from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.io import wavfile
from scipy.signal import stft
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.karaone_0715.codec import DiscreteEncodec, DiscreteEncodecConfig  # noqa: E402
from src.karaone_0715.data import AudioCodeBank, KaraOne0715Dataset, SplitManifest0715, load_audio, write_json  # noqa: E402
from src.karaone_0715.eval import balanced_accuracy  # noqa: E402
from src.utils import resample_audio  # noqa: E402

from train_karaone_0715 import (  # noqa: E402
    audio_dir,
    cache_path,
    default_device,
    eeg_dir,
    load_audio_model,
    load_eeg_model,
    move_eeg_batch,
    resolve,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate strict EEG-only 0715 EnCodec-code wavs and comparisons.")
    parser.add_argument("--config", default=str(APP_DIR / "configs" / "karaone_0715.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--audio-checkpoint", default=None)
    parser.add_argument("--eeg-checkpoint", default=None)
    parser.add_argument("--split", choices=("subject_train", "subject_val", "subject_test"), default="subject_val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-final-test", action="store_true")
    parser.add_argument("--allow-failed-gate", action="store_true")
    parser.add_argument("--maskgit-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    return parser.parse_args()


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, int(sample_rate), (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16))


def fixed_rms(audio: np.ndarray, target: float = 0.08) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(value), dtype=np.float64) + 1e-8))
    return np.clip(value * min(float(target) / max(rms, 1e-8), 10.0), -0.95, 0.95).astype(np.float32)


def match_length(audio: np.ndarray, length: int) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float32)
    if len(value) >= int(length):
        return value[: int(length)]
    return np.pad(value, (0, int(length) - len(value))).astype(np.float32)


def waveform_metrics(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> dict[str, float]:
    n = min(len(reference), len(candidate))
    reference = np.asarray(reference[:n], dtype=np.float64)
    candidate = np.asarray(candidate[:n], dtype=np.float64)
    ref = reference - reference.mean()
    pred = candidate - candidate.mean()
    correlation = float((ref @ pred) / (np.linalg.norm(ref) * np.linalg.norm(pred) + 1e-12))
    scale = float((candidate @ reference) / (reference @ reference + 1e-12))
    target = scale * reference
    noise = candidate - target
    si_sdr = float(10.0 * np.log10((target @ target + 1e-12) / (noise @ noise + 1e-12)))
    _, _, ref_spec = stft(reference, fs=sample_rate, nperseg=512, noverlap=384, boundary=None)
    _, _, pred_spec = stft(candidate, fs=sample_rate, nperseg=512, noverlap=384, boundary=None)
    frames = min(ref_spec.shape[1], pred_spec.shape[1])
    ref_db = 20.0 * np.log10(np.maximum(np.abs(ref_spec[:, :frames]), 1e-6))
    pred_db = 20.0 * np.log10(np.maximum(np.abs(pred_spec[:, :frames]), 1e-6))
    return {"waveform_correlation": correlation, "si_sdr_db": si_sdr, "log_spectrogram_mae_db": float(np.mean(np.abs(ref_db - pred_db)))}


def comparison_figure(
    path: Path,
    reference: np.ndarray,
    oracle: np.ndarray,
    label_only: np.ndarray,
    eeg_generated: np.ndarray,
    *,
    key: str,
    sample_rate: int,
) -> None:
    signals = [reference, oracle, label_only, eeg_generated]
    names = ["reference", "exact EnCodec-code oracle", "EEG-label-only prior", "full EEG-conditioned"]
    colors = ["#2563eb", "#16a34a", "#9333ea", "#dc2626"]
    duration = len(reference) / float(sample_rate)
    time = np.arange(len(reference)) / float(sample_rate)
    specs = []
    for signal in signals:
        _, _, value = stft(signal, fs=sample_rate, nperseg=512, noverlap=384, boundary=None)
        specs.append(20.0 * np.log10(np.maximum(np.abs(value), 1e-6)))
    low = float(min(value.min() for value in specs))
    high = float(max(value.max() for value in specs))
    fig, axes = plt.subplots(5, 1, figsize=(13, 12), constrained_layout=True)
    for signal, name, color in zip(signals, names, colors):
        axes[0].plot(time, signal, linewidth=0.55, alpha=0.78, label=name, color=color)
    axes[0].set(title=f"{key}: 0715 waveform comparison", xlim=(0, duration), ylabel="amplitude")
    axes[0].legend(loc="upper right", ncol=2)
    for axis, spec, name, color in zip(axes[1:], specs, names, ("Blues", "Greens", "Purples", "Reds")):
        axis.imshow(spec, origin="lower", aspect="auto", cmap=color, vmin=low, vmax=high, extent=(0, duration, 0, sample_rate / 2))
        axis.set(title=f"{name} log-spectrogram", ylabel="Hz")
    axes[-1].set(xlabel="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def summarise(metrics: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {
        name: {"mean": float(np.mean(values)), "median": float(np.median(values))}
        for name, values in metrics.items()
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(cfg["run"]["seed"])
    set_seed(seed)
    if args.split == "subject_test" and not args.allow_final_test:
        raise PermissionError("0715 MM21 synthesis requires --allow-final-test")
    gate_path = eeg_dir(cfg, seed) / "metrics" / "validation_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {"passed": False, "reasons": ["missing_validation_gate"]}
    if args.split == "subject_test" and not bool(gate.get("passed")) and not args.allow_failed_gate:
        raise PermissionError(f"0715 P02 gate failed; MM21 remains locked: {gate.get('reasons')}")
    device = torch.device(args.device) if args.device else default_device()
    manifest = SplitManifest0715.build(resolve(cfg["data"]["root"]))
    target_cache = Path(args.cache) if args.cache else cache_path(cfg, seed)
    bank = AudioCodeBank(target_cache, manifest)
    audio_path = Path(args.audio_checkpoint) if args.audio_checkpoint else audio_dir(cfg, seed) / "checkpoints" / "best.pt"
    eeg_path = Path(args.eeg_checkpoint) if args.eeg_checkpoint else eeg_dir(cfg, seed) / "checkpoints" / "best.pt"
    audio_model, audio_payload = load_audio_model(audio_path, bank, device)
    eeg_model, eeg_payload = load_eeg_model(eeg_path, bank, device)
    audio_model.eval()
    eeg_model.eval()
    root = resolve(cfg["data"]["root"])
    dataset = KaraOne0715Dataset(
        root,
        args.split,
        bank=bank,
        manifest=manifest,
        eeg_len=int(cfg["data"]["eeg_len"]),
        baseline_mode=str(cfg["data"]["baseline_mode"]),
        clip_value=float(cfg["data"]["baseline_clip"]),
    )
    if args.limit is not None:
        dataset.records = dataset.records[: int(args.limit)]
    loader = DataLoader(dataset, batch_size=int(cfg["evaluation"].get("synthesis_batch_size", 4)), shuffle=False, num_workers=0)
    codec_cfg = cfg["codec"]
    codec = DiscreteEncodec(
        DiscreteEncodecConfig(
            model_path=str(resolve(cfg["paths"]["encodec_model"])),
            sample_rate=int(codec_cfg["sample_rate"]),
            duration_sec=float(codec_cfg["duration_sec"]),
            bandwidth=float(codec_cfg["bandwidth"]),
        ),
        device,
    )
    steps = int(args.maskgit_steps or cfg["evaluation"]["maskgit_steps"])
    temperature = float(cfg["evaluation"]["synthesis_temperature"] if args.temperature is None else args.temperature)
    destination = eeg_dir(cfg, seed) / "wavs" / f"0715_{args.split}"
    folders = {name: destination / name for name in ("reference", "codec_oracle", "label_only", "eeg_conditioned", "comparison")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    eeg_metrics = {name: [] for name in ("waveform_correlation", "si_sdr_db", "log_spectrogram_mae_db")}
    label_metrics = {name: [] for name in ("waveform_correlation", "si_sdr_db", "log_spectrogram_mae_db")}
    oracle_metrics = {name: [] for name in ("waveform_correlation", "si_sdr_db", "log_spectrogram_mae_db")}
    target_labels, eeg_audio_labels, label_audio_labels = [], [], []
    coarse_correct = 0
    coarse_label_correct = 0
    coarse_total = 0
    files = []
    progress = tqdm(total=len(dataset), desc=f"[0715] synthesize {args.split}", unit="trial", dynamic_ncols=True)
    with torch.no_grad():
        for batch in loader:
            batch = move_eeg_batch(batch, device)
            output = eeg_model(batch["eeg"], batch["eeg_valid_len"])
            probabilities = torch.softmax(output["label_logits"], dim=-1)
            eeg_codes = audio_model.decoder.generate(output["condition"], probabilities, steps=steps, temperature=temperature)
            label_codes = audio_model.decoder.generate(torch.zeros_like(output["condition"]), probabilities, steps=steps, temperature=temperature)
            true_codes = batch["codes"]
            target_labels.extend(int(value) for value in batch["label_idx"].cpu().tolist())
            eeg_audio_labels.extend(int(value) for value in audio_model.encoder(eeg_codes)["label_logits"].argmax(dim=-1).cpu().tolist())
            label_audio_labels.extend(int(value) for value in audio_model.encoder(label_codes)["label_logits"].argmax(dim=-1).cpu().tolist())
            coarse_correct += int((eeg_codes[:, :2] == true_codes[:, :2]).sum().item())
            coarse_label_correct += int((label_codes[:, :2] == true_codes[:, :2]).sum().item())
            coarse_total += int(true_codes[:, :2].numel())
            for row in range(len(batch["key"])):
                key = str(batch["key"][row])
                safe_key = key.replace(":", "_")
                reference_16k = load_audio(root / str(batch["audio_path"][row]), sample_rate=16000, duration_sec=float(codec_cfg["duration_sec"]))
                reference = resample_audio(reference_16k, src_sr=16000, dst_sr=codec.codec_sample_rate)
                scale = batch["encodec_scale"][row].cpu().numpy() if bool(batch["encodec_scale_valid"][row].item()) else None
                oracle = codec.decode(true_codes[row].cpu().numpy(), scale=scale)
                label_audio = codec.decode(label_codes[row].cpu().numpy(), scale=None)
                eeg_audio = codec.decode(eeg_codes[row].cpu().numpy(), scale=None)
                reference = fixed_rms(reference)
                oracle = fixed_rms(match_length(oracle, len(reference)))
                label_audio = fixed_rms(match_length(label_audio, len(reference)))
                eeg_audio = fixed_rms(match_length(eeg_audio, len(reference)))
                for name, value in waveform_metrics(reference, oracle, codec.codec_sample_rate).items():
                    oracle_metrics[name].append(value)
                for name, value in waveform_metrics(reference, label_audio, codec.codec_sample_rate).items():
                    label_metrics[name].append(value)
                for name, value in waveform_metrics(reference, eeg_audio, codec.codec_sample_rate).items():
                    eeg_metrics[name].append(value)
                paths = {
                    "reference": folders["reference"] / f"{safe_key}.wav",
                    "codec_oracle": folders["codec_oracle"] / f"{safe_key}.wav",
                    "label_only": folders["label_only"] / f"{safe_key}.wav",
                    "eeg_conditioned": folders["eeg_conditioned"] / f"{safe_key}.wav",
                    "comparison": folders["comparison"] / f"{safe_key}.png",
                }
                write_wav(paths["reference"], reference, codec.codec_sample_rate)
                write_wav(paths["codec_oracle"], oracle, codec.codec_sample_rate)
                write_wav(paths["label_only"], label_audio, codec.codec_sample_rate)
                write_wav(paths["eeg_conditioned"], eeg_audio, codec.codec_sample_rate)
                comparison_figure(paths["comparison"], reference, oracle, label_audio, eeg_audio, key=key, sample_rate=codec.codec_sample_rate)
                files.append({"key": key, **{name: str(path.relative_to(destination)) for name, path in paths.items()}})
            progress.update(len(batch["key"]))
    progress.close()
    report = {
        "version": "0715",
        "phase": "synthesize",
        "split": args.split,
        "n_generated": len(files),
        "sample_rate": codec.codec_sample_rate,
        "maskgit_steps": steps,
        "temperature": temperature,
        "audio_checkpoint": str(audio_path),
        "audio_checkpoint_epoch": int(audio_payload["epoch"]),
        "eeg_checkpoint": str(eeg_path),
        "eeg_checkpoint_epoch": int(eeg_payload["epoch"]),
        "p02_gate_passed": bool(gate.get("passed")),
        "eeg_conditioned_code_classifier_balanced_accuracy": balanced_accuracy(np.asarray(target_labels), np.asarray(eeg_audio_labels)),
        "label_only_code_classifier_balanced_accuracy": balanced_accuracy(np.asarray(target_labels), np.asarray(label_audio_labels)),
        "eeg_conditioned_coarse_code_accuracy": float(coarse_correct / max(coarse_total, 1)),
        "label_only_coarse_code_accuracy": float(coarse_label_correct / max(coarse_total, 1)),
        "eeg_conditioned_coarse_gain": float((coarse_correct - coarse_label_correct) / max(coarse_total, 1)),
        "codec_oracle_vs_reference": summarise(oracle_metrics),
        "label_only_vs_reference": summarise(label_metrics),
        "eeg_conditioned_vs_reference": summarise(eeg_metrics),
        "inference_input": "clearing-calibrated overt EEG only",
        "predicted_label_source": "EEG label head; no true label at inference",
        "reference_audio_used_for_generation": False,
        "reference_encodec_scale_used_for_generation": False,
        "reference_audio_exported_after_generation_for_comparison": True,
        "fixed_rms_postprocessing": True,
        "claim_boundary": "label-only is a canonical-speech prior; full EEG must improve over it before claiming EEG-specific acoustic reconstruction",
        "test_accessed": args.split == "subject_test",
    }
    write_json(destination / "synthesis_manifest.json", {**report, "files": files})
    write_json(eeg_dir(cfg, seed) / "metrics" / f"{args.split}_audio_metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
