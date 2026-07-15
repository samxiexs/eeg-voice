from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.io import wavfile
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.karaone_0715.codec import DiscreteEncodec, DiscreteEncodecConfig  # noqa: E402
from src.karaone_0715.data import LABELS, AudioCodeBank, SplitManifest0715, load_audio, write_json  # noqa: E402
from src.karaone_0715.eval import balanced_accuracy  # noqa: E402
from src.utils import resample_audio  # noqa: E402

from synthesize_karaone_0715 import fixed_rms, match_length, waveform_metrics  # noqa: E402
from train_karaone_0715 import audio_dir, cache_path, default_device, load_audio_model, resolve, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit 0715 audio code encoder/MaskGIT/EnCodec round-trip before EEG training.")
    parser.add_argument("--config", default=str(APP_DIR / "configs" / "karaone_0715.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--export-limit", type=int, default=None)
    return parser.parse_args()


class IndexedCodes(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, bank: AudioCodeBank, indices: np.ndarray):
        self.bank = bank
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = {label: index for index, label in enumerate(LABELS)}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        return (
            torch.from_numpy(np.ascontiguousarray(self.bank.codes[index])).long(),
            torch.tensor(self.labels[str(self.bank.labels[index])], dtype=torch.long),
            torch.tensor(index, dtype=torch.long),
        )


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, int(sample_rate), (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16))


def metric_store() -> dict[str, list[float]]:
    return {name: [] for name in ("waveform_correlation", "si_sdr_db", "log_spectrogram_mae_db")}


def summarise(metrics: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {name: {"mean": float(np.mean(values)), "median": float(np.median(values))} for name, values in metrics.items()}


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg["run"]["seed"])
    set_seed(seed)
    device = torch.device(args.device) if args.device else default_device()
    root = resolve(cfg["data"]["root"])
    manifest = SplitManifest0715.build(root)
    target_cache = Path(args.cache) if args.cache else cache_path(cfg, seed)
    bank = AudioCodeBank(target_cache, manifest)
    checkpoint = Path(args.checkpoint) if args.checkpoint else audio_dir(cfg, seed) / "checkpoints" / "best.pt"
    model, payload = load_audio_model(checkpoint, bank, device)
    model.eval()
    settings = cfg["evaluation"]
    loader = DataLoader(
        IndexedCodes(bank, bank.indices("subject_val")),
        batch_size=int(settings.get("synthesis_batch_size", 4)),
        shuffle=False,
        num_workers=0,
    )
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
    maskgit_steps = int(settings["maskgit_steps"])
    temperature = float(settings["synthesis_temperature"])
    export_limit = int(args.export_limit if args.export_limit is not None else settings["audio_roundtrip_export_limit"])
    destination = audio_dir(cfg, seed) / "roundtrip_subject_val"
    for name in ("reference", "codec_oracle", "audio_condition", "audio_condition_plus_label", "label_only"):
        (destination / name).mkdir(parents=True, exist_ok=True)
    targets: list[int] = []
    predictions: dict[str, list[int]] = {name: [] for name in ("audio_condition", "audio_condition_plus_label", "label_only")}
    coarse_correct = {name: 0 for name in predictions}
    coarse_total = 0
    waveform = {name: metric_store() for name in ("codec_oracle", "audio_condition", "audio_condition_plus_label", "label_only")}
    exported = 0
    with torch.no_grad():
        for codes, labels, cache_indices in tqdm(loader, desc="[0715 audio] round-trip audit", unit="batch", dynamic_ncols=True):
            codes, labels = codes.to(device), labels.to(device)
            encoded = model.encoder(codes)
            true_probabilities = F.one_hot(labels, num_classes=model.cfg.num_labels).float()
            zero_probabilities = torch.zeros_like(true_probabilities)
            zero_condition = torch.zeros_like(encoded["condition"])
            generated = {
                "audio_condition": model.decoder.generate(encoded["condition"], zero_probabilities, steps=maskgit_steps, temperature=temperature),
                "audio_condition_plus_label": model.decoder.generate(encoded["condition"], true_probabilities, steps=maskgit_steps, temperature=temperature),
                "label_only": model.decoder.generate(zero_condition, true_probabilities, steps=maskgit_steps, temperature=temperature),
            }
            targets.extend(int(value) for value in labels.cpu().tolist())
            for name, generated_codes in generated.items():
                predictions[name].extend(int(value) for value in model.encoder(generated_codes)["label_logits"].argmax(dim=-1).cpu().tolist())
                coarse_correct[name] += int((generated_codes[:, :2] == codes[:, :2]).sum().item())
            coarse_total += int(codes[:, :2].numel())
            for row, cache_index in enumerate(cache_indices.cpu().tolist()):
                if exported >= export_limit:
                    continue
                key = str(bank.keys[int(cache_index)])
                safe_key = key.replace(":", "_")
                reference_16k = load_audio(root / str(bank.audio_paths[int(cache_index)]), sample_rate=16000, duration_sec=float(codec_cfg["duration_sec"]))
                reference = fixed_rms(resample_audio(reference_16k, src_sr=16000, dst_sr=codec.codec_sample_rate))
                scale = bank.scale[int(cache_index)] if bool(bank.scale_valid[int(cache_index)]) else None
                decoded = {"codec_oracle": codec.decode(codes[row].cpu().numpy(), scale=scale)}
                decoded.update({name: codec.decode(value[row].cpu().numpy(), scale=None) for name, value in generated.items()})
                write_wav(destination / "reference" / f"{safe_key}.wav", reference, codec.codec_sample_rate)
                for name, audio in decoded.items():
                    audio = fixed_rms(match_length(audio, len(reference)))
                    write_wav(destination / name / f"{safe_key}.wav", audio, codec.codec_sample_rate)
                    for metric, value in waveform_metrics(reference, audio, codec.codec_sample_rate).items():
                        waveform[name][metric].append(value)
                exported += 1
    content_accuracy = {name: balanced_accuracy(np.asarray(targets), np.asarray(value)) for name, value in predictions.items()}
    coarse_accuracy = {name: float(value / max(coarse_total, 1)) for name, value in coarse_correct.items()}
    condition_gain = coarse_accuracy["audio_condition"] - coarse_accuracy["label_only"]
    passed = bool(
        content_accuracy["audio_condition"] >= float(settings["min_audio_condition_content_accuracy"])
        and content_accuracy["audio_condition_plus_label"] >= float(settings["min_audio_condition_plus_label_accuracy"])
        and condition_gain > 0.0
    )
    reasons = []
    if content_accuracy["audio_condition"] < float(settings["min_audio_condition_content_accuracy"]):
        reasons.append("audio_condition_content_accuracy_below_threshold")
    if content_accuracy["audio_condition_plus_label"] < float(settings["min_audio_condition_plus_label_accuracy"]):
        reasons.append("audio_condition_plus_label_accuracy_below_threshold")
    if condition_gain <= 0.0:
        reasons.append("audio_condition_does_not_improve_coarse_codes_over_label_only")
    report: dict[str, Any] = {
        "version": "0715",
        "phase": "audio_roundtrip_gate",
        "passed": passed,
        "reasons": reasons,
        "split": "subject_val",
        "n_trials": len(targets),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "maskgit_steps": maskgit_steps,
        "content_balanced_accuracy": content_accuracy,
        "coarse_code_accuracy": coarse_accuracy,
        "audio_condition_coarse_gain_over_label_only": condition_gain,
        "waveform_metrics_exported_subset_n": exported,
        "waveform_metrics_exported_subset": {name: summarise(values) for name, values in waveform.items()},
        "requirements": {
            "audio_condition_content_accuracy_min": float(settings["min_audio_condition_content_accuracy"]),
            "audio_condition_plus_label_accuracy_min": float(settings["min_audio_condition_plus_label_accuracy"]),
            "audio_condition_coarse_gain_over_label_only_min_exclusive": 0.0,
        },
        "reference_audio_used_for_generation": False,
        "true_label_used_only_in_explicit_plus_label_and_label_only_audit_branches": True,
        "test_accessed": False,
    }
    write_json(audio_dir(cfg, seed) / "metrics" / "roundtrip_gate.json", report)
    write_json(destination / "roundtrip_manifest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
