from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-karaone-0711v1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.io import wavfile
from scipy.signal import stft
from torch.utils.data import DataLoader

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from scripts.train_karaone_0711v1 import AudioTargetBank, cache_paths, default_device, load_model, make_eeg_model, move_batch, resolve
from src.audio_features import AudioFeatureConfig, load_codec_backend
from src.karaone_0711v1.data import KaraOne0711Dataset, SplitManifest, make_run_manifest, run_name, write_json
from src.karaone_0711v1.eval import require_flow_gate, validate_gate_context
from src.karaone_0711v1.model import ConditionalFlowDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEG-only 0711v1 EnCodec-flow synthesis.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone_0711v1.yaml"))
    parser.add_argument("--encoder", required=True, help="Locked alignment checkpoint")
    parser.add_argument("--flow", required=True, help="Passed-gate flow checkpoint")
    parser.add_argument("--gate", required=True, help="Passed validation gate JSON")
    parser.add_argument("--stage", choices=("overt_like", "thinking"), default=None)
    parser.add_argument("--split", choices=("subject_train", "subject_val", "subject_test", "all"), default="subject_val")
    parser.add_argument("--allow-final-test", action="store_true")
    parser.add_argument("--allow-gate-bypass", action="store_true", help="Exploratory-only synthesis after a failed validation gate; MM21 additionally requires --allow-all-splits-diagnostic.")
    parser.add_argument("--allow-all-splits-diagnostic", action="store_true", help="Explicitly authorise all 1,913 reference/reconstruction wavs and comparison figures, including MM21 diagnostic audio.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    wavfile.write(path, sample_rate, (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16))


def write_comparison(path: Path, reference: np.ndarray, reconstruction: np.ndarray, key: str, sample_rate: int = 16000) -> None:
    """Waveform plus paired log-spectrogram comparison for one trial."""
    duration = min(len(reference), len(reconstruction)) / float(sample_rate)
    time = np.arange(min(len(reference), len(reconstruction))) / float(sample_rate)
    _, _, ref_spec = stft(reference, fs=sample_rate, nperseg=512, noverlap=384, boundary=None)
    _, _, rec_spec = stft(reconstruction, fs=sample_rate, nperseg=512, noverlap=384, boundary=None)
    ref_db = 20.0 * np.log10(np.maximum(np.abs(ref_spec), 1e-6))
    rec_db = 20.0 * np.log10(np.maximum(np.abs(rec_spec), 1e-6))
    low, high = float(min(ref_db.min(), rec_db.min())), float(max(ref_db.max(), rec_db.max()))
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), constrained_layout=True)
    axes[0].plot(time, reference[: len(time)], color="#2563eb", linewidth=0.7, label="reference")
    axes[0].plot(time, reconstruction[: len(time)], color="#dc2626", linewidth=0.7, alpha=0.75, label="EEG reconstruction")
    axes[0].set(title=f"{key}: waveform comparison", xlim=(0, duration), ylabel="amplitude")
    axes[0].legend(loc="upper right")
    axes[1].imshow(ref_db, origin="lower", aspect="auto", cmap="Blues", vmin=low, vmax=high, extent=(0, duration, 0, sample_rate / 2))
    axes[1].set(title="reference log-spectrogram", ylabel="Hz")
    axes[2].imshow(rec_db, origin="lower", aspect="auto", cmap="Reds", vmin=low, vmax=high, extent=(0, duration, 0, sample_rate / 2))
    axes[2].set(title="EEG reconstruction log-spectrogram", xlabel="seconds", ylabel="Hz")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    stage = args.stage or str(cfg["data"]["stage"])
    seed = int(args.seed if args.seed is not None else cfg["run"]["seed"])
    device = torch.device(args.device) if args.device else default_device()
    root = resolve(cfg["data"]["root"])
    manifest = SplitManifest.build(root)
    gate = validate_gate_context(args.gate, manifest) if args.allow_gate_bypass else require_flow_gate(args.gate, manifest)
    if args.split == "all" and not args.allow_all_splits_diagnostic:
        raise PermissionError("All-split wav/figure export requires --allow-all-splits-diagnostic")
    if args.split == "subject_test" and not args.allow_final_test:
        raise PermissionError("MM21 synthesis requires --allow-final-test")
    if args.split == "subject_test" and args.allow_gate_bypass and not gate.get("passed") and not args.allow_all_splits_diagnostic:
        raise PermissionError("Exploratory MM21 synthesis requires --allow-all-splits-diagnostic")
    bank = AudioTargetBank(cache_paths(cfg, stage, seed)["audio_targets"])
    if bank.encodec_latent is None:
        raise FileNotFoundError("Adapted target cache has no EnCodec latent array")
    encoder = make_eeg_model(cfg).to(device)
    load_model(args.encoder, encoder, device)
    encoder.eval()
    flow_cfg = cfg["flow"]
    flow = ConditionalFlowDecoder(
        latent_dim=int(bank.encodec_latent.shape[-1]),
        eeg_dim=int(cfg["model"]["d_model"]),
        d_model=int(flow_cfg["d_model"]),
        heads=int(flow_cfg["heads"]),
        layers=int(flow_cfg["layers"]),
    ).to(device)
    load_model(args.flow, flow, device)
    flow.eval()
    splits = ("subject_train", "subject_val", "subject_test") if args.split == "all" else (args.split,)
    output_label = "all_splits_diagnostic" if args.split == "all" else f"generated_{args.split}"
    output = resolve(cfg["paths"]["output_root"]) / run_name(stage, "flow", seed) / "wavs" / output_label
    output.mkdir(parents=True, exist_ok=True)
    codec_path = resolve(cfg["paths"]["encodec_model"])
    codec = load_codec_backend(AudioFeatureConfig(sample_rate=16000, duration_sec=2.0, target_kind="encodec_latent", backend="encodec_latent", codec_model_name_or_path=str(codec_path), local_files_only=True, codec_bandwidth=6.0))
    saved: list[dict[str, str]] = []
    total = 0
    with torch.no_grad():
        for split in splits:
            dataset = KaraOne0711Dataset(root, split, stage, manifest=manifest, eeg_len=int(cfg["data"]["eeg_len"]), sample_rate=int(cfg["data"]["eeg_sample_rate"]), include_audio=True)
            data = DataLoader(dataset, batch_size=int(flow_cfg["batch_size"]), shuffle=False, num_workers=0)
            ref_dir, rec_dir, fig_dir = (output / "reference" / split, output / "reconstructed" / split, output / "comparison" / split)
            for directory in (ref_dir, rec_dir, fig_dir):
                directory.mkdir(parents=True, exist_ok=True)
            for batch in data:
                batch = move_batch(batch, device)
                encoded = encoder(batch["eeg"], batch["eeg_valid_len"], batch["topography"])
                latent = flow.sample(encoded["tokens"], encoded["pred_onset_sec"], encoded["pred_duration_sec"], encoded["pred_active_logit"], steps=int(args.steps or flow_cfg["sample_steps"])).cpu().numpy()
                reference = batch["audio"].cpu().numpy()
                for idx, key in enumerate(batch["key"]):
                    safe_key = str(key).replace(":", "_")
                    reconstruction = codec.decode(latent[idx])
                    reference_path = ref_dir / f"{safe_key}.wav"
                    reconstruction_path = rec_dir / f"{safe_key}.wav"
                    figure_path = fig_dir / f"{safe_key}.png"
                    write_wav(reference_path, reference[idx])
                    write_wav(reconstruction_path, reconstruction)
                    write_comparison(figure_path, reference[idx], reconstruction, str(key))
                    saved.append({"key": str(key), "split": split, "reference_wav": str(reference_path.relative_to(output)), "reconstructed_wav": str(reconstruction_path.relative_to(output)), "comparison_png": str(figure_path.relative_to(output))})
                    total += 1
                    if args.limit and total >= int(args.limit):
                        break
                if args.limit and total >= int(args.limit):
                    break
            if args.limit and total >= int(args.limit):
                break
    write_json(output / "synthesis_manifest.json", {
        **make_run_manifest(repo_root=BUNDLE_DIR.parent.parent, config_path=args.config, split_manifest=manifest, phase="flow", stage=stage, seed=seed, input_paths=[args.encoder, args.flow, args.gate]),
        "inference_input": "eeg_only",
        "reference_audio_used_for_synthesis": False,
        "reference_audio_exported_for_diagnostic_comparison": True,
        "gate_bypassed": bool(args.allow_gate_bypass and not gate.get("passed")),
        "claim_status": "exploratory_all_splits_diagnostic_not_reportable" if args.allow_gate_bypass and not gate.get("passed") else "all_splits_diagnostic" if args.split == "all" else "gate_passed",
        "split": args.split,
        "test_accessed": args.split in {"subject_test", "all"},
        "n_generated": total,
        "files": saved,
    })
    print(str(output))


if __name__ == "__main__":
    main()
