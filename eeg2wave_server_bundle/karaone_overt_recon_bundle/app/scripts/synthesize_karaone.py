from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.karaone_recon.data import KaraOneTrialDataset
from src.karaone_recon.model import KaraOneConfig, KaraOneEEG2Codec
from src.karaone_recon.synth import build_codec_backend, denormalize_latent
from src.karaone_recon.targets import KaraOneTargets
from src.utils import ensure_dir, load_simple_yaml, load_wav_fixed, resolve_bundle_path, save_wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize KaraOne wavs from a reconstruction checkpoint.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "subject_test"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)) + 1e-12)


def _scale_to_rms(audio: np.ndarray, target_rms: float, max_gain: float = 20.0) -> np.ndarray:
    gain = min(float(target_rms) / _rms(audio), max_gain)
    return (audio * gain).astype(np.float32)


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = KaraOneEEG2Codec(KaraOneConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    targets = KaraOneTargets(resolve_bundle_path(cfg["data"]["target_cache"], BUNDLE_DIR), data_root=root)
    split_protocol = "subject_holdout" if args.split == "subject_test" else str(cfg["data"].get("split_protocol", "trial"))
    ds = KaraOneTrialDataset(
        data_root=root,
        targets=targets,
        split=args.split,
        stages=tuple(ckpt["stages"]),
        split_protocol=split_protocol,
        heldout_subjects=cfg["data"].get("heldout_subjects", ["P02", "MM21"]),
        eeg_len=int(cfg["data"].get("eeg_len", 1280)),
    )
    duration_sec = float(cfg["audio"].get("duration_sec", 2.0))
    backend = build_codec_backend(
        str(resolve_bundle_path(cfg["targets"]["codec_model_name_or_path"], BUNDLE_DIR)),
        duration_sec=duration_sec,
        bandwidth=float(cfg["targets"].get("codec_bandwidth", 6.0)),
        local_files_only=bool(cfg["targets"].get("local_files_only", True)),
    )
    sample_rate = int(backend.sample_rate)
    out_dir = ensure_dir(
        args.out_dir
        or (Path(args.checkpoint).resolve().parents[1] / f"wav_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}")
    )
    mean_wav = backend.decode(targets.global_mean_raw.astype(np.float32), decoder_scales=targets.default_decoder_scales)
    manifest = []
    for idx in range(min(int(args.limit), len(ds))):
        item = ds[idx]
        entry = ds.entries[idx]
        with torch.no_grad():
            pred_latent, pred_log_rms = model.generate_full(
                item["eeg"].unsqueeze(0).to(device),
                item["subject_idx"].view(1).to(device),
                item["stage_idx"].view(1).to(device),
            )
            zero_latent, zero_log_rms = model.generate_full(
                torch.zeros_like(item["eeg"]).unsqueeze(0).to(device),
                item["subject_idx"].view(1).to(device),
                item["stage_idx"].view(1).to(device),
            )
        pred = pred_latent.squeeze(0).cpu().numpy()
        zero = zero_latent.squeeze(0).cpu().numpy()
        pred_wav = backend.decode(
            denormalize_latent(pred, targets.target_mean, targets.target_std),
            decoder_scales=targets.default_decoder_scales,
        )
        zero_wav = backend.decode(
            denormalize_latent(zero, targets.target_mean, targets.target_std),
            decoder_scales=targets.default_decoder_scales,
        )
        oracle = backend.decode(
            targets.raw_target(entry.subject, entry.trial_index).astype(np.float32),
            decoder_scales=targets.decoder_scale(entry.subject, entry.trial_index),
        )
        original = load_wav_fixed(
            root / targets.audio_path(entry.subject, entry.trial_index),
            sample_rate=sample_rate,
            n_samples=int(round(sample_rate * duration_sec)),
            normalize=str(cfg["audio"].get("normalize", "rms")),
            target_rms=float(cfg["audio"].get("target_rms", 0.08)),
            max_gain=float(cfg["audio"].get("max_gain", 10.0)),
        )
        tag = f"{entry.subject}_{entry.label.replace('/', '')}_{entry.stage}_t{entry.trial_index:03d}"
        wavs = {
            "original": original,
            "oracle_codec": oracle,
            "mean_latent": mean_wav,
            "zeroeeg": zero_wav,
            "pred": pred_wav,
            "pred_scaled": _scale_to_rms(pred_wav, float(np.exp(float(pred_log_rms.item())))),
            "zeroeeg_scaled": _scale_to_rms(zero_wav, float(np.exp(float(zero_log_rms.item())))),
        }
        for kind, wav in wavs.items():
            filename = f"{tag}_{kind}.wav"
            save_wav(out_dir / filename, wav, sample_rate)
            manifest.append([entry.subject, entry.label, entry.stage, entry.trial_index, args.split, kind, filename, _rms(wav)])
    with (out_dir / "listening_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject", "label", "stage", "trial_index", "split", "wav_type", "file", "rms"])
        writer.writerows(manifest)
    print(f"[done] wrote {len(manifest)} wav rows to {out_dir}")


if __name__ == "__main__":
    main()
