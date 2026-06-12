"""Synthesize wavs from an EEG-only direct checkpoint.

Writes comparable wavs:
  original_ref, target_oracle, mean_latent, pred_unscaled, pred_scaled

  python scripts/direct_synthesize.py --config configs/direct_eeg2speech.yaml \
      --checkpoint ../artifacts/outputs_direct/direct_stimuli_thinking_v1/checkpoints/best.pt
"""

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

from src.direct_eeg2speech.model import DirectEEG2Speech, DirectEEG2SpeechConfig
from src.feis_factored.data import FactoredFEISDataset
from src.feis_factored.synth import build_codec_backend, denormalize_latent
from src.feis_factored.targets import FactoredTargets
from src.utils import ensure_dir, load_simple_yaml, load_wav_fixed, resolve_bundle_path, resolve_feis_root, save_wav


def parse_args():
    p = argparse.ArgumentParser(description="Synthesize wavs from EEG-only direct model.")
    p.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "direct_eeg2speech.yaml"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test_holdout")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--device", default=None)
    return p.parse_args()


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)) + 1e-12)


def _scale_to_rms(wav: np.ndarray, target_rms: float, max_gain: float = 20.0) -> np.ndarray:
    gain = min(float(target_rms) / _rms(wav), max_gain)
    return (wav * gain).astype(np.float32)


def main():
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DirectEEG2Speech(DirectEEG2SpeechConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    mean = np.asarray(ckpt["target_mean"], np.float32)
    std = np.asarray(ckpt["target_std"], np.float32)

    targets = FactoredTargets(resolve_bundle_path(cfg["data"]["target_cache"], BUNDLE_DIR))
    default_scales = np.asarray(ckpt.get("default_decoder_scales", targets.default_decoder_scales), np.float32)
    feis_root = resolve_feis_root(resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR))
    ds = FactoredFEISDataset(
        data_root=resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR),
        targets=targets,
        split=args.split,
        stages=tuple(str(s).strip() for s in ckpt["stages"] if str(s).strip()),
        include_anomalous=bool(cfg["data"].get("include_anomalous", False)),
        holdout_offset=int(ckpt.get("holdout_offset", cfg["data"].get("holdout_offset", 0))),
        holdout_random=bool(ckpt.get("holdout_random", cfg["data"].get("holdout_random", True))),
    )

    backend = build_codec_backend(
        str(resolve_bundle_path("../models/encodec_24khz", BUNDLE_DIR)),
        duration_sec=1.0,
    )
    sr = backend.sample_rate
    out_dir = ensure_dir(args.out_dir or (
        Path(args.checkpoint).resolve().parents[1] / f"direct_wav_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}"
    ))
    mean_wav = backend.decode(targets.global_mean_raw_seq().astype(np.float32), decoder_scales=default_scales)

    manifest = []
    for i in range(min(args.limit, len(ds))):
        item = ds[i]
        e = ds.entries[i]
        sub, lab, stg = item["subject"], item["label"], item["stage"]
        with torch.no_grad():
            pred_latent, pred_log_rms = model.generate_full(
                item["eeg"].unsqueeze(0).to(device),
                item["stage_idx"].view(1).to(device),
            )
        pred = pred_latent.squeeze(0).cpu().numpy()
        pred_rms = float(np.exp(float(pred_log_rms.item())))

        orig = load_wav_fixed(
            feis_root / targets.cell_audio_path(sub, lab),
            sample_rate=sr,
            n_samples=int(sr * 1.0),
            normalize="rms",
            target_rms=0.08,
        )
        oracle = backend.decode(
            targets.cell_raw_target(sub, lab).astype(np.float32),
            decoder_scales=targets.cell_decoder_scale(sub, lab),
        )
        pred_unscaled = backend.decode(denormalize_latent(pred, mean, std), decoder_scales=default_scales)
        pred_scaled = _scale_to_rms(pred_unscaled, pred_rms)

        tag = f"{sub}_{lab}_{stg}_t{e.trial_index}"
        for kind, wav in {
            "original_ref": orig,
            "target_oracle": oracle,
            "mean_latent": mean_wav,
            "pred_unscaled": pred_unscaled,
            "pred_scaled": pred_scaled,
        }.items():
            file_name = f"{tag}_{kind}.wav"
            save_wav(out_dir / file_name, wav, sr)
            manifest.append([sub, lab, stg, e.trial_index, args.split, kind, file_name, _rms(wav)])

    with (out_dir / "listening_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject", "label", "stage", "trial", "split", "wav_type", "file", "rms"])
        writer.writerows(manifest)
    print(f"[done] wrote {min(args.limit, len(ds))} EEG-only samples x 5 wavs to {out_dir}")


if __name__ == "__main__":
    main()
