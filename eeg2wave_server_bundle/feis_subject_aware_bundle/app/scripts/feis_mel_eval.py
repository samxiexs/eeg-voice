"""Evaluate a FEIS EEG-only mel-alignment checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.feis_mel.data import FEISMelDataset
from src.feis_mel.diffusion import FEISAcousticDiffusionConfig, FEISDiffusionInference, build_feis_acoustic_diffusion
from src.feis_mel.eval import evaluate_feis_mel
from src.feis_mel.model import FEISEEGToMel, FEISMelConfig
from src.feis_mel.targets import MelLabelTargets
from src.utils import load_simple_yaml, resolve_bundle_path, write_json


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate FEIS EEG-only mel checkpoint.")
    p.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "feis_mel_align.yaml"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test_holdout")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--sample-steps", type=int, default=None)
    return p.parse_args()


def _target_cache_path(cfg: dict, ckpt: dict) -> Path:
    target_cache = Path(str(ckpt.get("target_cache", "")))
    if target_cache.exists():
        return target_cache
    target_kind = str(ckpt.get("target_kind", cfg.get("target", {}).get("kind", "mel")))
    target_cfg = cfg.get("target", {})
    if target_kind == "encodec_latent":
        return resolve_bundle_path(target_cfg.get("cache_encodec", cfg["data"]["target_cache"]), BUNDLE_DIR)
    return resolve_bundle_path(target_cfg.get("cache_mel", cfg["data"]["target_cache"]), BUNDLE_DIR)


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    target_cache = _target_cache_path(cfg, ckpt)
    targets = MelLabelTargets(target_cache)
    ds = FEISMelDataset(
        data_root=resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR),
        targets=targets,
        split=args.split,
        stage=str(ckpt["stage"]),
        include_anomalous=bool(cfg["data"].get("include_anomalous", False)),
    )
    model = FEISEEGToMel(FEISMelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    eval_model = model
    if str(ckpt.get("decoder_kind", "regression")) == "diffusion":
        diff_cfg = FEISAcousticDiffusionConfig(**ckpt["diffusion_config"])
        diffusion = build_feis_acoustic_diffusion(diff_cfg).to(device)
        diffusion.load_state_dict(ckpt["diffusion_state"], strict=True)
        eval_model = FEISDiffusionInference(
            model,
            diffusion,
            target_steps=targets.T,
            target_dim=targets.D,
            sample_steps=int(args.sample_steps or diff_cfg.sample_steps),
        )
    metrics = evaluate_feis_mel(
        eval_model,
        ds,
        targets,
        device=device,
        batch_size=int(cfg["train"].get("batch_size", 64)),
        dtw_band=int(cfg["loss"].get("dtw_band", 10)),
    )
    out = Path(args.out) if args.out else Path(args.checkpoint).resolve().parents[1] / "metrics" / f"{args.split}_mel_eval.json"
    write_json(out, metrics)
    print(f"[done] {out}")
    print(metrics)


if __name__ == "__main__":
    main()
