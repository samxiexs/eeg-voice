"""Reconstruction-fidelity evaluation from a v3 checkpoint.

Measures how close the decoded prediction is to THIS subject's own recording,
with a cross-subject control proving subject-specificity.

  python scripts/v3_recon_eval.py --config configs/v3_encodec.yaml \
      --checkpoint <run>/checkpoints/best.pt --protocol G --stage thinking \
      --split test --limit 64 --out <run>/metrics/recon_eval.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.utils import load_simple_yaml, resolve_bundle_path, write_json
from src.v3.data import V3Dataset
from src.v3.model import EEG2SpeechV3, EEG2SpeechV3Config
from src.v3.recon_eval import evaluate_reconstruction


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reconstruction-fidelity eval for a v3 checkpoint.")
    p.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "v3_encodec.yaml"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--protocol", default=None)
    p.add_argument("--stage", default=None)
    p.add_argument("--subject", default=None)
    p.add_argument("--holdout-subject", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_simple_yaml(args.config)
    cfg_d, cfg_a, cfg_t = config["data"], config["audio"], config["targets"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = EEG2SpeechV3(EEG2SpeechV3Config(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    target_mean = np.asarray(ckpt["target_mean"], dtype=np.float32)
    target_std = np.asarray(ckpt["target_std"], dtype=np.float32)

    dataset = V3Dataset(
        data_root=str(resolve_bundle_path(args.data_root or cfg_d["root"], BUNDLE_DIR)),
        protocol=(args.protocol or cfg_d["protocol"]).upper(),
        stage=args.stage or cfg_d.get("stage", "thinking"),
        split=args.split,
        subject_id=args.subject or cfg_d.get("subject_id"),
        holdout_subject_id=args.holdout_subject or cfg_d.get("holdout_subject_id"),
        include_anomalous=bool(cfg_d.get("include_anomalous", False)),
        target_cache_path=str(resolve_bundle_path(cfg_t["cache_path"], BUNDLE_DIR)),
        require_targets=True,
        audio_sr=int(cfg_a["sample_rate"]),
        audio_dur=float(cfg_a["duration_sec"]),
    )

    metrics = evaluate_reconstruction(
        model,
        dataset,
        codec_model_path=str(resolve_bundle_path(cfg_t["codec_model_name_or_path"], BUNDLE_DIR)),
        target_mean=target_mean,
        target_std=target_std,
        device=device,
        duration_sec=float(cfg_a["duration_sec"]),
        bandwidth=float(cfg_t.get("codec_bandwidth", 6.0)),
        limit=args.limit,
    )
    print(metrics)
    if args.out:
        write_json(args.out, metrics)


if __name__ == "__main__":
    main()
