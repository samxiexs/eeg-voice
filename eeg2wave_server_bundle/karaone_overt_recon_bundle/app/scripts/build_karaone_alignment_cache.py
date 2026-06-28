from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.karaone_recon.alignment import build_alignment_cache
from src.karaone_recon.targets import KaraOneTargets
from src.utils import load_simple_yaml, resolve_bundle_path, resolve_target_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build KaraOne EEG/audio timing alignment cache.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--stages", default="overt_like", help="comma list; overt_like is the only high-confidence acoustic stage")
    parser.add_argument("--out", default="../artifacts/alignment/karaone_overt_like_alignment.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    out = resolve_bundle_path(args.out, BUNDLE_DIR)
    sample_rate = int(cfg.get("audio", {}).get("sample_rate", 16000))
    mel_hop = int(cfg.get("target", {}).get("mel_hop", 256))
    mel_hop_sec = float(mel_hop) / float(sample_rate)
    target_steps = 126
    try:
        _, mel_cache = resolve_target_cache(cfg, BUNDLE_DIR, "mel")
        if mel_cache.exists():
            targets = KaraOneTargets(mel_cache, data_root=root)
            target_steps = int(targets.T)
    except Exception:
        pass
    stages = tuple(item.strip() for item in str(args.stages).split(",") if item.strip())
    cache = build_alignment_cache(
        data_root=root,
        output_path=out,
        stages=stages,
        mel_hop_sec=mel_hop_sec,
        target_steps=target_steps,
    )
    payload = np.load(cache, allow_pickle=True)
    lag = payload["lag_sec"].astype(np.float32)
    summary = {
        "cache": str(cache),
        "csv": str(cache.with_suffix(".csv")),
        "stages": list(stages),
        "n": int(lag.shape[0]),
        "coverage_note": "cache rows are one per segment row for requested stages",
        "lag_sec_median": float(np.median(lag)) if lag.size else 0.0,
        "lag_sec_mean": float(np.mean(lag)) if lag.size else 0.0,
        "lag_sec_p05": float(np.percentile(lag, 5)) if lag.size else 0.0,
        "lag_sec_p95": float(np.percentile(lag, 95)) if lag.size else 0.0,
        "mel_hop_sec": float(payload["mel_hop_sec"]),
        "target_steps": int(payload["target_steps"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
