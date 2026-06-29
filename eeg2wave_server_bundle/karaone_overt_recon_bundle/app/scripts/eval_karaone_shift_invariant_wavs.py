from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize v5 shift-invariant KaraOne synthesis metrics.")
    parser.add_argument("--wav-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wav_dir = args.wav_dir.expanduser().resolve()
    metrics_path = wav_dir / "synth_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"synth_metrics.json not found: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    keys = [
        "active_segment_shape_corr_mean",
        "best_shift_full_env_corr_mean",
        "best_shift_sec_mean",
        "pred_active_local_scaled_active_duration_ratio_mean",
        "pred_active_local_scaled_voiced_rms_over_orig_mean",
        "pred_active_local_scaled_peak_over_orig_mean",
        "silence_leakage_wav_mean",
        "oracle_active_env_corr_mean",
        "oracle_voiced_rms_over_orig_mean",
    ]
    summary = {
        "wav_dir": str(wav_dir),
        "n": metrics.get("n"),
        "target_kind": metrics.get("target_kind"),
        "prediction_mode": metrics.get("prediction_mode"),
        "shift_invariant": {key: metrics.get(key) for key in keys},
    }
    out_path = wav_dir / "shift_invariant_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
