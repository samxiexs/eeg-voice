#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare three matched dense and Adapter-MoE validation runs")
    parser.add_argument("--dense-reports", type=Path, nargs=3, required=True)
    parser.add_argument("--moe-reports", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dense = [json.loads(path.resolve().read_text(encoding="utf-8")) for path in args.dense_reports]
    moe = [json.loads(path.resolve().read_text(encoding="utf-8")) for path in args.moe_reports]
    if any(report.get("test_accessed") or report.get("split") != "validation" for report in dense + moe):
        raise ValueError("Dense/MoE comparison is validation-only")
    dense_scores = [float(report["statistics"]["eeg_composite_subject_macro"]) for report in dense]
    moe_scores = [float(report["statistics"]["eeg_composite_subject_macro"]) for report in moe]
    gains = [moe_value - dense_value for moe_value, dense_value in zip(moe_scores, dense_scores)]
    report = {
        "schema_version": "openvoice-0722-dense-moe-comparison-v1",
        "seeds": [15, 31, 47], "dense_composite_by_seed": dense_scores,
        "moe_composite_by_seed": moe_scores, "moe_minus_dense_composite_by_seed": gains,
        "mean_gain": float(np.mean(gains)),
        "new_subject_not_worse": float(np.mean(moe_scores)) >= float(np.mean(dense_scores)),
        "test_accessed": False,
        "dense_reports": [str(path.resolve()) for path in args.dense_reports],
        "moe_reports": [str(path.resolve()) for path in args.moe_reports],
    }
    output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__": main()
