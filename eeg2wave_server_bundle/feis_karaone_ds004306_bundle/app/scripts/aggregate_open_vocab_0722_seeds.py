#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EXCLUDED = {"moe_beats_matched_dense_three_seeds", "two_of_three_seeds_and_pooled"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate exactly three OpenVoice seed validation reports")
    parser.add_argument("--reports", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.resolve().read_text(encoding="utf-8")) for path in args.reports]
    if any(report.get("split") != "validation" or report.get("test_accessed") for report in reports):
        raise ValueError("Seed aggregation accepts validation-only reports")
    core_passes = [all(value for key, value in report["criteria"].items() if key not in EXCLUDED) for report in reports]
    names = sorted({key for report in reports for key in report["statistics"]})
    pooled = {name: float(np.mean([report["statistics"][name] for report in reports])) for name in names}
    pooled_passed = sum(core_passes) >= 2 and pooled["envelope_gain_subject_macro"] >= 0.05 and pooled["log_mel_gain_db_subject_macro"] >= 0.5 and pooled["retrieval_r5"] >= 2.0 * pooled["retrieval_chance_r5"] and pooled["composite_gain_bootstrap_95ci_lower"] > 0.0
    output = {
        "schema_version": "openvoice-0722-three-seed-summary-v1",
        "seeds": [15, 31, 47], "seed_passed": core_passes, "pooled_statistics": pooled,
        "pooled_passed": pooled_passed, "test_accessed": False,
        "reports": [str(path.resolve()) for path in args.reports],
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__": main()
