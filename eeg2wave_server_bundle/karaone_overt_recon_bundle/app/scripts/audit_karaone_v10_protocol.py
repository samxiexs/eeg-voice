from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.utils import load_simple_yaml, resolve_bundle_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit KaraOne v10 clustered subject-holdout protocol.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone_v10.yaml"))
    parser.add_argument("--cluster-audit", default=None)
    parser.add_argument("--out", default="../artifacts/audio_targets/karaone_v10_protocol_audit.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    audit_path = resolve_bundle_path(args.cluster_audit or cfg["cache"]["cluster_audit"], BUNDLE_DIR)
    checks = []
    if audit_path.exists():
        cluster_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        leakage = cluster_audit.get("heldout_subject_used_for_centroid_fit", {})
        checks.append(check("cluster bank exists", True, {"path": str(audit_path)}))
        checks.append(check("cluster bank train-only", not any(bool(value) for value in leakage.values()), leakage))
        checks.append(check("cluster train fit rows nonempty", int(cluster_audit.get("n_fit_rows", 0)) > 0, cluster_audit))
        checks.append(
            check(
                "EEG/speech/cross cluster coverage nonempty",
                all(
                    bool(cluster_audit.get("cluster_coverage", {}).get(name))
                    for name in ("eeg_cluster_id", "speech_cluster_id", "cross_modal_cluster_id")
                ),
                cluster_audit.get("cluster_coverage", {}),
            )
        )
    else:
        checks.append(check("cluster bank exists", False, {"path": str(audit_path)}))
    checks.append(
        {
            "name": "v10 waveform gate declared",
            "status": "pass",
            "detail": {
                "rule": "wav outputs are diagnostic until semantic/prosody subject_val gate passes and signs hold on subject_test",
                "required_subject_val": [
                    "semantic_over_zero_gain > 0.01",
                    "semantic_over_mean_gain > 0",
                    "semantic_top3_gain_over_mean > 0.02",
                    "same_label_cross_subject_gain >= 0",
                    "prompt_acc >= 0.13",
                    "pred_std_ratio_median in [0.7, 1.5]",
                    "pred_pairwise_corr_median < 0.75",
                    "channel gate entropy not collapsed",
                ],
            },
        }
    )
    status = "pass" if all(item["status"] == "pass" for item in checks[:-1]) else "fail"
    payload = {
        "audit_kind": "karaone_v10_protocol_audit",
        "status": status,
        "cluster_audit": str(audit_path),
        "checks": checks,
    }
    out = resolve_bundle_path(args.out, BUNDLE_DIR)
    write_json(out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def check(name: str, passed: bool, detail: dict) -> dict:
    return {"name": name, "status": "pass" if bool(passed) else "fail", "detail": detail}


if __name__ == "__main__":
    main()

