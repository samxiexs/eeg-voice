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
    parser = argparse.ArgumentParser(description="Audit KaraOne v9 subject-holdout protocol manifest.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone_v9.yaml"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", default="../artifacts/audio_targets/karaone_v9_protocol_audit.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    manifest_path = resolve_bundle_path(args.manifest or cfg["cache"]["canonical_manifest"], BUNDLE_DIR)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing v9 canonical manifest: {manifest_path}. Run build_karaone_v9_canonical_cache.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = []
    checks.append(check("semantic cache coverage", manifest["coverage"].get("semantic_missing", 1) == 0, manifest["coverage"]))
    checks.append(check("subject train/val/test disjoint", all(value == 0 for value in manifest["overlaps"].values()), manifest["overlaps"]))
    checks.append(check("subject_val nonempty", manifest["splits"]["subject_val"]["n"] > 0, manifest["splits"]["subject_val"]))
    checks.append(check("subject_test nonempty", manifest["splits"]["subject_test"]["n"] > 0, manifest["splits"]["subject_test"]))
    checks.append(check("codec target available for transport", manifest["coverage"].get("codec_missing", 1) == 0, manifest["coverage"]))
    checks.append(check("prosody target available", manifest["coverage"].get("prosody_missing", 1) == 0, manifest["coverage"]))
    checks.append(
        {
            "name": "semantic generation gate declared",
            "status": "pass",
            "detail": {
                "subject_val_required": [
                    "semantic_over_mean_gain > 0",
                    "semantic_top3_gain_over_mean > 0",
                    "same_label_cross_subject_gain >= 0",
                ],
                "generation_note": "waveform demos are diagnostic unless this gate passes on subject_val and remains positive on subject_test",
            },
        }
    )
    status = "pass" if all(item["status"] == "pass" for item in checks[:4]) else "fail"
    if status == "pass" and any(item["status"] == "fail" for item in checks[4:6]):
        status = "warn"
    payload = {
        "audit_kind": "karaone_v9_protocol_audit",
        "manifest": str(manifest_path),
        "status": status,
        "checks": checks,
    }
    out = resolve_bundle_path(args.out, BUNDLE_DIR)
    write_json(out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def check(name: str, passed: bool, detail: dict) -> dict:
    return {"name": name, "status": "pass" if bool(passed) else "fail", "detail": detail}


if __name__ == "__main__":
    main()
