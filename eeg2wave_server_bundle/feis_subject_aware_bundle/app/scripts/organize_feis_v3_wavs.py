"""Verify FEIS v3 grouped wav organization."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Verify FEIS v3 grouped wav artifacts.")
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    manifest = run_dir / "wavs" / "listening_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8", newline="")))
    missing = []
    for row in rows:
        for key in ["grouped_original_reference", "grouped_retrieval_diagnostic", "grouped_generated_codec"]:
            if not Path(row[key]).exists():
                missing.append(row[key])
    if missing:
        raise FileNotFoundError(f"Missing grouped wavs: {missing[:5]}")
    print(f"[done] verified {len(rows)} grouped FEIS v3 wav triplets")


if __name__ == "__main__":
    main()
