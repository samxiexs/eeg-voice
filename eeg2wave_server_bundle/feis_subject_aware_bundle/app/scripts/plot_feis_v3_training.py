"""Regenerate FEIS v3 training figures from metrics/history.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from scripts.train_feis_v3 import _write_figures


def parse_args():
    p = argparse.ArgumentParser(description="Plot FEIS v3 training metrics.")
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    payload = json.loads((run_dir / "metrics" / "history.json").read_text(encoding="utf-8"))
    _write_figures(run_dir, payload.get("history", []))
    print(f"[done] figures under {run_dir / 'figures'}")


if __name__ == "__main__":
    main()
