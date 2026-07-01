from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from scripts.plot_karaone_v11_training import plot_run as plot_v11_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot KaraOne v12 token/time-anchor training curves.")
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = plot_run(args.run_dir)
    print(json.dumps({"figures": [str(path) for path in paths]}, ensure_ascii=False, indent=2))


def plot_run(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir).expanduser().resolve()
    paths = list(plot_v11_run(run_dir))
    hist_path = run_dir / "metrics" / "history.json"
    payload = json.loads(hist_path.read_text(encoding="utf-8"))
    rows = payload.get("history", [])
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths.append(plot_lines(rows, out_dir / "time_anchor_metrics.png", "Time Anchor Metrics", ["subject_val_lag_mae_sec", "subject_test_lag_mae_sec", "subject_val_onset_mae_sec", "subject_test_onset_mae_sec", "subject_val_active_iou", "subject_test_active_iou"]))
    return paths


def plot_lines(rows: list[dict], path: Path, title: str, keys: list[str]) -> Path:
    epochs = [int(row["epoch"]) for row in rows]
    plt.figure(figsize=(10, 5))
    plotted = False
    for key in keys:
        vals = [row.get(key) for row in rows]
        if any(isinstance(v, (int, float)) for v in vals):
            plt.plot(epochs, [float(v) if isinstance(v, (int, float)) else float("nan") for v in vals], label=key)
            plotted = True
    if not plotted:
        plt.plot(epochs, [0.0 for _ in epochs], label="no_numeric_values")
    plt.title(title)
    plt.xlabel("epoch")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


if __name__ == "__main__":
    main()
