"""Run EEG-only direct training separately for each FEIS stage.

Default stages:
  stimuli, thinking, speaking, resting

This script is intentionally orchestration-only. Each stage gets its own
checkpoint/run directory, so results can be compared without mixing stage ids.

Examples:
  python scripts/direct_stage_sweep.py --epochs 80
  python scripts/direct_stage_sweep.py --max-steps 2 --run-recon-eval --qc-cells 4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.utils import load_simple_yaml, resolve_bundle_path


def parse_args():
    p = argparse.ArgumentParser(description="Train direct EEG-only models stage by stage.")
    p.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "direct_eeg2speech.yaml"))
    p.add_argument("--stages", default="stimuli,thinking,speaking,resting")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--run-suffix", default="stage_only_v1")
    p.add_argument("--device", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--run-recon-eval", action="store_true")
    p.add_argument("--run-synthesize", action="store_true")
    p.add_argument("--split", default="test_holdout")
    p.add_argument("--qc-cells", type=int, default=24)
    p.add_argument("--save-wav", type=int, default=12)
    p.add_argument("--synth-limit", type=int, default=24)
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(BUNDLE_DIR), check=True)


def main() -> None:
    args = parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if not stages:
        raise ValueError("No stages requested")
    cfg = load_simple_yaml(args.config)
    output_root = resolve_bundle_path(cfg["output"]["root"], BUNDLE_DIR)

    py = sys.executable
    for stage in stages:
        print(f"\n===== Direct EEG-only stage: {stage} =====", flush=True)
        train_cmd = [
            py,
            "scripts/direct_train.py",
            "--config",
            args.config,
            "--stages",
            stage,
            "--run-suffix",
            args.run_suffix,
        ]
        if args.epochs is not None:
            train_cmd += ["--epochs", str(args.epochs)]
        if args.device:
            train_cmd += ["--device", args.device]
        if args.max_steps is not None:
            train_cmd += ["--max-steps", str(args.max_steps)]
        _run(train_cmd)

        ckpt = output_root / f"direct_{stage}_{args.run_suffix}" / "checkpoints" / "best.pt"
        if args.run_recon_eval:
            recon_cmd = [
                py,
                "scripts/direct_recon_eval.py",
                "--config",
                args.config,
                "--checkpoint",
                str(ckpt),
                "--split",
                args.split,
                "--qc-cells",
                str(args.qc_cells),
                "--save-wav",
                str(args.save_wav),
            ]
            if args.device:
                recon_cmd += ["--device", args.device]
            _run(recon_cmd)

        if args.run_synthesize:
            synth_cmd = [
                py,
                "scripts/direct_synthesize.py",
                "--config",
                args.config,
                "--checkpoint",
                str(ckpt),
                "--split",
                args.split,
                "--limit",
                str(args.synth_limit),
            ]
            if args.device:
                synth_cmd += ["--device", args.device]
            _run(synth_cmd)

    print("\n[done] stage sweep complete", flush=True)


if __name__ == "__main__":
    main()
