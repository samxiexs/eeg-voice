from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.karaone_recon.data import KaraOneTrialDataset
from src.karaone_recon.eval import evaluate
from src.karaone_recon.model import KaraOneConfig, KaraOneEEG2Codec
from src.karaone_recon.targets import KaraOneTargets
from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a KaraOne reconstruction checkpoint.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "subject_test"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = KaraOneEEG2Codec(KaraOneConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    targets = KaraOneTargets(resolve_bundle_path(cfg["data"]["target_cache"], BUNDLE_DIR), data_root=root)
    split_protocol = "subject_holdout" if args.split == "subject_test" else str(cfg["data"].get("split_protocol", "trial"))
    ds = KaraOneTrialDataset(
        data_root=root,
        targets=targets,
        split=args.split,
        stages=tuple(ckpt["stages"]),
        split_protocol=split_protocol,
        heldout_subjects=cfg["data"].get("heldout_subjects", ["P02", "MM21"]),
        eeg_len=int(cfg["data"].get("eeg_len", 1280)),
    )
    metrics = evaluate(model, ds, targets, device=device, batch_size=int(cfg["train"].get("batch_size", 48)))
    if args.out:
        out = resolve_bundle_path(args.out, BUNDLE_DIR)
    else:
        out = ensure_dir(Path(args.checkpoint).resolve().parents[1] / "metrics") / f"{args.split}_metrics.json"
    write_json(out, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
