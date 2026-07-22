#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ORDER = ("label_free_dense", "global_clip", "local_alignment", "semantic_text", "adapter_moe")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a lineage-safe OpenVoice 0722 ablation config")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=ORDER, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve(); cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = (config_path.parents[1] / "../artifacts" / f"open_vocab_0722_ablation_{args.stage}").resolve()
    cfg["version"] = f"openvoice-eeg-0722-ablation-{args.stage}-v1"
    cfg["paths"]["output_root"] = str(root)
    cfg["paths"]["eeg_pretrain_checkpoint"] = str(root / "eeg_pretrain/checkpoints/best.pt")
    cfg["paths"]["eeg_checkpoint"] = str(root / "eeg/checkpoints/selected.pt")
    cfg["paths"]["validation_report"] = str(root / "evaluation/validation_report.json")
    cfg["paths"]["validation_gate"] = str(root / "evaluation/validation_gate.json")
    cfg["eeg_model"]["adapter_moe_enabled"] = args.stage == "adapter_moe"
    enabled = ORDER.index(args.stage)
    cfg["loss"]["clip_global"] = 1.0 if enabled >= 1 else 0.0
    cfg["loss"]["clip_local"] = 0.5 if enabled >= 2 else 0.0
    cfg["loss"]["same_label_semantic"] = 0.15 if enabled >= 3 else 0.0
    cfg["loss"]["text"] = 0.05 if enabled >= 3 else 0.0
    for name in ("eeg_output_root", "audio_output_root", "subject_split_file", "label_holdout_file", "montage_registry", "track_b_output_root", "track_b_montage_registry"):
        value = Path(cfg["data"][name]); cfg["data"][name] = str(value if value.is_absolute() else (config_path.parent / value).resolve())
    for name in ("audio_checkpoint", "project_audio_cache", "teacher_cache", "public_audio_manifest", "public_audio_cache", "leakage_audit", "encodec_model"):
        value = Path(cfg["paths"][name]); cfg["paths"][name] = str(value if value.is_absolute() else (config_path.parent / value).resolve())
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__": main()
