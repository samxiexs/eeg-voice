#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an isolated 0722 EEG seed config while reusing the label-free audio prior/cache")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(15, 31, 47), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.config.resolve(); cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    base_output = (source.parents[1] / "../artifacts" / f"open_vocab_0722_v1_seed{args.seed}").resolve()
    cfg["version"] = f"openvoice-eeg-0722-v1-seed{args.seed}"
    cfg["run"]["seed"] = args.seed
    cfg["paths"]["output_root"] = str(base_output)
    cfg["paths"]["eeg_pretrain_checkpoint"] = str(base_output / "eeg_pretrain/checkpoints/best.pt")
    cfg["paths"]["eeg_checkpoint"] = str(base_output / "eeg/checkpoints/selected.pt")
    cfg["paths"]["validation_report"] = str(base_output / "evaluation/validation_report.json")
    cfg["paths"]["validation_gate"] = str(base_output / "evaluation/validation_gate.json")
    for name in ("eeg_output_root", "audio_output_root", "subject_split_file", "label_holdout_file", "montage_registry", "track_b_output_root", "track_b_montage_registry"):
        value = Path(cfg["data"][name]); cfg["data"][name] = str(value if value.is_absolute() else (source.parent / value).resolve())
    # The audio prior and immutable caches are shared. Audio checkpoint
    # validation uses the audio-only lineage scope.
    for name in ("audio_checkpoint", "project_audio_cache", "teacher_cache", "public_audio_manifest", "public_audio_cache", "leakage_audit", "encodec_model"):
        value = Path(cfg["paths"][name])
        cfg["paths"][name] = str(value.resolve() if value.is_absolute() else (source.parent / value).resolve())
    output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(output)


if __name__ == "__main__": main()
