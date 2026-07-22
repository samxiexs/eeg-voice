#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from tqdm import tqdm


APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_0722.data import resolve_config_path  # noqa: E402
from src.open_vocab_0722.audio_io import wav_info  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_speaker(path: Path, root: Path, corpus: str) -> str:
    relative = path.relative_to(root)
    parts = relative.parts
    if corpus == "libritts":
        return parts[0] if len(parts) > 1 else path.parent.name
    # AISHELL-1 normally stores wav/{train,dev,test}/SPEAKER/*.wav.
    return path.parent.name


def scan(root: Path, language: str, corpus: str) -> list[dict[str, object]]:
    files = sorted(path for path in root.rglob("*") if path.suffix.lower() == ".wav")
    if not files:
        raise FileNotFoundError(f"No WAV files found under {root}")
    rows: list[dict[str, object]] = []
    for path in tqdm(files, desc=f"[0722 public] scan {language}"):
        sample_rate, frames = wav_info(path)
        if frames <= 0 or sample_rate <= 0:
            continue
        rows.append(
            {
                "language": language,
                "corpus": corpus,
                "speaker_id": f"{corpus}:{infer_speaker(path, root, corpus)}",
                "source_path": str(path.resolve()),
                "source_sha256": sha256(path),
                "source_sample_rate": sample_rate,
                "source_frames": frames,
                "duration_sec": frames / sample_rate,
            }
        )
    return rows


def split_and_segment(
    files: list[dict[str, object]],
    *,
    max_hours: float,
    seed: int,
    segment_sec: float = 2.0,
) -> list[dict[str, object]]:
    by_speaker: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in files:
        by_speaker[str(row["speaker_id"])].append(row)
    speakers = sorted(by_speaker)
    random.Random(seed).shuffle(speakers)
    validation_count = max(1, round(len(speakers) * 0.10)) if len(speakers) > 1 else 0
    validation = set(speakers[:validation_count])
    ordered = sorted(files, key=lambda row: hashlib.sha256(f"{seed}:{row['source_path']}".encode()).hexdigest())
    budget = float(max_hours) * 3600.0
    used = 0.0
    segments: list[dict[str, object]] = []
    for row in ordered:
        if used >= budget:
            break
        sample_rate = int(row["source_sample_rate"])
        segment_samples = max(1, round(segment_sec * sample_rate))
        frames = int(row["source_frames"])
        for start in range(0, frames, segment_samples):
            valid = min(segment_samples, frames - start)
            if valid < round(0.25 * sample_rate) or used >= budget:
                continue
            duration = valid / sample_rate
            if used + duration > budget:
                break
            key_material = f"{row['source_sha256']}:{start}:{valid}".encode()
            segments.append(
                {
                    **row,
                    "audio_key": hashlib.sha256(key_material).hexdigest(),
                    "split": "validation" if row["speaker_id"] in validation else "train",
                    "segment_start_sample": start,
                    "segment_valid_samples": valid,
                    "segment_duration_sec": duration,
                }
            )
            used += duration
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic, speaker-disjoint LibriTTS/AISHELL manifest")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--english-root", type=Path, required=True)
    parser.add_argument("--chinese-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = resolve_config_path(config_path, cfg["paths"]["public_audio_manifest"])
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest already exists; use --overwrite: {output}")
    audit_path = resolve_config_path(config_path, cfg["paths"]["leakage_audit"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    forbidden = set(audit["heldout_generation_audio_sha256"])
    seed = int(cfg["public_audio"]["seed"])
    english = [row for row in scan(args.english_root.resolve(), "en", "libritts") if row["source_sha256"] not in forbidden]
    chinese = [row for row in scan(args.chinese_root.resolve(), "zh", "aishell1") if row["source_sha256"] not in forbidden]
    english_rows = split_and_segment(english, max_hours=float(cfg["public_audio"]["english_hours"]), seed=seed)
    chinese_rows = split_and_segment(chinese, max_hours=float(cfg["public_audio"]["chinese_hours"]), seed=seed + 1)
    for language, selected, requested in (
        ("en", english_rows, float(cfg["public_audio"]["english_hours"])),
        ("zh", chinese_rows, float(cfg["public_audio"]["chinese_hours"])),
    ):
        observed = sum(float(row["segment_duration_sec"]) for row in selected) / 3600.0
        if observed < requested - 0.01:
            raise ValueError(f"{language} corpus supplies only {observed:.2f} of requested {requested:.2f} hours")
        if {row["split"] for row in selected} != {"train", "validation"}:
            raise ValueError(f"{language} selection does not contain both speaker-disjoint train and validation")
    rows = english_rows + chinese_rows
    if any(str(row["source_sha256"]) in forbidden for row in rows):
        raise RuntimeError("A held-out EEG reference audio entered the public speech manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "output": str(output),
        "segments": len(rows),
        "hours": sum(float(row["segment_duration_sec"]) for row in rows) / 3600.0,
        "train_speakers": len({str(row["speaker_id"]) for row in rows if row["split"] == "train"}),
        "validation_speakers": len({str(row["speaker_id"]) for row in rows if row["split"] == "validation"}),
        "transcripts_used": False,
        "heldout_hash_overlap": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
