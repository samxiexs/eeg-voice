#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_FEIS_ROOT = Path(
    "/Users/samxie/Research/EEG-Voice/ref_github/speech_decoding/data/processed/thinking_waveform_pairs/feis"
)


def normalize_subject_id(subject_id: str | int) -> str:
    text = str(subject_id)
    return text.zfill(2) if text.isdigit() else text


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def protocol_s_split_counts(trial_count: int, label_count: int) -> tuple[int, int, int]:
    repeats = trial_count // max(label_count, 1)
    return label_count * max(repeats - 2, 0), label_count, label_count


def format_table(rows: list[list[object]], headers: list[str]) -> str:
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))
    lines = ["  ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit FEIS subject-aware metadata and split protocols.")
    parser.add_argument("--feis-root", type=Path, default=DEFAULT_FEIS_ROOT)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    feis_root = args.feis_root
    trials = read_csv_rows(feis_root / "trials.csv")
    segments = read_csv_rows(feis_root / "segments.csv")
    manifest = json.loads((feis_root / "manifest.json").read_text(encoding="utf-8"))

    label_hashes = defaultdict(set)
    subject_label_hashes = defaultdict(set)
    subject_stage_counts = defaultdict(Counter)
    subject_labels = defaultdict(set)
    subject_audio_durations = defaultdict(set)
    subject_eeg_durations = defaultdict(set)
    subject_eeg_channels = defaultdict(set)
    subject_clean = {}
    subject_audio_sources = defaultdict(set)

    for row in trials:
        subject_id = normalize_subject_id(row["subject_id"])
        label = str(row["label"])
        subject_labels[subject_id].add(label)
        if "audio_sha1" in row:
            label_hashes[label].add(str(row["audio_sha1"]))
            subject_label_hashes[(subject_id, label)].add(str(row["audio_sha1"]))
        if "audio_duration_sec" in row:
            subject_audio_durations[subject_id].add(float(row["audio_duration_sec"]))
        if "audio_source_kind" in row:
            subject_audio_sources[subject_id].add(str(row["audio_source_kind"]))
        if "is_clean_subject" in row:
            subject_clean[subject_id] = str(row["is_clean_subject"]).lower() == "true"

    for row in segments:
        subject_id = normalize_subject_id(row["subject_id"])
        stage = str(row["segment_stage"])
        subject_stage_counts[subject_id][stage] += 1
        if "eeg_num_samples" in row and "eeg_sfreq_hz" in row:
            eeg_duration = float(row["eeg_num_samples"]) / float(row["eeg_sfreq_hz"])
            subject_eeg_durations[subject_id].add(round(eeg_duration, 6))
        if "eeg_num_channels" in row:
            subject_eeg_channels[subject_id].add(int(row["eeg_num_channels"]))

    table_rows = []
    per_subject = []
    for subject_id in sorted(subject_labels):
        labels = len(subject_labels[subject_id])
        trials_per_subject = sum(1 for row in trials if normalize_subject_id(row["subject_id"]) == subject_id)
        train_count, val_count, test_count = protocol_s_split_counts(trials_per_subject, labels)
        table_rows.append([subject_id, labels, trials_per_subject, train_count, val_count, test_count])
        per_subject.append(
            {
                "subject_id": subject_id,
                "labels": labels,
                "trials": trials_per_subject,
                "thinking_trials": int(subject_stage_counts[subject_id].get("thinking", 0)),
                "speaking_trials": int(subject_stage_counts[subject_id].get("speaking", 0)),
                "audio_duration_sec": sorted(subject_audio_durations[subject_id]),
                "eeg_duration_sec": sorted(subject_eeg_durations[subject_id]),
                "eeg_channels": sorted(subject_eeg_channels[subject_id]),
                "train": train_count,
                "val": val_count,
                "test": test_count,
                "is_clean_subject": subject_clean.get(subject_id, subject_id != "05"),
                "audio_source_kinds": sorted(subject_audio_sources[subject_id]),
            }
        )

    label_hash_counts = {label: len(hashes) for label, hashes in sorted(label_hashes.items())}
    subject_label_uniqueness = {
        f"{subject_id}:{label}": len(hashes)
        for (subject_id, label), hashes in sorted(subject_label_hashes.items())
    }
    within_subject_repetitions = defaultdict(Counter)
    for row in trials:
        subject_id = normalize_subject_id(row["subject_id"])
        within_subject_repetitions[subject_id][str(row["label"])] += 1

    anomaly_05 = next((item for item in per_subject if item["subject_id"] == "05"), None)
    payload = {
        "feis_root": str(feis_root),
        "subject_table": [
            {
                "subject_id": row[0],
                "labels": row[1],
                "trials": row[2],
                "train": row[3],
                "val": row[4],
                "test": row[5],
            }
            for row in table_rows
        ],
        "per_subject": per_subject,
        "within_subject_repetitions": {
            subject_id: dict(counter)
            for subject_id, counter in sorted(within_subject_repetitions.items())
        },
        "cross_subject_unique_hashes_per_label": label_hash_counts,
        "unique_hashes_per_subject_label": subject_label_uniqueness,
        "manifest_clean_subject_ids": manifest.get("clean_subject_ids", []),
        "manifest_anomalous_subject_ids": manifest.get("anomalous_subject_ids", []),
        "protocol_definitions": {
            "Protocol S": "Within-subject evaluation with subject-known template bank.",
            "Protocol G": "Pooled multi-subject evaluation over seen subjects, headline metrics exclude anomalous subject 05.",
            "Protocol U": "Leave-one-subject-out evaluation on unseen subjects.",
        },
        "subject_05_anomaly": anomaly_05,
    }

    print("FEIS Subject Table")
    print(format_table(table_rows, headers=["Subject", "Labels", "Trials", "Train", "Val", "Test"]))
    print()
    print("Cross-subject unique waveform hashes per label")
    for label, count in label_hash_counts.items():
        print(f"  {label}: {count}")
    print()
    if anomaly_05 is not None:
        print("Subject 05 anomaly summary")
        print(json.dumps(anomaly_05, ensure_ascii=False, indent=2))
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved audit JSON to {args.output_json}")


if __name__ == "__main__":
    main()
