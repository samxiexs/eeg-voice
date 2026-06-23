from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import wave
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze processed KaraOne data.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--out-dir", default=str(BUNDLE_DIR.parent / "reports"))
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_scalar(value: str):
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [_parse_scalar(part.strip()) for part in inner.split(",") if part.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def load_simple_yaml(path: str | Path) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not value.strip():
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value.strip())
    return root


def resolve_bundle_path(path: str | Path, base_dir: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(base_dir) / candidate


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return float(handle.getnframes() / handle.getframerate())


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    out_dir = ensure_dir(resolve_bundle_path(args.out_dir, BUNDLE_DIR))
    trial_rows = list(csv.DictReader((root / "trials.csv").open("r", encoding="utf-8", newline="")))
    segment_rows = list(csv.DictReader((root / "segments.csv").open("r", encoding="utf-8", newline="")))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    subjects = sorted({row["subject_id"] for row in trial_rows})
    labels = sorted({row["label"] for row in trial_rows})
    label_counts = Counter(row["label"] for row in trial_rows)
    subject_counts = Counter(row["subject_id"] for row in trial_rows)
    stage_counts = Counter(row["segment_stage"] for row in segment_rows)

    stage_lengths = {}
    for stage in sorted(stage_counts):
        valid = [int(row["eeg_valid_num_samples"]) for row in segment_rows if row["segment_stage"] == stage]
        target = [int(row["eeg_num_samples"]) for row in segment_rows if row["segment_stage"] == stage]
        stage_lengths[stage] = {
            "n": len(valid),
            "valid_min": min(valid),
            "valid_median": statistics.median(valid),
            "valid_p95": sorted(valid)[int(0.95 * (len(valid) - 1))],
            "valid_max": max(valid),
            "target_lengths": dict(Counter(target).most_common()),
        }

    audio_durations = []
    missing_audio = []
    for row in trial_rows:
        path = root / row["audio_path"]
        if not path.exists():
            missing_audio.append(str(path.relative_to(root)))
            continue
        audio_durations.append(wav_duration(path))
    npz_entries = {}
    for path in sorted((root / "subjects").glob("*.npz")):
        with zipfile.ZipFile(path) as archive:
            npz_entries[path.stem] = sorted(archive.namelist())

    summary = {
        "root": str(root),
        "manifest": manifest,
        "subject_count": len(subjects),
        "subjects": subjects,
        "label_count": len(labels),
        "labels": labels,
        "trial_count": len(trial_rows),
        "segment_count": len(segment_rows),
        "label_counts": dict(sorted(label_counts.items())),
        "subject_trial_counts": dict(sorted(subject_counts.items())),
        "stage_counts": dict(sorted(stage_counts.items())),
        "stage_lengths": stage_lengths,
        "audio_duration_sec": {
            "n": len(audio_durations),
            "min": min(audio_durations) if audio_durations else None,
            "median": statistics.median(audio_durations) if audio_durations else None,
            "max": max(audio_durations) if audio_durations else None,
        },
        "missing_audio": missing_audio,
        "subject_bundle_count": len(npz_entries),
        "npz_entry_names": next(iter(npz_entries.values())) if npz_entries else [],
        "all_npz_same_structure": len({tuple(value) for value in npz_entries.values()}) <= 1,
    }
    write_json(out_dir / "karaone_data_summary.json", summary)

    lines = [
        "# KaraOne data analysis",
        "",
        f"- Data root: `{root}`",
        f"- Subjects: {len(subjects)} (`{' '.join(subjects)}`)",
        f"- Trials: {len(trial_rows)}",
        f"- Segments: {len(segment_rows)}",
        f"- Labels: {len(labels)} (`{'`, `'.join(labels)}`)",
        f"- Audio duration median: {summary['audio_duration_sec']['median']:.3f}s",
        f"- Subject bundles: {len(npz_entries)}, same structure: {summary['all_npz_same_structure']}",
        "",
        "## Trial counts by subject",
        "",
    ]
    lines.extend(f"- {subject}: {subject_counts[subject]}" for subject in subjects)
    lines.extend(["", "## Label counts", ""])
    lines.extend(f"- {label}: {label_counts[label]}" for label in labels)
    lines.extend(["", "## Stage length summary", ""])
    for stage, payload in stage_lengths.items():
        lines.append(
            f"- {stage}: n={payload['n']}, valid min/median/p95/max="
            f"{payload['valid_min']}/{payload['valid_median']}/{payload['valid_p95']}/{payload['valid_max']}"
        )
    lines.extend(
        [
            "",
            "## Recommended experiment use",
            "",
            "- Primary positive-control task: `overt_like` EEG -> same-trial wav.",
            "- Main imagined-speech task: initialize from overt reconstruction, then fine-tune/evaluate `thinking` EEG -> same-trial overt wav.",
            "- Always report zero-EEG, mean-latent, and oracle-codec controls.",
            "",
        ]
    )
    (out_dir / "karaone_data_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_dir / 'karaone_data_summary.json'}")
    print(f"Wrote {out_dir / 'karaone_data_analysis.md'}")


if __name__ == "__main__":
    main()
