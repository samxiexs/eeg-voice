from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))


DEFAULT_STAGE_ORDER = ("clearing", "stimulus_like", "thinking", "overt_like")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export KaraOne per-subject EEG bundles to FEIS-style wide CSV files "
            "(one row per trial/stage/time sample, one column per EEG channel)."
        )
    )
    parser.add_argument("--data-root", default=str(BUNDLE_DIR.parent / "data" / "karaone"))
    parser.add_argument("--out-dir", default=None, help="default: <data-root>/eeg_csv")
    parser.add_argument(
        "--subjects",
        default=None,
        help="comma-separated subject ids; default exports every subject in trials.csv",
    )
    parser.add_argument(
        "--stages",
        default=",".join(DEFAULT_STAGE_ORDER),
        help="comma-separated stages, e.g. thinking,overt_like; default exports all four stages",
    )
    parser.add_argument(
        "--split-by-stage",
        action="store_true",
        help="also write <stage>_eeg.csv under each subject directory",
    )
    parser.add_argument(
        "--no-full",
        action="store_true",
        help="only valid with --split-by-stage; skip the subject-level full_eeg.csv",
    )
    parser.add_argument(
        "--drop-padding",
        action="store_true",
        help="drop padded rows after the valid EEG length instead of preserving them with valid=false",
    )
    parser.add_argument(
        "--limit-trials-per-subject",
        type=int,
        default=None,
        help="debug/export subset: keep only the first N trials for each subject",
    )
    parser.add_argument("--sfreq-hz", type=float, default=256.0)
    parser.add_argument(
        "--generic-channel-names",
        action="store_true",
        help="force Ch001...Ch062 instead of channel_names stored in the subject bundle",
    )
    parser.add_argument("--float-format", default=".10g", help="Python format spec for EEG values")
    parser.add_argument("--time-format", default=".9f", help="Python format spec for Time:256Hz")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing CSV outputs")
    parser.add_argument("--dry-run", action="store_true", help="estimate rows without writing CSV files")
    return parser.parse_args()


def _split_csv_arg(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _trial_key(subject: str, trial_index: int) -> tuple[str, int]:
    return str(subject), int(trial_index)


def _segment_key(subject: str, trial_index: int, stage: str) -> tuple[str, int, str]:
    return str(subject), int(trial_index), str(stage)


def _channel_names(n_channels: int, bundle: np.lib.npyio.NpzFile, generic: bool) -> list[str]:
    if not generic and "channel_names" in bundle.files:
        names = [str(item) for item in bundle["channel_names"].tolist()]
        if len(names) == n_channels:
            return names
    return [f"Ch{idx:03d}" for idx in range(1, n_channels + 1)]


def _npz_array_shape(npz_path: Path, array_name: str) -> tuple[int, ...]:
    """Read an array shape from an NPZ member header without loading its data."""
    member_name = f"{array_name}.npy"
    with zipfile.ZipFile(npz_path) as archive:
        if member_name not in archive.namelist():
            raise KeyError(f"{npz_path} does not contain {member_name}")
        with archive.open(member_name) as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"Unsupported npy header version {version} in {member_name}")
            return tuple(int(dim) for dim in shape)


def _open_writer(path: Path, header: list[str], overwrite: bool) -> tuple[csv.writer, object]:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.writer(handle)
    writer.writerow(header)
    return writer, handle


def _format_float(value: float, spec: str) -> str:
    return format(float(value), spec)


def _selected_subjects(trial_rows: Iterable[dict[str, str]], requested: tuple[str, ...]) -> list[str]:
    subjects = sorted({row["subject_id"] for row in trial_rows})
    if not requested:
        return subjects
    unknown = sorted(set(requested) - set(subjects))
    if unknown:
        raise ValueError(f"Unknown subject(s): {', '.join(unknown)}")
    return list(requested)


def _estimate_subject_rows(
    bundle_path: Path,
    bundle: np.lib.npyio.NpzFile,
    trial_rows: list[dict[str, str]],
    stages: tuple[str, ...],
    drop_padding: bool,
) -> int:
    trial_indices = bundle["trial_indices"].astype(np.int32).tolist()
    trial_to_pos = {int(trial): idx for idx, trial in enumerate(trial_indices)}
    total = 0
    for trial in trial_rows:
        pos = trial_to_pos[int(trial["trial_index"])]
        for stage in stages:
            stage_shape = _npz_array_shape(bundle_path, f"stage__{stage}")
            n_time = int(stage_shape[-1])
            valid_key = f"stage__{stage}__valid_lengths"
            valid_len = int(bundle[valid_key][pos]) if valid_key in bundle.files else n_time
            total += min(valid_len, n_time) if drop_padding else n_time
    return total


def export_subject(
    *,
    subject: str,
    data_root: Path,
    out_dir: Path,
    trial_rows: list[dict[str, str]],
    segment_by_key: dict[tuple[str, int, str], dict[str, str]],
    stages: tuple[str, ...],
    sfreq_hz: float,
    float_format: str,
    time_format: str,
    split_by_stage: bool,
    no_full: bool,
    drop_padding: bool,
    overwrite: bool,
    dry_run: bool,
    generic_channel_names: bool,
) -> dict[str, str | int | float]:
    bundle_path = data_root / "subjects" / f"{subject}.npz"
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)

    with np.load(bundle_path, allow_pickle=True) as bundle:
        trial_indices = bundle["trial_indices"].astype(np.int32).tolist()
        trial_to_pos = {int(trial): idx for idx, trial in enumerate(trial_indices)}

        missing_trials = [row["trial_index"] for row in trial_rows if int(row["trial_index"]) not in trial_to_pos]
        if missing_trials:
            raise ValueError(f"{subject} missing trial indices in bundle: {missing_trials[:8]}")

        for stage in stages:
            if f"stage__{stage}" not in bundle.files:
                raise KeyError(f"{bundle_path} does not contain stage__{stage}")

        stage_shapes = {stage: _npz_array_shape(bundle_path, f"stage__{stage}") for stage in stages}
        first_stage = stages[0]
        n_channels = int(stage_shapes[first_stage][1])
        for stage in stages:
            stage_channels = int(stage_shapes[stage][1])
            if stage_channels != n_channels:
                raise ValueError(f"{subject} {stage} has {stage_channels} channels, expected {n_channels}")

        channels = _channel_names(n_channels, bundle, generic=generic_channel_names)
        header = [
            f"Time:{int(sfreq_hz) if float(sfreq_hz).is_integer() else sfreq_hz:g}Hz",
            "subject_id",
            "trial_index",
            "label",
            "stage",
            "sample_index",
            "valid",
            "audio_path",
            *channels,
        ]
        subject_dir = out_dir / subject
        full_csv = subject_dir / "full_eeg.csv"
        stage_csvs = {stage: subject_dir / f"{stage}_eeg.csv" for stage in stages}

        if dry_run:
            n_rows = _estimate_subject_rows(bundle_path, bundle, trial_rows, stages, drop_padding)
            return {
                "subject_id": subject,
                "csv_path": "" if no_full else str(full_csv),
                "stage_csv_paths": ";".join(str(stage_csvs[stage]) for stage in stages) if split_by_stage else "",
                "n_trials": len(trial_rows),
                "n_rows": n_rows,
                "stages": ";".join(stages),
                "n_channels": n_channels,
                "sfreq_hz": sfreq_hz,
                "drop_padding": str(bool(drop_padding)).lower(),
                "dry_run": "true",
            }

        stage_arrays = {stage: bundle[f"stage__{stage}"] for stage in stages}
        stage_valid_lengths = {
            stage: (
                bundle[f"stage__{stage}__valid_lengths"].astype(np.int32)
                if f"stage__{stage}__valid_lengths" in bundle.files
                else np.full(stage_shapes[stage][0], stage_shapes[stage][-1], dtype=np.int32)
            )
            for stage in stages
        }

        writers: dict[str, csv.writer] = {}
        handles: list[object] = []
        try:
            if not no_full:
                writer, handle = _open_writer(full_csv, header, overwrite)
                writers["full"] = writer
                handles.append(handle)
            if split_by_stage:
                for stage, path in stage_csvs.items():
                    writer, handle = _open_writer(path, header, overwrite)
                    writers[stage] = writer
                    handles.append(handle)

            n_rows = 0
            for trial in trial_rows:
                trial_index = int(trial["trial_index"])
                pos = trial_to_pos[trial_index]
                label = trial["label"]
                audio_path = trial["audio_path"]
                for stage in stages:
                    seg = segment_by_key.get(_segment_key(subject, trial_index, stage))
                    if seg is None:
                        raise KeyError(f"Missing segment row for {subject} trial={trial_index} stage={stage}")
                    arr = stage_arrays[stage][pos]  # [channels, time]
                    valid_len = int(stage_valid_lengths[stage][pos])
                    n_time = int(arr.shape[-1])
                    n_emit = min(valid_len, n_time) if drop_padding else n_time
                    for sample_idx in range(n_emit):
                        valid = sample_idx < valid_len
                        values = arr[:, sample_idx]
                        row = [
                            _format_float(sample_idx / float(sfreq_hz), time_format),
                            subject,
                            trial_index,
                            label,
                            stage,
                            sample_idx,
                            "true" if valid else "false",
                            audio_path,
                            *(_format_float(value, float_format) for value in values),
                        ]
                        if "full" in writers:
                            writers["full"].writerow(row)
                        if stage in writers:
                            writers[stage].writerow(row)
                        n_rows += 1
        finally:
            for handle in handles:
                handle.close()

    return {
        "subject_id": subject,
        "csv_path": "" if no_full else str(full_csv),
        "stage_csv_paths": ";".join(str(stage_csvs[stage]) for stage in stages) if split_by_stage else "",
        "n_trials": len(trial_rows),
        "n_rows": n_rows,
        "stages": ";".join(stages),
        "n_channels": n_channels,
        "sfreq_hz": sfreq_hz,
        "drop_padding": str(bool(drop_padding)).lower(),
        "dry_run": "false",
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else data_root / "eeg_csv"
    stages = _split_csv_arg(args.stages)
    if not stages:
        raise ValueError("At least one stage is required")
    if args.no_full and not args.split_by_stage:
        raise ValueError("--no-full requires --split-by-stage")

    trials_path = data_root / "trials.csv"
    segments_path = data_root / "segments.csv"
    trial_rows = _read_csv(trials_path)
    segment_rows = _read_csv(segments_path)
    subject_filter = _split_csv_arg(args.subjects)
    subjects = _selected_subjects(trial_rows, subject_filter)

    trial_by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in trial_rows:
        trial_by_subject[row["subject_id"]].append(row)
    for rows in trial_by_subject.values():
        rows.sort(key=lambda item: int(item["trial_index"]))

    segment_by_key = {
        _segment_key(row["subject_id"], int(row["trial_index"]), row["segment_stage"]): row
        for row in segment_rows
    }

    manifest_rows = []
    for subject in subjects:
        rows = trial_by_subject[subject]
        if args.limit_trials_per_subject is not None:
            rows = rows[: max(0, int(args.limit_trials_per_subject))]
        print(f"[export] subject={subject} trials={len(rows)} stages={','.join(stages)}")
        manifest_rows.append(
            export_subject(
                subject=subject,
                data_root=data_root,
                out_dir=out_dir,
                trial_rows=rows,
                segment_by_key=segment_by_key,
                stages=stages,
                sfreq_hz=float(args.sfreq_hz),
                float_format=str(args.float_format),
                time_format=str(args.time_format),
                split_by_stage=bool(args.split_by_stage),
                no_full=bool(args.no_full),
                drop_padding=bool(args.drop_padding),
                overwrite=bool(args.overwrite),
                dry_run=bool(args.dry_run),
                generic_channel_names=bool(args.generic_channel_names),
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / ("manifest_dry_run.csv" if args.dry_run else "manifest.csv")
    manifest_fields = [
        "subject_id",
        "csv_path",
        "stage_csv_paths",
        "n_trials",
        "n_rows",
        "stages",
        "n_channels",
        "sfreq_hz",
        "drop_padding",
        "dry_run",
    ]
    if manifest_path.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"{manifest_path} already exists; pass --overwrite to replace it")
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    total_rows = sum(int(row["n_rows"]) for row in manifest_rows)
    print(f"[done] wrote {manifest_path} subjects={len(manifest_rows)} rows={total_rows}")


if __name__ == "__main__":
    main()
