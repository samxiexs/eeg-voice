from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any


TYPE_ORDER = {
    "original": ("01", "original_reference"),
    "retrieval_diagnostic": ("02", "retrieval_diagnostic"),
    "pred_env_scaled": ("02", "pred_env_scaled_retrieval"),
    "generated_codec": ("03", "generated_codec"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group KaraOne synthesis wavs by sample for easier listening.")
    parser.add_argument("--wav-dir", required=True, type=Path)
    parser.add_argument("--manifest", default=None, type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    parser.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--clean", action="store_true", help="remove the grouped output directory before rebuilding it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = organize_wavs(
        wav_dir=args.wav_dir,
        manifest=args.manifest,
        out_dir=args.out_dir,
        mode=args.mode,
        clean=bool(args.clean),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def organize_wavs(
    *,
    wav_dir: str | Path,
    manifest: str | Path | None = None,
    out_dir: str | Path | None = None,
    mode: str = "symlink",
    clean: bool = False,
) -> dict[str, Any]:
    wav_dir = Path(wav_dir).expanduser().resolve()
    manifest = Path(manifest).expanduser().resolve() if manifest else wav_dir / "listening_manifest.csv"
    out_dir = Path(out_dir).expanduser().resolve() if out_dir else wav_dir / "grouped_wavs"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    by_sample_dir = out_dir / "by_sample"
    by_subject_dir = out_dir / "by_subject"
    flat_dir = out_dir / "flat"
    by_sample_dir.mkdir(parents=True, exist_ok=True)
    by_subject_dir.mkdir(parents=True, exist_ok=True)
    flat_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(manifest)
    index_rows: list[dict[str, Any]] = []
    linked = 0
    missing = 0
    for row in rows:
        sample_key = safe_name(row.get("sample_key", "sample"))
        subject = safe_name(row.get("subject", "unknown_subject"))
        wav_type = str(row.get("wav_type", "unknown"))
        source_name = str(row.get("file", ""))
        source = wav_dir / source_name
        if not source.exists():
            missing += 1
            continue
        order, type_name = TYPE_ORDER.get(wav_type, ("90", safe_name(wav_type)))
        ext = source.suffix or ".wav"
        grouped_name = f"{order}_{type_name}{ext}"
        flat_name = f"{sample_key}__{order}_{type_name}{ext}"
        sample_dir = by_sample_dir / sample_key
        subject_sample_dir = by_subject_dir / subject / sample_key
        sample_dir.mkdir(parents=True, exist_ok=True)
        subject_sample_dir.mkdir(parents=True, exist_ok=True)
        grouped_target = sample_dir / grouped_name
        subject_target = subject_sample_dir / grouped_name
        flat_target = flat_dir / flat_name
        link_file(source, grouped_target, mode=mode)
        link_file(source, subject_target, mode=mode)
        link_file(source, flat_target, mode=mode)
        linked += 3
        index_rows.append(
            {
                **row,
                "order": order,
                "grouped_file": str(grouped_target.relative_to(out_dir)),
                "subject_grouped_file": str(subject_target.relative_to(out_dir)),
                "flat_file": str(flat_target.relative_to(out_dir)),
                "source_file": str(source.relative_to(wav_dir)),
            }
        )

    write_csv(out_dir / "grouped_manifest.csv", index_rows)
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# KaraOne Grouped Wavs",
                "",
                "- `by_sample/<sample_key>/01_original_reference.wav`: ground-truth trial audio.",
                "- `by_subject/<subject>/<sample_key>/01_original_reference.wav`: same grouping under subject folders.",
                "- `by_sample/<sample_key>/02_pred_env_scaled_retrieval.wav`: semantic retrieval diagnostic wav.",
                "- `by_sample/<sample_key>/02_retrieval_diagnostic.wav`: v11 retrieval diagnostic wav.",
                "- `by_sample/<sample_key>/03_generated_codec.wav`: codec-flow generated wav attempt.",
                "- `flat/`: same files with sortable sample prefixes.",
                f"- link_mode: `{mode}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "event": "karaone_wavs_grouped",
        "wav_dir": str(wav_dir),
        "manifest": str(manifest),
        "out_dir": str(out_dir),
        "mode": mode,
        "manifest_rows": len(rows),
        "linked_files": linked,
        "missing_files": missing,
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def link_file(source: Path, target: Path, *, mode: str) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "hardlink":
        os.link(source, target)
    else:
        rel_source = os.path.relpath(source, start=target.parent)
        target.symlink_to(rel_source)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value)).strip("_") or "sample"


if __name__ == "__main__":
    main()
