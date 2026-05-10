#!/usr/bin/env python3
"""Prepare ignored one-subject/one-trial EEG-Voice dataset sample folders.

The script is intentionally conservative. It copies complete local examples
when they already exist, downloads small public metadata files, and writes
clear manual-access notes for datasets that require login, authorization, or
large file pulls.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path("data/voice_eeg_dataset_samples")
DEFAULT_TIMEOUT = 45


@dataclass
class RemoteFile:
    url: str
    relpath: str
    max_mb: float = 25.0
    required: bool = False


@dataclass
class LocalPattern:
    pattern: str
    dest: str
    limit: int = 20
    required: bool = False


@dataclass
class DatasetSpec:
    slug: str
    title: str
    category: str
    priority: str
    source_url: str
    sample_goal: str
    access_note: str
    local_patterns: list[LocalPattern] = field(default_factory=list)
    remote_files: list[RemoteFile] = field(default_factory=list)


def openneuro_file(dataset: str, path: str, relpath: str, max_mb: float = 25.0) -> RemoteFile:
    return RemoteFile(
        url=f"https://s3.amazonaws.com/openneuro.org/{dataset}/{path}",
        relpath=relpath,
        max_mb=max_mb,
    )


def zenodo_metadata(record_id: str, relpath: str = "remote/zenodo_record.json") -> RemoteFile:
    return RemoteFile(
        url=f"https://zenodo.org/api/records/{record_id}",
        relpath=relpath,
        max_mb=10.0,
    )


DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        slug="ds004408",
        title="EEG responses to continuous naturalistic speech",
        category="english",
        priority="P0",
        source_url="https://openneuro.org/datasets/ds004408",
        sample_goal="sub-001 run-01 BrainVision EEG + audio01 wav/TextGrid.",
        access_note="Public OpenNeuro. Full raw run can be large but local run-01 is already available.",
        local_patterns=[
            LocalPattern("data/raw/openneuro/ds004408/sub-001/eeg/sub-001_task-listening_run-01_eeg.*", "local/eeg", required=True),
            LocalPattern("data/meeting_examples/ds004408/raw/audio01.*", "local/stimuli", required=True),
            LocalPattern("outputs/probe_artifacts/ds004408/*", "probe_artifacts"),
        ],
        remote_files=[
            openneuro_file("ds004408", "dataset_description.json", "remote/dataset_description.json"),
            openneuro_file("ds004408", "participants.tsv", "remote/participants.tsv"),
        ],
    ),
    DatasetSpec(
        slug="weissbart_natural_speech",
        title="Weissbart natural speech EEG",
        category="english",
        priority="P1",
        source_url="https://zenodo.org/records/7086168",
        sample_goal="One subject continuous speech EEG and stimulus metadata.",
        access_note="Zenodo record is public; use record metadata first, then select a subject file manually if files are large.",
        remote_files=[zenodo_metadata("7086168")],
    ),
    DatasetSpec(
        slug="ds006434",
        title="ABR to natural speech and selective attention",
        category="english_controlled",
        priority="P0/P1",
        source_url="https://openneuro.org/datasets/ds006434",
        sample_goal="One dichotic subject EEG metadata/events plus one speech wav.",
        access_note="Public OpenNeuro. Full raw EEG can be large; current automatic sample uses metadata and previously probed snippets.",
        local_patterns=[
            LocalPattern("outputs/probe_artifacts/ds006434/*", "probe_artifacts", required=True),
        ],
        remote_files=[
            openneuro_file("ds006434", "dataset_description.json", "remote/dataset_description.json"),
            openneuro_file("ds006434", "sub-dichotic02/eeg/sub-dichotic02_task-exp2DichoticCortex_events.tsv", "remote/events.tsv"),
            openneuro_file("ds006434", "sub-dichotic02/eeg/sub-dichotic02_task-exp2DichoticCortex_channels.tsv", "remote/channels.tsv"),
            openneuro_file("ds006434", "stimuli/exp2Dichotic/wrinkle_alchemyst000.wav", "remote/stimuli/wrinkle_alchemyst000.wav", max_mb=80.0),
        ],
    ),
    DatasetSpec(
        slug="etard_continuous_speech_7086209",
        title="Etard continuous speech and competing speakers EEG",
        category="english",
        priority="P0/P1",
        source_url="https://zenodo.org/records/7086209",
        sample_goal="One English continuous speech / competing-speaker EEG subject with aligned audiobook metadata.",
        access_note="Zenodo public. Full HDF5/audio bundles can be large; automatic sample stores record metadata only.",
        remote_files=[zenodo_metadata("7086209")],
    ),
    DatasetSpec(
        slug="ds007591",
        title="Delineating neural contributions to EEG-based speech decoding",
        category="english_proxy",
        priority="P2",
        source_url="https://openneuro.org/datasets/ds007591",
        sample_goal="One minimally overt speech EDF + events sample.",
        access_note="Public OpenNeuro. Full EDF may be large; automatic sample stores metadata/probe snippets unless large downloads are allowed.",
        local_patterns=[
            LocalPattern("outputs/probe_artifacts/ds007591/*", "probe_artifacts", required=True),
        ],
        remote_files=[
            openneuro_file("ds007591", "dataset_description.json", "remote/dataset_description.json"),
            openneuro_file("ds007591", "participants.tsv", "remote/participants.tsv"),
        ],
    ),
    DatasetSpec(
        slug="sparrkulee_eegdash",
        title="SparrKULee / EEGDash speech corpus",
        category="english",
        priority="P1",
        source_url="https://eegdash.org/api/dataset/eegdash.dataset.NM000238.html",
        sample_goal="One EEGDash subject recording and speech metadata.",
        access_note="Use EEGDash metadata first; exact raw download route requires separate confirmation.",
        remote_files=[RemoteFile("https://eegdash.org/api/dataset/eegdash.dataset.NM000238.html", "remote/eegdash_record.html", max_mb=10.0)],
    ),
    DatasetSpec(
        slug="ds005345",
        title="Le Petit Prince Multi-talker",
        category="mandarin_synthetic",
        priority="P0",
        source_url="https://openneuro.org/datasets/ds005345",
        sample_goal="sub-01 run-1..4 preprocessed/derived EEG + single female/male/mix wav and annotation.",
        access_note="Public OpenNeuro. Local full sub-01 derived samples are already available.",
        local_patterns=[
            LocalPattern("data/meeting_examples/ds005345/raw/*", "local/stimuli", required=True),
            LocalPattern("data/derived/openneuro_full/ds005345/sub-01/run-*/*_full_eeg.npz", "local/eeg", required=True),
            LocalPattern("data/derived/openneuro_full/ds005345/annotation/*", "local/annotation"),
            LocalPattern("outputs/probe_artifacts/ds005345/*", "probe_artifacts"),
        ],
        remote_files=[
            openneuro_file("ds005345", "dataset_description.json", "remote/dataset_description.json"),
            openneuro_file("ds005345", "participants.tsv", "remote/participants.tsv"),
        ],
    ),
    DatasetSpec(
        slug="esaa_7078451",
        title="ESAA Mandarin auditory attention",
        category="mandarin",
        priority="P0",
        source_url="https://zenodo.org/records/7078451",
        sample_goal="One Mandarin AAD subject/trial with audio and attention label.",
        access_note="Zenodo record public. Current local cache has README/preprocess/baseline snippets; full files require selected pull.",
        local_patterns=[LocalPattern("outputs/probe_artifacts/zenodo-7078451/*", "probe_artifacts", required=True)],
        remote_files=[zenodo_metadata("7078451")],
    ),
    DatasetSpec(
        slug="nju_aad_7253438",
        title="NJU Mandarin AAD",
        category="mandarin",
        priority="P0/P1",
        source_url="https://zenodo.org/records/7253438",
        sample_goal="One Mandarin competing-speech subject/trial with audio and label.",
        access_note="Zenodo metadata can be downloaded; full file selection needs inspection.",
        remote_files=[zenodo_metadata("7253438")],
    ),
    DatasetSpec(
        slug="ds006465_3m_cpseed",
        title="3M-CPSEED Mandarin pinyin speech",
        category="mandarin_proxy",
        priority="P2",
        source_url="https://openneuro.org/datasets/ds006465",
        sample_goal="One overt/mouthed/imagined pinyin subject example.",
        access_note="Public OpenNeuro. Exact file paths need a dataset inventory before full sample pull.",
        remote_files=[openneuro_file("ds006465", "dataset_description.json", "remote/dataset_description.json")],
    ),
    DatasetSpec(
        slug="aasd_17413336",
        title="AASD spontaneous auditory attention switch decoding",
        category="mandarin",
        priority="P0/P1",
        source_url="https://zenodo.org/records/17413336",
        sample_goal="One Mandarin spontaneous attention-switch EEG subject with multi-speaker stimuli metadata.",
        access_note="Zenodo public. Full EEG and audio zips are large; automatic sample stores record metadata only.",
        remote_files=[zenodo_metadata("17413336")],
    ),
    DatasetSpec(
        slug="ms_aasd_17149387",
        title="MS-AASD mixed-speech attention switch decoding",
        category="mandarin",
        priority="P0/P1",
        source_url="https://zenodo.org/records/17149387",
        sample_goal="One Mandarin mixed-speech self-initiated attention-switch subject with metadata.",
        access_note="Zenodo public. Full EEG and audio bundles are large; automatic sample stores record metadata only.",
        remote_files=[zenodo_metadata("17149387")],
    ),
    DatasetSpec(
        slug="ds004718",
        title="LPPHK Cantonese natural speech",
        category="cantonese",
        priority="P0",
        source_url="https://openneuro.org/datasets/ds004718",
        sample_goal="sub-HK001 preprocessed EEG set + one Cantonese sentence wav and timing annotation.",
        access_note="Public OpenNeuro. Local meeting sample already contains a sentence wav and preprocessed set.",
        local_patterns=[
            LocalPattern("data/meeting_examples/ds004718/raw/*", "local", required=True),
            LocalPattern("outputs/probe_artifacts/ds004718/*", "probe_artifacts"),
        ],
        remote_files=[
            openneuro_file("ds004718", "dataset_description.json", "remote/dataset_description.json"),
            openneuro_file("ds004718", "participants.tsv", "remote/participants.tsv"),
        ],
    ),
    DatasetSpec(
        slug="cantonese_tone_syllable_7750292",
        title="Cantonese tone/syllable production ERP",
        category="cantonese_proxy",
        priority="P2",
        source_url="https://zenodo.org/records/7750292",
        sample_goal="One tone/syllable ERP subject/trial and stimulus table.",
        access_note="Zenodo metadata can be downloaded; file contents need inspection before EEG/audio pull.",
        remote_files=[zenodo_metadata("7750292")],
    ),
    DatasetSpec(
        slug="ds006104",
        title="EEG dataset for speech decoding",
        category="controlled_speech",
        priority="P0/P2",
        source_url="https://openneuro.org/datasets/ds006104",
        sample_goal="sub-P01/sub-S01 full derived EEG + events/channels + a few local stimuli wav.",
        access_note="Public OpenNeuro. Local derived samples and stimuli are already available.",
        local_patterns=[
            LocalPattern("data/derived/openneuro_full/ds006104/sub-P01/ses-01/*", "local/sub-P01_ses-01", required=True),
            LocalPattern("data/derived/openneuro_full/ds006104/sub-S01/ses-02/*", "local/sub-S01_ses-02", limit=12),
            LocalPattern("data/raw/openneuro/ds006104/stimuli/*.wav", "local/stimuli_sample", limit=8),
            LocalPattern("outputs/probe_artifacts/ds006104/*", "probe_artifacts"),
        ],
        remote_files=[
            openneuro_file("ds006104", "dataset_description.json", "remote/dataset_description.json"),
        ],
    ),
    DatasetSpec(
        slug="kul_aad_4004271",
        title="KUL AAD",
        category="aad_controlled",
        priority="P1",
        source_url="https://zenodo.org/records/4004271",
        sample_goal="One AAD subject/trial and competing speech audio.",
        access_note="Zenodo public. Full EEG/audio files are large; local cache currently has README/scripts.",
        local_patterns=[LocalPattern("outputs/probe_artifacts/zenodo-4004271/*", "probe_artifacts", required=True)],
        remote_files=[zenodo_metadata("4004271")],
    ),
    DatasetSpec(
        slug="dtu_aad_1199011",
        title="DTU EEG and audio dataset for AAD",
        category="aad_controlled",
        priority="P1",
        source_url="https://zenodo.org/records/1199011",
        sample_goal="One reverberant AAD subject/trial with speech audio.",
        access_note="Zenodo public. Local cache currently has preprocess script; full sample requires selecting files from record metadata.",
        local_patterns=[LocalPattern("outputs/probe_artifacts/zenodo-1199011/*", "probe_artifacts", required=True)],
        remote_files=[zenodo_metadata("1199011")],
    ),
    DatasetSpec(
        slug="eeg_aad_255ch_4518754",
        title="Ultra high-density 255-channel EEG-AAD",
        category="aad_controlled",
        priority="P1",
        source_url="https://zenodo.org/records/4518754",
        sample_goal="One high-density AAD subject/trial and audio.",
        access_note="Zenodo public. Data can be large; current local cache has misc/scripts zip samples.",
        local_patterns=[LocalPattern("outputs/probe_artifacts/zenodo-4518754/*", "probe_artifacts", required=True)],
        remote_files=[zenodo_metadata("4518754")],
    ),
    DatasetSpec(
        slug="openmiir",
        title="OpenMIIR",
        category="music_proxy",
        priority="P3",
        source_url="https://github.com/sstober/openmiir",
        sample_goal="One music perception/imagination subject/trial and beat annotation.",
        access_note="Public metadata available; full EEG/audio route follows OpenMIIR instructions.",
        local_patterns=[LocalPattern("outputs/probe_artifacts/openmiir/*", "probe_artifacts", required=True)],
    ),
    DatasetSpec(
        slug="musin_g_ds003774",
        title="MUSIN-G",
        category="music_proxy",
        priority="P3",
        source_url="https://openneuro.org/datasets/ds003774",
        sample_goal="One music listening EEG event/channel/stimulus sample.",
        access_note="Public OpenNeuro. Current local cache has metadata and small probe snippets.",
        local_patterns=[LocalPattern("outputs/probe_artifacts/ds003774/*", "probe_artifacts", required=True)],
        remote_files=[openneuro_file("ds003774", "dataset_description.json", "remote/dataset_description.json")],
    ),
    DatasetSpec(
        slug="mad_eeg_4537751",
        title="MAD-EEG",
        category="music_proxy",
        priority="P3",
        source_url="https://zenodo.org/records/4537751",
        sample_goal="One target-instrument EEG/audio sample.",
        access_note="Zenodo public. Current local cache has behavioral/raw YAML/sequences snippets.",
        local_patterns=[LocalPattern("outputs/probe_artifacts/zenodo-4537751/*", "probe_artifacts", required=True)],
        remote_files=[zenodo_metadata("4537751")],
    ),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for idx in range(2, 1000):
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not make unique destination for {path}")


def copy_local_files(spec: DatasetSpec, dataset_dir: Path) -> list[dict]:
    results: list[dict] = []
    for local in spec.local_patterns:
        matches = sorted(glob.glob(local.pattern))
        selected = matches[: local.limit]
        copied = []
        dest_dir = dataset_dir / local.dest
        ensure_dir(dest_dir)
        for match in selected:
            src = Path(match)
            if not src.is_file():
                continue
            dest = safe_name(dest_dir / src.name)
            shutil.copy2(src, dest)
            copied.append(str(dest.relative_to(dataset_dir)))
        results.append(
            {
                "pattern": local.pattern,
                "dest": local.dest,
                "required": local.required,
                "matches": len(matches),
                "copied": copied,
                "status": "copied" if copied else ("missing_required" if local.required else "missing_optional"),
            }
        )
    return results


def remote_size(url: str, timeout: int) -> int | None:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "eeg-voice-sample-downloader/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value else None
    except Exception:
        return None


def download_remote_file(remote: RemoteFile, dataset_dir: Path, allow_large: bool, default_max_mb: float, timeout: int) -> dict:
    dest = dataset_dir / remote.relpath
    ensure_dir(dest.parent)
    max_mb = remote.max_mb if remote.max_mb is not None else default_max_mb
    size = remote_size(remote.url, timeout)
    if size is not None and not allow_large and size > max_mb * 1024 * 1024:
        return {
            "url": remote.url,
            "relpath": remote.relpath,
            "required": remote.required,
            "status": "skipped_too_large",
            "content_length": size,
            "max_mb": max_mb,
        }
    try:
        request = urllib.request.Request(remote.url, headers={"User-Agent": "eeg-voice-sample-downloader/0.1"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
        if not allow_large and len(data) > max_mb * 1024 * 1024:
            return {
                "url": remote.url,
                "relpath": remote.relpath,
                "required": remote.required,
                "status": "skipped_too_large_after_read",
                "bytes_read": len(data),
                "max_mb": max_mb,
            }
        dest.write_bytes(data)
        return {
            "url": remote.url,
            "relpath": str(dest.relative_to(dataset_dir)),
            "required": remote.required,
            "status": "downloaded",
            "bytes": len(data),
        }
    except urllib.error.HTTPError as exc:
        return {
            "url": remote.url,
            "relpath": remote.relpath,
            "required": remote.required,
            "status": "http_error",
            "code": exc.code,
            "reason": str(exc.reason),
        }
    except Exception as exc:
        return {
            "url": remote.url,
            "relpath": remote.relpath,
            "required": remote.required,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def write_dataset_readme(spec: DatasetSpec, dataset_dir: Path, local_results: list[dict], remote_results: list[dict]) -> None:
    copied_count = sum(len(item.get("copied", [])) for item in local_results)
    downloaded_count = sum(1 for item in remote_results if item.get("status") == "downloaded")
    lines = [
        f"# {spec.title}",
        "",
        f"- slug: `{spec.slug}`",
        f"- category: `{spec.category}`",
        f"- priority: `{spec.priority}`",
        f"- source: {spec.source_url}",
        f"- sample goal: {spec.sample_goal}",
        f"- access note: {spec.access_note}",
        f"- local files copied: {copied_count}",
        f"- remote files downloaded: {downloaded_count}",
        "",
        "## Local Copy Results",
        "",
    ]
    if local_results:
        for item in local_results:
            lines.append(f"- `{item['pattern']}` -> `{item['dest']}`: {item['status']} ({item['matches']} matches)")
    else:
        lines.append("- No local file patterns configured.")
    lines.extend(["", "## Remote Download Results", ""])
    if remote_results:
        for item in remote_results:
            status = item.get("status")
            relpath = item.get("relpath")
            lines.append(f"- {status}: `{relpath}` from {item.get('url')}")
    else:
        lines.append("- No safe automatic remote downloads configured.")
    lines.extend(
        [
            "",
            "## Next Action",
            "",
            "Use this folder as the canonical ignored sample location for this dataset. If the status is manual or skipped, download one subject/trial into this folder after checking the source terms and file sizes.",
            "",
        ]
    )
    (dataset_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def prepare_dataset(spec: DatasetSpec, root: Path, allow_large: bool, default_max_mb: float, timeout: int) -> dict:
    dataset_dir = root / spec.category / spec.slug
    ensure_dir(dataset_dir)
    local_results = copy_local_files(spec, dataset_dir)
    remote_results = [
        download_remote_file(remote, dataset_dir, allow_large=allow_large, default_max_mb=default_max_mb, timeout=timeout)
        for remote in spec.remote_files
    ]
    has_sample = any(item.get("copied") for item in local_results) or any(item.get("status") == "downloaded" for item in remote_results)
    missing_required = [
        item
        for item in local_results
        if item.get("required") and not item.get("copied")
    ] + [
        item
        for item in remote_results
        if item.get("required") and item.get("status") != "downloaded"
    ]
    status = "ready_or_partial_sample" if has_sample else "manual_required"
    if missing_required:
        status = "missing_required_sample_files"
    record = {
        "slug": spec.slug,
        "title": spec.title,
        "category": spec.category,
        "priority": spec.priority,
        "source_url": spec.source_url,
        "sample_goal": spec.sample_goal,
        "access_note": spec.access_note,
        "dataset_dir": str(dataset_dir),
        "status": status,
        "local_results": local_results,
        "remote_results": remote_results,
    }
    (dataset_dir / "status.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    write_dataset_readme(spec, dataset_dir, local_results, remote_results)
    return record


def write_root_files(root: Path, records: list[dict]) -> None:
    ready = [r for r in records if r["status"] == "ready_or_partial_sample"]
    manual = [r for r in records if r["status"] != "ready_or_partial_sample"]
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "dataset_count": len(records),
        "ready_or_partial_count": len(ready),
        "manual_or_missing_count": len(manual),
        "records": records,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Voice EEG Dataset Samples",
        "",
        "This directory is ignored by git and stores one or two sample examples per dataset when available.",
        "",
        f"- dataset count: {len(records)}",
        f"- ready or partial sample folders: {len(ready)}",
        f"- manual or missing folders: {len(manual)}",
        "",
        "## Status Table",
        "",
        "| Dataset | Category | Priority | Status | Folder |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| `{record['slug']}` | `{record['category']}` | `{record['priority']}` | `{record['status']}` | `{record['dataset_dir']}` |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `ready_or_partial_sample` means the folder contains copied local examples and/or downloaded public metadata/small files.",
            "- `manual_required` means the source requires login, terms acceptance, exact file selection, or large downloads.",
            "- Run with `--allow-large` only after checking disk space and source terms.",
            "",
        ]
    )
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def selected_specs(names: Iterable[str] | None) -> list[DatasetSpec]:
    if not names:
        return DATASETS
    wanted = set(names)
    specs = [spec for spec in DATASETS if spec.slug in wanted]
    missing = sorted(wanted - {spec.slug for spec in specs})
    if missing:
        raise SystemExit(f"Unknown dataset slug(s): {', '.join(missing)}")
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", action="append", help="Dataset slug to prepare. May be repeated. Defaults to all.")
    parser.add_argument("--allow-large", action="store_true", help="Allow remote downloads larger than the configured max_mb.")
    parser.add_argument("--max-mb", type=float, default=25.0, help="Default max MB for remote files without an explicit limit.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    ensure_dir(args.root)
    records = [
        prepare_dataset(spec, root=args.root, allow_large=args.allow_large, default_max_mb=args.max_mb, timeout=args.timeout)
        for spec in selected_specs(args.dataset)
    ]
    write_root_files(args.root, records)
    print(json.dumps({"root": str(args.root), "datasets": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
