#!/usr/bin/env python3
"""Build an imagined/covert speech EEG dataset exploration notebook.

The notebook is intentionally lightweight: it consumes probe reports and small
metadata artifacts, then defines the unified trial-index schema to use after
selected full subjects are downloaded.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "explore_imagined_speech_datasets.ipynb"
STRATEGY_JSON = ROOT / "outputs" / "imagined_speech_dataset_strategy.json"
STRATEGY_CSV = ROOT / "outputs" / "imagined_speech_dataset_strategy.csv"
INDEX_TEMPLATE_CSV = ROOT / "outputs" / "imagined_speech_unified_index_template.csv"


STRATEGY_ROWS: list[dict[str, str]] = [
    {
        "rank": "1",
        "dataset": "feis_3554128",
        "title": "FEIS: Fourteen-channel EEG with Imagined Speech",
        "source_url": "https://zenodo.org/records/3554128",
        "role": "primary imagined-speech pilot",
        "language": "English, plus two Chinese subjects",
        "imagined_window": "thinking phase per epoch",
        "target_proxy": "prompt wav or same-label spoken wav",
        "overt_audio": "yes, per-subject recorded wavs",
        "eeg_quality": "14-channel Emotiv; low density but directly task-matched",
        "recommendation": "Start here for the first end-to-end imagined EEG to label/audio-embedding retrieval baseline.",
    },
    {
        "rank": "2",
        "dataset": "kara_one",
        "title": "KARA ONE imagined and articulated speech",
        "source_url": "https://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html",
        "role": "primary cross-check if archives are accessible",
        "language": "English",
        "imagined_window": "5 s imagined speech state",
        "target_proxy": "prompt label/audio; vocalized audio for same prompt",
        "overt_audio": "yes",
        "eeg_quality": "EEG plus face tracking and audio; smaller cohort",
        "recommendation": "Use after FEIS to test whether the same windowing and target proxy transfers.",
    },
    {
        "rank": "3",
        "dataset": "ugr_mindvoice",
        "title": "UGR-MINDVOICE",
        "source_url": "https://osf.io/6sh5d",
        "role": "modern overt/covert EEG-audio validation",
        "language": "Iberian Spanish",
        "imagined_window": "covert event windows from BIDS events",
        "target_proxy": "materials wav, event label, or same-session anonymized speech audio",
        "overt_audio": "yes",
        "eeg_quality": "BIDS EDF with audio folder; conceptually close but not English",
        "recommendation": "Use as the best concept match if language can be relaxed.",
    },
    {
        "rank": "4",
        "dataset": "ds003626",
        "title": "Thinking Out Loud / Inner Speech",
        "source_url": "https://openneuro.org/datasets/ds003626",
        "role": "inner-speech classification benchmark",
        "language": "Spanish",
        "imagined_window": "inner/pronounced/visualized trial epochs",
        "target_proxy": "word label only",
        "overt_audio": "no public per-trial audio target",
        "eeg_quality": "128-channel EEG; strong for classification, weak for waveform target",
        "recommendation": "Use for representation learning and sanity-check classification, not audio reconstruction.",
    },
    {
        "rank": "5",
        "dataset": "ds005170",
        "title": "Chisco Chinese imagined speech",
        "source_url": "https://openneuro.org/datasets/ds005170",
        "role": "imagined-speech supplementary benchmark",
        "language": "Chinese",
        "imagined_window": "imagine task runs",
        "target_proxy": "text stimulus label",
        "overt_audio": "not confirmed in public probe",
        "eeg_quality": "OpenNeuro raw EDF plus derivatives; large",
        "recommendation": "Keep as a secondary imagined-speech benchmark after FEIS/KARA/UGR.",
    },
    {
        "rank": "6",
        "dataset": "ds004408",
        "title": "EEG responses to continuous naturalistic speech",
        "source_url": "https://openneuro.org/datasets/ds004408",
        "role": "heard-speech pretraining only",
        "language": "English",
        "imagined_window": "none",
        "target_proxy": "wav + TextGrid word/phoneme timing",
        "overt_audio": "n/a",
        "eeg_quality": "128-channel heard speech with strong audio alignment",
        "recommendation": "Use for acoustic/phoneme pretraining, not as imagined-speech supervision.",
    },
    {
        "rank": "7",
        "dataset": "ds004940",
        "title": "Auditory N400 active/passive sentence EEG",
        "source_url": "https://openneuro.org/datasets/ds004940",
        "role": "heard-speech response/pretraining only",
        "language": "English",
        "imagined_window": "none",
        "target_proxy": "audio stimuli, word/sentence metadata, and active-response events",
        "overt_audio": "n/a",
        "eeg_quality": "high-density BDF heard-speech EEG; useful for response-event plumbing",
        "recommendation": "Use to test event/response handling and heard-speech pretraining, not imagined supervision.",
    },
]


INDEX_TEMPLATE_ROWS: list[dict[str, str]] = [
    {
        "dataset": "feis_3554128",
        "subject": "01",
        "trial": "pending_full_extraction",
        "mode": "imagined",
        "label": "f",
        "eeg_start": "",
        "eeg_end": "",
        "audio_path_or_label": "wavs/01/wavs/f.wav",
        "target_kind": "prompt_audio",
        "source_path": "experiments/01/thinking.zip",
        "notes": "Fill eeg_start/eeg_end after extracting FEIS thinking CSV trial timing.",
    },
    {
        "dataset": "feis_3554128",
        "subject": "01",
        "trial": "pending_full_extraction",
        "mode": "overt",
        "label": "f",
        "eeg_start": "",
        "eeg_end": "",
        "audio_path_or_label": "wavs/01/wavs/f.wav",
        "target_kind": "same_label_spoken_or_prompt_audio",
        "source_path": "experiments/01/speaking.zip",
        "notes": "Use overt EEG/audio for acoustic target pretraining or same-label pairing.",
    },
    {
        "dataset": "kara_one",
        "subject": "P02",
        "trial": "pending_archive_extraction",
        "mode": "imagined",
        "label": "iy|uw|piy|tiy|diy|m|n|pat|pot|knew|gnaw",
        "eeg_start": "",
        "eeg_end": "",
        "audio_path_or_label": "prompt label or vocalized audio for the same prompt",
        "target_kind": "prompt_label_or_same_label_overt_audio",
        "source_path": "P02.tar.bz2",
        "notes": "KARA archives need full participant extraction before trial timing can be indexed.",
    },
    {
        "dataset": "ugr_mindvoice",
        "subject": "sub-05",
        "trial": "event_row",
        "mode": "covert",
        "label": "events.trial_type",
        "eeg_start": "events.onset",
        "eeg_end": "events.onset + events.duration",
        "audio_path_or_label": "Materials/*.wav or same-session anonymized.wav",
        "target_kind": "materials_audio_or_event_label",
        "source_path": "Data/sub-05/ses-*/eeg/*_events.tsv",
        "notes": "Use BIDS events once selected subject files are downloaded.",
    },
    {
        "dataset": "ds003626",
        "subject": "sub-01",
        "trial": "epoch_row",
        "mode": "inner",
        "label": "direction word label",
        "eeg_start": "epoch start",
        "eeg_end": "epoch end",
        "audio_path_or_label": "word label only",
        "target_kind": "label",
        "source_path": "derivatives/sub-01/ses-01/*_events.dat",
        "notes": "Useful for classification/representation checks, not waveform supervision.",
    },
]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip("\n"),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip("\n"),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_notebook() -> dict:
    cells: list[dict] = []
    cells.append(
        md(
            """
            # Imagined Speech EEG Dataset Triage

            Goal: choose datasets for a first imagined/covert speech EEG reconstruction
            pipeline. The practical target is **imagined EEG -> prompt label/audio
            embedding retrieval**, not direct waveform generation on day one.

            This notebook consumes the lightweight remote probes generated by:

            ```bash
            python3 scripts/probe_eeg_audio_datasets.py \\
              --only feis_3554128 kara_one ugr_mindvoice ds003626 ds004306 ds005170 ds007591 ds004408 \\
              --artifact-dir outputs/imagined_speech_probe_artifacts \\
              --json-out outputs/imagined_speech_dataset_probe_results.json \\
              --md-out outputs/imagined_speech_dataset_probe_results.md \\
              --detail-md-out outputs/imagined_speech_dataset_probe_details.md
            ```
            """
        )
    )
    cells.append(
        code(
            """
            from __future__ import annotations

            import json
            from pathlib import Path

            import pandas as pd
            from IPython.display import Markdown, display

            ROOT = Path.cwd()
            PROBE_JSON = ROOT / "outputs" / "imagined_speech_dataset_probe_results.json"
            STRATEGY_JSON = ROOT / "outputs" / "imagined_speech_dataset_strategy.json"
            INDEX_TEMPLATE_CSV = ROOT / "outputs" / "imagined_speech_unified_index_template.csv"
            ARTIFACT_DIR = ROOT / "outputs" / "imagined_speech_probe_artifacts"

            assert STRATEGY_JSON.exists(), STRATEGY_JSON
            strategy = pd.DataFrame(json.loads(STRATEGY_JSON.read_text()))
            strategy
            """
        )
    )
    cells.append(
        md(
            """
            ## Probe Status

            The probe only verifies public availability and small metadata/audio/event
            snippets. It intentionally does not pull full EEG archives.
            """
        )
    )
    cells.append(
        code(
            """
            probe_results = json.loads(PROBE_JSON.read_text()) if PROBE_JSON.exists() else []

            def probe_status(row):
                if "error" in row:
                    return "ERROR", row["error"]
                errors = [t for t in row.get("targets", []) if "error" in t]
                status = "PARTIAL" if errors else "OK"
                evidence = []
                if row.get("file_count") is not None:
                    evidence.append(f"zenodo_files={row['file_count']}")
                for target in row.get("targets", [])[:6]:
                    if "error" in target:
                        evidence.append(f"{target.get('label')}: {target.get('error')}")
                    else:
                        parsed = target.get("parsed", {})
                        if isinstance(parsed, list):
                            evidence.append(f"{target['label']}: {len(parsed)} items")
                        elif isinstance(parsed, dict):
                            if target.get("kind") in {"tsv", "csv"}:
                                evidence.append(f"{target['label']}: {len(parsed.get('columns', []))} cols")
                            elif target.get("kind") == "wav":
                                evidence.append(f"{target['label']}: {parsed.get('sample_rate')} Hz")
                            elif target.get("kind") == "json":
                                evidence.append(f"{target['label']}: json")
                            else:
                                evidence.append(f"{target['label']}: {target.get('bytes_read')} bytes")
                return status, "; ".join(evidence)

            rows = []
            for row in probe_results:
                status, evidence = probe_status(row)
                rows.append({
                    "dataset": row.get("dataset_id"),
                    "source": row.get("source"),
                    "priority": row.get("priority"),
                    "status": status,
                    "evidence": evidence,
                })
            probe_table = pd.DataFrame(rows)
            probe_table
            """
        )
    )
    cells.append(
        md(
            """
            ## Decision Matrix

            Primary training data should have an imagined/covert condition and an
            explicit time window. Heard speech datasets are still useful, but only
            for acoustic or phoneme pretraining.
            """
        )
    )
    cells.append(
        code(
            """
            display(strategy[[
                "rank",
                "dataset",
                "role",
                "language",
                "imagined_window",
                "target_proxy",
                "overt_audio",
                "recommendation",
            ]])
            """
        )
    )
    cells.append(
        md(
            """
            ## Artifact Inventory

            This is a quick way to see what the probe saved locally. Binary and
            audio-like probe artifacts are ignored by git.
            """
        )
    )
    cells.append(
        code(
            """
            artifact_rows = []
            if ARTIFACT_DIR.exists():
                for path in sorted(ARTIFACT_DIR.rglob("*")):
                    if path.is_file():
                        artifact_rows.append({
                            "dataset": path.parent.name,
                            "file": str(path.relative_to(ROOT)),
                            "bytes": path.stat().st_size,
                        })
            pd.DataFrame(artifact_rows)
            """
        )
    )
    cells.append(
        md(
            """
            ## Unified Trial Index Schema

            Every downloaded dataset should be converted into this narrow table.
            The table is allowed to start with labels/proxy targets; direct waveform
            targets should only be used when the dataset provides overt/prompt audio.
            """
        )
    )
    cells.append(
        code(
            """
            index_template = pd.read_csv(INDEX_TEMPLATE_CSV)
            index_template
            """
        )
    )
    cells.append(
        code(
            """
            REQUIRED_COLUMNS = [
                "dataset",
                "subject",
                "trial",
                "mode",
                "label",
                "eeg_start",
                "eeg_end",
                "audio_path_or_label",
                "target_kind",
                "source_path",
                "notes",
            ]
            missing = [col for col in REQUIRED_COLUMNS if col not in index_template.columns]
            assert not missing, missing
            print("Unified index columns are ready:", REQUIRED_COLUMNS)
            """
        )
    )
    cells.append(
        md(
            """
            ## BIDS Event Helper

            Use this for UGR-MINDVOICE or any future BIDS-style imagined/covert
            speech dataset. It converts an events file into the same index schema.
            """
        )
    )
    cells.append(
        code(
            """
            def bids_events_to_unified_index(
                events_path: Path,
                dataset: str,
                subject: str,
                mode_default: str = "covert",
                label_col: str = "trial_type",
                mode_col: str | None = None,
            ) -> pd.DataFrame:
                events = pd.read_csv(events_path, sep="\\t", encoding="utf-8-sig")
                rows = []
                for trial, event in events.reset_index(drop=True).iterrows():
                    onset = event.get("onset")
                    duration = event.get("duration", 0)
                    label = event.get(label_col, "")
                    mode = event.get(mode_col, mode_default) if mode_col else mode_default
                    eeg_end = ""
                    try:
                        eeg_end = float(onset) + float(duration)
                    except Exception:
                        pass
                    rows.append({
                        "dataset": dataset,
                        "subject": subject,
                        "trial": trial,
                        "mode": mode,
                        "label": label,
                        "eeg_start": onset,
                        "eeg_end": eeg_end,
                        "audio_path_or_label": label,
                        "target_kind": "event_label",
                        "source_path": str(events_path),
                        "notes": "Generated from BIDS events; confirm task-specific label columns before training.",
                    })
                return pd.DataFrame(rows)

            possible_ugr_events = sorted(ARTIFACT_DIR.glob("ugr_mindvoice/*events*.tsv"))
            if possible_ugr_events:
                display(bids_events_to_unified_index(possible_ugr_events[0], "ugr_mindvoice", "sub-05").head())
            else:
                print("No UGR events artifact found yet. Re-run the probe after network access is available.")
            """
        )
    )
    cells.append(
        md(
            """
            ## Recommended Next Execution

            1. Extract one FEIS subject (`experiments/01/thinking.zip`,
               `stimuli.zip`, `speaking.zip`, and `wavs/01/wavs/*.wav`).
            2. Fill a real per-trial index for FEIS using the schema above.
            3. Train a retrieval baseline: imagined EEG epoch -> prompt audio
               embedding or label.
            4. Use DS004408 only as heard-speech pretraining, then test transfer to
               imagined FEIS/KARA/UGR windows.
            """
        )
    )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    STRATEGY_JSON.parent.mkdir(parents=True, exist_ok=True)
    STRATEGY_JSON.write_text(json.dumps(STRATEGY_ROWS, indent=2), encoding="utf-8")
    write_csv(STRATEGY_CSV, STRATEGY_ROWS)
    write_csv(INDEX_TEMPLATE_CSV, INDEX_TEMPLATE_ROWS)
    NOTEBOOK_PATH.write_text(
        json.dumps(build_notebook(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(NOTEBOOK_PATH)
    print(STRATEGY_JSON)
    print(STRATEGY_CSV)
    print(INDEX_TEMPLATE_CSV)


if __name__ == "__main__":
    main()
