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
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import quote


DEFAULT_ROOT = Path("data/voice_eeg_dataset_samples")
DEFAULT_TIMEOUT = 45


@dataclass
class RemoteFile:
    url: str
    relpath: str
    max_mb: float = 25.0
    required: bool = False
    range_bytes: int | None = None


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


def openneuro_file(
    dataset: str,
    path: str,
    relpath: str,
    max_mb: float = 25.0,
    range_bytes: int | None = None,
) -> RemoteFile:
    encoded_path = quote(path, safe="/%:_-.,~()")
    return RemoteFile(
        url=f"https://s3.amazonaws.com/openneuro.org/{dataset}/{encoded_path}",
        relpath=relpath,
        max_mb=max_mb,
        range_bytes=range_bytes,
    )


def zenodo_metadata(record_id: str, relpath: str = "remote/zenodo_record.json") -> RemoteFile:
    return RemoteFile(
        url=f"https://zenodo.org/api/records/{record_id}",
        relpath=relpath,
        max_mb=10.0,
    )


def zenodo_file(
    record_id: str,
    key: str,
    relpath: str,
    max_mb: float = 25.0,
    range_bytes: int | None = None,
) -> RemoteFile:
    encoded_key = quote(key, safe="")
    return RemoteFile(
        url=f"https://zenodo.org/api/records/{record_id}/files/{encoded_key}/content",
        relpath=relpath,
        max_mb=max_mb,
        range_bytes=range_bytes,
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
            openneuro_file("ds004408", "README", "remote/README"),
            openneuro_file("ds004408", "participants.tsv", "remote/participants.tsv"),
            openneuro_file("ds004408", "stimuli/audio01.TextGrid", "remote/stimuli/audio01.TextGrid"),
            openneuro_file("ds004408", "stimuli/audio01.wav", "remote/stimuli/audio01.wav.header.bin", range_bytes=65536),
            openneuro_file("ds004408", "sub-001/eeg/sub-001_task-listening_run-01_channels.tsv", "remote/eeg/sub-001_run01_channels.tsv"),
            openneuro_file("ds004408", "sub-001/eeg/sub-001_task-listening_run-01_eeg.json", "remote/eeg/sub-001_run01_eeg.json"),
            openneuro_file("ds004408", "sub-001/eeg/sub-001_task-listening_run-01_eeg.vhdr", "remote/eeg/sub-001_run01_eeg.vhdr"),
            openneuro_file("ds004408", "sub-001/eeg/sub-001_task-listening_run-01_eeg.vmrk", "remote/eeg/sub-001_run01_eeg.vmrk"),
            openneuro_file("ds004408", "sub-001/eeg/sub-001_task-listening_run-01_eeg.eeg", "remote/eeg/sub-001_run01_eeg.head.bin", range_bytes=65536),
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
        remote_files=[
            zenodo_metadata("7086168"),
            zenodo_file("7086168", "WeissbartSurprisal.zip", "remote/archive_headers/WeissbartSurprisal.zip.head.bin", range_bytes=65536),
        ],
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
            openneuro_file("ds006434", "sub-dichotic02/eeg/sub-dichotic02_task-exp2DichoticCortex_events.tsv", "remote/eeg/events.tsv"),
            openneuro_file("ds006434", "sub-dichotic02/eeg/sub-dichotic02_task-exp2DichoticCortex_events.json", "remote/eeg/events.json"),
            openneuro_file("ds006434", "sub-dichotic02/eeg/sub-dichotic02_task-exp2DichoticCortex_channels.tsv", "remote/eeg/channels.tsv"),
            openneuro_file("ds006434", "sub-dichotic02/eeg/sub-dichotic02_task-exp2DichoticCortex_eeg.json", "remote/eeg/eeg.json"),
            openneuro_file("ds006434", "sub-dichotic02/eeg/sub-dichotic02_task-exp2DichoticCortex_eeg.vhdr", "remote/eeg/eeg.vhdr"),
            openneuro_file("ds006434", "sub-dichotic02/eeg/sub-dichotic02_task-exp2DichoticCortex_eeg.vmrk", "remote/eeg/eeg.vmrk"),
            openneuro_file("ds006434", "sub-dichotic02/eeg/sub-dichotic02_task-exp2DichoticCortex_eeg.eeg", "remote/eeg/eeg.head.bin", range_bytes=65536),
            openneuro_file("ds006434", "stimuli/exp2Dichotic/wrinkle_alchemyst000.wav", "remote/stimuli/wrinkle_alchemyst000.wav.header.bin", range_bytes=65536),
        ],
    ),
    DatasetSpec(
        slug="ds007630_eeg_speech_brain_decoding",
        title="EEG-Speech Brain Decoding Dataset",
        category="english_controlled",
        priority="P0/P2",
        source_url="https://openneuro.org/datasets/ds007630",
        sample_goal="One speechopen run with events/channels/eeg sidecar plus vocal wav header and EDF byte-range probe.",
        access_note="Public OpenNeuro/EEGDash but about 955 GB. Direct S3 object GET returned 403 in the sample probe; use EEGDash/OpenNeuro client for full pulls.",
        remote_files=[
            RemoteFile("https://eegdash.org/api/dataset/eegdash.dataset.DS007630.html", "remote/eegdash_record.html", max_mb=5.0),
            openneuro_file("ds007630", "dataset_description.json", "remote/dataset_description.json"),
            openneuro_file("ds007630", "participants.tsv", "remote/participants.tsv"),
            openneuro_file("ds007630", "sub-01/ses-20230829/eeg/sub-01_ses-20230829_task-speechopen_acq-pangolin_run-01_events.tsv", "remote/eeg/events.tsv"),
            openneuro_file("ds007630", "sub-01/ses-20230829/eeg/sub-01_ses-20230829_task-speechopen_acq-pangolin_run-01_channels.tsv", "remote/eeg/channels.tsv"),
            openneuro_file("ds007630", "sub-01/ses-20230829/eeg/sub-01_ses-20230829_task-speechopen_acq-pangolin_run-01_eeg.json", "remote/eeg/eeg.json"),
            openneuro_file("ds007630", "sub-01/ses-20230829/eeg/sub-01_ses-20230829_task-speechopen_acq-pangolin_run-01_eeg.edf", "remote/eeg/eeg.edf.head.bin", range_bytes=65536),
            openneuro_file("ds007630", "sub-01/ses-20230829/beh/sub-01_ses-20230829_task-speechopen_acq-pangolin_run-01_recording-vocal_beh.json", "remote/audio/vocal_beh.json"),
            openneuro_file("ds007630", "sub-01/ses-20230829/beh/sub-01_ses-20230829_task-speechopen_acq-pangolin_run-01_recording-vocal_beh.wav", "remote/audio/vocal_beh.wav.header.bin", range_bytes=65536),
        ],
    ),
    DatasetSpec(
        slug="ds007602_eeg_speech_overt",
        title="EEG-Speech Brain Decoding Dataset overt speech subset",
        category="english_proxy",
        priority="P2",
        source_url="https://openneuro.org/datasets/ds007602",
        sample_goal="One overt speech production run with BIDS metadata and EDF byte-range probe; audio path is not exposed in the probed beh prefix.",
        access_note="Public OpenNeuro/EEGDash, about 49.6 GB. No beh/audio object was exposed under the mirrored sub-01 session prefix during the sample probe; use EEGDash/OpenNeuro client if direct S3 GET is denied.",
        remote_files=[
            RemoteFile("https://eegdash.org/api/dataset/eegdash.dataset.DS007602.html", "remote/eegdash_record.html", max_mb=5.0),
            openneuro_file("ds007602", "dataset_description.json", "remote/dataset_description.json"),
            openneuro_file("ds007602", "README", "remote/README"),
            openneuro_file("ds007602", "participants.tsv", "remote/participants.tsv"),
            openneuro_file("ds007602", "sub-01/ses-20230829/eeg/sub-01_ses-20230829_task-speechopen_acq-pangolin_run-01_events.tsv", "remote/eeg/events.tsv"),
            openneuro_file("ds007602", "sub-01/ses-20230829/eeg/sub-01_ses-20230829_task-speechopen_acq-pangolin_run-01_channels.tsv", "remote/eeg/channels.tsv"),
            openneuro_file("ds007602", "sub-01/ses-20230829/eeg/sub-01_ses-20230829_task-speechopen_acq-pangolin_run-01_eeg.json", "remote/eeg/eeg.json"),
            openneuro_file("ds007602", "sub-01/ses-20230829/eeg/sub-01_ses-20230829_task-speechopen_acq-pangolin_run-01_eeg.edf", "remote/eeg/eeg.edf.head.bin", range_bytes=65536),
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
        remote_files=[
            zenodo_metadata("7086209"),
            zenodo_file("7086209", "EtardBrainstemAndComprehension.zip", "remote/archive_headers/EtardBrainstemAndComprehension.zip.head.bin", range_bytes=65536),
        ],
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
            openneuro_file("ds007591", "sub-1/ses-20230511/eeg/sub-1_ses-20230511_task-minimallyovert_acq-calibration_run-01_events.tsv", "remote/eeg/events.tsv"),
            openneuro_file("ds007591", "sub-1/ses-20230511/eeg/sub-1_ses-20230511_task-minimallyovert_acq-calibration_run-01_channels.tsv", "remote/eeg/channels.tsv"),
            openneuro_file("ds007591", "sub-1/ses-20230511/eeg/sub-1_ses-20230511_task-minimallyovert_acq-calibration_run-01_eeg.json", "remote/eeg/eeg.json"),
            openneuro_file("ds007591", "sub-1/ses-20230511/eeg/sub-1_ses-20230511_task-minimallyovert_acq-calibration_run-01_eeg.edf", "remote/eeg/eeg.edf.head.bin", range_bytes=65536),
        ],
    ),
    DatasetSpec(
        slug="kara_one",
        title="Kara One imagined and articulated speech",
        category="english_proxy",
        priority="P2",
        source_url="https://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html",
        sample_goal="Dataset page and one participant archive after manual size/terms check.",
        access_note="Public academic-use dataset, 14 participant archives totaling about 24 GB. Automatic sample stores page metadata only.",
        remote_files=[
            RemoteFile("https://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html", "remote/kara_one.html", max_mb=5.0),
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
            openneuro_file("ds005345", "README", "remote/README"),
            openneuro_file("ds005345", "participants.tsv", "remote/participants.tsv"),
            openneuro_file("ds005345", "annotation/single_female_word_information.csv", "remote/annotation/single_female_word_information.csv"),
            openneuro_file("ds005345", "stimuli/single_female.wav", "remote/stimuli/single_female.wav.header.bin", range_bytes=65536),
            openneuro_file("ds005345", "sub-01/eeg/sub-01_task-multitalker_eeg.json", "remote/eeg/raw_eeg.json"),
            openneuro_file("ds005345", "sub-01/eeg/sub-01_task-multitalker_eeg.vhdr", "remote/eeg/raw_eeg.vhdr"),
            openneuro_file("ds005345", "sub-01/eeg/sub-01_task-multitalker_eeg.eeg", "remote/eeg/raw_eeg.head.bin", range_bytes=65536),
            openneuro_file("ds005345", "derivatives/sub-01/eeg/sub-01_task-multitalker_run-1_eeg_preprocessed.fif", "remote/eeg/preprocessed_run1.fif.head.bin", range_bytes=65536),
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
        remote_files=[
            zenodo_metadata("7078451"),
            zenodo_file("7078451", "readme.txt", "remote/readme.txt", max_mb=1.0),
            zenodo_file("7078451", "preprocess.zip", "remote/preprocess.zip", max_mb=1.0),
            zenodo_file("7078451", "S1.zip", "remote/archive_headers/S1.zip.head.bin", range_bytes=65536),
        ],
    ),
    DatasetSpec(
        slug="nju_aad_7253438",
        title="NJU Mandarin AAD",
        category="mandarin",
        priority="P0/P1",
        source_url="https://zenodo.org/records/7253438",
        sample_goal="One Mandarin competing-speech subject/trial with audio and label.",
        access_note="Zenodo metadata can be downloaded; full file selection needs inspection.",
        remote_files=[
            zenodo_metadata("7253438"),
            zenodo_file("7253438", "script.zip", "remote/script.zip", max_mb=1.0),
            zenodo_file("7253438", "NJUNCA_preprocessed_arte_removed.zip", "remote/archive_headers/NJUNCA_preprocessed_arte_removed.zip.head.bin", range_bytes=65536),
        ],
    ),
    DatasetSpec(
        slug="ds006465_3m_cpseed",
        title="3M-CPSEED Mandarin pinyin speech",
        category="mandarin_proxy",
        priority="P2",
        source_url="https://openneuro.org/datasets/ds006465",
        sample_goal="One overt/mouthed/imagined pinyin subject example.",
        access_note="Public OpenNeuro. It exposes EEG and preprocessed MAT files; no speech audio stimulus files were exposed in the probed public listing.",
        remote_files=[
            openneuro_file("ds006465", "dataset_description.json", "remote/dataset_description.json"),
            openneuro_file("ds006465", "README.md", "remote/README.md"),
            openneuro_file("ds006465", "sub-01/ses-1/eeg/sub-01_ses-1_task-imaginedspeech_events.tsv", "remote/eeg/events.tsv"),
            openneuro_file("ds006465", "sub-01/ses-1/eeg/sub-01_ses-1_task-imaginedspeech_channels.tsv", "remote/eeg/channels.tsv"),
            openneuro_file("ds006465", "sub-01/ses-1/eeg/sub-01_ses-1_task-imaginedspeech_eeg.json", "remote/eeg/eeg.json"),
            openneuro_file("ds006465", "sub-01/ses-1/eeg/sub-01_ses-1_task-imaginedspeech_eeg.edf", "remote/eeg/eeg.edf.head.bin", range_bytes=65536),
            openneuro_file("ds006465", "derivatives/preproc/sub-01/ses-1/sub-01_ses-1_speak.mat", "remote/eeg/sub-01_ses-1_speak.mat.head.bin", range_bytes=65536),
            openneuro_file("ds006465", "derivatives/preproc/sub-01/ses-1/sub-01_ses-1_imagine.mat", "remote/eeg/sub-01_ses-1_imagine.mat.head.bin", range_bytes=65536),
        ],
    ),
    DatasetSpec(
        slug="ds005170_chisco",
        title="Chisco Chinese imagined speech",
        category="mandarin_proxy",
        priority="P2",
        source_url="https://openneuro.org/datasets/ds005170",
        sample_goal="Dataset metadata, participants, README, text stimulus split, and one raw/preprocessed EEG byte-range probe.",
        access_note="Public OpenNeuro/EEGDash, about 90.7 GB. Use metadata first, then download selected subject/session shards.",
        remote_files=[
            openneuro_file("ds005170", "dataset_description.json", "remote/dataset_description.json"),
            openneuro_file("ds005170", "README", "remote/README"),
            openneuro_file("ds005170", "textdataset/split_data_1.xlsx", "remote/textdataset/split_data_1.xlsx"),
            openneuro_file("ds005170", "sub-01/ses-01/eeg/sub-01_ses-01_task-imagine_run-01_eeg.edf", "remote/eeg/raw_run01.edf.head.bin", range_bytes=65536),
            openneuro_file("ds005170", "derivatives/preprocessed_fif/sub-01/eeg/sub-01_task-imagine_run-01_eeg.fif", "remote/eeg/preprocessed_run01.fif.head.bin", range_bytes=65536),
        ],
    ),
    DatasetSpec(
        slug="cire_2025",
        title="CIRE Chinese prosodic emotion and speech intention EEG",
        category="mandarin_proxy",
        priority="P2",
        source_url="https://www.nature.com/articles/s41597-025-05957-y",
        sample_goal="Scientific Data page and ScienceDB metadata; then participants/sentences/events/audio files after repository probe.",
        access_note="Open Scientific Data descriptor. Full dataset is hosted via ScienceDB; automatic sample stores publication metadata first.",
        remote_files=[
            RemoteFile("https://www.nature.com/articles/s41597-025-05957-y", "remote/scientific_data_page.html", max_mb=5.0),
        ],
    ),
    DatasetSpec(
        slug="aasd_17413336",
        title="AASD spontaneous auditory attention switch decoding",
        category="mandarin",
        priority="P0/P1",
        source_url="https://zenodo.org/records/17413336",
        sample_goal="One Mandarin spontaneous attention-switch EEG subject with multi-speaker stimuli metadata.",
        access_note="Zenodo public. Full EEG and audio zips are large; automatic sample stores record metadata only.",
        remote_files=[
            zenodo_metadata("17413336"),
            zenodo_file("17413336", "Original EEG.zip", "remote/archive_headers/Original_EEG.zip.head.bin", range_bytes=65536),
            zenodo_file("17413336", "Stimuli Audio.zip", "remote/archive_headers/Stimuli_Audio.zip.head.bin", range_bytes=65536),
        ],
    ),
    DatasetSpec(
        slug="ms_aasd_17149387",
        title="MS-AASD mixed-speech attention switch decoding",
        category="mandarin",
        priority="P0/P1",
        source_url="https://zenodo.org/records/17149387",
        sample_goal="One Mandarin mixed-speech self-initiated attention-switch subject with metadata.",
        access_note="Zenodo public. Full EEG and audio bundles are large; automatic sample stores record metadata only.",
        remote_files=[
            zenodo_metadata("17149387"),
            zenodo_file("17149387", "StimList.xlsx", "remote/StimList.xlsx", max_mb=1.0),
            zenodo_file("17149387", ".cnt.zip", "remote/archive_headers/cnt.zip.head.bin", range_bytes=65536),
            zenodo_file("17149387", "Female_wav.zip", "remote/archive_headers/Female_wav.zip.head.bin", range_bytes=65536),
            zenodo_file("17149387", "Mix_wav_Nospace.zip", "remote/archive_headers/Mix_wav_Nospace.zip.head.bin", range_bytes=65536),
        ],
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
            openneuro_file("ds004718", "sub-HK001/eeg/sub-HK001_task-lppHK_eeg.json", "remote/eeg/eeg.json"),
            openneuro_file("ds004718", "sub-HK001/eeg/sub-HK001_task-lppHK_eeg.set", "remote/eeg/eeg.set.head.bin", range_bytes=65536),
            openneuro_file("ds004718", "sourcedata/stimuli/audio_files_segmented_by_sentence/Part 1/1.003.wav", "remote/stimuli/part1_1.003.wav", max_mb=5.0),
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
        remote_files=[
            zenodo_metadata("7750292"),
            zenodo_file("7750292", "sub01 Mon.cnt", "remote/eeg/sub01_Mon.cnt.head.bin", range_bytes=65536),
        ],
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
            openneuro_file("ds006104", "README", "remote/README"),
            openneuro_file("ds006104", "sub-P01/ses-01/eeg/sub-P01_ses-01_task-phonemes_events.tsv", "remote/eeg/events.tsv"),
            openneuro_file("ds006104", "sub-P01/ses-01/eeg/sub-P01_ses-01_task-phonemes_channels.tsv", "remote/eeg/channels.tsv"),
            openneuro_file("ds006104", "sub-P01/ses-01/eeg/sub-P01_ses-01_task-phonemes_eeg.json", "remote/eeg/eeg.json"),
            openneuro_file("ds006104", "sub-P01/ses-01/eeg/sub-P01_ses-01_task-phonemes_eeg.edf", "remote/eeg/eeg.edf.head.bin", range_bytes=65536),
            openneuro_file("ds006104", "derivatives/eeglab/sub-P01/ses-01/P01.set", "remote/eeg/P01.set.head.bin", range_bytes=65536),
        ],
    ),
    DatasetSpec(
        slug="ds003626_inner_speech",
        title="Inner Speech",
        category="controlled_speech",
        priority="P2",
        source_url="https://openneuro.org/datasets/ds003626",
        sample_goal="Dataset metadata, README, one session events.dat, and epoched FIF byte-range probe.",
        access_note="Public OpenNeuro/EEGDash, about 18.3 GB. Full FIF derivatives should be pulled by subject/session.",
        remote_files=[
            openneuro_file("ds003626", "dataset_description.json", "remote/dataset_description.json"),
            openneuro_file("ds003626", "README", "remote/README"),
            openneuro_file("ds003626", "derivatives/sub-01/ses-01/sub-01_ses-01_events.dat", "remote/eeg/sub-01_ses-01_events.dat"),
            openneuro_file("ds003626", "sub-01/ses-01/eeg/sub-01_ses-01_task-innerspeech_eeg.bdf", "remote/eeg/raw_bdf.head.bin", range_bytes=65536),
            openneuro_file("ds003626", "derivatives/sub-01/ses-01/sub-01_ses-01_eeg-epo.fif", "remote/eeg/eeg-epo.fif.head.bin", range_bytes=65536),
        ],
    ),
    DatasetSpec(
        slug="feis_3554128",
        title="Fourteen-channel EEG with Imagined Speech",
        category="controlled_speech",
        priority="P2",
        source_url="https://zenodo.org/records/3554128",
        sample_goal="Zenodo metadata and file list for heard/imagined/spoken English phonemes and Chinese syllables.",
        access_note="Zenodo public, 1.6 GB archive. Automatic sample stores record metadata; archive download is safe after terms/space check.",
        remote_files=[
            zenodo_metadata("3554128"),
            zenodo_file("3554128", "scottwellington/FEIS-v1.1.zip", "remote/archive_headers/FEIS-v1.1.zip.head.bin", range_bytes=65536),
        ],
    ),
    DatasetSpec(
        slug="ugr_mindvoice",
        title="UGR-MINDVOICE",
        category="controlled_speech",
        priority="P2",
        source_url="https://osf.io/6sh5d",
        sample_goal="OSF root listing plus GitHub README/config for overt/covert Spanish EEG-audio metadata.",
        access_note="OSF public dataset and GitHub code. Download data folders after probing per-subject file sizes and audio anonymization details.",
        remote_files=[
            RemoteFile("https://api.osf.io/v2/nodes/6sh5d/files/osfstorage/", "remote/osf_root_listing.json", max_mb=5.0),
            RemoteFile("https://raw.githubusercontent.com/owaismujtaba/mind-voice/main/Readme.md", "remote/Readme.md", max_mb=5.0),
            RemoteFile("https://raw.githubusercontent.com/owaismujtaba/mind-voice/main/config.yaml", "remote/config.yaml", max_mb=1.0),
        ],
    ),
    DatasetSpec(
        slug="ds004306_semantic_imagination",
        title="EEG Semantic Imagination and Perception Dataset",
        category="controlled_speech",
        priority="P2/P3",
        source_url="https://openneuro.org/datasets/ds004306",
        sample_goal="Dataset metadata, participants, README, one auditory stimulus, and one preprocessed FIF byte-range probe.",
        access_note="Public OpenNeuro/EEGDash. Small metadata and stimuli are safe; full preprocessed files are large.",
        remote_files=[
            openneuro_file("ds004306", "dataset_description.json", "remote/dataset_description.json"),
            openneuro_file("ds004306", "participants.tsv", "remote/participants.tsv"),
            openneuro_file("ds004306", "README", "remote/README"),
            openneuro_file("ds004306", "stimuli/audio/flower/1.ogg", "remote/stimuli/audio/flower_1.ogg", max_mb=5.0),
            openneuro_file("ds004306", "derivatives/preprocessed/sub-016/ses-01/eeg/sub16_sess1_50_ica_eeg-1.fif", "remote/eeg/sub16_sess1_50_ica_eeg-1.fif.head.bin", range_bytes=65536),
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
        remote_files=[
            zenodo_metadata("4004271"),
            zenodo_file("4004271", "README.txt.txt", "remote/README.txt", max_mb=1.0),
            zenodo_file("4004271", "S2.mat", "remote/eeg/S2.mat.head.bin", range_bytes=65536),
            zenodo_file("4004271", "stimuli.zip", "remote/archive_headers/stimuli.zip.head.bin", range_bytes=65536),
        ],
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
        remote_files=[
            zenodo_metadata("1199011"),
            zenodo_file("1199011", "preproc_data.m", "remote/preproc_data.m", max_mb=1.0),
            zenodo_file("1199011", "AUDIO.zip", "remote/archive_headers/AUDIO.zip.head.bin", range_bytes=65536),
            zenodo_file("1199011", "EEG.zip", "remote/eeg/EEG.zip.head.bin", range_bytes=65536),
        ],
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
        remote_files=[
            zenodo_metadata("4518754"),
            zenodo_file("4518754", "misc.zip", "remote/misc.zip", max_mb=5.0),
            zenodo_file("4518754", "S3.tar.gz", "remote/eeg/S3.tar.gz.head.bin", range_bytes=65536),
            zenodo_file("4518754", "stimuli.zip", "remote/archive_headers/stimuli.zip.head.bin", range_bytes=65536),
        ],
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
        remote_files=[
            RemoteFile("https://raw.githubusercontent.com/sstober/openmiir/master/README.md", "remote/README.md", max_mb=5.0),
            RemoteFile("https://raw.githubusercontent.com/sstober/openmiir/master/meta/Stimuli_Meta.v2.xlsx", "remote/meta/Stimuli_Meta.v2.xlsx", max_mb=5.0),
            RemoteFile("https://raw.githubusercontent.com/sstober/openmiir/master/meta/beats.v2/1_beats.txt", "remote/meta/1_beats.txt", max_mb=1.0),
        ],
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
        remote_files=[
            openneuro_file("ds003774", "dataset_description.json", "remote/dataset_description.json"),
            openneuro_file("ds003774", "README", "remote/README"),
            openneuro_file("ds003774", "sourcedata/sub-001/eeg/sub-001_task-ListeningandResponse_events.tsv", "remote/eeg/events.tsv"),
            openneuro_file("ds003774", "sourcedata/sub-001/eeg/sub-001_task-ListeningandResponse_channels.tsv", "remote/eeg/channels.tsv"),
            openneuro_file("ds003774", "sourcedata/sub-001/eeg/sub-001_task-ListeningandResponse_eeg.json", "remote/eeg/eeg.json"),
            openneuro_file("ds003774", "sourcedata/sub-001/eeg/sub-001_task-ListeningandResponse_eeg.set", "remote/eeg/eeg.set.head.bin", range_bytes=65536),
            openneuro_file("ds003774", "Code/ESongs/1.esh.wav", "remote/stimuli/1.esh.wav.header.bin", range_bytes=65536),
        ],
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
        remote_files=[
            zenodo_metadata("4537751"),
            zenodo_file("4537751", "behavioural_data.xlsx", "remote/behavioural_data.xlsx", max_mb=1.0),
            zenodo_file("4537751", "madeeg_raw.yaml", "remote/madeeg_raw.yaml", max_mb=1.0),
            zenodo_file("4537751", "madeeg_raw.hdf5", "remote/eeg/madeeg_raw.hdf5.head.bin", range_bytes=65536),
            zenodo_file("4537751", "stimuli.zip", "remote/archive_headers/stimuli.zip.head.bin", range_bytes=65536),
        ],
    ),
    DatasetSpec(
        slug="four_talker_aad_10803261",
        title="Four-Talker AAD (Yan 2024)",
        category="mandarin",
        priority="P0",
        source_url="https://zenodo.org/records/10803261",
        sample_goal="Zenodo record metadata for 4-speaker spatialized Mandarin AAD (64ch NeuSen + cEEGrid).",
        access_note="Zenodo public. Full EEG and audio bundles are large; automatic sample stores record metadata only.",
        remote_files=[
            zenodo_metadata("10803261"),
            zenodo_file("10803261", "ear_raw.zip", "remote/eeg/ear_raw.zip.head.bin", range_bytes=65536),
        ],
    ),
    DatasetSpec(
        slug="four_direction_aad_10803229",
        title="Four-Direction AAD (Yan 2024)",
        category="mandarin",
        priority="P0",
        source_url="https://zenodo.org/records/10803229",
        sample_goal="Zenodo record metadata for 4-direction spatialized Mandarin AAD (64ch, anechoic).",
        access_note="Zenodo public. Full EEG and audio bundles are large; automatic sample stores record metadata only.",
        remote_files=[
            zenodo_metadata("10803229"),
            zenodo_file("10803229", "EEG_raw.zip", "remote/eeg/EEG_raw.zip.head.bin", range_bytes=65536),
        ],
    ),
    DatasetSpec(
        slug="non_block_aad_14887886",
        title="Non-block Design AAD (Yan 2025)",
        category="mandarin",
        priority="P0/P1",
        source_url="https://zenodo.org/records/14887886",
        sample_goal="Zenodo record metadata for non-block 4-speaker Mandarin AAD with attention switching.",
        access_note="Zenodo public. Full EEG and audio bundles are large; automatic sample stores record metadata only.",
        remote_files=[
            zenodo_metadata("14887886"),
            zenodo_file("14887886", "eareeg.zip", "remote/eeg/eareeg.zip.head.bin", range_bytes=65536),
            zenodo_file("14887886", "scalpeeg.zip", "remote/eeg/scalpeeg.zip.head.bin", range_bytes=65536),
        ],
    ),
    DatasetSpec(
        slug="asa_lin2024_11541114",
        title="ASA multi-angle Mandarin AAD (Lin 2024)",
        category="mandarin",
        priority="P1",
        source_url="https://zenodo.org/records/11541114",
        sample_goal="Zenodo record metadata for multi-angle (±5°–±90°) Mandarin 2-speaker AAD.",
        access_note="Zenodo public. Full EEG and audio bundles are large; automatic sample stores record metadata only.",
        remote_files=[
            zenodo_metadata("11541114"),
            zenodo_file("11541114", "preproc.zip", "remote/preproc.zip", max_mb=1.0),
            zenodo_file("11541114", "S001.zip", "remote/eeg/S001.zip.head.bin", range_bytes=65536),
        ],
    ),
    DatasetSpec(
        slug="fuglsang2020_3618205",
        title="Fuglsang 2020 large-sample AAD",
        category="aad_controlled",
        priority="P1",
        source_url="https://zenodo.org/record/3618205",
        sample_goal="Zenodo record metadata for 44-subject Danish competing-speech AAD including hearing-impaired.",
        access_note="Zenodo public. Full EEG and audio bundles are large; automatic sample stores record metadata only.",
        remote_files=[zenodo_metadata("3618205")],
    ),
    DatasetSpec(
        slug="rotaru2024_11058711",
        title="Rotaru 2024 long-session AAD",
        category="aad_controlled",
        priority="P1",
        source_url="https://zenodo.org/records/11058711",
        sample_goal="Zenodo record metadata for 13-subject Dutch AAD with 80 min per subject.",
        access_note="Zenodo public. Full EEG and audio bundles are large; automatic sample stores record metadata only.",
        remote_files=[
            zenodo_metadata("11058711"),
            zenodo_file("11058711", "README.txt", "remote/README.txt", max_mb=1.0),
            zenodo_file("11058711", "2024-AV-GC-AAD-sub03_preprocessed.mat", "remote/eeg/sub03_preprocessed.mat.head.bin", range_bytes=65536),
        ],
    ),
    DatasetSpec(
        slug="geirnaert2025_16536441",
        title="Geirnaert 2025 multi-device AAD",
        category="aad_controlled",
        priority="P1",
        source_url="https://zenodo.org/records/16536441",
        sample_goal="Zenodo record metadata for simultaneous scalp/around-ear/in-ear EEG AAD (Danish, 15 subjects).",
        access_note="Zenodo public. Full EEG and audio bundles are large; automatic sample stores record metadata only.",
        remote_files=[
            zenodo_metadata("16536441"),
            zenodo_file("16536441", "experiment-manual.pdf", "remote/experiment-manual.pdf", max_mb=5.0),
            zenodo_file("16536441", "preprocessedData.zip", "remote/eeg/preprocessedData.zip.head.bin", range_bytes=65536),
            zenodo_file("16536441", "bids_dataset.zip", "remote/eeg/bids_dataset.zip.head.bin", range_bytes=65536),
        ],
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
    if remote.range_bytes is not None:
        try:
            request = urllib.request.Request(
                remote.url,
                headers={
                    "User-Agent": "eeg-voice-sample-downloader/0.1",
                    "Range": f"bytes=0-{remote.range_bytes - 1}",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read(remote.range_bytes + 1)
                headers = dict(response.headers.items())
            if len(data) > remote.range_bytes:
                data = data[: remote.range_bytes]
            dest.write_bytes(data)
            return {
                "url": remote.url,
                "relpath": str(dest.relative_to(dataset_dir)),
                "required": remote.required,
                "status": "downloaded_range",
                "bytes": len(data),
                "range_bytes": remote.range_bytes,
                "content_range": headers.get("Content-Range"),
                "content_length": headers.get("Content-Length"),
            }
        except urllib.error.HTTPError as exc:
            return {
                "url": remote.url,
                "relpath": remote.relpath,
                "required": remote.required,
                "status": "http_error",
                "code": exc.code,
                "reason": str(exc.reason),
                "range_bytes": remote.range_bytes,
            }
        except Exception as exc:
            return {
                "url": remote.url,
                "relpath": remote.relpath,
                "required": remote.required,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "range_bytes": remote.range_bytes,
            }
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
    range_count = sum(1 for item in remote_results if item.get("status") == "downloaded_range")
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
        f"- remote byte-range samples downloaded: {range_count}",
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
    has_sample = any(item.get("copied") for item in local_results) or any(
        item.get("status") in {"downloaded", "downloaded_range"} for item in remote_results
    )
    missing_required = [
        item
        for item in local_results
        if item.get("required") and not item.get("copied")
    ] + [
        item
        for item in remote_results
        if item.get("required") and item.get("status") not in {"downloaded", "downloaded_range"}
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


def classify_sample(relpath: str) -> str:
    path = relpath.lower()
    name = Path(path).name
    parts = Path(path).parts
    suffixes = Path(path).suffixes
    if "stimuli" in parts or "audio" in parts or "audio" in name or any(token in path for token in ["vocal", ".wav", ".ogg", ".mp3", ".flac"]):
        return "audio"
    eeg_suffixes = {".edf", ".fif", ".vhdr", ".vmrk", ".eeg", ".set", ".mat", ".bdf", ".cnt"}
    if (
        "eeg" in parts
        or any(suffix in eeg_suffixes for suffix in suffixes)
        or name.endswith(("channels.tsv", "events.tsv", "events.dat"))
    ):
        return "eeg"
    if name.endswith((".json", ".tsv", ".csv", ".xlsx", ".txt", ".md", ".html", ".pdf", "readme")):
        return "metadata"
    return "other"


def collect_sample_files(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        base = Path(record["dataset_dir"])
        for item in record.get("local_results", []):
            for relpath in item.get("copied", []):
                rows.append(
                    {
                        "slug": record["slug"],
                        "category": record["category"],
                        "priority": record["priority"],
                        "kind": classify_sample(relpath),
                        "source": "local",
                        "status": "copied",
                        "bytes": (base / relpath).stat().st_size if (base / relpath).exists() else "",
                        "path": str(base / relpath),
                    }
                )
        for item in record.get("remote_results", []):
            if item.get("status") not in {"downloaded", "downloaded_range"}:
                continue
            relpath = item.get("relpath", "")
            rows.append(
                {
                    "slug": record["slug"],
                    "category": record["category"],
                    "priority": record["priority"],
                    "kind": classify_sample(relpath),
                    "source": "remote",
                    "status": item.get("status"),
                    "bytes": item.get("bytes", ""),
                    "path": str(base / relpath),
                }
            )
    return rows


def write_unified_sample_links(root: Path, sample_files: list[dict]) -> tuple[Path, int]:
    unified_samples_dir = root / "_unified_samples"
    if unified_samples_dir.exists():
        shutil.rmtree(unified_samples_dir)
    unified_samples_dir.mkdir(parents=True, exist_ok=True)

    link_count = 0
    for row in sample_files:
        source = Path(row["path"])
        if not source.exists():
            continue
        dataset_dir = unified_samples_dir / row["slug"] / row["kind"]
        dataset_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{row['source']}__"
        link = safe_name(dataset_dir / f"{prefix}{source.name}")
        target = os.path.relpath(source, start=link.parent)
        link.symlink_to(target)
        row["unified_path"] = str(link)
        link_count += 1
    return unified_samples_dir, link_count


def write_root_files(root: Path, records: list[dict]) -> None:
    ready = [r for r in records if r["status"] == "ready_or_partial_sample"]
    manual = [r for r in records if r["status"] != "ready_or_partial_sample"]
    sample_files = collect_sample_files(records)
    unified_samples_dir, unified_link_count = write_unified_sample_links(root, sample_files)
    unified_dir = root / "_unified_index"
    unified_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "dataset_count": len(records),
        "ready_or_partial_count": len(ready),
        "manual_or_missing_count": len(manual),
        "sample_file_count": len(sample_files),
        "unified_index_dir": str(unified_dir),
        "unified_samples_dir": str(unified_samples_dir),
        "unified_sample_link_count": unified_link_count,
        "records": records,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (unified_dir / "manifest_compact.json").write_text(
        json.dumps(
            {
                "created_at": manifest["created_at"],
                "dataset_count": len(records),
                "ready_or_partial_count": len(ready),
                "manual_or_missing_count": len(manual),
                "sample_file_count": len(sample_files),
                "unified_samples_dir": str(unified_samples_dir),
                "unified_sample_link_count": unified_link_count,
                "records": [
                    {
                        "slug": r["slug"],
                        "category": r["category"],
                        "priority": r["priority"],
                        "status": r["status"],
                        "dataset_dir": r["dataset_dir"],
                    }
                    for r in records
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    tsv_lines = ["slug\tcategory\tpriority\tkind\tsource\tstatus\tbytes\tpath\tunified_path"]
    for row in sample_files:
        tsv_lines.append(
            "\t".join(str(row.get(key, "")).replace("\t", " ") for key in ["slug", "category", "priority", "kind", "source", "status", "bytes", "path", "unified_path"])
        )
    (unified_dir / "sample_files.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
    lines = [
        "# Voice EEG Dataset Samples",
        "",
        "This directory is ignored by git and stores one or two sample examples per dataset when available.",
        "",
        f"- dataset count: {len(records)}",
        f"- ready or partial sample folders: {len(ready)}",
        f"- manual or missing folders: {len(manual)}",
        f"- unified sample links: {unified_link_count}",
        f"- unified sample folder: `{unified_samples_dir}`",
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

    by_dataset: dict[str, dict[str, int]] = {}
    for row in sample_files:
        stats = by_dataset.setdefault(row["slug"], {"audio": 0, "eeg": 0, "metadata": 0, "other": 0})
        stats[row["kind"]] = stats.get(row["kind"], 0) + 1
    status_lines = [
        "# Unified Sample Status",
        "",
        f"- root: `{root}`",
        f"- unified sample folder: `{unified_samples_dir}`",
        f"- sample files indexed: `{len(sample_files)}`",
        f"- unified sample links: `{unified_link_count}`",
        "",
        "| Dataset | Status | Audio files | EEG files | Metadata/other files | Folder |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        stats = by_dataset.get(record["slug"], {"audio": 0, "eeg": 0, "metadata": 0, "other": 0})
        status_lines.append(
            f"| `{record['slug']}` | `{record['status']}` | {stats.get('audio', 0)} | {stats.get('eeg', 0)} | {stats.get('metadata', 0) + stats.get('other', 0)} | `{record['dataset_dir']}` |"
        )
    (unified_dir / "sample_status.md").write_text("\n".join(status_lines) + "\n", encoding="utf-8")


def selected_specs(names: Iterable[str] | None) -> list[DatasetSpec]:
    if not names:
        return DATASETS
    wanted = set(names)
    specs = [spec for spec in DATASETS if spec.slug in wanted]
    missing = sorted(wanted - {spec.slug for spec in specs})
    if missing:
        raise SystemExit(f"Unknown dataset slug(s): {', '.join(missing)}")
    return specs


def load_existing_records(root: Path, specs: list[DatasetSpec]) -> list[dict]:
    records: list[dict] = []
    for spec in specs:
        status_path = root / spec.category / spec.slug / "status.json"
        if not status_path.exists():
            continue
        try:
            records.append(json.loads(status_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", action="append", help="Dataset slug to prepare. May be repeated. Defaults to all.")
    parser.add_argument("--allow-large", action="store_true", help="Allow remote downloads larger than the configured max_mb.")
    parser.add_argument("--max-mb", type=float, default=25.0, help="Default max MB for remote files without an explicit limit.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    ensure_dir(args.root)
    specs = selected_specs(args.dataset)
    records = []
    for idx, spec in enumerate(specs, start=1):
        print(f"[{idx}/{len(specs)}] {spec.slug}", flush=True)
        records.append(
            prepare_dataset(spec, root=args.root, allow_large=args.allow_large, default_max_mb=args.max_mb, timeout=args.timeout)
        )
    if args.dataset:
        records = load_existing_records(args.root, DATASETS)
    write_root_files(args.root, records)
    print(json.dumps({"root": str(args.root), "datasets": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
