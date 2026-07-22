from __future__ import annotations

import csv
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from src.combined_0715.audio_eval import validate_cache_arrays


DATASETS = ("feis", "karaone", "ds004306")
DATASET_IDS = {name: index for index, name in enumerate(DATASETS)}
COMMON_CHANNELS = (
    "F3", "FC5", "AF3", "F7", "T7", "P7", "O1", "O2", "P8", "T8", "F8", "AF4", "FC6", "F4",
)
PAIRING_LEVELS = {
    "karaone_same_trial_overt": "exact_trial",
    "feis_subject_label": "subject_label",
    "weak_category_level": "category_prototype",
}


def read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().strip("/").lower())


def pairing_level(row: dict[str, str]) -> str:
    confidence = str(row.get("pairing_confidence", ""))
    if confidence not in PAIRING_LEVELS:
        raise ValueError(f"Unknown pairing confidence: {confidence!r}")
    return PAIRING_LEVELS[confidence]


@dataclass(frozen=True)
class Montage:
    montage_id: str
    channel_names: tuple[str, ...]
    channel_types: tuple[str, ...]
    channel_xyz: np.ndarray
    reference_scheme: str
    coordinate_source: str


class MontageRegistry:
    SCHEMA_VERSION = "openvoice-montage-v1"

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(f"Unsupported montage registry schema: {raw.get('schema_version')}")
        self.recordings = {str(key): str(value) for key, value in raw.get("recordings", {}).items()}
        self.montages: dict[str, Montage] = {}
        for montage_id, value in raw.get("montages", {}).items():
            names = tuple(map(str, value["channel_names"]))
            types = tuple(map(str, value["channel_types"]))
            xyz = np.asarray(value["channel_xyz"], dtype=np.float32)
            if len(names) != len(types) or xyz.shape != (len(names), 3):
                raise ValueError(f"Invalid montage geometry for {montage_id}")
            if len({name.upper() for name in names}) != len(names):
                raise ValueError(f"Duplicate channel names in montage {montage_id}")
            if not np.isfinite(xyz).all():
                raise ValueError(f"Non-finite channel coordinates in montage {montage_id}")
            self.montages[montage_id] = Montage(
                montage_id=str(montage_id),
                channel_names=names,
                channel_types=types,
                channel_xyz=xyz,
                reference_scheme=str(value["reference_scheme"]),
                coordinate_source=str(value["coordinate_source"]),
            )
        if not self.montages:
            raise ValueError("Montage registry has no montages")

    def for_recording(self, eeg_relpath: str) -> Montage:
        montage_id = self.recordings.get(str(eeg_relpath))
        if montage_id is None:
            raise KeyError(f"No montage registered for EEG payload {eeg_relpath}")
        return self.montages[montage_id]


def parse_asa_elc(path: str | Path) -> dict[str, np.ndarray]:
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    try:
        count = int(next(line.split("=", 1)[1] for line in lines if line.startswith("NumberPositions")))
        position_start = lines.index("Positions") + 1
        label_start = lines.index("Labels") + 1
    except (StopIteration, ValueError, IndexError) as error:
        raise ValueError(f"Invalid ASA electrode file: {path}") from error
    coordinates = np.asarray(
        [[float(value) for value in lines[position_start + index].split()] for index in range(count)],
        dtype=np.float32,
    )
    labels = lines[label_start : label_start + count]
    if coordinates.shape != (count, 3) or len(labels) != count:
        raise ValueError(f"ASA electrode count mismatch: {path}")
    scale = np.linalg.norm(coordinates, axis=1).max()
    coordinates = coordinates / max(float(scale), 1e-8)
    return {label.upper(): coordinates[index] for index, label in enumerate(labels)}


def channel_type(name: str) -> str:
    upper = name.upper()
    if "EOG" in upper:
        return "eog"
    if upper in {"M1", "M2", "A1", "A2"}:
        return "reference"
    return "eeg"


def build_montage_registry_payload(
    eeg_root: str | Path,
    rows: Iterable[dict[str, str]],
    coordinate_lookup: dict[str, np.ndarray],
    *,
    reference_scheme: str = "common_average",
    coordinate_source: str = "standard_1005",
) -> dict[str, Any]:
    root = Path(eeg_root).resolve()
    recordings: dict[str, str] = {}
    montages: dict[str, dict[str, Any]] = {}
    signature_to_id: dict[tuple[str, ...], str] = {}
    for row in rows:
        relative = str(row["eeg_relpath"])
        if relative in recordings:
            continue
        with np.load(root / relative, allow_pickle=False) as payload:
            names = tuple(str(value) for value in np.asarray(payload["channel_names"]).reshape(-1))
        types = tuple(channel_type(name) for name in names)
        eeg_names = tuple(name for name, kind in zip(names, types) if kind == "eeg")
        if not eeg_names:
            raise ValueError(f"No EEG channels in {relative}")
        unresolved = [name for name in eeg_names if name.upper() not in coordinate_lookup]
        if unresolved:
            raise ValueError(f"Unresolved standard coordinates in {relative}: {unresolved}")
        signature = tuple(name.upper() for name in eeg_names)
        montage_id = signature_to_id.get(signature)
        if montage_id is None:
            montage_id = f"montage_{len(signature_to_id):03d}_{len(eeg_names)}ch"
            signature_to_id[signature] = montage_id
            montages[montage_id] = {
                "channel_names": list(eeg_names),
                "channel_types": ["eeg"] * len(eeg_names),
                "channel_xyz": [coordinate_lookup[name.upper()].astype(float).tolist() for name in eeg_names],
                "reference_scheme": reference_scheme,
                "coordinate_source": coordinate_source,
            }
        recordings[relative] = montage_id
    return {
        "schema_version": MontageRegistry.SCHEMA_VERSION,
        "montages": montages,
        "recordings": recordings,
    }


@dataclass(frozen=True)
class OpenVoiceContext:
    config_path: Path
    config: dict[str, Any]
    eeg_root: Path
    audio_root: Path
    manifest_path: Path
    split_path: Path
    rows: tuple[dict[str, str], ...]
    split: dict[str, Any]
    montage_registry: MontageRegistry
    label_to_index: dict[str, int]
    subject_to_index: dict[str, int]

    def split_for(self, row: dict[str, str]) -> str:
        dataset_split = self.split["datasets"][row["dataset"]]
        group = row["subject_group_id"]
        for name in ("train", "validation", "test"):
            if group in dataset_split[name]:
                return name
        raise ValueError(f"Subject group not covered by split: {group}")


def resolve_config_path(config_path: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def load_context(config_path: str | Path) -> OpenVoiceContext:
    path = Path(config_path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    eeg_root = resolve_config_path(path, cfg["data"]["eeg_output_root"])
    audio_root = resolve_config_path(path, cfg["data"].get("audio_output_root", cfg["data"]["eeg_output_root"]))
    manifest_path = eeg_root / "manifests" / "unified_trials.csv"
    split_path = resolve_config_path(path, cfg["data"]["subject_split_file"])
    registry_path = resolve_config_path(path, cfg["data"]["montage_registry"])
    rows = read_csv(manifest_path)
    required = {
        "sample_key", "dataset", "subject_group_id", "subject_recording_id", "trial_index",
        "label", "eeg_relpath", "eeg_row", "eeg_valid_samples", "audio_key", "pairing_confidence",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Unified manifest lacks fields: {sorted(missing)}")
    keys = [row["sample_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("sample_key must be unique")
    split = yaml.safe_load(split_path.read_text(encoding="utf-8"))
    for dataset in DATASETS:
        groups = {name: set(split["datasets"][dataset][name]) for name in ("train", "validation", "test")}
        if (groups["train"] & groups["validation"]) or (groups["train"] & groups["test"]) or (groups["validation"] & groups["test"]):
            raise ValueError(f"Subject split overlap for {dataset}")
        observed = {row["subject_group_id"] for row in rows if row["dataset"] == dataset}
        if observed != set().union(*groups.values()):
            raise ValueError(f"Subject split coverage mismatch for {dataset}")
    labels = sorted({normalize_label(row["label"]) for row in rows})
    train_subjects = sorted(
        group for dataset in DATASETS for group in split["datasets"][dataset]["train"]
    )
    return OpenVoiceContext(
        config_path=path,
        config=cfg,
        eeg_root=eeg_root,
        audio_root=audio_root,
        manifest_path=manifest_path,
        split_path=split_path,
        rows=rows,
        split=split,
        montage_registry=MontageRegistry(registry_path),
        label_to_index={label: index for index, label in enumerate(labels)},
        subject_to_index={subject: index for index, subject in enumerate(train_subjects)},
    )


class AudioCodeBank:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        with np.load(self.path, allow_pickle=False) as raw:
            checks = validate_cache_arrays(raw)
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise ValueError(f"Invalid EnCodec cache ({', '.join(failed)}): {self.path}")
            self.keys = np.asarray(raw["keys"]).astype(str)
            self.datasets = np.asarray(raw["datasets"]).astype(str)
            self.labels = np.asarray(raw["labels"]).astype(str)
            self.codes = np.asarray(raw["encodec_codes"], dtype=np.int64)
            self.scale = np.asarray(raw["encodec_scale"], dtype=np.float32)
            self.scale_valid = np.asarray(raw["encodec_scale_valid"], dtype=np.bool_)
            self.envelope = np.asarray(raw["audio_envelope"], dtype=np.float32)
            self.onset = np.asarray(raw["onset"], dtype=np.float32)
            self.duration = np.asarray(raw["duration"], dtype=np.float32)
            self.code_valid_steps = np.asarray(raw["code_valid_steps"], dtype=np.int64)
            self.fit_split = np.asarray(raw["fit_split"], dtype=np.bool_)
            self.schema_version = str(np.asarray(raw["cache_schema_version"]).reshape(-1)[0])
        self.key_to_index = {key: index for index, key in enumerate(self.keys.tolist())}


class LabelFreeAudioDataset(Dataset[dict[str, Any]]):
    """Label-independent project/public audio prior dataset.

    Locked-test project audio is never added to ``records``.  Labels are not
    returned, so changing metadata cannot affect the audio model.
    """

    def __init__(
        self,
        context: "OpenVoiceContext",
        teachers: "TeacherBank",
        *,
        split: str,
        include_public: bool = True,
    ):
        if split not in {"train", "validation"}:
            raise ValueError("Audio prior split must be train or validation")
        cfg = context.config
        project_path = resolve_config_path(context.config_path, cfg["paths"]["project_audio_cache"])
        self.project = AudioCodeBank(project_path)
        self.public: dict[str, np.ndarray] | None = None
        self.teachers = teachers
        key_splits: dict[str, set[str]] = {}
        for row in context.rows:
            if row["dataset"] not in {"karaone", "feis"}:
                continue
            key_splits.setdefault(row["audio_key"], set()).add(context.split_for(row))
        self.records: list[tuple[str, int]] = []
        for index, key in enumerate(self.project.keys.tolist()):
            observed = key_splits.get(key, set())
            if split == "train" and observed == {"train"}:
                self.records.append(("project", index))
            elif split == "validation" and "validation" in observed and "train" not in observed and "test" not in observed:
                self.records.append(("project", index))
        public_path = resolve_config_path(context.config_path, cfg["paths"]["public_audio_cache"])
        if include_public:
            if not public_path.is_file():
                raise FileNotFoundError(f"Public audio cache is required: {public_path}")
            with np.load(public_path, allow_pickle=False) as raw:
                self.public = {key: np.asarray(raw[key]) for key in raw.files}
            fit = np.asarray(self.public["fit_split"], dtype=bool)
            selected = fit if split == "train" else ~fit
            self.records.extend(("public", index) for index in np.flatnonzero(selected))
        if not self.records:
            raise ValueError(f"No label-free audio records for {split}")
        missing = [self._key(source, index) for source, index in self.records if teachers.audio_tokens.get(self._key(source, index)) is None]
        if missing:
            raise ValueError(f"Teacher cache misses {len(missing)} audio keys")

    def _key(self, source: str, index: int) -> str:
        if source == "project":
            return str(self.project.keys[index])
        assert self.public is not None
        return str(self.public["keys"][index])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> dict[str, Any]:
        source, index = self.records[item]
        key = self._key(source, index)
        if source == "project":
            codes = self.project.codes[index]
            valid_steps = int(self.project.code_valid_steps[index])
        else:
            assert self.public is not None
            codes = np.asarray(self.public["encodec_codes"][index])
            valid_steps = int(self.public["code_valid_steps"][index])
        valid = np.arange(codes.shape[-1]) < valid_steps
        return {
            "codes": torch.from_numpy(np.ascontiguousarray(codes)).long(),
            "code_valid_mask": torch.from_numpy(np.broadcast_to(valid, codes.shape).copy()),
            "xlsr_tokens": torch.from_numpy(np.ascontiguousarray(self.teachers.audio_tokens.get(key))).float(),
            "audio_key": key,
            "source": source,
        }


class TeacherBank:
    SCHEMA_VERSION = "openvoice-teacher-v1"

    def __init__(self, path: str | Path | None):
        self.path = Path(path).resolve() if path else None
        self.audio_tokens: Any = {}
        self.text_embeddings: dict[str, np.ndarray] = {}
        self.audio_dimension = 0
        self.text_dimension = 0
        if self.path is None:
            return
        if self.path.is_dir():
            index_path = self.path / "index.json"
            raw_index = json.loads(index_path.read_text(encoding="utf-8"))
            if raw_index.get("schema_version") != self.SCHEMA_VERSION:
                raise ValueError(f"Unsupported teacher cache: {raw_index.get('schema_version')}")
            self.audio_dimension = int(raw_index["audio_dimension"])
            self.text_dimension = int(raw_index.get("text_dimension", 0))
            self.audio_tokens = _ShardedTeacherTokens(self.path, raw_index["audio_index"])
            text_file = raw_index.get("text_file")
            if text_file:
                with np.load(self.path / str(text_file), allow_pickle=False) as text_raw:
                    keys = np.asarray(text_raw["keys"]).astype(str)
                    values = np.asarray(text_raw["embeddings"], dtype=np.float32)
                self.text_embeddings = {key: values[index] for index, key in enumerate(keys)}
            return
        with np.load(self.path, allow_pickle=False) as raw:
            version = str(np.asarray(raw["schema_version"]).reshape(-1)[0])
            if version != self.SCHEMA_VERSION:
                raise ValueError(f"Unsupported teacher cache: {version}")
            audio_keys = np.asarray(raw["audio_keys"]).astype(str)
            audio_tokens = np.asarray(raw["xlsr_tokens"], dtype=np.float32)
            text_keys = np.asarray(raw["text_keys"]).astype(str)
            text_embeddings = np.asarray(raw["text_embeddings"], dtype=np.float32)
        if audio_tokens.ndim != 3 or len(audio_keys) != len(audio_tokens):
            raise ValueError("Teacher audio arrays are inconsistent")
        if text_embeddings.ndim != 2 or len(text_keys) != len(text_embeddings):
            raise ValueError("Teacher text arrays are inconsistent")
        self.audio_dimension = int(audio_tokens.shape[-1])
        self.text_dimension = int(text_embeddings.shape[-1])
        self.audio_tokens = {key: audio_tokens[index] for index, key in enumerate(audio_keys)}
        self.text_embeddings = {key: text_embeddings[index] for index, key in enumerate(text_keys)}


class _ShardedTeacherTokens:
    def __init__(self, root: Path, index: dict[str, list[Any]], max_open_shards: int = 2):
        self.root = root
        self.index = {str(key): (str(value[0]), int(value[1])) for key, value in index.items()}
        self.max_open_shards = max(1, int(max_open_shards))
        self._shards: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: str, default: Any = None) -> Any:
        location = self.index.get(str(key))
        if location is None:
            return default
        shard, row = location
        if shard not in self._shards:
            with np.load(self.root / shard, allow_pickle=False) as raw:
                self._shards[shard] = np.asarray(raw["tokens"], dtype=np.float32)
            self._shards.move_to_end(shard)
            while len(self._shards) > self.max_open_shards:
                self._shards.popitem(last=False)
        return self._shards[shard][row]


def _row_selected(
    context: OpenVoiceContext,
    row: dict[str, str],
    split: str,
    generalization: str,
    holdout_label: str | None,
) -> bool:
    actual = context.split_for(row)
    label = normalize_label(row["label"])
    holdout = normalize_label(holdout_label) if holdout_label else None
    if generalization == "g1":
        return actual == split
    if holdout is None:
        raise ValueError(f"{generalization} requires a holdout label")
    if split == "train":
        return actual == "train" and label != holdout
    if generalization == "g2":
        return actual == "train" and label == holdout
    if generalization == "g3":
        return actual == split and label == holdout
    raise ValueError(f"Unknown generalization setting: {generalization}")


class OpenVoiceEEGDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        context: OpenVoiceContext,
        bank: AudioCodeBank,
        *,
        split: str,
        generalization: str = "g1",
        holdout_label: str | None = None,
        datasets: Sequence[str] = DATASETS,
        eeg_samples: int | None = None,
        max_open_payloads: int = 4,
        teachers: TeacherBank | None = None,
    ):
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if generalization not in {"g1", "g2", "g3"}:
            raise ValueError("generalization must be g1, g2, or g3")
        selected = set(datasets)
        if not selected <= set(DATASETS):
            raise ValueError(f"Unknown datasets: {sorted(selected - set(DATASETS))}")
        self.context = context
        self.bank = bank
        self.split = split
        self.generalization = generalization
        self.holdout_label = holdout_label
        self.eeg_samples = int(eeg_samples or context.config["data"]["eeg_samples"])
        self.rows = tuple(
            row for row in context.rows
            if row["dataset"] in selected
            and _row_selected(context, row, split, generalization, holdout_label)
        )
        if not self.rows:
            raise ValueError(f"No rows for split={split}, generalization={generalization}, holdout={holdout_label}")
        missing_audio = sorted({row["audio_key"] for row in self.rows if row["audio_key"] not in bank.key_to_index})
        if missing_audio:
            raise ValueError(f"Audio cache misses {len(missing_audio)} keys")
        self._payloads: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self.max_open_payloads = max(1, int(max_open_payloads))
        self.teachers = teachers or TeacherBank(None)

    def __len__(self) -> int:
        return len(self.rows)

    def _payload(self, relative: str) -> dict[str, np.ndarray]:
        if relative not in self._payloads:
            with np.load(self.context.eeg_root / relative, allow_pickle=False) as raw:
                self._payloads[relative] = {name: np.asarray(raw[name]) for name in raw.files}
            self._payloads.move_to_end(relative)
            while len(self._payloads) > self.max_open_payloads:
                self._payloads.popitem(last=False)
        return self._payloads[relative]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        relative = row["eeg_relpath"]
        raw = self._payload(relative)
        eeg_row = int(row["eeg_row"])
        eeg = np.asarray(raw["eeg"][eeg_row], dtype=np.float32)
        montage = self.context.montage_registry.for_recording(relative)
        raw_names = [str(value) for value in np.asarray(raw["channel_names"]).reshape(-1)]
        by_upper = {name.upper(): idx for idx, name in enumerate(raw_names)}
        missing = [name for name in montage.channel_names if name.upper() not in by_upper]
        if missing:
            raise ValueError(f"NPZ/montage mismatch in {relative}: {missing}")
        eeg = eeg[[by_upper[name.upper()] for name in montage.channel_names], : self.eeg_samples]
        valid = min(max(int(row["eeg_valid_samples"]), 1), eeg.shape[-1])
        if eeg.shape[-1] < self.eeg_samples:
            eeg = np.pad(eeg, ((0, 0), (0, self.eeg_samples - eeg.shape[-1])))
        time_mask = np.arange(self.eeg_samples) < valid
        eeg[:, ~time_mask] = 0.0
        audio_index = self.bank.key_to_index[row["audio_key"]]
        code_steps = int(self.bank.code_valid_steps[audio_index])
        code_mask = np.arange(self.bank.codes.shape[-1]) < code_steps
        label_key = normalize_label(row["label"])
        subject_index = self.context.subject_to_index.get(row["subject_group_id"], -1)
        common = np.asarray([name.upper() in {value.upper() for value in COMMON_CHANNELS} for name in montage.channel_names])
        audio_teacher = self.teachers.audio_tokens.get(row["audio_key"])
        text_teacher = self.teachers.text_embeddings.get(label_key)
        audio_dimension = self.teachers.audio_dimension or int(self.context.config["audio_model"].get("xlsr_dimension", 1024))
        text_dimension = self.teachers.text_dimension or int(self.context.config["teachers"]["text_dimension"])
        teacher_steps = int(self.context.config["teachers"]["audio_token_steps"])
        return {
            "eeg": torch.from_numpy(np.ascontiguousarray(eeg)),
            "channel_xyz": torch.from_numpy(np.ascontiguousarray(montage.channel_xyz)),
            "channel_mask": torch.ones(len(montage.channel_names), dtype=torch.bool),
            "common_channel_mask": torch.from_numpy(common),
            "time_mask": torch.from_numpy(time_mask),
            "codes": torch.from_numpy(np.ascontiguousarray(self.bank.codes[audio_index])).long(),
            "code_valid_mask": torch.from_numpy(np.broadcast_to(code_mask, self.bank.codes[audio_index].shape).copy()),
            "audio_envelope": torch.from_numpy(np.ascontiguousarray(self.bank.envelope[audio_index])),
            "onset": torch.tensor(self.bank.onset[audio_index], dtype=torch.float32),
            "duration": torch.tensor(self.bank.duration[audio_index], dtype=torch.float32),
            "audio_idx": torch.tensor(audio_index, dtype=torch.long),
            "label_idx": torch.tensor(self.context.label_to_index[label_key], dtype=torch.long),
            "dataset_idx": torch.tensor(DATASET_IDS[row["dataset"]], dtype=torch.long),
            "subject_idx": torch.tensor(subject_index, dtype=torch.long),
            "xlsr_tokens": torch.from_numpy(np.ascontiguousarray(audio_teacher)).float()
            if audio_teacher is not None
            else torch.zeros(teacher_steps, audio_dimension, dtype=torch.float32),
            "text_embedding": torch.from_numpy(np.ascontiguousarray(text_teacher)).float()
            if text_teacher is not None
            else torch.zeros(text_dimension, dtype=torch.float32),
            "has_audio_teacher": torch.tensor(audio_teacher is not None, dtype=torch.bool),
            "has_text_teacher": torch.tensor(text_teacher is not None, dtype=torch.bool),
            "sample_key": row["sample_key"],
            "audio_key": row["audio_key"],
            "dataset": row["dataset"],
            "subject_group_id": row["subject_group_id"],
            "label": row["label"],
            "label_key": label_key,
            "pairing_confidence": row["pairing_confidence"],
            "pairing_level": pairing_level(row),
            "channel_names": montage.channel_names,
            "eeg_relpath": relative,
            "eeg_row": eeg_row,
        }


def collate_openvoice(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch = len(samples)
    max_channels = max(sample["eeg"].shape[0] for sample in samples)
    max_time = max(sample["eeg"].shape[1] for sample in samples)
    eeg = torch.zeros(batch, max_channels, max_time, dtype=torch.float32)
    xyz = torch.zeros(batch, max_channels, 3, dtype=torch.float32)
    channel_mask = torch.zeros(batch, max_channels, dtype=torch.bool)
    common_mask = torch.zeros_like(channel_mask)
    time_mask = torch.zeros(batch, max_time, dtype=torch.bool)
    for index, sample in enumerate(samples):
        channels, times = sample["eeg"].shape
        eeg[index, :channels, :times] = sample["eeg"]
        xyz[index, :channels] = sample["channel_xyz"]
        channel_mask[index, :channels] = sample["channel_mask"]
        common_mask[index, :channels] = sample["common_channel_mask"]
        time_mask[index, :times] = sample["time_mask"]
    tensor_keys = (
        "codes", "code_valid_mask", "audio_envelope", "onset", "duration", "audio_idx",
        "label_idx", "dataset_idx", "subject_idx",
        "xlsr_tokens", "text_embedding", "has_audio_teacher", "has_text_teacher",
    )
    output: dict[str, Any] = {
        "eeg": eeg,
        "channel_xyz": xyz,
        "channel_mask": channel_mask,
        "common_channel_mask": common_mask,
        "time_mask": time_mask,
    }
    for key in tensor_keys:
        output[key] = torch.stack([sample[key] for sample in samples])
    list_keys = (
        "sample_key", "audio_key", "dataset", "subject_group_id", "label", "label_key",
        "pairing_confidence", "pairing_level", "channel_names", "eeg_relpath", "eeg_row",
    )
    for key in list_keys:
        output[key] = [sample[key] for sample in samples]
    return output


def common_montage_view(batch: dict[str, Any]) -> dict[str, Any]:
    output = dict(batch)
    output["channel_mask"] = batch["channel_mask"] & batch["common_channel_mask"]
    if not output["channel_mask"].any(dim=1).all():
        raise ValueError("Every common-montage view must retain at least one channel")
    output["eeg"] = batch["eeg"] * output["channel_mask"].unsqueeze(-1).to(batch["eeg"].dtype)
    return output


def stochastic_channel_view(
    batch: dict[str, Any],
    *,
    drop_probability: float,
    coordinate_noise_std: float = 0.0,
    region_drop_fraction: float = 0.0,
    bad_channel_probability: float = 0.0,
    time_mask_fraction: float = 0.0,
    noise_std: float = 0.0,
) -> dict[str, Any]:
    if not 0.0 <= float(drop_probability) < 1.0:
        raise ValueError("drop_probability must be in [0,1)")
    output = dict(batch)
    original = batch["channel_mask"].bool()
    keep = (torch.rand(original.shape, device=original.device) >= float(drop_probability)) & original
    for index in range(len(keep)):
        valid = torch.nonzero(original[index], as_tuple=False).flatten()
        if region_drop_fraction > 0 and len(valid) > 1:
            seed = valid[torch.randint(len(valid), (1,), device=valid.device)[0]]
            distance = torch.linalg.vector_norm(batch["channel_xyz"][index, valid] - batch["channel_xyz"][index, seed], dim=-1)
            count = min(len(valid) - 1, max(1, round(len(valid) * float(region_drop_fraction))))
            keep[index, valid[torch.topk(distance, k=count, largest=False).indices]] = False
        if bad_channel_probability > 0:
            bad = (torch.rand_like(keep[index], dtype=torch.float32) < float(bad_channel_probability)) & original[index]
            keep[index, bad] = False
        if not keep[index].any():
            first = torch.nonzero(original[index], as_tuple=False)[0, 0]
            keep[index, first] = True
    output["channel_mask"] = keep
    output["eeg"] = batch["eeg"] * keep.unsqueeze(-1).to(batch["eeg"].dtype)
    if coordinate_noise_std > 0:
        noise = torch.randn_like(batch["channel_xyz"]) * float(coordinate_noise_std)
        output["channel_xyz"] = batch["channel_xyz"] + noise * keep.unsqueeze(-1)
    if noise_std > 0:
        output["eeg"] = output["eeg"] + torch.randn_like(output["eeg"]) * float(noise_std) * keep.unsqueeze(-1)
    if time_mask_fraction > 0:
        eeg = output["eeg"].clone()
        for index in range(len(eeg)):
            valid_time = int(batch["time_mask"][index].sum())
            width = min(max(1, round(valid_time * float(time_mask_fraction))), max(valid_time, 1))
            start = int(torch.randint(max(1, valid_time - width + 1), (1,), device=eeg.device)[0])
            eeg[index, :, start : start + width] = 0.0
        output["eeg"] = eeg
    return output


__all__ = [
    "AudioCodeBank",
    "COMMON_CHANNELS",
    "DATASETS",
    "DATASET_IDS",
    "MontageRegistry",
    "LabelFreeAudioDataset",
    "OpenVoiceContext",
    "OpenVoiceEEGDataset",
    "TeacherBank",
    "build_montage_registry_payload",
    "collate_openvoice",
    "common_montage_view",
    "load_context",
    "normalize_label",
    "pairing_level",
    "parse_asa_elc",
    "resolve_config_path",
    "stochastic_channel_view",
]
