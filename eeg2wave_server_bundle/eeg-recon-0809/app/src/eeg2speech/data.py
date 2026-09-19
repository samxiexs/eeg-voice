"""Manifest-backed datasets and alternating homogeneous-batch scheduling."""
from __future__ import annotations

import json
import hashlib
import itertools
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler


DATASET_IDS = {"ds004940": 0}


def _decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _stable_key(value: str) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _complete_grid(frame: pd.DataFrame, subject_count: int, content_count: int,
                   namespace: str) -> pd.DataFrame:
    """Choose a deterministic complete subject x content grid."""
    subjects = sorted(frame.subject.unique())
    if len(subjects) < subject_count:
        raise RuntimeError(f"{namespace}: need {subject_count} subjects, found {len(subjects)}")
    best: tuple[tuple[int, int, str], tuple[str, ...], list[str]] | None = None
    for combination in itertools.combinations(subjects, subject_count):
        selected = frame[frame.subject.isin(combination) & (frame.linguistic_content_id != "")]
        coverage = selected.groupby("linguistic_content_id").subject.nunique()
        common = sorted(coverage[coverage == subject_count].index)
        if len(common) < content_count:
            continue
        score = (len(common), len(selected), _stable_key(f"{namespace}|{'|'.join(combination)}"))
        if best is None or score[:2] > best[0][:2] or (score[:2] == best[0][:2] and score[2] < best[0][2]):
            best = (score, combination, common)
    if best is None:
        raise RuntimeError(f"{namespace}: no {subject_count}-subject set shares {content_count} contents")
    _, subjects_selected, common = best
    common.sort(key=lambda value: _stable_key(f"{namespace}|content|{value}"))
    contents_selected = set(common[:content_count])
    selected = frame[frame.subject.isin(subjects_selected) & frame.linguistic_content_id.isin(contents_selected)].copy()
    selected["_trial_order"] = selected.trial_id.map(lambda value: _stable_key(f"{namespace}|trial|{value}"))
    selected = selected.sort_values("_trial_order").drop_duplicates(["subject", "linguistic_content_id"])
    expected = subject_count * content_count
    if len(selected) != expected:
        raise RuntimeError(f"{namespace}: selected grid has {len(selected)} rather than {expected} cells")
    return selected.drop(columns="_trial_order")


def pilot_indices(dataset: "JointManifestDataset", config: dict, stage: str,
                  role: str = "train") -> list[int]:
    """Select the preregistered M0/M1 subset and enforce its cardinality.

    M0 is deliberately a complete subject × content grid.  Failing loudly here
    prevents older Stage-0 shards from silently changing the overfit task.
    """
    frame = dataset.frame.copy()
    pilot = config["pilot"]
    if frame.dataset.iloc[0] == "ds004940" and pilot.get("primary_ds004940_task"):
        frame = frame[frame.task == str(pilot["primary_ds004940_task"])]
        if not len(frame):
            raise RuntimeError(f"no rows remain for primary DS004940 task {pilot['primary_ds004940_task']}")
    if stage == "overfit":
        if role != "train":
            raise ValueError("the M0 overfit subset must come from the train role")
        subject_count = int(pilot["overfit_subjects_per_dataset"])
        content_count = int(pilot["overfit_contents_per_dataset"])
        pair_count = int(pilot["overfit_pairs_per_dataset"])
        frame = _complete_grid(frame, subject_count, content_count, f"M0|{frame.dataset.iloc[0]}")
        grid = frame.groupby(["subject", "linguistic_content_id"]).size()
        if (len(frame), frame.subject.nunique(), frame.linguistic_content_id.nunique()) != (pair_count, subject_count, content_count):
            raise RuntimeError("M0 selection does not satisfy the configured pair/subject/content counts")
        if pair_count == subject_count * content_count and (len(grid) != pair_count or not grid.eq(1).all()):
            raise RuntimeError("M0 selection is not a complete one-trial-per-subject×content grid")
    else:
        subject_roles = pilot["generalization_subjects_by_role"]
        content_roles = pilot["generalization_contents_by_role"]
        if sum(int(value) for value in subject_roles.values()) != int(pilot["generalization_subjects_per_dataset"]):
            raise RuntimeError("M1 subject role counts do not sum to generalization_subjects_per_dataset")
        if sum(int(value) for value in content_roles.values()) != int(pilot["generalization_contents_per_dataset"]):
            raise RuntimeError("M1 content role counts do not sum to generalization_contents_per_dataset")
        if role not in subject_roles or role not in content_roles:
            raise ValueError(f"unknown M1 role {role}")
        frame = _complete_grid(
            frame, int(subject_roles[role]), int(content_roles[role]),
            f"M1|{frame.dataset.iloc[0]}|{role}",
        )
        maximum = int(pilot[f"max_{role}_trials_per_dataset"])
        if len(frame) > maximum:
            raise RuntimeError(f"M1 balanced grid has {len(frame)} trials but max_{role}_trials_per_dataset={maximum}")
    return frame.index.astype(int).tolist()


def auxiliary_indices(dataset: "JointManifestDataset", config: dict, stage: str) -> list[int]:
    maximum = int(config["pilot"][f"label_only_max_{'overfit' if stage == 'overfit' else 'generalization'}_pairs"])
    frame = dataset.frame.copy()
    frame["_trial_order"] = frame.trial_id.map(lambda value: _stable_key(f"label-only|{stage}|{value}"))
    frame = frame.sort_values("_trial_order")
    frame["_repeat"] = frame.groupby(["subject", "linguistic_content_id"]).cumcount()
    frame = frame.sort_values(["_repeat", "_trial_order"]).head(maximum)
    if not len(frame) or not frame.phoneme_label.astype(str).ne("").all():
        raise RuntimeError("label-only pilot selection has no valid phoneme labels")
    return frame.index.astype(int).tolist()


def phoneme_vocabulary_from_manifest(manifest_path: Path) -> dict[str, int]:
    frame = pd.read_csv(manifest_path, keep_default_na=False, low_memory=False)
    labels = sorted({str(value) for value in frame.get("phoneme_label", []) if str(value)})
    return {label: index for index, label in enumerate(labels)}


class ContentGroupedBatchSampler(Sampler[list[int]]):
    """Emit ``contents_per_batch × subjects_per_content`` homogeneous batches.

    The sampler is for a complete subject×content grid.  It visits every
    selected record exactly once per epoch and makes cross-subject observations
    of the same linguistic content available as positives in each batch.
    """

    def __init__(self, frame: pd.DataFrame, indices: list[int], *, batch_size: int,
                 contents_per_batch: int = 4, subjects_per_content: int = 2, seed: int = 31):
        if batch_size != contents_per_batch * subjects_per_content:
            raise ValueError("content_grouped batch_size must equal contents_per_batch × subjects_per_content")
        self.contents_per_batch = int(contents_per_batch)
        self.subjects_per_content = int(subjects_per_content)
        selected = frame.iloc[indices].copy()
        # DataLoader wraps the original dataset in ``Subset(dataset, indices)``.
        # A batch sampler must therefore emit positions inside that Subset,
        # not the original manifest indices (which may be sparse/nonzero).
        selected["_subset_index"] = list(range(len(indices)))
        by_content: dict[str, list[list[int]]] = {}
        for _, group in selected.groupby("linguistic_content_id", sort=False):
            ordered = group.sort_values(["subject", "_subset_index"])._subset_index.astype(int).tolist()
            if len(ordered) % self.subjects_per_content:
                raise RuntimeError("content_grouped sampler requires a complete subjects-per-content grid")
            by_content[str(group.iloc[0].linguistic_content_id)] = [ordered[offset:offset + self.subjects_per_content]
                                                                      for offset in range(0, len(ordered), self.subjects_per_content)]
        rounds = {len(value) for value in by_content.values()}
        if len(rounds) != 1:
            raise RuntimeError("content_grouped sampler requires equal subject coverage for every content")
        self.rounds = [[chunks[round_index] for chunks in by_content.values()]
                       for round_index in range(next(iter(rounds), 0))]
        if not self.rounds or len(self.rounds[0]) % self.contents_per_batch:
            raise RuntimeError("content_grouped sampler cannot form complete batches")
        self.seed = int(seed); self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        for groups in self.rounds:
            order = torch.randperm(len(groups), generator=generator).tolist()
            for offset in range(0, len(order), self.contents_per_batch):
                yield [index for item in order[offset:offset + self.contents_per_batch] for index in groups[item]]

    def __len__(self) -> int:
        return sum(len(groups) // self.contents_per_batch for groups in self.rounds)


class JointManifestDataset(Dataset):
    def __init__(self, manifest_path: Path, split_path: Path, role: str, dataset: str,
                 speech_targets: Path | None = None, normalizer_path: Path | None = None,
                 weak_content_weight: float = 0.35, limit: int | None = None,
                 supervision_types: set[str] | None = None,
                 phoneme_vocabulary: dict[str, int] | None = None):
        frame = pd.read_csv(manifest_path, keep_default_na=False, low_memory=False)
        split = pd.read_csv(split_path, keep_default_na=False)
        selected_ids = set(split[split.role == role].trial_id)
        frame = frame[(frame.trial_id.isin(selected_ids)) & (frame.build_status == "included")]
        if dataset != "all":
            frame = frame[frame.dataset == dataset]
        if supervision_types is not None:
            frame = frame[frame.supervision_type.isin(supervision_types)]
        frame = frame.sort_values(["dataset", "subject", "linguistic_content_id", "trial_id"])
        if limit is not None:
            frame = frame.head(limit)
        self.frame = frame.reset_index(drop=True)
        if not len(self.frame):
            raise ValueError(f"no rows for role={role}, dataset={dataset}")
        self.weak_content_weight = float(weak_content_weight)
        self.shards: dict[str, h5py.File] = {}
        content_rows = self.frame[self.frame.supervision_type.isin(["paired_audio", "weak_audio"])]
        if len(content_rows) and (speech_targets is None or not speech_targets.exists()):
            raise RuntimeError("content-supervised dataset requires an existing speech-target cache")
        self.targets = h5py.File(speech_targets, "r") if speech_targets and speech_targets.exists() else None
        if self.targets is not None and len(content_rows):
            config_hashes = set(content_rows.preprocess_config_sha256.astype(str))
            if len(config_hashes) != 1 or self.targets.attrs.get("preprocess_config_sha256", "") not in config_hashes:
                raise RuntimeError("speech-target/preprocessing config provenance mismatch")
            missing = sorted({self._audio_id(row) for _, row in content_rows.iterrows() if self._audio_id(row) not in self.targets})
            if missing:
                raise RuntimeError(f"content-supervised rows are missing speech targets: {missing[:5]} (n={len(missing)})")
            for audio_id in sorted({self._audio_id(row) for _, row in content_rows.iterrows()}):
                group = self.targets[audio_id]
                for required in ("content_mfcc", "content_mask", "log_mel", "rms", "activity"):
                    if required not in group:
                        raise RuntimeError(f"speech target {audio_id} is missing {required}")
                mfcc = group["content_mfcc"][:]
                mask = group["content_mask"][:].astype(bool)
                if mfcc.shape != (39, 161) or mask.shape != (161,) or not mask.any() or not np.isfinite(mfcc).all():
                    raise RuntimeError(f"speech target {audio_id} has an invalid MFCC contract")
                if bool(self.targets.attrs.get("hubert_included", False)):
                    if "hubert_local" not in group or group["hubert_local"].shape != (96, 768):
                        raise RuntimeError(f"speech target {audio_id} has an invalid HuBERT contract")
                    if "hubert_global" not in group or group["hubert_global"].shape != (768,):
                        raise RuntimeError(f"speech target {audio_id} has an invalid HuBERT global contract")
                if "native_mel_contract" in self.targets.attrs:
                    if "native_speecht5_mel" not in group or "native_audio_mask" not in group:
                        raise RuntimeError(f"speech target {audio_id} is missing native SpeechT5 mel")
                    native = group["native_speecht5_mel"][:]
                    native_mask = group["native_audio_mask"][:].astype(bool)
                    if native.ndim != 2 or native.shape[0] != 80 or native.shape[1] != native_mask.shape[0] or not native_mask.any():
                        raise RuntimeError(f"speech target {audio_id} has an invalid native SpeechT5 mel contract")
        if normalizer_path is None or not normalizer_path.exists():
            raise RuntimeError("an existing train-fold normalizer is required")
        normalizer_payload = json.loads(normalizer_path.read_text())
        if normalizer_payload.get("split_csv_sha256") != _file_sha256(split_path):
            raise RuntimeError("normalizer was not fitted on the requested split CSV")
        self.normalizers = normalizer_payload.get("datasets", {})
        self.preprocessing_contract = normalizer_payload.get("preprocessing_contract", {})
        if not self.preprocessing_contract:
            raise RuntimeError("normalizer is missing its preprocessing contract")
        if self.targets is not None:
            for key in ("preprocess_config_sha256", "source_lock_sha256"):
                if str(self.targets.attrs.get(key, "")) != str(self.preprocessing_contract.get(key, "")):
                    raise RuntimeError(f"speech-target/normalizer preprocessing contract mismatch for {key}")
        labels = sorted({str(value) for value in frame.phoneme_label if str(value)}) if "phoneme_label" in frame else []
        self.phoneme_vocabulary = phoneme_vocabulary or {label: index for index, label in enumerate(labels)}
        unknown_labels = sorted(set(labels) - set(self.phoneme_vocabulary))
        if unknown_labels:
            raise RuntimeError(f"phoneme vocabulary is missing labels: {unknown_labels}")
        # This is a training-only diagnostic target.  It is never exposed to
        # the EEG model as an input and held-out subjects simply receive an
        # index that was not observed by the adversarial head during fitting.
        self.subject_vocabulary = {
            value: index for index, value in enumerate(sorted({str(value) for value in self.frame.subject}))
        }

    def __len__(self) -> int:
        return len(self.frame)

    def _shard(self, relative: str) -> h5py.File:
        if relative not in self.shards:
            shard = h5py.File(relative, "r")
            for key, expected in self.preprocessing_contract.items():
                if str(shard.attrs.get(key, "")) != str(expected):
                    shard.close()
                    raise RuntimeError(f"shard/normalizer preprocessing contract mismatch for {key}: {relative}")
            self.shards[relative] = shard
        return self.shards[relative]

    def _audio_id(self, row) -> str:
        value = str(row.get("audio_id", ""))
        return value or (f"audio-{row.audio_sha256[:16]}-{row.audio_semantics}" if row.audio_sha256 else "")

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        shard_path = str(row.shard_path)
        if not Path(shard_path).is_absolute():
            # Manifest paths are repository-relative; resolve from eeg-recon-0809.
            candidate = Path(__file__).resolve().parents[3] / shard_path
            shard_path = str(candidate if candidate.exists() else Path(shard_path))
        shard = self._shard(shard_path)
        # Empty shard rows on excluded records make pandas infer a floating
        # column after CSV round-trips; included records are still integral.
        shard_row = int(float(row.shard_row))
        eeg = shard["eeg"][shard_row].astype("float32")
        channel_mask = shard["channel_valid_mask"][shard_row].astype(bool)
        time_mask = shard["eeg_valid_mask"][shard_row].astype(bool)
        model_time_mask = (np.ones_like(time_mask, dtype=bool)
                           if str(shard.attrs.get("model_time_mask_policy", "")) == "fixed_full_epoch"
                           else time_mask.copy())
        normalizer = self.normalizers.get(row.dataset)
        if normalizer:
            center = np.asarray(normalizer["center_median_v"], dtype=np.float32)[:, None]
            scale = np.asarray(normalizer["scale_mad_v"], dtype=np.float32)[:, None]
            eeg = (eeg - center) / np.maximum(scale, 1e-9)
        eeg *= time_mask[None, :]
        eeg *= channel_mask[:, None]
        audio_id = self._audio_id(row)
        content = np.zeros((39, 161), dtype=np.float32)
        content_mask = np.zeros(161, dtype=bool)
        hubert = np.zeros((96, 768), dtype=np.float32)
        hubert_mask = np.zeros(96, dtype=bool)
        hubert_global = np.zeros(768, dtype=np.float32)
        acoustic = np.zeros((80, 161), dtype=np.float32)
        rms = np.zeros(161, dtype=np.float32)
        activity = np.zeros(161, dtype=bool)
        native_mel = np.zeros((80, 1), dtype=np.float32)
        native_mask = np.zeros(1, dtype=bool)
        duration_frames = 0
        if self.targets is not None and audio_id in self.targets:
            target = self.targets[audio_id]
            content = target["content_mfcc"][:].astype("float32")
            content_mask = target["content_mask"][:].astype(bool)
            if "hubert_local" in target:
                hubert = target["hubert_local"][:].astype("float32")
                hubert_mask[:] = True
            if "hubert_global" in target:
                hubert_global = target["hubert_global"][:].astype("float32")
                if hubert_global.shape != (768,) or not np.isfinite(hubert_global).all():
                    raise RuntimeError(f"speech target {audio_id} has an invalid HuBERT global contract")
            acoustic = F.interpolate(torch.from_numpy(target["log_mel"][:].astype("float32")).unsqueeze(0), size=161,
                                     mode="linear", align_corners=False).squeeze(0).numpy()
            rms = F.interpolate(torch.from_numpy(target["rms"][:].astype("float32"))[None, None], size=161,
                                mode="linear", align_corners=False).squeeze().numpy()
            activity = F.interpolate(torch.from_numpy(target["activity"][:].astype("float32"))[None, None], size=161,
                                     mode="nearest").squeeze().bool().numpy()
            if "native_speecht5_mel" in target:
                native_mel = target["native_speecht5_mel"][:].astype("float32")
                native_mask = target["native_audio_mask"][:].astype(bool)
                duration_frames = int(target.attrs.get("native_duration_frames", int(native_mask.sum())))
        pairing = str(row.pairing_level)
        pairing_weight = 1.0 if pairing == "verified_exact" else (self.weak_content_weight if pairing == "candidate_filename_timing" else 0.0)
        if pairing_weight > 0 and (self.targets is None or audio_id not in self.targets):
            raise RuntimeError(f"missing target for supervised trial {row.trial_id}")
        phoneme = str(row.get("phoneme_label", ""))
        return {
            "trial_id": str(row.trial_id), "dataset": str(row.dataset), "dataset_id": DATASET_IDS[str(row.dataset)],
            "subject": str(row.subject), "task": str(row.task), "condition": str(row.condition),
            "subject_index": torch.tensor(self.subject_vocabulary[str(row.subject)], dtype=torch.long),
            "linguistic_content_id": str(row.linguistic_content_id), "pairing_level": pairing,
            "supervision_type": str(row.supervision_type), "audio_id": audio_id,
            "eeg": torch.from_numpy(eeg), "channel_xyz": torch.from_numpy(shard["channel_xyz"][:].astype("float32")),
            "channel_mask": torch.from_numpy(channel_mask),
            "time_mask": torch.from_numpy(time_mask),
            "model_time_mask": torch.from_numpy(model_time_mask),
            "content_mfcc": torch.from_numpy(content), "content_mask": torch.from_numpy(content_mask),
            "hubert_local": torch.from_numpy(hubert), "hubert_global": torch.from_numpy(hubert_global),
            "hubert_mask": torch.from_numpy(hubert_mask),
            "acoustic_log_mel": torch.from_numpy(acoustic), "acoustic_rms": torch.from_numpy(rms),
            "acoustic_activity": torch.from_numpy(activity),
            "native_speecht5_mel": torch.from_numpy(native_mel),
            "native_audio_mask": torch.from_numpy(native_mask),
            "audio_duration_frames": torch.tensor(duration_frames, dtype=torch.long),
            "acoustic_supervision": torch.tensor(pairing == "verified_exact", dtype=torch.bool),
            "pairing_weight": torch.tensor(pairing_weight, dtype=torch.float32),
            "phoneme_index": torch.tensor(self.phoneme_vocabulary.get(phoneme, -1), dtype=torch.long),
        }

    def sampling_weights(self) -> torch.Tensor:
        subjects = Counter(f"{row.dataset}:{row.subject}" for _, row in self.frame.iterrows())
        contents = Counter(str(row.linguistic_content_id) for _, row in self.frame.iterrows() if str(row.linguistic_content_id))
        values = []
        for _, row in self.frame.iterrows():
            subject_weight = 1.0 / subjects[f"{row.dataset}:{row.subject}"]
            content = str(row.linguistic_content_id)
            content_weight = 1.0 / contents[content] if content else 1.0
            values.append((subject_weight * content_weight) ** 0.5)
        return torch.tensor(values, dtype=torch.double)

    def balanced_sampler(self, samples: int | None = None, seed: int = 31) -> WeightedRandomSampler:
        generator = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(self.sampling_weights(), samples or len(self), replacement=True, generator=generator)

    def close(self) -> None:
        for shard in self.shards.values():
            shard.close()
        self.shards.clear()
        if self.targets is not None:
            self.targets.close()
            self.targets = None


def homogeneous_collate(records: list[dict[str, Any]]) -> dict[str, Any]:
    datasets = {record["dataset"] for record in records}
    if len(datasets) != 1:
        raise ValueError("a batch must contain exactly one dataset")
    tensor_keys = ("dataset_id", "subject_index", "eeg", "channel_xyz", "channel_mask", "time_mask", "model_time_mask",
                   "content_mfcc", "content_mask", "hubert_local", "hubert_global", "hubert_mask", "pairing_weight", "phoneme_index",
                   "acoustic_log_mel", "acoustic_rms", "acoustic_activity", "acoustic_supervision",
                   "audio_duration_frames")
    batch = {key: torch.stack([torch.as_tensor(record[key]) for record in records]) for key in tensor_keys}
    # SpeechT5 target mel remains native-duration/ragged.  Padding is purely a
    # batch transport detail and is always accompanied by native_audio_mask.
    maximum = max(int(record["native_speecht5_mel"].shape[-1]) for record in records)
    native = []
    native_mask = []
    for record in records:
        value = torch.as_tensor(record["native_speecht5_mel"])
        mask = torch.as_tensor(record["native_audio_mask"], dtype=torch.bool)
        native.append(F.pad(value, (0, maximum - value.shape[-1])))
        native_mask.append(F.pad(mask, (0, maximum - mask.shape[-1]), value=False))
    batch["native_speecht5_mel"] = torch.stack(native)
    batch["native_audio_mask"] = torch.stack(native_mask)
    for key in ("trial_id", "dataset", "subject", "task", "condition", "linguistic_content_id", "pairing_level",
                "supervision_type", "audio_id"):
        batch[key] = [record[key] for record in records]
    return batch


class AlternatingBatchIterator:
    """Round-robin loaders without combining incompatible channel spaces."""

    def __init__(self, loaders: dict[str, Any], order: list[str] | None = None):
        if not loaders:
            raise ValueError("at least one dataset loader is required")
        self.loaders = loaders
        self.order = order or sorted(loaders)

    def __iter__(self) -> Iterator[tuple[str, dict[str, Any]]]:
        iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        while True:
            for name in self.order:
                try:
                    batch = next(iterators[name])
                except StopIteration:
                    iterators[name] = iter(self.loaders[name])
                    batch = next(iterators[name])
                yield name, batch
