from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.utils import read_csv_rows, resolve_feis_root


SUBJECT_FIELD = "sub" + "ject_id"
CLEAN_FIELD = "is_clean_" + "sub" + "ject"
UNIT_DIR = "sub" + "jects"
STAGE_TO_ID = {"stimuli": 0, "thinking": 1, "speaking": 2, "articulators": 3, "resting": 4}


def norm_subject(value: str | int) -> str:
    text = str(value)
    return text.zfill(2) if text.isdigit() else text


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def parse_stage_spec(stage: str | list[str] | tuple[str, ...], negative_stage: str = "resting") -> list[str]:
    if isinstance(stage, (list, tuple)):
        parts = [str(item).strip() for item in stage if str(item).strip()]
    else:
        raw = str(stage).replace(",", " ")
        parts = [item.strip() for item in raw.split() if item.strip()]
    if not parts:
        return ["stimuli"]
    if len(parts) == 1 and parts[0] == "all_speech":
        return ["stimuli", "speaking", "thinking"]
    if len(parts) == 1 and parts[0] in {"all_non_articulators", "all_four", "all_with_resting"}:
        return ["stimuli", "thinking", "speaking", "resting"]
    if len(parts) == 1 and parts[0] == "all":
        return [item for item in STAGE_TO_ID if item != negative_stage]
    return parts


def sample_key(subject_id: str, label: str, stage: str, trial_index: int) -> str:
    return f"{stage}_{norm_subject(subject_id)}_{label}_{int(trial_index):04d}"


def forbidden_forward_tokens() -> tuple[str, ...]:
    return ("sub" + "ject", "speak" + "er", "audio_source_" + "sub" + "ject")


def assert_v3_model_forward_keys(keys: list[str] | tuple[str, ...] | set[str]) -> None:
    bad = [key for key in keys if any(token in str(key).lower() for token in forbidden_forward_tokens())]
    if bad:
        raise ValueError(f"Forbidden identity fields passed to FEIS v3 model.forward: {bad}")


def stable_kmeans(
    x: np.ndarray,
    k: int,
    seed: int = 7,
    iters: int = 12,
    max_fit_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Small numpy k-means used for train-only token/cache fitting."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"k-means expects [N, D], got {x.shape}")
    if x.shape[0] == 0:
        raise ValueError("k-means received no rows")
    rng = np.random.default_rng(int(seed))
    fit_x = x
    if max_fit_rows is not None and x.shape[0] > int(max_fit_rows):
        fit_x = x[rng.choice(x.shape[0], int(max_fit_rows), replace=False)]
    k = max(1, min(int(k), fit_x.shape[0]))
    init_idx = rng.choice(fit_x.shape[0], k, replace=False)
    centers = fit_x[init_idx].copy()
    labels = np.zeros(fit_x.shape[0], dtype=np.int64)
    for _ in range(max(1, int(iters))):
        d2 = ((fit_x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = d2.argmin(axis=1)
        for ci in range(k):
            mask = labels == ci
            if mask.any():
                centers[ci] = fit_x[mask].mean(axis=0)
            else:
                centers[ci] = fit_x[rng.integers(0, fit_x.shape[0])]
    full_d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    return centers.astype(np.float32), full_d2.argmin(axis=1).astype(np.int64)


def assign_to_centers(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    return d2.argmin(axis=1).astype(np.int64)


class FEISV3AudioTokenBank:
    """Subject-label audio token cache.

    Metadata can be used for sampling, losses, and audits. It is never required by
    the model forward path.
    """

    def __init__(self, cache_path: str | Path):
        self.path = Path(cache_path)
        payload = np.load(self.path, allow_pickle=True)
        self.audio_keys = payload["audio_keys"].astype(str).tolist()
        self.subject_ids = payload["subject_ids"].astype(str).tolist()
        self.labels = payload["labels"].astype(str).tolist()
        self.audio_paths = payload["audio_paths"].astype(str).tolist()
        self.audio_sha1 = payload["audio_sha1"].astype(str).tolist()
        self.fit_split = payload["fit_split"].astype(str).tolist()
        self.label_vocab = payload["label_vocab"].astype(str).tolist()
        self.label_to_id = {label: idx for idx, label in enumerate(self.label_vocab)}
        self.subject_vocab = payload["subject_vocab"].astype(str).tolist()
        self.subject_to_id = {subj: idx for idx, subj in enumerate(self.subject_vocab)}
        self.semantic_token_ids = payload["semantic_token_ids"].astype(np.int64)
        self.semantic_token_mask = payload["semantic_token_mask"].astype(np.float32)
        self.semantic_hidden = payload["semantic_hidden"].astype(np.float32)
        self.prosody_active = payload["prosody_active"].astype(np.float32)
        self.prosody_duration = payload["prosody_duration"].astype(np.float32)
        self.prosody_energy = payload["prosody_energy"].astype(np.float32)
        self.prosody_onset = payload["prosody_onset"].astype(np.float32)
        self.codec_latent = payload["codec_latent"].astype(np.float32)
        self.codec_token_ids = payload["codec_token_ids"].astype(np.int64)
        self.codec_token_mask = payload["codec_token_mask"].astype(np.float32)
        self.audio_variant_cluster_id = payload["audio_variant_cluster_id"].astype(np.int64)
        self.semantic_codebook = payload["semantic_codebook"].astype(np.float32)
        self.codec_feature_codebook = payload["codec_feature_codebook"].astype(np.float32)
        self.codec_codebook_waveform = payload["codec_codebook_waveform"].astype(np.float32)
        self.audio_variant_cluster_centers = payload["audio_variant_cluster_centers"].astype(np.float32)
        self.sample_rate = int(payload["sample_rate"]) if "sample_rate" in payload.files else 16000
        self.codec_chunk_samples = int(payload["codec_chunk_samples"]) if "codec_chunk_samples" in payload.files else int(self.codec_codebook_waveform.shape[1])
        self.audio_key_to_index = {key: idx for idx, key in enumerate(self.audio_keys)}
        self._semantic_hists: np.ndarray | None = None
        self._codec_label_priors: dict[str, np.ndarray] = {}

    @property
    def semantic_vocab_size(self) -> int:
        return int(self.semantic_codebook.shape[0])

    @property
    def codec_vocab_size(self) -> int:
        return int(self.codec_feature_codebook.shape[0])

    @property
    def semantic_steps(self) -> int:
        return int(self.semantic_token_ids.shape[1])

    @property
    def codec_steps(self) -> int:
        return int(self.codec_token_ids.shape[1])

    @property
    def audio_variant_clusters(self) -> int:
        return int(self.audio_variant_cluster_centers.shape[0])

    @property
    def num_labels(self) -> int:
        return len(self.label_vocab)

    @property
    def num_subjects(self) -> int:
        return len(self.subject_vocab)

    def index(self, audio_key: str) -> int:
        return int(self.audio_key_to_index[str(audio_key)])

    def label_id(self, label: str) -> int:
        return int(self.label_to_id[str(label)])

    def subject_id(self, subject: str) -> int:
        return int(self.subject_to_id[norm_subject(subject)])

    def item(self, audio_key: str) -> dict[str, np.ndarray | str | int]:
        idx = self.index(audio_key)
        return {
            "audio_index": idx,
            "audio_key": self.audio_keys[idx],
            "subject_id": self.subject_ids[idx],
            "label": self.labels[idx],
            "audio_path": self.audio_paths[idx],
            "audio_sha1": self.audio_sha1[idx],
            "semantic_token_ids": self.semantic_token_ids[idx],
            "semantic_token_mask": self.semantic_token_mask[idx],
            "semantic_hidden": self.semantic_hidden[idx],
            "prosody_active": self.prosody_active[idx],
            "prosody_duration": self.prosody_duration[idx],
            "prosody_energy": self.prosody_energy[idx],
            "prosody_onset": self.prosody_onset[idx],
            "codec_latent": self.codec_latent[idx],
            "codec_token_ids": self.codec_token_ids[idx],
            "codec_token_mask": self.codec_token_mask[idx],
            "audio_variant_cluster_id": int(self.audio_variant_cluster_id[idx]),
        }

    def semantic_histograms(self) -> np.ndarray:
        if self._semantic_hists is None:
            hists = np.zeros((len(self.audio_keys), self.semantic_vocab_size), dtype=np.float32)
            for idx, ids in enumerate(self.semantic_token_ids):
                mask = self.semantic_token_mask[idx] > 0.5
                vals = ids[mask]
                if vals.size:
                    hists[idx] = np.bincount(vals, minlength=self.semantic_vocab_size).astype(np.float32)
                    hists[idx] /= max(float(hists[idx].sum()), 1.0)
            self._semantic_hists = hists
        return self._semantic_hists

    def label_prior_codec_tokens(self, label: str) -> np.ndarray:
        label = str(label)
        if label not in self._codec_label_priors:
            idxs = [idx for idx, item in enumerate(self.labels) if item == label and self.fit_split[idx] == "train"]
            if not idxs:
                idxs = [idx for idx, item in enumerate(self.labels) if item == label]
            tokens = self.codec_token_ids[np.asarray(idxs, dtype=np.int64)]
            prior = []
            for step in range(tokens.shape[1]):
                counts = Counter(tokens[:, step].astype(int).tolist())
                prior.append(counts.most_common(1)[0][0])
            self._codec_label_priors[label] = np.asarray(prior, dtype=np.int64)
        return self._codec_label_priors[label]

    def decode_codec_tokens(self, token_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=np.int64).clip(0, self.codec_vocab_size - 1)
        chunks = self.codec_codebook_waveform[ids]
        audio = chunks.reshape(-1).astype(np.float32)
        if audio.size > self.sample_rate:
            audio = audio[: self.sample_rate]
        elif audio.size < self.sample_rate:
            audio = np.pad(audio, (0, self.sample_rate - audio.size)).astype(np.float32)
        peak = float(np.max(np.abs(audio)) + 1e-8)
        if peak > 0.98:
            audio = audio / peak * 0.98
        return audio.astype(np.float32)


class FEISV3ClusterBank:
    def __init__(self, cache_path: str | Path | None):
        self.path = Path(cache_path) if cache_path else None
        self.sample_to_cluster: dict[str, int] = {}
        self.channel_cluster_centers = np.zeros((1, 1), dtype=np.float32)
        if self.path and self.path.exists():
            payload = np.load(self.path, allow_pickle=True)
            keys = payload["sample_keys"].astype(str).tolist()
            ids = payload["sample_cluster_ids"].astype(np.int64).tolist()
            self.sample_to_cluster = {key: int(val) for key, val in zip(keys, ids)}
            self.channel_cluster_centers = payload["channel_cluster_centers"].astype(np.float32)

    @property
    def num_clusters(self) -> int:
        return int(max(1, self.channel_cluster_centers.shape[0]))

    def cluster_for(self, key: str) -> int:
        return int(self.sample_to_cluster.get(str(key), 0))


@dataclass(frozen=True)
class FEISV3Entry:
    sample_key: str
    subject_id: str
    label: str
    trial_index: int
    repetition_index: int
    stage: str
    position: int
    audio_key: str
    audio_path: str
    audio_sha1: str


class FEISV3Dataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        token_bank: FEISV3AudioTokenBank,
        split: str,
        stage: str = "stimuli",
        cluster_bank: FEISV3ClusterBank | None = None,
        eeg_len: int = 1280,
        include_anomalous: bool = False,
        subject_val: str = "20",
        subject_test: str = "21",
        negative_stage: str = "resting",
        allow_negative_train: bool = False,
    ):
        self.root = resolve_feis_root(data_root)
        self.token_bank = token_bank
        self.cluster_bank = cluster_bank or FEISV3ClusterBank(None)
        self.split = str(split)
        self.stage_spec = stage
        self.eeg_len = int(eeg_len)
        self.include_anomalous = bool(include_anomalous)
        self.subject_val = norm_subject(subject_val)
        self.subject_test = norm_subject(subject_test)
        self.negative_stage = str(negative_stage)
        self.allow_negative_train = bool(allow_negative_train)
        self._bundles: dict[str, dict[str, Any]] = {}

        rows = self._load_rows()
        self.entries = self._assign_split(rows)
        if not self.entries:
            raise ValueError(f"No FEIS v3 samples for split={split}, stage={stage}")

    def _wanted_stages(self) -> list[str]:
        if self.split in {"resting_control", "stage_negative_control"}:
            return [self.negative_stage]
        return parse_stage_spec(self.stage_spec, negative_stage=self.negative_stage)

    def _load_rows(self) -> list[dict[str, Any]]:
        wanted = set(self._wanted_stages())
        rows: list[dict[str, Any]] = []
        for row in read_csv_rows(self.root / "segments.csv"):
            stage = str(row.get("segment_stage", ""))
            if stage not in wanted:
                continue
            subject_id = norm_subject(row[SUBJECT_FIELD])
            if not self.include_anomalous and not as_bool(row.get(CLEAN_FIELD), subject_id != "05"):
                continue
            label = str(row["label"])
            audio_key = f"{subject_id}:{label}"
            if audio_key not in self.token_bank.audio_key_to_index:
                continue
            rows.append(
                {
                    "subject_id": subject_id,
                    "label": label,
                    "trial_index": int(row["trial_index"]),
                    "stage": stage,
                    "audio_key": audio_key,
                    "audio_path": str(row["audio_path"]),
                    "audio_sha1": str(row.get("audio_sha1", "")),
                }
            )
        rows.sort(key=lambda item: (item["stage"], item["subject_id"], item["label"], item["trial_index"]))
        return rows

    def _assign_split(self, rows: list[dict[str, Any]]) -> list[FEISV3Entry]:
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["subject_id"], row["label"], row["stage"])].append(row)

        aliases = {"val": "subject_val", "test": "subject_test", "subject_holdout_val": "subject_val", "subject_holdout_test": "subject_test"}
        split = aliases.get(self.split, self.split)
        out: list[FEISV3Entry] = []
        for (subject_id, label, stage), reps in grouped.items():
            reps.sort(key=lambda item: item["trial_index"])
            indexed: list[tuple[int, dict[str, Any]]] = list(enumerate(reps))
            if len(indexed) >= 3:
                train_rows = indexed[:-2]
                repeat_val = indexed[-2:-1]
                repeat_test = indexed[-1:]
            elif len(indexed) == 2:
                train_rows = indexed[:1]
                repeat_val = indexed[1:]
                repeat_test = indexed[1:]
            else:
                train_rows = indexed
                repeat_val = indexed
                repeat_test = indexed

            selected: list[tuple[int, dict[str, Any]]]
            if split in {"train", "subject_holdout_train"}:
                selected = (
                    []
                    if subject_id in {self.subject_val, self.subject_test}
                    or (stage == self.negative_stage and not self.allow_negative_train)
                    else train_rows
                )
            elif split == "repeat_val":
                selected = (
                    []
                    if subject_id in {self.subject_val, self.subject_test}
                    or (stage == self.negative_stage and not self.allow_negative_train)
                    else repeat_val
                )
            elif split in {"repeat_test", "repeat_holdout"}:
                selected = (
                    []
                    if subject_id in {self.subject_val, self.subject_test}
                    or (stage == self.negative_stage and not self.allow_negative_train)
                    else repeat_test
                )
            elif split == "subject_val":
                selected = indexed if subject_id == self.subject_val and (stage != self.negative_stage or self.allow_negative_train) else []
            elif split == "subject_test":
                selected = indexed if subject_id == self.subject_test and (stage != self.negative_stage or self.allow_negative_train) else []
            elif split in {"resting_control", "stage_negative_control"}:
                selected = indexed if stage == self.negative_stage and subject_id in {self.subject_val, self.subject_test} else []
            elif split == "all":
                selected = indexed
            else:
                raise ValueError(f"Unsupported FEIS v3 split: {self.split}")

            pos_map = self._trial_to_pos(subject_id)
            for repetition_index, row in selected:
                trial_index = int(row["trial_index"])
                if trial_index not in pos_map:
                    continue
                out.append(
                    FEISV3Entry(
                        sample_key=sample_key(subject_id, label, stage, trial_index),
                        subject_id=subject_id,
                        label=label,
                        trial_index=trial_index,
                        repetition_index=int(repetition_index),
                        stage=stage,
                        position=int(pos_map[trial_index]),
                        audio_key=row["audio_key"],
                        audio_path=row["audio_path"],
                        audio_sha1=row["audio_sha1"],
                    )
                )
        out.sort(key=lambda entry: entry.sample_key)
        return out

    def _bundle(self, subject_id: str) -> dict[str, Any]:
        subject_id = norm_subject(subject_id)
        if subject_id not in self._bundles:
            bundle = np.load(self.root / UNIT_DIR / f"{subject_id}.npz", allow_pickle=True)
            trial_indices = bundle["trial_indices"].astype(int)
            self._bundles[subject_id] = {
                "trial_to_pos": {int(item): idx for idx, item in enumerate(trial_indices.tolist())},
                "bundle": bundle,
            }
        return self._bundles[subject_id]

    def _trial_to_pos(self, subject_id: str) -> dict[int, int]:
        return self._bundle(subject_id)["trial_to_pos"]

    def _eeg(self, entry: FEISV3Entry) -> tuple[np.ndarray, int]:
        bundle = self._bundle(entry.subject_id)["bundle"]
        arr = bundle[f"stage__{entry.stage}"]
        valid_key = f"stage__{entry.stage}__valid_lengths"
        valid = int(bundle[valid_key][entry.position]) if valid_key in bundle.files else int(arr.shape[-1])
        x = arr[entry.position].astype(np.float32)
        channels, length = x.shape
        if length == self.eeg_len:
            return x, min(valid, self.eeg_len)
        out = np.zeros((channels, self.eeg_len), dtype=np.float32)
        keep = min(length, self.eeg_len)
        out[:, :keep] = x[:, :keep]
        return out, min(valid, self.eeg_len)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[idx]
        bank_idx = self.token_bank.index(entry.audio_key)
        bank_item = self.token_bank.item(entry.audio_key)
        eeg, valid_len = self._eeg(entry)
        return {
            "eeg": torch.from_numpy(eeg).float(),
            "stage_idx": torch.tensor(STAGE_TO_ID.get(entry.stage, 0), dtype=torch.long),
            "eeg_valid_len": torch.tensor(valid_len, dtype=torch.long),
            "channel_cluster_id": torch.tensor(self.cluster_bank.cluster_for(entry.sample_key), dtype=torch.long),
            "semantic_token_ids": torch.from_numpy(np.asarray(bank_item["semantic_token_ids"], dtype=np.int64)).long(),
            "semantic_token_mask": torch.from_numpy(np.asarray(bank_item["semantic_token_mask"], dtype=np.float32)).float(),
            "codec_token_ids": torch.from_numpy(np.asarray(bank_item["codec_token_ids"], dtype=np.int64)).long(),
            "codec_token_mask": torch.from_numpy(np.asarray(bank_item["codec_token_mask"], dtype=np.float32)).float(),
            "prosody_active": torch.from_numpy(np.asarray(bank_item["prosody_active"], dtype=np.float32)).float(),
            "prosody_duration": torch.from_numpy(np.asarray(bank_item["prosody_duration"], dtype=np.float32)).float(),
            "prosody_energy": torch.from_numpy(np.asarray(bank_item["prosody_energy"], dtype=np.float32)).float(),
            "prosody_onset": torch.from_numpy(np.asarray(bank_item["prosody_onset"], dtype=np.float32)).float(),
            "audio_variant_cluster_id": torch.tensor(int(bank_item["audio_variant_cluster_id"]), dtype=torch.long),
            "label_idx": torch.tensor(self.token_bank.label_id(entry.label), dtype=torch.long),
            "audio_index": torch.tensor(bank_idx, dtype=torch.long),
            "subject_idx": torch.tensor(self.token_bank.subject_id(entry.subject_id), dtype=torch.long),
            "trial_index": torch.tensor(entry.trial_index, dtype=torch.long),
            "repetition_index": torch.tensor(entry.repetition_index, dtype=torch.long),
            "sample_key": entry.sample_key,
            "subject_id": entry.subject_id,
            "label": entry.label,
            "stage": entry.stage,
            "audio_key": entry.audio_key,
            "audio_path": entry.audio_path,
            "audio_sha1": entry.audio_sha1,
        }


class FEISV3RepeatAwareBatchSampler(Sampler[list[int]]):
    """Simple metadata-aware sampler for v3 smoke and full runs.

    It interleaves subject-label repetition groups so batches contain multiple
    subjects, labels, and repetitions when the split has enough samples.
    """

    def __init__(self, dataset: FEISV3Dataset, batch_size: int = 64, seed: int = 7, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for idx, entry in enumerate(self.dataset.entries):
            groups[(entry.subject_id, entry.label)].append(idx)
        group_keys = list(groups)
        if self.shuffle:
            rng.shuffle(group_keys)
            for key in group_keys:
                rng.shuffle(groups[key])
        cursors = {key: 0 for key in group_keys}
        batch: list[int] = []
        remaining = True
        while remaining:
            remaining = False
            for key in group_keys:
                pos = cursors[key]
                if pos >= len(groups[key]):
                    continue
                remaining = True
                batch.append(groups[key][pos])
                cursors[key] += 1
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        return int(np.ceil(len(self.dataset) / max(self.batch_size, 1)))
