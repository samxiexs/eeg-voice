from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio_features import TARGET_KIND_ENCODEC_LATENT
from .phonemes import build_phoneme_vocab, encode_label_phonemes
from .utils import load_wav_fixed, resolve_feis_root


ABLATION_MODES = {
    "none",
    "random_noise",
    "shuffle_eeg",
    "subject_mean",
    "label_only",
    "subject_only",
    "label_subject",
}


def _normalize_subject_id(subject_id: str | int) -> str:
    text = str(subject_id)
    return text.zfill(2) if text.isdigit() else text


def _as_bool(value: Any, default: bool = False) -> bool:
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


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class TrialEntry:
    trial_index: int
    position: int
    label: str
    audio_relpath: str


@dataclass(frozen=True)
class ProtocolEntry:
    index: int
    subject_id: str
    subject_index: int
    trial_index: int
    position: int
    label: str
    label_id: int
    audio_relpath: str
    audio_path: str
    template_id: str
    split_name: str
    protocol: str
    is_clean_subject: bool
    audio_source_subject: str
    audio_source_kind: str
    unique_hashes_per_subject_label: int
    unique_hashes_per_label_across_subjects: int
    eeg_valid_num_samples: int


class FEISThinkingDataset(Dataset):
    def __init__(
        self,
        subject_id: str,
        data_root: str | Path,
        stage: str = "thinking",
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        audio_sr: int = 16000,
        audio_dur: float = 1.5,
        audio_normalize: str = "rms",
        audio_target_rms: float = 0.08,
        audio_max_gain: float = 12.0,
    ):
        self.subject_id = _normalize_subject_id(subject_id)
        self.stage = str(stage)
        self.split = str(split)
        self.feis_root = resolve_feis_root(data_root)
        self.audio_root = self.feis_root
        self.audio_sr = int(audio_sr)
        self.audio_n_samples = int(round(audio_sr * audio_dur))
        self.audio_normalize = str(audio_normalize)
        self.audio_target_rms = float(audio_target_rms)
        self.audio_max_gain = float(audio_max_gain)
        self._audio_cache: dict[str, np.ndarray] = {}

        bundle_path = self.feis_root / "subjects" / f"{self.subject_id}.npz"
        if not bundle_path.exists():
            raise FileNotFoundError(f"Missing FEIS subject bundle: {bundle_path}")
        bundle = np.load(bundle_path, allow_pickle=True)
        stage_key = f"stage__{self.stage}"
        if stage_key not in bundle.files:
            raise KeyError(f"Stage {self.stage} is not present in {bundle_path.name}")

        self.eeg_array = bundle[stage_key].astype(np.float32)
        self.labels = bundle["labels"].astype(str)
        self.audio_relpaths = bundle["audio_relpaths"].astype(str)
        self.trial_indices = bundle["trial_indices"].astype(np.int32)
        self.channel_names = bundle["channel_names"].astype(str)
        self.trial_to_position = {int(trial_idx): pos for pos, trial_idx in enumerate(self.trial_indices.tolist())}

        entries = self._load_entries()
        self.label_vocab = sorted({entry.label for entry in entries})
        self.label_to_id = {label: idx for idx, label in enumerate(self.label_vocab)}
        self.entries = self._split_entries(entries, split, float(train_ratio), float(val_ratio))
        if not self.entries:
            raise ValueError(f"No FEIS samples left for subject={subject_id}, stage={stage}, split={split}")

    def _load_entries(self) -> list[TrialEntry]:
        rows: list[TrialEntry] = []
        segments_path = self.feis_root / "segments.csv"
        with segments_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row["subject_id"] != self.subject_id or row["segment_stage"] != self.stage:
                    continue
                trial_index = int(row["trial_index"])
                position = self.trial_to_position[trial_index]
                audio_relpath = row.get("audio_path") or self.audio_relpaths[position]
                rows.append(
                    TrialEntry(
                        trial_index=trial_index,
                        position=position,
                        label=str(row["label"]),
                        audio_relpath=str(audio_relpath),
                    )
                )
        rows.sort(key=lambda item: item.trial_index)
        return rows

    @staticmethod
    def _split_entries(entries: list[TrialEntry], split: str, train_ratio: float, val_ratio: float) -> list[TrialEntry]:
        count = len(entries)
        train_end = max(1, min(count, int(count * train_ratio)))
        val_end = max(train_end, min(count, int(count * (train_ratio + val_ratio))))
        if split == "train":
            return entries[:train_end]
        if split == "val":
            return entries[train_end:val_end]
        if split == "test":
            return entries[val_end:]
        raise ValueError(f"Unsupported split: {split}")

    def canonical_wavs_by_label(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for entry in self._load_entries():
            if entry.label not in out:
                out[entry.label] = self._load_audio(entry.audio_relpath)
        return out

    def _load_audio(self, relpath: str) -> np.ndarray:
        if relpath not in self._audio_cache:
            self._audio_cache[relpath] = load_wav_fixed(
                self.audio_root / relpath,
                sample_rate=self.audio_sr,
                n_samples=self.audio_n_samples,
                normalize=self.audio_normalize,
                target_rms=self.audio_target_rms,
                max_gain=self.audio_max_gain,
            )
        return self._audio_cache[relpath]

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def num_labels(self) -> int:
        return len(self.label_vocab)

    def __getitem__(self, index: int) -> dict[str, object]:
        entry = self.entries[index]
        eeg = torch.from_numpy(self.eeg_array[entry.position]).float()
        wav = torch.from_numpy(self._load_audio(entry.audio_relpath)).float()
        return {
            "eeg": eeg,
            "waveform": wav,
            "label": entry.label,
            "label_id": self.label_to_id[entry.label],
            "trial_index": entry.trial_index,
        }


class FEISProtocolDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        protocol: str,
        split: str,
        stage: str = "thinking",
        subject_id: str | None = None,
        subject_ids: list[str] | None = None,
        holdout_subject_id: str | None = None,
        include_anomalous: bool = False,
        ablation_mode: str = "none",
        target_cache_path: str | Path | None = None,
        require_targets: bool = False,
        audio_sr: int = 16000,
        audio_dur: float = 1.0,
        audio_normalize: str = "rms",
        audio_target_rms: float = 0.08,
        audio_max_gain: float = 10.0,
        seed: int = 7,
    ):
        self.protocol = str(protocol).upper()
        self.split = str(split)
        self.stage = str(stage)
        self.ablation_mode = str(ablation_mode)
        if self.ablation_mode not in ABLATION_MODES:
            raise ValueError(f"Unsupported ablation_mode: {self.ablation_mode}")

        self.feis_root = resolve_feis_root(data_root)
        self.audio_root = self.feis_root
        self.audio_sr = int(audio_sr)
        self.audio_n_samples = int(round(audio_sr * audio_dur))
        self.audio_normalize = str(audio_normalize)
        self.audio_target_rms = float(audio_target_rms)
        self.audio_max_gain = float(audio_max_gain)
        self.seed = int(seed)
        self._audio_cache: dict[str, np.ndarray] = {}
        self._bundle_cache: dict[str, dict[str, np.ndarray]] = {}
        self._subject_mean_cache: dict[str, np.ndarray] = {}

        manifest_path = self.feis_root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        self.clean_subject_ids = {
            _normalize_subject_id(value)
            for value in self.manifest.get("clean_subject_ids", [])
        }
        self.anomalous_subject_ids = {
            _normalize_subject_id(value)
            for value in self.manifest.get("anomalous_subject_ids", [])
        }

        all_rows = self._load_stage_rows()
        self.label_vocab = sorted({row["label"] for row in all_rows})
        self.label_to_id = {label: idx for idx, label in enumerate(self.label_vocab)}
        self.phoneme_vocab = build_phoneme_vocab(self.label_vocab)

        self.protocol_rows = self._select_protocol_rows(
            rows=all_rows,
            protocol=self.protocol,
            subject_id=subject_id,
            subject_ids=subject_ids,
            holdout_subject_id=holdout_subject_id,
            include_anomalous=include_anomalous,
        )
        self.subject_vocab = sorted({row["subject_id"] for row in self.protocol_rows})
        self.subject_to_index = {item: idx for idx, item in enumerate(self.subject_vocab)}
        self.protocol_splits = self._build_protocol_splits(self.protocol_rows, self.protocol, holdout_subject_id)
        if split not in self.protocol_splits:
            raise ValueError(f"Unsupported split: {split}")
        split_rows = self.protocol_splits[split]
        self.entries = self._rows_to_entries(split_rows)
        if not self.entries:
            raise ValueError(f"No FEIS samples left for protocol={protocol}, split={split}, stage={stage}")

        self.shuffle_indices = np.random.RandomState(self.seed).permutation(len(self.entries))
        self.target_cache = self._load_target_cache(target_cache_path, require_targets=require_targets)
        self.target_kind = "none" if self.target_cache is None else str(self.target_cache["target_kind"])
        self.target_sequence_steps = 0 if self.target_cache is None else int(self.target_cache["target_sequences"].shape[1])
        self.target_sequence_dim = 0 if self.target_cache is None else int(self.target_cache["target_sequences"].shape[2])
        self.target_embedding_dim = 0 if self.target_cache is None else int(self.target_cache["target_summaries"].shape[1])
        self.prosody_dim = 0 if self.target_cache is None else int(self.target_cache["prosody_targets"].shape[1])

    def _load_stage_rows(self) -> list[dict[str, Any]]:
        rows = _load_csv_rows(self.feis_root / "segments.csv")
        stage_rows: list[dict[str, Any]] = []
        for row in rows:
            if row.get("segment_stage") != self.stage:
                continue
            subject_id = _normalize_subject_id(row["subject_id"])
            audio_path = str(row.get("audio_path") or "")
            label = str(row["label"])
            stage_rows.append(
                {
                    "subject_id": subject_id,
                    "trial_index": int(row["trial_index"]),
                    "label": label,
                    "audio_path": audio_path,
                    "audio_relpath": audio_path,
                    "template_id": f"{subject_id}:{label}",
                    "is_clean_subject": _as_bool(row.get("is_clean_subject"), subject_id != "05"),
                    "audio_source_subject": _normalize_subject_id(row.get("audio_source_subject", subject_id)),
                    "audio_source_kind": str(row.get("audio_source_kind", "subject_wavs")),
                    "unique_hashes_per_subject_label": int(row.get("unique_hashes_per_subject_label", 1) or 1),
                    "unique_hashes_per_label_across_subjects": int(
                        row.get("unique_hashes_per_label_across_subjects", 1) or 1
                    ),
                    "eeg_valid_num_samples": int(row.get("eeg_valid_num_samples", 0) or 0),
                }
            )
        stage_rows.sort(key=lambda item: (item["subject_id"], item["label"], item["trial_index"]))
        return stage_rows

    def _select_protocol_rows(
        self,
        rows: list[dict[str, Any]],
        protocol: str,
        subject_id: str | None,
        subject_ids: list[str] | None,
        holdout_subject_id: str | None,
        include_anomalous: bool,
    ) -> list[dict[str, Any]]:
        protocol = protocol.upper()
        normalized_subject_ids = None if subject_ids is None else {_normalize_subject_id(item) for item in subject_ids}
        selected: list[dict[str, Any]] = []
        if protocol == "S":
            if subject_id is None:
                raise ValueError("Protocol S requires subject_id")
            keep_subject = _normalize_subject_id(subject_id)
            selected = [row for row in rows if row["subject_id"] == keep_subject]
        elif protocol == "G":
            selected = list(rows)
            if normalized_subject_ids is not None:
                selected = [row for row in selected if row["subject_id"] in normalized_subject_ids]
            elif not include_anomalous and self.clean_subject_ids:
                selected = [row for row in selected if row["subject_id"] in self.clean_subject_ids]
            elif not include_anomalous:
                selected = [row for row in selected if row["subject_id"] != "05"]
        elif protocol == "U":
            if holdout_subject_id is None:
                raise ValueError("Protocol U requires holdout_subject_id")
            holdout = _normalize_subject_id(holdout_subject_id)
            selected = list(rows)
            if normalized_subject_ids is not None:
                selected = [row for row in selected if row["subject_id"] in normalized_subject_ids or row["subject_id"] == holdout]
            elif not include_anomalous and self.clean_subject_ids:
                allowed = set(self.clean_subject_ids) | {holdout}
                selected = [row for row in selected if row["subject_id"] in allowed]
            elif not include_anomalous and holdout != "05":
                selected = [row for row in selected if row["subject_id"] != "05"]
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
        if not selected:
            raise ValueError(f"No rows selected for protocol={protocol}")
        return selected

    def _build_protocol_splits(
        self,
        rows: list[dict[str, Any]],
        protocol: str,
        holdout_subject_id: str | None,
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["subject_id"], row["label"])].append(row)
        for group_rows in grouped.values():
            group_rows.sort(key=lambda item: item["trial_index"])

        out: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        holdout = None if holdout_subject_id is None else _normalize_subject_id(holdout_subject_id)
        for (subject_key, _label), group_rows in grouped.items():
            count = len(group_rows)
            if protocol == "U" and subject_key == holdout:
                out["test"].extend(group_rows)
                continue
            if protocol == "U":
                train_cut = max(1, count - 1)
                out["train"].extend(group_rows[:train_cut])
                out["val"].extend(group_rows[train_cut:])
                continue
            if count < 3:
                raise ValueError(f"Need at least 3 repetitions per subject-label group, got {count} for {subject_key}")
            out["train"].extend(group_rows[:-2])
            out["val"].append(group_rows[-2])
            out["test"].append(group_rows[-1])

        for split_rows in out.values():
            split_rows.sort(key=lambda item: (item["subject_id"], item["trial_index"]))
        return out

    def _rows_to_entries(self, rows: list[dict[str, Any]]) -> list[ProtocolEntry]:
        entries: list[ProtocolEntry] = []
        for idx, row in enumerate(rows):
            bundle = self._load_bundle(row["subject_id"])
            trial_to_position = bundle["trial_to_position"]
            trial_index = int(row["trial_index"])
            if trial_index not in trial_to_position:
                raise KeyError(f"Trial {trial_index} missing from subject bundle {row['subject_id']}")
            position = int(trial_to_position[trial_index])
            label = str(row["label"])
            entries.append(
                ProtocolEntry(
                    index=idx,
                    subject_id=row["subject_id"],
                    subject_index=self.subject_to_index[row["subject_id"]],
                    trial_index=trial_index,
                    position=position,
                    label=label,
                    label_id=self.label_to_id[label],
                    audio_relpath=row["audio_relpath"],
                    audio_path=row["audio_path"],
                    template_id=row["template_id"],
                    split_name=self.split,
                    protocol=self.protocol,
                    is_clean_subject=bool(row["is_clean_subject"]),
                    audio_source_subject=str(row["audio_source_subject"]),
                    audio_source_kind=str(row["audio_source_kind"]),
                    unique_hashes_per_subject_label=int(row["unique_hashes_per_subject_label"]),
                    unique_hashes_per_label_across_subjects=int(row["unique_hashes_per_label_across_subjects"]),
                    eeg_valid_num_samples=int(row["eeg_valid_num_samples"]),
                )
            )
        return entries

    def _load_bundle(self, subject_id: str) -> dict[str, Any]:
        subject_id = _normalize_subject_id(subject_id)
        if subject_id not in self._bundle_cache:
            bundle_path = self.feis_root / "subjects" / f"{subject_id}.npz"
            if not bundle_path.exists():
                raise FileNotFoundError(f"Missing FEIS subject bundle: {bundle_path}")
            bundle = np.load(bundle_path, allow_pickle=True)
            stage_key = f"stage__{self.stage}"
            if stage_key not in bundle.files:
                raise KeyError(f"Stage {self.stage} is not present in {bundle_path.name}")
            trial_indices = bundle["trial_indices"].astype(np.int32)
            self._bundle_cache[subject_id] = {
                "eeg": bundle[stage_key].astype(np.float32),
                "trial_indices": trial_indices,
                "trial_to_position": {int(item): idx for idx, item in enumerate(trial_indices.tolist())},
                "channel_names": bundle["channel_names"].astype(str),
            }
        return self._bundle_cache[subject_id]

    def _load_audio(self, relpath: str) -> np.ndarray:
        if relpath not in self._audio_cache:
            self._audio_cache[relpath] = load_wav_fixed(
                self.audio_root / relpath,
                sample_rate=self.audio_sr,
                n_samples=self.audio_n_samples,
                normalize=self.audio_normalize,
                target_rms=self.audio_target_rms,
                max_gain=self.audio_max_gain,
            )
        return self._audio_cache[relpath]

    def _load_target_cache(self, target_cache_path: str | Path | None, require_targets: bool) -> dict[str, Any] | None:
        if target_cache_path is None:
            if require_targets:
                raise FileNotFoundError("Alignment datasets require target_cache_path")
            return None
        cache_path = Path(target_cache_path)
        candidate_paths = [cache_path]
        if not cache_path.is_absolute():
            candidate_paths.extend(
                [
                    Path.cwd() / cache_path,
                    self.feis_root.parent / cache_path,
                    self.feis_root.parent.parent / cache_path,
                ]
            )
        resolved_path = next((candidate for candidate in candidate_paths if candidate.exists()), None)
        if resolved_path is None:
            if require_targets:
                raise FileNotFoundError(f"Missing target cache: {cache_path}")
            return None
        payload = np.load(resolved_path, allow_pickle=True)
        template_ids = payload["template_ids"].astype(str)
        index = {template_id: idx for idx, template_id in enumerate(template_ids.tolist())}
        if "target_sequences" in payload.files:
            raw_target_sequences = payload["target_sequences"].astype(np.float32)
        else:
            raw_target_sequences = payload["speech_embeddings"].astype(np.float32)[:, None, :]
        if "target_masks" in payload.files:
            target_masks = payload["target_masks"].astype(np.float32)
        else:
            target_masks = np.ones(raw_target_sequences.shape[:2], dtype=np.float32)
        if "target_summaries" in payload.files:
            raw_target_summaries = payload["target_summaries"].astype(np.float32)
        else:
            raw_target_summaries = payload["speech_embeddings"].astype(np.float32)
        prosody_targets = (
            payload["prosody_targets"].astype(np.float32)
            if "prosody_targets" in payload.files
            else np.zeros((raw_target_sequences.shape[0], 0), dtype=np.float32)
        )
        target_kind = (
            str(payload["target_kind"].item())
            if "target_kind" in payload.files
            else ("hubert_pooled" if raw_target_sequences.shape[1] == 1 else "unknown")
        )
        target_mean = (
            payload["target_mean"].astype(np.float32)
            if "target_mean" in payload.files
            else raw_target_sequences.mean(axis=(0, 1)).astype(np.float32)
        )
        target_std = (
            payload["target_std"].astype(np.float32)
            if "target_std" in payload.files
            else raw_target_sequences.std(axis=(0, 1)).astype(np.float32)
        )
        target_std = np.maximum(target_std, 1e-6).astype(np.float32)
        if target_kind == TARGET_KIND_ENCODEC_LATENT:
            target_sequences = ((raw_target_sequences - target_mean.reshape(1, 1, -1)) / target_std.reshape(1, 1, -1)).astype(
                np.float32
            )
            target_summaries = target_sequences.mean(axis=1).astype(np.float32)
        else:
            target_sequences = raw_target_sequences
            target_summaries = raw_target_summaries
        target_rms = (
            payload["target_rms"].astype(np.float32)
            if "target_rms" in payload.files
            else np.exp(prosody_targets[:, 1]).astype(np.float32)
            if prosody_targets.shape[1] > 1
            else np.ones((target_sequences.shape[0],), dtype=np.float32)
        )
        target_log_rms = (
            payload["target_log_rms"].astype(np.float32)
            if "target_log_rms" in payload.files
            else np.log(np.maximum(target_rms, 1e-8)).astype(np.float32)
        )
        return {
            "path": resolved_path,
            "template_ids": template_ids,
            "subject_ids": payload["subject_ids"].astype(str),
            "labels": payload["labels"].astype(str),
            "audio_paths": payload["audio_paths"].astype(str),
            "speech_embeddings": target_summaries,
            "target_sequences": target_sequences,
            "target_masks": target_masks,
            "target_summaries": target_summaries,
            "raw_target_sequences": raw_target_sequences,
            "raw_target_summaries": raw_target_summaries,
            "target_mean": target_mean,
            "target_std": target_std,
            "targets_are_normalized": target_kind == TARGET_KIND_ENCODEC_LATENT,
            "target_rms": target_rms,
            "target_log_rms": target_log_rms,
            "prosody_targets": prosody_targets,
            "feature_backend": payload["feature_backend"].astype(str),
            "target_kind": target_kind,
            "decoder_scales": payload["decoder_scales"].astype(np.float32) if "decoder_scales" in payload.files else None,
            "default_decoder_scales": payload["default_decoder_scales"].astype(np.float32)
            if "default_decoder_scales" in payload.files
            else None,
            "index": index,
        }

    @property
    def num_labels(self) -> int:
        return len(self.label_vocab)

    @property
    def num_subjects(self) -> int:
        return len(self.subject_vocab)

    @property
    def channel_names(self) -> np.ndarray:
        first_subject = self.entries[0].subject_id
        return self._load_bundle(first_subject)["channel_names"]

    def unique_template_ids(self, split: str | None = None) -> list[str]:
        source_rows = self.protocol_splits[self.split if split is None else split]
        return sorted({row["template_id"] for row in source_rows})

    def template_metadata(self, template_id: str) -> dict[str, Any]:
        subject_id, label = template_id.split(":", 1)
        for row in self.protocol_rows:
            if row["subject_id"] == subject_id and row["label"] == label:
                return dict(row)
        raise KeyError(f"Unknown template_id: {template_id}")

    def get_template_target(self, template_id: str) -> dict[str, np.ndarray]:
        if self.target_cache is None:
            raise RuntimeError("Target cache is not loaded")
        idx = self.target_cache["index"][template_id]
        return {
            "speech_embedding": self.target_cache["target_summaries"][idx],
            "target_sequence": self.target_cache["target_sequences"][idx],
            "target_mask": self.target_cache["target_masks"][idx],
            "target_summary": self.target_cache["target_summaries"][idx],
            "raw_target_sequence": self.target_cache["raw_target_sequences"][idx],
            "raw_target_summary": self.target_cache["raw_target_summaries"][idx],
            "prosody_target": self.target_cache["prosody_targets"][idx],
            "target_log_rms": self.target_cache["target_log_rms"][idx],
            "decoder_scale": None if self.target_cache["decoder_scales"] is None else self.target_cache["decoder_scales"][idx],
        }

    def build_control_predictions(self, mode: str, use_oracle_for_unseen: bool = False) -> dict[str, np.ndarray]:
        if self.target_cache is None:
            raise RuntimeError("Target cache is not loaded")
        mode = str(mode)
        if mode not in {"label_only", "subject_only", "label_subject"}:
            raise ValueError(f"Unsupported control mode: {mode}")

        train_templates = self.unique_template_ids(split="train")
        reference_templates = list(train_templates)
        if use_oracle_for_unseen:
            reference_templates = self.unique_template_ids(split="test")

        label_pool: dict[str, list[np.ndarray]] = defaultdict(list)
        subject_pool: dict[str, list[np.ndarray]] = defaultdict(list)
        label_subject_pool: dict[tuple[str, str], np.ndarray] = {}
        for template_id in reference_templates:
            meta = self.template_metadata(template_id)
            target = self.get_template_target(template_id)
            label_pool[meta["label"]].append(target["target_sequence"])
            subject_pool[meta["subject_id"]].append(target["target_sequence"])
            label_subject_pool[(meta["subject_id"], meta["label"])] = target["target_sequence"]

        label_means = {key: np.mean(np.stack(values, axis=0), axis=0) for key, values in label_pool.items()}
        subject_means = {key: np.mean(np.stack(values, axis=0), axis=0) for key, values in subject_pool.items()}
        zero_sequence = np.zeros((self.target_sequence_steps, self.target_sequence_dim), dtype=np.float32)
        pred_sequences: list[np.ndarray] = []
        availability: list[float] = []
        for entry in self.entries:
            availability.append(1.0)
            if mode == "label_only":
                pred_sequences.append(label_means.get(entry.label, zero_sequence))
                continue
            if mode == "subject_only":
                available = entry.subject_id in subject_means
                availability[-1] = 1.0 if available else 0.0
                pred_sequences.append(subject_means.get(entry.subject_id, zero_sequence))
                continue
            key = (entry.subject_id, entry.label)
            available = key in label_subject_pool
            availability[-1] = 1.0 if available else 0.0
            pred_sequences.append(label_subject_pool.get(key, zero_sequence))
        sequences = np.stack(pred_sequences, axis=0).astype(np.float32)
        return {
            "target_sequences": sequences,
            "target_masks": np.ones((sequences.shape[0], sequences.shape[1]), dtype=np.float32),
            "target_summaries": sequences.mean(axis=1).astype(np.float32),
            "speech_embeddings": sequences.mean(axis=1).astype(np.float32),
            "availability": np.asarray(availability, dtype=np.float32),
        }

    def _subject_mean_eeg(self, subject_id: str) -> np.ndarray:
        subject_id = _normalize_subject_id(subject_id)
        if subject_id not in self._subject_mean_cache:
            train_rows = [row for row in self.protocol_splits["train"] if row["subject_id"] == subject_id]
            if not train_rows:
                matching = [entry for entry in self.entries if entry.subject_id == subject_id]
                train_rows = [self.protocol_rows[entry.index] for entry in matching]
            eeg_stack = []
            for row in train_rows:
                bundle = self._load_bundle(row["subject_id"])
                position = bundle["trial_to_position"][int(row["trial_index"])]
                eeg_stack.append(bundle["eeg"][position])
            self._subject_mean_cache[subject_id] = np.mean(np.stack(eeg_stack, axis=0), axis=0).astype(np.float32)
        return self._subject_mean_cache[subject_id]

    def _ablate_eeg(self, entry: ProtocolEntry, eeg: np.ndarray, index: int) -> np.ndarray:
        if self.ablation_mode == "none":
            return eeg
        if self.ablation_mode == "random_noise":
            return np.random.RandomState(self.seed + index).randn(*eeg.shape).astype(np.float32)
        if self.ablation_mode == "shuffle_eeg":
            shuffled_entry = self.entries[int(self.shuffle_indices[index])]
            shuffled_bundle = self._load_bundle(shuffled_entry.subject_id)
            return shuffled_bundle["eeg"][shuffled_entry.position].astype(np.float32)
        if self.ablation_mode == "subject_mean":
            return self._subject_mean_eeg(entry.subject_id)
        if self.ablation_mode in {"label_only", "subject_only", "label_subject"}:
            return np.zeros_like(eeg, dtype=np.float32)
        raise ValueError(f"Unsupported ablation mode: {self.ablation_mode}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, object]:
        entry = self.entries[index]
        bundle = self._load_bundle(entry.subject_id)
        eeg_np = bundle["eeg"][entry.position].astype(np.float32)
        eeg_np = self._ablate_eeg(entry, eeg_np, index)
        wav_np = self._load_audio(entry.audio_relpath)

        if self.target_cache is None:
            speech_embedding = np.zeros((0,), dtype=np.float32)
            target_sequence = np.zeros((0, 0), dtype=np.float32)
            target_mask = np.zeros((0,), dtype=np.float32)
            target_summary = np.zeros((0,), dtype=np.float32)
            raw_target_sequence = np.zeros((0, 0), dtype=np.float32)
            raw_target_summary = np.zeros((0,), dtype=np.float32)
            prosody_target = np.zeros((0,), dtype=np.float32)
            target_log_rms = np.asarray(0.0, dtype=np.float32)
            decoder_scale = np.zeros((0,), dtype=np.float32)
            target_kind = "none"
        else:
            template_target = self.get_template_target(entry.template_id)
            speech_embedding = template_target["speech_embedding"].astype(np.float32)
            target_sequence = template_target["target_sequence"].astype(np.float32)
            target_mask = template_target["target_mask"].astype(np.float32)
            target_summary = template_target["target_summary"].astype(np.float32)
            raw_target_sequence = template_target["raw_target_sequence"].astype(np.float32)
            raw_target_summary = template_target["raw_target_summary"].astype(np.float32)
            prosody_target = template_target["prosody_target"].astype(np.float32)
            target_log_rms = np.asarray(template_target["target_log_rms"], dtype=np.float32)
            decoder_scale = (
                np.asarray(template_target["decoder_scale"], dtype=np.float32)
                if template_target["decoder_scale"] is not None
                else np.zeros((0,), dtype=np.float32)
            )
            target_kind = self.target_kind
        phoneme_ids, phoneme_mask = encode_label_phonemes(entry.label, self.phoneme_vocab)

        return {
            "eeg": torch.from_numpy(eeg_np).float(),
            "waveform": torch.from_numpy(wav_np).float(),
            "label": entry.label,
            "label_id": torch.tensor(entry.label_id, dtype=torch.long),
            "trial_index": torch.tensor(entry.trial_index, dtype=torch.long),
            "subject_id": entry.subject_id,
            "subject_index": torch.tensor(entry.subject_index, dtype=torch.long),
            "template_id": entry.template_id,
            "audio_path": entry.audio_path,
            "speech_embedding": torch.from_numpy(speech_embedding).float(),
            "target_sequence": torch.from_numpy(target_sequence).float(),
            "target_mask": torch.from_numpy(target_mask).float(),
            "target_summary": torch.from_numpy(target_summary).float(),
            "raw_target_sequence": torch.from_numpy(raw_target_sequence).float(),
            "raw_target_summary": torch.from_numpy(raw_target_summary).float(),
            "prosody_target": torch.from_numpy(prosody_target).float(),
            "target_log_rms": torch.tensor(float(target_log_rms), dtype=torch.float32),
            "decoder_scale": torch.from_numpy(decoder_scale).float(),
            "phoneme_ids": torch.from_numpy(phoneme_ids).long(),
            "phoneme_mask": torch.from_numpy(phoneme_mask).float(),
            "target_kind": target_kind,
            "is_clean_subject": torch.tensor(float(entry.is_clean_subject), dtype=torch.float32),
            "split_name": entry.split_name,
            "protocol": entry.protocol,
            "audio_source_subject": entry.audio_source_subject,
            "audio_source_kind": entry.audio_source_kind,
            "unique_hashes_per_subject_label": torch.tensor(entry.unique_hashes_per_subject_label, dtype=torch.long),
            "unique_hashes_per_label_across_subjects": torch.tensor(
                entry.unique_hashes_per_label_across_subjects,
                dtype=torch.long,
            ),
            "eeg_valid_num_samples": torch.tensor(entry.eeg_valid_num_samples, dtype=torch.long),
            "ablation_mode": self.ablation_mode,
        }
