from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


class KaraOneTargets:
    """Trial-level EnCodec target cache for KaraOne."""

    def __init__(self, cache_path: str | Path, data_root: str | Path | None = None):
        payload = np.load(Path(cache_path), allow_pickle=True)
        if "template_ids" in payload.files:
            self.template_ids = payload["template_ids"].astype(str)
            self.subject_ids = payload["subject_ids"].astype(str)
            self.labels = payload["labels"].astype(str)
            self.trial_indices = payload["trial_indices"].astype(np.int32)
            self.audio_paths = (
                payload["audio_paths"].astype(str)
                if "audio_paths" in payload.files
                else np.array([""] * len(self.template_ids))
            )
        elif "target_keys" in payload.files:
            self.template_ids = payload["target_keys"].astype(str)
            parsed = [item.split(":", 1) for item in self.template_ids.tolist()]
            self.subject_ids = np.asarray([item[0] for item in parsed])
            self.trial_indices = np.asarray([int(item[1]) for item in parsed], dtype=np.int32)
            labels, audio_paths = self._metadata_from_trials(data_root)
            self.labels = np.asarray([labels.get(key, "") for key in self.template_ids.tolist()])
            self.audio_paths = np.asarray([audio_paths.get(key, "") for key in self.template_ids.tolist()])
        else:
            raise KeyError(f"Unsupported KaraOne target cache keys: {payload.files}")
        raw_seq = payload["target_sequences"].astype(np.float32)
        self.T, self.D = int(raw_seq.shape[1]), int(raw_seq.shape[2])

        if "target_mean" in payload.files:
            mean = payload["target_mean"].astype(np.float32)
            std = np.maximum(payload["target_std"].astype(np.float32), 1e-6)
        else:
            mean = raw_seq.reshape(-1, self.D).mean(axis=0).astype(np.float32)
            std = np.maximum(raw_seq.reshape(-1, self.D).std(axis=0), 1e-6).astype(np.float32)
        self.target_mean = mean
        self.target_std = std
        self.raw_seq = raw_seq
        self.seq = ((raw_seq - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)
        self.summary = self.seq.mean(axis=1).astype(np.float32)
        self.global_mean_raw = raw_seq.mean(axis=0).astype(np.float32)
        self.global_mean_norm = ((self.global_mean_raw - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)

        n = raw_seq.shape[0]
        self.target_rms = payload["target_rms"].astype(np.float32) if "target_rms" in payload.files else np.full(n, 0.08, np.float32)
        self.target_log_rms = (
            payload["target_log_rms"].astype(np.float32)
            if "target_log_rms" in payload.files
            else np.log(np.maximum(self.target_rms, 1e-8)).astype(np.float32)
        )
        if "decoder_scales" in payload.files:
            self.decoder_scales = payload["decoder_scales"].astype(np.float32)
        else:
            self.decoder_scales = np.ones((n, 1), np.float32)
        self.default_decoder_scales = (
            payload["default_decoder_scales"].astype(np.float32)
            if "default_decoder_scales" in payload.files
            else self.decoder_scales.mean(axis=0).astype(np.float32)
        )
        self.decoder_scale_dim = int(self.default_decoder_scales.reshape(-1).shape[0])
        self.has_complete_audio_metadata = all(
            key in payload.files
            for key in ("labels", "trial_indices", "audio_paths", "target_rms", "target_log_rms", "decoder_scales")
        )

        self.key_to_idx = {
            self.key(subject, int(trial)): idx
            for idx, (subject, trial) in enumerate(zip(self.subject_ids.tolist(), self.trial_indices.tolist()))
        }
        self.subject_vocab = sorted(set(self.subject_ids.tolist()))
        self.label_vocab = sorted(set(self.labels.tolist()))
        self.subject_to_id = {subject: idx for idx, subject in enumerate(self.subject_vocab)}
        self.label_to_id = {label: idx for idx, label in enumerate(self.label_vocab)}

        self.content_proto = np.stack(
            [self.summary[[i for i, label in enumerate(self.labels) if label == item]].mean(axis=0) for item in self.label_vocab],
            axis=0,
        ).astype(np.float32)
        self.subject_proto = np.stack(
            [
                self.summary[[i for i, subject in enumerate(self.subject_ids) if subject == item]].mean(axis=0)
                for item in self.subject_vocab
            ],
            axis=0,
        ).astype(np.float32)

    @staticmethod
    def _metadata_from_trials(data_root: str | Path | None) -> tuple[dict[str, str], dict[str, str]]:
        if data_root is None:
            return {}, {}
        path = Path(data_root) / "trials.csv"
        if not path.exists():
            return {}, {}
        labels: dict[str, str] = {}
        audio_paths: dict[str, str] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = KaraOneTargets.key(row["subject_id"], int(row["trial_index"]))
                labels[key] = str(row["label"])
                audio_paths[key] = str(row["audio_path"])
        return labels, audio_paths

    @staticmethod
    def key(subject: str, trial_index: int) -> str:
        return f"{subject}:{int(trial_index)}"

    def has_trial(self, subject: str, trial_index: int) -> bool:
        return self.key(subject, trial_index) in self.key_to_idx

    def index(self, subject: str, trial_index: int) -> int:
        return self.key_to_idx[self.key(subject, trial_index)]

    def target(self, subject: str, trial_index: int) -> np.ndarray:
        return self.seq[self.index(subject, trial_index)]

    def raw_target(self, subject: str, trial_index: int) -> np.ndarray:
        return self.raw_seq[self.index(subject, trial_index)]

    def target_summary(self, subject: str, trial_index: int) -> np.ndarray:
        return self.summary[self.index(subject, trial_index)]

    def target_log_rms_value(self, subject: str, trial_index: int) -> float:
        return float(self.target_log_rms[self.index(subject, trial_index)])

    def decoder_scale(self, subject: str, trial_index: int) -> np.ndarray:
        return self.decoder_scales[self.index(subject, trial_index)].astype(np.float32)

    def audio_path(self, subject: str, trial_index: int) -> str:
        return str(self.audio_paths[self.index(subject, trial_index)])

    def content_prototype(self, label: str) -> np.ndarray:
        return self.content_proto[self.label_to_id[label]]

    def subject_prototype(self, subject: str) -> np.ndarray:
        return self.subject_proto[self.subject_to_id[subject]]
