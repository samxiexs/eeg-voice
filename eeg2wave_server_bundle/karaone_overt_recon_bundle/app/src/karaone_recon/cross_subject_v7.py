from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from .data import KaraOneTrialDataset
from .encoder import SpatialTemporalEEGEncoder
from .losses import grad_reverse
from .model import _masked_time_mean


class KaraOneEEGFeatureCache:
    """Trial/stage EEG feature cache for v7 cross-subject representation learning."""

    def __init__(self, path: str | Path):
        payload = np.load(Path(path), allow_pickle=True)
        self.path = Path(path)
        self.template_ids = payload["template_ids"].astype(str)
        self.subject_ids = payload["subject_ids"].astype(str)
        self.labels = payload["labels"].astype(str)
        self.stages = payload["stages"].astype(str)
        self.trial_indices = payload["trial_indices"].astype(np.int32)
        self.valid_lengths = payload["valid_lengths"].astype(np.int32)
        self.feature_vectors = payload["feature_vectors"].astype(np.float32)
        self.envelopes = payload["envelopes"].astype(np.float32)
        self.feature_mean = payload["feature_mean"].astype(np.float32)
        self.feature_std = np.maximum(payload["feature_std"].astype(np.float32), 1e-6)
        self.feature_names = payload["feature_names"].astype(str) if "feature_names" in payload.files else np.array([])
        self.key_to_idx = {
            self.key(subject, stage, int(trial)): i
            for i, (subject, stage, trial) in enumerate(
                zip(self.subject_ids.tolist(), self.stages.tolist(), self.trial_indices.tolist())
            )
        }

    @staticmethod
    def key(subject: str, stage: str, trial_index: int) -> str:
        return f"{subject}:{stage}:{int(trial_index)}"

    @property
    def feature_dim(self) -> int:
        return int(self.feature_vectors.shape[1])

    @property
    def envelope_steps(self) -> int:
        return int(self.envelopes.shape[1])

    def has_trial(self, subject: str, stage: str, trial_index: int) -> bool:
        return self.key(subject, stage, trial_index) in self.key_to_idx

    def get(self, subject: str, stage: str, trial_index: int) -> tuple[np.ndarray, np.ndarray]:
        idx = self.key_to_idx[self.key(subject, stage, trial_index)]
        return self.feature_vectors[idx], self.envelopes[idx]


class KaraOneV7Dataset(Dataset):
    def __init__(self, base: KaraOneTrialDataset, features: KaraOneEEGFeatureCache):
        self.base = base
        self.features = features
        self.entries = base.entries
        self.targets = base.targets
        self.subject_vocab = base.subject_vocab
        self.label_vocab = base.label_vocab
        self.num_subjects = base.num_subjects
        self.num_labels = base.num_labels
        self.num_stages = base.num_stages

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        feat, env = self.features.get(str(item["subject"]), str(item["stage"]), int(item["trial_index"]))
        item["eeg_feature"] = torch.from_numpy(feat).float()
        item["eeg_envelope"] = torch.from_numpy(env).float()
        return item


class SubjectBalancedBatchSampler(Sampler[list[int]]):
    """Batch sampler that forces each batch to contain multiple train subjects.

    This prevents the InfoNCE task from degenerating into easy subject/session
    separation. Sampling is with replacement, which is acceptable for small KaraOne.
    """

    def __init__(
        self,
        dataset: KaraOneV7Dataset,
        batch_size: int,
        subjects_per_batch: int = 6,
        batches_per_epoch: int | None = None,
        seed: int = 7,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.subjects_per_batch = max(2, int(subjects_per_batch))
        self.rng = np.random.default_rng(int(seed))
        by_subject: dict[str, list[int]] = {}
        for i, entry in enumerate(dataset.entries):
            by_subject.setdefault(str(entry.subject), []).append(i)
        self.by_subject = {k: np.asarray(v, dtype=np.int64) for k, v in by_subject.items()}
        self.subjects = sorted(self.by_subject)
        self.batches_per_epoch = int(batches_per_epoch or max(1, int(np.ceil(len(dataset) / max(self.batch_size, 1)))))

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            chosen_subjects = self.rng.choice(
                self.subjects,
                size=min(self.subjects_per_batch, len(self.subjects)),
                replace=False,
            )
            per_subject = max(1, self.batch_size // max(len(chosen_subjects), 1))
            batch: list[int] = []
            for subject in chosen_subjects:
                pool = self.by_subject[str(subject)]
                take = self.rng.choice(pool, size=per_subject, replace=True)
                batch.extend(int(x) for x in take.tolist())
            while len(batch) < self.batch_size:
                subject = str(self.rng.choice(self.subjects))
                batch.append(int(self.rng.choice(self.by_subject[subject])))
            self.rng.shuffle(batch)
            yield batch[: self.batch_size]


@dataclass
class KaraOneV7Config:
    n_channels_eeg: int = 62
    d_model: int = 128
    cond_dim: int = 32
    num_labels: int = 11
    num_subjects: int = 14
    num_stages: int = 1
    core_steps: int = 64
    core_dim: int = 80
    feature_dim: int = 1
    envelope_steps: int = 64
    audio_embed_dim: int = 768
    num_blocks: int = 4
    kernel_size: int = 7
    channel_dropout: float = 0.2
    dropout: float = 0.2
    use_channel_reliability: bool = True


class KaraOneV7CrossSubject(nn.Module):
    """Small dual-branch EEG model for cross-subject speech representation learning."""

    def __init__(self, cfg: KaraOneV7Config):
        super().__init__()
        self.cfg = cfg
        self.stage_embedding = nn.Embedding(cfg.num_stages, cfg.cond_dim)
        nn.init.normal_(self.stage_embedding.weight, std=0.02)
        self.raw_encoder = SpatialTemporalEEGEncoder(
            in_channels=cfg.n_channels_eeg,
            d_model=cfg.d_model,
            cond_dim=cfg.cond_dim,
            target_steps=cfg.core_steps,
            num_blocks=cfg.num_blocks,
            kernel_size=cfg.kernel_size,
            channel_dropout=cfg.channel_dropout,
            dropout=cfg.dropout,
            num_channel_experts=1,
            instance_norm=True,
            use_channel_reliability=bool(cfg.use_channel_reliability),
        )
        self.feature_branch = nn.Sequential(
            nn.LayerNorm(cfg.feature_dim),
            nn.Linear(cfg.feature_dim, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.envelope_branch = nn.Sequential(
            nn.LayerNorm(cfg.envelope_steps),
            nn.Linear(cfg.envelope_steps, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        fusion_dim = cfg.d_model * 3
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
        )
        self.eeg_embed_head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.audio_embed_dim))
        self.core_delta_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.core_dim),
        )
        self.content_classifier = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Linear(cfg.d_model, cfg.num_labels))
        self.subject_classifier = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_subjects),
        )

    def forward(
        self,
        eeg: torch.Tensor,
        eeg_feature: torch.Tensor,
        eeg_envelope: torch.Tensor,
        stage_idx: torch.Tensor,
        eeg_valid_len: torch.Tensor | None = None,
        lambda_subject_adv: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        cond = self.stage_embedding(stage_idx.long())
        raw, aux = self.raw_encoder(eeg, cond, eeg_valid_len)
        seq = raw.transpose(1, 2)
        pooled = _masked_time_mean(seq, eeg_valid_len, int(eeg.shape[-1]))
        feat = self.feature_branch(eeg_feature.float())
        env = self.envelope_branch(eeg_envelope.float())
        fused = self.fusion(torch.cat([pooled, feat, env], dim=-1))
        fused_seq = fused.unsqueeze(1).expand(-1, self.cfg.core_steps, -1)
        out = {
            "eeg_embed": self.eeg_embed_head(fused),
            "pred_core_delta": self.core_delta_head(fused_seq),
            "content_logits": self.content_classifier(fused),
            "subject_logits": self.subject_classifier(grad_reverse(fused, float(lambda_subject_adv))),
            "pooled": fused,
        }
        out.update(aux)
        return out


def normalize_audio_embed(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)
