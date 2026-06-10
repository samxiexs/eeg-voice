"""EEG2SpeechV3 model: EEG -> EnCodec-latent sequence + class + contrastive embed.

Output `speech_sequence` has shape [B, target_steps, target_dim] and lives in
the *normalised* EnCodec-latent space used by the target cache. To synthesise
audio, denormalise with the cache's target_mean/target_std and feed the frozen
EnCodec decoder (see v3/synth.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from dataclasses import field

from .encoder import SpatialAdapter, SpatialTemporalEEGEncoder, TemporalTrunk


@dataclass
class EEG2SpeechV3Config:
    n_channels_eeg: int = 14
    d_model: int = 256
    cond_dim: int = 64
    num_subjects: int = 21
    target_steps: int = 75          # EnCodec 24k @ 75 Hz for 1.0 s
    target_dim: int = 128           # EnCodec latent dim
    num_labels: int = 16
    num_blocks: int = 5
    kernel_size: int = 5
    channel_dropout: float = 0.1
    dropout: float = 0.1
    embed_dim: int = 128            # contrastive projection dim


class EEG2SpeechV3(nn.Module):
    def __init__(self, config: EEG2SpeechV3Config):
        super().__init__()
        self.config = config
        # Subject embedding (+1 row for an "unknown" subject at index num_subjects).
        self.subject_embedding = nn.Embedding(config.num_subjects + 1, config.cond_dim)
        nn.init.normal_(self.subject_embedding.weight, std=0.02)

        self.encoder = SpatialTemporalEEGEncoder(
            in_channels=config.n_channels_eeg,
            d_model=config.d_model,
            cond_dim=config.cond_dim,
            target_steps=config.target_steps,
            num_blocks=config.num_blocks,
            kernel_size=config.kernel_size,
            channel_dropout=config.channel_dropout,
            dropout=config.dropout,
        )

        d = config.d_model
        # Content head: per-frame EnCodec latent prediction.
        self.content_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, config.target_dim),
        )
        # Class head: 16-way prompt identity from pooled latent.
        self.class_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, config.num_labels),
        )
        # Contrastive embedding head (projects pooled latent for InfoNCE).
        self.embed_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, config.embed_dim),
        )

    @property
    def unknown_subject_index(self) -> int:
        return self.config.num_subjects

    def forward(
        self,
        eeg: torch.Tensor,
        subject_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b = eeg.shape[0]
        if subject_indices is None:
            idx = torch.full((b,), self.unknown_subject_index, device=eeg.device, dtype=torch.long)
        else:
            idx = subject_indices.long().clamp(min=0, max=self.config.num_subjects)
        cond = self.subject_embedding(idx)                       # [B, cond_dim]

        seq = self.encoder(eeg, cond)                            # [B, d_model, T]
        seq = seq.transpose(1, 2)                                # [B, T, d_model]
        pooled = seq.mean(dim=1)                                 # [B, d_model]

        speech_sequence = self.content_head(seq)                # [B, T, target_dim]
        speech_embedding = speech_sequence.mean(dim=1)          # [B, target_dim]
        label_logits = self.class_head(pooled)                  # [B, num_labels]
        contrastive_embedding = self.embed_head(pooled)         # [B, embed_dim]

        return {
            "speech_sequence": speech_sequence,
            "speech_embedding": speech_embedding,
            "label_logits": label_logits,
            "contrastive_embedding": contrastive_embedding,
            "pooled_latent": pooled,
        }


@dataclass
class DatasetHead:
    """Per-dataset config for the multi-dataset model."""

    name: str
    n_channels: int
    num_labels: int


@dataclass
class EEG2SpeechMDConfig:
    """Multi-dataset model config.

    `datasets` declares one (name, n_channels, num_labels) per registered
    dataset. The trunk + content/contrastive heads are shared; input adapters
    and class heads are per-dataset. Subject ids are global across datasets.
    """

    datasets: list = field(default_factory=list)   # list[DatasetHead]
    d_model: int = 256
    cond_dim: int = 64
    num_subjects: int = 35                          # global subject count
    target_steps: int = 150                         # common EnCodec-frame window
    target_dim: int = 128
    num_blocks: int = 5
    kernel_size: int = 5
    channel_dropout: float = 0.1
    dropout: float = 0.1
    embed_dim: int = 128


class EEG2SpeechMD(nn.Module):
    """Multi-dataset EEG -> shared speech-latent model.

    forward(eeg, subject_indices, dataset_name): a batch is dataset-homogeneous
    (one dataset per batch) because channel counts differ; the adapter is
    selected by `dataset_name`.
    """

    def __init__(self, config: EEG2SpeechMDConfig):
        super().__init__()
        self.config = config
        specs = [d if isinstance(d, DatasetHead) else DatasetHead(**d) for d in config.datasets]
        self.dataset_specs = {s.name: s for s in specs}
        self.dataset_order = [s.name for s in specs]

        # Conditioning: global subject embedding (+1 unknown) and dataset embedding.
        self.subject_embedding = nn.Embedding(config.num_subjects + 1, config.cond_dim)
        self.dataset_embedding = nn.Embedding(max(len(specs), 1), config.cond_dim)
        nn.init.normal_(self.subject_embedding.weight, std=0.02)
        nn.init.normal_(self.dataset_embedding.weight, std=0.02)
        self.dataset_to_id = {name: i for i, name in enumerate(self.dataset_order)}

        # Per-dataset input adapters (channels -> d_model).
        self.adapters = nn.ModuleDict(
            {
                s.name: SpatialAdapter(s.n_channels, config.d_model, config.cond_dim, config.channel_dropout)
                for s in specs
            }
        )
        # Shared trunk.
        self.trunk = TemporalTrunk(
            d_model=config.d_model,
            cond_dim=config.cond_dim,
            target_steps=config.target_steps,
            num_blocks=config.num_blocks,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )
        d = config.d_model
        # Shared content + contrastive heads (the common representation space).
        self.content_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(d, config.target_dim),
        )
        self.embed_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, config.embed_dim))
        # Per-dataset classification heads (label spaces differ).
        self.class_heads = nn.ModuleDict(
            {s.name: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, s.num_labels))
             for s in specs}
        )

    @property
    def unknown_subject_index(self) -> int:
        return self.config.num_subjects

    def forward(
        self,
        eeg: torch.Tensor,
        subject_indices: torch.Tensor | None,
        dataset_name: str,
    ) -> dict[str, torch.Tensor]:
        if dataset_name not in self.adapters:
            raise KeyError(f"Unknown dataset '{dataset_name}'. Registered: {self.dataset_order}")
        b = eeg.shape[0]
        if subject_indices is None:
            sid = torch.full((b,), self.unknown_subject_index, device=eeg.device, dtype=torch.long)
        else:
            sid = subject_indices.long().clamp(min=0, max=self.config.num_subjects)
        ds_id = torch.full((b,), self.dataset_to_id[dataset_name], device=eeg.device, dtype=torch.long)
        cond = self.subject_embedding(sid) + self.dataset_embedding(ds_id)   # [B, cond_dim]

        h = self.adapters[dataset_name](eeg, cond)          # [B, d_model, L]
        h = self.trunk(h, cond)                             # [B, d_model, T]
        seq = h.transpose(1, 2)                             # [B, T, d_model]
        pooled = seq.mean(dim=1)

        speech_sequence = self.content_head(seq)            # [B, T, target_dim]
        return {
            "speech_sequence": speech_sequence,
            "speech_embedding": speech_sequence.mean(dim=1),
            "label_logits": self.class_heads[dataset_name](pooled),
            "contrastive_embedding": self.embed_head(pooled),
            "pooled_latent": pooled,
        }
