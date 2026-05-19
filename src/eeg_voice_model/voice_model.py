"""English-first EEG voice token foundation model V1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .heads import ClassificationHead, RegressionHead, RetrievalHead, SequenceClassificationHead, SpeakingModeHead, pool_tokens
from .losses import dataset_predictability_proxy, eeg_reconstruction_loss, token_usage_metrics
from .tokenizer import EEGVoiceTokenizerV1, EEGVoiceV1Config, GroupedRVQOutput


@dataclass
class VoiceAlignmentTargets:
    content_labels: torch.Tensor | None = None
    phoneme_labels: torch.Tensor | None = None
    pitch_target: torch.Tensor | None = None
    prosody_target: torch.Tensor | None = None
    timbre_target: torch.Tensor | None = None
    style_labels: torch.Tensor | None = None
    mode_labels: torch.Tensor | None = None


@dataclass
class EEGVoiceBatch:
    eeg: torch.Tensor
    sensor_pos: torch.Tensor
    channel_mask: torch.Tensor | None
    dataset_id: list[str]
    language: list[str]
    domain_group: list[str]
    speaker_id: list[str] | None = None
    audio_embedding: torch.Tensor | None = None
    targets: VoiceAlignmentTargets | None = None
    sensor_type: torch.Tensor | None = None
    acquisition_device_id: torch.Tensor | None = None
    montage_id: torch.Tensor | None = None
    reference_id: torch.Tensor | None = None
    sampling_rate_hz: torch.Tensor | None = None
    native_channel_count: torch.Tensor | None = None


class EEGVoiceTokenV1(nn.Module):
    """Grouped-token EEG voice model with attribute and retrieval heads."""

    head_groups = {
        "content": ("base", "content"),
        "prosody": ("base", "prosody"),
        "voice": ("base", "voice"),
        "mode": ("base", "content", "prosody", "voice"),
    }

    def __init__(self, config: EEGVoiceV1Config | None = None):
        super().__init__()
        self.config = config or EEGVoiceV1Config()
        cfg = self.config
        self.tokenizer = EEGVoiceTokenizerV1(cfg)
        self.content_head = SequenceClassificationHead(cfg.dim, cfg.content_classes, dropout=cfg.dropout)
        self.phoneme_head = SequenceClassificationHead(cfg.dim, cfg.phoneme_classes, dropout=cfg.dropout)
        self.pitch_head = RegressionHead(cfg.dim, cfg.pitch_dim, dropout=cfg.dropout)
        self.prosody_head = RegressionHead(cfg.dim, cfg.prosody_dim, dropout=cfg.dropout)
        self.timbre_head = RegressionHead(cfg.dim, cfg.timbre_dim, dropout=cfg.dropout)
        self.style_head = ClassificationHead(cfg.dim, cfg.style_classes, dropout=cfg.dropout)
        self.mode_head = SpeakingModeHead(
            cfg.dim,
            mode_count=len(cfg.mode_labels),
            dataset_adapter_count=cfg.dataset_adapter_count,
            dropout=cfg.dropout,
        )
        self.retrieval_head = RetrievalHead(
            eeg_dim=cfg.dim,
            audio_dim=cfg.audio_embedding_dim,
            proj_dim=cfg.projection_dim,
            queue_size=cfg.retrieval_queue_size,
            queue_negatives=cfg.retrieval_queue_negatives,
            temperature=cfg.retrieval_temperature,
        )

    def route(self, rvq: GroupedRVQOutput, groups: tuple[str, ...]) -> torch.Tensor:
        return self.tokenizer.group_latent(rvq, groups)

    @staticmethod
    def _metadata_labels(value: torch.Tensor | None, prefix: str) -> list[str] | None:
        if value is None:
            return None
        flat = value.detach().cpu().reshape(-1).tolist()
        return [f"{prefix}:{int(item)}" for item in flat]

    def q7_metrics(
        self,
        rvq: GroupedRVQOutput,
        dataset_id: list[str],
        dropout_ablation: torch.Tensor,
        device_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        residual_tokens = rvq.group_tokens["residual"]
        metrics = token_usage_metrics(residual_tokens, self.config.codebook_size)
        residual_embedding = pool_tokens(rvq.group_latents["residual"])
        out = {
            "q7_usage": metrics["usage"],
            "q7_perplexity": metrics["perplexity"],
            "q7_dead_code_ratio": metrics["dead_code_ratio"],
            "q7_dropout_ablation": dropout_ablation.detach(),
            "q7_dataset_predictability": dataset_predictability_proxy(residual_embedding.detach(), dataset_id),
        }
        device_labels = self._metadata_labels(device_id, "device")
        if device_labels is not None:
            out["q7_device_predictability"] = dataset_predictability_proxy(residual_embedding.detach(), device_labels)
        return out

    def forward(self, batch: EEGVoiceBatch) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        token_out = self.tokenizer(
            batch.eeg,
            batch.sensor_pos,
            channel_mask=batch.channel_mask,
            sensor_type=batch.sensor_type,
            acquisition_device_id=batch.acquisition_device_id,
            montage_id=batch.montage_id,
            reference_id=batch.reference_id,
            sampling_rate_hz=batch.sampling_rate_hz,
            native_channel_count=batch.native_channel_count,
        )
        rvq = token_out["rvq"]
        assert isinstance(rvq, GroupedRVQOutput)
        targets = batch.targets or VoiceAlignmentTargets()

        content_z = self.route(rvq, self.head_groups["content"])
        prosody_z = self.route(rvq, self.head_groups["prosody"])
        voice_z = self.route(rvq, self.head_groups["voice"])
        mode_z = self.route(rvq, self.head_groups["mode"])

        content_out = self.content_head(content_z, targets.content_labels)
        phoneme_out = self.phoneme_head(content_z, targets.phoneme_labels)
        pitch_out = self.pitch_head(prosody_z, targets.pitch_target)
        prosody_out = self.prosody_head(prosody_z, targets.prosody_target)
        timbre_out = self.timbre_head(voice_z, targets.timbre_target)
        style_out = self.style_head(voice_z, targets.style_labels)
        mode_out = self.mode_head(mode_z, batch.dataset_id, targets.mode_labels)

        aligned_loss = eeg_reconstruction_loss(token_out["recon_aligned"], token_out["target"])
        full_loss = eeg_reconstruction_loss(token_out["recon_full"], token_out["target"])
        losses: dict[str, torch.Tensor] = {
            "recon_aligned_loss": aligned_loss["loss"],
            "recon_full_loss": full_loss["loss"],
            "vq_loss": rvq.commitment_loss,
        }

        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "tokens": rvq.tokens,
            "group_tokens": rvq.group_tokens,
            "group_names": rvq.group_names,
            "z": rvq.z,
            "z_q": rvq.z_q,
            "device_context": token_out["device_context"],
            "group_latents": rvq.group_latents,
            "recon_aligned": token_out["recon_aligned"],
            "recon_full": token_out["recon_full"],
            "content_logits": content_out["logits"],
            "phoneme_logits": phoneme_out["logits"],
            "pitch_pred": pitch_out["pred"],
            "prosody_pred": prosody_out["pred"],
            "timbre_pred": timbre_out["pred"],
            "style_logits": style_out["logits"],
            "mode_logits": mode_out["logits"],
            "head_groups": self.head_groups,
            "token_metrics": rvq.token_metrics,
            "q7_metrics": self.q7_metrics(
                rvq,
                batch.dataset_id,
                token_out["q7_dropout_ablation"],
                device_id=batch.acquisition_device_id,
            ),
        }

        for name, branch in {
            "content_loss": content_out,
            "phoneme_loss": phoneme_out,
            "pitch_loss": pitch_out,
            "prosody_loss": prosody_out,
            "timbre_loss": timbre_out,
            "style_loss": style_out,
            "mode_loss": mode_out,
        }.items():
            if "loss" in branch:
                losses[name] = branch["loss"]

        if batch.audio_embedding is not None:
            retrieval_out = self.retrieval_head(voice_z, batch.audio_embedding)
            out["retrieval_logits"] = retrieval_out["logits"]
            out["retrieval_eeg_embedding"] = retrieval_out["eeg_embedding"]
            out["retrieval_audio_embedding"] = retrieval_out["audio_embedding"]
            out["retrieval_queue_filled"] = retrieval_out["queue_filled"]
            losses["retrieval_loss"] = retrieval_out["loss"]

        total = (
            losses["recon_aligned_loss"]
            + self.config.q7_full_recon_weight * losses["recon_full_loss"]
            + losses["vq_loss"]
        )
        for name, value in losses.items():
            if name not in {"recon_aligned_loss", "recon_full_loss", "vq_loss"}:
                total = total + value
        losses["loss"] = total
        out["losses"] = losses
        out["loss"] = total
        return out
