from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import FitAudit, LABELS, SplitManifest, TrialRecord, assert_train_only, compute_time_anchor, load_audio, read_trial_records, write_json


@dataclass(frozen=True)
class HubertAdaptationConfig:
    model_path: str
    sample_rate: int = 16000
    top_unfrozen_layers: int = 2
    projection_dim: int = 256
    semantic_steps: int = 50
    semantic_vocab: int = 64


class KaraOneHubertAdapter(nn.Module):
    """A deliberately small domain adapter around a local HuBERT checkpoint."""

    def __init__(self, hubert: nn.Module, hidden_dim: int, n_labels: int = len(LABELS), projection_dim: int = 256):
        super().__init__()
        self.hubert = hubert
        self.projection = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, projection_dim))
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, n_labels))

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        output = self.hubert(input_values=input_values, attention_mask=attention_mask)
        sequence = output.last_hidden_state
        summary = sequence.mean(dim=1)
        return {
            "sequence": sequence,
            "summary": summary,
            "embedding": self.projection(summary),
            "label_logits": self.classifier(summary),
        }


def load_hubert_adapter(config: HubertAdaptationConfig, *, device: torch.device) -> tuple[KaraOneHubertAdapter, Any]:
    from transformers import AutoFeatureExtractor, HubertModel

    extractor = AutoFeatureExtractor.from_pretrained(config.model_path, local_files_only=True)
    hubert = HubertModel.from_pretrained(config.model_path, local_files_only=True)
    freeze_hubert_bottom_layers(hubert, config.top_unfrozen_layers)
    model = KaraOneHubertAdapter(hubert, hidden_dim=int(hubert.config.hidden_size), projection_dim=config.projection_dim).to(device)
    return model, extractor


def freeze_hubert_bottom_layers(hubert: nn.Module, top_unfrozen_layers: int = 2) -> list[str]:
    """Freeze all HuBERT parameters except the top transformer blocks and layer norms."""
    for parameter in hubert.parameters():
        parameter.requires_grad = False
    layers = list(getattr(getattr(hubert, "encoder"), "layers"))
    if not 0 < int(top_unfrozen_layers) <= len(layers):
        raise ValueError(f"top_unfrozen_layers must be in [1, {len(layers)}]")
    trainable = []
    for layer in layers[-int(top_unfrozen_layers) :]:
        for name, parameter in layer.named_parameters():
            parameter.requires_grad = True
            trainable.append(f"encoder.{len(layers) - int(top_unfrozen_layers)}+.{name}")
    for module_name, module in hubert.named_modules():
        if isinstance(module, nn.LayerNorm):
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad = True
                trainable.append(f"{module_name}.layernorm")
    return sorted(set(trainable))


def audio_augment(audio: torch.Tensor, *, noise_std: float = 0.005, max_mask_samples: int = 1600) -> torch.Tensor:
    """Content-preserving waveform augmentation used only during audio SSL."""
    out = audio + torch.randn_like(audio) * float(noise_std)
    if max_mask_samples > 0 and out.shape[1] > 1:
        width = min(int(max_mask_samples), max(1, out.shape[1] // 10))
        start = torch.randint(0, max(1, out.shape[1] - width + 1), (out.shape[0],), device=out.device)
        for idx, value in enumerate(start.tolist()):
            out[idx, value : value + width] = 0.0
    gain = 0.95 + 0.10 * torch.rand((out.shape[0], 1), device=out.device, dtype=out.dtype)
    return (out * gain).clamp(-1.0, 1.0)


def mean_pool_to_steps(sequence: torch.Tensor, steps: int) -> torch.Tensor:
    return F.adaptive_avg_pool1d(sequence.transpose(1, 2), int(steps)).transpose(1, 2)


def kmeans_train_only(values: np.ndarray, n_clusters: int, *, iterations: int = 50, seed: int = 11) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or len(values) < 1:
        raise ValueError(f"kmeans expects [N,D], got {values.shape}")
    n_clusters = min(max(2, int(n_clusters)), values.shape[0])
    rng = np.random.default_rng(seed)
    centers = values[rng.choice(values.shape[0], n_clusters, replace=False)].copy()
    for _ in range(int(iterations)):
        ids = nearest_centers(values, centers)
        for cluster in range(n_clusters):
            matches = values[ids == cluster]
            if len(matches):
                centers[cluster] = matches.mean(axis=0)
    return centers.astype(np.float32)


def nearest_centers(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    distance = (values[:, None, :] - centers[None, :, :]).square().sum(axis=-1)
    return distance.argmin(axis=1).astype(np.int64)


@torch.no_grad()
def build_adapted_audio_cache(
    *,
    root: str | Path,
    manifest: SplitManifest,
    adapter: KaraOneHubertAdapter,
    feature_extractor: Any,
    output_path: str | Path,
    audit_path: str | Path,
    device: torch.device,
    stage: str = "overt_like",
    semantic_steps: int = 50,
    semantic_vocab: int = 64,
    batch_size: int = 4,
    codec_model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze the selected HuBERT adapter and make a fully auditable target cache.

    The cache has all rows for diagnostics, but its codebook is fitted only on
    subject_train.  Holding audio targets for P02/MM21 is evaluation data, not fit data.
    """
    root = Path(root)
    output_path = Path(output_path)
    records = read_trial_records(root)
    train_records = [row for row in records if manifest.split_for(row.subject) == "subject_train"]
    assert_train_only(train_records, manifest, "adapted_hubert_codebook")
    adapter.eval()
    sequences, summaries, labels, subjects, trial_indices, audio_paths, anchors = [], [], [], [], [], [], []
    for start in range(0, len(records), int(batch_size)):
        chunk = records[start : start + int(batch_size)]
        waveform = torch.from_numpy(np.stack([load_audio(root / row.audio_path) for row in chunk])).to(device)
        inputs = feature_extractor(waveform.detach().cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        values = inputs["input_values"].to(device)
        mask = inputs.get("attention_mask")
        out = adapter(values, attention_mask=mask.to(device) if mask is not None else None)
        seq = mean_pool_to_steps(out["sequence"], semantic_steps).detach().cpu().numpy().astype(np.float32)
        summary = out["summary"].detach().cpu().numpy().astype(np.float32)
        sequences.append(seq)
        summaries.append(summary)
        labels.extend(row.label for row in chunk)
        subjects.extend(row.subject for row in chunk)
        trial_indices.extend(row.trial_index for row in chunk)
        audio_paths.extend(row.audio_path for row in chunk)
        anchors.extend(compute_time_anchor(load_audio(root / row.audio_path)) for row in chunk)
    sequence_array = np.concatenate(sequences, axis=0)
    summary_array = np.concatenate(summaries, axis=0)
    train_mask = np.asarray([subject in manifest.train_subjects for subject in subjects], dtype=bool)
    centers = kmeans_train_only(sequence_array[train_mask].reshape(-1, sequence_array.shape[-1]), semantic_vocab)
    semantic_ids = nearest_centers(sequence_array.reshape(-1, sequence_array.shape[-1]), centers).reshape(sequence_array.shape[:2])
    codec_latents = _extract_encodec_latents(root, records, codec_model_path) if codec_model_path else None
    payload: dict[str, Any] = {
        "version": np.asarray("0711v1"),
        "stage": np.asarray(stage),
        "keys": np.asarray([f"{subject}:{trial}" for subject, trial in zip(subjects, trial_indices)]),
        "subjects": np.asarray(subjects),
        "trial_indices": np.asarray(trial_indices, dtype=np.int32),
        "labels": np.asarray(labels),
        "audio_paths": np.asarray(audio_paths),
        "fit_split": train_mask,
        "semantic_sequence": sequence_array,
        "semantic_summary": summary_array,
        "semantic_token_ids": semantic_ids.astype(np.int64),
        "semantic_token_mask": np.ones_like(semantic_ids, dtype=np.float32),
        "semantic_codebook": centers,
        "time_active_mask": np.stack([np.asarray(item["active_mask"], dtype=np.float32) for item in anchors]),
        "time_envelope": np.stack([np.asarray(item["envelope"], dtype=np.float32) for item in anchors]),
        "time_onset_sec": np.asarray([item["onset_sec"] for item in anchors], dtype=np.float32),
        "time_duration_sec": np.asarray([item["duration_sec"] for item in anchors], dtype=np.float32),
    }
    if codec_latents is not None:
        payload["encodec_latent"] = codec_latents
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    audit = FitAudit.from_records("adapted_hubert_semantic_codebook", manifest, train_records)
    audit_payload = {**audit.to_dict(), "cache": str(output_path), "semantic_vocab": int(centers.shape[0]), "semantic_steps": int(semantic_steps)}
    write_json(audit_path, audit_payload)
    return audit_payload


def _extract_encodec_latents(root: Path, records: list[TrialRecord], codec_model_path: str | Path | None) -> np.ndarray:
    """Use frozen local EnCodec only as a target encoder; no audio enters EEG inference."""
    from src.audio_features import AudioFeatureConfig, load_codec_backend

    cfg = AudioFeatureConfig(
        sample_rate=16000,
        duration_sec=2.0,
        target_kind="encodec_latent",
        backend="encodec_latent",
        codec_model_name_or_path=str(codec_model_path),
        local_files_only=True,
        codec_bandwidth=6.0,
    )
    backend = load_codec_backend(cfg)
    values = [backend.extract(load_audio(root / row.audio_path), 16000)["target_sequence"] for row in records]
    return np.stack(values).astype(np.float32)


def save_audio_adapter(path: str | Path, adapter: KaraOneHubertAdapter, config: HubertAdaptationConfig, audit: FitAudit) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": adapter.state_dict(), "config": asdict(config), "fit_audit": audit.to_dict()}, path)
    write_json(path.with_suffix(".audit.json"), audit.to_dict())
