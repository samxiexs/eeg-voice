from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import stft

from .utils import ensure_dir, load_wav_fixed, read_csv_rows, resolve_feis_root


@dataclass(frozen=True)
class AudioFeatureConfig:
    sample_rate: int = 16000
    duration_sec: float = 1.0
    normalize: str = "rms"
    target_rms: float = 0.08
    max_gain: float = 10.0
    backend: str = "auto"
    hubert_model_name_or_path: str = "facebook/hubert-base-ls960"
    local_files_only: bool = True
    spectral_bins: int = 48


def _normalize_subject_id(subject_id: str | int) -> str:
    text = str(subject_id)
    return text.zfill(2) if text.isdigit() else text


def _safe_log(value: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return np.log(np.maximum(value, eps))


def estimate_pitch_hz(audio: np.ndarray, sample_rate: int, fmin: float = 60.0, fmax: float = 400.0) -> float:
    if not np.any(audio):
        return 0.0
    centered = audio.astype(np.float64) - float(np.mean(audio))
    if np.sqrt(np.mean(centered**2)) < 1e-5:
        return 0.0
    corr = np.correlate(centered, centered, mode="full")
    corr = corr[corr.size // 2 :]
    min_lag = max(1, int(sample_rate / fmax))
    max_lag = min(len(corr) - 1, int(sample_rate / fmin))
    if max_lag <= min_lag:
        return 0.0
    window = corr[min_lag:max_lag]
    if window.size == 0:
        return 0.0
    lag = int(np.argmax(window)) + min_lag
    if corr[lag] <= 0:
        return 0.0
    return float(sample_rate / lag)


def compute_spectral_embedding(audio: np.ndarray, sample_rate: int, spectral_bins: int = 48) -> np.ndarray:
    _, _, spec = stft(audio, fs=sample_rate, nperseg=400, noverlap=240, nfft=512, boundary=None, padded=False)
    mag = np.abs(spec).astype(np.float32)
    if mag.size == 0:
        return np.zeros(spectral_bins * 4, dtype=np.float32)
    log_mag = _safe_log(mag)
    freq_axis = np.linspace(0, log_mag.shape[0] - 1, num=spectral_bins).astype(np.int32)
    reduced = log_mag[freq_axis]
    delta = np.diff(reduced, axis=1, prepend=reduced[:, :1])
    pooled = np.concatenate(
        [
            reduced.mean(axis=1),
            reduced.std(axis=1),
            delta.mean(axis=1),
            delta.std(axis=1),
        ],
        axis=0,
    )
    return pooled.astype(np.float32)


def compute_prosody_target(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)) + 1e-8)
    energy_db = float(np.log(rms + 1e-5))
    peak = float(np.max(np.abs(audio)) if audio.size else 0.0)
    pitch_hz = estimate_pitch_hz(audio, sample_rate=sample_rate)
    duration_sec = float(len(audio) / sample_rate)
    return np.asarray([pitch_hz, energy_db, duration_sec, peak], dtype=np.float32)


class _LocalHubertEmbedder:
    def __init__(self, model_name_or_path: str, local_files_only: bool = True):
        from transformers import AutoFeatureExtractor, AutoModel

        self.extractor = AutoFeatureExtractor.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        self.model.eval()

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        with torch.no_grad():
            inputs = self.extractor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            hidden = self.model(**inputs).last_hidden_state
            embedding = hidden.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        return embedding


def build_audio_feature_backend(config: AudioFeatureConfig) -> tuple[str, Any]:
    backend = str(config.backend)
    if backend in {"auto", "hubert_local"}:
        try:
            return "hubert_local", _LocalHubertEmbedder(
                model_name_or_path=config.hubert_model_name_or_path,
                local_files_only=config.local_files_only,
            )
        except Exception:
            if backend == "hubert_local":
                raise
    return "spectral_fallback_v1", None


def load_template_rows(feis_root: str | Path) -> list[dict[str, str]]:
    root = resolve_feis_root(feis_root)
    rows = read_csv_rows(root / "trials.csv")
    unique_rows: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        subject_id = _normalize_subject_id(row["subject_id"])
        label = str(row["label"])
        key = (subject_id, label)
        if key not in unique_rows:
            copied = dict(row)
            copied["subject_id"] = subject_id
            copied["template_id"] = f"{subject_id}:{label}"
            unique_rows[key] = copied
    return sorted(unique_rows.values(), key=lambda item: (item["subject_id"], item["label"]))


def extract_template_audio_features(
    feis_root: str | Path,
    output_path: str | Path,
    config: AudioFeatureConfig,
) -> dict[str, Any]:
    root = resolve_feis_root(feis_root)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    template_rows = load_template_rows(root)
    backend_name, backend = build_audio_feature_backend(config)
    feature_backend = []
    template_ids: list[str] = []
    subject_ids: list[str] = []
    labels: list[str] = []
    audio_paths: list[str] = []
    speech_embeddings: list[np.ndarray] = []
    prosody_targets: list[np.ndarray] = []

    for row in template_rows:
        relpath = str(row["audio_path"])
        audio = load_wav_fixed(
            root / relpath,
            sample_rate=config.sample_rate,
            n_samples=int(round(config.sample_rate * config.duration_sec)),
            normalize=config.normalize,
            target_rms=config.target_rms,
            max_gain=config.max_gain,
        )
        embedding = (
            backend.embed(audio, sample_rate=config.sample_rate)
            if backend is not None
            else compute_spectral_embedding(audio, sample_rate=config.sample_rate, spectral_bins=config.spectral_bins)
        )
        prosody = compute_prosody_target(audio, sample_rate=config.sample_rate)
        template_ids.append(str(row["template_id"]))
        subject_ids.append(str(row["subject_id"]))
        labels.append(str(row["label"]))
        audio_paths.append(relpath)
        speech_embeddings.append(np.asarray(embedding, dtype=np.float32))
        prosody_targets.append(np.asarray(prosody, dtype=np.float32))
        feature_backend.append(backend_name)

    payload = {
        "template_ids": np.asarray(template_ids),
        "subject_ids": np.asarray(subject_ids),
        "labels": np.asarray(labels),
        "audio_paths": np.asarray(audio_paths),
        "speech_embeddings": np.stack(speech_embeddings, axis=0).astype(np.float32),
        "prosody_targets": np.stack(prosody_targets, axis=0).astype(np.float32),
        "feature_backend": np.asarray(feature_backend),
    }
    np.savez_compressed(output_path, **payload)
    metadata = {
        "output_path": str(output_path),
        "num_templates": len(template_ids),
        "embedding_dim": int(payload["speech_embeddings"].shape[1]),
        "prosody_dim": int(payload["prosody_targets"].shape[1]),
        "feature_backend": backend_name,
        "sample_rate": config.sample_rate,
        "duration_sec": config.duration_sec,
        "normalize": config.normalize,
        "hubert_model_name_or_path": config.hubert_model_name_or_path,
        "local_files_only": config.local_files_only,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata
