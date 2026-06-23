from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import stft
from tqdm import tqdm

from .utils import ensure_dir, load_wav_fixed, pad_or_crop_audio, read_csv_rows, resample_audio, resolve_feis_root


TARGET_KIND_HUBERT_POOLED = "hubert_pooled"
TARGET_KIND_HUBERT_SEQUENCE = "hubert_sequence"
TARGET_KIND_ENCODEC_LATENT = "encodec_latent"


@dataclass(frozen=True)
class AudioFeatureConfig:
    sample_rate: int = 16000
    duration_sec: float = 1.0
    normalize: str = "rms"
    target_rms: float = 0.08
    max_gain: float = 10.0
    backend: str = "auto"
    target_kind: str = TARGET_KIND_HUBERT_POOLED
    ssl_model_name_or_path: str = "facebook/hubert-base-ls960"
    codec_model_name_or_path: str = "facebook/encodec_24khz"
    local_files_only: bool = True
    spectral_bins: int = 48
    sequence_target_steps: int = 16
    codec_bandwidth: float = 6.0


def _normalize_subject_id(subject_id: str | int) -> str:
    text = str(subject_id)
    return text.zfill(2) if text.isdigit() else text


def _safe_log(value: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return np.log(np.maximum(value, eps))


def _average_pool_hidden(hidden: torch.Tensor, target_steps: int) -> np.ndarray:
    hidden = hidden.transpose(0, 1).unsqueeze(0)
    pooled = F.adaptive_avg_pool1d(hidden, int(target_steps)).squeeze(0).transpose(0, 1)
    return pooled.cpu().numpy().astype(np.float32)


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


class _LocalSSLModelEmbedder:
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

    def hidden(self, audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.extractor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            hidden = self.model(**inputs).last_hidden_state.squeeze(0)
        return hidden

    def pooled(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        hidden = self.hidden(audio, sample_rate=sample_rate)
        return hidden.mean(dim=0).cpu().numpy().astype(np.float32)

    def sequence(self, audio: np.ndarray, sample_rate: int, target_steps: int) -> np.ndarray:
        hidden = self.hidden(audio, sample_rate=sample_rate)
        return _average_pool_hidden(hidden, target_steps=target_steps)


class _EncodecLatentBackend:
    def __init__(
        self,
        model_name_or_path: str,
        local_files_only: bool = True,
        bandwidth: float = 6.0,
        duration_sec: float = 1.0,
    ):
        from transformers import EncodecModel

        self.model = EncodecModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        self.model.eval()
        self.sample_rate = int(self.model.config.sampling_rate)
        self.audio_channels = int(self.model.config.audio_channels)
        self.duration_sec = float(duration_sec)
        self.bandwidth = float(bandwidth)
        supported = [float(item) for item in self.model.config.target_bandwidths]
        if self.bandwidth not in supported:
            raise ValueError(f"Unsupported EnCodec bandwidth {self.bandwidth}; expected one of {supported}")

    def _prepare_input(self, audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        resampled = resample_audio(audio, src_sr=int(sample_rate), dst_sr=self.sample_rate)
        target_len = int(round(self.sample_rate * self.duration_sec))
        resampled = pad_or_crop_audio(resampled, target_len=target_len)
        return torch.from_numpy(resampled).float().view(1, self.audio_channels, -1)

    def extract(self, audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        input_values = self._prepare_input(audio, sample_rate=sample_rate)
        padding_mask = torch.ones_like(input_values, dtype=torch.bool)
        with torch.no_grad():
            encoded = self.model.encode(
                input_values=input_values,
                padding_mask=padding_mask,
                bandwidth=self.bandwidth,
                return_dict=True,
            )
            audio_codes = encoded.audio_codes
            audio_scales = encoded.audio_scales
            if audio_codes is None:
                raise RuntimeError("EnCodec encode() returned no audio_codes")
            if audio_codes.ndim != 4 or audio_codes.shape[0] != 1:
                raise ValueError(f"Expected EnCodec audio_codes shape [1, B, Q, T], got {tuple(audio_codes.shape)}")
            frame_codes = audio_codes[0]
            embeddings = self.model.quantizer.decode(frame_codes.transpose(0, 1))
        sequence = embeddings.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.float32)
        summary = sequence.mean(axis=0).astype(np.float32)
        decoder_scales = np.ones((1,), dtype=np.float32)
        if audio_scales is not None and len(audio_scales) > 0 and audio_scales[0] is not None:
            decoder_scales = audio_scales[0].reshape(-1).cpu().numpy().astype(np.float32)
        return {
            "target_sequence": sequence,
            "target_mask": np.ones(sequence.shape[0], dtype=np.float32),
            "target_summary": summary,
            "decoder_scales": decoder_scales,
        }

    def decode(self, target_sequence: np.ndarray, decoder_scales: np.ndarray | None = None) -> np.ndarray:
        latents = torch.from_numpy(np.asarray(target_sequence, dtype=np.float32)).transpose(0, 1).unsqueeze(0)
        with torch.no_grad():
            audio = self.model.decoder(latents)
            if decoder_scales is not None:
                scale = torch.from_numpy(np.asarray(decoder_scales, dtype=np.float32)).view(-1, 1, 1)
                audio = audio * scale
        return audio.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def build_audio_feature_backend(config: AudioFeatureConfig) -> tuple[str, Any]:
    target_kind = str(config.target_kind)
    backend = str(config.backend)
    if target_kind == TARGET_KIND_ENCODEC_LATENT or backend == "encodec_latent":
        return "encodec_latent", _EncodecLatentBackend(
            model_name_or_path=config.codec_model_name_or_path,
            local_files_only=config.local_files_only,
            bandwidth=config.codec_bandwidth,
            duration_sec=config.duration_sec,
        )
    if backend in {"auto", "ssl_local", "hubert_local", "wav2vec2_local"}:
        return "ssl_local", _LocalSSLModelEmbedder(
            model_name_or_path=config.ssl_model_name_or_path,
            local_files_only=config.local_files_only,
        )
    if target_kind != TARGET_KIND_HUBERT_POOLED:
        raise ValueError(f"Unsupported backend {backend} for target_kind={target_kind}")
    return "spectral_fallback_v1", None


def load_codec_backend(config: AudioFeatureConfig) -> _EncodecLatentBackend:
    backend_name, backend = build_audio_feature_backend(config)
    if backend_name != "encodec_latent" or backend is None:
        raise ValueError("AudioFeatureConfig does not resolve to an EnCodec backend")
    return backend


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


def _extract_target_from_audio(
    backend_name: str,
    backend: Any,
    audio: np.ndarray,
    config: AudioFeatureConfig,
) -> dict[str, np.ndarray]:
    target_kind = str(config.target_kind)
    if target_kind == TARGET_KIND_HUBERT_SEQUENCE:
        if backend is None or not isinstance(backend, _LocalSSLModelEmbedder):
            raise RuntimeError("Sequence-level HuBERT targets require a local SSL backend")
        sequence = backend.sequence(audio, sample_rate=config.sample_rate, target_steps=config.sequence_target_steps)
        return {
            "target_sequence": sequence,
            "target_mask": np.ones(sequence.shape[0], dtype=np.float32),
            "target_summary": sequence.mean(axis=0).astype(np.float32),
        }
    if target_kind == TARGET_KIND_ENCODEC_LATENT:
        if backend is None or not isinstance(backend, _EncodecLatentBackend):
            raise RuntimeError("Codec latent targets require an EnCodec backend")
        return backend.extract(audio, sample_rate=config.sample_rate)
    if backend is not None and isinstance(backend, _LocalSSLModelEmbedder):
        summary = backend.pooled(audio, sample_rate=config.sample_rate)
    else:
        summary = compute_spectral_embedding(audio, sample_rate=config.sample_rate, spectral_bins=config.spectral_bins)
    return {
        "target_sequence": summary.reshape(1, -1).astype(np.float32),
        "target_mask": np.ones(1, dtype=np.float32),
        "target_summary": summary.astype(np.float32),
    }


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
    template_ids: list[str] = []
    subject_ids: list[str] = []
    labels: list[str] = []
    audio_paths: list[str] = []
    speech_embeddings: list[np.ndarray] = []
    target_sequences: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    target_summaries: list[np.ndarray] = []
    prosody_targets: list[np.ndarray] = []
    target_rms_values: list[float] = []
    target_log_rms_values: list[float] = []
    feature_backend: list[str] = []
    decoder_scales: list[np.ndarray] = []

    for row in tqdm(template_rows, desc=f"extract {config.target_kind}", unit="template"):
        relpath = str(row["audio_path"])
        audio = load_wav_fixed(
            root / relpath,
            sample_rate=config.sample_rate,
            n_samples=int(round(config.sample_rate * config.duration_sec)),
            normalize=config.normalize,
            target_rms=config.target_rms,
            max_gain=config.max_gain,
        )
        audio_rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)) + 1e-8)
        target = _extract_target_from_audio(backend_name=backend_name, backend=backend, audio=audio, config=config)
        prosody = compute_prosody_target(audio, sample_rate=config.sample_rate)
        template_ids.append(str(row["template_id"]))
        subject_ids.append(str(row["subject_id"]))
        labels.append(str(row["label"]))
        audio_paths.append(relpath)
        target_sequences.append(np.asarray(target["target_sequence"], dtype=np.float32))
        target_masks.append(np.asarray(target["target_mask"], dtype=np.float32))
        target_summaries.append(np.asarray(target["target_summary"], dtype=np.float32))
        speech_embeddings.append(np.asarray(target["target_summary"], dtype=np.float32))
        prosody_targets.append(np.asarray(prosody, dtype=np.float32))
        target_rms_values.append(audio_rms)
        target_log_rms_values.append(float(np.log(audio_rms + 1e-8)))
        feature_backend.append(backend_name)
        if "decoder_scales" in target:
            decoder_scales.append(np.asarray(target["decoder_scales"], dtype=np.float32))

    if not target_sequences:
        raise ValueError("No audio templates were extracted")
    target_shape = {tuple(item.shape) for item in target_sequences}
    if len(target_shape) != 1:
        raise ValueError(f"Expected a fixed target shape across templates, got {sorted(target_shape)}")
    mask_shape = {tuple(item.shape) for item in target_masks}
    if len(mask_shape) != 1:
        raise ValueError(f"Expected a fixed target mask shape across templates, got {sorted(mask_shape)}")
    stacked_sequences = np.stack(target_sequences, axis=0).astype(np.float32)
    target_mean = stacked_sequences.mean(axis=(0, 1)).astype(np.float32)
    target_std = stacked_sequences.std(axis=(0, 1)).astype(np.float32)
    target_std = np.maximum(target_std, 1e-6).astype(np.float32)

    payload: dict[str, np.ndarray] = {
        "template_ids": np.asarray(template_ids),
        "subject_ids": np.asarray(subject_ids),
        "labels": np.asarray(labels),
        "audio_paths": np.asarray(audio_paths),
        "speech_embeddings": np.stack(speech_embeddings, axis=0).astype(np.float32),
        "target_sequences": stacked_sequences,
        "target_masks": np.stack(target_masks, axis=0).astype(np.float32),
        "target_summaries": np.stack(target_summaries, axis=0).astype(np.float32),
        "prosody_targets": np.stack(prosody_targets, axis=0).astype(np.float32),
        "target_mean": target_mean,
        "target_std": target_std,
        "target_rms": np.asarray(target_rms_values, dtype=np.float32),
        "target_log_rms": np.asarray(target_log_rms_values, dtype=np.float32),
        "feature_backend": np.asarray(feature_backend),
        "target_kind": np.asarray(str(config.target_kind)),
        "target_steps": np.asarray(int(target_sequences[0].shape[0]), dtype=np.int32),
        "target_dim": np.asarray(int(target_sequences[0].shape[1]), dtype=np.int32),
        "target_sample_rate": np.asarray(
            int(getattr(backend, "sample_rate", config.sample_rate)),
            dtype=np.int32,
        ),
    }
    default_decoder_scales = None
    if decoder_scales:
        scale_shapes = {tuple(item.shape) for item in decoder_scales}
        if len(scale_shapes) != 1:
            raise ValueError(f"Expected fixed decoder scale shape, got {sorted(scale_shapes)}")
        payload["decoder_scales"] = np.stack(decoder_scales, axis=0).astype(np.float32)
        default_decoder_scales = np.mean(payload["decoder_scales"], axis=0).astype(np.float32)
        payload["default_decoder_scales"] = default_decoder_scales
    np.savez_compressed(output_path, **payload)

    metadata = {
        "output_path": str(output_path),
        "num_templates": len(template_ids),
        "target_kind": str(config.target_kind),
        "target_shape": list(payload["target_sequences"].shape[1:]),
        "embedding_dim": int(payload["speech_embeddings"].shape[1]),
        "prosody_dim": int(payload["prosody_targets"].shape[1]),
        "feature_backend": backend_name,
        "sample_rate": config.sample_rate,
        "duration_sec": config.duration_sec,
        "normalize": config.normalize,
        "ssl_model_name_or_path": config.ssl_model_name_or_path,
        "codec_model_name_or_path": config.codec_model_name_or_path,
        "codec_bandwidth": config.codec_bandwidth,
        "local_files_only": config.local_files_only,
        "default_decoder_scales": None if default_decoder_scales is None else default_decoder_scales.tolist(),
        "target_mean_shape": list(target_mean.shape),
        "target_std_min": float(target_std.min()) if target_std.size else None,
        "target_std_max": float(target_std.max()) if target_std.size else None,
        "target_rms_mean": float(np.mean(target_rms_values)) if target_rms_values else None,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata
