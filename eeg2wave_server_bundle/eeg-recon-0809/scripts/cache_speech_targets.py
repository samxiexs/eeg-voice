#!/usr/bin/env python3
"""Cache leakage-safe MFCC/HuBERT content and gain-preserving acoustic targets."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from scipy.signal import resample_poly

from prepare_training_data import ROOT, git_provenance, load_config, output_root, sha256_bytes, sha256_file

APP_SRC = ROOT / "app" / "src"
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))
from eeg2speech.speecht5 import CONTRACT as SPEECHT5_NATIVE_CONTRACT, HOP_SAMPLES, native_speecht5_mel


def _pcm_float(value: np.ndarray) -> np.ndarray:
    if np.issubdtype(value.dtype, np.integer):
        info = np.iinfo(value.dtype)
        scale = float(max(abs(info.min), info.max))
        return value.astype(np.float32) / scale
    return value.astype(np.float32)


def load_wave(path: Path, target_rate: int = 16000) -> tuple[np.ndarray, int, int]:
    source_rate, value = wavfile.read(path)
    source_channels = 1 if value.ndim == 1 else value.shape[1]
    value = _pcm_float(value)
    if value.ndim == 2:
        value = value.mean(1)
    if source_rate != target_rate:
        divisor = math.gcd(int(source_rate), int(target_rate))
        value = resample_poly(value, target_rate // divisor, source_rate // divisor).astype(np.float32)
    return value, int(source_rate), int(source_channels)


def _hz_to_mel(frequency: np.ndarray) -> np.ndarray:
    f_min, f_sp = 0.0, 200.0 / 3.0
    mel = (frequency - f_min) / f_sp
    minimum_log_hz = 1000.0
    minimum_log_mel = (minimum_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    logarithmic = frequency >= minimum_log_hz
    mel[logarithmic] = minimum_log_mel + np.log(frequency[logarithmic] / minimum_log_hz) / logstep
    return mel


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    f_min, f_sp = 0.0, 200.0 / 3.0
    frequency = f_min + f_sp * mel
    minimum_log_hz = 1000.0
    minimum_log_mel = (minimum_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    logarithmic = mel >= minimum_log_mel
    frequency[logarithmic] = minimum_log_hz * np.exp(logstep * (mel[logarithmic] - minimum_log_mel))
    return frequency


def slaney_filterbank(sample_rate: int, n_fft: int, n_mels: int, f_min: float, f_max: float) -> torch.Tensor:
    frequencies = np.linspace(0.0, sample_rate / 2.0, 1 + n_fft // 2)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(np.array([f_min]))[0], _hz_to_mel(np.array([f_max]))[0], n_mels + 2))
    lower = (frequencies[None] - edges[:-2, None]) / np.maximum(edges[1:-1] - edges[:-2], 1e-8)[:, None]
    upper = (edges[2:, None] - frequencies[None]) / np.maximum(edges[2:] - edges[1:-1], 1e-8)[:, None]
    filters = np.maximum(0.0, np.minimum(lower, upper))
    filters *= (2.0 / np.maximum(edges[2:] - edges[:-2], 1e-8))[:, None]
    return torch.from_numpy(filters.astype(np.float32))


def _dct(n_mfcc: int, n_mels: int) -> torch.Tensor:
    index = torch.arange(n_mels, dtype=torch.float32)
    basis = torch.cos(math.pi / n_mels * (index + 0.5)[None] * torch.arange(n_mfcc, dtype=torch.float32)[:, None])
    basis[0] *= math.sqrt(1.0 / n_mels)
    basis[1:] *= math.sqrt(2.0 / n_mels)
    return basis


def log_mel(wave: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
    value = torch.from_numpy(np.asarray(wave, dtype=np.float32))
    if len(value) < 400:
        value = F.pad(value, (0, 400 - len(value)))
    spectrum = torch.stft(value, n_fft=400, hop_length=160, win_length=400,
                          window=torch.hann_window(400), center=False, return_complex=True)
    power = spectrum.abs().square()
    return torch.log((slaney_filterbank(sample_rate, 400, 80, 50.0, 7600.0) @ power).clamp_min(1e-10))


def active_crop(wave: np.ndarray, threshold_db: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    if len(wave) < 400:
        return wave, np.ones(1, dtype=bool)
    frames = np.lib.stride_tricks.sliding_window_view(wave, 400)[::160]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    threshold = max(float(rms.max()) * 10 ** (-threshold_db / 20.0), 1e-5)
    active = rms >= threshold
    if not active.any():
        return wave, active
    indices = np.flatnonzero(active)
    start = int(indices[0] * 160)
    end = min(len(wave), int(indices[-1] * 160 + 400))
    return wave[start:end], active


def frame_rms_activity(wave: np.ndarray, threshold_db: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """Return gain-preserving waveform RMS and its activity mask.

    These frames intentionally use the same 400-sample window and 160-sample
    hop as log-mel.  RMS must be measured from waveform samples: summing a
    Slaney-normalized mel bank is not an amplitude-preserving substitute.
    """
    value = np.asarray(wave, dtype=np.float32)
    if len(value) < 400:
        value = np.pad(value, (0, 400 - len(value)))
    frames = np.lib.stride_tricks.sliding_window_view(value, 400)[::160]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12).astype(np.float32)
    threshold = max(float(rms.max()) * 10 ** (-threshold_db / 20.0), 1e-5)
    return rms, rms >= threshold


def content_features(wave: np.ndarray, frames: int = 161) -> tuple[np.ndarray, np.ndarray]:
    centered = wave.astype(np.float32) - float(np.mean(wave))
    cropped, activity = active_crop(centered)
    mel = log_mel(cropped)
    mfcc = (_dct(40, 80) @ mel)[1:]
    mfcc = (mfcc - mfcc.mean(1, keepdim=True)) / mfcc.std(1, keepdim=True).clamp_min(1e-5)
    mfcc = F.interpolate(mfcc.unsqueeze(0), size=frames, mode="linear", align_corners=False).squeeze(0)
    mask = np.ones(frames, dtype=bool) if activity.any() else np.zeros(frames, dtype=bool)
    return mfcc.numpy().astype(np.float32), mask


def load_hubert(config: dict, allow_download: bool):
    try:
        from transformers import HubertModel, Wav2Vec2FeatureExtractor
    except ImportError as exc:
        raise RuntimeError("HuBERT caching requires transformers; MFCC caching works without it") from exc
    content = config["audio"]["content"]
    local = content.get("hubert_local_path")
    local_path = (ROOT / local).resolve() if local else None
    model_name = str(local_path) if local_path and local_path.exists() else content["hubert_model"]
    revision = content.get("hubert_revision", "main")
    kwargs = {"local_files_only": True} if local_path and local_path.exists() else {"revision": revision, "local_files_only": not allow_download}
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, **kwargs)
    model = HubertModel.from_pretrained(model_name, **kwargs).eval()
    return processor, model, model_name


def hubert_features(wave: np.ndarray, config: dict, runtime) -> tuple[np.ndarray, np.ndarray]:
    processor, model, _ = runtime
    inputs = processor(wave, sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=True).hidden_states[int(config["audio"]["content"]["hubert_layer"])]
    local = F.interpolate(output.transpose(1, 2), size=96, mode="linear", align_corners=False).transpose(1, 2).squeeze(0)
    global_ = F.normalize(local.mean(0), dim=0)
    return local.numpy().astype(np.float32), global_.numpy().astype(np.float32)


def cache(config: dict, dataset: str, limit: int | None, include_hubert: bool,
          allow_download: bool, manifest_kind: str = "built",
          target_name: str = "speech_targets") -> Path:
    root = output_root(config)
    if not all(character.isalnum() or character in "_-" for character in manifest_kind + target_name):
        raise ValueError("manifest_kind and target_name must be safe artifact identifiers")
    manifest_path = root / "manifests" / f"manifest_{manifest_kind}.csv"
    if not manifest_path.exists():
        raise RuntimeError(f"requested manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path, keep_default_na=False, low_memory=False)
    lock = json.loads((root / "source_lock.json").read_text())
    selected = manifest[(manifest.build_status == "included") & manifest.supervision_type.isin(["paired_audio", "weak_audio"]) & (manifest.audio_path != "")]
    if dataset != "all":
        selected = selected[selected.dataset == dataset]
    selected = selected.drop_duplicates(["audio_sha256", "audio_semantics"]).sort_values(["dataset", "audio_sha256"])
    if limit is not None:
        selected = selected.head(limit)
    target = root / "speech_targets" / f"{target_name}.h5"
    partial = target.with_suffix(".h5.partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        partial.unlink()
    inventory = []
    hubert_runtime = load_hubert(config, allow_download) if include_hubert else None
    with h5py.File(partial, "w") as output:
        output.attrs["schema_version"] = "speech-targets-v1"
        output.attrs["preprocess_config_sha256"] = config["_config_sha256"]
        output.attrs["source_lock_sha256"] = lock["source_lock_sha256"]
        commit, diff = git_provenance()
        output.attrs["code_commit"] = commit
        output.attrs["code_diff_hash"] = sha256_bytes(diff.encode())
        output.attrs["target_code_sha256"] = sha256_file(Path(__file__))
        output.attrs["native_mel_contract"] = SPEECHT5_NATIVE_CONTRACT
        output.attrs["native_mel_hop_samples"] = HOP_SAMPLES
        output.attrs["hubert_included"] = include_hubert
        if hubert_runtime is not None:
            output.attrs["hubert_source"] = hubert_runtime[2]
            output.attrs["hubert_layer"] = int(config["audio"]["content"]["hubert_layer"])
        for _, row in selected.iterrows():
            source = ROOT / row.audio_path
            if sha256_file(source) != row.audio_sha256:
                raise RuntimeError(f"audio source changed after audit: {source}")
            wave, source_rate, source_channels = load_wave(source)
            mfcc, content_mask = content_features(wave, int(config["audio"]["content"]["relative_frames"]))
            acoustic = log_mel(wave).numpy().astype(np.float32)
            native_mel = native_speecht5_mel(torch.from_numpy(wave).unsqueeze(0)).squeeze(0).cpu().numpy().astype(np.float32)
            rms, acoustic_activity = frame_rms_activity(
                wave, float(config["audio"]["content"]["vad_threshold_db_below_peak"])
            )
            audio_id = str(row.get("audio_id") or f"audio-{row.audio_sha256[:16]}-{row.audio_semantics}")
            group = output.create_group(audio_id)
            group.create_dataset("content_mfcc", data=mfcc, compression="gzip")
            group.create_dataset("content_mask", data=content_mask)
            group.create_dataset("log_mel", data=acoustic, compression="gzip")
            group.create_dataset("rms", data=rms, compression="gzip")
            group.create_dataset("activity", data=acoustic_activity)
            # Native SpeechT5 mel is deliberately ragged.  The loader pads it
            # per batch with an explicit mask; it must never be compressed to
            # the 161-frame relative-content grid before waveform synthesis.
            group.create_dataset("native_speecht5_mel", data=native_mel, compression="gzip")
            group.create_dataset("native_audio_mask", data=np.ones(native_mel.shape[1], dtype=bool))
            group.attrs["native_duration_frames"] = int(native_mel.shape[1])
            if include_hubert:
                centered = wave - wave.mean()
                content_wave, _ = active_crop(
                    centered, float(config["audio"]["content"]["vad_threshold_db_below_peak"])
                )
                local, global_ = hubert_features(content_wave, config, hubert_runtime)
                group.create_dataset("hubert_local", data=local, compression="gzip")
                group.create_dataset("hubert_global", data=global_)
            for key in ("dataset", "pairing_level", "linguistic_content_id", "audio_sha256", "audio_semantics"):
                group.attrs[key] = str(row.get(key, ""))
            group.attrs["source_path"] = str(row.audio_path)
            group.attrs["source_rate_hz"] = source_rate
            group.attrs["source_channels"] = source_channels
            group.attrs["target_rate_hz"] = 16000
            inventory.append({"audio_id": audio_id, "dataset": row.dataset, "pairing_level": row.pairing_level,
                              "audio_sha256": row.audio_sha256, "mfcc_frames": mfcc.shape[1],
                              "acoustic_frames": acoustic.shape[1], "native_speecht5_frames": native_mel.shape[1],
                              "hubert_included": include_hubert})
    os.replace(partial, target)
    pd.DataFrame(inventory).to_csv(target.parent / f"{target_name}_inventory.csv", index=False)
    (target.parent / f"{target_name}.sha256").write_text(sha256_file(target) + "\n")
    print(f"cached {len(inventory)} speech targets at {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "training_data_v3.yaml")
    parser.add_argument("--dataset", choices=["all", "ds004940"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--manifest", default="built")
    parser.add_argument("--target-name", default="speech_targets")
    parser.add_argument("--include-hubert", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--hubert-local-path", type=Path)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    if args.hubert_local_path:
        config["audio"]["content"]["hubert_local_path"] = str(args.hubert_local_path.resolve())
    cache(config, args.dataset, args.limit, args.include_hubert, args.allow_download,
          args.manifest, args.target_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
