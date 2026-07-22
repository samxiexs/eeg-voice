from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import hashlib
import math


def _float_audio(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scale = float(max(abs(info.min), info.max))
        array = array.astype(np.float32) / scale
    else:
        array = array.astype(np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    if array.ndim != 1:
        raise ValueError(f"Expected mono/stereo WAV, got shape {array.shape}")
    return array


def read_wav(path: str | Path, *, start: int = 0, frames: int = -1) -> tuple[np.ndarray, int]:
    rate, raw = wavfile.read(str(path), mmap=True)
    stop = None if int(frames) < 0 else int(start) + int(frames)
    return _float_audio(raw[int(start) : stop]), int(rate)


def wav_info(path: str | Path) -> tuple[int, int]:
    rate, raw = wavfile.read(str(path), mmap=True)
    return int(rate), int(raw.shape[0])


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    wavfile.write(str(destination), int(sample_rate), np.asarray(value * 32767.0, dtype=np.int16))


def canonical_audio_sha256(audio: np.ndarray, source_rate: int, *, target_rate: int = 16000, duration_sec: float = 2.0) -> str:
    value = np.asarray(audio, dtype=np.float32).reshape(-1)
    if int(source_rate) != int(target_rate):
        divisor = math.gcd(int(source_rate), int(target_rate))
        value = resample_poly(value, target_rate // divisor, source_rate // divisor).astype(np.float32)
    target = round(int(target_rate) * float(duration_sec))
    normalized = np.zeros(target, dtype=np.float32)
    normalized[: min(target, len(value))] = value[:target]
    pcm = np.asarray(np.clip(normalized, -1.0, 1.0) * 32767.0, dtype="<i2")
    return hashlib.sha256(pcm.tobytes()).hexdigest()


__all__ = ["canonical_audio_sha256", "read_wav", "wav_info", "write_wav"]
