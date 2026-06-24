from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class MelConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    target_frames: int = 64
    f_min: float = 50.0
    f_max: float = 7600.0
    eps: float = 1e-5


def hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (np.power(10.0, np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(cfg: MelConfig) -> np.ndarray:
    n_freq = cfg.n_fft // 2 + 1
    f_max = min(float(cfg.f_max), cfg.sample_rate / 2.0)
    mel_points = np.linspace(hz_to_mel(cfg.f_min), hz_to_mel(f_max), cfg.n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((cfg.n_fft + 1) * hz_points / cfg.sample_rate).astype(int)
    bins = np.clip(bins, 0, n_freq - 1)
    fb = np.zeros((cfg.n_mels, n_freq), dtype=np.float32)
    for mel_idx in range(cfg.n_mels):
        left, center, right = bins[mel_idx], bins[mel_idx + 1], bins[mel_idx + 2]
        if center <= left:
            center = min(left + 1, n_freq - 1)
        if right <= center:
            right = min(center + 1, n_freq)
        if center > left:
            fb[mel_idx, left:center] = (np.arange(left, center) - left) / max(center - left, 1)
        if right > center:
            fb[mel_idx, center:right] = (right - np.arange(center, right)) / max(right - center, 1)
    return fb


def _fix_frames(mat: np.ndarray, target_frames: int) -> np.ndarray:
    if mat.shape[0] == target_frames:
        return mat.astype(np.float32)
    if mat.shape[0] > target_frames:
        return mat[:target_frames].astype(np.float32)
    pad = np.repeat(mat[-1:, :], target_frames - mat.shape[0], axis=0) if mat.size else np.zeros((target_frames, mat.shape[1]), np.float32)
    return np.concatenate([mat, pad], axis=0).astype(np.float32)


def wav_to_logmel(audio: np.ndarray, cfg: MelConfig) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    _, _, z = signal.stft(
        audio,
        fs=cfg.sample_rate,
        window="hann",
        nperseg=cfg.n_fft,
        noverlap=cfg.n_fft - cfg.hop_length,
        nfft=cfg.n_fft,
        boundary="zeros",
        padded=True,
    )
    power = np.maximum(np.abs(z) ** 2, cfg.eps)
    mel = mel_filterbank(cfg) @ power
    logmel = np.log(np.maximum(mel, cfg.eps)).T
    return _fix_frames(logmel, cfg.target_frames)


def logmel_to_wav(logmel: np.ndarray, cfg: MelConfig, iters: int = 48, seed: int = 0) -> np.ndarray:
    logmel = np.asarray(logmel, dtype=np.float32)
    fb = mel_filterbank(cfg)
    pinv = np.linalg.pinv(fb).astype(np.float32)
    mel_power = np.exp(logmel.T).astype(np.float32)
    linear_mag = np.sqrt(np.maximum(pinv @ mel_power, cfg.eps)).astype(np.float32)
    rng = np.random.default_rng(seed)
    phase = np.exp(2j * np.pi * rng.random(linear_mag.shape))
    spec = linear_mag * phase
    audio = np.zeros(1, dtype=np.float32)
    for _ in range(max(int(iters), 1)):
        _, audio = signal.istft(
            spec,
            fs=cfg.sample_rate,
            window="hann",
            nperseg=cfg.n_fft,
            noverlap=cfg.n_fft - cfg.hop_length,
            nfft=cfg.n_fft,
            input_onesided=True,
            boundary=True,
        )
        _, _, z = signal.stft(
            audio.astype(np.float32),
            fs=cfg.sample_rate,
            window="hann",
            nperseg=cfg.n_fft,
            noverlap=cfg.n_fft - cfg.hop_length,
            nfft=cfg.n_fft,
            boundary="zeros",
            padded=True,
        )
        phase = z[:, : linear_mag.shape[1]]
        phase = phase / np.maximum(np.abs(phase), 1e-8)
        spec = linear_mag * phase
    target_len = int(round(cfg.sample_rate * (cfg.target_frames - 1) * cfg.hop_length / cfg.sample_rate))
    target_len = max(target_len, cfg.sample_rate)
    if audio.shape[0] < cfg.sample_rate:
        audio = np.pad(audio, (0, cfg.sample_rate - audio.shape[0]))
    return audio[: cfg.sample_rate].astype(np.float32)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)) + 1e-12)


def scale_to_rms(audio: np.ndarray, target_rms: float, max_gain: float = 20.0) -> np.ndarray:
    gain = min(float(target_rms) / rms(audio), float(max_gain))
    return (np.asarray(audio, dtype=np.float32) * gain).clip(-0.95, 0.95)


def pearson_flat(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom < 1e-12:
        return float("nan")
    return float((av @ bv) / denom)

