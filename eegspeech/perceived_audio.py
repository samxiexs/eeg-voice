"""Frozen speech teachers, native Vocos features and differentiable decoding."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
from .utils import read_json


def read_wave(path: Path, rate: int = 24000) -> np.ndarray:
    wave, original_rate = sf.read(path, dtype="float32", always_2d=True)
    wave = wave.mean(1)
    if original_rate != rate:
        from math import gcd
        g = gcd(original_rate, rate)
        wave = resample_poly(wave, rate // g, original_rate // g).astype(np.float32)
    if not np.isfinite(wave).all(): raise ValueError(f"Nonfinite audio: {path}")
    return wave


def acoustic_features(wave: torch.Tensor, rate: int = 24000) -> torch.Tensor:
    """25 ms RMS + positive onset, [B,time64,2], with finite input gradients."""
    if wave.ndim == 1: wave = wave.unsqueeze(0)
    width, hop = round(rate * .025), rate // 64
    if rate % 64: raise ValueError("Acoustic feature sample rate must divide 64 Hz")
    wave = wave.float()
    padded = F.pad(wave.unsqueeze(1), (width // 2, width // 2), mode="reflect").squeeze(1)
    chunks = padded.unfold(-1, width, hop)
    envelope = (chunks.square().mean(-1) + 1e-8).sqrt()
    envelope = envelope[:, :wave.shape[-1] // hop]
    onset = F.relu(envelope[:, 1:] - envelope[:, :-1])
    onset = F.pad(onset, (1, 0))
    return torch.stack((envelope, onset), dim=-1)


def interpolate_features(values: torch.Tensor, source_times: torch.Tensor, target_times: torch.Tensor) -> torch.Tensor:
    """Linear sampling [B,T,D] on explicit time coordinates, not array lengths."""
    if source_times.ndim != 1 or target_times.ndim != 1: raise ValueError("Expected common 1D time grids")
    if len(source_times) < 2: raise ValueError("Interpolation needs two source frames")
    right = torch.searchsorted(source_times.contiguous(), target_times.contiguous()).clamp(1, len(source_times) - 1)
    left = right - 1
    fraction = ((target_times - source_times[left]) / (source_times[right] - source_times[left])).clamp(0, 1)
    return values[:, left] * (1 - fraction[None, :, None]) + values[:, right] * fraction[None, :, None]


class Standardizer(nn.Module):
    def __init__(self, stats: dict):
        super().__init__()
        self.register_buffer("mean", torch.tensor(stats["mean"], dtype=torch.float32))
        self.register_buffer("std", torch.tensor(stats["std"], dtype=torch.float32))

    def forward(self, x): return (x - self.mean) / self.std
    def inverse(self, x): return x * self.std + self.mean


class FrozenVocos(nn.Module):
    """Freeze weights, preserve the derivative with respect to native log-Mel."""
    def __init__(self, path: Path):
        super().__init__()
        from vocos import Vocos
        self.model = Vocos.from_hparams(str(path / "config.yaml"))
        weights = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights)
        self.model.requires_grad_(False).eval()
        frontend = self.model.feature_extractor
        mel = frontend.mel_spec
        if (mel.sample_rate, mel.n_mels, mel.spectrogram.n_fft, mel.spectrogram.hop_length) != (24000, 100, 1024, 256):
            raise ValueError("Wrong Vocos checkpoint/frontend")
        if mel.spectrogram.power != 1: raise ValueError("Vocos expects magnitude Mel")

    def train(self, mode=True):
        super().train(False)
        return self

    def features(self, wave):
        return self.model.feature_extractor(wave)

    def forward(self, log_mel):
        # model.decode() is decorated with inference_mode and breaks neural loss.
        return self.model.head(self.model.backbone(log_mel.float()))


class FrozenAudio:
    def __init__(self, cfg, device="cpu", teacher=True):
        if not (cfg.dir("models") / "manifest.json").exists():
            raise FileNotFoundError("Run download to pin and acquire the pretrained models first")
        self.cfg, self.device = cfg, device
        self.vocos = FrozenVocos(cfg.model_dir("vocos")).to(device)
        self.teacher = self.processor = None
        if teacher:
            from transformers import AutoFeatureExtractor, WavLMModel
            self.processor = AutoFeatureExtractor.from_pretrained(cfg.model_dir("wavlm"), local_files_only=True)
            self.teacher = WavLMModel.from_pretrained(cfg.model_dir("wavlm"), local_files_only=True).to(device).eval()
            self.teacher.requires_grad_(False)

    @torch.no_grad()
    def wavlm(self, waves: np.ndarray, crop_start: float = -1.0, core_seconds: float | None = 4.0):
        if self.teacher is None: raise RuntimeError("Teacher was not loaded")
        if waves.ndim == 1: waves = waves[None]
        inputs = self.processor(list(waves), sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.teacher(**inputs, output_hidden_states=True).hidden_states[self.cfg["audio"]["teacher_layer"]]
        receptive, stride = 1, 1
        for kernel, step in zip(self.teacher.config.conv_kernel, self.teacher.config.conv_stride):
            receptive += (kernel - 1) * stride
            stride *= step
        times = crop_start + ((receptive - 1) / 2 + torch.arange(features.shape[1], device=self.device, dtype=torch.float64) * stride) / 16000
        if core_seconds is not None:
            keep = (times >= 0) & (times < core_seconds)
            features, times = features[:, keep], times[keep]
        return features.float(), times.float()


class ASR:
    """Independent, fixed English ASR; no target prompts or candidate decoding."""
    def __init__(self, cfg, device="cpu"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from whisper.normalizers import EnglishTextNormalizer
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(cfg.model_dir("whisper"), local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(cfg.model_dir("whisper"), local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)
        self.model.generation_config.forced_decoder_ids = None
        self.model.config.forced_decoder_ids = None
        self.normalize = EnglishTextNormalizer()

    @torch.no_grad()
    def __call__(self, wave: np.ndarray, rate=24000) -> str:
        if rate != 16000:
            from math import gcd
            g = gcd(rate, 16000)
            wave = resample_poly(wave, 16000 // g, rate // g).astype(np.float32)
        encoded = self.processor(wave, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        features = encoded.input_features.to(self.device)
        tokens = self.model.generate(features, attention_mask=encoded.attention_mask.to(self.device), do_sample=False, num_beams=1, temperature=0.0, condition_on_prev_tokens=False, return_timestamps=False)
        return self.normalize(self.processor.batch_decode(tokens, skip_special_tokens=True)[0]).strip()
