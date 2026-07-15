from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.utils import pad_or_crop_audio, resample_audio


@dataclass(frozen=True)
class DiscreteEncodecConfig:
    model_path: str
    sample_rate: int = 16000
    duration_sec: float = 2.0
    bandwidth: float = 6.0


class DiscreteEncodec:
    """Frozen local EnCodec wrapper that preserves discrete codes and scales."""

    def __init__(self, config: DiscreteEncodecConfig, device: torch.device):
        from transformers import EncodecModel

        self.config = config
        self.device = device
        self.model = EncodecModel.from_pretrained(str(Path(config.model_path)), local_files_only=True).to(device)
        self.model.eval()
        self.codec_sample_rate = int(self.model.config.sampling_rate)
        self.audio_channels = int(self.model.config.audio_channels)
        supported = [float(value) for value in self.model.config.target_bandwidths]
        if float(config.bandwidth) not in supported:
            raise ValueError(f"Unsupported bandwidth {config.bandwidth}; expected {supported}")

    def prepare_batch(self, audio: np.ndarray) -> torch.Tensor:
        values = []
        for row in np.asarray(audio, dtype=np.float32):
            resampled = resample_audio(row, src_sr=int(self.config.sample_rate), dst_sr=self.codec_sample_rate)
            target = int(round(self.codec_sample_rate * float(self.config.duration_sec)))
            values.append(pad_or_crop_audio(resampled, target_len=target))
        array = np.stack(values).astype(np.float32)
        return torch.from_numpy(array).to(self.device).view(len(array), self.audio_channels, -1)

    @torch.no_grad()
    def encode(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        inputs = self.prepare_batch(audio)
        padding_mask = torch.ones_like(inputs, dtype=torch.bool)
        encoded = self.model.encode(
            input_values=inputs,
            padding_mask=padding_mask,
            bandwidth=float(self.config.bandwidth),
            return_dict=True,
        )
        codes = encoded.audio_codes
        if codes is None or codes.ndim != 4 or codes.shape[0] != 1:
            raise ValueError(f"Expected one EnCodec frame [1,B,Q,T], got {None if codes is None else tuple(codes.shape)}")
        frame_codes = codes[0]
        quantized = self.model.quantizer.decode(frame_codes.transpose(0, 1)).transpose(1, 2)
        batch = frame_codes.shape[0]
        scales = np.ones((batch, 1), dtype=np.float32)
        scale_valid = np.zeros(batch, dtype=bool)
        if encoded.audio_scales is not None and len(encoded.audio_scales) == 1 and encoded.audio_scales[0] is not None:
            raw_scale = encoded.audio_scales[0].detach().cpu().numpy().reshape(batch, -1).astype(np.float32)
            scales = raw_scale
            scale_valid[:] = True
        return {
            "codes": frame_codes.detach().cpu().numpy().astype(np.int16),
            "quantized_latent": quantized.detach().cpu().numpy().astype(np.float32),
            "scale": scales,
            "scale_valid": scale_valid,
        }

    @torch.no_grad()
    def decode(self, codes: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
        values = np.asarray(codes, dtype=np.int64)
        single = values.ndim == 2
        if single:
            values = values[None, ...]
        code_tensor = torch.from_numpy(values).long().to(self.device).unsqueeze(0)
        audio_scales = [None]
        if scale is not None:
            scale_array = np.asarray(scale, dtype=np.float32)
            if scale_array.ndim == 1:
                scale_array = scale_array[None, ...]
            audio_scales = [torch.from_numpy(scale_array).to(self.device)]
        output = self.model.decode(code_tensor, audio_scales, return_dict=True).audio_values
        audio = output[:, 0].detach().cpu().numpy().astype(np.float32)
        return audio[0] if single else audio
