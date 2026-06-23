from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def denormalize_latent(pred_norm: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray) -> np.ndarray:
    return (pred_norm * target_std.reshape(1, -1) + target_mean.reshape(1, -1)).astype(np.float32)


class EncodecDecoder:
    def __init__(self, model_path: str | Path, local_files_only: bool = True):
        from transformers import EncodecModel

        self.model = EncodecModel.from_pretrained(str(model_path), local_files_only=local_files_only)
        self.model.eval()
        self.sample_rate = int(self.model.config.sampling_rate)

    def decode(self, target_sequence: np.ndarray, decoder_scales: np.ndarray | None = None) -> np.ndarray:
        latents = torch.from_numpy(np.asarray(target_sequence, dtype=np.float32)).transpose(0, 1).unsqueeze(0)
        with torch.no_grad():
            audio = self.model.decoder(latents)
            if decoder_scales is not None:
                scale = torch.from_numpy(np.asarray(decoder_scales, dtype=np.float32)).view(-1, 1, 1)
                audio = audio * scale
        return audio.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def build_codec_backend(codec_model_path: str | Path, local_files_only: bool = True) -> EncodecDecoder:
    return EncodecDecoder(codec_model_path, local_files_only=local_files_only)
