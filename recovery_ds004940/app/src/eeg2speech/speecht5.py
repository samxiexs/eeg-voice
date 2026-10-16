"""Pinned native SpeechT5 mel and HiFi-GAN helpers.

The project used to invert a Slaney log-power mel through Griffin--Lim.  That
representation is useful for diagnostics but is not a waveform contract.  This
module keeps the exact SpeechT5 mel frontend and the vocoder backend together
so audio-oracle, renderer, and qualitative export cannot silently disagree.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch


CONTRACT = "speecht5_native_log_mel_v1"
SAMPLE_RATE = 16000
HOP_SAMPLES = 256
_FEATURE_EXTRACTOR = None


def native_speecht5_mel(waveform: torch.Tensor) -> torch.Tensor:
    """Return the time-major SpeechT5 target as ``[B, 80, T]``.

    ``SpeechT5FeatureExtractor`` is also the frontend used by the pinned
    HiFi-GAN checkpoint.  Keeping it here avoids an undocumented dB/log or hop
    conversion between target caching and synthesis.
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError("waveform must be [B,S] or [S]")
    global _FEATURE_EXTRACTOR
    if _FEATURE_EXTRACTOR is None:
        from transformers import SpeechT5FeatureExtractor
        _FEATURE_EXTRACTOR = SpeechT5FeatureExtractor(
            sampling_rate=SAMPLE_RATE, num_mel_bins=80, hop_length=16,
            win_length=64, win_function="hann_window", fmin=80, fmax=7600,
            mel_floor=1e-10, do_normalize=False,
        )
    values = []
    for row in waveform.detach().float().cpu().numpy():
        value = _FEATURE_EXTRACTOR._extract_mel_features(np.asarray(row, dtype=np.float32))
        values.append(torch.from_numpy(value).float().T)
    maximum = max(value.shape[-1] for value in values)
    padded = [torch.nn.functional.pad(value, (0, maximum - value.shape[-1]), value=float(value.min()))
              for value in values]
    return torch.stack(padded).to(waveform.device)


class SpeechT5HiFiGan:
    """Local-only SpeechT5 HiFi-GAN; never downloads or changes a base model."""

    def __init__(self, root: Path, *, device: torch.device):
        if not root.is_dir():
            raise FileNotFoundError(f"SpeechT5 HiFi-GAN cache is missing: {root}")
        from transformers import SpeechT5HifiGan
        self.root = root
        self.device = device
        self.model = SpeechT5HifiGan.from_pretrained(str(root), local_files_only=True).to(device).eval()
        hop = int(np.prod(self.model.config.upsample_rates))
        if hop != HOP_SAMPLES:
            raise RuntimeError(f"unexpected SpeechT5 hop {hop}; expected {HOP_SAMPLES}")

    @torch.no_grad()
    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.ndim != 3 or mel.shape[1] != 80:
            raise ValueError("SpeechT5 mel must be [B,80,T]")
        return self.model(mel.transpose(1, 2).to(self.device))


def model_manifest(root: Path) -> dict[str, object]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"relative_path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest})
    return {"backend": "microsoft/speecht5_hifigan", "contract": CONTRACT, "local_path": str(root), "files": files}
