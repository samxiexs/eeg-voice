from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.audio_features import AudioFeatureConfig, load_codec_backend  # noqa: E402
from src.utils import pad_or_crop_audio, resample_audio  # noqa: E402


def _tone(freq_hz: float, samples: int = 16000, sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32) / float(sample_rate)
    return np.sin(2.0 * np.pi * float(freq_hz) * t).astype(np.float32)


class AudioFeatureTests(unittest.TestCase):
    @unittest.skipUnless((APP_DIR.parent / "models" / "encodec_24khz").exists(), "Local EnCodec weights not available")
    def test_encodec_roundtrip_smoke(self):
        config = AudioFeatureConfig(
            sample_rate=16000,
            duration_sec=1.0,
            target_kind="encodec_latent",
            backend="encodec_latent",
            codec_model_name_or_path=str(APP_DIR.parent / "models" / "encodec_24khz"),
            local_files_only=True,
            codec_bandwidth=6.0,
        )
        backend = load_codec_backend(config)
        audio = _tone(220.0)
        target = backend.extract(audio, sample_rate=16000)
        decoded = backend.decode(target["target_sequence"], decoder_scales=target["decoder_scales"])
        decoded = pad_or_crop_audio(
            resample_audio(decoded, src_sr=backend.sample_rate, dst_sr=16000),
            target_len=16000,
        )
        self.assertEqual(target["target_sequence"].ndim, 2)
        self.assertEqual(decoded.shape[0], 16000)
        self.assertGreater(float(np.mean(np.abs(decoded))), 0.0)


if __name__ == "__main__":
    unittest.main()
