from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.feis_v3.data import FEISV3AudioTokenBank


def test_token_bank_label_prior_uses_train_variants_only(tmp_path):
    path = tmp_path / "tokens.npz"
    np.savez_compressed(
        path,
        audio_keys=np.asarray(["01:a", "20:a", "21:a"]),
        subject_ids=np.asarray(["01", "20", "21"]),
        labels=np.asarray(["a", "a", "a"]),
        audio_paths=np.asarray(["audio/01/a.wav", "audio/20/a.wav", "audio/21/a.wav"]),
        audio_sha1=np.asarray(["x", "y", "z"]),
        fit_split=np.asarray(["train", "subject_val", "subject_test"]),
        label_vocab=np.asarray(["a"]),
        subject_vocab=np.asarray(["01", "20", "21"]),
        semantic_hidden=np.zeros((3, 2, 2), dtype=np.float32),
        semantic_token_ids=np.asarray([[1, 1], [2, 2], [3, 3]], dtype=np.int64),
        semantic_token_mask=np.ones((3, 2), dtype=np.float32),
        prosody_active=np.ones((3, 2), dtype=np.float32),
        prosody_duration=np.ones((3, 2), dtype=np.float32),
        prosody_energy=np.zeros((3, 2), dtype=np.float32),
        prosody_onset=np.zeros((3, 2), dtype=np.float32),
        codec_latent=np.zeros((3, 2, 2), dtype=np.float32),
        codec_token_ids=np.asarray([[4, 4], [5, 5], [6, 6]], dtype=np.int64),
        codec_token_mask=np.ones((3, 2), dtype=np.float32),
        audio_variant_cluster_id=np.zeros(3, dtype=np.int64),
        semantic_codebook=np.zeros((8, 2), dtype=np.float32),
        codec_feature_codebook=np.zeros((8, 2), dtype=np.float32),
        codec_codebook_waveform=np.zeros((8, 4), dtype=np.float32),
        audio_variant_cluster_centers=np.zeros((1, 5), dtype=np.float32),
        sample_rate=np.asarray(16000, dtype=np.int32),
        codec_chunk_samples=np.asarray(4, dtype=np.int32),
    )
    bank = FEISV3AudioTokenBank(path)
    assert bank.label_prior_codec_tokens("a").tolist() == [4, 4]
    assert {split for split in bank.fit_split if split != "train"} == {"subject_val", "subject_test"}


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        class Tmp:
            def __truediv__(self, name):
                return Path(tmp) / name

        test_token_bank_label_prior_uses_train_variants_only(Tmp())
    print("FEIS v3 no-leakage smoke passed")
