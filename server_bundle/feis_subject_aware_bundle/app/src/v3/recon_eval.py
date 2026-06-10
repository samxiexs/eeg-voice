"""Reconstruction-fidelity evaluation (not retrieval).

Decodes the predicted EnCodec latent to a waveform and measures how close it is
to *this subject's own* target recording — log-mel L1 + multi-resolution STFT.

Crucially it also reports a subject-specificity control: the same distance
against a DIFFERENT subject's recording of the SAME prompt. If the model truly
reconstructs the user's voice, `dist_to_own` should be clearly smaller than
`dist_to_other_subject_same_label`. The gap (`subject_specificity_gap`) is the
direct evidence that we are reconstructing subject 01's "f", not a generic "f".
"""

from __future__ import annotations

from collections import defaultdict

import librosa
import numpy as np
import torch

from .synth import build_codec_backend, latent_to_wav


def _to_16k(wav: np.ndarray, sr: int, n_samples: int) -> np.ndarray:
    if sr != 16000:
        wav = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=16000)
    wav = np.asarray(wav, dtype=np.float32)
    if len(wav) >= n_samples:
        return wav[:n_samples]
    return np.pad(wav, (0, n_samples - len(wav)))


def _log_mel(wav: np.ndarray, sr: int = 16000, n_mels: int = 64) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=1024, hop_length=256, n_mels=n_mels)
    return np.log(mel + 1e-6).astype(np.float32)


def mel_l1(a: np.ndarray, b: np.ndarray, sr: int = 16000) -> float:
    ma, mb = _log_mel(a, sr), _log_mel(b, sr)
    t = min(ma.shape[1], mb.shape[1])
    return float(np.abs(ma[:, :t] - mb[:, :t]).mean())


def multires_stft(a: np.ndarray, b: np.ndarray) -> float:
    dist = 0.0
    for n_fft in (512, 1024, 2048):
        hop = n_fft // 4
        sa = np.abs(librosa.stft(a, n_fft=n_fft, hop_length=hop))
        sb = np.abs(librosa.stft(b, n_fft=n_fft, hop_length=hop))
        t = min(sa.shape[1], sb.shape[1])
        sa, sb = sa[:, :t], sb[:, :t]
        mag = np.linalg.norm(sa - sb) / (np.linalg.norm(sa) + 1e-8)
        log = np.abs(np.log(sa + 1e-6) - np.log(sb + 1e-6)).mean()
        dist += float(mag + log)
    return dist / 3.0


@torch.no_grad()
def evaluate_reconstruction(
    model,
    dataset,
    codec_model_path: str,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: str = "cpu",
    duration_sec: float = 1.0,
    bandwidth: float = 6.0,
    limit: int | None = None,
    forward_kwargs: dict | None = None,
) -> dict:
    model.eval()
    fkw = forward_kwargs or {}
    backend = build_codec_backend(codec_model_path, duration_sec=duration_sec, bandwidth=bandwidth)
    sr = backend.sample_rate
    n_samples = int(round(16000 * duration_sec))

    # Index target wavs by (label -> {subject: wav16k}) for the cross-subject control.
    by_label_subject: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for tid in dataset.unique_template_ids(split="train") + dataset.unique_template_ids(split="test"):
        meta = dataset.template_metadata(tid)
        raw = np.asarray(dataset.get_template_target(tid)["raw_target_sequence"], dtype=np.float32)
        wav = _to_16k(backend.decode(raw), sr, n_samples)
        by_label_subject[meta["label"]][meta["subject_id"]] = wav

    n = 0
    sum_own_mel = sum_own_stft = 0.0
    sum_other_mel = 0.0
    n_other = 0
    rng = np.random.RandomState(0)
    count = len(dataset) if limit is None else min(limit, len(dataset))
    for i in range(count):
        item = dataset[i]
        eeg = item["eeg"].unsqueeze(0).to(device)
        subj = item["subject_index"].view(1).to(device)
        pred_norm = model(eeg, subj, **fkw)["speech_sequence"].squeeze(0).cpu().numpy()
        pred_wav = _to_16k(latent_to_wav(backend, pred_norm, target_mean, target_std), sr, n_samples)

        label, subject_id = item["label"], item["subject_id"]
        own = by_label_subject[label][subject_id]
        sum_own_mel += mel_l1(pred_wav, own)
        sum_own_stft += multires_stft(pred_wav, own)

        others = [s for s in by_label_subject[label] if s != subject_id]
        if others:
            other = by_label_subject[label][others[rng.randint(len(others))]]
            sum_other_mel += mel_l1(pred_wav, other)
            n_other += 1
        n += 1

    own_mel = sum_own_mel / max(n, 1)
    other_mel = sum_other_mel / max(n_other, 1)
    return {
        "num_trials": n,
        "mel_l1_to_own": own_mel,
        "stft_to_own": sum_own_stft / max(n, 1),
        "mel_l1_to_other_subject_same_label": other_mel,
        # > 0 means the reconstruction is closer to the user's OWN recording.
        "subject_specificity_gap": other_mel - own_mel,
    }
