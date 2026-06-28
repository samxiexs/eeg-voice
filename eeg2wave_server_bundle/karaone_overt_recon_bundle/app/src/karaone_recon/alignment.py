from __future__ import annotations

import csv
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def shift_sequence_np(seq: np.ndarray, shift: int) -> np.ndarray:
    """Shift [T, D] or [T] with zero padding. Positive shift moves content later."""
    arr = np.asarray(seq)
    out = np.zeros_like(arr)
    t = int(arr.shape[0])
    s = int(shift)
    if t <= 0:
        return out
    if s == 0:
        return arr.copy()
    if abs(s) >= t:
        return out
    if s > 0:
        out[s:] = arr[: t - s]
    else:
        out[: t + s] = arr[-s:]
    return out


def shift_sequence_torch(seq: torch.Tensor, shifts: torch.Tensor | int) -> torch.Tensor:
    """Batch shift [B, T, D] with zero padding. Positive shift moves content later."""
    if isinstance(shifts, int):
        shifts_t = torch.full((seq.shape[0],), int(shifts), device=seq.device, dtype=torch.long)
    else:
        shifts_t = shifts.to(device=seq.device, dtype=torch.long).view(-1)
    out = torch.zeros_like(seq)
    b, t = int(seq.shape[0]), int(seq.shape[1])
    for i in range(b):
        s = int(shifts_t[i].item())
        if s == 0:
            out[i] = seq[i]
        elif abs(s) >= t:
            continue
        elif s > 0:
            out[i, s:] = seq[i, : t - s]
        else:
            out[i, : t + s] = seq[i, -s:]
    return out


def _smooth(values: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    if n <= 1 or values.size <= 2:
        return values.astype(np.float64, copy=False)
    kernel = np.ones(n, dtype=np.float64) / float(n)
    return np.convolve(values, kernel, mode="same")


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        sr = int(handle.getframerate())
        n = int(handle.getnframes())
        ch = int(handle.getnchannels())
        sampwidth = int(handle.getsampwidth())
        data = handle.readframes(n)
    if sampwidth == 2:
        audio = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(data, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise RuntimeError(f"unsupported wav sample width: {sampwidth}")
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    return audio.astype(np.float64), sr


def _eeg_envelope(eeg: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(eeg, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    x = x / (x.std(axis=1, keepdims=True) + 1e-6)
    env = np.sqrt(np.mean(x * x, axis=0))
    env = _smooth(env, round(0.100 * float(fs)))
    t = np.arange(env.shape[0], dtype=np.float64) / float(fs)
    return t, env


def _audio_envelope(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    win = max(1, int(round(0.025 * sr)))
    hop = max(1, int(round(0.010 * sr)))
    if audio.shape[0] < win:
        return np.array([audio.shape[0] / max(2.0 * sr, 1.0)]), np.array([float(np.sqrt(np.mean(audio * audio) + 1e-12))])
    starts = np.arange(0, audio.shape[0] - win + 1, hop)
    env = np.empty(starts.shape[0], dtype=np.float64)
    for idx, start in enumerate(starts):
        frame = audio[start : start + win]
        env[idx] = np.sqrt(np.mean(frame * frame) + 1e-12)
    env = _smooth(env, 5)
    t = (starts + win / 2.0) / float(sr)
    return t, env


def _env_stats(t: np.ndarray, env: np.ndarray) -> tuple[float, float, float, float]:
    if env.size == 0 or float(np.max(env)) <= 1e-12:
        return float("nan"), float("nan"), float("nan"), 0.0
    base = np.percentile(env, 10)
    mass = np.maximum(env - base, 0.0)
    peak_t = float(t[int(np.argmax(env))])
    com_t = float(np.sum(t * mass) / (np.sum(mass) + 1e-12))
    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med)))
    threshold = max(med + 2.0 * mad, float(0.20 * np.max(env)))
    active = np.flatnonzero(env >= threshold)
    onset_t = float(t[int(active[0])]) if active.size else peak_t
    return peak_t, com_t, onset_t, float(active.size / max(env.size, 1))


def _active_mask_from_energy(energy: np.ndarray) -> np.ndarray:
    energy = np.asarray(energy, dtype=np.float64)
    mean = energy.mean()
    std = energy.std()
    peak = energy.max(initial=0.0)
    threshold = max(mean + 0.5 * std, 0.1 * peak)
    mask = energy >= threshold
    if not mask.any() and energy.size:
        mask[int(np.argmax(energy))] = True
    return mask.astype(np.float32)


@dataclass(frozen=True)
class AlignmentRecord:
    lag_sec: float
    lag_mel_frames: int
    lag_confidence: float
    eeg_valid_sec: float
    audio_sec: float
    eeg_peak_t: float
    audio_peak_t: float
    eeg_com_t: float
    audio_com_t: float
    eeg_onset_t: float
    audio_onset_t: float


class KaraOneAlignment:
    def __init__(self, path: str | Path):
        payload = np.load(Path(path), allow_pickle=True)
        self.path = Path(path)
        self.keys = payload["keys"].astype(str)
        self.subject_median_lag_sec = {
            str(subject): float(value)
            for subject, value in zip(payload["median_subjects"].astype(str), payload["median_lag_sec"].astype(np.float32))
        }
        self.mel_hop_sec = float(payload["mel_hop_sec"]) if "mel_hop_sec" in payload.files else 0.016
        self._records: dict[tuple[str, str, int], AlignmentRecord] = {}
        stages = payload["stages"].astype(str)
        subjects = payload["subjects"].astype(str)
        trials = payload["trial_indices"].astype(np.int32)
        for i, key in enumerate(self.keys):
            rec = AlignmentRecord(
                lag_sec=float(payload["lag_sec"][i]),
                lag_mel_frames=int(payload["lag_mel_frames"][i]),
                lag_confidence=float(payload["lag_confidence"][i]),
                eeg_valid_sec=float(payload["eeg_valid_sec"][i]),
                audio_sec=float(payload["audio_sec"][i]),
                eeg_peak_t=float(payload["eeg_peak_t"][i]),
                audio_peak_t=float(payload["audio_peak_t"][i]),
                eeg_com_t=float(payload["eeg_com_t"][i]),
                audio_com_t=float(payload["audio_com_t"][i]),
                eeg_onset_t=float(payload["eeg_onset_t"][i]),
                audio_onset_t=float(payload["audio_onset_t"][i]),
            )
            self._records[(str(subjects[i]), str(stages[i]), int(trials[i]))] = rec

    @staticmethod
    def key(subject: str, stage: str, trial_index: int) -> str:
        return f"{subject}:{stage}:{int(trial_index)}"

    def get(self, subject: str, stage: str, trial_index: int) -> AlignmentRecord | None:
        return self._records.get((str(subject), str(stage), int(trial_index)))


def build_alignment_cache(
    data_root: str | Path,
    output_path: str | Path,
    stages: Iterable[str] = ("overt_like",),
    mel_hop_sec: float = 0.016,
    target_steps: int = 126,
    csv_output_path: str | Path | None = None,
) -> Path:
    root = Path(data_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    wanted = set(str(stage) for stage in stages)
    rows: list[dict[str, str]] = []
    with (root / "segments.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row["segment_stage"]) in wanted:
                rows.append(row)

    bundle_cache: dict[str, dict[str, object]] = {}
    records: list[dict[str, object]] = []
    active_masks: list[np.ndarray] = []
    mel_energy: list[np.ndarray] = []
    for row in rows:
        subject = str(row["subject_id"])
        stage = str(row["segment_stage"])
        trial = int(row["trial_index"])
        if subject not in bundle_cache:
            bundle = np.load(root / "subjects" / f"{subject}.npz", allow_pickle=True)
            pos = {int(item): idx for idx, item in enumerate(bundle["trial_indices"].astype(np.int32).tolist())}
            stage_arrays = {}
            valid_arrays = {}
            for stage_name in wanted:
                stage_key = f"stage__{stage_name}"
                if stage_key in bundle.files:
                    stage_arrays[stage_key] = bundle[stage_key]
                    valid_key_name = f"{stage_key}__valid_lengths"
                    valid_arrays[valid_key_name] = (
                        bundle[valid_key_name]
                        if valid_key_name in bundle.files
                        else np.full(stage_arrays[stage_key].shape[0], stage_arrays[stage_key].shape[-1], dtype=np.int32)
                    )
            bundle_cache[subject] = {
                "trial_to_pos": pos,
                "stage_arrays": stage_arrays,
                "valid_arrays": valid_arrays,
                "eeg_sfreq_hz": float(np.asarray(bundle["eeg_sfreq_hz"]).reshape(-1)[0]),
            }
        cached = bundle_cache[subject]
        pos = int(cached["trial_to_pos"][trial])  # type: ignore[index]
        key = str(row["segment_array_key"])
        valid_key = f"{key}__valid_lengths"
        fs = float(cached["eeg_sfreq_hz"])
        stage_arrays = cached["stage_arrays"]  # type: ignore[assignment]
        valid_arrays = cached["valid_arrays"]  # type: ignore[assignment]
        valid = int(valid_arrays[valid_key][pos]) if valid_key in valid_arrays else int(row["eeg_valid_num_samples"])
        eeg_t, eeg_env = _eeg_envelope(stage_arrays[key][pos, :, :valid], fs)
        audio, sr = _read_wav(root / row["audio_path"])
        audio_t, audio_env = _audio_envelope(audio, sr)
        eeg_peak, eeg_com, eeg_onset, eeg_active = _env_stats(eeg_t, eeg_env)
        audio_peak, audio_com, audio_onset, audio_active = _env_stats(audio_t, audio_env)
        lag_sec = float(eeg_com - audio_com)
        lag_frames = int(round(lag_sec / max(float(mel_hop_sec), 1e-6)))
        confidence = float(np.clip(0.5 * (eeg_active + audio_active), 0.05, 1.0))
        audio_energy = np.interp(
            np.linspace(0.0, 1.0, int(target_steps)),
            np.linspace(0.0, 1.0, audio_env.shape[0]),
            audio_env,
        ).astype(np.float32)
        audio_energy = np.maximum(audio_energy, 1e-8)
        active_masks.append(_active_mask_from_energy(audio_energy))
        mel_energy.append(audio_energy)
        records.append(
            {
                "key": KaraOneAlignment.key(subject, stage, trial),
                "subject": subject,
                "label": str(row["label"]),
                "stage": stage,
                "trial_index": trial,
                "eeg_valid_sec": float(valid / fs),
                "audio_sec": float(audio.shape[0] / sr),
                "eeg_peak_t": eeg_peak,
                "audio_peak_t": audio_peak,
                "eeg_com_t": eeg_com,
                "audio_com_t": audio_com,
                "eeg_onset_t": eeg_onset,
                "audio_onset_t": audio_onset,
                "lag_sec": lag_sec,
                "lag_mel_frames": lag_frames,
                "lag_confidence": confidence,
            }
        )

    subjects = sorted({str(item["subject"]) for item in records})
    median_lag = np.asarray(
        [np.median([float(item["lag_sec"]) for item in records if str(item["subject"]) == subject]) for subject in subjects],
        dtype=np.float32,
    )
    payload = {
        "keys": np.asarray([item["key"] for item in records]),
        "subjects": np.asarray([item["subject"] for item in records]),
        "labels": np.asarray([item["label"] for item in records]),
        "stages": np.asarray([item["stage"] for item in records]),
        "trial_indices": np.asarray([item["trial_index"] for item in records], dtype=np.int32),
        "eeg_valid_sec": np.asarray([item["eeg_valid_sec"] for item in records], dtype=np.float32),
        "audio_sec": np.asarray([item["audio_sec"] for item in records], dtype=np.float32),
        "eeg_peak_t": np.asarray([item["eeg_peak_t"] for item in records], dtype=np.float32),
        "audio_peak_t": np.asarray([item["audio_peak_t"] for item in records], dtype=np.float32),
        "eeg_com_t": np.asarray([item["eeg_com_t"] for item in records], dtype=np.float32),
        "audio_com_t": np.asarray([item["audio_com_t"] for item in records], dtype=np.float32),
        "eeg_onset_t": np.asarray([item["eeg_onset_t"] for item in records], dtype=np.float32),
        "audio_onset_t": np.asarray([item["audio_onset_t"] for item in records], dtype=np.float32),
        "lag_sec": np.asarray([item["lag_sec"] for item in records], dtype=np.float32),
        "lag_mel_frames": np.asarray([item["lag_mel_frames"] for item in records], dtype=np.int32),
        "lag_confidence": np.asarray([item["lag_confidence"] for item in records], dtype=np.float32),
        "audio_active_mask": np.stack(active_masks, axis=0).astype(np.float32) if active_masks else np.zeros((0, target_steps), dtype=np.float32),
        "mel_energy": np.stack(mel_energy, axis=0).astype(np.float32) if mel_energy else np.zeros((0, target_steps), dtype=np.float32),
        "median_subjects": np.asarray(subjects),
        "median_lag_sec": median_lag,
        "mel_hop_sec": np.asarray(float(mel_hop_sec), dtype=np.float32),
        "target_steps": np.asarray(int(target_steps), dtype=np.int32),
    }
    np.savez_compressed(output, **payload)
    csv_path = Path(csv_output_path) if csv_output_path is not None else output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "key",
            "subject",
            "label",
            "stage",
            "trial_index",
            "eeg_valid_sec",
            "audio_sec",
            "eeg_peak_t",
            "audio_peak_t",
            "eeg_com_t",
            "audio_com_t",
            "eeg_onset_t",
            "audio_onset_t",
            "lag_sec",
            "lag_mel_frames",
            "lag_confidence",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return output
