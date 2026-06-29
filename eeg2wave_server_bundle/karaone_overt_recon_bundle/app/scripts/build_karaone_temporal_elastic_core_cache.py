from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.karaone_recon.targets import KaraOneTargets
from src.utils import load_simple_yaml, resolve_bundle_path, resolve_target_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build KaraOne v5 temporal-elastic active speech-core Mel cache.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--mel-cache", default=None, help="full 2s KaraOne Mel cache; defaults to target.cache_mel")
    parser.add_argument("--out", default="../artifacts/audio_targets/karaone_temporal_elastic_core_v5.npz")
    parser.add_argument("--ignore-initial-sec", type=float, default=0.10)
    parser.add_argument("--energy-smooth-frames", type=int, default=5)
    parser.add_argument("--threshold-mad", type=float, default=2.0)
    parser.add_argument("--threshold-peak-frac", type=float, default=0.20)
    parser.add_argument("--pre-margin-sec", type=float, default=0.06)
    parser.add_argument("--post-margin-sec", type=float, default=0.08)
    parser.add_argument("--core-len-frames", type=int, default=64)
    return parser.parse_args()


def _smooth(values: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    if n <= 1 or values.size <= 2:
        return values.astype(np.float64, copy=False)
    kernel = np.ones(n, dtype=np.float64) / float(n)
    return np.convolve(values, kernel, mode="same")


def _largest_component(mask: np.ndarray) -> tuple[int, int] | None:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks] + 1, idx[-1] + 1]
    best = int(np.argmax(ends - starts))
    return int(starts[best]), int(ends[best])


def _resample_1d(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    length = int(length)
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    if values.size == 0:
        return np.zeros(length, dtype=np.float32)
    if values.size == length:
        return values.astype(np.float32)
    src = np.linspace(0.0, 1.0, values.size)
    dst = np.linspace(0.0, 1.0, length)
    return np.interp(dst, src, values).astype(np.float32)


def _resample_2d(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    length = int(length)
    if values.shape[0] == length:
        return values.astype(np.float32)
    src = np.linspace(0.0, 1.0, values.shape[0])
    dst = np.linspace(0.0, 1.0, length)
    out = np.empty((length, values.shape[1]), dtype=np.float32)
    for j in range(values.shape[1]):
        out[:, j] = np.interp(dst, src, values[:, j]).astype(np.float32)
    return out


def _active_component(
    raw_mel: np.ndarray,
    hop_sec: float,
    ignore_initial_sec: float,
    smooth_frames: int,
    mad_weight: float,
    peak_frac: float,
) -> tuple[np.ndarray, int, int, int, int, float, float, float]:
    energy = np.exp(np.clip(raw_mel, -12.0, 6.0)).mean(axis=-1).astype(np.float64).clip(min=1e-12)
    smooth = _smooth(energy, smooth_frames)
    ignore_frames = min(int(np.ceil(ignore_initial_sec / max(hop_sec, 1e-8))), max(0, smooth.size - 1))
    search = smooth.copy()
    if ignore_frames > 0:
        search[:ignore_frames] = np.min(search[ignore_frames:]) if ignore_frames < search.size else np.min(search)
    peak_frame = int(np.argmax(search))
    tail = search[ignore_frames:] if ignore_frames < search.size else search
    median = float(np.median(tail))
    mad = float(np.median(np.abs(tail - median))) + 1e-12
    peak = float(np.max(tail)) if tail.size else float(np.max(search))
    threshold = max(median + float(mad_weight) * mad, float(peak_frac) * peak)
    active = (search >= threshold) & (np.arange(search.shape[0]) >= ignore_frames)
    component = _largest_component(active)
    if component is None:
        onset, offset = peak_frame, min(peak_frame + 1, raw_mel.shape[0])
        active = np.zeros(raw_mel.shape[0], dtype=bool)
        active[peak_frame] = True
    else:
        onset, offset = component
        keep = np.zeros_like(active, dtype=bool)
        keep[onset:offset] = True
        active = keep
    local_energy = energy[active]
    if local_energy.size == 0:
        local_energy = np.asarray([energy[peak_frame]], dtype=np.float64)
    rms = float(np.sqrt(np.mean(local_energy) + 1e-12))
    peak_amp = float(np.sqrt(np.max(local_energy) + 1e-12))
    center = int(round((int(onset) + int(offset) - 1) / 2.0))
    return active.astype(bool), int(onset), int(offset), int(center), int(peak_frame), float(peak), rms, peak_amp


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    mel_cache = resolve_bundle_path(args.mel_cache, BUNDLE_DIR) if args.mel_cache else resolve_target_cache(cfg, BUNDLE_DIR, "mel")[1]
    out = resolve_bundle_path(args.out, BUNDLE_DIR)
    out.parent.mkdir(parents=True, exist_ok=True)

    sample_rate = int(cfg.get("audio", {}).get("sample_rate", 16000))
    mel_hop = int(cfg.get("target", {}).get("mel_hop", 256))
    hop_sec = float(mel_hop) / float(sample_rate)
    targets = KaraOneTargets(mel_cache, data_root=root)
    raw = targets.raw_seq.astype(np.float32)
    n, t, d = raw.shape
    core_len = int(args.core_len_frames)
    ignore_frames = min(int(np.ceil(float(args.ignore_initial_sec) / max(hop_sec, 1e-8))), max(0, t - 1))
    pre = int(np.ceil(float(args.pre_margin_sec) / max(hop_sec, 1e-8)))
    post = int(np.ceil(float(args.post_margin_sec) / max(hop_sec, 1e-8)))

    silence_floor_frame = np.percentile(raw, 10, axis=(0, 1)).astype(np.float32)
    silence_floor_raw = np.tile(silence_floor_frame.reshape(1, -1), (t, 1)).astype(np.float32)
    core_raw = np.zeros((n, core_len, d), dtype=np.float32)
    core_mask = np.ones((n, core_len), dtype=np.float32)
    core_active_mask = np.zeros((n, core_len), dtype=np.float32)
    core_pre_noise_mask = np.zeros((n, core_len), dtype=np.float32)
    active_env_raw = np.zeros((n, core_len), dtype=np.float32)
    audio_active_mask = np.zeros((n, t), dtype=np.float32)
    silence_mask = np.ones((n, t), dtype=np.float32)
    pre_noise_mask = np.zeros((n, t), dtype=np.float32)
    ignore_initial_noise_mask = np.zeros((n, t), dtype=np.float32)
    active_duration = np.zeros(n, dtype=np.int32)
    active_start = np.zeros(n, dtype=np.int32)
    active_end = np.zeros(n, dtype=np.int32)
    active_center = np.zeros(n, dtype=np.int32)
    active_rms = np.zeros(n, dtype=np.float32)
    active_peak = np.zeros(n, dtype=np.float32)
    core_start = np.zeros(n, dtype=np.int32)
    core_end = np.zeros(n, dtype=np.int32)
    peak_frames = np.zeros(n, dtype=np.int32)

    for i in range(n):
        active, onset, offset, center, peak_frame, _peak_energy, rms, peak_amp = _active_component(
            raw[i],
            hop_sec=hop_sec,
            ignore_initial_sec=float(args.ignore_initial_sec),
            smooth_frames=int(args.energy_smooth_frames),
            mad_weight=float(args.threshold_mad),
            peak_frac=float(args.threshold_peak_frac),
        )
        start = max(0, onset - pre)
        end = min(t, offset + post)
        if end <= start:
            start = max(0, min(peak_frame, t - 1))
            end = min(t, start + 1)
        segment = raw[i, start:end]
        segment_active = active[start:end].astype(np.float32)
        segment_pre_noise = (np.arange(start, end) < ignore_frames).astype(np.float32)
        segment_energy = np.exp(np.clip(segment, -12.0, 6.0)).mean(axis=-1).clip(min=1e-12)
        core_raw[i] = _resample_2d(segment, core_len)
        core_active_mask[i] = (_resample_1d(segment_active, core_len) >= 0.5).astype(np.float32)
        if not bool(core_active_mask[i].any()):
            core_active_mask[i, int(np.argmax(_resample_1d(segment_energy, core_len)))] = 1.0
        core_pre_noise_mask[i] = (_resample_1d(segment_pre_noise, core_len) >= 0.5).astype(np.float32)
        active_env_raw[i] = _resample_1d(np.sqrt(segment_energy), core_len)
        audio_active_mask[i] = active.astype(np.float32)
        silence_mask[i] = (~active).astype(np.float32)
        pre_noise_mask[i, :ignore_frames] = 1.0
        ignore_initial_noise_mask[i, :ignore_frames] = 1.0
        active_duration[i] = max(1, int(offset - onset))
        active_start[i] = int(onset)
        active_end[i] = int(offset)
        active_center[i] = int(center)
        active_rms[i] = float(rms)
        active_peak[i] = float(peak_amp)
        core_start[i] = int(start)
        core_end[i] = int(end)
        peak_frames[i] = int(peak_frame)

    env_mean = float(active_env_raw.mean())
    env_std = float(max(active_env_raw.std(), 1e-6))
    active_env_norm = ((active_env_raw - env_mean) / env_std).astype(np.float32)
    core_norm = ((core_raw - targets.target_mean.reshape(1, 1, -1)) / targets.target_std.reshape(1, 1, -1)).astype(np.float32)
    core_log_rms = np.log(np.maximum(active_rms, 1e-8)).astype(np.float32)
    payload = {
        "speech_core_kind": np.asarray("temporal_elastic_active_core_v5"),
        "temporal_elastic_core_kind": np.asarray("active_core_resampled"),
        "source_mel_cache": np.asarray(str(mel_cache)),
        "template_ids": targets.template_ids.astype(str),
        "subject_ids": targets.subject_ids.astype(str),
        "labels": targets.labels.astype(str),
        "trial_indices": targets.trial_indices.astype(np.int32),
        "audio_paths": targets.audio_paths.astype(str),
        "target_sequences": core_raw.astype(np.float32),
        "core_mel": core_norm.astype(np.float32),
        "active_core_mel_norm": core_norm.astype(np.float32),
        "active_core_mel_raw": core_raw.astype(np.float32),
        "target_mean": targets.target_mean.astype(np.float32),
        "target_std": targets.target_std.astype(np.float32),
        "target_rms": active_rms.astype(np.float32),
        "target_log_rms": core_log_rms,
        "decoder_scales": targets.decoder_scales.astype(np.float32),
        "default_decoder_scales": targets.default_decoder_scales.astype(np.float32),
        "core_mask": core_mask,
        "core_active_mask": core_active_mask,
        "core_pre_noise_mask": core_pre_noise_mask,
        "audio_active_mask": audio_active_mask,
        "silence_mask": silence_mask,
        "pre_noise_mask": pre_noise_mask,
        "ignore_initial_noise_mask": ignore_initial_noise_mask,
        "active_envelope_raw": active_env_raw.astype(np.float32),
        "active_envelope_norm": active_env_norm.astype(np.float32),
        "active_envelope_mean": np.asarray(env_mean, dtype=np.float32),
        "active_envelope_std": np.asarray(env_std, dtype=np.float32),
        "active_duration_frames": active_duration,
        "active_start_frame": active_start,
        "active_end_frame": active_end,
        "active_center_frame": active_center,
        "active_rms": active_rms.astype(np.float32),
        "active_peak": active_peak.astype(np.float32),
        "core_start_frame": core_start,
        "core_end_frame": core_end,
        "core_insert_frame": active_center,
        "core_log_rms": core_log_rms,
        "core_peak": active_peak.astype(np.float32),
        "core_energy": np.square(active_rms).astype(np.float32),
        "audio_onset_frame": active_start,
        "audio_peak_frame": peak_frames,
        "audio_com_frame": active_center,
        "core_len_frames": np.asarray(core_len, dtype=np.int32),
        "full_target_steps": np.asarray(t, dtype=np.int32),
        "full_target_dim": np.asarray(d, dtype=np.int32),
        "global_core_insert_frame": np.asarray(float(np.median(active_center)), dtype=np.float32),
        "silence_floor_raw": silence_floor_raw.astype(np.float32),
        "mel_hop_sec": np.asarray(hop_sec, dtype=np.float32),
        "ignore_initial_sec": np.asarray(float(args.ignore_initial_sec), dtype=np.float32),
        "pre_margin_sec": np.asarray(float(args.pre_margin_sec), dtype=np.float32),
        "post_margin_sec": np.asarray(float(args.post_margin_sec), dtype=np.float32),
    }
    np.savez_compressed(out, **payload)
    summary = {
        "out": str(out),
        "source_mel_cache": str(mel_cache),
        "trials": int(n),
        "core_shape": [int(core_len), int(d)],
        "full_target_steps": int(t),
        "audio_active_frame_rate_mean": float(audio_active_mask.mean()),
        "core_active_rate_mean": float(core_active_mask.mean()),
        "duration_frames_mean": float(np.mean(active_duration)),
        "duration_frames_median": float(np.median(active_duration)),
        "duration_sec_median": float(np.median(active_duration) * hop_sec),
        "active_center_median": float(np.median(active_center)),
        "active_center_sec_median": float(np.median(active_center) * hop_sec),
        "active_rms_median": float(np.median(active_rms)),
        "active_peak_median": float(np.median(active_peak)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
