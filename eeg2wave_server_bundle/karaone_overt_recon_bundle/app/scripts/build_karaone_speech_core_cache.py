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
    parser = argparse.ArgumentParser(description="Build KaraOne fixed-length speech-core Mel target cache.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--mel-cache", default=None, help="full 2s KaraOne Mel cache; defaults to target.cache_mel")
    parser.add_argument("--out", default="../artifacts/audio_targets/karaone_speech_core_targets.npz")
    parser.add_argument("--ignore-initial-sec", type=float, default=0.10)
    parser.add_argument("--energy-smooth-frames", type=int, default=5)
    parser.add_argument("--threshold-mad", type=float, default=2.0)
    parser.add_argument("--threshold-peak-frac", type=float, default=0.20)
    parser.add_argument("--pre-margin-sec", type=float, default=0.08)
    parser.add_argument("--post-margin-sec", type=float, default=0.12)
    parser.add_argument("--core-len-frames", type=int, default=48)
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
    lengths = ends - starts
    best = int(np.argmax(lengths))
    return int(starts[best]), int(ends[best])


def _active_component(
    raw_mel: np.ndarray,
    hop_sec: float,
    ignore_initial_sec: float,
    smooth_frames: int,
    mad_weight: float,
    peak_frac: float,
) -> tuple[np.ndarray, int, int, int, float, float, float]:
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
    mass = np.maximum(search - np.percentile(search, 10), 0.0)
    frames = np.arange(search.shape[0], dtype=np.float64)
    com_frame = float(np.sum(frames * mass) / (np.sum(mass) + 1e-12)) if mass.size else float(peak_frame)
    return active.astype(bool), int(onset), int(offset), int(peak_frame), com_frame, float(peak), float(energy[active].mean())


def _core_window(
    onset: int,
    offset: int,
    peak_frame: int,
    total_frames: int,
    hop_sec: float,
    pre_margin_sec: float,
    post_margin_sec: float,
    core_len: int,
) -> tuple[int, int, int]:
    pre = int(np.ceil(pre_margin_sec / max(hop_sec, 1e-8)))
    post = int(np.ceil(post_margin_sec / max(hop_sec, 1e-8)))
    start = max(0, int(onset) - pre)
    end = min(int(total_frames), int(offset) + post)
    if end <= start:
        start = max(0, min(int(peak_frame), total_frames - 1))
        end = min(total_frames, start + 1)
    if end - start > core_len:
        crop_start = int(np.clip(int(peak_frame) - core_len // 2, start, end - core_len))
        start, end = crop_start, crop_start + core_len
    return int(start), int(end), int(start)


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    if args.mel_cache:
        mel_cache = resolve_bundle_path(args.mel_cache, BUNDLE_DIR)
    else:
        _, mel_cache = resolve_target_cache(cfg, BUNDLE_DIR, "mel")
    out = resolve_bundle_path(args.out, BUNDLE_DIR)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = int(cfg.get("audio", {}).get("sample_rate", 16000))
    mel_hop = int(cfg.get("target", {}).get("mel_hop", 256))
    hop_sec = float(mel_hop) / float(sample_rate)
    targets = KaraOneTargets(mel_cache, data_root=root)
    raw = targets.raw_seq.astype(np.float32)
    n, t, d = raw.shape
    core_len = int(args.core_len_frames)
    silence_floor_frame = np.percentile(raw, 10, axis=(0, 1)).astype(np.float32)
    silence_floor_raw = np.tile(silence_floor_frame.reshape(1, -1), (t, 1)).astype(np.float32)
    core_raw = np.tile(silence_floor_frame.reshape(1, 1, -1), (n, core_len, 1)).astype(np.float32)
    core_mask = np.zeros((n, core_len), dtype=np.float32)
    core_active_mask = np.zeros((n, core_len), dtype=np.float32)
    core_pre_noise_mask = np.zeros((n, core_len), dtype=np.float32)
    audio_active_mask = np.zeros((n, t), dtype=np.float32)
    silence_mask = np.ones((n, t), dtype=np.float32)
    pre_noise_mask = np.zeros((n, t), dtype=np.float32)
    core_start = np.zeros(n, dtype=np.int32)
    core_end = np.zeros(n, dtype=np.int32)
    core_insert = np.zeros(n, dtype=np.int32)
    core_log_rms = np.zeros(n, dtype=np.float32)
    core_peak = np.zeros(n, dtype=np.float32)
    core_energy = np.zeros(n, dtype=np.float32)
    onset_frames = np.zeros(n, dtype=np.int32)
    peak_frames = np.zeros(n, dtype=np.int32)
    com_frames = np.zeros(n, dtype=np.int32)
    ignore_frames = min(int(np.ceil(float(args.ignore_initial_sec) / max(hop_sec, 1e-8))), max(0, t - 1))
    for i in range(n):
        active, onset, offset, peak_frame, com_frame, peak_energy, active_energy = _active_component(
            raw[i],
            hop_sec=hop_sec,
            ignore_initial_sec=float(args.ignore_initial_sec),
            smooth_frames=int(args.energy_smooth_frames),
            mad_weight=float(args.threshold_mad),
            peak_frac=float(args.threshold_peak_frac),
        )
        start, end, insert = _core_window(
            onset,
            offset,
            peak_frame,
            total_frames=t,
            hop_sec=hop_sec,
            pre_margin_sec=float(args.pre_margin_sec),
            post_margin_sec=float(args.post_margin_sec),
            core_len=core_len,
        )
        valid = max(0, min(core_len, end - start))
        if valid > 0:
            core_raw[i, :valid] = raw[i, start : start + valid]
            core_mask[i, :valid] = 1.0
            local_active = active[start : start + valid]
            core_active_mask[i, :valid] = local_active.astype(np.float32)
            local_frames = np.arange(start, start + valid)
            core_pre_noise_mask[i, :valid] = (local_frames < ignore_frames).astype(np.float32)
        audio_active_mask[i] = active.astype(np.float32)
        silence_mask[i] = (~active).astype(np.float32)
        pre_noise_mask[i, :ignore_frames] = 1.0
        core_start[i] = int(start)
        core_end[i] = int(start + valid)
        core_insert[i] = int(insert)
        onset_frames[i] = int(onset)
        peak_frames[i] = int(peak_frame)
        com_frames[i] = int(round(com_frame))
        core_peak[i] = float(peak_energy)
        core_energy[i] = float(active_energy)
        active_values = np.exp(np.clip(core_raw[i, core_active_mask[i] > 0.0], -12.0, 6.0)).mean(axis=-1)
        if active_values.size == 0:
            active_values = np.exp(np.clip(core_raw[i, core_mask[i] > 0.0], -12.0, 6.0)).mean(axis=-1)
        core_log_rms[i] = float(0.5 * np.log(float(np.mean(active_values)) + 1e-8)) if active_values.size else -4.0
    core_norm = ((core_raw - targets.target_mean.reshape(1, 1, -1)) / targets.target_std.reshape(1, 1, -1)).astype(np.float32)
    payload = {
        "speech_core_kind": np.asarray("speech_core_mel"),
        "source_mel_cache": np.asarray(str(mel_cache)),
        "template_ids": targets.template_ids.astype(str),
        "subject_ids": targets.subject_ids.astype(str),
        "labels": targets.labels.astype(str),
        "trial_indices": targets.trial_indices.astype(np.int32),
        "audio_paths": targets.audio_paths.astype(str),
        "target_sequences": core_raw.astype(np.float32),
        "core_mel": core_norm.astype(np.float32),
        "target_mean": targets.target_mean.astype(np.float32),
        "target_std": targets.target_std.astype(np.float32),
        "target_rms": np.exp(core_log_rms).astype(np.float32),
        "target_log_rms": core_log_rms.astype(np.float32),
        "decoder_scales": targets.decoder_scales.astype(np.float32),
        "default_decoder_scales": targets.default_decoder_scales.astype(np.float32),
        "core_mask": core_mask,
        "core_active_mask": core_active_mask,
        "core_pre_noise_mask": core_pre_noise_mask,
        "audio_active_mask": audio_active_mask,
        "silence_mask": silence_mask,
        "pre_noise_mask": pre_noise_mask,
        "core_start_frame": core_start,
        "core_end_frame": core_end,
        "core_insert_frame": core_insert,
        "core_log_rms": core_log_rms,
        "core_peak": core_peak,
        "core_energy": core_energy,
        "audio_onset_frame": onset_frames,
        "audio_peak_frame": peak_frames,
        "audio_com_frame": com_frames,
        "core_len_frames": np.asarray(core_len, dtype=np.int32),
        "full_target_steps": np.asarray(t, dtype=np.int32),
        "full_target_dim": np.asarray(d, dtype=np.int32),
        "global_core_insert_frame": np.asarray(float(np.median(core_insert)), dtype=np.float32),
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
        "active_frame_rate_mean": float(audio_active_mask.mean()),
        "core_mask_rate_mean": float(core_mask.mean()),
        "core_active_rate_mean": float(core_active_mask.mean()),
        "core_insert_median": float(np.median(core_insert)),
        "core_insert_p05": float(np.percentile(core_insert, 5)),
        "core_insert_p95": float(np.percentile(core_insert, 95)),
        "audio_onset_sec_median": float(np.median(onset_frames) * hop_sec),
        "audio_peak_sec_median": float(np.median(peak_frames) * hop_sec),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
