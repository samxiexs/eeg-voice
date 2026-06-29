from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.audio_features import AudioFeatureConfig, load_mel_vocoder
from src.karaone_recon.alignment import shift_sequence_np
from src.karaone_recon.data import KaraOneTrialDataset
from src.karaone_recon.model import KaraOneConfig, KaraOneEEG2Codec
from src.karaone_recon.prototypes import KaraOneSemanticMelPrototypes
from src.karaone_recon.rendered_metrics import load_whisper_asr, transcribe_label_metrics
from src.karaone_recon.semantic_tokens import KaraOneSemanticTokenTargets
from src.karaone_recon.synth import build_codec_backend, denormalize_latent
from src.karaone_recon.targets import KaraOneTargets
from src.utils import ensure_dir, load_simple_yaml, load_wav_fixed, resolve_bundle_path, resolve_target_cache, save_wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize KaraOne wavs from a reconstruction checkpoint.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "subject_test"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--limit", type=int, default=24, help="number of trials; <=0 means ALL trials in the split")
    parser.add_argument("--device", default=None)
    parser.add_argument("--asr-model", default=None, help="optional local/cached Whisper model for rendered-audio ASR metrics")
    parser.add_argument("--asr-allow-download", action="store_true", help="allow Whisper to download --asr-model if not cached")
    parser.add_argument("--asr-download-root", default=None, help="optional Whisper model cache/download directory")
    return parser.parse_args()


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)) + 1e-12)


def _scale_to_rms(audio: np.ndarray, target_rms: float, max_gain: float = 20.0) -> np.ndarray:
    gain = min(float(target_rms) / _rms(audio), max_gain)
    return (audio * gain).astype(np.float32)


def _resample_1d(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if values.size == 0:
        return np.ones(n, dtype=np.float64)
    if values.size == n:
        return values
    src = np.linspace(0.0, 1.0, num=values.size)
    dst = np.linspace(0.0, 1.0, num=n)
    return np.interp(dst, src, values)


def _resample_2d(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = int(n)
    if n <= 0:
        return np.zeros((0, values.shape[1] if values.ndim == 2 else 1), dtype=np.float32)
    if values.ndim != 2:
        values = values.reshape(-1, 1)
    if values.shape[0] == n:
        return values.astype(np.float32)
    src = np.linspace(0.0, 1.0, num=values.shape[0])
    dst = np.linspace(0.0, 1.0, num=n)
    out = np.empty((n, values.shape[1]), dtype=np.float32)
    for j in range(values.shape[1]):
        out[:, j] = np.interp(dst, src, values[:, j]).astype(np.float32)
    return out


def _calibrate_frame_envelope(
    audio: np.ndarray,
    frame_log_energy: np.ndarray | None,
    hop: int = 256,
    max_gain: float = 12.0,
) -> np.ndarray:
    if frame_log_energy is None:
        return audio.astype(np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    n_frames = max(1, int(np.ceil(len(audio) / float(hop))))
    current = _envelope(audio, hop=hop)
    if current.size < n_frames:
        current = np.pad(current, (0, n_frames - current.size), mode="edge")
    current = current[:n_frames].clip(min=1e-5)
    target = np.sqrt(np.exp(np.asarray(frame_log_energy, dtype=np.float64).clip(min=-20.0, max=8.0)))
    target = _resample_1d(target, n_frames).clip(min=1e-5)
    # Use the predicted frame energy as a contour, not an absolute loudness source.
    target = target / (np.median(target) + 1e-8) * (np.median(current) + 1e-8)
    gain = np.clip(target / current, 1.0 / max_gain, max_gain)
    sample_gain = np.repeat(gain, hop)
    if sample_gain.size < len(audio):
        sample_gain = np.pad(sample_gain, (0, len(audio) - sample_gain.size), mode="edge")
    return (audio * sample_gain[: len(audio)]).astype(np.float32)


def _insert_core_mel(
    core_raw: np.ndarray,
    silence_floor_raw: np.ndarray,
    insert_frame: int,
    full_steps: int,
) -> np.ndarray:
    core = np.asarray(core_raw, dtype=np.float32)
    floor = np.asarray(silence_floor_raw, dtype=np.float32)
    if floor.ndim == 1:
        floor = np.tile(floor.reshape(1, -1), (int(full_steps), 1))
    if floor.shape[0] != int(full_steps):
        if floor.shape[0] > int(full_steps):
            floor = floor[: int(full_steps)]
        else:
            pad = np.repeat(floor[-1:, :], int(full_steps) - floor.shape[0], axis=0)
            floor = np.concatenate([floor, pad], axis=0)
    canvas = floor.copy()
    start = int(np.clip(int(insert_frame), 0, max(0, int(full_steps) - 1)))
    valid = min(core.shape[0], int(full_steps) - start)
    if valid > 0:
        canvas[start : start + valid] = core[:valid]
    return canvas.astype(np.float32)


def _insert_core_mel_centered(
    core_raw: np.ndarray,
    silence_floor_raw: np.ndarray,
    center_frame: int,
    full_steps: int,
) -> tuple[np.ndarray, int]:
    start = int(np.rint(float(center_frame) - 0.5 * float(core_raw.shape[0])))
    start = int(np.clip(start, 0, max(0, int(full_steps) - int(core_raw.shape[0]))))
    return _insert_core_mel(core_raw, silence_floor_raw, start, full_steps), start


def _predicted_core_insert_frame(
    out: dict[str, torch.Tensor],
    ckpt: dict,
    full_steps: int,
    core_steps: int,
) -> int:
    base = int(ckpt.get("speech_core_default_insert_frame", 0))
    shift = 0
    if "pred_shift_logits" in out:
        logits = out["pred_shift_logits"].view(-1)
        idx = int(torch.argmax(logits).detach().cpu())
        lo = int(ckpt.get("soft_shift_min_frames", -12))
        hi = int(ckpt.get("soft_shift_max_frames", 62))
        bins = max(1, int(logits.numel()))
        if bins > 1:
            shift = int(round(lo + (hi - lo) * idx / float(bins - 1)))
        else:
            shift = 0
    max_start = max(0, int(full_steps) - int(core_steps))
    return int(np.clip(base + shift, 0, max_start))


def _predicted_core_center_frame(out: dict[str, torch.Tensor], ckpt: dict, full_steps: int) -> int:
    base = int(ckpt.get("speech_core_default_insert_frame", 0))
    shift = 0
    if "pred_shift_logits" in out:
        logits = out["pred_shift_logits"].view(-1)
        idx = int(torch.argmax(logits).detach().cpu())
        lo = int(ckpt.get("soft_shift_min_frames", -12))
        hi = int(ckpt.get("soft_shift_max_frames", 62))
        bins = max(1, int(logits.numel()))
        shift = int(round(lo + (hi - lo) * idx / float(max(bins - 1, 1)))) if bins > 1 else 0
    return int(np.clip(base + shift, 0, max(0, int(full_steps) - 1)))


def _predicted_duration_frames(out: dict[str, torch.Tensor], full_steps: int, fallback: int) -> int:
    if "pred_duration_mu" in out:
        value = float(out["pred_duration_mu"].view(-1)[0].detach().cpu())
    elif "pred_duration_logits" in out:
        idx = int(torch.argmax(out["pred_duration_logits"].view(-1)).detach().cpu())
        value = float(idx + 1)
    else:
        value = float(fallback)
    return int(np.clip(int(round(value)), 2, max(2, int(full_steps))))


def _core_energy_to_full(frame_log_energy: np.ndarray | None, insert_frame: int, full_steps: int, floor: float = -16.0) -> np.ndarray | None:
    if frame_log_energy is None:
        return None
    core = np.asarray(frame_log_energy, dtype=np.float64).reshape(-1)
    full = np.full(int(full_steps), float(floor), dtype=np.float64)
    start = int(np.clip(insert_frame, 0, max(0, int(full_steps) - 1)))
    valid = min(core.size, int(full_steps) - start)
    if valid > 0:
        full[start : start + valid] = core[:valid]
    return full


def _envelope(audio: np.ndarray, hop: int = 256) -> np.ndarray:
    n = len(audio) // hop
    if n < 2:
        return np.zeros(2, dtype=np.float64)
    frames = np.asarray(audio[: n * hop], dtype=np.float64).reshape(n, hop)
    return np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)


def _env_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of frame-energy envelopes — a phase-robust waveform-level
    similarity (raw waveform PCC is meaningless under Griffin-Lim's lost phase)."""
    ea, eb = _envelope(a), _envelope(b)
    m = min(len(ea), len(eb))
    if m < 2:
        return 0.0
    ea = ea[:m] - ea[:m].mean()
    eb = eb[:m] - eb[:m].mean()
    den = float(np.linalg.norm(ea) * np.linalg.norm(eb)) + 1e-8
    return float((ea * eb).sum() / den)


def _best_shift_env_corr(a: np.ndarray, b: np.ndarray, hop: int = 256, max_shift_sec: float = 0.75, sr: int = 16000) -> tuple[float, float]:
    ea, eb = _envelope(a, hop=hop), _envelope(b, hop=hop)
    max_shift = int(round(float(max_shift_sec) * float(sr) / float(max(hop, 1))))
    best_corr = -1.0
    best_shift = 0
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            aa, bb = ea[-shift:], eb[: ea.size + shift]
        elif shift > 0:
            aa, bb = ea[: ea.size - shift], eb[shift:]
        else:
            aa, bb = ea, eb
        m = min(aa.size, bb.size)
        if m < 2:
            continue
        aa = aa[:m] - aa[:m].mean()
        bb = bb[:m] - bb[:m].mean()
        corr = float((aa * bb).sum() / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-8))
        if corr > best_corr:
            best_corr = corr
            best_shift = int(shift)
    return float(best_corr), float(best_shift * hop / max(sr, 1))


def _active_mask(env: np.ndarray) -> np.ndarray:
    env = np.asarray(env, dtype=np.float64)
    if env.size == 0 or float(env.max(initial=0.0)) <= 1e-10:
        return np.ones(max(env.size, 1), dtype=bool)
    median = float(np.median(env))
    mad = float(np.median(np.abs(env - median))) + 1e-12
    threshold = max(0.2 * float(env.max()), median + 2.0 * mad)
    mask = env >= threshold
    if not mask.any():
        mask[int(np.argmax(env))] = True
    return mask


def _samples_from_frame_mask(mask: np.ndarray, n: int, hop: int = 256) -> np.ndarray:
    sample_mask = np.repeat(mask.astype(bool), hop)
    if sample_mask.size < n:
        sample_mask = np.pad(sample_mask, (0, n - sample_mask.size), constant_values=False)
    return sample_mask[:n]


def _samples_from_frame_span(start_frame: int, n_frames: int, n: int, hop: int = 256) -> np.ndarray:
    mask = np.zeros(int(n), dtype=bool)
    start = int(max(0, int(start_frame) * int(hop)))
    end = int(min(int(n), (int(start_frame) + max(1, int(n_frames))) * int(hop)))
    if end > start:
        mask[start:end] = True
    return mask


def _local_loudness_mask(
    frame_log_energy: np.ndarray | None,
    insert_frame: int,
    core_frames: int,
    n_samples: int,
    hop: int = 256,
) -> np.ndarray:
    fallback = _samples_from_frame_span(insert_frame, core_frames, n_samples, hop=hop)
    if frame_log_energy is None:
        return fallback
    energy = np.exp(np.asarray(frame_log_energy, dtype=np.float64).reshape(-1).clip(min=-20.0, max=8.0))
    if energy.size == 0 or float(energy.max(initial=0.0)) <= 1e-10:
        return fallback
    median = float(np.median(energy))
    mad = float(np.median(np.abs(energy - median))) + 1e-12
    threshold = max(median + 2.0 * mad, 0.2 * float(energy.max()))
    frame_mask = energy >= threshold
    if not bool(frame_mask.any()):
        frame_mask[int(np.argmax(energy))] = True
    sample_mask = _samples_from_frame_mask(frame_mask, n_samples, hop=hop)
    # Keep the active mask inside the predicted core placement. If the energy head is
    # too flat, fall back to the whole core span rather than amplifying silence.
    core_sample_mask = fallback
    sample_mask = sample_mask & core_sample_mask
    return sample_mask if bool(sample_mask.any()) else fallback


def _limit_audio(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    out = np.asarray(audio, dtype=np.float32).copy()
    max_abs = float(np.max(np.abs(out))) if out.size else 0.0
    if max_abs > float(peak):
        out *= float(peak) / max_abs
    return np.clip(out, -float(peak), float(peak)).astype(np.float32)


def _scale_region_to_rms(
    audio: np.ndarray,
    target_rms: float,
    region_mask: np.ndarray | None,
    max_gain: float = 12.0,
    outside_attenuation: float = 0.12,
    max_full_rms: float = 0.16,
) -> np.ndarray:
    out = np.asarray(audio, dtype=np.float32).copy()
    if out.size == 0:
        return out
    if region_mask is None or region_mask.size != out.size or not bool(region_mask.any()):
        region_mask = np.ones(out.size, dtype=bool)
    else:
        region_mask = region_mask.astype(bool)
    gain = float(target_rms) / max(_rms(out[region_mask]), 1e-8)
    gain = float(np.clip(gain, 1.0 / max(float(max_gain), 1.0), float(max_gain)))
    out[region_mask] *= gain
    if bool((~region_mask).any()):
        out[~region_mask] *= float(np.clip(outside_attenuation, 0.0, 1.0))
    full_rms = _rms(out)
    if full_rms > float(max_full_rms):
        out *= float(max_full_rms) / max(full_rms, 1e-8)
    return _limit_audio(out)


def _silence_leakage_wav(audio: np.ndarray, active_sample_mask: np.ndarray | None) -> float:
    if active_sample_mask is None or active_sample_mask.size != len(audio) or not bool(active_sample_mask.any()):
        return 0.0
    active_sample_mask = active_sample_mask.astype(bool)
    if not bool((~active_sample_mask).any()):
        return 0.0
    inside = _rms(np.asarray(audio)[active_sample_mask])
    outside = _rms(np.asarray(audio)[~active_sample_mask])
    return float(outside / max(inside, 1e-8))


def _active_metrics(candidate: np.ndarray, original: np.ndarray, hop: int = 256) -> dict[str, float]:
    cand_env = _envelope(candidate, hop=hop)
    orig_env = _envelope(original, hop=hop)
    m = min(len(cand_env), len(orig_env))
    cand_env, orig_env = cand_env[:m], orig_env[:m]
    orig_active = _active_mask(orig_env)
    cand_active = _active_mask(cand_env)
    orig_sample_mask = _samples_from_frame_mask(orig_active, min(len(candidate), len(original)), hop=hop)
    cand = np.asarray(candidate[: orig_sample_mask.size], dtype=np.float64)
    orig = np.asarray(original[: orig_sample_mask.size], dtype=np.float64)
    if not orig_sample_mask.any():
        orig_sample_mask = np.ones_like(orig_sample_mask, dtype=bool)
    cand_voiced = cand[orig_sample_mask]
    orig_voiced = orig[orig_sample_mask]
    voiced_rms_ratio = _rms(cand_voiced) / max(_rms(orig_voiced), 1e-8)
    peak_ratio = float(np.max(np.abs(cand_voiced)) / max(float(np.max(np.abs(orig_voiced))), 1e-8))
    duration_ratio = float(cand_active.mean() / max(float(orig_active.mean()), 1e-8))
    active_corr = 0.0
    if orig_active.sum() >= 2:
        ca = cand_env[orig_active] - cand_env[orig_active].mean()
        oa = orig_env[orig_active] - orig_env[orig_active].mean()
        active_corr = float((ca * oa).sum() / (np.linalg.norm(ca) * np.linalg.norm(oa) + 1e-8))
    return {
        "active_env_corr": active_corr,
        "voiced_rms_over_orig": float(voiced_rms_ratio),
        "peak_over_orig": peak_ratio,
        "active_duration_ratio": duration_ratio,
    }


def _active_segment_shape_corr(candidate: np.ndarray, original: np.ndarray, hop: int = 256, core_len: int = 64) -> float:
    cand_env = _envelope(candidate, hop=hop)
    orig_env = _envelope(original, hop=hop)
    cand_active = _active_mask(cand_env)
    orig_active = _active_mask(orig_env)
    if not bool(cand_active.any()) or not bool(orig_active.any()):
        return 0.0
    cand_seg = cand_env[np.flatnonzero(cand_active)[0] : np.flatnonzero(cand_active)[-1] + 1]
    orig_seg = orig_env[np.flatnonzero(orig_active)[0] : np.flatnonzero(orig_active)[-1] + 1]
    cand_rs = _resample_1d(cand_seg, core_len)
    orig_rs = _resample_1d(orig_seg, core_len)
    cand_rs = cand_rs - cand_rs.mean()
    orig_rs = orig_rs - orig_rs.mean()
    return float((cand_rs * orig_rs).sum() / (np.linalg.norm(cand_rs) * np.linalg.norm(orig_rs) + 1e-8))


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    asr_model, asr_status = load_whisper_asr(args.asr_model, device, args.asr_allow_download, args.asr_download_root)
    print(f"[asr] {asr_status}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = KaraOneEEG2Codec(KaraOneConfig(**ckpt["model_config"])).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"[synth] checkpoint missing {len(missing)} new keys; using initialized defaults for: {missing[:4]}")
    if unexpected:
        print(f"[synth] checkpoint has {len(unexpected)} unexpected keys; ignored: {unexpected[:4]}")
    model.eval()
    has_trained_scale_head = any(str(key).startswith("decoder_scale_head.") for key in ckpt.get("model_state", {}))
    residual_mean = bool(ckpt.get("residual_mean", False)) or str(ckpt.get("prediction_mode", "")) == "residual_global_mean"
    semantic_proto_residual = bool(ckpt.get("semantic_prototype_residual", False)) or str(ckpt.get("prediction_mode", "")) == "semantic_prototype_residual"
    eeg_audio_direct = bool(ckpt.get("eeg_audio_direct", False))
    speech_core_objective = bool(ckpt.get("speech_core_objective", False)) or str(ckpt.get("prediction_mode", "")) == "speech_core_residual_mean"
    local_loudness_synthesis = bool(ckpt.get("local_loudness_synthesis", False))
    temporal_elastic_objective = bool(ckpt.get("temporal_elastic_objective", False))
    use_lag_correction = bool(ckpt.get("alignment_objective", False))
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    target_kind = str(ckpt.get("target_kind", cfg.get("target", {}).get("kind", "encodec_latent")))
    if speech_core_objective:
        target_kind = "mel"
        core_path_raw = ckpt.get("speech_core_cache")
        if not core_path_raw:
            raise ValueError("speech-core checkpoint is missing speech_core_cache")
        core_path = Path(str(core_path_raw))
        if not core_path.is_absolute():
            core_path = resolve_bundle_path(str(core_path_raw), BUNDLE_DIR)
        targets = KaraOneTargets(core_path, data_root=root)
        full_path_raw = ckpt.get("full_mel_cache")
        if full_path_raw:
            full_path = Path(str(full_path_raw))
            if not full_path.is_absolute():
                full_path = resolve_bundle_path(str(full_path_raw), BUNDLE_DIR)
        else:
            _, full_path = resolve_target_cache(cfg, BUNDLE_DIR, "mel")
        render_targets = KaraOneTargets(full_path, data_root=root)
    else:
        _, cache = resolve_target_cache(cfg, BUNDLE_DIR, target_kind)
        targets = KaraOneTargets(cache, data_root=root)
        render_targets = targets
    prototype_cache = None
    prototype_tensors = None
    token_targets = None
    if semantic_proto_residual or eeg_audio_direct:
        proto_path_raw = ckpt.get("semantic_prototype_cache")
        token_path_raw = ckpt.get("semantic_token_cache")
        if semantic_proto_residual and not proto_path_raw:
            raise ValueError("v4 checkpoint is missing semantic_prototype_cache")
        if proto_path_raw:
            proto_path = Path(str(proto_path_raw))
            if not proto_path.is_absolute():
                proto_path = resolve_bundle_path(str(proto_path_raw), BUNDLE_DIR)
            if proto_path.exists():
                prototype_cache = KaraOneSemanticMelPrototypes(proto_path)
                prototype_tensors = prototype_cache.to_tensors(device)
            elif semantic_proto_residual:
                raise FileNotFoundError(f"Missing semantic prototype cache: {proto_path}")
        if token_path_raw:
            token_path = Path(str(token_path_raw))
            if not token_path.is_absolute():
                token_path = resolve_bundle_path(str(token_path_raw), BUNDLE_DIR)
            if token_path.exists():
                token_targets = KaraOneSemanticTokenTargets(token_path)
    split_protocol = "subject_holdout" if args.split == "subject_test" else str(cfg["data"].get("split_protocol", "trial"))
    ds = KaraOneTrialDataset(
        data_root=root,
        targets=targets,
        split=args.split,
        stages=tuple(ckpt["stages"]),
        split_protocol=split_protocol,
        heldout_subjects=cfg["data"].get("heldout_subjects", ["P02", "MM21"]),
        eeg_len=int(cfg["data"].get("eeg_len", 1280)),
        semantic_token_targets=token_targets,
    )
    duration_sec = float(cfg["audio"].get("duration_sec", 2.0))
    audio_cfg = cfg.get("audio", {})
    tgt_cfg = cfg.get("target", {})
    if target_kind == "mel":
        backend = load_mel_vocoder(
            AudioFeatureConfig(
                sample_rate=int(audio_cfg.get("sample_rate", 16000)),
                n_mels=int(tgt_cfg.get("n_mels", 80)),
                mel_n_fft=int(tgt_cfg.get("mel_n_fft", 1024)),
                mel_hop=int(tgt_cfg.get("mel_hop", 256)),
                griffinlim_iters=int(cfg.get("vocoder", {}).get("griffinlim_iters", 100)),
                griffinlim_momentum=float(cfg.get("vocoder", {}).get("griffinlim_momentum", 0.99)),
            )
        )
    else:
        backend = build_codec_backend(
            str(resolve_bundle_path(cfg["targets"]["codec_model_name_or_path"], BUNDLE_DIR)),
            duration_sec=duration_sec,
            bandwidth=float(cfg["targets"].get("codec_bandwidth", 6.0)),
            local_files_only=bool(cfg["targets"].get("local_files_only", True)),
        )
    sample_rate = int(backend.sample_rate)
    print(
        f"[synth] target={target_kind} vocoder={'griffinlim' if target_kind=='mel' else 'encodec'} "
        f"mode={'speech_core_residual_mean' if speech_core_objective else ('semantic_prototype_residual' if semantic_proto_residual else ('eeg_audio_direct' if eeg_audio_direct else ('residual_global_mean' if residual_mean else 'direct_target')))} "
        f"lag_correction={use_lag_correction} local_loudness={local_loudness_synthesis} "
        f"temporal_elastic={temporal_elastic_objective} sr={sample_rate}"
    )
    out_dir = ensure_dir(
        args.out_dir
        or (Path(args.checkpoint).resolve().parents[1] / f"wav_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}")
    )
    mean_wav = backend.decode(render_targets.global_mean_raw.astype(np.float32), decoder_scales=render_targets.default_decoder_scales)
    n_out = len(ds) if int(args.limit) <= 0 else min(int(args.limit), len(ds))
    print(f"[synth] reconstructing {n_out}/{len(ds)} trials of split={args.split}")
    manifest = []
    metric_rows: list[dict] = []
    for idx in range(n_out):
        item = ds[idx]
        entry = ds.entries[idx]
        with torch.no_grad():
            valid_len = item["eeg_valid_len"].view(1).to(device)
            pred_out = model(
                item["eeg"].unsqueeze(0).to(device),
                item["subject_idx"].view(1).to(device),
                item["stage_idx"].view(1).to(device),
                valid_len,
            )
            zero_out = model(
                torch.zeros_like(item["eeg"]).unsqueeze(0).to(device),
                item["subject_idx"].view(1).to(device),
                item["stage_idx"].view(1).to(device),
                valid_len,
            )
        pred = pred_out["pred_latent"].squeeze(0).cpu().numpy()
        zero = zero_out["pred_latent"].squeeze(0).cpu().numpy()
        semantic_proto = None
        zero_semantic_proto = None
        oracle_label_proto = None
        oracle_semantic_proto = None
        if prototype_tensors is not None and "semantic_token_logits" in pred_out:
            if prototype_tensors is None:
                raise RuntimeError("semantic prototype tensors are not initialized")
            with torch.no_grad():
                semantic_proto_t = prototype_tensors.prototype_from_logits(pred_out["semantic_token_logits"], None)
                zero_semantic_proto_t = prototype_tensors.prototype_from_logits(zero_out["semantic_token_logits"], None)
                oracle_label_proto_t = prototype_tensors.label_prototype(item["label_idx"].view(1).to(device))
                if "semantic_token_targets" in item:
                    oracle_semantic_proto_t = prototype_tensors.prototype_from_token_targets(
                        item["semantic_token_targets"].view(1, -1).to(device),
                        item["semantic_token_mask"].view(1, -1).to(device) if "semantic_token_mask" in item else None,
                    )
                else:
                    oracle_semantic_proto_t = None
            semantic_proto = semantic_proto_t.squeeze(0).cpu().numpy()
            zero_semantic_proto = zero_semantic_proto_t.squeeze(0).cpu().numpy()
            oracle_label_proto = oracle_label_proto_t.squeeze(0).cpu().numpy()
            oracle_semantic_proto = (
                oracle_semantic_proto_t.squeeze(0).cpu().numpy() if oracle_semantic_proto_t is not None else None
            )
        if semantic_proto_residual and semantic_proto is not None and zero_semantic_proto is not None:
            pred = semantic_proto + pred
            zero = zero_semantic_proto + zero
        elif residual_mean:
            pred = targets.global_mean_norm.astype(np.float32) + pred
            zero = targets.global_mean_norm.astype(np.float32) + zero
        if semantic_proto_residual and prototype_tensors is not None:
            with torch.no_grad():
                pred_lag = prototype_tensors.lag_from_logits(pred_out["semantic_token_logits"], None) + pred_out.get(
                    "pred_lag_mu", torch.zeros(1, device=device)
                ).view(-1)
                zero_lag = prototype_tensors.lag_from_logits(zero_out["semantic_token_logits"], None) + zero_out.get(
                    "pred_lag_mu", torch.zeros(1, device=device)
                ).view(-1)
            pred_lag_sec = float(pred_lag[0].detach().cpu())
            zero_lag_sec = float(zero_lag[0].detach().cpu())
        else:
            pred_lag_sec = float(pred_out.get("pred_lag_mu", torch.zeros(1, device=device)).view(-1)[0].detach().cpu())
            zero_lag_sec = float(zero_out.get("pred_lag_mu", torch.zeros(1, device=device)).view(-1)[0].detach().cpu())
        lag_hop_sec = float(cfg.get("target", {}).get("mel_hop", 256)) / float(cfg.get("audio", {}).get("sample_rate", 16000))
        pred_lag_frames = int(np.clip(np.rint(pred_lag_sec / max(lag_hop_sec, 1e-6)), -pred.shape[0] + 1, pred.shape[0] - 1))
        zero_lag_frames = int(np.clip(np.rint(zero_lag_sec / max(lag_hop_sec, 1e-6)), -zero.shape[0] + 1, zero.shape[0] - 1))
        pred_unaligned = pred.copy()
        zero_unaligned = zero.copy()
        if use_lag_correction and not speech_core_objective:
            pred = shift_sequence_np(pred, -pred_lag_frames)
            zero = shift_sequence_np(zero, -zero_lag_frames)
        pred_log_rms = pred_out["pred_log_rms"].squeeze(0)
        zero_log_rms = zero_out["pred_log_rms"].squeeze(0)
        pred_frame_log_energy = (
            pred_out["pred_frame_log_energy"].squeeze(0).cpu().numpy()
            if "pred_frame_log_energy" in pred_out
            else None
        )
        pred_decoder_scale = targets.default_decoder_scales
        zero_decoder_scale = targets.default_decoder_scales
        if has_trained_scale_head:
            pred_decoder_scale = np.exp(pred_out["pred_log_decoder_scale"].squeeze(0).cpu().numpy()).astype(np.float32)
            zero_decoder_scale = np.exp(zero_out["pred_log_decoder_scale"].squeeze(0).cpu().numpy()).astype(np.float32)
            pred_decoder_scale = np.clip(pred_decoder_scale, 1e-4, 20.0)
            zero_decoder_scale = np.clip(zero_decoder_scale, 1e-4, 20.0)
        pred_core_wav = None
        zero_core_wav = None
        pred_active_unshifted_wav = None
        zero_active_local_scaled = None
        mean_active_core_scaled = None
        oracle_shift_pred_wav = None
        pred_duration_frames = int(pred.shape[0])
        if speech_core_objective:
            full_steps = int(getattr(targets, "full_target_steps", render_targets.T))
            pred_core_raw = denormalize_latent(pred, targets.target_mean, targets.target_std)
            zero_core_raw = denormalize_latent(zero, targets.target_mean, targets.target_std)
            pred_unaligned_raw = denormalize_latent(pred_unaligned, targets.target_mean, targets.target_std)
            if temporal_elastic_objective:
                pred_duration_frames = _predicted_duration_frames(pred_out, full_steps, pred.shape[0])
                zero_duration_frames = _predicted_duration_frames(zero_out, full_steps, zero.shape[0])
                pred_core_raw = _resample_2d(pred_core_raw, pred_duration_frames)
                zero_core_raw = _resample_2d(zero_core_raw, zero_duration_frames)
                pred_unaligned_raw = _resample_2d(pred_unaligned_raw, pred_duration_frames)
                pred_center_frame = _predicted_core_center_frame(pred_out, ckpt, full_steps)
                zero_center_frame = _predicted_core_center_frame(zero_out, ckpt, full_steps)
                pred_full_raw, pred_insert_frame = _insert_core_mel_centered(
                    pred_core_raw, targets.silence_floor_raw, pred_center_frame, full_steps
                )
                zero_full_raw, zero_insert_frame = _insert_core_mel_centered(
                    zero_core_raw, targets.silence_floor_raw, zero_center_frame, full_steps
                )
                pred_unaligned_full_raw, _ = _insert_core_mel_centered(
                    pred_unaligned_raw,
                    targets.silence_floor_raw,
                    int(ckpt.get("speech_core_default_insert_frame", 0)),
                    full_steps,
                )
                if pred_frame_log_energy is not None:
                    pred_frame_log_energy = _resample_1d(pred_frame_log_energy, pred_duration_frames)
                pred_frame_log_energy = _core_energy_to_full(pred_frame_log_energy, pred_insert_frame, full_steps)
                pred_active_unshifted_wav = backend.decode(
                    pred_unaligned_full_raw, decoder_scales=render_targets.default_decoder_scales
                )
                if "active_center_frame" in item:
                    oracle_full_raw, _ = _insert_core_mel_centered(
                        pred_core_raw,
                        targets.silence_floor_raw,
                        int(item["active_center_frame"].item()),
                        full_steps,
                    )
                    oracle_shift_pred_wav = backend.decode(
                        oracle_full_raw, decoder_scales=render_targets.default_decoder_scales
                    )
                mean_core = _resample_2d(targets.global_mean_raw.astype(np.float32), pred_duration_frames)
                mean_full_raw, _ = _insert_core_mel_centered(
                    mean_core,
                    targets.silence_floor_raw,
                    int(ckpt.get("speech_core_default_insert_frame", 0)),
                    full_steps,
                )
                mean_active_core_scaled = backend.decode(mean_full_raw, decoder_scales=render_targets.default_decoder_scales)
            else:
                pred_insert_frame = _predicted_core_insert_frame(pred_out, ckpt, full_steps, pred.shape[0])
                zero_insert_frame = _predicted_core_insert_frame(zero_out, ckpt, full_steps, zero.shape[0])
                pred_full_raw = _insert_core_mel(pred_core_raw, targets.silence_floor_raw, pred_insert_frame, full_steps)
                zero_full_raw = _insert_core_mel(zero_core_raw, targets.silence_floor_raw, zero_insert_frame, full_steps)
                pred_unaligned_full_raw = _insert_core_mel(
                    pred_unaligned_raw,
                    targets.silence_floor_raw,
                    int(ckpt.get("speech_core_default_insert_frame", 0)),
                    full_steps,
                )
                pred_frame_log_energy = _core_energy_to_full(pred_frame_log_energy, pred_insert_frame, full_steps)
            pred_wav = backend.decode(pred_full_raw, decoder_scales=render_targets.default_decoder_scales)
            zero_wav = backend.decode(zero_full_raw, decoder_scales=render_targets.default_decoder_scales)
            pred_core_wav = backend.decode(pred_core_raw, decoder_scales=render_targets.default_decoder_scales)
            zero_core_wav = backend.decode(zero_core_raw, decoder_scales=render_targets.default_decoder_scales)
        else:
            pred_insert_frame = 0
            zero_insert_frame = 0
            pred_wav = backend.decode(
                denormalize_latent(pred, targets.target_mean, targets.target_std),
                decoder_scales=pred_decoder_scale,
            )
            zero_wav = backend.decode(
                denormalize_latent(zero, targets.target_mean, targets.target_std),
                decoder_scales=zero_decoder_scale,
            )
            pred_unaligned_full_raw = denormalize_latent(pred_unaligned, targets.target_mean, targets.target_std)
        semantic_proto_wav = (
            backend.decode(
                denormalize_latent(semantic_proto, targets.target_mean, targets.target_std),
                decoder_scales=pred_decoder_scale,
            )
            if semantic_proto is not None
            else None
        )
        oracle_label_proto_wav = (
            backend.decode(
                denormalize_latent(oracle_label_proto, targets.target_mean, targets.target_std),
                decoder_scales=targets.default_decoder_scales,
            )
            if oracle_label_proto is not None
            else None
        )
        oracle_semantic_proto_wav = (
            backend.decode(
                denormalize_latent(oracle_semantic_proto, targets.target_mean, targets.target_std),
                decoder_scales=targets.default_decoder_scales,
            )
            if oracle_semantic_proto is not None
            else None
        )
        oracle = backend.decode(
            render_targets.raw_target(entry.subject, entry.trial_index).astype(np.float32),
            decoder_scales=render_targets.decoder_scale(entry.subject, entry.trial_index),
        )
        original = load_wav_fixed(
            root / render_targets.audio_path(entry.subject, entry.trial_index),
            sample_rate=sample_rate,
            n_samples=int(round(sample_rate * duration_sec)),
            normalize=str(cfg["audio"].get("normalize", "rms")),
            target_rms=float(cfg["audio"].get("target_rms", 0.08)),
            max_gain=float(cfg["audio"].get("max_gain", 10.0)),
        )
        tag = f"{entry.subject}_{entry.label.replace('/', '')}_{entry.stage}_t{entry.trial_index:03d}"
        oracle_kind = "oracle_encodec" if target_kind == "encodec_latent" else "oracle_griffinlim"
        mel_hop = int(tgt_cfg.get("mel_hop", 256))
        pred_target_rms = float(np.exp(float(pred_log_rms.item())))
        zero_target_rms = float(np.exp(float(zero_log_rms.item())))
        pred_env_wav = _calibrate_frame_envelope(pred_wav, pred_frame_log_energy, hop=mel_hop)
        pred_scaled_wav = _scale_to_rms(pred_wav, pred_target_rms)
        pred_env_scaled_wav = _scale_to_rms(pred_env_wav, pred_target_rms)
        zeroeeg_scaled_wav = _scale_to_rms(zero_wav, zero_target_rms)
        local_loudness_mask = None
        pred_core_local_scaled = None
        pred_env_local_scaled = None
        if speech_core_objective and local_loudness_synthesis:
            local_loudness_mask = _local_loudness_mask(
                pred_frame_log_energy,
                pred_insert_frame,
                pred_duration_frames if temporal_elastic_objective else pred.shape[0],
                len(pred_wav),
                hop=mel_hop,
            )
            pred_env_local_scaled = _scale_region_to_rms(
                pred_env_wav,
                pred_target_rms,
                local_loudness_mask,
                max_gain=12.0,
                outside_attenuation=0.08,
                max_full_rms=max(0.12, 1.5 * pred_target_rms),
            )
            if pred_core_wav is not None:
                pred_core_local_scaled = _scale_region_to_rms(
                    pred_core_wav,
                    pred_target_rms,
                    np.ones(len(pred_core_wav), dtype=bool),
                    max_gain=12.0,
                    outside_attenuation=1.0,
                    max_full_rms=max(0.12, 1.5 * pred_target_rms),
                )
        wavs = {
            "original": original,
            oracle_kind: oracle,
            "mean_latent": mean_wav,
            "zeroeeg": zero_wav,
            "pred_unaligned": backend.decode(pred_unaligned_full_raw, decoder_scales=render_targets.default_decoder_scales),
            "pred": pred_wav,
            "pred_scaled": pred_scaled_wav,
            "pred_env_scaled": pred_env_scaled_wav,
            "zeroeeg_scaled": zeroeeg_scaled_wav,
        }
        if temporal_elastic_objective and pred_env_local_scaled is not None:
            wavs["pred_active_local_scaled"] = pred_env_local_scaled
        if temporal_elastic_objective and pred_active_unshifted_wav is not None:
            wavs["pred_active_local_unshifted"] = _scale_to_rms(pred_active_unshifted_wav, pred_target_rms)
        if temporal_elastic_objective:
            wavs["zeroeeg_active_local_scaled"] = _scale_to_rms(zero_wav, zero_target_rms)
        if mean_active_core_scaled is not None:
            wavs["mean_active_core_scaled"] = _scale_to_rms(mean_active_core_scaled, pred_target_rms)
        if oracle_shift_pred_wav is not None:
            wavs["oracle_shift_pred"] = _scale_to_rms(oracle_shift_pred_wav, pred_target_rms)
        if pred_core_wav is not None:
            wavs["pred_core"] = pred_core_wav
        if pred_core_local_scaled is not None:
            wavs["pred_core_local_scaled"] = pred_core_local_scaled
        if pred_env_local_scaled is not None:
            wavs["pred_env_local_scaled"] = pred_env_local_scaled
        if zero_core_wav is not None:
            wavs["zeroeeg_core"] = zero_core_wav
        if semantic_proto_wav is not None:
            wavs["semantic_proto"] = semantic_proto_wav
            wavs["semantic_proto_scaled"] = _scale_to_rms(semantic_proto_wav, float(np.exp(float(pred_log_rms.item()))))
        if oracle_label_proto_wav is not None:
            wavs["oracle_label_proto"] = oracle_label_proto_wav
        if oracle_semantic_proto_wav is not None:
            wavs["oracle_semantic_proto"] = oracle_semantic_proto_wav
        for kind, wav in wavs.items():
            filename = f"{tag}_{kind}.wav"
            save_wav(out_dir / filename, wav, sample_rate)
            manifest.append([entry.subject, entry.label, entry.stage, entry.trial_index, args.split, kind, filename, _rms(wav)])
        # Oracle = GT target through the SAME vocoder => the vocoder ceiling. Reporting
        # pred vs oracle (not just vs original) separates "model error" from "vocoder loss".
        pred_active = _active_metrics(wavs["pred_scaled"], original)
        pred_env_active = _active_metrics(wavs["pred_env_scaled"], original)
        pred_env_local_active = (
            _active_metrics(wavs["pred_env_local_scaled"], original) if "pred_env_local_scaled" in wavs else None
        )
        pred_active_local_active = (
            _active_metrics(wavs["pred_active_local_scaled"], original) if "pred_active_local_scaled" in wavs else None
        )
        pred_best_shift_corr, pred_best_shift_sec = _best_shift_env_corr(
            wavs.get("pred_active_local_scaled", wavs["pred_env_scaled"]),
            original,
            hop=mel_hop,
            sr=sample_rate,
        )
        oracle_active = _active_metrics(oracle, original)
        semantic_proto_active = _active_metrics(wavs["semantic_proto_scaled"], original) if "semantic_proto_scaled" in wavs else None
        oracle_label_proto_active = _active_metrics(wavs["oracle_label_proto"], original) if "oracle_label_proto" in wavs else None
        metric_row = {
            "subject": entry.subject,
            "label": entry.label,
            "trial_index": int(entry.trial_index),
            "pred_env_corr": _env_corr(wavs["pred_scaled"], original),
            "pred_env_scaled_env_corr": _env_corr(wavs["pred_env_scaled"], original),
            "oracle_env_corr": _env_corr(oracle, original),
            "pred_rms_over_orig": _rms(wavs["pred_scaled"]) / max(_rms(original), 1e-8),
            "pred_env_scaled_rms_over_orig": _rms(wavs["pred_env_scaled"]) / max(_rms(original), 1e-8),
            "oracle_rms_over_orig": _rms(oracle) / max(_rms(original), 1e-8),
            "pred_active_env_corr": pred_active["active_env_corr"],
            "pred_env_scaled_active_env_corr": pred_env_active["active_env_corr"],
            "oracle_active_env_corr": oracle_active["active_env_corr"],
            "pred_voiced_rms_over_orig": pred_active["voiced_rms_over_orig"],
            "pred_env_scaled_voiced_rms_over_orig": pred_env_active["voiced_rms_over_orig"],
            "oracle_voiced_rms_over_orig": oracle_active["voiced_rms_over_orig"],
            "pred_peak_over_orig": pred_active["peak_over_orig"],
            "pred_env_scaled_peak_over_orig": pred_env_active["peak_over_orig"],
            "oracle_peak_over_orig": oracle_active["peak_over_orig"],
            "pred_active_duration_ratio": pred_active["active_duration_ratio"],
            "pred_env_scaled_active_duration_ratio": pred_env_active["active_duration_ratio"],
            "oracle_active_duration_ratio": oracle_active["active_duration_ratio"],
            "best_shift_full_env_corr": pred_best_shift_corr,
            "best_shift_sec": pred_best_shift_sec,
            "active_segment_shape_corr": _active_segment_shape_corr(
                wavs.get("pred_active_local_scaled", wavs["pred_env_scaled"]),
                original,
                hop=mel_hop,
            ),
            "pred_lag_sec": pred_lag_sec,
            "pred_lag_frames": pred_lag_frames,
            "zero_lag_sec": zero_lag_sec,
            "pred_core_insert_frame": int(pred_insert_frame),
            "zero_core_insert_frame": int(zero_insert_frame),
            "lag_corrected": bool(use_lag_correction),
        }
        if pred_env_local_active is not None:
            metric_row.update(
                {
                    "pred_env_local_scaled_env_corr": _env_corr(wavs["pred_env_local_scaled"], original),
                    "pred_env_local_scaled_active_env_corr": pred_env_local_active["active_env_corr"],
                    "pred_env_local_scaled_voiced_rms_over_orig": pred_env_local_active["voiced_rms_over_orig"],
                    "pred_env_local_scaled_peak_over_orig": pred_env_local_active["peak_over_orig"],
                    "pred_env_local_scaled_active_duration_ratio": pred_env_local_active["active_duration_ratio"],
                    "pred_env_local_scaled_rms_over_orig": _rms(wavs["pred_env_local_scaled"]) / max(_rms(original), 1e-8),
                    "silence_leakage_wav": _silence_leakage_wav(wavs["pred_env_local_scaled"], local_loudness_mask),
                }
            )
        if pred_active_local_active is not None:
            metric_row.update(
                {
                    "pred_active_local_scaled_env_corr": _env_corr(wavs["pred_active_local_scaled"], original),
                    "pred_active_local_scaled_active_env_corr": pred_active_local_active["active_env_corr"],
                    "pred_active_local_scaled_voiced_rms_over_orig": pred_active_local_active["voiced_rms_over_orig"],
                    "pred_active_local_scaled_peak_over_orig": pred_active_local_active["peak_over_orig"],
                    "pred_active_local_scaled_active_duration_ratio": pred_active_local_active["active_duration_ratio"],
                    "pred_active_local_scaled_rms_over_orig": _rms(wavs["pred_active_local_scaled"]) / max(_rms(original), 1e-8),
                }
            )
        if semantic_proto_active is not None:
            metric_row.update(
                {
                    "semantic_proto_env_corr": _env_corr(wavs["semantic_proto_scaled"], original),
                    "semantic_proto_active_env_corr": semantic_proto_active["active_env_corr"],
                    "semantic_proto_voiced_rms_over_orig": semantic_proto_active["voiced_rms_over_orig"],
                    "semantic_proto_peak_over_orig": semantic_proto_active["peak_over_orig"],
                }
            )
        if oracle_label_proto_active is not None:
            metric_row.update(
                {
                    "oracle_label_proto_env_corr": _env_corr(wavs["oracle_label_proto"], original),
                    "oracle_label_proto_active_env_corr": oracle_label_proto_active["active_env_corr"],
                    "oracle_label_proto_voiced_rms_over_orig": oracle_label_proto_active["voiced_rms_over_orig"],
                    "oracle_label_proto_peak_over_orig": oracle_label_proto_active["peak_over_orig"],
                }
            )
        if asr_model is not None:
            asr_items = [
                ("pred_scaled", "pred_asr"),
                ("pred_env_scaled", "pred_env_scaled_asr"),
            ]
            if "pred_env_local_scaled" in wavs:
                asr_items.append(("pred_env_local_scaled", "pred_env_local_scaled_asr"))
            asr_items.extend(
                [
                    (oracle_kind, "oracle_asr"),
                    ("zeroeeg_scaled", "zeroeeg_asr"),
                    ("mean_latent", "mean_asr"),
                    ("original", "original_asr"),
                ]
            )
            for wav_kind, prefix in asr_items:
                wav_path = out_dir / f"{tag}_{wav_kind}.wav"
                try:
                    asr_metrics = transcribe_label_metrics(asr_model, wav_path, entry.label, fp16=str(device).startswith("cuda"))
                except Exception as exc:  # noqa: BLE001
                    asr_metrics = {"text": "", "label_hit": False, "cer": None, "wer": None, "candidate": "", "error": str(exc)}
                metric_row.update({f"{prefix}_{name}": value for name, value in asr_metrics.items()})
        metric_rows.append(metric_row)
    with (out_dir / "listening_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject", "label", "stage", "trial_index", "split", "wav_type", "file", "rms"])
        writer.writerows(manifest)

    def _mean(key: str) -> float:
        return float(np.mean([row[key] for row in metric_rows])) if metric_rows else 0.0

    def _mean_optional(key: str) -> float | None:
        values = [row[key] for row in metric_rows if key in row and row[key] is not None]
        return float(np.mean(values)) if values else None

    def _hit_rate(key: str) -> float | None:
        values = [float(bool(row[key])) for row in metric_rows if key in row]
        return float(np.mean(values)) if values else None

    synth_metrics = {
        "split": args.split,
        "n": len(metric_rows),
        "target_kind": "speech_core_mel" if speech_core_objective else target_kind,
        "vocoder": "griffinlim" if target_kind == "mel" else "encodec",
        "prediction_mode": "speech_core_residual_mean" if speech_core_objective else ("semantic_prototype_residual" if semantic_proto_residual else ("eeg_audio_direct" if eeg_audio_direct else ("residual_global_mean" if residual_mean else "direct_target"))),
        "lag_corrected": bool(use_lag_correction),
        "asr": asr_status,
        "oracle_kind": oracle_kind if metric_rows else ("oracle_encodec" if target_kind == "encodec_latent" else "oracle_griffinlim"),
        "pred_env_corr_mean": _mean("pred_env_corr"),
        "pred_env_scaled_env_corr_mean": _mean("pred_env_scaled_env_corr"),
        "pred_env_local_scaled_env_corr_mean": _mean_optional("pred_env_local_scaled_env_corr"),
        "pred_active_local_scaled_env_corr_mean": _mean_optional("pred_active_local_scaled_env_corr"),
        "oracle_env_corr_mean": _mean("oracle_env_corr"),  # vocoder ceiling
        "pred_rms_over_orig_mean": _mean("pred_rms_over_orig"),
        "pred_env_scaled_rms_over_orig_mean": _mean("pred_env_scaled_rms_over_orig"),
        "pred_env_local_scaled_rms_over_orig_mean": _mean_optional("pred_env_local_scaled_rms_over_orig"),
        "pred_active_local_scaled_rms_over_orig_mean": _mean_optional("pred_active_local_scaled_rms_over_orig"),
        "oracle_rms_over_orig_mean": _mean("oracle_rms_over_orig"),
        "pred_active_env_corr_mean": _mean("pred_active_env_corr"),
        "pred_env_scaled_active_env_corr_mean": _mean("pred_env_scaled_active_env_corr"),
        "pred_env_local_scaled_active_env_corr_mean": _mean_optional("pred_env_local_scaled_active_env_corr"),
        "pred_active_local_scaled_active_env_corr_mean": _mean_optional("pred_active_local_scaled_active_env_corr"),
        "oracle_active_env_corr_mean": _mean("oracle_active_env_corr"),
        "pred_voiced_rms_over_orig_mean": _mean("pred_voiced_rms_over_orig"),
        "pred_env_scaled_voiced_rms_over_orig_mean": _mean("pred_env_scaled_voiced_rms_over_orig"),
        "pred_env_local_scaled_voiced_rms_over_orig_mean": _mean_optional("pred_env_local_scaled_voiced_rms_over_orig"),
        "pred_active_local_scaled_voiced_rms_over_orig_mean": _mean_optional("pred_active_local_scaled_voiced_rms_over_orig"),
        "oracle_voiced_rms_over_orig_mean": _mean("oracle_voiced_rms_over_orig"),
        "pred_peak_over_orig_mean": _mean("pred_peak_over_orig"),
        "pred_env_scaled_peak_over_orig_mean": _mean("pred_env_scaled_peak_over_orig"),
        "pred_env_local_scaled_peak_over_orig_mean": _mean_optional("pred_env_local_scaled_peak_over_orig"),
        "pred_active_local_scaled_peak_over_orig_mean": _mean_optional("pred_active_local_scaled_peak_over_orig"),
        "oracle_peak_over_orig_mean": _mean("oracle_peak_over_orig"),
        "pred_active_duration_ratio_mean": _mean("pred_active_duration_ratio"),
        "pred_env_scaled_active_duration_ratio_mean": _mean("pred_env_scaled_active_duration_ratio"),
        "pred_env_local_scaled_active_duration_ratio_mean": _mean_optional("pred_env_local_scaled_active_duration_ratio"),
        "pred_active_local_scaled_active_duration_ratio_mean": _mean_optional("pred_active_local_scaled_active_duration_ratio"),
        "oracle_active_duration_ratio_mean": _mean("oracle_active_duration_ratio"),
        "best_shift_full_env_corr_mean": _mean("best_shift_full_env_corr"),
        "best_shift_sec_mean": _mean("best_shift_sec"),
        "active_segment_shape_corr_mean": _mean("active_segment_shape_corr"),
        "silence_leakage_wav_mean": _mean_optional("silence_leakage_wav"),
        "pred_lag_sec_mean": _mean("pred_lag_sec"),
        "semantic_proto_env_corr_mean": _mean_optional("semantic_proto_env_corr"),
        "semantic_proto_active_env_corr_mean": _mean_optional("semantic_proto_active_env_corr"),
        "semantic_proto_voiced_rms_over_orig_mean": _mean_optional("semantic_proto_voiced_rms_over_orig"),
        "semantic_proto_peak_over_orig_mean": _mean_optional("semantic_proto_peak_over_orig"),
        "oracle_label_proto_env_corr_mean": _mean_optional("oracle_label_proto_env_corr"),
        "oracle_label_proto_active_env_corr_mean": _mean_optional("oracle_label_proto_active_env_corr"),
        "oracle_label_proto_voiced_rms_over_orig_mean": _mean_optional("oracle_label_proto_voiced_rms_over_orig"),
        "oracle_label_proto_peak_over_orig_mean": _mean_optional("oracle_label_proto_peak_over_orig"),
        "pred_asr_label_acc": _hit_rate("pred_asr_label_hit"),
        "pred_env_scaled_asr_label_acc": _hit_rate("pred_env_scaled_asr_label_hit"),
        "pred_env_local_scaled_asr_label_acc": _hit_rate("pred_env_local_scaled_asr_label_hit"),
        "oracle_asr_label_acc": _hit_rate("oracle_asr_label_hit"),
        "zeroeeg_asr_label_acc": _hit_rate("zeroeeg_asr_label_hit"),
        "mean_asr_label_acc": _hit_rate("mean_asr_label_hit"),
        "original_asr_label_acc": _hit_rate("original_asr_label_hit"),
        "pred_asr_cer_mean": _mean_optional("pred_asr_cer"),
        "pred_env_scaled_asr_cer_mean": _mean_optional("pred_env_scaled_asr_cer"),
        "pred_env_local_scaled_asr_cer_mean": _mean_optional("pred_env_local_scaled_asr_cer"),
        "oracle_asr_cer_mean": _mean_optional("oracle_asr_cer"),
        "zeroeeg_asr_cer_mean": _mean_optional("zeroeeg_asr_cer"),
        "mean_asr_cer_mean": _mean_optional("mean_asr_cer"),
        "original_asr_cer_mean": _mean_optional("original_asr_cer"),
        "per_trial": metric_rows,
    }
    (out_dir / "synth_metrics.json").write_text(json.dumps(synth_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[oracle] env_corr pred={synth_metrics['pred_env_corr_mean']:.3f} "
        f"env_scaled={synth_metrics['pred_env_scaled_env_corr_mean']:.3f} "
        f"oracle(ceiling)={synth_metrics['oracle_env_corr_mean']:.3f} | "
        f"active_env pred={synth_metrics['pred_active_env_corr_mean']:.3f} "
        f"env_scaled={synth_metrics['pred_env_scaled_active_env_corr_mean']:.3f} "
        f"oracle={synth_metrics['oracle_active_env_corr_mean']:.3f} | "
        f"voiced_rms/orig pred={synth_metrics['pred_voiced_rms_over_orig_mean']:.3f} "
        f"env_scaled={synth_metrics['pred_env_scaled_voiced_rms_over_orig_mean']:.3f} "
        f"oracle={synth_metrics['oracle_voiced_rms_over_orig_mean']:.3f}"
    )
    if synth_metrics["pred_env_local_scaled_env_corr_mean"] is not None:
        print(
            f"[local] env_corr={synth_metrics['pred_env_local_scaled_env_corr_mean']:.3f} "
            f"active_env={synth_metrics['pred_env_local_scaled_active_env_corr_mean']:.3f} "
            f"voiced_rms/orig={synth_metrics['pred_env_local_scaled_voiced_rms_over_orig_mean']:.3f} "
            f"peak/orig={synth_metrics['pred_env_local_scaled_peak_over_orig_mean']:.3f} "
            f"silence_leakage={synth_metrics['silence_leakage_wav_mean']:.3f}"
        )
    if synth_metrics["pred_active_local_scaled_env_corr_mean"] is not None:
        print(
            f"[v5-active] best_shift_env={synth_metrics['best_shift_full_env_corr_mean']:.3f} "
            f"active_shape={synth_metrics['active_segment_shape_corr_mean']:.3f} "
            f"duration_ratio={synth_metrics['pred_active_local_scaled_active_duration_ratio_mean']:.3f} "
            f"rms/orig={synth_metrics['pred_active_local_scaled_voiced_rms_over_orig_mean']:.3f} "
            f"peak/orig={synth_metrics['pred_active_local_scaled_peak_over_orig_mean']:.3f}"
        )
    print(f"[done] wrote {len(manifest)} wav rows to {out_dir}")


if __name__ == "__main__":
    main()
