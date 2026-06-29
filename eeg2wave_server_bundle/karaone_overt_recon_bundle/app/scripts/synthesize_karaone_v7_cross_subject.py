from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.audio_features import AudioFeatureConfig, load_mel_vocoder
from src.karaone_recon.cross_subject_v7 import (
    KaraOneEEGFeatureCache,
    KaraOneV7Config,
    KaraOneV7CrossSubject,
    KaraOneV7Dataset,
)
from src.karaone_recon.data import KaraOneTrialDataset
from src.karaone_recon.targets import KaraOneTargets
from src.utils import ensure_dir, load_simple_yaml, load_wav_fixed, resolve_bundle_path, save_wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize KaraOne wavs from v7 cross-subject checkpoint.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="subject_test", choices=["train", "val", "test", "subject_val", "subject_test"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--device", default=None)
    parser.add_argument("--retrieval-topk", type=int, default=None)
    parser.add_argument("--retrieval-temperature", type=float, default=None)
    return parser.parse_args()


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)) + 1e-12)


def _limit_audio(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    out = np.asarray(audio, dtype=np.float32)
    max_abs = float(np.max(np.abs(out))) if out.size else 0.0
    if max_abs > peak:
        out = out * (peak / max_abs)
    return np.clip(out, -peak, peak).astype(np.float32)


def _resample_1d(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros(int(n), dtype=np.float64)
    if values.size == int(n):
        return values
    return np.interp(np.linspace(0, 1, int(n)), np.linspace(0, 1, values.size), values)


def _resample_2d(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] == int(n):
        return values.astype(np.float32)
    out = np.empty((int(n), values.shape[1]), dtype=np.float32)
    src = np.linspace(0, 1, values.shape[0])
    dst = np.linspace(0, 1, int(n))
    for j in range(values.shape[1]):
        out[:, j] = np.interp(dst, src, values[:, j]).astype(np.float32)
    return out


def _denorm(core_norm: np.ndarray, targets: KaraOneTargets) -> np.ndarray:
    return (core_norm * targets.target_std.reshape(1, -1) + targets.target_mean.reshape(1, -1)).astype(np.float32)


def _insert_centered(core_raw: np.ndarray, floor_raw: np.ndarray, center_frame: int, full_steps: int) -> tuple[np.ndarray, int]:
    floor = np.asarray(floor_raw, dtype=np.float32)
    if floor.ndim == 1:
        floor = np.tile(floor.reshape(1, -1), (int(full_steps), 1))
    if floor.shape[0] < int(full_steps):
        floor = np.concatenate([floor, np.repeat(floor[-1:], int(full_steps) - floor.shape[0], axis=0)], axis=0)
    canvas = floor[: int(full_steps)].copy()
    start = int(round(float(center_frame) - 0.5 * core_raw.shape[0]))
    start = int(np.clip(start, 0, max(0, int(full_steps) - core_raw.shape[0])))
    valid = min(core_raw.shape[0], int(full_steps) - start)
    if valid > 0:
        canvas[start : start + valid] = core_raw[:valid]
    return canvas.astype(np.float32), start


def _scale_region_bounded(
    audio: np.ndarray,
    target_rms: float,
    median_rms: float,
    start_frame: int,
    n_frames: int,
    hop: int,
) -> np.ndarray:
    out = np.asarray(audio, dtype=np.float32).copy()
    start = int(max(0, start_frame * hop))
    end = int(min(out.size, (start_frame + max(1, n_frames)) * hop))
    if end <= start:
        return _limit_audio(out)
    target = float(np.clip(target_rms, 0.6 * median_rms, 1.4 * median_rms))
    gain = target / max(_rms(out[start:end]), 1e-8)
    gain = float(np.clip(gain, 0.25, 4.0))
    out[start:end] *= gain
    if start > 0:
        out[:start] *= 0.10
    if end < out.size:
        out[end:] *= 0.10
    full = _rms(out)
    if full > 0.10:
        out *= 0.10 / max(full, 1e-8)
    return _limit_audio(out)


def _envelope(audio: np.ndarray, hop: int = 256) -> np.ndarray:
    n = len(audio) // hop
    if n < 2:
        return np.zeros(2, dtype=np.float64)
    frames = np.asarray(audio[: n * hop], dtype=np.float64).reshape(n, hop)
    return np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _env_corr(a: np.ndarray, b: np.ndarray, hop: int = 256) -> float:
    ea, eb = _envelope(a, hop), _envelope(b, hop)
    m = min(ea.size, eb.size)
    return _corr(ea[:m], eb[:m])


def _best_shift_env_corr(a: np.ndarray, b: np.ndarray, hop: int = 256, max_shift: int = 48) -> tuple[float, float]:
    ea, eb = _envelope(a, hop), _envelope(b, hop)
    best, shift_best = -1.0, 0
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
        c = _corr(aa[:m], bb[:m])
        if c > best:
            best, shift_best = c, shift
    return float(best), float(shift_best * hop / 16000.0)


def _active_mask(env: np.ndarray) -> np.ndarray:
    if env.size == 0 or float(env.max(initial=0.0)) <= 1e-10:
        return np.ones(max(env.size, 1), dtype=bool)
    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med))) + 1e-12
    mask = env >= max(0.2 * float(env.max()), med + 2.0 * mad)
    if not mask.any():
        mask[int(np.argmax(env))] = True
    return mask


def _active_shape(candidate: np.ndarray, original: np.ndarray, hop: int = 256, n: int = 64) -> float:
    ce, oe = _envelope(candidate, hop), _envelope(original, hop)
    cm, om = _active_mask(ce), _active_mask(oe)
    c = ce[np.flatnonzero(cm)[0] : np.flatnonzero(cm)[-1] + 1]
    o = oe[np.flatnonzero(om)[0] : np.flatnonzero(om)[-1] + 1]
    return _corr(_resample_1d(c, n), _resample_1d(o, n))


def _active_metrics(candidate: np.ndarray, original: np.ndarray, hop: int = 256) -> dict[str, float]:
    ce, oe = _envelope(candidate, hop), _envelope(original, hop)
    m = min(ce.size, oe.size)
    ce, oe = ce[:m], oe[:m]
    om = _active_mask(oe)
    cm = _active_mask(ce)
    sample_mask = np.repeat(om, hop)[: min(candidate.size, original.size)]
    if sample_mask.size < min(candidate.size, original.size):
        sample_mask = np.pad(sample_mask, (0, min(candidate.size, original.size) - sample_mask.size))
    sample_mask = sample_mask.astype(bool)
    if not sample_mask.any():
        sample_mask[:] = True
    cand = candidate[: sample_mask.size][sample_mask]
    orig = original[: sample_mask.size][sample_mask]
    return {
        "active_env_corr": _corr(ce[om], oe[om]) if om.sum() >= 2 else 0.0,
        "voiced_rms_over_orig": float(_rms(cand) / max(_rms(orig), 1e-8)),
        "peak_over_orig": float(np.max(np.abs(cand)) / max(float(np.max(np.abs(orig))), 1e-8)),
        "active_duration_ratio": float(cm.mean() / max(float(om.mean()), 1e-8)),
    }


def _silence_leakage(audio: np.ndarray, start_frame: int, n_frames: int, hop: int = 256) -> float:
    mask = np.zeros(audio.size, dtype=bool)
    start = int(max(0, start_frame * hop))
    end = int(min(audio.size, (start_frame + max(1, n_frames)) * hop))
    if end > start:
        mask[start:end] = True
    if not mask.any() or not (~mask).any():
        return 0.0
    return float(_rms(audio[~mask]) / max(_rms(audio[mask]), 1e-8))


def _retrieve(query: np.ndarray, bank_audio: np.ndarray, bank_core: np.ndarray, topk: int, temperature: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = query / np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-8)
    b = bank_audio / np.linalg.norm(bank_audio, axis=1, keepdims=True).clip(min=1e-8)
    scores = q @ b.T
    k = max(1, min(int(topk), bank_core.shape[0]))
    idx = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
    vals = np.take_along_axis(scores, idx, axis=1)
    weights = np.exp((vals - vals.max(axis=1, keepdims=True)) / max(float(temperature), 1e-4))
    weights = weights / weights.sum(axis=1, keepdims=True).clip(min=1e-8)
    return (bank_core[idx] * weights[..., None, None]).sum(axis=1).astype(np.float32), idx, weights.astype(np.float32)


def _weighted(values: np.ndarray, idx: np.ndarray, weights: np.ndarray, fallback: float) -> np.ndarray:
    if values is None or values.size == 0:
        return np.full(idx.shape[0], float(fallback), dtype=np.float32)
    return (values[idx] * weights).sum(axis=1).astype(np.float32)


def _audio_path(root: Path, targets: KaraOneTargets, subject: str, trial: int) -> Path:
    raw = Path(targets.audio_path(subject, trial))
    return raw if raw.is_absolute() else root / raw


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = KaraOneV7CrossSubject(KaraOneV7Config(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    core_path = Path(str(ckpt["core_cache"]))
    if not core_path.is_absolute():
        core_path = resolve_bundle_path(str(ckpt["core_cache"]), BUNDLE_DIR)
    feature_path = Path(str(ckpt["feature_cache"]))
    if not feature_path.is_absolute():
        feature_path = resolve_bundle_path(str(ckpt["feature_cache"]), BUNDLE_DIR)
    targets = KaraOneTargets(core_path, data_root=root)
    features = KaraOneEEGFeatureCache(feature_path)
    stages = tuple(str(x) for x in ckpt["stages"])
    subject_val, subject_test = str(ckpt.get("subject_val", "P02")), str(ckpt.get("subject_test", "MM21"))
    if args.split == "subject_val":
        heldout, actual_split, protocol = [subject_val], "subject_test", "subject_holdout"
    elif args.split == "subject_test":
        heldout, actual_split, protocol = [subject_test], "subject_test", "subject_holdout"
    else:
        heldout, actual_split, protocol = sorted({subject_val, subject_test}), args.split, "trial"
    base = KaraOneTrialDataset(
        data_root=root,
        targets=targets,
        split=actual_split,
        stages=stages,
        split_protocol=protocol,
        heldout_subjects=heldout,
        eeg_len=int(cfg["data"].get("eeg_len", 1280)),
    )
    ds = KaraOneV7Dataset(base, features)
    audio_cfg, tgt_cfg = cfg.get("audio", {}), cfg.get("target", {})
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
    sr, hop = int(backend.sample_rate), int(tgt_cfg.get("mel_hop", 256))
    n_samples = int(round(sr * float(audio_cfg.get("duration_sec", 2.0))))
    bank: dict[str, Any] = ckpt["train_bank"]
    bank_audio = np.asarray(bank["audio_embed"], dtype=np.float32)
    bank_core = np.asarray(bank["core_seq"], dtype=np.float32)
    bank_duration = np.asarray(bank.get("active_duration_frames", []), dtype=np.float32)
    bank_center = np.asarray(bank.get("active_center_frame", []), dtype=np.float32)
    bank_rms = np.asarray(bank.get("active_rms", []), dtype=np.float32)
    topk = int(args.retrieval_topk or ckpt.get("retrieval_topk", 3))
    temp = float(args.retrieval_temperature or ckpt.get("retrieval_temperature", 0.05))
    gate_passed = bool(ckpt.get("gate_passed", False))
    residual_scale = float(ckpt.get("lambda_mel_residual", 0.0)) if gate_passed else 0.0
    median_rms = float(np.median(bank_rms)) if bank_rms.size else 0.08
    out_dir = ensure_dir(args.out_dir or (Path(args.checkpoint).resolve().parents[1] / f"wav_v7_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}"))
    n_out = len(ds) if int(args.limit) <= 0 else min(int(args.limit), len(ds))
    print(f"[synth-v7] reconstructing {n_out}/{len(ds)} split={args.split} topk={topk} gate={gate_passed} residual_scale={residual_scale}")
    floor = np.asarray(getattr(targets, "silence_floor_raw", np.zeros((getattr(targets, "full_target_steps", targets.T), targets.D))), dtype=np.float32)
    full_steps = int(getattr(targets, "full_target_steps", 122))
    manifest: list[list[Any]] = []
    metric_rows: list[dict[str, float]] = []

    def render(core_norm: np.ndarray, duration: int, center: int, rms: float) -> tuple[np.ndarray, int, int]:
        core_raw = _resample_2d(_denorm(core_norm, targets), int(np.clip(duration, 2, full_steps)))
        canvas, start = _insert_centered(core_raw, floor, center, full_steps)
        wav = backend.decode(canvas)
        wav = wav[:n_samples] if wav.size >= n_samples else np.pad(wav, (0, n_samples - wav.size))
        wav = _scale_region_bounded(wav, rms, median_rms, start, core_raw.shape[0], hop)
        return wav, start, int(core_raw.shape[0])

    mean_core = bank_core.mean(axis=0).astype(np.float32)
    mean_dur = int(round(float(np.mean(bank_duration)))) if bank_duration.size else int(targets.T)
    mean_center = int(round(float(np.mean(bank_center)))) if bank_center.size else int(getattr(targets, "global_core_insert_frame", 0))
    for idx in range(n_out):
        item = ds[idx]
        entry = ds.entries[idx]
        with torch.no_grad():
            out = model(
                item["eeg"].unsqueeze(0).to(device),
                item["eeg_feature"].unsqueeze(0).to(device),
                item["eeg_envelope"].unsqueeze(0).to(device),
                item["stage_idx"].view(1).to(device),
                item["eeg_valid_len"].view(1).to(device),
            )
        pred_embed = out["eeg_embed"].detach().cpu().numpy().astype(np.float32)
        prior, ridx, weights = _retrieve(pred_embed, bank_audio, bank_core, topk, temp)
        duration = int(round(float(_weighted(bank_duration, ridx, weights, targets.T)[0])))
        center = int(round(float(_weighted(bank_center, ridx, weights, getattr(targets, "global_core_insert_frame", 0))[0])))
        rms = float(_weighted(bank_rms, ridx, weights, median_rms)[0])
        delta = out["pred_core_delta"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        pred_core = prior[0] + residual_scale * delta
        pred_wav, pred_start, pred_frames = render(pred_core, duration, center, rms)
        prior_wav, _, _ = render(prior[0], duration, center, rms)
        mean_wav, _, _ = render(mean_core, mean_dur, mean_center, median_rms)
        oracle_norm = item["target_seq"].numpy().astype(np.float32)
        oracle_duration = int(item.get("active_duration_frames", torch.tensor(targets.T)).item())
        oracle_center = int(item.get("active_center_frame", torch.tensor(getattr(targets, "global_core_insert_frame", 0))).item())
        oracle_rms = float(item.get("active_rms", torch.tensor(median_rms)).item())
        oracle_wav, _, _ = render(oracle_norm, oracle_duration, oracle_center, oracle_rms)
        original = load_wav_fixed(
            _audio_path(root, targets, entry.subject, entry.trial_index),
            sample_rate=sr,
            n_samples=n_samples,
            normalize=str(audio_cfg.get("normalize", "rms")),
            target_rms=float(audio_cfg.get("target_rms", 0.08)),
            max_gain=float(audio_cfg.get("max_gain", 10.0)),
        )
        wavs = {
            "original": original,
            "pred_v7": pred_wav,
            "pred_v7_retrieved_prior": prior_wav,
            "mean_train_core": mean_wav,
            "oracle_griffinlim": oracle_wav,
        }
        stem = f"{entry.subject}_trial{entry.trial_index:03d}_{entry.stage}_{entry.label}"
        for wav_type, audio in wavs.items():
            name = f"{stem}_{wav_type}.wav"
            save_wav(out_dir / name, audio, sr)
            manifest.append([entry.subject, entry.label, entry.stage, entry.trial_index, args.split, wav_type, name, _rms(audio)])
        active = _active_metrics(pred_wav, original, hop)
        best_corr, best_sec = _best_shift_env_corr(pred_wav, original, hop)
        metric_rows.append(
            {
                "env_corr": _env_corr(pred_wav, original, hop),
                "best_shift_full_env_corr": best_corr,
                "best_shift_sec": best_sec,
                "active_segment_shape_corr": _active_shape(pred_wav, original, hop),
                "active_env_corr": active["active_env_corr"],
                "voiced_rms_over_orig": active["voiced_rms_over_orig"],
                "peak_over_orig": active["peak_over_orig"],
                "active_duration_ratio": active["active_duration_ratio"],
                "silence_leakage": _silence_leakage(pred_wav, pred_start, pred_frames, hop),
                "loudness_in_bounds": float(0.6 <= active["voiced_rms_over_orig"] <= 1.4 and 0.6 <= active["peak_over_orig"] <= 1.4),
            }
        )
    with (out_dir / "listening_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject", "label", "stage", "trial_index", "split", "wav_type", "file", "rms"])
        writer.writerows(manifest)
    metrics: dict[str, Any] = {"n": int(n_out), "target_kind": "speech_core_mel", "prediction_mode": "cross_subject_v7", "gate_passed": gate_passed, "retrieval_topk": int(topk), "lambda_mel_residual": float(residual_scale)}
    if metric_rows:
        for key in metric_rows[0]:
            metrics[f"pred_v7_{key}_mean"] = float(np.mean([row[key] for row in metric_rows]))
        metrics["active_segment_shape_corr_mean"] = metrics["pred_v7_active_segment_shape_corr_mean"]
        metrics["best_shift_full_env_corr_mean"] = metrics["pred_v7_best_shift_full_env_corr_mean"]
        metrics["pred_active_local_scaled_active_duration_ratio_mean"] = metrics["pred_v7_active_duration_ratio_mean"]
        metrics["pred_active_local_scaled_voiced_rms_over_orig_mean"] = metrics["pred_v7_voiced_rms_over_orig_mean"]
        metrics["pred_active_local_scaled_peak_over_orig_mean"] = metrics["pred_v7_peak_over_orig_mean"]
        metrics["silence_leakage_wav_mean"] = metrics["pred_v7_silence_leakage_mean"]
    (out_dir / "synth_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wav_dir": str(out_dir), "n": n_out, "gate_passed": gate_passed, "synth_metrics": str(out_dir / "synth_metrics.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
