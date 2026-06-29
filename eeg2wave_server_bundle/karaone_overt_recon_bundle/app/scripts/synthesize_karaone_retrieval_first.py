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
import torch.nn.functional as F

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.audio_features import AudioFeatureConfig, load_mel_vocoder
from src.karaone_recon.data import KaraOneTrialDataset
from src.karaone_recon.retrieval_first import KaraOneRetrievalFirst, RetrievalFirstConfig
from src.karaone_recon.semantic_tokens import KaraOneSemanticTokenTargets
from src.karaone_recon.targets import KaraOneTargets
from src.utils import ensure_dir, load_simple_yaml, load_wav_fixed, resolve_bundle_path, save_wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize KaraOne wavs from a v6.1 retrieval-first checkpoint.")
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
    out = np.asarray(audio, dtype=np.float32).copy()
    max_abs = float(np.max(np.abs(out))) if out.size else 0.0
    if max_abs > float(peak):
        out *= float(peak) / max_abs
    return np.clip(out, -float(peak), float(peak)).astype(np.float32)


def _resample_1d(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if values.size == 0:
        return np.zeros(n, dtype=np.float64)
    if values.size == n:
        return values
    src = np.linspace(0.0, 1.0, values.size)
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, values)


def _resample_2d(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        values = values.reshape(-1, 1)
    if n <= 0:
        return np.zeros((0, values.shape[1]), dtype=np.float32)
    if values.shape[0] == n:
        return values.astype(np.float32)
    src = np.linspace(0.0, 1.0, values.shape[0])
    dst = np.linspace(0.0, 1.0, int(n))
    out = np.empty((int(n), values.shape[1]), dtype=np.float32)
    for j in range(values.shape[1]):
        out[:, j] = np.interp(dst, src, values[:, j]).astype(np.float32)
    return out


def _denorm_core(core_norm: np.ndarray, targets: KaraOneTargets) -> np.ndarray:
    return (core_norm * targets.target_std.reshape(1, -1) + targets.target_mean.reshape(1, -1)).astype(np.float32)


def _insert_centered(core_raw: np.ndarray, floor_raw: np.ndarray, center_frame: int, full_steps: int) -> tuple[np.ndarray, int]:
    core = np.asarray(core_raw, dtype=np.float32)
    floor = np.asarray(floor_raw, dtype=np.float32)
    if floor.ndim == 1:
        floor = np.tile(floor.reshape(1, -1), (int(full_steps), 1))
    if floor.shape[0] < int(full_steps):
        floor = np.concatenate([floor, np.repeat(floor[-1:, :], int(full_steps) - floor.shape[0], axis=0)], axis=0)
    floor = floor[: int(full_steps)]
    start = int(round(float(center_frame) - 0.5 * float(core.shape[0])))
    start = int(np.clip(start, 0, max(0, int(full_steps) - int(core.shape[0]))))
    canvas = floor.copy()
    valid = min(core.shape[0], int(full_steps) - start)
    if valid > 0:
        canvas[start : start + valid] = core[:valid]
    return canvas.astype(np.float32), start


def _scale_region(audio: np.ndarray, target_rms: float, start_frame: int, n_frames: int, hop: int, max_full_rms: float = 0.16) -> np.ndarray:
    out = np.asarray(audio, dtype=np.float32).copy()
    start = int(max(0, start_frame * hop))
    end = int(min(out.size, (start_frame + max(1, n_frames)) * hop))
    if end <= start:
        return _limit_audio(out)
    region = out[start:end]
    gain = float(target_rms) / max(_rms(region), 1e-8)
    gain = float(np.clip(gain, 1.0 / 12.0, 12.0))
    out[start:end] *= gain
    if start > 0:
        out[:start] *= 0.12
    if end < out.size:
        out[end:] *= 0.12
    full = _rms(out)
    if full > max_full_rms:
        out *= max_full_rms / max(full, 1e-8)
    return _limit_audio(out)


def _envelope(audio: np.ndarray, hop: int = 256) -> np.ndarray:
    n = len(audio) // hop
    if n < 2:
        return np.zeros(2, dtype=np.float64)
    frames = np.asarray(audio[: n * hop], dtype=np.float64).reshape(n, hop)
    return np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)


def _env_corr(a: np.ndarray, b: np.ndarray, hop: int = 256) -> float:
    ea, eb = _envelope(a, hop), _envelope(b, hop)
    m = min(ea.size, eb.size)
    if m < 2:
        return 0.0
    ea, eb = ea[:m] - ea[:m].mean(), eb[:m] - eb[:m].mean()
    return float((ea * eb).sum() / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))


def _best_shift_env_corr(a: np.ndarray, b: np.ndarray, hop: int = 256, max_shift: int = 48) -> tuple[float, float]:
    ea, eb = _envelope(a, hop), _envelope(b, hop)
    best_corr, best_shift = -1.0, 0
    for shift in range(-int(max_shift), int(max_shift) + 1):
        if shift < 0:
            aa, bb = ea[-shift:], eb[: ea.size + shift]
        elif shift > 0:
            aa, bb = ea[: ea.size - shift], eb[shift:]
        else:
            aa, bb = ea, eb
        m = min(aa.size, bb.size)
        if m < 2:
            continue
        aa, bb = aa[:m] - aa[:m].mean(), bb[:m] - bb[:m].mean()
        corr = float((aa * bb).sum() / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-8))
        if corr > best_corr:
            best_corr, best_shift = corr, shift
    return float(best_corr), float(best_shift * hop / 16000.0)


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


def _active_segment_shape_corr(candidate: np.ndarray, original: np.ndarray, hop: int = 256, core_len: int = 64) -> float:
    ce, oe = _envelope(candidate, hop), _envelope(original, hop)
    cm, om = _active_mask(ce), _active_mask(oe)
    if not cm.any() or not om.any():
        return 0.0
    c = ce[np.flatnonzero(cm)[0] : np.flatnonzero(cm)[-1] + 1]
    o = oe[np.flatnonzero(om)[0] : np.flatnonzero(om)[-1] + 1]
    c, o = _resample_1d(c, core_len), _resample_1d(o, core_len)
    c, o = c - c.mean(), o - o.mean()
    return float((c * o).sum() / (np.linalg.norm(c) * np.linalg.norm(o) + 1e-8))


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
    active_corr = 0.0
    if om.sum() >= 2:
        ca, oa = ce[om] - ce[om].mean(), oe[om] - oe[om].mean()
        active_corr = float((ca * oa).sum() / (np.linalg.norm(ca) * np.linalg.norm(oa) + 1e-8))
    return {
        "active_env_corr": active_corr,
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
    q = query / np.linalg.norm(query, axis=-1, keepdims=True).clip(min=1e-8)
    b = bank_audio / np.linalg.norm(bank_audio, axis=1, keepdims=True).clip(min=1e-8)
    scores = q @ b.T
    k = max(1, min(int(topk), bank_core.shape[0]))
    idx = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
    vals = np.take_along_axis(scores, idx, axis=1)
    weights = np.exp((vals - vals.max(axis=1, keepdims=True)) / max(float(temperature), 1e-4))
    weights = weights / weights.sum(axis=1, keepdims=True).clip(min=1e-8)
    core = (bank_core[idx] * weights[..., None, None]).sum(axis=1).astype(np.float32)
    return core, idx, weights.astype(np.float32)


def _weighted_meta(values: np.ndarray, idx: np.ndarray, weights: np.ndarray, fallback: float) -> np.ndarray:
    if values is None or values.size == 0:
        return np.full(idx.shape[0], float(fallback), dtype=np.float32)
    return (values[idx] * weights).sum(axis=1).astype(np.float32)


def _audio_path(root: Path, targets: KaraOneTargets, subject: str, trial: int) -> Path:
    raw = Path(targets.audio_path(subject, trial))
    return raw if raw.is_absolute() else root / raw


def _make_dataset(
    *,
    root: Path,
    core_targets: KaraOneTargets,
    split: str,
    stages: tuple[str, ...],
    heldout_subjects: list[str],
    eeg_len: int,
    token_targets: KaraOneSemanticTokenTargets | None,
) -> KaraOneTrialDataset:
    protocol = "subject_holdout" if split in {"subject_val", "subject_test"} else "trial"
    actual_split = "subject_test" if split in {"subject_val", "subject_test"} else split
    return KaraOneTrialDataset(
        data_root=root,
        targets=core_targets,
        split=actual_split,
        stages=stages,
        split_protocol=protocol,
        heldout_subjects=heldout_subjects,
        eeg_len=eeg_len,
        semantic_token_targets=token_targets,
    )


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = KaraOneRetrievalFirst(RetrievalFirstConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    core_path = Path(str(ckpt["core_cache"]))
    if not core_path.is_absolute():
        core_path = resolve_bundle_path(str(ckpt["core_cache"]), BUNDLE_DIR)
    targets = KaraOneTargets(core_path, data_root=root)
    token_path = str(ckpt.get("semantic_token_cache", ""))
    token_targets = None
    if token_path:
        token_candidate = Path(token_path)
        if not token_candidate.is_absolute():
            token_candidate = resolve_bundle_path(token_path, BUNDLE_DIR)
        if token_candidate.exists():
            token_targets = KaraOneSemanticTokenTargets(token_candidate)
    stages = tuple(str(item) for item in ckpt["stages"])
    subject_val = str(ckpt.get("subject_val", "P02"))
    subject_test = str(ckpt.get("subject_test", "MM21"))
    if args.split == "subject_val":
        heldout = [subject_val]
    elif args.split == "subject_test":
        heldout = [subject_test]
    else:
        heldout = sorted({subject_val, subject_test})
    ds = _make_dataset(
        root=root,
        core_targets=targets,
        split=args.split,
        stages=stages,
        heldout_subjects=heldout,
        eeg_len=int(cfg["data"].get("eeg_len", 1280)),
        token_targets=token_targets,
    )
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
    sr = int(backend.sample_rate)
    hop = int(tgt_cfg.get("mel_hop", 256))
    duration_sec = float(audio_cfg.get("duration_sec", 2.0))
    n_samples = int(round(sr * duration_sec))
    bank: dict[str, Any] = ckpt["train_bank"]
    bank_audio = np.asarray(bank["audio_embed"], dtype=np.float32)
    bank_core = np.asarray(bank["core_seq"], dtype=np.float32)
    bank_duration = np.asarray(bank.get("active_duration_frames", []), dtype=np.float32)
    bank_center = np.asarray(bank.get("active_center_frame", []), dtype=np.float32)
    bank_rms = np.asarray(bank.get("active_rms", []), dtype=np.float32)
    topk = int(args.retrieval_topk or ckpt.get("retrieval_topk", 3))
    temperature = float(args.retrieval_temperature or ckpt.get("retrieval_temperature", 0.05))
    residual_scale = float(ckpt.get("lambda_mel_residual", 0.0))
    out_dir = ensure_dir(
        args.out_dir
        or (Path(args.checkpoint).resolve().parents[1] / f"wav_retrieval_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}")
    )
    n_out = len(ds) if int(args.limit) <= 0 else min(int(args.limit), len(ds))
    print(
        f"[synth-v61] reconstructing {n_out}/{len(ds)} split={args.split} "
        f"topk={topk} residual_scale={residual_scale} sr={sr}"
    )
    manifest: list[list[Any]] = []
    metric_rows: list[dict[str, float]] = []
    mean_core_norm = bank_core.mean(axis=0).astype(np.float32)
    mean_core_raw = _denorm_core(mean_core_norm, targets)
    mean_duration = int(round(float(np.mean(bank_duration)))) if bank_duration.size else int(targets.T)
    mean_center = int(round(float(np.mean(bank_center)))) if bank_center.size else int(getattr(targets, "global_core_insert_frame", 0))
    mean_rms = float(np.mean(bank_rms)) if bank_rms.size else 0.08
    for idx in range(n_out):
        item = ds[idx]
        entry = ds.entries[idx]
        with torch.no_grad():
            out = model(
                item["eeg"].unsqueeze(0).to(device),
                item["stage_idx"].view(1).to(device),
                item["eeg_valid_len"].view(1).to(device),
            )
            zero_out = model(
                torch.zeros_like(item["eeg"]).unsqueeze(0).to(device),
                item["stage_idx"].view(1).to(device),
                item["eeg_valid_len"].view(1).to(device),
            )
        pred_embed = out["eeg_embed"].detach().cpu().numpy().astype(np.float32)
        zero_embed = zero_out["eeg_embed"].detach().cpu().numpy().astype(np.float32)
        pred_prior, pred_idx, pred_w = _retrieve(pred_embed, bank_audio, bank_core, topk, temperature)
        zero_prior, zero_idx, zero_w = _retrieve(zero_embed, bank_audio, bank_core, topk, temperature)
        pred_delta = out["pred_core_delta"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        pred_resid = pred_prior[0] + residual_scale * pred_delta
        pred_duration = int(round(float(_weighted_meta(bank_duration, pred_idx, pred_w, targets.T)[0])))
        zero_duration = int(round(float(_weighted_meta(bank_duration, zero_idx, zero_w, targets.T)[0])))
        pred_center = int(round(float(_weighted_meta(bank_center, pred_idx, pred_w, getattr(targets, "global_core_insert_frame", 0))[0])))
        zero_center = int(round(float(_weighted_meta(bank_center, zero_idx, zero_w, getattr(targets, "global_core_insert_frame", 0))[0])))
        pred_rms = float(_weighted_meta(bank_rms, pred_idx, pred_w, 0.08)[0])
        zero_rms = float(_weighted_meta(bank_rms, zero_idx, zero_w, 0.08)[0])
        floor = np.asarray(getattr(targets, "silence_floor_raw", np.zeros((getattr(targets, "full_target_steps", targets.T), targets.D))), dtype=np.float32)
        full_steps = int(getattr(targets, "full_target_steps", 122))

        def render(core_norm: np.ndarray, duration: int, center: int, rms: float) -> tuple[np.ndarray, int, int]:
            core_raw = _resample_2d(_denorm_core(core_norm, targets), int(np.clip(duration, 2, full_steps)))
            canvas, start = _insert_centered(core_raw, floor, center, full_steps)
            wav = backend.decode(canvas)
            wav = wav[:n_samples] if wav.size >= n_samples else np.pad(wav, (0, n_samples - wav.size))
            wav = _scale_region(wav, rms, start, core_raw.shape[0], hop=hop)
            return wav.astype(np.float32), start, int(core_raw.shape[0])

        pred_prior_wav, pred_start, pred_frames = render(pred_prior[0], pred_duration, pred_center, pred_rms)
        pred_resid_wav, pred_resid_start, pred_resid_frames = render(pred_resid, pred_duration, pred_center, pred_rms)
        zero_wav, zero_start, zero_frames = render(zero_prior[0], zero_duration, zero_center, zero_rms)
        mean_wav, mean_start, mean_frames = render(mean_core_norm, mean_duration, mean_center, mean_rms)
        oracle_norm = item["target_seq"].numpy().astype(np.float32)
        oracle_duration = int(item.get("active_duration_frames", torch.tensor(targets.T)).item())
        oracle_center = int(item.get("active_center_frame", torch.tensor(getattr(targets, "global_core_insert_frame", 0))).item())
        oracle_rms = float(item.get("active_rms", torch.tensor(0.08)).item())
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
            "pred_retrieved_prior": pred_prior_wav,
            "pred_retrieval_residual": pred_resid_wav,
            "zeroeeg_retrieved_prior": zero_wav,
            "mean_train_core": mean_wav,
            "oracle_griffinlim": oracle_wav,
        }
        stem = f"{entry.subject}_trial{entry.trial_index:03d}_{entry.stage}_{entry.label}"
        for wav_type, audio in wavs.items():
            file_name = f"{stem}_{wav_type}.wav"
            save_wav(out_dir / file_name, audio, sr)
            manifest.append([entry.subject, entry.label, entry.stage, entry.trial_index, args.split, wav_type, file_name, _rms(audio)])
        pred_metrics = _active_metrics(pred_resid_wav, original, hop=hop)
        best_corr, best_sec = _best_shift_env_corr(pred_resid_wav, original, hop=hop)
        metric_rows.append(
            {
                "env_corr": _env_corr(pred_resid_wav, original, hop=hop),
                "best_shift_full_env_corr": best_corr,
                "best_shift_sec": best_sec,
                "active_segment_shape_corr": _active_segment_shape_corr(pred_resid_wav, original, hop=hop),
                "active_env_corr": pred_metrics["active_env_corr"],
                "voiced_rms_over_orig": pred_metrics["voiced_rms_over_orig"],
                "peak_over_orig": pred_metrics["peak_over_orig"],
                "active_duration_ratio": pred_metrics["active_duration_ratio"],
                "silence_leakage": _silence_leakage(pred_resid_wav, pred_resid_start, pred_resid_frames, hop=hop),
            }
        )
    with (out_dir / "listening_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject", "label", "stage", "trial_index", "split", "wav_type", "file", "rms"])
        writer.writerows(manifest)
    metrics = {
        "n": int(n_out),
        "target_kind": "speech_core_mel",
        "prediction_mode": "retrieval_first_v61",
        "retrieval_topk": int(topk),
        "lambda_mel_residual": float(residual_scale),
    }
    if metric_rows:
        for key in metric_rows[0]:
            metrics[f"pred_retrieval_residual_{key}_mean"] = float(np.mean([row[key] for row in metric_rows]))
        metrics["active_segment_shape_corr_mean"] = metrics["pred_retrieval_residual_active_segment_shape_corr_mean"]
        metrics["best_shift_full_env_corr_mean"] = metrics["pred_retrieval_residual_best_shift_full_env_corr_mean"]
        metrics["pred_active_local_scaled_active_duration_ratio_mean"] = metrics["pred_retrieval_residual_active_duration_ratio_mean"]
        metrics["pred_active_local_scaled_voiced_rms_over_orig_mean"] = metrics["pred_retrieval_residual_voiced_rms_over_orig_mean"]
        metrics["pred_active_local_scaled_peak_over_orig_mean"] = metrics["pred_retrieval_residual_peak_over_orig_mean"]
        metrics["silence_leakage_wav_mean"] = metrics["pred_retrieval_residual_silence_leakage_mean"]
    (out_dir / "synth_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wav_dir": str(out_dir), "n": n_out, "synth_metrics": str(out_dir / "synth_metrics.json")}, indent=2))


if __name__ == "__main__":
    main()
