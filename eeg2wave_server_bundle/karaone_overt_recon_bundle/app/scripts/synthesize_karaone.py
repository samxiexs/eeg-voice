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
from src.karaone_recon.data import KaraOneTrialDataset
from src.karaone_recon.model import KaraOneConfig, KaraOneEEG2Codec
from src.karaone_recon.rendered_metrics import load_whisper_asr, transcribe_label_metrics
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
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    target_kind = str(ckpt.get("target_kind", cfg.get("target", {}).get("kind", "encodec_latent")))
    _, cache = resolve_target_cache(cfg, BUNDLE_DIR, target_kind)
    targets = KaraOneTargets(cache, data_root=root)
    split_protocol = "subject_holdout" if args.split == "subject_test" else str(cfg["data"].get("split_protocol", "trial"))
    ds = KaraOneTrialDataset(
        data_root=root,
        targets=targets,
        split=args.split,
        stages=tuple(ckpt["stages"]),
        split_protocol=split_protocol,
        heldout_subjects=cfg["data"].get("heldout_subjects", ["P02", "MM21"]),
        eeg_len=int(cfg["data"].get("eeg_len", 1280)),
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
        f"mode={'residual_global_mean' if residual_mean else 'direct_target'} sr={sample_rate}"
    )
    out_dir = ensure_dir(
        args.out_dir
        or (Path(args.checkpoint).resolve().parents[1] / f"wav_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}")
    )
    mean_wav = backend.decode(targets.global_mean_raw.astype(np.float32), decoder_scales=targets.default_decoder_scales)
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
        if residual_mean:
            pred = targets.global_mean_norm.astype(np.float32) + pred
            zero = targets.global_mean_norm.astype(np.float32) + zero
        pred_log_rms = pred_out["pred_log_rms"].squeeze(0)
        zero_log_rms = zero_out["pred_log_rms"].squeeze(0)
        pred_decoder_scale = targets.default_decoder_scales
        zero_decoder_scale = targets.default_decoder_scales
        if has_trained_scale_head:
            pred_decoder_scale = np.exp(pred_out["pred_log_decoder_scale"].squeeze(0).cpu().numpy()).astype(np.float32)
            zero_decoder_scale = np.exp(zero_out["pred_log_decoder_scale"].squeeze(0).cpu().numpy()).astype(np.float32)
            pred_decoder_scale = np.clip(pred_decoder_scale, 1e-4, 20.0)
            zero_decoder_scale = np.clip(zero_decoder_scale, 1e-4, 20.0)
        pred_wav = backend.decode(
            denormalize_latent(pred, targets.target_mean, targets.target_std),
            decoder_scales=pred_decoder_scale,
        )
        zero_wav = backend.decode(
            denormalize_latent(zero, targets.target_mean, targets.target_std),
            decoder_scales=zero_decoder_scale,
        )
        oracle = backend.decode(
            targets.raw_target(entry.subject, entry.trial_index).astype(np.float32),
            decoder_scales=targets.decoder_scale(entry.subject, entry.trial_index),
        )
        original = load_wav_fixed(
            root / targets.audio_path(entry.subject, entry.trial_index),
            sample_rate=sample_rate,
            n_samples=int(round(sample_rate * duration_sec)),
            normalize=str(cfg["audio"].get("normalize", "rms")),
            target_rms=float(cfg["audio"].get("target_rms", 0.08)),
            max_gain=float(cfg["audio"].get("max_gain", 10.0)),
        )
        tag = f"{entry.subject}_{entry.label.replace('/', '')}_{entry.stage}_t{entry.trial_index:03d}"
        oracle_kind = "oracle_encodec" if target_kind == "encodec_latent" else "oracle_griffinlim"
        wavs = {
            "original": original,
            oracle_kind: oracle,
            "mean_latent": mean_wav,
            "zeroeeg": zero_wav,
            "pred": pred_wav,
            "pred_scaled": _scale_to_rms(pred_wav, float(np.exp(float(pred_log_rms.item())))),
            "zeroeeg_scaled": _scale_to_rms(zero_wav, float(np.exp(float(zero_log_rms.item())))),
        }
        for kind, wav in wavs.items():
            filename = f"{tag}_{kind}.wav"
            save_wav(out_dir / filename, wav, sample_rate)
            manifest.append([entry.subject, entry.label, entry.stage, entry.trial_index, args.split, kind, filename, _rms(wav)])
        # Oracle = GT target through the SAME vocoder => the vocoder ceiling. Reporting
        # pred vs oracle (not just vs original) separates "model error" from "vocoder loss".
        pred_active = _active_metrics(wavs["pred_scaled"], original)
        oracle_active = _active_metrics(oracle, original)
        metric_row = {
            "subject": entry.subject,
            "label": entry.label,
            "trial_index": int(entry.trial_index),
            "pred_env_corr": _env_corr(wavs["pred_scaled"], original),
            "oracle_env_corr": _env_corr(oracle, original),
            "pred_rms_over_orig": _rms(wavs["pred_scaled"]) / max(_rms(original), 1e-8),
            "oracle_rms_over_orig": _rms(oracle) / max(_rms(original), 1e-8),
            "pred_active_env_corr": pred_active["active_env_corr"],
            "oracle_active_env_corr": oracle_active["active_env_corr"],
            "pred_voiced_rms_over_orig": pred_active["voiced_rms_over_orig"],
            "oracle_voiced_rms_over_orig": oracle_active["voiced_rms_over_orig"],
            "pred_peak_over_orig": pred_active["peak_over_orig"],
            "oracle_peak_over_orig": oracle_active["peak_over_orig"],
            "pred_active_duration_ratio": pred_active["active_duration_ratio"],
            "oracle_active_duration_ratio": oracle_active["active_duration_ratio"],
        }
        if asr_model is not None:
            for wav_kind, prefix in (
                ("pred_scaled", "pred_asr"),
                (oracle_kind, "oracle_asr"),
                ("zeroeeg_scaled", "zeroeeg_asr"),
                ("mean_latent", "mean_asr"),
                ("original", "original_asr"),
            ):
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
        "target_kind": target_kind,
        "vocoder": "griffinlim" if target_kind == "mel" else "encodec",
        "prediction_mode": "residual_global_mean" if residual_mean else "direct_target",
        "asr": asr_status,
        "oracle_kind": oracle_kind if metric_rows else ("oracle_encodec" if target_kind == "encodec_latent" else "oracle_griffinlim"),
        "pred_env_corr_mean": _mean("pred_env_corr"),
        "oracle_env_corr_mean": _mean("oracle_env_corr"),  # vocoder ceiling
        "pred_rms_over_orig_mean": _mean("pred_rms_over_orig"),
        "oracle_rms_over_orig_mean": _mean("oracle_rms_over_orig"),
        "pred_active_env_corr_mean": _mean("pred_active_env_corr"),
        "oracle_active_env_corr_mean": _mean("oracle_active_env_corr"),
        "pred_voiced_rms_over_orig_mean": _mean("pred_voiced_rms_over_orig"),
        "oracle_voiced_rms_over_orig_mean": _mean("oracle_voiced_rms_over_orig"),
        "pred_peak_over_orig_mean": _mean("pred_peak_over_orig"),
        "oracle_peak_over_orig_mean": _mean("oracle_peak_over_orig"),
        "pred_active_duration_ratio_mean": _mean("pred_active_duration_ratio"),
        "oracle_active_duration_ratio_mean": _mean("oracle_active_duration_ratio"),
        "pred_asr_label_acc": _hit_rate("pred_asr_label_hit"),
        "oracle_asr_label_acc": _hit_rate("oracle_asr_label_hit"),
        "zeroeeg_asr_label_acc": _hit_rate("zeroeeg_asr_label_hit"),
        "mean_asr_label_acc": _hit_rate("mean_asr_label_hit"),
        "original_asr_label_acc": _hit_rate("original_asr_label_hit"),
        "pred_asr_cer_mean": _mean_optional("pred_asr_cer"),
        "oracle_asr_cer_mean": _mean_optional("oracle_asr_cer"),
        "zeroeeg_asr_cer_mean": _mean_optional("zeroeeg_asr_cer"),
        "mean_asr_cer_mean": _mean_optional("mean_asr_cer"),
        "original_asr_cer_mean": _mean_optional("original_asr_cer"),
        "per_trial": metric_rows,
    }
    (out_dir / "synth_metrics.json").write_text(json.dumps(synth_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[oracle] env_corr pred={synth_metrics['pred_env_corr_mean']:.3f} "
        f"oracle(ceiling)={synth_metrics['oracle_env_corr_mean']:.3f} | "
        f"active_env pred={synth_metrics['pred_active_env_corr_mean']:.3f} "
        f"oracle={synth_metrics['oracle_active_env_corr_mean']:.3f} | "
        f"voiced_rms/orig pred={synth_metrics['pred_voiced_rms_over_orig_mean']:.3f} "
        f"oracle={synth_metrics['oracle_voiced_rms_over_orig_mean']:.3f}"
    )
    print(f"[done] wrote {len(manifest)} wav rows to {out_dir}")


if __name__ == "__main__":
    main()
