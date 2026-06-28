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
from src.karaone_recon.diffusion import DiffusionConfig, EEGLatentDiffusion
from src.karaone_recon.rendered_metrics import load_whisper_asr, transcribe_label_metrics
from src.karaone_recon.synth import build_codec_backend, denormalize_latent
from src.karaone_recon.targets import KaraOneTargets
from src.utils import ensure_dir, load_simple_yaml, load_wav_fixed, resolve_bundle_path, resolve_target_cache, save_wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample wavs from a KaraOne latent-diffusion checkpoint.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "subject_test"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--limit", type=int, default=8, help="number of trials; <=0 means ALL trials in the split")
    parser.add_argument("--steps", type=int, default=None, help="DDIM sampling steps (default: ckpt's)")
    parser.add_argument("--num-samples", type=int, default=2, help="draws per trial (shows generative diversity)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--asr-model", default=None, help="optional local/cached Whisper model for rendered-audio ASR metrics")
    parser.add_argument("--asr-allow-download", action="store_true", help="allow Whisper to download --asr-model if not cached")
    parser.add_argument("--asr-download-root", default=None, help="optional Whisper model cache/download directory")
    return parser.parse_args()


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)) + 1e-12)


def _scale_to_rms(audio: np.ndarray, target_rms: float, max_gain: float = 20.0) -> np.ndarray:
    return (audio * min(float(target_rms) / _rms(audio), max_gain)).astype(np.float32)


def _envelope(audio: np.ndarray, hop: int = 256) -> np.ndarray:
    n = len(audio) // hop
    if n < 2:
        return np.zeros(2, dtype=np.float64)
    frames = np.asarray(audio[: n * hop], dtype=np.float64).reshape(n, hop)
    return np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)


def _env_corr(a: np.ndarray, b: np.ndarray) -> float:
    ea, eb = _envelope(a), _envelope(b)
    m = min(len(ea), len(eb))
    if m < 2:
        return 0.0
    ea = ea[:m] - ea[:m].mean()
    eb = eb[:m] - eb[:m].mean()
    return float((ea * eb).sum() / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))


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
    sample_mask = _samples_from_frame_mask(orig_active, min(len(candidate), len(original)), hop=hop)
    if not sample_mask.any():
        sample_mask = np.ones_like(sample_mask, dtype=bool)
    cand = np.asarray(candidate[: sample_mask.size], dtype=np.float64)[sample_mask]
    orig = np.asarray(original[: sample_mask.size], dtype=np.float64)[sample_mask]
    active_corr = 0.0
    if orig_active.sum() >= 2:
        ca = cand_env[orig_active] - cand_env[orig_active].mean()
        oa = orig_env[orig_active] - orig_env[orig_active].mean()
        active_corr = float((ca * oa).sum() / (np.linalg.norm(ca) * np.linalg.norm(oa) + 1e-8))
    return {
        "active_env_corr": active_corr,
        "voiced_rms_over_orig": _rms(cand) / max(_rms(orig), 1e-8),
        "peak_over_orig": float(np.max(np.abs(cand)) / max(float(np.max(np.abs(orig))), 1e-8)),
        "active_duration_ratio": float(cand_active.mean() / max(float(orig_active.mean()), 1e-8)),
    }


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    asr_model, asr_status = load_whisper_asr(args.asr_model, device, args.asr_allow_download, args.asr_download_root)
    print(f"[asr] {asr_status}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = EEGLatentDiffusion(DiffusionConfig(**ckpt["diffusion_config"])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    steps = int(args.steps or ckpt.get("ddim_steps", 50))

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
    print(f"[synth] target={target_kind} vocoder={'griffinlim' if target_kind=='mel' else 'encodec'} sr={sample_rate}")
    target_rms = float(cfg["audio"].get("target_rms", 0.08))
    out_dir = ensure_dir(
        args.out_dir or (Path(args.checkpoint).resolve().parents[1] / f"wav_diff_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}")
    )
    mean_wav = backend.decode(targets.global_mean_raw.astype(np.float32), decoder_scales=targets.default_decoder_scales)
    n_out = len(ds) if int(args.limit) <= 0 else min(int(args.limit), len(ds))
    print(f"[synth] reconstructing {n_out}/{len(ds)} trials of split={args.split} (ddim_steps={steps})")
    manifest = []
    metric_rows = []
    for idx in range(n_out):
        item = ds[idx]
        entry = ds.entries[idx]
        eeg = item["eeg"].unsqueeze(0).to(device)
        valid = item["eeg_valid_len"].view(1).to(device)
        tag = f"{entry.subject}_{entry.label.replace('/', '')}_{entry.stage}_t{entry.trial_index:03d}"

        oracle_kind = "oracle_encodec" if target_kind == "encodec_latent" else "oracle_griffinlim"
        original = load_wav_fixed(
            root / targets.audio_path(entry.subject, entry.trial_index),
            sample_rate=sample_rate,
            n_samples=int(round(sample_rate * duration_sec)),
            normalize=str(cfg["audio"].get("normalize", "rms")),
            target_rms=target_rms,
            max_gain=float(cfg["audio"].get("max_gain", 10.0)),
        )
        wavs = {
            "original": original,
            oracle_kind: backend.decode(
                targets.raw_target(entry.subject, entry.trial_index).astype(np.float32),
                decoder_scales=targets.decoder_scale(entry.subject, entry.trial_index),
            ),
            "mean_latent": mean_wav,
        }
        with torch.no_grad():
            zero_latent = model.sample(torch.zeros_like(eeg), valid, steps=steps).squeeze(0).cpu().numpy()
        zero_wav = backend.decode(
            denormalize_latent(zero_latent, targets.target_mean, targets.target_std),
            decoder_scales=targets.default_decoder_scales,
        )
        wavs["zeroeeg"] = _scale_to_rms(zero_wav, target_rms)
        # Multiple diffusion draws -> demonstrates the output is NOT a fixed mean.
        for s in range(int(args.num_samples)):
            with torch.no_grad():
                latent = model.sample(eeg, valid, steps=steps).squeeze(0).cpu().numpy()
            wav = backend.decode(
                denormalize_latent(latent, targets.target_mean, targets.target_std),
                decoder_scales=targets.default_decoder_scales,
            )
            wavs[f"sample{s + 1}"] = _scale_to_rms(wav, target_rms)

        for kind, wav in wavs.items():
            filename = f"{tag}_{kind}.wav"
            save_wav(out_dir / filename, wav, sample_rate)
            manifest.append([entry.subject, entry.label, entry.stage, entry.trial_index, args.split, kind, filename, _rms(wav)])

        sample1 = wavs.get("sample1")
        if sample1 is not None:
            sample_active = _active_metrics(sample1, original)
            oracle_active = _active_metrics(wavs[oracle_kind], original)
            metric_row = {
                "subject": entry.subject,
                "label": entry.label,
                "trial_index": int(entry.trial_index),
                "sample1_env_corr": _env_corr(sample1, original),
                "oracle_env_corr": _env_corr(wavs[oracle_kind], original),
                "sample1_rms_over_orig": _rms(sample1) / max(_rms(original), 1e-8),
                "oracle_rms_over_orig": _rms(wavs[oracle_kind]) / max(_rms(original), 1e-8),
                "sample1_active_env_corr": sample_active["active_env_corr"],
                "oracle_active_env_corr": oracle_active["active_env_corr"],
                "sample1_voiced_rms_over_orig": sample_active["voiced_rms_over_orig"],
                "oracle_voiced_rms_over_orig": oracle_active["voiced_rms_over_orig"],
                "sample1_peak_over_orig": sample_active["peak_over_orig"],
                "oracle_peak_over_orig": oracle_active["peak_over_orig"],
                "sample1_active_duration_ratio": sample_active["active_duration_ratio"],
                "oracle_active_duration_ratio": oracle_active["active_duration_ratio"],
            }
            if asr_model is not None:
                for wav_kind, prefix in (
                    ("sample1", "sample1_asr"),
                    (oracle_kind, "oracle_asr"),
                    ("zeroeeg", "zeroeeg_asr"),
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
        "asr": asr_status,
        "oracle_kind": "oracle_encodec" if target_kind == "encodec_latent" else "oracle_griffinlim",
        "sample1_env_corr_mean": _mean("sample1_env_corr"),
        "oracle_env_corr_mean": _mean("oracle_env_corr"),
        "sample1_active_env_corr_mean": _mean("sample1_active_env_corr"),
        "oracle_active_env_corr_mean": _mean("oracle_active_env_corr"),
        "sample1_voiced_rms_over_orig_mean": _mean("sample1_voiced_rms_over_orig"),
        "oracle_voiced_rms_over_orig_mean": _mean("oracle_voiced_rms_over_orig"),
        "sample1_peak_over_orig_mean": _mean("sample1_peak_over_orig"),
        "oracle_peak_over_orig_mean": _mean("oracle_peak_over_orig"),
        "sample1_active_duration_ratio_mean": _mean("sample1_active_duration_ratio"),
        "oracle_active_duration_ratio_mean": _mean("oracle_active_duration_ratio"),
        "sample1_asr_label_acc": _hit_rate("sample1_asr_label_hit"),
        "oracle_asr_label_acc": _hit_rate("oracle_asr_label_hit"),
        "zeroeeg_asr_label_acc": _hit_rate("zeroeeg_asr_label_hit"),
        "mean_asr_label_acc": _hit_rate("mean_asr_label_hit"),
        "original_asr_label_acc": _hit_rate("original_asr_label_hit"),
        "sample1_asr_cer_mean": _mean_optional("sample1_asr_cer"),
        "oracle_asr_cer_mean": _mean_optional("oracle_asr_cer"),
        "zeroeeg_asr_cer_mean": _mean_optional("zeroeeg_asr_cer"),
        "mean_asr_cer_mean": _mean_optional("mean_asr_cer"),
        "original_asr_cer_mean": _mean_optional("original_asr_cer"),
        "per_trial": metric_rows,
    }
    (out_dir / "synth_metrics.json").write_text(json.dumps(synth_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[oracle] env_corr sample1={synth_metrics['sample1_env_corr_mean']:.3f} "
        f"oracle={synth_metrics['oracle_env_corr_mean']:.3f} | "
        f"active_env sample1={synth_metrics['sample1_active_env_corr_mean']:.3f} "
        f"oracle={synth_metrics['oracle_active_env_corr_mean']:.3f}"
    )
    print(f"[done] wrote {len(manifest)} wav rows to {out_dir} (ddim_steps={steps})")


if __name__ == "__main__":
    main()
