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


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = KaraOneEEG2Codec(KaraOneConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
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
            pred_latent, pred_log_rms = model.generate_full(
                item["eeg"].unsqueeze(0).to(device),
                item["subject_idx"].view(1).to(device),
                item["stage_idx"].view(1).to(device),
                valid_len,
            )
            zero_latent, zero_log_rms = model.generate_full(
                torch.zeros_like(item["eeg"]).unsqueeze(0).to(device),
                item["subject_idx"].view(1).to(device),
                item["stage_idx"].view(1).to(device),
                valid_len,
            )
        pred = pred_latent.squeeze(0).cpu().numpy()
        zero = zero_latent.squeeze(0).cpu().numpy()
        pred_wav = backend.decode(
            denormalize_latent(pred, targets.target_mean, targets.target_std),
            decoder_scales=targets.default_decoder_scales,
        )
        zero_wav = backend.decode(
            denormalize_latent(zero, targets.target_mean, targets.target_std),
            decoder_scales=targets.default_decoder_scales,
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
        wavs = {
            "original": original,
            "oracle_codec": oracle,
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
        metric_rows.append(
            {
                "subject": entry.subject,
                "label": entry.label,
                "trial_index": int(entry.trial_index),
                "pred_env_corr": _env_corr(wavs["pred_scaled"], original),
                "oracle_env_corr": _env_corr(oracle, original),
                "pred_rms_over_orig": _rms(wavs["pred_scaled"]) / max(_rms(original), 1e-8),
                "oracle_rms_over_orig": _rms(oracle) / max(_rms(original), 1e-8),
            }
        )
    with (out_dir / "listening_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject", "label", "stage", "trial_index", "split", "wav_type", "file", "rms"])
        writer.writerows(manifest)

    def _mean(key: str) -> float:
        return float(np.mean([row[key] for row in metric_rows])) if metric_rows else 0.0

    synth_metrics = {
        "split": args.split,
        "n": len(metric_rows),
        "target_kind": target_kind,
        "vocoder": "griffinlim" if target_kind == "mel" else "encodec",
        "pred_env_corr_mean": _mean("pred_env_corr"),
        "oracle_env_corr_mean": _mean("oracle_env_corr"),  # vocoder ceiling
        "pred_rms_over_orig_mean": _mean("pred_rms_over_orig"),
        "oracle_rms_over_orig_mean": _mean("oracle_rms_over_orig"),
        "per_trial": metric_rows,
    }
    (out_dir / "synth_metrics.json").write_text(json.dumps(synth_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[oracle] env_corr pred={synth_metrics['pred_env_corr_mean']:.3f} "
        f"oracle(ceiling)={synth_metrics['oracle_env_corr_mean']:.3f} | "
        f"rms/orig pred={synth_metrics['pred_rms_over_orig_mean']:.3f} oracle={synth_metrics['oracle_rms_over_orig_mean']:.3f}"
    )
    print(f"[done] wrote {len(manifest)} wav rows to {out_dir}")


if __name__ == "__main__":
    main()
