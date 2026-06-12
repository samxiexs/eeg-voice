"""Stage-0 codec QC + collapse diagnostics for the factored model (V2_PLAN).

Generates, per sampled (subject,label) cell, FIVE comparable wavs:
  original_ref   : the source recording (rms-normalised, as fed to extraction)
  target_oracle  : decode the TRUE target latent (+ cell decoder_scales)  = upper bound
  mean_latent    : decode the GLOBAL MEAN latent                          = collapse floor
  pred_unscaled  : decode the model's predicted latent
  pred_scaled    : pred decode rescaled to the MODEL-predicted RMS (no target leak)

and quantifies:
  - codec health : oracle vs original (RMS ratio, log-STFT distance) and
                   oracle clearly beating mean-latent  -> is the codec path itself OK?
  - collapse     : pred/target std ratio, pred vs ref pairwise corr, decoded RMS ratio
  - content gain : reuses the honest eval (top1 - zeroeeg)

Outputs (timestamped dir): audio_qc.json, collapse_diagnostics.json,
recon_eval.json, recon_pairs.csv, listening_manifest.csv, and a few saved wavs.

  python scripts/factored_recon_eval.py --config configs/factored.yaml \
      --checkpoint <run>/checkpoints/best.pt --split test_holdout \
      --qc-cells 24 --save-wav 12
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import stft

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.utils import (ensure_dir, load_simple_yaml, load_wav_fixed, resolve_bundle_path,
                       resolve_feis_root, save_wav, write_json)
from src.feis_factored.data import FactoredFEISDataset
from src.feis_factored.eval import evaluate
from src.feis_factored.model import FactoredConfig, FactoredEEG2Speech
from src.feis_factored.synth import denormalize_latent
from src.feis_factored.targets import FactoredTargets


def parse_args():
    p = argparse.ArgumentParser(description="Stage-0 codec QC + collapse diagnostics.")
    p.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "factored.yaml"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test_holdout")
    p.add_argument("--qc-cells", type=int, default=24, help="# unique cells to decode 5-ways")
    p.add_argument("--save-wav", type=int, default=12, help="# cells to actually write wavs for")
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)) + 1e-12)


def _logstft(x: np.ndarray) -> np.ndarray:
    _, _, z = stft(x, nperseg=512, noverlap=384, boundary=None, padded=False)
    return np.log(np.abs(z).astype(np.float32) + 1e-5)


def _stft_dist(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 512:
        return float("nan")
    A, B = _logstft(a[:n]), _logstft(b[:n])
    m = min(A.shape[1], B.shape[1])
    return float(np.mean(np.abs(A[:, :m] - B[:, :m])))


def _scale_to_rms(wav: np.ndarray, target_rms: float, max_gain: float = 20.0) -> np.ndarray:
    g = min(float(target_rms) / _rms(wav), max_gain)
    return (wav * g).astype(np.float32)


def main():
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    targets = FactoredTargets(resolve_bundle_path(cfg["data"]["target_cache"], BUNDLE_DIR))
    feis_root = resolve_feis_root(resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR))
    ds = FactoredFEISDataset(
        data_root=resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR), targets=targets,
        split=args.split, stages=tuple(ckpt["stages"]),
        include_anomalous=bool(cfg["data"].get("include_anomalous", False)),
        holdout_offset=int(ckpt.get("holdout_offset", cfg["data"].get("holdout_offset", 0))),
        holdout_random=bool(ckpt.get("holdout_random", cfg["data"].get("holdout_random", False))))

    model = FactoredEEG2Speech(FactoredConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True); model.eval()
    mean = np.asarray(ckpt["target_mean"], np.float32); std = np.asarray(ckpt["target_std"], np.float32)
    default_scales = np.asarray(ckpt.get("default_decoder_scales", targets.default_decoder_scales), np.float32)

    out_dir = ensure_dir(args.out_dir or (Path(args.checkpoint).resolve().parents[1] /
                                          f"recon_eval_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}"))
    wav_dir = ensure_dir(out_dir / "wav")

    # ---- latent-space honest eval (no codec needed) ----
    bs = int(cfg["train"].get("batch_size", 64))
    metrics = evaluate(model, ds, targets, device=device, batch_size=bs)

    # ---- codec backend ----
    from src.feis_factored.synth import build_codec_backend
    backend = build_codec_backend(
        str(resolve_bundle_path("../models/encodec_24khz", BUNDLE_DIR)), duration_sec=1.0)
    sr = backend.sample_rate
    mean_wav = backend.decode(targets.global_mean_raw_seq().astype(np.float32),
                              decoder_scales=default_scales)

    # ---- pick unique cells from the split ----
    seen_cells: list[tuple[str, str, int]] = []
    seen_keys = set()
    for i, e in enumerate(ds.entries):
        if (e.subject, e.label) in seen_keys:
            continue
        seen_keys.add((e.subject, e.label)); seen_cells.append((e.subject, e.label, i))
        if len(seen_cells) >= args.qc_cells:
            break

    rows = []
    pairs = []
    n_save = 0
    for (sub, lab, idx) in seen_cells:
        item = ds[idx]
        with torch.no_grad():
            pred_latent, pred_log_rms = model.generate_full(
                item["eeg"].unsqueeze(0).to(device), item["subject_idx"].view(1).to(device),
                item["stage_idx"].view(1).to(device))
        pred_norm = pred_latent.squeeze(0).cpu().numpy()
        pred_rms = float(np.exp(float(pred_log_rms.item())))

        # original recording (rms-normalised at the codec sample rate)
        orig = load_wav_fixed(feis_root / targets.cell_audio_path(sub, lab),
                              sample_rate=sr, n_samples=int(sr * 1.0),
                              normalize="rms", target_rms=0.08)
        oracle = backend.decode(targets.cell_raw_target(sub, lab).astype(np.float32),
                                decoder_scales=targets.cell_decoder_scale(sub, lab))
        pred_unscaled = backend.decode(denormalize_latent(pred_norm, mean, std),
                                       decoder_scales=default_scales)
        pred_scaled = _scale_to_rms(pred_unscaled, pred_rms)

        r = {
            "subject": sub, "label": lab, "stage": item["stage"], "split": args.split,
            "orig_rms": _rms(orig), "oracle_rms": _rms(oracle), "mean_rms": _rms(mean_wav),
            "pred_unscaled_rms": _rms(pred_unscaled), "pred_scaled_rms": _rms(pred_scaled),
            "pred_target_rms": pred_rms, "cell_target_rms": targets.cell_rms(sub, lab),
            "oracle_to_orig_stft": _stft_dist(oracle, orig),
            "mean_to_orig_stft": _stft_dist(mean_wav, orig),
            "pred_to_orig_stft": _stft_dist(pred_scaled, orig),
            "pred_to_oracle_stft": _stft_dist(pred_scaled, oracle),
        }
        rows.append(r)
        pairs.append({**r})
        if n_save < args.save_wav:
            tag = f"{sub}_{lab}_{item['stage']}"
            save_wav(wav_dir / f"{tag}_original_ref.wav", orig, sr)
            save_wav(wav_dir / f"{tag}_target_oracle.wav", oracle, sr)
            save_wav(wav_dir / f"{tag}_mean_latent.wav", mean_wav, sr)
            save_wav(wav_dir / f"{tag}_pred_unscaled.wav", pred_unscaled, sr)
            save_wav(wav_dir / f"{tag}_pred_scaled.wav", pred_scaled, sr)
            n_save += 1

    arr = {k: np.asarray([row[k] for row in rows], dtype=np.float64)
           for k in rows[0] if isinstance(rows[0][k], (int, float))}

    def med(k):
        v = arr[k][np.isfinite(arr[k])]
        return float(np.median(v)) if v.size else float("nan")

    oracle_rms_ratio = float(np.median(arr["oracle_rms"] / np.maximum(arr["orig_rms"], 1e-9)))
    pred_rms_ratio = float(np.median(arr["pred_scaled_rms"] / np.maximum(arr["orig_rms"], 1e-9)))
    codec_ok = (0.5 <= oracle_rms_ratio <= 2.0) and (med("oracle_to_orig_stft") < med("mean_to_orig_stft"))

    audio_qc = {
        "split": args.split, "n_cells": len(rows), "sample_rate": sr,
        "oracle_rms_ratio_median": oracle_rms_ratio,
        "pred_scaled_rms_ratio_median": pred_rms_ratio,
        "oracle_to_orig_stft_median": med("oracle_to_orig_stft"),
        "mean_to_orig_stft_median": med("mean_to_orig_stft"),
        "pred_to_orig_stft_median": med("pred_to_orig_stft"),
        "pred_to_oracle_stft_median": med("pred_to_oracle_stft"),
        "codec_path_healthy": bool(codec_ok),
        "verdict": ("CODEC OK — oracle decode faithful & beats mean-latent; collapse/content is the model's"
                    if codec_ok else
                    "CODEC SUSPECT — oracle decode not clearly better than mean-latent or wrong loudness; fix extraction/scales FIRST"),
    }
    collapse = {
        "pred_std_ratio_median": metrics["pred_std_ratio_median"],
        "pred_pairwise_corr_median": metrics["pred_pairwise_corr_median"],
        "target_pairwise_corr_median": metrics["target_pairwise_corr_median"],
        "pred_scaled_rms_ratio_median": pred_rms_ratio,
        "collapsed": bool(metrics["pred_pairwise_corr_median"] > 0.25
                          or (metrics["pred_std_ratio_median"] < 0.5)),
    }

    write_json(out_dir / "audio_qc.json", audio_qc)
    write_json(out_dir / "collapse_diagnostics.json", collapse)
    write_json(out_dir / "recon_eval.json", {"latent_metrics": metrics, "audio_qc": audio_qc,
                                             "collapse": collapse})
    with (out_dir / "recon_pairs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with (out_dir / "listening_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "label", "stage", "wav_type", "rel_path", "rms"])
        for row in rows[:args.save_wav]:
            tag = f"{row['subject']}_{row['label']}_{row['stage']}"
            for kind, rms_key in [("original_ref", "orig_rms"), ("target_oracle", "oracle_rms"),
                                  ("mean_latent", "mean_rms"), ("pred_unscaled", "pred_unscaled_rms"),
                                  ("pred_scaled", "pred_scaled_rms")]:
                w.writerow([row["subject"], row["label"], row["stage"], kind,
                            f"wav/{tag}_{kind}.wav", row[rms_key]])

    print(f"[codec QC] {audio_qc['verdict']}")
    print(f"  oracle/orig RMS={oracle_rms_ratio:.3f} | oracle_stft={audio_qc['oracle_to_orig_stft_median']:.3f}"
          f" < mean_stft={audio_qc['mean_to_orig_stft_median']:.3f} ? {codec_ok}")
    print(f"[collapse] std_ratio={collapse['pred_std_ratio_median']:.3f} "
          f"pred_corr={collapse['pred_pairwise_corr_median']:.3f} "
          f"(ref_corr={collapse['target_pairwise_corr_median']:.3f}) collapsed={collapse['collapsed']}")
    print(f"[content] top1={metrics['within_subject_content_top1']:.4f} "
          f"zeroeeg={metrics['within_subject_content_top1_zeroeeg']:.4f} "
          f"gain={metrics['content_gain']:+.4f}")
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
