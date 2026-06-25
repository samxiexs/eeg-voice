"""Synthesize wavs from a FEIS EEG-only mel-alignment checkpoint."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.feis_mel.audio import MelConfig, logmel_to_wav, rms, scale_to_rms
from src.feis_mel.data import FEISMelDataset
from src.feis_mel.diffusion import FEISAcousticDiffusionConfig, FEISDiffusionInference, build_feis_acoustic_diffusion
from src.feis_mel.model import FEISEEGToMel, FEISMelConfig
from src.feis_mel.targets import MelLabelTargets
from src.direct_eeg2speech.synth import build_codec_backend
from src.utils import ensure_dir, load_simple_yaml, load_wav_fixed, resolve_bundle_path, resolve_feis_root, save_wav


def parse_args():
    p = argparse.ArgumentParser(description="Synthesize FEIS EEG-only mel wavs.")
    p.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "feis_mel_align.yaml"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test_holdout")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--diverse-labels", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--sample-steps", type=int, default=None)
    return p.parse_args()


def _target_cache_path(cfg: dict, ckpt: dict) -> Path:
    target_cache = Path(str(ckpt.get("target_cache", "")))
    if target_cache.exists():
        return target_cache
    target_kind = str(ckpt.get("target_kind", cfg.get("target", {}).get("kind", "mel")))
    target_cfg = cfg.get("target", {})
    if target_kind == "encodec_latent":
        return resolve_bundle_path(target_cfg.get("cache_encodec", cfg["data"]["target_cache"]), BUNDLE_DIR)
    return resolve_bundle_path(target_cfg.get("cache_mel", cfg["data"]["target_cache"]), BUNDLE_DIR)


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    target_cache = _target_cache_path(cfg, ckpt)
    targets = MelLabelTargets(target_cache)
    target_kind = str(ckpt.get("target_kind", targets.target_kind))
    feis_root = resolve_feis_root(resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR))
    ds = FEISMelDataset(
        data_root=resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR),
        targets=targets,
        split=args.split,
        stage=str(ckpt["stage"]),
        include_anomalous=bool(cfg["data"].get("include_anomalous", False)),
    )
    model = FEISEEGToMel(FEISMelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    synth_model = model
    if str(ckpt.get("decoder_kind", "regression")) == "diffusion":
        diff_cfg = FEISAcousticDiffusionConfig(**ckpt["diffusion_config"])
        diffusion = build_feis_acoustic_diffusion(diff_cfg).to(device)
        diffusion.load_state_dict(ckpt["diffusion_state"], strict=True)
        synth_model = FEISDiffusionInference(
            model,
            diffusion,
            target_steps=targets.T,
            target_dim=targets.D,
            sample_steps=int(args.sample_steps or diff_cfg.sample_steps),
        )
    synth_model.eval()
    mel_cfg = None
    codec_backend = None
    if target_kind == "mel":
        mel_cfg = MelConfig(
            sample_rate=int(cfg["audio"].get("sample_rate", 16000)),
            n_mels=int(cfg["target"].get("n_mels", 80)),
            n_fft=int(cfg["target"].get("n_fft", 1024)),
            hop_length=int(cfg["target"].get("hop_length", 256)),
            target_frames=targets.T,
            f_min=float(cfg["target"].get("f_min", 50.0)),
            f_max=float(cfg["target"].get("f_max", 7600.0)),
        )
        sample_rate = mel_cfg.sample_rate
    elif target_kind == "encodec_latent":
        codec_backend = build_codec_backend(
            resolve_bundle_path(cfg.get("vocoder", {}).get("encodec_model_name_or_path", "../models/encodec_24khz"), BUNDLE_DIR),
            local_files_only=bool(cfg.get("vocoder", {}).get("encodec_local_files_only", True)),
        )
        sample_rate = int(codec_backend.sample_rate)
    else:
        raise ValueError(f"Unsupported checkpoint target_kind={target_kind!r}")
    out_dir = ensure_dir(args.out_dir or (
        Path(args.checkpoint).resolve().parents[1] / f"final_wavs_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}"
    ))
    if args.diverse_labels:
        indices = []
        seen = set()
        for idx, item_entry in enumerate(ds.entries):
            lab = item_entry.label
            if lab in seen:
                continue
            seen.add(lab)
            indices.append(idx)
            if len(indices) >= args.limit:
                break
    else:
        indices = list(range(min(args.limit, len(ds))))
    if target_kind == "mel":
        assert mel_cfg is not None
        mean_wav = logmel_to_wav(targets.global_mean_raw, mel_cfg, iters=int(cfg["vocoder"].get("iters", 48)))
    else:
        assert codec_backend is not None
        mean_wav = codec_backend.decode(targets.global_mean_raw, decoder_scales=targets.default_decoder_scales)
    manifest = []
    for idx in indices:
        item = ds[idx]
        lab = str(item["label"])
        label_idx = int(item["label_idx"].item())
        with torch.no_grad():
            out = synth_model(item["eeg"].unsqueeze(0).to(device))
        pred_norm = out["pred_mel"].squeeze(0).cpu().numpy()
        pred_raw = targets.denormalize(pred_norm)
        pred_rms = float(np.exp(float(out["pred_log_rms"].item())))
        if target_kind == "mel":
            assert mel_cfg is not None
            pred_wav = logmel_to_wav(pred_raw, mel_cfg, iters=int(cfg["vocoder"].get("iters", 48)), seed=idx + 1)
            oracle = logmel_to_wav(targets.raw_bank_for_label_id(label_idx)[0], mel_cfg, iters=int(cfg["vocoder"].get("iters", 48)), seed=idx + 100)
        else:
            assert codec_backend is not None
            pred_wav = codec_backend.decode(pred_raw, decoder_scales=targets.decoder_scale_for_label_id(label_idx, 0))
            oracle = codec_backend.decode(targets.raw_bank_for_label_id(label_idx)[0], decoder_scales=targets.decoder_scale_for_label_id(label_idx, 0))
        pred_scaled = scale_to_rms(pred_wav, pred_rms)
        ref_path = targets.canonical_path_for_label_id(label_idx, 0)
        original = load_wav_fixed(
            feis_root / ref_path,
            sample_rate=sample_rate,
            n_samples=int(round(sample_rate * float(cfg["audio"].get("duration_sec", 1.0)))),
            normalize=str(cfg["audio"].get("normalize", "rms")),
            target_rms=float(cfg["audio"].get("target_rms", 0.08)),
            max_gain=float(cfg["audio"].get("max_gain", 12.0)),
        )
        tag = f"{item['sample_key']}_{lab}_{ckpt['stage']}"
        for kind, wav in {
            "original_ref": original,
            "target_oracle": oracle,
            "mean_mel": mean_wav,
            "pred_unscaled": pred_wav,
            "pred_scaled": pred_scaled,
        }.items():
            file_name = f"{tag}_{kind}.wav"
            save_wav(out_dir / file_name, wav, sample_rate)
            manifest.append([item["sample_key"], lab, ckpt["stage"], args.split, kind, file_name, rms(wav)])
    with (out_dir / "listening_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample_key", "label", "stage", "split", "wav_type", "file", "rms"])
        writer.writerows(manifest)
    print(f"[done] wrote {len(indices)} EEG-only {target_kind} samples x 5 wavs to {out_dir}")


if __name__ == "__main__":
    main()
