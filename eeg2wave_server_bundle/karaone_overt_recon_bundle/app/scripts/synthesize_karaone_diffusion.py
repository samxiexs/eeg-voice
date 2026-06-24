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

from src.audio_features import AudioFeatureConfig, load_mel_vocoder
from src.karaone_recon.data import KaraOneTrialDataset
from src.karaone_recon.diffusion import DiffusionConfig, EEGLatentDiffusion
from src.karaone_recon.synth import build_codec_backend, denormalize_latent
from src.karaone_recon.targets import KaraOneTargets
from src.utils import ensure_dir, load_simple_yaml, load_wav_fixed, resolve_bundle_path, resolve_target_cache, save_wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample wavs from a KaraOne latent-diffusion checkpoint.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "subject_test"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--steps", type=int, default=None, help="DDIM sampling steps (default: ckpt's)")
    parser.add_argument("--num-samples", type=int, default=2, help="draws per trial (shows generative diversity)")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)) + 1e-12)


def _scale_to_rms(audio: np.ndarray, target_rms: float, max_gain: float = 20.0) -> np.ndarray:
    return (audio * min(float(target_rms) / _rms(audio), max_gain)).astype(np.float32)


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
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
                griffinlim_iters=int(cfg.get("vocoder", {}).get("griffinlim_iters", 60)),
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
    manifest = []
    for idx in range(min(int(args.limit), len(ds))):
        item = ds[idx]
        entry = ds.entries[idx]
        eeg = item["eeg"].unsqueeze(0).to(device)
        valid = item["eeg_valid_len"].view(1).to(device)
        tag = f"{entry.subject}_{entry.label.replace('/', '')}_{entry.stage}_t{entry.trial_index:03d}"

        wavs = {
            "original": load_wav_fixed(
                root / targets.audio_path(entry.subject, entry.trial_index),
                sample_rate=sample_rate,
                n_samples=int(round(sample_rate * duration_sec)),
                normalize=str(cfg["audio"].get("normalize", "rms")),
                target_rms=target_rms,
                max_gain=float(cfg["audio"].get("max_gain", 10.0)),
            ),
            "oracle_codec": backend.decode(
                targets.raw_target(entry.subject, entry.trial_index).astype(np.float32),
                decoder_scales=targets.decoder_scale(entry.subject, entry.trial_index),
            ),
            "mean_latent": mean_wav,
        }
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
    with (out_dir / "listening_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject", "label", "stage", "trial_index", "split", "wav_type", "file", "rms"])
        writer.writerows(manifest)
    print(f"[done] wrote {len(manifest)} wav rows to {out_dir} (ddim_steps={steps})")


if __name__ == "__main__":
    main()
