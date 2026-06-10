"""Synthesise natural-sounding wavs from a trained v3 checkpoint.

EEG -> predicted EnCodec latent -> denormalise -> frozen EnCodec decoder -> wav.
Writes one wav per test trial plus the per-label canonical reference for A/B.

  python scripts/v3_synthesize.py --config configs/v3_encodec.yaml \
      --checkpoint <run>/checkpoints/best.pt --protocol G --stage thinking \
      --out-dir <run>/recon_wavs --limit 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path
from src.v3.data import V3Dataset
from src.v3.model import EEG2SpeechV3, EEG2SpeechV3Config
from src.v3.synth import build_codec_backend, latent_to_wav


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthesise wavs from a v3 checkpoint.")
    p.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "v3_encodec.yaml"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--protocol", default=None)
    p.add_argument("--stage", default=None)
    p.add_argument("--subject", default=None)
    p.add_argument("--holdout-subject", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_simple_yaml(args.config)
    cfg_d, cfg_a, cfg_t = config["data"], config["audio"], config["targets"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = EEG2SpeechV3(EEG2SpeechV3Config(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    target_mean = np.asarray(ckpt["target_mean"], dtype=np.float32)
    target_std = np.asarray(ckpt["target_std"], dtype=np.float32)

    root = resolve_bundle_path(args.data_root or cfg_d["root"], BUNDLE_DIR)
    cache = resolve_bundle_path(cfg_t["cache_path"], BUNDLE_DIR)
    dataset = V3Dataset(
        data_root=str(root),
        protocol=(args.protocol or cfg_d["protocol"]).upper(),
        stage=args.stage or cfg_d.get("stage", "thinking"),
        split=args.split,
        subject_id=args.subject or cfg_d.get("subject_id"),
        holdout_subject_id=args.holdout_subject or cfg_d.get("holdout_subject_id"),
        include_anomalous=bool(cfg_d.get("include_anomalous", False)),
        target_cache_path=str(cache),
        require_targets=True,
        audio_sr=int(cfg_a["sample_rate"]),
        audio_dur=float(cfg_a["duration_sec"]),
    )

    backend = build_codec_backend(
        codec_model_path=str(resolve_bundle_path(cfg_t["codec_model_name_or_path"], BUNDLE_DIR)),
        duration_sec=float(cfg_a["duration_sec"]),
        bandwidth=float(cfg_t.get("codec_bandwidth", 6.0)),
        local_files_only=bool(cfg_t.get("local_files_only", True)),
    )
    sr = backend.sample_rate

    out_dir = ensure_dir(args.out_dir)
    n = min(args.limit, len(dataset))
    for i in range(n):
        item = dataset[i]
        eeg = item["eeg"].unsqueeze(0).to(device)
        subj = item["subject_index"].view(1).to(device)
        with torch.no_grad():
            pred = model(eeg, subj)["speech_sequence"].squeeze(0).cpu().numpy()  # [T, D] normalised
        wav = latent_to_wav(backend, pred, target_mean, target_std)
        label = item["label"]
        subject_id = item["subject_id"]
        trial = int(item["trial_index"])
        sf.write(out_dir / f"{subject_id}_{label}_{trial}_pred.wav", wav, sr)
        # Reference: decode the canonical target latent (upper bound for this pipeline).
        tgt = dataset.get_template_target(item["template_id"])
        raw_tgt = tgt["raw_target_sequence"]
        ref = backend.decode(np.asarray(raw_tgt, dtype=np.float32))
        sf.write(out_dir / f"{subject_id}_{label}_{trial}_ref.wav", ref, sr)

    print(f"[done] wrote {n} predicted + reference wavs to {out_dir}")


if __name__ == "__main__":
    main()
