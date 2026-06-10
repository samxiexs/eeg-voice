"""Extract trial-level EnCodec-latent targets for KaraOne.

Unlike FEIS (canonical, template-level), KaraOne has a trial-synchronous overt
wav per trial, so each thinking trial gets its own target keyed by
"subject:trial_index". Output npz is consumed by src/v3/datasets._TargetCache.

  python scripts/v3_extract_karaone_targets.py \
      --karaone-root ../data/karaone \
      --codec-model ../models/encodec_24khz \
      --out ../artifacts/audio_targets/karaone_trial_encodec_latents.npz \
      --extract-steps 200 --stage thinking
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.audio_features import AudioFeatureConfig, load_codec_backend
from src.utils import ensure_dir, load_wav_fixed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract KaraOne trial-level EnCodec targets.")
    p.add_argument("--karaone-root", default=str(BUNDLE_DIR / "data" / "karaone"))
    p.add_argument("--codec-model", default=str(BUNDLE_DIR / "models" / "encodec_24khz"))
    p.add_argument("--out", required=True)
    p.add_argument("--stage", default="thinking")
    p.add_argument("--duration-sec", type=float, default=2.0, help="EnCodec input window")
    p.add_argument("--extract-steps", type=int, default=200, help="Max stored EnCodec frames")
    p.add_argument("--bandwidth", type=float, default=6.0)
    p.add_argument("--audio-sr", type=int, default=16000)
    p.add_argument("--limit", type=int, default=None, help="Debug: only extract the first N trials")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.karaone_root)
    backend = load_codec_backend(AudioFeatureConfig(
        duration_sec=float(args.duration_sec),
        target_kind="encodec_latent",
        backend="encodec_latent",
        codec_model_name_or_path=args.codec_model,
        codec_bandwidth=float(args.bandwidth),
        local_files_only=True,
    ))

    # Collect unique (subject, trial, audio_path) for the chosen stage.
    rows = []
    with (root / "segments.csv").open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("segment_stage") != args.stage:
                continue
            sid = str(r["subject_id"])
            trial = int(r["trial_index"])
            rows.append((sid, trial, str(r["audio_path"])))
    rows = sorted(set(rows), key=lambda x: (x[0], x[1]))
    if args.limit is not None:
        rows = rows[: int(args.limit)]

    keys, seqs, valids = [], [], []
    n_audio_samples = int(round(args.audio_sr * args.duration_sec))
    for sid, trial, audio_rel in rows:
        wav = load_wav_fixed(root / audio_rel, sample_rate=args.audio_sr,
                             n_samples=n_audio_samples, normalize="rms")
        feat = backend.extract(wav, sample_rate=args.audio_sr)        # [T, D]
        seq = feat["target_sequence"].astype(np.float32)
        t = seq.shape[0]
        valid = min(t, args.extract_steps)
        out = np.zeros((args.extract_steps, seq.shape[1]), dtype=np.float32)
        out[:valid] = seq[:valid]
        keys.append(f"{sid}:{trial}")
        seqs.append(out)
        valids.append(valid)

    seqs_arr = np.stack(seqs, 0).astype(np.float32)
    valids_arr = np.asarray(valids, dtype=np.int32)
    # Normalisation stats over valid frames only.
    flat = np.concatenate([seqs_arr[i, : valids_arr[i]] for i in range(len(seqs))], 0)
    mean = flat.mean(0).astype(np.float32)
    std = np.maximum(flat.std(0), 1e-6).astype(np.float32)

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    np.savez(
        out_path,
        target_keys=np.asarray(keys),
        target_sequences=seqs_arr,
        target_valid_steps=valids_arr,
        target_mean=mean,
        target_std=std,
        target_kind=np.asarray("encodec_latent"),
    )
    print(f"[done] {len(keys)} trial targets -> {out_path} | seq {seqs_arr.shape}")


if __name__ == "__main__":
    main()
