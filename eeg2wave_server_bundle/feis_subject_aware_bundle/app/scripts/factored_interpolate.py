"""Speaker-interpolation demo: fix the content (from one EEG trial), sweep the
speaker embedding between two subjects -> a gradient of "same phoneme, different
voice" wavs. Demonstrates that cross-subject VOICE characteristics were learned
and that the voice axis is continuous (voice-conversion style).

  python scripts/factored_interpolate.py --config configs/factored.yaml \
      --checkpoint <run>/checkpoints/best.pt --label f --subjects 01,10 \
      --steps 5 --out-dir <run>/interp
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
from src.feis_factored.data import FactoredFEISDataset
from src.feis_factored.model import FactoredConfig, FactoredEEG2Speech
from src.feis_factored.targets import FactoredTargets
from src.feis_factored.synth import build_codec_backend, latent_to_wav


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "factored.yaml"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--label", default="f", help="content (prompt) to hold fixed")
    p.add_argument("--subjects", default="01,10", help="two subject ids to interpolate voice between")
    p.add_argument("--stage", default="stimuli")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = FactoredEEG2Speech(FactoredConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True); model.eval()
    mean = np.asarray(ckpt["target_mean"], np.float32); std = np.asarray(ckpt["target_std"], np.float32)
    sub_vocab = ckpt["subject_vocab"]; sub_to_id = {s: i for i, s in enumerate(sub_vocab)}

    targets = FactoredTargets(resolve_bundle_path(cfg["data"]["target_cache"], BUNDLE_DIR))
    subA, subB = [s.strip() for s in args.subjects.split(",")][:2]
    # find one EEG trial of (subA, label) to provide the CONTENT — search ALL splits
    # (the cell may be a held-out one, absent from train).
    common = dict(data_root=resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR), targets=targets,
                  stages=(args.stage,),
                  include_anomalous=bool(cfg["data"].get("include_anomalous", False)),
                  holdout_offset=int(ckpt.get("holdout_offset", cfg["data"].get("holdout_offset", 0))),
                  holdout_random=bool(ckpt.get("holdout_random", cfg["data"].get("holdout_random", False))))
    src = None
    for split in ("train", "val_seen", "test_seen", "test_holdout"):
        try:
            ds = FactoredFEISDataset(split=split, **common)
        except ValueError:
            continue
        src = next((ds[i] for i in range(len(ds))
                    if ds.entries[i].subject == subA and ds.entries[i].label == args.label), None)
        if src is not None:
            break
    if src is None:
        raise SystemExit(f"no {subA}:{args.label} trial in stage {args.stage} (any split)")

    backend = build_codec_backend(str(resolve_bundle_path("../models/encodec_24khz", BUNDLE_DIR)))
    sr = backend.sample_rate
    out_dir = ensure_dir(args.out_dir)

    with torch.no_grad():
        eeg = src["eeg"].unsqueeze(0).to(device)
        stage = src["stage_idx"].view(1).to(device)
        # content comes from the EEG; speaker is what we sweep
        cond = model.stage_embedding(stage)
        seq = model.encoder(eeg, cond).transpose(1, 2)
        content_seq = model.content_seq_head(seq)                    # [1, T, content_dim]
        embA = model.speaker_embedding(torch.tensor([sub_to_id[subA]], device=device))
        embB = model.speaker_embedding(torch.tensor([sub_to_id[subB]], device=device))
        T = content_seq.shape[1]
        for k in range(args.steps):
            a = k / max(args.steps - 1, 1)
            spk = (1 - a) * embA + a * embB                          # interpolate voice
            gen_in = torch.cat([content_seq, spk.unsqueeze(1).expand(-1, T, -1)], dim=-1)
            pred = model.generator(gen_in).squeeze(0).cpu().numpy()
            wav = latent_to_wav(backend, pred, mean, std)
            sf.write(out_dir / f"{args.label}_voice_{subA}to{subB}_{a:.2f}.wav", wav, sr)
    print(f"[done] {args.steps} interpolated voices ('{args.label}', {subA}->{subB}) in {out_dir}")


if __name__ == "__main__":
    main()
