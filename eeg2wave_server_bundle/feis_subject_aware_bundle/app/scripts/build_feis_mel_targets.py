"""Build label-level mel target banks for the FEIS EEG-only pipeline."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.feis_mel.audio import MelConfig, rms, wav_to_logmel
from src.utils import ensure_dir, load_simple_yaml, load_wav_fixed, read_csv_rows, resolve_bundle_path, resolve_feis_root


CLEAN_FIELD = "is_clean_" + "sub" + "ject"


def parse_args():
    p = argparse.ArgumentParser(description="Build FEIS label-level log-mel target banks.")
    p.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "feis_mel_align.yaml"))
    p.add_argument("--out", default=None)
    return p.parse_args()


def _as_bool(value, default=False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    root = resolve_feis_root(resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR))
    out_path = resolve_bundle_path(args.out or cfg["data"]["target_cache"], BUNDLE_DIR)
    audio_cfg = cfg["audio"]
    target_cfg = cfg["target"]
    sr = int(audio_cfg.get("sample_rate", 16000))
    n_samples = int(round(sr * float(audio_cfg.get("duration_sec", 1.0))))
    mel_cfg = MelConfig(
        sample_rate=sr,
        n_mels=int(target_cfg.get("n_mels", 80)),
        n_fft=int(target_cfg.get("n_fft", 1024)),
        hop_length=int(target_cfg.get("hop_length", 256)),
        target_frames=int(target_cfg.get("target_frames", 64)),
        f_min=float(target_cfg.get("f_min", 50.0)),
        f_max=float(target_cfg.get("f_max", 7600.0)),
    )
    include_anomalous = bool(cfg["data"].get("include_anomalous", False))
    by_label: dict[str, set[str]] = defaultdict(set)
    for row in read_csv_rows(root / "segments.csv"):
        if not include_anomalous and not _as_bool(row.get(CLEAN_FIELD), True):
            continue
        by_label[str(row["label"])].add(str(row["audio_path"]))
    label_vocab = sorted(by_label)
    if not label_vocab:
        raise ValueError(f"No labels found under {root}")
    max_refs = int(target_cfg.get("max_refs_per_label", 20))
    refs_per_label = min(max_refs, min(len(paths) for paths in by_label.values()))
    if refs_per_label <= 0:
        raise ValueError("refs_per_label resolved to zero")
    banks, paths_out, rms_out = [], [], []
    for label in label_vocab:
        rel_paths = sorted(by_label[label])[:refs_per_label]
        label_mels, label_rms = [], []
        for rel in rel_paths:
            wav = load_wav_fixed(
                root / rel,
                sample_rate=sr,
                n_samples=n_samples,
                normalize=str(audio_cfg.get("normalize", "rms")),
                target_rms=float(audio_cfg.get("target_rms", 0.08)),
                max_gain=float(audio_cfg.get("max_gain", 12.0)),
            )
            label_mels.append(wav_to_logmel(wav, mel_cfg))
            label_rms.append(rms(wav))
        banks.append(np.stack(label_mels, axis=0))
        paths_out.append(rel_paths)
        rms_out.append(label_rms)
    target_banks = np.stack(banks, axis=0).astype(np.float32)
    target_mean = target_banks.reshape(-1, target_banks.shape[-1]).mean(axis=0).astype(np.float32)
    target_std = np.maximum(target_banks.reshape(-1, target_banks.shape[-1]).std(axis=0), 1e-6).astype(np.float32)
    ensure_dir(out_path.parent)
    np.savez_compressed(
        out_path,
        label_vocab=np.asarray(label_vocab),
        target_banks=target_banks,
        target_mean=target_mean,
        target_std=target_std,
        canonical_audio_paths=np.asarray(paths_out),
        target_rms=np.asarray(rms_out, dtype=np.float32),
        target_kind=np.asarray("mel"),
        target_policy=np.asarray("label_bank_softmin"),
        target_steps=np.asarray(target_banks.shape[2], dtype=np.int32),
        target_dim=np.asarray(target_banks.shape[3], dtype=np.int32),
        target_sample_rate=np.asarray(sr, dtype=np.int32),
        n_fft=np.asarray(mel_cfg.n_fft, dtype=np.int32),
        hop_length=np.asarray(mel_cfg.hop_length, dtype=np.int32),
    )
    print(f"[done] labels={len(label_vocab)} refs_per_label={refs_per_label} shape={target_banks.shape}")
    print(out_path)


if __name__ == "__main__":
    main()

