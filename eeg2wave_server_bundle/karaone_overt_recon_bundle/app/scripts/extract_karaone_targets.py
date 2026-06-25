from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.audio_features import AudioFeatureConfig, _extract_target_from_audio, build_audio_feature_backend, compute_prosody_target
from src.utils import ensure_dir, load_simple_yaml, load_wav_fixed, resolve_bundle_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract trial-level KaraOne acoustic targets (mel or EnCodec latent).")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--karaone-root", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--target", choices=["mel", "encodec_latent", "hubert_sequence"], default=None, help="target kind: mel/encodec_latent (renderable) or hubert_sequence (semantic aux). Default: config target.kind")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    root = resolve_bundle_path(args.karaone_root or cfg["data"]["root"], BUNDLE_DIR)
    audio_cfg = cfg["audio"]
    target_cfg = cfg["targets"]
    tgt = cfg.get("target", {})
    kind = args.target or str(tgt.get("kind", target_cfg.get("target_kind", "encodec_latent")))

    # Choose output cache by target kind (mel / encodec / hubert caches coexist).
    if args.out:
        out_default = args.out
    elif kind == "mel":
        out_default = tgt.get("cache_mel", "../artifacts/audio_targets/karaone_trial_mel.npz")
    elif kind == "hubert_sequence":
        out_default = tgt.get("cache_hubert", "../artifacts/audio_targets/karaone_trial_hubert.npz")
    else:
        out_default = tgt.get("cache_encodec", cfg["data"]["target_cache"])
    output_path = resolve_bundle_path(out_default, BUNDLE_DIR)
    ensure_dir(output_path.parent)

    if kind == "mel":
        backend_kind = "mel"
    elif kind == "hubert_sequence":
        backend_kind = "ssl_local"  # local HuBERT/wav2vec2 embedder (offline)
    else:
        backend_kind = str(target_cfg.get("backend", "encodec_latent"))
    # Local SSL (HuBERT) weights; the bundled copy lives in the FEIS bundle.
    ssl_path = resolve_bundle_path(
        tgt.get("hubert_ssl_path", "../../feis_subject_aware_bundle/models/hubert-base-ls960"), BUNDLE_DIR
    )

    feature_cfg = AudioFeatureConfig(
        sample_rate=int(audio_cfg["sample_rate"]),
        duration_sec=float(audio_cfg["duration_sec"]),
        normalize=str(audio_cfg.get("normalize", "rms")),
        target_rms=float(audio_cfg.get("target_rms", 0.08)),
        max_gain=float(audio_cfg.get("max_gain", 10.0)),
        backend=backend_kind,
        target_kind=kind,
        ssl_model_name_or_path=str(ssl_path),
        sequence_target_steps=int(tgt.get("hubert_steps", 50)),
        codec_model_name_or_path=str(resolve_bundle_path(target_cfg["codec_model_name_or_path"], BUNDLE_DIR)),
        codec_bandwidth=float(target_cfg.get("codec_bandwidth", 6.0)),
        local_files_only=bool(target_cfg.get("local_files_only", True)),
        n_mels=int(tgt.get("n_mels", 80)),
        mel_n_fft=int(tgt.get("mel_n_fft", 1024)),
        mel_hop=int(tgt.get("mel_hop", 256)),
    )
    print(f"[extract] target kind={kind} -> {output_path}")
    backend_name, backend = build_audio_feature_backend(feature_cfg)
    rows = list(csv.DictReader((root / "trials.csv").open("r", encoding="utf-8", newline="")))
    if args.limit:
        rows = rows[: int(args.limit)]

    template_ids, subject_ids, trial_indices, labels, audio_paths = [], [], [], [], []
    target_sequences, target_masks, target_summaries, prosody_targets = [], [], [], []
    target_rms_values, target_log_rms_values, decoder_scales = [], [], []
    feature_backend = []
    for row in tqdm(rows, desc="extract karaone targets", unit="trial"):
        subject = str(row["subject_id"])
        trial = int(row["trial_index"])
        relpath = str(row["audio_path"])
        audio = load_wav_fixed(
            root / relpath,
            sample_rate=feature_cfg.sample_rate,
            n_samples=int(round(feature_cfg.sample_rate * feature_cfg.duration_sec)),
            normalize=feature_cfg.normalize,
            target_rms=feature_cfg.target_rms,
            max_gain=feature_cfg.max_gain,
        )
        audio_rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)) + 1e-8)
        target = _extract_target_from_audio(backend_name, backend, audio, feature_cfg)
        template_ids.append(f"{subject}:{trial}")
        subject_ids.append(subject)
        trial_indices.append(trial)
        labels.append(str(row["label"]))
        audio_paths.append(relpath)
        target_sequences.append(np.asarray(target["target_sequence"], dtype=np.float32))
        target_masks.append(np.asarray(target["target_mask"], dtype=np.float32))
        target_summaries.append(np.asarray(target["target_summary"], dtype=np.float32))
        prosody_targets.append(compute_prosody_target(audio, feature_cfg.sample_rate))
        target_rms_values.append(audio_rms)
        target_log_rms_values.append(float(np.log(audio_rms + 1e-8)))
        feature_backend.append(backend_name)
        decoder_scales.append(np.asarray(target.get("decoder_scales", np.ones((1,), dtype=np.float32)), dtype=np.float32))

    if not target_sequences:
        raise ValueError("No KaraOne target sequences were extracted")
    sequence_shapes = {tuple(item.shape) for item in target_sequences}
    if len(sequence_shapes) != 1:
        raise ValueError(f"Expected fixed EnCodec sequence shapes, got {sorted(sequence_shapes)}")
    scale_shapes = {tuple(item.shape) for item in decoder_scales}
    if len(scale_shapes) != 1:
        raise ValueError(f"Expected fixed decoder scale shapes, got {sorted(scale_shapes)}")

    stacked_sequences = np.stack(target_sequences, axis=0).astype(np.float32)
    target_mean = stacked_sequences.mean(axis=(0, 1)).astype(np.float32)
    target_std = np.maximum(stacked_sequences.std(axis=(0, 1)), 1e-6).astype(np.float32)
    scales = np.stack(decoder_scales, axis=0).astype(np.float32)
    payload = {
        "template_ids": np.asarray(template_ids),
        "subject_ids": np.asarray(subject_ids),
        "trial_indices": np.asarray(trial_indices, dtype=np.int32),
        "labels": np.asarray(labels),
        "audio_paths": np.asarray(audio_paths),
        "target_sequences": stacked_sequences,
        "target_masks": np.stack(target_masks, axis=0).astype(np.float32),
        "target_summaries": np.stack(target_summaries, axis=0).astype(np.float32),
        "prosody_targets": np.stack(prosody_targets, axis=0).astype(np.float32),
        "target_mean": target_mean,
        "target_std": target_std,
        "target_rms": np.asarray(target_rms_values, dtype=np.float32),
        "target_log_rms": np.asarray(target_log_rms_values, dtype=np.float32),
        "decoder_scales": scales,
        "default_decoder_scales": scales.mean(axis=0).astype(np.float32),
        "feature_backend": np.asarray(feature_backend),
        "target_kind": np.asarray(feature_cfg.target_kind),
        "target_steps": np.asarray(stacked_sequences.shape[1], dtype=np.int32),
        "target_dim": np.asarray(stacked_sequences.shape[2], dtype=np.int32),
    }
    np.savez_compressed(output_path, **payload)
    metadata = {
        "output_path": str(output_path),
        "karaone_root": str(root),
        "num_trials": len(template_ids),
        "target_shape": list(stacked_sequences.shape[1:]),
        "codec_model_name_or_path": feature_cfg.codec_model_name_or_path,
        "codec_bandwidth": feature_cfg.codec_bandwidth,
        "duration_sec": feature_cfg.duration_sec,
        "sample_rate": feature_cfg.sample_rate,
        "target_rms_mean": float(np.mean(target_rms_values)),
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved target cache to {output_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

