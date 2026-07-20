from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
KARA_APP = APP_DIR.parents[1] / "karaone_overt_recon_bundle" / "app"
if str(KARA_APP) not in sys.path:
    sys.path.insert(0, str(KARA_APP))

from src.combined_0715.audio_eval import CACHE_SCHEMA_VERSION, scalar_text  # noqa: E402
from src.combined_0715.data import load_context, sha256_bytes  # noqa: E402
from src.karaone_0715.codec import DiscreteEncodec, DiscreteEncodecConfig  # noqa: E402
from src.karaone_0715.data import audio_envelope, load_audio  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deduplicated combined 0715 EnCodec cache.")
    parser.add_argument("--config", default=str(APP_DIR / "configs" / "combined_0715_v1.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else APP_DIR / "configs" / path


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    context = load_context(config_path)
    destination = Path(args.output) if args.output else resolve(cfg["paths"]["cache_root"]) / "combined_0715_encodec_codes.npz"
    destination = destination if destination.is_absolute() else APP_DIR / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.rebuild:
        with np.load(destination, allow_pickle=False) as existing:
            existing_version = scalar_text(existing["cache_schema_version"]) if "cache_schema_version" in existing.files else "legacy"
        if existing_version != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Existing cache uses schema {existing_version!r}; v2 is required. "
                "Re-run this command with --rebuild."
            )
        print(f"[combined cache] already exists: {destination}", flush=True)
        return

    unique: dict[str, dict[str, str]] = {}
    for row in context.rows:
        key = row["audio_key"]
        if key not in unique:
            unique[key] = dict(row)
        else:
            fields = ("audio_relpath", "audio_valid_samples", "dataset", "label")
            inconsistent = [field for field in fields if unique[key].get(field) != row.get(field)]
            if inconsistent:
                raise ValueError(f"audio_key {key} has inconsistent metadata fields: {inconsistent}")
    records = list(unique.values())
    records.sort(key=lambda row: row["audio_key"])
    device = torch.device(args.device) if args.device else default_device()
    codec_cfg = cfg["codec"]
    backend = DiscreteEncodec(
        DiscreteEncodecConfig(
            model_path=str(resolve(cfg["paths"]["encodec_model"])),
            sample_rate=int(codec_cfg["sample_rate"]),
            duration_sec=float(codec_cfg["duration_sec"]),
            bandwidth=float(codec_cfg["bandwidth"]),
        ),
        device,
    )
    batch_size = int(args.batch_size or codec_cfg["extraction_batch_size"])
    code_parts, scale_parts, scale_valid_parts = [], [], []
    envelopes, onsets, durations, valid_steps, audio_valid_samples = [], [], [], [], []
    target_audio_samples = int(round(int(codec_cfg["sample_rate"]) * float(codec_cfg["duration_sec"])))
    for start in tqdm(range(0, len(records), batch_size), desc="[combined cache] EnCodec encode", unit="batch", dynamic_ncols=True):
        chunk = records[start : start + batch_size]
        audio = np.stack([load_audio(context.root / row["audio_relpath"], sample_rate=int(codec_cfg["sample_rate"]), duration_sec=float(codec_cfg["duration_sec"])) for row in chunk])
        encoded = backend.encode(audio)
        code_parts.append(encoded["codes"])
        scale_parts.append(encoded["scale"])
        scale_valid_parts.append(encoded["scale_valid"])
        for row, waveform in zip(chunk, audio):
            envelope, onset, duration = audio_envelope(waveform, steps=int(codec_cfg["code_steps"]))
            envelopes.append(envelope)
            onsets.append(onset)
            durations.append(duration)
            actual_samples = min(max(int(row.get("audio_valid_samples") or target_audio_samples), 1), len(waveform))
            audio_valid_samples.append(actual_samples)
            valid_steps.append(max(1, min(int(codec_cfg["code_steps"]), int(np.ceil(actual_samples / len(waveform) * int(codec_cfg["code_steps"]))))))
    keys = [row["audio_key"] for row in records]
    train_keys = {row["audio_key"] for row in context.rows if context.split_for(row) == "train"}
    fit_split = np.asarray([key in train_keys for key in keys], dtype=bool)
    payload = {
        "version": np.asarray(CACHE_SCHEMA_VERSION),
        "cache_schema_version": np.asarray(CACHE_SCHEMA_VERSION),
        "config_sha256": np.asarray(sha256_bytes(config_path.read_bytes())),
        "keys": np.asarray(keys),
        "datasets": np.asarray([row["dataset"] for row in records]),
        "labels": np.asarray([row["label"] for row in records]),
        "audio_relpaths": np.asarray([row["audio_relpath"] for row in records]),
        "audio_valid_samples": np.asarray(audio_valid_samples, dtype=np.int64),
        "encodec_codes": np.concatenate(code_parts, axis=0).astype(np.int16),
        "encodec_scale": np.concatenate(scale_parts, axis=0).astype(np.float32),
        "encodec_scale_valid": np.concatenate(scale_valid_parts, axis=0).astype(bool),
        "audio_envelope": np.asarray(envelopes, dtype=np.float32),
        "onset": np.asarray(onsets, dtype=np.float32),
        "duration": np.asarray(durations, dtype=np.float32),
        "code_valid_steps": np.asarray(valid_steps, dtype=np.int64),
        "fit_split": fit_split,
        "audio_sample_rate": np.asarray(int(codec_cfg["sample_rate"]), dtype=np.int32),
        "codec_sample_rate": np.asarray(int(backend.codec_sample_rate), dtype=np.int32),
        "codec_duration_sec": np.asarray(float(codec_cfg["duration_sec"]), dtype=np.float32),
        "codec_bandwidth": np.asarray(float(codec_cfg["bandwidth"]), dtype=np.float32),
    }
    if payload["encodec_codes"].shape != (len(records), int(codec_cfg["codebooks"]), int(codec_cfg["code_steps"])):
        raise ValueError(f"Unexpected code shape: {payload['encodec_codes'].shape}")
    np.savez_compressed(destination, **payload)
    audit = {
        "version": CACHE_SCHEMA_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "config_sha256": sha256_bytes(config_path.read_bytes()),
        "cache": str(destination),
        "cache_sha256": sha256_bytes(destination.read_bytes()),
        "unique_audio_keys": len(records),
        "fit_audio_keys": int(fit_split.sum()),
        "code_shape": list(payload["encodec_codes"].shape),
        "audio_sample_rate": int(codec_cfg["sample_rate"]),
        "codec_sample_rate": int(backend.codec_sample_rate),
        "codec_duration_sec": float(codec_cfg["duration_sec"]),
        "codec_bandwidth": float(codec_cfg["bandwidth"]),
        "device": str(device),
        "test_audio_encoded_but_not_fit": True,
        "ds004306_category_candidate_audio_cached": True,
        "ds004306_trial_level_pairing_claim_allowed": False,
    }
    destination.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
