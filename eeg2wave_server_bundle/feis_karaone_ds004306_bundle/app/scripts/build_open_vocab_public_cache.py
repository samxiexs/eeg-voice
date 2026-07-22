#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.signal import resample_poly
from tqdm import tqdm


APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
KARA_APP = APP.parents[1] / "karaone_overt_recon_bundle/app"
if str(KARA_APP) not in sys.path:
    sys.path.insert(0, str(KARA_APP))

from src.open_vocab_0722.data import read_csv, resolve_config_path  # noqa: E402
from src.open_vocab_0722.audio_io import canonical_audio_sha256, read_wav  # noqa: E402
from src.open_vocab_0722.lineage import file_sha256  # noqa: E402
from src.karaone_0715.codec import DiscreteEncodec, DiscreteEncodecConfig  # noqa: E402
from src.karaone_0715.data import audio_envelope  # noqa: E402


SCHEMA = "openvoice-public-audio-cache-v1"


def device_default() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_segment(row: dict[str, str], target_rate: int, target_samples: int) -> tuple[np.ndarray, int]:
    path = Path(row["source_path"])
    start = int(row["segment_start_sample"])
    frames = int(row["segment_valid_samples"])
    audio, source_rate = read_wav(path, start=start, frames=frames)
    if source_rate != target_rate:
        divisor = math.gcd(int(source_rate), int(target_rate))
        audio = resample_poly(audio, target_rate // divisor, source_rate // divisor).astype(np.float32)
    valid = min(len(audio), target_samples)
    output = np.zeros(target_samples, dtype=np.float32)
    output[:valid] = audio[:valid]
    peak = float(np.max(np.abs(output[:valid]))) if valid else 0.0
    if peak > 0.99:
        output /= peak / 0.99
    return output, max(1, valid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode the fixed public speech manifest with frozen EnCodec")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = resolve_config_path(config_path, cfg["paths"]["public_audio_manifest"])
    output = resolve_config_path(config_path, cfg["paths"]["public_audio_cache"])
    if output.exists() and not args.rebuild:
        print(f"[0722 public cache] already exists: {output}")
        return
    rows = list(read_csv(manifest))
    leakage_audit = json.loads(resolve_config_path(config_path, cfg["paths"]["leakage_audit"]).read_text(encoding="utf-8"))
    forbidden_canonical = set(leakage_audit.get("heldout_generation_audio_canonical_sha256", []))
    codec = cfg["codec"]
    device = torch.device(args.device) if args.device else device_default()
    backend = DiscreteEncodec(
        DiscreteEncodecConfig(
            model_path=str(resolve_config_path(config_path, cfg["paths"]["encodec_model"])),
            sample_rate=int(codec["sample_rate"]),
            duration_sec=float(codec["duration_sec"]),
            bandwidth=float(codec["bandwidth"]),
        ),
        device,
    )
    target_samples = round(float(codec["duration_sec"]) * int(codec["sample_rate"]))
    codes, scales, scale_valid = [], [], []
    envelopes, onsets, durations, valid_samples, canonical_hashes = [], [], [], [], []
    for start in tqdm(range(0, len(rows), args.batch_size), desc="[0722 public] EnCodec", unit="batch"):
        chunk = rows[start : start + args.batch_size]
        loaded = [load_segment(row, int(codec["sample_rate"]), target_samples) for row in chunk]
        waveforms = np.stack([value[0] for value in loaded])
        chunk_hashes = [canonical_audio_sha256(waveform, int(codec["sample_rate"]), target_rate=int(codec["sample_rate"]), duration_sec=float(codec["duration_sec"])) for waveform in waveforms]
        overlap = sorted(set(chunk_hashes) & forbidden_canonical)
        if overlap:
            raise RuntimeError(f"Public audio contains {len(overlap)} canonical held-out EEG reference waveforms")
        canonical_hashes.extend(chunk_hashes)
        encoded = backend.encode(waveforms)
        codes.append(encoded["codes"])
        scales.append(encoded["scale"])
        scale_valid.append(encoded["scale_valid"])
        for waveform, valid in loaded:
            envelope, onset, duration = audio_envelope(waveform, steps=int(codec["code_steps"]))
            envelopes.append(envelope)
            onsets.append(onset)
            durations.append(duration)
            valid_samples.append(valid)
    fit = np.asarray([row["split"] == "train" for row in rows], dtype=bool)
    payload = {
        "schema_version": np.asarray(SCHEMA),
        "manifest_sha256": np.asarray(file_sha256(manifest)),
        "keys": np.asarray([row["audio_key"] for row in rows]),
        "languages": np.asarray([row["language"] for row in rows]),
        "speakers": np.asarray([row["speaker_id"] for row in rows]),
        "source_paths": np.asarray([row["source_path"] for row in rows]),
        "source_sha256": np.asarray([row["source_sha256"] for row in rows]),
        "canonical_audio_sha256": np.asarray(canonical_hashes),
        "segment_start_sample": np.asarray([int(row["segment_start_sample"]) for row in rows]),
        "segment_source_frames": np.asarray([int(row["segment_valid_samples"]) for row in rows]),
        "encodec_codes": np.concatenate(codes).astype(np.int16),
        "encodec_scale": np.concatenate(scales).astype(np.float32),
        "encodec_scale_valid": np.concatenate(scale_valid).astype(bool),
        "audio_envelope": np.asarray(envelopes, dtype=np.float32),
        "onset": np.asarray(onsets, dtype=np.float32),
        "duration": np.asarray(durations, dtype=np.float32),
        "audio_valid_samples": np.asarray(valid_samples, dtype=np.int64),
        "code_valid_steps": np.asarray([max(1, math.ceil(value / target_samples * int(codec["code_steps"]))) for value in valid_samples]),
        "fit_split": fit,
        "audio_sample_rate": np.asarray(int(codec["sample_rate"])),
        "codec_sample_rate": np.asarray(int(backend.codec_sample_rate)),
        "codec_duration_sec": np.asarray(float(codec["duration_sec"])),
        "codec_bandwidth": np.asarray(float(codec["bandwidth"])),
        "transcripts_used": np.asarray(False),
    }
    expected = (len(rows), int(codec["codebooks"]), int(codec["code_steps"]))
    if payload["encodec_codes"].shape != expected:
        raise ValueError(f"Unexpected EnCodec shape {payload['encodec_codes'].shape}, expected {expected}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    print(json.dumps({"output": str(output), "sha256": file_sha256(output), "segments": len(rows), "fit": int(fit.sum()), "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
