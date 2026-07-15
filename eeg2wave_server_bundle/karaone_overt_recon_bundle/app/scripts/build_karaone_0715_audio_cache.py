from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.karaone_0715.codec import DiscreteEncodec, DiscreteEncodecConfig  # noqa: E402
from src.karaone_0715.data import (  # noqa: E402
    SplitManifest0715,
    audio_envelope,
    fit_audit,
    load_audio,
    read_trial_records,
    records_for_split,
    sha256_bytes,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build independent 0715 exact EnCodec-code cache.")
    parser.add_argument("--config", default=str(APP_DIR / "configs" / "karaone_0715.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--keep-parts", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else APP_DIR / path


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cache_path(cfg: dict[str, Any], seed: int) -> Path:
    return resolve(cfg["paths"]["cache_root"]) / f"karaone_0715_encodec_codes_s{seed}.npz"


def valid_part(path: Path, keys: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        raw = np.load(path, allow_pickle=False)
        return raw["keys"].astype(str).tolist() == keys and "encodec_codes" in raw.files
    except Exception:  # noqa: BLE001 - a corrupt partial batch must be regenerated
        return False


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg["run"]["seed"])
    data_root = resolve(cfg["data"]["root"])
    destination = Path(args.output) if args.output else cache_path(cfg, seed)
    destination = destination if destination.is_absolute() else APP_DIR / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.rebuild:
        print(f"[0715 cache] already exists: {destination}", flush=True)
        return
    manifest = SplitManifest0715.build(data_root)
    records = read_trial_records(data_root)
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
    part_dir = destination.parent / f".{destination.stem}.parts"
    if args.rebuild and part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)
    part_paths: list[Path] = []
    batches = list(range(0, len(records), batch_size))
    for part_index, start in enumerate(tqdm(batches, desc="[0715 cache] EnCodec encode", unit="batch", dynamic_ncols=True)):
        chunk = records[start : start + batch_size]
        keys = [record.key for record in chunk]
        part_path = part_dir / f"part_{part_index:04d}.npz"
        part_paths.append(part_path)
        if valid_part(part_path, keys):
            continue
        audio = np.stack(
            [load_audio(data_root / record.audio_path, sample_rate=int(codec_cfg["sample_rate"]), duration_sec=float(codec_cfg["duration_sec"])) for record in chunk]
        )
        encoded = backend.encode(audio)
        envelopes, onsets, durations = [], [], []
        for waveform in audio:
            envelope, onset, duration = audio_envelope(waveform, steps=int(codec_cfg["code_steps"]))
            envelopes.append(envelope)
            onsets.append(onset)
            durations.append(duration)
        np.savez_compressed(
            part_path,
            keys=np.asarray(keys),
            subjects=np.asarray([record.subject for record in chunk]),
            labels=np.asarray([record.label for record in chunk]),
            audio_paths=np.asarray([record.audio_path for record in chunk]),
            encodec_codes=encoded["codes"],
            encodec_scale=encoded["scale"],
            encodec_scale_valid=encoded["scale_valid"],
            audio_envelope=np.stack(envelopes).astype(np.float32),
            onset=np.asarray(onsets, dtype=np.float32),
            duration=np.asarray(durations, dtype=np.float32),
        )
    arrays: dict[str, list[np.ndarray]] = {}
    for part_path in tqdm(part_paths, desc="[0715 cache] merge", unit="part", dynamic_ncols=True):
        raw = np.load(part_path, allow_pickle=False)
        for key in raw.files:
            arrays.setdefault(key, []).append(np.asarray(raw[key]))
    payload = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    payload["fit_split"] = np.asarray([record.subject in manifest.train_subjects for record in records], dtype=bool)
    payload["version"] = np.asarray("0715")
    payload["codec_sample_rate"] = np.asarray(backend.codec_sample_rate, dtype=np.int32)
    payload["codec_bandwidth"] = np.asarray(float(codec_cfg["bandwidth"]), dtype=np.float32)
    if payload["keys"].astype(str).tolist() != [record.key for record in records]:
        raise ValueError("Merged 0715 cache order differs from trials.csv")
    expected_shape = (len(records), int(codec_cfg["codebooks"]), int(codec_cfg["code_steps"]))
    if payload["encodec_codes"].shape != expected_shape:
        raise ValueError(f"Expected code shape {expected_shape}, got {payload['encodec_codes'].shape}")
    np.savez_compressed(destination, **payload)
    audit = {
        **fit_audit(records_for_split(data_root, manifest, "subject_train"), manifest, "0715_audio_code_model_fit_boundary"),
        "cache": str(destination),
        "cache_sha256": sha256_bytes(destination.read_bytes()),
        "all_targets_cached": len(records),
        "heldout_targets_encoded_but_not_fit": True,
        "codec_codes_shape": list(payload["encodec_codes"].shape),
        "codec_code_range": [int(payload["encodec_codes"].min()), int(payload["encodec_codes"].max())],
        "codec_sample_rate": int(backend.codec_sample_rate),
        "codec_bandwidth": float(codec_cfg["bandwidth"]),
        "device": str(device),
        "test_accessed_for_model_selection": False,
    }
    write_json(destination.with_suffix(".audit.json"), audit)
    manifest.write(destination.parent / "karaone_0715_split_manifest.json")
    if not args.keep_parts:
        shutil.rmtree(part_dir)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
