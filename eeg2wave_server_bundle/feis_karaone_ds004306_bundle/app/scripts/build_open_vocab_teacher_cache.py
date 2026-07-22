#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.signal import resample_poly
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer


APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_0722.data import AudioCodeBank, TeacherBank, load_context, normalize_label, resolve_config_path  # noqa: E402
from src.open_vocab_0722.audio_io import read_wav  # noqa: E402
from src.open_vocab_0722.lineage import file_sha256  # noqa: E402


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32)
    divisor = math.gcd(int(source_rate), int(target_rate))
    return resample_poly(audio, target_rate // divisor, source_rate // divisor).astype(np.float32)


def load_audio(path: Path, target_rate: int, start: int = 0, frames: int = -1) -> np.ndarray:
    audio, rate = read_wav(path, start=start, frames=frames)
    return resample(audio, int(rate), int(target_rate))


def project_records(context: object, bank: AudioCodeBank) -> list[dict[str, object]]:
    by_key = {row["audio_key"]: row for row in context.rows}
    records = []
    for index, key in enumerate(bank.keys):
        row = by_key[str(key)]
        records.append(
            {
                "key": str(key),
                "path": str(context.audio_root / row["audio_relpath"]),
                "start": 0,
                "frames": -1,
                "fit": bool(bank.fit_split[index]) and row["dataset"] != "ds004306",
                "source": "project",
            }
        )
    return records


def public_records(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with np.load(path, allow_pickle=False) as raw:
        return [
            {
                "key": str(raw["keys"][index]),
                "path": str(raw["source_paths"][index]),
                "start": int(raw["segment_start_sample"][index]),
                "frames": int(raw["segment_source_frames"][index]),
                "fit": bool(raw["fit_split"][index]),
                "source": "public",
            }
            for index in range(len(raw["keys"]))
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen XLS-R acoustic tokens and XLM-R label auxiliaries")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--project-only", action="store_true", help="Smoke/debug only; excludes the 100h public corpus")
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    context = load_context(config_path)
    output = resolve_config_path(config_path, cfg["paths"]["teacher_cache"])
    if output.exists() and not args.rebuild:
        print(f"[0722 teacher] already exists: {output}")
        return
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    project_cache = resolve_config_path(config_path, cfg["paths"]["project_audio_cache"])
    public_cache = resolve_config_path(config_path, cfg["paths"]["public_audio_cache"])
    records = project_records(context, AudioCodeBank(project_cache))
    if not args.project_only:
        if not public_cache.is_file():
            raise FileNotFoundError("Public audio cache is missing; use --project-only only for smoke/debug")
        records += public_records(public_cache)
    keys = [str(row["key"]) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Teacher audio keys must be unique across project/public sources")
    device = torch.device(args.device) if args.device else default_device()
    xlsr_name = str(cfg["teachers"]["xlsr_model"])
    processor = AutoProcessor.from_pretrained(xlsr_name)
    xlsr = AutoModel.from_pretrained(xlsr_name).to(device).eval()
    for parameter in xlsr.parameters():
        parameter.requires_grad_(False)
    sample_rate = int(cfg["codec"]["sample_rate"])
    token_steps = int(cfg["teachers"]["audio_token_steps"])
    all_tokens: list[np.ndarray] = []
    audio_index: dict[str, list[object]] = {}
    shard_number = 0

    def flush() -> None:
        nonlocal all_tokens, shard_number
        if not all_tokens:
            return
        shard_name = f"audio_{shard_number:05d}.npz"
        # Entries are assigned before this flush; only the file name is needed here.
        np.savez_compressed(output / shard_name, tokens=np.asarray(all_tokens, dtype=np.float16))
        all_tokens = []
        shard_number += 1

    with torch.inference_mode():
        for start in tqdm(range(0, len(records), args.batch_size), desc="[0722 teacher] XLS-R", unit="batch"):
            chunk = records[start : start + args.batch_size]
            waveforms = [load_audio(Path(str(row["path"])), sample_rate, int(row["start"]), int(row["frames"])) for row in chunk]
            encoded = processor(waveforms, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = xlsr(**encoded).last_hidden_state
            pooled = F.adaptive_avg_pool1d(hidden.transpose(1, 2), token_steps).transpose(1, 2)
            values = pooled.float().cpu().numpy()
            for row, value in zip(chunk, values):
                if len(all_tokens) >= args.shard_size:
                    flush()
                shard_name = f"audio_{shard_number:05d}.npz"
                audio_index[str(row["key"])] = [shard_name, len(all_tokens)]
                all_tokens.append(value)
    flush()

    text_name = str(cfg["teachers"]["text_model"])
    tokenizer = AutoTokenizer.from_pretrained(text_name)
    text_model = AutoModel.from_pretrained(text_name).to(device).eval()
    for parameter in text_model.parameters():
        parameter.requires_grad_(False)
    labels = sorted({normalize_label(row["label"]) for row in context.rows})
    prompts = [f"language=en; spoken sound={label}" for label in labels]
    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(prompts), 32), desc="[0722 teacher] XLM-R", unit="batch"):
            encoded = tokenizer(prompts[start : start + 32], padding=True, truncation=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            embeddings.extend(text_model(**encoded).last_hidden_state[:, 0].float().cpu().numpy())
    np.savez_compressed(output / "text.npz", keys=np.asarray(labels), embeddings=np.asarray(embeddings, dtype=np.float16))
    audio_dimension = int(next(iter(np.load(output / "audio_00000.npz", allow_pickle=False)["tokens"])).shape[-1])
    cache_files = sorted(output.glob("audio_*.npz")) + [output / "text.npz"]
    file_digests = {path.name: file_sha256(path) for path in tqdm(cache_files, desc="[0722 teacher] shard hashes", unit="file")}
    aggregate = hashlib.sha256()
    for name, digest in sorted(file_digests.items()):
        aggregate.update(name.encode("utf-8")); aggregate.update(b"\0"); aggregate.update(digest.encode("ascii")); aggregate.update(b"\0")
    index = {
        "schema_version": TeacherBank.SCHEMA_VERSION,
        "xlsr_model": xlsr_name,
        "text_model": text_name,
        "audio_dimension": audio_dimension,
        "text_dimension": int(np.asarray(embeddings).shape[-1]),
        "audio_token_steps": token_steps,
        "audio_index": audio_index,
        "audio_fit": {str(row["key"]): bool(row["fit"]) for row in records},
        "audio_source": {str(row["key"]): str(row["source"]) for row in records},
        "text_file": "text.npz",
        "project_only": bool(args.project_only),
        "transcripts_used_for_generator": False,
        "project_cache_sha256": file_sha256(project_cache),
        "public_cache_sha256": file_sha256(public_cache) if public_cache.is_file() and not args.project_only else "absent",
        "file_sha256": file_digests,
        "content_sha256": aggregate.hexdigest(),
    }
    (output / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "audio_items": len(records), "text_labels": len(labels), "shards": shard_number, "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
