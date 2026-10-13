"""Atomic, reproducible experiment bookkeeping; no implicit downloads."""
from __future__ import annotations
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any
import numpy as np


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def seed_all(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_name(requested: str = "auto") -> str:
    import torch
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def atomic_checkpoint(path: Path, value: Any) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, device: str = "cpu") -> dict:
    import torch
    return torch.load(path, map_location=device, weights_only=False)


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


class Moments:
    """Streaming per-feature moments; masks exclude bad channels."""
    def __init__(self, features: int):
        self.count = np.zeros(features, dtype=np.float64)
        self.total = np.zeros(features, dtype=np.float64)
        self.square = np.zeros(features, dtype=np.float64)

    def add(self, x: np.ndarray, mask: np.ndarray | None = None) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1, len(self.count))
        valid = np.isfinite(x)
        if mask is not None:
            valid &= np.broadcast_to(mask, x.shape)
        safe = np.where(valid, x, 0)
        self.count += valid.sum(0)
        self.total += safe.sum(0)
        self.square += (safe * safe).sum(0)

    def result(self, floor: float = 1e-4) -> dict:
        if np.any(self.count == 0):
            raise ValueError("Some features have no training observations")
        mean = self.total / self.count
        std = np.sqrt(np.maximum(self.square / self.count - mean**2, floor**2))
        return {"mean": mean.tolist(), "std": std.tolist(), "count": self.count.tolist()}
