from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_simple_yaml(path: str | Path) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not value.strip():
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str):
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [_parse_scalar(part.strip()) for part in inner.split(",") if part.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def resolve_feis_root(path: str | Path) -> Path:
    candidate = Path(path)
    if (candidate / "segments.csv").exists() and (candidate / "subjects").exists():
        return candidate
    if (candidate / "feis" / "segments.csv").exists() and (candidate / "feis" / "subjects").exists():
        return candidate / "feis"
    raise FileNotFoundError(f"Could not resolve FEIS processed root from {candidate}")


def resolve_bundle_path(path: str | Path, base_dir: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(base_dir) / candidate


def audio_to_float32(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.integer):
        scale = float(np.iinfo(audio.dtype).max)
        return (audio.astype(np.float32) / scale).clip(-1.0, 1.0)
    return audio.astype(np.float32)


def resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    gcd = np.gcd(src_sr, dst_sr)
    up = dst_sr // gcd
    down = src_sr // gcd
    return resample_poly(audio, up=up, down=down).astype(np.float32)


def pad_or_crop_audio(audio: np.ndarray, target_len: int) -> np.ndarray:
    if audio.shape[0] >= target_len:
        return audio[:target_len].astype(np.float32)
    return np.pad(audio, (0, target_len - audio.shape[0])).astype(np.float32)


def load_wav_fixed(path: str | Path, sample_rate: int, n_samples: int) -> np.ndarray:
    sr, audio = wavfile.read(str(path))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio_to_float32(audio)
    audio = resample_audio(audio, int(sr), int(sample_rate))
    return pad_or_crop_audio(audio, n_samples)


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    audio = np.asarray(audio, dtype=np.float32).clip(-1.0, 1.0)
    wavfile.write(str(path), sample_rate, (audio * 32767.0).astype(np.int16))
