from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
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


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def resolve_stage_output_root(path: str | Path, base_dir: str | Path, stage: str, explicit: bool = False) -> Path:
    root = resolve_bundle_path(path, base_dir)
    if explicit:
        return root
    return root.parent / f"{root.name}-{stage}"


def resolve_optional_path(path: str | Path | None, base_dir: str | Path | None = None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute() or base_dir is None:
        return candidate
    return Path(base_dir) / candidate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def mean_pool_masked(sequence: torch.Tensor, valid_steps: torch.Tensor | None = None) -> torch.Tensor:
    if valid_steps is None:
        return sequence.mean(dim=-1)
    if sequence.ndim != 3:
        raise ValueError(f"Expected [B, C, T], got {tuple(sequence.shape)}")
    time_steps = sequence.shape[-1]
    device = sequence.device
    mask = torch.arange(time_steps, device=device).unsqueeze(0) < valid_steps.unsqueeze(1)
    mask = mask.unsqueeze(1).to(sequence.dtype)
    denom = mask.sum(dim=-1).clamp_min(1.0)
    return (sequence * mask).sum(dim=-1) / denom


def cosine_similarity_batch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_norm = pred / pred.norm(dim=-1, keepdim=True).clamp_min(eps)
    target_norm = target / target.norm(dim=-1, keepdim=True).clamp_min(eps)
    return (pred_norm * target_norm).sum(dim=-1)


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


def normalize_rms(audio: np.ndarray, target_rms: float = 0.08, max_gain: float = 12.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)) + 1e-8)
    gain = min(float(target_rms) / rms, float(max_gain))
    return (audio.astype(np.float32) * gain).clip(-0.95, 0.95)


def load_wav_fixed(
    path: str | Path,
    sample_rate: int,
    n_samples: int,
    normalize: str = "rms",
    target_rms: float = 0.08,
    max_gain: float = 12.0,
) -> np.ndarray:
    sr, audio = wavfile.read(str(path))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio_to_float32(audio)
    audio = resample_audio(audio, int(sr), int(sample_rate))
    audio = pad_or_crop_audio(audio, n_samples)
    if normalize == "rms":
        audio = normalize_rms(audio, target_rms=target_rms, max_gain=max_gain)
    return audio.astype(np.float32)


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    audio = np.asarray(audio, dtype=np.float32).clip(-1.0, 1.0)
    wavfile.write(str(path), sample_rate, (audio * 32767.0).astype(np.int16))


def save_training_curves(path: str | Path, history: list[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not history:
        return

    epochs = [item["epoch"] for item in history]
    train_total = [item["train"]["total"] for item in history]
    val_total = [item["val"]["total"] for item in history]
    train_cls = [item["train"]["cls_acc"] for item in history]
    val_cls = [item["val"]["cls_acc"] for item in history]
    perplexity = [item["perplexity"] for item in history]
    train_l1 = [item["train"]["l1"] for item in history]
    train_stft = [item["train"]["stft"] for item in history]
    train_log_stft = [item["train"].get("log_stft", 0.0) for item in history]
    train_rms = [item["train"].get("rms", 0.0) for item in history]
    train_envelope = [item["train"].get("envelope", 0.0) for item in history]
    train_vq = [item["train"]["vq"] for item in history]
    train_cls_loss = [item["train"]["cls"] for item in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(epochs, train_total, label="train")
    axes[0, 0].plot(epochs, val_total, label="val")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs, train_cls, label="train")
    axes[0, 1].plot(epochs, val_cls, label="val")
    axes[0, 1].set_title("Classification Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(epochs, perplexity, color="tab:green")
    axes[1, 0].set_title("VQ Perplexity")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(epochs, train_l1, label="l1")
    axes[1, 1].plot(epochs, train_stft, label="stft")
    axes[1, 1].plot(epochs, train_log_stft, label="log_stft")
    axes[1, 1].plot(epochs, train_rms, label="rms")
    axes[1, 1].plot(epochs, train_envelope, label="env")
    axes[1, 1].plot(epochs, train_vq, label="vq")
    axes[1, 1].plot(epochs, train_cls_loss, label="cls")
    axes[1, 1].set_title("Train Loss Components")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
