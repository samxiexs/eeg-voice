from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def label_candidates(label: str) -> list[str]:
    """Return text candidates for KaraOne's mixed word/phoneme labels."""
    raw = str(label).strip().lower()
    core = raw.strip("/")
    phoneme_aliases = {
        "iy": ["ee", "e", "iy"],
        "uw": ["oo", "you", "u", "uw"],
        "m": ["m", "em"],
        "n": ["n", "en"],
        "piy": ["pee", "pea", "p", "piy"],
        "diy": ["dee", "d", "diy"],
    }
    candidates = phoneme_aliases.get(core, [core])
    return [_normalize_text(item) for item in candidates if _normalize_text(item)]


def _edit_distance(a: list[str] | str, b: list[str] | str) -> int:
    if isinstance(a, str):
        a = list(a)
    if isinstance(b, str):
        b = list(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + int(ca != cb)))
        prev = cur
    return prev[-1]


def _word_error_rate(pred: str, target: str) -> float:
    pred_words = pred.split()
    target_words = target.split()
    return float(_edit_distance(pred_words, target_words) / max(len(target_words), 1))


def _char_error_rate(pred: str, target: str) -> float:
    return float(_edit_distance(pred, target) / max(len(target), 1))


def asr_label_metrics(transcript: str, label: str) -> dict[str, Any]:
    transcript_norm = _normalize_text(transcript)
    candidates = label_candidates(label)
    if not candidates:
        return {
            "text": transcript_norm,
            "label_hit": False,
            "cer": None,
            "wer": None,
            "candidate": "",
        }
    hit = any(candidate in transcript_norm.split() or candidate == transcript_norm for candidate in candidates)
    cer_scores = [_char_error_rate(transcript_norm, candidate) for candidate in candidates]
    wer_scores = [_word_error_rate(transcript_norm, candidate) for candidate in candidates]
    best_idx = int(min(range(len(candidates)), key=lambda idx: cer_scores[idx]))
    return {
        "text": transcript_norm,
        "label_hit": bool(hit),
        "cer": float(cer_scores[best_idx]),
        "wer": float(wer_scores[best_idx]),
        "candidate": candidates[best_idx],
    }


def load_whisper_asr(
    model_name: str | None,
    device: str,
    allow_download: bool = False,
    download_root: str | None = None,
):
    """Load Whisper only when explicitly requested.

    Named Whisper models can trigger network downloads. By default we only load an
    existing local model path or an already cached checkpoint; pass
    --asr-allow-download for first-time downloads.
    """
    if not model_name:
        return None, {"enabled": False, "reason": "no asr model requested"}
    try:
        import whisper  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return None, {"enabled": False, "reason": f"whisper import failed: {exc}"}

    candidate = Path(model_name).expanduser()
    kwargs: dict[str, Any] = {"device": device}
    if download_root:
        kwargs["download_root"] = str(Path(download_root).expanduser())
    if candidate.exists():
        return whisper.load_model(str(candidate), **kwargs), {"enabled": True, "model": str(candidate)}
    if allow_download:
        return whisper.load_model(str(model_name), **kwargs), {"enabled": True, "model": str(model_name)}

    root = Path(download_root).expanduser() if download_root else Path.home() / ".cache" / "whisper"
    cached = root / f"{model_name}.pt"
    if cached.exists():
        return whisper.load_model(str(model_name), **kwargs), {"enabled": True, "model": str(model_name), "cache": str(cached)}
    return None, {
        "enabled": False,
        "reason": f"Whisper model {model_name!r} is not cached at {cached}; use --asr-allow-download or pass a local .pt path",
    }


def transcribe_label_metrics(model, wav_path: Path, label: str, fp16: bool = False) -> dict[str, Any]:
    result = model.transcribe(str(wav_path), language="en", fp16=bool(fp16))
    return asr_label_metrics(str(result.get("text", "")), label)
