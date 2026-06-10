from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

LABEL_PHONEMES: dict[str, tuple[str, ...]] = {
    "f": ("F",),
    "fleece": ("F", "L", "IY", "S"),
    "goose": ("G", "UW", "S"),
    "k": ("K",),
    "m": ("M",),
    "n": ("N",),
    "ng": ("NG",),
    "p": ("P",),
    "s": ("S",),
    "sh": ("SH",),
    "t": ("T",),
    "thought": ("TH", "AO", "T"),
    "trap": ("T", "R", "AE", "P"),
    "v": ("V",),
    "z": ("Z",),
    "zh": ("ZH",),
}


@dataclass(frozen=True)
class PhonemeVocab:
    tokens: tuple[str, ...]
    token_to_id: dict[str, int]
    max_steps: int

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def size(self) -> int:
        return len(self.tokens)


def build_phoneme_vocab(labels: list[str] | tuple[str, ...] | None = None) -> PhonemeVocab:
    selected_labels = sorted(LABEL_PHONEMES) if labels is None else sorted({str(item) for item in labels})
    phoneme_tokens: set[str] = set()
    max_steps = 1
    for label in selected_labels:
        sequence = LABEL_PHONEMES.get(label, tuple(str(label).upper()))
        phoneme_tokens.update(sequence)
        max_steps = max(max_steps, len(sequence))
    tokens = (PAD_TOKEN, UNK_TOKEN, *tuple(sorted(phoneme_tokens)))
    return PhonemeVocab(
        tokens=tokens,
        token_to_id={token: idx for idx, token in enumerate(tokens)},
        max_steps=max_steps,
    )


def encode_label_phonemes(label: str, vocab: PhonemeVocab) -> tuple[np.ndarray, np.ndarray]:
    sequence = LABEL_PHONEMES.get(str(label), tuple(str(label).upper()))
    ids = np.full((vocab.max_steps,), vocab.pad_id, dtype=np.int64)
    mask = np.zeros((vocab.max_steps,), dtype=np.float32)
    for idx, token in enumerate(sequence[: vocab.max_steps]):
        ids[idx] = vocab.token_to_id.get(token, vocab.unk_id)
        mask[idx] = 1.0
    return ids, mask
