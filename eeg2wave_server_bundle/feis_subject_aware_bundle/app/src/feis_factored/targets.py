"""Factored targets from the FEIS EnCodec-latent cache (no transformers needed).

From the 336-cell cache (subject:label -> EnCodec latent), derive:
  - normalised per-cell target latent sequence  [T, D]
  - per-label CONTENT prototype  (mean over the 21 speakers; speaker-independent)
  - per-subject SPEAKER prototype (mean over the 16 labels; content-independent)
  - coarse phonological category maps (manner / voicing / vowel-consonant)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# FEIS 16 prompts -> coarse phonological categories.
MANNER = {
    "p": "plosive", "t": "plosive", "k": "plosive",
    "f": "fricative", "s": "fricative", "sh": "fricative",
    "v": "fricative", "z": "fricative", "zh": "fricative",
    "m": "nasal", "n": "nasal", "ng": "nasal",
    "fleece": "vowel", "goose": "vowel", "trap": "vowel", "thought": "vowel",
}
VOICING = {  # voiceless vs voiced (vowels are voiced)
    "p": "voiceless", "t": "voiceless", "k": "voiceless",
    "f": "voiceless", "s": "voiceless", "sh": "voiceless",
    "v": "voiced", "z": "voiced", "zh": "voiced",
    "m": "voiced", "n": "voiced", "ng": "voiced",
    "fleece": "voiced", "goose": "voiced", "trap": "voiced", "thought": "voiced",
}
VOWEL_CONSONANT = {l: ("vowel" if MANNER[l] == "vowel" else "consonant") for l in MANNER}


class FactoredTargets:
    def __init__(self, cache_path: str | Path):
        payload = np.load(Path(cache_path), allow_pickle=True)
        self.template_ids = payload["template_ids"].astype(str)
        self.subject_ids = payload["subject_ids"].astype(str)
        self.labels = payload["labels"].astype(str)
        raw_seq = payload["target_sequences"].astype(np.float32)          # [N, T, D]
        self.T, self.D = int(raw_seq.shape[1]), int(raw_seq.shape[2])

        if "target_mean" in payload.files:
            mean = payload["target_mean"].astype(np.float32)
            std = np.maximum(payload["target_std"].astype(np.float32), 1e-6)
        else:
            mean = raw_seq.reshape(-1, self.D).mean(0).astype(np.float32)
            std = np.maximum(raw_seq.reshape(-1, self.D).std(0), 1e-6).astype(np.float32)
        self.target_mean, self.target_std = mean, std
        self.seq = ((raw_seq - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)
        self.raw_seq = raw_seq
        self.summary = self.seq.mean(axis=1).astype(np.float32)           # [N, D]
        # global RAW mean latent sequence -> decode = "mean-latent" collapse reference
        self.global_mean_raw = raw_seq.mean(axis=0).astype(np.float32)    # [T, D]

        # --- energy / scale / audio-path fields (codec QC + scaled synthesis) ---
        n = raw_seq.shape[0]
        self.target_rms = (payload["target_rms"].astype(np.float32)
                           if "target_rms" in payload.files else np.full(n, 0.08, np.float32))
        if "target_log_rms" in payload.files:
            self.target_log_rms = payload["target_log_rms"].astype(np.float32)
        else:
            self.target_log_rms = np.log(np.maximum(self.target_rms, 1e-8)).astype(np.float32)
        if "decoder_scales" in payload.files:
            self.decoder_scales = payload["decoder_scales"].astype(np.float32)   # [N, S]
        else:
            self.decoder_scales = np.ones((n, 1), np.float32)
        if "default_decoder_scales" in payload.files:
            self.default_decoder_scales = payload["default_decoder_scales"].astype(np.float32)
        else:
            self.default_decoder_scales = self.decoder_scales.mean(0).astype(np.float32)
        self.audio_paths = (payload["audio_paths"].astype(str)
                            if "audio_paths" in payload.files else np.array([""] * n))

        self.cell_to_idx = {t: i for i, t in enumerate(self.template_ids.tolist())}
        self.label_vocab = sorted(set(self.labels.tolist()))
        self.subject_vocab = sorted(set(self.subject_ids.tolist()))
        self.label_to_id = {l: i for i, l in enumerate(self.label_vocab)}
        self.subject_to_id = {s: i for i, s in enumerate(self.subject_vocab)}

        # CONTENT prototype: mean normalised summary over speakers, per label.
        self.content_proto = np.stack(
            [self.summary[[i for i, l in enumerate(self.labels) if l == lab]].mean(0)
             for lab in self.label_vocab], 0).astype(np.float32)          # [n_label, D]
        # SPEAKER prototype: mean normalised summary over labels, per subject.
        self.speaker_proto = np.stack(
            [self.summary[[i for i, s in enumerate(self.subject_ids) if s == sub]].mean(0)
             for sub in self.subject_vocab], 0).astype(np.float32)        # [n_subject, D]

        # coarse category ids
        self.manner_vocab = sorted(set(MANNER.values()))
        self.voicing_vocab = sorted(set(VOICING.values()))
        self.vc_vocab = sorted(set(VOWEL_CONSONANT.values()))

    # --- accessors ---
    def has_cell(self, subject: str, label: str) -> bool:
        return f"{subject}:{label}" in self.cell_to_idx

    def cell_target(self, subject: str, label: str) -> np.ndarray:
        return self.seq[self.cell_to_idx[f"{subject}:{label}"]]            # [T, D] normalised

    def cell_raw_target(self, subject: str, label: str) -> np.ndarray:
        return self.raw_seq[self.cell_to_idx[f"{subject}:{label}"]]        # [T, D] raw

    def content_prototype(self, label: str) -> np.ndarray:
        return self.content_proto[self.label_to_id[label]]

    def speaker_prototype(self, subject: str) -> np.ndarray:
        """Audio-derived voice prototype for a subject (mean latent over their labels)."""
        return self.speaker_proto[self.subject_to_id[subject]]

    def coarse_ids(self, label: str) -> dict[str, int]:
        return {
            "manner": self.manner_vocab.index(MANNER[label]),
            "voicing": self.voicing_vocab.index(VOICING[label]),
            "vc": self.vc_vocab.index(VOWEL_CONSONANT[label]),
        }

    # --- energy / scale / audio accessors (v2: codec QC + scaled synthesis) ---
    def cell_log_rms(self, subject: str, label: str) -> float:
        return float(self.target_log_rms[self.cell_to_idx[f"{subject}:{label}"]])

    def cell_rms(self, subject: str, label: str) -> float:
        return float(self.target_rms[self.cell_to_idx[f"{subject}:{label}"]])

    def cell_decoder_scale(self, subject: str, label: str) -> np.ndarray:
        return self.decoder_scales[self.cell_to_idx[f"{subject}:{label}"]].astype(np.float32)

    def cell_audio_path(self, subject: str, label: str) -> str:
        return str(self.audio_paths[self.cell_to_idx[f"{subject}:{label}"]])

    def global_mean_raw_seq(self) -> np.ndarray:
        """Raw global mean latent [T, D]; decode -> the mean-collapse reference wav."""
        return self.global_mean_raw
