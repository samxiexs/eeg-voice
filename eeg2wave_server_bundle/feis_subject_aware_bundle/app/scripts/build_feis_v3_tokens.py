"""Build train-only FEIS v3 subject-label audio token caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.feis_v3.data import CLEAN_FIELD, SUBJECT_FIELD, as_bool, assign_to_centers, norm_subject, parse_stage_spec, stable_kmeans
from src.utils import ensure_dir, load_simple_yaml, load_wav_fixed, read_csv_rows, resolve_bundle_path, resolve_feis_root, write_json


def parse_args():
    p = argparse.ArgumentParser(description="Build FEIS v3 tokenized audio cache.")
    p.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "feis_v3_tokenized_generation.yaml"))
    p.add_argument("--stage", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _format_path(raw: str, stage: str) -> str:
    return str(raw).replace("{stage}", stage)


def _audio_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _frame_bounds(n: int, steps: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n, int(steps) + 1).round().astype(int)
    return [(int(edges[i]), max(int(edges[i + 1]), int(edges[i]) + 1)) for i in range(int(steps))]


def _semantic_features(audio: np.ndarray, steps: int, dim: int) -> np.ndarray:
    feats = []
    for lo, hi in _frame_bounds(audio.shape[0], steps):
        frame = audio[lo:hi].astype(np.float32)
        if frame.size < 8:
            frame = np.pad(frame, (0, 8 - frame.size))
        win = np.hanning(frame.size).astype(np.float32)
        mag = np.log1p(np.abs(np.fft.rfft(frame * win))).astype(np.float32)
        xp = np.linspace(0, mag.size - 1, int(dim))
        feats.append(np.interp(xp, np.arange(mag.size), mag).astype(np.float32))
    arr = np.stack(feats, axis=0)
    arr = (arr - arr.mean(axis=0, keepdims=True)) / np.maximum(arr.std(axis=0, keepdims=True), 1e-5)
    return arr.astype(np.float32)


def _codec_chunks_and_features(audio: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    chunks = []
    feats = []
    for lo, hi in _frame_bounds(audio.shape[0], steps):
        chunk = audio[lo:hi].astype(np.float32)
        target = int(np.ceil(audio.shape[0] / steps))
        if chunk.size < target:
            chunk = np.pad(chunk, (0, target - chunk.size)).astype(np.float32)
        elif chunk.size > target:
            chunk = chunk[:target]
        chunks.append(chunk)
        rms = float(np.sqrt(np.mean(chunk**2) + 1e-8))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(chunk).astype(np.float32))))) if chunk.size > 1 else 0.0
        mag = np.log1p(np.abs(np.fft.rfft(chunk * np.hanning(chunk.size)))).astype(np.float32)
        low = np.interp(np.linspace(0, max(mag.size - 1, 1), 5), np.arange(mag.size), mag).astype(np.float32)
        feats.append(
            np.asarray(
                [
                    float(chunk.mean()),
                    float(chunk.std()),
                    rms,
                    float(chunk.max(initial=0.0)),
                    float(chunk.min(initial=0.0)),
                    float(np.mean(np.abs(chunk))),
                    zcr,
                    *low.tolist(),
                ],
                dtype=np.float32,
            )
        )
    return np.stack(chunks, axis=0).astype(np.float32), np.stack(feats, axis=0).astype(np.float32)


def _prosody(audio: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    energies = []
    for lo, hi in _frame_bounds(audio.shape[0], steps):
        frame = audio[lo:hi].astype(np.float32)
        energies.append(float(np.sqrt(np.mean(frame**2) + 1e-8)))
    energy = np.asarray(energies, dtype=np.float32)
    thresh = max(float(np.percentile(energy, 60)), float(energy.max() * 0.15))
    active = (energy >= thresh).astype(np.float32)
    onset = np.zeros_like(active)
    transitions = np.flatnonzero((active > 0.5) & (np.pad(active[:-1], (1, 0)) < 0.5))
    if transitions.size:
        onset[transitions[0]] = 1.0
    duration = np.full_like(active, float(active.mean()))
    norm_energy = (np.log1p(energy) - np.log1p(energy).mean()) / max(float(np.log1p(energy).std()), 1e-5)
    return active, duration.astype(np.float32), norm_energy.astype(np.float32), onset


def _variant_summary(semantic_ids: np.ndarray, codec_ids: np.ndarray, active: np.ndarray, energy: np.ndarray, semantic_vocab: int, codec_vocab: int) -> np.ndarray:
    sem_hist = np.bincount(semantic_ids.reshape(-1), minlength=semantic_vocab).astype(np.float32)
    sem_hist /= max(float(sem_hist.sum()), 1.0)
    codec_hist = np.bincount(codec_ids.reshape(-1), minlength=codec_vocab).astype(np.float32)
    codec_hist /= max(float(codec_hist.sum()), 1.0)
    pros = np.asarray([active.mean(), active.std(), energy.mean(), energy.std(), energy.max(initial=0.0)], dtype=np.float32)
    return np.concatenate([sem_hist, codec_hist, pros]).astype(np.float32)


def _collect_audio_rows(root: Path, stage: str, include_anomalous: bool, subject_val: str, subject_test: str, negative_stage: str) -> list[dict]:
    stages = set(parse_stage_spec(stage, negative_stage=negative_stage))
    by_key: dict[str, dict] = {}
    for row in read_csv_rows(root / "segments.csv"):
        if str(row.get("segment_stage", "")) not in stages:
            continue
        subject_id = norm_subject(row[SUBJECT_FIELD])
        if not include_anomalous and not as_bool(row.get(CLEAN_FIELD), subject_id != "05"):
            continue
        label = str(row["label"])
        key = f"{subject_id}:{label}"
        if key not in by_key:
            if subject_id == subject_val:
                fit_split = "subject_val"
            elif subject_id == subject_test:
                fit_split = "subject_test"
            else:
                fit_split = "train"
            by_key[key] = {
                "audio_key": key,
                "subject_id": subject_id,
                "label": label,
                "audio_path": str(row["audio_path"]),
                "audio_sha1": str(row.get("audio_sha1") or ""),
                "fit_split": fit_split,
            }
    return [by_key[key] for key in sorted(by_key)]


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    stage = args.stage or str(cfg.get("data", {}).get("stage", "stimuli"))
    token_cfg = cfg.get("tokens", {})
    data_cfg = cfg.get("data", {})
    audio_cfg = cfg.get("audio", {})
    root = resolve_feis_root(resolve_bundle_path(data_cfg.get("root", "../data/feis"), BUNDLE_DIR))
    subject_val = norm_subject(data_cfg.get("subject_val", "20"))
    subject_test = norm_subject(data_cfg.get("subject_test", "21"))
    negative_stage = str(data_cfg.get("negative_stage", "resting"))
    include_anomalous = bool(data_cfg.get("include_anomalous", False))
    out_path = resolve_bundle_path(
        args.out or _format_path(token_cfg.get("cache", "../artifacts/audio_targets/feis_v3_tokens_{stage}.npz"), stage),
        BUNDLE_DIR,
    )
    audit_path = resolve_bundle_path(
        _format_path(token_cfg.get("audit", "../artifacts/audio_targets/feis_v3_token_audit_{stage}.json"), stage),
        BUNDLE_DIR,
    )
    sr = int(audio_cfg.get("sample_rate", 16000))
    n_samples = int(round(sr * float(audio_cfg.get("duration_sec", 1.0))))
    semantic_steps = int(token_cfg.get("semantic_steps", 64))
    semantic_dim = int(token_cfg.get("semantic_dim", 32))
    codec_steps = int(token_cfg.get("codec_steps", 64))
    semantic_vocab = int(token_cfg.get("semantic_vocab", 64))
    codec_vocab = int(token_cfg.get("codec_vocab", 128))
    variant_clusters = int(token_cfg.get("audio_variant_clusters", 32))
    seed = int(cfg.get("train", {}).get("seed", 7))
    kmeans_iters = int(token_cfg.get("kmeans_iters", 10))

    rows = _collect_audio_rows(root, stage, include_anomalous, subject_val, subject_test, negative_stage)
    if not rows:
        raise ValueError(f"No FEIS v3 audio rows for stage={stage}")
    semantic_hidden, codec_latent, codec_chunks = [], [], []
    active, duration, energy, onset = [], [], [], []
    for row in rows:
        wav_path = root / row["audio_path"]
        wav = load_wav_fixed(
            wav_path,
            sample_rate=sr,
            n_samples=n_samples,
            normalize=str(audio_cfg.get("normalize", "rms")),
            target_rms=float(audio_cfg.get("target_rms", 0.08)),
            max_gain=float(audio_cfg.get("max_gain", 12.0)),
        )
        row["audio_sha1"] = row["audio_sha1"] or _audio_sha1(wav_path)
        semantic_hidden.append(_semantic_features(wav, semantic_steps, semantic_dim))
        chunks, features = _codec_chunks_and_features(wav, codec_steps)
        codec_chunks.append(chunks)
        codec_latent.append(features)
        p_active, p_duration, p_energy, p_onset = _prosody(wav, semantic_steps)
        active.append(p_active)
        duration.append(p_duration)
        energy.append(p_energy)
        onset.append(p_onset)

    semantic_hidden_arr = np.stack(semantic_hidden).astype(np.float32)
    codec_latent_arr = np.stack(codec_latent).astype(np.float32)
    codec_chunks_arr = np.stack(codec_chunks).astype(np.float32)
    active_arr = np.stack(active).astype(np.float32)
    duration_arr = np.stack(duration).astype(np.float32)
    energy_arr = np.stack(energy).astype(np.float32)
    onset_arr = np.stack(onset).astype(np.float32)
    train_mask = np.asarray([row["fit_split"] == "train" for row in rows], dtype=bool)
    if not train_mask.any():
        raise ValueError("No train audio variants available for train-only token fitting")

    sem_train = semantic_hidden_arr[train_mask].reshape(-1, semantic_dim)
    semantic_codebook, _ = stable_kmeans(
        sem_train,
        semantic_vocab,
        seed=seed,
        iters=kmeans_iters,
        max_fit_rows=int(token_cfg.get("max_fit_frames", 20000)),
    )
    semantic_ids = assign_to_centers(semantic_hidden_arr.reshape(-1, semantic_dim), semantic_codebook).reshape(len(rows), semantic_steps)
    semantic_mask = np.ones_like(semantic_ids, dtype=np.float32)

    codec_feat_dim = codec_latent_arr.shape[-1]
    codec_train = codec_latent_arr[train_mask].reshape(-1, codec_feat_dim)
    codec_codebook, train_assign = stable_kmeans(
        codec_train,
        codec_vocab,
        seed=seed + 13,
        iters=kmeans_iters,
        max_fit_rows=int(token_cfg.get("max_fit_chunks", 20000)),
    )
    codec_ids = assign_to_centers(codec_latent_arr.reshape(-1, codec_feat_dim), codec_codebook).reshape(len(rows), codec_steps)
    train_chunks = codec_chunks_arr[train_mask].reshape(-1, codec_chunks_arr.shape[-1])
    codec_wave = np.zeros((codec_codebook.shape[0], codec_chunks_arr.shape[-1]), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for code in range(codec_codebook.shape[0]):
        mask = train_assign == code
        if mask.any():
            codec_wave[code] = train_chunks[mask].mean(axis=0)
        else:
            codec_wave[code] = train_chunks[rng.integers(0, train_chunks.shape[0])]
    codec_mask = np.ones_like(codec_ids, dtype=np.float32)

    summaries = np.stack(
        [
            _variant_summary(semantic_ids[i], codec_ids[i], active_arr[i], energy_arr[i], semantic_codebook.shape[0], codec_codebook.shape[0])
            for i in range(len(rows))
        ],
        axis=0,
    )
    variant_centers, _ = stable_kmeans(
        summaries[train_mask],
        min(variant_clusters, int(train_mask.sum())),
        seed=seed + 29,
        iters=kmeans_iters,
        max_fit_rows=None,
    )
    variant_ids = assign_to_centers(summaries, variant_centers)

    label_vocab = sorted({row["label"] for row in rows})
    subject_vocab = sorted({row["subject_id"] for row in rows})
    ensure_dir(out_path.parent)
    np.savez_compressed(
        out_path,
        audio_keys=np.asarray([row["audio_key"] for row in rows]),
        subject_ids=np.asarray([row["subject_id"] for row in rows]),
        labels=np.asarray([row["label"] for row in rows]),
        audio_paths=np.asarray([row["audio_path"] for row in rows]),
        audio_sha1=np.asarray([row["audio_sha1"] for row in rows]),
        fit_split=np.asarray([row["fit_split"] for row in rows]),
        label_vocab=np.asarray(label_vocab),
        subject_vocab=np.asarray(subject_vocab),
        semantic_hidden=semantic_hidden_arr,
        semantic_token_ids=semantic_ids.astype(np.int64),
        semantic_token_mask=semantic_mask,
        prosody_active=active_arr,
        prosody_duration=duration_arr,
        prosody_energy=energy_arr,
        prosody_onset=onset_arr,
        codec_latent=codec_latent_arr,
        codec_token_ids=codec_ids.astype(np.int64),
        codec_token_mask=codec_mask,
        audio_variant_cluster_id=variant_ids.astype(np.int64),
        semantic_codebook=semantic_codebook,
        codec_feature_codebook=codec_codebook,
        codec_codebook_waveform=codec_wave,
        audio_variant_cluster_centers=variant_centers,
        sample_rate=np.asarray(sr, dtype=np.int32),
        codec_chunk_samples=np.asarray(codec_chunks_arr.shape[-1], dtype=np.int32),
        stage=np.asarray(stage),
        generated_artifact=np.asarray("generated_codec"),
        retrieval_name=np.asarray("retrieval_diagnostic"),
    )
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["fit_split"]].append(row["subject_id"])
    audit = {
        "stage": stage,
        "token_cache": str(out_path),
        "n_audio_variants": len(rows),
        "semantic_vocab": int(semantic_codebook.shape[0]),
        "codec_vocab": int(codec_codebook.shape[0]),
        "audio_variant_clusters": int(variant_centers.shape[0]),
        "subject_val": subject_val,
        "subject_test": subject_test,
        "train_fit_subjects": sorted(set(by_split["train"])),
        "heldout_subjects": [subject_val, subject_test],
        "heldout_subjects_used_for_fit": False,
        "train_only_fit": True,
        "fit_counts": {key: len(vals) for key, vals in by_split.items()},
        "audio_variants_per_label": {label: int(sum(row["label"] == label for row in rows)) for label in label_vocab},
        "generated_codec_is_generation_artifact": True,
        "retrieval_is_diagnostic_only": True,
        "negative_stage": negative_stage,
    }
    write_json(audit_path, audit)
    print(json.dumps({"token_cache": str(out_path), "audit": str(audit_path), "n_audio_variants": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
