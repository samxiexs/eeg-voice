from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path


BANDS = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 55.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v7 KaraOne EEG cross-subject feature cache.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--out", default="../artifacts/audio_targets/karaone_eeg_features_v7.npz")
    parser.add_argument("--stages", default="clearing,stimulus_like,thinking,overt_like")
    parser.add_argument("--fs", type=float, default=256.0)
    parser.add_argument("--envelope-steps", type=int, default=64)
    return parser.parse_args()


def _read_segments(root: Path, stages: set[str]) -> list[dict]:
    out: list[dict] = []
    with (root / "segments.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            stage = str(row["segment_stage"])
            if stage not in stages:
                continue
            out.append(
                {
                    "subject": str(row["subject_id"]),
                    "label": str(row["label"]),
                    "stage": stage,
                    "trial_index": int(row["trial_index"]),
                }
            )
    out.sort(key=lambda r: (r["subject"], r["trial_index"], r["stage"]))
    return out


def _robust_zscore(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=1, keepdims=True)
    q25 = np.percentile(x, 25, axis=1, keepdims=True)
    q75 = np.percentile(x, 75, axis=1, keepdims=True)
    scale = np.maximum((q75 - q25) / 1.349, 1e-4)
    return ((x - med) / scale).astype(np.float32)


def _resample_1d(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros(int(n), dtype=np.float32)
    if values.size == int(n):
        return values.astype(np.float32)
    src = np.linspace(0.0, 1.0, values.size)
    dst = np.linspace(0.0, 1.0, int(n))
    return np.interp(dst, src, values).astype(np.float32)


def _bandpower(z: np.ndarray, fs: float) -> np.ndarray:
    n = int(z.shape[1])
    if n < 8:
        return np.zeros((z.shape[0], len(BANDS)), dtype=np.float32)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))
    spec = np.abs(np.fft.rfft(z, axis=1)) ** 2
    feats = []
    for _, lo, hi in BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        if not bool(mask.any()):
            feats.append(np.zeros(z.shape[0], dtype=np.float32))
        else:
            feats.append(np.log(spec[:, mask].mean(axis=1) + 1e-8).astype(np.float32))
    return np.stack(feats, axis=1).astype(np.float32)


def _cov_upper(z: np.ndarray) -> np.ndarray:
    if z.shape[1] < 2:
        cov = np.eye(z.shape[0], dtype=np.float32)
    else:
        cov = (z @ z.T) / float(max(z.shape[1] - 1, 1))
        trace = float(np.trace(cov))
        cov = cov / max(trace / max(cov.shape[0], 1), 1e-6)
    idx = np.triu_indices(z.shape[0])
    return cov[idx].astype(np.float32)


def _feature_vector(arr: np.ndarray, valid_len: int, fs: float, envelope_steps: int) -> tuple[np.ndarray, np.ndarray]:
    valid = arr[:, : max(2, min(int(valid_len), arr.shape[1]))].astype(np.float32)
    z = _robust_zscore(valid)
    logvar = np.log(np.var(z, axis=1) + 1e-6).astype(np.float32)
    bp = _bandpower(z, fs).reshape(-1)
    cov = _cov_upper(z)
    # A subject-robust low-frequency activity envelope; this is not an acoustic
    # alignment target, just a compact temporal EEG descriptor.
    env = np.sqrt(np.mean(np.square(z), axis=0) + 1e-8)
    env = _resample_1d(env, int(envelope_steps))
    env = (env - float(np.median(env))) / max(float(np.std(env)), 1e-6)
    feat = np.concatenate([logvar, bp, cov], axis=0).astype(np.float32)
    return feat, env.astype(np.float32)


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    stages = {item.strip() for item in str(args.stages).split(",") if item.strip()}
    rows = _read_segments(root, stages)
    heldout = {str(item) for item in cfg["data"].get("heldout_subjects", ["P02", "MM21"])}

    bundle_cache: dict[str, dict] = {}

    def load_subject(subject: str) -> dict:
        if subject not in bundle_cache:
            path = root / "subjects" / f"{subject}.npz"
            payload = np.load(path, allow_pickle=True)
            trial_indices = payload["trial_indices"].astype(np.int32)
            stage_arrays: dict[str, np.ndarray] = {}
            stage_valid: dict[str, np.ndarray] = {}
            for stage in stages:
                key = f"stage__{stage}"
                if key not in payload.files:
                    continue
                arr = payload[key].astype(np.float32)
                valid_key = f"{key}__valid_lengths"
                valid = (
                    payload[valid_key].astype(np.int32)
                    if valid_key in payload.files
                    else np.full(arr.shape[0], arr.shape[-1], dtype=np.int32)
                )
                stage_arrays[stage] = arr
                stage_valid[stage] = valid
            bundle_cache[subject] = {
                "trial_to_pos": {int(t): i for i, t in enumerate(trial_indices.tolist())},
                "stages": stage_arrays,
                "valid": stage_valid,
            }
        return bundle_cache[subject]

    template_ids: list[str] = []
    subject_ids: list[str] = []
    labels: list[str] = []
    stage_values: list[str] = []
    trial_indices: list[int] = []
    valid_lengths: list[int] = []
    feats: list[np.ndarray] = []
    envs: list[np.ndarray] = []

    for row in rows:
        subject, stage, trial = row["subject"], row["stage"], int(row["trial_index"])
        bundle = load_subject(subject)
        pos = bundle["trial_to_pos"].get(trial)
        if pos is None:
            continue
        if stage not in bundle["stages"]:
            continue
        arr = bundle["stages"][stage][pos].astype(np.float32)
        valid_len = int(bundle["valid"][stage][pos])
        feat, env = _feature_vector(arr, valid_len, float(args.fs), int(args.envelope_steps))
        template_ids.append(f"{subject}:{trial}")
        subject_ids.append(subject)
        labels.append(str(row["label"]))
        stage_values.append(stage)
        trial_indices.append(trial)
        valid_lengths.append(valid_len)
        feats.append(feat)
        envs.append(env)

    if not feats:
        raise ValueError("No EEG features were extracted")
    raw_features = np.stack(feats, axis=0).astype(np.float32)
    train_mask = np.asarray([subject not in heldout for subject in subject_ids], dtype=bool)
    stat_source = raw_features[train_mask] if bool(train_mask.any()) else raw_features
    feature_mean = np.median(stat_source, axis=0).astype(np.float32)
    q25 = np.percentile(stat_source, 25, axis=0)
    q75 = np.percentile(stat_source, 75, axis=0)
    feature_std = np.maximum(((q75 - q25) / 1.349).astype(np.float32), 1e-4)
    features = ((raw_features - feature_mean.reshape(1, -1)) / feature_std.reshape(1, -1)).astype(np.float32)
    features = np.clip(features, -8.0, 8.0)
    n_ch = int((np.sqrt(8 * (_cov_upper(np.zeros((62, 2), dtype=np.float32)).shape[0]) + 1) - 1) / 2)
    del n_ch
    feature_names = (
        [f"logvar_ch{c:02d}" for c in range(62)]
        + [f"bp_{band}_ch{c:02d}" for c in range(62) for band, _, _ in BANDS]
        + [f"cov_u{i:04d}" for i in range(features.shape[1] - 62 - 62 * len(BANDS))]
    )
    out = resolve_bundle_path(args.out, BUNDLE_DIR)
    ensure_dir(out.parent)
    np.savez_compressed(
        out,
        feature_kind=np.asarray("karaone_eeg_features_v7"),
        template_ids=np.asarray(template_ids).astype(str),
        subject_ids=np.asarray(subject_ids).astype(str),
        labels=np.asarray(labels).astype(str),
        stages=np.asarray(stage_values).astype(str),
        trial_indices=np.asarray(trial_indices, dtype=np.int32),
        valid_lengths=np.asarray(valid_lengths, dtype=np.int32),
        feature_vectors=features,
        raw_feature_vectors=raw_features,
        envelopes=np.stack(envs, axis=0).astype(np.float32),
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_names=np.asarray(feature_names).astype(str),
        fs=np.asarray(float(args.fs), dtype=np.float32),
        envelope_steps=np.asarray(int(args.envelope_steps), dtype=np.int32),
        train_stat_excludes=np.asarray(sorted(heldout)).astype(str),
    )
    summary = {
        "out": str(out),
        "n": len(template_ids),
        "feature_dim": int(features.shape[1]),
        "envelope_steps": int(args.envelope_steps),
        "stages": sorted(stages),
        "train_stat_excludes": sorted(heldout),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
