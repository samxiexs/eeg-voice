from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.karaone_v9.data import KaraOneV9TargetBank
from src.karaone_v10.data import KaraOneV10ClusterBank, KaraOneV10ClusteredDataset
from src.karaone_v12.time_anchor import extract_time_anchor
from src.utils import ensure_dir, load_simple_yaml, load_wav_fixed, resolve_bundle_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train-only KaraOne v12 time-anchor cache and leakage audit.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone_v12.yaml"))
    parser.add_argument("--stages", default=None)
    parser.add_argument("--cluster-bank", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--audit-out", default=None)
    parser.add_argument("--alignment-cache", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    stages = tuple(item.strip() for item in str(args.stages or cfg["data"].get("stages", "overt_like")).split(",") if item.strip())
    cache_cfg = cfg.get("cache", {})
    time_cfg = cfg.get("time_anchor", {})
    subject_val = str(cfg["data"].get("subject_val", "P02"))
    subject_test = str(cfg["data"].get("subject_test", "MM21"))
    sample_rate = int(time_cfg.get("sample_rate", cfg.get("synthesis", {}).get("sample_rate", 16000)))
    duration_sec = float(time_cfg.get("duration_sec", cfg.get("synthesis", {}).get("duration_sec", 2.0)))
    n_samples = int(round(sample_rate * duration_sec))
    targets = KaraOneV9TargetBank(
        resolve_bundle_path(cache_cfg["semantic"], BUNDLE_DIR),
        codec_cache=resolve_bundle_path(cache_cfg["codec"], BUNDLE_DIR) if cache_cfg.get("codec") else None,
        prosody_cache=resolve_bundle_path(cache_cfg["prosody"], BUNDLE_DIR) if cache_cfg.get("prosody") else None,
        semantic_token_cache=resolve_bundle_path(cache_cfg["semantic_tokens"], BUNDLE_DIR) if cache_cfg.get("semantic_tokens") else None,
        data_root=root,
    )
    cluster_path = args.cluster_bank or cache_cfg.get("cluster_bank", "")
    cluster_bank = KaraOneV10ClusterBank(resolve_bundle_path(cluster_path, BUNDLE_DIR) if cluster_path else None)
    datasets = [
        KaraOneV10ClusteredDataset(root, targets, split, cluster_bank=cluster_bank, stages=stages, subject_val=subject_val, subject_test=subject_test, eeg_len=int(cfg["data"].get("eeg_len", 1280)))
        for split in ("subject_train", "subject_val", "subject_test")
    ]
    alignment = load_alignment_cache(resolve_bundle_path(args.alignment_cache or time_cfg.get("alignment_cache", ""), BUNDLE_DIR) if (args.alignment_cache or time_cfg.get("alignment_cache")) else None)
    rows = []
    for dataset in datasets:
        for idx in range(len(dataset)):
            rows.append(dataset[idx])
            if args.max_rows and len(rows) >= int(args.max_rows):
                break
        if args.max_rows and len(rows) >= int(args.max_rows):
            break
    if not rows:
        raise RuntimeError("No rows found for v12 time-anchor cache")

    keys, subjects, labels, stages_out, trials, splits, fit_split = [], [], [], [], [], [], []
    onset, duration, center, lag, confidence = [], [], [], [], []
    active_masks, envelopes = [], []
    for item in rows:
        subject = str(item["subject"])
        stage = str(item["stage"])
        trial = int(item["trial_index"])
        split = "subject_val" if subject == subject_val else "subject_test" if subject == subject_test else "subject_train"
        audio_path = resolve_audio_path(root, targets.semantic.audio_path(subject, trial))
        audio = load_wav_fixed(audio_path, sample_rate, n_samples, normalize="none") if audio_path.exists() else np.zeros(n_samples, dtype=np.float32)
        anchor = extract_time_anchor(
            audio,
            sample_rate=sample_rate,
            duration_sec=duration_sec,
            hop_sec=float(time_cfg.get("hop_sec", 0.01)),
            min_active_sec=float(time_cfg.get("min_active_sec", 0.12)),
            merge_gap_sec=float(time_cfg.get("merge_gap_sec", 0.08)),
            threshold_mad=float(time_cfg.get("threshold_mad", 1.5)),
            threshold_peak_ratio=float(time_cfg.get("threshold_peak_ratio", 0.15)),
        )
        lag_value, lag_conf = lookup_lag(alignment, subject=subject, stage=stage, trial=trial)
        keys.append(f"{subject}:{stage}:{trial}")
        subjects.append(subject)
        labels.append(str(item["label"]))
        stages_out.append(stage)
        trials.append(trial)
        splits.append(split)
        fit_split.append(split == "subject_train")
        onset.append(anchor.onset_sec)
        duration.append(anchor.duration_sec)
        center.append(anchor.center_sec)
        lag.append(lag_value)
        confidence.append(max(anchor.confidence, lag_conf))
        active_masks.append(anchor.active_mask)
        envelopes.append(anchor.envelope)

    fit = np.asarray(fit_split, dtype=bool)
    stages_arr = np.asarray(stages_out, dtype=object)
    stage_prior = {}
    for stage in sorted(set(stages_out)):
        mask = fit & (stages_arr == stage)
        values = np.asarray(lag, dtype=np.float32)[mask]
        stage_prior[stage] = float(np.median(values)) if values.size else 0.0
    for idx, is_fit in enumerate(fit):
        if confidence[idx] <= 0.0:
            lag[idx] = stage_prior.get(stages_out[idx], 0.0)

    out = resolve_bundle_path(args.out or cache_cfg.get("v12_time_anchor_bank", "../artifacts/audio_targets/karaone_v12_time_anchors_overt_like.npz"), BUNDLE_DIR)
    ensure_dir(out.parent)
    np.savez_compressed(
        out,
        keys=np.asarray(keys, dtype=object),
        subjects=np.asarray(subjects, dtype=object),
        labels=np.asarray(labels, dtype=object),
        stages=np.asarray(stages_out, dtype=object),
        trials=np.asarray(trials, dtype=np.int64),
        split=np.asarray(splits, dtype=object),
        fit_split=fit,
        onset_sec=np.asarray(onset, dtype=np.float32),
        duration_active_sec=np.asarray(duration, dtype=np.float32),
        center_sec=np.asarray(center, dtype=np.float32),
        lag_sec=np.asarray(lag, dtype=np.float32),
        confidence=np.asarray(confidence, dtype=np.float32),
        active_mask=np.stack(active_masks, axis=0).astype(np.float32),
        envelope=np.stack(envelopes, axis=0).astype(np.float32),
        sample_rate=np.asarray(sample_rate, dtype=np.int64),
        duration_sec=np.asarray(duration_sec, dtype=np.float32),
        stage_lag_prior_keys=np.asarray(list(stage_prior.keys()), dtype=object),
        stage_lag_prior_values=np.asarray(list(stage_prior.values()), dtype=np.float32),
    )
    audit = {
        "audit_kind": "karaone_v12_train_only_time_anchor_bank",
        "status": "pass",
        "time_anchor_bank": str(out),
        "stages": list(stages),
        "n_rows": len(keys),
        "n_train_fit_rows": int(fit.sum()),
        "heldout_subject_used_for_time_prior": {subject_val: False, subject_test: False},
        "heldout_reference_use": "diagnostic_targets_only",
        "sample_rate": sample_rate,
        "duration_sec": duration_sec,
        "active_mask_steps": int(active_masks[0].shape[0]),
        "stage_lag_prior": stage_prior,
        "alignment_cache": str(alignment.get("_path", "")) if alignment else "",
    }
    audit_out = resolve_bundle_path(args.audit_out or cache_cfg.get("v12_time_anchor_audit", "../artifacts/audio_targets/karaone_v12_time_anchor_audit_overt_like.json"), BUNDLE_DIR)
    write_json(audit_out, audit)
    print(json.dumps({"time_anchor_bank": str(out), "audit": str(audit_out), **audit}, ensure_ascii=False, indent=2), flush=True)


def load_alignment_cache(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = np.load(path, allow_pickle=True)
    keys = [str(item) for item in payload["keys"].tolist()]
    out = {"_path": str(path), "by_key": {}}
    lag = payload["lag_sec"].astype(np.float32) if "lag_sec" in payload.files else np.zeros(len(keys), dtype=np.float32)
    conf = payload["lag_confidence"].astype(np.float32) if "lag_confidence" in payload.files else np.ones(len(keys), dtype=np.float32)
    for idx, key in enumerate(keys):
        out["by_key"][key] = (float(lag[idx]), float(conf[idx]))
    return out


def lookup_lag(alignment: dict[str, Any] | None, *, subject: str, stage: str, trial: int) -> tuple[float, float]:
    if not alignment:
        return 0.0, 0.0
    by_key = alignment["by_key"]
    for key in (f"{subject}:{stage}:{trial}", f"{subject}:overt_like:{trial}"):
        if key in by_key:
            return by_key[key]
    return 0.0, 0.0


def resolve_audio_path(root: Path, audio_path: str | Path) -> Path:
    path = Path(audio_path)
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    main()
