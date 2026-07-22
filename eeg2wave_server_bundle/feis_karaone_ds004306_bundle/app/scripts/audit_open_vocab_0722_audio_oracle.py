#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import resample_poly
from tqdm import tqdm


APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
KARA_APP = APP.parents[1] / "karaone_overt_recon_bundle/app"
if str(KARA_APP) not in sys.path:
    sys.path.insert(0, str(KARA_APP))

from src.open_vocab_0722.audio_gate import AUDIO_FREEZE_SCHEMA, AUDIO_ORACLE_GATE_SCHEMA, require_frozen_audio_checkpoint  # noqa: E402
from src.open_vocab_0722.audio_io import read_wav, write_wav  # noqa: E402
from src.open_vocab_0722.data import LabelFreeAudioDataset, TeacherBank, load_context, resolve_config_path  # noqa: E402
from src.open_vocab_0722.lineage import build_lineage, file_sha256, validate_checkpoint  # noqa: E402
from src.open_vocab_0722.metrics import reconstruction_metrics, summarize  # noqa: E402
from src.open_vocab_0722.model import LabelFreeAudioConfig, LabelFreeAudioModel  # noqa: E402
from src.karaone_0715.codec import DiscreteEncodec, DiscreteEncodecConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode validation audio from its frozen HuBERT condition and gate the project-only audio prior"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=-1, help="Diagnostic only; a limited audit cannot freeze a checkpoint")
    parser.add_argument("--write-examples-per-dataset", type=int, default=None)
    parser.add_argument("--freeze-on-pass", action="store_true")
    parser.add_argument("--verify-frozen", action="store_true", help="Verify hashes/lineage without decoding audio")
    parser.add_argument("--project-audio-only", action="store_true")
    return parser.parse_args()


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def audio_config(cfg: dict[str, Any]) -> LabelFreeAudioConfig:
    value, codec = cfg["audio_model"], cfg["codec"]
    return LabelFreeAudioConfig(
        codebooks=int(codec["codebooks"]), code_steps=int(codec["code_steps"]),
        vocab_size=int(codec["vocab_size"]), d_model=int(value["d_model"]),
        condition_steps=int(value["condition_steps"]), encoder_layers=int(value["encoder_layers"]),
        decoder_layers=int(value["decoder_layers"]), heads=int(value["heads"]),
        dropout=float(value["dropout"]), text_dimension=int(cfg["teachers"]["text_dimension"]),
        xlsr_dimension=int(value["xlsr_dimension"]),
    )


def resample(audio: np.ndarray, source: int, target: int) -> np.ndarray:
    if source == target:
        return np.asarray(audio, dtype=np.float32)
    divisor = math.gcd(source, target)
    return resample_poly(audio, target // divisor, source // divisor).astype(np.float32)


def reference_waveform(context: Any, row: dict[str, str], rate: int, length: int) -> tuple[np.ndarray, int]:
    audio, source_rate = read_wav(context.audio_root / row["audio_relpath"])
    audio = resample(audio, int(source_rate), rate)
    output = np.zeros(length, dtype=np.float32)
    output[: min(length, len(audio))] = audio[:length]
    valid = round(int(row["audio_valid_samples"]) * rate / int(source_rate))
    return output, max(16, min(valid, length))


def rms_normalize(audio: np.ndarray) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float32)
    rms = math.sqrt(float(np.mean(value.astype(np.float64) ** 2)) + 1e-12)
    return value / rms * 0.08 if rms > 1e-8 else value


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "audio"


def label_free_derangement(datasets: list[str]) -> list[int]:
    """Rotate within each dataset without using labels or audio content."""

    permutation = list(range(len(datasets)))
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index, dataset in enumerate(datasets):
        grouped[dataset].append(index)
    for indices in grouped.values():
        if len(indices) < 2:
            raise ValueError("Audio oracle audit needs at least two validation audio items per dataset")
        for position, index in enumerate(indices):
            permutation[index] = indices[(position + 1) % len(indices)]
    if any(index == source for index, source in enumerate(permutation)):
        raise RuntimeError("Condition shuffle contains a self-loop")
    return permutation


def code_accuracy(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray, codebook: int) -> float:
    mask = np.asarray(valid[codebook], dtype=bool)
    return float(np.mean(np.asarray(prediction[codebook])[mask] == np.asarray(target[codebook])[mask])) if mask.any() else 0.0


def median(values: list[dict[str, float]], key: str) -> float:
    return float(np.median([row[key] for row in values]))


def main() -> None:
    args = parse_args()
    if args.freeze_on_pass and args.limit >= 0:
        raise ValueError("A limited audio audit is diagnostic only and cannot freeze a checkpoint")
    context = load_context(args.config)
    cfg = context.config
    settings = cfg.get("audio_oracle_gate") or {}
    device = torch.device(args.device) if args.device else default_device()
    lineage = build_lineage(context, require_optional_caches=not args.project_audio_only)
    checkpoint = resolve_config_path(context.config_path, cfg["paths"]["audio_checkpoint"])
    if args.verify_frozen:
        freeze = require_frozen_audio_checkpoint(context.config_path, cfg, lineage, checkpoint)
        print(json.dumps({"verified": True, "audio_checkpoint": str(checkpoint), "audio_checkpoint_sha256": file_sha256(checkpoint), "freeze": freeze}, indent=2, sort_keys=True))
        return
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint(payload, phase="audio", lineage=lineage, source=str(checkpoint))
    initialization = payload.get("initialization") or {}
    model = LabelFreeAudioModel(audio_config(cfg)).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    teachers = TeacherBank(resolve_config_path(context.config_path, cfg["paths"]["teacher_cache"]))
    dataset = LabelFreeAudioDataset(context, teachers, split="validation", include_public=not args.project_audio_only)
    indices = list(range(len(dataset)))
    if args.limit >= 0:
        indices = indices[: args.limit]
    items = [dataset[index] for index in indices]
    keys = [str(item["audio_key"]) for item in items]
    row_by_key = {
        row["audio_key"]: row
        for row in context.rows
        if context.split_for(row) == "validation" and row["dataset"] in {"feis", "karaone"}
    }
    missing_rows = [key for key in keys if key not in row_by_key]
    if missing_rows:
        raise ValueError(f"Validation audio keys have no manifest reference: {missing_rows[:3]}")
    datasets = [row_by_key[key]["dataset"] for key in keys]
    permutation = label_free_derangement(datasets)
    batch_size = int(args.batch_size or settings.get("batch_size", 8))

    conditions: list[torch.Tensor] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(items), batch_size), desc="[0722 audio-oracle] HuBERT conditions", unit="batch"):
            tokens = torch.stack([item["xlsr_tokens"] for item in items[start : start + batch_size]]).to(device)
            conditions.append(model.xlsr_encoder(tokens)["condition"].cpu())
    condition = torch.cat(conditions)

    correct_codes: list[np.ndarray] = []
    shuffled_codes: list[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(items), batch_size), desc="[0722 audio-oracle] MaskGIT decode", unit="batch"):
            stop = min(start + batch_size, len(items))
            correct = condition[start:stop].to(device)
            shuffled = condition[[permutation[index] for index in range(start, stop)]].to(device)
            correct_codes.extend(
                model.decoder.generate(correct, steps=int(cfg["evaluation"]["maskgit_steps"]), temperature=float(cfg["evaluation"]["synthesis_temperature"])).cpu().numpy()
            )
            shuffled_codes.extend(
                model.decoder.generate(shuffled, steps=int(cfg["evaluation"]["maskgit_steps"]), temperature=float(cfg["evaluation"]["synthesis_temperature"])).cpu().numpy()
            )

    codec_cfg = cfg["codec"]
    codec = DiscreteEncodec(
        DiscreteEncodecConfig(
            model_path=str(resolve_config_path(context.config_path, cfg["paths"]["encodec_model"])),
            sample_rate=int(codec_cfg["sample_rate"]), duration_sec=float(codec_cfg["duration_sec"]),
            bandwidth=float(codec_cfg["bandwidth"]),
        ),
        device,
    )
    target_length = round(codec.codec_sample_rate * float(codec_cfg["duration_sec"]))
    output_root = resolve_config_path(context.config_path, cfg["paths"]["output_root"])
    example_root = output_root / "audio/oracle_audit/examples"
    write_limit = int(args.write_examples_per_dataset if args.write_examples_per_dataset is not None else settings.get("write_examples_per_dataset", 4))
    written: defaultdict[str, int] = defaultdict(int)
    records: list[dict[str, Any]] = []

    for index in tqdm(range(len(items)), desc="[0722 audio-oracle] waveform metrics", unit="audio"):
        row = row_by_key[keys[index]]
        reference, valid_samples = reference_waveform(context, row, codec.codec_sample_rate, target_length)
        correct_audio = np.asarray(codec.decode(correct_codes[index], scale=None), dtype=np.float32).reshape(-1)
        shuffled_audio = np.asarray(codec.decode(shuffled_codes[index], scale=None), dtype=np.float32).reshape(-1)
        correct_audio = np.pad(correct_audio[:target_length], (0, max(0, target_length - len(correct_audio))))
        shuffled_audio = np.pad(shuffled_audio[:target_length], (0, max(0, target_length - len(shuffled_audio))))
        target_codes = items[index]["codes"].numpy()
        valid_codes = items[index]["code_valid_mask"].numpy()
        correct_metrics = reconstruction_metrics(reference[:valid_samples], correct_audio[:valid_samples], codec.codec_sample_rate, max_lag_ms=float(cfg["evaluation"]["max_envelope_lag_ms"]))
        shuffled_metrics = reconstruction_metrics(reference[:valid_samples], shuffled_audio[:valid_samples], codec.codec_sample_rate, max_lag_ms=float(cfg["evaluation"]["max_envelope_lag_ms"]))
        for name, prediction in (("correct", correct_codes[index]), ("shuffled", shuffled_codes[index])):
            target = correct_metrics if name == "correct" else shuffled_metrics
            target["q0_accuracy"] = code_accuracy(prediction, target_codes, valid_codes, 0)
            target["q1_accuracy"] = code_accuracy(prediction, target_codes, valid_codes, 1)
        record = {
            "audio_key": keys[index], "dataset": datasets[index], "label": row["label"],
            "subject_group_id": row["subject_group_id"],
            "shuffled_audio_key": keys[permutation[index]],
            "correct_condition": correct_metrics, "shuffled_condition": shuffled_metrics,
            "envelope_gain_over_shuffled": correct_metrics["lag_envelope_correlation"] - shuffled_metrics["lag_envelope_correlation"],
            "log_mel_gain_db_over_shuffled": shuffled_metrics["log_mel_mae_db"] - correct_metrics["log_mel_mae_db"],
        }
        records.append(record)
        if written[datasets[index]] < write_limit:
            stem = f"{written[datasets[index]]:02d}_{safe(keys[index])}"
            for mode, waveform in (("reference", reference), ("audio_condition_oracle", correct_audio), ("shuffled_condition", shuffled_audio)):
                write_wav(example_root / datasets[index] / mode / f"{stem}.wav", rms_normalize(waveform), codec.codec_sample_rate)
            written[datasets[index]] += 1

    thresholds = {
        "median_lag_envelope_correlation": float(settings["minimum_median_lag_envelope_correlation"]),
        "median_modulation_correlation": float(settings["minimum_median_modulation_correlation"]),
        "median_log_mel_mae_db_max": float(settings["maximum_median_log_mel_mae_db"]),
        "median_q0_accuracy": float(settings["minimum_median_q0_accuracy"]),
        "median_envelope_gain_over_shuffled": float(settings["minimum_median_envelope_gain_over_shuffled"]),
        "median_log_mel_gain_db_over_shuffled": float(settings["minimum_median_log_mel_gain_db_over_shuffled"]),
    }
    dataset_reports: dict[str, Any] = {}
    expected_bootstrap_root = output_root / str((cfg.get("audio_bootstrap") or {}).get("output_subdir", "bootstrap_0715"))
    source_checkpoint = Path(str(initialization.get("source_checkpoint", ""))).resolve()
    try:
        source_checkpoint.relative_to(expected_bootstrap_root.resolve())
        bootstrap_is_experiment_local = True
    except ValueError:
        bootstrap_is_experiment_local = False
    all_checks: dict[str, bool] = {
        "shared_nonlabel_initialization": initialization.get("mode") == "shared_nonlabel_weight_extraction",
        "compatible_tensors_copied": int(initialization.get("copied_tensor_count", 0)) > 0,
        "fresh_bootstrap_is_experiment_local": bootstrap_is_experiment_local,
        "validation_not_limited": args.limit < 0,
    }
    for dataset_name in sorted(set(datasets)):
        rows = [row for row in records if row["dataset"] == dataset_name]
        correct = [row["correct_condition"] for row in rows]
        shuffled = [row["shuffled_condition"] for row in rows]
        values = {
            "median_lag_envelope_correlation": median(correct, "lag_envelope_correlation"),
            "median_modulation_correlation": median(correct, "modulation_correlation"),
            "median_log_mel_mae_db": median(correct, "log_mel_mae_db"),
            "median_q0_accuracy": median(correct, "q0_accuracy"),
            "median_envelope_gain_over_shuffled": float(np.median([row["envelope_gain_over_shuffled"] for row in rows])),
            "median_log_mel_gain_db_over_shuffled": float(np.median([row["log_mel_gain_db_over_shuffled"] for row in rows])),
        }
        checks = {
            "envelope_absolute": values["median_lag_envelope_correlation"] >= thresholds["median_lag_envelope_correlation"],
            "modulation_absolute": values["median_modulation_correlation"] >= thresholds["median_modulation_correlation"],
            "log_mel_absolute": values["median_log_mel_mae_db"] <= thresholds["median_log_mel_mae_db_max"],
            "q0_absolute": values["median_q0_accuracy"] >= thresholds["median_q0_accuracy"],
            "envelope_condition_specific": values["median_envelope_gain_over_shuffled"] >= thresholds["median_envelope_gain_over_shuffled"],
            "log_mel_condition_specific": values["median_log_mel_gain_db_over_shuffled"] >= thresholds["median_log_mel_gain_db_over_shuffled"],
        }
        all_checks.update({f"{dataset_name}:{key}": value for key, value in checks.items()})
        dataset_reports[dataset_name] = {
            "n_unique_validation_audio": len(rows), "values": values, "checks": checks,
            "correct_summary": summarize(correct), "shuffled_summary": summarize(shuffled),
        }

    passed = bool(all(all_checks.values()))
    gate_path = resolve_config_path(context.config_path, cfg["paths"]["audio_oracle_gate"])
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": AUDIO_ORACLE_GATE_SCHEMA, "passed": passed,
        "failed_checks": sorted(key for key, value in all_checks.items() if not value),
        "checks": all_checks, "thresholds": thresholds, "datasets": dataset_reports,
        "audio_checkpoint": str(checkpoint), "audio_checkpoint_sha256": file_sha256(checkpoint),
        "initialization": initialization, "lineage": lineage,
        "selection_split": "validation", "labels_used_for_generation": False,
        "shuffled_condition_uses_labels": False, "test_accessed": False,
        "examples": str(example_root), "samples": records,
    }
    gate_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    freeze_path = resolve_config_path(context.config_path, cfg["paths"]["audio_freeze_manifest"])
    if passed and args.freeze_on_pass:
        freeze = {
            "schema_version": AUDIO_FREEZE_SCHEMA,
            "audio_checkpoint": str(checkpoint), "audio_checkpoint_sha256": file_sha256(checkpoint),
            "audio_oracle_gate": str(gate_path), "audio_oracle_gate_sha256": file_sha256(gate_path),
            "initialization": initialization, "lineage": lineage,
            "policy": "reuse_without_retraining_unless_RETRAIN_AUDIO=1", "test_accessed": False,
        }
        freeze_path.parent.mkdir(parents=True, exist_ok=True)
        freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif freeze_path.exists():
        freeze_path.unlink()

    print(json.dumps({"passed": passed, "gate": str(gate_path), "freeze": str(freeze_path) if passed and args.freeze_on_pass else None, "failed_checks": report["failed_checks"], "datasets": {key: value["values"] for key, value in dataset_reports.items()}}, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
