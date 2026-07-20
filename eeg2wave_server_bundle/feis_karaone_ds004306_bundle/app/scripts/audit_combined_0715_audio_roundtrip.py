from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
KARA_APP = APP_DIR.parents[1] / "karaone_overt_recon_bundle" / "app"
if str(KARA_APP) not in sys.path:
    sys.path.insert(0, str(KARA_APP))

from src.combined_0715.audio_eval import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    REQUIRED_CACHE_FIELDS,
    decode_cached_sample,
    select_stratified_validation_audio,
    summarise_metric_records,
    validate_cache_arrays,
    waveform_metrics,
)
from src.combined_0715.data import DATASETS, load_context, sha256_bytes  # noqa: E402
from src.karaone_0715.codec import DiscreteEncodec, DiscreteEncodecConfig  # noqa: E402
from src.karaone_0715.data import load_audio  # noqa: E402
from src.utils import resample_audio  # noqa: E402


THRESHOLDS = {
    "median_waveform_correlation_min": 0.65,
    "median_si_sdr_db_min": 0.0,
    "median_log_spectrogram_mae_db_max": 12.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode stratified validation EnCodec targets and audit the combined v2 cache round trip."
    )
    parser.add_argument("--config", default=str(APP_DIR / "configs" / "combined_0715_v1.yaml"))
    parser.add_argument("--cache", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve(config_path: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def _metadata_checks(raw: Any, context: Any, *, target_audio_samples: int) -> dict[str, bool]:
    keys = np.asarray(raw["keys"]).astype(str)
    datasets = np.asarray(raw["datasets"]).astype(str)
    labels = np.asarray(raw["labels"]).astype(str)
    paths = np.asarray(raw["audio_relpaths"]).astype(str)
    valid_samples = np.asarray(raw["audio_valid_samples"], dtype=np.int64)
    fit_split = np.asarray(raw["fit_split"], dtype=bool)
    manifest_by_key: dict[str, dict[str, str]] = {}
    consistent = True
    for row in context.rows:
        key = str(row["audio_key"])
        if key not in manifest_by_key:
            manifest_by_key[key] = row
        else:
            expected = manifest_by_key[key]
            consistent = consistent and all(
                str(expected[field]) == str(row[field]) for field in ("dataset", "label", "audio_relpath", "audio_valid_samples")
            )
    cache_metadata_matches = consistent
    for index, key in enumerate(keys.tolist()):
        row = manifest_by_key.get(key)
        cache_metadata_matches = bool(
            cache_metadata_matches
            and row is not None
            and datasets[index] == str(row["dataset"])
            and labels[index] == str(row["label"])
            and paths[index] == str(row["audio_relpath"])
            and valid_samples[index] == min(max(int(row["audio_valid_samples"]), 1), int(target_audio_samples))
        )
    train_keys = {str(row["audio_key"]) for row in context.rows if context.split_for(row) == "train"}
    expected_fit = np.asarray([key in train_keys for key in keys], dtype=bool)
    return {
        "manifest_audio_keys_covered": bool(set(manifest_by_key) == set(keys.tolist())),
        "cache_metadata_matches_manifest": bool(cache_metadata_matches),
        "fit_split_matches_locked_train": bool(np.array_equal(fit_split, expected_fit)),
    }


def _dataset_gate(
    summary: dict[str, dict[str, float]],
    thresholds: dict[str, float] | None = None,
) -> tuple[bool, list[str]]:
    thresholds = thresholds or THRESHOLDS
    reasons: list[str] = []
    if not summary:
        return False, ["no_validation_audio_selected"]
    if summary["waveform_correlation"]["median"] < thresholds["median_waveform_correlation_min"]:
        reasons.append("median_waveform_correlation_below_threshold")
    if summary["si_sdr_db"]["median"] < thresholds["median_si_sdr_db_min"]:
        reasons.append("median_si_sdr_below_threshold")
    if summary["log_spectrogram_mae_db"]["median"] > thresholds["median_log_spectrogram_mae_db_max"]:
        reasons.append("median_log_spectrogram_mae_above_threshold")
    return not reasons, reasons


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    context = load_context(config_path)
    cache_path = Path(args.cache).resolve() if args.cache else resolve(
        config_path, Path(cfg["paths"]["cache_root"]) / "combined_0715_encodec_codes.npz"
    )
    output_path = Path(args.output).resolve() if args.output else cache_path.with_name("audio_roundtrip_audit.json")
    seed = int(args.seed if args.seed is not None else cfg["run"]["seed"])
    codec_cfg = cfg["codec"]
    evaluation_cfg = cfg["evaluation"]
    max_per_label = int(
        args.max_per_label
        if args.max_per_label is not None
        else evaluation_cfg["roundtrip_max_per_label"]
    )
    thresholds = {
        "median_waveform_correlation_min": float(
            evaluation_cfg["minimum_codec_oracle_median_correlation"]
        ),
        "median_si_sdr_db_min": float(
            evaluation_cfg["minimum_codec_oracle_median_si_sdr_db"]
        ),
        "median_log_spectrogram_mae_db_max": float(
            evaluation_cfg["maximum_codec_oracle_median_log_spectrogram_mae_db"]
        ),
    }
    base_report: dict[str, Any] = {
        "version": CACHE_SCHEMA_VERSION,
        "phase": "audio_roundtrip_gate",
        "config": str(config_path),
        "config_sha256": sha256_bytes(config_path.read_bytes()),
        "cache": str(cache_path),
        "cache_sha256": sha256_bytes(cache_path.read_bytes()),
        "split": "validation",
        "sampling": {"seed": seed, "max_unique_audio_per_dataset_label": max_per_label},
        "thresholds": thresholds,
        "test_audio_waveforms_decoded": False,
        "test_manifest_rows_used_for_sampling": False,
    }

    with np.load(cache_path, allow_pickle=False) as raw:
        fields = set(raw.files)
        codes = np.asarray(raw["encodec_codes"]) if "encodec_codes" in fields else np.empty(0)
        checks = validate_cache_arrays(
            raw,
            codebooks=int(codec_cfg["codebooks"]),
            code_steps=int(codec_cfg["code_steps"]),
            vocab_size=int(codec_cfg["vocab_size"]),
            audio_sample_rate=int(codec_cfg["sample_rate"]),
            duration_sec=float(codec_cfg["duration_sec"]),
        )
        if REQUIRED_CACHE_FIELDS <= fields:
            checks.update(
                _metadata_checks(
                    raw,
                    context,
                    target_audio_samples=int(round(int(codec_cfg["sample_rate"]) * float(codec_cfg["duration_sec"]))),
                )
            )
        else:
            checks.update(
                {
                    "manifest_audio_keys_covered": False,
                    "cache_metadata_matches_manifest": False,
                    "fit_split_matches_locked_train": False,
                }
            )
        structure_passed = bool(all(checks.values()))
        report = {
            **base_report,
            "passed": False,
            "structure_passed": structure_passed,
            "checks": checks,
            "missing_fields": sorted(REQUIRED_CACHE_FIELDS - fields),
            "observed_codes_shape": list(codes.shape),
        }
        if not structure_passed:
            report["reasons"] = [name for name, passed in checks.items() if not passed]
            write_report(output_path, report)
            raise SystemExit(1)

        keys = np.asarray(raw["keys"]).astype(str)
        index_by_key = {key: index for index, key in enumerate(keys.tolist())}
        selected = select_stratified_validation_audio(
            context,
            keys,
            max_per_label=max_per_label,
            seed=seed,
        )
        expected_groups = {
            (str(row["dataset"]), str(row["label"]))
            for row in context.rows
            if context.split_for(row) == "validation"
        }
        selected_groups = {(str(row["dataset"]), str(row["label"])) for row in selected}
        checks["validation_dataset_label_coverage"] = bool(selected_groups == expected_groups)
        if not checks["validation_dataset_label_coverage"]:
            report.update(
                {
                    "checks": checks,
                    "reasons": ["validation_dataset_label_coverage"],
                    "selected_audio_keys": [str(row["audio_key"]) for row in selected],
                }
            )
            write_report(output_path, report)
            raise SystemExit(1)

        # Copy the selected arrays while the npz handle is open. This also makes
        # it impossible for later decoding code to inspect unselected test rows.
        selected_cache = []
        for row in selected:
            index = index_by_key[str(row["audio_key"])]
            selected_cache.append(
                {
                    "row": row,
                    "codes": np.asarray(raw["encodec_codes"][index]).copy(),
                    "scale": np.asarray(raw["encodec_scale"][index]).copy(),
                    "scale_valid": bool(raw["encodec_scale_valid"][index]),
                    "audio_valid_samples": int(raw["audio_valid_samples"][index]),
                    "audio_relpath": str(raw["audio_relpaths"][index]),
                }
            )
        cached_codec_sample_rate = int(np.asarray(raw["codec_sample_rate"]).reshape(-1)[0])
        cached_bandwidth = float(np.asarray(raw["codec_bandwidth"]).reshape(-1)[0])

    device = torch.device(args.device) if args.device else default_device()
    codec = DiscreteEncodec(
        DiscreteEncodecConfig(
            model_path=str(resolve(config_path, cfg["paths"]["encodec_model"])),
            sample_rate=int(codec_cfg["sample_rate"]),
            duration_sec=float(codec_cfg["duration_sec"]),
            bandwidth=float(codec_cfg["bandwidth"]),
        ),
        device,
    )
    if codec.codec_sample_rate != cached_codec_sample_rate:
        raise ValueError(
            f"Loaded codec sample rate {codec.codec_sample_rate} disagrees with cache {cached_codec_sample_rate}"
        )

    sample_metrics: list[dict[str, Any]] = []
    for item in tqdm(selected_cache, desc="[combined cache] validation round-trip", unit="audio", dynamic_ncols=True):
        row = item["row"]
        reference_16k = load_audio(
            context.root / item["audio_relpath"],
            sample_rate=int(codec_cfg["sample_rate"]),
            duration_sec=float(codec_cfg["duration_sec"]),
        )
        reference = resample_audio(
            reference_16k,
            src_sr=int(codec_cfg["sample_rate"]),
            dst_sr=codec.codec_sample_rate,
        )
        decoded = decode_cached_sample(
            codec,
            item["codes"],
            item["scale"],
            item["scale_valid"],
        )
        valid_codec_samples = max(
            2,
            int(round(item["audio_valid_samples"] * codec.codec_sample_rate / int(codec_cfg["sample_rate"]))),
        )
        metrics = waveform_metrics(
            reference,
            decoded,
            codec.codec_sample_rate,
            valid_samples=valid_codec_samples,
        )
        sample_metrics.append(
            {
                "dataset": str(row["dataset"]),
                "label": str(row["label"]),
                "audio_key": str(row["audio_key"]),
                "audio_relpath": item["audio_relpath"],
                "audio_valid_samples_16khz": item["audio_valid_samples"],
                "encodec_scale_used": item["scale_valid"],
                **metrics,
            }
        )

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in sample_metrics:
        by_dataset[str(record["dataset"])].append(record)
    summaries: dict[str, Any] = {}
    dataset_reasons: dict[str, list[str]] = {}
    for dataset in DATASETS:
        summary = summarise_metric_records(by_dataset.get(dataset, []))
        passed, reasons = _dataset_gate(summary, thresholds)
        summaries[dataset] = {
            "n_audio": len(by_dataset.get(dataset, [])),
            "passed": passed,
            "metrics": summary,
        }
        if reasons:
            dataset_reasons[dataset] = reasons

    report.update(
        {
            "passed": bool(all(value["passed"] for value in summaries.values())),
            "reasons": dataset_reasons,
            "device": str(device),
            "codec": {
                "input_sample_rate": int(codec_cfg["sample_rate"]),
                "codec_sample_rate": codec.codec_sample_rate,
                "duration_sec": float(codec_cfg["duration_sec"]),
                "bandwidth": cached_bandwidth,
                "codebooks": int(codec_cfg["codebooks"]),
                "code_steps": int(codec_cfg["code_steps"]),
            },
            "metric_definition": {
                "interval": "audio_valid_samples_only",
                "spectrogram": "unit-RMS, STFT 512/128-hop, -80 dB floor",
                "scale": "cached EnCodec scale when encodec_scale_valid is true, otherwise None",
            },
            "selected_audio_keys": [record["audio_key"] for record in sample_metrics],
            "n_selected_audio": len(sample_metrics),
            "samples": sample_metrics,
            "metrics_by_dataset": summaries,
        }
    )
    write_report(output_path, report)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
