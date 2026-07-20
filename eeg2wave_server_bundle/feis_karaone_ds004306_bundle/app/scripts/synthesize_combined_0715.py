from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from scipy.io import wavfile


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
KARA_APP = APP_DIR.parents[1] / "karaone_overt_recon_bundle" / "app"
if str(KARA_APP) not in sys.path:
    sys.path.insert(0, str(KARA_APP))

from src.combined_0715.data import (  # noqa: E402
    AudioCodeBank,
    CombinedEEGDataset,
    load_context,
)
from src.combined_0715.audio_eval import (  # noqa: E402
    METRIC_NAMES,
    decode_cached_sample,
    summarise_metric_records,
    waveform_metrics,
)
from src.combined_0715.lineage import (  # noqa: E402
    build_run_lineage,
    file_sha256,
    preauthorize_locked_test,
    validate_checkpoint_payload,
    validate_gate_binding,
    validate_lineage,
)
from src.combined_0715.model import (  # noqa: E402
    AudioCodeAutoencoder,
    AudioCodeModelConfig,
    EEGConditionEncoder,
    EEGModelConfig,
)
from src.karaone_0715.codec import DiscreteEncodec, DiscreteEncodecConfig  # noqa: E402
from src.karaone_0715.data import load_audio  # noqa: E402
from src.utils import resample_audio  # noqa: E402


DATASETS = ("feis", "karaone", "ds004306")
OUTPUT_KINDS = (
    "reference",
    "codec_oracle",
    "eeg_conditioned",
    "label_only",
    "zero_eeg",
    "shuffled_eeg",
    "dataset_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reference plus six controlled EnCodec reconstructions from combined EEG."
    )
    parser.add_argument("--config", default=str(APP_DIR / "configs/combined_0715_v1.yaml"))
    parser.add_argument("--cache", required=True)
    parser.add_argument("--audio-checkpoint", required=True)
    parser.add_argument("--eeg-checkpoint", required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-final-test", action="store_true")
    parser.add_argument(
        "--allow-failed-gate",
        action="store_true",
        help="Validation only: explicitly export exploratory samples when the audio round-trip gate is absent/failed.",
    )
    parser.add_argument("--roundtrip-gate", default=None)
    parser.add_argument("--validation-gate", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--maskgit-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=12)
    return parser.parse_args()


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_from_config(config_path: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required gate is missing: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def deterministic_label_derangement(rows: Iterable[dict[str, str]], seed: int) -> np.ndarray:
    """Return a deterministic same-label permutation with no fixed points."""

    materialized = tuple(rows)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(materialized):
        groups[str(row["label"])].append(index)
    permutation = np.full(len(materialized), -1, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    for label in sorted(groups):
        indices = sorted(groups[label], key=lambda index: str(materialized[index].get("sample_key", index)))
        if len(indices) < 2:
            raise ValueError(
                f"Cannot build a no-self shuffled-EEG control for label {label!r}: only one trial in this split"
            )
        offset = int(rng.integers(1, len(indices)))
        shifted = np.roll(np.asarray(indices, dtype=np.int64), offset)
        for target, source in zip(indices, shifted.tolist()):
            if target == source:
                raise AssertionError("Internal derangement error: shuffled EEG contains a self-loop")
            permutation[target] = source
    if np.any(permutation < 0):
        raise AssertionError("Internal derangement error: not every trial received a shuffled source")
    return permutation


def dataset_label_prior(context: Any, dataset: str, *, num_labels: int = 30) -> torch.Tensor:
    counts = Counter(
        row["label"]
        for row in context.rows
        if row["dataset"] == dataset and context.split_for(row) == "train"
    )
    if not counts:
        raise ValueError(f"No training labels are available for dataset-only prior: {dataset}")
    probabilities = torch.zeros(num_labels, dtype=torch.float32)
    total = float(sum(counts.values()))
    for label, count in counts.items():
        probabilities[context.label_to_global[(dataset, label)]] = float(count) / total
    if not torch.isclose(probabilities.sum(), torch.tensor(1.0), atol=1e-6):
        raise AssertionError("Dataset prior does not sum to one")
    return probabilities


def build_control_inputs(
    target_output: dict[str, torch.Tensor],
    zero_output: dict[str, torch.Tensor],
    shuffled_output: dict[str, torch.Tensor],
    prior: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Build conditions/probabilities without using the true trial label."""

    target_probabilities = torch.softmax(target_output["label_logits"], dim=-1)
    zero_probabilities = torch.softmax(zero_output["label_logits"], dim=-1)
    shuffled_probabilities = torch.softmax(shuffled_output["label_logits"], dim=-1)
    zero_condition = torch.zeros_like(target_output["condition"])
    prior = prior.to(device=zero_condition.device, dtype=zero_condition.dtype).view(1, -1)
    return {
        "eeg_conditioned": (target_output["condition"], target_probabilities),
        "label_only": (zero_condition, target_probabilities),
        "zero_eeg": (zero_output["condition"], zero_probabilities),
        "shuffled_eeg": (shuffled_output["condition"], shuffled_probabilities),
        "dataset_only": (zero_condition, prior),
    }


def ensure_output_layout(root: Path) -> dict[str, Path]:
    folders = {name: root / name for name in OUTPUT_KINDS}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "sample"


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    value = np.asarray(audio, dtype=np.float32).reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, int(sample_rate), np.asarray(np.clip(value, -1.0, 1.0) * 32767.0, dtype=np.int16))


def match_length(audio: np.ndarray, length: int) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(value) >= int(length):
        return value[: int(length)]
    return np.pad(value, (0, int(length) - len(value))).astype(np.float32)


def rms_normalize(audio: np.ndarray, target: float = 0.08) -> np.ndarray:
    value = np.asarray(audio, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(value), dtype=np.float64) + 1e-12))
    if rms <= 1e-8:
        return np.zeros_like(value)
    return np.clip(value * min(float(target) / rms, 10.0), -0.95, 0.95).astype(np.float32)


def code_metrics(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    result: dict[str, float] = {}
    for codebook in (0, 1):
        selected = mask[codebook].bool()
        result[f"q{codebook}_accuracy"] = float(
            (predicted[codebook][selected] == target[codebook][selected]).float().mean().item()
        ) if selected.any() else float("nan")
    return result


def summarise(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {name: float("nan") for name in ("mean", "median", "p05", "min")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "min": float(np.min(array)),
    }


def tensor_batch(sample: dict[str, Any], key: str, device: torch.device) -> torch.Tensor:
    value = sample[key]
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.unsqueeze(0).to(device)


def load_models(
    audio_path: Path,
    eeg_path: Path,
    device: torch.device,
    lineage: dict[str, Any],
) -> tuple[AudioCodeAutoencoder, EEGConditionEncoder, dict[str, Any], dict[str, Any], str, str]:
    audio_sha = file_sha256(audio_path)
    eeg_sha = file_sha256(eeg_path)
    audio_payload = torch.load(audio_path, map_location=device, weights_only=False)
    validate_checkpoint_payload(
        audio_payload,
        expected_phase="audio",
        expected_lineage=lineage,
        expected_dependencies={},
        source=str(audio_path),
    )
    eeg_payload = torch.load(eeg_path, map_location=device, weights_only=False)
    validate_checkpoint_payload(
        eeg_payload,
        expected_phase="eeg",
        expected_lineage=lineage,
        expected_dependencies={"audio_checkpoint_sha256": audio_sha},
        source=str(eeg_path),
    )
    audio = AudioCodeAutoencoder(AudioCodeModelConfig(**audio_payload["model_config"])).to(device)
    audio.load_state_dict(audio_payload["state_dict"], strict=True)
    audio.eval()
    eeg = EEGConditionEncoder(EEGModelConfig(**eeg_payload["model_config"])).to(device)
    eeg.load_state_dict(eeg_payload["state_dict"], strict=True)
    eeg.eval()
    return audio, eeg, audio_payload, eeg_payload, audio_sha, eeg_sha


def check_access_policy(
    *,
    split: str,
    allow_final_test: bool,
    allow_failed_gate: bool,
    roundtrip_gate_path: Path,
    validation_gate_path: Path,
    lineage: dict[str, Any],
    audio_checkpoint_sha256: str,
    eeg_checkpoint_sha256: str,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Validate all gates before a test Dataset can be instantiated."""

    if split == "test" and not allow_final_test:
        raise PermissionError("Locked test synthesis requires --allow-final-test")
    if split == "test" and allow_failed_gate:
        raise PermissionError("--allow-failed-gate is forbidden for locked test synthesis")

    exploratory_reasons: list[str] = []
    try:
        roundtrip_gate = read_json(roundtrip_gate_path)
        if not bool(roundtrip_gate.get("passed")):
            raise PermissionError(
                f"Audio round-trip gate is not passed: {roundtrip_gate.get('reasons') or roundtrip_gate.get('reason')}"
            )
        expected_roundtrip_binding = {
            "config_sha256": lineage["config_sha256"],
            "cache_sha256": lineage["cache_sha256"],
            "version": lineage["cache_version"],
            "test_audio_waveforms_decoded": False,
        }
        stale = {
            key: {"gate": roundtrip_gate.get(key), "current": value}
            for key, value in expected_roundtrip_binding.items()
            if roundtrip_gate.get(key) != value
        }
        if stale:
            raise PermissionError(f"Audio round-trip gate binding mismatch: {json.dumps(stale, sort_keys=True)}")
        if "lineage" in roundtrip_gate:
            validate_lineage(roundtrip_gate["lineage"], lineage, source="audio round-trip gate")
    except (FileNotFoundError, PermissionError, ValueError) as error:
        if split == "test" or not allow_failed_gate:
            raise PermissionError(
                f"Audio round-trip audit is required before non-exploratory synthesis: {error}"
            ) from error
        roundtrip_gate = {"passed": False, "error": str(error), "path": str(roundtrip_gate_path)}
        exploratory_reasons.append(f"audio_roundtrip_gate: {error}")

    if split == "test":
        validation_gate = read_json(validation_gate_path)
        validate_gate_binding(
            validation_gate,
            lineage=lineage,
            audio_checkpoint_sha256=audio_checkpoint_sha256,
            eeg_checkpoint_sha256=eeg_checkpoint_sha256,
        )
    return bool(exploratory_reasons), exploratory_reasons, roundtrip_gate


def trial_claim_allowed(row: dict[str, str]) -> bool:
    return row.get("pairing_confidence") == "karaone_same_trial_overt"


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(cfg["run"]["seed"])
    set_seed(seed)
    # Reject unauthorized test requests before loading manifests, caches, checkpoints, or models.
    if args.split == "test" and not args.allow_final_test:
        raise PermissionError("Locked test synthesis requires --allow-final-test")
    if args.split == "test" and args.allow_failed_gate:
        raise PermissionError("--allow-failed-gate is forbidden for locked test synthesis")
    audio_path = Path(args.audio_checkpoint).resolve()
    eeg_path = Path(args.eeg_checkpoint).resolve()
    output_root = resolve_from_config(config_path, cfg["paths"]["output_root"])
    roundtrip_gate_path = (
        Path(args.roundtrip_gate).resolve()
        if args.roundtrip_gate
        else resolve_from_config(config_path, cfg["paths"]["cache_root"]) / "audio_roundtrip_audit.json"
    )
    validation_gate_path = (
        Path(args.validation_gate).resolve()
        if args.validation_gate
        else output_root / "eeg" / "metrics" / "validation_gate.json"
    )

    # First-stage authorization uses only config, checkpoint/gate metadata and
    # the validation report.  It must precede loading the manifest, cache, or
    # hashing any locked-test EEG payload.  Full current-lineage validation is
    # repeated below after this authorization succeeds.
    if args.split == "test":
        preliminary_gate = preauthorize_locked_test(
            validation_gate_path,
            config_path=config_path,
            audio_checkpoint_path=audio_path,
            eeg_checkpoint_path=eeg_path,
        )
        preliminary_lineage = preliminary_gate["lineage"]
        check_access_policy(
            split=args.split,
            allow_final_test=True,
            allow_failed_gate=False,
            roundtrip_gate_path=roundtrip_gate_path,
            validation_gate_path=validation_gate_path,
            lineage=preliminary_lineage,
            audio_checkpoint_sha256=file_sha256(audio_path),
            eeg_checkpoint_sha256=file_sha256(eeg_path),
        )

    context = load_context(config_path)
    bank = AudioCodeBank(Path(args.cache).resolve())
    lineage = build_run_lineage(config_path, context, bank)
    device = torch.device(args.device) if args.device else default_device()
    audio, eeg, audio_payload, eeg_payload, audio_sha, eeg_sha = load_models(
        audio_path, eeg_path, device, lineage
    )
    exploratory, exploratory_reasons, roundtrip_gate = check_access_policy(
        split=args.split,
        allow_final_test=bool(args.allow_final_test),
        allow_failed_gate=bool(args.allow_failed_gate),
        roundtrip_gate_path=roundtrip_gate_path,
        validation_gate_path=validation_gate_path,
        lineage=lineage,
        audio_checkpoint_sha256=audio_sha,
        eeg_checkpoint_sha256=eeg_sha,
    )
    validation_gate_record = {
        "required": args.split == "test",
        "path": str(validation_gate_path),
        "sha256": file_sha256(validation_gate_path)
        if args.split == "test" and validation_gate_path.is_file()
        else None,
    }

    # Gate validation above must remain before construction/access of the locked test Dataset.
    dataset = CombinedEEGDataset(
        context, bank, args.dataset, args.split, eeg_len=int(cfg["data"]["eeg_len"])
    )
    shuffle_indices = deterministic_label_derangement(dataset.rows, seed)
    limit = len(dataset) if args.limit is None or int(args.limit) < 0 else min(len(dataset), int(args.limit))
    if limit < 1:
        raise ValueError("Synthesis requires at least one selected trial; use --limit >= 1")
    prior = dataset_label_prior(context, args.dataset).to(device)

    codec_cfg = cfg["codec"]
    codec = DiscreteEncodec(
        DiscreteEncodecConfig(
            model_path=str(resolve_from_config(config_path, cfg["paths"]["encodec_model"])),
            sample_rate=int(codec_cfg["sample_rate"]),
            duration_sec=float(codec_cfg["duration_sec"]),
            bandwidth=float(codec_cfg["bandwidth"]),
        ),
        device,
    )
    steps = int(args.maskgit_steps or cfg["evaluation"]["maskgit_steps"])
    temperature = float(
        cfg["evaluation"]["synthesis_temperature"] if args.temperature is None else args.temperature
    )
    destination = Path(args.output).resolve() / args.dataset / args.split
    folders = ensure_output_layout(destination)
    aggregate: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in (*METRIC_NAMES, "q0_accuracy", "q1_accuracy")}
        for name in OUTPUT_KINDS[1:]
    }
    files: list[dict[str, Any]] = []

    with torch.no_grad():
        for index in range(limit):
            row = dataset.rows[index]
            sample = dataset[index]
            shuffled_index = int(shuffle_indices[index])
            shuffled_row = dataset.rows[shuffled_index]
            shuffled_sample = dataset[shuffled_index]
            target_output = eeg(
                tensor_batch(sample, "eeg", device),
                tensor_batch(sample, "eeg_valid_len", device),
                tensor_batch(sample, "dataset_idx", device),
            )
            zero_output = eeg(
                torch.zeros_like(tensor_batch(sample, "eeg", device)),
                tensor_batch(sample, "eeg_valid_len", device),
                tensor_batch(sample, "dataset_idx", device),
            )
            shuffled_output = eeg(
                tensor_batch(shuffled_sample, "eeg", device),
                tensor_batch(shuffled_sample, "eeg_valid_len", device),
                tensor_batch(shuffled_sample, "dataset_idx", device),
            )
            inputs = build_control_inputs(target_output, zero_output, shuffled_output, prior)
            generated_codes = {
                name: audio.decoder.generate(condition, probabilities, steps=steps, temperature=temperature)[0]
                for name, (condition, probabilities) in inputs.items()
            }
            true_codes = tensor_batch(sample, "codes", device)[0]
            code_mask = tensor_batch(sample, "code_mask", device)[0]
            audio_index = int(
                sample["audio_idx"].item() if torch.is_tensor(sample["audio_idx"]) else sample["audio_idx"]
            )
            decoded: dict[str, np.ndarray] = {
                "codec_oracle": decode_cached_sample(
                    codec,
                    true_codes.cpu().numpy(),
                    bank.scale[audio_index],
                    bool(bank.scale_valid[audio_index]),
                ),
                **{
                    name: codec.decode(codes.cpu().numpy(), scale=None)
                    for name, codes in generated_codes.items()
                },
            }

            reference_16k = load_audio(
                context.root / str(row["audio_relpath"]),
                sample_rate=int(codec_cfg["sample_rate"]),
                duration_sec=float(codec_cfg["duration_sec"]),
            )
            reference = resample_audio(
                reference_16k,
                src_sr=int(codec_cfg["sample_rate"]),
                dst_sr=codec.codec_sample_rate,
            )
            target_length = len(reference)
            valid_samples_16k = max(
                1,
                min(
                    int(row.get("audio_valid_samples") or bank.audio_valid_samples[audio_index]),
                    len(reference_16k),
                ),
            )
            valid_samples_codec = max(
                16,
                min(
                    target_length,
                    int(round(valid_samples_16k * codec.codec_sample_rate / int(codec_cfg["sample_rate"]))),
                ),
            )
            decoded = {name: match_length(value, target_length) for name, value in decoded.items()}

            sample_key = str(row.get("sample_key") or f"{row['subject_recording_id']}:{row['trial_index']}")
            stem = f"{index:04d}_{safe_stem(sample_key)}"
            paths = {name: folders[name] / f"{stem}.wav" for name in OUTPUT_KINDS}
            write_wav(paths["reference"], rms_normalize(reference), codec.codec_sample_rate)
            for name, value in decoded.items():
                write_wav(paths[name], rms_normalize(value), codec.codec_sample_rate)

            mode_metrics: dict[str, dict[str, float]] = {}
            for name, candidate in decoded.items():
                values = waveform_metrics(
                    reference,
                    candidate,
                    codec.codec_sample_rate,
                    valid_samples=valid_samples_codec,
                )
                predicted = true_codes if name == "codec_oracle" else generated_codes[name]
                values.update(code_metrics(predicted, true_codes, code_mask))
                mode_metrics[name] = values
                for metric, value in values.items():
                    aggregate[name][metric].append(float(value))

            files.append(
                {
                    "sample_key": sample_key,
                    "audio_key": str(row["audio_key"]),
                    "subject_group_id": str(row["subject_group_id"]),
                    "subject_recording_id": str(row["subject_recording_id"]),
                    "trial_index": int(row["trial_index"]),
                    "label": str(row["label"]),
                    "audio_pairing": str(row.get("audio_pairing", "unknown")),
                    "pairing_confidence": str(row.get("pairing_confidence", "unknown")),
                    "audio_valid_samples_16khz": valid_samples_16k,
                    "shuffle_source_sample_key": str(
                        shuffled_row.get("sample_key")
                        or f"{shuffled_row['subject_recording_id']}:{shuffled_row['trial_index']}"
                    ),
                    "shuffle_source_subject_group_id": str(shuffled_row["subject_group_id"]),
                    "shuffle_same_dataset": shuffled_row["dataset"] == row["dataset"],
                    "shuffle_same_label": shuffled_row["label"] == row["label"],
                    "shuffle_no_self": shuffled_index != index,
                    "trial_level_claim_allowed": trial_claim_allowed(row),
                    "files": {name: str(path.relative_to(destination)) for name, path in paths.items()},
                    "mode_metrics": mode_metrics,
                }
            )

    summary = {}
    for mode, metrics in aggregate.items():
        waveform_records = [
            {name: metrics[name][index] for name in METRIC_NAMES}
            for index in range(len(metrics[METRIC_NAMES[0]]))
        ]
        summary[mode] = {
            **summarise_metric_records(waveform_records),
            "q0_accuracy": summarise(metrics["q0_accuracy"]),
            "q1_accuracy": summarise(metrics["q1_accuracy"]),
        }
    manifest: dict[str, Any] = {
        "version": "combined-0715-synthesis-v2",
        "phase": "synthesis_controls",
        "dataset": args.dataset,
        "split": args.split,
        "n_generated": len(files),
        "sample_rate_hz": codec.codec_sample_rate,
        "maskgit_steps": steps,
        "temperature": temperature,
        "seed": seed,
        "output_kinds": list(OUTPUT_KINDS),
        "audio_checkpoint": str(audio_path),
        "audio_checkpoint_sha256": audio_sha,
        "audio_checkpoint_epoch": int(audio_payload["epoch"]),
        "eeg_checkpoint": str(eeg_path),
        "eeg_checkpoint_sha256": eeg_sha,
        "eeg_checkpoint_epoch": int(eeg_payload["epoch"]),
        "lineage": lineage,
        "audio_roundtrip_gate": {
            "path": str(roundtrip_gate_path),
            "passed": bool(roundtrip_gate.get("passed")),
            "sha256": file_sha256(roundtrip_gate_path) if roundtrip_gate_path.is_file() else None,
        },
        "validation_gate": validation_gate_record,
        "exploratory": exploratory,
        "exploratory_reasons": exploratory_reasons,
        "trial_level_claim_allowed": bool(files) and all(
            bool(item["trial_level_claim_allowed"]) for item in files
        ),
        "ds004306_trial_level_claim_allowed": False if args.dataset == "ds004306" else None,
        "reference_waveform_used_for_generation": False,
        "cached_target_codes_and_scale_used_only_for_codec_oracle": True,
        "true_trial_label_used_for_eeg_conditioned": False,
        "true_trial_label_used_for_label_only": False,
        "true_trial_label_used_for_zero_eeg": False,
        "true_trial_label_used_for_shuffled_pair_selection": True,
        "true_trial_label_used_for_dataset_only": False,
        "label_only_probability_source": "same-trial EEG label head",
        "dataset_only_probability_source": "train-manifest empirical dataset label prior",
        "shuffled_eeg_policy": "deterministic same-dataset same-label derangement without self-loops",
        "waveform_metrics_use_valid_audio_only": True,
        "log_spectrogram_metrics_rms_normalized": True,
        "test_accessed": args.split == "test",
        "aggregate_metrics": summary,
        "files": files,
    }
    manifest_path = destination / "synthesis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"wrote {len(files)} samples and manifest to {destination}", flush=True)


if __name__ == "__main__":
    main()
