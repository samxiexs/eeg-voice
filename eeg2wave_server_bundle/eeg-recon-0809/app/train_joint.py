#!/usr/bin/env python3
"""Run the gated DS004940/DS006104 content pilot; never launches full training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler, Subset, WeightedRandomSampler

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(APP / "src"))

from eeg2speech.data import (AlternatingBatchIterator, ContentGroupedBatchSampler, JointManifestDataset, auxiliary_indices,
                             homogeneous_collate, phoneme_vocabulary_from_manifest, pilot_indices)
from eeg2speech.gates import require_registered_m0_gates
from eeg2speech.losses import counterfactual_eeg, joint_content_loss
from eeg2speech.model import JointEEGContentModel


def resolve(path: str | Path, base: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (base / value).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_code_sha256() -> str:
    digest = hashlib.sha256()
    paths = sorted([Path(__file__), *list((APP / "src").rglob("*.py"))], key=lambda path: str(path))
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode()); digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def model_code_sha256() -> str:
    """Fingerprint model/data/loss semantics, excluding training control flow."""
    digest = hashlib.sha256()
    paths = sorted((APP / "src").rglob("*.py"), key=lambda path: str(path))
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode()); digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def stable_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Write a checkpoint atomically so an interrupt never leaves a half file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def optimizer_to(optimizer: torch.optim.Optimizer, target: torch.device) -> None:
    """Move optimizer tensors after loading a CPU checkpoint onto the active device."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(target)


def resume_contract(*, args: argparse.Namespace, cfg: dict, artifact_hashes: dict[str, str],
                    split_protocol: str, artifact_set: str, target_name: str,
                    normalizer_name: str, source_lock: dict) -> dict:
    return {
        "mode": args.mode, "stage": args.stage, "seed": args.seed,
        "run_kind": "smoke" if args.smoke_model else ("explore" if args.explore else "pilot"),
        "split_protocol": split_protocol, "artifact_set": artifact_set,
        "target_name": target_name, "normalizer_name": normalizer_name,
        "model_config_sha256": stable_sha256(cfg["model"]),
        "batch_size": int(cfg["training"]["batch_size"]),
        "content_grouped_batch": cfg["training"].get("content_grouped_batch"),
        "pilot_config_sha256": sha256_file(args.config),
        "model_code_sha256": model_code_sha256(),
        "source_lock_sha256": source_lock["source_lock_sha256"],
        "artifact_hashes": artifact_hashes,
    }


def contract_mismatches(saved: dict, current: dict, *, allow_legacy_explore_control_upgrade: bool = False) -> list[str]:
    """Compare semantic checkpoint inputs without invalidating control-plane upgrades.

    Checkpoints created before resumability existed used one hash for both the
    model and train_joint.py.  A change to checkpoint cadence then made every
    completed exploratory run look incompatible.  Legacy explore artifacts may
    migrate only if every non-code input remains identical. Formal pilot runs
    stay strict until they have the model-only fingerprint.
    """
    code_keys = {"runtime_code_sha256", "model_code_sha256"}
    changed = [key for key in sorted((set(saved) | set(current)) - code_keys)
               if saved.get(key) != current.get(key)]
    if changed:
        return changed
    saved_model = saved.get("model_code_sha256")
    if saved_model:
        return ([] if saved_model == current.get("model_code_sha256") else ["model_code_sha256"])
    # A code-less contract is useful in focused unit tests and is not a
    # persisted legacy checkpoint.  Only a saved runtime fingerprint denotes
    # the pre-migration checkpoint format handled below.
    if "runtime_code_sha256" not in saved:
        return []
    if allow_legacy_explore_control_upgrade and saved.get("run_kind") == "explore":
        return []
    return ["legacy_runtime_code_sha256"]


def resume_maximum_steps(requested: int, state: dict) -> tuple[int, bool]:
    """Do not truncate an already-started run when a later budget is smaller.

    A lower max-steps value is a budget for future runs. A partial checkpoint
    must retain its original finish line, otherwise the runner could label an
    incomplete model as completed merely because its new budget is below the
    saved step count.
    """
    completed = int(state.get("completed_steps", 0))
    original = int(state.get("maximum_steps", requested))
    if completed < 0 or original < completed:
        raise RuntimeError(f"partial checkpoint has invalid completed_steps={completed} / maximum_steps={original}")
    return (original, True) if completed > requested else (requested, False)


def learning_rate_scheduler(optimizer: torch.optim.Optimizer, training: dict,
                            maximum_steps: int):
    """Create an optional resumable schedule; legacy configs stay constant-LR."""
    kind = str(training.get("lr_schedule", "constant")).lower()
    if kind == "constant":
        return None
    if kind != "cosine":
        raise ValueError(f"unsupported lr_schedule={kind!r}; choose constant or cosine")
    floor = float(training.get("min_learning_rate", 0.0))
    initial = float(training["learning_rate"])
    if not 0.0 <= floor <= initial:
        raise ValueError("min_learning_rate must be between zero and learning_rate")
    if maximum_steps < 1:
        raise ValueError("cosine learning-rate schedule requires positive maximum_steps")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=maximum_steps, eta_min=floor,
    )


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sampler_for(dataset: JointManifestDataset, indices: list[int], seed: int,
                strategy: str = "weighted_with_replacement"):
    """Create the deterministic sampler declared by an experiment config.

    The legacy pilot balances incomplete grids with weighted replacement.  A
    large, already complete subject×content grid instead needs every cell
    exactly once per epoch, so it uses a seeded shuffled sampler.
    """
    subset = Subset(dataset, indices)
    if strategy == "weighted_with_replacement":
        weights = dataset.sampling_weights()[indices]
        return WeightedRandomSampler(weights, num_samples=max(len(indices), 1), replacement=True,
                                     generator=torch.Generator().manual_seed(seed))
    if strategy == "shuffled_without_replacement":
        return RandomSampler(subset, replacement=False, generator=torch.Generator().manual_seed(seed))
    raise ValueError("sampling_strategy must be weighted_with_replacement or shuffled_without_replacement")


def loader_for(dataset: JointManifestDataset, indices: list[int], batch_size: int, seed: int,
               sampling_strategy: str = "weighted_with_replacement",
               content_grouped: dict | None = None) -> DataLoader:
    if not indices:
        raise RuntimeError("pilot selection produced zero trials")
    subset = Subset(dataset, indices)
    if content_grouped:
        sampler = ContentGroupedBatchSampler(
            dataset.frame, indices, batch_size=batch_size,
            contents_per_batch=int(content_grouped.get("contents_per_batch", 4)),
            subjects_per_content=int(content_grouped.get("subjects_per_content", 2)), seed=seed,
        )
        return DataLoader(subset, batch_sampler=sampler, collate_fn=homogeneous_collate)
    sampler = sampler_for(dataset, indices, seed, sampling_strategy)
    return DataLoader(subset, batch_size=batch_size, sampler=sampler, collate_fn=homogeneous_collate, drop_last=False)


def evaluation_loader_for(dataset: JointManifestDataset, indices: list[int], batch_size: int) -> DataLoader:
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                      collate_fn=homogeneous_collate, drop_last=False)


def move(batch: dict, target: torch.device) -> dict:
    return {key: value.to(target) if torch.is_tensor(value) else value for key, value in batch.items()}


def model_mask(batch: dict) -> torch.Tensor:
    """Return the EEG encoder mask, never the legacy duration-derived mask."""
    return batch.get("model_time_mask", batch["time_mask"])


def train_fold_templates(loader: DataLoader, target: torch.device) -> dict[str, torch.Tensor]:
    """Fit target templates from *training* pairs only.

    This is intentionally a loader-level operation rather than an HDF5 global
    statistic: the active split, artifact and any M0/M1 subset are therefore
    exactly the data allowed to define the zero-EEG baseline.
    """
    mfcc_sum = mfcc_sq_sum = mfcc_count = None
    hubert_sum = hubert_count = None
    durations: list[torch.Tensor] = []
    for batch in loader:
        batch = move(batch, target)
        eligible = batch["pairing_weight"] > 0
        if not eligible.any():
            continue
        value = batch["content_mfcc"][eligible]
        mask = batch["content_mask"][eligible].unsqueeze(1).to(value.dtype)
        current_sum = (value * mask).sum(0)
        current_sq = (value.square() * mask).sum(0)
        current_count = mask.sum(0).expand_as(current_sum)
        mfcc_sum = current_sum if mfcc_sum is None else mfcc_sum + current_sum
        mfcc_sq_sum = current_sq if mfcc_sq_sum is None else mfcc_sq_sum + current_sq
        mfcc_count = current_count if mfcc_count is None else mfcc_count + current_count
        if "hubert_local" in batch and batch["hubert_mask"][eligible].any():
            local = batch["hubert_local"][eligible]
            local_mask = batch["hubert_mask"][eligible].unsqueeze(-1).to(local.dtype)
            current_hubert_sum = (local * local_mask).sum((0, 1))
            current_hubert_count = local_mask.sum((0, 1))
            hubert_sum = current_hubert_sum if hubert_sum is None else hubert_sum + current_hubert_sum
            hubert_count = current_hubert_count if hubert_count is None else hubert_count + current_hubert_count
        if "audio_duration_frames" in batch:
            durations.append(batch["audio_duration_frames"][eligible].detach().float().cpu())
    if mfcc_sum is None:
        raise RuntimeError("cannot fit zero-centered template: training selection has no audio pairs")
    mean = mfcc_sum / mfcc_count.clamp_min(1)
    variance = (mfcc_sq_sum / mfcc_count.clamp_min(1) - mean.square()).clamp_min(1e-6)
    template = {"mfcc_mean": mean.detach(), "mfcc_scale": variance.sqrt().detach()}
    if hubert_sum is not None:
        template["hubert_mean"] = (hubert_sum / hubert_count.clamp_min(1)).detach()
    if durations:
        duration = torch.cat(durations).clamp_min(1).log()
        template.update({
            "duration_log_mean": duration.mean().detach(),
            "duration_log_scale": duration.std(unbiased=False).clamp_min(1e-4).detach(),
            "duration_min_frames": duration.exp().min().detach(),
            "duration_max_frames": duration.exp().max().detach(),
        })
    return template


def fit_train_fold_speech_teacher(loader: DataLoader, dimension: int) -> dict[str, object]:
    """Fit a frozen HuBERT whitening/PCA teacher from training content only."""
    if dimension < 1:
        raise ValueError("teacher dimension must be positive")
    grouped: dict[str, list[torch.Tensor]] = {}
    for batch in loader:
        eligible = batch["pairing_weight"] > 0
        if not eligible.any() or "hubert_global" not in batch:
            continue
        for index in eligible.nonzero(as_tuple=False).flatten().tolist():
            label = str(batch["linguistic_content_id"][index])
            grouped.setdefault(label, []).append(batch["hubert_global"][index].detach().float().cpu())
    labels = sorted(grouped)
    if len(labels) < 2:
        raise RuntimeError("frozen speech teacher requires at least two training contents")
    rows = torch.stack([torch.stack(grouped[label]).mean(0) for label in labels])
    mean = rows.mean(0)
    centered = rows - mean
    # Components are fitted over *unique content means*, so subject repetition
    # cannot dominate the audio teacher.  A 50-pair M0 has rank <= 9; pad the
    # requested 64-D contract with zero columns for stable model interfaces.
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    effective = min(int(dimension), int(vh.shape[0]), max(len(labels) - 1, 1))
    components = torch.zeros(rows.shape[1], int(dimension), dtype=rows.dtype)
    components[:, :effective] = vh[:effective].T
    projected = centered @ components
    scale = projected.std(0, unbiased=False).clamp_min(1e-4)
    bank = projected / scale
    return {"mean": mean, "components": components, "scale": scale, "bank": bank, "labels": labels,
            "effective_dimension": effective}


def retrieval_r1(prediction: torch.Tensor, target: torch.Tensor, eligible: torch.Tensor,
                 labels: list[str]) -> float:
    if eligible.sum() < 2:
        return float("nan")
    left = torch.nn.functional.normalize(prediction[eligible].flatten(1), dim=-1)
    right = torch.nn.functional.normalize(target[eligible].flatten(1), dim=-1)
    names = [labels[index] for index in eligible.nonzero(as_tuple=False).flatten().tolist()]
    nearest = (left @ right.T).argmax(1).tolist()
    return float(np.mean([names[index] == names[target_index] for index, target_index in enumerate(nearest)]))


def retrieval_metrics(prediction: torch.Tensor, target: torch.Tensor, labels: list[str]) -> dict[str, float | int]:
    """Multi-positive content retrieval for validation-time model selection."""
    if len(labels) < 2:
        return {"r1": float("nan"), "mrr": float("nan"), "chance_r1": float("nan"), "unique_contents": len(set(labels))}
    left = torch.nn.functional.normalize(prediction.flatten(1), dim=-1)
    right = torch.nn.functional.normalize(target.flatten(1), dim=-1)
    order = (left @ right.T).argsort(1, descending=True)
    positive = torch.tensor([[left_label == right_label for right_label in labels] for left_label in labels],
                            dtype=torch.bool, device=order.device)
    ranked_positive = positive.gather(1, order)
    first = ranked_positive.float().argmax(1) + 1
    return {"r1": float((first == 1).float().mean()), "mrr": float((1.0 / first.float()).mean()),
            "chance_r1": float(positive.float().mean()), "unique_contents": len(set(labels))}


def evaluate_batch(model, batch: dict, target: torch.device) -> dict[str, float]:
    batch = move(batch, target)
    with torch.no_grad():
        state = model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], model_mask(batch), batch["dataset_id"])
        eligible = batch["pairing_weight"] > 0
        correct = torch.nn.functional.l1_loss(state.mfcc[eligible], batch["content_mfcc"][eligible]) if eligible.any() else state.mfcc.new_tensor(float("nan"))
        metrics = {"content_retrieval_r1": retrieval_r1(state.mfcc, batch["content_mfcc"], eligible, batch["linguistic_content_id"]), "correct_mfcc_l1": float(correct)}
        for control in ("zero", "time_shuffle", "channel_shuffle"):
            controlled = counterfactual_eeg(batch["eeg"], control, time_mask=model_mask(batch), channel_mask=batch["channel_mask"])
            output = model(controlled, batch["channel_xyz"], batch["channel_mask"], model_mask(batch), batch["dataset_id"])
            value = torch.nn.functional.l1_loss(output.mfcc[eligible], batch["content_mfcc"][eligible]) if eligible.any() else output.mfcc.new_tensor(float("nan"))
            metrics[f"{control}_mfcc_l1"] = float(value)
    return metrics


def full_content_retrieval(model, loader: DataLoader, target: torch.device) -> float:
    predictions = []; teachers = []; labels = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, target)
            state = model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], model_mask(batch), batch["dataset_id"])
            eligible = batch["pairing_weight"] > 0
            predictions.append(state.mfcc[eligible]); teachers.append(batch["content_mfcc"][eligible])
            labels.extend(batch["linguistic_content_id"][index] for index in eligible.nonzero(as_tuple=False).flatten().tolist())
    if was_training:
        model.train()
    prediction = torch.cat(predictions)
    teacher = torch.cat(teachers)
    return retrieval_r1(prediction, teacher, torch.ones(len(prediction), dtype=torch.bool, device=prediction.device), labels)


def validation_metrics(model, loader: DataLoader, target: torch.device,
                       control_names: tuple[str, ...] = ("zero", "time_shuffle", "channel_shuffle")) -> dict:
    """Evaluate the model-selection fold without touching the locked test fold."""
    predictions = []; teachers = []; labels: list[str] = []
    global_predictions = []; global_teachers = []; durations = []; duration_targets = []
    residual_predictions = []; residual_teachers = []
    controls: dict[str, list[float]] = {name: [] for name in ("correct", *control_names)}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, target)
            eligible = batch["pairing_weight"] > 0
            if not eligible.any():
                continue
            state = model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], model_mask(batch), batch["dataset_id"])
            prediction = state.mfcc[eligible]
            teacher = batch["content_mfcc"][eligible]
            predictions.append(prediction.cpu()); teachers.append(teacher.cpu())
            selected = eligible.nonzero(as_tuple=False).flatten().tolist()
            labels.extend(batch["linguistic_content_id"][index] for index in selected)
            if model.teacher_dimension > 0 and "hubert_global" in batch:
                global_predictions.append(state.global_embedding[eligible].cpu())
                global_teachers.append(model.teacher_target(batch["hubert_global"][eligible]).cpu())
            if state.predicted_duration is not None:
                durations.append(state.predicted_duration[eligible].cpu())
                duration_targets.append(batch["audio_duration_frames"][eligible].float().cpu())
            if state.residual_mfcc is not None and state.baseline_mfcc is not None:
                residual_predictions.append(state.residual_mfcc[eligible].cpu())
                residual_teachers.append((teacher - state.baseline_mfcc[eligible]).cpu())
            controls["correct"].extend((prediction - teacher).abs().mean((1, 2)).cpu().tolist())
            for control in control_names:
                if control == "wrong_trial":
                    if len(batch["eeg"]) < 2:
                        continue
                    order = torch.arange(len(batch["eeg"]), device=target).roll(1)
                    eeg = batch["eeg"][order]
                else:
                    eeg = counterfactual_eeg(batch["eeg"], control, time_mask=model_mask(batch), channel_mask=batch["channel_mask"])
                output = model(eeg, batch["channel_xyz"], batch["channel_mask"], model_mask(batch), batch["dataset_id"])
                controls[control].extend((output.mfcc[eligible] - teacher).abs().mean((1, 2)).cpu().tolist())
    if was_training:
        model.train()
    if not predictions:
        raise RuntimeError("validation fold has no audio-supervised pairs")
    prediction, teacher = torch.cat(predictions), torch.cat(teachers)
    result = {"pairs": len(labels), "mfcc_l1": float((prediction - teacher).abs().mean()),
              "retrieval": retrieval_metrics(prediction, teacher, labels),
              "controls": {name: float(np.mean(values)) if values else float("nan") for name, values in controls.items()}}
    if global_predictions:
        result["hubert_global_retrieval"] = retrieval_metrics(torch.cat(global_predictions), torch.cat(global_teachers), labels)
    if durations:
        left, right = torch.cat(durations), torch.cat(duration_targets)
        result["duration_std_retention"] = float(left.std(unbiased=False) / right.std(unbiased=False).clamp_min(1e-8))
    if residual_predictions:
        left, right = torch.cat(residual_predictions).flatten(1), torch.cat(residual_teachers).flatten(1)
        result["residual_variance_ratio"] = float(left.var(0, unbiased=False).mean() /
                                                    right.var(0, unbiased=False).mean().clamp_min(1e-8))
    return result


def epoch_horizon(*, max_steps: int | None, max_epochs: int | None, training: dict,
                  mode: str, loader_count: int, steps_per_epoch: int) -> tuple[int, int | None]:
    """Resolve legacy step runs or a single-dataset epoch-defined experiment."""
    configured_epochs = training.get("max_epochs")
    if max_steps is not None and max_epochs is not None:
        raise ValueError("--max-steps and --max-epochs are mutually exclusive")
    if max_steps is not None and configured_epochs is not None:
        raise ValueError("--max-steps cannot override a config that declares max_epochs")
    epochs = max_epochs if max_epochs is not None else configured_epochs
    if epochs is None:
        maximum = max_steps or int(training["max_steps"])
        return maximum, None
    if mode == "joint" or loader_count != 1:
        raise RuntimeError("epoch-defined training currently requires exactly one audio-supervised dataset")
    if int(epochs) < 1 or steps_per_epoch < 1:
        raise ValueError("max_epochs and steps_per_epoch must be positive")
    return int(epochs) * steps_per_epoch, int(epochs)


def validation_improved(current: dict, best: dict | None, minimum_delta: float,
                        requirements: dict | None = None) -> bool:
    """Controls are a prerequisite; MRR selects only among control-valid models."""
    requirements = requirements or {}
    selection = str(requirements.get("selection_metric", "mfcc_mrr"))
    key = "hubert_global_retrieval" if selection == "hubert_global_mrr" else "retrieval"
    value = float(current[key]["mrr"])
    if not np.isfinite(value):
        raise RuntimeError("validation retrieval MRR is nonfinite")
    # Preserve the public helper contract used by legacy configurations and
    # focused tests.  v2 always supplies control errors below.
    if "controls" not in current:
        return best is None or value > float(best[key]["mrr"]) + float(minimum_delta)
    errors = current["controls"]
    correct = float(errors["correct"])
    margin = min(float(value) - correct for name, value in errors.items()
                 if name != "correct" and np.isfinite(value))
    current_valid = margin > 0
    if "minimum_residual_variance" in requirements:
        current_valid = current_valid and float(current.get("residual_variance_ratio", 0.0)) >= float(requirements["minimum_residual_variance"])
    if "minimum_duration_std_retention" in requirements:
        current_valid = current_valid and float(current.get("duration_std_retention", 0.0)) >= float(requirements["minimum_duration_std_retention"])
    if best is None:
        return True
    best_errors = best["controls"]
    best_correct = float(best_errors["correct"])
    best_margin = min(float(candidate) - best_correct for name, candidate in best_errors.items()
                      if name != "correct" and np.isfinite(candidate))
    best_valid = best_margin > 0
    if "minimum_residual_variance" in requirements:
        best_valid = best_valid and float(best.get("residual_variance_ratio", 0.0)) >= float(requirements["minimum_residual_variance"])
    if "minimum_duration_std_retention" in requirements:
        best_valid = best_valid and float(best.get("duration_std_retention", 0.0)) >= float(requirements["minimum_duration_std_retention"])
    if current_valid != best_valid:
        return current_valid
    if current_valid:
        return value > float(best[key]["mrr"]) + float(minimum_delta)
    return margin > best_margin + float(minimum_delta)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "joint_pilot_v1.yaml")
    parser.add_argument("--mode", choices=["ds004940", "ds006104", "joint"], required=True)
    parser.add_argument("--stage", choices=["overfit", "generalization"], default="overfit")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-epochs", type=int,
                        help="single-dataset epoch horizon; mutually exclusive with --max-steps")
    parser.add_argument("--batch-size", type=int,
                        help="override batch size; recorded in the resumability contract")
    parser.add_argument("--contents-per-batch", type=int,
                        help="generalization only: content groups per DS004940 batch")
    parser.add_argument("--subjects-per-content", type=int,
                        help="generalization only: same-content subjects per DS004940 batch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-model", action="store_true", help="use a 48-dim/1-layer engineering smoke model")
    parser.add_argument("--explore", action="store_true",
                        help="bypass scientific gates and write only outputs/.../explore artifacts")
    parser.add_argument("--checkpoint-every", type=int,
                        help="write an atomic resumable state after this many optimizer steps")
    parser.add_argument("--output-root", type=Path,
                        help="isolated checkpoint root; defaults to outputs/joint_pilot_v1/<run-kind>")
    parser.add_argument("--restart", action="store_true",
                        help="ignore a compatible partial/final checkpoint and train this run from step 1")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        cfg["training"]["batch_size"] = int(args.batch_size)
    if args.contents_per_batch is not None or args.subjects_per_content is not None:
        if args.stage != "generalization":
            parser.error("content-group batch overrides are only valid for generalization")
        grouped = dict(cfg["training"].get("content_grouped_batch", {}))
        if args.contents_per_batch is not None:
            if args.contents_per_batch < 1:
                parser.error("--contents-per-batch must be positive")
            grouped["contents_per_batch"] = int(args.contents_per_batch)
        if args.subjects_per_content is not None:
            if args.subjects_per_content < 1:
                parser.error("--subjects-per-content must be positive")
            grouped["subjects_per_content"] = int(args.subjects_per_content)
        expected_batch = int(grouped.get("contents_per_batch", 4)) * int(grouped.get("subjects_per_content", 2))
        if int(cfg["training"]["batch_size"]) != expected_batch:
            parser.error("--batch-size must equal --contents-per-batch × --subjects-per-content")
        cfg["training"]["content_grouped_batch"] = grouped
    if args.smoke_model:
        cfg["model"].update({"dimension": 48, "heads": 4, "layers": 1, "local_layers": 1, "dropout": 0.0})
        cfg["training"]["batch_size"] = 2
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    data_cfg_path = resolve(cfg["data_config"], args.config.parent)
    data_cfg = yaml.safe_load(data_cfg_path.read_text())
    # v3 extends v2; output_root is explicitly overridden in the child.
    artifact_root = ROOT / data_cfg["output_root"]
    audit = json.loads((artifact_root / "qc" / "audit.json").read_text())
    expected_counts = cfg.get("audit_expected_included_counts")
    if expected_counts is None and str(data_cfg.get("schema_version", "")).startswith("training-data-v3"):
        expected_counts = {"ds004940": 17489, "ds006104": 10888}
    for dataset_name, expected in (expected_counts or {}).items():
        actual = int(audit.get("included_counts", {}).get(dataset_name, -1))
        if actual != int(expected):
            raise RuntimeError(f"Stage 0 audit gate failed: {dataset_name} expected {expected}, found {actual}")
    for dataset_name in cfg.get("audit_required_nonzero_datasets", []):
        if int(audit.get("included_counts", {}).get(dataset_name, 0)) <= 0:
            raise RuntimeError(f"Stage 0 audit gate failed: {dataset_name} has no included trials")
    if args.stage == "generalization" and not args.explore and bool(cfg["training"].get("stage2_requires_all_m0_gates", True)):
        require_registered_m0_gates(ROOT, cfg)
    stage2 = cfg.get("stage2", {})
    m0 = cfg.get("m0", {})
    split_protocol = cfg["split"]["protocol"] if args.stage == "overfit" else str(
        stage2.get("protocol", "stage2_joint_ood")
    )
    split_path = artifact_root / "splits" / f"{split_protocol}_fold-{cfg['split']['fold']}.csv"
    if args.stage == "overfit":
        artifact_set = str(m0.get("explore_artifact_set", "explore_m0") if args.explore else "built")
    else:
        artifact_set = str(stage2.get(
            "explore_artifact_set" if args.explore else "artifact_set",
            "explore_stage2" if args.explore else "stage2",
        ))
    manifest_path = artifact_root / "manifests" / f"manifest_{artifact_set}.csv"
    if args.stage == "overfit":
        target_name = str(m0.get("explore_target_name", "speech_targets_explore_m0") if args.explore else "speech_targets")
    else:
        target_name = str(stage2.get(
            "explore_target_name" if args.explore else "target_name",
            "speech_targets_explore_stage2" if args.explore else "speech_targets_stage2",
        ))
    target_path = artifact_root / "speech_targets" / f"{target_name}.h5"
    if args.stage == "overfit":
        normalizer_name = str(m0.get("explore_normalizer_name", f"explore_m0_{split_path.stem}") if args.explore else split_path.stem)
    else:
        normalizer_name = str(stage2.get(
            "explore_normalizer_name" if args.explore else "normalizer_name",
            "explore_stage2_joint_ood_fold-0" if args.explore else "stage2_joint_ood_fold-0",
        ))
    normalizer_path = artifact_root / "normalizers" / f"{normalizer_name}.json"
    source_lock_path = artifact_root / "source_lock.json"
    validation_path = artifact_root / "qc" / "validate.json"
    required_paths = [split_path, manifest_path, target_path, normalizer_path, source_lock_path]
    if not args.explore:
        required_paths.append(validation_path)
    for required in required_paths:
        if not required.exists():
            raise RuntimeError(f"required gated artifact is missing: {required}")
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
    if not args.explore and validation.get("status") != "pass":
        raise RuntimeError("Stage 0 validation gate is not passing")
    if args.stage == "overfit" and not args.dry_run and not args.smoke_model and not args.explore:
        blockers = list(validation.get("formal_m0_blockers", []))
        if args.mode == "ds006104":
            blockers = [value for value in blockers if value != "ds004940_human_pair_review"]
        if blockers:
            raise RuntimeError(f"formal M0 is blocked by Stage-0 gates: {blockers}")
    source_lock = json.loads(source_lock_path.read_text())
    artifact_hashes = {path.name: sha256_file(path) for path in
                       (source_lock_path, split_path, manifest_path, target_path, normalizer_path)}
    run_kind = "smoke" if args.smoke_model else ("explore" if args.explore else "pilot")
    checkpoint_root = resolve(args.output_root, ROOT) if args.output_root else ROOT / "outputs" / "joint_pilot_v1" / run_kind
    checkpoint_dir = checkpoint_root / args.stage / args.mode / f"seed-{args.seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_dir / "training_state.pt"
    final_checkpoint_path = checkpoint_dir / "checkpoint.pt"
    current_contract = resume_contract(args=args, cfg=cfg, artifact_hashes=artifact_hashes,
                                       split_protocol=split_protocol, artifact_set=artifact_set,
                                       target_name=target_name, normalizer_name=normalizer_name,
                                       source_lock=source_lock)
    if final_checkpoint_path.exists() and not args.dry_run and not args.restart:
        completed = torch.load(final_checkpoint_path, map_location="cpu", weights_only=False)
        saved_contract = completed.get("resume_contract")
        if saved_contract:
            changed = contract_mismatches(
                saved_contract, current_contract, allow_legacy_explore_control_upgrade=args.explore,
            )
            if changed:
                raise RuntimeError("existing completed checkpoint is incompatible with this run: "
                                   f"{changed}; use --restart to deliberately replace it")
            print(json.dumps({"status": "already_completed", "checkpoint": str(final_checkpoint_path),
                              "steps_completed": completed.get("steps_completed"), "run_kind": run_kind}))
            return 0

    names = [args.mode] if args.mode != "joint" else ["ds004940", "ds006104"]
    datasets = {}; loaders = {}; evaluation_loaders = {}; validation_loaders = {}
    selections = {}
    sampling_strategy = str(cfg["training"].get("sampling_strategy", "weighted_with_replacement"))
    vocabulary = phoneme_vocabulary_from_manifest(manifest_path)
    if len(vocabulary) > int(cfg["model"]["phoneme_classes"]):
        raise RuntimeError(f"phoneme vocabulary has {len(vocabulary)} labels but model has {cfg['model']['phoneme_classes']} classes")
    for name in names:
        dataset = JointManifestDataset(manifest_path, split_path, "train", name, target_path, normalizer_path,
                                       float(cfg["loss"]["weak_content_weight"]),
                                       supervision_types={"paired_audio", "weak_audio"},
                                       phoneme_vocabulary=vocabulary)
        indices = pilot_indices(dataset, cfg, args.stage, "train")
        datasets[name] = dataset
        grouped = (cfg["training"].get("content_grouped_batch")
                   if name == "ds004940" and args.stage == "generalization" else None)
        loaders[name] = loader_for(dataset, indices, int(cfg["training"]["batch_size"]), args.seed,
                                   sampling_strategy, grouped)
        evaluation_loaders[name] = evaluation_loader_for(dataset, indices, int(cfg["training"]["batch_size"]))
        selected = dataset.frame.iloc[indices]
        selections[name] = {"pairs": len(selected), "subjects": int(selected.subject.nunique()),
                            "contents": int(selected.linguistic_content_id.nunique()),
                            "subject_counts": selected.groupby("subject").size().astype(int).to_dict()}

    if args.mode in {"ds006104", "joint"}:
        auxiliary = JointManifestDataset(manifest_path, split_path, "train", "ds006104", target_path, normalizer_path,
                                         float(cfg["loss"]["weak_content_weight"]),
                                         supervision_types={"label_only"}, phoneme_vocabulary=vocabulary)
        aux_indices = auxiliary_indices(auxiliary, cfg, args.stage)
        aux_name = "ds006104_label_only"
        datasets[aux_name] = auxiliary
        loaders[aux_name] = loader_for(auxiliary, aux_indices, int(cfg["training"]["batch_size"]), args.seed + 1009,
                                       sampling_strategy)
        selected = auxiliary.frame.iloc[aux_indices]
        selections[aux_name] = {"pairs": len(selected), "subjects": int(selected.subject.nunique()),
                                "contents": int(selected.linguistic_content_id.nunique()),
                                "labels": int(selected.phoneme_label.nunique()),
                                "subject_counts": selected.groupby("subject").size().astype(int).to_dict()}

    # The same pilot config is shared by M0 and M1.  Early stopping is an M1
    # (generalization) protocol only; M0 is intentionally a fixed-step
    # overfit diagnostic and must not require --max-epochs.
    validation_spec = (cfg["training"].get("validation_early_stopping")
                       if args.stage == "generalization" else None)
    if validation_spec is not None:
        if args.mode == "joint" or len(names) != 1:
            raise RuntimeError("validation_early_stopping requires single-dataset generalization training")
        validation_dataset = JointManifestDataset(
            manifest_path, split_path, "validation", names[0], target_path, normalizer_path,
            float(cfg["loss"]["weak_content_weight"]), supervision_types={"paired_audio", "weak_audio"},
            phoneme_vocabulary=vocabulary,
        )
        validation_indices = pilot_indices(validation_dataset, cfg, args.stage, "validation")
        validation_loaders[names[0]] = evaluation_loader_for(
            validation_dataset, validation_indices, int(cfg["training"]["batch_size"]),
        )
        datasets[f"{names[0]}_validation"] = validation_dataset
        selected = validation_dataset.frame.iloc[validation_indices]
        selections[f"{names[0]}_validation"] = {
            "pairs": len(selected), "subjects": int(selected.subject.nunique()),
            "contents": int(selected.linguistic_content_id.nunique()),
            "subject_counts": selected.groupby("subject").size().astype(int).to_dict(),
        }

    model = JointEEGContentModel(**cfg["model"]).to(device())
    target_templates: dict[str, torch.Tensor] | None = None
    speech_teacher: dict[str, object] | None = None
    if bool(cfg["model"].get("zero_centered", False)):
        target_templates = train_fold_templates(evaluation_loaders[names[0]], device())
        model.set_target_templates(target_templates["mfcc_mean"], target_templates["mfcc_scale"],
                                   target_templates.get("hubert_mean"))
        if bool(cfg["model"].get("duration_standardized", False)):
            required_duration = ("duration_log_mean", "duration_log_scale", "duration_min_frames", "duration_max_frames")
            if not all(key in target_templates for key in required_duration):
                raise RuntimeError("zero-centered v3 model is missing train-fold duration statistics")
            model.set_duration_statistics(*(target_templates[key] for key in required_duration))
    if int(cfg["model"].get("teacher_dimension", 0)) > 0:
        speech_teacher = fit_train_fold_speech_teacher(
            evaluation_loaders[names[0]], int(cfg["model"]["teacher_dimension"]),
        )
        model.set_speech_teacher(speech_teacher["mean"], speech_teacher["components"], speech_teacher["scale"],
                                 speech_teacher["bank"], speech_teacher["labels"])
    dry_batches = {name: move(next(iter(loader)), device()) for name, loader in loaders.items()}
    first_batch = dry_batches[names[0]]
    state = model(first_batch["eeg"], first_batch["channel_xyz"], first_batch["channel_mask"], model_mask(first_batch), first_batch["dataset_id"])
    if state.mfcc.shape[1:] != (39, 161) or not torch.isfinite(state.mfcc).all(): raise RuntimeError("model forward contract failed")
    if args.dry_run:
        by_dataset = {}
        model.zero_grad(set_to_none=True)
        for name in loaders:
            batch = dry_batches[name]
            output = model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], model_mask(batch), batch["dataset_id"])
            loss, metrics = joint_content_loss(output, batch, model, cfg["loss"])
            (loss / len(loaders)).backward()
            by_dataset[name] = {"batch_shape": list(batch["eeg"].shape), "metrics": metrics}
        gradient_finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        print(json.dumps({"status": "pass" if gradient_finite else "fail", "mode": args.mode,
                          "datasets": by_dataset,
                          "selections": selections,
                          "gradient_finite": gradient_finite}, indent=2))
        for dataset in datasets.values(): dataset.close()
        return 0 if gradient_finite else 2

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]),
                                  weight_decay=float(cfg["training"]["weight_decay"]))
    steps_per_epoch = len(loaders[names[0]]) if len(loaders) == 1 else 0
    maximum, maximum_epochs = epoch_horizon(
        max_steps=args.max_steps, max_epochs=args.max_epochs, training=cfg["training"], mode=args.mode,
        loader_count=len(loaders), steps_per_epoch=steps_per_epoch,
    )
    validation_interval_steps: int | None = None
    validation_spec = (cfg["training"].get("validation_early_stopping")
                       if args.stage == "generalization" else None)
    if validation_spec is not None:
        if maximum_epochs is None:
            raise RuntimeError("validation_early_stopping requires max_epochs, not max_steps")
        validation_interval_epochs = int(validation_spec["interval_epochs"])
        if validation_interval_epochs < 1:
            raise ValueError("validation_early_stopping.interval_epochs must be positive")
        validation_interval_steps = validation_interval_epochs * steps_per_epoch
        if maximum % steps_per_epoch:
            raise RuntimeError("epoch-defined maximum must end on an epoch boundary")
    scheduler = learning_rate_scheduler(optimizer, cfg["training"], maximum)
    checkpoint_every = (args.checkpoint_every if args.checkpoint_every is not None
                        else int(cfg["training"].get("checkpoint_interval_steps", 25)))
    if checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be at least 1")
    schedule = list(names)
    if "ds006104_label_only" in loaders:
        interval = int(cfg["training"]["label_only_batch_interval"])
        if interval < 2:
            raise RuntimeError("label_only_batch_interval must be at least 2")
        schedule = [names[index % len(names)] for index in range(interval - 1)] + ["ds006104_label_only"]
    iterator = iter(AlternatingBatchIterator(loaders, schedule))
    history = []
    seen_batch_sources: set[str] = set()
    early_stopped = False
    completed_steps = 0
    completed_epochs = 0
    validation_history: list[dict] = []
    best_validation: dict | None = None
    validation_without_improvement = 0
    best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
    last_checkpoint_path = checkpoint_dir / "last_checkpoint.pt"

    def model_payload(*, checkpoint_kind: str, validation: dict | None = None) -> dict:
        return {
            "model": model.state_dict(), "model_config": cfg["model"], "pilot_config": cfg,
            "mode": args.mode, "stage": args.stage, "seed": args.seed, "run_kind": run_kind,
            "split_protocol": split_protocol, "artifact_set": artifact_set,
            "target_name": target_name, "normalizer_name": normalizer_name,
            "selections": selections, "source_lock_sha256": source_lock["source_lock_sha256"],
            "preprocess_config_sha256": source_lock["config_sha256"],
            "runtime_code_sha256": runtime_code_sha256(), "model_code_sha256": model_code_sha256(),
            "phoneme_vocabulary": vocabulary, "artifact_hashes": artifact_hashes,
            "target_templates": ({key: value.detach().cpu() for key, value in target_templates.items()}
                                 if target_templates is not None else None),
            "speech_teacher": ({key: (value.detach().cpu() if torch.is_tensor(value) else value)
                                for key, value in speech_teacher.items()} if speech_teacher is not None else None),
            "resume_contract": current_contract, "steps_completed": completed_steps,
            "epochs_completed": completed_epochs, "checkpoint_kind": checkpoint_kind,
            "validation_selection": validation,
        }

    def save_training_state(*, interrupted: bool = False, completed: bool = False) -> None:
        payload = {
            "resume_contract": current_contract, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "completed_steps": completed_steps,
            "maximum_steps": maximum, "maximum_epochs": maximum_epochs,
            "steps_per_epoch": steps_per_epoch, "completed_epochs": completed_epochs, "history": history,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "seen_batch_sources": sorted(seen_batch_sources), "early_stopped": early_stopped,
            "validation_history": validation_history, "best_validation": best_validation,
            "validation_without_improvement": validation_without_improvement,
            "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path.exists() else "",
            "batch_schedule": schedule, "target_templates": ({key: value.detach().cpu() for key, value in target_templates.items()}
                                                                 if target_templates is not None else None),
            "speech_teacher": ({key: (value.detach().cpu() if torch.is_tensor(value) else value)
                                for key, value in speech_teacher.items()} if speech_teacher is not None else None),
            "interrupted": interrupted, "completed": completed,
            "python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "mps_rng_state": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        atomic_torch_save(payload, state_path)

    if state_path.exists() and not args.restart:
        previous = torch.load(state_path, map_location="cpu", weights_only=False)
        changed = contract_mismatches(
            previous.get("resume_contract", {}), current_contract,
            allow_legacy_explore_control_upgrade=args.explore,
        )
        if changed:
            raise RuntimeError("partial checkpoint is incompatible with this run: "
                               f"{changed}; use --restart to deliberately discard its progress")
        maximum, preserved_original_maximum = resume_maximum_steps(maximum, previous)
        completed_steps = int(previous.get("completed_steps", 0))
        if int(previous.get("steps_per_epoch", steps_per_epoch)) != steps_per_epoch:
            raise RuntimeError("partial checkpoint has incompatible steps_per_epoch")
        if previous.get("maximum_epochs", maximum_epochs) != maximum_epochs:
            raise RuntimeError("partial checkpoint has incompatible maximum_epochs")
        completed_epochs = int(previous.get("completed_epochs", completed_steps // steps_per_epoch if steps_per_epoch else 0))
        model.load_state_dict(previous["model"])
        saved_templates = previous.get("target_templates")
        if bool(cfg["model"].get("zero_centered", False)):
            if not saved_templates:
                raise RuntimeError("zero-centered resume checkpoint is missing train-fold templates")
            model.set_target_templates(saved_templates["mfcc_mean"], saved_templates["mfcc_scale"],
                                       saved_templates.get("hubert_mean"))
            if bool(cfg["model"].get("duration_standardized", False)):
                required_duration = ("duration_log_mean", "duration_log_scale", "duration_min_frames", "duration_max_frames")
                if not all(key in saved_templates for key in required_duration):
                    raise RuntimeError("v3 resume checkpoint is missing duration statistics")
                model.set_duration_statistics(*(saved_templates[key] for key in required_duration))
        if int(cfg["model"].get("teacher_dimension", 0)) > 0:
            saved_teacher = previous.get("speech_teacher")
            if not saved_teacher:
                raise RuntimeError("v3 resume checkpoint is missing frozen speech teacher")
            model.set_speech_teacher(saved_teacher["mean"], saved_teacher["components"], saved_teacher["scale"],
                                     saved_teacher["bank"], saved_teacher["labels"])
        optimizer.load_state_dict(previous["optimizer"])
        optimizer_to(optimizer, device())
        if scheduler is not None:
            if previous.get("scheduler") is None:
                raise RuntimeError("partial checkpoint is missing the configured learning-rate scheduler state")
            scheduler.load_state_dict(previous["scheduler"])
        history = list(previous.get("history", []))
        seen_batch_sources = set(previous.get("seen_batch_sources", []))
        early_stopped = bool(previous.get("early_stopped", False))
        validation_history = list(previous.get("validation_history", []))
        best_validation = previous.get("best_validation")
        validation_without_improvement = int(previous.get("validation_without_improvement", 0))
        # The sampler has a deterministic private generator. Replaying its
        # consumed batches recreates the next batch without serialising
        # DataLoader, MNE, or HDF5 state.
        for _ in range(completed_steps):
            next(iterator)
        random.setstate(previous["python_random_state"])
        np.random.set_state(previous["numpy_random_state"])
        torch.set_rng_state(previous["torch_rng_state"])
        if torch.backends.mps.is_available() and previous.get("mps_rng_state") is not None:
            torch.mps.set_rng_state(previous["mps_rng_state"])
        if torch.cuda.is_available() and previous.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(previous["cuda_rng_state_all"])
        print(json.dumps({"status": "resumed", "state": str(state_path),
                          "completed_steps": completed_steps, "maximum_steps": maximum,
                          "original_maximum_preserved": preserved_original_maximum}))

    stop_requested: list[int | None] = [None]

    def request_stop(signum, _frame) -> None:
        if stop_requested[0] is not None:
            raise KeyboardInterrupt
        stop_requested[0] = signum
        print("interrupt received: finishing the current optimizer step, then saving resumable state", flush=True)

    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, request_stop)
    interrupted = False
    try:
        for step in range(completed_steps + 1, maximum + 1):
            name, batch = next(iterator)
            batch = move(batch, device())
            optimizer.zero_grad(set_to_none=True)
            state = model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], model_mask(batch), batch["dataset_id"])
            loss, metrics = joint_content_loss(state, batch, model, cfg["loss"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at step {step} ({name}); metrics={metrics}")
            loss.backward()
            nonfinite_gradients = [name for name, parameter in model.named_parameters()
                                   if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
            if nonfinite_gradients:
                raise RuntimeError(f"nonfinite gradients at step {step} ({name}): {nonfinite_gradients[:8]}; metrics={metrics}")
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip"]))
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"nonfinite gradient norm at step {step} ({name}); metrics={metrics}")
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            completed_steps = step
            if steps_per_epoch and completed_steps % steps_per_epoch == 0:
                completed_epochs = completed_steps // steps_per_epoch
            screen_interval = int(cfg["training"].get("train_screen_interval_steps", 100))
            if screen_interval < 1:
                raise ValueError("train_screen_interval_steps must be positive")
            if name not in seen_batch_sources or step % screen_interval == 0:
                seen_batch_sources.add(name)
                history.append({"step": step, "dataset": name,
                                "learning_rate": float(optimizer.param_groups[0]["lr"]), **metrics})
                if step % screen_interval == 0:
                    screen = {dataset_name: full_content_retrieval(model, evaluation_loaders[dataset_name], device())
                              for dataset_name in names}
                    history[-1]["full_content_retrieval_r1"] = screen
                    if args.stage == "overfit" and all(
                        value >= float(cfg["training"]["early_stop_pair_retrieval_r1"]) for value in screen.values()
                    ):
                        early_stopped = True
                print(json.dumps(history[-1]))
            if validation_interval_steps is not None and completed_steps % validation_interval_steps == 0:
                validation = validation_metrics(
                    model, validation_loaders[names[0]], device(),
                    tuple(validation_spec.get("required_controls", ("zero", "time_shuffle", "channel_shuffle"))),
                )
                validation.update({"step": completed_steps, "epoch": completed_epochs,
                                   "learning_rate": float(optimizer.param_groups[0]["lr"])})
                improved = validation_improved(
                    validation, best_validation, float(validation_spec.get("minimum_delta", 0.0)), validation_spec,
                )
                validation["improved"] = improved
                if improved:
                    best_validation = validation
                    validation_without_improvement = 0
                    atomic_torch_save(
                        model_payload(checkpoint_kind="best_validation", validation=validation), best_checkpoint_path,
                    )
                else:
                    validation_without_improvement += 1
                validation["validations_without_improvement"] = validation_without_improvement
                validation_history.append(validation)
                if (completed_epochs >= int(validation_spec["minimum_epochs"])
                        and validation_without_improvement >= int(validation_spec["patience_validations"])):
                    early_stopped = True
                print(json.dumps({"validation": validation, "early_stopped": early_stopped}))
            if completed_steps % checkpoint_every == 0 or early_stopped or stop_requested[0] is not None:
                save_training_state(interrupted=stop_requested[0] is not None)
            if early_stopped or stop_requested[0] is not None:
                interrupted = stop_requested[0] is not None
                break
    except KeyboardInterrupt:
        interrupted = True
        save_training_state(interrupted=True)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)

    if interrupted:
        (checkpoint_dir / "progress.json").write_text(json.dumps({
            "status": "interrupted_resumable", "completed_steps": completed_steps,
            "maximum_steps": maximum, "state": str(state_path), "run_kind": run_kind,
        }, indent=2) + "\n")
        print(json.dumps({"status": "interrupted_resumable", "completed_steps": completed_steps,
                          "resume": "rerun the same command", "state": str(state_path)}))
        for dataset in datasets.values(): dataset.close()
        return 130

    controls = {name: evaluate_batch(model, next(iter(loaders[name])), device()) for name in names}
    atomic_torch_save(model_payload(checkpoint_kind="last", validation=best_validation), last_checkpoint_path)
    if validation_spec is not None:
        if best_validation is None or not best_checkpoint_path.exists():
            raise RuntimeError("epoch-defined training completed without a validation-selected checkpoint")
        selected_payload = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
        selected_kind = "best_validation"
    else:
        selected_payload = model_payload(checkpoint_kind="last", validation=None)
        selected_kind = "last"
    selected_payload["selected_checkpoint_kind"] = selected_kind
    selected_payload["last_steps_completed"] = completed_steps
    atomic_torch_save(selected_payload, final_checkpoint_path)
    save_training_state(completed=True)
    interpretation = ("engineering_only_no_scientific_gate_claim" if args.smoke_model
                      else ("exploratory_only_gates_bypassed_not_registered" if args.explore
                            else "evaluate_against_registered_pilot_gates"))
    (checkpoint_dir / "metrics.json").write_text(json.dumps({"run_kind": run_kind, "interpretation": interpretation,
                                                              "selections": selections, "history": history,
                                                              "controls": controls, "steps_completed": completed_steps,
                                                              "epochs_completed": completed_epochs,
                                                              "steps_per_epoch": steps_per_epoch,
                                                              "maximum_epochs": maximum_epochs,
                                                              "early_stopped": early_stopped,
                                                              "validation_history": validation_history,
                                                              "best_validation": best_validation,
                                                              "selected_checkpoint_kind": selected_kind,
                                                              "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path.exists() else "",
                                                              "last_checkpoint": str(last_checkpoint_path),
                                                              "batch_schedule": schedule,
                                                              "learning_rate_schedule": cfg["training"].get("lr_schedule", "constant"),
                                                              "runtime_code_sha256": runtime_code_sha256(),
                                                              "artifact_hashes": artifact_hashes}, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(final_checkpoint_path), "run_kind": run_kind,
                      "interpretation": interpretation, "selected_checkpoint_kind": selected_kind,
                      "best_validation": best_validation, "controls": controls}, indent=2))
    for dataset in datasets.values(): dataset.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
