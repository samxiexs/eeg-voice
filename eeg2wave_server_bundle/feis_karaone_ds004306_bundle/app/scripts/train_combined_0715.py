from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.combined_0715.data import (  # noqa: E402
    DATASET_IDS,
    AudioCodeBank,
    AudioCodeDataset,
    CombinedEEGDataset,
    load_context,
)
from src.combined_0715.losses import (  # noqa: E402
    code_cross_entropy,
    condition_alignment_loss,
    multi_positive_contrastive_loss,
    soft_label_distillation,
    variance_regularizer,
)
from src.combined_0715.lineage import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
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
    random_code_mask,
)


DATASETS = ("feis", "karaone", "ds004306")
KARAONE_LABELS = ("/diy/", "/iy/", "/m/", "/n/", "/piy/", "/tiy/", "/uw/", "gnaw", "knew", "pat", "pot")
KARAONE_GLOBAL_OFFSET = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the combined FEIS/KaraOne/ds004306 0715-compatible pipeline.")
    parser.add_argument("--phase", required=True, choices=("audio", "eeg", "evaluate"))
    parser.add_argument("--config", default=str(APP_DIR / "configs" / "combined_0715_v1.yaml"))
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--audio-checkpoint", default=None)
    parser.add_argument(
        "--audio-init-checkpoint",
        default=None,
        help=(
            "Supervised KaraOne audio checkpoint used to initialize the combined audio model. "
            "Required for non-scratch audio training; 11-label KaraOne heads are expanded into the combined 30-label space."
        ),
    )
    parser.add_argument(
        "--allow-scratch-audio",
        action="store_true",
        help="Explicitly allow random-init audio training (diagnostic only; not the recommended research pipeline).",
    )
    parser.add_argument("--eeg-checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Limit optimizer steps per epoch for smoke tests.")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--allow-final-test", action="store_true")
    parser.add_argument("--allow-failed-gate", action="store_true")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else APP_DIR / "configs" / path


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


def config_hash(path: Path) -> str:
    return file_sha256(path)


def code_indices_for_rows(context, split: str) -> np.ndarray:
    keys = sorted({row["audio_key"] for row in context.rows if context.split_for(row) == split})
    return np.asarray(keys)


def bank_indices_for_split(context, bank: AudioCodeBank, split: str) -> np.ndarray:
    keys = set(code_indices_for_rows(context, split).tolist())
    return np.asarray([index for index, key in enumerate(bank.keys.tolist()) if key in keys], dtype=np.int64)


def audio_cfg(cfg: dict[str, Any], bank: AudioCodeBank) -> AudioCodeModelConfig:
    settings = cfg["audio_model"]
    return AudioCodeModelConfig(
        codebooks=bank.codes.shape[1],
        code_steps=bank.codes.shape[2],
        vocab_size=int(cfg["codec"]["vocab_size"]),
        num_labels=30,
        d_model=int(settings["d_model"]),
        condition_steps=int(settings["condition_steps"]),
        encoder_layers=int(settings["encoder_layers"]),
        decoder_layers=int(settings["decoder_layers"]),
        heads=int(settings["heads"]),
        dropout=float(settings["dropout"]),
    )


def eeg_cfg(cfg: dict[str, Any], context, bank: AudioCodeBank) -> EEGModelConfig:
    settings = cfg["eeg_model"]
    return EEGModelConfig(
        channels=int(cfg["data"]["channels"]),
        eeg_len=int(cfg["data"]["eeg_len"]),
        d_model=int(settings["d_model"]),
        condition_steps=int(settings["condition_steps"]),
        code_steps=bank.codes.shape[2],
        global_labels=30,
        label_dims=(16, 11, 3),
        num_train_subjects=len(context.subject_to_index),
        transformer_layers=int(settings["transformer_layers"]),
        heads=int(settings["heads"]),
        dropout=float(settings["dropout"]),
        temporal_kernels=tuple(int(value) for value in settings["temporal_kernels"]),
        stem_stride=int(settings["stem_stride"]),
    )


def output_root(cfg: dict[str, Any]) -> Path:
    return resolve(cfg["paths"]["output_root"])


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    model_config: dict[str, Any],
    epoch: int,
    history: list[dict[str, Any]],
    best_score: float,
    phase: str,
    cfg_path: Path,
    lineage: dict[str, Any],
    dependencies: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "epoch": int(epoch),
        "history": history,
        "best_score": float(best_score),
        "phase": phase,
        "lineage": lineage,
        "dependencies": dependencies or {},
        "metadata": metadata or {},
        "config_sha256": config_hash(cfg_path),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }, path)


def resume(
    path: str | None,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    expected_phase: str,
    expected_lineage: dict[str, Any],
    expected_dependencies: dict[str, str] | None = None,
) -> tuple[int, list[dict[str, Any]], float]:
    if not path:
        return 0, [], -float("inf")
    payload = torch.load(path, map_location=device, weights_only=False)
    validate_checkpoint_payload(
        payload,
        expected_phase=expected_phase,
        expected_lineage=expected_lineage,
        expected_dependencies=expected_dependencies,
        source=f"resume checkpoint {path}",
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    return int(payload["epoch"]), list(payload.get("history", [])), float(payload.get("best_score", -float("inf")))


def make_loader(dataset, batch_size: int, cfg: dict[str, Any], *, balanced: bool) -> DataLoader:
    if not balanced:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=int(cfg["run"]["num_workers"]))
    groups = [(row["subject_group_id"], row["label"]) for row in dataset.rows]
    counts: dict[tuple[str, str], int] = {}
    for group in groups:
        counts[group] = counts.get(group, 0) + 1
    weights = torch.tensor([1.0 / counts[group] for group in groups], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=int(cfg["run"]["num_workers"]))


def initialize_audio_model(
    model: AudioCodeAutoencoder,
    architecture: AudioCodeModelConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Initialize the combined audio model from a supervised KaraOne audio checkpoint.

    KaraOne has 11 labels while the combined model has 30 global labels. All
    codec/transformer weights are transferred exactly; the KaraOne label rows
    are copied into the combined KaraOne slice and the FEIS/ds004306 rows keep
    their fresh initialization. This is initialization for fine-tuning, not a
    resume path, so the source checkpoint need not carry combined lineage.
    """

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("phase") != "audio":
        raise ValueError(f"Audio initialization checkpoint must be an audio checkpoint: {checkpoint_path}")
    source_config_raw = payload.get("model_config")
    source_state = payload.get("state_dict")
    if not isinstance(source_config_raw, dict) or not isinstance(source_state, dict):
        raise ValueError(f"Audio initialization checkpoint is missing model_config/state_dict: {checkpoint_path}")
    try:
        source_config = AudioCodeModelConfig(**source_config_raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid audio initialization model_config in {checkpoint_path}: {error}") from error

    shared_fields = (
        "codebooks",
        "code_steps",
        "vocab_size",
        "d_model",
        "condition_steps",
        "encoder_layers",
        "decoder_layers",
        "heads",
        "dropout",
    )
    mismatches = {
        field: {"source": getattr(source_config, field), "target": getattr(architecture, field)}
        for field in shared_fields
        if getattr(source_config, field) != getattr(architecture, field)
    }
    if mismatches:
        raise ValueError(f"Audio initialization architecture mismatch: {json.dumps(mismatches, sort_keys=True)}")
    if source_config.num_labels not in {len(KARAONE_LABELS), architecture.num_labels}:
        raise ValueError(
            f"Unsupported source label count {source_config.num_labels}; expected {len(KARAONE_LABELS)} or {architecture.num_labels}"
        )

    target_state = model.state_dict()
    copied: list[str] = []
    expanded: list[str] = []
    missing: list[str] = []
    unexpected: list[str] = []

    with torch.no_grad():
        for key, target in target_state.items():
            if key not in source_state:
                missing.append(key)
                continue
            source = source_state[key]
            if not torch.is_tensor(source):
                raise ValueError(f"Non-tensor state entry in {checkpoint_path}: {key}")
            if tuple(source.shape) == tuple(target.shape):
                target.copy_(source.to(dtype=target.dtype, device=target.device))
                copied.append(key)
                continue
            label_expansion = {
                "encoder.label_head.1.weight": ("rows",),
                "encoder.label_head.1.bias": ("rows",),
                "decoder.label_embedding": ("rows",),
            }
            if key in label_expansion and source_config.num_labels == len(KARAONE_LABELS):
                expected_source_shape = (len(KARAONE_LABELS),) + tuple(target.shape[1:])
                if tuple(source.shape) != expected_source_shape or target.shape[0] != architecture.num_labels:
                    raise ValueError(f"Unexpected label-head shape for {key}: source={tuple(source.shape)}, target={tuple(target.shape)}")
                target[KARAONE_GLOBAL_OFFSET : KARAONE_GLOBAL_OFFSET + len(KARAONE_LABELS)].copy_(
                    source.to(dtype=target.dtype, device=target.device)
                )
                expanded.append(key)
                continue
            raise ValueError(f"Incompatible state entry {key}: source={tuple(source.shape)}, target={tuple(target.shape)}")

    for key in source_state:
        if key not in target_state:
            unexpected.append(key)
    non_label_missing = [key for key in missing if not key.startswith("encoder.label_head") and key != "decoder.label_embedding"]
    if non_label_missing or unexpected:
        raise ValueError(
            "Audio initialization state is not compatible: "
            f"missing_non_label={non_label_missing[:8]}, unexpected={unexpected[:8]}"
        )
    model.load_state_dict(target_state, strict=True)
    source_numel = sum(int(value.numel()) for value in source_state.values() if torch.is_tensor(value))
    copied_numel = sum(int(target_state[key].numel()) for key in copied)
    return {
        "mode": "karaone_supervised_finetune_init",
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_checkpoint_sha256": file_sha256(checkpoint_path),
        "source_phase": payload.get("phase"),
        "source_epoch": payload.get("epoch"),
        "source_num_labels": int(source_config.num_labels),
        "target_num_labels": int(architecture.num_labels),
        "karaone_label_offset": KARAONE_GLOBAL_OFFSET,
        "copied_keys": len(copied),
        "expanded_label_keys": expanded,
        "missing_label_keys": [key for key in missing if key.startswith("encoder.label_head") or key == "decoder.label_embedding"],
        "source_tensor_numel": source_numel,
        "copied_tensor_numel": copied_numel,
        "non_label_transfer_fraction": float(copied_numel / max(source_numel, 1)),
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def audio_metrics(model: AudioCodeAutoencoder, loader: DataLoader, device: torch.device, context, bank: AudioCodeBank) -> dict[str, float]:
    model.eval()
    totals = {dataset: [0, 0] for dataset in DATASETS}
    q_correct = {dataset: np.zeros(2, dtype=np.float64) for dataset in DATASETS}
    q_total = {dataset: np.zeros(2, dtype=np.float64) for dataset in DATASETS}
    for codes, labels, valid_mask in loader:
        codes, labels, valid_mask = codes.to(device), labels.to(device), valid_mask.to(device)
        encoded = model.encoder(codes)
        logits = model.decoder(codes, valid_mask, encoded["condition"], F.one_hot(labels, 30).float())
        pred = encoded["label_logits"].argmax(dim=-1).cpu().numpy()
        for index, label in enumerate(labels.cpu().numpy()):
            dataset = next(dataset for (dataset, name), value in context.label_to_global.items() if value == int(label))
            local_start = sum((16, 11, 3)[:DATASET_IDS[dataset]])
            local_target = int(label) - local_start
            local_pred = int(pred[index]) - local_start
            totals[dataset][0] += int(local_target == local_pred)
            totals[dataset][1] += 1
            q_pred = logits[index, :2].argmax(dim=-1).cpu().numpy()
            q_target = codes[index, :2].cpu().numpy()
            q_mask = valid_mask[index, :2].cpu().numpy()
            q_correct[dataset] += ((q_pred == q_target) & q_mask).sum(axis=1)
            q_total[dataset] += q_mask.sum(axis=1)
    result = {}
    for dataset in DATASETS:
        result[f"{dataset}_label_accuracy"] = float(totals[dataset][0] / max(totals[dataset][1], 1))
        result[f"{dataset}_q0_accuracy"] = float(q_correct[dataset][0] / max(q_total[dataset][0], 1))
        result[f"{dataset}_q1_accuracy"] = float(q_correct[dataset][1] / max(q_total[dataset][1], 1))
    return result


def train_audio(args, cfg, cfg_path, context, bank, device, lineage) -> Path:
    settings = cfg["audio_model"]
    architecture = audio_cfg(cfg, bank)
    model = AudioCodeAutoencoder(architecture).to(device)
    if args.audio_init_checkpoint:
        init_path = Path(args.audio_init_checkpoint).expanduser().resolve()
        if not init_path.is_file():
            raise FileNotFoundError(
                f"Missing supervised audio initialization checkpoint: {init_path}. "
                "Run KaraOne 0715 prepare+audio first."
            )
        initialization = initialize_audio_model(model, architecture, init_path, device)
        initialization_sha256 = str(initialization["source_checkpoint_sha256"])
        dependencies = {
            "audio_init_checkpoint_sha256": initialization_sha256,
            "audio_init_mode": "karaone_supervised_finetune_init",
        }
        print(
            "[combined audio] initialized from supervised KaraOne audio checkpoint: "
            + json.dumps(initialization, sort_keys=True),
            flush=True,
        )
    elif args.allow_scratch_audio:
        initialization = {
            "mode": "scratch_diagnostic",
            "source_checkpoint": None,
            "source_checkpoint_sha256": "scratch",
            "warning": "This is not the recommended label-grounded fine-tuning path.",
        }
        dependencies = {
            "audio_init_checkpoint_sha256": "scratch",
            "audio_init_mode": "scratch_diagnostic",
        }
        print("[combined audio] WARNING: random-init diagnostic audio training was explicitly enabled", flush=True)
    else:
        raise PermissionError(
            "Combined audio training requires --audio-init-checkpoint pointing to a supervised KaraOne 0715 "
            "audio checkpoint. Use --allow-scratch-audio only for a diagnostic smoke run."
        )
    learning_rate = float(settings.get("finetune_lr", settings["lr"])) if initialization["mode"] == "karaone_supervised_finetune_init" else float(settings["lr"])
    initialization["combined_learning_rate"] = learning_rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=float(settings["weight_decay"]))
    if args.resume and not args.audio_init_checkpoint:
        raise PermissionError(
            "Resuming a fine-tuned audio checkpoint requires the same --audio-init-checkpoint so its SHA256 can be verified."
        )
    start, history, best = resume(
        args.resume,
        model,
        optimizer,
        device,
        expected_phase="audio",
        expected_lineage=lineage,
        expected_dependencies=dependencies,
    )
    train = AudioCodeDataset(bank, bank_indices_for_split(context, bank, "train"), context)
    validation = AudioCodeDataset(bank, bank_indices_for_split(context, bank, "validation"), context)
    train_loader = DataLoader(train, batch_size=int(settings["batch_size"]), shuffle=True, num_workers=int(cfg["run"]["num_workers"]))
    val_loader = DataLoader(validation, batch_size=int(settings["batch_size"]), shuffle=False, num_workers=int(cfg["run"]["num_workers"]))
    weights = torch.tensor(settings["codebook_weights"], device=device, dtype=torch.float32)
    directory = output_root(cfg) / "audio"
    best_path, last_path = directory / "checkpoints/best.pt", directory / "checkpoints/last.pt"
    (directory / "metrics").mkdir(parents=True, exist_ok=True)
    (directory / "metrics/initialization_report.json").write_text(
        json.dumps(initialization, indent=2) + "\n", encoding="utf-8"
    )
    epochs = int(args.epochs or settings["epochs"])
    for epoch in range(start + 1, epochs + 1):
        model.train()
        values = []
        iterator = tqdm(itertools.islice(train_loader, args.max_steps) if args.max_steps else train_loader, desc=f"[combined audio] {epoch}/{epochs}", unit="batch", dynamic_ncols=True)
        for codes, labels, valid_mask in iterator:
            codes, labels, valid_mask = codes.to(device), labels.to(device), valid_mask.to(device)
            mask = random_code_mask(codes, min_ratio=float(settings["mask_ratio_min"]), max_ratio=float(settings["mask_ratio_max"]), full_mask_probability=float(settings["full_mask_probability"])) & valid_mask
            probabilities = F.one_hot(labels, 30).float()
            probabilities *= (torch.rand(len(labels), device=device) >= float(settings["label_dropout"])).float().unsqueeze(1)
            condition_drop = torch.rand(len(labels), device=device) < float(settings["condition_dropout"])
            output = model(codes, mask, probabilities, condition_dropout=condition_drop)
            loss = code_cross_entropy(output["code_logits"], codes, mask, weights)["total"] + float(settings["lambda_label"]) * F.cross_entropy(output["label_logits"], labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), float(cfg["run"]["grad_clip"]))
            optimizer.step()
            values.append(float(loss.detach().cpu()))
            iterator.set_postfix(loss=f"{values[-1]:.4f}")
        metrics = audio_metrics(model, val_loader, device, context, bank)
        score = float(np.mean([metrics[f"{dataset}_label_accuracy"] for dataset in DATASETS])) + metrics["karaone_q0_accuracy"] + metrics["karaone_q1_accuracy"]
        row = {"epoch": epoch, "train_loss": float(np.mean(values)), **metrics, "selection_split": "validation", "test_accessed": False}
        history.append(row)
        if score > best:
            best = score
            save_checkpoint(
                best_path,
                model,
                optimizer,
                model_config=asdict(architecture),
                epoch=epoch,
                history=history,
                best_score=best,
                phase="audio",
                cfg_path=cfg_path,
                lineage=lineage,
                dependencies=dependencies,
                metadata={"audio_initialization": initialization},
            )
        save_checkpoint(
            last_path,
            model,
            optimizer,
            model_config=asdict(architecture),
            epoch=epoch,
            history=history,
            best_score=best,
            phase="audio",
            cfg_path=cfg_path,
            lineage=lineage,
            dependencies=dependencies,
            metadata={"audio_initialization": initialization},
        )
        (directory / "metrics").mkdir(parents=True, exist_ok=True)
        (directory / "metrics/latest.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row), flush=True)
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False) if best_path.is_file() else None
    best_validation = (
        dict(best_payload.get("history", [])[-1])
        if isinstance(best_payload, dict) and best_payload.get("history")
        else {}
    )
    gate = {
        "passed": bool(
            best_validation
            and best_validation["karaone_label_accuracy"] >= 0.50
            and best_validation["feis_label_accuracy"] >= 0.60
        ),
        "best_validation": best_validation,
        "selection_epoch": int(best_payload["epoch"]) if isinstance(best_payload, dict) else None,
        "requirements": "KaraOne>=0.50 and FEIS>=0.60; label-assisted and q0/q1 gates require audit script",
        "lineage": lineage,
        "audio_initialization": initialization,
        "audio_init_checkpoint_sha256": initialization["source_checkpoint_sha256"],
        "audio_checkpoint_sha256": file_sha256(best_path) if best_path.is_file() else None,
    }
    (directory / "metrics/validation_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    return best_path


def subject_adversary_strength(epoch: int, total: int, maximum: float) -> float:
    progress = epoch / max(total, 1)
    return float(maximum) * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def eeg_loss(output, audio_target, batch, dataset, settings, epoch, total_epochs, codebook_weights, audio_model):
    label_loss = F.cross_entropy(output["label_logits"], batch["label_idx"])
    alignment = condition_alignment_loss(output["condition"], audio_target["condition"])["total"]
    if dataset in {"feis", "karaone"}:
        contrastive_result = multi_positive_contrastive_loss(
            output["pooled"],
            audio_target["pooled"],
            batch["label_idx"],
            batch["subject_idx"],
            batch["audio_idx"],
            temperature=float(settings.get("contrastive_temperature", 0.08)),
            cross_subject_weight=(
                float(settings.get("contrastive_cross_subject_weight_feis", 0.25))
                if dataset == "feis"
                else 0.0
            ),
        )
    else:
        zero = output["pooled"].sum() * 0.0
        contrastive_result = {
            "total": zero,
            "eeg_to_audio": zero.detach(),
            "audio_to_eeg": zero.detach(),
            "extra_positive_fraction": zero.detach(),
            "mean_positive_count": zero.detach(),
        }
    valid_code = batch["code_mask"].bool()
    decoder_mask = valid_code
    probabilities = torch.softmax(output["label_logits"], dim=-1)
    code_logits = audio_model.decoder(batch["codes"], decoder_mask, output["condition"], probabilities)
    codebook_weights = codebook_weights.to(code_logits)
    code_loss = code_cross_entropy(code_logits, batch["codes"], valid_code, codebook_weights)["total"]
    if dataset == "karaone":
        code_weight = float(settings["lambda_code_strong"]) * min(1.0, epoch / max(1, int(settings["code_warmup_epochs"])))
        envelope_mask = valid_code[:, 0].float()
        envelope_loss = ((torch.sigmoid(output["envelope_logits"]) - batch["audio_envelope"]) ** 2 * envelope_mask).sum() / envelope_mask.sum().clamp_min(1.0)
        timing_loss = F.smooth_l1_loss(output["onset"], batch["onset"]) + F.smooth_l1_loss(output["duration"], batch["duration"])
        alignment_weight = 1.0
    elif dataset == "feis":
        code_weight = float(settings["lambda_code_feis"]) * min(1.0, epoch / max(1, int(settings["code_warmup_epochs"])))
        code_loss = code_cross_entropy(code_logits, batch["codes"], valid_code, codebook_weights * torch.tensor([1.0, 1.0, 0, 0, 0, 0, 0, 0], device=code_logits.device))["total"]
        envelope_loss = output["pooled"].sum() * 0.0
        timing_loss = output["pooled"].sum() * 0.0
        alignment_weight = float(settings["lambda_alignment_feis"])
    else:
        code_weight = 0.0
        code_loss = output["pooled"].sum() * 0.0
        envelope_loss = output["pooled"].sum() * 0.0
        timing_loss = output["pooled"].sum() * 0.0
        alignment_weight = 0.0
    subject_selected = batch["subject_idx"] >= 0
    subject_loss = F.cross_entropy(output["subject_logits"][subject_selected], batch["subject_idx"][subject_selected]) if subject_selected.any() else output["pooled"].sum() * 0.0
    distill = soft_label_distillation(output["label_logits"], audio_target["label_logits"])
    variance = variance_regularizer(output["condition"])
    total = float(settings["lambda_label"]) * label_loss + alignment_weight * alignment + float(settings["lambda_contrastive"]) * contrastive_result["total"] + code_weight * code_loss + (0.30 if dataset == "karaone" else 0.0) * envelope_loss + (0.15 if dataset == "karaone" else 0.0) * timing_loss + float(settings["lambda_subject"]) * subject_loss + float(settings["lambda_distill"]) * distill + float(settings["lambda_variance"]) * variance
    return total, {
        "label": label_loss.detach(),
        "alignment": alignment.detach(),
        "code": code_loss.detach(),
        "subject": subject_loss.detach(),
        "contrastive": contrastive_result["total"].detach(),
        "contrastive_eeg_to_audio": contrastive_result["eeg_to_audio"],
        "contrastive_audio_to_eeg": contrastive_result["audio_to_eeg"],
        "contrastive_extra_positive_fraction": contrastive_result["extra_positive_fraction"],
        "contrastive_mean_positive_count": contrastive_result["mean_positive_count"],
    }


def load_audio_model(path: Path, bank: AudioCodeBank, device: torch.device, lineage: dict[str, Any]) -> tuple[AudioCodeAutoencoder, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    validate_checkpoint_payload(
        payload,
        expected_phase="audio",
        expected_lineage=lineage,
        source=f"audio checkpoint {path}",
    )
    model = AudioCodeAutoencoder(AudioCodeModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


def train_eeg(args, cfg, cfg_path, context, bank, device, lineage) -> Path:
    settings = cfg["eeg_model"]
    audio_path = Path(args.audio_checkpoint) if args.audio_checkpoint else output_root(cfg) / "audio/checkpoints/best.pt"
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio checkpoint: {audio_path}")
    audio_gate = output_root(cfg) / "audio/metrics/validation_gate.json"
    audio_checkpoint_sha256 = file_sha256(audio_path)
    gate_error: Exception | None = None
    if not audio_gate.exists():
        gate_error = FileNotFoundError(f"Audio validation gate is missing: {audio_gate}")
    else:
        try:
            audio_gate_payload = json.loads(audio_gate.read_text(encoding="utf-8"))
            if not isinstance(audio_gate_payload, dict):
                raise ValueError("Audio validation gate must be a JSON object")
            if not bool(audio_gate_payload.get("passed")):
                raise PermissionError("Audio validation gate is not passed")
            validate_lineage(audio_gate_payload.get("lineage"), lineage, source="audio validation gate")
            if audio_gate_payload.get("audio_checkpoint_sha256") != audio_checkpoint_sha256:
                raise PermissionError(
                    "Audio validation gate checkpoint mismatch: "
                    f"gate={audio_gate_payload.get('audio_checkpoint_sha256')!r}, "
                    f"current={audio_checkpoint_sha256!r}"
                )
        except (OSError, json.JSONDecodeError, PermissionError, ValueError) as error:
            gate_error = error
    if gate_error is not None:
        if not args.allow_failed_gate:
            raise PermissionError(
                f"Audio validation gate rejected this checkpoint: {gate_error}. "
                "Use --allow-failed-gate only for exploratory runs."
            ) from gate_error
        print(f"[combined EEG] exploratory audio-gate bypass: {gate_error}", flush=True)
    audio_model, _ = load_audio_model(audio_path, bank, device, lineage)
    audio_model.eval()
    for parameter in audio_model.parameters():
        parameter.requires_grad_(False)
    architecture = eeg_cfg(cfg, context, bank)
    model = EEGConditionEncoder(architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["lr"]), weight_decay=float(settings["weight_decay"]))
    dependencies = {"audio_checkpoint_sha256": audio_checkpoint_sha256}
    start, history, best = resume(
        args.resume,
        model,
        optimizer,
        device,
        expected_phase="eeg",
        expected_lineage=lineage,
        expected_dependencies=dependencies,
    )
    loaders = {}
    for dataset in DATASETS:
        train = CombinedEEGDataset(context, bank, dataset, "train", eeg_len=int(cfg["data"]["eeg_len"]))
        loaders[dataset] = make_loader(train, int(settings["batch_size"]), cfg, balanced=True)
    codebook_weights = torch.tensor(settings["codebook_weights"], device=device, dtype=torch.float32)
    directory = output_root(cfg) / "eeg"
    best_path, last_path = directory / "checkpoints/best.pt", directory / "checkpoints/last.pt"
    epochs = int(args.epochs or settings["epochs"])
    for epoch in range(start + 1, epochs + 1):
        model.train()
        iterators = {dataset: iter(loader) for dataset, loader in loaders.items()}
        steps = max(len(loader) for loader in loaders.values())
        if args.max_steps:
            steps = min(steps, int(args.max_steps))
        losses = []
        component_values: dict[str, list[float]] = {}
        for _ in tqdm(range(steps), desc=f"[combined EEG] {epoch}/{epochs}", unit="step", dynamic_ncols=True):
            optimizer.zero_grad(set_to_none=True)
            total_loss = None
            for dataset in DATASETS:
                try:
                    batch = next(iterators[dataset])
                except StopIteration:
                    iterators[dataset] = iter(loaders[dataset])
                    batch = next(iterators[dataset])
                batch = move_batch(batch, device)
                with torch.no_grad():
                    audio_target = audio_model.encoder(batch["codes"])
                eeg = model.augment(batch["eeg"].clone(), channel_dropout=float(settings["channel_dropout"]), time_mask_ratio=float(settings["time_mask_ratio"]), noise_std=float(settings["noise_std"]))
                output = model(eeg, batch["eeg_valid_len"], batch["dataset_idx"], subject_adversary_strength=subject_adversary_strength(epoch, epochs, float(settings["subject_adversary_max"])))
                loss, components = eeg_loss(output, audio_target, batch, dataset, settings, epoch, epochs, codebook_weights, audio_model)
                for name, value in components.items():
                    component_values.setdefault(f"{dataset}_{name}", []).append(float(value.cpu()))
                weight = float(settings["dataset_weights"][dataset])
                total_loss = loss * weight if total_loss is None else total_loss + loss * weight
            total_loss.backward()
            clip_grad_norm_(model.parameters(), float(cfg["run"]["grad_clip"]))
            optimizer.step()
            losses.append(float(total_loss.detach().cpu()))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "selection_split": "validation",
            "test_accessed": False,
            **{name: float(np.mean(values)) for name, values in component_values.items()},
        }
        history.append(row)
        score = -row["train_loss"]
        if score > best:
            best = score
            save_checkpoint(
                best_path,
                model,
                optimizer,
                model_config=asdict(architecture),
                epoch=epoch,
                history=history,
                best_score=best,
                phase="eeg",
                cfg_path=cfg_path,
                lineage=lineage,
                dependencies=dependencies,
            )
        save_checkpoint(
            last_path,
            model,
            optimizer,
            model_config=asdict(architecture),
            epoch=epoch,
            history=history,
            best_score=best,
            phase="eeg",
            cfg_path=cfg_path,
            lineage=lineage,
            dependencies=dependencies,
        )
        print(json.dumps(row), flush=True)
    gate = {
        "passed": False,
        "reason": "run validation phase before unlocking test",
        "checkpoint": str(best_path),
        "audio_checkpoint_sha256": audio_checkpoint_sha256,
        "eeg_checkpoint_sha256": file_sha256(best_path) if best_path.is_file() else None,
        "lineage": lineage,
        "test_accessed": False,
    }
    (directory / "metrics").mkdir(parents=True, exist_ok=True)
    (directory / "metrics/validation_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    return best_path


def evaluate(args, cfg, cfg_path, context, bank, device, lineage) -> Path:
    if args.split == "test" and not args.allow_final_test:
        raise PermissionError("Locked test requires --allow-final-test")
    if args.split == "test" and args.allow_failed_gate:
        raise PermissionError("--allow-failed-gate cannot bypass the locked test")
    eeg_path = Path(args.eeg_checkpoint) if args.eeg_checkpoint else output_root(cfg) / "eeg/checkpoints/best.pt"
    audio_path = Path(args.audio_checkpoint) if args.audio_checkpoint else output_root(cfg) / "audio/checkpoints/best.pt"
    if not eeg_path.exists() or not audio_path.exists():
        raise FileNotFoundError("Both audio and EEG checkpoints are required")
    audio_checkpoint_sha256 = file_sha256(audio_path)
    eeg_checkpoint_sha256 = file_sha256(eeg_path)
    if args.split == "test":
        gate_path = output_root(cfg) / "eeg/metrics/validation_gate.json"
        if not gate_path.is_file():
            raise PermissionError("Validation gate is missing; locked test remains unavailable")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        validate_gate_binding(
            gate,
            lineage=lineage,
            audio_checkpoint_sha256=audio_checkpoint_sha256,
            eeg_checkpoint_sha256=eeg_checkpoint_sha256,
        )
    audio_model, _ = load_audio_model(audio_path, bank, device, lineage)
    audio_model.eval()
    eeg_payload = torch.load(eeg_path, map_location=device, weights_only=False)
    validate_checkpoint_payload(
        eeg_payload,
        expected_phase="eeg",
        expected_lineage=lineage,
        expected_dependencies={"audio_checkpoint_sha256": audio_checkpoint_sha256},
        source=f"EEG checkpoint {eeg_path}",
    )
    eeg_model = EEGConditionEncoder(EEGModelConfig(**eeg_payload["model_config"])).to(device)
    eeg_model.load_state_dict(eeg_payload["state_dict"], strict=True)
    eeg_model.eval()
    metrics = {}
    for dataset in DATASETS:
        data = CombinedEEGDataset(context, bank, dataset, args.split, eeg_len=int(cfg["data"]["eeg_len"]))
        loader = DataLoader(data, batch_size=int(cfg["eeg_model"]["batch_size"]), shuffle=False, num_workers=int(cfg["run"]["num_workers"]))
        targets, predictions, gains = [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = move_batch(batch, device)
                output = eeg_model(batch["eeg"], batch["eeg_valid_len"], batch["dataset_idx"])
                targets.extend(batch["label_local"].cpu().tolist())
                start = sum((16, 11, 3)[:DATASET_IDS[dataset]])
                predictions.extend((output["label_logits"].argmax(dim=-1) - start).cpu().tolist())
                probs = torch.softmax(output["label_logits"], dim=-1)
                eeg_logits = audio_model.decoder(batch["codes"], batch["code_mask"], output["condition"], probs)
                label_logits = audio_model.decoder(batch["codes"], batch["code_mask"], torch.zeros_like(output["condition"]), probs)
                mask = batch["code_mask"][:, :2]
                gains.append(float((((eeg_logits[:, :2].argmax(-1) == batch["codes"][:, :2]) & mask).float().sum() - ((label_logits[:, :2].argmax(-1) == batch["codes"][:, :2]) & mask).float().sum()).cpu()))
        recalls = [float(np.mean(np.asarray(predictions)[np.asarray(targets) == label] == label)) for label in sorted(set(targets)) if np.any(np.asarray(targets) == label)]
        metrics[dataset] = {"n_trials": len(data), "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"), "coarse_code_gain_sum": float(np.sum(gains)), "split": args.split}
    report = {
        "version": "combined-0715-v1",
        "split": args.split,
        "metrics": metrics,
        "config_sha256": config_hash(cfg_path),
        "lineage": lineage,
        "audio_checkpoint_sha256": audio_checkpoint_sha256,
        "eeg_checkpoint_sha256": eeg_checkpoint_sha256,
        "test_accessed": args.split == "test",
    }
    destination = output_root(cfg) / "eeg/metrics" / f"{args.split}_evaluation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.split == "validation":
        gate = {
            "passed": False,
            "manual_review_required": True,
            "reason": "Review per-dataset BA, bootstrap CI, shuffled/zero controls, and KaraOne q0/q1 gain before setting passed=true",
            "validation_report": str(destination),
            "validation_report_sha256": file_sha256(destination),
            "lineage": lineage,
            "audio_checkpoint_sha256": audio_checkpoint_sha256,
            "eeg_checkpoint_sha256": eeg_checkpoint_sha256,
            "test_accessed": False,
        }
        (destination.parent / "validation_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return destination


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if args.phase == "evaluate" and args.split == "test":
        if not args.allow_final_test:
            raise PermissionError("Locked test requires --allow-final-test")
        if args.allow_failed_gate:
            raise PermissionError("--allow-failed-gate cannot bypass the locked test")
        audio_path = Path(args.audio_checkpoint) if args.audio_checkpoint else output_root(cfg) / "audio/checkpoints/best.pt"
        eeg_path = Path(args.eeg_checkpoint) if args.eeg_checkpoint else output_root(cfg) / "eeg/checkpoints/best.pt"
        preauthorize_locked_test(
            output_root(cfg) / "eeg/metrics/validation_gate.json",
            config_path=cfg_path,
            audio_checkpoint_path=audio_path,
            eeg_checkpoint_path=eeg_path,
        )
    context = load_context(cfg_path)
    bank = AudioCodeBank(args.cache)
    print("[combined-0715-v1] hashing config/split/manifest/preprocessed EEG/cache lineage", flush=True)
    lineage = build_run_lineage(cfg_path, context, bank)
    device = torch.device(args.device) if args.device else default_device()
    set_seed(int(cfg["run"]["seed"]))
    print(f"[combined-0715-v1] phase={args.phase}; device={device}; audio_keys={len(bank.keys)}", flush=True)
    if args.phase == "audio":
        train_audio(args, cfg, cfg_path, context, bank, device, lineage)
    elif args.phase == "eeg":
        train_eeg(args, cfg, cfg_path, context, bank, device, lineage)
    else:
        evaluate(args, cfg, cfg_path, context, bank, device, lineage)


if __name__ == "__main__":
    main()
