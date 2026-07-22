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
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
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
    multi_scale_envelope_correlation_loss,
    multi_positive_contrastive_loss,
    same_label_morphology_ranking_loss,
    soft_activity_dice_loss,
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
LABEL_SLICES = {"feis": (0, 16), "karaone": (16, 27), "ds004306": (27, 30)}
ENVELOPE_CORRELATION_KERNELS = (1, 5, 9)


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
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override the artifact directory without changing the lineage-bound model config.",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help=(
            "EEG phase only: save an inference candidate at epoch 1, every N epochs, and the final epoch. "
            "Use 0 to disable periodic candidates. Candidates are intended for decoded-validation selection."
        ),
    )
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


def output_root(cfg: dict[str, Any], override: str | Path | None = None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
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
    include_training_state: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "state_dict": model.state_dict(),
        "model_config": model_config,
        "epoch": int(epoch),
        "history": history,
        "best_score": float(best_score),
        "phase": phase,
        "lineage": lineage,
        "dependencies": dependencies or {},
        "metadata": metadata or {},
        "config_sha256": config_hash(cfg_path),
    }
    if include_training_state:
        payload.update({
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        })
    torch.save(payload, path)


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


class SameLabelPairBatchSampler(Sampler[list[int]]):
    """Build label-balanced batches containing different-audio same-label pairs.

    The 0721 morphology-ranking loss needs an in-batch counterfactual with the
    same KaraOne class but a different recording.  Ordinary weighted sampling
    leaves that loss inactive in many small batches, so each pair is sampled
    explicitly while retaining inverse-subject weighting within every label.
    """

    def __init__(self, rows, batch_size: int, *, seed: int) -> None:
        if batch_size < 2:
            raise ValueError("SameLabelPairBatchSampler requires batch_size >= 2")
        self.rows = tuple(rows)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.iteration = 0
        self.num_batches = math.ceil(len(self.rows) / self.batch_size)
        self.groups: dict[str, list[int]] = {}
        for index, row in enumerate(self.rows):
            self.groups.setdefault(str(row["label"]), []).append(index)
        self.groups = {
            label: indices
            for label, indices in self.groups.items()
            if len({str(self.rows[index]["audio_key"]) for index in indices}) >= 2
        }
        if not self.groups:
            raise ValueError("No KaraOne label contains at least two distinct audio recordings")
        self.labels = tuple(sorted(self.groups))

    def __len__(self) -> int:
        return self.num_batches

    def _choice(self, rng: np.random.Generator, indices: list[int]) -> int:
        subject_counts: dict[str, int] = {}
        for index in indices:
            subject = str(self.rows[index]["subject_group_id"])
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
        weights = np.asarray(
            [1.0 / subject_counts[str(self.rows[index]["subject_group_id"])] for index in indices],
            dtype=np.float64,
        )
        weights /= weights.sum()
        return int(rng.choice(np.asarray(indices, dtype=np.int64), p=weights))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.iteration)
        self.iteration += 1
        for _ in range(self.num_batches):
            batch: list[int] = []
            while len(batch) + 1 < self.batch_size:
                label = self.labels[int(rng.integers(len(self.labels)))]
                indices = self.groups[label]
                first = self._choice(rng, indices)
                first_audio = str(self.rows[first]["audio_key"])
                alternatives = [
                    index for index in indices if str(self.rows[index]["audio_key"]) != first_audio
                ]
                second = self._choice(rng, alternatives)
                batch.extend((first, second))
            if len(batch) < self.batch_size:
                # Odd batch sizes retain one extra balanced example; every
                # complete pair above still activates morphology ranking.
                label = self.labels[int(rng.integers(len(self.labels)))]
                batch.append(self._choice(rng, self.groups[label]))
            yield batch


def make_loader(
    dataset,
    batch_size: int,
    cfg: dict[str, Any],
    *,
    balanced: bool,
    ensure_same_label_pairs: bool = False,
) -> DataLoader:
    if not balanced:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=int(cfg["run"]["num_workers"]))
    if ensure_same_label_pairs:
        batch_sampler = SameLabelPairBatchSampler(
            dataset.rows,
            batch_size,
            seed=int(cfg["run"].get("seed", 715)),
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=int(cfg["run"]["num_workers"]),
        )
    groups = [(row["subject_group_id"], row["label"]) for row in dataset.rows]
    counts: dict[tuple[str, str], int] = {}
    for group in groups:
        counts[group] = counts.get(group, 0) + 1
    weights = torch.tensor([1.0 / counts[group] for group in groups], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=int(cfg["run"]["num_workers"]))


def make_audio_loader(dataset: AudioCodeDataset, batch_size: int, cfg: dict[str, Any], *, balanced: bool) -> DataLoader:
    """Load audio examples with explicit dataset/label-balanced sampling.

    The cache contains 1616 unique KaraOne audio files but only 288 FEIS files
    and 3 ds004306 category files. Plain shuffle therefore makes the shared
    audio model optimize almost entirely for KaraOne. The weighted sampler
    keeps the requested dataset proportions while making labels within each
    dataset equally likely.
    """

    if not balanced:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=int(cfg["run"]["num_workers"]))
    settings = cfg["audio_model"]
    target_dataset_weights = {
        "feis": 0.35,
        "karaone": 0.55,
        "ds004306": 0.10,
        **{str(key): float(value) for key, value in settings.get("dataset_sampling_weights", {}).items()},
    }
    groups: list[tuple[str, int]] = [
        (str(dataset.bank.datasets[int(index)]), int(label))
        for index, label in zip(dataset.indices_array.tolist(), dataset.labels.tolist())
    ]
    counts: dict[tuple[str, int], int] = {}
    for group in groups:
        counts[group] = counts.get(group, 0) + 1
    label_counts = {
        dataset_name: max(1, len({label for name, label in groups if name == dataset_name}))
        for dataset_name, _ in groups
    }
    weights = torch.tensor(
        [
            target_dataset_weights.get(dataset_name, 1.0)
            / label_counts[dataset_name]
            / max(counts[(dataset_name, label)], 1)
            for dataset_name, label in groups
        ],
        dtype=torch.double,
    )
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
    train_loader = make_audio_loader(train, int(settings["batch_size"]), cfg, balanced=True)
    val_loader = make_audio_loader(validation, int(settings["batch_size"]), cfg, balanced=False)
    weights = torch.tensor(settings["codebook_weights"], device=device, dtype=torch.float32)
    directory = output_root(cfg, args.output_root) / "audio"
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


def safe_vector_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    n = min(len(first), len(second))
    if n < 2:
        return 0.0
    first = first[:n] - first[:n].mean()
    second = second[:n] - second[:n].mean()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float((first @ second) / denominator) if denominator > 1e-12 else 0.0


def multi_scale_vector_correlation(
    first: np.ndarray,
    second: np.ndarray,
    *,
    kernel_sizes: tuple[int, ...] = ENVELOPE_CORRELATION_KERNELS,
) -> float:
    """Validation analogue of the differentiable multi-scale loss."""

    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    correlations: list[float] = []
    for size in kernel_sizes:
        if size == 1:
            first_scale, second_scale = first, second
        else:
            kernel = np.ones(int(size), dtype=np.float64) / float(size)
            first_scale = np.convolve(first, kernel, mode="same")
            second_scale = np.convolve(second, kernel, mode="same")
        correlations.append(safe_vector_correlation(first_scale, second_scale))
    return float(np.mean(correlations)) if correlations else 0.0


def eeg_selection_metrics(model, loader, device, *, label_offset: int) -> dict[str, float]:
    """Cheap validation morphology metrics used for EEG checkpoint selection.

    This evaluates the learned KaraOne envelope/timing heads without running
    MaskGIT or EnCodec, so it is practical after every training epoch.
    """

    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    envelope_correlations: list[float] = []
    onset_errors: list[float] = []
    duration_errors: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(batch["eeg"], batch["eeg_valid_len"], batch["dataset_idx"])
            targets.extend(batch["label_local"].cpu().tolist())
            predictions.extend((output["label_logits"].argmax(dim=-1) - int(label_offset)).cpu().tolist())
            predicted_envelope = torch.sigmoid(output["envelope_logits"]).cpu().numpy()
            target_envelope = batch["audio_envelope"].cpu().numpy()
            valid = batch["code_mask"][:, 0].bool().cpu().numpy()
            for predicted, target, mask in zip(predicted_envelope, target_envelope, valid, strict=True):
                envelope_correlations.append(
                    multi_scale_vector_correlation(predicted[mask], target[mask])
                )
            onset_errors.extend(torch.abs(output["onset"] - batch["onset"]).detach().cpu().tolist())
            duration_errors.extend(torch.abs(output["duration"] - batch["duration"]).detach().cpu().tolist())
    target_array = np.asarray(targets)
    prediction_array = np.asarray(predictions)
    recalls = [
        float(np.mean(prediction_array[target_array == label] == label))
        for label in sorted(set(targets))
        if np.any(target_array == label)
    ]
    return {
        "selection_karaone_label_balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "selection_karaone_envelope_correlation_median": float(np.median(envelope_correlations)) if envelope_correlations else 0.0,
        "selection_karaone_onset_mae": float(np.mean(onset_errors)) if onset_errors else 1.0,
        "selection_karaone_duration_mae": float(np.mean(duration_errors)) if duration_errors else 1.0,
    }


def effective_eeg_loss_recipe(settings: dict[str, Any]) -> dict[str, float]:
    """Resolve the 0721v1 loss recipe without mutating the audio-bound YAML."""

    return {
        "lambda_label": float(settings["lambda_label"]),
        "lambda_alignment_karaone": float(settings.get("lambda_alignment", 1.0)),
        "lambda_alignment_feis": float(settings["lambda_alignment_feis"]),
        "lambda_alignment_ds004306": float(settings.get("lambda_alignment_ds", 0.0)),
        "lambda_contrastive": float(settings["lambda_contrastive"]),
        "lambda_code_karaone": float(settings["lambda_code_strong"]),
        "lambda_code_feis": float(settings["lambda_code_feis"]),
        "lambda_envelope_mse": float(settings.get("lambda_envelope_mse", 0.30)),
        "lambda_envelope_correlation": float(settings.get("lambda_envelope_correlation", 0.35)),
        "lambda_activity_dice": float(settings.get("lambda_activity_dice", 0.20)),
        "lambda_timing": float(settings.get("lambda_timing", 0.15)),
        "lambda_morphology_ranking": float(settings.get("lambda_morphology_ranking", 0.15)),
        "morphology_ranking_margin": float(settings.get("morphology_ranking_margin", 0.03)),
        "activity_threshold": float(settings.get("activity_threshold", 0.10)),
        "activity_temperature": float(settings.get("activity_temperature", 0.05)),
        "lambda_subject": float(settings["lambda_subject"]),
        "lambda_distill": float(settings["lambda_distill"]),
        "lambda_variance": float(settings["lambda_variance"]),
    }


def eeg_loss(output, audio_target, batch, dataset, settings, epoch, total_epochs, codebook_weights, audio_model):
    recipe = effective_eeg_loss_recipe(settings)
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
    zero = output["pooled"].sum() * 0.0
    if dataset == "karaone":
        code_weight = recipe["lambda_code_karaone"] * min(1.0, epoch / max(1, int(settings["code_warmup_epochs"])))
        envelope_mask = valid_code[:, 0].float()
        predicted_envelope = torch.sigmoid(output["envelope_logits"])
        envelope_mse = ((predicted_envelope - batch["audio_envelope"]) ** 2 * envelope_mask).sum() / envelope_mask.sum().clamp_min(1.0)
        envelope_correlation_result = multi_scale_envelope_correlation_loss(
            predicted_envelope,
            batch["audio_envelope"],
            envelope_mask,
            kernel_sizes=ENVELOPE_CORRELATION_KERNELS,
        )
        activity_result = soft_activity_dice_loss(
            predicted_envelope,
            batch["audio_envelope"],
            envelope_mask,
            threshold=recipe["activity_threshold"],
            temperature=recipe["activity_temperature"],
        )
        ranking_result = same_label_morphology_ranking_loss(
            predicted_envelope,
            batch["audio_envelope"],
            envelope_mask,
            batch["label_idx"],
            batch["audio_idx"],
            margin=recipe["morphology_ranking_margin"],
        )
        timing_loss = F.smooth_l1_loss(output["onset"], batch["onset"]) + F.smooth_l1_loss(output["duration"], batch["duration"])
        alignment_weight = recipe["lambda_alignment_karaone"]
    elif dataset == "feis":
        code_weight = recipe["lambda_code_feis"] * min(1.0, epoch / max(1, int(settings["code_warmup_epochs"])))
        code_loss = code_cross_entropy(code_logits, batch["codes"], valid_code, codebook_weights * torch.tensor([1.0, 1.0, 0, 0, 0, 0, 0, 0], device=code_logits.device))["total"]
        envelope_mse = zero
        envelope_correlation_result = {"total": zero, "correlation": zero.detach()}
        activity_result = {"total": zero, "dice": zero.detach()}
        ranking_result = {
            "total": zero,
            "active_fraction": zero.detach(),
            "correct_correlation": zero.detach(),
            "shuffled_correlation": zero.detach(),
        }
        timing_loss = zero
        alignment_weight = recipe["lambda_alignment_feis"]
    else:
        code_weight = 0.0
        code_loss = zero
        envelope_mse = zero
        envelope_correlation_result = {"total": zero, "correlation": zero.detach()}
        activity_result = {"total": zero, "dice": zero.detach()}
        ranking_result = {
            "total": zero,
            "active_fraction": zero.detach(),
            "correct_correlation": zero.detach(),
            "shuffled_correlation": zero.detach(),
        }
        timing_loss = zero
        alignment_weight = recipe["lambda_alignment_ds004306"]
    subject_selected = batch["subject_idx"] >= 0
    subject_loss = F.cross_entropy(output["subject_logits"][subject_selected], batch["subject_idx"][subject_selected]) if subject_selected.any() else output["pooled"].sum() * 0.0
    label_start, label_stop = LABEL_SLICES[dataset]
    distill = soft_label_distillation(
        output["label_logits"][:, label_start:label_stop],
        audio_target["label_logits"][:, label_start:label_stop],
    )
    variance = variance_regularizer(output["condition"])
    total = (
        recipe["lambda_label"] * label_loss
        + alignment_weight * alignment
        + recipe["lambda_contrastive"] * contrastive_result["total"]
        + code_weight * code_loss
        + recipe["lambda_envelope_mse"] * envelope_mse
        + recipe["lambda_envelope_correlation"] * envelope_correlation_result["total"]
        + recipe["lambda_activity_dice"] * activity_result["total"]
        + recipe["lambda_timing"] * timing_loss
        + recipe["lambda_morphology_ranking"] * ranking_result["total"]
        + recipe["lambda_subject"] * subject_loss
        + recipe["lambda_distill"] * distill
        + recipe["lambda_variance"] * variance
    )
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
        "envelope_mse": envelope_mse.detach(),
        "envelope_correlation_loss": envelope_correlation_result["total"].detach(),
        "envelope_correlation": envelope_correlation_result["correlation"],
        "envelope_correlation_k1": envelope_correlation_result.get("correlation_k1", zero.detach()),
        "envelope_correlation_k5": envelope_correlation_result.get("correlation_k5", zero.detach()),
        "envelope_correlation_k9": envelope_correlation_result.get("correlation_k9", zero.detach()),
        "activity_dice_loss": activity_result["total"].detach(),
        "activity_dice": activity_result["dice"],
        "timing": timing_loss.detach(),
        "morphology_ranking": ranking_result["total"].detach(),
        "morphology_ranking_active_fraction": ranking_result["active_fraction"],
        "morphology_correct_correlation": ranking_result["correct_correlation"],
        "morphology_shuffled_correlation": ranking_result["shuffled_correlation"],
        "distill": distill.detach(),
        "variance": variance.detach(),
    }


def load_audio_model(path: Path, bank: AudioCodeBank, device: torch.device, lineage: dict[str, Any]) -> tuple[AudioCodeAutoencoder, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    saved_dependencies = payload.get("dependencies") or {}
    if not isinstance(saved_dependencies, dict):
        raise ValueError(f"Audio checkpoint dependencies must be an object: {path}")
    # Audio checkpoints are consumed by later phases. At this boundary we
    # validate the dependency record against itself; resume() is the boundary
    # that compares it with the caller's expected initialization/input SHA.
    validate_checkpoint_payload(
        payload,
        expected_phase="audio",
        expected_lineage=lineage,
        expected_dependencies={str(key): str(value) for key, value in saved_dependencies.items()},
        source=f"audio checkpoint {path}",
    )
    if "audio_init_checkpoint_sha256" in saved_dependencies:
        metadata = payload.get("metadata") or {}
        initialization = metadata.get("audio_initialization") if isinstance(metadata, dict) else None
        if not isinstance(initialization, dict) or initialization.get("source_checkpoint_sha256") != saved_dependencies.get("audio_init_checkpoint_sha256"):
            raise ValueError(
                f"Audio checkpoint initialization metadata does not match dependencies: {path}"
            )
    model = AudioCodeAutoencoder(AudioCodeModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


def train_eeg(args, cfg, cfg_path, context, bank, device, lineage) -> Path:
    settings = cfg["eeg_model"]
    audio_path = Path(args.audio_checkpoint) if args.audio_checkpoint else output_root(cfg, args.output_root) / "audio/checkpoints/best.pt"
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio checkpoint: {audio_path}")
    audio_gate = output_root(cfg, args.output_root) / "audio/metrics/validation_gate.json"
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
        loaders[dataset] = make_loader(
            train,
            int(settings["batch_size"]),
            cfg,
            balanced=True,
            ensure_same_label_pairs=(dataset == "karaone"),
        )
    karaone_validation = CombinedEEGDataset(
        context, bank, "karaone", "validation", eeg_len=int(cfg["data"]["eeg_len"])
    )
    karaone_validation_loader = DataLoader(
        karaone_validation,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["run"]["num_workers"]),
    )
    codebook_weights = torch.tensor(settings["codebook_weights"], device=device, dtype=torch.float32)
    directory = output_root(cfg, args.output_root) / "eeg"
    best_path, last_path = directory / "checkpoints/best.pt", directory / "checkpoints/last.pt"
    loss_recipe = effective_eeg_loss_recipe(settings)
    checkpoint_metadata = {
        "training_recipe_version": "combined-0721v1-structure-loss",
        "loss_recipe": loss_recipe,
        "karaone_same_label_pair_batching": True,
        "envelope_correlation_kernel_steps": list(ENVELOPE_CORRELATION_KERNELS),
        "dataset_sliced_label_distillation": True,
        "output_root": str(directory.parent.resolve()),
        "proxy_selection_score_definition": (
            "karaone median 1/5/9-step envelope correlation + 0.25*label BA "
            "- 0.10*onset MAE - 0.10*duration MAE"
        ),
        "final_reconstruction_selection": (
            "Run select_combined_0721_checkpoint.py on decoded KaraOne validation synthesis; "
            "best.pt is proxy-best only."
        ),
    }
    print(json.dumps({"training_recipe": checkpoint_metadata}, ensure_ascii=False), flush=True)
    epochs = int(args.epochs or settings["epochs"])
    save_every = int(args.save_every)
    if save_every < 0:
        raise ValueError("--save-every must be non-negative")
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
        selection_metrics = eeg_selection_metrics(
            model,
            karaone_validation_loader,
            device,
            label_offset=KARAONE_GLOBAL_OFFSET,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "selection_split": "validation",
            "test_accessed": False,
            **selection_metrics,
            **{name: float(np.mean(values)) for name, values in component_values.items()},
        }
        history.append(row)
        score = (
            row["selection_karaone_envelope_correlation_median"]
            + 0.25 * row["selection_karaone_label_balanced_accuracy"]
            - 0.10 * row["selection_karaone_onset_mae"]
            - 0.10 * row["selection_karaone_duration_mae"]
        )
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
                metadata=checkpoint_metadata,
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
            metadata=checkpoint_metadata,
        )
        save_candidate = save_every > 0 and (
            epoch == 1 or epoch % save_every == 0 or epoch == epochs
        )
        if save_candidate:
            candidate_path = directory / "checkpoints" / "candidates" / f"epoch_{epoch:03d}.pt"
            candidate_metadata = {
                **checkpoint_metadata,
                "candidate_checkpoint": True,
                "candidate_epoch": int(epoch),
                "proxy_selection_score": float(score),
                "decoded_validation_selected": False,
            }
            save_checkpoint(
                candidate_path,
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
                metadata=candidate_metadata,
                include_training_state=False,
            )
            print(f"[combined EEG] saved decoded-validation candidate: {candidate_path}", flush=True)
        print(json.dumps(row), flush=True)
    gate = {
        "passed": False,
        "reason": "run validation phase before unlocking test",
        "checkpoint": str(best_path),
        "audio_checkpoint_sha256": audio_checkpoint_sha256,
        "eeg_checkpoint_sha256": file_sha256(best_path) if best_path.is_file() else None,
        "lineage": lineage,
        "training_recipe": checkpoint_metadata,
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
    eeg_path = Path(args.eeg_checkpoint) if args.eeg_checkpoint else output_root(cfg, args.output_root) / "eeg/checkpoints/best.pt"
    audio_path = Path(args.audio_checkpoint) if args.audio_checkpoint else output_root(cfg, args.output_root) / "audio/checkpoints/best.pt"
    if not eeg_path.exists() or not audio_path.exists():
        raise FileNotFoundError("Both audio and EEG checkpoints are required")
    audio_checkpoint_sha256 = file_sha256(audio_path)
    eeg_checkpoint_sha256 = file_sha256(eeg_path)
    if args.split == "test":
        gate_path = output_root(cfg, args.output_root) / "eeg/metrics/validation_gate.json"
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
    destination = output_root(cfg, args.output_root) / "eeg/metrics" / f"{args.split}_evaluation.json"
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
        audio_path = Path(args.audio_checkpoint) if args.audio_checkpoint else output_root(cfg, args.output_root) / "audio/checkpoints/best.pt"
        eeg_path = Path(args.eeg_checkpoint) if args.eeg_checkpoint else output_root(cfg, args.output_root) / "eeg/checkpoints/best.pt"
        preauthorize_locked_test(
            output_root(cfg, args.output_root) / "eeg/metrics/validation_gate.json",
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
