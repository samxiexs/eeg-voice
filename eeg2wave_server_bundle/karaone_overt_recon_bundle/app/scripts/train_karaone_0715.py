from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.karaone_0715.data import (  # noqa: E402
    LABELS,
    AudioCodeBank,
    KaraOne0715Dataset,
    SplitManifest0715,
    records_for_split,
    sha256_bytes,
    write_json,
)
from src.karaone_0715.eval import balanced_accuracy, make_eeg_gate  # noqa: E402
from src.karaone_0715.losses import (  # noqa: E402
    code_cross_entropy,
    condition_alignment_loss,
    paired_contrastive_loss,
    soft_label_distillation,
    variance_regularizer,
)
from src.karaone_0715.model import (  # noqa: E402
    AudioCodeAutoencoder,
    AudioCodeModelConfig,
    EEGConditionEncoder,
    EEGModelConfig,
    random_code_mask,
)


PHASES = ("audio", "eeg", "evaluate")
SPLITS = ("subject_train", "subject_val", "subject_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train independent KaraOne 0715 codec-token EEG-to-voice pipeline.")
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--config", default=str(APP_DIR / "configs" / "karaone_0715.yaml"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--audio-checkpoint", default=None)
    parser.add_argument("--eeg-checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--audio-epochs", type=int, default=None)
    parser.add_argument("--eeg-epochs", type=int, default=None)
    parser.add_argument("--split", choices=SPLITS, default="subject_val")
    parser.add_argument("--allow-final-test", action="store_true")
    parser.add_argument("--allow-failed-gate", action="store_true", help="Exploratory only: bypass a failed upstream P02 gate.")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else APP_DIR / path


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


def one_hot(labels: torch.Tensor, classes: int) -> torch.Tensor:
    return F.one_hot(labels.long(), num_classes=int(classes)).float()


def cache_path(cfg: dict[str, Any], seed: int) -> Path:
    return resolve(cfg["paths"]["cache_root"]) / f"karaone_0715_encodec_codes_s{seed}.npz"


def output_root(cfg: dict[str, Any]) -> Path:
    return resolve(cfg["paths"]["output_root"])


def audio_dir(cfg: dict[str, Any], seed: int) -> Path:
    return output_root(cfg) / f"karaone_0715_audio_codec_s{seed}"


def eeg_dir(cfg: dict[str, Any], seed: int) -> Path:
    return output_root(cfg) / f"karaone_0715_eeg_align_s{seed}"


def audio_model_config(cfg: dict[str, Any], bank: AudioCodeBank) -> AudioCodeModelConfig:
    settings = cfg["audio_model"]
    return AudioCodeModelConfig(
        codebooks=bank.codebooks,
        code_steps=bank.code_steps,
        vocab_size=int(cfg["codec"]["vocab_size"]),
        num_labels=len(LABELS),
        d_model=int(settings["d_model"]),
        condition_steps=int(settings["condition_steps"]),
        encoder_layers=int(settings["encoder_layers"]),
        decoder_layers=int(settings["decoder_layers"]),
        heads=int(settings["heads"]),
        dropout=float(settings["dropout"]),
    )


def eeg_model_config(cfg: dict[str, Any], bank: AudioCodeBank) -> EEGModelConfig:
    settings = cfg["eeg_model"]
    return EEGModelConfig(
        channels=62,
        eeg_len=int(cfg["data"]["eeg_len"]),
        d_model=int(settings["d_model"]),
        condition_steps=int(settings["condition_steps"]),
        code_steps=bank.code_steps,
        num_labels=len(LABELS),
        num_train_subjects=len(cfg["data"]["train_subjects"]),
        transformer_layers=int(settings["transformer_layers"]),
        heads=int(settings["heads"]),
        dropout=float(settings["dropout"]),
        temporal_kernels=tuple(int(value) for value in settings["temporal_kernels"]),
        stem_stride=int(settings["stem_stride"]),
    )


class CodeDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, bank: AudioCodeBank, indices: np.ndarray):
        self.bank = bank
        self.indices = np.asarray(indices, dtype=np.int64)
        label_lookup = {label: index for index, label in enumerate(LABELS)}
        self.label_ids = np.asarray([label_lookup[str(bank.labels[index])] for index in self.indices], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        return torch.from_numpy(np.ascontiguousarray(self.bank.codes[index])).long(), torch.tensor(self.label_ids[item], dtype=torch.long)


def simple_loader(dataset: Dataset, batch_size: int, cfg: dict[str, Any], *, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle, num_workers=int(cfg["run"]["num_workers"]))


def eeg_loader(dataset: KaraOne0715Dataset, batch_size: int, cfg: dict[str, Any], *, balanced: bool) -> DataLoader:
    if not balanced:
        return DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=int(cfg["run"]["num_workers"]))
    groups = [(record.subject, record.label) for record in dataset.records]
    counts = Counter(groups)
    weights = torch.tensor([1.0 / counts[group] for group in groups], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    return DataLoader(dataset, batch_size=int(batch_size), sampler=sampler, num_workers=int(cfg["run"]["num_workers"]))


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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config,
            "epoch": int(epoch),
            "history": history,
            "best_score": float(best_score),
            "phase": phase,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        },
        path,
    )


def resume_training(path: str | Path | None, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[int, list[dict[str, Any]], float]:
    if path is None:
        return 0, [], -float("inf")
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    print(f"[0715] resumed after epoch {payload['epoch']}: {path}", flush=True)
    return int(payload["epoch"]), list(payload.get("history", [])), float(payload.get("best_score", -float("inf")))


def load_audio_model(path: Path, bank: AudioCodeBank, device: torch.device) -> tuple[AudioCodeAutoencoder, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = AudioCodeModelConfig(**payload["model_config"])
    expected = (bank.codebooks, bank.code_steps)
    if (config.codebooks, config.code_steps) != expected:
        raise ValueError(f"Audio checkpoint/cache mismatch: {(config.codebooks, config.code_steps)} != {expected}")
    model = AudioCodeAutoencoder(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


def load_eeg_model(path: Path, bank: AudioCodeBank, device: torch.device) -> tuple[EEGConditionEncoder, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = EEGModelConfig(**payload["model_config"])
    if config.code_steps != bank.code_steps:
        raise ValueError("EEG checkpoint/cache code-step mismatch")
    model = EEGConditionEncoder(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


@torch.no_grad()
def evaluate_audio(model: AudioCodeAutoencoder, loader: DataLoader, device: torch.device, codebook_weights: torch.Tensor) -> dict[str, float]:
    model.eval()
    label_target, label_prediction = [], []
    correct = np.zeros(model.cfg.codebooks, dtype=np.float64)
    label_only_correct = np.zeros(model.cfg.codebooks, dtype=np.float64)
    total = np.zeros(model.cfg.codebooks, dtype=np.float64)
    for codes, labels in tqdm(loader, desc="[0715 audio] validate", unit="batch", dynamic_ncols=True, leave=False):
        codes, labels = codes.to(device), labels.to(device)
        probabilities = one_hot(labels, model.cfg.num_labels)
        encoded = model.encoder(codes)
        mask = torch.ones_like(codes, dtype=torch.bool)
        logits = model.decoder(codes, mask, encoded["condition"], probabilities)
        label_only = model.decoder(codes, mask, torch.zeros_like(encoded["condition"]), probabilities)
        pred = logits.argmax(dim=-1)
        label_pred = label_only.argmax(dim=-1)
        for codebook in range(model.cfg.codebooks):
            correct[codebook] += float((pred[:, codebook] == codes[:, codebook]).sum().cpu())
            label_only_correct[codebook] += float((label_pred[:, codebook] == codes[:, codebook]).sum().cpu())
            total[codebook] += float(codes[:, codebook].numel())
        label_target.extend(labels.cpu().tolist())
        label_prediction.extend(encoded["label_logits"].argmax(dim=-1).cpu().tolist())
    result = {
        "audio_label_balanced_accuracy": balanced_accuracy(np.asarray(label_target), np.asarray(label_prediction), classes=model.cfg.num_labels),
        "all_mask_weighted_code_accuracy": float(np.average(correct / np.maximum(total, 1.0), weights=codebook_weights.cpu().numpy())),
        "label_only_weighted_code_accuracy": float(np.average(label_only_correct / np.maximum(total, 1.0), weights=codebook_weights.cpu().numpy())),
    }
    for codebook in range(model.cfg.codebooks):
        result[f"all_mask_q{codebook}_accuracy"] = float(correct[codebook] / max(total[codebook], 1.0))
        result[f"label_only_q{codebook}_accuracy"] = float(label_only_correct[codebook] / max(total[codebook], 1.0))
    result["coarse_gain_over_label_only"] = 0.5 * (
        result["all_mask_q0_accuracy"] + result["all_mask_q1_accuracy"] - result["label_only_q0_accuracy"] - result["label_only_q1_accuracy"]
    )
    return result


def train_audio(args: argparse.Namespace, cfg: dict[str, Any], bank: AudioCodeBank, device: torch.device, seed: int) -> Path:
    settings = cfg["audio_model"]
    directory = audio_dir(cfg, seed)
    for folder in (directory / "checkpoints", directory / "metrics"):
        folder.mkdir(parents=True, exist_ok=True)
    architecture = audio_model_config(cfg, bank)
    model = AudioCodeAutoencoder(architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["lr"]), weight_decay=float(settings["weight_decay"]))
    start_epoch, history, best_score = resume_training(args.resume, model, optimizer, device)
    train_data = simple_loader(CodeDataset(bank, bank.indices("subject_train")), int(settings["batch_size"]), cfg, shuffle=True)
    val_data = simple_loader(CodeDataset(bank, bank.indices("subject_val")), int(settings["batch_size"]), cfg, shuffle=False)
    epochs = int(args.audio_epochs or settings["epochs"])
    weights = torch.tensor(settings["codebook_weights"], device=device, dtype=torch.float32)
    best_path = directory / "checkpoints" / "best.pt"
    last_path = directory / "checkpoints" / "last.pt"
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        losses = []
        iterator = tqdm(train_data, desc=f"[0715 audio] train {epoch}/{epochs}", unit="batch", dynamic_ncols=True)
        for codes, labels in iterator:
            codes, labels = codes.to(device), labels.to(device)
            mask = random_code_mask(
                codes,
                min_ratio=float(settings["mask_ratio_min"]),
                max_ratio=float(settings["mask_ratio_max"]),
                full_mask_probability=float(settings["full_mask_probability"]),
            )
            probabilities = one_hot(labels, architecture.num_labels)
            label_drop = torch.rand(len(labels), device=device) < float(settings["label_dropout"])
            probabilities = probabilities * (~label_drop).float().unsqueeze(1)
            condition_drop = torch.rand(len(labels), device=device) < float(settings["condition_dropout"])
            output = model(codes, mask, probabilities, condition_dropout=condition_drop)
            code_loss = code_cross_entropy(output["code_logits"], codes, mask, weights)["total"]
            label_loss = F.cross_entropy(output["label_logits"], labels)
            loss = code_loss + float(settings["lambda_label"]) * label_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), float(cfg["run"]["grad_clip"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            iterator.set_postfix(loss=f"{losses[-1]:.4f}")
        metrics = evaluate_audio(model, val_data, device, weights.cpu())
        score = metrics["audio_label_balanced_accuracy"] + metrics["all_mask_q0_accuracy"] + metrics["all_mask_q1_accuracy"]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics, "selection_split": "subject_val", "test_accessed": False}
        history.append(row)
        if score > best_score:
            best_score = score
            save_checkpoint(best_path, model, optimizer, model_config=asdict(architecture), epoch=epoch, history=history, best_score=best_score, phase="audio")
            gate = {
                "passed": bool(metrics["audio_label_balanced_accuracy"] >= 0.75 and metrics["coarse_gain_over_label_only"] > 0.0),
                **metrics,
                "requirements": {"audio_label_balanced_accuracy_min": 0.75, "coarse_gain_over_label_only_min_exclusive": 0.0},
                "selection_split": "subject_val",
                "test_accessed": False,
                "checkpoint": str(best_path),
            }
            write_json(directory / "metrics" / "validation_gate.json", gate)
        save_checkpoint(last_path, model, optimizer, model_config=asdict(architecture), epoch=epoch, history=history, best_score=best_score, phase="audio")
        write_json(directory / "metrics" / "latest_metrics.json", row)
        write_json(directory / "metrics" / "history.json", {"history": history, "best_score": best_score, "selection_split": "subject_val", "test_accessed": False})
        print("[0715 audio] " + json.dumps(row), flush=True)
    return best_path


def move_eeg_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def evaluate_eeg(
    eeg_model: EEGConditionEncoder,
    audio_model: AudioCodeAutoencoder,
    loader: DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    eeg_model.eval()
    audio_model.eval()
    labels, predictions, eeg_conditions, audio_conditions = [], [], [], []
    coarse_correct = 0.0
    label_only_correct = 0.0
    coarse_total = 0.0
    for batch in tqdm(loader, desc="[0715 EEG] validate", unit="batch", dynamic_ncols=True, leave=False):
        batch = move_eeg_batch(batch, device)
        codes = batch["codes"]
        target = audio_model.encoder(codes)
        output = eeg_model(batch["eeg"], batch["eeg_valid_len"])
        probabilities = torch.softmax(output["label_logits"], dim=-1)
        mask = torch.ones_like(codes, dtype=torch.bool)
        logits = audio_model.decoder(codes, mask, output["condition"], probabilities)
        label_logits = audio_model.decoder(codes, mask, torch.zeros_like(output["condition"]), probabilities)
        code_prediction = logits[:, :2].argmax(dim=-1)
        label_prediction = label_logits[:, :2].argmax(dim=-1)
        coarse_correct += float((code_prediction == codes[:, :2]).sum().cpu())
        label_only_correct += float((label_prediction == codes[:, :2]).sum().cpu())
        coarse_total += float(codes[:, :2].numel())
        labels.extend(batch["label_idx"].cpu().tolist())
        predictions.extend(output["label_logits"].argmax(dim=-1).cpu().tolist())
        eeg_conditions.append(output["pooled"].cpu().numpy())
        audio_conditions.append(target["pooled"].cpu().numpy())
    arrays = {
        "labels": np.asarray(labels, dtype=np.int64),
        "predictions": np.asarray(predictions, dtype=np.int64),
        "eeg_condition": np.concatenate(eeg_conditions, axis=0).astype(np.float32),
        "audio_condition": np.concatenate(audio_conditions, axis=0).astype(np.float32),
    }
    gate = make_eeg_gate(
        labels=arrays["labels"],
        predictions=arrays["predictions"],
        eeg_condition=arrays["eeg_condition"],
        audio_condition=arrays["audio_condition"],
        coarse_code_accuracy=float(coarse_correct / max(coarse_total, 1.0)),
        label_only_coarse_code_accuracy=float(label_only_correct / max(coarse_total, 1.0)),
        min_balanced_accuracy=float(cfg["evaluation"]["min_p02_balanced_accuracy"]),
        chance_accuracy=float(cfg["evaluation"]["chance_accuracy"]),
        bootstrap_samples=int(cfg["evaluation"]["bootstrap_samples"]),
    )
    return gate.to_dict(), arrays


def train_eeg(args: argparse.Namespace, cfg: dict[str, Any], bank: AudioCodeBank, device: torch.device, seed: int, manifest: SplitManifest0715) -> Path:
    settings = cfg["eeg_model"]
    directory = eeg_dir(cfg, seed)
    for folder in (directory / "checkpoints", directory / "metrics"):
        folder.mkdir(parents=True, exist_ok=True)
    audio_path = Path(args.audio_checkpoint) if args.audio_checkpoint else audio_dir(cfg, seed) / "checkpoints" / "best.pt"
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio-code model checkpoint: {audio_path}")
    audio_gate_path = audio_dir(cfg, seed) / "metrics" / "roundtrip_gate.json"
    audio_gate = json.loads(audio_gate_path.read_text(encoding="utf-8")) if audio_gate_path.exists() else {"passed": False, "reason": "missing_audio_validation_gate"}
    if not bool(audio_gate.get("passed")) and not args.allow_failed_gate:
        raise PermissionError(f"0715 audio-only MaskGIT/wav round-trip gate failed; EEG alignment is blocked: {audio_gate}")
    audio_model, audio_payload = load_audio_model(audio_path, bank, device)
    audio_model.eval()
    for parameter in audio_model.parameters():
        parameter.requires_grad_(False)
    architecture = eeg_model_config(cfg, bank)
    model = EEGConditionEncoder(architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["lr"]), weight_decay=float(settings["weight_decay"]))
    start_epoch, history, best_score = resume_training(args.resume, model, optimizer, device)
    data_kwargs = {
        "bank": bank,
        "manifest": manifest,
        "eeg_len": int(cfg["data"]["eeg_len"]),
        "baseline_mode": str(cfg["data"]["baseline_mode"]),
        "clip_value": float(cfg["data"]["baseline_clip"]),
    }
    root = resolve(cfg["data"]["root"])
    train_data = eeg_loader(KaraOne0715Dataset(root, "subject_train", **data_kwargs), int(settings["batch_size"]), cfg, balanced=True)
    val_data = eeg_loader(KaraOne0715Dataset(root, "subject_val", **data_kwargs), int(settings["batch_size"]), cfg, balanced=False)
    epochs = int(args.eeg_epochs or settings["epochs"])
    codebook_weights = torch.tensor(settings["codebook_weights"], device=device, dtype=torch.float32)
    best_path = directory / "checkpoints" / "best.pt"
    last_path = directory / "checkpoints" / "last.pt"
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        values = []
        progress = min(1.0, epoch / max(1, int(settings["code_warmup_epochs"])))
        adversary_strength = float(settings["subject_adversary_max"]) * (2.0 / (1.0 + math.exp(-10.0 * epoch / epochs)) - 1.0)
        iterator = tqdm(train_data, desc=f"[0715 EEG] train {epoch}/{epochs}", unit="batch", dynamic_ncols=True)
        for batch in iterator:
            batch = move_eeg_batch(batch, device)
            eeg = model.augment(
                batch["eeg"].clone(),
                channel_dropout=float(settings["channel_dropout"]),
                time_mask_ratio=float(settings["time_mask_ratio"]),
                noise_std=float(settings["noise_std"]),
            )
            with torch.no_grad():
                audio_target = audio_model.encoder(batch["codes"])
            output = model(eeg, batch["eeg_valid_len"], subject_adversary_strength=adversary_strength)
            probabilities = torch.softmax(output["label_logits"], dim=-1)
            all_mask = torch.ones_like(batch["codes"], dtype=torch.bool)
            code_logits = audio_model.decoder(batch["codes"], all_mask, output["condition"], probabilities)
            code_loss = code_cross_entropy(code_logits, batch["codes"], all_mask, codebook_weights)["total"]
            alignment = condition_alignment_loss(output["condition"], audio_target["condition"])["total"]
            contrastive = paired_contrastive_loss(output["pooled"], audio_target["pooled"], batch["label_idx"])
            label_loss = F.cross_entropy(output["label_logits"], batch["label_idx"])
            envelope_loss = F.binary_cross_entropy_with_logits(output["envelope_logits"], batch["audio_envelope"])
            timing_loss = F.smooth_l1_loss(output["onset"], batch["onset"]) + F.smooth_l1_loss(output["duration"], batch["duration"])
            subject_loss = F.cross_entropy(output["subject_logits"], batch["subject_idx"])
            distill = soft_label_distillation(output["label_logits"], audio_target["label_logits"])
            variance = variance_regularizer(output["condition"])
            loss = (
                float(settings["lambda_label"]) * label_loss
                + float(settings["lambda_alignment"]) * alignment
                + float(settings["lambda_contrastive"]) * contrastive
                + progress * float(settings["lambda_code"]) * code_loss
                + float(settings["lambda_envelope"]) * envelope_loss
                + float(settings["lambda_timing"]) * timing_loss
                + float(settings["lambda_subject"]) * subject_loss
                + float(settings["lambda_distill"]) * distill
                + float(settings["lambda_variance"]) * variance
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), float(cfg["run"]["grad_clip"]))
            optimizer.step()
            values.append(float(loss.detach().cpu()))
            iterator.set_postfix(loss=f"{values[-1]:.4f}", code=f"{float(code_loss.detach().cpu()):.3f}")
        gate, _ = evaluate_eeg(model, audio_model, val_data, device, cfg)
        score = float(gate["class_balanced_accuracy"]) + float(gate["coarse_code_gain"]) + float(gate["paired_retrieval_top1"])
        row = {"epoch": epoch, "train_loss": float(np.mean(values)), **gate, "selection_split": "subject_val", "test_accessed": False}
        history.append(row)
        if score > best_score:
            best_score = score
            save_checkpoint(best_path, model, optimizer, model_config=asdict(architecture), epoch=epoch, history=history, best_score=best_score, phase="eeg")
            write_json(
                directory / "metrics" / "validation_gate.json",
                {
                    "version": "0715",
                    **gate,
                    "checkpoint": str(best_path),
                    "checkpoint_epoch": epoch,
                    "audio_checkpoint": str(audio_path),
                    "audio_checkpoint_epoch": int(audio_payload["epoch"]),
                    "split_checksum": manifest.checksum,
                    "selection_split": "subject_val",
                    "test_accessed": False,
                },
            )
        save_checkpoint(last_path, model, optimizer, model_config=asdict(architecture), epoch=epoch, history=history, best_score=best_score, phase="eeg")
        write_json(directory / "metrics" / "latest_metrics.json", row)
        write_json(directory / "metrics" / "history.json", {"history": history, "best_score": best_score, "selection_split": "subject_val", "test_accessed": False})
        print("[0715 EEG] " + json.dumps(row), flush=True)
    return best_path


def evaluate_phase(args: argparse.Namespace, cfg: dict[str, Any], bank: AudioCodeBank, device: torch.device, seed: int, manifest: SplitManifest0715) -> Path:
    if args.split == "subject_test" and not args.allow_final_test:
        raise PermissionError("0715 MM21 evaluation requires --allow-final-test")
    if args.split == "subject_test":
        gate_path = eeg_dir(cfg, seed) / "metrics" / "validation_gate.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {"passed": False, "reasons": ["missing_validation_gate"]}
        if not bool(gate.get("passed")) and not args.allow_failed_gate:
            raise PermissionError(f"0715 P02 gate failed; MM21 remains locked: {gate.get('reasons')}")
    audio_path = Path(args.audio_checkpoint) if args.audio_checkpoint else audio_dir(cfg, seed) / "checkpoints" / "best.pt"
    eeg_path = Path(args.eeg_checkpoint) if args.eeg_checkpoint else eeg_dir(cfg, seed) / "checkpoints" / "best.pt"
    audio_model, audio_payload = load_audio_model(audio_path, bank, device)
    eeg_model, eeg_payload = load_eeg_model(eeg_path, bank, device)
    root = resolve(cfg["data"]["root"])
    dataset = KaraOne0715Dataset(
        root,
        args.split,
        bank=bank,
        manifest=manifest,
        eeg_len=int(cfg["data"]["eeg_len"]),
        baseline_mode=str(cfg["data"]["baseline_mode"]),
        clip_value=float(cfg["data"]["baseline_clip"]),
    )
    data = eeg_loader(dataset, int(cfg["eeg_model"]["batch_size"]), cfg, balanced=False)
    gate, arrays = evaluate_eeg(eeg_model, audio_model, data, device, cfg)
    report = {
        "version": "0715",
        "phase": "evaluate",
        "split": args.split,
        "n_trials": len(dataset),
        **gate,
        "audio_checkpoint": str(audio_path),
        "audio_checkpoint_epoch": int(audio_payload["epoch"]),
        "eeg_checkpoint": str(eeg_path),
        "eeg_checkpoint_epoch": int(eeg_payload["epoch"]),
        "split_checksum": manifest.checksum,
        "inference_input": "clearing-calibrated overt EEG only",
        "reference_audio_used_for_prediction": False,
        "test_accessed": args.split == "subject_test",
        "label_targets": arrays["labels"].tolist(),
        "label_predictions": arrays["predictions"].tolist(),
    }
    destination = eeg_dir(cfg, seed) / "metrics" / f"{args.split}_evaluation.json"
    write_json(destination, report)
    print(json.dumps({key: value for key, value in report.items() if key not in {"label_targets", "label_predictions"}}, ensure_ascii=False, indent=2), flush=True)
    return destination


def write_run_manifest(cfg: dict[str, Any], config_path: Path, cache: Path, manifest: SplitManifest0715, seed: int) -> None:
    payload = {
        "version": "0715",
        "seed": seed,
        "config": str(config_path),
        "config_sha256": sha256_bytes(config_path.read_bytes()),
        "cache": str(cache),
        "cache_sha256": sha256_bytes(cache.read_bytes()),
        "split_checksum": manifest.checksum,
        "train_subjects": list(manifest.train_subjects),
        "subject_val": manifest.subject_val,
        "subject_test": manifest.subject_test,
        "generator_target": "exactly-decodable EnCodec discrete codes [8,150]",
        "eeg_only_inference": True,
        "test_accessed_for_selection": False,
    }
    write_json(output_root(cfg) / "karaone_0715_run_manifest.json", payload)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(cfg["run"]["seed"])
    set_seed(seed)
    device = torch.device(args.device) if args.device else default_device()
    root = resolve(cfg["data"]["root"])
    manifest = SplitManifest0715.build(root)
    path = Path(args.cache) if args.cache else cache_path(cfg, seed)
    if not path.exists():
        raise FileNotFoundError(f"Missing 0715 EnCodec code cache: {path}")
    bank = AudioCodeBank(path, manifest)
    output_root(cfg).mkdir(parents=True, exist_ok=True)
    write_run_manifest(cfg, config_path, path, manifest, seed)
    print(f"[0715] phase={args.phase}; device={device}; cache={path}; train={len(bank.indices('subject_train'))}; val={len(bank.indices('subject_val'))}; test_locked={not args.allow_final_test}", flush=True)
    if args.phase == "audio":
        train_audio(args, cfg, bank, device, seed)
    elif args.phase == "eeg":
        train_eeg(args, cfg, bank, device, seed, manifest)
    else:
        evaluate_phase(args, cfg, bank, device, seed, manifest)


if __name__ == "__main__":
    main()
