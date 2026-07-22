#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from src.open_vocab_0722.data import (  # noqa: E402
    AudioCodeBank,
    DATASETS,
    LabelFreeAudioDataset,
    OpenVoiceEEGDataset,
    TeacherBank,
    collate_openvoice,
    common_montage_view,
    load_context,
    resolve_config_path,
    stochastic_channel_view,
)
from src.open_vocab_0722.lineage import (  # noqa: E402
    build_lineage,
    checkpoint_payload,
    file_sha256,
    preauthorize_test,
    preauthorize_test_metadata,
    validate_checkpoint,
)
from src.open_vocab_0722.losses import (  # noqa: E402
    code_cross_entropy,
    condition_consistency_loss,
    exact_pair_contrastive_loss,
    loss_eligibility,
    masked_patch_reconstruction_loss,
    moe_regularization,
    monotonic_local_alignment_loss,
    semantic_positive_weights,
    structure_loss,
    symmetric_contrastive_loss,
    text_semantic_loss,
)
from src.open_vocab_0722.model import (  # noqa: E402
    LabelFreeAudioConfig,
    LabelFreeAudioModel,
    OpenVoiceEEGConfig,
    OpenVoiceEEGEncoder,
    random_code_mask,
    random_patch_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate label-free OpenVoice-EEG 0722")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("audio", "eeg-pretrain", "eeg", "evaluate"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--generalization", choices=("g1", "g2", "g3"), default="g1")
    parser.add_argument("--holdout-label", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--project-audio-only", action="store_true", help="exploratory smoke only")
    parser.add_argument("--allow-final-test", action="store_true")
    parser.add_argument("--no-pretrain-init", action="store_true")
    parser.add_argument("--shared-init-checkpoint", type=Path, default=None, help="Explicitly copy only same-name/same-shape non-label weights from a legacy audio checkpoint")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def audio_config(cfg: dict[str, Any]) -> LabelFreeAudioConfig:
    value = cfg["audio_model"]
    codec = cfg["codec"]
    return LabelFreeAudioConfig(
        codebooks=int(codec["codebooks"]), code_steps=int(codec["code_steps"]),
        vocab_size=int(codec["vocab_size"]), d_model=int(value["d_model"]),
        condition_steps=int(value["condition_steps"]), encoder_layers=int(value["encoder_layers"]),
        decoder_layers=int(value["decoder_layers"]), heads=int(value["heads"]),
        dropout=float(value["dropout"]), text_dimension=int(cfg["teachers"]["text_dimension"]),
        xlsr_dimension=int(value["xlsr_dimension"]),
    )


def eeg_config(cfg: dict[str, Any], subjects: int) -> OpenVoiceEEGConfig:
    value = cfg["eeg_model"]
    return OpenVoiceEEGConfig(
        eeg_samples=int(cfg["data"]["eeg_samples"]), patch_size=int(value["patch_size"]),
        patch_hop=int(value["patch_hop"]), d_model=int(value["d_model"]),
        condition_steps=int(value["condition_steps"]), code_steps=int(cfg["codec"]["code_steps"]),
        heads=int(value["heads"]), latent_layers=int(value["latent_layers"]), dropout=float(value["dropout"]),
        specialists=int(value["specialists"]), specialist_bottleneck=int(value["specialist_bottleneck"]),
        soft_routing_epochs=int(value["soft_routing_epochs"]), top_k_specialists=int(value["top_k_specialists"]),
        expert_dropout=float(value["expert_dropout"]), num_datasets=len(DATASETS), num_train_subjects=subjects,
        adapter_moe_enabled=bool(value.get("adapter_moe_enabled", True)),
        text_dimension=int(cfg["teachers"]["text_dimension"]),
    )


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_audio_checkpoint(path: Path, cfg: dict[str, Any], lineage: dict[str, Any], device: torch.device) -> LabelFreeAudioModel:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint(payload, phase="audio", lineage=lineage, source=str(path))
    model = LabelFreeAudioModel(audio_config(cfg))
    model.load_state_dict(payload["model_state"])
    return model.to(device)


@torch.no_grad()
def validate_audio(model: LabelFreeAudioModel, loader: DataLoader[Any], device: torch.device, cfg: dict[str, Any]) -> dict[str, float]:
    model.eval()
    totals: defaultdict[str, float] = defaultdict(float)
    count = 0
    for batch in tqdm(loader, desc="[0722 audio] validation", leave=False):
        batch = move(batch, device)
        mask = torch.ones_like(batch["codes"], dtype=torch.bool)
        output = model(batch["codes"], mask, code_valid_mask=batch["code_valid_mask"], xlsr_tokens=batch["xlsr_tokens"])
        metrics = code_cross_entropy(output["code_logits"], batch["codes"], mask & batch["code_valid_mask"])
        size = len(batch["codes"])
        totals["loss"] += float(metrics["total"]) * size
        for index in range(2):
            totals[f"q{index}_accuracy"] += float(metrics[f"q{index}_accuracy"]) * size
        count += size
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_audio(args: argparse.Namespace, context: Any, teachers: TeacherBank, lineage: dict[str, Any], device: torch.device) -> None:
    cfg = context.config
    value = cfg["audio_model"]
    train_set = LabelFreeAudioDataset(context, teachers, split="train", include_public=not args.project_audio_only)
    validation_set = LabelFreeAudioDataset(context, teachers, split="validation", include_public=not args.project_audio_only)
    train_loader = DataLoader(train_set, batch_size=int(value["batch_size"]), shuffle=True, num_workers=int(cfg["run"]["num_workers"]))
    validation_loader = DataLoader(validation_set, batch_size=int(value["batch_size"]), shuffle=False)
    model = LabelFreeAudioModel(audio_config(cfg)).to(device)
    initialization: dict[str, Any] = {"mode": "scratch_label_free"}
    if args.shared_init_checkpoint and not args.resume:
        legacy = torch.load(args.shared_init_checkpoint, map_location="cpu", weights_only=False)
        source = legacy.get("model_state") or legacy.get("model_state_dict") or legacy.get("state_dict") or legacy.get("audio_model")
        if not isinstance(source, dict):
            raise ValueError("Legacy initialization checkpoint has no recognizable model state")
        current = model.state_dict(); copied = []
        for key, value_state in source.items():
            clean = str(key).removeprefix("module.")
            forbidden = ("label", "dataset", "subject", "text_projector")
            if any(token in clean.lower() for token in forbidden):
                continue
            if clean in current and current[clean].shape == value_state.shape:
                current[clean] = value_state; copied.append(clean)
        model.load_state_dict(current)
        report_path = resolve_config_path(context.config_path, cfg["paths"]["output_root"]) / "audio/shared_initialization.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"source": str(args.shared_init_checkpoint.resolve()), "source_sha256": file_sha256(args.shared_init_checkpoint), "copied_keys": copied, "excluded_label_dataset_subject_weights": True}, indent=2) + "\n", encoding="utf-8")
        initialization = {"mode": "shared_nonlabel_weight_extraction", "source_sha256": file_sha256(args.shared_init_checkpoint), "copied_tensor_count": len(copied)}
        print(f"[0722 audio] copied {len(copied)} label-independent tensors from {args.shared_init_checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(value["lr"]), weight_decay=float(value["weight_decay"]))
    start = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_checkpoint(payload, phase="audio", lineage=lineage, source=str(args.resume))
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start = int(payload["epoch"]) + 1
    epochs = int(args.epochs or value["epochs"])
    weights = torch.tensor(value["codebook_weights"], device=device)
    best = math.inf
    output = resolve_config_path(context.config_path, cfg["paths"]["audio_checkpoint"])
    metrics_path = resolve_config_path(context.config_path, cfg["paths"]["output_root"]) / "audio/metrics/train.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(start, epochs):
        model.train()
        running = 0.0
        progress = tqdm(train_loader, desc=f"[0722 audio] {epoch + 1}/{epochs}", unit="batch")
        for batch in progress:
            batch = move(batch, device)
            mask = random_code_mask(
                batch["codes"], min_ratio=float(value["mask_ratio_min"]), max_ratio=float(value["mask_ratio_max"]),
                full_mask_probability=float(value["full_mask_probability"]),
            ) & batch["code_valid_mask"]
            dropout = torch.rand(len(batch["codes"]), device=device) < float(value["condition_dropout"])
            output_value = model(batch["codes"], mask, code_valid_mask=batch["code_valid_mask"], xlsr_tokens=batch["xlsr_tokens"], condition_dropout=dropout)
            metrics = code_cross_entropy(output_value["code_logits"], batch["codes"], mask, weights)
            optimizer.zero_grad(set_to_none=True)
            metrics["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["run"]["grad_clip"]))
            optimizer.step()
            running += float(metrics["total"])
            progress.set_postfix(loss=f"{running / max(1, progress.n + 1):.4f}")
        validation = validate_audio(model, validation_loader, device, cfg)
        summary = {"epoch": epoch + 1, "train_loss": running / max(len(train_loader), 1), **validation, "labels_used": False, "test_accessed": False}
        print(json.dumps(summary, sort_keys=True))
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
        if validation["loss"] < best:
            best = validation["loss"]
            saved = checkpoint_payload(phase="audio", lineage=lineage, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), epoch=epoch, metrics=summary)
            saved["initialization"] = initialization
            save(output, saved)


def balanced_loader(dataset: OpenVoiceEEGDataset, batch_size: int, workers: int) -> DataLoader[Any]:
    counts = defaultdict(int)
    for row in dataset.rows:
        counts[row["dataset"]] += 1
    weights = [1.0 / counts[row["dataset"]] for row in dataset.rows]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_openvoice, num_workers=workers)


def eeg_outputs(model: OpenVoiceEEGEncoder, batch: dict[str, Any], epoch: int, adversary: float, patch_ratio: float) -> dict[str, Any]:
    with torch.no_grad():
        _, valid = model._patches(batch["eeg"], batch["channel_mask"], batch["time_mask"])
    patch_mask = random_patch_mask(valid, patch_ratio)
    return model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], batch["time_mask"], epoch=epoch, patch_mask=patch_mask, adversary_strength=adversary)


@torch.no_grad()
def validate_eeg(
    eeg: OpenVoiceEEGEncoder,
    audio: LabelFreeAudioModel | None,
    loader: DataLoader[Any],
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    eeg.eval()
    if audio is not None:
        audio.eval()
    acoustic_eeg, acoustic_audio = [], []
    envelopes, targets, router_mass = [], [], []
    datasets, dataset_logits = [], []
    reconstruction_losses: list[float] = []
    pooled_values: list[torch.Tensor] = []
    for raw in tqdm(loader, desc="[0722 EEG] validation", leave=False):
        batch = move(raw, device)
        _, patch_valid = eeg._patches(batch["eeg"], batch["channel_mask"], batch["time_mask"])
        deterministic_mask = torch.zeros_like(patch_valid)
        deterministic_mask[:, :, ::3] = patch_valid[:, :, ::3]
        output = eeg(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], batch["time_mask"], epoch=epoch, patch_mask=deterministic_mask)
        reconstruction_losses.append(float(masked_patch_reconstruction_loss(output["patch_reconstruction"], output["patch_target"], output["patch_mask"])))
        pooled_values.append(output["pooled"].detach().cpu())
        router_mass.append(output["router"]["specialist_mass"].detach().cpu())
        datasets.extend(batch["dataset_idx"].cpu().tolist())
        dataset_logits.extend(output["dataset_logits"].argmax(dim=-1).cpu().tolist())
        exact = loss_eligibility(batch["dataset"], batch["pairing_confidence"], device=device).exact_acoustic
        if audio is not None and exact.any():
            teacher = audio.xlsr_encoder(batch["xlsr_tokens"])
            acoustic_eeg.append(output["acoustic_global"][exact].cpu())
            acoustic_audio.append(teacher["acoustic_global"][exact].cpu())
            envelopes.append(output["envelope"][exact].cpu())
            targets.append(batch["audio_envelope"][exact].cpu())
    mass = torch.stack(router_mass).mean(dim=0) if router_mass else torch.zeros(4)
    result: dict[str, float] = {f"router_mass_{index}": float(value) for index, value in enumerate(mass)}
    result["dataset_balanced_accuracy"] = float(balanced_accuracy_score(datasets, dataset_logits))
    result["masked_reconstruction_loss"] = float(np.mean(reconstruction_losses))
    result["condition_feature_std"] = float(torch.cat(pooled_values).std(dim=0).mean())
    if acoustic_eeg:
        first = F.normalize(torch.cat(acoustic_eeg), dim=-1)
        second = F.normalize(torch.cat(acoustic_audio), dim=-1)
        similarity = first @ second.T
        ranks = torch.argsort(similarity, dim=1, descending=True)
        truth = torch.arange(len(first)).unsqueeze(1)
        result["retrieval_r1"] = float((ranks[:, :1] == truth).any(dim=1).float().mean())
        result["retrieval_r5"] = float((ranks[:, :5] == truth).any(dim=1).float().mean())
        predicted = torch.cat(envelopes)
        target = torch.cat(targets)
        predicted = predicted - predicted.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)
        corr = (predicted * target).sum(dim=1) / torch.sqrt(predicted.square().sum(dim=1) * target.square().sum(dim=1)).clamp_min(1e-8)
        pred_mod = torch.log1p(torch.fft.rfft(predicted, dim=1).abs())
        target_mod = torch.log1p(torch.fft.rfft(target, dim=1).abs())
        mod = F.cosine_similarity(pred_mod, target_mod, dim=1)
        result["envelope_correlation"] = float(corr.mean())
        result["modulation_correlation"] = float(mod.mean())
        # q-accuracy is deliberately only a checkpoint proxy.  Formal model
        # selection uses decoded validation waveforms and log-mel MAE.
        result["proxy_composite"] = 0.45 * result["envelope_correlation"] + 0.30 * result["modulation_correlation"] + 0.25 * result["retrieval_r5"]
    else:
        result.update({"retrieval_r1": 0.0, "retrieval_r5": 0.0, "envelope_correlation": 0.0, "modulation_correlation": 0.0, "proxy_composite": -math.inf})
    return result


def domain_loss(output: dict[str, Any], batch: dict[str, Any]) -> torch.Tensor:
    dataset = F.cross_entropy(output["dataset_logits"], batch["dataset_idx"])
    router_dataset = F.cross_entropy(output["router_dataset_logits"], batch["dataset_idx"])
    valid_subject = batch["subject_idx"] >= 0
    if valid_subject.any():
        subject = F.cross_entropy(output["subject_logits"][valid_subject], batch["subject_idx"][valid_subject])
        router_subject = F.cross_entropy(output["router_subject_logits"][valid_subject], batch["subject_idx"][valid_subject])
    else:
        subject = dataset * 0.0
        router_subject = dataset * 0.0
    return dataset + router_dataset + subject + router_subject


def train_eeg(args: argparse.Namespace, context: Any, teachers: TeacherBank, lineage: dict[str, Any], device: torch.device, *, pretrain: bool) -> None:
    cfg = context.config
    value = cfg["eeg_model"]
    loss_cfg = cfg["loss"]
    bank = AudioCodeBank(resolve_config_path(context.config_path, cfg["paths"]["project_audio_cache"]))
    train_set = OpenVoiceEEGDataset(context, bank, split="train", generalization=args.generalization, holdout_label=args.holdout_label, teachers=teachers)
    val_split = "validation" if args.generalization in {"g1", "g3"} else "validation"
    validation_set = OpenVoiceEEGDataset(context, bank, split=val_split, generalization=args.generalization, holdout_label=args.holdout_label, teachers=teachers)
    train_loader = balanced_loader(train_set, int(value["batch_size"]), int(cfg["run"]["num_workers"]))
    validation_loader = DataLoader(validation_set, batch_size=int(value["batch_size"]), shuffle=False, collate_fn=collate_openvoice)
    eeg = OpenVoiceEEGEncoder(eeg_config(cfg, len(context.subject_to_index))).to(device)
    audio: LabelFreeAudioModel | None = None
    dependencies: dict[str, str] = {}
    if not pretrain:
        audio_path = resolve_config_path(context.config_path, cfg["paths"]["audio_checkpoint"])
        audio = load_audio_checkpoint(audio_path, cfg, lineage, device)
        audio.eval()
        for parameter in audio.parameters():
            parameter.requires_grad_(False)
        dependencies["audio_checkpoint_sha256"] = file_sha256(audio_path)
        pretrain_path = resolve_config_path(context.config_path, cfg["paths"]["eeg_pretrain_checkpoint"])
        if not args.no_pretrain_init:
            payload = torch.load(pretrain_path, map_location="cpu", weights_only=False)
            validate_checkpoint(payload, phase="eeg-pretrain", lineage=lineage, source=str(pretrain_path))
            eeg.load_state_dict(payload["model_state"])
            dependencies["eeg_pretrain_checkpoint_sha256"] = file_sha256(pretrain_path)
    optimizer = torch.optim.AdamW(eeg.parameters(), lr=float(value["lr"]), weight_decay=float(value["weight_decay"]))
    phase = "eeg-pretrain" if pretrain else "eeg"
    epochs = int(args.epochs or (value["pretrain_epochs"] if pretrain else value["paired_epochs"]))
    checkpoint = (
        resolve_config_path(context.config_path, cfg["paths"]["eeg_pretrain_checkpoint"])
        if pretrain
        else resolve_config_path(context.config_path, cfg["paths"]["output_root"]) / "eeg/checkpoints/proxy_best.pt"
    )
    candidate_dir = resolve_config_path(context.config_path, cfg["paths"]["output_root"]) / "eeg/checkpoints/candidates"
    start, best, patience = 0, -math.inf, 0
    baseline_score: float | None = None
    decoder_unfrozen = False
    router_bad_streak = [0] * int(value["specialists"])
    metrics_path = resolve_config_path(context.config_path, cfg["paths"]["output_root"]) / phase / "metrics/train.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_checkpoint(payload, phase=phase, lineage=lineage, dependencies=dependencies, source=str(args.resume))
        eeg.load_state_dict(payload["model_state"])
        if audio is not None and payload.get("audio_decoder_state") is not None:
            audio.decoder.load_state_dict(payload["audio_decoder_state"])
            decoder_unfrozen = bool(payload.get("decoder_unfrozen", False))
            if decoder_unfrozen:
                parameters = []
                for block in list(audio.decoder.decoder.layers)[-int(value["decoder_unfreeze_blocks"]):]:
                    for parameter in block.parameters():
                        parameter.requires_grad_(True)
                        parameters.append(parameter)
                optimizer.add_param_group({"params": parameters, "lr": float(value["lr"]) * float(value["decoder_unfreeze_lr_scale"])})
        optimizer.load_state_dict(payload["optimizer_state"])
        start = int(payload["epoch"]) + 1
    code_weights = torch.tensor(loss_cfg["codebook_weights"], device=device)
    for epoch in range(start, epochs):
        if (
            not pretrain
            and audio is not None
            and not decoder_unfrozen
            and epoch >= int(value["decoder_frozen_epochs"])
            and baseline_score is not None
            and best > baseline_score + 1e-4
        ):
            blocks = list(audio.decoder.decoder.layers)[-int(value["decoder_unfreeze_blocks"]):]
            parameters = []
            for block in blocks:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
                    parameters.append(parameter)
            optimizer.add_param_group({"params": parameters, "lr": float(value["lr"]) * float(value["decoder_unfreeze_lr_scale"])})
            decoder_unfrozen = True
            print("[0722 EEG] validation improved while decoder was frozen; unfroze the last MaskGIT blocks at 0.1x LR")
        eeg.train()
        running: defaultdict[str, float] = defaultdict(float)
        adversary = float(value["adversary_max"]) * min(1.0, (epoch + 1) / max(1, epochs // 3))
        progress = tqdm(train_loader, desc=f"[0722 {phase}] {epoch + 1}/{epochs}", unit="batch")
        for raw in progress:
            batch = move(raw, device)
            drop = random.choice(list(map(float, value["channel_drop_probabilities"])))
            augmented = stochastic_channel_view(
                batch, drop_probability=drop, coordinate_noise_std=float(value["coordinate_noise_std"]),
                region_drop_fraction=float(value["continuous_region_drop_fraction"]),
                bad_channel_probability=float(value["bad_channel_probability"]),
                time_mask_fraction=float(value["time_mask_fraction"]), noise_std=float(value["signal_noise_std"]),
            )
            common = common_montage_view(batch)
            first = eeg_outputs(eeg, batch, epoch, adversary, float(value["patch_mask_ratio"]))
            second = eeg_outputs(eeg, augmented, epoch, adversary, float(value["patch_mask_ratio"]))
            common_output = eeg_outputs(eeg, common, epoch, adversary, float(value["patch_mask_ratio"]))
            masked = (1.0 / 3.0) * (
                masked_patch_reconstruction_loss(first["patch_reconstruction"], first["patch_target"], first["patch_mask"])
                + masked_patch_reconstruction_loss(second["patch_reconstruction"], second["patch_target"], second["patch_mask"])
                + masked_patch_reconstruction_loss(common_output["patch_reconstruction"], common_output["patch_target"], common_output["patch_mask"])
            )
            consistency = 0.5 * (condition_consistency_loss(first["condition"], second["condition"]) + condition_consistency_loss(first["condition"], common_output["condition"]))
            domain = (domain_loss(first, batch) + domain_loss(second, batch) + domain_loss(common_output, batch)) / 3.0
            moe = (moe_regularization(first["router"]) + moe_regularization(second["router"]) + moe_regularization(common_output["router"])) / 3.0
            total = float(loss_cfg["eeg_masked_pretraining"]) * masked + float(loss_cfg["channel_consistency"]) * consistency + float(loss_cfg["domain_adversarial"]) * domain + float(loss_cfg["moe"]) * moe
            if not pretrain:
                assert audio is not None
                eligibility = loss_eligibility(batch["dataset"], batch["pairing_confidence"], device=device)
                with torch.no_grad():
                    audio_teacher = audio.xlsr_encoder(batch["xlsr_tokens"])
                exact = eligibility.exact_acoustic & batch["has_audio_teacher"]
                weak = eligibility.weak_semantic & batch["has_audio_teacher"]
                # EEG-to-code learning must work from condition alone; leaving
                # true codec tokens visible would create a teacher-forcing
                # shortcut unavailable at inference.
                code_mask = batch["code_valid_mask"].bool()
                if exact.any():
                    logits = audio.decoder(batch["codes"][exact], code_mask[exact], first["condition"][exact])
                    code = code_cross_entropy(logits, batch["codes"][exact], code_mask[exact], code_weights)["total"]
                else:
                    code = total * 0.0
                acoustic = exact_pair_contrastive_loss(first["acoustic_global"], audio_teacher["acoustic_global"], exact, temperature=float(loss_cfg["contrastive_temperature"]))["total"]
                feis = torch.tensor([name == "feis" for name in batch["dataset"]], device=device) & weak
                feis_positive = ((batch["audio_idx"][:, None] == batch["audio_idx"][None, :]) & feis[:, None] & feis[None, :]).to(torch.float32)
                feis_allowed = feis[:, None] & feis[None, :]
                feis_clip = symmetric_contrastive_loss(first["acoustic_global"], audio_teacher["acoustic_global"], feis_positive, allowed=feis_allowed, temperature=float(loss_cfg["contrastive_temperature"]))["total"]
                local = monotonic_local_alignment_loss(first["acoustic_local"], audio_teacher["acoustic_local"], exact, temperature=float(loss_cfg["contrastive_temperature"]))["total"]
                semantic_weights = semantic_positive_weights(batch["label_idx"], exact, weak, weak_weight=float(loss_cfg["same_label_weak_positive"]))
                semantic_allowed = weak[:, None] & weak[None, :]
                semantic = symmetric_contrastive_loss(first["semantic_global"], audio_teacher["semantic_global"], semantic_weights, allowed=semantic_allowed, temperature=float(loss_cfg["contrastive_temperature"]))["total"]
                text_mask = weak & batch["has_text_teacher"]
                text_projection = eeg.project_text(batch["text_embedding"])
                text = text_semantic_loss(first["semantic_global"], text_projection, text_mask)
                envelope_valid = batch["code_valid_mask"].any(dim=1)
                structure = structure_loss(first["envelope"], batch["audio_envelope"], envelope_valid, first["onset"], batch["onset"], first["duration"], batch["duration"], exact)["total"]
                total = total + float(loss_cfg["code"]) * code + float(loss_cfg["clip_global"]) * (acoustic + float(loss_cfg["feis_global_clip_weight"]) * feis_clip) + float(loss_cfg["clip_local"]) * local + float(loss_cfg["structure"]) * structure + float(loss_cfg["same_label_semantic"]) * semantic + float(loss_cfg["text"]) * text
                for key, value_metric in (("code", code), ("clip_global", acoustic), ("clip_local", local), ("semantic", semantic), ("structure", structure)):
                    running[key] += float(value_metric.detach())
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(eeg.parameters(), float(cfg["run"]["grad_clip"]))
            optimizer.step()
            running["total"] += float(total.detach())
            progress.set_postfix(loss=f"{running['total'] / max(1, progress.n + 1):.4f}")
        validation = validate_eeg(eeg, audio, validation_loader, device, epoch)
        score = (
            -validation["masked_reconstruction_loss"]
            - 0.10 * validation.get("dataset_balanced_accuracy", 1.0)
            + 0.10 * validation["condition_feature_std"]
            if pretrain else validation["proxy_composite"]
        )
        if baseline_score is None:
            baseline_score = score
        for index in range(int(value["specialists"])):
            mass = validation[f"router_mass_{index}"]
            bad = mass < float(value["expert_dying_threshold"]) or mass > float(value["expert_collapse_threshold"])
            router_bad_streak[index] = router_bad_streak[index] + 1 if bad else 0
        bad_router = any(streak >= int(value["expert_bad_epochs"]) for streak in router_bad_streak)
        summary = {"epoch": epoch + 1, "phase": phase, **{key: value_metric / max(len(train_loader), 1) for key, value_metric in running.items()}, **validation, "router_bad_streak": router_bad_streak, "router_checkpoint_eligible": not bad_router, "selection_split": "validation", "test_accessed": False}
        print(json.dumps(summary, sort_keys=True))
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, sort_keys=True) + "\n")
        if score > best and not bad_router:
            best, patience = score, 0
            saved = checkpoint_payload(phase=phase, lineage=lineage, model_state=eeg.state_dict(), optimizer_state=optimizer.state_dict(), epoch=epoch, metrics=summary, dependencies=dependencies)
            if audio is not None:
                saved["decoder_unfrozen"] = decoder_unfrozen
                saved["audio_decoder_state"] = audio.decoder.state_dict() if decoder_unfrozen else None
            save(checkpoint, saved)
        else:
            patience += 1
        if not pretrain and ((epoch + 1) % int(value["candidate_interval_epochs"]) == 0 or epoch + 1 == epochs):
            candidate = checkpoint_payload(phase=phase, lineage=lineage, model_state=eeg.state_dict(), optimizer_state=None, epoch=epoch, metrics=summary, dependencies=dependencies)
            candidate["decoder_unfrozen"] = decoder_unfrozen
            candidate["audio_decoder_state"] = audio.decoder.state_dict() if audio is not None and decoder_unfrozen else None
            candidate["candidate_for_decoded_validation_selection"] = True
            save(candidate_dir / f"epoch_{epoch + 1:03d}.pt", candidate)
        if not pretrain and patience >= int(value["early_stopping_patience"]):
            print(f"[0722 EEG] early stopping after {patience} non-improving epochs")
            break


def evaluate(args: argparse.Namespace, context: Any, teachers: TeacherBank, lineage: dict[str, Any], device: torch.device) -> None:
    cfg = context.config
    audio_path = resolve_config_path(context.config_path, cfg["paths"]["audio_checkpoint"])
    eeg_path = resolve_config_path(context.config_path, cfg["paths"]["eeg_checkpoint"])
    if args.split == "test":
        if not args.allow_final_test:
            raise PermissionError("Locked test requires --allow-final-test")
        preauthorize_test(resolve_config_path(context.config_path, cfg["paths"]["validation_gate"]), lineage=lineage, audio_checkpoint=audio_path, eeg_checkpoint=eeg_path)
    # Test authorization is complete before dataset construction/read.
    audio = load_audio_checkpoint(audio_path, cfg, lineage, device)
    dependencies = {"audio_checkpoint_sha256": file_sha256(audio_path)}
    pretrain_path = resolve_config_path(context.config_path, cfg["paths"]["eeg_pretrain_checkpoint"])
    payload = torch.load(eeg_path, map_location="cpu", weights_only=False)
    if "eeg_pretrain_checkpoint_sha256" in (payload.get("dependencies") or {}):
        if not pretrain_path.is_file():
            raise FileNotFoundError(f"EEG checkpoint requires missing pretraining checkpoint: {pretrain_path}")
        dependencies["eeg_pretrain_checkpoint_sha256"] = file_sha256(pretrain_path)
    validate_checkpoint(payload, phase="eeg", lineage=lineage, dependencies=dependencies, source=str(eeg_path))
    if payload.get("audio_decoder_state") is not None:
        audio.decoder.load_state_dict(payload["audio_decoder_state"])
    eeg = OpenVoiceEEGEncoder(eeg_config(cfg, len(context.subject_to_index))).to(device)
    eeg.load_state_dict(payload["model_state"])
    bank = AudioCodeBank(resolve_config_path(context.config_path, cfg["paths"]["project_audio_cache"]))
    dataset = OpenVoiceEEGDataset(context, bank, split=args.split, generalization=args.generalization, holdout_label=args.holdout_label, teachers=teachers)
    loader = DataLoader(dataset, batch_size=int(cfg["eeg_model"]["batch_size"]), shuffle=False, collate_fn=collate_openvoice)
    metrics = validate_eeg(eeg, audio, loader, device, int(payload["epoch"]))
    print(json.dumps({"split": args.split, "generalization": args.generalization, "holdout_label": args.holdout_label, "metrics": metrics, "test_accessed": args.split == "test"}, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.phase == "evaluate" and args.split == "test":
        if not args.allow_final_test:
            raise PermissionError("Locked test requires --allow-final-test")
        raw_cfg = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
        preauthorize_test_metadata(
            resolve_config_path(args.config.resolve(), raw_cfg["paths"]["validation_gate"]),
            config_path=args.config.resolve(),
            audio_checkpoint=resolve_config_path(args.config.resolve(), raw_cfg["paths"]["audio_checkpoint"]),
            eeg_checkpoint=resolve_config_path(args.config.resolve(), raw_cfg["paths"]["eeg_checkpoint"]),
        )
    context = load_context(args.config)
    seed = int(args.seed if args.seed is not None else context.config["run"]["seed"])
    seed_everything(seed)
    device = torch.device(args.device) if args.device else default_device()
    teacher_path = resolve_config_path(context.config_path, context.config["paths"]["teacher_cache"])
    teachers = TeacherBank(teacher_path)
    lineage = build_lineage(context, require_optional_caches=not args.project_audio_only)
    print(f"[openvoice-0722] phase={args.phase}; device={device}; seed={seed}; hashing complete")
    if args.phase == "audio":
        train_audio(args, context, teachers, lineage, device)
    elif args.phase == "eeg-pretrain":
        train_eeg(args, context, teachers, lineage, device, pretrain=True)
    elif args.phase == "eeg":
        train_eeg(args, context, teachers, lineage, device, pretrain=False)
    else:
        evaluate(args, context, teachers, lineage, device)


if __name__ == "__main__":
    main()
