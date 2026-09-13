#!/usr/bin/env python3
"""Train an optional audio-only diffusion refiner for native SpeechT5 mel."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(APP / "src"))

from eeg2speech.data import (JointManifestDataset, homogeneous_collate,
                             phoneme_vocabulary_from_manifest, pilot_indices)
from eeg2speech.diffusion import (ConditionalMelDiffusion, denormalize_mel,
                                  normalize_mel)
from eeg2speech.model import DurationConditionedNativeRenderer, JointEEGContentModel


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available()
                        else ("cuda" if torch.cuda.is_available() else "cpu"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def optimizer_to(optimizer: torch.optim.Optimizer, target: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(target)


def mel_statistics(dataset: JointManifestDataset, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    channels: list[list[torch.Tensor]] = [[] for _ in range(80)]
    for index in indices:
        record = dataset[index]
        mel = record["native_speecht5_mel"][:, record["native_audio_mask"]].float()
        for channel in range(80):
            channels[channel].append(mel[channel].cpu())
    if not indices:
        raise RuntimeError("diffusion train split has no native mel")
    mean = torch.tensor([float(torch.cat(values).mean()) for values in channels])
    scale = torch.tensor([float(torch.cat(values).std(unbiased=False).clamp_min(1e-4)) for values in channels])
    return mean, scale


def load_renderer(path: Path, target: torch.device) -> DurationConditionedNativeRenderer:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not payload.get("gate") or not all(payload["gate"].values()):
        raise RuntimeError("diffusion refuses an ungated native renderer")
    model = DurationConditionedNativeRenderer(**payload.get("model_config", {})).to(target)
    model.load_state_dict(payload["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_eeg_checkpoint(path: Path, target: torch.device) -> JointEEGContentModel:
    """Load a frozen v3 EEG encoder for qualitative-only diffusion context."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("model_config", {})
    if int(config.get("teacher_dimension", 0)) <= 0:
        raise RuntimeError("EEG-conditioned diffusion requires a v3 frozen-teacher checkpoint")
    model = JointEEGContentModel(**config).to(target)
    model.load_state_dict(payload["model"], strict=False)
    templates = payload.get("target_templates")
    if not templates:
        raise RuntimeError("EEG-conditioned diffusion checkpoint lacks train-fold templates")
    model.set_target_templates(templates["mfcc_mean"], templates["mfcc_scale"], templates.get("hubert_mean"))
    if bool(config.get("duration_standardized", False)):
        keys = ("duration_log_mean", "duration_log_scale", "duration_min_frames", "duration_max_frames")
        if not all(key in templates for key in keys):
            raise RuntimeError("EEG-conditioned diffusion checkpoint lacks duration statistics")
        model.set_duration_statistics(*(templates[key] for key in keys))
    teacher = payload.get("speech_teacher")
    if not teacher:
        raise RuntimeError("EEG-conditioned diffusion checkpoint lacks frozen speech teacher")
    model.set_speech_teacher(teacher["mean"], teacher["components"], teacher["scale"], teacher["bank"], teacher["labels"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def eeg_conditioned_rows(eeg_model: JointEEGContentModel, renderer: DurationConditionedNativeRenderer,
                         batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Condition on EEG-predicted mel/duration and resample target to its support."""
    model_mask = batch.get("model_time_mask", batch["time_mask"])
    with torch.no_grad():
        state = eeg_model(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], model_mask, batch["dataset_id"])
        frames = state.predicted_duration.round().long().clamp(1, 1000)
        coarse, mask = renderer(state.mfcc, frames)
        maximum = int(mask.sum(1).max())
        rows = []
        for index, count in enumerate(frames.tolist()):
            source = batch["native_speecht5_mel"][index:index + 1, :, batch["native_audio_mask"][index]]
            value = torch.nn.functional.interpolate(source, size=int(count), mode="linear", align_corners=False)
            rows.append(torch.nn.functional.pad(value, (0, maximum - int(count))))
    return coarse, torch.cat(rows, dim=0), mask, state.global_embedding.detach()


def validate(diffusion: ConditionalMelDiffusion, renderer: DurationConditionedNativeRenderer,
             loader: DataLoader, mean: torch.Tensor, scale: torch.Tensor,
             target: torch.device, sampling_steps: int,
             eeg_model: JointEEGContentModel | None = None) -> dict[str, float]:
    diffusion.eval()
    coarse_errors: list[float] = []
    refined_errors: list[float] = []
    generator = torch.Generator(device="cpu").manual_seed(91827)
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(target) if torch.is_tensor(value) else value
                     for key, value in batch.items()}
            if eeg_model is None:
                coarse, mask = renderer(batch["content_mfcc"], batch["audio_duration_frames"])
                clean_target = batch["native_speecht5_mel"]
                if not torch.equal(mask, batch["native_audio_mask"]):
                    raise RuntimeError("diffusion validation duration/mask contract mismatch")
                context = None
            else:
                coarse, clean_target, mask, context = eeg_conditioned_rows(eeg_model, renderer, batch)
            normalized_coarse = normalize_mel(coarse, mean, scale)
            noise = torch.randn(normalized_coarse.shape, generator=generator).to(target)
            normalized_refined = diffusion.refine(normalized_coarse, mask, steps=sampling_steps, noise=noise, context=context)
            refined = denormalize_mel(normalized_refined, mean, scale)
            coarse_error = (coarse - clean_target).abs().mean(1)
            refined_error = (refined - clean_target).abs().mean(1)
            for index in range(len(coarse)):
                valid = mask[index]
                coarse_errors.append(float(coarse_error[index, valid].mean()))
                refined_errors.append(float(refined_error[index, valid].mean()))
    diffusion.train()
    coarse_mae = float(sum(coarse_errors) / len(coarse_errors))
    refined_mae = float(sum(refined_errors) / len(refined_errors))
    return {
        "pairs": len(coarse_errors),
        "coarse_native_mel_mae": coarse_mae,
        "diffusion_native_mel_mae": refined_mae,
        "diffusion_mel_improvement": float(1.0 - refined_mae / max(coarse_mae, 1e-8)),
    }


def checkpoint_payload(model: ConditionalMelDiffusion, optimizer: torch.optim.Optimizer,
                       completed: int, maximum_steps: int, history: list[dict],
                       artifact_hashes: dict[str, str], model_config: dict,
                       mean: torch.Tensor, scale: torch.Tensor) -> dict:
    return {
        "schema_version": "native-mel-conditional-diffusion-v1",
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "completed": int(completed), "maximum_steps": int(maximum_steps),
        "history": history, "artifact_hashes": artifact_hashes,
        "model_config": model_config, "mel_mean": mean.cpu(), "mel_scale": scale.cpu(),
        "torch_rng_state": torch.get_rng_state(),
        "mps_rng_state": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ds004940_conditioned_v2.yaml")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--eeg-checkpoint", type=Path,
                        help="optional frozen v3 EEG checkpoint for qualitative EEG-conditioned diffusion")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    spec = cfg["native_audio"]["diffusion"]
    maximum_steps = int(args.max_steps or spec["train_steps"])
    if maximum_steps < 1 or args.checkpoint_every < 1:
        raise ValueError("max-steps and checkpoint-every must be positive")
    required = tuple(path for path in (args.config, args.manifest, args.split, args.targets,
                                       args.normalizer, args.renderer, args.eeg_checkpoint) if path is not None)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"diffusion inputs are missing: {missing}")
    artifact_hashes = {path.name: file_sha256(path) for path in required}
    vocabulary = phoneme_vocabulary_from_manifest(args.manifest)
    train = JointManifestDataset(args.manifest, args.split, "train", "ds004940",
                                 args.targets, args.normalizer,
                                 float(cfg["loss"]["weak_content_weight"]),
                                 supervision_types={"paired_audio"}, phoneme_vocabulary=vocabulary)
    validation = JointManifestDataset(args.manifest, args.split, "validation", "ds004940",
                                      args.targets, args.normalizer,
                                      float(cfg["loss"]["weak_content_weight"]),
                                      supervision_types={"paired_audio"}, phoneme_vocabulary=vocabulary)
    train_indices = pilot_indices(train, cfg, "generalization", "train")
    validation_indices = pilot_indices(validation, cfg, "generalization", "validation")
    mean, scale = mel_statistics(train, train_indices)
    target = device()
    mean = mean.to(target)
    scale = scale.to(target)
    renderer = load_renderer(args.renderer, target)
    eeg_model = load_eeg_checkpoint(args.eeg_checkpoint, target) if args.eeg_checkpoint else None
    model_config = {
        "mel_bins": 80,
        "hidden_dimension": int(spec["hidden_dimension"]),
        "layers": int(spec["layers"]),
        "dropout": float(spec["dropout"]),
        "timesteps": int(spec["timesteps"]),
        "beta_start": float(spec["beta_start"]),
        "beta_end": float(spec["beta_end"]),
        "conditioning_dimension": int(eeg_model.teacher_dimension) if eeg_model is not None else 0,
    }
    diffusion = ConditionalMelDiffusion(**model_config).to(target)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=float(spec["learning_rate"]),
                                  weight_decay=float(spec["weight_decay"]))
    args.output.mkdir(parents=True, exist_ok=True)
    complete = args.output / "checkpoint.pt"
    progress = args.output / "training_state.pt"
    if complete.exists():
        payload = torch.load(complete, map_location="cpu", weights_only=False)
        if payload.get("artifact_hashes") != artifact_hashes or payload.get("model_config") != model_config:
            raise RuntimeError("completed diffusion checkpoint provenance/config changed")
        if not payload.get("gate") or not all(payload["gate"].values()):
            raise RuntimeError("completed diffusion checkpoint failed its validation gate")
        print(json.dumps({"status": "already_completed", "checkpoint": str(complete)}))
        train.close(); validation.close(); return 0
    torch.manual_seed(int(spec["seed"]))
    loader_generator = torch.Generator().manual_seed(int(spec["seed"]))
    loader = DataLoader(Subset(train, train_indices), batch_size=int(spec["batch_size"]),
                        shuffle=True, generator=loader_generator, collate_fn=homogeneous_collate)
    validation_loader = DataLoader(Subset(validation, validation_indices),
                                   batch_size=int(spec["batch_size"]), shuffle=False,
                                   collate_fn=homogeneous_collate)
    completed = 0
    history: list[dict] = []
    saved = None
    if progress.exists():
        saved = torch.load(progress, map_location="cpu", weights_only=False)
        if int(saved.get("maximum_steps", -1)) != maximum_steps:
            raise RuntimeError("diffusion partial checkpoint has a different --max-steps")
        if saved.get("artifact_hashes") != artifact_hashes or saved.get("model_config") != model_config:
            raise RuntimeError("diffusion partial checkpoint provenance/config changed")
        if not torch.equal(saved["mel_mean"], mean.cpu()) or not torch.equal(saved["mel_scale"], scale.cpu()):
            raise RuntimeError("diffusion train-fold mel statistics changed")
        diffusion.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        optimizer_to(optimizer, target)
        completed = int(saved["completed"])
        history = list(saved.get("history", []))
        print(json.dumps({"status": "resumed", "completed_steps": completed}))
    iterator = iter(loader)
    for _ in range(completed):
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(loader)
            next(iterator)
    if saved is not None and saved.get("torch_rng_state") is not None:
        torch.set_rng_state(saved["torch_rng_state"])
        if torch.backends.mps.is_available() and saved.get("mps_rng_state") is not None:
            torch.mps.set_rng_state(saved["mps_rng_state"])
    try:
        for step in range(completed + 1, maximum_steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {key: value.to(target) if torch.is_tensor(value) else value
                     for key, value in batch.items()}
            if eeg_model is None:
                with torch.no_grad():
                    coarse, mask = renderer(batch["content_mfcc"], batch["audio_duration_frames"])
                if not torch.equal(mask, batch["native_audio_mask"]):
                    raise RuntimeError("diffusion train duration/mask contract mismatch")
                clean_target = batch["native_speecht5_mel"]
                context = None
            else:
                coarse, clean_target, mask, context = eeg_conditioned_rows(eeg_model, renderer, batch)
            clean = normalize_mel(clean_target, mean, scale)
            condition = normalize_mel(coarse, mean, scale)
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.denoising_loss(clean, condition, mask, context=context)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite diffusion loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), float(spec["grad_clip"]))
            optimizer.step()
            completed = step
            if step == 1 or step % 100 == 0:
                history.append({"step": step, "diffusion_noise_mse": float(loss.detach())})
                print(json.dumps(history[-1]))
            if step % int(args.checkpoint_every) == 0:
                atomic_save(checkpoint_payload(diffusion, optimizer, completed, maximum_steps,
                                               history, artifact_hashes, model_config, mean, scale), progress)
    except KeyboardInterrupt:
        atomic_save(checkpoint_payload(diffusion, optimizer, completed, maximum_steps,
                                       history, artifact_hashes, model_config, mean, scale), progress)
        train.close(); validation.close()
        print(json.dumps({"status": "interrupted_resumable", "completed_steps": completed}))
        return 130
    metrics = validate(diffusion, renderer, validation_loader, mean, scale, target,
                       int(spec["sampling_steps"]), eeg_model=eeg_model)
    gate = {
        "validation_mel_improvement": metrics["diffusion_mel_improvement"]
        >= float(spec["validation_mel_improvement_min"]),
    }
    final = checkpoint_payload(diffusion, optimizer, completed, maximum_steps,
                               history, artifact_hashes, model_config, mean, scale)
    final.update({"validation": metrics, "gate": gate, "sampling_steps": int(spec["sampling_steps"]),
                  "conditioning": "frozen_eeg" if eeg_model is not None else "audio_only",
                  "warning": "qualitative mel refiner; excluded from deterministic EEG efficacy metrics"})
    final.pop("optimizer", None)
    atomic_save(final, complete)
    (args.output / "metrics.json").write_text(json.dumps({"history": history, "validation": metrics,
                                                          "gate": gate}, indent=2) + "\n")
    train.close(); validation.close()
    print(json.dumps({"output": str(args.output), "validation": metrics, "gate": gate}, indent=2))
    return 0 if all(gate.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
