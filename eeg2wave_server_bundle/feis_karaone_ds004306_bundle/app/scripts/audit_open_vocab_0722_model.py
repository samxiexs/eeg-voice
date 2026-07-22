#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm


APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path: sys.path.insert(0, str(APP))

from src.open_vocab_0722.data import AudioCodeBank, DATASETS, OpenVoiceEEGDataset, TeacherBank, collate_openvoice, common_montage_view, load_context, resolve_config_path  # noqa: E402
from src.open_vocab_0722.audio_gate import require_frozen_audio_checkpoint  # noqa: E402
from src.open_vocab_0722.lineage import build_lineage, file_sha256, validate_checkpoint  # noqa: E402
from src.open_vocab_0722.model import LabelFreeAudioConfig, LabelFreeAudioModel, OpenVoiceEEGConfig, OpenVoiceEEGEncoder  # noqa: E402


def configs(cfg: dict[str, Any], subjects: int) -> tuple[LabelFreeAudioConfig, OpenVoiceEEGConfig]:
    a, c, e = cfg["audio_model"], cfg["codec"], cfg["eeg_model"]
    audio = LabelFreeAudioConfig(codebooks=int(c["codebooks"]), code_steps=int(c["code_steps"]), vocab_size=int(c["vocab_size"]), d_model=int(a["d_model"]), condition_steps=int(a["condition_steps"]), encoder_layers=int(a["encoder_layers"]), decoder_layers=int(a["decoder_layers"]), heads=int(a["heads"]), dropout=float(a["dropout"]), text_dimension=int(cfg["teachers"]["text_dimension"]), xlsr_dimension=int(a["xlsr_dimension"]))
    eeg = OpenVoiceEEGConfig(eeg_samples=int(cfg["data"]["eeg_samples"]), patch_size=int(e["patch_size"]), patch_hop=int(e["patch_hop"]), d_model=int(e["d_model"]), condition_steps=int(e["condition_steps"]), code_steps=int(c["code_steps"]), heads=int(e["heads"]), latent_layers=int(e["latent_layers"]), dropout=float(e["dropout"]), specialists=int(e["specialists"]), specialist_bottleneck=int(e["specialist_bottleneck"]), soft_routing_epochs=int(e["soft_routing_epochs"]), top_k_specialists=int(e["top_k_specialists"]), expert_dropout=float(e["expert_dropout"]), num_train_subjects=subjects, adapter_moe_enabled=bool(e.get("adapter_moe_enabled", True)), text_dimension=int(cfg["teachers"]["text_dimension"]))
    return audio, eeg


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def load_models(context: Any, lineage: dict[str, Any], device: torch.device) -> tuple[LabelFreeAudioModel, OpenVoiceEEGEncoder, dict[str, Any]]:
    cfg = context.config; audio_cfg, eeg_cfg = configs(cfg, len(context.subject_to_index))
    audio_path = resolve_config_path(context.config_path, cfg["paths"]["audio_checkpoint"])
    require_frozen_audio_checkpoint(context.config_path, cfg, lineage, audio_path)
    eeg_path = resolve_config_path(context.config_path, cfg["paths"]["eeg_checkpoint"])
    audio_payload = torch.load(audio_path, map_location="cpu", weights_only=False)
    validate_checkpoint(audio_payload, phase="audio", lineage=lineage, source=str(audio_path))
    audio = LabelFreeAudioModel(audio_cfg).to(device); audio.load_state_dict(audio_payload["model_state"])
    eeg_payload = torch.load(eeg_path, map_location="cpu", weights_only=False)
    dependencies = {"audio_checkpoint_sha256": file_sha256(audio_path)}
    pretrain = resolve_config_path(context.config_path, cfg["paths"]["eeg_pretrain_checkpoint"])
    if "eeg_pretrain_checkpoint_sha256" in (eeg_payload.get("dependencies") or {}): dependencies["eeg_pretrain_checkpoint_sha256"] = file_sha256(pretrain)
    validate_checkpoint(eeg_payload, phase="eeg", lineage=lineage, dependencies=dependencies, source=str(eeg_path))
    if eeg_payload.get("audio_decoder_state") is not None: audio.decoder.load_state_dict(eeg_payload["audio_decoder_state"])
    eeg = OpenVoiceEEGEncoder(eeg_cfg).to(device); eeg.load_state_dict(eeg_payload["model_state"])
    audio.eval(); eeg.eval()
    return audio, eeg, eeg_payload


@torch.no_grad()
def extract(dataset: OpenVoiceEEGDataset, eeg: OpenVoiceEEGEncoder, audio: LabelFreeAudioModel, device: torch.device, epoch: int, batch_size: int) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_openvoice)
    output: defaultdict[str, list[Any]] = defaultdict(list)
    for raw in tqdm(loader, desc=f"[0722 audit] {dataset.split}", leave=False):
        batch = move(raw, device)
        common = common_montage_view(batch)
        full_value = eeg(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], batch["time_mask"], epoch=epoch)
        common_value = eeg(common["eeg"], common["channel_xyz"], common["channel_mask"], common["time_mask"], epoch=epoch)
        # Deterministic 25% channel removal, independent of dataset/subject.
        missing = dict(batch); mask = batch["channel_mask"].clone()
        for item in range(len(mask)):
            valid = torch.nonzero(mask[item], as_tuple=False).flatten()
            remove = valid[::4]
            if len(valid) - len(remove) < 1: remove = remove[:0]
            mask[item, remove] = False
        missing["channel_mask"] = mask; missing["eeg"] = batch["eeg"] * mask.unsqueeze(-1)
        missing_value = eeg(missing["eeg"], missing["channel_xyz"], missing["channel_mask"], missing["time_mask"], epoch=epoch)
        audio_value = audio.xlsr_encoder(batch["xlsr_tokens"])
        for key, value in (("full", full_value), ("common", common_value), ("missing", missing_value)):
            output[f"{key}_pooled"].append(value["pooled"].cpu())
            output[f"{key}_condition"].append(value["condition"].cpu())
            output[f"{key}_acoustic"].append(value["acoustic_global"].cpu())
            output[f"{key}_router"].append(value["router"]["sample_specialist_mass"].cpu())
            output[f"{key}_envelope"].append(value["envelope"].cpu())
        output["audio_acoustic"].append(audio_value["acoustic_global"].cpu())
        output["target_envelope"].append(batch["audio_envelope"].cpu())
        output["dataset"].extend(batch["dataset"])
        output["subject"].extend(batch["subject_group_id"])
        output["exact"].extend([value == "karaone_same_trial_overt" for value in batch["pairing_confidence"]])
    return {key: torch.cat(value) if value and torch.is_tensor(value[0]) else value for key, value in output.items()}


def retrieval_r5(eeg: torch.Tensor, audio: torch.Tensor) -> float:
    if not len(eeg): return float("nan")
    ranks = torch.argsort(F.normalize(eeg, dim=-1) @ F.normalize(audio, dim=-1).T, dim=1, descending=True)
    truth = torch.arange(len(eeg)).unsqueeze(1)
    return float((ranks[:, :5] == truth).any(dim=1).float().mean())


def envelope_corr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction - prediction.mean(dim=1, keepdim=True); target = target - target.mean(dim=1, keepdim=True)
    return float(((prediction * target).sum(dim=1) / torch.sqrt(prediction.square().sum(dim=1) * target.square().sum(dim=1)).clamp_min(1e-8)).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit montage shortcuts, routing and missing-channel robustness")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--project-audio-only", action="store_true")
    args = parser.parse_args()
    context = load_context(args.config); cfg = context.config
    lineage = build_lineage(context, require_optional_caches=not args.project_audio_only)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    audio, eeg, payload = load_models(context, lineage, device)
    teachers = TeacherBank(resolve_config_path(context.config_path, cfg["paths"]["teacher_cache"]))
    bank = AudioCodeBank(resolve_config_path(context.config_path, cfg["paths"]["project_audio_cache"]))
    train = OpenVoiceEEGDataset(context, bank, split="train", teachers=teachers)
    validation = OpenVoiceEEGDataset(context, bank, split="validation", teachers=teachers)
    batch_size = int(cfg["eeg_model"]["batch_size"])
    train_value = extract(train, eeg, audio, device, int(payload["epoch"]), batch_size)
    validation_value = extract(validation, eeg, audio, device, int(payload["epoch"]), batch_size)
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=15))
    classifier.fit(train_value["common_pooled"].numpy(), train_value["dataset"])
    dataset_ba = balanced_accuracy_score(validation_value["dataset"], classifier.predict(validation_value["common_pooled"].numpy()))
    cosine = F.cosine_similarity(validation_value["full_condition"], validation_value["common_condition"], dim=-1).median()
    exact = torch.tensor(validation_value["exact"], dtype=torch.bool)
    full_r5 = retrieval_r5(validation_value["full_acoustic"][exact], validation_value["audio_acoustic"][exact])
    missing_r5 = retrieval_r5(validation_value["missing_acoustic"][exact], validation_value["audio_acoustic"][exact])
    full_env = envelope_corr(validation_value["full_envelope"][exact], validation_value["target_envelope"][exact])
    missing_env = envelope_corr(validation_value["missing_envelope"][exact], validation_value["target_envelope"][exact])
    full_performance = 0.5 * (full_r5 + full_env); missing_performance = 0.5 * (missing_r5 + missing_env)
    masses: dict[str, Any] = {}
    for view in ("full", "common", "missing"):
        values = validation_value[f"{view}_router"]
        masses[f"{view}:global"] = values.mean(dim=0).tolist()
        for dataset in DATASETS:
            selected = torch.tensor([name == dataset for name in validation_value["dataset"]])
            if selected.any(): masses[f"{view}:dataset:{dataset}"] = values[selected].mean(dim=0).tolist()
        for subject in sorted(set(validation_value["subject"])):
            selected = torch.tensor([name == subject for name in validation_value["subject"]])
            if selected.any(): masses[f"{view}:subject:{subject}"] = values[selected].mean(dim=0).tolist()
    specialist_dataset_dominance = []
    for specialist in range(int(cfg["eeg_model"]["specialists"])):
        values = [masses[f"common:dataset:{dataset}"][specialist] for dataset in DATASETS]
        specialist_dataset_dominance.append(max(values) / max(sum(values), 1e-8))
    report = {
        "schema_version": "openvoice-0722-model-audit-v1", "split": "validation", "test_accessed": False,
        "project_audio_only_exploratory": bool(args.project_audio_only),
        "common14_dataset_balanced_accuracy": float(dataset_ba),
        "full_common_condition_cosine_median": float(cosine),
        "missing25": {"full_performance": full_performance, "missing_performance": missing_performance, "drop_fraction": max(0.0, (full_performance - missing_performance) / max(abs(full_performance), 1e-8)), "full_r5": full_r5, "missing_r5": missing_r5, "full_envelope": full_env, "missing_envelope": missing_env},
        "router_mass": masses, "specialist_dataset_dominance": specialist_dataset_dominance,
        "no_dying_or_collapsed_expert": all(0.05 <= value <= 0.60 for value in masses["common:global"]),
        "no_single_dataset_specialist": all(value < 0.80 for value in specialist_dataset_dominance),
        "thresholds": cfg["evaluation"]["moe"], "lineage": lineage,
        "audio_checkpoint_sha256": file_sha256(resolve_config_path(context.config_path, cfg["paths"]["audio_checkpoint"])),
        "eeg_checkpoint_sha256": file_sha256(resolve_config_path(context.config_path, cfg["paths"]["eeg_checkpoint"])),
    }
    output = args.output.resolve() if args.output else resolve_config_path(context.config_path, cfg["paths"]["output_root"]) / "evaluation/model_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__": main()
