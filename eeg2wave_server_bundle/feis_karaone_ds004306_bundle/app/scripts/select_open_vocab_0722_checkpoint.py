#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.signal import resample_poly
from torch.utils.data import DataLoader
from tqdm import tqdm


APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path: sys.path.insert(0, str(APP))
KARA_APP = APP.parents[1] / "karaone_overt_recon_bundle/app"
if str(KARA_APP) not in sys.path: sys.path.insert(0, str(KARA_APP))

from src.open_vocab_0722.data import AudioCodeBank, OpenVoiceEEGDataset, TeacherBank, collate_openvoice, load_context, resolve_config_path  # noqa: E402
from src.open_vocab_0722.audio_gate import require_frozen_audio_checkpoint  # noqa: E402
from src.open_vocab_0722.audio_io import read_wav  # noqa: E402
from src.open_vocab_0722.lineage import build_lineage, file_sha256, validate_checkpoint  # noqa: E402
from src.open_vocab_0722.metrics import reconstruction_metrics  # noqa: E402
from src.open_vocab_0722.model import LabelFreeAudioConfig, LabelFreeAudioModel, OpenVoiceEEGConfig, OpenVoiceEEGEncoder  # noqa: E402
from src.karaone_0715.codec import DiscreteEncodec, DiscreteEncodecConfig  # noqa: E402


def model_configs(cfg: dict[str, Any], subjects: int) -> tuple[LabelFreeAudioConfig, OpenVoiceEEGConfig]:
    a, c, e = cfg["audio_model"], cfg["codec"], cfg["eeg_model"]
    return (
        LabelFreeAudioConfig(codebooks=int(c["codebooks"]), code_steps=int(c["code_steps"]), vocab_size=int(c["vocab_size"]), d_model=int(a["d_model"]), condition_steps=int(a["condition_steps"]), encoder_layers=int(a["encoder_layers"]), decoder_layers=int(a["decoder_layers"]), heads=int(a["heads"]), dropout=float(a["dropout"]), text_dimension=int(cfg["teachers"]["text_dimension"]), xlsr_dimension=int(a["xlsr_dimension"])),
        OpenVoiceEEGConfig(eeg_samples=int(cfg["data"]["eeg_samples"]), patch_size=int(e["patch_size"]), patch_hop=int(e["patch_hop"]), d_model=int(e["d_model"]), condition_steps=int(e["condition_steps"]), code_steps=int(c["code_steps"]), heads=int(e["heads"]), latent_layers=int(e["latent_layers"]), dropout=float(e["dropout"]), specialists=int(e["specialists"]), specialist_bottleneck=int(e["specialist_bottleneck"]), soft_routing_epochs=int(e["soft_routing_epochs"]), top_k_specialists=int(e["top_k_specialists"]), expert_dropout=float(e["expert_dropout"]), num_train_subjects=subjects, adapter_moe_enabled=bool(e.get("adapter_moe_enabled", True)), text_dimension=int(cfg["teachers"]["text_dimension"])),
    )


def resample(audio: np.ndarray, source: int, target: int) -> np.ndarray:
    if source == target: return np.asarray(audio, dtype=np.float32)
    divisor = math.gcd(source, target)
    return resample_poly(audio, target // divisor, source // divisor).astype(np.float32)


def reference(context: Any, row: dict[str, str], rate: int, length: int) -> tuple[np.ndarray, int]:
    audio, source_rate = read_wav(context.audio_root / row["audio_relpath"])
    audio = resample(audio, int(source_rate), rate)
    output = np.zeros(length, dtype=np.float32); output[: min(length, len(audio))] = audio[:length]
    valid = round(int(row["audio_valid_samples"]) * rate / int(source_rate))
    return output, max(16, min(valid, length))


def subject_macro(values: list[tuple[str, float]]) -> float:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for subject, value in values: grouped[subject].append(value)
    return float(np.mean([np.mean(items) for items in grouped.values()]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Select EEG checkpoint by decoded KaraOne validation composite")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--generalization", choices=("g1", "g2", "g3"), default="g1")
    parser.add_argument("--holdout-label", default=None)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project-audio-only", action="store_true")
    args = parser.parse_args()
    context = load_context(args.config); cfg = context.config
    lineage = build_lineage(context, require_optional_caches=not args.project_audio_only)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    audio_config, eeg_config = model_configs(cfg, len(context.subject_to_index))
    audio_path = resolve_config_path(context.config_path, cfg["paths"]["audio_checkpoint"])
    require_frozen_audio_checkpoint(context.config_path, cfg, lineage, audio_path)
    audio_payload = torch.load(audio_path, map_location="cpu", weights_only=False)
    validate_checkpoint(audio_payload, phase="audio", lineage=lineage, source=str(audio_path))
    base_audio = LabelFreeAudioModel(audio_config).to(device); base_audio.load_state_dict(audio_payload["model_state"]); base_audio.eval()
    dependencies = {"audio_checkpoint_sha256": file_sha256(audio_path)}
    pretrain = resolve_config_path(context.config_path, cfg["paths"]["eeg_pretrain_checkpoint"])
    if pretrain.is_file(): dependencies["eeg_pretrain_checkpoint_sha256"] = file_sha256(pretrain)
    candidate_dir = args.candidates.resolve() if args.candidates else resolve_config_path(context.config_path, cfg["paths"]["output_root"]) / "eeg/checkpoints/candidates"
    candidates = sorted(candidate_dir.glob("epoch_*.pt"))
    proxy = resolve_config_path(context.config_path, cfg["paths"]["output_root"]) / "eeg/checkpoints/proxy_best.pt"
    if proxy.is_file(): candidates.append(proxy)
    candidates = sorted(set(candidates))
    if not candidates: raise FileNotFoundError(f"No EEG candidates under {candidate_dir}")
    teachers = TeacherBank(resolve_config_path(context.config_path, cfg["paths"]["teacher_cache"]))
    bank = AudioCodeBank(resolve_config_path(context.config_path, cfg["paths"]["project_audio_cache"]))
    dataset = OpenVoiceEEGDataset(context, bank, split="validation", generalization=args.generalization, holdout_label=args.holdout_label, datasets=("karaone",), teachers=teachers)
    if args.limit >= 0: dataset.rows = dataset.rows[: args.limit]
    loader = DataLoader(dataset, batch_size=int(cfg["eeg_model"]["batch_size"]), shuffle=False, collate_fn=collate_openvoice)
    codec_cfg = cfg["codec"]
    codec = DiscreteEncodec(DiscreteEncodecConfig(model_path=str(resolve_config_path(context.config_path, cfg["paths"]["encodec_model"])), sample_rate=int(codec_cfg["sample_rate"]), duration_sec=float(codec_cfg["duration_sec"]), bandwidth=float(codec_cfg["bandwidth"])), device)
    target_length = round(codec.codec_sample_rate * float(codec_cfg["duration_sec"]))
    results = []
    for candidate_path in tqdm(candidates, desc="[0722 select] candidates", unit="checkpoint"):
        payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
        expected_dependencies = {key: value for key, value in dependencies.items() if key in (payload.get("dependencies") or {})}
        validate_checkpoint(payload, phase="eeg", lineage=lineage, dependencies=expected_dependencies, source=str(candidate_path))
        eeg = OpenVoiceEEGEncoder(eeg_config).to(device); eeg.load_state_dict(payload["model_state"]); eeg.eval()
        audio = LabelFreeAudioModel(audio_config).to(device); audio.load_state_dict(audio_payload["model_state"])
        if payload.get("audio_decoder_state") is not None: audio.decoder.load_state_dict(payload["audio_decoder_state"])
        audio.eval(); metrics_by_trial: list[tuple[str, dict[str, float]]] = []; eeg_embeddings = []; audio_embeddings = []
        offset = 0
        with torch.no_grad():
            for raw in tqdm(loader, desc=f"[0722 select] {candidate_path.stem}", unit="batch", leave=False):
                batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in raw.items()}
                output = eeg(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], batch["time_mask"], epoch=int(payload["epoch"]))
                teacher = audio.xlsr_encoder(batch["xlsr_tokens"])
                generated = audio.decoder.generate(output["condition"], steps=int(cfg["evaluation"]["maskgit_steps"]), temperature=float(cfg["evaluation"]["synthesis_temperature"]))
                waveforms = codec.decode(generated.cpu().numpy(), scale=None)
                eeg_embeddings.append(output["acoustic_global"].cpu()); audio_embeddings.append(teacher["acoustic_global"].cpu())
                for item, waveform in enumerate(waveforms):
                    row = dataset.rows[offset + item]; target, valid = reference(context, row, codec.codec_sample_rate, target_length)
                    candidate = np.pad(waveform[:target_length], (0, max(0, target_length - len(waveform))))
                    metrics_by_trial.append((row["subject_group_id"], reconstruction_metrics(target[:valid], candidate[:valid], codec.codec_sample_rate, max_lag_ms=float(cfg["evaluation"]["max_envelope_lag_ms"]))))
                offset += len(waveforms)
        first = F.normalize(torch.cat(eeg_embeddings), dim=-1); second = F.normalize(torch.cat(audio_embeddings), dim=-1)
        ranks = torch.argsort(first @ second.T, dim=1, descending=True); truth = torch.arange(len(first)).unsqueeze(1)
        r5 = float((ranks[:, :5] == truth).any(dim=1).float().mean())
        per_trial = [(subject, 0.35 * value["lag_envelope_correlation"] + 0.25 * value["modulation_correlation"] + 0.20 * (1.0 - np.clip(value["log_mel_mae_db"] / 12.0, 0.0, 1.0)) + 0.20 * r5) for subject, value in metrics_by_trial]
        score = subject_macro(per_trial)
        results.append({"checkpoint": str(candidate_path), "checkpoint_sha256": file_sha256(candidate_path), "epoch": int(payload["epoch"]) + 1, "composite": score, "retrieval_r5": r5, "envelope_correlation": subject_macro([(subject, value["lag_envelope_correlation"]) for subject, value in metrics_by_trial]), "modulation_correlation": subject_macro([(subject, value["modulation_correlation"]) for subject, value in metrics_by_trial]), "log_mel_mae_db": subject_macro([(subject, value["log_mel_mae_db"]) for subject, value in metrics_by_trial]), "router_checkpoint_eligible": bool((payload.get("metrics") or {}).get("router_checkpoint_eligible", False))})
    eligible = [value for value in results if value["router_checkpoint_eligible"]]
    if not eligible: raise RuntimeError("No decoded candidate is router-eligible")
    selected = max(eligible, key=lambda value: value["composite"])
    selected_payload = torch.load(selected["checkpoint"], map_location="cpu", weights_only=False)
    selected_payload["selection"] = {"schema_version": "openvoice-decoded-selection-v1", "selection_split": "validation", "dataset": "karaone", "test_accessed": False, "generalization": args.generalization, "holdout_label": args.holdout_label, "metric": "subject_macro_decoded_composite", **selected}
    destination = resolve_config_path(context.config_path, cfg["paths"]["eeg_checkpoint"])
    destination.parent.mkdir(parents=True, exist_ok=True); torch.save(selected_payload, destination)
    report_path = resolve_config_path(context.config_path, cfg["paths"]["output_root"]) / "eeg/metrics/checkpoint_selection.json"
    report = {"schema_version": "openvoice-0722-checkpoint-selection-v1", "selected_checkpoint": str(destination), "selected_checkpoint_sha256": file_sha256(destination), "selected": selected, "candidates": results, "selection_split": "validation", "test_accessed": False, "lineage": lineage}
    report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected": str(destination), "epoch": selected["epoch"], "composite": selected["composite"], "report": str(report_path)}, indent=2))


if __name__ == "__main__": main()
