#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
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
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
KARA_APP = APP.parents[1] / "karaone_overt_recon_bundle/app"
if str(KARA_APP) not in sys.path:
    sys.path.insert(0, str(KARA_APP))

from src.combined_0715.audio_eval import decode_cached_sample  # noqa: E402
from src.open_vocab_0722.data import AudioCodeBank, OpenVoiceEEGDataset, TeacherBank, collate_openvoice, load_context, normalize_label, resolve_config_path  # noqa: E402
from src.open_vocab_0722.audio_io import read_wav, write_wav  # noqa: E402
from src.open_vocab_0722.lineage import build_lineage, file_sha256, preauthorize_test, preauthorize_test_metadata, validate_checkpoint  # noqa: E402
from src.open_vocab_0722.metrics import reconstruction_metrics, summarize  # noqa: E402
from src.open_vocab_0722.model import LabelFreeAudioConfig, LabelFreeAudioModel, OpenVoiceEEGConfig, OpenVoiceEEGEncoder  # noqa: E402
from src.karaone_0715.codec import DiscreteEncodec, DiscreteEncodecConfig  # noqa: E402


MODES = (
    "reference", "codec_oracle", "audio_condition_oracle", "eeg_conditioned",
    "shuffled_eeg_same_label", "shuffled_eeg_any", "channel_shuffled", "zero_eeg",
    "text_only_ablation", "dataset_prior",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OpenVoice 0722 reference/control reconstruction WAVs")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("karaone", "feis", "ds004306"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--generalization", choices=("g1", "g2", "g3"), default="g1")
    parser.add_argument("--holdout-label", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-final-test", action="store_true")
    parser.add_argument("--compute-xlsr", action="store_true")
    parser.add_argument("--project-audio-only", action="store_true")
    return parser.parse_args()


def device_default() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def audio_cfg(cfg: dict[str, Any]) -> LabelFreeAudioConfig:
    model, codec = cfg["audio_model"], cfg["codec"]
    return LabelFreeAudioConfig(codebooks=int(codec["codebooks"]), code_steps=int(codec["code_steps"]), vocab_size=int(codec["vocab_size"]), d_model=int(model["d_model"]), condition_steps=int(model["condition_steps"]), encoder_layers=int(model["encoder_layers"]), decoder_layers=int(model["decoder_layers"]), heads=int(model["heads"]), dropout=float(model["dropout"]), text_dimension=int(cfg["teachers"]["text_dimension"]), xlsr_dimension=int(model["xlsr_dimension"]))


def eeg_cfg(cfg: dict[str, Any], subjects: int) -> OpenVoiceEEGConfig:
    model = cfg["eeg_model"]
    return OpenVoiceEEGConfig(eeg_samples=int(cfg["data"]["eeg_samples"]), patch_size=int(model["patch_size"]), patch_hop=int(model["patch_hop"]), d_model=int(model["d_model"]), condition_steps=int(model["condition_steps"]), code_steps=int(cfg["codec"]["code_steps"]), heads=int(model["heads"]), latent_layers=int(model["latent_layers"]), dropout=float(model["dropout"]), specialists=int(model["specialists"]), specialist_bottleneck=int(model["specialist_bottleneck"]), soft_routing_epochs=int(model["soft_routing_epochs"]), top_k_specialists=int(model["top_k_specialists"]), expert_dropout=float(model["expert_dropout"]), num_train_subjects=subjects, adapter_moe_enabled=bool(model.get("adapter_moe_enabled", True)), text_dimension=int(cfg["teachers"]["text_dimension"]))


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def load_models(context: Any, lineage: dict[str, Any], device: torch.device) -> tuple[LabelFreeAudioModel, OpenVoiceEEGEncoder, dict[str, Any], dict[str, Any], Path, Path]:
    cfg = context.config
    audio_path = resolve_config_path(context.config_path, cfg["paths"]["audio_checkpoint"])
    eeg_path = resolve_config_path(context.config_path, cfg["paths"]["eeg_checkpoint"])
    audio_payload = torch.load(audio_path, map_location="cpu", weights_only=False)
    validate_checkpoint(audio_payload, phase="audio", lineage=lineage, source=str(audio_path))
    audio = LabelFreeAudioModel(audio_cfg(cfg)).to(device)
    audio.load_state_dict(audio_payload["model_state"])
    eeg_payload = torch.load(eeg_path, map_location="cpu", weights_only=False)
    dependencies = {"audio_checkpoint_sha256": file_sha256(audio_path)}
    pretrain = resolve_config_path(context.config_path, cfg["paths"]["eeg_pretrain_checkpoint"])
    if "eeg_pretrain_checkpoint_sha256" in (eeg_payload.get("dependencies") or {}):
        dependencies["eeg_pretrain_checkpoint_sha256"] = file_sha256(pretrain)
    validate_checkpoint(eeg_payload, phase="eeg", lineage=lineage, dependencies=dependencies, source=str(eeg_path))
    if eeg_payload.get("audio_decoder_state") is not None:
        audio.decoder.load_state_dict(eeg_payload["audio_decoder_state"])
    eeg = OpenVoiceEEGEncoder(eeg_cfg(cfg, len(context.subject_to_index))).to(device)
    eeg.load_state_dict(eeg_payload["model_state"])
    audio.eval(); eeg.eval()
    return audio, eeg, audio_payload, eeg_payload, audio_path, eeg_path


def derangements(rows: tuple[dict[str, str], ...]) -> tuple[list[int], list[int]]:
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_label[normalize_label(row["label"])].append(index)
    same = [-1] * len(rows)
    for indices in by_label.values():
        if len(indices) < 2:
            continue
        for position, index in enumerate(indices):
            same[index] = indices[(position + 1) % len(indices)]
    any_shuffle = [(index + 1) % len(rows) for index in range(len(rows))]
    return same, any_shuffle


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def resample(audio: np.ndarray, source: int, target: int) -> np.ndarray:
    if source == target: return np.asarray(audio, dtype=np.float32)
    divisor = math.gcd(source, target)
    return resample_poly(audio, target // divisor, source // divisor).astype(np.float32)


def reference_waveform(context: Any, row: dict[str, str], target_rate: int, target_length: int) -> tuple[np.ndarray, int]:
    audio, rate = read_wav(context.audio_root / row["audio_relpath"])
    audio = resample(audio, int(rate), target_rate)
    valid = min(len(audio), round(int(row["audio_valid_samples"]) * target_rate / int(rate)))
    output = np.zeros(target_length, dtype=np.float32)
    output[: min(target_length, len(audio))] = audio[:target_length]
    return output, max(16, min(valid, target_length))


def rms_normalize(audio: np.ndarray) -> np.ndarray:
    rms = math.sqrt(float(np.mean(np.asarray(audio, dtype=np.float64) ** 2)) + 1e-12)
    return np.asarray(audio / rms * 0.08 if rms > 1e-8 else audio, dtype=np.float32)


@torch.no_grad()
def dataset_prior_condition(context: Any, dataset_name: str, teachers: TeacherBank, audio: LabelFreeAudioModel, device: torch.device) -> torch.Tensor:
    keys = sorted({row["audio_key"] for row in context.rows if row["dataset"] == dataset_name and context.split_for(row) == "train" and teachers.audio_tokens.get(row["audio_key"]) is not None})
    if not keys:
        return torch.zeros(1, audio.cfg.condition_steps, audio.cfg.d_model, device=device)
    accumulator = torch.zeros(1, audio.cfg.condition_steps, audio.cfg.d_model, device=device)
    for start in range(0, len(keys), 32):
        values = torch.from_numpy(np.stack([teachers.audio_tokens.get(key) for key in keys[start : start + 32]])).float().to(device)
        accumulator += audio.xlsr_encoder(values)["condition"].sum(dim=0, keepdim=True)
    return accumulator / len(keys)


def channel_shuffle(batch: dict[str, Any]) -> dict[str, Any]:
    output = dict(batch)
    eeg = batch["eeg"].clone()
    for index in range(len(eeg)):
        valid = torch.nonzero(batch["channel_mask"][index], as_tuple=False).flatten()
        if len(valid) > 1:
            eeg[index, valid] = eeg[index, torch.roll(valid, shifts=1)]
    output["eeg"] = eeg
    return output


def code_accuracy(codes: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    return {f"q{index}_accuracy": float((codes[index][valid[index]] == target[index][valid[index]]).float().mean()) for index in (0, 1)}


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    raw_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audio_path = resolve_config_path(config_path, raw_cfg["paths"]["audio_checkpoint"])
    eeg_path = resolve_config_path(config_path, raw_cfg["paths"]["eeg_checkpoint"])
    gate_path = resolve_config_path(config_path, raw_cfg["paths"]["validation_gate"])
    if args.split == "test":
        if not args.allow_final_test: raise PermissionError("Locked test synthesis requires --allow-final-test")
        preauthorize_test_metadata(gate_path, config_path=config_path, audio_checkpoint=audio_path, eeg_checkpoint=eeg_path)
    context = load_context(config_path)
    lineage = build_lineage(context, require_optional_caches=not args.project_audio_only)
    if args.split == "test":
        preauthorize_test(gate_path, lineage=lineage, audio_checkpoint=audio_path, eeg_checkpoint=eeg_path)
    device = torch.device(args.device) if args.device else device_default()
    teachers = TeacherBank(resolve_config_path(config_path, context.config["paths"]["teacher_cache"]))
    bank = AudioCodeBank(resolve_config_path(config_path, context.config["paths"]["project_audio_cache"]))
    audio, eeg, audio_payload, eeg_payload, audio_path, eeg_path = load_models(context, lineage, device)
    dataset = OpenVoiceEEGDataset(context, bank, split=args.split, generalization=args.generalization, holdout_label=args.holdout_label, datasets=(args.dataset,), teachers=teachers)
    same_shuffle, any_shuffle = derangements(dataset.rows)
    indices = list(range(len(dataset)))
    if args.limit >= 0: indices = indices[: args.limit]
    codec_cfg = context.config["codec"]
    codec = DiscreteEncodec(DiscreteEncodecConfig(model_path=str(resolve_config_path(config_path, context.config["paths"]["encodec_model"])), sample_rate=int(codec_cfg["sample_rate"]), duration_sec=float(codec_cfg["duration_sec"]), bandwidth=float(codec_cfg["bandwidth"])), device)
    target_length = round(codec.codec_sample_rate * float(codec_cfg["duration_sec"]))
    destination = (args.output.resolve() if args.output else resolve_config_path(config_path, context.config["paths"]["output_root"]) / "synthesis") / args.generalization / (args.holdout_label or "all") / args.dataset / args.split
    folders = {mode: destination / mode for mode in MODES}
    for path in folders.values(): path.mkdir(parents=True, exist_ok=True)
    prior = dataset_prior_condition(context, args.dataset, teachers, audio, device)
    xlsr_processor = xlsr_model = None
    if args.compute_xlsr:
        from transformers import AutoModel, AutoProcessor
        xlsr_processor = AutoProcessor.from_pretrained(context.config["teachers"]["xlsr_model"])
        xlsr_model = AutoModel.from_pretrained(context.config["teachers"]["xlsr_model"]).to(device).eval()
    records, aggregate = [], defaultdict(list)
    retrieval_eeg: list[torch.Tensor] = []
    retrieval_audio: list[torch.Tensor] = []
    for output_index, index in enumerate(tqdm(indices, desc=f"[0722 synthesis] {args.dataset}/{args.split}", unit="sample")):
        sample, row = dataset[index], dataset.rows[index]
        batch = move(collate_openvoice([sample]), device)
        main_output = eeg(batch["eeg"], batch["channel_xyz"], batch["channel_mask"], batch["time_mask"], epoch=int(eeg_payload["epoch"]))
        zero = dict(batch); zero["eeg"] = torch.zeros_like(batch["eeg"])
        zero_output = eeg(zero["eeg"], zero["channel_xyz"], zero["channel_mask"], zero["time_mask"], epoch=int(eeg_payload["epoch"]))
        channel_output = eeg(**{key: value for key, value in channel_shuffle(batch).items() if key in {"eeg", "channel_xyz", "channel_mask", "time_mask"}}, epoch=int(eeg_payload["epoch"]))
        same_index = same_shuffle[index]
        same_output = zero_output if same_index < 0 else eeg(**{key: value for key, value in move(collate_openvoice([dataset[same_index]]), device).items() if key in {"eeg", "channel_xyz", "channel_mask", "time_mask"}}, epoch=int(eeg_payload["epoch"]))
        any_index = any_shuffle[index]
        any_output = eeg(**{key: value for key, value in move(collate_openvoice([dataset[any_index]]), device).items() if key in {"eeg", "channel_xyz", "channel_mask", "time_mask"}}, epoch=int(eeg_payload["epoch"]))
        audio_features = audio.xlsr_encoder(batch["xlsr_tokens"])
        audio_oracle = audio_features["condition"]
        if row["pairing_confidence"] == "karaone_same_trial_overt":
            retrieval_eeg.append(main_output["acoustic_global"].detach().cpu())
            retrieval_audio.append(audio_features["acoustic_global"].detach().cpu())
        text_condition = eeg.project_text(batch["text_embedding"]).unsqueeze(1).expand(-1, audio.cfg.condition_steps, -1)
        conditions = {
            "audio_condition_oracle": audio_oracle, "eeg_conditioned": main_output["condition"],
            "shuffled_eeg_same_label": same_output["condition"], "shuffled_eeg_any": any_output["condition"],
            "channel_shuffled": channel_output["condition"], "zero_eeg": zero_output["condition"],
            "text_only_ablation": text_condition, "dataset_prior": prior,
        }
        generated = {name: audio.decoder.generate(condition, steps=int(context.config["evaluation"]["maskgit_steps"]), temperature=float(context.config["evaluation"]["synthesis_temperature"]))[0] for name, condition in conditions.items()}
        audio_index = int(sample["audio_idx"])
        decoded = {"codec_oracle": decode_cached_sample(codec, bank.codes[audio_index], bank.scale[audio_index], bool(bank.scale_valid[audio_index]))}
        decoded.update({name: codec.decode(codes.cpu().numpy(), scale=None) for name, codes in generated.items()})
        reference, valid = reference_waveform(context, row, codec.codec_sample_rate, target_length)
        decoded = {name: np.pad(value[:target_length], (0, max(0, target_length - len(value)))) for name, value in decoded.items()}
        stem = f"{output_index:04d}_{safe(row['sample_key'])}"
        write_wav(folders["reference"] / f"{stem}.wav", rms_normalize(reference), codec.codec_sample_rate)
        mode_metrics: dict[str, Any] = {}
        for name, waveform in decoded.items():
            write_wav(folders[name] / f"{stem}.wav", rms_normalize(waveform), codec.codec_sample_rate)
            metrics = reconstruction_metrics(reference[:valid], waveform[:valid], codec.codec_sample_rate, max_lag_ms=float(context.config["evaluation"]["max_envelope_lag_ms"]))
            prediction_codes = bank.codes[audio_index] if name == "codec_oracle" else generated[name]
            metrics.update(code_accuracy(torch.as_tensor(prediction_codes), sample["codes"], sample["code_valid_mask"]))
            metrics["xlsr_content_cosine"] = float("nan")
            mode_metrics[name] = metrics
            aggregate[name].append(metrics)
        if xlsr_model is not None and xlsr_processor is not None:
            candidates = [resample(decoded[name][:valid], codec.codec_sample_rate, 16000) for name in decoded]
            encoded = xlsr_processor(candidates, sampling_rate=16000, padding=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad(): candidate_content = xlsr_model(**encoded).last_hidden_state.mean(dim=1)
            reference_content = batch["xlsr_tokens"].mean(dim=1)
            for candidate_index, name in enumerate(decoded):
                cosine = float(F.cosine_similarity(reference_content, candidate_content[candidate_index : candidate_index + 1], dim=-1))
                mode_metrics[name]["xlsr_content_cosine"] = cosine
                aggregate[name][-1]["xlsr_content_cosine"] = cosine
        records.append({
            "sample_key": row["sample_key"], "audio_key": row["audio_key"], "subject_group_id": row["subject_group_id"],
            "label": row["label"], "pairing_confidence": row["pairing_confidence"],
            "trial_level_claim_allowed": row["pairing_confidence"] == "karaone_same_trial_overt",
            "same_label_shuffle_source": dataset.rows[same_index]["sample_key"] if same_index >= 0 else None,
            "same_label_shuffle_available": same_index >= 0,
            "any_shuffle_source": dataset.rows[any_index]["sample_key"],
            "mode_metrics": mode_metrics,
            "files": {mode: f"{mode}/{stem}.wav" for mode in MODES},
        })
    retrieval = {"r1": float("nan"), "r5": float("nan"), "mrr": float("nan"), "chance_r5": float("nan"), "n": 0}
    if retrieval_eeg:
        eeg_embedding = F.normalize(torch.cat(retrieval_eeg), dim=-1)
        audio_embedding = F.normalize(torch.cat(retrieval_audio), dim=-1)
        ranks = torch.argsort(eeg_embedding @ audio_embedding.T, dim=1, descending=True)
        truth = torch.arange(len(ranks)).unsqueeze(1)
        positions = (ranks == truth).nonzero(as_tuple=False)[:, 1].float() + 1.0
        retrieval = {
            "r1": float((ranks[:, :1] == truth).any(dim=1).float().mean()),
            "r5": float((ranks[:, :5] == truth).any(dim=1).float().mean()),
            "mrr": float((1.0 / positions).mean()),
            "chance_r5": min(1.0, 5.0 / len(ranks)), "n": len(ranks),
        }
    manifest = {
        "schema_version": "openvoice-0722-synthesis-v1", "dataset": args.dataset, "split": args.split,
        "generalization": args.generalization, "holdout_label": args.holdout_label, "modes": list(MODES),
        "audio_checkpoint": str(audio_path), "audio_checkpoint_sha256": file_sha256(audio_path),
        "eeg_checkpoint": str(eeg_path), "eeg_checkpoint_sha256": file_sha256(eeg_path),
        "lineage": lineage, "inference_inputs": ["eeg", "channel_xyz", "channel_mask", "time_mask"],
        "main_generation_uses_label": False, "main_generation_uses_dataset_id": False, "main_generation_uses_subject_id": False,
        "project_audio_only_exploratory": bool(args.project_audio_only),
        "text_only_is_ablation": True, "dataset_prior_is_ablation": True,
        "reference_audio_used_only_for_oracles_and_evaluation": True,
        "ds004306_trial_level_claim_allowed": False if args.dataset == "ds004306" else None,
        "xlsr_content_computed": bool(args.compute_xlsr), "phoneme_word_metrics_available": False,
        "summary": {name: summarize(values) for name, values in aggregate.items()}, "retrieval": retrieval, "samples": records,
        "test_accessed": args.split == "test",
    }
    (destination / "synthesis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "samples": len(records), "modes": len(MODES), "test_accessed": args.split == "test"}, indent=2))


if __name__ == "__main__":
    main()
