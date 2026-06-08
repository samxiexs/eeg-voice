from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.alignment_model import EEGSpeechAlignmentModel
from src.dataset import FEISThinkingDataset, FEISProtocolDataset
from src.model import EEG2WaveVQModel
from src.utils import count_parameters, load_simple_yaml, resolve_bundle_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit FEIS waveform/alignment pipelines.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "config.yaml"))
    parser.add_argument("--alignment-config", default=str(BUNDLE_DIR / "configs" / "alignment.yaml"))
    parser.add_argument("--subject", default="01")
    parser.add_argument("--stage", default="thinking")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--waveform-checkpoint", default=None)
    parser.add_argument("--alignment-checkpoint", default=None)
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def tensor_shape(value: torch.Tensor) -> list[int]:
    return list(value.shape)


def audit_waveform_pipeline(args: argparse.Namespace) -> dict[str, object]:
    config = load_simple_yaml(args.config)
    if args.stage is not None:
        config["data"]["stage"] = args.stage
    data_root = resolve_bundle_path(args.data_root or config["data"]["root"], BUNDLE_DIR)
    dataset = FEISThinkingDataset(
        subject_id=args.subject,
        data_root=str(data_root),
        stage=config["data"]["stage"],
        split="train",
        train_ratio=float(config["data"]["train_ratio"]),
        val_ratio=float(config["data"]["val_ratio"]),
        audio_sr=int(config["audio"]["sample_rate"]),
        audio_dur=float(config["audio"]["duration_sec"]),
        audio_normalize=str(config["audio"].get("normalize", "rms")),
        audio_target_rms=float(config["audio"].get("target_rms", 0.08)),
        audio_max_gain=float(config["audio"].get("max_gain", 10.0)),
    )
    sample = dataset[0]
    eeg = sample["eeg"].unsqueeze(0)
    model = EEG2WaveVQModel(
        n_channels_eeg=int(config["model"]["n_channels_eeg"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        codebook_size=int(config["model"]["codebook_size"]),
        vq_beta=float(config["model"]["vq_beta"]),
        vq_decay=float(config["model"]["vq_decay"]),
        output_samples=int(config["audio"]["n_samples"]),
        num_labels=int(dataset.num_labels),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.waveform_checkpoint:
        checkpoint = torch.load(args.waveform_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    with torch.no_grad():
        z_e = model.encoder(eeg.to(device))
        z_q, vq_loss, codes, perplexity = model.quantizer(z_e)
        recon = model.decoder(z_q)
        logits = model.classifier(z_q)
    return {
        "verdicts": {
            "preprocess.py": "yes",
            "dataset.py": "no",
            "train.py": "partial / problematic",
            "model.py": "no",
        },
        "sample_trace": {
            "input_eeg": tensor_shape(eeg),
            "encoder_output": tensor_shape(z_e.cpu()),
            "vq_codes": tensor_shape(codes.cpu()),
            "decoder_output": tensor_shape(recon.cpu()),
            "classifier_logits": tensor_shape(logits.cpu()),
        },
        "parameter_counts": {
            "total": count_parameters(model),
            "encoder": count_parameters(model.encoder),
            "quantizer_buffers": int(model.quantizer.embedding.numel()),
            "decoder": count_parameters(model.decoder),
            "classifier": count_parameters(model.classifier),
        },
        "vq_usage": {
            "perplexity": float(perplexity.item()),
            "unique_codes_in_sample": int(torch.unique(codes).numel()),
            "code_length": int(codes.shape[-1]),
            "vq_loss": float(vq_loss.item()),
        },
        "subject_conditioning": {
            "waveform_baseline_has_subject_conditioning": False,
            "notes": "The baseline waveform path uses subject-specific templates implicitly through single-subject loading, but no explicit subject embedding exists in model.py.",
        },
    }


def audit_alignment_pipeline(args: argparse.Namespace) -> dict[str, object]:
    config = load_simple_yaml(args.alignment_config)
    if args.stage is not None:
        config["data"]["stage"] = args.stage
    data_root = resolve_bundle_path(args.data_root or config["data"]["root"], BUNDLE_DIR)
    dataset = FEISProtocolDataset(
        data_root=str(data_root),
        protocol=str(config["data"]["protocol"]),
        split="train",
        stage=str(config["data"]["stage"]),
        subject_id=config["data"].get("subject_id"),
        holdout_subject_id=config["data"].get("holdout_subject_id"),
        include_anomalous=bool(config["data"].get("include_anomalous", False)),
        ablation_mode=str(config["data"].get("ablation_mode", "none")),
        target_cache_path=resolve_bundle_path(config["targets"]["cache_path"], BUNDLE_DIR),
        require_targets=True,
        audio_sr=int(config["audio"]["sample_rate"]),
        audio_dur=float(config["audio"]["duration_sec"]),
        audio_normalize=str(config["audio"].get("normalize", "rms")),
        audio_target_rms=float(config["audio"].get("target_rms", 0.08)),
        audio_max_gain=float(config["audio"].get("max_gain", 10.0)),
        seed=int(config["train"]["seed"]),
    )
    sample = dataset[0]
    eeg = sample["eeg"].unsqueeze(0)
    model = EEGSpeechAlignmentModel(
        n_channels_eeg=int(config["model"]["n_channels_eeg"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        speech_embedding_dim=int(dataset.target_embedding_dim),
        prosody_dim=int(dataset.prosody_dim),
        num_labels=int(dataset.num_labels),
        latent_dim=int(config["model"].get("latent_dim", 192)),
        use_label_head=bool(config["model"].get("use_label_head", True)),
        use_subject_demo_head=bool(config["model"].get("use_subject_demo_head", False)),
        num_subjects=int(dataset.num_subjects),
        subject_embedding_dim=int(config["model"].get("subject_embedding_dim", 64)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.alignment_checkpoint:
        checkpoint = torch.load(args.alignment_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    with torch.no_grad():
        outputs = model(
            eeg.to(device),
            subject_indices=sample["subject_index"].view(1).to(device),
            valid_steps=torch.tensor([max(1, int(sample["eeg_valid_num_samples"].item()) // 32)], device=device),
        )
    trace = {
        "input_eeg": tensor_shape(eeg),
        "sequence_latent": tensor_shape(outputs["sequence_latent"].cpu()),
        "neural_speech_latent": tensor_shape(outputs["neural_speech_latent"].cpu()),
        "speech_embedding": tensor_shape(outputs["speech_embedding"].cpu()),
        "prosody": tensor_shape(outputs["prosody"].cpu()),
    }
    if "label_logits" in outputs:
        trace["label_logits"] = tensor_shape(outputs["label_logits"].cpu())
    if "subject_conditioned_embedding" in outputs:
        trace["subject_conditioned_embedding"] = tensor_shape(outputs["subject_conditioned_embedding"].cpu())
    return {
        "parameter_counts": {
            "total": count_parameters(model),
            "encoder": count_parameters(model.encoder),
            "projector": count_parameters(model.projector),
            "speech_head": count_parameters(model.speech_head),
            "prosody_head": count_parameters(model.prosody_head),
            "label_head": count_parameters(model.label_head) if model.label_head is not None else 0,
            "subject_demo_head": count_parameters(model.subject_demo_head) if model.subject_demo_head is not None else 0,
        },
        "sample_trace": trace,
        "subject_conditioning": {
            "alignment_primary_objective_uses_subject_conditioning": False,
            "alignment_demo_head_enabled": bool(config["model"].get("use_subject_demo_head", False)),
        },
    }


def main() -> None:
    args = parse_args()
    payload = {
        "waveform_baseline": audit_waveform_pipeline(args),
        "alignment_pipeline": audit_alignment_pipeline(args),
    }
    if args.output_path:
        candidate = Path(args.output_path)
        output_path = candidate if candidate.is_absolute() else Path.cwd() / candidate
    else:
        output_path = resolve_bundle_path("outputs/pipeline_audit.json", BUNDLE_DIR)
    write_json(output_path, payload)
    print(f"Saved pipeline audit to {output_path}")
    print(payload["waveform_baseline"]["verdicts"])


if __name__ == "__main__":
    main()
