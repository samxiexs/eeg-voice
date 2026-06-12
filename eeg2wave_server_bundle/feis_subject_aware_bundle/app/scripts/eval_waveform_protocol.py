from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.dataset import FEISProtocolDataset
from src.losses import compute_total_loss, stft_distance_single
from src.model import EEG2WaveVQModel
from src.subject_conditioned_waveform import SubjectConditionedEEG2WaveVQModel
from src.utils import load_simple_yaml, resolve_bundle_path, save_wav, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FEIS waveform protocol baselines.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "waveform_protocol.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--holdout-subject", default=None)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--ablation-mode", default=None)
    parser.add_argument("--include-anomalous", action="store_true")
    parser.add_argument(
        "--retrieval-bank",
        default="auto",
        choices=["auto", "train_templates", "oracle_test_templates"],
    )
    return parser.parse_args()


def build_run_name(config: dict, args: argparse.Namespace) -> str:
    protocol = str(config["data"]["protocol"]).upper()
    run_name = f"{protocol.lower()}_{config['data']['stage']}_{config['data'].get('ablation_mode', 'none')}"
    if protocol == "S":
        run_name += f"_subject_{args.subject or config['data'].get('subject_id')}"
    if protocol == "U":
        run_name += f"_holdout_{args.holdout_subject or config['data'].get('holdout_subject_id')}"
    if config["model"].get("use_subject_conditioning", False):
        run_name += "_subject_conditioned"
    run_tag = os.environ.get("FEIS_RUN_TAG", "").strip()
    if run_tag:
        run_name += f"_{run_tag}"
    return run_name


def build_dataset(config: dict, args: argparse.Namespace, split: str, include_anomalous: bool | None = None) -> FEISProtocolDataset:
    cfg_data = config["data"]
    cfg_audio = config["audio"]
    if include_anomalous is None:
        include_anomalous = bool(args.include_anomalous or cfg_data.get("include_anomalous", False))
    return FEISProtocolDataset(
        data_root=str(resolve_bundle_path(args.data_root or cfg_data["root"], BUNDLE_DIR)),
        protocol=str(args.protocol or cfg_data["protocol"]).upper(),
        split=split,
        stage=str(args.stage or cfg_data["stage"]),
        subject_id=args.subject or cfg_data.get("subject_id"),
        holdout_subject_id=args.holdout_subject or cfg_data.get("holdout_subject_id"),
        include_anomalous=include_anomalous,
        ablation_mode=str(args.ablation_mode or cfg_data.get("ablation_mode", "none")),
        target_cache_path=None,
        require_targets=False,
        audio_sr=int(cfg_audio["sample_rate"]),
        audio_dur=float(cfg_audio["duration_sec"]),
        audio_normalize=str(cfg_audio.get("normalize", "rms")),
        audio_target_rms=float(cfg_audio.get("target_rms", 0.08)),
        audio_max_gain=float(cfg_audio.get("max_gain", 10.0)),
        seed=int(config["train"]["seed"]),
    )


def build_model(config: dict, dataset: FEISProtocolDataset) -> tuple[torch.nn.Module, bool]:
    cfg_model = config["model"]
    subject_conditioning = bool(cfg_model.get("use_subject_conditioning", False))
    common = dict(
        n_channels_eeg=int(cfg_model["n_channels_eeg"]),
        hidden_dim=int(cfg_model["hidden_dim"]),
        codebook_size=int(cfg_model["codebook_size"]),
        vq_beta=float(cfg_model["vq_beta"]),
        vq_decay=float(cfg_model["vq_decay"]),
        output_samples=int(config["audio"]["n_samples"]),
        num_labels=int(dataset.num_labels),
    )
    if subject_conditioning:
        model = SubjectConditionedEEG2WaveVQModel(
            **common,
            num_subjects=int(dataset.num_subjects),
            subject_embedding_dim=int(cfg_model.get("subject_embedding_dim", 64)),
        )
    else:
        model = EEG2WaveVQModel(**common)
    return model, subject_conditioning


def forward_model(model: torch.nn.Module, batch: dict[str, object], device: torch.device, subject_conditioning: bool):
    eeg = batch["eeg"].to(device)
    if subject_conditioning:
        return model(eeg, batch["subject_index"].to(device))
    return model(eeg)


def build_waveform_bank(dataset: FEISProtocolDataset, retrieval_bank: str) -> tuple[dict[str, np.ndarray], str]:
    if retrieval_bank == "auto":
        retrieval_bank = "oracle_test_templates" if dataset.protocol == "U" else "train_templates"
    split = "train" if retrieval_bank == "train_templates" else "test"
    bank: dict[str, np.ndarray] = {}
    for template_id in dataset.unique_template_ids(split=split):
        meta = dataset.template_metadata(template_id)
        bank[template_id] = dataset._load_audio(meta["audio_relpath"])
    return bank, retrieval_bank


def run_eval(config: dict, args: argparse.Namespace, include_anomalous: bool | None = None) -> dict[str, object]:
    dataset = build_dataset(config, args, split="test", include_anomalous=include_anomalous)
    loader = DataLoader(dataset, batch_size=int(config["train"]["batch_size"]), shuffle=False, num_workers=0)
    bank, bank_policy = build_waveform_bank(dataset, args.retrieval_bank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else resolve_bundle_path(config["output"]["root"], BUNDLE_DIR) / build_run_name(config, args)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model, subject_conditioning = build_model(config, dataset)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    recon_dir = checkpoint_path.parent.parent / "recon_wavs"
    total = 0
    cls_correct = 0
    nta_correct = 0
    waveform_l1 = 0.0
    stft_distance = 0.0
    predictions = []
    with torch.no_grad():
        for batch in loader:
            target = batch["waveform"].to(device)
            label_ids = batch["label_id"].to(device)
            recon, vq_loss, codes, _, logits = forward_model(model, batch, device, subject_conditioning)
            losses = compute_total_loss(
                recon,
                target,
                vq_loss,
                lambda_stft=config["train"]["lambda_stft"],
                lambda_log_stft=config["train"]["lambda_log_stft"],
                lambda_rms=config["train"]["lambda_rms"],
                lambda_envelope=config["train"]["lambda_envelope"],
                lambda_vq=config["train"]["lambda_vq"],
                cls_logits=logits,
                label_ids=label_ids,
                lambda_cls=config["train"]["lambda_cls"],
            )
            batch_size = target.shape[0]
            total += batch_size
            waveform_l1 += float(losses["l1"].item()) * batch_size
            stft_distance += float(losses["stft"].item()) * batch_size
            recon_np = recon.squeeze(1).cpu().numpy()
            target_np = target.cpu().numpy()
            pred_label_ids = torch.argmax(logits, dim=-1).cpu().tolist()
            for idx in range(batch_size):
                label = str(batch["label"][idx])
                subject_id = str(batch["subject_id"][idx])
                trial_index = int(batch["trial_index"][idx].item())
                pred_label = dataset.label_vocab[int(pred_label_ids[idx])]
                cls_correct += int(pred_label == label)
                save_wav_path = recon_dir / f"{subject_id}_{label}_trial{trial_index:04d}_recon.wav"
                save_wav_target = recon_dir / f"{subject_id}_{label}_trial{trial_index:04d}_target.wav"
                save_wav(save_wav_path, recon_np[idx], int(config["audio"]["sample_rate"]))
                save_wav(save_wav_target, target_np[idx], int(config["audio"]["sample_rate"]))
                distances = {
                    template_id: stft_distance_single(torch.from_numpy(recon_np[idx]).float(), torch.from_numpy(wav).float())
                    for template_id, wav in bank.items()
                }
                pred_template = min(distances, key=distances.get)
                nta_correct += int(pred_template == str(batch["template_id"][idx]))
                predictions.append(
                    {
                        "subject_id": subject_id,
                        "label": label,
                        "trial_index": trial_index,
                        "template_id": str(batch["template_id"][idx]),
                        "cls_pred_label": pred_label,
                        "nearest_template_id": pred_template,
                        "code_length": int(codes.shape[-1]),
                    }
                )
    return {
        "protocol": dataset.protocol,
        "checkpoint": str(checkpoint_path),
        "retrieval_bank_policy": bank_policy,
        "include_anomalous": bool(include_anomalous if include_anomalous is not None else args.include_anomalous),
        "num_test_trials": total,
        "waveform_l1": waveform_l1 / max(total, 1),
        "stft_distance": stft_distance / max(total, 1),
        "classification_accuracy": cls_correct / max(total, 1),
        "nearest_template_accuracy": nta_correct / max(total, 1),
        "predictions": predictions,
    }


def main() -> None:
    args = parse_args()
    config = load_simple_yaml(args.config)
    if args.protocol is not None:
        config["data"]["protocol"] = args.protocol
    if args.stage is not None:
        config["data"]["stage"] = args.stage
    if args.ablation_mode is not None:
        config["data"]["ablation_mode"] = args.ablation_mode
    result = run_eval(config, args, include_anomalous=None)
    protocol = str(config["data"]["protocol"]).upper()
    sensitivity = None
    if protocol in {"G", "U"} and not bool(args.include_anomalous or config["data"].get("include_anomalous", False)):
        sensitivity = run_eval(config, args, include_anomalous=True)
        sensitivity.pop("predictions", None)
    run_name = build_run_name(config, args)
    metrics_path = resolve_bundle_path(args.output_root or config["output"]["root"], BUNDLE_DIR) / run_name / "metrics" / "test_metrics.json"
    payload = dict(result)
    if sensitivity is not None:
        payload["sensitivity_including_subject_05"] = sensitivity
    write_json(metrics_path, payload)
    print(f"Saved waveform evaluation to {metrics_path}")
    print(f"NTA: {payload['nearest_template_accuracy']:.4f}")


if __name__ == "__main__":
    main()
