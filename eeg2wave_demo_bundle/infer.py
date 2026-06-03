from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import FEISThinkingDataset
from losses import compute_total_loss, stft_distance_single
from model import EEG2WaveVQModel
from utils import ensure_dir, load_simple_yaml, resolve_bundle_path, save_wav, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer minimal FEIS EEG2Wave demo.")
    parser.add_argument("--subject", required=True, help="FEIS subject id, e.g. 01")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def build_model(config: dict) -> EEG2WaveVQModel:
    cfg_model = config["model"]
    cfg_audio = config["audio"]
    return EEG2WaveVQModel(
        n_channels_eeg=cfg_model["n_channels_eeg"],
        hidden_dim=cfg_model["hidden_dim"],
        codebook_size=cfg_model["codebook_size"],
        vq_beta=cfg_model["vq_beta"],
        vq_decay=cfg_model["vq_decay"],
        output_samples=cfg_audio["n_samples"],
    )


def main() -> None:
    args = parse_args()
    bundle_dir = Path(__file__).resolve().parent
    config = load_simple_yaml(args.config)
    output_root = resolve_bundle_path(args.output_root or config["output"]["root"], bundle_dir)
    checkpoint_path = Path(args.checkpoint or output_root / "checkpoints" / f"subject_{args.subject}_best.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = FEISThinkingDataset(
        subject_id=args.subject,
        data_root=str(resolve_bundle_path(args.data_root or config["data"]["root"], bundle_dir)),
        stage=config["data"]["stage"],
        split="test",
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        audio_sr=config["audio"]["sample_rate"],
        audio_dur=config["audio"]["duration_sec"],
    )
    loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=False, num_workers=0)
    recon_dir = ensure_dir(output_root / "recon_wavs" / f"subject_{args.subject}")
    metrics_dir = ensure_dir(output_root / "metrics")

    canonical = {
        label: torch.from_numpy(wav).float()
        for label, wav in dataset.canonical_wavs_by_label().items()
    }

    total = 0
    correct = 0
    l1_sum = 0.0
    stft_sum = 0.0
    predictions: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device)
            target = batch["waveform"].to(device)
            labels = list(batch["label"])
            trial_indices = batch["trial_index"].tolist()
            recon, vq_loss, codes, _ = model(eeg)
            losses = compute_total_loss(
                recon,
                target,
                vq_loss,
                lambda_stft=config["train"]["lambda_stft"],
                lambda_vq=config["train"]["lambda_vq"],
            )
            batch_size = eeg.shape[0]
            l1_sum += float(losses["l1"].item()) * batch_size
            stft_sum += float(losses["stft"].item()) * batch_size

            recon_np = recon.squeeze(1).cpu().numpy()
            target_np = target.cpu().numpy()
            for idx, label in enumerate(labels):
                trial_index = int(trial_indices[idx])
                recon_wav = recon_np[idx]
                target_wav = target_np[idx]
                save_wav(recon_dir / f"{label}_trial{trial_index:04d}_recon.wav", recon_wav, config["audio"]["sample_rate"])
                save_wav(recon_dir / f"{label}_trial{trial_index:04d}_target.wav", target_wav, config["audio"]["sample_rate"])

                distances = {
                    template_label: stft_distance_single(
                        torch.from_numpy(recon_wav).float(),
                        template_wav,
                    )
                    for template_label, template_wav in canonical.items()
                }
                pred_label = min(distances, key=distances.get)
                correct += int(pred_label == label)
                total += 1
                predictions.append(
                    {
                        "trial_index": trial_index,
                        "label": label,
                        "pred_label": pred_label,
                        "code_length": int(codes.shape[-1]),
                    }
                )

    metrics = {
        "subject_id": args.subject,
        "checkpoint": str(checkpoint_path),
        "num_test_trials": total,
        "waveform_l1": l1_sum / max(total, 1),
        "stft_distance": stft_sum / max(total, 1),
        "nearest_template_accuracy": correct / max(total, 1),
        "random_baseline": 1.0 / max(len(canonical), 1),
        "predictions": predictions,
    }
    write_json(metrics_dir / f"subject_{args.subject}_metrics.json", metrics)
    print(f"Saved reconstructions to {recon_dir}")


if __name__ == "__main__":
    main()
