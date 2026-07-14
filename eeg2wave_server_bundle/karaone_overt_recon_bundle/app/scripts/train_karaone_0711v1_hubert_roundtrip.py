from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.io import wavfile
from scipy.signal import stft
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.audio_features import AudioFeatureConfig, load_codec_backend, resample_audio  # noqa: E402
from src.karaone_0711v1.data import (  # noqa: E402
    SplitManifest,
    load_audio,
    make_run_manifest,
    run_name,
    write_json,
)
from src.karaone_0711v1.hubert_roundtrip import (  # noqa: E402
    HubertRoundTripConfig,
    HubertToEncodecDecoder,
    per_example_latent_metrics,
)


PHASES = ("train", "evaluate", "synthesize", "all")
SPLITS = ("subject_train", "subject_val", "subject_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audio-only audit: frozen adapted-HuBERT sequence -> EnCodec latent -> wav."
    )
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone_0711v1.yaml"))
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--stage", choices=("overt_like", "thinking"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override hubert_roundtrip.epochs for training.")
    parser.add_argument("--checkpoint", default=None, help="Decoder checkpoint; defaults to this run's checkpoints/best.pt.")
    parser.add_argument("--resume-training", default=None, help="Resume a round-trip training checkpoint.")
    parser.add_argument("--split", choices=SPLITS, default="subject_val", help="Evaluation/synthesis split.")
    parser.add_argument("--allow-final-test", action="store_true", help="Explicitly authorise one MM21 evaluation or synthesis export.")
    parser.add_argument("--limit", type=int, default=None, help="Limit wav/figure export for a quick diagnostic run.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else BUNDLE_DIR / path


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


def cache_path(cfg: dict[str, Any], stage: str, seed: int) -> Path:
    root = resolve(cfg["paths"]["cache_root"])
    return root / f"karaone_0711v1_{stage}_adapted_audio_targets_s{seed}.npz"


class RoundTripBank:
    """Read-only view of the existing adapted-HuBERT/EnCodec cache."""

    def __init__(self, path: str | Path, manifest: SplitManifest):
        self.path = Path(path)
        raw = np.load(self.path, allow_pickle=False)
        required = {"keys", "subjects", "labels", "audio_paths", "fit_split", "semantic_sequence", "encodec_latent"}
        missing = required - set(raw.files)
        if missing:
            raise ValueError(f"Round-trip audit requires cache fields: {sorted(missing)}")
        self.keys = np.asarray(raw["keys"]).astype(str)
        self.subjects = np.asarray(raw["subjects"]).astype(str)
        self.labels = np.asarray(raw["labels"]).astype(str)
        self.audio_paths = np.asarray(raw["audio_paths"]).astype(str)
        self.fit_split = np.asarray(raw["fit_split"], dtype=bool)
        self.sequence = np.asarray(raw["semantic_sequence"], dtype=np.float32)
        self.latent = np.asarray(raw["encodec_latent"], dtype=np.float32)
        if len({len(self.keys), len(self.subjects), len(self.labels), len(self.audio_paths), len(self.fit_split), len(self.sequence), len(self.latent)}) != 1:
            raise ValueError("Audio cache arrays do not share a common trial count")
        if self.sequence.ndim != 3 or self.latent.ndim != 3:
            raise ValueError(f"Expected sequences [N,T,D], got {self.sequence.shape} and {self.latent.shape}")
        self.splits = np.asarray([manifest.split_for(subject) for subject in self.subjects])
        if not np.array_equal(self.fit_split, self.splits == "subject_train"):
            raise ValueError("fit_split disagrees with the fixed 0711v1 subject split")

    def indices(self, split: str) -> np.ndarray:
        if split not in SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        return np.flatnonzero(self.splits == split).astype(np.int64)


class IndexedCacheDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, bank: RoundTripBank, indices: np.ndarray, latent_mean: np.ndarray, latent_std: np.ndarray):
        self.bank = bank
        self.indices = np.asarray(indices, dtype=np.int64)
        self.latent_mean = np.asarray(latent_mean, dtype=np.float32)
        self.latent_std = np.asarray(latent_std, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        source = torch.from_numpy(np.ascontiguousarray(self.bank.sequence[index]))
        target = (self.bank.latent[index] - self.latent_mean[None, :]) / self.latent_std[None, :]
        return source, torch.from_numpy(np.ascontiguousarray(target))


def inner_train_validation_indices(bank: RoundTripBank, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, label-stratified validation split inside subject_train only."""

    if not 0.0 < float(fraction) < 0.5:
        raise ValueError("hubert_roundtrip.inner_val_fraction must be in (0, 0.5)")
    train_indices = bank.indices("subject_train")
    inner_val: list[int] = []
    for label in sorted(set(bank.labels[train_indices].tolist())):
        rows = train_indices[bank.labels[train_indices] == label]
        rows = np.asarray(sorted(rows.tolist(), key=lambda idx: str(bank.keys[idx])), dtype=np.int64)
        n_val = max(1, int(round(len(rows) * float(fraction))))
        positions = np.linspace(0, len(rows) - 1, num=n_val, dtype=np.int64)
        inner_val.extend(rows[positions].tolist())
    inner_val_array = np.asarray(sorted(set(inner_val)), dtype=np.int64)
    inner_train = np.setdiff1d(train_indices, inner_val_array, assume_unique=False)
    if len(inner_train) == 0 or len(inner_val_array) == 0:
        raise ValueError("Internal train/validation split is empty")
    return inner_train, inner_val_array


def roundtrip_config(settings: dict[str, Any], bank: RoundTripBank) -> HubertRoundTripConfig:
    return HubertRoundTripConfig(
        source_dim=int(bank.sequence.shape[-1]),
        source_steps=int(bank.sequence.shape[1]),
        latent_dim=int(bank.latent.shape[-1]),
        latent_steps=int(bank.latent.shape[1]),
        d_model=int(settings["d_model"]),
        heads=int(settings["heads"]),
        encoder_layers=int(settings["encoder_layers"]),
        refiner_layers=int(settings["refiner_layers"]),
        dropout=float(settings["dropout"]),
    )


def bootstrap_interval(values: np.ndarray, *, seed: int = 11, samples: int = 1000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(int(samples), values.size))].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarise_vectors(vectors: dict[str, np.ndarray]) -> dict[str, dict[str, float | list[float]]]:
    output: dict[str, dict[str, float | list[float]]] = {}
    for name, value in vectors.items():
        values = np.asarray(value, dtype=np.float64)
        output[name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "bootstrap_95_ci": bootstrap_interval(values),
        }
    return output


@torch.no_grad()
def predict_latents(
    model: HubertToEncodecDecoder,
    bank: RoundTripBank,
    indices: np.ndarray,
    latent_mean: np.ndarray,
    latent_std: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(indices), int(batch_size)):
        rows = indices[start : start + int(batch_size)]
        source = torch.from_numpy(np.ascontiguousarray(bank.sequence[rows])).to(device)
        predicted = model(source).cpu().numpy()
        output.append(predicted * latent_std[None, None, :] + latent_mean[None, None, :])
    return np.concatenate(output, axis=0).astype(np.float32)


def latent_vectors(prediction: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    values = per_example_latent_metrics(torch.from_numpy(prediction), torch.from_numpy(target))
    return {name: value.detach().cpu().numpy() for name, value in values.items()}


def label_mean_baseline(bank: RoundTripBank, target_indices: np.ndarray) -> np.ndarray:
    train_indices = bank.indices("subject_train")
    means = {
        label: bank.latent[train_indices[bank.labels[train_indices] == label]].mean(axis=0)
        for label in sorted(set(bank.labels[train_indices].tolist()))
    }
    return np.stack([means[str(bank.labels[index])] for index in target_indices]).astype(np.float32)


def checkpoint_path(args: argparse.Namespace, out_dir: Path) -> Path:
    path = Path(args.checkpoint) if args.checkpoint else out_dir / "checkpoints" / "best.pt"
    if not path.exists():
        raise FileNotFoundError(f"Round-trip decoder checkpoint does not exist: {path}")
    return path


def save_checkpoint(
    path: Path,
    model: HubertToEncodecDecoder,
    optimizer: torch.optim.Optimizer,
    *,
    cfg: HubertRoundTripConfig,
    latent_mean: np.ndarray,
    latent_std: np.ndarray,
    epoch: int,
    history: list[dict[str, float]],
    best_raw_mse: float,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "roundtrip_config": asdict(cfg),
            "latent_mean": np.asarray(latent_mean, dtype=np.float32),
            "latent_std": np.asarray(latent_std, dtype=np.float32),
            "epoch": int(epoch),
            "history": history,
            "best_inner_val_raw_mse": float(best_raw_mse),
            "inner_train_indices": np.asarray(inner_train, dtype=np.int64),
            "inner_val_indices": np.asarray(inner_val, dtype=np.int64),
        },
        path,
    )


def load_decoder(path: Path, bank: RoundTripBank, device: torch.device) -> tuple[HubertToEncodecDecoder, dict[str, Any], np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = HubertRoundTripConfig(**payload["roundtrip_config"])
    expected = (cfg.source_steps, cfg.source_dim, cfg.latent_steps, cfg.latent_dim)
    observed = (bank.sequence.shape[1], bank.sequence.shape[-1], bank.latent.shape[1], bank.latent.shape[-1])
    if expected != observed:
        raise ValueError(f"Checkpoint/cache shape mismatch: checkpoint={expected}, cache={observed}")
    model = HubertToEncodecDecoder(cfg).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload, np.asarray(payload["latent_mean"], dtype=np.float32), np.asarray(payload["latent_std"], dtype=np.float32)


def train_decoder(args: argparse.Namespace, cfg: dict[str, Any], bank: RoundTripBank, out_dir: Path, device: torch.device) -> Path:
    settings = cfg["hubert_roundtrip"]
    inner_train, inner_val = inner_train_validation_indices(bank, float(settings["inner_val_fraction"]))
    latent_mean = bank.latent[inner_train].mean(axis=(0, 1)).astype(np.float32)
    latent_std = np.maximum(bank.latent[inner_train].std(axis=(0, 1)), 1e-6).astype(np.float32)
    architecture = roundtrip_config(settings, bank)
    model = HubertToEncodecDecoder(architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["lr"]), weight_decay=float(settings["weight_decay"]))
    epochs = int(args.epochs or settings["epochs"])
    history: list[dict[str, float]] = []
    start_epoch = 0
    best_raw_mse = float("inf")

    if args.resume_training:
        resume_path = Path(args.resume_training)
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        latent_mean = np.asarray(payload["latent_mean"], dtype=np.float32)
        latent_std = np.asarray(payload["latent_std"], dtype=np.float32)
        inner_train = np.asarray(payload["inner_train_indices"], dtype=np.int64)
        inner_val = np.asarray(payload["inner_val_indices"], dtype=np.int64)
        history = list(payload.get("history", []))
        start_epoch = int(payload.get("epoch", 0))
        best_raw_mse = float(payload.get("best_inner_val_raw_mse", float("inf")))
        print(f"[hubert_roundtrip] resumed after epoch {start_epoch}: {resume_path}", flush=True)

    train_data = DataLoader(
        IndexedCacheDataset(bank, inner_train, latent_mean, latent_std),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    best_path = out_dir / "checkpoints" / "best.pt"
    last_path = out_dir / "checkpoints" / "last.pt"
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        losses: list[float] = []
        iterator = tqdm(train_data, desc=f"[hubert_roundtrip] train {epoch}/{epochs}", unit="batch", dynamic_ncols=True)
        for source, target in iterator:
            source = source.to(device)
            target = target.to(device)
            prediction = model(source)
            loss = F.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), float(cfg["run"]["grad_clip"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            iterator.set_postfix(normalized_latent_mse=f"{losses[-1]:.4f}")
        validation_prediction = predict_latents(model, bank, inner_val, latent_mean, latent_std, int(settings["batch_size"]), device)
        validation_target = bank.latent[inner_val]
        raw_mse = float(np.mean((validation_prediction - validation_target) ** 2))
        cosine = float(latent_vectors(validation_prediction, validation_target)["latent_cosine"].mean())
        row = {
            "epoch": float(epoch),
            "train_normalized_latent_mse": float(np.mean(losses)),
            "inner_val_raw_latent_mse": raw_mse,
            "inner_val_latent_cosine": cosine,
        }
        history.append(row)
        write_json(out_dir / "metrics" / "latest_metrics.json", row)
        write_json(out_dir / "metrics" / "history.json", {"history": history, "selection_split": "subject_train_internal", "test_accessed": False})
        save_checkpoint(
            last_path,
            model,
            optimizer,
            cfg=architecture,
            latent_mean=latent_mean,
            latent_std=latent_std,
            epoch=epoch,
            history=history,
            best_raw_mse=best_raw_mse,
            inner_train=inner_train,
            inner_val=inner_val,
        )
        if raw_mse < best_raw_mse:
            best_raw_mse = raw_mse
            save_checkpoint(
                best_path,
                model,
                optimizer,
                cfg=architecture,
                latent_mean=latent_mean,
                latent_std=latent_std,
                epoch=epoch,
                history=history,
                best_raw_mse=best_raw_mse,
                inner_train=inner_train,
                inner_val=inner_val,
            )
        print("[hubert_roundtrip] epoch=" + json.dumps(row), flush=True)

    write_json(
        out_dir / "metrics" / "training_split.json",
        {
            "fit_split": "subject_train",
            "inner_train_n": int(len(inner_train)),
            "inner_val_n": int(len(inner_val)),
            "inner_val_fraction": float(settings["inner_val_fraction"]),
            "checkpoint_selection": "lowest_inner_val_raw_latent_mse",
            "test_accessed": False,
        },
    )
    return best_path


def require_split_permission(args: argparse.Namespace) -> None:
    if args.split == "subject_test" and not args.allow_final_test:
        raise PermissionError("MM21 round-trip evaluation/export requires explicit --allow-final-test")


def evaluate_decoder(args: argparse.Namespace, bank: RoundTripBank, out_dir: Path, device: torch.device) -> Path:
    require_split_permission(args)
    path = checkpoint_path(args, out_dir)
    model, payload, latent_mean, latent_std = load_decoder(path, bank, device)
    indices = bank.indices(args.split)
    prediction = predict_latents(model, bank, indices, latent_mean, latent_std, batch_size=32, device=device)
    target = bank.latent[indices]
    train_indices = bank.indices("subject_train")
    mean_baseline = np.broadcast_to(bank.latent[train_indices].mean(axis=0, keepdims=True), target.shape).copy()
    label_baseline = label_mean_baseline(bank, indices)
    report = {
        "phase": "hubert_roundtrip_evaluate",
        "checkpoint": str(path),
        "checkpoint_epoch": int(payload["epoch"]),
        "split": args.split,
        "n_trials": int(len(indices)),
        "input": "frozen_adapted_hubert_semantic_sequence_[50,768]",
        "target": "cached_encodec_latent_[150,128]",
        "roundtrip": summarise_vectors(latent_vectors(prediction, target)),
        "mean_latent_baseline": summarise_vectors(latent_vectors(mean_baseline, target)),
        "privileged_label_latent_baseline": {
            "warning": "Uses the true label only as an upper-bound diagnostic; it is not a deployable input.",
            "metrics": summarise_vectors(latent_vectors(label_baseline, target)),
        },
        "test_accessed": args.split == "subject_test",
    }
    destination = out_dir / "metrics" / f"{args.split}_latent_metrics.json"
    write_json(destination, report)
    print(str(destination))
    return destination


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    wavfile.write(path, int(sample_rate), (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16))


def match_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    if len(values) >= target_length:
        return values[:target_length]
    return np.pad(values, (0, target_length - len(values))).astype(np.float32)


def waveform_vectors(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    n = min(len(reference), len(candidate))
    reference, candidate = reference[:n], candidate[:n]
    ref_centered = reference - reference.mean()
    cand_centered = candidate - candidate.mean()
    corr = float((ref_centered @ cand_centered) / (np.linalg.norm(ref_centered) * np.linalg.norm(cand_centered) + 1e-12))
    scale = float((candidate @ reference) / (reference @ reference + 1e-12))
    target = scale * reference
    noise = candidate - target
    si_sdr = float(10.0 * np.log10((target @ target + 1e-12) / (noise @ noise + 1e-12)))
    _, _, ref_spec = stft(reference, fs=sample_rate, nperseg=512, noverlap=384, boundary=None)
    _, _, cand_spec = stft(candidate, fs=sample_rate, nperseg=512, noverlap=384, boundary=None)
    frames = min(ref_spec.shape[1], cand_spec.shape[1])
    ref_db = 20.0 * np.log10(np.maximum(np.abs(ref_spec[:, :frames]), 1e-6))
    cand_db = 20.0 * np.log10(np.maximum(np.abs(cand_spec[:, :frames]), 1e-6))
    return {"waveform_correlation": corr, "si_sdr_db": si_sdr, "log_spectrogram_mae_db": float(np.mean(np.abs(ref_db - cand_db)))}


def write_comparison(path: Path, reference: np.ndarray, oracle: np.ndarray, reconstruction: np.ndarray, key: str, sample_rate: int) -> None:
    duration = len(reference) / float(sample_rate)
    time = np.arange(len(reference)) / float(sample_rate)
    _, _, ref_spec = stft(reference, fs=sample_rate, nperseg=512, noverlap=384, boundary=None)
    _, _, oracle_spec = stft(oracle, fs=sample_rate, nperseg=512, noverlap=384, boundary=None)
    _, _, recon_spec = stft(reconstruction, fs=sample_rate, nperseg=512, noverlap=384, boundary=None)
    specs = [20.0 * np.log10(np.maximum(np.abs(value), 1e-6)) for value in (ref_spec, oracle_spec, recon_spec)]
    low, high = float(min(value.min() for value in specs)), float(max(value.max() for value in specs))
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), constrained_layout=True)
    axes[0].plot(time, reference, color="#2563eb", linewidth=0.65, label="reference")
    axes[0].plot(time, oracle, color="#16a34a", linewidth=0.65, alpha=0.8, label="cached-latent oracle")
    axes[0].plot(time, reconstruction, color="#dc2626", linewidth=0.65, alpha=0.8, label="HuBERT round-trip")
    axes[0].set(title=f"{key}: waveform comparison", xlim=(0, duration), ylabel="amplitude")
    axes[0].legend(loc="upper right")
    labels = ("reference log-spectrogram", "cached-latent oracle log-spectrogram", "HuBERT round-trip log-spectrogram")
    colors = ("Blues", "Greens", "Reds")
    for axis, spec, label, cmap in zip(axes[1:], specs, labels, colors):
        axis.imshow(spec, origin="lower", aspect="auto", cmap=cmap, vmin=low, vmax=high, extent=(0, duration, 0, sample_rate / 2))
        axis.set(title=label, ylabel="Hz")
    axes[-1].set(xlabel="seconds")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def synthesise_decoder(args: argparse.Namespace, cfg: dict[str, Any], bank: RoundTripBank, out_dir: Path, device: torch.device) -> Path:
    require_split_permission(args)
    path = checkpoint_path(args, out_dir)
    model, payload, latent_mean, latent_std = load_decoder(path, bank, device)
    indices = bank.indices(args.split)
    if args.limit is not None:
        indices = indices[: int(args.limit)]
    prediction = predict_latents(model, bank, indices, latent_mean, latent_std, batch_size=16, device=device)
    root = resolve(cfg["data"]["root"])
    codec_path = resolve(cfg["paths"]["encodec_model"])
    codec = load_codec_backend(
        AudioFeatureConfig(
            sample_rate=16000,
            duration_sec=2.0,
            target_kind="encodec_latent",
            backend="encodec_latent",
            codec_model_name_or_path=str(codec_path),
            local_files_only=True,
            codec_bandwidth=6.0,
        )
    )
    destination = out_dir / "wavs" / f"hubert_roundtrip_{args.split}"
    reference_dir = destination / "reference"
    oracle_dir = destination / "cache_latent_oracle"
    reconstruction_dir = destination / "reconstructed"
    figures_dir = destination / "comparison"
    for folder in (reference_dir, oracle_dir, reconstruction_dir, figures_dir):
        folder.mkdir(parents=True, exist_ok=True)
    oracle_metrics: dict[str, list[float]] = {"waveform_correlation": [], "si_sdr_db": [], "log_spectrogram_mae_db": []}
    reconstruction_metrics: dict[str, list[float]] = {"waveform_correlation": [], "si_sdr_db": [], "log_spectrogram_mae_db": []}
    files: list[dict[str, Any]] = []
    for row, index in enumerate(tqdm(indices, desc=f"[hubert_roundtrip] synthesize {args.split}", unit="trial", dynamic_ncols=True)):
        key = str(bank.keys[index])
        safe_key = key.replace(":", "_")
        reference_16k = load_audio(root / str(bank.audio_paths[index]))
        reference = match_length(resample_audio(reference_16k, src_sr=16000, dst_sr=codec.sample_rate), len(codec.decode(bank.latent[index])))
        oracle = match_length(codec.decode(bank.latent[index]), len(reference))
        reconstruction = match_length(codec.decode(prediction[row]), len(reference))
        for metric_name, value in waveform_vectors(reference, oracle, codec.sample_rate).items():
            oracle_metrics[metric_name].append(value)
        for metric_name, value in waveform_vectors(reference, reconstruction, codec.sample_rate).items():
            reconstruction_metrics[metric_name].append(value)
        reference_path = reference_dir / f"{safe_key}.wav"
        oracle_path = oracle_dir / f"{safe_key}.wav"
        reconstruction_path = reconstruction_dir / f"{safe_key}.wav"
        figure_path = figures_dir / f"{safe_key}.png"
        write_wav(reference_path, reference, codec.sample_rate)
        write_wav(oracle_path, oracle, codec.sample_rate)
        write_wav(reconstruction_path, reconstruction, codec.sample_rate)
        write_comparison(figure_path, reference, oracle, reconstruction, key, codec.sample_rate)
        files.append(
            {
                "key": key,
                "reference_wav": str(reference_path.relative_to(destination)),
                "cache_latent_oracle_wav": str(oracle_path.relative_to(destination)),
                "reconstructed_wav": str(reconstruction_path.relative_to(destination)),
                "comparison_png": str(figure_path.relative_to(destination)),
            }
        )
    audio_report = {
        "phase": "hubert_roundtrip_synthesize",
        "checkpoint": str(path),
        "checkpoint_epoch": int(payload["epoch"]),
        "split": args.split,
        "n_generated": int(len(indices)),
        "waveform_sample_rate": int(codec.sample_rate),
        "reference_audio": "KaraOne source wav standardized to 16 kHz/2 s/RMS, then resampled to the EnCodec decoder sample rate for a fair comparison.",
        "oracle_definition": "cached_encodec_latent decoded without per-example EnCodec decoder scales; this is a cache-latent ceiling, not a lossless waveform oracle.",
        "roundtrip_input": "frozen adapted-HuBERT semantic_sequence [50,768] from the existing 0711v1 cache",
        "cache_latent_oracle_vs_reference": summarise_vectors({key: np.asarray(value) for key, value in oracle_metrics.items()}),
        "hubert_roundtrip_vs_reference": summarise_vectors({key: np.asarray(value) for key, value in reconstruction_metrics.items()}),
        "test_accessed": args.split == "subject_test",
    }
    write_json(out_dir / "metrics" / f"{args.split}_audio_metrics.json", audio_report)
    write_json(
        destination / "synthesis_manifest.json",
        {
            "version": "0711v1",
            "phase": "hubert_roundtrip",
            "checkpoint": str(path),
            "split": args.split,
            "n_generated": int(len(indices)),
            "inference_input": "frozen_adapted_hubert_semantic_sequence_only",
            "reference_audio_used_for_reconstruction": False,
            "cache_latent_used_for_reconstruction": False,
            "cache_latent_used_only_for_oracle_export_and_metrics": True,
            "test_accessed": args.split == "subject_test",
            "files": files,
        },
    )
    print(str(destination))
    return destination


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    stage = args.stage or str(cfg["data"]["stage"])
    seed = int(args.seed if args.seed is not None else cfg["run"]["seed"])
    set_seed(seed)
    device = torch.device(args.device) if args.device else default_device()
    root = resolve(cfg["data"]["root"])
    manifest = SplitManifest.build(root)
    cache = cache_path(cfg, stage, seed)
    if not cache.exists():
        raise FileNotFoundError(f"Missing adapted audio cache: {cache}")
    bank = RoundTripBank(cache, manifest)
    out_dir = resolve(cfg["paths"]["output_root"]) / run_name(stage, "hubert_roundtrip", seed)
    for folder in (out_dir, out_dir / "checkpoints", out_dir / "metrics"):
        folder.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "run_manifest.json",
        {
            **make_run_manifest(
                repo_root=BUNDLE_DIR.parent.parent,
                config_path=args.config,
                split_manifest=manifest,
                phase="hubert_roundtrip",
                stage=stage,
                seed=seed,
                input_paths=[cache],
            ),
            "experiment": "audio_only_frozen_adapted_hubert_roundtrip",
            "input": "semantic_sequence [50,768]",
            "target": "encodec_latent [150,128]",
            "eeg_used": False,
            "test_accessed": False,
        },
    )
    if args.phase in {"train", "all"}:
        train_decoder(args, cfg, bank, out_dir, device)
    if args.phase in {"evaluate", "all"}:
        evaluate_decoder(args, bank, out_dir, device)
    if args.phase in {"synthesize", "all"}:
        synthesise_decoder(args, cfg, bank, out_dir, device)


if __name__ == "__main__":
    main()
