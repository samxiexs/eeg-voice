from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.karaone_0711v1.audio import (  # noqa: E402
    HubertAdaptationConfig,
    audio_augment,
    build_adapted_audio_cache,
    load_hubert_adapter,
    save_audio_adapter,
)
from src.karaone_0711v1.data import (  # noqa: E402
    FitAudit,
    KaraOne0711Dataset,
    SplitManifest,
    make_run_manifest,
    records_for_split,
    run_name,
    write_json,
)
from src.karaone_0711v1.eval import (  # noqa: E402
    evaluate_global_gate,
    final_test_report,
    require_flow_gate,
    validate_gate_context,
    write_gate,
)
from src.karaone_0711v1.losses import (  # noqa: E402
    flow_matching_loss,
    group_dro,
    masked_token_cross_entropy,
    multi_positive_clip_loss,
    symmetric_view_contrastive,
    variance_covariance_regularizer,
)
from src.karaone_0711v1.model import ConditionalFlowDecoder, EEG0711Config, EEG0711Encoder  # noqa: E402


PHASES = ("audit", "audio_ssl", "audio_cache", "eeg_ssl", "align_global", "align_token", "flow", "evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KaraOne 0711v1 strict cross-subject trainer.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone_0711v1.yaml"))
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--stage", choices=("overt_like", "thinking"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None, help="Required for align/flow/evaluate phases as applicable.")
    parser.add_argument("--gate", default=None, help="Validation gate JSON; required for flow.")
    parser.add_argument("--allow-gate-bypass", action="store_true", help="Exploratory-only: permit token/flow after a failed P02 gate; MM21 remains prohibited.")
    parser.add_argument("--allow-final-test", action="store_true", help="Explicitly authorise the one-time MM21 evaluation phase.")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def phase_dir(cfg: dict[str, Any], stage: str, phase: str, seed: int) -> Path:
    path = resolve(cfg["paths"]["output_root"]) / run_name(stage, phase, seed)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_paths(cfg: dict[str, Any], stage: str, seed: int) -> dict[str, Path]:
    root = resolve(cfg["paths"]["cache_root"])
    root.mkdir(parents=True, exist_ok=True)
    base = f"karaone_0711v1_{stage}"
    return {
        "split": root / "karaone_0711v1_split_manifest.json",
        "audio_adapter": root / f"{base}_audio_ssl_s{seed}.pt",
        "audio_targets": root / f"{base}_adapted_audio_targets_s{seed}.npz",
        "audio_audit": root / f"{base}_adapted_audio_targets_s{seed}.audit.json",
    }


class AudioTargetBank:
    def __init__(self, path: str | Path):
        raw = np.load(path, allow_pickle=False)
        required = {"keys", "clip_embedding", "semantic_token_ids", "labels", "subjects", "fit_split"}
        missing = required - set(raw.files)
        if missing:
            raise ValueError(f"0711v1 target cache missing keys: {sorted(missing)}")
        self.keys = [str(item) for item in raw["keys"].tolist()]
        self.index = {key: idx for idx, key in enumerate(self.keys)}
        self.clip_embedding = raw["clip_embedding"].astype(np.float32)
        self.token_ids = raw["semantic_token_ids"].astype(np.int64)
        self.labels = raw["labels"].astype(str)
        self.subjects = raw["subjects"].astype(str)
        self.fit_split = raw["fit_split"].astype(bool)
        self.encodec_latent = raw["encodec_latent"].astype(np.float32) if "encodec_latent" in raw.files else None

    def batch(self, keys: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        indices = [self.index[str(key)] for key in keys]
        return {
            "audio_embed": torch.from_numpy(self.clip_embedding[indices]).to(device),
            "token_ids": torch.from_numpy(self.token_ids[indices]).to(device),
            "encodec_latent": torch.from_numpy(self.encodec_latent[indices]).to(device) if self.encodec_latent is not None else None,
        }

    def train_bank(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.clip_embedding[self.fit_split], self.labels[self.fit_split], self.token_ids[self.fit_split]


def make_dataset(cfg: dict[str, Any], manifest: SplitManifest, split: str, stage: str, *, include_audio: bool = False) -> KaraOne0711Dataset:
    return KaraOne0711Dataset(
        resolve(cfg["data"]["root"]),
        split,
        stage,
        manifest=manifest,
        eeg_len=int(cfg["data"]["eeg_len"]),
        sample_rate=int(cfg["data"]["eeg_sample_rate"]),
        include_audio=include_audio,
    )


def loader(dataset: KaraOne0711Dataset, batch_size: int, cfg: dict[str, Any], *, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=int(cfg["run"].get("num_workers", 0)))


def make_eeg_model(cfg: dict[str, Any]) -> EEG0711Encoder:
    model_cfg = cfg["model"]
    return EEG0711Encoder(
        EEG0711Config(
            channels=int(model_cfg["channels"]),
            d_model=int(model_cfg["d_model"]),
            layers=int(model_cfg["layers"]),
            heads=int(model_cfg["heads"]),
            dropout=float(model_cfg["dropout"]),
            embed_dim=int(model_cfg["embed_dim"]),
            semantic_steps=int(model_cfg["semantic_steps"]),
            semantic_vocab=int(model_cfg["semantic_vocab"]),
        )
    )


def save_model(path: Path, model: torch.nn.Module, *, cfg: dict[str, Any], phase: str, epoch: int, metrics: dict[str, Any]) -> None:
    torch.save({"state_dict": model.state_dict(), "phase": phase, "epoch": epoch, "metrics": metrics, "config": cfg}, path)


def load_model(path: str | Path, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    return payload


def augment_eeg(batch: dict[str, Any], cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    eeg = batch["eeg"].clone()
    topo = batch["topography"].clone()
    options = cfg["eeg_ssl"]
    channel_drop = float(options["channel_dropout"])
    time_mask_ratio = float(options["time_mask_ratio"])
    noise = float(options["noise_std"])
    channel_keep = (torch.rand(eeg.shape[:2], device=eeg.device) > channel_drop).to(eeg.dtype).unsqueeze(-1)
    eeg = eeg * channel_keep
    width = max(1, int(eeg.shape[-1] * time_mask_ratio))
    for row, valid in enumerate(batch["eeg_valid_len"].tolist()):
        start = int(torch.randint(0, max(1, int(valid) - width + 1), (1,), device=eeg.device).item())
        eeg[row, :, start : start + width] = 0.0
    eeg = eeg + torch.randn_like(eeg) * noise
    topo = topo + torch.randn_like(topo) * (noise * 0.5)
    return eeg, topo


def evaluate_eeg_ssl(model: EEG0711Encoder, data: DataLoader, cfg: dict[str, Any], device: torch.device) -> float:
    model.eval()
    values = []
    with torch.no_grad():
        for batch in data:
            batch = move_batch(batch, device)
            out = model(batch["eeg"], batch["eeg_valid_len"], batch["topography"])
            values.append(float(symmetric_view_contrastive(out["raw_embed"], out["topo_embed"], float(cfg["eeg_ssl"]["temperature"]))["total"].cpu()))
    return float(np.mean(values)) if values else float("inf")


def collect_alignment(model: EEG0711Encoder, data: DataLoader, bank: AudioTargetBank, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    payload: dict[str, list[np.ndarray]] = {key: [] for key in ("embed", "zero", "audio", "token_logits", "token_ids", "labels")}
    with torch.no_grad():
        for batch in data:
            batch = move_batch(batch, device)
            out = model(batch["eeg"], batch["eeg_valid_len"], batch["topography"])
            zero = model(torch.zeros_like(batch["eeg"]), batch["eeg_valid_len"], torch.zeros_like(batch["topography"]))
            targets = bank.batch(batch["key"], device)
            payload["embed"].append(out["eeg_embed"].cpu().numpy())
            payload["zero"].append(zero["eeg_embed"].cpu().numpy())
            payload["audio"].append(targets["audio_embed"].cpu().numpy())
            payload["token_logits"].append(out["token_logits"].cpu().numpy())
            payload["token_ids"].append(targets["token_ids"].cpu().numpy())
            payload["labels"].append(batch["label_idx"].cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in payload.items()}


def train_audio_ssl(args: argparse.Namespace, cfg: dict[str, Any], manifest: SplitManifest, stage: str, seed: int, device: torch.device, out_dir: Path) -> None:
    settings = cfg["audio_ssl"]
    root = resolve(cfg["data"]["root"])
    adapter_cfg = HubertAdaptationConfig(
        model_path=str(resolve(cfg["paths"]["hubert_model"])),
        top_unfrozen_layers=int(settings["top_unfrozen_layers"]),
        projection_dim=int(settings["projection_dim"]),
        semantic_steps=int(settings["semantic_steps"]),
        semantic_vocab=int(settings["semantic_vocab"]),
    )
    model, extractor = load_hubert_adapter(adapter_cfg, device=device)
    train_data = loader(make_dataset(cfg, manifest, "subject_train", stage, include_audio=True), int(settings["batch_size"]), cfg, shuffle=True)
    val_data = loader(make_dataset(cfg, manifest, "subject_val", stage, include_audio=True), int(settings["batch_size"]), cfg, shuffle=False)
    optimizer = torch.optim.AdamW((item for item in model.parameters() if item.requires_grad), lr=float(settings["lr"]), weight_decay=float(settings["weight_decay"]))
    labels = {label: idx for idx, label in enumerate(("/diy/", "/iy/", "/m/", "/n/", "/piy/", "/tiy/", "/uw/", "gnaw", "knew", "pat", "pot"))}
    epochs = 1 if args.smoke else int(args.epochs or settings["epochs"])
    best, history = -1.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        iterator = tqdm(train_data, desc=f"[0711v1] audio_ssl {epoch}/{epochs}", unit="batch", dynamic_ncols=True)
        for batch in iterator:
            batch = move_batch(batch, device)
            a = audio_augment(batch["audio"])
            b = audio_augment(batch["audio"])
            first = extractor(a.detach().cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
            second = extractor(b.detach().cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
            one = model(first["input_values"].to(device), first.get("attention_mask").to(device) if first.get("attention_mask") is not None else None)
            two = model(second["input_values"].to(device), second.get("attention_mask").to(device) if second.get("attention_mask") is not None else None)
            label = torch.as_tensor([labels[item] for item in batch["label"]], device=device)
            contrast = multi_positive_clip_loss(one["embedding"], two["embedding"], label, batch["subject"], cross_subject_weight=float(settings["lambda_cross_subject"]))
            loss = float(settings["lambda_view"]) * contrast["total"] + float(settings["lambda_label"]) * 0.5 * (F.cross_entropy(one["label_logits"], label) + F.cross_entropy(two["label_logits"], label))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), float(cfg["run"]["grad_clip"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            iterator.set_postfix(loss=f"{losses[-1]:.4f}")
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in tqdm(val_data, desc=f"[0711v1] audio_ssl val {epoch}/{epochs}", unit="batch", leave=False, dynamic_ncols=True):
                batch = move_batch(batch, device)
                values = extractor(batch["audio"].detach().cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
                out = model(values["input_values"].to(device), values.get("attention_mask").to(device) if values.get("attention_mask") is not None else None)
                actual = torch.as_tensor([labels[item] for item in batch["label"]], device=device)
                correct += int((out["label_logits"].argmax(dim=-1) == actual).sum().item())
                total += int(actual.numel())
        metric = correct / max(total, 1)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "subject_val_audio_label_acc": metric}
        history.append(row)
        write_json(out_dir / "metrics" / "latest_metrics.json", row)
        print("[0711v1] audio_ssl epoch=" + json.dumps(row, ensure_ascii=False), flush=True)
        if metric > best:
            best = metric
            audit = FitAudit.from_records("hubert_domain_adapter", manifest, records_for_split(root, manifest, "subject_train"))
            save_audio_adapter(out_dir / "checkpoints" / "best.pt", model, adapter_cfg, audit)
    write_json(out_dir / "metrics" / "history.json", {"history": history, "selection_split": "subject_val", "test_accessed": False})
    best_path = out_dir / "checkpoints" / "best.pt"
    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    paths = cache_paths(cfg, stage, seed)
    codec = resolve(cfg["paths"]["encodec_model"])
    build_adapted_audio_cache(root=root, manifest=manifest, adapter=model, feature_extractor=extractor, output_path=paths["audio_targets"], audit_path=paths["audio_audit"], device=device, stage=stage, semantic_steps=adapter_cfg.semantic_steps, semantic_vocab=adapter_cfg.semantic_vocab, codec_model_path=codec if codec.exists() else None)


def rebuild_audio_cache(args: argparse.Namespace, cfg: dict[str, Any], manifest: SplitManifest, stage: str, seed: int, device: torch.device, out_dir: Path) -> None:
    """Recover from cache extraction failures without retraining adapted HuBERT."""
    if not args.resume:
        raise ValueError("audio_cache requires --resume <audio_ssl/checkpoints/best.pt>")
    settings = cfg["audio_ssl"]
    adapter_cfg = HubertAdaptationConfig(
        model_path=str(resolve(cfg["paths"]["hubert_model"])),
        top_unfrozen_layers=int(settings["top_unfrozen_layers"]),
        projection_dim=int(settings["projection_dim"]),
        semantic_steps=int(settings["semantic_steps"]),
        semantic_vocab=int(settings["semantic_vocab"]),
    )
    model, extractor = load_hubert_adapter(adapter_cfg, device=device)
    payload = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    paths = cache_paths(cfg, stage, seed)
    root = resolve(cfg["data"]["root"])
    codec = resolve(cfg["paths"]["encodec_model"])
    print(f"[0711v1] rebuilding audio target cache from {args.resume}; HuBERT will not be retrained.", flush=True)
    audit = build_adapted_audio_cache(
        root=root,
        manifest=manifest,
        adapter=model,
        feature_extractor=extractor,
        output_path=paths["audio_targets"],
        audit_path=paths["audio_audit"],
        device=device,
        stage=stage,
        semantic_steps=adapter_cfg.semantic_steps,
        semantic_vocab=adapter_cfg.semantic_vocab,
        codec_model_path=codec if codec.exists() else None,
    )
    write_json(out_dir / "metrics" / "audio_cache.json", {"source_checkpoint": str(args.resume), "retrained_hubert": False, **audit})


def train_eeg_ssl(args: argparse.Namespace, cfg: dict[str, Any], manifest: SplitManifest, stage: str, device: torch.device, out_dir: Path) -> None:
    settings = cfg["eeg_ssl"]
    model = make_eeg_model(cfg).to(device)
    train_data = loader(make_dataset(cfg, manifest, "subject_train", stage), int(settings["batch_size"]), cfg, shuffle=True)
    val_data = loader(make_dataset(cfg, manifest, "subject_val", stage), int(settings["batch_size"]), cfg, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["lr"]), weight_decay=float(settings["weight_decay"]))
    epochs = 1 if args.smoke else int(args.epochs or settings["epochs"])
    best, history = float("inf"), []
    for epoch in range(1, epochs + 1):
        model.train()
        values = []
        iterator = tqdm(train_data, desc=f"[0711v1] eeg_ssl {epoch}/{epochs}", unit="batch", dynamic_ncols=True)
        for batch in iterator:
            batch = move_batch(batch, device)
            eeg, topo = augment_eeg(batch, cfg)
            out = model(eeg, batch["eeg_valid_len"], topo)
            loss = symmetric_view_contrastive(out["raw_embed"], out["topo_embed"], float(settings["temperature"]))["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), float(cfg["run"]["grad_clip"]))
            optimizer.step()
            values.append(float(loss.detach().cpu()))
            iterator.set_postfix(view_loss=f"{values[-1]:.4f}")
        val_loss = evaluate_eeg_ssl(model, val_data, cfg, device)
        row = {"epoch": epoch, "train_view_loss": float(np.mean(values)), "subject_val_view_loss": val_loss}
        history.append(row)
        write_json(out_dir / "metrics" / "latest_metrics.json", row)
        print("[0711v1] eeg_ssl epoch=" + json.dumps(row, ensure_ascii=False), flush=True)
        if val_loss < best:
            best = val_loss
            save_model(out_dir / "checkpoints" / "best.pt", model, cfg=cfg, phase="eeg_ssl", epoch=epoch, metrics=row)
    write_json(out_dir / "metrics" / "history.json", {"history": history, "selection_split": "subject_val", "test_accessed": False})


def train_alignment(args: argparse.Namespace, cfg: dict[str, Any], manifest: SplitManifest, stage: str, seed: int, device: torch.device, out_dir: Path, phase: str) -> None:
    if not args.resume:
        raise ValueError(f"{phase} requires --resume <eeg_ssl_or_align_checkpoint>")
    bank = AudioTargetBank(cache_paths(cfg, stage, seed)["audio_targets"])
    model = make_eeg_model(cfg).to(device)
    load_model(args.resume, model, device)
    settings = cfg["alignment"]
    train_data = loader(make_dataset(cfg, manifest, "subject_train", stage), int(settings["batch_size"]), cfg, shuffle=True)
    val_data = loader(make_dataset(cfg, manifest, "subject_val", stage), int(settings["batch_size"]), cfg, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["lr"]), weight_decay=float(settings["weight_decay"]))
    epochs = 1 if args.smoke else int(args.epochs or settings["epochs"])
    best_score, history, best_gate = -float("inf"), [], None
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        iterator = tqdm(train_data, desc=f"[0711v1] {phase} {epoch}/{epochs}", unit="batch", dynamic_ncols=True)
        for batch in iterator:
            batch = move_batch(batch, device)
            out = model(batch["eeg"], batch["eeg_valid_len"], batch["topography"])
            targets = bank.batch(batch["key"], device)
            clip = multi_positive_clip_loss(out["eeg_embed"], targets["audio_embed"], batch["label_idx"], batch["subject"], temperature=float(settings["temperature"]), cross_subject_weight=float(settings["cross_subject_positive_weight"]))
            loss = group_dro(clip["per_example"], batch["subject"], float(settings["group_dro_eta"]))
            reg = variance_covariance_regularizer(out["eeg_embed"])["total"]
            loss = loss + float(settings["lambda_vicreg"]) * reg
            if phase == "align_token":
                loss = loss + float(settings["lambda_token"]) * masked_token_cross_entropy(out["token_logits"], targets["token_ids"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), float(cfg["run"]["grad_clip"]))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            iterator.set_postfix(loss=f"{train_losses[-1]:.4f}")
        collected = collect_alignment(model, val_data, bank, device)
        train_embed, train_labels, train_tokens = bank.train_bank()
        # Validation labels are recovered from numeric ids without ever constructing the test split.
        label_vocab = np.asarray(("/diy/", "/iy/", "/m/", "/n/", "/piy/", "/tiy/", "/uw/", "gnaw", "knew", "pat", "pot"))
        gate = evaluate_global_gate(eeg_embed=collected["embed"], zero_embed=collected["zero"], target_embed=collected["audio"], target_labels=label_vocab[collected["labels"]], token_logits=collected["token_logits"], target_tokens=collected["token_ids"], train_audio_embed=train_embed, train_audio_labels=train_labels, train_token_ids=train_tokens)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), **gate.to_dict(), "selection_split": "subject_val", "test_accessed": False}
        history.append(row)
        write_json(out_dir / "metrics" / "latest_metrics.json", row)
        print(f"[0711v1] {phase} epoch=" + json.dumps(row, ensure_ascii=False), flush=True)
        score = gate.semantic_retrieval_gain + gate.token_retrieval_gain + gate.semantic_over_zero_gain
        if score > best_score:
            best_score, best_gate = score, gate
            checkpoint = out_dir / "checkpoints" / "best.pt"
            save_model(checkpoint, model, cfg=cfg, phase=phase, epoch=epoch, metrics=row)
            write_gate(out_dir / "metrics" / "validation_gate.json", gate, split="subject_val", manifest=manifest, checkpoint=checkpoint)
    write_json(out_dir / "metrics" / "history.json", {"history": history, "best_score": best_score, "selection_split": "subject_val", "test_accessed": False})


def train_flow(args: argparse.Namespace, cfg: dict[str, Any], manifest: SplitManifest, stage: str, seed: int, device: torch.device, out_dir: Path) -> None:
    if not args.resume or not args.gate:
        raise ValueError("flow requires --resume <align_token_checkpoint> and --gate <validation_gate.json>")
    gate = validate_gate_context(args.gate, manifest) if args.allow_gate_bypass else require_flow_gate(args.gate, manifest)
    bank = AudioTargetBank(cache_paths(cfg, stage, seed)["audio_targets"])
    if bank.encodec_latent is None:
        raise FileNotFoundError("Flow is blocked: adapted audio cache has no EnCodec latent targets")
    encoder = make_eeg_model(cfg).to(device)
    load_model(args.resume, encoder, device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    settings = cfg["flow"]
    flow = ConditionalFlowDecoder(latent_dim=int(bank.encodec_latent.shape[-1]), eeg_dim=int(cfg["model"]["d_model"]), d_model=int(settings["d_model"]), heads=int(settings["heads"]), layers=int(settings["layers"])).to(device)
    data = loader(make_dataset(cfg, manifest, "subject_train", stage), int(settings["batch_size"]), cfg, shuffle=True)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=float(settings["lr"]), weight_decay=float(settings["weight_decay"]))
    epochs = 1 if args.smoke else int(args.epochs or settings["epochs"])
    history = []
    for epoch in range(1, epochs + 1):
        flow.train()
        values = []
        iterator = tqdm(data, desc=f"[0711v1] flow {epoch}/{epochs}", unit="batch", dynamic_ncols=True)
        for batch in iterator:
            batch = move_batch(batch, device)
            targets = bank.batch(batch["key"], device)
            z0 = targets["encodec_latent"]
            noise = torch.randn_like(z0)
            t = torch.rand(z0.shape[0], device=device, dtype=z0.dtype).clamp(0.02, 0.98)
            zt = (1.0 - t[:, None, None]) * noise + t[:, None, None] * z0
            with torch.no_grad():
                out = encoder(batch["eeg"], batch["eeg_valid_len"], batch["topography"])
            velocity = flow(zt, t, out["tokens"], out["pred_onset_sec"], out["pred_duration_sec"], out["pred_active_logit"])
            target_velocity = z0 - noise
            loss = flow_matching_loss(velocity, target_velocity)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(flow.parameters(), float(cfg["run"]["grad_clip"]))
            optimizer.step()
            values.append(float(loss.detach().cpu()))
            iterator.set_postfix(flow_loss=f"{values[-1]:.4f}")
        row = {"epoch": epoch, "train_flow_loss": float(np.mean(values)), "selection_split": "subject_val", "test_accessed": False}
        history.append(row)
        write_json(out_dir / "metrics" / "latest_metrics.json", row)
        print("[0711v1] flow epoch=" + json.dumps(row, ensure_ascii=False), flush=True)
        save_model(out_dir / "checkpoints" / "last.pt", flow, cfg=cfg, phase="flow", epoch=epoch, metrics=row)
    write_json(out_dir / "metrics" / "history.json", {
        "history": history,
        "gate": str(args.gate),
        "gate_bypassed": bool(args.allow_gate_bypass and not gate.get("passed")),
        "claim_status": "exploratory_not_reportable" if args.allow_gate_bypass and not gate.get("passed") else "gate_passed",
        "test_accessed": False,
    })


def final_evaluate(args: argparse.Namespace, cfg: dict[str, Any], manifest: SplitManifest, stage: str, seed: int, device: torch.device, out_dir: Path) -> None:
    if not args.allow_final_test or not args.resume:
        raise PermissionError("Final MM21 evaluation requires --resume and explicit --allow-final-test")
    bank = AudioTargetBank(cache_paths(cfg, stage, seed)["audio_targets"])
    model = make_eeg_model(cfg).to(device)
    payload = load_model(args.resume, model, device)
    if payload.get("phase") not in {"align_global", "align_token"}:
        raise ValueError("Final evaluation requires a locked alignment checkpoint")
    test_data = loader(make_dataset(cfg, manifest, "subject_test", stage), int(cfg["alignment"]["batch_size"]), cfg, shuffle=False)
    collected = collect_alignment(model, test_data, bank, device)
    train_embed, train_labels, train_tokens = bank.train_bank()
    label_vocab = np.asarray(("/diy/", "/iy/", "/m/", "/n/", "/piy/", "/tiy/", "/uw/", "gnaw", "knew", "pat", "pot"))
    result = evaluate_global_gate(eeg_embed=collected["embed"], zero_embed=collected["zero"], target_embed=collected["audio"], target_labels=label_vocab[collected["labels"]], token_logits=collected["token_logits"], target_tokens=collected["token_ids"], train_audio_embed=train_embed, train_audio_labels=train_labels, train_token_ids=train_tokens)
    final_test_report(out_dir / "metrics" / "subject_test_final.json", result.to_dict(), manifest=manifest, checkpoint=args.resume)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stage = args.stage or str(cfg["data"]["stage"])
    seed = int(args.seed if args.seed is not None else cfg["run"]["seed"])
    set_seed(seed)
    device = torch.device(args.device) if args.device else default_device()
    root = resolve(cfg["data"]["root"])
    manifest = SplitManifest.build(root)
    paths = cache_paths(cfg, stage, seed)
    paths["split"].parent.mkdir(parents=True, exist_ok=True)
    manifest.write(paths["split"])
    out_dir = phase_dir(cfg, stage, args.phase, seed)
    for folder in ("checkpoints", "metrics"):
        (out_dir / folder).mkdir(exist_ok=True)
    write_json(out_dir / "run_manifest.json", make_run_manifest(repo_root=BUNDLE_DIR.parent.parent, config_path=args.config, split_manifest=manifest, phase=args.phase, stage=stage, seed=seed, input_paths=[paths["split"]]))
    if args.phase == "audit":
        audit = FitAudit.from_records("0711v1_split", manifest, records_for_split(root, manifest, "subject_train"))
        write_json(out_dir / "metrics" / "audit.json", {**audit.to_dict(), "subject_val_n": len(records_for_split(root, manifest, "subject_val")), "subject_test_n": len(records_for_split(root, manifest, "subject_test")), "test_accessed": False})
    elif args.phase == "audio_ssl":
        train_audio_ssl(args, cfg, manifest, stage, seed, device, out_dir)
    elif args.phase == "audio_cache":
        rebuild_audio_cache(args, cfg, manifest, stage, seed, device, out_dir)
    elif args.phase == "eeg_ssl":
        train_eeg_ssl(args, cfg, manifest, stage, device, out_dir)
    elif args.phase == "align_global":
        train_alignment(args, cfg, manifest, stage, seed, device, out_dir, args.phase)
    elif args.phase == "align_token":
        if not args.gate:
            raise ValueError("align_token requires --gate <passed align_global validation gate>")
        if args.allow_gate_bypass:
            validate_gate_context(args.gate, manifest)
        else:
            require_flow_gate(args.gate, manifest)
        train_alignment(args, cfg, manifest, stage, seed, device, out_dir, args.phase)
    elif args.phase == "flow":
        train_flow(args, cfg, manifest, stage, seed, device, out_dir)
    elif args.phase == "evaluate":
        final_evaluate(args, cfg, manifest, stage, seed, device, out_dir)
    print(json.dumps({"run_dir": str(out_dir), "phase": args.phase, "stage": stage, "device": str(device)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
