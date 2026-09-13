#!/usr/bin/env python3
"""Train-fold-only HuBERT masked distillation and HiFi-GAN spectral adaptation.

These are domain adaptation objectives, not a reimplementation of HuBERT
pretraining or adversarial HiFi-GAN training. Never update on validation/test.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import HubertModel, Wav2Vec2FeatureExtractor, SpeechT5HifiGan
import aligned_speech as runner
from eeg2speech.aligned import sha256, tree_hash, atomic_json


def spectral_loss(prediction, target):
    """Phase-insensitive multi-resolution spectral loss, with differentiable CPU STFT on MPS."""
    prediction = prediction[..., :target.shape[-1]]
    if prediction.shape != target.shape:
        raise ValueError("vocoder output is shorter than the target")
    if prediction.device.type == "mps":
        prediction, target = prediction.cpu(), target.cpu()
    losses = []
    for size in (256, 512, 1024):
        window = torch.hann_window(size, device=prediction.device)
        p = torch.stft(prediction, size, size // 4, window=window, return_complex=True).abs().clamp_min(1e-5)
        t = torch.stft(target, size, size // 4, window=window, return_complex=True).abs().clamp_min(1e-5)
        convergence = (p - t).flatten(1).norm(dim=1) / t.flatten(1).norm(dim=1).clamp_min(1e-5)
        losses.append(convergence.mean() + F.l1_loss(p.log(), t.log()))
    return torch.stack(losses).mean()


def teacher_loss(model, processor, batch, seed):
    inputs = processor(batch["wave"].cpu().numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(model.device)
    teacher = batch["teacher"].to(model.device)
    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand(teacher.shape[:2], generator=generator) < .35
    mask[:, 0] = True
    mask = mask.to(model.device)
    prediction = model(inputs, mask_time_indices=mask, output_hidden_states=True).hidden_states[9]
    # Fixed pretrained clean targets prevent teacher/student moving together.
    return F.mse_loss(prediction[mask], teacher[mask]) + .1 * F.mse_loss(prediction, teacher)


def train(args):
    cfg = runner.config(args.config)
    target = runner.device(args.device)
    runner.seed_all(31)
    base = Path(args.base).resolve()
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    train_data = runner.dataset_for(cfg, "train", audio=True, cache=args.cache)
    validation = runner.dataset_for(cfg, "validation", audio=True, cache=args.cache)
    signature = dict(stage=args.kind, base=tree_hash(base), cache=sha256(Path(args.cache)),
                     manifest=sha256(runner.artifact_paths(cfg)[1]), epochs=args.epochs,
                     lr=args.lr, runtime=runner.runtime_hash(), code=sha256(Path(__file__)),
                     device=str(target), batch_size=args.batch_size)
    processor = None
    if args.kind == "hubert":
        if train_data.teacher_sha256 != signature["base"]:
            raise ValueError("HuBERT clean targets must come from the exact base teacher")
        model = HubertModel.from_pretrained(base, local_files_only=True)
        processor = Wav2Vec2FeatureExtractor.from_pretrained(base, local_files_only=True)
        # Layers above the consumed ninth hidden state cannot receive this loss.
        model.feature_extractor._freeze_parameters()
        for layer in model.encoder.layers[9:]:
            layer.requires_grad_(False)
        model.config.layerdrop = 0.
        model.config.mask_time_prob = .05
        model.config.mask_feature_prob = 0.
        model.gradient_checkpointing_enable()
    else:
        model = SpeechT5HifiGan.from_pretrained(base, local_files_only=True)
        if int(np.prod(model.config.upsample_rates)) != 256:
            raise ValueError("vocoder does not use the native 256 sample hop")
    model.to(target)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    progress = output / "training_state.pt"
    start_epoch = 0; best = float("inf"); history = []
    if progress.exists():
        state = torch.load(progress, map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise ValueError("adaptation resume configuration differs; use a new output directory")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        for item in optimizer.state.values():
            for key, value in item.items():
                if torch.is_tensor(value): item[key] = value.to(target)
        start_epoch, best, history = state["epoch"], state["best"], state["history"]

    def objective(batch, seed):
        if args.kind == "hubert":
            return teacher_loss(model, processor, batch, seed)
        mel = batch["mel"].to(target)
        wave = batch["wave"].to(target)
        # Train on a contiguous one-second crop with 16 frames of context on
        # each side. Validation covers the full four-second waveform.
        if model.training:
            gen = torch.Generator().manual_seed(seed)
            start = int(torch.randint(0, mel.shape[-1] - 64 + 1, (1,), generator=gen))
            left, right = max(0, start - 16), min(mel.shape[-1], start + 64 + 16)
            prediction = model(mel[..., left:right].transpose(1, 2))
            prediction = prediction[..., (start-left)*256:(start-left+64)*256]
            wave = wave[..., start*256:min((start+64)*256, wave.shape[-1])]
        else:
            prediction = model(mel.transpose(1, 2))
        return spectral_loss(prediction, wave)

    for epoch in range(start_epoch, args.epochs):
        runner.seed_all(31 + epoch)
        started = time.monotonic(); model.train(); losses = []
        loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(31 + epoch))
        for i, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(batch, epoch * 10000 + i)
            if not torch.isfinite(loss): raise ValueError("nonfinite adaptation loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); losses.append(float(loss.detach()))
            if i % 10 == 0:
                print(json.dumps(dict(stage=args.kind, epoch=epoch+1, batch=i, loss=losses[-1])), flush=True)
        model.eval(); errors = []
        with torch.no_grad():
            for i, batch in enumerate(DataLoader(validation, batch_size=1)):
                errors.append(float(objective(batch, 900000 + i)))
        metric = float(np.mean(errors))
        if not np.isfinite(metric): raise ValueError("nonfinite validation loss")
        history.append(dict(epoch=epoch+1, train_loss=float(np.mean(losses)), validation_loss=metric,
                            seconds=time.monotonic()-started))
        if metric < best:
            best = metric
            # Final published directory is immutable once downstream caches use it.
            model.save_pretrained(output / "best")
            if processor is not None: processor.save_pretrained(output / "best")
        runner.atomic_save(progress, dict(signature=signature, epoch=epoch+1, best=best,
                                          model=model.state_dict(), optimizer=optimizer.state_dict(), history=history))
        print(json.dumps(history[-1]), flush=True)
    atomic_json(output / "adaptation_report.json", dict(signature=signature, history=history,
                fit_role="train", selection_role="validation", test_used=False,
                train_unique_audio=len(train_data), validation_unique_audio=len(validation),
                selected_model_sha256=tree_hash(output / "best"), status="complete"))


def download(args):
    from huggingface_hub import HfApi, snapshot_download
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    for name, repo in (("hubert", "facebook/hubert-base-ls960"), ("hifigan", "microsoft/speecht5_hifigan")):
        folder = root / name
        lock = root / f"{name}_download.json"
        if lock.exists() and (folder / "config.json").exists() and (folder / "pytorch_model.bin").exists() and (name != "hubert" or (folder / "preprocessor_config.json").exists()):
            print(f"using downloaded {name}: {folder}", flush=True)
            continue
        revision = json.loads(lock.read_text())["revision"] if lock.exists() else HfApi().model_info(repo).sha
        atomic_json(lock, dict(repository=repo, revision=revision))
        snapshot_download(repo, revision=revision, local_dir=folder,
                          allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin"])
        # Store download metadata outside the tree consumed by cache hashing.
        print(json.dumps(dict(model=name, path=str(folder), revision=revision)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download"); p.add_argument("--output", required=True)
    p = sub.add_parser("train")
    p.add_argument("--kind", choices=("hubert", "hifigan"), required=True)
    p.add_argument("--config", required=True); p.add_argument("--base", required=True)
    p.add_argument("--cache", required=True); p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto"); p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1); p.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    if args.command == "download": download(args)
    else:
        if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0: parser.error("positive training settings required")
        train(args)

if __name__ == "__main__":
    main()
