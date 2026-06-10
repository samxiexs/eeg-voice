"""Train the v3 EEG->speech model (see NEW_DESIGN_eeg2speech_v3.md).

Curriculum usage:
  P1 (teacher / upper-bound probe) — train on the strong overt-speech stage:
      python scripts/v3_train.py --config configs/v3_encodec.yaml \
          --protocol G --stage speaking --run-suffix speaking_teacher

  P2 (main thinking model, warm-started from P1, optional KD):
      python scripts/v3_train.py --config configs/v3_encodec.yaml \
          --protocol G --stage thinking --run-suffix thinking_main \
          --init-from <P1 best.pt> --distill-teacher <P1 best.pt> --teacher-stage speaking
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, set_seed, write_json
from src.v3.data import V3Dataset
from src.v3.eval import evaluate
from src.v3.losses import compute_v3_losses
from src.v3.model import EEG2SpeechV3, EEG2SpeechV3Config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train v3 EEG-to-speech model.")
    p.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "v3_encodec.yaml"))
    p.add_argument("--data-root", default=None)
    p.add_argument("--output-root", default=None)
    p.add_argument("--protocol", default=None, help="S | G | U")
    p.add_argument("--subject", default=None, help="Required for Protocol S")
    p.add_argument("--holdout-subject", default=None, help="Required for Protocol U")
    p.add_argument("--stage", default=None, help="thinking | speaking | stimuli")
    p.add_argument("--run-suffix", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--init-from", default=None, help="Warm-start weights from a checkpoint")
    p.add_argument("--distill-teacher", default=None, help="Frozen teacher checkpoint for KD")
    p.add_argument("--teacher-stage", default=None, help="EEG stage fed to the KD teacher")
    p.add_argument("--max-steps", type=int, default=None, help="Debug: cap optimisation steps")
    return p.parse_args()


def build_datasets(config: dict, args: argparse.Namespace):
    cfg_d, cfg_a, cfg_t = config["data"], config["audio"], config["targets"]
    root = resolve_bundle_path(args.data_root or cfg_d["root"], BUNDLE_DIR)
    cache = resolve_bundle_path(cfg_t["cache_path"], BUNDLE_DIR)
    protocol = (args.protocol or cfg_d["protocol"]).upper()
    stage = args.stage or cfg_d.get("stage", "thinking")
    common = dict(
        data_root=str(root),
        protocol=protocol,
        stage=stage,
        subject_id=args.subject or cfg_d.get("subject_id"),
        holdout_subject_id=args.holdout_subject or cfg_d.get("holdout_subject_id"),
        include_anomalous=bool(cfg_d.get("include_anomalous", False)),
        target_cache_path=str(cache),
        require_targets=True,
        audio_sr=int(cfg_a["sample_rate"]),
        audio_dur=float(cfg_a["duration_sec"]),
        teacher_stage=args.teacher_stage,
    )
    train = V3Dataset(split="train", **common)
    val = V3Dataset(split="val", **common)
    test = V3Dataset(split="test", **common)
    return train, val, test, protocol, stage


def make_model(config: dict, dataset) -> EEG2SpeechV3:
    m = config["model"]
    cfg = EEG2SpeechV3Config(
        n_channels_eeg=int(m.get("n_channels_eeg", 14)),
        d_model=int(m.get("d_model", 256)),
        cond_dim=int(m.get("cond_dim", 64)),
        num_subjects=int(dataset.num_subjects),
        target_steps=int(dataset.target_sequence_steps),
        target_dim=int(dataset.target_sequence_dim),
        num_labels=int(dataset.num_labels),
        num_blocks=int(m.get("num_blocks", 5)),
        kernel_size=int(m.get("kernel_size", 5)),
        channel_dropout=float(m.get("channel_dropout", 0.1)),
        dropout=float(m.get("dropout", 0.1)),
        embed_dim=int(m.get("embed_dim", dataset.target_sequence_dim)),
    )
    return EEG2SpeechV3(cfg)


def run_epoch(model, loader, device, cfg_train, optimizer=None, teacher=None, max_steps=None):
    is_train = optimizer is not None
    model.train(is_train)
    agg: dict[str, float] = {}
    count = 0
    steps = 0
    for batch in loader:
        eeg = batch["eeg"].to(device)
        subj = batch["subject_index"].to(device)
        target_seq = batch["target_sequence"].to(device)
        target_sum = batch["target_summary"].to(device)
        label_ids = batch["label_id"].to(device)
        out = model(eeg, subj)

        teacher_out = None
        if teacher is not None and "teacher_eeg" in batch:
            with torch.no_grad():
                teacher_out = teacher(batch["teacher_eeg"].to(device), subj)

        losses = compute_v3_losses(
            out,
            target_sequence=target_seq,
            target_summary=target_sum,
            label_ids=label_ids,
            lambda_contrastive=float(cfg_train.get("lambda_contrastive", 1.0)),
            lambda_latent_cosine=float(cfg_train.get("lambda_latent_cosine", 1.0)),
            lambda_latent_mse=float(cfg_train.get("lambda_latent_mse", 0.5)),
            lambda_cls=float(cfg_train.get("lambda_cls", 0.5)),
            contrastive_temperature=float(cfg_train.get("contrastive_temperature", 0.07)),
            teacher_outputs=teacher_out,
            lambda_kd_latent=float(cfg_train.get("lambda_kd_latent", 0.0)) if teacher_out else 0.0,
            lambda_kd_logits=float(cfg_train.get("lambda_kd_logits", 0.0)) if teacher_out else 0.0,
        )
        if is_train:
            optimizer.zero_grad()
            losses["total"].backward()
            clip_grad_norm_(model.parameters(), float(cfg_train.get("grad_clip", 1.0)))
            optimizer.step()

        bs = eeg.shape[0]
        count += bs
        for k, v in losses.items():
            agg[k] = agg.get(k, 0.0) + float(v.detach()) * bs
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return {k: v / max(count, 1) for k, v in agg.items()}


def main() -> None:
    args = parse_args()
    config = load_simple_yaml(args.config)
    cfg_train = config["train"]
    set_seed(int(cfg_train.get("seed", 7)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds, protocol, stage = build_datasets(config, args)
    model = make_model(config, train_ds).to(device)

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[init-from] loaded {args.init_from} | missing={len(missing)} unexpected={len(unexpected)}")

    teacher = None
    if args.distill_teacher:
        teacher = make_model(config, train_ds).to(device)
        tck = torch.load(args.distill_teacher, map_location=device)
        teacher.load_state_dict(tck["model_state"], strict=False)
        teacher.eval()
        for prm in teacher.parameters():
            prm.requires_grad_(False)
        print(f"[distill] teacher loaded from {args.distill_teacher}")

    epochs = args.epochs or int(cfg_train.get("epochs", 80))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_train.get("lr", 2e-4)),
        weight_decay=float(cfg_train.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    num_workers = int(cfg_train.get("num_workers", 0))
    bs = int(cfg_train.get("batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers)

    suffix = args.run_suffix or config.get("output", {}).get("run_suffix", "v3")
    run_name = f"{protocol.lower()}_{stage}_{suffix}"
    out_root = resolve_bundle_path(args.output_root or config["output"]["root"], BUNDLE_DIR)
    run_dir = Path(out_root) / run_name
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "metrics")

    target_mean = np.asarray(train_ds.target_cache["target_mean"], dtype=np.float32)
    target_std = np.asarray(train_ds.target_cache["target_std"], dtype=np.float32)

    best_score = -1e9
    history = []
    for epoch in range(epochs):
        tr = run_epoch(model, train_loader, device, cfg_train, optimizer=optimizer,
                       teacher=teacher, max_steps=args.max_steps)
        va = run_epoch(model, val_loader, device, cfg_train, optimizer=None,
                       teacher=teacher, max_steps=args.max_steps)
        scheduler.step()
        # Selection score: high val latent cosine + low contrastive loss.
        score = va.get("latent_cosine", 0.0) - 0.1 * va.get("contrastive", 0.0)
        history.append({"epoch": epoch, "train": tr, "val": va, "score": score})
        tqdm.write(
            f"epoch {epoch:03d} | train total {tr['total']:.3f} cls_acc {tr['cls_acc']:.3f} "
            f"| val cos {va['latent_cosine']:.3f} contr {va['contrastive']:.3f} cls_acc {va['cls_acc']:.3f}"
        )
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": vars(model.config),
                    "label_vocab": list(train_ds.label_vocab),
                    "subject_vocab": list(train_ds.subject_vocab),
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "protocol": protocol,
                    "stage": stage,
                    "epoch": epoch,
                },
                run_dir / "checkpoints" / "best.pt",
            )

    write_json(run_dir / "metrics" / "history.json", history)

    # Final retrieval evaluation on the test split.
    # Protocol U's held-out subject is absent from train, so its subject-specific
    # template only exists in the test bank (oracle); G/S use the train bank.
    bank_split = "test" if protocol == "U" else "train"
    test_metrics = evaluate(model, test_ds, device=device, top_k=int(config.get("eval", {}).get("top_k", 5)),
                            batch_size=bs, bank_split=bank_split)
    summary = {k: v for k, v in test_metrics.items() if k != "predictions"}
    print("[test]", summary)
    write_json(run_dir / "metrics" / "test_metrics.json", test_metrics)
    print(f"[done] run dir: {run_dir}")


if __name__ == "__main__":
    main()
