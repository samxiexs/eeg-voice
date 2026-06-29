from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.karaone_recon.data import KaraOneTrialDataset
from src.karaone_recon.retrieval_first import KaraOneRetrievalFirst, RetrievalFirstConfig
from src.karaone_recon.semantic_tokens import KaraOneSemanticTokenTargets
from src.karaone_recon.targets import KaraOneTargets
from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, resolve_target_cache, set_seed, write_json

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


def _progress_bar(iterable, *, total: int, desc: str):
    if tqdm is None or os.environ.get("DISABLE_TQDM", "0") == "1":
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KaraOne v6.1 retrieval-first EEG-to-speech model.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--stages", default=None)
    parser.add_argument("--model", choices=["baseline", "moe"], default="baseline")
    parser.add_argument("--core-cache", default="../artifacts/audio_targets/karaone_temporal_elastic_core_v5.npz")
    parser.add_argument("--token-cache", default="../artifacts/audio_targets/karaone_trial_hubert_tokens_k64_trainonly.npz")
    parser.add_argument("--subject-val", default=None)
    parser.add_argument("--subject-test", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-suffix", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--lambda-trial-infonce", type=float, default=1.0)
    parser.add_argument("--lambda-hubert-summary-reg", type=float, default=0.5)
    parser.add_argument("--lambda-vicreg-variance", type=float, default=0.3)
    parser.add_argument("--lambda-vicreg-covariance", type=float, default=0.1)
    parser.add_argument("--lambda-semantic-token-ce", type=float, default=0.3)
    parser.add_argument("--lambda-speech-token-ctc", type=float, default=0.2)
    parser.add_argument("--lambda-content-ce", type=float, default=0.05)
    parser.add_argument("--lambda-mel-residual", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--retrieval-topk", type=int, default=3)
    parser.add_argument("--retrieval-temperature", type=float, default=0.05)
    return parser.parse_args()


def _sample_pcc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.reshape(a.shape[0], -1).astype(np.float64)
    b = b.reshape(b.shape[0], -1).astype(np.float64)
    am = a - a.mean(axis=1, keepdims=True)
    bm = b - b.mean(axis=1, keepdims=True)
    return (am * bm).sum(axis=1) / (
        np.sqrt((am * am).sum(axis=1)) * np.sqrt((bm * bm).sum(axis=1)) + 1e-8
    )


def _corr_median(x: np.ndarray, max_items: int = 256) -> float:
    if x.shape[0] < 2:
        return 0.0
    x = x[:max_items].reshape(min(max_items, x.shape[0]), -1).astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)
    corr = x @ x.T
    upper = corr[np.triu_indices(corr.shape[0], k=1)]
    return float(np.median(upper)) if upper.size else 0.0


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    return x.flatten()[:-1].view(n - 1, m + 1)[:, 1:].flatten()


def _vicreg_losses(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    var_loss = torch.mean(F.relu(1.0 - std))
    if z.shape[0] < 2:
        cov_loss = torch.zeros((), device=z.device, dtype=z.dtype)
    else:
        cov = (z.T @ z) / float(z.shape[0] - 1)
        cov_loss = _off_diagonal(cov).pow(2).sum() / float(max(z.shape[1], 1))
    return var_loss, cov_loss


def _symmetric_infonce(eeg: torch.Tensor, audio: torch.Tensor, temperature: float) -> torch.Tensor:
    eeg = F.normalize(eeg, dim=-1)
    audio = F.normalize(audio, dim=-1)
    logits = eeg @ audio.T / max(float(temperature), 1e-4)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _semantic_token_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, vocab: int) -> torch.Tensor:
    logits = logits[..., :vocab]
    loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1).clamp(min=0, max=vocab - 1), reduction="none")
    mask_flat = mask.reshape(-1).float()
    return (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)


def _speech_token_ctc(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, blank: int) -> torch.Tensor:
    if logits.shape[-1] <= blank:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
    input_lengths = torch.full((logits.shape[0],), logits.shape[1], device=logits.device, dtype=torch.long)
    pieces: list[torch.Tensor] = []
    lengths: list[int] = []
    target_cpu = targets.detach().cpu()
    mask_cpu = mask.detach().cpu()
    for i in range(target_cpu.shape[0]):
        values = target_cpu[i][mask_cpu[i].bool()].long()
        if values.numel() == 0:
            values = torch.zeros(1, dtype=torch.long)
        collapsed = [int(values[0])]
        for item in values[1:].tolist():
            if int(item) != collapsed[-1]:
                collapsed.append(int(item))
        seq = torch.tensor(collapsed, device=logits.device, dtype=torch.long)
        pieces.append(seq)
        lengths.append(int(seq.numel()))
    flat = torch.cat(pieces, dim=0)
    target_lengths = torch.tensor(lengths, device=logits.device, dtype=torch.long)
    return F.ctc_loss(log_probs, flat, input_lengths, target_lengths, blank=blank, zero_infinity=True)


def _make_dataset(
    *,
    root: Path,
    core_targets: KaraOneTargets,
    hubert_targets: KaraOneTargets,
    token_targets: KaraOneSemanticTokenTargets | None,
    split: str,
    stages: tuple[str, ...],
    split_protocol: str,
    heldout_subjects: list[str],
    eeg_len: int,
) -> KaraOneTrialDataset:
    return KaraOneTrialDataset(
        data_root=root,
        targets=core_targets,
        aux_targets=hubert_targets,
        semantic_token_targets=token_targets,
        split=split,
        stages=stages,
        split_protocol=split_protocol,
        heldout_subjects=heldout_subjects,
        eeg_len=eeg_len,
    )


def _collect_bank(dataset: KaraOneTrialDataset) -> dict[str, Any]:
    rows = [dataset[i] for i in range(len(dataset))]
    return {
        "audio_embed": np.stack([row["hubert_summary"].numpy() for row in rows], axis=0).astype(np.float32),
        "core_seq": np.stack([row["target_seq"].numpy() for row in rows], axis=0).astype(np.float32),
        "template_ids": np.asarray([row["template_id"] for row in rows]).astype(str),
        "subjects": np.asarray([row["subject"] for row in rows]).astype(str),
        "labels": np.asarray([row["label"] for row in rows]).astype(str),
        "trial_indices": np.asarray([int(row["trial_index"]) for row in rows], dtype=np.int32),
        "active_duration_frames": np.asarray(
            [int(row.get("active_duration_frames", torch.tensor(dataset.targets.T)).item()) for row in rows],
            dtype=np.float32,
        ),
        "active_center_frame": np.asarray(
            [int(row.get("active_center_frame", torch.tensor(getattr(dataset.targets, "global_core_insert_frame", 0))).item()) for row in rows],
            dtype=np.float32,
        ),
        "active_rms": np.asarray(
            [float(row.get("active_rms", torch.tensor(0.08)).item()) for row in rows],
            dtype=np.float32,
        ),
        "active_peak": np.asarray(
            [float(row.get("active_peak", torch.tensor(0.1)).item()) for row in rows],
            dtype=np.float32,
        ),
    }


def _retrieve_prior_torch(
    query: torch.Tensor,
    bank_audio: torch.Tensor,
    bank_core: torch.Tensor,
    *,
    topk: int,
    temperature: float,
    bank_ids: list[str] | None = None,
    query_ids: list[str] | None = None,
) -> torch.Tensor:
    query = F.normalize(query, dim=-1)
    scores = query @ bank_audio.T
    if bank_ids is not None and query_ids is not None:
        id_to_idx = {item: idx for idx, item in enumerate(bank_ids)}
        for row, item in enumerate(query_ids):
            idx = id_to_idx.get(str(item))
            if idx is not None:
                scores[row, idx] = -1e9
    k = max(1, min(int(topk), int(bank_core.shape[0])))
    vals, idx = torch.topk(scores, k=k, dim=-1)
    weights = torch.softmax(vals / max(float(temperature), 1e-4), dim=-1)
    return (bank_core[idx] * weights[..., None, None]).sum(dim=1)


@torch.no_grad()
def _collect_model_outputs(
    model: KaraOneRetrievalFirst,
    dataset: KaraOneTrialDataset,
    device: str | torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    pred, zero, audio, delta, target = [], [], [], [], []
    subjects, labels, trials, template_ids = [], [], [], []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        stage = batch["stage_idx"].to(device)
        valid = batch["eeg_valid_len"].to(device)
        out = model(eeg, stage, valid)
        zero_out = model(torch.zeros_like(eeg), stage, valid)
        pred.append(out["eeg_embed"].detach().cpu().numpy())
        zero.append(zero_out["eeg_embed"].detach().cpu().numpy())
        audio.append(batch["hubert_summary"].numpy())
        delta.append(out["pred_core_delta"].detach().cpu().numpy())
        target.append(batch["target_seq"].numpy())
        subjects.extend([str(item) for item in batch["subject"]])
        labels.extend([str(item) for item in batch["label"]])
        trials.extend([int(item) for item in batch["trial_index"]])
        template_ids.extend([str(item) for item in batch["template_id"]])
    return {
        "pred": np.concatenate(pred, axis=0),
        "zero": np.concatenate(zero, axis=0),
        "audio": np.concatenate(audio, axis=0),
        "delta": np.concatenate(delta, axis=0),
        "target_core": np.concatenate(target, axis=0),
        "subjects": subjects,
        "labels": labels,
        "trials": trials,
        "template_ids": template_ids,
    }


def _pair_retrieval(query: np.ndarray, audio: np.ndarray, prefix: str) -> dict[str, float]:
    q = query / np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-8)
    a = audio / np.linalg.norm(audio, axis=1, keepdims=True).clip(min=1e-8)
    scores = q @ a.T
    order = np.argsort(scores, axis=1)[:, ::-1]
    ranks = np.zeros(scores.shape[0], dtype=np.int32)
    for i in range(scores.shape[0]):
        ranks[i] = int(np.where(order[i] == i)[0][0]) + 1
    out = {
        f"{prefix}_trial_top1": float(np.mean(ranks <= 1)),
        f"{prefix}_trial_top3": float(np.mean(ranks <= min(3, scores.shape[0]))),
        f"{prefix}_trial_top5": float(np.mean(ranks <= min(5, scores.shape[0]))),
        f"{prefix}_mrr": float(np.mean(1.0 / ranks)),
    }
    return out


def _retrieve_prior_np(
    query: np.ndarray,
    bank_audio: np.ndarray,
    bank_core: np.ndarray,
    topk: int,
    temperature: float,
) -> np.ndarray:
    q = query / np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-8)
    b = bank_audio / np.linalg.norm(bank_audio, axis=1, keepdims=True).clip(min=1e-8)
    scores = q @ b.T
    k = max(1, min(int(topk), bank_core.shape[0]))
    idx = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
    vals = np.take_along_axis(scores, idx, axis=1)
    weights = np.exp((vals - vals.max(axis=1, keepdims=True)) / max(float(temperature), 1e-4))
    weights = weights / weights.sum(axis=1, keepdims=True).clip(min=1e-8)
    return (bank_core[idx] * weights[..., None, None]).sum(axis=1).astype(np.float32)


def _evaluate_retrieval(
    model: KaraOneRetrievalFirst,
    dataset: KaraOneTrialDataset,
    train_bank: dict[str, Any],
    device: str | torch.device,
    batch_size: int,
    *,
    residual_scale: float,
    topk: int,
    retrieval_temperature: float,
) -> dict[str, Any]:
    data = _collect_model_outputs(model, dataset, device, batch_size)
    audio = data["audio"].astype(np.float32)
    pred = data["pred"].astype(np.float32)
    zero = data["zero"].astype(np.float32)
    shuffled = np.roll(pred, 1, axis=0)
    mean_query = np.repeat(train_bank["audio_embed"].mean(axis=0, keepdims=True), pred.shape[0], axis=0).astype(np.float32)
    metrics: dict[str, Any] = {"n": int(pred.shape[0])}
    metrics.update(_pair_retrieval(pred, audio, "pred_pair"))
    metrics.update(_pair_retrieval(zero, audio, "zeroeeg_pair"))
    metrics.update(_pair_retrieval(shuffled, audio, "shuffled_pair"))
    metrics.update(_pair_retrieval(mean_query, audio, "mean_pair"))
    pred_cos = np.sum(
        pred / np.linalg.norm(pred, axis=1, keepdims=True).clip(min=1e-8)
        * audio / np.linalg.norm(audio, axis=1, keepdims=True).clip(min=1e-8),
        axis=1,
    )
    zero_cos = np.sum(
        zero / np.linalg.norm(zero, axis=1, keepdims=True).clip(min=1e-8)
        * audio / np.linalg.norm(audio, axis=1, keepdims=True).clip(min=1e-8),
        axis=1,
    )
    mean_cos = np.sum(
        mean_query / np.linalg.norm(mean_query, axis=1, keepdims=True).clip(min=1e-8)
        * audio / np.linalg.norm(audio, axis=1, keepdims=True).clip(min=1e-8),
        axis=1,
    )
    metrics["pred_hubert_cos"] = float(pred_cos.mean())
    metrics["zeroeeg_hubert_cos"] = float(zero_cos.mean())
    metrics["mean_hubert_cos"] = float(mean_cos.mean())
    metrics["pred_hubert_cos_gain"] = float(pred_cos.mean() - max(float(zero_cos.mean()), float(mean_cos.mean())))

    bank_audio = train_bank["audio_embed"].astype(np.float32)
    bank_core = train_bank["core_seq"].astype(np.float32)
    pred_prior = _retrieve_prior_np(pred, bank_audio, bank_core, topk, retrieval_temperature)
    zero_prior = _retrieve_prior_np(zero, bank_audio, bank_core, topk, retrieval_temperature)
    mean_core = np.repeat(bank_core.mean(axis=0, keepdims=True), pred.shape[0], axis=0).astype(np.float32)
    target_core = data["target_core"].astype(np.float32)
    pred_core = pred_prior + float(residual_scale) * data["delta"].astype(np.float32)
    pred_shape = float(_sample_pcc(pred_core, target_core).mean())
    prior_shape = float(_sample_pcc(pred_prior, target_core).mean())
    zero_shape = float(_sample_pcc(zero_prior, target_core).mean())
    mean_shape = float(_sample_pcc(mean_core, target_core).mean())
    metrics.update(
        {
            "pred_active_shape_corr": pred_shape,
            "retrieved_prior_active_shape_corr": prior_shape,
            "zeroeeg_active_shape_corr": zero_shape,
            "mean_active_shape_corr": mean_shape,
            "pred_over_zero_active_shape_gain": pred_shape - zero_shape,
            "pred_over_mean_active_shape_gain": pred_shape - mean_shape,
            "retrieved_prior_over_mean_active_shape_gain": prior_shape - mean_shape,
            "pred_pairwise_corr_median": _corr_median(pred_core),
            "pred_std_ratio_median": float(
                np.median(pred_core.reshape(pred_core.shape[0], -1).std(axis=0) / target_core.reshape(target_core.shape[0], -1).std(axis=0).clip(min=1e-6))
            ),
        }
    )
    metrics["pred_pair_trial_top3_gain"] = float(
        metrics["pred_pair_trial_top3"]
        - max(metrics["zeroeeg_pair_trial_top3"], metrics["mean_pair_trial_top3"], metrics["shuffled_pair_trial_top3"])
    )
    metrics["pred_pair_mrr_gain"] = float(
        metrics["pred_pair_mrr"] - max(metrics["zeroeeg_pair_mrr"], metrics["mean_pair_mrr"], metrics["shuffled_pair_mrr"])
    )
    metrics["selection_score"] = _selection_score(metrics)
    return metrics


def _selection_score(metrics: dict[str, Any]) -> float:
    base = (
        1.0 * float(metrics.get("pred_pair_trial_top3_gain", 0.0))
        + 0.8 * float(metrics.get("pred_pair_mrr_gain", 0.0))
        + 0.8 * float(metrics.get("pred_hubert_cos_gain", 0.0))
        + 0.5 * float(metrics.get("pred_over_zero_active_shape_gain", 0.0))
        + 0.3 * float(metrics.get("pred_over_mean_active_shape_gain", 0.0))
    )
    penalty = 0.3 * max(0.0, float(metrics.get("pred_pairwise_corr_median", 0.0)) - 0.85)
    return float(base - penalty)


def main() -> None:
    args = parse_args()
    cfg = load_simple_yaml(args.config)
    train_cfg, model_cfg = cfg["train"], cfg["model"]
    set_seed(int(train_cfg.get("seed", 7)))
    device = args.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    stages = tuple(item.strip() for item in (args.stages or cfg["data"].get("stages", "overt_like")).split(",") if item.strip())
    root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
    _, hubert_cache = resolve_target_cache(cfg, BUNDLE_DIR, "hubert_sequence")
    core_cache = resolve_bundle_path(args.core_cache, BUNDLE_DIR)
    token_path = resolve_bundle_path(args.token_cache, BUNDLE_DIR)
    hubert_targets = KaraOneTargets(hubert_cache, data_root=root)
    core_targets = KaraOneTargets(core_cache, data_root=root)
    token_targets = KaraOneSemanticTokenTargets(token_path) if token_path.exists() else None
    heldout = [str(item) for item in cfg["data"].get("heldout_subjects", ["P02", "MM21"])]
    subject_val = str(args.subject_val or heldout[0])
    subject_test = str(args.subject_test or (heldout[1] if len(heldout) > 1 else heldout[0]))
    heldout_pair = sorted(set([subject_val, subject_test]))
    eeg_len = int(cfg["data"].get("eeg_len", 1280))
    train_ds = _make_dataset(
        root=root,
        core_targets=core_targets,
        hubert_targets=hubert_targets,
        token_targets=token_targets,
        split="train",
        stages=stages,
        split_protocol="subject_holdout",
        heldout_subjects=heldout_pair,
        eeg_len=eeg_len,
    )
    subject_val_ds = _make_dataset(
        root=root,
        core_targets=core_targets,
        hubert_targets=hubert_targets,
        token_targets=token_targets,
        split="subject_test",
        stages=stages,
        split_protocol="subject_holdout",
        heldout_subjects=[subject_val],
        eeg_len=eeg_len,
    )
    subject_test_ds = _make_dataset(
        root=root,
        core_targets=core_targets,
        hubert_targets=hubert_targets,
        token_targets=token_targets,
        split="subject_test",
        stages=stages,
        split_protocol="subject_holdout",
        heldout_subjects=[subject_test],
        eeg_len=eeg_len,
    )
    val_ds = _make_dataset(
        root=root,
        core_targets=core_targets,
        hubert_targets=hubert_targets,
        token_targets=token_targets,
        split="val",
        stages=stages,
        split_protocol=str(cfg["data"].get("split_protocol", "trial")),
        heldout_subjects=heldout_pair,
        eeg_len=eeg_len,
    )
    test_ds = _make_dataset(
        root=root,
        core_targets=core_targets,
        hubert_targets=hubert_targets,
        token_targets=token_targets,
        split="test",
        stages=stages,
        split_protocol=str(cfg["data"].get("split_protocol", "trial")),
        heldout_subjects=heldout_pair,
        eeg_len=eeg_len,
    )
    run = f"karaone_retrieval_first_{args.model}_{'_'.join(stages)}_{args.run_suffix or 'v61'}"
    run_dir = resolve_bundle_path(cfg["output"]["root"], BUNDLE_DIR) / run
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "metrics")
    train_bank = _collect_bank(train_ds)
    audit = {
        "stages": list(stages),
        "subject_val": subject_val,
        "subject_test": subject_test,
        "splits": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
            "subject_val": len(subject_val_ds),
            "subject_test": len(subject_test_ds),
        },
        "train_bank": {
            "n": int(train_bank["audio_embed"].shape[0]),
            "audio_embed_dim": int(train_bank["audio_embed"].shape[1]),
            "core_shape": list(train_bank["core_seq"].shape),
            "core_pairwise_corr_median": _corr_median(train_bank["core_seq"]),
        },
    }
    write_json(run_dir / "metrics" / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.audit_only:
        print(f"[done] audit only: {run_dir}")
        return

    num_channel_experts = 1 if args.model == "baseline" else max(4, int(model_cfg.get("num_channel_experts", 4)))
    token_vocab = int(token_targets.vocab_size + 1) if token_targets is not None else 0
    model = KaraOneRetrievalFirst(
        RetrievalFirstConfig(
            n_channels_eeg=int(model_cfg.get("n_channels_eeg", 62)),
            d_model=int(model_cfg.get("d_model", 256)),
            cond_dim=int(model_cfg.get("cond_dim", 64)),
            num_labels=train_ds.num_labels,
            num_stages=train_ds.num_stages,
            core_steps=core_targets.T,
            core_dim=core_targets.D,
            audio_embed_dim=hubert_targets.D,
            num_blocks=int(model_cfg.get("num_blocks", 6)),
            kernel_size=int(model_cfg.get("kernel_size", 5)),
            channel_dropout=float(model_cfg.get("channel_dropout", 0.15)),
            dropout=float(model_cfg.get("dropout", 0.15)),
            num_channel_experts=num_channel_experts,
            encoder_kind=str(model_cfg.get("encoder_kind", "cnn")),
            transformer_layers=int(model_cfg.get("transformer_layers", 4)),
            transformer_heads=int(model_cfg.get("transformer_heads", 4)),
            patch_stride=int(model_cfg.get("patch_stride", 4)),
            instance_norm=True,
            use_channel_reliability=True,
            semantic_token_vocab=token_vocab,
            semantic_token_steps=int(token_targets.T) if token_targets is not None else 50,
        )
    ).to(device)

    bank_audio_t = torch.from_numpy(train_bank["audio_embed"]).to(device).float()
    bank_audio_t = F.normalize(bank_audio_t, dim=-1)
    bank_core_t = torch.from_numpy(train_bank["core_seq"]).to(device).float()
    bank_ids = [str(item) for item in train_bank["template_ids"].tolist()]
    epochs = int(args.epochs or train_cfg.get("epochs", 30))
    batch_size = int(train_cfg.get("batch_size", 48))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=int(train_cfg.get("num_workers", 0)), drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 3e-4)), weight_decay=float(train_cfg.get("weight_decay", 1e-3)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    history = run_dir / "metrics" / "history.csv"
    with history.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(["epoch", "train_total", "infonce", "hubert_reg", "mel_residual", "subject_val_score", "subject_val_top3_gain", "subject_val_hubert_gain", "subject_val_active_gain"])

    def save_checkpoint(path: Path, score: float) -> None:
        torch.save(
            {
                "model_kind": "retrieval_first_v61",
                "model_state": model.state_dict(),
                "model_config": vars(model.cfg),
                "stages": list(stages),
                "subject_val": subject_val,
                "subject_test": subject_test,
                "target_kind": "hubert_sequence",
                "hubert_cache": str(hubert_cache),
                "core_cache": str(core_cache),
                "semantic_token_cache": str(token_path) if token_path.exists() else "",
                "lambda_mel_residual": float(args.lambda_mel_residual),
                "retrieval_topk": int(args.retrieval_topk),
                "retrieval_temperature": float(args.retrieval_temperature),
                "selection": "subject_holdout_retrieval_generation",
                "val_selection_score": float(score),
                "train_bank": train_bank,
                "core_target_mean": core_targets.target_mean,
                "core_target_std": core_targets.target_std,
                "speech_core_default_insert_frame": int(getattr(core_targets, "global_core_insert_frame", 0)),
                "speech_core_full_target_steps": int(getattr(core_targets, "full_target_steps", core_targets.T)),
                "speech_core_silence_floor_raw": getattr(core_targets, "silence_floor_raw", None),
            },
            path,
        )

    best = -1e9
    patience = int(train_cfg.get("early_stop_patience", 15))
    stale = 0
    for epoch in range(epochs):
        model.train()
        agg: dict[str, float] = {}
        seen = 0
        pbar = _progress_bar(loader, total=len(loader), desc=f"epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(pbar):
            eeg = batch["eeg"].to(device)
            stage = batch["stage_idx"].to(device)
            valid = batch["eeg_valid_len"].to(device)
            audio = batch["hubert_summary"].to(device)
            target_core = batch["target_seq"].to(device)
            out = model(eeg, stage, valid)
            pred_embed = out["eeg_embed"]
            loss_infonce = _symmetric_infonce(pred_embed, audio, args.temperature)
            loss_hubert = F.smooth_l1_loss(F.normalize(pred_embed, dim=-1), F.normalize(audio, dim=-1))
            var_loss, cov_loss = _vicreg_losses(F.normalize(pred_embed, dim=-1))
            loss = (
                float(args.lambda_trial_infonce) * loss_infonce
                + float(args.lambda_hubert_summary_reg) * loss_hubert
                + float(args.lambda_vicreg_variance) * var_loss
                + float(args.lambda_vicreg_covariance) * cov_loss
            )
            loss_content = F.cross_entropy(out["content_logits"], batch["label_idx"].to(device))
            loss = loss + float(args.lambda_content_ce) * loss_content
            loss_sem = torch.zeros((), device=device)
            loss_ctc = torch.zeros((), device=device)
            if token_targets is not None and "semantic_token_logits" in out and "semantic_token_targets" in batch:
                tok_t = batch["semantic_token_targets"].to(device)
                tok_m = batch["semantic_token_mask"].to(device)
                loss_sem = _semantic_token_ce(out["semantic_token_logits"], tok_t, tok_m, int(token_targets.vocab_size))
                loss = loss + float(args.lambda_semantic_token_ce) * loss_sem
                if float(args.lambda_speech_token_ctc) > 0.0:
                    loss_ctc = _speech_token_ctc(out["semantic_token_logits"], tok_t, tok_m, blank=int(token_targets.vocab_size))
                    loss = loss + float(args.lambda_speech_token_ctc) * loss_ctc
            loss_mel = torch.zeros((), device=device)
            if float(args.lambda_mel_residual) > 0.0:
                prior = _retrieve_prior_torch(
                    pred_embed.detach(),
                    bank_audio_t,
                    bank_core_t,
                    topk=int(args.retrieval_topk),
                    temperature=float(args.retrieval_temperature),
                    bank_ids=bank_ids,
                    query_ids=[str(item) for item in batch["template_id"]],
                )
                pred_core = prior + out["pred_core_delta"]
                loss_mel = F.smooth_l1_loss(pred_core, target_core) + 0.5 * F.smooth_l1_loss(
                    out["pred_core_delta"], target_core - prior
                )
                loss = loss + float(args.lambda_mel_residual) * loss_mel
            opt.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            opt.step()
            b = int(eeg.shape[0])
            seen += b
            vals = {
                "total": loss,
                "infonce": loss_infonce,
                "hubert_reg": loss_hubert,
                "vicreg_var": var_loss,
                "vicreg_cov": cov_loss,
                "semantic_ce": loss_sem,
                "speech_ctc": loss_ctc,
                "content_ce": loss_content,
                "mel_residual": loss_mel,
            }
            for name, value in vals.items():
                agg[name] = agg.get(name, 0.0) + float(value.detach()) * b
            if tqdm is not None and hasattr(pbar, "set_postfix"):
                pbar.set_postfix(total=f"{float(loss.detach()):.3f}", nce=f"{float(loss_infonce.detach()):.3f}")
            if args.max_steps and step + 1 >= args.max_steps:
                break
        sched.step()
        train_metrics = {name: value / max(seen, 1) for name, value in agg.items()}
        subject_val_metrics = _evaluate_retrieval(
            model,
            subject_val_ds,
            train_bank,
            device,
            batch_size,
            residual_scale=float(args.lambda_mel_residual),
            topk=int(args.retrieval_topk),
            retrieval_temperature=float(args.retrieval_temperature),
        )
        score = float(subject_val_metrics["selection_score"])
        print(
            f"epoch {epoch:03d} total={train_metrics['total']:.3f} "
            f"nce={train_metrics.get('infonce', 0.0):.3f} hub={train_metrics.get('hubert_reg', 0.0):.3f} "
            f"mel={train_metrics.get('mel_residual', 0.0):.3f} "
            f"subject_val top3_gain={subject_val_metrics['pred_pair_trial_top3_gain']:+.3f} "
            f"hub_gain={subject_val_metrics['pred_hubert_cos_gain']:+.3f} "
            f"active_gain={subject_val_metrics['pred_over_mean_active_shape_gain']:+.3f} "
            f"select={score:+.3f}"
        )
        with history.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    epoch,
                    train_metrics["total"],
                    train_metrics.get("infonce", 0.0),
                    train_metrics.get("hubert_reg", 0.0),
                    train_metrics.get("mel_residual", 0.0),
                    score,
                    subject_val_metrics["pred_pair_trial_top3_gain"],
                    subject_val_metrics["pred_hubert_cos_gain"],
                    subject_val_metrics["pred_over_mean_active_shape_gain"],
                ]
            )
        write_json(run_dir / "metrics" / "subject_val_latest.json", subject_val_metrics)
        if score > best:
            best = score
            stale = 0
            save_checkpoint(run_dir / "checkpoints" / "best.pt", best)
        else:
            stale += 1
        if args.max_steps:
            break
        if patience > 0 and stale >= patience:
            print(f"[early-stop] no subject_val selection improvement for {patience} epochs (best={best:+.4f}); stopping at epoch {epoch}")
            break

    save_checkpoint(run_dir / "checkpoints" / "last.pt", best)
    best_path = run_dir / "checkpoints" / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[eval] loaded best checkpoint for final metrics: {best_path}")
    final = {
        "selection": {"criterion": "subject_holdout_retrieval_generation", "best_subject_val_score": float(best)},
        "val": _evaluate_retrieval(model, val_ds, train_bank, device, batch_size, residual_scale=float(args.lambda_mel_residual), topk=int(args.retrieval_topk), retrieval_temperature=float(args.retrieval_temperature)),
        "test": _evaluate_retrieval(model, test_ds, train_bank, device, batch_size, residual_scale=float(args.lambda_mel_residual), topk=int(args.retrieval_topk), retrieval_temperature=float(args.retrieval_temperature)),
        "subject_val": _evaluate_retrieval(model, subject_val_ds, train_bank, device, batch_size, residual_scale=float(args.lambda_mel_residual), topk=int(args.retrieval_topk), retrieval_temperature=float(args.retrieval_temperature)),
        "subject_test": _evaluate_retrieval(model, subject_test_ds, train_bank, device, batch_size, residual_scale=float(args.lambda_mel_residual), topk=int(args.retrieval_topk), retrieval_temperature=float(args.retrieval_temperature)),
    }
    write_json(run_dir / "metrics" / "test_metrics.json", final)
    print(json.dumps(final["selection"], indent=2))
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()
