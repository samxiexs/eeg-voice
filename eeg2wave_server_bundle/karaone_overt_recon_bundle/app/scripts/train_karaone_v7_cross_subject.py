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

from src.karaone_recon.cross_subject_v7 import (
    KaraOneEEGFeatureCache,
    KaraOneV7Config,
    KaraOneV7CrossSubject,
    KaraOneV7Dataset,
    SubjectBalancedBatchSampler,
    normalize_audio_embed,
)
from src.karaone_recon.data import KaraOneTrialDataset
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
    parser = argparse.ArgumentParser(description="Train KaraOne v7 cross-subject EEG representation model.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone.yaml"))
    parser.add_argument("--stages", default=None)
    parser.add_argument("--model", choices=["baseline", "moe"], default="baseline")
    parser.add_argument("--core-cache", default="../artifacts/audio_targets/karaone_temporal_elastic_core_v5.npz")
    parser.add_argument("--feature-cache", default="../artifacts/audio_targets/karaone_eeg_features_v7.npz")
    parser.add_argument("--subject-val", default=None)
    parser.add_argument("--subject-test", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-suffix", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--retrieval-topk", type=int, default=3)
    parser.add_argument("--retrieval-temperature", type=float, default=0.05)
    parser.add_argument("--lambda-trial-infonce", type=float, default=1.0)
    parser.add_argument("--lambda-hubert-cos", type=float, default=0.8)
    parser.add_argument("--lambda-vicreg-variance", type=float, default=0.3)
    parser.add_argument("--lambda-vicreg-covariance", type=float, default=0.1)
    parser.add_argument("--lambda-hard-neg", type=float, default=0.2)
    parser.add_argument("--lambda-content-ce", type=float, default=0.05)
    parser.add_argument("--lambda-subject-adv", type=float, default=0.05)
    parser.add_argument("--lambda-mel-residual", type=float, default=0.0)
    parser.add_argument("--lambda-soft-infonce", type=float, default=0.0)
    parser.add_argument("--soft-target-temperature", type=float, default=0.08)
    parser.add_argument("--feature-dropout-prob", type=float, default=0.0)
    parser.add_argument("--feature-noise-std", type=float, default=0.0)
    parser.add_argument("--selection", choices=["v7_subject_generalization", "v8_soft_subject_generalization"], default="v7_subject_generalization")
    return parser.parse_args()


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
    eeg = normalize_audio_embed(eeg)
    audio = normalize_audio_embed(audio)
    logits = eeg @ audio.T / max(float(temperature), 1e-4)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _soft_symmetric_infonce(
    eeg: torch.Tensor,
    audio: torch.Tensor,
    *,
    pred_temperature: float,
    target_temperature: float,
) -> torch.Tensor:
    """Speech-SSL soft-positive contrastive loss.

    KaraOne has repeated prompts and weak cross-subject acoustic consistency. A
    strict diagonal-only InfoNCE forces every EEG trial to retrieve exactly its
    paired waveform, which is often too strong for held-out subjects, especially
    thinking. This target distribution treats speech-SSL-near audio trials as
    partial positives while still making the supervision label-free.
    """
    if eeg.shape[0] < 2:
        return eeg.new_tensor(0.0)
    eeg = normalize_audio_embed(eeg)
    audio = normalize_audio_embed(audio)
    with torch.no_grad():
        target_logits = audio @ audio.T / max(float(target_temperature), 1e-4)
        targets = torch.softmax(target_logits, dim=-1)
    logits_e2a = eeg @ audio.T / max(float(pred_temperature), 1e-4)
    logits_a2e = audio @ eeg.T / max(float(pred_temperature), 1e-4)
    loss_e2a = -(targets * torch.log_softmax(logits_e2a, dim=-1)).sum(dim=-1).mean()
    loss_a2e = -(targets.T * torch.log_softmax(logits_a2e, dim=-1)).sum(dim=-1).mean()
    return 0.5 * (loss_e2a + loss_a2e)


def _hard_negative_loss(
    eeg: torch.Tensor,
    audio: torch.Tensor,
    labels: torch.Tensor,
    subjects: torch.Tensor,
    margin: float = 0.10,
) -> torch.Tensor:
    eeg = normalize_audio_embed(eeg)
    audio = normalize_audio_embed(audio)
    sim = eeg @ audio.T
    pos = sim.diag().unsqueeze(1)
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
    same_label_diff_subject = (labels[:, None] == labels[None, :]) & (subjects[:, None] != subjects[None, :]) & (~eye)
    diff_label_same_subject = (labels[:, None] != labels[None, :]) & (subjects[:, None] == subjects[None, :]) & (~eye)
    losses = []
    for mask in (same_label_diff_subject, diff_label_same_subject):
        if bool(mask.any()):
            neg = sim.masked_fill(~mask, -1e4).max(dim=1).values.unsqueeze(1)
            valid = mask.any(dim=1).float().unsqueeze(1)
            losses.append((F.relu(neg - pos + float(margin)) * valid).sum() / valid.sum().clamp(min=1.0))
    if not losses:
        return torch.zeros((), device=eeg.device, dtype=eeg.dtype)
    return sum(losses) / float(len(losses))


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


def _make_dataset(
    *,
    root: Path,
    core_targets: KaraOneTargets,
    hubert_targets: KaraOneTargets,
    features: KaraOneEEGFeatureCache,
    split: str,
    stages: tuple[str, ...],
    split_protocol: str,
    heldout_subjects: list[str],
    eeg_len: int,
) -> KaraOneV7Dataset:
    base = KaraOneTrialDataset(
        data_root=root,
        targets=core_targets,
        aux_targets=hubert_targets,
        split=split,
        stages=stages,
        split_protocol=split_protocol,
        heldout_subjects=heldout_subjects,
        eeg_len=eeg_len,
    )
    return KaraOneV7Dataset(base, features)


def _collect_bank(dataset: KaraOneV7Dataset) -> dict[str, Any]:
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
        "active_rms": np.asarray([float(row.get("active_rms", torch.tensor(0.08)).item()) for row in rows], dtype=np.float32),
    }


def _retrieve_prior_np(query: np.ndarray, bank_audio: np.ndarray, bank_core: np.ndarray, topk: int, temperature: float) -> np.ndarray:
    q = query / np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-8)
    b = bank_audio / np.linalg.norm(bank_audio, axis=1, keepdims=True).clip(min=1e-8)
    scores = q @ b.T
    k = max(1, min(int(topk), bank_core.shape[0]))
    idx = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
    vals = np.take_along_axis(scores, idx, axis=1)
    weights = np.exp((vals - vals.max(axis=1, keepdims=True)) / max(float(temperature), 1e-4))
    weights = weights / weights.sum(axis=1, keepdims=True).clip(min=1e-8)
    return (bank_core[idx] * weights[..., None, None]).sum(axis=1).astype(np.float32)


def _retrieve_prior_torch(query: torch.Tensor, bank_audio: torch.Tensor, bank_core: torch.Tensor, topk: int, temperature: float) -> torch.Tensor:
    scores = normalize_audio_embed(query) @ normalize_audio_embed(bank_audio).T
    k = max(1, min(int(topk), int(bank_core.shape[0])))
    vals, idx = torch.topk(scores, k=k, dim=-1)
    weights = torch.softmax(vals / max(float(temperature), 1e-4), dim=-1)
    return (bank_core[idx] * weights[..., None, None]).sum(dim=1)


@torch.no_grad()
def _collect_outputs(model: KaraOneV7CrossSubject, dataset: KaraOneV7Dataset, device: str | torch.device, batch_size: int) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    pred, zero, audio, delta, target = [], [], [], [], []
    subject_logits, subject_idx = [], []
    subjects, labels, trials = [], [], []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        feat = batch["eeg_feature"].to(device)
        env = batch["eeg_envelope"].to(device)
        stage = batch["stage_idx"].to(device)
        valid = batch["eeg_valid_len"].to(device)
        out = model(eeg, feat, env, stage, valid, lambda_subject_adv=0.0)
        zero_out = model(torch.zeros_like(eeg), feat * 0.0, env * 0.0, stage, valid, lambda_subject_adv=0.0)
        pred.append(out["eeg_embed"].detach().cpu().numpy())
        zero.append(zero_out["eeg_embed"].detach().cpu().numpy())
        audio.append(batch["hubert_summary"].numpy())
        delta.append(out["pred_core_delta"].detach().cpu().numpy())
        target.append(batch["target_seq"].numpy())
        subject_logits.append(out["subject_logits"].detach().cpu().numpy())
        subject_idx.append(batch["subject_idx"].numpy())
        subjects.extend([str(item) for item in batch["subject"]])
        labels.extend([str(item) for item in batch["label"]])
        trials.extend([int(item) for item in batch["trial_index"]])
    return {
        "pred": np.concatenate(pred, axis=0),
        "zero": np.concatenate(zero, axis=0),
        "audio": np.concatenate(audio, axis=0),
        "delta": np.concatenate(delta, axis=0),
        "target_core": np.concatenate(target, axis=0),
        "subject_logits": np.concatenate(subject_logits, axis=0),
        "subject_idx": np.concatenate(subject_idx, axis=0),
        "subjects": subjects,
        "labels": labels,
        "trials": trials,
    }


def _pair_retrieval(query: np.ndarray, audio: np.ndarray, prefix: str) -> dict[str, float]:
    q = query / np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-8)
    a = audio / np.linalg.norm(audio, axis=1, keepdims=True).clip(min=1e-8)
    scores = q @ a.T
    order = np.argsort(scores, axis=1)[:, ::-1]
    ranks = np.zeros(scores.shape[0], dtype=np.int32)
    for i in range(scores.shape[0]):
        ranks[i] = int(np.where(order[i] == i)[0][0]) + 1
    return {
        f"{prefix}_trial_top1": float(np.mean(ranks <= 1)),
        f"{prefix}_trial_top3": float(np.mean(ranks <= min(3, scores.shape[0]))),
        f"{prefix}_trial_top5": float(np.mean(ranks <= min(5, scores.shape[0]))),
        f"{prefix}_mrr": float(np.mean(1.0 / ranks)),
    }


def _semantic_neighborhood_retrieval(
    query: np.ndarray,
    audio: np.ndarray,
    prefix: str,
    *,
    topk: int = 3,
    positive_frac: float = 0.10,
) -> dict[str, float]:
    """Retrieval against an SSL-derived positive neighborhood, not only the diagonal."""
    q = query / np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-8)
    a = audio / np.linalg.norm(audio, axis=1, keepdims=True).clip(min=1e-8)
    scores = q @ a.T
    audio_sim = a @ a.T
    n = int(scores.shape[0])
    k_pos = max(int(topk), int(math.ceil(n * float(positive_frac))))
    k_pos = min(max(k_pos, 1), n)
    order = np.argsort(scores, axis=1)[:, ::-1]
    pos_order = np.argsort(audio_sim, axis=1)[:, ::-1][:, :k_pos]
    hits1, hits3, hits5, rr = [], [], [], []
    for i in range(n):
        positives = set(int(x) for x in pos_order[i].tolist())
        ranks = [rank + 1 for rank, idx in enumerate(order[i].tolist()) if int(idx) in positives]
        first = ranks[0] if ranks else n + 1
        hits1.append(float(first <= 1))
        hits3.append(float(first <= min(int(topk), n)))
        hits5.append(float(first <= min(5, n)))
        rr.append(1.0 / float(first))
    return {
        f"{prefix}_semantic_top1": float(np.mean(hits1)),
        f"{prefix}_semantic_top3": float(np.mean(hits3)),
        f"{prefix}_semantic_top5": float(np.mean(hits5)),
        f"{prefix}_semantic_mrr": float(np.mean(rr)),
    }


def _bank_label_retrieval(
    query: np.ndarray,
    query_labels: list[str],
    query_subjects: list[str],
    bank_audio: np.ndarray,
    bank_labels: np.ndarray,
    bank_subjects: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    q = query / np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-8)
    b = bank_audio / np.linalg.norm(bank_audio, axis=1, keepdims=True).clip(min=1e-8)
    scores = q @ b.T
    top1 = top3 = cross_top3 = evaluated = 0
    for i, (label, subject) in enumerate(zip(query_labels, query_subjects)):
        order = np.argsort(scores[i])[::-1]
        top = order[:3]
        top1 += int(str(bank_labels[order[0]]) == str(label))
        top3 += int(np.any(bank_labels[top].astype(str) == str(label)))
        cross_mask = bank_subjects.astype(str) != str(subject)
        if bool(cross_mask.any()):
            cross_order = np.argsort(np.where(cross_mask, scores[i], -1e9))[::-1][:3]
            cross_top3 += int(np.any(bank_labels[cross_order].astype(str) == str(label)))
        evaluated += 1
    n = max(1, evaluated)
    return {
        f"{prefix}_bank_label_top1": float(top1 / n),
        f"{prefix}_bank_label_top3": float(top3 / n),
        f"{prefix}_bank_cross_subject_label_top3": float(cross_top3 / n),
    }


def _evaluate_v7(
    model: KaraOneV7CrossSubject,
    dataset: KaraOneV7Dataset,
    train_bank: dict[str, Any],
    device: str | torch.device,
    batch_size: int,
    *,
    residual_scale: float,
    topk: int,
    retrieval_temperature: float,
    subject_leakage_score: float = 0.0,
) -> dict[str, Any]:
    data = _collect_outputs(model, dataset, device, batch_size)
    pred = data["pred"].astype(np.float32)
    zero = data["zero"].astype(np.float32)
    audio = data["audio"].astype(np.float32)
    shuffled = np.roll(pred, 1, axis=0)
    mean_query = np.repeat(train_bank["audio_embed"].mean(axis=0, keepdims=True), pred.shape[0], axis=0).astype(np.float32)
    metrics: dict[str, Any] = {"n": int(pred.shape[0])}
    for query, prefix in ((pred, "pred_pair"), (zero, "zeroeeg_pair"), (shuffled, "shuffled_pair"), (mean_query, "mean_pair")):
        metrics.update(_pair_retrieval(query, audio, prefix))
    for query, prefix in ((pred, "pred"), (zero, "zeroeeg"), (shuffled, "shuffled"), (mean_query, "mean")):
        metrics.update(_semantic_neighborhood_retrieval(query, audio, prefix, topk=3, positive_frac=0.10))
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
    for query, prefix in ((pred, "pred"), (zero, "zeroeeg"), (mean_query, "mean"), (shuffled, "shuffled")):
        metrics.update(
            _bank_label_retrieval(
                query,
                data["labels"],
                data["subjects"],
                train_bank["audio_embed"].astype(np.float32),
                train_bank["labels"].astype(str),
                train_bank["subjects"].astype(str),
                prefix,
            )
        )
    metrics["same_label_cross_subject_gain"] = float(
        metrics["pred_bank_cross_subject_label_top3"]
        - max(metrics["zeroeeg_bank_cross_subject_label_top3"], metrics["mean_bank_cross_subject_label_top3"], metrics["shuffled_bank_cross_subject_label_top3"])
    )
    pred_prior = _retrieve_prior_np(pred, train_bank["audio_embed"], train_bank["core_seq"], topk, retrieval_temperature)
    zero_prior = _retrieve_prior_np(zero, train_bank["audio_embed"], train_bank["core_seq"], topk, retrieval_temperature)
    mean_core = np.repeat(train_bank["core_seq"].mean(axis=0, keepdims=True), pred.shape[0], axis=0).astype(np.float32)
    target_core = data["target_core"].astype(np.float32)
    pred_core = pred_prior + float(residual_scale) * data["delta"].astype(np.float32)
    metrics["pred_active_shape_corr"] = float(_sample_pcc(pred_core, target_core).mean())
    metrics["retrieved_prior_active_shape_corr"] = float(_sample_pcc(pred_prior, target_core).mean())
    metrics["zeroeeg_active_shape_corr"] = float(_sample_pcc(zero_prior, target_core).mean())
    metrics["mean_active_shape_corr"] = float(_sample_pcc(mean_core, target_core).mean())
    metrics["pred_over_zero_active_shape_gain"] = metrics["pred_active_shape_corr"] - metrics["zeroeeg_active_shape_corr"]
    metrics["pred_over_mean_active_shape_gain"] = metrics["pred_active_shape_corr"] - metrics["mean_active_shape_corr"]
    flat = pred_core.reshape(pred_core.shape[0], -1)
    target_flat = target_core.reshape(target_core.shape[0], -1)
    metrics["pred_pairwise_corr_median"] = _corr_median(flat)
    metrics["pred_std_ratio_median"] = float(np.median(flat.std(axis=0) / target_flat.std(axis=0).clip(min=1e-6)))
    metrics["subject_adv_acc"] = float(np.mean(data["subject_logits"].argmax(axis=1) == data["subject_idx"]))
    metrics["subject_leakage_score"] = float(subject_leakage_score)
    metrics["pred_pair_trial_top3_gain"] = float(
        metrics["pred_pair_trial_top3"]
        - max(metrics["zeroeeg_pair_trial_top3"], metrics["mean_pair_trial_top3"], metrics["shuffled_pair_trial_top3"])
    )
    metrics["pred_pair_mrr_gain"] = float(
        metrics["pred_pair_mrr"] - max(metrics["zeroeeg_pair_mrr"], metrics["mean_pair_mrr"], metrics["shuffled_pair_mrr"])
    )
    metrics["pred_semantic_top3_gain"] = float(
        metrics["pred_semantic_top3"] - max(metrics["zeroeeg_semantic_top3"], metrics["mean_semantic_top3"], metrics["shuffled_semantic_top3"])
    )
    metrics["pred_semantic_mrr_gain"] = float(
        metrics["pred_semantic_mrr"] - max(metrics["zeroeeg_semantic_mrr"], metrics["mean_semantic_mrr"], metrics["shuffled_semantic_mrr"])
    )
    metrics["selection_score_v7"] = _selection_score(metrics)
    metrics["selection_score_v8"] = _selection_score_v8(metrics)
    metrics["selection_score"] = metrics["selection_score_v7"]
    return metrics


def _selection_score(metrics: dict[str, Any]) -> float:
    return float(
        metrics.get("pred_pair_trial_top3_gain", 0.0)
        + metrics.get("pred_pair_mrr_gain", 0.0)
        + 0.8 * metrics.get("pred_hubert_cos_gain", 0.0)
        + 0.5 * metrics.get("same_label_cross_subject_gain", 0.0)
        - 0.5 * metrics.get("subject_leakage_score", 0.0)
        - 0.3 * max(0.0, metrics.get("pred_pairwise_corr_median", 0.0) - 0.85)
    )


def _selection_score_v8(metrics: dict[str, Any]) -> float:
    return float(
        1.2 * metrics.get("pred_semantic_top3_gain", 0.0)
        + 1.0 * metrics.get("pred_semantic_mrr_gain", 0.0)
        + 0.6 * metrics.get("pred_pair_trial_top3_gain", 0.0)
        + 0.8 * metrics.get("pred_hubert_cos_gain", 0.0)
        + 0.5 * metrics.get("same_label_cross_subject_gain", 0.0)
        - 0.8 * metrics.get("subject_leakage_score", 0.0)
        - 0.3 * max(0.0, metrics.get("pred_pairwise_corr_median", 0.0) - 0.85)
    )


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
    feature_cache_path = resolve_bundle_path(args.feature_cache, BUNDLE_DIR)
    hubert_targets = KaraOneTargets(hubert_cache, data_root=root)
    core_targets = KaraOneTargets(core_cache, data_root=root)
    features = KaraOneEEGFeatureCache(feature_cache_path)
    heldout = [str(item) for item in cfg["data"].get("heldout_subjects", ["P02", "MM21"])]
    subject_val = str(args.subject_val or heldout[0])
    subject_test = str(args.subject_test or (heldout[1] if len(heldout) > 1 else heldout[0]))
    heldout_pair = sorted(set([subject_val, subject_test]))
    eeg_len = int(cfg["data"].get("eeg_len", 1280))
    train_ds = _make_dataset(root=root, core_targets=core_targets, hubert_targets=hubert_targets, features=features, split="train", stages=stages, split_protocol="subject_holdout", heldout_subjects=heldout_pair, eeg_len=eeg_len)
    val_ds = _make_dataset(root=root, core_targets=core_targets, hubert_targets=hubert_targets, features=features, split="val", stages=stages, split_protocol=str(cfg["data"].get("split_protocol", "trial")), heldout_subjects=heldout_pair, eeg_len=eeg_len)
    test_ds = _make_dataset(root=root, core_targets=core_targets, hubert_targets=hubert_targets, features=features, split="test", stages=stages, split_protocol=str(cfg["data"].get("split_protocol", "trial")), heldout_subjects=heldout_pair, eeg_len=eeg_len)
    subject_val_ds = _make_dataset(root=root, core_targets=core_targets, hubert_targets=hubert_targets, features=features, split="subject_test", stages=stages, split_protocol="subject_holdout", heldout_subjects=[subject_val], eeg_len=eeg_len)
    subject_test_ds = _make_dataset(root=root, core_targets=core_targets, hubert_targets=hubert_targets, features=features, split="subject_test", stages=stages, split_protocol="subject_holdout", heldout_subjects=[subject_test], eeg_len=eeg_len)
    run_family = "karaone_v8_soft_align" if str(args.selection) == "v8_soft_subject_generalization" else "karaone_v7_cross_subject"
    run = f"{run_family}_{args.model}_{'_'.join(stages)}_{args.run_suffix or 'v7'}"
    run_dir = resolve_bundle_path(cfg["output"]["root"], BUNDLE_DIR) / run
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "metrics")
    train_bank = _collect_bank(train_ds)
    audit = {
        "stages": list(stages),
        "subject_val": subject_val,
        "subject_test": subject_test,
        "splits": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds), "subject_val": len(subject_val_ds), "subject_test": len(subject_test_ds)},
        "feature_cache": {"path": str(feature_cache_path), "feature_dim": int(features.feature_dim), "envelope_steps": int(features.envelope_steps)},
        "train_bank": {"n": int(train_bank["audio_embed"].shape[0]), "audio_embed_dim": int(train_bank["audio_embed"].shape[1]), "core_shape": list(train_bank["core_seq"].shape), "core_pairwise_corr_median": _corr_median(train_bank["core_seq"])},
    }
    write_json(run_dir / "metrics" / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.audit_only:
        print(f"[done] audit only: {run_dir}")
        return
    model = KaraOneV7CrossSubject(
        KaraOneV7Config(
            n_channels_eeg=int(model_cfg.get("n_channels_eeg", 62)),
            d_model=128,
            cond_dim=32,
            num_labels=train_ds.num_labels,
            num_subjects=train_ds.num_subjects,
            num_stages=train_ds.num_stages,
            core_steps=core_targets.T,
            core_dim=core_targets.D,
            feature_dim=features.feature_dim,
            envelope_steps=features.envelope_steps,
            audio_embed_dim=hubert_targets.D,
        )
    ).to(device)
    epochs = int(args.epochs or train_cfg.get("epochs", 30))
    batch_size = int(train_cfg.get("batch_size", 48))
    batch_sampler = SubjectBalancedBatchSampler(
        train_ds,
        batch_size=batch_size,
        subjects_per_batch=min(6, max(2, len({e.subject for e in train_ds.entries}))),
        seed=int(train_cfg.get("seed", 7)),
    )
    loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=int(train_cfg.get("num_workers", 0)))
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 3e-4)), weight_decay=float(train_cfg.get("weight_decay", 1e-3)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    bank_audio_t = torch.from_numpy(train_bank["audio_embed"]).to(device).float()
    bank_core_t = torch.from_numpy(train_bank["core_seq"]).to(device).float()
    history = run_dir / "metrics" / "history.csv"
    with history.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow([
            "epoch",
            "train_total",
            "infonce",
            "soft",
            "hubert_cos",
            "hard_neg",
            "subject_val_score",
            "subject_val_top3_gain",
            "subject_val_semantic_top3_gain",
            "subject_val_hubert_gain",
            "same_label_gain",
            "leakage",
        ])

    def save_checkpoint(path: Path, score: float, metrics: dict[str, Any], gate_passed: bool) -> None:
        torch.save(
            {
                "model_kind": "cross_subject_v8_soft_align" if str(args.selection) == "v8_soft_subject_generalization" else "cross_subject_v7",
                "model_state": model.state_dict(),
                "model_config": vars(model.cfg),
                "stages": list(stages),
                "subject_val": subject_val,
                "subject_test": subject_test,
                "hubert_cache": str(hubert_cache),
                "core_cache": str(core_cache),
                "feature_cache": str(feature_cache_path),
                "lambda_mel_residual": float(args.lambda_mel_residual),
                "retrieval_topk": int(args.retrieval_topk),
                "retrieval_temperature": float(args.retrieval_temperature),
                "selection": str(args.selection),
                "val_selection_score": float(score),
                "gate_passed": bool(gate_passed),
                "gate_metrics": metrics,
                "train_bank": train_bank,
                "core_target_mean": core_targets.target_mean,
                "core_target_std": core_targets.target_std,
                "speech_core_default_insert_frame": int(getattr(core_targets, "global_core_insert_frame", 0)),
                "speech_core_full_target_steps": int(getattr(core_targets, "full_target_steps", core_targets.T)),
                "speech_core_silence_floor_raw": getattr(core_targets, "silence_floor_raw", None),
            },
            path,
        )

    best, stale = -1e9, 0
    patience = int(train_cfg.get("early_stop_patience", 15))
    for epoch in range(epochs):
        model.train()
        agg: dict[str, float] = {}
        seen = 0
        # Schedule GRL in [0, 1]. The loss multiplier below supplies
        # lambda_subject_adv; multiplying both made v7's encoder-side
        # adversarial signal too weak and allowed high subject leakage.
        grl = 2.0 / (1.0 + math.exp(-10.0 * epoch / max(epochs - 1, 1))) - 1.0
        pbar = _progress_bar(loader, total=len(loader), desc=f"epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(pbar):
            eeg = batch["eeg"].to(device)
            feat = batch["eeg_feature"].to(device)
            env = batch["eeg_envelope"].to(device)
            stage = batch["stage_idx"].to(device)
            valid = batch["eeg_valid_len"].to(device)
            label_idx = batch["label_idx"].to(device)
            subject_idx = batch["subject_idx"].to(device)
            audio = batch["hubert_summary"].to(device)
            target_core = batch["target_seq"].to(device)
            if float(args.feature_dropout_prob) > 0.0:
                keep = (torch.rand(feat.shape[0], 1, device=device) >= float(args.feature_dropout_prob)).float()
                feat = feat * keep
                env = env * keep
            if float(args.feature_noise_std) > 0.0:
                feat = feat + torch.randn_like(feat) * float(args.feature_noise_std)
                env = env + torch.randn_like(env) * float(args.feature_noise_std)
            out = model(eeg, feat, env, stage, valid, lambda_subject_adv=grl)
            pred = out["eeg_embed"]
            loss_infonce = _symmetric_infonce(pred, audio, args.temperature)
            loss_soft = _soft_symmetric_infonce(
                pred,
                audio,
                pred_temperature=args.temperature,
                target_temperature=args.soft_target_temperature,
            )
            loss_hubert = 1.0 - F.cosine_similarity(normalize_audio_embed(pred), normalize_audio_embed(audio), dim=-1).mean()
            var_loss, cov_loss = _vicreg_losses(normalize_audio_embed(pred))
            loss_hard = _hard_negative_loss(pred, audio, label_idx, subject_idx)
            loss_content = F.cross_entropy(out["content_logits"], label_idx)
            loss_subject = F.cross_entropy(out["subject_logits"], subject_idx)
            loss_mel = torch.zeros((), device=device)
            if float(args.lambda_mel_residual) > 0.0:
                prior = _retrieve_prior_torch(pred.detach(), bank_audio_t, bank_core_t, int(args.retrieval_topk), float(args.retrieval_temperature))
                pred_core = prior + out["pred_core_delta"]
                loss_mel = F.smooth_l1_loss(pred_core, target_core)
            loss = (
                float(args.lambda_trial_infonce) * loss_infonce
                + float(args.lambda_soft_infonce) * loss_soft
                + float(args.lambda_hubert_cos) * loss_hubert
                + float(args.lambda_vicreg_variance) * var_loss
                + float(args.lambda_vicreg_covariance) * cov_loss
                + float(args.lambda_hard_neg) * loss_hard
                + float(args.lambda_content_ce) * loss_content
                + float(args.lambda_subject_adv) * loss_subject
                + float(args.lambda_mel_residual) * loss_mel
            )
            opt.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            opt.step()
            b = int(eeg.shape[0])
            seen += b
            vals = {"total": loss, "infonce": loss_infonce, "soft": loss_soft, "hubert_cos": loss_hubert, "hard_neg": loss_hard, "subject": loss_subject, "mel": loss_mel}
            for name, value in vals.items():
                agg[name] = agg.get(name, 0.0) + float(value.detach()) * b
            if tqdm is not None and hasattr(pbar, "set_postfix"):
                pbar.set_postfix(total=f"{float(loss.detach()):.3f}", nce=f"{float(loss_infonce.detach()):.3f}", hub=f"{float(loss_hubert.detach()):.3f}")
            if args.max_steps and step + 1 >= args.max_steps:
                break
        sched.step()
        train_metrics = {k: v / max(seen, 1) for k, v in agg.items()}
        leakage_eval = _evaluate_v7(model, val_ds, train_bank, device, batch_size, residual_scale=float(args.lambda_mel_residual), topk=int(args.retrieval_topk), retrieval_temperature=float(args.retrieval_temperature))
        leakage_score = max(0.0, float(leakage_eval.get("subject_adv_acc", 0.0)) - 1.0 / max(train_ds.num_subjects, 1))
        subject_val_metrics = _evaluate_v7(model, subject_val_ds, train_bank, device, batch_size, residual_scale=float(args.lambda_mel_residual), topk=int(args.retrieval_topk), retrieval_temperature=float(args.retrieval_temperature), subject_leakage_score=leakage_score)
        if str(args.selection) == "v8_soft_subject_generalization":
            score = float(subject_val_metrics["selection_score_v8"])
            gate_passed = bool(
                subject_val_metrics.get("pred_semantic_top3_gain", 0.0) > 0.0
                and subject_val_metrics.get("pred_hubert_cos_gain", 0.0) > 0.0
            )
        else:
            score = float(subject_val_metrics["selection_score_v7"])
            gate_passed = bool(
                subject_val_metrics.get("pred_pair_trial_top3_gain", 0.0) > 0.0
                and subject_val_metrics.get("pred_hubert_cos_gain", 0.0) > 0.0
            )
        print(
            f"epoch {epoch:03d} total={train_metrics['total']:.3f} nce={train_metrics.get('infonce',0.0):.3f} "
            f"hub={train_metrics.get('hubert_cos',0.0):.3f} hard={train_metrics.get('hard_neg',0.0):.3f} "
            f"subject_val top3_gain={subject_val_metrics['pred_pair_trial_top3_gain']:+.3f} "
            f"sem_top3={subject_val_metrics.get('pred_semantic_top3_gain',0.0):+.3f} "
            f"hub_gain={subject_val_metrics['pred_hubert_cos_gain']:+.3f} "
            f"same_label={subject_val_metrics['same_label_cross_subject_gain']:+.3f} leakage={leakage_score:.3f} "
            f"gate={int(gate_passed)} select={score:+.3f}"
        )
        with history.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow([
                epoch,
                train_metrics["total"],
                train_metrics.get("infonce", 0.0),
                train_metrics.get("soft", 0.0),
                train_metrics.get("hubert_cos", 0.0),
                train_metrics.get("hard_neg", 0.0),
                score,
                subject_val_metrics["pred_pair_trial_top3_gain"],
                subject_val_metrics.get("pred_semantic_top3_gain", 0.0),
                subject_val_metrics["pred_hubert_cos_gain"],
                subject_val_metrics["same_label_cross_subject_gain"],
                leakage_score,
            ])
        write_json(run_dir / "metrics" / "subject_val_latest.json", subject_val_metrics)
        if score > best:
            best, stale = score, 0
            save_checkpoint(run_dir / "checkpoints" / "best.pt", best, subject_val_metrics, gate_passed)
        else:
            stale += 1
        if args.max_steps:
            break
        if patience > 0 and stale >= patience:
            print(f"[early-stop] no subject_val selection improvement for {patience} epochs (best={best:+.4f}); stopping at epoch {epoch}")
            break
    save_checkpoint(run_dir / "checkpoints" / "last.pt", best, subject_val_metrics, gate_passed)
    best_path = run_dir / "checkpoints" / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[eval] loaded best checkpoint for final metrics: {best_path}")
    leakage_eval = _evaluate_v7(model, val_ds, train_bank, device, batch_size, residual_scale=float(args.lambda_mel_residual), topk=int(args.retrieval_topk), retrieval_temperature=float(args.retrieval_temperature))
    leakage_score = max(0.0, float(leakage_eval.get("subject_adv_acc", 0.0)) - 1.0 / max(train_ds.num_subjects, 1))
    final = {
        "selection": {"criterion": str(args.selection), "best_subject_val_score": float(best)},
        "val": leakage_eval,
        "test": _evaluate_v7(model, test_ds, train_bank, device, batch_size, residual_scale=float(args.lambda_mel_residual), topk=int(args.retrieval_topk), retrieval_temperature=float(args.retrieval_temperature), subject_leakage_score=leakage_score),
        "subject_val": _evaluate_v7(model, subject_val_ds, train_bank, device, batch_size, residual_scale=float(args.lambda_mel_residual), topk=int(args.retrieval_topk), retrieval_temperature=float(args.retrieval_temperature), subject_leakage_score=leakage_score),
        "subject_test": _evaluate_v7(model, subject_test_ds, train_bank, device, batch_size, residual_scale=float(args.lambda_mel_residual), topk=int(args.retrieval_topk), retrieval_temperature=float(args.retrieval_temperature), subject_leakage_score=leakage_score),
    }
    write_json(run_dir / "metrics" / "test_metrics.json", final)
    print(json.dumps(final["selection"], indent=2))
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()
