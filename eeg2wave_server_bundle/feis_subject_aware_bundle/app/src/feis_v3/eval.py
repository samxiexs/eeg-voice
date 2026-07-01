from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.feis_v3.data import FEISV3AudioTokenBank


def _to_device(batch: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _masked_token_acc(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, k: int) -> float:
    pred = logits.topk(k=min(k, logits.shape[-1]), dim=-1).indices
    ok = (pred == target.unsqueeze(-1)).any(dim=-1).float() * mask.float()
    return float((ok.sum() / mask.sum().clamp_min(1.0)).detach().cpu())


def _entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    ent = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=-1)
    return float(np.mean(ent / np.log(max(probs.shape[-1], 2))))


def _pairwise_corr_median(x: np.ndarray) -> float:
    if x.shape[0] < 3:
        return 0.0
    corr = np.corrcoef(x)
    tri = corr[np.triu_indices_from(corr, k=1)]
    tri = tri[np.isfinite(tri)]
    if tri.size == 0:
        return 0.0
    return float(np.median(tri))


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    return a @ b.T


def _repeat_metrics(embeds: np.ndarray, subjects: list[str], labels: list[str]) -> dict[str, float | bool]:
    same, shuffled = [], []
    if len(subjects) < 2:
        return {
            "same_subject_label_repeat_consistency": 0.0,
            "shuffled_repeat_consistency": 0.0,
            "repeat_reliability_gate_pass": False,
        }
    norm = embeds / np.maximum(np.linalg.norm(embeds, axis=1, keepdims=True), 1e-8)
    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            sim = float((norm[i] * norm[j]).sum())
            if subjects[i] == subjects[j] and labels[i] == labels[j]:
                same.append(sim)
            else:
                shuffled.append(sim)
    same_mean = float(np.mean(same)) if same else 0.0
    shuf_mean = float(np.mean(shuffled)) if shuffled else 0.0
    return {
        "same_subject_label_repeat_consistency": same_mean,
        "shuffled_repeat_consistency": shuf_mean,
        "repeat_reliability_gate_pass": bool(same and same_mean > shuf_mean),
    }


def _self_recording_metrics(
    pred_hist: np.ndarray,
    audio_indices: list[int],
    labels: list[str],
    token_bank: FEISV3AudioTokenBank,
) -> dict[str, float | bool]:
    bank_hist = token_bank.semantic_histograms()
    top1 = []
    top3 = []
    subjects_per_label = defaultdict(set)
    for subj, label in zip(token_bank.subject_ids, token_bank.labels):
        subjects_per_label[label].add(subj)
    for row_idx, audio_idx in enumerate(audio_indices):
        label = labels[row_idx]
        candidates = [idx for idx, lab in enumerate(token_bank.labels) if lab == label]
        if not candidates:
            continue
        sims = _cosine_matrix(pred_hist[row_idx : row_idx + 1], bank_hist[np.asarray(candidates)])[0]
        order = np.asarray(candidates)[np.argsort(-sims)]
        top1.append(int(order[0] == audio_idx))
        top3.append(int(audio_idx in order[:3]))
    top1_acc = float(np.mean(top1)) if top1 else 0.0
    top3_acc = float(np.mean(top3)) if top3 else 0.0
    denom = float(np.mean([len(subjects_per_label[label]) for label in set(labels)] or [1.0]))
    chance1 = 1.0 / max(denom, 1.0)
    chance3 = min(1.0, 3.0 / max(denom, 1.0))
    return {
        "same_label_audio_variant_top1": top1_acc,
        "same_label_audio_variant_top3": top3_acc,
        "same_label_audio_variant_chance_top1": chance1,
        "same_label_audio_variant_chance_top3": chance3,
        "self_recording_specificity_gate_pass": bool(top1_acc > chance1 and top3_acc >= chance3),
    }


def evaluate_feis_v3(
    model: torch.nn.Module,
    dataset,
    token_bank: FEISV3AudioTokenBank,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    split_name: str = "eval",
    compute_controls: bool = True,
    max_samples: int | None = None,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    semantic_top1_num = semantic_top3_num = semantic_den = 0.0
    codec_top1_num = codec_top3_num = codec_den = 0.0
    prompt_ok = 0
    total = 0
    zero_codec_ok = shuf_codec_ok = 0.0
    label_prior_codec_ok = 0.0
    pred_hist_rows, embed_rows = [], []
    subjects: list[str] = []
    labels: list[str] = []
    audio_indices: list[int] = []
    prompt_probs = []
    with torch.no_grad():
        for raw in loader:
            if max_samples is not None and total >= int(max_samples):
                break
            batch = _to_device(raw, device)
            out = model(
                batch["eeg"],
                stage_idx=batch["stage_idx"],
                eeg_valid_len=batch["eeg_valid_len"],
                channel_cluster_id=batch["channel_cluster_id"],
            )
            sem_mask = batch["semantic_token_mask"].float()
            codec_mask = batch["codec_token_mask"].float()
            semantic_top1_num += _masked_token_acc(out["semantic_logits"], batch["semantic_token_ids"], sem_mask, 1) * float(sem_mask.sum().cpu())
            semantic_top3_num += _masked_token_acc(out["semantic_logits"], batch["semantic_token_ids"], sem_mask, 3) * float(sem_mask.sum().cpu())
            semantic_den += float(sem_mask.sum().cpu())
            codec_top1_num += _masked_token_acc(out["codec_logits"], batch["codec_token_ids"], codec_mask, 1) * float(codec_mask.sum().cpu())
            codec_top3_num += _masked_token_acc(out["codec_logits"], batch["codec_token_ids"], codec_mask, 3) * float(codec_mask.sum().cpu())
            codec_den += float(codec_mask.sum().cpu())
            prompt = out["prompt_logits"].argmax(dim=-1)
            prompt_ok += int((prompt == batch["label_idx"]).sum().detach().cpu())
            total += int(prompt.shape[0])
            sem_probs = out["semantic_logits"].softmax(dim=-1).mean(dim=1).detach().cpu().numpy()
            pred_hist_rows.append(sem_probs)
            embed_rows.append(out["content_embed"].detach().cpu().numpy())
            prompt_probs.append(out["prompt_logits"].softmax(dim=-1).detach().cpu().numpy())
            subjects.extend(list(raw["subject_id"]))
            labels.extend(list(raw["label"]))
            audio_indices.extend([int(x) for x in raw["audio_index"].numpy().tolist()])
            if compute_controls:
                zero_out = model(
                    torch.zeros_like(batch["eeg"]),
                    stage_idx=batch["stage_idx"],
                    eeg_valid_len=batch["eeg_valid_len"],
                    channel_cluster_id=batch["channel_cluster_id"],
                )
                perm = torch.arange(batch["eeg"].shape[0] - 1, -1, -1, device=batch["eeg"].device)
                shuf_out = model(
                    batch["eeg"][perm],
                    stage_idx=batch["stage_idx"],
                    eeg_valid_len=batch["eeg_valid_len"],
                    channel_cluster_id=batch["channel_cluster_id"],
                )
                target = batch["codec_token_ids"]
                zero_codec_ok += float(((zero_out["codec_logits"].argmax(-1) == target).float() * codec_mask).sum().cpu())
                shuf_codec_ok += float(((shuf_out["codec_logits"].argmax(-1) == target).float() * codec_mask).sum().cpu())
                for bi, label in enumerate(raw["label"]):
                    prior = torch.from_numpy(token_bank.label_prior_codec_tokens(label)).to(target.device)
                    label_prior_codec_ok += float(((prior == target[bi]).float() * codec_mask[bi]).sum().cpu())

    pred_hist = np.concatenate(pred_hist_rows, axis=0) if pred_hist_rows else np.zeros((0, token_bank.semantic_vocab_size), dtype=np.float32)
    embeds = np.concatenate(embed_rows, axis=0) if embed_rows else np.zeros((0, 1), dtype=np.float32)
    prompt_prob = np.concatenate(prompt_probs, axis=0) if prompt_probs else np.zeros((0, token_bank.num_labels), dtype=np.float32)
    sem_top1 = semantic_top1_num / max(semantic_den, 1.0)
    sem_top3 = semantic_top3_num / max(semantic_den, 1.0)
    codec_top1 = codec_top1_num / max(codec_den, 1.0)
    codec_top3 = codec_top3_num / max(codec_den, 1.0)
    prompt_acc = prompt_ok / max(total, 1)
    chance_prompt = 1.0 / max(token_bank.num_labels, 1)
    semantic_top3_chance = min(1.0, 3.0 / max(token_bank.semantic_vocab_size, 1))
    codec_chance = 1.0 / max(token_bank.codec_vocab_size, 1)

    bank_hist = token_bank.semantic_histograms()
    train_candidates = [idx for idx, split in enumerate(token_bank.fit_split) if split == "train"]
    if pred_hist.size and train_candidates:
        sims = _cosine_matrix(pred_hist, bank_hist[np.asarray(train_candidates)])
        retrieved = np.asarray(train_candidates)[sims.argmax(axis=1)]
        retrieval_acc = float(np.mean([token_bank.labels[ridx] == labels[i] for i, ridx in enumerate(retrieved)]))
    else:
        retrieval_acc = 0.0
    repeat = _repeat_metrics(embeds, subjects, labels)
    selfrec = _self_recording_metrics(pred_hist, audio_indices, labels, token_bank) if pred_hist.size else {}
    zero_acc = zero_codec_ok / max(codec_den, 1.0)
    shuf_acc = shuf_codec_ok / max(codec_den, 1.0)
    prior_acc = label_prior_codec_ok / max(codec_den, 1.0)
    alignment_gate = bool(
        (sem_top3 - semantic_top3_chance) > 0.02
        and prompt_acc > chance_prompt + 0.03
        and (retrieval_acc - chance_prompt) > 0
        and _entropy(pred_hist) > 0.1
        and _pairwise_corr_median(pred_hist) < 0.75
    )
    generation_gate = bool(
        alignment_gate
        and repeat.get("repeat_reliability_gate_pass", False)
        and codec_top1 > zero_acc
        and codec_top1 > shuf_acc
        and codec_top1 > prior_acc
    )
    stage_values = set(getattr(dataset, "_wanted_stages", lambda: [])())
    is_resting = stage_values == {"resting"}
    metrics: dict[str, Any] = {
        "split": split_name,
        "n_samples": total,
        "prompt_acc": prompt_acc,
        "prompt_chance": chance_prompt,
        "prompt_gate_threshold": chance_prompt + 0.03,
        "semantic_token_top1": sem_top1,
        "semantic_token_top3": sem_top3,
        "semantic_token_top3_prior": semantic_top3_chance,
        "semantic_token_top3_gain_over_prior": sem_top3 - semantic_top3_chance,
        "token_retrieval_cross_subject_top1": retrieval_acc,
        "token_retrieval_cross_subject_gain": retrieval_acc - chance_prompt,
        "pred_token_entropy": _entropy(pred_hist) if pred_hist.size else 0.0,
        "pred_pairwise_corr_median": _pairwise_corr_median(pred_hist) if pred_hist.size else 0.0,
        "codec_token_top1": codec_top1,
        "codec_token_top3": codec_top3,
        "codec_token_chance": codec_chance,
        "zero_eeg_codec_top1": zero_acc,
        "shuffled_eeg_codec_top1": shuf_acc,
        "label_prior_codec_top1": prior_acc,
        "generated_over_zero_codec_margin": codec_top1 - zero_acc,
        "generated_over_shuffled_codec_margin": codec_top1 - shuf_acc,
        "generated_over_labelprior_codec_margin": codec_top1 - prior_acc,
        "alignment_gate_pass": alignment_gate,
        "generation_gate_pass": generation_gate,
        "claim_status": "EEG-to-Speech generation gate passed" if generation_gate and not is_resting else "diagnostic generated codec attempt",
        "retrieval_name": "retrieval_diagnostic",
        "retrieval_is_diagnostic_only": True,
        "resting_negative_control_gate_reported": is_resting,
        "resting_negative_control_pass": bool(is_resting and not alignment_gate and not generation_gate),
        "prompt_confidence_mean": float(prompt_prob.max(axis=1).mean()) if prompt_prob.size else 0.0,
    }
    metrics.update(repeat)
    metrics.update(selfrec)
    return metrics
