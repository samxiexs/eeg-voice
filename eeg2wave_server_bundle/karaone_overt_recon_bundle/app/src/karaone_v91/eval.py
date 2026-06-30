from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.karaone_v9.eval import compute_v9_metrics, normalize_rows, outputs_to_bank, retrieval_metrics
from src.utils import ensure_dir


@torch.no_grad()
def collect_v91_outputs(model: torch.nn.Module, dataset, *, device: str | torch.device = "cpu", batch_size: int = 32) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    model.eval()
    pred, zero, target = [], [], []
    prompt_logits, subject_logits = [], []
    label_idx, subject_idx = [], []
    eeg_cluster, speech_cluster, cross_cluster = [], [], []
    channel_gate, channel_assign = [], []
    labels, subjects, stages, trials = [], [], [], []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        stage = batch["stage_idx"].to(device)
        valid = batch["eeg_valid_len"].to(device)
        out = model(eeg, stage, valid, mask_ratio=0.0, lambda_subject_adv=0.0)
        zero_out = model(torch.zeros_like(eeg), stage, valid, mask_ratio=0.0, lambda_subject_adv=0.0)
        pred.append(out["pred_semantic_summary"].detach().cpu().numpy())
        zero.append(zero_out["pred_semantic_summary"].detach().cpu().numpy())
        target.append(batch["semantic_summary"].numpy())
        prompt_logits.append(out["prompt_logits"].detach().cpu().numpy())
        subject_logits.append(out["subject_logits"].detach().cpu().numpy())
        label_idx.append(batch["label_idx"].numpy())
        subject_idx.append(batch["subject_idx"].numpy())
        eeg_cluster.append(batch.get("eeg_cluster_id", torch.zeros_like(batch["label_idx"])).numpy())
        speech_cluster.append(batch.get("speech_cluster_id", torch.zeros_like(batch["label_idx"])).numpy())
        cross_cluster.append(batch.get("cross_modal_cluster_id", torch.zeros_like(batch["label_idx"])).numpy())
        channel_gate.append(out["channel_gate"].detach().cpu().numpy())
        channel_assign.append(out["channel_assign"].detach().cpu().numpy())
        labels.extend([str(item) for item in batch["label"]])
        subjects.extend([str(item) for item in batch["subject"]])
        stages.extend([str(item) for item in batch["stage"]])
        trials.extend([int(item) for item in batch["trial_index"]])
    return {
        "pred": np.concatenate(pred, axis=0).astype(np.float32),
        "zero": np.concatenate(zero, axis=0).astype(np.float32),
        "target": np.concatenate(target, axis=0).astype(np.float32),
        "prompt_logits": np.concatenate(prompt_logits, axis=0).astype(np.float32),
        "subject_logits": np.concatenate(subject_logits, axis=0).astype(np.float32),
        "label_idx": np.concatenate(label_idx, axis=0).astype(np.int64),
        "subject_idx": np.concatenate(subject_idx, axis=0).astype(np.int64),
        "eeg_cluster_id": np.concatenate(eeg_cluster, axis=0).astype(np.int64),
        "speech_cluster_id": np.concatenate(speech_cluster, axis=0).astype(np.int64),
        "cross_modal_cluster_id": np.concatenate(cross_cluster, axis=0).astype(np.int64),
        "channel_gate": np.concatenate(channel_gate, axis=0).astype(np.float32),
        "channel_assign": np.concatenate(channel_assign, axis=0).astype(np.float32),
        "labels": labels,
        "subjects": subjects,
        "stages": stages,
        "trials": trials,
    }


def compute_v91_metrics(
    outputs: dict[str, Any],
    *,
    train_bank: dict[str, Any] | None = None,
    prefix: str = "",
) -> dict[str, float | bool]:
    metrics = compute_v9_metrics(outputs, train_bank=train_bank, prefix="")
    speech_clusters = np.asarray(outputs.get("speech_cluster_id", np.zeros(len(outputs["labels"]))), dtype=np.int64)
    eeg_clusters = np.asarray(outputs.get("eeg_cluster_id", np.zeros(len(outputs["labels"]))), dtype=np.int64)
    if train_bank and "speech_cluster_id" in train_bank:
        bank_clusters = np.asarray(train_bank["speech_cluster_id"], dtype=np.int64)
        cluster_retrieval = cluster_retrieval_metrics(outputs["pred"], train_bank["target"], speech_clusters, bank_clusters)
    else:
        cluster_retrieval = cluster_retrieval_metrics(outputs["pred"], outputs["target"], speech_clusters, speech_clusters)
    metrics.update(cluster_retrieval)
    metrics["eeg_cluster_label_purity"] = label_purity(eeg_clusters, np.asarray(outputs["labels"]).astype(str))
    metrics["speech_cluster_label_purity"] = label_purity(speech_clusters, np.asarray(outputs["labels"]).astype(str))
    metrics["cluster_subject_leakage_proxy"] = label_purity(eeg_clusters, np.asarray(outputs["subjects"]).astype(str))
    if "channel_gate" in outputs:
        gate = np.asarray(outputs["channel_gate"], dtype=np.float32)
        norm = gate / np.maximum(gate.sum(axis=1, keepdims=True), 1e-8)
        entropy = -np.sum(norm * np.log(np.maximum(norm, 1e-8)), axis=1) / np.log(max(gate.shape[1], 2))
        metrics["channel_gate_entropy_mean"] = float(np.mean(entropy))
        metrics["channel_gate_active_ratio"] = float(np.mean(gate > 1e-4))
        metrics["channel_gate_top16_mass"] = float(np.mean(np.sort(gate, axis=1)[:, -min(16, gate.shape[1]) :].sum(axis=1) / np.maximum(gate.sum(axis=1), 1e-8)))
    metrics["v91_research_gate_pass"] = bool(
        metrics.get("semantic_over_zero_gain", 0.0) > 0.01
        and metrics.get("semantic_over_mean_gain", 0.0) > 0.0
        and metrics.get("semantic_top3_gain_over_mean", 0.0) > 0.02
        and metrics.get("same_label_cross_subject_gain", -1.0) >= 0.0
        and metrics.get("prompt_acc", 0.0) >= 0.13
        and 0.7 <= metrics.get("pred_std_ratio_median", 0.0) <= 1.5
        and metrics.get("pred_pairwise_corr_median", 1.0) < 0.75
        and metrics.get("channel_gate_entropy_mean", 0.0) > 0.20
    )
    if prefix:
        return {f"{prefix}_{key}": value for key, value in metrics.items()}
    return metrics


def outputs_to_v91_bank(outputs: dict[str, Any]) -> dict[str, Any]:
    bank = outputs_to_bank(outputs)
    bank["speech_cluster_id"] = np.asarray(outputs.get("speech_cluster_id", []), dtype=np.int64)
    bank["eeg_cluster_id"] = np.asarray(outputs.get("eeg_cluster_id", []), dtype=np.int64)
    return bank


def cluster_retrieval_metrics(query: np.ndarray, bank: np.ndarray, clusters: np.ndarray, bank_clusters: np.ndarray) -> dict[str, float]:
    if query.shape[0] == 0 or bank.shape[0] == 0:
        return {"cluster_top1": 0.0, "cluster_top3": 0.0, "cluster_mrr": 0.0}
    return retrieval_metrics(
        query,
        bank,
        labels=clusters.astype(str),
        label_bank=bank_clusters.astype(str),
        prefix="cluster",
    )


def label_purity(cluster_ids: np.ndarray, labels: np.ndarray) -> float:
    if cluster_ids.size == 0:
        return 0.0
    total = 0
    majority = 0
    for cluster in np.unique(cluster_ids):
        mask = cluster_ids == cluster
        values, counts = np.unique(labels[mask], return_counts=True)
        if values.size:
            total += int(mask.sum())
            majority += int(counts.max())
    return float(majority / max(total, 1))


def write_channel_reports(out_dir: str | Path, outputs: dict[str, Any], channel_names: list[str]) -> dict[str, str]:
    out_dir = ensure_dir(out_dir)
    gate = np.asarray(outputs["channel_gate"], dtype=np.float32)
    if gate.ndim != 2:
        raise ValueError(f"channel_gate must be [N,C], got {tuple(gate.shape)}")
    names = channel_names[: gate.shape[1]]
    if len(names) < gate.shape[1]:
        names.extend([f"Ch{idx + 1:03d}" for idx in range(len(names), gate.shape[1])])
    paths = {
        "summary": str(out_dir / "channel_gate_summary.csv"),
        "stage": str(out_dir / "channel_importance_by_stage.csv"),
        "label": str(out_dir / "channel_importance_by_label.csv"),
        "cluster": str(out_dir / "channel_importance_by_cluster.csv"),
        "report": str(out_dir / "top_channels_report.md"),
    }
    write_summary_csv(paths["summary"], gate, names)
    write_group_csv(paths["stage"], gate, np.asarray(outputs.get("stages", ["all"] * gate.shape[0])).astype(str), names, "stage")
    write_group_csv(paths["label"], gate, np.asarray(outputs.get("labels", ["all"] * gate.shape[0])).astype(str), names, "label")
    write_group_csv(paths["cluster"], gate, np.asarray(outputs.get("speech_cluster_id", np.zeros(gate.shape[0]))).astype(str), names, "speech_cluster")
    write_top_report(paths["report"], gate, names, outputs)
    return paths


def write_summary_csv(path: str | Path, gate: np.ndarray, channel_names: list[str]) -> None:
    mean = gate.mean(axis=0)
    std = gate.std(axis=0)
    active = (gate > 1e-4).mean(axis=0)
    order = np.argsort(mean)[::-1]
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(order) + 1)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel", "rank", "mean_gate", "std_gate", "active_rate"])
        for idx, name in enumerate(channel_names):
            writer.writerow([name, int(rank[idx]), float(mean[idx]), float(std[idx]), float(active[idx])])


def write_group_csv(path: str | Path, gate: np.ndarray, groups: np.ndarray, channel_names: list[str], group_name: str) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([group_name, "channel", "mean_gate", "active_rate", "n"])
        for group in sorted(set(groups.tolist())):
            mask = groups == group
            if not bool(mask.any()):
                continue
            mean = gate[mask].mean(axis=0)
            active = (gate[mask] > 1e-4).mean(axis=0)
            for idx, name in enumerate(channel_names):
                writer.writerow([group, name, float(mean[idx]), float(active[idx]), int(mask.sum())])


def write_top_report(path: str | Path, gate: np.ndarray, channel_names: list[str], outputs: dict[str, Any]) -> None:
    mean = gate.mean(axis=0)
    order = np.argsort(mean)[::-1]
    top = order[: min(16, len(order))]
    norm_gate = gate / np.maximum(gate.sum(axis=1, keepdims=True), 1e-8)
    entropy = -np.sum(norm_gate * np.log(np.maximum(norm_gate, 1e-8)), axis=1) / np.log(max(gate.shape[1], 2))
    lines = [
        "# KaraOne v9.1 Channel-MoE Top Channels",
        "",
        f"- samples: {gate.shape[0]}",
        f"- channels: {gate.shape[1]}",
        f"- mean gate entropy: {float(np.mean(entropy)):.4f}",
        f"- active channel ratio: {float(np.mean(gate > 1e-4)):.4f}",
        "",
        "## Top Channels",
        "",
    ]
    for rank, idx in enumerate(top, start=1):
        lines.append(f"{rank}. {channel_names[int(idx)]}: mean_gate={float(mean[int(idx)]):.6f}")
    lines.append("")
    lines.append("Note: these are gate-based importance diagnostics. Use leave-channel-out or permutation runs before making neurophysiology claims.")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
