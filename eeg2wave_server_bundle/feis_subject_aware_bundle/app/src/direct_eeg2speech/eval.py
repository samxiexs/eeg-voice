"""Evaluation for the EEG-only speech reconstruction path."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _pairwise_corr_median(mat: np.ndarray) -> float:
    if mat.shape[0] < 2:
        return float("nan")
    x = mat - mat.mean(axis=1, keepdims=True)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    c = x @ x.T
    iu = np.triu_indices(c.shape[0], k=1)
    return float(np.median(c[iu]))


@torch.no_grad()
def evaluate_direct(model, dataset, targets, device="cpu", batch_size=64) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n = 0
    content_correct = 0
    sum_recon_cos = 0.0
    sum_mean_distance = 0.0
    stage_counts = defaultdict(lambda: [0, 0])
    pred_summ_list: list[np.ndarray] = []
    tgt_summ_list: list[np.ndarray] = []
    voice_list: list[np.ndarray] = []
    subjects: list[str] = []
    std_ratios: list[float] = []

    mean_norm = ((targets.global_mean_raw_seq() - targets.target_mean.reshape(1, -1))
                 / targets.target_std.reshape(1, -1)).astype(np.float32)
    mean_t = torch.from_numpy(mean_norm).to(device)

    for batch in loader:
        eeg = batch["eeg"].to(device)
        stage = batch["stage_idx"].to(device)
        out = model(eeg, stage)
        pred = out["pred_latent"]
        tgt = batch["target_seq"].to(device)
        label = batch["label_idx"].to(device)

        cls = out["content_logits"].argmax(-1)
        content_correct += int((cls == label).sum().item())
        rcos = F.cosine_similarity(pred, tgt, dim=-1).mean(dim=1)
        sum_recon_cos += float(rcos.sum().item())
        dist = torch.sqrt(torch.mean((pred - mean_t.unsqueeze(0)) ** 2, dim=(1, 2)) + 1e-8)
        sum_mean_distance += float(dist.sum().item())

        ps = pred.reshape(-1, pred.shape[-1]).std(dim=0)
        ts = tgt.reshape(-1, tgt.shape[-1]).std(dim=0)
        std_ratios.append(float((ps / ts.clamp_min(1e-6)).median().cpu()))
        pred_summ_list.append(pred.mean(dim=1).cpu().numpy())
        tgt_summ_list.append(tgt.mean(dim=1).cpu().numpy())
        voice_list.append(out["voice_embed"].cpu().numpy())

        for i, stg in enumerate(batch["stage"]):
            stage_counts[stg][0] += int(cls[i].item() == label[i].item())
            stage_counts[stg][1] += 1
            subjects.append(str(batch["subject"][i]))
        n += int(eeg.shape[0])

    pred_summ = np.concatenate(pred_summ_list, 0) if pred_summ_list else np.zeros((0, 1))
    tgt_summ = np.concatenate(tgt_summ_list, 0) if tgt_summ_list else np.zeros((0, 1))
    voices = np.concatenate(voice_list, 0) if voice_list else np.zeros((0, 1))
    voice_gap = float("nan")
    if voices.shape[0] > 2:
        z = voices / (np.linalg.norm(voices, axis=1, keepdims=True) + 1e-8)
        sim = z @ z.T
        same, diff = [], []
        for i in range(len(subjects)):
            for j in range(i + 1, len(subjects)):
                (same if subjects[i] == subjects[j] else diff).append(float(sim[i, j]))
        if same and diff:
            voice_gap = float(np.mean(same) - np.mean(diff))

    return {
        "split": dataset.split,
        "num_trials": n,
        "content_chance": 1.0 / max(dataset.num_labels, 1),
        "content_top1": content_correct / max(n, 1),
        "content_top1_by_stage": {
            k: {"top1": v[0] / max(v[1], 1), "n": v[1]} for k, v in stage_counts.items()
        },
        "latent_recon_cos": sum_recon_cos / max(n, 1),
        "mean_latent_distance": sum_mean_distance / max(n, 1),
        "pred_std_ratio_median": float(np.median(std_ratios)) if std_ratios else float("nan"),
        "pred_pairwise_corr_median": _pairwise_corr_median(pred_summ),
        "target_pairwise_corr_median": _pairwise_corr_median(tgt_summ),
        "speaker_retrieval_same_subject_gap": voice_gap,
    }
