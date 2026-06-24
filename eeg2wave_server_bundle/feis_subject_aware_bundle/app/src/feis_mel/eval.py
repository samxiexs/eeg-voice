from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.feis_mel.audio import pearson_flat
from src.feis_mel.losses import _dtw_path


def _dtw_distance(a: np.ndarray, b: np.ndarray, band: int = 10) -> float:
    cost = np.mean(np.abs(a[:, None, :] - b[None, :, :]), axis=-1)
    _, _, path_cost = _dtw_path(cost, band=band)
    return float(path_cost)


def _best_bank_metrics(pred: np.ndarray, bank: np.ndarray, mean_dist: float, band: int = 10, top_k: int = 3) -> tuple[float, float, float, float]:
    naive = np.mean(np.abs(bank - pred.reshape(1, *pred.shape)), axis=(1, 2))
    candidates = np.argsort(naive)[: max(1, min(int(top_k), bank.shape[0]))]
    best_dist = float("inf")
    best_ref = bank[int(candidates[0])]
    for ref_idx in candidates:
        ref = bank[int(ref_idx)]
        dist = _dtw_distance(pred, ref, band=band)
        if dist < best_dist:
            best_dist = dist
            best_ref = ref
    pcc = pearson_flat(pred, best_ref)
    mcd_like = float(np.mean(np.sqrt(np.sum((pred[: min(len(pred), len(best_ref))] - best_ref[: min(len(pred), len(best_ref))]) ** 2, axis=-1))))
    return best_dist, mean_dist, pcc, mcd_like


@torch.no_grad()
def evaluate_feis_mel(model, dataset, targets, device="cpu", batch_size=64, dtw_band: int = 10, dtw_top_k: int = 3) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n = 0
    content_correct = 0
    retrieval_correct = 0
    pccs: list[float] = []
    pred_dists: list[float] = []
    mean_dists: list[float] = []
    mcds: list[float] = []
    proto_raw = targets.raw_banks.mean(axis=1)
    mean_baseline = [
        min(_dtw_distance(targets.global_mean_raw, ref, band=dtw_band) for ref in targets.raw_bank_for_label_id(label_idx))
        for label_idx in range(targets.num_labels)
    ]
    for batch in loader:
        eeg = batch["eeg"].to(device)
        label_idx = batch["label_idx"].to(device)
        out = model(eeg)
        pred_norm = out["pred_mel"].detach().cpu().numpy()
        logits = out["content_logits"]
        content_correct += int((logits.argmax(dim=-1) == label_idx).sum().item())
        for i in range(pred_norm.shape[0]):
            lab = int(batch["label_idx"][i].item())
            pred_raw = targets.denormalize(pred_norm[i])
            pred_dist, mean_dist, pcc, mcd_like = _best_bank_metrics(
                pred_raw,
                targets.raw_bank_for_label_id(lab),
                mean_baseline[lab],
                band=dtw_band,
                top_k=dtw_top_k,
            )
            pred_dists.append(pred_dist)
            mean_dists.append(mean_dist)
            pccs.append(pcc)
            mcds.append(mcd_like)
            label_dist = np.mean(np.abs(proto_raw - pred_raw.reshape(1, *pred_raw.shape)), axis=(1, 2))
            retrieval_correct += int(label_dist.argmin() == lab)
        n += int(eeg.shape[0])
    pred_med = float(np.median(pred_dists)) if pred_dists else float("nan")
    mean_med = float(np.median(mean_dists)) if mean_dists else float("nan")
    return {
        "split": dataset.split,
        "stage": dataset.stage,
        "num_trials": n,
        "content_chance": 1.0 / max(targets.num_labels, 1),
        "content_top1": content_correct / max(n, 1),
        "retrieval_top1": retrieval_correct / max(n, 1),
        "mel_PCC": float(np.nanmean(pccs)) if pccs else float("nan"),
        "DTW_MCD": float(np.nanmedian(mcds)) if mcds else float("nan"),
        "pred_to_label_bank_dtw": pred_med,
        "mean_mel_baseline_dtw": mean_med,
        "pred_beats_mean": bool(pred_med < mean_med) if np.isfinite(pred_med) and np.isfinite(mean_med) else False,
    }
