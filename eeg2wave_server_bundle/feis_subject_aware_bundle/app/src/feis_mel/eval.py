from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.feis_mel.audio import pearson_flat
from src.feis_mel.losses import _dtw_path

# Min PCC gain over the STRICT controls (zero-EEG / shuffled-EEG) before the
# reconstruction counts as "using EEG". The weak label-prior baseline is NOT used for
# the verdict (a generic smooth mel beats it even from a zero input).
EEG_INFO_PCC_EPS = 0.01


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


def _bank_metrics_for(pred_raw, lab, targets, proto_raw, mean_baseline, dtw_band, dtw_top_k):
    """(dtw_dist, pcc, mcd, retrieval_hit) of one prediction against label `lab`'s bank."""
    dist, _, pcc, mcd = _best_bank_metrics(
        pred_raw, targets.raw_bank_for_label_id(lab), mean_baseline[lab], band=dtw_band, top_k=dtw_top_k
    )
    label_dist = np.mean(np.abs(proto_raw - pred_raw.reshape(1, *pred_raw.shape)), axis=(1, 2))
    return dist, pcc, mcd, int(label_dist.argmin() == lab)


def _agg(preds_and_labels, targets, proto_raw, mean_baseline, dtw_band, dtw_top_k, label_prior_cache=None):
    """Aggregate bank metrics over a list of (pred_raw, label) pairs.

    `label_prior_cache`: optional dict caching per-label results (used by the
    label-prior baseline, where the 'prediction' is identical for a given label)."""
    dists, pccs, mcds, retr = [], [], [], 0
    for pred_raw, lab in preds_and_labels:
        if label_prior_cache is not None and lab in label_prior_cache:
            d, p, m, r = label_prior_cache[lab]
        else:
            d, p, m, r = _bank_metrics_for(pred_raw, lab, targets, proto_raw, mean_baseline, dtw_band, dtw_top_k)
            if label_prior_cache is not None:
                label_prior_cache[lab] = (d, p, m, r)
        dists.append(d); pccs.append(p); mcds.append(m); retr += r
    n = max(len(preds_and_labels), 1)
    return {
        "pcc": float(np.nanmean(pccs)) if pccs else float("nan"),
        "dtw": float(np.median(dists)) if dists else float("nan"),
        "mcd": float(np.nanmedian(mcds)) if mcds else float("nan"),
        "retrieval_top1": retr / n,
    }


@torch.no_grad()
def evaluate_feis_mel(model, dataset, targets, device="cpu", batch_size=64, dtw_band: int = 10, dtw_top_k: int = 3, controls: bool = True) -> dict:
    """Evaluate the EEG-only mel model.

    Beyond the (gameable) PCC/DTW-vs-true-label-bank numbers, the honest question is
    whether the EEG *contributes anything beyond knowing the label*. With `controls=True`
    we add three baselines and report the gain of the real model over each:
      * zero-EEG   — model(0): an EEG-independent prior (a single constant mel).
      * label-prior — the label's bank-mean mel (uses the label, zero EEG).
      * shuffled-EEG — each prediction scored against a *mismatched* label's bank.
    If `pred_over_labelprior_pcc_gain` ~ 0 and content/retrieval ~ chance, the pipeline
    is reproducing a class average, not decoding EEG (the resting~=speaking finding).
    Controls add DTW cost, so per-epoch val passes controls=False; test runs the full suite.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    proto_raw = targets.raw_banks.mean(axis=1)  # [num_labels, T, D] per-label bank mean
    mean_baseline = [
        min(_dtw_distance(targets.global_mean_raw, ref, band=dtw_band) for ref in targets.raw_bank_for_label_id(label_idx))
        for label_idx in range(targets.num_labels)
    ]

    preds_raw: list[np.ndarray] = []
    labels: list[int] = []
    content_correct = 0
    zero_pred_raw = None
    zero_content_class = None
    for batch in loader:
        eeg = batch["eeg"].to(device)
        label_idx = batch["label_idx"].to(device)
        out = model(eeg)
        pred_norm = out["pred_mel"].detach().cpu().numpy()
        content_correct += int((out["content_logits"].argmax(dim=-1) == label_idx).sum().item())
        for i in range(pred_norm.shape[0]):
            preds_raw.append(targets.denormalize(pred_norm[i]))
            labels.append(int(batch["label_idx"][i].item()))
        if controls and zero_pred_raw is None:
            zout = model(torch.zeros_like(eeg))
            zero_pred_raw = targets.denormalize(zout["pred_mel"][0].detach().cpu().numpy())
            zero_content_class = int(zout["content_logits"][0].argmax().item())
    n = len(labels)
    chance = 1.0 / max(targets.num_labels, 1)

    pred = _agg(list(zip(preds_raw, labels)), targets, proto_raw, mean_baseline, dtw_band, dtw_top_k)
    mean_med = float(np.median([mean_baseline[lab] for lab in labels])) if labels else float("nan")
    result = {
        "split": dataset.split,
        "stage": dataset.stage,
        "num_trials": n,
        "content_chance": chance,
        "content_top1": content_correct / max(n, 1),
        "retrieval_top1": pred["retrieval_top1"],
        "mel_PCC": pred["pcc"],
        "DTW_MCD": pred["mcd"],
        "pred_to_label_bank_dtw": pred["dtw"],
        "mean_mel_baseline_dtw": mean_med,
        "pred_beats_mean": bool(pred["dtw"] < mean_med) if np.isfinite(pred["dtw"]) and np.isfinite(mean_med) else False,
    }
    if not controls or n == 0:
        return result

    # --- honest controls: does EEG add anything beyond the label prior? ---
    zero = _agg([(zero_pred_raw, lab) for lab in labels], targets, proto_raw, mean_baseline, dtw_band, dtw_top_k)
    prior = _agg([(proto_raw[lab], lab) for lab in labels], targets, proto_raw, mean_baseline, dtw_band, dtw_top_k, label_prior_cache={})
    shuffled_pairs = [(preds_raw[i], labels[(i + 1) % n]) for i in range(n)]
    shuffled = _agg(shuffled_pairs, targets, proto_raw, mean_baseline, dtw_band, dtw_top_k)
    zero_content_top1 = float(np.mean([1.0 if lab == zero_content_class else 0.0 for lab in labels]))

    # --- honest verdict (STRICT) ---------------------------------------------
    # "Informative" must be judged against the STRICT controls (zero-EEG and shuffled-
    # EEG), NOT the weak label-prior: a model that emits a generic smooth mel for ANY
    # input (even zeros) already beats the raw label-mean, so beating label-prior is not
    # evidence of decoding. We require the prediction to beat BOTH strict controls on
    # PCC by a real margin, OR a content accuracy that is statistically above chance.
    pred_over_zeroeeg_pcc = pred["pcc"] - zero["pcc"]
    pred_over_shuffled_pcc = pred["pcc"] - shuffled["pcc"]
    content_se = (chance * (1.0 - chance) / max(n, 1)) ** 0.5  # binomial SE at chance
    content_z = (result["content_top1"] - chance) / max(content_se, 1e-8)
    content_significant = bool(content_z > 2.0)  # > ~2 SD above chance (not noise)
    recon_beats_controls = bool(pred_over_zeroeeg_pcc > EEG_INFO_PCC_EPS and pred_over_shuffled_pcc > EEG_INFO_PCC_EPS)
    eeg_informative = bool(recon_beats_controls or content_significant)

    result.update({
        # baselines (PCC higher=better; DTW lower=better)
        "zeroeeg_mel_PCC": zero["pcc"],
        "labelprior_mel_PCC": prior["pcc"],
        "shuffled_mel_PCC": shuffled["pcc"],
        "zeroeeg_pred_to_label_bank_dtw": zero["dtw"],
        "labelprior_pred_to_label_bank_dtw": prior["dtw"],
        "zeroeeg_content_top1": zero_content_top1,
        "zeroeeg_retrieval_top1": zero["retrieval_top1"],
        # gains: the headline honesty metrics (label-prior gain is THE one to watch)
        "pred_over_labelprior_pcc_gain": pred["pcc"] - prior["pcc"],
        "pred_over_zeroeeg_pcc_gain": pred["pcc"] - zero["pcc"],
        "pred_over_shuffled_pcc_gain": pred["pcc"] - shuffled["pcc"],
        "pred_over_labelprior_dtw_gain": prior["dtw"] - pred["dtw"],  # >0 means pred closer than the label prior
        "pred_over_zeroeeg_dtw_gain": zero["dtw"] - pred["dtw"],
        "content_over_chance": result["content_top1"] - chance,
        "retrieval_over_chance": result["retrieval_top1"] - chance,
        # STRICT honest verdict: reconstruction must beat BOTH zero-EEG and shuffled-EEG
        # on PCC by >= EEG_INFO_PCC_EPS, OR content must be statistically above chance.
        "content_z": content_z,
        "content_significant": content_significant,
        "eeg_recon_beats_controls": recon_beats_controls,
        "eeg_info_pcc_eps": EEG_INFO_PCC_EPS,
        "eeg_informative": eeg_informative,
    })
    return result
