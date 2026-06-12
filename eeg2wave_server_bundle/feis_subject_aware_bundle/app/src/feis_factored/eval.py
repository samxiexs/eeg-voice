"""Evaluation for the factored model (v2: honest, gain-centred).

Reports the metrics that actually answer the research questions:
  - within_subject_content_top1 : decode WHAT within this subject's 16 prompts (chance 1/16)
  - content_top1_zeroeeg        : zero-EEG control (only stage + speaker id) -> baseline
  - content_gain                : top1 - zeroeeg  ==  THE decisive, non-gameable number
  - coarse manner/voicing/vc    : with their OWN zero-EEG baselines (gain, not raw)
  - recon_cos_to_cell           : predicted latent vs the cell's own target (latent fidelity)
  - latent collapse diagnostics : pred/target std ratio, pred vs ref pairwise correlation
  - holdout split: the SAME metrics on unseen (subject x label) cells -> beyond-classification
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _pairwise_corr_median(mat: np.ndarray) -> float:
    """Median off-diagonal Pearson correlation between rows of [N, D]."""
    if mat.shape[0] < 2:
        return float("nan")
    x = mat - mat.mean(axis=1, keepdims=True)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    c = x @ x.T
    iu = np.triu_indices(c.shape[0], k=1)
    return float(np.median(c[iu]))


@torch.no_grad()
def evaluate(model, dataset, targets, device="cpu", batch_size=64) -> dict:
    model.eval()
    # per-subject label prototype bank (the subject's own 16 cell summaries) for within-subject retrieval
    subj_label_summ: dict[str, np.ndarray] = {}
    for sub in dataset.subject_vocab:
        mat = np.stack([targets.cell_target(sub, lab).mean(0) for lab in dataset.label_vocab], 0)
        subj_label_summ[sub] = mat.astype(np.float32)              # [16, D]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n = 0
    ws_top1 = ws_top1_zero = 0
    manner_c = voicing_c = vc_c = 0
    manner_c0 = voicing_c0 = vc_c0 = 0
    sum_recon_cos = 0.0
    per_stage = defaultdict(lambda: [0, 0, 0])   # stage -> [eeg_correct, zero_correct, n]
    pred_summ_list: list[np.ndarray] = []
    tgt_summ_list: list[np.ndarray] = []
    std_ratios: list[float] = []

    for batch in loader:
        eeg = batch["eeg"].to(device)
        subj = batch["subject_idx"].to(device)
        stage = batch["stage_idx"].to(device)
        out = model(eeg, subj, stage)
        out0 = model(torch.zeros_like(eeg), subj, stage)            # zero-EEG control
        emb = F.normalize(out["content_embed"], dim=-1).cpu().numpy()
        emb0 = F.normalize(out0["content_embed"], dim=-1).cpu().numpy()
        cls_pred = out["content_logits"].argmax(-1).cpu().numpy()
        cls_pred0 = out0["content_logits"].argmax(-1).cpu().numpy()
        pred = out["pred_latent"]
        tgt = batch["target_seq"].to(device)
        rcos = F.cosine_similarity(pred, tgt, dim=-1).mean(dim=1).cpu().numpy()
        # collapse diagnostics
        ps = pred.reshape(-1, pred.shape[-1]).std(dim=0)
        ts = tgt.reshape(-1, tgt.shape[-1]).std(dim=0)
        std_ratios.append(float((ps / ts.clamp_min(1e-6)).median().cpu()))
        pred_summ_list.append(pred.mean(dim=1).cpu().numpy())       # [b, D]
        tgt_summ_list.append(tgt.mean(dim=1).cpu().numpy())

        for i in range(eeg.shape[0]):
            sub = batch["subject"][i]; lab = batch["label"][i]; stg = batch["stage"][i]
            bank = subj_label_summ[sub]                              # [16, D]
            bankn = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-8)
            r = int(np.argmax(bankn @ emb[i]))
            r0 = int(np.argmax(bankn @ emb0[i]))
            true_lab_id = dataset.label_to_id[lab]
            hit = (r == true_lab_id); hit0 = (r0 == true_lab_id)
            ws_top1 += int(hit); ws_top1_zero += int(hit0)
            per_stage[stg][0] += int(hit); per_stage[stg][1] += int(hit0); per_stage[stg][2] += 1
            # coarse via classifier prediction (EEG) and zero-EEG control
            cz = targets.coarse_ids(lab)
            pz = targets.coarse_ids(dataset.label_vocab[cls_pred[i]])
            pz0 = targets.coarse_ids(dataset.label_vocab[cls_pred0[i]])
            manner_c += int(pz["manner"] == cz["manner"]); manner_c0 += int(pz0["manner"] == cz["manner"])
            voicing_c += int(pz["voicing"] == cz["voicing"]); voicing_c0 += int(pz0["voicing"] == cz["voicing"])
            vc_c += int(pz["vc"] == cz["vc"]); vc_c0 += int(pz0["vc"] == cz["vc"])
            sum_recon_cos += float(rcos[i])
            n += 1

    stage_acc = {s: {"eeg": c / max(t, 1), "zeroeeg": z / max(t, 1),
                     "gain": (c - z) / max(t, 1)} for s, (c, z, t) in per_stage.items()}
    pred_summ = np.concatenate(pred_summ_list, 0) if pred_summ_list else np.zeros((0, 1))
    tgt_summ = np.concatenate(tgt_summ_list, 0) if tgt_summ_list else np.zeros((0, 1))

    top1 = ws_top1 / max(n, 1)
    zero = ws_top1_zero / max(n, 1)
    return {
        "split": dataset.split,
        "num_trials": n,
        "content_chance": 1.0 / max(dataset.num_labels, 1),
        "within_subject_content_top1": top1,
        "within_subject_content_top1_zeroeeg": zero,
        "content_gain": top1 - zero,                       # <-- THE decisive number
        "content_top1_by_stage": stage_acc,
        "coarse_manner_acc": manner_c / max(n, 1),
        "coarse_manner_acc_zeroeeg": manner_c0 / max(n, 1),
        "coarse_manner_gain": (manner_c - manner_c0) / max(n, 1),
        "coarse_voicing_acc": voicing_c / max(n, 1),
        "coarse_voicing_acc_zeroeeg": voicing_c0 / max(n, 1),
        "coarse_voicing_gain": (voicing_c - voicing_c0) / max(n, 1),
        "coarse_vc_acc": vc_c / max(n, 1),
        "coarse_vc_acc_zeroeeg": vc_c0 / max(n, 1),
        "coarse_vc_gain": (vc_c - vc_c0) / max(n, 1),
        "recon_cos_to_cell": sum_recon_cos / max(n, 1),
        # collapse diagnostics (secondary; std-match can game these)
        "pred_std_ratio_median": float(np.median(std_ratios)) if std_ratios else float("nan"),
        "pred_pairwise_corr_median": _pairwise_corr_median(pred_summ),
        "target_pairwise_corr_median": _pairwise_corr_median(tgt_summ),
    }
