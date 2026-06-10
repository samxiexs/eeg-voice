"""v3 retrieval evaluation — subject-specific by design.

FEIS targets are per-subject recordings: subject 01's "f" and subject 02's "f"
are different waveforms (336 = 21 subjects x 16 prompts distinct targets). So
the PRIMARY metric retrieves against the 336 subject-specific templates and
counts a hit only when the *correct subject's correct prompt* is recovered
(`template_top1/topk`). Label-only retrieval (`label_top1/topk`) is reported as
a secondary diagnostic — it ignores the subject identity and is the weaker,
"which of 16 prompts" view.

bank_split:
  "train"  — bank from train templates (Protocol G/S: test subjects appear in
             train via other trials, so subject-specific retrieval is valid).
  "test"   — oracle bank from test templates (Protocol U: the held-out subject
             is absent from train, so template-level retrieval needs the test
             templates as the bank).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _build_banks(dataset, bank_split: str = "train"):
    """Return template bank (subject-specific) and label bank (subject-agnostic)."""
    template_ids: list[str] = []
    template_vecs: list[np.ndarray] = []
    template_label: dict[str, str] = {}
    label_pools: dict[str, list[np.ndarray]] = defaultdict(list)
    for template_id in dataset.unique_template_ids(split=bank_split):
        meta = dataset.template_metadata(template_id)
        summary = np.asarray(dataset.get_template_target(template_id)["target_summary"], dtype=np.float32)
        template_ids.append(template_id)
        template_vecs.append(summary)
        template_label[template_id] = meta["label"]
        label_pools[meta["label"]].append(summary)
    template_mat = np.stack(template_vecs, 0).astype(np.float32)
    labels = sorted(label_pools.keys())
    label_mat = np.stack([np.mean(np.stack(label_pools[l], 0), 0) for l in labels], 0).astype(np.float32)
    return template_ids, template_mat, template_label, labels, label_mat


@torch.no_grad()
def evaluate(
    model,
    dataset,
    device: str = "cpu",
    top_k: int = 5,
    batch_size: int = 32,
    bank_split: str = "train",
    forward_kwargs: dict | None = None,
) -> dict:
    """Subject-specific + label-only retrieval and class-head accuracy.

    `forward_kwargs` lets the multi-dataset model pass e.g. {"dataset_name": "feis"}.
    """
    model.eval()
    fkw = forward_kwargs or {}
    template_ids, template_mat, template_label, labels, label_mat = _build_banks(dataset, bank_split)
    tpl_t = F.normalize(torch.from_numpy(template_mat).to(device), dim=-1)   # [N_tpl, D]
    lbl_t = F.normalize(torch.from_numpy(label_mat).to(device), dim=-1)      # [N_lbl, D]
    label_set = set(labels)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n = 0
    tpl_top1 = tpl_topk = 0
    lbl_top1 = lbl_topk = 0
    cls_correct = 0
    n_tpl_evaluable = 0   # trials whose true template exists in the bank
    predictions = []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        subj = batch["subject_index"].to(device)
        out = model(eeg, subj, **fkw)
        emb = F.normalize(out["contrastive_embedding"], dim=-1)             # [B, D]
        tpl_rank = (emb @ tpl_t.transpose(0, 1)).argsort(dim=-1, descending=True).cpu().numpy()
        lbl_rank = (emb @ lbl_t.transpose(0, 1)).argsort(dim=-1, descending=True).cpu().numpy()
        cls_pred = out["label_logits"].argmax(dim=-1).cpu().numpy()
        for i in range(eeg.shape[0]):
            true_label = batch["label"][i]
            true_tpl = batch["template_id"][i]
            # Subject-specific template retrieval (the primary metric).
            tpl_order = tpl_rank[i]
            tpl_r1 = template_ids[tpl_order[0]]
            tpl_rk = {template_ids[c] for c in tpl_order[:top_k]}
            if true_tpl in template_ids:
                n_tpl_evaluable += 1
                if tpl_r1 == true_tpl:
                    tpl_top1 += 1
                if true_tpl in tpl_rk:
                    tpl_topk += 1
            # Label-only retrieval (secondary).
            lbl_order = lbl_rank[i]
            lbl_r1 = labels[lbl_order[0]]
            lbl_rk = {labels[c] for c in lbl_order[:top_k]}
            if true_label in label_set:
                if lbl_r1 == true_label:
                    lbl_top1 += 1
                if true_label in lbl_rk:
                    lbl_topk += 1
            cls_label = dataset.label_vocab[cls_pred[i]]
            if cls_label == true_label:
                cls_correct += 1
            predictions.append(
                {
                    "subject_id": batch["subject_id"][i],
                    "trial_index": int(batch["trial_index"][i]),
                    "template_id": true_tpl,
                    "label": true_label,
                    "template_top1": tpl_r1,
                    "label_top1": lbl_r1,
                    "cls_pred": cls_label,
                }
            )
            n += 1

    denom_tpl = max(n_tpl_evaluable, 1)
    return {
        "num_trials": n,
        "num_templates": len(template_ids),
        "num_labels": len(labels),
        "bank_split": bank_split,
        "template_evaluable_trials": n_tpl_evaluable,
        "template_chance": 1.0 / max(len(template_ids), 1),
        "label_chance": 1.0 / max(len(labels), 1),
        # PRIMARY: correct subject AND correct prompt.
        "template_top1": tpl_top1 / denom_tpl,
        f"template_top{top_k}": tpl_topk / denom_tpl,
        # SECONDARY: correct prompt only (subject-agnostic).
        "label_top1": lbl_top1 / max(n, 1),
        f"label_top{top_k}": lbl_topk / max(n, 1),
        "class_head_accuracy": cls_correct / max(n, 1),
        "predictions": predictions,
    }


@torch.no_grad()
def evaluate_unified(model, train_dataset, test_dataset, dataset_name: str,
                     device: str = "cpu", top_k: int = 5, batch_size: int = 32) -> dict:
    """Retrieval eval for the multi-dataset model on one `UnifiedEEGSpeechDataset`.

    Self-contained against the unified interface (items carry `target_key`,
    `target_summary`, `label`). Two banks are built from the train split:

    - key bank (subject-specific): key = "subject:label" for FEIS (reused across
      its trials -> evaluable) or "subject:trial" for KaraOne (unique per trial ->
      a test key never appears in train, so template retrieval is not evaluable;
      for KaraOne the meaningful signal is reconstruction fidelity, see
      recon_eval.evaluate_reconstruction).
    - label bank: per-label centroid (always evaluable).
    """
    model.eval()
    key_pool: dict[str, list[np.ndarray]] = defaultdict(list)
    label_pool: dict[str, list[np.ndarray]] = defaultdict(list)
    for i in range(len(train_dataset)):
        it = train_dataset[i]
        summ = it["target_summary"].numpy().astype(np.float32)
        key_pool[it["target_key"]].append(summ)
        label_pool[it["label"]].append(summ)
    keys = sorted(key_pool.keys())
    key_mat = np.stack([np.mean(np.stack(key_pool[k], 0), 0) for k in keys], 0).astype(np.float32)
    key_label = {k: k.split(":", 1)[1] if train_dataset.spec.target_key == "template" else None for k in keys}
    labels = sorted(label_pool.keys())
    label_mat = np.stack([np.mean(np.stack(label_pool[l], 0), 0) for l in labels], 0).astype(np.float32)
    key_t = F.normalize(torch.from_numpy(key_mat).to(device), dim=-1)
    lbl_t = F.normalize(torch.from_numpy(label_mat).to(device), dim=-1)
    key_set = set(keys)

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    n = tpl_eval = tpl_top1 = tpl_topk = lbl_top1 = lbl_topk = cls_correct = 0
    for batch in loader:
        out = model(batch["eeg"].to(device), batch["subject_index"].to(device), dataset_name)
        emb = F.normalize(out["contrastive_embedding"], dim=-1)
        key_rank = (emb @ key_t.transpose(0, 1)).argsort(dim=-1, descending=True).cpu().numpy()
        lbl_rank = (emb @ lbl_t.transpose(0, 1)).argsort(dim=-1, descending=True).cpu().numpy()
        cls_pred = out["label_logits"].argmax(dim=-1).cpu().numpy()
        for i in range(len(batch["label"])):
            true_label = batch["label"][i]
            true_key = batch["target_key"][i]
            if true_key in key_set:
                tpl_eval += 1
                ko = key_rank[i]
                if keys[ko[0]] == true_key:
                    tpl_top1 += 1
                if true_key in {keys[c] for c in ko[:top_k]}:
                    tpl_topk += 1
            lo = lbl_rank[i]
            if labels[lo[0]] == true_label:
                lbl_top1 += 1
            if true_label in {labels[c] for c in lo[:top_k]}:
                lbl_topk += 1
            if test_dataset.label_vocab[cls_pred[i]] == true_label:
                cls_correct += 1
            n += 1

    return {
        "dataset": dataset_name,
        "num_trials": n,
        "num_keys": len(keys),
        "num_labels": len(labels),
        "template_evaluable_trials": tpl_eval,
        "template_chance": 1.0 / max(len(keys), 1),
        "label_chance": 1.0 / max(len(labels), 1),
        "template_top1": tpl_top1 / max(tpl_eval, 1),
        f"template_top{top_k}": tpl_topk / max(tpl_eval, 1),
        "label_top1": lbl_top1 / max(n, 1),
        f"label_top{top_k}": lbl_topk / max(n, 1),
        "class_head_accuracy": cls_correct / max(n, 1),
    }
