from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .eval_utils import cosine_similarity_matrix


RETRIEVAL_POLICIES = (
    "auto",
    "same_subject_train",
    "pooled_train",
    "unseen_strict_seen_subjects",
    "unseen_oracle_holdout",
)

MATCH_MODE_BY_POLICY = {
    "same_subject_train": "exact",
    "pooled_train": "exact",
    "unseen_strict_seen_subjects": "label",
    "unseen_oracle_holdout": "exact",
}

AUTO_POLICY_BY_PROTOCOL = {
    "S": "same_subject_train",
    "G": "pooled_train",
    "U": "unseen_strict_seen_subjects",
}


@dataclass(frozen=True)
class RetrievalBank:
    policy: str
    match_mode: str
    target_kind: str
    template_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    labels: tuple[str, ...]
    audio_paths: tuple[str, ...]
    sequences: np.ndarray
    masks: np.ndarray
    summaries: np.ndarray
    waveforms: np.ndarray

    @property
    def size(self) -> int:
        return len(self.template_ids)

    @property
    def embeddings(self) -> np.ndarray:
        return self.summaries


@dataclass(frozen=True)
class _STFTBankFeature:
    fft_size: int
    hop_size: int
    win_size: int
    magnitudes: torch.Tensor
    norms: torch.Tensor


@dataclass(frozen=True)
class WaveformDistanceBank:
    bank: RetrievalBank
    features: tuple[_STFTBankFeature, ...]


def resolve_retrieval_policy(protocol: str, policy: str) -> str:
    protocol = str(protocol).upper()
    policy = str(policy)
    if policy not in RETRIEVAL_POLICIES:
        raise ValueError(f"Unsupported retrieval policy: {policy}")
    if policy == "auto":
        if protocol not in AUTO_POLICY_BY_PROTOCOL:
            raise ValueError(f"Unsupported protocol for auto policy: {protocol}")
        return AUTO_POLICY_BY_PROTOCOL[protocol]
    return policy


def build_retrieval_bank(dataset, policy: str) -> RetrievalBank:
    resolved_policy = resolve_retrieval_policy(dataset.protocol, policy)
    if resolved_policy == "same_subject_train":
        if str(dataset.protocol).upper() != "S":
            raise ValueError("same_subject_train is only valid for Protocol S")
        template_ids = dataset.unique_template_ids(split="train")
    elif resolved_policy in {"pooled_train", "unseen_strict_seen_subjects"}:
        template_ids = dataset.unique_template_ids(split="train")
    elif resolved_policy == "unseen_oracle_holdout":
        if str(dataset.protocol).upper() != "U":
            raise ValueError("unseen_oracle_holdout is only valid for Protocol U")
        template_ids = dataset.unique_template_ids(split="test")
    else:
        raise ValueError(f"Unsupported resolved policy: {resolved_policy}")

    subject_ids: list[str] = []
    labels: list[str] = []
    audio_paths: list[str] = []
    sequences: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    summaries: list[np.ndarray] = []
    waveforms: list[np.ndarray] = []
    for template_id in template_ids:
        metadata = dataset.template_metadata(template_id)
        target = dataset.get_template_target(template_id)
        subject_ids.append(str(metadata["subject_id"]))
        labels.append(str(metadata["label"]))
        audio_paths.append(str(metadata.get("audio_path") or metadata["audio_relpath"]))
        sequences.append(np.asarray(target["target_sequence"], dtype=np.float32))
        masks.append(np.asarray(target["target_mask"], dtype=np.float32))
        summaries.append(np.asarray(target["target_summary"], dtype=np.float32))
        waveforms.append(np.asarray(dataset._load_audio(metadata["audio_relpath"]), dtype=np.float32))

    if not sequences:
        raise ValueError(f"No templates found for policy={resolved_policy}")

    return RetrievalBank(
        policy=resolved_policy,
        match_mode=MATCH_MODE_BY_POLICY[resolved_policy],
        target_kind=str(getattr(dataset, "target_kind", "unknown")),
        template_ids=tuple(str(item) for item in template_ids),
        subject_ids=tuple(subject_ids),
        labels=tuple(labels),
        audio_paths=tuple(audio_paths),
        sequences=np.stack(sequences, axis=0).astype(np.float32),
        masks=np.stack(masks, axis=0).astype(np.float32),
        summaries=np.stack(summaries, axis=0).astype(np.float32),
        waveforms=np.stack(waveforms, axis=0).astype(np.float32),
    )


def target_match_available_rate(
    bank: RetrievalBank,
    target_template_ids: Sequence[str],
    target_labels: Sequence[str],
    match_mode: str,
) -> float:
    if match_mode == "exact":
        bank_ids = set(bank.template_ids)
        available = [str(template_id) in bank_ids for template_id in target_template_ids]
    elif match_mode == "label":
        bank_labels = set(bank.labels)
        available = [str(label) in bank_labels for label in target_labels]
    else:
        raise ValueError(f"Unsupported match_mode: {match_mode}")
    return float(np.mean(available)) if available else 0.0


def _candidate_matches(
    candidate: dict[str, object],
    target_template_id: str,
    target_label: str,
    match_mode: str,
) -> bool:
    if match_mode == "exact":
        return str(candidate["template_id"]) == str(target_template_id)
    if match_mode == "label":
        return str(candidate["label"]) == str(target_label)
    raise ValueError(f"Unsupported match_mode: {match_mode}")


def sequence_similarity_matrix(
    predicted_sequences: np.ndarray,
    bank_sequences: np.ndarray,
    predicted_masks: np.ndarray | None = None,
    bank_masks: np.ndarray | None = None,
) -> np.ndarray:
    predicted_sequences = np.asarray(predicted_sequences, dtype=np.float32)
    bank_sequences = np.asarray(bank_sequences, dtype=np.float32)
    if predicted_sequences.ndim != 3 or bank_sequences.ndim != 3:
        raise ValueError(
            f"Expected [N, T, D] and [M, T, D], got {tuple(predicted_sequences.shape)} and {tuple(bank_sequences.shape)}"
        )
    if predicted_sequences.shape[1:] != bank_sequences.shape[1:]:
        raise ValueError(
            f"Sequence shape mismatch: pred={tuple(predicted_sequences.shape[1:])} bank={tuple(bank_sequences.shape[1:])}"
        )
    predicted_masks = (
        np.ones(predicted_sequences.shape[:2], dtype=np.float32)
        if predicted_masks is None
        else np.asarray(predicted_masks, dtype=np.float32)
    )
    bank_masks = np.ones(bank_sequences.shape[:2], dtype=np.float32) if bank_masks is None else np.asarray(bank_masks, dtype=np.float32)
    pred_norm = predicted_sequences / np.clip(np.linalg.norm(predicted_sequences, axis=-1, keepdims=True), 1e-8, None)
    bank_norm = bank_sequences / np.clip(np.linalg.norm(bank_sequences, axis=-1, keepdims=True), 1e-8, None)
    frame_cos = np.einsum("ntd,mtd->nmt", pred_norm, bank_norm)
    weights = predicted_masks[:, None, :] * bank_masks[None, :, :]
    return ((frame_cos * weights).sum(axis=-1) / np.clip(weights.sum(axis=-1), 1e-8, None)).astype(np.float32)


def _availability_mask(
    bank: RetrievalBank,
    target_template_ids: Sequence[str],
    target_labels: Sequence[str],
) -> np.ndarray:
    if bank.match_mode == "exact":
        bank_ids = set(bank.template_ids)
        return np.asarray([str(template_id) in bank_ids for template_id in target_template_ids], dtype=bool)
    bank_labels = set(bank.labels)
    return np.asarray([str(label) in bank_labels for label in target_labels], dtype=bool)


def rank_bank_by_cosine(
    bank: RetrievalBank,
    predicted_sequences: np.ndarray | None = None,
    predicted_summaries: np.ndarray | None = None,
    predicted_masks: np.ndarray | None = None,
    top_k: int = 5,
) -> dict[str, object]:
    if predicted_sequences is None and predicted_summaries is None:
        raise ValueError("Either predicted_sequences or predicted_summaries must be provided")
    if predicted_sequences is not None:
        predicted_sequences = np.asarray(predicted_sequences, dtype=np.float32)
    if predicted_summaries is not None:
        predicted_summaries = np.asarray(predicted_summaries, dtype=np.float32)
    if predicted_sequences is not None and predicted_sequences.ndim == 3:
        sims = sequence_similarity_matrix(
            predicted_sequences=predicted_sequences,
            bank_sequences=bank.sequences,
            predicted_masks=predicted_masks,
            bank_masks=bank.masks,
        )
    else:
        if predicted_summaries is None:
            if predicted_sequences is None or predicted_sequences.ndim != 2:
                raise ValueError("Summary predictions are required when predicted_sequences is not [N, T, D]")
            predicted_summaries = predicted_sequences
        sims = cosine_similarity_matrix(predicted_summaries, bank.summaries)
    order = np.argsort(-sims, axis=1)
    top_k = max(1, min(int(top_k), bank.size))
    topk_order = order[:, :top_k]
    ranked_candidates: list[list[dict[str, object]]] = []
    for row_idx in range(topk_order.shape[0]):
        row: list[dict[str, object]] = []
        for rank, col_idx in enumerate(topk_order[row_idx].tolist(), start=1):
            row.append(
                {
                    "rank": rank,
                    "template_id": bank.template_ids[col_idx],
                    "subject_id": bank.subject_ids[col_idx],
                    "label": bank.labels[col_idx],
                    "audio_path": bank.audio_paths[col_idx],
                    "cosine_similarity": float(sims[row_idx, col_idx]),
                }
            )
        ranked_candidates.append(row)
    return {
        "similarity_matrix": sims,
        "order": order,
        "topk_order": topk_order,
        "ranked_candidates": ranked_candidates,
    }


def compute_rank_metrics(
    ranked_candidates: Sequence[Sequence[dict[str, object]]],
    order: np.ndarray,
    bank: RetrievalBank,
    target_template_ids: Sequence[str],
    target_labels: Sequence[str],
    top_k_values: Sequence[int] = (1, 5),
    evaluation_mask: np.ndarray | None = None,
) -> dict[str, float | int | None | str]:
    num_samples = len(target_template_ids)
    available_mask = _availability_mask(bank, target_template_ids=target_template_ids, target_labels=target_labels)
    if evaluation_mask is None:
        evaluation_mask = available_mask
    else:
        evaluation_mask = np.asarray(evaluation_mask, dtype=bool) & available_mask
    evaluated_count = int(evaluation_mask.sum())
    suffix = "exact" if bank.match_mode == "exact" else "label"
    metrics: dict[str, float | int | None | str] = {
        "match_mode": bank.match_mode,
        "evaluation_count": evaluated_count,
        "availability_rate": float(evaluated_count / max(num_samples, 1)),
    }
    first_ranks: list[int] = []
    for idx, ordered_cols in enumerate(order.tolist()):
        if not evaluation_mask[idx]:
            continue
        rank = None
        for position, col_idx in enumerate(ordered_cols, start=1):
            candidate = {
                "template_id": bank.template_ids[col_idx],
                "label": bank.labels[col_idx],
            }
            if _candidate_matches(candidate, target_template_ids[idx], target_labels[idx], bank.match_mode):
                rank = position
                break
        if rank is not None:
            first_ranks.append(rank)

    for k in top_k_values:
        metric_key = f"retrieval_top{int(k)}_{suffix}"
        if evaluated_count == 0:
            metrics[metric_key] = None
        else:
            metrics[metric_key] = float(np.mean([rank <= int(k) for rank in first_ranks])) if first_ranks else 0.0
    if evaluated_count == 0:
        metrics["MRR"] = None
        metrics["mean_rank"] = None
    else:
        metrics["MRR"] = float(np.mean([1.0 / rank for rank in first_ranks])) if first_ranks else 0.0
        metrics["mean_rank"] = float(np.mean(first_ranks)) if first_ranks else float(bank.size + 1)
    return metrics


def compute_first_match_ranks(
    order: np.ndarray,
    bank: RetrievalBank,
    target_template_ids: Sequence[str],
    target_labels: Sequence[str],
    evaluation_mask: np.ndarray | None = None,
) -> list[int | None]:
    available_mask = _availability_mask(bank, target_template_ids=target_template_ids, target_labels=target_labels)
    if evaluation_mask is None:
        evaluation_mask = available_mask
    else:
        evaluation_mask = np.asarray(evaluation_mask, dtype=bool) & available_mask
    ranks: list[int | None] = []
    for idx, ordered_cols in enumerate(order.tolist()):
        if not evaluation_mask[idx]:
            ranks.append(None)
            continue
        rank = None
        for position, col_idx in enumerate(ordered_cols, start=1):
            candidate = {
                "template_id": bank.template_ids[col_idx],
                "label": bank.labels[col_idx],
            }
            if _candidate_matches(candidate, target_template_ids[idx], target_labels[idx], bank.match_mode):
                rank = position
                break
        ranks.append(rank)
    return ranks


def compute_mean_top1_similarity(
    ranked_candidates: Sequence[Sequence[dict[str, object]]],
    evaluation_mask: np.ndarray | None = None,
) -> float | None:
    if evaluation_mask is None:
        evaluation_mask = np.ones(len(ranked_candidates), dtype=bool)
    else:
        evaluation_mask = np.asarray(evaluation_mask, dtype=bool)
    scores = [
        float(candidates[0]["cosine_similarity"])
        for idx, candidates in enumerate(ranked_candidates)
        if evaluation_mask[idx] and candidates
    ]
    if not scores:
        return None
    return float(np.mean(scores))


def evaluate_embedding_retrieval(
    bank: RetrievalBank,
    target_template_ids: Sequence[str],
    target_labels: Sequence[str],
    predicted_sequences: np.ndarray | None = None,
    predicted_summaries: np.ndarray | None = None,
    predicted_masks: np.ndarray | None = None,
    top_k: int = 5,
    evaluation_mask: np.ndarray | None = None,
) -> dict[str, object]:
    ranked = rank_bank_by_cosine(
        bank=bank,
        predicted_sequences=predicted_sequences,
        predicted_summaries=predicted_summaries,
        predicted_masks=predicted_masks,
        top_k=top_k,
    )
    effective_mask = _availability_mask(bank, target_template_ids=target_template_ids, target_labels=target_labels)
    if evaluation_mask is not None:
        effective_mask = effective_mask & np.asarray(evaluation_mask, dtype=bool)
    metrics = compute_rank_metrics(
        ranked_candidates=ranked["ranked_candidates"],
        order=ranked["order"],
        bank=bank,
        target_template_ids=target_template_ids,
        target_labels=target_labels,
        top_k_values=(1, top_k),
        evaluation_mask=effective_mask,
    )
    metrics["target_match_available_rate"] = target_match_available_rate(
        bank=bank,
        target_template_ids=target_template_ids,
        target_labels=target_labels,
        match_mode=bank.match_mode,
    )
    metrics["mean_top1_cosine_similarity"] = compute_mean_top1_similarity(
        ranked_candidates=ranked["ranked_candidates"],
        evaluation_mask=effective_mask,
    )
    return {
        **ranked,
        "metrics": metrics,
    }


def _waveform_2d(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim == 3:
        waveform = waveform.squeeze(1)
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform [B, T] or [B, 1, T], got {tuple(waveform.shape)}")
    return waveform


def _stft_magnitude_batch(waveform: torch.Tensor, fft_size: int, hop_size: int, win_size: int) -> torch.Tensor:
    waveform = _waveform_2d(waveform)
    window = torch.hann_window(win_size, device=waveform.device, dtype=waveform.dtype)
    spec = torch.stft(
        waveform,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=True,
        return_complex=True,
    )
    return spec.abs()


def build_waveform_distance_bank(
    bank: RetrievalBank,
    fft_sizes: Sequence[int] = (512, 1024, 2048),
    hop_sizes: Sequence[int] = (128, 256, 512),
    win_sizes: Sequence[int] = (512, 1024, 2048),
) -> WaveformDistanceBank:
    waveforms = torch.from_numpy(bank.waveforms).float()
    features: list[_STFTBankFeature] = []
    for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
        magnitudes = _stft_magnitude_batch(waveforms, fft_size=fft_size, hop_size=hop_size, win_size=win_size).cpu()
        norms = torch.linalg.norm(magnitudes, dim=(-2, -1)).cpu()
        features.append(
            _STFTBankFeature(
                fft_size=int(fft_size),
                hop_size=int(hop_size),
                win_size=int(win_size),
                magnitudes=magnitudes,
                norms=norms,
            )
        )
    return WaveformDistanceBank(bank=bank, features=tuple(features))


def stft_distance_to_bank(query_waveform: np.ndarray, bank_features: WaveformDistanceBank) -> np.ndarray:
    query = torch.from_numpy(np.asarray(query_waveform, dtype=np.float32)).reshape(1, -1)
    total = torch.zeros(bank_features.bank.size, dtype=torch.float32)
    for feature in bank_features.features:
        query_mag = _stft_magnitude_batch(
            query,
            fft_size=feature.fft_size,
            hop_size=feature.hop_size,
            win_size=feature.win_size,
        ).squeeze(0).cpu()
        diff = feature.magnitudes - query_mag.unsqueeze(0)
        mag_loss = diff.abs().mean(dim=(-2, -1))
        sc_loss = torch.linalg.norm(diff, dim=(-2, -1)) / (feature.norms + 1e-8)
        total = total + mag_loss + sc_loss
    return (total / max(len(bank_features.features), 1)).numpy().astype(np.float32)


def evaluate_waveform_nta(
    output_waveforms: Sequence[np.ndarray],
    target_template_ids: Sequence[str],
    target_labels: Sequence[str],
    bank_features: WaveformDistanceBank,
    evaluation_mask: np.ndarray | None = None,
    cache_keys: Sequence[str] | None = None,
) -> dict[str, object]:
    num_samples = len(target_template_ids)
    available_mask = _availability_mask(
        bank_features.bank,
        target_template_ids=target_template_ids,
        target_labels=target_labels,
    )
    if evaluation_mask is None:
        evaluation_mask = available_mask
    else:
        evaluation_mask = np.asarray(evaluation_mask, dtype=bool) & available_mask
    cache: dict[str, np.ndarray] = {}
    nearest_rows: list[dict[str, object] | None] = []
    nta_hits = 0
    evaluated_count = int(evaluation_mask.sum())
    for idx in range(num_samples):
        if not evaluation_mask[idx]:
            nearest_rows.append(None)
            continue
        cache_key = None if cache_keys is None else str(cache_keys[idx])
        if cache_key is not None and cache_key in cache:
            distances = cache[cache_key]
        else:
            distances = stft_distance_to_bank(output_waveforms[idx], bank_features)
            if cache_key is not None:
                cache[cache_key] = distances
        best_idx = int(np.argmin(distances))
        nearest = {
            "template_id": bank_features.bank.template_ids[best_idx],
            "subject_id": bank_features.bank.subject_ids[best_idx],
            "label": bank_features.bank.labels[best_idx],
            "audio_path": bank_features.bank.audio_paths[best_idx],
            "stft_distance": float(distances[best_idx]),
        }
        nearest_rows.append(nearest)
        nta_hits += int(
            _candidate_matches(
                nearest,
                target_template_id=target_template_ids[idx],
                target_label=target_labels[idx],
                match_mode=bank_features.bank.match_mode,
            )
        )

    suffix = "exact" if bank_features.bank.match_mode == "exact" else "label"
    metrics: dict[str, float | int | None | str] = {
        "match_mode": bank_features.bank.match_mode,
        "availability_rate": float(evaluated_count / max(num_samples, 1)),
        "evaluation_count": evaluated_count,
        f"NTA_{suffix}": None if evaluated_count == 0 else nta_hits / max(evaluated_count, 1),
    }
    return {
        "metrics": metrics,
        "nearest_rows": nearest_rows,
    }


def _expected_random_first_rank(num_candidates: int, num_matches: int) -> tuple[float, float]:
    ranks = np.arange(1, num_candidates - num_matches + 2, dtype=np.float64)
    numerators = np.asarray(
        [math.comb(num_candidates - int(rank), num_matches - 1) for rank in ranks.tolist()],
        dtype=np.float64,
    )
    probabilities = numerators / max(math.comb(num_candidates, num_matches), 1)
    mean_rank = float(np.sum(ranks * probabilities))
    mrr = float(np.sum((1.0 / ranks) * probabilities))
    return mean_rank, mrr


def expected_random_retrieval_metrics(
    bank: RetrievalBank,
    target_template_ids: Sequence[str],
    target_labels: Sequence[str],
    top_k: int = 5,
    evaluation_mask: np.ndarray | None = None,
) -> dict[str, float | int | None | str]:
    num_samples = len(target_template_ids)
    available_mask = _availability_mask(bank, target_template_ids=target_template_ids, target_labels=target_labels)
    if evaluation_mask is None:
        evaluation_mask = available_mask
    else:
        evaluation_mask = np.asarray(evaluation_mask, dtype=bool) & available_mask
    evaluated_indices = np.flatnonzero(evaluation_mask)
    evaluated_count = int(len(evaluated_indices))
    suffix = "exact" if bank.match_mode == "exact" else "label"
    metrics: dict[str, float | int | None | str] = {
        "match_mode": bank.match_mode,
        "availability_rate": float(evaluated_count / max(num_samples, 1)),
        "evaluation_count": evaluated_count,
        "target_match_available_rate": target_match_available_rate(
            bank=bank,
            target_template_ids=target_template_ids,
            target_labels=target_labels,
            match_mode=bank.match_mode,
        ),
    }
    if evaluated_count == 0:
        metrics[f"retrieval_top1_{suffix}"] = None
        metrics[f"retrieval_top{int(top_k)}_{suffix}"] = None
        metrics["MRR"] = None
        metrics["mean_rank"] = None
        metrics[f"NTA_{suffix}"] = None
        return metrics

    bank_size = bank.size
    top1_scores: list[float] = []
    topk_scores: list[float] = []
    mrr_scores: list[float] = []
    rank_scores: list[float] = []
    bank_ids = list(bank.template_ids)
    label_counts: dict[str, int] = {}
    for label in bank.labels:
        label_counts[str(label)] = label_counts.get(str(label), 0) + 1
    for idx in evaluated_indices.tolist():
        if bank.match_mode == "exact":
            match_count = 1 if str(target_template_ids[idx]) in bank_ids else 0
        elif bank.match_mode == "label":
            match_count = label_counts.get(str(target_labels[idx]), 0)
        else:
            raise ValueError(f"Unsupported match_mode: {bank.match_mode}")
        if match_count <= 0:
            continue
        top1_scores.append(match_count / max(bank_size, 1))
        k = min(int(top_k), bank_size)
        if k >= bank_size:
            topk_scores.append(1.0)
        else:
            numerator = math.comb(bank_size - match_count, k)
            denominator = math.comb(bank_size, k)
            topk_scores.append(1.0 - float(numerator / max(denominator, 1)))
        mean_rank, mrr = _expected_random_first_rank(bank_size, match_count)
        rank_scores.append(mean_rank)
        mrr_scores.append(mrr)

    metrics[f"retrieval_top1_{suffix}"] = float(np.mean(top1_scores)) if top1_scores else 0.0
    metrics[f"retrieval_top{int(top_k)}_{suffix}"] = float(np.mean(topk_scores)) if topk_scores else 0.0
    metrics["MRR"] = float(np.mean(mrr_scores)) if mrr_scores else 0.0
    metrics["mean_rank"] = float(np.mean(rank_scores)) if rank_scores else float(bank_size + 1)
    metrics[f"NTA_{suffix}"] = metrics[f"retrieval_top1_{suffix}"]
    return metrics


def summarize_candidates(
    ranked_candidates: Sequence[Sequence[dict[str, object]]],
    nearest_rows: Sequence[dict[str, object] | None],
    target_template_ids: Sequence[str],
    target_labels: Sequence[str],
    match_mode: str,
) -> list[dict[str, object]]:
    suffix = "exact" if match_mode == "exact" else "label"
    rows: list[dict[str, object]] = []
    for idx, candidates in enumerate(ranked_candidates):
        top1 = candidates[0] if candidates else None
        nta_row = nearest_rows[idx] if idx < len(nearest_rows) else None
        first_match_rank = None
        for candidate in candidates:
            if _candidate_matches(candidate, target_template_ids[idx], target_labels[idx], match_mode):
                first_match_rank = int(candidate["rank"])
                break
        rows.append(
            {
                "retrieved_template_id": None if top1 is None else top1["template_id"],
                "retrieved_subject_id": None if top1 is None else top1["subject_id"],
                "retrieved_label": None if top1 is None else top1["label"],
                "top1_cosine_similarity": None if top1 is None else top1["cosine_similarity"],
                "first_match_rank": first_match_rank,
                "first_match_reciprocal_rank": None if first_match_rank is None else float(1.0 / first_match_rank),
                "retrieved_matches_target": False
                if top1 is None
                else _candidate_matches(top1, target_template_ids[idx], target_labels[idx], match_mode),
                f"NTA_{suffix}_match": False
                if nta_row is None
                else _candidate_matches(nta_row, target_template_ids[idx], target_labels[idx], match_mode),
                "nearest_waveform_template_id": None if nta_row is None else nta_row["template_id"],
                "nearest_waveform_label": None if nta_row is None else nta_row["label"],
                "nearest_waveform_distance": None if nta_row is None else nta_row["stft_distance"],
                "topk": list(candidates),
            }
        )
    return rows
