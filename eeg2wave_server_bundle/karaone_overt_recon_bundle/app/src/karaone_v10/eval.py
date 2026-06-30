from __future__ import annotations

from typing import Any

from src.karaone_v91.eval import collect_v91_outputs, compute_v91_metrics, outputs_to_v91_bank, write_channel_reports


collect_v10_outputs = collect_v91_outputs
outputs_to_v10_bank = outputs_to_v91_bank


def compute_v10_metrics(
    outputs: dict[str, Any],
    *,
    train_bank: dict[str, Any] | None = None,
    prefix: str = "",
) -> dict[str, float | bool]:
    metrics = compute_v91_metrics(outputs, train_bank=train_bank, prefix="")
    metrics["v10_research_gate_pass"] = v10_gate_pass(metrics)
    metrics["v10_waveform_claim_allowed"] = bool(metrics["v10_research_gate_pass"])
    if prefix:
        return {f"{prefix}_{key}": value for key, value in metrics.items()}
    return metrics


def v10_gate_pass(metrics: dict[str, Any]) -> bool:
    return bool(
        float(metrics.get("semantic_over_zero_gain", 0.0)) > 0.01
        and float(metrics.get("semantic_over_mean_gain", 0.0)) > 0.0
        and float(metrics.get("semantic_top3_gain_over_mean", 0.0)) > 0.02
        and float(metrics.get("same_label_cross_subject_gain", -1.0)) >= 0.0
        and float(metrics.get("prompt_acc", 0.0)) >= 0.13
        and 0.7 <= float(metrics.get("pred_std_ratio_median", 0.0)) <= 1.5
        and float(metrics.get("pred_pairwise_corr_median", 1.0)) < 0.75
        and float(metrics.get("channel_gate_entropy_mean", 0.0)) > 0.20
    )


def v10_selection_score(row: dict[str, Any], *, prefix: str = "subject_val") -> float:
    """Gate-aware model selection score.

    The score rewards semantic gains and prompt accuracy, penalizes collapse,
    and gives hard penalties to the failure modes seen in v9.1.
    """

    def get(name: str, default: float = 0.0) -> float:
        return float(row.get(f"{prefix}_{name}", default))

    std_ratio = get("pred_std_ratio_median", 1.0)
    corr = get("pred_pairwise_corr_median", 1.0)
    entropy = get("channel_gate_entropy_mean", 0.0)
    score = (
        1.20 * get("semantic_over_zero_gain")
        + 1.50 * get("semantic_over_mean_gain")
        + 1.25 * get("semantic_top3_gain_over_mean")
        + 1.50 * get("same_label_cross_subject_gain")
        + 0.45 * get("prompt_acc")
        + 0.15 * get("cluster_label_top3")
        - 0.35 * max(0.0, corr - 0.75)
        - 0.25 * max(0.0, 0.7 - std_ratio)
        - 0.25 * max(0.0, std_ratio - 1.5)
        - 0.20 * max(0.0, 0.20 - entropy)
        - 0.05 * get("subject_leakage_acc")
    )
    if get("same_label_cross_subject_gain") < 0.0:
        score -= 0.10
    if get("semantic_over_zero_gain") <= 0.0:
        score -= 0.10
    if get("semantic_top3_gain_over_mean") <= 0.0:
        score -= 0.05
    return float(score)


def row_gate_summary(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "subject_val_semantic_over_zero_gain",
        "subject_val_semantic_over_mean_gain",
        "subject_val_semantic_top3_gain_over_mean",
        "subject_val_same_label_cross_subject_gain",
        "subject_val_prompt_acc",
        "subject_val_pred_std_ratio_median",
        "subject_val_pred_pairwise_corr_median",
        "subject_val_channel_gate_entropy_mean",
        "subject_val_v10_research_gate_pass",
        "subject_test_semantic_over_zero_gain",
        "subject_test_semantic_top3_gain_over_mean",
        "subject_test_same_label_cross_subject_gain",
        "subject_test_v10_research_gate_pass",
        "selection_score",
    ]
    return {key: row.get(key) for key in keys}


__all__ = [
    "collect_v10_outputs",
    "compute_v10_metrics",
    "outputs_to_v10_bank",
    "row_gate_summary",
    "v10_gate_pass",
    "v10_selection_score",
    "write_channel_reports",
]

