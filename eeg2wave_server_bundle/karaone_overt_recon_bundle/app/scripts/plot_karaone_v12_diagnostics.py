from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from scripts.train_karaone_v12 import load_checkpoint, make_model_config  # noqa: E402
from src.karaone_v9.data import KaraOneV9TargetBank  # noqa: E402
from src.karaone_v12.data import KaraOneV10ClusterBank, KaraOneV11TokenBank, KaraOneV12Dataset, KaraOneV12TimeAnchorBank  # noqa: E402
from src.karaone_v12.eval import collect_v12_outputs  # noqa: E402
from src.karaone_v12.model import KaraOneV12TokenGenerator  # noqa: E402
from src.utils import load_simple_yaml, resolve_bundle_path, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate v12 diagnostic figures: confusion/AUC, active ROC-PR, token top-k, codec top-k, lag-aware summaries.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone_v12.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--stages", default=None)
    parser.add_argument("--cluster-bank", default=None)
    parser.add_argument("--token-bank", default=None)
    parser.add_argument("--time-anchor-bank", default=None)
    parser.add_argument("--wav-dir", default=None, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    fig_dir = run_dir / "figures" / "diagnostics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.checkpoint).expanduser() if args.checkpoint else run_dir / "checkpoints" / "best.pt"
    written: list[str] = []
    summary_rows: list[dict[str, Any]] = []

    if ckpt.exists():
        cfg = load_simple_yaml(args.config)
        stages = tuple(item.strip() for item in str(args.stages or cfg["data"].get("stages", "overt_like")).split(",") if item.strip())
        device = torch.device(args.device)
        root = resolve_bundle_path(cfg["data"]["root"], BUNDLE_DIR)
        cache_cfg = cfg.get("cache", {})
        targets = KaraOneV9TargetBank(
            resolve_bundle_path(cache_cfg["semantic"], BUNDLE_DIR),
            codec_cache=resolve_bundle_path(cache_cfg["codec"], BUNDLE_DIR) if cache_cfg.get("codec") else None,
            prosody_cache=resolve_bundle_path(cache_cfg["prosody"], BUNDLE_DIR) if cache_cfg.get("prosody") else None,
            semantic_token_cache=resolve_bundle_path(cache_cfg["semantic_tokens"], BUNDLE_DIR) if cache_cfg.get("semantic_tokens") else None,
            data_root=root,
        )
        subject_val = str(cfg["data"].get("subject_val", "P02"))
        subject_test = str(cfg["data"].get("subject_test", "MM21"))
        eeg_len = int(cfg["data"].get("eeg_len", 1280))
        cluster_bank = KaraOneV10ClusterBank(resolve_bundle_path(args.cluster_bank or cache_cfg.get("cluster_bank", ""), BUNDLE_DIR))
        token_bank = KaraOneV11TokenBank(resolve_bundle_path(args.token_bank or cache_cfg.get("v11_token_bank", ""), BUNDLE_DIR))
        time_bank = KaraOneV12TimeAnchorBank(resolve_bundle_path(args.time_anchor_bank or cache_cfg.get("v12_time_anchor_bank", ""), BUNDLE_DIR))
        train_ds = KaraOneV12Dataset(root, targets, str(cfg["data"].get("train_split", "subject_train")), cluster_bank=cluster_bank, token_bank=token_bank, time_anchor_bank=time_bank, stages=stages, subject_val=subject_val, subject_test=subject_test, eeg_len=eeg_len)
        val_ds = KaraOneV12Dataset(root, targets, "subject_val", cluster_bank=cluster_bank, token_bank=token_bank, time_anchor_bank=time_bank, stages=stages, subject_val=subject_val, subject_test=subject_test, eeg_len=eeg_len)
        test_ds = KaraOneV12Dataset(root, targets, "subject_test", cluster_bank=cluster_bank, token_bank=token_bank, time_anchor_bank=time_bank, stages=stages, subject_val=subject_val, subject_test=subject_test, eeg_len=eeg_len)
        model = KaraOneV12TokenGenerator(make_model_config(cfg, targets, train_ds, token_bank, time_bank, eeg_len=eeg_len, stages=stages), codec_codebook=torch.from_numpy(token_bank.codec_codebook)).to(device)
        payload = load_checkpoint(model, resolve_bundle_path(ckpt, BUNDLE_DIR))
        if isinstance(payload, dict) and payload.get("aligner"):
            model.cfg.aligner = str(payload["aligner"]).lower()
        train_prior = collect_token_priors(train_ds, batch_size=int(args.batch_size))
        for split_name, dataset in [("subject_val", val_ds), ("subject_test", test_ds)]:
            outputs = collect_v12_outputs(model, dataset, device=device, batch_size=int(args.batch_size))
            label_names = infer_label_names(outputs)
            paths, rows = plot_split_diagnostics(outputs, train_prior, fig_dir, split_name, label_names)
            written.extend(str(path) for path in paths)
            summary_rows.extend(rows)
    else:
        summary_rows.append({"split": "all", "metric": "checkpoint_missing", "value": str(ckpt)})

    wav_dir = args.wav_dir.expanduser().resolve() if args.wav_dir else run_dir / "wavs"
    if wav_dir.exists():
        paths, rows = plot_lagaware_summaries(wav_dir, fig_dir)
        written.extend(str(path) for path in paths)
        summary_rows.extend(rows)

    write_summary(run_dir, fig_dir, written, summary_rows)
    print(json.dumps({"figures": written, "summary_rows": len(summary_rows)}, ensure_ascii=False, indent=2))


def collect_token_priors(dataset: KaraOneV12Dataset, *, batch_size: int) -> dict[str, np.ndarray]:
    semantic_tokens, semantic_mask, codec_tokens, codec_mask = [], [], [], []
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    for batch in loader:
        semantic_tokens.append(batch["audio_semantic_tokens"].numpy())
        semantic_mask.append(batch["audio_semantic_token_mask"].numpy())
        codec_tokens.append(batch["codec_token_targets"].numpy())
        codec_mask.append(batch["codec_token_mask"].numpy())
    return {
        "semantic_tokens": np.concatenate(semantic_tokens, axis=0).astype(np.int64),
        "semantic_mask": np.concatenate(semantic_mask, axis=0).astype(np.float32),
        "codec_tokens": np.concatenate(codec_tokens, axis=0).astype(np.int64),
        "codec_mask": np.concatenate(codec_mask, axis=0).astype(np.float32),
    }


def plot_split_diagnostics(outputs: dict[str, Any], train_prior: dict[str, np.ndarray], fig_dir: Path, split_name: str, label_names: list[str]) -> tuple[list[Path], list[dict[str, Any]]]:
    paths: list[Path] = []
    rows: list[dict[str, Any]] = []
    paths.append(plot_prompt_confusion(outputs, fig_dir / f"prompt_confusion_matrix_{split_name}.png", split_name, label_names))
    path, auc_rows = plot_prompt_auc(outputs, fig_dir / f"prompt_ovr_auc_{split_name}.png", split_name, label_names)
    paths.append(path)
    rows.extend(auc_rows)
    path, active_rows = plot_active_roc_pr(outputs, fig_dir / f"active_mask_roc_pr_{split_name}.png", split_name)
    paths.append(path)
    rows.extend(active_rows)
    path, semantic_rows = plot_topk_curve(
        outputs["semantic_token_logits"],
        outputs["target_semantic_tokens"],
        outputs["target_semantic_token_mask"],
        train_prior["semantic_tokens"],
        train_prior["semantic_mask"],
        fig_dir / f"semantic_token_topk_curve_{split_name}.png",
        split_name,
        "Semantic Token Top-k Accuracy",
        "semantic_token",
    )
    paths.append(path)
    rows.extend(semantic_rows)
    path, codec_rows = plot_topk_curve(
        outputs["codec_token_logits"],
        outputs["codec_token_targets"],
        outputs["codec_token_mask"],
        train_prior["codec_tokens"],
        train_prior["codec_mask"],
        fig_dir / f"codec_token_topk_curve_{split_name}.png",
        split_name,
        "Codec Token Top-k Accuracy",
        "codec_token",
    )
    paths.append(path)
    rows.extend(codec_rows)
    return paths, rows


def infer_label_names(outputs: dict[str, Any]) -> list[str]:
    label_idx = np.asarray(outputs["label_idx"], dtype=np.int64)
    labels = np.asarray(outputs["labels"]).astype(str)
    n = int(max(label_idx.max(initial=0) + 1, outputs["prompt_logits"].shape[-1]))
    names = [f"label_{idx}" for idx in range(n)]
    for idx in range(n):
        vals = labels[label_idx == idx]
        if vals.size:
            unique, counts = np.unique(vals, return_counts=True)
            names[idx] = str(unique[np.argmax(counts)])
    return names


def plot_prompt_confusion(outputs: dict[str, Any], path: Path, split_name: str, label_names: list[str]) -> Path:
    y_true = np.asarray(outputs["label_idx"], dtype=np.int64)
    y_pred = np.asarray(outputs["prompt_logits"]).argmax(axis=-1).astype(np.int64)
    n = len(label_names)
    mat = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n and 0 <= p < n:
            mat[t, p] += 1
    denom = mat.sum(axis=1, keepdims=True).clip(min=1)
    norm = mat / denom
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=max(0.2, float(norm.max(initial=0.0))))
    ax.set_title(f"Prompt Confusion Matrix ({split_name})")
    ax.set_xlabel("predicted prompt")
    ax.set_ylabel("true prompt")
    ax.set_xticks(range(n), label_names, rotation=45, ha="right")
    ax.set_yticks(range(n), label_names)
    for i in range(n):
        for j in range(n):
            if mat[i, j] > 0:
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized rate")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_prompt_auc(outputs: dict[str, Any], path: Path, split_name: str, label_names: list[str]) -> tuple[Path, list[dict[str, Any]]]:
    y_true = np.asarray(outputs["label_idx"], dtype=np.int64)
    scores = softmax(np.asarray(outputs["prompt_logits"], dtype=np.float64), axis=-1)
    aucs = []
    rows = []
    for idx, name in enumerate(label_names):
        binary = (y_true == idx).astype(np.int64)
        auc = binary_auc(binary, scores[:, idx]) if binary.min(initial=0) != binary.max(initial=0) else float("nan")
        aucs.append(auc)
        if np.isfinite(auc):
            rows.append({"split": split_name, "metric": "prompt_ovr_auc", "class": name, "value": float(auc)})
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(label_names))
    ax.bar(x, [0.0 if not np.isfinite(v) else v for v in aucs], color="#4c78a8")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("one-vs-rest ROC-AUC")
    ax.set_title(f"Prompt One-vs-Rest AUC ({split_name})")
    ax.set_xticks(x, label_names, rotation=45, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path, rows


def plot_active_roc_pr(outputs: dict[str, Any], path: Path, split_name: str) -> tuple[Path, list[dict[str, Any]]]:
    target = np.asarray(outputs["target_active_mask"], dtype=np.float64)
    pred = np.asarray(outputs["pred_active_mask"], dtype=np.float64)
    if target.shape[1] != pred.shape[1]:
        target = resize_rows(target, pred.shape[1])
    y_true = (target.reshape(-1) >= 0.5).astype(np.int64)
    scores = pred.reshape(-1)
    fpr, tpr, roc_auc = roc_curve_auc(y_true, scores)
    recall, precision, ap = pr_curve_ap(y_true, scores)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, color="#4c78a8", label=f"AUC={roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("Active Mask ROC")
    axes[0].legend(frameon=False)
    axes[1].plot(recall, precision, color="#f58518", label=f"AP={ap:.3f}")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Active Mask PR")
    axes[1].legend(frameon=False)
    fig.suptitle(f"Active Speech Window Diagnostics ({split_name})")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path, [
        {"split": split_name, "metric": "active_mask_roc_auc", "value": float(roc_auc)},
        {"split": split_name, "metric": "active_mask_average_precision", "value": float(ap)},
    ]


def plot_topk_curve(
    logits: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    prior_tokens: np.ndarray,
    prior_mask: np.ndarray,
    path: Path,
    split_name: str,
    title: str,
    metric_prefix: str,
) -> tuple[Path, list[dict[str, Any]]]:
    logits = np.asarray(logits, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    mask_bool = np.asarray(mask, dtype=np.float32) > 0
    vocab = int(logits.shape[-1])
    ks = [k for k in [1, 2, 3, 5, 10, 20, 50] if k <= vocab]
    pred_order = np.argsort(logits, axis=-1)
    counts = np.bincount(np.asarray(prior_tokens, dtype=np.int64)[np.asarray(prior_mask, dtype=np.float32) > 0].clip(0, vocab - 1).reshape(-1), minlength=vocab)
    prior_order = np.argsort(counts)
    model_values, prior_values, gain_values, rows = [], [], [], []
    for k in ks:
        pred_top = pred_order[..., -k:]
        model_acc = float(np.any(pred_top[mask_bool] == targets[mask_bool, None], axis=-1).mean()) if mask_bool.any() else 0.0
        prior_top = set(prior_order[-k:].tolist())
        prior_acc = float(np.asarray([int(tok) in prior_top for tok in targets[mask_bool].reshape(-1)]).mean()) if mask_bool.any() else 0.0
        gain = model_acc - prior_acc
        model_values.append(model_acc)
        prior_values.append(prior_acc)
        gain_values.append(gain)
        rows.extend(
            [
                {"split": split_name, "metric": f"{metric_prefix}_top{k}_acc", "value": model_acc},
                {"split": split_name, "metric": f"{metric_prefix}_top{k}_prior_acc", "value": prior_acc},
                {"split": split_name, "metric": f"{metric_prefix}_top{k}_gain", "value": gain},
            ]
        )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, model_values, marker="o", label="EEG-pred")
    ax.plot(ks, prior_values, marker="o", label="train prior")
    ax.plot(ks, gain_values, marker="o", label="gain over prior")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_xlabel("k")
    ax.set_ylabel("accuracy / gain")
    ax.set_title(f"{title} ({split_name})")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path, rows


def plot_lagaware_summaries(wav_dir: Path, fig_dir: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    lag_dir = wav_dir / "waveform_compare_lagaware"
    manifests = sorted(lag_dir.glob("lagaware_manifest_*.csv"))
    if not manifests:
        return [], []
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    if not rows:
        return [], []
    metrics = [
        "zero_lag_envelope_corr",
        "best_lag_envelope_corr",
        "pred_lag_envelope_corr",
        "zero_lag_mel_corr",
        "best_lag_mel_corr",
        "pred_lag_mel_corr",
        "zero_lag_waveform_corr",
        "best_lag_waveform_corr",
        "pred_lag_waveform_corr",
    ]
    recon_types = sorted({str(row.get("reconstruction_type", "")) for row in rows if row.get("reconstruction_type")})
    means = {metric: [mean_float([row.get(metric) for row in rows if row.get("reconstruction_type") == recon]) for recon in recon_types] for metric in metrics}
    path = fig_dir / "lagaware_reconstruction_metrics.png"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    groups = [
        ("Envelope", ["zero_lag_envelope_corr", "best_lag_envelope_corr", "pred_lag_envelope_corr"]),
        ("Mel proxy", ["zero_lag_mel_corr", "best_lag_mel_corr", "pred_lag_mel_corr"]),
        ("Waveform", ["zero_lag_waveform_corr", "best_lag_waveform_corr", "pred_lag_waveform_corr"]),
    ]
    x = np.arange(len(recon_types))
    width = 0.25
    for ax, (name, keys) in zip(axes, groups):
        for offset, key in enumerate(keys):
            short = key.replace("_corr", "").replace("_lag", "").replace("_", " ")
            ax.bar(x + (offset - 1) * width, means[key], width=width, label=short)
        ax.set_title(name)
        ax.set_xticks(x, recon_types, rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("mean correlation")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Lag-aware waveform diagnostics")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    summary_rows = []
    for recon in recon_types:
        for metric in metrics:
            summary_rows.append({"split": "wav", "reconstruction_type": recon, "metric": metric, "value": mean_float([row.get(metric) for row in rows if row.get("reconstruction_type") == recon])})
    return [path], summary_rows


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-12)


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    _, _, auc = roc_curve_auc(y_true, scores)
    return auc


def roc_curve_auc(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = float((y_true == 1).sum())
    neg = float((y_true == 0).sum())
    if pos <= 0 or neg <= 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order]
    tps = np.cumsum(y == 1) / pos
    fps = np.cumsum(y == 0) / neg
    tpr = np.concatenate([[0.0], tps, [1.0]])
    fpr = np.concatenate([[0.0], fps, [1.0]])
    return fpr, tpr, float(np.trapz(tpr, fpr))


def pr_curve_ap(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = int((y_true == 1).sum())
    if pos <= 0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0]), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(pos, 1)
    ap = float((precision[y == 1]).sum() / max(pos, 1))
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return recall, precision, ap


def resize_rows(values: np.ndarray, steps: int) -> np.ndarray:
    x_new = np.linspace(0.0, 1.0, int(steps))
    out = []
    for row in values:
        x_old = np.linspace(0.0, 1.0, row.shape[0])
        out.append(np.interp(x_new, x_old, row))
    return np.asarray(out, dtype=np.float32)


def mean_float(values: list[Any]) -> float:
    vals = []
    for value in values:
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            vals.append(f)
    return float(np.mean(vals)) if vals else 0.0


def write_summary(run_dir: Path, fig_dir: Path, written: list[str], rows: list[dict[str, Any]]) -> None:
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json(fig_dir / "diagnostic_figures_manifest.json", {"figures": written, "metrics_rows": rows})
    write_json(metrics_dir / "diagnostic_metrics_summary.json", {"rows": rows})
    csv_path = metrics_dir / "diagnostic_metrics_summary.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()}) or ["metric", "value"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
