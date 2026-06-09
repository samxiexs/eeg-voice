from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from src.utils import ensure_dir, load_simple_yaml, resolve_bundle_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze FEIS speech-embedding template space.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "alignment_ssl_local.yaml"))
    parser.add_argument("--target-cache", default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def load_target_cache(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    summaries = (
        payload["target_summaries"].astype(np.float32)
        if "target_summaries" in payload.files
        else payload["speech_embeddings"].astype(np.float32)
    )
    return {
        "template_ids": payload["template_ids"].astype(str),
        "subject_ids": payload["subject_ids"].astype(str),
        "labels": payload["labels"].astype(str),
        "audio_paths": payload["audio_paths"].astype(str),
        "speech_embeddings": summaries,
        "target_kind": str(payload["target_kind"].item()) if "target_kind" in payload.files else "hubert_pooled",
        "target_sequences": payload["target_sequences"].astype(np.float32) if "target_sequences" in payload.files else None,
        "prosody_targets": payload["prosody_targets"].astype(np.float32),
        "feature_backend": payload["feature_backend"].astype(str),
    }


def pca_projection(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :n_components] * s[:n_components]
    return coords.astype(np.float32)


def plot_projection(coords: np.ndarray, groups: np.ndarray, title: str, output_path: Path) -> None:
    unique_groups = sorted(set(groups.tolist()))
    cmap = plt.get_cmap("tab20", max(len(unique_groups), 1))
    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, group in enumerate(unique_groups):
        mask = groups == group
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=42,
            alpha=0.8,
            color=cmap(idx),
            label=str(group),
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    if len(unique_groups) <= 24:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.clip(norms, 1e-8, None)
    sims = normalized @ normalized.T
    return (1.0 - sims).astype(np.float32)


def summarize_distances(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float32)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def bucket_pairwise_distances(
    template_ids: np.ndarray,
    subject_ids: np.ndarray,
    labels: np.ndarray,
    distances: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    num_templates = len(template_ids)
    for i in range(num_templates):
        for j in range(i + 1, num_templates):
            same_subject = subject_ids[i] == subject_ids[j]
            same_label = labels[i] == labels[j]
            value = float(distances[i, j])
            if same_label:
                buckets["within_label"].append(value)
            if same_subject:
                buckets["within_subject"].append(value)
            if same_label and not same_subject:
                buckets["cross_subject_same_label"].append(value)
            if same_subject and not same_label:
                buckets["cross_label_same_subject"].append(value)
            if (not same_subject) and (not same_label):
                buckets["global_cross"].append(value)
    return {name: summarize_distances(values) for name, values in buckets.items()}


def leave_one_out_centroid_probe(embeddings: np.ndarray, group_ids: np.ndarray) -> dict[str, object]:
    unique_groups = sorted(set(group_ids.tolist()))
    if not unique_groups:
        return {"accuracy": 0.0, "num_classes": 0}

    members: dict[str, np.ndarray] = {}
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for group in unique_groups:
        mask = group_ids == group
        group_embeddings = embeddings[mask]
        members[group] = group_embeddings
        sums[group] = group_embeddings.sum(axis=0)
        counts[group] = int(group_embeddings.shape[0])

    normalized_embeddings = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8, None)
    correct = 0
    for idx, group in enumerate(group_ids.tolist()):
        centroids: list[np.ndarray] = []
        centroid_ids: list[str] = []
        for centroid_group in unique_groups:
            centroid_sum = sums[centroid_group].copy()
            centroid_count = counts[centroid_group]
            if centroid_group == group and centroid_count > 1:
                centroid_sum = centroid_sum - embeddings[idx]
                centroid_count -= 1
            centroid = centroid_sum / max(centroid_count, 1)
            centroid_ids.append(centroid_group)
            centroids.append(centroid)
        centroid_matrix = np.stack(centroids, axis=0)
        centroid_matrix = centroid_matrix / np.clip(np.linalg.norm(centroid_matrix, axis=1, keepdims=True), 1e-8, None)
        sims = centroid_matrix @ normalized_embeddings[idx]
        pred_group = centroid_ids[int(np.argmax(sims))]
        correct += int(pred_group == group)
    return {
        "accuracy": float(correct / max(len(group_ids), 1)),
        "num_classes": len(unique_groups),
    }


def compute_space_summary(payload: dict[str, np.ndarray]) -> dict[str, object]:
    embeddings = payload["speech_embeddings"]
    template_ids = payload["template_ids"]
    subject_ids = payload["subject_ids"]
    labels = payload["labels"]
    distances = cosine_distance_matrix(embeddings)
    pairwise = bucket_pairwise_distances(template_ids, subject_ids, labels, distances)
    subject_probe = leave_one_out_centroid_probe(embeddings, subject_ids)
    label_probe = leave_one_out_centroid_probe(embeddings, labels)
    within_label_mean = pairwise.get("within_label", {}).get("mean")
    within_subject_mean = pairwise.get("within_subject", {}).get("mean")
    cross_subject_same_label_mean = pairwise.get("cross_subject_same_label", {}).get("mean")
    global_cross_mean = pairwise.get("global_cross", {}).get("mean")
    if within_label_mean is None or within_subject_mean is None:
        dominant_factor = "undetermined"
    elif within_label_mean < within_subject_mean and (
        global_cross_mean is None or within_label_mean < global_cross_mean
    ):
        dominant_factor = "label"
    elif within_subject_mean < within_label_mean:
        dominant_factor = "subject"
    else:
        dominant_factor = "both"
    return {
        "num_templates": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "target_kind": str(payload.get("target_kind", "hubert_pooled")),
        "num_subjects": int(len(set(subject_ids.tolist()))),
        "num_labels": int(len(set(labels.tolist()))),
        "feature_backend": str(payload["feature_backend"][0]),
        "pairwise_cosine_distance": pairwise,
        "subject_centroid_probe": subject_probe,
        "label_centroid_probe": label_probe,
        "dominant_structure": dominant_factor,
        "same_label_more_compact_than_same_subject": None
        if within_label_mean is None or within_subject_mean is None
        else bool(within_label_mean < within_subject_mean),
        "cross_subject_same_label_mean": cross_subject_same_label_mean,
    }


def main() -> None:
    args = parse_args()
    config = load_simple_yaml(args.config)
    target_cache_path = resolve_bundle_path(args.target_cache or config["targets"]["cache_path"], BUNDLE_DIR)
    payload = load_target_cache(target_cache_path)
    output_root = resolve_bundle_path(args.output_root or config["output"]["root"], BUNDLE_DIR) / "template_space"
    cache_stem = Path(target_cache_path).stem
    output_dir = ensure_dir(output_root / cache_stem)

    coords = pca_projection(payload["speech_embeddings"], n_components=2)
    plot_projection(
        coords=coords,
        groups=payload["subject_ids"],
        title="Template Space PCA by Subject",
        output_path=output_dir / "pca_by_subject.png",
    )
    plot_projection(
        coords=coords,
        groups=payload["labels"],
        title="Template Space PCA by Label",
        output_path=output_dir / "pca_by_label.png",
    )

    summary = compute_space_summary(payload)
    summary["target_cache_path"] = str(target_cache_path)
    summary["outputs"] = {
        "pca_by_subject": str(output_dir / "pca_by_subject.png"),
        "pca_by_label": str(output_dir / "pca_by_label.png"),
    }
    write_json(output_dir / "space_summary.json", summary)
    (output_dir / "space_summary.md").write_text(
        "\n".join(
            [
                "# FEIS Alignment Space Summary",
                "",
                f"- target cache: `{target_cache_path}`",
                f"- templates: `{summary['num_templates']}`",
                f"- embedding dim: `{summary['embedding_dim']}`",
                f"- feature backend: `{summary['feature_backend']}`",
                f"- dominant structure: `{summary['dominant_structure']}`",
                f"- subject centroid probe accuracy: `{summary['subject_centroid_probe']['accuracy']:.4f}`",
                f"- label centroid probe accuracy: `{summary['label_centroid_probe']['accuracy']:.4f}`",
                "",
                "## Pairwise Cosine Distance",
                "",
                json.dumps(summary["pairwise_cosine_distance"], ensure_ascii=False, indent=2),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved template-space analysis to {output_dir}")


if __name__ == "__main__":
    main()
