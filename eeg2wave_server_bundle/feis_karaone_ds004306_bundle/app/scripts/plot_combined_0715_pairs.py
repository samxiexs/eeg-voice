from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.io import wavfile  # noqa: E402


OUTPUT_MODES = (
    "codec_oracle",
    "eeg_conditioned",
    "label_only",
    "zero_eeg",
    "shuffled_eeg",
    "dataset_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reference-vs-reconstruction waveform pair plots from combined 0715 synthesis manifests."
    )
    parser.add_argument("--synthesis-root", required=True, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--limit", type=int, default=-1, help="Limit plotted pairs per manifest; -1 plots all generated samples.")
    parser.add_argument("--sample-points", type=int, default=4000)
    return parser.parse_args()


def read_wave(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, values = wavfile.read(path)
    array = np.asarray(values, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=1)
    if np.issubdtype(values.dtype, np.integer):
        scale = float(np.iinfo(values.dtype).max)
        array = array / max(scale, 1.0)
    return int(sample_rate), array.reshape(-1)


def downsample(values: np.ndarray, points: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) <= points:
        return values
    positions = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target = np.linspace(0.0, 1.0, int(points), dtype=np.float64)
    return np.interp(target, positions, values).astype(np.float32)


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in str(value))


def plot_pair(
    reference_path: Path,
    mode_paths: dict[str, Path],
    destination: Path,
    *,
    sample_key: str,
    label: str,
    sample_rate: int,
    metrics: dict[str, dict[str, float]],
    max_points: int,
) -> None:
    reference_rate, reference = read_wave(reference_path)
    if reference_rate != sample_rate:
        sample_rate = reference_rate
    reference = downsample(reference, max_points)
    time_axis = np.arange(len(reference), dtype=np.float64) / float(sample_rate)
    rows = 1 + len(mode_paths)
    fig, axes = plt.subplots(rows, 1, figsize=(15.0, max(4.0, rows * 2.0)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    axes[0].plot(time_axis, reference, color="#1f2937", linewidth=0.75)
    axes[0].set_title(f"reference | label={label} | sample={sample_key}", loc="left", fontsize=10)
    axes[0].set_ylabel("ref")
    axes[0].grid(alpha=0.2)
    for axis, (mode, path) in zip(axes[1:], mode_paths.items(), strict=True):
        candidate_rate, candidate = read_wave(path)
        candidate = downsample(candidate, max_points)
        length = min(len(reference), len(candidate))
        axis.plot(time_axis[:length], reference[:length], color="#9ca3af", linewidth=0.6, alpha=0.85, label="reference")
        axis.plot(time_axis[:length], candidate[:length], color="#2563eb", linewidth=0.7, label=mode)
        values = metrics.get(mode, {})
        annotation = (
            f"corr={values.get('waveform_correlation', float('nan')):.3f}; "
            f"SI-SDR={values.get('si_sdr_db', float('nan')):.2f} dB; "
            f"spec-MAE={values.get('log_spectrogram_mae_db', float('nan')):.2f} dB"
        )
        axis.set_title(f"{mode} | {annotation}", loc="left", fontsize=9)
        axis.set_ylabel(mode[:8])
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("time (s)")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle("Combined 0715 reference vs reconstruction pairs", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.synthesis_root.expanduser().resolve()
    output_root = (args.output or root).expanduser().resolve()
    manifests = sorted(root.glob("*/validation/synthesis_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No validation synthesis manifests found under {root}")
    total_pairs = 0
    total_plots = 0
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        destination = manifest_path.parent
        dataset = str(payload.get("dataset", destination.parent.name))
        split = str(payload.get("split", destination.name))
        pair_root = output_root / dataset / split / "comparison_pairs"
        pair_manifest = pair_root / "pair_manifest.csv"
        pair_root.mkdir(parents=True, exist_ok=True)
        rows = list(payload.get("files", []))
        if args.limit >= 0:
            rows = rows[: int(args.limit)]
        dataset_plots = 0
        dataset_pairs = 0
        with pair_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "dataset",
                    "split",
                    "sample_key",
                    "label",
                    "mode",
                    "reference_wav",
                    "reconstruction_wav",
                    "figure",
                    "waveform_correlation",
                    "si_sdr_db",
                    "log_spectrogram_mae_db",
                ),
            )
            writer.writeheader()
            for item in rows:
                files = item.get("files", {})
                reference_path = destination / str(files.get("reference", ""))
                available = {
                    mode: destination / str(files[mode])
                    for mode in OUTPUT_MODES
                    if files.get(mode)
                }
                if not reference_path.is_file() or not available:
                    continue
                sample_key = str(item.get("sample_key", "sample"))
                figure = pair_root / f"{safe_name(sample_key)}.png"
                plot_pair(
                    reference_path,
                    available,
                    figure,
                    sample_key=sample_key,
                    label=str(item.get("label", "unknown")),
                    sample_rate=int(payload.get("sample_rate_hz", 24000)),
                    metrics=item.get("mode_metrics", {}),
                    max_points=int(args.sample_points),
                )
                total_pairs += len(available)
                total_plots += 1
                dataset_pairs += len(available)
                dataset_plots += 1
                for mode, reconstruction_path in available.items():
                    values = item.get("mode_metrics", {}).get(mode, {})
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "split": split,
                            "sample_key": sample_key,
                            "label": str(item.get("label", "unknown")),
                            "mode": mode,
                            "reference_wav": str(reference_path),
                            "reconstruction_wav": str(reconstruction_path),
                            "figure": str(figure),
                            "waveform_correlation": values.get("waveform_correlation"),
                            "si_sdr_db": values.get("si_sdr_db"),
                            "log_spectrogram_mae_db": values.get("log_spectrogram_mae_db"),
                        }
                    )
        summary = {
            "dataset": dataset,
            "split": split,
            "source_manifest": str(manifest_path),
            "pair_manifest": str(pair_manifest),
            "plots_written": dataset_plots,
            "pairs_written": dataset_pairs,
            "plot_definition": "reference waveform overlaid with every reconstruction mode; metrics from synthesis manifest",
        }
        (pair_root / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(json.dumps({"manifests": len(manifests), "plots_written": total_plots, "pairs_written": total_pairs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
