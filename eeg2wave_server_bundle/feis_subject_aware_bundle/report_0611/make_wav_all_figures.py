from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-feis")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WAV_TYPES = ("original_ref", "target_oracle", "mean_latent", "pred_unscaled", "pred_scaled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full five-way waveform/spectrogram pages for wav_all.")
    parser.add_argument("--wav-root", required=True, help="Path to wav_all directory.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults next to wav_all with timestamp.")
    parser.add_argument("--groups-per-page", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--max-pages", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--image-format", default="png", choices=["png", "jpg", "jpeg"])
    return parser.parse_args()


def parse_wav_name(path: Path) -> tuple[str, str] | None:
    stem = path.stem
    for wav_type in WAV_TYPES:
        suffix = f"_{wav_type}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], wav_type
    return None


def read_audio(path: Path) -> tuple[int, np.ndarray]:
    sr, audio = wavfile.read(str(path))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / max(float(np.iinfo(audio.dtype).max), 1.0)
    else:
        audio = audio.astype(np.float32)
    if audio.size == 0:
        audio = np.zeros(1, dtype=np.float32)
    return int(sr), audio.astype(np.float32)


def audio_stats(audio: np.ndarray) -> dict[str, float]:
    audio = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)) + 1e-12)
    peak = float(np.max(np.abs(audio)) if audio.size else 0.0)
    return {"rms": rms, "peak": peak}


def plot_audio_cell(ax, path: Path | None, title: str) -> dict[str, float] | None:
    ax.set_xticks([])
    ax.set_yticks([])
    if path is None or not path.exists():
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
        ax.set_title(title, fontsize=8)
        return None

    sr, audio = read_audio(path)
    stats = audio_stats(audio)
    nperseg = min(512, max(64, int(2 ** math.floor(math.log2(max(len(audio) // 8, 64))))))
    noverlap = int(nperseg * 0.75)
    _, times, spec = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=None, padded=False)
    mag = np.log10(np.abs(spec) + 1e-5)
    if mag.size:
        vmin, vmax = np.percentile(mag, [5, 98])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(np.nanmin(mag)), float(np.nanmax(mag) + 1e-6)
        ax.imshow(mag, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)

    # Overlay waveform shape as a white trace in spectrogram coordinates.
    if len(audio) > 1 and mag.shape[1] > 1:
        xs = np.linspace(0, mag.shape[1] - 1, min(len(audio), 600))
        idx = np.linspace(0, len(audio) - 1, len(xs)).astype(int)
        wave = audio[idx]
        denom = np.max(np.abs(wave)) + 1e-8
        y_mid = mag.shape[0] * 0.82
        y_amp = mag.shape[0] * 0.12
        ys = y_mid + (wave / denom) * y_amp
        ax.plot(xs, ys, color="white", lw=0.35, alpha=0.9)

    ax.set_title(f"{title}\nrms={stats['rms']:.3f} peak={stats['peak']:.2f}", fontsize=7)
    return stats


def split_key_sort(item: tuple[str, dict[str, Path]]) -> tuple:
    key = item[0]
    parts = key.split("_")
    subject = parts[0] if parts else ""
    trial = ""
    for part in reversed(parts):
        if part.startswith("t") and part[1:].isdigit():
            trial = part[1:].zfill(5)
            break
    return subject, "_".join(parts[1:-2]) if len(parts) > 3 else key, trial, key


def main() -> None:
    args = parse_args()
    wav_root = Path(args.wav_root).resolve()
    if not wav_root.exists():
        raise FileNotFoundError(wav_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else wav_root.parent / f"wav_all_figures_{timestamp}"
    pages_root = out_dir / "pages"
    pages_root.mkdir(parents=True, exist_ok=True)

    groups_by_split: dict[str, dict[str, dict[str, Path]]] = defaultdict(lambda: defaultdict(dict))
    for wav in wav_root.glob("*/*.wav"):
        parsed = parse_wav_name(wav)
        if parsed is None:
            continue
        key, wav_type = parsed
        groups_by_split[wav.parent.name][key][wav_type] = wav

    manifest_rows: list[dict[str, object]] = []
    summary = {
        "wav_root": str(wav_root),
        "out_dir": str(out_dir),
        "wav_types": list(WAV_TYPES),
        "groups_per_page": int(args.groups_per_page),
        "splits": {},
    }

    for split in sorted(groups_by_split):
        split_groups = sorted(groups_by_split[split].items(), key=split_key_sort)
        split_page_dir = pages_root / split
        split_page_dir.mkdir(parents=True, exist_ok=True)
        page_count = 0
        complete_groups = 0
        total_groups = len(split_groups)

        for page_start in range(0, total_groups, args.groups_per_page):
            if args.max_pages is not None and page_count >= args.max_pages:
                break
            chunk = split_groups[page_start : page_start + args.groups_per_page]
            rows = len(chunk)
            fig, axes = plt.subplots(
                rows,
                len(WAV_TYPES),
                figsize=(15.5, max(2.0, 1.65 * rows)),
                squeeze=False,
            )
            for row_idx, (key, wavs) in enumerate(chunk):
                if all(w in wavs for w in WAV_TYPES):
                    complete_groups += 1
                for col_idx, wav_type in enumerate(WAV_TYPES):
                    stats = plot_audio_cell(axes[row_idx][col_idx], wavs.get(wav_type), wav_type)
                    manifest_rows.append(
                        {
                            "split": split,
                            "group_key": key,
                            "wav_type": wav_type,
                            "path": "" if wav_type not in wavs else str(wavs[wav_type]),
                            "page": page_count + 1,
                            "rms": None if stats is None else stats["rms"],
                            "peak": None if stats is None else stats["peak"],
                        }
                    )
                axes[row_idx][0].set_ylabel(key, fontsize=7, rotation=0, labelpad=55, va="center")
            fig.suptitle(
                f"{split}: five-way wav comparison | groups {page_start + 1}-{page_start + len(chunk)} / {total_groups}",
                fontsize=12,
            )
            fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.97))
            page_count += 1
            page_path = split_page_dir / f"page_{page_count:04d}.{args.image_format}"
            save_kwargs = {"dpi": int(args.dpi)}
            if args.image_format in {"jpg", "jpeg"}:
                save_kwargs["pil_kwargs"] = {"quality": 88}
            fig.savefig(page_path, **save_kwargs)
            plt.close(fig)

        summary["splits"][split] = {
            "groups": total_groups,
            "complete_groups_seen_on_generated_pages": complete_groups,
            "pages_generated": page_count,
            "page_dir": str(split_page_dir),
        }

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "group_key", "wav_type", "path", "page", "rms", "peak"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# wav_all visual pages\n\n"
        "Each page shows five columns: `original_ref`, `target_oracle`, `mean_latent`, "
        "`pred_unscaled`, and `pred_scaled`. Each cell is a log-STFT heat map with a white waveform overlay.\n\n"
        f"- Source wav root: `{wav_root}`\n"
        f"- Manifest: `manifest.csv`\n"
        f"- Summary: `summary.json`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
