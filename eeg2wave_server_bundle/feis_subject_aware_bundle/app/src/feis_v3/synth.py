from __future__ import annotations

from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from src.feis_v3.data import FEISV3AudioTokenBank
from src.utils import ensure_dir, load_wav_fixed, save_wav


def decode_generated_codec(token_bank: FEISV3AudioTokenBank, token_ids: np.ndarray) -> np.ndarray:
    return token_bank.decode_codec_tokens(token_ids)


def retrieve_diagnostic_audio(
    token_bank: FEISV3AudioTokenBank,
    pred_semantic_hist: np.ndarray,
    label: str | None = None,
) -> int:
    bank_hist = token_bank.semantic_histograms()
    candidates = list(range(len(token_bank.audio_keys)))
    if label is not None:
        label_candidates = [idx for idx in candidates if token_bank.labels[idx] == label]
        if label_candidates:
            candidates = label_candidates
    cand = np.asarray(candidates, dtype=np.int64)
    pred = pred_semantic_hist.reshape(1, -1).astype(np.float32)
    pred = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-8)
    target = bank_hist[cand]
    target = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-8)
    return int(cand[(pred @ target.T).argmax(axis=1)[0]])


def write_grouped_triplet(
    out_root: str | Path,
    subject_id: str,
    label: str,
    trial_index: int,
    reference: np.ndarray,
    retrieval: np.ndarray,
    generated: np.ndarray,
    sample_rate: int,
) -> dict[str, str]:
    group_dir = ensure_dir(Path(out_root) / "wavs" / "grouped_wavs" / "by_subject" / subject_id / label / f"trial_{int(trial_index):04d}")
    files = {
        "original_reference": group_dir / "01_original_reference.wav",
        "retrieval_diagnostic": group_dir / "02_retrieval_diagnostic.wav",
        "generated_codec": group_dir / "03_generated_codec.wav",
    }
    save_wav(files["original_reference"], reference, sample_rate)
    save_wav(files["retrieval_diagnostic"], retrieval, sample_rate)
    save_wav(files["generated_codec"], generated, sample_rate)
    return {key: str(path) for key, path in files.items()}


def write_audio_comparison_figure(
    path: str | Path,
    reference: np.ndarray,
    comparison: np.ndarray,
    sample_rate: int,
    title: str,
) -> str:
    """Write a compact waveform + spectrogram comparison PNG."""
    path = Path(path)
    ensure_dir(path.parent)
    reference = np.asarray(reference, dtype=np.float32)
    comparison = np.asarray(comparison, dtype=np.float32)
    n = min(reference.shape[0], comparison.shape[0])
    reference = reference[:n]
    comparison = comparison[:n]
    t = np.arange(n, dtype=np.float32) / float(sample_rate)

    f_ref, tt_ref, s_ref = spectrogram(reference, fs=sample_rate, nperseg=512, noverlap=384)
    f_cmp, tt_cmp, s_cmp = spectrogram(comparison, fs=sample_rate, nperseg=512, noverlap=384)
    s_ref_db = 10.0 * np.log10(np.maximum(s_ref, 1e-10))
    s_cmp_db = 10.0 * np.log10(np.maximum(s_cmp, 1e-10))

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(t, reference, lw=0.9, label="original_reference", color="#1f77b4")
    axes[0].plot(t, comparison, lw=0.8, label="comparison", color="#d62728", alpha=0.75)
    axes[0].set_title(title)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].pcolormesh(tt_ref, f_ref, s_ref_db, shading="auto", cmap="magma")
    axes[1].set_ylim(0, min(8000, sample_rate // 2))
    axes[1].set_ylabel("Hz")
    axes[1].set_title("original_reference spectrogram")

    axes[2].pcolormesh(tt_cmp, f_cmp, s_cmp_db, shading="auto", cmap="magma")
    axes[2].set_ylim(0, min(8000, sample_rate // 2))
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Hz")
    axes[2].set_title("comparison spectrogram")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def write_waveform_contact_sheet(path: str | Path, rows: list[dict[str, str]], generated: bool = True) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    gen_label = "generated_codec" if generated else "retrieval_diagnostic"
    fig_key = "generated_codec_figure" if generated else "retrieval_diagnostic_figure"
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<style>body{font-family:Arial,sans-serif;margin:24px} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:6px 8px;font-size:13px} img{max-width:560px}</style>",
        f"<title>FEIS v3 original vs {escape(gen_label)}</title></head><body>",
        f"<h1>FEIS v3 original vs {escape(gen_label)}</h1>",
        "<table><thead><tr><th>sample</th><th>subject</th><th>label</th><th>stage</th><th>original</th><th>comparison</th><th>figure</th></tr></thead><tbody>",
    ]
    for row in rows:
        comp = row["generated_codec"] if generated else row["retrieval_diagnostic"]
        fig = row.get(fig_key, "")
        fig_html = f"<a href='{escape(fig)}'><img src='{escape(fig)}'></a>" if fig else ""
        parts.append(
            "<tr>"
            f"<td>{escape(row['sample_key'])}</td>"
            f"<td>{escape(row['subject_id'])}</td>"
            f"<td>{escape(row['label'])}</td>"
            f"<td>{escape(row['stage'])}</td>"
            f"<td>{escape(row['original_reference'])}</td>"
            f"<td>{escape(comp)}</td>"
            f"<td>{fig_html}</td>"
            "</tr>"
        )
    parts.append("</tbody></table></body></html>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def load_reference_audio(feis_root: Path, rel_path: str, sample_rate: int, duration_sec: float) -> np.ndarray:
    return load_wav_fixed(
        feis_root / rel_path,
        sample_rate=sample_rate,
        n_samples=int(round(sample_rate * float(duration_sec))),
        normalize="rms",
        target_rms=0.08,
        max_gain=12.0,
    )
