from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BUNDLE_DIR = Path(__file__).resolve().parents[2]
DATA_ROOT = BUNDLE_DIR / "data" / "karaone"
DEFAULT_OUT_DIR = BUNDLE_DIR / "reports" / "figures" / "karaone_current_examples"


def _load_subject(subject: str) -> dict[str, np.ndarray]:
    return dict(np.load(DATA_ROOT / "subjects" / f"{subject}.npz", allow_pickle=True))


def _trial_positions(payload: dict[str, np.ndarray]) -> dict[int, int]:
    return {int(trial): idx for idx, trial in enumerate(payload["trial_indices"].astype(int).tolist())}


def _read_wav(path: Path, target_sec: float = 2.0) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = int(handle.getframerate())
        n_channels = int(handle.getnchannels())
        sample_width = int(handle.getsampwidth())
        frames = handle.readframes(handle.getnframes())
    if sample_width == 2:
        values = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        values = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        values = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        values = (values - 128.0) / 128.0
    if n_channels > 1:
        values = values.reshape(-1, n_channels).mean(axis=1)
    target_len = int(round(float(target_sec) * sample_rate))
    if values.shape[0] < target_len:
        values = np.pad(values, (0, target_len - values.shape[0]))
    else:
        values = values[:target_len]
    return values.astype(np.float32), sample_rate


def _moving_rms(values: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    squared = np.square(values.astype(np.float32))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.sqrt(np.convolve(squared, kernel, mode="same"))


def _select_trials(subject: str, label: str, n_trials: int, stage: str) -> list[int]:
    rows: list[dict[str, str]] = []
    with (DATA_ROOT / "segments.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["subject_id"] == subject and row["label"] == label and row["segment_stage"] == stage:
                rows.append(row)
    trials = sorted(int(row["trial_index"]) for row in rows)
    if len(trials) < n_trials:
        raise ValueError(f"Need {n_trials} trials for {subject} {label} {stage}, found {len(trials)}")
    center = len(trials) // 2
    start = max(0, center - n_trials // 2)
    return trials[start : start + n_trials]


def _next_trial_index(payload: dict[str, np.ndarray], trial: int) -> int:
    trials = payload["trial_indices"].astype(int).tolist()
    pos = trials.index(int(trial))
    if pos + 1 >= len(trials):
        raise ValueError(f"Trial {trial} has no following trial for next-trial clearing/resting segment")
    return int(trials[pos + 1])


def _available_channel_indices(channel_names: list[str], wanted: list[str]) -> list[int]:
    lookup = {name.upper(): idx for idx, name in enumerate(channel_names)}
    indices = [lookup[name.upper()] for name in wanted if name.upper() in lookup]
    if not indices:
        return list(range(min(6, len(channel_names))))
    return indices


def _stage_segment(payload: dict[str, np.ndarray], stage: str, trial: int, positions: dict[int, int]) -> tuple[np.ndarray, int]:
    stage_key = f"stage__{stage}"
    valid_key = f"{stage_key}__valid_lengths"
    pos = positions[int(trial)]
    eeg = payload[stage_key][pos].astype(np.float32)
    valid = int(payload[valid_key][pos]) if valid_key in payload else eeg.shape[1]
    return eeg[:, :valid], valid


def _plot_eeg_comparison(
    *,
    examples: list[tuple[str, int]],
    label: str,
    payloads: dict[str, dict[str, np.ndarray]],
    out_path: Path,
    channels: list[str],
) -> None:
    first_payload = payloads[examples[0][0]]
    channel_names = [str(item) for item in first_payload["channel_names"].tolist()]
    channel_indices = _available_channel_indices(channel_names, channels)
    selected_names = [channel_names[idx] for idx in channel_indices]
    sfreq = float(first_payload["eeg_sfreq_hz"].reshape(-1)[0])

    stage_order = [
        ("clearing", "rest / clearing", "current"),
        ("stimulus_like", "stimulus cue", "current"),
        ("thinking", "thinking", "current"),
        ("overt_like", "speaking", "current"),
        ("clearing", "next rest / clearing", "next"),
    ]
    stage_colors = {
        "clearing": "#e9ecef",
        "stimulus_like": "#fff3bf",
        "thinking": "#edf6f9",
        "overt_like": "#ffe3e3",
    }
    colors = ["#264653", "#2a9d8f", "#e76f51", "#6a4c93", "#457b9d", "#bc6c25", "#606c38"]

    prepared: list[tuple[str, int, list[tuple[str, str, np.ndarray]], np.ndarray]] = []
    max_total_len = 0
    for subject, trial in examples:
        payload = payloads[subject]
        positions = _trial_positions(payload)
        next_trial = _next_trial_index(payload, int(trial))
        pieces: list[tuple[str, str, np.ndarray]] = []
        for stage_name, display_name, trial_source in stage_order:
            source_trial = next_trial if trial_source == "next" else int(trial)
            eeg, _valid = _stage_segment(payload, stage_name, int(source_trial), positions)
            pieces.append((stage_name, display_name, eeg[channel_indices]))
        concat = np.concatenate([piece for _stage_name, _display_name, piece in pieces], axis=1)
        mean = np.mean(concat, axis=1, keepdims=True)
        std = np.std(concat, axis=1, keepdims=True) + 1e-6
        z = np.clip((concat - mean) / std, -2.5, 2.5)
        max_total_len = max(max_total_len, int(z.shape[1]))
        prepared.append((subject, int(trial), pieces, z))

    fig, axes = plt.subplots(len(prepared), 1, figsize=(14.0, 8.8), sharex=True, constrained_layout=True)
    if len(prepared) == 1:
        axes = [axes]
    offsets = np.arange(len(channel_indices), dtype=np.float32) * 4.0
    for ax, (subject, trial, pieces, z) in zip(axes, prepared, strict=True):
        time = np.arange(z.shape[1], dtype=np.float32) / sfreq
        cursor = 0
        for stage_name, display_name, piece in pieces:
            start = cursor / sfreq
            end = (cursor + piece.shape[1]) / sfreq
            ax.axvspan(start, end, color=stage_colors[stage_name], alpha=0.72, linewidth=0)
            ax.axvline(end, color="#495057", linewidth=0.9, alpha=0.75)
            ax.text(
                (start + end) / 2.0,
                offsets[-1] + 2.35,
                display_name,
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#212529",
            )
            cursor += piece.shape[1]
        for row, (name, offset) in enumerate(zip(selected_names, offsets, strict=True)):
            ax.plot(time, z[row] + offset, color=colors[row % len(colors)], linewidth=0.8, label=name)
        ax.set_yticks(offsets)
        ax.set_yticklabels(selected_names)
        ax.set_ylim(-2.8, offsets[-1] + 3.3)
        ax.set_xlim(0.0, max_total_len / sfreq)
        ax.set_ylabel("EEG channels")
        ax.set_title(f"{subject}  trial {trial}  label {label}", fontsize=10.8, weight="bold")
        ax.grid(True, axis="x", color="#dee2e6", linewidth=0.8)
        ax.grid(False, axis="y")
    axes[-1].set_xlabel("Trial time in original experimental order (s)")
    axes[0].legend(loc="upper right", ncol=len(selected_names), frameon=False, fontsize=8)
    fig.suptitle(
        f"Same-label EEG comparison: rest -> stimulus -> thinking -> speaking -> next rest ({label})",
        fontsize=14,
        weight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_wave_comparison(*, examples: list[tuple[str, int]], label: str, out_path: Path) -> None:
    waves = []
    rates = []
    for subject, trial in examples:
        wav_path = DATA_ROOT / "audio" / subject / f"{trial:03d}.wav"
        values, sample_rate = _read_wav(wav_path)
        waves.append(values)
        rates.append(sample_rate)
    y_lim = max(float(np.max(np.abs(w))) for w in waves)
    y_lim = max(y_lim, 0.05)

    fig, axes = plt.subplots(len(examples), 1, figsize=(14.0, 8.2), sharex=True, constrained_layout=True)
    if len(examples) == 1:
        axes = [axes]
    for ax, (subject, trial), values, sample_rate in zip(axes, examples, waves, rates, strict=True):
        t = np.arange(values.shape[0], dtype=np.float32) / float(sample_rate)
        env = _moving_rms(values, int(round(sample_rate * 0.01)))
        ax.plot(t, values, color="#234c7c", linewidth=0.75, label="waveform")
        ax.plot(t, env, color="#c44e52", linewidth=1.2, alpha=0.9, label="10 ms RMS envelope")
        ax.plot(t, -env, color="#c44e52", linewidth=1.2, alpha=0.9)
        ax.set_ylim(-1.08 * y_lim, 1.08 * y_lim)
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{subject}  trial {trial}  label {label}  overt/reference wav", fontsize=10.8, weight="bold")
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
    axes[-1].set_xlabel("Time in audio window (s)")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle(
        f"Same-label overt waveform targets across subjects/trials: label {label}, 16000 Hz x 2 s",
        fontsize=14,
        weight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot KaraOne report examples for same-label trial variability.")
    parser.add_argument("--subjects", nargs=2, default=("MM14", "MM18"))
    parser.add_argument("--label", default="/tiy/")
    parser.add_argument("--stage", default="thinking")
    parser.add_argument("--n-trials", type=int, default=2)
    parser.add_argument("--channels", nargs="+", default=("FZ", "C3", "CZ", "PZ"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest_rows: list[dict[str, str | int]] = []
    safe_label = args.label.replace("/", "")
    payloads = {subject: _load_subject(subject) for subject in args.subjects}
    examples: list[tuple[str, int]] = []
    eeg_path = out_dir / f"same_label_{safe_label}_complete_trial_eeg_subject_comparison.png"
    wav_path = out_dir / f"same_label_{safe_label}_overt_waveform_subject_comparison.png"
    for subject in args.subjects:
        trials = _select_trials(subject, args.label, args.n_trials, args.stage)
        examples.extend((subject, int(trial)) for trial in trials)
        for trial in trials:
            manifest_rows.append(
                {
                    "subject": subject,
                    "label": args.label,
                    "stage_order": "clearing,stimulus_like,thinking,overt_like,next_trial_clearing",
                    "trial_index": int(trial),
                    "next_resting_trial_index": int(_next_trial_index(payloads[subject], int(trial))),
                    "eeg_channels": ",".join(args.channels),
                    "eeg_figure": str(eeg_path),
                    "wav_figure": str(wav_path),
                    "audio_path": str(DATA_ROOT / "audio" / subject / f"{trial:03d}.wav"),
                }
            )

    _plot_eeg_comparison(
        examples=examples,
        label=args.label,
        payloads=payloads,
        out_path=eeg_path,
        channels=list(args.channels),
    )
    _plot_wave_comparison(examples=examples, label=args.label, out_path=wav_path)
    print(f"wrote {eeg_path}")
    print(f"wrote {wav_path}")

    manifest_path = out_dir / "karaone_report_trial_examples_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject",
                "label",
                "stage_order",
                "trial_index",
                "next_resting_trial_index",
                "eeg_channels",
                "eeg_figure",
                "wav_figure",
                "audio_path",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
