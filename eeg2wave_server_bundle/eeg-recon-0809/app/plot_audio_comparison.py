#!/usr/bin/env python3
"""Spectrogram and envelope figures for the exported reconstructions.

One row per condition (original, pipeline oracle, real EEG, zero-EEG,
wrong-trial), one column per example sentence, plus an envelope panel that
overlays real EEG and zero-EEG on the original.  Intended as the visual
companion to app/audio_comparison.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from audio_comparison import envelope, log_mel, speech_length, RATE

ROWS = [('original', 'presented sentence'), ('teacher_oracle', 'pipeline oracle (audio → decoder → vocoder)'),
        ('correct', 'real EEG'), ('zero', 'zero-EEG control'), ('wrong_trial', 'wrong-trial EEG')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', default=str(ROOT / 'outputs/aligned_recovery_v3/eval_validation_positional'))
    parser.add_argument('--output', default=str(ROOT / 'outputs/aligned_recovery_v3/audio_comparison'))
    parser.add_argument('--examples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    export = Path(args.export); index = json.loads((export / 'index.json').read_text())['samples']
    rng = np.random.default_rng(args.seed)
    chosen = [index[i] for i in sorted(rng.choice(len(index), size=args.examples, replace=False))]
    figure, axes = plt.subplots(len(ROWS) + 1, len(chosen), figsize=(4.1 * len(chosen), 1.7 * (len(ROWS) + 1)), constrained_layout=True)
    for column, entry in enumerate(chosen):
        folder = export / 'waveforms' / entry['folder']; samples = speech_length(entry)
        seconds = samples / RATE
        waves = {name: sf.read(folder / f'{name}.wav', dtype='float32')[0][:samples] for name, _ in ROWS}
        for row, (name, label) in enumerate(ROWS):
            spectrogram = log_mel(waves[name])
            axes[row, column].imshow(spectrogram, origin='lower', aspect='auto', cmap='magma',
                                     extent=[0, seconds, 0, spectrogram.shape[0]], vmin=-12, vmax=4)
            axes[row, column].set_xticks([])
            if column == 0:
                axes[row, column].set_ylabel(label, fontsize=7)
            axes[row, column].set_yticks([])
            if row == 0:
                text = entry.get('reference_transcript', '')
                axes[row, column].set_title((text[:38] + '…') if len(text) > 38 else text, fontsize=8)
        panel = axes[len(ROWS), column]
        time = np.arange(len(envelope(waves['original']))) / 100.
        for name, style in (('original', dict(color='black', lw=1.4)), ('correct', dict(color='tab:red', lw=1.1)), ('zero', dict(color='tab:blue', lw=.9, ls='--'))):
            values = envelope(waves[name]); values = values / (values.max() + 1e-9)
            panel.plot(time[:len(values)], values[:len(time)], label=name if column == 0 else None, **style)
        panel.set_xlabel('seconds', fontsize=7); panel.tick_params(labelsize=6); panel.set_ylim(0, 1.05)
        if column == 0:
            panel.set_ylabel('envelope', fontsize=7); panel.legend(fontsize=6, loc='upper right')
    figure.suptitle('DS004940 validation sentences: mel spectrograms and envelopes by condition', fontsize=10)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        figure.savefig(output / f'audio_comparison.{extension}', dpi=200)
    print(json.dumps(dict(figure=str(output / 'audio_comparison.png'), examples=[e['trial_id'] for e in chosen])))


if __name__ == '__main__':
    main()
