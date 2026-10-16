#!/usr/bin/env python3
"""Render the Di Liberto MIDI stimuli to 24 kHz audio aligned to the EEG stimulus clock.

The dataset ships the ten MIDI files but not the audio that was played, so
the audio is re-rendered: with FluidSynth and a piano SoundFont when
available (``--synth fluidsynth``), otherwise with a plain additive piano-like
synthesiser (``--synth additive``; exact onsets and pitch, approximate
timbre).

Nothing is assumed about which MIDI file is which piece or about timing.
Every rendering's 64 Hz amplitude envelope is cross-correlated (lags up to
``--max-lag-s``) with the envelope vector the dataset provides for every
piece; MIDI files are assigned to pieces by maximum total correlation
(Hungarian), each assignment must reach ``--min-r``, and the audio is shifted
by the best lag so that sample 0 is the stimulus onset of the EEG trials.
The report (alignment.json) records r, lag and durations per piece.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src'))
from music import parse_midi, render_additive

ENVELOPE_RATE = 64


def render_fluidsynth(midi, soundfont, sample_rate, gain=.6):
    import soundfile as sf
    if shutil.which('fluidsynth') is None:
        raise SystemExit('fluidsynth not found: `brew install fluid-synth`, or use --synth additive')
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder) / 'render.wav'
        subprocess.run(['fluidsynth', '-ni', '-g', str(gain), '-r', str(sample_rate), '-F', str(out), str(soundfont), str(midi)],
                       check=True, capture_output=True)
        wave, rate = sf.read(out, dtype='float32', always_2d=True)
    wave = wave.mean(1)
    if rate != sample_rate:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(rate, sample_rate)
        wave = resample_poly(wave, sample_rate // g, rate // g).astype(np.float32)
    return wave


def rms_envelope(wave, sample_rate):
    hop = sample_rate // ENVELOPE_RATE
    frames = len(wave) // hop
    power = np.square(wave[:frames * hop].astype(np.float64)).reshape(frames, hop).mean(1)
    return np.sqrt(power)


def best_lag(rendered, reference, max_lag):
    """(r, lag in frames) maximising Pearson r of rendered[t + lag] against reference[t]."""
    reference = (reference - reference.mean()) / (reference.std() + 1e-12)
    best = (-2., 0)
    for lag in range(-max_lag, max_lag + 1):
        a = rendered[max(lag, 0):]; b = reference[max(-lag, 0):]
        n = min(len(a), len(b))
        if n < ENVELOPE_RATE * 10:
            continue
        a = a[:n]; b = b[:n]
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else -2.
        if r > best[0]:
            best = (r, lag)
    return best


def main():
    import soundfile as sf
    from scipy.optimize import linear_sum_assignment
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--midi', default=str(ROOT / 'data/diliberto2020/midi'), help='folder containing audio1.mid ... audio10.mid')
    parser.add_argument('--stimulus', default=str(ROOT / 'artifacts/music/diliberto2020/stimulus.h5'))
    parser.add_argument('--output', default=str(ROOT / 'artifacts/music/diliberto2020/audio'))
    parser.add_argument('--synth', choices=['fluidsynth', 'additive'], default='fluidsynth')
    parser.add_argument('--soundfont', default=str(ROOT / 'models/soundfonts/MuseScore_General.sf2'))
    parser.add_argument('--sample-rate', type=int, default=24000)
    parser.add_argument('--max-lag-s', type=float, default=2.)
    parser.add_argument('--min-r', type=float, default=.5)
    args = parser.parse_args()
    midis = sorted(Path(args.midi).rglob('*.mid'))
    if not midis:
        raise SystemExit(f'no .mid files under {args.midi}; run scripts/download_music.sh data first')
    if args.synth == 'fluidsynth' and not Path(args.soundfont).is_file():
        raise SystemExit(f'SoundFont {args.soundfont} missing: scripts/download_music.sh soundfont, or --synth additive')
    with h5py.File(args.stimulus, 'r') as h5:
        pieces = sorted(h5['pieces'].keys(), key=int)
        references = {int(p): h5['pieces'][p][h5['pieces'][p].attrs['envelope_feature']][:].astype(np.float64) for p in pieces}
        durations = {int(p): float(h5['pieces'][p].attrs['duration_s']) for p in pieces}
        if float(h5.attrs['fs']) != ENVELOPE_RATE:
            raise SystemExit(f'stimulus features are at {h5.attrs["fs"]} Hz, expected {ENVELOPE_RATE}')
    if len(midis) != len(references):
        raise SystemExit(f'{len(midis)} MIDI files for {len(references)} pieces')
    renders = {}
    for midi in midis:
        wave = (render_fluidsynth(midi, args.soundfont, args.sample_rate) if args.synth == 'fluidsynth'
                else render_additive(parse_midi(midi), args.sample_rate))
        renders[midi.name] = wave
        print(f'rendered {midi.name}: {len(wave) / args.sample_rate:.1f} s', flush=True)
    names = sorted(renders); ids = sorted(references)
    max_lag = int(args.max_lag_s * ENVELOPE_RATE)
    score = np.full((len(names), len(ids)), -2.); lags = np.zeros_like(score, dtype=int)
    for i, name in enumerate(names):
        env = rms_envelope(renders[name], args.sample_rate)
        for j, piece in enumerate(ids):
            score[i, j], lags[i, j] = best_lag(env, references[piece], max_lag)
    rows, cols = linear_sum_assignment(-score)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    report, failures = {}, []
    for i, j in zip(rows, cols):
        name, piece = names[i], ids[j]
        lag_s = lags[i, j] / ENVELOPE_RATE
        wave = renders[name]
        shift = int(round(lag_s * args.sample_rate))
        # rendered[t + lag] matches reference[t]: drop (or pad) the first ``lag`` of the rendering.
        wave = wave[shift:] if shift >= 0 else np.concatenate([np.zeros(-shift, np.float32), wave])
        length = int(round(durations[piece] * args.sample_rate))
        wave = wave[:length] if len(wave) >= length else np.concatenate([wave, np.zeros(length - len(wave), np.float32)])
        rms = float(np.sqrt(np.mean(np.square(wave)))) or 1.
        wave = np.clip(wave * (.1 / rms), -.99, .99).astype(np.float32)
        path = out / f'piece_{piece:02d}.wav'
        sf.write(path, wave, args.sample_rate, subtype='FLOAT')
        report[piece] = dict(midi=name, r=float(score[i, j]), lag_s=float(lag_s), second_best_r=float(np.sort(score[i])[-2]),
                             rendered_s=len(renders[name]) / args.sample_rate, stimulus_s=durations[piece], audio=str(path))
        if score[i, j] < args.min_r:
            failures.append(piece)
    (out / 'alignment.json').write_text(json.dumps(dict(synth=args.synth, sample_rate=args.sample_rate,
                                                        soundfont=args.soundfont if args.synth == 'fluidsynth' else None,
                                                        min_r=args.min_r, pieces=report), indent=1))
    for piece, item in sorted(report.items()):
        print(f'piece {piece:2d} <- {item["midi"]:14s} r={item["r"]:.3f} (next best {item["second_best_r"]:.3f}) lag={item["lag_s"]:+.3f} s', flush=True)
    if failures:
        raise SystemExit(f'alignment below r={args.min_r} for pieces {failures}; inspect {out / "alignment.json"}')


if __name__ == '__main__':
    main()
