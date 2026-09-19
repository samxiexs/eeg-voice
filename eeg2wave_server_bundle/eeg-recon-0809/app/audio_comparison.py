#!/usr/bin/env python3
"""Do the reconstructed waveforms actually sound like the presented sentence?

Metric-level claims (mel MAE, envelope correlation) say nothing about audio.
This script scores the exported WAVs directly, always against the same
counterfactual conditions, and always with the two oracle chains that bound
what the frozen decoder and vocoder can possibly deliver:

  original.wav            the presented stimulus
  native_mel_oracle.wav   the stimulus' own mel through the vocoder  (vocoder ceiling)
  teacher_oracle.wav      audio -> HuBERT -> frozen decoder -> vocoder (pipeline ceiling)
  correct.wav             real EEG
  zero / wrong_trial / time_block_shuffle / channel_shuffle   counterfactuals

Objective measures (all implemented here, no extra packages):

* **STOI** (Taal et al. 2011): short-time objective intelligibility, the
  standard predictor of word intelligibility. 1.0 = perfect, ~0 = no relation.
* **MCD**: mel-cepstral distortion in dB, the standard speech-synthesis
  distance (lower is better; < 5 dB is "close", > 8 dB is "different").
* **Broadband envelope correlation** and the **modulation-spectrum
  correlation** (0.5-16 Hz), which is what EEG can plausibly carry.
* **2AFC machine listening test**: for every trial, is the reconstruction
  closer to its own original than to a random other sentence?  Chance is 50%
  and no oracle information is used, so this is the audio-domain analogue of
  a two-alternative forced-choice listening test.

Nothing here can replace a human listening test; `--listening-test` writes a
randomised forced-choice bundle (audio + answer key + scoring script) for one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
from scipy.fft import dct
from scipy.signal import get_window, resample_poly, stft

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('correct', 'zero', 'wrong_trial', 'time_block_shuffle', 'channel_shuffle', 'native_mel_oracle', 'teacher_oracle')
RATE = 16000


try:                                   # reference implementations when installed
    from pystoi import stoi as _reference_stoi
except Exception:
    _reference_stoi = None
try:
    from pesq import pesq as _reference_pesq
except Exception:
    _reference_pesq = None


# ----------------------------------------------------------------------------- STOI
def third_octave_matrix(fft_size: int, rate: int, bands: int = 15, first_centre: float = 150.):
    """15 one-third-octave bands from 150 Hz (Taal et al. 2011, Table 1)."""
    freqs = np.linspace(0, rate / 2, fft_size // 2 + 1)
    centres = first_centre * 2 ** (np.arange(bands) / 3)
    lower = centres / 2 ** (1 / 6); upper = centres * 2 ** (1 / 6)
    matrix = np.zeros((bands, len(freqs)))
    for b in range(bands):
        matrix[b] = ((freqs >= lower[b]) & (freqs < upper[b])).astype(float)
    return matrix


def remove_silent_frames(a, b, dynamic_range_db=40., frame=256, hop=128):
    window = get_window('hann', frame, fftbins=True)
    count = 1 + (len(a) - frame) // hop
    frames = np.arange(count)[:, None] * hop + np.arange(frame)[None, :]
    energy = 20 * np.log10(np.linalg.norm(a[frames] * window, axis=1) / np.sqrt(frame) + 1e-12)
    keep = energy > energy.max() - dynamic_range_db
    if keep.sum() < 2:
        return a, b
    def rebuild(x):
        out = np.zeros(len(a))
        for i, index in enumerate(np.flatnonzero(keep)):
            out[i * hop: i * hop + frame] += x[frames[index]] * window
        return out[:keep.sum() * hop + frame - hop]
    return rebuild(a), rebuild(b)


def stoi(reference: np.ndarray, degraded: np.ndarray, rate: int = RATE, beta_db: float = -15., segment: int = 30) -> float:
    """Short-time objective intelligibility (Taal et al. 2011).

    Uses the reference ``pystoi`` implementation when it is installed; the
    local fallback below follows the same specification but is slightly
    optimistic for unrelated signals, so a mixture of the two must not be
    compared.  Note the floor: STOI of two *unrelated* sentences is about
    0.47, not 0, so only differences against the counterfactual conditions
    are meaningful.
    """
    if _reference_stoi is not None:
        try:
            return float(_reference_stoi(reference, degraded, rate))
        except Exception:
            return float('nan')
    reference = resample_poly(reference, 10000, rate); degraded = resample_poly(degraded, 10000, rate)
    reference, degraded = remove_silent_frames(reference, degraded)
    if len(reference) < 512:
        return float('nan')
    kwargs = dict(fs=10000, window='hann', nperseg=256, noverlap=128, nfft=512, boundary=None, padded=False)
    _, _, x = stft(reference, **kwargs); _, _, y = stft(degraded, **kwargs)
    bands = third_octave_matrix(512, 10000)
    X = np.sqrt(bands @ np.abs(x) ** 2); Y = np.sqrt(bands @ np.abs(y) ** 2)
    if X.shape[1] < segment:
        return float('nan')
    beta = 10 ** (beta_db / 20)
    values = []
    for m in range(segment - 1, X.shape[1]):
        a = X[:, m - segment + 1: m + 1]; b = Y[:, m - segment + 1: m + 1]
        scale = np.linalg.norm(a, axis=1, keepdims=True) / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        b = np.minimum(b * scale, a * (1 + beta))
        a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
        values.append(((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)))
    return float(np.mean(values))


# ----------------------------------------------------------------------------- other measures
def log_mel(x: np.ndarray, rate: int = RATE, n_fft: int = 512, hop: int = 160, bands: int = 40):
    _, _, spectrum = stft(x, fs=rate, window='hann', nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft, boundary=None, padded=False)
    power = np.abs(spectrum) ** 2
    freqs = np.linspace(0, rate / 2, n_fft // 2 + 1)
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    edges = np.linspace(mel(50), mel(7600), bands + 2)
    hertz = 700 * (10 ** (edges / 2595) - 1)
    filters = np.zeros((bands, len(freqs)))
    for b in range(bands):
        left, centre, right = hertz[b], hertz[b + 1], hertz[b + 2]
        rising = (freqs >= left) & (freqs <= centre); falling = (freqs > centre) & (freqs <= right)
        filters[b, rising] = (freqs[rising] - left) / (centre - left + 1e-9)
        filters[b, falling] = (right - freqs[falling]) / (right - centre + 1e-9)
    return np.log(filters @ power + 1e-10)


def mcd(reference: np.ndarray, degraded: np.ndarray, coefficients: int = 13) -> float:
    """Mel-cepstral distortion in dB over frames where the reference is not silent."""
    a = log_mel(reference); b = log_mel(degraded)
    length = min(a.shape[1], b.shape[1]); a, b = a[:, :length], b[:, :length]
    voiced = a.mean(0) > a.mean(0).max() - 4.
    if voiced.sum() < 5:
        return float('nan')
    ca = dct(a[:, voiced], type=2, axis=0, norm='ortho')[1:coefficients + 1]
    cb = dct(b[:, voiced], type=2, axis=0, norm='ortho')[1:coefficients + 1]
    return float((10 / np.log(10)) * np.sqrt(2) * np.mean(np.sqrt(((ca - cb) ** 2).sum(0))))


def envelope(x: np.ndarray, rate: int = RATE, hop: int = 160) -> np.ndarray:
    frames = 1 + (len(x) - 400) // hop
    index = np.arange(frames)[:, None] * hop + np.arange(400)[None, :]
    return np.sqrt((x[index] ** 2).mean(1) + 1e-12)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    length = min(len(a), len(b)); a, b = a[:length] - a[:length].mean(), b[:length] - b[:length].mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-12 else 0.


def modulation_correlation(a: np.ndarray, b: np.ndarray, rate: float = 100., low: float = .5, high: float = 16.) -> float:
    """Correlation of the envelope modulation spectra between 0.5 and 16 Hz."""
    length = min(len(a), len(b))
    spectra = []
    for x in (a[:length], b[:length]):
        x = x - x.mean()
        magnitude = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freqs = np.fft.rfftfreq(len(x), 1 / rate)
        spectra.append(np.log(magnitude[(freqs >= low) & (freqs <= high)] + 1e-9))
    return correlation(*spectra)


def speech_length(index_entry, rate: int = RATE) -> int:
    return int(index_entry['oracle_duration_frames']) * 256


# ----------------------------------------------------------------------------- per-trial scoring
def score_trial(folder: Path, samples: int, conditions) -> dict:
    original = sf.read(folder / 'original.wav', dtype='float32')[0][:samples]
    reference_envelope = envelope(original)
    out = {}
    for name in conditions:
        path = folder / f'{name}.wav'
        if not path.exists():
            continue
        degraded = sf.read(path, dtype='float32')[0][:samples]
        entry = dict(stoi=stoi(original, degraded), mcd=mcd(original, degraded),
                     envelope_corr=correlation(reference_envelope, envelope(degraded)),
                     modulation_corr=modulation_correlation(reference_envelope, envelope(degraded)))
        if _reference_pesq is not None:
            try:
                entry['pesq'] = float(_reference_pesq(RATE, original, degraded, 'wb'))
            except Exception:
                entry['pesq'] = float('nan')
        out[name] = entry
    return out


def bootstrap_interval(values, rng, repeats: int = 2000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return [float('nan'), float('nan')]
    draws = [float(rng.choice(values, len(values)).mean()) for _ in range(repeats)]
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def foil_pool(keys, lengths, contents, key, duration_matched: bool, size: int = 12):
    """Candidate foils: always a DIFFERENT sentence (not another trial of the same one).

    Every sentence was presented to ~17 participants, so a foil drawn per
    trial is usually the same sentence and the comparison becomes a tie.
    With ``duration_matched`` the pool is further restricted to the sentences
    whose duration is closest to the target's, removing the length shortcut.
    """
    others = [k for k in keys if contents[k] != contents[key]]
    if not duration_matched:
        return others
    others.sort(key=lambda k: abs(lengths[k] - lengths[key]))
    # keep one trial per sentence, nearest durations first
    seen, pool = set(), []
    for k in others:
        if contents[k] not in seen:
            seen.add(contents[k]); pool.append(k)
        if len(pool) >= size:
            break
    return pool


def two_alternative(folders, index, conditions, contents, rng, repeats: int = 5, duration_matched: bool = True) -> dict:
    """For each trial: is the reconstruction closer to its own original than to a foil?

    Chance is 0.50; intervals are bootstrapped over comparisons.  See
    ``foil_pool`` for how the foil is chosen.
    """
    results = {name: {'stoi': [], 'mcd': [], 'envelope_corr': []} for name in conditions}
    lengths = {entry['folder']: speech_length(entry) for entry in index}
    keys = list(folders)
    for key in keys:
        own = sf.read(folders[key] / 'original.wav', dtype='float32')[0][:lengths[key]]
        window = foil_pool(keys, lengths, contents, key, duration_matched)
        if not window:
            continue
        for name in conditions:
            path = folders[key] / f'{name}.wav'
            if not path.exists():
                continue
            degraded = sf.read(path, dtype='float32')[0][:lengths[key]]
            for _ in range(repeats):
                foil_key = window[rng.integers(len(window))]
                foil = sf.read(folders[foil_key] / 'original.wav', dtype='float32')[0][:lengths[foil_key]]
                length = min(len(own), len(foil), len(degraded))
                if length < 8000:
                    continue
                a, b, d = own[:length], foil[:length], degraded[:length]
                results[name]['stoi'].append(float(stoi(a, d) > stoi(b, d)))
                results[name]['mcd'].append(float(mcd(a, d) < mcd(b, d)))
                results[name]['envelope_corr'].append(float(correlation(envelope(a), envelope(d)) > correlation(envelope(b), envelope(d))))
    return {name: {measure: dict(accuracy=float(np.mean(values)) if values else float('nan'),
                                 ci95=bootstrap_interval(values, rng), comparisons=len(values))
                   for measure, values in entry.items()} for name, entry in results.items()}


def listening_bundle(folders, index, contents, output: Path, count: int, rng, condition: str = 'correct'):
    """Randomised 2AFC listening test: reconstruction + two candidate originals."""
    output.mkdir(parents=True, exist_ok=True)
    keys = list(folders); chosen = [keys[i] for i in rng.choice(len(keys), size=min(count, len(keys)), replace=False)]
    references = {entry['folder']: entry for entry in index}
    answers = []
    lengths = {entry['folder']: speech_length(entry) for entry in index}
    for number, key in enumerate(chosen, start=1):
        pool = foil_pool(keys, lengths, contents, key, duration_matched=True)
        if not pool:
            continue
        foil = pool[rng.integers(len(pool))]
        item = output / f'item{number:02d}'; item.mkdir(exist_ok=True)
        order = ['A', 'B'] if rng.random() < .5 else ['B', 'A']
        mapping = dict(zip(order, [key, foil]))
        for label, source in mapping.items():
            sf.write(item / f'candidate_{label}.wav', sf.read(folders[source] / 'original.wav', dtype='float32')[0], RATE, subtype='FLOAT')
        sf.write(item / 'reconstruction.wav', sf.read(folders[key] / f'{condition}.wav', dtype='float32')[0], RATE, subtype='FLOAT')
        answers.append(dict(item=number, correct='A' if mapping['A'] == key else 'B', trial=references[key]['trial_id'],
                            transcript=references[key].get('reference_transcript', ''), condition=condition))
    (output / 'answer_key.json').write_text(json.dumps(answers, indent=2) + '\n')
    (output / 'INSTRUCTIONS.txt').write_text(
        f'Two-alternative forced choice, {len(answers)} items.\n\n'
        'For every item folder: listen to reconstruction.wav, then candidate_A.wav and\n'
        'candidate_B.wav.  Which candidate is the sentence the reconstruction was made\n'
        'from?  Answer A or B; guess when unsure (chance is 50%).  Do not read\n'
        'answer_key.json before finishing.\n\n'
        'Write your answers in this folder as responses.txt, one line per item:\n'
        '  1 A\n  2 B\n  ...\n'
        '(a bare string such as ABBA... in item order also works, as does\n'
        ' responses.json: [{"item": 1, "answer": "A"}, ...])\n\n'
        'Then score with:\n'
        '  python app/audio_comparison.py --score-listening <this folder>\n')
    return len(answers)


def read_responses(folder: Path, expected: list[int]) -> dict[int, str]:
    """Accept responses.json, responses.txt ("1 A" per line) or a bare A/B string."""
    path_json, path_text = folder / 'responses.json', folder / 'responses.txt'
    if path_json.exists():
        payload = json.loads(path_json.read_text())
        if isinstance(payload, dict):
            payload = [dict(item=k, answer=v) for k, v in payload.items()]
        return {int(entry['item']): str(entry['answer']).strip().upper() for entry in payload}
    if path_text.exists():
        text = path_text.read_text().strip()
        rows = [line.split() for line in text.splitlines() if line.strip()]
        if all(len(row) >= 2 for row in rows) and rows:
            return {int(row[0].rstrip('.:')): row[1].strip().upper() for row in rows}
        letters = [c for c in text.upper() if c in 'AB']       # e.g. "ABBABA..."
        return dict(zip(expected, letters))
    raise FileNotFoundError('no responses file')


def score_listening(folder: Path) -> dict:
    key = json.loads((folder / 'answer_key.json').read_text())
    answers = {entry['item']: entry['correct'] for entry in key}
    responses = read_responses(folder, sorted(answers))
    unknown = sorted(set(responses) - set(answers))
    invalid = sorted(item for item, value in responses.items() if value not in ('A', 'B'))
    missing = sorted(set(answers) - set(responses))
    if unknown or invalid:
        raise ValueError(f'unknown items {unknown}; answers must be A or B (bad: {invalid})')
    scored = {item: value for item, value in responses.items() if item in answers}
    correct = [scored[item] == answers[item] for item in sorted(scored)]
    n = len(correct); k = int(sum(correct))
    from scipy.stats import binomtest
    result = binomtest(k, n, .5, alternative='greater')
    return dict(items=n, unanswered=missing, correct=k, accuracy=k / n if n else float('nan'),
                p_value=float(result.pvalue), ci95=[float(v) for v in result.proportion_ci(.95)],
                by_item={item: dict(answer=scored[item], correct=answers[item], right=scored[item] == answers[item]) for item in sorted(scored)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', default=str(ROOT / 'outputs/aligned_recovery_v3/eval_validation_positional'))
    parser.add_argument('--manifest', default=str(ROOT / 'artifacts/aligned_speech_local_v1/manifest.csv'),
                        help='manifest that maps trial_id to content_group (foils must be a different sentence)')
    parser.add_argument('--output', default=str(ROOT / 'outputs/aligned_recovery_v3/audio_comparison'))
    parser.add_argument('--limit', type=int, default=0, help='number of trials to score (0 = all)')
    parser.add_argument('--afc-trials', type=int, default=60)
    parser.add_argument('--listening-test', type=int, default=0, help='write a human 2AFC bundle with this many items')
    parser.add_argument('--score-listening', help='score responses.json in this bundle folder')
    parser.add_argument('--seed', type=int, default=31)
    args = parser.parse_args()
    if args.score_listening:
        folder = Path(args.score_listening)
        if not (folder / 'answer_key.json').exists():
            raise SystemExit(f'{folder} is not a listening-test bundle (no answer_key.json); create one with --listening-test N')
        try:
            report = score_listening(folder)
        except FileNotFoundError:
            items = [entry['item'] for entry in json.loads((folder / 'answer_key.json').read_text())]
            raise SystemExit(
                f'No answers found yet in {folder}.\n\n'
                f'  1. For each of the {len(items)} items, listen to item<NN>/reconstruction.wav, then\n'
                f'     item<NN>/candidate_A.wav and item<NN>/candidate_B.wav, and decide which candidate\n'
                f'     the reconstruction was made from (guess when unsure; chance is 50%).\n'
                f'  2. Write the answers into ONE of these files in that folder:\n'
                f'       responses.txt   one line per item, e.g.   1 A\n'
                f'                                                 2 B\n'
                f'       responses.txt   or just the letters in order: ABBA...\n'
                f'       responses.json  [{{"item": 1, "answer": "A"}}, ...]\n'
                f'  3. Re-run this command.\n')
        summary = {k: v for k, v in report.items() if k != 'by_item'}
        print(json.dumps(summary, indent=2))
        print(f"\n{report['correct']}/{report['items']} correct = {report['accuracy']:.1%} "
              f"(chance 50%, one-sided binomial p = {report['p_value']:.4f}, 95% CI "
              f"[{report['ci95'][0]:.1%}, {report['ci95'][1]:.1%}])")
        (folder / 'score.json').write_text(json.dumps(report, indent=2) + '\n')
        return
    export = Path(args.export); index = json.loads((export / 'index.json').read_text())['samples']
    folders = {entry['folder']: export / 'waveforms' / entry['folder'] for entry in index}
    import pandas as pd
    manifest = pd.read_csv(args.manifest, keep_default_na=False)
    by_trial = dict(zip(manifest.trial_id, manifest.content_group))
    contents = {entry['folder']: by_trial[entry['trial_id']] for entry in index}
    rng = np.random.default_rng(args.seed)
    selected = index if not args.limit else [index[i] for i in sorted(rng.choice(len(index), size=min(args.limit, len(index)), replace=False))]
    rows = {name: [] for name in CONDITIONS}
    for n, entry in enumerate(selected, start=1):
        scores = score_trial(folders[entry['folder']], speech_length(entry), CONDITIONS)
        for name, value in scores.items():
            rows[name].append(value)
        if n % 25 == 0:
            print(json.dumps(dict(scored=n, total=len(selected))), flush=True)
    measures = [m for m in ('stoi', 'pesq', 'mcd', 'envelope_corr', 'modulation_corr') if any(m in r for values in rows.values() for r in values)]
    summary = {name: {m: float(np.nanmean([r[m] for r in values if m in r])) for m in measures}
               for name, values in rows.items() if values}
    afc_keys = [entry['folder'] for entry in selected[:args.afc_trials]]
    afc = two_alternative({k: folders[k] for k in afc_keys}, [e for e in index if e['folder'] in afc_keys], CONDITIONS, contents, rng)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / 'audio_comparison.json').write_text(json.dumps(dict(contract='audio_comparison_v1', export=str(export),
                                                                 trials=len(selected), per_condition=summary,
                                                                 two_alternative_forced_choice=afc), indent=2) + '\n')
    print(f"\n{'condition':22s} {'STOI':>6s} {'PESQ':>6s} {'MCD dB':>7s} {'env r':>6s} {'mod r':>6s} |  2AFC vs foil (chance .50)  {'STOI':>5s} {'MCD':>5s} {'env':>5s}")
    for name in CONDITIONS:
        if name not in summary:
            continue
        value = summary[name]; a = afc.get(name, {})
        print(f"{name:22s} {value['stoi']:6.3f} {value.get('pesq', float('nan')):6.3f} {value['mcd']:7.2f} {value['envelope_corr']:6.3f} {value['modulation_corr']:6.3f} |"
              f"{'':27s}{a.get('stoi', {}).get('accuracy', float('nan')):5.2f} {a.get('mcd', {}).get('accuracy', float('nan')):5.2f} {a.get('envelope_corr', {}).get('accuracy', float('nan')):5.2f}")
    for name in ('correct', 'zero', 'wrong_trial'):
        if name in afc:
            entry = afc[name]['stoi']
            print(f"  2AFC[STOI] {name:18s} {entry['accuracy']:.3f} [{entry['ci95'][0]:.3f}, {entry['ci95'][1]:.3f}]  n={entry['comparisons']}")
    if args.listening_test:
        bundle = output / 'listening_test'
        print(f"\nwrote {listening_bundle(folders, index, contents, bundle, args.listening_test, rng)} listening items to {bundle}")


if __name__ == '__main__':
    main()
