#!/usr/bin/env python3
"""DS004940 ingest for the VAE + diffusion route (``vaediff``).

One subject at a time: download that subject's BDF recordings (both tasks) from the public OpenNeuro
bucket, preprocess, cut fixed epochs, store them as float16, verify the stored copy, and delete the raw
files before the next subject.  The stimuli are processed once (Vocos mel, WavLM layers, transcript, text
embedding); afterwards the stimulus WAVs are deleted too, the 24 kHz waveform is kept inside the
processed audio table.

Preprocessing follows the September pipeline (``scripts/prepare_training_data.py``): 128 BioSemi channels,
declared-bad channels interpolated (a recording above 15 % bad is skipped), average reference, 0.5-45 Hz
band-pass.  Two differences, both listed in README_VAEDIFF.md: the signal is resampled to 128 Hz instead of
256 Hz (nothing above 45 Hz survives the band-pass), and each continuous run is scaled per channel by its
own median / MAD instead of a train-fold normaliser.  Epochs span [-0.5, 4.5) s around the sentence trigger;
the model crops its own window (default [-0.25, 3.6) s, which ends before the Active-task prompt that starts
exactly 2.0 s after every sentence).

Stages: ``metadata``, ``audio``, ``eeg``, ``index``, ``all`` (the four in order).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
RAW = ROOT / 'data/ds004940'
OUT = ROOT / 'artifacts/vaediff_ds004940'
MODELS = PROJECT / 'models'
S3 = 's3://openneuro.org/ds004940'
TEXT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

CHANNELS = [f'{bank}{i}' for bank in 'ABCD' for i in range(1, 33)]
SOURCE_RATE = 512
EEG_RATE = 128
EPOCH_START = -0.5                       # seconds relative to the sentence trigger
EPOCH_SAMPLES = 640                      # 5.0 s at 128 Hz
BANDPASS = (0.5, 45.)
MAX_BAD_FRACTION = .15
AUDIO_RATE = 24000
AUDIO_SECONDS = 4.5
WAVLM_LAYERS = (3, 6, 9, 12)
TASKS = ('N400Active', 'N400Passive')
SUBJECTS = [f'sub-{i:03d}' for i in range(1, 23)]
MIN_FREE_GB = 8.


def log(**row):
    print(json.dumps(row, default=str), flush=True)


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(value, indent=1, default=str))
    tmp.replace(path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    tmp = path.with_name(path.stem + '.part.npy')
    np.save(tmp, value)
    tmp.replace(path)


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def s3_sync(target: Path, includes: list[str]) -> None:
    args = ['aws', 's3', 'sync', '--no-sign-request', S3, str(target), '--exclude', '*', '--only-show-errors']
    for pattern in includes:
        args += ['--include', pattern]
    subprocess.run(args, check=True)


# ----------------------------------------------------------------------------- sentence table
def pair_of(stim_file: str) -> str:
    """NPC_car.wav and NPI_car(chest).wav share the sentence frame 'car': one pair, one split."""
    match = re.match(r'^NP[CI]_([^(.]+)', Path(stim_file).name)
    if not match:
        raise ValueError(f'not an experimental stimulus: {stim_file}')
    return match.group(1)


def sentence_table(path: Path) -> list[dict]:
    import pandas as pd
    table = pd.read_csv(path, sep='\t')
    rows = []
    for _, r in table.iterrows():
        name = str(r['stim_file'])
        if not re.match(r'^NP[CI]_', name):
            continue
        words = [str(r[str(i)]) for i in range(1, 9) if isinstance(r[str(i)], str) and r[str(i)] != 'n/a']
        rows.append(dict(stim_file=name, text=' '.join(words), words=words, pair=pair_of(name),
                         condition=name[:3], duration=float(r['stim_dur(s)']), target_word=words[-1].strip('.?!'),
                         target_onset=float(r['target_onset(s)']), target_end=float(r['target_end(s)']),
                         cloze=float(r['cloze-probability%_div'])))
    rows.sort(key=lambda row: row['stim_file'])
    for i, row in enumerate(rows):
        row['index'] = i
    return rows


# ----------------------------------------------------------------------------- metadata
def metadata(args) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    s3_sync(RAW, ['README', 'participants.*', 'dataset_description.json', 'task-*.json',
                  'N400PvsA_stimuli_parameters.*', 'stimuli/NPC_*.wav', 'stimuli/NPI_*.wav'])
    rows = sentence_table(RAW / 'N400PvsA_stimuli_parameters.tsv')
    missing = [r['stim_file'] for r in rows if not (RAW / 'stimuli' / r['stim_file']).exists()]
    log(stage='metadata', sentences=len(rows), pairs=len({r['pair'] for r in rows}), missing_wavs=len(missing))
    if missing and not (OUT / 'audio' / 'complete.json').exists():
        raise SystemExit(f'stimulus WAVs missing: {missing[:5]}')


# ----------------------------------------------------------------------------- audio
def text_model_dir() -> Path:
    """MiniLM sentence encoder (22 M parameters): the frozen text space of the text loss."""
    folder = MODELS / 'all-MiniLM-L6-v2'
    if (folder / 'revision.json').exists():
        return folder
    from huggingface_hub import HfApi, snapshot_download
    revision = HfApi().model_info(TEXT_MODEL).sha
    snapshot_download(TEXT_MODEL, revision=revision, local_dir=folder,
                      allow_patterns=['config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt',
                                      'special_tokens_map.json', 'model.safetensors'])
    atomic_json(folder / 'revision.json', dict(repo_id=TEXT_MODEL, revision=revision))
    return folder


def text_embeddings(texts: list[str], folder: Path) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    model = AutoModel.from_pretrained(folder, local_files_only=True).eval()
    with torch.no_grad():
        batch = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        hidden = model(**batch).last_hidden_state
        mask = batch['attention_mask'][:, :, None].float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(pooled, dim=-1).numpy().astype(np.float32)


def vocos_frontend():
    from vocos import Vocos
    import torch
    model = Vocos.from_hparams(str(MODELS / 'vocos-mel-24khz/config.yaml'))
    model.load_state_dict(torch.load(MODELS / 'vocos-mel-24khz/pytorch_model.bin', map_location='cpu', weights_only=True))
    return model.eval()


def read_resampled(path: Path, rate: int) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    wave, source = sf.read(path, dtype='float32', always_2d=True)
    wave = wave.mean(1)
    if source != rate:
        g = math.gcd(source, rate)
        wave = resample_poly(wave, rate // g, source // g).astype(np.float32)
    return wave


def fixed_length(wave: np.ndarray, samples: int) -> np.ndarray:
    return np.pad(wave[:samples], (0, max(0, samples - len(wave)))).astype(np.float32)


def wavlm_frame_times(frames: int) -> np.ndarray:
    """Centre of WavLM frame k: receptive field 400 samples, stride 320, at 16 kHz."""
    return (np.arange(frames) * 320 + 200) / 16000.


def audio(args) -> None:
    import torch
    from transformers import WavLMModel
    folder = OUT / 'audio'
    if (folder / 'complete.json').exists():
        log(stage='audio', status='already complete'); return
    folder.mkdir(parents=True, exist_ok=True)
    rows = sentence_table(RAW / 'N400PvsA_stimuli_parameters.tsv')
    vocos = vocos_frontend()
    wavlm = WavLMModel.from_pretrained(MODELS / 'wavlm-base', local_files_only=True).eval()
    mels, layers, waves, envelopes, mismatches = [], [], [], [], []
    started = time.monotonic()
    for i, row in enumerate(rows):
        path = RAW / 'stimuli' / row['stim_file']
        w24 = read_resampled(path, AUDIO_RATE)
        w16 = read_resampled(path, 16000)
        row['wav_seconds'] = len(w24) / AUDIO_RATE
        if abs(row['wav_seconds'] - row['duration']) > .05:
            mismatches.append((row['stim_file'], row['wav_seconds'], row['duration']))
        w24 = fixed_length(w24, int(AUDIO_SECONDS * AUDIO_RATE)); w16 = fixed_length(w16, int(AUDIO_SECONDS * 16000))
        with torch.no_grad():
            mels.append(vocos.feature_extractor(torch.from_numpy(w24)[None])[0].numpy().astype(np.float32))
            hidden = wavlm(torch.from_numpy(w16)[None], output_hidden_states=True).hidden_states
            layers.append(np.stack([hidden[k][0].numpy() for k in WAVLM_LAYERS]).astype(np.float16))
        frame = 160
        index = np.arange(1 + (len(w16) - 400) // frame)[:, None] * frame + np.arange(400)[None]
        envelopes.append(np.sqrt((w16[index] ** 2).mean(1) + 1e-12).astype(np.float32))      # 100 Hz RMS
        waves.append(w24.astype(np.float16))
        if i % 50 == 0:
            log(stage='audio', done=i + 1, total=len(rows), seconds=round(time.monotonic() - started, 1))
    text = text_embeddings([r['text'] for r in rows], text_model_dir())
    mel = np.stack(mels)
    atomic_npy(folder / 'mel.npy', mel)
    atomic_npy(folder / 'wavlm.npy', np.stack(layers))
    atomic_npy(folder / 'wave24.npy', np.stack(waves))
    atomic_npy(folder / 'envelope100.npy', np.stack(envelopes))
    atomic_npy(folder / 'text.npy', text)
    atomic_json(folder / 'sentences.json', rows)
    mel_times = np.arange(mel.shape[-1]) * 256 / AUDIO_RATE                               # Vocos 'center' padding
    atomic_npy(folder / 'mel_times.npy', mel_times.astype(np.float32))
    atomic_npy(folder / 'wavlm_times.npy', wavlm_frame_times(layers[0].shape[1]).astype(np.float32))
    revision = json.loads((text_model_dir() / 'revision.json').read_text())
    atomic_json(folder / 'complete.json', dict(sentences=len(rows), mel_shape=mel.shape, wavlm_layers=WAVLM_LAYERS,
                                               wavlm_shape=np.stack(layers).shape, text_model=revision,
                                               duration_mismatches=mismatches, seconds=AUDIO_SECONDS))
    log(stage='audio', status='complete', sentences=len(rows), duration_mismatches=len(mismatches))
    if not args.keep_raw:
        wavs = sorted((RAW / 'stimuli').glob('*.wav'))
        for path in wavs:
            path.unlink()
        log(stage='audio', removed_stimulus_wavs=len(wavs))


# ----------------------------------------------------------------------------- EEG
def bad_channels(path: Path) -> list[str]:
    import pandas as pd
    if not path.exists():
        return []
    table = pd.read_csv(path, sep='\t')
    bad = table[table['status'].astype(str).str.lower() == 'bad']['name'].astype(str)
    return sorted({name.split('_')[0].strip() for name in bad})


def sentence_events(path: Path) -> list[dict]:
    """Experimental sentence triggers (NPC/NPI), with the onset of the next audio event of any kind."""
    import pandas as pd
    table = pd.read_csv(path, sep='\t')
    onsets = pd.to_numeric(table['stim_onset_s_'], errors='coerce').to_numpy()
    events = []
    for i, r in table.iterrows():
        if str(r['type']) != 'Sound' or str(r['trial_type']).strip().upper() not in ('NPC', 'NPI'):
            continue
        later = [o for o in onsets[i + 1:] if np.isfinite(o)]
        events.append(dict(row=int(i), onset=float(onsets[i]), stim_file=Path(str(r['stim_file'])).name,
                           condition=str(r['trial_type']).strip().upper(),
                           next_audio=float(later[0] - onsets[i]) if later else float('inf')))
    return events


def recording_runs(folder: Path, subject: str, task: str) -> list[Path]:
    exact = folder / f'{subject}_task-{task}_eeg.bdf'
    split = sorted(folder.glob(f'{subject}_task-{task}_run-*_eeg.bdf'),
                   key=lambda p: int(re.search(r'_run-(\d+)_eeg', p.name).group(1)))
    return split or ([exact] if exact.exists() else [])


def robust_scale(data: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-channel (x - median) / (1.4826 * MAD) over the whole continuous run; invalid channels become 0."""
    median = np.median(data, axis=1, keepdims=True)
    mad = np.median(np.abs(data - median), axis=1, keepdims=True) * 1.4826
    ok = valid[:, None] & (mad > 0)
    return np.where(ok, (data - median) / np.where(ok, mad, 1.), 0.).astype(np.float32)


def preprocess_run(path: Path, bad: list[str]):
    """One BDF run -> (128 x samples at 128 Hz, scaled; valid-channel mask; interpolated list)."""
    import mne
    raw = mne.io.read_raw_bdf(path, preload=False, verbose='ERROR')
    names = {name.split('_')[0].strip(): name for name in raw.ch_names}
    missing = [c for c in CHANNELS if c not in names]
    raw.pick([names[c] for c in CHANNELS if c in names])
    raw.rename_channels({names[c]: c for c in CHANNELS if c in names})
    raw.load_data(verbose='ERROR')
    declared = [c for c in bad if c in raw.ch_names]
    fraction = len(set(missing) | set(declared)) / len(CHANNELS)
    if fraction > MAX_BAD_FRACTION:
        raise ValueError(f'bad/missing channel fraction {fraction:.3f} > {MAX_BAD_FRACTION}')
    raw.set_montage(mne.channels.make_standard_montage('biosemi128'), on_missing='raise')
    if declared:
        raw.info['bads'] = declared
        raw.interpolate_bads(reset_bads=True, verbose='ERROR')
    raw.set_eeg_reference('average', projection=False, verbose='ERROR')
    raw.resample(EEG_RATE, npad='auto', verbose='ERROR')
    raw.filter(*BANDPASS, verbose='ERROR')
    present = raw.get_data() * 1e6
    data = np.zeros((len(CHANNELS), present.shape[1]), dtype=np.float32)
    index = {c: i for i, c in enumerate(raw.ch_names)}
    for k, c in enumerate(CHANNELS):
        if c in index:
            data[k] = present[index[c]]
    valid = np.array([c not in missing for c in CHANNELS])
    return robust_scale(data, valid), valid, declared, float(raw.info['sfreq'])


def epoch_start(onset: float, run_offset: float) -> int:
    return int(round((onset - run_offset + EPOCH_START) * EEG_RATE))


def process_task(folder: Path, subject: str, task: str, sentences: dict) -> dict:
    runs = recording_runs(folder, subject, task)
    events_path = folder / f'{subject}_task-{task}_events.tsv'
    if not runs or not events_path.exists():
        return dict(subject=subject, task=task, status='missing')
    bad = bad_channels(folder / f'{subject}_task-{task}_channels.tsv')
    events = sentence_events(events_path)
    epochs, rows, offset = [], [], 0.
    excluded = dict(outside_run=0, unknown_stimulus=0)
    for run_number, path in enumerate(runs, start=1):
        data, valid, interpolated, rate = preprocess_run(path, bad)
        if rate != EEG_RATE:
            raise ValueError(f'unexpected rate {rate}')
        length = data.shape[1] / EEG_RATE
        for event in events:
            if not offset <= event['onset'] < offset + length:
                continue
            if event['stim_file'] not in sentences:
                excluded['unknown_stimulus'] += 1; continue
            start = epoch_start(event['onset'], offset)
            if start < 0 or start + EPOCH_SAMPLES > data.shape[1]:
                excluded['outside_run'] += 1; continue
            x = data[:, start:start + EPOCH_SAMPLES]
            epochs.append(x.astype(np.float16))
            rows.append(dict(subject=subject, task=task, run=run_number, event_row=event['row'], onset=event['onset'],
                             stim_file=event['stim_file'], sentence=sentences[event['stim_file']],
                             condition=event['condition'], next_audio=event['next_audio'],
                             peak=float(np.abs(x).max()), missing=int((~valid).sum()), interpolated=interpolated))
        offset += length
    if not epochs:
        return dict(subject=subject, task=task, status='no_epochs', excluded=excluded)
    stem = OUT / 'eeg' / f'{subject}_task-{task}'
    atomic_npy(stem.with_suffix('.npy'), np.stack(epochs))
    atomic_json(stem.with_suffix('.json'), rows)
    check = np.load(stem.with_suffix('.npy'), mmap_mode='r')
    if check.shape != (len(rows), len(CHANNELS), EPOCH_SAMPLES) or not np.isfinite(check[:]).all():
        raise RuntimeError(f'verification failed for {stem}')
    return dict(subject=subject, task=task, status='ok', trials=len(rows), runs=len(runs), bad=bad,
                excluded=excluded, events=len(events))


def eeg(args) -> None:
    (OUT / 'eeg').mkdir(parents=True, exist_ok=True)
    rows = json.loads((OUT / 'audio/sentences.json').read_text())
    sentences = {r['stim_file']: r['index'] for r in rows}
    status_path = OUT / 'eeg' / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    subjects = SUBJECTS if args.subjects == 'all' else [f'sub-{int(s):03d}' for s in args.subjects.split(',')]
    for subject in subjects:
        if subject in status and all(t in status[subject] for t in TASKS):
            log(stage='eeg', subject=subject, status='already done'); continue
        if free_gb(OUT) < MIN_FREE_GB:
            raise SystemExit(f'less than {MIN_FREE_GB} GB free; stopping before {subject}')
        started = time.monotonic()
        s3_sync(RAW, [f'{subject}/eeg/*'])
        folder = RAW / subject / 'eeg'
        log(stage='eeg', subject=subject, downloaded_gb=round(sum(p.stat().st_size for p in folder.glob('*')) / 1e9, 2),
            seconds=round(time.monotonic() - started, 1))
        status[subject] = {}
        for task in TASKS:
            try:
                result = process_task(folder, subject, task, sentences)
            except Exception as exc:                       # a skipped recording is recorded, not fatal
                result = dict(subject=subject, task=task, status='skipped', reason=f'{type(exc).__name__}: {exc}')
            status[subject][task] = result
            log(stage='eeg', **result, seconds=round(time.monotonic() - started, 1))
        atomic_json(status_path, status)
        if not args.keep_raw:
            shutil.rmtree(RAW / subject)
            log(stage='eeg', subject=subject, raw_deleted=True, free_gb=round(free_gb(OUT), 1))


# ----------------------------------------------------------------------------- index and splits
def ranked(values, seed: str, namespace: str):
    return sorted(values, key=lambda v: hashlib.sha256(f'{seed}:{namespace}:{v}'.encode()).hexdigest())


def make_splits(pairs, subjects, seed: str = 'vaediff-ds004940-v1', val: int = 20, test: int = 20, unseen: int = 3,
                eligible=None) -> dict:
    """Sentence pairs (NPC + NPI of one frame) go to one split; ``unseen`` subjects are test-only.

    The unseen people are drawn by hash from ``eligible`` (default: everyone); the ingest passes the people whose
    both recordings survived quality control, so the unseen-people test is not limited to one task.
    """
    order = ranked(sorted(set(pairs)), seed, 'pair')
    role = {p: 'train' for p in order[:len(order) - val - test]}
    role.update({p: 'val' for p in order[len(order) - val - test:len(order) - test]})
    role.update({p: 'test' for p in order[len(order) - test:]})
    pool = sorted(set(subjects) if eligible is None else set(eligible) & set(subjects))
    held = sorted(ranked(pool, seed, 'subject')[:unseen])
    return dict(seed=seed, pair_role=role, unseen_subjects=held)


def trial_role(pair_role: str, unseen: bool) -> str:
    if unseen:
        return 'test_unseen' if pair_role == 'test' else 'unused'
    return dict(train='train', val='dev', test='test_seen')[pair_role]


def build_index(root: Path = OUT, **split_args) -> dict:
    """trials.json (every stored epoch with its role) and splits.json, from the per-recording files in ``root/eeg``."""
    sentences = json.loads((root / 'audio/sentences.json').read_text())
    files = sorted((root / 'eeg').glob('sub-*_task-*.json'))
    if not files:
        raise SystemExit('no EEG files; run the eeg stage first')
    trials = []
    for f in files:
        for row, trial in enumerate(json.loads(f.read_text())):
            trials.append(dict(trial, file=f.stem, row=row))
    tasks = {}
    for t in trials:
        tasks.setdefault(t['subject'], set()).add(t['task'])
    complete = sorted(s for s, ts in tasks.items() if len(ts) == len(TASKS))
    splits = make_splits([s['pair'] for s in sentences], set(tasks), eligible=split_args.pop('eligible', complete), **split_args)
    unseen = set(splits['unseen_subjects'])
    counts = {}
    for t in trials:
        t['pair'] = sentences[t['sentence']]['pair']
        t['role'] = trial_role(splits['pair_role'][t['pair']], t['subject'] in unseen)
        counts[t['role']] = counts.get(t['role'], 0) + 1
    atomic_json(root / 'trials.json', trials)
    atomic_json(root / 'splits.json', dict(splits, counts=counts, subjects=sorted({t['subject'] for t in trials})))
    return dict(trials=len(trials), files=len(files), counts=counts, unseen=splits['unseen_subjects'])


def index(args) -> None:
    log(stage='index', **build_index(OUT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['metadata', 'audio', 'eeg', 'index', 'all'])
    parser.add_argument('--subjects', default='all', help='"all" or comma-separated numbers, e.g. 1,2,3')
    parser.add_argument('--keep-raw', action='store_true', help='do not delete raw files after processing')
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    stages = ['metadata', 'audio', 'eeg', 'index'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        if stage == 'metadata' and (OUT / 'audio/complete.json').exists():
            continue
        globals()[stage](args)


if __name__ == '__main__':
    sys.exit(main())
