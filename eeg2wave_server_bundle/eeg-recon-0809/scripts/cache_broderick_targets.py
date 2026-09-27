#!/usr/bin/env python3
"""Speech targets for the Broderick continuous-speech windows, in the app/music.py PieceTargets format.

Per audiobook run (ds004408 ``stimuli/audio<r>.wav``, from AUDIO_OFFSET_S on,
see scripts/prepare_broderick_windows.py):

* ``teacher``  - layer 9 of the train-fold fine-tuned HuBERT that produced the
  DS004940 targets (outputs/aligned_speech_local_v1/hubert/best, the same
  Wav2Vec2 feature extractor), 50 Hz.  Runs are processed in 16 s chunks with
  2 s of context on each side and the chunk centres stitched, because a
  3-minute input does not fit the attention in memory;
* ``mel``      - the native SpeechT5 log-mel (80 bins, 16 ms hop) that the
  speech AcousticDecoder and the HiFi-GAN use (so MFCC-80 follows from it);
* ``acoustic`` - envelope and onset strength at 64 Hz (universal_model.acoustic_targets).

The window grids stored with it are DS004940's (199 teacher frames, 251 mel
frames over 4 s), so the frozen speech decoder reads Broderick windows
exactly as it reads DS004940 trials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src')); sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('ALIGNED_TARGET_CACHE_NAME', 'targets_adapted.h5')
import aligned_speech as legacy
from eeg2speech.aligned import convolution_times
from eeg2speech.speecht5 import native_speecht5_mel
from music import TARGET_CONTRACT
from prepare_broderick_windows import AUDIO_OFFSET_S, RUNS
from universal_model import WINDOW_S, acoustic_targets

RATE = 16000
LAYER = 9


def load(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    wave, rate = sf.read(path, dtype='float32', always_2d=True)
    wave = wave.mean(1)
    if rate != RATE:
        g = math.gcd(rate, RATE)
        wave = resample_poly(wave, RATE // g, rate // g).astype(np.float32)
    return wave[int(round(AUDIO_OFFSET_S * RATE)):]


@torch.no_grad()
def teacher_features(model, processor, wave, device, core_s=16., context_s=2.):
    kernels, strides = model.config.conv_kernel, model.config.conv_stride
    core, context = int(core_s * RATE), int(context_s * RATE)
    features, times = [], []
    for a in range(0, len(wave), core):
        b = min(len(wave), a + core)
        lo, hi = max(0, a - context), min(len(wave), b + context)
        inputs = processor(wave[lo:hi], sampling_rate=RATE, return_tensors='pt')
        hidden = model(**{k: v.to(device) for k, v in inputs.items()}, output_hidden_states=True).hidden_states[LAYER][0].float().cpu().numpy()
        t = convolution_times(hi - lo, kernels, strides).numpy() + lo / RATE
        if len(t) != len(hidden):
            raise RuntimeError('teacher time axis does not match output')
        keep = (t >= a / RATE) & (t < b / RATE)
        features.append(hidden[keep]); times.append(t[keep])
    return np.concatenate(features), np.concatenate(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--audio', default=str(ROOT / 'data/ds004408/stimuli'))
    parser.add_argument('--hubert', default=str(ROOT / 'outputs/aligned_speech_local_v1/hubert/best'))
    parser.add_argument('--output', default=str(ROOT / 'artifacts/speech_continuous/broderick2018/targets_hubert.h5'))
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    from transformers import HubertModel, Wav2Vec2FeatureExtractor
    device = legacy.device(args.device)
    output = Path(args.output)
    if output.exists():
        print(f'{output} exists'); return
    cfg = legacy.config(str(ROOT / 'configs/aligned_speech_local_v1.yaml'))
    _, _, target_cache, _ = legacy.artifact_paths(cfg)
    with h5py.File(target_cache, 'r') as h5:
        if h5.attrs['teacher_sha256'] != legacy.tree_hash(Path(args.hubert)) or int(h5.attrs['teacher_layer']) != LAYER:
            raise ValueError('--hubert is not the teacher of the DS004940 target cache')
        speech_times, mel_times = h5['speech_times'][:], h5['mel_times'][:]
    model = HubertModel.from_pretrained(args.hubert, local_files_only=True).eval().requires_grad_(False).to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.hubert, local_files_only=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix('.partial.h5')
    with h5py.File(partial, 'w') as h5:
        h5.attrs.update(contract=TARGET_CONTRACT, teacher=f'hubert_adapted_l{LAYER}', teacher_path=str(args.hubert),
                        teacher_sha256=legacy.tree_hash(Path(args.hubert)), mel='speecht5_native_log_mel_v1',
                        acoustic_rate=64, window_s=WINDOW_S, audio_offset_s=AUDIO_OFFSET_S, source=str(args.audio))
        h5.create_dataset('grid/teacher_times', data=speech_times.astype(np.float32))
        h5.create_dataset('grid/mel_times', data=mel_times.astype(np.float32))
        for r in RUNS:
            wave = load(Path(args.audio) / f'audio{r:02d}.wav')
            features, times = teacher_features(model, processor, wave, device)
            mel = native_speecht5_mel(torch.from_numpy(wave))[0].numpy()
            acoustic = acoustic_targets(torch.from_numpy(wave), RATE, frames=int(np.ceil(len(wave) / RATE * 64)))[0].numpy()
            for name, value in (('teacher', features), ('mel', mel), ('acoustic', acoustic)):
                if not np.isfinite(value).all():
                    raise ValueError(f'run {r}: nonfinite {name}')
            g = h5.create_group(f'pieces/{r}')
            g.create_dataset('teacher', data=features.astype(np.float32)); g.create_dataset('teacher_times', data=times)
            g.create_dataset('mel', data=mel.astype(np.float32))
            g.create_dataset('mel_times', data=np.arange(mel.shape[1]) * 256 / RATE)
            g.create_dataset('acoustic', data=acoustic.astype(np.float32))
            g.attrs.update(duration_s=len(wave) / RATE, audio_sha256=hashlib.sha256(wave.tobytes()).hexdigest())
            print(f'run {r:2d}: {len(wave) / RATE:6.1f} s, teacher {features.shape}, mel {mel.shape}, acoustic {acoustic.shape}', flush=True)
    partial.replace(output)
    print(json.dumps(dict(output=str(output), runs=len(RUNS))))


if __name__ == '__main__':
    main()
