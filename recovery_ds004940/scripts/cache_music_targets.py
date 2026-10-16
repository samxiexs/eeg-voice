#!/usr/bin/env python3
"""Per-piece reconstruction targets for the music route (the music analogue of the HuBERT target cache).

For every aligned piece (``scripts/render_music_audio.py``) this stores, over
the whole piece:

* ``teacher`` - the frozen music teacher's hidden states, (frames, D):
  ``--teacher mert`` = m-a-p/MERT-v1-95M layer ``--layer`` (HuBERT-style music
  SSL, 75 Hz at 24 kHz; loaded with trust_remote_code), or
  ``--teacher encodec`` = facebook/encodec_24khz encoder latents (128-d, 75 Hz;
  native in transformers, the fallback if MERT's remote code does not load).
  Long pieces are processed in ``--chunk-s`` windows with ``--context-s`` of
  context on each side; only frames whose centre lies in a window's core are kept.
* ``mel`` - BigVGAN-v2 24 kHz 100-band log-mel (hop 256), so predictions can be
  vocoded by nvidia/bigvgan_v2_24khz_100band_256x.
* ``acoustic`` - broadband log-energy envelope and onset strength at 64 Hz,
  the same function as for DS004940 speech (universal_model.acoustic_targets).

It also stores the fixed 4 s window grids (``grid/teacher_times``,
``grid/mel_times``) on which the music AcousticDecoder operates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src'))
from eeg2speech.aligned import convolution_times, tree_hash
from music import BIGVGAN_24K, bigvgan_mel, bigvgan_mel_times
from universal_model import WINDOW_S, acoustic_targets

TARGET_CONTRACT = 'music_targets_v1'
SAMPLE_RATE = 24000


class Teacher:
    def __init__(self, kind, path, layer, device):
        from transformers import AutoFeatureExtractor, AutoModel, EncodecModel
        self.kind, self.layer, self.device = kind, layer, device
        self.path = Path(path)
        if not (self.path / 'config.json').is_file():
            raise SystemExit(f'{path}: local teacher weights missing; run scripts/download_music.sh models')
        self.processor = AutoFeatureExtractor.from_pretrained(self.path)
        if kind == 'mert':
            self.model = AutoModel.from_pretrained(self.path, trust_remote_code=True).to(device).eval()
            config = self.model.config
            self.kernels, self.strides = list(config.conv_kernel), list(config.conv_stride)
            if layer < 1 or layer > config.num_hidden_layers:
                raise SystemExit(f'--layer must be in 1..{config.num_hidden_layers}')
        else:
            self.model = EncodecModel.from_pretrained(self.path).to(device).eval()
        if int(self.processor.sampling_rate) != SAMPLE_RATE:
            raise SystemExit(f'teacher expects {self.processor.sampling_rate} Hz audio')

    def frame_times(self, samples):
        """Centre times (s) of the teacher frames of a clip of ``samples`` samples."""
        if self.kind == 'mert':
            return convolution_times(samples, self.kernels, self.strides, SAMPLE_RATE).numpy()
        hop = int(np.prod(self.model.config.upsampling_ratios))
        return (np.arange(int(np.ceil(samples / hop))) + .5) * hop / SAMPLE_RATE

    @torch.inference_mode()
    def __call__(self, wave):
        inputs = self.processor(wave, sampling_rate=SAMPLE_RATE, return_tensors='pt')
        values = inputs['input_values'].to(self.device)
        if self.kind == 'mert':
            hidden = self.model(values, output_hidden_states=True).hidden_states[self.layer][0]
        else:
            hidden = self.model.encoder(values if values.ndim == 3 else values[:, None])[0].T
        times = self.frame_times(len(wave))
        if abs(len(times) - len(hidden)) > 1:
            raise RuntimeError(f'teacher returned {len(hidden)} frames, expected {len(times)}')
        count = min(len(times), len(hidden))                  # codec padding may add one frame
        return hidden[:count].float().cpu().numpy(), times[:count]


def teacher_sequence(teacher, wave, chunk_s, context_s):
    step, context = int(chunk_s * SAMPLE_RATE), int(context_s * SAMPLE_RATE)
    features, times = [], []
    for core in range(0, len(wave), step):
        first = max(0, core - context); last = min(len(wave), core + step + context)
        hidden, local = teacher(wave[first:last])
        absolute = local + first / SAMPLE_RATE
        keep = (absolute >= core / SAMPLE_RATE) & (absolute < min(core + step, len(wave)) / SAMPLE_RATE)
        features.append(hidden[keep]); times.append(absolute[keep])
    return np.concatenate(features), np.concatenate(times)


def main():
    import soundfile as sf
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--audio', default=str(ROOT / 'artifacts/music/diliberto2020/audio'))
    parser.add_argument('--teacher', choices=['mert', 'encodec'], default='mert')
    parser.add_argument('--teacher-path', default=None, help='local model folder (default models/mert_v1_95m or models/encodec_24khz)')
    parser.add_argument('--layer', type=int, default=6, help='MERT hidden layer (1-12)')
    parser.add_argument('--chunk-s', type=float, default=10.)
    parser.add_argument('--context-s', type=float, default=2.)
    parser.add_argument('--output', default=None, help='default artifacts/music/diliberto2020/targets_<teacher>.h5')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    audio = Path(args.audio)
    alignment = json.loads((audio / 'alignment.json').read_text())
    path = Path(args.teacher_path or ROOT / ('models/mert_v1_95m' if args.teacher == 'mert' else 'models/encodec_24khz'))
    output = Path(args.output or ROOT / f'artifacts/music/diliberto2020/targets_{args.teacher}.h5')
    teacher = Teacher(args.teacher, path, args.layer, torch.device(args.device))
    window = int(WINDOW_S * SAMPLE_RATE)
    mel_frames = bigvgan_mel(np.zeros(window, np.float32)).shape[1]
    working = output.with_suffix('.partial.h5')
    with h5py.File(working, 'w') as h5:
        h5.attrs.update(contract=TARGET_CONTRACT, teacher=args.teacher, teacher_path=str(path), teacher_sha256=tree_hash(path),
                        teacher_layer=args.layer if args.teacher == 'mert' else -1, sample_rate=SAMPLE_RATE,
                        mel=json.dumps(BIGVGAN_24K), acoustic_rate=64, window_s=WINDOW_S,
                        alignment_sha256=hashlib.sha256((audio / 'alignment.json').read_bytes()).hexdigest())
        h5.create_dataset('grid/teacher_times', data=teacher.frame_times(window).astype(np.float32))
        h5.create_dataset('grid/mel_times', data=bigvgan_mel_times(mel_frames).astype(np.float32))
        for piece, item in sorted(alignment['pieces'].items(), key=lambda kv: int(kv[0])):
            wave, rate = sf.read(item['audio'], dtype='float32')
            if rate != SAMPLE_RATE or wave.ndim != 1:
                raise SystemExit(f'{item["audio"]}: expected mono {SAMPLE_RATE} Hz')
            features, times = teacher_sequence(teacher, wave, args.chunk_s, args.context_s)
            mel = bigvgan_mel(wave)
            acoustic = acoustic_targets(torch.from_numpy(wave), SAMPLE_RATE, frames=int(np.ceil(len(wave) / SAMPLE_RATE * 64)))[0]
            g = h5.create_group(f'pieces/{int(piece)}')
            g.create_dataset('teacher', data=features.astype(np.float32)); g.create_dataset('teacher_times', data=times)
            g.create_dataset('mel', data=mel.astype(np.float32)); g.create_dataset('mel_times', data=bigvgan_mel_times(mel.shape[1]))
            g.create_dataset('acoustic', data=acoustic.numpy().astype(np.float32))
            g.attrs.update(duration_s=len(wave) / SAMPLE_RATE, audio_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
                           alignment_r=float(item['r']))
            print(f'piece {int(piece):2d}: teacher {features.shape}, mel {mel.shape}, acoustic {tuple(acoustic.shape)}', flush=True)
    working.replace(output)
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
