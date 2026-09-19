#!/usr/bin/env python3
"""Fixed-voice text-to-speech for the content-first route.

EEG decides *what* is said; this module decides *how*.  SpeechT5 TTS produces
the same 80-bin / hop-256 / 16 kHz mel that the already-pinned SpeechT5
HiFi-GAN consumes, so the synthesis path introduces no new vocoder: it is the
same back end whose oracle ceiling was measured at STOI 0.931.

The speaker embedding is a single frozen x-vector, so every synthesised item
has the same voice and that voice is audibly not the stimulus speaker — a
listener can never mistake synthesis for playback of the presented audio.

This module is deliberately content-agnostic: it takes text and returns a
waveform.  Whether that text was decoded correctly from EEG is a separate
measurement and must never be inferred from how clear the audio sounds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / 'models/aligned_local_base/speecht5_tts'
DEFAULT_VOCODER = ROOT / 'outputs/aligned_speech_local_v1/hifigan/best'
RATE = 16000


class FixedVoiceSynthesizer:
    def __init__(self, model_root: Path = DEFAULT_MODEL, vocoder_root: Path = DEFAULT_VOCODER, device=None):
        from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor
        import sys
        sys.path.insert(0, str(ROOT / 'app/src'))
        from eeg2speech.speecht5 import SpeechT5HiFiGan
        self.device = torch.device(device or 'cpu')
        self.processor = SpeechT5Processor.from_pretrained(str(model_root), local_files_only=True)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(str(model_root), local_files_only=True).to(self.device).eval()
        self.speaker = torch.from_numpy(np.load(model_root / 'speaker_embedding.npy')).unsqueeze(0).to(self.device)
        self.vocoder = SpeechT5HiFiGan(vocoder_root, device=self.device)

    @torch.no_grad()
    def mel(self, text: str) -> torch.Tensor:
        """(80, frames) native SpeechT5 mel for one sentence."""
        inputs = self.processor(text=text, return_tensors='pt').to(self.device)
        spectrogram = self.model.generate_speech(inputs['input_ids'], self.speaker, vocoder=None)
        return spectrogram.T.contiguous()

    @torch.no_grad()
    def waveform(self, text: str) -> np.ndarray:
        return self.vocoder.synthesize(self.mel(text)[None])[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', nargs='*', help='sentences to synthesise')
    parser.add_argument('--from-transcripts', type=int, default=0, help='synthesise this many validation reference transcripts instead')
    parser.add_argument('--output', default=str(ROOT / 'outputs/aligned_recovery_v3/synthesis_check'))
    parser.add_argument('--model', default=str(DEFAULT_MODEL)); parser.add_argument('--vocoder', default=str(DEFAULT_VOCODER))
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    import soundfile as sf
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    texts = list(args.text or [])
    reference = {}
    if args.from_transcripts:
        import pandas as pd
        manifest = pd.read_csv(ROOT / 'artifacts/aligned_speech_local_v1/manifest.csv', keep_default_na=False)
        transcripts = pd.read_csv(ROOT / 'artifacts/aligned_speech_local_v1/official_reference_transcripts.csv', keep_default_na=False)
        by_trial = dict(zip(transcripts.trial_id, transcripts.reference_transcript))
        rows = manifest[manifest.role == 'validation'].drop_duplicates('content_group')
        for _, row in rows.head(args.from_transcripts).iterrows():
            text = by_trial.get(row.trial_id, '')
            if text:
                texts.append(text); reference[text] = ROOT / row.audio_path
    synthesizer = FixedVoiceSynthesizer(Path(args.model), Path(args.vocoder), args.device)
    index = []
    for number, text in enumerate(texts, start=1):
        wave = synthesizer.waveform(text)
        path = output / f'synth{number:02d}.wav'
        sf.write(path, wave, RATE, subtype='FLOAT')
        entry = dict(item=number, text=text, seconds=round(len(wave) / RATE, 2), file=path.name)
        if text in reference and reference[text].exists():
            original, _ = sf.read(reference[text], dtype='float32')
            sf.write(output / f'original{number:02d}.wav', original, RATE, subtype='FLOAT')
            entry.update(original=f'original{number:02d}.wav', original_seconds=round(len(original) / RATE, 2))
        index.append(entry); print(json.dumps(entry), flush=True)
    (output / 'index.json').write_text(json.dumps(index, indent=2) + '\n')


if __name__ == '__main__':
    main()
