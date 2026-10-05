"""Speech targets, the generator's acoustic space and acoustic measures.

Every vocabulary item is synthesised with several voices and speaking rates (macOS ``say``,
16 kHz), so the generator learns a distribution of natural renderings rather than one waveform.

    mel      SpeechT5 log10 mel (80 bins, 16 ms hop = 62.5 frames/s, 80-7600 Hz): the space the
             diffusion model generates in and the input of the SpeechT5 HiFi-GAN vocoder
    hubert   HuBERT-base (LibriSpeech 960 h) hidden states of one layer averaged over the voiced
             part: the speech embedding behind the CLIP space and one identification measure
    mcd      mel-cepstral distance (dB) along a dynamic-time-warping path between two log-mels
    listener Whisper-small: forced choice among the vocabulary (what a listener who knows the
             words would pick) and a free transcription
    speaker  WavLM speaker-verification embedding (x-vector: timbre, gender) and median F0 (pitch):
             the voice traits that condition the generator and are checked in its output

Synthesis needs macOS (``say``); everything else runs anywhere (CUDA, MPS or CPU).
"""
from __future__ import annotations

import copy
from pathlib import Path
import shutil
import subprocess

import numpy as np
import soundfile as sf
import torch

from . import ROOT

SAMPLE_RATE = 16000
MEL_BINS = 80
SILENCE = -10.                   # log10 mel of digital silence (mel floor 1e-10)
MODELS = ROOT / 'models'
VOICES = {                       # natural voices only (no novelty voices); several accents per language
    'en': ['Samantha', 'Daniel', 'Karen', 'Moira', 'Tessa', 'Rishi', 'Eddy (English (US))', 'Flo (English (UK))',
           'Reed (English (US))', 'Sandy (English (UK))', 'Shelley (English (US))', 'Grandpa (English (UK))'],
    'es': ['Mónica', 'Paulina', 'Eddy (Spanish (Spain))', 'Flo (Spanish (Mexico))', 'Reed (Spanish (Spain))',
           'Sandy (Spanish (Mexico))', 'Shelley (Spanish (Spain))', 'Grandpa (Spanish (Mexico))'],
    'zh': ['Tingting', 'Meijia', 'Eddy (Chinese (China mainland))', 'Flo (Chinese (China mainland))',
           'Reed (Chinese (China mainland))', 'Sandy (Chinese (China mainland))', 'Shelley (Chinese (China mainland))'],
}
RATES_WPM = (150, 185, 220)
FEMALE = {'Samantha', 'Karen', 'Moira', 'Tessa', 'Mónica', 'Paulina', 'Tingting', 'Meijia', 'Flo', 'Sandy', 'Shelley'}


def sex_of(voice):
    """'F' or 'M' for a synthetic voice (by the first word of its name)."""
    return 'F' if voice.split()[0] in FEMALE else 'M'


def synthesize(text, voice, wpm, path):
    """One rendering with macOS ``say`` as 16-bit 16 kHz WAV."""
    if shutil.which('say') is None:
        raise RuntimeError('speech synthesis needs macOS `say`; build artifacts/audio on a Mac and copy it here')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['say', '-v', voice, '-r', str(wpm), '-o', str(path), '--file-format=WAVE',
                    f'--data-format=LEI16@{SAMPLE_RATE}', text], check=True)
    return path


def read(path):
    wave, rate = sf.read(path, dtype='float32')
    if rate != SAMPLE_RATE:
        raise ValueError(f'{path}: {rate} Hz, expected {SAMPLE_RATE}')
    return wave if wave.ndim == 1 else wave.mean(1)


def voiced_span(wave, threshold_db=-35., frame=320, margin=2):
    """(start, stop) samples of the part louder than ``threshold_db`` below the loudest 20 ms frame."""
    n = len(wave) // frame
    if n == 0:
        return 0, len(wave)
    energy = 10 * np.log10(np.square(wave[:n * frame].reshape(n, frame)).mean(1) + 1e-12)
    loud = np.flatnonzero(energy > energy.max() + threshold_db)
    first, last = max(loud[0] - margin, 0), min(loud[-1] + 1 + margin, n)
    return first * frame, last * frame


def trim(wave, threshold_db=-35.):
    a, b = voiced_span(wave, threshold_db)
    return wave[a:b]


def level(wave, rms=.05):
    """Scale a (trimmed) waveform to a common loudness: room recordings and synthetic voices alike."""
    w = np.asarray(wave, np.float32) - np.mean(wave)
    r = float(np.sqrt(np.mean(np.square(w)))) if len(w) else 0.
    return w * (rms / r) if r > 0 else w


_EXTRACTOR = None


def log_mel(wave):
    """SpeechT5 log10 mel (80, frames) of one 16 kHz waveform (the vocoder's own frontend)."""
    global _EXTRACTOR
    if _EXTRACTOR is None:
        from transformers import SpeechT5FeatureExtractor
        _EXTRACTOR = SpeechT5FeatureExtractor(sampling_rate=SAMPLE_RATE, num_mel_bins=MEL_BINS, hop_length=16,
                                              win_length=64, win_function='hann_window', fmin=80, fmax=7600,
                                              mel_floor=1e-10, do_normalize=False)
    return _EXTRACTOR._extract_mel_features(np.asarray(wave, np.float32)).T.astype(np.float32)


class Vocoder:
    """SpeechT5 HiFi-GAN: log10 mel (B, 80, frames) -> waveforms (B, frames * 256) at 16 kHz."""

    def __init__(self, device='cpu', path=MODELS / 'speecht5_hifigan'):
        from transformers import SpeechT5HifiGan
        self.device = torch.device(device)
        self.model = SpeechT5HifiGan.from_pretrained(str(path), local_files_only=True).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, mel, batch=64):
        mel = torch.as_tensor(np.asarray(mel, np.float32))
        out = [self.model(mel[i:i + batch].transpose(1, 2).to(self.device)).cpu() for i in range(0, len(mel), batch)]
        return torch.cat(out).numpy()


class Hubert:
    """Mean HuBERT hidden state of the voiced part of each waveform, for every layer (13, 768)."""

    def __init__(self, device='cpu', path=MODELS / 'hubert_base_ls960'):
        from transformers import HubertModel
        self.device = torch.device(device)
        self.model = HubertModel.from_pretrained(str(path), local_files_only=True).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, waves, threshold_db=-35.):
        out = []
        for wave in waves:                     # one at a time: the group-normalised CNN frontend is padding-sensitive
            w = trim(np.asarray(wave, np.float32), threshold_db)
            if len(w) < 1600:                  # under 0.1 s of signal: pad to give the model a few frames
                w = np.pad(w, (0, 1600 - len(w)))
            w = (w - w.mean()) / (w.std() + 1e-7)
            hidden = self.model(torch.from_numpy(w)[None].to(self.device), output_hidden_states=True).hidden_states
            out.append(torch.stack([h[0].mean(0) for h in hidden]).cpu().numpy())
        return np.stack(out).astype(np.float32)


def place(mel, frames, lead=6):
    """A trimmed log-mel placed ``lead`` frames from the start of a ``frames``-long silent canvas."""
    canvas = np.full((mel.shape[0], frames), SILENCE, np.float32)
    n = min(mel.shape[1], frames - lead)
    canvas[:, lead:lead + n] = mel[:, :n]
    return canvas


def cepstra(mel, coefficients=24):
    """Mel cepstra c1..c24 of the frames louder than 35 dB below the peak.

    The mel is a log10 amplitude spectrum; c = sqrt(2 / bins) * DCT-II(ln amplitude) matches the
    SPTK mel-cepstrum scale, so ``dtw_mcd`` gives the usual dB values (resynthesis ~2 dB).
    """
    from scipy.fft import dct
    mel = np.asarray(mel, np.float64)
    energy = mel.mean(0)
    keep = energy > energy.max() - 1.75                      # log10 amplitude: 35 dB
    c = dct(mel[:, keep] * np.log(10), type=2, norm='ortho', axis=0) * np.sqrt(2 / mel.shape[0])
    return c[1:coefficients + 1].T                           # frames, coefficients


def dtw_mcd(pairs, chunk=1024):
    """Mel-cepstral distance (dB) along the DTW path for each (cepstra a, cepstra b) pair, batched over pairs."""
    out = np.zeros(len(pairs))
    k = 10 / np.log(10) * np.sqrt(2)
    for start in range(0, len(pairs), chunk):
        part = pairs[start:start + chunk]
        rows = np.arange(len(part))
        la = np.array([len(a) for a, _ in part]); lb = np.array([len(b) for _, b in part])
        A = np.zeros((len(part), la.max(), part[0][0].shape[1])); Bm = np.zeros((len(part), lb.max(), part[0][1].shape[1]))
        for i, (a, b) in enumerate(part):
            A[i, :len(a)], Bm[i, :len(b)] = a, b
        square = (np.square(A).sum(-1)[:, :, None] + np.square(Bm).sum(-1)[:, None, :]
                  - 2 * np.einsum('pic,pjc->pij', A, Bm))
        cost = k * np.sqrt(np.maximum(square, 0))                                       # P, Ta, Tb
        D = np.full((len(part), la.max() + 1, lb.max() + 1), np.inf); D[:, 0, 0] = 0
        L = np.zeros_like(D)
        for i in range(1, la.max() + 1):
            for j in range(1, lb.max() + 1):
                options = np.stack([D[:, i - 1, j - 1], D[:, i - 1, j], D[:, i, j - 1]])
                best = options.argmin(0)
                D[:, i, j] = options[best, rows] + cost[:, i - 1, j - 1]
                L[:, i, j] = np.stack([L[:, i - 1, j - 1], L[:, i - 1, j], L[:, i, j - 1]])[best, rows] + 1
        out[start:start + len(part)] = D[rows, la, lb] / L[rows, la, lb]
    return out


class Listener:
    """Whisper-small as a listener: forced choice among the vocabulary and a free transcription.

    ``choose`` scores each item by the log-likelihood of its text given the audio (the best of a few
    spellings: capitalisation and final punctuation, as Whisper writes them), like a listener who
    knows the vocabulary; ``transcribe`` says what was heard.  Whisper pads every clip to 30 s, so
    its encoder attention costs ~0.1 GB per clip and layer: batches stay small (32 clips swapped a
    16 GB Mac to a standstill).  On a GPU the model runs in half precision, as Whisper usually does.
    """

    def __init__(self, language, device='cpu', path=MODELS / 'whisper_small'):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.device, self.language = torch.device(device), language
        self.dtype = torch.float16 if self.device.type in ('cuda', 'mps') else torch.float32
        self.processor = WhisperProcessor.from_pretrained(str(path), local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(str(path), local_files_only=True).to(
            device=self.device, dtype=self.dtype).eval()
        tokenizer = self.processor.tokenizer
        self.prefix = tokenizer.convert_tokens_to_ids(['<|startoftranscript|>', f'<|{language}|>', '<|transcribe|>',
                                                       '<|notimestamps|>'])
        self.end = tokenizer.convert_tokens_to_ids('<|endoftext|>')

    def spellings(self, text):
        """Token ids of the ways Whisper may write ``text`` (space-separated languages start with a space)."""
        if self.language in ('zh', 'ja'):
            forms = [text, text + '。', text + '！']
        else:
            forms = [text, text + '.'] + [text.capitalize() + end for end in ('', '.', '!', '?')]
            forms = [' ' + f for f in dict.fromkeys(forms)]
        return [self.processor.tokenizer.encode(f, add_special_tokens=False) for f in forms]

    @torch.no_grad()
    def _encode(self, waves):
        features = self.processor.feature_extractor([np.asarray(w, np.float32) for w in waves], sampling_rate=SAMPLE_RATE,
                                                     return_tensors='pt').input_features
        return self.model.model.encoder(features.to(self.device, self.dtype)).last_hidden_state

    @torch.no_grad()
    def choose(self, waves, texts, batch=4):
        """Log-likelihood (n, K) that each text is what the waveform says.

        The task prefix runs once per batch; its cache, whose cross-attention keys and values over the
        1500 encoder frames are most of the decoder's cost, is shared by every spelling of every text.
        """
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache
        decoder, head = self.model.model.decoder, self.model.proj_out
        variants = [(k, ids) for k, text in enumerate(texts) for ids in self.spellings(text)]
        scores = np.full((len(waves), len(texts)), -np.inf)
        for i in range(0, len(waves), batch):
            encoded = self._encode(waves[i:i + batch])
            n = len(encoded)
            prefix = decoder(input_ids=torch.tensor([self.prefix] * n, device=self.device), encoder_hidden_states=encoded,
                             past_key_values=EncoderDecoderCache(DynamicCache(), DynamicCache()), use_cache=True)
            first = head(prefix.last_hidden_state[:, -1]).float().log_softmax(-1)    # predicts each text's first token
            cache = prefix.past_key_values
            for k, ids in variants:
                own = EncoderDecoderCache(copy.deepcopy(cache.self_attention_cache), cache.cross_attention_cache)
                hidden = decoder(input_ids=torch.tensor([ids] * n, device=self.device), encoder_hidden_states=encoded,
                                 past_key_values=own, use_cache=True).last_hidden_state
                log_prob = head(hidden).float().log_softmax(-1)                       # predicts ids[1:] + <|endoftext|>
                target = torch.tensor(ids[1:] + [self.end], device=self.device)
                ll = first[:, ids[0]] + log_prob.gather(-1, target[None, :, None].expand(n, -1, 1)).squeeze(-1).sum(-1)
                scores[i:i + n, k] = np.maximum(scores[i:i + n, k], ll.float().cpu().numpy())
        return scores

    @torch.no_grad()
    def transcribe(self, waves, batch=4):
        out = []
        for i in range(0, len(waves), batch):
            features = self.processor.feature_extractor([np.asarray(w, np.float32) for w in waves[i:i + batch]],
                                                         sampling_rate=SAMPLE_RATE, return_tensors='pt').input_features
            ids = self.model.generate(features.to(self.device, self.dtype), language=self.language, task='transcribe',
                                      max_new_tokens=24)
            out += [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]
        return out


class Speaker:
    """WavLM speaker-verification embedding (unit x-vector, 512-d) of the voiced part of each waveform."""

    def __init__(self, device='cpu', path=MODELS / 'wavlm_base_plus_sv'):
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        self.device = torch.device(device)
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(path), local_files_only=True)
        self.model = WavLMForXVector.from_pretrained(str(path), local_files_only=True).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, waves):
        out = []
        for wave in waves:                     # one at a time: clips differ in length
            w = trim(np.asarray(wave, np.float32))
            if len(w) < SAMPLE_RATE:             # short words are repeated to 1 s (the x-vector needs > 0.3 s)
                w = np.tile(w, int(np.ceil(SAMPLE_RATE / max(len(w), 1))))
            inputs = self.extractor(w, sampling_rate=SAMPLE_RATE, return_tensors='pt')
            embedding = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).embeddings
            out.append(torch.nn.functional.normalize(embedding, dim=-1)[0].cpu().numpy())
        return np.stack(out).astype(np.float32)


def pitch(wave, rate=SAMPLE_RATE, fmin=60., fmax=500., frame=640, hop=160, threshold=.15):
    """Median F0 (Hz) over the voiced frames (YIN, de Cheveigne & Kawahara 2002); NaN without voicing."""
    w = np.asarray(wave, np.float64)
    if len(w) <= frame:
        return float('nan')
    frames = np.lib.stride_tricks.sliding_window_view(w, frame)[::hop]
    loud = frames.std(1)
    frames = frames[loud > .1 * loud.max()]
    lag_max, lag_min = min(int(rate / fmin), frame // 2), int(rate / fmax)
    width = frame - lag_max
    d = np.stack([np.square(frames[:, :width] - frames[:, lag:lag + width]).sum(1) for lag in range(lag_max)], 1)
    norm = d[:, 1:] / np.maximum(np.cumsum(d[:, 1:], 1) / np.arange(1, lag_max), 1e-12)
    f0 = []
    for row in norm:                           # row[l - 1] is the normalised difference at lag l
        below = np.flatnonzero(row[lag_min - 1:] < threshold)
        if len(below):
            lag = lag_min + below[0]
            while lag < lag_max - 1 and row[lag] < row[lag - 1]:      # walk down to the local minimum
                lag += 1
            f0.append(rate / lag)
    return float(np.median(f0)) if len(f0) >= 3 else float('nan')

