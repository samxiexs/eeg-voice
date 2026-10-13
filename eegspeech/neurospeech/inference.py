"""EEG-only generation. No target waveform, teacher, transcript or test length input."""
from __future__ import annotations
import torch
from .protocol import encoder_dir, run_dir, require_checkpoint_scope, section
from ..utils import load_checkpoint, read_json

EEG_KEYS = ('eeg', 'auditory', 'positions', 'channel_mask', 'times', 'frame_mask', 'eeg_times')


def eeg_only(batch):
    if not all(key in batch for key in ('eeg', 'positions', 'channel_mask', 'times')):
        raise ValueError('Inference requires EEG, montage, channel mask and fixed core time grid')
    return {key: batch[key] for key in EEG_KEYS if key in batch}


def condition_from_output(output, calibrator=None, system='S2'):
    teacher = output.get('teacher', output.get('content'))
    if teacher is None: raise ValueError('Missing predicted content condition')
    rhythm = output.get('phi', output.get('rhythm'))
    if rhythm is None: rhythm = teacher.new_zeros((*teacher.shape[:2], 2))
    log_variance = output.get('log_variance')
    if system in ('S0', 'S1'):
        rhythm = torch.zeros_like(rhythm)
        reliability = teacher.new_ones((*teacher.shape[:2], 2))
    elif calibrator is not None and log_variance is not None:
        reliability = 1 / (1 + calibrator(log_variance))
    else:
        # A failed/missing calibration is explicitly neutral rather than optimistic.
        reliability = teacher.new_ones((*teacher.shape[:2], 2))
    condition = {'content': teacher, 'rhythm': rhythm, 'reliability': reliability}
    if output.get('times') is not None: condition['times'] = output['times']
    return condition



def load_encoder_state(encoder,state):
    incompatible=encoder.load_state_dict(state,strict=False)
    invalid_missing=[k for k in incompatible.missing_keys if not (k.startswith('reve.model.') and '.down.' not in k and '.up.' not in k)]
    if invalid_missing or incompatible.unexpected_keys:
        raise ValueError(f'Incompatible adapted encoder checkpoint: missing={invalid_missing}, unexpected={incompatible.unexpected_keys}')
    return incompatible

def detach_condition(condition):
    return {key: value.detach() if torch.is_tensor(value) else value for key, value in condition.items()}


def native_decode(mel, stats, vocos):
    mel_stats = stats.get('mel', stats)
    mean = torch.as_tensor(mel_stats['mean'], device=mel.device, dtype=mel.dtype).reshape(1, -1, 1)
    std = torch.as_tensor(mel_stats['std'], device=mel.device, dtype=mel.dtype).reshape(1, -1, 1)
    return vocos(mel.float() * std.float() + mean.float())


class EEGSpeechGenerator:
    def __init__(self, encoder, flow, stats, vocos=None, calibrator=None, system='S2', adapter=None):
        self.encoder = encoder.eval().requires_grad_(False)
        self.flow = flow.eval().requires_grad_(False)
        self.stats, self.vocos, self.calibrator = stats, vocos, calibrator
        self.system, self.adapter = system, adapter
        if calibrator is not None: calibrator.eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, eeg_batch, seed=0, return_waveform=True):
        clean = eeg_only(eeg_batch)
        output = self.encoder(clean, adapter=self.adapter)
        output['times'] = clean['times']
        condition = condition_from_output(output, self.calibrator, self.system)
        device = clean['eeg'].device
        generator = torch.Generator(device=device).manual_seed(int(seed))
        # Fixed core grid. This frame count comes from protocol configuration, never test audio.
        frames = int(getattr(self.flow, 'mel_frames', 376))
        noise = torch.randn(len(clean['eeg']), 100, frames, device=device, generator=generator)
        mel = self.flow.rollout(noise, condition, steps=32, method='heun')
        if not return_waveform: return mel
        if self.vocos is None: raise ValueError('Waveform generation requires the frozen native Vocos decoder')
        return native_decode(mel, self.stats, self.vocos)


def load_generator(cfg, protocol, fold, seed, system, device='cpu', vocos=None, adapter=None):
    from .models import build_encoder
    from .flow import ConditionalFlow
    from .reliability import MonotoneVarianceCalibrator
    encoder = build_encoder(cfg, 'S2' if system == 'S3' else system).to(device)
    enc_saved = load_checkpoint(encoder_dir(cfg, protocol, fold, seed, system) / 'best.pt', device)
    require_checkpoint_scope(enc_saved, cfg, protocol, fold)
    load_encoder_state(encoder,enc_saved['encoder'])
    flow = ConditionalFlow(cfg).to(device)
    saved = load_checkpoint(run_dir(cfg, protocol, fold, seed, system) / 'best.pt', device)
    require_checkpoint_scope(saved, cfg, protocol, fold)
    flow.load_state_dict(saved['flow'])
    calibrator = None
    if system in ('S2', 'S3'):
        path = encoder_dir(cfg, protocol, fold, seed, system) / 'reliability.pt'
        if path.exists():
            cal = load_checkpoint(path, device)
            require_checkpoint_scope(cal, cfg, protocol, fold)
            if cal.get('passed'):
                calibrator = MonotoneVarianceCalibrator().to(device)
                calibrator.load_state_dict(cal['calibrator'])
    stats = enc_saved.get('stats', saved.get('stats'))
    if stats is None: raise ValueError('Checkpoint does not contain train-side normalization statistics')
    return EEGSpeechGenerator(encoder, flow, stats, vocos, calibrator, system, adapter)
