#!/usr/bin/env python3
"""Evaluate and optionally export a v3 checkpoint; no human forms required.

Metrics are speech-frame-masked (aligned_recovery_eval).  Exported EEG-driven
waveforms are silenced after the model's own predicted duration by default
(``--tail predicted``); ``--tail oracle`` uses the presented duration and is
labelled as such in the export manifest; ``--tail none`` keeps the raw output.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import aligned_recovery as recovery
from aligned_recovery_model import RecoveryEEGModel, apply_tail
from aligned_recovery_eval import evaluate_recovery, passes_validation_controls
from eeg2speech.aligned_data import wrong_trial_indices

legacy = recovery.legacy


def check_role(payload, role):
    mode = payload['signature']['mode']
    if mode == 'm0' and role != 'train':
        raise ValueError('M0 checkpoints are training-only')
    if role == 'test':
        report = payload.get('evaluation') or {}
        try:
            passed = mode == 'full' and passes_validation_controls(report)
        except KeyError:
            passed = False
        if not passed:
            raise ValueError('test requires a full checkpoint passing validation controls; pilot cannot select on test')


def subject_index_of(payload):
    return ({s: i for i, s in enumerate(payload['signature']['subjects'])}
            if payload['signature']['spec'].get('subjects') else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--role', choices=['train', 'validation', 'test'], default='train')
    parser.add_argument('--config', default=str(recovery.ROOT / 'configs/aligned_speech_local_v1.yaml'))
    parser.add_argument('--device', default='auto'); parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', required=True); parser.add_argument('--export-wavs', action='store_true')
    parser.add_argument('--tail', choices=['predicted', 'oracle', 'none'], default='predicted')
    parser.add_argument('--hifigan', default=str(recovery.ROOT / 'outputs/aligned_speech_local_v1/hifigan/best'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = legacy.config(args.config); target = legacy.device(args.device)
    path = Path(args.checkpoint); payload = recovery.load_checkpoint(path)
    check_role(payload, args.role)
    m0 = payload['signature']['mode'] == 'm0'
    train = legacy.dataset_for(cfg, 'train', m0=m0)
    full_train = train if not m0 else legacy.dataset_for(cfg, 'train')
    signature = recovery.dataset_signature(cfg, train)
    for key, value in signature.items():
        if payload['signature'][key] != value:
            raise ValueError('evaluation data mismatch: ' + key)
    data = legacy.dataset_for(cfg, args.role, m0=m0)
    model = RecoveryEEGModel(legacy.decoder_from(payload), **payload['signature']['spec']).to(target)
    model.load_state_dict(payload['model']); model.eval()
    subject_index = subject_index_of(payload)
    report = evaluate_recovery(model, data, full_train, target, args.batch_size, subject_index, include_records=True)
    report.update(contract=recovery.CONTRACT, role=args.role, m0=m0, seed=payload['signature']['seed'],
                  checkpoint_sha256=legacy.sha256(path), manifest_sha256=signature['manifest'],
                  interpretation='exploratory_previously_inspected_dataset')
    if not m0:
        report['m0_passed'] = False
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    legacy.atomic_json(output / 'evaluation.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}), flush=True)
    if not args.export_wavs:
        return
    references, source = legacy.official_reference_transcripts(data.frame)
    vocoder = legacy.SpeechT5HiFiGan(Path(args.hifigan), device=target)
    wrong = wrong_trial_indices(data.frame)
    passport = dict(contract=recovery.CONTRACT, checkpoint_sha256=legacy.sha256(path),
                    role=args.role, manifest_sha256=signature['manifest'], tail_policy=args.tail,
                    vocoder_sha256=legacy.tree_hash(Path(args.hifigan)), references_source=source)
    marker = output / 'export_manifest.json'
    if marker.exists() and json.loads(marker.read_text()) != passport:
        raise ValueError('existing waveform export provenance differs')
    legacy.atomic_json(marker, passport)
    index = []; frames = len(model.decoder.mel_times)
    with torch.inference_mode():
        for i in range(len(data)):
            batch = legacy.move(torch.utils.data.default_collate([data[i]]), target)
            swapped = legacy.move(torch.utils.data.default_collate([data[wrong[i]]]), target)
            subject = torch.tensor([subject_index[batch['subject'][0]]], device=target) if subject_index else None
            key = hashlib.sha256(batch['trial_id'][0].encode()).hexdigest()[:20]
            folder = output / 'waveforms' / key; folder.mkdir(parents=True, exist_ok=True)
            sf.write(folder / 'original.wav', batch['wave'][0].cpu().numpy(), 16000, subtype='FLOAT')
            mels = {'native_mel_oracle': batch['mel'], 'teacher_oracle': model.decoder(batch['teacher'])}
            predicted_frames = {}
            for control in ['correct', 'zero', 'wrong_trial', 'time_block_shuffle', 'channel_shuffle']:
                if control == 'correct':
                    eeg = batch['eeg']
                elif control == 'zero':
                    eeg = torch.zeros_like(batch['eeg'])
                elif control == 'wrong_trial':
                    eeg = swapped['eeg']
                else:
                    eeg = legacy.counterfactual_eeg(batch['eeg'], control, time_mask=batch['time_mask'], channel_mask=batch['channel_mask'])
                state = model(eeg, batch['channel_xyz'], batch['channel_mask'], batch['time_mask'], subject)
                if args.tail == 'predicted':
                    cut = (state.duration_fraction * frames).round().long().clamp(1, frames)
                elif args.tail == 'oracle':
                    cut = batch['oracle_duration_frames']
                else:
                    cut = None
                predicted_frames[control] = int(cut[0]) if cut is not None else None
                mels[control] = apply_tail(state.native_mel, cut) if cut is not None else state.native_mel
            for name, mel in mels.items():
                wave = vocoder.synthesize(mel)[0].detach().cpu().numpy()[:64000]
                wave = np.pad(wave, (0, max(0, 64000 - len(wave))))
                sf.write(folder / (name + '.wav'), wave, 16000, subtype='FLOAT')
            index.append(dict(trial_id=batch['trial_id'][0], folder=key, wrong_trial_id=swapped['trial_id'][0],
                              oracle_duration_frames=int(batch['oracle_duration_frames'][0]),
                              tail_frames=predicted_frames, **references[batch['trial_id'][0]]))
            if i % 25 == 0:
                print(json.dumps(dict(exported=i + 1, total=len(data))), flush=True)
    legacy.atomic_json(output / 'index.json', dict(samples=index))


if __name__ == '__main__':
    main()
