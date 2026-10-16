#!/usr/bin/env python3
"""Reports for recovery-v3 checkpoints: one checkpoint in detail, or several runs pooled.

=== 1. evaluate ===

Evaluate and optionally export a v3 checkpoint; no human forms required.

Metrics are speech-frame-masked (aligned_recovery_eval).  Exported EEG-driven
waveforms are silenced after the model's own predicted duration by default
(``--tail predicted``); ``--tail oracle`` uses the presented duration and is
labelled as such in the export manifest; ``--tail none`` keeps the raw output.

=== 2. aggregate ===

Aggregate recovery-v3 runs (seeds / variants) into one seed-level summary.

For every run directory the selected checkpoint is the evaluation with the
lowest speech-frame mel MAE among evaluations that pass the validation
controls (the same rule that writes ``best_passed.pt``); runs with no passing
evaluation are reported with their best metric and flagged.  Seed-level means
carry a t-interval so that a claim rests on replication, not on one seed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
import soundfile as sf
import aligned_recovery as recovery
from aligned_recovery_model import RecoveryEEGModel, apply_tail
from aligned_recovery_eval import evaluate_recovery, passes_validation_controls
from eeg2speech.aligned_data import wrong_trial_indices
from aligned_recovery_eval import passes_validation_controls


# --- 1. One checkpoint: bootstrap report + waveform export (python app/recovery_reports.py evaluate) ----

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


def evaluate_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--role', choices=['train', 'validation', 'test'], default='train')
    parser.add_argument('--config', default=str(recovery.ROOT / 'configs/aligned_speech_local_v1.yaml'))
    parser.add_argument('--device', default='auto'); parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', required=True); parser.add_argument('--export-wavs', action='store_true')
    parser.add_argument('--tail', choices=['predicted', 'oracle', 'none'], default='predicted')
    parser.add_argument('--hifigan', default=str(recovery.ROOT / 'outputs/aligned_speech_local_v1/hifigan/best'))
    args = parser.parse_args(argv)
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


# --- 2. Several runs: seed-level summary (python app/recovery_reports.py aggregate) ----

KEYS = ('native_mel_mae', 'template_mae', 'zero_gain', 'wrong_trial_gain', 'time_block_shuffle_gain',
        'envelope_corr', 'envelope_template_corr', 'envelope_zero_gain', 'retrieval_r1', 'retrieval_mrr', 'duration_corr')


T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def select(history):
    passing = [row for row in history if passes_validation_controls(row['validation'])]
    pool = passing or history
    best = min(pool, key=lambda row: row['validation']['native_mel_mae'])
    return best, bool(passing)


def summarize(runs):
    rows = []
    for path in runs:
        metrics = json.loads((Path(path) / 'metrics.json').read_text())
        best, passed = select(metrics['history'])
        rows.append(dict(run=str(path), seed=metrics['signature'].get('seed'), update=best['update'], passed=passed,
                         **{k: best['validation'].get(k) for k in KEYS}))
    summary = {}
    n = len(rows)
    for key in KEYS:
        values = [r[key] for r in rows if r[key] is not None]
        if not values:
            continue
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else float('nan')
        half = T95.get(len(values) - 1, 1.96) * sd / math.sqrt(len(values)) if len(values) > 1 else float('nan')
        summary[key] = dict(mean=mean, sd=sd, ci95=[mean - half, mean + half], n=len(values))
    gates = dict(runs=n, passing_runs=sum(r['passed'] for r in rows),
                 zero_gain_positive=sum(1 for r in rows if (r['zero_gain'] or 0) > 0),
                 wrong_trial_gain_positive=sum(1 for r in rows if (r['wrong_trial_gain'] or 0) > 0),
                 time_shuffle_gain_positive=sum(1 for r in rows if (r['time_block_shuffle_gain'] or 0) > 0))
    return dict(runs=rows, seed_level=summary, gates=gates)


def aggregate_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', help='run directories containing metrics.json')
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    result = summarize(args.runs)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"{'seed':>5} {'update':>6} {'pass':>5} " + ' '.join(f'{k[:14]:>14}' for k in KEYS))
    for r in result['runs']:
        print(f"{str(r['seed']):>5} {r['update']:>6} {str(r['passed']):>5} " +
              ' '.join(f"{(r[k] if r[k] is not None else float('nan')):>14.4f}" for k in KEYS))
    print('seed-level mean [95% t-interval]:')
    for key, v in result['seed_level'].items():
        print(f"  {key:26s} {v['mean']:+.4f}  [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  n={v['n']}")
    print('gates:', json.dumps(result['gates']))


def main():
    commands = {'evaluate': evaluate_main, 'aggregate': aggregate_main}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit('usage: python app/recovery_reports.py evaluate|aggregate [options]')
    commands[sys.argv[1]](sys.argv[2:])


if __name__ == '__main__':
    main()
