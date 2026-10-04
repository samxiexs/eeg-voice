#!/usr/bin/env python3
"""Subject-free joint training on DS004940 speech EEG and (optionally) music EEG.

The model (universal_model.UniversalEEGModel) receives EEG, electrode
positions and a validity mask only.  Participant identity is used on the data
side alone: to hold participants out, to pick same-participant background
trials, and to estimate the spatial covariances used by recolouring.

Protocol
* DS004940: sentences are split as before (train / validation / test); in
  addition ``--heldout-subjects`` participants are removed from training
  entirely.  Validation is reported for two cohorts, validation sentences x
  training participants (``seen``) and validation sentences x held-out
  participants (``unseen``); checkpoints are selected on ``unseen`` by
  default, because an encoder feeding a generative decoder must work for
  people it has never seen.  The test partition is never read here.
* Music (optional, ``--music-targets``; the music preparation scripts were
  removed on 2026-09-27, the prepared MUSIN-G set stays in artifacts/music):
  pieces and participants are split the same way; music metrics are reported
  for the validation piece, never used for selection.
* Selection rule: lowest speech-frame mel MAE among evaluations that beat
  every same-decoder control (zero EEG, matched wrong trial, time-block
  shuffle, retrieval above chance) - the v3 gate, now on unseen participants.

* Spectral target (``--spectral-space``): ``mfcc`` (default) replaces the
  mel L1 by a weighted L1 on standardised MFCC-80 (app/mfcc.py; exactly
  invertible, so nothing is lost for the vocoder): c0-c12 at full weight,
  c13-c79 at ``--mfcc-high-weight``, because scalp EEG reaches the envelope
  and spectral tilt but not the harmonic fine structure
  (app/feature_decodability.py).  Evaluation adds the MFCC metrics with the
  same controls, and selection (``--select-metric mfcc``) uses the weighted
  MFCC MAE among evaluations that pass the v3 gate and the MFCC controls.
* Continuous speech (``--continuous-root/--continuous-targets``, Broderick
  2018 audiobook, scripts/prepare_broderick_windows.py): 4 s windows through
  the same speech head and frozen decoder, time-resolved CLIP with windows of
  the same run and time excluded as negatives, no onset clock (``locked`` 0).

Runs are driven by scripts/run_mfcc_queue.sh (the subject-free speech recipe
plus the MFCC-80 target and continuous speech; ``--content-half`` builds the
two cross-fitting encoders of the diffusion decoder, ``--evaluate-only``
scores checkpoints side by side).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

os.environ.setdefault('ALIGNED_TARGET_CACHE_NAME', 'targets_adapted.h5')
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
import aligned_speech as legacy
from aligned_recovery import EEGBank, dataset_signature, learning_rate, subset, training_probe
from aligned_recovery_eval import evaluate_recovery, passes_validation_controls, train_templates
from aligned_recovery_model import (augment_eeg, average_eeg_many, background_partners, diverse_batches,
                                    duration_fraction, mix_background, mix_probability, recovery_loss,
                                    same_content_partners, same_content_pool, speech_frame_masks)
from eeg2speech.losses import counterfactual_eeg
from aligned_recovery_eval import CONTROLS
from eeg2speech.aligned import two_way_bootstrap
from mfcc import COEFFICIENTS, LOW, MFCCScaler
from universal_model import (ACOUSTIC_FRAMES, ACOUSTIC_RATE, SpatialAugmenter, SpatialConfig, UniversalEEGModel,
                             acoustic_targets, aligned_loss, consistency_loss, correlation_loss, decoder_from_spec,
                             group_covariances, load_trunk, music_negatives, person_contrastive, split_views)

ROOT = legacy.ROOT
CONTRACT = 'universal_eeg_v1'
SPEECH_RATE = 16000
SPEECH_DATASET = 'ds004940'
SUMMARY_KEYS = ('mfcc_mae', 'mfcc_template_mae', 'mfcc_low_mae', 'mfcc_zero_gain', 'mfcc_wrong_trial_gain',
                'mfcc_time_block_shuffle_gain', 'native_mel_mae', 'template_mae', 'retrieval_mrr', 'chance_mrr', 'zero_gain',
                'wrong_trial_gain', 'time_block_shuffle_gain', 'envelope_corr', 'envelope_zero_gain', 'duration_corr')
MEL_FLOOR = -7.                           # native SpeechT5 log-mel floor used before the MFCC rotation (as in generative_recovery)


def runtime_hash():
    here = Path(__file__).resolve().parent
    files = [Path(__file__)] + [here / n for n in ('universal_model.py', 'music.py', 'mfcc.py',
                                                   'aligned_recovery_model.py', 'aligned_recovery_eval.py')]
    return hashlib.sha256(json.dumps([legacy.runtime_hash(), *[legacy.sha256(p) for p in files]]).encode()).hexdigest()


def content_halves(contents, seed: int):
    """The cross-fitting halves of app/generative_recovery.py (same hash), so fold f of an encoder here
    matches fold f of the diffusion decoder's conditioning."""
    ordered = sorted(contents, key=lambda c: hashlib.sha256(f'{seed}:{c}'.encode()).hexdigest())
    return ordered[:len(ordered) // 2], ordered[len(ordered) // 2:]


def heldout_subjects(subjects, count, salt=SPEECH_DATASET):
    """``count`` participants fixed by a hash of their ids (never by results)."""
    return sorted(sorted(set(subjects), key=lambda s: hashlib.sha256(f'heldout:{salt}:{s}'.encode()).hexdigest())[:count])


def acoustic_mask(duration_frames):
    """(B, 64 Hz frames) mask of frames inside the presented sentence."""
    seconds = duration_frames.float() * 256 / SPEECH_RATE
    times = torch.arange(ACOUSTIC_FRAMES, device=duration_frames.device).float() / ACOUSTIC_RATE
    return times[None, :] < seconds[:, None]


def spatial_config(args):
    return SpatialConfig(geometry=args.spatial_geometry, mirror=args.spatial_mirror, subset=args.spatial_subset,
                         keep_low=args.spatial_keep[0], keep_high=args.spatial_keep[1], regional=args.spatial_regional,
                         subset_mode=args.spatial_subset_mode, reference=args.spatial_reference,
                         conduction=args.spatial_conduction, recolour=args.spatial_recolour).validate()


def model_spec(args):
    spec = dict(virtual=args.virtual, width=args.width, dropout=args.dropout, positional=bool(args.positional),
                lag_ms=args.lag_ms, harmonics=args.harmonics, input_norm=args.input_norm)
    if args.person_dim:
        spec.update(person_dim=args.person_dim, person_rank=args.person_rank)
    return spec


def calibration_rows(frame, subject, count, exclude_contents=(), salt=''):
    """``count`` trial indices of one person, deterministic (hash order), avoiding the given sentences."""
    rows = [i for i, (s, c) in enumerate(zip(frame.subject, frame.content_group)) if s == subject and c not in exclude_contents]
    rows.sort(key=lambda i: hashlib.sha256(f'calibration:{salt}:{frame.trial_id.iat[i]}'.encode()).hexdigest())
    return rows[:count]


class SpeechCalibration:
    """Fixed calibration windows of every person of the speech cohorts: their training-sentence trials
    (never a validation sentence), loaded once (float16), turned into person vectors by the current model."""
    def __init__(self, full_train, subjects, count):
        self.windows = {}
        for subject in sorted(subjects):
            rows = calibration_rows(full_train.frame, subject, count)
            items = [full_train[i] for i in rows]
            self.windows[subject] = (torch.stack([item['eeg'] for item in items]).half(),
                                     torch.stack([item['channel_mask'] for item in items]), items[0]['channel_xyz'])

    @torch.no_grad()
    def vectors(self, model, target):
        out = {}
        for subject, (eeg, mask, xyz) in self.windows.items():
            out[subject] = model.person_embedding(eeg.float().to(target), xyz.to(target), mask.to(target), [len(eeg)])[0]
        return out


def load_universal(path):
    """Rebuild a trained model from a checkpoint (for evaluation, export or a generative stage)."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('contract') != CONTRACT:
        raise ValueError(f'{path}: not a {CONTRACT} checkpoint')
    decoders = {name: decoder_from_spec(spec) for name, spec in payload['decoder_specs'].items()}
    model = UniversalEEGModel(decoders, **payload['model_spec'])
    model.load_state_dict(payload['model'])
    return model, payload


def summarize(report):
    return {k: round(float(report[k]), 4) for k in SUMMARY_KEYS if k in report}


# --- MFCC-80 spectral target -----------------------------------------------------------------

def mfcc_weights(low=LOW, high_weight=.1):
    weights = torch.full((COEFFICIENTS,), float(high_weight)); weights[:low] = 1.
    return weights


def mfcc_loss(prediction, target, mask, scaler, weights):
    """Weighted L1 on standardised MFCC-80 over masked frames (+ 0.2 x the same on frame differences).

    The coefficient weights are normalised to sum to one, so the value is a
    per-frame weighted mean absolute error in standard deviations.
    """
    zp, zt = scaler.encode(prediction), scaler.encode(target)
    w = (weights / weights.sum()).to(zp)[None, :, None]
    m = mask[:, None, :].to(zp.dtype)
    l1 = ((zp - zt).abs() * w * m).sum() / m.sum().clamp_min(1.)
    both = (mask[:, 1:] & mask[:, :-1])[:, None, :].to(zp.dtype)
    delta = ((zp.diff(dim=-1) - zt.diff(dim=-1)).abs() * w * both).sum() / both.sum().clamp_min(1.)
    return l1 + .2 * delta


def target_mels(cfg, keys):
    """{audio_key: (native mel (80, 251), speech frames)} from the DS004940 target cache."""
    import h5py
    _, _, cache, _ = legacy.artifact_paths(cfg)
    out = {}
    with h5py.File(cache, 'r') as h5:
        for k in sorted(set(keys)):
            g = h5['targets'][k]
            out[k] = (torch.from_numpy(g['mel'][:].astype(np.float32)), int(g.attrs['source_samples_16k']) // 256 + 1)
    return out


def fit_mfcc_scaler(mels):
    """Per-coefficient MFCC-80 statistics over the speech frames of the train-fold sentences."""
    frames = torch.cat([mel[:, :min(n, mel.shape[1])] for mel, n in mels.values()], 1)
    return MFCCScaler.fit(frames[None], MEL_FLOOR)


class Recorder(torch.nn.Module):
    """Forwards to the model and keeps every native-mel output, so evaluate_recovery's own passes
    (one per control, in CONTROLS order) also yield the MFCC metrics without extra forward passes.

    With ``persons`` ({subject: vector}) every call also receives the person vector of each row: the calls
    arrive batch by batch (``batch_size`` rows of ``frame`` in order), one per control, and every control of
    a row (zero, within-person wrong trial, shuffles) keeps that row's person - the calibration EEG is
    separate data, the controls only replace the trial's own EEG."""
    def __init__(self, model, persons=None, frame=None, batch_size=None):
        super().__init__()
        self.inner = model
        self.outputs = []
        self.persons, self.frame, self.batch_size = persons, frame, batch_size

    @property
    def decoder(self):
        return self.inner.decoder

    def forward(self, *args, **kwargs):
        if self.persons is not None:
            batch = len(self.outputs) // len(CONTROLS)
            first = batch * self.batch_size
            subjects = self.frame.subject.iloc[first:first + len(args[0])]
            kwargs['person'] = torch.stack([self.persons[s] for s in subjects])
        state = self.inner(*args, **kwargs)
        self.outputs.append(state.native_mel.detach().float().cpu())
        return state


def evaluate_speech(model, dataset, train_dataset, target, batch_size, templates, scaler, weights, mels, *,
                    bootstrap=False, include_records=False, persons=None):
    """evaluate_recovery (mel metrics, v3 gate) plus standardised MFCC-80 metrics under the same controls.

    ``mfcc_mae`` is the training-loss weighting (c0-c12 full, the rest
    ``--mfcc-high-weight``); ``mfcc_low_mae`` is c0-c12 unweighted;
    ``mfcc_template_mae`` scores the train-fold speech-median template the
    same way.  Gains are control minus real (positive = real EEG better).
    """
    recorder = Recorder(model, persons, dataset.frame, batch_size)
    report = evaluate_recovery(recorder, dataset, train_dataset, target, batch_size, None, bootstrap=bootstrap,
                               templates=templates, include_records=True)
    records = report.pop('records')
    outputs, k = recorder.outputs, 0
    per_control = {c: [] for c in CONTROLS}
    for offset in range(0, len(dataset), batch_size):
        for control in CONTROLS:
            per_control[control].append(outputs[k]); k += 1
    if k != len(outputs):
        raise RuntimeError('evaluate_recovery call order changed; MFCC metrics would be misaligned')
    per_control = {c: torch.cat(v) for c, v in per_control.items()}
    w = (weights / weights.sum())[:, None]
    template = templates[1].float().cpu()
    rows = {c: [] for c in CONTROLS}; low = []; template_rows = []
    for i, key in enumerate(dataset.frame.audio_key):
        truth, _ = mels[key]
        n = int(records[i]['speech_frames'])
        zt = scaler.encode(truth[:, :n])
        for c in CONTROLS:
            diff = (scaler.encode(per_control[c][i][:, :n]) - zt).abs()
            rows[c].append(float((diff * w).sum(0).mean()))
            if c == 'correct':
                low.append(float(diff[:LOW].mean()))
        template_rows.append(float(((scaler.encode(template[:, :n]) - zt).abs() * w).sum(0).mean()))
    report['mfcc_mae'] = float(np.mean(rows['correct']))
    report['mfcc_low_mae'] = float(np.mean(low))
    report['mfcc_template_mae'] = float(np.mean(template_rows))
    for c in CONTROLS[1:]:
        report[f'mfcc_{c}_gain'] = float(np.mean(np.array(rows[c]) - np.array(rows['correct'])))
    for i, record in enumerate(records):
        record.update(mfcc_mae=rows['correct'][i], mfcc_low_mae=low[i], mfcc_template_gain=template_rows[i] - rows['correct'][i],
                      **{f'mfcc_{c}_gain': rows[c][i] - rows['correct'][i] for c in CONTROLS[1:]})
    if bootstrap:
        report['mfcc_bootstrap'] = {key: two_way_bootstrap(records, key) for key in MFCC_BOOTSTRAP_KEYS}
    if include_records:
        report['records'] = records
    return report


MFCC_BOOTSTRAP_KEYS = ('mfcc_mae', 'mfcc_low_mae', 'mfcc_template_gain', 'mfcc_zero_gain', 'mfcc_wrong_trial_gain',
                       'mfcc_time_block_shuffle_gain')


def evaluate_checkpoints(args, paths, cohorts, full_train, templates, scaler, mels):
    """--evaluate-only: every checkpoint on every cohort with the MFCC + mel metrics and crossed
    subject/content bootstrap CIs; with two checkpoints, paired per-trial differences (second minus first,
    positive MFCC-MAE difference = the second is worse) with the same bootstrap."""
    target = legacy.device(args.device)
    weights = mfcc_weights(args.mfcc_low, args.mfcc_high_weight)
    out = dict(checkpoints=[str(p) for p in paths], mfcc_low=args.mfcc_low, mfcc_high_weight=args.mfcc_high_weight, results={})
    records = {}
    for path in paths:
        model, payload = load_universal(path)
        model.to(target).eval()
        persons = None
        if model.person_dim:
            persons = SpeechCalibration(full_train, {s for c in cohorts.values() for s in c.frame.subject},
                                        args.person_eval_windows).vectors(model, target)
        entry = dict(update=payload.get('step'), selection_cohort=payload.get('selection_cohort'), personal=bool(model.person_dim))
        for name, cohort in cohorts.items():
            report = evaluate_speech(model, cohort, full_train, target, args.batch_size, templates, scaler, weights, mels,
                                     bootstrap=True, include_records=True, persons=persons)
            records[(str(path), name)] = {r['trial_id']: r for r in report.pop('records')}
            entry[name] = report
            print(json.dumps({'checkpoint': str(path), 'cohort': name, **summarize(report)}), flush=True)
        out['results'][str(path)] = entry
    if len(paths) == 2:
        out['paired'] = {}
        for name in cohorts:
            first, second = records[(str(paths[0]), name)], records[(str(paths[1]), name)]
            rows = [dict(subject=a['subject'], content=a['content'],
                         **{k: second[t][k] - a[k] for k in MFCC_BOOTSTRAP_KEYS + ('native_mel_mae', 'envelope_corr', 'wrong_trial_gain')})
                    for t, a in first.items()]
            out['paired'][name] = {k: two_way_bootstrap(rows, k) for k in MFCC_BOOTSTRAP_KEYS + ('native_mel_mae', 'envelope_corr', 'wrong_trial_gain')}
            print(json.dumps({'paired': name, **{k: round(v['mean'], 4) for k, v in out['paired'][name].items()}}), flush=True)
    return out


def passes_mfcc_controls(report):
    """The v3 gate plus: real EEG beats zero, matched wrong-trial and time-block-shuffle EEG in MFCC-80."""
    return bool(passes_validation_controls(report) and all(report[f'mfcc_{c}_gain'] > 0
                                                          for c in ('zero', 'wrong_trial', 'time_block_shuffle')))


# --- music evaluation ------------------------------------------------------------------

@torch.no_grad()
def evaluate_music(model, windows, keys, target, batch_size=32, frame_step=4, *, domain='music', scaler=None, weights=None,
                   person_fn=None):
    """Paired controls on continuous windows (music, or continuous speech with ``domain='speech'``):
    zero EEG, time-block shuffle, and a wrong window of the same trial.  With ``scaler`` (speech) the
    weighted standardised MFCC-80 MAE is reported per control as well.

    Mel MAE and low-level correlations per control; time-resolved retrieval
    of each window's teacher sequence among the distinct (piece, time)
    targets of the evaluation set (targets heard by several participants are
    one candidate).
    """
    model.eval()
    decoder = model.decoders[domain]
    controls = ('correct', 'zero', 'time_block_shuffle', 'wrong_window')
    mae = {c: [] for c in controls}; env = {c: [] for c in controls}; onset = {c: [] for c in controls}
    cepstral = {c: [] for c in controls}
    w = None if scaler is None else (weights / weights.sum()).to(target)[None, :, None]
    predictions, identities, targets = [], [], {}
    for offset in range(0, len(keys), batch_size):
        chunk = keys[offset:offset + batch_size]
        data = legacy.move(windows.load(chunk), target)
        wrong = legacy.move(windows.load([windows.shifted(k) for k in chunk]), target)
        person = person_fn(chunk) if person_fn is not None else None
        for control in controls:
            eeg, mask = data['eeg'], data['channel_mask']
            if control == 'zero':
                eeg = torch.zeros_like(eeg)
            elif control == 'time_block_shuffle':
                eeg = counterfactual_eeg(eeg, control, time_mask=data['time_mask'], channel_mask=mask)
            elif control == 'wrong_window':
                eeg, mask = wrong['eeg'], wrong['channel_mask']
            state = model(eeg, data['channel_xyz'], mask, data['time_mask'], domain=domain,
                          locked=torch.zeros(len(chunk), dtype=torch.bool, device=target), person=person)
            mae[control].append((state.native_mel - data['mel']).abs().mean((1, 2)).cpu())
            if scaler is not None:
                cepstral[control].append(((scaler.encode(state.native_mel) - scaler.encode(data['mel'])).abs() * w).sum(1).mean(1).cpu())
            _, r = correlation_loss(state.acoustic, data['acoustic'])
            env[control].append(r[:, 0].cpu()); onset[control].append(r[:, 1].cpu())
            if control == 'correct':
                z = F.normalize(decoder.normalizer(state.aligned_sequence)[:, ::frame_step], dim=-1)
                predictions.append(z.flatten(1).cpu())
                normalized = F.normalize(decoder.normalizer(data['teacher'])[:, ::frame_step], dim=-1).flatten(1).cpu()
                for i, (piece, start) in enumerate(zip(data['piece'], data['start'])):
                    identity = (piece, round(start, 3))
                    identities.append(identity); targets.setdefault(identity, normalized[i])
    out = {}
    for control in controls:
        out[f'{control}_mel_mae'] = float(torch.cat(mae[control]).mean())
        out[f'{control}_envelope_r'] = float(torch.cat(env[control]).mean())
        out[f'{control}_onset_r'] = float(torch.cat(onset[control]).mean())
    for control in controls[1:]:
        out[f'{control}_gain'] = out[f'{control}_mel_mae'] - out['correct_mel_mae']
        out[f'{control}_envelope_gain'] = out['correct_envelope_r'] - out[f'{control}_envelope_r']
    if scaler is not None:
        for control in controls:
            out[f'{control}_mfcc_mae'] = float(torch.cat(cepstral[control]).mean())
        for control in controls[1:]:
            out[f'{control}_mfcc_gain'] = out[f'{control}_mfcc_mae'] - out['correct_mfcc_mae']
    names = list(targets)
    candidates = torch.stack([targets[n] for n in names])
    similarity = torch.cat(predictions) @ candidates.T
    truth = torch.tensor([names.index(i) for i in identities])
    rank = (similarity > similarity[torch.arange(len(truth)), truth][:, None]).sum(1) + 1
    out.update(windows=len(keys), candidates=len(names), retrieval_mrr=float((1. / rank.float()).mean()),
               chance_mrr=float(sum(1. / k for k in range(1, len(names) + 1)) / len(names)))
    return out


# --- training ------------------------------------------------------------------------------

def train(args, *, speech_train, speech_full_train, speech_cohorts, templates, speech_decoder_payload,
          signature_extra, music_train=None, music_cohorts=None, music_decoder_payload=None, bank_factory=EEGBank,
          evaluate=evaluate_recovery, mfcc_scaler=None, mels=None, continuous_train=None, continuous_cohorts=None):
    legacy.seed_all(args.seed)
    target = legacy.device(args.device)
    rng = np.random.default_rng(args.seed)
    decoders = {'speech': decoder_from_spec(speech_decoder_payload['decoder_spec'], speech_decoder_payload['decoder'])}
    specs = {'speech': speech_decoder_payload['decoder_spec']}
    if music_train is not None:
        decoders['music'] = decoder_from_spec(music_decoder_payload['decoder_spec'], music_decoder_payload['decoder'])
        specs['music'] = music_decoder_payload['decoder_spec']
    model = UniversalEEGModel(decoders, **model_spec(args)).to(target)
    if args.initialize:
        initial, _ = load_universal(args.initialize)
        missing, unexpected = model.load_state_dict(initial.state_dict(), strict=False)
        if unexpected or any(k.split('.')[0] in ('spatial_attention', 'spatial', 'temporal', 'blocks') for k in missing):
            raise ValueError(f'--initialize does not match this architecture (missing {missing[:3]}, unexpected {unexpected[:3]})')
        print(f'initialised from {args.initialize}', flush=True)
    elif args.initialize_trunk:
        print(f'loaded trunk tensors {load_trunk(model, args.initialize_trunk)[:3]}... from {args.initialize_trunk}', flush=True)
    select_on = args.select_on if args.select_on != 'auto' else ('unseen' if 'unseen' in speech_cohorts else 'seen')
    if select_on not in speech_cohorts:
        raise ValueError(f'no {select_on} speech cohort (use --heldout-subjects > 0 for unseen)')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    use_mfcc = args.spectral_space == 'mfcc'
    if use_mfcc and (mfcc_scaler is None or mels is None):
        raise ValueError('the MFCC spectral space needs the train-fold MFCC scaler and target mels')
    weights = mfcc_weights(args.mfcc_low, args.mfcc_high_weight)
    if use_mfcc:
        signature_extra = dict(signature_extra, mfcc_scaler=mfcc_scaler.state())
    signature = dict(signature_extra, contract=CONTRACT, seed=args.seed, args={k: v for k, v in sorted(vars(args).items())
                                                                                if k not in ('output', 'device', 'throttle')})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    step = epoch = next_batch = 0; best = best_passing = math.inf; history = []
    progress = output / 'training_state.pt'
    if progress.exists():
        saved = torch.load(progress, map_location='cpu', weights_only=False)
        if saved.get('runtime_hash') != runtime_hash() or saved['signature'] != signature:
            raise ValueError('resume arguments/data/code changed; use a new output directory')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        for item in optimizer.state.values():
            for key, value in item.items():
                if torch.is_tensor(value):
                    item[key] = value.to(target)
        step, epoch, next_batch = saved['step'], saved['epoch'], saved['next_batch']
        best, best_passing, history = saved['best'], saved['best_passing'], saved['history']
        torch.set_rng_state(saved['torch_rng']); np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
        rng.bit_generator.state = saved['generator']
        if target.type == 'mps' and saved.get('mps_rng') is not None:
            torch.mps.set_rng_state(saved['mps_rng'])
        if saved['complete']:
            print('universal training already complete'); return
        print(f'resuming universal update {step}', flush=True)

    # --- data-side helpers (participant ids appear only here) ---
    fetch_cache = {}
    def fetch(i):
        if i not in fetch_cache:
            if len(fetch_cache) > 512:
                fetch_cache.clear()
            fetch_cache[i] = speech_train[i]
        return fetch_cache[i]
    contents = {c: i for i, c in enumerate(sorted(set(speech_full_train.frame.content_group)))}
    personal = args.person_dim > 0
    needs_bank = (args.background_mix > 0 or max(args.mix_same_content, args.mix_start or 0.) > 0 or args.spatial_recolour > 0
                  or personal)
    bank = bank_factory(speech_train) if needs_bank else None
    by_subject = speech_train.frame.groupby('subject').indices
    train_contents = speech_train.frame.content_group.to_numpy()
    calibration = (SpeechCalibration(speech_full_train, {s for cohort in speech_cohorts.values() for s in cohort.frame.subject},
                                     args.person_eval_windows) if personal else None)

    def person_views(sets, xyz):
        """sets = [(view_a windows, view_b windows)] of (eeg, mask) pairs -> (person vectors, contrastive loss)."""
        eeg = torch.stack([w[0] for pair in sets for view in pair for w in view]).to(target)
        mask = torch.stack([w[1] for pair in sets for view in pair for w in view]).to(target)
        counts = [len(view) for pair in sets for view in pair]
        vectors = model.person_embedding(eeg, xyz, mask, counts)
        a, b = vectors[0::2], vectors[1::2]
        loss = person_contrastive(a, b, args.person_temperature) if len(sets) > 1 else vectors.sum() * 0.
        return (a + b) / 2, loss

    def speech_persons(subjects, batch_contents, xyz):
        """Two disjoint calibration sets per person of the batch, from their other sentences only."""
        order = list(dict.fromkeys(subjects))
        sets = []
        for subject in order:
            pool = [j for j in by_subject[subject] if train_contents[j] not in batch_contents]
            picks = rng.choice(pool, size=2 * args.person_windows, replace=len(pool) < 2 * args.person_windows)
            windows = [bank[int(j)] for j in picks]
            sets.append((windows[:args.person_windows], windows[args.person_windows:]))
        vectors, loss = person_views(sets, xyz)
        return vectors[torch.tensor([order.index(s) for s in subjects], device=target)], loss

    def continuous_persons(windows, keys, xyz, generator, count):
        """Calibration windows from the same listener's other runs (never the run of the window)."""
        order, sets = [], []
        for row, _ in keys:
            subject = windows.frame.subject.iat[row]
            if subject in order:
                continue
            order.append(subject)
            others = [int(j) for j in windows.by_subject[subject] if int(j) != row] or [row]
            picks = [(int(others[generator.integers(len(others))]), None) for _ in range(2 * count)]
            picks = [(j, float(generator.uniform(0., windows.limits[j]))) for j, _ in picks]
            loaded = [tuple(torch.from_numpy(v) for v in windows.eeg(k)) for k in picks]
            sets.append((loaded[:count], loaded[count:]))
        vectors, loss = person_views(sets, xyz)
        return vectors[torch.tensor([order.index(windows.frame.subject.iat[r]) for r, _ in keys], device=target)], loss
    covariances = {}
    if args.spatial_recolour > 0:
        groups = [f'{SPEECH_DATASET}:{s}' for s in speech_train.frame.subject]
        covariances.update(group_covariances(bank.eeg, bank.mask, groups, np.random.default_rng(args.seed + 1)))
        if music_train is not None:
            eeg, masks, music_groups = music_train.eeg_bank_sample(40, np.random.default_rng(args.seed + 2))
            covariances.update(group_covariances(eeg, masks, music_groups, np.random.default_rng(args.seed + 3), per_group=40))
        if continuous_train is not None:
            eeg, masks, continuous_groups = continuous_train.eeg_bank_sample(40, np.random.default_rng(args.seed + 4))
            covariances.update(group_covariances(eeg, masks, continuous_groups, np.random.default_rng(args.seed + 5), per_group=40))
    augmenter = SpatialAugmenter(spatial_config(args), covariances)
    speech_decoder = model.decoders['speech']
    music_decoder = model.decoders['music'] if music_train is not None else None
    mel_frames = len(speech_decoder.mel_times)

    def pool(eeg, mask, rows):
        """Average each anchor with its partners (rows[i] = list of partner windows' (eeg, mask))."""
        chosen = [i for i, r in enumerate(rows) if r]
        if not chosen:
            return eeg
        slots = max(len(r) for r in rows)
        stacked = torch.zeros(len(rows), slots, *eeg.shape[1:], device=eeg.device)
        stacked_mask = torch.zeros(len(rows), slots, eeg.shape[1], dtype=torch.bool, device=eeg.device)
        for i, r in enumerate(rows):
            for k, (x, m) in enumerate(r):
                stacked[i, k] = x.to(eeg.device); stacked_mask[i, k] = m.to(eeg.device)
        use = torch.zeros(len(rows), 1, 1, dtype=torch.bool, device=eeg.device); use[chosen] = True
        return torch.where(use, average_eeg_many(eeg, stacked, mask, stacked_mask), eeg)

    def background_mix(eeg, mask, noise):
        """noise[i] = (eeg, mask) of a same-participant, other-content trial or None."""
        if not any(n is not None for n in noise):
            return eeg
        noise_eeg = torch.stack([n[0] if n is not None else torch.zeros_like(eeg[0].cpu()) for n in noise]).to(eeg.device)
        noise_mask = torch.stack([n[1] if n is not None else torch.zeros_like(mask[0].cpu()) for n in noise]).to(eeg.device)
        low, high = args.background_alpha
        alpha = (low + torch.rand(len(noise), device=eeg.device) * (high - low)) * torch.tensor(
            [n is not None for n in noise], device=eeg.device)
        return mix_background(eeg, noise_eeg, mask, noise_mask, alpha)

    def augment(eeg, mask, xyz, groups):
        eeg, mask = augmenter(eeg, mask, xyz, groups, rng)
        if args.augment:
            eeg = augment_eeg(eeg, mask, shift_max=args.shift_max, channel_gain=args.channel_gain)
        return eeg, mask

    def speech_loss(ids, partners, background, mel_weight):
        batch = legacy.move(torch.utils.data.default_collate([fetch(i) for i in ids]), target)
        eeg, mask = batch['eeg'], batch['channel_mask']
        eeg = pool(eeg, mask, [[bank[j] for j in partners.get(i, [])] for i in ids])
        eeg = background_mix(eeg, mask, [bank[background[i]] if i in background else None for i in ids])
        eeg, mask = augment(eeg, mask, batch['channel_xyz'], [f'{SPEECH_DATASET}:{s}' for s in batch['subject']])
        person = person_loss = None
        if personal:
            person, person_loss = speech_persons(list(batch['subject']), set(batch['content']), batch['channel_xyz'][0])
        state = model(eeg, batch['channel_xyz'], mask, batch['time_mask'], domain='speech', person=person)
        mel_mask, speech_mask = speech_frame_masks(batch['oracle_duration_frames'], speech_decoder.mel_times,
                                                   speech_decoder.speech_times)
        indices = torch.tensor([contents[c] for c in batch['content']], device=target)
        loss, parts = recovery_loss(state, batch['teacher'], batch['mel'], speech_decoder.normalizer, speech_mask, mel_mask,
                                    duration_fraction(batch['oracle_duration_frames'], mel_frames), indices,
                                    mel_weight=0. if use_mfcc else mel_weight, temperature=args.temperature,
                                    contrastive_weight=args.contrastive_weight, sequence_weight=args.sequence_weight,
                                    delta_weight=args.delta_weight, duration_weight=args.duration_weight)
        if use_mfcc:
            spectral = mfcc_loss(state.native_mel, batch['mel'], mel_mask, mfcc_scaler, weights)
            loss = loss + mel_weight * spectral; parts['mfcc'] = float(spectral.detach())
        if args.acoustic_weight > 0:
            wanted = acoustic_targets(batch['wave'].float().cpu(), SPEECH_RATE).to(target)
            low, r = correlation_loss(state.acoustic, wanted, acoustic_mask(batch['oracle_duration_frames']))
            loss = loss + args.acoustic_weight * low
            parts.update(acoustic=float(low.detach()), envelope_r=float(r[:, 0].mean().detach()), onset_r=float(r[:, 1].mean().detach()))
        if args.consistency_weight > 0:
            first, second = split_views(mask, rng)
            a = model(eeg * first[:, :, None], batch['channel_xyz'], first, batch['time_mask'], domain='speech', person=person)
            b = model(eeg * second[:, :, None], batch['channel_xyz'], second, batch['time_mask'], domain='speech', person=person)
            agree = consistency_loss(a, b, speech_decoder.normalizer, speech_mask)
            loss = loss + args.consistency_weight * agree; parts['consistency'] = float(agree.detach())
        if personal:
            loss = loss + args.person_weight * person_loss; parts['person'] = float(person_loss.detach())
        return loss, parts

    def music_loss(mel_weight):
        keys = music_train.random_keys(args.music_batch, rng)
        batch = legacy.move(music_train.load(keys), target)
        eeg, mask = batch['eeg'], batch['channel_mask']
        rows = []
        for key in keys:
            if rng.random() < args.music_mix:
                partners = music_train.partners(key, int(rng.integers(1, args.music_partners + 1)), rng)
                rows.append([tuple(torch.from_numpy(v) for v in music_train.eeg(p)) for p in partners])
            else:
                rows.append([])
        eeg = pool(eeg, mask, rows)
        noise = []
        for key in keys:
            other = music_train.background(key, rng) if rng.random() < args.music_background else None
            noise.append(tuple(torch.from_numpy(v) for v in music_train.eeg(other)) if other is not None else None)
        eeg = background_mix(eeg, mask, noise)
        eeg, mask = augment(eeg, mask, batch['channel_xyz'], batch['group'])
        state = model(eeg, batch['channel_xyz'], mask, batch['time_mask'], domain='music',
                      locked=torch.zeros(len(keys), dtype=torch.bool, device=target))
        frames = torch.ones(batch['teacher'].shape[:2], dtype=torch.bool, device=target)
        mel_frames_mask = torch.ones(len(keys), batch['mel'].shape[-1], dtype=torch.bool, device=target)
        negatives = music_negatives(batch['piece'], batch['start'], music_decoder.normalizer(batch['teacher']),
                                    similarity=args.music_false_negative)
        loss, parts = aligned_loss(state, batch['teacher'], batch['mel'], music_decoder.normalizer, frames, mel_frames_mask,
                                   negatives, temperature=args.temperature, contrastive_weight=args.contrastive_weight,
                                   sequence_weight=args.sequence_weight, delta_weight=args.delta_weight, mel_weight=mel_weight)
        if args.acoustic_weight > 0:
            low, r = correlation_loss(state.acoustic, batch['acoustic'])
            loss = loss + args.acoustic_weight * low
            parts.update(acoustic=float(low.detach()), envelope_r=float(r[:, 0].mean().detach()), onset_r=float(r[:, 1].mean().detach()))
        return loss, parts

    def continuous_loss(spectral_weight):
        """Broderick audiobook windows through the speech head: no onset clock, CLIP negatives exclude same run & time."""
        keys = continuous_train.random_keys(args.continuous_batch, rng)
        batch = legacy.move(continuous_train.load(keys), target)
        eeg, mask = batch['eeg'], batch['channel_mask']
        rows = []
        for key in keys:
            if rng.random() < args.continuous_mix:
                partners = continuous_train.partners(key, int(rng.integers(1, args.continuous_partners + 1)), rng)
                rows.append([tuple(torch.from_numpy(v) for v in continuous_train.eeg(p)) for p in partners])
            else:
                rows.append([])
        eeg = pool(eeg, mask, rows)
        eeg, mask = augment(eeg, mask, batch['channel_xyz'], batch['group'])
        person = person_loss = None
        if personal:
            person, person_loss = continuous_persons(continuous_train, keys, batch['channel_xyz'][0], rng, args.person_windows)
        state = model(eeg, batch['channel_xyz'], mask, batch['time_mask'], domain='speech',
                      locked=torch.zeros(len(keys), dtype=torch.bool, device=target), person=person)
        frames = torch.ones(batch['teacher'].shape[:2], dtype=torch.bool, device=target)
        mel_frames_mask = torch.ones(len(keys), batch['mel'].shape[-1], dtype=torch.bool, device=target)
        negatives = music_negatives(batch['piece'], batch['start'], speech_decoder.normalizer(batch['teacher']),
                                    similarity=args.continuous_false_negative)
        loss, parts = aligned_loss(state, batch['teacher'], batch['mel'], speech_decoder.normalizer, frames, mel_frames_mask,
                                   negatives, temperature=args.temperature, contrastive_weight=args.contrastive_weight,
                                   sequence_weight=args.sequence_weight, delta_weight=args.delta_weight,
                                   mel_weight=0. if use_mfcc else spectral_weight)
        if use_mfcc:
            spectral = mfcc_loss(state.native_mel, batch['mel'], mel_frames_mask, mfcc_scaler, weights)
            loss = loss + spectral_weight * spectral; parts['mfcc'] = float(spectral.detach())
        if args.acoustic_weight > 0:
            low, r = correlation_loss(state.acoustic, batch['acoustic'])
            loss = loss + args.acoustic_weight * low
            parts.update(acoustic=float(low.detach()), envelope_r=float(r[:, 0].mean().detach()), onset_r=float(r[:, 1].mean().detach()))
        if personal:
            loss = loss + args.person_weight * person_loss; parts['person'] = float(person_loss.detach())
        return loss, parts

    def save(path, evaluation=None, complete=False, stop_reason=None):
        legacy.atomic_save(path, dict(
            contract=CONTRACT, runtime_hash=runtime_hash(), signature=signature, model_spec=model_spec(args),
            decoder_specs=specs, model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, epoch=epoch,
            next_batch=next_batch, best=best, best_passing=best_passing, history=history, evaluation=evaluation,
            selection_cohort=select_on, heldout_subjects=signature_extra.get('heldout_subjects'),
            complete=complete, stop_reason=stop_reason, torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
            python_rng=random.getstate(), generator=rng.bit_generator.state,
            mps_rng=torch.mps.get_rng_state() if target.type == 'mps' else None))

    probe = training_probe(speech_train)
    music_eval_keys = {name: windows.grid_keys(limit=args.music_eval_windows, rng=np.random.default_rng(7))
                       for name, windows in (music_cohorts or {}).items()}
    continuous_eval_keys = {name: windows.grid_keys(limit=args.continuous_eval_windows, rng=np.random.default_rng(11))
                            for name, windows in (continuous_cohorts or {}).items()}
    stopping = [False]
    old_handler = signal.getsignal(signal.SIGINT)
    def interrupt(signum, frame):
        stopping[0] = True
        print('Finishing this update and saving universal progress.', flush=True)
    signal.signal(signal.SIGINT, interrupt)
    print(json.dumps(dict(select_on=select_on, speech_train_trials=len(speech_train), cohorts={k: len(v) for k, v in speech_cohorts.items()},
                          music=music_train is not None, spectral_space=args.spectral_space, select_metric=args.select_metric,
                          continuous_train_trials=len(continuous_train) if continuous_train is not None else 0,
                          spatial=spatial_config(args).as_dict(),
                          heldout_subjects=signature_extra.get('heldout_subjects'))), flush=True)
    start, collected = time.monotonic(), []
    try:
        while step < args.updates:
            batches = diverse_batches(speech_train.frame, args.batch_size, args.seed + epoch)
            probability = mix_probability(epoch, args.mix_same_content, args.mix_start, args.mix_anneal_epochs)
            plan_rng = np.random.default_rng(args.seed * 7919 + epoch)
            partners = (same_content_pool(speech_train.frame, probability, plan_rng, args.mix_partners) if args.mix_partners > 1
                        else {i: [j] for i, j in same_content_partners(speech_train.frame, probability, plan_rng).items()})
            background = background_partners(speech_train.frame, args.background_mix, np.random.default_rng(args.seed * 104729 + epoch))
            while next_batch < len(batches) and step < args.updates:
                tick = time.monotonic()
                model.train(); optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate(step, args.lr, args.updates, args.warmup)
                mel_weight = args.mel_weight * min(1., (step + 1) / max(1, args.mel_warmup))
                loss, parts = speech_loss(batches[next_batch], partners, background, mel_weight)
                parts = {f'speech_{k}': v for k, v in parts.items()}
                if music_train is not None and args.music_weight > 0:
                    music, music_parts = music_loss(mel_weight)
                    loss = loss + args.music_weight * music
                    parts.update({f'music_{k}': v for k, v in music_parts.items()})
                if continuous_train is not None and args.continuous_weight > 0:
                    continuous, continuous_parts = continuous_loss(mel_weight)
                    loss = loss + args.continuous_weight * continuous
                    parts.update({f'continuous_{k}': v for k, v in continuous_parts.items()})
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('nonfinite universal loss')
                loss.backward()
                parts['grad_norm'] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True))
                parts['lr'] = optimizer.param_groups[0]['lr']; parts['loss'] = float(loss.detach())
                optimizer.step(); step += 1; next_batch += 1; collected.append(parts)
                if args.throttle > 0:
                    time.sleep(args.throttle * (time.monotonic() - tick))
                if step % 10 == 0:
                    keys = sorted({k for p in collected for k in p})
                    print(json.dumps(dict(update=step, epoch=epoch + 1, seconds=round(time.monotonic() - start, 2),
                                          **{k: round(float(np.mean([p[k] for p in collected if k in p])), 5) for k in keys})), flush=True)
                    collected = []
                if stopping[0]:
                    save(progress); return
                if step % args.eval_every == 0 or step == args.updates:
                    tick = time.monotonic()
                    model.eval()
                    persons = calibration.vectors(model, target) if personal else None
                    if use_mfcc:
                        reports = {name: evaluate_speech(model, cohort, speech_full_train, target, args.batch_size, templates,
                                                         mfcc_scaler, weights, mels, persons=persons)
                                   for name, cohort in speech_cohorts.items()}
                    else:
                        reports = {name: evaluate(model, cohort, speech_full_train, target, args.batch_size, None,
                                                  bootstrap=False, templates=templates) for name, cohort in speech_cohorts.items()}
                    training = evaluate(model, probe, speech_full_train, target, args.batch_size, None, bootstrap=False, templates=templates)
                    music_reports = {name: evaluate_music(model, windows, music_eval_keys[name], target)
                                     for name, windows in (music_cohorts or {}).items()}
                    def continuous_person_fn(windows):
                        if not personal:
                            return None
                        def fn(chunk):
                            with torch.no_grad():
                                return continuous_persons(windows, chunk, torch.from_numpy(windows.channel_xyz).to(target),
                                                          np.random.default_rng(13), args.person_eval_windows // 2)[0]
                        return fn
                    continuous_reports = {name: evaluate_music(model, windows, continuous_eval_keys[name], target, domain='speech',
                                                               scaler=mfcc_scaler if use_mfcc else None, weights=weights,
                                                               person_fn=continuous_person_fn(windows))
                                          for name, windows in (continuous_cohorts or {}).items()}
                    report = reports[select_on]
                    score = report['mfcc_mae'] if args.select_metric == 'mfcc' else report['native_mel_mae']
                    eligible = passes_mfcc_controls(report) if use_mfcc else passes_validation_controls(report)
                    row = dict(update=step, selection=select_on, select_metric=args.select_metric, eligible=eligible, speech=reports,
                               train_probe=training, music=music_reports, continuous=continuous_reports)
                    history.append(row)
                    if score < best:
                        best = score; save(output / 'best_metric.pt', row)
                    if eligible and score < best_passing:
                        best_passing = score; save(output / 'best_passed.pt', row)
                    legacy.atomic_json(output / 'metrics.json', dict(contract=CONTRACT, signature=signature, history=history))
                    print(json.dumps(dict(update=step, eligible=eligible, selection=select_on, best=round(best, 4),
                                          **{name: summarize(r) for name, r in reports.items()},
                                          **{f'music_{name}': {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}
                                             for name, r in music_reports.items()},
                                          **{f'continuous_{name}': {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}
                                             for name, r in continuous_reports.items()})), flush=True)
                    collapsed = len(history) >= 5 and all(
                        h['train_probe']['prediction_variance_ratio'] < .001 and
                        h['train_probe']['retrieval_r1'] <= h['train_probe']['chance_r1'] + .02 for h in history[-3:])
                    save(progress, row, complete=(step >= args.updates or collapsed),
                         stop_reason='collapse' if collapsed else ('budget' if step >= args.updates else None))
                    if collapsed:
                        raise RuntimeError('universal model collapsed at three evaluations; inspect metrics')
                    if args.throttle > 0:
                        time.sleep(args.throttle * (time.monotonic() - tick))
                elif step % 10 == 0:
                    save(progress)
            if next_batch == len(batches):
                epoch += 1; next_batch = 0
        print(f'Finished. Review {output / "metrics.json"}; best_passed.pt exists only when the {select_on} cohort passes every control.',
              flush=True)
    finally:
        signal.signal(signal.SIGINT, old_handler)
        if music_train is not None:
            music_train.close()
        for windows in (music_cohorts or {}).values():
            windows.close()
        if continuous_train is not None:
            continuous_train.close()
        for windows in (continuous_cohorts or {}).values():
            windows.close()
        if stopping[0]:
            raise SystemExit(130)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=str(ROOT / 'configs/aligned_speech_local_v1.yaml'))
    p.add_argument('--decoder', default=str(ROOT / 'outputs/aligned_speech_local_v1/adapt/best_checkpoint.pt'),
                   help='speech AcousticDecoder (the v3 adapt checkpoint)')
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=322)
    p.add_argument('--device', default='auto')
    p.add_argument('--updates', type=int, default=4000)
    p.add_argument('--eval-every', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup', type=int, default=200)
    p.add_argument('--weight-decay', type=float, default=.05)
    p.add_argument('--throttle', type=float, default=0., help='sleep this fraction of every update\'s wall time (cooler, slower)')
    # model
    p.add_argument('--virtual', type=int, default=128, help='virtual channels produced by the spatial attention')
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--harmonics', type=int, default=12)
    p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--lag-ms', type=int, choices=[0, 100, 200, 300, 400], default=0)
    p.add_argument('--no-positional', dest='positional', action='store_false')
    p.add_argument('--input-norm', choices=['trial_rms', 'none'], default='trial_rms')
    p.add_argument('--initialize', help='universal checkpoint to start from (e.g. the acoustic pretraining stage)')
    p.add_argument('--initialize-trunk', help='Broderick trunk or v3 recovery checkpoint: temporal trunk only')
    # objective (defaults = the v3 passing recipe plus the low-level head)
    p.add_argument('--temperature', type=float, default=.1)
    p.add_argument('--contrastive-weight', type=float, default=.5)
    p.add_argument('--sequence-weight', type=float, default=0.)
    p.add_argument('--delta-weight', type=float, default=0.)
    p.add_argument('--duration-weight', type=float, default=.5)
    p.add_argument('--mel-weight', type=float, default=1.)
    p.add_argument('--mel-warmup', type=int, default=100)
    p.add_argument('--acoustic-weight', type=float, default=.5, help='envelope + onset correlation loss (shared head)')
    p.add_argument('--consistency-weight', type=float, default=0., help='two disjoint electrode halves must agree (2 extra passes)')
    p.add_argument('--spectral-space', choices=['mfcc', 'mel'], default='mfcc',
                   help='spectral target of --mel-weight: weighted standardised MFCC-80 (default) or the v3 mel L1')
    p.add_argument('--mfcc-low', type=int, default=LOW, help='MFCC coefficients at full weight (c0 .. c{low-1})')
    p.add_argument('--mfcc-high-weight', type=float, default=.1, help='weight of the higher MFCC coefficients')
    p.add_argument('--select-metric', choices=['auto', 'mfcc', 'mel'], default='auto', help='auto = the spectral space')
    # participants
    p.add_argument('--heldout-subjects', type=int, default=4, help='DS004940 participants removed from training entirely')
    p.add_argument('--select-on', choices=['auto', 'seen', 'unseen'], default='auto')
    p.add_argument('--person-dim', type=int, default=0,
                   help='data-derived person vector (0 = off): a person encoder over the person\'s own calibration EEG '
                        'drives a low-rank channel mixing and a FiLM; participant ids only label which windows share a person')
    p.add_argument('--person-rank', type=int, default=8)
    p.add_argument('--person-windows', type=int, default=8, help='calibration windows per view during training')
    p.add_argument('--person-weight', type=float, default=.1, help='person-contrastive loss weight')
    p.add_argument('--person-temperature', type=float, default=.1)
    p.add_argument('--person-eval-windows', type=int, default=32, help='calibration trials per person at evaluation')
    p.add_argument('--content-half', type=int, choices=[0, 1], default=None,
                   help='cross-fitting: train on this half of the DS004940 training sentences only (content_halves)')
    p.add_argument('--crossfit-seed', type=int, default=322)
    p.add_argument('--evaluate-only', nargs='+', default=None,
                   help='no training: evaluate these universal checkpoints (MFCC + mel metrics, bootstrap CIs, paired if two); '
                        'writes <output>/evaluation.json')
    # temporal / trial-level augmentation (same meaning as app/aligned_recovery.py)
    p.add_argument('--no-augment', dest='augment', action='store_false')
    p.add_argument('--shift-max', type=int, default=0)
    p.add_argument('--channel-gain', type=float, default=0.)
    p.add_argument('--background-mix', type=float, default=0.)
    p.add_argument('--background-alpha', type=float, nargs=2, default=(.2, .8))
    p.add_argument('--mix-same-content', type=float, default=0.)
    p.add_argument('--mix-partners', type=int, default=1)
    p.add_argument('--mix-start', type=float, default=None)
    p.add_argument('--mix-anneal-epochs', type=int, default=0)
    # spatial augmentation (universal_model.py, part 2); probabilities per sample
    p.add_argument('--spatial-geometry', type=float, default=0., help='cap rotation/tilt/shift/electrode jitter')
    p.add_argument('--spatial-mirror', type=float, default=0.)
    p.add_argument('--spatial-subset', type=float, default=0., help='montage dropout (random or regional)')
    p.add_argument('--spatial-keep', type=float, nargs=2, default=(.25, 1.))
    p.add_argument('--spatial-regional', type=float, default=.5)
    p.add_argument('--spatial-subset-mode', choices=['mask', 'interpolate'], default='mask')
    p.add_argument('--spatial-reference', type=float, default=0.)
    p.add_argument('--spatial-conduction', type=float, default=0.)
    p.add_argument('--spatial-recolour', type=float, default=0., help='cross-participant covariance recolouring')
    # music
    p.add_argument('--music-root', default=str(ROOT / 'artifacts/music/diliberto2020'))
    p.add_argument('--music-targets', default=None, help='targets_<teacher>.h5; enables the music domain')
    p.add_argument('--music-decoder', default=None, help='app/music.py best.pt')
    p.add_argument('--music-weight', type=float, default=.5)
    p.add_argument('--music-batch', type=int, default=16)
    p.add_argument('--music-mix', type=float, default=0., help='probability of averaging a window with other presentations')
    p.add_argument('--music-partners', type=int, default=3)
    p.add_argument('--music-background', type=float, default=0.)
    p.add_argument('--music-false-negative', type=float, default=.8, help='teacher similarity above which two windows are not negatives')
    p.add_argument('--music-eval-windows', type=int, default=192)
    # continuous speech (Broderick 2018 audiobook; scripts/prepare_broderick_windows.py + cache_broderick_targets.py)
    p.add_argument('--continuous-root', default=None, help='e.g. artifacts/speech_continuous/broderick2018; enables the domain')
    p.add_argument('--continuous-targets', default=None, help='targets_hubert.h5 of that root (default: <root>/targets_hubert.h5)')
    p.add_argument('--continuous-weight', type=float, default=.5)
    p.add_argument('--continuous-batch', type=int, default=16)
    p.add_argument('--continuous-mix', type=float, default=0., help='probability of averaging a window with other listeners at the same time')
    p.add_argument('--continuous-partners', type=int, default=3)
    p.add_argument('--continuous-false-negative', type=float, default=.8)
    p.add_argument('--continuous-eval-windows', type=int, default=192)
    return p


def check_args(p, args):
    probabilities = [args.background_mix, args.mix_same_content, args.music_mix, args.music_background, args.continuous_mix, args.spatial_geometry,
                     args.spatial_mirror, args.spatial_subset, args.spatial_regional, args.spatial_reference,
                     args.spatial_conduction, args.spatial_recolour] + ([args.mix_start] if args.mix_start is not None else [])
    if (args.batch_size < 2 or args.music_batch < 2 or min(args.updates, args.eval_every, args.width, args.virtual) < 1
            or not all(0 <= v <= 1 for v in probabilities) or args.mix_partners < 1 or args.music_partners < 1
            or args.heldout_subjects < 0 or args.throttle < 0 or not 0 <= args.channel_gain < 1 or args.shift_max < 0
            or not 0 <= args.background_alpha[0] <= args.background_alpha[1] or not math.isfinite(args.lr) or args.lr <= 0
            or (args.initialize and args.initialize_trunk) or bool(args.music_targets) != bool(args.music_decoder)
            or args.continuous_batch < 2 or args.continuous_partners < 1 or not 0 < args.mfcc_low <= COEFFICIENTS
            or args.mfcc_high_weight < 0 or (args.continuous_targets and not args.continuous_root)
            or (args.select_metric == 'mfcc' and args.spectral_space != 'mfcc')):
        p.error('invalid arguments: check batch sizes, probabilities in [0, 1], ranges, one initialisation, '
                'give --music-targets with --music-decoder, --continuous-targets needs --continuous-root, '
                'and --select-metric mfcc needs --spectral-space mfcc')
    if args.select_metric == 'auto':
        args.select_metric = args.spectral_space


def main():
    p = parser(); args = p.parse_args(); check_args(p, args)
    torch.set_num_threads(4)
    cfg = legacy.config(args.config)
    full_train = legacy.dataset_for(cfg, 'train')
    validation = legacy.dataset_for(cfg, 'validation')
    held = heldout_subjects(full_train.frame.subject, args.heldout_subjects)
    keep = set(content_halves(set(full_train.frame.content_group), args.crossfit_seed)[args.content_half]) if args.content_half is not None else None
    speech_train = subset(full_train, [i for i, (s, c) in enumerate(zip(full_train.frame.subject, full_train.frame.content_group))
                                       if s not in held and (keep is None or c in keep)])
    cohorts = {'seen': subset(validation, [i for i, s in enumerate(validation.frame.subject) if s not in held])}
    if held:
        cohorts['unseen'] = subset(validation, [i for i, s in enumerate(validation.frame.subject) if s in held])
    source = legacy.load_payload(Path(args.decoder))
    legacy.check_eeg_artifacts(source, cfg)
    if source['stage'] != 'adapt' or source['teacher_sha256'] != full_train.teacher_sha256:
        raise ValueError('matching train-fold adapted speech decoder required')
    signature_extra = dict(dataset_signature(cfg, full_train), speech_decoder=legacy.sha256(Path(args.decoder)),
                           heldout_subjects=held, content_half=args.content_half,
                           content_half_sentences=len(keep) if keep is not None else None)
    music_train = music_cohorts = music_payload = None
    if args.music_targets:
        from music import MusicWindows, load_music_decoder
        root = Path(args.music_root)
        manifest, normalizer = root / 'manifest.csv', root / 'normalizer.json'
        _, music_payload = load_music_decoder(args.music_decoder)
        if music_payload['targets_sha256'] != legacy.sha256(Path(args.music_targets)):
            raise ValueError('music decoder was trained on different targets')
        music_train = MusicWindows(ROOT, manifest, args.music_targets, normalizer, content_roles=('train',), subject_roles=('train',))
        music_cohorts = {'seen': MusicWindows(ROOT, manifest, args.music_targets, normalizer, content_roles=('validation',),
                                              subject_roles=('train',)),
                         'unseen': MusicWindows(ROOT, manifest, args.music_targets, normalizer, content_roles=('validation',),
                                                subject_roles=('heldout',))}
        signature_extra.update(music_manifest=legacy.sha256(manifest), music_targets=music_payload['targets_sha256'],
                               music_normalizer=legacy.sha256(normalizer), music_decoder=legacy.sha256(Path(args.music_decoder)))
    mfcc_scaler = mels = None
    if args.evaluate_only:
        mels = target_mels(cfg, list(full_train.frame.audio_key) + list(validation.frame.audio_key))
        mfcc_scaler = fit_mfcc_scaler({k: mels[k] for k in set(full_train.frame.audio_key)})
        result = evaluate_checkpoints(args, [Path(p) for p in args.evaluate_only], cohorts, full_train, train_templates(full_train),
                                      mfcc_scaler, mels)
        Path(args.output).mkdir(parents=True, exist_ok=True)
        legacy.atomic_json(Path(args.output) / 'evaluation.json', result)
        print(f'wrote {Path(args.output) / "evaluation.json"}'); return
    if args.spectral_space == 'mfcc':
        mels = target_mels(cfg, list(full_train.frame.audio_key) + list(validation.frame.audio_key))
        mfcc_scaler = fit_mfcc_scaler({k: mels[k] for k in set(full_train.frame.audio_key)})
    continuous_train = continuous_cohorts = None
    if args.continuous_root:
        from music import MusicWindows
        root = Path(args.continuous_root)
        manifest, normalizer = root / 'manifest.csv', root / 'normalizer.json'
        targets = Path(args.continuous_targets) if args.continuous_targets else root / 'targets_hubert.h5'
        continuous_train = MusicWindows(ROOT, manifest, targets, normalizer, content_roles=('train',), subject_roles=('train',))
        continuous_cohorts = {'seen': MusicWindows(ROOT, manifest, targets, normalizer, content_roles=('validation',), subject_roles=('train',)),
                              'unseen': MusicWindows(ROOT, manifest, targets, normalizer, content_roles=('validation',),
                                                     subject_roles=('heldout',))}
        if continuous_train.targets.teacher_dimension != source['decoder_spec'].get('speech_dimension', 768):
            raise ValueError('continuous-speech teacher does not match the speech decoder')
        signature_extra.update(continuous_manifest=legacy.sha256(manifest), continuous_targets=legacy.sha256(targets),
                               continuous_normalizer=legacy.sha256(normalizer))
    train(args, speech_train=speech_train, speech_full_train=full_train, speech_cohorts=cohorts,
          templates=train_templates(full_train), speech_decoder_payload=source, signature_extra=signature_extra,
          music_train=music_train, music_cohorts=music_cohorts, music_decoder_payload=music_payload,
          mfcc_scaler=mfcc_scaler, mels=mels, continuous_train=continuous_train, continuous_cohorts=continuous_cohorts)


if __name__ == '__main__':
    main()
