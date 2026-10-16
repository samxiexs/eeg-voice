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
* Music (Di Liberto 2020, if given): pieces and participants are split the
  same way (scripts/prepare_diliberto.py); music metrics are reported for the
  validation piece, never used for selection.
* Selection rule: lowest speech-frame mel MAE among evaluations that beat
  every same-decoder control (zero EEG, matched wrong trial, time-block
  shuffle, retrieval above chance) - the v3 gate, now on unseen participants.

Stages are flag presets (app/run_universal.sh): acoustic pretraining with
music (contrastive off, mel + low-level targets on), then joint fine-tuning
(``--initialize`` from the pretraining checkpoint).
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
from universal_model import (ACOUSTIC_FRAMES, ACOUSTIC_RATE, SpatialAugmenter, SpatialConfig, UniversalEEGModel,
                             acoustic_targets, aligned_loss, consistency_loss, correlation_loss, decoder_from_spec,
                             group_covariances, load_trunk, music_negatives, split_views)

ROOT = legacy.ROOT
CONTRACT = 'universal_eeg_v1'
SPEECH_RATE = 16000
SPEECH_DATASET = 'ds004940'
SUMMARY_KEYS = ('native_mel_mae', 'template_mae', 'retrieval_mrr', 'chance_mrr', 'zero_gain', 'wrong_trial_gain',
                'time_block_shuffle_gain', 'envelope_corr', 'envelope_zero_gain', 'duration_corr')


def runtime_hash():
    here = Path(__file__).resolve().parent
    files = [Path(__file__)] + [here / n for n in ('universal_model.py', 'music.py',
                                                   'aligned_recovery_model.py', 'aligned_recovery_eval.py')]
    return hashlib.sha256(json.dumps([legacy.runtime_hash(), *[legacy.sha256(p) for p in files]]).encode()).hexdigest()


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
    return dict(virtual=args.virtual, width=args.width, dropout=args.dropout, positional=bool(args.positional),
                lag_ms=args.lag_ms, harmonics=args.harmonics, input_norm=args.input_norm)


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


# --- music evaluation ------------------------------------------------------------------

@torch.no_grad()
def evaluate_music(model, windows, keys, target, batch_size=32, frame_step=4):
    """Paired controls on music windows: zero EEG, time-block shuffle, and a wrong window of the same trial.

    Mel MAE and low-level correlations per control; time-resolved retrieval
    of each window's teacher sequence among the distinct (piece, time)
    targets of the evaluation set (targets heard by several participants are
    one candidate).
    """
    model.eval()
    decoder = model.decoders['music']
    controls = ('correct', 'zero', 'time_block_shuffle', 'wrong_window')
    mae = {c: [] for c in controls}; env = {c: [] for c in controls}; onset = {c: [] for c in controls}
    predictions, identities, targets = [], [], {}
    for offset in range(0, len(keys), batch_size):
        chunk = keys[offset:offset + batch_size]
        data = legacy.move(windows.load(chunk), target)
        wrong = legacy.move(windows.load([windows.shifted(k) for k in chunk]), target)
        for control in controls:
            eeg, mask = data['eeg'], data['channel_mask']
            if control == 'zero':
                eeg = torch.zeros_like(eeg)
            elif control == 'time_block_shuffle':
                eeg = counterfactual_eeg(eeg, control, time_mask=data['time_mask'], channel_mask=mask)
            elif control == 'wrong_window':
                eeg, mask = wrong['eeg'], wrong['channel_mask']
            state = model(eeg, data['channel_xyz'], mask, data['time_mask'], domain='music',
                          locked=torch.zeros(len(chunk), dtype=torch.bool, device=target))
            mae[control].append((state.native_mel - data['mel']).abs().mean((1, 2)).cpu())
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
          evaluate=evaluate_recovery):
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
    needs_bank = (args.background_mix > 0 or max(args.mix_same_content, args.mix_start or 0.) > 0 or args.spatial_recolour > 0)
    bank = bank_factory(speech_train) if needs_bank else None
    covariances = {}
    if args.spatial_recolour > 0:
        groups = [f'{SPEECH_DATASET}:{s}' for s in speech_train.frame.subject]
        covariances.update(group_covariances(bank.eeg, bank.mask, groups, np.random.default_rng(args.seed + 1)))
        if music_train is not None:
            eeg, masks, music_groups = music_train.eeg_bank_sample(40, np.random.default_rng(args.seed + 2))
            covariances.update(group_covariances(eeg, masks, music_groups, np.random.default_rng(args.seed + 3), per_group=40))
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
        state = model(eeg, batch['channel_xyz'], mask, batch['time_mask'], domain='speech')
        mel_mask, speech_mask = speech_frame_masks(batch['oracle_duration_frames'], speech_decoder.mel_times,
                                                   speech_decoder.speech_times)
        indices = torch.tensor([contents[c] for c in batch['content']], device=target)
        loss, parts = recovery_loss(state, batch['teacher'], batch['mel'], speech_decoder.normalizer, speech_mask, mel_mask,
                                    duration_fraction(batch['oracle_duration_frames'], mel_frames), indices,
                                    mel_weight=mel_weight, temperature=args.temperature,
                                    contrastive_weight=args.contrastive_weight, sequence_weight=args.sequence_weight,
                                    delta_weight=args.delta_weight, duration_weight=args.duration_weight)
        if args.acoustic_weight > 0:
            wanted = acoustic_targets(batch['wave'].float().cpu(), SPEECH_RATE).to(target)
            low, r = correlation_loss(state.acoustic, wanted, acoustic_mask(batch['oracle_duration_frames']))
            loss = loss + args.acoustic_weight * low
            parts.update(acoustic=float(low.detach()), envelope_r=float(r[:, 0].mean().detach()), onset_r=float(r[:, 1].mean().detach()))
        if args.consistency_weight > 0:
            first, second = split_views(mask, rng)
            a = model(eeg * first[:, :, None], batch['channel_xyz'], first, batch['time_mask'], domain='speech')
            b = model(eeg * second[:, :, None], batch['channel_xyz'], second, batch['time_mask'], domain='speech')
            agree = consistency_loss(a, b, speech_decoder.normalizer, speech_mask)
            loss = loss + args.consistency_weight * agree; parts['consistency'] = float(agree.detach())
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
    stopping = [False]
    old_handler = signal.getsignal(signal.SIGINT)
    def interrupt(signum, frame):
        stopping[0] = True
        print('Finishing this update and saving universal progress.', flush=True)
    signal.signal(signal.SIGINT, interrupt)
    print(json.dumps(dict(select_on=select_on, speech_train_trials=len(speech_train), cohorts={k: len(v) for k, v in speech_cohorts.items()},
                          music=music_train is not None, spatial=spatial_config(args).as_dict(),
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
                    reports = {name: evaluate(model, cohort, speech_full_train, target, args.batch_size, None,
                                              bootstrap=False, templates=templates) for name, cohort in speech_cohorts.items()}
                    training = evaluate(model, probe, speech_full_train, target, args.batch_size, None, bootstrap=False, templates=templates)
                    music_reports = {name: evaluate_music(model, windows, music_eval_keys[name], target)
                                     for name, windows in (music_cohorts or {}).items()}
                    report = reports[select_on]
                    score = report['native_mel_mae']; eligible = passes_validation_controls(report)
                    row = dict(update=step, selection=select_on, eligible=eligible, speech=reports, train_probe=training, music=music_reports)
                    history.append(row)
                    if score < best:
                        best = score; save(output / 'best_metric.pt', row)
                    if eligible and score < best_passing:
                        best_passing = score; save(output / 'best_passed.pt', row)
                    legacy.atomic_json(output / 'metrics.json', dict(contract=CONTRACT, signature=signature, history=history))
                    print(json.dumps(dict(update=step, eligible=eligible, selection=select_on, best=round(best, 4),
                                          **{name: summarize(r) for name, r in reports.items()},
                                          **{f'music_{name}': {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}
                                             for name, r in music_reports.items()})), flush=True)
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
    # participants
    p.add_argument('--heldout-subjects', type=int, default=4, help='DS004940 participants removed from training entirely')
    p.add_argument('--select-on', choices=['auto', 'seen', 'unseen'], default='auto')
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
    return p


def check_args(p, args):
    probabilities = [args.background_mix, args.mix_same_content, args.music_mix, args.music_background, args.spatial_geometry,
                     args.spatial_mirror, args.spatial_subset, args.spatial_regional, args.spatial_reference,
                     args.spatial_conduction, args.spatial_recolour] + ([args.mix_start] if args.mix_start is not None else [])
    if (args.batch_size < 2 or args.music_batch < 2 or min(args.updates, args.eval_every, args.width, args.virtual) < 1
            or not all(0 <= v <= 1 for v in probabilities) or args.mix_partners < 1 or args.music_partners < 1
            or args.heldout_subjects < 0 or args.throttle < 0 or not 0 <= args.channel_gain < 1 or args.shift_max < 0
            or not 0 <= args.background_alpha[0] <= args.background_alpha[1] or not math.isfinite(args.lr) or args.lr <= 0
            or (args.initialize and args.initialize_trunk) or bool(args.music_targets) != bool(args.music_decoder)):
        p.error('invalid arguments: check batch sizes, probabilities in [0, 1], ranges, one initialisation, '
                'and give --music-targets with --music-decoder')


def main():
    p = parser(); args = p.parse_args(); check_args(p, args)
    torch.set_num_threads(4)
    cfg = legacy.config(args.config)
    full_train = legacy.dataset_for(cfg, 'train')
    validation = legacy.dataset_for(cfg, 'validation')
    held = heldout_subjects(full_train.frame.subject, args.heldout_subjects)
    speech_train = subset(full_train, [i for i, s in enumerate(full_train.frame.subject) if s not in held])
    cohorts = {'seen': subset(validation, [i for i, s in enumerate(validation.frame.subject) if s not in held])}
    if held:
        cohorts['unseen'] = subset(validation, [i for i, s in enumerate(validation.frame.subject) if s in held])
    source = legacy.load_payload(Path(args.decoder))
    legacy.check_eeg_artifacts(source, cfg)
    if source['stage'] != 'adapt' or source['teacher_sha256'] != full_train.teacher_sha256:
        raise ValueError('matching train-fold adapted speech decoder required')
    signature_extra = dict(dataset_signature(cfg, full_train), speech_decoder=legacy.sha256(Path(args.decoder)),
                           heldout_subjects=held)
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
    train(args, speech_train=speech_train, speech_full_train=full_train, speech_cohorts=cohorts,
          templates=train_templates(full_train), speech_decoder_payload=source, signature_extra=signature_extra,
          music_train=music_train, music_cohorts=music_cohorts, music_decoder_payload=music_payload)


if __name__ == '__main__':
    main()
