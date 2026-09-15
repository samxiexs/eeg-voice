"""Speech-frame-masked evaluation for recovery v3 checkpoints.

Every gate metric (``native_mel_mae``, ``template_mae``, the ``*_gain``
controls, retrieval, envelope correlation) is computed on presented-speech
frames only.  Full-window values are reported under ``full_*`` for comparison
with the v2 numbers, never for selection.  The template baseline is the
per-bin MEDIAN over train-fold sentences still being presented at each frame
(the L1-optimal time-only prior); the mean version is ``template_mean_mae``.
"""
from __future__ import annotations

import warnings

import h5py
import numpy as np
import torch
import torch.nn.functional as F

import hashlib

from eeg2speech.aligned import two_way_bootstrap
from eeg2speech.aligned_data import wrong_trial_indices
from eeg2speech.losses import counterfactual_eeg
from aligned_recovery_model import speech_frame_masks, duration_fraction

CONTROLS = ('correct', 'zero', 'wrong_trial', 'time_block_shuffle', 'channel_shuffle')


def matched_wrong_trial_indices(frame):
    """Same-subject, different-content control whose sentence covers the target's speech frames.

    The legacy hash-based choice often picks a shorter sentence, so the swapped
    EEG carries post-offset activity inside the target's speech frames and the
    control loses for a duration reason rather than a content reason.  Prefer
    the closest duration at least as long as the target; fall back to the
    closest duration overall; ties break by trial-id hash.
    """
    if 'stimulus_duration_seconds' not in frame:
        return wrong_trial_indices(frame)
    durations = frame.stimulus_duration_seconds.astype(float).to_numpy()
    tasks = frame.task.astype(str).to_numpy() if 'task' in frame else None
    result = []
    for i, row in frame.iterrows():
        candidates = frame.index[(frame.subject == row.subject) & (frame.content_group != row.content_group)].tolist()
        if not candidates:
            raise ValueError(f'no within-subject wrong trial for {row.trial_id}')
        if tasks is not None:
            # With several tasks per participant, keep the control inside the
            # same task so attention state is not what differs.
            candidates = [j for j in candidates if tasks[j] == tasks[i]] or candidates
        covering = [j for j in candidates if durations[j] >= durations[i]] or candidates
        covering.sort(key=lambda j: (abs(durations[j] - durations[i]),
                                     hashlib.sha256(f"{row.trial_id}:{frame.iloc[j].trial_id}".encode()).hexdigest()))
        result.append(covering[0])
    return result


def chance_mrr(count):
    return float(sum(1 / k for k in range(1, count + 1)) / count)


def train_templates(train_dataset):
    """(speech mean, speech median, full mean, full median) mel templates from train-fold audio.

    The speech templates pool, at every frame, only the sentences that were
    still being presented there, so they never predict padding silence inside
    a validation sentence; frames beyond the longest train sentence repeat the
    last speaking frame.  The full-window templates are the v2 baselines.
    """
    with h5py.File(train_dataset.cache, 'r') as h5:
        keys = sorted(set(train_dataset.frame.audio_key))
        mels = np.stack([h5['targets'][key]['mel'][:] for key in keys])
        frames = np.array([int(h5['targets'][key].attrs['source_samples_16k']) // 256 + 1 for key in keys])
    speaking = np.arange(mels.shape[-1])[None, :] < frames[:, None]
    masked = np.where(speaking[:, None, :], mels, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)       # all-NaN frames past the longest sentence
        speech_mean, speech_median = np.nanmean(masked, 0), np.nanmedian(masked, 0)
    last = int(speaking.any(0).nonzero()[0].max())
    speech_mean[:, last + 1:] = speech_mean[:, last:last + 1]; speech_median[:, last + 1:] = speech_median[:, last:last + 1]
    return tuple(torch.from_numpy(np.asarray(t)).float() for t in (speech_mean, speech_median, mels.mean(0), np.median(mels, 0)))


def move(batch, target):
    return {k: v.to(target) if torch.is_tensor(v) else v for k, v in batch.items()}


def pearson(a, b):
    # CPU float64: MPS has no float64 and per-trial vectors are tiny.
    a = a.detach().cpu().double(); b = b.detach().cpu().double()
    a = a - a.mean(); b = b - b.mean()
    denominator = a.norm() * b.norm()
    return float((a @ b / denominator).clamp(-1, 1)) if denominator > 1e-12 else 0.


def subject_tensor(batch, subject_index, target):
    if subject_index is None:
        return None
    try:
        return torch.tensor([subject_index[s] for s in batch['subject']], device=target)
    except KeyError as error:
        raise ValueError(f'unknown subject for a known-subject model: {error}') from None


def evaluate_recovery(model, dataset, train_dataset, target, batch_size, subject_index=None, *,
                      bootstrap=True, include_records=False, templates=None):
    model.eval()
    speech_mean, speech_median, full_mean, full_median = [t.to(target) for t in
                                                          (templates if templates is not None else train_templates(train_dataset))]
    wrong = matched_wrong_trial_indices(dataset.frame)
    tasks_by_trial = (dict(zip(dataset.frame.trial_id.astype(str), dataset.frame.task.astype(str)))
                      if 'task' in dataset.frame else {})
    records, predictions, targets, masks, embeddings, teachers, durations = [], [], [], [], [], [], []
    with torch.inference_mode():
        for offset in range(0, len(dataset), batch_size):
            ids = list(range(offset, min(len(dataset), offset + batch_size)))
            batch = move(torch.utils.data.default_collate([dataset[i] for i in ids]), target)
            swapped = move(torch.utils.data.default_collate([dataset[wrong[i]] for i in ids]), target)
            subject = subject_tensor(batch, subject_index, target)
            mel_mask, speech_mask = speech_frame_masks(batch['oracle_duration_frames'], model.decoder.mel_times,
                                                       model.decoder.speech_times)
            states = {}
            for control in CONTROLS:
                if control == 'correct':
                    eeg = batch['eeg']
                elif control == 'wrong_trial':
                    eeg = swapped['eeg']
                else:
                    eeg = counterfactual_eeg(batch['eeg'], control, time_mask=batch['time_mask'], channel_mask=batch['channel_mask'])
                states[control] = model(eeg, batch['channel_xyz'], batch['channel_mask'], batch['time_mask'], subject)
            state = states['correct']
            normalized_teacher = model.decoder.normalizer(batch['teacher'])
            weight = speech_mask[:, :, None].float()
            teachers.append((F.normalize((normalized_teacher * weight).sum(1) / weight.sum(1), dim=-1)).cpu())
            embeddings.append((F.normalize((model.decoder.normalizer(state.aligned_sequence) * weight).sum(1) / weight.sum(1), dim=-1)).cpu())
            predictions.append(state.native_mel.cpu()); targets.append(batch['mel'].cpu()); masks.append(mel_mask.cpu())
            fraction = duration_fraction(batch['oracle_duration_frames'], len(model.decoder.mel_times))
            durations.append(torch.stack([state.duration_fraction.cpu(), fraction.cpu()], 1))
            for i in range(len(ids)):
                truth = batch['mel'][i]; m = mel_mask[i]; s = speech_mask[i]
                envelope = truth[:, m].mean(0)
                record = {'trial_id': batch['trial_id'][i], 'subject': batch['subject'][i], 'content': batch['content'][i],
                          'task': tasks_by_trial.get(batch['trial_id'][i], ''), 'speech_frames': int(m.sum()),
                          'template_mae': float((speech_median[:, m] - truth[:, m]).abs().mean()),
                          'template_mean_mae': float((speech_mean[:, m] - truth[:, m]).abs().mean()),
                          'full_template_mae': float((full_median - truth).abs().mean()),
                          'full_template_mean_mae': float((full_mean - truth).abs().mean()),
                          'sequence_mae': float((model.decoder.normalizer(state.aligned_sequence[i])[s] - normalized_teacher[i][s]).abs().mean()),
                          'envelope_template_corr': pearson(speech_median[:, m].mean(0), envelope)}
                errors, full, corr = {}, {}, {}
                for name, value in states.items():
                    errors[name] = float((value.native_mel[i][:, m] - truth[:, m]).abs().mean())
                    full[name] = float((value.native_mel[i] - truth).abs().mean())
                    corr[name] = pearson(value.native_mel[i][:, m].mean(0), envelope)
                record.update(native_mel_mae=errors['correct'], full_native_mel_mae=full['correct'], envelope_corr=corr['correct'])
                for name in CONTROLS[1:]:
                    record[name + '_gain'] = errors[name] - errors['correct']
                    record['full_' + name + '_gain'] = full[name] - full['correct']
                    record['envelope_' + name + '_gain'] = corr['correct'] - corr[name]
                records.append(record)
    pred, truth_mel, mask = torch.cat(predictions), torch.cat(targets), torch.cat(masks)
    emb, teacher = torch.cat(embeddings), torch.cat(teachers)
    duration = torch.cat(durations)
    labels = [r['content'] for r in records]; unique = sorted(set(labels))
    prototypes = torch.stack([teacher[torch.tensor([x == label for x in labels])].mean(0) for label in unique])
    order = (emb @ F.normalize(prototypes, dim=-1).T).argsort(1, descending=True)
    indices = torch.tensor([unique.index(label) for label in labels])
    ranks = (order == indices[:, None]).nonzero()[:, 1].float() + 1
    # Trial-to-trial variance over frames that every evaluated trial contains.
    common = int(mask.sum(1).min())
    variance = float(pred[:, :, :common].var(0, unbiased=False).mean() /
                     truth_mel[:, :, :common].var(0, unbiased=False).mean().clamp_min(1e-8))
    result = {'pairs': len(records), 'unique_contents': len(unique), 'retrieval_r1': float((ranks == 1).float().mean()),
              'retrieval_mrr': float((1 / ranks).mean()), 'chance_r1': 1 / len(unique), 'chance_mrr': chance_mrr(len(unique)),
              'prediction_variance_ratio': variance, 'common_speech_frames': common,
              'duration_corr': pearson(duration[:, 0], duration[:, 1]) if len(duration) > 2 else 0.,
              'duration_mae_s': float((duration[:, 0] - duration[:, 1]).abs().mean() * len(model.decoder.mel_times) * 256 / 16000),
              'metric_scope': 'speech_frames_only; full_* keys use the whole four-second window'}
    numeric = [key for key in records[0] if key not in ('trial_id', 'subject', 'content', 'task')]
    for key in numeric:
        result[key] = float(np.mean([r[key] for r in records]))
    tasks = sorted({r['task'] for r in records} - {''})
    if len(tasks) > 1:
        # Per-task breakdown (e.g. Active vs Passive listening) of the gate quantities.
        result['per_task'] = {task: {key: float(np.mean([r[key] for r in records if r['task'] == task]))
                                     for key in ('native_mel_mae', 'template_mae', 'zero_gain', 'wrong_trial_gain',
                                                 'time_block_shuffle_gain', 'envelope_corr', 'envelope_zero_gain')}
                              | {'pairs': sum(1 for r in records if r['task'] == task)} for task in tasks}
    result['template_improvement'] = 1 - result['native_mel_mae'] / max(result['template_mae'], 1e-8)
    result['beats_template'] = result['template_improvement'] > 0
    result['beats_template_envelope'] = result['envelope_corr'] > result['envelope_template_corr']
    result['beats_wrong_trial'] = result['wrong_trial_gain'] > 0
    result['beats_chance'] = result['retrieval_mrr'] > result['chance_mrr']
    result['m0_passed'] = bool(result['retrieval_r1'] >= .9 - 1e-6 and result['template_improvement'] >= .1 and
                               variance >= .25 and all(result[c + '_gain'] > 0 for c in ('zero', 'wrong_trial', 'time_block_shuffle')))
    if bootstrap:
        result['subject_content_bootstrap'] = {key: two_way_bootstrap(records, key) for key in
                                               ('native_mel_mae', 'zero_gain', 'wrong_trial_gain', 'time_block_shuffle_gain',
                                                'envelope_corr', 'envelope_zero_gain')}
    if include_records:
        result['records'] = records
    return result


def passes_validation_controls(report):
    """Eligibility for best_passed.pt: real EEG must beat every same-decoder counterfactual on speech frames.

    Zero-EEG, duration-matched wrong-trial and time-block-shuffle inputs share
    the frozen decoder, so they are the architecture-fair no-information
    references (mel MAE and, for zero EEG, envelope correlation); retrieval
    must exceed chance MRR.  The blur-optimal median template is reported
    (``beats_template``, ``beats_template_envelope``) but is not a gate: an
    L1 metric rewards blur that a natural-spectrogram decoder cannot emit.
    """
    return bool(report['beats_wrong_trial'] and report['beats_chance'] and report['zero_gain'] > 0
                and report['time_block_shuffle_gain'] > 0 and report['envelope_zero_gain'] > 0)
