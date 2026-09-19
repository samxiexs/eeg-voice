#!/usr/bin/env python3
"""S1: EEG decides *what* was said; a fixed-voice synthesiser decides *how*.

The regression route is abandoned (reports/DS004940_PROGRAMME_STATUS.md §6-7):
single-trial EEG carries no composable unit-level content, so the content head
here is restricted to the closed-set decisions the data does support.

Two decision tasks, deliberately labelled by how much is given away:

* ``sentence41`` — **not cued**: rank all 41 held-out sentences by the
  encoder's fixed-window (1.5 s, no oracle duration) similarity to each
  candidate's speech-teacher sequence.  Chance R1 = 1/41.
* ``ending2`` — **cued**: the carrier context is given, so the decision is
  which of its two possible endings (congruent / incongruent) was heard,
  from the ERP after the critical word.  Chance = 1/2.  This is a
  cue-dependent task in the sense of a speller with a known carrier phrase;
  it must never be reported as free decoding.

Every decision carries a confidence.  Below the threshold the system
**abstains** and produces no audio: with a synthesiser in the loop the output
is always fluent, so a system that always speaks would fabricate fluent
mistakes.  The result is therefore a risk-coverage curve — accuracy among the
most confident x% of trials — reported next to zero-EEG and wrong-trial
controls, which must stay at chance at every coverage.

Audio is synthesised only for accepted decisions, in one fixed voice that is
audibly not the stimulus speaker.  **Clarity of that audio carries no
information about decoding**: it is the synthesiser's, not the brain's.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('ALIGNED_TARGET_CACHE_NAME', 'targets_adapted.h5')
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src'))
import aligned_recovery as recovery
from aligned_recovery_model import RecoveryEEGModel
from aligned_recovery_eval import FIXED_WINDOW_S, matched_wrong_trial_indices
from eeg2speech.losses import counterfactual_eeg

legacy = recovery.legacy
CONTRACT = 'content_pipeline_v1'
CONDITIONS = ('correct', 'zero', 'wrong_trial')


def transcripts() -> dict[str, str]:
    """content group -> reference sentence.

    The transcript table carries one representative trial per sentence, so a
    per-trial lookup silently misses ~94% of trials; the text is a property of
    the sentence and is resolved through the content group instead.
    """
    table = pd.read_csv(ROOT / 'artifacts/aligned_speech_local_v1/official_reference_transcripts.csv', keep_default_na=False)
    manifest = pd.read_csv(ROOT / 'artifacts/aligned_speech_local_v1/manifest.csv', keep_default_na=False)
    joined = manifest.merge(table[['trial_id', 'reference_transcript']], on='trial_id', how='inner')
    missing = set(manifest.content_group) - set(joined.content_group)
    if missing:
        raise RuntimeError(f'{len(missing)} sentences have no reference transcript')
    return dict(zip(joined.content_group, joined.reference_transcript))


def candidate_table(dataset, text_of_content: dict[str, str]) -> pd.DataFrame:
    """One row per held-out sentence: content group, audio key, text, carrier context."""
    rows = dataset.frame.drop_duplicates('content_group')[['content_group', 'audio_key', 'trial_id']].copy()
    rows['text'] = rows.content_group.map(text_of_content).fillna('')
    rows['context'] = rows.text.str.rstrip('.').str.split().str[:-1].str.join(' ').str.lower()
    rows['ending'] = rows.text.str.rstrip('.').str.split().str[-1].str.lower()
    return rows.reset_index(drop=True)


@torch.no_grad()
def sentence_scores(model, dataset, candidates: pd.DataFrame, subjects, device, batch_size: int) -> dict:
    """Per-trial similarity to every candidate sentence, for each EEG condition."""
    fixed = (model.decoder.speech_times < FIXED_WINDOW_S)
    with h5py.File(dataset.cache, 'r') as h5:
        prototypes = torch.stack([F.normalize(model.decoder.normalizer(
            torch.from_numpy(h5['targets'][key]['teacher'][:])) [fixed], dim=-1) for key in candidates.audio_key])
    prototypes = prototypes.to(device)                       # C, F, D
    wrong = matched_wrong_trial_indices(dataset.frame)
    order = {c: i for i, c in enumerate(candidates.content_group)}
    out = {name: [] for name in CONDITIONS}; truth = []; trials = []; subjects_seen = []
    for offset in range(0, len(dataset), batch_size):
        ids = list(range(offset, min(len(dataset), offset + batch_size)))
        batch = legacy.move(torch.utils.data.default_collate([dataset[i] for i in ids]), device)
        swapped = legacy.move(torch.utils.data.default_collate([dataset[wrong[i]] for i in ids]), device)
        subject = torch.tensor([subjects[s] for s in batch['subject']], device=device) if subjects else None
        for name in CONDITIONS:
            if name == 'correct':
                eeg = batch['eeg']
            elif name == 'wrong_trial':
                eeg = swapped['eeg']
            else:
                eeg = counterfactual_eeg(batch['eeg'], name, time_mask=batch['time_mask'], channel_mask=batch['channel_mask'])
            state = model(eeg, batch['channel_xyz'], batch['channel_mask'], batch['time_mask'], subject)
            embedding = F.normalize(model.decoder.normalizer(state.aligned_sequence)[:, fixed], dim=-1)
            out[name].append(torch.einsum('nfd,cfd->nc', embedding, prototypes).cpu() / int(fixed.sum()))
        truth.extend(order[str(c)] for c in batch['content'])
        trials.extend(batch['trial_id']); subjects_seen.extend(batch['subject'])
    return {name: torch.cat(values).numpy() for name, values in out.items()} | dict(
        truth=np.array(truth), trials=trials, subjects=subjects_seen)


def decisions_from_scores(scores: np.ndarray, truth: np.ndarray) -> dict:
    """Top-1 choice and a confidence: the top-two margin in units of the score spread."""
    order = np.argsort(-scores, axis=1)
    best, second = order[:, 0], order[:, 1]
    spread = scores.std(axis=1) + 1e-9
    margin = (np.take_along_axis(scores, best[:, None], 1)[:, 0] - np.take_along_axis(scores, second[:, None], 1)[:, 0]) / spread
    return dict(choice=best, confidence=margin, correct=(best == truth))


def risk_coverage(confidence: np.ndarray, correct: np.ndarray, points=(1., .75, .5, .25, .1)) -> list[dict]:
    order = np.argsort(-confidence)
    rows = []
    for coverage in points:
        take = max(1, int(round(coverage * len(order))))
        selected = correct[order[:take]]
        rows.append(dict(coverage=coverage, accepted=int(take), accuracy=float(selected.mean())))
    return rows


def pooled(scores: np.ndarray, truth: np.ndarray, groups: np.ndarray, sizes, rng, repeats: int = 5) -> dict:
    """Accuracy when k trials of the same sentence are averaged before the decision."""
    report = {}
    for k in sizes:
        chosen, targets = [], []
        for content in np.unique(truth):
            index = np.flatnonzero(truth == content)
            size = len(index) if k == 0 or k >= len(index) else k
            for _ in range(1 if size == len(index) else repeats):
                pick = rng.choice(index, size=size, replace=False)
                chosen.append(scores[pick].mean(0)); targets.append(content)
        stacked = np.stack(chosen); targets = np.array(targets)
        decision = decisions_from_scores(stacked, targets)
        report[str(k)] = dict(trials_per_decision=('all' if k == 0 else k), decisions=len(targets),
                              accuracy=float(decision['correct'].mean()),
                              risk_coverage=risk_coverage(decision['confidence'], decision['correct']))
    return report


# ----------------------------------------------------------------------------- cued ending decision
def ending_features(dataset, device, batch_size: int, conditions=CONDITIONS) -> dict:
    """ERP bin amplitudes in the 0.9 s after the critical word, per EEG condition.

    The critical word's onset comes from the trial's carrier context, which the
    cued task gives away by construction; nothing about the *ending* is used.
    """
    from congruency_probe import BINS, EEG_START, RATE, WINDOW_S, critical_word_onsets, erp_features, labels_of
    frame = dataset.frame.reset_index(drop=True)
    onsets = critical_word_onsets(frame); labels = labels_of(frame)
    wrong = matched_wrong_trial_indices(frame)
    out = {name: [] for name in conditions}
    keep, kept_labels, kept_contexts, kept_trials, kept_subjects = [], [], [], [], []
    texts = transcripts()
    contexts = [str(texts.get(c, '')).rstrip('.').lower() for c in frame.content_group]
    if not all(contexts):
        raise RuntimeError('a trial has no reference sentence; the cued task needs its carrier context')
    with torch.inference_mode():
        for offset in range(0, len(frame), batch_size):
            ids = [i for i in range(offset, min(len(frame), offset + batch_size)) if labels[i] >= 0]
            if not ids:
                continue
            batch = legacy.move(torch.utils.data.default_collate([dataset[i] for i in ids]), device)
            swapped = legacy.move(torch.utils.data.default_collate([dataset[wrong[i]] for i in ids]), device)
            for j, i in enumerate(ids):
                start = int(round((onsets[i] - EEG_START) * RATE))
                if start < 0 or start + int(WINDOW_S * RATE) > batch['eeg'].shape[-1]:
                    continue
                for name in conditions:
                    if name == 'correct':
                        eeg = batch['eeg'][j]
                    elif name == 'wrong_trial':
                        eeg = swapped['eeg'][j]
                    else:
                        eeg = torch.zeros_like(batch['eeg'][j])
                    out[name].append(erp_features(eeg.cpu().numpy(), start))
                keep.append(i); kept_labels.append(int(labels[i]))
                kept_contexts.append(' '.join(contexts[i].split()[:-1])); kept_trials.append(batch['trial_id'][j]); kept_subjects.append(batch['subject'][j])
    return {name: np.stack(values) for name, values in out.items()} | dict(
        labels=np.array(kept_labels), contexts=kept_contexts, trials=kept_trials, subjects=kept_subjects, index=np.array(keep))


def ending_candidates() -> dict[str, dict[int, str]]:
    """context -> {1: congruent sentence, 0: incongruent sentence} over the whole corpus.

    The partner sentence's *text* is part of the cue, not something decoded; it
    may therefore come from any role.
    """
    manifest = pd.read_csv(ROOT / 'artifacts/aligned_speech_local_v1/manifest.csv', keep_default_na=False)
    texts = transcripts()
    rows = manifest.drop_duplicates('content_group')[['content_group', 'stim_file']].copy()
    rows['text'] = rows.content_group.map(texts).fillna('')
    rows = rows[rows.text.astype(bool)]
    rows['label'] = np.where(rows.stim_file.str[:3] == 'NPC', 1, np.where(rows.stim_file.str[:3] == 'NPI', 0, -1))
    rows['context'] = rows.text.str.rstrip('.').str.split().str[:-1].str.join(' ').str.lower()
    table = {}
    for context, group in rows[rows.label >= 0].groupby('context'):
        table[context] = {int(r.label): r.text for _, r in group.iterrows()}
    return table


def run_ending(cfg, args, device, rng) -> dict:
    from karaone_baselines import ShrinkageLDA, Standardizer
    train = legacy.dataset_for(cfg, 'train'); evaluation = legacy.dataset_for(cfg, args.role)
    print(f'extracting ERP features: train {len(train)} trials, {args.role} {len(evaluation)} trials', flush=True)
    fitted = ending_features(train, device, args.batch_size, conditions=('correct',))
    held = ending_features(evaluation, device, args.batch_size)
    scaler = Standardizer().fit(fitted['correct'])
    model = ShrinkageLDA(.2).fit(scaler(fitted['correct']), fitted['labels'])
    report = dict(contract=CONTRACT, task='ending2', cued=True, role=args.role, chance_accuracy=.5,
                  trials=len(held['labels']), conditions={})
    decisions = {}
    for name in CONDITIONS:
        scores = model.decision(scaler(held[name]))
        choice = model.classes[scores.argmax(1)]
        spread = scores.std(1) + 1e-9
        confidence = (scores.max(1) - np.sort(scores, axis=1)[:, -2]) / spread
        correct = choice == held['labels']
        decisions[name] = dict(choice=choice, confidence=confidence, correct=correct)
        report['conditions'][name] = dict(accuracy=float(correct.mean()),
                                          risk_coverage=risk_coverage(confidence, correct))
    # per-sentence vote across the ~17 participants who heard it
    votes = pd.DataFrame(dict(context=held['contexts'], label=held['labels'],
                              choice=decisions['correct']['choice'])).groupby('context').agg(
        label=('label', 'first'), vote=('choice', lambda v: int(v.mean() > .5)), n=('choice', 'size'))
    report['sentence_vote'] = dict(sentences=int(len(votes)), accuracy=float((votes.label == votes.vote).mean()),
                                   trials_per_sentence=float(votes.n.mean()))
    report['decisions'] = decisions; report['held'] = held
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(ROOT / 'outputs/aligned_recovery_v3/full_seed322_positional/best_passed.pt'))
    parser.add_argument('--config', default=str(ROOT / 'configs/aligned_speech_local_v1.yaml'))
    parser.add_argument('--role', choices=['validation', 'test'], default='validation')
    parser.add_argument('--task', choices=['sentence41', 'ending2'], default='sentence41',
                        help='sentence41 = not cued (41 candidates); ending2 = carrier context given, two endings')
    parser.add_argument('--output', default=str(ROOT / 'outputs/content_pipeline/validation'))
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--coverage', type=float, default=.25, help='fraction of trials to accept when synthesising')
    parser.add_argument('--speak', type=int, default=12, help='how many accepted decisions to synthesise')
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4); device = torch.device(args.device); rng = np.random.default_rng(args.seed)
    cfg = legacy.config(args.config)
    payload = recovery.load_checkpoint(Path(args.checkpoint))
    dataset = legacy.dataset_for(cfg, args.role)
    model = RecoveryEEGModel(legacy.decoder_from(payload), **payload['signature']['spec']).to(device)
    model.load_state_dict(payload['model']); model.eval()
    subjects = {s: i for i, s in enumerate(payload['signature']['subjects'])} if payload['signature']['spec'].get('subjects') else None
    if args.task == 'ending2':
        return run_ending_task(cfg, args, device, rng)
    candidates = candidate_table(dataset, transcripts())
    scored = sentence_scores(model, dataset, candidates, subjects, device, args.batch_size)
    truth = scored['truth']
    report = dict(contract=CONTRACT, task='sentence41', cued=False, role=args.role,
                  checkpoint=str(args.checkpoint), candidates=len(candidates), trials=len(truth),
                  chance_accuracy=1 / len(candidates), conditions={})
    for name in CONDITIONS:
        decision = decisions_from_scores(scored[name], truth)
        report['conditions'][name] = dict(accuracy=float(decision['correct'].mean()),
                                          risk_coverage=risk_coverage(decision['confidence'], decision['correct']))
        if name == 'correct':
            best = decision
    report['pooled'] = pooled(scored['correct'], truth, np.array(truth), (1, 2, 4, 8, 0), rng)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(dict(trial_id=scored['trials'], subject=scored['subjects'],
                              true_sentence=candidates.text.values[truth], decoded_sentence=candidates.text.values[best['choice']],
                              confidence=best['confidence'], correct=best['correct']))
    frame.sort_values('confidence', ascending=False).to_csv(output / 'decisions.csv', index=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"{'condition':12s} {'accuracy':>9s} " + ' '.join(f'{int(100*p["coverage"]):>3d}%' for p in report['conditions']['correct']['risk_coverage']))
    for name in CONDITIONS:
        entry = report['conditions'][name]
        print(f"{name:12s} {entry['accuracy']:9.3f} " + ' '.join(f"{p['accuracy']:4.2f}" for p in entry['risk_coverage']))
    print(f"chance {report['chance_accuracy']:.3f}; columns are accuracy among the most confident x% of trials")
    print('pooled (k trials of one sentence averaged): ' + ', '.join(
        f"k={v['trials_per_decision']}: {v['accuracy']:.3f}" for v in report['pooled'].values()))
    # ---- speak only the accepted decisions
    if args.speak:
        from synthesize_text import FixedVoiceSynthesizer
        accepted = frame.sort_values('confidence', ascending=False).head(max(1, int(round(args.coverage * len(frame)))))
        speak = accepted.head(args.speak)
        synthesizer = FixedVoiceSynthesizer(device=args.device)
        import soundfile as sf
        folder = output / 'spoken'; folder.mkdir(exist_ok=True)
        index = []
        for number, (_, row) in enumerate(speak.iterrows(), start=1):
            wave = synthesizer.waveform(row.decoded_sentence)
            sf.write(folder / f'decision{number:02d}.wav', wave, 16000, subtype='FLOAT')
            index.append(dict(item=number, trial_id=row.trial_id, subject=row.subject, confidence=float(row.confidence),
                              decoded=row.decoded_sentence, truth=row.true_sentence, correct=bool(row.correct)))
            print(json.dumps(index[-1], ensure_ascii=False), flush=True)
        (folder / 'index.json').write_text(json.dumps(dict(
            note='audio clarity is the synthesiser, not the decoding; correctness is the `correct` field',
            coverage=args.coverage, accepted=int(len(accepted)), accuracy_at_coverage=float(accepted.correct.mean()),
            items=index), indent=2, ensure_ascii=False) + '\n')
        print(f'accepted {len(accepted)}/{len(frame)} trials at coverage {args.coverage}: accuracy {accepted.correct.mean():.3f}')


def run_ending_task(cfg, args, device, rng) -> None:
    report = run_ending(cfg, args, device, rng)
    decisions = report.pop('decisions'); held = report.pop('held')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    table = ending_candidates()
    spoken_text = [table.get(c, {}).get(int(v), '') for c, v in zip(held['contexts'], decisions['correct']['choice'])]
    truth_text = [table.get(c, {}).get(int(v), '') for c, v in zip(held['contexts'], held['labels'])]
    frame = pd.DataFrame(dict(trial_id=held['trials'], subject=held['subjects'], context=held['contexts'],
                              decoded_sentence=spoken_text, true_sentence=truth_text,
                              confidence=decisions['correct']['confidence'], correct=decisions['correct']['correct']))
    frame.sort_values('confidence', ascending=False).to_csv(output / 'decisions.csv', index=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"{'condition':12s} {'accuracy':>9s} " + ' '.join(f'{int(100*p["coverage"]):>3d}%' for p in report['conditions']['correct']['risk_coverage']))
    for name in CONDITIONS:
        entry = report['conditions'][name]
        print(f"{name:12s} {entry['accuracy']:9.3f} " + ' '.join(f"{p['accuracy']:4.2f}" for p in entry['risk_coverage']))
    print(f"chance 0.500 | per-sentence vote over {report['sentence_vote']['trials_per_sentence']:.0f} participants: "
          f"{report['sentence_vote']['accuracy']:.3f} of {report['sentence_vote']['sentences']} sentences")
    if args.speak:
        from synthesize_text import FixedVoiceSynthesizer
        import soundfile as sf
        accepted = frame[frame.decoded_sentence.astype(bool)].sort_values('confidence', ascending=False)
        accepted = accepted.head(max(1, int(round(args.coverage * len(frame)))))
        synthesizer = FixedVoiceSynthesizer(device=args.device)
        folder = output / 'spoken'; folder.mkdir(exist_ok=True); index = []
        for number, (_, row) in enumerate(accepted.head(args.speak).iterrows(), start=1):
            sf.write(folder / f'decision{number:02d}.wav', synthesizer.waveform(row.decoded_sentence), 16000, subtype='FLOAT')
            index.append(dict(item=number, trial_id=row.trial_id, subject=row.subject, confidence=float(row.confidence),
                              decoded=row.decoded_sentence, truth=row.true_sentence, correct=bool(row.correct)))
            print(json.dumps(index[-1], ensure_ascii=False), flush=True)
        (folder / 'index.json').write_text(json.dumps(dict(
            note='cued task: the carrier context is given; audio clarity is the synthesiser, not the decoding',
            coverage=args.coverage, accepted=int(len(accepted)), accuracy_at_coverage=float(accepted.correct.mean()),
            items=index), indent=2, ensure_ascii=False) + '\n')
        print(f'accepted {len(accepted)}/{len(frame)} trials at coverage {args.coverage}: accuracy {accepted.correct.mean():.3f}')


if __name__ == '__main__':
    main()
