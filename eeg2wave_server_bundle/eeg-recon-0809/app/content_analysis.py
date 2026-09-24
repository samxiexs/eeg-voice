#!/usr/bin/env python3
"""The content route: does EEG say *what* was heard, and can a fixed voice speak it?

Four parts in dependency order, each with its own entry point
(``python app/content_analysis.py {congruency|sentence|pooled|speak} ...``):

=== 1. congruency - Semantic-congruency (N400) probe on DS004940 Active trials ===

Every Active sentence ends in a congruent (NPC) or incongruent (NPI) final
word.  This probe asks whether the final word's congruency can be read from
(a) the raw EEG after the critical word (ERP mean amplitudes, Riemannian
covariance) and (b) the trained encoder's internal tokens over the same
window — i.e. whether the representation carries linguistic information
beyond the acoustic envelope.

Protocol: fit on train-role trials, evaluate on validation-role trials
(content-disjoint, known participants); permutation null by shuffling the
training labels (200 times).  Controls: the same window *before* the
critical word (must be at chance unless context or acoustics differ), and
audio-only features of the final word (if the audio alone separates NPC from
NPI, an EEG result could be acoustic rather than semantic).

=== 2. sentence - S1: EEG decides *what* was said; a fixed-voice synthesiser decides *how* ===

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

=== 3. pooled - S2: pooled sentence identification — EEG picks *which* sentence, TTS says it ===

S1 (`app/py`) asked one trial to choose among 41 held-out
sentences and reached 5.1% (chance 2.4%): too weak to speak.  But §4 of the
programme report showed the same decision is far from dead when the trials of a
sentence are pooled **at the encoder input** — 14.6% R1 and 46.3% top-5 — while
pooling the *scores* of separately decoded trials, which is what S1's
``pooled()`` did, stays at 4.9%.  The encoder is nonlinear and only input
averaging raises its input SNR, so S1 measured the weaker of the two poolings
and concluded the pooled task was hopeless.

This script runs the decision the way §4 says it should be run, and carries it
through to audio:

* pooling is **input averaging** of the participant-mixed EEG (k = 1, 2, 4, 8,
  all ~17 presentations of a sentence);
* two oracle-free evidence channels are scored and compared —
  ``embedding`` (the recovery encoder's aligned sequence against each
  candidate's speech-teacher sequence over a fixed 1.5 s window) and
  ``envelope`` (the envelope decoder's predicted envelope against each
  candidate's envelope, with the time-only prior regressed out of both, over
  the same window) — plus their equal-weight combination;
* every decision carries the top-two margin as a confidence, so the system can
  **abstain**; accuracy is reported against coverage;
* the zero-EEG and wrong-trial (pooling a *different* sentence's trials)
  controls run through the identical path, and a label permutation gives an
  exact null for the 41-way accuracy, which matters because k = all leaves only
  41 decisions;
* accepted decisions are spoken in the one fixed voice.  **The clarity of that
  audio is the synthesiser's, never evidence about the brain**: correctness
  lives only in the ``correct`` field of the manifest.

Nothing here is a single-trial BCI claim.  Pooling presentations of a known
sentence across participants is a group-level statement about the information
in EEG, and the candidate set is closed and known.

=== 4. speak - Fixed-voice text-to-speech for the content-first route ===

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
import os
import sys
from pathlib import Path
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
from karaone import ShrinkageLDA, Standardizer, tangent_features
from aligned_recovery_eval import FIXED_WINDOW_S, matched_wrong_trial_indices
from eeg2speech.losses import counterfactual_eeg
import envelope_decoder as envelope
from aligned_recovery_eval import FIXED_WINDOW_S
from eeg2speech.aligned import EEG_SAMPLES


# --- 1. congruency: N400 semantic-congruency probe -----------------------------

legacy = recovery.legacy


RATE = 256; EEG_START = -.25; WINDOW_S = .9






BINS = ((0., .3), (.3, .5), (.5, .7), (.7, .9))


def critical_word_onsets(frame: pd.DataFrame) -> np.ndarray:
    """Seconds from stimulus onset to the critical (final) word, from the BIDS events."""
    onsets = np.full(len(frame), np.nan)
    for path, rows in frame.groupby('source_event_path'):
        events = pd.read_csv(ROOT / path, sep='\t', keep_default_na=False)
        for index, row in rows.iterrows():
            event = events.iloc[int(row.source_event_row)]
            onsets[index] = float(event['onset']) - float(event['stim_onset_s_'])
    return onsets


def labels_of(frame: pd.DataFrame) -> np.ndarray:
    prefix = frame.stim_file.str[:3]
    return np.where(prefix == 'NPC', 1, np.where(prefix == 'NPI', 0, -1))


def erp_features(eeg: np.ndarray, start: int) -> np.ndarray:
    """Mean amplitude per channel in four post-onset bins; eeg is (channels, samples)."""
    return np.concatenate([eeg[:, start + int(a * RATE): start + int(b * RATE)].mean(1) for a, b in BINS])


def collect(model, dataset, subjects, device, onsets, labels, pre_word: bool):
    """Per-trial feature sets for the window after (or before) the critical word."""
    hooks = {}
    handle = model.output_norm.register_forward_hook(lambda module, inputs, output: hooks.__setitem__('tokens', output.detach()))
    token_times = model.token_times.cpu().numpy()
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(dataset), 32):
            ids = [i for i in range(offset, min(len(dataset), offset + 32)) if labels[i] >= 0]
            if not ids:
                continue
            batch = legacy.move(torch.utils.data.default_collate([dataset[i] for i in ids]), device)
            subject = torch.tensor([subjects[s] for s in batch['subject']], device=device) if subjects else None
            model(batch['eeg'], batch['channel_xyz'], batch['channel_mask'], batch['time_mask'], subject)
            tokens = hooks['tokens'].cpu().numpy(); eeg = batch['eeg'].cpu().numpy(); mel = batch['mel'].cpu().numpy()
            for j, i in enumerate(ids):
                onset = onsets[i] - WINDOW_S if pre_word else onsets[i]
                if onset < 0:
                    continue
                start = int(round((onset - EEG_START) * RATE))
                if start + int(WINDOW_S * RATE) > eeg.shape[-1]:
                    continue
                window = eeg[j, :, start: start + int(WINDOW_S * RATE)]
                selected = (token_times >= onset) & (token_times < onset + WINDOW_S)
                mel_start = int(round(onset / (256 / 16000)))
                rows.append(dict(index=i, label=int(labels[i]), subject=batch['subject'][j], content=batch['content'][j],
                                 erp=erp_features(eeg[j], start), tangent=tangent_features(window[None])[0],
                                 tokens=tokens[j, selected].mean(0), audio=mel[j][:, mel_start: mel_start + int(WINDOW_S * 16000 / 256)].mean(1)))
    handle.remove()
    return rows


def content_permutation(rows, rng):
    """Permute labels at the SENTENCE level, not the trial level.

    Each sentence is heard by ~17 participants and carries one label, so a
    trial-level permutation destroys that structure and yields an
    over-optimistic null: any feature that merely identifies the sentence
    beats it.  Permuting whole sentences keeps the structure in the null.
    """
    contents = sorted({r['content'] for r in rows})
    labels = [next(r['label'] for r in rows if r['content'] == c) for c in contents]
    mapping = dict(zip(contents, rng.permutation(labels)))
    return np.array([mapping[r['content']] for r in rows])


def evaluate(train_rows, test_rows, key, permutations, rng):
    x_train = np.stack([r[key] for r in train_rows]); y_train = np.array([r['label'] for r in train_rows])
    x_test = np.stack([r[key] for r in test_rows]); y_test = np.array([r['label'] for r in test_rows])
    scaler = Standardizer().fit(x_train)
    def accuracy(y):
        model = ShrinkageLDA(.2).fit(scaler(x_train), y)
        predictions = model.predict(scaler(x_test))
        return float((predictions == y_test).mean()), predictions
    observed, predictions = accuracy(y_train)
    null = [accuracy(content_permutation(train_rows, rng))[0] for _ in range(permutations)]
    subjects = sorted({r['subject'] for r in test_rows})
    per_subject = {s: float(np.mean([p == r['label'] for p, r in zip(predictions, test_rows) if r['subject'] == s])) for s in subjects}
    values = np.array(list(per_subject.values())); n = len(values)
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120}.get(n, 1.96)
    half = t * values.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
    balanced = float(np.mean([(predictions[y_test == c] == c).mean() for c in (0, 1)]))
    contents = sorted({r['content'] for r in test_rows})
    per_content = [float(np.mean([p == r['label'] for p, r in zip(predictions, test_rows) if r['content'] == c])) for c in contents]
    votes = [float(np.mean([p for p, r in zip(predictions, test_rows) if r['content'] == c])) for c in contents]
    truths = [next(r['label'] for r in test_rows if r['content'] == c) for c in contents]
    majority = float(np.mean([(v > .5) == bool(y) for v, y in zip(votes, truths)]))
    return dict(content_mean_accuracy=float(np.mean(per_content)), content_majority_accuracy=majority, n_contents=len(contents),
                accuracy=observed, balanced_accuracy=balanced, p_value=float((np.sum(np.array(null) >= observed) + 1) / (permutations + 1)) if permutations else None,
                null_mean=float(np.mean(null)) if null else None, subject_mean=float(values.mean()), subject_ci95=[float(values.mean() - half), float(values.mean() + half)],
                subjects_above_0_5=int((values > .5).sum()), n_subjects=n, n_test=len(test_rows), positive_rate_test=float(y_test.mean()))


def congruency_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(ROOT / 'outputs/aligned_recovery_v3/full_seed322_positional/best_passed.pt'))
    parser.add_argument('--config', default=str(ROOT / 'configs/aligned_speech_local_v1.yaml'))
    parser.add_argument('--roles', nargs='*', default=['validation'])
    parser.add_argument('--permutations', type=int, default=200)
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default=str(ROOT / 'outputs/aligned_recovery_v3/congruency_probe'))
    args = parser.parse_args(argv)
    torch.set_num_threads(4); device = torch.device(args.device); rng = np.random.default_rng(args.seed)
    cfg = legacy.config(args.config)
    payload = recovery.load_checkpoint(Path(args.checkpoint))
    train = legacy.dataset_for(cfg, 'train')
    model = RecoveryEEGModel(legacy.decoder_from(payload), **payload['signature']['spec']).to(device)
    model.load_state_dict(payload['model']); model.eval()
    subjects = {s: i for i, s in enumerate(payload['signature']['subjects'])} if payload['signature']['spec'].get('subjects') else None
    features = {}
    for role in ['train', *args.roles]:
        dataset = train if role == 'train' else legacy.dataset_for(cfg, role)
        onsets = critical_word_onsets(dataset.frame); labels = labels_of(dataset.frame)
        features[role] = {window: collect(model, dataset, subjects, device, onsets, labels, pre_word=(window == 'pre_word')) for window in ('post_word', 'pre_word')}
        print(json.dumps(dict(role=role, trials={w: len(v) for w, v in features[role].items()})), flush=True)
    report = dict(contract='congruency_probe_v1', checkpoint=str(args.checkpoint), window_s=WINDOW_S, bins=BINS, permutations=args.permutations, results={})
    for role in args.roles:
        report['results'][role] = {}
        for window in ('post_word', 'pre_word'):
            report['results'][role][window] = {}
            for key, name in (('erp', 'eeg_erp_bins'), ('tangent', 'eeg_riemannian'), ('tokens', 'encoder_tokens'), ('audio', 'audio_only_mel')):
                result = evaluate(features['train'][window], features[role][window], key, args.permutations, rng)
                report['results'][role][window][name] = result
                print(f"{role:10s} {window:9s} {name:16s} acc {result['accuracy']:.3f} bal {result['balanced_accuracy']:.3f} p={result['p_value']:.3f} (sentence-null {result['null_mean']:.3f}) | subjects {result['subject_mean']:.3f} [{result['subject_ci95'][0]:.3f}, {result['subject_ci95'][1]:.3f}] {result['subjects_above_0_5']}/{result['n_subjects']}>0.5 | per-sentence {result['content_mean_accuracy']:.3f}, vote {result['content_majority_accuracy']:.3f} of {result['n_contents']}", flush=True)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / 'congruency_probe.json').write_text(json.dumps(report, indent=2) + '\n')


# --- 2. speak: fixed-voice SpeechT5 synthesis (the output end) -----------------

DEFAULT_MODEL = ROOT / 'models/aligned_local_base/speecht5_tts'


DEFAULT_VOCODER = ROOT / 'outputs/aligned_speech_local_v1/hifigan/best'


AUDIO_RATE = 16000


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


def speak_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', nargs='*', help='sentences to synthesise')
    parser.add_argument('--from-transcripts', type=int, default=0, help='synthesise this many validation reference transcripts instead')
    parser.add_argument('--output', default=str(ROOT / 'outputs/aligned_recovery_v3/synthesis_check'))
    parser.add_argument('--model', default=str(DEFAULT_MODEL)); parser.add_argument('--vocoder', default=str(DEFAULT_VOCODER))
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
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
        sf.write(path, wave, AUDIO_RATE, subtype='FLOAT')
        entry = dict(item=number, text=text, seconds=round(len(wave) / AUDIO_RATE, 2), file=path.name)
        if text in reference and reference[text].exists():
            original, _ = sf.read(reference[text], dtype='float32')
            sf.write(output / f'original{number:02d}.wav', original, AUDIO_RATE, subtype='FLOAT')
            entry.update(original=f'original{number:02d}.wav', original_seconds=round(len(original) / AUDIO_RATE, 2))
        index.append(entry); print(json.dumps(entry), flush=True)
    (output / 'index.json').write_text(json.dumps(index, indent=2) + '\n')


# --- 3. sentence: single-trial closed-set content decisions --------------------



SENTENCE_CONTRACT = 'content_pipeline_v1'


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


def ending_features(dataset, device, batch_size: int, conditions=CONDITIONS) -> dict:
    """ERP bin amplitudes in the 0.9 s after the critical word, per EEG condition.

    The critical word's onset comes from the trial's carrier context, which the
    cued task gives away by construction; nothing about the *ending* is used.
    """
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
    from karaone import ShrinkageLDA, Standardizer
    train = legacy.dataset_for(cfg, 'train'); evaluation = legacy.dataset_for(cfg, args.role)
    print(f'extracting ERP features: train {len(train)} trials, {args.role} {len(evaluation)} trials', flush=True)
    fitted = ending_features(train, device, args.batch_size, conditions=('correct',))
    held = ending_features(evaluation, device, args.batch_size)
    scaler = Standardizer().fit(fitted['correct'])
    model = ShrinkageLDA(.2).fit(scaler(fitted['correct']), fitted['labels'])
    report = dict(contract=SENTENCE_CONTRACT, task='ending2', cued=True, role=args.role, chance_accuracy=.5,
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


def sentence_main(argv=None):
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
    args = parser.parse_args(argv)
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
    report = dict(contract=SENTENCE_CONTRACT, task='sentence41', cued=False, role=args.role,
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


# --- 4. pooled: pooled sentence identification ---------------------------------



POOLED_CONTRACT = 'group_content_v1'




CHANNELS = ('embedding', 'envelope', 'mel', 'combined', 'rank_fused')


def load_recovery(checkpoint: Path, cfg, role: str, device):
    payload = recovery.load_checkpoint(checkpoint)
    train = legacy.dataset_for(cfg, 'train'); data = legacy.dataset_for(cfg, role)
    model = RecoveryEEGModel(legacy.decoder_from(payload), **payload['signature']['spec']).to(device)
    model.load_state_dict(payload['model']); model.eval()
    subjects = ({s: i for i, s in enumerate(payload['signature']['subjects'])}
                if payload['signature']['spec'].get('subjects') else None)
    return model, train, data, subjects


def load_envelope(checkpoint: Path, channels: int, device):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    saved = payload['args']
    model = envelope.EnvelopeDecoder(channels=channels, width=saved['width'], subjects=len(payload['subjects']),
                                     bands=len(envelope.BANDS) if saved['band_split'] else 1,
                                     per_subject_spatial=saved['per_subject_spatial']).to(device)
    model.load_state_dict(payload['model']); model.eval()
    return model, {s: i for i, s in enumerate(payload['subjects'])}, payload


def subject_projected(model, eeg, subject):
    """Everything in the envelope decoder up to (and including) the spatial map.

    The per-participant part is linear, so applying it per trial and averaging
    afterwards is the same input averaging §4 used for the recovery encoder —
    the nonlinear trunk still sees one averaged, higher-SNR input.
    """
    x = envelope.band_split(eeg) if model.bands > 1 else eeg
    if model.per_subject_spatial:
        return torch.bmm(model.spatial_weight[subject], x)
    if subject is not None and model.subjects:
        x = torch.bmm(torch.eye(x.shape[1], device=x.device, dtype=x.dtype) + model.subject_delta[subject], x)
    return model.spatial(x)


def envelope_from_features(model, x, prior):
    x = F.gelu(model.temporal(x))
    if model.positional:
        x = x + model.position
    x = model.output_norm(model.blocks(x).transpose(1, 2)).transpose(1, 2)
    return model.head(x).squeeze(1) + model.prior_gain * prior


def recovery_mixed(model, batch, subjects, device):
    eeg = batch['eeg'] * batch['channel_mask'][:, :, None]
    if subjects is None or not model.subjects:
        return eeg
    index = torch.tensor([subjects[s] for s in batch['subject']], device=device)
    return model.mix_subject(eeg, index)


def recovery_embedding(model, eeg, fixed, device, mel_window=None):
    """Aligned-sequence embedding over the fixed window, and the predicted mel on it.

    The envelope channel uses the mel's mean over frequency bins; everything the
    spectrum says beyond that level has never been used as evidence, and it
    costs nothing to read off the same forward pass.
    """
    n = len(eeg)
    xyz = torch.zeros(n, model.channels, 3, device=device)
    mask = torch.ones(n, model.channels, dtype=torch.bool, device=device)
    times = torch.ones(n, eeg.shape[-1], dtype=torch.bool, device=device)
    state = model(eeg, xyz, mask, times, None)
    embedding = F.normalize(model.decoder.normalizer(state.aligned_sequence)[:, fixed], dim=-1)
    if mel_window is None:
        return embedding, None
    return embedding, state.native_mel[:, :, mel_window].flatten(1)


def zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(axis=1, keepdims=True)) / (values.std(axis=1, keepdims=True) + 1e-9)


def rank_fuse(*channels: np.ndarray) -> np.ndarray:
    """Borda fusion: each channel votes with its ranking, not with its scale.

    The z-sum still lets a channel with a long tail of near-ties dominate the
    sum; averaging ranks cannot, which matters here because the two channels
    have different score distributions (the envelope channel wins R1 while the
    embedding channel wins top-5).
    """
    total = np.zeros_like(channels[0])
    for values in channels:
        order = np.argsort(-values, axis=1)
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(values.shape[1])[None].repeat(len(values), 0), axis=1)
        total = total + rank
    return -total


def background_corrected(scores: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Remove each candidate's attractiveness, measured on TRAINING EEG.

    `hubness_corrected` estimates the same quantity from the batch of queries
    being decoded, which is transductive -- and on the test partition it lifted
    the wrong-trial control to MRR 0.151 against a chance of 0.105, i.e. it was
    reading the balanced batch rather than the brain.  Estimating each
    candidate's mean score against a fixed set of *training* trials uses no
    information from the queries at all, so it is available to a system
    decoding a single presentation and cannot inflate a control through batch
    structure.
    """
    return (scores - background.mean(axis=0, keepdims=True)) / (background.std(axis=0, keepdims=True) + 1e-9)


def hubness_corrected(scores: np.ndarray) -> np.ndarray:
    """Remove each candidate's own popularity across the batch of queries.

    A handful of candidates win many queries at once (three of the ten most
    confident validation decisions named the same sentence), which is the
    standard hubness pathology of high-dimensional retrieval and caps R1 on its
    own.  Centring each candidate's column across queries removes it.  This is
    **transductive**: it uses the batch of queries, not any label, so it is only
    available to a system that decodes a batch rather than one presentation.
    """
    return (scores - scores.mean(axis=0, keepdims=True)) / (scores.std(axis=0, keepdims=True) + 1e-9)


def assignment_report(scores: np.ndarray, truth: np.ndarray, rng, draws: int = 2000) -> dict | None:
    """Decode the batch as a one-to-one assignment instead of independent argmax.

    At k = all the groups *are* the candidate sentences, one each, so the
    one-to-one constraint is true by construction.  It is a stronger assumption
    than identifying a sentence on its own -- the system is told the batch is a
    permutation of the candidate set -- and is reported separately for that
    reason, never as the headline.

    Two things make this rule easy to over-read, and both are handled here:

    * a **degenerate** score matrix (every query scoring every candidate
      identically, which is exactly what the zero-EEG control produces) still
      yields a full permutation, and the solver's tie-breaking can hand it any
      accuracy at all -- 1.000 on an 8-sentence smoke test.  There is no
      decision in that permutation, so it is reported as ``None``;
    * accuracy under a permutation decoder is not compared against 1/n by
      eye but against its own label-permutation null.
    """
    # Degeneracy is relative: the zero-EEG control's rows differ only by float32
    # reduction noise (across-query spread ~1e-7 against a within-query spread
    # of ~1), while a real condition's ratio is of order 1.
    if float(scores.std(axis=0).max()) < 1e-4 * float(scores.std(axis=1).mean() + 1e-12):
        return None
    from scipy.optimize import linear_sum_assignment
    rows, columns = linear_sum_assignment(-scores)
    observed = float((columns == truth[rows]).mean())
    null = np.array([float((columns == rng.permutation(truth)[rows]).mean()) for _ in range(draws)])
    return dict(accuracy=observed, null_mean=float(null.mean()),
                p_value=float(((null >= observed).sum() + 1) / (draws + 1)))


def metrics_for(scores: np.ndarray, truth: np.ndarray) -> dict:
    order = np.argsort(-scores, axis=1)
    rank = np.array([int(np.flatnonzero(order[i] == truth[i])[0]) + 1 for i in range(len(truth))])
    decision = decisions_from_scores(scores, truth)
    return dict(decisions=len(truth), accuracy=float((rank == 1).mean()), top5=float((rank <= 5).mean()),
                mrr=float((1 / rank).mean()),
                risk_coverage=risk_coverage(decision['confidence'], decision['correct']))


def permutation_p(scores: np.ndarray, truth: np.ndarray, rng, draws: int = 2000) -> dict:
    """Null for the 41-way accuracy: the same scores, the sentence labels shuffled."""
    choice = np.argmax(scores, axis=1)
    observed = float((choice == truth).mean())
    null = np.array([float((choice == rng.permutation(truth)).mean()) for _ in range(draws)])
    return dict(accuracy=observed, null_mean=float(null.mean()),
                p_value=float(((null >= observed).sum() + 1) / (draws + 1)))


def bootstrap_accuracy(correct: np.ndarray, rng, draws: int = 2000) -> list[float]:
    n = len(correct)
    values = np.array([correct[rng.integers(0, n, n)].mean() for _ in range(draws)])
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def pooled_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(ROOT / 'outputs/aligned_recovery_v3/full_seed322_positional/best_passed.pt'))
    parser.add_argument('--envelope-checkpoint', default=str(ROOT / 'outputs/envelope_decoder/base/best.pt'))
    parser.add_argument('--config', default=str(ROOT / 'configs/aligned_speech_local_v1.yaml'))
    parser.add_argument('--role', choices=['validation', 'test'], default='validation')
    parser.add_argument('--ks', nargs='*', type=int, default=[1, 2, 4, 8, 0], help='presentations pooled; 0 = all')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=31)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--speak-channel', default='combined', choices=list(CHANNELS))
    parser.add_argument('--coverage', type=float, default=.25)
    parser.add_argument('--speak', type=int, default=0, help='synthesise this many accepted decisions (0 = none)')
    parser.add_argument('--max-sentences', type=int, default=0, help='smoke-test cap on candidate sentences')
    parser.add_argument('--background', type=int, default=0,
                        help='training trials used to estimate each candidate\'s attractiveness (0 = off)')
    parser.add_argument('--output', default=str(ROOT / 'outputs/group_content'))
    args = parser.parse_args(argv)
    legacy.seed_all(args.seed); torch.set_num_threads(4)
    device = legacy.device(args.device); cfg = legacy.config(args.config)
    rng = np.random.default_rng(args.seed)

    model, train, data, subjects = load_recovery(Path(args.checkpoint), cfg, args.role, device)
    candidates = candidate_table(data, transcripts())
    if args.max_sentences:
        candidates = candidates.iloc[:args.max_sentences].reset_index(drop=True)
    order = {c: i for i, c in enumerate(candidates.content_group)}
    fixed = (model.decoder.speech_times < FIXED_WINDOW_S)

    # ---- candidate prototypes: speech-teacher sequences over the fixed window
    with h5py.File(data.cache, 'r') as h5:
        prototypes = torch.stack([F.normalize(model.decoder.normalizer(
            torch.from_numpy(h5['targets'][key]['teacher'][:]))[fixed], dim=-1) for key in candidates.audio_key])
    prototypes = prototypes.to(device)

    # ---- mel channel: candidate spectra and the train-fold median template on the window
    from aligned_recovery_eval import train_templates
    mel_window = (model.decoder.mel_times < FIXED_WINDOW_S)
    _, speech_median, _, _ = train_templates(train)
    mel_prior = speech_median[:, mel_window].flatten().to(device)
    with h5py.File(data.cache, 'r') as h5:
        candidate_mels = torch.stack([torch.from_numpy(h5['targets'][key]['mel'][:, mel_window])
                                      for key in candidates.audio_key]).flatten(1).to(device)
    mel_mask = torch.ones(1, candidate_mels.shape[-1], dtype=torch.bool, device=device)

    # ---- envelope channel: prior from train (as in training) and candidate envelopes
    count = (EEG_SAMPLES + 2 * 7 - 15) // envelope.TOKEN_STRIDE + 1
    token_times = -.25 + np.arange(count) * envelope.TOKEN_STRIDE / envelope.RATE
    envelope_model = envelope_subjects = None
    envelope_path = Path(args.envelope_checkpoint)
    if envelope_path.exists():
        envelope_model, envelope_subjects, envelope_payload = load_envelope(envelope_path, data[0]['eeg'].shape[0], device)
        train_targets = envelope.envelope_targets(train, token_times)
        stack = np.stack([v for v, _ in train_targets.values()]); masks = np.stack([m for _, m in train_targets.values()])
        prior = torch.from_numpy(np.where(masks.sum(0) > 0, np.nansum(stack * masks, 0) / np.maximum(masks.sum(0), 1),
                                          stack.mean(0)).astype(np.float32)).to(device)
        role_targets = envelope.envelope_targets(data, token_times)
        window = torch.from_numpy((token_times >= 0) & (token_times < FIXED_WINDOW_S)).to(device)
        candidate_envelopes = torch.stack([torch.from_numpy(role_targets[k][0]) for k in candidates.audio_key]).to(device)
    else:
        print(json.dumps(dict(warning='no envelope checkpoint; embedding channel only', path=str(envelope_path))), flush=True)

    # ---- one pass over the role: participant-mixed inputs for both models
    mixed, features, truth_of_trial, subject_of_trial = [], [], [], []
    for offset in range(0, len(data), args.batch_size):
        ids = list(range(offset, min(len(data), offset + args.batch_size)))
        batch = legacy.move(torch.utils.data.default_collate([data[i] for i in ids]), device)
        keep = [i for i, c in enumerate(batch['content']) if str(c) in order]
        if not keep:
            continue
        eeg = recovery_mixed(model, batch, subjects, device)
        mixed.extend(eeg[i].cpu() for i in keep)
        truth_of_trial.extend(order[str(batch['content'][i])] for i in keep)
        subject_of_trial.extend(batch['subject'][i] for i in keep)
        if envelope_model is not None:
            raw = batch['eeg'] * batch['channel_mask'][:, :, None]
            index = torch.tensor([envelope_subjects[s] for s in batch['subject']], device=device)
            projected = subject_projected(envelope_model, raw, index)
            features.extend(projected[i].cpu() for i in keep)
    truth_of_trial = np.array(truth_of_trial)
    print(json.dumps(dict(role=args.role, trials=len(mixed), sentences=len(candidates),
                          chance=1 / len(candidates))), flush=True)

    background_mixed, background_features = [], []
    if args.background:
        picked = rng.choice(len(train), size=min(args.background, len(train)), replace=False)
        for offset in range(0, len(picked), args.batch_size):
            ids = [int(i) for i in picked[offset:offset + args.batch_size]]
            batch = legacy.move(torch.utils.data.default_collate([train[i] for i in ids]), device)
            background_mixed.extend(recovery_mixed(model, batch, subjects, device)[i].cpu() for i in range(len(ids)))
            if envelope_model is not None:
                raw = batch['eeg'] * batch['channel_mask'][:, :, None]
                index = torch.tensor([envelope_subjects[s] for s in batch['subject']], device=device)
                projected = subject_projected(envelope_model, raw, index)
                background_features.extend(projected[i].cpu() for i in range(len(ids)))
        print(json.dumps(dict(background_trials=len(background_mixed))), flush=True)

    @torch.no_grad()
    def score_groups(picks: list[np.ndarray], condition: str, pool=None, feature_pool=None) -> dict:
        """Pool each group's trials at the input, decode once, score every candidate."""
        pool = mixed if pool is None else pool
        feature_pool = features if feature_pool is None else feature_pool
        embeddings, envelopes, mels = [], [], []
        for start in range(0, len(picks), args.batch_size):
            chunk = picks[start:start + args.batch_size]
            eeg = torch.stack([torch.stack([pool[j] for j in pick]).mean(0) for pick in chunk]).to(device)
            if condition == 'zero':
                eeg = torch.zeros_like(eeg)
            embedding, mel = recovery_embedding(model, eeg, fixed, device, mel_window)
            embeddings.append(torch.einsum('nfd,cfd->nc', embedding, prototypes).cpu() / int(fixed.sum()))
            c = len(candidates)
            mels.append(torch.stack([
                envelope.partial_correlation(mel[i][None].expand(c, -1), candidate_mels,
                                             mel_prior[None].expand(c, -1), mel_mask.expand(c, -1)).cpu()
                for i in range(len(chunk))]))
            if envelope_model is not None:
                x = torch.stack([torch.stack([feature_pool[j] for j in pick]).mean(0) for pick in chunk]).to(device)
                if condition == 'zero':
                    x = torch.zeros_like(x)   # the per-participant map is linear, so zero EEG gives zero features
                prior_batch = prior[None].expand(len(chunk), -1)
                prediction = envelope_from_features(envelope_model, x, prior_batch)
                rows = []
                for i in range(len(chunk)):
                    c = len(candidates)
                    rows.append(envelope.partial_correlation(prediction[i][None].expand(c, -1), candidate_envelopes,
                                                             prior[None].expand(c, -1), window[None].expand(c, -1)).cpu())
                envelopes.append(torch.stack(rows))
        out = dict(embedding=torch.cat(embeddings).numpy(), mel=torch.cat(mels).numpy())
        if envelopes:
            out['envelope'] = torch.cat(envelopes).numpy()
            out['combined'] = zscore(out['embedding']) + zscore(out['envelope'])
            out['rank_fused'] = rank_fuse(out['embedding'], out['envelope'])
        return out

    background_scores = None
    if background_mixed:
        background_scores = score_groups([np.array([i]) for i in range(len(background_mixed))], 'correct',
                                         pool=background_mixed, feature_pool=background_features)

    by_sentence = {c: np.flatnonzero(truth_of_trial == c) for c in range(len(candidates))}
    results = {}
    saved: dict[str, np.ndarray] = {}
    speak_rows = None
    for k in args.ks:
        label = 'all' if k == 0 else str(k)
        results[label] = {}
        for condition in CONDITIONS:
            picks, targets = [], []
            if k == 1:
                # Every trial decides once: 670 decisions instead of 41 x repeats,
                # which is the difference between a marginal and a usable p-value
                # on the row that matters most.
                for i in range(len(mixed)):
                    c = int(truth_of_trial[i])
                    if condition == 'wrong_trial':
                        others = [o for o in by_sentence if o != c]
                        picks.append(rng.choice(by_sentence[int(rng.choice(others))], size=1, replace=False))
                    else:
                        picks.append(np.array([i]))
                    targets.append(c)
            else:
                for c, index in by_sentence.items():
                    if condition == 'wrong_trial':
                        others = [o for o in by_sentence if o != c]
                        source = by_sentence[int(rng.choice(others))]
                    else:
                        source = index
                    size = len(source) if k == 0 or k >= len(source) else k
                    for _ in range(1 if (k == 0 or k >= len(source)) else args.repeats):
                        picks.append(rng.choice(source, size=size, replace=False)); targets.append(c)
            targets = np.array(targets)
            scored = score_groups(picks, condition)
            row = {}
            for name, values in list(scored.items()):
                row[name] = metrics_for(values, targets)
                corrected = hubness_corrected(values)
                row[name]['hubness_corrected'] = metrics_for(corrected, targets)
                if background_scores is not None and name in background_scores:
                    row[name]['background_corrected'] = metrics_for(
                        background_corrected(values, background_scores[name]), targets)
                if k == 0 and len(targets) == len(candidates):
                    row[name]['assignment'] = assignment_report(values, targets, np.random.default_rng(args.seed))
                    row[name]['hubness_corrected']['assignment'] = assignment_report(
                        corrected, targets, np.random.default_rng(args.seed))
                if k in (0, 1):
                    row[name]['permutation'] = permutation_p(values, targets, np.random.default_rng(args.seed))
                    correct = (np.argmax(values, axis=1) == targets)
                    row[name]['accuracy_ci'] = bootstrap_accuracy(correct, np.random.default_rng(args.seed))
            results[label][condition] = row
            if k in (0, 1):
                saved.setdefault(f'{label}_{condition}_truth', targets)
                for name, values in scored.items():
                    saved[f'{label}_{condition}_{name}'] = values
            if k == 0 and condition == 'correct':
                values = scored.get(args.speak_channel, scored['embedding'])
                decision = decisions_from_scores(values, targets)
                speak_rows = pd.DataFrame(dict(
                    sentence=[candidates.content_group.iloc[c] for c in targets],
                    decoded=[candidates.text.iloc[c] for c in decision['choice']],
                    truth=[candidates.text.iloc[c] for c in targets],
                    confidence=decision['confidence'], correct=decision['correct'],
                    presentations=[len(p) for p in picks]))
        for condition in CONDITIONS:
            for name in [n for n in CHANNELS if n in results[label][condition]]:
                r = results[label][condition][name]
                h = r['hubness_corrected']
                print(f"k={label:>3s} {condition:11s} {name:10s} R1 {r['accuracy']:.3f} top5 {r['top5']:.3f} "
                      f"MRR {r['mrr']:.3f} | coverage " +
                      ' '.join(f"{d['coverage']:.2f}:{d['accuracy']:.3f}" for d in r['risk_coverage']) +
                      f" || hub R1 {h['accuracy']:.3f} MRR {h['mrr']:.3f}" +
                      (f" || bg R1 {r['background_corrected']['accuracy']:.3f} "
                       f"top5 {r['background_corrected']['top5']:.3f} MRR {r['background_corrected']['mrr']:.3f}"
                       if 'background_corrected' in r else '') +
                      (' assign ' + '/'.join('degenerate' if a is None else f"{a['accuracy']:.3f}(p={a['p_value']:.3f})"
                                              for a in (r['assignment'], h['assignment']))
                       if 'assignment' in r else '') +
                      (f" | p={r['permutation']['p_value']:.4f} CI {r['accuracy_ci'][0]:.3f}-{r['accuracy_ci'][1]:.3f}"
                       if 'permutation' in r else ''), flush=True)

    output = Path(args.output) / args.role; output.mkdir(parents=True, exist_ok=True)
    summary = dict(contract=POOLED_CONTRACT, role=args.role, checkpoint=str(args.checkpoint),
                   envelope_checkpoint=str(envelope_path) if envelope_model is not None else None,
                   pooling='input_averaging', sentences=len(candidates), trials=len(mixed),
                   chance_accuracy=1 / len(candidates), repeats=args.repeats, results=results)
    (output / 'report.json').write_text(json.dumps(summary, indent=2) + '\n')
    if speak_rows is not None:
        speak_rows.to_csv(output / 'decisions.csv', index=False)
    if saved:
        np.savez_compressed(output / 'scores.npz', **saved)
    if args.speak and speak_rows is not None:
        import soundfile as sf
        accepted = speak_rows.sort_values('confidence', ascending=False)
        accepted = accepted.iloc[:max(1, int(round(args.coverage * len(accepted))))].iloc[:args.speak]
        synthesizer = FixedVoiceSynthesizer(device=args.device)
        spoken = output / 'spoken'; spoken.mkdir(parents=True, exist_ok=True)
        index = []
        for n, (_, row) in enumerate(accepted.iterrows(), 1):
            wave = synthesizer.waveform(row.decoded)
            name = f'decision{n:02d}.wav'
            sf.write(spoken / name, wave, 16000)
            index.append(dict(file=name, decoded=row.decoded, truth=row.truth, correct=bool(row.correct),
                              confidence=float(row.confidence), presentations=int(row.presentations)))
        (spoken / 'index.json').write_text(json.dumps(dict(
            contract=POOLED_CONTRACT, channel=args.speak_channel, coverage=args.coverage, items=index,
            note='audio clarity is the synthesiser, not the decoding; correctness is the `correct` field',
        ), indent=2) + '\n')
        print(json.dumps(dict(spoken=len(index), correct=int(sum(i['correct'] for i in index)))))
    print(json.dumps(dict(report=str(output / 'report.json'))))


def main():
    commands = {'congruency': congruency_main, 'sentence': sentence_main, 'pooled': pooled_main, 'speak': speak_main}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit('usage: python app/content_analysis.py congruency|sentence|pooled|speak [options]')
    commands[sys.argv[1]](sys.argv[2:])


if __name__ == '__main__':
    main()
