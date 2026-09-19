#!/usr/bin/env python3
"""S2: pooled sentence identification — EEG picks *which* sentence, TTS says it.

S1 (`app/content_pipeline.py`) asked one trial to choose among 41 held-out
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
import content_pipeline
import envelope_decoder as envelope
from aligned_recovery_model import RecoveryEEGModel
from aligned_recovery_eval import FIXED_WINDOW_S
from eeg2speech.aligned import EEG_SAMPLES

legacy = recovery.legacy
CONTRACT = 'group_content_v1'
CONDITIONS = ('correct', 'zero', 'wrong_trial')
CHANNELS = ('embedding', 'envelope', 'mel', 'combined', 'rank_fused')


# ----------------------------------------------------------------------------- models


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


@torch.no_grad()
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


@torch.no_grad()
def envelope_from_features(model, x, prior):
    x = F.gelu(model.temporal(x))
    if model.positional:
        x = x + model.position
    x = model.output_norm(model.blocks(x).transpose(1, 2)).transpose(1, 2)
    return model.head(x).squeeze(1) + model.prior_gain * prior


@torch.no_grad()
def recovery_mixed(model, batch, subjects, device):
    eeg = batch['eeg'] * batch['channel_mask'][:, :, None]
    if subjects is None or not model.subjects:
        return eeg
    index = torch.tensor([subjects[s] for s in batch['subject']], device=device)
    return model.mix_subject(eeg, index)


@torch.no_grad()
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


# ----------------------------------------------------------------------------- scoring


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
    decision = content_pipeline.decisions_from_scores(scores, truth)
    return dict(decisions=len(truth), accuracy=float((rank == 1).mean()), top5=float((rank <= 5).mean()),
                mrr=float((1 / rank).mean()),
                risk_coverage=content_pipeline.risk_coverage(decision['confidence'], decision['correct']))


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


# ----------------------------------------------------------------------------- main


def main():
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
    args = parser.parse_args()
    legacy.seed_all(args.seed); torch.set_num_threads(4)
    device = legacy.device(args.device); cfg = legacy.config(args.config)
    rng = np.random.default_rng(args.seed)

    model, train, data, subjects = load_recovery(Path(args.checkpoint), cfg, args.role, device)
    candidates = content_pipeline.candidate_table(data, content_pipeline.transcripts())
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
                decision = content_pipeline.decisions_from_scores(values, targets)
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
    summary = dict(contract=CONTRACT, role=args.role, checkpoint=str(args.checkpoint),
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
        from synthesize_text import FixedVoiceSynthesizer
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
            contract=CONTRACT, channel=args.speak_channel, coverage=args.coverage, items=index,
            note='audio clarity is the synthesiser, not the decoding; correctness is the `correct` field',
        ), indent=2) + '\n')
        print(json.dumps(dict(spoken=len(index), correct=int(sum(i['correct'] for i in index)))))
    print(json.dumps(dict(report=str(output / 'report.json'))))


if __name__ == '__main__':
    main()
