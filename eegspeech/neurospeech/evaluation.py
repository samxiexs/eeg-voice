"""Source-balanced held-person evaluation with fixed pairing and free English ASR."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import default_collate
from ..metrics import word_errors, bootstrap_mean
from ..utils import stable_int, atomic_json, append_jsonl
from .protocol import row_source, subject_identity, assert_scope, require_lock, run_dir
from .inference import condition_from_output, native_decode, eeg_only


def wrong_time_mapping(rows, minimum=10.0):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        key = (row_source(row), str(row.get('subject', 'audio')), str(row.get('recording', row.get('audio'))))
        groups[key].append(i)
    mapping = {}
    for indices in groups.values():
        for i in indices:
            row = rows[i]
            start = float(row.get('canonical_start', row['start']))
            candidates = [j for j in indices if abs(float(rows[j].get('canonical_start', rows[j]['start'])) - start) >= minimum]
            if not candidates:
                raise ValueError(f'No valid same-person/stimulus wrong-time pairing ≥{minimum}s for {row.get("id", i)}')
            mapping[i] = min(candidates, key=lambda j: stable_int(f'wrong-v2.1:{row.get("id", i)}:{rows[j].get("id", j)}'))
    return mapping


def aggregate_rows(rows):
    # Sum word counts over windows per generation/training seed, then average seeds.
    summed = defaultdict(lambda: [0, 0])
    for row in rows:
        key = (row['dataset'], row['subject'], row['system'], row['control'], row['training_seed'], row['generation_seed'])
        summed[key][0] += row['errors']; summed[key][1] += row['words']
    by_training = defaultdict(list)
    for (source, person, system, control, training, generation), (errors, words) in summed.items():
        if not words: raise ValueError('Reference words missing for an entire evaluation subject/control')
        by_training[(source, person, system, control, training)].append(errors / words)
    by_subject = defaultdict(list)
    for (source, person, system, control, _), values in by_training.items():
        by_subject[(source, person, system, control)].append(float(np.mean(values)))
    subjects = [{'dataset': source, 'subject': person, 'system': system, 'control': control,
                 'wer': float(np.mean(values)), 'training_seeds': len(values)}
                for (source, person, system, control), values in sorted(by_subject.items())]
    by_source = defaultdict(list)
    for row in subjects: by_source[(row['dataset'], row['system'], row['control'])].append(row['wer'])
    sources = [{'dataset': s, 'system': m, 'control': c, 'wer': float(np.mean(v)), 'subjects': len(v)}
               for (s,m,c),v in sorted(by_source.items())]
    by_system = defaultdict(list)
    for row in sources: by_system[(row['system'], row['control'])].append(row['wer'])
    overall = [{'system':m, 'control':c, 'wer':float(np.mean(v)), 'sources':len(v)} for (m,c),v in sorted(by_system.items())]
    return {'subjects': subjects, 'sources': sources, 'overall': overall}


def condition_controls(correct, wrong=None, control='correct'):
    if control == 'correct': return correct
    if control == 'wrong':
        if wrong is None: raise ValueError('wrong-time condition missing')
        return wrong
    if control == 'null':
        return {**correct, 'content': torch.zeros_like(correct['content']),
                'rhythm': torch.zeros_like(correct['rhythm']),
                'reliability': torch.zeros_like(correct['reliability'])}
    if control == 'content-shuffle':
        if wrong is None: raise ValueError('Wrong content missing')
        return {**correct, 'content': wrong['content']}
    raise ValueError(control)


@torch.no_grad()
def score_model(cfg, dataset, flow, audio, asr, encoder=None, calibrator=None, system='S2',
                seed=11, controls=('correct',), generation_seeds=(0,), device='cpu', adapter=None,
                output_path=None):
    from .training import move_batch
    flow.eval()
    if encoder is not None: encoder.eval()
    mapping = wrong_time_mapping(dataset.rows) if any(c in controls for c in ('wrong','content-shuffle')) else None
    rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        batch = move_batch(default_collate([sample]), device)
        if encoder is None:
            correct = {'content': batch['teacher'], 'rhythm': batch['phi'],
                       'reliability': torch.ones_like(batch['phi']), 'times': batch['times']}
        else:
            output = encoder(eeg_only(batch), adapter=adapter)
            output['times'] = batch['times']
            correct = condition_from_output(output, calibrator, system)
        wrong = None
        if mapping is not None:
            alternate = move_batch(default_collate([dataset[mapping[index]]]), device)
            out = encoder(eeg_only(alternate), adapter=adapter)
            out['times'] = batch['times']
            wrong = condition_from_output(out, calibrator, system)
        for generation_seed in generation_seeds:
            # Identical noise per sample/control/system enables a paired comparison.
            noise_seed = stable_int(f'gen:{generation_seed}:{sample["id"]}')
            rng = torch.Generator(device=device).manual_seed(noise_seed)
            frames = int(getattr(flow, 'mel_frames', 376))
            noise = torch.randn(1, 100, frames, generator=rng, device=device)
            for control in controls:
                condition = condition_controls(correct, wrong, control)
                mel = flow.rollout(noise.clone(), condition, steps=32, method='heun')
                wave = native_decode(mel, dataset.stats, audio.vocos)[0].cpu().numpy()
                hypothesis = asr(wave)
                reference = asr.normalize(sample['text']).strip()
                if not reference: raise ValueError(f'No independently verified WER reference for {sample["id"]}')
                errors = word_errors(reference, hypothesis)
                row = {'id': sample['id'], 'dataset': row_source(dataset.rows[index]), 'subject': sample.get('subject','audio'),
                       'audio':sample['audio'], 'content_group': dataset.rows[index].get('content_group', sample['audio']),
                       'system':system,'control':control,'training_seed':seed,'generation_seed':generation_seed,
                       'reference':reference,'hypothesis':hypothesis,**errors,
                       'mel_mae': float((mel - batch['mel']).abs().mean())}
                rows.append(row)
                if output_path is not None: append_jsonl(output_path, row)
    summaries = aggregate_rows(rows)
    correct_rows = [r for r in summaries['overall'] if r['control'] == 'correct']
    return {**summaries, 'wer':correct_rows[0]['wer'] if correct_rows else None,
            'mel_mae':float(np.mean([r['mel_mae'] for r in rows])) if rows else None, 'rows':rows}


@torch.no_grad()
def copy_synthesis_gate(cfg, dataset, audio, asr, device='cpu'):
    rows = []; length_errors = []
    for i in range(len(dataset)):
        sample = dataset[i]
        wave = sample['wave'].to(device)[None]
        original = asr(wave[0].cpu().numpy())
        reconstructed = audio.vocos(audio.vocos.features(wave))[0]
        copy = asr(reconstructed.cpu().numpy())
        reference = asr.normalize(sample['text']).strip()
        if not reference: raise ValueError('Copy gate requires an independent, nonempty transcript')
        for system, hypothesis in [('original', original), ('copy', copy)]:
            rows.append({'id':sample['id'],'dataset':row_source(dataset.rows[i]),'subject':sample.get('subject','audio'),
                         'system':system,'control':'correct','training_seed':0,'generation_seed':0,
                         **word_errors(reference, hypothesis)})
        length_errors.append(abs(len(reconstructed)-len(wave[0])))
    summary = aggregate_rows(rows)
    values = {(r['dataset'],r['system']):r['wer'] for r in summary['sources']}
    sources = sorted({key[0] for key in values})
    per_source = {s:{'original_wer':values[(s,'original')], 'copy_wer':values[(s,'copy')],
                     'passed':values[(s,'copy')]-values[(s,'original')] <= .05} for s in sources}
    return {'sources':per_source,'passed':bool(sources) and all(v['passed'] for v in per_source.values()) and max(length_errors,default=99999)<=256,
            'max_length_error_samples':max(length_errors,default=0)}


@torch.no_grad()
def encoder_validation(cfg, encoder, dataset, device='cpu'):
    from .training import move_batch
    encoder.eval()
    mapping = wrong_time_mapping(dataset.rows)
    by_subject = defaultdict(list)
    for i in range(len(dataset)):
        batch = move_batch(default_collate([dataset[i]]), device)
        wrong = move_batch(default_collate([dataset[mapping[i]]]), device)
        out, alternate = encoder(eeg_only(batch)), encoder(eeg_only(wrong))
        good = float((out['teacher']-batch['teacher']).square().mean())
        bad = float((alternate['teacher']-batch['teacher']).square().mean())
        by_subject[(row_source(dataset.rows[i]), batch['subject'][0])].append((good,bad))
    subjects = [{'dataset':s,'subject':p,'mse':float(np.mean(v,axis=0)[0]),'wrong_mse':float(np.mean(v,axis=0)[1])}
                for (s,p),v in sorted(by_subject.items())]
    by_source = defaultdict(list)
    for row in subjects: by_source[row['dataset']].append(row['mse'])
    # Source-stratified paired bootstrap, not windows as independent observations.
    from .reporting import stratified_paired
    ci = stratified_paired({s:[r['wrong_mse']-r['mse'] for r in subjects if r['dataset']==s] for s in by_source})
    return {'mse':float(np.mean([np.mean(v) for v in by_source.values()])), 'subjects':subjects,
            'correct_vs_wrong':ci,'passed':ci['lower']>0}


def evaluate(cfg, protocol, fold, seed, system, split='test', device='cpu', mode='zero-shot'):
    from .data import WindowDataset,calibration_dataset
    from .inference import load_generator
    from ..perceived_audio import FrozenAudio, ASR
    from .adaptation import calibrate_adapter
    require_lock(cfg, protocol)
    audio = FrozenAudio(cfg, device, teacher=False); asr = ASR(cfg, device)
    generator = load_generator(cfg, protocol, fold, seed, system, device, audio.vocos)
    if mode == 'noise' and protocol != 'P1': raise ValueError('Etard noise evaluation is defined for P1 held-out subjects only')
    dataset = WindowDataset(cfg,protocol=protocol,fold=fold,split=split,stats=generator.stats)
    assert_scope(dataset.rows,protocol,split)
    folder = run_dir(cfg, protocol, fold, seed, system) / 'evaluation' / mode / split
    path = folder / 'scores.jsonl'
    if path.exists(): raise FileExistsError('Scores already exist; use a fresh immutable evaluation namespace')
    controls = ('correct','wrong','null','content-shuffle') if system in ('S2','S3') else ('correct','wrong','null')
    if mode == 'zero-shot':
        result = score_model(cfg,dataset,generator.flow,audio,asr,generator.encoder,generator.calibrator,system,seed,
                             controls,(0,1,2),device,output_path=path)
    elif mode == 'few-shot':
        result_rows = []
        for person in sorted({subject_identity(r) for r in dataset.rows}):
            indices = [r for r in dataset.rows if subject_identity(r)==person]
            target = WindowDataset(cfg,protocol=protocol,fold=fold,split=split,rows=indices,stats=generator.stats)
            paired = calibration_dataset(cfg,person,protocol,fold,generator.stats)
            adapter = calibrate_adapter(cfg,generator.encoder,paired,protocol,fold,seed,device,folder/'adapters'/f'{stable_int(person)}.pt')
            scored = score_model(cfg,target,generator.flow,audio,asr,generator.encoder,generator.calibrator,system,seed,
                                 controls,(0,1,2),device,adapter,output_path=path)
            result_rows.extend(scored['rows'])
        result = {**aggregate_rows(result_rows), 'rows':result_rows, 'reliability':'source-only-not-recalibrated'}
    elif mode == 'noise':
        # Etard babble conditions are never trained on. No English story part is shared with clean
        # (reader identity audit), so every noisy window is unseen content and has no clean pair.
        from .data import rows_for
        people = [s for s in cfg.fold_subjects(protocol,fold,split) if s.startswith('etard:')]
        result = {'clean_counterpart':'none released: all English story parts are different speech','conditions':{},'rows':[]}
        for condition in ('lb','mb','hb'):
            rows = rows_for(cfg,people,None,(condition,))
            if not rows: result['conditions'][condition] = 'missing'; continue
            noisy = WindowDataset(cfg,protocol=protocol,fold=fold,split=split,rows=rows,stats=generator.stats)
            scored = score_model(cfg,noisy,generator.flow,audio,asr,generator.encoder,generator.calibrator,system,seed,
                                 controls,(0,1,2),device,output_path=path)
            result['conditions'][condition] = {k:v for k,v in scored.items() if k!='rows'}
            result['rows'].extend({**r,'condition':condition} for r in scored['rows'])
    else: raise ValueError(mode)
    atomic_json(folder/'summary.json', {k:v for k,v in result.items() if k!='rows'})
    return result
