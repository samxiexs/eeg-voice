"""Subject-level, source-stratified uncertainty and complete matrix accounting."""
from __future__ import annotations
from collections import defaultdict
import json
import numpy as np
from ..metrics import holm
from ..utils import atomic_json
from .evaluation import aggregate_rows
from .protocol import main_matrix, artifact_root


def stratified_paired(source_deltas, repetitions=10000, seed=11):
    groups = {s:np.asarray(v,dtype=float) for s,v in source_deltas.items()}
    if not groups or any(len(v)==0 or not np.isfinite(v).all() for v in groups.values()):
        raise ValueError('Complete nonempty source-level paired observations required')
    rng = np.random.default_rng(seed)
    observed = float(np.mean([v.mean() for v in groups.values()]))
    bootstrap = np.mean([v[rng.integers(len(v), size=(repetitions,len(v)))].mean(1) for v in groups.values()],axis=0)
    lower,upper = np.quantile(bootstrap,[.025,.975])
    # The sign flip is per subject within source, then the sources receive equal weight.
    null = np.mean([(rng.choice([-1,1],size=(100000,len(v)))*v).mean(1) for v in groups.values()],axis=0)
    p = float((1+(np.abs(null)>=abs(observed)-1e-12).sum())/(len(null)+1))
    return {'mean':observed,'lower':float(lower),'upper':float(upper),'p_two_sided':p,
            'subjects':{s:len(v) for s,v in groups.items()},'unit':'subject within source','weighting':'equal sources'}


def paired_comparison(rows, a='S2', b='S3', control_a='correct',control_b='correct'):
    aggregates = aggregate_rows(rows)['subjects']
    left = {(r['dataset'],r['subject']):r['wer'] for r in aggregates if r['system']==a and r['control']==control_a}
    right = {(r['dataset'],r['subject']):r['wer'] for r in aggregates if r['system']==b and r['control']==control_b}
    if set(left)!=set(right): raise ValueError('Systems must have identical held-out subjects and window references')
    # Audit matching sample/seed/reference sets before aggregating totals.
    def keys(system,control):
        return {(r['dataset'],r['subject'],r['id'],r['training_seed'],r['generation_seed'],r['words'],r.get('reference'))
                for r in rows if r['system']==system and r['control']==control}
    if keys(a,control_a)!=keys(b,control_b): raise ValueError('Paired systems lack identical windows/seeds/references')
    groups = defaultdict(list)
    for key in sorted(left): groups[key[0]].append(left[key]-right[key])
    return {'comparison':f'{a}/{control_a} minus {b}/{control_b}',**stratified_paired(groups)}


def two_way_stratified(rows,a='S2',b='S3',repetitions=10000,seed=11):
    per_unit = defaultdict(lambda:[0,0])
    for r in rows:
        if r['system'] not in (a,b) or r['control']!='correct': continue
        key=(r['dataset'],r['subject'],r.get('content_group',r['audio']),r['system'],r['training_seed'],r['generation_seed'])
        per_unit[key][0]+=r['errors'];per_unit[key][1]+=r['words']
    averaged=defaultdict(list)
    for (s,p,c,m,_,_),(errors,words) in per_unit.items():
        if not words: raise ValueError('Empty WER content unit')
        averaged[(s,p,c,m)].append(errors/words)
    tables={}
    for source in sorted({k[0] for k in averaged}):
        people=sorted({k[1] for k in averaged if k[0]==source})
        contents=sorted({k[2] for k in averaged if k[0]==source})
        values=np.array([[np.mean(averaged.get((source,p,c,a),[np.nan]))-np.mean(averaged.get((source,p,c,b),[np.nan]))
                          for c in contents] for p in people])
        if not np.isfinite(values).all(): raise ValueError('Incomplete subject × canonical content pairs')
        tables[source]=(values,people,contents)
    if not tables: raise ValueError('No complete two-way evidence')
    rng=np.random.default_rng(seed);estimates=[]
    for _ in range(repetitions):
        means=[]
        for matrix,people,contents in tables.values():
            means.append(matrix[np.ix_(rng.integers(len(people),size=len(people)),rng.integers(len(contents),size=len(contents)))].mean())
        estimates.append(np.mean(means))
    low,high=np.quantile(estimates,[.025,.975])
    return {'mean':float(np.mean([v[0].mean() for v in tables.values()])), 'lower':float(low),'upper':float(high),
            'sources':{s:{'subjects':len(p),'content_groups':len(c)} for s,(_,p,c) in tables.items()},
            'limitation':'Four sources and limited canonical passages do not identify general laboratory-level effects'}


def report(cfg, protocol='P1'):
    root=artifact_root(cfg,protocol)
    rows=[]
    for path in sorted(root.glob('runs/*/*/*/evaluation/zero-shot/test/scores.jsonl')):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    if not rows: raise ValueError('No frozen test scores found')
    required=[r for r in main_matrix() if r['protocol']==protocol]
    present={(r['system'],r['training_seed']) for r in rows}
    missing=[r for r in required if not (root/'runs'/r['fold']/str(r['seed'])/r['system']/'evaluation/zero-shot/test/summary.json').exists()]
    value={'protocol':protocol,'aggregate':aggregate_rows(rows),'completed':not missing,'missing_runs':missing}
    # Incomplete runs are reported, never silently presented as the complete main matrix.
    if not missing:
        value['primary']=paired_comparison(rows)
        value['two_way']=two_way_stratified(rows)
        comparisons=[('S3','S1','correct','correct'),('S3','S0','correct','correct'),
                     ('S3','S3','wrong','correct'),('S3','S3','null','correct')]
        secondary=[paired_comparison(rows,*args) for args in comparisons]
        for item,p in zip(secondary,holm([r['p_two_sided'] for r in secondary])):item['p_holm']=p
        value['secondary']=secondary
    atomic_json(root/'report.json',value)
    return value
