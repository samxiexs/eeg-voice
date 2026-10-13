"""Immutable experiment scopes, independent P1/P2 locks, and train-side OOF groups."""
from __future__ import annotations
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
from ..utils import atomic_json, fingerprint, read_json, stable_int

SYSTEMS = ('S0', 'S1', 'S2', 'S3')
SEEDS = (11, 22, 33)
PERCEIVED = ('ds004408', 'brennan2019', 'etard', 'kymata_english')


def section(cfg, key):
    try: return cfg[key]
    except KeyError: return {}


def artifact_root(cfg, protocol='P1'):
    if protocol not in ('P1', 'P2'): raise ValueError(protocol)
    return Path(cfg.dir('artifacts')) / protocol


def encoder_dir(cfg, protocol, fold, seed, system, replica=None):
    shared = 'S2' if system == 'S3' else system
    if shared not in SYSTEMS: raise ValueError(system)
    path = artifact_root(cfg, protocol) / 'encoders' / fold / str(seed) / shared
    return path if replica is None else path / 'oof' / str(replica)


def run_dir(cfg, protocol, fold, seed, system):
    if system not in SYSTEMS: raise ValueError(system)
    return artifact_root(cfg, protocol) / 'runs' / fold / str(seed) / system


def oracle_dir(cfg, protocol, seed):
    return artifact_root(cfg, protocol) / 'oracle' / str(seed)


def main_matrix():
    return [{'protocol': p, 'fold': f, 'seed': s, 'system': m}
            for p, folds in [('P1', [f'F{i}' for i in range(5)]), ('P2', ['external'])]
            for f in folds for s in SEEDS for m in SYSTEMS]


def row_source(row):
    return str(row.get('dataset', row.get('source', ''))).lower()


def content_identity(row):
    return str(row.get('content_group', row.get('content_id', row.get('audio', ''))))


def subject_identity(row):
    return f'{row_source(row)}:{row["subject"]}'


def assert_scope(rows, protocol, split='train'):
    if protocol not in ('P1', 'P2'): raise ValueError(protocol)
    for row in rows:
        source = row_source(row)
        if source == 'karaone': raise ValueError('KaraOne cannot enter Perceived training or evaluation')
        if protocol == 'P2' and split in ('train', 'validation') and source in ('brennan', 'brennan2019', 'nm000180'):
            raise ValueError('External Brennan EEG/audio cannot enter P2 training or model selection')
        if split == 'train' and source == 'etard' and row.get('condition', 'clean') != 'clean':
            raise ValueError('Etard noisy/repeated conditions cannot enter clean training')
        if row.get('raw_path') or row.get('staging_path'):
            raise ValueError('NeuroSpeech readers must use processed-only samples')
        if not content_identity(row): raise ValueError('Missing canonical content identity')


def scope_manifest(cfg, protocol, rows, split='train', excluded_subjects=(), excluded_contents=()):
    assert_scope(rows, protocol, split)
    return {'config_hash': cfg.hash, 'protocol': protocol, 'split': split,
            'subjects': sorted({subject_identity(r) for r in rows}),
            'contents': sorted({content_identity(r) for r in rows}),
            'sample_hash': fingerprint(sorted(str(r.get('id', r.get('key', ''))) for r in rows)),
            'excluded_subjects': sorted(excluded_subjects), 'excluded_contents': sorted(excluded_contents)}


def require_checkpoint_scope(saved, cfg, protocol, fold=None):
    info = saved.get('provenance', {})
    if info.get('config_hash') != cfg.hash or info.get('protocol') != protocol:
        raise ValueError('Checkpoint belongs to a different protocol/configuration')
    if fold is not None and info.get('fold') != fold:
        raise ValueError('Checkpoint belongs to a different subject fold')


def provenance(cfg, protocol, fold=None, scope=None):
    return {'config_hash': cfg.hash, 'protocol': protocol, 'fold': fold, 'scope': scope}


def freeze_protocol(cfg, protocol, decisions):
    """Atomic once-only lock. P2 decisions must identify source-side development."""
    if protocol == 'P2' and decisions.get('selection_scope') != 'source-only':
        raise ValueError('P2 lock requires source-only selection evidence')
    record = {'version': 'neurospeech_v2.1', 'config_hash': cfg.hash,
              'protocol': protocol, 'decisions': decisions}
    record['lock_hash'] = fingerprint(record)
    path = artifact_root(cfg, protocol) / 'protocol_lock.json'
    if path.exists():
        if read_json(path) != record: raise ValueError('Protocol is already frozen with different decisions')
    else: atomic_json(path, record)
    return record


def require_lock(cfg, protocol):
    path = artifact_root(cfg, protocol) / 'protocol_lock.json'
    lock = read_json(path)
    if lock['config_hash'] != cfg.hash or lock['protocol'] != protocol:
        raise ValueError('Protocol lock/configuration mismatch')
    return lock


def oof_groups(rows, folds=3, seed=20261013):
    """Double holdout inside formal train only, independently grouping persons/content.

    Every internal fit excludes the chosen persons AND chosen contents. Only their
    intersection is a calibration prediction; all other omitted rows are ignored.
    """
    if folds < 2: raise ValueError('OOF requires at least two folds')
    by_source = defaultdict(list)
    for i, row in enumerate(rows): by_source[row_source(row)].append(i)
    assignments = {}
    for source, indices in sorted(by_source.items()):
        people = sorted({subject_identity(rows[i]) for i in indices}, key=lambda v: stable_int(f'{seed}:subject:{v}'))
        contents = sorted({content_identity(rows[i]) for i in indices}, key=lambda v: stable_int(f'{seed}:content:{v}'))
        if len(people) < folds or len(contents) < folds:
            raise ValueError(f'{source} needs {folds} independent training subjects AND content groups for reliability OOF')
        assignments[source] = ({p: i % folds for i, p in enumerate(people)},
                               {c: i % folds for i, c in enumerate(contents)})
    result = []
    for fold in range(folds):
        train, held, excluded_subjects, excluded_contents = [], [], set(), set()
        for i, row in enumerate(rows):
            people, contents = assignments[row_source(row)]
            p, c = subject_identity(row), content_identity(row)
            person_held, content_held = people[p] == fold, contents[c] == fold
            if person_held: excluded_subjects.add(p)
            if content_held: excluded_contents.add(c)
            if not person_held and not content_held: train.append(i)
            elif person_held and content_held: held.append(i)
        if not train or not held: raise ValueError('Empty double-held OOF intersection')
        result.append({'fold': fold, 'train_indices': train, 'held_indices': held,
                       'excluded_subjects': sorted(excluded_subjects), 'excluded_contents': sorted(excluded_contents)})
    return result
