#!/usr/bin/env python3
"""Prepare a multi-task DS004940 aligned-speech manifest (v2 data route).

The legacy ``aligned_speech.py prepare`` audits and materializes only the
``N400Active`` task and rejects any recording with more than
``max_bad_fraction`` declared bad channels (five of the 22 participants, all
with 20-23 bad electrodes of 128).  This runner reuses the legacy adapters
unchanged but

* audits and materializes every task listed under ``tasks`` in the aligned
  config (``N400Active`` and ``N400Passive`` present the same 402 sentences);
* takes the bad-channel limit from the (new) data config, so the five
  excluded participants can be re-admitted with interpolation and masking;
* pins content roles to an existing assignment (``role_source``) so the
  validation/test sentences stay identical to v1 and the train-fold-adapted
  HuBERT/HiFi-GAN from v1 remain leak-free for v2.

No legacy module is modified, so legacy runtime hashes are untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
sys.path.insert(0, str(ROOT / 'app/src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import pandas as pd

import aligned_speech as legacy
from eeg2speech.aligned import CONTRACT, atomic_json, sha256
from eeg2speech.aligned_data import content_split, fit_eeg_normalizer, local_path, truth, validate_split

DEFAULT_TASKS = ('N400Active',)


def tasks_of(cfg):
    tasks = tuple(cfg.get('tasks', DEFAULT_TASKS))
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError('tasks must be a nonempty list of distinct DS004940 task names')
    return tasks


def audit_sources(data_cfg, tasks):
    """Audit every selected task's events/EEG/audio; mirrors the legacy Active-only audit."""
    from prepare_training_data import (_ds004_trial_rows, inventory_sha256, output_root, source_lock_entry,
                                       stable_json, sha256_bytes, write_frame)
    root = output_root(data_cfg); root.mkdir(parents=True, exist_ok=True)
    spec = data_cfg['sources']['ds004940']
    source = ROOT / spec['data_root']
    description = source / 'dataset_description.json'
    if sha256(description) != spec['dataset_description_sha256']:
        raise RuntimeError('DS004940 dataset description differs from the pinned version')
    if inventory_sha256(source / 'stimuli') != spec['stimulus_inventory_sha256']:
        raise RuntimeError('DS004940 stimulus inventory differs from the pinned version')
    scope = 'ds004940_' + '+'.join(tasks)
    qc = {'actual_subjects': {}, 'exclusions': Counter(), 'warnings': []}
    lock = {'schema_version': data_cfg['schema_version'], 'config_sha256': data_cfg['_config_sha256'],
            'files': [source_lock_entry(description, 'dataset_description')], 'official_aux': {}, 'scope': scope}
    all_rows = _ds004_trial_rows(data_cfg, lock, qc)
    if len(all_rows) != spec['expected_trials'] or qc['actual_subjects'].get('ds004940') != spec['expected_subjects']:
        raise RuntimeError('DS004940 event/subject count differs from pinned inventory')
    rows = [row for row in all_rows if row['task'] in tasks]
    if not rows:
        raise RuntimeError(f'no DS004940 events for tasks {tasks}')
    paths = sorted({row[k] for row in rows for k in ('source_eeg_path', 'source_channels_path', 'source_event_path', 'audio_path') if row.get(k)})
    digests = {}
    for i, relative in enumerate(paths):
        entry = source_lock_entry(ROOT / relative, 'source')
        lock['files'].append(entry); digests[relative] = entry['sha256']
        if i % 25 == 0:
            print(json.dumps({'audited_sources': i + 1, 'total': len(paths)}), flush=True)
    lock['files'].sort(key=lambda x: x['path'])
    lock['source_lock_sha256'] = sha256_bytes(stable_json(lock))
    stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    for row in rows:
        row.update(source_eeg_sha256=digests.get(row['source_eeg_path'], ''), source_lock_sha256=lock['source_lock_sha256'],
                   preprocess_config_sha256=data_cfg['_config_sha256'], code_commit='aligned-prepare-v2',
                   code_diff_hash=legacy.runtime_hash(), audit_timestamp_utc=stamp)
    write_frame(pd.DataFrame(rows), root / 'manifests/manifest_all', pd)
    atomic_json(root / 'source_lock.json', lock)
    qc.update(status='warning' if qc['warnings'] else 'pass', scope=scope,
              source_lock_sha256=lock['source_lock_sha256'], trial_counts={'ds004940': len(rows)},
              trials_per_task={task: sum(1 for r in rows if r['task'] == task) for task in tasks})
    qc['exclusions'] = dict(qc['exclusions'])
    atomic_json(root / 'qc/audit.json', qc)
    if qc['warnings']:
        raise RuntimeError('source audit warnings; inspect the isolated audit report')


def pin_roles(frame, assignment_path):
    """Copy content roles from an earlier assignment; unseen contents fall to train.

    Returns the pinned frame and the list of content groups that had no
    earlier role.  Roles are per content group, so no content can straddle
    roles and the earlier held-out audio stays held out.
    """
    earlier = pd.read_csv(assignment_path, keep_default_na=False)
    roles = earlier.drop_duplicates('content_group').set_index('content_group').role.to_dict()
    if earlier.groupby('content_group').role.nunique().max() != 1:
        raise ValueError('role source assigns one content to several roles')
    frame = frame.copy()
    unpinned = sorted(set(frame.content_group) - set(roles))
    frame['role'] = frame.content_group.map(lambda c: roles.get(c, 'train'))
    missing = sorted(set(roles) - set(frame.content_group))
    validate_split(frame)
    return frame, unpinned, missing


def prepare(cfg, *, materialize):
    from prepare_training_data import make_splits, load_config, output_root, build
    from prepare_m0_artifacts import _buildable_rows
    tasks = tasks_of(cfg)
    tag = str(cfg.get('artifact_set', 'aligned_v2'))
    data_cfg, _ = load_config(legacy.path_of(cfg, 'data_config'))
    data_root = output_root(data_cfg)
    base, manifest, _, normalizer = legacy.artifact_paths(cfg)
    base.mkdir(parents=True, exist_ok=True)
    if not (data_root / 'manifests/manifest_all.csv').exists():
        audit_sources(data_cfg, tasks)
    source_lock = json.loads((data_root / 'source_lock.json').read_text())
    if source_lock['config_sha256'] != data_cfg['_config_sha256']:
        raise ValueError('preprocessing config changed after audit; choose a new artifact directory')
    if source_lock.get('scope') != 'ds004940_' + '+'.join(tasks):
        raise ValueError('existing audit covers different tasks; choose a new artifact directory')
    if not (data_root / 'splits/assignment.json').exists():
        make_splits(data_cfg)
    original = pd.read_csv(data_root / 'manifests/manifest_all.csv', keep_default_na=False, low_memory=False)
    eligible = original[(original.dataset == 'ds004940') & original.task.isin(tasks) &
                        (original.pairing_level == 'verified_exact') & (original.build_status == 'included') & truth(original.qc_pass)]
    eligible = _buildable_rows(eligible, data_cfg)
    m0 = pd.read_csv(legacy.path_of(cfg, 'legacy_m0_manifest'), keep_default_na=False)
    m0_ids = set(m0.loc[(m0.build_status == 'included') & (m0.dataset == 'ds004940'), 'trial_id'])
    if len(m0_ids) != 50 or not m0_ids <= set(eligible.trial_id):
        raise RuntimeError('the existing 50-pair M0 is missing or fails current QC')
    selected = content_split(eligible, seed=cfg['split_seed'], reserved_train_ids=m0_ids)
    pinned = {}
    if cfg.get('role_source'):
        selected, unpinned, missing = pin_roles(selected, legacy.path_of(cfg, 'role_source'))
        if selected.loc[selected.trial_id.isin(m0_ids), 'role'].ne('train').any():
            raise RuntimeError('pinned roles move an M0 trial out of the train fold')
        pinned = dict(role_source=str(cfg['role_source']), unpinned_contents=unpinned, contents_absent_here=missing)
    selected['is_m0'] = selected.trial_id.isin(m0_ids)
    selected['audio_key'] = selected.audio_sha256
    split_path = data_root / f'splits/{tag}_fold-0.csv'
    cols = ['trial_id', 'role', 'fold', 'content_group', 'is_m0', 'audio_key']
    csv_text = selected[cols].sort_values('trial_id').to_csv(index=False)
    if split_path.exists() and split_path.read_text() != csv_text:
        raise RuntimeError('existing split differs; use a new version directory')
    split_path.write_text(csv_text)
    if materialize:
        for entry in source_lock['files']:
            if sha256(local_path(ROOT, entry['path'])) != entry['sha256']:
                raise ValueError(f"source changed since audit: {entry['path']}")
        transport = selected.loc[selected.role != 'excluded', cols].copy()
        transport['role'] = 'materialize'
        transport_path = data_root / f'splits/{tag}_materialize_fold-0.csv'
        transport_text = transport.sort_values('trial_id').to_csv(index=False)
        if transport_path.exists() and transport_path.read_text() != transport_text:
            raise RuntimeError('build selection changed; use a new version directory')
        transport_path.write_text(transport_text)
        build(data_cfg, 'ds004940', 'all', ','.join(tasks), None, None, None,
              'materialize', f'{tag}_materialize', 0, True, False, tag)
        built = pd.read_csv(data_root / f'manifests/manifest_{tag}.csv', keep_default_na=False)
        built = built[built.build_status == 'included']
        absent = set(selected.loc[selected.role != 'excluded', 'trial_id']) - set(built.trial_id)
        if absent:
            atomic_json(base / 'build_failures.json', {'missing_trials': sorted(absent)})
            raise RuntimeError('selected trials failed materialization; inspect build_failures.json')
        built = built.drop(columns=[c for c in cols[1:] if c in built], errors='ignore')
        selected = built.merge(selected[cols], on='trial_id', validate='one_to_one')
        import h5py
        for name, indices in selected.groupby('shard_path').groups.items():
            with h5py.File(local_path(ROOT, name), 'r') as h5:
                actual = str(h5.attrs['split_index_sha256'])
                if actual != sha256(transport_path):
                    raise ValueError('built shard pins an unexpected materialization selection')
            selected.loc[indices, 'split_index_sha256'] = actual
        selected = selected.sort_values('trial_id')
        text = selected.to_csv(index=False)
        if manifest.exists() and manifest.read_text() != text:
            raise RuntimeError('materialized manifest changed; preserve existing experiment')
        manifest.write_text(text)
        if not normalizer.exists():
            fit_eeg_normalizer(ROOT, manifest, normalizer)
    selected[cols].to_csv(base / 'assignment.csv', index=False)
    report = {'contract': CONTRACT, 'protocol': 'known_subject_unseen_content', 'tasks': list(tasks),
              'max_bad_fraction': float(data_cfg['harmonized']['interpolation']['max_bad_fraction']),
              'role_counts': selected.role.value_counts().to_dict(),
              'unique_contents': selected.groupby('role').content_group.nunique().to_dict(),
              'subjects': selected.groupby('role').subject.nunique().to_dict(),
              'trials_per_task': selected.groupby('task').size().to_dict(),
              'm0_pairs': int(selected.is_m0.sum()), 'materialized': bool(materialize),
              'historical_test_status': 'exploratory_previously_inspected_dataset', 'split_sha256': sha256(split_path),
              **pinned}
    atomic_json(base / 'split_report.json', report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/aligned_speech_local_v2.yaml'))
    parser.add_argument('--materialize', action='store_true')
    args = parser.parse_args()
    prepare(legacy.config(args.config), materialize=args.materialize)


if __name__ == '__main__':
    main()
