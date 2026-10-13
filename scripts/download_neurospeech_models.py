#!/usr/bin/env python3
"""Pin public model revisions, download one weight copy, verify all SHA256s.

REVE requires audited repository Python files for trust_remote_code. No external
code executes during acquisition; interface checks are a separate explicit step.
Existing legacy model pins are preserved and their files verified without redownload.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def acquire(root=ROOT):
    configuration=root/'configs/neurospeech_v2/models.json'
    specs=json.loads(configuration.read_text());manifest_path=root/'models/manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    session=requests.Session();session.headers['User-Agent']='NeuroSpeech-research-acquisition/2.1'
    for kind,spec in specs.items():
        prior=manifest.get(kind)
        if prior:
            if prior['repo_id']!=spec['repo_id']:raise ValueError(f'Model repo identity conflict: {kind}')
            if spec.get('revision') and prior['revision']!=spec['revision']:raise ValueError(f'Model pin conflict: {kind}')
            folder=root/'models'/prior['local_dir']
            for name,sha in prior['files'].items():
                path=folder/name
                if not path.is_file() or digest(path)!=sha:raise ValueError(f'Existing pinned model corrupt/missing: {kind}/{name}')
            specs[kind]={**spec,**prior};atomic_json(configuration,specs)
            print(f'Verified existing {kind} revision {prior["revision"]}',flush=True)
            continue
        repo=spec['repo_id'];requested=spec.get('revision') or 'main'
        response=session.get(f'https://huggingface.co/api/models/{repo}/revision/{requested}',params={'blobs':'true'},timeout=(30,120));response.raise_for_status();meta=response.json()
        revision=meta['sha']
        if len(revision)!=40:raise ValueError('Expected full immutable model commit SHA')
        siblings={f['rfilename']:f for f in meta['siblings']}
        weights='model.safetensors' if 'model.safetensors' in siblings else 'pytorch_model.bin'
        if weights not in siblings:raise ValueError(f'No known model weights in {repo}')
        names=[n for n in siblings if '/' not in n and n.endswith(('.json','.yaml','.txt','.py','.md'))]+[weights]
        folder=root/'models'/spec['local_dir'];folder.mkdir(parents=True,exist_ok=True);receipts={}
        for name in sorted(set(names)):
            item=siblings[name];target=folder/name
            expected=(item.get('lfs') or {}).get('sha256');size=item.get('size')
            if target.is_file() and (expected and digest(target)==expected):pass
            else:
                temporary=target.with_suffix(target.suffix+'.part')
                url=f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
                with session.get(url,stream=True,timeout=(30,180)) as body:
                    body.raise_for_status()
                    with temporary.open('wb') as output:
                        for chunk in body.iter_content(1<<20):
                            if chunk:output.write(chunk)
                        output.flush();os.fsync(output.fileno())
                if size is not None and temporary.stat().st_size!=size:raise ValueError(f'Model byte count mismatch {kind}/{name}')
                actual=digest(temporary)
                if expected and actual!=expected:raise ValueError(f'Model upstream LFS SHA256 mismatch {kind}/{name}')
                temporary.replace(target)
            receipts[name]=digest(target)
            print(f'{kind}/{name}: {target.stat().st_size} bytes, SHA256 verified',flush=True)
        entry={**spec,'revision':revision,'files':receipts}
        manifest[kind]=entry;specs[kind]=entry
        atomic_json(manifest_path,manifest);atomic_json(configuration,specs)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();acquire(args.root.resolve())
