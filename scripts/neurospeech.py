#!/usr/bin/env python3
"""NeuroSpeech v2.1: processed-only acquisition and explicitly invoked experiments."""
from pathlib import Path
import os
import sys
import argparse
import json
import shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'artifacts/neurospeech_v2/runtime/numba'))
os.environ.setdefault('HF_MODULES_CACHE',str(ROOT/'artifacts/neurospeech_v2/runtime/hf_modules'))
os.environ.setdefault('NUMBA_DISABLE_JIT','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path)
    sub=p.add_subparsers(dest='command',required=True)
    ingest=sub.add_parser('ingest');ingest.add_argument('dataset',choices=['ds004408','brennan2019','etard','kymata_english','karaone','all']);ingest.add_argument('--keep-raw',action='store_true')
    sub.add_parser('status');verify=sub.add_parser('verify-data');verify.add_argument('--dataset')
    annotate=sub.add_parser('annotate');annotate.add_argument('dataset',choices=['brennan2019','etard','kymata_english'])
    prep=sub.add_parser('prepare');prep.add_argument('--allow-incomplete',action='store_true')
    sub.add_parser('matrix');freeze=sub.add_parser('freeze');freeze.add_argument('--protocol',choices=['P1','P2'],required=True);freeze.add_argument('--decisions',type=Path,required=True)
    for name in ['copy-gate','oracle','trf','encoder','reliability','flow','evaluate']:
        cmd=sub.add_parser(name);cmd.add_argument('--protocol',choices=['P1','P2'],default='P1');cmd.add_argument('--fold');cmd.add_argument('--seed',type=int,default=11);cmd.add_argument('--system',choices=['S0','S1','S2','S3'],default='S2');cmd.add_argument('--device',default='auto');cmd.add_argument('--resume',action='store_true')
    cmd.add_argument('--mode',choices=['zero-shot','few-shot','noise'],default='zero-shot');cmd.add_argument('--split',choices=['test','validation'],default='test')
    report=sub.add_parser('report');report.add_argument('--protocol',default='P1')
    a=p.parse_args()
    from eegspeech.neurospeech.config import Config
    from eegspeech.neurospeech.storage import verify_dataset
    from eegspeech.utils import read_json
    cfg=Config(a.config)
    if a.command=='status':
        result={'free_GB':shutil.disk_usage(ROOT).free/1e9,'datasets':{}}
        for source in ['ds004408','brennan2019','etard','kymata_english','karaone']:
            path=cfg.dir('processed')/source/'manifest.json'
            if not path.exists():result['datasets'][source]={'status':'pending'};continue
            m=read_json(path);result['datasets'][source]={'complete':m['complete'],'processed_records':len(m['records']),'contents':len(m['contents']),'subjects':m.get('subjects',sorted({r['subject'] for r in m['records'].values()}))}
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    if a.command=='ingest':
        sources=['ds004408','brennan2019','etard','kymata_english','karaone'] if a.dataset=='all' else [a.dataset]
        for source in sources:
            if a.keep_raw and source!='ds004408':raise ValueError('--keep-raw currently applies only to existing ds004408 migration')
            if source=='ds004408':
                from eegspeech.neurospeech.sources.ds004408 import ingest
                result=ingest(root=cfg.dir('processed'),cleanup=not a.keep_raw)
            else:
                import importlib
                name={'brennan2019':'brennan','kymata_english':'kymata','karaone':'karaone','etard':'etard'}[source]
                module=importlib.import_module('eegspeech.neurospeech.sources.'+name)
                result=module.run(root=cfg.dir('processed'))
            print(json.dumps({'dataset':source,'result':result},default=str,ensure_ascii=False))
        return
    if a.command=='annotate':
        from eegspeech.neurospeech import annotation
        if a.dataset=='brennan2019':result=annotation.annotate_brennan(cfg.dir('processed'))
        elif a.dataset=='etard':
            drafts,book=annotation.etard_reference(cfg.dir('processed'))
            result=annotation.annotate_etard(drafts,book,cfg.dir('processed'),source=annotation.BOOK)
        else:result=annotation.lock_content_mask('kymata_english','kymata_english_conversation','untranscribed-or-overlapping-speech',cfg.path.parent/'content_masks.json',cfg.dir('processed'))
        print(json.dumps(result,indent=2));return
    if a.command=='verify-data':
        sources=[a.dataset] if a.dataset else ['ds004408','brennan2019','etard','kymata_english','karaone']
        print(json.dumps([verify_dataset(s,cfg.dir('processed'),hashes=True) for s in sources],indent=2));return
    if a.command=='prepare':
        from eegspeech.neurospeech.data import build_registry,all_rows
        r=build_registry(cfg,require_complete=not a.allow_incomplete);rows=all_rows(cfg)
        print(json.dumps({'registry_hash':r['registry_hash'],'sources':r['sources'],'accepted_windows':len(rows)},ensure_ascii=False,indent=2));return
    if a.command=='matrix':
        from eegspeech.neurospeech.protocol import main_matrix
        print(json.dumps(main_matrix(),indent=2));return
    if a.command=='freeze':
        from eegspeech.neurospeech.protocol import freeze_protocol
        print(json.dumps(freeze_protocol(cfg,a.protocol,read_json(a.decisions)),indent=2));return
    if a.command=='report':
        from eegspeech.neurospeech.reporting import report
        print(report(cfg,a.protocol));return
    from eegspeech.utils import device_name
    from eegspeech.neurospeech import training
    device=device_name(a.device);fold=a.fold or ('external' if a.protocol=='P2' else 'F0')
    if a.command=='copy-gate':
        from eegspeech.neurospeech.data import AudioDataset
        from eegspeech.neurospeech.evaluation import copy_synthesis_gate
        from eegspeech.perceived_audio import FrozenAudio,ASR
        train=AudioDataset(cfg,protocol=a.protocol,fold=fold,split='train')
        val=AudioDataset(cfg,protocol=a.protocol,fold=fold,split='validation',fixed_validation=True,stats=train.stats)
        result=copy_synthesis_gate(cfg,val,FrozenAudio(cfg,device),ASR(cfg,device),device=device)
    elif a.command=='oracle':result=training.train_oracle(cfg,protocol=a.protocol,seed=a.seed,device=device,resume=a.resume)
    elif a.command=='trf':result=training.train_trfs(cfg,protocol=a.protocol,fold=fold,seed=a.seed,device=device)
    elif a.command=='encoder':result=training.train_encoder(cfg,protocol=a.protocol,fold=fold,seed=a.seed,system=a.system,device=device,resume=a.resume)
    elif a.command=='reliability':result=training.calibrate_reliability(cfg,protocol=a.protocol,fold=fold,seed=a.seed,device=device)
    elif a.command=='flow':result=training.train_flow(cfg,protocol=a.protocol,fold=fold,seed=a.seed,system=a.system,device=device,resume=a.resume)
    else:
        from eegspeech.neurospeech.evaluation import evaluate
        result=evaluate(cfg,protocol=a.protocol,fold=fold,seed=a.seed,system=a.system,split=a.split,device=device,mode=a.mode)
    print(json.dumps(result,default=str,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
