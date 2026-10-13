#!/usr/bin/env python3
"""Single entry point for perceived EEG-to-speech acquisition and experiments."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eegspeech.config import Config
from eegspeech.utils import device_name,read_json


def parser():
    root=argparse.ArgumentParser(description=__doc__)
    sub=root.add_subparsers(dest="command",required=True)
    for name in ["download","prepare","oracle","fit-trf","train-encoder","train-flow","matrix","freeze","evaluate","adapt","report","infer","neural-diagnostic"]:
        p=sub.add_parser(name)
        p.add_argument("--config",type=Path,default=Path(__file__).resolve().parents[1]/"configs/perceived_flow.yaml")
        p.add_argument("--device",default="auto",choices=["auto","cpu","cuda"])
        if name in {"fit-trf","train-encoder","train-flow","evaluate","adapt","infer","neural-diagnostic"}:p.add_argument("--fold",choices=["F0","F1","F2","F3","F4","all"],default="F0" if name not in {"evaluate","adapt"} else "all")
        if name in {"oracle","train-encoder","train-flow","evaluate","adapt","infer","neural-diagnostic"}:p.add_argument("--seed",type=int,choices=[11,22,33],default=None if name in {"oracle","evaluate","adapt"} else 11)
        if name in {"train-flow","evaluate","infer"}:p.add_argument("--system",choices=["B0","B1","B2","all"],default="all" if name=="evaluate" else "B2")
        if name in {"oracle","train-encoder","train-flow","matrix"}:p.add_argument("--resume",action="store_true")
        if name=="train-encoder":p.add_argument("--encoder",choices=["baseline","physiology","both"],default="both")
        if name=="download":
            p.add_argument("--subjects",nargs="+")
            p.add_argument("--workers",type=int,default=4)
            p.add_argument("--models-only",action="store_true")
            p.add_argument("--data-only",action="store_true")
        if name=="prepare":p.add_argument("--subjects",nargs="+")
        if name in {"evaluate","report"}:p.add_argument("--split",choices=["validation","test"],default="validation" if name=="evaluate" else "test")
        if name=="evaluate":p.add_argument("--adapted",action="store_true")
        if name=="adapt":p.add_argument("--seconds",type=int,default=60)
        if name=="infer":
            p.add_argument("--input",type=Path,required=True,help="Processed µV EEG .npy, shape [832,128]")
            p.add_argument("--mask",type=Path)
            p.add_argument("--output",type=Path,required=True)
            p.add_argument("--noise-seed",type=int,default=0)
    return root


def main():
    args=parser().parse_args();cfg=Config(args.config)
    device=device_name(args.device)
    folds=list(cfg.splits["test_folds"]) if getattr(args,"fold","all")=="all" else [args.fold]
    seeds=cfg["training"]["seeds"] if getattr(args,"seed",None) is None else [args.seed]
    systems=["B0","B1","B2"] if getattr(args,"system","all")=="all" else [args.system]
    if args.command=="download":
        from eegspeech.acquire import download_dataset,download_models
        if args.models_only and args.data_only:raise ValueError("Choose one download scope")
        if not args.models_only:download_dataset(cfg.dir("raw"),cfg["dataset"]["revision"],args.subjects,args.workers)
        if not args.data_only:download_models(cfg)
    elif args.command=="prepare":
        from eegspeech.perceived_audio import FrozenAudio
        from eegspeech.perceived_data import prepare,fit_fold_statistics
        from eegspeech.pipeline import environment_manifest
        manifest=prepare(cfg,FrozenAudio(cfg,device),args.subjects)
        environment_manifest(cfg)
        if manifest["complete_subjects"]:
            for fold in cfg.splits["test_folds"]:fit_fold_statistics(cfg,fold)
        print(json.dumps({"subjects":len(manifest["subjects"]),"windows":len(manifest["windows"]),"complete":manifest["complete_subjects"]}))
    elif args.command=="oracle":
        from eegspeech.training import train_oracle
        for seed in seeds:train_oracle(cfg,seed,device,args.resume)
    elif args.command=="fit-trf":
        from eegspeech.neural_forward import fit_trfs
        for fold in folds:fit_trfs(cfg,fold)
    elif args.command=="train-encoder":
        from eegspeech.training import train_encoder
        variants=[False,True] if args.encoder=="both" else [args.encoder=="physiology"]
        for fold in folds:
            for seed in seeds:
                for variant in variants:train_encoder(cfg,fold,seed,variant,device,args.resume)
    elif args.command=="train-flow":
        from eegspeech.training import train_flow
        for fold in folds:
            for seed in seeds:
                for system in systems:train_flow(cfg,fold,seed,system,device,args.resume)
    elif args.command=="matrix":
        from eegspeech.training import train_oracle,train_encoder,train_flow
        from eegspeech.neural_forward import fit_trfs
        for seed in cfg["training"]["seeds"]:
            gate=train_oracle(cfg,seed,device,args.resume)
            if not gate["passed"]:raise ValueError("Oracle gate failed; stop formal matrix")
        for fold in cfg.splits["test_folds"]:
            gate=fit_trfs(cfg,fold)
            if not gate["passed"]:raise ValueError(f"TRF gate failed for {fold}; stop B2")
            for seed in cfg["training"]["seeds"]:
                for variant in [False,True]:train_encoder(cfg,fold,seed,variant,device,args.resume)
                for system in ["B0","B1","B2"]:
                    gate=train_flow(cfg,fold,seed,system,device,args.resume)
                    if fold=="F0" and seed==11 and system=="B0" and not gate["passed"]:raise ValueError("Pilot EEG-conditioning gate failed")
    elif args.command=="freeze":
        from eegspeech.pipeline import freeze_protocol
        freeze_protocol(cfg);print("Formal protocol frozen; final test evaluation is enabled")
    elif args.command=="evaluate":
        from eegspeech.evaluation import evaluate_matrix
        rows=evaluate_matrix(cfg,args.split,folds,seeds,systems,device,args.adapted)
        print(f"Saved {len(rows)} content/acoustic/control records")
    elif args.command=="adapt":
        from eegspeech.adaptation import adapt_subjects
        for fold in folds:
            for seed in seeds:adapt_subjects(cfg,fold,seed,device,args.seconds)
    elif args.command=="report":
        from eegspeech.evaluation import summarize_results
        value=summarize_results(cfg,args.split)
        from eegspeech.reporting import result_figures
        result_figures(cfg,value)
        print(f"Reported {len(value['subject_results'])} subject/system/control aggregates")
    elif args.command=="neural-diagnostic":
        from eegspeech.reporting import neural_diagnostics
        for fold in folds:
            for seed in seeds:neural_diagnostics(cfg,fold,seed,device)
    elif args.command=="infer":
        import numpy as np
        import soundfile as sf
        from eegspeech.inference import Reconstructor
        if len(folds)!=1 or len(seeds)!=1 or len(systems)!=1:raise ValueError("Inference requires one fold/seed/system")
        mask=np.load(args.mask) if args.mask else None
        model=Reconstructor(cfg,folds[0],seeds[0],systems[0],device)
        wave=model(np.load(args.input),mask,args.noise_seed).squeeze(0).cpu().numpy()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        sf.write(args.output,wave,24000,subtype="FLOAT")
        print(str(args.output.resolve()))


if __name__=="__main__":main()
