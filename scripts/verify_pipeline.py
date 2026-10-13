#!/usr/bin/env python3
"""Post-implementation real-file contract checks; never a scientific experiment."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    import torch
    from eegspeech.config import Config
    from eegspeech.flow import ConditionalFlow
    from eegspeech.neural_forward import FrozenTRF,neural_consistency
    from eegspeech.perceived_audio import ASR,FrozenAudio,read_wave
    from eegspeech.perceived_data import prepare_recording,recording_dir,textgrid_words,windows_for_recording,reference
    from eegspeech.pipeline import environment_manifest
    from eegspeech.temporal import EEGEncoder
    from eegspeech.utils import atomic_json,read_json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=Path(__file__).resolve().parents[1]/"configs/perceived_flow.yaml")
    p.add_argument("--device",default="cpu")
    p.add_argument("--asr",action="store_true")
    args=p.parse_args();cfg=Config(args.config)
    torch.set_num_threads(4)
    acquired=read_json(cfg.dir("raw")/"acquisition.json")
    assert acquired["complete"] and acquired["revision"]==cfg["dataset"]["revision"]
    for item in acquired["files"]:assert (cfg.dir("raw")/item["path"]).stat().st_size==item["size"]
    print("Paired source inventory: passed",flush=True)
    source=cfg.dir("raw")/"sub-007/eeg/sub-007_task-listening_run-01_eeg.vhdr"
    meta=prepare_recording(cfg,source)
    rows,rejected=windows_for_recording(cfg,meta)
    if not rows:raise ValueError("Real training run has no QC-approved windows")
    row=rows[0];a=row["start"]
    wave24=read_wave(cfg.dir("raw")/"stimuli/audio01.wav",24000)
    wave16=read_wave(cfg.dir("raw")/"stimuli/audio01.wav",16000)
    core=wave24[round(a*24000):round((a+4)*24000)]
    model=FrozenAudio(cfg,args.device)
    teacher,times=model.wavlm(wave16[round((a-1)*16000):round((a+5)*16000)])
    assert teacher.shape==(1,200,768)
    assert torch.isfinite(teacher).all()
    print("Real WavLM frame sequence: passed",flush=True)
    tensor=torch.from_numpy(core)[None].to(args.device)
    with torch.no_grad():mel=model.vocos.features(tensor);copy=model.vocos(mel)
    assert mel.shape==(1,100,376) and copy.shape==(1,96000)
    assert torch.isfinite(copy).all()
    print("Native Vocos feature/decode contract: passed",flush=True)
    log_mel=mel.detach().clone().requires_grad_(True)
    decoded=model.vocos(log_mel)
    decoded.square().mean().backward()
    assert log_mel.grad is not None and torch.isfinite(log_mel.grad).all() and log_mel.grad.abs().sum()>0
    assert all(v.grad is None for v in model.vocos.parameters())
    print("Frozen Vocos input gradients: passed",flush=True)
    raw=np.load(recording_dir(cfg,"sub-007","audio01")/"eeg.npy",mmap_mode="r")
    first=round((a-1)*128)
    eeg=np.array(raw[:,first:first+832].T)
    # This scaling is only for a contract smoke test, not saved fold statistics.
    eeg=(eeg-eeg.mean(0))/(eeg.std(0)+1e-3)
    encoder=EEGEncoder(cfg,meta["montage"]["groups"]).to(args.device).eval()
    mask=torch.tensor(row["good_channels"],device=args.device)[None]
    predicted=encoder(torch.tensor(eeg,dtype=torch.float32,device=args.device)[None],times,mask)["teacher"]
    assert predicted.shape==teacher.shape
    flow=ConditionalFlow(cfg).to(args.device)
    # Real architecture + real frozen vocoder, synthetic TRF weights for gradient QA.
    noise=torch.randn_like(mel)
    generated=flow.rollout(noise,predicted.detach(),2,"euler",checkpoint_grad=True)
    mel_mean=mel.detach().mean(-1,keepdim=True)
    mel_std=mel.detach().std(-1,keepdim=True).clamp_min(.1)
    wave=model.vocos(generated*mel_std+mel_mean)
    trf=FrozenTRF({"weights":torch.randn(67,128)*.01,"selected":torch.ones(128,dtype=torch.bool),"variance":torch.ones(128),"phi_mean":torch.zeros(2),"phi_std":torch.ones(2),"lags":33}).to(args.device)
    loss=neural_consistency(wave,torch.zeros(1,256,128,device=args.device),mask,trf)
    loss.backward()
    gradients=[p.grad for p in flow.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients)>0
    assert all(p.grad is None for p in model.vocos.parameters())
    print("Real CFM → Vocos → acoustic features → frozen TRF gradients: passed",flush=True)
    report={"status":"VERIFIED_CODE_CONTRACTS","scientific_performance":"UNVERIFIED; no oracle/EEG training or TRF physiology gate has run","dataset_files":len(acquired["files"]),"eeg_recordings":sum(f["path"].endswith("_eeg.eeg") for f in acquired["files"]),"audio_pairs":20,"dataset_revision":acquired["revision"],"training_source":str(source.relative_to(cfg.root)),"accepted_source_windows":len(rows),"rejected_source_windows":rejected,"wavlm_shape":list(teacher.shape),"mel_shape":list(mel.shape),"waveform_samples":copy.shape[-1],"vocos_input_gradient":True,"cfm_neural_gradient":True,"trf_for_gradient_check":"synthetic weights, not a fitted scientific forward model","environment":environment_manifest(cfg)}
    if args.asr:
        listener=ASR(cfg,args.device)
        words=textgrid_words(cfg.dir("raw")/"stimuli/audio01.TextGrid")
        report["asr_smoke"]={"reference":listener.normalize(reference(words,a)),"original":listener(core),"copy":listener(copy.squeeze(0).cpu().numpy())}
        print("Frozen English ASR interface: passed",flush=True)
    atomic_json(cfg.root/"report/implementation_checks.json",report)
    print("Saved report/implementation_checks.json; scientific experiments were not run",flush=True)


if __name__=="__main__":main()
