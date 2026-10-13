"""One supervised 60-second adapter; all temporal/filter support stays inside budget."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from .perceived_data import calibrated_microvolts,filter_eeg, qc_window, recording_dir
from .pipeline import require_provenance, require_test_lock
from .temporal import Adapter
from .training import BalancedBatches, build_encoder
from .utils import atomic_checkpoint, atomic_json, digest, load_checkpoint, read_json, seed_all


class CalibrationDataset(Dataset):
    def __init__(self,cfg,fold,subject,seconds=60):
        import mne
        if seconds!=cfg["adapter"]["seconds"]:raise ValueError("The locked adapter budget is 60 seconds")
        self.cfg=cfg
        meta=read_json(recording_dir(cfg,subject,"audio01")/"meta.json")
        source=cfg.dir("raw")/meta["eeg_source"]
        raw=mne.io.read_raw_brainvision(source,preload=False,verbose="ERROR")
        names=meta["montage"]["names"]
        raw.pick(names);raw.reorder_channels(names)
        rate=raw.info["sfreq"]
        start=cfg["adapter"]["start_seconds"]
        first=round((start+meta["offset"])*rate);last=first+round(seconds*rate)
        if last>raw.n_times:raise ValueError("Recording is shorter than calibration budget")
        # Read exactly 60 seconds, then filter this block alone with reflected padding.
        uv=calibrated_microvolts(raw.get_data(start=first,stop=last),cfg)
        self.good=np.array(meta["good_channels"])
        uv-=uv[self.good].mean(0,keepdims=True);uv[~self.good]=0
        self.eeg,_,self.filter_support=filter_eeg(uv,rate,cfg)
        self.stats=read_json(cfg.dir("artifacts")/"stats"/f"{fold}.json")["eeg"]
        self.audio_stats=read_json(cfg.dir("artifacts")/"stats/audio.json")
        self.rows=[]
        guard=cfg["preprocessing"]["guard_seconds"]
        for local in np.arange(guard,seconds-guard-4+1e-6,2):
            core=self.eeg[:,round(local*128):round((local+4)*128)]
            mask,accepted=qc_window(core,self.good,cfg)
            if accepted:self.rows.append({"subject":subject,"audio":"audio01","start":float(start+local),"local":float(local),"key":f"audio01_{start+local:.3f}","good":mask.tolist()})
        if not self.rows:raise ValueError("Calibration has no accepted internal windows")
        self.receipt={"subject":subject,"audio":"audio01","eeg_start_sample":first,"eeg_stop_sample":last,"raw_rate":rate,"raw_seconds":seconds,"source_sha256":meta["eeg_sha256"],"filter_uses_only_calibration_block":True,"valid_core_windows":len(self.rows)}

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index];start=round((row["local"]-1)*128)
        eeg=self.eeg[:,start:start+832].T.copy()
        eeg=(eeg-self.stats["mean"])/self.stats["std"]
        mask=np.array(row["good"]);eeg[:,~mask]=0
        with np.load(self.cfg.dir("artifacts")/"audio_targets"/f"{row['key']}.npz") as target:
            teacher=(target["teacher"].astype(np.float32)-self.audio_stats["teacher"]["mean"])/self.audio_stats["teacher"]["std"]
            phi=(target["phi"]-self.audio_stats["phi"]["mean"])/self.audio_stats["phi"]["std"]
            return {"subject":row["subject"],"audio":"audio01","eeg":torch.tensor(eeg,dtype=torch.float32),"channel_mask":torch.from_numpy(mask),"times":torch.from_numpy(target["times"].copy()),"teacher":torch.tensor(teacher,dtype=torch.float32),"phi":torch.tensor(phi,dtype=torch.float32)}


def adapt_subjects(cfg,fold,seed,device="cpu",seconds=60):
    path=cfg.run_dir(fold,seed,"B2")/"best.pt"
    require_test_lock(cfg,[path,cfg.dir("artifacts")/"stats"/f"{fold}.json"])
    state=load_checkpoint(path,device);require_provenance(state,cfg,fold)
    encoder=build_encoder(cfg,True).to(device).eval();encoder.load_state_dict(state["encoder"])
    encoder.requires_grad_(False)
    folder=cfg.dir("artifacts")/"adapters"/fold/str(seed);folder.mkdir(parents=True,exist_ok=True)
    results={}
    for subject in cfg.fold_subjects(fold,"test"):
        seed_all(seed)
        try:dataset=CalibrationDataset(cfg,fold,subject,seconds)
        except (FileNotFoundError,ValueError) as error:
            results[subject]={"status":"missing_calibration","reason":str(error)};continue
        batches=BalancedBatches(dataset,seed)
        adapter=Adapter(cfg["encoder"]["width"],cfg["encoder"]["adapter_rank"]).to(device)
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=cfg["adapter"]["lr"],weight_decay=cfg["training"]["weight_decay"])
        for step in range(cfg["adapter"]["steps"]):
            batch=batches.next(cfg["adapter"]["batch"],device)
            with torch.no_grad():hidden=encoder.hidden(batch["eeg"],batch["times"],batch["channel_mask"])
            delta=adapter.residual(hidden);adapted=hidden+delta
            predicted=encoder.content(adapted);phi=encoder.acoustics(adapted)
            loss=(predicted-batch["teacher"]).square().mean()+.1*(phi[...,0]-batch["phi"][...,0]).square().mean()+.05*(phi[...,1]-batch["phi"][...,1]).square().mean()+cfg["adapter"]["residual_penalty"]*delta.square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(),cfg["training"]["grad_clip"],error_if_nonfinite=True)
            optimizer.step()
        saved={"adapter":adapter.state_dict(),"base_checkpoint_hash":digest(path),"provenance":state["provenance"],"calibration":dataset.receipt,"steps":cfg["adapter"]["steps"],"parameters":sum(p.numel() for p in adapter.parameters()),"labels":"known real stimulus audio01; supervised"}
        atomic_checkpoint(folder/f"{subject}.pt",saved)
        results[subject]={"status":"completed",**dataset.receipt,"parameters":saved["parameters"]}
    atomic_json(folder/"summary.json",results)
    return results
