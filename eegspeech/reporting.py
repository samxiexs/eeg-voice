"""Publication figures and forward-consistency diagnostics from actual results."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import default_collate
from .utils import atomic_json,load_checkpoint,read_json,stable_int


def result_figures(cfg,summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=summary["subject_results"]
    table={(r["subject"],r["system"],r["control"]):100*r["wer"] for r in rows}
    folder=cfg.dir("outputs")/"figures"/summary["split"]
    folder.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    comparisons=[[("B0","correct"),("B1","correct"),("B2","correct")],[("B2","correct"),("B2","wrong"),("B2","null")]]
    for ax,keys,labels in zip(axes,comparisons,[["B0","B1","B2"],["Correct EEG","Wrong EEG","Null"]]):
        people=sorted({s for s,_,_ in table if all((s,m,c) in table for m,c in keys)})
        if people:
            values=np.array([[table[s,m,c] for m,c in keys] for s in people])
            for line in values:ax.plot(range(3),line,color=".7",linewidth=.7,alpha=.6)
            ax.plot(range(3),values.mean(0),"o-",color="#235c9b",linewidth=2)
        ax.set_xticks(range(3),labels);ax.set_ylabel("Free ASR WER (%)");ax.spines[["top","right"]].set_visible(False)
    fig.suptitle(f"Perceived EEG-to-speech · {summary['split']} · {summary['status']}")
    for suffix in ["png","pdf"]:fig.savefig(folder/f"content_and_controls.{suffix}",dpi=220)
    plt.close(fig)
    people=sorted({s for s,_,_ in table if (s,"B2","correct") in table and (s,"B2_60s","correct") in table})
    if people:
        fig,ax=plt.subplots(figsize=(4,4),constrained_layout=True)
        for s in people:ax.plot([0,1],[table[s,"B2","correct"],table[s,"B2_60s","correct"]],"o-",alpha=.45)
        ax.set_xticks([0,1],["Zero-shot","60 s supervised"]);ax.set_ylabel("Free ASR WER (%)")
        ax.set_title(f"Same B2 model; {len(people)} paired participants")
        for suffix in ["png","pdf"]:fig.savefig(folder/f"adapter.{suffix}",dpi=220)
        plt.close(fig)
    return str(folder)


@torch.no_grad()
def neural_diagnostics(cfg,fold,seed,device="cpu"):
    from .flow import ConditionalFlow
    from .neural_forward import load_trfs,neural_consistency
    from .perceived_audio import FrozenAudio
    from .perceived_data import WindowDataset
    from .pipeline import decode_mel,require_provenance
    from .training import build_encoder,move_batch
    trfs=load_trfs(cfg,fold,device)
    dataset=WindowDataset(cfg,fold,"validation",subjects=cfg.fold_subjects(fold,"train"),fixed_validation=True)
    dataset.rows=[r for r in dataset.rows if r["subject"] in trfs]
    audio=FrozenAudio(cfg,device,teacher=False)
    stats=read_json(cfg.dir("artifacts")/"stats/audio.json")
    results=[]
    for system in ["B1","B2"]:
        state=load_checkpoint(cfg.run_dir(fold,seed,system)/"best.pt",device);require_provenance(state,cfg,fold)
        encoder=build_encoder(cfg,True).to(device).eval();encoder.load_state_dict(state["encoder"])
        flow=ConditionalFlow(cfg).to(device).eval();flow.load_state_dict(state["flow"])
        for index,row in enumerate(dataset.rows):
            batch=move_batch(default_collate([dataset[index]]),device)
            condition=encoder(batch["eeg"],batch["times"],batch["channel_mask"])["teacher"]
            mel=flow.sample(condition,376,stable_int(row["key"]+"|0"),cfg["flow"]["sample_steps"])
            wave=decode_mel(mel,stats["mel"],audio.vocos)
            trf=trfs[row["subject"]]
            valid=int((trf.selected & batch["channel_mask"][0]).sum())
            if not valid:continue
            loss=neural_consistency(wave,batch["neural"],batch["channel_mask"],trf)
            results.append({"subject":row["subject"],"audio":row["audio"],"key":row["key"],"system":system,"training_seed":seed,"valid_neural_channels":valid,"consistency_mse":float(loss)})
    value={"scope":"training subjects; held-out audio15-17; frozen training-only TRFs","config_hash":cfg.hash,"rows":results}
    atomic_json(cfg.dir("outputs")/"neural_diagnostics"/fold/str(seed)/"neural_scores.json",value)
    return value
