"""Free ASR, paired EEG controls, native acoustics and fixed candidate probes."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import default_collate
from .metrics import aggregate_wer, bootstrap_mean, holm, paired_test, two_way_bootstrap, word_errors
from .perceived_audio import ASR, FrozenAudio, acoustic_features, read_wave
from .perceived_data import WindowDataset, reference, textgrid_words
from .pipeline import decode_mel, provenance, require_provenance, require_test_lock
from .utils import atomic_json, digest, load_checkpoint, read_json, stable_int


def wrong_time_mapping(rows, min_seconds=10):
    result = {}
    for i, row in enumerate(rows):
        candidates = [j for j,r in enumerate(rows) if r.get("subject")==row.get("subject") and r["audio"]==row["audio"] and abs(r["start"]-row["start"])>=min_seconds]
        if not candidates: raise ValueError(f"No disjoint wrong-time control for {row['key']}")
        candidates.sort(key=lambda j: stable_int(row["key"]+"|"+rows[j]["key"]))
        result[i]=candidates[0]
    return result


def numpy_correlation(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    a=a-a.mean();b=b-b.mean();denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denominator) if denominator>1e-12 else 0.0


@torch.no_grad()
def copy_synthesis_gate(cfg,audio,asr):
    from .training import AudioDataset
    path=cfg.dir("artifacts")/"copy_validation.json"
    if path.exists():
        saved=read_json(path)
        if saved.get("provenance")==provenance(cfg): return saved
        raise ValueError("Copy-synthesis cache has stale provenance")
    rows=[];length_errors=[]
    dataset=AudioDataset(cfg,"validation",fixed_validation=True)
    for row in dataset.rows:
        full=read_wave(cfg.dir("raw")/"stimuli"/f"{row['audio']}.wav")
        a=row["start"];wave=full[round(a*24000):round((a+4)*24000)]
        words=textgrid_words(cfg.dir("raw")/"stimuli"/f"{row['audio']}.TextGrid")
        ref=asr.normalize(reference(words,a))
        tensor=torch.from_numpy(wave)[None].to(audio.device)
        mel=audio.vocos.features(tensor);copy=audio.vocos(mel).squeeze(0).cpu().numpy()
        length_errors.append(abs(len(copy)-len(wave)))
        original=word_errors(ref,asr(wave));synthesized=word_errors(ref,asr(copy))
        rows.append({**row,"original":original,"copy":synthesized})
    count=sum(r["original"]["words"] for r in rows)
    if not count: raise ValueError("Validation references contain no words")
    original=sum(r["original"]["errors"] for r in rows)/count
    copy=sum(r["copy"]["errors"] for r in rows)/count
    value={"provenance":provenance(cfg),"original_wer":original,"copy_wer":copy,"max_length_error":max(length_errors),"passed":(copy-original)*100<=cfg["evaluation"]["copy_max_wer_delta_pp"] and max(length_errors)<=cfg["audio"]["hop_length"],"rows":rows}
    atomic_json(path,value)
    return value


class RetrievalProbe:
    def __init__(self,cfg,rows,audio,asr):
        if audio.teacher is None:raise ValueError("Retrieval requires a frozen evaluation teacher")
        self.cfg,self.audio=cfg,audio
        self.rows={r["key"]:r for r in rows}
        self.targets={};self.reference={}
        for key,row in self.rows.items():
            words=textgrid_words(cfg.dir("raw")/"stimuli"/f"{row['audio']}.TextGrid")
            self.reference[key]=asr.normalize(reference(words,row["start"]))
        self.wave_cache={}

    @torch.no_grad()
    def feature(self,wave):
        from scipy.signal import resample_poly
        wave16=resample_poly(wave,2,3).astype(np.float32)
        features,_=self.audio.wavlm(wave16,crop_start=0,core_seconds=None)
        pooled=features.mean(1).squeeze(0)
        return torch.nn.functional.normalize(pooled,dim=0).cpu().numpy()

    def candidates(self,key):
        anchor=self.rows[key]
        pool=[]
        for other,row in self.rows.items():
            if other==key:continue
            if row["audio"]==anchor["audio"] and abs(row["start"]-anchor["start"])<4:continue
            if self.reference[other]==self.reference[key]:continue
            pool.append(other)
        pool.sort(key=lambda candidate:stable_int(key+"|"+candidate))
        return [key]+pool[:self.cfg["evaluation"]["retrieval_candidates"]-1]

    def target(self,key):
        if key not in self.targets:
            row=self.rows[key]
            if row["audio"] not in self.wave_cache:self.wave_cache[row["audio"]]=read_wave(self.cfg.dir("raw")/"stimuli"/f"{row['audio']}.wav")
            a=row["start"];wave=self.wave_cache[row["audio"]][round(a*24000):round((a+4)*24000)]
            self.targets[key]=self.feature(wave)
        return self.targets[key]

    def score(self,key,wave):
        candidates=self.candidates(key)
        similarity=np.stack([self.target(c) for c in candidates])@self.feature(wave)
        return {"retrieval_top1":int(similarity.argmax()==0),"retrieval_candidates":len(candidates),"retrieval_chance":1/len(candidates)}


@torch.no_grad()
def score_model(cfg,dataset,flow,audio,asr,encoder=None,controls=None,generation_seeds=None,training_seed=11,system="B2",save_dir=None,adapters=None,retrieval=False):
    from .training import move_batch
    controls=controls or ["correct","wrong","null"]
    generation_seeds=generation_seeds or cfg["evaluation"]["generation_seeds"]
    device=audio.device
    flow.eval()
    if encoder is not None:encoder.eval()
    stats=read_json(cfg.dir("artifacts")/"stats/audio.json")
    wrong=wrong_time_mapping(dataset.rows,cfg["evaluation"]["wrong_min_seconds"]) if "wrong" in controls else {}
    probe=RetrievalProbe(cfg,dataset.rows,audio,asr) if retrieval else None
    wave_cache={};words_cache={};rows=[]
    listening_keys=set()
    for person in {r.get("subject","audio") for r in dataset.rows}:
        for clip in {r["audio"] for r in dataset.rows}:
            pool=[r for r in dataset.rows if r.get("subject","audio")==person and r["audio"]==clip]
            if pool:listening_keys.add(min(pool,key=lambda r:stable_int(r["key"]))["key"])
    frames=round(cfg["window"]["seconds"]*cfg["audio"]["sample_rate"]//cfg["audio"]["hop_length"])+1
    if save_dir:save_dir.mkdir(parents=True,exist_ok=True)
    for index,row in enumerate(dataset.rows):
        batch=move_batch(default_collate([dataset[index]]),device)
        clip=row["audio"]
        if clip not in wave_cache:
            wave_cache[clip]=read_wave(cfg.dir("raw")/"stimuli"/f"{clip}.wav")
            words_cache[clip]=textgrid_words(cfg.dir("raw")/"stimuli"/f"{clip}.TextGrid")
        a=row["start"];truth=wave_cache[clip][round(a*24000):round((a+4)*24000)]
        target_phi=acoustic_features(torch.from_numpy(truth)[None]).squeeze(0).numpy()
        ref=asr.normalize(reference(words_cache[clip],a))
        adapter=(adapters or {}).get(row.get("subject"))
        if encoder is None:condition=batch["teacher"]
        else:condition=encoder(batch["eeg"],batch["times"],batch["channel_mask"],adapter)["teacher"]
        for control in controls:
            if control=="correct":cond=condition
            elif control=="null":cond=torch.zeros_like(condition)
            elif control=="wrong":
                other=move_batch(default_collate([dataset[wrong[index]]]),device)
                cond=encoder(other["eeg"],batch["times"],other["channel_mask"],adapter)["teacher"]
            else:raise ValueError(control)
            for sample_seed in generation_seeds:
                noise_seed=stable_int(f"{row['key']}|{sample_seed}")
                generated=flow.sample(cond,frames,noise_seed,cfg["flow"]["sample_steps"])
                wave=decode_mel(generated,stats["mel"],audio.vocos).squeeze(0).float().cpu().numpy()
                hyp=asr(wave);errors=word_errors(ref,hyp)
                phi=acoustic_features(torch.from_numpy(wave)[None]).squeeze(0).numpy()
                scale=torch.tensor(stats["mel"]["std"],device=device)[None,:,None]
                native_mae=float(((generated-batch["mel"])*scale).abs().mean())
                native_pred=torch.exp(generated*scale+torch.tensor(stats["mel"]["mean"],device=device)[None,:,None])
                native_true=torch.exp(batch["mel"]*scale+torch.tensor(stats["mel"]["mean"],device=device)[None,:,None])
                sc=float(torch.linalg.vector_norm(native_pred-native_true)/torch.linalg.vector_norm(native_true).clamp_min(1e-8))
                result={"id":row.get("id",row["key"]),"subject":row.get("subject","audio"),"audio":clip,"key":row["key"],"start":a,"system":system,"control":control,"training_seed":training_seed,"generation_seed":sample_seed,"noise_seed":noise_seed,"reference":ref,"hypothesis":hyp,**errors,"mel_mae":native_mae,"spectral_convergence":sc,"envelope_r":numpy_correlation(phi[:,0],target_phi[:,0]),"onset_r":numpy_correlation(phi[:,1],target_phi[:,1]),"waveform_peak":float(np.abs(wave).max())}
                if probe:result.update(probe.score(row["key"],wave))
                if save_dir and training_seed==11 and sample_seed==0 and row["key"] in listening_keys:
                    # Fixed sample IDs; no content-dependent selection for listening.
                    name=f"{row.get('subject','audio')}_{row['key']}_{control}_{sample_seed}.wav"
                    sf.write(save_dir/name,wave,24000,subtype="FLOAT")
                    result["waveform"]=str((save_dir/name).relative_to(cfg.root))
                rows.append(result)
    aggregates=aggregate_wer(rows)
    per_person={}
    for r in aggregates:per_person.setdefault(r["subject"],{})[r["control"]]=r["wer"]
    control_wer={c:float(np.mean([v[c] for v in per_person.values()])) for c in controls}
    result={"wer":control_wer["correct"],"control_wer":control_wer,"subject_control_wer":per_person,"mel_mae":float(np.mean([r["mel_mae"] for r in rows if r["control"]=="correct"])),"rows":rows}
    if save_dir:atomic_json(save_dir/"scores.json",result)
    return result


def evaluate_matrix(cfg,split,folds,seeds,systems,device="cpu",adapted=False):
    from .flow import ConditionalFlow
    from .training import build_encoder
    from .temporal import Adapter
    audio=FrozenAudio(cfg,device,teacher=True);asr=ASR(cfg,device)
    all_rows=[]
    for fold in folds:
        for seed in seeds:
            for system in systems:
                path=cfg.run_dir(fold,seed,system)/"best.pt"
                if split=="test":require_test_lock(cfg,[path,cfg.dir("artifacts")/"stats"/f"{fold}.json"])
                saved=load_checkpoint(path,device);require_provenance(saved,cfg,fold)
                encoder=build_encoder(cfg,system!="B0").to(device);encoder.load_state_dict(saved["encoder"])
                flow=ConditionalFlow(cfg).to(device);flow.load_state_dict(saved["flow"])
                dataset=WindowDataset(cfg,fold,split,fixed_validation=split=="validation")
                adapters={}
                if adapted:
                    if system!="B2":raise ValueError("Adapter is only defined for B2")
                    for subject in cfg.fold_subjects(fold,"test"):
                        source=cfg.dir("artifacts")/"adapters"/fold/str(seed)/f"{subject}.pt"
                        if source.exists():
                            state=load_checkpoint(source,device)
                            if state["base_checkpoint_hash"]!=digest(path):raise ValueError("Adapter belongs to a different base model")
                            module=Adapter(cfg["encoder"]["width"],cfg["encoder"]["adapter_rank"]).to(device)
                            module.load_state_dict(state["adapter"]);module.eval();adapters[subject]=module
                    dataset.rows=[r for r in dataset.rows if r["subject"] in adapters]
                    if not dataset.rows:raise ValueError("No eligible calibrated test subjects")
                tag=system+"_60s" if adapted else system
                destination=cfg.dir("outputs")/split/fold/str(seed)/tag
                value=score_model(cfg,dataset,flow,audio,asr,encoder,save_dir=destination,adapters=adapters,retrieval=True,training_seed=seed,system=tag)
                atomic_json(destination/"provenance.json",{**provenance(cfg,fold),"checkpoint_hash":digest(path),"adapted":adapted})
                all_rows.extend(value["rows"])
    return all_rows


def summarize_results(cfg,split="test"):
    paths=sorted((cfg.dir("outputs")/split).rglob("scores.json"))
    if not paths:raise FileNotFoundError("No completed evaluation scores")
    rows=[]
    for path in paths:rows.extend(read_json(path)["rows"])
    summary=aggregate_wer(rows)
    table={(r["subject"],r["system"],r["control"]):r["wer"] for r in summary}
    people=sorted({r["subject"] for r in summary})
    comparisons={}
    for name,a,b,ca,cb in [("primary_B1_minus_B2","B1","B2","correct","correct"),("encoder_B0_minus_B1","B0","B1","correct","correct"),("wrong_minus_correct","B2","B2","wrong","correct"),("null_minus_correct","B2","B2","null","correct"),("zero_shot_minus_60s","B2","B2_60s","correct","correct")]:
        paired=[s for s in people if (s,a,ca) in table and (s,b,cb) in table]
        if paired:
            comparison=paired_test([table[s,a,ca]-table[s,b,cb] for s in paired],cfg["evaluation"]["bootstrap"])
            comparison["subjects"]=paired
            subset_a=[r for r in rows if r["subject"] in paired and r["system"]==a and r["control"]==ca]
            subset_b=[r for r in rows if r["subject"] in paired and r["system"]==b and r["control"]==cb]
            comparison["two_way_sensitivity"]=two_way_bootstrap(subset_a,subset_b,cfg["evaluation"]["bootstrap"])
            comparisons[name]=comparison
    secondaries=[k for k in ["encoder_B0_minus_B1","wrong_minus_correct","null_minus_correct"] if k in comparisons]
    for name,p in zip(secondaries,holm([comparisons[k]["p_two_sided"] for k in secondaries])):comparisons[name]["p_holm"]=p
    observed={(r["system"],r["training_seed"],r["subject"]) for r in rows if r["system"] in {"B0","B1","B2"}}
    value={"split":split,"config_hash":cfg.hash,"subject_results":summary,"comparisons":comparisons,"independent_unit":"subject","test_stimuli":cfg.splits["audio"].get(split,[]),"matrix_subject_seed_coverage":len(observed),"expected_test_coverage":144 if split=="test" else None,"status":"COMPLETE" if split=="test" and len(observed)==144 else "PARTIAL"}
    atomic_json(cfg.dir("outputs")/f"summary_{split}.json",value)
    lines=[f"# Perceived EEG-to-speech results ({split})","",f"Coverage status: {value['status']}. All numbers below come from saved evaluation records.","","| System / condition | Subjects | Macro WER |","|---|---:|---:|"]
    for model,control in sorted({(r["system"],r["control"]) for r in summary}):
        selected=[r["wer"] for r in summary if r["system"]==model and r["control"]==control]
        lines.append(f"| {model} / {control} | {len(selected)} | {np.mean(selected)*100:.2f}% |")
    lines.extend(["","## Paired content improvements","","| Contrast | n | WER improvement (pp) | 95% CI (pp) | Two-sided p |","|---|---:|---:|---:|---:|"])
    for name,c in comparisons.items():lines.append(f"| {name} | {c['n']} | {100*c['mean']:.2f} | [{100*c['lower']:.2f}, {100*c['upper']:.2f}] | {c['p_two_sided']:.4f} |")
    lines.extend(["","Only three held-out stimuli are available. Seeds and generated samples are not independent participants.","","The primary endpoint is B1 WER minus B2 WER. Acoustic or TRF improvements alone do not establish content recovery."])
    (cfg.root/"report"/f"perceived_flow_results_{split}.md").write_text("\n".join(lines)+"\n")
    return value
