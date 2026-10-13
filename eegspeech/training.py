"""Resumable oracle, shared TCN and physiology-guided flow training."""
from __future__ import annotations
from contextlib import nullcontext
import math
import time
import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
from .flow import ConditionalFlow
from .neural_forward import load_trfs, neural_consistency
from .perceived_audio import ASR, FrozenAudio
from .perceived_data import WindowDataset
from .pipeline import audio_window_rows, decode_mel, provenance, require_full_data, require_provenance
from .temporal import EEGEncoder, encoder_loss
from .utils import append_jsonl, atomic_checkpoint, atomic_json, load_checkpoint, read_json, seed_all


def move_batch(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


class BalancedBatches:
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)
        self.groups = {}
        for i, row in enumerate(dataset.rows):
            self.groups.setdefault(row.get("subject", "audio"), {}).setdefault(row["audio"], []).append(i)
        self.people = sorted(self.groups)

    def next(self, count, device):
        indices = []
        for _ in range(count):
            person = self.people[self.rng.integers(len(self.people))]
            clips = sorted(self.groups[person])
            pool = self.groups[person][clips[self.rng.integers(len(clips))]]
            indices.append(pool[self.rng.integers(len(pool))])
        return move_batch(default_collate([self.dataset[i] for i in indices]), device)


class AudioDataset(Dataset):
    def __init__(self, cfg, split, fixed_validation=False):
        self.cfg = cfg
        self.rows = audio_window_rows(cfg, split)
        if fixed_validation:
            from .utils import stable_int
            selected = set()
            for audio in cfg.splits["audio"][split]:
                pool = sorted([r for r in self.rows if r["audio"] == audio], key=lambda r: stable_int(r["key"]))
                selected.update(r["key"] for r in pool[:cfg["training"]["validation_windows_per_audio"]])
            self.rows = [r for r in self.rows if r["key"] in selected]
        self.stats = read_json(cfg.dir("artifacts") / "stats/audio.json")

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(self.cfg.dir("artifacts") / "audio_targets" / f"{row['key']}.npz") as target:
            teacher = (target["teacher"].astype(np.float32) - self.stats["teacher"]["mean"]) / self.stats["teacher"]["std"]
            mel = (target["mel"] - np.array(self.stats["mel"]["mean"])[:, None]) / np.array(self.stats["mel"]["std"])[:, None]
            return {**row, "subject": "audio", "id": row["key"], "teacher": torch.tensor(teacher, dtype=torch.float32), "mel": torch.tensor(mel, dtype=torch.float32), "times": torch.from_numpy(target["times"].copy())}


def build_encoder(cfg, physiological=True):
    manifest = read_json(cfg.dir("artifacts") / "manifest.json")
    groups = manifest["recordings"][0]["montage"]["groups"]
    return EEGEncoder(cfg, groups, physiological)


def learning_rate(peak, step, total, warmup):
    if step < warmup: return peak * (step + 1) / max(warmup, 1)
    phase = (step - warmup) / max(total - warmup, 1)
    return peak * .5 * (1 + math.cos(math.pi * min(phase, 1)))


def amp_context(cfg, device):
    enabled = device.startswith("cuda") and cfg["training"]["amp"] and torch.cuda.is_bf16_supported()
    return torch.autocast("cuda", dtype=torch.bfloat16) if enabled else nullcontext()


def restore_rng(saved, batches):
    if "torch_rng" in saved: torch.set_rng_state(saved["torch_rng"].cpu())
    if "batch_rng" in saved: batches.rng.bit_generator.state = saved["batch_rng"]
    if torch.cuda.is_available() and saved.get("cuda_rng") is not None: torch.cuda.set_rng_state_all(saved["cuda_rng"])


def rng_state(batches):
    return {"torch_rng": torch.get_rng_state(), "batch_rng": batches.rng.bit_generator.state, "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def train_oracle(cfg, seed, device="cpu", resume=False):
    from .evaluation import copy_synthesis_gate, score_model
    seed_all(seed)
    folder = cfg.dir("artifacts") / "oracle" / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    audio = FrozenAudio(cfg, device, teacher=False)
    asr = ASR(cfg, device)
    baseline = copy_synthesis_gate(cfg, audio, asr)
    if not baseline["passed"]: raise ValueError("Native Mel/Vocos copy-synthesis gate failed")
    train, validation = AudioDataset(cfg, "train"), AudioDataset(cfg, "validation", fixed_validation=True)
    batches = BalancedBatches(train, seed)
    flow = ConditionalFlow(cfg).to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=cfg["training"]["oracle_lr"], weight_decay=cfg["training"]["weight_decay"])
    total = cfg["training"]["oracle_steps"]
    start, best_metric = 0, float("inf")
    elapsed_before = 0.0
    if resume and (folder / "latest.pt").exists():
        saved = load_checkpoint(folder / "latest.pt", device); require_provenance(saved, cfg)
        elapsed_before = float(saved.get("elapsed_seconds",0.0))
        flow.load_state_dict(saved["flow"]); optimizer.load_state_dict(saved["optimizer"])
        start, best_metric = saved["step"], saved["best_metric"]
        restore_rng(saved, batches)
    timer = time.perf_counter()
    for step in range(start, total):
        flow.train(); optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(cfg["training"]["oracle_lr"], step, total, cfg["training"]["warmup_steps"])
        for group in optimizer.param_groups: group["lr"] = lr
        batch = batches.next(cfg["training"]["flow_batch"], device)
        loss_value = 0.0
        micro = cfg["training"]["microbatch"]
        for a in range(0, len(batch["mel"]), micro):
            stop = min(a + micro, len(batch["mel"]))
            with amp_context(cfg, device): loss = flow.matching_loss(batch["mel"][a:stop], batch["teacher"][a:stop], cfg["flow"]["null_fraction"]) * (stop-a) / len(batch["mel"])
            loss.backward(); loss_value += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(flow.parameters(), cfg["training"]["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        if (step+1) % 50 == 0: append_jsonl(folder / "training.jsonl", {"step": step+1, "fm": loss_value, "lr": lr})
        if (step+1) % cfg["training"]["validate_every"] == 0 or step+1 == total:
            flow.eval()
            scores = score_model(cfg, validation, flow, audio, asr, encoder=None, controls=["correct"], generation_seeds=[0], training_seed=seed, system="oracle")
            value = scores["wer"]
            checkpoint_value = {"flow": flow.state_dict(), "optimizer": optimizer.state_dict(), "step": step+1, "best_metric": min(best_metric, value), "provenance": provenance(cfg), "elapsed_seconds": elapsed_before+time.perf_counter()-timer, **rng_state(batches)}
            if value < best_metric:
                best_metric = value
                atomic_checkpoint(folder / "best.pt", {**checkpoint_value, "validation": scores, "selected_step": step+1})
            atomic_checkpoint(folder / "latest.pt", checkpoint_value)
            print(f"Oracle seed={seed} step={step+1}: WER={value:.4f}", flush=True)
    saved = load_checkpoint(folder / "best.pt")
    saved.update(completed=True, trained_updates=total, elapsed_seconds=elapsed_before+time.perf_counter()-timer)
    atomic_checkpoint(folder / "best.pt", saved)
    wer = saved["validation"]["wer"]
    gate = {"config_hash": cfg.hash, "passed": wer <= cfg["evaluation"]["oracle_max_wer"] and (wer-baseline["copy_wer"])*100 <= cfg["evaluation"]["oracle_max_wer_delta_pp"], "wer": wer, "copy_wer": baseline["copy_wer"], "selected_step": saved["selected_step"]}
    atomic_json(folder / "gate.json", gate)
    return gate


@torch.no_grad()
def encoder_validation(cfg, encoder, dataset, device):
    from .evaluation import wrong_time_mapping
    from .metrics import bootstrap_mean
    encoder.eval()
    mapping = wrong_time_mapping(dataset.rows, cfg["evaluation"]["wrong_min_seconds"])
    by_person = {}
    for index in range(len(dataset)):
        batch = move_batch(default_collate([dataset[index]]), device)
        wrong = move_batch(default_collate([dataset[mapping[index]]]), device)
        correct = encoder(batch["eeg"], batch["times"], batch["channel_mask"])["teacher"]
        shuffled = encoder(wrong["eeg"], batch["times"], wrong["channel_mask"])["teacher"]
        mse = (correct - batch["teacher"]).square().mean().item()
        wrong_mse = (shuffled - batch["teacher"]).square().mean().item()
        by_person.setdefault(batch["subject"][0], []).append([mse, wrong_mse])
    people = {s: {"mse": float(np.mean(v, axis=0)[0]), "wrong_mse": float(np.mean(v, axis=0)[1])} for s, v in by_person.items()}
    delta = [v["wrong_mse"] - v["mse"] for v in people.values()]
    ci = bootstrap_mean(delta)
    return {"mse": float(np.mean([v["mse"] for v in people.values()])), "subjects": people, "correct_vs_wrong": ci, "passed": ci["lower"] > 0}


def train_encoder(cfg, fold, seed, physiological=True, device="cpu", resume=False):
    require_full_data(cfg); seed_all(seed)
    name = "physiology" if physiological else "baseline"
    folder = cfg.dir("artifacts") / "encoders" / fold / str(seed) / name
    folder.mkdir(parents=True, exist_ok=True)
    train = WindowDataset(cfg, fold, "train")
    validation = WindowDataset(cfg, fold, "validation", fixed_validation=True)
    batches = BalancedBatches(train, seed)
    encoder = build_encoder(cfg, physiological).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=cfg["training"]["encoder_lr"], weight_decay=cfg["training"]["weight_decay"])
    start, best_metric = 0, float("inf")
    elapsed_before = 0.0
    total = cfg["training"]["encoder_steps"]
    if resume and (folder / "latest.pt").exists():
        saved = load_checkpoint(folder / "latest.pt", device); require_provenance(saved, cfg, fold)
        elapsed_before = float(saved.get("elapsed_seconds",0.0))
        encoder.load_state_dict(saved["encoder"]); optimizer.load_state_dict(saved["optimizer"])
        start, best_metric = saved["step"], saved["best_metric"]; restore_rng(saved, batches)
    timer = time.perf_counter()
    for step in range(start, total):
        encoder.train(); optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(cfg["training"]["encoder_lr"], step, total, cfg["training"]["warmup_steps"])
        for group in optimizer.param_groups: group["lr"] = lr
        batch = batches.next(cfg["training"]["encoder_batch"], device)
        output = encoder(batch["eeg"], batch["times"], batch["channel_mask"])
        loss, parts = encoder_loss(output, batch, cfg, physiological)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg["training"]["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        if (step+1) % 50 == 0: append_jsonl(folder / "training.jsonl", {"step": step+1, "loss": float(loss.detach()), **{k: float(v) for k,v in parts.items()}})
        if (step+1) % cfg["training"]["validate_every"] == 0 or step+1 == total:
            scores = encoder_validation(cfg, encoder, validation, device)
            value = scores["mse"]
            saved = {"encoder": encoder.state_dict(), "optimizer": optimizer.state_dict(), "step": step+1, "best_metric": min(best_metric,value), "physiological": physiological, "provenance": provenance(cfg,fold), "elapsed_seconds": elapsed_before+time.perf_counter()-timer, **rng_state(batches)}
            if value < best_metric:
                best_metric=value; atomic_checkpoint(folder / "best.pt", {**saved,"validation":scores,"selected_step":step+1})
            atomic_checkpoint(folder / "latest.pt", saved)
            print(f"Encoder {fold}/{seed}/{name} step={step+1} MSE={value:.4f}", flush=True)
    saved = load_checkpoint(folder / "best.pt")
    saved.update(completed=True, trained_updates=total, elapsed_seconds=elapsed_before+time.perf_counter()-timer)
    atomic_checkpoint(folder / "best.pt", saved)
    atomic_json(folder / "gate.json", {"config_hash":cfg.hash, **saved["validation"]})
    return saved["validation"]


def train_flow(cfg, fold, seed, system, device="cpu", resume=False):
    from .evaluation import score_model
    require_full_data(cfg); seed_all(seed)
    physiological = system != "B0"
    if system not in {"B0","B1","B2"}: raise ValueError(system)
    oracle_folder = cfg.dir("artifacts") / "oracle" / str(seed)
    if not read_json(oracle_folder / "gate.json")["passed"]: raise ValueError("Oracle gate failed")
    enc_folder = cfg.dir("artifacts") / "encoders" / fold / str(seed) / ("physiology" if physiological else "baseline")
    if fold == "F0" and seed == 11 and not read_json(enc_folder / "gate.json")["passed"]:
        raise ValueError("Pilot temporal alignment gate failed")
    if system != "B0" and not read_json(cfg.run_dir("F0",11,"B0") / "gate.json")["passed"]:
        raise ValueError("Pilot EEG conditioning gate failed")
    folder = cfg.run_dir(fold, seed, system); folder.mkdir(parents=True, exist_ok=True)
    train = WindowDataset(cfg,fold,"train"); validation=WindowDataset(cfg,fold,"validation",fixed_validation=True)
    batches=BalancedBatches(train,seed)
    encoder=build_encoder(cfg,physiological).to(device)
    initial=load_checkpoint(enc_folder / "best.pt",device); require_provenance(initial,cfg,fold)
    encoder.load_state_dict(initial["encoder"])
    flow=ConditionalFlow(cfg).to(device)
    oracle=load_checkpoint(oracle_folder / "best.pt",device); require_provenance(oracle,cfg)
    flow.load_state_dict(oracle["flow"])
    flow_optimizer=torch.optim.AdamW(flow.parameters(),lr=cfg["training"]["flow_lr"],weight_decay=cfg["training"]["weight_decay"])
    encoder_optimizer=torch.optim.AdamW(encoder.parameters(),lr=cfg["training"]["joint_encoder_lr"],weight_decay=cfg["training"]["weight_decay"])
    trfs=load_trfs(cfg,fold,device) if system=="B2" else {}
    audio=FrozenAudio(cfg,device,teacher=False); asr=ASR(cfg,device)
    audio_stats=read_json(cfg.dir("artifacts")/"stats/audio.json")
    start,best_metric=0,float("inf")
    elapsed_before=0.0
    total=cfg["training"]["flow_steps"]
    if resume and (folder/"latest.pt").exists():
        saved=load_checkpoint(folder/"latest.pt",device);require_provenance(saved,cfg,fold)
        elapsed_before=float(saved.get("elapsed_seconds",0.0))
        flow.load_state_dict(saved["flow"]);encoder.load_state_dict(saved["encoder"])
        flow_optimizer.load_state_dict(saved["flow_optimizer"]);encoder_optimizer.load_state_dict(saved["encoder_optimizer"])
        start,best_metric=saved["step"],saved["best_metric"];restore_rng(saved,batches)
    timer=time.perf_counter()
    if device.startswith("cuda"):torch.cuda.reset_peak_memory_stats()
    for step in range(start,total):
        joint=step>=cfg["training"]["joint_after"]
        encoder.requires_grad_(joint);encoder.train(joint);flow.train()
        flow_optimizer.zero_grad(set_to_none=True);encoder_optimizer.zero_grad(set_to_none=True)
        for group in flow_optimizer.param_groups:group["lr"]=learning_rate(cfg["training"]["flow_lr"],step,total,cfg["training"]["warmup_steps"])
        if joint:
            for group in encoder_optimizer.param_groups:group["lr"]=learning_rate(cfg["training"]["joint_encoder_lr"],step-cfg["training"]["joint_after"],total-cfg["training"]["joint_after"],cfg["training"]["warmup_steps"])
        batch=batches.next(cfg["training"]["flow_batch"],device)
        output=encoder(batch["eeg"],batch["times"],batch["channel_mask"])
        condition=output["teacher"]
        fm_value=0.0;nc_value=0.0
        micro=cfg["training"]["microbatch"]
        for a in range(0,len(condition),micro):
            stop=min(a+micro,len(condition))
            with amp_context(cfg,device):loss=flow.matching_loss(batch["mel"][a:stop],condition[a:stop],cfg["flow"]["null_fraction"]) * (stop-a)/len(condition)
            loss.backward(retain_graph=joint);fm_value+=float(loss.detach())
        if joint:
            alignment,_=encoder_loss(output,batch,cfg,physiological)
            (cfg["training"]["encoder_joint_weight"]*alignment).backward()
        if system=="B2" and step>=cfg["training"]["joint_after"] and (step+1)%cfg["training"]["neural_every"]==0:
            eligible=[i for i,s in enumerate(batch["subject"]) if s in trfs]
            if eligible:
                i=eligible[step%len(eligible)]
                noise=torch.randn_like(batch["mel"][i:i+1])
                predicted=flow.rollout(noise,condition[i:i+1].detach().float(),cfg["flow"]["neural_steps"],"euler",checkpoint_grad=True)
                wave=decode_mel(predicted,audio_stats["mel"],audio.vocos)
                nc=neural_consistency(wave,batch["neural"][i:i+1],batch["channel_mask"][i:i+1],trfs[batch["subject"][i]])
                ramp=min(1,(step+1-cfg["training"]["joint_after"])/max(1,cfg["training"]["neural_ramp_end"]-cfg["training"]["joint_after"]))
                (cfg["training"]["neural_weight"]*ramp*nc).backward();nc_value=float(nc.detach())
        parameters=list(flow.parameters())+(list(encoder.parameters()) if joint else [])
        torch.nn.utils.clip_grad_norm_(parameters,cfg["training"]["grad_clip"],error_if_nonfinite=True)
        flow_optimizer.step()
        if joint:encoder_optimizer.step()
        if (step+1)%50==0:append_jsonl(folder/"training.jsonl",{"step":step+1,"fm":fm_value,"nc":nc_value,"joint":joint})
        if (step+1)%cfg["training"]["validate_every"]==0 or step+1==total:
            flow.eval();encoder.eval()
            scores=score_model(cfg,validation,flow,audio,asr,encoder,controls=["correct"],generation_seeds=[0],training_seed=seed,system=system)
            value=(scores["wer"],scores["mel_mae"])
            candidate_metric=value[0]+1e-9*value[1]
            saved={"flow":flow.state_dict(),"encoder":encoder.state_dict(),"flow_optimizer":flow_optimizer.state_dict(),"encoder_optimizer":encoder_optimizer.state_dict(),"step":step+1,"best_metric":min(best_metric,candidate_metric),"system":system,"provenance":provenance(cfg,fold),"elapsed_seconds":elapsed_before+time.perf_counter()-timer,**rng_state(batches)}
            if candidate_metric<best_metric:
                best_metric=candidate_metric;atomic_checkpoint(folder/"best.pt",{**saved,"validation":scores,"selected_step":step+1})
            atomic_checkpoint(folder/"latest.pt",saved)
            print(f"Flow {fold}/{seed}/{system} step={step+1} WER={scores['wer']:.4f}",flush=True)
    best=load_checkpoint(folder/"best.pt",device)
    flow.load_state_dict(best["flow"]);encoder.load_state_dict(best["encoder"])
    scores=score_model(cfg,validation,flow,audio,asr,encoder,controls=["correct","wrong","null"],generation_seeds=cfg["evaluation"]["generation_seeds"],training_seed=seed,system=system)
    by_control=scores["control_wer"]
    people=scores["subject_control_wer"]
    majority=sum(v["correct"]<min(v["wrong"],v["null"]) for v in people.values())>len(people)/2
    gate={"config_hash":cfg.hash,"passed":by_control["correct"]<min(by_control["wrong"],by_control["null"]) and majority,"control_wer":by_control,"subjects":people}
    best.update(completed=True,trained_updates=total,elapsed_seconds=elapsed_before+time.perf_counter()-timer,trainable_parameters=sum(p.numel() for p in flow.parameters())+sum(p.numel() for p in encoder.parameters()),peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0)
    atomic_checkpoint(folder/"best.pt",best);atomic_json(folder/"gate.json",gate)
    return gate
