"""Staged NeuroSpeech training with frozen generation conditions and train-only OOF.

No entrypoint downloads or fits models implicitly. Acquisition and processed-data
verification must be completed before any of these explicitly invoked operations.
"""
from __future__ import annotations
from collections import defaultdict
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import default_collate
from ..utils import (atomic_checkpoint, atomic_json, append_jsonl, load_checkpoint,
                     read_json, seed_all, stable_int)
from .protocol import (section, encoder_dir, oracle_dir, run_dir, artifact_root,
                       assert_scope, scope_manifest, provenance, require_checkpoint_scope,
                       row_source, subject_identity, content_identity, oof_groups)
from .inference import condition_from_output, detach_condition, native_decode, eeg_only, load_encoder_state


def move_batch(batch, device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}


def learning_rate(peak,step,total,warmup=1000):
    if step<warmup:return peak*(step+1)/max(1,warmup)
    return peak*.5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total-warmup))))


def mask_mean(values,mask=None):
    if mask is None:return values.mean()
    mask=mask.to(values.dtype)
    while mask.ndim<values.ndim:mask=mask.unsqueeze(-1)
    mask=mask.expand_as(values)
    return (values*mask).sum()/mask.sum().clamp_min(1)


def temporal_contrast(output,batch,temperature=.07):
    """One-second symmetric multi-positive local InfoNCE; repeats are not negatives."""
    prediction,target=output['teacher'],batch['teacher']
    step=50
    embedding=[];teachers=[];anchors=[]
    frame_mask=batch.get('frame_mask', torch.ones(prediction.shape[:2],device=prediction.device,dtype=torch.bool))
    for b in range(len(prediction)):
        content=str(batch.get('content_group',batch['audio'])[b])
        canonical=str(batch.get('canonical_audio',batch['audio'])[b])
        start=float(batch.get('canonical_start',batch['start'])[b])
        for a in range(0,prediction.shape[1]-step+1,step):
            if not bool(frame_mask[b,a:a+step].all()):continue
            embedding.append(prediction[b,a:a+step].mean(0));teachers.append(target[b,a:a+step].mean(0))
            anchors.append((content,canonical,start+a/50))
    if len(embedding)<2:return prediction.sum()*0
    eeg=F.normalize(torch.stack(embedding),dim=-1);audio=F.normalize(torch.stack(teachers),dim=-1)
    scores=eeg@audio.T/temperature
    positive=torch.zeros_like(scores,dtype=torch.bool);allowed=torch.zeros_like(positive)
    for i,(ci,ai,ti) in enumerate(anchors):
        for j,(cj,aj,tj) in enumerate(anchors):
            same=(ai==aj or ci==cj)
            pos=same and abs(ti-tj)<.02
            # Core windows overlap if their anchor time distance is under four seconds.
            neg=not same or abs(ti-tj)>=4.0
            positive[i,j]=pos;allowed[i,j]=pos or neg
    def directional(logits,pos,valid):
        enough=(valid&~pos).any(1)&pos.any(1)
        if not enough.any():return logits.sum()*0
        return (torch.logsumexp(logits.masked_fill(~valid,-torch.inf),1)-
                torch.logsumexp(logits.masked_fill(~pos,-torch.inf),1))[enough].mean()
    return .5*(directional(scores,positive,allowed)+directional(scores.T,positive.T,allowed.T))


def alignment_loss(output,batch,cfg=None,system='S2',contrast=True,uncertainty=True):
    frame_mask=batch.get('frame_mask')
    prediction=output['teacher'];teacher=batch['teacher']
    feature=mask_mean(F.smooth_l1_loss(prediction,teacher,reduction='none'),frame_mask)+.5*mask_mean(1-F.cosine_similarity(prediction,teacher,dim=-1),frame_mask)
    local=temporal_contrast(output,batch) if contrast else feature*0
    rhythm=feature*0;phoneme=feature*0;variance=feature*0
    if system in ('S2','S3'):
        phi=output['phi'];target=batch['phi']
        rhythm=mask_mean((phi-target).square(),frame_mask)
        valid=torch.ones_like(phi[...,0],dtype=torch.bool) if frame_mask is None else frame_mask.bool()
        correlations=[]
        for b in range(len(phi)):
            a,z=phi[b,:,0][valid[b]],target[b,:,0][valid[b]]
            if len(a)>1 and float(z.var(unbiased=False))>1e-6:
                a=a-a.mean();z=z-z.mean()
                correlations.append(1-(a*z).sum()/(a.square().sum().sqrt()*z.square().sum().sqrt()).clamp_min(1e-8))
        if correlations:rhythm=rhythm+torch.stack(correlations).mean()
        if 'phoneme_targets' in batch and 'phoneme_logits' in output:
            targets=batch['phoneme_targets'];logits=output['phoneme_logits']
            phoneme_mask=batch.get('phoneme_mask',frame_mask)
            pos_weight=batch.get('phoneme_pos_weight')
            if pos_weight is not None and pos_weight.ndim>1:pos_weight=pos_weight[0]
            phoneme=mask_mean(F.binary_cross_entropy_with_logits(logits,targets,reduction='none',pos_weight=pos_weight),phoneme_mask)
        if uncertainty and 'log_variance' in output:
            errors=torch.stack(((prediction-teacher).square().mean(-1),(phi-target).square().mean(-1)),dim=-1).detach()
            logvar=output['log_variance'].clamp(math.log(1e-4),math.log(10))
            variance=.5*mask_mean(errors*torch.exp(-logvar)+logvar,frame_mask)
    weights=section(cfg,'alignment') if cfg is not None else {}
    encoder_weights=section(cfg,'encoder') if cfg is not None else {}
    weights={**{k:encoder_weights.get(k+'_weight',v) for k,v in [('feature',1.),('contrast',.1),('rhythm',.2),('phoneme',.05)]},**weights}
    loss=float(weights.get('feature',1.))*feature+float(weights.get('contrast',.1))*local+float(weights.get('rhythm',.2))*rhythm+float(weights.get('phoneme',.05))*phoneme+.05*variance
    return loss,{'feature':feature,'contrast':local,'rhythm':rhythm,'phoneme':phoneme,'variance':variance}


class BalancedBatches:
    """Sample sources equally, then persons and content; variable montage microbatches."""
    def __init__(self,dataset,seed,curriculum=None):
        self.dataset,self.rng,self.curriculum=dataset,np.random.default_rng(seed),curriculum
        self.groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for i,row in enumerate(dataset.rows):self.groups[row_source(row)][str(row['subject'])][content_identity(row)].append(i)
        if not self.groups:raise ValueError('No training samples')
    def indices(self,count,progress=1.):
        result=[];sources=sorted(self.groups)
        for _ in range(count):
            source=sources[int(self.rng.integers(len(sources)))];people=sorted(self.groups[source])
            if self.curriculum is not None:
                order=self.curriculum.orders[source]
                fraction=.25 if progress<1/3 else .5 if progress<2/3 else 1.
                people=order if self.rng.random()<.2 else order[:max(1,math.ceil(len(order)*fraction))]
                people=[p for p in people if p in self.groups[source]]
            if not people:raise ValueError('Curriculum produced empty eligible person set')
            person=people[int(self.rng.integers(len(people)))];contents=sorted(self.groups[source][person])
            content=contents[int(self.rng.integers(len(contents)))];pool=self.groups[source][person][content]
            result.append(pool[int(self.rng.integers(len(pool)))])
        return result
    def samples(self,count,device,progress=1.):
        return [move_batch(default_collate([self.dataset[i]]),device) for i in self.indices(count,progress)]


def concatenate_outputs(outputs,batches):
    output={k:torch.cat([v[k] for v in outputs]) for k in outputs[0] if torch.is_tensor(outputs[0][k]) and all(k in v for v in outputs)}
    keys=('teacher','phi','times','mel','frame_mask','phoneme_targets','phoneme_mask','phoneme_pos_weight')
    batch={k:torch.cat([v[k] for v in batches]) for k in keys if all(k in v for v in batches)}
    for k in ('subject','dataset','audio','start','canonical_start','canonical_audio','content_group','id'):
        if all(k in v for v in batches):
            vals=[v[k] for v in batches]
            batch[k]=torch.cat(vals) if torch.is_tensor(vals[0]) else [x for v in vals for x in v]
    return output,batch


def rng_state(batches):
    return {'torch_rng':torch.get_rng_state(),'batch_rng':batches.rng.bit_generator.state,
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(saved,batches):
    if 'torch_rng' in saved:torch.set_rng_state(saved['torch_rng'].cpu())
    if 'batch_rng' in saved:batches.rng.bit_generator.state=saved['batch_rng']
    if torch.cuda.is_available() and saved.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(saved['cuda_rng'])


def _trainable_state(model):
    # Persist adapters/new modules only; public 69M REVE weights are stored once.
    return {n:v.detach().cpu() for n,v in model.adaptation_state_dict().items()}


def _stage_settings(cfg,stage):
    train=section(cfg,'training');defaults={'A1':(20000,2e-4),'B1':(10000,3e-4),'B2':(30000,3e-4),'C1':(30000,1e-4)}
    steps,lr=defaults[stage]
    spec=train.get(stage,{})
    legacy={'A1':'oracle','B1':'B1','B2':'B2','C1':'flow'}[stage]
    budget=train.get(legacy+'_steps',train.get({'A1':'oracle_steps','B1':'heads_steps','B2':'adaptation_steps','C1':'flow_steps'}[stage],steps))
    rate=train.get(legacy+'_lr',train.get({'A1':'oracle_lr','B1':'heads_lr','B2':'encoder_lr','C1':'flow_lr'}[stage],lr))
    return int(spec.get('updates',spec.get('steps',budget))),float(spec.get('lr',rate))


def _fit_encoder(cfg,train,validation,protocol,fold,seed,system,folder,device='cpu',resume=False,
                 scope=None,allow_support=True):
    from .models import build_encoder
    from .evaluation import encoder_validation
    from .curriculum import SubjectAdversary
    seed_all(seed);assert_scope(train.rows,protocol,'train')
    folder.mkdir(parents=True,exist_ok=True)
    encoder=build_encoder(cfg,system).to(device)
    batches=BalancedBatches(train,seed)
    subject_map={s:sorted(g) for s,g in batches.groups.items()}
    adversary=SubjectAdversary(256,subject_map).to(device) if system in ('S2','S3') and allow_support else None
    train_cfg=section(cfg,'training');effective=int(train_cfg.get('encoder_batch',32))
    validate_every=int(train_cfg.get('validate_every',2000))
    stages=('B1','B2') if system in ('S2','S3') else ('B1',)
    best=float('inf');global_step=0;simulation=None;curriculum=None
    support={'grl_max':.05,'grl_halvings':0,'grl_stopped':False,'simulation':'not-enabled-without-validated-training-artifact'}
    checkpoint=load_checkpoint(folder/'latest.pt',device) if resume and (folder/'latest.pt').exists() else None
    if checkpoint:
        require_checkpoint_scope(checkpoint,cfg,protocol,fold);load_encoder_state(encoder,checkpoint['encoder'])
        global_step=checkpoint.get('global_step',0);best=checkpoint['best_metric'];support=checkpoint.get('support',support)
        restore_rng(checkpoint,batches)
    warmup_error=None;bad_validations=0
    sim_mode=section(cfg,'training').get('simulation_mode','candidate')
    if allow_support and system in ('S2','S3') and sim_mode in ('candidate','retained'):
        trf_folder=folder/'support_trf' if scope and scope.get('excluded_subjects') else artifact_root(cfg,protocol)/'trf'/fold
        if not (trf_folder/'gate.json').exists():
            train_trfs(cfg,protocol,fold,seed,device,dataset=train,folder=trf_folder)
        simulation=TRFSimulationBatches(train,trf_folder,seed)
        support['simulation']=sim_mode
    for stage in stages:
        encoder.set_stage(stage)
        steps,lr=_stage_settings(cfg,stage)
        parameters=[{'params':[p for n,p in encoder.named_parameters() if p.requires_grad and not (n.startswith('reve.model.') and ('.down.' in n or '.up.' in n))], 'base_lr':lr},
                    {'params':[p for n,p in encoder.named_parameters() if p.requires_grad and n.startswith('reve.model.') and ('.down.' in n or '.up.' in n)], 'base_lr':1e-4}]
        if adversary is not None:parameters.append({'params':list(adversary.parameters()),'base_lr':lr})
        parameters=[p for p in parameters if len(p['params'])]
        optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=.01)
        start=0
        if checkpoint and checkpoint.get('stage')==stage:
            start=checkpoint['stage_step'];optimizer.load_state_dict(checkpoint['optimizer'])
            if adversary is not None and checkpoint.get('adversary'):adversary.load_state_dict(checkpoint['adversary'])
        elif checkpoint and checkpoint.get('stage')=='B2' and stage=='B1':continue
        for step in range(start,steps):
            encoder.train();optimizer.zero_grad(set_to_none=True)
            strength=0.
            if stage=='B2' and step>=10000 and adversary is not None and not support['grl_stopped']:
                strength=support['grl_max']*min(1,(step-10000)/10000)
                if curriculum is None:
                    curriculum=_build_curriculum(encoder,train,device)
                    batches.curriculum=curriculum
            for group in optimizer.param_groups:group['lr']=learning_rate(group['base_lr'],step,steps)
            sim_count=int(batches.rng.binomial(effective,.2)) if stage=='B2' and step>=10000 and simulation is not None else 0
            sim_count=min(sim_count,effective-1)
            microbatches=batches.samples(effective-sim_count,device,progress=max(0,(step-10000)/max(1,steps-10000)))
            from torch.utils.checkpoint import checkpoint as activation_checkpoint
            outputs=[activation_checkpoint(encoder,eeg_only(batch),use_reentrant=False) for batch in microbatches]
            combined,batch=concatenate_outputs(outputs,microbatches)
            loss,parts=alignment_loss(combined,batch,cfg,system)
            if strength:
                adversarial=adversary(combined['hidden'],batch['dataset'],batch['subject'],strength,batch.get('frame_mask'))
                loss=loss+adversarial;parts['subject_adversarial']=adversarial
            # Only accept simulated batches with independent train-side residual provenance.
            if sim_count:
                simulated_losses=[]
                for _ in range(sim_count):
                    simulated=simulation.next(device)
                    out=activation_checkpoint(encoder,eeg_only(simulated),use_reentrant=False)
                    simulated_losses.append(mask_mean((out['phi']-simulated['phi']).square(),simulated.get('frame_mask')))
                rhythm=torch.stack(simulated_losses).mean()
                loss=loss+.1*rhythm;parts['simulation_rhythm']=rhythm
            loss.backward();torch.nn.utils.clip_grad_norm_([p for group in parameters for p in group['params']],1,error_if_nonfinite=True);optimizer.step()
            global_step+=1
            if global_step%50==0:append_jsonl(folder/'training.jsonl',{'stage':stage,'step':step+1,'global_step':global_step,'grl_strength':strength,**{k:float(v.detach()) for k,v in parts.items()}})
            if (step+1)%validate_every==0 or step+1==steps:
                scores=encoder_validation(cfg,encoder,validation,device) if validation is not None else {'mse':None,'passed':False,'selection':'fixed-final-update-no-held-OOF-selection'}
                # 'identification' selects on the mean rank of each window's own audio, which a constant output cannot improve.
                value=(scores['identification']['mean_rank_percentile'] if train_cfg.get('selection')=='identification' else scores['mse']) if validation is not None else float(steps-step-1)
                if stage=='B2' and step<10000:warmup_error=value
                if strength and warmup_error is not None:
                    bad_validations=bad_validations+1 if value>warmup_error*1.1 else 0
                    if bad_validations>=2:
                        if support['grl_halvings']==0:support['grl_max']/=2;support['grl_halvings']=1
                        else:support['grl_stopped']=True;batches.curriculum=None
                        bad_validations=0
                saved={'encoder':_trainable_state(encoder),'stage':stage,'stage_step':step+1,'global_step':global_step,
                       'optimizer':optimizer.state_dict(),'adversary':adversary.state_dict() if adversary else None,
                       'stats':train.stats,'support':support,'best_metric':min(best,value),'validation':scores,
                       'provenance':provenance(cfg,protocol,fold,scope),**rng_state(batches)}
                if value<best or validation is None:
                    best=value;atomic_checkpoint(folder/'best.pt',{**saved,'selected_step':global_step})
                atomic_checkpoint(folder/'latest.pt',saved)
        checkpoint=None
    saved=load_checkpoint(folder/'best.pt',device);saved['completed']=True;saved['trained_updates']=global_step
    atomic_checkpoint(folder/'best.pt',saved);atomic_json(folder/'gate.json',{'config_hash':cfg.hash,**saved['validation']})
    load_encoder_state(encoder,saved['encoder'])
    return encoder,saved


def _build_curriculum(encoder,dataset,device):
    from .curriculum import medoid_order,ProgressiveCurriculum
    anchors=defaultdict(lambda:defaultdict(dict))
    was_training=encoder.training;encoder.eval()
    with torch.no_grad():
        for i,row in enumerate(dataset.rows):
            source=row_source(row);person=str(row['subject'])
            key=(content_identity(row),round(float(row['start']),3))
            if len(anchors[source][person])>=32:continue
            batch=move_batch(default_collate([dataset[i]]),device)
            out=encoder(eeg_only(batch))
            anchors[source][person][key]=out['hidden'][0].mean(0).cpu().numpy()
    order={}
    for source,people in anchors.items():
        common=set.intersection(*(set(v) for v in people.values()))
        if not common:
            raise ValueError(f'{source}: curriculum requires same training content/time anchors across persons')
        keys=sorted(common)
        order[source]=medoid_order({p:np.stack([v[k] for k in keys]) for p,v in people.items()})
    encoder.train(was_training)
    return ProgressiveCurriculum(order)


def train_encoder(cfg,protocol='P1',fold='F0',seed=11,system='S2',device='cpu',resume=False):
    from .data import WindowDataset
    if system=='S3':system='S2'
    train=WindowDataset(cfg,protocol=protocol,fold=fold,split='train')
    validation=WindowDataset(cfg,protocol=protocol,fold=fold,split='validation',fixed_validation=True,stats=train.stats)
    scope=scope_manifest(cfg,protocol,train.rows)
    _,saved=_fit_encoder(cfg,train,validation,protocol,fold,seed,system,encoder_dir(cfg,protocol,fold,seed,system),device,resume,scope)
    return saved['validation']


def _calibration_diagnostics(encoder,calibrator,dataset,constant,device='cpu'):
    from scipy.stats import spearmanr
    from .reporting import stratified_paired
    records=[];encoder.eval();calibrator.approved.fill_(True)
    with torch.no_grad():
        for i,row in enumerate(dataset.rows):
            batch=move_batch(default_collate([dataset[i]]),device);out=encoder(eeg_only(batch))
            error=torch.stack(((out['teacher']-batch['teacher']).square().mean(-1),(out['phi']-batch['phi']).square().mean(-1)),-1)
            predicted=calibrator(out['log_variance']).clamp(1e-4,10)
            valid=batch.get('frame_mask',torch.ones(error.shape[:2],device=device,dtype=torch.bool))[0].bool()
            for d in range(2):
                x=predicted[0,valid,d].cpu().numpy();y=error[0,valid,d].cpu().numpy()
                r=float(spearmanr(x,y).statistic) if len(x)>1 and np.std(x)>1e-8 and np.std(y)>1e-8 else 0.
                nll=float(.5*(y/x+np.log(x)).mean());base=float(.5*(y/constant[d]+np.log(constant[d])).mean())
                records.append({'dataset':row_source(row),'subject':row['subject'],'kind':d,'rho':r,'nll':nll,'constant_nll':base})
    by_person=defaultdict(list)
    for row in records:by_person[(row['dataset'],row['subject'],row['kind'])].append(row)
    summaries=[]
    for (source,person,kind),items in by_person.items():
        summaries.append({'dataset':source,'subject':person,'kind':kind,
                          **{k:float(np.mean([r[k] for r in items])) for k in ('rho','nll','constant_nll')}})
    gates={}
    for kind in (0,1):
        groups={s:[r['rho'] for r in summaries if r['kind']==kind and r['dataset']==s] for s in {r['dataset'] for r in summaries}}
        ci=stratified_paired(groups)
        nll=np.mean([np.mean([r['nll'] for r in summaries if r['kind']==kind and r['dataset']==s]) for s in groups])
        base=np.mean([np.mean([r['constant_nll'] for r in summaries if r['kind']==kind and r['dataset']==s]) for s in groups])
        gates[str(kind)]={'spearman':ci,'nll':float(nll),'constant_nll':float(base),'passed':ci['lower']>0 and nll<base}
    passed=all(v['passed'] for v in gates.values());calibrator.approved.fill_(passed)
    return {'passed':passed,'kinds':gates,'subjects':summaries,'selection_scope':'source-only'}


def calibrate_reliability(cfg,protocol='P1',fold='F0',seed=11,device='cpu',resume=False):
    from .data import WindowDataset,fit_statistics
    from .models import build_encoder
    from .reliability import MonotoneVarianceCalibrator
    train=WindowDataset(cfg,protocol=protocol,fold=fold,split='train')
    assert_scope(train.rows,protocol,'train');folder=encoder_dir(cfg,protocol,fold,seed,'S2')
    groups=oof_groups(train.rows);x=[];errors=[];record_ids=[]
    for spec in groups:
        fitrows=[train.rows[i] for i in spec['train_indices']];heldrows=[train.rows[i] for i in spec['held_indices']]
        stats=fit_statistics(cfg,fitrows,protocol,fold,write=False)
        fit=WindowDataset(cfg,protocol=protocol,fold=fold,split='train',rows=fitrows,stats=stats)
        held=WindowDataset(cfg,protocol=protocol,fold=fold,split='train',rows=heldrows,stats=stats)
        scope=scope_manifest(cfg,protocol,fitrows,excluded_subjects=spec['excluded_subjects'],excluded_contents=spec['excluded_contents'])
        replica_folder=encoder_dir(cfg,protocol,fold,seed,'S2',spec['fold'])
        # A fixed final update is selected; double-held OOF rows never select checkpoints.
        encoder,_=_fit_encoder(cfg,fit,None,protocol,fold,seed,'S2',replica_folder,device,resume,scope)
        encoder.eval()
        with torch.no_grad():
            for i,row in enumerate(heldrows):
                batch=move_batch(default_collate([held[i]]),device);out=encoder(eeg_only(batch))
                observed=torch.stack(((out['teacher']-batch['teacher']).square().mean(-1),(out['phi']-batch['phi']).square().mean(-1)),-1)
                mask=batch.get('frame_mask',torch.ones(observed.shape[:2],device=device,dtype=torch.bool)).bool()
                x.append(out['log_variance'][mask].cpu());errors.append(observed[mask].cpu())
                record_ids.append({'id':row.get('id',row.get('key')),'oof_fold':spec['fold'],'subject':subject_identity(row),'content':content_identity(row),'valid_frames':int(mask.sum())})
        # Optimizer resume copies are not needed once this fixed training replica completed.
        if (replica_folder/'latest.pt').exists():(replica_folder/'latest.pt').unlink()
    if not x:raise ValueError('No independent double-held training OOF predictions')
    x=torch.cat(x);errors=torch.cat(errors)
    calibrator=MonotoneVarianceCalibrator().to(device).fit(x.to(device),errors.to(device))
    constant=errors.mean(0).clamp(1e-4,10).numpy()
    encoder=build_encoder(cfg,'S2').to(device)
    best=load_checkpoint(folder/'best.pt',device);require_checkpoint_scope(best,cfg,protocol,fold)
    load_encoder_state(encoder,best['encoder'])
    validation=WindowDataset(cfg,protocol=protocol,fold=fold,split='validation',fixed_validation=True,stats=best['stats'])
    gate=_calibration_diagnostics(encoder,calibrator,validation,constant,device)
    atomic_checkpoint(folder/'reliability.pt',{'calibrator':calibrator.state_dict(),**gate,
                      'provenance':provenance(cfg,protocol,fold),'constant_variance':constant.tolist(),
                      'oof_records':record_ids,'oof_stats_scope':'each replica excludes both held persons and content'})
    # Compact frame errors/variances only. Never persist 768-dimensional predictions.
    atomic_checkpoint(folder/'oof_errors.pt',{'log_variance':x,'errors':errors,'records':record_ids,
                      'provenance':provenance(cfg,protocol,fold)})
    atomic_json(folder/'reliability_gate.json',gate)
    return gate


def train_oracle(cfg,protocol='P1',seed=11,device='cpu',resume=False):
    from .data import AudioDataset
    from .flow import ConditionalFlow
    from .evaluation import copy_synthesis_gate,score_model
    from ..perceived_audio import FrozenAudio,ASR
    seed_all(seed);folder=oracle_dir(cfg,protocol,seed);folder.mkdir(parents=True,exist_ok=True)
    train=AudioDataset(cfg,protocol=protocol,split='train')
    validation=AudioDataset(cfg,protocol=protocol,split='validation',fixed_validation=True,stats=train.stats)
    assert_scope(train.rows,protocol,'train');assert_scope(validation.rows,protocol,'validation')
    audio=FrozenAudio(cfg,device,teacher=False);asr=ASR(cfg,device)
    copy=copy_synthesis_gate(cfg,validation,audio,asr,device);atomic_json(folder/'copy_gate.json',copy)
    if not copy['passed']:raise ValueError('Native Mel→Vocos copy gate failed for at least one source')
    flow=ConditionalFlow(cfg).to(device);batches=BalancedBatches(train,seed)
    steps,lr=_stage_settings(cfg,'A1');optimizer=torch.optim.AdamW(flow.parameters(),lr=lr,weight_decay=.01)
    start=0;best=(float('inf'),float('inf'));train_cfg=section(cfg,'training')
    if resume and (folder/'latest.pt').exists():
        saved=load_checkpoint(folder/'latest.pt',device);require_checkpoint_scope(saved,cfg,protocol)
        flow.load_state_dict(saved['flow']);optimizer.load_state_dict(saved['optimizer']);start=saved['step'];best=tuple(saved['best_metric']);restore_rng(saved,batches)
    for step in range(start,steps):
        flow.train();optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=learning_rate(lr,step,steps)
        effective=int(train_cfg.get('flow_batch',32));total_loss=0.
        for batch in batches.samples(effective,device):
            condition={'content':batch['teacher'],'rhythm':batch['phi'],'reliability':torch.ones_like(batch['phi']),'times':batch['times']}
            loss=flow.matching_loss(batch['mel'],condition)/effective;loss.backward();total_loss+=float(loss.detach())
        torch.nn.utils.clip_grad_norm_(flow.parameters(),1,error_if_nonfinite=True);optimizer.step()
        if (step+1)%50==0:append_jsonl(folder/'training.jsonl',{'step':step+1,'fm':total_loss})
        if (step+1)%int(train_cfg.get('validate_every',2000))==0 or step+1==steps:
            scores=score_model(cfg,validation,flow,audio,asr,system='oracle',seed=seed,device=device)
            value=(scores['wer'],scores['mel_mae'])
            saved={'flow':flow.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,'best_metric':min(best,value),
                   'stats':train.stats,'provenance':provenance(cfg,protocol),'validation':{k:v for k,v in scores.items() if k!='rows'},**rng_state(batches)}
            if value<best:best=value;atomic_checkpoint(folder/'best.pt',saved)
            atomic_checkpoint(folder/'latest.pt',saved)
    saved=load_checkpoint(folder/'best.pt');per_source={r['dataset']:r['wer'] for r in saved['validation']['sources']}
    gate_sources={s:{'wer':wer,'copy_wer':copy['sources'][s]['copy_wer'],
                     'passed':wer<=.4 and wer-copy['sources'][s]['copy_wer']<=.15} for s,wer in per_source.items()}
    gate={'config_hash':cfg.hash,'passed':bool(gate_sources) and all(v['passed'] for v in gate_sources.values()),'sources':gate_sources}
    atomic_json(folder/'gate.json',gate);saved['completed']=True;atomic_checkpoint(folder/'best.pt',saved)
    return gate


def train_trfs(cfg,protocol='P1',fold='F0',seed=11,device='cpu',dataset=None,folder=None):
    """Train nested, subject-specific forward models without touching held-person fits."""
    from .data import WindowDataset
    from .neural import AcousticFrontend,nested_crossfit,FrozenTRF
    from .reporting import stratified_paired
    custom_scope=dataset is not None
    dataset=dataset if dataset is not None else WindowDataset(cfg,protocol=protocol,fold=fold,split='train')
    assert_scope(dataset.rows,protocol,'train')
    folder=Path(folder) if folder else artifact_root(cfg,protocol)/'trf'/fold
    folder.mkdir(parents=True,exist_ok=True)
    frontend=AcousticFrontend().to(device);groups=defaultdict(list)
    for i,row in enumerate(dataset.rows):groups[(row_source(row),str(row['subject']))].append(i)
    records=[];saved_paths={}
    for (source,person),indices in sorted(groups.items()):
        # Full support lets cross-fitted residual replay preserve real temporal correlation.
        chunks=[];retained=[];last={}
        for index in sorted(indices,key=lambda i:(str(dataset.rows[i]['audio']),float(dataset.rows[i]['start']))):
            row=dataset.rows[index];audio=row['audio'];start=float(row['start'])
            if start-last.get(audio,-1e9)<6.5:continue
            sample=dataset[index]
            if 'wave_context' in sample and 'neural_context' in sample:
                wave=sample['wave_context'];target=sample['neural_context'];support='full-6.5s'
            else:
                raise ValueError('TRF fitting needs full 6.5-second processed support for valid ≥2-second shift nulls and simulation')
            phi=frontend(wave.to(device)[None])[0].detach().cpu().numpy()
            y=target.detach().cpu().numpy();mask=np.broadcast_to(sample['channel_mask'].cpu().numpy().astype(bool),y.shape).copy()
            if len(phi)!=len(y):raise ValueError('TRF acoustic/neural time-grid mismatch')
            chunks.append({'phi':phi,'y':y,'mask':mask,'content':content_identity(row),'id':row.get('id',row.get('key')),'support':support})
            retained.append(index);last[audio]=start
        if len({c['content'] for c in chunks})<3:
            # Only guarded NONOVERLAPPING train blocks. No main split changes.
            guarded=[];guarded_indices=[]
            for chunk,index in zip(chunks,retained):
                row=dataset.rows[index];within=float(row['start'])%60
                if within<11 or within+5.5>50:continue
                chunk['content']=f'{content_identity(row)}:train-block:{int(float(row["start"])//60)}'
                guarded.append(chunk);guarded_indices.append(index)
            chunks,retained=guarded,guarded_indices
        if len({c['content'] for c in chunks})<3:
            raise ValueError(f'{source}/{person}: insufficient train-only guarded groups for nested TRF fitting')
        state=nested_crossfit(chunks,seed=seed)
        model=FrozenTRF(state).to(device)
        # Save capped FULL-CONTEXT OOF residual blocks and corresponding source IDs.
        available={str(c['content']):[] for c in chunks}
        for chunk,index in zip(chunks,retained):available[str(chunk['content'])].append((chunk,index))
        residual_refs=[];bytes_used=0
        residual_cap=1_000_000
        for residual in state.pop('oof_chunks'):
            chunk,index=available[str(residual['content'])].pop(0)
            value=np.asarray(residual['residual'],dtype=np.float32)
            if chunk['support']=='full-6.5s' and len(value)==416 and bytes_used+value.nbytes<=residual_cap:
                # Edges remain invalid and are not exposed as a valid rhythm target.
                residual_refs.append({'index':index,'id':chunk['id'],'residual':value,'mask':residual['mask'],'content':chunk['content']})
                bytes_used+=value.nbytes
        state.update(dataset=source,subject=person,provenance=provenance(cfg,protocol,fold,scope_manifest(cfg,protocol,dataset.rows)),
                     residual_refs=residual_refs,stats=dataset.stats)
        path=folder/f'{stable_int(source+":"+person)}.pt';atomic_checkpoint(path,state);saved_paths[(source,person)]=path
        records.append({'dataset':source,'subject':person,'qualified':state['qualified'],'channels':int(state['selected'].sum()),
                        'path':str(path),'residual_bytes':bytes_used,'train_chunks':len(chunks)})
    # Source-side held CONTENT for the SAME training persons; no test-person scorer is fit.
    validation=WindowDataset(cfg,protocol=protocol,fold=fold,split='train_subject_validation',fixed_validation=True,stats=dataset.stats) if not custom_scope else None
    train_people=set(groups);by_person=defaultdict(list)
    for i,row in enumerate(validation.rows if validation is not None else []):
        key=(row_source(row),str(row['subject']))
        if key not in train_people:continue
        sample=validation[i];state=load_checkpoint(saved_paths[key],device);model=FrozenTRF(state).to(device)
        if not model.qualified:continue
        phi=frontend(sample['wave'].to(device)[None]);truth=sample['neural'].cpu().numpy()[39:250]
        full_prediction=model(phi)[0].detach().cpu().numpy()
        prediction=full_prediction[39:250]
        valid=sample['channel_mask'].cpu().numpy().astype(bool)&state['selected']
        if valid.sum()<4:continue
        def correlation(a,b):
            a=a[:,valid]-a[:,valid].mean(0);b=b[:,valid]-b[:,valid].mean(0)
            return float(np.mean((a*b).sum(0)/np.maximum(np.sqrt((a*a).sum(0)*(b*b).sum(0)),1e-8)))
        good=correlation(prediction,truth)
        wrong=correlation(np.roll(full_prediction,128,axis=0)[39:250],truth)
        by_person[key].append((good,wrong))
    gates={}
    for source in sorted({r['dataset'] for r in records}):
        people=[r for r in records if r['dataset']==source];qualified=sum(r['qualified'] for r in people)/len(people)
        validation_values={p:np.mean(v,axis=0).tolist() for (s,p),v in by_person.items() if s==source}
        if validation_values:
            advantages={source:[v[0]-v[1] for v in validation_values.values()]};ci=stratified_paired(advantages)
            median=float(np.median([v[0] for v in validation_values.values()]))
            passed=qualified>=.8 and median>=.05 and ci['lower']>0
        else:ci=None;median=None;passed=False
        gates[source]={'passed':passed,'qualified_fraction':qualified,'median_validation_r':median,'paired_advantage':ci,'subjects':validation_values}
    gate={'protocol':protocol,'config_hash':cfg.hash,'sources':gates,'models':records,
          'passed':bool(gates) and all(v['passed'] for v in gates.values())}
    atomic_json(folder/'gate.json',gate)
    return gate


def load_trfs(cfg,protocol,fold,device='cpu',folder=None,require_source_gate=True):
    from .neural import FrozenTRF
    folder=Path(folder) if folder else artifact_root(cfg,protocol)/'trf'/fold
    gate=read_json(folder/'gate.json');result={}
    if gate['config_hash']!=cfg.hash or gate['protocol']!=protocol:raise ValueError('TRF protocol/configuration mismatch')
    for record in gate['models']:
        if not record['qualified']:continue
        if require_source_gate and not gate['sources'][record['dataset']]['passed']:continue
        state=load_checkpoint(Path(record['path']),device)
        require_checkpoint_scope(state,cfg,protocol,fold)
        result[(record['dataset'],record['subject'])]=FrozenTRF(state).to(device)
    return result


class TRFSimulationBatches:
    """Bounded cross-fitted residual replay with ONLY rhythm supervision."""
    def __init__(self,dataset,folder,seed=11):
        self.dataset=dataset;self.rng=np.random.default_rng(seed);self.records=[];self.cache={};self.cache_bytes=0
        self.rows_by_id={str(row.get('id',row.get('key'))):i for i,row in enumerate(dataset.rows)}
        gate=read_json(Path(folder)/'gate.json')
        for record in gate['models']:
            state=load_checkpoint(Path(record['path']))
            fit_scope=state['provenance']['scope']
            source_ids={subject_identity(r) for r in dataset.rows};content_ids={content_identity(r) for r in dataset.rows}
            if not set(fit_scope['subjects'])<=source_ids or not set(fit_scope['contents'])<=content_ids:
                raise ValueError('Simulation artifact fitted on subjects/content outside current training scope')
            for block in state.get('residual_refs',[]):
                if str(block['id']) in self.rows_by_id:self.records.append((Path(record['path']),block))
        if not self.records:raise ValueError('TRF Simulation requires full-context cross-fitted residual blocks')
    def next(self,device):
        from .neural import FrozenTRF,AcousticFrontend
        from .curriculum import simulate_response
        path,block=self.records[int(self.rng.integers(len(self.records)))]
        state=load_checkpoint(path);model=FrozenTRF(state).to(device)
        sample=self.dataset[self.rows_by_id[str(block['id'])]]
        batch=move_batch(default_collate([sample]),device)
        if 'wave_context' not in batch:raise ValueError('Simulation requires full EEG/audio support')
        phi=AcousticFrontend().to(device)(batch['wave_context'])
        residual=torch.as_tensor(block['residual'],device=device)[None]
        simulated=simulate_response(model,phi,residual)
        # Exact same fixed source-independent auditory filtering as processed reader.
        from .data import auditory_filter
        signal=simulated.detach().cpu().numpy()[0]
        auditory=auditory_filter(signal)
        def normalize(value,kind):
            stats=self.dataset.stats[kind]
            return (value-torch.as_tensor(stats['mean'],device=device))/torch.as_tensor(stats['std'],device=device)
        batch['eeg']=normalize(simulated,'eeg')
        batch['auditory']=normalize(torch.tensor(auditory,device=device,dtype=torch.float32)[None],'auditory')
        good=batch['channel_mask'][:,None,:]
        batch['eeg']=batch['eeg'].masked_fill(~good,0)
        batch['auditory']=batch['auditory'].masked_fill(~good,0)
        batch['simulation']=True
        # Teacher/phoneme targets are removed so an accidental semantic loss fails.
        batch.pop('teacher',None);batch.pop('phoneme_targets',None);batch.pop('phoneme_mask',None)
        return batch


def train_flow(cfg,protocol='P1',fold='F0',seed=11,system='S2',device='cpu',resume=False):
    from .data import WindowDataset
    from .models import build_encoder
    from .flow import ConditionalFlow
    from .reliability import MonotoneVarianceCalibrator
    from .neural import neural_consistency
    from .evaluation import score_model
    from ..perceived_audio import FrozenAudio,ASR
    if system not in ('S0','S1','S2','S3'):raise ValueError(system)
    seed_all(seed)
    oracle_folder=oracle_dir(cfg,protocol,seed)
    if not read_json(oracle_folder/'gate.json')['passed']:raise ValueError('Protocol-specific oracle gate failed')
    enc_folder=encoder_dir(cfg,protocol,fold,seed,system)
    if not read_json(enc_folder/'gate.json')['passed']:raise ValueError('Local EEG alignment gate failed')
    encoder=build_encoder(cfg,'S2' if system=='S3' else system).to(device)
    enc=load_checkpoint(enc_folder/'best.pt',device);require_checkpoint_scope(enc,cfg,protocol,fold)
    load_encoder_state(encoder,enc['encoder']);encoder.set_stage('frozen')
    calibrator=None
    if system in ('S2','S3'):
        saved=load_checkpoint(enc_folder/'reliability.pt',device);require_checkpoint_scope(saved,cfg,protocol,fold)
        if saved['passed']:
            calibrator=MonotoneVarianceCalibrator().to(device);calibrator.load_state_dict(saved['calibrator']);calibrator.eval().requires_grad_(False)
    train=WindowDataset(cfg,protocol=protocol,fold=fold,split='train',stats=enc['stats'])
    validation=WindowDataset(cfg,protocol=protocol,fold=fold,split='validation',fixed_validation=True,stats=enc['stats'])
    assert_scope(train.rows,protocol,'train')
    flow=ConditionalFlow(cfg).to(device)
    oracle=load_checkpoint(oracle_folder/'best.pt',device);require_checkpoint_scope(oracle,cfg,protocol)
    flow.load_state_dict(oracle['flow']);flow.freeze_style()
    optimizer=torch.optim.AdamW([p for p in flow.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    folder=run_dir(cfg,protocol,fold,seed,system);folder.mkdir(parents=True,exist_ok=True)
    trfs=load_trfs(cfg,protocol,fold,device) if system=='S3' else {}
    audio=FrozenAudio(cfg,device,teacher=False);asr=ASR(cfg,device)
    batches=BalancedBatches(train,seed);steps,lr=_stage_settings(cfg,'C1')
    start=0;best=(float('inf'),float('inf'));train_cfg=section(cfg,'training')
    if resume and (folder/'latest.pt').exists():
        saved=load_checkpoint(folder/'latest.pt',device);require_checkpoint_scope(saved,cfg,protocol,fold)
        flow.load_state_dict(saved['flow']);optimizer.load_state_dict(saved['optimizer']);start=saved['step'];best=tuple(saved['best_metric']);restore_rng(saved,batches)
    nc_triggered=0;nc_attempted=0
    for step in range(start,steps):
        flow.train();optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=learning_rate(lr,step,steps)
        effective=int(train_cfg.get('flow_batch',32));value=0.;eligible=[]
        for batch in batches.samples(effective,device):
            with torch.no_grad():out=encoder(eeg_only(batch));condition=detach_condition(condition_from_output(out,calibrator,system))
            loss=flow.matching_loss(batch['mel'],condition)/effective;loss.backward();value+=float(loss.detach())
            key=(batch['dataset'][0],batch['subject'][0])
            if key in trfs:eligible.append((batch,condition,trfs[key]))
        nc_value=0.
        if system=='S3' and step>=5000 and (step+1)%4==0:
            nc_attempted+=1
            if eligible:
                batch,condition,trf=eligible[step%len(eligible)]
                # Pure noise trajectory is independent of the real-Mel FM interpolation.
                noise=torch.randn(1,100,376,device=device)
                generated=flow.rollout(noise,condition,steps=8,method='euler',checkpoint_grad=True)
                waveform=native_decode(generated,train.stats,audio.vocos)
                nc,coverage=neural_consistency(waveform,batch['neural'],batch['channel_mask'],trf,return_coverage=True)
                if coverage['active']:
                    weight=.1*min(1,(step+1-5000)/5000);(weight*nc).backward();nc_value=float(nc.detach());nc_triggered+=1
        torch.nn.utils.clip_grad_norm_([p for p in flow.parameters() if p.requires_grad],1,error_if_nonfinite=True);optimizer.step()
        if any(p.grad is not None for p in encoder.parameters()):raise RuntimeError('Frozen encoder received CFM generation gradients')
        if (step+1)%50==0:append_jsonl(folder/'training.jsonl',{'step':step+1,'fm':value,'neural':nc_value,'nc_triggered':nc_triggered,'nc_attempted':nc_attempted})
        if (step+1)%int(train_cfg.get('validate_every',2000))==0 or step+1==steps:
            scored=score_model(cfg,validation,flow,audio,asr,encoder,calibrator,system,seed,device=device)
            metric=(scored['wer'],scored['mel_mae'])
            saved={'flow':flow.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,'best_metric':min(best,metric),
                   'stats':enc['stats'],'system':system,'encoder_path':str(enc_folder/'best.pt'),'oracle_path':str(oracle_folder/'best.pt'),
                   'validation':{k:v for k,v in scored.items() if k!='rows'},'provenance':provenance(cfg,protocol,fold),
                   'nc_triggered':nc_triggered,'nc_attempted':nc_attempted,**rng_state(batches)}
            if metric<best:best=metric;atomic_checkpoint(folder/'best.pt',saved)
            atomic_checkpoint(folder/'latest.pt',saved)
    best=load_checkpoint(folder/'best.pt',device);flow.load_state_dict(best['flow'])
    scored=score_model(cfg,validation,flow,audio,asr,encoder,calibrator,system,seed,
                       controls=('correct','wrong','null'),device=device)
    from .reporting import paired_comparison
    wrong=paired_comparison(scored['rows'],system,system,'wrong','correct')
    null=paired_comparison(scored['rows'],system,system,'null','correct')
    subjects=aggregate_condition_advantages(scored['subjects'])
    gate={'config_hash':cfg.hash,'passed':wrong['lower']>0 and null['lower']>0 and subjects['majority_positive'],
          'wrong_minus_correct':wrong,'null_minus_correct':null,**subjects}
    atomic_json(folder/'gate.json',gate);best['completed']=True;atomic_checkpoint(folder/'best.pt',best)
    return gate


def aggregate_condition_advantages(subject_rows):
    table=defaultdict(dict)
    for row in subject_rows:table[(row['dataset'],row['subject'])][row['control']]=row['wer']
    values=[v['correct']<v['wrong'] and v['correct']<v['null'] for v in table.values()]
    return {'majority_positive':bool(values) and np.mean(values)>.5,'positive_subjects':sum(values),'subjects':len(values)}


def simulation_selection(real_scores,augmented_scores):
    """Three-seed source-side development rule; noise/test scores are rejected by caller."""
    if set(real_scores)!=set(augmented_scores) or set(real_scores)!={11,22,33}:
        raise ValueError('Simulation decision needs matched three training seeds')
    sources=set.intersection(*[set(real_scores[s]) for s in real_scores])
    if not sources:raise ValueError('No complete development source comparison')
    per_source={source:{'real':float(np.mean([real_scores[s][source] for s in real_scores])),
                        'augmented':float(np.mean([augmented_scores[s][source] for s in augmented_scores]))} for source in sources}
    real=np.mean([v['real'] for v in per_source.values()]);aug=np.mean([v['augmented'] for v in per_source.values()])
    passed=aug<=real*.98 and all(v['augmented']<=v['real']*1.05 for v in per_source.values())
    return {'passed':bool(passed),'sources':per_source,'relative_improvement':float(1-aug/max(real,1e-12)),
            'selection_scope':'source-side-real-development-only'}
