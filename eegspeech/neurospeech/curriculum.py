"""Training-only centered-progressive subject curriculum and GRL support."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,strength):ctx.strength=strength;return value.view_as(value)
    @staticmethod
    def backward(ctx,gradient):return -ctx.strength*gradient,None


def gradient_reverse(value,strength):return _Reverse.apply(value,float(strength))


class SubjectAdversary(nn.Module):
    def __init__(self,width,subjects_by_dataset):
        super().__init__();self.lookup={d:{s:i for i,s in enumerate(sorted(subjects))} for d,subjects in subjects_by_dataset.items()}
        self.keys={d:f'dataset_{i}' for i,d in enumerate(sorted(self.lookup))}
        self.classifiers=nn.ModuleDict({self.keys[d]:nn.Sequential(nn.Linear(width,128),nn.GELU(),nn.Linear(128,len(v))) for d,v in self.lookup.items()})
    def forward(self,hidden,datasets,subject_ids,strength,frame_mask=None):
        if frame_mask is None:h=hidden.mean(1)
        else:h=(hidden*frame_mask[...,None]).sum(1)/frame_mask.sum(1).clamp_min(1)[...,None]
        h=gradient_reverse(h,strength);losses=[]
        for dataset,lookup in self.lookup.items():
            ix=[i for i,d in enumerate(datasets) if d==dataset]
            if ix and len(lookup)>1:
                if any(subject_ids[i] not in lookup for i in ix):raise ValueError('Subject classifier received unseen subject')
                labels=torch.tensor([lookup[subject_ids[i]] for i in ix],device=h.device)
                losses.append(F.cross_entropy(self.classifiers[self.keys[dataset]](h[ix]),labels))
        return torch.stack(losses).mean() if losses else h.sum()*0


def medoid_order(anchors):
    """Each subject must provide the SAME ordered stimulus/time anchor vectors.

    Callers must intersect canonical anchor IDs first; mismatched shapes fail.
    """
    names=sorted(anchors)
    if not names:raise ValueError('No training subjects to define center')
    values=[np.asarray(anchors[s],dtype=float) for s in names]
    if any(x.shape!=values[0].shape for x in values):raise ValueError('Subject anchors do not share stimulus/time support')
    x=np.stack(values)
    if not np.isfinite(x).all():raise ValueError('Nonfinite curriculum representations')
    difference=x[:,None]-x[None]
    distance=np.sqrt(np.square(difference).mean(axis=tuple(range(2,difference.ndim))))
    center=int(distance.mean(1).argmin());order=np.argsort(distance[center],kind='stable')
    return {'center':names[center],'subjects':[names[i] for i in order],'distances':[float(distance[center,i]) for i in order]}


class ProgressiveCurriculum:
    def __init__(self,orders,seed=11,warmup=10000,ramp=10000,total=30000,replay=.2):
        self.orders={d:(v['subjects'] if isinstance(v,dict) else list(v)) for d,v in orders.items()};self.rng=np.random.default_rng(seed)
        self.warmup,self.ramp,self.total,self.replay=warmup,ramp,total,replay
    def fraction(self,step):
        if step<self.warmup:return 1.
        progress=(step-self.warmup)/max(1,self.total-self.warmup)
        return .25 if progress<1/3 else .5 if progress<2/3 else 1.
    def subject(self,dataset,step):
        order=self.orders[dataset]
        count=max(1,math.ceil(len(order)*self.fraction(step)))
        pool=order if self.rng.random()<self.replay else order[:count]
        return str(self.rng.choice(pool))
    def grl_strength(self,step,maximum=.05):return maximum*min(1.,max(0.,(step-self.warmup)/self.ramp))


class GRLGuard:
    """One predeclared halving, then stop after two consecutive >=10% harms."""
    def __init__(self,baseline):self.baseline=float(baseline);self.maximum=.05;self.strikes=0;self.halved=False;self.stopped=False
    def update(self,error):
        self.strikes=self.strikes+1 if error>self.baseline*1.1 else 0
        if self.strikes>=2:
            if self.halved:self.maximum=0.;self.stopped=True
            else:self.maximum=.025;self.halved=True
            self.strikes=0
        return self.maximum


def simulate_response(model,phi_context,residual_context,*,generator=None,gain_range=(1.,1.),max_delay=0):
    """Full-context TRF simulation; residuals must come from train OOF blocks.

    No synthetic sample can supervise content/phonemes/InfoNCE. The training
    orchestrator enforces that rule. Ranges must be estimated in TRAIN sources.
    """
    if phi_context.shape[1]<416:raise ValueError('Simulation needs full 6.5s support, not zero-padded 4s core')
    clean=model(phi_context)
    if clean.shape!=residual_context.shape:raise ValueError('Residual blocks require exact montage/time support')
    gain=torch.empty((len(clean),1,1),device=clean.device).uniform_(*gain_range,generator=generator)
    if max_delay:
        if max_delay<0:raise ValueError('Delay perturbation bound must be nonnegative')
        delay=int(torch.randint(max_delay+1,(1,),device=clean.device,generator=generator))
        clean=F.pad(clean.transpose(1,2),(delay,0)).transpose(1,2)[:,:clean.shape[1]]
    response=gain*clean+residual_context
    eeg=F.interpolate(response.transpose(1,2),size=1300,mode='linear',align_corners=False).transpose(1,2)
    # Full context and training-only residual blocks are returned for the source
    # preprocessing filter to derive the exact same 1–8 Hz auditory signal.
    return eeg
