"""Identical differentiable acoustic frontends and genuinely nested TRF fitting."""
from __future__ import annotations
import importlib.metadata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..neural_forward import lag_design, correlations, ridge_fit, circular_null, bh_mask


class AcousticFrontend(nn.Module):
    """Broadband + eight fixed-band 25ms RMS + onset, at exactly 64 Hz.

    Source audio and reconstructed audio MUST use this same frontend. FIR
    boundaries and coefficients are fixed, not fitted from test audio.
    """
    def __init__(self,rate=24000):
        super().__init__()
        from scipy.signal import firwin
        if rate!=24000:raise ValueError('Frontend contract requires 24 kHz')
        self.rate=rate;self.width=600;self.hop=375
        edges=[80,160,320,640,1280,2560,4000,6000,10000]
        bank=np.stack([firwin(513,[lo,hi],pass_zero=False,fs=rate) for lo,hi in zip(edges[:-1],edges[1:])])
        self.register_buffer('filters',torch.tensor(bank,dtype=torch.float32)[:,None])
        self.edges=edges
    def forward(self,wave):
        if wave.ndim==1:wave=wave[None]
        wave=wave.float()
        if wave.shape[-1]<601:raise ValueError('Acoustic crop too short')
        bands=F.conv1d(F.pad(wave[:,None],(256,256),mode='reflect'),self.filters)
        channels=torch.cat((wave[:,None],bands),1)
        frames=F.pad(channels,(300,300),mode='reflect').unfold(-1,self.width,self.hop)
        env=(frames.square().mean(-1)+1e-8).sqrt()[...,:wave.shape[-1]//self.hop].transpose(1,2)
        onset=F.pad(F.relu(env[:,1:,0]-env[:,:-1,0]),(1,0))[...,None]
        return torch.cat((env,onset),-1)
    @torch.no_grad()
    def numpy(self,wave):return self(torch.as_tensor(wave,dtype=torch.float32)).cpu().numpy()


def _fit(chunks,penalty,lags=33,backend='mtrfpy'):
    phi=np.concatenate([c['phi'] for c in chunks]);mean=phi.mean(0);std=phi.std(0).clip(1e-8)
    xs=[lag_design((c['phi']-mean)/std,lags) for c in chunks]
    ys=[np.asarray(c['y'],dtype=float) for c in chunks]
    masks=[np.asarray(c.get('mask',np.ones_like(y,dtype=bool)),dtype=bool) for c,y in zip(chunks,ys)]
    # Reset and mask each concatenation boundary; acoustic response lags cannot
    # traverse independent contents, even if files happen to be contiguous.
    for m in masks:m[:lags-1]=False;m[-(lags-1):]=False
    if backend=='mtrfpy':
        # mTRFpy's API and weights are normalized explicitly to this design:
        # per-trial fitting is legal only with fully observed EEG. Masked EEG
        # uses our declared sample-normalized ridge convention instead.
        try:version=importlib.metadata.version('mtrf')
        except importlib.metadata.PackageNotFoundError:
            try:version=importlib.metadata.version('mtrfpy')
            except importlib.metadata.PackageNotFoundError:raise ImportError('Install the pinned mtrfpy dependency before TRF fitting; no silent backend substitution')
        from mtrf.model import TRF
        # Group channels sharing an identical validity history. A permanently
        # bad sensor cannot remove every fitting observation for the other ones.
        histories=np.concatenate(masks)
        validity_groups={}
        for channel in range(histories.shape[1]):
            key=np.packbits(histories[:,channel]).tobytes()
            validity_groups.setdefault(key,[]).append(channel)
        weights=np.zeros((lags*phi.shape[1]+1,ys[0].shape[1]))
        fitted_groups=0
        for columns in validity_groups.values():
            stimuli=[];responses=[]
            for c,m,y in zip(chunks,masks,ys):
                valid=m[:,columns[0]]
                changes=np.diff(np.r_[False,valid,False].astype(int))
                starts=np.flatnonzero(changes==1);stops=np.flatnonzero(changes==-1)
                for a,b in zip(starts,stops):
                    if b-a>=128:
                        stimuli.append((c['phi'][a:b]-mean)/std)
                        responses.append(y[a:b][:,columns])
            if not stimuli:continue
            model=TRF(direction=1)
            model.train(stimuli,responses,64,0,(lags-1)/64,regularization=penalty)
            raw=np.asarray(model.weights)
            if raw.shape!=(phi.shape[1],lags,len(columns)):
                raise ValueError(f'Unexpected mTRFpy weights shape {raw.shape}')
            # Official predict multiplies BOTH weights and bias by 1/fs.
            weights[:,columns]=np.concatenate((raw.transpose(1,0,2).reshape(lags*phi.shape[1],-1),np.asarray(model.bias).reshape(1,-1)),0)/64.0
            fitted_groups+=1
        if not fitted_groups:raise ValueError('No channels have >=2s contiguous valid TRF fitting segments')
        convention=f'mtrfpy {version} native regularization; exported weights AND bias divided by fs=64'
    elif backend=='sample_normalized_ridge':
        weights=ridge_fit(np.concatenate(xs),np.concatenate(ys),np.concatenate(masks),penalty)
        convention='XTX/n + lambda*I, unpenalized bias'
    else:raise ValueError('Unknown TRF ridge convention')
    return dict(weights=weights,phi_mean=mean,phi_std=std,lags=lags,ridge_convention=convention)


def _predict(model,phi):return lag_design((phi-model['phi_mean'])/model['phi_std'],model['lags'])@model['weights']


def nested_crossfit(chunks,ridge=(.01,.1,1,10,100),lags=33,permutations=500,min_shift_seconds=2.,fdr=.05,max_channels=32,variance_floor=.1,seed=11,backend='mtrfpy'):
    """Chunks carry phi[T,10], y[T,C], mask[T,C], content group.

    Entire content groups remain out of outer fitting AND inner lambda choice.
    With <3 groups caller must prepare guarded train-only 60s blocks explicitly.
    Returns frozen final parameters plus train OOF diagnostics and residuals.
    """
    groups=sorted({str(c['content']) for c in chunks})
    if len(groups)<3:raise ValueError('TRF needs >=3 independent groups or predeclared guarded train blocks')
    oof=[];chosen=[]
    for outer,group in enumerate(groups):
        outer_train=[c for c in chunks if str(c['content'])!=group];outer_test=[c for c in chunks if str(c['content'])==group]
        train_groups=[g for g in groups if g!=group];scores=[]
        for alpha in ridge:
            losses=[]
            for inner in train_groups:
                fit=[c for c in outer_train if str(c['content'])!=inner];validation=[c for c in outer_train if str(c['content'])==inner]
                fitted=_fit(fit,alpha,lags,backend)
                for c in validation:
                    m=np.asarray(c.get('mask',np.ones_like(c['y'],dtype=bool))).copy();m[:lags-1]=False;m[-(lags-1):]=False
                    if m.any():losses.append(float(np.square((_predict(fitted,c['phi'])-c['y'])[m]).mean()))
            scores.append(float(np.mean(losses)) if losses else float('inf'))
        best=float(ridge[int(np.argmin(scores))]);chosen.append(best);fitted=_fit(outer_train,best,lags,backend)
        for c in outer_test:
            mask=np.asarray(c.get('mask',np.ones_like(c['y'],dtype=bool))).copy();mask[:lags-1]=False;mask[-(lags-1):]=False
            oof.append((_predict(fitted,c['phi']),np.asarray(c['y']),mask,str(c['content'])))
    pred=np.concatenate([c[0] for c in oof]);truth=np.concatenate([c[1] for c in oof]);mask=np.concatenate([c[2] for c in oof])
    r=correlations(pred,truth,mask)
    null=circular_null([(c[0],c[1],c[2]) for c in oof],permutations,round(min_shift_seconds*64),seed)
    p=(1+(null>=r).sum(0))/(permutations+1);selected=bh_mask(p,fdr)&(r>0)
    order=np.flatnonzero(selected);order=order[np.argsort(-r[order])[:max_channels]];selected[:]=False;selected[order]=True
    residual=truth-pred;cov=np.eye(truth.shape[1])*variance_floor
    # Estimate covariance only from simultaneous valid selected observations.
    if len(order):
        valid=mask[:,order].all(1);values=residual[valid][:,order]
        if len(values)>len(order):
            from sklearn.covariance import LedoitWolf
            small=LedoitWolf().fit(values).covariance_;v,u=np.linalg.eigh(small);small=(u*np.maximum(v,variance_floor))@u.T
            cov[np.ix_(order,order)]=small
        else:selected[:]=False
    # Final alpha median selected only by TRAIN nested folds, never val/test.
    alpha=min(ridge,key=lambda a:abs(np.log(a)-np.median(np.log(chosen))))
    final=_fit(chunks,alpha,lags,backend)
    final.update(selected=selected,covariance=cov,ridge=float(alpha),crossfit_r=r,pvalues=p,qualified=int(selected.sum())>=8,outer_ridges=chosen,backend=backend)
    final['oof_chunks']=[{'content':g,'residual':y-pr,'mask':m} for pr,y,m,g in oof]
    return final


class FrozenTRF(nn.Module):
    def __init__(self,state):
        super().__init__()
        for k in ('weights','selected','covariance','phi_mean','phi_std'):
            self.register_buffer(k,torch.as_tensor(state[k],dtype=torch.bool if k=='selected' else torch.float32))
        self.lags=int(state.get('lags',33));self.qualified=bool(state.get('qualified',int(self.selected.sum())>=8))
    def forward(self,phi):
        phi=(phi.float()-self.phi_mean)/self.phi_std
        columns=[F.pad(phi.transpose(1,2),(lag,0)).transpose(1,2)[:,:phi.shape[1]] for lag in range(self.lags)]
        design=torch.cat(columns+[torch.ones_like(phi[...,:1])],-1)
        return design@self.weights


def neural_consistency(wave,target_eeg,channel_mask,model,frontend=None,return_coverage=False):
    frontend=AcousticFrontend().to(wave.device) if frontend is None else frontend
    prediction=model(frontend(wave));target_eeg=target_eeg.float()
    if prediction.shape!=target_eeg.shape:raise ValueError('Neural target/acoustic 64 Hz grids disagree')
    values=[];channels=[]
    for b in range(len(wave)):
        valid=channel_mask[b].bool()&model.selected
        count=int(valid.sum());channels.append(count)
        if not model.qualified or count<4:continue
        residual=(prediction[b,39:250]-target_eeg[b,39:250])[:,valid]
        # Invert AFTER selecting covariance intersection, never mask a precision.
        covariance=model.covariance[valid][:,valid]
        precision=torch.linalg.inv(covariance)
        values.append(torch.einsum('ti,ij,tj->t',residual,precision,residual).mean()/count)
    loss=torch.stack(values).mean() if values else prediction.sum()*0
    return (loss,{'active':len(values),'total':len(wave),'channels':channels}) if return_coverage else loss
