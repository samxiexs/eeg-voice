"""Training-side uncertainty supervision and OOF monotone calibration."""
from __future__ import annotations
import torch
from torch import nn


def uncertainty_loss(output, teacher, rhythm, mask=None):
    errors=torch.stack(((output['teacher'].detach()-teacher).square().mean(-1),(output['phi'].detach()-rhythm).square().mean(-1)),-1)
    s=output['log_variance'].clamp(-9.210340372,2.302585093)
    value=.5*(errors*s.neg().exp()+s)
    if mask is not None:
        if mask.ndim==2: mask=mask[...,None].expand_as(value)
        return value[mask.bool()].mean() if mask.any() else value.sum()*0
    return value.mean()


class MonotoneVarianceCalibrator(nn.Module):
    """PAVA isotonic log-variance→error, fitted only on grouped train OOF.

    Neutral gates are used until independent development gate approves calibration.
    Fixed buffer dimensions make serialization independent of number of PAVA blocks.
    """
    def __init__(self,bins=64):
        super().__init__(); self.bins=bins
        self.register_buffer('knots',torch.linspace(-9.21034,2.302585,bins).repeat(2,1))
        self.register_buffer('values',torch.ones(2,bins))
        self.register_buffer('approved',torch.tensor(False))
        self.register_buffer('fitted',torch.tensor(False))

    @torch.no_grad()
    def fit(self,log_variance,errors):
        import numpy as np
        x=log_variance.detach().double().cpu().reshape(-1,2).numpy()
        y=errors.detach().double().cpu().reshape(-1,2).numpy()
        if len(x)<4 or not np.isfinite(x).all() or not np.isfinite(y).all() or (y<0).any(): raise ValueError('Invalid grouped OOF calibration records')
        for d in range(2):
            order=np.argsort(x[:,d]); xx=x[order,d]; yy=y[order,d]
            unique,first,counts=np.unique(xx,return_index=True,return_counts=True)
            sums=np.add.reduceat(yy,first)
            blocks=[]
            for i,(total,n) in enumerate(zip(sums,counts)):
                blocks.append([i,i,float(total),int(n)])
                while len(blocks)>1 and blocks[-2][2]/blocks[-2][3]>blocks[-1][2]/blocks[-1][3]:
                    right=blocks.pop();left=blocks.pop();blocks.append([left[0],right[1],left[2]+right[2],left[3]+right[3]])
            fitted=np.empty(len(unique))
            for start,stop,total,n in blocks: fitted[start:stop+1]=total/n
            knots=np.linspace(unique[0],unique[-1],self.bins)
            vals=np.interp(knots,unique,fitted).clip(1e-4,10)
            self.knots[d].copy_(torch.tensor(knots,device=self.knots.device));self.values[d].copy_(torch.tensor(vals,device=self.values.device))
        self.fitted.fill_(True);self.approved.fill_(False)
        return self

    def forward(self,s):
        if not bool(self.approved):
            # Variance zero gives exactly neutral R=1, independent of error ranking.
            return s*0
        if not bool(self.fitted): raise RuntimeError('Calibration approved without train OOF fit')
        outputs=[]
        for d in range(2):
            knots=self.knots[d].to(s); values=self.values[d].to(s)
            right=torch.searchsorted(knots.contiguous(),s[...,d].contiguous()).clamp(1,self.bins-1);left=right-1
            a=((s[...,d]-knots[left])/(knots[right]-knots[left]).clamp_min(1e-8)).clamp(0,1)
            outputs.append(values[left]*(1-a)+values[right]*a)
        return torch.stack(outputs,-1).clamp(1e-4,10)
