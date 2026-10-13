"""One conditional flow with distinct content, rhythm and reliability paths."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ..flow import sinusoidal


def condition_from_output(output,detach=True):
    c={'content':output['teacher'],'rhythm':output['phi'],'reliability':output['reliability'],'times':output['times']}
    return {k:v.detach() if detach else v for k,v in c.items()}


def interpolate(values,times,target):
    if times.ndim==1: times=times[None].expand(len(values),-1)
    rows=[]
    for b in range(len(values)):
        right=torch.searchsorted(times[b].contiguous(),target).clamp(1,len(times[b])-1);left=right-1
        a=((target-times[b,left])/(times[b,right]-times[b,left])).clamp(0,1)
        rows.append(values[b,left]*(1-a[:,None])+values[b,right]*a[:,None])
    return torch.stack(rows)


class FlowBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__()
        self.norms=nn.ModuleList([nn.LayerNorm(width,elementwise_affine=False) for _ in range(3)])
        self.self_attention=nn.MultiheadAttention(width,heads,batch_first=True)
        self.cross_attention=nn.MultiheadAttention(width,heads,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
        self.modulation=nn.Sequential(nn.SiLU(),nn.Linear(width,9*width))
        nn.init.zeros_(self.modulation[-1].weight);nn.init.zeros_(self.modulation[-1].bias)
        self.rhythm=nn.Sequential(nn.Linear(2,width),nn.SiLU(),nn.Linear(width,2*width))
    def forward(self,x,content,rhythm,time):
        shifts=self.modulation(time).chunk(9,-1)
        local_shift,local_scale=self.rhythm(rhythm).chunk(2,-1)
        x=x+local_shift*.1
        def mod(h,i):return self.norms[i](h)*(1+shifts[3*i+1][:,None])+shifts[3*i][:,None]
        h=mod(x,0);x=x+shifts[2][:,None]*self.self_attention(h,h,h,need_weights=False)[0]
        h=mod(x,1);x=x+shifts[5][:,None]*self.cross_attention(h,content,content,need_weights=False)[0]
        h=mod(x,2)*(1+.1*local_scale)
        return x+shifts[8][:,None]*self.ff(h)


class ConditionalFlow(nn.Module):
    def __init__(self,cfg):
        super().__init__();f=cfg['flow'];a=cfg['audio'];self.width=f.get('width',256);self.mel_bins=a.get('mel_bins',100)
        self.hop=a.get('hop_length',256)/a.get('sample_rate',24000)
        self.mel_input=nn.Linear(self.mel_bins,self.width);self.content_input=nn.Linear(a.get('teacher_dim',768),self.width)
        self.time=nn.Sequential(nn.Linear(self.width,self.width),nn.SiLU(),nn.Linear(self.width,self.width))
        self.style_token=nn.Parameter(torch.zeros(128));self.style=nn.Linear(128,self.width)
        self.blocks=nn.ModuleList([FlowBlock(self.width,f.get('heads',4)) for _ in range(f.get('layers',6))])
        self.norm=nn.LayerNorm(self.width);self.output=nn.Linear(self.width,self.mel_bins)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        if sum(p.numel() for p in self.parameters())>15_000_000: raise ValueError('Generator exceeds locked 15M parameter ceiling')

    def freeze_style(self):self.style_token.requires_grad_(False);self.style.requires_grad_(False)

    def forward(self,mel,tau,condition):
        if tau.ndim==0:tau=tau.expand(len(mel))
        content,rhythm,confidence,times=(condition[k] for k in ('content','rhythm','reliability','times'))
        mel_times=torch.arange(mel.shape[-1],device=mel.device,dtype=times.dtype)*self.hop
        x=self.mel_input(mel.transpose(1,2))+sinusoidal(mel_times*50,self.width)[None].to(mel)
        # Multiply after learned projection so zero content remains truly empty.
        cond=self.content_input(content)*confidence[...,:1]
        cond=cond+sinusoidal(times*50,self.width).to(content)
        r=interpolate(rhythm*confidence[...,1:2],times,mel_times)
        time=self.time(sinusoidal(tau*1000,self.width)).to(x)+self.style(self.style_token)[None]
        for block in self.blocks:x=block(x,cond,r,time)
        return self.output(self.norm(x)).transpose(1,2)

    def matching_loss(self,target,condition,null_fraction=.1):
        noise=torch.randn_like(target);tau=torch.rand(len(target),device=target.device)
        c={k:v.detach() for k,v in condition.items()}
        if null_fraction:
            keep=(torch.rand(len(target),device=target.device)>=null_fraction).to(target)[:,None,None]
            c['content']=c['content']*keep;c['rhythm']=c['rhythm']*keep;c['reliability']=c['reliability']*keep
        x=(1-tau[:,None,None])*noise+tau[:,None,None]*target
        return F.mse_loss(self(x,tau,c),target-noise)

    def rollout(self,noise,condition,steps=32,method='heun',checkpoint_grad=False):
        if steps<1 or method not in {'euler','heun'}:raise ValueError('Invalid ODE protocol')
        x=noise.float();dt=1/steps;c={k:v.detach() for k,v in condition.items()}
        def velocity(value,tau):return self(value,tau,c)
        for k in range(steps):
            tau=torch.full((len(x),),k*dt,device=x.device)
            v=checkpoint(velocity,x,tau,use_reentrant=False) if checkpoint_grad else velocity(x,tau)
            candidate=x+dt*v
            if method=='heun':
                v1=checkpoint(velocity,candidate,tau+dt,use_reentrant=False) if checkpoint_grad else velocity(candidate,tau+dt)
                x=x+dt*(v+v1)/2
            else:x=candidate
        return x

    @torch.no_grad()
    def sample(self,condition,frames=376,seed=0,steps=32):
        device=condition['content'].device;generator=torch.Generator(device=device).manual_seed(seed)
        noise=torch.randn((len(condition['content']),self.mel_bins,frames),device=device,generator=generator)
        return self.rollout(noise,condition,steps,'heun')
