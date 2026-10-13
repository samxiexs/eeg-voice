"""Speech-specific REVE adaptation on real electrode layouts.

Input coordinates are metres in the REVE position-bank / MNE native montage
frame. Acquisition must perform and record any head-to-template transform.
Invalid channels are removed before REVE, whose official API has no pad mask.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from ..temporal import TemporalBlock, Adapter


def time_encoding(times, width):
    f = torch.exp(-math.log(10000) * torch.arange(width // 2, device=times.device) / max(1, width // 2))
    phase = times.float()[..., None] * 50 * f
    return torch.cat((phase.sin(), phase.cos()), -1)


def sample_times(values, times, rate=200, left=1.):
    """values [B,T,D], query seconds relative to the audio core."""
    indices = (times + left) * rate
    if bool((indices < 0).any() or (indices > values.shape[1] - 1).any()):
        raise ValueError('EEG context does not cover requested positive response lags')
    lo = indices.floor().long(); hi = (lo + 1).clamp_max(values.shape[1] - 1)
    b = torch.arange(len(values), device=values.device).reshape((-1,) + (1,) * (lo.ndim - 1))
    a = (indices - lo)[..., None]
    return values[b, lo] * (1 - a) + values[b, hi] * a


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=.05):
        super().__init__(); self.base = base; base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout); self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5)); nn.init.zeros_(self.up.weight)
    def forward(self, x):
        return self.base(x) + self.up(self.down(self.dropout(x))) * self.scale


class ReveBackbone(nn.Module):
    def __init__(self, cfg, model=None, lora=True):
        super().__init__(); e = cfg['encoder']
        if model is None:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(cfg.model_dir('reve'), trust_remote_code=True, local_files_only=True)
        self.model = model; self.model.requires_grad_(False)
        self.patch_size = model.patch_size; self.patch_step = model.patch_size - model.overlap_size
        if self.patch_size != 200 or self.patch_step != 180:
            raise ValueError('REVE checkpoint differs from locked 200/20 patch interface')
        self.width = model.embed_dim; self.lora = bool(lora)
        if lora:
            for attention, _ in model.transformer.layers[-e.get('reve_last_layers',4):]:
                for name in ('to_qkv', 'to_out'):
                    base = getattr(attention, name)
                    setattr(attention,name,LoRALinear(base,e.get('reve_rank',8),e.get('reve_alpha',16),e.get('reve_dropout',.05)))
        self.set_stage('B1')

    def set_stage(self, stage):
        self.model.requires_grad_(False)
        if self.lora and stage == 'B2':
            for module in self.model.modules():
                if isinstance(module, LoRALinear):
                    module.down.requires_grad_(True); module.up.requires_grad_(True)
        self.stage = stage

    def train(self, mode=True):
        super().train(mode)
        # Official REVE has no stochastic backbone dropout. Keep only LoRA active.
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, LoRALinear): module.train(mode and self.stage=='B2')
        return self

    def forward(self, eeg, positions, mask):
        if positions.shape != (len(eeg), eeg.shape[-1], 3): raise ValueError('Invalid coordinate dimensions')
        if not torch.isfinite(positions[mask]).all() or (positions[mask].abs().max() > .5):
            raise ValueError('REVE coordinates must be finite metres in audited montage frame')
        results = []
        # Remove rather than zero-pad: official REVE does not implement masks.
        for b in range(len(eeg)):
            good = mask[b].bool()
            if good.sum() < 4: raise ValueError('REVE requires at least four valid scalp channels')
            tokens = self.model(eeg[b:b+1,:,good].transpose(1,2),positions[b:b+1,good])
            if tokens.ndim != 4: raise ValueError('Expected official REVE [B,C,N_patch,D] output')
            n = tokens.shape[2]
            centers = (torch.arange(n,device=eeg.device)*self.patch_step+(self.patch_size-1)/2)/200-1.
            results.append((tokens.flatten(1,2),positions[b,good].repeat_interleave(n,0)[None],centers.repeat(int(good.sum()))[None]))
        return results


class SpatialTemporalEncoder(nn.Module):
    """Layout-independent scalp pooling; soft priors retain every valid channel."""
    def __init__(self, width=128, physiological=True, dropout=.1):
        super().__init__(); self.physiological = physiological
        # Native MNE standard1020 / official REVE-bank positions: Fz,Cz,T7,T8,POz.
        self.register_buffer('anchors',torch.tensor([[.0003122,.058512,.066462],[.0004009,-.009167,.100244],[-.084161,-.016019,-.009346],[.085080,-.015020,-.009490],[.00021,-.102178,.050608]]))
        self.position = nn.Sequential(nn.Linear(3,32),nn.GELU(),nn.Linear(32,width))
        self.value = nn.Linear(8,width); self.score = nn.Linear(width,8)
        self.prior_strength = nn.Parameter(torch.tensor(.01),requires_grad=physiological)
        self.blocks = nn.ModuleList([TemporalBlock(width,d,dropout) for d in (1,2,4,8,16,32)])
        self.register_buffer('lags',torch.tensor([0,.05,.1,.2,.3,.5] if physiological else [0.]))
        self.lag_logits = nn.Parameter(torch.zeros(len(self.lags)),requires_grad=physiological)

    def forward(self, eeg, times, positions, mask):
        positions=positions.masked_fill(~mask.bool()[...,None],0)
        eeg=eeg.masked_fill(~mask.bool()[:,None],0)
        p = self.position(positions*10)
        scores = self.score(p).transpose(1,2)
        if self.physiological:
            bias = -torch.cdist(positions,self.anchors[None].expand(len(eeg),-1,-1)).square().amin(-1)/(.06**2)
            scores = scores + self.prior_strength * bias[:,None]
        scores = scores.masked_fill(~mask.bool()[:,None],-torch.inf)
        weights = scores.softmax(-1)
        # Eight soft scalp pools preserve complementary topographies without allocating
        # a giant [B,T,C,W] tensor; every valid channel can contribute to each pool.
        pooled = torch.einsum('btc,bkc->btk',eeg,weights)
        x = self.value(pooled)+torch.einsum('bcw,bkc->bkw',p,weights).mean(1)[:,None]
        x = x.transpose(1,2)
        for block in self.blocks: x = block(x)
        x = x.transpose(1,2)
        requested = times[...,None]+self.lags
        return (sample_times(x,requested)*self.lag_logits.softmax(0)[None,None,:,None]).sum(2)


class TemporalReadout(nn.Module):
    def __init__(self, width=256, reve_width=512, auditory_width=128, heads=4, layers=2):
        super().__init__(); self.width = width
        self.query = nn.Linear(auditory_width,width)
        self.tokens = nn.Linear(reve_width,width); self.position = nn.Linear(3,width)
        self.attention = nn.ModuleList([nn.MultiheadAttention(width,heads,batch_first=True) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.gates = nn.ModuleList([nn.Linear(width*2,width) for _ in range(layers)])
    def forward(self, local, times, token_list):
        x = self.query(local)+time_encoding(times,self.width).to(local)
        result=[]
        for b,(tokens,positions,centers) in enumerate(token_list):
            h=x[b:b+1]
            k=self.tokens(tokens)+self.position(positions*10)+time_encoding(centers,self.width).to(tokens)
            # A +/-1.5s local readout; global context already exists inside REVE.
            disallowed=(times[b,:,None]-centers[0,None,:]).abs()>1.5
            if disallowed.all(-1).any(): raise ValueError('No nearby REVE token for time query')
            for attention,norm,gate in zip(self.attention,self.norm,self.gates):
                read=attention(norm(h),k,k,attn_mask=disallowed,need_weights=False)[0]
                h=h+torch.sigmoid(gate(torch.cat((h,read),-1)))*read
            result.append(h)
        return torch.cat(result)


class NeuroSpeechEncoder(nn.Module):
    def __init__(self,cfg,system='S2',reve_model=None):
        super().__init__()
        if system not in {'S0','S1','S2','S3'}: raise ValueError('Unknown encoder system')
        self.cfg,self.system=cfg,system; e=cfg['encoder']; w=e.get('width',128); fw=e.get('fusion_width',256)
        self.physiological=system in {'S2','S3'}
        self.auditory = SpatialTemporalEncoder(w,self.physiological,e.get('dropout',.1)) if system!='S1' else None
        self.reve = ReveBackbone(cfg,reve_model,lora=self.physiological) if system!='S0' else None
        self.readout = TemporalReadout(fw,self.reve.width,w) if self.reve is not None else nn.Linear(w,fw)
        self.query_token = nn.Parameter(torch.zeros(w)) if system=='S1' else None
        self.content=nn.Linear(fw,cfg['audio'].get('teacher_dim',768)); self.acoustics=nn.Linear(fw,2)
        self.phoneme=nn.Linear(fw,e.get('phoneme_dim',19))
        self.variance=nn.Sequential(nn.Linear(fw,64),nn.GELU(),nn.Linear(64,2))
        from .reliability import MonotoneVarianceCalibrator
        self.calibrator=MonotoneVarianceCalibrator()
        self.set_stage('B1')

    def set_stage(self,stage):
        if stage not in {'B1','B2','frozen'}: raise ValueError('Invalid encoder stage')
        self.requires_grad_(stage!='frozen')
        if self.reve is not None: self.reve.set_stage(stage if self.physiological else 'B1')
        self.stage=stage
        if stage=='frozen': self.eval()

    def hidden(self,batch):
        eeg=batch['eeg']; positions=batch['positions']; mask=batch['channel_mask']; times=batch['times']
        if positions.ndim==2: positions=positions[None].expand(len(eeg),-1,-1)
        if mask.ndim==1: mask=mask[None].expand(len(eeg),-1)
        if times.ndim==1: times=times[None].expand(len(eeg),-1)
        if self.auditory is None: local=self.query_token[None,None].expand(len(eeg),times.shape[1],-1)
        else:
            signal=batch.get('auditory') if self.physiological else eeg
            if signal is None: raise ValueError('Auditory branch requires audited 1–8 Hz input')
            local=self.auditory(signal,times,positions,mask)
        if self.reve is None: return self.readout(local)
        return self.readout(local,times,self.reve(eeg,positions,mask))

    def heads(self,hidden):
        logvar=self.variance(hidden.detach()).clamp(math.log(1e-4),math.log(10))
        variance=self.calibrator(logvar)
        reliability=1/(1+variance) if self.physiological else torch.ones_like(variance)
        return {'teacher':self.content(hidden),'phi':self.acoustics(hidden),'phoneme_logits':self.phoneme(hidden),'hidden':hidden,'log_variance':logvar,'reliability':reliability}

    def forward(self,batch,adapter=None):
        h=self.hidden(batch)
        if adapter is not None: h=adapter(h)
        out=self.heads(h); out['times']=batch['times']
        if not self.physiological: out['phi']=out['phi']*0
        return out

    def adaptation_state_dict(self):
        """Checkpoint only additions and LoRA; never duplicate frozen 69M weights."""
        return {k:v for k,v in self.state_dict().items() if not k.startswith('reve.model.') or '.down.' in k or '.up.' in k}


def build_encoder(cfg,system='S2',reve_model=None): return NeuroSpeechEncoder(cfg,system,reve_model)
