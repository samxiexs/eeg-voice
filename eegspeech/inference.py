"""EEG-only inference: no target audio, text, embeddings or duration input."""
from __future__ import annotations
import torch
from .flow import ConditionalFlow
from .perceived_audio import FrozenVocos
from .pipeline import decode_mel, require_provenance
from .training import build_encoder
from .utils import load_checkpoint, read_json


def core_teacher_times(device="cpu"):
    times=-1+(torch.arange(299,device=device,dtype=torch.float64)*320+(400-1)/2)/16000
    return times[(times>=0)&(times<4)].float()


class Reconstructor:
    def __init__(self,cfg,fold="F0",seed=11,system="B2",device="cpu"):
        self.cfg,self.device=cfg,device
        state=load_checkpoint(cfg.run_dir(fold,seed,system)/"best.pt",device)
        require_provenance(state,cfg,fold)
        self.encoder=build_encoder(cfg,system!="B0").to(device).eval()
        self.encoder.load_state_dict(state["encoder"])
        self.flow=ConditionalFlow(cfg).to(device).eval();self.flow.load_state_dict(state["flow"])
        self.vocos=FrozenVocos(cfg.model_dir("vocos")).to(device)
        self.eeg_stats=read_json(cfg.dir("artifacts")/"stats"/f"{fold}.json")["eeg"]
        self.mel_stats=read_json(cfg.dir("artifacts")/"stats/audio.json")["mel"]

    @torch.no_grad()
    def __call__(self,eeg_uv,channel_mask=None,noise_seed=0,adapter=None):
        """Input: already filtered/CAR EEG128 Hz, [B,832,128] in µV.

        The 6.5s block contains 1s before and 1.5s after the 4s reconstruction.
        Audio output is always four seconds; true utterance length is unused.
        """
        eeg=torch.as_tensor(eeg_uv,dtype=torch.float32,device=self.device)
        if eeg.ndim==2:eeg=eeg[None]
        if eeg.shape[1:]!=(832,128):raise ValueError("Expected [batch,832,128] processed µV EEG")
        if not torch.isfinite(eeg).all():raise ValueError("Nonfinite EEG input")
        mask=torch.ones((len(eeg),128),dtype=torch.bool,device=self.device) if channel_mask is None else torch.as_tensor(channel_mask,dtype=torch.bool,device=self.device).reshape(-1,128).expand(len(eeg),-1)
        mean=torch.tensor(self.eeg_stats["mean"],device=self.device)
        std=torch.tensor(self.eeg_stats["std"],device=self.device)
        normalized=(eeg-mean)/std
        condition=self.encoder(normalized,core_teacher_times(self.device),mask,adapter)["teacher"]
        mel=self.flow.sample(condition,376,noise_seed,self.cfg["flow"]["sample_steps"])
        return decode_mel(mel,self.mel_stats,self.vocos)
