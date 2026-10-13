#!/usr/bin/env python3
"""Small real-model API contract check; does not train or score research data."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for key,folder in {'HF_MODULES_CACHE':'hf_modules','NUMBA_CACHE_DIR':'numba','MPLCONFIGDIR':'matplotlib'}.items():
    path=ROOT/'artifacts/neurospeech_v2/cache'/folder;path.mkdir(parents=True,exist_ok=True);os.environ.setdefault(key,str(path))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import sys
sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoModel
from eegspeech.neurospeech.config import Config
from eegspeech.neurospeech.models import build_encoder
from eegspeech.utils import atomic_json


def main():
    torch.manual_seed(20261013);torch.set_num_threads(2);cfg=Config()
    bank=AutoModel.from_pretrained(cfg.model_dir('reve_positions'),trust_remote_code=True,local_files_only=True)
    positions=bank(['Fz','Cz','T7','T8']).float()[None]
    model=AutoModel.from_pretrained(cfg.model_dir('reve'),trust_remote_code=True,local_files_only=True).eval()
    eeg=torch.randn(1,1300,4)
    with torch.no_grad():tokens=model(eeg.transpose(1,2),positions)
    assert tokens.shape==(1,4,7,512) and torch.isfinite(tokens).all()
    count=sum(p.numel() for p in model.parameters())
    encoder=build_encoder(cfg,'S2',reve_model=model).eval()
    batch={'eeg':eeg,'auditory':eeg,'positions':positions,'channel_mask':torch.ones(1,4,dtype=torch.bool),'times':torch.arange(200)[None]/50}
    with torch.no_grad():out=encoder(batch)
    assert out['teacher'].shape==(1,200,768) and out['phi'].shape==(1,200,2)
    assert all(torch.isfinite(v).all() for v in out.values() if torch.is_tensor(v))
    frozen_trainable=sum(p.numel() for p in encoder.reve.parameters() if p.requires_grad)
    assert frozen_trainable==0
    encoder.set_stage('B2');lora_trainable=sum(p.numel() for p in encoder.reve.parameters() if p.requires_grad)
    assert lora_trainable>0
    result={'status':'passed','scope':'synthetic small-input real-model interface check, no training or scientific evaluation','revisions':{k:v['revision'] for k,v in cfg.models.items()},'reve_parameters':count,'official_output_shape':list(tokens.shape),'aligned_output_shape':list(out['teacher'].shape),'lora_trainable_parameters':lora_trainable,'all_outputs_finite':True,'coordinate_source':'official pinned brain-bzh/reve-positions','coordinate_unit':'metres','weight_bytes':(cfg.model_dir('reve')/'model.safetensors').stat().st_size}
    atomic_json(ROOT/'report/neurospeech_model_interface_check_2026-10-13.json',result)
    print(result)


if __name__=='__main__':main()
