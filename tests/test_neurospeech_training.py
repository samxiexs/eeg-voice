"""Behavioral contracts; tiny deterministic fixtures, no scientific training."""
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from eegspeech.neurospeech.protocol import (oof_groups, main_matrix, assert_scope, freeze_protocol,
                                          scope_manifest, subject_identity, content_identity)
from eegspeech.neurospeech.training import alignment_loss, BalancedBatches, simulation_selection, _fit_encoder
from eegspeech.neurospeech.evaluation import aggregate_rows, wrong_time_mapping, condition_controls
from eegspeech.neurospeech.inference import EEGSpeechGenerator
from eegspeech.neurospeech.adaptation import validate_calibration
from eegspeech.utils import load_checkpoint


class TinyConfig:
    hash='fixture-config'
    def __init__(self,root):
        self.root=Path(root)
        self.values={'training':{'B1_steps':2,'B2_steps':2,'encoder_batch':4,'validate_every':1,'simulation_mode':'off'},
                     'encoder':{},'alignment':{}}
    def __getitem__(self,key):return self.values[key]
    def dir(self,key):return self.root/key


def make_rows():
    return [{'dataset':d,'subject':f'p{p}','audio':f'{d}-a{a}','content_group':f'{d}-c{a}',
             'id':f'{d}:{p}:{a}:{t}','start':t,'recording':f'{d}:{p}:{a}'}
            for d in ('ds004408','etard') for p in range(6) for a in range(6) for t in (0.,16.)]


def test_oof_double_holds_within_formal_training():
    rows=make_rows()
    for fold in oof_groups(rows):
        fit=[rows[i] for i in fold['train_indices']];held=[rows[i] for i in fold['held_indices']]
        assert {subject_identity(r) for r in fit}.isdisjoint(subject_identity(r) for r in held)
        assert {content_identity(r) for r in fit}.isdisjoint(content_identity(r) for r in held)
        assert set(fold['excluded_subjects']).isdisjoint(subject_identity(r) for r in fit)
        assert set(fold['excluded_contents']).isdisjoint(content_identity(r) for r in fit)


def test_p2_has_its_own_lock_and_target_exclusion(tmp_path):
    cfg=TinyConfig(tmp_path)
    assert len(main_matrix())==72
    assert_scope([make_rows()[0]],'P2')
    with pytest.raises(ValueError):assert_scope([{'dataset':'brennan2019','subject':'p1','audio':'a'}],'P2')
    with pytest.raises(ValueError):freeze_protocol(cfg,'P2',{'selection_scope':'target-test'})
    freeze_protocol(cfg,'P2',{'selection_scope':'source-only','simulation':False})
    with pytest.raises(ValueError):freeze_protocol(cfg,'P2',{'selection_scope':'source-only','simulation':True})
    freeze_protocol(cfg,'P1',{'selection_scope':'P1-dev','simulation':True})


def test_wrong_pairing_preserves_stimulus_person():
    rows=make_rows();mapping=wrong_time_mapping(rows)
    for i,j in mapping.items():
        assert rows[i]['recording']==rows[j]['recording']
        assert abs(rows[i]['start']-rows[j]['start'])>=10
    with pytest.raises(ValueError):wrong_time_mapping(rows[:1])


def test_source_balance_and_word_count_aggregation():
    rows=[]
    for subject in ('many1','many2','many3'):
        rows.extend([{'dataset':'A','subject':subject,'system':'S3','control':'correct','training_seed':11,'generation_seed':0,'errors':1,'words':1},
                     {'dataset':'A','subject':subject,'system':'S3','control':'correct','training_seed':11,'generation_seed':0,'errors':0,'words':9}])
    rows.append({'dataset':'B','subject':'single','system':'S3','control':'correct','training_seed':11,'generation_seed':0,'errors':15,'words':10})
    result=aggregate_rows(rows)
    assert result['overall'][0]['wer']==pytest.approx(.8)
    assert next(r for r in result['subjects'] if r['subject']=='single')['wer']==1.5


def test_masked_phoneme_and_detached_variance_loss():
    hidden=torch.randn(2,200,8,requires_grad=True);teacher=torch.randn_like(hidden)
    content=hidden*2;phi=hidden[...,:2]*2;logvar=torch.zeros(2,200,2,requires_grad=True)
    out={'teacher':content,'phi':phi,'phoneme_logits':hidden[...,:3],'log_variance':logvar}
    batch={'teacher':teacher,'phi':torch.randn(2,200,2),'audio':['a','b'],'content_group':['a','b'],'start':[0.,0.],
           'phoneme_targets':torch.ones(2,200,3),'phoneme_mask':torch.zeros(2,200,dtype=torch.bool)}
    loss,parts=alignment_loss(out,batch)
    assert parts['phoneme']==0
    variance_gradient=torch.autograd.grad(parts['variance'],hidden,allow_unused=True,retain_graph=True)[0]
    assert variance_gradient is None
    loss.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert logvar.grad is not None


def test_null_zeros_reliability_and_shuffle_preserves_rhythm():
    correct={k:torch.rand(1,200,2) for k in ('content','rhythm','reliability')};wrong={k:torch.rand(1,200,2) for k in correct}
    null=condition_controls(correct,wrong,'null')
    assert not null['content'].any() and not null['rhythm'].any() and not null['reliability'].any()
    shuffled=condition_controls(correct,wrong,'content-shuffle')
    assert shuffled['rhythm'] is correct['rhythm'] and shuffled['content'] is wrong['content']


def test_eeg_only_generation_strips_target_inputs():
    class Encoder(nn.Module):
        def forward(self,batch,adapter=None):
            assert 'teacher' not in batch and 'mel' not in batch and 'text' not in batch
            return {'teacher':torch.zeros(1,200,768),'phi':torch.zeros(1,200,2),'log_variance':torch.zeros(1,200,2)}
    class Flow(nn.Module):
        def rollout(self,noise,condition,steps,method):
            assert steps==32 and method=='heun'
            return noise
    generator=EEGSpeechGenerator(Encoder(),Flow(),{},system='S2')
    batch={'eeg':torch.zeros(1,1300,3),'positions':torch.zeros(1,3,3),'channel_mask':torch.ones(1,3,dtype=torch.bool),
           'times':torch.arange(200)[None]/50,'teacher':torch.rand(1,200,768),'mel':torch.rand(1,100,376),'text':['target leak']}
    assert torch.equal(generator(batch,seed=0,return_waveform=False),generator(batch,seed=0,return_waveform=False))
    assert not torch.equal(generator(batch,seed=0,return_waveform=False),generator(batch,seed=1,return_waveform=False))


def test_simulation_three_seed_source_only_rule():
    real={s:{'A':1.,'B':1.} for s in (11,22,33)}
    good={s:{'A':.9,'B':.99} for s in real};harm={s:{'A':.5,'B':1.06} for s in real}
    assert simulation_selection(real,good)['passed']
    assert not simulation_selection(real,harm)['passed']
    with pytest.raises(ValueError):simulation_selection({11:real[11]},{11:good[11]})


def test_calibration_context_support_budget():
    class Data:
        rows=[{'dataset':'ds004408','subject':'p1','audio':'a','calibration':{'start':0,'stop':60,'independently_filtered':True}}]
        def __len__(self):return len(self.rows)
    assert validate_calibration(Data())=='ds004408:p1'
    Data.rows.append({'dataset':'ds004408','subject':'p1','audio':'b','calibration':{'start':0,'stop':60,'independently_filtered':True}})
    with pytest.raises(ValueError):validate_calibration(Data())


def test_staged_head_checkpoint_does_not_need_raw(tmp_path,monkeypatch):
    from eegspeech.neurospeech import models,evaluation
    class Encoder(nn.Module):
        def __init__(self):super().__init__();self.content=nn.Linear(3,8)
        def set_stage(self,stage):self.stage=stage
        def forward(self,batch):
            z=self.content(batch['eeg'][:,:200])
            return {'teacher':z,'hidden':z,'phi':z[...,:2],'log_variance':z[...,:2].detach()}
        def adaptation_state_dict(self):return self.state_dict()
    class Dataset:
        rows=make_rows()[:8]
        stats={'teacher':{'mean':[0]*8,'std':[1]*8}}
        def __len__(self):return len(self.rows)
        def __getitem__(self,i):
            return {**self.rows[i],'eeg':torch.ones(1300,3),'positions':torch.zeros(3,3),'channel_mask':torch.ones(3,dtype=torch.bool),
                    'times':torch.arange(200)/50,'teacher':torch.zeros(200,8),'phi':torch.zeros(200,2)}
    cfg=TinyConfig(tmp_path);dataset=Dataset()
    monkeypatch.setattr(models,'build_encoder',lambda cfg,system:Encoder())
    monkeypatch.setattr(evaluation,'encoder_validation',lambda *args:{'mse':.5,'passed':True})
    encoder,saved=_fit_encoder(cfg,dataset,dataset,'P2','external',11,'S0',tmp_path/'run',scope=scope_manifest(cfg,'P2',dataset.rows))
    assert saved['completed'] and saved['trained_updates']==2
    assert load_checkpoint(tmp_path/'run'/'best.pt')['provenance']['protocol']=='P2'
    assert not any(p.name=='raw' for p in tmp_path.rglob('*'))
