"""Exercise complete training orchestration with explicit synthetic test fixtures."""
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from eegspeech.flow import ConditionalFlow
from eegspeech.neural_forward import FrozenTRF
from eegspeech.temporal import EEGEncoder
from eegspeech.inference import core_teacher_times
from eegspeech.utils import atomic_checkpoint,atomic_json,load_checkpoint


def test_three_training_entrypoints_accumulation_joint_nc_and_resume(cfg,tmp_path,monkeypatch):
    import eegspeech.training as train
    import eegspeech.evaluation as evaluation
    cfg=copy.deepcopy(cfg);cfg.root=tmp_path
    cfg.values["encoder"].update(width=8,roi_width=4,dilations=[1,2],dropout=0)
    cfg.values["flow"].update(width=16,layers=2,heads=2,neural_steps=2)
    cfg.values["training"].update(oracle_steps=4,encoder_steps=4,encoder_batch=4,flow_steps=4,flow_batch=4,microbatch=2,joint_after=1,neural_every=1,neural_ramp_end=3,validate_every=2,warmup_steps=1)
    groups=[i%5 for i in range(128)]
    class ToyDataset(Dataset):
        def __init__(self,*args,**kwargs):
            self.rows=[{"subject":"sub-007","audio":"audio01","start":a,"key":str(a)} for a in [10.,22.,34.]]
            rng=torch.Generator().manual_seed(7)
            self.eeg=torch.randn(3,832,128,generator=rng)
            self.teacher=torch.randn(3,200,768,generator=rng)
            self.phi=torch.randn(3,200,2,generator=rng)
            self.mel=torch.randn(3,100,376,generator=rng)
        def __len__(self):return 3
        def __getitem__(self,i):
            return {**self.rows[i],"eeg":self.eeg[i],"teacher":self.teacher[i],"phi":self.phi[i],"mel":self.mel[i],"times":core_teacher_times(),"channel_mask":torch.ones(128,dtype=torch.bool),"neural":torch.zeros(256,128)}
    class FakeVocoder(torch.nn.Module):
        def forward(self,mel):return torch.nn.functional.interpolate(mel.mean(1,keepdim=True),size=96000,mode="linear",align_corners=False).squeeze(1)
    class FakeAudio:
        def __init__(self,cfg,device,teacher=False):self.vocos=FakeVocoder();self.device=device
    def fake_scores(*args,controls=None,**kwargs):
        result={"correct":.3,"wrong":.6,"null":.7}
        chosen={k:result[k] for k in controls}
        return {"wer":.3,"mel_mae":1.,"control_wer":chosen,"subject_control_wer":{"sub-001":chosen,"sub-002":chosen,"sub-003":chosen},"rows":[]}
    monkeypatch.setattr(train,"WindowDataset",ToyDataset)
    monkeypatch.setattr(train,"AudioDataset",ToyDataset)
    monkeypatch.setattr(train,"build_encoder",lambda cfg,physiological:EEGEncoder(cfg,groups,physiological))
    monkeypatch.setattr(train,"FrozenAudio",FakeAudio)
    monkeypatch.setattr(train,"ASR",lambda *args:None)
    monkeypatch.setattr(train,"require_full_data",lambda *args:None)
    monkeypatch.setattr(train,"provenance",lambda *args:{"test_fixture":True})
    monkeypatch.setattr(train,"require_provenance",lambda *args:None)
    monkeypatch.setattr(evaluation,"score_model",fake_scores)
    monkeypatch.setattr(evaluation,"copy_synthesis_gate",lambda *args:{"passed":True,"copy_wer":.25})
    fake_trf=FrozenTRF({"weights":torch.randn(67,128)*.01,"selected":torch.ones(128,dtype=torch.bool),"variance":torch.ones(128),"phi_mean":torch.zeros(2),"phi_std":torch.ones(2),"lags":33})
    monkeypatch.setattr(train,"load_trfs",lambda *args:{"sub-007":fake_trf})
    atomic_json(cfg.dir("artifacts")/"stats/audio.json",{"mel":{"mean":[0]*100,"std":[1]*100}})
    oracle=cfg.dir("artifacts")/"oracle/11"
    atomic_json(oracle/"gate.json",{"passed":True})
    oracle_gate=train.train_oracle(cfg,11)
    assert oracle_gate["passed"]
    assert load_checkpoint(oracle/"best.pt")["trained_updates"]==4
    for variant in [False,True]:
        folder=cfg.dir("artifacts")/"encoders/F0/11"/("physiology" if variant else "baseline")
        train.train_encoder(cfg,"F0",11,variant)
        assert load_checkpoint(folder/"best.pt")["trained_updates"]==4
        atomic_json(folder/"gate.json",{"passed":True})
    for system in ["B0","B1","B2"]:
        gate=train.train_flow(cfg,"F0",11,system)
        assert gate["passed"]
        state=load_checkpoint(cfg.run_dir("F0",11,system)/"best.pt")
        assert state["completed"] and state["trained_updates"]==4
    # Completed run can resume without performing additional optimization updates.
    train.train_flow(cfg,"F0",11,"B2",resume=True)
    latest=load_checkpoint(cfg.run_dir("F0",11,"B2")/"latest.pt")
    assert latest["step"]==4
