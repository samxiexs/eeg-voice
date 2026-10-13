"""Train-only preparation statistics with deliberately distinct synthetic targets."""
import copy
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from eegspeech.inference import core_teacher_times
from eegspeech.perceived_data import WindowDataset,fit_fold_statistics,recording_dir
from eegspeech.pipeline import prepare_audio_targets,require_test_lock
from eegspeech.utils import atomic_json,read_json


def test_audio_and_eeg_statistics_never_fit_heldout_values(cfg,tmp_path):
    cfg=copy.deepcopy(cfg);cfg.root=tmp_path
    raw=cfg.dir("raw")/"stimuli";raw.mkdir(parents=True)
    for i in range(1,21):
        sf.write(raw/f"audio{i:02d}.wav",np.full(30*24000,i,dtype=np.float32),24000,subtype="FLOAT")
    atomic_json(cfg.dir("models")/"manifest.json",{"synthetic_fixture":True})
    class FakeVocoder:
        def features(self,wave):return wave.mean().reshape(1,1,1).expand(1,100,376)
    class FakeAudio:
        device="cpu"
        vocos=FakeVocoder()
        def wavlm(self,wave):return torch.full((1,200,768),float(np.mean(wave))),core_teacher_times()
    prepare_audio_targets(cfg,FakeAudio())
    stats=read_json(cfg.dir("artifacts")/"stats/audio.json")
    np.testing.assert_allclose(stats["teacher"]["mean"],7.5)
    np.testing.assert_allclose(stats["mel"]["mean"],7.5)
    records=[];windows=[];rejections={}
    for subject,clip,value in [("sub-007","audio01",3),("sub-004","audio18",9999)]:
        folder=recording_dir(cfg,subject,clip);folder.mkdir(parents=True)
        np.save(folder/"eeg.npy",np.full((128,30*128),value,dtype=np.float32))
        np.save(folder/"neural.npy",np.full((128,30*64),value,dtype=np.float32))
        records.append({"subject":subject,"audio":clip,"duration":30,"offset":0,"good_channels":[True]*128})
        for a in [10.,14.]:windows.append({"subject":subject,"audio":clip,"start":a,"offset":0,"id":f"{subject}/{clip}_{a:.3f}","key":f"{clip}_{a:.3f}","good_channels":[True]*128})
        rejections[f"{subject}/{clip}"]={"accepted":2,"rejected":0}
    atomic_json(cfg.dir("artifacts")/"manifest.json",{"config_hash":cfg.hash,"recordings":records,"windows":windows,"rejections":rejections,"words":{}})
    fold=fit_fold_statistics(cfg,"F0")
    np.testing.assert_allclose(fold["eeg"]["mean"],3)
    np.testing.assert_allclose(fold["neural"]["mean"],3)
    train=WindowDataset(cfg,"F0","train")
    held=WindowDataset(cfg,"F0","test")
    sample=train[0];assert sample["eeg"].shape==(832,128) and sample["teacher"].shape==(200,768)
    assert sample["mel"].shape==(100,376)
    assert torch.all(sample["eeg"]==0)
    assert held[0]["eeg"].mean()>1e6


def test_final_test_requires_explicit_completed_protocol_lock(cfg,tmp_path):
    cfg=copy.deepcopy(cfg);cfg.root=tmp_path
    import pytest
    with pytest.raises(ValueError,match="locked"):
        require_test_lock(cfg)
