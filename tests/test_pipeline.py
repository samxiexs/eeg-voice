import copy
import inspect
import numpy as np
import pytest
import torch
from eegspeech.flow import ConditionalFlow
from eegspeech.inference import Reconstructor,core_teacher_times
from eegspeech.metrics import aggregate_wer,bootstrap_mean,holm,paired_test,word_errors
from eegspeech.neural_forward import FrozenTRF,bh_mask,circular_null,correlations,lag_design,neural_consistency,ridge_fit
from eegspeech.perceived_audio import acoustic_features,interpolate_features
from eegspeech.perceived_data import qc_window,reference,zero_phase_fir
from eegspeech.perceived_data import calibrated_microvolts
from eegspeech.evaluation import wrong_time_mapping
from eegspeech.temporal import Adapter,EEGEncoder,encoder_loss,temporal_contrastive
from eegspeech.utils import Moments


def tiny(cfg):
    cfg=copy.deepcopy(cfg)
    cfg.values["encoder"].update(width=8,roi_width=4,dilations=[1,2],dropout=0)
    cfg.values["flow"].update(width=16,layers=2,heads=2)
    return cfg


def test_global_content_and_subject_split(cfg):
    for fold,held in cfg.splits["test_folds"].items():
        assert set(held).isdisjoint(cfg.fold_subjects(fold,"train"))
        assert set(cfg.fold_subjects(fold,"validation")).isdisjoint(cfg.fold_subjects(fold,"train"))
    assert len(set(sum(cfg.splits["test_folds"].values(),[])))==16
    cfg.splits["audio"]["test"].append("audio01")
    with pytest.raises(ValueError,match="overlap"):cfg.validate()


def test_training_moments_exclude_bad_channels():
    moments=Moments(2)
    moments.add(np.array([[1,1000],[3,2000]]),np.array([[True,False],[True,False]]))
    moments.add(np.array([[5,2],[7,4]]))
    stats=moments.result()
    np.testing.assert_allclose(stats["mean"],[4,3])
    np.testing.assert_allclose(stats["count"],[4,2])


def test_real_time_interpolation():
    data=torch.tensor([[[0.],[10.],[20.]]])
    out=interpolate_features(data,torch.tensor([0.,1.,2.]),torch.tensor([.25,1.5]))
    torch.testing.assert_close(out,torch.tensor([[[2.5],[15.]]]))


def test_teacher_grid_is_fixed_and_content_free():
    times=core_teacher_times()
    assert len(times)==200
    assert 0<times[0]<.02 and 3.98<times[-1]<4
    torch.testing.assert_close(times[1:]-times[:-1],torch.full((199,),.02),atol=3e-7,rtol=0)
    parameters=inspect.signature(Reconstructor.__call__).parameters
    assert not ({"mel","text","teacher","duration","audio"}&set(parameters))


def test_filter_has_no_group_delay():
    x=np.zeros((1,31));x[0,15]=1
    kernel=np.array([.25,.5,.25])
    result=zero_phase_fir(x,kernel)
    assert result.shape==x.shape
    assert result.argmax()==15
    np.testing.assert_allclose(result[0,14:17],kernel,atol=1e-12)


def test_qc_masks_bads_before_statistics(cfg):
    rng=np.random.default_rng(2)
    x=rng.normal(size=(128,512))*10
    x[0]=0;x[1,0]=1000
    mask,accepted=qc_window(x,np.ones(128,dtype=bool),cfg)
    assert accepted and not mask[0] and not mask[1]
    x[:20]=0
    assert not qc_window(x,np.ones(128,dtype=bool),cfg)[1]


def test_tcn_shapes_and_delay_context(cfg):
    groups=[i%5 for i in range(128)]
    encoder=EEGEncoder(tiny(cfg),groups).eval()
    result=encoder(torch.randn(2,832,128),core_teacher_times(),torch.ones(2,128,dtype=torch.bool))
    assert result["teacher"].shape==(2,200,768)
    assert result["phi"].shape==(2,200,2)
    with pytest.raises(ValueError,match="context"):
        encoder(torch.randn(2,300,128),core_teacher_times())


def test_capacity_matching_default(cfg):
    groups=[i%5 for i in range(128)]
    a=EEGEncoder(cfg,groups,False);b=EEGEncoder(cfg,groups,True)
    na=sum(p.numel() for p in a.parameters());nb=sum(p.numel() for p in b.parameters())
    assert na<1_000_000 and nb<1_000_000 and abs(na-nb)/max(na,nb)<.05


def test_production_flow_stays_within_parameter_budget(cfg):
    model=ConditionalFlow(cfg)
    assert sum(p.numel() for p in model.parameters())<10_000_000


def test_duplicate_audio_is_a_positive_group():
    # No negatives exist: duplicated EEG views of the same audio must have zero loss.
    teacher=torch.randn(1,200,768).expand(2,-1,-1)
    loss=temporal_contrastive(teacher,teacher,["audio01","audio01"],torch.tensor([10.,10.]),core_teacher_times())
    assert abs(float(loss))<1e-5


def test_contrastive_gradients_are_finite():
    target=torch.randn(2,200,768)
    pred=target.clone().requires_grad_(True)
    loss=temporal_contrastive(pred,target,["audio01","audio02"],torch.tensor([10.,20.]),core_teacher_times())
    loss.backward()
    assert torch.isfinite(pred.grad).all()


def test_flow_orientation_and_seed_reproducibility(cfg):
    flow=ConditionalFlow(tiny(cfg)).eval()
    condition=torch.randn(1,200,768)
    first=flow.sample(condition,376,11,2)
    second=flow.sample(condition,376,11,2)
    torch.testing.assert_close(first,second)
    assert first.shape==(1,100,376)
    other=flow.sample(condition,376,22,2)
    assert not torch.equal(first,other)


def test_ode_linear_velocity(cfg):
    flow=ConditionalFlow(tiny(cfg))
    flow.forward=lambda mel,tau,condition:torch.ones_like(mel)*3
    noise=torch.randn(1,100,20);cond=torch.zeros(1,10,768)
    for method in ["euler","heun"]:
        torch.testing.assert_close(flow.rollout(noise,cond,4,method),noise+3)


def test_synthetic_end_to_end_update_and_nc(cfg):
    cfg=tiny(cfg)
    encoder=EEGEncoder(cfg,[i%5 for i in range(128)])
    flow=ConditionalFlow(cfg)
    batch={"eeg":torch.randn(2,832,128),"times":core_teacher_times().repeat(2,1),"teacher":torch.randn(2,200,768),"phi":torch.randn(2,200,2),"audio":["audio01","audio02"],"start":torch.tensor([10.,20.])}
    optimizer=torch.optim.AdamW(list(encoder.parameters())+list(flow.parameters()),lr=.001)
    before=flow.output.weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        output=encoder(batch["eeg"],batch["times"])
        alignment,_=encoder_loss(output,batch,cfg)
        loss=flow.matching_loss(torch.randn(2,100,376),output["teacher"])+.1*alignment
        loss.backward();optimizer.step()
    assert not torch.equal(before,flow.output.weight)
    condition=output["teacher"][:1].detach()
    generated=flow.rollout(torch.randn(1,100,376),condition,2,"euler",True)
    # Synthetic vocoder is used only in unit tests. Real Vocos has a separate check.
    wave=torch.nn.functional.interpolate(generated.mean(1,keepdim=True),size=96000,mode="linear",align_corners=False).squeeze(1)
    model=FrozenTRF({"weights":torch.randn(67,128)*.01,"selected":torch.ones(128,dtype=torch.bool),"variance":torch.ones(128),"phi_mean":torch.zeros(2),"phi_std":torch.ones(2),"lags":33})
    optimizer.zero_grad()
    nc=neural_consistency(wave,torch.randn(1,256,128),torch.ones(1,128,dtype=torch.bool),model)
    nc.backward()
    assert flow.output.weight.grad is not None and torch.isfinite(flow.output.weight.grad).all()
    assert flow.output.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.parameters())


def test_numpy_and_differentiable_trf_match():
    rng=np.random.default_rng(4);phi=rng.normal(size=(200,2)).astype(np.float32)
    weights=rng.normal(size=(67,3)).astype(np.float32)
    model=FrozenTRF({"weights":torch.from_numpy(weights),"selected":torch.ones(3,dtype=torch.bool),"variance":torch.ones(3),"phi_mean":torch.zeros(2),"phi_std":torch.ones(2),"lags":33})
    output=model(torch.from_numpy(phi)[None]).squeeze(0).detach().numpy()
    np.testing.assert_allclose(output,lag_design(phi)@weights,atol=1e-5,rtol=1e-5)


def test_ridge_handles_missing_channels():
    rng=np.random.default_rng(2);x=rng.normal(size=(300,4));x[:,-1]=1
    weights=rng.normal(size=(4,3));y=x@weights
    mask=np.ones_like(y,dtype=bool);mask[:100,0]=False;mask[:,2]=False
    fitted=ridge_fit(x,y,mask,1e-6)
    np.testing.assert_allclose(fitted[:,:2],weights[:,:2],atol=1e-5)
    assert np.all(fitted[:,2]==0)


def test_fft_shift_null_matches_direct_masking():
    rng=np.random.default_rng(10)
    x=rng.normal(size=(1000,3));y=x*.5+rng.normal(size=(1000,3))
    mask=rng.random((1000,3))>.1
    result=circular_null([(x,y,mask)],5,128,seed=11)
    shifts=np.random.default_rng(11).integers(128,1000-128+1,5)
    mask[:33]=False;mask[-33:]=False
    direct=np.stack([correlations(x,np.roll(y,-shift,axis=0),mask&np.roll(mask,-shift,axis=0)) for shift in shifts])
    np.testing.assert_allclose(result,direct,atol=1e-10)


def test_bh_fdr():
    np.testing.assert_array_equal(bh_mask(np.array([.001,.01,.3,.8])),[True,True,False,False])


def test_wrong_controls_keep_subject_and_content_group():
    rows=[{"subject":s,"audio":"audio01","start":a,"key":f"{s}_{a}"} for s in ["s1","s2"] for a in [10.,22.,34.]]
    mapping=wrong_time_mapping(rows)
    for i,j in mapping.items():
        assert rows[i]["subject"]==rows[j]["subject"] and abs(rows[i]["start"]-rows[j]["start"])>=10


def test_adapter_starts_as_identity_and_only_adapter_is_trainable():
    adapter=Adapter();x=torch.randn(2,200,128)
    torch.testing.assert_close(adapter(x),x)
    assert sum(p.numel() for p in adapter.parameters())==4240
    adapter(x).square().mean().backward()
    assert adapter.up.weight.grad.abs().sum()>0


def test_envelope_grid_and_gradient():
    wave=torch.randn(1,96000,requires_grad=True)
    phi=acoustic_features(wave)
    assert phi.shape==(1,256,2)
    phi.square().mean().backward()
    assert torch.isfinite(wave.grad).all() and wave.grad.abs().sum()>0


def test_wer_includes_insertions_and_empty_window_errors():
    assert word_errors("a","a b c")["wer"]==2
    rows=[{"subject":"s","system":"B2","control":"correct","training_seed":11,"generation_seed":0,"errors":2,"words":1},{"subject":"s","system":"B2","control":"correct","training_seed":11,"generation_seed":0,"errors":3,"words":0}]
    assert aggregate_wer(rows)[0]["wer"]==5


def test_independent_unit_is_subject():
    result=paired_test([.1,.2,.3])
    assert result["n"]==3 and result["unit"]=="subject"
    assert result["p_two_sided"]==.25
    np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])


def test_textgrid_window_reference():
    words=[{"start":.1,"end":.3,"word":"one"},{"start":3.9,"end":4.3,"word":"two"}]
    assert reference(words,0)=="one"


def test_unit_restoration_is_source_bound_and_never_guessed(cfg):
    nominal=np.array([2.,3.,4.])
    np.testing.assert_allclose(calibrated_microvolts(nominal,cfg),nominal)
    cfg.values["dataset"]["header_unit_multiplier"]=1.0
    with pytest.raises(ValueError,match="calibration"):
        calibrated_microvolts(nominal,cfg)
