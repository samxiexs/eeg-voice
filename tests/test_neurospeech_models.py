"""Scientific contract tests: signed lags, masking, uncertainty, gradients."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from eegspeech.neurospeech.models import ReveBackbone, LoRALinear, sample_times, build_encoder
from eegspeech.neurospeech.flow import ConditionalFlow
from eegspeech.neurospeech.neural import AcousticFrontend, FrozenTRF, neural_consistency, _fit, _predict
from eegspeech.neurospeech.reliability import MonotoneVarianceCalibrator, uncertainty_loss
from eegspeech.neurospeech.curriculum import gradient_reverse, medoid_order, ProgressiveCurriculum


class FakeReve(nn.Module):
    def __init__(self):
        super().__init__();self.patch_size=200;self.overlap_size=20;self.embed_dim=16
        self.transformer=nn.Module();self.transformer.layers=nn.ModuleList()
        for _ in range(6):
            attention=nn.Module();attention.to_qkv=nn.Linear(16,48,bias=False);attention.to_out=nn.Linear(16,16,bias=False)
            self.transformer.layers.append(nn.ModuleList([attention,nn.Linear(16,16)]))
        self.channel_counts=[]
    def forward(self,eeg,pos):
        self.channel_counts.append(eeg.shape[1])
        # Nonlinear cross-channel information lets padding pollution be detected.
        patches=eeg.unfold(-1,200,180).mean(-1)[...,None].expand(-1,-1,-1,16)
        return patches+pos.square().sum(-1)[:,:,None,None]


def cfg():return {'encoder':{'width':16,'fusion_width':16,'reve_rank':2,'phoneme_dim':19},'audio':{'teacher_dim':8,'mel_bins':4},'flow':{'width':16,'layers':2,'heads':4}}


def test_signed_response_lags():
    x=torch.arange(1300,dtype=torch.float32)[None,:,None]
    query=torch.tensor([[[0.,.5],[1.,1.5]]])
    value=sample_times(x,query)
    assert torch.allclose(value[...,0],torch.tensor([[[200.,300.],[400.,500.]]]))
    with pytest.raises(ValueError):sample_times(x,torch.tensor([[6.]]))


def test_reve_bad_channels_removed_and_exact_lora_boundary():
    model=FakeReve();backbone=ReveBackbone(cfg(),model)
    masks=torch.tensor([[1,1,1,1,0],[1,1,1,1,1]],dtype=torch.bool)
    x=torch.randn(2,1300,5);positions=torch.randn(2,5,3)*.04
    x[0,:,4]=float('nan');positions[0,4]=float('nan')
    result=backbone(x,positions,masks)
    assert model.channel_counts==[4,5]
    assert result[0][0].shape==(1,28,16)
    assert torch.isfinite(result[0][0]).all()
    assert all(not p.requires_grad for p in backbone.parameters())
    backbone.set_stage('B2')
    trainable=[name for name,p in backbone.named_parameters() if p.requires_grad]
    assert len(trainable)==16 and all('.down.' in n or '.up.' in n for n in trainable)
    assert not isinstance(model.transformer.layers[1][0].to_qkv,LoRALinear)
    assert isinstance(model.transformer.layers[2][0].to_qkv,LoRALinear)


def test_uncertainty_detaches_mean_and_monotone_calibration():
    mean=torch.randn(1,8,3,requires_grad=True);rhythm=torch.randn(1,8,2,requires_grad=True);s=torch.zeros(1,8,2,requires_grad=True)
    uncertainty_loss({'teacher':mean,'phi':rhythm,'log_variance':s},torch.zeros_like(mean),torch.zeros_like(rhythm)).backward()
    assert mean.grad is None and rhythm.grad is None and s.grad.abs().sum()>0
    calibrator=MonotoneVarianceCalibrator()
    x=torch.linspace(-3,2,100)[:,None].repeat(1,2);errors=torch.exp(x).clamp_max(10)
    calibrator.fit(x,errors)
    assert torch.equal(calibrator(x),torch.zeros_like(x))
    calibrator.approved.fill_(True);variance=calibrator(x)
    assert ((variance[1:]-variance[:-1])>=-1e-7).all()
    loaded=MonotoneVarianceCalibrator();loaded.load_state_dict(calibrator.state_dict())
    assert torch.allclose(loaded(x),variance)


def test_full_noise_rollout_gradient_conditions_frozen():
    model=ConditionalFlow(cfg())
    # Output zero-init creates first-step gradient only for output; verify chain
    # with trained-like tiny output perturbation so every conditioning path acts.
    nn.init.normal_(model.output.weight,std=.01)
    c={'content':torch.randn(1,20,8,requires_grad=True),'rhythm':torch.randn(1,20,2,requires_grad=True),'reliability':torch.ones(1,20,2),'times':torch.arange(20)[None]/50}
    noise=torch.randn(1,4,38);x=model.rollout(noise,c,steps=3,method='euler',checkpoint_grad=True)
    x.square().mean().backward()
    assert model.output.weight.grad.abs().sum()>0
    assert c['content'].grad is None and c['rhythm'].grad is None
    model.freeze_style();assert not model.style_token.requires_grad


def test_frontend_numeric_identity_and_gradient():
    frontend=AcousticFrontend();wave=torch.randn(1,24000,requires_grad=True)*.01
    result=frontend(wave);assert result.shape==(1,64,10)
    numpy=frontend.numpy(wave.detach().numpy());assert np.allclose(result.detach().numpy(),numpy)
    grad=torch.autograd.grad(result.square().mean(),wave)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0


def test_covariance_mask_inverse_order():
    C=5;cov=torch.eye(C);cov[0,4]=cov[4,0]=.8
    trf=FrozenTRF({'weights':torch.zeros(331,C),'selected':torch.ones(C,dtype=torch.bool),'covariance':cov,'phi_mean':torch.zeros(10),'phi_std':torch.ones(10),'qualified':True})
    wave=torch.zeros(1,96000);truth=torch.zeros(1,256,C);truth[:,:,0]=1
    mask=torch.tensor([[1,1,1,1,0]],dtype=torch.bool)
    value=neural_consistency(wave,truth,mask,trf)
    assert torch.allclose(value,torch.tensor(.25),atol=1e-6)
    mask[0,3]=False
    assert neural_consistency(wave,truth,mask,trf)==0


def test_mtrf_export_matches_official_predict():
    pytest.importorskip('mtrf')
    from mtrf.model import TRF
    rng=np.random.default_rng(5);phi=rng.normal(size=(600,2));y=np.roll(phi,3,axis=0)@rng.normal(size=(2,4))+.02*rng.normal(size=(600,4))
    result=_fit([{'phi':phi,'y':y,'mask':np.ones_like(y,dtype=bool)}],.1,lags=9)
    # _fit trims both boundaries by eight before official fitting.
    official=TRF(direction=1);standard=(phi-result['phi_mean'])/result['phi_std']
    official.train([standard[8:-8]],[y[8:-8]],64,0,8/64,.1)
    pred=official.predict([standard])[0]
    assert np.allclose(_predict(result,phi),pred,atol=1e-10)


def test_grl_and_same_anchor_medoid():
    x=torch.tensor([1.,2.],requires_grad=True);gradient_reverse(x,.05).sum().backward()
    assert torch.allclose(x.grad,torch.tensor([-.05,-.05]))
    order=medoid_order({'A':np.zeros((3,2)),'B':np.ones((3,2)),'C':np.ones((3,2))*10})
    assert order['center']=='B'
    curriculum=ProgressiveCurriculum({'data':order})
    assert curriculum.fraction(11000)==.25 and curriculum.grl_strength(15000)==.025


def test_encoder_all_variants_and_adapter_input_contract():
    batch={'eeg':torch.randn(2,1300,5),'auditory':torch.randn(2,1300,5),'positions':torch.randn(2,5,3)*.04,'channel_mask':torch.tensor([[1,1,1,1,0],[1,1,1,1,1]],dtype=torch.bool),'times':torch.arange(20)[None].expand(2,-1)/50}
    # A bad sensor has arbitrary nonfinite measurements/coordinates; no valid
    # EEG path may depend on those values, including soft scalp attention.
    batch['eeg'][0,:,4]=float('nan');batch['auditory'][0,:,4]=float('nan');batch['positions'][0,4]=float('nan')
    for system in ('S0','S1','S2','S3'):
        encoder=build_encoder(cfg(),system,FakeReve() if system!='S0' else None)
        out=encoder(batch)
        assert out['teacher'].shape==(2,20,8) and torch.isfinite(out['teacher']).all()
        assert out['phoneme_logits'].shape==(2,20,19)
        if system in {'S0','S1'}:assert torch.count_nonzero(out['phi'])==0
        encoder.set_stage('frozen');assert not any(p.requires_grad for p in encoder.parameters())
        assert not any(k.startswith('reve.model.') and '.down.' not in k and '.up.' not in k for k in encoder.adaptation_state_dict())


def test_mtrf_bad_sensor_does_not_disable_other_channels():
    pytest.importorskip('mtrf')
    rng=np.random.default_rng(9);phi=rng.normal(size=(400,2));y=np.roll(phi,2,axis=0)@rng.normal(size=(2,5));mask=np.ones_like(y,dtype=bool);mask[:,4]=False
    fitted=_fit([{'phi':phi,'y':y,'mask':mask}],.1,lags=9)
    assert np.count_nonzero(fitted['weights'][:,4])==0
    assert np.count_nonzero(fitted['weights'][:,:4])>0


def test_nested_crossfit_outer_content_excluded_from_inner(monkeypatch):
    import eegspeech.neurospeech.neural as neural
    calls=[]
    def fake_fit(chunks,penalty,lags,backend):
        groups={c['content'] for c in chunks};calls.append(groups)
        return {'weights':np.zeros((lags+1,4)),'phi_mean':np.zeros(1),'phi_std':np.ones(1),'lags':lags}
    monkeypatch.setattr(neural,'_fit',fake_fit)
    monkeypatch.setattr(neural,'circular_null',lambda chunks,permutations,shift,seed:np.zeros((permutations,4)))
    rng=np.random.default_rng(0)
    chunks=[{'content':name,'phi':rng.normal(size=(400,1)),'y':rng.normal(size=(400,4)),'mask':np.ones((400,4),bool)} for name in ('A','B','C')]
    result=neural.nested_crossfit(chunks,ridge=(.1,1),lags=3,backend='sample_normalized_ridge',permutations=3)
    # Each outer group: 2 lambdas x 2 singleton inner fits, then 1 two-group
    # outer fit. The final refit is the sole fit seeing all three contents.
    for i,held in enumerate(('A','B','C')):
        outer_calls=calls[i*5:(i+1)*5]
        assert all(held not in groups for groups in outer_calls)
        assert all(len(groups)==1 for groups in outer_calls[:4])
        assert len(outer_calls[4])==2
    assert calls[-1]=={'A','B','C'}
    assert len(result['oof_chunks'])==3


def test_lora_starts_at_pretrained_projection_without_base_gradient():
    base=nn.Linear(7,9,bias=False);lora=LoRALinear(base,rank=2,alpha=4,dropout=0)
    x=torch.randn(3,7);assert torch.equal(lora(x),base(x))
    lora(x).square().mean().backward()
    assert base.weight.grad is None and lora.up.weight.grad.abs().sum()>0


def test_independent_content_rhythm_reliability_paths():
    model=ConditionalFlow(cfg()).eval()
    nn.init.normal_(model.output.weight,std=.2)
    for block in model.blocks:nn.init.normal_(block.modulation[-1].weight,std=.2)
    x=torch.randn(1,4,38);tau=torch.tensor([.5]);c={'content':torch.randn(1,20,8),'rhythm':torch.randn(1,20,2),'reliability':torch.ones(1,20,2),'times':torch.arange(20)[None]/50}
    y=model(x,tau,c)
    content=dict(c,content=torch.randn_like(c['content']));rhythm=dict(c,rhythm=torch.randn_like(c['rhythm']))
    assert not torch.allclose(model(x,tau,content),y)
    assert not torch.allclose(model(x,tau,rhythm),y)
    confidence=torch.ones_like(c['reliability']);confidence[...,0]=0
    nullcontent=dict(c,reliability=confidence)
    nullcontent2=dict(content,reliability=confidence)
    assert torch.allclose(model(x,tau,nullcontent),model(x,tau,nullcontent2))
    confidence=confidence.clone();confidence[...,1]=0
    neutral=dict(c,reliability=confidence)
    neutral2=dict(rhythm,reliability=confidence)
    assert torch.allclose(model(x,tau,neutral),model(x,tau,neutral2))
