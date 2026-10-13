"""Versioned configuration; no raw paths or implicit network model loads."""
from pathlib import Path
import yaml
from eegspeech import ROOT
from eegspeech.utils import fingerprint,read_json,digest

class Config:
    def __init__(self,path=None,root=None):
        self.path=Path(path or ROOT/'configs/neurospeech_v2/plan.yaml').resolve()
        self.root=Path(root or ROOT).resolve()
        self.values=yaml.safe_load(self.path.read_text())
        self.models=read_json(self.path.parent/'models.json')
        code_root=Path(__file__).parent
        self.code_hashes={str(p.relative_to(self.root)):digest(p) for p in sorted(code_root.rglob('*.py'))}
        for p in [self.root/'eegspeech/perceived_audio.py',self.root/'eegspeech/utils.py']:
            self.code_hashes[str(p.relative_to(self.root))]=digest(p)
        registry_path=self.dir('processed')/'registry.json'
        self.dataset_registry_hash=read_json(registry_path).get('registry_hash') if registry_path.exists() else None
        self.hash=fingerprint({'config':self.values,'models':self.models,'code':self.code_hashes,'dataset_registry_hash':self.dataset_registry_hash})
        self.validate()

    def __getitem__(self,key):return self.values[key]
    def get(self,key,default=None):return self.values.get(key,default)
    def dir(self,key):return self.root/self.values['paths'][key]
    def model_dir(self,kind):return self.dir('models')/self.models[kind]['local_dir']
    def registry(self):return read_json(self.dir('processed')/'registry.json')
    def subjects(self,protocol='P1'):
        registry=self.registry()
        sources=self['protocols'][protocol]['sources']
        return [s for d in sources for s in registry['subjects'][d]]
    def fold_subjects(self,protocol,fold,split):
        r=self.registry();p=r['protocols'][protocol]
        if protocol=='P1':
            held=p['test_folds'][fold];dev=p['development_subjects']
            if split=='test':return held
            if split=='validation':return dev
            if split=='calibration':return held
            if split=='train':return [s for s in self.subjects(protocol) if s not in held+dev]
        else:
            if split in {'test','calibration'}:return p['test_subjects']
            if split=='validation':return p['development_subjects']
            if split=='train':return [s for s in self.subjects(protocol) if s not in p['development_subjects']]
        raise ValueError(f'Unknown split {split}')
    def run_dir(self,protocol,fold,seed,system):return self.dir('artifacts')/protocol/'runs'/fold/str(seed)/system
    def validate(self):
        p,w,a=self['preprocessing'],self['window'],self['audio']
        if p['eeg_rate']!=200 or p['neural_rate']!=64:raise ValueError('Locked EEG200/neural64 interface')
        if w['seconds']!=4 or a['teacher_dim']!=768:raise ValueError('Locked4sec/WavLM-base interface')
        if (a['sample_rate'],a['mel_bins'],a['n_fft'],a['hop_length'])!=(24000,100,1024,256):raise ValueError('Wrong Vocos frontend')
        if 'brennan2019' in self['protocols']['P2']['sources']:raise ValueError('External target leaked into P2 training')
        if 'karaone' in self['protocols']['P1']['sources']:raise ValueError('Imagined data leaked into Perceived experiment')
