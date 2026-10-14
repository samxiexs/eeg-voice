"""Processed-only, content-safe windows and independently fitted fold statistics."""
from __future__ import annotations
from collections import defaultdict
import csv
from pathlib import Path
import copy
import hashlib
import os
import re
import time
import numpy as np
from scipy import signal
import soundfile as sf
import torch
from torch.utils.data import Dataset
from eegspeech.utils import read_json,atomic_json,stable_int,fingerprint,Moments,digest
from .storage import PROCESSING,dataset_dir,verify_dataset,manifest_lock

PHONETIC_FEATURES=['vowel','voiced','stop','fricative','affricate','nasal','approximant','labial','dental','alveolar','postalveolar','palatal','velar','glottal','high','mid','low','rounded','boundary']
ARPABET={
 'AA':('vowel','voiced','low'),'AE':('vowel','voiced','low'),'AH':('vowel','voiced','mid'),'AO':('vowel','voiced','mid','rounded'),'AW':('vowel','voiced','low'),'AY':('vowel','voiced','low'),
 'EH':('vowel','voiced','mid'),'ER':('vowel','voiced','mid'),'EY':('vowel','voiced','mid'),'IH':('vowel','voiced','high'),'IY':('vowel','voiced','high'),'OW':('vowel','voiced','mid','rounded'),'OY':('vowel','voiced','mid','rounded'),'UH':('vowel','voiced','high','rounded'),'UW':('vowel','voiced','high','rounded'),
 'B':('voiced','stop','labial'),'P':('stop','labial'),'D':('voiced','stop','alveolar'),'T':('stop','alveolar'),'G':('voiced','stop','velar'),'K':('stop','velar'),
 'F':('fricative','labial'),'V':('voiced','fricative','labial'),'TH':('fricative','dental'),'DH':('voiced','fricative','dental'),'S':('fricative','alveolar'),'Z':('voiced','fricative','alveolar'),'SH':('fricative','postalveolar'),'ZH':('voiced','fricative','postalveolar'),'HH':('fricative','glottal'),
 'CH':('affricate','postalveolar'),'JH':('voiced','affricate','postalveolar'),'M':('voiced','nasal','labial'),'N':('voiced','nasal','alveolar'),'NG':('voiced','nasal','velar'),'L':('voiced','approximant','alveolar'),'R':('voiced','approximant','postalveolar'),'W':('voiced','approximant','labial','velar','rounded'),'Y':('voiced','approximant','palatal')}


_AUDITORY_KERNEL=signal.firwin(1601,[1,8],pass_zero=False,fs=200,window='hamming')


def auditory_filter(x):
    """Fixed 1–8Hz, 8sec Hamming FIR; x[time,channel], no fitted parameters."""
    x=np.asarray(x,dtype=np.float64);kernel=signal.firwin(1601,[1,8],pass_zero=False,fs=200,window='hamming')
    return signal.fftconvolve(np.pad(x,((800,800),(0,0)),mode='reflect'),kernel[:,None],mode='valid',axes=0).astype(np.float32)


def content_masks(cfg):
    """Reviewed canonical-audio intervals withheld from every window, e.g. Kymata speaker overlap."""
    path=cfg.path.parent/'content_masks.json'
    masks=read_json(path) if path.exists() else {}
    for key,value in masks.items():
        if value.get('status') not in {'pending','locked'} or any(len(v)!=2 or not 0<=v[0]<v[1] for v in value.get('intervals',[])):
            raise ValueError(f'Invalid content mask {key}')
    return masks


def _split_contents(contents,dataset):
    keys=sorted(set(contents),key=lambda x:stable_int('neurospeech-v2.1:content:'+x))
    if dataset=='ds004408':
        return {k:'train' if int(re.search(r'audio(\d+)',k).group(1))<=14 else 'validation' if int(re.search(r'audio(\d+)',k).group(1))<=17 else 'test' for k in keys}
    n=len(keys)
    if n<5:raise ValueError(f'{dataset}: insufficient independent content groups ({n})')
    nt=max(1,round(n*.15));nv=max(1,round(n*.15))
    return {k:'test' if i<nt else 'validation' if i<nt+nv else 'train' for i,k in enumerate(keys)}


def build_registry(cfg,require_complete=True):
    """No scientific selection: deterministic source/subject/content membership."""
    subjects={};groups={};sources={};duplicates=defaultdict(list);masks=content_masks(cfg)
    for dataset in ['ds004408','brennan2019','etard','kymata_english','karaone']:
        folder=cfg.dir('processed')/dataset
        if not (folder/'manifest.json').exists():
            if require_complete:raise FileNotFoundError(f'Pending dataset {dataset}')
            continue
        m=read_json(folder/'manifest.json')
        if require_complete and not m['complete']:raise ValueError(f'{dataset} acquisition incomplete')
        people=m.get('subjects',sorted({r['subject'] for r in m['records'].values()}))
        valid=[r for r in m['records'].values() if r.get('task')=='perceived']
        paired_people=sorted({r['subject'] for r in valid});policy=(cfg.get('subject_policy') or {}).get(dataset);dropped=[]
        if policy=='authors_analyzed':
            # The release carries the original study's quality control; only its analyzed sample is used.
            with (folder/'metadata/participants.tsv').open(encoding='utf-8-sig') as handle:kept={r['participant_id'] for r in csv.DictReader(handle,delimiter='\t') if r['analyzed']=='yes'}
            dropped=sorted(set(paired_people)-kept);paired_people=[s for s in paired_people if s in kept]
        elif policy is not None:raise ValueError(f'Unknown subject policy {policy}')
        subjects[dataset]=[f'{dataset}:{s}' for s in (people if dataset=='karaone' else paired_people)]
        if dataset!='karaone' and valid:
            groups[dataset]=_split_contents([r['content_group'] for r in valid],dataset)
            clean={groups[dataset][r['content_group']] for r in valid if r['metadata'].get('condition','clean')=='clean'}
            if clean!={'train','validation','test'}:raise ValueError(f'{dataset}: clean content lacks a {sorted({"train","validation","test"}-clean)} group')
        sources[dataset]={'complete':m['complete'],'acquired_subjects':len(people),'paired_subjects':len(paired_people),'subjects':len(people),'records':len(m['records']),'usable_pairings':len(valid),'excluded_subjects':sorted(set(people)-set(paired_people)) if dataset!='karaone' else [],'subject_policy':policy,'policy_excluded_paired_subjects':dropped,'manifest_sha256':digest(folder/'manifest.json')}
        for c in m['contents'].values():duplicates[c['sha256']].append((dataset,c['id']))
    # Repeated conditions already share canonical group; identical cross-source
    # stimuli are blocked conservatively until a joint assignment is reviewed.
    collisions=[v for v in duplicates.values() if len({q[0] for q in v})>1]
    if collisions:raise ValueError(f'Cross-source duplicate stimulus audit required: {collisions}')
    masks={k:v for k,v in masks.items() if k.split(':')[0] in sources}
    pending=sorted(k for k,v in masks.items() if v['status']!='locked')
    if require_complete and pending:raise ValueError(f'Content mask review pending (configs/neurospeech_v2/content_masks.json): {pending}')
    development=[];folds={f'F{i}':[] for i in range(5)}
    for dataset in cfg['protocols']['P1']['sources']:
        people=sorted(subjects.get(dataset,[]),key=lambda x:stable_int('neurospeech-v2.1:person:'+x))
        # Plan 3.1: ds004408 keeps sub-001..003 as development subjects; other rosters follow the fixed hash order.
        if dataset=='ds004408':people=sorted(people,key=lambda x:x not in {f'ds004408:sub-{i:03d}' for i in (1,2,3)})
        n=cfg['protocols']['P1']['development_counts'][dataset];development.extend(people[:n])
        for i,p in enumerate(people[n:]):folds[f'F{i%5}'].append(p)
    p2dev=[p for p in development if p.split(':')[0] in cfg['protocols']['P2']['sources']]
    value={'version':'neurospeech-v2.1','subjects':subjects,'content_splits':groups,'sources':sources,
           'duplicate_audit':{'exact_canonical_audio_collisions':collisions,'method':'processed PCM SHA256; condition identities checked by source readers'},
           'content_masks':{k:{'kind':v.get('kind'),'status':v['status'],'intervals':v.get('intervals',[])} for k,v in masks.items()},
           'protocols':{'P1':{'development_subjects':development,'test_folds':folds},
                        'P2':{'development_subjects':p2dev,'test_subjects':subjects.get('brennan2019',[])}},
           'processing':PROCESSING,'all_sources_complete':len(sources)==5 and all(v['complete'] for v in sources.values())}
    value['registry_hash']=fingerprint(value);atomic_json(cfg.dir('processed')/'registry.json',value)
    return value


def _window_rows(cfg):
    registry=cfg.registry();result=[];qc=[]
    for dataset in cfg['protocols']['P1']['sources']:
        folder=cfg.dir('processed')/dataset
        if not (folder/'manifest.json').exists():continue
        manifest=read_json(folder/'manifest.json')
        for r in manifest['records'].values():
            if r.get('task')!='perceived':continue
            if r['audio'] not in manifest['contents']:raise ValueError('Missing paired processed audio')
            x=np.load(folder/r['path']/'eeg.npy',mmap_mode='r');base=np.asarray(r['good_channels'],bool)
            # Full source filter support contains no neighboring content. Derive
            # auditory stream inside that same independent record, before crops.
            guard=float(cfg['preprocessing']['guard_seconds']);seconds=cfg['window']['seconds']
            first=guard+cfg['window']['eeg_left'];last=r['duration']-guard-seconds-cfg['window']['eeg_right']
            split=registry['content_splits'][dataset][r['content_group']]
            hidden=registry.get('content_masks',{}).get(f"{dataset}:{r['audio']}",{}).get('intervals',[])
            stride=cfg['window']['train_stride'] if split=='train' else cfg['window']['evaluation_stride']
            for start in np.arange(first,last+1e-7,stride):
                a=round(start*200);core=np.asarray(x[:,a:a+round(seconds*200)])
                valid=base&(core.std(1)>=cfg['preprocessing']['flat_uv'])&(np.ptp(core,axis=1)<=cfg['preprocessing']['peak_to_peak_uv'])
                # Annotated bad channels are masked, as the sources interpolate them; only artifacts on the
                # remaining channels count toward the transient limit, with a floor on what is left.
                rejected=(base&~valid).sum()/max(base.sum(),1)>cfg['preprocessing']['max_bad_fraction'] or valid.sum()/len(valid)<cfg['preprocessing']['min_valid_fraction']
                key=f"{r['id']}:window-{start:.3f}";canonical=r['audio_start']+float(start)
                if any(lo<canonical+seconds and hi>canonical for lo,hi in hidden):rejected=True
                qc.append({'id':key,'dataset':dataset,'subject':r['subject'],'content_split':split,'rejected':bool(rejected),'valid_channels':int(valid.sum()),'total_channels':len(valid)})
                if rejected:continue
                compact={k:r[k] for k in ('dataset','subject','recording','path','audio','content_group','duration','audio_start','task')}
                result.append({**compact,'id':key,'record_id':r['id'],'start':float(start),'canonical_start':r['audio_start']+float(start),
                               'canonical_audio':f"{dataset}:{r['audio']}",'channel_mask':valid.tolist(),'content_split':split,
                               'condition':r['metadata'].get('condition','clean'),'audio_path':manifest['contents'][r['audio']]['path']})
    atomic_json(cfg.dir('processed')/'window_qc.json',{'windows':qc})
    return result


_ROWS={}


def all_rows(cfg):
    """Window list keyed by the data contract only, so configs that differ in training settings share it."""
    path=cfg.dir('processed')/'windows.json';registry=cfg.registry()
    key=fingerprint({'preprocessing':cfg['preprocessing'],'window':cfg['window'],'registry':registry['registry_hash']})
    if key in _ROWS:return _ROWS[key]
    if path.exists():
        saved=read_json(path)
        if saved.get('window_hash')==key:_ROWS[key]=saved['rows'];return saved['rows']
    rows=_window_rows(cfg);atomic_json(path,{'window_hash':key,'registry_hash':registry['registry_hash'],'rows':rows});_ROWS[key]=rows
    return rows


def rows_for(cfg,subjects,content_split,conditions=('clean',)):
    subjects=set(subjects)
    return [r for r in all_rows(cfg) if f"{r['dataset']}:{r['subject']}" in subjects and (content_split is None or r['content_split']==content_split) and (conditions is None or r['condition'] in conditions)]


def reference(words,start,seconds=4):
    return ' '.join(str(w.get('word',w.get('text',''))) for w in words if start<=(float(w['start'])+float(w['end']))/2<start+seconds).strip()


def _audio_slice(path,start,seconds):
    with sf.SoundFile(path) as f:
        if f.samplerate!=24000:raise ValueError('Processed audio must be 24kHz')
        first=round(start*24000);count=round(seconds*24000)
        if first<0 or first+count>len(f):raise ValueError('Audio context outside processed canonical content')
        f.seek(first);x=f.read(count,dtype='float32')
    if len(x)!=count:raise ValueError('Truncated audio context')
    return x


def _phonemes(content,start):
    times=start+(np.arange(200)+.5)/50;target=np.zeros((200,19),np.float32);mask=np.zeros((200,19),bool)
    for p in content.get('phonemes',[]):
        if p.get('quality_status') in {'pending','unreviewed','unverified'}:continue
        phone=re.sub(r'\d','',str(p.get('phone',p.get('text','')))).upper().strip()
        if phone not in ARPABET:continue
        a,b=float(p['start']),float(p['end']);inside=(times>=a)&(times<b)
        mask[inside]=True
        for label in ARPABET[phone]:target[inside,PHONETIC_FEATURES.index(label)]=1
        # A boundary is a 40ms neighborhood, with the same reliable annotation mask.
        boundary=np.abs(times-a)<.02;target[boundary,-1]=1;mask[boundary,-1]=True
    return target,mask


class TargetCache:
    """One bounded native target cache per unique canonical crop, never person."""
    def __init__(self,cfg):
        self.cfg=cfg;self.audio=None;self.manifests={};self.tiers={};self.memory={}
        self.fixed={'audio':cfg['audio'],'models':cfg.models,'frontend_sha':digest(Path(__file__).parent/'neural.py')}
    def get(self,row):
        cfg=self.cfg;folder=cfg.dir('processed')/row['dataset']
        if row['dataset'] not in self.manifests:self.manifests[row['dataset']]=read_json(folder/'manifest.json')
        c=self.manifests[row['dataset']]['contents'][row['audio']]
        # Phoneme tiers can be added after ingest; cached targets must not outlive the labels they were built from.
        tier=self.tiers.setdefault((row['dataset'],row['audio']),fingerprint(c.get('phonemes',[])))
        key=fingerprint({'wave':c['sha256'],'phonemes':tier,'start':row['canonical_start'],'seconds':4,**self.fixed})
        if key in self.memory:return self.memory[key]
        dest=cfg.dir('artifacts')/'target_cache';path=dest/(key+'.npz');dest.mkdir(parents=True,exist_ok=True)
        with manifest_lock(dest):
            if path.exists():
                os.utime(path,None)
                with np.load(path,allow_pickle=False) as z:value={k:z[k].copy() for k in z.files}
                if len(self.memory)<2500:self.memory[key]=value
                return value
            from eegspeech.perceived_audio import FrozenAudio
            from .neural import AcousticFrontend
            if self.audio is None:self.audio=FrozenAudio(cfg,device='cpu')
            absolute=row['canonical_start'];wave=_audio_slice(folder/c['path'],absolute,4)
            teacherwave=_audio_slice(folder/c['path'],absolute-1,6)
            wave16=signal.resample_poly(teacherwave,2,3).astype(np.float32)
            feature,t=self.audio.wavlm(wave16,crop_start=-1,core_seconds=4)
            times=(np.arange(200)+.5)/50
            feature=feature[0].cpu().numpy();t=t.cpu().numpy()
            teacher=np.stack([np.interp(times,t,feature[:,i]) for i in range(feature.shape[1])],1).astype(np.float32)
            with torch.no_grad():mel=self.audio.vocos.features(torch.from_numpy(wave)[None])[0].cpu().numpy()
            # Use full6.5s context for fixed FIR/RMS envelopes; crop true64Hzcore.
            context=_audio_slice(folder/c['path'],absolute-1,6.5)
            phi64=AcousticFrontend().numpy(context)[0][64:320][:,[0,9]]
            phi=np.stack([np.interp(times,np.arange(256)/64,phi64[:,i]) for i in range(2)],1).astype(np.float32)
            phones,mask=_phonemes(c,absolute)
            result={'teacher':teacher,'mel':mel,'phi':phi,'phoneme_targets':phones,'phoneme_mask':mask}
            if teacher.shape!=(200,768) or mel.shape!=(100,376):raise ValueError('Native audio frontend contract mismatch')
            tmp=path.with_suffix('.part.npz');np.savez_compressed(tmp,**result);os.replace(tmp,path)
            cap=cfg['storage']['teacher_cache_bytes'];files=sorted(dest.glob('*.npz'),key=lambda f:f.stat().st_mtime);total=sum(p.stat().st_size for p in files)
            for old in files:
                if total<=cap:break
                if old==path:continue
                total-=old.stat().st_size;old.unlink()
            return result


def _identity_stats(cfg):
    return {'eeg':{'mean':[0.],'std':[1.]},'auditory':{'mean':[0.],'std':[1.]},
            'teacher':{'mean':[0.]*768,'std':[1.]*768},'phi':{'mean':[0.]*2,'std':[1.]*2},'mel':{'mean':[0.]*100,'std':[1.]*100}}


class WindowDataset(Dataset):
    def __init__(self,cfg,protocol='P1',fold='F0',split='train',targets=True,fixed_validation=False,rows=None,stats=None,stats_override=None):
        self.cfg,self.protocol,self.fold,self.split,self.targets=cfg,protocol,fold,split,targets
        self.all_rows=all_rows(cfg)
        if rows is None:
            if split=='train_subject_validation':rows=rows_for(cfg,cfg.fold_subjects(protocol,fold,'train'),'validation')
            elif split=='calibration':raise ValueError('Use calibration_dataset for exactly one unseen person and one independent 60s shard')
            else:rows=rows_for(cfg,cfg.fold_subjects(protocol,fold,split),split)
        self.rows=list(rows)
        from .protocol import assert_scope
        assert_scope(self.rows,protocol,'validation' if split=='train_subject_validation' else split)
        if not self.rows:raise ValueError(f'Empty {protocol}/{fold}/{split}; acquisition/synchronization/QC may be incomplete')
        if fixed_validation:
            groups=defaultdict(list)
            for row in self.rows:groups[(row['dataset'],row['subject'],row['canonical_audio'])].append(row)
            limit=cfg['training']['validation_windows_per_audio'];chosen=[]
            for group in groups.values():
                # Evenly spaced, then retain ends for ≥10sec negative controls.
                indices=np.linspace(0,len(group)-1,min(limit,len(group))).round().astype(int)
                chosen.extend(group[i] for i in indices)
            self.rows=chosen
        selected=stats_override if stats_override is not None else stats
        if selected is False or not targets:self.stats=_identity_stats(cfg) if selected is None or selected is False else selected
        elif selected is not None:self.stats=selected
        elif split=='train':self.stats=fit_statistics(cfg,self.rows,protocol,fold,write=True)
        else:raise ValueError('Validation/test require explicit training-only statistics')
        self.cache=TargetCache(cfg);self._auditory={};self._auditory_bytes=0
        self._manifests={};self._record_meta={}
    def __len__(self):return len(self.rows)
    def with_stats(self,stats):return WindowDataset(self.cfg,self.protocol,self.fold,self.split,self.targets,rows=self.rows,stats=stats)
    def __getitem__(self,index):
        r=self.rows[index];cfg=self.cfg;folder=cfg.dir('processed')/r['dataset'];record=folder/r['path'];start=r['start'];left=cfg['window']['eeg_left']
        if str(record) not in self._record_meta:self._record_meta[str(record)]=read_json(record/'meta.json')
        meta=self._record_meta[str(record)]
        x=np.load(record/'eeg.npy',mmap_mode='r');a=round((start-left)*200);eeg=np.array(x[:,a:a+1300].T,dtype=np.float32)
        if eeg.shape!=(1300,len(meta['names'])):raise ValueError('Truncated processed EEG context')
        # The 8 s FIR needs 800 samples on each side; windows keep a 10 s guard, so the support is real EEG
        # of the same record and the result equals filtering the whole record.
        if a<800 or a+2100>x.shape[1]:auditory=auditory_filter(np.array(x.T))[a:a+1300]
        else:auditory=signal.fftconvolve(np.asarray(x[:,a-800:a+2100].T,dtype=np.float64),_AUDITORY_KERNEL[:,None],mode='valid',axes=0).astype(np.float32)
        y=np.load(record/'neural.npy',mmap_mode='r');b=round(start*64);bc=round((start-left)*64)
        neural=np.array(y[:,b:b+256].T);neural_context=np.array(y[:,bc:bc+416].T)
        good=np.asarray(r['channel_mask'],bool)
        def norm(x,kind):return (x-np.asarray(self.stats[kind]['mean'],np.float32))/np.asarray(self.stats[kind]['std'],np.float32)
        eeg=norm(eeg,'eeg');auditory=norm(auditory,'auditory');eeg[:,~good]=0;auditory[:,~good]=0
        result={'eeg':torch.tensor(eeg),'auditory':torch.tensor(auditory),'neural':torch.tensor(neural),'neural_context':torch.tensor(neural_context),
                'positions':torch.tensor(meta['positions'],dtype=torch.float32),'channel_mask':torch.tensor(good),
                'times':torch.tensor((np.arange(200)+.5)/50,dtype=torch.float32),'eeg_times':torch.tensor(np.arange(1300)/200-left,dtype=torch.float32),
                'frame_mask':torch.ones(200,dtype=torch.bool),'dataset':r['dataset'],'subject':r['subject'],'recording':r['recording'],
                'audio':r['audio'],'start':r['start'],'canonical_start':r['canonical_start'],'canonical_audio':r['canonical_audio'],'content_group':r['content_group'],'id':r['id']}
        if self.targets:
            if r['dataset'] not in self._manifests:self._manifests[r['dataset']]=read_json(folder/'manifest.json')
            c=self._manifests[r['dataset']]['contents'][r['audio']];native=self.cache.get(r)
            for k,v in native.items():
                if k in ('teacher','phi'):v=norm(v,k)
                elif k=='mel':v=norm(v.T,k).T
                result[k]=torch.tensor(v)
            result['wave']=torch.tensor(_audio_slice(folder/c['path'],r['canonical_start'],4))
            result['wave_context']=torch.tensor(_audio_slice(folder/c['path'],r['canonical_start']-1,6.5))
            result['text']=reference(c.get('words',[]),r['canonical_start'])
        return result


class AudioDataset(WindowDataset):
    def __init__(self,cfg,protocol='P1',fold='F0',split='train',targets=True,fixed_validation=False,rows=None,stats=None,**kwargs):
        if rows is None:rows=rows_for(cfg,cfg.fold_subjects(protocol,fold,split),split)
        dedup={}
        for r in rows:dedup.setdefault((r['canonical_audio'],r['canonical_start']),r)
        super().__init__(cfg,protocol,fold,split,targets,fixed_validation,list(dedup.values()),stats,**kwargs)


def fit_statistics(cfg,rows,protocol='P1',fold='F0',write=False):
    from .protocol import assert_scope,scope_manifest
    assert_scope(rows,protocol,'train');scope=scope_manifest(cfg,protocol,rows)
    key=fingerprint(scope);path=cfg.dir('artifacts')/protocol/'statistics'/fold/(key+'.json')
    if path.exists():return read_json(path)
    moments={k:Moments(n) for k,n in [('eeg',1),('auditory',1),('teacher',768),('phi',2),('mel',100)]}
    dataset=WindowDataset(cfg,protocol,fold,rows=rows,stats=False);seen=set()
    for i,r in enumerate(dataset.rows):
        sample=dataset[i];good=np.asarray(r['channel_mask'],bool)
        for k in ['eeg','auditory']:moments[k].add(sample[k].numpy()[200:1000,good].reshape(-1,1))
        canonical=(r['canonical_audio'],r['canonical_start'])
        if canonical not in seen:
            seen.add(canonical)
            for k in ['teacher','phi']:moments[k].add(sample[k].numpy())
            moments['mel'].add(sample['mel'].numpy().T)
    value={k:m.result() for k,m in moments.items()};value.update(provenance=scope,statistics_hash=key)
    if write:atomic_json(path,value)
    return value


def calibration_dataset(cfg,subject,protocol,fold,stats):
    source,person=subject.split(':',1);folder=cfg.dir('processed')/source;m=read_json(folder/'manifest.json')
    candidates=[r for r in m['calibration'].values() if r['subject']==person]
    # Calibration content is formal train-side stimulus, withheld person. Never
    # inspect adaptation performance to select the shard.
    groups=cfg.registry()['content_splits'][source];candidates=[r for r in candidates if any(g.startswith(source+':'+r['audio']) and s=='train' for g,s in groups.items())]
    if not candidates:raise ValueError('No independent training-content calibration shard')
    r=min(candidates,key=lambda v:stable_int('calibration:'+v['audio']));rows=[]
    for start in np.arange(11,44.5,2):
        rows.append({**r,'id':f"calibration:{subject}:{start}",'record_id':f"calibration:{subject}",'task':'perceived','start':float(start),
                     'canonical_start':r['audio_start']+float(start),'canonical_audio':source+':'+r['audio'],'content_group':source+':'+r['audio'],
                     'channel_mask':r['good_channels'],'condition':'clean','calibration':{'independently_filtered':True,'start':0,'stop':60}})
    return WindowDataset(cfg,protocol,fold,split='calibration',rows=rows,stats=stats)
