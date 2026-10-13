"""KaraOne state-isolated physical EEG and genuine articulation audio.

The practice prompt list is authoritative. The 2N ``speaking_inds`` cells
alternate stimulus/preparation and articulation; imagined audio is never truth.
"""
from __future__ import annotations
import json
import os
import shutil
import tarfile
from bisect import bisect_right
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
import numpy as np
from scipy.io import loadmat
import soundfile as sf
from .brennan import download,sha256

PROMPTS={"iy","uw","piy","tiy","diy","m","n","pat","pot","knew","gnaw"}

def extract_selected(archive,folder):
    """One sequential pass, skip FDT/ICA/features/Kinect3D/animation payloads."""
    folder=Path(folder);inventory=[]
    with ExitStack() as stack:
        try:
            import indexed_bzip2
            stream=stack.enter_context(indexed_bzip2.open(str(archive),parallelization=4))
            tar=stack.enter_context(tarfile.open(fileobj=stream,mode='r|'))
        except ImportError:
            tar=stack.enter_context(tarfile.open(archive,mode='r|bz2'))
        for member in tar:
            part=PurePosixPath(member.name);basename=part.name
            wanted=member.isfile() and (part.suffix.lower() in {'.cnt','.wav','.txt','.set','.loc','.sfp','.ced'} or basename=='epoch_inds.mat')
            inventory.append({'path':member.name,'bytes':member.size,'retained_for_processing':wanted})
            if not wanted:continue
            if part.is_absolute() or '..' in part.parts:raise ValueError('Unsafe archive path')
            destination=folder/str(part);destination.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src,destination.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024*1024)
    return inventory

def phase_indices(path,n):
    """Per-trial state ranges anchored on rest and imagined; speaking ranges are placed by position.

    ``clearing_inds`` and ``thinking_inds`` are ordered in every session. The 2N
    ``speaking_inds`` cells are not: some sessions store them out of order, twice,
    or repeat the anchor ranges (MM15, MM16, MM20). A speaking range is the prompt
    of the trial whose rest-to-imagined gap contains it, or the articulation of the
    trial whose imagined-to-next-rest gap contains it. A slot without a range is
    withheld, never guessed. Inclusive MATLAB bounds may share one sample, which
    stays with the earlier state.
    """
    m=loadmat(path,squeeze_me=True,struct_as_record=False)
    rest=list(np.atleast_1d(m['clearing_inds']));imagined=list(np.atleast_1d(m['thinking_inds']));speech=list(np.atleast_1d(m['speaking_inds']))
    if len(rest)!=n or len(imagined)!=n or len(speech)!=2*n:raise ValueError('KaraOne phase/prompt count mismatch')
    span=lambda v:[int(np.asarray(v).ravel()[0])-1,int(np.asarray(v).ravel()[1])]
    anchors=[(span(rest[i]),span(imagined[i])) for i in range(n)];slack=2
    for i,(r,t) in enumerate(anchors):
        if r[0]<0 or r[1]<=r[0] or t[1]<=t[0] or r[1]>t[0]+slack or (i and anchors[i-1][1][1]>r[0]+slack):
            raise ValueError('Overlapping KaraOne rest/imagined trial anchors')
    starts=[r[0] for r,_ in anchors];slots=[{'stimulus_preparation':[],'articulated':[]} for _ in range(n)];unplaced=[]
    for cell in speech:
        a,b=span(cell);i=bisect_right(starts,a+slack)-1;state=None
        if i>=0 and b>a:
            r,t=anchors[i];following=anchors[i+1][0][0] if i+1<n else None
            if r[1]-slack<=a and b<=t[0]+slack:state='stimulus_preparation'
            elif t[1]-slack<=a and (following is None or b<=following+slack):state='articulated'
        if state is None:unplaced.append([a,b])
        elif [a,b] not in slots[i][state]:slots[i][state].append([a,b])
    result=[];withheld=[]
    for (r,t),slot in zip(anchors,slots):
        kept={'rest':list(r)};notes={}
        for name in ('stimulus_preparation','imagined','articulated'):
            if name=='imagined':kept[name]=list(t);continue
            if not slot[name]:notes[name]={'reason':'no speaking range inside this trial slot'};continue
            kept[name]=list(slot[name][0])
            if len(slot[name])>1:notes['duplicate_'+name]={'reason':'first source range used','source_indices':slot[name][1:]}
        previous=None
        for name in kept:
            if previous is not None:kept[name][0]=max(kept[name][0],kept[previous][1])
            previous=name
        result.append(kept);withheld.append(notes)
    return result,withheld,unplaced

def source_geometry(raw,setfile):
    """Use provided EEGLAB cap locations, never implausible CNT 2D display coords."""
    from ..storage import geometry_positions
    import mne
    e=loadmat(setfile,squeeze_me=True,struct_as_record=False)['EEG'];locations=list(np.atleast_1d(e.chanlocs))
    names=[str(c.labels) for c in locations]
    if len(names)!=62 or any(n not in raw.ch_names for n in names):raise ValueError('KaraOne source cap layout mismatch')
    xyz=np.array([[-float(c.Y),float(c.X),float(c.Z)] for c in locations])/1000
    if not np.isfinite(xyz).all() or np.linalg.norm(xyz,axis=1).max()>.2:raise ValueError('Invalid EEGLAB mm coordinates')
    montage=mne.channels.make_dig_montage(ch_pos=dict(zip(names,xyz)),coord_frame='head')
    positions,evidence=geometry_positions(names,montage)
    evidence.update(positions_source='source EEGLAB .set 62-channel cap template; individual digitization not claimed',source_coordinate_conversion='EEGLAB anterior/left/superior mm -> RAS head (-Y,X,Z)/1000',coordinate_quality='source-cap-template')
    return names,positions,evidence

def run(root=None,subjects=None,retry_withheld=False):
    from ..storage import register_audio,publish_record,reclaim_sources,ensure_space,mark_dataset_complete,manifest_batch
    os.environ.setdefault('NUMBA_DISABLE_JIT','1')
    import mne
    workspace=Path.cwd();root=Path(root or workspace/'data/processed/neurospeech');out=root/'karaone';meta=out/'metadata';meta.mkdir(parents=True,exist_ok=True)
    stage=workspace/'data/staging/neurospeech/karaone';audit=json.loads((workspace/'report/dataset_storage_audit_2026-10-13.json').read_text())['karaone'];completed=[]
    for item in audit['files']:
        subject=item['subject']
        if subjects and subject not in subjects:continue
        marker=meta/f'{subject}_complete.json'
        if marker.exists() and not (retry_withheld and json.loads(marker.read_text()).get('withheld_states')):completed.append(subject);continue
        folder=stage/subject;folder.mkdir(parents=True,exist_ok=True);archive=folder/f'{subject}.tar.bz2'
        ensure_space(folder,item['bytes']+2_000_000_000)
        download(item['url'],archive,size=item['bytes']);archive_sha=sha256(archive)
        # First pilot may already have selected files; normal batches need one scan.
        if not (folder/'extraction_complete.json').exists():
            inventory=extract_selected(archive,folder);(folder/'extraction_complete.json').write_text(json.dumps(inventory))
        else:inventory=json.loads((folder/'extraction_complete.json').read_text())
        cnts=list(folder.rglob('*.cnt'));epochs=list(folder.rglob('epoch_inds.mat'))
        if len(cnts)!=1 or len(epochs)!=1:raise ValueError(f'KaraOne expected one CNT/epoch index for {subject}')
        cnt=cnts[0];d=cnt.parent;kinect=d/'kinect_data'
        practice=kinect/(subject+'_p.txt');practice=practice if practice.exists() else kinect/'ID_p.txt'
        practice_labels=[v.strip().strip('/') for v in practice.read_text().splitlines() if v.strip()]
        n_epochs=len(np.atleast_1d(loadmat(epochs[0],squeeze_me=True,struct_as_record=False)['thinking_inds']))
        labels=practice_labels;label_source=practice
        if len(labels)!=n_epochs:
            accepted=kinect/'labels.txt'
            if not accepted.exists():raise ValueError('Practice sequence includes failed trials but accepted source labels missing')
            labels=[v.strip().strip('/') for v in accepted.read_text().splitlines() if v.strip()];label_source=accepted
            # Source accepted trials must retain chronological practice order.
            pointer=0
            for label in labels:
                while pointer<len(practice_labels) and practice_labels[pointer]!=label:pointer+=1
                if pointer==len(practice_labels):raise ValueError('Accepted labels are not a subsequence of practiced prompts')
                pointer+=1
        if not set(labels)<=PROMPTS:raise ValueError(f'Unrecognized KaraOne prompts {set(labels)-PROMPTS}')
        phases,withheld,unplaced=phase_indices(epochs[0],len(labels));raw=mne.io.read_raw_cnt(cnt,preload=False,data_format='int32',eog=['VEO','HEO'],ecg=['EKG'],emg=['EMG'],misc=['Trigger','M1','M2'],verbose='ERROR')
        if raw.info['sfreq']!=1000:raise ValueError('Unexpected KaraOne sampling rate')
        sets=[p for p in d.glob('*.set')];names,positions,geometry=source_geometry(raw,sets[0]);picks=[raw.ch_names.index(n) for n in names]
        source={'url':item['url'],'archive_bytes':item['bytes'],'archive_sha256':archive_sha,'cnt_sha256':sha256(cnt),'epoch_sha256':sha256(epochs[0]),'cnt_data_format':'int32','cnt_units':'MNE calibration converted V to uV','native_rate':1000,'prompt_source':label_source.name,'practiced_prompt_count':len(practice_labels),'accepted_trial_count':len(labels),'accepted_prompt_order_verified':True,'trial_identity':'accepted epoch and WAV index; failed practice retries are not fabricated as EEG trials'}
        for p in [epochs[0],practice,*kinect.glob('*.txt'),*sets]:shutil.copy2(p,meta/f'{subject}_{p.name}')
        (meta/f'{subject}_archive_inventory.json').write_text(json.dumps(inventory,indent=2));records=[];trial_receipts=[]
        with manifest_batch('karaone',root=root):
            for trial,(prompt,states) in enumerate(zip(labels,phases)):
                wav=kinect/f'{trial}.wav'
                if not wav.exists():raise ValueError(f'Missing KaraOne articulation WAV {subject}/{trial}')
                content=f'kara_{subject}_{trial:04d}';audio=register_audio('karaone',content,wav,words=[{'start':0,'end':sf.info(wav).duration,'text':prompt}],metadata={'task':'articulated','subject':subject,'trial':trial,'prompt':prompt,'recorded_waveform':True,'imagined_synchronous_truth':False,'source_url':item['url'],'source_trial_zero_based':True},root=root)
                receipt={'trial':trial,'prompt':prompt,'source_indices':states,'withheld_states':withheld[trial],'audio':content,'imagined_audio_truth':False};trial_receipts.append(receipt)
                for state,(first,stop) in states.items():
                    if stop>raw.n_times:raise ValueError('KaraOne phase extends outside EEG')
                    eeg=raw.get_data(picks=picks,start=first,stop=stop)*1e6
                    duration=eeg.shape[1]/1000;group=f'karaone:{subject}:trial-{trial:04d}'
                    records.extend(publish_record('karaone',subject,f'trial-{trial:04d}_{state}',eeg,1000,names,positions,[{'audio':content,'content_group':group,'eeg_start':0.,'audio_start':0.,'duration':duration,'minimum_seconds':.001,'task':state}],metadata={**geometry,**source,'task':state,'prompt':prompt,'trial':trial,'guard_seconds':min(.5,duration/4),'source_eeg_start_seconds':first/1000,'source_eeg_stop_seconds':stop/1000,'filter_state_isolated':True,'short_phase_reflected_fir_support':True,'audio_relationship':'recorded articulation of same prompt/trial; no independently measured sample synchronization','target_audio_synchronous':False,'source_epoch_index_convention':'MATLAB inclusive indices converted [first-1,last)','stimulus_state_includes_preparation':state=='stimulus_preparation','excluded_channels':[n for n in raw.ch_names if n not in names]},root=root))
        raw.close();(meta/f'{subject}_trials.json').write_text(json.dumps(trial_receipts,indent=2));marker.write_text(json.dumps({'subject':subject,'trials':len(labels),'records':len(records),'withheld_states':sum(k in ('stimulus_preparation','articulated') for w in withheld for k in w),'unplaced_speaking_ranges':unplaced,'source':source},indent=2))
        reclaimed=reclaim_sources([p for p in folder.rglob('*') if p.is_file()],'karaone',records,root=root)
        shutil.rmtree(folder);completed.append(subject);print(json.dumps({'dataset':'karaone','subject':subject,'trials':len(labels),'records':len(records),'released_bytes':reclaimed}),flush=True)
    all_completed=[p.name.removesuffix('_complete.json') for p in meta.glob('*_complete.json')]
    return mark_dataset_complete('karaone',all_completed,14,root=root,details={'phase_isolated':True,'imagined_audio_is_synchronous_truth':False,'source':audit['source']})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--subject',action='append');p.add_argument('--retry-withheld',action='store_true')
    args=p.parse_args();run(subjects=args.subject,retry_withheld=args.retry_withheld)
