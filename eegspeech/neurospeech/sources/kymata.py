"""Kymata Soto English conversation: whitelisted, run-by-run EEG on the audio clock.

Every task-en FIF mixes EEG with MEG, so one file at a time is downloaded,
checked against the audited SHA256, reduced to scalp EEG and deleted. The
analog audio monitor (MISC) measures each repetition's onset and the audio /
acquisition clock ratio; it is used for pairing only and is never published.
"""
from __future__ import annotations
import csv
import http.client
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
from scipy import signal
import soundfile as sf

DATASET='kymata_english';CONTENT='kymata_english_conversation'
PREFIX='/english-native-english-conversation/'
STIMULI=('stimulus.wav','mfa_word.txt','mfa_word_stime.txt','mfa_phoneme.txt','mfa_phoneme_stime.txt',
         'script_for_bbc_english_podcast_cutting/cuts.csv','script_for_bbc_english_podcast_cutting/reconstruct_from_csv.py')
BLOCK=60.;PRE=1000;CHUNK=10000;SEARCH=600;LINEAR_MS=4.

def catalog(workspace):
    """Audited English whitelist only; Russian, anatomy, visual and empty-room paths never appear."""
    audit=json.loads((Path(workspace)/'report/dataset_storage_audit_2026-10-13.json').read_text())[DATASET]
    files={f['path'].removeprefix(PREFIX):f for f in audit['files'] if f['path'].startswith(PREFIX)}
    runs={}
    for path in files:
        if path.endswith('_meg.fif'):
            if '_task-en_' not in path or '/meg/' not in path or 'split' in path:raise ValueError(f'Unexpected Kymata recording in whitelist: {path}')
            runs.setdefault(path.split('/')[0],[]).append(path)
    if len(runs)!=21 or sum(map(len,runs.values()))!=audit['task_en_fif_files']:raise ValueError('Kymata whitelist differs from audited English task-en catalog')
    return audit,files,{d:sorted(v) for d,v in sorted(runs.items())}

def person(directory):
    """sub-E3b is the second session of sub-E3, never a 21st participant."""
    return 'sub-E3' if directory=='sub-E3b' else directory

def fetch(item,path,attempts=8):
    """Audited OSF object. OSF throttles bursts, so wait and resume the .part file."""
    from ..storage import download_file
    for attempt in range(attempts):
        try:return download_file(item['download_url'],path,size=item['bytes'],sha256=item['sha256'])
        except (OSError,http.client.HTTPException) as error:
            if 'Insufficient disk space' in str(error) or attempt+1==attempts:raise
            time.sleep(min(30*2**attempt,600))

def tokens(labels,starts,duration,key):
    """MFA tier as released: one label and one start per line; a token ends where the next starts."""
    names=[v.strip() for v in Path(labels).read_text().splitlines() if v.strip()]
    times=[float(v) for v in Path(starts).read_text().split()]
    if len(names)!=len(times) or times!=sorted(times) or times[-1]>=duration:raise ValueError('Kymata annotation tier disagrees with its start times')
    return [{'start':a,'end':b,key:n} for n,a,b in zip(names,times,times[1:]+[duration]) if n!='<sp>' and b>a]

def envelope(x):
    """60–300 Hz magnitude smoothed to 25 Hz at 1 kHz; identical for audio and monitor."""
    band=signal.firwin(501,[60,300],pass_zero=False,fs=1000);smooth=signal.firwin(201,25,fs=1000)
    x=np.asarray(x,dtype=np.float64)
    return signal.fftconvolve(np.abs(signal.fftconvolve(x-x.mean(),band,mode='same')),smooth,mode='same')

def chunk_lag(monitor,reference,start,center):
    """Monitor sample (relative to the trigger) where reference[start:start+CHUNK] is played."""
    seg=reference[start:start+CHUNK];lo=PRE+start+center-SEARCH;cand=monitor[max(lo,0):lo+CHUNK+2*SEARCH]
    if lo<0 or len(seg)<CHUNK or len(cand)<CHUNK+2*SEARCH:return None
    xc=signal.correlate(cand-cand.mean(),seg-seg.mean(),mode='valid',method='fft');i=int(np.argmax(xc));fine=0.
    if 0<i<len(xc)-1:
        a,b,c=xc[i-1:i+2];fine=float(.5*(a-c)/(a-2*b+c)) if a-2*b+c else 0.
    r=np.corrcoef(cand[i:i+CHUNK],seg)[0,1]
    return {'audio_time':(start+CHUNK/2)/1000,'lag_ms':center+i+fine-SEARCH,'r':float(r) if np.isfinite(r) else 0.,'edge':i in (0,len(xc)-1)}

def clock_fit(monitor,reference):
    """Onset and drift of the played audio from the monitor alone, never from EEG or model scores.

    ``monitor`` is the 1 kHz envelope starting PRE samples before the repetition
    trigger; ``reference`` is the canonical audio envelope at 1 kHz. Ten-second
    chunks are tracked through the stimulus and a Theil-Sen line gives
    sample(audio second t) = trigger + offset + t*(1000 + drift). Clean playback
    stays within about 1 ms of that line. When playback slipped mid-stimulus the
    line misses by more than LINEAR_MS; the status is then ``measured-piecewise``
    and every content block is placed by ``block_lag`` at the within-block drift.
    """
    checks=[];center=0
    for start in range(0,len(reference)-CHUNK+1,CHUNK):
        q=chunk_lag(monitor,reference,start,center)
        if q is None:break
        checks.append(q)
        if q['r']>=.5 and not q['edge']:center=int(round(q['lag_ms']))
    good=[q for q in checks if q['r']>=.5 and not q['edge']]
    if len(good)<max(10,.8*len(checks)):raise ValueError(f'Audio monitor does not follow the stimulus: {len(good)}/{len(checks)} chunks')
    t=np.array([q['audio_time'] for q in good]);lag=np.array([q['lag_ms'] for q in good])
    i,j=np.triu_indices(len(t),1);slopes=(lag[j]-lag[i])/(t[j]-t[i]);drift=float(np.median(slopes));offset=float(np.median(lag-drift*t))
    residual=np.abs(lag-(offset+drift*t))
    return {'status':'measured-aligned' if residual.max()<=LINEAR_MS else 'measured-piecewise','offset_ms':offset,'drift_ms_per_second':drift,'clock_ratio':1+drift/1000,
            'local_drift_ms_per_second':float(np.median(slopes[t[j]-t[i]<=BLOCK])),
            'maximum_residual_seconds':float(residual.max())/1000,'median_residual_seconds':float(np.median(residual))/1000,
            'chunks':len(checks),'accepted_chunks':len(good),'minimum_r':float(min(q['r'] for q in good)),
            'covered_audio_seconds':float(t.max()+CHUNK/2000),'audio_monitor_excluded':True,'checks':checks}

def block_lag(sync,start,seconds):
    """Lag (ms) at the start of one content block from its own chunks; None if the block holds the slip."""
    rows=[q for q in sync['checks'] if q['r']>=.5 and not q['edge'] and start<=q['audio_time']<start+seconds]
    if len(rows)<3:return None
    values=np.array([q['lag_ms']-sync['local_drift_ms_per_second']*(q['audio_time']-start) for q in rows]);lag=float(np.median(values))
    return lag if np.abs(values-lag).max()<=LINEAR_MS else None

def repetitions(raw):
    """File samples of trigger 3 (audio repetition onset); button bits above 255 are ignored."""
    import mne
    events=mne.find_events(raw,stim_channel='STI101',mask=255,mask_type='and',min_duration=.002,shortest_event=1,verbose='ERROR')
    return [int(sample-raw.first_samp) for sample,_,code in events if code==3]

def monitor_onsets(raw,misc,reference,seconds=60,threshold=.5):
    """Repetition onsets without a trigger: where the first minute of the stimulus plays on the audio monitor.

    Candidates only; each still has to pass ``clock_fit`` over the whole stimulus.
    """
    head=reference[:seconds*1000]-reference[:seconds*1000].mean();best=[]
    for name in misc:
        a=envelope(raw.get_data(picks=[name])[0]);a=a-a.mean()
        if len(a)<=len(head) or not a.std():continue
        xc=signal.correlate(a,head,mode='valid',method='fft');energy=np.cumsum(np.r_[0,a*a])
        r=xc/(np.sqrt(np.maximum(energy[len(head):]-energy[:-len(head)],1e-30))*np.linalg.norm(head));found=[]
        while r.max()>=threshold:
            i=int(np.argmax(r));found.append(i);r[max(0,i-300000):i+300000]=-1
        if len(found)>len(best):best=sorted(found)
    return best

def stimulus(files,stage,root,meta,cleanup=True):
    """Canonical audio and released MFA tiers; source WAV is cleared once the FLAC reopens."""
    from ..storage import register_audio,dataset_dir
    from ...utils import read_json
    manifest=dataset_dir(DATASET,root)/'manifest.json'
    if manifest.exists() and CONTENT in read_json(manifest).get('contents',{}):return read_json(manifest)['contents'][CONTENT]
    folder=stage/'stimuli/audio';receipts={}
    for name in STIMULI:
        item=files['stimuli/audio/'+name];fetch(item,folder/name);receipts[name]={k:item[k] for k in ('path','bytes','sha256','download_url')}
        if name!='stimulus.wav':
            (meta/'stimuli').mkdir(parents=True,exist_ok=True);shutil.copy2(folder/name,meta/'stimuli'/Path(name).name)
    info=sf.info(folder/'stimulus.wav');duration=info.frames/info.samplerate
    words=tokens(folder/'mfa_word.txt',folder/'mfa_word_stime.txt',duration,'text')
    phones=tokens(folder/'mfa_phoneme.txt',folder/'mfa_phoneme_stime.txt',duration,'phone')
    content=register_audio(DATASET,CONTENT,folder/'stimulus.wav',words,phones,{'source':'https://osf.io/qdzgh/','sources':receipts,
        'narrative':'three-speaker English podcast conversation','annotation':'released Montreal Forced Aligner word/phoneme tiers',
        'unaligned_intervals':[[p['start'],p['end']] for p in phones if p['phone']=='spn'],
        'speaker_overlap_annotation':'not released by source; see configs/neurospeech_v2/content_masks.json',
        'concurrent_visual_stimulus':True},root=root)
    if cleanup:shutil.rmtree(stage/'stimuli')
    return content

def process_run(raw,recording,subject,reference,duration,sources,root,meta):
    """Publish every repetition of one FIF as guarded 60 s content blocks on the audio clock."""
    from ..storage import publish_record,geometry_positions
    types=raw.get_channel_types();montage=raw.get_montage();located=montage.get_positions()['ch_pos'] if montage is not None else {}
    names=[n for n,t in zip(raw.ch_names,types) if t=='eeg' and n in located and np.isfinite(located[n]).all()]
    if raw.info['sfreq']!=1000 or len(names)<60:raise ValueError(f'Unexpected Kymata EEG interface {recording}: {raw.info["sfreq"]} Hz, {len(names)} located EEG')
    rows=list(csv.DictReader((meta/f'{recording}_channels.tsv').open(encoding='utf-8-sig'),delimiter='\t'))
    bads=sorted({r['name'] for r in rows if r.get('status','good').lower()=='bad'}|set(raw.info['bads']))
    positions,geometry=geometry_positions(names,montage)
    geometry.update(positions_source='individual Polhemus digitization stored in the FIF',coordinate_quality='measured',
                    original_positions=np.round(geometry['original_positions'],5).tolist())
    picks=[raw.ch_names.index(n) for n in names];misc=[n for n,t in zip(raw.ch_names,types) if t=='misc']
    listed=[float(r['onset'])-raw.first_samp/1000 for r in csv.DictReader((meta/f'{recording}_events.tsv').open(encoding='utf-8-sig'),delimiter='\t') if r['trial_type']=='3']
    excluded={t:types.count(t) for t in sorted(set(types)) if t!='eeg'}
    records=[];summary=[];onsets=repetitions(raw);origin='STI101 value 3'
    if not onsets and misc:onsets=monitor_onsets(raw,misc,reference);origin='audio monitor search; the recording has no trigger 3'
    for k,trigger in enumerate(onsets,1):
        identity=f'{recording}_rep-{k}';stop=min(raw.n_times,trigger+round(duration*1000)+3000)
        evidence={'trigger_sample':trigger,'trigger_source':origin,'events_tsv_agreement_seconds':min((abs(v-trigger/1000) for v in listed),default=None)}
        try:
            if trigger<PRE or not misc:raise ValueError('No audio monitor support around the repetition trigger')
            head=raw.get_data(picks=misc,start=trigger-PRE,stop=trigger+CHUNK+SEARCH)
            scores=[(chunk_lag(envelope(v),reference,0,0) or {'r':0.})['r'] for v in head]
            channel=misc[int(np.argmax(scores))]
            sync=clock_fit(envelope(raw.get_data(picks=[channel],start=trigger-PRE,stop=stop)[0]),reference)
        except ValueError as error:
            eeg=raw.get_data(picks=picks,start=trigger,stop=min(raw.n_times,trigger+round(duration*1000)))*1e6
            (meta/'sync').mkdir(exist_ok=True)
            if misc and trigger>=PRE:np.save(meta/'sync'/f'{identity}_monitor.npy',raw.get_data(picks=misc,start=trigger-PRE,stop=stop).astype(np.float32))
            if eeg.shape[1]>=2000:records.extend(publish_record(DATASET,subject,identity,eeg,1000,names,positions,[{'audio':'unpaired-no-audio-target',
                'content_group':f'{DATASET}:unpaired:{identity}','duration':eeg.shape[1]/1000,'task':'quarantined','minimum_seconds':1}],bads,
                {**geometry,'task':'quarantined','condition':'quarantined_unaligned','failed_synchronization':str(error),**evidence,
                 'no_trainable_audio_pairing':True,'sources':sources,'units':'uV'},root))
            summary.append({'repetition':k,'status':'quarantined','reason':str(error)});continue
        linear=sync['status']=='measured-aligned';first=round(trigger+sync['offset_ms'])
        rate=1000+(sync['drift_ms_per_second'] if linear else sync['local_drift_ms_per_second'])
        eeg=raw.get_data(picks=picks,start=first,stop=min(raw.n_times,first+int(np.ceil(duration*rate))+2))*1e6 if linear else None
        available=min(duration,((eeg.shape[1] if linear else raw.n_times-first)-2)/rate)
        if sync['covered_audio_seconds']<duration-2*CHUNK/1000:available=min(available,sync['covered_audio_seconds'])
        checks=sync.pop('checks');(meta/'sync').mkdir(exist_ok=True)
        (meta/'sync'/f'{identity}.json').write_text(json.dumps({**sync,**evidence,'monitor_channel':channel,'checks':checks},indent=1))
        details={**geometry,'task':'perceived','condition':'clean','repetition':k,'synchronization':{**sync,**evidence,'monitor_channel':channel},
                 'recording_first_sample':first,'nominal_rate':1000.,'time_base':'canonical audio seconds; raw_rate is EEG samples per audio second',
                 'sources':sources,'units':'uV','unit_conversion':'FIF SI volts times 1e6','reference':'nose, re-referenced to valid-channel average',
                 'excluded_channel_types':excluded,'concurrent_visual_stimulus':True}
        published=0;slipped=[]
        for block,start in enumerate(np.arange(0,duration,BLOCK)):
            seconds=min(BLOCK,duration-start,available-start)
            if seconds<24:continue
            source,origin,note=eeg,float(start),{}
            if not linear:
                # Playback slipped: this block is cut again at its own measured onset.
                lag=block_lag({**sync,'checks':checks},start,seconds)
                if lag is None:slipped.append(block);continue
                begin=round(trigger+lag+start*1000);origin=0.;note={'block_lag_ms':lag,'recording_first_sample':begin}
                source=raw.get_data(picks=picks,start=begin,stop=min(raw.n_times,begin+int(np.ceil(seconds*rate))+2))*1e6
            records.extend(publish_record(DATASET,subject,identity,source,rate,names,positions,[{'audio':CONTENT,'eeg_start':origin,
                'audio_start':float(start),'duration':float(seconds),'content_group':f'{DATASET}:{CONTENT}:block-{block:02d}','minimum_seconds':24}],
                bads,{**details,**note,'content_block':block},root));published+=1
        summary.append({'repetition':k,'status':sync['status'],'blocks':published,'withheld_blocks':slipped,'available_audio_seconds':available,
                        **{key:sync[key] for key in ('offset_ms','drift_ms_per_second','maximum_residual_seconds','minimum_r')}})
    if not summary:
        eeg=raw.get_data(picks=picks)*1e6
        records.extend(publish_record(DATASET,subject,recording,eeg,1000,names,positions,[{'audio':'unpaired-no-audio-target',
            'content_group':f'{DATASET}:unpaired:{recording}','duration':eeg.shape[1]/1000,'task':'quarantined'}],bads,
            {**geometry,'task':'quarantined','condition':'quarantined_unaligned','failed_synchronization':'no repetition trigger',
             'no_trainable_audio_pairing':True,'sources':sources,'units':'uV'},root))
        summary.append({'repetition':None,'status':'quarantined','reason':'no repetition trigger'})
    return records,summary

def unsettled(marker):
    """Repetitions an earlier pass quarantined, or accepted on a line looser than LINEAR_MS."""
    return any(q['status']=='quarantined' or (q['status']=='measured-aligned' and q.get('maximum_residual_seconds',0)*1000>LINEAR_MS) for q in marker['repetitions'])

def run(root=None,subjects=None,cleanup=True,limit=None,retry_quarantined=False):
    from ..storage import (reclaim_sources,mark_dataset_complete,manifest_batch,source_metadata,verify_record,dataset_dir)
    from ...utils import atomic_json,read_json
    os.environ.setdefault('NUMBA_DISABLE_JIT','1')
    import mne
    from eegspeech import ROOT
    root=Path(root or ROOT/'data/processed/neurospeech');out=dataset_dir(DATASET,root);meta=out/'metadata';meta.mkdir(parents=True,exist_ok=True)
    stage=ROOT/'data/staging/neurospeech'/DATASET;audit,files,runs=catalog(ROOT)
    source_metadata(DATASET,'osf',{'source':audit['source'],'doi':'10.17605/OSF.IO/QDZGH','paper':'10.1038/s41597-026-06579-8','english_root':audit['english_root'],
        'license':'CC-BY 4.0 per data descriptor; dataset_description.json states CC0','whitelist_files':len(files),'whitelist_bytes':audit['selected_download_bytes'],
        'acquisition':'one audited FIF at a time; EEG extracted, FIF deleted after verified publication'},root)
    for name in ('README','dataset_description.json'):fetch(files[name],meta/name)
    content=stimulus(files,stage,root,meta,cleanup);duration=float(content['duration'])
    wave,rate=sf.read(out/content['path'],dtype='float64')
    reference=envelope(signal.resample_poly(wave,1,rate//1000));released=0;done=0
    for directory,paths in runs.items():
        if subjects and directory not in subjects and person(directory) not in subjects:continue
        folder=meta/directory;folder.mkdir(exist_ok=True);fetch(files[f'{directory}/meg/{directory}_coordsystem.json'],folder/f'{directory}_coordsystem.json')
        for path in paths:
            recording=Path(path).name.removesuffix('_meg.fif').replace('_task-en','');marker=folder/f'{recording}_complete.json'
            if marker.exists() and not (retry_quarantined and unsettled(read_json(marker))):
                for record in read_json(marker)['records']:verify_record(out/record['path'],hashes=True)
                continue
            if limit is not None and done>=limit:continue
            sources=[]
            for suffix in ('_channels.tsv','_events.tsv','_meg.json'):
                item=files[path.removesuffix('_meg.fif')+suffix];fetch(item,folder/f'{recording}{suffix}');sources.append({k:item[k] for k in ('path','bytes','sha256')})
            item=files[path];fif=stage/directory/'meg'/Path(path).name;fetch(item,fif)
            sources.append({k:item[k] for k in ('path','bytes','sha256','download_url')})
            raw=mne.io.read_raw_fif(fif,preload=False,allow_maxshield=True,verbose='ERROR')
            with manifest_batch(DATASET,root=root):
                records,summary=process_run(raw,recording,person(directory),reference,duration,sources,root,folder)
            raw.close()
            compact=[{k:r[k] for k in ('id','path','task','content_group','duration')} for r in records]
            atomic_json(marker,{'recording':recording,'subject':person(directory),'records':compact,'repetitions':summary,'sources':sources})
            if cleanup:
                released+=reclaim_sources([fif],DATASET,records,root=root)
                for parent in (fif.parent,fif.parent.parent):
                    if parent.exists() and not any(parent.iterdir()):parent.rmdir()
            done+=1
            print(json.dumps({'dataset':DATASET,'recording':recording,'records':len(records),'repetitions':summary,'released_bytes_total':released}),flush=True)
    finished=[person(d) for d,paths in runs.items() if all((meta/d/f'{Path(p).name.removesuffix("_meg.fif").replace("_task-en","")}_complete.json').exists() for p in paths)]
    complete=[s for s in set(finished) if finished.count(s)==sum(person(d)==s for d in runs)]
    markers=[read_json(p) for p in meta.glob('sub-*/*_complete.json')]
    return mark_dataset_complete(DATASET,complete,audit['unique_subjects_after_E3b_maps_to_E3'],root=root,details={
        'scope':'English task-en EEG only; MEG, EOG, ECG, triggers and audio monitor excluded','recordings':len(markers),
        'expected_recordings':audit['task_en_fif_files'],'released_bytes_this_run':released,
        'aligned_repetitions':sum(q['status'].startswith('measured') for m in markers for q in m['repetitions']),
        'piecewise_repetitions':sum(q['status']=='measured-piecewise' for m in markers for q in m['repetitions']),
        'quarantined_repetitions':sum(q['status']=='quarantined' for m in markers for q in m['repetitions']),
        'pairing_status':'monitor-measured onset and clock ratio per repetition; unmeasured repetitions preserved but excluded'})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--subject',action='append');p.add_argument('--limit',type=int);p.add_argument('--keep-raw',action='store_true');p.add_argument('--retry-quarantined',action='store_true')
    args=p.parse_args();print(json.dumps(run(subjects=args.subject,cleanup=not args.keep_raw,limit=args.limit,retry_quarantined=args.retry_quarantined)['completion']))
