"""Pinned NeMAR Brennan2019 acquisition with AUD-only synchronization QC."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np
from scipy.signal import correlate, resample_poly
import soundfile as sf

REVISION = "52c493c409dfb6cf87c7a1ef9f36d5eed500f9c7"
GITHUB = f"https://raw.githubusercontent.com/nemarDatasets/nm000180/{REVISION}/"
S3 = "https://nemar.s3.us-east-2.amazonaws.com/nm000180/objects/"

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()

def download(url,path,expected=None,size=None):
    """Resumable HTTPS payload, SHA256 validated before use."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and (size is None or path.stat().st_size==size) and (not expected or sha256(path)==expected):return path
    subprocess.run(["curl","--fail","--location","--retry","5","--retry-delay","2","--continue-at","-","--silent","--show-error",url,"-o",str(path)],check=True)
    if size is not None and path.stat().st_size!=size:raise ValueError(f"Wrong size {path}")
    if expected and sha256(path)!=expected:raise ValueError(f"Wrong SHA256 {path}")
    return path

def acquire(relative,path):
    """Resolve annex pointers using the pinned repository and official S3."""
    path=Path(path)
    receipt=path.with_name(path.name+'.source.json')
    if path.exists() and receipt.exists():
        previous=json.loads(receipt.read_text())
        if previous['revision']==REVISION and previous['sha256']==sha256(path):return path
    pointer_path=path.with_name(path.name+'.pointer')
    pointer=download(GITHUB+relative,pointer_path).read_bytes()
    match=re.search(rb"SHA256E-s(\d+)--([0-9a-f]{64})[^/\s]*",pointer) if len(pointer)<500 else None
    if match:
        key=match.group(0).decode()
        download(S3+key,path,match.group(2).decode(),int(match.group(1)))
    else:
        os.replace(pointer_path,path)
    pointer_path.unlink(missing_ok=True)
    receipt.write_text(json.dumps({'revision':REVISION,'source_url':GITHUB+relative,'sha256':sha256(path),'bytes':path.stat().st_size}))
    return path

def synchronization(aud,audio,rate,expected):
    """Measure three within-stimulus delays; timestamps never fitted to EEG scores."""
    y=resample_poly(audio,rate,44100) if rate==500 else resample_poly(audio,rate,44100)
    length=min(int(8*rate),len(y)//3); checks=[]
    for start in [0,max(0,len(y)//2-length//2),max(0,len(y)-length)]:
        reference=y[start:start+length]; lo=max(0,int(round(expected*rate))+start-int(.15*rate)); hi=min(len(aud),int(round(expected*rate))+start+length+int(.15*rate))
        candidate=aud[lo:hi]
        if len(candidate)<length:raise ValueError('Stimulus synchronization support extends outside source EEG')
        xc=correlate(candidate-candidate.mean(),reference-reference.mean(),mode="valid",method="fft")
        lag=int(np.argmax(abs(xc))); observed=(lo+lag-start)/rate
        r=float(np.corrcoef(candidate[lag:lag+length],reference)[0,1])
        checks.append({"audio_time":start/rate,"recording_start":observed,"r":r,"residual_seconds":observed-expected})
    starts=np.array([q["recording_start"] for q in checks]); rs=np.array([abs(q["r"]) for q in checks]); residual=float(np.max(abs(starts-np.median(starts))))
    if not np.isfinite(rs).all() or residual>.010 or np.min(rs)<.15:raise ValueError(f"AUD synchronization failed: {checks}")
    return {"status":"measured-aligned","source_start_seconds":expected,"measured_start_seconds":float(np.median(starts)),"source_clock_correction_seconds":float(np.median(starts)-expected),"maximum_residual_seconds":residual,"checks":checks,"audio_monitor_excluded":True}

def locate(aud,audio,rate,top=3):
    """Events-free candidate onsets: whole-stimulus normalized correlation over the entire AUD monitor.

    The re-release labels some recordings one segment late and omits events.tsv for
    others. Candidates still have to pass ``synchronization``; they are never trusted alone.
    """
    y=resample_poly(audio,rate,44100);y=y-y.mean();a=aud-aud.mean()
    if len(a)<len(y) or not a.std():return []
    xc=correlate(a,y,mode="valid",method="fft");energy=np.cumsum(np.r_[0,a*a])
    r=np.abs(xc)/(np.sqrt(np.maximum(energy[len(y):]-energy[:-len(y)],1e-20))*np.linalg.norm(y));found=[]
    for _ in range(top):
        i=int(np.argmax(r));found.append(i/rate);r[max(0,i-rate):i+rate]=0
    return found

def run(root=None,subjects=None,retry_quarantined=False):
    from ..storage import publish_record,register_audio,geometry_positions,mark_dataset_complete,reclaim_sources,manifest_batch
    os.environ.setdefault("NUMBA_DISABLE_JIT","1")
    import mne
    workspace=Path.cwd(); root=Path(root or workspace/"data/processed/neurospeech"); out=root/"brennan2019"; stage=workspace/"data/staging/neurospeech/brennan2019"; meta=out/"metadata";meta.mkdir(parents=True,exist_ok=True)
    tree=json.loads((meta/"source_tree.json").read_text())
    sourcepaths={q['path'] for q in tree['tree']}
    allsubjects=sorted({q["path"].split('/')[0] for q in tree['tree'] if q['path'].startswith('sub-')})
    annotations=list(csv.DictReader(acquire("sourcedata/AliceChapterOne-EEG.csv",meta/"AliceChapterOne-EEG.csv").open()))
    audio={};contents=[]
    registered=json.loads((out/'manifest.json').read_text()).get('contents',{}) if (out/'manifest.json').exists() else {}
    for segment in range(1,13):
        name=f"DownTheRabbitHoleFinal_SoundFile{segment}.wav"; p=acquire("stimuli/"+name,stage/"stimuli"/name); y,sr=sf.read(p)
        if sr!=44100 or y.ndim!=1:raise ValueError("Unexpected Brennan audio")
        audio[segment]=y; words=[]
        for q in annotations:
            if int(q['Segment'])!=segment:continue
            # The official CSV lists clitics ('s, 'd) as their own rows with the apostrophe stored as \x1a.
            if q['Word'].startswith('\x1a') and words:words[-1]['text']+="'"+q['Word'][1:];words[-1]['end']=float(q['offset']);continue
            words.append({"start":float(q['onset']),"end":float(q['offset']),"text":q['Word']})
        content=f"brennan_alice_{segment:02d}"
        # An unchanged stimulus keeps its entry, so tiers added later (forced-aligned phonemes) survive reader reruns.
        if registered.get(content,{}).get('source_sha256')==sha256(p) and registered[content]['words']==words:
            contents.append(registered[content]);p.unlink();continue
        receipt=register_audio("brennan2019",content,p,words=words,metadata={"source_url":S3,"revision":REVISION,"segment":segment,"narrative":"Alice chapter 1"},root=root)
        contents.append(receipt);p.unlink()
    records=[];release=0
    acquire("participants.tsv",meta/"participants.tsv");acquire("participants.json",meta/"participants.json")
    # Offset between the authors' trial table and the audio actually measured on AUD, over every verified stimulus.
    measured=[q['measured_start_seconds']-q['source_start_seconds'] for r in (json.loads((out/'manifest.json').read_text())['records'].values() if (out/'manifest.json').exists() else [])
              if r.get('task')=='perceived' for q in [r['metadata']['synchronization']] if q.get('status')=='measured-aligned' and q.get('onset_source','official events.tsv')=='official events.tsv']
    cohort={'median_seconds':float(np.median(measured)),'mad_seconds':float(np.median(np.abs(np.asarray(measured)-np.median(measured)))),'minimum_seconds':float(min(measured)),'maximum_seconds':float(max(measured)),'stimuli':len(measured)} if len(measured)>=100 else None
    for subject in subjects or allsubjects:
        marker=out/"metadata"/f"{subject}_complete.json"
        if marker.exists():
            saved=json.loads(marker.read_text())
            if not (retry_quarantined and saved.get('failed_segments')):records.extend(saved['records']);continue
        d=stage/subject;d.mkdir(parents=True,exist_ok=True)
        prefix=f"{subject}_task-alicelistening";source=[]
        for suffix in ["_eeg.eeg","_eeg.vhdr","_eeg.vmrk","_channels.tsv","_events.tsv","_space-CapTrak_electrodes.tsv","_space-CapTrak_coordsystem.json","_eeg.json"]:
            filename=(subject+suffix if suffix.startswith('_space') else prefix+suffix)
            if suffix=='_events.tsv' and f'{subject}/eeg/{filename}' not in sourcepaths:
                source.append({'path':filename,'missing_at_pinned_source':True,'revision':REVISION});continue
            p=acquire(f"{subject}/eeg/{filename}",d/filename);source.append({"path":filename,"sha256":sha256(p),"bytes":p.stat().st_size,"revision":REVISION})
            if p.suffix not in ['.eeg','.vhdr']:shutil.copy2(p,meta/f"{subject}_{filename}")
        raw=mne.io.read_raw_brainvision(d/(prefix+"_eeg.vhdr"),preload=False,verbose="ERROR")
        channels=list(csv.DictReader((d/(prefix+"_channels.tsv")).open(encoding='utf-8-sig'),delimiter='\t')); eegnames=[q['name'] for q in channels if q['type'].upper()=='EEG']; pick=[raw.ch_names.index(q) for q in eegnames]
        monitor=[i for i,q in enumerate(channels) if q['type'].upper() in ('MISC','AUDIO') or q['name'] in ('AUD','Aux5')]
        if len(pick)!=60 or len(monitor)!=1:raise ValueError(f"Unexpected Brennan channel roles {subject}")
        positionsrows={q['name']:q for q in csv.DictReader((d/(subject+"_space-CapTrak_electrodes.tsv")).open(encoding='utf-8-sig'),delimiter='\t')}
        coords=json.loads((d/(subject+"_space-CapTrak_coordsystem.json")).read_text());scale={'m':1,'cm':.01,'mm':.001}[coords['EEGCoordinateUnits']]
        positions=np.array([[float(positionsrows[n][v])*scale for v in ['x','y','z']] for n in eegnames])
        landmarks=coords['AnatomicalLandmarkCoordinates']; montage=mne.channels.make_dig_montage(ch_pos=dict(zip(eegnames,positions)),nasion=landmarks['NAS'],lpa=landmarks['LPA'],rpa=landmarks['RPA'],coord_frame='head')
        positions,geometry=geometry_positions(eegnames,montage)
        geometry.update(positions_source="source easyCAP-M10 cap template, not individual digitization",coordinate_quality="source-cap-template")
        aud=raw.get_data(picks=monitor)[0];eventpath=d/(prefix+'_events.tsv');events=list(csv.DictReader(eventpath.open(encoding='utf-8-sig'),delimiter='\t')) if eventpath.exists() else []; anno={int(q['Order']):q for q in annotations}; subjectrecords=[]
        def publish_segment(segment,sync):
            first=round(sync.get('measured_start_seconds',sync.get('applied_start_seconds'))*500);last=first+round(len(audio[segment])*500/44100)
            eeg=raw.get_data(picks=pick,start=first,stop=last)*1e6
            return publish_record("brennan2019",subject,f"segment-{segment:02d}",eeg,500,eegnames,positions,[{"audio":f"brennan_alice_{segment:02d}","eeg_start":0.,"audio_start":0.,"duration":eeg.shape[1]/500}],metadata={**geometry,"coordinate_system":coords,"task":"perceived","condition":"clean","synchronization":sync,"original_recording_start":first/500,"sources":source,"units":"uV","word_annotation_source":"official AliceChapterOne-EEG.csv","excluded_channels":[q['name'] for q in channels if q['name'] not in eegnames]},root=root)
        with manifest_batch('brennan2019',root=root):
            failed=[];previous=0.
            for segment in range(1,13):
                rows=[q for q in events if int(q['segment'])==segment]
                candidates=[(float(np.median([float(q['onset'])-float(anno[int(q['word_index'])]['onset']) for q in rows])),'official events.tsv')] if rows else []
                candidates+=[(t,'AUD whole-recording search') for t in locate(aud,audio[segment],500)]
                sync=None;reason='missing source word events'
                for expected,origin in candidates:
                    try:sync={**synchronization(aud,audio[segment],500,expected),'onset_source':origin};break
                    except (ValueError,IndexError) as exc:
                        if origin=='official events.tsv':reason=str(exc)
                if sync is not None and sync['measured_start_seconds']<previous-.05:sync=None;reason='AUD match precedes the previous stimulus'
                if sync is None:
                    failed.append({'segment':segment,'source_start_seconds':candidates[0][0] if rows else None,'reason':reason,'whole_recording_search':'no verified AUD match'});continue
                previous=sync['measured_start_seconds']+len(audio[segment])/44100;subjectrecords.extend(publish_segment(segment,sync))
            # No segment could be measured: the monitor carries no audio. The authors' trial table is accepted only
            # when it lists all twelve stimuli back to back in order, and is shifted by the cohort's measured offset.
            table={q['segment']:q['source_start_seconds'] for q in failed if q.get('source_start_seconds') is not None}
            if not subjectrecords and cohort and len(table)==12 and all(0<=table[k+1]-table[k]-len(audio[k])/44100<=.15 for k in range(1,12)):
                for segment in range(1,13):subjectrecords.extend(publish_segment(segment,{'status':'source-aligned','onset_source':'official events.tsv','source_start_seconds':table[segment],'applied_start_seconds':table[segment]+cohort['median_seconds'],'cohort_clock_correction':cohort,'evidence':"authors' trial table (proc.trl); twelve stimuli back to back in order; AUD monitor carries no audio",'audio_monitor_excluded':True}))
                failed=[]
            if failed:
                physical=raw.get_data(picks=pick)*1e6
                quarantined=publish_record('brennan2019',subject,'unaligned-complete-recording',physical,500,eegnames,positions,[{'audio':'unpaired-no-audio-target','content_group':f'brennan2019:unpaired:{subject}','duration':physical.shape[1]/500,'task':'quarantined'}],metadata={**geometry,'task':'quarantined','condition':'quarantined_unaligned','sources':source,'no_trainable_audio_pairing':True,'failed_synchronization':failed,'full_recording_preserved':True,'coordinate_system':coords},root=root)
                subjectrecords.extend(quarantined);(meta/f'{subject}_synchronization_failures.json').write_text(json.dumps(failed,indent=2))
        raw.close(); marker.write_text(json.dumps({"records":subjectrecords,"source":source,'quarantined':bool(failed),'failed_segments':len(failed)},indent=2));records.extend(subjectrecords)
        release+=reclaim_sources(list(d.iterdir()),"brennan2019",subjectrecords,root=root)
        d.rmdir();print(json.dumps({"dataset":"brennan2019","subject":subject,"records":len(subjectrecords),"released_bytes_total":release}),flush=True)
    markers=[json.loads(p.read_text()) for p in meta.glob('sub-*_complete.json')]
    return mark_dataset_complete("brennan2019",[p.name.removesuffix('_complete.json') for p in meta.glob('sub-*_complete.json')],45,root=root,details={"revision":REVISION,"released_bytes_this_run":release,'quarantined_subjects':sum(bool(m.get('quarantined')) for m in markers),'pairing_status':'AUD-verified segments only are trainable; failed records preserved but excluded'})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--subject',action='append');p.add_argument('--retry-quarantined',action='store_true')
    args=p.parse_args();run(subjects=args.subject,retry_quarantined=args.retry_quarantined)
