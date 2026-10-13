"""Migrate the checksum-verified corpus to independent content-safe EEG200."""
import csv
from pathlib import Path
import re
import shutil
import numpy as np
import soundfile as sf
from eegspeech import ROOT
from eegspeech.utils import read_json,digest,atomic_json
from eegspeech.perceived_data import textgrid_words
from ..storage import dataset_dir,register_audio,publish_record,geometry_positions,source_metadata,reclaim_sources,mark_dataset_complete,verify_record,verify_dataset

def ingest(raw=None,root=None,cleanup=True):
    import mne
    from praatio import textgrid
    raw=Path(raw or ROOT/'data/raw/ds004408');dataset='ds004408';dest=dataset_dir(dataset,root)
    if not (raw/'acquisition.json').exists() and (dest/'manifest.json').exists():
        saved=read_json(dest/'manifest.json')
        if saved['complete']:
            verify_dataset(dataset,root,hashes=True)
            return saved
    receipt=read_json(raw/'acquisition.json')
    if not receipt['complete'] or receipt['revision']!='b2d91784a90c18e1d76bd77dcb2b29b0ce27a0be':raise ValueError('Unverified raw source')
    original={r['path']:r for r in receipt['files']}
    source_metadata(dataset,'acquisition',receipt,root)
    audit=read_json(ROOT/'report/dataset_unit_audit.json')
    if not audit['source_matches'] or audit['header_unit_multiplier']!=1e-6:raise ValueError('Pinned source unit audit missing')
    source_metadata(dataset,'unit_audit',audit,root)
    for i in range(1,21):
        audio=f'audio{i:02d}';wave=raw/'stimuli'/f'{audio}.wav';tg=wave.with_suffix('.TextGrid')
        existing=read_json(dest/'manifest.json').get('contents',{}).get(audio)
        if existing:continue
        for path in [wave,tg]:
            expected=original[str(path.relative_to(raw))]
            if digest(path)!=expected['sha256']:raise ValueError(f'Source changed: {path}')
        words=textgrid_words(tg);grid=textgrid.openTextgrid(str(tg),includeEmptyIntervals=True)
        tiers=[n for n in grid.tierNames if any(s in n.lower() for s in ['phone','phoneme'])]
        phones=[{'start':float(a),'end':float(b),'phone':s} for a,b,s in grid.getTier(tiers[0]).entries if s.strip()] if len(tiers)==1 else []
        register_audio(dataset,audio,wave,words,phones,{'source_revision':receipt['revision'],'alignment':'source-aligned'},root)
        (dest/'metadata'/'textgrid').mkdir(parents=True,exist_ok=True)
        shutil.copy2(tg,dest/'metadata'/'textgrid'/tg.name)
    subjects=[]
    names=[f'{l}{n}' for l in 'ABCD' for n in range(1,33)]
    positions,geometry=geometry_positions(names)
    for s in range(1,20):
        subject=f'sub-{s:03d}';success=0
        for run in range(1,21):
            recording=f'{subject}_task-listening_run-{run:02d}'
            header=raw/subject/'eeg'/f'{recording}_eeg.vhdr'
            audio=f'audio{run:02d}'
            registry=read_json(dest/'manifest.json')
            rows=[r for r in registry.get('records',{}).values() if r['subject']==subject and r['recording']==recording]
            if rows:
                for row in rows:verify_record(dest/row['path'],True)
                success+=1;continue
            if not header.exists():raise FileNotFoundError(f'Missing unprocessed {header}')
            binary=header.with_suffix('.eeg');marker=header.with_suffix('.vmrk')
            for path in [header,binary,marker]:
                expected=original[str(path.relative_to(raw))]
                if digest(path)!=expected['sha256']:raise ValueError('Pinned raw checksum changed')
            channel_path=header.with_name(recording+'_channels.tsv')
            with channel_path.open(encoding='utf-8-sig') as handle:channels=list(csv.DictReader(handle,delimiter='\t'))
            scalp=[c['name'] for c in channels if c['type'].upper()=='EEG']
            if set(scalp)!=set(names):raise ValueError('Unexpected 128-channel montage')
            bads=[c['name'] for c in channels if c.get('status','good').lower()=='bad']
            source=mne.io.read_raw_brainvision(header,preload=False,verbose='ERROR').pick(names).reorder_channels(names)
            duration=registry['contents'][audio]['duration']
            if source.n_times/source.info['sfreq']<duration:raise ValueError('EEG shorter than mapped stimulus')
            physical=source.get_data()*1e6*audit['header_unit_multiplier']
            evidence={**geometry,'source_revision':receipt['revision'],'source_sha256':original[str(binary.relative_to(raw))]['sha256'],
                      'source_units':'BrainVision nominal V corrected using fixed corpus audit','scale_to_uV':1.,
                      'alignment':'source-aligned','mapping_evidence':'Pinned README: one run per corresponding stimulus, starts aligned',
                      'source_annotations':[{'onset':float(a['onset']),'duration':float(a['duration']),'description':str(a['description'])} for a in source.annotations]}
            rows=publish_record(dataset,subject,recording,physical,source.info['sfreq'],names,positions,
                [{'audio':audio,'eeg_start':0,'audio_start':0,'duration':duration}],bads,evidence,root)
            source.close();del physical
            if cleanup:reclaim_sources([binary,marker,header],dataset,rows,root)
            success+=1
            if run%5==0:print(f'ds004408 {subject}: {run}/20 recordings processed; raw reclaimed={cleanup}',flush=True)
        if success==20:subjects.append(subject)
    result=mark_dataset_complete(dataset,subjects,19,root,{'expected_recordings':380,'source_revision':receipt['revision']})
    if cleanup and result['complete']:
        verify_dataset(dataset,root,hashes=True)
        if not raw.resolve().is_relative_to((ROOT/'data/raw').resolve()):raise ValueError('Raw cleanup outside approved dataset root')
        # All EEG/audio metadata and canonical waveforms are now independently published.
        source_metadata(dataset,'final_raw_inventory',{'files':receipt['files']},root)
        total=0
        for path in raw.rglob('*'):
            if path.is_file():total+=path.stat().st_size;path.unlink()
        for path in sorted(raw.rglob('*'),key=lambda p:len(p.parts),reverse=True):
            if path.is_dir():path.rmdir()
        from eegspeech.utils import append_jsonl
        append_jsonl(dest/'cleanup.jsonl',{'cleared_completed_raw_root':str(raw),'remaining_released_bytes':total})
    return result
