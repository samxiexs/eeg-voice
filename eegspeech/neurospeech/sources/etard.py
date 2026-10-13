"""Zenodo7778289 English-only range acquisition and verified HDF5 publication.

ZIP members are range-fetched and CRC checked, avoiding a20GB resident archive.
Published archive MD5 values are preserved as source metadata; a partial range
acquisition cannot claim to have verified the full-archive MD5.
"""
from __future__ import annotations
import http.client
import io
import json
import hashlib
import os
from pathlib import Path
import struct
import time
from urllib.request import Request,urlopen
import zipfile
import zlib
import shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import soundfile as sf
from ..storage import (download_file,ensure_space,publish_record,register_audio,geometry_positions,
                       mark_dataset_complete,reclaim_sources,source_metadata,verify_record)
from ...utils import atomic_json,digest

RECORD='7778289'
ENGLISH=('clean','lb','mb','hb')


def response_range(url,start,stop,total):
    """Require genuine206 and exactContent-Range; never silently fetch wholeZIP."""
    request=Request(url,headers={'Range':f'bytes={start}-{stop-1}','User-Agent':'NeuroSpeech/2.1'})
    error=None
    for attempt in range(6):
        try:
            response=urlopen(request,timeout=150)
            if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{stop-1}/{total}':
                response.close();raise IOError('Source does not honor exact HTTP Range')
            return response
        except OSError as exc:
            error=exc
            if attempt<5:time.sleep(min(2**attempt,30))
    raise IOError(f'Cannot get source range {start}:{stop}: {error}')



def range_chunks(url,start,stop,total,block=16*1024*1024,parallel=4):
    """Four bounded ordered HTTP Range requests; CRC still covers full member.

    At most64MB compressed RAM is queued. Progress never writes a raw HDF5.
    """
    def fetch(first,last):
        # A stalled or dropped connection mid-chunk is retried; the member CRC still covers every byte.
        error=None
        for attempt in range(6):
            try:
                with response_range(url,first,last,total) as response:value=response.read()
                if len(value)==last-first:return value
                error=IOError('Truncated member range chunk')
            except (OSError,http.client.HTTPException) as exc:error=exc
            time.sleep(min(2**attempt,30))
        raise IOError(f'Cannot read source range {first}:{last}: {error}')
    ranges=iter((a,min(stop,a+block)) for a in range(start,stop,block))
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        pending=[]
        for _ in range(parallel):
            interval=next(ranges,None)
            if interval is not None:pending.append(pool.submit(fetch,*interval))
        while pending:
            first=pending.pop(0);value=first.result()
            interval=next(ranges,None)
            if interval is not None:pending.append(pool.submit(fetch,*interval))
            yield value

class RemoteZip(io.RawIOBase):
    def __init__(self,url,size):self.url,self.size,self.position=url,int(size),0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if self.position<0:raise ValueError('Negative ZIP position')
        return self.position
    def read(self,count=-1):
        stop=self.size if count<0 else min(self.size,self.position+count)
        if stop<=self.position:return b''
        with response_range(self.url,self.position,stop,self.size) as response:value=response.read()
        if len(value)!=stop-self.position:raise IOError('Incomplete ZIP directory range')
        self.position=stop;return value


def member_acquire(archive,member,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    info=member
    if path.exists():
        if path.stat().st_size==info.file_size:return {'path':path.name,'bytes':info.file_size,'sha256':digest(path),'zip_member':info.filename,'crc32':f'{info.CRC:08x}'}
        path.unlink()
    ensure_space(path.parent,info.file_size+100_000_000)
    start=info.header_offset
    with response_range(archive.url,start,start+30,archive.size) as response:header=response.read()
    signature,version,flags,method,mtime,mdate,crc,csize,usize,namelen,extra=struct.unpack('<IHHHHHIIIHH',header)
    if signature!=0x04034b50 or method!=info.compress_type:raise ValueError('ZIP local header mismatch')
    data_start=start+30+namelen+extra;stop=data_start+info.compress_size
    temporary=path.with_suffix(path.suffix+'.part');checksum=0;written=0
    decompressor=zlib.decompressobj(-15) if method==zipfile.ZIP_DEFLATED else None
    if method not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED):raise ValueError('Unsupported ZIP compression')
    with temporary.open('wb') as output:
        for chunk in range_chunks(archive.url,data_start,stop,archive.size):
            value=decompressor.decompress(chunk) if decompressor is not None else chunk
            checksum=zlib.crc32(value,checksum);output.write(value);written+=len(value)
        if decompressor is not None:
            value=decompressor.flush();checksum=zlib.crc32(value,checksum);output.write(value);written+=len(value)
            if not decompressor.eof:raise IOError('Incomplete compressed member')
    if written!=info.file_size or checksum&0xffffffff!=info.CRC:raise ValueError('ZIP memberCRC/size mismatch; source retained')
    os.replace(temporary,path)
    return {'path':path.name,'bytes':written,'sha256':digest(path),'zip_member':info.filename,'crc32':f'{info.CRC:08x}',
            'archive_url':archive.url,'archive_byte_range':[data_start,stop],'archive_full_md5_verified':False}



def member_memory(archive,info):
    """Verified transient RAM source; no HDF5 raw file exists or is deleted.

    English-session HDF5s also contain ignored Dutch conditions. Holding the
    author-required mixed container in RAM avoids deleting any source files.
    """
    start=info.header_offset
    with response_range(archive.url,start,start+30,archive.size) as response:header=response.read()
    fields=struct.unpack('<IHHHHHIIIHH',header);signature,method=fields[0],fields[3]
    if signature!=0x04034b50 or method!=info.compress_type:raise ValueError('ZIP local header mismatch')
    if method not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED):raise ValueError('Unsupported ZIP compression')
    data_start=start+30+fields[-2]+fields[-1];stop=data_start+info.compress_size
    decompressor=zlib.decompressobj(-15) if method==zipfile.ZIP_DEFLATED else None
    memory=io.BytesIO();checksum=0;sha=hashlib.sha256();written=0
    for chunk in range_chunks(archive.url,data_start,stop,archive.size):
        value=decompressor.decompress(chunk) if decompressor is not None else chunk
        checksum=zlib.crc32(value,checksum);sha.update(value);memory.write(value);written+=len(value)
    if decompressor is not None:
        value=decompressor.flush();checksum=zlib.crc32(value,checksum);sha.update(value);memory.write(value);written+=len(value)
        if not decompressor.eof:raise IOError('Incomplete compressed member')
    if written!=info.file_size or checksum&0xffffffff!=info.CRC:raise ValueError('ZIP memberCRC/size mismatch')
    memory.seek(0)
    return memory,{'bytes':written,'sha256':sha.hexdigest(),'zip_member':info.filename,'crc32':f'{info.CRC:08x}',
                   'archive_url':archive.url,'archive_byte_range':[data_start,stop],
                   'archive_full_md5_verified':False,'source_storage':'RAM-only; no rawHDF5 file created or deleted'}

def _audio_digest(wave):
    """Quantized normalized PCM identity; exact repeat detection after noise padding."""
    return hashlib.sha256(np.round(np.asarray(wave,dtype=float)*2**23).astype('<i4').tobytes()).hexdigest()


def run(root=None,subjects=None):
    import h5py,mne
    os.environ.setdefault('NUMBA_DISABLE_JIT','1')
    from eegspeech import ROOT
    root=Path(root or ROOT/'data/processed/neurospeech');out=root/'etard';metadata=out/'metadata';metadata.mkdir(parents=True,exist_ok=True)
    stage=ROOT/'data/staging/neurospeech/etard';stage.mkdir(parents=True,exist_ok=True)
    api=metadata/'zenodo_record.json'
    if not api.exists():
        cached=Path('/private/tmp/neurospeech-etard-metadata.json')
        if cached.exists():shutil.copy2(cached,api)
        else:download_file(f'https://zenodo.org/api/records/{RECORD}',api)
    manifest=json.loads(api.read_text());files={f['key']:f for f in manifest['files']}
    for name in ('session_info.json','align_data.py'):
        file=files[name];download_file(file['links']['self'],metadata/name,file['size'],md5=file['checksum'].split(':')[1])
    session=json.loads((metadata/'session_info.json').read_text());people=session['english_session_participants']
    if len(people)!=18:raise ValueError('Official English-session participants differ from locked18')
    if subjects and not set(subjects)<=set(people):raise ValueError('Non-English-session participant requested')
    remote={key:RemoteZip(files[key]['links']['self'],files[key]['size']) for key in ('eeg.zip','audiobooks.zip')}
    archives={key:zipfile.ZipFile(value) for key,value in remote.items()}
    catalog={key:[{'filename':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size,'crc32':f'{i.CRC:08x}','header_offset':i.header_offset} for i in value.infolist()] for key,value in archives.items()}
    atomic_json(metadata/'zip_member_catalog.json',catalog)
    source_metadata('etard','zenodo',{'doi':manifest['doi'],'record':RECORD,'version':manifest['metadata']['version'],
                     'license':manifest['metadata']['license'],'files':files,'acquisition':'HTTP Range ZIP memberCRC + SHA256, no fullZIPMD5verification'},root)
    # Only English story + babble streams. No Dutch, fM/fW or embeddedEnglishDutch.
    waves={};audio_receipts={};identities={};content_map={};audio_sources=[]
    existing=json.loads((out/'manifest.json').read_text()).get('contents',{}) if (out/'manifest.json').exists() else {}
    for condition in ENGLISH:
        for part in range(1,5):
            basename=f'part_{part}_story.wav'
            candidates=[i for i in archives['audiobooks.zip'].infolist() if i.filename.endswith(f'{condition}/{basename}')]
            if len(candidates)!=1:raise ValueError(f'Expected one English audio member {condition}/{basename}')
            previous=[v for v in existing.values() if v.get('metadata',{}).get('source_member')==candidates[0].filename]
            if previous:
                value=previous[0];sf.info(out/value['path'])
                content=value['id'];identities[value['metadata']['canonical_pcm_sha256']]=content
                waves[(condition,part)]=float(value['duration']);content_map[f'{condition}/part_{part}']=content
                audio_sources.append(value['metadata']['source_receipt']);continue
            source=stage/'audio'/condition/basename;receipt=member_acquire(remote['audiobooks.zip'],candidates[0],source)
            y,rate=sf.read(source,dtype='float32',always_2d=True);y=y.mean(1)
            if rate!=44100:raise ValueError('Official Etard source audio is not44100Hz')
            if condition!='clean':
                if len(y)<88200 or np.max(np.abs(y[:44100]))>1e-7 or np.max(np.abs(y[-44100:]))>1e-7:
                    raise ValueError('English noisy story is missing author-declared1second zero padding')
                y=y[44100:-44100]
            key=_audio_digest(y)
            if key in identities:content=identities[key]
            else:
                content=f'etard_english_{condition}_part{part}';identities[key]=content
                # Reuse only canonical cropped speech after proving duplicate content identity.
                cropped=stage/'audio'/f'{content}.wav';sf.write(cropped,y,rate,subtype='FLOAT')
                audio_receipts[content]=register_audio('etard',content,cropped,metadata={'doi':manifest['doi'],'source_member':candidates[0].filename,
                  'condition':condition,'part':part,'trim_source_seconds':[1,1] if condition!='clean' else [0,0],
                  'annotation_status':'transcript-not-released-in-aligned-source; independent reference required beforeWER',
                  'source_receipt':receipt,'canonical_pcm_sha256':key},root=root)
                cropped.unlink()
            waves[(condition,part)]=len(y)/44100;content_map[f'{condition}/part_{part}']=content;audio_sources.append(receipt);source.unlink()
    atomic_json(metadata/'english_content_map.json',{'mapping':content_map,'unique_contents':len(identities),'identity_method':'exact after official noisepad removal normalized23bitPCM',
                'clean_noise_shared_content':[{k:v for k,v in content_map.items() if v==content} for content in identities.values()]})
    atomic_json(metadata/'audio_source_receipts.json',audio_sources)
    released=0
    for person in subjects or people:
        marker=metadata/f'{person}_complete.json'
        if marker.exists():
            prior=json.loads(marker.read_text())
            for record in prior['records']:verify_record(out/record['path'],hashes=True)
            print(json.dumps({'dataset':'etard','subject':person,'status':'already-verified'}),flush=True);continue
        members=[i for i in archives['eeg.zip'].infolist() if Path(i.filename).name==person+'.h5']
        if len(members)!=1:raise ValueError(f'Expected one HDF5 member for {person}')
        source_memory,source_receipt=member_memory(remote['eeg.zip'],members[0])
        records=[];quarantine=[]
        with h5py.File(source_memory,'r') as hdf:
            names=[n.decode() if isinstance(n,bytes) else str(n) for n in hdf['ch_names'][:]]
            rate=int(np.asarray(hdf['srate']).squeeze())
            if rate!=1000:raise ValueError('Official Etard alignedEEG rate differs1000Hz')
            # Source already droppedSound; select actual cap channels by standard nomenclature.
            montage=mne.channels.make_standard_montage('standard_1005');available=montage.get_positions()['ch_pos']
            eegnames=[n for n in names if n in available and not any(k in n.upper() for k in ('EOG','EMG','ECG','SOUND','AUD','MIC','STIM'))]
            if len(eegnames)<60:raise ValueError(f'Unexpected Etard scalpEEG roles: {names}')
            pick=[names.index(n) for n in eegnames];positions,geometry=geometry_positions(eegnames,montage)
            for condition in ENGLISH:
                for part in range(1,5):
                    dataset=hdf[f'{condition}/part_{part}'];x=dataset[pick,:].astype(np.float64)*1e6
                    if condition!='clean':x=x[:,1000:-1000]
                    duration=waves[(condition,part)]
                    difference=x.shape[1]/rate-duration
                    # Author aligns starts and usesfloor(duration)+1sample. End mismatch≤2msaccepted.
                    sync={'status':'source-aligned','evidence':'official align_data.py Sound crosscorrelation softwaretrigger difference<=5ms',
                          'duration_error_seconds':difference,'removed_eeg_edge_seconds':[1,1] if condition!='clean' else [0,0],
                          'removed_audio_edge_seconds':[1,1] if condition!='clean' else [0,0],'independent_monitor_in_release':False,
                          'source_audio_monitor_excluded':True}
                    if abs(difference)>.010:
                        quarantine.append({'condition':condition,'part':part,'duration_error_seconds':difference});task='quarantined'
                    else:task='perceived'
                    content=content_map[f'{condition}/part_{part}'];seconds=min(duration,x.shape[1]/rate)
                    record=publish_record('etard',person,f'{condition}-part-{part}',x,rate,eegnames,positions,
                      [{'audio':content,'duration':seconds,'task':task,'content_group':f'etard:{content}','block_seconds':60,'minimum_seconds':24}],
                      metadata={**geometry,'task':task,'condition':condition,'part':part,'sources':[source_receipt],
                                'units':'uV','unit_conversion':'official mne.get_data SIvolts times1e6','synchronization':sync,
                                'excluded_channels':[n for n in names if n not in eegnames],'audio_content_id':content},root=root)
                    records.extend(record)
        atomic_json(marker,{'subject':person,'records':records,'source_receipt':source_receipt,'quarantine':quarantine})
        for record in records:verify_record(out/record['path'],hashes=True)
        source_memory.close()
        released+=source_receipt['bytes']
        print(json.dumps({'dataset':'etard','subject':person,'records':len(records),'quarantine':len(quarantine),'RAM_source_bytes_processed_total':released}),flush=True)
    for archive in archives.values():archive.close()
    completed=[p.name.removesuffix('_complete.json') for p in metadata.glob('P*_complete.json')]
    for directory in sorted(stage.rglob('*'),reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):directory.rmdir()
    return mark_dataset_complete('etard',completed,18,root=root,details={'scope':'English clean/lb/mb/hb only','content_map':content_map,
      'RAM_source_bytes_processed_this_run':released,'full_archive_downloaded':False,'English_HDF5_member_may_contain_ignoredDutch':True,
      'transcripts':'Notreleased; missingreferencesexplicitly blockWER/oraclegates'})

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--subject',action='append');args=parser.parse_args();run(subjects=args.subject)
