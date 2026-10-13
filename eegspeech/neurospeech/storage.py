"""Source-independent physical EEG storage and verified raw reclamation.

Only source readers know original formats. This module stores unnormalized
physical units, independent content/filter boundaries and calibration blocks.
Folds fit their own statistics later; no full raw corpus is kept permanently.
"""
from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np
from scipy import signal
import soundfile as sf

from eegspeech import ROOT
from eegspeech.utils import atomic_json, digest, read_json, append_jsonl, fingerprint

PROCESSED = ROOT / "data/processed/neurospeech"
STAGING = ROOT / "data/staging/neurospeech"
PROCESSING = {"version": "neurospeech-preprocessing-v2.1", "eeg_rate":200,
              "neural_rate":64,"highpass":.5,"lowpass":40.,"neural_lowpass":15.,
              "fir_seconds":8.,"guard_seconds":10.,"units":"uV"}
_BATCHES = {}


def _empty_manifest(dataset):
    return {"version":"neurospeech-v2.1","dataset":dataset,"records":{},
            "contents":{},"calibration":{},"source_metadata":{},"complete":False}


@contextmanager
def manifest_batch(dataset,root=None):
    """Publish one atomic catalog per bounded source batch, including failures.

    Individual EEG files are still verified and published atomically. Holding
    the dataset lock avoids lost concurrent writers while avoiding quadratic
    JSON rewrites for thousands of short trials.
    """
    folder=dataset_dir(dataset,root);key=str(folder.resolve())
    if key in _BATCHES:
        yield
        return
    with manifest_lock(folder):
        path=folder/'manifest.json'
        _BATCHES[key]=read_json(path) if path.exists() else _empty_manifest(dataset)
        try:yield
        finally:atomic_json(path,_BATCHES.pop(key))


def safe_id(value):
    value = str(value)
    if not value or value in {".",".."}:
        raise ValueError("Empty/unsafe sample identifier")
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", value)


def dataset_dir(dataset, root=None):
    return Path(root or PROCESSED) / safe_id(dataset)


@contextmanager
def manifest_lock(folder):
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/".manifest.lock").open("a") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(handle,fcntl.LOCK_UN)


def update_manifest(dataset, key, value, root=None):
    folder=dataset_dir(dataset,root)
    active=_BATCHES.get(str(folder.resolve()))
    if active is not None:
        collection,identity=key;active.setdefault(collection,{})[identity]=value
        return
    with manifest_lock(folder):
        path=folder/"manifest.json"
        manifest=read_json(path) if path.exists() else _empty_manifest(dataset)
        collection,identity=key
        manifest.setdefault(collection,{})[identity]=value
        atomic_json(path,manifest)


def mark_dataset_complete(dataset, subjects, expected, root=None, details=None):
    folder=dataset_dir(dataset,root)
    with manifest_lock(folder):
        path=folder/"manifest.json"; value=read_json(path)
        value.update(subjects=sorted(set(subjects)), expected_subjects=int(expected),
                     complete=len(set(subjects))==expected, completion=details or {})
        atomic_json(path,value)
    return value


def ensure_space(path, required_bytes, reserve_bytes=5_000_000_000):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    free=shutil.disk_usage(path).free
    if free < required_bytes+reserve_bytes:
        raise OSError(f"Insufficient disk space: need {(required_bytes+reserve_bytes)/1e9:.2f} GB; have {free/1e9:.2f} GB")


def download_file(url, path, size=None, sha256=None, md5=None, retries=5):
    """Bounded, resumable HTTP download; validate bytes before publication."""
    if urlsplit(url).scheme != "https": raise ValueError("HTTPS source required")
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if size is not None and path.stat().st_size!=size:
            raise ValueError(f"Existing source size mismatch: {path}")
        if sha256 and digest(path)!=sha256: raise ValueError(f"Source SHA256 mismatch: {path}")
        if md5 and digest(path,"md5")!=md5: raise ValueError(f"Source MD5 mismatch: {path}")
        return {"url":url,"bytes":path.stat().st_size,"sha256":digest(path)}
    partial=path.with_name(path.name+".part")
    ensure_space(path.parent,max(0,(size or 2_000_000_000)-(partial.stat().st_size if partial.exists() else 0)))
    error=None
    for attempt in range(retries):
        try:
            have=partial.stat().st_size if partial.exists() else 0
            headers={"User-Agent":"NeuroSpeech/2.1"}
            if have:headers["Range"]=f"bytes={have}-"
            with urlopen(Request(url,headers=headers),timeout=90) as response:
                if have and response.status!=206:have=0
                if have and not response.headers.get("Content-Range","").startswith(f"bytes {have}-"):
                    raise IOError("Invalid HTTP resume range")
                with partial.open("ab" if have else "wb") as output:
                    while block:=response.read(1<<20): output.write(block)
            if size is not None and partial.stat().st_size!=size:raise IOError("Incomplete source download")
            if sha256 and digest(partial)!=sha256:raise ValueError("Source SHA256 mismatch")
            if md5 and digest(partial,"md5")!=md5:raise ValueError("Source MD5 mismatch")
            os.replace(partial,path)
            return {"url":url,"bytes":path.stat().st_size,"sha256":digest(path)}
        except ValueError:
            partial.unlink(missing_ok=True);raise
        except (OSError,TimeoutError) as exc:
            error=exc
            if attempt+1<retries:time.sleep(min(2**attempt,15))
    raise IOError(f"Cannot acquire {url}: {error}")


def source_metadata(dataset, identity, value, root=None):
    update_manifest(dataset,("source_metadata",identity),value,root)


def geometry_positions(names, montage=None, frame="native"):
    """Coordinates in REVE's standard1005 native-meter frame, with evidence.

    Measured head coordinates are transformed, never matched by array index.
    Original head/native positions and both fiducial transforms are retained.
    """
    import mne
    from mne.transforms import apply_trans, invert_transform
    if montage is None:
        kind="biosemi128" if set(names)=={f"{l}{n}" for l in "ABCD" for n in range(1,33)} else "standard_1005"
        montage=mne.channels.make_standard_montage(kind)
        source=f"MNE {kind} template"
    else:source="source-provided montage"
    positions=montage.get_positions(); coords=positions["ch_pos"]
    missing=[n for n in names if n not in coords]
    if missing:raise ValueError(f"No electrode coordinates for {missing}")
    xyz=np.asarray([coords[n] for n in names],dtype=float)
    actual_frame=positions.get("coord_frame",frame)
    native_to_head=mne.channels.compute_native_head_t(montage)
    head=xyz if actual_frame=="head" else apply_trans(native_to_head,xyz)
    standard=mne.channels.make_standard_montage("standard_1005")
    canonical_to_head=mne.channels.compute_native_head_t(standard)
    reve=apply_trans(invert_transform(canonical_to_head),head)
    if not np.isfinite(reve).all() or np.linalg.norm(reve,axis=1).max()>.3:
        raise ValueError("Invalid electrode geometry; expected meters")
    return reve.astype(np.float32),{"positions_source":source,"positions_frame":"standard1005-native",
        "positions_units":"m","original_positions":xyz.tolist(),"original_frame":actual_frame,
        "source_native_to_head":native_to_head["trans"].tolist(),
        "canonical_native_to_head":canonical_to_head["trans"].tolist()}


def _filter_centered(x,kernel):
    half=len(kernel)//2
    if x.shape[1]<2:raise ValueError("EEG region too short")
    padded=np.pad(x,((0,0),(half,half)),mode="reflect")
    return signal.fftconvolve(padded,kernel[None],mode="valid",axes=-1)


def physical_preprocess(eeg_uv,raw_rate,known_good, profile=None):
    p=profile or PROCESSING
    x=np.asarray(eeg_uv,dtype=np.float64).copy()
    good=np.asarray(known_good,dtype=bool)
    if x.ndim!=2 or x.shape[0]!=len(good) or not good.any():raise ValueError("Invalid EEG/channel contract")
    if not np.isfinite(x).all():raise ValueError("Nonfinite source EEG")
    x-=x[good].mean(0,keepdims=True);x[~good]=0
    taps=round(p["fir_seconds"]*raw_rate)|1
    kernel=signal.firwin(taps,[p["highpass"],p["lowpass"]],pass_zero=False,fs=raw_rate,window="hamming")
    ratio=Fraction(p["eeg_rate"]/raw_rate).limit_denominator(10000)
    main=signal.resample_poly(_filter_centered(x,kernel),ratio.numerator,ratio.denominator,axis=-1)
    nk=signal.firwin(p["eeg_rate"]+1,p["neural_lowpass"],fs=p["eeg_rate"],window="hamming")
    nr=Fraction(p["neural_rate"]/p["eeg_rate"]).limit_denominator(10000)
    low=signal.resample_poly(_filter_centered(main,nk),nr.numerator,nr.denominator,axis=-1)
    return main.astype(np.float32),low.astype(np.float32),{
        **p,"main_kernel_samples":taps,"main_support_seconds":(taps-1)/raw_rate,
        "main_kernel_sha256":hashlib.sha256(kernel.tobytes()).hexdigest(),
        "neural_kernel_samples":len(nk),"neural_kernel_sha256":hashlib.sha256(nk.tobytes()).hexdigest(),
        "boundary":"reflect only inside current canonical content region",
        "resample_ratio_main":[ratio.numerator,ratio.denominator],"resample_ratio_neural":[nr.numerator,nr.denominator]}


def _write_record(folder,main,low,meta):
    folder.mkdir(parents=True,exist_ok=True)
    for name,array in [("eeg.npy",main),("neural.npy",low)]:
        temporary=folder/(name+".part")
        with temporary.open("wb") as handle:np.save(handle,array,allow_pickle=False)
        os.replace(temporary,folder/name)
    meta["files"]={name:{"sha256":digest(folder/name),"bytes":(folder/name).stat().st_size}
                   for name in ["eeg.npy","neural.npy"]}
    atomic_json(folder/"meta.json",meta)
    verify_record(folder,hashes=True)
    return meta


def verify_record(folder,hashes=False):
    folder=Path(folder);meta=read_json(folder/"meta.json")
    for name,rate in [("eeg.npy",200),("neural.npy",64)]:
        value=np.load(folder/name,mmap_mode="r",allow_pickle=False)
        expected=round(meta["duration"]*rate)
        if value.ndim!=2 or value.shape[0]!=len(meta["names"]) or abs(value.shape[1]-expected)>1:
            raise ValueError(f"Processed shape mismatch {folder/name}: {value.shape}")
        for a in range(0,value.shape[1],30000):
            if not np.isfinite(value[:,a:a+30000]).all():raise ValueError("Nonfinite processed EEG")
        if hashes and digest(folder/name)!=meta["files"][name]["sha256"]:raise ValueError("Processed checksum mismatch")
    if len(meta["positions"])!=len(meta["names"]):raise ValueError("Electrode coordinate count mismatch")
    return True


def publish_record(dataset,subject,recording,eeg_uv,raw_rate,names,positions,segments,
                   bads=None,metadata=None,root=None):
    """Publish separate filter-support-safe content blocks, then source can clear.

    Segment times are relative to the supplied raw physical EEG. Audio identity
    and audio_start refer to one canonical processed waveform. Whole canonical
    stimuli or fixed 60sec Kymata blocks are processed independently.
    """
    folder=dataset_dir(dataset,root); meta_extra=dict(metadata or {})
    x=np.asarray(eeg_uv); positions=np.asarray(positions,dtype=np.float32)
    if x.ndim!=2 or x.shape[0]!=len(names) or positions.shape!=(len(names),3):raise ValueError("Source dimensions disagree")
    known=np.array([n not in set(bads or []) for n in names])
    published=[]
    for seg in segments:
        seconds=float(seg["duration"]);audio=str(seg.get("audio",seg.get("content_id")))
        if audio=="None":raise ValueError("Canonical audio required")
        eeg_start=float(seg.get("eeg_start",0)); audio_start=float(seg.get("audio_start",0))
        block=float(seg.get("block_seconds",seconds))
        for local in np.arange(0,seconds-1e-7,block):
            duration=min(block,seconds-local)
            if duration<float(seg.get("minimum_seconds",24)):continue
            first=round((eeg_start+local)*raw_rate);stop=first+round(duration*raw_rate)
            if first<0 or stop>x.shape[1]+1:raise ValueError("EEG/audio segment extends outside recording")
            core=x[:,first:min(stop,x.shape[1])]
            duration=core.shape[1]/raw_rate
            content_group=seg.get("content_group") or f"{dataset}:{audio}"
            if block<seconds:content_group=f"{dataset}:{audio}:block-{audio_start+local:.3f}-{audio_start+local+duration:.3f}"
            identity=f"{safe_id(recording)}_{safe_id(content_group)}"
            dest=folder/"records"/safe_id(subject)/identity
            if (dest/"meta.json").exists():
                saved=read_json(dest/"meta.json");verify_record(dest,hashes=True)
                if saved["processing_version"]!=PROCESSING["version"]:raise ValueError("Stale preprocessing")
            else:
                main,low,kernels=physical_preprocess(core,raw_rate,known)
                saved={"dataset":dataset,"subject":str(subject),"recording":str(recording),
                       "id":f"{dataset}:{subject}:{identity}","audio":audio,"content_group":content_group,
                       "audio_start":audio_start+float(local),"duration":duration,"offset":0.,
                       "raw_rate":float(raw_rate),"source_eeg_first":first,"source_eeg_stop":stop,
                       "names":list(names),"positions":positions.tolist(),"good_channels":known.tolist(),
                       "units":"uV","kernels":kernels,"processing_version":PROCESSING["version"],
                       "processing_hash":fingerprint(PROCESSING),"path":str(dest.relative_to(folder)),
                       "metadata":meta_extra,"task":seg.get("task",meta_extra.get("task","perceived"))}
                saved=_write_record(dest,main,low,saved)
            update_manifest(dataset,("records",saved["id"]),saved,root);published.append(saved)
        # Independent exactly60sec support, before any raw source reclamation.
        # Use10..70 when possible, otherwise a complete60sec source interval.
        if seconds>=60 and seg.get("task",meta_extra.get("task","perceived"))=="perceived":
            key=f"{dataset}:{subject}:{audio}"
            dest=folder/"calibration"/safe_id(subject)/safe_id(audio)
            if not (dest/"meta.json").exists():
                calibration_start=min(10.,seconds-60.)
                first=round((eeg_start+calibration_start)*raw_rate);stop=first+round(60*raw_rate)
                main,low,kernels=physical_preprocess(x[:,first:stop],raw_rate,known)
                value={"dataset":dataset,"subject":str(subject),"recording":str(recording),"audio":audio,
                       "audio_start":audio_start+calibration_start,"duration":60.,"raw_rate":float(raw_rate),
                       "source_eeg_first":first,"source_eeg_stop":stop,"raw_seconds":60.,
                       "names":list(names),"positions":positions.tolist(),"good_channels":known.tolist(),
                       "units":"uV","kernels":kernels,"metadata":meta_extra,
                       "filter_uses_only_calibration_block":True,"processing_version":PROCESSING["version"],
                       "path":str(dest.relative_to(folder))}
                value=_write_record(dest,main,low,value)
                update_manifest(dataset,("calibration",key),value,root)
    if not published:raise ValueError("No publishable source segments; raw must remain")
    return published


def register_audio(dataset,content_id,path,words=None,phonemes=None,metadata=None,root=None):
    folder=dataset_dir(dataset,root); identity=safe_id(content_id)
    destination=folder/"audio"/(identity+".flac");destination.parent.mkdir(parents=True,exist_ok=True)
    source=Path(path); source_sha=digest(source)
    wave,rate=sf.read(source,dtype="float32",always_2d=True)
    wave=wave.mean(1)
    ratio=Fraction(24000/int(rate)).limit_denominator(100000)
    if rate!=24000:wave=signal.resample_poly(wave,ratio.numerator,ratio.denominator).astype(np.float32)
    if not len(wave) or not np.isfinite(wave).all():raise ValueError("Invalid canonical waveform")
    if np.max(np.abs(wave))>1.001:raise ValueError("Audio outside normalized PCM range; source scaling audit required")
    temp=destination.with_suffix(".part.flac")
    sf.write(temp,wave,24000,subtype="PCM_24")
    decoded,sr=sf.read(temp,dtype="float32")
    if sr!=24000 or decoded.shape!=wave.shape or np.max(np.abs(decoded-wave))>2e-7:
        raise ValueError("Processed audio numerical contract failed")
    os.replace(temp,destination)
    value={"dataset":dataset,"id":str(content_id),"path":str(destination.relative_to(folder)),
           "sample_rate":24000,"duration":len(wave)/24000,"sha256":digest(destination),
           "source_sha256":source_sha,"source_rate":int(rate),"source_path":source.name,
           "resample_ratio":[ratio.numerator,ratio.denominator],"words":words or [],
           "phonemes":phonemes or [],"metadata":metadata or {},"processing_version":PROCESSING["version"]}
    update_manifest(dataset,("contents",str(content_id)),value,root)
    return value


def reclaim_sources(paths,dataset,published,root=None,allowed_roots=None):
    """Delete only explicitly supplied sources after reopening ALL outputs."""
    folder=dataset_dir(dataset,root)
    for record in published:verify_record(folder/record["path"],hashes=True)
    if not published:raise ValueError("Cannot clear sources without verified processed records")
    allowed=[Path(p).resolve() for p in (allowed_roots or [STAGING,ROOT/"data/raw"])]
    cleared=[]
    for path in paths:
        path=Path(path);resolved=path.resolve()
        if not any(resolved.is_relative_to(r) for r in allowed):raise ValueError(f"Source cleanup escaped staging: {path}")
        if path.is_symlink():raise ValueError("Refuse source cleanup through symlink")
        if path.is_file():
            size=path.stat().st_size;path.unlink();cleared.append({"path":str(path),"bytes":size})
    append_jsonl(folder/"cleanup.jsonl",{"at_unix":time.time(),"cleared":cleared,"verified_records":[r.get("id",r.get("recording")) for r in published]})
    return sum(r["bytes"] for r in cleared)


def verify_dataset(dataset,root=None,hashes=True):
    """Reopen all persistent EEG, calibration blocks and canonical audio."""
    folder=dataset_dir(dataset,root);manifest=read_json(folder/'manifest.json')
    for collection in ('records','calibration'):
        for row in manifest.get(collection,{}).values():
            verify_record(folder/row['path'],hashes)
            if row.get('task','perceived')=='perceived' and row['audio'] not in manifest['contents']:
                raise ValueError('Processed EEG lacks its canonical audio')
    for row in manifest.get('contents',{}).values():
        path=folder/row['path'];info=sf.info(path)
        if info.samplerate!=24000 or info.channels!=1 or abs(info.duration-row['duration'])>1/24000:
            raise ValueError('Processed audio format/duration mismatch')
        if hashes and digest(path)!=row['sha256']:raise ValueError('Processed audio checksum mismatch')
    return {'dataset':dataset,'records':len(manifest['records']),
            'contents':len(manifest['contents']),'calibration':len(manifest['calibration']),
            'complete':manifest['complete']}
