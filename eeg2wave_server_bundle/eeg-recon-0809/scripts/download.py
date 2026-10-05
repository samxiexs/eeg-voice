"""Download the public datasets this project uses (only the files it needs).

    python scripts/download.py thinking_out_loud      # OpenNeuro ds003626 derivatives (7.2 GB)
    python scripts/download.py cpseed                 # OpenNeuro ds006465 session-level preproc .mat (4.0 GB)
    python scripts/download.py bci2020                # OSF pq7vb, Track 3 imagined speech (2.3 GB)
    python scripts/download.py sparrkulee             # KU Leuven RDR: preprocessed EEG + stimulus audio (29.5 GB)
    python scripts/download.py chisco --subject 01    # OpenNeuro ds005170 preprocessed fif, one subject (~12 GB)
    python scripts/download.py karaone --subject MM05 # Toronto KaraOne raw archive, one person (1.3-2.4 GB)
    python scripts/download.py models                 # SpeechT5 HiFi-GAN, HuBERT-base, Whisper-small -> models/ (1.4 GB)

Files land in ``data/raw/<dataset>/`` keeping the source layout.  Downloads resume
(``.part`` files) and every file is checked against the size the source reports.
The ``prepare_*`` scripts convert these trees into ``artifacts/store`` and the raw
trees can then be deleted; they are only needed to rebuild the store.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fnmatch
from pathlib import Path
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
S3 = 'https://s3.amazonaws.com/openneuro.org'
RDR = 'https://rdr.kuleuven.be'
SPARRKULEE_DOI = 'doi:10.48804/K3VSND'
OSF_NODE = 'pq7vb'
SESSION = requests.Session()
SESSION.headers['User-Agent'] = 'eeg-recon-downloader/1.0 (research use)'


def s3_list(prefix):
    """(key, size) of every OpenNeuro object under ``prefix``."""
    out, token = [], None
    ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    while True:
        params = {'list-type': '2', 'prefix': prefix}
        if token:
            params['continuation-token'] = token
        response = SESSION.get(S3, params=params, timeout=120)
        response.raise_for_status()
        tree = ET.fromstring(response.content)
        for item in tree.findall('s3:Contents', ns):
            out.append((item.find('s3:Key', ns).text, int(item.find('s3:Size', ns).text)))
        if tree.find('s3:IsTruncated', ns).text != 'true':
            return out
        token = tree.find('s3:NextContinuationToken', ns).text


def openneuro_jobs(dataset, patterns):
    """Download jobs for the objects of ``dataset`` matching any glob in ``patterns`` (paths relative to the dataset)."""
    jobs = []
    for key, size in s3_list(dataset + '/'):
        relative = key[len(dataset) + 1:]
        if any(fnmatch.fnmatch(relative, p) for p in patterns):
            jobs.append((f'{S3}/{urllib.parse.quote(key)}', RAW / dataset / relative, size))
    return jobs


def fetch(url, path, size, attempts=6):
    """Resumable download of one file, verified by size; dropped connections are retried with backoff."""
    for attempt in range(attempts):
        try:
            return _fetch(url, path, size)
        except (requests.RequestException, IOError) as error:
            if attempt == attempts - 1:
                raise
            wait = 30 * 2 ** attempt
            print(f'  {path.name}: {type(error).__name__}, retrying in {wait} s', flush=True)
            time.sleep(wait)


def _fetch(url, path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (size is None or path.stat().st_size == size):
        return 0
    part = path.with_name(path.name + '.part')
    have = part.stat().st_size if part.exists() else 0
    headers = {'Range': f'bytes={have}-'} if have else {}
    with SESSION.get(url, headers=headers, stream=True, timeout=300) as response:
        if have and response.status_code != 206:          # server ignored the range: start over
            have = 0
        response.raise_for_status()
        with open(part, 'ab' if have else 'wb') as handle:
            for chunk in response.iter_content(1 << 20):
                handle.write(chunk)
    if size is not None and part.stat().st_size != size:
        raise IOError(f'{path}: got {part.stat().st_size} bytes, expected {size}')
    part.rename(path)
    return size or path.stat().st_size


def run(jobs, workers=6):
    total = sum(s or 0 for _, _, s in jobs)
    print(f'{len(jobs)} files, {total / 1e9:.2f} GB', flush=True)
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fetch, *job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            done += future.result() or 0
            if i % 20 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} files, {done / 1e9:.2f} GB new', flush=True)


def thinking_out_loud(_):
    return openneuro_jobs('ds003626', ['derivatives/*', 'README', 'dataset_description.json', 'participants.tsv'])


def cpseed(_):
    edf_channel_orders()
    return openneuro_jobs('ds006465', ['derivatives/preproc/sub-*/ses-*/*.mat', 'sub-*/ses-*/eeg/*.tsv',
                                       'README.md', 'dataset_description.json', 'participants.tsv'])


def edf_channel_orders():
    """Channel order of each 3M-CPSEED raw EDF, read from the file header only (HTTP range request).

    The authors' preprocessed .mat files keep the EDF order, not the channels.tsv order.
    """
    import json
    out, path = {}, RAW / 'ds006465' / 'edf_channels.json'
    for key, _ in s3_list('ds006465/sub-'):
        if key.endswith('_ses-1_task-imaginedspeech_eeg.edf'):
            head = SESSION.get(f'{S3}/{urllib.parse.quote(key)}', headers={'Range': 'bytes=0-65535'}, timeout=120).content
            count = int(head[252:256].decode().strip())
            out[key.split('/')[1]] = [head[256 + 16 * i:272 + 16 * i].decode('latin-1').strip() for i in range(count)]
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, 'w'), indent=0)


def chisco(args):
    if not args.subject:
        raise SystemExit('chisco is downloaded one subject at a time: --subject 01')
    return openneuro_jobs('ds005170', [f'derivatives/preprocessed_fif/sub-{args.subject}/eeg/*.fif', 'textdataset/*',
                                       'json/*', 'README*', 'dataset_description.json', 'participants.tsv'])


def bci2020(_):
    def walk(url, prefix):
        out = []
        while url:
            page = SESSION.get(url, timeout=120).json()
            for item in page['data']:
                attributes = item['attributes']
                if attributes['kind'] == 'folder':
                    out += walk(item['relationships']['files']['links']['related']['href'], prefix / attributes['name'])
                else:
                    out.append((item['links']['download'], prefix / attributes['name'], attributes['size']))
            url = page['links'].get('next')
        return out
    root = SESSION.get(f'https://api.osf.io/v2/nodes/{OSF_NODE}/files/osfstorage/', timeout=120).json()
    track = next(i for i in root['data'] if i['attributes']['name'].startswith('Track#3'))
    return walk(track['relationships']['files']['links']['related']['href'], RAW / 'bci2020')


def sparrkulee(_):
    meta = SESSION.get(f'{RDR}/api/datasets/:persistentId/', params={'persistentId': SPARRKULEE_DOI}, timeout=120).json()
    jobs = []
    for item in meta['data']['latestVersion']['files']:
        directory, info = item.get('directoryLabel') or '', item['dataFile']
        name = info['filename']
        wanted = ((directory.startswith('derivatives/preprocessed_eeg/') and name.endswith('_eeg.npy'))
                  or (directory == 'stimuli/eeg' and name.endswith('.npz.gz'))
                  or (directory == 'derivatives/preprocessed_stimuli' and name.endswith('_envelope.npy'))
                  or (not directory and name in ('README.md', 'participants.tsv', 'dataset_description.json')))
        if wanted and not item.get('restricted'):
            jobs.append((f'{RDR}/api/access/datafile/{info["id"]}', RAW / 'sparrkulee' / directory / name, info['filesize']))
    return jobs


KARAONE = 'https://www.cs.toronto.edu/~complingweb/data/karaOne'
KARAONE_PEOPLE = ['MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14', 'MM15', 'MM16', 'MM18', 'MM19', 'MM20',
                  'MM21', 'P02']


def karaone(args):
    """KaraOne raw archives (EEG .cnt at 1 kHz, Kinect audio/video), one participant at a time with --subject."""
    people = [args.subject] if args.subject else KARAONE_PEOPLE
    jobs = []
    for person in people:
        head = SESSION.head(f'{KARAONE}/{person}.tar.bz2', timeout=120, allow_redirects=True)
        jobs.append((f'{KARAONE}/{person}.tar.bz2', RAW / 'karaone' / f'{person}.tar.bz2',
                     int(head.headers['Content-Length'])))
    return jobs


SOURCES = dict(thinking_out_loud=thinking_out_loud, cpseed=cpseed, bci2020=bci2020, sparrkulee=sparrkulee, chisco=chisco,
               karaone=karaone)
MODELS = {                       # folder in models/ -> (Hugging Face repository, files)
    'speecht5_hifigan': ('microsoft/speecht5_hifigan', ['*.json', 'pytorch_model.bin']),
    'hubert_base_ls960': ('facebook/hubert-base-ls960', ['*.json', 'pytorch_model.bin']),
    'whisper_small': ('openai/whisper-small', ['*.json', '*.txt', 'model.safetensors']),
}


def models():
    """The pretrained speech models of the reconstruction: vocoder, HuBERT (CLIP space) and Whisper (listener)."""
    from huggingface_hub import snapshot_download
    for folder, (repo, patterns) in MODELS.items():
        print(repo, '->', snapshot_download(repo, local_dir=ROOT / 'models' / folder, allow_patterns=patterns), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', choices=sorted(SOURCES) + ['models'])
    parser.add_argument('--subject', help='chisco: two-digit subject id; karaone: participant (e.g. MM05)')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--list', action='store_true', help='print the file list and total size, download nothing')
    args = parser.parse_args()
    if args.dataset == 'models':
        models()
        sys.exit(0)
    jobs = SOURCES[args.dataset](args)
    if args.list:
        for url, path, size in jobs:
            print(f'{(size or 0) / 1e6:10.1f} MB  {path.relative_to(RAW)}')
        print(f'{len(jobs)} files, {sum(s or 0 for _, _, s in jobs) / 1e9:.2f} GB')
        sys.exit(0)
    run(jobs, args.workers)
