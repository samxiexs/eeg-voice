"""Download the public datasets this project uses (only the files it needs).

    python scripts/download.py thinking_out_loud      # OpenNeuro ds003626 derivatives (7.2 GB)
    python scripts/download.py cpseed                 # OpenNeuro ds006465 session-level preproc .mat (4.0 GB)
    python scripts/download.py bci2020                # OSF pq7vb, Track 3 imagined speech (2.3 GB)
    python scripts/download.py sparrkulee             # KU Leuven RDR: preprocessed EEG + stimulus audio (29.5 GB)
    python scripts/download.py chisco --subject 01    # OpenNeuro ds005170 preprocessed fif, one subject (~12 GB)

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


def fetch(url, path, size):
    """Resumable download of one file, verified by size."""
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
    return openneuro_jobs('ds006465', ['derivatives/preproc/sub-*/ses-*/*.mat', 'sub-*/ses-*/eeg/*.tsv',
                                       'README.md', 'dataset_description.json', 'participants.tsv'])


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


SOURCES = dict(thinking_out_loud=thinking_out_loud, cpseed=cpseed, bci2020=bci2020, sparrkulee=sparrkulee, chisco=chisco)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', choices=sorted(SOURCES))
    parser.add_argument('--subject', help='chisco only: two-digit subject id')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--list', action='store_true', help='print the file list and total size, download nothing')
    args = parser.parse_args()
    jobs = SOURCES[args.dataset](args)
    if args.list:
        for url, path, size in jobs:
            print(f'{(size or 0) / 1e6:10.1f} MB  {path.relative_to(RAW)}')
        print(f'{len(jobs)} files, {sum(s or 0 for _, _, s in jobs) / 1e9:.2f} GB')
        sys.exit(0)
    run(jobs, args.workers)
