#!/usr/bin/env python3
"""Download the SpeechT5 text-to-speech model and a fixed speaker embedding.

The content-first route needs a mature synthesiser: EEG decides *what* is said,
this model decides *how*.  SpeechT5 TTS is chosen because its mel output is
exactly what the already-pinned SpeechT5 HiFi-GAN consumes (80 bins, hop 256,
16 kHz), so no new vocoder is introduced.

A single fixed speaker embedding is stored alongside it: the synthetic voice
must be constant and clearly different from the stimulus speaker, so that
nothing in the listening material can be mistaken for the presented audio.

Files are fetched with curl against the pinned revision.  Hugging Face serves
large files from ``cdn-lfs.hf.co`` / ``us.aws.cdn.hf.co``, which the system
resolver here does not resolve, so the CDN hosts are resolved through public
DNS and pinned with ``curl --resolve``; the download itself still goes to
huggingface.co over TLS with the correct SNI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'microsoft/speecht5_tts'
REVISION = '30fcde30f19b87502b8435427b5f5068e401d5f6'
FILES = ('config.json', 'preprocessor_config.json', 'tokenizer_config.json', 'special_tokens_map.json',
         'added_tokens.json', 'spm_char.model', 'pytorch_model.bin')
SPEAKER_REPOSITORY = 'Matthijs/cmu-arctic-xvectors'
SPEAKER_FILE = 'spkrec-xvect.zip'
CDN_HOSTS = ('cdn-lfs.hf.co', 'us.aws.cdn.hf.co', 'cdn-lfs-us-1.hf.co')
PUBLIC_RESOLVERS = ('8.8.8.8', '1.1.1.1')


def resolve(host: str) -> list[str]:
    for server in PUBLIC_RESOLVERS:
        try:
            output = subprocess.run(['nslookup', host, server], capture_output=True, text=True, timeout=20).stdout
        except Exception:
            continue
        addresses = re.findall(r'^Address:\s*([0-9.]+)$', output, flags=re.M)
        if addresses:
            return addresses
    return []


def curl_arguments() -> list[str]:
    arguments = []
    for host in CDN_HOSTS:
        for address in resolve(host)[:3]:
            arguments += ['--resolve', f'{host}:443:{address}']
    return arguments


def fetch(url: str, destination: Path, resolves: list[str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ['curl', '-L', '--fail', '--retry', '5', '--retry-delay', '5', '-C', '-', *resolves, '-o', str(destination), url]
    print(f'[speecht5] {destination.name}', flush=True)
    subprocess.run(command, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def speaker_embedding(archive: Path, index: int):
    """One fixed x-vector from the CMU-Arctic table (npy files inside the archive)."""
    import zipfile
    import numpy as np
    with zipfile.ZipFile(archive) as bundle:
        names = sorted(n for n in bundle.namelist() if n.endswith('.npy'))
        if not names:
            raise RuntimeError('no x-vector arrays in the speaker archive')
        name = names[index % len(names)]
        with bundle.open(name) as handle:
            vector = np.load(handle).astype('float32').reshape(-1)
    if vector.shape != (512,):
        raise RuntimeError(f'unexpected speaker embedding shape {vector.shape}')
    return name, vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT / 'models/aligned_local_base/speecht5_tts'))
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--speaker-index', type=int, default=7306, help='row of the x-vector table frozen as the voice')
    args = parser.parse_args()
    import numpy as np
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    resolves = curl_arguments()
    if not resolves:
        print('[speecht5] warning: could not resolve the CDN through public DNS; using the system resolver', flush=True)
    for name in FILES:
        target = output / name
        if target.exists() and target.stat().st_size:
            print(f'[speecht5] have {name}', flush=True); continue
        fetch(f'https://huggingface.co/{REPOSITORY}/resolve/{args.revision}/{name}', target, resolves)
    archive = output / SPEAKER_FILE
    if not archive.exists():
        fetch(f'https://huggingface.co/datasets/{SPEAKER_REPOSITORY}/resolve/main/{SPEAKER_FILE}', archive, resolves)
    name, vector = speaker_embedding(archive, args.speaker_index)
    np.save(output / 'speaker_embedding.npy', vector)
    provenance = dict(repository=REPOSITORY, revision=args.revision,
                      files={item: sha256(output / item) for item in FILES},
                      speaker_repository=SPEAKER_REPOSITORY, speaker_archive_sha256=sha256(archive),
                      speaker_index=args.speaker_index, speaker_array=name,
                      speaker_sha256=hashlib.sha256(vector.tobytes()).hexdigest())
    (output.parent / 'speecht5_tts_download.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({k: v for k, v in provenance.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
