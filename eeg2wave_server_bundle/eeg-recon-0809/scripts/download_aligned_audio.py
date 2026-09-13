#!/usr/bin/env python3
"""Fetch the two official LibriSpeech archives needed by aligned_speech_v1."""
import argparse
import hashlib
from pathlib import Path
import tarfile
import urllib.request

BASE = "https://www.openslr.org/resources/12/"
ARCHIVES = ("train-clean-100.tar.gz", "dev-clean.tar.gz")


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BASE + "md5sum.txt", timeout=60) as response:
        checksums = {line.split()[-1].lstrip("*").split("/")[-1]: line.split()[0]
                     for line in response.read().decode().splitlines() if len(line.split()) == 2}
    for name in ARCHIVES:
        if name not in checksums:
            raise RuntimeError(f"official checksum absent: {name}")
        archive = args.output / name
        if not archive.exists() or md5(archive) != checksums[name]:
            temporary = archive.with_suffix(archive.suffix + ".partial")
            print(f"Downloading {name}", flush=True)
            with urllib.request.urlopen(BASE + name, timeout=60) as response, temporary.open("wb") as stream:
                count = 0
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block); count += len(block)
                    if count % (256 * 1024 * 1024) == 0:
                        print(f"{name}: {count // (1024 * 1024)} MiB", flush=True)
            if md5(temporary) != checksums[name]:
                raise RuntimeError(f"checksum mismatch: {name}")
            temporary.replace(archive)
        marker = args.output / (name + ".extracted")
        if marker.exists() and marker.read_text().strip() == checksums[name]:
            continue
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            for member in members:
                destination = (args.output / member.name).resolve()
                if not destination.is_relative_to(args.output.resolve()) or not (member.isfile() or member.isdir()):
                    raise RuntimeError("refusing unsafe archive member")
            tar.extractall(args.output, members=members, filter="data")
        marker.write_text(checksums[name] + "\n")
    print(f"LIBRISPEECH_ROOT={args.output.resolve() / 'LibriSpeech'}")


if __name__ == "__main__":
    main()
