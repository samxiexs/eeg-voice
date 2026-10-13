"""Resumable public EEG AND audio acquisition, checked against a pinned Git tree.

Standard-library only; downloading is independent of model execution.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from . import ROOT

REVISION = "b2d91784a90c18e1d76bd77dcb2b29b0ce27a0be"
S3 = "https://s3.amazonaws.com/openneuro.org"
GITHUB = "https://raw.githubusercontent.com/OpenNeuroDatasets/ds004408"


def request(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "perceived-eeg-speech/1.0", **(headers or {})}), timeout=60)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def source_inventory(raw: Path, revision: str) -> list[dict]:
    cache = raw / "source_inventory.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved["revision"] != revision:
            raise ValueError("Raw directory belongs to another dataset revision")
        return saved["files"]
    with request(f"https://api.github.com/repos/OpenNeuroDatasets/ds004408/git/trees/{revision}?recursive=1") as response:
        tree = json.load(response)
    if tree.get("truncated"):
        raise ValueError("Truncated official Git tree")
    objects, token = {}, None
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        params = {"list-type": "2", "prefix": "ds004408/"}
        if token: params["continuation-token"] = token
        with request(S3 + "?" + urllib.parse.urlencode(params)) as response:
            listing = ET.fromstring(response.read())
        for entry in listing.findall("s:Contents", ns):
            key = entry.find("s:Key", ns).text
            objects[key.removeprefix("ds004408/")] = int(entry.find("s:Size", ns).text)
        if listing.find("s:IsTruncated", ns).text != "true": break
        token = listing.find("s:NextContinuationToken", ns).text
    entries = [x for x in tree["tree"] if x["type"] == "blob" and not x["path"].startswith(".")]

    def describe(entry):
        relative = entry["path"]
        if relative not in objects:
            raise FileNotFoundError(f"Snapshot object unavailable: {relative}")
        item = {"path": relative, "size": objects[relative], "git_blob": entry["sha"]}
        if entry["mode"] == "120000":
            with request(f"{GITHUB}/{revision}/{urllib.parse.quote(relative)}") as response:
                pointer = response.read().decode()
            match = re.search(r"(MD5E|SHA256E)-s(\d+)--([a-f0-9]+)(?:\.|/|$)", pointer)
            if not match: raise ValueError(f"Unsupported annex checksum: {relative}")
            algo, size, expected = match.groups()
            if int(size) != item["size"]:
                raise ValueError(f"S3 and pinned annex sizes disagree: {relative}")
            item.update(algorithm="md5" if algo == "MD5E" else "sha256", expected=expected)
        else:
            item.update(algorithm="git-sha1", expected=entry["sha"])
        return item

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(describe, entries))
    paths = {x["path"] for x in files}
    for i in range(1, 21):
        for suffix in ("wav", "TextGrid"):
            if f"stimuli/audio{i:02d}.{suffix}" not in paths:
                raise ValueError("Missing paired audio or timing")
    save_json(cache, {"revision": revision, "files": files})
    return files


def hashes(path: Path, item: dict) -> dict:
    expected = hashlib.sha1() if item["algorithm"] == "git-sha1" else hashlib.new(item["algorithm"])
    if item["algorithm"] == "git-sha1":
        expected.update(f"blob {path.stat().st_size}\0".encode())
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            expected.update(block); sha.update(block)
    if expected.hexdigest() != item["expected"]:
        raise ValueError(f"Pinned snapshot checksum mismatch: {item['path']}")
    return {**item, "sha256": sha.hexdigest()}


def fetch(raw: Path, item: dict) -> dict:
    path = raw / item["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == item["size"]: return hashes(path, item)
    part = path.with_suffix(path.suffix + ".part")
    last_error = None
    for attempt in range(6):
        try:
            have = part.stat().st_size if part.exists() else 0
            if have == item["size"]:
                receipt = hashes(part, item); part.replace(path); return receipt
            url = S3 + "/ds004408/" + urllib.parse.quote(item["path"])
            with request(url, {"Range": f"bytes={have}-"} if have else None) as response:
                if have and response.status != 206: have = 0
                with part.open("ab" if have else "wb") as handle:
                    shutil.copyfileobj(response, handle, 1 << 20)
            if part.stat().st_size != item["size"]: raise IOError(f"Incomplete: {item['path']}")
            receipt = hashes(part, item); part.replace(path); return receipt
        except ValueError:
            part.unlink(missing_ok=True); raise
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            time.sleep(min(2**attempt, 30))
    raise IOError(f"Download failed: {item['path']}: {last_error}")


def download_dataset(raw: Path, revision=REVISION, subjects=None, workers=4) -> dict:
    raw.mkdir(parents=True, exist_ok=True)
    files = source_inventory(raw, revision)
    subjects = set(subjects or [f"sub-{i:03d}" for i in range(1, 20)])
    selected = [f for f in files if not f["path"].startswith("sub-") or f["path"].split("/")[0] in subjects]
    pending = sum(f["size"] for f in selected if not (raw / f["path"]).exists())
    if shutil.disk_usage(raw).free < pending + 3_000_000_000:
        raise OSError("Insufficient space for EEG/audio download")
    print(f"Downloading {len(selected)} files, {sum(f['size'] for f in selected)/1e9:.2f} GB; EEG + 20 WAV/TextGrid pairs", flush=True)
    receipts, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, raw, f): f for f in selected}
        for future in concurrent.futures.as_completed(futures):
            try: receipts.append(future.result())
            except Exception as error: errors.append({"path": futures[future]["path"], "error": str(error)})
            if (len(receipts) + len(errors)) % 50 == 0:
                print(f"Completed {len(receipts)}/{len(selected)}, failures={len(errors)}", flush=True)
                save_json(raw / "acquisition.json", {"revision": revision, "complete": False, "files": receipts, "errors": errors})
    result = {"revision": revision, "subjects": sorted(subjects), "complete": not errors, "files": sorted(receipts, key=lambda x: x["path"]), "errors": errors}
    save_json(raw / "acquisition.json", result)
    if errors: raise IOError(f"{len(errors)} files failed; acquisition.json contains errors; rerun to resume")
    print("EEG, WAV and TextGrid download complete and source-verified", flush=True)
    return result


def download_models(cfg) -> dict:
    from huggingface_hub import HfApi, snapshot_download
    entries = {}
    existing = cfg.dir("models") / "manifest.json"
    old = json.loads(existing.read_text()) if existing.exists() else {}
    entries = dict(old)  # Preserve every existing pin if a later download fails.
    for kind, spec in cfg.models.items():
        revision = old.get(kind, {}).get("revision") or spec.get("revision") or HfApi().model_info(spec["repo_id"]).sha
        inventory = HfApi().list_repo_files(spec["repo_id"], revision=revision)
        small = [p for p in inventory if "/" not in p and p.endswith((".json", ".yaml", ".txt"))]
        if kind == "vocos": weights = ["pytorch_model.bin"]
        elif "model.safetensors" in inventory: weights = ["model.safetensors"]
        else: weights = [p for p in inventory if "/" not in p and p.endswith(".bin")]
        if not weights: raise ValueError(f"No supported pretrained weights: {kind}")
        snapshot_download(spec["repo_id"], revision=revision, local_dir=cfg.model_dir(kind), allow_patterns=small + weights)
        hashes_by_file = {}
        for p in cfg.model_dir(kind).glob("*"):
            if p.is_file():
                h = hashlib.sha256()
                with p.open("rb") as handle:
                    for block in iter(lambda: handle.read(1 << 20), b""): h.update(block)
                hashes_by_file[p.name] = h.hexdigest()
        entries[kind] = {**spec, "revision": revision, "files": hashes_by_file}
        save_json(existing, entries)
    return entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/ds004408")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--subjects", nargs="+")
    args = parser.parse_args()
    download_dataset(args.raw, subjects=args.subjects, workers=args.workers)
