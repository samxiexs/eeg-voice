#!/usr/bin/env python3
"""Fetch and verify the public DS004940 sentence-reference table."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


URL = "https://raw.githubusercontent.com/OpenNeuroDatasets/ds004940/master/N400PvsA_stimuli_parameters.tsv"
SHA256 = "26b15a495f32faf8348ccc44d4a438e6e806c566c32e62c69864d4b93b130ed1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and hashlib.sha256(output.read_bytes()).hexdigest() == SHA256:
        print(f"verified existing table: {output}")
        return
    with urlopen(URL) as response:
        value = response.read()
    digest = hashlib.sha256(value).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"unexpected DS004940 stimulus-table hash: {digest}")
    temporary = output.with_suffix(".tmp")
    temporary.write_bytes(value)
    temporary.replace(output)
    print(f"downloaded and verified: {output}")


if __name__ == "__main__":
    main()
