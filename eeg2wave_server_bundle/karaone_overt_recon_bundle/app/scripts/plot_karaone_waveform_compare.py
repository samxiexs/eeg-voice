from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[3]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from generate_all_waveform_comparisons import generate_for_dir, pairs_from_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate waveform comparison PNG/SVG/HTML files for KaraOne synthesis outputs."
    )
    parser.add_argument("--wav-dir", required=True, type=Path)
    parser.add_argument("--original-type", default="original")
    parser.add_argument("--reconstruction-type", default="pred_env_scaled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wav_dir = args.wav_dir.expanduser().resolve()
    if not wav_dir.exists():
        raise FileNotFoundError(f"wav dir not found: {wav_dir}")
    manifest = wav_dir / "listening_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"listening manifest not found: {manifest}")

    pairs = pairs_from_manifest(wav_dir, args.original_type, args.reconstruction_type)
    if not pairs:
        raise RuntimeError(
            "no waveform pairs found; check --original-type and --reconstruction-type "
            f"for manifest: {manifest}"
        )
    count = generate_for_dir(wav_dir, args.original_type, args.reconstruction_type)
    out_dir = wav_dir / "waveform_compare"
    summary = {
        "wav_dir": str(wav_dir),
        "original_type": args.original_type,
        "reconstruction_type": args.reconstruction_type,
        "pairs_found": len(pairs),
        "figures_written": count,
        "figure_dir": str(out_dir),
        "manifest": str(out_dir / "waveform_compare_manifest.csv"),
        "contact_sheet": str(out_dir / f"original_vs_{args.reconstruction_type}_contact_sheet.html"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
