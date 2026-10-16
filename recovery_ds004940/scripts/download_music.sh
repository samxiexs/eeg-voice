#!/usr/bin/env bash
# Downloads for the music EEG route.  Every file is resumable (curl -C -) and checksum-verified where
# the source publishes a checksum.  Nothing here runs automatically; call the part you need.
#
#   bash scripts/download_music.sh data        # Di Liberto et al. 2020 EEG (CND, 5.98 GB) + MIDI + README, Zenodo mirror of Dryad
#   bash scripts/download_music.sh musin-g     # MUSIN-G, OpenNeuro ds003774: per-song EEG of 20 participants + the 12 songs (10.9 GB)
#   bash scripts/download_music.sh models      # MERT-v1-95M (teacher), EnCodec 24 kHz (fallback teacher), BigVGAN-v2 24 kHz 100-band (vocoder)
#   bash scripts/download_music.sh soundfont   # MuseScore_General.sf2 (216 MB, MIT) for rendering the MIDI stimuli with FluidSynth
#   bash scripts/download_music.sh all
#
# Dataset: "Cortical encoding of melodic expectations in human temporal cortex", Di Liberto, Pelofi, Bianco,
# Patel, Mehta, Herrero, de Cheveigne, Shamma, Mesgarani, eLife 2020.  CC0.  DOI 10.5061/dryad.g1jwstqmh
# (Dryad's API now requires a login token; Zenodo record 5083410 is the same deposit with md5 checksums.)
# 20 participants (1-10 non-musicians, 11-20 pianists), 64-channel BioSemi at 512 Hz, 10 monophonic Bach
# pieces (~150 s) x 3 presentations.  The audio played in the experiment is NOT included, only the MIDI files.
#
# Disk: ~6 GB for the zip plus ~6 GB extracted (the zip may be deleted afterwards with KEEP_ZIP=0).
# FluidSynth is a system tool: `brew install fluid-synth` (or render with --synth additive, no install).
#
# MUSIN-G ("Music Listening- Genre EEG dataset", Miyapuram et al.; CC0; DOI 10.18112/openneuro.ds003774.v1.0.0):
# 128-channel EGI HydroCel at 250 Hz, 12 songs of different genres (~2 min each, presented at 8 kHz; the wav
# files ship in Code/ESongs).  Only the per-song BIDS files are fetched: each is an exact cut of the continuous
# recording in sourcedata/ (13.7 GB, not needed), from 10 s before the song to its end.  Needs the aws CLI.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
DATA_ROOT="${DILIBERTO_ROOT:-$PROJECT_ROOT/data/diliberto2020}"
MUSIN_ROOT="${MUSIN_G_ROOT:-$PROJECT_ROOT/data/ds003774}"
MODELS_ROOT="${MUSIC_MODELS_ROOT:-$PROJECT_ROOT/models}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv-aligned-local/bin/python}"
ZENODO="https://zenodo.org/api/records/5083410/files"

md5_of() { if command -v md5 >/dev/null; then md5 -q "$1"; else md5sum "$1" | cut -d' ' -f1; fi; }

fetch() {  # url destination [md5]
  local url="$1" dest="$2" sum="${3:-}"
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ] && { [ -z "$sum" ] || [ "$(md5_of "$dest")" = "$sum" ]; }; then echo "ok  $dest"; return; fi
  echo "get $url"
  curl -L --fail --retry 8 --retry-delay 15 --retry-all-errors -C - -o "$dest.part" "$url"
  if [ -n "$sum" ] && [ "$(md5_of "$dest.part")" != "$sum" ]; then
    echo "md5 mismatch for $dest.part (expected $sum); delete it and retry" >&2; exit 1
  fi
  mv "$dest.part" "$dest"
}

data() {
  df -g "$PROJECT_ROOT" 2>/dev/null | awk 'NR==2 && $4 < 14 {print "warning: less than 14 GB free for the 6 GB zip + extraction" > "/dev/stderr"}'
  fetch "$ZENODO/readme_DiliBach_EEG.txt/content" "$DATA_ROOT/readme_DiliBach_EEG.txt" 7250e1919ef8201c04a9cc8a301c2c56
  fetch "$ZENODO/diliBach_midi_4dryad.zip/content" "$DATA_ROOT/diliBach_midi_4dryad.zip" aee63ab3769f134fabeec20ddec7cd04
  fetch "$ZENODO/diliBach_4dryad_CND.zip/content" "$DATA_ROOT/diliBach_4dryad_CND.zip" 7a4cf6e0ca39d4b50c519fad2f797329
  mkdir -p "$DATA_ROOT/midi" "$DATA_ROOT/CND"
  unzip -n -q "$DATA_ROOT/diliBach_midi_4dryad.zip" -d "$DATA_ROOT/midi"
  unzip -n -q "$DATA_ROOT/diliBach_4dryad_CND.zip" -d "$DATA_ROOT/CND"
  if [ "${KEEP_ZIP:-1}" = 0 ]; then rm -f "$DATA_ROOT/diliBach_4dryad_CND.zip"; fi
  echo "MIDI files:  $(find "$DATA_ROOT/midi" -name '*.mid' | wc -l | tr -d ' ')"
  echo "CND files:   $(find "$DATA_ROOT/CND" -name 'data*.mat' | wc -l | tr -d ' ') (expected 21: dataStim + 20 dataSub)"
}

musin_g() {
  command -v aws >/dev/null || { echo "aws CLI required (brew install awscli); no credentials needed" >&2; exit 127; }
  mkdir -p "$MUSIN_ROOT"
  aws s3 sync --no-sign-request s3://openneuro.org/ds003774 "$MUSIN_ROOT" --only-show-errors --exclude '*' \
    --include 'README' --include 'CHANGES' --include 'dataset_description.json' \
    --include 'Code/*' --include 'Code/**' --include 'stimuli/*' --include 'sub-*/ses-*/eeg/*'
  echo "per-song EEG files: $(find "$MUSIN_ROOT" -path '*/ses-*/eeg/*_eeg.set' | wc -l | tr -d ' ') (expected 240 = 20 participants x 12 songs)"
  echo "songs:              $(find "$MUSIN_ROOT/Code/ESongs" -name '*.wav' | wc -l | tr -d ' ') (expected 12)"
}

models() {
  "$PYTHON_BIN" - "$MODELS_ROOT" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path(sys.argv[1])
for repo, folder, patterns in [
    ('m-a-p/MERT-v1-95M', 'mert_v1_95m', ['*.json', '*.py', 'pytorch_model.bin']),     # skip the fairseq .pt
    ('facebook/encodec_24khz', 'encodec_24khz', ['*.json', 'model.safetensors']),
    ('nvidia/bigvgan_v2_24khz_100band_256x', 'bigvgan_v2_24khz_100band_256x', None),   # code + weights; vocoding only
]:
    path = snapshot_download(repo, local_dir=root / folder, allow_patterns=patterns)
    print(f'{repo} -> {path}')
PY
}

soundfont() {
  fetch "https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2" \
        "$MODELS_ROOT/soundfonts/MuseScore_General.sf2"
  command -v fluidsynth >/dev/null || echo "note: fluidsynth not installed; run 'brew install fluid-synth' (or use --synth additive)"
}

case "${1:-}" in
  data) data ;;
  musin-g) musin_g ;;
  models) models ;;
  soundfont) soundfont ;;
  all) data; musin_g; models; soundfont ;;
  *) echo 'usage: bash scripts/download_music.sh data|musin-g|models|soundfont|all' >&2; exit 2 ;;
esac
