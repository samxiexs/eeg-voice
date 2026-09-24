#!/usr/bin/env bash
# Subject-free speech + music route.  One heavy process at a time (16 GB machine); throttled by default.
#
# Music data preparation (after scripts/download_music.sh data|musin-g models [soundfont]); MUSIC_DATASET picks the set:
#   bash app/run_universal.sh prepare         # EEG -> harmonised shards, manifest, splits, normalizer (MUSIN-G also writes its audio)
#   bash app/run_universal.sh audio           # Di Liberto only: MIDI -> 24 kHz audio aligned to the EEG clock (r gate vs CND envelope)
#   bash app/run_universal.sh targets         # MERT (or EnCodec) teacher, BigVGAN mel, envelope/onset per piece
#   bash app/run_universal.sh music-decoder   # teacher -> 100-band mel decoder (audio only)
#
# Training (DS004940 held-out participants = unseen cohort, used for selection):
#   bash app/run_universal.sh speech          # A1: speech only, temporal + spatial augmentation
#   bash app/run_universal.sh pretrain        # stage 1: speech + music, low-level acoustic + mel objectives
#   bash app/run_universal.sh joint           # stage 2: from stage 1, full objective, speech + music
#   bash app/run_universal.sh speech-ft       # control for stage 2: from stage 1, music weight 0
#
# Knobs: UNIVERSAL_SEED (322)  UNIVERSAL_DEVICE (auto)  UNIVERSAL_THROTTLE (0.5)  UNIVERSAL_HELDOUT (4)
#        UNIVERSAL_TRUNK (Broderick trunk if present; "none" to disable)  MUSIC_TEACHER (mert|encodec)
#        MUSIC_SYNTH (fluidsynth|additive)  MUSIC_DATASET (diliberto2020|musin_g)  UNIVERSAL_EXTRA (extra trainer flags)
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv-aligned-local/bin/python}"
export ALIGNED_TARGET_CACHE_NAME=targets_adapted.h5 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
SEED="${UNIVERSAL_SEED:-322}"
DEVICE="${UNIVERSAL_DEVICE:-auto}"
THROTTLE="${UNIVERSAL_THROTTLE:-0.5}"
HELDOUT="${UNIVERSAL_HELDOUT:-4}"
TEACHER="${MUSIC_TEACHER:-mert}"
SYNTH="${MUSIC_SYNTH:-fluidsynth}"
EXTRA="${UNIVERSAL_EXTRA:-}"
MUSIC_DATASET="${MUSIC_DATASET:-diliberto2020}"
case "$MUSIC_DATASET" in diliberto2020|musin_g) ;; *) echo "MUSIC_DATASET must be diliberto2020 or musin_g" >&2; exit 2;; esac
MUSIC_ROOT="$PROJECT_ROOT/artifacts/music/$MUSIC_DATASET"
TARGETS="$MUSIC_ROOT/targets_$TEACHER.h5"
MUSIC_DECODER_DIR="$PROJECT_ROOT/outputs/music_decoder/$MUSIC_DATASET/$TEACHER"
MUSIC_DECODER="$MUSIC_DECODER_DIR/best.pt"
BASE="$PROJECT_ROOT/outputs/universal"
DEFAULT_TRUNK="$PROJECT_ROOT/outputs/broderick2018/trunk/best.pt"
TRUNK="${UNIVERSAL_TRUNK:-$([ -f "$DEFAULT_TRUNK" ] && echo "$DEFAULT_TRUNK" || echo none)}"
TRUNK_FLAG=()
if [ "$TRUNK" != none ]; then TRUNK_FLAG=(--initialize-trunk "$TRUNK"); fi

# Temporal / trial augmentation: the two variants that beat the v3 baseline on 2026-09-22 (seed 322), combined.
AUG_TRIAL=(--shift-max 10 --channel-gain 0.15 --background-mix 0.5 --background-alpha 0.2 0.8
           --mix-same-content 0.2 --mix-partners 3 --mix-start 0.8 --mix-anneal-epochs 12)
# Spatial augmentation (app/universal_model.py, part 2).  Mirror stays off (hemispheric asymmetry).
AUG_SPATIAL=(--spatial-geometry 0.5 --spatial-subset 0.5 --spatial-keep 0.25 1 --spatial-reference 0.3
             --spatial-conduction 0.3 --spatial-recolour 0.3)
MUSIC=(--music-root "$MUSIC_ROOT" --music-targets "$TARGETS" --music-decoder "$MUSIC_DECODER" --music-mix 0.3 --music-background 0.3)

guard() {
  if pgrep -f "aligned_recovery.py|universal_train.py|envelope_decoder.py|generative_recovery.py|music.py|broderick_pretrain.py" >/dev/null; then
    echo "another training process is running; refusing to overlap (one heavy job at a time)" >&2; exit 1
  fi
}

train() {  # output-name, flags...
  local name="$1"; shift
  guard
  mkdir -p logs
  # shellcheck disable=SC2086  # UNIVERSAL_EXTRA is a flag list
  "$PYTHON_BIN" app/universal_train.py --seed "$SEED" --device "$DEVICE" --throttle "$THROTTLE" \
    --heldout-subjects "$HELDOUT" --output "$BASE/${name}_seed$SEED" "$@" $EXTRA 2>&1 | tee -a "logs/universal_${name}_seed$SEED.log"
}

case "${1:-}" in
  prepare) guard; "$PYTHON_BIN" "scripts/prepare_${MUSIC_DATASET/2020/}.py" --output "$MUSIC_ROOT" ;;
  audio)
    if [ "$MUSIC_DATASET" = musin_g ]; then echo "MUSIN-G ships its audio; prepare already wrote $MUSIC_ROOT/audio"; exit 0; fi
    "$PYTHON_BIN" scripts/render_music_audio.py --synth "$SYNTH" --stimulus "$MUSIC_ROOT/stimulus.h5" --output "$MUSIC_ROOT/audio" ;;
  targets) "$PYTHON_BIN" scripts/cache_music_targets.py --teacher "$TEACHER" --audio "$MUSIC_ROOT/audio" --output "$TARGETS" ;;
  music-decoder) guard; "$PYTHON_BIN" app/music.py --targets "$TARGETS" --manifest "$MUSIC_ROOT/manifest.csv" \
                   --output "$MUSIC_DECODER_DIR" --device "$DEVICE" --throttle "$THROTTLE" ;;
  speech) train speech ${TRUNK_FLAG[@]+"${TRUNK_FLAG[@]}"} "${AUG_TRIAL[@]}" "${AUG_SPATIAL[@]}" ;;
  pretrain) train "pretrain_$MUSIC_DATASET" ${TRUNK_FLAG[@]+"${TRUNK_FLAG[@]}"} "${AUG_TRIAL[@]}" "${AUG_SPATIAL[@]}" "${MUSIC[@]}" --music-weight 1 \
              --contrastive-weight 0 --duration-weight 0 --acoustic-weight 1 --updates 3000 ;;
  joint) train "joint_$MUSIC_DATASET" --initialize "$BASE/pretrain_${MUSIC_DATASET}_seed$SEED/best_metric.pt" "${AUG_TRIAL[@]}" "${AUG_SPATIAL[@]}" "${MUSIC[@]}" \
           --music-weight 0.5 ;;
  speech-ft) train "speech_ft_$MUSIC_DATASET" --initialize "$BASE/pretrain_${MUSIC_DATASET}_seed$SEED/best_metric.pt" "${AUG_TRIAL[@]}" "${AUG_SPATIAL[@]}" "${MUSIC[@]}" \
               --music-weight 0 ;;
  *) echo 'usage: bash app/run_universal.sh prepare|audio|targets|music-decoder|speech|pretrain|joint|speech-ft' >&2; exit 2 ;;
esac
