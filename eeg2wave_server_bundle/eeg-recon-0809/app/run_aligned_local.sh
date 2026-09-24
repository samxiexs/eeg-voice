#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv-aligned-local/bin/python}"
export ALIGNED_CONFIG="$PROJECT_ROOT/configs/aligned_speech_local_v1.yaml"
export ALIGNED_OUTPUT="$PROJECT_ROOT/outputs/aligned_speech_local_v1"
export ALIGNED_BATCH="${ALIGNED_BATCH:-1}"
export ALIGNED_DEVICE="${ALIGNED_DEVICE:-auto}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PROJECT_ROOT/artifacts/local_matplotlib"
mkdir -p "$MPLCONFIGDIR"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
BASE="$PROJECT_ROOT/models/aligned_local_base"
ARTIFACTS="$PROJECT_ROOT/artifacts/aligned_speech_local_v1"
export ALIGNED_TARGET_CACHE_NAME="${ALIGNED_TARGET_CACHE_NAME:-targets_adapted.h5}"
BOOTSTRAP_CACHE="$ARTIFACTS/bootstrap_features_v2.h5"
export ALIGNED_OBJECTIVE_ONLY=1
export HUBERT_LOCAL_PATH="$ALIGNED_OUTPUT/hubert/best"
export HIFIGAN_LOCAL_PATH="$ALIGNED_OUTPUT/hifigan/best"
run() { "$PYTHON_BIN" app/aligned_speech.py --config "$ALIGNED_CONFIG" "$@"; }
adapt() { "$PYTHON_BIN" app/adapt_local_audio.py "$@"; }
case "${1:-status}" in
  setup)
    BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-/opt/anaconda3/envs/eegvoice/bin/python}"
    if [[ ! -x "$PYTHON_BIN" ]]; then "$BOOTSTRAP_PYTHON" -m venv --system-site-packages "$PROJECT_ROOT/.venv-aligned-local"; fi
    "$PYTHON_BIN" -m pip install 'soundfile==0.13.1'
    "$PYTHON_BIN" -c 'import torch,transformers,mne,h5py,pandas,scipy,yaml,soundfile; print("torch", torch.__version__, "transformers", transformers.__version__); assert transformers.__version__ == "4.57.6", "Use the tested eegvoice environment"'
    mkdir -p "$ALIGNED_OUTPUT"
    "$PYTHON_BIN" -m pip freeze > "$ALIGNED_OUTPUT/environment.txt"
    ;;
  download) adapt download --output "$BASE" ;;
  stimuli)
    "$PYTHON_BIN" scripts/fetch_ds004940_stimuli.py \
      --output "$PROJECT_ROOT/data/external_metadata/N400PvsA_stimuli_parameters.tsv"
    ;;
  prepare) run prepare --materialize ;;
  references) run references ;;
  bootstrap-cache) run cache --hubert "$BASE/hubert" --output "$BOOTSTRAP_CACHE" --device "$ALIGNED_DEVICE" ;;
  hubert|hifigan)
    kind="$1"
    adapt train --kind "$kind" --config "$ALIGNED_CONFIG" --base "$BASE/$kind" \
      --cache "$BOOTSTRAP_CACHE" --output "$ALIGNED_OUTPUT/$kind" \
      --device "$ALIGNED_DEVICE" --batch-size 1 --epochs 10 --lr 0.00001
    ;;
  cache) run cache --device "$ALIGNED_DEVICE" ;;
  audio)
    run train-audio --stage adapt --device "$ALIGNED_DEVICE" --batch-size "$ALIGNED_BATCH" --output "$ALIGNED_OUTPUT/adapt"
    run train-audio --stage mfcc --device "$ALIGNED_DEVICE" --batch-size "$ALIGNED_BATCH" --output "$ALIGNED_OUTPUT/mfcc"
    run export --kind audio --role validation --device "$ALIGNED_DEVICE" \
      --checkpoint "$ALIGNED_OUTPUT/adapt/best_checkpoint.pt" --mfcc-checkpoint "$ALIGNED_OUTPUT/mfcc/best_checkpoint.pt" \
      --output "$ALIGNED_OUTPUT/audio_listening"
    ;;
  all)
    mkdir -p "$PROJECT_ROOT/logs"
    bash "$0" _all-stages 2>&1 | tee -a "$PROJECT_ROOT/logs/aligned_local_all.log"
    ;;
  _all-stages)
    bash "$0" start
    bash "$0" status
    ;;
  start)
    for stage in download stimuli prepare references bootstrap-cache hubert hifigan cache audio; do
      bash "$0" "$stage"
    done
    echo "Audio-side models are ready. Train the EEG encoder with app/run_aligned_recovery.sh (or app/run_universal.sh)."
    ;;
  status) run readiness; run report --output-root "$ALIGNED_OUTPUT" ;;
  *) echo 'usage: bash app/run_aligned_local.sh all|start|setup|download|stimuli|prepare|references|bootstrap-cache|hubert|hifigan|cache|audio|status' >&2; exit 2 ;;
esac
