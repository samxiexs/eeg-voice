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
  bootstrap-cache) run cache --hubert "$BASE/hubert" --output "$ARTIFACTS/bootstrap_targets.h5" --device "$ALIGNED_DEVICE" ;;
  hubert|hifigan)
    kind="$1"
    adapt train --kind "$kind" --config "$ALIGNED_CONFIG" --base "$BASE/$kind" \
      --cache "$ARTIFACTS/bootstrap_targets.h5" --output "$ALIGNED_OUTPUT/$kind" \
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
  review)
    run score-review --export-root "$ALIGNED_OUTPUT/audio_listening" \
      --transcriptions "$ALIGNED_OUTPUT/audio_listening/blind/transcriptions.csv" \
      --references "$ALIGNED_OUTPUT/audio_listening/reference_transcripts.csv" \
      --output "$ALIGNED_OUTPUT/audio_review.json"
    ;;
  start)
    for stage in download stimuli prepare references bootstrap-cache hubert hifigan cache audio m0; do
      bash "$0" "$stage"
    done
    echo "Training preparation and M0 finished. Complete real listening CSVs, then run review and full."
    ;;
  status) run readiness; run report --output-root "$ALIGNED_OUTPUT" ;;
  probe|m0|full) bash app/run_aligned_speech_v1.sh "$1" ;;
  full-objective) ALIGNED_OBJECTIVE_ONLY=1 bash app/run_aligned_speech_v1.sh full ;;
  *) echo 'usage: bash app/run_aligned_local.sh setup|download|stimuli|prepare|references|bootstrap-cache|hubert|hifigan|cache|audio|m0|review|full|full-objective|start|probe|status' >&2; exit 2 ;;
esac
