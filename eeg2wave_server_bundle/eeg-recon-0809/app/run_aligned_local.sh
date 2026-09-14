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
check_m0() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    sys.exit(f"Missing M0 report: {p}")
r = json.loads(p.read_text())
if r.get('m0_passed') is not True:
    sys.exit(f"M0 FAILED: retrieval={r.get('retrieval_r1')}, variance={r.get('prediction_variance_ratio')}, "
             f"template_improvement={r.get('template_improvement')}. Full training stopped. "
             "No manual forms are required; inspect/retrain M0. See " + str(p))
print('M0 objective gate passed.')
PY
}
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
    bash "$0" full
    bash "$0" status
    ;;
  start)
    for stage in download stimuli prepare references bootstrap-cache hubert hifigan cache audio m0; do
      bash "$0" "$stage"
    done
    echo "Audio training and M0 objective gate passed. Run full for objective EEG training; no manual forms required."
    ;;
  status) run readiness; run report --output-root "$ALIGNED_OUTPUT" ;;
  full|full-objective)
    for required in "$ALIGNED_OUTPUT/adapt/best_checkpoint.pt" "$ALIGNED_OUTPUT/m0_report.json"; do
      if [[ ! -f "$required" ]]; then
        echo "Missing prerequisite: $required. Run all to complete earlier stages." >&2
        exit 1
      fi
    done
    check_m0 "$ALIGNED_OUTPUT/m0_report.json"
    bash app/run_aligned_speech_v1.sh full
    ;;
  m0)
    bash app/run_aligned_speech_v1.sh m0
    check_m0 "$ALIGNED_OUTPUT/m0_report.json"
    ;;
  m0-retry)
    # A new experiment: never resume the collapsed optimizer or retrain audio.
    retry="$ALIGNED_OUTPUT/m0_lr1e-5"
    run train-eeg --stage align --m0 --lag-ms 0 --max-steps 5000 --lr 0.00001 \
      --device "$ALIGNED_DEVICE" --batch-size 1 \
      --initialize "$ALIGNED_OUTPUT/adapt/best_checkpoint.pt" --output "$retry/align"
    run train-eeg --stage finetune --m0 --lag-ms 0 --max-steps 5000 --lr 0.00001 \
      --resume-early-stop-fix \
      --device "$ALIGNED_DEVICE" --batch-size 1 \
      --initialize "$retry/align/best_checkpoint.pt" --output "$retry/finetune"
    checkpoint="$retry/finetune/best_checkpoint.pt"
    if [[ ! -f "$checkpoint" ]]; then checkpoint="$retry/finetune/last_checkpoint.pt"; fi
    if [[ -f "$retry/finetune/best_m0_checkpoint.pt" ]]; then checkpoint="$retry/finetune/best_m0_checkpoint.pt"; fi
    run evaluate --m0 --role train --device "$ALIGNED_DEVICE" \
      --checkpoint "$checkpoint" --output "$retry/report.json"
    check_m0 "$retry/report.json"
    echo "Retry passed. Report: $retry/report.json. Review this experiment before choosing full-training hyperparameters."
    ;;
  m0-refine)
    refine="$ALIGNED_OUTPUT/m0_lr3e-6_refine"
    run train-eeg --stage finetune --m0 --refine-finetune --lag-ms 0 \
      --epochs 20 --max-steps 1000 --lr 0.000003 --device "$ALIGNED_DEVICE" --batch-size 1 \
      --initialize "$ALIGNED_OUTPUT/m0_lr1e-5/finetune/best_checkpoint.pt" --output "$refine"
    checkpoint="$refine/best_m0_checkpoint.pt"
    if [[ ! -f "$checkpoint" ]]; then checkpoint="$refine/best_checkpoint.pt"; fi
    if [[ ! -f "$checkpoint" ]]; then checkpoint="$refine/last_checkpoint.pt"; fi
    run evaluate --m0 --role train --device "$ALIGNED_DEVICE" \
      --checkpoint "$checkpoint" --output "$refine/report.json"
    check_m0 "$refine/report.json"
    run export --kind eeg --m0 --role train --device "$ALIGNED_DEVICE" \
      --checkpoint "$checkpoint" --output "$refine/waveforms"
    echo "M0 refinement passed; report and waveform controls saved in $refine. Full training is a separate experiment."
    ;;
  full-refined|full-refined-all)
    mkdir -p "$PROJECT_ROOT/logs"
    # Preserve an explicit caller seed (for example ALIGNED_SEEDS=322).
    # The all variant intentionally runs the registered three-seed set.
    if [[ "$1" == "full-refined-all" ]]; then
      export ALIGNED_SEEDS="31 47 73"
    else
      export ALIGNED_SEEDS="${ALIGNED_SEEDS:-31}"
    fi
    bash "$0" _full-refined 2>&1 | tee -a "$PROJECT_ROOT/logs/aligned_full_refined.log"
    ;;
  _full-refined)
    export ALIGNED_M0_REPORT="$ALIGNED_OUTPUT/m0_lr3e-6_refine/report.json"
    check_m0 "$ALIGNED_M0_REPORT"
    "$PYTHON_BIN" - "$ALIGNED_OUTPUT" "$ALIGNED_CONFIG" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'app')
import aligned_speech as runner
root = Path(sys.argv[1]); cfg = runner.config(sys.argv[2])
report = json.loads((root / 'm0_lr3e-6_refine/report.json').read_text())
checkpoint = root / 'm0_lr3e-6_refine/best_m0_checkpoint.pt'
payload = runner.load_payload(checkpoint)
runner.check_eeg_artifacts(payload, cfg)
decoder = root / 'adapt/best_checkpoint.pt'
decoder_payload = runner.load_payload(decoder)
runner.check_eeg_artifacts(decoder_payload, cfg)
if (report.get('checkpoint_sha256') != runner.sha256(checkpoint)
    or report.get('manifest_sha256') != payload['signature']['manifest']
    or report.get('decoder_origin_sha256') != runner.sha256(decoder)
    or payload.get('decoder_origin_sha256') != runner.sha256(decoder)
    or payload['stage'] != 'finetune' or not payload['signature']['m0']
    or report.get('role') != 'train' or report.get('m0') is not True
    or report.get('stage') != 'finetune'):
    sys.exit('M0 report/checkpoint/decoder provenance mismatch')
print('Verified passing M0 checkpoint, split and frozen acoustic decoder.')
PY
    export ALIGNED_FULL_OUTPUT="$ALIGNED_OUTPUT/full_lr1e-5_mel3e-6"
    export ALIGNED_ALIGN_LR=0.00001
    export ALIGNED_FINETUNE_LR=0.000003
    export ALIGNED_BATCH=1
    bash app/run_aligned_speech_v1.sh full
    ;;
  probe) bash app/run_aligned_speech_v1.sh probe ;;
  *) echo 'usage: bash app/run_aligned_local.sh all|setup|download|stimuli|prepare|references|bootstrap-cache|hubert|hifigan|cache|audio|m0|m0-retry|m0-refine|full-refined|full-refined-all|full|start|probe|status' >&2; exit 2 ;;
esac
