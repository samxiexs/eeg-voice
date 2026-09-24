#!/usr/bin/env bash
# Sequential augmentation ablation for the recovery-v3 encoder (DS004940).
#
# One training process at a time (16 GB machine), throttled for temperature.
# Every variant reuses the seed's existing M0 gate and the v3 positional recipe;
# only the augmentation flags differ, so the runs compare against
# outputs/aligned_recovery_v3/full_seed<seed>_positional (no augmentation beyond
# the v3 defaults).  Finished variants are skipped, so the queue can be re-run.
#
#   bash scripts/run_augmentation_queue.sh            # seeds 322, then 323 324 for every variant
#   SEEDS="322" bash scripts/run_augmentation_queue.sh
#   VARIANTS="robust pool" bash scripts/run_augmentation_queue.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export ALIGNED_THROTTLE="${ALIGNED_THROTTLE:-0.5}"
BASE="outputs/aligned_recovery_v3"
SEEDS="${SEEDS:-322 323 324}"
VARIANTS="${VARIANTS:-robust pool both}"

flags_for() {
  case "$1" in
    # Single-trial robustness: +/-40 ms latency jitter, per-channel gain, and the
    # participant's own other-sentence EEG as real background noise (p 0.5, fraction 0.2-0.8).
    robust) echo "--shift-max 10 --channel-gain 0.15 --background-mix 0.5 --background-alpha 0.2 0.8" ;;
    # SNR curriculum: average up to 3 other presentations of the sentence into the partner half,
    # with probability 0.8 at the start decaying to 0.2 by epoch 12 (of ~24).
    pool) echo "--mix-partners 3 --mix-start 0.8 --mix-anneal-epochs 12" ;;
    both) echo "$(flags_for robust) $(flags_for pool)" ;;
    *) echo "unknown variant: $1" >&2; exit 2 ;;
  esac
}
mix_for() { case "$1" in pool|both) echo 0.2 ;; *) echo 0 ;; esac; }

if pgrep -f "app/aligned_recovery.py" >/dev/null; then
  echo "another aligned_recovery.py process is running; refusing to overlap" >&2; exit 1
fi
for seed in $SEEDS; do
  for variant in $VARIANTS; do
    dir="$BASE/full_seed${seed}_positional_$variant"
    if [ -f "$dir/training_state.pt" ] && .venv-aligned-local/bin/python - "$dir" <<'PY'
import sys, torch
state = torch.load(f'{sys.argv[1]}/training_state.pt', map_location='cpu', weights_only=False)
sys.exit(0 if state.get('complete') else 1)
PY
    then echo "skip $dir (complete)"; continue; fi
    [ -f "$BASE/m0_seed$seed/best_passed.pt" ] || { echo "no M0 gate for seed $seed; run 'bash app/run_aligned_recovery.sh m0' first" >&2; exit 1; }
    echo "=== $(date '+%F %T') start seed=$seed variant=$variant"
    ALIGNED_SEED="$seed" ALIGNED_TAG="$variant" ALIGNED_MIX="$(mix_for "$variant")" ALIGNED_EXTRA="$(flags_for "$variant")" \
      bash app/run_aligned_recovery.sh full
    echo "=== $(date '+%F %T') done seed=$seed variant=$variant"
  done
done
dirs=()
for seed in $SEEDS; do dirs+=("$BASE/full_seed${seed}_positional"); done
.venv-aligned-local/bin/python app/recovery_reports.py aggregate "${dirs[@]}" --output "$BASE/augmentation_baseline_summary.json"
for variant in $VARIANTS; do
  dirs=()
  for seed in $SEEDS; do [ -f "$BASE/full_seed${seed}_positional_$variant/metrics.json" ] && dirs+=("$BASE/full_seed${seed}_positional_$variant"); done
  [ ${#dirs[@]} -gt 0 ] && .venv-aligned-local/bin/python app/recovery_reports.py aggregate "${dirs[@]}" --output "$BASE/augmentation_${variant}_summary.json"
done
echo "=== $(date '+%F %T') queue finished"
