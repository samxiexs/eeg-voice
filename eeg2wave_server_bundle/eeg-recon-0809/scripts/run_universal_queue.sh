#!/usr/bin/env bash
# Sequential subject-free ablation (one training process at a time, throttled).
#
#   speech     DS004940 only, trial + spatial augmentation              -> is the subject-free encoder any good on unseen people?
#   pretrain   DS004940 + music, acoustic objectives (3000 updates)     -> stage 1
#   joint      from pretrain, full objective, speech + music            -> does music help?
#   speech-ft  from pretrain, full objective, music weight 0            -> control for joint (music only in pre-training)
#
# Every run resumes from its training_state.pt and skips when complete, so the queue can be restarted.
#   MUSIC_DATASET=musin_g UNIVERSAL_THROTTLE=1.0 nohup bash scripts/run_universal_queue.sh > logs/universal_queue.log 2>&1 &
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export MUSIC_DATASET="${MUSIC_DATASET:-musin_g}"
export UNIVERSAL_THROTTLE="${UNIVERSAL_THROTTLE:-1.0}"
for stage in ${STAGES:-speech pretrain joint speech-ft}; do
  echo "=== $(date '+%F %T') start $stage (music=$MUSIC_DATASET, throttle=$UNIVERSAL_THROTTLE)"
  if ! bash app/run_universal.sh "$stage"; then
    echo "=== $(date '+%F %T') FAILED $stage; queue stopped"; exit 1
  fi
  echo "=== $(date '+%F %T') done $stage"
done
echo "=== $(date '+%F %T') queue finished"
