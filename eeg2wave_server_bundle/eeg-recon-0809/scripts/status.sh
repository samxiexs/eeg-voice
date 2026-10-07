#!/usr/bin/env bash
# What is running, how far it got, and whether anything stopped or failed.
# Live view: while true; do clear; bash scripts/status.sh; sleep 30; done
cd "$(dirname "$0")/.."
mtime() { stat -f %m "$1" 2>/dev/null || stat -c %Y "$1"; }      # macOS / Linux
count() { ls $1 2>/dev/null | wc -l | tr -d ' '; }
date
echo "--- running"
jobs=$(ps -eo pid,etime,%cpu,command | grep -E "scripts/(prepare|download|reconstruct|sentences)\.py|run_plan\.sh" | grep -v grep | cut -c1-150)
if [ -n "$jobs" ]; then echo "$jobs"; else echo "!! nothing is running"; fi
echo "--- latest logs"
for log in $(ls -t logs/decoder*.log logs/reconstruct*.log logs/sentences*.log 2>/dev/null | head -2); do
  age=$(( $(date +%s) - $(mtime "$log") ))
  echo "$log: $(grep -v Warn "$log" | tail -1 | cut -c1-140) (updated ${age}s ago)"
done
echo "--- folds finished: items $(count 'outputs/reconstruct/f*/*/summary.json') / 10, sentences linear" \
     "$(count 'outputs/sentences/f*/*/summary.json') / 5, deep $(count 'outputs/sentences_deep/f*/*/summary.json') / 5"
for log in $(find logs -name '*.log' -mmin -120 2>/dev/null); do        # logs written in the last 2 h
  error=$(grep -E "Error|FAILED|non-finite" "$log" | grep -v Warning | tail -1 | cut -c1-120)
  [ -n "$error" ] && echo "!! $log: $error"
done
exit 0
