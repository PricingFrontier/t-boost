#!/usr/bin/env bash
# Run each line of a job file in order, one at a time.
# Usage: queue.sh <jobfile> <logfile>
# "band <args>" runs band_exp.py, "grad <args>" grad_exp.py, "es <args>" es_check.py.
# Fits run with TBOOST_PROFILE=1 so every bag's early-stopping line ([mc-es] trained/best round)
# and the post-fit stage timings land in the log; the profiler's per-phase tables are dropped.
set -u
PY=/home/ralph/suite/t-boost/.claude/worktrees/agent-aa9f4791fc8a2b51d/.venv/bin/python
RES=/home/ralph/suite/t-boost/specs/roadmap/repro/post-fit-stages/results/pf06_07
KEEP='mc-es|mc-prune\] (deploy_fit|ranked_path|keepset_apply|banding )|test |noise|base repro|d_test|gfirst|^\{|Traceback|Error|error'
cd "$RES" || exit 1
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue ;; esac
  kind=${line%% *}
  args=${line#* }
  echo "=== $(date +%H:%M:%S) $line" >> "$2"
  case "$kind" in
    band) script=band_exp.py ;;
    grad) script=grad_exp.py ;;
    es) script=es_check.py ;;
    *) echo "unknown job $line" >> "$2"; continue ;;
  esac
  eval "TBOOST_PROFILE=1 $PY $script $args" 2>&1 | grep -E --line-buffered "$KEEP" >> "$2"
  echo "=== $(date +%H:%M:%S) done ${PIPESTATUS[0]}" >> "$2"
done < "$1"
echo "=== QUEUE FINISHED $(date +%H:%M:%S)" >> "$2"
