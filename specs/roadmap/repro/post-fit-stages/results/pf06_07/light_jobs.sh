#!/usr/bin/env bash
# Light follow-ups, run one at a time in the free slot.
PY=/home/ralph/suite/t-boost/.claude/worktrees/agent-aa9f4791fc8a2b51d/.venv/bin/python
cd /home/ralph/suite/t-boost/specs/roadmap/repro/post-fit-stages/results/pf06_07 || exit 1
{
  echo "=== ovr"; $PY ovr_check.py fr_instal 0 0,2
  echo "=== centred s1"; $PY centre_exp.py fr_instal 1 8
  echo "=== centred s2"; $PY centre_exp.py fr_instal 2 8
  echo "=== order pg16"; $PY order_split.py pg16_fleet 0 8
  echo "=== order fr1"; $PY order_split.py fr_instal 1 8
  echo "=== LIGHT DONE"
} 2>&1 | grep -vE "^\[" >> light_jobs.log
