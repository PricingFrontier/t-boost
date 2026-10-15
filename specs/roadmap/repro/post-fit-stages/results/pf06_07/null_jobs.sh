#!/usr/bin/env bash
# The unit-free null-space refusal rule on the stored stages (graduation is mostly single-core GCV).
PY=/home/ralph/suite/t-boost/.claude/worktrees/agent-aa9f4791fc8a2b51d/.venv/bin/python
cd /home/ralph/suite/t-boost/specs/roadmap/repro/post-fit-stages/results/pf06_07 || exit 1
for s in 0 1 2; do
  $PY grad_exp.py fr_instal $s 8 B,U null_refuse 2>&1 | grep -E "d_test|Traceback|Error" >> null_jobs.log
done
$PY grad_exp.py pg16_fleet 0 8 B,U null_refuse 2>&1 | grep -E "d_test|Traceback|Error" >> null_jobs.log
$PY grad_exp.py pg17_cover 0 8 B null_refuse 2>&1 | grep -E "d_test|Traceback|Error" >> null_jobs.log
$PY grad_exp.py fr_instal 0 8 B,U null_refuse nt12k 2>&1 | grep -E "d_test|Traceback|Error" >> null_jobs.log
echo "NULL DONE" >> null_jobs.log
