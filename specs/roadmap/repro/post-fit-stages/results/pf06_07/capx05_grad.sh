#!/usr/bin/env bash
# Graduation (shipped rule, brake recorded) on the half-cap banks of fr_instal.
PY=/home/ralph/suite/t-boost/.claude/worktrees/agent-aa9f4791fc8a2b51d/.venv/bin/python
cd /home/ralph/suite/t-boost/specs/roadmap/repro/post-fit-stages/results/pf06_07 || exit 1
until grep -q "CSIG GRAD DONE" csig_grad.log 2>/dev/null; do sleep 10; done
for s in 0 1 2; do
  $PY grad_exp.py fr_instal $s 8 band_capx0.5 shipped 2>&1 | grep -E "d_test|Traceback|Error" >> capx05_grad.log
done
echo "CAPX05 DONE" >> capx05_grad.log
