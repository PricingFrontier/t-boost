#!/usr/bin/env bash
# Graduation (shipped rule, brake recorded) on the centred-noise banks, then the paired SEs.
PY=/home/ralph/suite/t-boost/.claude/worktrees/agent-aa9f4791fc8a2b51d/.venv/bin/python
OUT=/tmp/claude-1000/-home-ralph-suite-t-boost/c55508b8-d4e7-4235-9479-9cc2164ee349/tasks/b5fm5m29o.output
cd /home/ralph/suite/t-boost/specs/roadmap/repro/post-fit-stages/results/pf06_07 || exit 1
until grep -q "pg17_cover s0 csig\|Traceback" "$OUT" 2>/dev/null; do sleep 10; done
for s in 0 1 2; do
  $PY grad_exp.py fr_instal $s 8 band_csig shipped 2>&1 | grep -E "d_test|Traceback|Error" >> csig_grad.log
done
$PY grad_exp.py pg16_fleet 0 8 band_csig shipped 2>&1 | grep -E "d_test|Traceback|Error" >> csig_grad.log
$PY grad_exp.py pg17_cover 0 8 band_csig shipped 2>&1 | grep -E "d_test|Traceback|Error" >> csig_grad.log
for st in "fr_instal 0" "fr_instal 1" "fr_instal 2" "pg16_fleet 0" "pg17_cover 0"; do
  $PY paired_se.py $st 8 2>&1 | grep -E "^\{|Traceback|Error" >> csig_grad.log
done
echo "CSIG GRAD DONE" >> csig_grad.log
