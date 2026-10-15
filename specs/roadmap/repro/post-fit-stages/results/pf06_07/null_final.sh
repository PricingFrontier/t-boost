#!/usr/bin/env bash
# The null-space refusal rule on the shipped banks of pg17_instal (s0-2) and prudential s1, each
# once queue C3 has finished that stage's graduation job.
PY=/home/ralph/suite/t-boost/.claude/worktrees/agent-aa9f4791fc8a2b51d/.venv/bin/python
cd /home/ralph/suite/t-boost/specs/roadmap/repro/post-fit-stages/results/pf06_07 || exit 1
next_job_started() {  # $1 = the job line that follows the stage's graduation job
  grep -q "$1" queue_C3.log 2>/dev/null
}
for pair in "pg17_instal 0|band pg17_instal 1 8" "pg17_instal 1|band pg17_instal 2 8" \
            "pg17_instal 2|band prudential 1 8" "prudential 1|band pg16_fleet 0 1"; do
  stage=${pair%%|*}
  marker=${pair#*|}
  until next_job_started "$marker"; do sleep 30; done
  $PY grad_exp.py $stage 8 B null_refuse 2>&1 | grep -E "d_test|Traceback|Error" >> null_final.log
done
echo "NULL FINAL DONE" >> null_final.log
