"""Run the centred-noise follow-ups (csig re-band, graduation on it, paired SEs) on each 8-bag stage
once the queues have finished its banding job (its graduation job has started). One at a time.
Usage: csig_watch.py   (exits when every target is done; log: csig_watch.log)"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

RES = Path(__file__).resolve().parent
PY = "/home/ralph/suite/t-boost/.claude/worktrees/agent-aa9f4791fc8a2b51d/.venv/bin/python"
TARGETS = [("pg17_cover", 1), ("pg17_cover", 2), ("prudential", 0), ("pg16_fleet", 1),
           ("pg16_fleet", 2), ("pg17_instal", 0), ("pg17_instal", 1), ("pg17_instal", 2),
           ("prudential", 1)]
LOG = RES / "csig_watch.log"


def band_done(ds: str, sp: int) -> bool:
    for f in RES.glob("queue_*.log"):
        if f"grad {ds} {sp} 8 " in f.read_text(errors="replace"):
            return True
    return False


def run(args: list[str]) -> None:
    out = subprocess.run([PY, *args], cwd=RES, capture_output=True, text=True)
    keep = [ln for ln in (out.stdout + out.stderr).splitlines()
            if any(k in ln for k in ("csig", "d_test", "band_vs_none", "Traceback", "Error"))]
    with open(LOG, "a") as fh:
        fh.write("\n".join(keep) + "\n")


pending = list(TARGETS)
while pending:
    for ds, sp in list(pending):
        if band_done(ds, sp):
            run(["csig_exp.py", ds, str(sp)])
            run(["grad_exp.py", ds, str(sp), "8", "band_csig", "shipped"])
            run(["paired_se.py", ds, str(sp), "8"])
            pending.remove((ds, sp))
            break
    else:
        time.sleep(30)
with open(LOG, "a") as fh:
    fh.write("CSIG WATCH DONE\n")
