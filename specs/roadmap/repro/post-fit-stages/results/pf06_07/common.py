"""Shared helpers for the PF-06/PF-07 investigation (multiclass banding and graduation).

Run every script with the WORKTREE's interpreter, so the worktree build is imported:
    /home/ralph/suite/t-boost/.claude/worktrees/agent-aa9f4791fc8a2b51d/.venv/bin/python <script>

`load` and `deviance` come from the team harness (team_bench.py); the worktree copy is imported.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

WT = Path("/home/ralph/suite/t-boost/.claude/worktrees/agent-aa9f4791fc8a2b51d")
RES = Path("/home/ralph/suite/t-boost/specs/roadmap/repro/post-fit-stages/results/pf06_07")
sys.path.insert(0, str(WT / "specs/roadmap/repro/post-fit-stages"))
if __import__("os").environ.get("PF_PROTO"):
    # the prototype build (target/proto), so running jobs keep the installed extension
    sys.path.insert(0, str(WT / "target/proto"))

import team_bench as tb  # noqa: E402
import t_boost  # noqa: E402

assert str(WT) in t_boost.__file__, f"wrong t_boost imported: {t_boost.__file__}"

MULTI = ["fr_instal", "pg16_fleet", "pg17_instal", "pg17_cover", "prudential"]


def ce(y: np.ndarray, proba: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted mean multiclass cross-entropy (the harness's deviance for multiclass)."""
    return tb.deviance("multiclass", y, proba, None, w)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")


def census(model: Any) -> dict[str, Any]:
    """`deployed_census` over every class bank: tables and dense cells per order, and totals."""
    c = model.deployed_census()
    cells = {int(k): int(v) for k, v in c["cells"].items()}
    tables = {int(k): int(v) for k, v in c["tables"].items()}
    return {"cells": sum(cells.values()), "cells_by_order": cells,
            "tables": sum(tables.values()), "tables_by_order": tables}


class Timer:
    def __init__(self) -> None:
        self.t = time.time()

    def lap(self) -> float:
        now = time.time()
        d, self.t = now - self.t, now
        return round(d, 2)
