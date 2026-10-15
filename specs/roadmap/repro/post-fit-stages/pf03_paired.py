"""Paired per-row test-deviance change of the per-bag rebalance (Q) against no rebalance (R),
with its standard error, on small sets where single-split differences may be noise.

    uv run --no-sync python specs/roadmap/repro/post-fit-stages/pf03_paired.py tic2000,ussev 0,1,2,3,4
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import team_bench as tb  # noqa: E402


def unit_dev(task, y, pred, e):
    if task == "poisson":
        mu = pred * (1.0 if e is None else e)
        t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
        return 2.0 * (t - (y - mu))
    if task == "gamma":
        return 2.0 * (-np.log(y / pred) + (y - pred) / pred)
    if task == "binary":
        p = np.clip(pred[:, 1], 1e-15, 1 - 1e-15)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return -np.log(np.clip(pred[np.arange(len(y)), y.astype(int)], 1e-15, None))


for name in sys.argv[1].split(","):
    for split in (int(s) for s in sys.argv[2].split(",")):
        d = tb.load(name, split)
        task = d["task"]
        X, y, e, w = d["test"]
        w = np.ones(len(y)) if w is None else w
        devs = {}
        for arm, kw in (("Q", {}), ("R", {"prune_rebalance": False})):
            os.environ.pop("TBOOST_EXP_SLICE_CHECK", None)
            est = tb.fit(task, kw, d["train"])
            devs[arm] = unit_dev(task, y, tb.predict(task, est, X), e)
        diff = w * (devs["Q"] - devs["R"])
        base = np.sum(w * devs["R"])
        rel = 100 * diff.sum() / base
        se = 100 * np.sqrt(len(diff)) * diff.std(ddof=1) / base
        print(json.dumps({"dataset": name, "split": split, "q_vs_r_pct": round(rel, 4),
                          "paired_se_pct": round(se, 4)}), flush=True)
