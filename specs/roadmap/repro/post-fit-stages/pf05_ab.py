"""PF-05 A/B: post-fit stages scoring the fit rows on the serve map (A, the old behaviour) or
as the trees saw them (B). Needs a build with the temporary TBOOST_EXP_SERVE_FIT_ROWS switch.

    uv run --no-sync python specs/roadmap/repro/post-fit-stages/pf05_ab.py \
        --datasets belgian16,ussev --splits 0,1,2 --out results.jsonl

One JSON line per (dataset, split, arm): test/train deviance, fit seconds, the kept tables and
how many touch a target-encoded raw feature, the ranked path's cutoff and its out-of-bag
deviances, the rebalance report, and a hash of the model bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import team_bench as tb  # noqa: E402


def ts_raw_ids(X: pl.DataFrame, direct_max: int = 16) -> set[int]:
    """Raw ids of the categoricals with more levels than the direct bypass (numeric first)."""
    num = [c for c in X.columns if X[c].dtype != pl.String]
    cat = [c for c in X.columns if X[c].dtype == pl.String]
    return {len(num) + j for j, c in enumerate(cat) if X[c].n_unique() > direct_max}


def run(name: str, split: int, arm: str, kwargs: dict) -> dict:
    data = tb.load(name, split)
    task = data["task"]
    # A: the old behaviour (serve map everywhere, cross-fit categoricals excluded from the
    # rebalance); B: out-of-fold placements for the evidence only (stage-1 build); C: out-of-fold
    # placements for the evidence and the rebalance, no exclusion; D: serve-map evidence,
    # out-of-fold rebalance, no exclusion.
    env = {"A": {"TBOOST_EXP_SERVE_FIT_ROWS": "1", "TBOOST_EXP_TS_EXCLUSION": "1"},
           "B": {"TBOOST_EXP_TS_EXCLUSION": "1"},
           "C": {},
           "D": {"TBOOST_EXP_SERVE_OOB": "1"},
           # E: as C, but only categoricals whose levels share bins (more than 254 levels)
           "E": {"TBOOST_EXP_SHARED_ONLY": "1"}}[arm]
    for key in ("TBOOST_EXP_SERVE_FIT_ROWS", "TBOOST_EXP_TS_EXCLUSION", "TBOOST_EXP_SERVE_OOB",
                "TBOOST_EXP_SHARED_ONLY"):
        os.environ.pop(key, None)
    os.environ.update(env)
    t0 = time.time()
    est = tb.fit(task, kwargs, data["train"])
    secs = time.time() - t0
    out = {"dataset": name, "split": split, "arm": arm, "fit_s": round(secs, 1)}
    for side in ("test", "train"):
        X, y, e, w = data[side]
        out[f"{side}_dev"] = tb.deviance(task, y, tb.predict(task, est, X), e, w)
    rep = est.pruning_report_ or {}
    ts = ts_raw_ids(data["train"][0])
    kept = [tuple(u) for u in rep.get("kept") or []]
    out["kept"] = len(kept)
    out["kept_ts"] = sum(1 for u in kept if set(u) & ts)
    out["kept_ts_inter"] = sum(1 for u in kept if len(u) > 1 and set(u) & ts)
    path = rep.get("path") or {}
    out["path_best"] = path.get("best")
    out["path_oob"] = path.get("oob_deviance")
    out["rebalance"] = rep.get("rebalance")
    try:
        from _artifact import model_bytes  # python/tests helper
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "python" / "tests"))
        from _artifact import model_bytes
    out["bytes"] = hashlib.sha256(model_bytes(est)).hexdigest()[:16]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--splits", default="0")
    ap.add_argument("--arms", default="A,B")
    ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kwargs = json.loads(a.kwargs)
    with open(a.out, "a") as fh:
        for name in a.datasets.split(","):
            for split in (int(s) for s in a.splits.split(",")):
                for arm in a.arms.split(","):
                    r = run(name, split, arm, kwargs)
                    fh.write(json.dumps(r, default=str) + "\n")
                    fh.flush()
                    print(json.dumps({k: r[k] for k in ("dataset", "split", "arm", "test_dev",
                                                         "kept", "kept_ts", "path_best",
                                                         "fit_s", "bytes")}), flush=True)


if __name__ == "__main__":
    main()
