"""PF-03 A/B on the PF-05 build: the rebalance judged on the in-sample slice (P), per bag on
out-of-bag rows (Q), switched off (R, the reference), or per bag deploying the mean of the bags'
own corrections (M). P and M need the temporary TBOOST_EXP_SLICE_CHECK / TBOOST_EXP_MEAN_BAG_CORR
switches of the 2026-10-15 experiment build (removed before commit); Q and R run on any build.

    uv run --no-sync python specs/roadmap/repro/post-fit-stages/pf03_ab.py \
        --datasets swmc,ussev --splits 0,1 --arms P,Q,R --out results.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "python" / "tests"))
import team_bench as tb  # noqa: E402
from _artifact import model_bytes  # noqa: E402

ARMS = {"P": ({"TBOOST_EXP_SLICE_CHECK": "1"}, {}),
        "Q": ({}, {}),
        "R": ({}, {"prune_rebalance": False}),
        # M: as Q, deploying the mean of the bags' own corrections instead of an all-rows solve
        "M": ({"TBOOST_EXP_MEAN_BAG_CORR": "1"}, {})}


def run(name: str, split: int, arm: str, extra: dict) -> dict:
    data = tb.load(name, split)
    task = data["task"]
    env, kwargs = ARMS[arm]
    for key in ("TBOOST_EXP_SLICE_CHECK", "TBOOST_EXP_MEAN_BAG_CORR"):
        os.environ.pop(key, None)
    os.environ.update(env)
    t0 = time.time()
    est = tb.fit(task, {**extra, **kwargs}, data["train"])
    out = {"dataset": name, "split": split, "arm": arm, "fit_s": round(time.time() - t0, 1)}
    for side in ("test", "train"):
        X, y, e, w = data[side]
        out[f"{side}_dev"] = tb.deviance(task, y, tb.predict(task, est, X), e, w)
    rep = est.pruning_report_ or {}
    out["kept"] = len(rep.get("kept") or [])
    out["rebalance"] = rep.get("rebalance")
    out["bytes"] = hashlib.sha256(model_bytes(est)).hexdigest()[:16]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--splits", default="0")
    ap.add_argument("--arms", default="P,Q,R")
    ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    extra = json.loads(a.kwargs)
    with open(a.out, "a") as fh:
        for name in a.datasets.split(","):
            for split in (int(s) for s in a.splits.split(",")):
                for arm in a.arms.split(","):
                    r = run(name, split, arm, extra)
                    fh.write(json.dumps(r, default=str) + "\n")
                    fh.flush()
                    rb = r["rebalance"] or {}
                    print(json.dumps({k: r[k] for k in ("dataset", "split", "arm", "test_dev",
                                                         "fit_s", "bytes")}
                                     | {"adopted": rb.get("adopted"), "step": rb.get("step"),
                                        "evidence": rb.get("evidence")}), flush=True)


if __name__ == "__main__":
    main()
