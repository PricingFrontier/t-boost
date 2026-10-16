"""Does early stopping fire before the tree cap? Single-output fits report n_trees_per_bag_;
multiclass fits print `[mc-es] trained_rounds=.. best_round=..` under TBOOST_PROFILE (the
caller greps stderr). Usage: es_check.py <dataset> <split> [json kwargs]"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, Timer, append_jsonl, ce, tb  # noqa: E402

name, split = sys.argv[1], int(sys.argv[2])
kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
d = tb.load(name, split)
task = d["task"]
t = Timer()
est = tb.fit(task, {"graduate": False, **kw}, d["train"], 8)
secs = t.lap()
X, y, e, w = d["test"]
rec = {"dataset": name, "split": split, "kwargs": kw, "fit_s": secs,
       "test": tb.deviance(task, y, tb.predict(task, est, X), e, w),
       "n_trees_per_bag": getattr(est, "n_trees_per_bag_", None),
       "stop": getattr(est, "stopping_reason_per_bag_", None)}
print(json.dumps(rec), flush=True)
append_jsonl(RES / "es_check.jsonl", rec)
