"""Time one default multiclass fit (graduate=False) with the stage profiler on."""
from __future__ import annotations

import os
import sys

os.environ["TBOOST_PROFILE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, Timer, append_jsonl, ce, census, tb  # noqa: E402

name = sys.argv[1]
split = int(sys.argv[2]) if len(sys.argv) > 2 else 0
d = tb.load(name, split)
t = Timer()
est = tb.fit("multiclass", {"graduate": False}, d["train"], 8)
fit_s = t.lap()
Xte, yte, _, wte = d["test"]
rec = {"dataset": name, "split": split, "fit_s": fit_s,
       "test": ce(yte, est.predict_proba(Xte), wte), **census(est._multi_model)}
print(rec, flush=True)
append_jsonl(RES / "time_probe.jsonl", rec)
