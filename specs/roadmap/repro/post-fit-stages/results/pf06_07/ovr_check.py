"""Convergence of one-vs-rest binary fits on a multiclass set (same rows, default settings):
does a binary fit on the same data early-stop long before the multiclass fit's 4000-tree cap?
Usage: ovr_check.py <dataset> <split> <class,class,...>"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, Timer, append_jsonl, tb  # noqa: E402

name, split = sys.argv[1], int(sys.argv[2])
classes = [int(c) for c in sys.argv[3].split(",")]
d = tb.load(name, split)
X, y, e, w = d["train"]
Xt, yt, et, wt = d["test"]
for c in classes:
    t = Timer()
    yb = (y == c).astype(int)
    est = tb.fit("binary", {"graduate": False}, (X, yb, e, w), 8)
    rec = {"dataset": name, "split": split, "ovr_class": c, "rate": float(yb.mean()),
           "fit_s": t.lap(), "n_trees_per_bag": est.n_trees_per_bag_,
           "stop": est.stopping_reason_per_bag_,
           "test": tb.deviance("binary", (yt == c).astype(int), est.predict_proba(Xt), None, wt)}
    print(rec, flush=True)
    append_jsonl(RES / "ovr_check.jsonl", rec)
