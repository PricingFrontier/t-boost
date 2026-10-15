"""PF-07: the centred graduation rule on the unbanded bank U (see centre_lib). Usage:
    centre_exp.py <dataset> <split> <n_bags> [tag]
Appends to grad_exp.jsonl with base "unbanded", rule "centred" (brake lifted; the training ratio
the brake would judge is recorded)."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, Timer, append_jsonl, census  # noqa: E402
from exp_lib import Stage, decompose, softmax  # noqa: E402
from grad_lib import apply  # noqa: E402
from centre_lib import centred_updates  # noqa: E402

name, split, n_bags = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tag = sys.argv[4] if len(sys.argv) > 4 else ""
st = Stage(name, split, n_bags, tag=tag)
a_tr, a_te = st.logits(st.U)
w = st.w_tr.astype(np.float64)
priors = (softmax(a_tr) * w[:, None]).sum(axis=0) / w.sum()
t = Timer()
ups, stats = centred_updates(st, st.U, priors)
secs = t.lap()
cand, ratio = apply(st, st.U, [(k, i, v) for k, i, v, _ in ups])
ce0, ce1 = st.ce(st.U), st.ce(cand)
b_tr, b_te = st.logits(cand)
rec = {"dataset": name, "split": split, "n_bags": n_bags, "base": "unbanded", "rule": "centred",
       "tag": tag, "test0": ce0[1], "train0": ce0[0], "test": ce1[1], "train": ce1[0],
       "d_test_pct": 100 * (ce1[1] - ce0[1]) / ce0[1], "train_ratio": ratio,
       "adopted_by_brake": bool(ratio <= 1.01), "n_smoothed": len(ups),
       "n_by_class": np.bincount([k for k, *_ in ups], minlength=st.K).tolist(),
       "n_ceiling": sum(1 for *_, a in ups if a == max(type(st.est)._GRADUATION_ALPHAS)),
       "cells0": census(st.U)["cells"], "centre_stats": stats, "priors": priors.tolist(),
       "curve_s": secs, "dec_test": decompose(a_te, b_te, st.y_te, st.w_te),
       "dec_train": decompose(a_tr, b_tr, st.yi_tr, st.w_tr)}
append_jsonl(RES / "grad_exp.jsonl", rec)
print(f"{name} s{split} b{n_bags} {tag} unbanded centred: d_test {rec['d_test_pct']:+.3f}% train x{ratio:.4f} "
      f"smoothed {len(ups)} {rec['n_by_class']} stats {stats} joint {rec['dec_test']['joint2']:.3f} "
      f"first {rec['dec_test']['first']:+.3f}", flush=True)
