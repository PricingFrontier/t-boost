"""PF-06 candidate: banding's noise budget from the softmax-relevant noise. Each class's sigma is
the curvature-weighted RMS bag error of its logit after removing the p-weighted common deviation
(the part of the bag spread that moves every class alike and so never changes a probability).
Re-bands the stored unbanded bank with those sigmas (cap and tolerance unchanged).
Usage: csig_exp.py <dataset> <split> [tag]   (8 bags; appends variant "csig" to band_exp.jsonl)"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, Timer, append_jsonl, census  # noqa: E402
from exp_lib import Stage, band_summary, decompose, softmax  # noqa: E402

name, split = sys.argv[1], int(sys.argv[2])
tag = sys.argv[3] if len(sys.argv) > 3 else ""
st = Stage(name, split, 8, tag=tag)
S = np.stack([st.bag_scores(k) for k in range(st.K)], axis=0)  # [K, bags, n]
a_tr, a_te = st.logits(st.U)
p = softmax(a_tr).T
w = st.w_tr.astype(np.float64)
dev = S - S.mean(axis=1, keepdims=True)
nb = S.shape[1]
cen = dev - (p[:, None, :] * dev).sum(axis=0)[None]
var_cen = (cen ** 2).sum(axis=1) / (nb - 1)
h = w[None] * p * (1 - p)
sig_c = np.sqrt((h * var_cen / nb).sum(axis=1) / h.sum(axis=1))
t = Timer()
m, rep = st.U.exp_band(st.x_tr, st.y_tr, st.w_tr, [float(s) for s in sig_c], 0.75, 0.001, "",
                       cat_x=st.cat_tr, n_jobs=8)
secs = t.lap()
import json  # noqa: E402
reps = json.loads(rep)
(st.dir / "band_csig.bin").write_bytes(m.to_bytes())
ce_u, ce_b, ce_m = st.ce(st.U), st.ce(st.B), st.ce(m)
b_tr, b_te = st.logits(m)
rec = {"dataset": name, "split": split, "n_bags": 8, "variant": "csig", "tag": tag,
       "fit_s": st.fit_s, "band_s": secs, "train": ce_m[0], "test": ce_m[1],
       "train_u": ce_u[0], "test_u": ce_u[1],
       "d_test_pct": 100 * (ce_m[1] - ce_u[1]) / ce_u[1],
       "d_test_vs_ship_pct": 100 * (ce_m[1] - ce_b[1]) / ce_b[1],
       **{f"cells_{k}": v for k, v in census(m).items()}, "cells_u": census(st.U)["cells"],
       "dec_train": decompose(a_tr, b_tr, st.yi_tr, st.w_tr),
       "dec_test": decompose(a_te, b_te, st.y_te, st.w_te),
       "band": band_summary(reps), "sigmas": sig_c.tolist(), "sigmas_shipped": st.sigmas()}
append_jsonl(RES / "band_exp.jsonl", rec)
print(f"{name} s{split} csig: test {ce_m[1]:.6f} ({rec['d_test_pct']:+.3f}% vs U, "
      f"{rec['d_test_vs_ship_pct']:+.3f}% vs ship) cells {rec['cells_cells']} vs shipped "
      f"{census(st.B)['cells']} sigma {np.round(sig_c, 4)} vs {np.round(st.sigmas(), 4)}", flush=True)
