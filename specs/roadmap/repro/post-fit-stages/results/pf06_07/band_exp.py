"""PF-06: multiclass banding on one soup. Usage:
    band_exp.py <dataset> <split> <n_bags> [mode,mode,...]

One fit (graduate=False) with the banding dump on; then the shipped banded model B, the
unbanded U and every banding mode re-banded from U, each scored on train and test with the
second-order decomposition of its move. Appends one JSON line per model to band_exp.jsonl.
Banded models are saved as <dump>/band_<mode>.bin for the graduation runs.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, Timer, append_jsonl, census  # noqa: E402
from exp_lib import Stage, band_summary, decompose, softmax  # noqa: E402

name, split, n_bags = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
modes = [m for m in (sys.argv[4] if len(sys.argv) > 4 else "").split(",") if m]
tag = sys.argv[5] if len(sys.argv) > 5 else ""
fit_kwargs = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
out = RES / "band_exp.jsonl"

st = Stage(name, split, n_bags, tag=tag, fit_kwargs=fit_kwargs)
ce_u = st.ce(st.U)
ce_b = st.ce(st.B)
cu, cb = census(st.U), census(st.B)
a_tr, a_te = st.logits(st.U)


def record(label: str, model, reports, secs: float | None) -> None:
    b_tr, b_te = st.logits(model)
    ce_tr, ce_te = st.ce(model)
    rec = {"dataset": name, "split": split, "n_bags": n_bags, "variant": label, "tag": tag,
           "fit_s": st.fit_s, "band_s": secs,
           "train": ce_tr, "test": ce_te, "train_u": ce_u[0], "test_u": ce_u[1],
           "d_test_pct": 100 * (ce_te - ce_u[1]) / ce_u[1],
           "d_test_vs_ship_pct": 100 * (ce_te - ce_b[1]) / ce_b[1],
           **{f"cells_{k}": v for k, v in census(model).items()},
           "cells_u": cu["cells"],
           "dec_train": decompose(a_tr, b_tr, st.yi_tr, st.w_tr),
           "dec_test": decompose(a_te, b_te, st.y_te, st.w_te),
           "band": band_summary(reports), "sigmas": st.sigmas(), "hs": st.meta["hs"],
           "dev_mean": st.meta["dev_mean"]}
    append_jsonl(out, rec)
    print(f"{name} s{split} b{n_bags} {label:>14}: test {ce_te:.6f} ({rec['d_test_pct']:+.3f}% vs U, "
          f"{rec['d_test_vs_ship_pct']:+.3f}% vs ship) cells {rec['cells_cells']} "
          f"first {rec['dec_test']['first']:+.3f} joint {rec['dec_test']['joint2']:.3f} "
          f"diag {rec['dec_test']['diag2']:.3f}", flush=True)


record("unbanded", st.U, None, None)
record("shipped", st.B, st.band_reports, None)

# Noise structure (8 bags): bag spread of class logits vs of softmax-centred logits, and the
# joint (softmax Hessian) noise against its diagonal stand-in.
if n_bags >= 2:
    S = [st.bag_scores(k) for k in range(st.K)]
    if all(s is not None for s in S):
        S = np.stack(S, axis=0)  # [K, bags, n]
        p = softmax(a_tr).T  # [K, n]
        w = st.w_tr.astype(np.float64)
        dev = S - S.mean(axis=1, keepdims=True)  # bag deviations [K, B, n]
        nb = S.shape[1]
        var_k = (dev ** 2).sum(axis=1) / (nb - 1)  # [K, n]
        cbar = (p[:, None, :] * dev).sum(axis=0)  # p-weighted common deviation per bag/row
        cen = dev - cbar[None]
        var_cen = (cen ** 2).sum(axis=1) / (nb - 1)
        h = w[None] * p * (1 - p)
        sig_raw = np.sqrt((h * var_k / nb).sum(axis=1) / h.sum(axis=1))
        sig_cen = np.sqrt((h * var_cen / nb).sum(axis=1) / h.sum(axis=1))
        # joint noise (per row, tr(H Σ) with H = diag(p) - pp') vs diagonal Σ p(1-p) Σ_kk
        joint = ((p[:, None, :] * cen ** 2).sum(axis=0)).sum(axis=0) / (nb - 1)
        diag = ((p * (1 - p))[:, None, :] * dev ** 2).sum(axis=0).sum(axis=0) / (nb - 1)
        noise = {"dataset": name, "split": split, "n_bags": n_bags, "variant": "_noise", "tag": tag,
                 "sigma_shipped": st.sigmas(), "sigma_raw": sig_raw.tolist(),
                 "sigma_centred": sig_cen.tolist(),
                 "joint_over_diag_noise": float((w * joint).sum() / (w * diag).sum())}
        append_jsonl(out, noise)
        print("noise", noise, flush=True)

for mode in modes:
    t = Timer()
    m, reps = st.band("" if mode == "base" else mode)
    secs = t.lap()
    (st.dir / f"band_{mode}.bin").write_bytes(m.to_bytes())
    if mode == "base":
        same = m.to_bytes() == st.B.to_bytes()
        print("base reproduces shipped bytes:", same, flush=True)
    record(mode, m, reps, secs)
