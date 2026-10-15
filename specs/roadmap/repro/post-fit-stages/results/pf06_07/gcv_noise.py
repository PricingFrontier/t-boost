"""PF-07 diagnostic: GCV's implied cell noise against the measured between-bag noise, on the
unbanded bank U (whose tables sit on the soup's grids, like the bag banks). Needs a stage dumped
with TBOOST_EXP_BAGTABLES=1 (bagtables_<k>.json). Usage:
    gcv_noise.py <dataset> <split> <n_bags> <tag>

Per eligible table: sigma2_gcv = RSS(a*)/(n - df(a*)) under GCV's model Var(v_c) = sigma2/s_c, and
sigma2_bag = mean_c s_c var_b(v_c) / B (banding's noise convention for the bag mean) over the
observed cells. Then graduation with Mallows' Cp at the bag noise scale, RSS(a) + 2 sigma2 df(a)
against the unsmoothed table (Cp(0) = 2 sigma2 n), at sigma2_bag and 4 sigma2_bag, applied to U.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, append_jsonl  # noqa: E402
from exp_lib import Stage  # noqa: E402
from grad_lib import apply, curves, select_shipped  # noqa: E402

name, split, n_bags, tag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
st = Stage(name, split, n_bags, tag=tag)
recs = curves(st.est, st.U, st.w_tr)
bag = {}
for k in range(st.K):
    banks = json.loads((st.dir / f"bagtables_{k}.json").read_text())
    by_u: dict[tuple, list[np.ndarray]] = {}
    for b in banks:
        for t in b:
            by_u.setdefault(tuple(t["u"]), []).append((tuple(t["shape"]), np.asarray(t["values"])))
    bag[k] = by_u
B = None
rows = []
for r in recs:
    entries = bag[r["k"]].get(tuple(r["features"]))
    if not entries or len({e[0] for e in entries}) != 1 or entries[0][0] != tuple(r["shape"]):
        r["bag"] = None
        continue
    V = np.stack([e[1] for e in entries])  # [bags, cells]
    B = V.shape[0]
    full_var = V.var(axis=0, ddof=1) / B
    # restrict to the smoothed block (the same cells the curve sees), via the assemble mapping:
    # the block is everything except the reserved missing slots; recover it by elimination
    v_full, vv = r["v"], r["vv"]
    mask = np.zeros(v_full.size, dtype=bool)
    # cells of the block are where assemble(v_s=vv+1) differs from v
    probe = r["assemble"](vv + 1.0, 0.0)
    mask = ~np.isclose(probe, v_full)
    var_c = full_var[mask]
    ss = r["ss"]
    obs = ss > 0
    if obs.sum() < 3 or var_c.size != ss.size:
        r["bag"] = None
        continue
    sigma2_bag = float(np.mean(ss[obs] * var_c[obs]))
    r["bag"] = sigma2_bag
    r["bagmean_gap"] = float(np.max(np.abs(V.mean(axis=0) - v_full)))
    c = r["curve"] or []
    vals, a_star = select_shipped(r)
    if c:
        n = r["n_data"]
        if a_star > 0:
            _, _, vs, df = next(t for t in c if t[1] == a_star)
        else:
            _, _, vs, df = min(c, key=lambda t: t[1])
        rss = float(np.sum(ss * (vv - vs) ** 2))
        s2g = rss / max(n - df, 1e-9)
    else:
        s2g = float("nan")
    rows.append({"k": r["k"], "u": list(r["features"]), "cells": int(ss.size), "alpha": a_star,
                 "sigma2_gcv": s2g, "sigma2_bag": sigma2_bag,
                 "ratio": s2g / sigma2_bag if sigma2_bag > 0 else float("inf"),
                 "bagmean_gap": r["bagmean_gap"]})


def cp_updates(scale: float) -> list[tuple[int, int, list[float]]]:
    ups = []
    for r in recs:
        if r.get("bag") is None or not r["curve"]:
            continue
        s2 = scale * r["bag"]
        n = r["n_data"]
        best = (2.0 * s2 * n, 0.0, None)  # Cp(0): RSS 0, df n
        for _, a, vs, df in r["curve"]:
            cp = float(np.sum(r["ss"] * (r["vv"] - vs) ** 2)) + 2.0 * s2 * df
            if cp < best[0]:
                best = (cp, a, vs)
        if best[1] > 0:
            ups.append((r["k"], r["idx"], r["assemble"](best[2], best[1]).tolist()))
    return ups


smoothed = [x for x in rows if x["alpha"] > 0]
summary = {"dataset": name, "split": split, "n_bags": n_bags, "tag": tag, "bags": B,
           "tables_matched": len(rows), "tables_smoothed_gcv": len(smoothed),
           "median_ratio_smoothed": float(np.median([x["ratio"] for x in smoothed])) if smoothed else None,
           "median_ratio_all": float(np.median([x["ratio"] for x in rows])) if rows else None,
           "max_bagmean_gap": float(max(x["bagmean_gap"] for x in rows)) if rows else None}
ce0 = st.ce(st.U)
for label, scale in (("cp_bag", 1.0), ("cp_bag4", 4.0), ("cp_bag16", 16.0)):
    ups = cp_updates(scale)
    if ups:
        cand, ratio = apply(st, st.U, ups)
        d = 100 * (st.ce(cand)[1] - ce0[1]) / ce0[1]
        st.forget(cand)
    else:
        ratio, d = 1.0, 0.0
    summary[label] = {"n": len(ups), "d_test_pct": d, "train_ratio": ratio}
append_jsonl(RES / "gcv_noise.jsonl", {**summary, "tables": rows})
print(json.dumps(summary, indent=1), flush=True)
