"""PF-07: multiclass graduation on one stored soup (run band_exp.py first). Usage:
    grad_exp.py <dataset> <split> <n_bags> [base_models] [rules]

base_models: comma list of `U` (unbanded), `B` (shipped banded), `band_<mode>` (a stored
re-banding). rules: comma list of selection rules (see RULES). For every (base, rule) the
candidate is built with the brake lifted; the record holds the test change, the training ratio
the 1% brake judges, counts of smoothed tables, and per-class / ceiling breakdowns for the
shipped rule. Appends to grad_exp.jsonl.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, Timer, append_jsonl, census  # noqa: E402
from exp_lib import N_JOBS, Stage, decompose  # noqa: E402
from grad_lib import apply, at_alpha, curves, select_null_refuse, select_shipped  # noqa: E402

from t_boost._t_boost import _MultiClassTableModel  # noqa: E402

name, split, n_bags = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
bases = (sys.argv[4] if len(sys.argv) > 4 else "B,U").split(",")
rules = (sys.argv[5] if len(sys.argv) > 5 else "shipped").split(",")
tag = sys.argv[6] if len(sys.argv) > 6 else ""
out = RES / "grad_exp.jsonl"
st = Stage(name, split, n_bags, tag=tag)
CEIL = max(type(st.est)._GRADUATION_ALPHAS)
hs = np.asarray(st.meta["hs"], dtype=float)
cw = hs / hs.sum()  # class curvature-mass shares


def pooled(recs: list[dict[str, Any]], weights: np.ndarray | None) -> dict[int, float]:
    """One alpha per support: the pooled GCV over the support's class tables (weighted by
    `weights[k]`, or equally), adopted when it beats the pooled least-smoothed alpha."""
    by_u: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        by_u[r["features"]].append(r)
    choice: dict[int, float] = {}
    for u, rs in by_u.items():
        rs = [r for r in rs if r["curve"]]
        if not rs:
            continue
        common = set.intersection(*[{a for _, a, _, _ in r["curve"]} for r in rs])
        if not common:
            continue
        tot = {}
        for a in sorted(common):
            tot[a] = sum((1.0 if weights is None else weights[r["k"]]) *
                         next(g for g, aa, _, _ in r["curve"] if aa == a) for r in rs)
        best = min(tot, key=lambda a: (tot[a], a))
        if tot[best] < tot[min(tot)]:
            for r in rs:
                choice[id(r)] = best
    return choice


def updates_for(recs: list[dict[str, Any]], rule: str) -> list[tuple[int, int, list[float], float]]:
    ups = []
    if rule == "shipped":
        for r in recs:
            vals, a = select_shipped(r)
            if a > 0:
                ups.append((r["k"], r["idx"], vals.tolist(), a))
        return ups
    if rule == "null_refuse":
        for r in recs:
            vals, a = select_null_refuse(r, type(st.est))
            if a > 0:
                ups.append((r["k"], r["idx"], vals.tolist(), a))
        return ups
    if rule in ("shared_h", "shared_eq"):
        ch = pooled(recs, cw if rule == "shared_h" else None)
        for r in recs:
            a = ch.get(id(r), 0.0)
            if a > 0:
                ups.append((r["k"], r["idx"], at_alpha(r, a).tolist(), a))
        return ups
    raise SystemExit(f"unknown rule {rule}")


def run(base_label: str, model: Any) -> None:
    t = Timer()
    recs = curves(st.est, model, st.w_tr)
    curve_s = t.lap()
    ce0 = st.ce(model)
    for rule in rules:
        ups = updates_for(recs, rule)
        t.lap()
        cand, ratio = apply(st, model, [(k, i, v) for k, i, v, _ in ups])
        ce1 = st.ce(cand)
        rec = {"dataset": name, "split": split, "n_bags": n_bags, "base": base_label,
               "rule": rule, "tag": tag, "test0": ce0[1], "train0": ce0[0], "test": ce1[1], "train": ce1[0],
               "d_test_pct": 100 * (ce1[1] - ce0[1]) / ce0[1], "train_ratio": ratio,
               "adopted_by_brake": bool(ratio <= 1.01), "n_tables": len(recs),
               "n_smoothed": len(ups),
               "n_by_class": np.bincount([k for k, *_ in ups], minlength=st.K).tolist(),
               "n_ceiling": sum(1 for *_, a in ups if a == CEIL),
               "cells0": census(model)["cells"], "curve_s": curve_s, "apply_s": t.lap()}
        a_tr, a_te = st.logits(model)
        b_tr, b_te = st.logits(cand)
        rec["dec_test"] = decompose(a_te, b_te, st.y_te, st.w_te)
        rec["dec_train"] = decompose(a_tr, b_tr, st.yi_tr, st.w_tr)
        if rule == "shipped":
            # sanity: the replica's choices equal the estimator's own per-table solve
            # per-class and ceiling breakdowns (each subset applied alone)
            per_class = []
            for k in range(st.K):
                sub = [(kk, i, v) for kk, i, v, _ in ups if kk == k]
                if not sub:
                    per_class.append(None)
                    continue
                c, r_k = apply(st, model, sub)
                per_class.append({"n": len(sub), "d_test_pct": 100 * (st.ce(c)[1] - ce0[1]) / ce0[1],
                                  "train_ratio": r_k})
                st.forget(c)
            rec["per_class"] = per_class
            for label, keep in (("ceiling", lambda a: a == CEIL), ("below_ceiling", lambda a: a < CEIL)):
                sub = [(k, i, v) for k, i, v, a in ups if keep(a)]
                if sub:
                    c, r_s = apply(st, model, sub)
                    rec[label] = {"n": len(sub), "d_test_pct": 100 * (st.ce(c)[1] - ce0[1]) / ce0[1],
                                  "train_ratio": r_s}
                    st.forget(c)
        append_jsonl(out, rec)
        print(f"{name} s{split} b{n_bags} {base_label:>12} {rule:>10}: d_test {rec['d_test_pct']:+.3f}% "
              f"train x{ratio:.4f} smoothed {len(ups)} {rec['n_by_class']} ceiling {rec['n_ceiling']} "
              f"per-class {[None if p is None else round(p['d_test_pct'], 3) for p in rec.get('per_class', [])]}",
              flush=True)
        st.forget(cand)


def graduate_first(rule: str) -> None:
    """Graduation before banding: smooth the unbanded bank (brake judged there), then band the
    graduated model exactly as the shipped banding bands (same noise, cap and tolerance)."""
    t = Timer()
    recs = curves(st.est, st.U, st.w_tr)
    ups = updates_for(recs, rule)
    cand, ratio = apply(st, st.U, [(k, i, v) for k, i, v, _ in ups])
    gb, _ = cand.exp_band(st.x_tr, st.y_tr, st.w_tr, st.sigmas(), 0.75, 0.001, "",
                          cat_x=st.cat_tr, n_jobs=N_JOBS)
    (st.dir / f"gfirst_{rule}.bin").write_bytes(gb.to_bytes())
    ce_b = st.ce(st.B)
    ce_g = st.ce(gb)
    gated = ce_g if ratio <= 1.01 else ce_b
    rec = {"dataset": name, "split": split, "n_bags": n_bags, "base": "gfirst", "rule": rule,
           "tag": tag,
           "test0": ce_b[1], "train0": ce_b[0], "test": ce_g[1], "train": ce_g[0],
           "d_test_pct": 100 * (ce_g[1] - ce_b[1]) / ce_b[1], "train_ratio": ratio,
           "adopted_by_brake": bool(ratio <= 1.01),
           "d_test_gated_pct": 100 * (gated[1] - ce_b[1]) / ce_b[1],
           "n_smoothed": len(ups), "n_by_class": np.bincount([k for k, *_ in ups], minlength=st.K).tolist(),
           "n_ceiling": sum(1 for *_, a in ups if a == CEIL),
           "cells0": census(st.B)["cells"], "cells": census(gb)["cells"], "secs": t.lap()}
    append_jsonl(out, rec)
    print(f"{name} s{split} b{n_bags} gfirst {rule:>10}: d_test vs band-only {rec['d_test_pct']:+.3f}% "
          f"(gated {rec['d_test_gated_pct']:+.3f}%) train x{ratio:.4f} smoothed {len(ups)} "
          f"cells {rec['cells0']} -> {rec['cells']}", flush=True)
    st.forget(cand)
    st.forget(gb)


for b in bases:
    if b == "U":
        run("unbanded", st.U)
    elif b == "B":
        run("shipped", st.B)
    elif b == "G":
        for rule in rules:
            graduate_first(rule)
    else:
        m = _MultiClassTableModel.from_bytes((st.dir / f"{b}.bin").read_bytes())
        run(b, m)
