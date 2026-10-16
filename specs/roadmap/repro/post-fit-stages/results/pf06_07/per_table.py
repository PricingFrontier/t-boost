"""PF-07: which graduated tables carry the test harm. Each shipped-rule update applied alone.
Usage: per_table.py <dataset> <split> <n_bags> <base U|B> [tag]"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, append_jsonl  # noqa: E402
from exp_lib import Stage  # noqa: E402
from grad_lib import apply, curves, select_shipped  # noqa: E402

name, split, n_bags, base = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
tag = sys.argv[5] if len(sys.argv) > 5 else ""
st = Stage(name, split, n_bags, tag=tag)
model = st.U if base == "U" else st.B
recs = curves(st.est, model, st.w_tr)
ce0 = st.ce(model)
names = model.raw_feature_names()
CEIL = max(type(st.est)._GRADUATION_ALPHAS)
out = []
for r in recs:
    vals, a = select_shipped(r)
    if a <= 0:
        continue
    cand, ratio = apply(st, model, [(r["k"], r["idx"], vals.tolist())])
    d = 100 * (st.ce(cand)[1] - ce0[1]) / ce0[1]
    st.forget(cand)
    move = vals - r["v"]
    j = int(np.argmax(np.abs(move)))
    sup = np.asarray(r["sup"]).ravel()
    out.append({"k": r["k"], "u": [names[i] for i in r["features"]], "shape": list(r["shape"]),
                "alpha": a, "ceiling": a == CEIL, "n_data": r["n_data"], "d_test_pct": d,
                "train_ratio": ratio, "max_move": float(move[j]), "max_move_support": float(sup[j]),
                "max_move_from": float(r["v"][j]), "rms_move": float(np.sqrt(np.average(
                    move ** 2, weights=np.maximum(sup, 1e-12)))) if sup.sum() > 0 else 0.0})
out.sort(key=lambda e: -e["d_test_pct"])
tot = sum(e["d_test_pct"] for e in out)
append_jsonl(RES / "per_table.jsonl", {"dataset": name, "split": split, "n_bags": n_bags,
                                       "base": base, "tag": tag, "sum_d_test_pct": tot, "tables": out})
print(f"{name} s{split} {base} {tag}: {len(out)} tables, sum of single-table effects {tot:+.3f}%")
for e in out[:12]:
    print(f"  {e['d_test_pct']:+.4f}%  class {e['k']} {e['u']} shape {e['shape']} alpha {e['alpha']:g} "
          f"max move {e['max_move']:+.2f} (from {e['max_move_from']:+.2f}, support {e['max_move_support']:.0f})")
print("  ...")
for e in out[-5:]:
    print(f"  {e['d_test_pct']:+.4f}%  class {e['k']} {e['u']} shape {e['shape']} alpha {e['alpha']:g}")
by_order = {}
for e in out:
    o = len(e["u"])
    by_order.setdefault(o, []).append(e["d_test_pct"])
print("by order:", {o: (len(v), round(sum(v), 3)) for o, v in by_order.items()})
by_ceil = {c: round(sum(e["d_test_pct"] for e in out if e["ceiling"] == c), 3) for c in (True, False)}
print("ceiling vs not:", by_ceil)
