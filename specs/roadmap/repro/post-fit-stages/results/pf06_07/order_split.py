"""Where banding's held-out cost sits: the move B - U split by table order (purified contributions)
and its first-order term Σ(p - y)δ per order and class, on test rows. Usage:
    order_split.py <dataset> <split> <n_bags> [stored banding, default: shipped]
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, append_jsonl  # noqa: E402
from exp_lib import Stage, ce_logits, softmax  # noqa: E402

from t_boost._t_boost import _MultiClassTableModel  # noqa: E402

name, split, n_bags = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
which = sys.argv[4] if len(sys.argv) > 4 else "shipped"
st = Stage(name, split, n_bags)
B = st.B if which == "shipped" else _MultiClassTableModel.from_bytes(
    (st.dir / f"{which}.bin").read_bytes())


def by_order(model) -> np.ndarray:
    """[K, n_test, 4]: intercept, order-1, order-2, order-3 contributions on test rows."""
    parts = model.effect_contributions(st.x_te, **st.kw_te, n_jobs=8)
    out = np.zeros((len(parts), st.x_te.shape[0], 4))
    for k, (f0, contrib, sups) in enumerate(parts):
        c = np.asarray(contrib, dtype=np.float64)
        out[k, :, 0] = f0
        for j, u in enumerate(sups):
            out[k, :, min(len(u), 3)] += c[:, j]
    return out


cu, cb = by_order(st.U), by_order(B)
a_tr, a_te = st.logits(st.U)
assert np.allclose(cu.sum(axis=2).T, a_te, atol=1e-3), "contributions do not sum to the logits"
p = softmax(a_te)
Y = np.zeros_like(p)
Y[np.arange(len(st.y_te)), st.y_te] = 1.0
g = (p - Y).T  # [K, n]
w = st.w_te
base = ce_logits(a_te, st.y_te, w)
d = cb - cu  # [K, n, 4]
first = np.einsum("kn,kno,n->ko", g, d, w) / w.sum() / base * 100  # [K, order]
rms = np.sqrt(np.einsum("kno,n->ko", d ** 2, w) / w.sum())
rec = {"dataset": name, "split": split, "n_bags": n_bags, "banding": which,
       "first_by_class_order": first.tolist(), "first_by_order": first.sum(axis=0).tolist(),
       "rms_move_by_class_order": rms.tolist()}
append_jsonl(RES / "order_split.jsonl", rec)
print(name, split, n_bags, which, "first-order by order [f0, 1, 2, 3]:",
      np.round(first.sum(axis=0), 4), "\n per class:\n", np.round(first, 4), "\n rms move:\n",
      np.round(rms, 4), flush=True)
