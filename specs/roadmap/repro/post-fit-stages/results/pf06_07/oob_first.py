"""PF-06: can out-of-bag rows see banding's held-out first-order cost? For each training row the
jury is the bags that left it out; its residual p_jury - y is honest. The banding move is
B - U on the same rows (training rows, so the distil has pinned it). Compares the OOB first-order
estimate with the test and training first-order terms. Usage:
    oob_first.py <dataset> <split> <n_bags> <tag> [stored banding ...]"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, append_jsonl  # noqa: E402
from exp_lib import Stage, ce_logits, decompose, softmax  # noqa: E402

from t_boost._t_boost import _MultiClassTableModel  # noqa: E402

name, split, n_bags, tag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
extra = sys.argv[5:]
st = Stage(name, split, n_bags, tag=tag)
n = int(st.meta["n"])
inb = np.fromfile(st.dir / "inbag.u8", dtype=np.uint8).reshape(-1, n).astype(bool)
S = np.stack([st.bag_scores(k) for k in range(st.K)], axis=0)  # [K, bags, n]
oob = ~inb  # [bags, n]
cnt = oob.sum(axis=0)
has = cnt > 0
zj = (S * oob[None]).sum(axis=1) / np.maximum(cnt, 1)[None]  # [K, n]
pj = softmax(zj.T)
y = st.yi_tr
w = st.w_tr.astype(np.float64)
Y = np.zeros_like(pj)
Y[np.arange(n), y] = 1.0
a_tr, a_te = st.logits(st.U)
base_tr = ce_logits(a_tr, y, w)
models = {"shipped": st.B}
for e in extra:
    models[e] = _MultiClassTableModel.from_bytes((st.dir / f"{e}.bin").read_bytes())
res = {"dataset": name, "split": split, "n_bags": n_bags, "tag": tag,
       "oob_rows": int(has.sum()), "mean_jury": float(cnt[has].mean()),
       "ce_jury_oob": ce_logits(zj.T[has], y[has], w[has]),
       "ce_U_oob_rows": ce_logits(a_tr[has], y[has], w[has])}
for label, m in models.items():
    b_tr, b_te = st.logits(m)
    d = b_tr - a_tr
    g = pj - Y
    first_oob = float(np.sum((w[:, None] * g * d)[has]) / w[has].sum()) / base_tr * 100
    dt = decompose(a_te, b_te, st.y_te, st.w_te)
    dr = decompose(a_tr, b_tr, y, w)
    res[label] = {"first_oob": first_oob, "first_test": dt["first"], "first_train": dr["first"],
                  "joint_test": dt["joint2"], "joint_train": dr["joint2"], "realized_test": dt["realized"]}
append_jsonl(RES / "oob_first.jsonl", res)
print(res, flush=True)
