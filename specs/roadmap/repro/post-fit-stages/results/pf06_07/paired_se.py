"""Row-level paired standard errors (same test rows) of the key comparisons on a stored stage:
banding vs none, and every stored re-banding vs the shipped banding. Percent of the reference's
mean test cross-entropy. Usage: paired_se.py <dataset> <split> <n_bags> [tag]"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, append_jsonl  # noqa: E402
from exp_lib import Stage  # noqa: E402

from t_boost._t_boost import _MultiClassTableModel  # noqa: E402

name, split, n_bags = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tag = sys.argv[4] if len(sys.argv) > 4 else ""
st = Stage(name, split, n_bags, tag=tag)


def row_nll(model) -> np.ndarray:
    _, z = st.logits(model)
    m = z.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
    return lse - z[np.arange(len(st.y_te)), st.y_te]


w = st.w_te
lu, lb = row_nll(st.U), row_nll(st.B)


def paired(a: np.ndarray, b: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    d = (a - b) * w / w.mean()
    base = float(np.mean(ref * w) / w.mean())
    return 100 * float(d.mean()) / base, 100 * float(d.std(ddof=1) / np.sqrt(d.size)) / base


out = {"dataset": name, "split": split, "n_bags": n_bags, "tag": tag}
out["band_vs_none"] = paired(lb, lu, lu)
for f in sorted(st.dir.glob("band_*.bin")):
    label = f.stem
    m = _MultiClassTableModel.from_bytes(f.read_bytes())
    out[f"{label}_vs_shipped"] = paired(row_nll(m), lb, lb)
    st.forget(m)
append_jsonl(RES / "paired_se.jsonl", out)
print(out, flush=True)
