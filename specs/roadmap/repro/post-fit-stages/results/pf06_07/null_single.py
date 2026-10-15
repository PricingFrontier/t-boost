"""The unit-free null-space refusal rule on single-output fits, as it would ship (brake on):
each fit runs twice on the same split and seed, with the shipped `_gcv_smooth` and with a version
that keeps a table as fitted when GCV is lowest at the penalty's null-space limit (alpha -> inf).
Usage: null_single.py <dataset,...> <split,...>"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, Timer, append_jsonl, tb  # noqa: E402

from t_boost.sklearn import _BaseTBoost  # noqa: E402

SHIPPED = _BaseTBoost._gcv_smooth.__func__


def null_rule_gcv(cls, v, s, penalty, fixed_alpha, family=None):  # type: ignore[no-untyped-def]
    v_s, alpha = SHIPPED(cls, v, s, penalty, fixed_alpha, family=family)
    if alpha <= 0.0 or fixed_alpha is not None:
        return v_s, alpha
    fam = family if family is not None else (
        cls._dr_family(v, s, penalty) if penalty is not None else None)
    if fam is None:
        return v_s, alpha
    n = int(np.sum(s > 0))

    def gcv(a: float) -> float | None:
        vs, df = fam(a)
        d = 1.0 - df / n
        if d <= 1e-6 or not np.all(np.isfinite(vs)):
            return None
        return float(np.sum(s * (v - vs) ** 2)) / (d * d) / n

    g_inf = gcv(1e12)
    g_min = min((g for g in (gcv(float(a)) for a in cls._GRADUATION_ALPHAS) if g is not None),
                default=None)
    if g_inf is not None and g_min is not None and g_inf <= g_min:
        return v, 0.0
    return v_s, alpha


names = sys.argv[1].split(",")
splits = [int(x) for x in sys.argv[2].split(",")]
for name in names:
    for split in splits:
        d = tb.load(name, split)
        task = d["task"]
        X, y, e, w = d["test"]
        out = {"dataset": name, "split": split}
        for label in ("shipped", "null_refuse"):
            _BaseTBoost._gcv_smooth = classmethod(SHIPPED if label == "shipped" else null_rule_gcv)
            t = Timer()
            est = tb.fit(task, {}, d["train"], 8)
            gv = getattr(est, "graduation_validation_", {}) or {}
            out[label] = {"test": tb.deviance(task, y, tb.predict(task, est, X), e, w),
                          "adopted": gv.get("adopted"), "n_smoothed": gv.get("n_smoothed"),
                          "fit_s": t.lap()}
        _BaseTBoost._gcv_smooth = classmethod(SHIPPED)
        out["d_test_pct"] = 100 * (out["null_refuse"]["test"] - out["shipped"]["test"]) / out["shipped"]["test"]
        print(out, flush=True)
        append_jsonl(RES / "null_single.jsonl", out)
