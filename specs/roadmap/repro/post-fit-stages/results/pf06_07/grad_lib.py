"""PF-07: graduation of a multiclass table model with the per-table GCV curves exposed.

`curves(est, model, w)` mirrors `_graduate_table_model` + `_graduate_one` + `_gcv_smooth`
(sklearn.py) but keeps every alpha's (gcv, smoothed values, df), so a selection rule can be
swapped. `select_shipped` reproduces the shipped rule exactly (checked against `_graduate_one`).
`apply(st, model, updates)` builds the native candidate (re-anchored) with the 1% brake lifted
(env TBOOST_EXP_GRAD_SLACK, experiment build only) and reports the training-deviance ratio the
brake would have judged.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import threadpoolctl

from exp_lib import N_JOBS, ce_logits


def _curve(cls: Any, v: np.ndarray, s: np.ndarray, penalty: np.ndarray | None,
           family: Callable | None, keep: dict | None = None
           ) -> list[tuple[float, float, np.ndarray, float]] | None:
    n_data = int(np.sum(s > 0))
    if n_data < 3 or (penalty is not None and penalty.shape[0] != v.shape[0]):
        return None
    if family is None:
        if penalty is None:
            return None
        family = cls._dr_family(v, s, penalty)
    if keep is not None:
        keep["family"] = family
    out = []
    for alpha in cls._GRADUATION_ALPHAS:
        if family is not None:
            v_s, df = family(float(alpha))
        else:
            w_mat = np.diag(s)
            try:
                x_mat = np.linalg.solve(w_mat + float(alpha) * penalty, w_mat)
            except np.linalg.LinAlgError:
                continue
            v_s, df = x_mat @ v, float(np.trace(x_mat))
        denom = 1.0 - df / n_data
        if denom <= 1e-6 or not np.all(np.isfinite(v_s)):
            continue
        gcv = float(np.sum(s * (v - v_s) ** 2)) / (denom * denom) / n_data
        out.append((gcv, float(alpha), v_s, float(df)))
    return out or None


def table_curve(cls: Any, item: tuple, w_scale: float) -> dict[str, Any] | None:
    """`_graduate_one` with the GCV curve kept. Returns None for the same tables it skips."""
    table_idx, features, shape, values, support, axis_cat = item
    v = np.asarray(values, dtype=float)
    sup = np.asarray(support, dtype=float) * w_scale
    rec: dict[str, Any] = {"idx": int(table_idx), "features": tuple(features),
                           "shape": tuple(shape), "v": v, "sup": sup}
    if len(shape) == 1:
        if axis_cat and axis_cat[0]:
            return None
        if v.size > cls._GRADUATION_MAX_CELLS:
            return None
        start = 1 if v.size > 1 else 0
        pen_d = cls._d2(v.size - start)
        pen = pen_d.T @ pen_d
        vv, ss = v[start:], sup[start:]
        kept: dict = {}
        curve = _curve(cls, vv, ss, pen, None, kept)

        def assemble(v_s: np.ndarray, alpha: float) -> np.ndarray:
            if alpha > 0.0:
                v_s = cls._clamp_extrapolated_fill(v_s, ss)
            out = v.copy()
            out[start:] = v_s
            return out
        rec.update(kind="1d", vv=vv, ss=ss)
    elif len(shape) == 2:
        if len(axis_cat) == 2 and all(axis_cat):
            return None
        v2, s2 = v.reshape(shape), sup.reshape(shape)
        r0 = 1 if (shape[0] > 1 and not (axis_cat and axis_cat[0])) else 0
        c0 = 1 if (shape[1] > 1 and not (len(axis_cat) == 2 and axis_cat[1])) else 0
        vv2, ss2 = v2[r0:, c0:], s2[r0:, c0:]
        m, n = vv2.shape
        if m * n < 4 or (m < 3 and n < 3):
            return None
        row_smooth = m >= 3 and not (axis_cat and axis_cat[0])
        col_smooth = n >= 3 and not (len(axis_cat) == 2 and axis_cat[1])
        if not (row_smooth or col_smooth):
            return None
        if row_smooth and col_smooth:
            if m * n > cls._GRADUATION_MAX_CELLS:
                return None
        else:
            chain_len = m if row_smooth else n
            if chain_len > cls._GRADUATION_MAX_SEPARABLE_CHAIN:
                return None
        family = None
        pen = None
        if row_smooth != col_smooth:
            v_o = np.ascontiguousarray(vv2 if row_smooth else vv2.T)
            s_o = np.ascontiguousarray(ss2 if row_smooth else ss2.T)
            sep = cls._dr_family_separable(v_o, s_o)
            if sep is not None:
                if row_smooth:
                    family = sep
                else:
                    def family(alpha: float, _sep: Any = sep,
                               _shape: tuple[int, ...] = v_o.shape) -> tuple[np.ndarray, float]:
                        v_s, df = _sep(alpha)
                        return v_s.reshape(_shape).T.ravel(), df
        if family is None:
            pen = np.zeros((m * n, m * n))
            if row_smooth:
                d = cls._d2(m)
                pen += np.kron(d.T @ d, np.eye(n))
            if col_smooth:
                d = cls._d2(n)
                pen += np.kron(np.eye(m), d.T @ d)
        vv, ss = vv2.ravel(), ss2.ravel()
        kept = {}
        curve = _curve(cls, vv, ss, pen, family, kept)

        def assemble(v_s: np.ndarray, alpha: float) -> np.ndarray:
            if alpha > 0.0:
                v_s = cls._clamp_extrapolated_fill(v_s, ss)
            out2 = v2.copy()
            out2[r0:, c0:] = v_s.reshape(m, n)
            return out2.ravel()
        rec.update(kind="2d", vv=vv, ss=ss, axis_cat=tuple(axis_cat))
    else:
        return None
    rec["curve"] = curve
    rec["family_fn"] = kept.get("family")
    rec["assemble"] = assemble
    rec["n_data"] = int(np.sum(rec["ss"] > 0))
    return rec


def curves(est: Any, model: Any, w: np.ndarray) -> list[dict[str, Any]]:
    """Every eligible table of every class bank, with its GCV curve (`k` = class)."""
    cls = type(est)
    w_scale = float(np.sum(w)) / float(max(len(w), 1))
    if est._measure_kwargs()["ref_measure"] == "exposure":
        w_scale = 1.0
    tables = model.graduation_tables(cls._GRADUATION_MAX_CELLS_RAW)
    out = []
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        for k, class_tables in enumerate(tables):
            for it in class_tables:
                rec = table_curve(cls, it, w_scale)
                if rec is not None:
                    rec["k"] = k
                    out.append(rec)
    return out


def select_shipped(rec: dict[str, Any]) -> tuple[np.ndarray, float]:
    """`_gcv_smooth`'s rule on the kept curve: (full table values, alpha)."""
    c = rec["curve"]
    if not c:
        return rec["v"], 0.0
    srt = sorted(c, key=lambda t: (t[0], t[1]))
    g_min, alpha, v_s, _ = srt[0]
    g_small = min(c, key=lambda t: t[1])[0]
    if g_min < g_small:
        return rec["assemble"](v_s, alpha), alpha
    return rec["v"], 0.0


def gcv_null(rec: dict[str, Any], cls: Any) -> float | None:
    """GCV at the penalty's null-space limit (alpha -> infinity: the weighted fit of the penalty's
    null space, straight lines along each smoothed axis), evaluated at a huge alpha through the
    table's own ridge family. Unit-free: the limit does not depend on the scale of the weights."""
    fam = rec.get("family_fn")
    if fam is None:
        return None
    try:
        v_s, df = fam(1e12)
    except Exception:  # noqa: BLE001 - numerical failure: no verdict
        return None
    n = rec["n_data"]
    denom = 1.0 - df / n
    if denom <= 1e-6 or not np.all(np.isfinite(v_s)):
        return None
    return float(np.sum(rec["ss"] * (rec["vv"] - v_s) ** 2)) / (denom * denom) / n


def select_null_refuse(rec: dict[str, Any], cls: Any) -> tuple[np.ndarray, float]:
    """The shipped rule, except that a table whose GCV is lowest at the null-space limit (the
    straight line GCV prefers but the grid can only approach) is kept as fitted."""
    vals, a = select_shipped(rec)
    if a <= 0:
        return vals, a
    g_inf = gcv_null(rec, cls)
    g_min = min(t[0] for t in rec["curve"])
    if g_inf is not None and g_inf <= g_min:
        return rec["v"], 0.0
    return vals, a


def at_alpha(rec: dict[str, Any], alpha: float) -> np.ndarray:
    """The table smoothed at exactly `alpha` (0 = as fitted)."""
    if alpha <= 0.0 or not rec["curve"]:
        return rec["v"]
    for _, a, v_s, _ in rec["curve"]:
        if a == alpha:
            return rec["assemble"](v_s, alpha)
    return rec["v"]  # alpha infeasible for this table (df too large): keep as fitted


def apply(st: Any, model: Any, updates: list[tuple[int, int, list[float]]]) -> tuple[Any, float]:
    """Native candidate (re-anchored per class) with the brake lifted, and its training
    multinomial-deviance ratio against `model` (what the 1% brake compares)."""
    os.environ["TBOOST_EXP_GRAD_SLACK"] = "1e300"
    try:
        cand, adopted, _ = model.apply_graduation(
            st.x_tr, st.y_tr, st.w_tr, updates, alpha=0.0, cat_x=st.cat_tr, n_jobs=N_JOBS,
            box_budget=max(0, int(st.est.prune_box_budget or 0)))
    finally:
        os.environ.pop("TBOOST_EXP_GRAD_SLACK", None)
    a, _ = st.logits(model)
    b, _ = st.logits(cand)
    w = st.w_tr.astype(np.float64)
    ratio = ce_logits(b, st.yi_tr, w) / ce_logits(a, st.yi_tr, w)
    return cand, ratio
