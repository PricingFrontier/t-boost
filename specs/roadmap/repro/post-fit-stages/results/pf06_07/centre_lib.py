"""PF-07 prototype: graduation of the softmax-centred class tables.

A multiclass model's class logits are defined up to a common function g(x) (adding g to every class
leaves every probability unchanged). Per-class GCV is not invariant to that gauge: it smooths each
class's share of the common component with its own alpha, which turns part of a free move into a
differential (costly) one, and the common component's roughness steers each class's alpha choice.

Centring: on every support u, map each class table onto the union of the classes' grids, take the
common component c = sum_k pi_k f_k (pi = class priors, any fixed weights summing to 1 make the
result gauge-invariant), project c back onto each class grid (row-mass-weighted means), smooth
d_k = f_k - c_k with the shipped GCV, and add c_k back unsmoothed.

Numeric axes only map through borders; a categorical axis must have the same level cells in every
class (otherwise the support falls back to the shipped per-class rule).
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import threadpoolctl

from grad_lib import select_shipped, table_curve


def _axis_maps(models_axes: list[list[dict]], r: int) -> tuple[list[np.ndarray], int, Any] | None:
    """For raw feature r: per class, the class cell of every union cell; the union cell count;
    and the union borders (numeric) or None (categorical, identical in all classes)."""
    axes = [ma[r] for ma in models_axes]
    if axes[0].get("levels") is not None:
        ref = json.dumps(axes[0])
        if any(json.dumps(a) != ref for a in axes[1:]):
            return None
        n = int(axes[0]["cells"])
        return [np.arange(n) for _ in axes], n, None
    borders = [np.asarray(a["borders"], dtype=float) for a in axes]
    union = np.unique(np.concatenate(borders)) if borders else np.zeros(0)
    # union cells: 0 = missing, then len(union)+1 intervals; a representative point per interval
    edges = np.concatenate([[union[0] - 1.0] if union.size else [0.0],
                            (union[:-1] + union[1:]) / 2.0 if union.size > 1 else [],
                            [union[-1] + 1.0] if union.size else []])
    maps = []
    for b in borders:
        cls = np.searchsorted(b, edges, side="right") + 1
        maps.append(np.concatenate([[0], cls]).astype(np.int64))
    return maps, union.size + 2, union


def _union_cells(x: np.ndarray, union: Any, n: int) -> np.ndarray:
    """Union cell of every row's raw value (numeric: 0 = missing, then intervals)."""
    out = np.zeros(x.shape[0], dtype=np.int64)
    ok = np.isfinite(x)
    out[ok] = np.searchsorted(union, x[ok], side="right") + 1
    return np.minimum(out, n - 1)


def centred_updates(st: Any, model: Any, priors: np.ndarray) -> tuple[list[tuple], dict]:
    """Updates (k, idx, values, alpha) of the centred rule on `model` (unbanded: merged grids)."""
    est = st.est
    cls = type(est)
    K = st.K
    w = st.w_tr.astype(np.float64)
    w_scale = 1.0 if est._measure_kwargs()["ref_measure"] == "exposure" else float(w.mean())
    axes = [json.loads(model.merged_axes(k)) for k in range(K)]
    tables = model.graduation_tables(cls._GRADUATION_MAX_CELLS_RAW)
    by_u: dict[tuple, dict[int, tuple]] = {}
    for k, ct in enumerate(tables):
        for it in ct:
            by_u.setdefault(tuple(it[1]), {})[k] = it
    raw_names = model.raw_feature_names()
    cols = {c: i for i, c in enumerate(st.Xtr.columns)}
    union_rows: dict[int, tuple] = {}
    cell_cache: dict[int, np.ndarray] = {}

    def model_cells(k: int) -> np.ndarray:
        if k not in cell_cache:
            cell_cache[k] = np.asarray(model.cell_indices(st.x_tr, cat_x=st.cat_tr, class_index=k))
        return cell_cache[k]

    def rows_cells(r: int, union: Any, n: int) -> np.ndarray | None:
        if r in union_rows:
            return union_rows[r]
        name = raw_names[r]
        if name not in cols:
            union_rows[r] = None
            return None
        x = st.Xtr[name].to_numpy()
        if union is None:  # categorical: the model's own cells (same in every class)
            union_rows[r] = None
            return None
        cells = _union_cells(np.asarray(x, dtype=float), union, n)
        # the union binning must agree with the model's own cells in every class
        for k in range(K):
            mk = _axis_maps(axes, r)[0][k]
            own = model_cells(k)[:, r]
            agree = float(np.mean(mk[cells] == own))
            stats["agree_min"] = min(stats.get("agree_min", 1.0), agree)
        union_rows[r] = cells
        return cells

    ups: list[tuple] = []
    stats = {"supports": 0, "centred": 0, "fallback": 0}
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        for u, per_k in by_u.items():
            stats["supports"] += 1
            maps = []
            ok = True
            for r in u:
                m = _axis_maps(axes, r)
                if m is None:
                    ok = False
                    break
                maps.append(m)
            # union supports from the rows (numeric axes); categorical axes keep the class cells
            if ok:
                ushape = tuple(m[1] for m in maps)
                cell_ids = []
                for r, m in zip(u, maps):
                    if m[2] is None:
                        # categorical: rows' cells are the class cells; read them from the model
                        cell_ids.append(None)
                    else:
                        c = rows_cells(r, m[2], m[1])
                        if c is None:
                            ok = False
                            break
                        cell_ids.append(c)
            if ok and any(c is None for c in cell_ids):
                # a categorical axis (same cells in every class): the rows' cells from the model
                ci = model_cells(0)
                for j, r in enumerate(u):
                    if cell_ids[j] is None:
                        cell_ids[j] = ci[:, r].astype(np.int64)
            if not ok:
                for k, it in per_k.items():
                    rec = table_curve(cls, it, w_scale)
                    if rec is None:
                        continue
                    vals, a = select_shipped(rec)
                    if a > 0:
                        ups.append((k, rec["idx"], vals.tolist(), a))
                stats["fallback"] += 1
                continue
            flat = np.ravel_multi_index(tuple(cell_ids), ushape)
            S = np.bincount(flat, weights=w, minlength=int(np.prod(ushape))).reshape(ushape)
            # every class's table on the union grid (absent table = 0)
            V = np.zeros((K,) + ushape)
            for k in range(K):
                it = per_k.get(k)
                if it is None:
                    continue
                v = np.asarray(it[3], dtype=float).reshape(it[2])
                idx = np.ix_(*[m[0][k] for m in maps])
                if any(m[0][k].max() >= s for m, s in zip(maps, v.shape)):
                    ok = False
                    break
                V[k] = v[idx]
            if not ok:
                stats["fallback"] += 1
                continue
            c = np.tensordot(priors, V, axes=1)
            stats["centred"] += 1
            for k, it in per_k.items():
                shape = tuple(it[2])
                # project c back onto class k's grid: row-mass-weighted means over its union cells
                size = int(np.prod(shape))
                grids = np.meshgrid(*[m[0][k] for m in maps], indexing="ij")
                tgt = np.ravel_multi_index(tuple(g.ravel() for g in grids), shape)
                ck_flat = np.bincount(tgt, weights=(S * c).ravel(), minlength=size)
                m_flat = np.bincount(tgt, weights=S.ravel(), minlength=size)
                c_flat = np.bincount(tgt, weights=c.ravel(), minlength=size)
                n_flat = np.bincount(tgt, minlength=size)
                ckv = np.where(m_flat > 0, ck_flat / np.where(m_flat > 0, m_flat, 1.0),
                               c_flat / np.maximum(n_flat, 1))
                v = np.asarray(it[3], dtype=float)
                d_item = (it[0], it[1], it[2], (v - ckv).tolist(), it[4], it[5])
                rec = table_curve(cls, d_item, w_scale)
                if rec is None:
                    continue
                dvals, a = select_shipped(rec)
                if a > 0:
                    ups.append((k, rec["idx"], (dvals + ckv).tolist(), a))
    return ups, stats
