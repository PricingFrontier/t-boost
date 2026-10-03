"""Canonical ranking + deviance metrics — numpy-only (no scikit-learn dependency).

The ranking metrics are Python ports of the Rust `xtask` implementations and match
`xtask/src/main.rs` (`ordered_gini` / `lift_curve` / `concentration_gini`). Weight-aware;
NaN/degenerate inputs degrade to 0 just like the Rust (`finite_or_zero`).

The deviance metrics (`mean_tweedie_deviance` and its Poisson/Gamma wrappers) use the standard
GLM unit-deviance definitions — numerically identical to scikit-learn's functions of the same
names — so evaluating a t-boost fit needs nothing outside this package.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "ordered_gini",
    "lift_curve",
    "top_bucket_lift",
    "concentration_gini",
    "mean_tweedie_deviance",
    "mean_poisson_deviance",
    "mean_gamma_deviance",
]

_EPS = float(np.finfo(np.float64).eps)


def _finite_or_zero(v: float) -> float:
    return float(v) if np.isfinite(v) else 0.0


def _total_order_asc_key(score: np.ndarray) -> np.ndarray:
    """Int64 sort key: ascending order on this key equals ascending IEEE-754 `totalOrder`,
    bit-for-bit matching Rust's `f64::total_cmp` (`left ^= (((left >> 63) as u64) >> 1) as i64`).
    Unlike a plain float compare, this distinguishes signed zero and signed NaN:
    -NaN < -inf < ... < -0.0 < +0.0 < ... < +inf < +NaN.
    """
    bits = np.ascontiguousarray(score, dtype=np.float64).view(np.int64)
    shifted = bits >> np.int64(63)  # arithmetic shift: -1 if the sign bit is set, else 0
    mask = (shifted.view(np.uint64) >> np.uint64(1)).view(np.int64)  # 0x7FFF...FFFF or 0
    key: np.ndarray = bits ^ mask
    return key


def _order_desc(score: np.ndarray) -> np.ndarray:
    """Indices sorting `score` descending, ties broken by ascending index — matching the Rust
    `score[b].total_cmp(&score[a]).then(a.cmp(&b))`, including its signed NaN/zero total order.

    A plain `-score` key (the previous implementation) agrees with `total_cmp` everywhere except
    NaN: numpy always sorts NaN last regardless of sign, while `total_cmp` ranks +NaN first and
    -NaN last. `~_total_order_asc_key(score)` is a strictly order-reversing bitwise NOT (no
    overflow, unlike negating the raw int64 key), so ascending-sorting it reproduces the Rust
    descending comparator exactly.
    """
    n = score.shape[0]
    return np.lexsort((np.arange(n), ~_total_order_asc_key(score)))


def concentration_gini(y: Any, score: Any, weight: Any | None = None) -> float:
    """Weight-aware concentration (Lorenz) Gini of `y` when ranked by `score`, descending.

    Mirrors the Rust `concentration_gini`: `y` and `weight` are clamped at 0, the curve is the
    weighted cumulative share of `y` vs the weighted cumulative share of rows, and the result is
    `2·area − 1` (trapezoid rule). Returns 0 for empty/degenerate input.
    """
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    n = y.shape[0]
    weight = np.ones(n, dtype=np.float64) if weight is None else np.asarray(weight, dtype=np.float64)
    if n == 0 or score.shape[0] != n or weight.shape[0] != n:
        return 0.0
    w = np.maximum(weight, 0.0)
    yc = np.maximum(y, 0.0)
    total_w = float(w.sum())
    total_yw = float((yc * w).sum())
    if total_w <= 0.0 or total_yw <= 0.0:
        return 0.0
    order = _order_desc(score)
    ws = w[order]
    ys = yc[order] * ws
    # A tied score carries no ordering evidence. Traverse each tie as one
    # straight segment, independent of the input row order within that group.
    ranked = score[order]
    starts = np.r_[0, np.flatnonzero(ranked[1:] != ranked[:-1]) + 1]
    ws = np.add.reduceat(ws, starts)
    ys = np.add.reduceat(ys, starts)
    x = np.cumsum(ws) / total_w
    yy = np.cumsum(ys) / total_yw
    prev_x = np.concatenate(([0.0], x[:-1]))
    prev_y = np.concatenate(([0.0], yy[:-1]))
    area = float(np.sum((x - prev_x) * (yy + prev_y) * 0.5))
    return _finite_or_zero(2.0 * area - 1.0)


def ordered_gini(y: Any, pred: Any, weight: Any | None = None) -> float:
    """Concentration Gini of `y` ranked by `pred`, normalized by the perfect-model Gini (ranking by
    `y` itself). 1.0 = perfect ranking. Mirrors the Rust `ordered_gini`; the release-gate
    `ordered_gini` field.
    """
    y = np.asarray(y, dtype=np.float64)
    model = concentration_gini(y, pred, weight)
    perfect = concentration_gini(y, y, weight)
    if abs(perfect) <= _EPS:
        return 0.0
    return _finite_or_zero(model / perfect)


def lift_curve(
    y: Any, pred: Any, weight: Any | None = None, buckets: int = 10
) -> list[dict[str, float]]:
    """Weight-aware lift curve: rows ranked by `pred` descending, split into `buckets` equal-count
    groups; each group reports weighted `mean_y`, `mean_pred`, and `lift = mean_y / overall_mean_y`.
    Mirrors the Rust `lift_curve` (note: `y` is NOT clamped here, matching the Rust). Returns `[]`
    for empty/degenerate input.
    """
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    n = y.shape[0]
    weight = np.ones(n, dtype=np.float64) if weight is None else np.asarray(weight, dtype=np.float64)
    if n == 0 or pred.shape[0] != n or weight.shape[0] != n or buckets == 0:
        return []
    w = np.maximum(weight, 0.0)
    total_w = float(w.sum())
    total_yw = float((y * w).sum())
    overall = total_yw / total_w if total_w > 0.0 else 0.0
    order = _order_desc(pred)
    out: list[dict[str, float]] = []
    for bucket in range(buckets):
        start = bucket * n // buckets
        end = min((bucket + 1) * n // buckets, n)
        if start >= end:
            continue
        idx = order[start:end]
        wb = w[idx]
        w_sum = float(wb.sum())
        mean_y = float((y[idx] * wb).sum() / w_sum) if w_sum > 0.0 else 0.0
        mean_pred = float((pred[idx] * wb).sum() / w_sum) if w_sum > 0.0 else 0.0
        lift = mean_y / overall if abs(overall) > _EPS else 0.0
        out.append(
            {
                "bucket": float(bucket + 1),
                "rows": float(end - start),
                "mean_y": _finite_or_zero(mean_y),
                "mean_pred": _finite_or_zero(mean_pred),
                "lift": _finite_or_zero(lift),
            }
        )
    return out


def top_bucket_lift(y: Any, pred: Any, weight: Any | None = None, buckets: int = 10) -> float:
    """The lift of the top (highest-`pred`) bucket — the natural scalar `lift` for the release-gate
    rival-adapter artifact. 0.0 if the curve is empty.
    """
    curve = lift_curve(y, pred, weight, buckets)
    return curve[0]["lift"] if curve else 0.0


def mean_tweedie_deviance(y: Any, pred: Any, weight: Any | None = None, power: float = 0.0) -> float:
    """Weight-averaged Tweedie unit deviance of predictions `pred` against observations `y`.

    The standard GLM deviance (numerically identical to scikit-learn's
    ``mean_tweedie_deviance``): the weighted mean of the per-row unit deviance for the
    Tweedie family with variance power ``power`` — 0 = squared error, 1 = Poisson,
    2 = Gamma, 1 < power < 2 = compound Poisson-Gamma. Lower is better; the natural
    goodness-of-fit for the matching t-boost ``objective``.

    Unlike the ranking metrics above, domain violations raise ``ValueError`` rather than
    degrade to 0 — a deviance evaluated outside its family's support is a bug in the
    caller, not a degenerate ranking.
    """
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if y.ndim != 1 or pred.ndim != 1 or y.size == 0:
        raise ValueError("y and pred must be nonempty vectors")
    n = y.shape[0]
    if pred.shape[0] != n:
        raise ValueError(f"pred has {pred.shape[0]} rows but y has {n}")
    weight = np.ones(n, dtype=np.float64) if weight is None else np.asarray(weight, dtype=np.float64)
    if weight.shape != (n,):
        raise ValueError(f"weight must contain {n} values")
    if not all(np.all(np.isfinite(v)) for v in (y, pred, weight)):
        raise ValueError("y, pred, and weight must be finite")
    if np.any(weight < 0) or weight.sum() <= 0:
        raise ValueError("weight must be non-negative with positive total mass")

    p = float(power)
    if not np.isfinite(p) or p < 0.0 or 0.0 < p < 1.0:
        raise ValueError(f"power={p} is not supported (use 0, or any power >= 1)")
    if p == 0.0:
        dev = (y - pred) ** 2
    else:
        if np.any(pred <= 0.0):
            raise ValueError(f"predictions must be strictly positive for power={p}")
        if p == 1.0:
            if np.any(y < 0.0):
                raise ValueError("y must be non-negative for power=1 (Poisson)")
            # xlogy semantics: the y=0 rows contribute no y*log(y/pred) term.
            ylog = np.zeros(n, dtype=np.float64)
            pos = y > 0.0
            ylog[pos] = y[pos] * np.log(y[pos] / pred[pos])
            dev = 2.0 * (ylog - y + pred)
        elif p == 2.0:
            if np.any(y <= 0.0):
                raise ValueError("y must be strictly positive for power=2 (Gamma)")
            dev = 2.0 * (np.log(pred / y) + y / pred - 1.0)
        else:
            if p < 2.0 and np.any(y < 0.0):
                raise ValueError(f"y must be non-negative for power={p}")
            if p > 2.0 and np.any(y <= 0.0):
                raise ValueError(f"y must be strictly positive for power={p}")
            dev = 2.0 * (
                np.power(np.maximum(y, 0.0), 2.0 - p) / ((1.0 - p) * (2.0 - p))
                - y * np.power(pred, 1.0 - p) / (1.0 - p)
                + np.power(pred, 2.0 - p) / (2.0 - p)
            )
    return float(np.average(dev, weights=weight))


def mean_poisson_deviance(y: Any, pred: Any, weight: Any | None = None) -> float:
    """Weight-averaged Poisson unit deviance — `mean_tweedie_deviance` at ``power=1``, the
    goodness-of-fit for ``objective="poisson"`` frequency models."""
    return mean_tweedie_deviance(y, pred, weight, power=1.0)


def mean_gamma_deviance(y: Any, pred: Any, weight: Any | None = None) -> float:
    """Weight-averaged Gamma unit deviance — `mean_tweedie_deviance` at ``power=2``, the
    goodness-of-fit for ``objective="gamma"`` severity models."""
    return mean_tweedie_deviance(y, pred, weight, power=2.0)
