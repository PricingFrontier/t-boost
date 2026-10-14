"""The model report (`pricing_report`, version 2): everything a reviewer needs to decide whether
a fitted model is ready for production, as plain JSON.

`build_report` assembles it from the estimator and one or two datasets; `_html_report.report_html`
renders it. The report is the record (log it as JSON), the page is a view of it. Every number the
page shows is computed here, and every statistic follows a stated convention:

* deviance: the objective's unit deviance of ``y`` against the expected value per row (exposure
  and offset included), weighted by ``sample_weight`` only — the convention of the fit's own
  deviance, so the training curve and the fit-quality table agree;
* null deviance: the best constant model on the same rows, with the same exposure and offset;
* Gini: the Lorenz (concentration) Gini of ``y`` ranked by predicted rate, with the exposure as
  the x-axis mass for a log-link model; ``gini_normalised`` divides it by the perfect ranking's;
* lift: up to ``_LIFT_BANDS`` bands of equal mass, ascending predicted rate, tied predictions kept
  in one band (deterministic, independent of row order);
* PSI: per feature on at most ``_PSI_BINS`` bins of the merged grid, shares floored at
  ``_PSI_FLOOR``.

The report describes the data passed to it; it cannot verify which rows were held out.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from . import metrics
from ._html_report import run_label
from ._t_boost import _report_cell_quantiles, _report_cell_sums, _report_ranking

if TYPE_CHECKING:
    from .sklearn import _BaseTBoost

REPORT_VERSION = 2

DEFAULT_THRESHOLDS: dict[str, float] = {
    "ae_train": 0.02,
    "ae_validation": 0.05,
    "ae_test": 0.05,
    "factor_ae": 0.10,
    "factor_min_mass": 0.01,
    "factor_z": 3.0,
    "decile_ae": 0.05,
    "near_miss_z": 2.0,
    "missing_effect": 0.05,
    "thin_cell_mass": 0.001,
    "psi_warn": 0.10,
    "psi_critical": 0.25,
    "train_gap": 0.05,
    "unseen_mass": 0.01,
}
THRESHOLD_DESCRIPTIONS: dict[str, str] = {
    "ae_train": "overall A/E on the train data within 1 ± value",
    "ae_validation": "overall A/E on the validation data within 1 ± value",
    "ae_test": "overall A/E on the test data within 1 ± value",
    "factor_ae": "factor bands whose A/E is off by more than value",
    "factor_min_mass": "only bands holding at least this share of the mass",
    "factor_z": ("only bands and deciles whose A - E is at least value standard deviations "
                 "from 0 (None: the tolerance alone)"),
    "decile_ae": "predicted-rate deciles whose A/E is off by more than value (and |z| ≥ factor_z)",
    "near_miss_z": "bands and deciles off by more than their tolerance with z ≥ value (below "
                   "factor_z) are listed as near misses",
    "missing_effect": "a missing-value level with no exposure whose effect differs from no "
                      "effect by more than value",
    "thin_cell_mass": "one-way table cells under this share of the table's exposure count as thin",
    "psi_warn": "PSI per feature above value: warning",
    "psi_critical": "PSI per feature above value: critical",
    "train_gap": "drop in deviance explained from train to each later dataset",
    "unseen_mass": "mass share of unseen categorical values in a dataset",
}

_LIFT_BANDS = 10
_LORENZ_POINTS = 101
_PSI_BINS = 10
_PSI_FLOOR = 1e-4
_PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
_HISTOGRAM_BINS = 40
# The spread of each level's Shapley value across policies, and of an interaction's effect
# across exposure.
_SHAP_QUANTILES = (0.1, 0.9)
_SPREAD_QUANTILES = (0.05, 0.95)
_MAX_KEY = 250
_PROB_CLIP = 1e-12


# --------------------------------------------------------------------------- arguments --------

def resolve_thresholds(thresholds: dict[str, Any] | None) -> dict[str, float | None]:
    """The defaults overridden by ``thresholds``: unknown keys, non-finite or negative values and
    ``psi_warn > psi_critical`` raise ``ValueError``; ``None`` disables a check."""
    out: dict[str, float | None] = dict(DEFAULT_THRESHOLDS)
    if thresholds is None:
        return out
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be a dict")
    unknown = sorted(set(thresholds) - set(DEFAULT_THRESHOLDS))
    if unknown:
        raise ValueError(
            f"unknown threshold(s) {unknown}; known: {sorted(DEFAULT_THRESHOLDS)}"
        )
    for key, value in thresholds.items():
        if value is None:
            out[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"threshold {key} must be a number or None, got {value!r}")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"threshold {key} must be finite and non-negative, got {value!r}")
        out[key] = float(value)
    warn, crit = out["psi_warn"], out["psi_critical"]
    if warn is not None and crit is not None and warn > crit:
        raise ValueError(f"psi_warn ({warn}) must not exceed psi_critical ({crit})")
    return out


def dataset_roles(n: int) -> tuple[str, ...]:
    """The role of each of ``n`` datasets: train, then test (two datasets), or train,
    validation and test (three). t-boost's bags early-stop and prune on rows they hold back, so
    a separate validation dataset is the exception, not the default."""
    return ("train", "test") if n == 2 else ("train", "validation", "test")[:n]


def resolve_labels(labels: Any, n: int) -> list[str]:
    """The names of the ``n`` datasets: ``labels`` (the first ``n``), which must be distinct
    non-empty strings, or by default their roles (`dataset_roles`)."""
    if labels is None:
        return list(dataset_roles(n))
    if (not isinstance(labels, (tuple, list)) or len(labels) < n
            or not all(isinstance(v, str) and v.strip() for v in labels[:n])
            or len(set(labels[:n])) != n):
        raise ValueError(
            f"labels must give {n} distinct non-empty strings, one per dataset, e.g. "
            f"{dataset_roles(n)!r}"
        )
    return [str(v) for v in labels[:n]]


def resolve_eval_sets(eval_set: Any, weights: Any, exposures: Any, offsets: Any
                      ) -> list[tuple[Any, Any, Any, Any, Any]]:
    """``eval_set`` as ``(X, y, sample_weight, exposure, offset)`` per dataset: one ``(X, y)``
    pair (the test data, by default) with single vectors, or a list of up to two pairs
    (validation, then test) with the ``eval_*`` vectors as lists of the same length."""
    if eval_set is None:
        if any(v is not None for v in (weights, exposures, offsets)):
            raise ValueError("eval_sample_weight, eval_exposure and eval_offset need an eval_set")
        return []
    if isinstance(eval_set, tuple):
        if len(eval_set) != 2:
            raise ValueError("eval_set must be a pair (X, y) or a list of up to two pairs")
        return [(eval_set[0], eval_set[1], weights, exposures, offsets)]
    if not isinstance(eval_set, list) or not 1 <= len(eval_set) <= 2 or not all(
            isinstance(pair, tuple) and len(pair) == 2 for pair in eval_set):
        raise ValueError(
            "eval_set must be a pair (X_test, y_test) or a list of up to two pairs: "
            "[(X_validation, y_validation), (X_test, y_test)]"
        )
    n = len(eval_set)
    per_set = []
    for name, value in (("eval_sample_weight", weights), ("eval_exposure", exposures),
                        ("eval_offset", offsets)):
        if value is None:
            per_set.append([None] * n)
        elif isinstance(value, list) and len(value) == n:
            per_set.append(value)
        else:
            raise ValueError(f"with a list eval_set, {name} must be a list of {n} entries "
                             "(None for a dataset without one)")
    return [(pair[0], pair[1], *(v[i] for v in per_set)) for i, pair in enumerate(eval_set)]


# --------------------------------------------------------------------------- JSON helpers -----

def json_safe(value: Any) -> Any:
    """``value`` as strict JSON: numpy types to Python, tuples to lists, non-finite floats to
    ``None``, dict keys to strings."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _ratio(num: float, den: float) -> float | None:
    return float(num / den) if den != 0 and math.isfinite(num / den) else None


# --------------------------------------------------------------------------- datasets ---------

class Prepared:
    """One dataset scored by the model: the per-row arrays every statistic reads."""

    label: str
    rows: int
    w: np.ndarray
    e: np.ndarray
    offset: np.ndarray | None
    y: np.ndarray            # single output: the target (0/1 for a binary classifier)
    codes: np.ndarray | None  # multiclass: class index per row
    pred: np.ndarray | None   # single output: expected value per row (exposure, offset applied)
    proba: np.ndarray | None  # multiclass: (n, K) probabilities
    cells: list[np.ndarray]   # per class bank (one entry for single output): (n, n_raw) cells
    n_cells: list[list[int]]
    x32: np.ndarray | None    # the scored design and its categorical codes (`_serve_kwargs`)
    cat_kw: dict[str, Any]
    names: list[str]
    unseen: list[dict[str, Any]]
    error: str | None


def _bank_models(est: "_BaseTBoost") -> tuple[Any, bool]:
    """The deployed table model and whether it is multiclass; raises for a tree ensemble saved by
    an older t-boost (it has no fixed cell grid)."""
    from ._t_boost import _MultiClassTableModel, _TableModel

    multi = getattr(est, "_multi_model", None)
    if multi is not None:
        if not isinstance(multi, _MultiClassTableModel):
            raise ValueError(
                "this multiclass model is a tree ensemble saved by an older t-boost, with no "
                "fixed cell grid to aggregate over. Refit it to get rating tables."
            )
        return multi, True
    model = getattr(est, "_model", None)
    if not isinstance(model, _TableModel):
        raise ValueError(
            "this model is a tree ensemble saved by an older t-boost, with no fixed cell grid "
            "to aggregate over. Refit it to get rating tables."
        )
    return model, False


def prepare(
    est: "_BaseTBoost", X: Any, y: Any, *, sample_weight: Any | None, exposure: Any | None,
    offset: Any | None, label: str = "data", check_unseen: bool = False,
) -> Prepared:
    """Validate one dataset and score it. With ``check_unseen``, unseen categorical values are
    counted first, and under ``unknown_category="error"`` the dataset is returned unscored with
    ``error`` set instead of raising."""
    from .sklearn import _as_float32_1d, _offset_array

    model, multiclass = _bank_models(est)
    X, vectors = est._resolve_fit_vectors(
        X, {"y": y, "sample_weight": sample_weight, "exposure": exposure, "offset": offset}
    )
    y, sample_weight, exposure, offset = (
        vectors[k] for k in ("y", "sample_weight", "exposure", "offset")
    )
    if multiclass and (exposure is not None or offset is not None):
        raise ValueError("exposure and offset are not supported for multiclass (K >= 3) models")
    for name, value, flag in (
        ("sample_weight", sample_weight, "_ae_requires_weight_"),
        ("exposure", exposure, "_ae_requires_exposure_"),
        ("offset", offset, "_ae_requires_offset_"),
    ):
        if value is None and getattr(est, flag, False):
            raise ValueError(f"actual_vs_expected requires explicit {name} aligned to X")
    p = Prepared()
    p.label, p.error, p.unseen = label, None, []
    if check_unseen and getattr(est, "_cat_indices_", None):
        frame = est.unseen_values(X, sample_weight=sample_weight, exposure=exposure)
        p.unseen = json_safe(frame.to_dicts())
        if p.unseen and getattr(est, "unknown_category", "rare") == "error":
            p.error = ("unknown_category='error' and the data hold categorical values the fit "
                       "never saw, so the model cannot score them")
    # Categoricals as codes into distinct labels: the `predict` path, no per-value Python.
    x32, cat_kw = est._serve_kwargs(X, model) if p.error is None else (None, {})
    n = int(x32.shape[0]) if x32 is not None else _n_rows(X)
    p.rows = n
    p.x32, p.cat_kw = x32, cat_kw
    classes = getattr(est, "classes_", None)
    if classes is not None:
        labels = np.asarray(y)
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels[:, 0]
        if labels.ndim != 1 or not np.all(np.isin(labels, classes)):
            raise ValueError("y must contain a vector of fitted class labels")
        codes = np.searchsorted(np.asarray(classes), labels)
        if multiclass:
            p.codes, yv = codes.astype(np.int64), codes.astype(np.float64)
        else:
            p.codes, yv = None, (labels == classes[1]).astype(np.float64)
    else:
        p.codes = None
        yv = _as_float32_1d(y, "y").astype(np.float64)
    if yv.shape[0] != n:
        raise ValueError(f"y has {yv.shape[0]} rows but X has {n}")
    w = (_as_float32_1d(sample_weight, "sample_weight").astype(np.float64)
         if sample_weight is not None else np.ones(n, dtype=np.float64))
    e = (_as_float32_1d(exposure, "exposure").astype(np.float64)
         if exposure is not None else np.ones(n, dtype=np.float64))
    for name, values in (("y", yv), ("sample_weight", w), ("exposure", e)):
        if values.shape != (n,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain {n} finite values")
        if name != "y" and np.any(values < 0):
            raise ValueError(f"{name} must be non-negative")
    p.y, p.w, p.e = yv, w, e
    p.offset = None if offset is None else _offset_array(offset, n)
    p.names = list(model.raw_feature_names())
    p.pred = p.proba = None
    p.cells, p.n_cells = [], []
    if p.error is not None:
        return p
    if multiclass:
        p.proba = np.asarray(est.predict_proba(X), dtype=np.float64)
        banks = range(int(model.n_classes))
    else:
        p.pred = np.asarray(est._expected_response(X, exposure, p.offset), dtype=np.float64)
        banks = range(1)
    for k in banks:
        kw = {"class_index": k} if multiclass else {}
        p.cells.append(np.ascontiguousarray(model.cell_indices(x32, **cat_kw, **kw),
                                            dtype=np.uint32))
        p.n_cells.append([int(a["cells"]) for a in json.loads(model.merged_axes(**kw))])
    return p


def _f64(a: Any) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def _n_rows(X: Any) -> int:
    shape = getattr(X, "shape", None)
    return int(shape[0]) if shape is not None else len(X)


def factor_ae(p: Prepared, bank: int = 0, *, actual: np.ndarray | None = None,
              expected: np.ndarray | None = None, variance: np.ndarray | None = None
              ) -> list[dict[str, Any]]:
    """Actual versus expected per merged-grid cell for every raw feature (one value per cell of
    the grid), summed natively. ``actual``/``expected`` default to ``y``/``pred`` (single
    output). With a per-row ``variance`` (dispersion × V(mu)), each cell also gets ``z``:
    how many standard deviations its actual - expected is from zero."""
    yv = p.y if actual is None else actual
    pred = p.pred if expected is None else expected
    assert pred is not None
    f64 = _f64
    sums = _report_cell_sums(
        p.cells[bank], p.n_cells[bank], f64(yv), f64(pred), f64(p.w), f64(p.e),
        None if variance is None else f64(variance),
    )
    out: list[dict[str, Any]] = []
    for j, (name, (act, exp, mass, rows, var)) in enumerate(zip(p.names, sums)):
        with np.errstate(divide="ignore", invalid="ignore"):
            ae = np.where(exp != 0, act / exp, np.nan)
        entry: dict[str, Any] = {
            "feature": name, "raw": j, "actual": act.tolist(), "expected": exp.tolist(),
            "mass": mass.tolist(), "rows": rows.astype(np.int64).tolist(), "ae": ae.tolist(),
        }
        if variance is not None:
            sd = np.sqrt(var)
            entry["z"] = [float(d / v) if v > 0 else None for d, v in zip(act - exp, sd)]
            entry["_var"] = var.tolist()  # for the bands; removed before the report is returned
        out.append(entry)
    return out


def table_measure(est: "_BaseTBoost", X: Any, sample_weight: Any | None, exposure: Any | None,
                  ref_measure: str | None) -> dict[str, Any]:
    """The measure arguments ``est.tables(X, ref_measure=..., sample_weight=..., exposure=...)``
    hands the native export (resolved the same way, fit-time fallbacks included), so the
    Shapley values can be read from the bank the report's tables come from."""
    model, _ = _bank_models(est)
    X, sample_weight, exposure = est._resolve_explain_inputs(X, sample_weight, exposure)
    weight32, exposure32 = est._explain_mass(sample_weight, exposure, model,
                                             joint=ref_measure == "joint")
    return {**est._export_measure_kwargs(ref_measure, 1.0, None),
            "weight": weight32, "exposure": exposure32}


def interaction_ae(report: dict[str, Any], prepared: list[Prepared], obj: str,
                   rho: float) -> None:
    """For each interaction table the page draws, the A/E by band of its first feature (the
    factor check's bands) within each band of its partner (the chart's bands of equal exposure),
    per dataset (``interactions[k]["ae"]``), with each pair's mass, z and the standard deviation
    ``sd`` of its A/E; and, by level of the first feature within each partner band, the actual,
    expected and mass (``levels``) behind the actual and expected rates. One native pass per
    dataset sums every table. A multiclass model's first class, one-vs-rest, as the page draws."""
    from ._html_report import _INTERACTION_ROWS, _classes, _table_cells, interaction_layout

    cls = _classes(report)[0]
    plans = []
    for inter in (report.get("interactions") or [])[:_INTERACTION_ROWS]:
        lay = interaction_layout(report, inter, cls)
        if lay is None or not lay["bands"]:
            continue
        xe, be = lay["entries"][lay["xf"]], lay["entries"][lay["by"]]
        x_bands = [[int(c) for c in b["cells"]] for b in xe.get("bands") or []]
        if not x_bands:
            continue
        x_band = np.full(len(xe["mass"]), -1, dtype=np.int64)
        for i, cells in enumerate(x_bands):
            x_band[cells] = i
        band_of = {tc: i for i, band in enumerate(lay["bands"]) for tc in band}
        by_band = np.array([-1 if tc is None else band_of.get(tc, -1)
                            for tc in _table_cells(be["axis"], lay["t"]["axes"][lay["iby"]])],
                           dtype=np.int64)
        plans.append((inter, int(xe["raw"]), int(be["raw"]), x_band, by_band,
                      [str(b["label"]) for b in xe["bands"]], lay))
    if not plans:
        return
    per_ds: list[list[dict[str, Any]]] = [[] for _ in plans]
    for p, d in zip(prepared, report["datasets"]):
        if p.error is not None or not d.get("actual_vs_expected"):
            continue
        row_cells = p.cells[0]
        # Two columns per table: (x band, partner band), then (x level, partner band).
        codes = np.empty((row_cells.shape[0], 2 * len(plans)), dtype=np.uint32)
        n_cells = []
        for j, (_, rx, rb, x_band, by_band, x_labels, lay) in enumerate(plans):
            nb, size = len(lay["bands"]), len(x_labels) * len(lay["bands"])
            cx = np.minimum(row_cells[:, rx], len(x_band) - 1)
            pb = by_band[np.minimum(row_cells[:, rb], len(by_band) - 1)]
            xb = x_band[cx]
            codes[:, 2 * j] = np.where((xb >= 0) & (pb >= 0), xb * nb + pb, size)  # last: neither
            levels = len(x_band) * nb
            codes[:, 2 * j + 1] = np.where(pb >= 0, cx * nb + pb, levels)
            n_cells += [size + 1, levels + 1]
        if p.proba is not None and p.codes is not None:
            pk = p.proba[:, 0]
            actual, expected, var = (p.codes == 0).astype(np.float64), pk, pk * (1.0 - pk)
        else:
            assert p.pred is not None
            actual, expected = p.y, p.pred
            var = dispersion(obj, p.y, p.pred, p.w, rho) * _variance(obj, p.pred, rho)
        sums = _report_cell_sums(np.ascontiguousarray(codes), n_cells, _f64(actual),
                                 _f64(expected), _f64(p.w), _f64(p.e), _f64(var))
        for j, (_, _, _, x_band, _, x_labels, lay) in enumerate(plans):
            (a, e, m, _, v), (la, le, lm, _, _) = sums[2 * j], sums[2 * j + 1]
            nb = len(lay["bands"])

            def grid(values: list[float | None]) -> list[list[float | None]]:
                return [values[i * nb:(i + 1) * nb] for i in range(len(x_labels))]

            size = len(x_labels) * nb
            per_ds[j].append({
                "label": d["label"],
                "ae": grid([_ratio(a[c], e[c]) if e[c] > 0 else None for c in range(size)]),
                "mass": grid([float(m[c]) for c in range(size)]),
                "z": grid([float((a[c] - e[c]) / math.sqrt(v[c])) if v[c] > 0 else None
                           for c in range(size)]),
                "sd": grid([float(math.sqrt(v[c]) / e[c]) if v[c] > 0 and e[c] > 0 else None
                            for c in range(size)]),
                "levels": {key: [[float(arr[c * nb + b]) for b in range(nb)]
                                 for c in range(len(x_band))]
                           for key, arr in (("actual", la), ("expected", le), ("mass", lm))},
            })
    for (inter, _, _, _, _, x_labels, lay), datasets in zip(plans, per_ds):
        inter["ae"] = {"x": lay["xf"], "by": lay["by"], "x_labels": x_labels,
                       "x_bands": [[int(c) for c in b["cells"]]
                                   for b in lay["entries"][lay["xf"]]["bands"]],
                       "by_bands": lay["bands"], "by_labels": lay["band_labels"],
                       "datasets": datasets}


def shap_by_cell(est: "_BaseTBoost", p: Prepared,
                 measure: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Per class bank and raw feature: each merged-grid cell's exposure-weighted mean and 10th
    and 90th percentile of the feature's exact Shapley value (every table shared equally among
    its features, on the link scale) over the rows in that cell. The spread is what the other
    factors, through the interactions, do to the feature's effect across these policies.
    ``measure`` (`table_measure`) reads them from the bank the report's tables are exported
    from, so with no interactions the mean is the one-way table."""
    model, multiclass = _bank_models(est)
    banks = model.feature_contributions(p.x32, **p.cat_kw, n_jobs=est._resolve_n_jobs(),
                                        **measure)
    weight = _f64(p.w * p.e)
    out = []
    for k, (_, values) in enumerate(banks if multiclass else [banks]):
        stats = _report_cell_quantiles(p.cells[k], p.n_cells[k], _f64(values), weight,
                                       list(_SHAP_QUANTILES))
        out.append([
            {"mean": mean.tolist(), "p10": q[:, 0].tolist(), "p90": q[:, 1].tolist()}
            for _, mean, q in stats
        ])
    return out


# --------------------------------------------------------------------------- statistics -------

def _variance(obj: str, mu: np.ndarray, rho: float) -> np.ndarray:
    """The objective's variance function V(mu), per row."""
    if obj == "poisson":
        v = np.maximum(mu, 0.0)
    elif obj == "gamma":
        v = mu * mu
    elif obj == "tweedie":
        v = np.power(np.maximum(mu, 0.0), rho)
    elif obj == "logistic":
        v = np.clip(mu, 0.0, 1.0) * (1.0 - np.clip(mu, 0.0, 1.0))
    else:
        v = np.ones_like(mu)
    return np.asarray(v, dtype=np.float64)


def dispersion(obj: str, y: np.ndarray, mu: np.ndarray, w: np.ndarray, rho: float) -> float:
    """Pearson dispersion: 1 for a binomial outcome, else Σ w (y - mu)² / V(mu) / Σ w."""
    if obj == "logistic":
        return 1.0
    v = _variance(obj, mu, rho)
    ok = (v > 0) & (w > 0)
    if not ok.any():
        return 1.0
    return float(np.sum(w[ok] * (y[ok] - mu[ok]) ** 2 / v[ok]) / np.sum(w[ok]))



def _null_prediction(obj: str, link: str, p: Prepared, rho: float) -> np.ndarray | None:
    """The best constant model's expected value per row, with the same exposure and offset."""
    y, w, e = p.y, p.w, p.e
    o = p.offset if p.offset is not None else np.zeros_like(y)
    if link == "log":
        a = e * np.exp(o)
        power = {"poisson": 1.0, "gamma": 2.0}.get(obj, rho)
        den = float(np.sum(w * a ** (2.0 - power)))
        if den <= 0:
            return None
        c = float(np.sum(w * y * a ** (1.0 - power))) / den
        return np.asarray(c * a, dtype=np.float64) if c > 0 else None
    if link == "identity":
        if w.sum() <= 0:
            return None
        return np.asarray(np.full_like(y, float(np.sum(w * (y - o)) / np.sum(w))) + o)
    # logit: the intercept c solving Σ w (y - σ(c + base)) = 0, with the offset and
    # log-exposure as fixed terms. The sum falls strictly in c, so the root is bracketed and
    # Newton steps that leave the bracket (a large offset saturates σ) bisect instead.
    base = o + np.log(np.where(e > 0, e, 1.0))
    mean = float(np.sum(w * y) / np.sum(w)) if w.sum() > 0 else 0.0
    if not 0.0 < mean < 1.0:
        return None

    def sigmoid(c: float) -> np.ndarray:
        return np.asarray(0.5 * (1.0 + np.tanh(0.5 * (c + base))))

    def grad(c: float) -> float:
        return float(np.sum(w * (y - sigmoid(c))))

    c = math.log(mean / (1.0 - mean)) - float(np.sum(w * base) / np.sum(w))
    lo, hi, step = c, c, 1.0
    while grad(lo) <= 0.0 and step < 1e6:
        lo, step = lo - step, step * 2.0
    step = 1.0
    while grad(hi) >= 0.0 and step < 1e6:
        hi, step = hi + step, step * 2.0
    for _ in range(100):
        mu = sigmoid(c)
        g = float(np.sum(w * (y - mu)))
        if g == 0.0:
            break
        if g > 0.0:
            lo = c
        else:
            hi = c
        hess = float(np.sum(w * mu * (1.0 - mu)))
        nxt = c + g / hess if hess > 0.0 else math.nan
        if not lo < nxt < hi:
            nxt = 0.5 * (lo + hi)
        tol = 1e-12 * max(1.0, abs(c))
        done = abs(nxt - c) <= tol or hi - lo <= tol
        c = nxt
        if done:
            break
    return sigmoid(c)


def _deviance(obj: str, y: np.ndarray, mu: np.ndarray, w: np.ndarray, rho: float) -> float:
    if obj == "logistic":
        return metrics.mean_binomial_deviance(y, mu, w)
    power = {"squared_error": 0.0, "poisson": 1.0, "gamma": 2.0}.get(obj, rho)
    return metrics.mean_tweedie_deviance(y, mu, w, power=power)


def _guarded(fn: Any, *args: Any) -> tuple[float | None, str | None]:
    try:
        return _finite(fn(*args)), None
    except (ValueError, ZeroDivisionError, FloatingPointError) as exc:
        return None, str(exc)


def _rate_and_mass(link: str, p: Prepared, pred: np.ndarray, y: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(predicted rate, actual rate, row mass, keep mask) for ranking: per unit exposure (and
    offset multiplier) for a log link, per row otherwise. Rows without mass are dropped."""
    if link == "log":
        a = p.e * (np.exp(p.offset) if p.offset is not None else 1.0)
        keep = a > 0
        safe = np.where(keep, a, 1.0)
        return pred / safe, y / safe, p.w * a, keep & (p.w > 0)
    return pred, y, p.w.copy(), p.w > 0


def _ranking(link: str, p: Prepared, pred: np.ndarray, y: np.ndarray,
             variance: np.ndarray | None = None) -> dict[str, Any]:
    """Gini, lift and Lorenz from one native sort (see ``t_boost_core::report::ranking``).
    ``variance`` is each row's variance of its weighted actual total (w · dispersion · V(mu)),
    summed per decile for the decile's A/E standard error and z."""
    rate, y_rate, mass, keep = _rate_and_mass(link, p, pred, y)
    f64 = _f64
    r = _report_ranking(f64(rate), f64(y_rate), f64(np.where(keep, mass, 0.0)),
                        _LIFT_BANDS, _LORENZ_POINTS,
                        None if variance is None else f64(np.where(keep, variance, 0.0)))
    out: dict[str, Any] = {"excluded_rows": int(p.rows - r["rows"])}
    if link == "identity" and np.any(y_rate[keep] < 0):
        out.update(gini=None, gini_normalised=None,
                   gini_reason="Gini needs non-negative targets")
    else:
        out["gini"] = _finite(r["gini"])
        perfect = r["gini_perfect"]
        out["gini_normalised"] = _finite(r["gini"] / perfect) if abs(perfect) > 1e-12 else None
    total = sum(b[1] for b in r["lift"])
    out["lift"] = [
        {"band": i + 1, "rows": int(rows), "mass": m, "mass_share": m / total if total else None,
         "actual": _ratio(a, m), "predicted": _ratio(pr, m), "rate_min": lo, "rate_max": hi,
         "ae": _ratio(a, pr) if pr > 0 else None,
         "sd": (math.sqrt(var) / pr) if variance is not None and var > 0 and pr > 0 else None,
         "z": ((a - pr) / math.sqrt(var)) if variance is not None and var > 0 else None}
        for i, (rows, m, a, pr, lo, hi, var) in enumerate(r["lift"])
    ]
    out["lorenz"] = ({"x": np.linspace(0.0, 1.0, _LORENZ_POINTS).tolist(), "y": r["lorenz"]}
                     if r["lorenz"] else None)
    kept = rate[keep]
    out["predicted_rate_percentiles"] = (
        {str(q): float(v) for q, v in zip(_PERCENTILES, np.percentile(kept, _PERCENTILES))}
        if kept.size else None
    )
    return out


def prediction_histograms(est: "_BaseTBoost", prepared: list[Prepared],
                          datasets: list[dict[str, Any]]) -> None:
    """Each scored single-output dataset's mass by predicted rate, on bins every dataset shares
    (``metrics["predicted_rate_histogram"]``): from the first dataset's 0.5th to 99.5th
    percentile, log-spaced when the rates are positive and span more than a factor of 3, with
    rates beyond in the end bins."""
    parts = []
    for p, d in zip(prepared, datasets):
        if p.error is None and p.pred is not None and d.get("metrics"):
            rate, _, mass, keep = _rate_and_mass(est.link, p, p.pred, p.y)
            parts.append((d, rate[keep], mass[keep]))
    if not parts or parts[0][0] is not datasets[0] or not parts[0][1].size:
        return
    first = parts[0][1]
    lo, hi = (float(v) for v in np.quantile(first, [0.005, 0.995]))
    if hi <= lo:
        lo, hi = float(np.min(first)), float(np.max(first))
    if hi <= lo:  # constant predictions: one narrow range
        hi = lo + (abs(lo) * 0.1 or 1.0)
    log = lo > 0 and hi / lo > 3
    edges = (np.geomspace(lo, hi, _HISTOGRAM_BINS + 1) if log
             else np.linspace(lo, hi, _HISTOGRAM_BINS + 1))
    for d, rate, mass in parts:
        counts, _ = np.histogram(np.clip(rate, edges[0], edges[-1]), bins=edges, weights=mass)
        d["metrics"]["predicted_rate_histogram"] = {
            "edges": edges.tolist(), "mass": counts.tolist(), "log": bool(log)}


def single_output_metrics(est: "_BaseTBoost", p: Prepared) -> dict[str, Any]:
    from .sklearn import _canonical_objective

    obj = _canonical_objective(est.objective)
    link = est.link
    rho = float(getattr(est, "tweedie_rho", 1.5) or 1.5)
    assert p.pred is not None
    out: dict[str, Any] = {
        "rows": p.rows, "weight": float(p.w.sum()), "mass": float(np.sum(p.w * p.e)),
        "actual": float(np.sum(p.w * p.y)), "expected": float(np.sum(p.w * p.pred)),
    }
    out["ae"] = _ratio(out["actual"], out["expected"]) if out["expected"] > 0 else None
    out["mean_difference"] = _ratio(out["actual"] - out["expected"], out["weight"])
    out["deviance"], out["deviance_reason"] = _guarded(_deviance, obj, p.y, p.pred, p.w, rho)
    null = _null_prediction(obj, link, p, rho)
    if null is None:
        out["null_deviance"], out["null_deviance_reason"] = None, "no constant model fits"
    else:
        out["null_deviance"], out["null_deviance_reason"] = _guarded(
            _deviance, obj, p.y, null, p.w, rho
        )
    d, d0 = out["deviance"], out["null_deviance"]
    out["deviance_explained"] = (1.0 - d / d0) if d is not None and d0 else None
    var = p.w * dispersion(obj, p.y, p.pred, p.w, rho) * _variance(obj, p.pred, rho)
    out.update(_ranking(link, p, p.pred, p.y, var))
    return out


def multiclass_metrics(est: "_BaseTBoost", p: Prepared) -> dict[str, Any]:
    assert p.proba is not None and p.codes is not None
    proba, codes, w = p.proba, p.codes, p.w
    k = proba.shape[1]
    labels = [str(c) for c in est.classes_]
    out: dict[str, Any] = {"rows": p.rows, "weight": float(w.sum()), "mass": float(w.sum())}
    out["deviance"], out["deviance_reason"] = _guarded(
        metrics.mean_multinomial_deviance, codes, proba, w
    )
    freq = np.bincount(codes, weights=w, minlength=k) / max(float(w.sum()), 1e-300)
    out["null_deviance"], out["null_deviance_reason"] = _guarded(
        metrics.mean_multinomial_deviance, codes, np.tile(freq, (len(codes), 1)), w
    )
    d, d0 = out["deviance"], out["null_deviance"]
    out["deviance_explained"] = (1.0 - d / d0) if d is not None and d0 else None
    picked = np.argmax(proba, axis=1)
    out["accuracy"] = _ratio(float(np.sum(w * (picked == codes))), float(w.sum()))
    expected = np.zeros((k, k))
    confusion = np.zeros((k, k))
    for c in range(k):
        rows = codes == c
        expected[c] = np.sum(w[rows, None] * proba[rows], axis=0)
        confusion[c] = np.bincount(picked[rows], weights=w[rows], minlength=k)
    out["class_matrix"] = {
        "classes": labels, "rows": "actual class", "columns": "predicted class",
        "expected": expected.tolist(), "confusion": confusion.tolist(),
    }
    per_class = []
    for c in range(k):
        yk = (codes == c).astype(np.float64)
        act, exp = float(np.sum(w * yk)), float(np.sum(w * proba[:, c]))
        entry: dict[str, Any] = {"class": labels[c], "actual": act, "expected": exp,
                                 "ae": _ratio(act, exp)}
        entry.update(_ranking("logit", p, proba[:, c], yk, w * proba[:, c] * (1.0 - proba[:, c])))
        per_class.append(entry)
    out["per_class"] = per_class
    return out


# --------------------------------------------------------------------------- drift ------------

def _psi_bins(mass: np.ndarray, levels: list[dict[str, Any]] | None,
              n_bins: int = _PSI_BINS) -> list[list[int]]:
    """Groups of merged cells for PSI and the factor bands: the missing cell alone, then the rest
    coarsened to at most ``n_bins - 1`` bins — contiguous runs of equal first-dataset mass on a
    numeric axis, the largest levels plus one "other" bin on a categorical axis."""
    cells = list(range(1, len(mass)))
    room = n_bins - 1
    if len(cells) <= room:
        groups = [[c] for c in cells]
    elif levels is not None:
        order = sorted(cells, key=lambda c: (-mass[c], c))
        groups = [[c] for c in order[: room - 1]] + [sorted(order[room - 1:])]
    else:
        total = float(mass[1:].sum())
        groups, current, acc = [], [], 0.0
        for c in cells:
            current.append(c)
            acc += float(mass[c])
            if total > 0 and acc >= total * (len(groups) + 1) / room and len(groups) < room - 1:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
    return [[0]] + groups


def whole_valued(p: Prepared) -> list[bool]:
    """Per numeric column of the scored design (the raw feature ids of the numeric axes):
    whether every value present is a whole number."""
    x = p.x32
    if x is None:
        return []
    out = []
    for j in range(x.shape[1]):
        col = x[:, j]
        col = col[np.isfinite(col)]
        out.append(bool(col.size) and bool(np.all(col == np.round(col))))
    return out


def mark_integer_axes(axes: list[dict[str, Any]], whole: list[bool]) -> None:
    """Set ``integer`` on each numeric axis: its feature took only whole values in the first
    dataset, so its cells read as integers (``51``, ``22–25``). Borders alone cannot tell: 0.1
    and 0.9 split at 0.5 too."""
    for a in axes:
        raw = a.get("raw")
        if not a.get("levels") and raw is not None:
            a["integer"] = 0 <= int(raw) < len(whole) and whole[int(raw)]


def band_label(cells: list[int], axis: dict[str, Any]) -> str:
    """A band's label: ``missing``; the values of a run of numeric cells (``51``, ``22–25``,
    ``≤ 18`` on an integer feature, ``(a, b]`` on a continuous one); the level names (or
    ``n other levels``) for categorical cells."""
    if cells == [0]:
        return "missing"
    levels = axis.get("levels")
    if levels:
        names = [str(lv["label"]) for lv in levels if int(lv["cell"]) in cells]
        return ", ".join(names) if len(names) <= 3 else f"{len(names)} other levels"
    return run_label(axis, cells[0], cells[-1])


def factor_bands(entry: dict[str, Any], bins: list[list[int]], axis: dict[str, Any]
                 ) -> list[dict[str, Any]]:
    """The entry's cells summed over ``bins`` (from the first dataset), with A/E and z."""
    out = []
    var = entry.get("_var")
    for cells in bins:
        act = sum(entry["actual"][c] for c in cells)
        exp = sum(entry["expected"][c] for c in cells)
        v = sum(var[c] for c in cells) if var is not None else 0.0
        mass = sum(entry["mass"][c] for c in cells)
        rows = sum(entry["rows"][c] for c in cells)
        if rows == 0 and cells == [0]:
            continue
        out.append({"label": band_label(cells, axis), "cells": cells, "rows": int(rows),
                    "mass": mass, "actual": act, "expected": exp,
                    "ae": _ratio(act, exp) if exp > 0 else None,
                    "sd": float(math.sqrt(v) / exp) if v > 0 and exp > 0 else None,
                    "z": float((act - exp) / math.sqrt(v)) if v > 0 else None})
    return out


def psi(first: list[dict[str, Any]], second: list[dict[str, Any]],
        axes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for a, b, axis in zip(first, second, axes):
        ma, mb = np.asarray(a["mass"], dtype=float), np.asarray(b["mass"], dtype=float)
        if ma.sum() <= 0 or mb.sum() <= 0:
            out.append({"feature": a["feature"], "psi": None, "bins": []})
            continue
        bins = _psi_bins(ma, axis.get("levels"))
        pa = np.array([ma[g].sum() for g in bins]) / ma.sum()
        pb = np.array([mb[g].sum() for g in bins]) / mb.sum()
        fa, fb = np.maximum(pa, _PSI_FLOOR), np.maximum(pb, _PSI_FLOOR)
        terms = (fb - fa) * np.log(fb / fa)
        out.append({
            "feature": a["feature"], "psi": float(terms.sum()),
            "bins": [{"cells": g, "first": float(x), "second": float(y)}
                     for g, x, y in zip(bins, pa, pb)],
        })
    return out


# --------------------------------------------------------------------------- review -----------

def _flag(check: str, status: str, message: str, *, value: Any = None,
          threshold: Any = None, dataset: str | None = None, feature: str | None = None,
          cls: str | None = None) -> dict[str, Any]:
    return {"check": check, "status": status, "message": message, "value": value,
            "threshold": threshold, "dataset": dataset, "feature": feature, "class": cls}


def review(est: "_BaseTBoost", report: dict[str, Any], th: dict[str, float | None]
           ) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    datasets = report["datasets"]
    multiclass = report["model"]["multiclass"]
    roles = dataset_roles(len(datasets))
    for i, ds in enumerate(datasets):
        label = ds["label"]
        tol = th[f"ae_{roles[i]}"]
        if ds.get("error"):
            flags.append(_flag("scoring", "critical", ds["error"], dataset=label))
            continue
        m = ds["metrics"]
        scopes = ([(c["class"], c["ae"]) for c in m["per_class"]] if multiclass
                  else [(None, m["ae"])])
        if tol is not None:
            for cls, ae in scopes:
                if ae is None:
                    flags.append(_flag("overall_ae", "info", "A/E undefined (no expected value)",
                                       dataset=label, cls=cls, threshold=tol))
                    continue
                off = abs(ae - 1.0) > tol
                flags.append(_flag(
                    "overall_ae", "warning" if off else "pass",
                    f"overall A/E {ae:.4f} is {'outside' if off else 'within'} 1 ± {tol:g}",
                    value=ae, threshold=tol, dataset=label, cls=cls))
        ftol, fmin, fz = th["factor_ae"], th["factor_min_mass"], th["factor_z"]
        nz = th["near_miss_z"]
        # Near misses are listed for out-of-sample data (or the only dataset there is).
        out_of_sample = i > 0 or len(datasets) == 1
        dtol = th["decile_ae"]
        if dtol is not None:
            lifts = ([(c["class"], c.get("lift") or []) for c in m["per_class"]] if multiclass
                     else [(None, m.get("lift") or [])])
            decile_flagged = len(flags)
            for cls, lift in lifts:
                for b in lift:
                    ae_b, z_b = b.get("ae"), b.get("z")
                    if ae_b is None or abs(ae_b - 1.0) <= dtol:
                        continue
                    text = f"decile {b['band']} A/E {ae_b:.3f}" + (
                        f" (z {z_b:+.1f})" if z_b is not None else "")
                    # As for the factor bands: with `factor_z` off, the tolerance alone flags.
                    if fz is None or (z_b is not None and abs(z_b) >= fz):
                        flags.append(_flag("decile_ae", "warning", text, value=ae_b,
                                           threshold=dtol, dataset=label, cls=cls))
                    elif (nz is not None and z_b is not None and abs(z_b) >= nz
                          and out_of_sample):
                        flags.append(_flag("decile_ae", "info", "near miss: " + text,
                                           value=ae_b, threshold=dtol, dataset=label, cls=cls))
            if not any(f["status"] == "warning" for f in flags[decile_flagged:]):
                flags.append(_flag("decile_ae", "pass",
                                   f"no predicted-rate decile off by more than {dtol:g}"
                                   + (f" and {fz:g} standard deviations" if fz is not None else ""),
                                   threshold=dtol, dataset=label))
        for factor in ds["actual_vs_expected"]:  # a fresh verdict on every band
            for b in factor.get("bands") or []:
                b.pop("flag", None)
        if ftol is not None:
            flagged = len(flags)
            covered, total_mass = 0.0, 0.0
            for factor in ds["actual_vs_expected"]:
                bands = factor.get("bands") or []
                total = sum(b["mass"] for b in bands)
                eligible = [b for b in bands
                            if total > 0 and b["mass"] / total >= (fmin or 0.0)]
                covered += sum(b["mass"] for b in eligible)
                total_mass += total
                bad = [b for b in eligible if b["ae"] is not None and abs(b["ae"] - 1.0) > ftol
                       and (fz is None or (b["z"] is not None and abs(b["z"]) >= fz))]
                near = [b for b in eligible if b not in bad and nz is not None and out_of_sample
                        and b["ae"] is not None and abs(b["ae"] - 1.0) > ftol
                        and b["z"] is not None and abs(b["z"]) >= nz]
                # The verdict on the band itself, for the page's charts.
                for b in bad:
                    b["flag"] = "warning"
                for b in near:
                    b["flag"] = "near_miss"
                for b in near:
                    flags.append(_flag(
                        "factor_ae", "info",
                        f"near miss: {b['label']} A/E {b['ae']:.3f}, z {b['z']:+.1f} (flag at "
                        f"|z| ≥ {fz:g})" if fz is not None else
                        f"near miss: {b['label']} A/E {b['ae']:.3f}, z {b['z']:+.1f}",
                        value=b["ae"], threshold=ftol, dataset=label,
                        feature=factor["feature"], cls=factor.get("class")))
                if bad:
                    worst = max(bad, key=lambda b: abs(b["ae"] - 1.0))
                    flags.append(_flag(
                        "factor_ae", "warning",
                        f"{len(bad)} band(s) off by more than {ftol:g}"
                        + (f" and at least {fz:g} standard deviations" if fz is not None else "")
                        + f" (worst: {worst['label']}, A/E {worst['ae']:.3f}"
                        + (f", z {worst['z']:.1f})" if worst["z"] is not None else ")"),
                        value=len(bad), threshold=ftol, dataset=label,
                        feature=factor["feature"], cls=factor.get("class")))
            if not any(f["status"] == "warning" for f in flags[flagged:]):
                coverage = covered / total_mass if total_mass else 0.0
                flags.append(_flag(
                    "factor_ae", "pass",
                    f"no factor band off by more than {ftol:g}"
                    + (f" and {fz:g} standard deviations" if fz is not None else "")
                    + f"; the bands checked hold {coverage:.1%} of the mass",
                    threshold=ftol, dataset=label))
        if ds.get("unseen") and th["unseen_mass"] is not None:
            total = ds["metrics"]["mass"] or 0.0
            for feature in sorted({u["feature"] for u in ds["unseen"]}):
                rows = [u for u in ds["unseen"] if u["feature"] == feature]
                unseen_mass = sum(float(u.get("mass", u["rows"])) for u in rows)
                share = unseen_mass / total if total else None
                over = share is not None and share > th["unseen_mass"]
                flags.append(_flag(
                    "unseen_values", "warning" if over else "info",
                    f"{len(rows)} unseen value(s), {share:.2%} of the mass" if share is not None
                    else f"{len(rows)} unseen value(s)",
                    value=share, threshold=th["unseen_mass"], dataset=label, feature=feature))
    if len(datasets) == 1:
        flags.append(_flag("holdout_data", "info",
                           "train data only: every metric is in-sample. Pass test data as "
                           "eval_set for out-of-sample evidence",
                           dataset=datasets[0]["label"]))
    # Train and test need no note: the bags early-stop and prune on rows they hold back. A fit
    # that early-stopped on an eval_set gets an info note, not a warning: the report cannot tell
    # whether that eval_set is one of these datasets, and stopping on (say) a later period while
    # reporting on a separate test set is a sound workflow that a warning would flag.
    if len(datasets) == 2 and report["fit"]["early_stopping"]["eval_set"]:
        flags.append(_flag(
            "holdout_data", "info",
            f"the fit early-stopped on an eval_set: if {datasets[1]['label']} is that data, its "
            "metrics are not fully out-of-sample",
            dataset=datasets[1]["label"]))
    elif len(datasets) == 3 and report["fit"]["early_stopping"]["eval_set"]:
        flags.append(_flag(
            "holdout_data", "info",
            f"the fit early-stopped on an eval_set: if that was {datasets[1]['label']}, its "
            f"metrics are not fully out-of-sample; {datasets[2]['label']} is the independent "
            "estimate",
            dataset=datasets[1]["label"]))
    train = datasets[0]
    first = None if train.get("error") else train["metrics"]["deviance_explained"]
    for ds in datasets[1:]:
        if ds.get("error"):
            continue
        later = ds["metrics"]["deviance_explained"]
        if later is not None:
            flags.append(_flag(
                "beats_null", "critical" if later <= 0 else "pass",
                f"deviance explained on {ds['label']}: {later:.4f}",
                value=later, threshold=0.0, dataset=ds["label"]))
        gap = th["train_gap"]
        if gap is not None and first is not None and later is not None:
            drop = first - later
            flags.append(_flag(
                "train_gap", "warning" if drop > gap else "pass",
                f"deviance explained drops by {drop:.4f} from {train['label']} to "
                f"{ds['label']}", value=drop, threshold=gap, dataset=ds["label"]))
    reasons = report["fit"]["stopping_reason_per_bag"]["value"]
    if reasons is None:
        flags.append(_flag("early_stopping", "info",
                           report["fit"]["stopping_reason_per_bag"]["reason"]))
    else:
        capped = sum(r == "max_trees" for r in reasons)
        flags.append(_flag(
            "early_stopping", "warning" if capped else "pass",
            f"{capped} of {len(reasons)} bag(s) reached n_trees without early stopping; the "
            "model may be under-fitted" if capped else
            f"every bag stopped before n_trees ({', '.join(sorted(set(reasons)))})",
            value=capped))
    crit, warn = th["psi_critical"], th["psi_warn"]
    for pair in report.get("drift") or []:
        if warn is None and crit is None:
            break
        for d in pair["features"]:
            if d["psi"] is None:
                continue
            status = ("critical" if crit is not None and d["psi"] > crit
                      else "warning" if warn is not None and d["psi"] > warn else "pass")
            flags.append(_flag(
                "psi", status, f"PSI {d['psi']:.4f} from {pair['reference']} to {pair['compared']}",
                value=d["psi"], threshold=warn if status != "critical" else crit,
                dataset=pair["compared"], feature=d["feature"]))
    # The deployed tables themselves: missing-value levels and thin cells.
    meff = th["missing_effect"]
    for f in report.get("features") or []:
        ow = f.get("one_way")
        if not ow:
            continue
        miss = ow.get("missing")
        if meff is not None and miss and miss["support"] == 0:
            if ow["multiplicative"]:
                off = abs(miss["effect"] - 1.0) > meff
                shown = f"×{miss['effect']:.3g}"
            else:
                rng = ow["supported_range"] or ow["range"] or [0.0, 0.0]
                off = abs(miss["effect"]) > meff * max(abs(rng[1] - rng[0]), 1e-12)
                shown = f"{miss['effect']:+.3g}"
            if off:
                flags.append(_flag(
                    "missing_level", "warning",
                    f"a missing {f['feature']} would score {shown}, with no exposure behind it",
                    value=miss["effect"], threshold=meff, feature=f["feature"]))
        if ow["thin_cells"] and ow["thin_range"] and ow["supported_range"]:
            lo, hi = ow["thin_range"]
            slo, shi = ow["supported_range"]
            if lo < slo or hi > shi:
                def fmt(v: float, mult: bool = ow["multiplicative"]) -> str:
                    return f"×{v:.3g}" if mult else f"{v:+.3g}"

                flags.append(_flag(
                    "thin_cells", "info",
                    f"{ow['thin_cells']} thin cell(s) ({ow['thin_share']:.2%} of exposure) span "
                    f"{fmt(lo)}–{fmt(hi)}, beyond the {fmt(slo)}–{fmt(shi)} of the rest",
                    value=ow["thin_cells"], threshold=th["thin_cell_mass"],
                    feature=f["feature"]))
    grad = report["fit"]["graduation"]["value"] or {}
    grad_tables = report["fit"]["graduation_tables"]["value"] or []
    # A declined fixed `graduation_alpha` is a binding finding (below); a declined GCV choice
    # is reported here.
    if (grad.get("candidate_count") == 1 and grad.get("adopted") is False
            and getattr(est, "graduation_alpha", None) is None):
        flags.append(_flag(
            "graduation", "info",
            "graduation was not applied: " + str(grad.get("rejected") or "it was declined")))
    if grad and grad.get("adopted") and grad.get("evidence") == "none":
        smoothed = [g for g in grad_tables if (g.get("alpha") or 0) > 0 and not g.get("rejected")]
        plain = [g["features"][0] for g in grad_tables
                 if len(g["features"]) == 1 and not (g.get("alpha") or 0) > 0]
        flags.append(_flag(
            "graduation", "info",
            f"graduation smoothed {len(smoothed) if grad_tables else grad.get('n_smoothed')} "
            "table(s) without holdout evidence"
            + (f"; left as fitted: {', '.join(plain)}" if plain else "")))

    binding = report["fit"]["binding_report"]["value"] or []
    for row in binding:
        binding_status = {"INERT": "warning", "OVERRIDDEN": "warning",
                          "UNAVAILABLE": "info"}.get(row["status"])
        if binding_status:
            flags.append(_flag("binding", binding_status,
                               f"{row['param']}={row['value']!r} {row['status']}: {row['reason']}",
                               value=row["status"]))
    return flags


# --------------------------------------------------------------------------- summary ----------

_KEY_CHARS = re.compile(r"[^A-Za-z0-9_]")


def summary_metrics(report: dict[str, Any]) -> dict[str, float]:
    """Flat, finite, MLflow-safe metrics: ``<label>_<metric>`` and, for multiclass,
    ``<label>_<metric>_<class>``."""
    out: dict[str, float] = {}

    def put(key: str, value: Any) -> None:
        if value is None or isinstance(value, bool):
            return
        v = float(value)
        if not math.isfinite(v):
            return
        key = _KEY_CHARS.sub("_", key)[:_MAX_KEY]
        base, i = key, 2
        while key in out:
            suffix = f"_{i}"
            key = base[: _MAX_KEY - len(suffix)] + suffix
            i += 1
        out[key] = v

    for ds in report["datasets"]:
        if ds.get("error"):
            continue
        m, label = ds["metrics"], ds["label"]
        for name in ("deviance", "null_deviance", "deviance_explained", "gini",
                     "gini_normalised", "ae", "accuracy", "rows", "mass"):
            put(f"{label}_{name}", m.get(name))
        for c in m.get("per_class") or []:
            for name in ("ae", "gini"):
                put(f"{label}_{name}_{c['class']}", c.get(name))
    fit = report["fit"]
    put("n_trees", fit["n_trees"]["value"])
    put("deployed_tables", report["structure"]["deployed_tables"])
    flags = report["review"]
    put("review_warnings", sum(f["status"] == "warning" for f in flags))
    put("review_critical", sum(f["status"] == "critical" for f in flags))
    return out


# --------------------------------------------------------------------------- model sections ---

def _display(value: Any) -> Any:
    try:
        return json_safe(value)
    except Exception:  # pragma: no cover - json_safe already falls back to str
        return str(value)


def model_section(est: "_BaseTBoost", multiclass: bool, names: list[str]) -> dict[str, Any]:
    from . import __version__
    from .sklearn import _canonical_objective, _ctor_defaults

    defaults = _ctor_defaults(type(est))
    params = est.get_params(deep=False)
    columns = getattr(est, "feature_names_in_", None)
    classes = getattr(est, "classes_", None)
    return {
        "class": type(est).__name__,
        "objective": _canonical_objective(est.objective),
        "tweedie_rho": est.tweedie_rho if _canonical_objective(est.objective) == "tweedie" else None,
        "link": est.link,
        "multiclass": multiclass,
        "classes": None if classes is None else [str(c) for c in classes],
        "t_boost_version": __version__,
        "columns": ([str(c) for c in columns] if columns is not None else
                    [f"f{j}" for j in range(int(est.n_features_in_))]),
        "columns_named": columns is not None,
        "native_feature_order": names,
        "metadata": _display(getattr(est, "_metadata", None) or {}),
        "fitted_at": getattr(est, "fitted_at_", None),
        "fit_duration_seconds": getattr(est, "fit_duration_", None),
        "seed": _display(params.get("seed")),
        "params": [
            {"name": k, "value": _display(v),
             "default": _display(defaults.get(k)),
             "non_default": k in defaults and not _same(v, defaults[k])}
            for k, v in sorted(params.items())
        ],
        "exposure_column": getattr(est, "_exposure_spec", None),
        "weight_column": getattr(est, "_sample_weight_spec", None),
        "requires": {
            "sample_weight": bool(getattr(est, "_ae_requires_weight_", False)),
            "exposure": bool(getattr(est, "_ae_requires_exposure_", False)),
            "offset": bool(getattr(est, "_ae_requires_offset_", False)),
        },
    }


def _same(a: Any, b: Any) -> bool:
    try:
        return bool(a == b) if not isinstance(a, np.ndarray) else False
    except Exception:
        return False


def _available(value: Any, reason: str) -> dict[str, Any]:
    return {"value": _display(value), "available": value is not None,
            "reason": None if value is not None else reason}


def _graduation_tables(est: "_BaseTBoost", names: list[str]) -> list[dict[str, Any]] | None:
    """Per smoothed-candidate table: its features, the strength chosen (0 = left as fitted),
    whether the training-deviance check rejected it and, on a multiclass fit, its class, from
    ``graduation_report_``."""
    rows = getattr(est, "graduation_report_", None)
    if not rows:
        return None
    labels = getattr(est, "classes_", None)
    classes = [] if labels is None else [str(c) for c in labels]
    out = []
    for g in rows:
        ids = g.get("features") or g.get("u") or []
        k = g.get("class")
        out.append({
            "features": [names[i] if isinstance(i, int) and 0 <= i < len(names) else str(i)
                         for i in ids],
            "alpha": _display(g.get("alpha")),
            "rejected": bool(g.get("rejected", False)),
            "method": _display(g.get("method")),
            "class": (classes[k] if isinstance(k, int) and 0 <= k < len(classes)
                      else None if k is None else str(k)),
        })
    return out


def fit_section(est: "_BaseTBoost", multiclass: bool,
                names: list[str] | None = None) -> dict[str, Any]:
    loaded_or_multi = ("not recorded: a multiclass fit has no per-bag boosting report"
                       if multiclass else "not recorded with this model (saved before 0.7.0)")
    history = getattr(est, "evals_result_", None)
    pruning = getattr(est, "pruning_report_", None)
    why = ("" if pruning is not None
           else "the pruning report was not stored with this model (saved before it was)")
    try:
        binding = est.binding_report_
    except Exception as exc:  # pragma: no cover - defensive
        binding, why_b = None, str(exc)
    else:
        why_b = ""
    return {
        "n_trees": _available(getattr(est, "n_trees_", None), loaded_or_multi),
        "n_trees_per_bag": _available(getattr(est, "n_trees_per_bag_", None), loaded_or_multi),
        "stopping_reason": _available(getattr(est, "stopping_reason_", None), loaded_or_multi),
        "stopping_reason_per_bag": _available(
            getattr(est, "stopping_reason_per_bag_", None), loaded_or_multi),
        "evals_result": _available(
            history, "the per-round history is recorded only on a live fit with eval_set or "
            "callbacks, and is not saved with the model"),
        "pruning": _available(pruning, why),
        "graduation": _available(getattr(est, "graduation_validation_", None),
                                 "no graduation diagnostics recorded"),
        "graduation_tables": _available(
            _graduation_tables(est, names or []),
            "no per-table graduation recorded (graduation off, or a model saved before it was "
            "stored)"),
        "binding_report": _available(binding, why_b),
        "early_stopping": {
            "eval_set": bool(getattr(est, "_fit_eval_set_", False)),
            "validation_fraction": _display(getattr(est, "validation_fraction", None)),
            "early_stopping_rounds": _display(getattr(est, "early_stopping_rounds", None)),
        },
    }


def structure_section(est: "_BaseTBoost", banks: dict[str, Any]) -> dict[str, Any]:
    """Per bank: intercept and one row per effect (dense or factored), by Sobol share."""
    out_banks = []
    for label, bank in banks.items():
        rows = []
        for t in bank.get("tables", []):
            rows.append({"features": t["feature_names"], "order": len(t["feature_names"]),
                         "kind": "dense", "cells": int(np.prod(t["shape"])) if t["shape"] else 1,
                         "sobol": t.get("sobol"), "variance": t.get("variance")})
        for f in bank.get("factored", []) or []:
            names = f.get("feature_names") or [str(u) for u in f.get("feature_set", [])]
            rows.append({"features": names, "order": len(names), "kind": "factored",
                         "cells": None, "sobol": f.get("sobol"), "variance": f.get("variance")})
        rows.sort(key=lambda r: (-(r["sobol"] or 0.0), r["features"]))
        by_order: dict[str, int] = {}
        for r in rows:
            by_order[str(r["order"])] = by_order.get(str(r["order"]), 0) + 1
        out_banks.append({"class": label, "f0": bank.get("f0"), "effects": rows,
                          "by_order": by_order})
    return {"deployed_tables": est._deployed_table_count(), "banks": out_banks}


def features_section(report: dict[str, Any], th: dict[str, float | None]) -> list[dict[str, Any]]:
    """One row per raw feature: importance split into its own one-way table and its share
    through interactions, and the one-way table's effect range, thin cells, missing-value level
    and smoothing strength (first class bank for a multiclass model)."""
    m = report["model"]
    link = m["link"]
    banks = report["structure"]["banks"]
    n_banks = max(len(banks), 1)
    tables = report["tables"]
    first_bank = next(iter(tables.values())) if m["multiclass"] else tables
    grad = report["fit"]["graduation_tables"]["value"] or []
    smoothing = {tuple(g["features"]): g["alpha"] for g in grad}
    thin_mass = th.get("thin_cell_mass") or 0.0
    datasets = [d for d in report["datasets"] if not d.get("error")]
    out = []
    for name in m["native_feature_order"]:
        effects = [e for b in banks for e in b["effects"] if name in e["features"]]
        own = sum((e["sobol"] or 0.0) for e in effects if e["features"] == [name]) / n_banks
        imp = sum((e["sobol"] or 0.0) / len(e["features"]) for e in effects) / n_banks
        entry: dict[str, Any] = {"feature": name, "importance": imp, "own_share": own,
                                 "interaction_share": imp - own, "used": bool(effects),
                                 "interactions": sorted({tuple(e["features"]) for e in effects
                                                         if len(e["features"]) > 1})}
        entry["interactions"] = [list(u) for u in entry["interactions"]]
        if datasets:
            ae = next((a for a in datasets[0]["actual_vs_expected"] if a["feature"] == name), None)
            entry["levels"] = sum(1 for n in (ae["rows"] if ae else []) if n)
        t = next((t for t in first_bank.get("tables", []) if t["feature_names"] == [name]), None)
        if t is not None:
            if link == "log" and t.get("relativities"):
                eff, kind, mult = list(t["relativities"]), "relativity", True
            elif link == "logit":
                eff, kind, mult = [math.exp(v) for v in t["values"]], "odds multiplier", True
            else:
                eff, kind, mult = list(t["values"]), "value", False
            sup = list(t.get("support") or [0.0] * len(eff))
            total = sum(sup)
            live = [i for i, x in enumerate(sup) if x > 0]
            thin = [i for i in live if total > 0 and sup[i] < thin_mass * total]
            solid = [i for i in live if i not in thin]

            def span(idx: list[int]) -> list[float] | None:
                return [min(eff[i] for i in idx), max(eff[i] for i in idx)] if idx else None

            entry["one_way"] = {
                "kind": kind, "multiplicative": mult, "cells": len(eff),
                "range": span(live), "supported_range": span(solid),
                "thin_cells": len(thin), "thin_share": (sum(sup[i] for i in thin) / total
                                                        if total else 0.0),
                "thin_range": span(thin),
                "missing": {"effect": eff[0], "support": sup[0]} if eff else None,
                "smoothing": smoothing.get((name,)),
            }
        out.append(entry)
    out.sort(key=lambda e: -e["importance"])
    return out


def _weighted_quantiles(values: list[float], weights: list[float], qs: tuple[float, ...]
                        ) -> list[float] | None:
    """The weighted lower quantiles (smallest value whose cumulative weight reaches q of the
    total), as `cell_quantiles` defines them."""
    pairs = sorted((v, w) for v, w in zip(values, weights) if w > 0)
    total = sum(w for _, w in pairs)
    if not pairs or total <= 0:
        return None
    cumulative = np.cumsum([w for _, w in pairs])
    return [pairs[min(int(np.searchsorted(cumulative, q * total)), len(pairs) - 1)][0] for q in qs]


def interactions_section(report: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per deployed interaction table (the first class bank of a multiclass model), by
    share of model variance: its features, share, cells, the range of its effect over the cells
    holding exposure, the range holding 90% of the exposure (the support-weighted 5th to 95th
    percentile) and its smoothing strength."""
    m = report["model"]
    link = m["link"]
    tables = report["tables"]
    bank = next(iter(tables.values())) if m["multiclass"] else tables
    grad = {tuple(g["features"]): g["alpha"]
            for g in report["fit"]["graduation_tables"]["value"] or []}
    out = []
    for t in bank.get("tables", []):
        names = list(t["feature_names"])
        if len(names) < 2:
            continue
        if link == "log" and t.get("relativities"):
            eff, mult = [float(v) for v in t["relativities"]], True
        elif link == "logit":
            eff, mult = [math.exp(float(v)) for v in t["values"]], True
        else:
            eff, mult = [float(v) for v in t["values"]], False
        sup = [float(v) for v in t.get("support") or [0.0] * len(eff)]
        live = [i for i, w in enumerate(sup) if w > 0]
        out.append({
            "features": names, "order": len(names), "share": t.get("sobol"), "cells": len(eff),
            "multiplicative": mult,
            "range": [min(eff[i] for i in live), max(eff[i] for i in live)] if live else None,
            "spread": _weighted_quantiles(eff, sup, _SPREAD_QUANTILES),
            "smoothing": grad.get(tuple(names)),
        })
    out.sort(key=lambda e: (-(e["share"] or 0.0), e["features"]))
    return out


# --------------------------------------------------------------------------- assembly ---------

def build_report(
    est: "_BaseTBoost", X: Any, y: Any, *, sample_weight: Any | None, exposure: Any | None,
    offset: Any | None, ref_measure: str | None, eval_set: Any,
    eval_sample_weight: Any | None, eval_exposure: Any | None, eval_offset: Any | None,
    labels: Any, name: str | None, thresholds: dict[str, Any] | None,
) -> dict[str, Any]:
    from ._pricing import annotate_unseen_cells

    th = resolve_thresholds(thresholds)
    model, multiclass = _bank_models(est)
    sets = [(X, y, sample_weight, exposure, offset)] + resolve_eval_sets(
        eval_set, eval_sample_weight, eval_exposure, eval_offset)
    names = resolve_labels(labels, len(sets))
    specs = [(lab, *spec) for lab, spec in zip(names, sets)]
    prepared = [
        prepare(est, Xd, yd, sample_weight=wd, exposure=ed, offset=od, label=lab,
                check_unseen=True)
        for lab, Xd, yd, wd, ed, od in specs
    ]
    first = prepared[0]
    if first.error is not None:
        raise ValueError(f"the first dataset cannot be scored: {first.error}")
    policy = getattr(est, "unknown_category", "rare")
    n_banks = int(model.n_classes) if multiclass else 1
    axes = [
        json.loads(annotate_unseen_cells(
            json.dumps({"axes": json.loads(
                model.merged_axes(class_index=k) if multiclass else model.merged_axes())}),
            policy,
        ))["axes"]
        for k in range(n_banks)
    ]
    whole = whole_valued(first)
    for bank_axes in axes:
        mark_integer_axes(bank_axes, whole)
    class_labels = [str(c) for c in est.classes_] if multiclass else [None]

    from .sklearn import _canonical_objective

    obj = _canonical_objective(est.objective)
    rho = float(getattr(est, "tweedie_rho", 1.5) or 1.5)

    def ae_lists(p: Prepared) -> list[dict[str, Any]]:
        out = []
        for k in range(n_banks):
            if multiclass:
                assert p.codes is not None and p.proba is not None
                pk = p.proba[:, k]
                rows = factor_ae(p, k, actual=(p.codes == k).astype(np.float64), expected=pk,
                                 variance=pk * (1.0 - pk))
            else:
                assert p.pred is not None
                rows = factor_ae(p, k, variance=dispersion(obj, p.y, p.pred, p.w, rho)
                                 * _variance(obj, p.pred, rho))
            for r in rows:
                r["ae"] = [float(v) if np.isfinite(v) else None for v in r["ae"]]
                r["axis"] = {**axes[k][int(r["raw"])], "band_of": None}
                if multiclass:
                    r["class"] = class_labels[k]
            out.extend(rows)
        return out

    datasets = []
    for p in prepared:
        entry: dict[str, Any] = {"label": p.label, "rows": p.rows, "unseen": p.unseen,
                                 "error": p.error}
        if p.error is None:
            entry["metrics"] = (multiclass_metrics(est, p) if multiclass
                                else single_output_metrics(est, p))
            entry["actual_vs_expected"] = ae_lists(p)
            # Cell 0 is the missing-value cell of every grid.
            entry["missing"] = [
                {"feature": f["feature"], "rows": f["rows"][0],
                 "mass_share": _ratio(f["mass"][0], sum(f["mass"]))}
                for f in entry["actual_vs_expected"][: len(p.names)]
            ]
        datasets.append(entry)
    if not multiclass:
        prediction_histograms(est, prepared, datasets)

    # Factor bands: at most `_PSI_BINS` per feature, from the first dataset's mass, applied to
    # every dataset, so the factor check and the level summary read the same groups everywhere.
    if datasets[0].get("error") is None:
        base_entries = datasets[0]["actual_vs_expected"]
        bins_of = [
            _psi_bins(np.asarray(e["mass"], dtype=float), e["axis"].get("levels"))
            for e in base_entries
        ]
        for d in datasets:
            if d.get("error") is None:
                for e, bins in zip(d["actual_vs_expected"], bins_of):
                    e["bands"] = factor_bands(e, bins, e["axis"])
    for d in datasets:
        for e in d.get("actual_vs_expected") or []:
            e.pop("_var", None)
    design, vectors = est._resolve_fit_vectors(
        X, {"y": y, "sample_weight": sample_weight, "exposure": exposure, "offset": offset}
    )
    # Shapley values per level on the first dataset (train): one per (class bank, raw feature),
    # the order `ae_lists` lays the entries out in, from the bank the tables below come from.
    if datasets[0].get("error") is None:
        measure = table_measure(est, design, vectors["sample_weight"], vectors["exposure"],
                                ref_measure)
        shap = [cell for per_bank in shap_by_cell(est, first, measure) for cell in per_bank]
        for e, stats in zip(datasets[0]["actual_vs_expected"], shap):
            e["shap"] = stats

    # Drift from the train data to each later dataset, on the train data's bins.
    drift = []
    base = datasets[0]["actual_vs_expected"][: len(first.names)]
    for d in datasets[1:]:
        if d.get("error") is None:
            drift.append({"reference": datasets[0]["label"], "compared": d["label"],
                          "features": psi(base, d["actual_vs_expected"][: len(first.names)],
                                          axes[0])})

    exported = json.loads(est.tables(
        design, ref_measure=ref_measure, sample_weight=vectors["sample_weight"],
        exposure=vectors["exposure"],
    ))
    banks = exported if multiclass else {"": exported}
    for bank in banks.values():
        for t in bank.get("tables", []):
            mark_integer_axes(t.get("axes") or [], whole)

    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "name": name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "caveat": ("This report describes the data passed to it. It cannot verify which rows "
                   "were used to fit the model or which were held out."),
        "prediction_units": (
            "per unit exposure" if getattr(est, "_ae_requires_exposure_", False)
            else "class probability" if multiclass
            else "target units (including any rate units defined by the caller)"
        ),
        "reference_normalization": "zero mean on the link scale; not arithmetic mean rate",
        "band_interpretation": "bag stability; not claim prediction or credibility intervals",
        "model": model_section(est, multiclass, first.names),
        "fit": fit_section(est, multiclass, first.names),
        "thresholds": th,
        "threshold_descriptions": THRESHOLD_DESCRIPTIONS,
        "datasets": datasets,
        "drift": drift,
        "tables": exported,
        "actual_vs_expected": datasets[0]["actual_vs_expected"],
        "pruning": getattr(est, "pruning_report_", None),
        "graduation": getattr(est, "graduation_validation_", None),
    }
    report["structure"] = structure_section(est, banks)
    report["features"] = features_section(report, th)
    report["interactions"] = interactions_section(report)
    interaction_ae(report, prepared, obj, rho)
    report["review"] = review(est, report, th)
    report["summary_metrics"] = summary_metrics(report)
    return json_safe(report)  # type: ignore[no-any-return]
