"""The DEPLOYED-BOX BUDGET at the estimator surface (`prune_box_budget`).

The unit of deployed size for a factored effect is its BOX count, not its table count: one
rank-1 region box per realized tree-region, folded across trees only where the per-axis masks
agree exactly. A depth-3 tree contributes one box; a depth-6 tree contributes up to `Pi k_i`.
Lifting `max_depth` past 3 therefore barely moves the table count and multiplies the boxes
instead, which is what this knob exists to cap. Everything here is stated against the two guarantees the gate
sells: OFF is bit-identical, and ON lands under the number it was given.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from t_boost.sklearn import TBoostClassifier, TBoostRegressor


def _frame(n: int = 1200, seed: int = 5, k: int = 0):
    """A genuine 3-way sign product plus a main-effect decoy: only a deep interaction tree
    reaches it, so a depth-6 fit deploys real factored boxes to cap."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    lin = (
        2.0 * np.sign(x[:, 0]) * np.sign(x[:, 1]) * np.sign(x[:, 2])
        + 0.7 * x[:, 3]
        + 0.1 * rng.normal(size=n)
    )
    frame = pl.DataFrame({f"n{i}": x[:, i] for i in range(4)})
    if k == 0:
        return frame, lin
    cuts = np.quantile(lin, np.linspace(0, 1, k + 1)[1:-1])
    return frame, np.digitize(lin, cuts)


def _boxes(model) -> int:
    """Total deployed rank-1 boxes across every bank the model carries (one per class for a
    multiclass fit — a reader faces all of them)."""
    total = 0

    def walk(node):
        nonlocal total
        if isinstance(node, str) and node.lstrip().startswith("{"):
            try:
                walk(json.loads(node))
            except ValueError:
                return
            return
        if isinstance(node, dict):
            if isinstance(node.get("tables"), list):
                for f in node.get("factored") or []:
                    total += len(f.get("boxes") or [])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(json.loads(model.to_json()))
    return total


def _fit(box_budget: int, *, k: int = 0, n: int = 1200, n_trees: int = 60, **kw):
    frame, y = _frame(n=n, k=k)
    est = (TBoostClassifier if k else TBoostRegressor)(
        n_trees=n_trees,
        seed=3,
        n_jobs=4,
        max_depth=6,
        prune_box_budget=box_budget,
        **kw,
    )
    est.fit(frame, y)
    return est


def test_fixture_actually_deploys_boxes_to_cap() -> None:
    """Guard the guard: every assertion below is vacuous on a bank with no factored effect."""
    est = _fit(0)
    assert _boxes(est) > 50


def test_budget_off_is_bit_identical_and_reports_nothing() -> None:
    off = _fit(0)
    default = TBoostRegressor(n_trees=60, seed=3, n_jobs=4, max_depth=6)
    frame, y = _frame()
    default.fit(frame, y)
    assert off.to_bytes() == default.to_bytes()
    assert "box_budget" not in off.pruning_report_


def test_a_generous_budget_is_a_bit_identical_no_op() -> None:
    off = _fit(0)
    n = _boxes(off)
    for budget in (n, n + 1, 10 * n):
        on = _fit(budget)
        assert on.to_bytes() == off.to_bytes(), budget
        assert on.pruning_report_["box_budget"]["engaged"] is False, budget
        assert on.pruning_report_["box_budget"]["boxes_before"] == n


@pytest.mark.parametrize("divisor", [2, 4, 32])
def test_a_tight_budget_lands_under_it(divisor: int) -> None:
    off = _fit(0)
    n = _boxes(off)
    budget = max(1, n // divisor)
    on = _fit(budget)
    got = _boxes(on)
    assert got <= budget, f"budget {budget}, deployed {got}"
    rep = on.pruning_report_["box_budget"]
    assert rep["engaged"] is True
    assert rep["max_boxes"] == budget
    assert rep["boxes_after"] == got
    assert rep["effects_after"] <= rep["effects_before"]
    # Still a working model: the budget drops effects, it does not corrupt the bank.
    frame, _ = _frame()
    assert np.all(np.isfinite(np.asarray(on.predict(frame), dtype=float)))


def test_the_budget_never_spends_the_dense_bank() -> None:
    """Dense tables cost zero boxes and are governed by the cell firewall, not by this gate:
    a budget of 1 must leave the main effects and pairs exactly where they were."""
    off = _fit(0)
    on = _fit(1)
    doc_off = json.loads(off.to_json())
    doc_on = json.loads(on.to_json())

    def dense(doc):
        out = []

        def walk(node):
            if isinstance(node, str) and node.lstrip().startswith("{"):
                try:
                    walk(json.loads(node))
                except ValueError:
                    return
                return
            if isinstance(node, dict):
                if isinstance(node.get("tables"), list):
                    out.append(sorted(json.dumps(t["axes"], sort_keys=True)
                                      for t in node["tables"]))
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(doc)
        return out

    assert dense(doc_on) == dense(doc_off)


def test_multiclass_honors_the_budget() -> None:
    """The K>=3 path selects ONE keep-set shared by every class, so a support's box cost is
    summed over the per-class banks — and the same knob has to reach it."""
    mc = dict(k=4, n=4000, n_trees=150)
    off = _fit(0, **mc)
    n = _boxes(off)
    assert n > 0, "the multiclass fixture must deploy boxes"
    on = _fit(max(1, n // 4), **mc)
    assert _boxes(on) <= max(1, n // 4)
    assert on.pruning_report_["box_budget"]["engaged"] is True
    generous = _fit(10 * n, **mc)
    assert generous.to_bytes() == off.to_bytes()


def test_budget_without_pruning_raises_rather_than_no_ops() -> None:
    frame, y = _frame()
    est = TBoostRegressor(n_trees=20, seed=3, n_jobs=2, prune=False, prune_box_budget=100)
    with pytest.raises(ValueError, match="prune_box_budget has no effect with prune=False"):
        est.fit(frame, y)
    clf = TBoostClassifier(n_trees=20, seed=3, n_jobs=2, prune=False, prune_box_budget=100)
    fr3, y3 = _frame(k=3)
    with pytest.raises(ValueError, match="prune_box_budget has no effect with prune=False"):
        clf.fit(fr3, y3)


def test_negative_budget_is_refused() -> None:
    frame, y = _frame()
    est = TBoostRegressor(n_trees=20, seed=3, n_jobs=2, prune_box_budget=-5)
    with pytest.raises(ValueError, match="prune_box_budget must be >= 0"):
        est.fit(frame, y)


def test_budget_round_trips_through_get_params() -> None:
    est = TBoostRegressor(prune_box_budget=1234)
    assert est.get_params()["prune_box_budget"] == 1234
    assert TBoostClassifier(prune_box_budget=7).get_params()["prune_box_budget"] == 7
    est.set_params(prune_box_budget=9)
    assert est.get_params()["prune_box_budget"] == 9


@pytest.fixture(autouse=True)
def _legacy_fold_vote_unbanded(monkeypatch: pytest.MonkeyPatch) -> None:
    """This module pins the fold-vote selector (its guard, evidence and box-budget mechanics) on
    unbanded tables: the ranked path and banding became the defaults on 2026-09-26."""
    from t_boost import TBoostClassifier as _C
    from t_boost import TBoostRegressor as _R

    for cls in (_R, _C):
        defaults = cls.__init__.__kwdefaults__
        monkeypatch.setitem(defaults, "prune_selector", "fold_vote")
        monkeypatch.setitem(defaults, "band_tolerance", None)
