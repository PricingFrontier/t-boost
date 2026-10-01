"""The MULTI-WAY TABLE budget and price at the estimator surface.

Ralph moved the explainability bar on 2026-08-27: cells within a table are FREE, and the count
of >=3-way tables is the only size constraint ("100 3-way tables is not explainable").
`prune_box_budget` prices the other quantity — resolution — and bounds table count only as a
side effect of dropping supports binarily, so it buys its table discipline by starving the
tables it keeps. `prune_table_budget` caps the count directly; `prune_lambda_tables` is the
selection-time analogue.

The Rust side is covered by `crates/t-boost-core/tests/table_budget.rs`. This file covers what
only the estimator can break: the sklearn parameter surface, the deploy assembly's new 3-tuple
unpack, the `pruning_report_` block, and the CLASSIFIER (multiclass) path.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from t_boost.sklearn import TBoostClassifier, TBoostRegressor


def _frame(n: int = 1200, seed: int = 5, k: int = 0):
    """A genuine 3-way sign product plus a main-effect decoy, so a depth-6 fit deploys real
    >=3-way tables for the cap to bite on."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5)).astype(np.float32)
    lin = (
        2.0 * np.sign(x[:, 0]) * np.sign(x[:, 1]) * np.sign(x[:, 2])
        + 1.2 * np.sign(x[:, 1]) * np.sign(x[:, 3]) * np.sign(x[:, 4])
        + 0.7 * x[:, 3]
        + 0.1 * rng.normal(size=n)
    )
    frame = pl.DataFrame({f"n{i}": x[:, i] for i in range(5)})
    if k == 0:
        return frame, lin
    cuts = np.quantile(lin, np.linspace(0, 1, k + 1)[1:-1])
    return frame, np.digitize(lin, cuts)


def _arity(model) -> dict[int, int]:
    """Kept-table counts by interaction order, summed over every bank the model carries (one
    per class for a multiclass fit — a filing reader faces all of them)."""
    out: dict[int, int] = {}

    def walk(node):
        # A classifier's `to_json` carries the model as a nested JSON STRING, so a walker that
        # only descends dicts finds nothing and every assertion below reads as vacuously true.
        if isinstance(node, str) and node.lstrip().startswith("{"):
            try:
                walk(json.loads(node))
            except ValueError:
                return
            return
        if isinstance(node, dict):
            if isinstance(node.get("tables"), list):
                for t in (node.get("tables") or []) + (node.get("factored") or []):
                    order = len(t.get("u") or [])
                    out[order] = out.get(order, 0) + 1
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(json.loads(model.to_json()))
    return out


def _n_ge3(model) -> int:
    return sum(v for k, v in _arity(model).items() if k >= 3)


# Sized for a test suite, not for a battery: n=900 keeps the fit above
# `_PRUNE_GUARD_MIN_ROWS` (500) so the prune guard is LIVE — the regime the no-op regression
# below was hiding in — while 30 trees keep each fit to a few seconds. The first cut ran
# 1,200 rows x 60 trees and took over twenty minutes, which is a defect of its own.
def _fit(*, k: int = 0, n: int | None = None, n_trees: int | None = None, **kw):
    """The multiclass case needs its own, heavier settings and that is a finding, not a knob.

    A K-class softmax fit splits the same signal across K one-vs-rest boosters, so each sees a
    fraction of it and the interaction hurdle refuses the 3-way structure the regressor finds
    easily: at the regressor's own settings (900 rows, 30 trees, hurdle 0.1) the classifier
    deploys ZERO >=3-way supports, and every cap assertion would pass vacuously. 1,500 rows, 60
    trees and hurdle 0.0 put four of them in the bank, which is enough to cap.
    """
    mc = k > 0
    frame, y = _frame(n=n if n is not None else (1500 if mc else 900), k=k)
    est = (TBoostClassifier if mc else TBoostRegressor)(
        n_trees=n_trees if n_trees is not None else (60 if mc else 30),
        seed=3,
        n_jobs=4,
        max_depth=6,
        interaction_gain_hurdle=0.0 if mc else 0.1,
        **kw,
    )
    est.fit(frame, y)
    return est


def test_fixture_actually_deploys_multi_way_tables_to_cap() -> None:
    """Guard the guard: every assertion below is vacuous on a bank with no >=3-way table."""
    assert _n_ge3(_fit()) >= 8


@pytest.mark.parametrize("k", [0, 3])
def test_budget_and_price_off_are_bit_identical_and_report_nothing(k: int) -> None:
    """OFF is the shipping claim, so it is checked on BYTES, not on a summary — and on the
    classifier too, which the Rust fingerprint battery never exercises.

    "The default is inert" and "an explicit zero is inert" are different statements; both are
    checked, along with an exotic arity floor, which is the one new field whose default is not
    zero and so the one that could leak through an unguarded read.
    """
    baseline = _fit(k=k)
    for kw in (
        {"prune_table_budget": 0, "prune_lambda_tables": 0.0},
        # The floor is only ever consulted behind the armed check, so an EXTREME legal value
        # must still be inert while both knobs are off — 1 would price the whole bank if it
        # ever leaked through an unguarded read.
        {"prune_table_min_arity": 1},
    ):
        est = _fit(k=k, **kw)
        assert est.to_json() == baseline.to_json(), f"{kw} moved the artifact"
        assert "table_budget" not in (est.pruning_report_ or {}), kw


@pytest.mark.parametrize("k", [0, 3])
def test_a_cap_lands_at_or_under_the_number_it_was_given(k: int) -> None:
    """The deploy assembly returns a 3-tuple now; a cap that silently did nothing, or a report
    describing a different bank, would both show up here.

    The rungs are DERIVED from the fixture's own uncapped count rather than hardcoded. A
    hardcoded rung is a bug waiting for the fixture to drift under it: the first cut of this
    test asked for `engaged is True` at cap=4 on a bank that carries 4, which is a no-op by
    contract — the test was wrong, not the code.
    """
    base = _fit(k=k)
    uncapped = sum(1 for u in base.pruning_report_["kept"] if len(u) >= 3)
    assert uncapped >= 3, f"fixture must carry >=3-way supports to cap: {uncapped}"
    for cap in {1, max(1, uncapped // 2)}:
        est = _fit(k=k, prune_table_budget=cap)
        rep = est.pruning_report_["table_budget"]
        assert rep["engaged"] is True, (cap, rep)
        assert rep["min_arity"] == 3
        assert rep["tables_before"] == uncapped, (cap, rep)
        assert rep["tables_after"] <= cap, rep
        # For a K-class fit the cap counts SUPPORTS in the shared keep-set while the artifact
        # carries one copy per class, so the deployed census is at most `cap` per class.
        assert _n_ge3(est) <= cap * max(1, k), (cap, _arity(est))
        assert _n_ge3(est) < _n_ge3(base)


def test_a_cap_at_or_above_the_banks_own_count_is_a_verbatim_no_op() -> None:
    """Monotone/no-op is what makes "budget off is byte-identical" checkable rather than
    hopeful, and it must hold at the BOUNDARY, not only far above it.

    This is the assertion that caught a real regression. An earlier cut skipped the prune
    guard's out-of-bag path whenever a budget was armed — reasoning that the guard scores the
    pre-budget keep-set and so cannot describe a budgeted deploy. But `_PRUNE_GUARD_MIN_ROWS`
    is 500, so the guard is live on this 1,200-row fixture, and suppressing it turned a
    NON-BINDING cap into a model change: 10 three-way tables became 4. The guard's blindness is
    now reported (`pruning_report_["guard"]["budget_blind"]`) rather than routed around.
    """
    baseline = _fit()
    n = _n_ge3(baseline)
    for cap in (n, n + 1):
        est = _fit(prune_table_budget=cap)
        assert est.to_json() == baseline.to_json(), cap
        assert est.pruning_report_["table_budget"]["engaged"] is False, cap


def test_a_guard_that_cannot_see_the_budget_says_so_in_the_report() -> None:
    """The out-of-bag guard scores raw per-support sums over the SELECTION keep-set, so once a
    budget ENGAGES its verdict describes a bank that did not ship. That is a real limitation and
    the report has to carry it; a silent wrong verdict is worse than a flagged one."""
    base = _fit()
    assert base.pruning_report_["guard"].get("budget_blind") is None
    tight = _fit(prune_table_budget=1)
    assert tight.pruning_report_["table_budget"]["engaged"] is True
    if tight.pruning_report_["guard"].get("evidence") == "oob":
        assert tight.pruning_report_["guard"]["budget_blind"] is True


def test_the_cap_never_touches_a_main_effect_or_a_pair() -> None:
    """The property that distinguishes this from "prune harder". Mains and pairs are what a
    filing reads; if the tightest cap could drop one, the knob would be a second selection rule
    wearing a readability label."""
    base = _arity(_fit())
    tight = _arity(_fit(prune_table_budget=1))
    assert tight.get(1, 0) == base.get(1, 0), (base, tight)
    assert tight.get(2, 0) == base.get(2, 0), (base, tight)


def test_the_arity_floor_selects_what_the_cap_counts() -> None:
    """At floor 4 a bank whose top arity is 3 has nothing to cap, so the tightest budget is a
    no-op — the floor is doing the work, not the cap."""
    baseline = _fit()
    assert max(_arity(baseline)) == 3, "fixture must top out at order 3 for this to mean anything"
    est = _fit(prune_table_budget=1, prune_table_min_arity=4)
    assert est.to_json() == baseline.to_json()
    assert est.pruning_report_["table_budget"]["engaged"] is False


def test_the_price_is_monotone_in_the_priced_band() -> None:
    """A selection-time price must never GROW the priced band — that monotonicity is what makes
    it usable as a tuned dial. It is deliberately not asserted to reach zero: it re-picks a
    waypoint on a fixed deviance-greedy path and saturates, which is why the BUDGET exists."""
    last = _n_ge3(_fit())
    for lam in (1e-2, 1.0, 100.0):
        n = _n_ge3(_fit(prune_lambda_tables=lam))
        assert n <= last, f"lambda={lam} grew the >=3-way count {last} -> {n}"
        last = n


def test_the_knobs_round_trip_through_get_params() -> None:
    """sklearn clones estimators through get_params/set_params, so a param that is transformed
    in `__init__` silently breaks every CV wrapper. All three must be plain attributes."""
    est = TBoostRegressor(
        prune_table_budget=7, prune_lambda_tables=0.25, prune_table_min_arity=2
    )
    params = est.get_params()
    assert params["prune_table_budget"] == 7
    assert params["prune_lambda_tables"] == 0.25
    assert params["prune_table_min_arity"] == 2
    clone = TBoostRegressor(**params)
    assert clone.get_params() == params


@pytest.mark.parametrize(
    "kw, message",
    [
        ({"prune_table_budget": -1}, "prune_table_budget must be >= 0"),
        ({"prune_lambda_tables": -1.0}, "prune_lambda_tables must be a finite value >= 0"),
        ({"prune_lambda_tables": float("nan")}, "prune_lambda_tables must be a finite value"),
        ({"prune_table_min_arity": 0}, "prune_table_min_arity must be in 1..8"),
        ({"prune_table_min_arity": 9}, "prune_table_min_arity must be in 1..8"),
    ],
)
def test_a_malformed_knob_raises_rather_than_silently_degrading(kw, message) -> None:
    frame, y = _frame(n=400)
    with pytest.raises(ValueError, match=message.replace("..", r"\.\.")):
        TBoostRegressor(n_trees=10, seed=3, n_jobs=2, **kw).fit(frame, y)


def test_a_budget_without_pruning_raises_rather_than_no_opping() -> None:
    """The budget trims the prune's keep-set; an unpruned fit has none, so honoring the
    parameter is impossible and ignoring it would be a silent lie."""
    frame, y = _frame(n=400)
    with pytest.raises(ValueError, match="prune_table_budget has no effect with prune=False"):
        TBoostRegressor(
            n_trees=10, seed=3, n_jobs=2, prune=False, prune_table_budget=4
        ).fit(frame, y)


@pytest.fixture(autouse=True)
def _legacy_fold_vote_unbanded(monkeypatch: pytest.MonkeyPatch) -> None:
    """This module pins the fold-vote selector (its guard, evidence gate and slope machinery) on
    unbanded tables: the ranked path and banding became the defaults on 2026-09-26."""
    from t_boost import TBoostClassifier as _C
    from t_boost import TBoostRegressor as _R

    for cls in (_R, _C):
        defaults = cls.__init__.__kwdefaults__
        monkeypatch.setitem(defaults, "prune_selector", "fold_vote")
        monkeypatch.setitem(defaults, "band_tolerance", None)
