"""Monotone constraints run the whole post-fit pipeline (2026-10-14).

A monotone fit used to keep its full table bank, with selection, banding, the guard and graduation
all switched off. Now every stage runs as it does for an unconstrained fit, and the deployed tables
are projected back onto the signs last (`t_boost_core::monotone`). The checks below sweep each
constrained feature across a grid of every other feature, missing values and every category
included, rather than along a few training rows.
"""

from __future__ import annotations

import itertools
import json
import re

import numpy as np
import polars as pl
import pytest

from t_boost import TBoostClassifier, TBoostRegressor

SIGNS = {"a": 1, "b": -1}


def _data(n: int = 4000, seed: int = 0) -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2, 2, n)
    b = rng.uniform(-2, 2, n)
    d = rng.uniform(-2, 2, n)
    c = rng.choice(list("pqrs"), n)
    eta = 0.6 * np.tanh(a) - 0.4 * np.tanh(b) + 0.35 * a * b * (c == "p") + 0.2 * d
    y = rng.poisson(np.exp(eta - 1.0)).astype(float)
    return pl.DataFrame({"a": a, "b": b, "c": c, "d": d}), y


def _worst_step(est, feat: str, sign: int) -> float:
    """The most negative step of the raw score along `feat` (sign applied), over a grid of every
    other feature: numeric quantiles plus a missing value, every category plus a missing one."""
    others = {
        "a": [*np.linspace(-1.9, 1.9, 7), None],
        "b": [*np.linspace(-1.9, 1.9, 7), None],
        "d": [*np.linspace(-1.9, 1.9, 5), None],
        "c": ["p", "q", "r", "s", None],
    }
    sweep = np.linspace(-1.99, 1.99, 40)
    names = [k for k in ("a", "b", "c", "d") if k != feat]
    combos = list(itertools.product(*(others[k] for k in names)))
    rows: dict[str, list] = {k: [] for k in ("a", "b", "c", "d")}
    for combo in combos:
        for v in sweep:
            rows[feat].append(float(v))
            for k, x in zip(names, combo):
                rows[k].append(x)
    frame = pl.DataFrame(rows, schema={"a": pl.Float64, "b": pl.Float64, "c": pl.String,
                                       "d": pl.Float64})
    score = est.decision_function if isinstance(est, TBoostClassifier) else est.predict_raw
    raw = np.asarray(score(frame), dtype=np.float64).reshape(len(combos), len(sweep))
    return float((sign * np.diff(raw, axis=1)).min())


def _assert_monotone(est) -> None:
    for feat, sign in SIGNS.items():
        assert _worst_step(est, feat, sign) >= -1e-6, feat


@pytest.fixture(scope="module")
def data() -> tuple[pl.DataFrame, np.ndarray]:
    return _data()


@pytest.fixture(scope="module")
def fitted(data) -> TBoostRegressor:
    X, y = data
    return TBoostRegressor(objective="poisson", n_trees=150, seed=0, n_jobs=2,
                           monotone_constraints=SIGNS).fit(X, y)


def test_every_stage_runs_and_the_deployed_tables_are_monotone(fitted) -> None:
    rep = fitted.pruning_report_
    assert rep["selector"] == "ranked_path"
    assert len(rep["kept"]) < rep["n_candidates"], "selection must actually drop tables"
    assert "banding" in rep
    assert "skipped" not in fitted.graduation_validation_
    mono = rep["monotone"]
    assert {f["feature"] for f in mono["features"]} == {0, 1}
    for f in mono["features"]:
        assert f["worst_step_after"] >= -1e-9
    _assert_monotone(fitted)


def test_the_repaired_model_round_trips_and_its_exports_agree(fitted, data) -> None:
    X, _ = data
    for loaded in (TBoostRegressor.from_bytes(fitted.to_bytes()),
                   TBoostRegressor.from_json(fitted.to_json())):
        np.testing.assert_array_equal(loaded.predict(X), fitted.predict(X))
    # Contributions reconstruct the raw score exactly (validated inside), and a re-centred
    # export of the same function is still exact.
    fitted.predict_contributions(X[:200], validate=True)
    assert json.loads(fitted.tables(X[:200], ref_measure="uniform"))["mode"] == "Exact"


def test_a_single_bag_fit_keeps_its_factored_effects_and_is_repaired(data) -> None:
    # One bag: no out-of-bag rows, so the fold vote selects. Unbanded, the three-way effects
    # stay in factored form for the repair to account for.
    X, y = data
    est = TBoostRegressor(objective="poisson", n_trees=150, n_bags=1, seed=2, n_jobs=2,
                          band_tolerance=None, monotone_constraints=SIGNS).fit(X, y)
    assert est.pruning_report_["selector"] != "ranked_path"
    assert json.loads(est.tables(X[:50]))["factored"], "the fixture must carry factored effects"
    assert est.pruning_report_["monotone"]["certified"] is False, "the fixture must need repair"
    _assert_monotone(est)


@pytest.mark.parametrize("kwargs", [
    {"prune_main_effects": True},
    {"prune_table_budget": 1},
    {"prune_box_budget": 4},
    {"graduation_high_order_alpha": 0.5},
    {"prune_rebalance": False, "graduate": False},
])
def test_settings_that_used_to_be_refused_now_combine(data, kwargs) -> None:
    X, y = data
    est = TBoostRegressor(objective="poisson", n_trees=150, seed=0, n_jobs=2,
                          monotone_constraints=SIGNS, **kwargs).fit(X, y)
    _assert_monotone(est)


def test_an_unpruned_fit_is_banded_graduated_and_repaired(data) -> None:
    X, y = data
    est = TBoostRegressor(objective="poisson", n_trees=150, seed=0, n_jobs=2, prune=False,
                          monotone_constraints=SIGNS).fit(X, y)
    rep = est.pruning_report_
    assert rep["selector"] == "none"
    assert rep["banding"].get("skipped") is None
    assert "skipped" not in est.graduation_validation_
    assert "monotone" in rep
    _assert_monotone(est)


@pytest.mark.parametrize("signs", [{"c": 1}, [0, 0, 1, 0], {2: -1}])
def test_a_sign_on_a_categorical_raises(data, signs) -> None:
    X, y = data
    with pytest.raises(ValueError, match="numeric features only"):
        TBoostRegressor(n_trees=5, monotone_constraints=signs).fit(X, y)


def test_a_binary_classifier_is_repaired_too(data) -> None:
    X, _ = data
    rng = np.random.default_rng(3)
    a, b = X["a"].to_numpy(), X["b"].to_numpy()
    p = 1 / (1 + np.exp(-(np.tanh(a) - 0.6 * np.tanh(b) + 0.5 * a * b)))
    yb = (rng.uniform(size=len(p)) < p).astype(int)
    est = TBoostClassifier(n_trees=150, seed=0, n_jobs=2, monotone_constraints=SIGNS).fit(X, yb)
    assert "monotone" in est.pruning_report_
    _assert_monotone(est)


def test_the_html_report_shows_the_repair(fitted, data) -> None:
    X, y = data
    html = fitted.html_report(X.with_columns(pl.Series("y", y)), "y", name="monotone")
    assert "Monotone constraints" in html
    for feat in SIGNS:
        assert re.search(rf"<td[^>]*>{feat}</td><td[^>]*>(in|de)creasing</td>", html), feat


def test_a_repaired_fit_is_byte_identical_across_thread_counts(data) -> None:
    from _artifact import model_bytes

    X, y = data
    fits = [TBoostRegressor(objective="poisson", n_trees=150, seed=0, n_jobs=jobs,
                            monotone_constraints=SIGNS).fit(X, y) for jobs in (1, 4)]
    assert fits[0].pruning_report_["monotone"]["certified"] is False
    assert model_bytes(fits[0]) == model_bytes(fits[1])
