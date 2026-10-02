"""Python-surface tests for the gated `max_delta_step` default (spec §05.6 addendum, av35).

The Rust gate G-MDS (`crates/t-boost-core/tests/gated_delta_step.rs`) owns the engine
semantics. This file owns the knob surface: the tri-state parameter parses the way the
docstring says, the shipped default is tweedie-only, an explicit ``max_delta_step`` still
wins, and the fitted estimator publishes a usable ``delta_step_gate_`` diagnostic.
"""

from __future__ import annotations

import numpy as np
import pytest

from t_boost import TBoostRegressor
from _artifact import model_bytes


def _collapse_data(n: int = 2000, seed: int = 0):
    """Aggregated-exposure pure-premium data with a cleanly-separable zero-claim subgroup.

    Rows with ``x0 == 0`` never have a claim, so boosting drives their predicted rate toward
    zero without bound — the production failure mode the gate exists to catch.
    """
    rng = np.random.default_rng(seed)
    x0 = (np.arange(n) % 11 == 0).astype(np.float32)  # 1 == the zero-claim subgroup
    x1 = (np.arange(n) % 7).astype(np.float32)
    x2 = rng.normal(size=n).astype(np.float32)
    expo = (1.0 + (np.arange(n) % 3)).astype(np.float32)
    claim = ((np.arange(n) % 4 == 0) & (x0 == 0)).astype(np.float32)
    y = (claim * (50.0 + 40.0 * x1)).astype(np.float32)
    return np.column_stack([x0, x1, x2]).astype(np.float32), y, expo


def _benign_data(n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x0 = (np.arange(n) % 6).astype(np.float32)
    x1 = (np.arange(n) % 4).astype(np.float32)
    x2 = rng.normal(size=n).astype(np.float32)
    expo = (1.0 + (np.arange(n) % 3)).astype(np.float32)
    y = (10.0 + 5.0 * x0 + 3.0 * x1).astype(np.float32)
    return np.column_stack([x0, x1, x2]).astype(np.float32), y, expo


def _fit(X, y, w, **kw):
    params = dict(
        objective="tweedie",
        tweedie_rho=1.7,
        n_trees=120,
        learning_rate=0.3,
        prune=False,
        n_bags=0,
        validation_fraction=None,
        seed=3,
        n_jobs=2,
    )
    params.update(kw)
    est = TBoostRegressor(**params)
    est.fit(X, y, sample_weight=w)
    return est


def test_tweedie_default_arms_the_gate_and_fires_on_a_collapse():
    X, y, w = _collapse_data()
    est = _fit(X, y, w)
    rep = est.delta_step_gate_
    assert rep is not None, "tweedie must arm the gate by default"
    assert rep["engaged"] is True
    assert rep["engaged_round"] >= 1  # round 0 cannot fire: every row's score offset is 0
    assert rep["capped_step"] == pytest.approx(0.3, abs=1e-6)
    assert rep["log_threshold"] == pytest.approx(np.log(1e-3), abs=1e-6)
    assert rep["min_log_rate_ratio"] < rep["log_threshold"]


def test_silent_gate_is_bit_identical_to_gate_off():
    X, y, w = _benign_data()
    on = _fit(X, y, w)
    off = _fit(X, y, w, max_delta_step_gated=False)
    assert on.delta_step_gate_ is not None
    assert on.delta_step_gate_["engaged"] is False
    assert off.delta_step_gate_ is None
    # Predictions AND the serialized model must match exactly, not approximately.
    np.testing.assert_array_equal(on.predict(X), off.predict(X))
    assert model_bytes(on) == model_bytes(off)


def test_firing_gate_changes_the_model_and_gate_off_does_not():
    X, y, w = _collapse_data()
    on = _fit(X, y, w)
    off = _fit(X, y, w, max_delta_step_gated=False)
    assert on.delta_step_gate_["engaged"] is True
    assert model_bytes(on) != model_bytes(off)


def test_explicit_max_delta_step_outranks_the_gate():
    X, y, w = _collapse_data()
    explicit = _fit(X, y, w, max_delta_step=0.7)
    assert explicit.delta_step_gate_ is None, "a caller-named cap is never gated"
    plain = _fit(X, y, w, max_delta_step=0.7, max_delta_step_gated=False)
    assert model_bytes(explicit) == model_bytes(plain)
    # ...and it is genuinely the uncapped-by-the-gate fit, not a coincidence.
    gated = _fit(X, y, w)
    assert model_bytes(explicit) != model_bytes(gated)


@pytest.mark.parametrize("objective", ["gamma", "poisson", "squared_error"])
def test_other_objectives_ship_no_gate(objective):
    X, y, w = _collapse_data()
    yy = np.maximum(y, 0.01).astype(np.float32) if objective == "gamma" else y
    auto = _fit(X, yy, w, objective=objective)
    off = _fit(X, yy, w, objective=objective, max_delta_step_gated=False)
    assert auto.delta_step_gate_ is None
    assert model_bytes(auto) == model_bytes(off)


def test_explicit_parameters_arm_the_gate_on_any_objective():
    X, y, w = _collapse_data()
    yy = np.maximum(y, 0.01).astype(np.float32)
    armed = _fit(X, yy, w, objective="gamma", max_delta_step_gated=(0.5, 0.3))
    assert armed.delta_step_gate_ is not None
    assert armed.delta_step_gate_["engaged"] is True


@pytest.mark.parametrize(
    "value,threshold",
    [
        ((0.02, 0.25), 0.02),
        ([0.02, 0.25], 0.02),
        ({"collapse_threshold": 0.02, "capped_step": 0.25}, 0.02),
    ],
)
def test_explicit_forms_all_parse(value, threshold):
    X, y, w = _collapse_data()
    est = _fit(X, y, w, max_delta_step_gated=value)
    assert est.delta_step_gate_["log_threshold"] == pytest.approx(np.log(threshold), abs=1e-6)
    assert est.delta_step_gate_["capped_step"] == pytest.approx(0.25, abs=1e-6)


@pytest.mark.parametrize("value", [None, True, "auto", "objective"])
def test_auto_forms_all_mean_the_objective_default(value):
    X, y, w = _collapse_data()
    est = _fit(X, y, w, max_delta_step_gated=value)
    assert est.delta_step_gate_["log_threshold"] == pytest.approx(np.log(1e-3), abs=1e-6)


@pytest.mark.parametrize("value", [False, "off", "none"])
def test_off_forms_all_disable(value):
    X, y, w = _collapse_data()
    est = _fit(X, y, w, max_delta_step_gated=value)
    assert est.delta_step_gate_ is None


@pytest.mark.parametrize(
    "bad",
    [(0.0, 0.3), (1.0, 0.3), (0.01, 0.0), (0.01, float("inf")), (0.01,), (0.01, 0.3, 0.5)],
)
def test_invalid_parameters_are_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        TBoostRegressor(objective="tweedie", max_delta_step_gated=bad).fit(
            *(_collapse_data()[:2])
        )


def test_bagged_fit_reports_per_bag_firing():
    X, y, w = _collapse_data()
    est = _fit(X, y, w, n_bags=4)
    rep = est.delta_step_gate_
    assert rep["bags_total"] == 4
    assert 0 < rep["bags_engaged"] <= 4
    assert rep["engaged"] is (rep["bags_engaged"] > 0)


def test_gate_is_deterministic_across_thread_counts():
    X, y, w = _collapse_data()
    a = _fit(X, y, w, n_jobs=1)
    b = _fit(X, y, w, n_jobs=8)
    assert a.delta_step_gate_["engaged_round"] == b.delta_step_gate_["engaged_round"]
    assert a.delta_step_gate_["min_log_rate_ratio"] == b.delta_step_gate_["min_log_rate_ratio"]
    assert model_bytes(a) == model_bytes(b)


def test_parameter_survives_get_params_and_clone():
    from sklearn.base import clone

    est = TBoostRegressor(objective="tweedie", max_delta_step_gated=(0.02, 0.25))
    assert est.get_params()["max_delta_step_gated"] == (0.02, 0.25)
    assert clone(est).get_params()["max_delta_step_gated"] == (0.02, 0.25)
