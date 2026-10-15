"""The post-prune rebalance runs on every pruned fit and says what it did (2026-10-14).

It used to return the pruned bank untouched, without a word, whenever the fit carried a cell
correction or any categorical was encoded by more than one channel. Now it re-solves every kept
main and pair it can address, leaves the tables on a multi-channel categorical as fitted, and
records the outcome in `pruning_report_["rebalance"]`.

Tables on a categorical binned by cross-fit target statistics are left as fitted too: after the
fit, the training rows are binned by full-data statistics that contain their own targets.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from t_boost import TBoostClassifier, TBoostRegressor

KEYS = {"adopted", "step", "supports", "excluded", "holdout_rows", "holdout_deviance_before",
        "holdout_deviance_after"}


@pytest.fixture(scope="module")
def data() -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 6000
    a, b = rng.normal(size=n), rng.normal(size=n)
    levels = [f"L{i}" for i in range(40)]
    effect = dict(zip(levels, rng.normal(scale=0.4, size=len(levels))))
    c = rng.choice(levels, n)
    eta = 0.4 * a - 0.3 * b + np.array([effect[v] for v in c]) + 0.3 * a * b
    y = rng.poisson(np.exp(eta - 1.0)).astype(float)
    return pl.DataFrame({"a": a, "b": b, "c": c}), y


def _fit(data, **kw) -> TBoostRegressor:
    X, y = data
    return TBoostRegressor(objective="poisson", n_trees=200, seed=0, n_jobs=2, **kw).fit(X, y)


def test_a_pruned_fit_reports_its_rebalance(data) -> None:
    rep = _fit(data).pruning_report_["rebalance"]
    assert set(rep) == KEYS
    assert rep["supports"] > 0
    assert rep["holdout_rows"] > 0
    assert _fit(data, prune_rebalance=False).pruning_report_.get("rebalance") is None


def test_tables_on_a_target_encoded_categorical_are_left_as_fitted(data) -> None:
    # "c" (raw 2) has 40 levels: more than `cat_direct_max_levels`, so it is binned by
    # cross-fit target statistics and its tables keep their values. Let it bin each level on
    # its own and they are re-solved like any other.
    est = _fit(data)
    rep = est.pruning_report_["rebalance"]
    kept = [u for u in est.pruning_report_["kept"] if len(u) <= 2]
    assert rep["excluded"] == [u for u in kept if 2 in u] != []
    assert rep["supports"] == len(kept) - len(rep["excluded"]) > 0
    direct = _fit(data, cat_direct_max_levels=64)
    rep = direct.pruning_report_["rebalance"]
    kept = [u for u in direct.pruning_report_["kept"] if len(u) <= 2]
    assert rep["excluded"] == [] and rep["supports"] == len(kept)


def test_tables_on_a_multichannel_categorical_are_left_as_fitted(data) -> None:
    # 40 levels clears `cat_count_min_levels`, so "c" (raw 2) gets a count channel too; its
    # mean channel bins each level on its own, so the channels are the only reason.
    est = _fit(data, cat_channels=["mean", "count"], cat_direct_max_levels=64)
    rep = est.pruning_report_["rebalance"]
    kept = [u for u in est.pruning_report_["kept"] if len(u) <= 2]
    assert rep["excluded"] == [u for u in kept if 2 in u] != []
    assert rep["supports"] == len(kept) - len(rep["excluded"]) > 0


@pytest.fixture(scope="module")
def three_class() -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(5)
    n = 6000
    x = rng.normal(size=(n, 3))
    c = rng.choice([f"L{i}" for i in range(12)], n)
    logits = np.stack([np.zeros(n), 1.5 * np.tanh(x[:, 0]) + x[:, 0] * x[:, 1],
                       -x[:, 1] + 0.8 * x[:, 2] * (c < "L6")], axis=1)
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = (rng.uniform(size=(n, 1)) > np.cumsum(p, axis=1)).sum(axis=1)
    return pl.DataFrame({"a": x[:, 0], "b": x[:, 1], "d": x[:, 2], "c": c}), y


def _fit_mc(data, **kw) -> TBoostClassifier:
    X, y = data
    # Few, large trees underfit: the dropped tables then leave a residual the re-solve recovers.
    return TBoostClassifier(n_trees=8, learning_rate=0.3, validation_fraction=None, seed=0,
                            n_jobs=2, **kw).fit(X, y)


def test_a_multiclass_fit_rebalances_with_one_joint_step(three_class) -> None:
    est = _fit_mc(three_class)
    rep = est.pruning_report_["rebalance"]
    assert set(rep) == KEYS
    assert rep["adopted"] and 0 < rep["step"] <= 1
    assert rep["holdout_deviance_after"] < rep["holdout_deviance_before"]
    off = _fit_mc(three_class, prune_rebalance=False)
    assert off.pruning_report_.get("rebalance") is None
    X, _ = three_class
    assert not np.array_equal(est.predict_proba(X), off.predict_proba(X))
    np.testing.assert_allclose(est.predict_proba(X).sum(axis=1), 1.0, rtol=1e-5)


def test_per_class_channels_leave_the_categorical_as_fitted(three_class) -> None:
    # `class_freq` gives "c" (raw 3) one channel per class, so its tables have no single axis.
    est = _fit_mc(three_class, cat_channels=["mean", "class_freq"])
    rep = est.pruning_report_["rebalance"]
    kept = [u for u in est.pruning_report_["kept"] if len(u) <= 2]
    assert sorted(rep["excluded"]) == sorted(u for u in kept if 3 in u) != []
    assert rep["supports"] == len(kept) - len(rep["excluded"])
