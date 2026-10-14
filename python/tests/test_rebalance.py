"""The post-prune rebalance runs on every pruned fit and says what it did (2026-10-14).

It used to return the pruned bank untouched, without a word, whenever the fit carried a cell
correction or any categorical was encoded by more than one channel. Now it re-solves every kept
main and pair it can address, leaves the tables on a multi-channel categorical as fitted, and
records the outcome in `pruning_report_["rebalance"]`.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from t_boost import TBoostRegressor

KEYS = {"adopted", "supports", "excluded", "holdout_rows", "holdout_deviance_before",
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
    assert rep["supports"] > 0 and rep["excluded"] == []
    assert rep["holdout_rows"] > 0
    assert _fit(data, prune_rebalance=False).pruning_report_.get("rebalance") is None


def test_tables_on_a_multichannel_categorical_are_left_as_fitted(data) -> None:
    # 40 levels clears `cat_count_min_levels`, so "c" (raw 2) gets a count channel too.
    est = _fit(data, cat_channels=["mean", "count"])
    rep = est.pruning_report_["rebalance"]
    kept = [u for u in est.pruning_report_["kept"] if len(u) <= 2]
    assert rep["excluded"] == [u for u in kept if 2 in u] != []
    assert rep["supports"] == len(kept) - len(rep["excluded"]) > 0
