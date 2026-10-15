"""The post-prune rebalance runs on every pruned fit and says what it did (2026-10-14).

It used to return the pruned bank untouched, without a word, whenever the fit carried a cell
correction or any categorical was encoded by more than one channel. Now it re-solves every kept
main and pair it can address, leaves the tables on a multi-channel categorical as fitted, and
records the outcome in `pruning_report_["rebalance"]`.

The bags judge the re-solve on the rows each left out; a single bag falls back to a held-out slice
of the training rows (2026-10-15). A categorical whose levels share bins places its training rows
as the trees saw them (their out-of-fold statistics), so its tables are re-solved like any other.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from t_boost import TBoostClassifier, TBoostRegressor

KEYS = {"adopted", "step", "supports", "excluded", "evidence", "holdout_rows",
        "holdout_deviance_before", "holdout_deviance_after"}


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
    params = {"objective": "poisson", "n_trees": 200, "seed": 0, "n_jobs": 2, **kw}
    return TBoostRegressor(**params).fit(X, y)


def test_a_pruned_fit_reports_its_rebalance(data) -> None:
    est = _fit(data)
    rep = est.pruning_report_["rebalance"]
    assert set(rep) == KEYS
    # "c" (raw 2) is target-encoded (40 levels), and its tables are re-solved with the rest.
    kept = [u for u in est.pruning_report_["kept"] if len(u) <= 2]
    assert rep["excluded"] == [] and rep["supports"] == len(kept) > 0
    assert any(2 in u for u in kept)
    # The bags judge the re-solve: a row counts once for every bag that left it out.
    assert rep["evidence"] == "out_of_bag" and rep["holdout_rows"] > len(data[1]) // 2
    assert rep["adopted"] == (rep["holdout_deviance_after"] <= rep["holdout_deviance_before"])
    assert _fit(data, prune_rebalance=False).pruning_report_.get("rebalance") is None


def test_a_single_bag_is_judged_on_a_held_out_slice(data) -> None:
    rep = _fit(data, n_bags=1).pruning_report_["rebalance"]
    assert set(rep) == KEYS
    # Without bags no row is held out of the trees: about 15% of the rows judge it.
    assert rep["evidence"] == "in_sample" and 0 < rep["holdout_rows"] < len(data[1]) // 4


def test_tables_on_a_multichannel_categorical_are_left_as_fitted(data) -> None:
    # 40 levels clears `cat_count_min_levels`, so "c" (raw 2) gets a count channel too.
    est = _fit(data, cat_channels=["mean", "count"])
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
    params = {"n_trees": 8, "learning_rate": 0.3, "validation_fraction": None, "seed": 0,
              "n_jobs": 2, **kw}
    return TBoostClassifier(**params).fit(X, y)


def test_a_multiclass_fit_rebalances_with_one_joint_step(three_class) -> None:
    est = _fit_mc(three_class)
    rep = est.pruning_report_["rebalance"]
    assert set(rep) == KEYS
    assert rep["evidence"] == "out_of_bag"
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


def test_the_multiclass_step_is_judged_on_the_model_it_deploys(three_class) -> None:
    # The deploy re-anchors the class intercepts after the correction; each trial step is now
    # re-anchored the same way before it is scored, so with one bag the reported held-out
    # deviance is the deployed model's own on the held-out slice (banding and graduation off,
    # so nothing else moves it).
    X, y = three_class
    est = _fit_mc(three_class, n_bags=1, band_tolerance=None, graduate=False)
    rep = est.pruning_report_["rebalance"]
    assert rep["evidence"] == "in_sample" and rep["adopted"]
    r = np.arange(len(y), dtype=np.uint64)
    held = ((r * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(56)) < 38  # the core's slice
    assert int(held.sum()) == rep["holdout_rows"]
    p = est.predict_proba(X)[held, y[held]]
    np.testing.assert_allclose(-np.log(p).mean(), rep["holdout_deviance_after"], rtol=1e-5)


def test_the_bags_judge_alike_at_any_thread_count(data, three_class) -> None:
    # The bags are judged in parallel waves and summed in bag order: byte-identical models.
    from _artifact import model_bytes

    for fit, d in ((_fit, data), (_fit_mc, three_class)):
        one, many = fit(d, n_jobs=1), fit(d, n_jobs=4)
        assert one.pruning_report_["rebalance"] == many.pruning_report_["rebalance"]
        assert model_bytes(one) == model_bytes(many)


def test_a_categorical_whose_levels_share_bins_is_re_solved_as_the_trees_saw_it() -> None:
    # 400 levels outnumber the 254 bins, so levels share bins and the serve map would place a
    # training row partly by its own target. The selection evidence and the rebalance score
    # the fit rows at their out-of-fold placements instead, and its tables are re-solved.
    rng = np.random.default_rng(3)
    n = 8000
    levels = [f"L{i}" for i in range(400)]
    effect = dict(zip(levels, rng.normal(scale=0.5, size=len(levels))))
    a, c = rng.normal(size=n), rng.choice(levels, n)
    y = rng.poisson(np.exp(0.4 * a + np.array([effect[v] for v in c]) - 1.0)).astype(float)
    X = pl.DataFrame({"a": a, "c": c})
    est = TBoostRegressor(objective="poisson", n_trees=150, seed=0, n_jobs=2).fit(X, y)
    rep = est.pruning_report_["rebalance"]
    assert rep["evidence"] == "out_of_bag" and rep["excluded"] == []
    assert any(1 in u for u in est.pruning_report_["kept"])
    assert np.isfinite(est.predict(X)).all()
