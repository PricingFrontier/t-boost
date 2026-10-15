"""A multiclass fit reports its boosting, as a regression or binary fit does (2026-10-15).

It used to record nothing: `n_trees_per_bag_` and its siblings were None, and the model report
said there was no per-bag boosting report. A fit cut short by the `n_trees` cap then looked like
any other, so nobody could tell that raising `n_trees` would help. On a multiclass fit a boosting
round grows one tree per class, so the counts are rounds, which `n_trees` caps.
"""

from __future__ import annotations

import numpy as np
import pytest

from t_boost import TBoostClassifier

STOP_REASONS = {"early_stopping", "max_trees", "no_split", "callback"}


@pytest.fixture(scope="module")
def three_class() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    n = 3000
    x = rng.normal(size=(n, 3)).astype(np.float32)
    logits = np.stack([np.zeros(n), 1.2 * x[:, 0] + 0.5 * x[:, 0] * x[:, 1], -x[:, 1]], axis=1)
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = (rng.uniform(size=(n, 1)) > np.cumsum(p, axis=1)).sum(axis=1)
    return x, y


def test_a_capped_multiclass_fit_says_so_and_the_report_warns(three_class) -> None:
    x, y = three_class
    est = TBoostClassifier(n_trees=5, n_bags=3, seed=0, n_jobs=2).fit(x, y)
    assert est.n_trees_per_bag_ == [5, 5, 5]
    assert est.stopping_reason_per_bag_ == ["max_trees"] * 3
    assert est.n_trees_ == 5 and est.stopping_reason_ == "max_trees"
    flag = next(f for f in est.pricing_report(x, y)["review"] if f["check"] == "early_stopping")
    assert flag["status"] == "warning"
    assert "3 of 3 bag(s) reached n_trees" in flag["message"]


def test_an_early_stopped_multiclass_fit_reports_its_rounds(three_class) -> None:
    x, y = three_class
    est = TBoostClassifier(n_trees=3000, n_bags=2, learning_rate=0.3, seed=0, n_jobs=2).fit(x, y)
    assert len(est.n_trees_per_bag_) == 2
    assert set(est.stopping_reason_per_bag_) <= STOP_REASONS
    assert "early_stopping" in est.stopping_reason_per_bag_
    assert all(0 < n < 3000 for n in est.n_trees_per_bag_)
    for loaded in (TBoostClassifier.from_bytes(est.to_bytes()),
                   TBoostClassifier.from_json(est.to_json())):
        assert loaded.n_trees_per_bag_ == est.n_trees_per_bag_
        assert loaded.stopping_reason_per_bag_ == est.stopping_reason_per_bag_


@pytest.mark.parametrize("options", [{}, {"prune": False}, {"n_bags": 1}],
                         ids=["pruned", "unpruned", "single-bag"])
def test_a_multiclass_fit_keeps_each_bags_early_stopping_curve(three_class, options) -> None:
    # The stopping rule computes the early-stopping deviance every round anyway, so every fit
    # keeps it (as the multinomial deviance); the model report draws it. Not saved.
    x, y = three_class
    est = TBoostClassifier(**{"n_trees": 3000, "n_bags": 2, "learning_rate": 0.3, "seed": 0,
                              "n_jobs": 2, "early_stopping_min_delta": 0.0, **options}).fit(x, y)
    assert set(est.evals_result_) == {"eval"}
    curves = est.evals_result_["eval"]["deviance"]
    assert len(curves) == len(est.n_trees_per_bag_)
    for kept, curve in zip(est.n_trees_per_bag_, curves):
        assert np.all(np.isfinite(curve)) and min(curve) > 0
        assert kept == int(np.argmin(curve)) + 1
    assert est.pricing_report(x, y)["fit"]["evals_result"]["available"]
    assert not hasattr(TBoostClassifier.from_bytes(est.to_bytes()), "evals_result_")


def test_an_unpruned_or_single_bag_multiclass_fit_reports_too(three_class) -> None:
    x, y = three_class
    unpruned = TBoostClassifier(n_trees=5, n_bags=2, prune=False, seed=0, n_jobs=2).fit(x, y)
    assert unpruned.n_trees_per_bag_ == [5, 5]
    assert unpruned.stopping_reason_per_bag_ == ["max_trees", "max_trees"]
    single = TBoostClassifier(n_trees=5, n_bags=1, seed=0, n_jobs=2).fit(x, y)
    assert single.n_trees_per_bag_ == [5] and single.stopping_reason_ == "max_trees"
