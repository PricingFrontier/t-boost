"""A single-bag fit runs every post-fit stage (2026-10-14).

One bag leaves no out-of-bag rows and no spread between bags. Pruning used to keep a multiclass
fit's full bank and skip banding on every path; now every path falls back to the fold vote, and
banding runs with the deviance cap (`band_deviance_cap`) alone as its budget.
"""

from __future__ import annotations

import numpy as np
import pytest

from t_boost import TBoostClassifier, TBoostRegressor


def _cells(model) -> int:
    return int(sum(model.deployed_census()["cells"].values()))


@pytest.fixture(scope="module")
def poisson() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 4)).astype(np.float32)
    mu = np.exp(0.4 * x[:, 0] - 0.3 * x[:, 1] + 0.3 * x[:, 0] * x[:, 1] + 0.2 * x[:, 2] * x[:, 3])
    return x, rng.poisson(mu).astype(np.float32)


def _fit(data, **kw) -> TBoostRegressor:
    x, y = data
    return TBoostRegressor(objective="poisson", n_bags=1, n_trees=200, seed=0, n_jobs=2,
                           **kw).fit(x, y)


def test_a_single_bag_fit_is_banded_within_the_deviance_cap(poisson) -> None:
    est = _fit(poisson)
    band = est.pruning_report_["banding"]
    assert band["skipped"] is None
    assert band["noise_budget"] is None, "no spread between bags: an infinite noise budget"
    assert band["budget"] == band["deviance_budget"] > 0
    assert _cells(est._model) < _cells(_fit(poisson, band_tolerance=None)._model)


def test_one_bag_and_no_deviance_cap_leave_nothing_to_bound_the_banding(poisson) -> None:
    est = _fit(poisson, band_deviance_cap=0.0)
    assert "nothing bounds the banding" in est.pruning_report_["banding"]["skipped"]


def test_a_single_bag_multiclass_fit_falls_back_to_the_fold_vote_and_is_banded() -> None:
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.normal(size=(n, 3)).astype(np.float32)
    logits = np.stack([np.zeros(n), x[:, 0] + 0.5 * x[:, 0] * x[:, 1], -x[:, 1]], axis=1)
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = (rng.uniform(size=(n, 1)) > np.cumsum(p, axis=1)).sum(axis=1)
    est = TBoostClassifier(n_bags=1, n_trees=60, seed=0, n_jobs=2).fit(x, y)
    rep = est.pruning_report_
    assert rep["selector"] == "cv_fold_vote"
    assert len(rep["banding"]) == 3
    assert all(b["noise_budget"] is None and b["skipped"] is None for b in rep["banding"])
    np.testing.assert_allclose(est.predict_proba(x).sum(axis=1), 1.0, rtol=1e-5)
