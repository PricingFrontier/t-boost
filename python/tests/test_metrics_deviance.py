"""Deviance metrics: numpy-only, numerically identical to scikit-learn's definitions."""

from __future__ import annotations

import numpy as np
import pytest

from t_boost.metrics import (
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_tweedie_deviance,
)


def fixtures() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.poisson(0.3, n).astype(np.float64)  # includes zeros (the xlogy edge)
    pred = rng.uniform(0.05, 2.0, n)
    weight = rng.uniform(0.1, 1.0, n)
    return y, pred, weight


def test_matches_sklearn_exactly() -> None:
    skm = pytest.importorskip("sklearn.metrics")
    y, pred, weight = fixtures()
    assert mean_poisson_deviance(y, pred, weight) == skm.mean_poisson_deviance(
        y, pred, sample_weight=weight
    )
    assert mean_poisson_deviance(y, pred) == skm.mean_poisson_deviance(y, pred)
    assert mean_gamma_deviance(y + 0.5, pred, weight) == skm.mean_gamma_deviance(
        y + 0.5, pred, sample_weight=weight
    )
    for power in (0.0, 1.5, 3.0):
        y_p = y + 0.5 if power > 2.0 else y
        assert mean_tweedie_deviance(
            y_p, pred, weight, power=power
        ) == skm.mean_tweedie_deviance(y_p, pred, sample_weight=weight, power=power)


def test_zero_y_rows_contribute_no_log_term() -> None:
    # unit deviance at y=0 is exactly 2*pred (the xlogy(0, .) = 0 convention)
    assert mean_poisson_deviance([0.0], [0.7]) == pytest.approx(1.4)


def test_weight_semantics() -> None:
    y, pred, _ = fixtures()
    # weight of 1 everywhere == unweighted; doubling all weights changes nothing
    w1 = np.ones_like(y)
    assert mean_poisson_deviance(y, pred, w1) == mean_poisson_deviance(y, pred)
    assert mean_poisson_deviance(y, pred, 2.0 * w1) == pytest.approx(
        mean_poisson_deviance(y, pred)
    )


def test_domain_violations_raise() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        mean_poisson_deviance([1.0], [0.0])
    with pytest.raises(ValueError, match="non-negative"):
        mean_poisson_deviance([-1.0], [1.0])
    with pytest.raises(ValueError, match="strictly positive"):
        mean_gamma_deviance([0.0], [1.0])
    with pytest.raises(ValueError, match="not supported"):
        mean_tweedie_deviance([1.0], [1.0], power=0.5)
    with pytest.raises(ValueError, match="rows"):
        mean_poisson_deviance([1.0, 2.0], [1.0])
