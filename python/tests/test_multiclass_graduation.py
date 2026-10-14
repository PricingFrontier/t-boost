"""Multiclass (K>=3) fits graduate every class bank (2026-10-14).

Graduation used to raise for multiclass. Now each class's dense tables are smoothed by the same
per-table solves as a single-output fit's, the class intercepts are re-anchored together, and the
candidate is kept only if the training multinomial deviance rises by at most 1%.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from t_boost import TBoostClassifier


@pytest.fixture(scope="module")
def data() -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(11)
    n = 5000
    a, b = rng.normal(size=n), rng.uniform(-2, 2, size=n)
    c = rng.choice(list("pqrst"), n)
    logits = np.stack([np.zeros(n), np.sin(2 * a) + 0.4 * b,
                       0.5 * b ** 2 - 1 + 0.6 * (c == "p")], axis=1)
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = (rng.uniform(size=(n, 1)) > np.cumsum(p, axis=1)).sum(axis=1)
    return pl.DataFrame({"a": a, "b": b, "c": c}), y


def _fit(data, **kw) -> TBoostClassifier:
    X, y = data
    return TBoostClassifier(n_trees=120, seed=0, n_jobs=2, **kw).fit(X, y)


def _ce(est: TBoostClassifier, data) -> float:
    X, y = data
    p = est.predict_proba(X)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


@pytest.fixture(scope="module")
def graduated(data) -> TBoostClassifier:
    return _fit(data, graduation_alpha=30.0)


def test_every_class_bank_is_graduated_and_checked(graduated, data) -> None:
    gv = graduated.graduation_validation_
    assert gv["candidate_count"] == 1 and gv["adopted"] is True
    assert graduated.pruning_report_["graduation_validation"] is gv
    smoothed = [r for r in graduated.graduation_report_ if r["alpha"] > 0]
    assert {r["class"] for r in smoothed} == {0, 1, 2}
    raw = _fit(data, graduate=False)
    assert not hasattr(raw, "graduation_report_")
    assert raw.graduation_validation_["skipped"] == "disabled"
    # the training-deviance check: kept only within 1% of the tables as fitted
    assert _ce(graduated, data) <= 1.01 * _ce(raw, data)
    X, _ = data
    assert not np.array_equal(graduated.predict_proba(X), raw.predict_proba(X))


def test_the_graduated_model_is_a_valid_round_tripping_simplex(graduated, data) -> None:
    X, _ = data
    proba = graduated.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    restored = TBoostClassifier.from_bytes(graduated.to_bytes())
    np.testing.assert_array_equal(restored.predict_proba(X), proba)


def test_the_re_anchor_reproduces_the_class_masses(graduated, data) -> None:
    X, y = data
    observed = np.bincount(y, minlength=3) / len(y)
    np.testing.assert_allclose(graduated.predict_proba(X).mean(axis=0), observed, atol=1e-4)


def test_a_strength_the_check_declines_keeps_the_tables_as_fitted(data) -> None:
    X, _ = data
    with pytest.warns(RuntimeWarning, match="graduation_alpha"):
        declined = _fit(data, graduation_alpha=1e9)
    assert declined.graduation_validation_["adopted"] is False
    assert all(r["rejected"] for r in declined.graduation_report_)
    raw = _fit(data, graduate=False)
    np.testing.assert_array_equal(declined.predict_proba(X), raw.predict_proba(X))


def test_the_html_report_names_each_tables_class(graduated, data) -> None:
    X, y = data
    labelled = X.with_columns(pl.Series("y", y))
    html = graduated.html_report(labelled, "y", name="multiclass graduation")
    assert "<th>class</th>" in html or ">class<" in html
