"""Multiclass (K>=3) silent-no-op guards: the native softmax path never applies leaf
refinement, the fully-corrective ridge leaf refit, DART, intercept/slope re-anchoring, or the
scale-invariant lambda rescale (see `TBoostClassifier`'s "Honesty notes" and
`engine::boost::fit_multiclass`'s doc). SIX params whose own no-op value is their constructor
default raise ValueError when explicitly set for a K>=3 fit; `leaf_refine_steps` ships a
non-inert recipe default, so it warns once per estimator instance instead of raising.

Params that have LEFT this family (each is now live for K>=3, and each has its own test file):
`n_bags`/`bag_subsample` since 2026-07-14 (outer bags soup per class — covered by the
bagging-is-honored test below), `subsample`/`mvs_min_rows` since 2026-08-23 (§06.5 MVS row
sampling, one joint-norm row draw per round shared by the K class trees — see
test_multiclass_mvs.py), and `cell_refit_base`/`cell_refit_gamma` since 2026-08-25 (the §G1 OOB
cell refit as K decoupled diagonal-Hessian solves with ONE joint backtrack on the multinomial
loss — see test_multiclass_cell_refit.py)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from t_boost.sklearn import TBoostClassifier, TBoostRegressor


def multiclass_fixture(n: int = 240, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.random(size=(n, 3)).astype(np.float32)
    y = np.where(x[:, 0] < 0.33, "a", np.where(x[:, 0] < 0.66, "b", "c"))
    return x, y


def binary_fixture(n: int = 120, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.random(size=(n, 3)).astype(np.float32)
    y = (x[:, 0] < 0.5).astype(int)
    return x, y


# (constructor kwarg, a value guaranteed to differ from its default) for the RAISE-family
# params -- each independently no-ops the multiclass path per the class docstring / the Rust
# `fit_multiclass` doc comment.
RAISE_PARAMS: list[tuple[str, object]] = [
    ("ridge_refit_l2", 0.1),
    ("ridge_refit_max_iter", 10),
    ("dart_drop_rate", 0.1),
    ("reanchor", True),
    ("reanchor_slope", True),
    ("lambda_scale_invariant", True),
]


@pytest.mark.parametrize("param, value", RAISE_PARAMS)
def test_raises_on_explicit_opt_in(param: str, value: object) -> None:
    x, y = multiclass_fixture()
    clf = TBoostClassifier(n_trees=5, n_bags=1, leaf_refine_steps=0, **{param: value})
    with pytest.raises(ValueError, match=f"{param} has no effect on multiclass"):
        clf.fit(x, y)


def test_raise_family_defaults_do_not_raise() -> None:
    # None of the raise-family params fire at their own shipped defaults -- only n_bags/
    # leaf_refine_steps are non-inert by default, silenced here so this isolates the raise
    # family specifically.
    x, y = multiclass_fixture()
    TBoostClassifier(n_trees=5, n_bags=1, leaf_refine_steps=0).fit(x, y)


def test_raises_on_pruned_multiclass_path_too() -> None:
    # The guard is called once before branching into the pruned/unpruned _fit_multiclass arms;
    # confirm the pruned arm is equally covered (not just the plain fit path above).
    x, y = multiclass_fixture()
    clf = TBoostClassifier(
        n_trees=5, n_bags=1, leaf_refine_steps=0, prune=True, dart_drop_rate=0.1
    )
    with pytest.raises(ValueError, match="dart_drop_rate has no effect on multiclass"):
        clf.fit(x, y)


def test_bagging_is_honored_for_multiclass_no_warning_and_bags_change_the_fit() -> None:
    # n_bags is LIVE for K>=3 (2026-07-14): outer bags soup per class, so a bagged default
    # must fit silently, and the bagged model must differ from the single-bag fit.
    x, y = multiclass_fixture()
    bagged = TBoostClassifier(n_trees=5, n_bags=4, leaf_refine_steps=0, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bagged.fit(x, y)
    single = TBoostClassifier(n_trees=5, n_bags=1, leaf_refine_steps=0, seed=0).fit(x, y)
    assert not np.array_equal(bagged.predict_proba(x), single.predict_proba(x))
    # Determinism: same config + seed reproduces the bagged fit exactly.
    again = TBoostClassifier(n_trees=5, n_bags=4, leaf_refine_steps=0, seed=0).fit(x, y)
    assert np.array_equal(bagged.predict_proba(x), again.predict_proba(x))


def test_multiclass_leaf_refine_is_inert() -> None:
    # K>=3 has no leaf refinement: a default fit must be byte-identical to leaf_refine_steps=0.
    x, y = multiclass_fixture()
    default = TBoostClassifier(n_trees=5, n_bags=1, seed=0).fit(x, y)
    plain = TBoostClassifier(n_trees=5, n_bags=1, leaf_refine_steps=0, seed=0).fit(x, y)
    assert np.array_equal(default.predict_proba(x), plain.predict_proba(x))


def test_n_bags_one_silences_the_bagging_warning() -> None:
    # n_bags=1 is this library's documented "disable bagging" value.
    x, y = multiclass_fixture()
    clf = TBoostClassifier(n_trees=5, n_bags=1, leaf_refine_steps=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clf.fit(x, y)


def test_leaf_refine_steps_zero_silences_the_leaf_refine_warning() -> None:
    x, y = multiclass_fixture()
    clf = TBoostClassifier(n_trees=5, n_bags=1, leaf_refine_steps=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clf.fit(x, y)


def test_default_multiclass_fit_emits_no_inert_param_warnings() -> None:
    # n_bags=8 (outer bags souped per class) AND leaf_refine_steps=4 (per-round Jacobi
    # re-linearization) are both HONORED for K>=3 as of 2026-07-14 — a completely unmodified
    # default fit must be warning-silent on those params.
    x, y = multiclass_fixture()
    clf = TBoostClassifier(n_trees=5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf.fit(x, y)
    messages = [str(w.message) for w in caught]
    assert not any("have no effect on multiclass" in m for m in messages)


@pytest.mark.parametrize("param, value", RAISE_PARAMS)
def test_binary_classifier_is_unaffected(param: str, value: object) -> None:
    # The guard lives inside _fit_multiclass only -- a K=2 fit must never see it, even with the
    # exact same non-default values that raise for K>=3.
    x, y = binary_fixture()
    TBoostClassifier(n_trees=5, **{param: value}).fit(x, y)


def test_binary_classifier_does_not_warn_for_bagging_or_leaf_refine() -> None:
    x, y = binary_fixture()
    clf = TBoostClassifier(n_trees=5)  # recipe defaults: n_bags=8, leaf_refine_steps=4
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clf.fit(x, y)


@pytest.mark.parametrize("param, value", RAISE_PARAMS)
def test_regressor_is_unaffected(param: str, value: object) -> None:
    # TBoostRegressor never reaches _fit_multiclass at all (regression has no K>=3 concept).
    rng = np.random.default_rng(0)
    x = rng.random(size=(80, 3)).astype(np.float32)
    y = (2.0 * x[:, 0] - x[:, 1]).astype(np.float32)
    TBoostRegressor(n_trees=5, **{param: value}).fit(x, y)
