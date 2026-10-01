"""Group-honest carves: `groups` on TBoostClassifier.fit (binary K=2 AND multiclass K>=3)
and TBoostRegressor.fit.

On panel data the row-level ES/prune-select carves leak near-duplicate rows of the same
entity into validation — measured driving fits to the 4000-tree cap. `groups` assigns
WHOLE groups to one side of every internal carve: the ES holdout (all objectives) and the
prune-CV fold assignment (regressor/binary K-fold via `_carve_group_folds`; multiclass
train/select split via `_carve_group_holdout`). `groups=None` is byte-compatible with the
pre-`groups` behavior everywhere.
"""
import numpy as np
import pytest

from t_boost.sklearn import (
    TBoostClassifier,
    TBoostRegressor,
    _carve_group_folds,
    _carve_group_holdout,
)


def _panel(n_groups=400, rows_per=4, k=3, seed=7):
    """Sticky panel: each group's rows share features + label (the leak-prone regime)."""
    rng = np.random.default_rng(seed)
    gx = rng.normal(size=(n_groups, 5))
    gy = rng.integers(0, k, size=n_groups)
    X = np.repeat(gx, rows_per, axis=0) + rng.normal(scale=0.01, size=(n_groups * rows_per, 5))
    y = np.repeat(gy, rows_per)
    groups = np.repeat(np.arange(n_groups), rows_per)
    return X.astype(np.float32), y, groups


def test_carve_assigns_whole_groups_and_is_deterministic():
    _, _, groups = _panel()
    mask1 = _carve_group_holdout(groups, 0.15, [0, 1])
    mask2 = _carve_group_holdout(groups, 0.15, [0, 1])
    assert np.array_equal(mask1, mask2)
    # Whole groups on one side: within every group the mask is constant.
    for g in np.unique(groups):
        vals = np.unique(mask1[groups == g])
        assert vals.shape[0] == 1, f"group {g} straddles the carve"
    # Fraction is approximated from above at group granularity, and both sides are non-empty.
    frac = mask1.mean()
    assert 0.10 <= frac <= 0.25
    assert 0 < mask1.sum() < mask1.shape[0]


def test_carve_different_seed_keys_differ_and_single_group_raises():
    _, _, groups = _panel()
    a = _carve_group_holdout(groups, 0.15, [0, 0])
    b = _carve_group_holdout(groups, 0.15, [0, 1])
    assert not np.array_equal(a, b)
    with pytest.raises(ValueError, match="two distinct values"):
        _carve_group_holdout(np.zeros(10), 0.15, [0, 0])


def test_carve_extreme_fractions_leave_both_sides_nonempty():
    _, _, groups = _panel(n_groups=5, rows_per=3)
    tiny = _carve_group_holdout(groups, 1e-9, [1, 0])
    huge = _carve_group_holdout(groups, 1.0, [1, 0])
    assert 0 < tiny.sum() < tiny.shape[0]
    assert 0 < huge.sum() < huge.shape[0]


def test_multiclass_fit_accepts_groups_and_is_deterministic():
    X, y, groups = _panel()
    kw = dict(n_trees=60, seed=0, n_jobs=2, prune=False)
    m1 = TBoostClassifier(**kw).fit(X, y, groups=groups)
    m2 = TBoostClassifier(**kw).fit(X, y, groups=groups)
    p1, p2 = m1.predict_proba(X), m2.predict_proba(X)
    assert np.array_equal(p1, p2)
    # And the grouped carve actually changes the fit vs the row-level carve.
    m3 = TBoostClassifier(**kw).fit(X, y)
    assert not np.array_equal(p1, m3.predict_proba(X))


def test_multiclass_pruned_fit_accepts_groups():
    X, y, groups = _panel(n_groups=300, rows_per=4)
    m = TBoostClassifier(n_trees=60, seed=0, n_jobs=2, prune=True).fit(X, y, groups=groups)
    p = m.predict_proba(X)
    assert p.shape == (X.shape[0], 3)
    assert np.all(np.isfinite(p))


def test_binary_fit_accepts_groups_and_is_deterministic():
    X, y, groups = _panel(k=2)
    kw = dict(n_trees=60, seed=0, n_jobs=2, prune=False)
    m1 = TBoostClassifier(**kw).fit(X, y, groups=groups)
    m2 = TBoostClassifier(**kw).fit(X, y, groups=groups)
    p1, p2 = m1.predict_proba(X), m2.predict_proba(X)
    assert np.array_equal(p1, p2)
    # And the grouped carve actually changes the fit vs the row-level carve.
    m3 = TBoostClassifier(**kw).fit(X, y)
    assert not np.array_equal(p1, m3.predict_proba(X))


def test_binary_pruned_fit_accepts_groups():
    X, y, groups = _panel(n_groups=300, rows_per=4, k=2)
    m = TBoostClassifier(n_trees=60, seed=0, n_jobs=2, prune=True).fit(X, y, groups=groups)
    p = m.predict_proba(X)
    assert p.shape == (X.shape[0], 2)
    assert np.all(np.isfinite(p))


def test_regressor_fit_accepts_groups_and_is_deterministic():
    X, y, groups = _panel(k=2)
    y = y.astype(np.float32)
    kw = dict(n_trees=60, seed=0, n_jobs=2, prune=False)
    m1 = TBoostRegressor(**kw).fit(X, y, groups=groups)
    m2 = TBoostRegressor(**kw).fit(X, y, groups=groups)
    p1, p2 = m1.predict(X), m2.predict(X)
    assert np.array_equal(p1, p2)
    # And the grouped carve actually changes the fit vs the row-level carve.
    m3 = TBoostRegressor(**kw).fit(X, y)
    assert not np.array_equal(p1, m3.predict(X))


def test_regressor_pruned_fit_accepts_groups():
    X, y, groups = _panel(n_groups=300, rows_per=4, k=2)
    y = y.astype(np.float32)
    m = TBoostRegressor(n_trees=60, seed=0, n_jobs=2, prune=True).fit(X, y, groups=groups)
    p = m.predict(X)
    assert p.shape == (X.shape[0],)
    assert np.all(np.isfinite(p))


def test_groups_length_mismatch_raises():
    X, y, groups = _panel()
    with pytest.raises(ValueError, match="groups has"):
        TBoostClassifier(n_trees=30, seed=0, prune=False).fit(X, y, groups=groups[:-1])
    with pytest.raises(ValueError, match="groups has"):
        TBoostRegressor(n_trees=30, seed=0, prune=False).fit(
            X, y.astype(np.float32), groups=groups[:-1]
        )


def test_carve_group_folds_assigns_whole_groups_and_is_deterministic():
    _, _, groups = _panel(n_groups=300, rows_per=4)
    fold1 = _carve_group_folds(groups, 5, [0, 0])
    fold2 = _carve_group_folds(groups, 5, [0, 0])
    assert np.array_equal(fold1, fold2)
    assert set(np.unique(fold1).tolist()) == set(range(5))
    # Whole groups on one fold: within every group the fold id is constant.
    for g in np.unique(groups):
        vals = np.unique(fold1[groups == g])
        assert vals.shape[0] == 1, f"group {g} straddles a fold"
    # Reasonably balanced fold sizes (greedy bin-packing over near-equal group sizes).
    _, counts = np.unique(fold1, return_counts=True)
    assert counts.max() / counts.min() < 1.5


def test_carve_group_folds_different_seed_keys_differ_and_too_few_groups_raises():
    _, _, groups = _panel(n_groups=300, rows_per=4)
    a = _carve_group_folds(groups, 5, [0, 0])
    b = _carve_group_folds(groups, 5, [0, 1])
    assert not np.array_equal(a, b)
    with pytest.raises(ValueError, match="fewer than k_folds"):
        _carve_group_folds(np.repeat(np.arange(3), 4), 5, [0, 0])


# --- ES-runaway fix #2 (2026-07-20): y>0-stratified group carve for Poisson/Tweedie ---------


def _skewed_event_panel(n_event_groups=15, n_zero_groups=285, rows_per=4, seed=11):
    """Panel with a small minority of "event" groups (any row y>0) among a majority of
    all-zero groups — the group-carve analog of the row-level zero-dominance fixture."""
    rng = np.random.default_rng(seed)
    event_groups = np.arange(n_event_groups)
    zero_groups = np.arange(n_event_groups, n_event_groups + n_zero_groups)
    groups = np.concatenate(
        [np.repeat(event_groups, rows_per), np.repeat(zero_groups, rows_per)]
    )
    strata = np.concatenate(
        [
            np.ones(n_event_groups * rows_per, dtype=np.int64),
            np.zeros(n_zero_groups * rows_per, dtype=np.int64),
        ]
    )
    perm = rng.permutation(groups.shape[0])
    return groups[perm], strata[perm], event_groups, zero_groups


def test_carve_group_holdout_stratified_keeps_both_strata_represented():
    groups, strata, event_groups, zero_groups = _skewed_event_panel()
    # A small, unstratified 5% holdout over 300 groups (15 of them event groups) easily misses
    # the event groups entirely; the stratified carve must not.
    mask = _carve_group_holdout(groups, 0.05, [0, 1], strata=strata)
    held_groups = np.unique(groups[mask])
    assert np.isin(held_groups, event_groups).any(), "dropped ALL event groups from holdout"
    assert np.isin(held_groups, zero_groups).any(), "dropped ALL zero groups from holdout"
    # Whole groups on one side still holds under stratification.
    for g in np.unique(groups):
        vals = np.unique(mask[groups == g])
        assert vals.shape[0] == 1, f"group {g} straddles the carve"


def test_carve_group_holdout_stratified_is_deterministic_and_strata_shape_checked():
    groups, strata, _, _ = _skewed_event_panel()
    m1 = _carve_group_holdout(groups, 0.05, [0, 1], strata=strata)
    m2 = _carve_group_holdout(groups, 0.05, [0, 1], strata=strata)
    assert np.array_equal(m1, m2)
    with pytest.raises(ValueError, match="strata has"):
        _carve_group_holdout(groups, 0.05, [0, 1], strata=strata[:-1])


def test_carve_group_holdout_strata_none_matches_undocumented_default():
    # strata=None must take the EXACT original (pre-strata) code path — not a re-derivation of
    # it — so this is unchanged from test_carve_assigns_whole_groups_and_is_deterministic.
    _, _, groups = _panel()
    assert np.array_equal(
        _carve_group_holdout(groups, 0.15, [0, 1]),
        _carve_group_holdout(groups, 0.15, [0, 1], strata=None),
    )


def test_regressor_poisson_zero_dominated_fit_accepts_groups():
    # End-to-end: a grouped Poisson panel with a small minority of claim groups fits cleanly
    # with groups=, exercising the y>0-stratified deploy/fold ES holdout through _fit_and_prune.
    groups, strata, _, _ = _skewed_event_panel(n_event_groups=15, n_zero_groups=85, rows_per=4)
    rng = np.random.default_rng(5)
    n = groups.shape[0]
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = np.where(strata > 0, rng.poisson(3.0, size=n) + 1, 0).astype(np.float32)
    kw = dict(n_trees=60, seed=0, n_jobs=2, prune=True, objective="poisson")
    m1 = TBoostRegressor(**kw).fit(X, y, groups=groups)
    m2 = TBoostRegressor(**kw).fit(X, y, groups=groups)
    assert np.array_equal(m1.predict(X), m2.predict(X))
    assert np.all(np.isfinite(m1.predict(X)))
