"""Degenerate-grouping routing (av35): all-singleton `groups` IS `groups=None`.

Group-aware carves exist to stop one ENTITY from straddling a carve boundary. At max group
size 1 an entity is a row, so the guarantee is vacuous — but the grouped path still charges
for it, carving a shared ~`validation_fraction` of rows out of every bag's training data for
the ES holdout an ungrouped fit gets free from each bag's own internal slice. `_fit_model` /
`_fit_multiclass` therefore normalize such a grouping to None at ingest.

The claim these tests pin is an IDENTITY, not an improvement: a fit given all-singleton groups
must be byte-identical to the same fit given no groups at all, on every path. The converse
matters just as much — a genuine panel (some group with >1 row) must still take the grouped
path, so the tests below also assert the panel fit stays DIFFERENT from its ungrouped twin.
"""
import numpy as np
import pytest

from t_boost.sklearn import (
    TBoostClassifier,
    TBoostRegressor,
    _is_degenerate_grouping,
)


def _rows(n=600, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    rate = np.exp(0.3 * X[:, 0] - 0.2 * X[:, 1])
    y = rng.poisson(rate).astype(np.float64)
    return X, y


def _panel(n_groups=150, rows_per=4, seed=5):
    """Sticky panel: rows of a group share features + label (the leak-prone regime)."""
    rng = np.random.default_rng(seed)
    gx = rng.normal(size=(n_groups, 4))
    gy = rng.poisson(np.exp(0.4 * gx[:, 0]))
    X = np.repeat(gx, rows_per, axis=0) + rng.normal(scale=0.01, size=(n_groups * rows_per, 4))
    y = np.repeat(gy, rows_per).astype(np.float64)
    groups = np.repeat(np.arange(n_groups), rows_per)
    return X.astype(np.float32), y, groups


# --------------------------------------------------------------------- predicate


def test_predicate_singleton_vs_panel():
    assert _is_degenerate_grouping(np.arange(10))
    assert _is_degenerate_grouping(np.array([f"id{i}" for i in range(10)], dtype=object))
    assert _is_degenerate_grouping(np.arange(10, dtype=np.float64))
    assert not _is_degenerate_grouping(np.repeat(np.arange(5), 2))
    # one duplicated id in an otherwise unique column is a real (if tiny) panel
    assert not _is_degenerate_grouping(np.array([0, 1, 2, 3, 3]))


def test_predicate_nan_ids_are_not_singletons():
    """A NaN id is not an identity — a column of them must not read as a column of singletons
    (the prune guard's OOB-honesty test has always taken this line; both now share it)."""
    g = np.arange(10, dtype=np.float64)
    g[3] = np.nan
    assert not _is_degenerate_grouping(g)
    assert not _is_degenerate_grouping(np.full(10, np.nan))


def test_predicate_empty_is_not_degenerate():
    assert not _is_degenerate_grouping(np.array([], dtype=np.int64))


# ------------------------------------------------------- THE identity, every path


@pytest.mark.parametrize("prune", [True, False])
def test_singleton_groups_identical_to_no_groups_regressor(prune):
    X, y = _rows()
    singleton = np.arange(X.shape[0])
    a = TBoostRegressor(objective="poisson", n_trees=60, n_bags=2, seed=11, prune=prune)
    b = TBoostRegressor(objective="poisson", n_trees=60, n_bags=2, seed=11, prune=prune)
    a.fit(X, y, groups=singleton)
    b.fit(X, y)
    assert a.to_json() == b.to_json()
    assert np.array_equal(a.predict(X), b.predict(X))


def test_singleton_groups_identical_to_no_groups_shuffled_ids():
    """Identity is over the GROUPING, not the id values: any injective labelling routes the
    same way (a string per-row policy id is the arena's real case)."""
    X, y = _rows()
    ids = np.array([f"pol-{i:05d}" for i in np.random.default_rng(0).permutation(X.shape[0])],
                   dtype=object)
    a = TBoostRegressor(objective="poisson", n_trees=60, n_bags=2, seed=11)
    b = TBoostRegressor(objective="poisson", n_trees=60, n_bags=2, seed=11)
    a.fit(X, y, groups=ids)
    b.fit(X, y)
    assert a.to_json() == b.to_json()


def test_singleton_groups_identical_to_no_groups_binary():
    X, _ = _rows()
    y = (X[:, 0] + np.random.default_rng(1).normal(scale=0.5, size=X.shape[0]) > 0).astype(int)
    singleton = np.arange(X.shape[0])
    a = TBoostClassifier(n_trees=60, n_bags=2, seed=11)
    b = TBoostClassifier(n_trees=60, n_bags=2, seed=11)
    a.fit(X, y, groups=singleton)
    b.fit(X, y)
    assert a.to_json() == b.to_json()


def test_singleton_groups_identical_to_no_groups_multiclass():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(600, 4)).astype(np.float32)
    y = rng.integers(0, 3, size=600)
    singleton = np.arange(X.shape[0])
    a = TBoostClassifier(n_trees=60, n_bags=2, seed=11)
    b = TBoostClassifier(n_trees=60, n_bags=2, seed=11)
    a.fit(X, y, groups=singleton)
    b.fit(X, y)
    assert a.to_json() == b.to_json()


def test_singleton_groups_recover_the_es_carve_rows():
    """The mechanism, not just the artifact: the grouped path holds ~`validation_fraction` of
    rows out of the deploy fit's training data; the routed fit trains on all of them, which is
    what makes it identical to the ungrouped fit rather than merely close to it."""
    X, y = _rows()
    est = TBoostRegressor(objective="poisson", n_trees=60, n_bags=2, seed=11,
                            validation_fraction=0.1)
    est.fit(X, y, groups=np.arange(X.shape[0]))
    ungrouped = TBoostRegressor(objective="poisson", n_trees=60, n_bags=2, seed=11,
                                  validation_fraction=0.1)
    ungrouped.fit(X, y)
    assert est.to_json() == ungrouped.to_json()


# ------------------------------------------------------- the converse: panels untouched


@pytest.mark.parametrize("prune", [True, False])
def test_genuine_panel_still_takes_the_grouped_path(prune):
    X, y, groups = _panel()
    grouped = TBoostRegressor(objective="poisson", n_trees=60, n_bags=2, seed=11, prune=prune)
    plain = TBoostRegressor(objective="poisson", n_trees=60, n_bags=2, seed=11, prune=prune)
    grouped.fit(X, y, groups=groups)
    plain.fit(X, y)
    assert not _is_degenerate_grouping(groups)
    assert grouped.to_json() != plain.to_json()


def test_genuine_panel_multiclass_still_grouped():
    rng = np.random.default_rng(9)
    gx = rng.normal(size=(150, 4))
    gy = rng.integers(0, 3, size=150)
    X = np.repeat(gx, 4, axis=0).astype(np.float32)
    y = np.repeat(gy, 4)
    groups = np.repeat(np.arange(150), 4)
    grouped = TBoostClassifier(n_trees=60, n_bags=2, seed=11)
    plain = TBoostClassifier(n_trees=60, n_bags=2, seed=11)
    grouped.fit(X, y, groups=groups)
    plain.fit(X, y)
    assert grouped.to_json() != plain.to_json()


def test_mismatched_length_still_raises_for_singleton_ids():
    """Normalization runs AFTER validation: a degenerate grouping of the wrong length is still
    a caller error, never silently dropped."""
    X, y = _rows(n=100)
    est = TBoostRegressor(objective="poisson", n_trees=20, n_bags=1, seed=0)
    with pytest.raises(ValueError, match="groups has"):
        est.fit(X, y, groups=np.arange(99))
