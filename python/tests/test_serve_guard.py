"""Serve-time guards: the cat_x provenance check on numeric-only binning paths, and
apply_keepset's user-id canonicalization before FeatureSet construction."""

from __future__ import annotations

import numpy as np
import pytest

from t_boost._t_boost import _Booster, _Model, _TableModel


def _numeric_and_category(n: int = 1500, seed: int = 0):
    rng = np.random.default_rng(seed)
    x_num = rng.normal(size=(n, 2)).astype(np.float32)
    levels = ["north", "south", "east"]
    codes = rng.integers(0, len(levels), size=n)
    cat_labels = [levels[c] for c in codes]
    bump = np.select([codes == 0, codes == 1], [0.5, -0.3], default=0.0)
    y = (0.4 * x_num[:, 0] + 0.3 * x_num[:, 1] + bump + rng.normal(scale=0.05, size=n)).astype(
        np.float32
    )
    return x_num, cat_labels, codes, y


def _noisy_poisson_pair(n: int = 3000, seed: int = 0):
    # Same fixture as test_prune.py's `_noisy_poisson`: real signal (+interaction) in
    # features 0,1, pure noise in 2,3 — proven to realize a {0,1} pairwise table.
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    mu = np.exp(0.3 * x[:, 0] + 0.2 * x[:, 1] + 0.2 * x[:, 0] * x[:, 1])
    y = rng.poisson(mu).astype(np.float32)
    return x, y


def test_predict_without_cat_x_raises_for_native_categorical_model() -> None:
    x_num, cat_labels, codes, y = _numeric_and_category()
    booster = _Booster(objective="squared_error", n_trees=40, seed=0)
    model = booster.fit(x_num, y, cat_x=[cat_labels])
    assert isinstance(model, _Model)

    # The natural mistake this guards against: append the ordinal-encoded category as
    # an extra numeric column, so width happens to equal model.grids.len() (2 + 1 = 3).
    x_wide = np.column_stack([x_num, codes.astype(np.float32)])
    with pytest.raises(Exception, match="cat_x"):
        model.predict(x_wide)


def test_predict_with_cat_x_still_works() -> None:
    x_num, cat_labels, codes, y = _numeric_and_category()
    booster = _Booster(objective="squared_error", n_trees=40, seed=0)
    model = booster.fit(x_num, y, cat_x=[cat_labels])

    pred = model.predict(x_num, cat_x=[cat_labels])
    assert pred.shape == (x_num.shape[0],)
    assert np.all(np.isfinite(pred))


def test_table_model_predict_without_cat_x_also_raises() -> None:
    x_num, cat_labels, codes, y = _numeric_and_category()
    booster = _Booster(objective="squared_error", n_trees=40, seed=0)
    model = booster.fit(x_num, y, cat_x=[cat_labels])

    weight = np.ones_like(y)
    sel_rows = list(range(0, len(y), 5))
    tm, _report = model.prune_to_tables(
        x_num, y, weight, sel_rows, cat_x=[cat_labels], se_rule=1.0
    )
    assert isinstance(tm, _TableModel)

    x_wide = np.column_stack([x_num, codes.astype(np.float32)])
    with pytest.raises(Exception, match="cat_x"):
        tm.predict(x_wide)

    pred = tm.predict(x_num, cat_x=[cat_labels])
    assert np.all(np.isfinite(pred))


def test_numeric_only_model_predicts_without_cat_x() -> None:
    # The guard must not over-fire: a model fit with no categorical columns at all has
    # no CategoricalTS axes, so omitting cat_x at serve time is the normal, valid call.
    rng = np.random.default_rng(2)
    x = rng.normal(size=(400, 3)).astype(np.float32)
    y = (0.5 * x[:, 0] - 0.2 * x[:, 1] + rng.normal(scale=0.05, size=400)).astype(np.float32)
    booster = _Booster(objective="squared_error", n_trees=30, seed=0)
    model = booster.fit(x, y)

    pred = model.predict(x)
    assert pred.shape == (400,)
    assert np.all(np.isfinite(pred))


def test_apply_keepset_unsorted_ids_match_sorted() -> None:
    import json

    x, y = _noisy_poisson_pair()
    booster = _Booster(objective="poisson", n_trees=300, seed=0)
    model = booster.fit(x, y)

    # Discover a real multi-feature table from the fitted bank rather than assuming one
    # exists at a hardcoded id — robust to exactly which support the greedy search picks.
    full_tables = json.loads(model.tables(x))["tables"]
    pair = next((t["feature_set"] for t in full_tables if len(t["feature_set"]) >= 2), None)
    assert pair is not None, "fixture must realize at least one multi-feature table"
    sorted_ids = sorted(pair)
    unsorted_ids = list(reversed(sorted_ids))
    assert unsorted_ids != sorted_ids, "need a genuine order permutation to exercise the fix"

    def feature_sets(tm) -> set[tuple[int, ...]]:
        return {tuple(sorted(t["feature_set"])) for t in json.loads(tm.tables())["tables"]}

    weight = np.ones_like(y)
    tm_sorted = model.apply_keepset(x, y, weight, [sorted_ids])
    tm_unsorted = model.apply_keepset(x, y, weight, [unsorted_ids])

    sets_sorted = feature_sets(tm_sorted)
    sets_unsorted = feature_sets(tm_unsorted)
    assert tuple(sorted_ids) in sets_sorted
    assert sets_sorted == sets_unsorted, (
        "an unsorted keep entry must retain the same table as its sorted form"
    )

    pred_sorted = tm_sorted.predict(x[:50])
    pred_unsorted = tm_unsorted.predict(x[:50])
    assert np.allclose(pred_sorted, pred_unsorted)
