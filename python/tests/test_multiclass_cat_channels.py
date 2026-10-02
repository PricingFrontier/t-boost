"""P3 per-class frequency channels for multiclass categoricals
(``design/multichannel-categoricals.md`` §8).

The K>=3 softmax path bins categoricals once against the INTEGER class labels, so its
mean-TS channel computes ``E[class_index | level]`` — an ordinal statistic over nominal
classes (``design/multiclass-design.md`` §9a §5a). ``cat_channels=["class_freq"]`` replaces
that single axis with one channel per class carrying ``E[1[y == k] | level]``.

These tests pin: the default stays untouched, the knob is validated rather than silently
ignored, the extra axes actually appear (and only where they can help), and the whole
glass-box stack — exact decomposition, one collapsed table per RAW feature, level-labelled
export, pruning, serialization — survives K channels on one feature.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from t_boost.sklearn import TBoostClassifier, TBoostRegressor
from _artifact import model_bytes

pl = pytest.importorskip("polars")


def _frame(n: int = 900, k: int = 3, n_levels: int = 15, seed: int = 5):
    """A fixture whose high-cardinality categorical is NOMINALLY informative.

    Each level's class is a permutation of the level index, so the label-mean statistic
    ``E[class | level]`` orders levels almost arbitrarily while the per-class frequencies
    identify them exactly. A 2-level control categorical rides along: it sits below the
    per-class channels' level floor and must keep its single mean axis.
    """
    rng = np.random.default_rng(seed)
    level = rng.integers(0, n_levels, size=n)
    # Non-monotone level -> class map (a fixed shuffle, not `level % k`).
    perm = rng.permutation(n_levels) % k
    y = perm[level]
    # 10% label noise so the fit is not degenerate.
    flip = rng.random(n) < 0.10
    y = np.where(flip, rng.integers(0, k, size=n), y)
    num = rng.normal(size=(n, 2)).astype(np.float32)
    frame = pl.DataFrame(
        {
            "n0": num[:, 0],
            "n1": num[:, 1],
            "hi": [f"L{v}" for v in level],
            "lo": [f"S{v}" for v in (level % 2)],
        }
    )
    return frame, y.astype(int)


def _clf(**kw):
    params = dict(n_trees=30, seed=1, n_jobs=2, prune=False)
    params.update(kw)
    return TBoostClassifier(**params)


# --- the knob is validated, never silently ignored ------------------------------------


def test_unknown_channel_name_raises():
    x, y = _frame()
    with pytest.raises(ValueError, match="cat_channels entries must be"):
        _clf(cat_channels=["mean", "nonsense"]).fit(x, y)


def test_class_freq_on_a_regressor_raises():
    x, y = _frame()
    with pytest.raises(ValueError, match="no effect on a regression fit"):
        TBoostRegressor(n_trees=5, prune=False, cat_channels=["class_freq"]).fit(
            x, y.astype(float)
        )


def test_class_freq_on_a_binary_fit_is_satisfied_by_the_default():
    """K=2's mean-TS on the 0/1 indicator IS the class-1 frequency: the request is met by
    the default behavior, so it must neither raise nor add an axis."""
    x, y = _frame(k=2)
    base = _clf().fit(x, y)
    asked = _clf(cat_channels=["class_freq"]).fit(x, y)
    assert model_bytes(base) == model_bytes(asked)


# --- the default is untouched ---------------------------------------------------------


def test_mean_only_spellings_are_byte_identical_to_the_default():
    x, y = _frame()
    default = model_bytes(_clf().fit(x, y))
    assert model_bytes(_clf(cat_channels=["mean"]).fit(x, y)) == default
    assert model_bytes(_clf(cat_channels=[]).fit(x, y)) == default


def test_class_freq_actually_changes_the_multiclass_fit():
    x, y = _frame()
    assert model_bytes(_clf().fit(x, y)) != model_bytes(_clf(cat_channels=["class_freq"]).fit(x, y))


# --- the axes appear where (and only where) they can help -----------------------------


def test_one_axis_per_class_on_the_high_card_feature_and_a_fallback_on_the_low_card_one():
    x, y = _frame(k=3)
    est = _clf(cat_channels=["class_freq"]).fit(x, y)
    doc = json.loads(est.tables(x))
    # One entry per class, each an ordinary single-model table bank.
    assert sorted(doc) == ["0", "1", "2"]
    for bank in doc.values():
        names = [n for t in bank["tables"] for n in t["feature_names"]]
        # The collapse contract: RAW feature names only, never a per-channel axis name.
        assert not any("#cls" in n or "#count" in n or "#mean" in n for n in names)
        assert set(names) <= {"n0", "n1", "hi", "lo"}


def test_per_class_channels_keep_the_decomposition_exact():
    x, y = _frame(k=4)
    est = _clf(cat_channels=["class_freq"]).fit(x, y)
    doc = json.loads(est.tables(x, ref_measure="uniform"))
    assert sorted(doc) == ["0", "1", "2", "3"]
    for bank in doc.values():
        assert bank["mode"] == "Exact"


def test_the_collapsed_table_is_one_row_per_level_keyed_by_label():
    x, y = _frame(k=3)
    est = _clf(cat_channels=["class_freq"]).fit(x, y)
    doc = json.loads(est.tables(x))
    for bank in doc.values():
        for table in bank["tables"]:
            if table["feature_names"] != ["hi"]:
                continue
            axes = table["axes"]
            assert len(axes) == 1, "K channels collapse to ONE axis for the raw feature"
            labels = [level["label"] for level in axes[0]["levels"]]
            assert len(labels) == len(set(labels)) == 15
            assert all(lab.startswith("L") for lab in labels)
            # Every level's cell is a real cell of the exported table.
            assert all(0 <= level["cell"] < axes[0]["cells"] for level in axes[0]["levels"])


# --- the rest of the stack survives ---------------------------------------------------


def test_predict_proba_is_a_simplex_and_round_trips_through_bytes():
    x, y = _frame(k=4)
    est = _clf(cat_channels=["class_freq"]).fit(x, y)
    proba = est.predict_proba(x)
    assert proba.shape == (x.height, 4)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    restored = TBoostClassifier.from_bytes(est.to_bytes())
    assert np.allclose(restored.predict_proba(x), proba)


def test_pruning_survives_per_class_channels():
    """`prune=True` is the shipped default, and it runs `.explain()` internally — the
    Stage-B joint-cell flatten has to hold for K channels, not just two."""
    x, y = _frame(k=3)
    est = _clf(prune=True, cat_channels=["class_freq"]).fit(x, y)
    proba = est.predict_proba(x)
    assert proba.shape == (x.height, 3)
    assert np.isfinite(proba).all()


def test_mean_and_class_freq_together_also_works():
    x, y = _frame(k=3)
    both = _clf(cat_channels=["mean", "class_freq"]).fit(x, y)
    only = _clf(cat_channels=["class_freq"]).fit(x, y)
    assert model_bytes(both) != model_bytes(only)
    assert np.isfinite(both.predict_proba(x)).all()


def test_class_freq_composes_with_the_count_channel():
    x, y = _frame(k=3)
    est = _clf(cat_channels=["count", "class_freq"], cat_count_min_levels=5).fit(x, y)
    doc = json.loads(est.tables(x, ref_measure="uniform"))
    for bank in doc.values():
        assert bank["mode"] == "Exact"
        names = [n for t in bank["tables"] for n in t["feature_names"]]
        assert not any("#" in n for n in names)


def test_multi_channel_models_round_trip_their_input_column_count():
    """Regression (P1 defect surfaced by P3's K channels): `_attach_model` reads the model's
    AXIS count, which exceeds the input column count once a categorical owns more than one
    channel — so a reloaded model used to reject the very frame it was fit on. Covers the
    count channel too, not just the per-class ones."""
    x, y = _frame(k=3)
    for channels, extra in (
        (["mean", "count"], {"cat_count_min_levels": 5}),
        (["class_freq"], {}),
    ):
        est = _clf(cat_channels=channels, **extra).fit(x, y)
        assert est.n_features_in_ == x.width
        restored = TBoostClassifier.from_bytes(est.to_bytes())
        assert restored.n_features_in_ == x.width
        assert np.allclose(restored.predict_proba(x), est.predict_proba(x))
        from_json = TBoostClassifier.from_json(est.to_json())
        assert np.allclose(from_json.predict_proba(x), est.predict_proba(x))


def test_class_freq_min_levels_gates_by_cardinality():
    """The per-class channels can be restricted to high-cardinality features, the way
    `cat_count_min_levels` restricts the count channel. At a floor above every categorical's
    cardinality the fit falls all the way back to the mean-only default."""
    x, y = _frame(k=3)  # 'hi' has 15 levels, 'lo' has 2
    gated = _clf(cat_channels=["class_freq"], cat_class_freq_min_levels=100).fit(x, y)
    assert model_bytes(gated) == model_bytes(_clf().fit(x, y))
    live = _clf(cat_channels=["class_freq"], cat_class_freq_min_levels=10).fit(x, y)
    assert model_bytes(live) != model_bytes(gated)


def test_binary_fit_with_count_and_class_freq_equals_mean_plus_count():
    """The full K=2 contract for a multiclass-shaped knob: `class_freq` is inert (the binary
    mean-TS on the 0/1 indicator IS the class-1 frequency), so `["count","class_freq"]` must
    reduce to exactly `["mean","count"]` — NOT to a count-only fit with the target statistic
    silently deleted."""
    x, y = _frame(k=2)
    asked = _clf(cat_channels=["count", "class_freq"], cat_count_min_levels=5).fit(x, y)
    equivalent = _clf(cat_channels=["mean", "count"], cat_count_min_levels=5).fit(x, y)
    assert model_bytes(asked) == model_bytes(equivalent)


def test_multiclass_feature_below_the_class_gate_keeps_its_mean_channel():
    """Per-feature replacement: with the class gate raised above every categorical's
    cardinality but the count gate low enough to admit them, each feature must fall back to
    mean+count — never to count alone."""
    x, y = _frame(k=3)
    gated = _clf(
        cat_channels=["count", "class_freq"],
        cat_count_min_levels=5,
        cat_class_freq_min_levels=100,
    ).fit(x, y)
    equivalent = _clf(cat_channels=["mean", "count"], cat_count_min_levels=5).fit(x, y)
    assert model_bytes(gated) == model_bytes(equivalent)
