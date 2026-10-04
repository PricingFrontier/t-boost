"""The Haute integration surface (specs/roadmap/t-boost-improvements.md): the metadata slot
(R5), the unknown-category policy (R7), exposure refused under squared_error (R8), the
contribution matrix (R9), public cell indices (R10) and typed top-level imports (R11)."""

from __future__ import annotations

import json
import warnings

import numpy as np
import polars as pl
import pytest

import t_boost
from t_boost import SerializationError
from t_boost.sklearn import ContributionMatrix, TBoostClassifier, TBoostRegressor

FAST = dict(n_trees=60, n_bags=2, seed=3, n_jobs=2)


@pytest.fixture(autouse=True)
def _quiet() -> None:
    warnings.simplefilter("ignore")


def _frame(n: int = 3000, seed: int = 0) -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    a[:40] = np.nan
    d = rng.choice(list("pqrstu"), n).tolist()
    d[:3] = ["rare1", "rare2", None]
    frame = pl.DataFrame({"a:b": a, "c": rng.normal(size=n), "d": d})
    y = (
        np.sin(np.nan_to_num(a) * 2) * frame["c"].to_numpy()
        + (frame["d"] == "p").fill_null(False).to_numpy()
        + rng.normal(size=n) * 0.3
    )
    return frame, y


@pytest.fixture(scope="module")
def fitted() -> tuple[TBoostRegressor, pl.DataFrame]:
    warnings.simplefilter("ignore")
    frame, y = _frame()
    return TBoostRegressor(**FAST).fit(frame, y), frame


# --- R5 ------------------------------------------------------------------------------------


def test_metadata_round_trips_through_both_formats(fitted) -> None:
    model, frame = fitted
    model.metadata = {"haute": {"task": "regression", "levels": ["a", "b"], "n": 3}}
    try:
        for loaded in (
            TBoostRegressor.from_bytes(model.to_bytes()),
            TBoostRegressor.from_json(model.to_json()),
        ):
            assert loaded.metadata == model.metadata
            np.testing.assert_array_equal(loaded.predict(frame), model.predict(frame))
    finally:
        model.metadata = {}


def test_metadata_defaults_empty_and_leaves_the_header_alone(fitted) -> None:
    model, _ = fitted
    assert model.metadata == {}
    assert "metadata" not in json.loads(model.to_json())


def test_metadata_must_be_json(fitted) -> None:
    model, _ = fitted
    for bad in ({"x": object()}, {"x": float("nan")}, {1: "int key"}):
        model.metadata = bad
        try:
            with pytest.raises(SerializationError):
                model.to_bytes()
        finally:
            model.metadata = {}
    with pytest.raises(TypeError):
        model.metadata = ["not", "a", "dict"]  # type: ignore[assignment]


def test_metadata_survives_a_refit() -> None:
    frame, y = _frame(800)
    model = TBoostRegressor(n_trees=10, n_bags=1, prune=False)
    model.metadata = {"owner": "haute"}
    model.fit(frame, y)
    assert model.metadata == {"owner": "haute"}


# --- R7 ------------------------------------------------------------------------------------


def test_categories_lists_every_fitted_level(fitted) -> None:
    model, _ = fitted
    levels = model.categories_["d"]
    assert set(levels) == {"p", "q", "r", "s", "t", "u", "rare1", "rare2", None}
    assert list(model.categories_) == ["d"]


def test_unknown_category_error_covers_every_scoring_entry_point(fitted) -> None:
    model, _ = fitted
    bad = pl.DataFrame({"a:b": [0.0, 0.0], "c": [0.0, 0.0], "d": ["never", "p"]})
    model.set_params(unknown_category="error")
    try:
        assert hasattr(model, "_model"), "a scoring-only parameter must not unfit the model"
        for call in (
            model.predict,
            model.predict_raw,
            lambda x: model.predict_contributions(x, return_format="matrix"),
            model.tables,
            model.cell_indices,
        ):
            with pytest.raises(ValueError, match="never"):
                call(bad)
        # A null is never unknown, and seen values (pooled rare members too) score.
        ok = pl.DataFrame({"a:b": [0.0] * 3, "c": [0.0] * 3, "d": [None, "rare1", "p"]})
        assert np.isfinite(model.predict(ok)).all()
    finally:
        model.set_params(unknown_category="default_cell")
    assert np.isfinite(model.predict(bad)).all()


def test_unknown_category_rejects_a_bad_policy(fitted) -> None:
    model, frame = fitted
    model.set_params(unknown_category="ignore")
    try:
        with pytest.raises(ValueError, match="unknown_category"):
            model.predict(frame)
    finally:
        model.set_params(unknown_category="default_cell")


def test_unknown_category_survives_serialization() -> None:
    frame, y = _frame(800)
    model = TBoostRegressor(n_trees=10, n_bags=1, unknown_category="error").fit(frame, y)
    loaded = TBoostRegressor.from_bytes(model.to_bytes())
    assert loaded.unknown_category == "error"
    assert loaded.categories_ == model.categories_


# --- R8 ------------------------------------------------------------------------------------


def test_exposure_is_refused_under_squared_error() -> None:
    frame, y = _frame(500)
    with pytest.raises(ValueError, match="squared_error"):
        TBoostRegressor(n_trees=5, n_bags=1).fit(frame, y, exposure=np.ones(len(y)))


# --- R9 ------------------------------------------------------------------------------------


def test_matrix_contributions_are_exact_and_keyed_by_tuples(fitted) -> None:
    model, frame = fitted
    result = model.predict_contributions(frame, return_format="matrix")
    assert isinstance(result, ContributionMatrix)
    assert result.values.shape == (frame.height, len(result.terms))
    assert all(isinstance(t, tuple) for t in result.terms)
    assert ("a:b",) in result.terms, "a feature name containing ':' stays one name"
    for term, kind in zip(result.terms, result.term_types):
        assert kind == ("main" if len(term) == 1 else "interaction")
    raw = model.predict_raw(frame).astype(np.float64)
    np.testing.assert_allclose(result.base_value + result.values.sum(axis=1), raw, atol=1e-5)


def test_matrix_split_interactions_columns_are_the_fitted_features(fitted) -> None:
    model, frame = fitted
    full = model.predict_contributions(frame, return_format="matrix")
    split = model.predict_contributions(frame, return_format="matrix", split_interactions=True)
    assert split.terms == [("a:b",), ("c",), ("d",)]
    assert split.term_types == ["feature"] * 3
    np.testing.assert_allclose(split.values.sum(axis=1), full.values.sum(axis=1), atol=1e-9)


def test_matrix_carries_exposure_as_its_own_column() -> None:
    frame, y = _frame(1500)
    exposure = np.full(len(y), 2.0)
    model = TBoostRegressor(objective="poisson", **FAST).fit(frame, np.abs(y), exposure=exposure)
    result = model.predict_contributions(frame, return_format="matrix", exposure=exposure)
    assert result.terms[-1] == ("exposure",) and result.term_types[-1] == "exposure"
    np.testing.assert_allclose(result.values[:, -1], np.log(exposure))


def test_matrix_intercept_only_model_has_zero_columns() -> None:
    frame, _ = _frame(400)
    model = TBoostRegressor(n_trees=20, n_bags=1).fit(frame, np.full(frame.height, 3.0))
    result = model.predict_contributions(frame, return_format="matrix")
    assert result.values.shape == (frame.height, 0) and result.terms == []
    np.testing.assert_allclose(result.base_value, model.predict_raw(frame), atol=1e-6)


def test_matrix_multiclass_has_a_leading_class_axis() -> None:
    frame, y = _frame(1500)
    labels = np.digitize(y, [-0.5, 0.5])
    model = TBoostClassifier(n_trees=40, n_bags=1, seed=0).fit(frame, labels)
    result = model.predict_contributions(frame, return_format="matrix")
    assert result.classes == [0, 1, 2]
    assert result.values.shape == (3, frame.height, len(result.terms))
    logits = result.base_value + result.values.sum(axis=2)
    proba = np.exp(logits - logits.max(axis=0))
    np.testing.assert_allclose((proba / proba.sum(axis=0)).T, model.predict_proba(frame), atol=1e-5)


# --- R10 -----------------------------------------------------------------------------------


def test_cell_indices_index_each_exported_table(fitted) -> None:
    model, frame = fitted
    export = {tuple(t["feature_names"]): t for t in json.loads(model.tables(frame))["tables"]}
    cells = model.cell_indices(frame)
    assert set(cells) == set(export)
    contributions = model.predict_contributions(frame, return_format="matrix")
    for key, block in cells.items():
        table = export[key]
        assert block.shape == (frame.height, len(table["axes"])) and block.dtype == np.uint32
        flat = np.ravel_multi_index(tuple(block.T.astype(np.intp)), table["shape"])
        np.testing.assert_array_equal(
            np.asarray(table["values"])[flat],
            contributions.values[:, contributions.terms.index(key)],
        )
        for k, axis in enumerate(table["axes"]):
            if axis["levels"] is None:
                # Documented rule: cell 0 is missing, else 1 + #{borders < float32(v)}.
                v = frame[axis["name"]].to_numpy().astype(np.float32)
                b = np.asarray(axis["borders"], dtype=np.float32)
                rule = np.where(np.isnan(v), 0, 1 + np.searchsorted(b, v, side="left"))
                np.testing.assert_array_equal(block[:, k], rule)


def test_cell_borders_are_right_closed_in_float32(fitted) -> None:
    model, _ = fitted
    export = {tuple(t["feature_names"]): t for t in json.loads(model.tables(_frame()[0]))["tables"]}
    borders = np.asarray(export[("c",)]["axes"][0]["borders"], dtype=np.float32)
    b = borders[len(borders) // 2]
    just_above = float(np.nextafter(b, np.float32(np.inf)))
    rounds_onto = float(b) + abs(float(b)) * 1e-12  # float64 above b, float32 rounds onto it
    assert np.float32(rounds_onto) == b and rounds_onto != float(b)
    probe = pl.DataFrame({"a:b": [0.0] * 3, "c": [float(b), just_above, rounds_onto], "d": ["p"] * 3})
    got = model.cell_indices(probe)[("c",)][:, 0]
    lower = 1 + int(np.searchsorted(borders, b, side="left"))
    assert got.tolist() == [lower, lower + 1, lower]


def test_cell_indices_refuses_multiclass() -> None:
    frame, y = _frame(600)
    model = TBoostClassifier(n_trees=10, n_bags=1).fit(frame, np.digitize(y, [-0.5, 0.5]))
    with pytest.raises(ValueError, match="multiclass|binary"):
        model.cell_indices(frame)


# --- R11 -----------------------------------------------------------------------------------


def test_top_level_names_resolve_to_the_estimators() -> None:
    assert t_boost.TBoostRegressor is TBoostRegressor
    assert t_boost.TBoostClassifier is TBoostClassifier
