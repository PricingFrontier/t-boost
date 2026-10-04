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


# --- R1 / R2 / R3: eval_set, fit report, callbacks ------------------------------------------

from _artifact import model_bytes  # noqa: E402

ES = dict(n_trees=400, n_bags=2, seed=5, early_stopping_adaptive=None, early_stopping_rounds=40,
          early_stopping_min_delta=0.0, learning_rate=0.2)


@pytest.fixture(scope="module")
def split() -> tuple[pl.DataFrame, np.ndarray, pl.DataFrame, np.ndarray]:
    frame, y = _frame(6000, seed=11)
    return frame[:4500], y[:4500], frame[4500:], y[4500:]


def test_eval_set_stops_each_bag_at_its_best_eval_round(split) -> None:
    x, y, xv, yv = split
    model = TBoostRegressor(**ES, prune=False).fit(x, y, eval_set=(xv, yv))
    assert "train" not in model.evals_result_, "the training curve is computed for callbacks only"
    history = model.evals_result_["eval"]["deviance"]
    assert len(history) == 2 == len(model.n_trees_per_bag_)
    for kept, curve, reason in zip(model.n_trees_per_bag_, history, model.stopping_reason_per_bag_):
        assert kept == int(np.argmin(curve)) + 1
        assert reason in ("early_stopping", "max_trees")
    report = {row["param"]: row for row in model.binding_report_}
    assert report["validation_fraction"]["overridden_by"] == "eval_set"


def test_eval_rows_never_reach_training(split) -> None:
    x, y, xv, yv = split
    moved = yv.copy()
    moved[0] = 1e4
    histories = [
        TBoostRegressor(**ES)
        .fit(x, y, eval_set=(xv, target), callbacks=lambda info: None)
        .evals_result_["train"]["deviance"]
        for target in (yv, moved)
    ]
    for a, b in zip(*histories):
        k = min(len(a), len(b))
        assert k > 0 and a[:k] == b[:k]


def test_eval_set_is_identical_across_n_jobs(split) -> None:
    x, y, xv, yv = split
    blobs = {
        model_bytes(TBoostRegressor(**{**ES, "n_jobs": jobs}).fit(x, y, eval_set=(xv, yv)))
        for jobs in (1, 4)
    }
    assert len(blobs) == 1


def test_eval_set_binary_classifier_and_column_names(split) -> None:
    x, y, xv, yv = split
    labels, val_labels = (y > 0.5).astype(int), (yv > 0.5).astype(int)
    frame_v = xv.with_columns(pl.Series("target", val_labels))
    model = TBoostClassifier(**ES).fit(x, labels, eval_set=(frame_v, "target"))
    assert model.n_trees_per_bag_ is not None and "eval" in model.evals_result_


def test_eval_set_refusals(split) -> None:
    x, y, xv, yv = split
    with pytest.raises(ValueError, match="eval_exposure"):
        TBoostRegressor(objective="poisson", **ES).fit(
            x, np.abs(y), exposure=np.ones(len(y)), eval_set=(xv, np.abs(yv))
        )
    with pytest.raises(ValueError, match="reanchor_slope"):
        TBoostRegressor(reanchor_slope=True, **ES).fit(x, y, eval_set=(xv, yv))
    with pytest.raises(ValueError, match="binary"):
        TBoostClassifier(**ES).fit(x, np.digitize(y, [-0.5, 0.5]), eval_set=(xv, np.digitize(yv, [-0.5, 0.5])))


def test_stopping_reasons_and_counts(split) -> None:
    x, y, xv, yv = split
    capped = TBoostRegressor(n_trees=30, n_bags=2, validation_fraction=None, prune=False).fit(x, y)
    assert capped.n_trees_per_bag_ == [30, 30] and capped.stopping_reason_ == "max_trees"
    assert capped.n_trees_ == 30
    flat = TBoostRegressor(n_trees=30, n_bags=1, prune=False).fit(x, np.full(len(y), 2.0))
    assert flat.stopping_reason_ == "no_split" and flat.n_trees_ == 0
    stopped = TBoostRegressor(**ES).fit(x, y, eval_set=(xv, yv))
    assert stopped.stopping_reason_ == "early_stopping"
    assert "early_stopping" in stopped.stopping_reason_per_bag_


def test_fit_report_survives_serialization(split) -> None:
    x, y, xv, yv = split
    model = TBoostRegressor(**ES).fit(x, y, eval_set=(xv, yv))
    for loaded in (TBoostRegressor.from_bytes(model.to_bytes()), TBoostRegressor.from_json(model.to_json())):
        assert loaded.n_trees_per_bag_ == model.n_trees_per_bag_
        assert loaded.stopping_reason_per_bag_ == model.stopping_reason_per_bag_
        assert loaded.n_trees_ == model.n_trees_ and loaded.stopping_reason_ == model.stopping_reason_
    doc = json.loads(model.to_json())
    del doc["fit_report"]
    old = TBoostRegressor.from_json(json.dumps(doc))
    assert old.n_trees_per_bag_ is None and old.stopping_reason_ is None


def test_callbacks_see_rounds_in_order_and_match_history(split) -> None:
    x, y, xv, yv = split
    seen: list[dict] = []
    model = TBoostRegressor(**ES).fit(x, y, eval_set=(xv, yv), callbacks=[seen.append])
    for bag in range(2):
        rounds = [e for e in seen if e["bag"] == bag]
        assert [e["round"] for e in rounds] == list(range(1, len(rounds) + 1))
        assert [e["eval_deviance"] for e in rounds] == model.evals_result_["eval"]["deviance"][bag]
        assert [e["train_deviance"] for e in rounds] == model.evals_result_["train"]["deviance"][bag]
    assert {e["phase"] for e in seen} == {"fit"} and {e["n_bags"] for e in seen} == {2}


def test_callback_true_stops_and_exceptions_propagate(split) -> None:
    x, y, _, _ = split
    model = TBoostRegressor(n_trees=200, n_bags=2, prune=False).fit(
        x, y, callbacks=lambda info: info["round"] >= 5
    )
    assert model.n_trees_per_bag_ == [5, 5]
    assert model.stopping_reason_ == "callback"

    class Cancelled(Exception):
        pass

    def cancel(info: dict) -> None:
        raise Cancelled(info["round"])

    with pytest.raises(Cancelled):
        TBoostRegressor(n_trees=200, n_bags=2).fit(x, y, callbacks=cancel)


# --- R6 ------------------------------------------------------------------------------------


def test_offset_under_squared_error_is_a_target_shift(split) -> None:
    x, y, _, _ = split
    rng = np.random.default_rng(1)
    offset = rng.normal(size=len(y)) * 3
    model = TBoostRegressor(**FAST).fit(x, y + offset, offset=offset)
    plain = TBoostRegressor(**FAST).fit(x, y)
    np.testing.assert_allclose(model.predict(x, offset=offset) - offset, plain.predict(x), atol=1e-3)
    np.testing.assert_array_equal(model.predict_raw(x, offset=offset), model.predict_raw(x) + offset)
    result = model.predict_contributions(x, return_format="matrix", offset=offset)
    assert result.terms[-1] == ("offset",) and result.term_types[-1] == "offset"
    np.testing.assert_array_equal(result.values[:, -1], offset)


def test_offset_column_name_and_eval_offset(split) -> None:
    x, y, xv, yv = split
    frame = x.with_columns(pl.Series("off", np.linspace(-1, 1, len(y))))
    frame_v = xv.with_columns(pl.Series("off", np.zeros(len(yv))))
    model = TBoostRegressor(**ES).fit(
        frame, y, offset="off", eval_set=(frame_v, yv), eval_offset="off"
    )
    assert "off" not in list(model.feature_names_in_)
    np.testing.assert_array_equal(
        model.predict_raw(frame, offset="off"), model.predict_raw(frame) + frame["off"].to_numpy()
    )
    with pytest.raises(ValueError, match="eval_offset"):
        TBoostRegressor(**ES).fit(frame, y, offset="off", eval_set=(frame_v, yv))


def test_offset_under_log_link_equals_exposure(split) -> None:
    x, y, _, _ = split
    exposure = np.exp(np.random.default_rng(2).normal(size=len(y)) * 0.3)
    counts = np.random.default_rng(3).poisson(exposure)
    by_offset = TBoostRegressor(objective="poisson", **FAST).fit(x, counts, offset=np.log(exposure))
    by_exposure = TBoostRegressor(objective="poisson", **FAST).fit(x, counts, exposure=exposure)
    np.testing.assert_array_equal(by_offset.predict(x), by_exposure.predict(x))


def test_offset_under_logistic_scores_around_the_offset(split) -> None:
    x, y, _, _ = split
    rng = np.random.default_rng(4)
    offset = rng.normal(size=len(y)) * 2
    labels = (rng.uniform(size=len(y)) < 1 / (1 + np.exp(-offset))).astype(int)
    model = TBoostClassifier(**FAST).fit(x, labels, offset=offset)
    p = model.predict_proba(x, offset=offset)[:, 1]
    np.testing.assert_allclose(
        p, 1 / (1 + np.exp(-(model.decision_function(x) + offset))), rtol=1e-12
    )
    assert np.mean(np.abs(model.decision_function(x))) < 0.5, "the offset carries the signal"
    records = model.predict_contributions(x[:3], offset=offset[:3])
    assert records[0]["contributions"][-1]["term"] == "offset"


def test_offset_refused_for_multiclass(split) -> None:
    x, y, _, _ = split
    with pytest.raises(ValueError, match="offset"):
        TBoostClassifier(n_trees=5, n_bags=1).fit(x, np.digitize(y, [-0.5, 0.5]), offset=np.zeros(len(y)))


def test_multiclass_fit_report_is_none() -> None:
    frame, y = _frame(600)
    model = TBoostClassifier(n_trees=10, n_bags=1).fit(frame, np.digitize(y, [-0.5, 0.5]))
    assert model.n_trees_per_bag_ is None and model.stopping_reason_ is None


@pytest.mark.parametrize(
    "options",
    [
        {"n_bags": 1},
        {"cat_channels": ["mean", "count"], "cat_count_min_levels": 2},
        {"prune": False},
    ],
    ids=["single-bag", "multi-channel", "unpruned"],
)
def test_eval_set_paths(split, options) -> None:
    x, y, xv, yv = split
    model = TBoostRegressor(**{**ES, **options}).fit(x, y, eval_set=(xv, yv))
    curves = model.evals_result_["eval"]["deviance"]
    assert len(curves) == len(model.n_trees_per_bag_) == int(model.n_bags)
    for kept, curve in zip(model.n_trees_per_bag_, curves):
        assert kept == int(np.argmin(curve)) + 1
    assert np.isfinite(model.predict(xv)).all()


def test_eval_set_with_groups(split) -> None:
    x, y, xv, yv = split
    groups = np.arange(len(y)) // 3
    model = TBoostRegressor(**ES).fit(x, y, groups=groups, eval_set=(xv, yv))
    assert len(model.n_trees_per_bag_) == 2 and np.isfinite(model.predict(xv)).all()


def test_eval_set_unknown_levels_follow_the_policy(split) -> None:
    x, y, xv, yv = split
    bad = xv.with_columns(pl.lit("never").alias("d"))
    with pytest.raises(ValueError, match="never"):
        TBoostRegressor(**ES, unknown_category="error").fit(x, y, eval_set=(bad, yv))
    TBoostRegressor(**ES).fit(x, y, eval_set=(bad, yv))  # default_cell: accepted


def test_contribution_matrix_is_exported() -> None:
    assert t_boost.ContributionMatrix is ContributionMatrix
