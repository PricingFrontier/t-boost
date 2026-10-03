"""Polars-native ingestion: frames in, rustystats-style column-name vectors, no pandas.

Every test here runs against the estimator layer only (no Rust changes): a polars
``DataFrame``/``LazyFrame`` X, column-name ``y``/``sample_weight``/``exposure``/``groups``,
dtype-driven auto-categoricals, and name-based serve-time column selection. Parity tests
assert bit-identical predictions against the plain-numpy path, which is the exactness bar
this repo holds derived views to.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from t_boost.sklearn import PrecisionWarning, TBoostClassifier, TBoostRegressor

CHEAP = dict(n_trees=40, n_bags=1, validation_fraction=None, seed=0)


def frame_fixture(n: int = 240) -> pl.DataFrame:
    """Mixed-dtype frequency-style frame: float32 numerics, String + Categorical cats."""
    rng = np.random.default_rng(0)
    age = rng.uniform(18.0, 80.0, n).astype(np.float32)
    bm = rng.integers(50, 200, n).astype(np.float32)
    region = rng.choice(["N", "S", "E", "W"], n)
    fuel = rng.choice(["petrol", "diesel"], n)
    expo = rng.uniform(0.1, 1.0, n).astype(np.float32)
    lam = 0.1 * np.exp(0.01 * (age - 40.0) + (region == "N") * 0.3)
    claims = rng.poisson(lam * expo).astype(np.float32)
    return pl.DataFrame(
        {
            "age": age,
            "bm": bm,
            "region": region,
            "fuel": pl.Series(fuel).cast(pl.Categorical),
            "expo": expo,
            "claims": claims,
        }
    )


def fit_column_names(df: pl.DataFrame) -> TBoostRegressor:
    model = TBoostRegressor(objective="poisson", **CHEAP)
    return model.fit(df, "claims", exposure="expo")


# --- fit: column-name vectors, feature capture --------------------------------------------------


def test_fit_with_column_name_y_and_exposure_excludes_consumed_columns() -> None:
    df = frame_fixture()
    model = fit_column_names(df)
    assert list(model.feature_names_in_) == ["age", "bm", "region", "fuel"]
    assert model.n_features_in_ == 4
    # String + Categorical dtype columns were auto-detected without declaration.
    assert model._cat_indices_ == [2, 3]


def test_fit_column_names_matches_explicit_arrays() -> None:
    df = frame_fixture()
    by_name = fit_column_names(df)
    by_array = TBoostRegressor(objective="poisson", **CHEAP).fit(
        df.drop(["claims", "expo"]),
        df["claims"].to_numpy(),
        exposure=df["expo"].to_numpy(),
    )
    x_serve = df.drop(["claims", "expo"])
    assert np.array_equal(by_name.predict(x_serve), by_array.predict(x_serve))


def test_polars_series_vectors_accepted() -> None:
    df = frame_fixture()
    model = TBoostRegressor(objective="poisson", **CHEAP).fit(
        df.drop(["claims", "expo"]), df["claims"], exposure=df["expo"]
    )
    assert np.array_equal(model.predict(df), fit_column_names(df).predict(df))


def test_numeric_only_frame_bit_matches_numpy_f_order() -> None:
    df = frame_fixture().select(["age", "bm", "claims", "expo"])
    by_frame = fit_column_names(df)
    x_np = np.asfortranarray(df.select(["age", "bm"]).to_numpy().astype(np.float32))
    by_numpy = TBoostRegressor(objective="poisson", **CHEAP).fit(
        x_np, df["claims"].to_numpy(), exposure=df["expo"].to_numpy()
    )
    assert np.array_equal(by_frame.predict(df), by_numpy.predict(x_np))


def test_categorical_frame_matches_numpy_object_path() -> None:
    df = frame_fixture()
    by_frame = fit_column_names(df)
    # The pre-polars route: object ndarray + declared categorical indices.
    x_obj = np.empty((df.height, 4), dtype=object)
    x_obj[:, 0] = df["age"].to_numpy()
    x_obj[:, 1] = df["bm"].to_numpy()
    x_obj[:, 2] = df["region"].to_numpy()
    x_obj[:, 3] = df["fuel"].to_numpy()
    by_numpy = TBoostRegressor(
        objective="poisson", categorical_features=[2, 3], **CHEAP
    ).fit(x_obj, df["claims"].to_numpy(), exposure=df["expo"].to_numpy())
    assert np.array_equal(by_frame.predict(df), by_numpy.predict(x_obj))


def test_declared_categorical_on_numeric_polars_column() -> None:
    df = frame_fixture()
    model = TBoostRegressor(
        objective="poisson", categorical_features=["bm"], **CHEAP
    ).fit(df, "claims", exposure="expo")
    # declared numeric-dtype cat unions with the dtype-detected ones
    assert model._cat_indices_ == [1, 2, 3]
    assert np.isfinite(model.predict(df)).all()


# --- LazyFrame ------------------------------------------------------------------------------------


def test_lazyframe_fit_and_serve_match_eager() -> None:
    df = frame_fixture()
    eager = fit_column_names(df)
    lazy = TBoostRegressor(objective="poisson", **CHEAP).fit(
        df.lazy(), "claims", exposure="expo"
    )
    assert np.array_equal(lazy.predict(df.lazy()), eager.predict(df))
    assert list(lazy.feature_names_in_) == list(eager.feature_names_in_)


# --- serve: name-based selection -------------------------------------------------------------------


def test_serve_ignores_extra_columns_and_column_order() -> None:
    df = frame_fixture()
    model = fit_column_names(df)
    base = model.predict(df.select(["age", "bm", "region", "fuel"]))
    shuffled = df.select(["fuel", "expo", "bm", "claims", "age", "region"])
    assert np.array_equal(model.predict(shuffled), base)
    assert np.array_equal(model.predict_raw(shuffled), model.predict_raw(df))


def test_serve_missing_feature_column_raises() -> None:
    df = frame_fixture()
    model = fit_column_names(df)
    with pytest.raises(ValueError, match="missing feature column"):
        model.predict(df.drop("age"))


def test_numpy_fitted_model_serves_polars_positionally() -> None:
    df = frame_fixture().select(["age", "bm", "claims"])
    x_np = df.select(["age", "bm"]).to_numpy().astype(np.float32)
    model = TBoostRegressor(**CHEAP).fit(x_np, df["claims"].to_numpy())
    assert np.array_equal(
        model.predict(df.select(["age", "bm"])),
        model.predict(np.asfortranarray(x_np)),
    )


# --- nulls & dtypes ---------------------------------------------------------------------------------


def test_numeric_nulls_become_nan_missing_bin() -> None:
    df = frame_fixture()
    holed = df.with_columns(
        pl.when(pl.int_range(pl.len()) % 7 == 0).then(None).otherwise(pl.col("age")).alias("age")
    )
    model = fit_column_names(holed)
    x_nan = holed.drop(["claims", "expo"])
    preds = model.predict(x_nan)
    assert np.isfinite(preds).all()
    # NaN-for-null equivalence: same fit from a numpy design with NaNs in place of nulls.
    x_obj = np.empty((holed.height, 4), dtype=object)
    x_obj[:, 0] = holed["age"].to_numpy()  # nulls -> NaN
    x_obj[:, 1] = holed["bm"].to_numpy()
    x_obj[:, 2] = holed["region"].to_numpy()
    x_obj[:, 3] = holed["fuel"].to_numpy()
    by_numpy = TBoostRegressor(
        objective="poisson", categorical_features=[2, 3], **CHEAP
    ).fit(x_obj, holed["claims"].to_numpy(), exposure=holed["expo"].to_numpy())
    assert np.array_equal(preds, by_numpy.predict(x_obj))


def test_categorical_nulls_are_distinct_from_literal_reserved_marker() -> None:
    from t_boost._ingest import _CAT_MISSING, _cat_level

    df = frame_fixture()
    holed = df.with_columns(
        pl.when(pl.int_range(pl.len()) % 5 == 0)
        .then(None)
        .otherwise(pl.col("region"))
        .alias("region")
    )
    by_null = fit_column_names(holed)
    by_marker = fit_column_names(
        holed.with_columns(pl.col("region").fill_null(_CAT_MISSING))
    )
    x_serve = holed.drop(["claims", "expo"])
    assert _cat_level(None) == _CAT_MISSING
    assert _cat_level(_CAT_MISSING) != _CAT_MISSING
    assert np.isfinite(by_null.predict(x_serve)).all()
    assert np.isfinite(by_marker.predict(x_serve)).all()


def test_categorical_and_enum_nulls_collapse_to_missing_level() -> None:
    # Regression guard for the vectorized extraction: Enum.fill_null with an out-of-set
    # value silently KEEPS the null (no error), so the fast path must cast to String first
    # — otherwise None leaks into the core as a level.
    from t_boost._ingest import _CAT_MISSING, split_polars_columns

    for dtype in (pl.Categorical, pl.Enum(["x", "y"])):
        df = pl.DataFrame({"c": pl.Series(["x", None, "y"], dtype=dtype)})
        _, cat_x, _ = split_polars_columns(df, [0])
        assert cat_x == [["x", _CAT_MISSING, "y"]], f"nulls leaked for {dtype}"


def test_vectorized_string_extraction_matches_cat_level() -> None:
    from t_boost._ingest import _cat_level, split_polars_columns

    rng = np.random.default_rng(3)
    values = rng.choice(["a", "b", None, "c"], 500).tolist()
    df = pl.DataFrame({"c": pl.Series(values, dtype=pl.String)})
    _, cat_x, _ = split_polars_columns(df, [0])
    assert cat_x == [[_cat_level(v) for v in values]]


def test_enum_dtype_is_auto_categorical() -> None:
    df = frame_fixture().with_columns(
        pl.col("region").cast(pl.Enum(["N", "S", "E", "W"]))
    )
    model = fit_column_names(df)
    assert model._cat_indices_ == [2, 3]
    assert np.array_equal(model.predict(df), fit_column_names(frame_fixture()).predict(frame_fixture()))


def test_boolean_column_is_numeric() -> None:
    df = frame_fixture().with_columns((pl.col("age") > 40.0).alias("senior"))
    model = fit_column_names(df)
    assert "senior" in list(model.feature_names_in_)
    assert model._cat_indices_ == [2, 3]  # senior (col 4 -> numeric) did not join the cats


def test_unsupported_dtype_raises_clear_error() -> None:
    df = frame_fixture().with_columns(
        pl.lit(None).cast(pl.Datetime).alias("when")
    )
    with pytest.raises(ValueError, match="treated as numeric have non-numeric dtypes"):
        fit_column_names(df)


def test_null_in_named_vector_column_raises() -> None:
    df = frame_fixture().with_columns(
        pl.when(pl.int_range(pl.len()) == 3).then(None).otherwise(pl.col("expo")).alias("expo")
    )
    with pytest.raises(ValueError, match="expo.*null"):
        fit_column_names(df)


# --- PrecisionWarning parity -------------------------------------------------------------------------


def test_all_float32_frame_does_not_warn() -> None:
    df = frame_fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionWarning)
        fit_column_names(df)


def test_non_float32_numeric_warns_once() -> None:
    df = frame_fixture().with_columns(pl.col("bm").cast(pl.Int64))
    model = TBoostRegressor(objective="poisson", **CHEAP)
    with pytest.warns(PrecisionWarning):
        model.fit(df, "claims", exposure="expo")
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionWarning)
        model.predict(df)  # once per estimator, like the numpy path


# --- errors: misuse ------------------------------------------------------------------------------------


def test_column_name_y_with_numpy_x_raises_typeerror() -> None:
    with pytest.raises(TypeError, match="requires X to be a polars"):
        TBoostRegressor(**CHEAP).fit(np.ones((6, 2), dtype=np.float32), "claims")


def test_unknown_column_name_vector_raises() -> None:
    df = frame_fixture()
    with pytest.raises(ValueError, match="not a column of X"):
        TBoostRegressor(**CHEAP).fit(df, "claimz")


# --- tables ---------------------------------------------------------------------------------------------


def test_tables_exact_with_column_name_exposure() -> None:
    df = frame_fixture()
    model = fit_column_names(df)
    bank = json.loads(model.tables(df, exposure="expo"))
    assert bank["mode"] == "Exact"
    assert bank["tables"]
    named = json.loads(model.tables(df, exposure=df["expo"].to_numpy()))
    assert bank == named


def test_tables_reproduce_predict_raw_from_polars_fit() -> None:
    df = frame_fixture().select(["age", "bm", "claims", "expo"])
    model = TBoostRegressor(
        objective="poisson", max_interaction_order=2, **CHEAP
    ).fit(df, "claims", exposure="expo")
    bank = json.loads(model.tables(df))
    x = df.select(["age", "bm"]).to_numpy()
    raw = np.full(len(x), bank["f0"])
    column = {"age": 0, "bm": 1}
    for t in bank["tables"]:
        cells = []
        for a in t["axes"]:
            v = x[:, column[a["name"]]]
            idx = 1 + np.searchsorted(np.asarray(a["borders"]), v, side="left")
            cells.append(np.where(np.isnan(v), 0, idx))
        raw += np.asarray(t["values"])[np.ravel_multi_index(cells, t["shape"])]
    assert np.abs(raw - model.predict_raw(df)).max() < 1e-4


# --- classifier -------------------------------------------------------------------------------------------


def test_binary_classifier_column_name_y() -> None:
    df = frame_fixture().with_columns(
        (pl.col("claims") > 0).cast(pl.String).alias("any_claim")
    )
    model = TBoostClassifier(**CHEAP).fit(df.drop(["claims", "expo"]), "any_claim")
    assert list(model.classes_) == ["false", "true"]
    assert list(model.feature_names_in_) == ["age", "bm", "region", "fuel"]
    proba = model.predict_proba(df)
    assert proba.shape == (df.height, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)


def test_multiclass_classifier_with_groups_column() -> None:
    rng = np.random.default_rng(1)
    n = 300
    df = pl.DataFrame(
        {
            "x0": rng.uniform(0, 1, n).astype(np.float32),
            "x1": rng.uniform(0, 1, n).astype(np.float32),
            "kind": rng.choice(["a", "b", "c"], n),
            "entity": rng.integers(0, 40, n),
        }
    )
    model = TBoostClassifier(
        n_trees=30, n_bags=1, validation_fraction=0.2, early_stopping_rounds=10, seed=0
    ).fit(df, "kind", groups="entity")
    assert list(model.classes_) == ["a", "b", "c"]
    assert list(model.feature_names_in_) == ["x0", "x1"]
    proba = model.predict_proba(df)
    assert proba.shape == (n, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)


# --- persistence -------------------------------------------------------------------------------------------


def test_bytes_roundtrip_preserves_polars_serving() -> None:
    df = frame_fixture()
    model = fit_column_names(df)
    loaded = TBoostRegressor.from_bytes(model.to_bytes())
    shuffled = df.select(["fuel", "expo", "bm", "claims", "age", "region"])
    assert np.array_equal(loaded.predict(shuffled), model.predict(df))


def test_pickle_roundtrip_preserves_polars_serving() -> None:
    import pickle

    df = frame_fixture()
    model = fit_column_names(df)
    loaded = pickle.loads(pickle.dumps(model))
    assert np.array_equal(loaded.predict(df), model.predict(df))
