"""Serve-path speedups must not change a single prediction.

The table models take categoricals as codes into distinct labels (``cat_codes``) instead of one
string per row (``cat_x``), pandas columns are factorized instead of stringified value by value,
and polars columns become ``Enum`` codes. Each of those is only allowed where it reproduces the
per-row labels exactly, so every test here compares against the per-row label path bit for bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from t_boost import _ingest
from t_boost._ingest import _cat_level, split_polars_columns
from t_boost._t_boost import _MultiClassTableModel, _TableModel
from t_boost.sklearn import (
    TBoostClassifier,
    TBoostRegressor,
    _column_as_codes,
    _column_as_str_list,
    _factorized_levels,
)

CHEAP = dict(n_trees=40, n_bags=1, validation_fraction=None, seed=0)


def _adversarial_columns(n: int = 200) -> dict[str, pd.Series]:
    rng = np.random.default_rng(0)
    return {
        "obj_str_nan_none": pd.Series(rng.choice(np.array(["a", "b", None, np.nan, "c"], dtype=object), n)),
        "obj_mixed_1_1.0_True": pd.Series(rng.choice(np.array([1, 1.0, True, "x"], dtype=object), n)),
        "obj_pd_na": pd.Series(rng.choice(np.array(["a", pd.NA, "b"], dtype=object), n)),
        "string_dtype": pd.Series(rng.choice(["a", "b", None], n), dtype="string"),
        "category_str": pd.Series(rng.choice(["a", "b", None], n), dtype="category"),
        "category_float": pd.Series(rng.choice([1.0, 2.5, np.nan], n)).astype("category"),
        "int64": pd.Series(rng.integers(-3, 3, n)),
        "bool": pd.Series(rng.integers(0, 2, n).astype(bool)),
        "nullable_int": pd.Series(rng.choice([1, 2, None], n), dtype="Int64"),
        "float_signed_zero": pd.Series(rng.choice([0.0, -0.0, 1.5, np.nan], n)),
        "datetime": pd.Series(pd.to_datetime(rng.choice(["2020-01-01", None], n))),
    }


@pytest.mark.parametrize("name", list(_adversarial_columns()))
def test_factorized_levels_equal_the_per_value_loop_or_fall_back(name: str) -> None:
    col = _adversarial_columns()[name]
    got = _factorized_levels(col)
    assert got is None or got == [_cat_level(v) for v in col.to_numpy()]


def test_factorize_refuses_columns_where_equal_values_stringify_differently() -> None:
    cols = _adversarial_columns()
    # 1 == 1.0 == True and -0.0 == 0.0, yet each has its own label.
    assert _factorized_levels(cols["obj_mixed_1_1.0_True"]) is None
    assert _factorized_levels(cols["float_signed_zero"]) is None


@pytest.mark.parametrize("n", [5, 200])
def test_column_codes_reconstruct_the_per_row_labels(n: int) -> None:
    df = pd.DataFrame({k: v.iloc[:n] for k, v in _adversarial_columns().items()})
    for j in range(df.shape[1]):
        codes, labels = _column_as_codes(df, j)
        assert codes.dtype == np.uint32 and len(codes) == n
        assert [labels[c] for c in codes] == _column_as_str_list(df, j)


@pytest.mark.parametrize("n", [10, _ingest._POLARS_CODES_MIN_ROWS + 500])
def test_polars_codes_reconstruct_the_string_list_labels(n: int) -> None:
    rng = np.random.default_rng(1)
    s = pl.Series(rng.choice(["a", "b", "c"], n)).scatter(rng.choice(n, n // 10), None)
    df = pl.DataFrame(
        {
            "string": s,
            "categorical": s.cast(pl.Categorical),
            "enum": s.cast(pl.Enum(["c", "a", "b"])),
            "declared_numeric": pl.Series(rng.choice([1.0, -0.0, 0.0], n)),
        }
    )
    idx = list(range(df.width))
    _, labels_per_row, _ = split_polars_columns(df, idx)
    _, coded, _ = split_polars_columns(df, idx, coded=True)
    for want, (codes, labels) in zip(labels_per_row, coded):
        assert [labels[c] for c in codes] == want


def _frame(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    region = rng.choice(np.array(["N", "S", "E", "W", None], dtype=object), n)
    return pd.DataFrame(
        {
            "age": rng.uniform(18.0, 80.0, n),
            "region": region,
            "fuel": pd.Series(rng.choice(["petrol", "diesel"], n), dtype="category"),
            "bm": rng.integers(50, 200, n),
        }
    )


def _labels_path(est, X):  # the pre-speedup serve: one label string per row
    x32, cat_x = est._serve_design(X, coded=False)
    return x32, cat_x


@pytest.mark.parametrize("n_serve", [7, 3000])
def test_regressor_coded_predictions_are_bit_identical_to_label_predictions(n_serve: int) -> None:
    X = _frame(3000)
    rng = np.random.default_rng(3)
    y = rng.poisson(0.1 + 0.2 * (X["region"] == "N").to_numpy()).astype(np.float64)
    est = TBoostRegressor(objective="poisson", categorical_features=["region", "fuel"], **CHEAP)
    est.fit(X, y)
    assert isinstance(est._model, _TableModel)  # the coded route is table-model only
    Xs = X.iloc[:n_serve]
    assert "cat_codes" in est._serve_kwargs(Xs, est._model)[1]
    x32, cat_x = _labels_path(est, Xs)
    want = np.asarray(est._model.predict(x32, cat_x=cat_x), dtype=np.float64)
    np.testing.assert_array_equal(est.predict(Xs), want)
    want_raw = np.asarray(est._model.predict_raw(x32, cat_x=cat_x), dtype=np.float64)
    np.testing.assert_array_equal(est.predict_raw(Xs), want_raw)
    # polars input takes its own coded route (Enum codes at this size).
    np.testing.assert_array_equal(est.predict(pl.from_pandas(Xs)), want)


def test_binary_and_multiclass_coded_probabilities_are_bit_identical() -> None:
    X = _frame(2000)
    rng = np.random.default_rng(4)
    for n_classes in (2, 3):
        y = rng.integers(0, n_classes, len(X))
        est = TBoostClassifier(categorical_features=["region", "fuel"], **CHEAP).fit(X, y)
        x32, cat_x = _labels_path(est, X)
        model = est._multi_model if est._multi_model is not None else est._model
        assert isinstance(model, (_TableModel, _MultiClassTableModel))
        assert "cat_codes" in est._serve_kwargs(X, model)[1]
        want = np.asarray(model.predict_proba(x32, cat_x=cat_x), dtype=np.float64)
        np.testing.assert_array_equal(est.predict_proba(X), want)


def test_cat_codes_guards() -> None:
    X = _frame(300)
    y = np.random.default_rng(5).normal(size=len(X))
    est = TBoostRegressor(categorical_features=["region", "fuel"], **CHEAP).fit(X, y)
    x32, cat_x = _labels_path(est, X)
    _, coded = est._serve_kwargs(X, est._model)
    with pytest.raises(ValueError, match="not both"):
        est._model.predict(x32, cat_x=cat_x, **coded)
    codes, labels = coded["cat_codes"][0]
    bad = [(np.full_like(codes, len(labels)), labels)] + coded["cat_codes"][1:]
    with pytest.raises(ValueError, match="outside its"):
        est._model.predict(x32, cat_codes=bad)
