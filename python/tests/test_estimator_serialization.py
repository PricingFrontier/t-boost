"""Estimator-level `to_bytes`/`from_bytes`/`to_json`/`from_json` serialization contract.

Covers two fixed HIGH findings (2026-07-11 review):
  H8 — a pruned multiclass model (`_MultiClassTableModel`, TBMT magic) wrapped in the TBP1
       envelope used to record kind='multi', misrouting it to the TBMC (`_MultiClassModel`)
       decoder on load. `_estimator_metadata` now distinguishes 'multi_tables'; `_unpack_bytes`/
       `_unpack_json` additionally sniff the inner payload's own magic/kind as a fallback so
       blobs already saved with the buggy 'multi' kind still load.
  H9 — the TBP1 envelope (which carries `classes_`) used to be emitted only when the estimator
       had categorical features, so a NUMERIC classifier's `to_bytes`/`from_bytes` round-trip
       silently rebuilt `classes_` from stringified schema labels (`predict()` returned strings
       instead of the original label dtype). The envelope is now emitted for any estimator that
       carries `classes_`, i.e. every classifier; numeric regressors are unaffected.
"""

from __future__ import annotations

import json
import pickle
from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score

import t_boost
from t_boost import SerializationError
from t_boost._t_boost import _MultiClassTableModel
from t_boost.sklearn import _ESTIMATOR_MAGIC, TBoostClassifier, TBoostRegressor


def _int_label_classifier_fixture() -> tuple[np.ndarray, np.ndarray]:
    x = np.array(
        [[float(i % 6), float((i // 2) % 5), float((i // 3) % 4)] for i in range(96)],
        dtype=np.float64,
    )
    y_reg = np.where(x[:, 0] <= 2.0, 1.5, -0.75) + np.where(x[:, 1] <= 2.0, 0.5, -0.25)
    y = (y_reg > np.median(y_reg)).astype(np.int64)
    return x, y


def _small_regression_fixture() -> tuple[np.ndarray, np.ndarray]:
    x = np.array(
        [[float(i % 6), float((i // 2) % 5), float((i // 3) % 4)] for i in range(96)],
        dtype=np.float64,
    )
    y = np.where(x[:, 0] <= 2.0, 1.5, -0.75) + np.where(x[:, 1] <= 2.0, 0.5, -0.25)
    return x, y.astype(np.float32)


def _categorical_multiclass_fixture(n: int = 2400, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    num = rng.normal(size=(n, 4)).astype(np.float32)
    levels = np.asarray(["north", "south", "east"], dtype=object)
    region = levels[rng.integers(0, levels.shape[0], size=n)]
    lin = np.stack(
        [
            0.7 * num[:, 0] + np.where(region == "north", 1.0, 0.0),
            0.6 * num[:, 1] + np.where(region == "south", 1.0, 0.0),
            0.4 * num[:, 0] * num[:, 1],
        ],
        axis=1,
    )
    y = np.argmax(lin + rng.normal(scale=0.5, size=(n, 3)), axis=1)
    x = np.empty((n, 5), dtype=object)
    x[:, :4] = num
    x[:, 4] = region
    return x, y


def test_numeric_binary_classifier_int_labels_round_trip() -> None:
    x, y = _int_label_classifier_fixture()
    clf = TBoostClassifier(
        n_trees=20,
        learning_rate=0.25,
        lambda_=1.0,
        max_bin=32,
        seed=3,
        validation_fraction=None,
        n_bags=0,
        leaf_refine_steps=0,
        colsample_bytree=1.0,
    ).fit(x, y)
    assert np.issubdtype(clf.classes_.dtype, np.integer)
    pred = clf.predict(x)
    assert np.issubdtype(pred.dtype, np.integer)

    blob = clf.to_bytes()
    assert blob[:4] == b"TBP1"  # H9: envelope now carries classes_ even without categoricals
    restored = TBoostClassifier.from_bytes(blob)
    assert restored.classes_.dtype.kind == clf.classes_.dtype.kind
    np.testing.assert_array_equal(restored.classes_, clf.classes_)
    restored_pred = restored.predict(x)
    assert np.issubdtype(restored_pred.dtype, np.integer)
    np.testing.assert_array_equal(restored_pred, pred)
    accuracy_score(y, restored_pred)  # would TypeError on a str/int dtype mismatch pre-fix

    restored_json = TBoostClassifier.from_json(clf.to_json())
    np.testing.assert_array_equal(restored_json.classes_, clf.classes_)
    np.testing.assert_array_equal(restored_json.predict(x), pred)


def test_pruned_multiclass_with_categoricals_round_trips_bytes_and_json() -> None:
    x, y = _categorical_multiclass_fixture()
    clf = TBoostClassifier(
        objective="logistic",
        n_trees=120,
        n_bags=1,
        seed=0,
        categorical_features=[4],
        prune=True,
        prune_se_rule=1.0,
    ).fit(x, y)
    assert isinstance(clf._multi_model, _MultiClassTableModel)
    proba = clf.predict_proba(x[:20])

    blob = clf.to_bytes()
    assert blob[:4] == b"TBP1"  # categoricals force the envelope; H8 was in its 'kind' field
    restored = TBoostClassifier.from_bytes(blob)
    np.testing.assert_allclose(restored.predict_proba(x[:20]), proba, atol=1e-5)
    np.testing.assert_array_equal(restored.classes_, clf.classes_)

    restored_json = TBoostClassifier.from_json(clf.to_json())
    np.testing.assert_allclose(restored_json.predict_proba(x[:20]), proba, atol=1e-5)
    np.testing.assert_array_equal(restored_json.classes_, clf.classes_)


def test_old_buggy_multi_kind_envelope_still_loads_via_magic_sniff() -> None:
    """Backward compat: a blob saved by the pre-fix code — envelope kind='multi' wrapping the
    TBMT tables container, the exact H8 bug — must still load via the sniff fallback."""
    x, y = _categorical_multiclass_fixture(n=500, seed=2)
    clf = TBoostClassifier(
        objective="logistic",
        n_trees=40,
        n_bags=1,
        seed=0,
        categorical_features=[4],
        prune=True,
        prune_se_rule=1.0,
    ).fit(x, y)
    assert isinstance(clf._multi_model, _MultiClassTableModel)
    proba = clf.predict_proba(x[:10])

    # -- bytes path: hand-pack the OLD (buggy) envelope directly -----------------------------
    inner = clf._multi_model.to_bytes()
    assert inner[:4] == b"TBMT"
    old_md = {
        "kind": "multi",  # pre-fix bug: multi_tables mis-recorded as plain 'multi'
        "cat_indices": [int(i) for i in clf._cat_indices_],
        "classes_": np.asarray(clf.classes_).tolist(),
    }
    old_header = json.dumps(old_md).encode("utf-8")
    old_blob = _ESTIMATOR_MAGIC + len(old_header).to_bytes(4, "big") + old_header + inner

    restored = TBoostClassifier.from_bytes(old_blob)
    np.testing.assert_allclose(restored.predict_proba(x[:10]), proba, atol=1e-5)
    np.testing.assert_array_equal(restored.classes_, clf.classes_)

    # -- json path: same old-style envelope, JSON inner model --------------------------------
    inner_json = clf._multi_model.to_json()
    assert '"t-boost-multiclass-tables"' in inner_json
    old_text = json.dumps(
        {
            "kind": "multi",
            "__tri_estimator__": 1,
            "model": inner_json,
            "cat_indices": [int(i) for i in clf._cat_indices_],
            "classes_": np.asarray(clf.classes_).tolist(),
        }
    )
    restored_json = TBoostClassifier.from_json(old_text)
    np.testing.assert_allclose(restored_json.predict_proba(x[:10]), proba, atol=1e-5)
    np.testing.assert_array_equal(restored_json.classes_, clf.classes_)


def _small_regressor(**kw: Any) -> TBoostRegressor:
    params: dict[str, Any] = dict(
        n_trees=16,
        learning_rate=0.25,
        max_bin=32,
        seed=7,
        validation_fraction=None,
        n_bags=0,
        leaf_refine_steps=0,
        colsample_bytree=1.0,
        prune=False,
    )
    params.update(kw)
    return TBoostRegressor(**params)


def _header(blob: bytes) -> dict[str, Any]:
    hlen = int.from_bytes(blob[4:8], "big")
    header: dict[str, Any] = json.loads(blob[8 : 8 + hlen])
    return header


def _with_header(blob: bytes, header: dict[str, Any]) -> bytes:
    hlen = int.from_bytes(blob[4:8], "big")
    raw = json.dumps(header).encode("utf-8")
    return _ESTIMATOR_MAGIC + len(raw).to_bytes(4, "big") + raw + blob[8 + hlen :]


def _frequency_frame(n: int = 400, seed: int = 5) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    exposure = rng.uniform(0.2, 1.0, size=n)
    return pl.DataFrame(
        {
            "a": a,
            "b": rng.normal(size=n),
            "region": rng.choice(["north", "south", "east"], size=n),
            "Exposure": exposure,
            "Weight": rng.uniform(0.5, 2.0, size=n),
            "ClaimCount": rng.poisson(np.exp(0.4 * a) * exposure).astype(np.float64),
        }
    )


def test_numeric_regressor_is_enveloped_and_legacy_raw_blobs_still_load() -> None:
    """Every estimator now writes the envelope (it carries params and column specs); a raw
    model blob, which releases up to 0.6.1 wrote for numeric regressors, still loads."""
    x, y = _small_regression_fixture()
    reg = _small_regressor().fit(x, y)
    pred = reg.predict(x)

    blob = reg.to_bytes()
    assert blob[:4] == b"TBP1"
    text = reg.to_json()
    assert "__t_boost_estimator__" in text and "__tri" not in text
    for restored in (TBoostRegressor.from_bytes(blob), TBoostRegressor.from_json(text)):
        np.testing.assert_array_equal(restored.predict(x), pred)
        assert restored.get_params() == reg.get_params()

    for legacy in (
        TBoostRegressor.from_bytes(reg._model.to_bytes()),
        TBoostRegressor.from_json(reg._model.to_json()),
    ):
        np.testing.assert_array_equal(legacy.predict(x), pred)


def test_header_records_version_estimator_and_params() -> None:
    x, y = _small_regression_fixture()
    reg = _small_regressor(n_trees=12).fit(x, y)
    header = _header(reg.to_bytes())
    assert header["schema_version"] == 2
    assert header["t_boost_version"] == t_boost.__version__
    assert header["estimator"] == "TBoostRegressor"
    assert header["params"]["n_trees"] == 12
    assert json.loads(reg.to_json())["params"]["n_trees"] == 12


def test_params_round_trip_including_non_json_types() -> None:
    """Tuples, numpy arrays and int-keyed dicts come back equal, so a loaded estimator can be
    cloned and refit with the same configuration."""
    x, y = _small_regression_fixture()
    reg = _small_regressor(
        monotone_constraints={0: 1, 1: -1},
        categorical_features=np.array([False, False, True]),
        cat_channels=["mean"],
        max_delta_step_gated=(0.01, 0.5),
    ).fit(x, y)
    expected = reg.get_params()
    for restored in (
        TBoostRegressor.from_bytes(reg.to_bytes()),
        TBoostRegressor.from_json(reg.to_json()),
    ):
        got = restored.get_params()
        assert got["monotone_constraints"] == {0: 1, 1: -1}
        assert got["max_delta_step_gated"] == (0.01, 0.5)
        np.testing.assert_array_equal(got["categorical_features"], expected["categorical_features"])
        assert got["categorical_features"].dtype == np.bool_
        rest = {k: v for k, v in got.items() if k != "categorical_features"}
        assert rest == {k: v for k, v in expected.items() if k != "categorical_features"}
        np.testing.assert_array_equal(restored.predict(x), reg.predict(x))
        refit = clone(restored).fit(x, y)
        np.testing.assert_array_equal(refit.predict(x), reg.predict(x))


def test_unserializable_param_fails_loudly() -> None:
    x, y = _small_regression_fixture()
    reg = _small_regressor().fit(x, y)
    reg.cat_channels = {"not", "a", "list"}  # bypass set_params, which would unfit it
    with pytest.raises(SerializationError, match="cat_channels"):
        reg.to_bytes()


def test_column_specs_and_required_columns_round_trip() -> None:
    df = _frequency_frame()
    reg = _small_regressor(objective="poisson").fit(
        df, "ClaimCount", sample_weight="Weight", exposure="Exposure"
    )
    assert reg.required_columns == ["a", "b", "region"]
    pred = reg.predict(df)
    for restored in (
        TBoostRegressor.from_bytes(reg.to_bytes()),
        TBoostRegressor.from_json(reg.to_json()),
        pickle.loads(pickle.dumps(reg)),
    ):
        assert restored._response_spec == "ClaimCount"
        assert restored._weights_spec == "Weight"
        assert restored._exposure_spec == "Exposure"
        assert restored._groups_spec is None
        assert restored.required_columns == ["a", "b", "region"]
        np.testing.assert_array_equal(restored.predict(df.select(restored.required_columns)), pred)

    # Arrays leave no spec, and a refit replaces the previous fit's specs.
    reg.fit(
        df.select("a", "b", "region"),
        df["ClaimCount"].to_numpy(),
        exposure=df["Exposure"].to_numpy(),
    )
    assert reg._exposure_spec is None and reg._response_spec is None
    assert TBoostRegressor.from_bytes(reg.to_bytes())._exposure_spec is None

    reg.set_params(n_trees=8)  # unfits: the specs described the discarded fit
    assert not hasattr(reg, "_exposure_spec")


def test_classifier_column_specs_round_trip() -> None:
    df = _frequency_frame().with_columns(Claimed=(pl.col("ClaimCount") > 0).cast(pl.Int64))
    clf = TBoostClassifier(
        n_trees=16, validation_fraction=None, n_bags=0, leaf_refine_steps=0, prune=False
    ).fit(df.drop("ClaimCount", "Exposure"), "Claimed", sample_weight="Weight")
    restored = TBoostClassifier.from_bytes(clf.to_bytes())
    assert restored._response_spec == "Claimed" and restored._weights_spec == "Weight"
    assert restored.get_params() == clf.get_params()
    np.testing.assert_array_equal(restored.predict_proba(df), clf.predict_proba(df))


def test_required_columns_needs_named_columns() -> None:
    x, y = _small_regression_fixture()
    reg = _small_regressor().fit(x, y)
    with pytest.raises(RuntimeError, match="named columns"):
        reg.required_columns
    # The round trip must not invent the native model's `f{i}` placeholder names either.
    restored = TBoostRegressor.from_bytes(reg.to_bytes())
    assert not hasattr(restored, "feature_names_in_")
    with pytest.raises(NotFittedError):
        TBoostRegressor().required_columns


def test_newer_schema_is_refused() -> None:
    x, y = _small_regression_fixture()
    reg = _small_regressor().fit(x, y)
    blob = reg.to_bytes()
    header = _header(blob)
    header["schema_version"] = 3
    with pytest.raises(SerializationError, match="schema_version 3"):
        TBoostRegressor.from_bytes(_with_header(blob, header))
    doc = json.loads(reg.to_json())
    doc["schema_version"] = 3
    with pytest.raises(SerializationError, match="schema_version 3"):
        TBoostRegressor.from_json(json.dumps(doc))


def test_loading_with_the_wrong_estimator_class_is_refused() -> None:
    x, y = _small_regression_fixture()
    reg = _small_regressor().fit(x, y)
    with pytest.raises(SerializationError, match="TBoostRegressor"):
        TBoostClassifier.from_bytes(reg.to_bytes())
    with pytest.raises(SerializationError, match="TBoostRegressor"):
        TBoostClassifier.from_json(reg.to_json())


def test_pickles_from_older_releases_still_load() -> None:
    """Releases up to 0.6.1 pickled the native model under `__tri_model_bytes__`."""
    x, y = _small_regression_fixture()
    reg = _small_regressor().fit(x, y)
    state = reg.__getstate__()
    state["__tri_model_bytes__"] = state.pop("__t_boost_model_bytes__")
    old = TBoostRegressor.__new__(TBoostRegressor)
    old.__setstate__(state)
    np.testing.assert_array_equal(old.predict(x), reg.predict(x))
