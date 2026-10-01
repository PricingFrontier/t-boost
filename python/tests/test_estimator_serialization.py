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

import numpy as np
from sklearn.metrics import accuracy_score

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


def test_numeric_regressor_bytes_and_json_stay_raw_wire_format() -> None:
    x, y = _small_regression_fixture()
    reg = TBoostRegressor(
        n_trees=16,
        learning_rate=0.25,
        max_bin=32,
        seed=7,
        validation_fraction=None,
        n_bags=0,
        leaf_refine_steps=0,
        colsample_bytree=1.0,
    ).fit(x.astype(np.float32), y)

    blob = reg.to_bytes()
    assert blob[:4] != b"TBP1"  # no classes_, no categoricals: unchanged raw wire format (H9)
    text = reg.to_json()
    assert "__tri_estimator__" not in text

    restored = TBoostRegressor.from_bytes(blob)
    np.testing.assert_array_equal(
        restored.predict(x.astype(np.float32)), reg.predict(x.astype(np.float32))
    )
    restored_json = TBoostRegressor.from_json(text)
    np.testing.assert_array_equal(
        restored_json.predict(x.astype(np.float32)), reg.predict(x.astype(np.float32))
    )
