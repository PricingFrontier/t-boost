"""Regression oracles for the confirmed roadmap findings.

Each test names its finding so coverage can be audited without changing the backlog.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from t_boost import TBoostClassifier, TBoostRegressor
from t_boost._t_boost import _Booster, SerializationError
from t_boost.metrics import (
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_tweedie_deviance,
    ordered_gini,
)
from t_boost.sklearn import _guard_mu, _guard_reanchor


OPTIONS = dict(
    n_trees=8, n_bags=1, prune=False, graduate=False, band_tolerance=None,
    validation_fraction=None, min_data_in_leaf=2, n_jobs=1,
)


@pytest.mark.parametrize("multiclass", [False, True])
@pytest.mark.parametrize("failure", ["shape", "weight", "negative_weight", "categorical", "native"])
def test_bug002_failed_refit_keeps_complete_state_or_invalidates(multiclass, failure):
    x = pl.DataFrame({"a": np.arange(90, dtype=np.float32), "b": np.zeros(90)})
    y = np.repeat(["a", "b", "c"] if multiclass else ["a", "b"], 30 if multiclass else 45)
    model = TBoostClassifier(**OPTIONS).fit(x, y)
    if failure == "native":
        model.max_bin = 0
    reordered = x.select("b", "a")
    before = model.predict(reordered)
    blob = model.to_bytes()
    bad_y = np.repeat(["x", "y", "z"] if multiclass else ["x", "y"], 30 if multiclass else 45)
    bad_x = pd.DataFrame({"a": ["not-a-number"] * 90, "b": np.zeros(90)}) if failure == "categorical" else x
    weight = np.ones(89) if failure == "weight" else np.r_[-1., np.ones(89)] if failure == "negative_weight" else None
    with pytest.raises(ValueError):
        model.fit(bad_x, bad_y[:-1] if failure == "shape" else bad_y, sample_weight=weight)
    if model.__sklearn_is_fitted__():
        np.testing.assert_array_equal(model.predict(reordered), before)
        assert model.to_bytes() == blob
    else:
        with pytest.raises((ValueError, AttributeError)):
            model.predict(reordered)


def test_bug002_failed_regressor_refit_preserves_column_alignment():
    x = pl.DataFrame({"a": np.arange(80, dtype=np.float32), "b": np.zeros(80)})
    model = TBoostRegressor(**OPTIONS).fit(x, x["a"].to_numpy())
    query = x.select("b", "a")
    before = model.predict(query)
    with pytest.raises(ValueError):
        model.fit(x, x["a"].to_numpy()[:-1])
    if model.__sklearn_is_fitted__():
        np.testing.assert_array_equal(model.predict(query), before)
        assert list(model.feature_names_in_) == ["a", "b"]
    else:
        with pytest.raises((ValueError, AttributeError)):
            model.predict(query)


@pytest.mark.parametrize("channels", [None, ["class_freq"], ["mean", "count", "class_freq"]])
@pytest.mark.parametrize("bags", [1, 2])
def test_bug003_multiclass_encoder_excludes_holdout_labels(channels, bags):
    x = np.empty((160, 0), dtype=np.float32)
    cats = [["a"] * 40 + ["b"] * 40 + ["c"] * 40 + ["heldout_only"] * 40]
    holdout = [False] * 120 + [True] * 40
    y = np.repeat([0, 1, 2, 0], 40).astype(np.float32)
    changed = y.copy()
    changed[120:] = 2
    booster = _Booster(n_trees=1, n_jobs=1, n_bags=bags, cat_smooth=0,
                      cat_min_data_per_group=0, cat_channels=channels,
                      cat_count_min_levels=0, cat_class_freq_min_levels=0)
    encoders = []
    for target in (y, changed):
        model = booster.fit_multiclass(x, target, 3, ["a", "b", "c"], cat_x=cats,
                                      es_holdout=holdout)
        encoders.append(json.loads(model.to_json())["model"]["classes"][0]["schema"]["cat_encoders"])
    assert encoders[0] == encoders[1]


@pytest.mark.parametrize("objective", ["logistic", "poisson"])
@pytest.mark.parametrize("varying", [False, True])
def test_bug008_guard_intercept_balances_weighted_response(objective, varying):
    rng = np.random.default_rng(0)
    raw = rng.normal(size=100) if varying else np.zeros(100)
    w = rng.uniform(0.5, 2, 100)
    y = np.r_[np.ones(10), np.zeros(90)]
    shifted = _guard_reanchor(objective, raw, y, w, True)
    assert np.dot(w, _guard_mu(objective, shifted)) == pytest.approx(np.dot(w, y), rel=1e-9)
    np.testing.assert_array_equal(_guard_reanchor(objective, raw, y, w, False), raw)


@pytest.mark.parametrize("marker", ["t-boost-multiclass", "t-boost-multiclass-tables", "t-boost-tables"])
@pytest.mark.parametrize("classifier", [False, True])
def test_bug011_json_dispatch_reads_discriminator(marker, classifier):
    x = pl.DataFrame({marker: np.arange(80, dtype=np.float32)})
    kind = TBoostClassifier if classifier else TBoostRegressor
    y = np.repeat([0, 1], 40) if classifier else np.arange(80, dtype=np.float32)
    model = kind(**OPTIONS).fit(x, y)
    restored = kind.from_json(model.to_json())
    np.testing.assert_array_equal(restored.predict(x), model.predict(x))


@pytest.mark.parametrize("params", [dict(early_stopping=20), dict(early_stopping=1.2),
                                    dict(prune_size_penalty=3.0)])
def test_bug015_set_params_resolves_merged_aliases(params):
    constructed = TBoostRegressor(**params)
    updated = TBoostRegressor().set_params(**params)
    assert updated.get_params() == constructed.get_params()
    assert clone(updated).get_params() == updated.get_params()


@pytest.mark.parametrize("objective", ["squared-error", "l2", "regression", "SQUARED_ERROR"])
def test_bug016_objective_alias_has_identity_explanation_semantics(objective):
    x = np.arange(80, dtype=np.float32).reshape(-1, 1)
    model = TBoostRegressor(objective=objective, **OPTIONS).fit(x, x[:, 0] + 1)
    assert model.link == "identity"
    model.predict_contributions(x[:2])
    assert TBoostRegressor.from_bytes(model.to_bytes()).link == "identity"


def test_bug016_case_insensitive_poisson_reporting_uses_exposure():
    x = np.arange(80, dtype=np.float32).reshape(-1, 1)
    e = np.full(80, 2, dtype=np.float32)
    model = TBoostRegressor(objective="POISSON", **OPTIONS).fit(x, np.ones(80), exposure=e)
    report = model.actual_vs_expected(x, np.ones(80), exposure=e)[0]
    assert sum(report["expected"]) == pytest.approx(np.dot(model.predict(x), e), rel=1e-6)


@pytest.mark.parametrize("labels", [[1, 2], [-1, 1], [False, True], ["a", "b"]])
def test_bug017_binary_ae_counts_encoded_positive_labels(labels):
    x = np.arange(80, dtype=np.float32).reshape(-1, 1)
    y = np.repeat(labels, 40)
    model = TBoostClassifier(**OPTIONS).fit(x, y)
    report = model.actual_vs_expected(x, y)[0]
    assert sum(report["actual"]) == 40


def test_bug018_pruning_report_includes_deployed_factored_effects():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(300, 4)).astype(np.float32)
    y = 2 * np.prod(np.sign(x[:, :3]), axis=1) + .7 * x[:, 3] + .1 * rng.normal(size=300)
    model = TBoostRegressor(n_trees=10, n_bags=1, max_depth=6, interaction_gain_hurdle=0.,
        validation_fraction=None, n_jobs=1, prune_selector="fold_vote", band_tolerance=None,
        graduate=False).fit(x, y)
    bank = json.loads(model._model.to_json())["model"]["bank"]
    factored = {tuple(f["u"]) for f in bank["factored"]}
    assert factored, "fixture must deploy a factored effect"
    assert factored <= {tuple(u) for u in model.pruning_report_["deployed"]}
    assert not factored.intersection(map(tuple, model.pruning_report_["kept_not_deployed"]))


def _fork_prediction(model, x, queue, width):
    model.n_jobs = width
    try:
        queue.put(("predictions", model.predict(x).tolist()))
    except ValueError as error:
        queue.put(("error", str(error)))


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="requires fork")
@pytest.mark.parametrize("width", [None, 2])
def test_bug022_serving_after_fork_completes(width):
    x = np.arange(80, dtype=np.float32).reshape(-1, 1)
    model = TBoostRegressor(**{**OPTIONS, "n_jobs": 2}).fit(x, x[:, 0])
    query = np.tile(x, (1000, 1))
    expected = model.predict(query)[:2]
    context = mp.get_context("fork")
    queue = context.Queue()
    child = context.Process(target=_fork_prediction, args=(model, query, queue, width))
    child.start()
    try:
        status, actual = queue.get(timeout=10)
        child.join(5)
        assert child.exitcode == 0
        if status == "error":
            assert "fork" in actual and "spawn" in actual
        else:
            np.testing.assert_array_equal(actual[:2], expected)
    finally:
        if child.is_alive():
            child.terminate()
            child.join(5)
        queue.close()


@pytest.mark.parametrize("method", ["predict_proba", "decision_function"])
def test_bug023_multiclass_serving_validates_thread_budget(method):
    x = np.arange(180, dtype=np.float32).reshape(-1, 1)
    model = TBoostClassifier(**OPTIONS).fit(x, np.repeat(np.arange(3), 60))
    model.n_jobs = 0
    with pytest.raises(ValueError):
        getattr(model, method)(x)


@pytest.mark.parametrize("train_container", ["numpy", "pandas", "polars"])
def test_bug024_float32_categories_keep_identity_across_containers(train_container):
    x = np.tile(np.array([.1, .2], dtype=np.float32), 100).reshape(-1, 1)
    frames = dict(numpy=x, pandas=pd.DataFrame(x, columns=["f"]),
                  polars=pl.DataFrame({"f": x[:, 0]}))
    model = TBoostRegressor(**{**OPTIONS, "categorical_features": [0], "n_trees": 20,
                               "min_data_in_leaf": 1}).fit(frames[train_container], np.tile([0., 10.], 100))
    expected = model.predict(frames[train_container])
    assert abs(expected[0] - expected[1]) > 1
    for frame in frames.values():
        np.testing.assert_array_equal(model.predict(frame), expected)


def test_bug024_pandas_category_routing_is_batch_size_invariant():
    x = pd.DataFrame({"cat": pd.Categorical(np.tile(np.array([.1, .2], np.float32), 100))})
    model = TBoostRegressor(**{**OPTIONS, "categorical_features": [0]}).fit(x, np.tile([0., 10.], 100))
    expected = model.predict(x)[1]
    for size in (1, 63, 64, 100):
        query = pd.concat([x.iloc[[1]]] * size, ignore_index=True)
        np.testing.assert_array_equal(model.predict(query), np.full(size, expected))


def test_bug024_missing_category_is_independent_of_optional_pandas():
    scenario = """
import sys
if sys.argv[1] == 'blocked':
    sys.modules['pandas'] = None
import numpy as np
from t_boost._ingest import _cat_level
assert _cat_level(np.float32('nan')) == _cat_level(None)
"""
    for availability in ("available", "blocked"):
        result = subprocess.run([sys.executable, "-c", scenario, availability],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("format", ["bytes", "json", "pickle"])
@pytest.mark.parametrize("classifier", [False, True])
@pytest.mark.parametrize("container", ["pandas_float32", "polars_string"])
def test_bug024_legacy_category_encoding_survives_load_resave_and_refit(monkeypatch, format, classifier, container):
    import t_boost.sklearn as sk
    from t_boost._ingest import _cat_level

    if container == "pandas_float32":
        x = pd.DataFrame({"cat": np.tile(np.array([.1, .2], np.float32), 100)})
    else:
        x = pl.DataFrame({"cat": ["ordinary", "__t_boost_missing__:literal"] * 100})
    kind = TBoostClassifier if classifier else TBoostRegressor
    y = np.tile([0, 1] if classifier else [0., 10.], 100)
    # Reproduce the v2 writer's ingestion at its stringification seam. Native
    # category maps store precisely these strings, with no original dtype.
    with monkeypatch.context() as patch:
        patch.setattr(sk, "_cat_level", lambda v, **kw: _cat_level(v, legacy=True))
        import t_boost._ingest as ingest
        patch.setattr(ingest, "_cat_level", lambda v, **kw: _cat_level(v, legacy=True))
        model = kind(**{**OPTIONS, "n_trees": 20, "categorical_features": [0]}).fit(x, y)
        expected = model.predict(x)
        assert expected[0] != expected[1]
        if format == "pickle":
            state = model.__getstate__()
            state[sk._PICKLE_SCHEMA_KEY] = 2
            state.pop("_categorical_encoding_version_", None)
        else:
            doc = json.loads(model.to_json())
            doc["schema_version"] = 2
            doc.pop("categorical_encoding_version")
            if format == "json":
                blob = json.dumps(doc)
            else:
                native = model._model.to_bytes()
                doc.pop("model")
                header = json.dumps(doc).encode()
                blob = b"TBP1" + len(header).to_bytes(4, "big") + header + native
    if format == "pickle":
        restored = kind.__new__(kind)
        restored.__setstate__(state)
    else:
        restored = getattr(kind, "from_" + format)(blob)
    for size in (2, 64, 200):
        np.testing.assert_array_equal(restored.predict(x[:size]), expected[:size])
    for again in (kind.from_json(restored.to_json()), kind.from_bytes(restored.to_bytes()),
                  pickle.loads(pickle.dumps(restored))):
        np.testing.assert_array_equal(again.predict(x), expected)
    restored.fit(x, y)
    fresh = kind(**{**OPTIONS, "n_trees": 20, "categorical_features": [0]}).fit(x, y)
    np.testing.assert_array_equal(restored.predict(x), fresh.predict(x))
    assert restored._categorical_encoding_version_ == 3


@pytest.mark.parametrize("format", ["bytes", "json", "pickle"])
@pytest.mark.parametrize("classes", [[1, 2**63 + 1], [1, 2**63 + 1, 2**63 + 2]])
def test_bug025_unsigned_class_labels_roundtrip_exactly(format, classes):
    x = np.tile(np.arange(len(classes), dtype=np.float32), 100).reshape(-1, 1)
    y = np.tile(np.array(classes, dtype=np.uint64), 100)
    model = TBoostClassifier(**OPTIONS).fit(x, y)
    restored = (pickle.loads(pickle.dumps(model)) if format == "pickle" else
                getattr(TBoostClassifier, "from_" + format)(getattr(model, "to_" + format)()))
    np.testing.assert_array_equal(restored.classes_, model.classes_)
    assert restored.classes_.dtype == model.classes_.dtype
    np.testing.assert_array_equal(restored.predict(x), model.predict(x))


@pytest.mark.parametrize("format", ["bytes", "json"])
@pytest.mark.parametrize("classes", [[1, 2**63 + 1], [1, 2**63 + 1, 2**63 + 2]])
def test_bug025_legacy_integer_labels_load_without_loss(format, classes):
    x = np.tile(np.arange(len(classes), dtype=np.float32), 100).reshape(-1, 1)
    y = np.tile(np.array(classes, dtype=np.uint64), 100)
    model = TBoostClassifier(**OPTIONS).fit(x, y)
    if format == "json":
        header = json.loads(model.to_json())
    else:
        blob = model.to_bytes()
        size = int.from_bytes(blob[4:8], "big")
        header = json.loads(blob[8:8 + size])
    header["schema_version"] = 2
    header.pop("classes_dtype")
    header.pop("categorical_encoding_version")
    if format == "json":
        restored = TBoostClassifier.from_json(json.dumps(header))
    else:
        encoded = json.dumps(header).encode()
        restored = TBoostClassifier.from_bytes(blob[:4] + len(encoded).to_bytes(4, "big") + encoded + blob[8 + size:])
    assert restored.classes_.tolist() == classes
    np.testing.assert_array_equal(restored.predict(x), model.predict(x))
    np.testing.assert_array_equal(TBoostClassifier.from_bytes(restored.to_bytes()).predict(x), model.predict(x))


def _runtime_only(scenario):
    blocker = """
import sys
class BlockSklearn:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'sklearn' or fullname.startswith('sklearn.'):
            raise ImportError('runtime-only regression')
sys.meta_path.insert(0, BlockSklearn())
import numpy as np
from t_boost import TBoostClassifier, TBoostRegressor
from t_boost._compat import SKLEARN_AVAILABLE
assert not SKLEARN_AVAILABLE
"""
    result = subprocess.run([sys.executable, "-c", blocker + scenario], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_bug026_runtime_only_classifier_rejects_nonfinite_labels():
    _runtime_only("""
for labels in ([0., np.inf], [0., 1., np.nan], [0., 1., np.inf], [0., np.nan]):
    y = np.tile(labels, 30)
    x = np.arange(len(y), dtype=np.float32).reshape(-1, 1)
    try:
        TBoostClassifier(n_trees=1, n_bags=1, prune=False, validation_fraction=None, n_jobs=1).fit(x, y)
    except ValueError:
        pass
    else:
        raise AssertionError('accepted nonfinite class labels')
""")


@pytest.mark.parametrize("format", ["bytes", "json", "pickle"])
def test_bug027_loaded_partial_export_mass_requires_missing_exposure(format):
    x = np.tile(np.array([0., 1.], np.float32), 100).reshape(-1, 1)
    e = np.tile(np.array([1., 10.], np.float32), 100)
    model = TBoostRegressor(**{**OPTIONS, "objective": "poisson"}).fit(x, np.tile([1., 30.], 100), exposure=e)
    restored = (pickle.loads(pickle.dumps(model)) if format == "pickle" else
                getattr(TBoostRegressor, "from_" + format)(getattr(model, "to_" + format)()))
    if format == "pickle":
        assert json.loads(restored.tables(x, sample_weight=np.ones(200))) == json.loads(
            model.tables(x, sample_weight=np.ones(200)))
    else:
        with pytest.raises(ValueError, match="exposure|mass"):
            restored.tables(x, sample_weight=np.ones(200, dtype=np.float32))
    expected = json.loads(model.tables(x, sample_weight=np.ones(200), exposure=e))
    assert json.loads(restored.tables(x, sample_weight=np.ones(200), exposure=e)) == expected


@pytest.mark.parametrize("indices", [[], (), np.array([], dtype=int)])
def test_bug028_empty_categorical_indices_mean_no_categories(indices):
    x = np.arange(80, dtype=np.float32).reshape(-1, 1)
    model = TBoostRegressor(**{**OPTIONS, "categorical_features": indices}).fit(x, x[:, 0])
    assert np.isfinite(model.predict(x)).all()


def test_bug029_no_argument_set_params_preserves_fitted_model():
    x = np.arange(80, dtype=np.float32).reshape(-1, 1)
    model = TBoostRegressor(**OPTIONS).fit(x, x[:, 0])
    before = model.to_bytes()
    assert model.set_params() is model
    assert model.__sklearn_is_fitted__()
    assert model.to_bytes() == before


@pytest.fixture(scope="module")
def banded_model():
    rng = np.random.default_rng(80)
    x = rng.uniform(-2, 2, (600, 2)).astype(np.float32)
    y = (.5 * (x[:, 0] + x[:, 1]) + np.sin(3*x[:, 0])*np.cos(2*x[:, 1])
         + .1*rng.normal(size=600)).astype(np.float32)
    model = TBoostRegressor(n_trees=40, n_bags=2, max_depth=4, interaction_gain_hurdle=0.,
        validation_fraction=None, n_jobs=1, band_tolerance=.75, graduate=False).fit(x, y)
    return model, x, y


def test_bug030_rebanding_respects_existing_band_maps(banded_model):
    model, x, _ = banded_model
    raw = np.asarray(model._model.predict_raw(x), dtype=np.float64)
    banded, report = model._model.band(x, np.ones(600), np.ones(600), raw, 1.,
        tolerance=1e-8, pseudo_rows=1000, n_jobs=1)
    report = json.loads(report)
    actual_mse = np.mean((np.asarray(banded.predict_raw(x), dtype=np.float64) - raw)**2)
    assert actual_mse <= report["budget"] + 1e-12
    assert actual_mse == pytest.approx(report["combined_mse"], abs=1e-9)


@pytest.mark.parametrize("objective,power", [("poisson", 1.), ("gamma", 2.), ("tweedie", 1.5)])
def test_bug034_exposure_intercept_solves_objective_gradient(objective, power):
    x = np.ones((100, 1), dtype=np.float32)
    y = np.full(100, 10., dtype=np.float32)
    e = np.tile([1., 10.], 50)
    model = TBoostRegressor(**{**OPTIONS, "objective": objective, "reanchor": False}).fit(x, y, exposure=e)
    optimum = np.sum(y * e**(1-power)) / np.sum(e**(2-power))
    np.testing.assert_allclose(model.predict(x), optimum, rtol=1e-6)


def test_bug039_zero_weight_uninformative_feature_does_not_abort_fit():
    x = np.column_stack([np.r_[np.arange(300), np.full(100, np.nan)], np.arange(400)]).astype(np.float32)
    w = np.r_[np.zeros(300), np.ones(100)].astype(np.float32)
    model = TBoostRegressor(**OPTIONS).fit(x, x[:, 1], sample_weight=w)
    control = TBoostRegressor(**OPTIONS).fit(x[:, 1:], x[:, 1], sample_weight=w)
    np.testing.assert_array_equal(model.predict(x), control.predict(x[:, 1:]))


def test_bug040_training_scores_agree_with_table_accumulation_at_large_intercept():
    x = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32), (20, 1))
    y = np.tile(np.array([6, 2, 2, 0], np.float32), 20) + np.float32(10_000_000)
    model = TBoostRegressor(**{**OPTIONS, "n_trees": 20, "learning_rate": .2,
                               "leaf_refine_steps": 0, "reanchor": False})
    native = model._new_booster(fit_pool_width=1).fit(x, y)
    tables = model._full_tables(native, x, y, None, None, None)
    np.testing.assert_allclose(native.predict_raw(x), tables.predict_raw(x), rtol=0, atol=1)


@pytest.mark.parametrize("booster", [dict(dart_drop_rate=1e-30), dict(ridge_refit_l2=1.)])
@pytest.mark.parametrize("validation", [None, .2])
def test_bug040_optional_boosters_preserve_small_updates(booster, validation):
    x = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32), (20, 1))
    y = np.tile(np.array([6, 2, 2, 0], np.float32), 20)
    model = TBoostRegressor(**{**OPTIONS, **booster, "n_trees": 100, "learning_rate": .2,
        "leaf_refine_steps": 0, "reanchor": False, "validation_fraction": validation}).fit(x, y + 1e7)
    np.testing.assert_allclose(model.predict(x) - 1e7, y, atol=1, rtol=0)


def test_bug041_pricing_ae_labels_use_aggregation_grid(banded_model):
    model, x, y = banded_model
    report = model.pricing_report(x, y)
    for axis in report["actual_vs_expected"]:
        assert len(axis["rows"]) == axis["axis"]["cells"]
        assert len(axis["mass"]) == axis["axis"]["cells"]


def test_bug042_multiclass_dataframe_handles_intercept_only_class():
    x = np.tile(np.array([-1, -1, -1, 1, 1, 1], np.float32), 100).reshape(-1, 1)
    y = np.tile([0, 1, 1, 0, 2, 2], 100)
    model = TBoostClassifier(**{**OPTIONS, "n_trees": 10, "colsample_bytree": 1,
                                "leaf_refine_steps": 0}).fit(x, y)
    records = model.predict_contributions(x[:2])
    assert any(not c["contributions"] for c in records[0]["classes"])
    frame = model.predict_contributions(x[:2], return_format="dataframe")
    assert frame["class"].n_unique() == len(model.classes_)
    assert frame["row_index"].n_unique() == 2


def test_bug043_gini_ties_are_neutral_and_permutation_invariant():
    y = np.array([0., 0., 1., 1.])
    assert ordered_gini(y, np.ones(4)) == pytest.approx(0., abs=1e-12)
    rng = np.random.default_rng(1)
    y = (rng.random(200) < .3).astype(float)
    scores = np.round(rng.random(200), 1)
    perm = rng.permutation(200)
    expected = 2 * roc_auc_score(y, scores) - 1
    assert ordered_gini(y, scores) == pytest.approx(expected, abs=1e-12)
    assert ordered_gini(y[perm], scores[perm]) == pytest.approx(expected, abs=1e-12)


def test_bug044_categorical_export_preserves_rare_routing_identity():
    y = np.array([0]*100 + [2]*100 + [4]*2 + [6]*20, np.float32)
    exports = []
    for rare in ("rareA", "rareB"):
        x = pl.DataFrame({"cat": ["a"]*100 + ["b"]*100 + [rare]*2 + [None]*20})
        model = TBoostRegressor(**{**OPTIONS, "n_trees": 30}).fit(x, y)
        exports.append(json.loads(model.tables(x)))
    assert exports[0] != exports[1], "different serve routing must be present in a rating export"


def test_bug044_literal_rare_label_does_not_collide_with_pooled_level():
    x = pl.DataFrame({"c": ["<rare>"]*100 + ["a"]*100 + ["thin"]*2})
    y = np.array([0]*100 + [2]*100 + [8]*2, np.float32)
    model = TBoostRegressor(**{**OPTIONS, "n_trees": 30}).fit(x, y)
    axis = json.loads(model.tables(x))["tables"][0]["axes"][0]
    assert len(axis["levels"]) == len({level["label"] for level in axis["levels"]})


@pytest.mark.parametrize("channels", [None, ["mean", "count"]])
@pytest.mark.parametrize("band_tolerance", [None, .02])
def test_bug044_export_metadata_alone_routes_categories(channels, band_tolerance):
    x = pl.DataFrame({"cat": ["a"]*100 + ["b"]*100 + ["thin"]*2 + [None]*20 + ["<rare>"]*100})
    y = np.array([0]*100 + [2]*100 + [4]*2 + [6]*20 + [8]*100, np.float32)
    model = TBoostRegressor(**{**OPTIONS, "n_trees": 30, "cat_channels": channels,
        "cat_count_min_levels": 0, "band_tolerance": band_tolerance}).fit(x, y)
    bank = json.loads(model.tables(x))
    assert bank["tables"] and not bank["factored"]
    labels = ["a", "b", "thin", "never-seen", None, "<rare>"]
    predicted = []
    for label in labels:
        value = bank["f0"]
        key = "__t_boost_missing__" if label is None else label
        for table in bank["tables"]:
            axis = table["axes"][0]
            cells = {member: level["cell"] for level in axis["levels"] for member in level["members"]}
            cell = cells.get(key, axis["default_cell"])
            assert cell is not None
            value += table["values"][cell]
        predicted.append(value)
    np.testing.assert_allclose(predicted, model.predict_raw(pl.DataFrame({"cat": labels})), rtol=0, atol=1e-5)


@pytest.mark.parametrize("metric,y,pred,weight,extra", [
    (mean_poisson_deviance, [np.nan, 1.], [1., 1.], None, {}),
    (mean_gamma_deviance, [1., 1.], [np.inf, 1.], None, {}),
    (mean_poisson_deviance, [1., 1.], [1., 1.], [np.nan, 1.], {}),
    (mean_tweedie_deviance, [2.], [5.], None, {"power": np.nan}),
    (mean_tweedie_deviance, [2.], [5.], None, {"power": np.inf}),
    (mean_tweedie_deviance, [np.inf], [1.], None, {"power": 0}),
])
def test_bug045_deviance_rejects_nonfinite_inputs(metric, y, pred, weight, extra):
    with pytest.raises(ValueError):
        metric(y, pred, weight, **extra)


def test_bug046_negative_expected_total_has_valid_ae():
    x = np.arange(20, dtype=np.float32).reshape(-1, 1)
    y = np.full(20, -3., np.float32)
    model = TBoostRegressor(**OPTIONS).fit(x, y)
    report = model.actual_vs_expected(x, y)[0]
    populated = np.asarray(report["mass"]) > 0
    np.testing.assert_allclose(np.asarray(report["ae"])[populated], 1.)
    pricing = model.pricing_report(x, y)["actual_vs_expected"][0]
    np.testing.assert_allclose(np.asarray(pricing["ae"], dtype=float)[populated], 1.)


def test_bug047_empty_ae_returns_empty_aggregates():
    x = np.arange(20, dtype=np.float32).reshape(-1, 1)
    model = TBoostRegressor(**OPTIONS).fit(x, x[:, 0])
    report = model.actual_vs_expected(x[:0], x[:0, 0])
    assert len(report) == 1
    assert sum(report[0]["mass"]) == 0


@pytest.mark.parametrize("bags", [2, 8])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("validation", [None, .2])
@pytest.mark.parametrize("fraction", [.8, 1.])
def test_bug049_bags_without_positive_weight_do_not_abort_valid_fit(bags, seed, validation, fraction):
    x = np.arange(100, dtype=np.float32).reshape(-1, 1)
    w = np.zeros(100, np.float32)
    w[17] = 1
    model = TBoostRegressor(**{**OPTIONS, "n_trees": 3, "n_bags": bags, "seed": seed, "validation_fraction": validation, "bag_subsample": fraction}).fit(x, x[:, 0], sample_weight=w)
    np.testing.assert_allclose(model.predict(x), 17., atol=1e-5)


def test_bug054_incremental_poisson_preserves_scores_beyond_clamp():
    x = np.repeat(np.array([0, 1], np.float32), 100).reshape(-1, 1)
    y = np.repeat(np.array([1e12, 7.9e13], np.float32), 100)
    predictions = []
    for incremental in (False, True):
        model = TBoostRegressor(**{**OPTIONS, "objective": "poisson", "n_trees": 20,
            "learning_rate": .1, "leaf_refine_steps": 0, "reanchor": False,
            "incremental_mu": incremental}).fit(x, y)
        predictions.append(model.predict(x))
    np.testing.assert_allclose(*predictions, rtol=1e-5)


def test_bug055_runtime_only_score_validates_target_shape_and_singleton():
    _runtime_only("""
x = np.arange(20, dtype=np.float32).reshape(-1, 1)
options = dict(n_trees=1, n_bags=1, prune=False, graduate=False, validation_fraction=None, n_jobs=1)
for model, y in ((TBoostRegressor(**options), np.zeros(20)), (TBoostClassifier(**options), np.tile([0, 1], 10))):
    model.fit(x, y)
    for bad_y in (np.array([0.]), np.column_stack([y, y])[:10]):
        try:
            model.score(x, bad_y)
        except ValueError:
            pass
        else:
            raise AssertionError('score accepted incompatible target shape')
    if isinstance(model, TBoostRegressor):
        assert np.isnan(model.score(x[:1], y[:1]))
""")


@pytest.mark.parametrize("prune", [False, True])
def test_bug056_binary_ae_applies_logit_exposure_offset(prune):
    x = np.ones((200, 1), np.float32)
    y = np.repeat([0, 1], 100)
    e = np.repeat([1., 4.], 100).astype(np.float32)
    model = TBoostClassifier(**{**OPTIONS, "prune": prune}).fit(x, y, exposure=e)
    raw = model.decision_function(x).astype(float)
    expected = np.sum(1 / (1 + np.exp(-(raw + np.log(e)))))
    report = model.actual_vs_expected(x, y, exposure=e)[0]
    assert sum(report["expected"]) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("container", ["polars", "numpy"])
def test_bug057_literal_missing_sentinel_is_distinct_or_rejected(container):
    values = [None]*100 + ["__t_boost_missing__"]*100
    x = pl.DataFrame({"cat": values}) if container == "polars" else np.array(values, object).reshape(-1, 1)
    try:
        model = TBoostRegressor(**{**OPTIONS, "categorical_features": [0], "n_trees": 10}).fit(x, np.repeat([0., 10.], 100))
    except ValueError as error:
        assert "reserved" in str(error).lower() or "sentinel" in str(error).lower()
    else:
        pred = model.predict(x)
        assert pred[-1] - pred[0] > 1, "literal category merged with actual missing values"


@pytest.mark.parametrize("format", ["json", "bytes"])
@pytest.mark.parametrize("field,value", [("classes_", ["A", "B", "C"]),
                                        ("classes_", []),
                                        ("feature_names_in_", ["noise", "signal"])])
def test_bug058_envelope_rejects_metadata_inconsistent_with_native_model(format, field, value):
    x = pl.DataFrame({"signal": np.tile([0., 1.], 100), "noise": np.zeros(200)})
    model = TBoostClassifier(**OPTIONS).fit(x, np.tile([0, 1], 100))
    if format == "json":
        doc = json.loads(model.to_json())
        doc[field] = value
        blob = json.dumps(doc)
    else:
        blob = model.to_bytes()
        length = int.from_bytes(blob[4:8], "big")
        header = json.loads(blob[8:8+length])
        header[field] = value
        encoded = json.dumps(header).encode()
        blob = blob[:4] + len(encoded).to_bytes(4, "big") + encoded + blob[8+length:]
    with pytest.raises(SerializationError):
        getattr(TBoostClassifier, "from_" + format)(blob)


def test_bug059_runtime_only_repr_supports_numpy_parameters():
    _runtime_only("""
for params in (dict(monotone_constraints=np.array([1, 0])), dict(categorical_features=np.array([True, False]))):
    assert isinstance(repr(TBoostRegressor(**params)), str)
""")


@pytest.mark.parametrize("correlation", [0, 1, -1])
def test_bug064_joint_variance_shares_use_total_model_variance(correlation):
    x = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32), (20, 1))
    model = TBoostRegressor(**{**OPTIONS, "n_trees": 100, "max_depth": 3,
        "max_interaction_order": 1, "leaf_refine_steps": 0, "reanchor": False}).fit(x, x[:, 0]+0.5*x[:, 1])
    query = x.copy()
    if correlation:
        query[:, 1] = query[:, 0] if correlation > 0 else 1-query[:, 0]
    weights = np.tile([1., 2., 3., 4.], 20).astype(np.float32)
    export = json.loads(model.tables(query, ref_measure="joint", sample_weight=weights))
    raw = model.predict_raw(query).astype(float)
    mean = np.average(raw, weights=weights)
    variance = np.average((raw-mean)**2, weights=weights)
    for table in export["tables"]:
        assert table["sobol"] == pytest.approx(table["variance"]/variance, rel=2e-5, abs=1e-8)


@pytest.mark.parametrize("classifier", [False, True])
@pytest.mark.parametrize("exposure", [False, True])
def test_bug064_joint_exports_preserve_fit_mass_and_require_it_after_loading(classifier, exposure):
    x = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32), (20, 1))
    mass = np.tile([1., 2., 3., 10.], 20).astype(np.float32)
    key = "exposure" if exposure else "sample_weight"
    kind = TBoostClassifier if classifier else TBoostRegressor
    y = x[:, 0] if classifier else x[:, 0] + 2 * x[:, 1]
    # Exposure is refused under squared_error (R8), so the exposure regressor is a rate model.
    options = {**OPTIONS, "objective": "poisson"} if exposure and not classifier else OPTIONS
    model = kind(**options).fit(x, y, **{key: mass})
    expected = json.loads(model.tables(x, ref_measure="joint", **{key: mass}))
    assert json.loads(model.tables(x, ref_measure="joint")) == expected
    for table in expected["tables"]:
        assert sum(table["support"]) == pytest.approx(mass.sum())
    for loaded in (kind.from_json(model.to_json()), kind.from_bytes(model.to_bytes())):
        with pytest.raises(ValueError, match=key):
            loaded.tables(x, ref_measure="joint")
        assert json.loads(loaded.tables(x, ref_measure="joint", **{key: mass})) == expected
