from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from t_boost.sklearn import PrecisionWarning, TBoostClassifier, TBoostRegressor
from _artifact import assert_exports_close, ensemble_fit, model_bytes


def regression_fixture() -> tuple[np.ndarray, np.ndarray]:
    x = np.array(
        [[float(i % 6), float((i // 2) % 5), float((i // 3) % 4)] for i in range(96)],
        dtype=np.float64,
    )
    y = np.where(x[:, 0] <= 2.0, 1.5, -0.75) + np.where(x[:, 1] <= 2.0, 0.5, -0.25)
    return x, y.astype(np.float32)


def classifier_fixture() -> tuple[np.ndarray, np.ndarray]:
    x, y_reg = regression_fixture()
    y = np.where(y_reg > np.median(y_reg), "yes", "no")
    return x, y


def mixed_categorical_fixture() -> tuple[np.ndarray, np.ndarray]:
    n = 96
    num = (np.arange(n) % 8).astype(np.float32)
    levels = np.asarray(["alpha", "beta", "gamma", "delta"], dtype=object)
    brand = levels[np.arange(n) % levels.shape[0]]
    x = np.empty((n, 2), dtype=object)
    x[:, 0] = num
    x[:, 1] = brand
    y = (
        0.15 * num
        + np.where(brand == "beta", 1.25, 0.0)
        + np.where(brand == "gamma", -0.75, 0.25)
    )
    return x, y.astype(np.float32)


def small_regressor(**kwargs) -> TBoostRegressor:
    # Minimal, NEUTRAL engine config for unit tests: pin the recommended-recipe levers OFF
    # (early stopping / leaf refinement / outer bagging / column subsampling) so each test
    # isolates the behavior it probes. The product constructor defaults enable these levers
    # (see `_BaseTBoost.__init__`); any kwarg here still overrides.
    params = dict(
        n_trees=16,
        learning_rate=0.25,
        lambda_=1.0,
        max_bin=32,
        seed=7,
        validation_fraction=None,
        early_stopping_rounds=50,
        early_stopping_adaptive=None,
        interaction_gain_hurdle=0.0,
        leaf_refine_steps=0,
        n_bags=0,
        colsample_bytree=1.0,
        prune=False,
    )
    params.update(kwargs)
    return TBoostRegressor(**params)


def test_regressor_fit_predict_serialize_and_warns_once() -> None:
    x, y = regression_fixture()
    est = small_regressor()
    with pytest.warns(PrecisionWarning):
        est.fit(x, y)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pred1 = est.predict(x)
        pred2 = est.predict(x)
    assert not [w for w in caught if issubclass(w.category, PrecisionWarning)]
    np.testing.assert_array_equal(pred1, pred2)
    # The wrapper's predict returns float64; the low-level _model.predict fills an f32 buffer.
    out = np.empty(pred1.shape[0], dtype=np.float32)
    returned = est._model.predict(x.astype(np.float32), out=out)
    assert returned is out
    np.testing.assert_array_equal(out, pred1)

    loaded = TBoostRegressor.from_bytes(est.to_bytes())
    np.testing.assert_array_equal(pred1, loaded.predict(x.astype(np.float32)))
    loaded_json = TBoostRegressor.from_json(est.to_json())
    np.testing.assert_array_equal(pred1, loaded_json.predict(x.astype(np.float32)))

    export = json.loads(est.tables(x.astype(np.float32), ref_measure="uniform"))
    assert export["mode"] == "Exact"
    assert export["link"] == "Identity"
    assert export["tables"]


# --- X ingest preserves the caller's memory order (spec §12.3, appendix X-ingest zero-copy) ---
# _as_float32_2d used to force order="C" unconditionally, which silently defeated the native
# F-contiguous fast path (raw_columns_from_array, crates/t-boost-py/src/lib.rs) for every
# estimator caller by copying F-order input into C-order before it ever reached Rust.


def test_serve_design_passes_through_an_already_fortran_float32_x_with_no_copy() -> None:
    x, y = regression_fixture()
    x_f = np.asfortranarray(x.astype(np.float32))
    assert x_f.flags["F_CONTIGUOUS"] and not x_f.flags["C_CONTIGUOUS"]
    est = small_regressor().fit(x, y)
    x32, _ = est._serve_design(x_f)
    assert x32 is x_f, "an already-float32 F-contiguous X must pass through with no copy"


def test_serve_design_passes_through_an_already_c_contiguous_float32_x_with_no_copy() -> None:
    x, y = regression_fixture()
    x_c = np.ascontiguousarray(x.astype(np.float32))
    assert x_c.flags["C_CONTIGUOUS"]
    est = small_regressor().fit(x, y)
    x32, _ = est._serve_design(x_c)
    assert x32 is x_c, "an already-float32 C-contiguous X must also pass through with no copy"


def test_fit_predict_tables_byte_identical_c_order_vs_fortran_order_x() -> None:
    # The estimator-level analogue of the native _Booster test (test_error_mapping.py); this is
    # the layer that was silently forcing everything to C-order before reaching the native fast
    # path, so it needs its own end-to-end check, not just the native one.
    x, y = regression_fixture()
    x_c = np.ascontiguousarray(x.astype(np.float32))
    x_f = np.asfortranarray(x.astype(np.float32))
    np.testing.assert_array_equal(x_c, x_f)  # same logical values, different memory layout

    est_c = small_regressor().fit(x_c, y)
    est_f = small_regressor().fit(x_f, y)
    assert est_c.to_bytes() == est_f.to_bytes()
    np.testing.assert_array_equal(est_c.predict(x_c), est_c.predict(x_f))
    assert est_c.tables(x_c, ref_measure="uniform") == est_c.tables(x_f, ref_measure="uniform")


def test_regressor_clone_set_params_and_not_fitted_contract() -> None:
    x, y = regression_fixture()
    est = small_regressor()
    cloned = clone(est)
    assert cloned.get_params()["n_trees"] == 16
    with pytest.raises(NotFittedError):
        cloned.predict(x)

    est.fit(x.astype(np.float32), y)
    est.set_params(n_trees=4)
    with pytest.raises(NotFittedError):
        est.predict(x.astype(np.float32))


def test_stale_pruning_and_graduation_reports_cleared_on_set_params_and_refit() -> None:
    # set_params's clear-list used to omit pruning_report_/graduation_report_, and a direct
    # refit (bypassing set_params) never deleted them either — so a prune=True fit followed by
    # set_params(prune=False) + refit left the DISCARDED fit's reports still attached.
    rng = np.random.default_rng(0)
    n = 1200
    xn = rng.normal(size=(n, 3)).astype(np.float32)
    mu = np.exp(0.3 * xn[:, 0] + 0.2 * xn[:, 1])
    yn = rng.poisson(mu).astype(np.float32)
    common = dict(objective="poisson", n_trees=60, n_bags=1, seed=0, prune=True, graduate=True)

    # set_params alone (no refit yet) must drop both reports immediately.
    est = TBoostRegressor(**common).fit(xn, yn)
    assert hasattr(est, "pruning_report_") and hasattr(est, "graduation_report_")
    est.set_params(prune=False)
    assert not hasattr(est, "pruning_report_")
    assert not hasattr(est, "graduation_report_")

    # A direct refit with prune/graduate turned off must not leave the PREVIOUS fit's reports.
    est2 = TBoostRegressor(**common).fit(xn, yn)
    assert hasattr(est2, "pruning_report_") and hasattr(est2, "graduation_report_")
    est2.set_params(prune=False, graduate=False)
    est2.fit(xn, yn)
    assert not hasattr(est2, "pruning_report_")
    assert not hasattr(est2, "graduation_report_")


def test_stale_feature_names_in_cleared_when_refitting_without_names() -> None:
    pd = pytest.importorskip("pandas")
    x, y = regression_fixture()
    frame = pd.DataFrame(x, columns=["a", "b", "c"])
    est = small_regressor().fit(frame, y)
    assert est.feature_names_in_.tolist() == ["a", "b", "c"]

    # Refitting directly on a plain ndarray (no names) must drop the stale feature_names_in_,
    # not leave it describing the PREVIOUS (DataFrame) fit.
    est.fit(x.astype(np.float32), y)
    assert not hasattr(est, "feature_names_in_")


def test_classifier_predict_proba_classes_and_roundtrip() -> None:
    x, y = classifier_fixture()
    clf = TBoostClassifier(
        n_trees=18,
        learning_rate=0.2,
        lambda_=1.0,
        max_bin=32,
        seed=11,
    )
    with pytest.warns(PrecisionWarning):
        clf.fit(x, y)
    assert clf.classes_.tolist() == ["no", "yes"]
    proba = clf.predict_proba(x.astype(np.float32))
    assert proba.shape == (x.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-6)
    pred = clf.predict(x.astype(np.float32))
    assert set(pred.tolist()) <= {"no", "yes"}

    loaded = TBoostClassifier.from_bytes(clf.to_bytes())
    assert loaded.classes_.tolist() == ["no", "yes"]
    np.testing.assert_array_equal(proba, loaded.predict_proba(x.astype(np.float32)))


def test_python_fit_is_thread_count_deterministic() -> None:
    x, y = regression_fixture()
    a = model_bytes(small_regressor(n_jobs=1).fit(x.astype(np.float32), y))
    b = model_bytes(small_regressor(n_jobs=2).fit(x.astype(np.float32), y))
    assert a == b


def test_credibility_floor_params_round_trip_and_path_smooth_shrinks() -> None:
    x, y = regression_fixture()
    x32 = x.astype(np.float32)
    # Credibility kwargs survive get_params / clone (the sklearn config contract).
    est = small_regressor(
        min_data_in_leaf=4,
        min_sum_hessian_in_leaf=1.0,
        min_weight_sum_in_leaf=4.0,
        path_smooth=2.0,
    )
    params = est.get_params()
    assert params["min_data_in_leaf"] == 4
    assert params["path_smooth"] == 2.0
    assert clone(est).get_params()["min_weight_sum_in_leaf"] == 4.0

    # path_smooth is value-level: same structure, but it shifts the served predictions,
    # and the model stays exactly decomposable.
    plain = small_regressor().fit(x32, y)
    floored = est.fit(x32, y)
    assert json.loads(floored.tables(x32, ref_measure="uniform"))["mode"] == "Exact"
    assert not np.allclose(plain.predict(x32), floored.predict(x32))


def test_negative_credibility_floor_is_rejected() -> None:
    x, y = regression_fixture()
    with pytest.raises(Exception):
        small_regressor(min_sum_hessian_in_leaf=-1.0).fit(x.astype(np.float32), y)


def test_nesterov_is_rejected_as_unstable() -> None:
    # AGBM/Nesterov acceleration currently diverges (momentum correction unimplemented),
    # so the Python surface refuses it loudly rather than return a blown-up model.
    x, y = regression_fixture()
    with pytest.raises(Exception, match="nesterov"):
        small_regressor(nesterov=True).fit(x.astype(np.float32), y)


def test_all_categorical_input_stays_exact_and_predicts() -> None:
    # No numeric features at all — the model is built entirely from native categoricals.
    rng = np.random.default_rng(0)
    n = 1500
    a = rng.integers(0, 5, n)
    b = rng.integers(0, 4, n)
    x = np.array([[f"A{ai}", f"B{bi}"] for ai, bi in zip(a, b)], dtype=object)
    y = (a + 2.0 * b).astype(np.float32)
    m = TBoostRegressor(
        objective="squared_error", n_trees=80, learning_rate=0.1, seed=0,
        categorical_features=[0, 1],
    ).fit(x, y)
    pred = np.asarray(m.predict(x))
    assert pred.shape == (n,) and np.isfinite(pred).all()
    exp = json.loads(m.tables(x[:128]))
    assert exp["mode"] == "Exact"


def test_empty_feature_matrix_is_rejected() -> None:
    with pytest.raises(ValueError, match="no features"):
        TBoostRegressor(n_trees=5, seed=0).fit(
            np.empty((50, 0), dtype=np.float32), np.zeros(50, dtype=np.float32)
        )


def test_booster_knobs_round_trip_and_stay_exact() -> None:
    # The §06/§09/§07 levers (ensemble, sampling, hist precision, refit, interaction order)
    # are now reachable from Python; they survive get_params/clone and stay exactly
    # decomposable (every booster is leaf-scalar / tree-alpha / intercept level).
    x, y = regression_fixture()
    x32 = x.astype(np.float32)
    est = small_regressor(
        n_bags=3,
        subsample=0.8,
        hist_precision="quantized",
        ridge_refit_l2=0.5,
        random_strength=0.1,
        reanchor=True,
        max_interaction_order=2,
    )
    params = est.get_params()
    assert params["n_bags"] == 3
    assert params["hist_precision"] == "quantized"
    assert params["max_interaction_order"] == 2
    assert clone(est).get_params()["subsample"] == 0.8

    est.fit(x32, y)
    assert json.loads(est.tables(x32, ref_measure="uniform"))["mode"] == "Exact"
    assert est.predict(x32).shape == (x32.shape[0],)


def test_new_accuracy_knobs_round_trip_and_stay_exact() -> None:
    x, y = regression_fixture()
    x32 = x.astype(np.float32)
    est = small_regressor(
        l1_leaf=0.01,
        colsample_bytree=0.67,
        learning_rate_decay=0.05,
        validation_fraction=0.2,
        early_stopping_rounds=3,
        leaf_refine_steps=1,
        leaf_refine_backtracks=3,
    )
    params = est.get_params()
    assert params["l1_leaf"] == 0.01
    assert params["colsample_bytree"] == 0.67
    assert params["learning_rate_decay"] == 0.05
    assert params["validation_fraction"] == 0.2
    assert params["early_stopping_rounds"] == 3
    assert params["leaf_refine_steps"] == 1
    assert params["leaf_refine_backtracks"] == 3
    assert clone(est).get_params()["colsample_bytree"] == 0.67

    est.fit(x32, y)
    assert json.loads(est.tables(x32, ref_measure="uniform"))["mode"] == "Exact"
    assert est.predict(x32).shape == (x32.shape[0],)


def test_new_accuracy_invalid_params_are_rejected() -> None:
    x, y = regression_fixture()
    x32 = x.astype(np.float32)
    for kwargs in (
        {"l1_leaf": -1.0},
        {"colsample_bytree": 0.0},
        {"learning_rate_decay": -0.1},
        {"validation_fraction": 0.0},
        {"validation_fraction": 0.2, "early_stopping_rounds": 0},
        {"leaf_refine_steps": 1, "leaf_refine_backtracks": 0},
        {"early_stopping_adaptive": 0.0},  # ratio must be finite and > 0 when set
        {"early_stopping_adaptive": -1.5},
        {"early_stopping_min_delta": -0.1},  # tolerance must be finite and in [0.0, 1.0)
        {"early_stopping_min_delta": 1.0},
        {"interaction_gain_hurdle": -0.5},  # hurdle must be finite and >= 0
        {"interaction_gain_hurdle": float("nan")},
        {"interaction_gain_hurdle_mode": "later_please"},
    ):
        with pytest.raises(Exception):
            small_regressor(**kwargs).fit(x32, y)


def test_graduation_gcv_roundtrips_and_is_deterministic() -> None:
    # Whittaker-Henderson graduation (post-fit, per-table GCV): params round-trip, the fit
    # produces a graduation_report_ with one entry per assessed dense table (alpha=0 = left
    # verbatim), predictions stay finite, and the whole path is seed-deterministic.
    x, y = regression_fixture()
    x, y = np.tile(x, (16, 1)), np.tile(y, 16)
    est = small_regressor(prune=True, graduate=True)
    assert est.get_params()["graduate"] is True
    assert est.get_params()["graduation_alpha"] is None
    assert clone(est).get_params()["graduate"] is True
    est.fit(x, y)
    rep = est.graduation_report_
    assert isinstance(rep, list) and rep, "graduation report missing"
    assert all(r["alpha"] >= 0.0 for r in rep)
    p1 = est.predict(x)
    assert np.isfinite(np.asarray(p1)).all()
    est2 = small_regressor(prune=True, graduate=True).fit(x, y)
    assert np.array_equal(np.asarray(p1), np.asarray(est2.predict(x)))
    # fixed-alpha override forces smoothing on an ELIGIBLE (>=3 supported cells) table; build a
    # one-feature signal that pruning keeps as a wide dense main-effect table.
    rng = np.random.default_rng(3)
    xw = (np.arange(1600) % 20).astype(np.float64)[:, None]
    yw = (
        np.sin(xw[:, 0] / 3.0)
        + 0.1 * xw[:, 0]
        + rng.normal(scale=0.2, size=1600)
    ).astype(np.float32)
    est3 = small_regressor(prune=True, graduate=True, graduation_alpha=1e4).fit(xw, yw)
    rep3 = est3.graduation_report_
    # forced heavy smoothing either sticks (alpha recorded) or trips the train-deviance
    # no-harm guard, which reverts wholesale and marks every entry rejected — never silence.
    assert rep3
    assert any(r["alpha"] > 0 for r in rep3) or all(r.get("rejected") for r in rep3)
    assert np.isfinite(np.asarray(est3.predict(xw))).all()


def test_wh_graduation_never_smooths_the_populated_missing_bin() -> None:
    # H-finding: bin 0 (numeric axis) is the RESERVED missing slot regardless of support --
    # missing is not ordinally adjacent to the lowest real numeric bin, so it must never enter
    # the D2 difference chain, even when it IS populated (NaNs present at train). Exercised
    # directly against `_graduate_table_model`'s orchestration with a mocked table_model (the
    # `apply_graduation` Rust call is mocked out, so this needs no .so rebuild — it tests the
    # pure-Python `start`/`r0`/`c0` construction the finding is about), covering both the 1-D
    # and mixed-axis 2-D branches.
    import unittest.mock as um

    est = small_regressor(prune=True, graduate=True, graduation_alpha=1.0)

    # 1-D: bin 0 (missing) is an outlier (9.0) against an otherwise near-flat, populated run.
    v1 = [9.0, 1.0, 1.1, 0.9, 1.05, 5.0, 1.0]
    s1 = [12.0, 20.0, 22.0, 19.0, 21.0, 18.0, 20.0]  # bin 0 IS populated
    # 2-D: axis 0 numeric (row 0 = missing, populated, an outlier), axis 1 categorical (no
    # missing convention at all -- graduation smooths along the numeric direction only).
    shape2 = [4, 3]
    v2 = [
        9.0, 9.0, 9.0,
        1.0, 1.1, 0.95,
        2.0, 2.1, 1.9,
        3.0, 3.2, 2.8,
    ]
    s2 = [5.0] * 12

    table_model = um.Mock()
    table_model.graduation_tables.return_value = [
        (0, [0], [len(v1)], v1, s1, [False]),
        (1, [0, 1], shape2, v2, s2, [False, True]),
    ]
    captured: dict[str, Any] = {}

    def fake_apply_graduation(x, y, w, updates, **kwargs):
        captured["updates"] = updates
        return table_model, True

    table_model.apply_graduation.side_effect = fake_apply_graduation

    x32 = np.zeros((1, 2), dtype=np.float32)
    y32 = np.zeros(1, dtype=np.float32)
    w_full = np.ones(1, dtype=np.float32)
    est.graduation_validation_ = {}
    table_model.predict_raw.return_value = np.zeros(1)
    est._graduate_table_model(table_model, x32, y32, w_full, None, None)

    updates = dict(captured["updates"])
    out1 = np.asarray(updates[0])
    assert out1[0] == pytest.approx(v1[0]), "1-D: the populated missing bin must stay verbatim"
    assert not np.allclose(out1[1:], v1[1:]), "1-D: the real numeric bins must actually smooth"

    out2 = np.asarray(updates[1]).reshape(shape2)
    v2_arr = np.asarray(v2).reshape(shape2)
    np.testing.assert_array_equal(
        out2[0, :], v2_arr[0, :]
    )  # 2-D: the missing ROW (numeric axis 0) stays verbatim
    assert not np.allclose(out2[1:, :], v2_arr[1:, :]), "2-D: the real rows must actually smooth"


def test_graduate_without_prune_raises_but_auto_default_stays_silent() -> None:
    x, y = regression_fixture()
    with pytest.raises(ValueError, match="graduate=True has no effect without prune=True"):
        small_regressor(graduate=True, prune=False).fit(x, y)

    # The auto-default (graduate=None, resolves to True for poisson/gamma) must NOT raise when
    # prune is off -- the whole resolution block lives inside the pruned path, so it truthfully
    # never engages (not a special-cased silence).
    est = small_regressor(
        objective="poisson", graduate=None, prune=False
    ).fit(x.astype(np.float32), np.abs(y))
    assert not hasattr(est, "graduation_report_")
    # Explicit False must never raise either, prune or not.
    small_regressor(graduate=False, prune=False).fit(x, y)
    small_regressor(graduate=False, prune=True).fit(x, y)

    # Binary classifier (K=2) routes through the same _fit_model, so the same rule applies.
    xc, yc = classifier_fixture()
    with pytest.raises(ValueError, match="graduate=True has no effect without prune=True"):
        TBoostClassifier(n_trees=10, seed=0, graduate=True, prune=False).fit(xc, yc)

    # Multiclass (K>=3): graduate=True always raises, prune or not -- there is no graduation
    # path for the multiclass per-class banks at all.
    xm, ym = multiclass_fixture()
    with pytest.raises(ValueError, match="graduate=True has no effect for multiclass"):
        TBoostClassifier(n_trees=10, seed=0, graduate=True, prune=False).fit(xm, ym)
    with pytest.raises(ValueError, match="graduate=True has no effect for multiclass"):
        TBoostClassifier(n_trees=10, seed=0, graduate=True, prune=True).fit(xm, ym)


def test_graduated_table_variance_reflects_post_smoothing_values() -> None:
    # H3: apply_graduation_updates must recompute EffectTable.variance (the field sobol()/the
    # rating export read) from the SMOOTHED values, not leave it at the pre-smoothing cache.
    rng = np.random.default_rng(3)
    xw = (np.arange(1600) % 20).astype(np.float64)[:, None]
    yw = (
        np.sin(xw[:, 0] / 3.0) + 0.1 * xw[:, 0] + rng.normal(scale=0.2, size=1600)
    ).astype(np.float32)

    baseline = small_regressor(prune=True, graduate=True, graduation_alpha=0.0).fit(xw, yw)
    graduated = small_regressor(prune=True, graduate=True, graduation_alpha=1.0).fit(xw, yw)
    rep = graduated.graduation_report_
    assert rep and rep[0]["alpha"] > 0.0 and not rep[0].get("rejected"), (
        "the forced alpha must actually be adopted for this test to mean anything "
        f"(report: {rep})"
    )

    def main_table(est: TBoostRegressor) -> dict[str, Any]:
        doc = json.loads(est.tables(xw[:20].astype(np.float32)))
        for t in doc["tables"]:
            if t["feature_set"] == [0]:
                return t
        raise AssertionError("no main-effect table found")

    base_t, grad_t = main_table(baseline), main_table(graduated)
    assert base_t["values"] != grad_t["values"], "graduation must have changed the values"
    assert base_t["variance"] != grad_t["variance"], (
        "variance must be recomputed from the smoothed values, not left stale at the "
        "pre-smoothing (baseline) number"
    )


def test_ordered_ts_supports_internal_early_stopping() -> None:
    # One-honest-holdout (design/ordered-ts-early-stopping.md): ordered target statistics now
    # work WITH validation_fraction — the holdout is carved before encoding, encoders are blind
    # to it, and bags share it. LOO stays gated. Same seed => byte-deterministic predictions.
    x, y = mixed_categorical_fixture()
    kw = dict(
        categorical_features=[1],
        validation_fraction=0.25,
        early_stopping_rounds=10,
        cat_leakage="ordered",
        cat_n_perms=2,
        n_bags=2,
        bag_subsample=0.8,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrecisionWarning)
        m1 = small_regressor(**kw).fit(x, y)
        m2 = small_regressor(**kw).fit(x, y)
        p1, p2 = m1.predict(x), m2.predict(x)
    assert p1.shape == (x.shape[0],)
    assert np.array_equal(p1, p2)

    with pytest.raises(Exception, match="loo"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PrecisionWarning)
            small_regressor(
                categorical_features=[1],
                validation_fraction=0.25,
                cat_leakage="loo",
            ).fit(x, y)


def test_interaction_gain_hurdle_roundtrips_and_caps_table_order() -> None:
    # Param round-trip + the structural contract: a prohibitive hurdle rejects new factors after
    # the first level, so the fitted bank holds ONLY main-effect (order-1) tables; the tree may still
    # reuse that first factor to refine the main surface. The default 0.0 grows multi-factor tables
    # on the same data.
    import json

    x, y = regression_fixture()
    est = small_regressor(interaction_gain_hurdle=2.5)
    assert est.get_params()["interaction_gain_hurdle"] == 2.5
    assert clone(est).get_params()["interaction_gain_hurdle"] == 2.5
    assert small_regressor(interaction_gain_hurdle_mode="fixed").get_params()[
        "interaction_gain_hurdle_mode"
    ] == "fixed"
    assert (
        clone(small_regressor(interaction_gain_hurdle_mode="adaptive")).get_params()[
            "interaction_gain_hurdle_mode"
        ]
        == "adaptive"
    )

    def max_order(model, xref) -> int:
        doc = json.loads(model.tables(xref))
        orders = [len(t["feature_set"]) for t in doc["tables"]]
        orders += [len(f["feature_set"]) for f in doc.get("factored", [])]
        return max(orders) if orders else 0

    hurdled = small_regressor(interaction_gain_hurdle=1e6).fit(x, y)
    assert max_order(hurdled, x) == 1
    unfloored = small_regressor(interaction_gain_hurdle=0.0).fit(x, y)
    assert max_order(unfloored, x) >= 2

    # recommended_recipe: silent -> adaptive shipped default; explicit overrides pass through.
    from t_boost import recommended_recipe

    assert recommended_recipe("squared_error", budget=50).interaction_gain_hurdle == 2.0
    assert recommended_recipe("squared_error", budget=50).interaction_gain_hurdle_mode == "adaptive"
    assert (
        recommended_recipe("squared_error", budget=50, interaction_gain_hurdle=0.7)
        .interaction_gain_hurdle
        == 0.7
    )
    assert (
        recommended_recipe("squared_error", budget=50, interaction_gain_hurdle_mode="fixed")
        .interaction_gain_hurdle_mode
        == "fixed"
    )


def test_recipe_bag_count_is_objective_aware() -> None:
    # 2026-07-23: n_bags is 8 for EVERY objective again — the 2026-07-22 gamma/tweedie 16 was
    # reverted once event-stratified subagging proved to be the actual variance fix (16 bags no
    # longer beat 8 anywhere on the post-stratification screen). Guards against the doubling
    # silently coming back. An explicit override always wins.
    from t_boost import recommended_recipe

    for obj in ("poisson", "gamma", "tweedie", "logistic", "squared_error"):
        assert recommended_recipe(obj, budget=50, tuned=True).get_params()["n_bags"] == 8
    assert recommended_recipe("tweedie", budget=50, tuned=True, n_bags=4).get_params()["n_bags"] == 4


def test_defaults_are_objective_aware_psm_and_reanchor() -> None:
    # 2026-07-23 batch: (1) path_smooth None resolves to 10.0 for gamma/tweedie (sparse-evidence
    # cell variance smoother; six-set screen) and 0.0 elsewhere; (2) reanchor None resolves ON
    # for binary logistic (the old off-default was measured against a silently-dropped
    # correction — post-fix 6-set battery: no harm where balance is right, repairs it where
    # off); reanchor_slope keeps the log-link-only default.
    from unittest import mock

    import t_boost.sklearn as S
    from t_boost import TBoostClassifier, TBoostRegressor

    assert TBoostRegressor().get_params()["path_smooth"] is None

    def resolved(est) -> dict:
        captured: dict = {}

        class _Recorder:
            def __init__(self, **kw):
                captured.update(kw)

        with mock.patch.object(S, "_Booster", _Recorder):
            est._new_booster()
        return captured

    assert resolved(TBoostRegressor(objective="gamma", n_trees=4))["path_smooth"] == 10.0
    assert resolved(TBoostRegressor(objective="tweedie", n_trees=4))["path_smooth"] == 10.0
    assert resolved(TBoostRegressor(objective="poisson", n_trees=4))["path_smooth"] == 0.0
    clf = resolved(TBoostClassifier(n_trees=4))
    assert clf["path_smooth"] == 0.0
    assert clf["reanchor"] is True                # K=2 logistic now reanchors by default
    assert clf["reanchor_slope"] is False         # slope stays log-link-only
    assert resolved(TBoostRegressor(objective="tweedie", path_smooth=0.0, n_trees=4))["path_smooth"] == 0.0


def test_regressor_native_categorical_object_array_stays_exact_and_cloneable() -> None:
    x, y = mixed_categorical_fixture()
    est = small_regressor(
        categorical_features=[1],
        cat_smooth=5.0,
        cat_target="mean",
        cat_leakage="kfold",
        cat_k=3,
        cat_min_data_per_group=0.0,
    )
    params = est.get_params()
    assert params["categorical_features"] == [1]
    assert params["cat_smooth"] == 5.0
    assert params["cat_target"] == "mean"
    assert params["cat_leakage"] == "kfold"
    assert params["cat_k"] == 3
    assert params["cat_min_data_per_group"] == 0.0
    assert clone(est).get_params()["categorical_features"] == [1]

    with pytest.warns(PrecisionWarning):
        est.fit(x, y)
    pred = est.predict(x)
    assert pred.shape == (x.shape[0],)
    assert est.n_features_in_ == 2
    assert est._cat_indices_ == [1]
    export = json.loads(est.tables(x, ref_measure="uniform"))
    assert export["mode"] == "Exact"

    loaded = TBoostRegressor.from_bytes(est.to_bytes())
    loaded.categorical_features = [1]
    with pytest.warns(PrecisionWarning):
        loaded_pred = loaded.predict(x)
    np.testing.assert_array_equal(loaded_pred, pred)


def test_monotone_with_native_categorical_remaps_and_is_enforced() -> None:
    # Categorical at index 0, numeric at index 1 — exercises the positional->axis-order remap
    # (native cats reorder numeric-first), so monotone_constraints=[0, +1] must land the +1 on
    # the numeric axis, not the categorical one. Verify it fits, stays Exact, and is enforced.
    n = 300
    rng = np.random.default_rng(1)
    cat = rng.choice(list("PQR"), n)
    num = rng.uniform(0.0, 5.0, n).astype(np.float32)
    x = np.empty((n, 2), dtype=object)
    x[:, 0] = cat
    x[:, 1] = num
    y = (
        0.6 * num
        + np.where(cat == "P", 1.0, np.where(cat == "Q", -0.5, 0.0))
        + rng.normal(0.0, 0.3, n)
    ).astype(np.float32)
    est = TBoostRegressor(
        objective="squared_error", n_trees=120, learning_rate=0.25, max_bin=32, seed=7,
        categorical_features=[0], monotone_constraints=[0, 1],
    )
    est.fit(x, y)  # must NOT raise
    exp = json.loads(est.tables(x[:64], ref_measure="uniform"))
    assert exp["mode"] == "Exact"
    # For each fixed category, prediction is non-decreasing in the (constrained) numeric.
    for c in ["P", "Q", "R"]:
        grid = np.empty((20, 2), dtype=object)
        grid[:, 0] = c
        grid[:, 1] = np.linspace(0.0, 5.0, 20).astype(np.float32)
        pred = np.asarray(est.predict(grid))
        assert np.all(np.diff(pred) >= -1e-6), f"monotone +1 violated for cat {c}: {pred}"


def test_native_categorical_early_stopping_kfold_allowed_ordered_gated() -> None:
    # KFold cross-fit (the default) gives OOF per-row encodings, so the carved validation
    # fold excludes each row's own target → internal early-stopping is leakage-free + allowed.
    x, y = mixed_categorical_fixture()
    est = small_regressor(
        categorical_features=[1], validation_fraction=0.2, cat_leakage="kfold"
    )
    est.fit(x, y)  # must NOT raise
    assert np.isfinite(np.asarray(est.predict(x))).all()
    # ordered now runs under the one-honest-holdout carve (design/ordered-ts-early-stopping.md);
    # LOO keeps the guard (own-target mean-shift pathology).
    est_o = small_regressor(
        categorical_features=[1], validation_fraction=0.2, cat_leakage="ordered"
    )
    est_o.fit(x, y)  # must NOT raise
    assert np.isfinite(np.asarray(est_o.predict(x))).all()
    with pytest.raises(Exception, match="loo"):
        small_regressor(
            categorical_features=[1], validation_fraction=0.2, cat_leakage="loo"
        ).fit(x, y)


def test_regressor_native_categorical_dataframe_names_and_monotone_guard() -> None:
    pd = pytest.importorskip("pandas")
    x, y = mixed_categorical_fixture()
    frame = pd.DataFrame(
        {
            "brand": x[:, 1],
            "age": x[:, 0].astype(np.float32),
        }
    )

    est = small_regressor(categorical_features=["brand"])
    est.fit(frame, y)
    assert est.feature_names_in_.tolist() == ["brand", "age"]
    assert est._model.feature_names == ["age", "brand"]
    pred = est.predict(frame)
    assert pred.shape == (frame.shape[0],)

    # Monotone on a numeric feature with a native categorical present is now supported
    # (the sign vector is remapped to the numeric-first axis order); fits + stays Exact.
    mono = small_regressor(categorical_features=["brand"], monotone_constraints={"age": 1})
    mono.fit(frame, y)
    assert json.loads(mono.tables(frame.iloc[:48], ref_measure="uniform"))["mode"] == "Exact"


def test_classifier_native_categorical_predict_proba() -> None:
    x, y_reg = mixed_categorical_fixture()
    y = np.where(y_reg > np.median(y_reg), "high", "low")
    clf = TBoostClassifier(
        n_trees=18,
        learning_rate=0.2,
        lambda_=1.0,
        max_bin=32,
        seed=11,
        categorical_features=[1],
    )
    with pytest.warns(PrecisionWarning):
        clf.fit(x, y)
    proba = clf.predict_proba(x)
    assert proba.shape == (x.shape[0], 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-6)
    export = json.loads(clf.tables(x, ref_measure="uniform"))
    assert export["mode"] == "Exact"


def test_outer_bag_is_thread_count_deterministic() -> None:
    # Bagging folds convex weights into tree alphas — still byte-identical across n_jobs.
    x, y = regression_fixture()
    a = model_bytes(small_regressor(n_bags=4, n_jobs=1).fit(x.astype(np.float32), y))
    b = model_bytes(small_regressor(n_bags=4, n_jobs=2).fit(x.astype(np.float32), y))
    assert a == b


def test_invalid_hist_precision_is_rejected() -> None:
    x, y = regression_fixture()
    with pytest.raises(Exception):
        small_regressor(hist_precision="nonsense").fit(x.astype(np.float32), y)


def test_reanchor_defaults_on_for_log_link_only() -> None:
    # reanchor=None ⇒ link-aware default: ON for log-link (gamma/poisson/tweedie),
    # OFF for identity/logit. Removes post-shrinkage aggregate bias for free.
    rng = np.random.RandomState(0)
    x = rng.rand(400, 3).astype(np.float32)
    y = (1.0 + 2.0 * x[:, 0] + 0.5 * rng.rand(400)).astype(np.float32)  # positive (gamma-safe)
    common = dict(n_trees=40, max_bin=32, seed=0, prune=False)
    # Gamma (log link): default reanchors → differs from explicitly-off.
    g_def = TBoostRegressor(objective="gamma", **common).fit(x, y).predict(x)
    g_off = TBoostRegressor(objective="gamma", reanchor=False, **common).fit(x, y).predict(x)
    assert not np.allclose(g_def, g_off)
    # SquaredError (identity link): default does NOT reanchor → identical to explicitly-off.
    s_def = TBoostRegressor(objective="squared_error", **common).fit(x, y).predict(x)
    s_off = TBoostRegressor(
        objective="squared_error", reanchor=False, **common
    ).fit(x, y).predict(x)
    np.testing.assert_array_equal(s_def, s_off)


def multiclass_fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(0)
    n = 600
    x = rng.rand(n, 4).astype(np.float32)
    y = np.where(x[:, 0] < 0.33, "a", np.where(x[:, 0] < 0.66, "b", "c"))
    return x, y


def test_multiclass_softmax_fit_predict_and_simplex() -> None:
    x, y = multiclass_fixture()
    clf = TBoostClassifier(n_trees=100, learning_rate=0.3, seed=0).fit(x, y)
    np.testing.assert_array_equal(clf.classes_, np.array(["a", "b", "c"], dtype=object))
    proba = clf.predict_proba(x)
    assert proba.shape == (x.shape[0], 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()
    # decision_function is (n_samples, n_classes) raw logits for multiclass.
    assert clf.decision_function(x).shape == (x.shape[0], 3)
    pred = clf.predict(x)
    assert (pred == y).mean() > 0.9


def test_multiclass_requires_at_least_two_classes() -> None:
    x, _ = multiclass_fixture()
    with pytest.raises(ValueError, match="at least two classes"):
        TBoostClassifier().fit(x, np.zeros(x.shape[0]))


def test_multiclass_rejects_exposure() -> None:
    x, y = multiclass_fixture()
    with pytest.raises(ValueError, match="exposure is not supported for multiclass"):
        TBoostClassifier().fit(x, y, exposure=np.ones(x.shape[0], dtype=np.float32))


def test_multiclass_serialization_round_trip() -> None:
    x, y = multiclass_fixture()
    clf = TBoostClassifier(n_trees=60, learning_rate=0.3, seed=0).fit(x, y)
    proba = clf.predict_proba(x)

    blob = clf.to_bytes()
    assert blob[:4] == b"TBP1"  # classes_ is always envelope-wrapped for classifiers now (H9)
    restored = TBoostClassifier.from_bytes(blob)
    np.testing.assert_array_equal(restored.classes_, clf.classes_)
    np.testing.assert_array_equal(restored.predict_proba(x), proba)

    restored_json = TBoostClassifier.from_json(clf.to_json())
    np.testing.assert_array_equal(restored_json.predict_proba(x), proba)


def test_multiclass_tables_are_per_class_and_exact() -> None:
    x, y = multiclass_fixture()
    clf = TBoostClassifier(n_trees=40, learning_rate=0.3, seed=0, prune=False).fit(x, y)
    tables = json.loads(clf.tables(x[:128]))
    # One rating export per class label; each is an exact per-class decomposition.
    assert set(tables) == {"a", "b", "c"}
    for export in tables.values():
        assert export["mode"] == "Exact"


# --- tables(sample_weight=..., exposure=...): spec §08.7 exposure-weighted support, wired ------
# through explain_weighted/explain_with_budget_weighted (crates/t-boost-core/src/explain.rs).
# These are explain-TIME weights aligned to the X passed to `tables()`, independent of whatever
# sample_weight/exposure (if any) `fit` was called with.


def test_tables_no_weight_or_exposure_is_the_plain_unweighted_export() -> None:
    # Regression guard: adding the weighted path must not perturb the default (no-args) call.
    x, y = regression_fixture()
    est = TBoostRegressor(n_trees=30, learning_rate=0.3, seed=0, prune=False).fit(x, y)
    assert est.tables(x) == est.tables(x, sample_weight=None, exposure=None)


def test_tables_all_ones_weight_and_exposure_matches_unweighted_exactly() -> None:
    x, y = regression_fixture()
    est = TBoostRegressor(n_trees=30, learning_rate=0.3, seed=0, prune=False).fit(x, y)
    ones = np.ones(x.shape[0], dtype=np.float32)
    assert_exports_close(est.tables(x), est.tables(x, sample_weight=ones, exposure=ones))
    assert_exports_close(est.tables(x), est.tables(x, sample_weight=ones))
    assert_exports_close(est.tables(x), est.tables(x, exposure=ones))


def test_tables_sample_weight_changes_support_and_is_not_display_only() -> None:
    # A real per-row weight must change the export -- support (and, under the default
    # product_marginals reference measure, values/importances/SE) reflect effective row mass,
    # not a flat row count (spec §08.7's whole point; NOT a cosmetic reweighting).
    x, y = regression_fixture()
    est = TBoostRegressor(n_trees=30, learning_rate=0.3, seed=0, prune=False).fit(x, y)
    heavy = np.where(x[:, 0] <= 2.0, 25.0, 1.0).astype(np.float32)
    assert est.tables(x) != est.tables(x, sample_weight=heavy)


def test_tables_sample_weight_and_exposure_combine_multiplicatively() -> None:
    # weight=w, exposure=1 must equal weight=1, exposure=w (both reduce to the same effective
    # row mass), and weight=w1, exposure=w2 must equal weight=w2, exposure=w1 (commutative).
    x, y = regression_fixture()
    est = TBoostRegressor(n_trees=30, learning_rate=0.3, seed=0, prune=False).fit(x, y)
    ones = np.ones(x.shape[0], dtype=np.float32)
    w1 = np.where(x[:, 0] <= 2.0, 5.0, 1.0).astype(np.float32)
    w2 = np.where(x[:, 1] <= 2.0, 3.0, 1.0).astype(np.float32)
    assert est.tables(x, sample_weight=w1, exposure=ones) == est.tables(x, exposure=w1)
    assert est.tables(x, sample_weight=w1, exposure=w2) == est.tables(x, sample_weight=w2, exposure=w1)


def test_tables_mismatched_weight_length_raises_value_error() -> None:
    x, y = regression_fixture()
    est = TBoostRegressor(n_trees=30, learning_rate=0.3, seed=0, prune=False).fit(x, y)
    with pytest.raises(ValueError, match="shape mismatch"):
        est.tables(x, sample_weight=np.ones(x.shape[0] - 1, dtype=np.float32))


def test_tables_negative_weight_raises_value_error() -> None:
    x, y = regression_fixture()
    est = TBoostRegressor(n_trees=30, learning_rate=0.3, seed=0, prune=False).fit(x, y)
    bad = np.ones(x.shape[0], dtype=np.float32)
    bad[0] = -1.0
    with pytest.raises(ValueError, match="invalid input"):
        est.tables(x, sample_weight=bad)


def test_binary_classifier_tables_honors_sample_weight() -> None:
    x, y = classifier_fixture()
    clf = TBoostClassifier(n_trees=30, learning_rate=0.3, seed=0, prune=False).fit(x, y)
    heavy = np.where(x[:, 0] <= 2.0, 25.0, 1.0).astype(np.float32)
    assert clf.tables(x) != clf.tables(x, sample_weight=heavy)


def test_multiclass_tables_weight_changes_every_class_bank() -> None:
    # PyMultiClassModel.tables loops over K per-class banks; the same combined row mass must
    # reach every class's explain call, not just the first.
    x, y = multiclass_fixture()
    clf = TBoostClassifier(n_trees=40, learning_rate=0.3, seed=0, prune=False).fit(x, y)
    heavy = np.where(x[:, 0] < 0.33, 25.0, 1.0).astype(np.float32)
    unweighted = json.loads(clf.tables(x))
    weighted = json.loads(clf.tables(x, sample_weight=heavy))
    assert set(unweighted) == set(weighted) == {"a", "b", "c"}
    for label in unweighted:
        assert unweighted[label] != weighted[label], f"class {label!r} bank did not reweight"


def test_multiclass_tables_exposure_is_accepted_unlike_fit() -> None:
    # fit() rejects exposure for multiclass (K>=3); tables()'s exposure is a different,
    # explain-time support-weighting concept and must NOT inherit that restriction.
    x, y = multiclass_fixture()
    clf = TBoostClassifier(n_trees=40, learning_rate=0.3, seed=0, prune=False).fit(x, y)
    with pytest.raises(ValueError, match="exposure is not supported for multiclass"):
        clf.fit(x, y, exposure=np.ones(x.shape[0], dtype=np.float32))
    heavy = np.where(x[:, 0] < 0.33, 25.0, 1.0).astype(np.float32)
    assert clf.tables(x) != clf.tables(x, exposure=heavy)


def _supports_scale_values_hold(plain: str, scaled: str, factor: float) -> None:
    """A uniform call-time mass scales every support by `factor` and leaves every value: the
    measure is normalized, so only the displayed mass moves."""
    a, b = json.loads(plain), json.loads(scaled)
    banks = list(zip(a.values(), b.values())) if "tables" not in a else [(a, b)]
    for bank_a, bank_b in banks:
        for ta, tb in zip(bank_a["tables"], bank_b["tables"]):
            np.testing.assert_allclose(tb["values"], ta["values"], rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(tb["support"], np.asarray(ta["support"]) * factor, rtol=1e-6)


def test_pruned_regressor_tables_recentre_on_call_time_weight_and_exposure() -> None:
    # A pruned model's tables are stored, and a call-time weight/exposure re-centres them on the
    # passed rows. A uniform mass of 7*7 changes only the supports.
    x, y = regression_fixture()
    est = TBoostRegressor(
        n_trees=30, learning_rate=0.3, seed=0, prune=True, prune_n_folds=2
    ).fit(x, y)
    heavy = np.full(x.shape[0], 7.0, dtype=np.float32)
    _supports_scale_values_hold(
        est.tables(x), est.tables(x, sample_weight=heavy, exposure=heavy), 49.0
    )
    skewed = np.where(x[:, 0] <= 2.0, 25.0, 1.0).astype(np.float32)
    assert est.tables(x) != est.tables(x, sample_weight=skewed)


def test_pruned_multiclass_tables_recentre_on_call_time_weight() -> None:
    x, y = multiclass_fixture()
    # (No `prune_validation_fraction` here: since 2026-09-07 the K>=3 prune selects on fold
    # refits and refuses that legacy single-split knob rather than silently ignoring it.)
    clf = TBoostClassifier(n_trees=40, learning_rate=0.3, seed=0, prune=True).fit(x, y)
    heavy = np.full(x.shape[0], 7.0, dtype=np.float32)
    _supports_scale_values_hold(clf.tables(x), clf.tables(x, sample_weight=heavy), 7.0)


# --- tables() defaults to the fit-time sample_weight/exposure when the caller passes neither ---


def test_tables_defaults_to_fit_time_sample_weight() -> None:
    x, y = regression_fixture()
    heavy = np.where(x[:, 0] <= 2.0, 25.0, 1.0).astype(np.float32)
    fit_weighted = TBoostRegressor(n_trees=30, learning_rate=0.3, seed=0, prune=False).fit(
        x, y, sample_weight=heavy
    )
    fit_plain = TBoostRegressor(n_trees=30, learning_rate=0.3, seed=0, prune=False).fit(x, y)
    # No args: the weighted fit must report weighted tables automatically (matching an explicit
    # call with the same weight), and must differ from an unweighted fit's plain tables.
    assert_exports_close(fit_weighted.tables(x), fit_weighted.tables(x, sample_weight=heavy))
    assert fit_weighted.tables(x) != fit_plain.tables(x)


def test_tables_defaults_to_fit_time_exposure() -> None:
    x, y = regression_fixture()
    heavy = np.where(x[:, 1] <= 2.0, 10.0, 1.0).astype(np.float32)
    fit_weighted = TBoostRegressor(n_trees=30, learning_rate=0.3, seed=0, prune=False).fit(
        x, y, exposure=heavy
    )
    fit_plain = TBoostRegressor(n_trees=30, learning_rate=0.3, seed=0, prune=False).fit(x, y)
    assert_exports_close(fit_weighted.tables(x), fit_weighted.tables(x, exposure=heavy))
    assert fit_weighted.tables(x) != fit_plain.tables(x)


def test_tables_explicit_call_time_weight_overrides_fit_time_weight() -> None:
    # fit-time sample_weight and explain-time (tables()) weight are independent mechanisms --
    # the former shapes gradient/hessian weighting during boosting (so it changes the fitted
    # tree structure/values themselves), the latter reweights an already-fitted model's table
    # export. A cross-model comparison would conflate the two, so this isolates the override on
    # ONE fitted model: an explicit call-time weight must produce the same export whether or not
    # a fit-time default is even present, proving it replaces rather than combines with it.
    x, y = regression_fixture()
    fit_time = np.where(x[:, 0] <= 2.0, 25.0, 1.0).astype(np.float32)
    call_time = np.where(x[:, 1] <= 2.0, 3.0, 1.0).astype(np.float32)
    est = TBoostRegressor(n_trees=30, learning_rate=0.3, seed=0, prune=False).fit(x, y, sample_weight=fit_time)
    with_override = est.tables(x, sample_weight=call_time)
    assert with_override != est.tables(x)  # differs from the fit-time-default export
    assert hasattr(est, "_fit_sample_weight_")
    stashed = est._fit_sample_weight_
    del est._fit_sample_weight_
    try:
        assert est.tables(x, sample_weight=call_time) == with_override
    finally:
        est._fit_sample_weight_ = stashed


def test_tables_all_ones_call_time_weight_forces_unweighted_export() -> None:
    # The documented escape hatch: pass an explicit all-ones array to opt back out of the
    # fit-time weight for one call. The stored tables carry the weighted fit's ledger; the
    # all-ones override re-centres them on flat row counts instead.
    x, y = regression_fixture()
    heavy = np.where(x[:, 0] <= 2.0, 25.0, 1.0).astype(np.float32)
    est = TBoostRegressor(n_trees=30, learning_rate=0.3, seed=0, prune=False).fit(x, y, sample_weight=heavy)
    ones = np.ones(x.shape[0], dtype=np.float32)
    with_all_ones = json.loads(est.tables(x, sample_weight=ones))
    stored = json.loads(est.tables(x))
    for table in with_all_ones["tables"]:
        assert sum(table["support"]) == pytest.approx(x.shape[0])
    assert any(
        sum(t["support"]) != pytest.approx(x.shape[0]) for t in stored["tables"]
    ), "the stored ledger is the weighted fit's"


def test_tables_fit_time_weight_does_not_leak_across_refit_or_set_params() -> None:
    x, y = regression_fixture()
    heavy = np.where(x[:, 0] <= 2.0, 25.0, 1.0).astype(np.float32)
    est = TBoostRegressor(n_trees=30, learning_rate=0.3, seed=0, prune=False)
    est.fit(x, y, sample_weight=heavy)
    weighted_tables = est.tables(x)

    # Direct refit without sample_weight must drop the stale fit-time weight.
    est.fit(x, y)
    assert not hasattr(est, "_fit_sample_weight_")
    refit_plain_tables = est.tables(x)
    assert refit_plain_tables != weighted_tables
    assert refit_plain_tables == TBoostRegressor(n_trees=30, learning_rate=0.3, seed=0, prune=False).fit(
        x, y
    ).tables(x)

    # set_params must clear it too, even before any refit happens.
    est2 = TBoostRegressor(n_trees=30, learning_rate=0.3, seed=0, prune=False).fit(x, y, sample_weight=heavy)
    assert hasattr(est2, "_fit_sample_weight_")
    est2.set_params(seed=1)
    assert not hasattr(est2, "_fit_sample_weight_")


def test_multiclass_tables_defaults_to_fit_time_sample_weight() -> None:
    x, y = multiclass_fixture()
    heavy = np.where(x[:, 0] < 0.33, 25.0, 1.0).astype(np.float32)
    fit_weighted = TBoostClassifier(n_trees=40, learning_rate=0.3, seed=0, prune=False).fit(
        x, y, sample_weight=heavy
    )
    fit_plain = TBoostClassifier(n_trees=40, learning_rate=0.3, seed=0, prune=False).fit(x, y)
    assert_exports_close(fit_weighted.tables(x), fit_weighted.tables(x, sample_weight=heavy))
    assert fit_weighted.tables(x) != fit_plain.tables(x)


def test_multiclass_fit_never_sets_fit_time_exposure() -> None:
    # fit() rejects exposure for K>=3 before _fit_multiclass runs, so there is never a
    # fit-time exposure to default to for a multiclass model.
    x, y = multiclass_fixture()
    clf = TBoostClassifier(n_trees=40, learning_rate=0.3, seed=0, prune=False).fit(x, y)
    assert not hasattr(clf, "_fit_exposure_")


def test_tables_sample_weight_and_exposure_default_independently() -> None:
    # sample_weight and exposure each fall back to their own fit-time value independently: an
    # explicit override of one must still combine with the other's fit-time default, not clear
    # it. Isolated on a single fitted model, same rationale as the override test above.
    x, y = regression_fixture()
    fit_weight = np.where(x[:, 0] <= 2.0, 25.0, 1.0).astype(np.float32)
    fit_exposure = np.where(x[:, 1] <= 2.0, 10.0, 1.0).astype(np.float32)
    call_weight = np.where(x[:, 1] <= 2.0, 3.0, 1.0).astype(np.float32)
    est = TBoostRegressor(n_trees=30, learning_rate=0.3, seed=0, prune=False).fit(
        x, y, sample_weight=fit_weight, exposure=fit_exposure
    )
    # Override only sample_weight; exposure must still come from the fit-time default.
    mixed = est.tables(x, sample_weight=call_weight)
    explicit_both = est.tables(x, sample_weight=call_weight, exposure=fit_exposure)
    assert mixed == explicit_both
    # And it must differ from overriding exposure to all-ones (i.e. the fit-time exposure really
    # is in effect, not silently dropped by the sample_weight override).
    ones = np.ones(x.shape[0], dtype=np.float32)
    assert mixed != est.tables(x, sample_weight=call_weight, exposure=ones)


def test_multiclass_integrates_with_clone_and_cross_val_score() -> None:
    from sklearn.model_selection import cross_val_score

    x, y = multiclass_fixture()
    est = TBoostClassifier(n_trees=40, learning_rate=0.3, seed=0, prune=False)
    cloned = clone(est)  # unfitted clone must succeed (params only)
    scores = cross_val_score(cloned, x, y, cv=3)
    assert scores.shape == (3,)
    assert scores.mean() > 0.8


def test_metrics_ordered_gini_matches_auc_identity() -> None:
    from sklearn.metrics import roc_auc_score

    from t_boost.metrics import lift_curve, ordered_gini, top_bucket_lift

    rng = np.random.RandomState(0)
    n = 4000
    y = (rng.rand(n) < 0.3).astype(float)
    score = 0.6 * y + rng.rand(n)
    # Perfect ranking normalizes to 1.
    assert abs(ordered_gini(y, y) - 1.0) < 1e-9
    # Normalized Gini == Somers' D == 2*AUC-1 for binary targets (uniform weights).
    assert abs(ordered_gini(y, score) - (2 * roc_auc_score(y, score) - 1)) < 1e-6
    # Lift curve: correct bucket count, rows partition n, informative top bucket.
    curve = lift_curve(y, score, buckets=10)
    assert len(curve) == 10
    assert sum(int(b["rows"]) for b in curve) == n
    assert curve[0]["lift"] > 1.0
    assert top_bucket_lift(y, score) == curve[0]["lift"]
    # Weighted path with uniform weights equals the unweighted result.
    assert ordered_gini(y, score) == ordered_gini(y, score, np.ones(n))


def test_order_desc_matches_rust_total_cmp_on_signed_nan_and_zero() -> None:
    # Regression test: a plain `-score` lexsort (the pre-fix implementation) always sorts NaN
    # last regardless of sign, but Rust's `f64::total_cmp` ranks +NaN as the maximum and -NaN as
    # the minimum, and distinguishes -0.0 < +0.0. Reference order cross-checked against an actual
    # `f64::total_cmp` run (not hand-derived) during the fix.
    from t_boost.metrics import _order_desc

    pos_nan = np.uint64(0x7FF8000000000000).view(np.float64)
    neg_nan = np.uint64(0xFFF8000000000000).view(np.float64)
    score = np.array([1.0, pos_nan, -1.0, neg_nan, 0.0, -0.0, 2.0], dtype=np.float64)
    # descending total_cmp order: +NaN, 2.0, 1.0, +0.0, -0.0, -1.0, -NaN
    expected = [1, 6, 0, 4, 5, 2, 3]
    assert _order_desc(score).tolist() == expected

    # Ties still break by ascending index (the secondary lexsort key is unaffected by the fix).
    tie_order = _order_desc(np.array([5.0, 3.0, 5.0, 3.0, 5.0]))
    assert tie_order.tolist() == [0, 2, 4, 1, 3]


def test_lift_curve_places_positive_nan_prediction_in_top_bucket() -> None:
    # The finding's exact failure scenario: a rival adapter emits a NaN prediction for one row.
    # Pre-fix, Python's lift_curve buried it in the bottom bucket while the Rust release-gate
    # (total_cmp) put it in the top bucket — the "identical" metrics silently disagreed.
    from t_boost.metrics import lift_curve

    rng = np.random.RandomState(0)
    n = 100
    y = (rng.rand(n) < 0.3).astype(float)
    score = rng.rand(n)
    pos_nan = np.uint64(0x7FF8000000000000).view(np.float64)
    score[7] = pos_nan
    curve = lift_curve(y, score, buckets=10)
    assert sum(int(b["rows"]) for b in curve) == n

    from t_boost.metrics import _order_desc

    assert _order_desc(score)[0] == 7


def test_pickle_round_trip_regressor_binary_multiclass() -> None:
    import pickle

    # Regressor
    xr, yr = regression_fixture()
    r = TBoostRegressor(n_trees=40, seed=0).fit(xr.astype(np.float32), yr)
    r2 = pickle.loads(pickle.dumps(r))
    np.testing.assert_array_equal(r.predict(xr.astype(np.float32)), r2.predict(xr.astype(np.float32)))

    # Binary classifier
    xc, yc = classifier_fixture()
    c = TBoostClassifier(n_trees=40, seed=0).fit(xc.astype(np.float32), yc)
    c2 = pickle.loads(pickle.dumps(c))
    np.testing.assert_array_equal(c2.classes_, c.classes_)
    np.testing.assert_array_equal(
        c.predict_proba(xc.astype(np.float32)), c2.predict_proba(xc.astype(np.float32))
    )

    # Multiclass classifier (the _MultiClassModel container round-trips too)
    xm, ym = multiclass_fixture()
    m = TBoostClassifier(n_trees=40, learning_rate=0.3, seed=0, prune=False).fit(xm, ym)
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(m2.classes_, m.classes_)
    np.testing.assert_array_equal(m.predict_proba(xm), m2.predict_proba(xm))


def test_recommended_recipe_carries_the_load_bearing_knobs() -> None:
    from t_boost import recommended_recipe

    r = recommended_recipe("squared_error", budget=50, n_jobs=1)
    assert isinstance(r, TBoostRegressor)
    assert r.early_stopping_rounds == 500  # the load-bearing knob (also the estimator default now)
    assert r.n_trees == 50 and r.validation_fraction == 0.1 and r.leaf_refine_steps == 4
    xr, yr = regression_fixture()
    r.fit(xr.astype(np.float32), yr)
    assert r.predict(xr.astype(np.float32)).shape == (xr.shape[0],)
    # Classifier objective → TBoostClassifier; log-link tuned recipe enables reanchor.
    assert isinstance(recommended_recipe("logistic", budget=50), TBoostClassifier)
    assert recommended_recipe("poisson", budget=50).reanchor is True
    # Explicit overrides win.
    assert recommended_recipe("squared_error", early_stopping_rounds=10).early_stopping_rounds == 10
    # Adaptive early-stopping patience is ON in the recipe (1.5, both regimes) and plumbs through
    # overrides — including back to None for the fixed-patience, bit-identical path.
    assert recommended_recipe("squared_error", budget=50).early_stopping_adaptive == 1.5
    assert recommended_recipe("poisson", budget=50).early_stopping_adaptive == 1.5
    assert (
        recommended_recipe(
            "poisson", budget=50, early_stopping_adaptive=None
        ).early_stopping_adaptive
        is None
    )
    assert (
        recommended_recipe("poisson", budget=50, early_stopping_adaptive=0.4).early_stopping_adaptive
        == 0.4
    )
    # early_stopping_min_delta: the recipe stays silent (inherits the shipped ctor default), and an
    # explicit override — including back to the legacy 0.0 — passes cleanly through **overrides.
    from t_boost.sklearn import _ES_MIN_DELTA_DEFAULT

    assert (
        recommended_recipe("squared_error", budget=50).early_stopping_min_delta
        == _ES_MIN_DELTA_DEFAULT
    )
    assert (
        recommended_recipe(
            "poisson", budget=50, early_stopping_min_delta=0.0
        ).early_stopping_min_delta
        == 0.0
    )


def test_early_stopping_adaptive_plumbs_through_and_exposes_tree_count() -> None:
    x, y = regression_fixture()
    x32 = x.astype(np.float32)
    # Round-trips through get_params / clone (a real, tunable constructor param).
    est = small_regressor(
        validation_fraction=0.2, early_stopping_rounds=50, early_stopping_adaptive=1.5, n_trees=200
    )
    assert est.get_params()["early_stopping_adaptive"] == 1.5
    assert clone(est).get_params()["early_stopping_adaptive"] == 1.5

    # Fits and exposes the retained (best-validation) tree count, bounded by the n_trees cap.
    ensemble_fit(est, x32, y)
    n_kept = est._model.n_trees
    assert isinstance(n_kept, int)
    assert 0 < n_kept <= 200
    assert est.predict(x32).shape == (x32.shape[0],)

    # The fixed-patience default (None) shares the same estimator surface / tree-count getter.
    base = small_regressor(validation_fraction=0.2, early_stopping_rounds=50, n_trees=200)
    assert base.get_params()["early_stopping_adaptive"] is None
    ensemble_fit(base, x32, y)
    assert base._model.n_trees > 0


def test_early_stopping_min_delta_round_trips_and_stops_before_cap() -> None:
    from t_boost.sklearn import _ES_MIN_DELTA_DEFAULT

    x, y = regression_fixture()
    x32 = x.astype(np.float32)
    # Shipped constructor default is the recommended tolerance, NOT the neutral engine 0.0.
    assert TBoostRegressor().early_stopping_min_delta == _ES_MIN_DELTA_DEFAULT
    assert TBoostClassifier().early_stopping_min_delta == _ES_MIN_DELTA_DEFAULT
    # A real, tunable constructor param: round-trips through get_params / set_params / clone.
    est = small_regressor(
        validation_fraction=0.25, early_stopping_rounds=5, early_stopping_min_delta=0.5, n_trees=300
    )
    assert est.get_params()["early_stopping_min_delta"] == 0.5
    assert clone(est).get_params()["early_stopping_min_delta"] == 0.5
    est.set_params(early_stopping_min_delta=0.1)
    assert est.get_params()["early_stopping_min_delta"] == 0.1

    # A huge tolerance makes early stopping actually terminate before the n_trees cap, and never
    # retains more trees than the legacy 0.0 any-improvement rule under otherwise identical params.
    big = ensemble_fit(small_regressor(
        validation_fraction=0.25, early_stopping_rounds=5, early_stopping_min_delta=0.5, n_trees=300
    ), x32, y)
    legacy = ensemble_fit(small_regressor(
        validation_fraction=0.25, early_stopping_rounds=5, early_stopping_min_delta=0.0, n_trees=300
    ), x32, y)
    assert 0 < big._model.n_trees < 300
    assert big._model.n_trees <= legacy._model.n_trees


def test_multiclass_early_stopping_min_delta_truncates_before_cap() -> None:
    x, y = multiclass_fixture()
    cap = 120
    # Huge tolerance + a validation carve ⇒ the softmax early-stopping loop treats tiny cross-entropy
    # gains as immaterial, so the best round stops advancing and the fit truncates before the cap.
    stopped = TBoostClassifier(
        n_trees=cap,
        learning_rate=0.3,
        seed=0,
        n_bags=0,
        leaf_refine_steps=0,
        validation_fraction=0.2,
        early_stopping_rounds=5,
        early_stopping_min_delta=0.5,
    ).fit(x, y)
    assert stopped.get_params()["early_stopping_min_delta"] == 0.5
    # A full-cap fit with early stopping OFF keeps every round; the early-stopped model is strictly
    # smaller (fewer retained trees), evidencing the multiclass loop terminated before the cap.
    full = TBoostClassifier(
        n_trees=cap,
        learning_rate=0.3,
        seed=0,
        n_bags=0,
        leaf_refine_steps=0,
        validation_fraction=None,
    ).fit(x, y)
    assert len(stopped.to_bytes()) < len(full.to_bytes())
    proba = stopped.predict_proba(x)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)


def test_refine_closed_form_tier2_round_trips_and_plumbs_through() -> None:
    x, y = regression_fixture()
    x32 = x.astype(np.float32)
    yp = np.abs(y)  # Poisson domain (non-negative)
    # A real, tunable constructor param: round-trips through get_params / clone; ships ON by default.
    est = small_regressor(objective="poisson", refine_closed_form_tier2=False)
    assert est.get_params()["refine_closed_form_tier2"] is False
    assert clone(est).get_params()["refine_closed_form_tier2"] is False
    assert small_regressor().get_params()["refine_closed_form_tier2"] is True

    # Both tiers fit + predict on a log-link (Poisson) objective WITH leaf refinement, where the tier
    # actually branches: Tier 1 (False) keeps the leaf updates byte-identical to the exact per-row
    # line search; Tier 2 (True, the default) is the closed-form fast path (~1e-7 leaf drift). Both
    # stay finite and exactly decomposable.
    preds: dict[bool, np.ndarray] = {}
    for tier2 in (True, False):
        m = small_regressor(
            objective="poisson", refine_closed_form_tier2=tier2, leaf_refine_steps=4, n_trees=80
        ).fit(x32, yp)
        p = m.predict(x32)
        assert p.shape == (x32.shape[0],) and np.all(np.isfinite(p))
        assert json.loads(m.tables(x32, ref_measure="uniform"))["mode"] == "Exact"
        preds[tier2] = p
    # The tiers track each other closely (the closed-form deltas follow the exact path) but need not
    # be bit-identical — Tier 2 trades a tiny leaf drift for dropping the per-row refine passes.
    assert np.allclose(preds[True], preds[False], rtol=1e-2, atol=1e-3)


def test_incremental_mu_round_trips_and_plumbs_through() -> None:
    x, y = regression_fixture()
    x32 = x.astype(np.float32)
    yp = np.abs(y)  # Poisson domain (non-negative)
    # A real constructor param: round-trips through get_params / clone; ships OFF by default (an
    # opt-in speed knob for log-link fits — not bit-identical, so not a default).
    est = small_regressor(objective="poisson", incremental_mu=True)
    assert est.get_params()["incremental_mu"] is True
    assert clone(est).get_params()["incremental_mu"] is True
    assert small_regressor().get_params()["incremental_mu"] is False

    # ON vs OFF fit + predict on a plain (no bagging / early stopping) Poisson fit: the multiplicative
    # mu=exp(F) cache tracks the exact per-round exp(F) closely, so predictions match within the
    # documented small drift (NOT bit-identical). On the plain path there is no early-stop/bagging to
    # amplify the drift, so a tight relative tolerance holds.
    preds: dict[bool, np.ndarray] = {}
    for inc in (False, True):
        m = small_regressor(
            objective="poisson", incremental_mu=inc, leaf_refine_steps=4, n_trees=80
        ).fit(x32, yp)
        p = m.predict(x32)
        assert p.shape == (x32.shape[0],) and np.all(np.isfinite(p))
        assert json.loads(m.tables(x32, ref_measure="uniform"))["mode"] == "Exact"
        preds[inc] = p
    assert np.allclose(preds[True], preds[False], rtol=1e-3, atol=1e-4)


def test_constructor_defaults_embed_the_recommended_recipe() -> None:
    # A bare estimator ships the winning recipe so untuned users perform well — the whole point of
    # making these the *defaults*. Guards against a refactor silently resetting them to neutral.
    for est in (TBoostRegressor(), TBoostClassifier()):
        p = est.get_params()
        assert p["n_trees"] == 4000
        assert p["validation_fraction"] == 0.1
        assert p["early_stopping_rounds"] == 500
        assert p["leaf_refine_steps"] == 4
        assert p["n_bags"] == 8      # 2026-07-10: ohlsson-gap study — row-bag diversity is the top ranking lever
        assert p["colsample_bytree"] == 0.8  # 2026-07-10: column diversity stacks with row bagging
        # Tier-2 log-link closed-form leaf refinement ships ON (the fast log-link refine default).
        assert p["refine_closed_form_tier2"] is True
        # Incremental mu=exp(F) cache ships OFF (opt-in, not bit-identical).
        assert p["incremental_mu"] is False
        # knobs the recipe leaves at their neutral values
        assert p["learning_rate"] == 0.05 and p["lambda_"] == 1.0 and p["max_bin"] == 254
        assert p["max_interaction_order"] == 3
        # 2026-07-15 benchmark parity (Ralph: the library must match what insur-arena ran):
        # adaptive early-stopping patience and post-fit table pruning ship ON — these were the
        # last two deltas between a bare constructor and the arena's deployed cells.
        assert p["early_stopping_adaptive"] == 1.5
        assert p["prune"] is True


def test_bare_constructor_is_the_benchmarked_recipe() -> None:
    # The insur-arena adapter fits `recommended_recipe(obj, ...)` with prune=True forced on every
    # deployed cell. A bare constructor must be THAT configuration: the only allowed deltas vs the
    # factory are its two harness conveniences (n_jobs) — budget maps 1:1 onto the n_trees default.
    from t_boost.sklearn import recommended_recipe

    for objective, cls in (("poisson", TBoostRegressor), ("logistic", TBoostClassifier)):
        recipe = recommended_recipe(objective, tuned=False).get_params()
        bare = cls() if objective == "logistic" else cls(objective=objective)
        deltas = {k for k, v in bare.get_params().items() if recipe[k] != v}
        assert deltas <= {"n_jobs"}, (
            f"bare {cls.__name__} diverged from the benchmarked recipe on: {sorted(deltas)}"
        )
        assert recipe["prune"] is True  # what every scored insur-arena cell deployed


def test_bare_regressor_fits_and_predicts_through_default_bagged_path() -> None:
    # Drive the default (early-stopping + outer-bagging + leaf-refine + colsample) path end to end
    # on a bare constructor, and confirm the bagged model is still exactly decomposable. n_trees is
    # capped only for test speed; every other lever is the shipped default.
    x, y = regression_fixture()
    x32 = x.astype(np.float32)
    est = TBoostRegressor(n_trees=64, n_jobs=1).fit(x32, y)
    assert est.get_params()["n_bags"] == 8  # bagging really is on by default
    pred = est.predict(x32)
    assert pred.shape == (x32.shape[0],) and np.all(np.isfinite(pred))
    assert json.loads(est.tables(x32, ref_measure="uniform"))["mode"] == "Exact"


def test_cat_level_collapses_all_missing_markers() -> None:
    from t_boost.sklearn import _CAT_MISSING, _cat_level

    assert _cat_level(None) == _CAT_MISSING
    assert _cat_level(float("nan")) == _CAT_MISSING
    assert _cat_level(np.nan) == _CAT_MISSING
    assert _cat_level("alpha") == "alpha"
    assert _cat_level(3) == "3"
    try:
        import pandas as pd

        assert _cat_level(pd.NA) == _CAT_MISSING
    except ImportError:
        pass


def test_outputs_are_float64() -> None:
    xr, yr = regression_fixture()
    r = TBoostRegressor(n_trees=20, seed=0).fit(xr.astype(np.float32), yr)
    assert r.predict(xr.astype(np.float32)).dtype == np.float64
    assert r.predict_raw(xr.astype(np.float32)).dtype == np.float64
    xc, yc = classifier_fixture()
    c = TBoostClassifier(n_trees=20, seed=0).fit(xc.astype(np.float32), yc)
    assert c.predict_proba(xc.astype(np.float32)).dtype == np.float64
    assert c.decision_function(xc.astype(np.float32)).dtype == np.float64


def test_bad_inputs_raise_valueerror() -> None:
    from scipy.sparse import csr_matrix

    xr, yr = regression_fixture()
    x32 = xr.astype(np.float32)
    with pytest.raises(ValueError, match="sparse"):
        TBoostRegressor().fit(csr_matrix(x32), yr)
    with pytest.raises(ValueError):
        TBoostRegressor().fit(
            np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
        )
    with pytest.raises(ValueError, match="sample_weight sums to zero"):
        TBoostRegressor().fit(x32, yr, sample_weight=np.zeros(yr.shape[0], dtype=np.float32))
    # A genuinely continuous target handed to the classifier → sklearn-style message.
    cont_y = np.linspace(0.0, 1.0, x32.shape[0]).astype(np.float64)
    with pytest.raises(ValueError, match="Unknown label type"):
        TBoostClassifier().fit(x32, cont_y)


def test_categorical_model_serialize_round_trip() -> None:
    x, y = mixed_categorical_fixture()
    r = TBoostRegressor(n_trees=30, seed=0, categorical_features=[1]).fit(x, y)
    pred = r.predict(x)
    # Before the fix this raised ("could not convert string to float") because _cat_indices_ was
    # not restored; now the TBP1 envelope carries the categorical layout.
    r2 = TBoostRegressor.from_bytes(r.to_bytes())
    assert r.to_bytes()[:4] == b"TBP1"
    np.testing.assert_array_equal(r2.predict(x), pred)
    r3 = TBoostRegressor.from_json(r.to_json())
    np.testing.assert_array_equal(r3.predict(x), pred)
    # A numeric model is enveloped too (the header carries params and column specs).
    xr, yr = regression_fixture()
    rn = TBoostRegressor(n_trees=20, seed=0).fit(xr.astype(np.float32), yr)
    assert rn.to_bytes()[:4] == b"TBP1"


def test_binary_path_unchanged_no_multiclass_container() -> None:
    x, y = classifier_fixture()
    clf = TBoostClassifier(n_trees=40, seed=0).fit(x, y)
    assert clf.predict_proba(x).shape == (x.shape[0], 2)
    blob = clf.to_bytes()
    assert blob[:4] != b"TBMC"  # binary never routes through the multiclass container
    assert blob[:4] == b"TBP1"  # classes_ envelope-wrapped for classifiers now (H9)


@pytest.mark.skipif((__import__("os").cpu_count() or 1) < 16,
                    reason="heuristic asserts collapse to min(cores, x) below 16 cores")
def test_default_fit_pool_heuristic() -> None:
    # P1.1 (plan/speed-campaign-2.md): n_jobs=None sizes the FIT pool from bags+features
    # instead of grabbing every core — small per-round tasks lose to fork-join churn on
    # wide pools (measured: full default MTPL fit 101.2s -> 78.9s under the heuristic).
    import os

    cores = os.cpu_count() or 1
    est = TBoostRegressor()  # n_bags=8 default
    assert est._default_fit_pool(9) == min(cores, 12)      # narrow shape: 1.5x bags
    assert est._default_fit_pool(130) == min(cores, 130)   # wide shape: saturates to cores
    assert TBoostRegressor(n_bags=1)._default_fit_pool(3) == min(cores, 4)  # floor
    # explicit n_jobs always wins, including negatives
    assert TBoostRegressor(n_jobs=5)._default_fit_pool(9) == 5
    assert TBoostRegressor(n_jobs=-1)._default_fit_pool(9) == cores


@pytest.mark.skipif((__import__("os").cpu_count() or 1) < 16,
                    reason="width delta unobservable below 16 cores")
def test_default_fit_does_not_pin_the_global_pool() -> None:
    # Review catch (2026-07-16): the heuristic width must apply to the fit's LOCAL scoped
    # pool only — if it reaches cap_global_pool_once, the process-global pool is pinned
    # narrow forever and serve-time predict loses full width. Assert in a SUBPROCESS (the
    # global pool builds once per process) that a default fit followed by a big predict
    # spins up more threads than the fit's heuristic width.
    import subprocess
    import sys

    code = """
import numpy as np, os
def n_threads():
    with open('/proc/self/status') as f:
        return int(next(l for l in f if l.startswith('Threads:')).split()[1])
from t_boost.sklearn import TBoostRegressor
rng = np.random.default_rng(0)
x = rng.uniform(size=(60_000, 3)).astype(np.float32)   # heuristic width = max(4, 12, 3) = 12
y = (2.0 * x[:, 0]).astype(np.float32)
est = TBoostRegressor(n_trees=16, n_bags=8, validation_fraction=None, prune=False, seed=0).fit(x, y)
base = n_threads()
xl = rng.uniform(size=(400_000, 3)).astype(np.float32)
est.predict(xl)
grown = n_threads() - base
width = est._default_fit_pool(3)
cores = os.cpu_count() or 1
assert width < cores, "test needs headroom between heuristic width and cores"
assert grown >= 0
# the global pool the predict ran on must NOT be capped at the fit width
total_after = n_threads()
print(f"THREADS {total_after} WIDTH {width} CORES {cores}")
assert total_after > width + 4, f"serve pool pinned near fit width: {total_after} vs {width}"
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
