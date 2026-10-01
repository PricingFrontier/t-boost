"""Pricing diagnostics and evidence must respect row units and fitting boundaries."""
import json

import numpy as np
import pytest

from t_boost import TBoostRegressor
from t_boost.sklearn import _Booster, _guard_mean_deviance, _guard_reanchor


def fixture(n=600):
    rng = np.random.default_rng(62)
    x = rng.normal(size=(n, 2)).astype(np.float32)
    exposure = rng.uniform(0.1, 0.8, n).astype(np.float32)
    y = rng.poisson(exposure * np.exp(0.3 * x[:, 0])).astype(np.float32)
    return x, y, exposure


def estimator(**kw):
    return TBoostRegressor(n_trees=25, n_bags=3, prune_n_folds=2,
                           validation_fraction=None, n_jobs=1, **kw)


@pytest.mark.parametrize('wire', ['live', 'json', 'bytes'])
def test_ae_requires_aligned_exposure_even_after_reload(wire):
    x, y, e = fixture()
    m = estimator(objective='poisson', graduate=False).fit(x, y, exposure=e)
    if wire == 'json':
        m = TBoostRegressor.from_json(m.to_json())
    elif wire == 'bytes':
        m = TBoostRegressor.from_bytes(m.to_bytes())
    for design in (x, x[::-1], x[:31]):
        with pytest.raises(ValueError, match='explicit exposure'):
            m.actual_vs_expected(design, y[:len(design)])
    r = m.actual_vs_expected(x[::-1], y[::-1], exposure=e[::-1])[0]
    assert sum(r['expected']) == pytest.approx(float(np.sum(m.predict(x) * e)))
    assert sum(r['mass']) == pytest.approx(float(np.sum(e)), rel=1e-6)


def test_ae_weight_requirement_and_column_arguments():
    import polars as pl
    x, y, e = fixture()
    w = np.linspace(1, 2, len(y)).astype(np.float32)
    df = pl.DataFrame({'a': x[:, 0], 'b': x[:, 1], 'claims': y, 'expo': e, 'w': w})
    m = estimator(objective='poisson', graduate=False).fit(
        df, 'claims', exposure='expo', sample_weight='w')
    with pytest.raises(ValueError, match='explicit sample_weight'):
        m.actual_vs_expected(df, 'claims', exposure='expo')
    row = m.actual_vs_expected(df, 'claims', exposure='expo', sample_weight='w')[0]
    assert sum(row['actual']) == pytest.approx(float(np.sum(w * y)))
    assert sum(row['expected']) == pytest.approx(float(np.sum(w * e * m.predict(df))))
    with pytest.raises(ValueError, match='finite values'):
        m.actual_vs_expected(df, y, exposure=e[:-1], sample_weight=w)


@pytest.mark.parametrize('grouped', [False, True])
def test_graduation_withholds_no_rows_from_the_fit(monkeypatch, grouped):
    # The inverse of the 2026-09-09..2026-09-20 invariant. Graduation used to carve
    # `prune_validation_fraction` of the rows before any fitting and never return them,
    # which cost +1.29% mean test deviance on the arena catalog. Every row must now reach
    # the prune core AND the smoother, and every row's outcome must be able to move the
    # deployed model.
    n = 1600
    x, y, e = fixture(n)
    x[:, 0] = np.arange(n)
    groups = np.arange(n) // 4 if grouped else None
    calls = []
    original = TBoostRegressor._fit_and_prune_core

    def core(self, booster, xx, yy, ww, ee, names, labels, mono, cats, gg):
        calls.append((xx.copy(), yy.copy(), gg))
        return original(self, booster, xx, yy, ww, ee, names, labels, mono, cats, gg)

    graduation_rows = []
    def inspect_graduation(self, model, xx, yy, ww, ee, cats, **kwargs):
        assert 'evaluation' not in kwargs      # no holdout is passed any more
        graduation_rows.append(xx[:, 0].astype(int))
        return model

    monkeypatch.setattr(TBoostRegressor, '_fit_and_prune_core', core)
    monkeypatch.setattr(TBoostRegressor, '_graduate_table_model', inspect_graduation)
    a = estimator(objective='poisson').fit(x, y, exposure=e, groups=groups)
    # the prune core and the smoother both saw the WHOLE design, in the caller's row order
    np.testing.assert_array_equal(calls[0][0][:, 0].astype(int), np.arange(n))
    np.testing.assert_array_equal(graduation_rows[0], np.arange(n))
    assert a.graduation_validation_['holdout_rows'] == 0
    assert a.graduation_validation_['training_rows'] == n
    assert a.graduation_validation_['evidence'] == 'none'
    # no row is inert: perturbing ANY row changes the deployed model
    changed = y.copy()
    changed[np.arange(0, n, 97)] += 50
    b = estimator(objective='poisson').fit(x, changed, exposure=e, groups=groups)
    assert a._model.to_bytes() != b._model.to_bytes()


@pytest.mark.parametrize('train_adopted', [True, False])
def test_graduation_adopts_on_the_native_reanchor_verdict(monkeypatch, train_adopted):
    # There is no accuracy gate: the smoothed bank ships whenever the native re-anchor
    # accepts it. Nothing is scored against held-out outcomes, because nothing is held out.
    class Model:
        def graduation_tables(self, limit):
            return [None]
        def apply_graduation(self, x, y, w, updates, **kw):
            assert 'eval_rows' not in kw
            return Model(), train_adopted
    m = estimator(objective='squared_error')
    m.graduation_validation_ = {}
    monkeypatch.setattr(m, '_graduate_one', lambda *a: (0, ['x0'], np.ones(2), 1.0))
    x = np.zeros((10, 1), dtype=np.float32)
    base = Model()
    result = m._graduate_table_model(base, x, np.zeros(10), np.ones(10), None, None)
    assert m.graduation_validation_['adopted'] is train_adopted
    assert m.graduation_validation_['n_smoothed'] == 1
    assert (result is not base) is train_adopted


def test_offset_guard_measures_expected_counts(monkeypatch):
    x, y, e = fixture()
    captured = []
    fit = _Booster.fit
    def capture(self, *a, **kw):
        model = fit(self, *a, **kw)
        captured.append(model)
        return model
    monkeypatch.setattr(_Booster, 'fit', capture)
    m = estimator(objective='poisson', graduate=False, prune_guard_min_rows=20).fit(x, y, exposure=e)
    g = m.pruning_report_['guard']
    assert g['evidence'] == 'oob' and g['exposure_offset']
    assert 'skipped' not in g
    sums, _, _, counts = captured[0].bag_oob_group_raw(
        x, [[]], weight=np.ones(len(y), dtype=np.float32), exposure=e,
        n_jobs=1, **m._measure_kwargs())
    counts = np.asarray(counts)
    valid = counts > 0
    raw = np.asarray(sums)[valid] / counts[valid] + np.log(e[valid].astype(float))
    w = np.ones(valid.sum())
    raw = _guard_reanchor('poisson', raw, y[valid], w, True)
    expected = _guard_mean_deviance('poisson', y[valid], raw, w, m.tweedie_rho)
    assert g['dev_full'] == pytest.approx(expected, rel=1e-10)
    assert 'fixed offset' in m.pruning_report_['slope']['skipped']


def test_monotone_final_tables_retain_full_function():
    x, y, _ = fixture()
    m = estimator(objective='squared_error', monotone_constraints=[1, 0]).fit(x, y)
    grid = np.column_stack([np.linspace(-4, 4, 100), np.full(100, 0.4)]).astype(np.float32)
    assert np.all(np.diff(m.predict(grid)) >= -1e-6)
    assert 'full table bank' in m.pruning_report_['constraints']
    assert m.graduation_validation_['holdout_rows'] == 0


def test_pricing_report_and_joint_residuals_are_explicit():
    x, y, e = fixture()
    model = estimator(objective='poisson', graduate=False, interaction_gain_hurdle=0).fit(x, y, exposure=e)
    report = model.pricing_report(x, y, exposure=e, ref_measure='joint')
    json.dumps(report, allow_nan=False)
    joint = report['tables']['joint_reallocation']
    assert joint['ridge'] == .001 and joint['joint_floor'] == 1e-6
    assert joint['exact_conditional_purity'] is False
    assert joint['shapley_interpretation'] is False
    for pair in joint['pairs']:
        assert np.isfinite(pair['max_abs_observed_conditional_mean'])
    assert report['actual_vs_expected'][0]['axis']['name']
    assert report['prediction_units'] == 'per unit exposure'


def test_legacy_unwrapped_log_model_restores_objective():
    x, y, e = fixture()
    model = estimator(objective='poisson', graduate=False).fit(x, y, exposure=e)
    loaded = TBoostRegressor.from_json(model._model.to_json())
    assert loaded.objective == 'poisson'
    a = loaded.actual_vs_expected(x, y, exposure=e)[0]
    assert sum(a['expected']) == pytest.approx(float(np.sum(loaded.predict(x) * e)))


@pytest.fixture(autouse=True)
def _legacy_fold_vote_unbanded(monkeypatch: pytest.MonkeyPatch) -> None:
    """This module pins the fold-vote selector (its guard, evidence gate and slope machinery) on
    unbanded tables: the ranked path and banding became the defaults on 2026-09-26."""
    from t_boost import TBoostClassifier as _C
    from t_boost import TBoostRegressor as _R

    for cls in (_R, _C):
        defaults = cls.__init__.__kwdefaults__
        monkeypatch.setitem(defaults, "prune_selector", "fold_vote")
        monkeypatch.setitem(defaults, "band_tolerance", None)
