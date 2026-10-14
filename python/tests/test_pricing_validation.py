"""Pricing diagnostics and evidence must respect row units and fitting boundaries."""
import json

import numpy as np
import pytest

from t_boost import TBoostRegressor
from t_boost.sklearn import _Booster


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


def test_ae_covers_unpruned_and_multiclass_fits():
    # Every fit deploys rating tables, so an unpruned fit has a grid too. Multiclass has one bank
    # (and grid) per class: one-vs-rest entries per class and feature.
    from t_boost import TBoostClassifier

    x, y, e = fixture()
    m = estimator(objective='poisson', prune=False).fit(x, y, exposure=e)
    assert m.actual_vs_expected(x, y, exposure=e)
    labels = np.digitize(x[:, 0], [-0.5, 0.5])
    multi = TBoostClassifier(n_trees=25, n_bags=3, n_jobs=1).fit(x, labels)
    rows = multi.actual_vs_expected(x, labels)
    assert {r['class'] for r in rows} == {0, 1, 2} and len(rows) == 3 * x.shape[1]
    for r in rows:
        assert sum(r['actual']) == pytest.approx(float(np.sum(labels == r['class'])))


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
    # No held-out gate: the smoothed bank ships whenever the native check (the training
    # deviance rises by at most 1%) accepts it. Nothing held out is scored, because nothing is.
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
    assert ('rejected' in m.graduation_validation_) is not train_adopted
    assert (result is not base) is train_adopted


def test_a_declined_graduation_alpha_warns_and_does_not_bind():
    # A fixed strength that would flatten a real alternating effect raises the training
    # deviance by far more than 1%: the tables stay as fitted, and that is never silent.
    rng = np.random.default_rng(11)
    x = rng.integers(0, 40, 4000).astype(float)[:, None]
    y = rng.poisson(np.exp(-0.2 + 0.9 * (x[:, 0] % 2)))
    with pytest.warns(RuntimeWarning, match="graduation_alpha=1000000000.0.*NO EFFECT"):
        m = estimator(objective='poisson', graduation_alpha=1e9).fit(x, y)
    assert m.graduation_validation_['adopted'] is False
    row = next(r for r in m.binding_report_ if r['param'] == 'graduation_alpha')
    assert row['status'] == 'INERT' and 'training deviance' in row['reason']
    with pytest.raises(ValueError, match='graduation_alpha'):
        m.check_bindings()
    flags = m.pricing_report(x, y)['review']
    assert any(f['check'] == 'binding' and f['status'] == 'warning'
               and f['message'].startswith('graduation_alpha=') for f in flags)
    # A strength the check accepts binds as usual.
    ok = estimator(objective='poisson', graduation_alpha=1e-3).fit(x, y)
    assert ok.graduation_validation_['adopted'] is True
    assert not any(r['param'] == 'graduation_alpha' and r['status'] == 'INERT'
                   for r in ok.binding_report_)


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
    """This module pins the fold-vote selector (its evidence gate and vote machinery) on
    unbanded tables: the ranked path and banding became the defaults on 2026-09-26."""
    from t_boost import TBoostClassifier as _C
    from t_boost import TBoostRegressor as _R

    for cls in (_R, _C):
        defaults = cls.__init__.__kwdefaults__
        monkeypatch.setitem(defaults, "prune_selector", "fold_vote")
        monkeypatch.setitem(defaults, "band_tolerance", None)


def _offset_fixture(n=800, seed=63):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)).astype(np.float32)
    exposure = rng.uniform(0.1, 0.8, n).astype(np.float32)
    offset = rng.normal(scale=0.5, size=n)
    return x, exposure, offset, rng


@pytest.mark.parametrize('wire', ['live', 'bytes'])
@pytest.mark.parametrize('objective', ['poisson', 'squared_error', 'logistic'])
def test_ae_requires_and_applies_the_fit_offset(objective, wire):
    from t_boost import TBoostClassifier

    x, e, o, rng = _offset_fixture()
    if objective == 'poisson':
        y = rng.poisson(e * np.exp(0.3 * x[:, 0] + o)).astype(np.float32)
        m, fit_kw, ae_kw = estimator(objective='poisson'), {'exposure': e}, {'exposure': e}
    elif objective == 'squared_error':
        y = (x[:, 0] + o + rng.normal(scale=0.3, size=len(o))).astype(np.float32)
        m, fit_kw, ae_kw = estimator(objective='squared_error'), {}, {}
    else:
        y = (rng.uniform(size=len(o)) < 1 / (1 + np.exp(-(0.5 * x[:, 0] + o)))).astype(int)
        m = TBoostClassifier(n_trees=25, n_bags=3, prune_n_folds=2,
                             validation_fraction=None, n_jobs=1)
        fit_kw = ae_kw = {}
    m.fit(x, y, offset=o, **fit_kw)
    if wire == 'bytes':
        m = type(m).from_bytes(m.to_bytes())
    with pytest.raises(ValueError, match='explicit offset'):
        m.actual_vs_expected(x, y, **ae_kw)
    report = m.actual_vs_expected(x, y, offset=o, **ae_kw)
    if objective == 'poisson':
        want = m.predict(x, offset=o) * e
    elif objective == 'squared_error':
        want = m.predict(x, offset=o)
    else:
        want = m.predict_proba(x, offset=o)[:, 1]
    for factor in report:
        assert sum(factor['expected']) == pytest.approx(float(np.sum(want)), rel=1e-9)
    # An explicit zero offset is the offset-free evaluation, not a missing argument.
    zero = m.actual_vs_expected(x, y, offset=np.zeros(len(o)), **ae_kw)[0]
    assert sum(zero['expected']) != pytest.approx(sum(report[0]['expected']), rel=1e-6)
    pricing = m.pricing_report(x, y, offset=o, **ae_kw)['actual_vs_expected']
    assert sum(pricing[0]['expected']) == pytest.approx(sum(report[0]['expected']), rel=1e-12)


def test_ae_offset_is_not_required_without_a_fit_offset():
    x, y, e = fixture()
    m = estimator(objective='poisson').fit(x, y, exposure=e)
    assert m._ae_requires_offset_ is False
    m.actual_vs_expected(x, y, exposure=e)


def test_ae_lists_cover_every_grid_cell_even_when_the_top_cells_are_empty():
    x, y, e = fixture(n=1500)
    m = estimator(objective='poisson').fit(x, y, exposure=e)
    low = x[:, 0] < np.quantile(x[:, 0], 0.3)
    report = m.pricing_report(x[low], y[low], exposure=e[low])
    for factor in report['actual_vs_expected']:
        cells = factor['axis']['cells']
        assert cells == len(factor['axis']['borders']) + 2
        for key in ('actual', 'expected', 'mass', 'rows', 'ae'):
            assert len(factor[key]) == cells, (factor['feature'], key)
    first = report['actual_vs_expected'][0]
    assert first['rows'][-1] == 0 and first['ae'][-1] is None

