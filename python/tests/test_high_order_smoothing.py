"""Higher-order diffusion: deployed scoring, export and independent acceptance."""
import json

import numpy as np
import pytest
from sklearn.base import clone

from t_boost import TBoostClassifier, TBoostRegressor
from t_boost.sklearn import _TableModel


@pytest.mark.parametrize('alpha', [-0.1, 1.1, np.nan, np.inf, True, '0.5', None])
def test_invalid_strength_fails_before_fit(alpha):
    with pytest.raises(ValueError, match='graduation_high_order_alpha'):
        TBoostRegressor(graduation_high_order_alpha=alpha).fit(
            np.zeros((20, 3), dtype=np.float32), np.zeros(20, dtype=np.float32))


@pytest.mark.parametrize('kwargs', [dict(prune=False), dict(graduate=False),
                                   dict(monotone_constraints=[1, 0, 0])])
def test_unsupported_configuration_is_not_silently_ignored(kwargs):
    with pytest.raises(ValueError, match='smoothing|graduation_high_order_alpha'):
        TBoostRegressor(graduation_high_order_alpha=0.5, **kwargs).fit(
            np.zeros((20, 3), dtype=np.float32), np.zeros(20, dtype=np.float32))


def test_classifier_parameter_clone_and_multiclass_rejection():
    estimator = TBoostClassifier(graduation_high_order_alpha=0.4)
    assert clone(estimator).graduation_high_order_alpha == 0.4
    with pytest.raises(ValueError, match='higher-order smoothing.*multiclass'):
        estimator.fit(np.zeros((30, 3), dtype=np.float32), np.arange(30) % 3)


@pytest.mark.parametrize('candidate_score,train_adopted,expected', [
    (1.0, True, True), (100.0, True, True), (1.0, False, False)])
def test_high_order_only_candidate_adopts_on_the_reanchor_verdict(candidate_score, train_adopted, expected):
    class Model:
        def __init__(self, score):
            self.score = score
        def graduation_tables(self, limit):
            return []  # A higher-order-only candidate must still be built and adopted.
        def apply_high_order_graduation(self, x, y, weight, updates, **kwargs):
            assert np.all(x == 0) and np.all(y == 0)
            assert updates == [] and kwargs['alpha'] == 0.4
            return Model(candidate_score), train_adopted, json.dumps([
                dict(features=[0, 1, 2], factored_idx=0, alpha=0.4, applied=True)])
        def predict_raw(self, x, **kwargs):
            return np.full(len(x), self.score)
    estimator = TBoostRegressor(graduation_high_order_alpha=0.4)
    estimator.graduation_validation_ = {}
    x = np.zeros((10, 3), dtype=np.float32)
    source = Model(0.0)
    result = estimator._graduate_table_model(
        source, x, np.zeros(10), np.ones(10), None, None)
    assert (result is not source) is expected
    assert estimator.graduation_validation_['candidate_count'] == 1
    assert estimator.graduation_validation_['adopted'] is expected
    assert estimator.graduation_report_[0]['alpha'] == (0.4 if expected else 0.0)


def exported_score(bank, x):
    """Independent interpretation of the public threshold-box export."""
    result = np.full(len(x), bank['f0'])
    for table in bank['tables']:
        indices = []
        for axis in table['axes']:
            column = x[:, axis['raw']]
            borders = np.asarray(axis['borders'], dtype=np.float32)
            indices.append(np.where(np.isnan(column), 0,
                                    np.searchsorted(borders, column, side='left') + 1))
        result += np.asarray(table['values']).reshape(table['shape'])[tuple(indices)]
    for effect in bank['factored']:
        for box in effect['boxes']:
            corner = np.zeros(len(x), dtype=int)
            for d, raw in enumerate(effect['feature_set']):
                low = np.where(np.isnan(x[:, raw]), box['missing_left'][d],
                               x[:, raw] <= np.float32(box['thresholds'][d]))
                corner += low.astype(int) << d
            result += np.asarray(box['octants'])[corner]
    return result


@pytest.mark.parametrize('order', [3, 4])
def test_native_diffusion_export_roundtrip_and_box_budget(order):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(600, 4)).astype(np.float32)
    y = (1.2 * (x[:, 0] > 0) - .8 * (x[:, 1] > 0)
         + 3 * np.all(x[:, :order] > 0, axis=1)
         + rng.normal(size=len(x))).astype(np.float32)
    fitted = TBoostRegressor(n_trees=6, n_bags=1, n_jobs=1, validation_fraction=None,
                             max_depth=order + 1, max_interaction_order=order,
                             prune=False, interaction_gain_hurdle=0,
                             colsample_bytree=1, leaf_refine_steps=0).fit(x, y)
    weights = np.ones(len(y), dtype=np.float32)
    native = fitted._model
    source = native.apply_keepset(x, y, weights, native.table_supports(x), reanchor=False)
    before = source.to_bytes()
    candidate, adopted, report = source.apply_high_order_graduation(
        x, y, weights, [], 0.1, n_jobs=1)
    report = json.loads(report)
    assert any(r['applied'] and len(r['features']) == order for r in report)
    assert adopted, 'fixture must exercise the changed scorer, not a rejected candidate'
    assert source.to_bytes() == before
    assert candidate.to_bytes() != before
    bank = json.loads(candidate.tables())
    audit = [*x[:40], np.full(4, np.nan)]
    for effect in bank['factored']:
        for box in effect['boxes']:
            for raw, border in zip(effect['feature_set'], box['thresholds']):
                for value in [np.float32(border), np.nextafter(np.float32(border), np.float32(np.inf))]:
                    row = x[0].copy(); row[raw] = value; audit.append(row)
    audit = np.asarray(audit, dtype=np.float32)
    predictions = candidate.predict_raw(audit)
    np.testing.assert_allclose(exported_score(bank, audit), predictions, atol=1e-5, rtol=0)
    for restored in [_TableModel.from_json(candidate.to_json()), _TableModel.from_bytes(candidate.to_bytes())]:
        np.testing.assert_array_equal(restored.predict_raw(audit), predictions)
    zero, _, reports = source.apply_high_order_graduation(x, y, weights, [], 0.0)
    assert zero.to_bytes() == before and json.loads(reports) == []
    skipped, _, reports = source.apply_high_order_graduation(x, y, weights, [], 0.5, box_budget=1)
    assert skipped.to_bytes() == before
    assert all(not r['applied'] for r in json.loads(reports))


def test_estimator_fit_default_noop_and_enabled_reporting():
    rng = np.random.default_rng(24)
    x = rng.normal(size=(2000, 3)).astype(np.float32)
    y = (3 * np.all(x > 0, axis=1) + rng.normal(size=len(x))).astype(np.float32)
    params = dict(n_trees=6, n_bags=1, n_jobs=1, validation_fraction=None,
                  prune_n_folds=2, graduation_alpha=0, interaction_gain_hurdle=0,
                  leaf_refine_steps=0, colsample_bytree=1)
    default = TBoostRegressor(**params).fit(x, y)
    zero = TBoostRegressor(**params, graduation_high_order_alpha=0).fit(x, y)
    assert default.to_bytes() == zero.to_bytes()
    enabled = TBoostRegressor(**params, graduation_high_order_alpha=0.2).fit(x, y)
    high = [r for r in enabled.graduation_report_ if r.get('method') == 'reference_diffusion']
    assert high and any(r['applied'] for r in high)
    assert enabled.graduation_validation_['holdout_rows'] == 0
    assert enabled.graduation_validation_['candidate_count'] == 1
    for restored in [TBoostRegressor.from_json(enabled.to_json()), TBoostRegressor.from_bytes(enabled.to_bytes())]:
        np.testing.assert_array_equal(restored.predict(x), enabled.predict(x))
