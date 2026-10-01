"""The reference measure the fitted tables are purified against (2026-09-06).

`ref_measure=None` now resolves to `"exposure"` (exposure-weighted marginals with a
positivity floor); `"product_marginals"` is the pre-2026-09-06 half-uniform row-count
blend, kept for reproducibility. Changing the measure at fit time changes the prune and so
the deployed model; changing it at EXPORT time (`tables(ref_measure=...)`) re-expresses the
same function and never touches predictions.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from t_boost import TBoostRegressor


def _poisson_frame(n: int = 1500, seed: int = 0):
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 80, n)
    # licence length correlated with age (a thin joint support band)
    lic = np.clip(age - 17 - rng.exponential(3.0, n), 0, None)
    vgroup = rng.integers(0, 6, n).astype(np.float64)
    exposure = rng.uniform(0.2, 1.0, n).astype(np.float32)
    rate = np.exp(-2.0 + 0.6 * (age < 25) - 0.02 * lic + 0.1 * vgroup)
    y = rng.poisson(rate * exposure).astype(np.float32)
    X = np.column_stack([age, lic, vgroup]).astype(np.float32)
    return X, y, exposure


def _fit(**kw):
    X, y, e = _poisson_frame()
    est = TBoostRegressor(
        objective="poisson", n_trees=200, n_bags=2, max_depth=3, prune=True,
        early_stopping_rounds=30, **kw,
    )
    est.fit(X, y, exposure=e)
    return est, X, y, e


def test_default_measure_is_exposure_and_stamps_v6():
    est, X, _, _ = _fit()
    assert est._measure_kwargs()["ref_measure"] == "exposure"
    export = json.loads(est.tables(X))
    assert "ExposureMarginals" in json.dumps(export["reference_measure"])
    assert export["schema_version"] == 6


def test_legacy_measure_still_fits_and_stamps_v2():
    est, X, _, _ = _fit(ref_measure="product_marginals")
    export = json.loads(est.tables(X))
    assert "ProductMarginals" in json.dumps(export["reference_measure"])
    assert export["schema_version"] == 2


def test_export_time_measure_changes_the_ledger_not_the_predictions():
    est, X, _, e = _fit()
    pred = est.predict(X)
    deployed = json.loads(est.tables(X))
    legacy_view = json.loads(est.tables(X, ref_measure="product_marginals"))
    uniform_view = json.loads(est.tables(X, ref_measure="uniform"))
    # Same function, three ledgers: predictions are untouched by any export call ...
    np.testing.assert_array_equal(pred, est.predict(X))
    assert "ProductMarginals" in json.dumps(legacy_view["reference_measure"])
    assert "Uniform" in json.dumps(uniform_view["reference_measure"])
    # ... and the table sets match while at least one main effect's values differ.
    def mains(ex):
        return {t["feature_names"][0]: t for t in ex["tables"] if len(t["feature_names"]) == 1}
    a, b = mains(deployed), mains(legacy_view)
    assert set(a) == set(b) and a
    diffs = [
        float(np.max(np.abs(np.asarray(a[k]["values"], dtype=float)
                            - np.asarray(b[k]["values"], dtype=float))))
        for k in a
    ]
    assert max(diffs) > 1e-9


def test_measure_floor_must_be_positive():
    X, y, e = _poisson_frame(300)
    est = TBoostRegressor(objective="poisson", n_trees=20, n_bags=1, prune=True,
                          measure_floor=0.0)
    with pytest.raises(ValueError, match="measure_floor"):
        est.fit(X, y, exposure=e)


def test_unpruned_tables_accept_every_measure_name():
    X, y, e = _poisson_frame(500)
    est = TBoostRegressor(objective="poisson", n_trees=30, n_bags=1, prune=False,
                          early_stopping_rounds=10)
    est.fit(X, y, exposure=e)
    for name in ("exposure", "product_marginals", "uniform", "joint"):
        json.loads(est.tables(X, ref_measure=name))
    with pytest.raises(Exception):
        est.tables(X, ref_measure="hooker")


def test_joint_ledger_is_export_only_and_moves_mass_between_pairs_and_mains():
    est, X, _, e = _fit()
    pred = est.predict(X)
    deployed = json.loads(est.tables(X))
    joint = json.loads(est.tables(X, ref_measure="joint"))
    np.testing.assert_array_equal(pred, est.predict(X))
    assert json.dumps(joint["reference_measure"]) == '"Joint"'
    # Same supports, same intercept-plus-tables function; different owner of shared mass.
    names = lambda ex: sorted(tuple(t["feature_names"]) for t in ex["tables"])
    assert names(joint) == names(deployed)
    pairs_j = [t for t in joint["tables"] if len(t["feature_names"]) == 2]
    if pairs_j:
        moved = 0.0
        for tj in pairs_j:
            td = next(t for t in deployed["tables"] if t["feature_names"] == tj["feature_names"])
            moved += float(np.max(np.abs(np.asarray(tj["values"]) - np.asarray(td["values"]))))
        assert moved > 1e-9
    # The intercept absorbs each pair's joint mean, so it is allowed to differ; the sum of
    # every table at a data row must not. Check the score identity on the fit rows.
    cells = np.asarray(est._model.cell_indices(X.astype(np.float32)), dtype=np.int64)

    def score(ex):
        raw = np.full(cells.shape[0], float(ex["f0"]))
        by = {tuple(t["feature_names"]): t for t in ex["tables"]}
        fn = list(est._model.raw_feature_names())
        for key, t in by.items():
            idx = [fn.index(k) for k in key]
            shape = tuple(int(v) for v in t["shape"])
            vals = np.asarray(t["values"], dtype=float).reshape(shape)
            raw += vals[tuple(cells[:, j] for j in idx)]
        return raw

    if not deployed["factored"]:
        np.testing.assert_allclose(score(joint), score(deployed), atol=1e-8)


def test_actual_vs_expected_balances_in_total_and_covers_every_row():
    est, X, y, e = _fit()
    rows = est.actual_vs_expected(X, y, exposure=e)
    assert [r["feature"] for r in rows] == ["x0", "x1", "x2"] or len(rows) == 3
    n = X.shape[0]
    for r in rows:
        assert sum(r["rows"]) == n
        assert len(r["actual"]) == len(r["expected"]) == len(r["mass"]) == len(r["ae"])
        # Poisson fits are rebalanced in total, so the whole-book A/E is ~1 on the fit rows.
        assert abs(sum(r["actual"]) / sum(r["expected"]) - 1.0) < 0.05
        assert sum(r["mass"]) == pytest.approx(float(np.sum(e)), rel=1e-5)


def test_training_row_subset_is_rejected_as_graduation_evidence():
    est, X, y, e = _fit()
    with pytest.raises(ValueError, match="not an independent holdout"):
        est._graduate_table_model(
            est._model, X, y, np.ones(len(y), dtype=np.float32), e, None,
            eval_rows=np.arange(0, len(y), 2, dtype=np.uint32),
        )


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
