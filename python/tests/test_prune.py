"""Table-pruning surface tests (the `prune=` post-fit tables-only model path)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from t_boost._t_boost import _Booster, _Model, _MultiClassTableModel, _TableModel
from t_boost.sklearn import TBoostClassifier, TBoostRegressor
from _artifact import ensemble_fit, model_bytes


def _noisy_poisson(n: int = 3000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    # Signal in features 0,1 (+ weak interaction); features 2,3 are pure noise.
    mu = np.exp(0.3 * x[:, 0] + 0.2 * x[:, 1] + 0.2 * x[:, 0] * x[:, 1])
    y = rng.poisson(mu).astype(np.float32)
    return x, y


def _n_tables(est: TBoostRegressor, x: np.ndarray) -> int:
    import json

    return len(json.loads(est.tables(x))["tables"])


def test_prune_reduces_tables_and_populates_report() -> None:
    x, y = _noisy_poisson()
    common = dict(objective="poisson", n_trees=300, n_bags=1, seed=0, graduate=False)
    full = TBoostRegressor(**common).fit(x, y)
    pruned = TBoostRegressor(**common, prune=True).fit(x, y)

    assert isinstance(pruned._model, _TableModel)
    assert _n_tables(pruned, x) <= _n_tables(full, x)
    rep = pruned.pruning_report_
    # Contribution-stability pruning reports the selected support directly; there is no transferred
    # drop fraction because held-out contribution decides which tables survive.
    assert set(rep) >= {"kept", "effective_order", "cv_folds", "selector", "table_scores"}
    assert rep["selector"] == "heldout_contribution_stability"
    assert rep["main_effect_policy"] == "sticky"
    assert 0 <= rep["effective_order"] <= 3
    assert all(len(fs) >= 1 for fs in rep["kept"])
    assert all(row["selected"] for row in rep["table_scores"] if row["sticky"])
    # The report reconciles selection with the shipped artifact: deployed lists what actually
    # survived purification into the bank, and kept ∖ deployed is surfaced, never silent.
    deployed = {tuple(u) for u in rep["deployed"]}
    kept = {tuple(sorted(u)) for u in rep["kept"]}
    assert deployed <= kept
    assert {tuple(u) for u in rep["kept_not_deployed"]} == kept - deployed
    # predictions are finite and the right shape
    p = pruned.predict(x[:20])
    assert p.shape == (20,) and np.all(np.isfinite(p))


def test_pruned_model_round_trips_through_bytes_and_json() -> None:
    x, y = _noisy_poisson(n=1500)
    est = TBoostRegressor(objective="poisson", n_trees=200, n_bags=1, seed=0, prune=True, graduate=False).fit(x, y)
    p = est.predict(x[:50])

    blob = est.to_bytes()
    assert bytes(blob[:4]) == b"TBP1"
    assert model_bytes(est)[:4] == b"TBTM", "pruned model must serialize with the tables-only magic"
    back = TBoostRegressor.from_bytes(blob)
    assert isinstance(back._model, _TableModel)
    assert np.allclose(p, back.predict(x[:50]), atol=1e-5)

    back_json = TBoostRegressor.from_json(est.to_json())
    assert np.allclose(p, back_json.predict(x[:50]), atol=1e-5)


def test_se_rule_raises_on_scalar_path_when_non_default() -> None:
    # prune_se_rule reaches this path's native call, but not its OUTCOME: the per-fold selections
    # the SE band steers are aggregated by a rule that ORs `kept_rate` against `positive_rate`
    # behind an se_rule-independent `mean_gain > 0`, so the band never decides the deployed
    # keep-set. Measured 2026-09-03 on French MTPL frequency: se_rule in {0, 0.5, 1, 2, 10, 100}
    # gives bit-identical banks and test deviance at 8k/25k/80k/406k rows. This test previously
    # asserted `strict >= loose`, which passed only because the two are EQUAL — the assertion was
    # satisfied by the no-op it was meant to catch. It must raise instead.
    x, y = _noisy_poisson(n=1000)
    common = dict(objective="poisson", n_trees=60, n_bags=1, seed=0, prune=True, graduate=False)
    with pytest.raises(ValueError, match="prune_se_rule"):
        TBoostRegressor(**common, prune_se_rule=1.0).fit(x, y)
    # The default (0.0) still fits normally.
    est = TBoostRegressor(**common, prune_se_rule=0.0).fit(x, y)
    assert isinstance(est._model, _TableModel)


def test_se_rule_is_honored_on_the_multiclass_path() -> None:
    # The K>=3 path selects on ONE train/select split (`fit_multiclass_pruned`), so the walk's
    # chosen waypoint IS the keep-set and the SE band decides it directly — the band is live here
    # exactly where it is dead on the scalar path above. se_rule=0 takes the held-out minimum;
    # a wide band prunes strictly harder.
    rng = np.random.default_rng(3)
    n = 3000
    x = rng.normal(size=(n, 5)).astype(np.float32)
    lin = np.stack(
        [0.9 * x[:, 0] + 0.5 * x[:, 1] * x[:, 2], 0.7 * x[:, 3] - 0.4 * x[:, 1], 0.3 * x[:, 4]],
        axis=1,
    )
    y = np.argmax(lin + rng.normal(scale=0.5, size=(n, 3)), axis=1)
    # (The legacy single-split regime, `multiclass_prune_cv=False`: under the default CV
    # regime the band acts inside each fold walk and the fold VOTE decides the keep-set.)
    common = dict(
        objective="logistic", n_trees=120, n_bags=1, seed=0, prune=True, multiclass_prune_cv=False
    )
    strict = TBoostClassifier(**common, prune_se_rule=0.0).fit(x, y)
    loose = TBoostClassifier(**common, prune_se_rule=5.0).fit(x, y)
    n_strict = len(strict.pruning_report_["kept"])
    n_loose = len(loose.pruning_report_["kept"])
    assert n_loose < n_strict, (n_strict, n_loose)


def test_prune_default_on_deploys_table_model() -> None:
    # 2026-07-15 benchmark parity: prune defaults ON (every insur-arena cell deployed the
    # CV-pruned tables-only artifact); prune=False keeps the full table bank.
    x, y = _noisy_poisson(n=1000)
    est = TBoostRegressor(objective="poisson", n_trees=100, n_bags=1, seed=0).fit(x, y)
    assert isinstance(est._model, _TableModel)
    assert hasattr(est, "pruning_report_")

    off = TBoostRegressor(
        objective="poisson", n_trees=100, n_bags=1, seed=0, prune=False
    ).fit(x, y)
    assert isinstance(off._model, _TableModel)
    assert off._model.deployed_table_count() >= est._model.deployed_table_count()
    # No selection ran: every table is kept, and the report says so.
    assert off.pruning_report_["selector"] == "none"
    assert off.pruning_report_["kept"] == off.pruning_report_["deployed"]
    assert "table_scores" not in off.pruning_report_


def test_an_unpruned_multiclass_fit_is_banded() -> None:
    # Only the selection is skipped on an unpruned fit (2026-10-14): every class bank is banded,
    # against the spread of the bags' full banks, as a pruned fit's are.
    rng = np.random.default_rng(5)
    n = 3000
    x = rng.normal(size=(n, 3)).astype(np.float32)
    logits = np.stack([np.zeros(n), x[:, 0] + 0.5 * x[:, 0] * x[:, 1], -x[:, 1]], axis=1)
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = (rng.uniform(size=(n, 1)) > np.cumsum(p, axis=1)).sum(axis=1)
    kw = dict(n_trees=60, n_bags=4, seed=0, n_jobs=2, prune=False)
    # (explicit tolerance: this module pins `band_tolerance=None`, see the fixture at its end)
    banded = TBoostClassifier(**kw, band_tolerance=0.75).fit(x, y)
    lossless = TBoostClassifier(**kw, band_tolerance=None).fit(x, y)
    rep = banded.pruning_report_
    assert rep["selector"] == "none"
    assert len(rep["banding"]) == 3
    assert "banding" not in lossless.pruning_report_

    def cells(est: TBoostClassifier) -> int:
        return sum(est._multi_model.deployed_census()["cells"].values())

    assert cells(banded) < cells(lossless)
    np.testing.assert_allclose(banded.predict_proba(x).sum(axis=1), 1.0, rtol=1e-5)
    restored = TBoostClassifier.from_bytes(banded.to_bytes())
    np.testing.assert_array_equal(restored.predict_proba(x), banded.predict_proba(x))


def test_scalar_prune_accepts_sklearn_all_core_n_jobs() -> None:
    x, y = _noisy_poisson(n=500, seed=12)
    est = TBoostRegressor(
        objective="poisson",
        n_trees=35,
        n_bags=1,
        seed=0,
        prune=True,
        graduate=False,
        n_jobs=-1,
    ).fit(x, y)
    assert isinstance(est._model, _TableModel)


def test_multiclass_prune_refit_full_raises_too() -> None:
    # prune_refit_full is dead on the multiclass path too (never read by fit_multiclass_pruned);
    # it must raise there as well, not just on the scalar path.
    rng = np.random.default_rng(2)
    n = 300
    x = rng.normal(size=(n, 3)).astype(np.float32)
    y = np.argmax(np.stack([x[:, 0], x[:, 1], -x[:, 0]], axis=1), axis=1)
    with pytest.raises(ValueError, match="prune_refit_full"):
        TBoostClassifier(
            objective="logistic", n_trees=30, n_bags=1, seed=0, prune=True, prune_refit_full=True
        ).fit(x, y)


def test_multiclass_prune_produces_tables_model_and_round_trips() -> None:
    # K>=3: prune the K per-class banks with a shared keep-set (joint softmax deviance) → a
    # tables-only multiclass model that still predicts a proper simplex and round-trips via TBMT.
    rng = np.random.default_rng(1)
    n = 2400
    x = rng.normal(size=(n, 4)).astype(np.float32)
    lin = np.stack([0.7 * x[:, 0], 0.6 * x[:, 1], 0.4 * x[:, 0] * x[:, 1]], axis=1)
    y = np.argmax(lin + rng.normal(scale=0.5, size=(n, 3)), axis=1)
    est = TBoostClassifier(
        objective="logistic",
        n_trees=120,
        n_bags=1,
        seed=0,
        prune=True,
        prune_se_rule=1.0,
        n_jobs=2,
    ).fit(x, y)
    assert isinstance(est._multi_model, _MultiClassTableModel)
    proba = est.predict_proba(x[:10])
    assert proba.shape == (10, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert "effective_order" in est.pruning_report_
    blob = est.to_bytes()
    # classes_ ([0, 1, 2] ints from np.argmax) forces the TBP1 envelope even without
    # categoricals (H9); the tables-only TBMT container lives just past the header.
    assert bytes(blob[:4]) == b"TBP1"
    back = TBoostClassifier.from_bytes(blob)
    assert np.allclose(proba, back.predict_proba(x[:10]), atol=1e-5)


def test_native_multiclass_fit_prune_matches_two_call_path() -> None:
    rng = np.random.default_rng(17)
    n = 700
    x = rng.normal(size=(n, 4)).astype(np.float32)
    logits = np.stack((x[:, 0], x[:, 1], -x[:, 0] - x[:, 1]), axis=1)
    y = np.argmax(logits + rng.normal(scale=0.4, size=logits.shape), axis=1)
    y_f32 = y.astype(np.float32)
    labels = ["0", "1", "2"]
    weight = rng.uniform(0.5, 2.0, size=n).astype(np.float32)
    sel = np.sort(rng.permutation(n)[:140])
    selected = np.zeros(n, dtype=bool)
    selected[sel] = True
    train = np.flatnonzero(~selected)
    booster = _Booster(
        objective="logistic", n_trees=45, learning_rate=0.1, seed=5, n_jobs=2
    )

    fitted = booster.fit_multiclass(
        np.ascontiguousarray(x[train]),
        np.ascontiguousarray(y_f32[train]),
        3,
        labels,
        weight=np.ascontiguousarray(weight[train]),
    )
    expected, expected_report = fitted.prune_to_tables(
        x,
        y.astype(np.uint32),
        weight,
        sel.tolist(),
        se_rule=0.0,
        n_folds=3,
        n_jobs=2,
    )
    actual, actual_report = booster.fit_multiclass_pruned(
        x,
        y_f32,
        3,
        labels,
        sel.tolist(),
        weight=weight,
        se_rule=0.0,
        n_folds=3,
    )
    # The SELECTION contract is unchanged: the keep-set is chosen on a complement fit's
    # held-out slice, so the report must match the manual two-call path exactly. Compared as
    # PARSED JSON: re-serializing through `serde_json::Value` to attach the selector fields
    # below also normalizes key order, which is not a selection change.
    actual_doc = json.loads(actual_report)
    # The binding also names the selection regime it ran (`sel_rows` given, no `fold_of`),
    # and whether main effects were pruning candidates (the default keeps them all).
    assert actual_doc.pop("selector") == "single_split_walk"
    assert actual_doc.pop("cv_folds") == 0
    assert actual_doc.pop("main_effect_policy") == "sticky"
    assert actual_doc == json.loads(expected_report)
    # The DEPLOYED model contract changed (2026-07-14): the keep-set is applied to a fit on
    # ALL rows (the old design withheld the sel slice from the deployed model), so the table
    # payload must legitimately DIFFER from the complement-fit tables while staying valid.
    assert actual.to_bytes() != expected.to_bytes()
    p = actual.predict_proba(x)
    assert p.shape == (n, 3)
    assert np.all(np.isfinite(p))
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_reanchor_balances_pruned_poisson_aggregate() -> None:
    # Dropping tables shifts the exposure-weighted aggregate; for a log-link model the prune's
    # reanchor (default on for poisson) restores Σŷ/Σy ≈ 1 over the training population.
    rng = np.random.default_rng(3)
    nn = 12000
    x = rng.normal(size=(nn, 4)).astype(np.float32)
    mu = np.exp(-0.5 + 0.4 * x[:, 0] + 0.3 * x[:, 1] + 0.25 * x[:, 0] * x[:, 1])
    y = rng.poisson(mu).astype(np.float32)
    est = TBoostRegressor(
        objective="poisson", n_trees=250, n_bags=1, seed=0, prune=True
    ).fit(x, y)
    bal = float(np.asarray(est.predict(x), np.float64).sum() / float(y.sum()))
    assert abs(bal - 1.0) < 0.02, f"pruned poisson aggregate should be ~balanced, got {bal}"


def test_prune_with_exposure_balances() -> None:
    # Exposure-aware pruning: the pruned rate model times exposure should sum to the counts
    # (Σ(ŷ·exposure)/Σy ≈ 1) — the held-out deviance + reanchor now apply the exposure log-offset.
    rng = np.random.default_rng(4)
    n = 12000
    x = rng.normal(size=(n, 4)).astype(np.float32)
    expo = rng.uniform(0.3, 1.0, size=n).astype(np.float32)
    mu = np.exp(-0.5 + 0.4 * x[:, 0] + 0.3 * x[:, 1] + 0.25 * x[:, 0] * x[:, 1])
    y = rng.poisson(mu * expo).astype(np.float32)
    est = TBoostRegressor(
        objective="poisson", n_trees=250, n_bags=1, seed=0, prune=True
    ).fit(x, y, exposure=expo)
    rate = np.asarray(est.predict(x), np.float64)
    bal = float((rate * expo).sum() / float(y.sum()))
    assert abs(bal - 1.0) < 0.02, f"exposure-weighted aggregate should balance, got {bal}"


def test_prune_refit_full_flag_raises_instead_of_silently_no_opping() -> None:
    # Scalar pruning always deploys full-data tables after CV support selection; there is no
    # separate subset-pruned fit for the legacy prune_refit_full flag to conditionally refit, and
    # neither native prune entry point takes a parameter it could map to. It used to be silently
    # ignored; it must now raise rather than let the user believe the request was honored.
    x, y = _noisy_poisson(n=1000)
    common = dict(objective="poisson", n_trees=60, n_bags=1, seed=0, prune=True, graduate=False)
    with pytest.raises(ValueError, match="prune_refit_full"):
        TBoostRegressor(**common, prune_refit_full=True).fit(x, y)
    # The default (False) still fits normally.
    est = TBoostRegressor(**common, prune_refit_full=False).fit(x, y)
    assert isinstance(est._model, _TableModel)


def _marginal_poisson(n: int = 5000, p: int = 8, seed: int = 0, noise: float = 0.1):
    """A fit with genuinely MARGINAL tables — ones a fold vote can disagree about.

    `_noisy_poisson` is too clean for the stability bar: every kept table there scores in every
    fold, so the threshold never binds and a test built on it would pass vacuously.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p)).astype(np.float32)
    mu = np.exp(
        0.35 * x[:, 0] + 0.25 * x[:, 1] + 0.2 * x[:, 0] * x[:, 1] + noise * x[:, 2] * x[:, 3]
    )
    return x, rng.poisson(mu).astype(np.float32)


def test_exposed_prune_internals_are_bit_identical_at_their_hardcoded_defaults() -> None:
    # The 2026-09-04 exposure of seven previously-hardcoded prune internals must not move a
    # single shipped artifact: passing each one explicitly at the value it used to be frozen at
    # has to reproduce the bare estimator exactly, and the report must echo them.
    x, y = _noisy_poisson(n=1500)
    common = dict(objective="poisson", n_trees=80, n_bags=1, seed=0, prune=True, graduate=False)
    bare = TBoostRegressor(**common).fit(x, y)
    explicit = TBoostRegressor(
        **common,
        prune_min_stability=0.5,
        prune_min_mean_gain=0.0,
        prune_fold_min_rows=125,
        prune_fold_es_patience=None,
        prune_guard_min_rows=500,
    ).fit(x, y)
    np.testing.assert_array_equal(bare.predict(x), explicit.predict(x))
    assert bare.pruning_report_["kept"] == explicit.pruning_report_["kept"]
    # The report used to write a literal 0.0 for a threshold the caller could not set.
    assert explicit.pruning_report_["min_stability"] == 0.5
    assert explicit.pruning_report_["min_mean_gain"] == 0.0


def test_prune_min_stability_tightens_the_keep_set() -> None:
    # The fold-stability bar: keep a table only if it showed signal in at least this fraction of
    # the prune folds. Demanding ALL folds must keep strictly fewer tables than the shipped 0.5
    # — on a fixture that actually has marginal tables to disagree about.
    x, y = _marginal_poisson()
    common = dict(objective="poisson", n_trees=200, n_bags=1, seed=0, prune=True, graduate=False)
    lenient = TBoostRegressor(**common).fit(x, y)  # 0.5 = 3 of 5 folds
    strict = TBoostRegressor(**common, prune_min_stability=1.0).fit(x, y)  # 5 of 5
    n_lenient = len(lenient.pruning_report_["kept"])
    n_strict = len(strict.pruning_report_["kept"])
    assert n_strict < n_lenient, (n_lenient, n_strict)
    assert strict.pruning_report_["min_stability"] == 1.0


def test_prune_min_mean_gain_is_a_parsimony_bar_only_with_the_evidence_gate_off() -> None:
    # The gain floor applies to the LEGACY keep rule. With the av37 gate armed, a table the
    # floor rejects becomes an AMBIGUOUS candidate, and the gate's rule is "keep unless the fold
    # evidence is significantly negative" — so raising the floor hands tables to the gate and
    # the bank can GROW. Both directions are pinned here because the surprising one is the
    # default configuration, and a test that only checked the gate-off case would let the
    # docstring drift back to claiming a monotone parsimony bar.
    x, y = _marginal_poisson(n=8000, noise=0.05)
    common = dict(objective="poisson", n_trees=200, n_bags=1, seed=0, prune=True, graduate=False)

    gated_lo = TBoostRegressor(**common, prune_min_mean_gain=0.0).fit(x, y)
    gated_hi = TBoostRegressor(**common, prune_min_mean_gain=1e-3).fit(x, y)
    assert gated_hi.pruning_report_["legacy_selected"] < gated_lo.pruning_report_["legacy_selected"]
    assert len(gated_hi.pruning_report_["kept"]) >= len(gated_lo.pruning_report_["kept"])

    open_lo = TBoostRegressor(**common, prune_drop_z=None, prune_min_mean_gain=0.0).fit(x, y)
    open_hi = TBoostRegressor(**common, prune_drop_z=None, prune_min_mean_gain=1e-3).fit(x, y)
    assert len(open_hi.pruning_report_["kept"]) < len(open_lo.pruning_report_["kept"])


def test_aggregator_knobs_are_live_under_multiclass_cv_and_raise_on_the_legacy_split() -> None:
    # prune_min_stability / prune_min_mean_gain live in `aggregate_prune_selection`. Since
    # 2026-09-07 the K>=3 path selects on fold refits through that same aggregator, so the
    # knobs are honored and echoed in the report; the legacy single-split regime
    # (`multiclass_prune_cv=False`) still has no fold vote to threshold and raises.
    rng = np.random.default_rng(2)
    n = 600
    x = rng.normal(size=(n, 3)).astype(np.float32)
    y = np.argmax(np.stack([x[:, 0], x[:, 1], -x[:, 0]], axis=1), axis=1)
    common = dict(objective="logistic", n_trees=30, n_bags=1, seed=0, prune=True)
    cv = TBoostClassifier(**common, prune_min_stability=1.0, prune_min_mean_gain=1e-3).fit(x, y)
    sel = cv.pruning_report_["selection"]
    assert sel["min_stability"] == 1.0 and sel["min_mean_gain"] == 1e-3
    assert sel["cv_folds"] == cv.pruning_report_["cv_folds"] >= 2
    assert "fold_table_scores" in sel and "evidence_scores" in sel
    legacy = dict(common, multiclass_prune_cv=False)
    with pytest.raises(ValueError, match="prune_min_stability"):
        TBoostClassifier(**legacy, prune_min_stability=1.0).fit(x, y)
    with pytest.raises(ValueError, match="prune_min_mean_gain"):
        TBoostClassifier(**legacy, prune_min_mean_gain=1e-3).fit(x, y)
    # prune_validation_fraction only governs the legacy split: dead (and so refused) under CV.
    with pytest.raises(ValueError, match="prune_validation_fraction"):
        TBoostClassifier(**common, prune_validation_fraction=0.3).fit(x, y)
    old = TBoostClassifier(**legacy, prune_validation_fraction=0.3).fit(x, y)
    assert old.pruning_report_["selector"] == "single_split_walk"
    assert old.pruning_report_["cv_folds"] == 0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(prune_min_stability=0.0), "prune_min_stability"),
        (dict(prune_min_stability=1.5), "prune_min_stability"),
        (dict(prune_min_mean_gain=-1.0), "prune_min_mean_gain"),
        (dict(prune_fold_min_rows=0), "prune_fold_min_rows"),
        (dict(prune_fold_es_patience=0), "prune_fold_es_patience"),
        (dict(prune_guard_min_rows=0), "prune_guard_min_rows"),
    ],
)
def test_exposed_prune_internals_reject_out_of_range_values(kwargs, match) -> None:
    x, y = _noisy_poisson(n=600)
    with pytest.raises(ValueError, match=match):
        TBoostRegressor(
            objective="poisson", n_trees=30, n_bags=1, seed=0, prune=True, **kwargs
        ).fit(x, y)


def test_prune_fold_es_patience_param_explicit_and_default_both_fit() -> None:
    # An explicit value and None (the shipped 250) must both produce a pruned model.
    x, y = _noisy_poisson(n=800, seed=1)
    common = dict(objective="poisson", n_trees=40, n_bags=1, seed=0, prune=True, graduate=False)
    explicit = TBoostRegressor(**common, prune_fold_es_patience=3).fit(x, y)
    assert isinstance(explicit._model, _TableModel)
    inherited = TBoostRegressor(**common, prune_fold_es_patience=None).fit(x, y)
    assert isinstance(inherited._model, _TableModel)


def test_prune_validation_fraction_raises_on_scalar_path_when_non_default() -> None:
    # prune_validation_fraction only has a target on the multiclass path (its single
    # train/select split); the scalar (regressor/binary) path uses prune_n_folds-fold CV instead,
    # so a non-default value here cannot be honored and must raise rather than be ignored.
    x, y = _noisy_poisson(n=1000)
    common = dict(objective="poisson", n_trees=60, n_bags=1, seed=0, prune=True, graduate=False)
    with pytest.raises(ValueError, match="prune_validation_fraction"):
        TBoostRegressor(**common, prune_validation_fraction=0.3).fit(x, y)
    # The default (0.15) is inert on this path but must not raise.
    est = TBoostRegressor(**common, prune_validation_fraction=0.15).fit(x, y)
    assert isinstance(est._model, _TableModel)


def test_binary_classifier_prune_round_trips() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2000, 4)).astype(np.float32)
    y = (rng.uniform(size=2000) < 1.0 / (1.0 + np.exp(-(0.8 * x[:, 0] - 0.5 * x[:, 1])))).astype(int)
    est = TBoostClassifier(objective="logistic", n_trees=200, n_bags=1, seed=0, prune=True).fit(x, y)
    assert isinstance(est._model, _TableModel)
    proba = est.predict_proba(x[:30])
    back = TBoostClassifier.from_bytes(est.to_bytes())
    assert np.allclose(proba, back.predict_proba(x[:30]), atol=1e-5)


def _expand_1d(dep_axes: list, rep_table: dict) -> np.ndarray:
    """Rep table values expanded onto a deployed 1-D axis grid via its borders."""
    b = np.asarray(dep_axes[0]["borders"], dtype=float)
    mids = (b[:-1] + b[1:]) / 2.0
    pts = np.concatenate([[b[0] - 1.0], mids, [b[-1] + 1.0]])
    rb = np.asarray(rep_table["axes"][0]["borders"], dtype=float)
    rv = np.asarray(rep_table["values"]["data"]["Dense"], dtype=float)
    idx = np.concatenate([[0], 1 + np.searchsorted(rb, pts, side="left")]).astype(int)
    return rv[np.clip(idx, 0, len(rv) - 1)]


def test_bag_bank_jsons_returns_honest_replicates_behind_the_soup() -> None:
    import json

    # Near-discrete single feature so every bag realizes the same exact bin grid.
    rng = np.random.default_rng(5)
    x = (rng.integers(0, 9, size=4000).astype(np.float32)).reshape(-1, 1)
    mu = np.exp(0.25 * np.sin(x[:, 0]))
    y = rng.poisson(mu).astype(np.float32)
    est = ensemble_fit(
        TBoostRegressor(objective="poisson", n_trees=200, n_bags=4, seed=0, prune=False), x, y
    )  # bag_bank_jsons is the UNPRUNED _Model's replicate API
    assert isinstance(est._model, _Model)

    keep = [[0]]
    bag_jsons = est._model.bag_bank_jsons(np.ascontiguousarray(x), keep)
    assert len(bag_jsons) == 4, "one bank per bag, in bag order"
    banks = [json.loads(b) for b in bag_jsons]
    for b in banks:
        assert {tuple(t["u"]) for t in b["tables"]} <= {(0,)}

    # The bag banks are the replicate values BEHIND the soup: their cell-mean reproduces the
    # drop-only (rebalance=False) deployed bank. Expansion maps each bag's realized-cut grid
    # onto the deployed grid; tolerance covers the Laplace ref-measure asymmetry between
    # coarse per-bag merged grids and the soup's finer one.
    plain = est._model.apply_keepset(
        np.ascontiguousarray(x), np.ascontiguousarray(y),
        np.ascontiguousarray(np.ones_like(y)), keep, reanchor=True, rebalance=False,
    )
    dep = json.loads(plain.to_json())["model"]["bank"]["tables"][0]
    v = np.asarray(dep["values"]["data"]["Dense"], dtype=float)
    acc = np.zeros_like(v)
    for b in banks:
        rep = {tuple(t["u"]): t for t in b["tables"]}.get((0,))
        acc += _expand_1d(dep["axes"], rep) if rep is not None else 0.0
    # Tightened from 0.02 after the shared-grid/shared-measure per-bag purify fix: every bag
    # now purifies on the soup's own MergedGrids and weights, so only f32 fit noise remains.
    assert np.allclose(acc / len(banks), v, atol=1e-4), (
        f"mean-of-bags must reproduce the soup bank; max diff "
        f"{np.max(np.abs(acc / len(banks) - v)):.6f}"
    )

    # No bag partition -> typed error: single fits and wire round-trips both lack it.
    single = ensemble_fit(
        TBoostRegressor(objective="poisson", n_trees=50, n_bags=1, seed=0, prune=False), x, y
    )
    with pytest.raises(Exception, match="bag partition"):
        single._model.bag_bank_jsons(np.ascontiguousarray(x), keep)
    back = TBoostRegressor.from_bytes(est.to_bytes())
    with pytest.raises(Exception, match="bag partition"):
        back._model.bag_bank_jsons(np.ascontiguousarray(x), keep)


def test_graduate_default_smooths_every_scalar_objective_on_the_whole_fit() -> None:
    # Any scalar objective may smooth, and `graduate=None` resolves to ON. No rows are
    # withheld to decide it. Explicit False disables smoothing outright.
    x, y = _noisy_poisson(1500, 3)
    p = TBoostRegressor(objective="poisson", n_trees=120, n_bags=1, seed=0, prune=True).fit(x, y)
    assert p.get_params()["graduate"] is None
    assert isinstance(getattr(p, "graduation_report_", None), list)
    t = TBoostRegressor(objective="tweedie", n_trees=120, n_bags=1, seed=0, prune=True).fit(x, y)
    assert isinstance(t.graduation_report_, list)
    assert t.graduation_validation_["evidence"] == "none"
    assert t.graduation_validation_["holdout_rows"] == 0
    assert t.graduation_validation_["training_rows"] == len(y)
    f = TBoostRegressor(
        objective="poisson", n_trees=120, n_bags=1, seed=0, prune=True, graduate=False
    ).fit(x, y)
    assert getattr(f, "graduation_report_", None) is None


def test_graduation_smooths_ordinal_axes_only() -> None:
    # Categorical-TS axes have no ordinal adjacency: pure-cat tables never enter the
    # graduation report; numeric mains do. (euhealth 2026-07-10: smoothing along the
    # risk-sorted cat encoding carried a consistent test-deviance regression.)
    rng = np.random.default_rng(11)
    n = 2400
    num = (rng.integers(0, 30, size=n)).astype(np.float32)
    levels = np.asarray(["a", "b", "c", "d", "e"], dtype=object)
    cat = levels[rng.integers(0, 5, size=n)]
    mu = np.exp(0.03 * num + np.where(cat == "a", 0.6, np.where(cat == "b", -0.5, 0.0)))
    y = rng.poisson(mu).astype(np.float32)
    x = np.empty((n, 2), dtype=object)
    x[:, 0] = num
    x[:, 1] = cat
    est = TBoostRegressor(
        objective="poisson", n_trees=150, n_bags=1, seed=0, prune=True, graduate=True,
        categorical_features=[1],
    ).fit(x, y)
    feats = [r["features"] for r in est.graduation_report_]
    assert [1] not in feats, "pure categorical tables must not be graduated"
    assert [0] in feats, "the numeric main should be considered for graduation"


def test_dr_family_matches_dense_gcv_solve() -> None:
    # The Demmler-Reinsch factorization must reproduce the dense per-alpha solve exactly
    # (same estimator, same GCV curve, same chosen alpha) — including tables with
    # zero-support cells, which are eliminated via the alpha-free Schur complement.
    from t_boost.sklearn import TBoostRegressor as R

    rng = np.random.default_rng(7)
    for k, zeros in ((21, 0), (40, 5), (12, 3)):
        d2 = R._d2(k)
        pen = d2.T @ d2
        v = rng.normal(scale=0.2, size=k)
        s = rng.uniform(20.0, 500.0, size=k)
        if zeros:
            s[rng.choice(k, size=zeros, replace=False)] = 0.0
        fam = R._dr_family(v, s, pen)
        assert fam is not None
        w = np.diag(s)
        for alpha in (1.0, 30.0, 1e3, 3e4):
            v_dr, df_dr = fam(alpha)
            x = np.linalg.solve(w + alpha * pen, w)
            v_dense = x @ v
            assert np.allclose(v_dr, v_dense, atol=1e-9), (k, zeros, alpha)
            assert abs(df_dr - float(np.trace(x))) < 1e-8, (k, zeros, alpha)
            v_fixed, fixed_alpha = R._gcv_smooth(v, s, pen, alpha)
            assert fixed_alpha == alpha
            assert np.allclose(v_fixed, v_dense, atol=1e-9), (k, zeros, alpha)
        # end-to-end: same alpha chosen by _gcv_smooth either way
        got = R._gcv_smooth(v, s, pen, None)
        import unittest.mock as um
        with um.patch.object(R, "_dr_family", staticmethod(lambda *a: None)):
            want = R._gcv_smooth(v, s, pen, None)
        assert got[1] == want[1]
        assert np.allclose(got[0], want[0], atol=1e-9)


def test_graduation_penalty_kron_identity_is_bit_exact() -> None:
    # The 2026-07-16 penalty assembly (kron of the small Gram, not the big-operator dgemm)
    # must be BIT-identical to the operator form it replaced — graduation output feeds the
    # deployed model, which sits behind the bit-repro release gate.
    d2 = TBoostRegressor._d2
    for m, n in ((3, 4), (7, 5), (23, 11), (50, 50)):
        d = d2(m)
        dr = np.kron(d, np.eye(n))
        assert np.array_equal(np.kron(d.T @ d, np.eye(n)), dr.T @ dr), (m, n)
        d = d2(n)
        dc = np.kron(np.eye(m), d)
        assert np.array_equal(np.kron(np.eye(m), d.T @ d), dc.T @ dc), (m, n)


def test_graduation_family_handles_singular_zero_support_block() -> None:
    # Sparse 2-way surfaces (empty regions with flat D2 directions) used to fall off the
    # Demmler-Reinsch path into a ~30x O(k^3)-per-table dense fallback that also silently
    # SKIPPED every alpha whose full system was singular. The pinv Schur reduction
    # (2026-07-16, Ralph-ratified) keeps them on the one-eigendecomposition path and
    # scores the whole alpha grid.
    # The construction mirrors the real failing MTPL tables: a numeric x CATEGORICAL grid
    # (the penalty smooths only along the numeric axis), with one categorical level fully
    # unobserved. That level's cells form an unanchored D2 chain — null space = constants
    # + linears, exactly singular (min eig ~1e-16 on the real tables).
    rng = np.random.default_rng(0)
    m, n = 10, 3
    v = rng.normal(size=m * n)
    s = rng.uniform(1.0, 5.0, size=(m, n))
    s[:, 2] = 0.0  # fully-empty categorical level -> unanchored numeric chain
    s = s.ravel()
    d2 = TBoostRegressor._d2
    pen = np.kron(d2(m).T @ d2(m), np.eye(n))  # cat axis contributes no smoothing direction
    p00 = pen[np.ix_(s == 0, s == 0)]
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(p00)  # guards the test's premise: the block IS singular

    fam = TBoostRegressor._dr_family(v, s, pen)
    assert fam is not None, "singular P00 must take the pinv family path, not the dense fallback"
    n_obs = int((s > 0).sum())
    for alpha in (0.5, 5.0):
        v_s, df = fam(alpha)
        assert np.all(np.isfinite(v_s))
        assert 0.0 < df <= n_obs
        try:
            dense = np.linalg.solve(np.diag(s) + alpha * pen, s * v)
        except np.linalg.LinAlgError:
            continue  # full system singular at this alpha: the family is the only evaluator
        np.testing.assert_allclose(v_s[s > 0], dense[s > 0], rtol=1e-8, atol=1e-10)


def test_graduation_separable_family_matches_joint_path() -> None:
    # Single-smoothed-direction 2-D tables (numeric x cat) decompose exactly: kron(D'D, I)
    # is block-diagonal per categorical level. The separable family must reproduce the
    # joint eigendecomposition's alphas and values (2026-07-16; measured 2.6x on the
    # graduation loop, 1e-16 agreement on the real MTPL tables).
    rng = np.random.default_rng(7)
    m, n = 24, 6
    item = (
        0,
        [0, 5],
        (m, n),
        rng.normal(size=m * n).tolist(),
        np.where(rng.uniform(size=m * n) < 0.35, 0.0, rng.uniform(1, 50, m * n)).tolist(),
        [False, True],  # numeric x categorical -> one smoothed direction
    )
    # make one categorical level fully unobserved (the empty-level zero-assignment case)
    s = np.asarray(item[4]).reshape(m, n)
    s[:, 4] = 0.0
    item = (*item[:4], s.ravel().tolist(), item[5])

    new = TBoostRegressor._graduate_one(item, None, 1.0)
    orig = TBoostRegressor._dr_family_separable
    try:
        TBoostRegressor._dr_family_separable = classmethod(lambda cls, v2, s2: None)
        joint = TBoostRegressor._graduate_one(item, None, 1.0)
    finally:
        TBoostRegressor._dr_family_separable = orig
    assert new is not None and joint is not None
    assert new[3] == joint[3], "separable path must select the same alpha"
    np.testing.assert_allclose(new[2], joint[2], rtol=1e-10, atol=1e-12)


def test_graduation_fill_clamp_bounds_extrapolation() -> None:
    # Zero-support stationary fills are UNBOUNDED affine extrapolations on the smoothed
    # scale (2026-07-16, review panel finding): a categorical level observed at only 2
    # ADJACENT chain positions gets a straight line through those 2 points projected across
    # the whole rest of the chain. Fixed alpha (rather than GCV selection) exercises the
    # same fill+clamp path deterministically.
    m_raw, n = 25, 3  # raw numeric chain; 24 once row 0 (reserved missing bin) is excluded
    shape = (m_raw, n)
    v2 = np.zeros(shape)
    s2 = np.zeros(shape)
    # Well-observed, smooth columns (1, 2): solid support, slowly varying values.
    chain = np.arange(m_raw, dtype=float)
    for col in (1, 2):
        v2[:, col] = 0.01 * chain + 0.1 * col
        s2[:, col] = 40.0
    # Sparse column (0): observed ONLY at raw rows 1 and 2 (the first two positions of the
    # smoothed slice once row 0 is excluded), far apart in value -> steep local slope.
    v2[1, 0], v2[2, 0] = 0.0, 5.0
    s2[1, 0], s2[2, 0] = 10.0, 10.0
    axis_cat = [False, True]  # numeric x categorical
    item = (0, [0, 3], shape, v2.ravel().tolist(), s2.ravel().tolist(), axis_cat)

    result = TBoostRegressor._graduate_one(item, 3.0, 1.0)
    assert result is not None
    _, _, out, alpha = result
    assert alpha == 3.0

    out2 = out.reshape(shape)
    smoothed_support = s2[1:, :]  # excludes the reserved bin-0 row
    smoothed_out = out2[1:, :]
    observed = smoothed_support > 0
    lo = min(float(smoothed_out[observed].min()), 0.0)
    hi = max(float(smoothed_out[observed].max()), 0.0)
    assert np.all(smoothed_out[~observed] >= lo - 1e-9)
    assert np.all(smoothed_out[~observed] <= hi + 1e-9)

    # Prove the test actually bites: WITHOUT the clamp, the raw stationary fill for the
    # sparse column would have exceeded these bounds (recompute it inline via the same
    # D2 penalty the separable path uses internally for a chain of this length).
    chain_len = m_raw - 1  # the smoothed slice's own chain length (24)
    d2 = TBoostRegressor._d2(chain_len)
    pen = d2.T @ d2
    sub_v, sub_s = v2[1:, 0], s2[1:, 0]
    fam = TBoostRegressor._dr_family(sub_v, sub_s, pen)
    assert fam is not None
    unclamped, _ = fam(3.0)
    sub_observed = sub_s > 0
    assert np.any(unclamped[~sub_observed] < lo - 1e-6) or np.any(
        unclamped[~sub_observed] > hi + 1e-6
    ), "construction must produce an extrapolation the clamp actually needs to bite"


def test_graduation_separable_cap_admits_long_chains() -> None:
    # A [300, 12] numeric x categorical table (3600 cells, over the 2500-cell JOINT cap)
    # must still be graduated: its true cost is 12 independent O(300^3) per-level chains
    # (the SEPARABLE path), not one O(3600^3) joint solve, so it is gated on the smoothed
    # chain length (<=512), not total cells (2026-07-16 panel finding).
    m, n = 300, 12
    chain = np.arange(m, dtype=float)
    v2 = np.sin(chain / 30.0)[:, None] + 0.1 * np.arange(n)[None, :]
    s2 = np.full((m, n), 20.0)
    item = (0, [0, 1], (m, n), v2.ravel().tolist(), s2.ravel().tolist(), [False, True])
    result = TBoostRegressor._graduate_one(item, 1.0, 1.0)
    assert result is not None, "a cheap separable table must not be excluded by the joint-cost cap"

    # A [60, 60] BOTH-numeric table (3600 cells too) IS a joint 2-D solve (one dense
    # O((m*n)^3) eigendecomposition over the flattened grid) and must stay capped on cells.
    rng = np.random.default_rng(0)
    m2 = n2 = 60
    v2b = rng.normal(scale=0.2, size=(m2, n2))
    s2b = np.full((m2, n2), 20.0)
    item2 = (0, [0, 1], (m2, n2), v2b.ravel().tolist(), s2b.ravel().tolist(), [False, False])
    result2 = TBoostRegressor._graduate_one(item2, 1.0, 1.0)
    assert result2 is None, "the joint 2-D path must stay capped on total cells"


def test_fold_patience_override_shortens_the_fold_booster() -> None:
    # The prune-CV fold booster's early_stopping_rounds override, never the deployed model's.
    # Direct plumbing check: the pyo3 `_Booster` exposes no getter for its resolved
    # early_stopping_rounds, so the override is verified by its EFFECT — a 1-round patience
    # must retain far fewer trees than a 50-round patience on identical data (both isolated
    # from the adaptive-patience and min-delta knobs so only the override itself varies).
    base = TBoostRegressor(
        objective="poisson", n_trees=300, n_bags=1, seed=0,
        validation_fraction=0.2, early_stopping_rounds=50,
        early_stopping_adaptive=None, early_stopping_min_delta=0.0,
    )
    tight = base._new_booster(early_stopping_rounds_override=1)
    default = base._new_booster()
    # seed=11 (not the module's usual seed=2): since the y>0-stratified Poisson ES holdout
    # (ES-runaway fix #2, 2026-07-20) is honest rather than occasionally zero-starved, the
    # validation curve converges more cleanly in general and several seeds no longer wobble
    # past their best round at all — collapsing tight/default to the SAME n_trees regardless of
    # patience (a real, desirable effect of the fix, but it stops THIS plumbing check from
    # telling override-does-nothing apart from override-had-nothing-left-to-shorten). seed=11
    # keeps a comfortably wide margin (tight << default) so the assertion still isolates the
    # override's effect.
    x2, y2 = _noisy_poisson(n=2000, seed=11)
    tight_model = tight.fit(x2, y2)
    default_model = default.fit(x2, y2)
    assert 0 < int(tight_model.n_trees) < int(default_model.n_trees), (
        "early_stopping_rounds_override must actually shorten the fold booster's patience"
    )


def test_fold_patience_default_is_250_and_explicit_250_reproduces_it() -> None:
    # Promoted default (2026-07-16): fold fits vote at patience 250; passing the same value
    # explicitly must be byte-identical to the default.
    from t_boost.sklearn import _PRUNE_FOLD_ES_PATIENCE

    assert _PRUNE_FOLD_ES_PATIENCE == 250
    x, y = _noisy_poisson(n=1200, seed=3)
    a = TBoostRegressor(
        objective="poisson", n_trees=120, n_bags=1, seed=0, prune=True, graduate=False
    ).fit(x, y)
    b = TBoostRegressor(
        objective="poisson", n_trees=120, n_bags=1, seed=0, prune=True, graduate=False,
        prune_fold_es_patience=250,
    ).fit(x, y)
    assert model_bytes(a) == model_bytes(b)


@pytest.mark.parametrize("objective", ["poisson", "gamma", "tweedie"])
def test_grouped_prune_fit_honors_explicit_reanchor(objective: str) -> None:
    # Regression (2026-07-21): `_fit_and_prune` bound `obj` only inside its `reanchor is None`
    # branch, then read it unconditionally for the ES-strata group carve. An EXPLICIT reanchor —
    # which `recommended_recipe(tuned=True)` sets for every log-link objective — left `obj`
    # unbound, so every grouped tuned deploy fit died with UnboundLocalError (insur-arena
    # fremotor_prem: 9/9 splits NaN). Explicit True and the resolved default must both fit.
    rng = np.random.default_rng(11)
    n, n_groups = 2400, 800
    groups = rng.integers(0, n_groups, size=n)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    mu = np.exp(-0.4 + 0.4 * x[:, 0] + 0.3 * x[:, 1])
    y = (
        rng.gamma(2.0, mu / 2.0) if objective == "gamma" else rng.poisson(mu)
    ).astype(np.float32)
    preds = {}
    for reanchor in (True, None):
        est = TBoostRegressor(
            objective=objective, n_trees=80, n_bags=1, seed=0, prune=True, reanchor=reanchor
        ).fit(x, y, groups=groups)
        p = np.asarray(est.predict(x), np.float64)
        assert np.isfinite(p).all(), f"{objective}/reanchor={reanchor} produced non-finite predictions"
        preds[reanchor] = p
    # These objectives all resolve the None default to True, so both fits must agree exactly:
    # hoisting `obj` changed no behavior on the path that already worked.
    assert np.array_equal(preds[True], preds[None])


def test_recipe_tuned_grouped_deploy_fit_is_finite() -> None:
    # End-to-end guard on the exact insur-arena tuned deploy configuration that failed:
    # recommended_recipe(tuned=True) on a log-link objective + panel groups + prune=True.
    from t_boost import recommended_recipe

    rng = np.random.default_rng(12)
    n, n_groups = 2400, 800
    groups = rng.integers(0, n_groups, size=n)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    mu = np.exp(0.3 * x[:, 0] + 0.2 * x[:, 1] + 1.0)
    y = rng.gamma(2.0, mu / 2.0).astype(np.float32)
    est = recommended_recipe("gamma", budget=80, n_jobs=2, tuned=True, seed=0)
    assert est.reanchor is True and est.prune is True  # the configuration under test
    est.fit(x, y, sample_weight=np.ones(n), groups=groups)
    assert np.isfinite(np.asarray(est.predict(x), np.float64)).all()


def test_bag_membership_and_oob_scorer_are_thread_and_layout_deterministic() -> None:
    # The native primitives behind the ranked path's evidence: the recorded membership, the per-bag
    # keep-restricted raw scorer, and the one-pass OOB group accumulator.
    rng = np.random.default_rng(11)
    n = 4000
    x = np.column_stack([rng.uniform(0, 1, n), rng.uniform(0, 1, n)]).astype(np.float32)
    mu = np.exp(0.4 * np.sin(5 * x[:, 0]) + 0.3 * (x[:, 1] > 0.5))
    y = rng.poisson(mu).astype(np.float32)
    est = ensemble_fit(TBoostRegressor(graduate=False, objective="poisson", n_trees=120, n_bags=4,
                                       bag_subsample=0.7, seed=0, prune=False), x, y)
    m = est._model
    assert isinstance(m, _Model)

    mask = m.bag_in_bag_mask()
    assert mask.shape == (4, n) and mask.dtype == np.bool_
    # bag_subsample=0.7 without replacement ⇒ exactly round(0.7n) distinct rows per bag.
    assert (mask.sum(axis=1) == round(0.7 * n)).all()
    assert (~mask).any(axis=1).all(), "every bag must leave rows out of bag"

    xc = np.ascontiguousarray(x)
    supports = sorted({tuple(int(i) for i in u) for u in m.table_supports(xc)})
    keep = [list(supports[0])]
    full = m.bag_raw_scores(xc, n_jobs=1)
    assert full.shape == (4, n)
    assert np.array_equal(full, m.bag_raw_scores(xc, n_jobs=4)), "thread count must not move it"
    # A keep-restriction can only remove table mass, and an empty keep-set is intercept-only.
    restricted = m.bag_raw_scores(xc, keep=keep)
    assert restricted.shape == full.shape and not np.array_equal(restricted, full)
    intercept_only = m.bag_raw_scores(xc, keep=[])
    assert np.allclose(intercept_only, intercept_only[:, :1], atol=1e-12)
    # A row subset scores the same values as the corresponding full-row slots.
    sub = m.bag_raw_scores(xc, rows=[7, 3])
    assert np.allclose(sub[:, 0], full[:, 7]) and np.allclose(sub[:, 1], full[:, 3])

    # The one-pass accumulator: jury = the OOB complement, and groups sum back to the full bank.
    groups = [[list(u) for u in supports[:1]], [list(u) for u in supports[1:]]]
    fs, f0s, gs, counts = m.bag_oob_group_raw(xc, groups, n_jobs=1)
    assert np.array_equal(counts, (~mask).sum(axis=0).astype(counts.dtype))
    cov = counts > 0
    assert np.allclose(f0s[cov] + gs[:, cov].sum(axis=0), fs[cov], atol=1e-8)
    # The jury really is the OOB set: rebuild the sum from the per-bag scorer.
    expected = (full * (~mask)).sum(axis=0)
    assert np.allclose(fs, expected, atol=1e-8)
    for arr, ref in zip(m.bag_oob_group_raw(xc, groups, n_jobs=4), (fs, f0s, gs, counts)):
        assert np.array_equal(arr, ref), "thread count must not move the evidence"

    # Membership and per-bag banks are runtime-only: a wire round-trip has neither.
    back = TBoostRegressor.from_bytes(est.to_bytes())
    with pytest.raises(Exception, match="bag"):
        back._model.bag_in_bag_mask()
    single = TBoostRegressor(graduate=False, objective="poisson", n_trees=50, n_bags=1, seed=0,
                               prune=False).fit(x, y)
    with pytest.raises(Exception, match="bag"):
        single._model.bag_in_bag_mask()


# --- EVIDENCE-GATED DROPPING (av37) ---------------------------------------------------------


def _egp_fixture(n: int = 2400, seed: int = 3):
    """A small, noisy Poisson frame with four weakly-informative features: the regime the
    forensic identified, where the prune-CV folds cannot resolve an interaction's sign and the
    pre-av37 rule therefore dropped on a coin flip."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    mu = np.exp(0.25 * x[:, 0] + 0.2 * x[:, 1] + 0.1 * x[:, 2]
                + 0.12 * x[:, 0] * x[:, 1] + 0.08 * x[:, 1] * x[:, 2])
    y = rng.poisson(mu).astype(np.float32)
    return x, y


def _kept(est) -> set:
    return {tuple(sorted(int(i) for i in u)) for u in est.pruning_report_["kept"]}


def test_evidence_gate_off_is_the_legacy_selection_and_reports_it_as_such() -> None:
    # `prune_drop_z=None` must leave the aggregator exactly where av36 had it: no evidence
    # scores, no admissions, and the report's own bookkeeping saying so.
    x, y = _egp_fixture()
    kw = dict(objective="poisson", n_trees=200, n_bags=4, seed=0, prune=True, n_jobs=2)
    off = TBoostRegressor(graduate=False, **kw, prune_drop_z=None).fit(x, y)
    r = off.pruning_report_
    assert r["drop_z"] is None
    assert r["evidence_scores"] == [] and r["evidence_admitted"] == []
    assert r["evidence_candidates"] == 0 and r["evidence_budget_bound"] is False
    # A zero budget can admit nothing (budget = max(0, legacy set size) = the legacy set), so
    # the gate is forced back onto the legacy selection and must reproduce it TO THE BYTE.
    pinned = TBoostRegressor(graduate=False, **kw, prune_drop_z=2.0, prune_keep_budget=0).fit(x, y)
    assert pinned.pruning_report_["evidence_admitted"] == []
    assert model_bytes(pinned) == model_bytes(off)


def test_evidence_gate_only_ever_keeps_more_never_fewer() -> None:
    # The monotonicity that makes the gate safe to reason about: the legacy rule keeps on
    # `mean_gain > 0` and the gate drops on `mean < -z*se`, so the two can never disagree in
    # the shrinking direction. Rising z can only widen the keep-set.
    x, y = _egp_fixture()
    kw = dict(objective="poisson", n_trees=200, n_bags=4, seed=0, prune=True, n_jobs=2,
              prune_keep_budget=1000)
    legacy = _kept(TBoostRegressor(graduate=False, **kw, prune_drop_z=None).fit(x, y))
    prev = legacy
    for z in (0.5, 1.0, 2.0, 4.0):
        now = _kept(TBoostRegressor(graduate=False, **kw, prune_drop_z=z).fit(x, y))
        assert legacy <= now, f"z={z} dropped a table the legacy rule kept"
        assert prev <= now, f"z={z} kept fewer tables than a smaller z"
        prev = now
    assert legacy < prev, "fixture must actually exercise the gate"


def test_keep_budget_caps_ambiguous_admissions_and_says_when_it_bound() -> None:
    x, y = _egp_fixture()
    kw = dict(objective="poisson", n_trees=200, n_bags=4, seed=0, prune=True, n_jobs=2,
              prune_drop_z=3.0)
    wide = TBoostRegressor(graduate=False, **kw, prune_keep_budget=1000).fit(x, y)
    rw = wide.pruning_report_
    assert rw["evidence_budget_bound"] is False
    assert len(rw["evidence_admitted"]) == rw["evidence_candidates"] > 0
    # A budget strictly between the legacy set and the unconstrained one must bind, admit the
    # HIGHEST-ranked candidates only, and land inside the two.
    tight_budget = rw["legacy_selected"] + 1
    tight = TBoostRegressor(graduate=False, **kw, prune_keep_budget=tight_budget).fit(x, y)
    rt = tight.pruning_report_
    assert rt["evidence_budget_bound"] is True
    assert rt["evidence_budget"] == tight_budget
    assert 0 < len(rt["evidence_admitted"]) < len(rw["evidence_admitted"])
    assert [tuple(u) for u in rt["evidence_admitted"]] == \
        [tuple(u) for u in rw["evidence_admitted"][: len(rt["evidence_admitted"])]]
    assert len(_kept(tight)) < len(_kept(wide))


def test_evidence_gate_is_deterministic_across_thread_counts() -> None:
    x, y = _egp_fixture()
    kw = dict(objective="poisson", n_trees=200, n_bags=4, seed=0, prune=True,
              prune_drop_z=1.5, prune_keep_budget=1000)
    a = TBoostRegressor(graduate=False, **kw, n_jobs=1).fit(x, y)
    b = TBoostRegressor(graduate=False, **kw, n_jobs=4).fit(x, y)
    assert a.pruning_report_["kept"] == b.pruning_report_["kept"]
    assert a.pruning_report_["evidence_admitted"] == b.pruning_report_["evidence_admitted"]
    assert model_bytes(a) == model_bytes(b)


def test_evidence_gate_params_roundtrip() -> None:
    est = TBoostRegressor(graduate=False, prune_drop_z=2.5, prune_keep_budget=41)
    p = est.get_params()
    assert (p["prune_drop_z"], p["prune_keep_budget"]) == (2.5, 41)
    est2 = TBoostRegressor(graduate=False, ).set_params(prune_drop_z=None, prune_keep_budget=7)
    assert est2.prune_drop_z is None and est2.prune_keep_budget == 7
    clf = TBoostClassifier(prune_drop_z=0.5, prune_keep_budget=9)
    assert clf.get_params()["prune_drop_z"] == 0.5
    assert clf.get_params()["prune_keep_budget"] == 9


# --- PINNED-BANK FOLD FIDELITY (`prune_fold_fidelity`) ---------------------------------------


def _fidelity_fit(**kw):
    """A 6-feature Poisson fit wide enough that the single-bag prune folds cannot realize every
    support the deploy bank does — which is the whole condition the fidelity knob addresses."""
    rng = np.random.default_rng(7)
    n = 3000
    x = rng.normal(size=(n, 6)).astype(np.float32)
    mu = np.exp(0.3 * x[:, 0] + 0.2 * x[:, 1] + 0.2 * x[:, 0] * x[:, 1] + 0.1 * x[:, 2])
    y = rng.poisson(mu).astype(np.float32)
    common = dict(
        objective="poisson",
        n_trees=120,
        n_bags=4,
        seed=0,
        prune=True,
        graduate=False,
        prune_drop_z=2.0,
        max_interaction_order=3,
    )
    common.update(kw)
    return TBoostRegressor(**common).fit(x, y)


def test_fold_fidelity_default_off_is_byte_identical() -> None:
    """The knob's whole safety claim: absent or False, the fit is the SAME BYTES. Not "close",
    not "same score" — the serialized artifact, which is what the benchmark cache keys on."""
    a = _fidelity_fit()
    b = _fidelity_fit(prune_fold_fidelity=False)
    assert a._model.to_bytes() == b._model.to_bytes()
    assert a.pruning_report_["fold_fidelity"] is False


def test_fold_fidelity_covers_every_candidate_with_paired_evidence() -> None:
    """OFF, most candidates carry NO paired per-fold evidence (`gain_n < 2`) and the av37 gate
    falls back to the legacy verdict on absence. ON, every candidate is scored by every fold."""
    off = _fidelity_fit().pruning_report_
    on = _fidelity_fit(prune_fold_fidelity=True).pruning_report_
    assert on["fold_fidelity"] is True
    uncovered_off = [r for r in off["evidence_scores"] if r["gain_n"] < 2]
    uncovered_on = [r for r in on["evidence_scores"] if r["gain_n"] < 2]
    # The defect is real on this fit, and the knob closes it completely.
    assert len(uncovered_off) > 0
    assert uncovered_on == []
    n_folds = on["cv_folds"]
    assert all(r["gain_n"] == n_folds for r in on["evidence_scores"])
    # And the evidence is a real measurement, not the rejected imputed zero: the paired gains
    # have spread, so `se_gain` is a number the gate can divide by.
    assert any((r["se_gain"] or 0.0) > 0.0 for r in on["evidence_scores"])


def test_fold_fidelity_report_field_survives_round_trip() -> None:
    est = _fidelity_fit(prune_fold_fidelity=True)
    assert json.loads(json.dumps(est.pruning_report_))["fold_fidelity"] is True
    assert isinstance(est._model, _TableModel)


def test_fold_fidelity_raises_on_multiclass_instead_of_no_opping() -> None:
    """The K>=3 prune selects on ONE train/select split and has no fold refits to pin a bank
    into, so a non-default value there must raise rather than be silently ignored."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(900, 4)).astype(np.float32)
    y = (x[:, 0] + 0.5 * rng.normal(size=900) > 0).astype(np.int64) + (
        x[:, 1] + 0.5 * rng.normal(size=900) > 0
    ).astype(np.int64)
    with pytest.raises(ValueError, match="prune_fold_fidelity"):
        TBoostClassifier(
            n_trees=40, n_bags=1, seed=0, prune=True, graduate=False, prune_fold_fidelity=True
        ).fit(x, y)


def test_fold_fidelity_is_deterministic_under_thread_count() -> None:
    """The augmentation is a backfit solve inside the fold task; like every other reduction on
    this path it must not depend on how many workers ran it."""
    a = _fidelity_fit(prune_fold_fidelity=True, n_jobs=1)
    b = _fidelity_fit(prune_fold_fidelity=True, n_jobs=8)
    assert a._model.to_bytes() == b._model.to_bytes()


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
