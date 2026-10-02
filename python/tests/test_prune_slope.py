"""Post-prune OOB slope re-anchor (`pruning_report_["slope"]`).

`apply_keepset` re-solves the deployed bank's intercept but never its scale, so a heavy prune
ships a score whose log-scale spread is compressed. These tests pin the three things that make
the correction safe to ship: the solve is the objective's own GLM, the fold is an EXACT affine
on the score that leaves the table structure alone, and the gate keeps every fit that does not
need it byte-identical.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from t_boost._t_boost import _TableModel
from _artifact import model_bytes
from t_boost.sklearn import (
    _fit_link_slope,
    _fold_slope_into_bank_json,
    TBoostClassifier,
    TBoostRegressor,
)


def _guard_fixture(n: int = 8000, seed: int = 0):
    """The prune guard's own OOB fixture (mirrors `test_prune.py`), so the slope is exercised
    on exactly the fits the guard runs on."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.uniform(0, 1, n), rng.uniform(0, 1, n)]).astype(np.float32)
    cat = rng.choice([f"L{i}" for i in range(8)], size=n)
    mu = np.exp(0.3 * np.sin(6 * x[:, 0]) + 0.2 * (x[:, 1] > 0.5) + 0.15 * (cat == "L3"))
    y = rng.poisson(mu).astype(np.float32)
    X = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "c": cat})
    return X, y


# --- the solve ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "obj,b_true", [("gamma", 1.25), ("gamma", 0.8), ("poisson", 1.15), ("tweedie", 1.2)]
)
def test_slope_solver_recovers_a_known_compression(obj: str, b_true: float) -> None:
    # Construct the exact situation the prune creates: a TRUE log-scale score, and a shipped
    # score that is a COMPRESSED version of it (`raw = true/b_true`). The deviance-minimising
    # rescale of the shipped score is then `b_true` by construction, and the solver must find it.
    rng = np.random.default_rng(0)
    n = 40000
    eta_true = 1.5 + 0.9 * rng.normal(size=n)
    mu = np.exp(eta_true)
    if obj == "gamma":
        y = rng.gamma(shape=4.0, scale=mu / 4.0)
    elif obj == "poisson":
        y = rng.poisson(np.exp(eta_true - 1.0)).astype(float) * np.exp(1.0)
        y = rng.poisson(mu).astype(float)
    else:
        y = rng.poisson(mu * 0.5).astype(float) * rng.gamma(2.0, 0.5, size=n)
    w = np.ones(n)
    raw = eta_true / b_true  # the compressed (shipped) score
    a, b, se = _fit_link_slope(obj, raw, y, w, 1.5)
    assert b == pytest.approx(b_true, rel=0.05), f"{obj}: recovered b={b}"
    assert np.isfinite(se) and se > 0.0
    # A real compression on 40k rows must be many standard errors from 1.
    assert abs(b - 1.0) / se > 10.0


def test_solver_refuses_a_non_log_link_instead_of_silently_returning_the_identity() -> None:
    # A LANDMINE, disarmed. The solve is log-link all the way down (`mu = exp(eta)`), so on a
    # logit it does not degrade gracefully — it converges to `b = 1` and reports "nothing to
    # correct" for a score that is genuinely compressed. Verified below: a logit compressed by a
    # true factor of 1.3 must NOT come back as a silent no-op. Anyone widening
    # `_SLOPE_OBJECTIVES` to a new link has to bring that link's own IRLS with them.
    from t_boost.sklearn import _slope_variance_power

    rng = np.random.default_rng(0)
    n = 20000
    eta = 0.9 * rng.normal(size=n)
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    with pytest.raises(ValueError, match="log-link only"):
        _fit_link_slope("logistic", eta / 1.3, y, np.ones(n), None)
    with pytest.raises(ValueError, match="log-link only"):
        _slope_variance_power("squared_error", None)
    # the shipped domain is exactly the three log-link objectives
    for obj, p in (("poisson", 1.0), ("gamma", 2.0), ("tweedie", 1.9)):
        assert _slope_variance_power(obj, 1.9) == p


def test_slope_solver_returns_the_exact_identity_on_a_constant_score() -> None:
    # No slope is identifiable from a constant score: the solver must hand back the exact
    # identity (and infinite standard error), never a fitted value.
    rng = np.random.default_rng(1)
    y = rng.gamma(3.0, 2.0, size=2000)
    raw = np.full(2000, 1.7)
    assert _fit_link_slope("gamma", raw, y, np.ones(2000), None) == (0.0, 1.0, float("inf"))


def test_slope_solver_beats_ols_on_gamma_where_the_two_disagree() -> None:
    # The correction is the objective's GLM fit, not an OLS of log y on the score. On gamma the
    # two genuinely disagree, and the GLM one is the one that lowers the GAMMA deviance.
    from t_boost.sklearn import _guard_mean_deviance

    rng = np.random.default_rng(3)
    n = 60000
    eta = 1.2 + 1.0 * rng.normal(size=n)
    y = rng.gamma(shape=1.5, scale=np.exp(eta) / 1.5)  # heavy tail, small shape
    w = np.ones(n)
    raw = eta / 1.3
    a_glm, b_glm, _ = _fit_link_slope("gamma", raw, y, w, None)
    # OLS of log y on raw.
    ols = np.polyfit(raw, np.log(y), 1)
    d_glm = _guard_mean_deviance("gamma", y, a_glm + b_glm * raw, w, None)
    d_ols = _guard_mean_deviance("gamma", y, ols[1] + ols[0] * raw, w, None)
    assert d_glm < d_ols


# --- the fold ----------------------------------------------------------------------------


def _fit_pruned(**kw) -> TBoostRegressor:
    X, y = _guard_fixture()
    base = dict(objective="poisson", n_trees=150, n_bags=2, graduate=False, seed=0, prune=True,
                categorical_features=["c"])
    base.update(kw)
    return TBoostRegressor(**base).fit(X, y)


def test_fold_is_an_exact_affine_on_the_score_and_leaves_the_structure_alone() -> None:
    est = _fit_pruned()
    X, _ = _guard_fixture()
    tm = est._model
    doc = json.loads(tm.to_json())
    before = json.loads(tm.to_json())
    a, b = -0.37, 1.21
    _fold_slope_into_bank_json(doc, a, b)
    scaled = _TableModel.from_json(json.dumps(doc))

    cat_x = [list(X["c"].to_numpy(dtype=object))]
    xs = np.ascontiguousarray(X[["a", "b"]].to_numpy(np.float32))
    old = np.asarray(tm.predict_raw(xs, cat_x=cat_x), np.float64)
    new = np.asarray(scaled.predict_raw(xs, cat_x=cat_x), np.float64)
    assert np.allclose(new, a + b * old, atol=1e-4 * max(1.0, np.abs(a + b * old).max()))

    # TABLE STRUCTURE AND COUNT UNTOUCHED — the whole point of representing the correction as a
    # uniform multiplier plus an intercept shift rather than as extra tables.
    assert scaled.deployed_supports() == tm.deployed_supports()
    aft = json.loads(scaled.to_json())
    assert len(aft["model"]["bank"]["tables"]) == len(before["model"]["bank"]["tables"])
    assert aft["model"]["mode"] == before["model"]["mode"]
    assert aft["model"]["bank"]["merged_grids"] == before["model"]["bank"]["merged_grids"]
    assert aft["model"]["bank"]["w"] == before["model"]["bank"]["w"]

    # f0 and every cell moved by exactly the affine / the multiplier.
    assert aft["model"]["bank"]["f0"] == pytest.approx(a + b * before["model"]["bank"]["f0"])
    for t_old, t_new in zip(before["model"]["bank"]["tables"], aft["model"]["bank"]["tables"]):
        assert t_new["u"] == t_old["u"] and t_new["axes"] == t_old["axes"]
        vo = np.asarray(t_old["values"]["data"]["Dense"], float)
        vn = np.asarray(t_new["values"]["data"]["Dense"], float)
        assert np.array_equal(vn, b * vo)
        # `support` is training mass, NOT a link-scale value: it must not move.
        assert t_new["support"] == t_old["support"]
        # `variance` is a second moment, so it rides on b^2 — the I2 variance-sum gate and the
        # absolute Sobol figure both read it.
        assert t_new["variance"] == pytest.approx(t_old["variance"] * b * b, rel=1e-12)


def test_fold_preserves_fanova_purity_and_mass_by_linearity() -> None:
    # The five I2 gates split cleanly under an affine fold, and this pins the split.
    #
    # PURITY, MASS CONSERVATION and VARIANCE SUM are statements INTERNAL to the bank, and every
    # one of them is preserved: purity says each axis-slice has w-weighted mean zero, and
    # `Sum w*(b*v) = b * Sum w*v = 0` for ANY reference measure and any b; mass conservation is
    # re-centred on `a + b*f0` by construction; the cached variances are rewritten to `b^2*v`.
    # Because every cell is EXACTLY `b` times the old cell (asserted here), every linear or
    # quadratic invariant of the values follows by linearity — no measure needed.
    #
    # RECONSTRUCTION and THREE-WAY-EQUAL are the two gates that tie the bank to the tree
    # ensemble. They hold in affine-mapped form -- the bank reconstructs `a + b*F_ens`, which is
    # the deliberate change -- and that is what the predict_raw check above verifies.
    est = _fit_pruned()
    tm = est._model
    doc = json.loads(tm.to_json())
    before = json.loads(tm.to_json())
    b = 0.83
    _fold_slope_into_bank_json(doc, 0.4, b)
    for t_old, t_new in zip(before["model"]["bank"]["tables"], doc["model"]["bank"]["tables"]):
        vo = np.asarray(t_old["values"]["data"]["Dense"], float)
        vn = np.asarray(t_new["values"]["data"]["Dense"], float)
        nz = vo != 0.0
        # exact uniform multiplier, cell for cell
        assert np.allclose(vn[nz] / vo[nz], b, rtol=0, atol=0) or np.array_equal(vn, b * vo)
    # the artifact still loads through the §10 validate gate
    _TableModel.from_json(json.dumps(doc))


def test_folded_model_rating_export_is_consistent() -> None:
    # `tables()` is the shipped deliverable — the rating export an insurer actually deploys — so
    # the fold has to leave it coherent, not merely leave the bank loadable.
    est = _fit_pruned()
    tm = est._model
    a, b = -0.31, 1.17
    doc = json.loads(tm.to_json())
    _fold_slope_into_bank_json(doc, a, b)
    scaled = _TableModel.from_json(json.dumps(doc))
    ex0, ex1 = json.loads(tm.tables()), json.loads(scaled.tables())

    assert len(ex1["tables"]) == len(ex0["tables"])
    assert ex1["mode"] == ex0["mode"] and ex1["link"] == ex0["link"]
    assert ex1["f0"] == pytest.approx(a + b * ex0["f0"])
    for t0, t1 in zip(ex0["tables"], ex1["tables"]):
        assert np.array_equal(np.asarray(t1["values"]), b * np.asarray(t0["values"]))
        assert t1["variance"] == pytest.approx(t0["variance"] * b * b, rel=1e-12)
        if t0.get("relativities"):
            # relativities are exp(value), so they move MULTIPLICATIVELY — the one export
            # field that is not linear in b, and it is right that it is not.
            assert np.allclose(
                np.asarray(t1["relativities"]),
                np.exp(b * np.log(np.asarray(t0["relativities"]))),
            )
    # Sobol shares are variance RATIOS, and every variance moved by the same b^2, so the
    # importance ordering an actuary reads off the export is untouched.
    assert np.allclose([t["sobol"] for t in ex1["tables"]],
                       [t["sobol"] for t in ex0["tables"]])


def test_folded_model_round_trips_through_bytes_and_json() -> None:
    est = _fit_pruned()
    doc = json.loads(est._model.to_json())
    _fold_slope_into_bank_json(doc, -0.2, 1.3)
    scaled = _TableModel.from_json(json.dumps(doc))
    again = _TableModel.from_bytes(scaled.to_bytes())
    assert json.loads(again.to_json()) == json.loads(scaled.to_json())


# --- the gate ----------------------------------------------------------------------------


def test_slope_is_always_reported_on_a_log_link_oob_pruned_fit() -> None:
    est = _fit_pruned()
    s = est.pruning_report_["slope"]
    # reported whether or not it fires — the self-gating safety story is that you can always see
    # what it decided and why
    assert s["evidence"] == "oob" and s["rows"] > 0
    for k in ("a", "b", "se_b", "z_b", "sd_ratio", "rel_gain", "dev_before", "dev_after"):
        assert k in s and np.isfinite(s[k]) or k == "sd_ratio"
    assert "applied" in s


def test_healthy_oob_fit_is_byte_identical_to_the_guard_off_build() -> None:
    # THE SAFETY CONTRACT. On a fit where the prune did not compress the score, the correction
    # must not merely be small — it must not touch the artifact at all. Here `b` lands ~1.03 off
    # a 2-bag jury, comfortably past the |b-1| epsilon, and it is the z gate (z ~ 0.3) that
    # keeps the bytes identical.
    X, y = _guard_fixture()
    kw = dict(objective="poisson", n_trees=150, n_bags=2, graduate=False, seed=0, prune=True,
              categorical_features=["c"])
    on = TBoostRegressor(**kw).fit(X, y)
    off = TBoostRegressor(**kw, prune_guard=False).fit(X, y)
    s = on.pruning_report_["slope"]
    assert s["applied"] is False and s["z_b"] < s["min_z"]
    assert model_bytes(on) == model_bytes(off)


def test_slope_gate_needs_both_a_real_slope_and_real_evidence() -> None:
    # The two gates are independent and both are load-bearing: `eps` rejects a slope that is
    # too small to matter even if it is precisely measured, `min_z` rejects a slope that is
    # large but indistinguishable from out-of-bag noise. Every battery harm came from the
    # second kind, which is why `|b-1|` alone is not a gate.
    from t_boost.sklearn import _SLOPE_EPS, _SLOPE_MIN_Z

    est = _fit_pruned()
    s = est.pruning_report_["slope"]
    fired = abs(s["b"] - 1.0) > _SLOPE_EPS and s["z_b"] >= _SLOPE_MIN_Z
    assert s["applied"] is fired
    assert s["min_z"] == _SLOPE_MIN_Z and s["eps"] == _SLOPE_EPS
    # the reported reason always names the gate that stopped it
    if not s["applied"]:
        assert ("|b-1|" in s["skipped"]) or ("z " in s["skipped"])


def test_when_the_gate_opens_the_shipped_model_is_the_explicitly_rescaled_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # END TO END through the APPLY path. Relaxing both gates to zero forces the correction to
    # fire on a fixture small enough to be a unit test; what is then checked is the thing that
    # matters — the model the estimator SHIPS is exactly the model you get by taking the
    # un-corrected bank and folding the reported (a, b) into it by hand.
    import t_boost.sklearn as tbs

    X, y = _guard_fixture()
    # The two gates became constructor parameters on 2026-09-04 (`prune_slope_eps` /
    # `prune_slope_min_z`); relaxing them to zero is now a supported request rather than a
    # monkeypatch of module constants the estimator no longer reads.
    kw = dict(objective="poisson", n_trees=150, n_bags=2, graduate=False, seed=0, prune=True,
              categorical_features=["c"], prune_slope_eps=0.0, prune_slope_min_z=0.0)
    fired = TBoostRegressor(**kw).fit(X, y)
    s = fired.pruning_report_["slope"]
    assert s["applied"] is True and abs(s["b"] - 1.0) > 1e-6

    # The un-corrected build: the same fit with the objective gate emptied, which is the code
    # path as it stood before this feature.
    monkeypatch.setattr(tbs, "_SLOPE_OBJECTIVES", frozenset())
    plain = TBoostRegressor(**kw).fit(X, y)
    assert "slope" not in plain.pruning_report_

    hand = json.loads(plain._model.to_json())
    _fold_slope_into_bank_json(hand, s["a"], s["b"])
    # Cell for cell the same artifact — not merely the same predictions.
    assert json.loads(fired._model.to_json()) == hand
    assert _TableModel.from_json(json.dumps(hand)).to_bytes() == fired._model.to_bytes()

    # ...and the correction really is a rescale of the score, not a re-fit: same tables, and the
    # predictions differ by exactly the reported affine.
    assert fired.pruning_report_["deployed"] == plain.pruning_report_["deployed"]
    p_plain = np.log(np.asarray(plain.predict(X), np.float64))
    p_fired = np.log(np.asarray(fired.predict(X), np.float64))
    assert np.allclose(p_fired, s["a"] + s["b"] * p_plain, atol=1e-4)


def test_slope_untouched_for_a_classifier() -> None:
    # Non-log-link: logistic is OUT of the shipped domain. An affine rescale of a logit is
    # Platt scaling — principled, but a different claim about a different loss, and it ships
    # only where the battery measured it.
    rng = np.random.default_rng(0)
    n = 8000
    x = rng.normal(size=(n, 4)).astype(np.float32)
    p = 1.0 / (1.0 + np.exp(-(0.7 * x[:, 0] + 0.4 * x[:, 1])))
    y = (rng.uniform(size=n) < p).astype(np.int32)
    est = TBoostClassifier(objective="logistic", n_trees=150, n_bags=2, graduate=False, seed=0,
                             prune=True).fit(x, y)
    assert "slope" not in est.pruning_report_
    off = TBoostClassifier(objective="logistic", n_trees=150, n_bags=2, graduate=False, seed=0, prune=True,
                             prune_guard=False).fit(x, y)
    assert model_bytes(est) == model_bytes(off)


def test_slope_untouched_on_the_grouped_carve_path() -> None:
    # A real panel fit cannot supply honest out-of-bag rows (a group straddles the bag
    # boundary), so the guard falls back to the shared carve — and the slope, which is defined
    # on the OOB evidence, must not run at all there.
    import pandas as pd

    rng = np.random.default_rng(0)
    n_g, per = 900, 12
    n = n_g * per
    groups = np.repeat(np.arange(n_g), per)
    ge = rng.normal(0.0, 0.5, size=n_g)[groups]
    x = np.column_stack([rng.uniform(0, 1, n), rng.uniform(0, 1, n)]).astype(np.float32)
    mu = np.exp(0.3 * np.sin(6 * x[:, 0]) + 0.25 * (x[:, 1] > 0.5) + ge)
    y = rng.poisson(mu).astype(np.float32)
    X = pd.DataFrame({"a": x[:, 0], "b": x[:, 1]})
    est = TBoostRegressor(objective="poisson", n_trees=150, n_bags=2, graduate=False, seed=0,
                            prune=True, validation_fraction=0.1).fit(X, y, groups=groups)
    if est.pruning_report_["guard"].get("evidence") == "carve":
        assert "slope" not in est.pruning_report_


def test_slope_is_deterministic() -> None:
    a = _fit_pruned()
    b = _fit_pruned()
    assert a.pruning_report_["slope"] == b.pruning_report_["slope"]
    assert model_bytes(a) == model_bytes(b)


@pytest.mark.parametrize("jobs", [1, 4])
def test_slope_is_byte_identical_across_thread_counts(jobs: int) -> None:
    # The house determinism contract: a fit is byte-identical whatever the thread count. The
    # slope must not become the one stage that breaks it — its evidence comes from a parallel
    # out-of-bag pass and its verification scores on a bounded pool, so the report AND the
    # artifact are both pinned here.
    one = _fit_pruned(n_jobs=1)
    many = _fit_pruned(n_jobs=jobs)
    assert one.pruning_report_["slope"] == many.pruning_report_["slope"]
    assert model_bytes(one) == model_bytes(many)


def test_slope_survives_a_heavier_prune_with_more_bags() -> None:
    # A wider fit with a real bank and an 8-bag jury — closer to the arena's shape than the
    # 3-feature fixture — exercised end to end: whatever the gate decides, the artifact must be
    # loadable, finite, and structurally intact.
    rng = np.random.default_rng(5)
    n = 20000
    x = rng.normal(size=(n, 8)).astype(np.float32)
    mu = np.exp(0.4 * x[:, 0] + 0.3 * x[:, 1] + 0.2 * x[:, 0] * x[:, 2] + 0.15 * x[:, 3])
    y = rng.gamma(shape=2.0, scale=mu / 2.0).astype(np.float32)
    est = TBoostRegressor(objective="gamma", n_trees=250, n_bags=8, seed=0,
                            prune=True).fit(x, y)
    s = est.pruning_report_["slope"]
    assert s["evidence"] == "oob" and np.isfinite(s["b"]) and s["b"] > 0.0
    p = np.asarray(est.predict(x), np.float64)
    assert np.isfinite(p).all() and (p > 0).all()
    assert len(est.pruning_report_["deployed"]) > 0
    if s["applied"]:
        # when it fires, the fold must have verified as an exact affine
        assert s["verify_max_abs_err"] < 1e-3


def test_slope_is_fitted_on_the_final_rung_not_a_pre_guard_one() -> None:
    # THE REGRESSION THIS FILE EXISTS FOR. The correction must be fitted on the OOB scores of
    # the keep-set the guard SETTLED ON. Fitting it on an earlier rung and applying it to the
    # bank the ladder subsequently grew is catastrophic, not merely suboptimal: on allstate_sev
    # split 3 the pre-guard rung's b = 1.23 (a 1,934-table bank) applied to the 10,307-table
    # bank actually deployed took the split from +0.0926 to +0.0091.
    #
    # `tol < 0` forces the ladder to climb, so `steps > 0` and the final rung is NOT rung 0.
    X, y = _guard_fixture()
    # `prune_guard_z=0.0` keeps the FIXED-tolerance breach test: a negative tolerance is the
    # forcing device this test needs, and the av37 SE-aware bar would override it (that bar is
    # exercised in test_prune.py, not here).
    est = TBoostRegressor(objective="poisson", n_trees=150, n_bags=2, graduate=False, seed=0, prune=True,
                            categorical_features=["c"], prune_guard_tol=-0.5,
                            prune_guard_z=0.0, prune_guard_z_dn=0.0).fit(X, y)
    g = est.pruning_report_["guard"]
    assert g["fired"] is True and g["steps"] >= 1
    assert est.pruning_report_["slope"]["rung"] == g["steps"]
    # The last rung is the full support set, which is un-pruned and therefore un-compressed:
    # the correction must find nothing to do there.
    assert est.pruning_report_["slope"]["sd_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert est.pruning_report_["slope"]["b"] == pytest.approx(1.0, abs=1e-3)


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
