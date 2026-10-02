"""The K>=3 set-level prune no-harm guard (`multiclass_prune_guard=True`).

The single-output path has carried a set-level guard since 2026-08-04: the per-table
leave-one-out gains the keep-set is chosen from are structurally blind to mass carried JOINTLY
by correlated tables, so a whole correlated family can be dropped while the fold evidence
claims the cost is ~0. The multiclass path had no such guard at all — these tests cover the
softmax analogue (`prune::multiclass_prune_guard`), which lives in Rust because the K>=3
fit-select-deploy flow is orchestrated entirely inside `fit_multiclass_pruned`.

The load-bearing properties, in order of how much they would cost to get wrong:

1. OFF BY DEFAULT and inert when off — a K>=3 fit that does not ask for the guard is
   byte-identical to the pre-guard build (the standing identity gate).
2. SILENT ⇒ IDENTICAL — when the breach test does not trip, the shipped artifact is exactly
   the unguarded one, not merely close to it.
3. THE LADDER IS EXACT — driving the guard to re-admit every candidate reproduces the full
   arm's deviance to the last bit, which is the arithmetic proof that the prefix-union
   accumulation over per-bag/per-class banks reconstructs a real bank score.
4. BOTH EVIDENCE PATHS RUN — out-of-bag juries for an ungrouped fit, the shared group-honest
   carve for a panel fit (an out-of-bag row's panel-mates are almost surely in bag).
5. K<=2 AND REGRESSION NEVER SEE IT.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from t_boost.sklearn import TBoostClassifier, TBoostRegressor
from _artifact import model_bytes


def _fixture(n: int = 8000, p: int = 12, seed: int = 3, k_cuts=(0.3, 0.6, 0.85)):
    """A K=4 set wide enough that the selector actually drops tables (so the guard has a
    ladder to climb) and noisy enough that dropping them is a real decision."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p)).astype(np.float32)
    lin = 1.0 * x[:, 0] - 0.8 * x[:, 1] + 0.6 * x[:, 2] + 0.5 * x[:, 0] * x[:, 3]
    lin = lin + rng.normal(scale=1.2, size=n)
    cuts = np.quantile(lin, k_cuts)
    y = np.digitize(lin, cuts)
    return pl.DataFrame({f"n{i}": x[:, i] for i in range(p)}), y


def _fit(frame, y, *, groups=None, **kw):
    clf = TBoostClassifier(n_trees=200, seed=1, n_jobs=4, prune=True, **kw)
    return clf.fit(frame, y, **({"groups": groups} if groups is not None else {}))


def test_guard_is_on_by_default_with_an_se_aware_bar() -> None:
    # Since 2026-09-07 a default K>=3 fit runs the guard: the multiclass prune selects on fold
    # refits like the single-output path, and the guard measures the shipped keep-set against
    # the full deploy bank with an SE-aware bar (`prune_guard_z_dn` SEs of the paired gap,
    # clamped by the raw `prune_guard_tol`), because the raw relative tolerance alone can never
    # fire on a multiclass log-loss dominated by class entropy.
    frame, y = _fixture(n=3000, p=6)
    model = _fit(frame, y)
    assert model.multiclass_prune_guard is True
    guard = model.pruning_report_["guard"]
    assert guard["enabled"] and guard["skipped"] is None, guard
    assert guard["evidence"] == "oob"
    assert np.isfinite(guard["gap_se"]) and guard["gap_se"] >= 0.0
    assert np.isfinite(guard["dev_null"]) and guard["dev_null"] > guard["dev_full"]
    assert 0.0 <= guard["bar"] <= guard["dev_full"] * guard["tol"] + 1e-12
    assert model.pruning_report_["selector"] == "cv_fold_vote"
    assert model.pruning_report_["cv_folds"] >= 2


def test_guard_off_is_byte_identical_to_guard_on_when_silent() -> None:
    # Property 2. The guard is post-fit SELECTION only: with no breach, the same keep-set is
    # applied to the same deploy fit, so the artifact must be identical byte for byte — not
    # "close", not "within tolerance".
    frame, y = _fixture()
    off = _fit(frame, y, multiclass_prune_guard=False)
    on = _fit(frame, y, multiclass_prune_guard=True)
    guard = on.pruning_report_["guard"]
    assert guard["enabled"] and not guard["fired"], guard
    assert model_bytes(off) == model_bytes(on)
    assert np.array_equal(off.predict_proba(frame), on.predict_proba(frame))


def test_guard_reports_the_out_of_bag_jury_it_measured_on() -> None:
    frame, y = _fixture()
    guard = _fit(frame, y, multiclass_prune_guard=True).pruning_report_["guard"]
    assert guard["evidence"] == "oob"
    assert guard["skipped"] is None
    assert guard["n_bags"] == 8
    # At the shipped n_bags=8 / bag_subsample=0.8 recipe most rows get a jury of ~2 bags. The
    # exact numbers are the single-output guard's measured regime; these bounds only assert we
    # are in it, not a golden value.
    assert guard["oob_rows"] > 0.7 * len(y)
    assert 1.0 <= guard["oob_mean_jury"] <= float(guard["n_bags"])
    assert guard["oob_rows"] + guard["oob_rows_uncovered"] == len(y)
    # Every class was re-anchored (one shift per class), on BOTH arms.
    assert len(guard["level_shift_selected"]) == len(np.unique(y))
    assert len(guard["level_shift_full"]) == len(np.unique(y))


def test_full_ladder_reproduces_the_full_arm_exactly() -> None:
    # Property 3, the arithmetic gate. A negative tolerance can never be satisfied, so the
    # ladder runs to exhaustion and the shipped keep-set is the whole candidate universe. Its
    # deviance must then EQUAL the full arm's: the rungs are prefix unions of disjoint table
    # groups accumulated on one intercept, so the last rung is the full arm by construction.
    # Any drift here means the accumulation is not reconstructing a real bank score.
    frame, y = _fixture()
    guard = _fit(
        frame, y, multiclass_prune_guard=True, prune_guard_tol=-1.0
    ).pruning_report_["guard"]
    assert guard["fired"]
    assert guard["steps"] == guard["rungs_available"]
    assert guard["kept_final"] > guard["kept_initial"]
    assert guard["dev_selected_final"] == pytest.approx(guard["dev_full"], rel=1e-12)


def test_a_fired_guard_grows_the_keepset_and_the_report_describes_the_artifact() -> None:
    # The report must describe what SHIPPED, not the pre-guard selection: `kept` is rewritten
    # to the grown keep-set, and the deployed bank must actually carry the re-admitted tables.
    frame, y = _fixture()
    unguarded = _fit(frame, y, multiclass_prune_guard=False)
    fired = _fit(frame, y, multiclass_prune_guard=True, prune_guard_tol=-1.0)
    guard = fired.pruning_report_["guard"]
    assert guard["fired"]
    assert len(fired.pruning_report_["kept"]) == guard["kept_final"]
    assert len(unguarded.pruning_report_["kept"]) == guard["kept_initial"]
    kept_before = {tuple(u) for u in unguarded.pruning_report_["kept"]}
    kept_after = {tuple(u) for u in fired.pruning_report_["kept"]}
    # Only ever GROWS: binary table dropping is preserved, nothing kept is taken away.
    assert kept_before < kept_after
    assert not np.array_equal(
        unguarded.predict_proba(frame), fired.predict_proba(frame)
    )


def test_grouped_fit_takes_the_out_of_bag_path_with_group_aware_bags() -> None:
    # Property 4 (revised 2026-09-07). Bags are now drawn at GROUP granularity when `groups=`
    # is given, so no group straddles a bag boundary and an out-of-bag row's group-mates are
    # out of bag with it: the deploy soup's own out-of-bag rows are honest evidence on a panel
    # and the guard no longer has to fall back to the 10% shared carve.
    frame, y = _fixture()
    groups = np.repeat(np.arange(len(y) // 4), 4)[: len(y)]
    guard = _fit(
        frame, y, groups=groups, multiclass_prune_guard=True
    ).pruning_report_["guard"]
    assert guard["evidence"] == "oob", guard
    assert guard["skipped"] is None
    assert guard["oob_mean_jury"] >= 1.0
    # Out-of-bag coverage, far more rows than the ~validation_fraction carve.
    assert guard["oob_rows"] > 0.5 * len(y)


def test_grouped_out_of_bag_ladder_is_also_exact() -> None:
    frame, y = _fixture()
    groups = np.repeat(np.arange(len(y) // 4), 4)[: len(y)]
    guard = _fit(
        frame, y, groups=groups, multiclass_prune_guard=True, prune_guard_tol=-1.0
    ).pruning_report_["guard"]
    assert guard["evidence"] == "oob"
    assert guard["fired"]
    assert guard["dev_selected_final"] == pytest.approx(guard["dev_full"], rel=1e-12)


def test_singleton_grouping_still_takes_the_out_of_bag_path() -> None:
    # A degenerate (all-singleton) grouping cannot straddle the in-bag/out-of-bag boundary, so
    # out-of-bag evidence is exactly as honest as on an ungrouped fit AND covers ~8x more rows
    # than the carve would. Same predicate as the single-output guard (`_is_degenerate_grouping`).
    frame, y = _fixture()
    guard = _fit(
        frame,
        y,
        groups=np.arange(len(y)),
        multiclass_prune_guard=True,
    ).pruning_report_["guard"]
    assert guard["evidence"] == "oob", guard


def test_single_bag_fit_reports_no_evidence_instead_of_silently_skipping() -> None:
    # n_bags=1 has no out-of-bag rows at all. That must be REPORTED, not silently read as "the
    # guard ran and was happy" — the distinction is the whole point of the `skipped` field.
    frame, y = _fixture(n=4000, p=8)
    guard = _fit(
        frame, y, n_bags=1, multiclass_prune_guard=True
    ).pruning_report_["guard"]
    assert guard["enabled"] and not guard["fired"]
    assert guard["evidence"] is None
    assert "single-bag" in guard["skipped"]


def test_guard_is_deterministic() -> None:
    frame, y = _fixture()
    a = _fit(frame, y, multiclass_prune_guard=True, prune_guard_tol=-1.0)
    b = _fit(frame, y, multiclass_prune_guard=True, prune_guard_tol=-1.0)
    assert model_bytes(a) == model_bytes(b)
    # `evidence_seconds` is a wall clock, deliberately excluded.
    drop = {"evidence_seconds"}
    assert {k: v for k, v in a.pruning_report_["guard"].items() if k not in drop} == {
        k: v for k, v in b.pruning_report_["guard"].items() if k not in drop
    }


def test_spread_diagnostics_are_reported_for_both_arms() -> None:
    # The compression statistic the slope/temperature question turns on: the weighted mean
    # per-row standard deviation of the K class logits, selected arm vs full arm. A pruned bank
    # that is SHRUNK relative to the full one shows spread_selected < spread_full.
    frame, y = _fixture()
    guard = _fit(frame, y, multiclass_prune_guard=True).pruning_report_["guard"]
    assert guard["spread_selected"] > 0.0
    assert guard["spread_full"] > 0.0


@pytest.mark.parametrize("k", [2])
def test_binary_classifier_never_sees_the_multiclass_guard(k: int) -> None:
    # Property 5. `multiclass_prune_guard` is read only inside `_fit_multiclass`; a K=2 fit
    # takes `_fit_and_prune` and its own (separately-defaulted) `prune_guard`.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 5)).astype(np.float32)
    y = (x[:, 0] + rng.normal(scale=1.0, size=3000) > 0).astype(int)
    frame = pl.DataFrame({f"n{i}": x[:, i] for i in range(5)})
    off = TBoostClassifier(n_trees=60, seed=0, n_jobs=4, prune=True).fit(frame, y)
    on = TBoostClassifier(
        n_trees=60, seed=0, n_jobs=4, prune=True, multiclass_prune_guard=True
    ).fit(frame, y)
    assert model_bytes(off) == model_bytes(on)


def test_regressor_never_sees_the_multiclass_guard() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 5)).astype(np.float32)
    y = (2.0 * x[:, 0] - x[:, 1] + rng.normal(scale=0.5, size=3000)).astype(np.float32)
    frame = pl.DataFrame({f"n{i}": x[:, i] for i in range(5)})
    off = TBoostRegressor(n_trees=60, seed=0, n_jobs=4, prune=True).fit(frame, y)
    on = TBoostRegressor(
        n_trees=60, seed=0, n_jobs=4, prune=True, multiclass_prune_guard=True
    ).fit(frame, y)
    assert model_bytes(off) == model_bytes(on)


# --- av40: the guard and the TABLE BUDGET, composed --------------------------------------------
#
# These two features were built on branches that never met: the guard (av39) may only GROW the
# shared keep-set, the >=3-way table budget (depth-cells) CAPS it. `fit_multiclass_pruned_owned`
# composes them GUARD FIRST, BUDGET SECOND — the order the single-output path already uses, and
# the only order under which the cap is a cap: a guard that grew after the budget could ship past
# it. The consequence is that the guard's verdict describes the PRE-BUDGET keep-set, which the
# report has to say out loud (`guard["budget_blind"]`) exactly as the single-output path does.
#
# Nothing below can perturb a default fit: both knobs are off at their defaults.


def test_a_budget_never_lets_the_guard_ship_past_the_cap() -> None:
    frame, y = _fixture(n=6000, p=10)
    # `prune_guard_tol=-1.0` forces the breach test to trip, so the ladder re-admits every
    # dropped candidate — the most aggressive growth the guard can produce. The cap must still
    # hold on the artifact that ships.
    cap = 4
    fit = _fit(frame, y, multiclass_prune_guard=True, prune_guard_tol=-1.0,
               prune_table_budget=cap, prune_table_min_arity=3)
    rep = fit.pruning_report_
    assert rep["guard"]["enabled"] is True
    assert rep["guard"]["fired"] is True
    tb = rep["table_budget"]
    assert tb["max_tables"] == cap
    assert tb["min_arity"] == 3
    assert tb["tables_after"] <= cap
    # The keep-set the guard handed over is what the budget priced, so `tables_before` is the
    # POST-guard count — that is the whole reason the two have to be ordered.
    assert tb["tables_before"] >= tb["tables_after"]


def test_an_engaged_budget_marks_the_guards_verdict_budget_blind() -> None:
    frame, y = _fixture(n=6000, p=10)
    loose = _fit(frame, y, multiclass_prune_guard=True, prune_guard_tol=-1.0,
                 prune_table_budget=0)
    tight = _fit(frame, y, multiclass_prune_guard=True, prune_guard_tol=-1.0,
                 prune_table_budget=4)
    # No budget asked for => no budget block at all, and no flag: the guard's verdict is about
    # the bank that shipped.
    assert "table_budget" not in loose.pruning_report_
    assert loose.pruning_report_["guard"].get("budget_blind") is None
    # A budget that actually BOUND => the flag, so a reader can never mistake a verdict about
    # the pre-budget bank for one about the artifact.
    assert tight.pruning_report_["table_budget"]["engaged"] is True
    assert tight.pruning_report_["guard"]["budget_blind"] is True


def test_an_idle_budget_is_bit_identical_and_raises_no_flag() -> None:
    # A budget above what the bank carries leaves the keep-set untouched, so the guarded fit is
    # the same artifact as the unbudgeted one and the guard's verdict still describes it.
    frame, y = _fixture(n=6000, p=10)
    base = _fit(frame, y, multiclass_prune_guard=True, prune_guard_tol=-1.0)
    idle = _fit(frame, y, multiclass_prune_guard=True, prune_guard_tol=-1.0,
                prune_table_budget=10_000)
    assert idle.pruning_report_["table_budget"]["engaged"] is False
    assert idle.pruning_report_["guard"].get("budget_blind") is None
    assert idle.predict_proba(frame).tobytes() == base.predict_proba(frame).tobytes()


@pytest.fixture(autouse=True)
def _legacy_fold_vote_unbanded(monkeypatch: pytest.MonkeyPatch) -> None:
    """This module pins the fold-vote selector (its guard and evidence mechanics) on unbanded
    tables: the ranked path and banding became the defaults on 2026-09-26."""
    from t_boost import TBoostClassifier as _C
    from t_boost import TBoostRegressor as _R

    for cls in (_R, _C):
        defaults = cls.__init__.__kwdefaults__
        monkeypatch.setitem(defaults, "prune_selector", "fold_vote")
        monkeypatch.setitem(defaults, "band_tolerance", None)
