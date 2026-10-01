"""The 2026-09-04 surface work: keyword-only barrier, deprecation registry, merged spellings,
and `binding_report_` / `check_bindings`.

The governing constraint for every merge here is that it is a SPELLING, not a behaviour: the new
name resolves onto the legacy attributes and the fit is byte-identical either way. That is what
makes the change free against a benchmark cache whose cell keys do not fingerprint this library.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from t_boost import TBoostClassifier, TBoostRegressor


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 800
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "c": rng.integers(0, 12, n).astype(str)})
    y = rng.poisson(np.exp(0.3 * X.a - 0.4 * X.b)).astype(float)
    return X, y


def _mk(**kw):
    base = dict(objective="poisson", n_trees=40, categorical_features=["c"], seed=0, n_jobs=2)
    base.update(kw)
    return TBoostRegressor(**base)


# --------------------------------------------------------------- keyword-only barrier ----------

def test_only_the_three_plausible_positionals_remain_positional():
    import inspect
    for cls in (TBoostRegressor, TBoostClassifier):
        ps = inspect.signature(cls.__init__).parameters
        pos = [k for k, v in ps.items() if k != "self" and v.kind is v.POSITIONAL_OR_KEYWORD]
        assert pos == ["n_trees", "learning_rate", "lambda_"], cls.__name__


def test_keyword_only_params_survive_get_params_and_clone():
    # The reason the tail is behind `*` and NOT behind `**kwargs`: sklearn builds
    # `_get_param_names` by excluding VAR_POSITIONAL/VAR_KEYWORD only, so KEYWORD_ONLY round-trips
    # while a kwargs tail would vanish from `get_params` and be silently dropped by `clone`.
    m = _mk(max_bin=64, prune_min_stability=0.8)
    assert "max_bin" in m.get_params() and "prune_min_stability" in m.get_params()
    c = clone(m)
    assert c.max_bin == 64 and c.prune_min_stability == 0.8


def test_a_fourth_positional_argument_is_rejected():
    with pytest.raises(TypeError):
        TBoostRegressor(1000, 0.1, 2.0, True)          # 4th would have been a silent re-aim


# ------------------------------------------------------------------- merged spellings ----------

def test_prune_size_penalty_is_byte_identical_to_the_legacy_lambda(data):
    X, y = data
    a = _mk(prune=False, prune_lambda_tables=0.02).fit(X, y).predict(X)
    b = _mk(prune=False, prune_size_penalty=0.02).fit(X, y).predict(X)
    assert np.array_equal(a, b)


def test_early_stopping_int_and_float_resolve_onto_the_legacy_pair():
    assert _mk(early_stopping=99).early_stopping_rounds == 99      # int -> patience
    assert _mk(early_stopping=1.2).early_stopping_adaptive == 1.2  # float -> ratio


@pytest.mark.parametrize("scheme,legacy,tagged", [
    ("kfold", {"cat_leakage": "kfold", "cat_k": 3}, {"cat_leakage": "kfold:3"}),
    ("ordered", {"cat_leakage": "ordered", "cat_n_perms": 2}, {"cat_leakage": "ordered:2"}),
])
def test_tagged_cat_leakage_is_byte_identical_to_the_legacy_triple(data, scheme, legacy, tagged):
    X, y = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        a = _mk(prune=False, **legacy).fit(X, y).predict(X)
    b = _mk(prune=False, **tagged).fit(X, y).predict(X)
    assert np.array_equal(a, b), scheme


@pytest.mark.parametrize("kw,match", [
    ({"cat_leakage": "kfold:x"}, "must be an integer"),
    ({"cat_leakage": "loo:3"}, "takes none"),
    ({"cat_leakage": "kfold:3", "cat_k": 7}, "conflicts with"),
    ({"prune_size_penalty": 0.02, "prune_lambda_tables": 0.01}, "conflicts with"),
    ({"early_stopping": 99, "early_stopping_rounds": 200}, "conflicts with"),
])
def test_disagreeing_spellings_raise(data, kw, match):
    X, y = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match=match):
            _mk(prune=False, **kw).fit(X, y)


def test_agreeing_spellings_do_not_raise_because_clone_reproduces_them():
    # `clone` rebuilds from `get_params()`, which returns BOTH the new spelling and the legacy
    # attribute the resolution wrote. A presence test would raise on every clone; the conflict
    # test is DISAGREEMENT, which makes the resolution idempotent.
    m = _mk(prune_size_penalty=0.02, early_stopping=99, cat_leakage="kfold:3")
    c = clone(m)
    assert clone(c).get_params() == c.get_params()


# --------------------------------------------------------------------- deprecations ------------

def test_a_deprecated_spelling_warns_at_fit_not_at_construction(data):
    X, y = data
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        est = _mk(prune=False, cat_k=3)
        assert not [x for x in w if issubclass(x.category, DeprecationWarning)], "warned in __init__"
        est.fit(X, y)
        assert len([x for x in w if issubclass(x.category, DeprecationWarning)]) == 1


def test_a_clean_fit_is_silent(data):
    X, y = data
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _mk(prune=False).fit(X, y)
        assert not [x for x in w if issubclass(x.category, DeprecationWarning)]


def test_early_stopping_rounds_is_NOT_deprecated():
    # The ecosystem-standard spelling (LightGBM/XGBoost/CatBoost all use it) and the clamp's own
    # ceiling. Only the t-boost-specific `early_stopping_adaptive` is deprecated.
    assert "early_stopping_rounds" not in TBoostRegressor._DEPRECATED
    assert "early_stopping_adaptive" in TBoostRegressor._DEPRECATED


# ------------------------------------------------------------------ binding report -------------

def test_stability_is_reported_OVERRIDDEN_when_the_keep_budget_is_unbounded(data):
    # THE case this whole feature exists for. Tightening the stability gate cannot remove a table
    # while the av37 evidence path has budget to re-admit it, and nothing said so before.
    X, y = data
    m = _mk(n_trees=80, prune=True, prune_min_stability=1.0, prune_keep_budget=1_000_000).fit(X, y)
    row = next(r for r in m.binding_report_ if r["param"] == "prune_min_stability")
    assert row["status"] == "OVERRIDDEN"
    assert row["overridden_by"] == "prune_keep_budget"
    with pytest.raises(ValueError, match="did not bind"):
        m.check_bindings()


def test_prune_knobs_are_reported_INERT_when_pruning_is_off(data):
    X, y = data
    m = _mk(prune=False, prune_min_stability=1.0).fit(X, y)
    row = next(r for r in m.binding_report_ if r["param"] == "prune_min_stability")
    assert (row["status"], row["overridden_by"]) == ("INERT", "prune")


def test_a_cap_above_the_bank_is_reported_INERT(data):
    X, y = data
    m = _mk(n_trees=80, prune=True, prune_table_budget=100_000).fit(X, y)
    row = next(r for r in m.binding_report_ if r["param"] == "prune_table_budget")
    assert row["status"] == "INERT"


def test_a_default_fit_reports_nothing_and_passes_strict(data):
    X, y = data
    m = _mk(n_trees=80, prune=True).fit(X, y)
    assert m.binding_report_ == []
    m.check_bindings()


def test_deployed_table_count_comes_from_the_artifact_not_the_prune_report(data):
    # `pruning_report_["deployed"]` counts DENSE tables only; the order-3 bank is stored as
    # `factored` and is absent from it. The artifact is the only honest source for a size.
    X, y = data
    m = _mk(n_trees=80, prune=True).fit(X, y)
    artifact = m._deployed_table_count()
    assert artifact is not None
    assert artifact >= len(m.pruning_report_.get("deployed") or [])


def test_deployed_table_count_is_None_when_there_is_no_table_bank(data):
    X, y = data
    assert _mk(prune=False).fit(X, y)._deployed_table_count() is None


# ------------------------------------------------- proactive warning: knobs that do nothing ----
# `binding_report_` only helps someone who already suspects a problem. These cover the proactive
# half: a parameter the caller set that another parameter CANCELLED interrupts at fit.

def _fixture():
    rng = np.random.default_rng(0)
    n = 2500
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "c": rng.integers(0, 25, n).astype(str)})
    y = rng.poisson(np.exp(0.3 * X.a - 0.4 * X.b)).astype(float)
    return X, y


def _fit(**kw):
    X, y = _fixture()
    base = dict(objective="poisson", n_trees=100, categorical_features=["c"], seed=0,
                n_jobs=2, prune=True)
    base.update(kw)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = TBoostRegressor(**base).fit(X, y)
    return m, [x for x in w if issubclass(x.category, RuntimeWarning)]


def test_a_gate_its_own_evidence_path_undoes_warns_at_fit():
    # THE case. `prune_min_stability=1.0` cannot remove a table while the av37 evidence path has
    # budget to re-admit every one it refused, so the fit is what it would have been without the
    # gate at all — and nothing said so before this warning existed.
    m, warns = _fit(prune_min_stability=1.0, prune_keep_budget=1_000_000)
    assert len(warns) == 1
    assert "had NO EFFECT" in str(warns[0].message)
    assert "prune_keep_budget" in str(warns[0].message)


def test_the_warning_tracks_reality_not_just_the_setting():
    # The discriminator: turning the evidence path OFF lets the same gate genuinely bite, and the
    # warning must then stay silent. Asserted against the deployed bank, not against the flag.
    baseline, _ = _fit()
    voided, w_voided = _fit(prune_min_stability=1.0, prune_keep_budget=1_000_000)
    working, w_working = _fit(prune_min_stability=1.0, prune_drop_z=None)

    assert voided._deployed_table_count() == baseline._deployed_table_count()
    assert len(w_voided) == 1                       # did nothing -> warns

    assert working._deployed_table_count() != baseline._deployed_table_count()
    assert len(w_working) == 0                      # did something -> silent


def test_an_ordinary_fit_does_not_warn():
    _m, warns = _fit()
    assert warns == []


def test_a_merely_non_binding_cap_does_not_warn():
    # INERT is the weaker finding and is deliberately silent: a size cap set above the bank did
    # nothing, which is harmless. Measured on the insur-arena campaign, the depth gate arms
    # `prune_table_budget=100` on a lifted candidate whose bank holds 24 tables — warning there
    # would fire on ordinary campaign fits for no benefit. It stays in the report.
    m, warns = _fit(prune_table_budget=100_000)
    assert warns == []
    assert any(r["param"] == "prune_table_budget" and r["status"] == "INERT"
               for r in m.binding_report_)


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
