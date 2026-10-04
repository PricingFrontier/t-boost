"""Main-effect pruning (`prune_main_effects`, 2026-10-04).

Off by default: every main effect the fit realized is deployed and pruning only simplifies
interactions. On, main effects are candidates in every selector, under hierarchy: a main effect
leaves only when it does not earn its place and no kept interaction contains it."""
from __future__ import annotations

import json

import numpy as np
import pytest

from t_boost import TBoostClassifier, TBoostRegressor
from t_boost.sklearn import _heredity_sequence, _mains_before_their_interactions

Support = tuple[int, ...]


def _book(n: int = 5000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Poisson book: features 0-3 carry signal (0 and 2 interact), 4 and 5 are noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n, 6)).astype(np.float32)
    eta = (-1.0 + 0.6 * x[:, 0] - 0.4 * x[:, 1] + 0.8 * (x[:, 0] > 0.5) * (x[:, 2] > 0.4)
           + 0.3 * x[:, 3])
    return x, rng.poisson(np.exp(eta)).astype(np.float32)


def _mc_book(n: int = 6000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Three classes driven by features 0-2 (0 and 2 interact); 3 and 4 are noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n, 5)).astype(np.float32)
    logits = np.stack([np.zeros(n), 1.5 * x[:, 0] - 0.7,
                       1.2 * x[:, 1] + 0.9 * (x[:, 0] > 0.5) * (x[:, 2] > 0.5) - 1.0], 1)
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = (rng.uniform(size=n)[:, None] > np.cumsum(p, 1)).sum(1)
    return x, y


def _deviance(y: np.ndarray, mu: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(np.mean(2 * (t - y + mu)))


def _kept(est: TBoostRegressor | TBoostClassifier) -> set[Support]:
    return {tuple(int(i) for i in u) for u in est.pruning_report_["kept"]}


def _mains(kept: set[Support]) -> set[int]:
    return {u[0] for u in kept if len(u) == 1}


def _interactions(kept: set[Support]) -> set[Support]:
    return {u for u in kept if len(u) > 1}


def _hierarchical(kept: set[Support]) -> bool:
    return all((i,) in kept for u in kept if len(u) > 1 for i in u)


def _reg(**kw: object) -> TBoostRegressor:
    x, y = _book()
    return TBoostRegressor(objective="poisson", n_trees=200, seed=3, **kw).fit(x, y)


@pytest.fixture(scope="module")
def sticky() -> TBoostRegressor:
    return _reg()


@pytest.fixture(scope="module")
def prunable() -> TBoostRegressor:
    return _reg(prune_main_effects=True)


def test_main_effects_are_sticky_by_default(sticky: TBoostRegressor) -> None:
    assert TBoostRegressor().get_params()["prune_main_effects"] is False
    assert TBoostClassifier().get_params()["prune_main_effects"] is False
    rep = sticky.pruning_report_
    assert rep["selector"] == "ranked_path" and rep["main_effect_policy"] == "sticky"
    assert _mains(_kept(sticky)) == set(range(6))
    assert all(t["sticky"] for t in rep["table_scores"] if t["order"] == 1)


def test_ranked_path_drops_noise_main_effects_and_keeps_the_interactions(
    sticky: TBoostRegressor, prunable: TBoostRegressor
) -> None:
    rep = prunable.pruning_report_
    assert rep["selector"] == "ranked_path" and rep["main_effect_policy"] == "prunable"
    assert not any(t["sticky"] for t in rep["table_scores"])
    kept = _kept(prunable)
    assert _mains(kept) == {0, 1, 2, 3}
    assert _interactions(kept) == _interactions(_kept(sticky)) == {(0, 2)}
    assert _hierarchical(kept)
    # the path starts from the intercept-only model
    assert rep["path"]["sizes"][0] == 0
    assert rep["path"]["oob_deviance"][0] > rep["path"]["oob_deviance"][rep["path"]["best"]]
    importances = prunable.feature_importances_
    assert importances[4] == importances[5] == 0.0
    # the noise mains were noise: fresh data scores no worse without them (here, better)
    x, y = _book(20000, seed=9)
    assert _deviance(y, prunable.predict(x)) <= _deviance(y, sticky.predict(x))


def test_ranked_path_admits_a_main_with_its_first_interaction() -> None:
    # By variance, (1, 2) outranks both mains it needs and (0, 1, 2) waits for (0, 1) and
    # (0, 2). Mains (1) and (2) enter with (1, 2); (0) and (3) at their own ranks. The same
    # case as the native `ranked_path_sequence` test, so the two implementations agree.
    ranked = [(0,), (1, 2), (0, 1, 2), (1,), (0, 1), (3,), (0, 2), (2,)]
    seq = _heredity_sequence([], ranked)
    assert seq == [(0,), (1,), (2,), (1, 2), (0, 1), (3,), (0, 2), (0, 1, 2)]
    sticky = _heredity_sequence([(0,), (1,), (2,), (3,)], [u for u in ranked if len(u) > 1])
    assert [u for u in seq if len(u) > 1] == sticky


def test_guard_ladder_readmits_a_main_before_its_interactions() -> None:
    ranked = [(0, 3), (1,), (3,), (2, 3), (5,)]
    assert _mains_before_their_interactions(ranked) == [(3,), (0, 3), (1,), (2, 3), (5,)]
    interactions_only = [(0, 1), (1, 2)]
    assert _mains_before_their_interactions(interactions_only) == interactions_only


def test_fold_vote_judges_main_effects() -> None:
    # A single bag has no out-of-bag rows, so the fold vote selects. The legacy keep rule
    # (`prune_drop_z=None`) refuses the noise main outright.
    kw = dict(n_bags=1, max_interaction_order=1, prune_drop_z=None)
    off, on = _reg(**kw), _reg(prune_main_effects=True, **kw)
    assert on.pruning_report_["selector"] == "heldout_contribution_stability"
    assert on.pruning_report_["main_effect_policy"] == "prunable"
    assert 4 in _mains(_kept(off))
    assert _mains(_kept(on)) == _mains(_kept(off)) - {4}


def test_fold_vote_keeps_a_refused_main_that_a_kept_interaction_needs() -> None:
    est = _reg(n_bags=1, prune_drop_z=None, prune_main_effects=True)
    rep = est.pruning_report_
    restored = [tuple(u) for u in rep["closure_added"] if len(u) == 1]
    assert restored, "the fixture must have a refused main that a kept interaction contains"
    kept = _kept(est)
    assert _hierarchical(kept)
    assert all(any(m[0] in u for u in _interactions(kept)) for m in restored)


def test_guard_can_readmit_a_dropped_main() -> None:
    kw = dict(n_bags=2, prune_selector="fold_vote", band_tolerance=None, graduate=False,
              max_interaction_order=1, prune_drop_z=None, prune_guard_z=0.0, prune_guard_z_dn=0.0,
              prune_main_effects=True)
    quiet = _reg(**kw)
    assert quiet.pruning_report_["guard"]["fired"] is False
    dropped = set(range(6)) - _mains(_kept(quiet))
    assert dropped
    forced = _reg(prune_guard_tol=-0.5, **kw)  # a bar no keep-set can clear: re-admit everything
    assert forced.pruning_report_["guard"]["fired"] is True
    assert dropped <= _mains(_kept(forced))


@pytest.mark.parametrize(
    "kw",
    [
        {},
        dict(prune_selector="fold_vote", max_interaction_order=1, prune_drop_z=None),
        dict(prune_selector="fold_vote", multiclass_prune_cv=False, max_interaction_order=1),
    ],
    ids=["ranked_path", "cv_fold_vote", "single_split_walk"],
)
def test_multiclass_selectors_drop_noise_main_effects(kw: dict[str, object]) -> None:
    x, y = _mc_book()
    off = TBoostClassifier(n_trees=150, seed=3, **kw).fit(x, y)
    on = TBoostClassifier(n_trees=150, seed=3, prune_main_effects=True, **kw).fit(x, y)
    assert off.pruning_report_["main_effect_policy"] == "sticky"
    assert on.pruning_report_["main_effect_policy"] == "prunable"
    assert _mains(_kept(off)) == set(range(5))
    assert {0, 1, 2} <= _mains(_kept(on)) < _mains(_kept(off))
    assert _hierarchical(_kept(on))
    proba = on.predict_proba(x[:50])
    assert np.allclose(proba.sum(1), 1.0)


def test_an_intercept_only_model_deploys_and_round_trips() -> None:
    # A target independent of every feature: the out-of-bag path is lowest at the intercept, so
    # every table goes and the model predicts the observed mean.
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(5000, 4)).astype(np.float32)
    y = rng.poisson(0.3, size=5000).astype(np.float32)
    est = TBoostRegressor(objective="poisson", n_trees=200, seed=1, prune_main_effects=True)
    est.fit(x, y)
    assert est.pruning_report_["kept"] == []
    pred = est.predict(x[:10])
    assert np.allclose(pred, y.mean(), rtol=1e-4)
    assert not np.any(est.feature_importances_)
    rows = est.predict_contributions(x[:2])
    assert all(r["contributions"] == [] for r in rows)
    assert json.loads(est.tables(x))["tables"] == []
    for loaded in (TBoostRegressor.from_bytes(est.to_bytes()), TBoostRegressor.from_json(est.to_json())):
        assert loaded.prune_main_effects is True
        np.testing.assert_array_equal(loaded.predict(x), est.predict(x))


def test_a_dropped_categorical_main_still_scores_every_level() -> None:
    import polars as pl

    rng = np.random.default_rng(0)
    n = 6000
    a, b = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    colour = rng.choice(["red", "green", "blue", "grey", "pink"], n)  # noise
    y = rng.poisson(np.exp(-1.0 + 0.9 * a - 0.6 * b)).astype(np.float32)
    df = pl.DataFrame({"a": a, "b": b, "colour": colour, "y": y})
    est = TBoostRegressor(objective="poisson", n_trees=200, seed=1, prune_main_effects=True)
    est.fit(df, "y")
    assert _kept(est) == {(0,), (1,)}
    new = pl.DataFrame({"a": [0.5, 0.5], "b": [0.1, 0.1], "colour": ["red", "purple"]})
    pred = est.predict(new)
    assert pred[0] == pred[1]  # colour no longer moves the prediction, seen level or not
    assert set(est.cell_indices(new)) == {("a",), ("b",)}


def test_prune_main_effects_needs_pruning() -> None:
    x, y = _book(500)
    with pytest.raises(ValueError, match="prune_main_effects has no effect with prune=False"):
        TBoostRegressor(n_trees=20, prune=False, prune_main_effects=True).fit(x, y)
    with pytest.raises(ValueError, match="prune_main_effects has no effect with prune=False"):
        TBoostClassifier(n_trees=20, prune=False, prune_main_effects=True).fit(x, (y > 0).astype(int))


def test_prune_main_effects_refuses_monotone_constraints() -> None:
    x, y = _book(500)
    est = TBoostRegressor(n_trees=20, monotone_constraints=[1, 0, 0, 0, 0, 0], prune_main_effects=True)
    with pytest.raises(ValueError, match="monotone_constraints"):
        est.fit(x, y)


def test_a_document_saved_before_the_parameter_loads_with_sticky_mains(
    sticky: TBoostRegressor,
) -> None:
    doc = json.loads(sticky.to_json())
    del doc["params"]["prune_main_effects"]
    loaded = TBoostRegressor.from_json(json.dumps(doc))
    assert loaded.prune_main_effects is False
    x, _ = _book(500, seed=4)
    np.testing.assert_array_equal(loaded.predict(x), sticky.predict(x))
