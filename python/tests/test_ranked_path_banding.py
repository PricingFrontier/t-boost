"""The default deploy pipeline (2026-09-26): ranked-path prune, then banding of every interaction.

Small synthetic Poisson book with a real pair and a real three-way effect, so both the prune and the
banding have something to decide."""
from __future__ import annotations

import json

import numpy as np
import pytest

from sklearn.base import clone

from t_boost import TBoostClassifier, TBoostRegressor


def _book(n: int = 6000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n, 5)).astype(np.float32)
    eta = (-1.0 + 0.6 * x[:, 0] - 0.4 * x[:, 1] + 0.8 * (x[:, 0] > 0.5) * (x[:, 2] > 0.4)
           + 0.5 * (x[:, 1] < 0.3) * (x[:, 3] > 0.6) * (x[:, 4] > 0.5))
    y = rng.poisson(np.exp(eta)).astype(np.float32)
    return x, y


def _fit(**kw: object) -> TBoostRegressor:
    x, y = _book()
    return TBoostRegressor(objective="poisson", n_trees=300, max_depth=4, seed=3, **kw).fit(x, y)


def _deviance(y: np.ndarray, mu: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(np.mean(2 * (t - y + mu)))


@pytest.fixture(scope="module")
def banded() -> TBoostRegressor:
    return _fit()


@pytest.fixture(scope="module")
def unbanded() -> TBoostRegressor:
    return _fit(band_tolerance=None)


def test_ranked_path_is_the_default_selector_and_reports_its_path(banded: TBoostRegressor) -> None:
    rep = banded.pruning_report_
    assert rep["selector"] == "ranked_path"
    path = rep["path"]
    assert len(path["sizes"]) == len(path["oob_deviance"]) >= 2
    assert 0 <= path["best"] < len(path["sizes"])
    assert "guard" not in rep and "slope" not in rep  # both removed on 2026-10-14


def test_banding_is_the_default_and_shrinks_every_interaction(banded: TBoostRegressor) -> None:
    rep = banded.pruning_report_["banding"]
    assert rep["tolerance"] == pytest.approx(0.75)
    before = sum(rep["cells_before"].values())
    after = sum(rep["cells_after"].values())
    assert 0 < after < before
    # the contract: the combined cross-fitted move is within the binding budget (or banding says
    # why it skipped); the binding budget is the tighter of the noise and deviance sides
    assert rep["budget"] == pytest.approx(min(rep["noise_budget"], rep["deviance_budget"]))
    assert rep.get("skipped") or rep["combined_mse"] <= rep["budget"] * (1 + 1e-9)


def test_banded_predictions_stay_close_to_the_unbanded_model(banded: TBoostRegressor, unbanded: TBoostRegressor) -> None:
    x, y = _book(4000, seed=11)
    d_band = _deviance(y, banded.predict(x))
    d_full = _deviance(y, unbanded.predict(x))
    assert abs(d_band - d_full) / d_full < 0.01


def test_banded_model_round_trips_through_json_and_bytes(banded: TBoostRegressor) -> None:
    from t_boost._t_boost import _TableModel

    x, _ = _book(500, seed=5)
    tm = banded._model
    x32 = np.ascontiguousarray(x, dtype=np.float32)
    ref = np.asarray(tm.predict_raw(x32))
    doc = json.loads(tm.to_json())
    assert doc["schema_version"] >= 7
    assert any(a.get("band_of") for t in doc["model"]["bank"]["tables"] for a in t["axes"])
    for back in (_TableModel.from_json(tm.to_json()), _TableModel.from_bytes(bytes(tm.to_bytes()))):
        assert np.array_equal(np.asarray(back.predict_raw(x32)), ref)


def test_rating_export_reports_band_level_tables(banded: TBoostRegressor) -> None:
    export = json.loads(banded._model.tables(None))
    for t in export["tables"]:
        assert len(t["values"]) == int(np.prod(t["shape"]))
        for ax, dim in zip(t["axes"], t["shape"]):
            assert ax["cells"] == dim
            if not ax.get("levels"):
                assert ax["cells"] == len(ax["borders"]) + 2


def test_banding_is_deterministic(banded: TBoostRegressor) -> None:
    again = _fit()
    x, _ = _book(500, seed=5)
    assert np.array_equal(banded.predict(x), again.predict(x))


def test_band_tolerance_none_leaves_tables_on_the_merged_grid(unbanded: TBoostRegressor) -> None:
    assert "banding" not in unbanded.pruning_report_
    doc = json.loads(unbanded._model.to_json())
    assert not any(a.get("band_of") for t in doc["model"]["bank"]["tables"] for a in t["axes"])


def _heredity_passes(mains: list, inter: list) -> list:
    """The ranked path's original admission loop (before 2026-09-27): append each support to a
    pending list, then repeat full passes over it until a pass admits nothing."""
    admitted = set(mains)
    seq: list = []
    pending: list = []
    for u in inter:
        pending.append(u)
        moved = True
        while moved:
            moved = False
            for v in list(pending):
                if len(v) == 2 or all(
                    tuple(x for j, x in enumerate(v) if j != i) in admitted for i in range(len(v))
                ):
                    admitted.add(v)
                    seq.append(v)
                    pending.remove(v)
                    moved = True
    return seq


def test_heredity_sequence_reproduces_the_pass_loop_order() -> None:
    import itertools
    import random

    from t_boost.sklearn import _heredity_sequence

    rng = random.Random(0)
    for _ in range(1500):
        n_feat = rng.randint(3, 9)
        top_order = rng.choice([2, 3, 3, 4, 5])
        tops = {
            tuple(sorted(rng.sample(range(n_feat), rng.randint(2, min(top_order, n_feat)))))
            for _ in range(rng.randint(1, 25))
        }
        sup = {c for t in tops for k in range(1, len(t) + 1) for c in itertools.combinations(t, k)}
        if rng.random() < 0.2:  # a missing subset: its supersets must never enter
            pairs = sorted(s for s in sup if len(s) == 2)
            if pairs:
                sup.discard(rng.choice(pairs))
        share = {u: (rng.random() if rng.random() < 0.9 else 0.0) for u in sup}
        mains = [u for u in sorted(sup) if len(u) == 1]
        inter = sorted((u for u in sup if len(u) > 1), key=lambda u: (-share[u], u))
        assert _heredity_sequence(mains, inter) == _heredity_passes(mains, inter)


def test_pricing_report_level_labels_index_the_merged_grid_of_a_banded_feature():
    import polars as pl

    rng = np.random.default_rng(7)
    n = 30000
    levels = [f"L{i}" for i in range(12)]
    veh = rng.choice(levels, n)
    age = rng.integers(18, 80, n).astype(float)
    effect = np.array([int(v[1:]) for v in veh]) * 0.05
    exposure = rng.uniform(0.1, 1, n)
    mu = np.exp(-2 + 0.01 * (age - 40) ** 2 / 40 + effect + (age > 50) * (effect - 0.3))
    X = pl.DataFrame({"age": age, "veh": veh})
    y = rng.poisson(mu * exposure)
    m = TBoostRegressor(objective='poisson', n_trees=300, n_jobs=1).fit(X, y, exposure=exposure)
    bank = json.loads(m.tables(X))
    banded = [a for t in bank['tables'] for a in t['axes']
              if a['name'] == 'veh' and len(t['axes']) > 1]
    assert banded and banded[0]['cells'] < 13, 'the fixture must band the veh axis'
    report = m.pricing_report(X, y, exposure=exposure)
    factor = next(f for f in report['actual_vs_expected'] if f['feature'] == 'veh')
    x32, cat_x = m._serve_design(X)
    raw = m._model.raw_feature_names().index('veh')
    cells = np.asarray(m._model.cell_indices(x32, cat_x=cat_x))[:, raw]
    assert factor['axis']['cells'] == len(factor['rows'])
    for level in factor['axis']['levels']:
        rows = veh == level['label']
        if rows.any():
            assert set(cells[rows]) == {level['cell']}, level
            assert factor['rows'][level['cell']] == int(rows.sum()), level


def _floor_fixture(n=6000, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    eta = 0.4 * X[:, 0] - 0.3 * X[:, 1] + 0.25 * X[:, 0] * X[:, 1] + 0.02 * X[:, 2] * X[:, 3]
    return X, rng.poisson(np.exp(eta - 1.0)), rng


def test_min_variance_share_drops_only_small_tables_nothing_kept_needs():
    X, y, _ = _floor_fixture()
    base = TBoostRegressor(objective="poisson", n_trees=150, n_jobs=1, seed=0,
                           band_tolerance=None)
    off = clone(base).fit(X, y)
    path = off.pruning_report_["path"]
    assert path["min_variance_share"] == 0.0 and path["dropped_low_share"] == []
    scores = {tuple(s["u"]): s["variance_share"] for s in off.pruning_report_["table_scores"]}
    total = sum(max(v, 0.0) for v in scores.values())
    share = {u: max(v, 0.0) / total for u, v in scores.items()}
    kept_inter = [tuple(u) for u in off.pruning_report_["kept"] if len(u) > 1]
    assert kept_inter, "the fixture must keep interactions"
    floor = sorted(share[u] for u in kept_inter)[len(kept_inter) // 2] + 1e-12
    on = clone(base).set_params(prune_min_variance_share=floor).fit(X, y)
    rep = on.pruning_report_
    dropped = {tuple(u) for u in rep["path"]["dropped_low_share"]}
    kept = {tuple(u) for u in rep["kept"]}
    assert dropped and not dropped & kept
    anchors = [set(u) for u in kept if len(u) == 1 or share[u] >= floor]
    for u in dropped:  # below the floor, and no kept table at or above it is built on it
        assert share[u] < floor and not any(set(u) <= a for a in anchors)
    for u in kept:  # clears the floor, or a table that does is built on it
        assert any(set(u) <= a for a in anchors)
        assert all(tuple(sorted(v)) in kept for v in _proper_subsets(u))  # heredity
    deployed = {tuple(int(name[1:]) for name in key) for key in on.cell_indices(X)}
    assert not dropped & deployed
    assert on.get_params()["prune_min_variance_share"] == floor
    html = on.html_report(X, y)
    assert "below the floor" in html and "prune_min_variance_share" in html


def _proper_subsets(u):
    from itertools import combinations

    return [c for k in range(1, len(u)) for c in combinations(u, k)]


def test_min_variance_share_keeps_a_small_main_a_significant_pair_needs():
    # Prunable mains: a main effect under the floor stays when a kept interaction above it
    # contains it (dropping it would have to drop the interaction too).
    rng = np.random.default_rng(11)
    n = 8000
    X = rng.normal(size=(n, 4))
    eta = 0.5 * X[:, 0] + 0.4 * X[:, 0] * np.sign(X[:, 1])  # x1 matters only through the pair
    y = rng.poisson(np.exp(eta - 1.0))
    m = TBoostRegressor(objective="poisson", n_trees=150, n_jobs=1, seed=0, band_tolerance=None,
                        prune_main_effects=True, prune_min_variance_share=0.02).fit(X, y)
    scores = {tuple(s["u"]): s["variance_share"] for s in m.pruning_report_["table_scores"]}
    total = sum(max(v, 0.0) for v in scores.values())
    kept = {tuple(u) for u in m.pruning_report_["kept"]}
    if (0, 1) in kept and scores[(0, 1)] / total >= 0.02:
        assert (1,) in kept


@pytest.mark.parametrize("bad", [-0.1, 1.0, 2.0])
def test_min_variance_share_is_validated(bad):
    X, y, _ = _floor_fixture(n=1500)
    with pytest.raises(ValueError, match="prune_min_variance_share"):
        TBoostRegressor(objective="poisson", n_trees=40, n_jobs=1,
                        prune_min_variance_share=bad).fit(X, y)


def test_min_variance_share_multiclass_and_binding():
    X, _, rng = _floor_fixture(n=3000)
    y = np.where(X[:, 0] * X[:, 1] > 0.3, 2, (X[:, 2] > 0).astype(int))
    m = TBoostClassifier(n_trees=60, n_jobs=1, seed=0, band_tolerance=None,
                         prune_min_variance_share=0.05).fit(X, y)
    sel = m.pruning_report_["selection"]
    assert sel["min_variance_share"] == 0.05 and "dropped_low_share" in sel
    for u in m.pruning_report_["kept"]:
        assert u not in sel["dropped_low_share"]
    fv = TBoostRegressor(objective="poisson", n_trees=40, n_jobs=1, prune_selector="fold_vote",
                         prune_min_variance_share=0.01).fit(X, rng.poisson(1.0, len(X)))
    row = next(r for r in fv.binding_report_ if r["param"] == "prune_min_variance_share")
    assert row["status"] == "INERT"


def test_min_variance_share_leaves_a_skipped_multiclass_path_alone():
    # One bag leaves no out-of-bag rows, so the path is skipped and the full bank kept: the floor
    # must not drop tables then (the single-output path does not), as `binding_report_` says.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 3))
    y = rng.integers(0, 3, 120)
    fits = [TBoostClassifier(n_trees=12, n_bags=1, n_jobs=1, seed=0, validation_fraction=None,
                             prune_min_variance_share=share).fit(X, y) for share in (0.0, 0.99)]
    assert "skipped" in fits[1].pruning_report_["selection"]
    assert "dropped_low_share" not in fits[1].pruning_report_["selection"]
    banks = [{c: [(t["feature_names"], t["values"]) for t in b["tables"]]
              for c, b in json.loads(m.tables(X)).items()} for m in fits]
    assert banks[1] == banks[0]
    np.testing.assert_array_equal(fits[1].predict_proba(X), fits[0].predict_proba(X))
    row = next(r for r in fits[1].binding_report_ if r["param"] == "prune_min_variance_share")
    assert row["status"] == "INERT"
