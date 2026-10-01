"""The default deploy pipeline (2026-09-26): ranked-path prune, then banding of every interaction.

Small synthetic Poisson book with a real pair and a real three-way effect, so both the prune and the
banding have something to decide."""
from __future__ import annotations

import json

import numpy as np
import pytest

from t_boost import TBoostRegressor


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
    assert rep["guard"]["enabled"] is False


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
