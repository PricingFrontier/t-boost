"""Group-aware outer bagging (2026-09-07): with ``groups=`` every bag is a subsample of WHOLE
groups, so a group never straddles a bag's in-bag/out-of-bag boundary and the out-of-bag rows
are honest evidence on panel data. Exercised through the public ``fit(groups=...)`` surface."""
from __future__ import annotations

import pytest
import numpy as np
import polars as pl

from t_boost import TBoostClassifier, TBoostRegressor


def _panel(n_groups: int = 600, rows_per_group: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), rows_per_group)
    rng.shuffle(g)
    x = rng.normal(size=(g.size, 4)).astype(np.float32)
    # A group-level random effect so the panel structure is real, not just labels.
    u = rng.normal(size=n_groups)[g]
    lin = 0.8 * x[:, 0] + 0.5 * x[:, 1] * x[:, 2] + u
    return pl.DataFrame({f"n{i}": x[:, i] for i in range(4)}), lin, g


def test_bags_never_split_a_group_on_the_single_output_path() -> None:
    frame, lin, g = _panel()
    y = (lin + np.random.default_rng(1).normal(scale=0.5, size=lin.size) > 0).astype(int)
    est = TBoostClassifier(n_trees=60, n_bags=4, seed=0, prune=False).fit(frame, y, groups=g)
    mask = np.asarray(est._model.bag_in_bag_mask())  # (n_bags, n_rows)
    assert mask.shape == (4, y.size)
    for bag in mask:
        per_group = {}
        for row, grp in enumerate(g):
            per_group.setdefault(int(grp), set()).add(bool(bag[row]))
        assert all(len(v) == 1 for v in per_group.values()), "a group straddled a bag boundary"
    # It is still a subsample: no bag holds every row, none is empty.
    assert all(0 < bag.sum() < y.size for bag in mask)


def test_group_bags_are_deterministic_and_differ_from_row_bags() -> None:
    frame, lin, g = _panel()
    y = lin + np.random.default_rng(2).normal(scale=0.3, size=lin.size)
    a = TBoostRegressor(n_trees=40, n_bags=3, seed=0, prune=False).fit(frame, y, groups=g)
    b = TBoostRegressor(n_trees=40, n_bags=3, seed=0, prune=False).fit(frame, y, groups=g)
    assert a._model.to_bytes() == b._model.to_bytes()
    rows = TBoostRegressor(n_trees=40, n_bags=3, seed=0, prune=False).fit(frame, y)
    assert rows._model.to_bytes() != a._model.to_bytes()


def test_grouped_multiclass_prune_guard_reads_honest_out_of_bag_rows() -> None:
    frame, lin, g = _panel(n_groups=900, rows_per_group=3)
    rng = np.random.default_rng(3)
    y = np.digitize(lin + rng.normal(scale=0.5, size=lin.size), np.quantile(lin, [0.33, 0.66]))
    est = TBoostClassifier(n_trees=120, n_bags=4, seed=0, prune=True).fit(frame, y, groups=g)
    guard = est.pruning_report_["guard"]
    # Before group-aware bags a grouped K>=3 fit had to fall back to the 10% carve; now the
    # deploy soup's own out-of-bag rows are the evidence.
    assert guard["enabled"] and guard["evidence"] == "oob", guard
    assert guard["oob_rows"] > 0.5 * y.size


@pytest.fixture(autouse=True)
def _legacy_fold_vote_unbanded(monkeypatch: pytest.MonkeyPatch) -> None:
    """This module pins the fold-vote selector (its guard, evidence and box-budget mechanics) on
    unbanded tables: the ranked path and banding became the defaults on 2026-09-26."""
    from t_boost import TBoostClassifier as _C
    from t_boost import TBoostRegressor as _R

    for cls in (_R, _C):
        defaults = cls.__init__.__kwdefaults__
        monkeypatch.setitem(defaults, "prune_selector", "fold_vote")
        monkeypatch.setitem(defaults, "band_tolerance", None)
