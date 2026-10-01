"""The order lift's PRODUCT bar, at the layer that actually enforces it.

`crates/t-boost-core/tests/order_lift.rs` pins exactness and growth. The bar that
matters commercially — *a 4-way table appears only where there is genuine signal* — is
enforced one layer up, by the av37 evidence-gated prune plus its SE-aware no-harm guard,
both of which live in the PyO3/sklearn layer. So it is tested here.

The mechanism, measured: the evidence gate itself selects order <= 3 (it ranks candidates
by held-out per-fold contribution and spends a flat keep budget), and a 4-way survives only
when the guard finds that the selected bank is materially worse on held-out deviance than
the full one and re-admits by mean gain. In other words a 4-way table has to be worth a
measurable out-of-sample deviance gap before anything keeps it.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from t_boost.sklearn import _MAX_DEPTH, _MAX_ORDER, TBoostRegressor

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _make(kind: str, n: int = 6000, seed: int = 5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5)).astype(np.float32)
    if kind == "pair":
        # Highest true interaction is a PAIR (n0 x n2).
        lin = (
            1.2 * np.tanh(x[:, 0])
            - 0.7 * np.sin(1.5 * x[:, 1])
            + 0.9 * x[:, 0] * x[:, 2]
            + 0.3 * x[:, 3]
        )
    else:
        # A genuine 4-way sign product that NO lower-order structure explains.
        quad = np.sign(x[:, 0]) * np.sign(x[:, 1]) * np.sign(x[:, 2]) * np.sign(x[:, 3])
        lin = (
            0.9 * np.tanh(x[:, 0])
            - 0.6 * np.sin(1.5 * x[:, 1])
            + 0.4 * x[:, 4]
            + 2.2 * quad
        )
    frame = pl.DataFrame({f"n{i}": x[:, i] for i in range(5)})
    return frame, lin + rng.normal(scale=0.15, size=n), lin


def _fit(kind: str, max_order: int):
    frame, y, truth = _make(kind)
    model = TBoostRegressor(
        n_trees=200,
        seed=7,
        n_jobs=4,
        prune=True,
        max_interaction_order=max_order,
        max_depth=4,
    )
    model.fit(frame, y)
    report = model.pruning_report_
    kept: dict[int, int] = {}
    for u in report.get("kept", []):
        kept[len(u)] = kept.get(len(u), 0) + 1
    mse = float(np.mean((np.asarray(model.predict(frame), dtype=float) - truth) ** 2))
    return kept, report, mse


def test_an_order_three_sufficient_target_is_untouched_by_raising_the_cap():
    """The bar, in its strongest form: on data whose highest true interaction is a pair,
    `max_interaction_order=4` must produce the SAME deployed bank as `=3`.

    Not "few 4-way tables" — none, and the rest of the keep set identical too. If raising
    the cap perturbed an order-3-sufficient book at all, the knob would be unsafe to expose
    to a tuner, because the tuner would sometimes pick it for noise reasons."""
    kept3, _, mse3 = _fit("pair", 3)
    kept4, report4, mse4 = _fit("pair", 4)

    assert kept4.get(4, 0) == 0, f"a pairwise-only target kept {kept4.get(4)} 4-way tables"
    assert kept3 == kept4, f"raising the cap moved the keep set: {kept3} -> {kept4}"
    assert mse3 == pytest.approx(mse4, rel=1e-12), (mse3, mse4)
    assert report4["effective_order"] <= 3


def test_a_genuine_four_way_target_admits_four_way_tables_and_fits_better():
    """The other half: where the 4-way signal is real, the lift must actually deliver."""
    kept3, _, mse3 = _fit("quad", 3)
    kept4, report4, mse4 = _fit("quad", 4)

    assert kept4.get(4, 0) > 0, "a genuine 4-way target kept no 4-way table"
    assert mse4 < mse3, f"order 4 must beat order 3 on 4-way data ({mse4} vs {mse3})"
    # The 4-ways got in through the no-harm guard, not through the evidence gate's own
    # budget: the gate selects order <= 3, then the guard measures that the selected bank
    # is materially worse than the full one and re-admits. That is the mechanism the
    # "genuine signal" claim rests on, so pin it rather than just the outcome.
    assert report4["guard"]["fired"] is True


def test_the_four_way_tables_kept_are_a_small_minority():
    """Minimality: a 4-way table is the exception in the deployed bank, not the bulk of it.

    A bank whose 4-way tables outnumbered its pairs would be unreadable regardless of how
    well it scored, so this is a product constraint, not a statistical one."""
    kept, _, _ = _fit("quad", 4)
    total = sum(kept.values())
    assert kept.get(4, 0) <= max(2, total // 5), (
        f"4-way tables are {kept.get(4, 0)} of {total} kept effects; they are supposed to "
        "be the exception"
    )
    # And heredity holds downward: nothing of order 4 without pairs and triples beneath it.
    assert kept.get(2, 0) >= kept.get(4, 0)
    assert kept.get(3, 0) >= kept.get(4, 0)


def test_an_order_above_the_cap_is_refused_in_python():
    """The cap is `_MAX_ORDER`, whatever it currently is -- pinned against the constant.

    This test used to hard-code `max_interaction_order=5`, which the order lift made a
    refusal and the high-order lift makes a legal fit. A cap test written as a literal
    silently becomes a test of nothing the moment the cap moves, so read the constant.
    """
    frame, y, _ = _make("pair", n=500)
    over = _MAX_ORDER + 1
    with pytest.raises(ValueError, match="max_interaction_order"):
        TBoostRegressor(
            max_interaction_order=over, max_depth=_MAX_DEPTH, n_trees=5
        ).fit(frame, y)


def test_an_order_exceeding_the_depth_is_refused_in_python():
    """`max_order > max_depth` is unreachable, not tighter — a tree needs one level per
    distinct feature. Refuse it rather than silently behaving as `max_depth`."""
    frame, y, _ = _make("pair", n=500)
    with pytest.raises(ValueError, match="max_depth"):
        TBoostRegressor(max_interaction_order=4, max_depth=3, n_trees=5).fit(frame, y)


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
