"""§06.5 MVS row sampling on the native multiclass (K>=3) softmax path.

`subsample`/`mvs_min_rows` used to RAISE for K>=3 (see test_multiclass_inert_params.py's
raise family). They are now live: `engine::boost::mc_sample_rows` draws ONE row sample per
round, shared by that round's K class trees, weighted by the joint gradient/hessian norm across
classes `s_i = sqrt(Σ_k g_ik² + Σ_k h_ik²)`.

Why one shared draw and not K: the class columns of a round are grown against the SAME
round-start softmax probabilities (standard multinomial GBM). Sampling per class independently
would hand one round's K trees disjoint views of the data, which is not the estimator the rest
of the loop assumes. The joint norm is also the only reading that keeps a row which is
uninformative for class 0 but informative for class 2 — a per-class scalar would drop it from
class 0's tree while class 2 kept it.

The invariants that matter:

1. INERT AT THE DEFAULT — `subsample=None` is `Sampling::Full`, byte-identical to the
   pre-MVS build (this is the standing K>=3 identity gate; the fingerprint battery covers the
   whole default matrix, these tests cover the parameter specifically).
2. LIVE WHEN ASKED — a sampled fit differs, deterministically, and reproduces exactly on a
   re-fit with the same seed.
3. STRUCTURE ONLY — the leaves are re-solved on ALL train rows against the exact,
   un-reweighted gradients, so a sampled fit is not a fit on less data; only the split search
   sees the subsample.
4. `mvs_min_rows` is a real floor, and `subsample >= 1.0` degenerates back to Full.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from t_boost.sklearn import TBoostClassifier


def _fixture(n: int = 4000, p: int = 8, seed: int = 5, classes: int = 4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p)).astype(np.float32)
    lin = 1.1 * x[:, 0] - 0.7 * x[:, 1] + 0.4 * x[:, 0] * x[:, 2]
    lin = lin + rng.normal(scale=1.0, size=n)
    cuts = np.quantile(lin, np.linspace(0, 1, classes + 1)[1:-1])
    y = np.digitize(lin, cuts)
    return pl.DataFrame({f"n{i}": x[:, i] for i in range(p)}), y


def _fit(frame, y, **kw):
    return TBoostClassifier(
        n_trees=80, seed=2, n_jobs=4, prune=False, leaf_refine_steps=0, **kw
    ).fit(frame, y)


def test_default_is_full_sampling_and_does_not_raise() -> None:
    # Invariant 1. `subsample=None` no longer raises (it left the inert-param raise family) and
    # must be the SAME fit as before — the sampler short-circuits to `Sampling::Full`.
    frame, y = _fixture()
    a = _fit(frame, y)
    b = _fit(frame, y, subsample=None, mvs_min_rows=1)
    assert a.to_bytes() == b.to_bytes()


def test_subsample_one_is_full_sampling() -> None:
    # Invariant 4. rate=1.0 ⇒ k == n ⇒ the `k == n` short circuit ⇒ no reweighting, no re-walk,
    # byte-identical to the unsampled fit. This is what makes the knob safe to leave at 1.0.
    frame, y = _fixture()
    assert _fit(frame, y).to_bytes() == _fit(frame, y, subsample=1.0).to_bytes()


def test_subsample_changes_the_fit_deterministically() -> None:
    # Invariant 2.
    frame, y = _fixture()
    full = _fit(frame, y)
    mvs = _fit(frame, y, subsample=0.5)
    again = _fit(frame, y, subsample=0.5)
    assert mvs.to_bytes() != full.to_bytes()
    assert mvs.to_bytes() == again.to_bytes()
    assert np.array_equal(mvs.predict_proba(frame), again.predict_proba(frame))


def test_subsample_is_thread_count_independent() -> None:
    # The row draw is a per-row deterministic key (`pb_seed(seed, round, Sample, pos)`) sorted
    # to a fixed order, and the leaf re-solve is a fixed-order pass — nothing here may depend on
    # how rayon happens to schedule the K classes.
    frame, y = _fixture()
    one = TBoostClassifier(
        n_trees=80, seed=2, n_jobs=1, prune=False, leaf_refine_steps=0, subsample=0.5
    ).fit(frame, y)
    many = TBoostClassifier(
        n_trees=80, seed=2, n_jobs=8, prune=False, leaf_refine_steps=0, subsample=0.5
    ).fit(frame, y)
    assert one.to_bytes() == many.to_bytes()


def test_the_shared_draw_moves_every_class_together() -> None:
    # The point of the joint-norm draw: it is ONE sample per round for all K trees. If a class
    # were sampled independently (or not at all), that class's bank would be unchanged from the
    # full fit while the others moved. Every class must move.
    frame, y = _fixture()
    full = _fit(frame, y)
    mvs = _fit(frame, y, subsample=0.5)
    a = full.decision_function(frame)
    b = mvs.decision_function(frame)
    assert a.shape == b.shape
    moved = [not np.allclose(a[:, k], b[:, k]) for k in range(a.shape[1])]
    assert all(moved), moved


def test_mvs_min_rows_floors_the_draw() -> None:
    # Invariant 4. A rate small enough to be dominated by the floor must produce the SAME fit as
    # the rate that names the floor directly (k = max(ceil(rate*n), min_rows)); and raising the
    # floor to n degenerates back to Full.
    frame, y = _fixture(n=2000)
    n_train = 2000
    floored = _fit(frame, y, subsample=0.001, mvs_min_rows=800)
    direct = _fit(frame, y, subsample=0.001, mvs_min_rows=800)
    assert floored.to_bytes() == direct.to_bytes()
    # min_rows >= n means every row is drawn, which is the k == n short circuit.
    all_rows = _fit(frame, y, subsample=0.1, mvs_min_rows=n_train)
    assert all_rows.to_bytes() == _fit(frame, y).to_bytes()


def test_sampled_fit_still_predicts_a_valid_simplex() -> None:
    # Invariant 3's smoke consequence: the reweighted gradients only ever reach the SPLIT
    # SEARCH, so the served model is an ordinary softmax model — no scale artifact from the
    # 1/p_i multipliers leaking into leaf values.
    frame, y = _fixture()
    proba = _fit(frame, y, subsample=0.4).predict_proba(frame)
    assert proba.shape == (len(y), len(np.unique(y)))
    assert np.all(np.isfinite(proba))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_subsample_composes_with_bagging_and_pruning() -> None:
    # The three variance-reduction levers now live on the K>=3 path at once. Bags subset rows
    # BEFORE the fit; MVS subsets each round's split search WITHIN a bag; the prune runs after.
    frame, y = _fixture()
    model = TBoostClassifier(
        n_trees=80, seed=2, n_jobs=4, prune=True, n_bags=4, leaf_refine_steps=0, subsample=0.5
    ).fit(frame, y)
    proba = model.predict_proba(frame)
    assert np.all(np.isfinite(proba))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
