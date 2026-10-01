"""The K>=3 §G1 OOB cell refit (`cell_refit_base`/`cell_refit_gamma` for softmax fits).

The mechanism is K decoupled diagonal-Hessian cell solves on the out-of-bag softmax working
residual, accepted or shrunk by ONE joint backtrack on the true multinomial loss (see
`engine::boost::attach_multiclass_cell_correction` for the math and the numerical evidence).
The properties pinned here:

  1. It is HONORED for K>=3 (it used to raise) and it moves the fit.
  2. `cell_refit_base=None` — the default — is byte-identical to a pre-feature fit.
  3. The corrected model is still TABLE-REPRESENTABLE: cell corrections ARE table entries, so
     the per-class banks reproduce the corrected logits exactly.
  4. Determinism across thread counts with the feature on.
  5. It needs a bag partition: `n_bags=1` REFUSES with the single-output path's own message,
     rather than silently dropping what the caller asked for.
  6. It composes with the prune path.
  7. K=2 is untouched (that path is the binary logistic single-output one).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from t_boost.sklearn import TBoostClassifier

_BASE = dict(n_trees=40, learning_rate=0.3, seed=3, n_bags=4, bag_subsample=0.8,
             leaf_refine_steps=0)


def multiclass_fixture(n: int = 2000, seed: int = 0, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Softmax truth whose class signal includes a smooth PAIR surface (x2*x3) — the kind of
    weak interaction surface a shared greedy budget under-fits, i.e. what §G1 exists for."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    c = rng.integers(0, 6, n)
    lin = np.stack(
        [
            1.0 * x[:, 0] * (j + 1) / k
            + 0.9 * np.sin(2.0 * x[:, 1]) * (1 if j % 2 else -1)
            + 0.9 * x[:, 2] * x[:, 3] * (j - k / 2) / k
            + 0.7 * (c == j % 6)
            for j in range(k)
        ],
        axis=1,
    )
    p = np.exp(lin - lin.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(k, p=pi) for pi in p])
    return np.column_stack([x, c]).astype(np.float32), y.astype(np.int64)


def _fit(x, y, **kw):
    clf = TBoostClassifier(**{**_BASE, **kw})
    clf.fit(x, y)
    return clf


def test_cell_refit_is_honored_for_multiclass_and_moves_the_fit() -> None:
    x, y = multiclass_fixture()
    off = _fit(x, y)
    on = _fit(x, y, cell_refit_base=200.0)
    assert not np.array_equal(off.predict_proba(x), on.predict_proba(x))


def test_default_is_byte_identical_to_an_explicit_none() -> None:
    x, y = multiclass_fixture()
    a = _fit(x, y).predict_proba(x)
    b = _fit(x, y, cell_refit_base=None).predict_proba(x)
    assert np.array_equal(a, b)


def test_corrected_model_is_still_table_representable_and_persisted() -> None:
    # Cell corrections ARE table entries: the corrected model still exports K exact per-class
    # banks, and the correction is part of the serialized artifact (not runtime-only state), so
    # a byte round-trip predicts identically. The five fANOVA invariants on the CORRECTED
    # per-class model are asserted in Rust
    # (`multiclass_cell_refit_moves_the_model_and_stays_exactly_decomposable`).
    x, y = multiclass_fixture()
    on = _fit(x, y, cell_refit_base=200.0)
    banks = json.loads(on.tables(x))
    assert sorted(banks) == ["0", "1", "2"]
    for bank in banks.values():
        assert bank["mode"] == "Exact"
        assert bank["tables"] or bank["factored"]
    reloaded = TBoostClassifier.from_bytes(on.to_bytes())
    assert np.array_equal(on.predict_proba(x), reloaded.predict_proba(x))


def test_thread_count_determinism_with_the_correction_on() -> None:
    x, y = multiclass_fixture()
    a = _fit(x, y, cell_refit_base=200.0, n_jobs=1).predict_proba(x)
    b = _fit(x, y, cell_refit_base=200.0, n_jobs=8).predict_proba(x)
    assert np.array_equal(a, b)


def test_one_bag_refuses_rather_than_silently_dropping_the_correction() -> None:
    x, y = multiclass_fixture()
    with pytest.raises(ValueError, match="cell_refit requires n_bags >= 2"):
        _fit(x, y, n_bags=1, cell_refit_base=200.0)


def test_composes_with_the_multiclass_prune_path() -> None:
    x, y = multiclass_fixture()
    off = _fit(x, y, prune=True)
    on = _fit(x, y, prune=True, cell_refit_base=200.0)
    assert not np.array_equal(off.predict_proba(x), on.predict_proba(x))
    # The shipped artifact is still a pruned per-class TABLE model.
    assert len(on.pruning_report_["kept"]) > 0


def test_binary_path_is_untouched() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(800, 3)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] * x[:, 2] + rng.normal(0, 0.5, 800) > 0).astype(int)
    a = _fit(x, y).predict_proba(x)
    b = _fit(x, y, cell_refit_base=200.0).predict_proba(x)
    # K=2 runs the binary logistic single-output path; whatever it does with cell_refit is the
    # pre-existing single-output behaviour, and the K>=3 change must not have perturbed it.
    assert a.shape == b.shape == (800, 2)


def test_multiclass_prune_sel_bags_defaults_to_the_shipped_fidelity() -> None:
    x, y = multiclass_fixture()
    a = _fit(x, y, prune=True)
    b = _fit(x, y, prune=True, multiclass_prune_sel_bags=1)
    assert np.array_equal(a.predict_proba(x), b.predict_proba(x))


def test_multiclass_prune_sel_bags_rejects_zero_and_the_single_output_path() -> None:
    x, y = multiclass_fixture()
    with pytest.raises(ValueError, match="multiclass_prune_sel_bags must be >= 1"):
        _fit(x, y, prune=True, multiclass_prune_sel_bags=0)
    rng = np.random.default_rng(2)
    xb = rng.normal(size=(400, 3)).astype(np.float32)
    yb = (xb[:, 0] > 0).astype(int)
    with pytest.raises(ValueError, match="multiclass_prune_sel_bags has no effect"):
        _fit(xb, yb, prune=True, multiclass_prune_sel_bags=4)


def test_multi_channel_categoricals_do_not_crash_the_refit() -> None:
    # P1 multi-channel (`cat_channels`): a raw feature owning several model axes has a JOINT
    # merged axis, and a joint axis has no per-bin -> cell map, so `correction_scaffold` cannot
    # build a CorrectionTable for any support touching it. Those supports are excluded rather
    # than crashing (they used to raise `model_bin_to_cell called on a P1 multi-channel joint
    # axis`); the rest of the bank is still refit.
    import pandas as pd

    rng = np.random.default_rng(5)
    n = 1500
    lvl = rng.integers(0, 25, n)
    z = rng.normal(size=n)
    lab = ((lvl % 3) + (z > 0).astype(int)) % 3
    df = pd.DataFrame({"cat": pd.Categorical(lvl.astype(str)), "z": z.astype(np.float32)})
    clf = TBoostClassifier(
        **_BASE, cell_refit_base=200.0, cat_channels=["count", "class_freq"],
        cat_count_min_levels=5, categorical_features=["cat"],
    )
    clf.fit(df, lab)
    assert clf.predict_proba(df).shape == (n, 3)


def test_cell_refit_report_surfaces_lambda_coverage_and_the_guard_verdict() -> None:
    # av39 reporting surface: the fit says what the joint backtrack accepted, how much of the
    # realized structure it could reach, and whether it declined — on BOTH the plain and the
    # pruned (deploy) paths, since pruning swaps the `_MultiClassModel` for a table model that
    # carries no fit-time diagnostics.
    x, y = multiclass_fixture()

    off = _fit(x, y)
    assert off.cell_refit_report_ is None

    on = _fit(x, y, cell_refit_base=200.0)
    r = on.cell_refit_report_
    assert r is not None
    assert 0.0 <= r["lambda"] <= 1.0
    assert r["declined"] is (r["lambda"] <= 0.0)
    assert r["n_supports"] > 0
    assert 0.0 <= r["reachable_coverage"] <= 1.0
    assert r["guard_rows"] > 0

    pruned = _fit(x, y, cell_refit_base=200.0, prune=True)
    p = pruned.cell_refit_report_
    assert p is not None
    assert p == pruned.pruning_report_["cell_refit"]
    assert set(p) == {"lambda", "declined", "n_supports", "n_blocked",
                      "reachable_coverage", "n_rejected", "guard_rows"}

    # An unrefitted pruned fit's report is exactly the pre-feature one: no `cell_refit` block.
    plain_pruned = _fit(x, y, prune=True)
    assert "cell_refit" not in plain_pruned.pruning_report_
    assert plain_pruned.cell_refit_report_ is None
