"""PbError -> Python exception typing (spec §12.7) and the param-contract guards that ride
along the same lib.rs error-mapping funnel: fold_of range validation, cell_refit_base's
n_bags>=1 requirement, and the n_jobs=0 guard on per-call thread-pool builders.

Every InvalidInput/ShapeMismatch/InvalidConfig/DtypeMismatch is raised as a class built via
genuine Python multiple inheritance -- TBoostValueError(TBoostError, ValueError) /
TBoostTypeError(TBoostError, TypeError) -- so BOTH `except <builtin>` (the spec-promised
type) and `except TBoostError` (the pre-existing catch-all) match the same raised instance."""

from __future__ import annotations

import numpy as np
import pytest

from t_boost._t_boost import TBoostError, TBoostTypeError, TBoostValueError, _Booster


def _tiny_fit(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)).astype(np.float32)
    y = (0.4 * x[:, 0] - 0.2 * x[:, 1] + rng.normal(scale=0.05, size=n)).astype(np.float32)
    booster = _Booster(objective="squared_error", n_trees=20, seed=0)
    model = booster.fit(x, y)
    return booster, model, x, y


def test_dual_exception_classes_have_the_promised_mro() -> None:
    # Sanity on the classes themselves. This also exercises the dynamic `type(name,
    # (TBoostError, builtin), ns)` construction at module-init time for BOTH classes: if
    # that build were broken, `import t_boost._t_boost` itself would already have failed,
    # which matters for TBoostTypeError since no live PbError::DtypeMismatch path exists
    # today to trigger it end-to-end (grepped both crates: nothing constructs that variant).
    assert issubclass(TBoostValueError, TBoostError)
    assert issubclass(TBoostValueError, ValueError)
    assert issubclass(TBoostTypeError, TBoostError)
    assert issubclass(TBoostTypeError, TypeError)


def test_invalid_config_raises_value_error_and_t_boost_error() -> None:
    # Objective::parse returns PbError::InvalidConfig for an unrecognized objective name.
    with pytest.raises(ValueError, match="invalid config") as excinfo:
        _Booster(objective="not-a-real-objective")
    assert isinstance(excinfo.value, TBoostError), (
        "except TBoostError must still catch this (spec §12.7 dual inheritance)"
    )
    assert isinstance(excinfo.value, TBoostValueError)


def test_invalid_input_raises_value_error_and_t_boost_error() -> None:
    booster = _Booster(objective="squared_error", n_trees=10, seed=0)
    x = np.zeros((5, 0), dtype=np.float32)
    y = np.zeros(5, dtype=np.float32)
    with pytest.raises(ValueError, match="invalid input") as excinfo:
        booster.fit(x, y)
    assert isinstance(excinfo.value, TBoostError)
    assert isinstance(excinfo.value, TBoostValueError)


def test_shape_mismatch_raises_value_error_and_t_boost_error() -> None:
    _, model, x, _ = _tiny_fit()
    x_wrong_width = np.ascontiguousarray(x[:, :2])  # fitted on 3 columns
    with pytest.raises(ValueError, match="shape mismatch") as excinfo:
        model.predict(x_wrong_width)
    assert isinstance(excinfo.value, TBoostError)
    assert isinstance(excinfo.value, TBoostValueError)


def test_noncontiguous_array_raises_value_error_not_type_error() -> None:
    # spec §12.3: ".as_slice() (1-D y/weight/exposure) is zero-copy only when contiguous,
    # else PbError::InvalidInput surfaces as ValueError" -- it was raised as a bare
    # PyTypeError, bypassing PbError entirely. This path does NOT go through PbError/py_err
    # (it's a hand-rolled pyo3 error in array1_to_vec and its duplicated inline siblings), so
    # it's a plain ValueError rather than the dual-inheriting TBoostValueError -- it was
    # never TBoostError-catchable even before this fix, so there's no back-compat surface
    # to preserve here, unlike the four PbError variants above.
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 2)).astype(np.float32)
    y_wide = rng.normal(size=(20, 2)).astype(np.float32)
    y_noncontig = y_wide[:, 0]  # strided column view, not C-contiguous
    assert not y_noncontig.flags["C_CONTIGUOUS"]
    booster = _Booster(objective="squared_error", n_trees=10, seed=0)
    with pytest.raises(ValueError, match="contiguous"):
        booster.fit(x, y_noncontig)


def test_fortran_contiguous_x_is_accepted_and_matches_c_contiguous() -> None:
    # spec §12.3: "an F-contiguous X therefore costs nothing on ingest" -- raw_columns_from_array
    # (lib.rs) used to reject F-order X outright with a TypeError (`.as_slice()` requires
    # C-contiguous). It now takes a fast column-copy path for F-order instead of transposing, so
    # both this test's acceptance check AND the byte-identical-output check matter: an F-order
    # bug could as easily scramble columns as raise.
    rng = np.random.default_rng(4)
    n, p = 250, 4
    x_c = rng.normal(size=(n, p)).astype(np.float32)
    assert x_c.flags["C_CONTIGUOUS"]
    x_f = np.asfortranarray(x_c)
    assert x_f.flags["F_CONTIGUOUS"] and not x_f.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(x_c, x_f)  # same logical values, different memory layout

    y = (0.4 * x_c[:, 0] - 0.2 * x_c[:, 1] + 0.1 * x_c[:, 2] * x_c[:, 3]).astype(np.float32)

    # Fit-time ingest: an F-contiguous X must no longer raise, and must fit a byte-identical
    # model to the C-contiguous fit on the same logical data (same seed, same values).
    booster_c = _Booster(objective="squared_error", n_trees=25, seed=0)
    model_c = booster_c.fit(x_c, y)
    booster_f = _Booster(objective="squared_error", n_trees=25, seed=0)
    model_f = booster_f.fit(x_f, y)
    assert model_c.to_bytes() == model_f.to_bytes()

    # Serve-time ingest: predicting on an F-contiguous view of the same rows must match
    # predicting on the C-contiguous view, against either model (both are identical above).
    np.testing.assert_array_equal(model_c.predict(x_c), model_c.predict(x_f))

    # A single-row / single-column edge (n_rows or n_features degenerate) must not panic.
    x_one_row = np.asfortranarray(x_c[:1])
    assert model_c.predict(x_one_row).shape == (1,)
    x_zero_rows = np.asfortranarray(x_c[:0])
    assert model_c.predict(x_zero_rows).shape == (0,)


def test_fold_of_out_of_range_raises() -> None:
    rng = np.random.default_rng(2)
    n = 60
    x = rng.normal(size=(n, 2)).astype(np.float32)
    y = rng.normal(size=n).astype(np.float32)
    booster = _Booster(objective="squared_error", n_trees=10, seed=0)

    negative_sentinel = np.zeros(n, dtype=np.int64)
    negative_sentinel[0] = -1  # e.g. an sklearn-splitter "unassigned" code
    with pytest.raises(ValueError, match="fold_of") as excinfo:
        booster.fit_prune_folds(x, y, negative_sentinel, k_folds=3)
    assert isinstance(excinfo.value, TBoostError)

    off_by_one = np.zeros(n, dtype=np.int64)
    off_by_one[5] = 3  # >= k_folds
    with pytest.raises(ValueError, match="fold_of"):
        booster.fit_prune_folds(x, y, off_by_one, k_folds=3)


def test_fold_of_in_range_still_works() -> None:
    # The new range check must not over-fire on ordinary, valid fold assignments.
    rng = np.random.default_rng(3)
    n = 90
    x = rng.normal(size=(n, 2)).astype(np.float32)
    y = rng.normal(size=n).astype(np.float32)
    fold_of = rng.integers(0, 3, size=n).astype(np.int64)
    booster = _Booster(objective="squared_error", n_trees=15, seed=0)
    reports = booster.fit_prune_folds(x, y, fold_of, k_folds=3)
    assert len(reports) == 3


def test_cell_refit_base_requires_bagging() -> None:
    with pytest.raises(ValueError, match="cell_refit_base") as excinfo:
        _Booster(objective="squared_error", n_bags=0, cell_refit_base=0.5)
    assert isinstance(excinfo.value, TBoostError)
    # n_bags >= 1 is the documented requirement; construction must succeed.
    _Booster(objective="squared_error", n_bags=2, cell_refit_base=0.5)


def test_n_jobs_zero_rejected_on_predict() -> None:
    _, model, x, _ = _tiny_fit()
    with pytest.raises(ValueError, match="n_jobs") as excinfo:
        model.predict(x, n_jobs=0)
    assert isinstance(excinfo.value, TBoostError)


def test_n_jobs_zero_rejected_on_apply_keepset() -> None:
    _, model, x, y = _tiny_fit()
    weight = np.ones_like(y)
    with pytest.raises(ValueError, match="n_jobs"):
        model.apply_keepset(x, y, weight, [[0]], n_jobs=0)
