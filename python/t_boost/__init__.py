"""t-boost: exact oblivious boosting with low-order fANOVA tables.

Trees are depth-3, order-3 by default (`max_depth`, 3..8; `max_interaction_order`, 1..8).
The two knobs are orthogonal: ORDER is how many distinct raw features an effect couples
(= the number of axes an exported table has), RESOLUTION is how finely a tree divides
them (levels past the order cap can only refine a feature already on the tree). Every
model is exactly decomposable at whatever order it was fitted at -- the purification
cascade is n-dimensional, so raising the order relaxes readability, never exactness.

What raising the order really costs is a COMBINATORIAL readability bill, because the
prune keeps a downward-closed order ideal (heredity): one surviving order-k table drags
in its whole subset lattice, `2**k - 1` tables of which `sum_{j>=4} C(k,j)` are
themselves order >= 4 -- 6 at k=5, 22 at k=6, 64 at k=7, 163 at k=8. Order 5 fits a
readable bank comfortably; order 6 spends the whole budget on one effect; 7 and 8 are
expressible and not readable. Choose accordingly.
"""

from __future__ import annotations

from ._t_boost import (
    _Booster,
    _Model,
    _MultiClassModel,
    _TableBank,
    ExactnessError,
    InternalError,
    InvariantError,
    SerializationError,
    TBoostError,
)

__all__ = [
    "TBoostClassifier",
    "TBoostRegressor",
    "PrecisionWarning",
    "recommended_recipe",
    "_Booster",
    "_Model",
    "_MultiClassModel",
    "_TableBank",
    "TBoostError",
    "InvariantError",
    "ExactnessError",
    "SerializationError",
    "InternalError",
]
try:
    # The installed distribution version (single source of truth: the workspace Cargo version,
    # which maturin stamps into the wheel). Mirrors the rustystats packaging approach.
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("t-boost")
except PackageNotFoundError:  # not installed as a distribution (e.g. a bare source tree)
    __version__ = "0.0.0+unknown"


def __getattr__(name: str) -> object:
    if name in {
        "TBoostClassifier",
        "TBoostRegressor",
        "PrecisionWarning",
        "recommended_recipe",
    }:
        # scikit-learn is NOT required here: the estimators run standalone via the
        # `_compat` shims, and become genuine sklearn estimators when sklearn is installed.
        from .sklearn import (
            PrecisionWarning,
            TBoostClassifier,
            TBoostRegressor,
            recommended_recipe,
        )

        return {
            "TBoostClassifier": TBoostClassifier,
            "TBoostRegressor": TBoostRegressor,
            "PrecisionWarning": PrecisionWarning,
            "recommended_recipe": recommended_recipe,
        }[name]
    raise AttributeError(name)
