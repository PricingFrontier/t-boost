"""Optional-scikit-learn compatibility layer for the estimator module.

scikit-learn is NOT a dependency of t-boost: the estimators run standalone on
numpy/polars. What sklearn buys — when it happens to be installed — is ecosystem interop
(``Pipeline``/``GridSearchCV``/``cross_val_score``/``clone`` and the ``__sklearn_tags__``
contract), so this module re-exports the real sklearn base classes/helpers when the import
succeeds and substitutes behavior-matching shims when it doesn't. The shims cover exactly
what the estimator module uses: ``get_params``/``set_params`` introspection, the two
``score`` mixins, the fitted-state guard, and the continuous-target rejection.

With sklearn installed, everything here IS sklearn — bit-identical estimator behavior.
"""

from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np


def _score_inputs(y: Any, pred: np.ndarray, sample_weight: Any) -> tuple[np.ndarray, Any]:
    target = np.asarray(y)
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 1 or target.shape != pred.shape or target.size == 0:
        raise ValueError("y must be a vector aligned to X")
    if target.dtype.kind in "fc" and not np.all(np.isfinite(target)):
        raise ValueError("y must be finite")
    weight = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if weight is not None and (weight.shape != target.shape or not np.all(np.isfinite(weight))
                               or np.any(weight < 0) or weight.sum() <= 0):
        raise ValueError("sample_weight must be a finite non-negative vector aligned to X")
    return target, weight

if TYPE_CHECKING:
    # mypy always analyzes against the real sklearn names (Any under the
    # ignore_missing_imports override) — the runtime shims below mirror them.
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.multiclass import type_of_target
    from sklearn.utils.validation import check_is_fitted

    SKLEARN_AVAILABLE: bool = True
else:
    try:
        from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.multiclass import type_of_target
        from sklearn.utils.validation import check_is_fitted

        SKLEARN_AVAILABLE = True
    except ImportError:
        SKLEARN_AVAILABLE = False

        class NotFittedError(ValueError, AttributeError):
            """Estimator used before fitting (same bases as sklearn's NotFittedError)."""

        class BaseEstimator:
            """get_params/set_params via __init__-signature introspection (sklearn's rule:
            every constructor argument is mirrored 1:1 to an attribute of the same name)."""

            @classmethod
            def _get_param_names(cls):
                sig = inspect.signature(cls.__init__)
                return sorted(
                    p.name
                    for p in sig.parameters.values()
                    if p.name != "self"
                    and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                )

            def get_params(self, deep=True):
                return {name: getattr(self, name) for name in self._get_param_names()}

            def set_params(self, **params):
                if not params:
                    return self
                valid = set(self._get_param_names())
                for key, value in params.items():
                    if key not in valid:
                        raise ValueError(
                            f"Invalid parameter {key!r} for estimator "
                            f"{type(self).__name__}. Valid parameters are: {sorted(valid)}."
                        )
                    setattr(self, key, value)
                return self

            def __repr__(self):
                sig = inspect.signature(type(self).__init__)
                diffs = []
                for name in self._get_param_names():
                    default = sig.parameters[name].default
                    value = getattr(self, name)
                    if value is not default and not np.array_equal(value, default):
                        diffs.append(f"{name}={value!r}")
                return f"{type(self).__name__}({', '.join(diffs)})"

        class RegressorMixin:
            def score(self, X, y, sample_weight=None):
                """Weighted R^2, matching sklearn's ``r2_score`` conventions."""
                pred = np.asarray(self.predict(X), dtype=np.float64)
                y, w = _score_inputs(y, pred, sample_weight)
                y = y.astype(np.float64)
                if len(y) < 2:
                    warnings.warn("R^2 score is not well-defined with less than two samples.", RuntimeWarning)
                    return float("nan")
                ss_res = float(np.average((y - pred) ** 2, weights=w))
                ss_tot = float(np.average((y - np.average(y, weights=w)) ** 2, weights=w))
                if ss_tot == 0.0:
                    return 1.0 if ss_res == 0.0 else 0.0
                return 1.0 - ss_res / ss_tot

        class ClassifierMixin:
            def score(self, X, y, sample_weight=None):
                """Weighted accuracy, matching sklearn's ``accuracy_score`` conventions."""
                pred = np.asarray(self.predict(X))
                y, w = _score_inputs(y, pred, sample_weight)
                hits = (pred == y).astype(np.float64)
                return float(np.average(hits, weights=w))

        def type_of_target(y, input_name="y"):
            """Just enough of sklearn's type_of_target for the classifier's guard: the
            caller only checks for the two 'continuous*' answers, so every discrete-ish
            input may collapse to 'multiclass'."""
            arr = np.asarray(y)
            suffix = "" if arr.ndim <= 1 else "-multioutput"
            if arr.dtype.kind == "f":
                finite = arr[np.isfinite(arr)]
                if finite.size and np.any(finite != np.round(finite)):
                    return "continuous" + suffix
            return "multiclass" + suffix

        def check_is_fitted(estimator, attributes=None):
            """sklearn's check order: explicit attributes if given, else the estimator's
            own __sklearn_is_fitted__."""
            if attributes is not None:
                attrs = [attributes] if isinstance(attributes, str) else list(attributes)
                fitted = all(hasattr(estimator, a) for a in attrs)
            elif hasattr(estimator, "__sklearn_is_fitted__"):
                fitted = bool(estimator.__sklearn_is_fitted__())
            else:
                fitted = any(
                    v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
                )
            if not fitted:
                raise NotFittedError(
                    f"This {type(estimator).__name__} instance is not fitted yet. Call "
                    "'fit' with appropriate arguments before using this estimator."
                )

__all__ = [
    "SKLEARN_AVAILABLE",
    "BaseEstimator",
    "ClassifierMixin",
    "RegressorMixin",
    "NotFittedError",
    "type_of_target",
    "check_is_fitted",
]
