"""The serialized model alone, without the estimator envelope.

`to_bytes()`/`to_json()` also record the constructor parameters and fit-time column specs, so two
estimators fitted with different (but model-equivalent) arguments no longer serialize to equal
blobs. Tests asserting that a parameter does or does not move the deployed artifact compare these
instead.
"""

from __future__ import annotations

from typing import Any


def _native(est: Any) -> Any:
    multi = getattr(est, "_multi_model", None)
    return multi if multi is not None else est._model


def model_bytes(est: Any) -> bytes:
    return bytes(_native(est).to_bytes())


def model_json(est: Any) -> str:
    return str(_native(est).to_json())


def _numbers(doc: Any, path: str = "") -> dict[str, Any]:
    if isinstance(doc, dict):
        out: dict[str, Any] = {}
        for key, value in doc.items():
            out.update(_numbers(value, f"{path}.{key}"))
        return out
    if isinstance(doc, list):
        out = {}
        for i, value in enumerate(doc):
            out.update(_numbers(value, f"{path}[{i}]"))
        return out
    return {path: doc}


def assert_exports_close(a: str, b: str, rtol: float = 1e-12) -> None:
    """Two `tables()` JSON exports agree structurally and numerically to ``rtol``. Re-centring
    stored tables on the rows they were purified on gives the same ledger up to float64
    rounding, not byte-identical JSON."""
    import json
    import math

    left, right = _numbers(json.loads(a)), _numbers(json.loads(b))
    assert left.keys() == right.keys()
    for key, x in left.items():
        y = right[key]
        if isinstance(x, (int, float)) and not isinstance(x, bool) and isinstance(y, (int, float)):
            assert math.isclose(x, y, rel_tol=rtol, abs_tol=rtol), (key, x, y)
        else:
            assert x == y, (key, x, y)


def ensemble_fit(est: Any, *args: Any, **kwargs: Any) -> Any:
    """Fit an unpruned single-output estimator but keep its raw tree ensemble as ``_model``.

    Every fit deploys rating tables, so a ``prune=False`` fit converts its ensemble to the full
    table bank before returning. Tests of the ensemble's own machinery (bag membership,
    out-of-bag evidence, the early-stopped tree count) need the ensemble itself.
    """
    est._full_tables = lambda model, *a, **k: model
    try:
        return est.fit(*args, **kwargs)
    finally:
        del est._full_tables
