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
