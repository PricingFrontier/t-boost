"""Shared test fixtures."""
from __future__ import annotations

from collections.abc import Iterator

import pytest

from t_boost.sklearn import _ctor_defaults_cached


@pytest.fixture(autouse=True)
def _fresh_ctor_defaults() -> Iterator[None]:
    # Several modules patch the estimators' constructor defaults for their own tests; the
    # cached defaults (`binding_report_`, deprecation warnings) must not outlive the patch.
    _ctor_defaults_cached.cache_clear()
    yield
    _ctor_defaults_cached.cache_clear()
