# Python package installation

!!! note

    t-boost Python package supports only [CPython Python implementation](https://en.wikipedia.org/wiki/CPython).

!!! warning

    Precompiled packages are only available for the 64-bit version of Python.

The package requires Python 3.10 or later.

Dependencies:

- `numpy (>=1.23)`
- `polars (>=1.0)`
- `threadpoolctl (>=3.0)`

!!! note

    Note that in most cases dependencies will be installed automatically using mechanisms built
    into `uv` or `pip`.

To install the Python package:

1. Choose an installation method:
    - [Install from PyPI](pip-install.md)
    - [Build from source](build-from-source.md)

1. (Optionally) Install [scikit-learn](https://scikit-learn.org/) to use the estimators with
   scikit-learn tools such as `clone`, pipelines and model selection.

    t-boost does not require scikit-learn: without it, `TBoostRegressor` and
    `TBoostClassifier` run standalone. scikit-learn is not listed in the package requirements
    that are installed automatically because it is not needed for other functionality.

1. (Optionally) [Test t-boost](test.md).
