# Exceptions and warnings

All errors raised by t-boost derive from `TBoostError`:

```python
from t_boost import TBoostError, SerializationError
```

## Exceptions {#exceptions}

### TBoostError

The base class of all t-boost errors. Catching it catches every error below.

### Invalid input and configuration

Malformed input data (for example, a non-finite feature value or a target outside the domain of
the objective), arrays whose shapes disagree and parameter values outside their range raise an
exception that is both a `TBoostError` and a built-in `ValueError`. A buffer of the wrong
element type raises one that is both a `TBoostError` and a `TypeError`. So `except ValueError`
and `except TBoostError` both catch them.

The estimators also raise a plain `ValueError` when a parameter cannot be honoured for the task,
for example a regression-only parameter on a multiclassification model.

### SerializationError

Saving or loading a model failed: for example, a document written by a newer version of t-boost,
or `metadata` that is not JSON-serializable.

### ExactnessError

An operation that would break the exact decomposition of the model into rating tables was
attempted.

### InvariantError

One of the internal checks of the exact decomposition failed. This indicates a bug: please
report it.

### InternalError

An internal error. This indicates a bug: please report it.

## Warnings {#warnings}

### PrecisionWarning

t-boost computes in `float32`. When a numeric feature is converted to `float32` before training
or scoring, a `PrecisionWarning` (a subclass of `UserWarning`) is issued once per estimator. Cast
the columns to `float32` to skip the conversion, or filter the warning:

```python
import warnings
from t_boost import PrecisionWarning

warnings.filterwarnings("ignore", category=PrecisionWarning)
```
