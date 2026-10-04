# metrics

The `t_boost.metrics` module calculates metrics separately from the training. It needs only
NumPy: no scikit-learn.

```python
from t_boost.metrics import mean_poisson_deviance, ordered_gini
```

All functions take the targets `y`, the predictions and optional weights as one-dimensional
array-like data.

## Deviances {#deviances}

The deviances are the standard GLM unit deviances, numerically identical to scikit-learn's
functions of the same names, and the natural goodness-of-fit for the matching
[objective](../objectives/regression.md). Lower is better. A value outside the domain of the
distribution (for example, a negative prediction) raises `ValueError`.

For a model trained with an exposure, pass the expected totals, `predict(X) * exposure`, as the
predictions.

### mean_tweedie_deviance {#mean_tweedie_deviance}

```python
mean_tweedie_deviance(y, pred, weight=None, power=0.0)
```

The weighted mean Tweedie unit deviance with variance power `power`: 0 is the squared error, 1
the Poisson deviance, 2 the Gamma deviance, and a value between 1 and 2 the compound
Poisson-Gamma deviance. Powers between 0 and 1 are not supported.

**Return value:** float

### mean_poisson_deviance {#mean_poisson_deviance}

```python
mean_poisson_deviance(y, pred, weight=None)
```

`mean_tweedie_deviance` with `power=1`: the goodness-of-fit of `objective="poisson"` frequency
models.

**Return value:** float

### mean_gamma_deviance {#mean_gamma_deviance}

```python
mean_gamma_deviance(y, pred, weight=None)
```

`mean_tweedie_deviance` with `power=2`: the goodness-of-fit of `objective="gamma"` severity
models.

**Return value:** float

## Ranking metrics {#ranking-metrics}

The ranking metrics measure how well the predictions order the objects. They are weight-aware,
and degenerate or non-finite inputs give 0 rather than an error.

### ordered_gini {#ordered_gini}

```python
ordered_gini(y, pred, weight=None)
```

The concentration Gini of `y` when the objects are ranked by `pred`, normalized by the Gini of
the perfect ranking (by `y` itself). 1 is a perfect ranking.

**Return value:** float

### concentration_gini {#concentration_gini}

```python
concentration_gini(y, score, weight=None)
```

The concentration (Lorenz) Gini of `y` when the objects are ranked by `score`, descending: the
weighted cumulative share of `y` against the weighted cumulative share of objects, as
$2 \cdot area - 1$. `y` and the weights are clamped at 0.

**Return value:** float

### lift_curve {#lift_curve}

```python
lift_curve(y, pred, weight=None, buckets=10)
```

The objects are ranked by `pred`, descending, and split into `buckets` groups with equal
numbers of objects.

**Return value:** a list with one dictionary per group: `bucket` (starting at 1), `rows`,
`mean_y` and `mean_pred` (weighted means) and `lift` (`mean_y` divided by the overall weighted
mean of `y`). An empty list for degenerate input.

### top_bucket_lift {#top_bucket_lift}

```python
top_bucket_lift(y, pred, weight=None, buckets=10)
```

The lift of the first group of `lift_curve` (the highest predictions). 0 if the curve is empty.

**Return value:** float

## Usage examples {#usage-examples}

```python
from t_boost.metrics import mean_poisson_deviance, ordered_gini

expected = model.predict(test_data) * test_data["Exposure"].to_numpy()
claims = test_data["ClaimCount"].to_numpy()
print(mean_poisson_deviance(claims, expected))
print(ordered_gini(claims, expected))
```
