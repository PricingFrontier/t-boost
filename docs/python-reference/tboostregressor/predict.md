# predict

Apply the model to the given dataset.

--8<-- "_snippets/feature-matching-note.md"

## Method call format {#call-format}

```python
predict(X, *, offset=None)
```

## Parameters {#parameters}

--8<-- "_snippets/params/X-apply.md"

--8<-- "_snippets/params/offset-apply.md"

## Return value {#output-format}

A one-dimensional `numpy.ndarray` of shape `(object_count,)` with the prediction for each object,
on the scale of the target.

For the `poisson`, `gamma` and `tweedie` objectives, the prediction is the rate per unit of
exposure, not the total for the object: the expected total is `predict(X) * exposure`. An exposure
column in `X` is ignored.

The values are `float64`. The model calculates in `float32`, and the wider type avoids
accumulating rounding errors in downstream metrics.

## Usage examples {#usage-examples}

```python
rate = model.predict(test_data)
expected_claims = rate * test_data["Exposure"].to_numpy()
```
