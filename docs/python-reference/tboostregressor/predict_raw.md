# predict_raw

Apply the model to the given dataset and return the raw score: the prediction on the link scale,
before the inverse of the link function is applied.

For the `squared_error` objective this is the same as [predict](predict.md). For the `poisson`,
`gamma` and `tweedie` objectives it is the logarithm of the rate.

--8<-- "_snippets/feature-matching-note.md"

## Method call format {#call-format}

```python
predict_raw(X, *, offset=None)
```

## Parameters {#parameters}

--8<-- "_snippets/params/X-apply.md"

--8<-- "_snippets/params/offset-apply.md"

## Return value {#output-format}

A one-dimensional `numpy.ndarray` of shape `(object_count,)` with the raw score of each object, as
`float64`.
