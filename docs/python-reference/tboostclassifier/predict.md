# predict

Apply the model to the given dataset to predict the class labels.

--8<-- "_snippets/feature-matching-note.md"

## Method call format {#call-format}

```python
predict(X, *, offset=None)
```

## Parameters {#parameters}

--8<-- "_snippets/params/X-apply.md"

--8<-- "_snippets/params/offset-apply.md"

## Return value {#output-format}

A one-dimensional `numpy.ndarray` of shape `(object_count,)` with labels from `classes_`. For
binary classification, the label is `classes_[1]` where `predict_proba(X)[:, 1] >= 0.5` and
`classes_[0]` otherwise. For multiclassification, it is the class with the highest probability.
