# decision_function

Apply the model to the given dataset and return the raw score: the prediction on the link
(logit) scale, before the sigmoid or the softmax is applied.

--8<-- "_snippets/feature-matching-note.md"

## Method call format {#call-format}

```python
decision_function(X, *, offset=None)
```

## Parameters {#parameters}

--8<-- "_snippets/params/X-apply.md"

--8<-- "_snippets/params/offset-apply.md"

## Return value {#output-format}

As `float64`:

- Binary classification — A one-dimensional `numpy.ndarray` of shape `(object_count,)` with the
  logit of the probability of `classes_[1]`.
- Multiclassification — A two-dimensional `numpy.ndarray` of shape
  `(object_count, number_of_classes)` with the raw score of each class. The model is a joint
  softmax, not independent one-vs-rest models.
