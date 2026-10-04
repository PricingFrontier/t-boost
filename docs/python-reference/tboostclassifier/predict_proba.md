# predict_proba

Apply the model to the given dataset to predict the probability that the object belongs to the
given classes.

--8<-- "_snippets/feature-matching-note.md"

## Method call format {#call-format}

```python
predict_proba(X, *, offset=None)
```

## Parameters {#parameters}

--8<-- "_snippets/params/X-apply.md"

--8<-- "_snippets/params/offset-apply.md"

## Return value {#output-format}

A two-dimensional `numpy.ndarray` of shape `(number_of_objects, number_of_classes)` with the
probability for every class for each object. Column $j$ is the probability of `classes_[j]`.

The values are `float64`. The model calculates in `float32`, and the wider type avoids
accumulating rounding errors in downstream metrics such as log loss and AUC.

## Usage examples {#usage-examples}

```python
proba = model.predict_proba(test_data)
p_lapse = proba[:, 1]
```
