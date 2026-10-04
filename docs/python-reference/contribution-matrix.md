# ContributionMatrix

```python
class ContributionMatrix
```

## Purpose {#purpose}

The result of [predict_contributions](tboostregressor/predict_contributions.md) with
`return_format="matrix"`: the contributions of every object as a dense `objects × terms`
matrix. This is the fastest format.

For every object, `base_value + values.sum(axis=-1)` equals the raw score on the link scale,
including any `exposure` or `offset` passed, which appear as columns of their own.

## Attributes {#attributes}

### base_value

The intercept for each object, on the link scale (`float64`). Shape `(object_count,)`, or
`(number_of_classes, object_count)` for a multiclassification model.

**Type:** numpy.ndarray

### values

The contributions (`float64`). Shape `(object_count, number_of_terms)`, or
`(number_of_classes, object_count, number_of_terms)` for a multiclassification model.

**Type:** numpy.ndarray

### terms

The feature names of each column, as a tuple: one name for a main effect (or, with
`split_interactions=True`, a feature) and several for an interaction. Because the names are
tuples, feature names may contain `:`. The `exposure` and `offset` columns carry the name of the
argument alone.

With `split_interactions=True` the columns are the features in input order, and a feature that no
table uses is a column of zeros. An intercept-only model has no columns.

**Type:** list of tuples of strings

### term_types

The kind of each column: `"main"`, `"interaction"`, `"feature"`, `"exposure"` or `"offset"`. A
feature that happens to be named `"exposure"` stays `"main"`.

**Type:** list of strings

### classes

The class labels along the first axis of a multiclassification result, else `None`.

**Type:** list or None

## Usage examples {#usage-examples}

```python
matrix = model.predict_contributions(test_data, return_format="matrix")
raw = matrix.base_value + matrix.values.sum(axis=-1)
for term, kind in zip(matrix.terms, matrix.term_types):
    print(":".join(term), kind)
```
