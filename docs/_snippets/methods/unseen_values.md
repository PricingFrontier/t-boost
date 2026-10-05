# unseen_values

Count the categorical values in the dataset that are absent from the training data.

These are the values the [`unknown_category`](../../training-parameters/categorical.md#unknown_category)
parameter acts on. A missing value is never unseen.

## Method call format {#call-format}

```python
unseen_values(X, *, sample_weight=None, exposure=None)
```

## Parameters {#parameters}

--8<-- "_snippets/params/X-apply.md"

### sample_weight, exposure

#### Description

The weight and the exposure of each object. When either is given, the result has a `mass`
column. A string names a column of a polars `X`.

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- list
- string

**Default value**

None

## Return value {#output-format}

A polars DataFrame with one row per `(feature, value)`, sorted by feature (in input order) and
then by the number of objects, descending:

- `feature` — The feature name.
- `value` — The unseen value, as the string t-boost matches on (a numeric category `1` is
  `"1"`).
- `rows` — The number of objects with the value.
- `mass` — Their total `sample_weight * exposure` (only when one of them is given).

## Usage examples {#usage-examples}

```python
print(model.unseen_values(new_business))
```
