# actual_vs_expected

Calculate actual versus expected totals by rating-factor level, for every feature.

This shows where observed and predicted totals differ. For each feature, the objects are
aggregated over the cells of the grid the deployed tables use (cell 0 holds missing values).

--8<-- "_snippets/feature-matching-note.md"

## Method call format {#call-format}

```python
actual_vs_expected(X, y, *, sample_weight=None, exposure=None, offset=None)
```

## Parameters {#parameters}

--8<-- "_snippets/params/X-apply.md"

### y

#### Description

The target values of the objects (for a classifier, the class labels). A string names a column
of a polars `X`.

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- list
- string

**Default value**

Required parameter

### sample_weight, exposure

#### Description

The weight and the exposure of each object. A string names a column of a polars `X`.

If the model was trained with a sample weight or an exposure, the corresponding value must be
passed explicitly, even on the training data: the training values are never reused, because a
matching number of objects does not prove the objects are the same. Pass a vector of ones for an
unweighted or unit-exposure evaluation.

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- list
- string

**Default value**

None

### offset

#### Description

The link-scale offset of each object, as passed to the `offset` parameter of `fit`. A string
names a column of a polars `X`.

If the model was trained with an offset, it must be passed explicitly, as for `sample_weight`
and `exposure`: without it the expected values would be computed on a zero offset. Pass a
vector of zeros for an offset-free evaluation.

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- list
- string

**Default value**

None

## Return value {#output-format}

A list with one dictionary per feature, holding one value per cell of the feature's grid,
empty cells included:

- `feature`, `raw` — The feature name and its zero-based index.
- `actual` — $\sum w \cdot t$.
- `expected` — $\sum w \cdot prediction$, where the prediction includes the offset and, for a
  model with a log link, the exposure.
- `mass` — $\sum w \cdot exposure$.
- `rows` — The number of objects.
- `ae` — `actual / expected` (`nan` where `expected` is 0).

Exact balance by factor level is not a general property of boosted models, so the ratios show
where the model and the data disagree.

A multiclassification model has one grid per class: the list holds one entry per class and
feature, with a `class` key, one-vs-rest — `actual` $= \sum w \cdot [t = class]$ and
`expected` $= \sum w \cdot P(class)$. It accepts no exposure or offset.

## Usage examples {#usage-examples}

```python
# model: a TBoostRegressor(objective="poisson") trained with exposure="Exposure"
ae = model.actual_vs_expected(train_data, "ClaimCount", exposure="Exposure")
for factor in ae:
    print(factor["feature"], factor["ae"])
```
