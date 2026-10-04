# tables

Export the exact rating tables of the model as a JSON document.

The tables are the model: for every object, the intercept plus the values of the cells it falls
in exactly reproduce the raw score, with no approximation and no surrogate model. See
[Rating tables](../../model-analysis/rating-tables.md) for the format of the document and how an
object finds its cell.

--8<-- "_snippets/feature-matching-note.md"

## Method call format {#call-format}

```python
tables(X,
       ref_measure=None,
       laplace=1.0,
       basis_json=None,
       measure_floor=None,
       overflow=None,
       sample_weight=None,
       exposure=None)
```

## Parameters {#parameters}

### X

#### Description

Feature values data. The objects determine which cells are reported, and with `sample_weight`
or `exposure` they are the objects the tables are re-centred on.

**Possible types**

- polars.DataFrame
- polars.LazyFrame
- numpy.ndarray of shape `(object_count, feature_count)`
- other array-like data of the same shape

**Default value**

Required parameter

### ref_measure

#### Description

The reference measure the tables are purified against.

`None` exports the tables as they are stored, purified at the fit under the
[`ref_measure`](../../training-parameters/purification.md#ref_measure) the model was trained with.
Any other value re-expresses the same model under that measure: the sum of the tables in every
cell, and so every prediction, is unchanged, and only the split between the tables moves.

Possible values:

- `exposure` — Each axis is weighted by the exposure-weighted distribution of its feature, plus
  a positivity floor ([`measure_floor`](#measure_floor)).
- `product_marginals` — The number of objects in each cell, blended with a uniform weighting
  ([`laplace`](#laplace)).
- `uniform` — Every cell has the same weight.
- `joint` — Re-expresses the main effects and the pairs using the joint distribution of each
  pair of features (a regularized reallocation). The total score is preserved, but the tables
  are no longer exactly purified, their variance shares no longer add up to one and the equal
  split of interactions is no longer a Shapley value. Higher-order tables stay purified against
  the product measure. Not supported for banded tables: the model must be trained with
  `band_tolerance=None`.

**Possible types**

string

**Default value**

None (the measure the model was trained with)

### laplace

#### Description

The smoothing constant of the `product_marginals` reference measure. Unused for the other
measures.

**Possible types**

float

**Default value**

1.0

### basis_json

#### Description

A JSON-encoded custom grid to export the tables on, instead of the model's own bin borders.

**Possible types**

string

**Default value**

None (the model's own grid)

### measure_floor

#### Description

The positivity floor of the `exposure` reference measure. `None` uses the value the model was
trained with.

**Possible types**

float

**Default value**

None

### overflow

#### Description

What to do with an effect whose dense table would exceed the cell budget.

- `factored` — Keep it exactly as a sum of rank-one boxes, under the `factored` key of the
  document.
- `sparse` — Store it as an exact sparse table.
- `error` — Raise an error.

All three keep the decomposition exact.

**Possible types**

string

**Default value**

None (`factored`)

### sample_weight, exposure

#### Description

The weight and the exposure of each object of `X`. A string names a column of a polars `X`.

When neither is given, the tables are exported as they are stored, purified on the training
data at the fit. When either is given, the tables are re-centred on the objects of `X` under the
mass $sample\_weight \cdot exposure$, which also changes the `support`, the variances and the
importances. Predictions never change either way.

When `None`, each falls back to the value the model was trained with, if any. That fallback only
makes sense when `X` is the training data: pass explicit values (for example `numpy.ones`)
whenever `X` holds other objects.

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- list
- string

**Default value**

None

## Return value {#output-format}

A JSON string. For a regression or binary classification model, one document with the intercept
(`f0`), the dense tables (`tables`) and the factored effects (`factored`). For a
multiclassification model, an object keyed by class label, with one such document per class.

## Usage examples {#usage-examples}

```python
import json

tables = json.loads(model.tables(train_data))
for table in tables["tables"]:
    print(table["feature_names"], table["shape"], table["sobol"])
```
