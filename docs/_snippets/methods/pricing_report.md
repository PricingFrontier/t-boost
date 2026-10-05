# pricing_report

Return the rating tables, the actual versus expected cells and the diagnostics of the pruning
and graduation stages, in one document for review.

The report describes the data passed to it; it is not evidence that the data were held out.
Save it beside the serialized model. Regression and binary classification only.

--8<-- "_snippets/feature-matching-note.md"

## Method call format {#call-format}

```python
pricing_report(X, y, *, sample_weight=None, exposure=None, ref_measure=None)
```

## Parameters {#parameters}

--8<-- "_snippets/params/X-apply.md"

### y

#### Description

The target values of the objects. A string names a column of a polars `X`.

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- list
- string

**Default value**

Required parameter

### sample_weight, exposure

#### Description

The weight and the exposure of each object, passed explicitly as for
[actual_vs_expected](actual_vs_expected.md).

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- list
- string

**Default value**

None

### ref_measure

#### Description

The reference measure of the exported tables, as for [tables](tables.md).

**Possible types**

string

**Default value**

None

## Return value {#output-format}

A dictionary:

- `report_version` — The version of the report format.
- `prediction_units`, `reference_normalization`, `band_interpretation` — How to read the
  numbers.
- `tables` — The rating tables, as returned by [tables](tables.md).
- `actual_vs_expected` — The cells returned by [actual_vs_expected](actual_vs_expected.md), each
  feature with its axis.
- `pruning` — The `pruning_report_` attribute.
- `graduation` — The diagnostics of graduation.

## Usage examples {#usage-examples}

```python
import json

# model: a TBoostRegressor(objective="poisson") trained with exposure="Exposure"
report = model.pricing_report(train_data, "ClaimCount", exposure="Exposure")
with open("freq_report.json", "w") as f:
    json.dump(report, f)
```
