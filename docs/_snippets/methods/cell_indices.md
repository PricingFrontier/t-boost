# cell_indices

Return the rating-table cell every object falls in, for every deployed dense table.

The cells follow the rules described in
[How an object finds its cell](../../model-analysis/rating-tables.md#cells): numeric values are
compared in `float32`, cell 0 holds missing values, and a value equal to a border lands in the
lower cell.

--8<-- "_snippets/feature-matching-note.md"

## Method call format {#call-format}

```python
cell_indices(X)
```

## Parameters {#parameters}

--8<-- "_snippets/params/X-apply.md"

## Return value {#output-format}

A dictionary with one entry per dense table, keyed by the tuple of the table's feature names
(the same keys as the `terms` of [ContributionMatrix](../contribution-matrix.md) and the
`feature_names` of [tables](tables.md)). Each value is a `numpy.uint32` array of shape
`(object_count, order)` with the object's cell on each axis of the table, in the table's axis
order. Effects stored in factored form have no cells and are not listed.

For an exported table, `values[numpy.ravel_multi_index(tuple(cells.T), shape)]` is the
contribution of the table to each object.

`ValueError` is raised for a multiclassification model (which has one set of tables per class)
and for a model saved by t-boost 0.6 or earlier.

## Usage examples {#usage-examples}

```python
import json
import numpy as np

cells = model.cell_indices(test_data)
tables = {tuple(t["feature_names"]): t for t in json.loads(model.tables(train_data))["tables"]}
table = tables[("DriverAge",)]
values = np.asarray(table["values"])[np.ravel_multi_index(tuple(cells[("DriverAge",)].T), table["shape"])]
```
