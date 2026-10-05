# Rating tables

The fitted model is a set of rating tables: an intercept and one table per main effect and per
interaction. For every object, the raw score (the prediction on the link scale) is exactly the
intercept plus the values of the table cells the object falls in:

$$
a(x) = f_0 + \sum\limits_{u \in U} f_{u}(x_{u}) { , where}
$$

- $f_0$ is the intercept.
- $U$ is the set of deployed effects. Each effect $u$ is a set of features: one feature for a
  main effect, two or more for an interaction.
- $f_u(x_u)$ is the value of the cell of table $u$ that the object's values of the features in
  $u$ fall in.

There is no approximation and no surrogate model: the tables *are* the model. A saved model is
stored as its rating tables, so what you review is exactly what gets deployed.

## Export the tables {#export}

The `tables` method returns the tables as a JSON document:

```python
import json

tables = json.loads(model.tables(train_data))
print(tables["f0"])                       # the intercept
for table in tables["tables"]:
    print(table["feature_names"], table["shape"])
```

The main keys of the document:

`f0`
:   The intercept.

`link`, `objective`
:   The link function and the objective the model was trained with.

`reference_measure`
:   The reference measure the tables are purified against (see
    [Purification](#purification)).

`tables`
:   The dense tables, one entry per effect.

`factored`
:   Effects whose dense grid would be too large to store. They are kept exactly, as sums of
    rank-one boxes, instead of as dense tables.

Each entry of `tables` contains:

`feature_names`, `feature_set`
:   The names and the zero-based indices of the features of the effect.

`axes`
:   One entry per dimension of the table: the feature, the cell borders and, for a
    categorical feature, its levels.

`shape`
:   The number of cells along each axis.

`values`
:   The table values on the link scale, flattened in row-major (C) order over `shape`.

`relativities`
:   `exp(values)` for a model with a log link (`poisson`, `gamma`, `tweedie`), the
    multiplicative relativities. `null` for other links.

`support`
:   The weight of the training objects in each cell: their number, or the sum of sample weight
    times exposure when either was given.

`variance`, `sobol`
:   The variance of the table under the reference measure, and its share of the model's
    variance.

## How an object finds its cell {#cells}

- A numeric axis with borders $b_0 < b_1 < \ldots < b_{m-1}$ has $m + 2$ cells. Cell 0 holds
  missing values. A value $v$ lands in cell $1 + \#\{i : b_i < v\}$. Cells are right-closed, so a
  value equal to a border lands in the lower cell.
- Values are converted to `float32` before the comparison, and the borders are exported as
  exactly the `float32` values compared against.
- A categorical axis maps each level to a cell through its `levels` list. A value the fit
  never saw is scored in the axis's `unseen_cell` (see
  [Unseen values](../features/categorical-features.md#unseen-values)).
- A table that was banded lists its band edges as borders, which is why two tables on the same
  feature may list different borders.

The `cell_indices` method returns, for every object and every dense table, the cell the object
falls in.

## Purification {#purification}

An interaction table could absorb part of a main effect, and a main effect part of the
intercept, without changing any prediction. Purification removes that freedom: each table is
centred so that its weighted mean along every axis is zero under a *reference measure*. As a
result each main effect carries as much of the signal as it can, and each interaction carries
only what the main effects cannot express.

By default the reference measure is the exposure-weighted distribution of the training data
(`ref_measure="exposure"`). The `ref_measure` argument of `tables` re-expresses the same model
under another measure. The sum of the tables in every cell, and so every prediction, is
unchanged: only the split between the tables moves.
