# Categorical features

!!! warning

    Do not use one-hot encoding during preprocessing. Pass categorical columns as they are:
    t-boost encodes them natively, and the rating tables then show one relativity per level of
    the original feature instead of one table per dummy column.

t-boost supports numerical and categorical features.

Categorical features are transformed to numerical features before the trees are built: each
level is replaced by a smoothed target statistic, and the trees split on that statistic. See
the [Transforming categorical features to numerical features](../algorithm/categorical-features.md)
section for details.

## Declaring categorical features {#declaring}

For a polars `DataFrame` or `LazyFrame`, `String`, `Categorical` and `Enum` columns are
treated as categorical automatically. No declaration is needed.

Use the `categorical_features` parameter to treat other columns as categorical: a numeric
polars column that holds codes rather than quantities, or any column of a NumPy array or other
array-like input. The parameter accepts a single name or index, a list of names or indices,
or a boolean mask with one entry per feature.

```python
from t_boost import TBoostRegressor

# "VehicleGroup" is an integer code, so declare it explicitly
model = TBoostRegressor(objective="poisson", categorical_features=["VehicleGroup"])
```

## Levels in the rating tables {#levels}

Every categorical axis of an exported [rating table](../model-analysis/rating-tables.md) lists
its levels: each level's label together with the cell it lands in. Levels the model cannot
distinguish share one cell, and levels pooled into the rare level are listed as its
`members`.

Rare levels are pooled before encoding: a level whose total weight is below
`cat_min_data_per_group` (10 by default) joins a shared `"<rare>"` level. Missing values
(`null`, `None` or `NaN`) form their own level.

The `categories_` attribute lists the levels the fit saw for each categorical feature.

## Unseen values {#unseen-values}

A value that is absent from the training data is scored according to the `unknown_category`
parameter:

- `"rare"` (default) — The value is scored exactly as a level that was pooled into the
  `"<rare>"` level, in every table that uses the feature. If the fit pooled no levels, the
  value is scored in the axis's default cell.
- `"default_cell"` — The value is always scored in the axis's default cell (the encoder's base
  level).
- `"error"` — Every scoring call raises `ValueError`, naming the feature and up to five unseen
  values.

A missing value is never unseen: it is scored in the missing level.

Use the `unseen_values` method to count the unseen values in a dataset before scoring it.
