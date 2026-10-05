# Missing values processing

The missing values processing mode depends on the feature type.

## Numerical features {#numerical-features}

t-boost interprets the value of a numerical feature as a missing value if it is equal to one of
the following values:

- `null` in a polars column, or `None`
- [Floating point NaN value](https://en.wikipedia.org/wiki/NaN)

Positive and negative infinity are not missing values: they fall into the highest and the
lowest bin of the feature.

Missing values are put into a reserved bin of their own. When a split on the feature is
selected, the missing values are tried on both sides of the split, and the side with the
better score is kept. In the [rating tables](../model-analysis/rating-tables.md), every numeric
axis has a cell for missing values (cell 0), so missing values get a relativity of their own.

## Categorical features {#categorical-features}

Missing values (`null`, `None` or `NaN`) of a categorical feature are collected into one
reserved level, which is [encoded](categorical-features.md) like any other level. A missing
value is never treated as an unseen value. The `categories_` attribute lists the missing level
as `None` when the fit saw missing values.
