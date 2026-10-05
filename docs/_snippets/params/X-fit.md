### X

#### Description

The input training dataset in the form of a two-dimensional feature matrix.

The description is different for each group of possible types.

**Possible types**

??? info "polars.DataFrame, polars.LazyFrame"

    The column names become the feature names (`feature_names_in_`). `String`, `Categorical`
    and `Enum` columns are treated as [categorical features](../../features/categorical-features.md),
    and numeric columns are converted to `float32`, with nulls treated as missing values. A
    `LazyFrame` is collected at the start of the fit.

    The columns named by the `y`, `sample_weight`, `exposure`, `groups` and `offset` parameters
    are not used as features.

??? info "numpy.ndarray, other array-like data"

    Every feature is numerical unless it is listed in the
    [`categorical_features`](../../training-parameters/categorical.md#categorical_features)
    parameter. For a pandas `DataFrame` the column names become the feature names, but at
    prediction time the features are matched by position. A Fortran-ordered (column-major)
    `float32` array is the cheapest input: its columns are copied without a transpose.

**Default value**

Required parameter
