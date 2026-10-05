### eval_set

#### Description

The validation dataset used by [early stopping](../../algorithm/early-stopping.md), as a tuple
`(X_val, y_val)`. Every bag stops at its own best iteration on it, and no validation
objects are set aside from `X` (`validation_fraction` is ignored).

The evaluation objects are used only by early stopping: they never reach the training,
the intercept, the cell refit, the tables or pruning. `X_val` is matched to the features as in
`predict`, and `y_val` may name a column of a polars `X_val`. Setting `reanchor_slope=True` with
an `eval_set` raises an error.

**Possible types**

tuple `(X_val, y_val)`

**Default value**

None

### eval_sample_weight, eval_exposure, eval_offset

#### Description

The weight, the exposure and the offset of the evaluation objects. A string names a column of
`X_val`. `eval_exposure` is required when `exposure` is given, and `eval_offset` when `offset` is
given.

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- list
- string

**Default value**

None
