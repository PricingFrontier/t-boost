### offset

#### Description

A link-scale offset added to the raw score of each object and never learned (the equivalent of
XGBoost's `base_margin` and LightGBM's `init_score`). A string names a column of a polars `X`.

For the `squared_error` objective, the fit is exactly the fit of `y - offset`. For the log and
logit links it is added to the exposure offset, $\log(exposure) + offset$.

Prediction methods add it only when it is passed to them again (their `offset` parameter);
otherwise objects are scored with a zero offset.

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- list
- string

**Default value**

None
