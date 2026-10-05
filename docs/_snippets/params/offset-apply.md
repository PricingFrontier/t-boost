### offset

#### Description

The link-scale offset to add to the raw score of each object, as passed to the `offset`
parameter of `fit`. A string names a column of a polars `X`. If omitted, the objects are scored
with a zero offset.

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- list
- string

**Default value**

None
