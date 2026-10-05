### sample_weight

#### Description

The weight of each object in the input data in the form of a one-dimensional array-like data.
A string names a column of a polars `X`.

By default, it is set to 1 for all objects.

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- list
- string

**Default value**

None
