# score

Calculate the R2 metric for the objects in the given dataset.

The metric compares `y` with [predict](predict.md), which for a model with a log link is the rate
per unit of exposure. For a model trained with an exposure, use the deviances of the
[t_boost.metrics](../metrics.md) module instead.

## Method call format {#call-format}

```python
score(X, y, sample_weight=None)
```

## Parameters {#parameters}

--8<-- "_snippets/params/X-apply.md"

### y

#### Description

The target values of the objects.

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- list

**Default value**

Required parameter

### sample_weight

#### Description

The weight of each object. By default, it is set to 1 for all objects.

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- list

**Default value**

None

## Return value {#output-format}

float
