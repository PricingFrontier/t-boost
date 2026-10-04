# score

Calculate the Accuracy metric for the objects in the given dataset.

## Method call format {#call-format}

```python
score(X, y, sample_weight=None)
```

## Parameters {#parameters}

--8<-- "_snippets/params/X-apply.md"

### y

#### Description

The class labels of the objects.

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
