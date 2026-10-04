# get_params

Return the values of all training parameters, as passed to the constructor or to
[set_params](set_params.md).

## Method call format {#call-format}

```python
get_params(deep=True)
```

## Parameters {#parameters}

### deep

#### Description

Kept for compatibility with the scikit-learn estimator API. The estimators have no nested
estimators, so the value has no effect.

**Possible types**

bool

**Default value**

True

## Return value {#output-format}

A dictionary with one entry per parameter of the constructor (see
[Training parameters](../../training-parameters/index.md)).
