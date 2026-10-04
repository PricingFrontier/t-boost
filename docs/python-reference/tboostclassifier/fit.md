# fit

Train a model.

The number of distinct classes in `y` decides the model: two classes train a logistic model, and
three or more train a softmax model with one set of rating tables per class.

## Method call format {#call-format}

```python
fit(X,
    y,
    sample_weight=None,
    exposure=None,
    groups=None,
    *,
    offset=None,
    eval_set=None,
    eval_sample_weight=None,
    eval_exposure=None,
    eval_offset=None,
    callbacks=None)
```

## Parameters {#parameters}

The training parameters are set in the constructor of the [TBoostClassifier](index.md) class.

--8<-- "_snippets/params/X-fit.md"

### y

#### Description

The class labels of the training objects: at least two distinct values, not a continuous
target. The labels can be of any type; they are sorted into `classes_`. A string names a column
of a polars `X`, which is then not used as a feature.

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- list
- string

**Default value**

Required parameter

--8<-- "_snippets/params/sample_weight-fit.md"

### exposure

#### Description

The exposure of each object. $\log(exposure)$ is added to the raw score (the logit) as an offset.
A string names a column of a polars `X`.

Supported for binary classification only: passing it with three or more classes raises an
error.

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- list
- string

**Default value**

None

--8<-- "_snippets/params/groups-fit.md"

--8<-- "_snippets/params/offset-fit.md"

--8<-- "_snippets/params/eval-fit.md"

--8<-- "_snippets/params/callbacks-fit.md"

## Return value {#output-format}

The estimator itself, trained, with `classes_` set to the sorted distinct labels found in `y`.

For binary classification, the attributes `n_trees_per_bag_`, `stopping_reason_per_bag_`,
`n_trees_`, `stopping_reason_` and `evals_result_` describe the boosting (see
[Attributes](attributes.md)).

## Usage examples {#usage-examples}

```python
from t_boost import TBoostClassifier

model = TBoostClassifier()
model.fit(train_data, "Lapsed")
print(model.classes_)
```
