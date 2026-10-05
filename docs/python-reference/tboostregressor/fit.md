# fit

Train a model.

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

The training parameters are set in the constructor of the [TBoostRegressor](index.md) class.

--8<-- "_snippets/params/X-fit.md"

### y

#### Description

The target variables (in other words, the objects' label values) for the training dataset, on
the natural scale (not the link scale). A string names a column of a polars `X`, which is then
not used as a feature.

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

The exposure of each object (for example, the duration of a policy), for the `poisson`, `gamma`
and `tweedie` objectives. The predicted rate is multiplied by the exposure before it is compared
with the target: $\log(exposure)$ is added to the raw score as an offset. A string names a column
of a polars `X`.

By default, it is set to 1 for all objects. Not supported for the `squared_error` objective.

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

The estimator itself, trained.

The attributes `n_trees_per_bag_`, `stopping_reason_per_bag_`, `n_trees_`, `stopping_reason_` and
`evals_result_` describe the boosting (see [Attributes](attributes.md)).

## Usage examples {#usage-examples}

```python
from t_boost import TBoostRegressor

model = TBoostRegressor(objective="poisson")
model.fit(train_data, "ClaimCount", exposure="Exposure",
          eval_set=(valid_data, "ClaimCount"), eval_exposure="Exposure")
print(model.n_trees_per_bag_, model.stopping_reason_)
```
