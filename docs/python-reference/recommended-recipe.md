# recommended_recipe

Return an estimator configured with the benchmark recipe.

The constructors of `TBoostRegressor` and `TBoostClassifier` already default to this recipe
(early stopping with adaptive patience, leaf refinement, bagging and pruning), so a bare
estimator is the benchmarked configuration. This function is the explicit entry point: it sets
the tree limit and the number of threads, and for the log-link objectives it also sets the
categorical encoding explicitly (`cat_target="log_mean"` for `gamma`) and `reanchor=True`.

## Method call format {#call-format}

```python
recommended_recipe(objective="squared_error",
                   *,
                   budget=4000,
                   n_jobs=4,
                   tuned=True,
                   seed=0,
                   **overrides)
```

## Parameters {#parameters}

### objective

#### Description

The objective. `squared_error`, `poisson`, `gamma` and `tweedie` return a `TBoostRegressor`;
`logistic` returns a `TBoostClassifier` (softmax is used automatically for three or more
classes).

**Possible types**

string

**Default value**

squared_error

### budget

#### Description

The maximum number of trees ([`n_trees`](../training-parameters/common.md#n_trees)). Early
stopping decides the actual number.

**Possible types**

int

**Default value**

4000

### n_jobs

#### Description

The number of threads (see [`n_jobs`](../training-parameters/performance.md#n_jobs)).

**Possible types**

int

**Default value**

4

### tuned

#### Description

Apply the tuned part of the recipe: `n_bags=8`, `colsample_bytree=0.8` and, for the log-link
objectives, `reanchor=True`, `cat_target` (`"log_mean"` for `gamma`, `"mean"` otherwise),
`cat_leakage="kfold"` and `cat_k=5`.

**Possible types**

bool

**Default value**

True

### seed

#### Description

The random seed used for training.

**Possible types**

int

**Default value**

0

### **overrides

#### Description

Any [training parameter](../training-parameters/index.md). These take precedence over the
recipe.

**Possible types**

key=value format

**Default value**

None

## Return value {#output-format}

An untrained `TBoostRegressor` or `TBoostClassifier`.

## Usage examples {#usage-examples}

```python
from t_boost import recommended_recipe

model = recommended_recipe("poisson", n_jobs=8, prune_main_effects=True)
model.fit(train_data, "ClaimCount", exposure="Exposure")
```
