# Training

t-boost provides two classes for training a model. Both accept polars `DataFrame` and
`LazyFrame` input directly, with the target and other per-object values named by column.

## Classes {#classes}

### [TBoostRegressor](../python-reference/tboostregressor/index.md)

**Class purpose**

Training and applying regression models: `squared_error`, `poisson`, `gamma` and `tweedie`.

**Method**

[fit](../python-reference/tboostregressor/fit.md)

### [TBoostClassifier](../python-reference/tboostclassifier/index.md)

**Class purpose**

Training and applying classification models: logistic for two classes, softmax for three or
more.

**Method**

[fit](../python-reference/tboostclassifier/fit.md)

## Usage examples {#usage-examples}

```python
from t_boost import TBoostRegressor

model = TBoostRegressor(objective="poisson")
model.fit(train_data, "ClaimCount", exposure="Exposure")
```

The default parameters already include early stopping, bagging, pruning, banding and
graduation, so a model is usually trained without tuning. See
[Training parameters](../training-parameters/index.md) for the full list and
[Parameter tuning](../parameter-tuning.md) for tips.

A trained model is deterministic: the same data and the same `seed` give a bit-identical model,
whatever the number of threads.
