# Quick start

Use one of the following examples after [installing](installation/index.md) the Python package
to get started:

- [TBoostRegressor](#regression)
- [TBoostClassifier](#classification)
- [Rating tables](#rating-tables)

## TBoostRegressor {#regression}

```python
import numpy as np
import polars as pl
from t_boost import TBoostRegressor

# initialize data
rng = np.random.default_rng(0)
n = 10_000
data = pl.DataFrame({
    "DriverAge": rng.integers(18, 80, n).astype(np.float32),
    "VehicleAge": rng.integers(0, 20, n).astype(np.float32),
    "Region": rng.choice(["North", "South", "East", "West"], n),
    "Exposure": rng.uniform(0.1, 1.0, n),
})
rate = np.exp(-2.0 + 0.5 * (data["DriverAge"].to_numpy() < 25))
data = data.with_columns(ClaimCount=rng.poisson(rate * data["Exposure"].to_numpy()))
train_data, test_data = data[:8_000], data[8_000:]

# specify the training parameters
model = TBoostRegressor(objective="poisson")
# train the model
model.fit(train_data, "ClaimCount", exposure="Exposure")
# make the prediction using the resulting model
preds = model.predict(test_data)
print(preds)
```

The target and the exposure are given as column names of the training frame, and those
columns are not used as features. `Region` is a `String` column, so it is treated as a
[categorical feature](features/categorical-features.md) automatically. At prediction time,
columns are matched by name and extra columns are ignored.

For a Poisson, Gamma or Tweedie model `predict` returns the rate per unit of exposure. The
expected number of claims for a row is `predict(X) * exposure`.

## TBoostClassifier {#classification}

```python
import numpy as np
import polars as pl
from t_boost import TBoostClassifier

# initialize data
rng = np.random.default_rng(0)
train_data = pl.DataFrame({
    "Age": rng.integers(18, 80, 1000).astype(np.float32),
    "Channel": rng.choice(["Broker", "Direct", "Online"], 1000),
    "Lapsed": rng.integers(0, 2, 1000),
})
test_data = train_data.drop("Lapsed").head(5)

model = TBoostClassifier()
# train the model
model.fit(train_data, "Lapsed")
# make the prediction using the resulting model
preds_class = model.predict(test_data)
preds_proba = model.predict_proba(test_data)
print("class = ", preds_class)
print("proba = ", preds_proba)
```

A target with two classes trains a logistic model. A target with three or more classes trains
a softmax model, with one set of rating tables per class.

## Rating tables {#rating-tables}

The fitted model is a set of rating tables. Continue the [TBoostRegressor](#regression)
example to look at them:

```python
import json

tables = json.loads(model.tables(train_data))     # the rating tables
for table in tables["tables"]:
    print(table["feature_names"], table["shape"])

contributions = model.predict_contributions(test_data.head(5))  # per-prediction breakdown
importances = model.feature_importances_          # share of variance per feature
```

For each prediction, the base value plus the sum of the contributions equals the raw score
on the link scale, so the explanation is exact rather than estimated. See
[Model analysis](model-analysis/rating-tables.md) for details.

!!! note

    t-boost computes in 32-bit floating point. Numeric feature columns are converted to
    `float32` before fitting and scoring, and `fit` issues a `PrecisionWarning` once per
    estimator if any numeric feature column has another dtype. Cast the columns to
    `pl.Float32` to skip the conversion.
