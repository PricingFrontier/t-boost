# Regular prediction

t-boost provides the following methods for applying a trained model. For a polars input, the
features are matched by name: extra columns are ignored and the order of the columns does not
matter.

## Classes {#classes}

### [TBoostRegressor](../python-reference/tboostregressor/index.md)

**Method** [predict](../python-reference/tboostregressor/predict.md)

**Description**

Apply the model to the given dataset. For the `poisson`, `gamma` and `tweedie` objectives the
result is the rate per unit of exposure.

**Method** [predict_raw](../python-reference/tboostregressor/predict_raw.md)

**Description**

Apply the model to the given dataset and return the raw score on the link scale.

### [TBoostClassifier](../python-reference/tboostclassifier/index.md)

**Method** [predict](../python-reference/tboostclassifier/predict.md)

**Description**

Apply the model to the given dataset to predict the class labels.

**Method** [predict_proba](../python-reference/tboostclassifier/predict_proba.md)

**Description**

Apply the model to the given dataset to predict the probability that the object belongs to the
given classes.

**Method** [decision_function](../python-reference/tboostclassifier/decision_function.md)

**Description**

Apply the model to the given dataset and return the raw score on the logit scale.

## Usage examples {#usage-examples}

```python
rate = model.predict(test_data)
expected_claims = rate * test_data["Exposure"].to_numpy()
```

Use the `required_columns` attribute to read only the columns the model needs:

```python
import polars as pl

data = pl.scan_parquet("policies.parquet").select(model.required_columns).collect()
preds = model.predict(data)
```
