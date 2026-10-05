# Feature importances

The importance of each input feature is its share of the model's variance: each table's share is
split equally among the features it involves. The importances are read from the model, need no
data, and sum to 1. See [Feature importance](../model-analysis/feature-importance.md) for the
calculation principles.

## Python package {#python-package}

**Attribute**

- [feature_importances_](../python-reference/tboostregressor/attributes.md#feature_importances_)
  ([TBoostRegressor](../python-reference/tboostregressor/index.md))
- [feature_importances_](../python-reference/tboostclassifier/attributes.md#feature_importances_)
  ([TBoostClassifier](../python-reference/tboostclassifier/index.md))

## Usage examples {#usage-examples}

```python
for name, importance in zip(model.feature_names_in_, model.feature_importances_):
    print(f"{name}: {importance:.3f}")
```

The share of each table is in the `sobol` field of the [exported tables](../model-analysis/rating-tables.md).
