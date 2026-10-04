# Objectives and metrics

This section contains basic information regarding the supported objectives for various machine
learning problems.

- [Regression](regression.md)
- [Classification](classification.md)
- [Multiclassification](multiclassification.md)

Refer to the [Variables used in formulas](variables.md) section for the description of commonly
used variables in the listed metrics.

The objective is set by the [`objective`](../training-parameters/common.md#objective) parameter.
Every objective is the deviance of a distribution with a link function, and the same deviance is
used throughout the fit: the trees minimize it, the
[overfitting detector](../algorithm/overfitting-detector.md) monitors it on the validation
dataset, and pruning compares sets of tables with it.

| Objective | Machine learning problem | Link | Class |
|-----------|--------------------------|------|-------|
| [`squared_error`](regression.md#squared_error) | regression | identity | `TBoostRegressor` |
| [`poisson`](regression.md#poisson) | claim frequency, counts | log | `TBoostRegressor` |
| [`gamma`](regression.md#gamma) | severity | log | `TBoostRegressor` |
| [`tweedie`](regression.md#tweedie) | pure premium | log | `TBoostRegressor` |
| [`logistic`](classification.md#logistic) | binary classification | logit | `TBoostClassifier` |
| [softmax](multiclassification.md#softmax) (automatic for 3+ classes) | multiclassification | softmax | `TBoostClassifier` |

Metrics can also be calculated separately from the training with the
[`t_boost.metrics`](../python-reference/metrics.md) module: the deviances of the regression
objectives and weight-aware ranking metrics (Gini, lift).
