# Python package

The t-boost Python package provides the following classes, functions and modules. See
[Quick start](../quickstart.md) for examples.

## Classes {#classes}

### [TBoostRegressor](tboostregressor/index.md)

**Class purpose**

Training and applying regression models: `squared_error`, `poisson`, `gamma` and `tweedie`.

### [TBoostClassifier](tboostclassifier/index.md)

**Class purpose**

Training and applying classification models: logistic for two classes, softmax for three or
more.

### [ContributionMatrix](contribution-matrix.md)

**Class purpose**

The dense result of `predict_contributions(return_format="matrix")`.

## Functions {#functions}

### [recommended_recipe](recommended-recipe.md)

Return an estimator configured with the benchmark recipe.

## Modules {#modules}

### [metrics](metrics.md)

Deviances and weight-aware ranking metrics, calculated separately from the training.

## Exceptions and warnings {#exceptions}

### [TBoostError and its subclasses, PrecisionWarning](exceptions.md)

The errors raised by t-boost, and the warning issued when features are converted to `float32`.
