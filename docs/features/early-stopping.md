# Using early stopping

If overfitting occurs, t-boost can stop the training earlier than the training parameters
dictate. For example, it can be stopped before the specified number of trees are built. Early
stopping is turned on by default. See [Early stopping](../algorithm/early-stopping.md) for how it
works.

## Python package {#python-package}

The following parameters can be set in the constructor of the
[TBoostRegressor](../python-reference/tboostregressor/index.md) and
[TBoostClassifier](../python-reference/tboostclassifier/index.md) classes and are used when the
model is trained:

`validation_fraction`
:   The fraction of the training objects set aside as the validation dataset (0.1). `None` turns
    early stopping off.

`early_stopping_rounds`
:   The number of iterations to continue the training after the iteration with the optimal
    metric value (500).

`early_stopping_adaptive`
:   Makes the number of iterations to wait grow with the iteration of the best result (1.5).

`early_stopping_min_delta`
:   The minimum relative improvement of the metric for an iteration to become the new best
    (0.0001).

See [Early stopping settings](../training-parameters/early-stopping.md) for details.

The following parameters can be set for the `fit` method:

`eval_set`, `eval_sample_weight`, `eval_exposure`, `eval_offset`
:   A separate validation dataset, used instead of the objects set aside from the training
    dataset.

`callbacks`
:   Functions called after every iteration, which can stop the training.

## Usage examples {#usage-examples}

```python
from t_boost import TBoostRegressor

model = TBoostRegressor(objective="poisson")
model.fit(train_data, "ClaimCount", exposure="Exposure",
          eval_set=(valid_data, "ClaimCount"), eval_exposure="Exposure")

print(model.n_trees_per_bag_)          # the trees each bag kept
print(model.stopping_reason_per_bag_)  # why each bag stopped
deviance = model.evals_result_["eval"]["deviance"]  # one curve per bag
```
