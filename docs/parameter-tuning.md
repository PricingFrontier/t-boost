# Parameter tuning

t-boost provides a flexible interface for parameter tuning and can be configured to suit
different tasks. The default parameters are already a complete recipe: early stopping, bagging,
pruning, banding and graduation are all turned on, so a model usually performs well without
tuning.

This section contains some tips on the possible parameter settings.

## Number of trees {#trees-number}

It is recommended to check that there is no obvious underfitting or overfitting before tuning
any other parameters. In order to do this it is necessary to analyze the metric value on the
validation dataset and the number of trees each bag kept.

By default the number of trees ([`n_trees`](training-parameters/common.md#n_trees)) is a large
cap and the [overfitting detector](algorithm/overfitting-detector.md) decides when to stop.
After the fit, check `stopping_reason_per_bag_`: bags that stopped with `"max_trees"` reached the
cap before the detector triggered, so raise `n_trees` or `learning_rate`. Pass an `eval_set` to
`fit` to monitor the deviance of every iteration in `evals_result_`.

## Learning rate {#learning-rate}

This setting is used for reducing the gradient step. It affects the overall time of training:
the smaller the value, the more iterations are required for training. Choose the value based on
the performance expectations.

Possible ways of adjusting the learning rate depending on the overfitting results:

- There is no overfitting on the last iterations of training (the training does not converge) —
  increase the learning rate.
- Overfitting is detected — decrease the learning rate.

## Tree depth and interaction order {#tree-depth}

[`max_interaction_order`](training-parameters/common.md#max_interaction_order) decides the
largest tables the model can have: 1 gives main effects only, 2 adds pairs, and the default 3
adds three-way tables. Higher orders are supported but quickly stop being readable.

[`max_depth`](training-parameters/common.md#max_depth) (3 by default) controls the resolution
of the trees, not their interaction order. Keep the default unless a held-out comparison shows
a gain, and keep `max_depth` equal to `max_interaction_order` at order 4 and above.

## L2 regularization {#l2-reg}

Try different values for the regularizer ([`lambda_`](training-parameters/common.md#lambda_))
to find the best possible. With large sample weights or exposures, set
[`lambda_scale_invariant=True`](training-parameters/advanced.md#lambda_scale_invariant) so that
the value keeps an effect.

## Bagging {#bagging}

[`n_bags`](training-parameters/bagging.md#n_bags) (8 by default) trades training time for
accuracy: the training costs about `n_bags` times a single model. For the cheapest single-model
baseline, pass `n_bags=1, validation_fraction=None, prune=False`.

## Smaller rating structures {#smaller-models}

The following parameters give fewer or smaller tables:

- [`prune_table_budget`](training-parameters/pruning.md#prune_table_budget) — Limit the number of
  three-way and higher tables.
- [`prune_path_fraction`](training-parameters/pruning.md#prune_path_fraction) — A lower value
  deploys a shorter prefix of the ranked tables.
- [`prune_main_effects`](training-parameters/pruning.md#prune_main_effects) — Allow pruning to
  drop main effects too.
- [`max_interaction_order`](training-parameters/common.md#max_interaction_order) — A lower order.
- [`interaction_gain_hurdle`](training-parameters/interaction.md#interaction_gain_hurdle) — A
  higher hurdle admits fewer interactions while the trees grow.
- [`band_tolerance`](training-parameters/banding.md#band_tolerance) — A higher tolerance gives
  coarser bands.

## Models for review or filing {#review}

- Set [`min_data_in_leaf`](training-parameters/common.md#min_data_in_leaf) explicitly, so that no
  cell rests on a handful of objects.
- Call [check_bindings](python-reference/tboostregressor/check_bindings.md) after the fit to make
  sure the parameters you set took effect.
- Save the [pricing report](model-analysis/actual-vs-expected.md#pricing-report) beside the
  serialized model, and record your context in the `metadata` attribute.

## Methods for hyperparameter search {#defining-optimal-parameter-values}

With scikit-learn installed, the estimators work with its model selection tools, such as
`GridSearchCV` and `RandomizedSearchCV`:

```python
from sklearn.model_selection import GridSearchCV
from t_boost import TBoostRegressor

grid = {"learning_rate": [0.05, 0.1], "lambda_": [1.0, 10.0]}
search = GridSearchCV(TBoostRegressor(), grid, cv=3).fit(X, y)
print(search.best_params_)
```

The default score is R2 for `TBoostRegressor` and accuracy for `TBoostClassifier`. For a model
with an exposure, pass a scorer based on the deviances of [t_boost.metrics](python-reference/metrics.md).
For panel data, use a group-aware splitter such as `GroupKFold`.
