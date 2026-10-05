# Early stopping

If overfitting occurs, t-boost can stop the training earlier than the training parameters
dictate. For example, it can be stopped before the specified number of trees are built. This
option is set in the starting parameters, and it is turned on by default.

After building each new tree, t-boost calculates the mean deviance of the objective on the
validation dataset. The iteration with the lowest deviance so far is the best iteration. An
iteration becomes the new best only if it improves on the previous best by more than
[`early_stopping_min_delta`](../training-parameters/early-stopping.md#early_stopping_min_delta)
(relative to the best deviance), so noise-level improvements are ignored.

The training stops when the number of iterations since the best iteration exceeds the
patience:

$$
patience = \min\left(\max\left(\lceil r \cdot best\_iteration \rceil, 50\right), early\_stopping\_rounds\right)
$$

where $r$ is [`early_stopping_adaptive`](../training-parameters/early-stopping.md#early_stopping_adaptive)
(1.5 by default). With `early_stopping_adaptive=None` the patience is
[`early_stopping_rounds`](../training-parameters/early-stopping.md#early_stopping_rounds).
When the training stops, the trees built after the best iteration are discarded.

With bagging, every bag stops on its own and keeps its trees up to its own best iteration. Each bag sets aside its own validation objects, unless `groups` or an `eval_set` is
given: the bags then share one validation dataset.

## Validation dataset {#validation-dataset}

The validation dataset is one of the following:

- A fraction of each bag's training objects,
  [`validation_fraction`](../training-parameters/early-stopping.md#validation_fraction)
  (10% by default). It is stratified by class for classification and by zero versus non-zero
  target for the `poisson` and `tweedie` objectives. When `groups` is passed to `fit`, whole
  groups are set aside (once, shared by the bags), so near-duplicate objects of the same entity
  cannot fall on both sides.
- The `eval_set` passed to `fit`. No objects are set aside from the training dataset, and the
  evaluation objects are used only by early stopping: they never reach the training,
  the tables or pruning.

## Results {#results}

After a regression or binary classification fit, the following attributes describe the
boosting (before pruning):

`n_trees_per_bag_`
:   The number of trees each bag kept.

`stopping_reason_per_bag_`
:   Why each bag stopped: `"early_stopping"`, `"max_trees"` (the `n_trees` limit was reached),
    `"no_split"` (no split cleared `min_split_gain`) or `"callback"`.

`n_trees_`, `stopping_reason_`
:   The largest number of trees, and one summary of the reasons.

`evals_result_`
:   With an `eval_set` or `callbacks`, the deviance of every iteration of every bag.

A function passed in `callbacks` is called after every iteration and can stop the training by
returning a true value. See [fit](../python-reference/tboostregressor/fit.md).
