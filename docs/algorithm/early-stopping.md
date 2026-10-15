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

With bagging, every bag stops on its own and keeps its trees up to its own best iteration. Each
bag sets aside its own validation objects, unless `groups` is given: the bags then share one
validation dataset.

## Validation dataset {#validation-dataset}

The validation dataset is a fraction of each bag's training objects,
[`validation_fraction`](../training-parameters/early-stopping.md#validation_fraction)
(10% by default). It is stratified by class for classification and by zero versus non-zero
target for the `poisson` and `tweedie` objectives. When `groups` is passed to `fit`, whole
groups are set aside (once, shared by the bags), so near-duplicate objects of the same entity
cannot fall on both sides.

There is no separate validation dataset for `fit`: the bags already stop and prune on objects
they hold back. To measure the model on data it never saw, pass that data to the
[model report](../model-analysis/model-report.md).

## Results {#results}

After a fit, the following attributes describe the boosting (before pruning). For
multiclassification the counts are boosting rounds, each of which grows one tree per class:

`n_trees_per_bag_`
:   The number of trees each bag kept.

`stopping_reason_per_bag_`
:   Why each bag stopped: `"early_stopping"`, `"max_trees"` (the `n_trees` limit was reached),
    `"no_split"` (no split cleared `min_split_gain`) or `"callback"`.

`n_trees_`, `stopping_reason_`
:   The largest number of trees, and one summary of the reasons.

`evals_result_`
:   The deviance of every iteration of every bag on its validation objects (and on its training
    objects when `callbacks` are given). It is kept on the trained model only, not saved.

A bag that stopped at `"max_trees"` was still improving on its validation objects when it reached
the `n_trees` limit, so the model may be under-fitted; the
[model report](../model-analysis/model-report.md) warns about it. Raise `n_trees` to let early
stopping end the training.

A function passed in `callbacks` is called after every iteration of every bag and can stop that
bag by returning a true value. See [fit](../python-reference/tboostregressor/fit.md).
