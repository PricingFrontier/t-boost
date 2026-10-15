## feature_importances_ {#feature_importances_}

#### Purpose

Return the importance of each input feature, in the order of the input columns. The values sum
to 1.

Each deployed effect's share of the model's variance (its Sobol index under the reference
measure) is split equally among the features it involves, as in the Shapley split of
`predict_contributions`. A multiclassification model averages this over its classes. It is read
from the model and needs no data. See [Feature importance](../../model-analysis/feature-importance.md).

#### Type

numpy.ndarray

## required_columns {#required_columns}

#### Purpose

The input columns needed to apply the model, in the order of the training data. Prediction
methods read these columns by name, and extra columns are ignored. The exposure column is not
among them. Use it to select the columns of a LazyFrame before collecting it:

```python
data.select(model.required_columns).collect()
```

#### Type

list of strings

## n_features_in_ {#n_features_in_}

#### Purpose

The number of features seen during `fit`.

#### Type

int

## feature_names_in_ {#feature_names_in_}

#### Purpose

The names of the features seen during `fit`. Only set when `X` had column names (for example, a
polars `DataFrame`): check with `getattr(model, "feature_names_in_", None)`.

#### Type

numpy.ndarray

## categories_ {#categories_}

#### Purpose

The levels of each categorical feature seen during `fit`, keyed by feature name. The list
includes the levels pooled into the `"<rare>"` level, and `None` for the missing level when the
fit saw missing values. The values are the strings t-boost matches on (a numeric category `1` is
`"1"`).

#### Type

dict

## link {#link}

#### Purpose

The link between the raw score and the prediction: `identity`, `log`, `logit` or, for a
multiclassification model, `softmax`.

#### Type

string

## n_trees_ {#n_trees_}

#### Purpose

The largest number of trees kept by a bag. This number can differ from the value specified in
the `n_trees` training parameter in the following cases:

- The training is stopped by [early stopping](../../algorithm/early-stopping.md).
- No split clears `min_split_gain`.
- A callback stops the bag.

For multiclassification, a boosting round grows one tree per class, and the counts are rounds,
which `n_trees` caps.

#### Type

int

## n_trees_per_bag_ {#n_trees_per_bag_}

#### Purpose

The number of trees each bag kept (boosting rounds, for multiclassification).

#### Type

list of ints

## stopping_reason_ {#stopping_reason_}

#### Purpose

Why the training stopped, summarized over the bags: `"callback"` if a callback stopped a bag, else
`"early_stopping"` if a bag was stopped by early stopping, else `"no_split"` if a bag
ran out of splits, else `"max_trees"`.

#### Type

string

## stopping_reason_per_bag_ {#stopping_reason_per_bag_}

#### Purpose

Why each bag stopped: `"early_stopping"`, `"max_trees"`, `"no_split"` or `"callback"`.

#### Type

list of strings

## evals_result_ {#evals_result_}

#### Purpose

Return the values of metrics calculated during the training: the mean deviance of every
iteration of every bag on the validation objects of
[early stopping](../../algorithm/early-stopping.md), and, when `callbacks` were passed to `fit`,
on the bag's training objects too.

Output format:

```
{"eval": {"deviance": [[value_1, value_2, ...], ...]}, "train": {"deviance": [[value_1, value_2, ...], ...]}}
```

with one list per bag. The `"eval"` curve is recorded on every training (early stopping
calculates it anyway); for multiclassification it is the multinomial deviance, one value per
iteration. The `"train"` curve takes an extra pass per iteration (5–11% of the training time),
so it is calculated only when `callbacks` are given. The attribute is absent when there is
neither (`validation_fraction=None` without `callbacks`) and on a loaded model: it is not saved
with the model.

#### Type

dict

## fitted_at_ {#fitted_at_}

#### Purpose

When the training finished, as an ISO 8601 UTC timestamp (for example
`"2026-10-13T14:02:11+00:00"`). Saved with the model; `None` for a model saved by an older
t-boost.

#### Type

string

## fit_duration_ {#fit_duration_}

#### Purpose

How long the training took, in seconds of wall-clock time. Saved with the model; `None` for a
model saved by an older t-boost.

#### Type

float

## pruning_report_ {#pruning_report_}

#### Purpose

The record of the table selection: the selector used (`selector`), the candidate tables and
their scores (`table_scores`), the kept tables (`kept`), the scored path (`path`) and, when they
ran, the reports of the rebalance (`rebalance`), of banding (`banding`) and of the budgets
(`table_budget`, `box_budget`). On an unpruned model (`prune=False`) the selector is `"none"`
and every table is kept. Saved with the model.

For the ranked-path selector, `path` holds the bank sizes (`sizes`) and their out-of-bag
deviance (`oob_deviance`), the total gain over the main-effects bank (`total_gain`), the step of
the minimum (`min_step`), the first step reaching `prune_path_fraction` of the gain
(`fraction_step`), the first within `prune_path_tolerance` of the minimum (`tolerance_step`),
the chosen cutoff (`best`, the later of the two), the variance floor (`min_variance_share`) with
the tables it removed (`dropped_low_share`), and the ranked tables with the step each one enters
the bank (`sequence`). A multiclassification model records its path under `selection`.

#### Type

dict

## graduation_report_ {#graduation_report_}

#### Purpose

The graduation diagnostics, one entry per table: the features, the smoothing strength (`alpha`)
and whether the smoothing was rejected. For multiclassification, each entry also names its
class (`class`, the index into `classes_`). Only set when graduation is turned on. Saved with
the model.

#### Type

list of dicts

## binding_report_ {#binding_report_}

#### Purpose

Which of the parameters set away from their defaults took effect on the fit. See
[check_bindings](check_bindings.md#binding_report_).

#### Type

list of dicts

## delta_step_gate_ {#delta_step_gate_}

#### Purpose

What the guard of [`max_delta_step_gated`](../../training-parameters/common.md#max_delta_step_gated)
saw: whether it engaged (`engaged`, `engaged_round`, `bags_engaged`), the smallest log ratio of a
predicted rate to the mean rate (`min_log_rate_ratio`) and the guard's settings. `None` when no
guard was armed.

#### Type

dict

## metadata {#metadata}

#### Purpose

Your own JSON metadata, saved with the model by `to_bytes` and `to_json` and returned unchanged by
`from_bytes` and `from_json`. t-boost never reads it: it has no effect on the training, the
predictions or the tables, and a new fit keeps it. The keys must be strings and the values
JSON-serializable (no NaN or infinity), otherwise saving the model raises `SerializationError`.

```python
model.metadata["portfolio"] = "motor"
```

#### Type

dict
