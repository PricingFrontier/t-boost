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

- The training is stopped by the [overfitting detector](../../algorithm/overfitting-detector.md).
- No split clears `min_split_gain`.
- A callback stops the training.

`None` for a multiclassification model.

#### Type

int

## n_trees_per_bag_ {#n_trees_per_bag_}

#### Purpose

The number of trees each bag kept. `None` for a multiclassification model.

#### Type

list of ints

## stopping_reason_ {#stopping_reason_}

#### Purpose

Why the training stopped, summarized over the bags: `"callback"` if a callback stopped it, else
`"early_stopping"` if a bag was stopped by the overfitting detector, else `"no_split"` if a bag
ran out of splits, else `"max_trees"`. `None` for a multiclassification model.

#### Type

string

## stopping_reason_per_bag_ {#stopping_reason_per_bag_}

#### Purpose

Why each bag stopped: `"early_stopping"`, `"max_trees"`, `"no_split"` or `"callback"`. `None` for
a multiclassification model.

#### Type

list of strings

## evals_result_ {#evals_result_}

#### Purpose

Return the values of metrics calculated during the training: the mean deviance of every
iteration of every bag, when an `eval_set` or `callbacks` were passed to `fit`.

Output format:

```
{"train": {"deviance": [[value_1, value_2, ...], ...]}, "eval": {"deviance": [[value_1, value_2, ...], ...]}}
```

with one list per bag. The `"eval"` curve (the validation objects) costs nothing extra. The
`"train"` curve takes an extra pass per iteration, so it is calculated only when `callbacks` are
given. Without either, the dictionary is empty.

#### Type

dict

## pruning_report_ {#pruning_report_}

#### Purpose

The record of the table selection: the selector used (`selector`), the candidate tables and
their scores (`table_scores`), the kept tables (`kept`), the scored path (`path`) and, when they
ran, the reports of banding (`banding`) and of the budgets (`table_budget`, `box_budget`). Only
set when `prune=True`.

#### Type

dict

## graduation_report_ {#graduation_report_}

#### Purpose

The graduation diagnostics, one entry per table: the features, the smoothing strength (`alpha`)
and whether the smoothing was rejected. Only set when graduation is turned on.

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
