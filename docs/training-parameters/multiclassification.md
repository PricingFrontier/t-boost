# Multiclassification settings

These parameters are used only by `TBoostClassifier` with three or more classes, and only when
the tables are selected by the `fold_vote` selector (`prune_selector="fold_vote"`). Setting one
of them away from its default for a regression or binary classification model raises an error
where noted.

A multiclassification model has one set of rating tables per class, and pruning selects one
set of effects shared by the classes. See also [Pruning settings](pruning.md) and
[Fold vote pruning settings](pruning-fold-vote.md).

## multiclass_prune_cv {#multiclass_prune_cv}

#### Description

The selection regime for multiclassification.

- `True` — K-fold cross-validation, as for regression and binary classification:
  [`prune_n_folds`](pruning-fold-vote.md#prune_n_folds) fold models are scored on their held-out
  folds, voted on with [`prune_min_stability`](pruning-fold-vote.md#prune_min_stability), gated
  by [`prune_drop_z`](pruning-fold-vote.md#prune_drop_z) and
  [`prune_keep_budget`](pruning-fold-vote.md#prune_keep_budget).
- `False` — A single train/select split (see
  [`prune_validation_fraction`](#prune_validation_fraction)). This is the earlier regime, kept
  for comparison: it tends to drop tables on selection noise.

**Type**

bool

**Default value**

True

## multiclass_prune_sel_bags {#multiclass_prune_sel_bags}

#### Description

The number of bags of the models the selection is trained on (the fold models, or the selection
model of the single-split regime). Must be at least 1. Any other value than 1 raises an error for
regression and binary classification.

**Type**

int

**Default value**

1

## prune_validation_fraction {#prune_validation_fraction}

#### Description

The fraction of the objects used to select the tables in the single-split regime
(`multiclass_prune_cv=False`). Any other value than the default raises an error in the other
regimes, and for regression and binary classification.

**Type**

float

**Default value**

0.15

## prune_se_rule {#prune_se_rule}

#### Description

The selection rule, in standard errors of the held-out deviance estimate. 0 selects the set with
the lowest held-out deviance; larger values (for example, the classic one-standard-error rule,
1.0) prune more aggressively and trade deviance for a smaller model.

Supported only for multiclassification: any other value than 0 raises an error for regression
and binary classification. Use [`prune_table_budget`](pruning.md#prune_table_budget) or
[`prune_box_budget`](pruning.md#prune_box_budget) for a smaller model there.

**Type**

float

**Default value**

0.0
