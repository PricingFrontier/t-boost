# Fold vote pruning settings

These parameters are used only when the tables are selected by the `fold_vote` selector:
with `prune_selector="fold_vote"`, or when a regression or binary classification fit has no
out-of-bag objects for the default `ranked_path` selector (for example, with `n_bags=1`). See
[Pruning settings](pruning.md) for the parameters that apply to both selectors.

The fold vote splits the objects into folds, trains a model on each fold's complement and
measures, on the fold, how much each candidate table improves the held-out deviance (its gain).
A table is kept when its mean gain is positive and it shows that signal in enough folds.

## prune_n_folds {#prune_n_folds}

#### Description

The number of cross-validation folds.

On small datasets the number adapts down, so that every fold keeps at least
[`prune_fold_min_rows`](#prune_fold_min_rows) objects:

$$
k = \max\left(2, \min\left(prune\_n\_folds, \left\lfloor\frac{n}{prune\_fold\_min\_rows}\right\rfloor\right)\right)
$$

**Type**

int

**Default value**

5

## prune_fold_min_rows {#prune_fold_min_rows}

#### Description

The minimum number of objects in each fold (see [`prune_n_folds`](#prune_n_folds)). Raising it
makes the number of folds adapt down earlier.

**Type**

int

**Default value**

125

## prune_fold_es_patience {#prune_fold_es_patience}

#### Description

The number of iterations early stopping of the fold models waits after the iteration with the
optimal metric value. The fold models only vote on which tables to keep, so they use a
shorter patience than the deployed model, which always uses
[`early_stopping_rounds`](early-stopping.md#early_stopping_rounds).

**Type**

int

**Default value**

None (250)

## prune_min_stability {#prune_min_stability}

#### Description

The fraction of the folds in which a table must show signal to be kept. A table is kept when its
mean gain exceeds [`prune_min_mean_gain`](#prune_min_mean_gain) and either it was kept by at least
this fraction of the folds' own selections, or its gain was positive in at least this fraction of
the folds.

The fractions are multiples of $1/k$, so with 5 folds only a few positions are distinct: values
in $(0.4; 0.6]$ require 3 of 5 folds, $(0.6; 0.8]$ 4 of 5, and $(0.8; 1]$ all 5. Higher values
deploy fewer tables.

**Type**

float

**Default value**

0.5

## prune_min_mean_gain {#prune_min_mean_gain}

#### Description

The minimum mean gain a table must exceed to be kept, in units of deviance per unit of weight.
0 keeps every table with a positive mean gain.

!!! note

    The floor applies to the vote only. A table it rejects becomes a candidate for the evidence
    gate ([`prune_drop_z`](#prune_drop_z)), which keeps it unless its gain is significantly
    negative, so raising the floor can deploy more tables, not fewer. Set `prune_drop_z=None`
    when using the floor to deploy fewer tables.

**Type**

float

**Default value**

0.0

## prune_drop_z {#prune_drop_z}

#### Description

The evidence bar for dropping a table. A table the vote would drop is kept instead unless its
gain is significantly negative: its mean over the folds must be below $-z \cdot SE$, where $z$ is
the value of this parameter and $SE$ the standard error of the mean. Tables kept this way are
ranked by mean gain and limited by [`prune_keep_budget`](#prune_keep_budget). Tables scored by
fewer than two folds keep the vote's verdict.

The gate can only keep more tables than the vote alone. `None` turns it off.

**Type**

float

**Default value**

2.0

## prune_keep_budget {#prune_keep_budget}

#### Description

The maximum number of tables after the evidence gate ([`prune_drop_z`](#prune_drop_z)) adds its
tables: the limit is the larger of this value and the number of tables the vote kept. A set that
the vote already made larger than the budget is left as the vote chose it.

**Type**

int

**Default value**

32

## prune_lambda_tables {#prune_lambda_tables}

_Alias:_ `prune_size_penalty`

#### Description

The price of one kept table of order
[`prune_table_min_arity`](pruning.md#prune_table_min_arity) or higher, in units of held-out
deviance. The selection minimizes the held-out deviance plus this price times the number of
such tables. 0 turns it off.

The price only chooses between the sets on the backward path the selection already took, so it
cannot reach every table count. Use [`prune_table_budget`](pruning.md#prune_table_budget) to
deploy a given number of tables.

**Type**

float

**Default value**

0.0

## prune_size_penalty {#prune_size_penalty}

#### Description

An alias of [`prune_lambda_tables`](#prune_lambda_tables). Setting both to different values
raises an error.

**Type**

float

**Default value**

None

## prune_lambda_boxes {#prune_lambda_boxes}

#### Description

The price of one deployed box (see [`prune_box_budget`](pruning.md#prune_box_budget)), in units
of held-out deviance per unit of weight. The selection minimizes the held-out deviance plus this
price times the number of boxes, so a large set of tables has to earn its size. Dense tables cost
no boxes. 0 turns it off.

The `path` entries of `pruning_report_` carry `n_boxes`, `mean_deviance` and `se`, which help
calibrate the price: for example, `path[0]["se"] / path[0]["n_boxes"]` prices the whole model at
one standard error.

**Type**

float

**Default value**

0.0

## prune_fold_fidelity {#prune_fold_fidelity}

#### Description

Score every candidate table in every fold. A fold model searches its own structure, so it may not
build some of the tables the deployed model has, and those tables then get no evidence from that
fold. With this parameter, the missing tables are given values in each fold by a ridge fit on the
fold's training objects, so every candidate gets a held-out gain.

Supported only for regression and binary classification.

**Type**

bool

**Default value**

False
