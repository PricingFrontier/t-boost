# Pruning settings

After the training, t-boost selects which tables to deploy and drops the rest. See
[Pruning, banding and graduation](../algorithm/readable-tables.md) for details.

Two selectors are available (see [`prune_selector`](#prune_selector)):

- `ranked_path` (default) — The interaction tables are ranked by their purified variance and
  added back in that order, subject to heredity: a $k$-way table enters only once all its
  $(k-1)$-way sub-tables are in. Each prefix is scored on the out-of-bag objects of the bags,
  and the deployed set is the smallest prefix that captures
  [`prune_path_fraction`](#prune_path_fraction) of the improvement and is within
  [`prune_path_tolerance`](#prune_path_tolerance) of the best out-of-bag deviance.
- `fold_vote` — The tables are judged by K-fold cross-validation and a vote across the folds.
  Its parameters are listed in [Fold vote pruning settings](pruning-fold-vote.md).

The ranked path needs out-of-bag objects: `n_bags` of at least 2, and at least
[`prune_guard_min_rows`](#prune_guard_min_rows) objects that are out of bag for some bag. For
regression and binary classification, a fit without them falls back to `fold_vote`. For
multiclassification, it keeps the full set of tables.

The selection is recorded in the `pruning_report_` attribute.

## prune {#prune}

#### Description

Select the tables to deploy after the training, and deploy the smaller set.

`False` deploys the full, unpruned set of tables: faster to train, but a much larger model.
Either way the model is stored as rating tables. [Banding](banding.md) and
[graduation](graduation.md) are only applied to a pruned model.

**Type**

bool

**Default value**

True

## prune_main_effects {#prune_main_effects}

#### Description

Allow pruning to drop main effects too.

`False` deploys every main effect the training built and prunes interactions only. `True` makes
the main effects candidates as well, under hierarchy: a main effect is dropped only when it does
not earn its place and no kept interaction contains it, so a kept interaction always keeps its
main effects. A feature whose main effect is dropped, and which no kept interaction uses, no
longer affects predictions.

On the ranked path, a main effect enters at its own rank, or just before the first interaction
that contains it, and the path starts from the intercept-only model instead of the main-effects
model. [`prune_path_fraction`](#prune_path_fraction) is then measured from the intercept-only
model, which can change the interactions that are kept as well.

Requires `prune=True` and cannot be combined with `monotone_constraints`.

**Type**

bool

**Default value**

False

## prune_selector {#prune_selector}

#### Description

The method used to select the tables.

Possible values:

- `ranked_path` — Rank the tables by purified variance and deploy the smallest heredity-closed
  prefix that is good enough on the out-of-bag objects.
- `fold_vote` — Judge every table by K-fold cross-validation (see
  [Fold vote pruning settings](pruning-fold-vote.md)).

**Type**

string

**Default value**

ranked_path

## prune_path_fraction {#prune_path_fraction}

#### Description

The fraction of the out-of-bag improvement the deployed tables must capture. The improvement is
measured from the model with main effects only (the intercept-only model with
`prune_main_effects=True`) to the prefix with the lowest out-of-bag deviance.

1 deploys the prefix with the lowest out-of-bag deviance. The value must be in the range
$(0; 1]$. Used only by the `ranked_path` selector.

**Type**

float

**Default value**

0.995

## prune_path_tolerance {#prune_path_tolerance}

#### Description

The maximum relative excess of the deployed prefix's out-of-bag deviance over the best prefix's.
The deployed prefix is the larger of the smallest prefix that captures
[`prune_path_fraction`](#prune_path_fraction) of the improvement and the smallest prefix within
this tolerance.

The fraction alone can concentrate the loss where interactions matter most, because a small
fraction of a large improvement can still be a large loss of deviance. This bound caps that loss
at the given fraction of the deviance. The value must be non-negative. Used only by the
`ranked_path` selector, and ignored when `prune_path_fraction` is 1.

**Type**

float

**Default value**

0.001

## prune_path_steps {#prune_path_steps}

#### Description

The number of prefixes scored on the ranked path. The prefix sizes are spaced geometrically
between one table and all the candidate tables. Used only by the `ranked_path` selector.

**Type**

int

**Default value**

32

## prune_guard_min_rows {#prune_guard_min_rows}

#### Description

The minimum number of objects with honest evidence needed to judge the tables on.

For the `ranked_path` selector, these are the objects that are out of bag for at least one bag:
with fewer, the ranked path is not used. For the `fold_vote` selector, it is the minimum number
of evidence objects for its [no-harm guard](pruning-fold-vote.md#prune_guard) to act.

**Type**

int

**Default value**

500

## prune_rebalance {#prune_rebalance}

#### Description

After tables are dropped, re-solve the values of the remaining tables for the smaller structure
(one ridge IRLS step, followed by purification). The new values are kept only if they lower the
held-out deviance. Models with multi-channel categorical features (see
[`cat_channels`](categorical.md#cat_channels)) are not rebalanced.

`False` deploys the remaining tables with their values unchanged.

**Type**

bool

**Default value**

True

## prune_table_budget {#prune_table_budget}

#### Description

The maximum number of deployed tables of order
[`prune_table_min_arity`](#prune_table_min_arity) or higher.

The tables are kept in order of their evidence, and the tables past the limit are dropped,
together with any table that contains them. Tables below the arity floor are never counted and
never dropped, and every kept table keeps all of its cells. A larger limit never keeps fewer
tables. 0 turns the limit off.

For multiclassification the limit counts the tables shared by the classes, not each class's
copy. Requires `prune=True`.

**Type**

int

**Default value**

0 (no limit)

## prune_table_min_arity {#prune_table_min_arity}

#### Description

The lowest interaction order counted by [`prune_table_budget`](#prune_table_budget) and by
[`prune_lambda_tables`](pruning-fold-vote.md#prune_lambda_tables). The default counts three-way
and higher tables, so main effects and pairs are never limited. Allowed values are integers
from 1 to 8.

**Type**

int

**Default value**

3

## prune_box_budget {#prune_box_budget}

#### Description

The maximum total number of boxes in the deployed model.

An effect whose dense table would be too large is stored in factored form, as a sum of rank-one
boxes (regions of the feature space), and each box is one row of its exported rating table. A
deeper tree contributes more boxes, so `max_depth` mostly multiplies the number of boxes rather
than the number of tables.

The boxes are kept in order of evidence per box, and the effects that do not fit are dropped,
together with any table that contains them. Dense tables cost no boxes. A budget at or above the
model's own total changes nothing. 0 turns the budget off. Requires `prune=True`.

**Type**

int

**Default value**

0 (no budget)
