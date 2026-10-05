# Bagging settings

t-boost trains several bags (independent models, each on a sample of the objects) and averages
them. The average is exact: the averaged rating tables are the average of the bags' tables, so
the deployed model is still one set of tables. The objects a bag did not see (its out-of-bag
objects) give honest evidence that [pruning](pruning.md) and [banding](banding.md) use. See
[Bagging](../algorithm/bagging.md) for details.

## n_bags {#n_bags}

#### Description

The number of bags. Each bag is trained on its own sample of the objects, with its own early
stopping, and the bags are averaged into one model. Training costs about `n_bags`
times as much as training a single model.

1 turns bagging off. Without bagging there are no out-of-bag objects: banding is skipped, and
pruning falls back to the cross-validated selector, or keeps the full set of tables for
multiclassification (see [Pruning settings](pruning.md)).

**Type**

int

**Default value**

8

## bag_subsample {#bag_subsample}

#### Description

The fraction of the objects sampled for each bag. Only used when [`n_bags`](#n_bags) is greater
than 1.

A value below 1 samples the objects without replacement (subagging). A value of 1 or more draws
a full-size bootstrap sample with replacement. Subagging keeps early stopping honest:
in a bootstrap sample duplicated objects can fall on both sides of the validation split, so the
validation deviance keeps improving and the training does not stop. When `groups` is passed to
`fit`, whole groups are sampled.

**Type**

float

**Default value**

0.8

## cell_refit_base {#cell_refit_base}

#### Description

The base penalty of the out-of-bag cell refit. After bagging, every cell of the averaged tables
is refit toward the residuals of the bags' out-of-bag objects under a ridge penalty shaped by
[`cell_refit_gamma`](#cell_refit_gamma), and the tables are purified again. The refit is kept
only if it improves the held-out deviance.

`None` turns the refit off. It requires `n_bags` of at least 2 and cannot be combined with
`monotone_constraints`.

**Type**

float

**Default value**

None (off)

## cell_refit_gamma {#cell_refit_gamma}

#### Description

The adaptive exponent of the penalty of the out-of-bag cell refit. 0 applies the same ridge
penalty to every cell; larger values penalize cells with a strong signal less. Only used when
[`cell_refit_base`](#cell_refit_base) is set.

**Type**

float

**Default value**

2.0
