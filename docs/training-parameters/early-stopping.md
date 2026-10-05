# Early stopping settings

Early stopping ends the training of each bag when the deviance of the objective on a
validation dataset stops improving, and keeps the trees up to the iteration with the best
deviance. By default the validation dataset is a fraction of the training objects
([`validation_fraction`](#validation_fraction)). Pass `eval_set` to `fit` to use a separate
dataset instead. See [Early stopping](../algorithm/early-stopping.md) for details.

## validation_fraction {#validation_fraction}

#### Description

The fraction of the training objects set aside as the validation dataset of early stopping.
Each bag sets aside its own validation objects from its own sample.

The validation objects are stratified for classification (by class) and for the `poisson` and
`tweedie` objectives (zero versus non-zero target). When `groups` is passed to `fit`, whole
groups are set aside instead of single objects, once, and the bags share them.

`None` turns early stopping off, so every bag builds [`n_trees`](common.md#n_trees)
trees. The value is ignored when an `eval_set` is passed to `fit`.

**Type**

float

**Default value**

0.1

## early_stopping_rounds {#early_stopping_rounds}

#### Description

Stops the training after the specified number of iterations since the iteration with the
optimal metric value. The metric is the mean deviance of the objective on the validation
objects.

With [`early_stopping_adaptive`](#early_stopping_adaptive) set, this is the upper bound of the
number of iterations. It is ignored when early stopping is off.

**Type**

int

**Default value**

500

## early_stopping_adaptive {#early_stopping_adaptive}

#### Description

Makes the number of iterations to wait grow with the iteration of the best result.

The number of iterations to wait is

$$
patience = \min\left(\max\left(\lceil r \cdot best\_iteration \rceil, 50\right), early\_stopping\_rounds\right)
$$

where $r$ is the value of this parameter. A fit whose best iteration comes early stops sooner.
`None` waits a fixed [`early_stopping_rounds`](#early_stopping_rounds) iterations.

**Type**

float

**Default value**

1.5

## early_stopping_min_delta {#early_stopping_min_delta}

#### Description

The minimum relative improvement of the metric for an iteration to become the new best. The
validation deviance must fall below $best \cdot (1 - early\_stopping\_min\_delta)$; smaller
improvements do not reset the count of iterations to wait and do not move the iteration the
model is truncated at.

0 counts any improvement. The value must be in the range $[0; 1)$.

**Type**

float

**Default value**

0.0001

## early_stopping {#early_stopping}

#### Description

A single setting for the patience of early stopping. An `int` sets
[`early_stopping_rounds`](#early_stopping_rounds), a `float` sets
[`early_stopping_adaptive`](#early_stopping_adaptive).

Setting it together with a different value of the parameter it sets raises an error.

**Type**

- int
- float

**Default value**

None
