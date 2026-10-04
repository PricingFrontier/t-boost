# Common parameters

## objective {#objective}

#### Description

The [metric](../objectives/index.md) to use in training. The specified value also determines
the machine learning problem to solve.

Possible values:

- `squared_error` — Regression with the identity link.
- `poisson` — Regression with the log link, for counts such as claim frequency.
- `gamma` — Regression with the log link, for positive targets such as claim severity.
- `tweedie` — Regression with the log link, for targets with a mass at zero such as pure
  premium. The power parameter is set by [`tweedie_rho`](#tweedie_rho).
- `logistic` — Classification. A target with two classes trains a logistic model, and a
  target with three or more classes trains a softmax model.

`TBoostRegressor` accepts the four regression objectives and `TBoostClassifier` accepts
`logistic`.

**Type**

string

**Default value**

Depends on the class:

- `TBoostRegressor`: `squared_error`
- `TBoostClassifier`: `logistic`

## tweedie_rho {#tweedie_rho}

#### Description

The power parameter of the Tweedie distribution: the variance is proportional to
$\mu^{\rho}$. Used only when `objective="tweedie"`.

The value must be in the range $(1; 2)$: the compound Poisson-Gamma case typical of insurance
pure-premium targets.

**Type**

float

**Default value**

1.5

## n_trees {#n_trees}

#### Description

The maximum number of trees that can be built when solving machine learning problems.

When using other parameters that limit the number of iterations, the final number of trees may
be less than the number specified in this parameter. By default the
[overfitting detector](overfitting-detection.md) decides how many trees are built, and this
value is a cap that is rarely reached. A fit also stops when no candidate split clears
[`min_split_gain`](#min_split_gain).

With bagging, the limit applies to each bag.

**Type**

int

**Default value**

4000

## learning_rate {#learning_rate}

#### Description

The learning rate.

Used for reducing the gradient step: the values in the leaves of every tree are multiplied by
it before the tree is added to the model.

**Type**

float

**Default value**

0.05

## learning_rate_decay {#learning_rate_decay}

#### Description

The decay of the learning rate over the iterations.

The learning rate of iteration $t$ is

$$
learning\_rate_{t} = \frac{learning\_rate}{1 + learning\_rate\_decay \cdot t}
$$

0 keeps the learning rate constant.

**Type**

float

**Default value**

0.0

## seed {#seed}

#### Description

The random seed used for training.

It seeds every source of randomness in the fit: bagging, object and feature sampling, the folds
of the categorical encoding, DART and [`random_strength`](#random_strength). The same seed and
the same input data reproduce a bit-identical model.

**Type**

int

**Default value**

0

## lambda_ {#lambda_}

#### Description

Coefficient at the L2 regularization term of the cost function.

It is the $\lambda$ in the leaf value $w = -\frac{G}{H + \lambda}$, where $G$ and $H$ are the sums of
the gradients and the Hessians of the objects in the leaf, and in the score of a split.

The trailing underscore is there because `lambda` is a Python keyword. See
[`lambda_scale_invariant`](advanced.md#lambda_scale_invariant) to make the value independent of
the scale of the weights.

**Type**

float

**Default value**

1.0

## l1_leaf {#l1_leaf}

#### Description

Coefficient at the L1 regularization term of the leaf values. The sum of the gradients of a
leaf is soft-thresholded by this value before the leaf value is calculated.

**Type**

float

**Default value**

0.0

## max_depth {#max_depth}

#### Description

Depth of the trees.

Every tree is symmetric (oblivious): each level applies one split, shared by all the nodes of
the level. Possible values are integers from 3 to 8, and the value must be at least
[`max_interaction_order`](#max_interaction_order).

The depth controls the resolution of the trees, not their interaction order. Once a tree uses
`max_interaction_order` distinct features, its deeper levels can only split again on features
it already uses. A deeper tree therefore gives rating tables with finer grids, never with more
axes.

Deeper trees put fewer objects in each leaf. Validate a depth above 3 on held-out data before
using it.

**Type**

int

**Default value**

3

## max_interaction_order {#max_interaction_order}

#### Description

The maximum number of distinct features a tree may use. This is the highest interaction order
in the model, and so the maximum number of axes of a rating table.

Possible values are integers from 1 to 8, and the value must not exceed
[`max_depth`](#max_depth). Set it to 1 to train a model with main effects only.

The cost of a higher order is readability. Pruning keeps an interaction only together with all
of its lower-order sub-tables, so one kept $k$-way table brings $2^k - 1$ tables with it.
Order 4 and 5 can still give a readable set of tables; orders 6 to 8 are supported but rarely
readable.

At order 4 and above, prefer `max_depth` equal to `max_interaction_order`. Effects of that
order are stored as sums of boxes, one per tree at that depth, and deeper trees multiply the
number of boxes.

**Type**

int

**Default value**

3

## min_data_in_leaf {#min_data_in_leaf}

#### Description

The minimum number of training samples in a leaf. A split that would leave fewer samples on
either side is rejected.

Set it explicitly for a model whose tables will be reviewed or filed: it stops the value of a
cell from resting on a handful of objects.

**Type**

int

**Default value**

None (0, no minimum)

## min_sum_hessian_in_leaf {#min_sum_hessian_in_leaf}

#### Description

The minimum sum of the Hessians of the objects in a leaf. A split that would leave less on
either side is rejected.

It complements [`min_data_in_leaf`](#min_data_in_leaf) for weighted data and for objectives
whose Hessian is not constant, where the number of objects alone misstates how much
information a leaf holds.

**Type**

float

**Default value**

0.0

## min_weight_sum_in_leaf {#min_weight_sum_in_leaf}

#### Description

The minimum sum of the sample weights of the objects in a leaf. A split that would leave less
on either side is rejected.

**Type**

float

**Default value**

0.0

## min_split_gain {#min_split_gain}

#### Description

The minimum score a split must exceed to be selected. A level whose best split does not
exceed it is not split, and its nodes stay leaves.

**Type**

float

**Default value**

0.0

## max_delta_step {#max_delta_step}

#### Description

The maximum absolute value of a leaf's Newton step, applied before
[`learning_rate`](#learning_rate). It keeps leaf values finite on sparse or zero-heavy targets.

An explicit value always overrides the objective's default.

**Type**

float

**Default value**

None. Depends on the objective:

- `poisson`, `gamma`, `tweedie`: 0.7
- `squared_error`, `logistic`: no limit

## max_delta_step_gated {#max_delta_step_gated}

#### Description

A guard against the predicted rate of a small group of objects collapsing toward zero during
the training of a log-link model.

At the start of every iteration, the smallest ratio of an object's predicted rate to the
weighted mean rate is checked on the training objects. The first time it falls below
`collapse_threshold`, the leaf step limit ([`max_delta_step`](#max_delta_step)) is tightened to
`capped_step`, for that iteration and every later one. Trees that are already built do not
change. With bagging, each bag is guarded independently.

Possible values:

- `None`, `True` or `"auto"` — Use the objective's default: `tweedie` uses
  `collapse_threshold=0.001` and `capped_step=0.3`, the other objectives use no guard.
- `False` or `"off"` — No guard.
- `(collapse_threshold, capped_step)`, or a dict with these keys — Use the given values. This
  also turns the guard on for objectives that have no default.

An explicit `max_delta_step` takes precedence over the guard. A fit in which the guard never
triggers is identical to a fit without it. After the fit, the `delta_step_gate_` attribute
reports what the guard saw.

**Type**

- None
- bool
- string
- tuple
- dict

**Default value**

None (the objective's default)

## path_smooth {#path_smooth}

#### Description

Path smoothing. The value of a leaf is shrunk toward the value of its parent node with the
credibility weight $Z = \frac{n}{n + path\_smooth}$, where $n$ is the number of objects in the
leaf, so leaves with little data stay close to the path above them. 0 turns it off.

**Type**

float

**Default value**

None. Depends on the objective:

- `gamma`, `tweedie`: 10.0
- other objectives: 0.0 (off)

## colsample_bytree {#colsample_bytree}

#### Description

The fraction of features randomly sampled for each tree.

The value must be in the range $(0; 1]$.

**Type**

float

**Default value**

0.8

## subsample {#subsample}

#### Description

Sample rate of the objects used to choose the structure of each tree.

`None` uses all objects. A value in the range $(0; 1)$ turns on minimal variance sampling (MVS):
the objects are sampled with probabilities that grow with their gradients, and only the sample
is used to choose the splits. The values in the leaves are always calculated on all objects.

Not to be confused with [`bag_subsample`](bagging.md#bag_subsample), the sample rate of each
bag.

**Type**

float

**Default value**

None (all objects are used)

## mvs_min_rows {#mvs_min_rows}

#### Description

The minimum number of objects in the sample when [`subsample`](#subsample) turns on minimal
variance sampling.

**Type**

int

**Default value**

1

## random_strength {#random_strength}

#### Description

The amount of randomness to use for scoring splits when the tree structure is selected. Use
this parameter to avoid overfitting the model.

A uniformly distributed random value is added to the score of each candidate split when the
splits are ranked. Its amplitude at iteration $t$ is the value of this parameter divided by
$\sqrt{t + 1}$, so the randomness decreases during the training. The noise only changes which
split is selected: [`min_split_gain`](#min_split_gain) is checked against the score without
noise.

The noise is seeded by [`seed`](#seed), so the fit stays reproducible. 0 turns it off.

**Type**

float

**Default value**

0.0

## monotone_constraints {#monotone_constraints}

#### Description

Impose monotonic constraints on numerical features.

Possible values:

- <q>1</q> — Increasing constraint on the feature. The algorithm forces the model to be a
  non-decreasing function of this features.
- <q>-1</q> — Decreasing constraint on the feature. The algorithm forces the model to be a
  non-increasing function of this features.
- <q>0</q> — constraints are disabled.

Supported formats for setting the value of this parameter:

- A sequence with one constraint per feature, in the order of the input columns:

    ```python
    monotone_constraints=[1, 0, -1]
    ```

- A dictionary of constraints for the explicitly specified features, keyed by feature name or
  zero-based index:

    ```python
    monotone_constraints={"VehicleAge": -1, "BonusMalus": 1}
    ```

!!! note

    A model with monotonic constraints keeps its full set of tables: pruning, banding and
    graduation are skipped, and `prune_main_effects`, `prune_table_budget` and
    `prune_box_budget` cannot be used with it.

**Type**

- list
- dict

**Default value**

None (no constraints)

## leaf_refine_steps {#leaf_refine_steps}

#### Description

t-boost might calculate leaf values using several Newton steps instead of a single one.

This parameter regulates how many extra steps are done in every tree, after its structure is
chosen, when calculating leaf values. It has no effect on multiclassification.

**Type**

int

**Default value**

4

## leaf_refine_backtracks {#leaf_refine_backtracks}

#### Description

The maximum number of times a leaf refinement step is halved when it does not reduce the loss
(see [`leaf_refine_steps`](#leaf_refine_steps)). It has no effect on multiclassification.

**Type**

int

**Default value**

4

## reanchor {#reanchor}

#### Description

Re-solve the intercept of the model exactly on the training data after the training. This
removes the overall bias that shrinkage leaves in the predictions.

Not supported for multiclassification: setting it explicitly raises an error.

**Type**

bool

**Default value**

None. Depends on the objective:

- `gamma`, `tweedie`: True
- other objectives: False

## reanchor_slope {#reanchor_slope}

#### Description

Recalibrate the raw score $a$ to $b_0 + b_1 \cdot a$ on the validation objects of the
[overfitting detector](overfitting-detection.md) after the training. This corrects a uniform
compression of the score scale that shrinkage and early stopping can leave, without changing the
ranking of the objects or the decomposition into rating tables.

The recalibration is shrunk toward the identity and only applied when its improvement on the
validation objects clears a penalty, so a small validation set leaves the model unchanged. It
has no effect when `validation_fraction` is `None`. Setting it to `True` together with an
`eval_set` raises an error.

Not supported for multiclassification: setting it explicitly raises an error.

**Type**

bool

**Default value**

None (the same as [`reanchor`](#reanchor))
