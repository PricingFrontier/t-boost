# Advanced settings

These parameters are rarely needed. None of them is supported for multiclassification: setting
one away from its default for a model with three or more classes raises an error.

## lambda_scale_invariant {#lambda_scale_invariant}

#### Description

Rescale [`lambda_`](common.md#lambda_) by the mean Hessian per object of each iteration instead
of using it as it is.

With large sample weights or exposures the sums of the Hessians in the leaves can be so large
that a value of `lambda_` in the usual range has no effect. Rescaled, `lambda_` acts as a number
of prior objects, whatever the scale of the weights.

**Type**

bool

**Default value**

False

## ridge_refit_l2 {#ridge_refit_l2}

#### Description

The L2 penalty of a fully corrective refit. After the trees are built, all their leaf values are
re-solved jointly by regularized IRLS, with the tree structures fixed.

`None` turns the refit off. Not supported with `max_depth` above 3.

**Type**

float

**Default value**

None (off)

## ridge_refit_max_iter {#ridge_refit_max_iter}

#### Description

The maximum number of IRLS iterations of the fully corrective refit. Only used when
[`ridge_refit_l2`](#ridge_refit_l2) is set.

**Type**

int

**Default value**

5

## dart_drop_rate {#dart_drop_rate}

#### Description

Turns on DART (Dropout Additive Regression Trees): at each iteration, the trees built so far are
dropped with this probability before the new tree is built, and the tree weights are normalized
as in DART. The value must be in the range $[0; 1)$.

**Type**

float

**Default value**

None (off)

## nesterov {#nesterov}

#### Description

Reserved for Nesterov-accelerated boosting, which is not implemented yet. Setting it to `True`
raises an error.

**Type**

bool

**Default value**

False

## prune_refit_full {#prune_refit_full}

#### Description

Deprecated, and without effect: the deployed model is always trained on all objects. Setting it
to `True` raises an error. It will be removed in a future release.

**Type**

bool

**Default value**

False
