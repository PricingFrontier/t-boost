# Overview

These parameters are for the Python package classes `TBoostRegressor` and `TBoostClassifier`.
Both classes accept the same parameters; only the default value of `objective` differs.

`n_trees`, `learning_rate` and `lambda_` can be passed by position. Every other parameter is
keyword-only.

## [Common parameters](common.md)

### [objective](common.md#objective)

The [metric](../objectives/index.md) to use in training. The specified value also determines
the machine learning problem to solve.

### [tweedie_rho](common.md#tweedie_rho)

The power parameter of the Tweedie distribution: the variance is proportional to
$\mu^{\rho}$. Used only when `objective="tweedie"`.

### [n_trees](common.md#n_trees)

The maximum number of trees that can be built when solving machine learning problems.

### [learning_rate](common.md#learning_rate)

The learning rate.

Used for reducing the gradient step: the values in the leaves of every tree are multiplied by
it before the tree is added to the model.

### [learning_rate_decay](common.md#learning_rate_decay)

The decay of the learning rate over the iterations.

### [seed](common.md#seed)

The random seed used for training.

It seeds every source of randomness in the fit: bagging, object and feature sampling, the folds
of the categorical encoding, DART and [`random_strength`](common.md#random_strength). The same seed and
the same input data reproduce a bit-identical model.

### [lambda_](common.md#lambda_)

Coefficient at the L2 regularization term of the cost function.

### [l1_leaf](common.md#l1_leaf)

Coefficient at the L1 regularization term of the leaf values. The sum of the gradients of a
leaf is soft-thresholded by this value before the leaf value is calculated.

### [max_depth](common.md#max_depth)

Depth of the trees.

Every tree is symmetric (oblivious): each level applies one split, shared by all the nodes of
the level. Possible values are integers from 3 to 8, and the value must be at least
[`max_interaction_order`](common.md#max_interaction_order).

### [max_interaction_order](common.md#max_interaction_order)

The maximum number of distinct features a tree may use. This is the highest interaction order
in the model, and so the maximum number of axes of a rating table.

### [min_data_in_leaf](common.md#min_data_in_leaf)

The minimum number of training samples in a leaf. A split that would leave fewer samples on
either side is rejected.

### [min_sum_hessian_in_leaf](common.md#min_sum_hessian_in_leaf)

The minimum sum of the Hessians of the objects in a leaf. A split that would leave less on
either side is rejected.

### [min_weight_sum_in_leaf](common.md#min_weight_sum_in_leaf)

The minimum sum of the sample weights of the objects in a leaf. A split that would leave less
on either side is rejected.

### [min_split_gain](common.md#min_split_gain)

The minimum score a split must exceed to be selected. A level whose best split does not
exceed it is not split, and its nodes stay leaves.

### [max_delta_step](common.md#max_delta_step)

The maximum absolute value of a leaf's Newton step, applied before
[`learning_rate`](common.md#learning_rate). It keeps leaf values finite on sparse or zero-heavy targets.

### [max_delta_step_gated](common.md#max_delta_step_gated)

A guard against the predicted rate of a small group of objects collapsing toward zero during
the training of a log-link model.

### [path_smooth](common.md#path_smooth)

Path smoothing. The value of a leaf is shrunk toward the value of its parent node with the
credibility weight $Z = \frac{n}{n + path\_smooth}$, where $n$ is the number of objects in the
leaf, so leaves with little data stay close to the path above them. 0 turns it off.

### [colsample_bytree](common.md#colsample_bytree)

The fraction of features randomly sampled for each tree.

The value must be in the range $(0; 1]$.

### [subsample](common.md#subsample)

Sample rate of the objects used to choose the structure of each tree.

### [mvs_min_rows](common.md#mvs_min_rows)

The minimum number of objects in the sample when [`subsample`](common.md#subsample) turns on minimal
variance sampling.

### [random_strength](common.md#random_strength)

The amount of randomness to use for scoring splits when the tree structure is selected. Use
this parameter to avoid overfitting the model.

### [monotone_constraints](common.md#monotone_constraints)

Impose monotonic constraints on numerical features.

### [leaf_refine_steps](common.md#leaf_refine_steps)

t-boost might calculate leaf values using several Newton steps instead of a single one.

### [leaf_refine_backtracks](common.md#leaf_refine_backtracks)

The maximum number of times a leaf refinement step is halved when it does not reduce the loss
(see [`leaf_refine_steps`](common.md#leaf_refine_steps)). It has no effect on multiclassification.

### [reanchor](common.md#reanchor)

Re-solve the intercept of the model exactly on the training data after the training. This
removes the overall bias that shrinkage leaves in the predictions.

### [reanchor_slope](common.md#reanchor_slope)

Recalibrate the raw score $a$ to $b_0 + b_1 \cdot a$ on the validation objects of the
[overfitting detector](overfitting-detection.md) after the training. This corrects a uniform
compression of the score scale that shrinkage and early stopping can leave, without changing the
ranking of the objects or the decomposition into rating tables.

## [Overfitting detection settings](overfitting-detection.md)

### [validation_fraction](overfitting-detection.md#validation_fraction)

The fraction of the training objects set aside as the validation dataset of the overfitting
detector. Each bag sets aside its own validation objects from its own sample.

### [early_stopping_rounds](overfitting-detection.md#early_stopping_rounds)

Stops the training after the specified number of iterations since the iteration with the
optimal metric value. The metric is the mean deviance of the objective on the validation
objects.

### [early_stopping_adaptive](overfitting-detection.md#early_stopping_adaptive)

Makes the number of iterations to wait grow with the iteration of the best result.

### [early_stopping_min_delta](overfitting-detection.md#early_stopping_min_delta)

The minimum relative improvement of the metric for an iteration to become the new best. The
validation deviance must fall below $best \cdot (1 - early\_stopping\_min\_delta)$; smaller
improvements do not reset the count of iterations to wait and do not move the iteration the
model is truncated at.

### [early_stopping](overfitting-detection.md#early_stopping)

A single setting for the patience of the overfitting detector. An `int` sets
[`early_stopping_rounds`](overfitting-detection.md#early_stopping_rounds), a `float` sets
[`early_stopping_adaptive`](overfitting-detection.md#early_stopping_adaptive).

## [Quantization settings](quantization.md)

### [max_bin](quantization.md#max_bin)

The maximum number of bins for numerical features, not counting the bin for missing values.
Allowed values are integers from 2 to 254 inclusively.

## [Interaction settings](interaction.md)

### [interaction_gain_hurdle](interaction.md#interaction_gain_hurdle)

The interaction hurdle. While a tree grows, a split on a feature the tree does not use yet
raises the tree's interaction order. Such a split is selected only if its gain is large enough
relative to the gain of the tree's first split, and if it beats the best split on a feature the
tree already uses. Otherwise the tree keeps refining the features it already has. This is soft
heredity: interactions are admitted only on real evidence. A split that would raise the tree's
order above 3 faces a doubled hurdle.

### [interaction_gain_hurdle_mode](interaction.md#interaction_gain_hurdle_mode)

How [`interaction_gain_hurdle`](interaction.md#interaction_gain_hurdle) is applied.

### [table_budget_cells](interaction.md#table_budget_cells)

The cell budget of the table-size prior, which steers the trees toward smaller tables.

### [table_budget_order_shrink](interaction.md#table_budget_order_shrink)

How much [`table_budget_cells`](interaction.md#table_budget_cells) shrinks for each interaction order above 3.
A $k$-way table with $k > 3$ is measured against
$\frac{budget}{table\_budget\_order\_shrink^{k - 3}}$ cells, because a table with more axes is
harder to read at the same number of cells. 1 turns the shrinking off.

## [Categorical features settings](categorical.md)

### [categorical_features](categorical.md#categorical_features)

The features to treat as categorical.

For a polars `DataFrame` or `LazyFrame`, `String`, `Categorical` and `Enum` columns are
categorical automatically, and this parameter is only needed to treat a numeric column as
categorical. For other input types, every feature is numerical unless it is listed here.

### [unknown_category](categorical.md#unknown_category)

How a categorical value that is absent from the training data is scored.

### [cat_smooth](categorical.md#cat_smooth)

The shrinkage strength $m$ of the target statistic of a level toward the mean target of the
whole dataset.

### [cat_target](categorical.md#cat_target)

The transformation of the target before the target statistic is calculated.

### [cat_leakage](categorical.md#cat_leakage)

The method used to keep an object's own target out of the target statistic it is trained on.
At prediction time, the statistics calculated on the whole training dataset are always used.

### [cat_k](categorical.md#cat_k)

The number of folds of the `kfold` method of [`cat_leakage`](categorical.md#cat_leakage).

### [cat_n_perms](categorical.md#cat_n_perms)

The number of random orders of the `ordered` method of [`cat_leakage`](categorical.md#cat_leakage).

### [cat_min_data_per_group](categorical.md#cat_min_data_per_group)

The minimum total weight (sample weight times exposure) of a level. Levels below it are
collapsed into one shared `"<rare>"` level before the encoding.

### [cat_direct_max_levels](categorical.md#cat_direct_max_levels)

The maximum number of levels of a low-cardinality feature. A feature with between 3 and this
many levels (after rare levels are pooled) skips the cross-fitting and the shrinkage: each level
keeps its target statistic calculated on the whole training dataset and gets its own bin.
Binary features always use the regular path. 0 turns this off.

### [cat_channels](categorical.md#cat_channels)

The numerical features (channels) built from each categorical feature.

### [cat_count_min_levels](categorical.md#cat_count_min_levels)

The minimum number of levels (after rare levels are pooled) a feature must have to get the
`count` channel of [`cat_channels`](categorical.md#cat_channels). Features with fewer levels behave as if the
channel was not requested. 0 gives the channel to every categorical feature.

### [cat_class_freq_min_levels](categorical.md#cat_class_freq_min_levels)

The minimum number of levels (after rare levels are pooled) a feature must have to get the
`class_freq` channels of [`cat_channels`](categorical.md#cat_channels). Features with fewer levels keep the
target statistic. Used only for multiclassification.

## [Bagging settings](bagging.md)

### [n_bags](bagging.md#n_bags)

The number of bags. Each bag is trained on its own sample of the objects, with its own
overfitting detector, and the bags are averaged into one model. Training costs about `n_bags`
times as much as training a single model.

### [bag_subsample](bagging.md#bag_subsample)

The fraction of the objects sampled for each bag. Only used when [`n_bags`](bagging.md#n_bags) is greater
than 1.

### [cell_refit_base](bagging.md#cell_refit_base)

The base penalty of the out-of-bag cell refit. After bagging, every cell of the averaged tables
is refit toward the residuals of the bags' out-of-bag objects under a ridge penalty shaped by
[`cell_refit_gamma`](bagging.md#cell_refit_gamma), and the tables are purified again. The refit is kept
only if it improves the held-out deviance.

### [cell_refit_gamma](bagging.md#cell_refit_gamma)

The adaptive exponent of the penalty of the out-of-bag cell refit. 0 applies the same ridge
penalty to every cell; larger values penalize cells with a strong signal less. Only used when
[`cell_refit_base`](bagging.md#cell_refit_base) is set.

## [Pruning settings](pruning.md)

### [prune](pruning.md#prune)

Select the tables to deploy after the training, and deploy the smaller set.

### [prune_main_effects](pruning.md#prune_main_effects)

Allow pruning to drop main effects too.

`False` deploys every main effect the training built and prunes interactions only. `True` makes
the main effects candidates as well, under hierarchy: a main effect is dropped only when it does
not earn its place and no kept interaction contains it, so a kept interaction always keeps its
main effects. A feature whose main effect is dropped, and which no kept interaction uses, no
longer affects predictions.

### [prune_selector](pruning.md#prune_selector)

The method used to select the tables.

### [prune_path_fraction](pruning.md#prune_path_fraction)

The fraction of the out-of-bag improvement the deployed tables must capture. The improvement is
measured from the model with main effects only (the intercept-only model with
`prune_main_effects=True`) to the prefix with the lowest out-of-bag deviance.

### [prune_path_tolerance](pruning.md#prune_path_tolerance)

The maximum relative excess of the deployed prefix's out-of-bag deviance over the best prefix's.
The deployed prefix is the larger of the smallest prefix that captures
[`prune_path_fraction`](pruning.md#prune_path_fraction) of the improvement and the smallest prefix within
this tolerance.

### [prune_path_steps](pruning.md#prune_path_steps)

The number of prefixes scored on the ranked path. The prefix sizes are spaced geometrically
between one table and all the candidate tables. Used only by the `ranked_path` selector.

### [prune_guard_min_rows](pruning.md#prune_guard_min_rows)

The minimum number of objects with honest evidence needed to judge the tables on.

### [prune_rebalance](pruning.md#prune_rebalance)

After tables are dropped, re-solve the values of the remaining tables for the smaller structure
(one ridge IRLS step, followed by purification). The new values are kept only if they lower the
held-out deviance. Models with multi-channel categorical features (see
[`cat_channels`](categorical.md#cat_channels)) are not rebalanced.

### [prune_table_budget](pruning.md#prune_table_budget)

The maximum number of deployed tables of order
[`prune_table_min_arity`](pruning.md#prune_table_min_arity) or higher.

### [prune_table_min_arity](pruning.md#prune_table_min_arity)

The lowest interaction order counted by [`prune_table_budget`](pruning.md#prune_table_budget) and by
[`prune_lambda_tables`](pruning-fold-vote.md#prune_lambda_tables). The default counts three-way
and higher tables, so main effects and pairs are never limited. Allowed values are integers
from 1 to 8.

### [prune_box_budget](pruning.md#prune_box_budget)

The maximum total number of boxes in the deployed model.

An effect whose dense table would be too large is stored in factored form, as a sum of rank-one
boxes (regions of the feature space), and each box is one row of its exported rating table. A
deeper tree contributes more boxes, so `max_depth` mostly multiplies the number of boxes rather
than the number of tables.

## [Fold vote pruning settings](pruning-fold-vote.md)

### [prune_n_folds](pruning-fold-vote.md#prune_n_folds)

The number of cross-validation folds.

### [prune_fold_min_rows](pruning-fold-vote.md#prune_fold_min_rows)

The minimum number of objects in each fold (see [`prune_n_folds`](pruning-fold-vote.md#prune_n_folds)). Raising it
makes the number of folds adapt down earlier.

### [prune_fold_es_patience](pruning-fold-vote.md#prune_fold_es_patience)

The number of iterations the overfitting detector of the fold models waits after the iteration
with the optimal metric value. The fold models only vote on which tables to keep, so they use a
shorter patience than the deployed model, which always uses
[`early_stopping_rounds`](overfitting-detection.md#early_stopping_rounds).

### [prune_min_stability](pruning-fold-vote.md#prune_min_stability)

The fraction of the folds in which a table must show signal to be kept. A table is kept when its
mean gain exceeds [`prune_min_mean_gain`](pruning-fold-vote.md#prune_min_mean_gain) and either it was kept by at least
this fraction of the folds' own selections, or its gain was positive in at least this fraction of
the folds.

### [prune_min_mean_gain](pruning-fold-vote.md#prune_min_mean_gain)

The minimum mean gain a table must exceed to be kept, in units of deviance per unit of weight.
0 keeps every table with a positive mean gain.

### [prune_drop_z](pruning-fold-vote.md#prune_drop_z)

The evidence bar for dropping a table. A table the vote would drop is kept instead unless its
gain is significantly negative: its mean over the folds must be below $-z \cdot SE$, where $z$ is
the value of this parameter and $SE$ the standard error of the mean. Tables kept this way are
ranked by mean gain and limited by [`prune_keep_budget`](pruning-fold-vote.md#prune_keep_budget). Tables scored by
fewer than two folds keep the vote's verdict.

### [prune_keep_budget](pruning-fold-vote.md#prune_keep_budget)

The maximum number of tables after the evidence gate ([`prune_drop_z`](pruning-fold-vote.md#prune_drop_z)) adds its
tables: the limit is the larger of this value and the number of tables the vote kept. A set that
the vote already made larger than the budget is left as the vote chose it.

### [prune_lambda_tables](pruning-fold-vote.md#prune_lambda_tables)

_Alias:_ `prune_size_penalty`

The price of one kept table of order
[`prune_table_min_arity`](pruning.md#prune_table_min_arity) or higher, in units of held-out
deviance. The selection minimizes the held-out deviance plus this price times the number of
such tables. 0 turns it off.

### [prune_size_penalty](pruning-fold-vote.md#prune_size_penalty)

An alias of [`prune_lambda_tables`](pruning-fold-vote.md#prune_lambda_tables). Setting both to different values
raises an error.

### [prune_lambda_boxes](pruning-fold-vote.md#prune_lambda_boxes)

The price of one deployed box (see [`prune_box_budget`](pruning.md#prune_box_budget)), in units
of held-out deviance per unit of weight. The selection minimizes the held-out deviance plus this
price times the number of boxes, so a large set of tables has to earn its size. Dense tables cost
no boxes. 0 turns it off.

### [prune_fold_fidelity](pruning-fold-vote.md#prune_fold_fidelity)

Score every candidate table in every fold. A fold model searches its own structure, so it may not
build some of the tables the deployed model has, and those tables then get no evidence from that
fold. With this parameter, the missing tables are given values in each fold by a ridge fit on the
fold's training objects, so every candidate gets a held-out gain.

### [prune_guard](pruning-fold-vote.md#prune_guard)

Check the selected set of tables as a whole. The selection judges tables one at a time, so it can
drop a group of correlated tables that only matter together. The guard compares the deviance of
the selected set with that of the full set on honest objects (the out-of-bag objects, or a
shared holdout for grouped data). If the relative gap exceeds the tolerance, dropped tables are
added back, best evidence first, until it does not.

### [prune_guard_tol](pruning-fold-vote.md#prune_guard_tol)

The tolerance of the [no-harm guard](pruning-fold-vote.md#prune_guard): the largest acceptable relative increase of
the deviance of the selected set over the full set.

### [prune_guard_z](pruning-fold-vote.md#prune_guard_z)

The multiplier of the standard error that can raise the guard's tolerance above
[`prune_guard_tol`](pruning-fold-vote.md#prune_guard_tol). 0 uses the fixed tolerance.

### [prune_guard_z_dn](pruning-fold-vote.md#prune_guard_z_dn)

The multiplier of the standard error that can lower the guard's tolerance below
[`prune_guard_tol`](pruning-fold-vote.md#prune_guard_tol) when the evidence is precise. It can only make the guard
act more often, never less. 0 turns it off.

### [prune_guard_tol_floor](pruning-fold-vote.md#prune_guard_tol_floor)

The lowest tolerance [`prune_guard_z_dn`](pruning-fold-vote.md#prune_guard_z_dn) can lower the guard's tolerance to,
so that very precise evidence cannot drive it to zero. Ignored when `prune_guard_z_dn` is 0.

### [prune_slope_eps](pruning-fold-vote.md#prune_slope_eps)

The dead band of the post-pruning slope correction. For a `poisson`, `gamma` or `tweedie` model
trained without an exposure, the scale $b$ of the pruned model's score is estimated on the
out-of-bag objects after the guard. The score is rescaled only if $|b - 1|$ exceeds this value
and $b$ differs from 1 by at least [`prune_slope_min_z`](pruning-fold-vote.md#prune_slope_min_z) standard errors.
The outcome is recorded in `pruning_report_["slope"]`.

### [prune_slope_min_z](pruning-fold-vote.md#prune_slope_min_z)

The number of standard errors by which the scale $b$ must differ from 1 for the post-pruning
slope correction to be applied (see [`prune_slope_eps`](pruning-fold-vote.md#prune_slope_eps)).

## [Banding settings](banding.md)

### [band_tolerance](banding.md#band_tolerance)

The tolerance of banding, as a multiple of the noise between the bags. The mean squared change
of the predictions caused by banding is held within $(band\_tolerance \cdot \sigma)^2$, and
within [`band_deviance_cap`](banding.md#band_deviance_cap). Larger values give coarser bands.

### [band_deviance_cap](banding.md#band_deviance_cap)

The maximum cost of banding, as a fraction of the model's training deviance. The noise
tolerance alone could let a very noisy model move far; this cap bounds what that can cost.

## [Graduation settings](graduation.md)

### [graduate](graduation.md#graduate)

Smooth the deployed tables with Whittaker-Henderson graduation.

### [graduation_alpha](graduation.md#graduation_alpha)

A fixed smoothing strength for every table, instead of the strength each table picks by
generalized cross-validation. 0 turns off the smoothing of the dense tables, which is useful
together with [`graduation_high_order_alpha`](graduation.md#graduation_high_order_alpha).

### [graduation_high_order_alpha](graduation.md#graduation_high_order_alpha)

The strength, in the range $[0; 1]$, of an additional neighbour smoothing step for effects of
order 3 to 8 that are stored in factored form. Unlike the smoothing of the dense tables, its
strength is fixed rather than selected by cross-validation.

## [Purification settings](purification.md)

### [ref_measure](purification.md#ref_measure)

The reference measure the tables are purified against.

### [measure_floor](purification.md#measure_floor)

The total mass of the floor of the `exposure` reference measure, as a fraction of the
training data's mass. It is spread evenly over the cells of each axis so that every weight is
strictly positive. The value must be finite and positive. Used only with
`ref_measure="exposure"`.

## [Multiclassification settings](multiclassification.md)

### [multiclass_prune_cv](multiclassification.md#multiclass_prune_cv)

The selection regime for multiclassification.

### [multiclass_prune_guard](multiclassification.md#multiclass_prune_guard)

Check the selected set of tables as a whole against the full set on honest objects, and add
dropped tables back while the gap exceeds the tolerance (see
[`multiclass_prune_guard_floor`](multiclassification.md#multiclass_prune_guard_floor)).

### [multiclass_prune_guard_floor](multiclassification.md#multiclass_prune_guard_floor)

The floor of the guard's tolerance, as a fraction of the improvement of the full model's
deviance over the class prior.

### [multiclass_prune_sel_bags](multiclassification.md#multiclass_prune_sel_bags)

The number of bags of the models the selection is trained on (the fold models, or the selection
model of the single-split regime). Must be at least 1. Any other value than 1 raises an error for
regression and binary classification.

### [prune_validation_fraction](multiclassification.md#prune_validation_fraction)

The fraction of the objects used to select the tables in the single-split regime
(`multiclass_prune_cv=False`). Any other value than the default raises an error in the other
regimes, and for regression and binary classification.

### [prune_se_rule](multiclassification.md#prune_se_rule)

The selection rule, in standard errors of the held-out deviance estimate. 0 selects the set with
the lowest held-out deviance; larger values (for example, the classic one-standard-error rule,
1.0) prune more aggressively and trade deviance for a smaller model.

## [Performance settings](performance.md)

### [n_jobs](performance.md#n_jobs)

The number of threads to use during the training and prediction.

### [hist_precision](performance.md#hist_precision)

The precision of the gradient histograms used to search for splits.

### [refine_closed_form_tier2](performance.md#refine_closed_form_tier2)

Calculate the leaf refinement steps (see
[`leaf_refine_steps`](common.md#leaf_refine_steps)) of a `poisson` model in closed form instead
of with an exact line search. This is faster, and the leaf values differ from the exact path by
about $10^{-7}$. `False` uses the exact path.

### [incremental_mu](performance.md#incremental_mu)

Update the predicted mean $\mu = e^{a}$ of a `poisson` model incrementally from one iteration to
the next instead of recalculating it. This is faster, and the predictions differ from the exact
path by about $10^{-9}$ to $10^{-6}$.

## [Advanced settings](advanced.md)

### [lambda_scale_invariant](advanced.md#lambda_scale_invariant)

Rescale [`lambda_`](common.md#lambda_) by the mean Hessian per object of each iteration instead
of using it as it is.

### [ridge_refit_l2](advanced.md#ridge_refit_l2)

The L2 penalty of a fully corrective refit. After the trees are built, all their leaf values are
re-solved jointly by regularized IRLS, with the tree structures fixed.

### [ridge_refit_max_iter](advanced.md#ridge_refit_max_iter)

The maximum number of IRLS iterations of the fully corrective refit. Only used when
[`ridge_refit_l2`](advanced.md#ridge_refit_l2) is set.

### [dart_drop_rate](advanced.md#dart_drop_rate)

Turns on DART (Dropout Additive Regression Trees): at each iteration, the trees built so far are
dropped with this probability before the new tree is built, and the tree weights are normalized
as in DART. The value must be in the range $[0; 1)$.

### [nesterov](advanced.md#nesterov)

Reserved for Nesterov-accelerated boosting, which is not implemented yet. Setting it to `True`
raises an error.

### [prune_refit_full](advanced.md#prune_refit_full)

Deprecated, and without effect: the deployed model is always trained on all objects. Setting it
to `True` raises an error. It will be removed in a future release.
