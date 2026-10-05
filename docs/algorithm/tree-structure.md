# Choosing the tree structure

The trees are symmetric (oblivious): every level of a tree applies one split, a pair of a feature
and a border, shared by all the nodes of the level. A tree of depth $d$ therefore has $2^{d}$
leaves, and the leaf of an object is determined by $d$ comparisons.

This is a greedy method. The tree is built level by level, and one split is selected for each
level:

1. A list is formed of possible candidates (<q>feature-split pairs</q>): every border of every
   feature that the level is allowed to use. Missing values are tried on both sides of each
   candidate, and the better side is kept.
1. Each candidate is scored by the decrease of the loss it gives, summed over all the leaves of
   the current level:

    $$
    score = \sum\limits_{leaves} \frac{1}{2}\left(\frac{G_{L}^{2}}{H_{L} + \lambda} + \frac{G_{R}^{2}}{H_{R} + \lambda} - \frac{G^{2}}{H + \lambda}\right) { , where}
    $$

    - $G$ and $H$ are the sums of the gradients and the Hessians of the objects in a leaf.
    - $G_L, H_L$ and $G_R, H_R$ are the same sums on the two sides of the split.
    - $\lambda$ is the value of [`lambda_`](../training-parameters/common.md#lambda_).

1. The split with the highest score is selected. If no candidate scores above
   [`min_split_gain`](../training-parameters/common.md#min_split_gain), the tree stops growing.

The procedure is repeated until the tree reaches [`max_depth`](../training-parameters/common.md#max_depth)
levels.

## Constraints on the features of a tree {#constraints}

A tree may use at most [`max_interaction_order`](../training-parameters/common.md#max_interaction_order)
distinct features (3 by default). Once a tree uses that many, its remaining levels can only split
again on the features it already uses. This is what bounds the interaction order of the model,
and so the number of axes of its rating tables.

Within that limit, a split on a new feature raises the tree's interaction order, so it must pass
the [interaction hurdle](../training-parameters/interaction.md#interaction_gain_hurdle): its gain
must be large enough relative to the gain of the tree's first split, and it must beat the best
split on a feature the tree already uses. Otherwise the tree keeps refining the features it
already has.

The score of a candidate can also be adjusted by:

- the [table-size prior](../training-parameters/interaction.md#table_budget_cells), which favours
  splits that keep the tables small;
- [monotonic constraints](../training-parameters/common.md#monotone_constraints);
- [`random_strength`](../training-parameters/common.md#random_strength), which adds noise to the
  ranking of the candidates.

Before each tree is built, a fraction of the features is sampled
([`colsample_bytree`](../training-parameters/common.md#colsample_bytree)), and optionally a
sample of the objects is drawn to score the candidates
([`subsample`](../training-parameters/common.md#subsample)).
