# Decomposing the trees into rating tables

Because of the [constraints on the trees](tree-structure.md#constraints), the trained ensemble can
be rewritten exactly as a functional ANOVA (fANOVA) decomposition: an intercept plus one table per
main effect and per interaction.

The decomposition includes the following stages:

1. Grouping the trees by their features.

    A tree that uses the features $u$ is a function of those features only: a lookup table on the
    grid formed by the borders it splits on. Adding up all the trees that use the same features
    gives one table per set of features, on the grid formed by all the borders the model uses for
    each feature.

1. Purification.

    A table on the features $u$ can still contain parts that depend on fewer features. Each table
    is centred so that its weighted mean along every axis is zero under the
    [reference measure](../training-parameters/purification.md#ref_measure), and the parts that
    are removed are pushed down to the tables of the subsets of $u$, and finally to the intercept.
    This is done for the highest interaction order first.

The result is

$$
a(x) = f_0 + \sum\limits_{u \in U} f_{u}(x_{u})
$$

where every table $f_u$ has zero weighted mean along each of its axes. The tables reproduce the
model's predictions exactly, to floating-point tolerance, and no surrogate model is fitted. Each
main effect carries as much of the signal as it can, and each interaction carries only what the
main effects cannot express.

An effect whose dense table would be too large is kept in factored form: as the exact sum of the
rank-one boxes the trees contributed, instead of as a dense table.

See [Rating tables](../model-analysis/rating-tables.md) for how to export the tables and how an
object finds its cell.
