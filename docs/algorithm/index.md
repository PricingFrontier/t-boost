# How training is performed

The goal of training is to select the *model* $y$, depending on a set of *features* $x_{i}$,
that best solves the given problem (regression, classification, or multiclassification) for any
input *object*. This model is found by using a *training dataset*, which is a set of objects
with known features and label values. Accuracy is checked on the *validation dataset*, which
has data in the same format as in the training dataset, but it is only used for evaluating the
quality of training (it is not used for training).

t-boost is based on gradient boosted decision trees. During training, a set of decision trees is
built consecutively. Each successive tree is built with reduced loss compared to the previous
trees. The trees are constrained so that the trained model can be rewritten exactly as a set of
rating tables.

The number of trees is controlled by the starting parameters. To prevent overfitting, use the
[overfitting detector](overfitting-detector.md). When it is triggered, trees stop being built.

Building stages for a single tree:

1. Preliminary calculation of splits: every numerical feature is
   [quantized](../training-parameters/quantization.md) into bins once, before the training.
1. (_Optional_) [Transforming categorical features to numerical features](categorical-features.md).
1. [Choosing the tree structure](tree-structure.md).
1. Calculating values in leaves: each leaf gets the Newton step
   $w = -\frac{G}{H + \lambda}$, where $G$ and $H$ are the sums of the gradients and the
   Hessians of the objects in the leaf, optionally refined by further Newton steps (see
   [`leaf_refine_steps`](../training-parameters/common.md#leaf_refine_steps)). The values are
   multiplied by the learning rate before the tree is added to the model.

By default several models (bags) are trained this way, each on its own sample of the objects
(see [Bagging](bagging.md)). Then the model is turned into the deployed set of rating tables:

1. [Decomposing the trees into rating tables](decomposition.md): the averaged trees are
   rewritten exactly as an intercept plus one table per main effect and per interaction.
1. [Pruning, banding and graduation](readable-tables.md): the tables that contribute little are
   dropped, the interaction tables are condensed into bands, and the tables are smoothed.
