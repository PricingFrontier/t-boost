# Transforming categorical features to numerical features

t-boost supports the following types of features:

- Numerical. Values of such features can be real numbers and `NaN` (the latter represents
  [missing values](missing-values.md)). Examples are the driver's age (<q>32</q>, <q>57</q>), or
  any binary feature (<q>0</q>, <q>1</q>).
- Categorical (cat). Such features can take one of a limited number of possible values. These
  values are usually fixed. Examples are the vehicle make (<q>Ford</q>, <q>Peugeot</q>) and the
  region (<q>North</q>, <q>South</q>).

Before the trees are built, categorical features are transformed to numerical. Each level is
replaced by a statistic of the target, and the levels are sorted by that statistic, so a split
on the transformed feature groups levels with similar target values.

The method of transforming categorical features to numerical includes the following stages:

1. Pooling rare levels.

    Levels whose total weight (sample weight times exposure) is below `cat_min_data_per_group`
    are collapsed into one shared level, `"<rare>"`. Missing values form a level of their own.

1. Calculating the target statistic of each level.

    The statistic is the level's mean target shrunk toward the mean target of the whole
    dataset:

    $$
    ctr_{c} = \frac{n_{c} \cdot \bar{y}_{c} + m \cdot \bar{y}}{n_{c} + m} { , where}
    $$

    - $n_c$ is the total weight of the objects with level $c$ (sample weight times exposure).
    - $\bar{y}_c$ is their weighted mean target per unit of exposure, $\sum w_i t_i / \sum w_i e_i$.
    - $\bar{y}$ is the same mean over the whole dataset.
    - $m$ is the shrinkage strength. By default it is estimated from the data for each feature
      (an empirical Bayes credibility estimate). Set `cat_smooth` to fix it.

    With `cat_target="log_mean"` the statistic is calculated on $\log(t)$ instead of $t$, which
    suits positive, heavy-tailed targets such as claim severity.

1. Avoiding target leakage.

    If an object's own target contributed to the statistic it is trained on, the trees would
    overfit to it. The training objects are therefore encoded with statistics that exclude
    them. The method is set by `cat_leakage`:

    - `"kfold"` (default) — The objects are split into `cat_k` folds, and each fold is encoded
      with statistics calculated on the other folds.
    - `"ordered"` — The objects are put in a random order, and each object is encoded with
      statistics calculated on the objects before it. `cat_n_perms` random orders are used.
    - `"loo"` — Each object is encoded with the statistics of its level with the object itself
      removed.

    At prediction time, levels are encoded with the statistics calculated on the whole
    training dataset, which are stored with the model.

## Low-cardinality features {#low-cardinality}

A feature with between 3 and `cat_direct_max_levels` levels (16 by default) after rare-level
pooling skips the cross-fitting and the shrinkage. Each level keeps its target statistic
calculated on the whole training dataset and gets its own bin, ordered by that statistic.
Binary features always use the regular path. Set `cat_direct_max_levels=0` to turn this off.

## Additional channels {#channels}

The `cat_channels` parameter adds more numerical features per categorical feature:

- `"count"` — A target-free statistic of how common the level is (the logarithm of its share
  of the total weight). It lets the trees separate rare but informative levels from common
  ones. It is only added for features with at least `cat_count_min_levels` levels.
- `"class_freq"` (multiclassification only) — One channel per class, holding the level's
  frequency of that class.

All channels of a feature collapse into one axis in the rating tables, so a table still shows
one relativity per level of the original feature.
