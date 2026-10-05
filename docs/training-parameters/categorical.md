# Categorical features settings

These parameters control how categorical features are transformed to numerical features. See
[Categorical features](../features/categorical-features.md) and
[Transforming categorical features to numerical features](../algorithm/categorical-features.md)
for details.

## categorical_features {#categorical_features}

#### Description

The features to treat as categorical.

For a polars `DataFrame` or `LazyFrame`, `String`, `Categorical` and `Enum` columns are
categorical automatically, and this parameter is only needed to treat a numeric column as
categorical. For other input types, every feature is numerical unless it is listed here.

Supported formats:

- A feature name or a zero-based feature index.
- A list of feature names or indices.
- A boolean mask with one value per feature.

**Type**

- string
- int
- list
- numpy.ndarray

**Default value**

None (categorical features are detected from the polars column types only)

## unknown_category {#unknown_category}

#### Description

How a categorical value that is absent from the training data is scored.

Possible values:

- `rare` — The value is scored exactly as a level that was pooled into the `"<rare>"` level,
  in every table that uses the feature. If the fit pooled no levels, the value is scored in the
  axis's default cell.
- `default_cell` — The value is always scored in the axis's default cell (the encoder's base
  level).
- `error` — Every scoring call (`predict`, `predict_proba`, `predict_raw`,
  `decision_function`, `predict_contributions`, `tables`, `cell_indices`,
  `actual_vs_expected`) raises `ValueError`, naming the feature and up to five unseen values.

The objects of an `eval_set` follow the same rule. A missing value is never unseen: it is scored
in the missing level.

**Type**

string

**Default value**

rare

## cat_smooth {#cat_smooth}

#### Description

The shrinkage strength $m$ of the target statistic of a level toward the mean target of the
whole dataset.

The target statistic of level $c$ is

$$
ctr_{c} = \frac{n_{c} \cdot \bar{y}_{c} + m \cdot \bar{y}}{n_{c} + m}
$$

`None` estimates $m$ from the data for each feature (an empirical Bayes credibility estimate).

**Type**

float

**Default value**

None (estimated for each feature)

## cat_target {#cat_target}

#### Description

The transformation of the target before the target statistic is calculated.

Possible values:

- `mean` — The weighted mean of the target per unit of exposure.
- `log_mean` — The weighted mean of $\log(t)$. Requires a positive target, and suits
  heavy-tailed targets such as claim severity.

**Type**

string

**Default value**

None (`mean`)

## cat_leakage {#cat_leakage}

#### Description

The method used to keep an object's own target out of the target statistic it is trained on.
At prediction time, the statistics calculated on the whole training dataset are always used.

Possible values:

- `kfold` — The objects are split into [`cat_k`](#cat_k) folds, and each fold is encoded with
  statistics calculated on the other folds.
- `ordered` — Each object is encoded with statistics calculated on the objects before it in a
  random order. [`cat_n_perms`](#cat_n_perms) random orders are used.
- `loo` — Each object is encoded with the statistics of its level with the object itself
  removed.

**Type**

string

**Default value**

None (`kfold`)

## cat_k {#cat_k}

#### Description

The number of folds of the `kfold` method of [`cat_leakage`](#cat_leakage).

**Type**

int

**Default value**

5

## cat_n_perms {#cat_n_perms}

#### Description

The number of random orders of the `ordered` method of [`cat_leakage`](#cat_leakage).

**Type**

int

**Default value**

1

## cat_min_data_per_group {#cat_min_data_per_group}

#### Description

The minimum total weight (sample weight times exposure) of a level. Levels below it are
collapsed into one shared `"<rare>"` level before the encoding.

**Type**

float

**Default value**

10.0

## cat_direct_max_levels {#cat_direct_max_levels}

#### Description

The maximum number of levels of a low-cardinality feature. A feature with between 3 and this
many levels (after rare levels are pooled) skips the cross-fitting and the shrinkage: each level
keeps its target statistic calculated on the whole training dataset and gets its own bin.
Binary features always use the regular path. 0 turns this off.

**Type**

int

**Default value**

16

## cat_channels {#cat_channels}

#### Description

The numerical features (channels) built from each categorical feature.

Possible values:

- `None` or `["mean"]` — The target statistic only.
- `["mean", "count"]` — The target statistic and a target-free statistic of how common the
  level is (the logarithm of its share of the total weight). The second channel lets the trees
  separate rare but informative levels from common ones, which can improve the accuracy on
  high-cardinality features. See [`cat_count_min_levels`](#cat_count_min_levels).
- `["class_freq"]` (multiclassification only) — One channel per class, holding the level's
  frequency of that class, instead of the target statistic. Add `"mean"` to keep the target
  statistic as well. See [`cat_class_freq_min_levels`](#cat_class_freq_min_levels).

However many channels a feature has, the rating tables show one relativity per level of the
original feature.

**Type**

list of strings

**Default value**

None (`["mean"]`)

## cat_count_min_levels {#cat_count_min_levels}

#### Description

The minimum number of levels (after rare levels are pooled) a feature must have to get the
`count` channel of [`cat_channels`](#cat_channels). Features with fewer levels behave as if the
channel was not requested. 0 gives the channel to every categorical feature.

**Type**

int

**Default value**

20

## cat_class_freq_min_levels {#cat_class_freq_min_levels}

#### Description

The minimum number of levels (after rare levels are pooled) a feature must have to get the
`class_freq` channels of [`cat_channels`](#cat_channels). Features with fewer levels keep the
target statistic. Used only for multiclassification.

**Type**

int

**Default value**

3
