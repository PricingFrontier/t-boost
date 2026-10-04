# Feature importance

The individual importance values for each of the input features.

For each feature, the importance is its share of the variance of the model's raw score: the
share of every table that involves the feature, split equally among the table's features.

??? info "Calculation principles"

    The tables are purified, so under the reference measure they are uncorrelated and the
    variance of the raw score is the sum of the variances of the tables. The share of table $u$
    is its Sobol index:

    $$
    S_{u} = \frac{\sigma^{2}(f_{u})}{\sum\limits_{v \in U} \sigma^{2}(f_{v})}
    $$

    The importance of feature $j$ is the sum of the shares of the tables that involve it, each
    divided by the number of features of the table:

    $$
    feature\_importance_{j} = \sum\limits_{u \ni j} \frac{S_{u}}{|u|}
    $$

    For a multiclassification model the importance is averaged over the classes.

#### Specifics

- Feature importance values are normalized so that the sum of importances of all features is
  equal to 1. This is possible because the values of these importances are always
  non-negative.
- The importances are read from the model and need no data. They depend on the
  [reference measure](../training-parameters/purification.md#ref_measure) the tables are
  purified against.
- The split of interactions is the same as the Shapley split of
  [prediction contributions](contributions.md#by-feature).

The importances are available in the `feature_importances_` attribute, and each table's share in
the `sobol` field of the [exported tables](rating-tables.md).
