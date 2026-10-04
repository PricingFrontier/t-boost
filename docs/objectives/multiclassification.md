# Multiclassification: objectives and metrics

- [Objectives and metrics](#objectives-and-metrics)
- [Used for optimization](#usage-information)

## Objectives and metrics {#objectives-and-metrics}

The formulas use the [variables](variables.md) common to all metrics.

### softmax {#softmax}

$$
\displaystyle\frac{-\sum\limits_{i=1}^{N}w_{i}\log\left(\displaystyle\frac{e^{a_{it_{i}}}}{ \sum\limits_{j=0}^{M - 1}e^{a_{ij}}} \right)}{\sum\limits_{i=1}^{N}w_{i}} { ,}
$$

$t \in \{0, ..., M - 1\}$

where $a_{ij}$ is the raw score of the i-th object for class $j$.

`TBoostClassifier` uses this objective automatically when the target has three or more distinct
values (with `objective="logistic"`). The model has one set of rating tables per class: the raw
score of a class is its intercept plus its tables, and the probabilities are the softmax of the
class scores.

!!! note

    Multiclassification does not support `exposure`, graduation, `reanchor`, `reanchor_slope`
    or the parameters listed in [Advanced settings](../training-parameters/advanced.md).

## Used for optimization {#usage-information}

| Name | Optimization | Link |
|------|--------------|------|
| [softmax](#softmax) | + | softmax |
