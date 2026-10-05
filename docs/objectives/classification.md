# Classification: objectives and metrics

- [Objectives and metrics](#objectives-and-metrics)
- [Used for optimization](#usage-information)

## Objectives and metrics {#objectives-and-metrics}

The formulas use the [variables](variables.md) common to all metrics.

### logistic {#logistic}

$$
\displaystyle\frac{ - \sum\limits_{i=1}^N 2 w_{i}\left(c_i \log(p_{i}) + (1-c_{i}) \log(1 - p_{i})\right)}{\sum\limits_{i = 1}^{N} w_{i}}
$$

where $c_i$ is 1 if the i-th object belongs to the second class (`classes_[1]`) and 0
otherwise. This is the binomial deviance: twice the log loss.

`TBoostClassifier` uses this objective when the target has exactly two distinct values. The
class labels can be of any type; they are sorted, and `classes_[1]` is the positive class.

## Used for optimization {#usage-information}

| Name | Optimization | Link |
|------|--------------|------|
| [logistic](#logistic) | + | logit |
