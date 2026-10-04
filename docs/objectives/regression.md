# Regression: objectives and metrics

- [Objectives and metrics](#objectives-and-metrics)
- [Used for optimization](#usage-information)

## Objectives and metrics {#objectives-and-metrics}

The formulas use the [variables](variables.md) common to all metrics. Each formula is the
weighted mean of the unit deviance of the distribution.

### squared_error {#squared_error}

$$
\displaystyle\frac{\sum\limits_{i=1}^{N} w_{i} (t_{i} - \mu_{i})^{2}}{\sum\limits_{i=1}^{N} w_{i}}
$$

The identity link: $\mu_i = a_i$.

### poisson {#poisson}

$$
\displaystyle\frac{\sum\limits_{i=1}^{N} 2 w_{i} \left(t_{i} \log\frac{t_{i}}{\mu_{i}} - (t_{i} - \mu_{i})\right)}{\sum\limits_{i=1}^{N} w_{i}}
$$

The log link: $\mu_i = e^{a_i}$. The term $t_i \log\frac{t_i}{\mu_i}$ is 0 when $t_i = 0$.

Labels $t_i$ should be non-negative. Pass the exposure (for example, the policy duration) in the
`exposure` parameter of `fit` rather than dividing the target by it.

### gamma {#gamma}

$$
\displaystyle\frac{\sum\limits_{i=1}^{N} 2 w_{i} \left(\frac{t_{i} - \mu_{i}}{\mu_{i}} - \log\frac{t_{i}}{\mu_{i}}\right)}{\sum\limits_{i=1}^{N} w_{i}}
$$

The log link: $\mu_i = e^{a_i}$.

Labels $t_i$ should be positive.

### tweedie {#tweedie}

$$
\displaystyle\frac{\sum\limits_{i=1}^{N} 2 w_{i} \left(\frac{t_{i}^{2-\rho}}{(1-\rho)(2-\rho)} - \frac{t_{i}\mu_{i}^{1-\rho}}{1-\rho} + \frac{\mu_{i}^{2-\rho}}{2-\rho}\right)}{\sum\limits_{i=1}^{N} w_{i}}
$$

The log link: $\mu_i = e^{a_i}$. $\rho$ is the value of the
[`tweedie_rho`](../training-parameters/common.md#tweedie_rho) parameter, in the range $(1; 2)$.

Labels $t_i$ should be non-negative.

## Used for optimization {#usage-information}

| Name | Optimization | Link | Metric function |
|------|--------------|------|-----------------|
| [squared_error](#squared_error) | + | identity | `mean_tweedie_deviance(power=0)` |
| [poisson](#poisson) | + | log | `mean_poisson_deviance` |
| [gamma](#gamma) | + | log | `mean_gamma_deviance` |
| [tweedie](#tweedie) | + | log | `mean_tweedie_deviance(power=tweedie_rho)` |

The metric functions are in the [`t_boost.metrics`](../python-reference/metrics.md) module. Pass
them the expected totals ($\mu_i$, which is `predict(X) * exposure` for a model trained with an
exposure).
