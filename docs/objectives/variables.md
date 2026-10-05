# Variables used in formulas

The following common variables are used in formulas of the described metrics:

- $t_{i}$ is the label value for the i-th object (from the input data for training).
- $a_{i}$ is the result of applying the model to the i-th object on the link scale (the raw
  score), including the offset passed to `fit` and, for the log and logit links,
  $\log(e_{i})$.
- $\mu_{i}$ is the predicted mean for the i-th object: $\mu_{i} = a_{i}$ for the identity link
  and $\mu_{i} = e^{a_{i}}$ for the log link. With an exposure, $\mu_i$ is the expected total for
  the object: `predict` returns $\mu_i / e_i$, the rate per unit of exposure.
- $p_{i}$ is the predicted success probability $\left(p_{i} = \frac{1}{1 + e^{-a_{i}}}\right)$.
- $e_{i}$ is the exposure of the i-th object. It is set in the `exposure` parameter of `fit`.
  The default is 1 for all objects.
- $w_{i}$ is the weight of the i-th object. It is set in the `sample_weight` parameter of `fit`.
  The default is 1 for all objects.
- $N$ is the total number of objects.
- $M$ is the number of classes.
- $\rho$ is the value of the [`tweedie_rho`](../training-parameters/common.md#tweedie_rho)
  parameter.
