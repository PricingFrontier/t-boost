# Banding settings

After pruning, each deployed interaction table is condensed into a small product grid of bands.
Adjacent cells are merged where the model barely distinguishes them, cheapest merge first. Every
table that contains a feature cuts it at the same nested places, so the bands line up across
tables. Missing values always keep their own band.

How coarse the bands get is set by the model's own noise: the change to the predictions is held
within $(band\_tolerance \cdot \sigma)^2$, where $\sigma$ is the spread between the bags. Banding
therefore needs bagging (`n_bags` of at least 2) and a pruned model (`prune=True`). The outcome
is recorded in `pruning_report_["banding"]`. See
[Pruning, banding and graduation](../algorithm/readable-tables.md) for details.

## band_tolerance {#band_tolerance}

#### Description

The tolerance of banding, as a multiple of the noise between the bags. The mean squared change
of the predictions caused by banding is held within $(band\_tolerance \cdot \sigma)^2$, and
within [`band_deviance_cap`](#band_deviance_cap). Larger values give coarser bands.

`None` turns banding off.

**Type**

float

**Default value**

0.75

## band_deviance_cap {#band_deviance_cap}

#### Description

The maximum cost of banding, as a fraction of the model's training deviance. The noise
tolerance alone could let a very noisy model move far; this cap bounds what that can cost.

**Type**

float

**Default value**

0.001 (0.1% of the training deviance)
