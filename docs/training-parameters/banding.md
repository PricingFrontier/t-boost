# Banding settings

Each deployed interaction table is condensed into a small product grid of bands.
Adjacent cells are merged where the model barely distinguishes them, cheapest merge first. Every
table that contains a feature cuts it at the same nested places, so the bands line up across
tables. Missing values always keep their own band.

How coarse the bands get is set by the model's own noise: the change to the predictions is held
within $(band\_tolerance \cdot \sigma)^2$, where $\sigma$ is the spread between the bags, and
within the [deviance cap](#band_deviance_cap). With one bag (`n_bags=1`) there is no spread to
measure, so the deviance cap alone bounds the change. Pruned and unpruned (`prune=False`) models
are banded alike. The outcome is recorded in `pruning_report_["banding"]`. See
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

How far banding may move the predictions, as a fraction of the model's training deviance. The
cap is a budget on banding's cross-fitted estimate of the move's expected cost: the deviance it
would add if the unbanded tables were exactly right, estimated with bands fitted on one half of the
rows and the move measured on the other. New data can exceed the estimate when the cap is tight.

The noise tolerance alone could let a very noisy model move far; this cap bounds the move. With
one bag (`n_bags=1`) there is no spread between bags to measure the noise by, so the cap is the
only bound. A model with three or more classes shares the cap between its classes.

The cap does not bound the rise in training deviance, which is often several times larger. The
cap leaves out the first-order term of the move: the banded tables move away from the training
rows' own optimum, which always costs training deviance, while on new data that term can go
either way.

Must be finite and greater than 0. `band_tolerance=None` turns banding off.

**Type**

float

**Default value**

0.001 (0.1% of the training deviance)
