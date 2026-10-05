# Pruning, banding and graduation

A default fit runs four steps that keep the rating tables few, small and smooth. Each one can be
tuned or switched off.

```python
TBoostRegressor(
    objective="poisson",
    interaction_gain_hurdle=2.0,            # 1. interaction hurdle (0.0 = off)
    interaction_gain_hurdle_mode="adaptive",
    prune=True,                             # 2. pruning
    prune_main_effects=False,               #    (True = main effects can be dropped too)
    band_tolerance=0.75,                    # 3. banding (None = off)
    band_deviance_cap=0.001,
    graduate=None,                          # 4. graduation (False = off)
)
```

## Interaction hurdle {#interaction-hurdle}

While a tree grows, a split that brings in a new feature raises the tree's interaction order.
That split must earn enough gain relative to the tree's first (main-effect) split, and must beat
the best split on a feature the tree already uses. Otherwise the tree keeps refining features it
already has. This is soft heredity: interactions are admitted only on real evidence.

In the default `"adaptive"` mode the hurdle starts at full strength and relaxes as main-effect
gains fade. Three-way admissions face a stricter bar than two-way ones. `"fixed"` applies the
scalar as given, and `interaction_gain_hurdle=0.0` turns the hurdle off. See
[Interaction settings](../training-parameters/interaction.md).

## Pruning {#pruning}

After the fit, the interaction tables are ranked by their purified variance and added back in
that order, subject to heredity: a $k$-way table enters only once all its $(k-1)$-way sub-tables
are in. Each prefix is scored on the bags' out-of-bag objects. The deployed set is the smallest
prefix that captures 99.5% of the available improvement over the main-effects-only model and is
within 0.1% of the best out-of-bag deviance. Main effects are kept by default.

`prune_main_effects=True` puts the main effects on the path too. The path then starts from the
intercept-only model, so the 99.5% is measured from there. A main effect enters at its own rank,
or just before the first interaction that contains it, so a kept interaction always keeps its
main effects. A feature whose main effect is dropped, and which no kept interaction uses, no
longer affects predictions.

A fit without out-of-bag objects (for example `n_bags=1`) falls back to a K-fold
cross-validated vote, which judges main effects the same way when `prune_main_effects=True`. The
selection is recorded in `pruning_report_`. `prune=False` deploys the full, unpruned set of
tables instead. See [Pruning settings](../training-parameters/pruning.md).

## Banding {#banding}

Each surviving interaction table is condensed into a small product grid of bands. Adjacent cells
are merged where the model barely distinguishes them, cheapest merge first. Every table that
contains a feature cuts it at the same nested places, so bands line up across tables. Missing
values always keep their own band.

How coarse the bands get is set by the model's own noise. The prediction change from banding is
held within $(band\_tolerance \cdot \sigma)^2$, where $\sigma$ is the spread between bags. It is
also capped at `band_deviance_cap` (0.1%) of the training deviance. Banding needs bagging to
measure $\sigma$. Its report is in `pruning_report_["banding"]`, and `band_tolerance=None` turns
it off. See [Banding settings](../training-parameters/banding.md).

## Graduation {#graduation}

Finally, the tables are smoothed with Whittaker-Henderson graduation, the actuarial smoother for
rating factors. Each table picks its own strength by generalized cross-validation, so a table
whose roughness is real shape is left untouched. No objects are held out for it.

`graduation_alpha` fixes one strength for every table. `graduation_high_order_alpha` (off by
default) adds a light neighbour smoothing for factored 3-way and higher interactions. Details are
in `graduation_report_`. `graduate=False` ships the unsmoothed tables. Graduation is skipped for
monotone-constrained fits and is not supported for models with three or more classes. See
[Graduation settings](../training-parameters/graduation.md).
