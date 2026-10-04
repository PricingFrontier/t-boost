# t-boost

**A Tabulating Boosting Machine (TBM): gradient boosting whose fitted model is *exactly* a set of
rating tables.**

Documentation: <https://pricingfrontier.github.io/t-boost/>

## Why

Gradient-boosted trees are usually more accurate than a GLM, but they are hard to read, review or
deploy in systems built around rating tables. t-boost aims to get boosting-level accuracy in a
model that *is* a set of main-effect and interaction tables, with no approximation and no
surrogate model.

## How it works

- **Constrained trees.** Each tree is symmetric (oblivious): every level applies one shared
  `(feature, threshold)` split. Each tree may use only a few distinct features, so the whole
  ensemble has a fixed maximum interaction order (up to 8th order).
- **Exact decomposition.** Because of that structure, the trained ensemble can be rewritten as a
  functional-ANOVA (fANOVA) decomposition: one table per main effect and per interaction. The
  tables reproduce the model's predictions exactly, to floating-point tolerance.
- **Purification.** The tables are centred on the training data (exposure-weighted when
  an exposure is given), so each main effect carries as much of the signal as it can.
- **The tables are the model.** A saved model is stored as its rating tables, so what you review
  is exactly what gets deployed.

It has a Rust core with Python bindings, takes polars DataFrames directly and is deterministic.

### Keeping the tables readable

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

#### 1. Interaction hurdle

While a tree grows, a split that brings in a new feature raises the tree's interaction order. That
split must earn enough gain relative to the tree's first (main-effect) split, and must beat the
best split on a feature the tree already uses. Otherwise the tree keeps refining features it
already has. This is soft heredity: interactions are admitted only on real evidence.

In the default `"adaptive"` mode the hurdle starts at full strength and relaxes as main-effect gains
fade. Three-way admissions face a stricter bar than two-way ones. `"fixed"` applies the scalar as
given, and `interaction_gain_hurdle=0.0` restores plain greedy splitting.

#### 2. Pruning

After the fit, the interaction tables are ranked by their purified variance and added back in that
order, subject to heredity: a k-way table enters only once all its (k-1)-way sub-tables are in.
Each prefix is scored on the bags' out-of-bag rows. The deployed set is the smallest prefix that
captures 99.5% of the available improvement over the main-effects-only model and is within 0.1% of
the best out-of-bag deviance. Main effects are kept by default.

`prune_main_effects=True` puts the main effects on the path too. The path then starts from the
intercept-only model, so the 99.5% is measured from there. A main effect enters at its own rank, or
just before the first interaction that contains it, so a kept interaction always keeps its main
effects. A feature whose main effect is dropped, and which no kept interaction uses, no longer
affects predictions.

A fit without out-of-bag rows (for example `n_bags=1`) falls back to a K-fold cross-validated
vote (`prune_n_folds`), which judges main effects the same way when `prune_main_effects=True`. The
selection is recorded in `pruning_report_`. `prune=False` deploys the full, unpruned table bank
instead.

#### 3. Banding

Each surviving interaction table is condensed into a small product grid of bands. Adjacent cells
are merged where the model barely distinguishes them, cheapest merge first. Every table that
contains a feature cuts it at the same nested places, so bands line up across tables. Missing values
always keep their own band.

How coarse the bands get is set by the model's own noise. The prediction change from banding is
held within `(band_tolerance × σ)²`, where σ is the spread between bags. It is also capped at
`band_deviance_cap` (0.1%) of the training deviance. Banding needs bagging to measure σ. Its
report is in `pruning_report_["banding"]`, and `band_tolerance=None` turns it off.

#### 4. Graduation

Finally, the tables are smoothed with Whittaker-Henderson graduation, the actuarial smoother for
rating factors. Each table picks its own strength by generalized cross-validation, so a table whose
roughness is real shape is left untouched. No rows are held out for it.

`graduation_alpha` fixes one strength for every table. `graduation_high_order_alpha` (off by
default) adds a light neighbour smoothing for factored 3-way and higher interactions. Details are in
`graduation_report_`. `graduate=False` ships the unsmoothed bank. Graduation is skipped for
monotone-constrained fits and is not supported for 3+ class models.

## Install

```bash
uv add t-boost
```

To build from source you need a Rust toolchain; run `uv sync`.

## Quickstart

```python
import polars as pl
from t_boost import TBoostClassifier, TBoostRegressor

train = pl.read_parquet("policies.parquet")

# Claim frequency: Poisson with an exposure offset
freq = TBoostRegressor(objective="poisson").fit(
    train.select(FEATURES + ["ClaimCount", "Exposure"]),
    "ClaimCount",              # target, by column name
    exposure="Exposure",
)
rate = freq.predict(test)

# Classification: binary, or softmax for 3+ classes
clf = TBoostClassifier().fit(train.select(FEATURES + ["Lapsed"]), "Lapsed")
proba = clf.predict_proba(test)
```

Categorical columns are encoded automatically. At prediction time, columns are matched by name.

## Rating tables and explanations

```python
import json
tables = json.loads(freq.tables(train))                  # the fANOVA rating tables

freq.predict_contributions(test.head(5))                 # per-prediction breakdown by table
freq.feature_importances_                                # share of variance per feature
freq.actual_vs_expected(train, "ClaimCount", exposure="Exposure")   # A/E by factor level
```

For each prediction, `base_value + sum(contributions)` equals the raw (link-scale) score, so the
explanation is exact rather than estimated. The output format matches rustystats'
`GLMModel.predict_contributions`.

## Saving and loading

```python
with open("freq.tboost", "wb") as f:
    f.write(freq.to_bytes())

with open("freq.tboost", "rb") as f:
    loaded = TBoostRegressor.from_bytes(f.read())
```

`to_json()` / `from_json()` give a diffable format. A loaded model predicts identically to the
original.

## Objectives

| Objective | Use | Link |
|-----------|-----|------|
| `squared_error` | regression | identity |
| `logistic` | binary classification | logit |
| softmax (automatic for 3+ classes) | multiclass | softmax |
| `poisson` | claim frequency / counts | log |
| `gamma` | severity | log |
| `tweedie` | pure premium | log |

## License

Apache-2.0
