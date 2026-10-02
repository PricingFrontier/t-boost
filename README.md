# t-boost

**A Tabulating Boosting Machine (TBM): gradient boosting whose fitted model is *exactly* a set of
rating tables.**

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
- **Purification and pruning.** The tables are centred on the training data (exposure-weighted when
  an exposure is given), so each main effect carries as much of the signal as it can. Tables that
  contribute little are pruned.
- **The tables are the model.** A saved model is stored as its rating tables, so what you review
  is exactly what gets deployed.

It has a Rust core with Python bindings, takes polars DataFrames directly and is deterministic.

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
