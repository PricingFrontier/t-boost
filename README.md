# t-boost

**An oblivious gradient-boosting machine that is *exactly* decomposable into functional-ANOVA (fANOVA)
"rating tables" of up to 8th order.**

Every tree is a symmetric (oblivious) tree with one shared `(feature, threshold)` test per level
and a bounded number of distinct raw features; deeper levels may reuse a feature to refine a surface
rather than add a new one, so the trained ensemble truncates at a fixed interaction order. That
structure lets the fitted model be rewritten, losslessly, as a set of
main-effect and interaction tables (with factored box representations for high-order effects) that reproduce the model's predictions with mathematical exactness within floating-point tolerance — a glass-box
GBM you can read, ship as lookup tables, or audit.

- **Rust core** (`t-boost-core`), thin [PyO3](https://pyo3.rs) bindings, and polars-native
  Python estimators. The core is `#![forbid(unsafe_code)]`, no-panic-gated, and deterministic
  (bit-identical across thread counts).
- **polars-native**: `TBoostRegressor` / `TBoostClassifier` take polars DataFrames and LazyFrames
  directly, with targets, weights and exposure named by column.
- **Objectives**: `squared_error`, `logistic` (binary), native-softmax **multiclass**, and the
  log-link `poisson` / `gamma` / `tweedie` families for insurance frequency & severity.
- **Exact decomposition**: `model.tables(X)` emits the fANOVA rating tables; the reconstruction
  is verified against the ensemble by lossless invariant checks. Tables are purified against
  the exposure-weighted marginals by default (`ref_measure="exposure"`); `tables(X, ref_measure="joint")`
  re-expresses the same model under each pair's joint exposure via regularized pairwise reallocation with explicit ridge regularization and residual diagnostics for correlated factors, and `actual_vs_expected(X, y, exposure=...)`
  gives the A/E by rating-factor level a reviewer asks for first (explicit exposure/weight arguments are required for reliable row alignment). Neither changes a prediction.

## Install

```bash
uv add t-boost
```

Wheels are built for Linux / macOS / Windows as a single abi3 wheel per platform (CPython 3.10–3.13).

### From source

Building from source needs a Rust toolchain (`rustup`); [uv](https://docs.astral.sh/uv/) drives
the [maturin](https://www.maturin.rs) build:

```bash
uv sync                          # builds the Rust extension into the project's .venv
```

## Quickstart

polars DataFrames and LazyFrames are the estimators' first-class frame type — no pandas anywhere.
`y` / `sample_weight` / `exposure` / `groups` may name columns of `X` (which are then excluded
from the features), String/Categorical/Enum columns are target-statistic encoded automatically,
and prediction matches feature columns by name, ignoring extras:

```python
import polars as pl
from t_boost import TBoostClassifier, TBoostRegressor

train = pl.read_parquet("policies.parquet")            # or pl.scan_parquet(...) for lazy input

# Claim frequency: Poisson with an exposure offset
freq = TBoostRegressor(objective="poisson").fit(
    train.select(FEATURES + ["ClaimCount", "Exposure"]),
    "ClaimCount",                                      # y, by column name
    exposure="Exposure",                               # per-row offset, by column name
)
rate = freq.predict(test)                              # extra columns ignored, any column order

# Classification: binary, or native softmax for K >= 3 classes
clf = TBoostClassifier().fit(train.select(FEATURES + ["Lapsed"]), "Lapsed")
proba = clf.predict_proba(test)                        # (n, K), rows sum to 1
```

### The exact decomposition

```python
import json
tables = json.loads(freq.tables(train))     # fANOVA rating tables (JSON)
```

Each fitted model decomposes into main effects and interactions that reproduce the raw
score with mathematical exactness within floating-point tolerance (for a multiclass model, one table bank per class logit).

## Objectives

| Objective | Task | Link |
|-----------|------|------|
| `squared_error` | regression | identity |
| `logistic` | binary classification | logit |
| softmax (automatic for `TBoostClassifier` with ≥3 classes) | multiclass | softmax |
| `poisson` | counts / frequency | log |
| `gamma` | positive severities | log |
| `tweedie` | compound Poisson-gamma | log |

## Status & license

Pre-1.0; the wire `schema_version` is versioned independently of the package version.

### Serialization compatibility

Serialized models and rating exports embed a wire `schema_version` (supported through `6`; see `SCHEMA_VERSION` in
[`crates/t-boost-core/src/serialize.rs`](crates/t-boost-core/src/serialize.rs)).
Rating table export schema versions are content-dependent based on model structure and reference measure (v2 for legacy ≤3-order unlifted models, v3 for depth-lifted trees up to 8, v4 for order-4 or factored box representations, v5 for high-order interactions 5..8, and v6 for exposure-marginal reference measures).
Known supported artifacts are validated and loaded according to the container's version rules.
JSON provides registered migration paths; binary compatibility is container- and schema-dependent,
not a promise that any older blob can load. Pin the producing package version with each artifact
and verify loading and prediction equivalence before upgrading.

Licensed under **Apache-2.0**.
