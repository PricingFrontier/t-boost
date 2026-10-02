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

### Per-prediction contributions

`predict_contributions` breaks each prediction into one contribution per rating table. The
record format is the same as rustystats' `GLMModel.predict_contributions`. Because the deployed
model *is* its tables, the breakdown is exact: `base_value + sum(contributions)` is the raw
(link-scale) score, and the inverse link of that is the prediction.

```python
rows = freq.predict_contributions(test.head(5))                  # one dict per row
rows[0]["contributions"]          # [{"term": "age", "term_type": "main", "contribution": ...},
                                  #  {"term": "age:region", "term_type": "interaction", ...}, ...]
freq.predict_contributions(test, exposure="Exposure")            # adds a log(exposure) term
freq.predict_contributions(test, split_interactions=True)        # exact Shapley value per feature
freq.predict_contributions(test, return_format="dataframe")      # long polars DataFrame
freq.feature_importances_                                        # Sobol shares per input feature
```

`exposure=` adds a `log(exposure)` contribution, so `prediction_value` becomes the expected total
(`predict(X) * exposure`) rather than the rate per unit exposure. A multiclass model decomposes
each class logit the same way: each record holds a `classes` list, and each class's
`prediction_value` is its softmax probability.

Every model is stored as its rating tables. `prune=True` (the default) keeps the tables that earn
their place. `prune=False` keeps all of them, which is quicker to fit but gives a much larger
file: tens of MB on wide data, against well under 1 MB for a pruned model.

`tables(X)` exports the tables as stored, centred on the training rows. Passing `sample_weight=`
or `exposure=` re-centres them on `X`'s rows under that mass instead. That changes how the effect
is shared between tables, but never a prediction.

## Saving and loading

A fitted estimator serializes to bytes (compact) or JSON (diffable) and loads back as a fitted
estimator that predicts identically:

```python
from t_boost import TBoostRegressor

with open("freq.tboost", "wb") as f:
    f.write(freq.to_bytes())

with open("freq.tboost", "rb") as f:
    loaded = TBoostRegressor.from_bytes(f.read())   # TBoostClassifier.from_bytes for classifiers

rate = loaded.predict(test)
test.select(loaded.required_columns)               # the raw feature columns predict reads
loaded._exposure_spec                              # "Exposure": the column named at fit, or None
```

`to_json()` / `from_json()` work the same way. Pickle and joblib work too. The JSON document is
the envelope: estimator metadata at the top level, and the model document as a JSON string under
`"model"`. In 0.6.1 and earlier, a numeric regressor's `to_json()` was the bare model document.

| Preserved | Notes |
|-----------|-------|
| The model (its rating tables, pruned or full) | Bit-identical predictions after loading |
| Constructor parameters | `get_params()` matches the fitted estimator |
| Feature names, categorical layout, `classes_` | Needed to serve by column name |
| Column specs | `_response_spec`, `_weights_spec`, `_exposure_spec`, `_groups_spec`: the column names given to `fit`, or `None` when passed as arrays |

Fit-time reports (`pruning_report_`, `graduation_report_`, …), training data, and fit-time
exposure/weights are not saved.

**Version compatibility.** Each blob records the t-boost version that wrote it and a
`schema_version`. A newer t-boost loads blobs written by older releases. A blob from a newer
schema than the running t-boost supports is refused with `SerializationError` rather than
half-read. So is loading a classifier blob with `TBoostRegressor.from_bytes`, or the reverse.

Two kinds of 0.6.x model are refused and must be refit: models with categorical features (0.6.x
labelled their missing-value level differently, so nulls would score wrongly), and pickles.
Numeric 0.6.x models saved with `to_bytes` / `to_json` still load.

## Objectives

| Objective | Task | Link |
|-----------|------|------|
| `squared_error` | regression | identity |
| `logistic` | binary classification | logit |
| softmax (automatic for `TBoostClassifier` with ≥3 classes) | multiclass | softmax |
| `poisson` | counts / frequency | log |
| `gamma` | positive severities | log |
| `tweedie` | compound Poisson-gamma | log |

## License

Apache-2.0
