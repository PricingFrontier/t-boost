# t-boost

t-boost is a machine learning algorithm that uses gradient boosting on oblivious decision
trees. Its fitted model is *exactly* a set of rating tables: a Tabulating Boosting Machine
(TBM). It is available as an open source library.

Gradient-boosted trees are usually more accurate than a GLM, but they are hard to read, review
or deploy in systems built around rating tables. t-boost aims to get boosting-level accuracy in
a model that *is* a set of main-effect and interaction tables, with no approximation and no
surrogate model. It has a Rust core with Python bindings, takes polars DataFrames directly and
is deterministic.

```python
from t_boost import TBoostRegressor

model = TBoostRegressor(objective="poisson")
model.fit(train_data, "ClaimCount", exposure="Exposure")
tables = model.tables(train_data)   # the model, as rating tables
```

<div class="grid cards" markdown>

-   :material-download: **Installation**

    ---

    - [Python package installation](installation/index.md)
    - [Quick start](quickstart.md)
    - [Build from source](installation/build-from-source.md)

-   :material-school: **Training**

    ---

    - [Training](features/training.md)
    - [Training parameters](training-parameters/index.md)
    - [Using the overfitting detector](features/overfitting-detector.md)
    - [Categorical features](features/categorical-features.md)
    - [Exposure and offsets](features/exposure.md)
    - [Panel data](features/panel-data.md)

-   :material-table: **Rating tables**

    ---

    - [Rating tables](model-analysis/rating-tables.md)
    - [Prediction contributions](model-analysis/contributions.md)
    - [Feature importance](model-analysis/feature-importance.md)
    - [Actual versus expected](model-analysis/actual-vs-expected.md)

-   :material-play-circle: **Applying models**

    ---

    - [Regular prediction](features/prediction.md)
    - [Saving and loading models](features/saving-models.md)

-   :material-function-variant: **Metrics**

    ---

    - [Objectives and metrics](objectives/index.md)
    - [t_boost.metrics](python-reference/metrics.md)

-   :material-book-open-variant: **Educational materials**

    ---

    - [How training is performed](algorithm/index.md)
    - [Pruning, banding and graduation](algorithm/readable-tables.md)
    - [Parameter tuning](parameter-tuning.md)

</div>

## How it works {#how-it-works}

- **Constrained trees.** Each tree is symmetric (oblivious): every level applies one shared
  `(feature, threshold)` split. Each tree may use only a few distinct features, so the whole
  ensemble has a fixed maximum interaction order.
- **Exact decomposition.** Because of that structure, the trained ensemble can be rewritten as
  a functional-ANOVA (fANOVA) decomposition: one table per main effect and per interaction. The
  tables reproduce the model's predictions exactly, to floating-point tolerance.
- **Purification.** The tables are centred on the training data (exposure-weighted when an
  exposure is given), so each main effect carries as much of the signal as it can.
- **The tables are the model.** A saved model is stored as its rating tables, so what you
  review is exactly what gets deployed.

See [How training is performed](algorithm/index.md) for details.
