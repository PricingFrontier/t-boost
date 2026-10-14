# pricing_report

Return the model report: fit quality, structure, pruning evidence, rating tables, actual versus
expected by factor level, drift and review flags, as one JSON-serializable document.

The report is the record of a model's review: log it beside the serialized model, and render it
with [report_html](../report-html.md) (or call [html_report](html_report.md)) for a page to read.
See [Model report](../../model-analysis/model-report.md) for what it contains and how each
statistic is defined.

The report describes the data passed to it; it is not evidence that the data were held out.

--8<-- "_snippets/feature-matching-note.md"

## Method call format {#call-format}

```python
pricing_report(X, y, *,
               sample_weight=None, exposure=None, offset=None,
               ref_measure=None,
               eval_set=None, eval_sample_weight=None, eval_exposure=None, eval_offset=None,
               labels=None,
               name=None,
               thresholds=None)
```

## Parameters {#parameters}

--8<-- "_snippets/params/X-apply.md"

### y

#### Description

The target values of the objects (for a classifier, the class labels). A string names a column
of a polars `X`.

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- list
- string

**Default value**

Required parameter

### sample_weight, exposure, offset

#### Description

The weight, the exposure and the offset of each object, passed explicitly as for
[actual_vs_expected](actual_vs_expected.md): a value the model was trained with must be passed
again, even on the training data. A multiclassification model accepts no exposure or offset.

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- list
- string

**Default value**

None

### ref_measure

#### Description

The reference measure of the exported tables, as for [tables](tables.md).

**Possible types**

string

**Default value**

None

### eval_set

#### Description

The test data, or the validation and the test data, described beside the train data (`X`, `y`):

- one pair `(X_test, y_test)`, the convention of `fit`;
- a list of two pairs `[(X_validation, y_validation), (X_test, y_test)]`.

Train and test are the usual pair: the bags early-stop and prune on objects they hold back, so
a model needs no separate validation data. Keep some only to early-stop on chosen data (for
example the latest period) or to compare settings.

Each dataset gets its own metrics, A/E, unseen values and review checks, and the drift from the
train data to it (PSI) is measured. A `y` may name a column of its polars `X`.

If the model was early-stopped on an `eval_set` passed to `fit`, that data is not fully
out-of-sample: pass it as the validation data and keep a test set for the independent estimate.

**Possible types**

- tuple `(X_test, y_test)`
- list of two tuples `[(X_validation, y_validation), (X_test, y_test)]`

**Default value**

None

### eval_sample_weight, eval_exposure, eval_offset

#### Description

The weight, the exposure and the offset of the `eval_set` objects. A string names a column of
that dataset's `X`. With a list `eval_set`, each is a list with one entry per dataset (`None`
for a dataset without one), for example `eval_exposure=["Exposure", "Exposure"]`.

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- string
- list of the above, one per dataset of a list `eval_set`

**Default value**

None

### labels

#### Description

The names of the datasets, in order. They label the page and prefix the `summary_metrics` keys.
`None` names them by their role: `("train", "test")` for two datasets, `("train", "validation",
"test")` for three. The review thresholds follow the role, not the name: `ae_test` applies to
the second of two datasets and `ae_validation` to the second of three, whatever they are called.

**Possible types**

- tuple of distinct strings
- None

**Default value**

None

### name

#### Description

The name of the report, shown as the heading of the page followed by the time of the fit.

**Possible types**

string

**Default value**

None (the class and the objective)

### thresholds

#### Description

Overrides of the review thresholds, by key. `None` as a value turns that check off. An unknown
key, a negative or non-finite value, or `psi_warn` above `psi_critical` raises `ValueError`.
The values in force are written to the report and printed next to every flag.

| key | default | check |
|---|---|---|
| `ae_train` | 0.02 | overall A/E on the train data within 1 ± value |
| `ae_validation` | 0.05 | overall A/E on the validation data within 1 ± value |
| `ae_test` | 0.05 | overall A/E on the test data within 1 ± value |
| `factor_ae` | 0.10 | factor bands whose A/E is off by more than value … |
| `factor_min_mass` | 0.01 | … among bands holding at least this share of the mass … |
| `factor_z` | 3.0 | … and at least this many standard deviations from balance (`None`: the tolerance alone flags, for the deciles too) |
| `decile_ae` | 0.05 | deciles of predicted rate whose A/E is off by more than value (and at least `factor_z` standard deviations) |
| `near_miss_z` | 2.0 | out-of-sample bands and deciles past their tolerance with \|z\| at least value, but below `factor_z`, are listed as near misses |
| `missing_effect` | 0.05 | a one-way table's missing-value level with no exposure whose effect is further than value from no effect |
| `thin_cell_mass` | 0.001 | one-way cells holding less than this share of the table's exposure count as thin |
| `psi_warn` | 0.10 | PSI per feature above value: warning |
| `psi_critical` | 0.25 | PSI per feature above value: critical |
| `train_gap` | 0.05 | drop in deviance explained from train to each later dataset |
| `unseen_mass` | 0.01 | mass share of unseen categorical values in a dataset |

The factor check reads up to ten bands per feature (the missing values, then runs of equal train
exposure on a numeric feature, or the largest levels and one "other" band on a categorical one),
the same bands for every dataset, so a feature with hundreds of fine levels is still checked on
nearly all of its exposure. A standard deviation is the objective's variance function times the
Pearson dispersion, summed over the band's rows.

**Possible types**

dict

**Default value**

None (the defaults above)

## Return value {#output-format}

A dictionary of plain JSON values (`json.dumps(report, allow_nan=False)` succeeds):

- `report_version` — The version of the report format: 2.
- `name`, `generated_at` — The report name and when it was built (UTC).
- `model` — Class, objective, link, t-boost version, input columns, metadata, the time of the
  fit (`fitted_at`, `fit_duration_seconds`), the parameters with their defaults.
- `fit` — Trees per bag, stopping reasons, the training history, the pruning report, the
  graduation diagnostics and the smoothing strength of each table (`graduation_tables`), and
  `binding_report_`, each with whether it is available on this model.
- `datasets` — Per dataset: its label, rows, `metrics` (deviance, null deviance, deviance
  explained, Gini, A/E, lift bands with their A/E, standard error `sd` and `z`, Lorenz curve,
  percentiles of the predicted rate and its histogram (`predicted_rate_histogram`: bin `edges`
  shared by every dataset, the `mass` in each and whether the bins are `log`-spaced),
  and for a multiclassification model accuracy, a class matrix and per-class values),
  `actual_vs_expected` (each cell with a `z` score; on the first dataset, `shap`: each cell's
  exposure-weighted mean and 10th and 90th percentile of the feature's exact Shapley value, on
  the link scale, read from the report's `tables`), `bands` (the factor check's bands per
  feature; a band the review flags carries `"flag": "warning"`, a near miss
  `"flag": "near_miss"`), missing values and unseen values. Each cell's `axis` is its feature's merged grid; a
  numeric axis carries `integer`, true when the feature took only whole values in the first
  dataset, and its cells are then labelled as integers (`51`, `22–25`). The numeric axes of
  `tables` carry the same flag.
- `drift` — PSI per feature from the train data to each later dataset: a list of
  `{"reference", "compared", "features"}`.
- `structure` — The deployed effects of each table bank with their variance shares.
- `interactions` — Per interaction table, by share of model variance: its features, share,
  cells, the range of its effect over the cells with exposure and the range holding 90% of the
  exposure (`spread`), and its smoothing strength. The 40 largest also carry `ae`: the A/E by
  band of the first feature (`x`, the factor check's bands) within each band of its partner
  (`by`, up to four bands of equal exposure), per dataset, with each pair's `mass`, `z` and
  standard deviation `sd`.
- `features` — Per raw feature, by importance: its importance split into its own one-way table
  and its share through interactions, the interactions it is in, and its one-way table's effect
  range, thin cells, missing-value level and smoothing strength.
- `review` — The flags: `check`, `status` (`critical`, `warning`, `info` or `pass`), `message`,
  `value`, `threshold` and the dataset, feature or class it concerns.
- `thresholds` — The thresholds in force.
- `summary_metrics` — A flat dictionary of finite floats, `<label>_<metric>` (and
  `<label>_<metric>_<class>`), plus `n_trees`, `deployed_tables`, `review_warnings` and
  `review_critical`. The keys are valid MLflow metric names.
- `tables`, `actual_vs_expected`, `pruning`, `graduation`, `prediction_units`,
  `reference_normalization`, `band_interpretation` — As in version 1: the rating tables, the
  first dataset's A/E cells with their axis, and the stage diagnostics.

## Usage examples {#usage-examples}

```python
import json

import t_boost

# model: a TBoostRegressor(objective="poisson") trained with exposure="Exposure"
report = model.pricing_report(train_data, "ClaimCount", exposure="Exposure",
                              eval_set=(test_data, "ClaimCount"), eval_exposure="Exposure",
                              name="Motor frequency v3")
with open("freq_report.json", "w") as f:
    json.dump(report, f)
with open("freq_report.html", "w") as f:
    f.write(t_boost.report_html(report))
print(report["summary_metrics"])
```
