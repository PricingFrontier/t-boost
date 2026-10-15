# Model report

The model report gathers, for one model and its train and test data, what a reviewer needs to
decide whether the model is ready for production: how well it fits in and out of sample, how it is
built, how the pruning chose its tables, every rating factor, the drift between the datasets
and a list of review flags.

- [pricing_report](../python-reference/tboostregressor/pricing_report.md) returns the report as
  a JSON-serializable dictionary: the record to keep beside the model.
- [report_html](../python-reference/report-html.md) renders a report as one self-contained HTML
  page; [html_report](../python-reference/tboostregressor/html_report.md) does both in one call.

```python
import json

import t_boost

report = model.pricing_report(train_data, "ClaimCount", exposure="Exposure",
                              eval_set=(test_data, "ClaimCount"), eval_exposure="Exposure",
                              name="Motor frequency v3")
with open("freq_report.json", "w") as f:
    json.dump(report, f)
with open("freq_report.html", "w") as f:
    f.write(t_boost.report_html(report))
```

The train data is `X`, `y`; `eval_set` adds the test data, or the validation and the test data
(a list of two pairs), named `train` and `test`, or `train`, `validation` and `test`, unless
`labels` says otherwise. Train and test are the usual pair: t-boost's bags early-stop and prune
on objects they hold back, so a model needs no separate validation data. Keep some only to
compare settings.

The report describes the data passed to it; it cannot verify which rows were used to fit the
model and which were held out.

All three kinds of model are supported: regression, binary classification and
multiclassification (one-vs-rest per class, with a class selector on the page). The per-row work
(cell lookup, the sums per cell, the sorts behind the Gini and the lift) runs natively, so a
report on a million rows takes about as long as scoring them.

## Logging the report with MLflow {#mlflow}

t-boost has no MLflow code; the report is built to be logged as it is:

```python
import mlflow

report = model.pricing_report(...)
mlflow.log_metrics(report["summary_metrics"])            # flat, finite, valid metric names
mlflow.log_dict(report, "t_boost/report.json")           # the full record
mlflow.log_text(t_boost.report_html(report), "t_boost/report.html")
```

The page has no scripts and loads nothing from the network, so it displays in MLflow's artifact
viewer.

## The page {#sections}

The page has seven tabs and one dataset switch (train and test, or train, validation and test;
it opens on the last dataset given, the most independent one). The selected dataset is drawn in
colour in every chart and the others in grey, so the datasets compare without scrolling; the
legends and the titles name the dataset each line or figure belongs to. Blue is
observed data and orange is the model throughout. Each chart has a one-line caption and its
numbers collapsed beneath it; hovering a point shows its values. The tabs, the switches and the
collapsed sections are plain HTML and CSS, so they work where scripts are blocked.

### Overview {#overview}

One screen (1440 × 900) with what a reviewer reads first: the deviance explained, Gini and A/E of
every dataset; the open review flags grouped by check; calibration by decile of predicted rate;
the feature importance; the largest one-way tables side by side on one scale; and the share of
the pruning gain against the number of tables. Chart titles state the result (for example "Top
decile 8.2× the bottom; test decile A/E 0.94–1.08").

### Performance {#performance}

Per dataset:

- **Deviance**: the objective's mean unit deviance of the target against the expected value of
  each object (exposure and offset included), weighted by the sample weight — the convention of
  the fit's own deviance. Binary and multiclassification models use the binomial and multinomial
  deviances (twice the negative log-likelihood).
- **Null deviance**: the deviance of the best constant model on the same objects, with the same
  exposure and offset. For a log link with variance power $p$ and $a_i = e_i \exp(o_i)$, the
  constant rate is $c = \sum w_i y_i a_i^{1-p} / \sum w_i a_i^{2-p}$.
- **Deviance explained**: $1 - deviance / null\ deviance$.
- **Gini**: the Lorenz (concentration) Gini of the actual rate ranked by the predicted rate, the
  exposure being the mass on the x-axis for a log-link model; tied predictions form one segment.
  The normalised Gini divides it by the Gini of the perfect ranking.
- **A/E**: $\sum w_i y_i / \sum w_i \mu_i$.

Then calibration by decile: ten bands of equal exposure by ascending predicted rate (a group of
tied predictions stays in one band), the actual and predicted rate of the selected dataset above
and the A/E of every dataset below, with ±2 standard deviations shaded for the selected one. A
standard deviation is the objective's variance function times the Pearson dispersion, summed over
the band's rows. Beside the Lorenz curves, every feature's A/E by band as a small chart on one
shared scale, each band as wide as its share of exposure: the selected dataset in colour with ±2
standard deviations shaded, the `factor_ae` tolerance dashed, the bands the review flags in red
and its near misses in amber, and each feature's band furthest from balance named above its
chart. The scale is set by the bands the review checks (`factor_min_mass`); thinner ones are
clipped at its edge. Below, the spread of the predictions as a histogram: the share of exposure
by predicted rate, on bins every dataset shares (log-spaced when the rates span more than a
factor of 3), the selected dataset as bars and the others outlined. A multiclassification model has
a class switch, one-vs-rest views per class and the class matrices.

### Factors {#factors}

A list of the features by importance (the share of model variance in the feature's own one-way
table, then its share of the interactions it is in, split evenly over their features), each
with its band furthest from balance on the selected dataset. Selecting a feature shows, on one
x-axis of its levels: the deployed one-way table (relativities on a log scale for a log link,
odds multipliers for a logit link; cells holding under `thin_cell_mass` of the table's exposure
faded); the actual and expected rate by level on the selected dataset (levels under
`thin_cell_mass` of its exposure faded); the A/E of the factor check's bands for every dataset,
with the `factor_ae` tolerance dashed and ±2 standard deviations shaded; and the train exposure.
Categorical levels are ordered by their effect. The findings about the feature are listed above
the chart; the numbers by band and by level are collapsed below it.

Over the one-way table, the chart draws the feature's SHAP value per level on the train data: its
mean (dashed) and the band holding 80% of the policies at that level (10th to 90th percentile,
weighted by exposure). A policy's SHAP value for a feature is its one-way table plus an equal
share of every interaction table the feature is in; with t-boost's centred tables this split is
the exact interventional Shapley value, and the features' values add up to the score. The values
are read from the tables the report shows, so where the feature is in no interaction the mean is
its one-way table. The band
shows how much the other factors, through the interactions, move this feature's effect: narrow
where the one-way table tells the whole story, wide where an interaction matters. A partial
dependence plot is not drawn because here it would repeat the one-way table: every interaction
table averages to no effect over its partner's exposure.

### Interactions {#interactions}

The interaction tables by share of model variance, each with the range of its effect over the
exposure (the support-weighted 5th to 95th percentile of its cells). Selecting one draws, on the
levels of its feature with the most levels, that feature's one-way table (dashed) and, for up to
four bands of equal exposure of the partner, the effect with this table added: where the lines
part, the interaction matters. Under it, the selected dataset's actual (dashed) and expected
(solid) rate by level of the first feature within each partner band (pairs under
`thin_cell_mass` of the exposure have their actual faded); then one strip per partner band with
the A/E by band of the first feature, ±2 standard deviations shaded and the `factor_ae`
tolerance dashed (band pairs under `factor_min_mass` faded). Lines that stay near 1 in every
strip mean the model captures the interaction. A three-way table
is averaged over its third feature. The 40 largest are drawn; every table's numbers are under
Tables.

### Tables {#tables}

Every deployed table, by its share of the model variance, with its effect range and smoothing
strength; selecting one shows its numbers: one-way tables as rows, two-way tables as a grid (rows
the first feature's levels, columns the second's) with the support grid below, and tables of
order three or more as one row per cell. Each number is tinted light green above no effect and
light yellow below, in five steps of its size against the table's largest; a cell with no
exposure is grey. Each cell is the relativity (log link),
the odds multiplier (logit link) or the value on the link scale. Interaction tables hold only the
adjustment beyond their features' one-way tables. Tables past the page's cell budget, and
factored effects, are in the JSON report only.

### Build {#build}

How the model was built. The share of the out-of-bag gain against the size of the bank, with the
cutoff; every candidate table as a column of its variance share, in the order the path adds it,
coloured kept, removed by the variance floor or past the cutoff; the ranked tables in collapsible
groups (main effects, kept, below the floor, past the cutoff). The cutoff is the later of the
first bank reaching `prune_path_fraction` of the total gain and the first within
`prune_path_tolerance` of the minimum; the floor (`prune_min_variance_share`) then removes the
tables below it that no kept table is built on. The pruning report is saved with the model. Then
the smoothing strength of each table, and the deviance on each bag's early-stopping objects by
boosting round, from 0 to the starting deviance, with the round each bag kept, and on each bag's
training objects too when `callbacks` were passed to `fit` (the curves are kept on the trained
model only: they are not saved with the model).

### Checks {#checks}

Every review flag with its status (`critical`, `warning`, `info` or `pass`), the value it
measured and the threshold it used, then the data checks, the parameters set away from their
defaults (the rest collapsed), the model's columns and metadata, and "How to read this report":
the definition of every statistic on the page. The flags support a review; they are not a
verdict. The thresholds are set with the `thresholds` parameter of `pricing_report`, so each
team can apply its own:

- the overall A/E of each dataset (`ae_train`, `ae_validation`, `ae_test`);
- whether the deviance explained on each later dataset is positive (the model beats a constant);
- the drop in deviance explained from train to each later dataset (`train_gap`);
- whether the report has out-of-sample data at all;
- whether graduation, the smoothing of the tables, was declined (see
  [Graduation settings](../training-parameters/graduation.md));
- whether any bag reached `n_trees` without [early stopping](../algorithm/early-stopping.md);
- deciles of predicted rate whose A/E is off by more than `decile_ae` and at least `factor_z`
  standard deviations (with `factor_z` set to `None`, the tolerance alone);
- factor bands whose A/E is off by more than `factor_ae`, among bands holding at least
  `factor_min_mass` of the mass and at least `factor_z` standard deviations from balance (with
  `factor_z` set to `None`, the tolerance alone). Each
  feature is read on up to ten bands of equal train exposure (the same bands for every dataset),
  and the passing check states the share of the exposure it covered. On out-of-sample data, bands
  and deciles past their tolerance with at least `near_miss_z` standard deviations are listed as
  near misses;
- one-way tables whose missing-value level has no exposure yet an effect further than
  `missing_effect` from no effect (what a missing value would be charged), and thin cells whose
  effects reach beyond those of the rest of the table;
- graduation adopted without holdout evidence;
- PSI per feature, from the train data to each later dataset (on at most 10 bins of each
  feature's grid, shares floored at 0.0001; it compares the datasets given, it is not a
  forecast);
- unseen categorical values;
- `binding_report_` rows that are `INERT`, `OVERRIDDEN` or `UNAVAILABLE`.
