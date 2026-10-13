# Actual versus expected

Actual versus expected (A/E) compares the observed and the predicted totals for each level of
each rating factor. It shows where the model and the data disagree: exact balance by factor
level is not a general property of boosted models, nor of every GLM.

For each feature, the objects are aggregated over the cells of the grid the deployed tables use
(cell 0 holds missing values):

- $actual = \sum w_i t_i$
- $expected = \sum w_i \mu_i$, where $\mu_i$ includes the offset and, for a model with a log
  link, the exposure
- $A/E = actual / expected$

```python
ae = model.actual_vs_expected(train_data, "ClaimCount", exposure="Exposure")
for factor in ae:
    print(factor["feature"], factor["ae"])
```

If the model was trained with a sample weight, an exposure or an offset, pass them explicitly,
even on the training data. See [actual_vs_expected](../python-reference/tboostregressor/actual_vs_expected.md).

## Multiclassification {#multiclassification}

A multiclassification model has one table bank, and so one grid, per class. Its actual versus
expected is one-vs-rest: one entry per class and feature, with a `class` key,
$actual = \sum w_i [t_i = class]$ and $expected = \sum w_i P_i(class)$.

## Model report {#model-report}

[pricing_report](../python-reference/tboostregressor/pricing_report.md) and
[html_report](../python-reference/tboostregressor/html_report.md) put the actual versus expected
of one or two datasets, with a z-score per level, beside the fit metrics, the structure and the
review flags. See [Model report](model-report.md).
