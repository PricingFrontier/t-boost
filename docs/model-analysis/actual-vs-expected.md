# Actual versus expected

Actual versus expected (A/E) compares the observed and the predicted totals for each level of
each rating factor. It shows where the model and the data disagree: exact balance by factor
level is not a general property of boosted models, nor of every GLM.

For each feature, the objects are aggregated over the cells of the grid the deployed tables use
(cell 0 holds missing values):

- $actual = \sum w_i t_i$
- $expected = \sum w_i \mu_i$, where $\mu_i$ includes the exposure for a model with a log link
- $A/E = actual / expected$

```python
ae = model.actual_vs_expected(train_data, "ClaimCount", exposure="Exposure")
for factor in ae:
    print(factor["feature"], factor["ae"])
```

If the model was trained with a sample weight or an exposure, pass them explicitly, even on the
training data. See [actual_vs_expected](../python-reference/tboostregressor/actual_vs_expected.md).

## Pricing report {#pricing-report}

[pricing_report](../python-reference/tboostregressor/pricing_report.md) collects, in one document
for review, the rating tables, the actual versus expected cells with their axes, and the
diagnostics of pruning and graduation:

```python
import json

report = model.pricing_report(test_data, "ClaimCount", exposure="Exposure")
with open("freq_report.json", "w") as f:
    json.dump(report, f)
```

The report describes the data passed to it; it is not evidence that the data were held out.
Save it beside the serialized model.
