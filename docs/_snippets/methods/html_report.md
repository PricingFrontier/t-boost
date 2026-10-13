# html_report

Return the model report as one self-contained HTML page: everything needed to judge how well the
model fits, its structure, its pruning, every rating factor and interaction, and the review
flags, in tabs led by a one-screen overview.

This is `report_html(pricing_report(...))`: it takes the same parameters as
[pricing_report](pricing_report.md). The page holds no scripts and loads nothing from the
network, so it opens offline, in a sandboxed viewer such as MLflow's artifact view, and prints.
To keep the data as well as the page, call `pricing_report` and log the dictionary too: the page
can be rebuilt from it at any time with [report_html](../report-html.md).

See [Model report](../../model-analysis/model-report.md#sections) for the tabs of the page.

--8<-- "_snippets/feature-matching-note.md"

## Method call format {#call-format}

```python
html_report(X, y, *,
            sample_weight=None, exposure=None, offset=None,
            ref_measure=None,
            eval_set=None, eval_sample_weight=None, eval_exposure=None, eval_offset=None,
            labels=("train", "validation", "test"),
            name=None,
            thresholds=None)
```

## Parameters {#parameters}

The parameters of [pricing_report](pricing_report.md#parameters).

## Return value {#output-format}

A string: the HTML document.

## Usage examples {#usage-examples}

```python
# model: a TBoostRegressor(objective="poisson") trained with exposure="Exposure"
html = model.html_report(train_data, "ClaimCount", exposure="Exposure",
                         eval_set=[(validation_data, "ClaimCount"),
                                   (test_data, "ClaimCount")],
                         eval_exposure=["Exposure", "Exposure"],
                         name="Motor frequency v3", thresholds={"ae_test": 0.03})
with open("freq_report.html", "w") as f:
    f.write(html)
```
