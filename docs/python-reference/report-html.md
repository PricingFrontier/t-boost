# report_html

Render a model report returned by `pricing_report` as one self-contained HTML page.

The page is a view of the report: it computes nothing that is not in the report, so a page can
be rebuilt from a report saved as JSON (for example, an MLflow artifact). It holds no scripts
and loads nothing from the network.

## Method call format {#call-format}

```python
report_html(report, *, name=None)
```

## Parameters {#parameters}

### report

#### Description

The dictionary returned by [pricing_report](tboostregressor/pricing_report.md), or the same
document read back from JSON. A report of another version raises `ValueError`.

**Possible types**

dict

**Default value**

Required parameter

### name

#### Description

The heading of the page, instead of the report's own `name`.

**Possible types**

string

**Default value**

None

## Return value {#output-format}

A string: the HTML document.

## Usage examples {#usage-examples}

```python
import json

from t_boost import report_html

with open("freq_report.json") as f:
    report = json.load(f)
with open("freq_report.html", "w") as f:
    f.write(report_html(report))
```
