# from_json

Load a model from the JSON document written by [to_json](to_json.md). This is a class method.

## Method call format {#call-format}

```python
TBoostClassifier.from_json(data)
```

## Parameters {#parameters}

### data

#### Description

The output of [to_json](to_json.md).

**Possible types**

string

**Default value**

Required parameter

## Return value {#output-format}

A new, trained TBoostClassifier. The compatibility rules are those of [from_bytes](from_bytes.md).

## Usage examples {#usage-examples}

```python
from t_boost import TBoostClassifier

with open("model.json") as f:
    model = TBoostClassifier.from_json(f.read())
```
