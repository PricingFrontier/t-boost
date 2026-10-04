# from_bytes

Load a model from the binary format written by [to_bytes](to_bytes.md). This is a class method.

## Method call format {#call-format}

```python
TBoostClassifier.from_bytes(data)
```

## Parameters {#parameters}

### data

#### Description

The output of [to_bytes](to_bytes.md).

**Possible types**

bytes

**Default value**

Required parameter

## Return value {#output-format}

A new, trained TBoostClassifier.

t-boost reads the files written by the same or an earlier version, and a loaded model predicts
identically to the version that wrote it. There are two exceptions: a file written by a newer
version raises `SerializationError` naming both versions, and a model with categorical features
saved by t-boost 0.6 is refused, because its missing level was labelled differently. See
[Saving and loading models](../../features/saving-models.md).

## Usage examples {#usage-examples}

```python
from t_boost import TBoostClassifier

with open("model.tboost", "rb") as f:
    model = TBoostClassifier.from_bytes(f.read())
```
