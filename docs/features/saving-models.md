# Saving and loading models

A trained model is stored as its rating tables, so what you review is exactly what gets
deployed. t-boost provides two formats, and does not write files itself: the methods return the
model as bytes or as a string, which you store where you need.

| Method | Format | Load with |
|--------|--------|-----------|
| [to_bytes](../python-reference/tboostregressor/to_bytes.md) | compact binary | [from_bytes](../python-reference/tboostregressor/from_bytes.md) |
| [to_json](../python-reference/tboostregressor/to_json.md) | JSON, diffable and readable without t-boost | [from_json](../python-reference/tboostregressor/from_json.md) |

`from_bytes` and `from_json` are class methods of `TBoostRegressor` and `TBoostClassifier`.

## Usage examples {#usage-examples}

```python
from t_boost import TBoostRegressor

with open("freq.tboost", "wb") as f:
    f.write(model.to_bytes())

with open("freq.tboost", "rb") as f:
    loaded = TBoostRegressor.from_bytes(f.read())
```

A loaded model predicts identically to the original.

## Metadata {#metadata}

The `metadata` attribute holds your own JSON metadata, which is saved with the model and
returned unchanged when it is loaded. t-boost never reads it.

```python
model.metadata["portfolio"] = "motor"
model.metadata["data_cutoff"] = "2026-06-30"
```

## Compatibility {#compatibility}

t-boost reads the files written by the same or an earlier version, and a loaded model predicts
identically to the version that wrote it. A file written by a newer version raises
`SerializationError` naming both versions. A model with categorical features saved by t-boost 0.6
is refused, because its missing level was labelled differently. Models saved by t-boost 0.6 or
earlier were stored as tree ensembles: they still load and predict, but do not support the
methods that need rating tables, such as `predict_contributions`.
