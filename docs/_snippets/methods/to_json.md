# to_json

Serialize the trained model to a JSON document.

The document holds the training parameters, the fitted metadata and the model (its rating
tables), and is read back with `from_json`. It is larger than the output of `to_bytes`, but it
can be diffed and inspected without t-boost. A loaded model predicts identically to the
original. See [Saving and loading models](../../features/saving-models.md).

## Method call format {#call-format}

```python
to_json()
```

## Return value {#output-format}

string

## Usage examples {#usage-examples}

```python
with open("model.json", "w") as f:
    f.write(model.to_json())
```
