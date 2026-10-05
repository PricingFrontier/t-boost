# to_bytes

Serialize the trained model to a compact binary format.

The result holds the training parameters, the fitted metadata and the model (its rating tables),
and is read back with `from_bytes`. A loaded model predicts identically to the original. See
[Saving and loading models](../../features/saving-models.md).

## Method call format {#call-format}

```python
to_bytes()
```

## Return value {#output-format}

bytes

## Usage examples {#usage-examples}

```python
with open("model.tboost", "wb") as f:
    f.write(model.to_bytes())
```
