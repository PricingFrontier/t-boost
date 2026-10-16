# check_bindings

Raise an error if a parameter set away from its default had no effect on the fit.

This is the strict counterpart of the `binding_report_` attribute, for a model whose parameters
are part of a filed or otherwise defended model, where a silently ignored setting is a defect.

## Method call format {#call-format}

```python
check_bindings()
```

## Return value {#output-format}

None. Raises `ValueError` listing every parameter whose status in `binding_report_` is `INERT`
(it had no effect on this fit), `OVERRIDDEN` (another parameter's value voided it) or
`UNAVAILABLE` (whether it had an effect cannot be determined). Call it before saving the model:
the pruning report is not stored with a saved model, so on a loaded model every pruning
parameter set away from its default is `UNAVAILABLE`.

## binding_report_ {#binding_report_}

One entry per checked parameter set away from its default, with `param`, `value`, `status`
(`HONOURED`, `INERT`, `OVERRIDDEN` or `UNAVAILABLE`), `reason` and `overridden_by`.

The report covers parameters whose effect depends on other settings: the pruning gates and
limits (for example, every pruning parameter when `prune=False`, and the parameters of the
pruning selector that did not run), `monotone_constraints` and deprecated spellings. It is not an audit of every parameter.

## Usage examples {#usage-examples}

```python
from t_boost import TBoostRegressor

model = TBoostRegressor(objective="poisson", prune_table_budget=4)
model.fit(train_data, "ClaimCount", exposure="Exposure")
for row in model.binding_report_:
    print(row["param"], row["status"], row["reason"])
model.check_bindings()
```
