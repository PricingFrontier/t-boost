# set_params

Set the training parameters.

## Method call format {#call-format}

```python
set_params(**params)
```

## Parameters {#parameters}

### **params

#### Description

A list of parameters to start training with.

Format:

```
parameter_1=<value>, parameter_2=<value>, ..., parameter_N=<value>
```

An example of the method call:

```python
model.set_params(n_trees=500, n_jobs=2, prune_main_effects=True)
```

See [Training parameters](../../training-parameters/index.md) for the full list of parameters.

**Possible types**

key=value format

**Default value**

Required parameter

## Return value {#output-format}

The estimator itself. The new values apply to the next call to `fit`.
