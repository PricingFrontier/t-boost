# Prediction contributions

A vector of contributions of each rating table to the prediction for every input object, and
the base value of the model (its intercept).

- $base\_value$ is the intercept $f_0$ of the model.
- The contribution of table $u$ to object $x$ is $f_u(x_u)$, the value of the cell the object
  falls in.

For a given object the sum of the base value and the contributions is equal to the raw score of
the model on this object, and applying the inverse link gives the prediction:

$$
base\_value + \sum\limits_{u \in U} f_{u}(x_{u}) = a(x), \qquad prediction = link^{-1}(a(x))
$$

The decomposition is exact, because the deployed model *is* its rating tables: the contributions
are not estimated. For a model with a log link, the exponentials of the contributions are the
multiplicative relativities of the object.

## Contributions by feature {#by-feature}

With `split_interactions=True`, each interaction is shared equally among its features, giving one
contribution per input feature:

$$
\phi_{j}(x) = \sum\limits_{u \ni j} \frac{f_{u}(x_{u})}{|u|}
$$

Because every table is purified, these are exact interventional Shapley values of the raw score:
no sampling and no approximation is involved.

## Exposure and offset {#exposure-offset}

`exposure` and `offset` passed to `predict_contributions` appear as contributions of their own
(`log(exposure)` and the offset), so the identity holds for the raw score including them. With
the exposure, the prediction is the expected total for the object rather than the rate per unit
of exposure.

## Usage examples {#usage-examples}

```python
records = model.predict_contributions(test_data.head(5))
row = records[0]
print(row["base_value"], row["prediction_value"])
for term in row["contributions"]:
    print(term["term"], term["feature_value"], term["contribution"])
```

See [predict_contributions](../python-reference/tboostregressor/predict_contributions.md) for the
parameters and the output formats. The output matches rustystats'
`GLMModel.predict_contributions`.
