# predict_contributions

Calculate the contribution of each rating table to the prediction for every object.

For every object

$$
base\_value + \sum contributions = raw\ score\ (link\ scale), \qquad inverse\_link(raw\ score) = prediction
$$

exactly, because the deployed model *is* its rating tables: `base_value` is the intercept and
each contribution is one table's value for the object. The output format matches rustystats'
`GLMModel.predict_contributions`. See [Prediction contributions](../../model-analysis/contributions.md).

--8<-- "_snippets/feature-matching-note.md"

## Method call format {#call-format}

```python
predict_contributions(X,
                      *,
                      exposure=None,
                      offset=None,
                      split_interactions=False,
                      return_format="records",
                      validate=True,
                      atol=1e-06,
                      rtol=1e-06)
```

## Parameters {#parameters}

--8<-- "_snippets/params/X-apply.md"

### exposure

#### Description

The exposure of each object, for a model with a log link. A string names a column of a polars
`X`. It adds an `"exposure"` contribution of $\log(exposure)$, so `prediction_value` becomes the
expected total (`predict(X) * exposure`) instead of the rate per unit of exposure that `predict`
returns.

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- list
- string

**Default value**

None

### offset

#### Description

A link-scale offset for each object, as passed to the `offset` parameter of `fit`. It adds an
`"offset"` contribution, so the identities above hold for the raw score including it. A string
names a column of a polars `X`. Regression and binary classification only.

**Possible types**

- numpy.ndarray of shape `(object_count,)`
- polars.Series
- list
- string

**Default value**

None

### split_interactions

#### Description

How interactions are reported.

- `False` — One contribution per table: main effects (`term_type="main"`) and interactions
  (`term_type="interaction"`, named `"a:b"`).
- `True` — One contribution per input feature (`term_type="feature"`): each interaction is
  shared equally among its features. These are exact interventional Shapley values.

**Possible types**

bool

**Default value**

False

### return_format

#### Description

The format of the result.

- `records` — One dictionary per object, with a nested `contributions` list.
- `dataframe` — A long polars DataFrame with one row per `(row_index, term)`, or per
  `(row_index, class, term)` for a multiclassification model.
- `matrix` — A [ContributionMatrix](../contribution-matrix.md) holding a dense
  `objects × terms` matrix. The fastest format.

**Possible types**

string

**Default value**

records

### validate

#### Description

Check both identities above against the model's own raw score and prediction (the class
probabilities for a classifier), and raise `ValueError` if they do not hold.

**Possible types**

bool

**Default value**

True

### atol, rtol

#### Description

The tolerance of the check: $|delta| \le atol + rtol \cdot |actual|$. Predictions are computed in
`float32`, so the defaults are looser than rustystats'.

**Possible types**

float

**Default value**

1e-06

## Return value {#output-format}

With `return_format="records"`, a list with one dictionary per object:

- `family`, `link`, `output_space`, `prediction_space` — Describe the model.
- `base_value` — The intercept.
- `sum_contributions` — The sum of the contributions.
- `prediction_from_contributions` — `base_value + sum_contributions`, the raw score.
- `prediction_value` — The prediction.
- `contributions` — One entry per term, with `term`, `term_type`, `feature`, `feature_value`,
  `contribution` and `rank` (by absolute size).

For a multiclassification model each object instead has `family="multinomial"`,
`link="softmax"`, `prediction_space="probability"` and a `classes` list with one entry per class
(`class` and the keys above): the base value and the contributions add up to that class's raw
score, and `prediction_value` is its probability.

`ValueError` is raised for an exposure with a model that does not have a log link, for a
non-positive exposure, and for a model saved by t-boost 0.6 or earlier, which was not stored as
rating tables.

## Usage examples {#usage-examples}

```python
records = model.predict_contributions(test_data.head(5))
for term in records[0]["contributions"]:
    print(term["term"], term["contribution"])

matrix = model.predict_contributions(test_data, return_format="matrix")
print(matrix.terms, matrix.values.shape)
```
