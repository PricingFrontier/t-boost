# Exposure and offsets

## Exposure {#exposure}

For rate models, such as claim frequency, the objects are observed over different exposures
(for example, policy durations). Pass the exposure to `fit` instead of dividing the target by it:

```python
model = TBoostRegressor(objective="poisson")
model.fit(train_data, "ClaimCount", exposure="Exposure")
```

The model then predicts a rate per unit of exposure: the predicted rate is multiplied by the
exposure before it is compared with the target, which adds $\log(exposure)$ to the raw score.
The exposure is supported for the `poisson`, `gamma` and `tweedie` objectives and for binary
classification.

`predict` returns the rate per unit of exposure, and an exposure column in the data is ignored
when predicting. The expected total for an object is `predict(X) * exposure`. Pass the exposure
to `predict_contributions` to decompose the expected total instead of the rate.

The exposure also weights the data the tables are purified on (see
[Rating tables](../model-analysis/rating-tables.md#purification)), and the `support` of every
table cell is the exposure-weighted count of its objects.

## Offset {#offset}

An offset is a link-scale value added to the raw score of each object and never learned, the
equivalent of XGBoost's `base_margin` and LightGBM's `init_score`. Use it, for example, to train
a model on top of an existing rating structure:

```python
model.fit(train_data, "ClaimCount", exposure="Exposure", offset="BaseLogRate")
preds = model.predict(test_data, offset="BaseLogRate")
```

Prediction methods add the offset only when it is passed to them again; otherwise objects are
scored with a zero offset. For the log and logit links, the offset is added to the exposure
offset.

## Sample weights {#sample-weights}

`sample_weight` multiplies each object's contribution to the loss. For rate models, use the
exposure rather than a weight to express the observation period: the exposure enters the
prediction, a weight does not.
