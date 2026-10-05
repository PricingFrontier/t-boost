# Purification settings

The rating tables are purified against a reference measure: each table is centred so that its
weighted mean along every axis is zero under that measure. The measure decides which table owns
the part of the score that could sit in more than one table. See
[Rating tables](../model-analysis/rating-tables.md#purification) for details.

## ref_measure {#ref_measure}

#### Description

The reference measure the tables are purified against.

Possible values:

- `exposure` — Each axis is weighted by the distribution of its feature in the training data,
  weighted by sample weight times exposure, plus a small floor
  ([`measure_floor`](#measure_floor)) that keeps every weight positive.
- `product_marginals` — The number of objects in each cell, blended half and half with a
  uniform weighting. Kept so that models trained with earlier versions can be reproduced.
- `uniform` — Every cell has the same weight.

The measure changes which tables pruning keeps, and so the deployed model. To look at a trained
model under another measure without changing its predictions, use the `ref_measure` argument of
the `tables` method.

**Type**

string

**Default value**

None (`exposure`)

## measure_floor {#measure_floor}

#### Description

The total mass of the floor of the `exposure` reference measure, as a fraction of the
training data's mass. It is spread evenly over the cells of each axis so that every weight is
strictly positive. The value must be finite and positive. Used only with
`ref_measure="exposure"`.

**Type**

float

**Default value**

0.001
