# Graduation settings

Finally, the deployed tables are smoothed with Whittaker-Henderson graduation, the actuarial
smoother for rating factors. Each table picks its own strength by generalized cross-validation
(GCV), so a table whose roughness is real shape is left untouched. No objects are held out for
it. The outcome is recorded in the `graduation_report_` attribute. See
[Pruning, banding and graduation](../algorithm/readable-tables.md) for details.

Graduation is skipped for models with `monotone_constraints` and is not supported for
multiclassification.

## graduate {#graduate}

#### Description

Smooth the deployed tables with Whittaker-Henderson graduation.

`None` and `True` turn graduation on, and `False` turns it off (the `graduation_report_`
attribute is then not set). An explicit `True` requires `prune=True`, and raises an error for
multiclassification. Whether a smoothed table is what you want to deploy is a modelling
decision: graduation has no accuracy gate.

**Type**

bool

**Default value**

None (on)

## graduation_alpha {#graduation_alpha}

#### Description

A fixed smoothing strength for every table, instead of the strength each table picks by
generalized cross-validation. 0 turns off the smoothing of the dense tables, which is useful
together with [`graduation_high_order_alpha`](#graduation_high_order_alpha).

**Type**

float

**Default value**

None (each table picks its own strength)

## graduation_high_order_alpha {#graduation_high_order_alpha}

#### Description

The strength, in the range $[0; 1]$, of an additional neighbour smoothing step for effects of
order 3 to 8 that are stored in factored form. Unlike the smoothing of the dense tables, its
strength is fixed rather than selected by cross-validation.

Numeric axes are smoothed along their order. Categorical axes and the edges between missing and
non-missing values are not smoothed. Effects with more than 1 024 stored boxes, or that would
add more than 4 096 boxes in total or 2 000 000 cells, are skipped; the details are in
`graduation_report_`.

0 turns it off. Requires `prune=True` and graduation turned on; not supported with
`monotone_constraints` or for multiclassification.

**Type**

float

**Default value**

0.0 (off)
