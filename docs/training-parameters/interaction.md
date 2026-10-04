# Interaction settings

These parameters control which interactions the trees build. The highest interaction order is
set by [`max_interaction_order`](common.md#max_interaction_order) and the resolution of the
trees by [`max_depth`](common.md#max_depth). See
[Choosing the tree structure](../algorithm/tree-structure.md) for details.

## interaction_gain_hurdle {#interaction_gain_hurdle}

#### Description

The interaction hurdle. While a tree grows, a split on a feature the tree does not use yet
raises the tree's interaction order. Such a split is selected only if its gain is large enough
relative to the gain of the tree's first split, and if it beats the best split on a feature the
tree already uses. Otherwise the tree keeps refining the features it already has. This is soft
heredity: interactions are admitted only on real evidence. A split that would raise the tree's
order above 3 faces a doubled hurdle.

The value is the strength of the hurdle and must be finite and non-negative. 0 turns the hurdle
off: each level takes the best split on a feature the tree does not use yet, and splits again on
a feature it already uses only when no new feature is available.

**Type**

float

**Default value**

2.0

## interaction_gain_hurdle_mode {#interaction_gain_hurdle_mode}

#### Description

How [`interaction_gain_hurdle`](#interaction_gain_hurdle) is applied.

Possible values:

- `adaptive` — The hurdle starts at full strength and relaxes as the gains of the first splits
  fade during the training. Three-way interactions face a stricter hurdle than two-way ones, and
  must also be supported by evidence for their pairs.
- `fixed` — The hurdle is applied as given throughout the training.

**Type**

string

**Default value**

adaptive

## table_budget_cells {#table_budget_cells}

#### Description

The cell budget of the table-size prior, which steers the trees toward smaller tables.

When the splits are ranked, the score of a candidate is multiplied by

$$
\sqrt{\frac{budget}{\max(budget, cells)}}
$$

where $cells$ is the number of cells of the table the split would contribute to (the product of
the numbers of cells of its features). The prior never rejects a split and never caps the
size of an exported table.

**Type**

int

**Default value**

None. Depends on the tree structure:

- `max_depth` and `max_interaction_order` both at most 3: 2 000 000 (effectively no prior)
- otherwise: 4096

## table_budget_order_shrink {#table_budget_order_shrink}

#### Description

How much [`table_budget_cells`](#table_budget_cells) shrinks for each interaction order above 3.
A $k$-way table with $k > 3$ is measured against
$\frac{budget}{table\_budget\_order\_shrink^{k - 3}}$ cells, because a table with more axes is
harder to read at the same number of cells. 1 turns the shrinking off.

**Type**

float

**Default value**

2.0
