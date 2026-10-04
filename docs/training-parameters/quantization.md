# Quantization settings

Before the training, every numerical feature is quantized into bins, and the trees only split
on bin borders. Missing values get a bin of their own (see
[Missing values processing](../algorithm/missing-values.md)).

## max_bin {#max_bin}

#### Description

The maximum number of bins for numerical features, not counting the bin for missing values.
Allowed values are integers from 2 to 254 inclusively.

If a feature has at most this many distinct values, each value gets its own bin, with the
borders halfway between neighbouring values. Otherwise the borders are weighted quantiles of the
feature's values, calculated on a deterministic sample of up to 200 000 values.

The borders bound the grid of the rating tables: a table axis never has more cells than its
feature has bins, plus the cell for missing values.

**Type**

int

**Default value**

254
