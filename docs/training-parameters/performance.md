# Performance settings

## n_jobs {#n_jobs}

#### Description

The number of threads to use during the training and prediction.

Optimizes the speed of execution. This parameter doesn't affect results: a model is
bit-identical whatever the number of threads.

Possible values:

- `None` — Use all processor cores.
- A positive integer — Use that many threads.
- A negative integer $n$ — Use $cpu\_count + 1 + n$ threads, so `-1` uses all cores and `-2`
  all but one.

**Type**

int

**Default value**

None (the number of threads is equal to the number of processor cores)

## hist_precision {#hist_precision}

#### Description

The precision of the gradient histograms used to search for splits.

Possible values:

- `full` — Exact, full-precision accumulation.
- `quantized` — Faster accumulation in quantized integers.

**Type**

string

**Default value**

None (`full`)

## refine_closed_form_tier2 {#refine_closed_form_tier2}

#### Description

Calculate the leaf refinement steps (see
[`leaf_refine_steps`](common.md#leaf_refine_steps)) of a `poisson` model in closed form instead
of with an exact line search. This is faster, and the leaf values differ from the exact path by
about $10^{-7}$. `False` uses the exact path.

**Type**

bool

**Default value**

True

## incremental_mu {#incremental_mu}

#### Description

Update the predicted mean $\mu = e^{a}$ of a `poisson` model incrementally from one iteration to
the next instead of recalculating it. This is faster, and the predictions differ from the exact
path by about $10^{-9}$ to $10^{-6}$.

**Type**

bool

**Default value**

False
