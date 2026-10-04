# t-boost improvements for the Haute integration

A handover specification for the t-boost project, first written on 4 October 2026 against
t-boost 0.6.2 and revised the same day after review. It lists the library changes that
would let Haute train, validate and explain t-boost models on the same terms as its other
gradient-boosting families (XGBoost and LightGBM), and remove the workarounds Haute's
adapter carries today. It is a supporting report: it owns no Haute work. The Haute change
that consumes these improvements is `TBOOST-11` in the [t-boost roadmap](t-boost.md).

Each requirement states the behaviour observed in 0.6.2, the API, its exact semantics, the
tests t-boost ships with it, and what it unlocks in Haute.

## Revision notes (4 October 2026)

- **Haute does not refit t-boost.** Haute investigated refitting on training plus
  validation rows after selection and found it not worth doing. The early-stopped selection
  fit is the deployed model. Everything that existed only to transfer a round count or a
  table set to a refit is dropped: ensemble-level (lockstep) stopping, `best_iteration_`,
  the fixed-round refit test, and `keep_tables`.
- **R1** is per-bag early stopping on one shared evaluation set, which the engine's
  existing shared-holdout design already supports.
- **R4** is withdrawn. `prune_on="eval_set"` would score tables on one model and one
  holdout with no standard error; single-split selection was measured to drop tables on
  noise, which is why pruning uses fold refits. `keep_tables` was only for refits.
- **R6** is named `offset` (rustystats' name for a link-scale offset), and is optional at
  scoring time, like `exposure`.
- **R9** adds a matrix format. The records and dataframe formats keep their shape, which
  matches rustystats' `GLMModel.predict_contributions`.
- **R11**: `PrecisionWarning` is already emitted once per estimator; only the typed
  imports remain.

## How Haute uses t-boost

Haute trains every model through one evaluation plan. Source rows are split into
development rows and an optional **final-test** partition that no fit or selection ever
sees. Development rows are split again into a **training** partition and a
**validation** partition (or cross-validation folds). A **selection fit** trains on the
training partition and is evaluated, and early-stopped, on the validation partition;
tuning compares candidates on that validation evidence. For t-boost the selection fit is
also the deployed model: there is no refit. The final-test partition is scored once at the
end.

t-boost 0.6.2 has no validation-set argument. Each fit carves its own early-stopping
holdout from the rows it is given (`validation_fraction`, per bag), so Haute's validation
partition only scores the fit and never drives stopping. Fit evidence reports no fitted
rounds and no stopping reason, and there is no loss curve and no progress during a fit.

## Summary

| ID | Requirement | Priority | Unlocks in Haute |
|---|---|---|---|
| R1 | External evaluation set for early stopping | P1 | Stopping on Haute's validation partition; every training row trains |
| R2 | Fitted round counts and stopping reasons | P1 | Fit evidence: fitted rounds and stopping reason |
| R3 | Per-round metrics history and an iteration callback | P2 | Live loss curve, training progress, cancellation |
| R4 | ~~Pruning against the validation set; keep-set for refits~~ | Withdrawn | — |
| R5 | An official metadata slot in the saved model | P2 | Self-describing `.tboost` files without relying on unknown-key tolerance |
| R6 | Arbitrary link-scale `offset` | P2 | Offsets under RMSE and Logloss |
| R7 | Unknown-category policy | P2 | Haute stops pre-checking categorical levels |
| R8 | Refuse `exposure` under `squared_error` and logistic | P3 | One less silent case |
| R9 | Contributions as a matrix keyed by feature tuples | P2 | Faster explanations; feature names may contain `:` |
| R10 | Public cell indices and documented border semantics | P2 | Cell-level validation and unfolding into rating tables |
| R11 | Typed top-level imports | P3 | Cleaner adapter |
| R12 | A documented saved-model compatibility rule | P3 | An exact retrain rule in Haute |

## Requirements

### R1. External evaluation set for early stopping

**Observed in 0.6.2.** `fit` takes `X, y, sample_weight, exposure, groups`. Early stopping
always uses an internal carve of the fit rows (`validation_fraction`, default 0.1), drawn
per bag; there is no way to supply the rows to stop on.

**API.**

```python
model.fit(
    X, y, sample_weight=w, exposure=e, groups=g,
    eval_set=(X_val, y_val),
    eval_sample_weight=w_val,     # optional
    eval_exposure=e_val,          # required when exposure is given
    eval_offset=o_val,            # required when offset is given (R6)
)
```

`X_val` is matched to the fit features by name, as `predict` matches a polars frame. A
string for `y_val`, `eval_sample_weight`, `eval_exposure` or `eval_offset` names a column
of `X_val`, as the fit-row arguments name columns of `X`.

**Semantics.**

- With `eval_set`, every bag early-stops on the evaluation rows. No internal holdout is
  carved, so every fit row the bag draws trains. `validation_fraction` is ignored and
  `binding_report_` records it as overridden by `eval_set` (sklearn-style parameters
  cannot tell an explicit 0.1 from the default).
- Bags still stop independently, each at the round minimising its own evaluation deviance
  (with `early_stopping_rounds`, `early_stopping_adaptive` and `early_stopping_min_delta`
  keeping their meaning). Each bag's count is reported by R2.
- The evaluation rows are binned on the fit rows' grid and encoded by the fit's
  categorical encoders. They never enter a bag, the intercept, the cell refit,
  purification, the reference measure, pruning or graduation.
- The holdout slope recalibration (`reanchor_slope`) fits on the stopping holdout, so it
  does not run with `eval_set`. An explicit `reanchor_slope=True` with `eval_set` raises.
- With `prune=True`, the pruning fold fits carve their own stopping holdouts from the
  fit rows, as without `eval_set`.
- `eval_set` is supported for regression and binary classification. A multiclass
  (K >= 3) fit with `eval_set` raises.
- Without `eval_set`, behaviour and model bytes are unchanged.

**Tests t-boost ships.**

- Each bag's evaluation-deviance curve has its minimum at the bag's reported round count.
- Evaluation rows never reach training: changing one evaluation row's target to an
  extreme value leaves every bag's per-round training-deviance history (R3) identical, so
  only the kept round counts can move.
- Exposure on the fit rows without `eval_exposure` raises; `reanchor_slope=True` with
  `eval_set` raises; multiclass with `eval_set` raises.
- `n_jobs` determinism holds with `eval_set`.

**Unlocks in Haute.** Selection fits early-stop on Haute's validation partition, every
training row trains, and the deployed model is the one validated.

### R2. Fitted round counts and stopping reasons

**Observed in 0.6.2.** The deployed model is a tables-only model with no round count.

**API.** Fitted attributes, kept through `to_json`/`from_json` and
`to_bytes`/`from_bytes`:

| Attribute | Meaning |
|---|---|
| `n_trees_per_bag_` | Boosting rounds each bag kept, in bag order |
| `stopping_reason_per_bag_` | Why each bag stopped: `"early_stopping"`, `"max_trees"`, `"no_split"` or `"callback"` |
| `n_trees_` | The largest per-bag count |
| `stopping_reason_` | One summary: `"callback"` if any bag was stopped by a callback, else `"early_stopping"` if any bag early-stopped, else `"no_split"` if any bag ran out of admissible splits, else `"max_trees"` |

**Semantics.** The counts describe the deployed fit's boosting, before pruning; pruning does
not change them. A multiclass fit reports one entry (its classes advance together). A
model saved by 0.6.2 or earlier loads with these attributes set to `None`.

**Tests t-boost ships.** Each stopping reason is produced by a constructed fit; the
attributes survive both formats; a fit without early stopping reports `"max_trees"` and
`n_trees_ == n_trees`.

**Unlocks in Haute.** Fit evidence records the configured ceiling, the fitted rounds and
the stopping reason (`none`, `validation` or `native_exhaustion` in Haute's terms).

### R3. Per-round metrics history and an iteration callback

**Observed in 0.6.2.** No per-round output.

**API.**

```python
model.fit(..., callbacks=[on_round])

def on_round(info: dict) -> bool | None:
    ...  # return True to stop boosting this fit after the current round
```

`info` holds `phase` (`"fit"` for the deployed fit, `"prune_fold"` for a pruning fold
fit), `bag`, `n_bags`, `round` (1-based), `n_trees` (the ceiling), `train_deviance` and
`eval_deviance` (the stopping deviance on the `eval_set` or the internal holdout; `None`
when there is none). And a fitted `evals_result_` for the deployed fit:

```python
{"train": {"deviance": [[...], ...]}, "eval": {"deviance": [[...], ...]}}  # per bag, per round
```

**Semantics.** Deviances are the objective's mean deviance, weighted by `sample_weight`
(with exposure as an offset for the log-link objectives), as early stopping measures it.
Bags run in parallel, so calls for different bags interleave; within a bag rounds arrive
in order. Callbacks run on the calling thread, so they may touch Python state and may
raise: an exception stops every bag and propagates unchanged. Returning `True` stops every
bag of that fit after its current round; a stopped deployed fit records
`stopping_reason_ == "callback"`. `evals_result_` lists the rounds trained, including any
past the kept round.

**Tests t-boost ships.** Rounds arrive in order per bag; the last `eval_deviance` of each
bag matches the recorded history; returning `True` stops the fit with reason `"callback"`;
an exception propagates.

**Unlocks in Haute.** The live loss chart during training, accurate progress for long fits,
and prompt cancellation of a training job.

### R4. Withdrawn

Pruning keeps its fold-refit selection. Haute does not refit, so it needs no keep-set; the
deployed tables are listed in `pruning_report_`.

### R5. An official metadata slot in the saved model

**Observed in 0.6.2.** Haute's `.tboost` file is `to_json()` with an added top-level
`"haute"` key, which works only because `from_json` ignores unknown keys. `to_bytes()`
cannot carry anything extra.

**API.** `model.metadata: dict[str, JSON]`, set by the caller after fitting, stored in the
envelope header of both formats and returned unchanged by `from_json`/`from_bytes`.
t-boost never reads it. A fit keeps it.

**Semantics.** Values must be JSON-serialisable; anything else raises at save time. It does
not affect predictions or `tables()`.

**Tests t-boost ships.** Round trip through both formats; a non-JSON value raises; a model
with metadata predicts identically to one without.

**Unlocks in Haute.** The Haute record (feature order, categorical levels, task, link,
offset, class labels) moves into the documented slot, and Haute can use the binary format.

### R6. Arbitrary link-scale offset

**Observed in 0.6.2.** `exposure` is the only offset: it enters as `log(exposure)` and is
meaningful only under the log-link objectives.

**API.** `fit(..., offset=o)` (array or a column name of `X`), and the same keyword on
`predict`, `predict_raw`, `predict_proba`, `decision_function` and
`predict_contributions`: a link-scale term added to every row's raw score, under any
regression or binary objective. `exposure` keeps its meaning; supplying both adds both.

**Semantics.** The offset is never learned, purified into the tables, or counted as mass.
At scoring time it is optional, like `exposure`: omitted, the model scores on a zero
offset (as XGBoost's `base_margin` and LightGBM's `init_score` do). Given, it is added to
the raw score, and `predict_contributions` reports it as an `"offset"` term, so
`base_value + sum(contributions)` equals the raw score including it. Multiclass rejects it.

**Tests t-boost ships.** A constant offset under `squared_error` shifts predictions by that
constant; a varying offset is added exactly in `predict_raw`; the fit learns around it;
contributions stay exact with it.

**Unlocks in Haute.** Offsets under every loss.

### R7. Unknown-category policy

**Observed in 0.6.2.** A categorical value not seen in training scores in the axis's
`default_cell` without any signal.

**API.** A constructor parameter `unknown_category="default_cell" | "error"` (default
unchanged), and a fitted `categories_` mapping each categorical feature name to its fitted
levels.

**Semantics.** With `"error"`, every scoring call (`predict*`, `predict_contributions`,
`tables` and `cell_indices` on new rows) raises a `ValueError` naming the feature and up
to five unseen values. Nulls are never unknown: they score in the missing level.

**Tests t-boost ships.** Each scoring entry point raises under `"error"` and scores the
default cell under `"default_cell"`; a null never raises.

**Unlocks in Haute.** Haute sets `"error"` and drops its own level check.

### R8. Refuse `exposure` under `squared_error` and logistic

**Observed in 0.6.2.** `TBoostRegressor(objective="squared_error")` and `TBoostClassifier`
accept `exposure` without complaint, although an exposure offset has no meaning under the
identity or logit link.

**API.** Raise a `ValueError` naming the objectives that accept exposure, and pointing at
`offset` (R6) for the others. This is a breaking change for callers who passed it.

### R9. Contributions as a matrix keyed by feature tuples

**Observed in 0.6.2.** `predict_contributions` returns records or a long polars frame keyed
by a term string that joins feature names with `:`. Haute rebuilds a matrix and splits
names on `:`.

**API.** `predict_contributions(X, return_format="matrix", split_interactions=...)`
returning a result object:

```python
result.base_value   # float64 array, one per row (link scale)
result.values       # float64 array, rows x terms
result.terms        # list[tuple[str, ...]]: each term's feature names
```

The records and dataframe formats are unchanged.

**Semantics.** `base_value + values.sum(axis=1)` equals `predict_raw` (plus any exposure and
offset passed, which appear as their own terms); an intercept-only model returns zero
columns; with `split_interactions=True` the columns are the fitted features in
`feature_names_in_` order, a feature no table uses being a zero column. Multiclass
returns one matrix per class.

**Tests t-boost ships.** Exactness against `predict_raw`; a feature named `a:b`
round-trips; zero columns for an intercept-only model; column order under
`split_interactions=True`.

### R10. Public cell indices and documented border semantics

**Observed in 0.6.2.** `tables()` exports each axis's `borders` and categorical `levels`,
but border closure, the float32 comparison, the missing cell and empty categorical cells
are undocumented, and the per-row cell lookup is private.

**API.**

- A public `cell_indices(X)` returning a dict keyed by each deployed table's feature-name
  tuple (the `feature_names` of its `tables()` entry, and the R9 matrix term), holding an
  `(n_rows, order)` array of every row's cell on each axis. Banded tables report their band.
  Regression and binary models only; factored over-budget effects have no cells.
- Documentation, in the `tables()` docstring, of border closure, the float32 comparison,
  the missing cell, empty categorical cells, the `default_cell`, and the row-major order of
  `values` and `support`.

**Tests t-boost ships.** For random rows, the table value at `cell_indices` equals that
row's contribution for every table; a value equal to a border lands in the lower cell; a
float64 value just above a border that rounds onto it in float32 lands in the lower cell.

**Unlocks in Haute.** Cell-level actual-versus-expected (`TBOOST-02`) and exact unfolding
into Haute's Banding and Rating Step nodes (`TBOOST-03`).

### R11. Typed top-level imports

**Observed in 0.6.2.** `t_boost.TBoostRegressor`, `TBoostClassifier` and `PrecisionWarning`
resolve through a module `__getattr__`, so type checkers see them as `object`.

**API.** Import the names under `if TYPE_CHECKING:` in `t_boost/__init__.py`.
`PrecisionWarning` is already emitted once per estimator.

### R12. A documented saved-model compatibility rule

**Observed in 0.6.2.** A saved document carries `schema_version`, `t_boost_version` and a
nested model schema version; which versions a release reads is not stated.

**API.** Document the rule in the `from_bytes`/`from_json` docstrings: which envelope
schema versions a release reads, that `t_boost_version` is informational, that any
readable document predicts identically under the reading release, and that an unreadable
one raises `SerializationError` naming both versions.

## Behaviour Haute relies on today

These hold in 0.6.2, Haute's integration depends on them, and they are kept:

- A fit is identical across `n_jobs` values for a fixed seed.
- `to_json`/`from_json` and `to_bytes`/`from_bytes` round-trip predictions exactly, and
  loading a classifier document as a regressor raises `SerializationError`.
- `tables(X)` on the fit rows reproduces every row's contribution, including cells with
  no training support, whose `support` is 0 and whose `values` are the scored values.
- `support` is the training mass per cell: the sum of `sample_weight` times `exposure`.
- `groups` keeps every internal carve and bag group-honest.
- String, Categorical and Enum columns are encoded natively; a null categorical value
  scores in the missing level.

## Not requested

A multiclass path for Haute, GPU training, and changes to the fitting algorithm, pruning
heuristics or defaults are out of scope. Apart from R8's new error, every requirement is
opt-in and leaves the predictions of an existing fit unchanged.
