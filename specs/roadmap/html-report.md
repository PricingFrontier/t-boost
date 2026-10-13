# HTML model report — plan

Status: plan, revision 2 (2026-10-13), not started. Branch `html-report` off `main` @ 1f480c2.
Revision 2 folds in two read-only reviews (Fable 5.1; Codex gpt-6-astra, ultra effort); the
review findings are summarised at the end.

## Goal

One self-contained HTML page that gives a data scientist what they need to decide whether a
fitted t-boost model is ready for production: how well it fits (in and out of sample), its
structure, charts and tables of every rating factor, and a list of review flags. Haute must be
able to attach it to an MLflow run with no extra work.

## Constraints

- **No MLflow code in t-boost** (haute owns MLflow; persistence mirrors rustystats). t-boost
  returns data and an HTML string; haute logs them.
- **No new runtime dependencies** (numpy, polars, threadpoolctl only). No CDN, no network.
- **No JavaScript.** The page must be fully usable where scripts are blocked or allowed
  (upstream MLflow's viewer uses `sandbox="allow-scripts"`; haute's pinned version is untested).
- **Regression, binary and multiclass** classification (user decision 2026-10-13).
- rustystats has `diagnostics(train_data, test_data=...)` and no HTML output; nothing to mirror.

## Pre-existing bugs (found by the reviews) — fixed 2026-10-13 on `html-report`

1. `actual_vs_expected`/`pricing_report` ignore `offset`: `_expected_response` calls
   `predict(X)` with no offset (sklearn.py:7514, classifier 9129), so an offset-fitted model's
   A/E is computed on a zero offset. There is no `_ae_requires_offset_` flag (4290-4291).
2. A/E arrays end at the highest *observed* cell (`minlength = c.max() + 1`, sklearn.py:3762),
   so they don't align with the grid's `n_bins`/`borders`.
3. `pricing_report`'s axis merge (3800-3807) overwrites a table axis's borders with the merged
   grid but keeps that table's categorical `levels`, which point at the table's *bands* when it
   is banded; its `joint_channels` guard reads a field `AxisExport` never writes.
4. Multiclass fits never set `_ae_requires_weight_` (only the single-output path does, 4290),
   so the explicit-weight rule is silently skipped for K ≥ 3.
5. `binding_report_` on a loaded model reads the missing `pruning_report_` (2796) and can
   report non-default parameters as INERT "because prune=False".

Fixes: (1) `offset=` on `actual_vs_expected`/`pricing_report`, `_expected_response` applies it,
new persisted `_ae_requires_offset_` flag; (2) lists padded to the grid's cell count; (3) new
native `_TableModel.merged_axes()` (core `TableBank::merged_axis_exports`) exports each raw
feature's merged-grid axis, used for `pricing_report`'s factor axes; (4) multiclass fit sets the
three flags; (5) prune knobs on a loaded pruned model report a new `UNAVAILABLE` status, which
`check_bindings` also raises on (persisting `pruning_report_` — open question 2 — would remove
it for new files).

## Public API

### `pricing_report` → `report_version: 2`

```python
report = model.pricing_report(
    X, y, *, sample_weight=None, exposure=None, offset=None, ref_measure=None,
    eval_set=None,               # (X_val, y_val), the fit() convention
    eval_sample_weight=None, eval_exposure=None, eval_offset=None,
    labels=("train", "eval"),    # dataset roles, used in the page and in metric keys
    name=None,                   # report name, shown as the heading
    thresholds=None,             # dict overriding the review thresholds
)
```

- Weight, exposure and offset must be passed explicitly when the fit used them (new
  `_ae_requires_offset_` flag, persisted like the other two). Strings name polars columns of
  `X` or `X_val`.
- `labels`: v1 described `X` as "the data passed", so v2 does not silently call it training
  data. The default is `("train", "eval")`; haute passes its own roles (e.g.
  `("train", "holdout")`). The page always states that it cannot verify which rows were held
  out.
- Version 2 keeps every version-1 key and adds:
  - `name`, `generated_at` (UTC), `model` (class, objective, link, `t_boost.__version__`,
    columns — `required_columns`, or positional `x0..` for a numpy-fitted model, whose
    `required_columns` raises — `metadata`, `fitted_at_`, `fit_duration_`, `classes_` as
    strings, `seed`, the full `get_params()` through the existing `_encode_param` codec with
    non-defaults marked);
  - `fit`: per-field availability (`{"value": ..., "available": bool, "reason": str}`) for
    `n_trees_`, `n_trees_per_bag_`, `stopping_reason_*`, `evals_result_`, pruning, graduation,
    `binding_report_`;
  - `datasets`: one entry per label with rows, mass, metrics, lift, Lorenz points, A/E by
    factor level (padded to the full grid), missing-value share, unseen values;
  - `drift`: PSI per feature between the two datasets;
  - `thresholds`: the values in force; `review`: the flags; `summary_metrics`.
- `deployed_tables` comes from `_deployed_table_count` (2751), not `pruning_report_["kept"]`.
- JSON-safe: `json.dumps(report, allow_nan=False)` succeeds (non-finite values become `null`
  with a reason).

### `t_boost.report_html(report, *, name=None) -> str`

Pure view of a version-2 report; needs no model, so the page can be rebuilt from the JSON
logged to MLflow. Raises on an unknown `report_version`. Deterministic for a given report
(`generated_at` comes from the report). Exported through `__init__`'s lazy `__getattr__` and
`__all__`.

### `model.html_report(X, y, ...) -> str`

`report_html(self.pricing_report(...))`, same arguments.

### Fit time: `fitted_at_`, `fit_duration_`

- Set at the end of a successful fit; rolled back by `_atomic_fit` on failure; added to both
  fitted-state clear-lists (set_params and `_clear_stale_fit_state`, 2628-2672) so a refit
  never keeps a stale time.
- Saved in the `to_bytes`/`to_json` envelope (schema version bumped); older files load with
  both `None` ("fit time unknown").
- Heading: `<name> · fitted 2026-10-13 14:02 UTC` (default name: class and objective).

### Persisted fit evidence

Haute loads models back from MLflow, so the evidence the report shows must survive
`to_bytes`: also save `pruning_report_`, `graduation_validation_` and `evals_result_` in the
envelope (tree counts and stopping reasons already are, sklearn.py:1439-1444). Older files: the
affected sections say "not stored with this model", and `binding_report_` is reported as
unavailable instead of claiming INERT.

### Thresholds

| key | default | check |
|---|---|---|
| `ae_train` | 0.02 | overall A/E on the first dataset within 1 ± value |
| `ae_eval` | 0.05 | overall A/E on the second dataset within 1 ± value |
| `factor_ae` | 0.10 | factor levels whose A/E is off by more than value… |
| `factor_min_mass` | 0.01 | …among levels holding at least this share of the mass |
| `psi_warn` | 0.10 | PSI per feature, warning |
| `psi_critical` | 0.25 | PSI per feature, critical |
| `train_eval_gap` | 0.05 | drop in deviance explained, in absolute points (not relative: that explodes near 0) |
| `unseen_mass` | 0.01 | mass share of unseen categorical values in the second dataset |

Unknown key → `ValueError`; non-finite or negative → `ValueError`; `psi_warn > psi_critical` →
`ValueError`; `None` disables a check. Values in force are written to the report and printed
next to every flag. Teams keep their own values in haute's config.

### `summary_metrics`

Flat `{str: float}`, finite values only. Keys `<label>_<metric>` (e.g. `eval_deviance`,
`eval_deviance_explained`, `eval_gini`, `eval_ae`) plus `n_trees`, `deployed_tables`,
`review_warnings`, `review_critical`; multiclass adds `<label>_<metric>_<class>` for per-class
values with the class label sanitised to `[A-Za-z0-9_]` and an index suffix on collision. Keys
match MLflow's name rules and stay under its length limit (checked in tests against the rules
as documented; MLflow is not installed here to test against).

## Statistics (contracts)

Notation: `w` sample weight, `e` exposure, `o` offset, `a = e·exp(o)` the multiplicative
offset for log links, `μ` the expected value of `y` per row (`_expected_response`, now with
the offset).

- **Deviance** on `y` against `μ`, weighted by `w` only: the same convention as the fit's
  guard and `evals_result_` (`_guard_unit_deviance`, sklearn.py:1232-1262), so the fit-quality
  table and the training curve agree. Squared error, Poisson, Gamma, Tweedie(`tweedie_rho`)
  via `metrics.mean_tweedie_deviance`; add `mean_binomial_deviance` and
  `mean_multinomial_deviance` to `metrics.py` (public, tested).
- **Null deviance** = deviance of the best constant model *on the same dataset*, with the same
  offsets: log link `μ0 = c·a` with Tweedie closed form `c = Σ w y a^{1-p} / Σ w a^{2-p}`
  (Poisson p=1, Gamma p=2 included); identity `μ0 = c + o`, `c` the weighted mean of `y − o`;
  logit and softmax: a 1-D (resp. K-dim) Newton solve for the intercept with the offset.
  Deviance explained = `1 − D/D0`; "unavailable" when `D0 = 0`.
- **Gini**: the exposure-weighted Lorenz (concentration) Gini — rows ranked by predicted rate
  `μ/a`, x = cumulative `w·e`, y = cumulative `w·y` — reported as `gini`; the normalised
  version (`/ perfect`) as `gini_normalised`. Rows with `e = 0` are excluded (counted in the
  report). Omitted for identity-link fits with negative targets (the helper clips at 0,
  metrics.py:81). Multiclass: one-vs-rest per class.
- **Lift**: own implementation (`metrics.lift_curve` is equal-count and does not merge ties):
  rows ranked by predicted rate, whole tie groups kept together, a group assigned to the band
  of the midpoint of its cumulative `w·e`; up to 10 bands, fewer when there are fewer distinct
  predictions. Deterministic, independent of row order. Per band: mass, rows, actual and
  predicted rate.
- **A/E**: by merged-grid cell, padded to `n_bins`; overall A/E `Σ w y / Σ w μ`; signed targets
  (identity link) report the difference `Σ w (y − μ)` instead of a ratio. Multiclass:
  one-vs-rest per class, `actual = Σ w·1[y=k]`, `expected = Σ w·p_k` (the all-class sum is
  identically 1, so not reported).
- **PSI** per feature between the two datasets on its own binning: numeric features on up to
  10 bins of equal first-dataset mass built from the merged-grid cells, categorical on levels
  (rare-pooled levels together), missing values as their own bin, unseen values as their own
  bin; shares floored at 1e-4 before the log. Thresholds 0.10/0.25 are calibrated for ≈10 bins,
  which is why the binning is capped. Multiclass uses class 0's grid (grids are validated equal
  across banks; merged grids are per bank).
- **Multiclass accuracy and class matrix**: weighted accuracy of the arg-max class; matrix rows
  = actual class, columns = predicted class, cells = Σ w·p_k (expected) beside Σ w (actual),
  class order = `classes_`.

## Content of the page

1. **Header** — name and fit time, class, objective/link, t-boost version, seed, dataset
   labels with rows/mass, columns, metadata; the hold-out caveat.
2. **Review flags** — every check with value, threshold and status (icon + label, never colour
   alone); a list for review, not a verdict:
   overall A/E per dataset; second dataset beats its null model; deviance-explained gap; early
   stopping (no second dataset and no validation; any bag ending on `max_trees`; info, not a
   warning, when the reason is unavailable); factor levels off by more than `factor_ae` with
   mass ≥ `factor_min_mass`; PSI; unseen categorical values; `binding_report_` INERT/OVERRIDDEN
   rows (only when available); per class for multiclass.
3. **Fit quality** per dataset — deviance, null deviance, deviance explained, Gini (both),
   overall A/E, rows, mass, Σ actual, mean predicted rate, predicted-rate distribution
   (percentiles); lift chart and Lorenz curve; multiclass adds accuracy and the class matrix,
   with lift/Lorenz per class.
4. **Training** — train/eval deviance by round when `evals_result_` exists (it is recorded only
   with `eval_set` or callbacks — the page says so otherwise); trees per bag; stopping
   reasons. Multiclass fits have no per-bag report: the section says so.
5. **Model structure** — intercept, tables by order (dense, sparse, factored counted
   separately), Sobol share per table, pruning path (OOB deviance against bank size, best
   marked), candidate tables kept or dropped, banding, graduation, guard. Joint-measure caveats
   (non-Shapley, not exactly purified, `_pricing.annotate_joint_export`) kept verbatim.
6. **Rating factors** — one card per feature, by importance:
   - effect chart of the 1-D table on the table's own axis: relativity (log link), odds
     multiplier (logit), additive effect (identity); bag-stability band (`se_band`) when present,
     labelled "bag stability, not a confidence interval"; mass panel below;
   - one-way chart: actual vs expected average by level on the merged grid, and, when the 1-D
     table is banded, also aggregated to its bands so both panels share an x-axis; mass panel
     below;
   - relativity range and rare-pooled share;
   - cell labels `missing`, `≤ b`, `(a, b]`, `> b`, or the category members;
   - CSS-only selectors for dataset and (multiclass) class.
   Two panels, never two y-scales on one chart.
7. **Interactions** — 2-D tables as SVG `<rect>` heatmaps (hover value from `<title>`),
   diverging blue–grey–red around no effect, zero support hatched; 2-D A/E on the same cells
   (via `cell_indices`). Drawn for the top N by Sobol share within a cell budget; sparse and
   factored effects listed, not drawn; order ≥ 3 listed with their Sobol share.
8. **Data checks** — missing share per feature per dataset, unseen values (from
   `unseen_values`, computed before any scoring), PSI table.
9. **Appendix** — parameters (non-defaults highlighted); rating tables up to a cell budget, the
   rest pointing to the logged JSON (`<details>` hides content but does not shrink the file).

## Robustness

- `unknown_category="error"` with unseen values in a dataset: scoring would raise
  (sklearn.py:3603-3607). Data checks run first; that dataset gets a critical flag and its
  prediction-based sections are marked unavailable; the report is still produced.
- Zero total weight, zero mass, empty datasets, absent classes in a dataset, `μ ≤ 0` where the
  deviance needs `μ > 0`: the metric is `null` with a reason, never an exception.
- Loaded models: per-field availability (see above); ≤ 0.6.x ensemble models without tables
  raise the existing clear error.
- Size budget: total heatmap cells and appendix cells capped (defaults chosen so a `prune=False`
  bank on 100k × 25 stays under ~5 MB; tested). Sparse tables are not densified for display.

## Rendering

- `python/t_boost/_report.py` assembles the report (metrics, lift, PSI, thresholds, flags);
  `python/t_boost/_html_report.py` renders it and only reads the dict.
- Inline SVG generated in Python; theme through CSS custom properties (SVG uses
  `var(--…)`), so one set of markup serves light and dark; print stylesheet.
- Palette and marks per the dataviz method (validated categorical palette, 2px lines, 24px max
  bars, recessive gridlines, legend for ≥ 2 series, a table view for every chart).
- Escaping: all text and attribute values HTML-escaped (including SVG `<title>`); element
  `id`s and selector names generated from indices, never from labels; no untrusted string in
  CSS.

## Multiclass support

1. Native (`crates/t-boost-py/src/lib.rs`, `_MultiClassTableModel`): `cell_indices(x, cat_x,
   class_index)` on the class bank's merged grid, and `raw_feature_names()`; stubs updated.
2. `actual_vs_expected`: multiclass path (one entry per class × feature, with a `class` key);
   the docs that say "binary only" updated.
3. Set `_ae_requires_weight_` in the multiclass fit (pre-existing bug 4).
4. `pricing_report`: tables per class (`tables()` already supports K ≥ 3; the per-class
   banks live under `["classes"]` in the envelope, the single-output `self._model.to_json()`
   path is not used); metrics, lift and checks per class. Multiclass `fit` refuses exposure and
   offset (sklearn.py:9311-9317), so the report refuses them too; `eval_set` is allowed (the
   report only scores it).
5. ≤ 0.6.x `_MultiClassModel` ensembles still raise: no grid.

## Haute integration (for haute, not done here)

```python
report = model.pricing_report(X, y, exposure="Exposure", eval_set=(X_val, y_val),
                              eval_exposure="Exposure", labels=("train", "holdout"),
                              name=run_name, thresholds=team_thresholds)
candidate.metrics.update(report["summary_metrics"])                      # log_batch
_log_json_artifact(client, run_id, report, "t_boost_report", "t_boost")  # full record
html = t_boost.report_html(report)  # temp .html → client.log_artifact(run_id, path, "t_boost")
```

## Tests — `python/tests/test_html_report.py` (+ native, metrics)

- Poisson with exposure, Gamma, Tweedie, squared error, binary with offset, multiclass
  (3 classes, categoricals); with and without `eval_set`; polars and pandas; string columns in
  `X_val`; numpy-fitted model.
- Pre-existing bugs: offset-fitted A/E equals the hand computation; A/E arrays have `n_bins`
  entries; banded categorical axis labels map to the right cells; multiclass weight rule.
- Statistics against independent hand computations (not just `t_boost.metrics`): null
  deviances per objective, Gini with exposure, lift tie handling and row-order invariance, PSI
  with empty bins.
- `from_bytes(to_bytes())`: fit time and fit evidence restored; an older-format file reports
  "not stored" and no false INERT rows.
- `unknown_category="error"` with unseen values: report produced, critical flag.
- `prune=False` fit: page under the size budget.
- Thresholds: override, unknown key, invalid values, `None` disables.
- `summary_metrics` key rules, finiteness, collisions.
- `report_html(json.loads(json.dumps(report)))` equals `report_html(report)`; unknown version
  raises.
- No `<script>`, no external URL; hostile feature names and levels (`<`, quotes, `</style>`)
  escaped; ids independent of labels.
- Native: multiclass `cell_indices` reproduces each class logit from its tables.

## Docs

`docs/_snippets/methods/html_report.md` (same layout as `pricing_report.md`); reference pages
for both classes; mkdocs nav and index listings; `pricing_report.md` for v2 and the new
arguments; `actual_vs_expected.md` for offset and multiclass; attributes page for `fitted_at_`
and `fit_duration_`; `metrics.md` for the two new deviances; a "Model report" page under Model
analysis; `report_html` in the Python reference. Wording: "early stopping".

## Gates

Release-mode core tests touched by the native change; `cargo +1.96.1` fmt/clippy with
`-D warnings` (default members and `-p t-boost-py`); pytest (one session); mypy `--strict`;
stubtest; `scripts/check_docs.py check`; strict mkdocs build. A sample page published as a
private artifact for a visual check before the PR.

## Order of work

1. Pre-existing bugs 1–5, with tests.
2. Native multiclass `cell_indices` + `raw_feature_names` + stubs + test.
3. `fitted_at_`/`fit_duration_`, persisted fit evidence, envelope bump.
4. `metrics.py` deviances; `_report.py` (statistics, thresholds, flags); `pricing_report` v2
   including multiclass.
5. `_html_report.py`; `report_html`, `html_report`.
6. Tests, docs, gates, sample artifact.

## Review summary (2026-10-13)

Both reviews independently found: the offset gap; deviance/null-deviance conventions;
exposure Gini and equal-mass lift needing their own contracts; A/E arrays shorter than the
grid; banded-axis label mismatch; PSI empty-bin and bin-count issues; loaded-model evidence
availability (`pruning_report_` is not persisted; tree counts are); `unknown_category="error"`
crashing the unseen check; JSON safety of `get_params()`; PNG heatmaps contradicting hover; size
not bounded by `<details>`.

Codex only: `required_columns` raising for numpy-fitted models; relabelling the v1 dataset as
`train`; MLflow metric key length/collision rules; logit effects are odds multipliers.
Fable only: multiclass also lacks `raw_feature_names` and the weight flag; relative
train/eval gap explodes near 0; new attributes must join the clear-lists; `_deployed_table_count`;
sign-off additions (SE bands, 2-D A/E, predicted distribution, relativity range).

Not adopted: both suggested cutting multiclass from the first release — kept, by user decision.
Fable suggested one function name instead of `html_report` + `report_html` — kept both
(method for convenience, function to render a stored report). Deferred: comparison against an
incumbent model (rustystats' `base_predictions`), split/date provenance — see open questions.

## Open questions

1. Incumbent comparison: accept `baseline=` predictions (column name or array) for a double-lift
   chart and side-by-side metrics, like rustystats' `base_predictions`? Useful for sign-off,
   adds scope.
2. Persisting `pruning_report_`/`evals_result_` grows the saved model (the pruning path and
   table scores are small; `evals_result_` is n_bags × rounds floats). OK?
