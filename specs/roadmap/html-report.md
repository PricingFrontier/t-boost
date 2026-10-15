# HTML model report — plan

Status: implemented 2026-10-13 on branch `html-report` (see "As built" at the end).
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

### Persisted fit evidence (decided 2026-10-13)

Haute loads models back from MLflow, so the pruning evidence must survive `to_bytes`: save
`pruning_report_` and `graduation_validation_` in the envelope (tree counts and stopping
reasons already are, sklearn.py:1439-1444). **`evals_result_` is not persisted** (user: "we
don't need previous training history"); the Training section shows the curve only on a live
fit and says so on a loaded one. Older files: the pruning section says "not stored with this
model", and `binding_report_` keeps its `UNAVAILABLE` status for them.

### Pruning transparency (user: "any transparency on the model fit is good")

The ranked-path selector (default) scores nested banks on out-of-bag rows: the main effects,
then interaction tables added in order of variance share (heredity-respecting), at a
geometric grid of bank sizes. With `devs` the OOB deviance per size, total gain =
`devs[0] − min(devs)`; the cutoff is the first size that reaches `prune_path_fraction` of the
total gain **and** is within `prune_path_tolerance` of the minimum (sklearn.py:4530-4541).

- At fit, add to `pruning_report_["path"]`: `sequence` (the ranked tables in the order they
  enter, with the size step each joins), `total_gain`, `cutoff` (the chosen step) and the two
  candidate steps it is the max of (`fraction_step`, `tolerance_step`, `min_step`).
- Page section "Pruning":
  - path chart: OOB deviance against bank size, with the minimum, the fraction-of-gain level,
    the tolerance level and the cutoff marked;
  - cumulative gain chart: share of the total gain captured against bank size, cutoff marked;
  - ranked table list: rank, table, order, variance share, cumulative share, the step it
    enters, kept / dropped (and kept but not deployed, from `kept_not_deployed`), with the
    cutoff drawn as a rule between the last kept and first dropped row;
  - one sentence in words, e.g. "kept 9 of 31 candidate tables, capturing 99.5% of the OOB
    deviance gain; the next step would have improved OOB deviance by 0.02%".
  - then the stages after selection: table/box budgets, guard, banding (cells before/after
    per table), graduation.
- Fold-vote selector (`prune_selector="fold_vote"`): its own evidence (stability, held-out
  gain per table, guard) in the same list layout.
- `prune=False`: "no pruning; full bank deployed" with the bank size.

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
   separately), Sobol share per table; then the **Pruning** section described under
   "Pruning transparency". Joint-measure caveats
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
- `from_bytes(to_bytes())`: fit time, pruning report and graduation restored, `evals_result_`
  absent; an older-format file reports
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
3. `fitted_at_`/`fit_duration_`, persisted pruning report + graduation, path `sequence`/
   `total_gain`/`cutoff` fields, envelope bump.
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

## Decisions (2026-10-13)

1. No comparison with an incumbent model yet (`baseline=` deferred).
2. Persist `pruning_report_` and `graduation_validation_`; do not persist `evals_result_`.

## As built (2026-10-13)

- `pricing_report` v2 (`python/t_boost/_report.py`), `report_html` (`_html_report.py`),
  `html_report` on both estimators, `fitted_at_`/`fit_duration_`, persisted `pruning_report_` and
  `graduation_validation_` (envelope schema 6), ranked-path `total_gain`/`min_step`/
  `fraction_step`/`tolerance_step`/`sequence`, multiclass `cell_indices`/`raw_feature_names`/
  `merged_axes` and one-vs-rest `actual_vs_expected`, `metrics.mean_binomial_deviance` and
  `mean_multinomial_deviance`.
- Performance (user, mid-implementation: "make any calculation in the rust side"): the per-row
  work is native. New `t_boost_core::report` (`cell_sums`, parallel over features; `ranking`:
  Gini, perfect Gini, equal-mass lift, Lorenz from one sort) exposed as `_report_cell_sums` /
  `_report_ranking`; `cell_indices` (both models) and `_TableModel.tables` take `cat_codes`, so
  the report and `tables()` use the coded categorical path `predict` uses. 1M rows × 15
  features: `pricing_report` 12.0 s → 0.76 s (the label path's per-value Python was 7 s).
- Factor A/E flag gained a credibility condition, `factor_z` (default 3): with only
  `factor_ae`/`factor_min_mass`, ordinary Poisson noise on ~150-claim levels flagged nearly every
  model. z uses the objective's variance function and the Pearson dispersion.
- Rendering: no JavaScript at all; SVG `<title>` hovers; CSS-only dataset/class selectors
  (verified with JS disabled in headless Chromium); heatmaps are SVG `<rect>`s coloured with
  `color-mix()` so one markup serves both themes. Checked at 1280 px and 390 px: no horizontal
  scroll.
- Datasets (user, 2026-10-13: "train / validation / test terminology … accept train / validation,
  or train / validation / test"): `eval_set` takes one pair or a list of up to two
  (validation, test), `eval_*` vectors as matching lists; labels default
  `("train", "validation", "test")`; thresholds `ae_train`/`ae_validation`/`ae_test`/`train_gap`
  (replacing `ae_eval`/`train_eval_gap`); PSI from train to each later dataset; an info flag when
  the fit early-stopped on an `eval_set`. Also fixed `binding_report_` flagging the DEFAULT
  `validation_fraction` as INERT on every eval_set fit (it made `check_bindings` raise).
- `prune_min_variance_share` (user, 2026-10-13: the ranked path kept tables with ~0.001% variance
  share; "users should be able to set a min variance share"): after the path cutoff, drop kept
  candidates below the share of the candidates' total variance, plus their supersets; mains exempt
  unless `prune_main_effects`; default 0 (off). Core `prune::drop_low_share` (multiclass, native)
  mirrored in `_ranked_path_select`; recorded as `path.min_variance_share`/`dropped_low_share`.
  freMTPL2 (60/20/20): 0 → 72 tables, test dev 0.304806; 1e-4 → 52 (+0.04%); 1e-3 → 28 (+0.18%);
  5e-3 → 19 (+0.30%).
- Tests: `python/tests/conftest.py` clears the cached constructor defaults around every test (ten
  modules patch `__kwdefaults__`, which leaked into `binding_report_`).
  `test_callback_true_stops_and_exceptions_propagate` is a pre-existing flake (~1 in 6 on
  origin/main too), not touched.
- Floor rule changed the same day: the first rule (drop a low table and everything containing it)
  dropped VehAge x VehGas (1.29%) once mains were prunable (VehGas main 0.085%). Now the kept set
  is the downward closure of the tables clearing the floor: a low table stays when a kept table at
  or above the floor is built on it. freMTPL2, 0.1% floor + `prune_main_effects=True`: 31 tables,
  test deviance 0.305104 (+0.10% vs no floor; old rule 24 tables, 0.305837); only Area's main
  effect dropped. Default stays 0 (user: "default off").
- Model-card review (Fable 5.1 + Codex gpt-6-astra ultra, 2026-10-13) applied: page order
  Review → Fit quality → Structure → Rating factors → Data checks → Pruning → Training → Appendix
  → Tables; header with base rate, table counts and a dataset × (deviance explained, Gini, A/E)
  grid; charts default to the last dataset; pruning/training/defaults/secondary metrics collapsed;
  one-line data checks; factor summary table; Structure rows link to Tables; categorical features
  as points; relativity axes span ≥ ±5%; no axis below zero; ranked tables grouped and collapsible.
  The factor A/E check now reads ≤ 10 equal-mass bands per feature (train bins, shared by every
  dataset; `bands` in the report) — per-cell checking covered only 6.4% of Density's exposure.
- Team review round 2 (six agents incl. Fable 5.1 and Codex Astra, 2026-10-13; user: "everything
  they need to make judgement on the model, without being overwhelmed, without scrolling up and
  down, and visuals they can use to summarise in meetings") implemented the same day. The page is
  now CSS-only tabs (radio inputs first in `.app`, labels in a sticky bar) with one page-wide
  dataset switch (default: the last dataset) and, for multiclass, one class switch:
  - **Overview** fits 1440 × 900 without scrolling: headline table (datasets × exposure, claims,
    A/E, deviance explained, Gini), the open review flags grouped by check (one line each,
    ellipsised, full text on hover), decile calibration (actual/predicted on top, A/E of every
    dataset below with the selected one's ±2σ), feature importance (own one-way table vs through
    interactions), the four largest one-way tables as small multiples on one log scale (thin
    cells faded), and the pruning gain path (Lorenz curve when no path was recorded). Titles are
    takeaways computed from the data ("Top decile 8.2× the bottom; test decile A/E 0.94–1.08").
  - **Performance**: full metrics table, full-width calibration (decile table collapsed), Lorenz
    beside an all-features A/E-by-band chart, spread of predictions; class matrices.
  - **Factors**: a rail of features (importance bar, worst band on the selected dataset) selects
    one figure per feature on one x-axis: relativity (log scale, thin cells faded), A/E by the
    review's bands for every dataset (±`factor_ae` dashed, ±2σ shaded) and train exposure (a
    dominant bar clipped and labelled); categorical levels ordered by effect; the feature's
    findings above; per-level numbers collapsed.
  - **Tables**: list (share, range, smoothing) selects a table shown as numbers beside it.
  - **Build**: gain path; every candidate as a column of its variance share in path order (kept /
    below the floor / past the cutoff, floor line); ranked groups; graduation strengths; training
    curve with the band where the bags stopped.
  - **Checks**: all flags, passed checks and thresholds collapsed; data; parameters; model; how to
    read (collapsed).
  - **Slides**: five charts as standalone light-palette SVG images (base64 data URIs, drawn at 2×,
    takeaway title and subtitle inside the image) to copy into a deck.
  Alignment: every chart is drawn at the inner pixel width of its card (558 half, 1166 full, 910
  factor) so 12 px text renders at 12 px and stacked panels share their left/right edges; titles
  and captions are one line; numeric columns right-aligned with one format per column; integer
  features labelled as integers (`51`, `22–25`, `≥ 189`, shared with the report's band labels via
  `run_label`); log ticks chosen to span the axis. Report additions: decile `ae`/`sd`/`z`, factor
  band `sd`, `features` section, `fit.graduation_tables`, `graduation_report_` persisted, review
  checks `decile_ae` (+ near misses), `missing_level`, `thin_cells`, `graduation`, thresholds
  `decile_ae`, `near_miss_z`, `missing_effect`, `thin_cell_mass`. The multiclass pruning report
  keeps its path under `selection` (no `sequence`): drawn kept-first.
- Round 3 (user, 2026-10-13: "as well as main effects for the factors, can we get a view of PDP or
  Shap to take into account other factors. And then also some more info on interactions … light
  pastel green (high) yellow low … I dont think we need the slides view"):
  - **SHAP, not PDP.** t-boost's interaction tables are centred on the train exposure marginals
    (freMTPL2 DrivAge × BonusMalus: |mean over the partner| ≤ 0.0005 on a table spanning −0.98 to
    +0.75), so a PDP equals the one-way table. The report now stores, on the first dataset's A/E
    entries, each merged cell's exposure-weighted mean and 10th/90th percentile of the feature's
    exact Shapley value (`shap`), from two native kernels: `TableModel::feature_contributions`
    (each effect split equally over its raw features; `_TableModel.feature_contributions`, per
    class on the multiclass model) and `report::cell_quantiles` (`_report_cell_quantiles`,
    weighted lower quantiles per cell, parallel over features). +0.11 s on 406k rows. The factor
    chart draws the SHAP mean (dashed) and the 80% band over the one-way table.
  - **Interactions tab** (replacing Slides): `interactions` section in the report (share, range,
    support-weighted 5th–95th percentile `spread`, smoothing); a rail by share and, per table,
    the first feature's one-way table (dashed) plus the table averaged over ≤ 4 bands of equal
    exposure of the partner (third feature averaged out), thin bands faded. The 40 largest are
    drawn (page size; all are in Tables).
  - **Tables tints**: five-step light green (above no effect) / yellow (below) per cell on the
    square root of |log effect| / the table's largest; grey for no exposure.
  - Slides tab, slide images and the takeaway variants for them removed.
- Train and test by default (user, 2026-10-14: "should we just have train and test?"). t-boost's
  bags early-stop on rows each holds back (`validation_fraction`) and prune on their out-of-bag
  rows, so validation data is opt-in: early-stopping on chosen data (the latest period, say) or
  comparing settings. On freMTPL2, fitting on 80% with the bags' own early stopping beat 60% plus
  a 20% `eval_set` by 0.12% of test deviance (8.30% vs 8.19% deviance explained; two seeds). So
  one eval pair is `test` (labels default by count: train/test, or train/validation/test), the
  second of two datasets is held to `ae_test`, the "no test data" note is gone, and a fit that
  early-stopped on an `eval_set` gets an info note naming the dataset that may be that data
  (info, not warning: stopping on a later period and reporting a separate test set is sound).
- Declined graduation is no longer silent (same day). The native check keeps smoothed tables
  only if the training deviance rises by at most 1%; the docs had said there was no accuracy
  gate. A fixed `graduation_alpha` it declines now warns at fit, is INERT in `binding_report_`
  (so `check_bindings()` raises and the report shows a warning), `graduation_validation_` keeps
  the reason under `rejected`, and the report notes a declined GCV graduation.
- Fit-time `eval_set` removed (2026-10-15, `post-fit-stages.md` PF-18): the report's `eval_set`
  stays (it is how test data arrives), the "early-stopped on an eval_set" note and
  `fit.early_stopping.eval_set` are gone, every live single-output fit has a training curve, and
  the Training chart draws one line per bag with the round it kept (per-bag holdouts sit at
  different levels, so a mean over the running bags jumped as they finished).
