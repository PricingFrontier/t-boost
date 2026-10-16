# Monotone constraints without switching the pipeline off — plan

Status: implemented 2026-10-14 on branch `report-0.9.1` (revision 2 of the plan, which folds in a
read-only review: Codex gpt-6-astra, ultra effort; its findings and what was done with each are at
the end). See "As built".

## Goal

A fit with `monotone_constraints` runs every post-fit stage that an unconstrained fit runs
(selection, budgets, rebalance, banding, graduation), and the deployed rating tables are still
monotone in every constrained feature, for every combination of the other features. This
follows the rule in `CLAUDE.md`: no setting switches a stage off; a stage that could break a
constraint is changed to respect it.

## Today

- `_fit_and_prune_core` (python/t_boost/sklearn.py:4818-4831) returns the full table bank as
  soon as any sign is non-zero: no selection, banding, guard or rebalance.
  `pruning_report_["constraints"]` says so.
- `_fit_and_prune` (sklearn.py:4506) skips graduation for monotone fits
  (`graduation_validation_["skipped"] = "monotone_constraints"`).
- Monotone constraints are refused in combination with:
  - `prune_main_effects` (sklearn.py:4395);
  - table/box budgets (sklearn.py:4820);
  - `graduation_high_order_alpha` (sklearn.py:4387);
  - `cell_refit_base` (sklearn.py:4393, and in the core at engine/boost.rs:69);
  - multiclass fits (sklearn.py:9574).
- `binding_report_` (sklearn.py:2966) lists every `prune_*` knob as unread "because the full bank
  is kept".

The bypass came in on 2026-09-21 (old-repo commit 3487ebc). Dropping a purified interaction
table can break monotonicity, and the fix was to switch the whole pipeline off.

## Why monotonicity can break

The trees enforce the constraint (split vetoes plus `clamp_monotone`, engine/split.rs:755), so
the ensemble is monotone, and so is the full purified bank, which equals it exactly.

Write `F_j` for the tables that contain the constrained feature `j`: its main effect `M_j`, and
its interactions `I_j`. Only `F_j` affects the steps along `j`. A pure interaction cannot be
monotone on its own (zero mean along every axis), so `M_j + Σ I_j` is monotone only as a sum.
Any stage that changes `F_j` (dropping a table, or rewriting values) can therefore break it.

Every supported reference measure is a positive product measure (`Joint` is rejected,
explain.rs:706). Under one, a pure interaction averages to zero over its partners, so `M_j` is
exactly the partner-average of the family. A monotone family therefore has a monotone main
effect. Conversely, a non-monotone `M_j` proves the family is infeasible.

## Evidence (2026-10-14)

Setup:
- freMTPL2freq: 80/20 train/test, Poisson with exposure, defaults.
- `BonusMalus` and `Density` constrained to be increasing.
- The bypass is removed by a subclass: the trees stay constrained, and every stage runs as if
  the fit were unconstrained. No repo code was changed for this.
- A reversal is the largest drop in prediction as the feature steps up, over 2,000 training rows
  times every distinct value (BonusMalus) or 300 quantiles (Density).

| Run | Tables | Test deviance | BonusMalus reversal | Density reversal |
|---|---|---|---|---|
| Unconstrained | 68 | 0.30444 | 216% | 37.8% |
| Today (full bank) | 113 | 0.30726 | 0 | 0 |
| Selection only | 51 | 0.30736 | 0.05% | 0.003% |
| + rebalance | 51 | 0.30686 | 14.9% | 3.6% |
| + rebalance + banding | 51 | 0.30692 | 158.6% | 3.7% |
| All defaults (+ graduation) | 51 | 0.30699 | 19.2% | 3.7% |
| No rebalance; banding + graduation | 51 | 0.30748 | 0 | 2.2% |
| No rebalance; banding, no graduation | 51 | — | 47% | 1.5% |

Breaking the reversals down by table (per-term contributions across the sweep):

- **Rebalance** makes the main effects themselves non-monotone. The Density main carries almost
  all of its reversal. It also enlarges pair swings (BonusMalus×VehBrand, worst step −0.14 on
  the log scale). It is an unconstrained ridge re-solve of main and pair cells
  (prune.rs:2331), so it relearns the shape the constraint held back.
- **Banding** leaves the main effects monotone (mains stay on the merged grid, banding.rs:1915)
  but puts large swings along the constrained axis into interaction tables: VehAge×BonusMalus
  −0.47 when graduation is off.
- **Graduation** keeps the main effects monotone. Its pair smoothing gives at most 1.6%
  (Density×Region).
- **Selection** on its own: below 0.05%.

The 0.9% between "Unconstrained" and "Today" is what the constraints cost on this data, not
damage from the pipeline.

Conclusion: the bypass was aimed at the wrong stage. Selection is nearly harmless; the stages
that rewrite table values are what break monotonicity.

This measures removing the bypass, not the repair. What the repair costs, and whether
interaction tables need editing at all (rather than only main effects), is measured in step 3
of the implementation order before that machinery is built.

## Design

### Stage order (single-output, `prune=True`)

1. Deploy fit.
2. Selection (ranked path, or the fold vote).
3. Keep-set assembly (re-anchor, rebalance).
4. Banding.
5. Graduation.
6. **New: the monotone repair**, only when a sign is non-zero.

The prune guards and the slope re-anchor are removed (see "Also on this branch"), so nothing
re-assembles the bank after banding.

There are no new constructor parameters. Unconstrained fits never call the repair. The other
changes on this branch do change some models deliberately; they are listed there.

### The monotone repair (native, new module `crates/t-boost-core/src/monotone.rs`)

**Inputs:**
- the deployed bank;
- the signs per **raw feature**, in native raw-feature order (numeric first, then categoricals,
  as `_fit_model` already remaps them), not per expanded channel axis;
- the training rows (binned), with a per-row curvature `h` from `_band_curvature`, as banding
  uses;
- the loss, weight and exposure for the re-anchor.

**Output:** the repaired bank, plus a JSON report.

**Steps and boundaries.** For a constrained (numeric, see "Refusals") feature `j` with sign `s`:
- The missing bin (cell 0) is exempt from the constraint; the trees' learned missing direction
  leaves it free.
- The constrained cells are the finite merged cells, in order. Boundary `b` sits between two
  adjacent ones.
- For an interaction `T` and a configuration `x` of its partners (every merged cell, including
  partners' missing cells and every categorical level), the step is
  `D_T(b, x) = s·(t_T(b⁺, x) − t_T(b⁻, x))`, read through each axis's band map.
- The family is monotone exactly when, for every `b`,
  `s·ΔM_j(b) + min_x Σ_T D_T(b, x) ≥ 0`.

**Certificate: the interaction envelope `E_j(b) = min_x Σ_T D_T(b, x)`.**
- Group `I_j` into components that share partners. Pairs with distinct partners are separate
  components; a triple `(j,k,l)` joins `(j,k)` and `(j,l)`. Pairs-only families are always
  exact.
- Enumerate a component exactly when its work is under a fixed cap (an internal constant). Work
  is counted honestly: cells × boundaries × the cost of each table lookup, so a factored effect
  counts its boxes (explain.rs:3003), and memory is bounded separately.
- Over the cap, use the sound bound `Σ_T min_x D_T(b, x)`. For a factored effect, each box
  contributes the minimum over its realizable corners. A bound never overstates the minimum,
  but it can call a feasible family infeasible. With pure steps `k`, `l` and `k·l` over
  `{−1, 0, 1}`, the exact minimum is −1 and the summed minima are −3.
- Factored effects appear at default settings whenever banding is skipped: single-bag fits fail
  `bag_oob_available()` (sklearn.py:5035), and orders 4–8 are factored. Fixing the audit's
  `n_bags=1` item (below) would shrink how often this path is reached.

**A certified bank is returned untouched**, byte-identical: no re-centring and no re-anchoring.
That promise holds when the certificate proves feasibility. It does not hold for a feasible
family that only a bound covers; the report says which certificate each feature got.

**Repair, in global phases.** Every phase finishes for all constrained features before the next
starts. A per-feature schedule is unsound: clipping a shared interaction for feature `k` can
undo feature `j`'s earlier polish.

1. **Mains.** Weighted isotonic regression (PAVA) of each constrained `s·M_j`. The weights are
   curvature mass per cell plus a product-measure floor, as banding builds its floor
   (banding.rs:2562), so every weight is positive. This step is necessary (see "Why
   monotonicity can break"), and it does nothing to a main effect that is already monotone.
2. **Interactions.** All interaction edits for all features happen here. Which edit, if any, is
   decided by the benchmark below.
3. **Re-purify once.** Re-purify the sub-bank of modified tables plus their down-closure (all
   dense) under the fit's reference measure, with `banding::repurify_bank`. Moving a marginal
   into a lower-order table never changes any family's steps along its own feature, including
   for tables shared by two constrained features.
4. **Polish every constrained main.** Recompute each envelope from the final interactions, then
   project `M_j` onto `{s·ΔM_j(b) ≥ −E_j(b)}` by shifted PAVA: make `u(b) = s·M_j(b) + Σ_{b'<b}
   E_j(b')` isotonic, then subtract the shift back out. That is the exact weighted L2 projection
   onto the set. Each `M_j` belongs to one family only, so this phase cannot break another
   feature. After it, every family satisfies its certificate.
5. **Maintain invariants.**
   - Re-centre each changed main effect: its mean under the reference measure moves into `f0`.
   - Recompute the cached `variance` of every modified table after this final centring.
   - Re-anchor the intercept by the `apply_keepset` rule (log and logit links, when the fit
     re-anchors).
   - `se_band` (display only) is kept and documented as the bag spread of the unrepaired
     estimate.
6. **Verify.** Every constrained feature must pass its certificate; a failure is an internal
   error, never a silent ship. `TableModel::validate` only checks structure (table_model.rs:94),
   so the tests also run the I2 gates on repaired banks: purity, mass conservation, variance sum
   and reconstruction.

The edits leave every table exact against its own lookup sum, so there is no exactness-firewall
downgrade and no schema bump.

**Interaction step: three candidates, measured before one is chosen.**
- **(i) None (main-only).** Phase 2 is empty. Phase 4 absorbs every interaction reversal by
  steepening the main effect.
- **(ii) Scaling.** Shrink each offending interaction by a single factor `α_T ∈ [0,1]`, the
  largest that satisfies its share of the envelope. A scaled pure table stays pure, so there is
  no re-purification and no new band edge.
- **(iii) Clipping.** Project every slice of an offending table by shifted PAVA, against a floor
  shared out in proportion to need.
  - Weights are positive: curvature plus the floor, per cell.
  - There is no floor at boundaries that already pass.
  - Factored need is taken from the budget first; if it exceeds the budget, clippable tables
    get a zero floor and phase 4 covers the rest.
  - Tables with two or more constrained features alternate their projections (Dykstra) for a
    bounded number of rounds; phase 4 absorbs any residue.

The benchmark compares repaired outputs on freMTPL2 and at least two arena datasets with natural
monotone factors, with banding and graduation each on and off. It records held-out deviance,
the largest change to any main effect (the visible distortion in a rating table), the tables
touched, and runtime. The simplest candidate within noise of the best wins. Clipping is built
only if it earns its place.

**Benchmark (2026-10-14).** A Python prototype applied each candidate to the deployed bank's
JSON. It skipped re-purification, which does not change predictions, and re-anchored the
intercept on the training rows. Each dataset was fitted once (80/20 split), with the bypass
removed by a subclass. Every candidate removed every reversal; they differ in cost.

freMTPL2 (Poisson with exposure; BonusMalus and Density increasing):

| Pipeline | Before repair (not monotone) | (i) main-only | (ii) scaling | (iii) clipping |
|---|---|---|---|---|
| All defaults | 0.30699 | +0.33%, BonusMalus main moves 28% | +0.32%, α = 0 (interactions zeroed) | −0.004%, mains move ≤ 4% |
| Graduation off | 0.30692 | **+24.8%**, BonusMalus main moves 972% | +0.32%, α = 0 | +0.02%, mains move ≤ 9% |
| Rebalance off | 0.30748 | −0.001% | +0.03% | −0.000% |

- **beMTPL97** (Poisson; `bm` and `power` increasing): the ranked path keeps no interaction
  with either feature. All three candidates coincide, and a main effect moves by at most 0.016%.
- **Swedish motorcycle** (Tweedie pure premium; `mcklass` increasing, `fordald` and `bonuskl`
  decreasing): `bonuskl` is unused and `mcklass`'s one interaction is already monotone, so every
  candidate is a no-op.

Conclusions:
- **Clipping (iii) is built.** Correcting only the main effect can be catastrophic when banding
  has put large swings into interactions. Scaling zeroes a constrained feature's interactions
  wherever its main effect is flat.
- **Rebalance stays on.** With the repair, all defaults reach 0.30698, against 0.30748 with
  rebalance off and 0.30726 for today's full bank, with 51 tables instead of 113.
- **Not yet exercised:**
  - a table holding two constrained features (no dataset produced one);
  - re-purification (skipped by the prototype);
  - factored three-way tables (banding was on throughout).

  The native implementation is tested on each of these first.

### Rebalance

Rebalance runs on everything, and the repair projects its output.

Excluding constrained supports instead (the earlier variant (a)) was dropped. When every
feature is constrained, it leaves no supports, and `rebalance_kept_cells` returns immediately
(prune.rs:2405): a whole-stage switch-off. The benchmark compares repaired outputs with
rebalance on and off, because rebalance's own acceptance check scores the unrepaired candidate
(prune.rs:2478).

The whole-model rebalance skip for multi-channel categoricals (prune.rs:2372) is an audit item
and is not changed here.

### Refusals

- **Lifted:** `prune_main_effects`, table/box budgets, and `graduation_high_order_alpha`
  combined with monotone constraints. The repair covers whatever they produce.
- **Kept:**
  - `cell_refit_base` (a fit-time ensemble correction; the core refuses it for the Rust API,
    BUG-048).
  - Multiclass (monotone logits do not make every softmax probability monotone).
- **New: monotone constraints are numeric-only** (maintainer decision, 2026-10-14). A sign on a
  categorical feature raises at fit time, naming the feature.
  - The docs always said "numerical features" (common.md:446).
  - Signs on categoricals got through by accident. The June 2026 change (old-repo 21e7957) let
    numeric constraints coexist with categoricals by removing two guards, and those guards had
    also refused a sign placed on a categorical itself.
  - Today such a sign constrains the model along the category's target-statistic encoding (and
    every channel with `cat_channels`, lib.rs:678): an order learned from the data, not the
    labels' order.
  - No Python test puts a sign on a categorical. `test_sklearn.py:681` and `:754` constrain a
    numeric feature next to a categorical, and keep passing.
  - Categorical *partners* of a constrained feature are unaffected.

### Reporting

- `pruning_report_["monotone"]`:
  - per feature: sign, family supports, the certificate used (exact or bound), the worst step
    before the repair, the largest main-effect change, the interaction tables changed and
    their largest change, and the polish change;
  - certified no-ops recorded as such;
  - the training-deviance change caused by the repair.
- `pruning_report_["kept"]` is reconciled with the deployed bank after the repair. Today it is
  only reconciled inside the guard (sklearn.py:5499), which is removed.
- `pruning_report_["constraints"]` and `graduation_validation_["skipped"] = "monotone_constraints"`
  go away.
- `binding_report_` stops listing `prune_*` knobs as unread for monotone fits. The monotone row
  points at the repair report.
- The HTML report's Build tab shows one line per constrained feature, outside the
  ranked-path-only "After selection" list (_html_report.py:2651).

### API, docs, tests

- **Binding:** `_TableModel.enforce_monotone(x, h, monotone, y, weight, exposure=None,
  cat_x=None, reanchor=True, n_jobs=None) -> (_TableModel, str)`, plus its stub entry. Python
  calls it at the end of `_fit_and_prune`.
- **Docs:**
  - `docs/training-parameters/common.md#monotone_constraints` (the "keeps its full set of
    tables" note; a sign on a categorical raises);
  - `graduation.md:9,68`, `bagging.md:60`, `pruning.md:59`;
  - `algorithm/readable-tables.md:73`;
  - both class docstrings and the `_fit_and_prune` docstring.
- **Rust tests** (`monotone.rs`):
  - shifted PAVA equals a brute-force QP on small random cases, including zero-mass cells, and
    is idempotent on feasible input;
  - the exact envelope equals a brute force over every partner configuration, including
    missing and categorical partners; the bound is ≤ exact, and the cap fallback path is
    exercised;
  - repairing banks built from monotone trees with interactions dropped or perturbed gives a
    monotone bank under a full-grid brute-force check, for:
    - both signs and mixed signs;
    - two and three constrained features sharing tables;
    - every product measure (`ProductMarginals`, `ExposureMarginals`, `Uniform`);
    - banded axes (no new band edges);
    - factored and sparse storage;
  - the I2 gates pass on every repaired bank;
  - a certified bank is returned byte-identical;
  - the output is byte-identical across thread counts and processes.
- **Python tests:**
  - a default pipeline on synthetic data with monotone numeric features (both signs) next to a
    categorical partner: pruning ran (kept < full), banding and graduation are reported, the
    brute-force grid check holds, and the repair report is present;
  - a single-bag default fit, which keeps factored effects;
  - `prune_main_effects`, budgets and higher-order smoothing combined with monotone;
  - `prune=False` is unaffected (it deploys the full bank, monotone by construction);
  - a `to_bytes`/`from_bytes` round trip; `tables(X)` re-centred exports, `predict_contributions`
    and the SQL export agree with `predict_raw` on a repaired model;
  - a sign on a categorical raises, for a positional list and a name-keyed dict;
  - updated refusal tests (`test_prune_main_effects.py:225`,
    `test_surface_merges_and_binding_report.py:345`).
- **Performance:** measured, not assumed, on freMTPL2, a single-bag default fit and an order-6
  fit. The freMTPL2 run is repeated manually (its data is not in the repo).

## Also on this branch: from the audit

Each bug fix gets its own regression test.

1. **Remove the prune guards** (maintainer decision, 2026-10-14). The audit found that a guard
   that fires re-assembles the bank through `_assemble_deploy` (sklearn.py:5278, 5461), which
   does not band, while `pruning_report_["banding"]` still describes the discarded banding.
   Rather than fix that inside the guard, the guards go.
   - **History.** The set-level guard came in on 2026-08-05 after two specific fold-vote failures
     (allstate_sev, homesite_conv), and grew a parameter or mechanism with most August
     iterations. On 2026-09-25 the ranked path replaced the fold vote as the default "with no
     guard" (old-repo 70d7ad2): measured on 35 arena datasets, it gave 27% fewer pairs, 72%
     fewer tables of order three and above, +0.016% test deviance, and 1.3–3x faster fits. The
     fold vote was kept for ablation and as the fallback for fits without out-of-bag rows.
   - **What runs it today.** Only an explicit `prune_selector="fold_vote"` with enough
     out-of-bag rows, or tiny panel data through the carve. A default fit never runs it.
   - **Removed:**
     - the single-output guard, and the slope re-anchor that only runs inside it;
     - the K≥3 guard (`prune::multiclass_prune_guard` and its arms);
     - parameters `prune_guard`, `prune_guard_tol`, `prune_guard_z`, `prune_guard_z_dn`,
       `prune_guard_tol_floor`, `prune_slope_eps`, `prune_slope_min_z`,
       `multiclass_prune_guard` and `multiclass_prune_guard_floor`;
     - their docs, report keys (`pruning_report_["guard"]`, `["slope"]`) and tests.
   - **Kept:** `prune_guard_min_rows`, which the ranked paths read as their minimum out-of-bag
     row count. Deviance helpers that the ranked path and banding share with the guard are
     renamed, not removed.
   - **Model change:** fold-vote fits no longer re-admit tables or rescale their slope. Default
     fits are unchanged.
2. **The binary classifier never warns.** `TBoostClassifier.fit` returns from the K=2 branch
   (sklearn.py:9466) before `_warn_voided_params()`, which only sklearn.py:7797 and 9478 call.
   Fix: call it before that return.
3. **Callbacks drop the group-aware holdout.** With callbacks and `prune=False` on panel data,
   `deploy` holds the observer, so `not deploy` (sklearn.py:4421) skips the group-aware
   early-stopping carve. Fix: test `"eval_x" not in deploy`, as the pruned path does (4752).
   This changes those fits' early stopping.
4. **Docstring errors:**
   - `leaf_refine_steps` "honoured for K≥3" (sklearn.py:3377);
   - `bag_oob_available` "needs no §G1 correction" (lib.rs:5607);
   - the "§G1 cell-corrected soup" example at sklearn.py:4660.

## Implementation order

1. Bugs 2–4 (small and independent), each with its test. Done: 52f4232.
2. Remove the guards (item 1). Done: 7e1f070.
3. A Python prototype of candidates (i)–(iii) and rebalance on/off, compared on repaired outputs
   (see "Benchmark"). Clipping (iii) won; rebalance stays on.
4. `monotone.rs` with clipping (iii), its binding, the Python wiring, reports, docs, tests and
   every gate in CONTRIBUTING.md.

## As built (2026-10-14)

- `crates/t-boost-core/src/monotone.rs`: `repair_bank` (the phases above) and
  `enforce_monotone` (repair plus the intercept re-anchor). Ten unit tests check the repair
  against brute force over every cell combination:
  - one, two and three constrained features, both signs, tables shared between features;
  - factored effects through both the exact and the bound path;
  - purity and stored variances of every repaired table;
  - PAVA against the min-max formula;
  - a certified bank returned unchanged, and determinism.
- Binding `_TableModel.enforce_monotone`. `_fit_and_prune` calls it last, after graduation,
  weighted by `_band_curvature`.
- `python/tests/test_monotone_pruning.py` covers:
  - the default pipeline;
  - a single-bag fit that keeps factored effects and needs a repair;
  - `prune_main_effects`, both budgets, higher-order smoothing, and rebalance and graduation off;
  - `prune=False`;
  - the categorical refusal;
  - a binary classifier;
  - serialization and exports;
  - the HTML report's Build-tab card.
- On freMTPL2 the real estimator reproduces the prototype: test deviance 0.306977 with all
  defaults and 0.306993 with graduation off, no reversals, 51 tables instead of 113 (today's
  bypass: 0.307258).

## Not in this plan: other stages that are switched off (audit, 2026-10-14)

These come from a read-only audit. "Verified" means read in the code; the rest is the audit's
reading. They are pending the maintainer's choice.

- **Slope re-anchor (verified).** It ran only inside the fold-vote guard's out-of-bag branch, so
  never on the default ranked path, and was skipped whenever exposure or an offset was given.
  It is removed with the guard (item 1 above). A slope correction on the ranked path would be a
  new proposal, to be measured first.
- **Rebalance skips silently (verified).** It skips the whole model when there is a cell
  correction or any multi-channel categorical (prune.rs:2346, 2372), and nothing records the
  skip. Idea: map raw ids to axes through `provenance`, and add a `rebalance` report key.
- **Graduation is all or nothing** (lib.rs:10564, 1% training-deviance check). Idea: back off
  the worst tables instead of declining all of them.
- **`n_bags=1` or fewer than 500 out-of-bag rows.** Selection falls back to the fold vote, and
  with `n_bags=1` banding is skipped too. For K≥3 the full bank is kept.
- **`prune=False` gets no banding or graduation.**
- Smaller items: the squared-error re-anchor; multiclass gaps; banding all or nothing; reasons
  missing from reports.

## Decisions and open questions

Decided by the maintainer, 2026-10-14:
1. `se_band` is kept on repaired tables, documented as the unrepaired estimate's bag spread.
2. Monotone constraints are numeric-only; a sign on a categorical raises (see "Refusals").

3. The prune guards (single-output and K≥3) and the slope re-anchor are removed rather than
   fixed (item 1 under "Also on this branch").

## Review (Codex gpt-6-astra, ultra effort, 2026-10-14)

Overall: sound direction, not sound to implement as written. The central mathematics was
confirmed:
- every supported measure is a positive product;
- sub-bank purification preserves every directional step, including shared constrained
  features;
- shifted PAVA is exact for fixed bounds and positive weights;
- no firewall downgrade or schema bump is needed;
- slope multipliers are positive;
- keeping the multiclass and `cell_refit_base` refusals is right.

| # | Finding | Disposition |
|---|---|---|
| 1 | Blocker: the per-feature schedule can invalidate an earlier feature's polish through a shared table. | Accepted: global phases, with the polish of every main last. |
| 2 | Cached variances go stale after the polish; `validate` does not check I2. | Accepted: recompute variances after final centring; I2 gates in tests. |
| 3 | Interaction PAVA has no zero-weight or boundary policy. | Accepted: positive floor weights everywhere; constraint set fully specified. |
| 4 | Rebalance variant (a) can filter every support away, which is a whole-stage switch-off. | Accepted: (a) dropped, rebalance always runs; compared on repaired outputs. |
| 5 | Factored effects occur at default settings (single-bag fits), and the work cap ignores box cost. | Accepted: honest work accounting; single-bag and order-6 tests. |
| 6 | A conservative bound contradicts "feasible banks are byte-identical". | Accepted: byte identity only for certified banks; early return before centring. |
| 7 | Missing-cell and categorical semantics are wrong. | Accepted, then simplified: constraints are numeric-only (maintainer decision), so only the missing bin is exempt and categorical order never arises; signs are given in raw-feature order. |
| 8 | The guard fix needs one common finalization path. | Superseded: the guards are removed (maintainer decision), so nothing re-assembles after banding. |
| 9 | Interaction clipping has not earned its complexity. | Accepted: main-only and scaling are measured first; clipping is built only if it wins. |
| 10 | Test coverage and report reconciliation are incomplete. | Accepted: tests listed above; `kept` reconciled after the repair; HTML line outside the ranked-path list. |

One point is only partly taken: the reviewer suggested invalidating `se_band` on modified
tables. It is kept and documented instead (decision 1).
