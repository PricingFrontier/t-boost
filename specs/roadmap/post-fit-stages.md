# Post-fit stages: open issues after the switch-off work

Status: open. Written 2026-10-15 on branch `report-0.9.1` at `85f40f1`. Nothing below has been
implemented; each issue says what is ready, what needs a decision and what needs more work.

## Where this comes from

On 2026-10-14 every setting that switched a post-fit stage off was closed (`7473e7a`, `f7db12f`,
`168a472`, `85f40f1`): unpruned fits are banded, graduated and repaired; one-bag fits are banded
with the deviance cap alone; the rebalance runs with cell corrections and on multiclass fits;
multiclass fits are graduated. That work left three open findings:

- **A.** Graduation's 1% training-deviance check looked too loose for classification.
- **B.** The rebalance's held-out check looked optimistic: pg17 instalment adopted a step-0.75
  correction and lost about 0.2% test cross-entropy.
- **C.** `band_deviance_cap` is described as a bound on banding's cost, but one-bag fits raise
  training deviance about 4× the cap. And `n_bags=1` with `band_deviance_cap=0` skipped
  banding.

A team of six agents (one investigator and one adversarial reviewer per finding) measured each
finding, built and benchmarked candidate fixes, and checked one another's evidence on fresh splits.
The causes first proposed for A and B were both wrong. This document records everything they
found, including problems nobody was looking for, and the candidates that were measured and
rejected, so they are not tried again.

### How it was measured

- **Datasets** (cached under `~/suite/hsv18-throwaway-cache/datasets`, freMTPL2 from
  `~/suite/t-boost-report-demo/data`):

  | name | task | rows | notes |
  |---|---|---|---|
  | fremtpl2 | Poisson, exposure | 678k | Region has 22 levels of about 25k rows |
  | norway | Poisson, exposure | 184k | 4 features |
  | swmc | Poisson, exposure | 62k | Swedish motorcycle |
  | ausbi | Gamma | 22k | |
  | ussev | Gamma | 6.8k | CLASS has 18 levels |
  | surrender, lifesurr, lapse | binary | 22k–29k | |
  | pg16_fleet | multiclass, 3 classes | 87k | |
  | pg17_instal | multiclass, 4 classes | 100k | vh_model 979 levels, pol_insee_code 16,303, vh_make 97 |
  | pg17_cover | multiclass, 4 classes | 100k | |
  | fr_instal | multiclass, 4 classes | 52k | rare class at 2.8% |
  | prudential | multiclass, 8 classes | 59k | 139 columns; also used as binary (y ≥ 6) |

- **Splits.** Test rows are `default_rng(split).uniform() >= 0.8`. Effects are paired: every
  variant shares the split, seed (0), build and, where stated, the same fitted soup.
- **Deviance.** Weighted mean unit deviance on the test rows (log loss for classification).
  Changes are given in % of that deviance; negative is better.
- **Noise.** On the small sets, the paired SE of a single split's effect is 0.02–0.35%. A single
  split rarely settles anything below 0.1% there; dataset-level means over 5–10 splits, and
  fresh splits not used to design a rule, carry the verdicts below.
- **Harness.** `repro/post-fit-stages/team_bench.py` loads every dataset, fits, and prints one
  JSON line of test/train deviance and post-fit reports per (dataset, split, config).
- **Patches.** Each investigator's patch is committed on a local branch, against `85f40f1`:

  | branch | contents | status |
  |---|---|---|
  | `team/a-gcv-ceiling-rule-rejected` | GCV interior-minimum rule, doc fixes | rejected, reference only |
  | `team/b-rebalance-leak-and-bag-check` | leak exclusion plus per-bag check (one commit) | split before landing |
  | `team/c-band-cap-wording` | cap rewording and validation | needs the reviewer's five edits |

  Port from them by hand (`git diff 85f40f1 <branch>`); don't merge them, because B's single
  commit bundles two changes and A's is rejected.
- **Primary source.** The six agents' final reports are in
  `repro/post-fit-stages/team_reports.txt`; their full transcripts are in the Claude Code session
  directory (`~/.claude/projects/-home-ralph-suite-t-boost/c55508b8-d4e7-4235-9479-9cc2164ee349/subagents/workflows/wf_f90f4e08-420/`,
  `journal.jsonl` and `agent-*.jsonl`). The agents' own scripts and raw JSONL results lived in a
  session scratchpad under `/tmp` that has since been cleared; only the harness was recovered.

## Summary

| ID | Issue | Recommendation | Cost | Decision |
|---|---|---|---|---|
| PF-01 | `band_deviance_cap` wording; 0, inf and nan switch banding off | Reword; raise unless finite and > 0 | none; byte-identical for valid caps | ready |
| PF-02 | Rebalance learns a target-encoding leak | Leave tables on cross-fit-encoded categoricals as fitted | none | ready |
| PF-03 | Rebalance check is biased against the re-solve | Per-bag out-of-bag check, correction fitted on every row | +6–37% fit time, +11–14% memory | maintainer |
| PF-04 | Graduation's 1% check | Keep it; say it is a brake, not an accuracy test | none | ready (docs) |
| PF-05 | The serve-map leak in selection, banding and graduation | Investigate | unknown | next |
| PF-06 | Multiclass banding too coarse on some sets | Measure more sets first | unknown | open |
| PF-07 | Multiclass graduation over-smoothing (fr_instal), held off only by the brake's scale | Investigate | unknown | open |
| PF-08 | GCV at the grid ceiling (Prudential lead) | A unit-free rule, validated on fresh splits | none expected | open |
| PF-09 | Graduation's re-anchor overrides `reanchor=False` | Honour it or report it as overridden | small | bug |
| PF-10 | `graduation_validation_` has no reason when nothing is smoothed | Set `skipped` and `candidate_count` on every path | small | bug |
| PF-11 | One-bag fits have no honest held-out evidence | Use the early-stopping carve; fold-model σ | medium | open |
| PF-12 | `band_tolerance` 0.375 beats 0.75 on small single-output sets | Measure on the arena and large sets | none | open |
| PF-13 | Banding can raise the cell count (densifies factored tables) | Investigate | unknown | open |
| PF-14 | `test_callback_true_stops_and_exceptions_propagate` is flaky under load | Fix the test | small | bug |
| PF-15 | Smaller defects and loose ends | See the list | small | mixed |

## PF-01. `band_deviance_cap` bounds an estimate, not the training rise

### Finding

`mse_cap = cap · D · Σw / Σh` bounds only the second-order term of banding's move,
`Σ h·δ² / Σw ≤ cap · D` (D the training deviance). The core holds that on its cross-fitted
estimate of the combined move (every table summed per row): `combined_mse ≤ budget` in every
measured fit. The docs and docstrings call the cap "the most banding may cost, as a fraction of
the training deviance", which is false.

Banding's exact training-deviance change splits into
`2·Σ g·δ / Σw + Σ h·δ² / Σw + remainder`, with g and h the gradient and curvature at the unbanded
tables. Over 39 fits (7 fast sets × 2 splits × 1 and 8 bags, fremtpl2, fr_instal, pg16_fleet):

- **The first-order term is the whole excess.** It is positive on the training rows in all 35 fits
  that moved anything, because the bagged, shrunk, early-stopped tables sit short of each cell's
  training optimum and banding moves them further away. The realized training rise is 0–5.9× the
  cap where the cap binds alone (one bag, single-output) and up to 6.6× on multiclass. The 12.3×
  maximum is swmc at 8 bags, where the noise budget bound, not the cap.
- **The second-order approximation holds.** The remainder is at most 0.025× the cap.
- **Interacting merges are not the cause.** The second-order term stays at most 0.87× the cap on
  training rows and 0.86× on test rows at the default cap.
- **On test rows the first-order term has either sign** (negative in 17 of 39 fits), from −0.240%
  to +0.405%. It decides whether banding helps (swmc and fremtpl2 at one bag) or hurts (ausbi,
  fr_instal, pg16_fleet).
- **At tight caps the estimate can be exceeded on new rows.** The test-row second-order term was
  1.01–1.18× the cap in force in 5 tighter-cap runs. On `test_single_bag.py`'s fixture with 200k
  fresh rows it is 1.35× at cap 0.00025 (0.39× in sample), 0.49× at 0.001 and 0.74× at 0.004.
- **The training rows misjudge sparse tables.** Banding's distil pins them: on surrender the
  training-row move is 0.00–0.07× the budget while the test-row move is 0.73–1.07×.

The zero and non-finite values are also inconsistent switches:

- Single-output, 0 (or negative, or nan) meant "no cap". At 8 bags the noise budget acted alone;
  at one bag banding was skipped, with only
  `pruning_report_["banding"]["skipped"] = "one bag and band_deviance_cap=0: nothing bounds the banding"`.
- Multiclass, 0 gave every class a zero budget, so the core's "cannot meet its fidelity budget"
  skip left every class bank unbanded at any bag count. This second silent skip was found during
  the investigation.
- Multiclass, inf or nan reached the core as an infinite budget, which bands without bound at one
  bag.

### Recommendation (ready)

Keep the mechanism. Apply `team/c-band-cap-wording` with the reviewer's five edits:

1. Describe the cap as a budget on banding's **cross-fitted estimate** of the move's expected
   (second-order) cost, which new data can exceed at tight caps. Limit any "held every time"
   wording to the default cap, or drop it. This applies to banding.md, index.md (regenerate with
   `check_docs.py fix`), readable-tables.md, both estimator docstrings, the
   `_BAND_DEVIANCE_CAP_DEFAULT` and `_band_deployed` comments, and the test docstring.
2. Replace "Banding strips out noise that the tables fitted to the training rows" with a neutral
   statement: the training deviance rises by more than the cap because the cap leaves out the
   first-order term, and on new data that term can go either way. (On ausbi and every multiclass
   fit, banding removed signal.)
3. Replace "up to 12× the cap" with the range where the cap binds: 0–5.9× single-output at one
   bag, at most 6.6× multiclass.
4. The contract test asserts what the core guarantees: `combined_mse <= budget`, and
   `budget == deviance_budget` at one bag. If an expected-cost check is kept, compute it on fresh
   rows with an explicit tolerance.
5. Scope "flat test response, needs no tuning" and "dropping it costs +0.10%" to single-output
   fits, and point to PF-06 for multiclass.

The validator, `_validate_band_deviance_cap`, raises `ValueError` at the start of fit on both
`_fit_model` and `_fit_multiclass` for a cap ≤ 0, inf, nan or bool, naming the parameter and its
one-bag role. `_band_deployed` loses the "nothing bounds the banding" skip and the 0/non-finite →
inf fallback, and `test_single_bag.py::test_one_bag_and_no_deviance_cap_leave_nothing_to_bound_the_banding`
becomes a test of the `ValueError`. Python only.

- **Byte-identical** for every finite cap above 0. Checked on 8 fits by the investigator, and 8 of
  8 again by the reviewer on a fresh split against the HEAD build.
- **No speed cost.**
- **Breaking:** `band_deviance_cap=0` now raises. Haute, insur-arena2, insur-arena-av40-wt and
  t-boost-report-demo never pass it.

### Rejected (measured)

| candidate | result | why rejected |
|---|---|---|
| Include the first-order term in the cap | constrains the same training-row quantity as calibration | Charges banding for removing fitted noise; the term has either sign out of sample |
| Calibrate the budget to the realized training rise (up to 5 passes) | 1 bag −0.052% at cells ×1.47; 8 bags −0.078% at ×2.89; fremtpl2 +0.02% at ×9–10 cells; banding time 9 → 71 s | Binds at every bag count, so it replaces `band_tolerance`; a plain tolerance change matches it at 8 bags; slower and larger |
| Price the move on out-of-bag rows | not built | No OOB rows at one bag, where the cap binds; at 8 bags the soup saw each row through ~80% of bags; the honest form costs n_bags banding passes |
| Drop the cap | single-output 8 bags: identical 14/14; fr_instal +0.10% | The only bound at one bag; binds on multiclass at 8 bags |
| Retune the cap (×0.25 to ×4, one bag) | test −0.020%, −0.019%, +0.037%, −0.013%; all medians 0 | Flat across 16× on single-output; cells move monotonically (×1.53 to ×0.71) |
| Keep 0 = "no cap", raise only at one bag | not built | A magic value that switches a budget off; loosens fr_instal by about 0.10% |

### Loose ends

- The validator accepts 1e300, which bands without bound at one bag just like inf. Consider an
  upper limit, or document it.
- The core still treats `BandingRows.mse_cap = INFINITY` and `_TableModel.band(mse_cap=inf)` as
  "no cap". Python never calls them that way after this change; rejecting a non-finite budget in
  the core would need the Rust gates.
- The core's "cannot meet its fidelity budget … tables left unbanded" skip is still reachable for
  valid but near-zero budgets. It is a pre-existing silent skip; it should fail loudly or band
  as far as the budget allows.

## PF-02. The rebalance learns a target-encoding leak

### Finding

A categorical with more levels than `cat_direct_max_levels` is fed to the trees as k-fold
**cross-fit** target statistics. Every post-fit stage bins the training rows through the frozen
**full-data** map instead, and that map's statistic for a level contains its own rows' targets.
A level whose rows happen to sit high is binned high. Levels also share cells, so the cell a row
lands in depends partly on its own target. A re-solve fitted to the training residual on those
cells learns the leak.

- **pg17_instal**, shipped (`85f40f1`): adopted step 0.75. Training deviance fell 1.08% / 1.05%;
  test rose +0.21% / +0.41% (splits 0, 1). On a fresh split 2: +0.216% test, −1.04% training.
- **Every slice endorses it**, out of bag or not, because each row's own target placed it. For the
  shipped correction the out-of-bag slice predicted −0.50% / −0.57% and the in-sample slice
  −0.42% / −0.46%. The out-of-bag minimum sat at the same step as the in-sample one on every
  multiclass fit.
- **Causal test:** with the shipped check unchanged, leaving tables on the target-encoded
  categoricals (vh_model, pol_insee_code, vh_make) as fitted turns pg17_instal into −0.021% /
  −0.015%. The training drop shrinks to 0.13% / 0.03%, and both checks then agree with test.
- **ussev is not explained by the leak.** Excluding CLASS moves the shipped result only from
  +0.099% to +0.089% (split 3 gets worse). The leak story holds for pg17_instal only.

### Recommendation (ready)

The "leak exclusion" (Tier 1) on `team/b-rebalance-leak-and-bag-check`, landed as its own commit:
a table on a categorical whose cells come from a cross-fit target statistic (more levels than
`cat_direct_max_levels`; binary, direct-bypass and count channels excepted) is left as fitted and
listed in `pruning_report_["rebalance"]["excluded"]`. Its parts:

- `prune.rs`: `cell_reads_own_target`; the `encoders` parameter and the `honest` line in
  `rebalance_supports` and its two call sites; the `excluded` doc line; the unit test
  `rebalance_leaves_tables_on_a_cross_fit_categorical_as_fitted`; the updated call in
  `rebalance_resolves_each_raw_feature_to_its_own_axis`.
- The `cat_direct_max_levels` clauses in the `prune_rebalance` docstrings and `pruning.md`, then
  `check_docs.py fix`.
- `test_rebalance.py`: `test_tables_on_a_target_encoded_categorical_are_left_as_fitted`,
  `cat_direct_max_levels=64` in the multichannel test, and the dropped `excluded == []`
  assertion.

About 40 core lines, no cost. Over 45 dataset-splits: −0.019% against no rebalance (9 better, 3
worse), pg17_instal +0.31% → −0.02%, multiclass −0.092%.

Two fixes to make while porting: `pruning.md`'s explanation should say the distinction is
cross-fit training against full-data post-fit binning, with levels sharing cells (it also
applies to direct-bypass categoricals as written); and drop the ussev attribution.

**Known cost:** the criterion is a proxy. It also freezes categoricals with large levels, where the
own-row leak is negligible: fremtpl2's Region (22 levels of about 25k rows) loses the shipped
re-solve's −0.066% on split 0. A data-derived criterion, or training-time binning (PF-05), would
fix that.

## PF-03. The rebalance check is biased against the re-solve

### Finding

On single-output fits the stage did almost nothing: it adopted the correction on 6 of 30 fast
fits, for a mean of +0.005%. The deterministic 15% slice that judges it is **pessimistic**, not
optimistic. The soup trained on the slice rows, so a correction fitted to the other rows'
residual is anti-correlated with the slice's own noise. An out-of-bag jury does not fix this,
because the jury's bags also trained on the slice.

- Applied unconditionally (with the leak exclusion), the correction lowers test deviance on 25
  single-output dataset-splits without a leaky categorical by 0.13% (in-sample residual) or 0.18%
  (out-of-bag residual) on average.
- ausbi split 0: the correction gains 0.59% on test while both slices predicted +0.57%.

### Proposal: per-bag check (maintainer's decision)

On the rest of `team/b-rebalance-leak-and-bag-check`: each bag's keep-set bank is anchored on its
in-bag rows, re-solved there exactly as the deploy re-solves the soup, and scored on its own
out-of-bag rows. The deployed correction is fitted on every row. Single-output keeps the binary
rule; multiclass keeps one joint step from the 8-step grid. `n_bags=1` keeps the slice check and
reports `evidence: "in_sample"`.

Accuracy, against no rebalance (test deviance):

| group | shipped | leak exclusion only | leak exclusion + per-bag check |
|---|---|---|---|
| all, 45 splits | −0.006% (11 better / 5 worse) | −0.019% (9/3) | −0.121% (33/4) |
| single-output, 35 | +0.001% | +0.002% | −0.097% (23/4) |
| multiclass, 10 | −0.031% | −0.092% | −0.205% (10/0) |

| dataset | shipped vs off | final vs off |
|---|---|---|
| swmc | −0.049% | −0.105% |
| ussev | +0.099% | +0.036% |
| surrender | 0 | −0.027% |
| ausbi | 0 | −0.358% |
| lifesurr | 0 | −0.081% |
| lapse | −0.023% | −0.118% |
| norway | −0.007% | −0.015% |
| fremtpl2 | −0.033% | −0.043% |
| pg16_fleet | −0.368% | −0.413% |
| pg17_instal | +0.306% | −0.071% |
| pg17_cover | −0.035% | −0.052% |
| fr_instal | −0.019% | −0.256% |
| prudential | −0.040% | −0.230% |

What the reviewer's independent checks change:

- **Dataset-level, not split-level.** Single-output −0.089% (8 datasets, t ≈ −2.1), and −0.050%
  without ausbi. Multiclass −0.204% (5 of 5 datasets, t ≈ −3.1). The winner was chosen among about
  10 variants on these splits, so these are optimistic.
- **Fresh splits.** Multiclass reproduces: pg17_instal split 2 −0.059% (HEAD +0.216%); fr_instal
  split 2 −0.241% (HEAD −0.055%; no leaky categorical, so this is the check alone).
  Single-output is small to nil on the same datasets (fremtpl2 −0.044%, ausbi split 5 −0.001%,
  ussev split 5 0). On a new wide task, prudential as binary (139 columns), it gains −0.091%,
  −0.139% and −0.196% on splits 0–2.
- **The single-output check is mostly insurance.** It equals always-adopt on 28 of 30 fits. Its two
  declines were one right (ussev split 3, saved 0.164%) and one wrong (surrender split 3). It let
  through ausbi split 4 +0.174% and ussev split 1 +0.188%. The single-output gain comes from no
  longer blocking the re-solve.

Cost, measured one fit at a time:

| fit | fit time | rebalance stage | peak memory |
|---|---|---|---|
| single-output, narrow (surrender, fremtpl2) | +5–10% | fremtpl2 1.6 → 6.6 s | fremtpl2 1.77 → 2.02 GB (+14%) |
| single-output, wide (prudential binary) | +22–25% | 4.2–5.0 → 11.4–12.7 s | +11% |
| multiclass | +17% (fr_instal) to +37% (prudential) | prudential keep-set apply 11 → 89 s | not measured |

The check costs one correction solve per bag (K per bag on multiclass), at most 8 at once, so more
cores don't shorten it.

### Defects to fix before landing

1. When no bag leaves a row out (tiny n, a grouped fit with few groups), single-output adopts
   unchecked and multiclass deploys step 1 unchecked, while the report says `out_of_bag`. Fall
   back to the slice and report `in_sample`.
2. `rebalance_bag_check` and `multiclass_rebalance_bag_check` read membership with
   `in_bag.get(r).unwrap_or(false)` and never check its length against `serve`. Error on a
   mismatch, as `bag_oob_group_sums_with` does.
3. Under `cell_refit_base`, each bag gets the whole soup-level §G1 correction, which saw this
   bag's out-of-bag rows. The `RebalanceEvidence::OutOfBag` doc ("honest for the trees and for the
   correction") is false there. Say it is approximate, or route that setting to the slice.
4. The docs say the deployed values are checked ("kept only if they do not raise the held-out
   deviance"). The check judges the re-solve **procedure** on each bag's out-of-bag rows; the
   deployed all-rows correction is never scored. Reword.

Also: land it as a commit separate from PF-02, so it can be reverted alone. Report semantics change:
`holdout_rows` counts (row, bag) pairs, single-output deviances are sums over bags, and a new
`evidence` key appears. The HTML report renders the block generically; haute's `holdout_rows` is an
unrelated field.

Ways to make it cheaper, none measured: shorten per-solve vectors to the rows used; run fewer
single-output solves at once; solve the deployed correction only after the check passes;
warm-start each bag from the soup's correction; deploy the mean of the bags' corrections, which is
exactly what was judged and saves a solve per class. Sharing bag banks with banding's
`bag_score_variance` would save little (about 2 s of pg17_instal's 11 s).

### Rejected (measured)

| candidate | single-output, 30 fits | multiclass | why rejected |
|---|---|---|---|
| Out-of-bag jury residual and check (the original proposal) | +0.006% (10 adopted) | pg17_instal still step 0.75, +0.21% / +0.52% | The leak is in the serving encoding, which OOB rows share; the jury's bags trained on the slice |
| OOB check only | −0.004% (9) | decisions unchanged | Same biases |
| Ridge ×10 | −0.012% (9) | | Tuned constant; both biases remain |
| 8-step grid on single-output | −0.010% (11); with the per-bag check equal to binary | | A second mechanism with no gain |
| Always adopt, correction on all rows, no check | −0.104% (20/5) | step 1 overshoots (pg16: 0.75 beats 1) | No safety net; multiclass needs a judged step |
| Remove the stage | 0 by definition | | Forgoes −0.121% (33 of 45 splits better with the per-bag check) |

### Related findings in the shipped multiclass rebalance

From the 2026-10-14 measurement on 5 multiclass sets (test cross-entropy, split 0):

| set | neither | graduation | rebalance | both |
|---|---|---|---|---|
| pg16_fleet | 0.971426 | 0.971423 | 0.967115 | 0.967100 |
| pg17_instal | 1.109522 | 1.109168 | 1.111731 | 1.111448 |
| pg17_cover | 0.674745 | 0.674667 | 0.674319 | 0.674383 |
| fr_instal | 0.844575 | 0.844575 | 0.844575 | 0.844575 |
| prudential | 1.148454 | 1.149103 | 1.148454 | 1.149103 |

- The multiclass check scores `raw + step·δ` without the intercept re-anchor the deploy applies
  afterwards (inherited from `168a472`).
- On the out-of-bag path the deployed correction is solved before the check, and that solve is
  wasted when the check declines.

## PF-04. Graduation's 1% check: keep it

### Finding

The 1% check compares training deviance, which rises whether smoothing removes noise or real shape.
No training threshold can tell good smoothing from bad, however it is scaled:

- On single-output fits, training cost is **anti-correlated** with the test effect:
  corr = −0.35 over 35 fits. Larger smoothing went with more benefit (swmc, ussev).
- On fr_instal, per-table training cost predicts test harm: r = +0.72, test cost about 45% of
  training cost.
- Because the sign flips between model types, no single threshold works. Over 76 single-output
  candidates the native check declined 1, and it was the best one (swmc split 9, −0.40%).

The harms first reported were mostly split noise. Paired SEs are 0.02–0.35% per split on the small
sets. ausbi's splits 0 and 1 (+0.34%, +0.13%) were the two worst of ten; over all ten, ausbi
averages −0.015 ± 0.058% and swmc −0.081 ± 0.054%. fremtpl2 averages +0.013% over two splits.

Graduation against `graduate=False` at HEAD, mean over 12 datasets: −0.0043% test. Per dataset
(± SE across splits):

| dataset | splits | HEAD vs graduate=False | no check at all |
|---|---|---|---|
| ausbi | 10 | −0.015 ± 0.058% | same |
| ussev | 10 | −0.029 ± 0.031% | same |
| swmc | 10 | −0.041 ± 0.042% | −0.081 ± 0.054% |
| surrender | 10 | +0.003 ± 0.015% | same |
| lapse | 10 | −0.006 ± 0.007% | same |
| lifesurr | 10 | −0.009 ± 0.007% | same |
| fremtpl2 | 2 | +0.013 ± 0.019% | same |
| pg16_fleet | 4 | +0.001 ± 0.001% | same |
| pg17_instal | 4 | −0.016 ± 0.005% | same |
| pg17_cover | 4 | +0.005 ± 0.006% | same |
| fr_instal | 4 | 0 (declined 4/4) | **+0.517 ± 0.033%** |
| prudential | 2 | +0.042 ± 0.014% | same |

The check is what keeps fr_instal out (see PF-07).

### Recommendation (ready, docs only)

No change to the check. Say in `graduation.md` that it is a brake on heavy smoothing, not a test
of accuracy (the wording in `team/a-gcv-ceiling-rule-rejected` is accurate; port only the docs).
Also fix `docs/algorithm/readable-tables.md`, which still says graduation is not supported for
three or more classes (stale since `85f40f1`).

### Rejected (measured)

| candidate | result | why rejected |
|---|---|---|
| Gate on explained deviance (null − model), c = 1%, 2%, 5% | pooled −0.0035%, −0.0038%, −0.0088% vs −0.0165% ungated | Passes ausbi split 0 (+0.34%), declines swmc split 9 (−0.40%) |
| Tighter slack, 0.3% and 0.1% | pooled −0.0012%, −0.0002% vs −0.0120% native | Declines helpful candidates with the harmful |
| Remove the check | small sets −0.0228%; fr_instal +0.517%; mean +0.0354% | fr_instal |
| Curvature-weighted GCV fidelity | lifesurr +203% (an 11,558-row cell at logit −4.0 pulled 7.6 logits) | Quadratic approximation fails on confident cells; GCV is scale-invariant anyway |
| Known-scale UBRE | surrender +0.83 / +1.44%, lifesurr ~+1%, ausbi +0.64 / +0.25% | Boosted tables are shrunk, so the GLM noise model overstates their noise |
| Semi-honest OOB check (bags' OOB score plus the deployed change) | corr −0.32 with test | The deployed table saw the row |
| Honest per-bag OOB check (smooth each bag, score its OOB rows) | corr −0.02 over 35 fits; sign agreement 15/35 | No resolution at ~0.02% effects (each row OOB for 1.6 of 8 bags) |
| Noise-referenced gate (move ≤ σ² from banding's bag noise) | fr_instal [1.06, 1.38, 2.92, 0.76] (harm) vs pg17_instal [1.42, 1.10, 0.83, 0.35] (benefit) | Doesn't separate |
| Skip banded axes in graduation | mean +0.0063%; fr_instal now adopted at +0.162% | Removes the training cost the brake keys on, not the harm |
| Skip banded axes plus the ceiling rule | mean −0.0046%; fr_instal adopted +0.090% | Same |
| Graduation off by default | 0 by construction | Forgoes the gain; violates "every stage runs" |
| GCV interior-minimum (ceiling) rule | see PF-08 | Reviewer rejected |

## PF-05. The serve-map leak beyond the rebalance (next investigation)

PF-02's leak is not specific to the rebalance. The ranked path's out-of-bag prefix deviances,
banding's curvature-weighted cells and graduation's cell weights all score training rows through
the full-data target-statistic map. Tables on a target-encoded categorical therefore look better
than they are at selection time, and banding and graduation see a distorted picture of them. This
affects **default** fits whenever a categorical has more levels than `cat_direct_max_levels`.

Not measured. Suggested approach:

1. Measure: on pg17_instal, prudential and a single-output set with a high-cardinality categorical,
   compare each stage's training-row evidence for tables on target-encoded categoricals between the
   full-data map and the training-time cross-fit encodings.
2. Root fix: bin training rows for post-fit stages with the same cross-fit encodings the trees saw,
   keeping the full-data map for serving only. That would also let PF-02's exclusion go, and with
   it the Region regression.

## PF-06. Multiclass banding is too coarse on some sets (open decision)

Banding costs +0.15% to +0.48% test (graduate=False) in all 8 investigator multiclass fits, while
raising cells 2.5–5.7×.

| set | setting | banded vs unbanded | cap ×0.25 vs default |
|---|---|---|---|
| fr_instal | 8 bags, splits 0 / 1 / 2 | +0.417% / +0.479% / +0.424% | −0.245% / −0.287% / −0.313% |
| fr_instal | 1 bag, splits 0 / 1 / 2 | +0.348% / +0.315% / +0.299% | split 2: −0.249% |
| pg16_fleet | 8 bags (noise budget binds) | +0.162% / +0.149% | |
| pg16_fleet | 1 bag | +0.313% / +0.279% | |
| pg17_instal | 8 bags, split 2 | +0.012% | |

On fr_instal at 8 bags the per-class share of the cap binds (noise/deviance budget ratio
1.00–1.34). Dropping the cap costs +0.10%, and a 4× tighter cap gains 0.25–0.31% on 3 of 3 splits
and at one bag. On fr_instal split 2 the default fit ships the banded, ungraduated bank
(graduation declined), so users get the full +0.42%.

Before deciding: measure pg17_cover and prudential, and a one-bag cap sweep on pg16_fleet. Options
then include a tighter multiclass cap, a different per-class split of the cap, or a data-derived
budget.

## PF-07. Multiclass graduation over-smoothing, held off only by scale

Ungated, graduation costs fr_instal +0.45% to +0.61% on all four splits (5–7 SE). The 1% check
declines it only because the training rise happens to be ×1.014–1.016; fresh splits 4 and 5 are
also declined. A slightly milder fr_instal-like set would be adopted with harm.

What is known:

- **Banded axes.** Graduation runs after banding, and on banded axes GCV smooths 3–5× more tables
  (fr_instal 177 against 60–77; pg17_cover about 530 against 110). With banding off, fr_instal's
  harm halves (+0.61 / +0.50% → +0.25 / +0.12%). Skipping banded axes alone gets the candidate
  adopted at +0.16% (above).
- **Grid-ceiling choices on rare classes.** fr_instal's two worst tables are rare-class main
  effects (+0.07%, +0.04%); one moves a 1,402-row cell from −0.50 to +0.36 in logit.
- **The softmax-redundant common component.** About 30% of the class-table variance is shared
  across classes, and each class's GCV smooths it along with its own shape. The class banks have
  different merged grids, so they can't be centred cell-wise.
- **Class frequency.** GCV with support weights treats every class's logit cells alike, whatever
  the class's frequency (2.8% for class 2).

Untested idea: graduate before banding. It would keep both stages active on interactions, but
banding's distil target would become the graduated model, unbanded pairs often exceed the
2,500-cell joint cap, and multiclass banding runs natively inside `fit_multiclass_pruned`.

## PF-08. GCV choices at the grid ceiling (Prudential lead)

The investigator proposed refusing a table whose GCV criterion is still falling at the top of the
strength grid (alpha = 3e4), mirroring the existing rule at the weak end. The reviewer rejected it:

- **No general gain out of sample.** On 120 fresh single-output fits it was +0.0044 ± 0.0035%
  against HEAD (swmc +0.032 ± 0.018% worse; lapse's "significant" gain fell to −0.002 ± 0.005%).
  On splits not used to design it, the dataset-level effect was −0.0021 ± 0.0024%. The headline
  came from the motivating splits and one brake flip (swmc split 9, ×1.0107 → ×1.0099).
- **Unit-dependent.** GCV's minimiser scales with the support (weight × exposure). Scaling the
  support by 1000 on identical tables cut the rule from 57 smoothed tables to 17 (HEAD: 87 → 83).
  A swmc fit with `sample_weight=1000` smoothed 0 tables (HEAD 10). Catalog sets with such units:
  swautoins (exposure about 1,090 policy-years a row), pg15training and catelematic13 (days).
- **Wrong rationale.** On a grid extended to 1e10, 20 of 21 ceiling tables have their minimum at
  the linear limit. The data did choose: the straight line.
- **pg17_instal** was worse on 4 of 5 splits (+0.0037 ± 0.0024%).

What stands: **Prudential gains about −0.04% on 3 of 3 splits**, including a fresh one
(−0.038 ± 0.026%). Ceiling tables have 4–24 observed cells.

Requirements for any revised rule:

1. The adopt/refuse decision must not depend on the unit of `sample_weight` or exposure. Test that
   a constant weight multiple, and exposure in days against years, leave the set of smoothed
   tables unchanged. One option: evaluate GCV at the penalty's null-space projection
   (alpha → ∞) and refuse only when that is the minimum. At natural units it matched the rejected
   rule on 6 of 6 fits; at ×1000 it kept 36 tables against 9.
2. Record the reason per table in `graduation_report_` (a refused table currently shows alpha 0,
   "as fitted", the same as a table whose roughness GCV judged real) and set `skipped` when
   nothing is smoothed (PF-10).
3. Keep an end-to-end test of a real GCV candidate declined by the brake.
4. Validate on splits not used to design it (prudential split 3+, small sets 10+), on the
   weighted and large-exposure sets above, and on the multiclass sets.
5. State the real rationale: GCV prefers the straight line, and that preference is not trusted
   on few-cell tables.
6. Measure readability too (the rejected rule smoothed 30–80% fewer tables).

Related: the `w_scale` comment in `_graduate_table_model` says row-count GCV over-smooths by the
mean weight. GCV picks the same smoother when every weight is scaled by one constant, apart from
the grid's edges, so `w_scale` only relocates the grid. It does carry the fixed-alpha semantics.
Fix the comment.

## PF-09. Graduation's re-anchor overrides `reanchor=False` (bug)

An adopted graduation on a log link re-anchors f0 natively (`apply_graduation_updates_impl`,
`if model.link == Link::Log`), whatever `reanchor` says. `reanchor=False` then predicts the same as
the default, and `binding_report_` has no row for it. The `reanchor` docs say an explicit
True/False always overrides the default. `test_reanchor_defaults_on_for_log_link_only` passes at
HEAD only because its fixture's graduation is declined. The rejected PF-08 patch made graduation
adopt there and hid the conflict with `graduate=False`; don't do that.

Fix: honour `reanchor=False` in the graduation re-anchor, or report it as OVERRIDDEN in
`binding_report_`, with a test either way.

## PF-10. `graduation_validation_` has no reason when nothing is smoothed (bug)

The early return in `_graduate_table_model` leaves out `skipped` and `candidate_count` (norway
shows `adopted: False` with no reason), while the later no-update path sets them. Set them on
every path. The monotone tests assert `"skipped" not in graduation_validation_` and will need
rewording.

## PF-11. One-bag fits have no honest held-out evidence

- The rebalance at `n_bags=1` keeps the slice check, which PF-03 showed is biased against the
  re-solve, so the stage is effectively inert there. On panel data the slice is hashed by row,
  not by group.
- Banding at one bag uses the deviance cap alone (σ = ∞).
- The early-stopping carve (`holdout_mask`, deterministic in seed, rows and strata) is honest
  held-out evidence that no post-fit stage uses. Recording it would give the rebalance (and
  possibly banding) a real check at one bag.
- For pruned one-bag fits, the fold vote already fits K fold models; their between-fold spread
  could give a data-derived σ for banding. It can't cover `prune=False` at one bag, and the
  single-output response to the cap is flat, so this is low priority.

## PF-12. `band_tolerance` 0.375 beats 0.75 on small single-output sets

At 8 bags, graduated, on the 7 small fast sets: `band_tolerance=0.375` −0.095% mean / −0.048%
median against 0.75 (8 better, 2 worse), cells ×1.90; 0.25 gives −0.076% at ×2.98. Largest:
ausbi −0.36%, swmc −0.27%. The arena figure of +0.015% pooled for 0.75 may not hold on small
data. Measure on the arena and the large sets before changing the default; a data-derived
tolerance would fit CLAUDE.md better than a new default.

## PF-13. Banding can increase the cell count

Banding densifies factored tables, so cells can rise rather than fall: surrender at one bag
78,915 → 119,209 at the default cap and 595,632 at cap ×0.25; fremtpl2 calibrated 7,240 → 73,077
against 54,066 unbanded; multiclass 2.5–5.7×. This works against readability, and it makes cell
counts (`deployed_census`) a poor readability measure when comparing banding settings. Find where
the densification happens and whether a factored table can be banded in factored form.

## PF-14. Flaky callback test

`test_haute_integration.py::test_callback_true_stops_and_exceptions_propagate` fails under load
(`assert [2, 5] == [5, 5]` on `n_trees_per_bag_` after a stop callback). It failed 2 of 5 times on
unmodified HEAD at load average 20–43. A load-dependent race in the test or in how bags interleave
with the callback; fix the test or make the stop deterministic.

## PF-15. Smaller items

- PF-01's loose ends: 1e300 accepted; the core's `mse_cap = INFINITY` = "no cap"; the core's
  near-zero-budget "left unbanded" silent skip.
- PF-03's: the multiclass check scores without the intercept re-anchor; the wasted deployed solve
  on a decline.
- PF-08's: the wrong `w_scale` comment.
- Coverage gap across all three investigations: no Tweedie, squared error, real weighted or
  grouped (panel) data in the accuracy evidence; B checked only that those configurations run.
  Monotone fits were checked for B on fremtpl2 (BonusMalus +1): no harm from the repair.

## Order of work (suggested)

1. PF-01, PF-02 and PF-04's doc fixes: no cost, ready, byte-identical apart from the exclusion.
2. PF-09, PF-10, PF-14: small bugs.
3. PF-05: the serve-map leak, which affects default fits.
4. PF-03: decide on the per-bag check, fixing its four defects first.
5. PF-06 and PF-07 together (both are multiclass strength problems), then PF-08, PF-11, PF-12,
   PF-13.
