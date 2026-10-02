# Bug remediation roadmap

This document records 64 confirmed findings from four codebase review passes on
2026-10-02. The second pass added BUG-022 through BUG-038 (17 new findings); the
third added BUG-039 through BUG-053 (15 new findings). The fourth added BUG-054
through BUG-064 (11 new findings) and another reproduction for BUG-024.
It is a backlog for fixes, not a record of completed remediation. This is not a
guarantee that all bugs have been identified.

- **Reviewed revision:** `6588084a35615677b9ca851169050ee4e12ad1f1`.
- **Reviewed package version:** `0.6.1`.
- **Status:** all findings are open at the time of writing. All 64 were
  independently re-verified on 2026-10-02 against the same revision; see
  [Verification pass](#verification-pass) for the outcome, the corrections it
  produced, and the per-finding verdicts in the index below.
- **Scope:** Rust training, data preparation, table decomposition and serving,
  serialization, Python bindings and estimators, development tooling, and CI.
- **Evidence:** runtime reproductions unless stated otherwise. BUG-001 was tested
  using an exact extraction of a private sampler; BUG-008 was reproduced at the
  helper level; BUG-019 and BUG-020 were established by workflow and upstream
  implementation/documentation review, without executing GitHub Actions.
- **Source references:** line numbers refer to the reviewed revision and may move
  as fixes land. Reproduction inputs and observed outputs are preserved below so
  this document does not depend on temporary review files.

## Priorities and index

**P1** means a high-priority correctness defect that can silently corrupt training
or predictions. **P2** means a substantive correctness, compatibility, validation,
or CI defect. **P3** means lower-priority API brittleness or a development safeguard failure.
Priority is not a claim that every default configuration reaches the defect.

Current totals: **3 P1**, **56 P2**, and **5 P3**. Findings identify their affected
surface explicitly: some occur in ordinary estimator use, some require optional
configurations or public Rust APIs, and others require malformed input.

| ID | Priority | Finding | Area | Verified 2026-10-02 |
| --- | --- | --- | --- | --- |
| [BUG-001](#bug-001) | P1 | MVS importance weights do not match the sampling distribution | Training | Reproduced |
| [BUG-002](#bug-002) | P1 | Failed refits preserve models with corrupted metadata | Python estimator state | Reproduced |
| [BUG-003](#bug-003) | P1 | Multiclass categorical encoders see early-stopping holdout labels | Training / binding | Reproduced |
| [BUG-004](#bug-004) | P2 | Cross-fitted categorical encodings leak targets through their prior | Categorical encoding | Reproduced |
| [BUG-005](#bug-005) | P2 | Missing-versus-present signals cannot produce a split | Split search | Reproduced |
| [BUG-006](#bug-006) | P2 | Numeric midpoint rounding merges distinct float32 values | Binning | Reproduced |
| [BUG-007](#bug-007) | P2 | Multiclass pruning normalizes fold loss twice | Pruning | Reproduced |
| [BUG-008](#bug-008) | P2 | Binary pruning guard uses a log-link intercept correction | Pruning / Python | Reproduced |
| [BUG-009](#bug-009) | P2 | Clipped component relativities fail to reconstruct predictions | Rating export | Reproduced |
| [BUG-010](#bug-010) | P2 | Pre-v7 table JSON fails the version gate | Serialization compatibility | Reproduced |
| [BUG-011](#bug-011) | P2 | Feature names can change JSON decoder selection | Python serialization | Reproduced |
| [BUG-012](#bug-012) | P2 | Unbounded factored order panics during model loading | Load validation | Reproduced (malformed input) |
| [BUG-013](#bug-013) | P2 | Nonfinite factored coefficients pass validation | Load validation | Reproduced (malformed input) |
| [BUG-014](#bug-014) | P2 | Empty bagged fits panic | Input validation | Reproduced (Rust API only) |
| [BUG-015](#bug-015) | P2 | Merged parameter aliases fail through `set_params` | Estimator parameters | Reproduced |
| [BUG-016](#bug-016) | P2 | Objective aliases produce wrong explanation and reporting semantics | Python API | Reproduced |
| [BUG-017](#bug-017) | P2 | Binary A/E reports sum class labels | Reporting | Reproduced |
| [BUG-018](#bug-018) | P2 | Deployed factored effects are reported as absent | Pruning reports | Reproduced |
| [BUG-019](#bug-019) | P2 | Fuzz workflow uses the wrong dictionary path | CI | Source-verified (not executed) |
| [BUG-020](#bug-020) | P2 | Release Gate can smoke-test a published package instead of its artifact | Release verification | Source-verified (not executed) |
| [BUG-021](#bug-021) | P3 | Multiline derives bypass serialized-field gates | Development tooling | Reproduced |
| [BUG-022](#bug-022) | P2 | Forked workers inherit a serving pool whose threads no longer exist | Process lifecycle | Reproduced |
| [BUG-023](#bug-023) | P2 | Multiclass prediction ignores the estimator's thread budget | Multiclass serving | Reproduced |
| [BUG-024](#bug-024) | P2 | Categorical identities change across containers and optional pandas availability | Categorical ingestion | Reproduced |
| [BUG-025](#bug-025) | P2 | Classifier serialization corrupts large unsigned integer labels | Serialization | Reproduced |
| [BUG-026](#bug-026) | P2 | Runtime-only classifier accepts NaN and infinite class labels | Optional dependencies | Reproduced |
| [BUG-027](#bug-027) | P2 | Loaded estimators silently drop the unspecified half of table export mass | Export / serialization | Reproduced |
| [BUG-028](#bug-028) | P3 | Empty categorical index lists are misclassified as Boolean masks | Input declaration | Reproduced |
| [BUG-029](#bug-029) | P3 | No-argument set_params discards a fitted model | Estimator state | Reproduced |
| [BUG-030](#bug-030) | P2 | Re-banding a banded model ignores its existing band maps | Banding | Reproduced |
| [BUG-031](#bug-031) | P2 | Cached CellMaps omit dependencies from compatibility checking | Core scoring cache | Reproduced; partial check is documented |
| [BUG-032](#bug-032) | P2 | Recentring a banded interaction-only bank loses axis templates for recreated mains | Recentring | Reproduced (Rust API only) |
| [BUG-033](#bug-033) | P2 | Model loading accepts mismatched feature-set/axis identities and misattributes effects | Load validation / explanations | Reproduced (malformed input) |
| [BUG-034](#bug-034) | P2 | Gamma/Tweedie exposure initialization uses the Poisson optimum | Loss initialization | Reproduced |
| [BUG-035](#bug-035) | P2 | Periodic ridge refits desynchronize AGBM's score cache from its trees | Training state | Reproduced (Rust API only) |
| [BUG-036](#bug-036) | P2 | Bootstrap copies cross the early-stopping train/validation boundary | Early stopping / bagging | Reproduced; documented tradeoff |
| [BUG-037](#bug-037) | P2 | Bagged core fitting panics on a short fixed-holdout mask | Core input validation | Reproduced (Rust API only) |
| [BUG-038](#bug-038) | P2 | Core multiclass fitting silently accepts invalid sample weights | Core input validation | Reproduced (Rust API only) |
| [BUG-039](#bug-039) | P2 | An uninformative weighted feature can abort an otherwise valid fit | Weighted binning | Reproduced |
| [BUG-040](#bug-040) | P2 | Small updates disappear from training scores but accumulate in deployed tables | Training / prediction precision | Reproduced |
| [BUG-041](#bug-041) | P2 | Pricing reports label merged-grid A/E aggregates with an interaction's compressed axis | Pricing reports | Reproduced |
| [BUG-042](#bug-042) | P2 | Multiclass contribution DataFrames crash when one class has only an intercept | Multiclass explanations | Reproduced |
| [BUG-043](#bug-043) | P2 | Gini awards arbitrary ranking skill to tied scores according to row order | Ranking metrics | Reproduced; tie-break is documented |
| [BUG-044](#bug-044) | P2 | Categorical rating exports omit and collide on routing metadata | Categorical rating export | Reproduced; Repro A as documented |
| [BUG-045](#bug-045) | P2 | Deviance metrics accept nonfinite inputs and an infinite Tweedie power | Metric validation | Reproduced |
| [BUG-046](#bug-046) | P2 | A/E reports suppress valid ratios for negative expected totals | Reporting | Reproduced |
| [BUG-047](#bug-047) | P3 | Empty A/E evaluation batches raise an internal indexing error | Empty evaluation batches | Reproduced |
| [BUG-048](#bug-048) | P2 | OOB cell correction silently breaks monotonicity in the Rust API | Core monotonicity | Reproduced (Rust API only) |
| [BUG-049](#bug-049) | P2 | Valid weighted fits fail when a bag omits all positive-weight observations | Weighted bagging | Reproduced |
| [BUG-050](#bug-050) | P2 | The cell-refit guard splits declared groups between fitting and validation | Grouped validation | Reproduced |
| [BUG-051](#bug-051) | P2 | Table-budget checks do not cover purification and factored shedding | Table memory budgets | Reproduced (Rust API only) |
| [BUG-052](#bug-052) | P2 | Public exactness assertion ignores the mass used to build weighted banks | Weighted certification | Reproduced; API gap |
| [BUG-053](#bug-053) | P2 | Variance certification loses precision after harmless intercept translation | Numerical certification | Reproduced (Rust API only) |
| [BUG-054](#bug-054) | P2 | Incremental Poisson scores lose the distance beyond the exponent clamp | Poisson training cache | Reproduced |
| [BUG-055](#bug-055) | P2 | Runtime-only score methods silently broadcast or flatten incompatible targets | Optional-dependency scoring | Reproduced |
| [BUG-056](#bug-056) | P2 | Binary A/E ignores the exposure offset that was used during training | Binary exposure reporting | Reproduced |
| [BUG-057](#bug-057) | P2 | A literal missing-sentinel category is silently merged with actual missing values | Categorical ingestion | Reproduced |
| [BUG-058](#bug-058) | P2 | Estimator envelopes accept metadata inconsistent with the native model | Envelope validation | Reproduced (malformed input) |
| [BUG-059](#bug-059) | P3 | Runtime-only estimator repr fails for valid NumPy-array parameters | Optional-dependency representation | Reproduced |
| [BUG-060](#bug-060) | P2 | Reconstructed bags reuse the soup intercept, leaking targets into OOB evidence | OOB pruning evidence | Reproduced |
| [BUG-061](#bug-061) | P2 | A fixed ten-step intercept reanchor can stop far from class balance | Multiclass calibration | Reproduced |
| [BUG-062](#bug-062) | P2 | Recentring a pruned high-order bank turns recreated categorical effects into numeric exports | Categorical table metadata | Reproduced (Rust API only) |
| [BUG-063](#bug-063) | P2 | Recentring an interaction-only bank reports zero support for recreated main effects | Recentring support | Reproduced (Rust API only) |
| [BUG-064](#bug-064) | P2 | Joint exports normalize component variances as though the effects were independent | Joint-reference importance | Reproduced |

## Validation baseline from the first pass

The existing checks below passed during the review, including a rebuild of the
Python extension before the final Python test run. Their success does not cover
the new counterexamples recorded in this document.

| Check | Result |
| --- | --- |
| `cargo test --release -p t-boost-core --all-features` | Passed |
| `cargo test --release -p t-boost-core --all-features --test '*' -- --ignored` | All 14 slow integration tests passed |
| `cargo test -p t-boost-core --doc` | Passed |
| `cargo test -p xtask` | All 9 tests passed |
| `cargo fmt --all --check` | Passed |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | Passed |
| `cargo run -p xtask -- check-all` | Passed |
| Two independent `xtask bit-repro --seed 7` runs, followed by `cmp` | Byte-identical |
| `uv sync --reinstall-package t-boost` | Extension rebuilt successfully |
| `uv run --no-sync pytest python/tests -q` | 517 passed; 85 warnings |
| `uv run --no-sync mypy --strict python/t_boost` | Passed |
| `uv run --no-sync python -m mypy.stubtest t_boost._t_boost` | Passed |

GitHub Actions, cross-platform builds, and the complete fuzz workflow were not
executed as part of the local review. The review did not modify production code.

**Second-pass validation.** The additional 17 findings have executed, targeted
counterexamples against the unchanged release core or the already-rebuilt Python
extension. These include fork/spawn controls, per-thread CPU observations,
cross-container and serialization comparisons, blocked optional-import
subprocesses, repeated banding, cached/uncached scoring, malformed load/input
cases, and instrumented production-loss callbacks for training state/holdouts.
The exposure initialization result was also checked independently against the
implemented gradients. Instrumentation and malformed-input cases are identified
in their entries; they are not presented as benchmark accuracy measurements.
The full suites above were not rerun during the second pass. No production code
or permanent regression tests were added.

This pass did not exhaust parameter combinations or establish absence of further
bugs. In particular, a source-review suspicion is not counted without a concrete
failure. Tested controls that behaved correctly include all-categorical
`mc_supports` with omitted versus explicit unit weights, pandas integer column
names, unbanded recentering, and AGBM without periodic refitting.

**Third-pass validation.** Fifteen additional findings have executed, targeted
counterexamples against the unchanged release core or rebuilt Python extension.
Checks include translated-target controls and native/table score comparisons,
weighted feature/binning and bag-sampling cases, banded report labels, contribution
DataFrame schemas, tied-score metrics, categorical export routing, empty/signed
A/E reports, constrained and grouped cell correction, small table-budget fixtures,
and weighted/numerical certification. Group-guard instrumentation delegates the
underlying loss calculations unchanged. Rust/Python mirror paths established only
by source inspection are identified in their entries. The Rust harnesses used
release optimization and overflow checks. No production code or permanent tests
were changed, and the full baseline suites were not rerun during this pass.

Controls that succeeded include unconstrained/no-correction comparisons, original
unshifted score and variance fixtures, dropping ineffective weighted data, native
categorical serialization/scoring, and an ensemble-average probe with an unused
all-missing feature. Unconfirmed leads were excluded. These passes do not exhaust
the combinations of data, parameters, and execution environments.

**Fourth-pass validation.** Eleven further findings have executed targeted
counterexamples. These cover clamped incremental Poisson scores, scoring and
representation without optional scikit-learn, binary exposure reporting, literal
missing-category collisions, corrupted Python envelope metadata, OOB bag
intercepts, multiclass intercept convergence, categorical metadata and support
propagation during re-centering, and joint-reference variance reporting. An
additional small/large-batch pandas reproduction extends BUG-024 without counting
it twice. Rust examples used release optimization with overflow checks. Selected
Python results and the cache's cause were independently checked. Malformed
envelope cases and handcrafted valid core models are identified explicitly; they
are not presented as ordinary writer failures or training benchmarks. No
production code or permanent tests changed, and the full baseline suites were
not rerun. Unconfirmed leads were excluded.

<a id="verification-pass"></a>
## Verification pass (2026-10-02)

An independent pass re-verified every finding against the reviewed revision
(HEAD `06d2700` differs from it only by this document). Each reproduction was
re-run — 27 Rust harnesses built as temporary binaries against the release core
with overflow checks, 44 Python scripts through `uv run --no-sync python` — and
every cited `file:line` was opened to confirm the stated cause. Docstrings,
code comments, README and the test suites were then searched for evidence that
the observed behaviour is deliberate. No production code was changed.

**Outcome.** All 64 findings reproduce. Sixty-one executed and matched the
quoted numbers to the printed digit (panic sites, overlap-ID lists and the
10,000-seed MVS sweep included); BUG-019 and BUG-020 were established from the
primary sources named in their entries. Every cited line is accurate to within
one line. The pass changed the reading of several entries:

- **BUG-036 is a documented tradeoff, not an undisclosed leak.** The bootstrap
  train/validation overlap is the stated reason the `bag_subsample` default is
  `0.8` (`boosters.rs:176-178`; the `bag_subsample` docstring at
  `sklearn.py:5846-5852` says so and tells the user "Set to `1.0` to restore
  classic bootstrap bagging"). The entry has been annotated; it is a known
  limitation and any fix is an enhancement.
- **BUG-005 named a type that does not exist** (`HistogramMode`); corrected to
  `Config.hist_precision: HistPrecision::{FullF64, QuantizedI32}`.
- **BUG-057's control claim is conditional**: the `__t_boost_rare__` guard is
  active only when `cat_min_data_per_group > 0`.
- **BUG-007's double division is acknowledged in the code**
  (`prune.rs:3625-3629`) on the premise that it "cancels in every comparison";
  the unequal-fold-mass reproduction falsifies that premise. The verdict stands;
  the entry now cites the comment and its `se_rule` concern.
- **BUG-019 is currently masked by an earlier failure**: the only recorded run
  of `fuzz.yml` dies in `cargo install cargo-fuzz --version 0.13.1 --locked`
  (its locked `rustix 0.36.5` no longer builds on nightly), before the
  dictionary step is reached.
- **BUG-020 is live, not hypothetical**: PyPI already serves `0.6.1`, equal to
  the workspace version, so the smoke step's choice between index and `dist`
  is already unspecified.

**Reproduced behaviour that matches a documented choice.** These reproduce
exactly but the behaviour is documented, so each is a policy call rather than
a plain defect: BUG-031 (`CellMaps` docs make the compatibility check a caller
obligation), BUG-036 (above), BUG-043 (the index tie-break is documented for
Rust parity; what is *not* documented is that ties earn ranking skill, and
`test_sklearn.py:1146` asserts the AUC identity — the verdict leans defect),
BUG-044 (Reproduction A is exactly what the `tables` docstring describes;
Reproduction B's literal `<rare>` collision is a real, narrow ambiguity), and
BUG-052 (the assertion's docstring says it rebuilds weights from row counts;
there is no weighted counterpart — an API gap).

**Reproduced contract violations reachable only through the Rust API or
malformed input** (each entry already says so): BUG-014, BUG-032, BUG-035,
BUG-037, BUG-038, BUG-048, BUG-051, BUG-053, BUG-062, BUG-063 (Rust API);
BUG-012, BUG-013, BUG-033, BUG-058 (malformed input). BUG-053 trips only at
the quoted magnitudes (coefficient `1e5` / offset `1e7`; `1e4` / `1e6` pass).
The remaining 44 findings are reachable from ordinary or opt-in Python use.

**Additional facts recorded in the entries.** BUG-037: an over-long
`fixed_holdout` with two bags is silently accepted. BUG-012: factored orders
9–63 load without error. BUG-010: a second version gate also rejects pre-v7
JSON. BUG-022: a post-fork `n_jobs=None` prediction on a large query also
hangs. BUG-011: `"t-boost-multiclass-tables"` misroutes too. BUG-016: the
defect survives bytes serialization. BUG-018: `deployed_census()` counts the
factored effect that `deployed_supports()` omits, and `test_prune.py:48-53`
asserts the guarantee being violated. BUG-048: the ridge-refit path does
preserve monotonicity (`boost.rs:4769`); cell refit is the exception.

**Priority note.** By this document's own P1 definition — silent corruption of
training or predictions — BUG-024, BUG-034, BUG-040 and BUG-049 are
candidates: all are reachable from default Python use with ordinary finite
input (BUG-049 fails on the default `n_bags=8`; BUG-040's error grows with tree
count), whereas BUG-001 needs `subsample` and BUG-003 an explicit `es_holdout`.
Priorities were left unchanged pending the maintainer's decision.

## Reproduction conventions

Run Python examples through `uv run --no-sync python` in an environment prepared
with `uv sync`. Rebuild with `uv sync --reinstall-package t-boost` after Rust
changes, as required by [CONTRIBUTING.md](../../CONTRIBUTING.md).

Several Python examples use this shared setup:

```python
import numpy as np
import polars as pl

from t_boost import TBoostClassifier, TBoostRegressor

OPTIONS = dict(
    n_trees=8,
    n_bags=1,
    prune=False,
    validation_fraction=None,
    min_data_in_leaf=2,
    n_jobs=1,
)
X = np.arange(80, dtype=np.float32).reshape(-1, 1)
```

Rust reproductions used the current release build with overflow checks enabled.
The fixture helpers named below are
`t_boost_core::explain::{fixture_model, fixture_serve}`. Proposed regression checks
are acceptance criteria for future fixes; they have not been added to the suite.

Every reproduction below, plus the controls run during the verification pass, is
checked in under [repro/](repro/README.md) as runnable scripts
(`repro/python/bugNNN.py`) and a standalone Rust harness crate
(`repro/rust/src/bin/bugNNN.rs`). Use those rather than retyping the listings.

<a id="bug-001"></a>
## BUG-001 — MVS importance weights do not match the sampling distribution

**Priority:** P1. **Status:** Open.

**Source:** [engine/boost.rs:5404](../../crates/t-boost-core/src/engine/boost.rs#L5404);
the multiclass implementation repeats the calculation around
[line 5477](../../crates/t-boost-core/src/engine/boost.rs#L5477).

**Trigger and impact.** Enable MVS row subsampling with heterogeneous gradient
magnitudes. Incorrect row multipliers bias histogram gradient/Hessian totals,
split gains, and credibility thresholds. Full-data leaf refitting cannot undo an
incorrect choice of tree structure. The default `subsample=None` uses full rows
and does not exercise this sampler.

**Cause.** `sample_rows` selects exactly `k` rows using weighted sampling without
replacement, ranking `ln(U) / s_i` where `s_i = sqrt(g_i^2 + h_i^2)`. It then
weights selected rows by `max(1, W / (k * s_i))`, as though their inclusion
probability were `min(1, k * s_i / W)`. That is not the inclusion probability of
the implemented sampling procedure. Capping heavy-row probabilities also means
that choosing the threshold as `W / k` does not generally make probabilities sum
to `k`.

**Reproduction.** The private scalar sampler was copied verbatim into an isolated
Rust harness and exercised with:

```text
g = [100, 0, 0]
h = [1, 1, 1]
all_rows = [0, 1, 2]
Sampling::Mvs { rate: 0.5, min_rows: 1 }
round = 0
```

Two rows are selected. The true population Hessian total is `3`, but a light row
receives multiplier `51.0024999375`. Every two-row sample necessarily contains at
least one light row.

```text
seed 0: rows = [0, 1]
multipliers = [1.0, 51.0024999375]
estimated Hessian = 52.0024999375

10,000 seeds: inclusion counts = [9997, 4887, 5116]
mean estimated Hessian = 52.0175006875
```

This establishes biased sampling statistics, not a measured end-to-end accuracy
delta on a benchmark. The multiclass defect was identified in its corresponding
implementation; the empirical inclusion-frequency run used the scalar sampler.

**Suggested fix.** Use a sampling design with known inclusion probabilities and
derive its correction from that same design. If using thresholded inclusion
probabilities, solve the threshold consistently after saturation. Share the
probability/correction logic between scalar and multiclass sampling.

**Acceptance checks.** Cover equal scores, one dominant score, multiple saturated
scores, full sampling, and minimum-row floors. Compare empirical weighted totals
with population gradient/Hessian totals using a deterministic seed sweep and a
justified tolerance. Preserve byte determinism across thread counts.

<a id="bug-002"></a>
## BUG-002 — Failed refits preserve models with corrupted metadata

**Priority:** P1. **Status:** Open.

**Source:** [_clear_stale_fit_state, sklearn.py:2237](../../python/t_boost/sklearn.py#L2237)
and [classifier fit, line 7898](../../python/t_boost/sklearn.py#L7898).

**Trigger and impact.** Fit an estimator successfully, then call `fit` again with
invalid input and catch its error. Subsequent prediction remains available but
can silently use the wrong class labels or feature-column order.

**Cause.** Refit clears or replaces metadata before validation/training succeeds,
while retaining the previous `_model`. Classifier fit also overwrites `classes_`
before the failing validation. The remaining state is neither the old fitted
estimator nor a consistently unfitted estimator.

**Reproduction.** Using the shared Python setup:

```python
m = TBoostClassifier(**OPTIONS).fit(X, np.repeat(["a", "b"], 40))
try:
    m.fit(X, np.repeat(["wrong1", "wrong2"], 40), sample_weight=np.ones(79))
except ValueError:
    pass
print(np.unique(m.predict(X)))  # ['wrong1', 'wrong2'] using the old model

frame = pl.DataFrame({"a": X[:, 0], "b": np.zeros(80, dtype=np.float32)})
r = TBoostRegressor(**OPTIONS).fit(frame, X[:, 0])
reordered = frame.select(["b", "a"])
before = r.predict(reordered)
try:
    r.fit(frame, X[:-1, 0])
except ValueError:
    pass
print(r.__sklearn_is_fitted__())               # True
print(hasattr(r, "feature_names_in_"))         # False
print(np.max(np.abs(before - r.predict(reordered))))  # 43.57222748
```

**Expected behavior.** A failed refit must either preserve the complete previous
fitted state or leave the estimator wholly unfitted, with prediction refused.

**Suggested fix.** Stage new model and metadata together and commit them only
after success, or consistently invalidate every fitted attribute before starting
the refit. Apply the same policy to scalar and multiclass state.

**Acceptance checks.** Test failures in shape validation, categorical ingestion,
weight validation, and native fitting. Check predictions, class labels, feature
name alignment, fitted-state detection, and serialization after each failure.

<a id="bug-003"></a>
## BUG-003 — Multiclass categorical encoders see early-stopping holdout labels

**Priority:** P1. **Status:** Open.

**Source:** [fit_multiclass_owned_bagged, lib.rs:7583](../../crates/t-boost-py/src/lib.rs#L7583).
The holdout-aware binning API is documented in
[data/bin.rs:198](../../crates/t-boost-core/src/data/bin.rs#L198).

**Trigger and impact.** Fit a multiclass model with native categorical features
and a caller-supplied early-stopping holdout, including holdouts produced for
grouped data. Validation labels influence the categorical representation used by
training and validation, making the early-stopping signal contaminated.

**Cause.** `es_holdout` reaches `FitSpec::fixed_holdout`, but categorical assembly
calls `bin_train_columns` without that mask. The scalar path already has a
`bin_train_columns_with_holdout` path that excludes holdout targets from encoders.

**Reproduction.** This example is independent of the shared Python fixture:

```python
import json
import numpy as np
from t_boost._t_boost import _Booster

x = np.empty((160, 0), dtype=np.float32)
cats = [["a"] * 40 + ["b"] * 40 + ["c"] * 40 + ["heldout_only"] * 40]
holdout = [False] * 120 + [True] * 40
y = np.array([0] * 40 + [1] * 40 + [2] * 40 + [0] * 40, dtype=np.float32)
changed = y.copy()
changed[120:] = 2
booster = _Booster(n_trees=1, n_jobs=1, cat_smooth=0, cat_min_data_per_group=0)

for target in (y, changed):
    m = booster.fit_multiclass(
        x, target, 3, ["a", "b", "c"], cat_x=cats, es_holdout=holdout
    )
    enc = json.loads(m.to_json())["model"]["classes"][0]["schema"]
    enc = enc["cat_encoders"]["encoders"][0]
    print([(v["label"], v["encoding"]) for v in enc["levels"]])
    print(enc["base"])
```

The held-out-only category is encoded as `0.0` in the first fit and `2.0` in the
second. The encoder base changes from `0.75` to `1.25`, although all train-proper
rows and their labels are unchanged.

**Suggested fix.** Pass the same holdout mask to multiclass categorical encoding
and training. Ensure rare-level pooling, smoothing, channel admission, and the
serve map cannot consult holdout targets. Audit automatically derived holdouts
for the same boundary; the concrete reproduction above uses an explicit mask.

**Acceptance checks.** Changing only holdout labels must not change the fitted
encoders or train-proper binned representation. Cover mean and class-frequency
channels, all-categorical inputs, grouped holdouts, bagging, and pruning folds.
Do not require the selected stopping round to remain unchanged: holdout labels
are legitimately allowed to affect that decision.

<a id="bug-004"></a>
## BUG-004 — Cross-fitted categorical encodings leak targets through their prior

**Priority:** P2. **Status:** Open.

**Source:** [cat.rs:1036](../../crates/t-boost-core/src/cat.rs#L1036),
[global smoothing resolution, line 1045](../../crates/t-boost-core/src/cat.rs#L1045),
and [KFold encoding, line 2002](../../crates/t-boost-core/src/cat.rs#L2002).

**Trigger and impact.** Use cross-fitted or ordered target statistics with
nonzero smoothing. Excluding a row from the local category total is insufficient
when the prior still contains its target. The documented own-target exclusion
contract fails and may encourage overfitting, especially for influential targets.

**Cause.** The encoder computes its base and automatic smoothing parameters from
all rows, then reuses them inside the withheld-row encoding schemes. Ordered and
LeaveOneOut encodings share the global-prior issue.

**Reproduction.** Call `fit_cat_encoder` with 30 identical category labels,
targets `0..29`, seed `123`, `Smooth::Fixed { m: 10.0 }`, and
`direct_max_levels=0`. Refit after changing only `y[0] += 10000` and inspect row
zero's training encoding:

| Scheme | Original encoding | After changing only its own target |
| --- | ---: | ---: |
| KFold, `k=5` | 14.8 | 110.0381 |
| LeaveOneOut | 14.871795 | 100.34188 |
| Ordered, one permutation | 16.619047 | 175.34921 |

The existing own-target-invariance test uses zero smoothing and therefore does
not exercise this dependency. This issue is distinct from BUG-003: it concerns
the encoder's own leakage scheme even without an outer early-stopping holdout.

**Suggested fix.** Compute priors and any target-dependent smoothing statistics
using only the information permitted by each fold, leave-one-out exclusion, or
ordered prefix. Define an explicit fallback for an empty permitted sample.

**Acceptance checks.** Test own-target invariance with positive fixed smoothing
and automatic smoothing under every leakage scheme, including weights and
exposure. Preserve intended serve-map fitting on the full permitted training set.

<a id="bug-005"></a>
## BUG-005 — Missing-versus-present signals cannot produce a split

**Priority:** P2. **Status:** Open.

**Source:** [split.rs:2266](../../crates/t-boost-core/src/engine/split.rs#L2266),
[shared scanner, line 1077](../../crates/t-boost-core/src/engine/split.rs#L1077),
and its candidate range around line 1155.

**Trigger and impact.** A feature has one distinct finite value and informative
missingness. The model cannot learn that signal, although its tree representation
can express the required split.

**Cause.** Axis prefiltering and scanning require at least two finite data bins.
Missing bin zero is excluded from that count, and the threshold search omits the
endpoints needed for a pure missing-versus-present split. Canonical routing in
`engine/mod.rs::low_bit` already supports `bin_le=0` with missing values routed
left and finite values routed right.

**Reproduction.** Use squared error, 100 trees, learning rate `0.5`, lambda `0`,
and no validation carve:

```text
x = [NaN repeated 50 times, 1 repeated 50 times]
y = [0 repeated 50 times, 10 repeated 50 times]
```

Both `Config.hist_precision` settings, `HistPrecision::FullF64` and
`HistPrecision::QuantizedI32`, produce zero trees and predict `5` for every
row. Replacing the missing half with finite zero gives approximately `0` and `10`
for the two groups under the same settings. (Verification 2026-10-02: an earlier
revision of this entry named a nonexistent `HistogramMode` type. The grid for the
missing/finite column has `borders = []`, `n_bins = 2`; the comment at
`split.rs:2262` treats "fewer than two data bins ⇒ no candidate split" as the
designed rule, so the fix changes a stated rule rather than an accident.)

**Suggested fix.** Admit pure missing/present split candidates where both sides
have sufficient support, including the single-finite-bin case. Update both the
axis filter and the shared threshold scanner.

**Acceptance checks.** Cover both histogram modes, missing-left/right behavior,
no-missing and all-missing controls, credibility floors, monotonic constraints,
and prediction/decomposition agreement for the new split representation.

<a id="bug-006"></a>
## BUG-006 — Numeric midpoint rounding merges distinct float32 values

**Priority:** P2. **Status:** Open.

**Source:** [data/grid.rs:138](../../crates/t-boost-core/src/data/grid.rs#L138).
The corresponding numeric refill logic should be audited with the same rule.

**Trigger and impact.** Adjacent representable float32 values have a midpoint
that rounds upward when converted back to float32. Distinct inputs collapse into
the same bin despite adequate bin capacity, destroying a potentially predictive
distinction. This is not precision lost by converting a float64 input to float32:
both example values are already distinct float32 values.

**Cause.** The grid stores `(f64(a) + f64(b)) / 2` as float32 without ensuring the
stored border is strictly less than `b`. Binning assigns values equal to a border
to its lower bin. The categorical grid code already contains a related rounding
correction.

**Reproduction.** `build_grid` with default configuration on
`[16777218f32, 16777220f32]` produces:

```text
borders = [16777220.0]
bin(16777218) = 1
bin(16777220) = 1
```

Repeating each value 50 times and using the two-group target from BUG-005 again
fits zero trees with predictions equal to `5`.

**Suggested fix.** For an upper-inclusive threshold separating distinct `a < b`,
enforce `a <= border < b` after conversion. Use the lower endpoint or a suitable
representable predecessor when midpoint rounding reaches the upper endpoint.

**Acceptance checks.** Cover adjacent float32 values on both rounding parities,
negative values, subnormals, and extreme finite values. Exercise both midpoint
construction and refill paths. Assert separation and strict border ordering.

<a id="bug-007"></a>
## BUG-007 — Multiclass pruning normalizes fold loss twice

**Priority:** P2. **Status:** Open.

**Source:** [prune.rs:2933](../../crates/t-boost-core/src/prune.rs#L2933).

**Trigger and impact.** Pruning folds have different total sample weights or row
counts. Smaller-mass folds receive disproportionate influence, changing the
selected tables and the reported loss scale.

**Cause.** `multiclass_deviance_for_rows` already returns a weighted mean. The
pruning evaluator divides it again by the fold's weight sum before computing the
mean and standard error across folds. The extra division is not a common scaling
factor when fold masses differ.

The division is at `prune.rs:2938`, inside the block beginning at the cited
line. It is acknowledged in the code: the comment at `prune.rs:3625-3629` calls
it "a long-standing constant rescale that cancels in every comparison it makes"
and says it was left alone because changing it would move the `se_rule` band.
The reproduction below falsifies that premise — the rescale is only constant
when every fold carries the same mass — so a fix must also re-derive the
`se_rule` band on the corrected scale. (`prune.rs:78` documents the reported
deviance as "per unit weight"; the scalar path divides once because
`Loss::deviance` returns a weighted sum.)

**Reproduction.** Build three class models from `[fixture_model(), zero, zero]`,
where each zero model is the same fixture with its trees removed. Use
`fixture_serve()`, class labels `a/b/c`, target indices `[1,0,0,0]`,
`RefMeasure::Uniform`, selection rows `[0,1,2,3]`, two folds, and `se_rule=0`.
Call `prune_multiclass_to_tables` with these two weight vectors:

| Weights | Reported path losses | Selected supports |
| --- | --- | --- |
| `[1,1,1,1]` | `[0.9478309, 0.8369084]` | Both main effects |
| `[100,1,100,1]` | `[0.1750752, 0.1831211]` | Both mains and their pair |

The change scales every weight in one fold by the same constant, leaving that
fold's actual weighted mean loss unchanged. Under the existing equal-fold mean
policy, the selected path should consequently be unchanged.

**Suggested fix.** Normalize exactly once and make the fold aggregation policy
explicit. Preserve the intended interpretation of loss-based penalties and SE
rules when correcting the loss scale.

**Acceptance checks.** Pin fold-local weight-scale invariance for the equal-fold
policy, test unequal fold sizes and weight totals, and check report losses against
a direct weighted cross-entropy calculation.

<a id="bug-008"></a>
## BUG-008 — Binary pruning guard uses a log-link intercept correction

**Priority:** P2. **Status:** Open.

**Source:** [_guard_reanchor, sklearn.py:836](../../python/t_boost/sklearn.py#L836).
Compare the logit bisection at
[engine/boost.rs:5316-5331](../../crates/t-boost-core/src/engine/boost.rs#L5316),
reached by the native pruning reanchor path (`prune.rs:1681-1683` and
`1809-1812` dispatch `Link::Logit` to it; `prune::reanchor_shift` at
`prune.rs:3216` is the log formula and is only dispatched for `Link::Log`).
The `_guard_reanchor` docstring (`sklearn.py:817-818`) claims to mirror
`prune::reanchor_shift` "exactly" — true for the log link, but the native
deployment path never uses that function for logistic models.

**Trigger and impact.** The binary OOB pruning guard reanchors logistic scores.
It evaluates predictions calibrated differently from the native deployed model,
which can affect the guard's decision to re-admit tables.

**Cause.** The Python helper adds `log(sum(w*y) / sum(w*mu))` for every objective.
That adjustment is appropriate for a multiplicative log-link intercept, but it
does not solve the logistic intercept equation. The native logit implementation
instead solves the sigmoid balance equation using bisection.

**Reproduction.** Independent of the shared estimator fixture:

```python
import numpy as np
from t_boost.sklearn import _guard_mu, _guard_reanchor

raw = np.zeros(100)
y = np.r_[np.ones(10), np.zeros(90)]
shifted = _guard_reanchor("logistic", raw, y, np.ones(100), True)
print(_guard_mu("logistic", shifted).mean())  # 0.1666666667
print(y.mean())                              # 0.1
```

The helper adds `-1.6094379`; the correct constant logistic intercept is
`log(0.1 / 0.9) = -2.1972246`. The helper mismatch was directly reproduced. A
complete fit demonstrating a different final keep-set was not constructed.

**Suggested fix.** Use objective-specific intercept correction consistent with
the native deployment path. Prefer sharing the implementation over maintaining
two independent formulas.

**Acceptance checks.** Compare guard and native shifts for weighted binary data,
imbalanced outcomes, constant and varying logits, and boundary prevalence. Keep
the disabled-reanchor path unchanged. Add an artifact-versus-guard comparison.

<a id="bug-009"></a>
## BUG-009 — Clipped component relativities fail to reconstruct predictions

**Priority:** P2. **Status:** Open.

**Source:** [serialize.rs:1351](../../crates/t-boost-core/src/serialize.rs#L1351).
`RatingTable::relativities` is documented as `exp(value)` around line 1046.

**Trigger and impact.** A log-link model contains large positive or negative
component effects that cancel in the total score. Multiplying the exported
relativities can give a materially different rate from serving the model,
undermining rating-table deployment and auditability.

**Cause.** Export calculates `exp(clamp(effect, -30, 30))` separately for each
component. Clipping individual effects is not equivalent to applying the serving
inverse link to their sum, even when the final total lies well inside the clamp.

**Reproduction.** Start from `fixture_model()`, set its intercept to `-35`, set
the first tree's weight to `20`, and set its link and objective metadata to
Poisson/log. Build a `TableModel` through `from_model` with `fixture_serve()` and
`RefMeasure::Uniform`; the resulting purified model passes validation. For the
row whose merged cells are `[1,2]`:

```text
served prediction = 148.41316
exp(export.f0) * product(selected exported relativities) = 0.57375342
```

This reproduction uses a valid purified bank, not manually inconsistent table
values.

**Suggested fix.** Preserve exact component values in the multiplicative export
and define where any final response clamp belongs. If component exponentiation
cannot be represented safely, return an explicit export error or a suitable
representation instead of silently clipping individual factors.

**Acceptance checks.** Include large cancelling effects, basis rebasing, small
ordinary effects, and final scores near serving clamp boundaries. Compare
export-based scoring with the deployed model within its documented tolerance.

<a id="bug-010"></a>
## BUG-010 — Pre-v7 table JSON fails the version gate

**Priority:** P2. **Status:** Open.

**Source:** [decode_tables_json, serialize.rs:708](../../crates/t-boost-core/src/serialize.rs#L708)
and [multiclass table JSON decoding, line 874](../../crates/t-boost-core/src/serialize.rs#L874).
The intended compatibility is stated in the
[v7 schema comment, line 86](../../crates/t-boost-core/src/serialize.rs#L86).

**Trigger and impact.** Load an older JSON table model after upgrading. The model
is rejected even though the new optional fields can be defaulted and the code
explicitly intends to retain JSON read compatibility.

**Cause.** Deserialization supplies the absent `AxisId.band_of` using
`#[serde(default)]`, then `validate_tables_doc` compares the old schema version
against `required_tables_version`, which unconditionally requires at least v7.
The existing `migrate` facade handles tree `ModelDoc`, not table documents.
There is a second, independent gate: `TableModel::validate`
(`table_model.rs:108-120`) rejects a model-level `schema_version` outside
`7..=7`, so a document whose envelope stamp is raised to 7 but whose model
stamp is still 2 is also rejected ("tables model schema_version 2 outside
7..=7"). Both gates must defer to JSON defaulting. (Verification 2026-10-02:
stripping `band_of` while leaving both stamps at 7 loads correctly, confirming
the `#[serde(default)]` path itself works; `serialize.rs:89-90` literally
promises that "`#[serde(default)]` loads a pre-v7 document".)

**Reproduction.** Serialize a simple unbanded fixture table model to JSON. Set
its document/model schema versions to `2` and remove `band_of` from its axes,
producing the corresponding pre-v7 document shape. Loading reports:

```text
tables schema_version 2 is below the 7 its contents require
```

The scalar rejection was reproduced directly; the multiclass table decoder uses
the same incompatible validation pattern.

**Suggested fix.** Apply JSON-specific migration/defaulting before the version
gate, while retaining rejection of truly unsupported future versions and features.
Do not casually relax binary validation: pre-v7 table binary incompatibility is
explicitly documented and is a separate format constraint.

**Acceptance checks.** Store representative pre-v7 scalar and multiclass JSON
fixtures and verify prediction preservation on load. Also retain tests that
reject future versions, mislabeled newer features, and unsupported old binaries.

<a id="bug-011"></a>
## BUG-011 — Feature names can change JSON decoder selection

**Priority:** P2. **Status:** Open.

**Source:** [_unpack_json, sklearn.py:1397](../../python/t_boost/sklearn.py#L1397).
Related loader branches also detect table formats by searching the payload text.

**Trigger and impact.** An otherwise ordinary model contains a feature name or
other serialized string equal to a model-kind marker. Its own JSON output can
then be dispatched to the wrong loader and fail to reload.

**Cause.** The loader searches for quoted strings such as
`"t-boost-multiclass"` anywhere in the inner JSON rather than reading the actual
top-level discriminator.

**Reproduction.** Independent of the shared fixture:

```python
import polars as pl
from t_boost import TBoostClassifier

frame = pl.DataFrame({"t-boost-multiclass": list(range(20))})
m = TBoostClassifier(
    n_trees=1, n_bags=1, prune=False, validation_fraction=None, n_jobs=1
).fit(frame, [0, 1] * 10)
TBoostClassifier.from_json(m.to_json())
```

Observed: `SerializationError: missing field 'classes'` from the multiclass
decoder. The corresponding binary round-trip succeeds and preserves predictions.
The quoted substring test is at `sklearn.py:1399` within `_unpack_json`; the
legacy branch at lines 1402–1405 uses the same pattern. A feature named
`"t-boost-multiclass-tables"` misroutes the same way; `"t-boost-tables"` is
inert in this build because every model is a tables model. The regressor
loader ignores the kind marker and is unaffected.

**Suggested fix.** Parse the discriminator as a structured JSON field. Preserve
the existing policy of trusting the inner document over stale outer metadata,
but apply it only to the actual inner kind field.

**Acceptance checks.** Round-trip each model family with every format marker
appearing in feature names, class labels, and category labels. Test both legacy
raw JSON and estimator envelopes, including stale outer kind metadata.

<a id="bug-012"></a>
## BUG-012 — Unbounded factored order panics during model loading

**Priority:** P2. **Status:** Open.

**Source:** [FactoredEffect::validate_shape, explain.rs:2682](../../crates/t-boost-core/src/explain.rs#L2682).

**Trigger and impact.** A malformed serialized factored effect has an excessively
large feature-set order. Loading panics instead of returning a typed error,
violating the library's no-panic input boundary. The reproduction establishes a
Rust panic; it does not establish an unconditional process abort in every caller.

**Cause.** Deserialization can create arbitrarily long feature sets. After checking
axis/vector counts, validation computes `1usize << k` without bounding `k` by the
supported maximum interaction order or using a checked shift.

**Reproduction.** Start from a fixture table JSON document and insert a factored
effect with 64 feature IDs, 64 individually valid axis records, 64 correctly sized
reference-weight vectors, and `boxes: []`. On the reviewed 64-bit release build,
`TableModel::from_json` panics with:

```text
attempt to shift left with overflow
```

The panic was caught with `catch_unwind`. The document is intentionally malformed
and should be rejected before evaluating order-dependent arithmetic. Orders 65
and 200 panic identically. Orders 9 and 63 — both above `MAX_ORDER = 8`, with
axis `raw` identities beyond the two-feature model — load *without error*;
nothing on the load path bounds the order or checks axis identities, so 64 is
merely the first order that overflows. `FactoredEffect::packed`
(`explain.rs:2615`) already uses `checked_shl` for the same quantity.

**Suggested fix.** Validate the supported order before shifting or allocating
order-dependent structures. Also validate feature-set/axis relationships, and
retain checked arithmetic at deserialization boundaries.

**Acceptance checks.** Reject orders above the structural maximum, including
values at and above the platform word width, through `PbError`. Cover JSON and
binary loaders and add the malformed cases to the deserialization fuzz corpus.

<a id="bug-013"></a>
## BUG-013 — Nonfinite factored coefficients pass validation

**Priority:** P2. **Status:** Open.

**Source:** [table_model.rs:240](../../crates/t-boost-core/src/table_model.rs#L240)
and [FactoredEffect::validate_shape, explain.rs:2671](../../crates/t-boost-core/src/explain.rs#L2671).

**Trigger and impact.** A binary model contains a nonfinite factored coefficient.
Loading and explicit validation both succeed, allowing `NaN` predictions to flow
out of an apparently valid model.

**Cause.** Factored-effect validation checks variance, axes, and vector lengths,
but not the numerical validity of box corners or reference weights. Dense table
tensors receive numerical validation that factored payloads do not.

**Reproduction.** Construct a valid three-feature, depth-three tree from the
fixture model, with leaves:

```text
[0.125, 1.25, 2.75, -3.875, 4.5, 5.125, 6.875, 8.625]
```

Score all eight finite-bin combinations and build its exact table model under
`RefMeasure::Uniform`. The generated bank contains one factored effect and passes
validation. Replace only the eight serialized bytes of its first box corner with
the bytes for `f64::NAN`; preserve every other byte.

```text
before: [8.625, 6.875, 5.125, 4.5, -3.875, 2.75, 1.25, 0.125]
from_bincode: Ok
validate: Ok
after:  [8.625, 6.875, 5.125, 4.5, -3.875, 2.75, 1.25, NaN]
```

This strengthened reproduction begins with a normally constructed valid bank;
the sole corruption is the nonfinite coefficient.

**Suggested fix.** Validate all factored numerical payloads where they are
accessible, including finite box coefficients and the required constraints on
reference weights. Propagate the resulting typed error through every load path.

**Acceptance checks.** Test `NaN` and both infinities in box coefficients and
invalid reference-weight entries. Valid factored models must retain identical
predictions and round-trip behavior. Add binary corruption seeds to fuzzing.

<a id="bug-014"></a>
## BUG-014 — Empty bagged fits panic

**Priority:** P2. **Status:** Open.

**Source:** [scalar outer bags, boost.rs:2790](../../crates/t-boost-core/src/engine/boost.rs#L2790)
and [multiclass bags, line 1494](../../crates/t-boost-core/src/engine/boost.rs#L1494).

**Trigger and impact.** Fit a zero-row dataset with two or more outer bags. Both
public core fitting APIs panic rather than returning an input-validation error.

**Cause.** Ensemble validation checks shapes but does not reject an empty dataset
before calculating a draw size with `.clamp(1, n_rows)`. For zero rows this has an
invalid minimum/maximum pair.

**Reproduction.** Create a zero-row binned matrix with
`bin_columns(&[&[]], None, &BinConfig::default(), 0)`, an empty target, and:

```rust
EnsembleSpec::OuterBag {
    n_bags: 2,
    bag_subsample: 0.8,
    cell_refit: None,
}
```

With an otherwise default `Config` and squared-error `FitSpec`, both
`Booster::fit` and `Booster::fit_multiclass` panic. The multiclass call uses three
class labels. Observed panic:

```text
min > max. min = 1, max = 0
```

**Suggested fix.** Reject empty training data before bag-size arithmetic, at a
validation boundary shared by scalar and multiclass ensemble fitting.

**Acceptance checks.** Exercise empty datasets for zero, one, and multiple bags,
with and without groups/holdouts, through both Rust and Python surfaces. Confirm
typed errors rather than panics while preserving valid singleton behavior.

<a id="bug-015"></a>
## BUG-015 — Merged parameter aliases fail through `set_params`

**Priority:** P2. **Status:** Open.

**Source:** [set_params, sklearn.py:2201](../../python/t_boost/sklearn.py#L2201);
constructor-only alias resolution is immediately above it, around lines 2165–2199.

**Trigger and impact.** Set merged parameter names after construction, including
through a parameter-search interface. Fits use stale effective values, and
cloning can fail. A search may therefore ignore candidate settings or error.

**Cause.** The constructor resolves `early_stopping` and `prune_size_penalty` onto
legacy attributes. `set_params` only assigns the new attributes and clears fitted
state; it does not repeat resolution. `get_params` then exposes an inconsistent
pair of alias and effective values.

**Reproduction.** With scikit-learn available:

```python
from sklearn.base import clone
from t_boost import TBoostRegressor

a = TBoostRegressor(early_stopping=20)
b = TBoostRegressor().set_params(early_stopping=20)
print(a.early_stopping_rounds, b.early_stopping_rounds)  # 20, 500
clone(b)  # RuntimeError: constructor modifies early_stopping_rounds

a = TBoostRegressor(prune_size_penalty=3.0)
b = TBoostRegressor().set_params(prune_size_penalty=3.0)
print(a.prune_lambda_tables, b.prune_lambda_tables)     # 3.0, 0.0
clone(b)  # RuntimeError: constructor modifies prune_lambda_tables
```

**Suggested fix.** Centralize alias resolution and conflict handling for every
parameter mutation path. Preserve clone-compatible constructor semantics and
define what happens when aliases or legacy spellings are subsequently reset.

**Acceptance checks.** Compare constructor and `set_params` behavior for integer
and float early-stopping forms, penalty aliases, conflicting values, and resets
to `None`. Verify cloning, representative parameter searches, effective native
configuration, and model equivalence for equivalent spellings.

<a id="bug-016"></a>
## BUG-016 — Objective aliases produce wrong explanation and reporting semantics

**Priority:** P2. **Status:** Open.

**Source:** [link property, sklearn.py:5312](../../python/t_boost/sklearn.py#L5312)
and [_expected_response, line 6418](../../python/t_boost/sklearn.py#L6418).
Compare the native objective parser in
[lib.rs:197](../../crates/t-boost-py/src/lib.rs#L197).

**Trigger and impact.** Use an accepted objective alias, alternative case, or
hyphenated spelling. Training succeeds, but explanations can fail validation or
report the wrong response transformation, and A/E expected totals can omit
exposure scaling.

**Cause.** The native parser canonicalizes names and accepts aliases. Python
reporting branches compare the original literal `self.objective` against only
canonical lowercase spellings. Further literal comparisons not cited above sit
at `sklearn.py:5478` (objective family) and `5555` (classifier predict branch).
The Python layer does normalize at fit time (`3288`, `3811`, `7871`), so alias
acceptance is deliberate; the reporting branches simply never see the
canonical form. The defect survives serialization: `_restore_metadata`
(`1331-1332`) rewrites the raw objective literal over the canonical value that
`_attach_model` (`5217`) derived, so a loaded alias-fit model still misreports
its link. The case variant `"SQUARED_ERROR"` fails in the same way.

**Reproduction.** Using the shared setup, fit
`TBoostRegressor(objective=name, **OPTIONS)` on `(X, X[:, 0] + 1)` for each of
`"squared-error"`, `"l2"`, and `"regression"`. Each reports `link == "log"`
instead of `"identity"`; `predict_contributions(X[:1])` fails its prediction
identity check. A classifier with `objective="Logistic"` similarly reports
`"log"` instead of `"logit"`.

For the exposure case:

```python
exposure = np.full(80, 2.0, dtype=np.float32)
y = np.ones(80, dtype=np.float32)
m = TBoostRegressor(objective="POISSON", **OPTIONS).fit(X, y, exposure=exposure)
report = m.actual_vs_expected(X, y, exposure=exposure)[0]
print(sum(report["expected"]))              # 40.0
print(np.sum(m.predict(X) * exposure))       # 80.0
```

**Suggested fix.** Resolve the fitted objective canonically once and use that
resolved identity or native model metadata for response semantics. Keep raw
constructor parameters separate if necessary for estimator cloning.

**Acceptance checks.** Parameterize all supported aliases/case variants across
prediction, link reporting, contributions, exposure totals, and save/load. Audit
other Python objective-dependent branches for the same literal-comparison issue.

<a id="bug-017"></a>
## BUG-017 — Binary A/E reports sum class labels

**Priority:** P2. **Status:** Open.

**Source:** [actual_vs_expected, sklearn.py:3158](../../python/t_boost/sklearn.py#L3158)
and [actual aggregation, line 3184](../../python/t_boost/sklearn.py#L3184).

**Trigger and impact.** A binary classifier is trained with labels other than
literal `0/1`. A/E and the dependent pricing report compute incorrect actual
totals for numeric labels, or fail for otherwise valid string labels.

**Cause.** Fit maps labels to the positive-class indicator, but reporting casts
the original supplied labels to floating point and sums them. Expected values
are positive-class probabilities, so actual and expected are on different scales.

**Reproduction.** Using the shared Python setup:

```python
y = np.repeat([1, 2], 40)
m = TBoostClassifier(**OPTIONS).fit(X, y)
report = m.actual_vs_expected(X, y)[0]
print(m.classes_)                          # [1, 2]
print(np.count_nonzero(y == m.classes_[1])) # 40
print(sum(report["actual"]))              # 120.0
print(sum(report["expected"]))            # approximately 40.0
```

Fitting and reporting with `np.repeat(["a", "b"], 40)` instead raises a string
to float conversion error.

**Suggested fix.** Validate observed labels against the fitted class set, then
convert them to the indicator for `classes_[1]` before weighted aggregation.
Retain an explicit contract for unsupported multiclass A/E behavior.

**Acceptance checks.** Cover `0/1`, `1/2`, negative numeric, boolean, and string
labels; sample weights; unknown labels; named target columns; loaded estimators;
and the dependent `pricing_report` path.

<a id="bug-018"></a>
## BUG-018 — Deployed factored effects are reported as absent

**Priority:** P2. **Status:** Open.

**Source:** [_TableModel.deployed_supports, lib.rs:5632](../../crates/t-boost-py/src/lib.rs#L5632)
and [report reconciliation, sklearn.py:4579](../../python/t_boost/sklearn.py#L4579).

**Trigger and impact.** A scalar fold-vote pruning fit deploys factored effects.
The report omits these effects from `deployed` and can list them under
`kept_not_deployed`, giving a false account of the model that is actually served.

**Cause.** `deployed_supports` iterates only `bank.tables`, excluding
`bank.factored`. Python interprets the result as the full deployed support set.
Existing test comments acknowledge the dense-only count; they do not make the
`kept_not_deployed` classification accurate. The binding is internally
inconsistent: `deployed_census()` (`lib.rs:5622-5624`) explicitly includes
`factored` and returns `{1: 4, 2: 5, 3: 1}` for the model below, counting the
effect that `deployed_supports()` omits. The two test files disagree too:
`test_surface_merges_and_binding_report.py:176-177` records that `deployed`
"counts DENSE tables only", while `test_prune.py:48-53` asserts that
"deployed lists what actually survived purification into the bank, and kept \
deployed is surfaced, never silent" — the guarantee the factored support
violates.

**Reproduction.** Independent Python fixture:

```python
import json
import numpy as np
from t_boost import TBoostRegressor

rng = np.random.default_rng(5)
x = rng.normal(size=(300, 4)).astype(np.float32)
y = (
    2 * np.sign(x[:, 0]) * np.sign(x[:, 1]) * np.sign(x[:, 2])
    + 0.7 * x[:, 3] + 0.1 * rng.normal(size=300)
)
m = TBoostRegressor(
    n_trees=10, n_bags=1, max_depth=6, interaction_gain_hurdle=0.0,
    validation_fraction=None, n_jobs=1, prune_selector="fold_vote",
    band_tolerance=None, graduate=False,
).fit(x, y)
bank = json.loads(m._model.to_json())["model"]["bank"]
print([(f["u"], len(f["boxes"])) for f in bank["factored"]])
print(m.pruning_report_["deployed"])
print(m.pruning_report_["kept_not_deployed"])
```

Observed: support `[0,2,3]` has 64 deployed boxes but is missing from `deployed`
and appears in `kept_not_deployed`. Nine dense effects are reported separately.

**Suggested fix.** Enumerate the union of dense/sparse table supports and factored
supports, with deterministic ordering and deduplication as appropriate. Reconcile
the report against that complete set.

**Acceptance checks.** Assert that report supports match the serialized artifact
for dense-only, factored-only, and mixed banks. Ensure a genuinely absent selected
support can still appear in `kept_not_deployed` and that count APIs stay consistent.

<a id="bug-019"></a>
## BUG-019 — Fuzz workflow uses the wrong dictionary path

**Priority:** P2. **Status:** Open.

**Source:** [.github/workflows/fuzz.yml:30](../../.github/workflows/fuzz.yml#L30).
The dictionary is [fuzz/fuzz_targets/fuzz_deserialize.dict](../../fuzz/fuzz_targets/fuzz_deserialize.dict).

**Trigger and impact.** The nightly/manual fuzz workflow runs from the repository
root. Its deserialization fuzz command cannot locate the dictionary and fails
before meaningful fuzzing. The following binning-fuzzer step is also skipped by
the failed job step.

**Cause.** The command passes:

```sh
cargo +nightly fuzz run fuzz_deserialize -- -max_total_time=300 -dict=fuzz_targets/fuzz_deserialize.dict
```

The pinned cargo-fuzz implementation forwards target arguments unchanged through
`cargo run --manifest-path ...`; it does not change the process directory to
`fuzz/`. The supplied relative path therefore lacks its `fuzz/` prefix.

**Evidence and limits.** The path mismatch and command forwarding were checked
against the [cargo-fuzz 0.13.1 implementation](https://raw.githubusercontent.com/rust-fuzz/cargo-fuzz/0.13.1/src/project.rs)
and [Cargo's working-directory contract](https://doc.rust-lang.org/cargo/commands/cargo-run.html).
The full workflow was not executed locally because cargo-fuzz was unavailable.

Verification 2026-10-02 traced cargo-fuzz 0.13.1 `src/project.rs` directly:
`find_package` walks up from the current directory to the first non-fuzz
`Cargo.toml` (the repository root), `cargo()` builds `cargo run
--manifest-path <root>/fuzz/Cargo.toml`, `exec_fuzz` appends the target
arguments verbatim and spawns; no `current_dir` call exists on any `Command`.
libFuzzer resolves `-dict=` in the target process and exits 1 when the file is
missing. The workflow pins `cargo install cargo-fuzz --version 0.13.1 --locked`
and sets no `working-directory`. **The defect is currently latent:** the only
recorded run of `fuzz.yml` (run `36960895029`, 2026-10-02, 24 s) fails one
step earlier, inside the install itself — the `--locked` 0.13.1 dependency set
pins `rustix 0.36.5`, which no longer compiles on current nightly
(`cannot find attribute rustc_layout_scalar_valid_range_start`). The install
must be repaired before the dictionary path can be exercised. The local smoke
commands in `fuzz/README.md` do not pass `-dict`, so nothing exercises the path
locally either.

**Suggested fix.** Use the repository-relative dictionary path or set a working
directory and make all associated paths consistent with it.

**Acceptance checks.** From a clean checkout, run the exact workflow command for
a short smoke duration and confirm the dictionary loads. Then verify both fuzz
targets run. Include a cheap dictionary existence check before the expensive
build/run steps if useful for diagnosing future path drift.

<a id="bug-020"></a>
## BUG-020 — Release Gate can smoke-test a published package instead of its artifact

**Priority:** P2. **Status:** Open.

**Source:** [.github/workflows/release-gate.yml:212](../../.github/workflows/release-gate.yml#L212).

**Trigger and impact.** Run the manual Release Gate against a revision whose
locally built version is older than an available PyPI release. The smoke test can
install the published release, pass, and leave the built artifact untested.

**Cause.** The fresh environment installs:

```sh
python -m pip install --find-links dist t-boost
```

`--find-links` adds a candidate location; it does not give that location priority
over the index. An unpinned requirement can select a newer indexed version. The
smoke code does not assert the installed version or the origin of the artifact.

**Evidence and limits.** This is a conditional workflow defect established from
the command and [pip's documented candidate selection](https://pip.pypa.io/en/stable/cli/pip_install/#finding-packages).
No publish action or GitHub workflow was run. The finding concerns
`release-gate.yml`; the main `release.yml` has separate version checks and is not
the reported unpinned install path.

Verification 2026-10-02: the step at line 212 has no version pin, no
`--no-index` and no `--only-binary`; the inline heredoc smoke (lines 214–226)
never asserts `__version__`, `__file__` or the build profile.
`scripts/package_smoke_check.py`, which asserts all three, is used only by
`release.yml:286`. `release.yml:278` installs `"t-boost==${VERSION}"` with
`--only-binary`/`--no-binary` from `$RUNNER_TEMP`, as the entry says. pip's
documentation states verbatim that "there is no priority in the locations that
are searched … the 'best' match … (in terms of version number)" is selected.
**The condition is already live:** the workspace version is `0.6.1` and PyPI
serves `0.6.1`, so for the current revision the index offers an *equal*
version and which `0.6.1` the gate installs is unspecified by the quoted rule.
The entry's "older local version" trigger covers only pre-bump revisions.

**Suggested fix.** Install the exact built wheel path for the current platform,
then verify the installed package's version and import location. Runtime
dependencies can still be resolved normally; selecting the project artifact
should not depend on index version ordering.

**Acceptance checks.** Exercise a fixture where the index offers a newer version
than the local artifact and confirm the local artifact is installed. Fail clearly
when no suitable built wheel exists. Check platform-specific wheel selection.

<a id="bug-021"></a>
## BUG-021 — Multiline derives bypass serialized-field gates

**Priority:** P3. **Status:** Open.

**Source:** [xtask/src/main.rs:1170](../../xtask/src/main.rs#L1170).

**Trigger and impact.** A serialized Rust type uses a multiline derive. The
`check-no-usize-serialized` and `check-no-hashmap-serialized` gates fail to inspect
its fields, allowing types that violate the repository's wire-width and
deterministic-container rules. This is a safeguard failure, not evidence that an
existing shipped field of either forbidden type was found.

**Cause.** The scanner recognizes a serialization attribute only if `#[` and
`Serialize` or `Deserialize` occur on the same line. Attribute continuation lines
do not maintain the required pending state. Normal formatting can introduce this
layout; multiline derives already occur in the codebase.

**Reproduction.** Place the following source under
`crates/example/src/lib.rs` in a temporary directory and run the already-built
`xtask check-all` binary from that directory:

```rust
#[derive(
    Serialize, Deserialize,
)]
pub struct Bad {
    pub n: usize,
    pub m: HashMap<u32, u32>,
}
```

The scanner does not compile this fixture; it reads its source. All four gates
report `[ok]` and the process exits `0`. Putting the identical derive on one
line makes the process exit `1` and report both forbidden fields.

**Suggested fix.** Parse complete attributes, including multiline forms, before
associating serialization derives with their items. Prefer a syntax-aware
approach or a well-tested tokenizer over independent line matching.

**Acceptance checks.** Pair single-line and multiline fixtures for structs,
enums, tuple structs, intervening attributes, qualified derive names, and comments.
Cover all four forbidden field types (`usize`, `isize`, `HashMap`, `HashSet`) and
allowed fixed-width/ordered alternatives. Both formatting forms must flag the
same forbidden fields without flagging nonserialized types, and existing
test-module exclusions must continue to work.

<a id="bug-022"></a>
## BUG-022 — Forked workers inherit a serving pool whose threads no longer exist

**Priority:** P2. **Status:** Open.

**Source:** [run_on_pool, lib.rs:7037](../../crates/t-boost-py/src/lib.rs#L7037) and
[serve_pool, line 7054](../../crates/t-boost-py/src/lib.rs#L7054).

**Trigger and impact.** On a platform supporting `multiprocessing`'s `fork` start
method, score a model in the parent and then score it in a forked child using the same
explicit `n_jobs`. The child hangs instead of returning a prediction. This affects
applications that load/warm models before forking workers, even when no prediction is in
progress at the instant of the fork.

**Cause.** `serve_pool` keeps Rayon pools in a process-global static map keyed only by
width. A fork copies this map and its pool bookkeeping but does not copy the worker
threads. The child retrieves the inherited pool and calls `install` on it. The cache has
no process identity check or child-process reset policy.

**Reproduction.** Save the following as a script and run it on Linux. The main guard and
an importable file are required for the `spawn` control:

```python
import multiprocessing as mp
import numpy as np
from t_boost import TBoostRegressor

def predict_child(model, x, queue, width):
    model.n_jobs = width
    queue.put(model.predict(x).tolist())

if __name__ == "__main__":
    x = np.arange(80, dtype=np.float32).reshape(-1, 1)
    m = TBoostRegressor(
        n_trees=4, n_bags=1, prune=False, graduate=False,
        band_tolerance=None, validation_fraction=None, n_jobs=2,
    ).fit(x, x[:, 0])
    print("parent", m.predict(x[:2]))
    for method, width in [("fork", 2), ("fork", 3), ("spawn", 2)]:
        ctx = mp.get_context(method)
        queue = ctx.Queue()
        child = ctx.Process(target=predict_child, args=(m, x[:2], queue, width))
        child.start()
        child.join(5)
        print(method, width, "still running:", child.is_alive())
        if child.is_alive():
            child.terminate()
            child.join(2)
        else:
            print(queue.get(timeout=1))
        queue.close()
```

Observed: `fork, 2` remained alive after the five-second timeout. Both `fork, 3` and
`spawn, 2` completed successfully and returned `[20.23057746887207, 20.23057746887207]`.
The different-width control creates a fresh local pool in the child and isolates the
inherited-cache problem. It is evidence for this serving path, not a claim that every
post-fork native operation is safe with a different width.

Verification 2026-10-02 reproduced the three results exactly (Python 3.11) and added
a control: with `n_jobs=None` the forked child returns for a two-row query but **also
hangs** for a parallelism-sized query (10,000 × 80 rows, 16 trees), so the inherited
process-global Rayon pool (`cap_global_pool_once`, `lib.rs:7021`) is equally dead; the
suggested-fix note about the global pool is warranted. Nothing in README, CONTRIBUTING
or any docstring mentions fork or `multiprocessing` start methods. Forking after
threads exist is unsupported by Rayon in general (and deprecated by CPython 3.12+),
so ownership is arguable, but a silent indefinite hang with no documented restriction
is still a defect in this library's surface.

**Suggested fix.** Give native worker-pool state an explicit process lifecycle. Detect
inherited state before accessing it and either initialize child-safe state or raise a
clear error directing callers to a supported process start method. Account for the
separate global Rayon pool and inherited locks; simply clearing a map after arbitrary
forks is not a sufficient general solution.

**Acceptance checks.** Run bounded subprocess tests for prediction before and after
fork, explicit cached widths, fresh widths, and `spawn`. No supported path may wait
indefinitely. Unsupported combinations must fail promptly and clearly.

<a id="bug-023"></a>
## BUG-023 — Multiclass prediction ignores the estimator's thread budget

**Priority:** P2. **Status:** Open.

**Source:** [predict_proba, sklearn.py:8214](../../python/t_boost/sklearn.py#L8214),
[decision_function, line 8245](../../python/t_boost/sklearn.py#L8245), and [multiclass
table prediction, lib.rs:6700](../../crates/t-boost-py/src/lib.rs#L6700). The public
[n_jobs documentation](../../python/t_boost/sklearn.py#L5961) promises the setting
applies to both fitting and prediction.

**Trigger and impact.** Use a multiclass classifier in a process whose global Rayon pool
has a different width from that estimator's `n_jobs`. Prediction uses the ambient pool
rather than the requested budget. Multiple estimators/workers can oversubscribe the
machine; conversely, an earlier narrow global pool can prevent a later prediction from
using its requested parallelism.

**Cause.** The binary branches resolve and pass `n_jobs`; the multiclass branches do
neither. Native `_MultiClassModel` and `_MultiClassTableModel` prediction methods do not
accept the argument or enter `run_on_pool`. Per-class table scoring still executes
parallel work, so omission does not make prediction serial. The process-global pool's
width depends on earlier calls.

**Reproduction.** This Linux script measures CPU time for each native thread during
repeated prediction. Run in a fresh process so the initial fit can set the global pool
to four workers:

```python
from pathlib import Path
import numpy as np
from t_boost import TBoostClassifier

def ticks():
    return {
        p.name: sum(map(int, (p / "stat").read_text().split()[13:15]))
        for p in Path("/proc/self/task").iterdir()
    }

x = np.arange(180, dtype=np.float32).reshape(-1, 1)
m = TBoostClassifier(
    n_trees=8, n_bags=1, prune=False, graduate=False,
    band_tolerance=None, validation_fraction=None, n_jobs=4,
).fit(x, np.repeat(np.arange(3), 60))
m.n_jobs = 1
query = np.tile(x, (10000, 1))
m.predict_proba(query)
before = ticks()
for _ in range(12):
    m.predict_proba(query)
after = ticks()
print({k: after[k] - before.get(k, 0)
       for k in after if after[k] > before.get(k, 0)})
```

Observed positive CPU tick deltas were `89` on the calling thread and `25, 24, 25, 24`
on four worker threads, despite `n_jobs=1`. Exact timings are machine-dependent. Setting
`m.n_jobs=0` afterward also allowed multiclass prediction to complete, confirming that
the normal prediction-time thread validation is skipped.

**Suggested fix.** Thread the resolved job budget through both multiclass container
types and install the same cached serving pool used by scalar prediction. Apply it to
probabilities and raw logits so `predict` inherits the same behavior.

**Acceptance checks.** Verify the executing pool width with instrumentation, including
differently configured estimators in one process and changed serving budgets. Check
invalid counts and confirm that probabilities/raw scores remain byte-identical across
supported thread counts.

<a id="bug-024"></a>
## BUG-024 — Categorical identities change across containers and optional pandas availability

**Priority:** P2. **Status:** Open.

**Source:** [python/t_boost/_ingest.py:30](../../python/t_boost/_ingest.py#L30);
[python/t_boost/sklearn.py:1480](../../python/t_boost/sklearn.py#L1480);
[python/t_boost/sklearn.py:1678](../../python/t_boost/sklearn.py#L1678);
[python/t_boost/_ingest.py:225](../../python/t_boost/_ingest.py#L225).

**Cause.** NumPy/polars materialize numeric category values as Python scalars before
`str`, while pandas passes NumPy float32 scalars. `str(np.float32(0.1)) == '0.1'` but
the identical float32 value converted to a Python float becomes `'0.10000000149011612'`.
The encoding has no common scalar normalization. Missing float32 NumPy scalars have
another consequence: `_cat_level`'s dependency-free NaN check accepts Python floats but
not np.float32; pandas' optional `isna` fallback is the only thing identifying these as
missing when they reside in an object array.

**Trigger and impact.** Changing the container of otherwise identical supported inputs
silently sends known categories down the unseen-category fallback. Moving a serialized
model from a development environment with pandas to a supported runtime environment
without pandas can likewise change missing-value predictions.

**Reproduction A.**

```python
import warnings
import numpy as np
import pandas as pd
import polars as pl
from t_boost import TBoostRegressor
warnings.simplefilter('ignore')
opts = dict(n_trees=20, n_bags=1, prune=False, validation_fraction=None,
            n_jobs=1, min_data_in_leaf=1, max_depth=3)
x = np.tile(np.array([0.1, 0.2], dtype=np.float32), 100).reshape(-1, 1)
y = np.tile(np.array([0., 10.], dtype=np.float32), 100)
frames = [('numpy', x), ('pandas', pd.DataFrame(x, columns=['f'])),
          ('polars', pl.DataFrame({'f': x[:, 0]}))]
for train_name, frame in frames:
    m = TBoostRegressor(categorical_features=[0], **opts).fit(frame, y)
    for serve_name, other in frames:
        print(train_name, serve_name, m.predict(other)[:2])
```

Observed:

```text
numpy numpy   [0.03118572 9.9688139]
numpy pandas  [0.03118572 0.03118572]
numpy polars  [0.03118572 9.9688139]
pandas numpy  [0.03118572 0.03118572]
pandas pandas [0.03118572 9.9688139]
pandas polars [0.03118572 0.03118572]
polars numpy  [0.03118572 9.9688139]
polars pandas [0.03118572 0.03118572]
polars polars [0.03118572 9.9688139]
```

**Reproduction B.** Run the following script first without an argument to fit/write,
then with any argument to load/serve without pandas:

```python
import sys, warnings
if len(sys.argv) > 1:
    sys.modules['pandas'] = None
import numpy as np
from t_boost import TBoostRegressor
from t_boost._ingest import _cat_level
warnings.simplefilter('ignore')
x = np.empty((300, 1), dtype=object)
x[:, 0] = [np.float32('nan'), 'a', 'b'] * 100
y = np.tile([0, 5, 10], 100).astype(np.float32)
path = '/tmp/tboost_deep_no_pandas.bin'
if len(sys.argv) > 1:
    m = TBoostRegressor.from_bytes(open(path, 'rb').read())
else:
    m = TBoostRegressor(categorical_features=[0], n_trees=20, n_bags=1,
        prune=False, validation_fraction=None, n_jobs=1).fit(x, y)
    open(path, 'wb').write(m.to_bytes())
print(_cat_level(np.float32('nan')), m.predict(x)[:3])
```

```text
pandas available: __t_boost_missing__ [0.03118572 5.0 9.9688139]
pandas absent:    nan                 [5.0 5.0 9.9688139]
```

**Suggested fix.** Canonicalize supported NumPy/Python scalar values before missing
detection and string formatting. Make missing detection for supported scalar types
independent of optional pandas. Ensure training and coded/uncoded serving share this
canonicalization; consider compatibility for models whose category maps used the
previous spelling.

**Acceptance checks.** Cross-container fit/serve matrix with explicit float32
categories, NaN held as np.float32 in object arrays, installed/blocked pandas
subprocesses, both small and factorization-sized batches, and bytes/JSON reload.
Predictions must agree for the same original category values. Preserve deliberate
distinctions such as strings versus numbers where required by the categorical API.

**Fourth-pass reproduction: prediction changes at the 64-row batch boundary.**

This strengthens the existing categorical-normalization defect: changing **batch size alone** within the same pandas container is sufficient.

**Source:** [sklearn.py:1610](../../python/t_boost/sklearn.py#L1610) switches off factorization below 64 rows; [sklearn.py:1615](../../python/t_boost/sklearn.py#L1615)–[sklearn.py:1616](../../python/t_boost/sklearn.py#L1616) treats all pandas Categorical dtypes as safe; `_factorized_labels` at [sklearn.py:1628](../../python/t_boost/sklearn.py#L1628) iterates categories as Python scalars, while the short path at [sklearn.py:1665](../../python/t_boost/sklearn.py#L1665)/[sklearn.py:1678](../../python/t_boost/sklearn.py#L1678) iterates NumPy float32 scalars. They stringify the same numeric value differently, as already documented in BUG-024.

Reproduction:

```python
import numpy as np
import pandas as pd
from t_boost import TBoostRegressor
x = pd.DataFrame({'cat': pd.Categorical(
    np.tile(np.array([.1, .2], dtype=np.float32), 100)
)})
y = np.tile([0., 10.], 100).astype(np.float32)
m = TBoostRegressor(categorical_features=[0], n_trees=10, n_bags=1,
    prune=False, validation_fraction=None, n_jobs=1).fit(x, y)
for n in [1, 63, 64, 100]:
    batch = pd.concat([x.iloc[[1]]] * n, ignore_index=True)
    print(n, m.predict(batch)[0])
```

```text
1   0.39487800002098083
63  0.39487800002098083
64  9.605121612548828
100 9.605121612548828
```

Every row in every batch is the same original float32 category 0.2. Bulk scoring versus individual scoring can therefore disagree even when the user never changes DataFrame type. Add this boundary case to BUG-024's acceptance checks.

<a id="bug-025"></a>
## BUG-025 — Classifier serialization corrupts large unsigned integer labels

**Priority:** P2. **Status:** Open.

**Source:** [python/t_boost/sklearn.py:1260](../../python/t_boost/sklearn.py#L1260);
[python/t_boost/sklearn.py:1348](../../python/t_boost/sklearn.py#L1348).

**Cause.** A uint64 class array containing both a small integer and an integer greater
than INT64_MAX becomes a mixed-range Python integer list. NumPy infers float64 on
reload, rounding the larger label. This affects both bytes and JSON envelopes. Native
string class labels remain exact, but `_restore_metadata` overwrites them with the lossy
inferred array.

**Trigger and impact.** Fitted models begin returning a different class identity after
save/load, not merely a different integer dtype.

```python
import warnings
import numpy as np
from t_boost import TBoostClassifier
warnings.simplefilter('ignore')
x = np.tile(np.array([0., 1.], dtype=np.float32), 100).reshape(-1, 1)
y = np.tile(np.array([1, 2**63 + 1], dtype=np.uint64), 100)
m = TBoostClassifier(n_trees=20, n_bags=1, prune=False,
    validation_fraction=None, n_jobs=1).fit(x, y)
for name, restored in [
    ('bytes', TBoostClassifier.from_bytes(m.to_bytes())),
    ('json', TBoostClassifier.from_json(m.to_json())),
]:
    print(name, m.classes_.dtype, restored.classes_.dtype,
          list(map(int, restored.classes_)))
    print(list(map(int, m.predict(x[:2]))), list(map(int, restored.predict(x[:2]))))
```

Both formats restore `float64`, with classes `[1, 9223372036854775808]` instead of
original `uint64` `[1, 9223372036854775809]`. The provided x makes predicted labels
visibly change too.

**Suggested fix.** Preserve class dtype using a versioned array representation or exact
typed-label encoding. For legacy envelopes infer a lossless integer/object
representation when the range cannot fit int64; never silently choose float64 for
integer labels.

**Acceptance checks.** Bytes/JSON/pickle round trips for binary and multiclass uint64
labels spanning the signed boundary (and adjacent large labels), asserting exact Python
integer values as well as prediction/class dtype where promised. Verify ordinary
numeric/string/bool labels and older envelopes.

<a id="bug-026"></a>
## BUG-026 — Runtime-only classifier accepts NaN and infinite class labels

**Priority:** P2. **Status:** Open.

**Source:** [python/t_boost/_compat.py:104](../../python/t_boost/_compat.py#L104);
[python/t_boost/sklearn.py:7887](../../python/t_boost/sklearn.py#L7887).

**Cause.** The no-scikit-learn target checker removes nonfinite values from its
continuous-label check, then calls the result multiclass. Fit derives classes from the
original invalid labels and encodes NaN/infinity as finite class indices, bypassing the
native target validation.

**Trigger and impact.** The supported default runtime (scikit-learn is optional) can
silently train a class corresponding to missing labels and return NaN/inf predictions.
Adding scikit-learn changes acceptance of the same inputs. Binary `[0, NaN]` happens to
fail later because there are no positive encoded labels; three-class NaN and
binary/infinite cases succeed.

**Reproduction.**

```python
import sys, warnings
sys.modules['sklearn'] = None
import numpy as np
from t_boost import TBoostClassifier
warnings.simplefilter('ignore')
for labels in [[0., np.inf], [0., 1., np.nan], [0., 1., np.inf]]:
    y = np.tile(labels, 100)
    x = np.tile(np.arange(len(labels), dtype=np.float32), 100).reshape(-1, 1)
    m = TBoostClassifier(n_trees=20, n_bags=1, prune=False,
        validation_fraction=None, n_jobs=1).fit(x, y)
    print(m.classes_, m.predict(x)[:len(labels)])
```

```text
[0. inf]     [0. inf]
[0. 1. nan]  [0. 1. nan]
[0. 1. inf]  [0. 1. inf]
```

**Suggested fix.** Validate target dimensionality, supported label types, and
missing/nonfinite numeric labels in common estimator code independent of optional
sklearn, or make the fallback faithfully implement the required validation. Reject
before mutating fitted metadata.

**Acceptance checks.** Separate subprocesses with sklearn available and blocked must
both reject NaN and ±inf in binary/multiclass targets, whether supplied as numpy arrays,
polars Series, or target columns; valid string, bool and integer classes remain
supported.

<a id="bug-027"></a>
## BUG-027 — Loaded estimators silently drop the unspecified half of table export mass

**Priority:** P2. **Status:** Open.

**Source:** [python/t_boost/sklearn.py:3004](../../python/t_boost/sklearn.py#L3004);
[python/t_boost/sklearn.py:1230](../../python/t_boost/sklearn.py#L1230);
[python/t_boost/sklearn.py:1327](../../python/t_boost/sklearn.py#L1327).

**Scope.** No-argument exports use the bank's stored ledger and remain correct. The bug
occurs when the caller overrides one mass component (`sample_weight` or `exposure`) and
expects the other to retain its documented fit-time default. That component disappears
after portable serialization. Pickle carries the full Python state and does not have
this particular loss.

**Trigger and impact.** Identical `tables` calls on original and loaded estimators
silently report different support, base values and effects. Ordinary predictions remain
unchanged because re-centering preserves the score.

**Reproduction.**

```python
import warnings, json
import numpy as np
from t_boost import TBoostRegressor
warnings.simplefilter('ignore')
x = np.tile(np.array([0., 1.], dtype=np.float32), 100).reshape(-1, 1)
e = np.tile(np.array([1., 10.], dtype=np.float32), 100)
y = np.tile(np.array([1., 30.], dtype=np.float32), 100)
m = TBoostRegressor(objective='poisson', n_trees=20, n_bags=1,
    prune=False, validation_fraction=None, n_jobs=1).fit(x, y, exposure=e)
for name, estimator in [('original', m),
    ('bytes', TBoostRegressor.from_bytes(m.to_bytes())),
    ('json', TBoostRegressor.from_json(m.to_json()))]:
    bank = json.loads(estimator.tables(x, sample_weight=np.ones(200, dtype=np.float32)))
    print(name, bank['tables'][0]['support'], bank['f0'], bank['tables'][0]['values'])
```

```text
original support [0.0, 100.0, 1000.0], f0 0.9991664179988653
         values [0.09906294153093073, -0.9877341847323329, 0.09906294153093073]
bytes/json support [0.0, 100.0, 100.0], f0 0.5550117483092251
         values [0.543217611220571, -0.5435795150426926, 0.543217611220571]
```

**Suggested fix.** Either preserve the default vectors in the serialization contract or
reject partial mass overrides after load when the unspecified component was used at fit
but is unavailable. Existing `_ae_requires_weight_`/`_ae_requires_exposure_` metadata
can support the latter without inflating portable model blobs. Do not silently equate
unavailable fitted exposure to one.

**Acceptance checks.** Before/after bytes and JSON load, cover fit with each mass
separately and both together; no override, each partial override, and both explicit
overrides. Require equivalent export data or a clear error asking for the absent aligned
component. Repeat with reordered/different-size evaluation rows to prevent accidental
reuse of misaligned masses.

<a id="bug-028"></a>
## BUG-028 — Empty categorical index lists are misclassified as Boolean masks

**Priority:** P3. **Status:** Open.

**Source:** [python/t_boost/sklearn.py:2909](../../python/t_boost/sklearn.py#L2909).

**Cause.** `all(isinstance(..., bool) for v in seq)` is true for an empty sequence. A
valid empty list of categorical indices therefore enters mask-length validation instead
of resolving to no categorical columns.

**Trigger and impact.** Dynamic pipelines that compute no categorical indices cannot use
the natural `categorical_features=[]`; a numeric fit fails before training.

```python
import numpy as np
from t_boost import TBoostRegressor
x = np.arange(80, dtype=np.float32).reshape(-1, 1)
TBoostRegressor(categorical_features=[], n_trees=5, n_bags=1,
    prune=False, validation_fraction=None, n_jobs=1).fit(x, x[:, 0])
```

Observed: `ValueError: categorical_features mask length 0 != n_features 1`.

**Suggested fix.** Resolve empty categorical declarations to `[]` before detecting mask
syntax. Preserve length checking for nonempty Boolean masks.

**Acceptance checks.** Empty list, tuple, and empty integer numpy array must behave like
`None` on numeric input. With polars input the explicit empty declaration should
continue unioning in automatic String/Categorical/Enum detection, just as `None` does.
Reject nonempty incorrectly sized masks.

<a id="bug-029"></a>
## BUG-029 — No-argument set_params discards a fitted model

**Priority:** P3. **Status:** Open.

**Source:** [python/t_boost/sklearn.py:2201](../../python/t_boost/sklearn.py#L2201).

**Cause.** `_BaseTBoost.set_params` always deletes fitted attributes after calling the
superclass, even when no parameters were provided. The superclass's no-argument no-op
does not stop this clear loop.

**Trigger and impact.** An empty parameter update can silently make a fitted estimator
unusable, affecting generic wrappers that forward a dynamically constructed parameter
dictionary. No model-setting parameter actually changed.

```python
import numpy as np
from t_boost import TBoostRegressor
x = np.arange(80, dtype=np.float32).reshape(-1, 1)
m = TBoostRegressor(n_trees=5, n_bags=1, prune=False,
    validation_fraction=None, n_jobs=1).fit(x, x[:, 0])
print(m.predict(x[:1]))
m.set_params()  # equivalent to m.set_params(**{})
print(m.predict(x[:1]))
```

Observed: the first call returns `[17.34382057]`; the second raises `NotFittedError:
This TBoostRegressor instance is not fitted yet`.

**Suggested fix.** Immediately return self for an empty parameter mapping. Keep
deliberate invalidation for actual parameter mutations.

**Acceptance checks.** Empty updates preserve prediction, fitted state, serialized bytes
and reports for regressor, binary classifier and multiclass classifier, with sklearn
installed and blocked. Confirm actual updates still invalidate state as intended.

<a id="bug-030"></a>
## BUG-030 — Re-banding a banded model ignores its existing band maps

**Priority:** P2. **Status:** Open.

**Source:**
[crates/t-boost-core/src/banding.rs:224](../../crates/t-boost-core/src/banding.rs#L224);
[crates/t-boost-core/src/banding.rs:251](../../crates/t-boost-core/src/banding.rs#L251);
[crates/t-boost-py/src/lib.rs:5946](../../crates/t-boost-py/src/lib.rs#L5946).

`band_bank` accepts a normally fitted, already-banded `TableBank`. For dense interaction
sources, `Inter::at` indexes `t.values` directly using **merged-grid** coordinates
rather than mapping through `t.axes[d].band_of`. A coordinate larger than the compressed
tensor extent becomes a silent zero via `unwrap_or(0.0)`. `prior_band_sums` similarly
walks compressed values using the full merged-grid extents. Consequently fidelity is
measured against the wrong function, and accepted re-banding changes predictions far
beyond its reported tolerance.

Minimal runnable Python reproduction (runtime about one second):

```python
import json
import numpy as np
from t_boost import TBoostRegressor

rng = np.random.default_rng(80)
x = rng.uniform(-2, 2, size=(600, 2)).astype(np.float32)
y = (np.sin(3*x[:, 0])*np.cos(2*x[:, 1])
     + 0.1*rng.normal(size=600)).astype(np.float32)
m = TBoostRegressor(
    n_trees=40, n_bags=2, max_depth=4, interaction_gain_hurdle=0.0,
    validation_fraction=None, n_jobs=1, band_tolerance=0.75,
    graduate=False,
).fit(x, y)
tm = m._model
p = np.asarray(tm.predict_raw(x), dtype=np.float64)
b, report = tm.band(
    x, np.ones(600), np.ones(600), p, 1.0,
    tolerance=1e-8, pseudo_rows=1000, n_jobs=1,
)
p2 = np.asarray(b.predict_raw(x), dtype=np.float64)
r = json.loads(report)
print(r['skipped'], r['combined_mse'], r['budget'])
print(np.mean((p2-p)**2), np.max(np.abs(p2-p)))
```

Observed:

```text
initial pair merged shape: [87, 85]
initial pair deployed shape: [87, 65] (axis 1 banded)
skipped: None
reported combined_mse: 3.484303750886758e-17
budget: 1.0000000000000001e-16
actual prediction MSE: 2.4263803986694343e-06
maximum raw-score move: 0.011296331882476807
```

At tolerance `0.01`: reported combined MSE `2.6235382786740404e-05`, budget `0.0001`,
actual prediction MSE `0.0016266059431928384`, maximum move `0.4050445109605789`. The
existing `.band` API has no unbanded-input restriction; it should either correctly
interpret existing band maps or explicitly reject already-banded inputs. This is a
**second band pass**, not an allegation that first-pass default fit banding suffers this
specific coordinate error. Add repeated-band tests under near-zero tolerance, including
ordinal and categorical axes, with the real served predictions as the differential
oracle.

<a id="bug-031"></a>
## BUG-031 — Cached CellMaps omit dependencies from compatibility checking

**Priority:** P2. **Status:** Open.

**Source:**
[crates/t-boost-core/src/table_model.rs:445](../../crates/t-boost-core/src/table_model.rs#L445);
[crates/t-boost-core/src/scoring.rs:1131](../../crates/t-boost-core/src/scoring.rs#L1131).

`CellMaps::build` depends on original axis grids, axis provenance and categorical
encoders as well as the bank's merged grids. The cache only stores merged grids;
`score_raw_with` only compares those. Two valid models can have identical realized split
borders but different unused binning borders, making a stale/cross-model cache silently
change scores. The incoming matrix still matches the receiving model exactly and both
models validate. This affects public Rust callers reusing the explicit cache; the Python
binding's own per-model cache was not shown to cross models.

Verification 2026-10-02: reproduced verbatim. The `CellMaps` docstring
(`scoring.rs:1127-1130`) states the maps are "valid only for the model it was built
from, and the scorer refuses it when the bank's merged grids differ", and
`score_raw_with` (`420-426`) documents the same limited guarantee — so the partial
check is a documented caller obligation with a best-effort guard, which makes this a
policy call. The Python binding builds maps per model object in a `OnceLock`
(`t-boost-py/src/lib.rs:8829-8858`), so no cross-model reuse exists there. The silent
wrong answer for a documented-invalid use is still worth closing with a typed error.

Standalone Rust reproduction:

```rust
use t_boost_core::{
    explain::{fixture_model, fixture_serve, RefMeasure},
    table_model::TableModel,
};

fn main() {
    let mut a = fixture_model();
    a.grids[0].borders = vec![0.0, 1.5];
    a.grids[0].n_bins = 4;
    a.trees[0].1.splits[0].bin_le = 2;
    let mut xa = fixture_serve();
    xa.0.grids = a.grids.clone();
    xa.0.data[0] = vec![2, 2, 3, 3];
    a.validate().unwrap();
    let ta = TableModel::from_model(&a, &xa, RefMeasure::Uniform).unwrap();
    ta.validate().unwrap();

    let mut b = a.clone();
    b.grids[0].borders = vec![1.5, 3.0];
    b.trees[0].1.splits[0].bin_le = 1;
    let mut xb = xa.clone();
    xb.0.grids = b.grids.clone();
    xb.0.data[0] = vec![1, 1, 2, 2];
    b.validate().unwrap();
    let tb = TableModel::from_model(&b, &xb, RefMeasure::Uniform).unwrap();
    tb.validate().unwrap();
    assert_eq!(ta.bank.merged_grids, tb.bank.merged_grids);
    println!("correct: {:?}", tb.score_raw(&xb.0, None).unwrap());
    println!(
        "stale cache: {:?}",
        tb.score_raw_with(&xb.0, None, Some(&ta.cell_maps().unwrap()))
            .unwrap()
    );
}
```

Observed:

```text
correct: [6.0, 2.0, 2.0, -1.110223e-16]
stale cache: [6.0, 2.0, 6.0, 2.0]
```

Expected: reject incompatible cache through a typed error. Include all mapping
dependencies in a compatibility signature or bind a cache to immutable model identity.
Regression variants should cover reordered provenance, changed original borders, changed
categorical encoder tuples, per-class maps, and intentional valid reuse across banks
that share the complete mapping inputs.

<a id="bug-032"></a>
## BUG-032 — Recentring a banded interaction-only bank loses axis templates for recreated mains

**Priority:** P2. **Status:** Open.

**Source:**
[crates/t-boost-core/src/banding.rs:1795](../../crates/t-boost-core/src/banding.rs#L1795);
[crates/t-boost-core/src/banding.rs:1834](../../crates/t-boost-core/src/banding.rs#L1834);
[crates/t-boost-core/src/banding.rs:1472](../../crates/t-boost-core/src/banding.rs#L1472).

The public core `retain_tables` API can validly retain an interaction while dropping its
main effects. `TableModel::validate` accepts the resulting bank. Recentring/re-purifying
under another reference measure creates the necessary missing main effects, but the
band-aware driver only saved axis templates keyed by the **original** full feature sets.
Newly created supports get an empty template list and fail with `Internal` instead of
completing the exact change of basis. The unbanded counterpart handles the same
operation successfully.

**Reproduction.** Run the Python example in BUG-030, then save its normally fitted model
with `open("/tmp/tboost_review_banded.json", "w").write(tm.to_json())`. Run the
following Rust program against that generated file:

```rust
use t_boost_core::{
    explain::{FeatureSet, RefMeasure},
    prune::retain_tables,
    table_model::TableModel,
};

fn main() {
    let mut tm =
        TableModel::from_json(&std::fs::read_to_string("/tmp/tboost_review_banded.json").unwrap())
            .unwrap();
    tm.bank = retain_tables(&tm.bank, &[FeatureSet::new(&[0, 1])]);
    println!("validate: {:?}", tm.validate());
    println!(
        "recenter: {:?}",
        tm.bank.recompute_under(RefMeasure::Uniform).map(|_| ())
    );
}
```

Observed:

```text
validate: Ok(())
recenter: Err(Internal { what: "banding: no axis template for raw 0" })
same operation on unbanded interaction-only bank: Ok(())
```

This is a narrower public-core composition; the current Python fit selector keeps mains,
so ordinary untouched default fits do not trigger it. Derive missing subset templates
from their parent axes or keep a raw-feature axis template map. Regression should retain
only a banded pair, change measure and call `recentre_on`, and verify successful score
preservation on every merged-cell tuple. Cover a missing pair/main beneath a
higher-order banded effect as well.

<a id="bug-033"></a>
## BUG-033 — Model loading accepts mismatched feature-set/axis identities and misattributes effects

**Priority:** P2. **Status:** Open.

**Source:**
[crates/t-boost-core/src/table_model.rs:199](../../crates/t-boost-core/src/table_model.rs#L199);
[crates/t-boost-core/src/explain.rs:5166](../../crates/t-boost-core/src/explain.rs#L5166).

This finding deliberately uses malformed model JSON. It is a structural load-validation
gap, distinct from the original nonfinite-coefficient and overflowing-order findings.
`TableModel::validate` checks that axis count matches feature-set order but never checks
that each axis's `raw` identity matches the corresponding feature ID in `u`. Serving
reads the axes; Shapley attribution and support naming read `u`. A single changed
feature ID therefore loads and validates successfully, preserves predictions, and
assigns an effect to the wrong feature. Rating export succeeds too, embedding
contradictory support/axis metadata.

Complete minimal reproduction:

```rust
use t_boost_core::{
    explain::{fixture_model, fixture_serve, RefMeasure},
    table_model::TableModel,
};

fn main() {
    let base =
        TableModel::from_model(&fixture_model(), &fixture_serve(), RefMeasure::Uniform).unwrap();
    let mut doc: serde_json::Value = serde_json::from_str(&base.to_json().unwrap()).unwrap();
    // This main-effect table has axes[0].raw == 0. Change only its claimed support.
    doc["model"]["bank"]["tables"][0]["u"] = serde_json::json!([1]);
    let corrupt = TableModel::from_json(&doc.to_string()).unwrap();
    println!("validate {:?}", corrupt.validate());
    println!(
        "original {:?}, corrupted {:?}",
        base.bank.shap(&[1, 2]),
        corrupt.bank.shap(&[1, 2])
    );
    println!(
        "scores {:?}, {:?}",
        base.bank.score(&[1, 2]),
        corrupt.bank.score(&[1, 2])
    );
    println!(
        "export {:?}",
        corrupt
            .bank
            .to_rating_export(
                corrupt.link,
                &corrupt.mode,
                &corrupt.schema,
                &corrupt.provenance,
                &corrupt.schema.cat_encoders,
                None,
            )
            .map(|_| ())
    );
}
```

Observed:

```text
validate: Ok(())
original SHAP: [1.5555555555555554, -1.111111111111111]
corrupt SHAP: [-0.2222222222222222, 0.6666666666666666]
both scores: 1.9999999999999996
rating export: Ok(())
```

Reject conflicting feature identities at all load boundaries, including dense and
factored effects. Also require feature IDs to be valid, ordered and unique, and table
axis metadata to match its bank grid, so all scoring/explanation/export paths describe
the same function. Regression should mutate one valid serialized feature ID, assert
typed load failure for JSON and binary, and verify successful round trips retain both
scores and per-feature attribution. This is confirmed malformed-input handling, not a
claim that ordinary fits create inconsistent metadata themselves.

<a id="bug-034"></a>
## BUG-034 — Gamma/Tweedie exposure initialization uses the Poisson optimum

**Priority:** P2. **Status:** Open.

**Source:** [loss.rs:373](../../crates/t-boost-core/src/loss.rs#L373), especially the
accumulation at lines 392–393; [Gamma::init_score, line
1447](../../crates/t-boost-core/src/loss.rs#L1447); [Tweedie::init_score, line
1588](../../crates/t-boost-core/src/loss.rs#L1588).

**Trigger and impact.** Fit Gamma or Tweedie with nonuniform exposure. The starting
intercept minimizes the Poisson objective rather than the configured objective. If the
features are constant, no tree can correct it and the completed model remains materially
worse than another constant model under its own loss. With informative features this
also changes every initial gradient and can spend tree capacity on correcting the global
level.

**Cause.** All three log-link objectives call the same helper, which sets the per-unit
rate to `sum(w*y) / sum(w*exposure)`. With `mu_i = exposure_i * rate`, Gamma instead
requires `sum(w*y/exposure) / sum(w)`. Tweedie with power `rho` requires
`sum(w*y*exposure**(1-rho)) / sum(w*exposure**(2-rho))`. The implemented expression is
the `rho=1` Poisson special case. No-exposure data, or uniform exposures, do not expose
this discrepancy.

**Reproduction.** This runs through the public Python estimators and compares
algebraically equivalent representations of the same objective:

```python
import numpy as np
from t_boost import TBoostRegressor

X = np.ones((100, 1), dtype=np.float32)
y = np.full(100, 10.0)
e = np.tile([1.0, 10.0], 50)
for objective, power in [("poisson", 1), ("gamma", 2), ("tweedie", 1.5)]:
    options = dict(
        objective=objective, n_trees=20, n_bags=1, prune=False,
        graduate=False, validation_fraction=None, n_jobs=1,
        reanchor=False, reanchor_slope=False,
    )
    offset_model = TBoostRegressor(**options).fit(X, y, exposure=e)
    equivalent_model = TBoostRegressor(**options).fit(
        X, y / e, sample_weight=e ** (2 - power)
    )
    print(objective, offset_model.predict(X)[0], equivalent_model.predict(X)[0])
```

```text
poisson 1.8181817531585693 1.8181817531585693
 gamma 1.8181817531585693 5.499999523162842
 tweedie 1.8181817531585693 3.1622776985168457
```

Evaluated against the original totals `y` using `rate*e`, summed Gamma deviance is
`294.3089053` for the fitted rate versus `110.6911091` for the optimum. Summed Tweedie
deviance is `544.9419705` versus `430.8549366`. The Poisson control agrees. Repeating the
exposure fits with `reanchor=None`, `False`, and `True` gives the same incorrect
Gamma/Tweedie rate: aggregate-balance reanchoring leaves this already-Poisson-balanced
rate unchanged. Thus the link-aware default does not mask this example. Reanchoring
deliberately has an aggregate-balance contract; this finding concerns the optimizer's
`init_score`, and the explicit `reanchor=False` reproduction isolates it.

**Suggested fix.** Make offset-aware initialization objective-specific, or pass the loss
power to a shared formula. Preserve the current no-offset behavior. Treat any
intentionally requested post-fit balance adjustment separately from the loss-optimal
initialization.

**Acceptance checks.** Verify the sum of initial gradients is approximately zero for
constant models under nonuniform exposure and nonuniform weights. Compare offset and
rate/adjusted-weight representations for Gamma and several Tweedie powers, including the
Poisson boundary/control. Cover constant features where later trees cannot hide the
problem.

<a id="bug-035"></a>
## BUG-035 — Periodic ridge refits desynchronize AGBM's score cache from its trees

**Priority:** P2. **Status:** Open.

**Source:** [engine/boost.rs:895](../../crates/t-boost-core/src/engine/boost.rs#L895),
[previous-score capture at line
1023](../../crates/t-boost-core/src/engine/boost.rs#L1023), and [periodic refit at line
1081](../../crates/t-boost-core/src/engine/boost.rs#L1081).

**Trigger and impact.** Use the public Rust API with `NesterovSpec::Agbm` and
`RefitSpec::Ridge { every_k_trees: Some(k), ... }`. Periodic refits mutate the old
trees' leaves, but AGBM retains the previous iteration's score vector from before those
mutations. Subsequent gradients are evaluated at scores which do not correspond to the
alphas and leaves in the current model. This affects split selection, leaf fitting, and
any validation evaluation using the accumulated score. Python constructs ridge refit
with `every_k_trees: None` and rejects `nesterov=True` entirely ([binding lines 3955
onward](../../crates/t-boost-py/src/lib.rs#L3955)). The demonstrated combination is
supported only through the Rust API; it cannot be requested through the current Python
estimator.

**Cause.** AGBM applies `a'_j=(1+beta)*a_j-beta*previous_a_j` to the current tree
coefficients, then computes `fit_raw=(1+beta)*raw-beta*raw_prev`. These operations are
equivalent only when both score vectors use the same underlying tree leaf values. The
ridge solve changes those values after `raw_prev` has been captured. It refreshes `raw`,
but leaves `raw_prev` stale. Once an existing tree has been refit and a nonzero old
coefficient is subtracted, the cached score no longer matches the stored model.

**Reproduction.** The complete Rust harness below wraps `SquaredError` solely to record
the score vectors supplied to `grad_hess`; numerical calculations still use the
production loss. It fits prefixes of one, two, and three rounds on 200 rows, `x_i=i`,
`y_i=5*sin(i/20)+i/20`, with learning rate `.3`, lambda `10`, leaf refinement off, ridge
`l2=10/max_iter=1/every_k_trees=1`, and AGBM momentum correction off. At the third
round, it compares the recorded gradient input against a fresh prediction from the
two-round model with the production alpha mixture applied. The previous coefficients
come from the separately fitted one-round prefix.

```text
nt=1 trees=1 gh_calls=2
nt=2 trees=2 gh_calls=4
nt=3 trees=3 gh_calls=6
AGBM + every-round ridge third-round gradient raw max difference: 0.41963005
first row: gradient raw=2.0824442, current tree/alpha score=2.106028
AGBM without ridge control max difference: 0.0000009536743
```

The difference is far larger than the ordinary float32 accumulation-order difference
shown by the no-refit control. This establishes state inconsistency; it is not a
measured benchmark accuracy delta.

**Suggested fix.** When leaves have been changed, recompute the prior-coefficient score
vector using the new leaves before the next AGBM blend; alternatively use a full
reconstruction for this combination or reject it explicitly until supported. Keep offset
contributions and rollback snapshots consistent as well.

**Acceptance checks.** Check cached-versus-reconstructed lookahead scores after periodic
refits, with and without momentum correction and validation truncation. Include a
control without refits, exposure offsets, and cross-thread determinism.

<a id="bug-036"></a>
## BUG-036 — Bootstrap copies cross the early-stopping train/validation boundary

**Priority:** P2. **Status:** Open — **reclassified on verification as a documented
limitation**, see below.

**Source:** [engine/boost.rs:2897](../../crates/t-boost-core/src/engine/boost.rs#L2897),
the bag-local fit specification at [line
2904](../../crates/t-boost-core/src/engine/boost.rs#L2904), and the later carve in
[fit_single, lines 657–665](../../crates/t-boost-core/src/engine/boost.rs#L657)
(`carve_validation_rows_stratified`; line 671 is the intercept-honesty comment that
refers to it). The multiclass path repeats the bootstrap-before-carve sequence at
lines 1566–1584, carving positionally at line 2083.

**Trigger and impact.** Configure `n_bags >= 2`, `bag_subsample >= 1.0` (documented
bootstrap sampling), and an internal `validation_fraction`, without a supplied fixed
holdout. Copies of the same original row can be used for both tree fitting and early
stopping. Validation is then partly in-sample, undermining selection of the best
iteration. This does not require categorical features. The shipped `bag_subsample=.8`
subagging default avoids duplicate rows, as does an honestly constructed fixed holdout
sampled only after its boundary is fixed.

**Verification 2026-10-02 — this is documented behaviour.** The overlap reproduces
exactly, but it is not an undisclosed defect: the leak is the stated reason the
default is `0.8`. `boosters.rs:176-178` describes subagging as giving "no within-bag
train/val leak"; the `bag_subsample` docstring (`sklearn.py:5846-5852`) explains that
"a bootstrap bag duplicates rows, which makes the per-bag early-stopping validation
carve overlap the bag's own training multiset, a train/validation leak that lets
validation deviance improve indefinitely and defeats early stopping. Set to `1.0` to
restore classic bootstrap bagging"; `sklearn.py:1726-1730` says the same internally.
Users who set `1.0` are told what they get. Treat this entry as a known limitation
whose remedy (an honest carve before bootstrap) is an enhancement, not a correctness
fix; it does not belong on the same footing as the other P2 items.

**Cause.** The outer loop first samples original row indices with replacement,
materializes a matrix containing duplicates, and then asks `fit_single` to carve its
validation set by the positions in that materialized matrix. The original row identities
have been discarded for the carve. A row and another copy of itself therefore receive
unrelated train/validation decisions.

**Reproduction.** The Rust harness below assigns unique targets equal to original row
IDs so the production loss callbacks can reveal exactly which original observations were
used for the intercept and for the first validation evaluation. This is identification
instrumentation, not a modified split or sampling implementation. Use `n_rows=100`,
`n_bags=2`, `n_trees=1`, `validation_fraction=Some(.2)`, seed `0`, and no leaf
refinement. Run with `RAYON_NUM_THREADS=1` only to make the callback event order easy to
pair by bag.

```text
bag_subsample=1 train_n=79 holdout_n=21 heldout_rows_also_in_train=13
bag_subsample=1 train_n=79 holdout_n=21 heldout_rows_also_in_train=14
bag_subsample=0.8 train_n=63 holdout_n=17 heldout_rows_also_in_train=0
bag_subsample=0.8 train_n=63 holdout_n=17 heldout_rows_also_in_train=0
```

The first bag's overlapping validation target IDs are
`[48,35,77,79,50,99,17,31,5,79,54,26,1]`. The counts are row occurrences, so duplicated
IDs can appear twice in validation. This proves overlap, not a quantified downstream
benchmark bias. The scalar path was instrumented at runtime; the corresponding
multiclass route was inspected.

**Suggested fix.** Select the honest validation boundary over original rows before
bootstrap sampling, then draw only from its training complement. Alternatively group all
copies of an original observation during the carve. Apply the same policy to scalar and
multiclass bagging.

**Acceptance checks.** For bootstrap bags, assert the sets of original row IDs used in
training and validation are disjoint. Cover multiple seeds, both objective families,
pre-specified holdouts, group sampling, and the unchanged subagging control.

<a id="bug-037"></a>
## BUG-037 — Bagged core fitting panics on a short fixed-holdout mask

**Priority:** P2. **Status:** Open.

**Source:** [engine/boost.rs:2829](../../crates/t-boost-core/src/engine/boost.rs#L2829),
[multiclass line 1514](../../crates/t-boost-core/src/engine/boost.rs#L1514); the scalar
outer validator at [line 2722](../../crates/t-boost-core/src/engine/boost.rs#L2722) does
not validate this mask.

**Trigger and impact.** A Rust caller passes a `FitSpec.fixed_holdout` shorter than the
matrix row count while using at least two outer bags. The bag loop indexes the slice
directly and panics inside Rayon, rather than returning `PbError::ShapeMismatch`. Both
scalar and multiclass fits were reproduced. The single-fit path has a proper mask-length
check in `split_rows_by_mask`; bagging accesses the mask before that check can run.
Binding-owned multiclass input does validate `es_holdout.len()` earlier, so the
demonstrated public surface is the Rust API.

**Reproduction.** In the complete combined Rust harness, a valid 200-row matrix and
targets are passed with `fixed_holdout=Some(&[false])`, `n_bags=2`, `bag_subsample=.8`.
`catch_unwind(AssertUnwindSafe(|| booster.fit(...)))` returns an unwind; the analogous
`fit_multiclass` call also unwinds:

```text
engine/boost.rs:2829:61: index out of bounds: the len is 1 but the index is 1
short fixed_holdout scalar bagged panic=true
engine/boost.rs:1514:57: index out of bounds: the len is 1 but the index is 1
short fixed_holdout multiclass bagged panic=true
```

Verification 2026-10-02 reproduced both panics and added two controls: with
`n_bags=1` both paths return `ShapeMismatch "fixed_holdout len 1 != n_rows 200"`
as expected, and — not previously recorded — a mask that is too **long** (201
entries for 200 rows) with two bags is silently accepted and returns `Ok` with
two trees; no error at all. The acceptance checks below already anticipate long
masks. The no-panic rule is a library-wide contract (`CONTRIBUTING.md:111-118`),
so the Rust-API-only scope does not make this acceptable behaviour.

**Suggested fix.** Validate fixed-holdout shape once, before entering either parallel
bag loop, and share the validation with scalar/multiclass single fits. Reject a mask
that leaves no training rows with a typed error too.

**Acceptance checks.** Short and long masks must return a shape error for single and
bagged scalar/multiclass fits without unwinding. Valid all-false and mixed masks should
retain their documented behavior; all-true masks should fail cleanly.

<a id="bug-038"></a>
## BUG-038 — Core multiclass fitting silently accepts invalid sample weights

**Priority:** P2. **Status:** Open.

**Source:** [engine/boost.rs:2418](../../crates/t-boost-core/src/engine/boost.rs#L2418)
(`multiclass_init`), [line 2496](../../crates/t-boost-core/src/engine/boost.rs#L2496)
(`fill_grad_hess_from_probs`), and the length-only fit check around line 2050.

**Trigger and impact.** A direct Rust caller supplies negative or NaN sample weights to
`Booster::fit_multiclass`. Unlike scalar losses, multiclass checks the slice length but
not each weight's domain/finiteness. It can return a completed model trained from
invalid priors/statistics instead of refusing the input. NaN prior sums are especially
misleading: `max(EPS)` turns the invalid class priors into finite values, hiding the
problem.

**Reproduction.** Use a pre-binned 200-row numeric matrix, `y_i=i%3`, three class
labels, one tree, no validation, and unit weights except `w[0]=-10` or `w[0]=NaN`. The
complete Rust harness returns these successful class intercepts:

```text
single, negative: Ok([-1.2163954, -1.0370544, -1.0520923])
single, NaN:      Ok([-27.631021, -27.631021, -27.631021])
2 bags, negative: Ok([-1.164468, -1.0692682, -1.0692682])
2 bags, NaN:      Ok([-14.358605, -14.367951, -14.367951])
```

The bagged example uses `bag_subsample=.8`. This is specifically a Rust core validation
omission. Public Python attempts with analogous weights were rejected by binning,
Newton-leaf checks, or later explanation-mass validation; no end-to-end successful
Python estimator with these invalid weights was established.

Verification 2026-10-02: all four intercept vectors reproduced exactly; `w[0] = +inf`
is accepted with the same intercepts as NaN; the scalar `Booster::fit` with
`SquaredError` rejects all three (`"weight[0] must be finite and >= 0"`), confirming
the scalar/multiclass asymmetry. The length-only check is at lines 2053–2058 (the
cited "around line 2050" is a blank line). The binding's `fit_multiclass`
(`lib.rs:4163`) has no weight-domain check either; Python is shielded only because
weights pass through `build_grid`'s finite/non-negative check (`grid.rs:46-50`),
which a pre-binned Rust caller bypasses.

**Suggested fix.** Validate all multiclass weights as finite and nonnegative before
initialization or resampling; reject nonpositive effective training weight and
class-prior corner cases explicitly. Keep finite-check guarantees aligned with the
scalar `Loss` implementations.

**Acceptance checks.** Negative, NaN, and infinite weights should fail identically
across scalar/multiclass and bagged/single paths. Include invalid weights at rows
excluded by a particular sample or holdout, and retain valid zero-weight behavior.

## Complete Rust reproductions for the training findings

Compile these as temporary binaries against the current release `t_boost_core` build, or
put each in a temporary Cargo project depending on `t-boost-core` by local path. No
production files were changed.

### Combined AGBM, native weights, and holdout-shape harness

```rust
use std::sync::Mutex;
use t_boost_core::boosters::{BoosterConfig, EnsembleSpec, NesterovSpec, RefitSpec};
use t_boost_core::constraints::{CredibilityFloor, InteractionPolicy};
use t_boost_core::data::{bin_columns, BinConfig};
use t_boost_core::engine::{Booster, Config, FitSpec};
use t_boost_core::loss::{GradHess, Link, Loss, Metric, ObjectiveTag, SquaredError};
use t_boost_core::PbError;
struct Recorded(Mutex<Vec<Vec<f32>>>);
impl Loss for Recorded {
    fn grad_hess(
        &self,
        y: &[f32],
        raw: &[f32],
        w: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError> {
        self.0.lock().unwrap().push(raw.to_vec());
        SquaredError.grad_hess(y, raw, w, out)
    }
    fn init_score(&self, y: &[f32], w: &[f32], off: Option<&[f32]>) -> Result<f64, PbError> {
        SquaredError.init_score(y, w, off)
    }
    fn link(&self) -> Link {
        Link::Identity
    }
    fn pred_from_raw(&self, r: f32) -> f32 {
        r
    }
    fn deviance(&self, y: &[f32], r: &[f32], w: &[f32]) -> Result<f32, PbError> {
        SquaredError.deviance(y, r, w)
    }
    fn default_metric(&self) -> Metric {
        Metric::Rmse
    }
    fn objective_tag(&self) -> ObjectiveTag {
        SquaredError.objective_tag()
    }
}
fn main() {
    let n = 200;
    let feature = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let y = (0..n)
        .map(|i| {
            let z = i as f32 / 20.;
            z.sin() * 5. + z
        })
        .collect::<Vec<_>>();
    let x = bin_columns(&[&feature], None, &BinConfig::default(), 0).unwrap();
    let loss = Recorded(Mutex::new(vec![]));
    let mut spec = FitSpec {
        loss: &loss,
        weight: None,
        exposure: None,
        monotone: Default::default(),
        interaction: InteractionPolicy::default(),
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed: 0,
    };
    let cfg = Config {
        n_trees: 2,
        learning_rate: 0.3,
        lambda: 10.,
        leaf_refine_steps: 0,
        boosters: BoosterConfig {
            refit_leaves: RefitSpec::Ridge {
                l2: 10.,
                max_iter: 1,
                every_k_trees: Some(1),
            },
            nesterov: NesterovSpec::Agbm {
                momentum_correction: false,
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut models = vec![];
    for nt in [1, 2, 3] {
        loss.0.lock().unwrap().clear();
        let m = Booster::with_config(Config {
            n_trees: nt,
            ..cfg.clone()
        })
        .fit(&x, &y, &spec)
        .unwrap();
        println!(
            "nt={nt} trees={} gh_calls={}",
            m.trees.len(),
            loss.0.lock().unwrap().len()
        );
        models.push(m);
    }
    let actual = loss.0.lock().unwrap()[4].clone();
    let mut expected_model = models[1].clone();
    for (i, (a, _)) in expected_model.trees.iter_mut().enumerate() {
        let prev = models[0].trees.get(i).map_or(0., |v| v.0);
        *a = 1.5 * *a - 0.5 * prev;
    }
    let expected = expected_model.predict(&x, None).unwrap();
    let max_diff = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max);
    println!("AGBM + every-round ridge third-round gradient raw max difference from persisted alpha mixture: {max_diff}; first actual={} expected={}",actual[0],expected[0]);
    let mut control_models = vec![];
    for nt in [1, 2, 3] {
        loss.0.lock().unwrap().clear();
        let mut c = cfg.clone();
        c.n_trees = nt;
        c.boosters.refit_leaves = RefitSpec::Off;
        control_models.push(Booster::with_config(c).fit(&x, &y, &spec).unwrap());
    }
    let actual = loss.0.lock().unwrap()[2].clone();
    let mut expected_model = control_models[1].clone();
    for (i, (a, _)) in expected_model.trees.iter_mut().enumerate() {
        let prev = control_models[0].trees.get(i).map_or(0., |v| v.0);
        *a = 1.5 * *a - 0.5 * prev;
    }
    let expected = expected_model.predict(&x, None).unwrap();
    println!(
        "AGBM without ridge control max difference: {}",
        actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max)
    );
    for bagged in [false, true] {
        let booster = Booster::with_config(Config {
            n_trees: 1,
            boosters: BoosterConfig {
                ensemble: if bagged {
                    EnsembleSpec::OuterBag {
                        n_bags: 2,
                        bag_subsample: 0.8,
                        cell_refit: None,
                    }
                } else {
                    EnsembleSpec::Off
                },
                ..Default::default()
            },
            ..Default::default()
        });
        let ym = (0..n).map(|i| (i % 3) as f32).collect::<Vec<_>>();
        let classes = vec!["a".into(), "b".into(), "c".into()];
        for kind in ["negative", "nan"] {
            let mut w = vec![1.; n];
            w[0] = if kind == "negative" { -10. } else { f32::NAN };
            let ws = FitSpec {
                weight: Some(&w),
                monotone: spec.monotone.clone(),
                interaction: spec.interaction.clone(),
                ..spec
            };
            let result = booster.fit_multiclass(&x, &ym, 3, &classes, &ws);
            println!(
                "multiclass bagged={bagged} weight={kind}: {:?}",
                result
                    .as_ref()
                    .map(|m| m.classes.iter().map(|x| x.f0).collect::<Vec<_>>())
            );
        }
        spec.weight = None;
    }
    let mask = [false];
    spec.fixed_holdout = Some(&mask);
    let booster = Booster::with_config(Config {
        n_trees: 1,
        boosters: BoosterConfig {
            ensemble: EnsembleSpec::OuterBag {
                n_bags: 2,
                bag_subsample: 0.8,
                cell_refit: None,
            },
            ..Default::default()
        },
        ..Default::default()
    });
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| booster.fit(&x, &y, &spec)));
    println!(
        "short fixed_holdout scalar bagged panic={}",
        result.is_err()
    );
    let ym = (0..n).map(|i| (i % 3) as f32).collect::<Vec<_>>();
    let classes = vec!["a".into(), "b".into(), "c".into()];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        booster.fit_multiclass(&x, &ym, 3, &classes, &spec)
    }));
    println!(
        "short fixed_holdout multiclass bagged panic={}",
        result.is_err()
    );
}
```

### Bootstrap early-stopping harness

Run this binary with `RAYON_NUM_THREADS=1` so instrumentation events for different bags
are sequential.

```rust
use std::sync::Mutex;
use t_boost_core::boosters::{BoosterConfig, EnsembleSpec};
use t_boost_core::constraints::{CredibilityFloor, InteractionPolicy};
use t_boost_core::data::{bin_columns, BinConfig};
use t_boost_core::engine::{Booster, Config, FitSpec};
use t_boost_core::loss::{GradHess, Link, Loss, Metric, ObjectiveTag, SquaredError};
use t_boost_core::PbError;
struct Recorded(Mutex<Vec<(bool, Vec<f32>)>>);
impl Loss for Recorded {
    fn grad_hess(
        &self,
        y: &[f32],
        raw: &[f32],
        w: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError> {
        SquaredError.grad_hess(y, raw, w, out)
    }
    fn init_score(&self, y: &[f32], w: &[f32], off: Option<&[f32]>) -> Result<f64, PbError> {
        self.0.lock().unwrap().push((true, y.to_vec()));
        SquaredError.init_score(y, w, off)
    }
    fn link(&self) -> Link {
        Link::Identity
    }
    fn pred_from_raw(&self, r: f32) -> f32 {
        r
    }
    fn deviance(&self, y: &[f32], r: &[f32], w: &[f32]) -> Result<f32, PbError> {
        self.0.lock().unwrap().push((false, y.to_vec()));
        SquaredError.deviance(y, r, w)
    }
    fn default_metric(&self) -> Metric {
        Metric::Rmse
    }
    fn objective_tag(&self) -> ObjectiveTag {
        SquaredError.objective_tag()
    }
}
fn main() {
    let feature = (0..100).map(|i| i as f32).collect::<Vec<_>>();
    let x = bin_columns(&[&feature], None, &BinConfig::default(), 0).unwrap();
    let loss = Recorded(Mutex::new(vec![]));
    let spec = FitSpec {
        loss: &loss,
        weight: None,
        exposure: None,
        monotone: Default::default(),
        interaction: InteractionPolicy::default(),
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed: 0,
    };
    for fraction in [1., 0.8] {
        loss.0.lock().unwrap().clear();
        let cfg = Config {
            n_trees: 1,
            validation_fraction: Some(0.2),
            leaf_refine_steps: 0,
            boosters: BoosterConfig {
                ensemble: EnsembleSpec::OuterBag {
                    n_bags: 2,
                    bag_subsample: fraction,
                    cell_refit: None,
                },
                ..Default::default()
            },
            ..Default::default()
        };
        Booster::with_config(cfg).fit(&x, &feature, &spec).unwrap();
        let mut train = vec![];
        let mut awaiting = false;
        for (init, y) in loss.0.lock().unwrap().iter() {
            if *init {
                train = y.clone();
                awaiting = true;
            } else if awaiting {
                let overlapping = y
                    .iter()
                    .filter(|r| train.contains(r))
                    .copied()
                    .collect::<Vec<_>>();
                println!("bag_subsample={fraction} train_n={} holdout_n={} heldout_rows_also_in_train={} overlap_ids={overlapping:?}",train.len(),y.len(),overlapping.len());
                awaiting = false;
            }
        }
    }
}
```

<a id="bug-039"></a>
## BUG-039 — An uninformative weighted feature can abort an otherwise valid fit

**Priority:** P2. **Status:** Open.

**Source:** [data/grid.rs:44](../../crates/t-boost-core/src/data/grid.rs#L44), the
quantile construction branches at [lines
84–91](../../crates/t-boost-core/src/data/grid.rs#L84) (line 81, cited earlier, is a
comment inside the midpoint branch at 79–83), and
[quantile weight check, line 163](../../crates/t-boost-core/src/data/grid.rs#L163).
The all-missing short-circuit that already returns a degenerate grid is at lines
62–68. Note that the unit test `all_zero_weight_on_quantile_path_errors`
(`grid.rs:516-523`) pins the per-column error on exactly this per-column situation,
so a fix must revisit that test, not just the production branch.

**Trigger and impact.** Supply finite, nonnegative sample weights with positive total
weight, and a numeric feature whose finite values occur only on zero-weight rows. If
that feature has more distinct values than `max_bin`, binning raises and prevents the
entire fit, even when another feature has usable signal. Removing the uninformative
feature or the zero-weight rows lets the same fit succeed. The all-missing version of
that feature is already supported.

**Cause.** The binner filters nonfinite feature values before constructing its weighted
distribution, but retains finite values with zero weight. Here the remaining
distribution has zero mass. The quantile path treats this as a fatal input error instead
of returning a degenerate grid for a feature with no effective observations. This is
distinct from rejecting a dataset whose *global* sample weight is zero: the reproduced
dataset has total weight `100`. It is also different from the subsampling failure in
BUG-049: this example uses one bag and fails before training starts.

**Reproduction.** Run through the public estimator:

```python
import numpy as np
from t_boost import TBoostRegressor

x = np.column_stack([
    np.r_[np.arange(300), np.full(100, np.nan)],
    np.arange(400),
]).astype(np.float32)
y = x[:, 1].copy()
w = np.r_[np.zeros(300), np.ones(100)].astype(np.float32)
options = dict(
    n_trees=5, n_bags=1, prune=False, graduate=False,
    band_tolerance=None, validation_fraction=None, n_jobs=1,
)
for label, design, target, mass in [
    ("original", x, y, w),
    ("drop zero-weight rows", x[w > 0], y[w > 0], w[w > 0]),
    ("drop uninformative feature", x[:, 1:], y, w),
]:
    try:
        model = TBoostRegressor(**options).fit(design, target, sample_weight=mass)
        print(label, "ok", model.predict(design[:2]))
    except ValueError as error:
        print(label, type(error).__name__, str(error))
```

```text
original TBoostValueError invalid input: non-positive total weight in binning subsample
drop zero-weight rows ok [321.28079224 321.28079224]
drop uninformative feature ok [321.28079224 321.28079224]
```

The existing `all_zero_weight_on_quantile_path_errors` unit test covers a column with
globally all-zero weights; it does not distinguish this valid global-weight,
missing-feature case. The failure is unnecessary input brittleness, not a claim that a
model returned incorrect predictions.

**Suggested fix.** Handle a feature with no positive-weight finite observations as an
uninformative axis, consistently with the existing all-missing case. Keep global
target/weight validation responsible for rejecting a truly empty effective training
dataset. Define a fallback for a random binning sample that happens to miss all
positive-weight finite rows as well.

**Acceptance checks.** Cover this two-feature fixture, an entirely missing axis, finite
observations with zero weight, cardinalities on both sides of `max_bin`, and all-zero
global weights. Valid weighted data should train without requiring the caller to
manually remove an irrelevant feature; invalid global weight must still raise a typed
error.

<a id="bug-040"></a>
## BUG-040 — Small updates disappear from training scores but accumulate in deployed tables

**Priority:** P2. **Status:** Open.

**Source:** [engine/boost.rs:696](../../crates/t-boost-core/src/engine/boost.rs#L696)
initializes the raw training scores as `f32`;
[update_raw:7258](../../crates/t-boost-core/src/engine/boost.rs#L7258), [precomputed
update:7281](../../crates/t-boost-core/src/engine/boost.rs#L7281), and [walk
update:7302](../../crates/t-boost-core/src/engine/boost.rs#L7302) add each tree's small
leaf directly to that large score. The validation-carve counterpart uses the same
accumulation at [line 7340](../../crates/t-boost-core/src/engine/boost.rs#L7340). Native
serving repeats float32 accumulation in
[scoring.rs:418](../../crates/t-boost-core/src/scoring.rs#L418), whereas
[engine/mod.rs:1224](../../crates/t-boost-core/src/engine/mod.rs#L1224) reconstructs the
retained trees in float64.

**Trigger and impact.** Fit ordinary squared-error regression with a large target
baseline and modest variation. When an individual tree update is below the float32
spacing at that baseline, the cached training score can discard it. The tree itself
retains the leaf value, and the deployed table model accumulates those values at higher
precision. Subsequent gradients keep attempting to correct an error that the training
cache never sees corrected. More trees then increase the deployed error. This reaches
the public Python estimator with finite inputs and without acceleration, periodic
refits, or malformed state.

**Reproduction.** All shifted target values in this example are exactly representable as
float32; this is not loss of target differences at input casting. The four unique
feature rows are repeated twenty times:

```python
import numpy as np
from t_boost import TBoostRegressor

x = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32), (20, 1))
y = np.tile(np.array([6, 2, 2, 0], dtype=np.float32), 20)
for n_trees in [20, 100]:
    for shift in [0.0, 10_000_000.0]:
        model = TBoostRegressor(
            n_trees=n_trees, n_bags=1, prune=False, graduate=False,
            band_tolerance=None, validation_fraction=None, n_jobs=1,
            learning_rate=0.2, leaf_refine_steps=0, reanchor=False,
        ).fit(x, y + np.float32(shift))
        centered = model.predict(x) - shift
        print(n_trees, shift, centered[:4], np.mean((centered - y) ** 2))
```

```text
20, shift 0:    [5.89285326, 2.02595282, 2.02595282, 0.05524150], MSE 0.0039697861
20, shift 1e7:  [10, 10, -2, -2],                              MSE 25
100, shift 0:   [6.00000429, 1.99999726, 1.99999726, -9.75e-8],  MSE 8.37e-12
100, shift 1e7: [41, 41, -18, -18],                            MSE 867.5
```

**Independent accumulation check.** For the same options and inputs, construct the
native booster through `model._new_booster(fit_pool_width=1).fit(x, target)` and convert
it through `model._full_tables(native, x, target, None, None, None)`. These are private
Python helpers used only to localize the failure; the public reproduction above requires
neither. At both 20 and 100 trees, the native `predict_raw(x) - 1e7` returns `[2, 2, 2,
2]` for the four distinct rows. The table scores are the public predictions above. The
maximum native/table discrepancy grows from `8` to `39`; the unshifted controls differ
by at most approximately `5.3e-6`. This isolates score accumulation from input
quantization and shows the two serving representations no longer agree at meaningful
precision.

This is distinct from BUG-035's periodic AGBM leaf-rewrite/cache mismatch and BUG-053's
cancellation in a variance-certification calculation. No such options or diagnostic
gates are required here. The confirmed accuracy result is this bounded translation
fixture, not an assertion about all large-valued datasets.

**Suggested fix.** Preserve small increments in the training state, for example with
higher-precision or compensated accumulation, or an explicitly centered score
representation where the objective permits it. Align the accumulation contract used for
gradients, validation, native serving, and deployed tables. Merely increasing an
equality tolerance or casting the already-rounded final cache to float64 cannot recover
the lost updates.

**Acceptance checks.** Compare translated and untranslated squared-error fits whose
target differences remain exactly representable, across learning rates and tree counts.
Check cached scores against reconstruction from retained trees and against the deployed
table model. Verify both ordinary and validation-carved update paths; additional trees
should not repeatedly accumulate a correction that was lost only in the training cache.

<a id="bug-041"></a>
## BUG-041 — Pricing reports label merged-grid A/E aggregates with an interaction's compressed axis

**Priority:** P2. **Status:** Open.

**Source:** [sklearn.py:3219](../../python/t_boost/sklearn.py#L3219) overwrites the axis
per feature; [line 3223](../../python/t_boost/sklearn.py#L3223) attaches it to the A/E
result. [Lines 3178 onward](../../python/t_boost/sklearn.py#L3178) aggregate native
merged-grid indices, while
[serialize.rs:1335](../../crates/t-boost-core/src/serialize.rs#L1335) exports each
table's own band axis.

**Trigger.** An ordinary first fit uses default banding, and an interaction with lower
Sobol share than its main effects has a compressed axis. Rating tables are exported in
descending Sobol order, so the interaction overwrites the correct uncompressed main
axis. Different interactions can have different band maps, so a raw feature does not in
general have one universal table band axis.

**Impact.** A/E counts, actual totals, expected totals and mass are associated with the
wrong ranges/levels in `pricing_report`. Some populated aggregate cells have indices
beyond the reported axis's entire extent. This is not BUG-030's repeated banding defect;
no second band pass is involved.

**Reproduction.**

```python
import numpy as np
from t_boost import TBoostRegressor
rng = np.random.default_rng(80)
x = rng.uniform(-2, 2, size=(600, 2)).astype(np.float32)
y = (0.5 * (x[:, 0] + x[:, 1])
     + np.sin(3*x[:, 0])*np.cos(2*x[:, 1])
     + 0.1*rng.normal(size=600)).astype(np.float32)
m = TBoostRegressor(
    n_trees=40, n_bags=2, max_depth=4, interaction_gain_hurdle=0,
    validation_fraction=None, n_jobs=1, band_tolerance=.75, graduate=False,
).fit(x, y)
r = m.pricing_report(x, y)
print([(t['feature_set'], t['shape']) for t in r['tables']['tables']])
for a in r['actual_vs_expected']:
    print(a['feature'], len(a['rows']), a['axis']['cells'],
          sum(a['mass'][a['axis']['cells']:]))
```

**Observed.**

```text
[([0], [83]), ([1], [79]), ([0, 1], [65, 65])]
f0 83 65 162.0
f1 79 65 129.0
```

The final value is the number of unit-mass rows whose merged-cell index is outside the
attached axis. Even indices within range can carry wrong labels once earlier cells have
merged.

**Suggested fix.** Export the canonical merged-grid axis used by `cell_indices` for A/E
labeling, independently of per-table compressed axes. Alternatively choose and
explicitly name a reporting band grid, and map/reaggregate the A/E arrays onto that
exact grid. Do not select axes by last table occurrence.

**Acceptance checks.** Verify every A/E cell's label by independently locating input
values in the reported axis, for banded numeric/categorical fits, different table export
orders, multiple differently banded interactions involving the same feature, named
inputs, and loaded models. Unused/no-table features need an explicit axis contract too.

<a id="bug-042"></a>
## BUG-042 — Multiclass contribution DataFrames crash when one class has only an intercept

**Priority:** P2. **Status:** Open.

**Source:** [sklearn.py:5491](../../python/t_boost/sklearn.py#L5491) constructs
per-class frames; [line 5494](../../python/t_boost/sklearn.py#L5494) makes empty class
lists, and [lines 5515 onward](../../python/t_boost/sklearn.py#L5515) infer schemas and
concatenate them strictly.

**Cause.** Each class is converted to a separate long DataFrame. A class with no effect
tables produces a zero-row frame whose string columns infer `Null`; a class with effects
infers `String`. Concatenation fails. Such a constant class bank occurs under valid
symmetric class distributions without any malformed state.

**Reproduction.**

```python
import numpy as np
from t_boost import TBoostClassifier
x = np.tile(np.array([-1, -1, -1, 1, 1, 1], dtype=np.float32), 100).reshape(-1, 1)
y = np.tile([0, 1, 1, 0, 2, 2], 100)
m = TBoostClassifier(
    n_trees=10, n_bags=1, prune=False, validation_fraction=None,
    n_jobs=1, colsample_bytree=1, leaf_refine_steps=0,
).fit(x, y)
records = m.predict_contributions(x[:2])
print([[len(c['contributions']) for c in row['classes']] for row in records])
print(m.predict_contributions(x[:2], return_format='dataframe'))
```

**Observed.**

```text
[[0, 1, 1], [0, 1, 1]]
SchemaError: type String is incompatible with expected type Null
This error occurred with the following context stack:
    [1] failed to vstack column 'class'
```

`predict_proba` and records contributions succeed; the probabilities are finite. A
constant scalar bank likewise produces an empty long frame, but zero rows can follow the
documented one-row-per-term convention when there are zero terms; that scalar behavior
is not counted as a separate defect.

**Suggested fix.** Declare a stable schema rather than infer from empty lists, or
otherwise handle empty class frames consistently before concatenation. Preserve the
documented one-row-per-term representation; an optional baseline row would be a separate
API choice.

**Acceptance checks.** Constant/nonconstant class banks in either order, all-constant
regression/multiclass fits, empty prediction batches, records/DataFrame consistency, and
serialization round trips. Every documented format should succeed for valid fitted
models.

<a id="bug-043"></a>
## BUG-043 — Gini awards arbitrary ranking skill to tied scores according to row order

**Priority:** P2. **Status:** Open.

**Source:** [metrics.py:48](../../python/t_boost/metrics.py#L48) breaks sort ties by
input index; [line 81](../../python/t_boost/metrics.py#L81) integrates individual rows
within each tie. [xtask/src/main.rs:865](../../xtask/src/main.rs#L865) repeats this
logic. The intended binary identity is asserted in
[test_sklearn.py:1146](../../python/tests/test_sklearn.py#L1146): normalized Gini equals
`2*AUC - 1`; that fixture has continuous scores and no ties.

**Impact.** A model with identical predictions for every row can be reported as
perfectly discriminating, perfectly reversed, or somewhere in between solely from
evaluation row order. Ties are ordinary for table/tree predictors. This affects the
public Python metric and, by identical source logic, the Rust release/accuracy metric.
Python behavior was executed; the Rust implementation was inspected, not separately
instrumented in this pass.

**Reproduction.**

```python
import numpy as np
from sklearn.metrics import roc_auc_score
from t_boost.metrics import ordered_gini
p = np.ones(4)
for y in [np.array([0., 0., 1., 1.]),
          np.array([1., 1., 0., 0.]),
          np.array([0., 1., 0., 1.])]:
    print(ordered_gini(y, p), 2 * roc_auc_score(y, p) - 1)
```

```text
-1.0 0.0
 1.0 0.0
-0.5 0.0
```

**Scope.** The index tie-break itself is documented and was deliberately chosen for
deterministic Python/Rust parity. The defect is treating that arbitrary ordering as
predictive discrimination rather than aggregating equal-score mass before integrating
the Gini curve; it violates the existing AUC identity and permutation invariance. This
does not demand changing the separately documented equal-count lift-bucket convention.

Verification 2026-10-02: the three constant-score results reproduced exactly. On a
tied-score fixture (scores rounded to 0.1) the metric is permutation-variant
(`-0.1152` versus `-0.1098` after a joint row permutation; `2·AUC−1 = -0.1139`), while
a no-ties control matches `2·AUC−1` to `1e-15`. `metrics.py:1-5` states the contract
is parity with `xtask`, and `metrics.py:49` documents the index tie-break, which makes
this a policy call; nothing, however, documents that ties earn ranking skill, and the
comment at `test_sklearn.py:1146` states the AUC identity for binary targets without a
no-ties qualification. The Rust implementation (`xtask/src/main.rs:865-897`) was
confirmed identical by reading.

**Suggested fix.** Preserve deterministic sorting, but integrate one segment per tied
score, summing its weight and weighted outcomes. Apply matching behavior to Python and
Rust and define signed-zero equality consistently.

**Acceptance checks.** Constant scores return zero for nondegenerate binary data,
tied-score groups are invariant to joint row permutations (also with nonuniform weights
and zero weights), and Gini agrees with `2*weighted AUC-1` on binary fixtures with ties.
Retain perfect/reversed controls and explicit nonfinite-input policy.

<a id="bug-044"></a>
## BUG-044 — Categorical rating exports omit and collide on routing metadata

**Priority:** P2. **Status:** Open.

**Source:** [serialize.rs:1004](../../crates/t-boost-core/src/serialize.rs#L1004)
defines exported levels with only `label` and `cell`; [line
1106](../../crates/t-boost-core/src/serialize.rs#L1106) substitutes the real string
`"<rare>"` for the internal rare bucket; [line
1158](../../crates/t-boost-core/src/serialize.rs#L1158) drops `CatLevel.members`, and
the axis has no unseen/default-cell field. Native routing uses original members and
encoder base at [cat.rs:280](../../crates/t-boost-core/src/cat.rs#L280).

**Contract and scope.** [README.md](../../README.md#L8) describes exact tables and
rating-table deployment; `TBoostRegressor.tables` at
[sklearn.py:6604](../../python/t_boost/sklearn.py#L6604) says exported table lookups
reconstruct `predict_raw`. Its detailed return documentation at [line
6685](../../python/t_boost/sklearn.py#L6685) explicitly describes **post-rare-pooling**
levels; it does not promise to list every original label. The demonstrated limitation is
that the exported artifact cannot independently reconstruct raw-category routing;
separate native model metadata is required but not carried or clearly required by the
artifact. Native model scoring and ordinary `to_json`/`to_bytes` retain the encoder and
are correct.

**Reproduction A.** Two ordinary fits produce identical parsed rating exports but
different predictions on the same raw category values, proving required information was
lost:

```python
import json
import numpy as np
import polars as pl
from t_boost import TBoostRegressor
opts = dict(n_trees=30, n_bags=1, prune=False, validation_fraction=None,
            n_jobs=1, graduate=False)
y = np.array([0]*100 + [2]*100 + [4]*2 + [6]*20, dtype=np.float32)
a = pl.DataFrame({'cat': ['a']*100 + ['b']*100 + ['rareA']*2 + [None]*20})
b = pl.DataFrame({'cat': ['a']*100 + ['b']*100 + ['rareB']*2 + [None]*20})
ma = TBoostRegressor(**opts).fit(a, y)
mb = TBoostRegressor(**opts).fit(b, y)
ea, eb = json.loads(ma.tables(a)), json.loads(mb.tables(b))
print(ea == eb)
print(ea['tables'][0]['axes'][0]['levels'])
query = pl.DataFrame({'cat': ['rareA', 'rareB', 'never-seen', None]})
print(ma.predict(query))
print(mb.predict(query))
```

```text
True
[{'label':'a','cell':1}, {'label':'b','cell':2},
 {'label':'<rare>','cell':3}, {'label':'__t_boost_missing__','cell':4}]
[3.99838114 1.99974251 1.99974251 5.99675703]
[1.99974251 3.99838114 1.99974251 5.99675703]
```

The exported artifact contains neither `rareA` nor `rareB`, and does not say which cell
is used for genuinely unseen levels. The native encoder distinguishes rare members from
unseen values.

**Reproduction B.** establishes ambiguity even within the documented displayed
label-to-cell list:

```python
import json
import numpy as np
import polars as pl
from t_boost import TBoostRegressor
x = pl.DataFrame({'c': ['<rare>']*100 + ['a']*100 + ['thin']*2})
y = np.array([0]*100 + [2]*100 + [8]*2, dtype=np.float32)
m = TBoostRegressor(n_trees=30, n_bags=1, prune=False,
    validation_fraction=None, n_jobs=1).fit(x, y)
print(json.loads(m.tables(x))['tables'][0]['axes'][0]['levels'])
print(m.predict(pl.DataFrame({'c': ['<rare>', 'thin']})))
```

```text
[{'label':'<rare>','cell':1}, {'label':'a','cell':2}, {'label':'<rare>','cell':3}]
[0.000526717806 7.95711708]
```

A real user level named `<rare>` and the synthetic bucket are indistinguishable in the
export but mean very different predictions. A consumer converting entries to a
label-keyed map selects the wrong one.

Verification 2026-10-02: both reproductions matched exactly; the exported axis carries
only `borders`, `cells`, `levels`, `name` and `raw`. Reproduction A is precisely what
the `tables` docstring (`sklearn.py:6684-6689`, "one per post-rare-pooling category
level (the reserved rare bucket shown as `<rare>`)") and `serialize.rs:996-1006`
describe, so on its own it is a documented limitation rather than a contradicted
contract. Reproduction B — a genuine level literally named `<rare>` — is addressed by
no documentation and is the concrete defect. `test_sql_export_contract.py:76-87` keys a
dictionary by `levels[].label`, exactly the consumer pattern that Reproduction B breaks.
The constant `RARE_LEVEL_EXPORT_LABEL` is at `serialize.rs:1107` (1106 is its comment).

**Suggested fix.** Export a complete unambiguous categorical routing contract: original
members, stable bucket identity/type separate from display labels, and unseen/missing
cell semantics. Alternatively explicitly identify the artifact as requiring the native
encoder, provide that encoder alongside it, and still eliminate duplicate display labels
masquerading as raw identities. Preserve compatibility/versioning deliberately.

**Acceptance checks.** An independent SQL/JSON-only scorer must match native scoring on
frequent labels, each pooled training member, genuinely unseen labels, missing values, a
literal `<rare>` label, multiple channels and banded axes. No need to inspect private
native encoders in the independent test. Existing categorical SQL-contract test reads
native `cell_indices`, so it cannot catch this information loss.

<a id="bug-045"></a>
## BUG-045 — Deviance metrics accept nonfinite inputs and an infinite Tweedie power

**Priority:** P2. **Status:** Open.

**Source:** [metrics.py:164](../../python/t_boost/metrics.py#L164) converts inputs and
checks lengths/power intervals without checking finiteness; [line
184](../../python/t_boost/metrics.py#L184) uses comparisons that do not reject NaN. Its
docstring promises `ValueError` for domain violations; current domain tests begin at
[test_metrics_deviance.py:56](../../python/tests/test_metrics_deviance.py#L56).

**Impact.** Invalid evaluation data silently produces NaN/infinity, which can flow into
score selection and reports. An infinite variance power is worse: finite unequal y/pred
can receive an apparently perfect zero deviance. This is a metrics-layer validation
defect, not BUG-026's classifier target validation and not BUG-038's core fit weights.

**Reproduction.**

```python
import numpy as np
from t_boost.metrics import mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance
print(mean_poisson_deviance([np.nan, 1.], [1., 1.]))
print(mean_gamma_deviance([1., 1.], [np.inf, 1.]))
print(mean_poisson_deviance([1., 1.], [1., 1.], [np.nan, 1.]))
print(mean_tweedie_deviance([2.], [5.], power=np.nan))
print(mean_tweedie_deviance([2.], [5.], power=np.inf))
```

```text
nan
inf
nan
nan
0.0
```

**Suggested fix.** Validate finite y, predictions, weights and power before mathematical
domain checks. Define invalid/zero-total weight behavior explicitly and raise a
consistent user-facing `ValueError`. Do not silently replace invalid data with zeros in
these deviance metrics (their contract differs from the ranking helpers).

**Acceptance checks.** NaN and ±inf in each vector and power are rejected for power 0,
1, 2 and representative Tweedie powers. Cover weighted/zero-weight inputs and ensure
finite valid cases still match the existing oracle. A perfect score must not arise
solely from an invalid power.

<a id="bug-046"></a>
## BUG-046 — A/E reports suppress valid ratios for negative expected totals

**Priority:** P2. **Status:** Open.

**Source:** [sklearn.py:3189](../../python/t_boost/sklearn.py#L3189) tests `expected >
0`; the [public docstring at line 3132](../../python/t_boost/sklearn.py#L3132) defines
`actual/expected` with NaN where expected is zero.

**Trigger.** A supported squared-error regressor predicts negative values. Negative
targets/predictions are valid for this objective and the reporting method is available
for generic regression.

**Reproduction.**

```python
import numpy as np
from t_boost import TBoostRegressor
x = np.arange(20, dtype=np.float32).reshape(-1, 1)
y = np.full(20, -3, dtype=np.float32)
m = TBoostRegressor(n_trees=5, n_bags=1, prune=False,
    validation_fraction=None, n_jobs=1).fit(x, y)
print(m.actual_vs_expected(x, y)[0])
print(m.pricing_report(x, y)['actual_vs_expected'][0]['ae'])
```

```text
actual: [0.0, -60.0]
expected: [0.0, -60.0]
rows: [0, 20]
ae: [nan, nan]
pricing_report ae: [None, None]
```

The populated cell has ratio 1, not an undefined ratio. This is distinct from BUG-017's
binary label coding; it uses a regressor and correct target values.

**Suggested fix.** Use a nonzero-denominator test for the documented generic ratio. If
the intended business contract disallows signed responses, reject that use explicitly
and document the restriction instead of silently marking perfectly defined ratios
missing.

**Acceptance checks.** Negative and positive expected totals, opposite-sign
actual/expected values, zero denominators, weighted aggregation, and JSON-null
conversion only for truly undefined ratios.

<a id="bug-047"></a>
## BUG-047 — Empty A/E evaluation batches raise an internal indexing error

**Priority:** P3. **Status:** Open.

**Source:** [binding lib.rs:5914](../../crates/t-boost-py/src/lib.rs#L5914) documents
`(n_rows,n_raw_features)` but calls `PyArray::from_vec2` on an empty vector.
[sklearn.py:3182](../../python/t_boost/sklearn.py#L3182) indexes missing feature
columns; the next line explicitly anticipates empty aggregation.

**Reproduction.**

```python
import numpy as np
from t_boost import TBoostRegressor
x = np.arange(20, dtype=np.float32).reshape(-1, 1)
m = TBoostRegressor(n_trees=5, n_bags=1, prune=False,
    validation_fraction=None, n_jobs=1).fit(x, x[:, 0])
print(m.predict(x[:0]).shape)
print(m.actual_vs_expected(x[:0], x[:0, 0]))
```

```text
(0,)
IndexError: index 0 is out of bounds for axis 1 with size 0
```

**Cause.** Native empty cell indices have shape `(0,0)` instead of `(0,n_raw_features)`,
despite the documented shape. A filtered evaluation set with zero matching rows can
therefore break report generation while prediction itself handles it.

**Suggested fix.** Construct the correctly shaped empty native array explicitly, as the
probability/raw-score binding already does for empty batches; then return a consistent
empty A/E report or explicitly reject emptiness with `ValueError`.

**Acceptance checks.** Empty numeric and categorical designs, several feature counts,
loaded models, direct `cell_indices` shape, and `pricing_report`. Avoid treating a
zero-row batch as a zero-feature model.

<a id="bug-048"></a>
## BUG-048 — OOB cell correction silently breaks monotonicity in the Rust API

**Priority:** P2. **Status:** Open.

**Source:** [engine/boost.rs:3091](../../crates/t-boost-core/src/engine/boost.rs#L3091)
(`attach_cell_correction`), [attachment at line
3190](../../crates/t-boost-core/src/engine/boost.rs#L3190), and
[cell_refit.rs:493](../../crates/t-boost-core/src/cell_refit.rs#L493).

**Trigger and impact.** A Rust caller combines `FitSpec.monotone` with
`EnsembleSpec::OuterBag { cell_refit: Some(..), .. }`. The trees obey the requested
constraint, but the unconstrained correction bank can reverse it. The returned model
therefore violates an explicit prediction contract on ordinary finite input.

**Cause.** The correction solve optimizes per-cell residuals without monotonicity
constraints, then attaches its values to the constrained tree ensemble. The no-harm
guard checks deviance, not monotonicity, and `Model::validate` does not carry or enforce
the original monotone specification. Neither fit-spec validation nor outer-bag
validation rejects this incompatible combination. Python already rejects the scalar
combination at [sklearn.py:3525](../../python/t_boost/sklearn.py#L3525), so this finding
concerns the public Rust fitting API. Python multiclass fitting rejects monotone
constraints entirely; that rejection was also verified.

Verification 2026-10-02: all four rows reproduced exactly (40 trees in each fit).
`cell_refit.rs` contains no reference to monotone constraints; `Model::validate`
(`engine/mod.rs:962`) has no monotone field and `EnsembleSpec::validate`
(`boosters.rs:198-230`) checks only `base`/`gamma` finiteness. The strongest evidence
that the core treats monotonicity as a post-refit invariant is the *ridge* refit path:
`boost.rs:4769` projects refit leaves back onto the monotone cone and the test
`monotone_holds_under_ridge_refit` (`boost.rs:12081`) pins it. Cell refit is therefore
the one leaf-rewriting stage that does not honour the constraint, which argues for the
"mirror the Python check in core" fix at minimum.

**Reproduction.** The complete Rust harness below uses 4,000 rows: every `(x,z)` pair in
`0..19 × 0..19` repeated ten times. The response is `2*x + .4*z` for `z<10`, otherwise
`-x + .4*z`; constrain `x` to be increasing. Fit four bags at `.8` subsampling, ten
trees per bag, learning rate `.2`, lambda `1`, and no leaf refinement. Compare the same
fit with and without the correction:

```text
cellrefit=None:       correction=false, minimum adjacent-x delta=0
cellrefit=base .01:   correction=true,  minimum adjacent-x delta=-3.3988252
cellrefit=base 1:     correction=true,  minimum adjacent-x delta=-4.067876
cellrefit=base 100:   correction=true,  minimum adjacent-x delta=-0.61279714
```

All corrections use `gamma=0`. At `z=15`, the `base=.01` model predicts `4.0003543` for
`x=0` and `-10.836588` for `x=19`, despite the increasing constraint. The no-correction
control predicts `-1.8932799` and `-1.3445102`, respectively. The negative adjacent
deltas compare rows with exactly the same second feature; they are not a marginal or
averaging artifact.

**Suggested fix.** Mirror the existing Python incompatibility check in core fitting, or
implement a joint correction constrained to preserve the final ensemble's monotonicity.
Do not rely on the deviance guard or unconstrained cell validation to guarantee it.

**Acceptance checks.** Exercise increasing/decreasing constraints with realized pair
effects and accepted cell correction. Either reject the combination before fitting or
verify the final predictions retain the signs over every supported feature slice.
Preserve unconstrained correction behavior and constrained no-correction controls.


### Complete Rust reproduction

```rust
use t_boost_core::boosters::{BoosterConfig, CellRefit, EnsembleSpec};
use t_boost_core::constraints::{CredibilityFloor, InteractionPolicy, MonoSign};
use t_boost_core::data::{bin_columns, BinConfig};
use t_boost_core::engine::{Booster, Config, FitSpec};
use t_boost_core::loss::SquaredError;
fn main() {
    let n = 4000;
    let xc = (0..n).map(|i| (i / 200) as f32).collect::<Vec<_>>();
    let zc = (0..n).map(|i| ((i / 10) % 20) as f32).collect::<Vec<_>>();
    let y = xc
        .iter()
        .zip(&zc)
        .map(|(&x, &z)| x * if z < 10. { 2. } else { -1. } + z * 0.4)
        .collect::<Vec<_>>();
    let x = bin_columns(&[&xc, &zc], None, &BinConfig::default(), 0).unwrap();
    let loss = SquaredError;
    let spec = FitSpec {
        loss: &loss,
        weight: None,
        exposure: None,
        monotone: [("f0".into(), MonoSign::Increasing)].into(),
        interaction: InteractionPolicy::default(),
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed: 0,
    };
    for base in [None, Some(0.01), Some(1.), Some(100.)] {
        let cfg = Config {
            n_trees: 10,
            learning_rate: 0.2,
            lambda: 1.,
            leaf_refine_steps: 0,
            boosters: BoosterConfig {
                ensemble: EnsembleSpec::OuterBag {
                    n_bags: 4,
                    bag_subsample: 0.8,
                    cell_refit: base.map(|base| CellRefit { base, gamma: 0. }),
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let m = Booster::with_config(cfg).fit(&x, &y, &spec).unwrap();
        let pred = m.predict(&x, None).unwrap();
        let min_delta = (0..n - 200)
            .map(|i| pred[i + 200] - pred[i])
            .fold(0., f32::min);
        println!("cellrefit={base:?} correction={} trees={} minimum_delta={min_delta} x0z15={} x19z15={}",m.correction.is_some(),m.trees.len(),pred[150],pred[3950]);
    }
}
```

<a id="bug-049"></a>
## BUG-049 — Valid weighted fits fail when a bag omits all positive-weight observations

**Priority:** P2. **Status:** Open.

**Source:** [engine/boost.rs:2887](../../crates/t-boost-core/src/engine/boost.rs#L2887)
(subagging draw), [row subset and fit at line
2902](../../crates/t-boost-core/src/engine/boost.rs#L2902), [weight-blind dispatch at
line 3719](../../crates/t-boost-core/src/engine/boost.rs#L3719).

**Trigger and impact.** A valid regression dataset has finite nonnegative sample weights
and positive total mass, but only a few observations carry that mass. Outer bags sample
row identities without considering effective weight. A bag can contain only zero-weight
rows; its intercept initialization fails and aborts the entire estimator fit. Success
then depends on the random seed and bag count even for a trivial constant effective
target. The shipped default of eight bags can reach this failure.

**Cause.** No stage ensures each selected bag has positive training weight, retries a
zero-mass draw, or removes irrelevant zero-weight rows before sampling. Scalar
initialization correctly rejects an all-zero *bag*, but that internal bag was generated
by the library from valid input. This is distinct from a caller providing globally
all-zero weights and from BUG-039's pre-training failure on an uninformative feature
containing NaNs.

**Reproduction.** All features here are ordinary finite numbers; there are no missing
values. Only row 17 has positive weight:

```python
import numpy as np
from t_boost import TBoostRegressor

X = np.arange(100, dtype=np.float32).reshape(-1, 1)
y = np.arange(100, dtype=np.float32)
w = np.zeros(100, dtype=np.float32)
w[17] = 1
for bags in [1, 2, 8]:
    for seed in [0, 1, 2]:
        try:
            model = TBoostRegressor(
                n_trees=3, n_bags=bags, prune=False, graduate=False,
                validation_fraction=None, n_jobs=1, seed=seed,
            ).fit(X, y, sample_weight=w)
            print(bags, seed, "ok", model.predict(X)[0])
        except ValueError as error:
            print(bags, seed, type(error).__name__, str(error))
```

```text
bags=1 seeds=0,1,2: succeeds, predicts17
bags=2 seed=0: fails; seeds=1,2: succeed, predict17
bags=8 seeds=0,1: fail; seed=2: succeeds, predicts17
error: TBoostValueError invalid input: squared-error init_score: all-zero (or non-positive) weights
```

**Suggested fix.** Define a deterministic policy for empty effective bag samples: sample
from positive-weight rows where compatible with group semantics, ensure positive mass
during sampling, or retry/fall back without changing the public dataset's validity.
Apply equivalent protection after any internal validation carve; class-specific
positive-mass requirements may need additional checks for classification.

**Acceptance checks.** Fit this fixture across several seeds and bag counts, including
bootstrap and grouped sampling. Positive-weight input should not fail merely because of
internally generated zero-mass bags. Globally all-zero weights should continue to fail.
No missing-value/binning changes are needed to expose or repair this defect.

<a id="bug-050"></a>
## BUG-050 — The cell-refit guard splits declared groups between fitting and validation

**Priority:** P2. **Status:** Open.

**Source:** [engine/boost.rs:3148](../../crates/t-boost-core/src/engine/boost.rs#L3148)
(row-hash holdout), [correction fit weights at line
3152](../../crates/t-boost-core/src/engine/boost.rs#L3152), [guard rows at line
3204](../../crates/t-boost-core/src/engine/boost.rs#L3204). The multiclass equivalent
repeats the row hash at [line
1799](../../crates/t-boost-core/src/engine/boost.rs#L1799).

**Trigger and impact.** Enable OOB cell refitting while providing repeated entity/group
IDs. Outer-bag sampling keeps each entity entirely in or out of each bag, but the
*second-stage correction* divides that entity's rows between its own fit and no-harm
guard. The guard is therefore not evaluating on held-out entities and can select the
correction using information from group-mates it fitted. This undermines the
grouped-validation guarantee for that stage, even though the base OOB prediction remains
group-honest.

**Contract and scope.** Public `fit(groups=...)` describes group-honest internal carves
([sklearn.py:6464](../../python/t_boost/sklearn.py#L6464)); `cell_refit_base` promises
held-out no-harm guarding ([line 5859](../../python/t_boost/sklearn.py#L5859)). The
implementation itself says the 15% slice makes the correction use honest data
(boost.rs:3146–3147). The narrower guarantee that a base bag never contains a group-mate
of an OOB row *does hold* in this reproduction. This finding is specifically the ignored
group boundary in the correction-fit/guard split, not a claim that the bag sampling
implementation violates its own boundary, and not a quantified predictive-accuracy
regression.

**Cause.** `attach_cell_correction` has access to `spec.bag_groups`, but selects guard
membership purely from a hash of the row index. The multiclass attachment does not
receive group IDs at all. Both can split all repeated entities despite having
group-honest bag memberships available.

**Reproduction.** The complete Rust harness records the actual target rows passed to the
loss's guard evaluations. There are 1,000 rows, 100 groups of ten consecutive rows each,
feature equal to group ID, and a unique target equal to original row index so the
callback identifies every observation. Fit four group-sampled bags at `.8`, five trees,
`.2` learning rate, and `CellRefit {base:1,gamma:0}`. All rows have unit weight and no
other holdout or leaf-refinement loss calls occur.

```text
deviance_calls=5
correction_kept=true
covered_rows=560
correction_fit_rows=473
guard_rows=87
guard_groups=56
guard_groups_also_fit=56
any_bag_splits_group=false
observed_guard_matches_production_hash=true
```

Thus every one of the 56 guard groups also has rows used to solve the correction. The
callback's observed guard equals the production hash predicate exactly, and the
complement among OOB-covered unit-weight rows is exactly the correction's
positive-weight fit set. Runtime instrumentation covered scalar fitting; the same
multiclass hash was established by source review.

**Suggested fix.** When groups are present, choose whole groups for the correction guard
and exclude every member of those groups from the correction solve. Thread group IDs
into the multiclass attachment. Keep row hashing for ungrouped data and define a clean
fallback when too few covered groups remain for a guard.

**Acceptance checks.** Assert that original group IDs in the correction-fit and guard
sets are disjoint, separately from checking each base bag's in-bag/OOB disjointness.
Cover scalar/multiclass, unequal group sizes, group relabeling, and limited OOB
coverage.


### Complete Rust reproduction

```rust
use std::collections::BTreeSet;
use std::sync::Mutex;
use t_boost_core::boosters::{BoosterConfig, CellRefit, EnsembleSpec};
use t_boost_core::constraints::{CredibilityFloor, InteractionPolicy};
use t_boost_core::data::{bin_columns, BinConfig};
use t_boost_core::engine::{Booster, Config, FitSpec};
use t_boost_core::loss::{GradHess, Link, Loss, Metric, ObjectiveTag, SquaredError};
use t_boost_core::PbError;
struct Recorded(Mutex<Vec<Vec<f32>>>);
impl Loss for Recorded {
    fn grad_hess(
        &self,
        y: &[f32],
        r: &[f32],
        w: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError> {
        SquaredError.grad_hess(y, r, w, out)
    }
    fn init_score(&self, y: &[f32], w: &[f32], o: Option<&[f32]>) -> Result<f64, PbError> {
        SquaredError.init_score(y, w, o)
    }
    fn link(&self) -> Link {
        Link::Identity
    }
    fn pred_from_raw(&self, r: f32) -> f32 {
        r
    }
    fn deviance(&self, y: &[f32], r: &[f32], w: &[f32]) -> Result<f32, PbError> {
        self.0.lock().unwrap().push(y.to_vec());
        SquaredError.deviance(y, r, w)
    }
    fn default_metric(&self) -> Metric {
        Metric::Rmse
    }
    fn objective_tag(&self) -> ObjectiveTag {
        SquaredError.objective_tag()
    }
}
fn main() {
    let n = 1000;
    let groups = (0..n).map(|i| (i / 10) as u32).collect::<Vec<_>>();
    let col = groups.iter().map(|&g| g as f32).collect::<Vec<_>>();
    let y = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let x = bin_columns(&[&col], None, &BinConfig::default(), 0).unwrap();
    let loss = Recorded(Mutex::new(vec![]));
    let spec = FitSpec {
        loss: &loss,
        weight: None,
        exposure: None,
        monotone: Default::default(),
        interaction: InteractionPolicy::default(),
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: Some(&groups),
        seed: 0,
    };
    let m = Booster::with_config(Config {
        n_trees: 5,
        learning_rate: 0.2,
        leaf_refine_steps: 0,
        boosters: BoosterConfig {
            ensemble: EnsembleSpec::OuterBag {
                n_bags: 4,
                bag_subsample: 0.8,
                cell_refit: Some(CellRefit {
                    base: 1.,
                    gamma: 0.,
                }),
            },
            ..Default::default()
        },
        ..Default::default()
    })
    .fit(&x, &y, &spec)
    .unwrap();
    let events = loss.0.lock().unwrap();
    let guard_rows = events
        .last()
        .unwrap()
        .iter()
        .map(|v| *v as usize)
        .collect::<BTreeSet<_>>();
    let bags = m.bag_in_bag.as_ref().unwrap();
    let covered = (0..n)
        .filter(|&r| bags.iter().any(|b| !b[r]))
        .collect::<BTreeSet<_>>();
    let fit_rows = covered
        .difference(&guard_rows)
        .copied()
        .collect::<BTreeSet<_>>();
    let guard_groups = guard_rows
        .iter()
        .map(|&r| groups[r])
        .collect::<BTreeSet<_>>();
    let fit_groups = fit_rows.iter().map(|&r| groups[r]).collect::<BTreeSet<_>>();
    let bags_split_group = bags
        .iter()
        .any(|b| (0..100).any(|g| (0..10).any(|i| b[g * 10 + i] != b[g * 10])));
    println!("deviance_calls={} correction_kept={} covered_rows={} correction_fit_rows={} guard_rows={} guard_groups={} guard_groups_also_fit={} any_bag_splits_group={}",events.len(),m.correction.is_some(),covered.len(),fit_rows.len(),guard_rows.len(),guard_groups.len(),guard_groups.intersection(&fit_groups).count(),bags_split_group);
    let hash_guard = covered
        .iter()
        .filter(|&&r| ((r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) < 38)
        .copied()
        .collect::<BTreeSet<_>>();
    println!(
        "observed_guard_matches_production_hash={}",
        guard_rows == hash_guard
    );
}
```

<a id="bug-051"></a>
## BUG-051 — Table-budget checks do not cover purification and factored shedding

**Priority:** P2. **Status:** Open.

**Source:** [explain.rs:1705](../../crates/t-boost-core/src/explain.rs#L1705) skips raw
allocation for factored supports; [line
1759](../../crates/t-boost-core/src/explain.rs#L1759) budgets directly accumulated dense
tables. [Line 2165](../../crates/t-boost-core/src/explain.rs#L2165) allocates new dense
subsets during purification and [line
2205](../../crates/t-boost-core/src/explain.rs#L2205) allocates their support tensors.
[Line 2002](../../crates/t-boost-core/src/explain.rs#L2002) expands sparse tables
without a density check.

**Trigger/impact.** A caller supplies explicit `TableBudget` limits to
`Model::explain_with_budget`. The returned bank can exceed the whole-bank ceiling, and
factored effects can create dense main/pair tables above the per-table ceiling. This
defeats the documented memory firewall. The counterexamples use at most 36 dense value
cells; they establish the missing enforcement without exhausting memory.

**Cause.** The budget and counter live only in raw tree accumulation. Later exact
decomposition creates missing subset tables and densifies sparse residuals, but those
paths receive no budget. A support sent to the factored path bypasses the raw counter
entirely, although shedding subsequently creates dense subsets.

**Minimal whole-bank reproduction:**

```rust
use t_boost_core::explain::{
    fixture_model, fixture_serve, OverflowPolicy, RefMeasure, TableBudget,
};

fn main() {
    let model = fixture_model();
    let x = fixture_serve();
    for cap in [8, 9, 14, 15] {
        let got = model.explain_with_budget(
            &x,
            RefMeasure::Uniform,
            TableBudget {
                max_table_cells: 9,
                max_bank_cells: cap,
                on_overflow: OverflowPolicy::Error,
            },
        );
        println!(
            "cap {cap}: {:?}",
            got.map(|b| b.tables.iter().map(|t| t.values.len()).sum::<usize>())
        );
    }
}
```

Observed:

```text
cap 8: Err(TableBudget { what: "bank", cells: 9, budget: 8 })
cap 9: Ok(15)
cap 14: Ok(15)
cap 15: Ok(15)
```

The raw pair contains 9 cells; the returned pair plus two generated mains contains 15.
The public `TableBudget.max_bank_cells` documentation says it limits the sum over all
tables.

**Factored/per-table variant:**

```rust
use t_boost_core::{
    data::{AxisKind, AxisProvenance, FeatureId},
    engine::Split,
    explain::{fixture_model, fixture_serve, OverflowPolicy, RefMeasure, TableBudget},
    table_model::TableModel,
};

fn main() {
    let mut m = fixture_model();
    m.grids.push(m.grids[0].clone());
    m.provenance.push(AxisProvenance {
        raw: FeatureId(2),
        kind: AxisKind::Numeric,
    });
    m.schema.feature_names.push("x2".into());
    m.schema.feature_kinds.push(AxisKind::Numeric);
    m.trees[0].1.depth = 3;
    m.trees[0].1.splits.push(Split {
        axis: 2,
        bin_le: 1,
        missing_left: false,
    });
    m.trees[0].1.leaves = vec![0., 0., 0., 0., 0., 0., 0., 8.];
    let mut x = fixture_serve();
    x.0.data.push(vec![1, 2, 1, 2]);
    x.0.grids = m.grids.clone();
    x.0.provenance = m.provenance.clone();
    m.validate().unwrap();
    let b = m
        .explain_with_budget(
            &x,
            RefMeasure::Uniform,
            TableBudget {
                max_table_cells: 1,
                max_bank_cells: 1,
                on_overflow: OverflowPolicy::Factored,
            },
        )
        .unwrap();
    println!(
        "cells {:?}",
        b.tables.iter().map(|t| t.values.len()).collect::<Vec<_>>()
    );
    println!(
        "valid {:?}",
        TableModel::from_model_and_bank(&m, b).validate()
    );
}
```

Observed dense sizes `[3, 9, 9, 3, 9, 3]`, total **36**, plus one factored effect. Model
validation succeeds despite both limits being **1**.

**Related sparse-density variant, same missing later-stage enforcement.** Replace the
pair fixture's leaf vector with `[0,0,0,9,0,0,0,0]` and use `max_table_cells=1`,
`max_bank_cells=100`, `SparseFallback { density_threshold: 0.2 }`. Raw occupancy is one
of nine cells and passes its cap. The returned pair remains sparse-backed but contains
**9/9 nonzero cells**, five times the allowed density; generated mains are dense.
Purification can introduce entries into previously zero cells without enforcing the
sparse density threshold.

**Suggested fix.** Thread budget accounting through the complete decomposition,
including generated subsets and sparse insertions. Decide and enforce the
memory-accounting policy for support tensors as well. Check before the relevant
allocation/expansion, rather than only rejecting an oversized finished bank. Keep sparse
threshold `1.0`'s documented permissive behavior while enforcing restrictive thresholds
after purification changes occupancy.

**Acceptance checks.** The examples above must return typed budget errors at ceilings
below their dense requirements; sufficient budgets must preserve predictions. Cover
direct dense, factored-to-dense subset creation, sparse residual densification, and
exact boundary values. Larger tests can extrapolate counts without allocating excessive
memory.

<a id="bug-052"></a>
## BUG-052 — Public exactness assertion ignores the mass used to build weighted banks

**Priority:** P2. **Status:** Open.

**Source:** [explain.rs:4408](../../crates/t-boost-core/src/explain.rs#L4408) calls
`build_weights(..., None)` unconditionally. [Weighted construction at line
4511](../../crates/t-boost-core/src/explain.rs#L4511) uses the supplied mass and its
matching internal gate path.

**Trigger/impact.** Construct a valid bank with `Model::explain_weighted` under
`ProductMarginals` or `ExposureMarginals`, then call public `assert_exact_decomposition`
with that bank and its original serve matrix. The assertion incorrectly reports
`MassConservation`, despite weighted construction having already passed the genuine
weighted checks and the table sum reconstructing the model exactly. This is a **public
diagnostic API failure**, not evidence that `explain_weighted`'s internal production
gates use the wrong mass or that predictions change.

**Cause.** The assertion reconstructs reference weights from unweighted row counts and
the measure kind. The actual bank was purified under different weighted marginals. There
is no mass parameter or weighted assertion counterpart in the public API.

Verification 2026-10-02: reproduced exactly for all three measures. Two added controls
isolate the failure to nonuniform mass: `explain_weighted` with unit weights
`[1, 1, 1, 1]` passes the assertion under every measure, and unweighted `explain` passes
under every measure. The assertion's own docstring (`explain.rs:4395-4397`) says it
"rebuilds the reference weights from `x` … and `bank.w`", i.e. the row-count rebuild is
documented, and `explain_weighted`'s docstring (`4488-4500`) describes the weighted
measure as "a different valid decomposition" by design. This is therefore an API gap —
a documented unweighted check with no weighted counterpart — rather than a contradicted
contract; the entry's own "public diagnostic API failure" framing is accurate.

**Complete reproduction:**

```rust
use t_boost_core::{
    assert_exact_decomposition,
    explain::{check_reconstruction, fixture_model, fixture_serve, RefMeasure},
};

fn main() {
    let model = fixture_model();
    let x = fixture_serve();
    for w in [
        RefMeasure::Uniform,
        RefMeasure::default(),
        RefMeasure::ExposureMarginals { floor: 0.001 },
    ] {
        let bank = model
            .explain_weighted(&x, w.clone(), &[1., 1., 1., 100.])
            .unwrap();
        println!(
            "{w:?}: reconstruction {:?}; assertion {:?}",
            check_reconstruction(&model, &bank),
            assert_exact_decomposition(&model, &bank, &x)
        );
    }
}
```

Observed: reconstruction passes for all three measures; the public assertion passes for
Uniform but raises `InvariantViolated { invariant: MassConservation }` for both weighted
measures.

**Suggested fix.** Provide a weighted assertion path and use the same mass as
construction, or recover the actual reference weights from reliable bank metadata.
Document the distinction if the existing function intentionally remains unweighted; do
not present its output as verification of any arbitrary weighted bank.

**Acceptance checks.** Weighted explained banks must pass the matching public assertion
for nonuniform sample weights/exposure. Include unit weights, zero-weight rows,
alternate positive floors, and a deliberately corrupted weighted bank that still fails
the appropriate invariant.

<a id="bug-053"></a>
## BUG-053 — Variance certification loses precision after harmless intercept translation

**Priority:** P2. **Status:** Open.

**Source:** [explain.rs:4285](../../crates/t-boost-core/src/explain.rs#L4285) computes
exhaustive variance as `E[F²] - E[F]²`; [line
4350](../../crates/t-boost-core/src/explain.rs#L4350) does the same for sampled
variance, and [line 4362](../../crates/t-boost-core/src/explain.rs#L4362) compares it
with the table variance sum. The exhaustive tolerance is absolute and depends on tree
count/order, not score scale.

**Trigger/impact.** A finite validated model with ordinary variation around a large
intercept cannot be explained or converted with the public gated table constructor:
`Model::explain` returns `VarianceSum`. Adding a constant does not change true variance
or per-effect values. This is a **production core explain/export gate false rejection**,
separate from the weighted public assertion issue above.

**Cause.** Subtracting two large, nearly equal uncentered moments destroys the small
variance. The actual table decomposition remains correct: independently run
reconstruction and three-way equality both pass. Merely loosening tolerance would
conceal the numerical calculation problem; center before accumulating moments or use a
stable deterministic variance algorithm.

**Complete primary reproduction:**

```rust
use t_boost_core::{
    assert_exact_decomposition,
    explain::{
        check_reconstruction, check_three_way_equal, fixture_model, fixture_serve, ExactTol,
        RefMeasure,
    },
};

fn main() {
    let original = fixture_model();
    let x = fixture_serve();
    let mut shifted = original.clone();
    shifted.f0 = 10_000_000.0;
    shifted.validate().unwrap();
    let mut bank = original.explain(&x, RefMeasure::Uniform).unwrap();
    bank.f0 += 10_000_000.0; // Exact constant translation; table values unchanged.
    println!("reconstruct {:?}", check_reconstruction(&shifted, &bank));
    println!("threeway {:?}", check_three_way_equal(&shifted, &bank));
    println!(
        "assert {:?}",
        assert_exact_decomposition(&shifted, &bank, &x)
    );
    println!(
        "explain {:?}",
        shifted.explain(&x, RefMeasure::Uniform).map(|_| ())
    );
    let (mut m1, mut m2, mut centered, mut mass) = (0., 0., 0., 0.);
    for i in 0..3u8 {
        for j in 0..3u8 {
            let v = shifted.ensemble_f64(&[i, j]).unwrap();
            let w = (1. / 3.) * (1. / 3.);
            m1 += w * v;
            m2 += w * v * v;
            centered += w * (v - bank.f0).powi(2);
            mass += w;
        }
    }
    println!(
        "uncentered {}, centered {}, table {}, tolerance {}",
        m2 / mass - (m1 / mass).powi(2),
        centered / mass,
        bank.tables.iter().map(|t| t.variance).sum::<f64>(),
        ExactTol::for_model(&shifted).var_tol
    );
}
```

Observed:

```text
model validation: Ok
reconstruction: Ok
three-way equality: Ok
assert_exact_decomposition: VarianceSum error
Model::explain: VarianceSum error
uncentered variance: 3.375
centered variance: 3.358024691358023
sum of table variances: 3.358024691358024
absolute tolerance: 0.00000762939453125
```

The original unshifted model explains successfully. Changing only its tree coefficient
from `1` to `100000` also makes explanation fail, showing separate sensitivity to scale
as well as translation. (Verification 2026-10-02: both reproduced exactly. A sweep
shows the gate trips only at these magnitudes — coefficients `10`, `100`, `1000`,
`10000` and offsets `10` through `1e6` all explain successfully; `var_tol` stays at
`7.63e-6` throughout because it scales with tree count, not score scale. The
`ExactTol` doc at `explain.rs:855-857` says a false alarm on this gate "is not
harmless … refusing to export a rating table that is in fact exact".)

**Scope check.** An ordinary Python `TBoostRegressor.fit` followed by `.tables` did
**not** raise this gate on the tested large-offset/scaled dataset: the deployed Python
table path skips these ensemble-anchored checks. This is not a demonstrated default
Python fit failure. The confirmed affected entry points are core `Model::explain`,
`TableModel::from_model`, and public exactness certification.

**Suggested fix.** Compute centered weighted moments deterministically in both
exhaustive and sampled branches, and review whether remaining floating-point
certification error needs scale-aware bounds. Keep all real invariant failures
detectable.

**Acceptance checks.** Exact decompositions should pass under constant translations and
moderate scalar multiplication across numeric/categorical, dense/factored,
exhaustive/sampled cases. Compare against a stable centered oracle. Deliberately corrupt
a table or variance field to verify the gate still rejects actual discrepancies.

<a id="bug-054"></a>
## BUG-054 — Incremental Poisson scores lose the distance beyond the exponent clamp

**Priority:** P2. **Status:** Open.

**Source:** [loss.rs:1257](../../crates/t-boost-core/src/loss.rs#L1257) seeds and
refreshes the inverse-link cache from `clamp_exp(raw)`;
[boost.rs:7287](../../crates/t-boost-core/src/engine/boost.rs#L7287) multiplies it by
each leaf's exponential. [loss.rs:1284](../../crates/t-boost-core/src/loss.rs#L1284)
then clamps the cached value again for gradients. Refresh is only periodic at
[boost.rs:921](../../crates/t-boost-core/src/engine/boost.rs#L921). The Python option
promises only small numerical drift at
[sklearn.py:5830](../../python/t_boost/sklearn.py#L5830).

**Trigger and impact.** Enable `incremental_mu=True` for a Poisson fit whose raw scores
are outside `[-30,30]` at initialization or a refresh. The cache starts from the
saturated value, discarding how far the raw score lies beyond the boundary. Multiplying
that value by subsequent updates no longer represents the clamped exponential of the
updated raw score. The resulting gradients can materially change the fitted model,
rather than exhibiting the documented approximately `1e-9` to `1e-6` drift. This is an
optional numerical edge case involving large but finite targets; the default option
remains off.

**Cause.** In general, `exp(clamp(F)) * exp(delta)` followed by clamping is not
`exp(clamp(F + delta))`. For example, if `F > 30` and a modest negative update leaves `F
+ delta > 30`, the exact inverse link remains at `exp(30)`, but the cache decreases
immediately. The same loss of distance occurs below `-30`. Periodic refresh cannot undo
the incorrect gradients and retained trees produced between refreshes. This differs from
BUG-040: no update needs to disappear through float32 rounding, and the error exists in
exact arithmetic because clipping does not commute with multiplication.

**Reproduction.** The public estimator below isolates the cache from leaf refinement,
reanchoring, pruning, and graduation:

```python
import numpy as np
from t_boost import TBoostRegressor

x = np.repeat(np.array([0, 1], dtype=np.float32), 100).reshape(-1, 1)
y = np.repeat(np.array([1e12, 7.9e13], dtype=np.float32), 100)
for n_trees in [20, 40]:
    predictions = []
    for incremental in [False, True]:
        model = TBoostRegressor(
            objective="poisson", n_trees=n_trees, n_bags=1,
            prune=False, graduate=False, band_tolerance=None,
            validation_fraction=None, n_jobs=1, learning_rate=0.1,
            leaf_refine_steps=0, reanchor=False, incremental_mu=incremental,
        ).fit(x, y)
        pred = model.predict(x)
        predictions.append(pred)
        print(n_trees, incremental, pred[[0, -1]], model.predict_raw(x)[[0, -1]])
    print("max relative difference",
          np.max(np.abs(predictions[0] - predictions[1]) / predictions[0]))
```

```text
20 False: first prediction 9.863882080256e12; first raw 29.91990089
20 True:  first prediction 9.952799227904e12; first raw 29.92887497
max relative difference: 0.0090144171
40 False: first prediction 2.476328091648e12; first raw 28.53779793
40 True:  first prediction 4.531199410176e12; first raw 29.14200783
max relative difference: 0.8298057618
```

Both configurations keep the second group's predictions at the existing exponent
ceiling. The finding does not demand removing that documented ceiling or claim either
configuration can fit arbitrarily large counts. It is the 83% discrepancy on the first
group's predictions, which are below the ceiling, caused by a purported speed-only
option. At two and ten trees the controls still agree because the negative updates
remain at the leaf-step cap; the difference emerges as cached gradients leave that
regime.

**Suggested fix.** Retain enough state to account for the raw distance beyond either
clamp, or use exact recomputation for saturated rows until their raw scores reenter the
interior. Make seeding, refresh, and incremental updates obey the same saturation
contract. Avoid replacing the current cache with an unbounded exponential that can
overflow for valid finite raw scores.

**Acceptance checks.** Compare cached and exact gradients after updates that remain
beyond, enter, leave, and cross both clamp boundaries. Exercise initialization and the
64-round refresh boundary, nonuniform exposure, and the public fixture above. Retain the
existing ordinary-range drift tests.

<a id="bug-055"></a>
## BUG-055 — Runtime-only score methods silently broadcast or flatten incompatible targets

**Priority:** P2. **Status:** Open.

**Source:** [python/t_boost/_compat.py:85](../../python/t_boost/_compat.py#L85)
implements the fallback regression score; [line 97](../../python/t_boost/_compat.py#L97)
implements classification accuracy. Both flatten targets and perform
arithmetic/comparison without validating their original shape or row count.

**Trigger and impact.** Use the supported installation without optional scikit-learn and
call `score` with a target whose number of rows differs from `X`. A one-element target
broadcasts over every prediction, returning an apparently legitimate metric. A
two-dimensional target can be flattened into a matching-length but incorrectly ordered
vector and silently scored as well. Installing the optional dependency changes these
cases into explicit validation errors. This can hide an evaluation-data alignment error
in a deployment environment that was caught during development.

**Reproduction.** Save this as a script and execute it twice, once normally and once
with `--block`. The import blocker is the same testing technique used by the
repository's no-scikit-learn suite; it does not alter estimator or metric code.

```python
import sys

if "--block" in sys.argv:
    class BlockSklearn:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "sklearn" or fullname.startswith("sklearn."):
                raise ImportError("simulate runtime-only installation")
    sys.meta_path.insert(0, BlockSklearn())

import numpy as np
from t_boost import TBoostClassifier, TBoostRegressor
from t_boost._compat import SKLEARN_AVAILABLE

print("sklearn", SKLEARN_AVAILABLE)
x = np.arange(20, dtype=np.float32).reshape(-1, 1)
options = dict(n_trees=5, n_bags=1, prune=False, graduate=False,
               validation_fraction=None, n_jobs=1)
for model, y in [
    (TBoostRegressor(**options), np.zeros(20)),
    (TBoostClassifier(**options), np.tile([0, 1], 10)),
]:
    model.fit(x, y)
    for design, target in [
        (x, np.array([0.])),
        (x[:1], y[:1]),
        (x, np.column_stack([y, y])[:10]),
    ]:
        try:
            print(type(model).__name__, design.shape, target.shape,
                  model.score(design, target))
        except ValueError as error:
            print(type(model).__name__, type(error).__name__, str(error))
```

**Observed.** With scikit-learn present, both estimators reject the one-element target
for 20 rows and the `(10,2)` target for 20 rows. With it blocked, the regression scores
are `1.0` for both invalid inputs, and the classifier returns `0.85` and `0.45`,
respectively. A related parity gap appears for valid single-row regression evaluation:
the fallback returns `1.0`, whereas scikit-learn reports undefined R² (`NaN` with a
warning). These are all deficiencies of the same fallback scoring contract; they are not
separate findings or a repeat of BUG-026's classifier fit-label validation.

**Suggested fix.** Validate target dimensionality and row alignment before any
flattening or broadcasting, and validate score weights consistently. Implement the
documented metric conventions for insufficient sample counts. Keep the dependency
optional by performing these checks locally.

**Acceptance checks.** Run a parity matrix with and without scikit-learn: correct
vectors, accepted column vectors, one-value mismatches, incompatible multioutput shapes,
one-row R², and mismatched weights. Invalid alignment must raise consistently rather
than return a plausible score.

<a id="bug-056"></a>
## BUG-056 — Binary A/E ignores the exposure offset that was used during training

**Priority:** P2. **Status:** Open.

**Source:**
[sklearn.py:7803](../../python/t_boost/sklearn.py#L7803)–[sklearn.py:7806](../../python/t_boost/sklearn.py#L7806)
(`TBoostClassifier._expected_response` ignores its exposure argument), invoked by
`actual_vs_expected` at [sklearn.py:3177](../../python/t_boost/sklearn.py#L3177). The
classifier's public fit documentation explicitly supports exposure for binary fits at
[sklearn.py:7854](../../python/t_boost/sklearn.py#L7854)–[sklearn.py:7857](../../python/t_boost/sklearn.py#L7857).
Native training converts exposure into a raw-score log offset at
[boost.rs:632](../../crates/t-boost-core/src/engine/boost.rs#L632)–[boost.rs:635](../../crates/t-boost-core/src/engine/boost.rs#L635).

**Trigger and impact.** Fit an ordinary binary classifier with an exposure offset, then
provide the required aligned exposure to `actual_vs_expected` or `pricing_report`. These
methods require exposure when fitting used it, but they compute expected totals from
baseline unit-exposure probabilities. The result disagrees with the response model
fitted to y. This is distinct from BUG-016's noncanonical objective spellings and
BUG-017's non-0/1 class labels: the reproduction uses canonical logistic and literal
labels 0/1.

Reproduction:

```python
import numpy as np
from t_boost import TBoostClassifier
x = np.ones((200, 1), dtype=np.float32)
y = np.repeat([0, 1], 100)
e = np.repeat([1., 4.], 100).astype(np.float32)
for prune in [False, True]:
    m = TBoostClassifier(
        n_trees=5, n_bags=1, prune=prune,
        validation_fraction=None, n_jobs=1,
    ).fit(x, y, exposure=e)
    raw = m.decision_function(x).astype(float)
    correct = 1 / (1 + np.exp(-(raw + np.log(e))))
    a = m.actual_vs_expected(x, y, exposure=e)[0]
    print(prune, raw[0], sum(a['actual']), sum(a['expected']), sum(correct))
```

Observed for both pruned and unpruned fits:

```text
raw score:                  -0.69314718
baseline probability:        0.33333334
actual total:              100.0
reported expected total:    66.66666865348816
offset-aware expected:     100.00000000000006
```

The starting score is `-log(2)`; the two exposure groups have fitted response
probabilities 1/3 and 2/3. Reporting the unit-exposure probability for both groups
spuriously reports underprediction. Ordinary `predict_proba` without a serve-time
exposure argument is not itself alleged wrong; the bug is the explicit exposure-aware
report silently ignoring its supplied vector.

**Suggested fix.** For binary reporting compute the deployed raw logit plus the exposure
log offset and apply the native logistic inverse-link convention. Do not multiply
baseline probability by exposure (exposure scales odds for the logit link). Validate
exposure consistently and preserve explicit unit-exposure evaluation.

**Acceptance checks.** Nonuniform and unit exposures, sample weights, named exposure
columns, binary numeric/string labels after BUG-017 is fixed, bytes/JSON load, and
comparison to the actual training loss's response transformation. Include
`pricing_report` as a dependent surface.

<a id="bug-057"></a>
## BUG-057 — A literal missing-sentinel category is silently merged with actual missing values

**Priority:** P2. **Status:** Open.

**Source:** [_ingest.py:27](../../python/t_boost/_ingest.py#L27) sets `_CAT_MISSING =
'__t_boost_missing__'`;
[_ingest.py:32](../../python/t_boost/_ingest.py#L32)–[_ingest.py:42](../../python/t_boost/_ingest.py#L42)
maps missing values to that string while passing an equal real string through unchanged.
Polars fast paths perform the same collision using `fill_null` at
[_ingest.py:221](../../python/t_boost/_ingest.py#L221) and
[_ingest.py:223](../../python/t_boost/_ingest.py#L223). Both coded and ordinary
ingestion share the ambiguity.

**Trigger and impact.** A real categorical level is literally `__t_boost_missing__` and
missing values also occur. The distinct input identities collapse before fitting, with
no rejection or escaping. The model cannot learn their different outcomes, and serving
cannot distinguish them. This corrupts native training/routing identities, unlike
BUG-044's export-only `<rare>` display collision. It also differs from BUG-024: the
result does not depend on container type, scalar precision, batch size, or optional
pandas.

Reproduction:

```python
import numpy as np
import polars as pl
from t_boost import TBoostRegressor
opts = dict(n_trees=10, n_bags=1, prune=False,
            validation_fraction=None, n_jobs=1)
y = np.repeat([0., 10.], 100).astype(np.float32)
for label in ['__t_boost_missing__', 'ordinary-value']:
    x = pl.DataFrame({'cat': [None]*100 + [label]*100})
    m = TBoostRegressor(**opts).fit(x, y)
    print(label, m.predict(x)[[0, -1]])
```

```text
__t_boost_missing__ [5.0, 5.0]
ordinary-value     [0.39487800, 9.60512161]
```

Control: a real `__t_boost_rare__` category is rejected by the native encoder
([cat.rs:1539](../../crates/t-boost-core/src/cat.rs#L1539)–[cat.rs:1542](../../crates/t-boost-core/src/cat.rs#L1542))
**only when `cat_min_data_per_group > 0`** (the default). The guard sits after the
`min_data_per_group <= 0` early return at `cat.rs:1531-1536`, so with
`cat_min_data_per_group=0` the same label is accepted and fits (verified 2026-10-02:
`[0.394878, 9.60512161]`). The reserved rare namespace therefore has a partial guard;
the Python missing namespace has none. A fix should make both reservations
unconditional.

**Suggested fix.** Encode missing identity separately from user strings
(tagging/escaping with a reversible representation), or reject a colliding literal
clearly before missing-value normalization. Cover the optimized polars paths as well as
`_cat_level`. Consider compatibility with existing encoders whose missing key uses the
literal string; existing already-collapsed training information cannot be recovered
retroactively.

**Acceptance checks.** Real sentinel values alone and alongside None/NaN, mixed
supported containers, coded/uncoded paths, small and large batches, and model round
trips. A distinct legitimate level must remain distinct or be rejected explicitly, never
silently conflated.

<a id="bug-058"></a>
## BUG-058 — Estimator envelopes accept metadata inconsistent with the native model

**Priority:** P2. **Status:** Open.

**Source:** [sklearn.py:1282](../../python/t_boost/sklearn.py#L1282) (`_check_envelope`
checks version and estimator class only);
[sklearn.py:1338](../../python/t_boost/sklearn.py#L1338)–[sklearn.py:1352](../../python/t_boost/sklearn.py#L1352)
(`_restore_metadata` overwrites native feature/class/shape metadata without validating
relationships). Both `from_json` and `from_bytes` attach a validated native model and
then call this restoration routine.

**Scope.** These reproductions deliberately alter the Python envelope while preserving a
valid native payload. No ordinary writer is alleged to emit these malformed
combinations. This is a Python load-boundary validation gap, separate from BUG-033's
mismatched native feature-set/axis identities and BUG-025's normal uint64 round-trip
corruption.

Reproduction A — invalid class cardinality and empty class list:

```python
import json
import numpy as np
import polars as pl
from t_boost import TBoostClassifier
x = pl.DataFrame({
    'signal': np.tile([0., 1.], 100).astype(np.float32),
    'noise': np.zeros(200, dtype=np.float32),
})
y = np.tile([0, 1], 100)
m = TBoostClassifier(n_trees=20, n_bags=1, prune=False,
    validation_fraction=None, n_jobs=1).fit(x, y)
for labels in [['A', 'B', 'C'], []]:
    doc = json.loads(m.to_json())
    doc['classes_'] = labels
    loaded = TBoostClassifier.from_json(json.dumps(doc))
    print('accepted classes', loaded.classes_,
          'probability shape', loaded.predict_proba(x[:2]).shape)
    try:
        print(loaded.predict(x[:2]))
    except Exception as exc:
        print(type(exc).__name__, str(exc))
```

```text
accepted classes ['A' 'B' 'C'] probability shape (2, 2)
['A' 'B']
accepted classes [] probability shape (2, 2)
IndexError: index 0 is out of bounds for axis 0 with size 0
```

The same invalid cardinality also loads through bytes when the header is changed and its
encoded length updated correctly:

```python
blob = m.to_bytes()
length = int.from_bytes(blob[4:8], 'big')
header = json.loads(blob[8:8+length])
header['classes_'] = ['A', 'B', 'C']
encoded = json.dumps(header).encode()
corrupt = blob[:4] + len(encoded).to_bytes(4, 'big') + encoded + blob[8+length:]
loaded = TBoostClassifier.from_bytes(corrupt)
print(loaded.classes_.tolist(), loaded.predict_proba(x[:1]).shape)
# ['A', 'B', 'C'] (1, 2)
```

Reproduction B — named-feature metadata changes scoring despite unchanged native model:

```python
from t_boost import TBoostRegressor
m = TBoostRegressor(n_trees=20, n_bags=1, prune=False,
    validation_fraction=None, n_jobs=1).fit(x, y.astype(np.float32)*10)
doc = json.loads(m.to_json())
print(doc['feature_names_in_'])
print(json.loads(doc['model'])['model']['schema']['feature_names'])
doc['feature_names_in_'] = ['noise', 'signal']
loaded = TBoostRegressor.from_json(json.dumps(doc))
print(m.predict(x[:2]), loaded.predict(x[:2]))
```

```text
['signal', 'noise']
['signal', 'noise']
[0.03118572, 9.9688139] [0.03118572, 0.03118572]
```

The header/native schema control matters: **do not demand naive equality of all name
arrays**. A normal mixed categorical fit can intentionally write header
`['category','value']` and native axis names `['value','category']`, with
`cat_indices=[0]`; multi-channel categoricals add further axis expansion. That control
was executed during this pass. The all-numeric mismatch above has no such permitted
mapping.

**Suggested fix.** Validate envelope field shapes/types and relationships to the
attached native model before exposing a fitted estimator. Require class cardinality to
match native probability width, reject empty/invalid class mappings, and validate
feature names/count/category positions through the supported input-to-native-axis
mapping rather than unconditional name equality. Keep dtype-preserving original labels
and valid historical metadata migration supported. Raise `SerializationError` for
inconsistent payloads.

**Acceptance checks.** Mutate one header field at a time in both bytes and JSON while
retaining valid framing/native contents. Include empty and wrong-size classes,
impossible feature counts, invalid categorical positions, swapped numeric names, and
proper mixed/multi-channel name remapping controls. Validate at load time rather than
waiting for prediction to crash or silently misroute.

<a id="bug-059"></a>
## BUG-059 — Runtime-only estimator repr fails for valid NumPy-array parameters

**Priority:** P3. **Status:** Open.

**Source:**
[_compat.py:74](../../python/t_boost/_compat.py#L74)–[_compat.py:82](../../python/t_boost/_compat.py#L82),
particularly `value != default` in a Boolean condition at
[_compat.py:80](../../python/t_boost/_compat.py#L80).

**Cause.** The no-scikit-learn BaseEstimator shim tries to detect nondefault parameters
through ordinary inequality. Valid array-valued parameters produce an array of Booleans,
whose truth value is ambiguous.

**Impact.** `repr`, printing, logging and notebook rendering can fail for otherwise
valid estimators using NumPy arrays for monotone/categorical declarations. Scikit-learn
is optional; the affected fallback is a supported normal runtime configuration. This is
separate from BUG-055's score validation failure.

Reproduction (fresh process):

```python
import sys
sys.modules['sklearn'] = None
import numpy as np
from t_boost import TBoostRegressor
m = TBoostRegressor(monotone_constraints=np.array([1, 0]))
print(repr(m))
```

```text
ValueError: The truth value of an array with more than one element is ambiguous.
Use a.any() or a.all()
```

**Suggested fix.** Make default comparison safe for structured/array parameters, or
simply render parameters without scalar truth-testing arbitrary values. Preserve
readable bounded output.

**Acceptance checks.** Repr before/after fit for ndarray monotone constraints and
categorical masks, lists/dicts/scalars, and sklearn present/blocked. Representation
should not mutate parameters or fitted state.

<a id="bug-060"></a>
## BUG-060 — Reconstructed bags reuse the soup intercept, leaking targets into OOB evidence

**Priority:** P2. **Status:** Open.

**Source:** [prune.rs:1927](../../crates/t-boost-core/src/prune.rs#L1927), especially
`f0: model.f0` at [line 1949](../../crates/t-boost-core/src/prune.rs#L1949). Original
bag intercepts are averaged in
[engine/boost.rs:4260](../../crates/t-boost-core/src/engine/boost.rs#L4260) and not
retained alongside the bag spans. The reconstructed models feed [OOB evidence at
prune.rs:2221](../../crates/t-boost-core/src/prune.rs#L2221) (the `bag_bank` call; 2223
is a comment) and [bag-score variance at
line 2089](../../crates/t-boost-core/src/prune.rs#L2089). `Model` carries only
`bag_spans`/`bag_in_bag` (`engine/mod.rs:866-886`); no per-bag intercept is retained
anywhere. The helper's own doc (`prune.rs:1911-1914`) says "the soup's f0" and claims
only the all-bag mean identity, which contradicts the per-bag `f0_b` promised at
`prune.rs:2117-2120`. The existing jury test (`prune.rs:4850-4855`) compares against
`bag_raw_scores_for_rows`, which goes through the same `bag_member_model`, so it is
circular on the intercept and could not catch this. (Verified 2026-10-02: both runs and
the `[0.0]` variance reproduced exactly; bag sizes are `[10, 10]`.)

**Trigger and impact.** Outer bags have different fitted intercepts. The pruning code
reconstructs their tree slices but substitutes the full soup's intercept into every bag.
An observation's purportedly out-of-bag score can consequently depend on its own target
through other bags that did train on it. The same reconstruction removes variation
between bag intercepts from the reported prediction variance. This affects the evidence
primitives used by ranked-path pruning, the pruning guard, and bag-noise estimation; the
fit-time OOB residual calculation inside `attach_cell_correction` still uses the
original bag models and is not the source of this particular defect.

**Cause.** The soup preserves per-bag tree spans and memberships, but only the averaged
intercept. `bag_member_model` multiplies a bag's tree coefficients back to standalone
scale and retains `model.f0`, the soup intercept. This does preserve the mean over *all*
bags, as its comment observes. It does not preserve the individual bag functions or the
mean over the particular subset of bags which excluded a row. The OOB API explicitly
promises the full standalone bag raw score and an honest jury
([prune.rs:2117](../../crates/t-boost-core/src/prune.rs#L2117)); substituting a mean
that includes in-bag targets violates that stronger claim.

**Reproduction.** The complete Rust harness below uses 20 finite constant-feature rows,
two bags with `.5` subsampling, one attempted tree, no validation, and ordinary squared
error. Constant features produce no trees, making each original bag's prediction exactly
its training-target mean. With seed zero, row zero is excluded from bag zero and
included in bag one. Change only its target from zero to 100:

```text
before: row=0 trees=0 bag_membership=[false,true] soup_f0=0 oob_count=1 reported_oob=0 honest_oob=0
after:  row=0 trees=0 bag_membership=[false,true] soup_f0=5 oob_count=1 reported_oob=5 honest_oob=0
```

The actual OOB bag saw ten zero targets in both runs, so its intercept and prediction
remain zero. The other bag has mean ten. Averaging their intercepts creates the leaked
value five used by the reported OOB score.

`bag_score_variance_for_rows` also returns `[0.0]` after the perturbation. The actual
standalone bag predictions are `[0,10]`, whose sample variance is `50`. This is missing
between-bag intercept variability, not floating-point cancellation and not BUG-053. The
reproduction checks the public core evidence primitives; it does not claim a measured
change in the final Python keep-set for this tiny, interaction-free dataset.

**Suggested fix.** Preserve each bag's fitted intercept as runtime metadata alongside
its tree span and membership, and use that intercept to reconstruct the bag.
Alternatively absorb each bag's intercept deviation from the soup mean into an exact
bag-specific constant component that survives extraction. Keep the all-bag mean
identity, individual-bag prediction identity, and OOB honesty as separate invariants.

**Acceptance checks.** Compare every reconstructed bag's full predictions with its
original model before souping. Perturb a target excluded by a particular bag and verify
that bag's score stays unchanged. Check the actual out-of-bag subset mean and sample
variance against independent calculations. Include bags with different class priors,
weights, and exposure-aware intercepts, while preserving deterministic aggregation.

```rust
use t_boost_core::boosters::{BoosterConfig, EnsembleSpec};
use t_boost_core::constraints::{CredibilityFloor, InteractionPolicy};
use t_boost_core::data::{bin_columns, BinConfig, ServeBinnedMatrix};
use t_boost_core::engine::{Booster, Config, FitSpec};
use t_boost_core::explain::RefMeasure;
use t_boost_core::loss::SquaredError;
use t_boost_core::prune::{bag_oob_group_raw_sums, bag_score_variance_for_rows};
fn main() {
    let n = 20;
    let col = vec![1.; n];
    let x = bin_columns(&[&col], None, &BinConfig::default(), 0).unwrap();
    let loss = SquaredError;
    let spec = FitSpec {
        loss: &loss,
        weight: None,
        exposure: None,
        monotone: Default::default(),
        interaction: InteractionPolicy::default(),
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed: 0,
    };
    let booster = Booster::with_config(Config {
        n_trees: 1,
        boosters: BoosterConfig {
            ensemble: EnsembleSpec::OuterBag {
                n_bags: 2,
                bag_subsample: 0.5,
                cell_refit: None,
            },
            ..Default::default()
        },
        ..Default::default()
    });
    let mut y = vec![0.; n];
    let before = booster.fit(&x, &y, &spec).unwrap();
    let bags = before.bag_in_bag.as_ref().unwrap();
    let r = (0..n).find(|&r| bags[0][r] != bags[1][r]).unwrap();
    y[r] = 100.;
    let after = booster.fit(&x, &y, &spec).unwrap();
    for (name, m) in [("before", before), ("after", after)] {
        let evidence = bag_oob_group_raw_sums(
            &m,
            &ServeBinnedMatrix(x.clone()),
            RefMeasure::Uniform,
            None,
            &[],
        )
        .unwrap();
        let bags = m.bag_in_bag.as_ref().unwrap();
        let honest = (0..bags.len())
            .filter(|&b| !bags[b][r])
            .map(|b| {
                y.iter()
                    .enumerate()
                    .filter(|(i, _)| bags[b][*i])
                    .map(|(_, v)| *v as f64)
                    .sum::<f64>()
                    / bags[b].iter().filter(|v| **v).count() as f64
            })
            .sum::<f64>()
            / evidence.counts[r] as f64;
        println!("{name}: row={r} trees={} bag_membership={:?} soup_f0={} oob_count={} reported_oob={} honest_oob={honest}",m.trees.len(),bags.iter().map(|b|b[r]).collect::<Vec<_>>(),m.f0,evidence.counts[r],evidence.full_sum[r]/evidence.counts[r] as f64);
        let variance = bag_score_variance_for_rows(
            &m,
            &ServeBinnedMatrix(x.clone()),
            RefMeasure::Uniform,
            None,
            None,
            &[r as u32],
        )
        .unwrap();
        println!("{name}: reconstructed bag score variance={variance:?}");
    }
}
```

<a id="bug-061"></a>
## BUG-061 — A fixed ten-step intercept reanchor can stop far from class balance

**Priority:** P2. **Status:** Open.

**Source:** [prune.rs:2511](../../crates/t-boost-core/src/prune.rs#L2511), especially
the unconditional ten-iteration limit at [line
2533](../../crates/t-boost-core/src/prune.rs#L2533) and unchecked return at line 2560.
The helper is used by the keepset deployment at [line
2747](../../crates/t-boost-core/src/prune.rs#L2747), multiclass guard arms, and the
temperature/slope profiling helper around line 4034.

**Trigger and impact.** Multiclass logits separate groups strongly, while the target
class totals require a material intercept shift. Reanchoring returns success after ten
iterative proportional-fitting updates even if the predicted class masses remain
substantially different from the observed masses. The deployed artifact then fails the
stated aggregate-balance contract; guard and slope calculations using the same routine
can evaluate incompletely reanchored candidates too.

**Cause.** The update is `delta_k += log(observed_mass_k / predicted_mass_k)`. This is
an iterative proportional-fitting update, not the exact joint multinomial Newton step
claimed by the comments. Concavity does not imply convergence within ten rounds. There
is no convergence test, adaptive continuation, or diagnostic for the remaining
class-mass error. (Verification 2026-10-02: the comments at `prune.rs:2500-2503` and
`2586-2588` do claim "each round is the exact Newton step … needs no convergence
test"; a Newton step would use the full Hessian `Σ w (diag p − p pᵀ)`, and the
implemented update drops the off-diagonal coupling. The unit test
`intercept_shifts_reproduce_the_observed_class_mass` (`prune.rs:4512-4515`) is
annotated "10 fixed IPF rounds, no convergence test — this asserts the budget is
actually enough"; the fixture below falsifies that assumption. The library and
reference ten-iteration masses agree to ~1e-6.)

**Reproduction.** The complete Rust harness below constructs a finite, validated
three-class model and calls the public `prune_multiclass_to_keepset`, retaining its only
feature support. There are 100 unit-weight rows: the first 50 have logits `[20,0,0]` and
the other 50 `[0,20,0]`. Observed class counts are `[10,80,10]`. All feature bins and
model structures are valid, no extreme exponents or nonfinite inputs are used.

```text
observed=[10,80,10]
library IPF10 predicted=[24.91346598,50.17312914,24.91340928]
reference iter10 predicted=[24.91345898,50.17312626,24.91341475]
reference iter100 predicted=[10.00000000,80.00000000,10.00000000]
```

The independent reference simply continues the identical update using stable float64
softmax. Agreement at ten iterations confirms that the library discrepancy is
insufficient convergence, not a mismatch in the input or reference calculation; by
iteration 100 the desired masses agree within approximately `1e-13`. The class expected
to have 80 observations instead receives about 50.17 after the library reports success.

This was reproduced through a public core pruning API on a handcrafted validated model,
not through an end-to-end Python training benchmark. Margins of 20 are sufficient;
malformed-model behavior is not involved. The result also demonstrates that the issue
exists even when retaining all supports, so it does not depend on selecting the wrong
table subset.

**Suggested fix.** Iterate until a deterministic residual criterion on weighted class
masses is satisfied, with a sufficiently large safe iteration cap and explicit handling
of nonconvergence. A properly implemented damped Newton method in a fixed logit gauge is
another option. Correct the documentation's claim that each IPF round is an exact Newton
step. Keep the same convergence policy across deployment, pruning guards, and slope
profiling.

**Acceptance checks.** Cover this two-region, three-class fixture; near-separated
logits; rare observed classes; nonuniform weights; and a constant-logit case that
converges quickly. Assert class-mass agreement after reanchoring, not only that deviance
decreases. Check deterministic results across thread counts and ensure empty-class
handling remains explicitly defined.

```rust
use t_boost_core::engine::{MultiClassModel, Split};
use t_boost_core::explain::{fixture_model, fixture_serve, FeatureSet, RefMeasure};
use t_boost_core::loss::LossId;
use t_boost_core::prune::prune_multiclass_to_keepset;
fn main() {
    let n = 100;
    let mut serve = fixture_serve();
    serve.0.data = vec![
        (0..n).map(|i| if i < 50 { 1 } else { 2 }).collect(),
        vec![1; n],
    ];
    serve.0.n_rows = n as u32;
    let classes = (0..3)
        .map(|k| {
            let mut m = fixture_model();
            m.schema.objective.loss = LossId::Softmax;
            let tree = &mut m.trees[0].1;
            tree.splits = vec![Split {
                axis: 0,
                bin_le: 1,
                missing_left: false,
            }];
            tree.depth = 1;
            tree.leaves = vec![0.; 8];
            if k == 0 {
                tree.leaves[1] = 20.;
            }
            if k == 1 {
                tree.leaves[0] = 20.;
            }
            m
        })
        .collect();
    let mc = MultiClassModel {
        classes,
        class_labels: vec!["a".into(), "b".into(), "c".into()],
        schema_version: 2,
        cell_refit: None,
    };
    mc.validate().unwrap();
    let labels = (0..n)
        .map(|i| {
            if i < 10 {
                0
            } else if i < 90 {
                1
            } else {
                2
            }
        })
        .collect::<Vec<_>>();
    let w = vec![1.; n];
    let tm = prune_multiclass_to_keepset(
        &mc,
        &serve,
        &labels,
        &w,
        RefMeasure::Uniform,
        &[FeatureSet::new(&[0])],
    )
    .unwrap();
    let pred = tm.predict_proba(&serve.0).unwrap();
    let masses = (0..3)
        .map(|k| (0..n).map(|r| pred[r * 3 + k] as f64).sum::<f64>())
        .collect::<Vec<_>>();
    println!("observed=[10,80,10] library IPF10 predicted={masses:?}");
    let raw = mc.predict_raw(&serve.0).unwrap();
    let mut d = vec![0.; 3];
    for iter in 0..1000 {
        let mut masses = vec![0.; 3];
        for r in 0..n {
            let logits = (0..3)
                .map(|k| raw[r * 3 + k] as f64 + d[k])
                .collect::<Vec<_>>();
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps = logits.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
            let z = exps.iter().sum::<f64>();
            for k in 0..3 {
                masses[k] += exps[k] / z;
            }
        }
        if [10, 100, 999].contains(&iter) {
            println!("reference iter={iter} predicted={masses:?}");
        }
        for k in 0..3 {
            d[k] += ([10., 80., 10.][k] / masses[k]).ln();
        }
    }
}
```

<a id="bug-062"></a>
## BUG-062 — Recentring a pruned high-order bank turns recreated categorical effects into numeric exports

**Priority:** P2. **Status:** Open.

**Source:** [explain.rs:5359](../../crates/t-boost-core/src/explain.rs#L5359) copies
stored factored axes only when the exact child support exists; otherwise it calls
`grids.axis_id`. [Grid reconstruction at line
5435](../../crates/t-boost-core/src/explain.rs#L5435) turns joint categorical
placeholders into fictitious borders `[0,1,...]`, with `joint: None`.
[FactoredEffect::export_boxes at line
2844](../../crates/t-boost-core/src/explain.rs#L2844) branches on `axis.joint_channels`
and therefore exports the regenerated axes as numeric thresholds.
[serialize.rs:1412](../../crates/t-boost-core/src/serialize.rs#L1412) forwards this
payload into rating export.

**Trigger and impact.** A valid bank includes an order-four effect involving a
categorical feature with multiple channels. Public `retain_tables` keeps that effect
while dropping its lower-order descendants. Public `TableModel::recentred_bank` with
nonuniform row mass recreates nonzero three-way descendants. Those descendants lose the
categorical axis identity even though the retained parent still contains the correct
axis template. In-memory predictions remain exact and `TableModel::validate` succeeds,
but exported boxes use fictitious numeric thresholds instead of the categorical cell
masks needed to serve the effect.

This differs from BUG-032: no banding is present, no missing-template error is raised,
and the operation returns a falsely typed export. It also differs from BUG-044's omitted
standalone routing metadata: even a consumer supplied all original encoders is handed
the wrong box predicate after this transformation.

**Complete reproduction.** The fixture functions are public Rust helpers. The model is
constructed with legitimate public fields and validates; no serialized state is edited.
The four-row fixture and 32 leaves stay small.

```rust
use t_boost_core::{
    data::{AxisKind, AxisProvenance, FeatureId},
    engine::Split,
    explain::{fixture_multichannel_model, fixture_multichannel_serve, RefMeasure},
    prune::retain_tables,
    table_model::TableModel,
};

fn cat_probe() {
    let mut m = fixture_multichannel_model();
    let mut x = fixture_multichannel_serve();
    for raw in 1..=3 {
        let axis = m.grids.len() as u32;
        m.grids.push(m.grids[0].clone());
        m.provenance.push(AxisProvenance {
            raw: FeatureId(raw),
            kind: AxisKind::Numeric,
        });
        m.schema.feature_names.push(format!("x{raw}"));
        m.schema.feature_kinds.push(AxisKind::Numeric);
        m.trees[0].1.splits.push(Split {
            axis,
            bin_le: 1,
            missing_left: false,
        });
        x.0.data.push(vec![1, 2, 1, 2]);
    }
    m.trees[0].1.depth = 5;
    m.trees[0].1.leaves = (0..32).map(|v| if v == 31 { 32. } else { 0. }).collect();
    m.schema_version = t_boost_core::serialize::SCHEMA_VERSION;
    x.0.grids = m.grids.clone();
    x.0.provenance = m.provenance.clone();
    println!("model validate {:?}", m.validate());
    let b = m.explain(&x, RefMeasure::Uniform).unwrap();
    let keep = b
        .factored
        .iter()
        .filter(|f| f.u.order() == 4)
        .map(|f| f.u.clone())
        .collect::<Vec<_>>();
    let p = retain_tables(&b, &keep);
    for (name, bank) in [("full", b), ("pruned", p)] {
        println!(
            "{name}: dense{} factored{}",
            bank.tables.len(),
            bank.factored.len()
        );
        let before = TableModel::from_model_and_bank(&m, bank.clone());
        let r = before
            .recentred_bank(
                &x.0,
                Some(&[1., 2., 3., 10.]),
                RefMeasure::ExposureMarginals { floor: 0.01 },
            )
            .unwrap();
        let tm = TableModel::from_model_and_bank(&m, r.clone());
        println!("recenter validates {:?}", tm.validate());
        let mut maxdiff = 0f64;
        for c in 0..5 {
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        let cells = [c, i, j, k];
                        maxdiff = maxdiff
                            .max((bank.score(&cells).unwrap() - r.score(&cells).unwrap()).abs());
                    }
                }
            }
        }
        println!("maxdiff {maxdiff}");
        let f = r
            .factored
            .iter()
            .find(|f| f.u.0.iter().map(|r| r.0).collect::<Vec<_>>() == vec![0, 1, 2])
            .unwrap();
        let cells = [1, 1, 1, 1];
        let exported = f
            .export_boxes()
            .unwrap()
            .iter()
            .map(|b| {
                let mut corner = 0;
                for (d, raw) in f.u.0.iter().enumerate() {
                    let low = match &b.categorical_low_cells[d] {
                        Some(ids) => ids.contains(&cells[raw.0 as usize]),
                        None => 1.0 <= b.thresholds[d], // aa mean encoding =1; numeric inputs =1
                    };
                    if low {
                        corner |= 1 << d;
                    }
                }
                b.octants[corner]
            })
            .sum::<f64>();
        println!(
            "effect{:?}: var {}; cat axis {:?}; native {} exported {}",
            f.u,
            f.variance,
            f.axes[0],
            f.eval(&cells).unwrap(),
            exported
        );
    }
}

fn main() {
    cat_probe();
}
```

**Observed.** Both original and recentred table models validate. With the full bank, the
regenerated/reference `{0,1,2}` effect retains `joint_channels=Some([0,1])`, empty
borders and `categorical_low_cells[0]=Some([1])`; native and exported values are both
`4.217232637338354` for category `aa` and numeric values `1,1`.

With only the order-four parent retained, that child is nonzero
(`variance=0.01471572292520605`) but has `joint_channels=None`, fictitious borders
`[0,1,2]`, and an exported numeric threshold `0` in the categorical position. The
correct child value is `-1.387247578479819`. Applying the published numeric predicate to
the original first channel (`aa` encoding `1`) produces `0.09463508066298845`. The other
original channel's encoding is `0.1`, also greater than the fictitious threshold, so
choosing it does not rescue the predicate. Correct routing needs joint-cell membership
`{1}`, which the transformed export has erased.

The exhaustive native bank-score difference before/after recentering is at most
`1.78e-15` over all `5×3×3×3` cells. This isolates exported axis semantics from model
score corruption and confirms a material nonzero descendant, rather than a harmless zero
table.

**Suggested fix.** Preserve per-raw axis templates across the bank, sourcing them from
any surviving dense or factored effect (especially the depositing parent), and propagate
them when lower-order effects are created. Never let `from_border_grids`' shape-only
padding become semantic axis metadata. Validate exported categorical effects against
provenance.

**Acceptance checks.** Retain only an order-four-or-higher effect, recenter on
nonuniform mass, and verify every newly created categorical child keeps its channel
identity. Reconstruct all known levels from exported boxes using original routing
metadata, compare with native child evaluation, and check numeric and full-bank
controls. Cover both explicit-mass recentering and reference-measure changes.

<a id="bug-063"></a>
## BUG-063 — Recentring an interaction-only bank reports zero support for recreated main effects

**Priority:** P2. **Status:** Open.

**Source:** [explain.rs:5281](../../crates/t-boost-core/src/explain.rs#L5281) fills
support only for tables present before repurification. [Line
5394](../../crates/t-boost-core/src/explain.rs#L5394) copies rebuilt support only when
the exact effect existed in the source bank; new effects retain the zero tensor
allocated at [line 2205](../../crates/t-boost-core/src/explain.rs#L2205).
[build_weights_from_support_with at line
5802](../../crates/t-boost-core/src/explain.rs#L5802) trusts only main-effect support
and falls back to uniform weights when it is empty.

**Trigger and impact.** Retain a valid numeric pair effect, then request public
`recentred_bank` with explicit nonuniform mass. Re-purification creates nonzero main
effects, but their reported support is zero in every cell. This is immediately wrong
support metadata for the supplied rows. A subsequent measure change or even
`recompute_under` with the same measure derives the wrong empirical weights from those
zeros and changes the allocation and variances of effects. Predictions still sum
correctly.

This is separate from BUG-032's banded template failure and BUG-052's public diagnostic
assertion using the wrong mass. Here an ordinary unbanded transformation produces an
internally inconsistent successful bank and wrong exported support; both branches are
production re-expression APIs.

**Complete reproduction:**

```rust
use t_boost_core::{
    data::{AxisKind, AxisProvenance, FeatureId},
    engine::Split,
    explain::{fixture_multichannel_model, fixture_multichannel_serve, RefMeasure},
    prune::retain_tables,
    table_model::TableModel,
};

fn support_probe() {
    use t_boost_core::explain::{fixture_model, fixture_serve};
    let m = fixture_model();
    let x = fixture_serve();
    let b = m.explain(&x, RefMeasure::Uniform).unwrap();
    let control = TableModel::from_model_and_bank(&m, b.clone())
        .recentred_bank(
            &x.0,
            Some(&[1., 2., 3., 10.]),
            RefMeasure::ExposureMarginals { floor: 0.01 },
        )
        .unwrap();
    println!(
        "support full-bank control {:?}",
        control
            .tables
            .iter()
            .filter(|t| t.u.order() == 1)
            .map(|t| (t.u.clone(), t.support.values().to_vec()))
            .collect::<Vec<_>>()
    );
    let keep = b
        .tables
        .iter()
        .filter(|t| t.u.order() == 2)
        .map(|t| t.u.clone())
        .collect::<Vec<_>>();
    let b = retain_tables(&b, &keep);
    let tm = TableModel::from_model_and_bank(&m, b);
    let mass = [1., 2., 3., 10.];
    let r = tm
        .recentred_bank(
            &x.0,
            Some(&mass),
            RefMeasure::ExposureMarginals { floor: 0.01 },
        )
        .unwrap();
    for t in &r.tables {
        println!(
            "support {:?}: values {:?} support {:?}",
            t.u,
            t.values.values(),
            t.support.values()
        );
    }
    let tm2 = TableModel::from_model_and_bank(&m, r.clone());
    let direct = tm2
        .recentred_bank(&x.0, Some(&mass), RefMeasure::default())
        .unwrap();
    let cached = r.recompute_under(RefMeasure::default()).unwrap();
    println!(
        "same measure f0 {} -> {}",
        r.f0,
        r.recompute_under(r.w.clone()).unwrap().f0
    );
    println!("default cached f0 {} explicit f0 {}", cached.f0, direct.f0);
    for t in &cached.tables {
        println!(
            "cached {:?}: values {:?} support {:?}",
            t.u,
            t.values.values(),
            t.support.values()
        );
    }
    let mut maxchange = 0f64;
    for i in 0..3 {
        for j in 0..3 {
            maxchange = maxchange
                .max((cached.score(&[i, j]).unwrap() - direct.score(&[i, j]).unwrap()).abs());
        }
    }
    println!("function cached vs explicit diff {maxchange}");
}
fn main() {
    support_probe();
}
```

**Observed.** For mass `[1,2,3,10]`, the correct mains' marginal supports are `[0,3,13]`
and `[0,4,12]`. The full-bank control reports exactly those. After retaining only the
pair, the pair support still contains `[0,0,0,0,1,2,0,3,10]`, total `16`, but both
recreated mains have `[0,0,0]` despite nonzero effects:

```text
main 0 values: [0.03117886046193051, -0.13383764122475325, 0.03117886046193051]
main 1 values: [0.07243298588360143, -0.21634589206809512, 0.07243298588360143]
same-measure recompute f0: 1.5793821956558527 -> 1.5555555555555554
ProductMarginals from cached support f0: 1.5555555555555554
ProductMarginals with original explicit mass f0: 1.561631944444444
```

Cached recomputation drives both main value vectors to floating-point zero, unlike
correct explicit-mass re-expression. The two representations' total score differs by at
most `6.67e-16`, so this is an explanation/reference-measure error, not a prediction
error.

**Suggested fix.** For `recentre_on`, refill supports after the final bank has been
formed so every resulting table is counted on the supplied rows. For bank-only
`recompute_under`, derive newly created subset supports by marginalizing an available
containing support; do not replace known row mass with zeros. If no surviving tensor
supplies the needed data, preserve suitable marginal metadata explicitly. An existing
helper, `support_for_subset`, already serves the analogous `purify_raw_effects` path.

**Acceptance checks.** Start with pair-only and factored-only retained banks, recenter
with nonuniform sample weights/exposure, and verify every generated table's support
against independent cell counts. Re-expression from stored support should match
explicit-mass re-expression, and reapplying the same measure should preserve
effects/variances within rounding. Include full-bank controls and roundtrips.

<a id="bug-064"></a>
## BUG-064 — Joint exports normalize component variances as though the effects were independent

**Priority:** P2. **Status:** Open.

**Source:** [explain.rs:5189](../../crates/t-boost-core/src/explain.rs#L5189) documents
`S_u = variance(f_u)/variance(F)` but divides by the sum of component variances.
[serialize.rs:1281](../../crates/t-boost-core/src/serialize.rs#L1281) and [line
1371](../../crates/t-boost-core/src/serialize.rs#L1371) forward those shares into rating
JSON. The public joint-export contract at
[sklearn.py:6633](../../python/t_boost/sklearn.py#L6633) says shares need no longer add
to one; the implementation forces them to sum to one whenever any variance is positive.
[serialize.rs:1059](../../crates/t-boost-core/src/serialize.rs#L1059) describes the
share as being under the bank reference measure.

**Trigger and impact.** Export an additive model under `ref_measure="joint"` on
correlated features. Under the joint distribution, distinct main effects can have
nonzero covariance. Therefore `Var(sum effects)` is not `sum Var(effects)`. The export
nevertheless uses the latter denominator and labels the result a Sobol/variance share.
On perfectly correlated equal main effects the reported values are twice the documented
variance fractions. This is reporting correctness, not a prediction defect; it does not
allege that classical independent-input Sobol theory uniquely extends to dependent
features. The implementation contradicts its own stated ratio and joint-export caveat.

**Complete public Python reproduction:**

```python
import json
import numpy as np
from t_boost import TBoostRegressor

x = np.tile(np.array([[0,0],[0,1],[1,0],[1,1]], np.float32), (20,1))
y = x[:,0]+x[:,1]
m = TBoostRegressor(n_trees=100,n_bags=1,max_depth=3,max_interaction_order=1,prune=False,
    graduate=False,band_tolerance=None,validation_fraction=None,n_jobs=1,
    leaf_refine_steps=0,reanchor=False).fit(x,y)
for correlated in [False, True]:
    z=x.copy()
    if correlated:z[:,1]=z[:,0]
    e=json.loads(m.tables(z,ref_measure='joint',sample_weight=np.ones(len(z),np.float32)))
    variance=np.var(m.predict_raw(z))
    print('correlated',correlated,'modelvariance',variance)
    for t in e['tables']:
        print(t['feature_set'],t['variance'],t['sobol'],'variance/modelvariance',t['variance']/variance)
```

**Observed:**

```text
independent rows: model variance 0.4213231907368484
main 0: variance 0.2106616293512848, reported sobol 0.5000000934662797, actual variance fraction 0.5000000806574653
main 1: variance 0.21066155059226443, reported sobol 0.4999999065337203, actual variance fraction 0.4999998937249106

perfectly correlated rows: model variance 0.8426463814736884
main 0: variance 0.2106616293512848, reported sobol 0.5000000934662797, actual variance fraction 0.2500000403287351
main 1: variance 0.21066155059226443, reported sobol 0.4999999065337203, actual variance fraction 0.24999994686245777
```

Only two main effects exist (`max_interaction_order=1`), so this discrepancy does not
depend on pairwise ridge regularization, unsupported higher-order hybrid behavior, or
cancellation. The correlated variance is doubled by covariance; the main variances are
unchanged, and the published shares remain unchanged.

An independent Rust additive fixture gives exact simple values: each main variance
`0.25`; model variance `0.5` under independent rows and `1.0` under perfectly correlated
rows; export reports `0.5` for each main in both cases. Full Rust control:

```rust
use t_boost_core::{
    data::{AxisKind, AxisProvenance, FeatureId},
    engine::Split,
    explain::{fixture_multichannel_model, fixture_multichannel_serve, RefMeasure},
    prune::retain_tables,
    table_model::TableModel,
};

fn joint_probe() {
    use t_boost_core::explain::{fixture_model, fixture_serve};
    use t_boost_core::joint::{rejoint, JointOptions};
    let mut m = fixture_model();
    let mut x = fixture_serve();
    m.trees[0].1.leaves = vec![0., 1., 1., 2., 0., 0., 0., 0.];
    for correlated in [false, true] {
        if correlated {
            x.0.data[1] = x.0.data[0].clone();
        }
        let b = m.explain(&x, RefMeasure::Uniform).unwrap();
        let j = rejoint(&b, &JointOptions::default()).unwrap();
        let tm = TableModel::from_model_and_bank(&m, j.clone());
        let p = tm.score_raw(&x.0, None).unwrap();
        let mean = p.iter().map(|v| f64::from(*v)).sum::<f64>() / 4.;
        let variance = p
            .iter()
            .map(|v| (f64::from(*v) - mean).powi(2))
            .sum::<f64>()
            / 4.;
        println!(
            "joint correlated {correlated} model variance {variance} mains {:?} sumsobol {}",
            j.tables
                .iter()
                .map(|t| (t.u.clone(), t.variance))
                .collect::<Vec<_>>(),
            j.sobol().iter().map(|(_, s)| s).sum::<f64>()
        );
        let export = j
            .to_rating_export(
                m.link,
                &m.mode,
                &m.schema,
                &m.provenance,
                &m.schema.cat_encoders,
                None,
            )
            .unwrap();
        println!(
            "joint exported {:?}",
            export
                .tables
                .iter()
                .map(|t| (t.feature_set.clone(), t.variance, t.sobol))
                .collect::<Vec<_>>()
        );
    }
}
fn main() {
    joint_probe();
}
```

**Suggested fix.** For a joint export, use an explicitly defined variance denominator
that includes covariance under that same reference measure, if sufficient joint
information is available. Otherwise omit/refuse the unsupported Sobol statistic, or give
the existing normalized-component statistic a different name and update the contract. Do
not silently reuse the product-measure identity. Ensure the treatment of hybrid
high-order effects is explicit.

**Acceptance checks.** Compare the exported denominator against independently scored
variance for independent, positively correlated and negatively correlated additive
effects, including weights. The perfectly correlated two-main fixture should give `0.25`
each under the documented ratio, not `0.5`. Retain product-measure controls and verify
explanation changes preserve total predictions.

## Suggested remediation sequence

1. Address the P1 training and estimator-state defects: BUG-001 through BUG-003.
   Keep BUG-003 and BUG-004 separate: outer holdout isolation and encoder-local
   own-target isolation require different checks.
2. Correct prediction, persistence, and export mismatches: BUG-024 through
   BUG-027, BUG-044, and BUG-057/BUG-058, alongside BUG-005 through BUG-009
   and BUG-015 through BUG-018. Record intentional changes in model bytes and selection decisions
   where mathematical corrections alter training behavior.
3. Correct loss initialization, training state, and sampling/correction contracts:
   BUG-034 through BUG-036, BUG-039/BUG-040, BUG-048 through BUG-050,
   BUG-054, and BUG-060/BUG-061. Use gradient, score-reconstruction, and
   group-boundary oracles in addition to
   benchmark metrics.
4. Close load/input validation gaps: BUG-010 through BUG-014, BUG-033,
   BUG-037/BUG-038, BUG-045, and BUG-055/BUG-058. Exercise malformed-model
   cases after correcting the fuzz path in BUG-019.
5. Repair serving-pool lifecycle and resource limits (BUG-022/BUG-023 and
   BUG-051), then banding/cache/recentring compositions (BUG-030 through BUG-032)
   and weighted/numerical certification (BUG-052/BUG-053). Restore metadata and
   support propagation during re-centering (BUG-062/BUG-063).
6. Correct reporting and metric results: BUG-041 through BUG-043 and
   BUG-046/BUG-047, BUG-056, and BUG-064. Fix artifact selection and development gates
   (BUG-020/BUG-021), and the remaining API edge cases
   (BUG-028/BUG-029 and BUG-059).
   These can proceed independently of training changes.

For each fix, add the relevant regression first, run the gates required by
[CONTRIBUTING.md](../../CONTRIBUTING.md), and update this document with the fix revision
and verification results. Do not mark an item resolved solely because the existing suite
passes; each counterexample needs explicit coverage.
