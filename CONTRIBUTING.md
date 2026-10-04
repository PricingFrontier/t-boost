# Contributing to t-boost

This file is written for **coding agents**. Changes to t-boost are made by agents working
for the maintainer, not by outside human contributors. Treat everything here as operating
instructions: the rules are enforced by CI gates, and the conventions are the maintainer's
standing preferences.

t-boost is an oblivious gradient-boosting machine that is **exactly decomposable** into
fANOVA rating tables. Two properties are non-negotiable: **predictiveness** and
**lossless explainability** (the five I2 invariant checks + the I1 feature budget). Every
rule below exists to protect one of them, or the bit-for-bit determinism they are tested
with.

## Repository map

| Path | What it is |
|------|------------|
| `crates/t-boost-core/` | The pure-Rust engine: binning (`data/`), boosting (`engine/`), decomposition (`explain.rs`, `explain/`), pruning (`prune.rs`), banding (`banding.rs`), serialization (`serialize.rs`), the error type (`error.rs`). No Python dependencies, ever. |
| `crates/t-boost-core/tests/` | Integration tests, including the determinism, invariants and overflow-trap gates. |
| `crates/t-boost-py/src/lib.rs` | The PyO3 binding (`t_boost._t_boost`: `_Booster`, `_Model`, `_TableModel`, …). |
| `python/t_boost/_t_boost.pyi` | Type stub for the binding. Must match it exactly (stubtest gate). |
| `python/t_boost/sklearn.py` | The public estimators `TBoostRegressor` / `TBoostClassifier`. Despite the module name they are polars-native and do not need scikit-learn; `_compat.py` makes them genuine sklearn estimators only when it happens to be installed. |
| `python/t_boost/_ingest.py`, `metrics.py`, `_pricing.py` | polars ingest, deviance metrics, pricing helpers. |
| `python/tests/` | The Python test suite (pytest). |
| `xtask/` | Dev-only Rust tasks: the grep-gates (`check-all`), `bit-repro`, `accuracy`, `release-preflight`. |
| `fuzz/` | cargo-fuzz targets (a separate workspace with its own lock file). |
| `docs/`, `mkdocs.yml` | The documentation site (MkDocs + Material), published to GitHub Pages. Hand-written, laid out and worded after the CatBoost documentation. `docs/_snippets/` holds fragments shared by several pages. |
| `scripts/` | Release tooling: `release_version.py`, `verify_pypi_release.py`, `package_smoke_check.py`; `check_docs.py` keeps the parameter reference in step with the estimators. |
| `.github/workflows/` | `ci.yml` (the gates), `docs.yml` (the documentation site), `release.yml` (PyPI), `release-gate.yml` (manual pre-publish checklist), `coverage.yml`, `fuzz.yml`. |

## Environment

- Rust: your default stable toolchain with rustfmt and clippy. CI pins 1.96.1 in each job; the
  MSRV is 1.85 (`rust-version` in `Cargo.toml`). Do not add a `rust-toolchain.toml`: it
  overrides the toolchains CI pins.
- Python: [uv](https://docs.astral.sh/uv/). `uv sync` creates `.venv` with the interpreter from
  `.python-version` (3.11, matching CI), builds the Rust extension into it, and installs the
  `dev` dependency group (pytest, mypy, pandas, pyarrow, scikit-learn — the tests exercise the
  optional interop paths). Do not use a newer interpreter for mypy: it resolves a numpy whose
  stubs mypy cannot parse under the configured `python_version = "3.10"`.
- **After any Rust change under `crates/`, run `uv sync --reinstall-package t-boost`.** Plain
  `uv sync` does not notice Rust edits, and the Python tests would silently run a stale build.
- Run Python tooling through `uv run` (e.g. `uv run pytest`), never a bare `python`/`pip`.

## Definition of done

**A task is done when the gates covering what you changed are green — not when the code is
written.** Run them yourself before reporting; do not hand work back for CI to discover
failures. Report results faithfully: name each gate you ran and its outcome, and quote the
failure output if any is red.

Run the gates for every area you touched:

| You changed | Run |
|-------------|-----|
| Any Rust | `cargo fmt --all --check` · `cargo clippy --workspace --all-targets --all-features -- -D warnings` · `cargo run -p xtask -- check-all` |
| `t-boost-core` | `cargo test --release -p t-boost-core --all-features` (~1 min) · the slow suite `cargo test --release -p t-boost-core --all-features --test '*' -- --ignored` (~13 min) · `cargo test -p t-boost-core --doc` · the bit-repro pair below |
| The binding (`t-boost-py`) or the stub | `uv sync --reinstall-package t-boost` · `uv run python -m mypy.stubtest t_boost._t_boost` · the Python row |
| Python (`python/t_boost`) | `uv run pytest python/tests -q` (~3 min) · `uv run mypy --strict python/t_boost` |
| A constructor parameter of the estimators | its entry under `docs/training-parameters/` (Description / Type / Default value) · `uv run --no-project python scripts/check_docs.py fix` · the documentation row |
| Documentation (`docs/`, `mkdocs.yml`) | `uv run --no-project python scripts/check_docs.py check` · `uv run --no-project --with-requirements docs/requirements.txt mkdocs build --strict` |
| `pyproject.toml` / dependencies | `uv lock` (commit `uv.lock`) · `uv sync --locked` · the Python row |
| `xtask/` | `cargo test -p xtask` · `cargo run -p xtask -- check-all` |
| Workflows | `uvx --from actionlint-py actionlint .github/workflows/*.yml` |

**Run the core tests in release mode.** In debug the suite takes hours. The core has no
`debug_assert!` and `overflow-checks = true` is set in every profile, so `--release` checks
exactly the same things.

**Slow tests.** Integration tests that take more than about 30 s in release are marked
``#[ignore = "slow: run with `cargo test --release -- --ignored`"]``, so the default suite stays
fast. Run them (the command above) whenever you change `t-boost-core`, `Cargo.toml` or
`Cargo.lock`. CI runs them only while a release is pending (see below). Give any new test of
that cost the same attribute. Keep slow tests in `tests/` (integration tests): the slow command
selects `--test '*'`, and the one ignored unit test, `bench_row_scatter`, is a benchmark that
must not run in CI.

The cross-run reproducibility gate (two processes, same seed, byte-identical output):

```bash
cargo run -p xtask -- bit-repro --seed 7 --output target/bit-repro-a.bin
cargo run -p xtask -- bit-repro --seed 7 --output target/bit-repro-b.bin
cmp target/bit-repro-a.bin target/bit-repro-b.bin
```

### What CI runs

`ci.yml` runs on every pull request and every push to `main`:

| Job | Checks |
|-----|--------|
| `lint` | fmt, clippy (incl. the no-panic deny set), `xtask check-all` |
| `py-lint` | clippy on the PyO3 binding |
| `python` | runtime-only install check (`uv sync --locked --no-dev`, no scikit-learn), then pytest, `mypy --strict`, stubtest |
| `test` | core tests (`--all-features`, `--doc`), determinism (`n_threads ∈ {1,2,8}`, byte-compared), invariants, overflow trap, bit-repro |
| `slow-tests` | the slow core tests, only while a release is pending — the workspace version is not on PyPI yet (`scripts/release_version.py pending`); always on nightly and manual runs |
| `m6-preflight` | `cargo test -p xtask`, `xtask accuracy` (plain + adversarial), `xtask release-preflight` |
| `msrv` | build + test on Rust 1.85 |
| `features` | the core builds under each feature combination (`arrow`, `nightly`) |
| `portability` | no `pyo3`/`numpy` in the core's dependency graph; the core builds for `wasm32-unknown-unknown` |
| `deny` | `cargo deny check` (licenses, advisories, bans, sources) |

On push to `main`, nightly and manual dispatch only (not on pull requests):
`platform-matrix` re-runs the core gates on Linux x86_64/aarch64, macOS arm64 and Windows
under Rust 1.96.1 and 1.85, and `bit-repro-cpu-baseline` byte-compares a
`target-cpu=x86-64-v3` build against the portable baseline.

`docs.yml` runs `scripts/check_docs.py check` and a strict MkDocs build on every pull request,
and deploys the site to GitHub Pages on every push to `main`. Preview the site locally with
`uv run --no-project --with-requirements docs/requirements.txt mkdocs serve`.

`coverage.yml` runs `cargo llvm-cov` over the core nightly and on demand. It is informational
(it gates nothing yet) and lives outside `ci.yml` because the instrumented suite runs for over
two hours.

## Code rules

### No panics in library code

The clippy deny set forbids `unwrap`, `expect`, `panic!`, `unreachable!` and
`indexing_slicing` everywhere except `tests`, `benches` and `xtask` (none of which ship).
Return failures through the single [`PbError`] enum (`Result<T, PbError>`), never
`Box<dyn Error>` (`xtask check-no-box-dyn` enforces this). Integer overflow traps in every
profile, so write arithmetic that provably cannot overflow: use `checked_*`/`saturating_*`,
and `u32::try_from(..)` rather than `as` for narrowing.

### `// JUSTIFIED:` for proven-unchecked code

A hot inner loop may use a proven-unchecked index only inside a small scoped function
carrying all of:

1. `#[allow(clippy::indexing_slicing)]` (or a scoped `clippy::arithmetic_side_effects`);
2. a `// JUSTIFIED:` comment on the same or preceding line proving it safe (e.g.
   "`idx ∈ 0..8` because it is built from three `bool` bits");
3. a unit test exercising the extreme indices.

`xtask check-justified` fails any such `#[allow]` without the proof. Prefer
`.get(..).ok_or(PbError::..)?` wherever the branch is not hot. Some older modules carry
module-scoped `#![allow(clippy::indexing_slicing)]` debt: do not add new module-scoped allows.

### Serialized state

Every serialized type uses **fixed-width** integers (`u32`/`u64`, never `usize`/`isize`) and
**deterministic-order** containers (`BTreeMap`/`BTreeSet`, never `HashMap`/`HashSet`). This
includes report structs that are only ever written to JSON. Keep `usize` for internal logic
and convert when building the serialized value, using the codebase's saturating pattern
`u32::try_from(n).unwrap_or(u32::MAX)`. Enforced by `xtask check-no-usize-serialized` and
`check-no-hashmap-serialized`.

The Python estimators wrap the native blob in a JSON-headed envelope (`python/t_boost/sklearn.py`,
`_ENVELOPE_SCHEMA_VERSION`). Bump that version when an older loader would misread a header
written by the new code. Keep reading every older header, and the raw blobs written before the
envelope was universal. Tests that check "this parameter does not move the deployed model" compare
`model_bytes(est)` from `python/tests/_artifact.py`, not `est.to_bytes()`, because the envelope
records the parameters.

### Determinism

Output must be bit-identical across thread counts, processes and runs with the same seed.
Do not introduce order-dependent float reductions, unseeded randomness or hash-order
iteration. Any change that alters model bytes must be deliberate, and you must say so when
you report it.

### The exactness firewall

An operation that cannot preserve exact decomposability (a nonlinear calibration warp, a
continuous target-statistic axis, linear leaves, a base margin above order 3) must flip the
model to `ExactnessMode::Approximate { reason }` and refuse an `Exact` table export. Never
let a change bend the I1/I2 invariants silently: gate it behind the firewall, and tell the
maintainer when you report.

### The binding and its stub move together

When you add, remove or change anything exposed by `crates/t-boost-py/src/lib.rs` (a
parameter, its default, a method, a property), update `python/t_boost/_t_boost.pyi` in the
same change. stubtest compares names, parameter order and defaults. One missing constructor
parameter shifts every later one and produces a cascade of errors.

### Python typing

`mypy --strict` must pass on `python/t_boost`. Give generics their type parameters
(`list[Any]`, `dict[str, Any]`), annotate nested helpers, and narrow `Optional`s explicitly
rather than with `assert`. Untyped third-party modules go in the `[[tool.mypy.overrides]]`
list in `pyproject.toml`.

## Writing rules (docs, comments, names)

- **Never name the predecessor project** (the repository this code was imported from)
  anywhere in this repository. The project is t-boost throughout, including in identifiers
  such as environment variables (the profiler switch is `TBOOST_PROFILE`).
- **README.md** is for library users:
  - install with `uv add t-boost`, not `pip install`;
  - examples use the polars-native estimators (`TBoostRegressor` / `TBoostClassifier` on
    polars frames, with targets/exposure named by column);
  - do not mention scikit-learn;
  - do not discuss hyperparameters or their defaults.
- `pyproject.toml`'s `description` must agree with the README headline.
- `§NN` / `§NN.M` in comments cite sections of the original design spec (`spec/NN-*.md`). The
  spec is not in this repository; keep these as section identifiers, but never cite a file
  path that does not exist here (`spec/…`, `docs/…`).
- Match the surrounding code's comment density and style. Explain *why*, not what.

## Making changes

- Work on a branch, in small commits that each leave the gates green. Do not commit to `main`.
- **Ask the maintainer first** before you push, open or merge a pull request, run the Release
  workflow, or publish anything. Approval for one of these does not carry over to the next.
- A new capability lands behind its gate: write the invariant check, determinism assertion or
  type contract first, so the feature cannot regress what is already proven.
- Prefer a thin, runnable, gated end-to-end slice over a pile of untested modules.
- If you cannot make a gate pass, stop and report it with the output. Never weaken a gate,
  skip a test, or add an `#[allow]`/`# type: ignore` just to get green without the
  maintainer's agreement.

## Releasing

Releases go to PyPI from `.github/workflows/release.yml`, a workflow the maintainer runs by
hand. An agent's part is preparing the version bump; run the workflow only if asked.

1. **Bump the version on a branch:** `uv run --no-project python scripts/release_version.py
   bump patch` (or `minor`, `major`, an explicit `X.Y.Z`). The version lives in the Cargo
   workspace (`pyproject.toml`'s is dynamic) and is recorded in five places that must agree:
   `Cargo.toml`'s `[workspace.package] version`, the `t-boost-core` requirement in
   `crates/t-boost-py/Cargo.toml` and `xtask/Cargo.toml`, and three `Cargo.lock` entries. The
   script rewrites all five; never edit them by hand.
2. **Merge, then wait for main's CI** (`ci.yml`) to pass on the merge commit. With the version
   bumped past PyPI's, this CI run includes the slow core tests (~25 min).
3. **Run Release** from the Actions tab, on `main`. It refuses to start unless main's CI
   passed on that exact commit, the five version records agree, and the version is newer than
   every release on PyPI and not yet tagged. It then:
   - builds one abi3 wheel per platform (Linux x86_64/aarch64, Windows x64, macOS
     x86_64/arm64) plus an sdist;
   - install-smoke-tests the Linux, Windows and macOS-arm64 wheels and the sdist in fresh
     environments;
   - publishes via PyPI trusted publishing;
   - waits until PyPI serves exactly the built files;
   - and only then tags `vX.Y.Z` and creates the GitHub release.

The workflow never commits to the repository. "Dry run" builds and smoke-tests without
publishing. If a run fails after publishing started, "Re-run failed jobs" completes it (the
upload skips files PyPI already has). The one-time GitHub-environment and PyPI
trusted-publisher setup is described at the top of `release.yml`.

`.github/workflows/release-gate.yml` is a separate manual pre-publish checklist. It runs the
full Rust matrix, wheel/sdist builds with a clean-environment smoke test,
`cargo publish -p t-boost-core --dry-run`, `cargo-semver-checks` and
`xtask release-preflight`. It never publishes.

### Build baseline: portable, not `target-cpu=x86-64-v3`

Wheels build at the default portable CPU baseline: `release.yml` sets no
`RUSTFLAGS`/`target-cpu`. This is a decided policy (2026-07-11), not unfinished work.
Bit-reproducibility comes before serve speed. A `-C target-cpu=x86-64-v3` baseline with
`multiversion` runtime dispatch is only safe once specific kernels are proven **order-exact**
(bit-identical regardless of vector width), and that audit has not happened. Only integer
kernels (histogram scatter/accumulate, table-lookup gather) are candidates; float reductions
are excluded. Do not add `target-cpu` flags or `multiversion` without the maintainer's
decision. The `bit-repro-cpu-baseline` jobs in `ci.yml` and `release-gate.yml` byte-compare
an x86-64-v3 build against the portable baseline on every run.

[`PbError`]: crates/t-boost-core/src/error.rs
