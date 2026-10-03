# Reproduction harnesses for `../bugs.md`

These are the scripts and binaries that re-verified every finding in
[bugs.md](../bugs.md) on 2026-10-02 against revision `6588084`. They are the
acceptance seeds for the fixes, not part of the test suite: nothing here runs
in CI, and the Rust crate is deliberately outside the workspace so the
repository gates never build or lint it.

| Directory | Contents |
|-----------|----------|
| `python/bugNNN.py` | One script per Python-reachable finding; the exact code from the entry plus the controls run during verification. Letters (`bug024a/b/c`, `bug030b/c`, `bug044a/b`) are the entry's separate reproductions or added controls. |
| `rust/src/bin/bugNNN.rs` | One binary per Rust-core finding. `bug037.rs` covers BUG-037 and BUG-038. `smoke.rs` just checks the crate links. |
| `xtask-fixtures/{multi,single}` | The BUG-021 fixtures: the same `#[derive(Serialize, Deserialize)]` struct on one line and on three. |
| `out/` | Scratch output (git-ignored). `bug030.py` writes the banded model that `bug032.rs` consumes; `bug024b.py` writes the model it reloads. |

## Running

Python, from the repository root (needs the venv from `uv sync`):

```sh
uv run --no-sync python specs/roadmap/repro/python/bug046.py
uv run --no-sync python specs/roadmap/repro/python/bug024b.py        # fit + write
uv run --no-sync python specs/roadmap/repro/python/bug024b.py load   # reload without pandas
uv run --no-sync python specs/roadmap/repro/python/bug055.py --block # without scikit-learn
```

Rust, from `specs/roadmap/repro/rust` (release + overflow checks, as the
entries require):

```sh
cargo run --release --bin bug012
RAYON_NUM_THREADS=1 cargo run --release --bin bug036   # BUG-036 pairs events by bag
uv run --no-sync python ../python/bug030.py && cargo run --release --bin bug032
```

BUG-021, from each fixture directory, with the xtask binary built by
`cargo build -p xtask`:

```sh
cd specs/roadmap/repro/xtask-fixtures/multi  && ../../../../../target/debug/xtask check-all; echo exit=$?
cd specs/roadmap/repro/xtask-fixtures/single && ../../../../../target/debug/xtask check-all; echo exit=$?
```

At the original 2026-10-02 verification, two findings were established from
sources rather than execution: BUG-019 (cargo-fuzz working directory) and
BUG-020 (release-gate install). The remediation audit also runs the fuzz targets.

## Remediation evidence

The scripts are retained as historical reproductions. Current regression tests and
per-finding remediation evidence are linked from the
[complete audit](../bugs.md#complete-remediation-audit--2026-10-03).
The BUG-001 script embeds the old sampler, so its output does not test the current
implementation. Other scripts may print the original observation without asserting
the corrected behavior; use the linked regression tests as the completion gates.
