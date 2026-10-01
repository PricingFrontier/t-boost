<!--
  t-boost PR template. A PR is "done" when its gates are green (see CONTRIBUTING.md),
  not when the code is written. Tick the gate checklist.
-->

## Summary

<!-- What this PR does, in one or two sentences. -->

## Definition of Done — gates

- [ ] **Fmt** — `cargo fmt --all --check`
- [ ] **Clippy** — `cargo clippy --all-targets --all-features -- -D warnings` (no-panic deny set)
- [ ] **Test** — `cargo test -p t-boost-core --all-features` and `--no-default-features`
- [ ] **Doctests** — `cargo test -p t-boost-core --doc`
- [ ] **Deny** — `cargo deny check`
- [ ] **MSRV** — builds + tests on Rust 1.85
- [ ] **NoPyo3 / Wasm** — core stays Python-free and wasm32-buildable
- [ ] **Determinism** — byte-equal across `n_threads ∈ {1,2,8}` (if training paths touched)
- [ ] **Invariants** — the five I2 checks + I1 budget still pass (if model/tables touched)
- [ ] **Exactness preflight** — `cargo run -p xtask -- release-preflight --seed 7`
- [ ] **Grep-gates** — `cargo run -p xtask -- check-all`

## Invariant / firewall impact

<!--
  Does this change touch I1 (feature budget) or I2 (decomposability)? If it could bend
  either, it MUST be firewall-gated (ExactnessMode::Approximate) and flagged here.
-->
- [ ] No I1/I2 impact, **or** the change is firewall-gated and the reason is documented.

## `// JUSTIFIED:` / unsafe

- [ ] No new `unwrap`/`expect`/`panic` outside `{tests, xtask}`.
- [ ] Any proven-unchecked `#[allow(clippy::indexing_slicing)]` carries a `// JUSTIFIED:` proof + boundary test.
