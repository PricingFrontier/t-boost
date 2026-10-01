# Fuzz Targets

This is a standalone `cargo-fuzz` crate, deliberately excluded from the main
workspace so normal `cargo test --workspace` and release builds do not pull in
libFuzzer.

Run the local compile smoke:

```sh
cargo +nightly fuzz build
cargo +nightly fuzz run fuzz_deserialize -- -runs=0
cargo +nightly fuzz run fuzz_binning -- -runs=0
```

The scheduled GitHub workflow runs both targets for five minutes each.

Targets:

- `fuzz_deserialize`: arbitrary bytes through bincode `ModelDoc` decoding and JSON
  model decoding. Valid outputs are typed serialization errors or validated models,
  never panics.
- `fuzz_binning`: arbitrary `f32` values and optional weights through grid
  construction plus per-value binning. Valid outputs are typed errors or in-range
  bins, never panics.
