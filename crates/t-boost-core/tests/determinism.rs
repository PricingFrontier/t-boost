//! Determinism gate (spec §13.4 / §02.10(2), plan F6): CI's "Determinism gate ({1,2,8}-
//! thread byte-equality)" step runs exactly this file (`--test determinism`), so it is the
//! one gate a contributor triaging that CI name will see.
//!
//! Two kinds of coverage live here:
//! - [`model_bytes_are_thread_count_independent`] / [`pb_seed_drives_a_stable_fold`]: a
//!   hand-built model whose leaves come from a FIXED-ORDER parallel fold (the §11 pattern:
//!   `par_chunks` mapped, partials combined in index order — never a steal-order `reduce`).
//!   This isolates and pins the fold PATTERN itself, independent of the boosting engine.
//! - `real_*_is_thread_count_independent` below: the real public `fit()` pipeline — the
//!   same entry point `crates/t-boost-core/src/engine/boost.rs`'s own inline
//!   `#[cfg(test)]` determinism tests (e.g. `fitted_model_is_byte_identical_across_thread_counts`,
//!   `cell_refit_outer_bag_is_g0_exact_and_thread_deterministic`) drive — fit at
//!   `n_threads ∈ {1, 2, 8}` and byte-compare the ENCODED model. Those boost.rs tests are
//!   real coverage too, but only via `cargo test --all-features` (no `--test` filter), so
//!   they are invisible when triaging a failure specifically under the "Determinism gate"
//!   CI step name; this file closes that gap at the same integration level the step name
//!   promises. (Historical note: earlier revisions of this file only had the synthetic
//!   fold-pattern tests above and a doc comment promising real-`fit` coverage "when §06
//!   lands" — §06 landed long ago; that promise is now implemented, not deferred.)

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use rayon::prelude::*;
use t_boost_core::boosters::CellRefit;
use t_boost_core::{
    bin_columns, bin_train_columns, encode_model, explain::fixture_model, pb_seed, BinConfig,
    Booster, BoosterConfig, CategoricalColumn, Config, CredibilityFloor, EnsembleSpec, FeatureId,
    FitSpec, InteractionPolicy, LeakageScheme, MonotoneMap, NumericColumn, Smooth, SquaredError,
    TsConfig, TsEncodingId, CHUNK_ROWS,
};

/// A per-leaf statistic computed as a fixed-order fold over a synthetic row set:
/// each chunk is summed in parallel, the partials are collected in index order, then
/// combined sequentially. The result is independent of how many threads ran it.
fn leaf_value(seed: u64, leaf: u32) -> f32 {
    const N_ROWS: u32 = 50_000;
    let rows: Vec<u32> = (0..N_ROWS).collect();
    let partials: Vec<f64> = rows
        .par_chunks(CHUNK_ROWS)
        .map(|chunk| {
            chunk
                .iter()
                .map(|&r| {
                    let s = pb_seed(seed, leaf, 0, r);
                    // Deterministic pseudo-gradient in [-1, 1).
                    (s as f64 / u64::MAX as f64) * 2.0 - 1.0
                })
                .sum::<f64>()
        })
        .collect(); // IndexedParallelIterator::collect preserves chunk order.
    let total: f64 = partials.iter().sum(); // combined in index order
    (total / f64::from(N_ROWS)) as f32
}

/// Build the fixture model with its 8 leaves replaced by fixed-order folds, inside a
/// rayon pool of exactly `n_threads`, and return its frozen-config bincode bytes.
fn model_bytes_in_pool(n_threads: usize, seed: u64) -> Vec<u8> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap();
    pool.install(|| {
        let mut model = fixture_model();
        let leaves: [f32; 8] = std::array::from_fn(|k| leaf_value(seed, k as u32));
        let (_, tree) = model.trees.get_mut(0).unwrap();
        tree.leaves = leaves.to_vec();
        encode_model(&model).unwrap()
    })
}

#[test]
fn model_bytes_are_thread_count_independent() {
    let seed = 0x00C0_FFEE;
    let b1 = model_bytes_in_pool(1, seed);
    let b2 = model_bytes_in_pool(2, seed);
    let b8 = model_bytes_in_pool(8, seed);

    assert!(!b1.is_empty(), "harness produced no bytes");
    assert_eq!(b1, b2, "n_threads 1 vs 2 produced different bytes");
    assert_eq!(b1, b8, "n_threads 1 vs 8 produced different bytes");
}

#[test]
fn pb_seed_drives_a_stable_fold() {
    // The fold is a pure function of the seed regardless of thread count.
    let a = model_bytes_in_pool(8, 123);
    let b = model_bytes_in_pool(1, 123);
    assert_eq!(a, b);
    // A different seed yields different bytes (the fold is not constant).
    assert_ne!(model_bytes_in_pool(1, 123), model_bytes_in_pool(1, 124));
}

// ---------------------------------------------------------------------------------------
// Real `fit()` pipeline, at the public API, byte-compared across {1, 2, 8}-thread pools.
// The encoded bytes are the equality oracle (spec §10): any field the wire format carries
// — trees, grids, the bag partition's ONE resulting soup, a cell-refit correction bank —
// is covered by one `assert_eq!` on `encode_model`'s output, not a field-by-field walk.
// ---------------------------------------------------------------------------------------

fn run_in_pool<R: Send>(n_threads: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap()
        .install(f)
}

fn fit_spec(loss: &SquaredError) -> FitSpec<'_> {
    FitSpec {
        loss,
        weight: None,
        exposure: None,
        monotone: MonotoneMap::new(),
        interaction: InteractionPolicy::default(),
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed: 0,
    }
}

/// Assert `model_bytes_in_pool_of(1)` is byte-identical to the same fit at 2 and 8 threads,
/// under the given human-readable label (used in the panic message).
fn assert_thread_count_independent(label: &str, bytes_at: impl Fn(usize) -> Vec<u8>) {
    let b1 = bytes_at(1);
    assert!(!b1.is_empty(), "{label}: harness produced no bytes");
    assert_eq!(
        b1,
        bytes_at(2),
        "{label}: 1 vs 2 threads produced different bytes"
    );
    assert_eq!(
        b1,
        bytes_at(8),
        "{label}: 1 vs 8 threads produced different bytes"
    );
}

#[test]
fn real_fit_default_config_is_thread_count_independent() {
    // Two integer features, a clean additive-plus-noise target — the same shape
    // `objective_invariants.rs` uses, small enough to fit in well under a second at each
    // of the three thread counts.
    let n = 220usize;
    let x0: Vec<f32> = (0..n).map(|i| (i % 8 + 1) as f32).collect();
    let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
    let y: Vec<f32> = (0..n)
        .map(|i| x0[i] + 2.0 * x1[i] + (i % 7) as f32 * 0.1)
        .collect();
    let refs: Vec<&[f32]> = vec![&x0, &x1];
    let x = bin_columns(&refs, None, &BinConfig::default(), 0).unwrap();
    let cfg = Config {
        lambda_scale_invariant: false,
        max_delta_step_gated: t_boost_core::engine::GatedStepPolicy::Objective,
        n_trees: 40,
        learning_rate: 0.3,
        ..Config::default()
    };
    let sqe = SquaredError;
    assert_thread_count_independent("default-config real fit", |nt| {
        run_in_pool(nt, || {
            let model = Booster::with_config(cfg.clone())
                .fit(&x, &y, &fit_spec(&sqe))
                .unwrap();
            encode_model(&model).unwrap()
        })
    });
}

#[test]
fn real_bagged_fit_is_thread_count_independent() {
    // n_bags >= 2 (the outer-bag soup path): the bag partition, the per-bag OOB fits, and
    // the final soup average must all fold in a thread-count-independent order.
    let n = 260usize;
    let x0: Vec<f32> = (0..n).map(|i| (i % 7 + 1) as f32).collect();
    let x1: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
    let y: Vec<f32> = (0..n)
        .map(|i| 2.0 * x0[i] - x1[i] + (i % 9) as f32 * 0.05)
        .collect();
    let refs: Vec<&[f32]> = vec![&x0, &x1];
    let x = bin_columns(&refs, None, &BinConfig::default(), 0).unwrap();
    let cfg = Config {
        lambda_scale_invariant: false,
        max_delta_step_gated: t_boost_core::engine::GatedStepPolicy::Objective,
        n_trees: 30,
        learning_rate: 0.3,
        boosters: BoosterConfig {
            ensemble: EnsembleSpec::OuterBag {
                n_bags: 4,
                bag_subsample: 0.8,
                cell_refit: None,
            },
            ..BoosterConfig::default()
        },
        ..Config::default()
    };
    let sqe = SquaredError;
    assert_thread_count_independent("bagged real fit (n_bags=4)", |nt| {
        run_in_pool(nt, || {
            let model = Booster::with_config(cfg.clone())
                .fit(&x, &y, &fit_spec(&sqe))
                .unwrap();
            encode_model(&model).unwrap()
        })
    });
}

#[test]
fn real_bagged_fit_with_cell_refit_is_thread_count_independent() {
    // Same bagged shape, PLUS the §G1 OOB fANOVA cell-refit: the OOB accumulation and its
    // CG solve are additional fold-order-sensitive machinery this must also cover, mirroring
    // `engine::boost`'s own `cell_refit_outer_bag_is_g0_exact_and_thread_deterministic` at
    // the public integration-test level instead of an inline `#[cfg(test)]` unit test.
    let n = 240usize;
    let x0: Vec<f32> = (0..n).map(|i| (i % 8 + 1) as f32).collect();
    let x1: Vec<f32> = (0..n).map(|i| ((i / 8) % 6 + 1) as f32).collect();
    let y: Vec<f32> = (0..n)
        .map(|i| {
            let a = if x0[i] <= 4.0 { 1.5 } else { -2.0 };
            let b = if x1[i] <= 3.0 { 1.0 } else { -0.5 };
            let ab = if x0[i] <= 4.0 && x1[i] <= 3.0 {
                2.0
            } else {
                0.0
            };
            a + b + ab + (i % 13) as f32 * 0.02
        })
        .collect();
    let refs: Vec<&[f32]> = vec![&x0, &x1];
    let x = bin_columns(&refs, None, &BinConfig::default(), 0).unwrap();
    let cfg = Config {
        lambda_scale_invariant: false,
        max_delta_step_gated: t_boost_core::engine::GatedStepPolicy::Objective,
        n_trees: 30,
        learning_rate: 0.25,
        boosters: BoosterConfig {
            ensemble: EnsembleSpec::OuterBag {
                n_bags: 4,
                bag_subsample: 0.8,
                cell_refit: Some(CellRefit {
                    base: 50.0,
                    gamma: 2.0,
                }),
            },
            ..BoosterConfig::default()
        },
        ..Config::default()
    };
    let sqe = SquaredError;
    assert_thread_count_independent("bagged + cell-refit real fit", |nt| {
        run_in_pool(nt, || {
            let model = Booster::with_config(cfg.clone())
                .fit(&x, &y, &fit_spec(&sqe))
                .unwrap();
            encode_model(&model).unwrap()
        })
    });
}

#[test]
fn real_fit_with_categorical_features_is_thread_count_independent() {
    // One numeric + one native categorical (TS-encoded) axis — the K-fold cross-fit
    // encoding path (the shipped default) has its own fold-order-sensitive accumulation,
    // separate from the boosting loop's.
    let n = 200usize;
    let numeric: Vec<f32> = (0..n).map(|i| (i % 6) as f32).collect();
    let cats = ["low", "mid", "high"];
    let levels: Vec<String> = (0..n).map(|i| cats[i % 3].to_owned()).collect();
    let y: Vec<f32> = (0..n)
        .map(|i| numeric[i] + [0.0_f32, 5.0, 10.0][i % 3])
        .collect();
    let ts = TsConfig {
        leakage: LeakageScheme::KFold { k: 3 },
        smooth: Smooth::Fixed { m: 0.0 },
        min_data_per_group: 0.0,
        ..TsConfig::default()
    };
    let sqe = SquaredError;
    let cfg = Config {
        lambda_scale_invariant: false,
        max_delta_step_gated: t_boost_core::engine::GatedStepPolicy::Objective,
        n_trees: 25,
        learning_rate: 0.4,
        ..Config::default()
    };
    assert_thread_count_independent("categorical real fit", |nt| {
        run_in_pool(nt, || {
            let fitted = bin_train_columns(
                &[NumericColumn {
                    raw: FeatureId(0),
                    values: &numeric,
                }],
                &[CategoricalColumn {
                    raw: FeatureId(1),
                    id: TsEncodingId(0),
                    levels: &levels,
                    config: &ts,
                }],
                &y,
                None,
                None,
                &BinConfig::default(),
                7,
            )
            .unwrap();
            let model = Booster::with_config(cfg.clone())
                .fit_train(
                    &fitted.train,
                    &y,
                    &fit_spec(&sqe),
                    fitted.cat_encoders.clone(),
                )
                .unwrap();
            encode_model(&model).unwrap()
        })
    });
}
