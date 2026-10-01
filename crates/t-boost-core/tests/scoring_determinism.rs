//! Scoring-side determinism gate (spec §10.7 "predict at n_threads in {1,2,8} produces
//! bit-identical output arrays" / spec §13.4's companion [GATE]).
//!
//! `tests/determinism.rs` pins the FIT side; this file pins the SCORE side, across every
//! parallel scoring surface reachable from the public API:
//! - [`Model::score_trees`]/[`Model::predict_binned`] are included below as a thread-count-
//!   stability reference. NOTE (post packed-scorer perf fix): `score_trees` now itself routes
//!   through `ScoringBank`/`score_tile` internally (`Model::score_trees_prevalidated`), so its
//!   agreement with the `ScoringBank::Packed` checks below is partly self-referential — it
//!   re-confirms thread-count stability, not independent correctness. The genuinely independent
//!   bit-exact oracle (still a separate scalar implementation, untouched by that fix) is
//!   `engine::mod`'s own `score_trees_row_batch_and_rows_agree_bit_exactly_with_a_multi_table_
//!   correction` test, which cross-checks against `score_trees_row`/`score_trees_rows`.
//! - [`ScoringBank`] (both the `Packed` and `Wide` tree-representation variants) via the
//!   production [`score_tile`] kernel, tiled by [`CHUNK_ROWS`] and driven in parallel here
//!   exactly as `simd.rs`'s own doc describes the intended calling convention. Includes a
//!   model with a non-empty `correction` bank (the §G1 OOB cell-refit) — the regression
//!   this gate exists to catch, per the review that requested it.
//! - [`TableScoringBank::score_binned`] and [`TableModel`]/[`MultiClassTableModel`]'s
//!   `score_bank_binned`-backed path (via `score_raw`/`predict_binned`/`predict_proba`) —
//!   both are `par_iter_mut` over disjoint per-row output slots internally, no harness
//!   needed beyond installing the call inside a sized pool.
//!
//! Every check compares `to_bits()` of every output element at 1 vs 2 vs 8 threads, 1 as
//! the reference — a `to_bits` compare (not `==`) so a `NaN`-vs-`NaN` divergence would still
//! be caught (`NaN != NaN` under `==`, but `to_bits()` differs only if the actual bit
//! pattern moved).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use rayon::prelude::*;
use t_boost_core::engine::{CorrectionBank, CorrectionTable};
use t_boost_core::explain::{fixture_model, fixture_serve};
use t_boost_core::table_model::{MultiClassTableModel, TableModel};
use t_boost_core::{
    bin_columns, score_tile, AxisKind, AxisProvenance, BinConfig, BinnedMatrix, Booster,
    BorderGrid, Config, CredibilityFloor, FeatureId, FitSpec, InteractionPolicy, ModelSchema,
    MonotoneMap, RefMeasure, ScoringBank, ServeBinnedMatrix, Split, SquaredError, TableScoringBank,
    CHUNK_ROWS, SCHEMA_VERSION_UNLIFTED,
};

fn run_in_pool<R: Send>(n_threads: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap()
        .install(f)
}

fn assert_f32_bits_eq(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}: row {i} diverged ({x} vs {y})"
        );
    }
}

fn assert_f64_bits_eq(a: &[f64], b: &[f64], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}: row {i} diverged ({x} vs {y})"
        );
    }
}

/// Score every row of `x` through `bank` in parallel via the production [`score_tile`]
/// kernel, tiled by [`CHUNK_ROWS`] (the calling convention `simd.rs` documents: `score_tile`
/// is scalar WITHIN a tile, parallelism is across tiles).
fn score_bank_via_tiles(bank: &ScoringBank, x: &BinnedMatrix, f0: f32) -> Vec<f32> {
    let n_rows = x.n_rows as usize;
    let rows: Vec<u32> = (0..n_rows as u32).collect();
    let offset = vec![f0; n_rows];
    let mut out = vec![0.0_f32; n_rows];
    out.par_chunks_mut(CHUNK_ROWS)
        .zip(rows.par_chunks(CHUNK_ROWS))
        .for_each(|(out_chunk, rows_chunk)| {
            score_tile(bank, x, rows_chunk, Some(&offset), out_chunk).unwrap();
        });
    out
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

fn small_numeric_fit() -> (t_boost_core::Model, ServeBinnedMatrix) {
    let n = 200usize;
    let x0: Vec<f32> = (0..n).map(|i| (i % 8 + 1) as f32).collect();
    let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
    let y: Vec<f32> = (0..n)
        .map(|i| x0[i] + 1.5 * x1[i] + (i % 6) as f32 * 0.1)
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
    let model = Booster::with_config(cfg)
        .fit(&x, &y, &fit_spec(&sqe))
        .unwrap();
    (model, ServeBinnedMatrix(x))
}

#[test]
fn model_predict_and_score_trees_are_thread_count_stable_reference() {
    let (model, serve) = small_numeric_fit();

    let raw_at = |nt: usize| -> Vec<f32> {
        run_in_pool(nt, || {
            let mut out = vec![0.0_f32; serve.0.n_rows as usize];
            model.score_trees(&serve.0, None, &mut out).unwrap();
            out
        })
    };
    let r1 = raw_at(1);
    assert!(!r1.is_empty());
    assert_f32_bits_eq(&r1, &raw_at(2), "Model::score_trees 1v2");
    assert_f32_bits_eq(&r1, &raw_at(8), "Model::score_trees 1v8");

    let pred_at = |nt: usize| -> Vec<f32> {
        run_in_pool(nt, || model.predict_binned(&serve.0, None).unwrap())
    };
    let p1 = pred_at(1);
    assert_f32_bits_eq(&p1, &pred_at(2), "Model::predict_binned 1v2");
    assert_f32_bits_eq(&p1, &pred_at(8), "Model::predict_binned 1v8");
}

#[test]
fn scoring_bank_packed_scoring_matches_model_and_is_thread_count_stable() {
    let (model, serve) = small_numeric_fit();
    let bank = ScoringBank::from_model(&model).unwrap();
    assert!(matches!(bank, ScoringBank::Packed { .. }));

    let scores_at = |nt: usize| -> Vec<f32> {
        run_in_pool(nt, || score_bank_via_tiles(&bank, &serve.0, model.f0))
    };
    let s1 = scores_at(1);
    assert!(!s1.is_empty());
    assert_f32_bits_eq(&s1, &scores_at(2), "ScoringBank::Packed 1v2");
    assert_f32_bits_eq(&s1, &scores_at(8), "ScoringBank::Packed 1v8");

    // Cross-check against the sequential Model oracle in raw-score space: `score_tile`
    // never folds f0 in on its own (its doc: offset is used verbatim), so pre-folding f0
    // into `offset` (done inside `score_bank_via_tiles`) must reproduce `score_trees`'s raw
    // output exactly.
    let mut reference = vec![0.0_f32; serve.0.n_rows as usize];
    model.score_trees(&serve.0, None, &mut reference).unwrap();
    assert_f32_bits_eq(&s1, &reference, "ScoringBank::Packed vs Model::score_trees");
}

#[test]
fn scoring_bank_wide_axis_variant_is_thread_count_stable() {
    // Force the `Wide` fallback (split axes that don't fit in u8): pad grids/provenance
    // past 255 entries and redirect one split onto the new wide axis, mirroring
    // `scoring.rs`'s own `wide_axis_fallback_matches_model_tree_walk` unit test.
    let mut model = fixture_model();
    let wide_axis = 300usize;
    while model.grids.len() <= wide_axis {
        model.grids.push(BorderGrid {
            borders: vec![1.5],
            n_bins: 3,
            missing_bin: 0,
        });
        let raw = u32::try_from(model.provenance.len()).unwrap();
        model.provenance.push(AxisProvenance {
            raw: FeatureId(raw),
            kind: AxisKind::Numeric,
        });
        model.schema.feature_names.push(format!("f{raw}"));
        model.schema.feature_kinds.push(AxisKind::Numeric);
    }
    model.trees[0].1.splits[0] = Split {
        axis: u32::try_from(wide_axis).unwrap(),
        bin_le: 1,
        missing_left: false,
    };
    model.schema = ModelSchema {
        feature_names: model.schema.feature_names.clone(),
        feature_kinds: model.schema.feature_kinds.clone(),
        cat_encoders: model.schema.cat_encoders.clone(),
        class_labels: None,
        objective: model.schema.objective.clone(),
    };
    let bank = ScoringBank::from_model(&model).unwrap();
    assert!(matches!(bank, ScoringBank::Wide { .. }));

    // Synthetic rows varying the wide axis and the tree's other referenced axis (1) so every
    // leaf is exercised. `score_row` reads only the axes a tree's splits reference, so these
    // rows need not be a real `BinnedMatrix` (no `score_tile`/column-length machinery).
    let n_rows = 1500usize;
    let rows: Vec<Vec<u8>> = (0..n_rows)
        .map(|i| {
            let mut row = vec![2_u8; model.grids.len()];
            row[wide_axis] = (i % 3) as u8;
            row[1] = ((i / 3) % 3) as u8;
            row
        })
        .collect();
    let scores_at = |nt: usize| -> Vec<f32> {
        run_in_pool(nt, || {
            rows.par_iter()
                .map(|row| bank.score_row(row, model.f0).unwrap())
                .collect()
        })
    };
    let s1 = scores_at(1);
    assert!(!s1.is_empty());
    assert_f32_bits_eq(&s1, &scores_at(2), "ScoringBank::Wide 1v2");
    assert_f32_bits_eq(&s1, &scores_at(8), "ScoringBank::Wide 1v8");
}

/// Mirrors `crates/t-boost-core/src/scoring.rs`'s own (private) `fixture_correction`: a
/// realized pair {0,1} PLUS a main effect on axis 1 -- two tables, so there is more than one
/// summation term (a single-table bank can't distinguish a per-table-cast-then-sum bug from
/// the correct f64-sum-then-cast-once accumulation `score_trees_prevalidated` uses).
fn small_correction() -> CorrectionBank {
    CorrectionBank {
        tables: vec![
            CorrectionTable {
                axes: vec![0, 1],
                shape: vec![3, 3],
                bin_to_cell: vec![vec![0, 1, 2], vec![0, 1, 2]],
                values: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            },
            CorrectionTable {
                axes: vec![1],
                shape: vec![3],
                bin_to_cell: vec![vec![0, 1, 2]],
                values: vec![0.0, 0.1, 0.2],
            },
        ],
    }
}

#[test]
fn scoring_bank_with_correction_bank_is_thread_count_stable() {
    // The §G1 OOB cell-refit case: `ScoringBank` carries `model.correction` and adds its
    // delta at serve. Scoring a corrected model across thread counts is exactly the
    // regression this gate exists to pin.
    let mut model = fixture_model();
    model.correction = Some(small_correction());
    let bank = ScoringBank::from_model(&model).unwrap();
    let serve = fixture_serve();

    let scores_at = |nt: usize| -> Vec<f32> {
        run_in_pool(nt, || score_bank_via_tiles(&bank, &serve.0, model.f0))
    };
    let s1 = scores_at(1);
    assert!(!s1.is_empty());
    assert_f32_bits_eq(&s1, &scores_at(2), "corrected ScoringBank 1v2");
    assert_f32_bits_eq(&s1, &scores_at(8), "corrected ScoringBank 1v8");

    // Non-triviality: the correction must actually move the score, so this test can't pass
    // by having every thread count agree on a value that silently dropped the correction.
    let uncorrected = fixture_model();
    let uncorrected_bank = ScoringBank::from_model(&uncorrected).unwrap();
    let baseline = score_bank_via_tiles(&uncorrected_bank, &serve.0, uncorrected.f0);
    assert_ne!(
        s1.first().copied(),
        baseline.first().copied(),
        "correction had no effect on the score"
    );

    // Cross-check against the sequential Model oracle, which also carries the correction.
    let mut reference = vec![0.0_f32; serve.0.n_rows as usize];
    model.score_trees(&serve.0, None, &mut reference).unwrap();
    assert_f32_bits_eq(
        &s1,
        &reference,
        "corrected ScoringBank vs Model::score_trees",
    );
}

#[test]
fn table_scoring_bank_score_binned_is_thread_count_stable() {
    let (model, serve) = small_numeric_fit();
    let bank = model.explain(&serve, RefMeasure::default()).unwrap();
    let flat = TableScoringBank::from_bank(&bank, &model.schema.cat_encoders).unwrap();

    let scores_at = |nt: usize| -> Vec<f64> {
        run_in_pool(nt, || {
            let mut out = vec![0.0_f64; serve.0.n_rows as usize];
            flat.score_binned(&serve.0, &mut out).unwrap();
            out
        })
    };
    let s1 = scores_at(1);
    assert!(!s1.is_empty());
    assert_f64_bits_eq(&s1, &scores_at(2), "TableScoringBank::score_binned 1v2");
    assert_f64_bits_eq(&s1, &scores_at(8), "TableScoringBank::score_binned 1v8");
}

#[test]
fn table_model_score_raw_and_predict_binned_are_thread_count_stable() {
    // TableModel::score_raw routes through the pub(crate) score_bank_binned -- the
    // par_iter_mut path that (unlike TableScoringBank) also sums factored order-3 tables --
    // there is no public direct entry point to it, only via TableModel/MultiClassTableModel.
    let (model, serve) = small_numeric_fit();
    let bank = model.explain(&serve, RefMeasure::default()).unwrap();
    let tm = TableModel::from_model_and_bank(&model, bank);

    let raw_at =
        |nt: usize| -> Vec<f32> { run_in_pool(nt, || tm.score_raw(&serve.0, None).unwrap()) };
    let r1 = raw_at(1);
    assert!(!r1.is_empty());
    assert_f32_bits_eq(&r1, &raw_at(2), "TableModel::score_raw 1v2");
    assert_f32_bits_eq(&r1, &raw_at(8), "TableModel::score_raw 1v8");

    let pred_at =
        |nt: usize| -> Vec<f32> { run_in_pool(nt, || tm.predict_binned(&serve.0, None).unwrap()) };
    let p1 = pred_at(1);
    assert_f32_bits_eq(&p1, &pred_at(2), "TableModel::predict_binned 1v2");
    assert_f32_bits_eq(&p1, &pred_at(8), "TableModel::predict_binned 1v8");
}

fn small_multiclass_fit() -> (t_boost_core::MultiClassModel, ServeBinnedMatrix) {
    let n = 240usize;
    let x0: Vec<f32> = (0..n).map(|i| (i % 24) as f32).collect();
    let x1: Vec<f32> = (0..n).map(|i| (i % 6) as f32).collect();
    let y: Vec<f32> = x0
        .iter()
        .map(|&v| {
            if v < 8.0 {
                0.0
            } else if v < 16.0 {
                1.0
            } else {
                2.0
            }
        })
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
    let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let sqe = SquaredError; // ignored by fit_multiclass (softmax gradient is coupled)
    let model = Booster::with_config(cfg)
        .fit_multiclass(&x, &y, 3, &labels, &fit_spec(&sqe))
        .unwrap();
    (model, ServeBinnedMatrix(x))
}

#[test]
fn multiclass_table_model_predict_proba_is_thread_count_stable() {
    // The multiclass batch-predict surface, built on K independent TableModel::score_raw
    // calls (the SAME score_bank_binned path table_model_score_raw_and_predict_binned_are_
    // thread_count_stable pins for a single class) plus the softmax response layer.
    let (mc_model, serve) = small_multiclass_fit();
    let classes: Vec<TableModel> = mc_model
        .classes
        .iter()
        .map(|m| {
            let bank = m.explain(&serve, RefMeasure::default()).unwrap();
            TableModel::from_model_and_bank(m, bank)
        })
        .collect();
    let mc_tables = MultiClassTableModel {
        classes,
        class_labels: mc_model.class_labels.clone(),
        schema_version: SCHEMA_VERSION_UNLIFTED,
    };

    let raw_at =
        |nt: usize| -> Vec<f32> { run_in_pool(nt, || mc_tables.predict_raw(&serve.0).unwrap()) };
    let r1 = raw_at(1);
    assert!(!r1.is_empty());
    assert_f32_bits_eq(&r1, &raw_at(2), "MultiClassTableModel::predict_raw 1v2");
    assert_f32_bits_eq(&r1, &raw_at(8), "MultiClassTableModel::predict_raw 1v8");

    let proba_at =
        |nt: usize| -> Vec<f32> { run_in_pool(nt, || mc_tables.predict_proba(&serve.0).unwrap()) };
    let p1 = proba_at(1);
    assert_f32_bits_eq(&p1, &proba_at(2), "MultiClassTableModel::predict_proba 1v2");
    assert_f32_bits_eq(&p1, &proba_at(8), "MultiClassTableModel::predict_proba 1v8");
}
