//! The §13.1 invariant matrix, objective columns (plan M2 / Gate G4): every v1 loss,
//! fit end-to-end, produces an `Exact` model whose purified `TableBank` passes all five
//! I2 checks under both reference measures. Because a loss is orthogonal to tree shape
//! (§05.8 — it sees only `(y, F, w)`), the firewall stays `Exact` by construction; this
//! gate *proves* it per objective rather than asserting it. Also exercises the Poisson
//! `max_delta_step` stability path and the exposure-offset frequency fit.
//!
//! [`i2_matrix_sweeps_loss_x_order_x_measure_x_monotone_x_early_stop`] extends G4 into the
//! full spec §13.1 promised sweep — `{loss} x {max_interaction_order} x {ref-measure} x
//! {monotone} x {early-termination}` — and
//! [`i2_matrix_cell_refit_stays_exact_per_loss`] /
//! [`i2_matrix_pruning_produces_valid_table_models_per_loss`] add the two post-matrix
//! features (OOB cell-refit, table pruning) the matrix never covered.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use t_boost_core::boosters::CellRefit;
use t_boost_core::prune::{prune_model_to_tables, PruneConfig};
use t_boost_core::{
    assert_exact_decomposition, bin_columns, BinConfig, Booster, BoosterConfig, Config,
    CredibilityFloor, EnsembleSpec, ExactnessMode, FitSpec, Gamma, InteractionPolicy, Link,
    Logistic, Loss, LossId, MonoSign, MonotoneMap, Poisson, RefMeasure, ServeBinnedMatrix,
    SquaredError, Tweedie,
};

fn spec<'a>(loss: &'a dyn Loss, exposure: Option<&'a [f32]>) -> FitSpec<'a> {
    FitSpec {
        loss,
        weight: None,
        exposure,
        monotone: MonotoneMap::new(),
        interaction: InteractionPolicy::default(),
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed: 0,
    }
}

fn cfg() -> Config {
    Config {
        lambda_scale_invariant: false,
        max_delta_step_gated: t_boost_core::engine::GatedStepPolicy::Objective,
        n_trees: 25,
        learning_rate: 0.3,
        lambda: 1.0,
        min_split_gain: 0.0,
        max_delta_step: None,
        sampling: Default::default(),
        hist_precision: Default::default(),
        l1_leaf: 0.0,
        colsample_bytree: 1.0,
        learning_rate_decay: 0.0,
        validation_fraction: None,
        early_stopping_rounds: 50,
        early_stopping_adaptive: None,
        early_stopping_min_delta: 0.0,
        interaction_gain_hurdle: 0.0,
        interaction_gain_hurdle_mode: t_boost_core::InteractionGainHurdleMode::Fixed,
        leaf_refine_steps: 0,
        leaf_refine_backtracks: 4,
        refine_closed_form_tier2: false,
        incremental_mu: false,
        boosters: Default::default(),
    }
}

/// Two integer features over `n` rows, binned once and reused across objectives.
fn features(n: usize) -> (Vec<f32>, Vec<f32>) {
    let x0: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
    let x1: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
    (x0, x1)
}

#[test]
fn g4_every_objective_passes_the_five_invariant_checks() {
    let n = 96usize;
    let (x0, x1) = features(n);

    // Per-objective in-domain targets (a real ≤order-2 structure on each link scale).
    let y_sqe: Vec<f32> = (0..n).map(|i| x0[i] + 2.0 * x1[i]).collect();
    let y_logit: Vec<f32> = (0..n)
        .map(|i| {
            if (x0[i] <= 3.0) ^ (x1[i] <= 2.0) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let y_pois: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect(); // counts ≥ 0 (incl. 0)
    let y_gamma: Vec<f32> = (0..n).map(|i| 1.0 + x0[i] + 0.5 * x1[i]).collect(); // > 0
    let y_tweedie: Vec<f32> = (0..n)
        .map(|i| (x0[i] + x1[i] - 4.0).max(0.0)) // ≥ 0, with genuine zeros
        .collect();

    let tweedie = Tweedie::new(1.5).unwrap();
    let cases: Vec<(&str, &dyn Loss, &[f32], Link)> = vec![
        ("squared_error", &SquaredError, &y_sqe, Link::Identity),
        ("logistic", &Logistic, &y_logit, Link::Logit),
        ("poisson", &Poisson, &y_pois, Link::Log),
        ("gamma", &Gamma, &y_gamma, Link::Log),
        ("tweedie", &tweedie, &y_tweedie, Link::Log),
    ];

    for (name, loss, y, link) in cases {
        let refs: Vec<&[f32]> = vec![&x0, &x1];
        let x = bin_columns(&refs, None, &BinConfig::default(), 0).unwrap();
        let model = Booster::with_config(cfg())
            .fit(&x, y, &spec(loss, None))
            .unwrap_or_else(|e| panic!("{name} fit failed: {e:?}"));
        assert_eq!(model.mode, ExactnessMode::Exact, "{name} must be Exact");
        assert_eq!(model.link, link, "{name} link");
        assert_eq!(model.schema.objective.link, link);

        let serve = ServeBinnedMatrix(x);
        for w in [RefMeasure::default(), RefMeasure::Uniform] {
            let bank = model
                .explain(&serve, w.clone())
                .unwrap_or_else(|e| panic!("{name} explain({w:?}) failed: {e:?}"));
            assert_exact_decomposition(&model, &bank, &serve)
                .unwrap_or_else(|e| panic!("{name} I2 gates failed under {w:?}: {e:?}"));
        }
    }
}

#[test]
fn poisson_default_fit_is_max_delta_step_stabilized() {
    // Poisson advertises max_delta_step = Some(0.7); with no Config override the engine
    // resolves it, so every per-tree leaf step is capped at lr·0.7 — the fit stays finite
    // even on data that would otherwise drive exp(F) explosive.
    let n = 80usize;
    let (x0, x1) = features(n);
    let y: Vec<f32> = (0..n).map(|i| (i % 9) as f32).collect();
    let refs: Vec<&[f32]> = vec![&x0, &x1];
    let x = bin_columns(&refs, None, &BinConfig::default(), 0).unwrap();
    let lr = 0.5_f32;
    let model = Booster::with_config(Config {
        lambda_scale_invariant: false,
        max_delta_step_gated: t_boost_core::engine::GatedStepPolicy::Objective,
        n_trees: 30,
        learning_rate: lr,
        lambda: 0.1,
        min_split_gain: 0.0,
        max_delta_step: None, // ⇒ falls back to Poisson's Some(0.7)
        sampling: Default::default(),
        hist_precision: Default::default(),
        l1_leaf: 0.0,
        colsample_bytree: 1.0,
        learning_rate_decay: 0.0,
        validation_fraction: None,
        early_stopping_rounds: 50,
        early_stopping_adaptive: None,
        early_stopping_min_delta: 0.0,
        interaction_gain_hurdle: 0.0,
        interaction_gain_hurdle_mode: t_boost_core::InteractionGainHurdleMode::Fixed,
        leaf_refine_steps: 0,
        leaf_refine_backtracks: 4,
        refine_closed_form_tier2: false,
        incremental_mu: false,
        boosters: Default::default(),
    })
    .fit(&x, &y, &spec(&Poisson, None))
    .unwrap();
    assert_eq!(model.schema.objective.loss, LossId::Poisson);
    // Every leaf value is bounded by lr·δ = 0.5·0.7 = 0.35 (the clamp is per Newton step).
    let bound = f64::from(lr) * 0.7 + 1e-5;
    for (_, tree) in &model.trees {
        for &v in &tree.leaves {
            assert!(
                f64::from(v).abs() <= bound,
                "leaf {v} exceeds lr·δ bound {bound}"
            );
        }
    }
    // And predictions are finite everywhere.
    for r in 0..n {
        let bins: Vec<u8> = x.data.iter().map(|c| c[r]).collect();
        assert!(model.ensemble_f64(&bins).unwrap().is_finite());
    }
}

#[test]
fn i2_matrix_sweeps_loss_x_order_x_measure_x_monotone_x_early_stop() {
    // The spec §13.1 promised sweep: {loss} x {max_interaction_order 1,2,3} x
    // {ProductMarginals, Uniform} x {monotone off/on} x {early-termination off/on} — a full
    // 5x3x2x2x2 = 120-cell cross product (appendix L501-514: G4 above only ever swept loss x
    // ref-measure). Bounded for per-PR runtime: n=60 rows, n_trees <= 40, so every cell's
    // fit+explain+assert_exact_decomposition stays cheap; the measured wall time is printed
    // and asserted against a generous ceiling so a future slowdown is visible rather than
    // silently eating CI budget instead of being caught here.
    let t0 = std::time::Instant::now();
    let n = 60usize;
    let (x0, x1) = features(n);

    let y_sqe: Vec<f32> = (0..n).map(|i| x0[i] + 2.0 * x1[i]).collect();
    let y_logit: Vec<f32> = (0..n)
        .map(|i| {
            if (x0[i] <= 3.0) ^ (x1[i] <= 2.0) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let y_pois: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
    let y_gamma: Vec<f32> = (0..n).map(|i| 1.0 + x0[i] + 0.5 * x1[i]).collect();
    let y_tweedie: Vec<f32> = (0..n).map(|i| (x0[i] + x1[i] - 4.0).max(0.0)).collect();
    let tweedie = Tweedie::new(1.5).unwrap();
    let cases: Vec<(&str, &dyn Loss, &[f32])> = vec![
        ("squared_error", &SquaredError, &y_sqe),
        ("logistic", &Logistic, &y_logit),
        ("poisson", &Poisson, &y_pois),
        ("gamma", &Gamma, &y_gamma),
        ("tweedie", &tweedie, &y_tweedie),
    ];

    const ORDERS: [u8; 3] = [1, 2, 3];
    const MONOTONE_ON: [bool; 2] = [false, true];
    const EARLY_STOP_ON: [bool; 2] = [false, true];

    let mut n_configs = 0usize;
    for (name, loss, y) in cases {
        let refs: Vec<&[f32]> = vec![&x0, &x1];
        let x = bin_columns(&refs, None, &BinConfig::default(), 0).unwrap();

        for &order in &ORDERS {
            for measure in [
                RefMeasure::ProductMarginals { laplace: 1.0 },
                RefMeasure::Uniform,
            ] {
                for &mono_on in &MONOTONE_ON {
                    for &es_on in &EARLY_STOP_ON {
                        n_configs += 1;
                        let mut monotone = MonotoneMap::new();
                        if mono_on {
                            // "f0" is the default name the engine assigns the first unnamed
                            // column (bin_columns(..., None, ...) below) — the same convention
                            // engine/boost.rs's own inline monotone tests rely on.
                            monotone.insert("f0".into(), MonoSign::Increasing);
                        }
                        let fit_spec = FitSpec {
                            loss,
                            weight: None,
                            exposure: None,
                            monotone,
                            interaction: InteractionPolicy {
                                max_order: order,
                                ..Default::default()
                            },
                            credibility: CredibilityFloor::default(),
                            fixed_holdout: None,
                            bag_groups: None,
                            seed: 0,
                        };
                        let config = Config {
                            lambda_scale_invariant: false,
                            max_delta_step_gated: t_boost_core::engine::GatedStepPolicy::Objective,
                            n_trees: if es_on { 40 } else { 15 },
                            learning_rate: 0.3,
                            validation_fraction: if es_on { Some(0.3) } else { None },
                            early_stopping_rounds: if es_on { 3 } else { 50 },
                            refine_closed_form_tier2: false,
                            ..Default::default()
                        };
                        let ctx = format!(
                            "{name} order={order} measure={measure:?} mono={mono_on} es={es_on}"
                        );
                        let model = Booster::with_config(config)
                            .fit(&x, y, &fit_spec)
                            .unwrap_or_else(|e| panic!("{ctx}: fit failed: {e:?}"));
                        assert_eq!(model.mode, ExactnessMode::Exact, "{ctx}: must be Exact");
                        let serve = ServeBinnedMatrix(x.clone());
                        let bank = model
                            .explain(&serve, measure.clone())
                            .unwrap_or_else(|e| panic!("{ctx}: explain failed: {e:?}"));
                        assert_exact_decomposition(&model, &bank, &serve)
                            .unwrap_or_else(|e| panic!("{ctx}: I2 gates failed: {e:?}"));
                    }
                }
            }
        }
    }
    assert_eq!(
        n_configs,
        5 * 3 * 2 * 2 * 2,
        "full cross product must be 120 cells"
    );
    let elapsed = t0.elapsed();
    println!("i2 matrix: {n_configs} configs in {elapsed:?}");
    assert!(
        elapsed.as_secs() < 60,
        "i2 matrix took {elapsed:?} — over the ~60s per-PR budget; switch to a fractional \
         pairwise-covering design (see the finding, appendix L501-514)"
    );
}

#[test]
fn i2_matrix_cell_refit_stays_exact_per_loss() {
    // Post-matrix feature axis (appendix L501-514): the OOB fANOVA cell-refit
    // (crates/t-boost-core/src/cell_refit.rs, 785 lines) had never appeared in any file
    // under tests/. `EnsembleSpec::OuterBag` with `cell_refit: Some(..)` is the product-facing
    // way to get a cell-refit-corrected `Model` directly from `Booster::fit` (requires
    // `n_bags >= 2` so an out-of-bag residual exists). The correction attaches to the `Model`
    // — it never converts it to a `TableModel` — so the SAME `assert_exact_decomposition`
    // check applies unchanged; this is exactly the "G1 keeps G0 exact" claim that was never
    // actually exercised by the shared invariant fixture.
    let n = 96usize; // a bit more room than the main matrix so bagging has real OOB coverage
    let (x0, x1) = features(n);
    let y_sqe: Vec<f32> = (0..n).map(|i| x0[i] + 2.0 * x1[i]).collect();
    let y_logit: Vec<f32> = (0..n)
        .map(|i| {
            if (x0[i] <= 3.0) ^ (x1[i] <= 2.0) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let y_pois: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
    let y_gamma: Vec<f32> = (0..n).map(|i| 1.0 + x0[i] + 0.5 * x1[i]).collect();
    let y_tweedie: Vec<f32> = (0..n).map(|i| (x0[i] + x1[i] - 4.0).max(0.0)).collect();
    let tweedie = Tweedie::new(1.5).unwrap();
    let cases: Vec<(&str, &dyn Loss, &[f32])> = vec![
        ("squared_error", &SquaredError, &y_sqe),
        ("logistic", &Logistic, &y_logit),
        ("poisson", &Poisson, &y_pois),
        ("gamma", &Gamma, &y_gamma),
        ("tweedie", &tweedie, &y_tweedie),
    ];

    for (name, loss, y) in cases {
        let refs: Vec<&[f32]> = vec![&x0, &x1];
        let x = bin_columns(&refs, None, &BinConfig::default(), 0).unwrap();
        let config = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: t_boost_core::engine::GatedStepPolicy::Objective,
            n_trees: 20,
            learning_rate: 0.3,
            refine_closed_form_tier2: false,
            boosters: BoosterConfig {
                ensemble: EnsembleSpec::OuterBag {
                    n_bags: 2,
                    bag_subsample: 0.8,
                    cell_refit: Some(CellRefit {
                        base: 1.0,
                        gamma: 0.0,
                    }),
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let model = Booster::with_config(config)
            .fit(&x, y, &spec(loss, None))
            .unwrap_or_else(|e| panic!("{name} cell-refit fit failed: {e:?}"));
        assert_eq!(
            model.mode,
            ExactnessMode::Exact,
            "{name} cell-refit model must be Exact"
        );
        let serve = ServeBinnedMatrix(x);
        for w in [RefMeasure::default(), RefMeasure::Uniform] {
            let bank = model
                .explain(&serve, w.clone())
                .unwrap_or_else(|e| panic!("{name} cell-refit explain({w:?}) failed: {e:?}"));
            assert_exact_decomposition(&model, &bank, &serve)
                .unwrap_or_else(|e| panic!("{name} cell-refit I2 gates failed under {w:?}: {e:?}"));
        }
    }
}

#[test]
fn i2_matrix_pruning_produces_valid_table_models_per_loss() {
    // Post-matrix feature axis (appendix L501-514): table pruning
    // (crates/t-boost-core/src/prune.rs, 1414 lines) had never appeared in any file under
    // tests/. `prune_model_to_tables` converts a `Model` into a `TableModel` (tables-only) — a
    // DIFFERENT type from `assert_exact_decomposition`'s `(Model, TableBank)` signature, so the
    // five I2 checks cannot run on it verbatim. The check here is the reconstruction-equivalent
    // that IS meaningful for a pruned artifact: `TableModel::validate()` (the type's own §10
    // load gate) plus a tight prediction match against the original ensemble on a no-op prune
    // (report.dropped.is_empty()) — exactly the "the pruned bank is a NEW exact model
    // reconstructed against its own LUT-sum" claim prune.rs's own module doc makes, and (below)
    // one representative GENUINELY reduced case, so this isn't only exercising the trivial path.
    let n = 96usize;
    let (x0, x1) = features(n);
    let y_sqe: Vec<f32> = (0..n).map(|i| x0[i] + 2.0 * x1[i]).collect();
    let y_logit: Vec<f32> = (0..n)
        .map(|i| {
            if (x0[i] <= 3.0) ^ (x1[i] <= 2.0) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let y_pois: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
    let y_gamma: Vec<f32> = (0..n).map(|i| 1.0 + x0[i] + 0.5 * x1[i]).collect();
    let y_tweedie: Vec<f32> = (0..n).map(|i| (x0[i] + x1[i] - 4.0).max(0.0)).collect();
    let tweedie = Tweedie::new(1.5).unwrap();
    let cases: Vec<(&str, &dyn Loss, &[f32])> = vec![
        ("squared_error", &SquaredError, &y_sqe),
        ("logistic", &Logistic, &y_logit),
        ("poisson", &Poisson, &y_pois),
        ("gamma", &Gamma, &y_gamma),
        ("tweedie", &tweedie, &y_tweedie),
    ];

    for (name, loss, y) in cases {
        let refs: Vec<&[f32]> = vec![&x0, &x1];
        let x = bin_columns(&refs, None, &BinConfig::default(), 0).unwrap();
        let model = Booster::with_config(cfg())
            .fit(&x, y, &spec(loss, None))
            .unwrap_or_else(|e| panic!("{name} fit failed: {e:?}"));
        assert_eq!(model.mode, ExactnessMode::Exact, "{name} must be Exact");
        let serve = ServeBinnedMatrix(x.clone());

        // No-op prune: the model's OWN predictions as the held-out target, so every table
        // earns its keep and nothing is dropped (mirrors prune.rs's own
        // `prune_model_to_tables_keeps_all_on_genuine_signal`, extended across every loss).
        let y_self = model.predict(&x, None).unwrap();
        let w = vec![1.0_f32; n];
        let sel: Vec<usize> = (0..n).collect();
        let (tm, report) = prune_model_to_tables(
            &model,
            &serve,
            &y_self,
            &w,
            None,
            loss,
            RefMeasure::default(),
            false,
            &sel,
            2,
            &PruneConfig::default(),
        )
        .unwrap_or_else(|e| panic!("{name} prune_model_to_tables failed: {e:?}"));
        tm.validate()
            .unwrap_or_else(|e| panic!("{name} pruned TableModel failed validate(): {e:?}"));
        let ens = model.predict(&x, None).unwrap();
        let got = tm.predict_binned(&x, None).unwrap();
        for (i, (e, g)) in ens.iter().zip(&got).enumerate() {
            assert!(
                (e - g).abs() < 1e-4,
                "{name} row {i}: pruned TableModel prediction {g} != ensemble prediction {e} \
                 (dropped={:?}) — a no-op prune must reconstruct the ensemble exactly",
                report.dropped
            );
        }
    }

    // One representative GENUINELY reduced case: real pair-interaction structure, then prune
    // against a zero-signal held-out target (constant == f0) so the interaction table reads as
    // pure overfit and gets dropped — verifying a truly pruned, reduced TableModel (not just
    // the no-op case above) still validates as well-formed.
    let y_interact: Vec<f32> = (0..n)
        .map(|i| x0[i] + 2.0 * x1[i] + if x0[i] > 3.0 && x1[i] > 2.0 { 5.0 } else { 0.0 })
        .collect();
    let refs: Vec<&[f32]> = vec![&x0, &x1];
    let x = bin_columns(&refs, None, &BinConfig::default(), 0).unwrap();
    let model = Booster::with_config(cfg())
        .fit(&x, &y_interact, &spec(&SquaredError, None))
        .unwrap();
    let serve = ServeBinnedMatrix(x.clone());
    let bank_pre = model.explain(&serve, RefMeasure::default()).unwrap();
    let y_flat = vec![bank_pre.f0 as f32; n];
    let w = vec![1.0_f32; n];
    let sel: Vec<usize> = (0..n).collect();
    let (tm_pruned, report) = prune_model_to_tables(
        &model,
        &serve,
        &y_flat,
        &w,
        None,
        &SquaredError,
        RefMeasure::default(),
        false,
        &sel,
        2,
        &PruneConfig::default(),
    )
    .unwrap();
    tm_pruned
        .validate()
        .unwrap_or_else(|e| panic!("genuinely-pruned TableModel failed validate(): {e:?}"));
    assert!(
        !report.dropped.is_empty(),
        "expected the zero-signal held-out target to drop >= 1 overfit interaction table \
         (dropped={:?}); if this now fails, the pruning heuristic's behavior changed under \
         test — investigate rather than loosen this fixture",
        report.dropped
    );
}

#[test]
fn poisson_exposure_fit_explains_exactly() {
    // The exposure-offset frequency path: offset = log(e) folded into raw, exposure-
    // weighted intercept. The fitted model still decomposes losslessly (G4 + §05.5).
    let n = 64usize;
    let (x0, x1) = features(n);
    let y: Vec<f32> = (0..n).map(|i| (i % 4) as f32).collect();
    let e: Vec<f32> = (0..n).map(|i| 0.5 + (i % 3) as f32).collect(); // exposure > 0
    let refs: Vec<&[f32]> = vec![&x0, &x1];
    let x = bin_columns(&refs, None, &BinConfig::default(), 0).unwrap();
    let model = Booster::with_config(cfg())
        .fit(&x, &y, &spec(&Poisson, Some(&e)))
        .unwrap();
    assert_eq!(model.mode, ExactnessMode::Exact);
    let serve = ServeBinnedMatrix(x);
    let bank = model.explain(&serve, RefMeasure::default()).unwrap();
    assert_exact_decomposition(&model, &bank, &serve).unwrap();
}
