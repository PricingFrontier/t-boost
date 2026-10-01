//! Gate G3 — the green spine (spec §13.1/§13.2, plan F7/F8 → §08). A real fitted
//! `Model` is turned into its purified `TableBank` by `Model::explain`, and the five I2
//! checks + the I1 feature budget are exercised against THAT bank: green on the real
//! decomposition, red with the correct `Invariant` when a table is perturbed. This is
//! the milestone where the invariant checks built in Phase 0 finally point at a real
//! model — "if these ever disagree there is no product."

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use t_boost_core::explain::{
    check_reconstruction, check_three_way_equal, fixture_model, fixture_over_budget_model,
    fixture_serve,
};
use t_boost_core::{
    assert_exact_decomposition, bin_columns, check_feature_budget, BinConfig, Booster, Config,
    CredibilityFloor, FitSpec, InteractionPolicy, Invariant, Model, MonotoneMap, PbError,
    RefMeasure, ServeBinnedMatrix, SquaredError,
};

fn fit_additive() -> (Model, ServeBinnedMatrix) {
    // An additive piecewise-constant target on two features, recovered exactly (λ=0).
    let n = 80usize;
    let x0: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
    let x1: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
    let y: Vec<f32> = (0..n)
        .map(|i| {
            let a = if x0[i] <= 3.0 { 10.0 } else { 20.0 };
            let b = if x1[i] <= 2.0 { 5.0 } else { 0.0 };
            a + b
        })
        .collect();
    let cols: Vec<&[f32]> = vec![&x0, &x1];
    let x = bin_columns(&cols, None, &BinConfig::default(), 0).unwrap();
    let spec = FitSpec {
        loss: &SquaredError,
        weight: None,
        exposure: None,
        monotone: MonotoneMap::new(),
        interaction: InteractionPolicy::default(),
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed: 0,
    };
    let model = Booster::with_config(Config {
        lambda_scale_invariant: false,
        max_delta_step_gated: t_boost_core::engine::GatedStepPolicy::Objective,
        n_trees: 30,
        learning_rate: 1.0,
        lambda: 0.0,
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
    })
    .fit(&x, &y, &spec)
    .unwrap();
    (model, ServeBinnedMatrix(x))
}

#[test]
fn g3_real_model_passes_all_five_checks() {
    let (model, x) = fit_additive();
    let bank = model.explain(&x, RefMeasure::default()).unwrap();
    assert_exact_decomposition(&model, &bank, &x).expect("exact decomposition must hold");
    check_feature_budget(&model).expect("feature budget must hold");
}

#[test]
fn g3_fixture_model_passes_under_both_measures() {
    let model = fixture_model();
    let x = fixture_serve();
    for w in [RefMeasure::Uniform, RefMeasure::default()] {
        let bank = model.explain(&x, w).unwrap();
        assert_exact_decomposition(&model, &bank, &x).expect("exact decomposition must hold");
    }
    check_feature_budget(&model).expect("feature budget must hold");
}

#[test]
fn g3_negative_reconstruction_is_caught() {
    let model = fixture_model();
    let x = fixture_serve();
    let mut bank = model.explain(&x, RefMeasure::Uniform).unwrap();
    // Perturb a single main-effect cell: tables no longer reconstruct the ensemble.
    let main = bank
        .tables
        .iter_mut()
        .find(|t| t.u.order() == 1)
        .expect("a main effect");
    main.values.add(&[1], 2.0).unwrap();
    match check_reconstruction(&model, &bank) {
        Err(PbError::InvariantViolated { invariant }) => {
            assert_eq!(invariant, Invariant::Reconstruction);
        }
        other => panic!("expected Reconstruction violation, got {other:?}"),
    }
    // And the end-to-end three-way check catches it too.
    assert!(matches!(
        check_three_way_equal(&model, &bank),
        Err(PbError::InvariantViolated {
            invariant: Invariant::ThreeWayEqual
        })
    ));
}

#[test]
fn g3_over_budget_model_violates_feature_budget() {
    match check_feature_budget(&fixture_over_budget_model()) {
        Err(PbError::InvariantViolated { invariant }) => {
            assert_eq!(invariant, Invariant::FeatureBudget);
        }
        other => panic!("expected FeatureBudget violation, got {other:?}"),
    }
}
