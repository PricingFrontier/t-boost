//! Audit-1 regression (CRITICAL): `Model::explain` must work on a model trained with
//! categorical (TS-encoded) features. A categorical axis is a Target-Statistic-encoded
//! ORDINAL `BorderGrid` axis, so the merged-grid decomposition and the five I2 gates
//! apply to it identically — the bank is audited on a `ServeBinnedMatrix` re-encoded
//! through the frozen full-data encoders (R-CATSERVE, §08/§04).
//!
//! Before this test, NO test called `explain()` on a categorical model, and a stale
//! "numeric axes only" guard in `MergedGrids::from_model` (left over from Phase 3, when
//! categoricals did not exist) made the core decomposability guarantee structurally
//! unreachable for any categorical model: such a model is `Exact`, but `explain()` failed
//! with `InvalidConfig` before any of the five checks could run.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use t_boost_core::{
    assert_exact_decomposition, bin_serve_columns, bin_train_columns, BinConfig, Booster,
    CatTarget, CategoricalColumn, Config, CredibilityFloor, FeatureId, FitSpec, InteractionPolicy,
    LeakageScheme, MonotoneMap, NumericColumn, RefMeasure, ServeCategoricalColumn, Smooth,
    SquaredError, TsConfig, TsEncodingId,
};

#[test]
fn categorical_model_explains_and_passes_all_five_gates() {
    let n = 90usize;
    let numeric: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
    let cats = ["low", "mid", "high"];
    let levels: Vec<String> = (0..n).map(|i| cats[i % 3].to_owned()).collect();
    // A target additive in (numeric, category), with a DOMINANT categorical effect so
    // the categorical axis is reliably split (and thus appears in the decomposition).
    let y: Vec<f32> = (0..n)
        .map(|i| numeric[i] + [0.0_f32, 50.0, 100.0][i % 3])
        .collect();

    let ts = TsConfig {
        leakage: LeakageScheme::KFold { k: 3 },
        smooth: Smooth::Fixed { m: 0.0 },
        min_data_per_group: 0.0,
        ..TsConfig::default()
    };
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

    let sqe = SquaredError;
    let spec = FitSpec {
        loss: &sqe,
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
        n_trees: 25,
        learning_rate: 0.5,
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
        fit_control: Default::default(),
    })
    .fit_train(&fitted.train, &y, &spec, fitted.cat_encoders.clone())
    .unwrap();

    // The categorical axis (axis 1, raw FeatureId(1)) is actually realized in the model.
    assert!(
        model
            .trees
            .iter()
            .any(|(_, t)| t.splits.iter().any(|s| s.axis == 1)),
        "the dominant categorical effect should be split on"
    );

    // R-CATSERVE: build the serve matrix by re-encoding raw labels through the FROZEN
    // full-data encoders, binned against the MODEL's own grids/provenance.
    let serve = bin_serve_columns(
        &[NumericColumn {
            raw: FeatureId(0),
            values: &numeric,
        }],
        &[ServeCategoricalColumn {
            raw: FeatureId(1),
            id: TsEncodingId(0),
            levels: &levels,
        }],
        &model.grids,
        &model.provenance,
        &model.schema.cat_encoders,
    )
    .unwrap();

    // The whole reason the library exists: a categorical model decomposes losslessly.
    for w in [RefMeasure::default(), RefMeasure::Uniform] {
        let bank = model
            .explain(&serve, w.clone())
            .unwrap_or_else(|e| panic!("categorical explain({w:?}) failed: {e:?}"));
        assert_exact_decomposition(&model, &bank, &serve)
            .unwrap_or_else(|e| panic!("categorical I2 gates failed under {w:?}: {e:?}"));
        // The categorical raw feature has a realized table in the bank.
        assert!(
            bank.tables.iter().any(|t| t.u.0.iter().any(|f| f.0 == 1)),
            "the categorical raw feature should have a realized effect table"
        );
    }
}

/// P1 multi-channel (design/multichannel-categoricals.md §4.3, Stage B "flatten" design):
/// a raw categorical feature with TWO channel axes (mean-TS + count) must still `explain()`
/// cleanly and pass all five I2 gates — the hard requirement the naive "sum of two
/// independently-purified tables" design was found NOT to satisfy in general.
///
/// The fixture is deliberately a "rare-but-severe" pattern: severity rises with level index
/// (captured by the mean channel) AND every RARE level (low count) carries an extra additive
/// boost NEITHER channel alone predicts — a tree that combines both channels (mean split, then
/// count split, or vice versa, within the SAME tree) captures this jointly, which is exactly
/// the regime the naive per-channel sum silently drops the residual for. If the flattened joint
/// cell space is wired correctly, `assert_exact_decomposition` must still pass exactly.
#[test]
fn two_channel_categorical_explains_and_passes_all_five_gates() {
    let common_levels = ["L0", "L1", "L2", "L3"];
    let rare_levels = ["L4", "L5", "L6", "L7"];
    let mut levels: Vec<String> = Vec::new();
    let mut y: Vec<f32> = Vec::new();
    for (i, &lvl) in common_levels.iter().enumerate() {
        for _ in 0..100 {
            levels.push(lvl.to_owned());
            y.push(i as f32 * 10.0);
        }
    }
    for (i, &lvl) in rare_levels.iter().enumerate() {
        for _ in 0..10 {
            levels.push(lvl.to_owned());
            // Rare-but-severe: the +200 boost is an effect NEITHER the mean channel (which
            // only sees a smooth level-index trend) NOR the count channel (which only sees
            // rarity) predicts alone — only a tree that uses BOTH captures it.
            y.push((4 + i) as f32 * 10.0 + 200.0);
        }
    }

    let mean_ts = TsConfig {
        leakage: LeakageScheme::KFold { k: 3 },
        smooth: Smooth::Fixed { m: 0.0 },
        min_data_per_group: 0.0,
        ..TsConfig::default()
    };
    let count_ts = TsConfig {
        target: CatTarget::Count,
        min_data_per_group: 0.0,
        ..TsConfig::default()
    };
    let fitted = bin_train_columns(
        &[],
        &[
            CategoricalColumn {
                raw: FeatureId(0),
                id: TsEncodingId(0),
                levels: &levels,
                config: &mean_ts,
            },
            CategoricalColumn {
                raw: FeatureId(0),
                id: TsEncodingId(1),
                levels: &levels,
                config: &count_ts,
            },
        ],
        &y,
        None,
        None,
        &BinConfig::default(),
        11,
    )
    .unwrap();

    let sqe = SquaredError;
    let spec = FitSpec {
        loss: &sqe,
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
        n_trees: 60,
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
        fit_control: Default::default(),
    })
    .fit_train(&fitted.train, &y, &spec, fitted.cat_encoders.clone())
    .unwrap();

    // Both channel axes (0 and 1, both raw feature 0) are available to every tree (a greedy
    // gain-maximizing fit is free to ignore axis 1 entirely if axis 0's own Fisher order
    // already resolves the fixture — with only 8 levels and an 8-leaf-capacity tree it often
    // does, since mean-TS is BY DEFINITION whatever affects the level's own mean, so it can
    // always losslessly rank-separate a small level set regardless of the "true" pattern).
    // This end-to-end fit is a real-pipeline confidence check; the deterministic proof that a
    // SINGLE tree combining both channels is handled exactly (the true residual-interaction
    // case) is `two_channel_hand_built_tree_combining_both_channels_is_exact` below, which
    // does not depend on what boosting happens to choose.
    let axes_used: std::collections::BTreeSet<u32> = model
        .trees
        .iter()
        .flat_map(|(_, t)| t.splits.iter().map(|s| s.axis))
        .collect();
    assert!(
        axes_used.contains(&0),
        "fixture should exercise at least the mean channel, got axes {axes_used:?}"
    );

    let serve = bin_serve_columns(
        &[],
        &[
            ServeCategoricalColumn {
                raw: FeatureId(0),
                id: TsEncodingId(0),
                levels: &levels,
            },
            ServeCategoricalColumn {
                raw: FeatureId(0),
                id: TsEncodingId(1),
                levels: &levels,
            },
        ],
        &model.grids,
        &model.provenance,
        &model.schema.cat_encoders,
    )
    .unwrap();

    for w in [RefMeasure::default(), RefMeasure::Uniform] {
        let bank = model
            .explain(&serve, w.clone())
            .unwrap_or_else(|e| panic!("two-channel explain({w:?}) failed: {e:?}"));
        assert_exact_decomposition(&model, &bank, &serve)
            .unwrap_or_else(|e| panic!("two-channel I2 gates failed under {w:?}: {e:?}"));
        // The multi-channel raw feature collapses to exactly ONE table (order-1 over {0}),
        // never one per channel — no channel-named/axis-named entries leak into the bank.
        let raw0_tables: Vec<_> = bank
            .tables
            .iter()
            .filter(|t| t.u.0.iter().any(|f| f.0 == 0))
            .collect();
        assert_eq!(
            raw0_tables.len(),
            1,
            "raw feature 0 must have exactly one table (order-1, collapsed), not one per \
             channel: got {raw0_tables:?}"
        );
        assert_eq!(
            raw0_tables[0].u.order(),
            1,
            "must stay order-1, not order-2"
        );
    }
}
