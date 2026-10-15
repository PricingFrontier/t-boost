//! A fresh categorical fit through the core API records where its rows sat in training
//! (`Model::fit_cat_bins`), so the post-fit stages that score the fit rows run on it. Only the
//! Python binding used to record them: a model from `Booster::fit_train` then failed every such
//! stage (the rebalance, the bagged out-of-bag evidence) with the error meant for loaded models,
//! even with a three-level categorical (codex review, 2026-10-15).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use t_boost_core::prune::fit_rows_matrix;
use t_boost_core::{
    bin_train_columns, BinConfig, Booster, CategoricalColumn, Config, CredibilityFloor, FeatureId,
    FitSpec, FittedBinnedData, InteractionPolicy, LeakageScheme, MonotoneMap, NumericColumn,
    Smooth, SquaredError, TsConfig, TsEncodingId,
};

const N: usize = 1200;

/// A numeric feature and a categorical with `n_levels` levels, binned with `max_bin` bins: more
/// levels than bins makes the levels share bins.
fn design(n_levels: usize, max_bin: u8, y: &[f32]) -> FittedBinnedData {
    let numeric: Vec<f32> = (0..N).map(|i| (i % 7) as f32).collect();
    let levels: Vec<String> = (0..N)
        .map(|i| format!("L{}", (i * 13) % n_levels))
        .collect();
    let ts = TsConfig {
        leakage: LeakageScheme::KFold { k: 3 },
        smooth: Smooth::Fixed { m: 1.0 },
        min_data_per_group: 0.0,
        ..TsConfig::default()
    };
    bin_train_columns(
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
        y,
        None,
        None,
        &BinConfig {
            max_bin,
            ..BinConfig::default()
        },
        7,
    )
    .unwrap()
}

fn target(n_levels: usize) -> Vec<f32> {
    (0..N)
        .map(|i| (i % 7) as f32 * 0.3 + ((i * 13) % n_levels) as f32 * 0.1 + (i % 5) as f32 * 0.2)
        .collect()
}

fn config() -> Config {
    Config {
        n_trees: 20,
        learning_rate: 0.3,
        validation_fraction: None,
        ..Config::default()
    }
}

fn spec(loss: &SquaredError) -> FitSpec<'_> {
    FitSpec {
        loss,
        weight: None,
        exposure: None,
        monotone: MonotoneMap::new(),
        interaction: InteractionPolicy::default(),
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed: 3,
    }
}

#[test]
fn a_fit_whose_levels_keep_their_own_bins_scores_its_rows_on_the_serve_map() {
    let y = target(3);
    let fitted = design(3, 64, &y);
    let model = Booster::with_config(config())
        .fit_train(
            &fitted.train,
            &y,
            &spec(&SquaredError),
            fitted.cat_encoders.clone(),
        )
        .unwrap();
    assert_eq!(model.fit_cat_bins, Some(Vec::new()));
    assert!(fit_rows_matrix(&model, &fitted.serve).unwrap().is_none());
}

#[test]
fn a_fit_whose_levels_share_bins_records_their_training_bins() {
    let y = target(40);
    let fitted = design(40, 8, &y);
    let model = Booster::with_config(config())
        .fit_train(
            &fitted.train,
            &y,
            &spec(&SquaredError),
            fitted.cat_encoders.clone(),
        )
        .unwrap();
    let bins = model.fit_cat_bins.as_ref().unwrap();
    assert_eq!(bins.len(), 1);
    let (axis, column) = &bins[0];
    assert_eq!(column, &fitted.train.0.data[*axis as usize]);
    let rows = fit_rows_matrix(&model, &fitted.serve).unwrap().unwrap();
    for (a, col) in rows.0.data.iter().enumerate() {
        let want = if a == *axis as usize {
            &fitted.train.0.data[a]
        } else {
            &fitted.serve.0.data[a]
        };
        assert_eq!(col, want, "axis {a}");
    }
}

#[test]
fn every_class_of_a_multiclass_fit_records_them_too() {
    let y: Vec<f32> = (0..N).map(|i| ((i * 13) % 40 / 14) as f32).collect();
    let fitted = design(40, 8, &y);
    let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let model = Booster::with_config(config())
        .fit_multiclass_train(
            &fitted.train,
            &y,
            3,
            &labels,
            &spec(&SquaredError),
            fitted.cat_encoders.clone(),
        )
        .unwrap();
    for class in &model.classes {
        let bins = class
            .fit_cat_bins
            .as_ref()
            .expect("a fresh fit records its bins");
        assert_eq!(bins.len(), 1);
        assert!(fit_rows_matrix(class, &fitted.serve).unwrap().is_some());
    }
}
