//! `Config::fit_control` and the per-bag fit report: the per-round history (the early-stopping
//! deviance always, the training deviance with an observer) and the observer. All of it is
//! run-time control: an observer that never stops changes nothing.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use t_boost_core::{
    bin_columns, encode_model, BinConfig, Booster, Config, CredibilityFloor, FitSpec,
    InteractionPolicy, Loss, Model, MonotoneMap, RoundObserver, SquaredError, StopReason,
};

fn fixture(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x0: Vec<f32> = (0..n).map(|i| (i % 17) as f32).collect();
    let x1: Vec<f32> = (0..n).map(|i| ((i * 7) % 13) as f32).collect();
    let y: Vec<f32> = x0
        .iter()
        .zip(&x1)
        .enumerate()
        .map(|(i, (a, b))| a * 0.3 - b * 0.2 + ((i * 31) % 11) as f32 * 0.1)
        .collect();
    (x0, x1, y)
}

fn fit(config: Config) -> Model {
    let (x0, x1, y) = fixture(2000);
    let x = bin_columns(&[&x0, &x1], None, &BinConfig::default(), 3).unwrap();
    let loss = SquaredError;
    let spec = FitSpec {
        loss: &loss as &dyn Loss,
        weight: None,
        exposure: None,
        monotone: MonotoneMap::new(),
        interaction: InteractionPolicy::default(),
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed: 9,
    };
    Booster::with_config(config).fit(&x, &y, &spec).unwrap()
}

fn base() -> Config {
    Config {
        n_trees: 60,
        learning_rate: 0.2,
        validation_fraction: Some(0.2),
        early_stopping_rounds: 10,
        ..Config::default()
    }
}

#[test]
fn report_counts_trees_and_reason() {
    let model = fit(base());
    let report = model.fit_report.as_ref().unwrap();
    assert_eq!(report.len(), 1);
    assert_eq!(report[0].trees_kept as usize, model.trees.len());
    assert!(report[0].rounds_trained >= report[0].trees_kept);
    assert!(
        report[0].train_deviance.is_empty(),
        "the training curve needs an observer"
    );
    // The early-stopping curve is always kept (free), and the kept round is its minimum.
    let curve = &report[0].eval_deviance;
    assert_eq!(curve.len(), report[0].rounds_trained as usize);
    assert_eq!(report[0].trees_kept as usize, argmin(curve) + 1);

    let capped = fit(Config {
        validation_fraction: None,
        n_trees: 25,
        ..base()
    });
    let r = &capped.fit_report.as_ref().unwrap()[0];
    assert_eq!((r.trees_kept, r.reason), (25, StopReason::MaxTrees));
}

#[test]
fn a_silent_observer_leaves_the_model_bit_identical() {
    let plain = fit(base());
    let calls = Arc::new(AtomicU32::new(0));
    let counter = Arc::clone(&calls);
    let mut watched = base();
    watched.fit_control.observer = Some(RoundObserver(Arc::new(move |event| {
        counter.fetch_add(1, Ordering::Relaxed);
        assert!(event.train_deviance.is_finite());
        assert!(event.eval_deviance.is_some());
        false
    })));
    let traced = fit(watched);
    assert_eq!(
        encode_model(&plain).unwrap(),
        encode_model(&traced).unwrap()
    );
    let r = &traced.fit_report.as_ref().unwrap()[0];
    assert_eq!(calls.load(Ordering::Relaxed), r.rounds_trained);
    assert_eq!(r.train_deviance.len(), r.rounds_trained as usize);
    assert_eq!(
        r.eval_deviance,
        plain.fit_report.as_ref().unwrap()[0].eval_deviance
    );
}

#[test]
fn a_multiclass_fit_keeps_its_early_stopping_curve() {
    let (x0, x1, y) = fixture(2000);
    let x = bin_columns(&[&x0, &x1], None, &BinConfig::default(), 3).unwrap();
    let classes: Vec<f32> = y
        .iter()
        .map(|&v| {
            if v < 0.5 {
                0.0
            } else if v < 2.5 {
                1.0
            } else {
                2.0
            }
        })
        .collect();
    let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let loss = SquaredError; // ignored: the softmax gradient is coupled across classes
    let spec = FitSpec {
        loss: &loss as &dyn Loss,
        weight: None,
        exposure: None,
        monotone: MonotoneMap::new(),
        interaction: InteractionPolicy::default(),
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed: 9,
    };
    let model = Booster::with_config(base())
        .fit_multiclass(&x, &classes, 3, &labels, &spec)
        .unwrap();
    for class in &model.classes {
        let r = &class.fit_report.as_ref().unwrap()[0];
        assert_eq!(r.eval_deviance.len(), r.rounds_trained as usize);
        assert!(r.eval_deviance.iter().all(|d| d.is_finite() && *d > 0.0));
        assert_eq!(r.trees_kept as usize, argmin(&r.eval_deviance) + 1);
        assert!(r.train_deviance.is_empty());
    }
}

fn argmin(curve: &[f64]) -> usize {
    curve
        .iter()
        .enumerate()
        .fold(
            (0, f64::INFINITY),
            |(bi, bv), (i, &v)| {
                if v < bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            },
        )
        .0
}

#[test]
fn an_observer_can_stop_the_fit() {
    let mut config = base();
    config.validation_fraction = None;
    config.fit_control.observer = Some(RoundObserver(Arc::new(|event| event.round >= 4)));
    let model = fit(config);
    let r = &model.fit_report.as_ref().unwrap()[0];
    assert_eq!((r.trees_kept, r.reason), (4, StopReason::Callback));
}
