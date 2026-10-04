//! `Config::fit_control`: the per-bag fit report, the per-round history and observer, and the
//! external evaluation holdout. All of it is run-time control: with the defaults a fit is
//! bit-identical, and history or an observer that never stops change nothing either.

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

fn fit(config: Config, holdout: Option<&[bool]>) -> Model {
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
        fixed_holdout: holdout,
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
    let model = fit(base(), None);
    let report = model.fit_report.as_ref().unwrap();
    assert_eq!(report.len(), 1);
    assert_eq!(report[0].trees_kept as usize, model.trees.len());
    assert!(report[0].rounds_trained >= report[0].trees_kept);
    assert!(report[0].train_deviance.is_empty(), "history is opt-in");

    let capped = fit(
        Config {
            validation_fraction: None,
            n_trees: 25,
            ..base()
        },
        None,
    );
    let r = &capped.fit_report.as_ref().unwrap()[0];
    assert_eq!((r.trees_kept, r.reason), (25, StopReason::MaxTrees));
}

#[test]
fn history_and_a_silent_observer_leave_the_model_bit_identical() {
    let plain = fit(base(), None);
    let calls = Arc::new(AtomicU32::new(0));
    let counter = Arc::clone(&calls);
    let mut watched = base();
    watched.fit_control.record_history = true;
    watched.fit_control.observer = Some(RoundObserver(Arc::new(move |event| {
        counter.fetch_add(1, Ordering::Relaxed);
        assert!(event.train_deviance.is_finite());
        assert!(event.eval_deviance.is_some());
        false
    })));
    let traced = fit(watched, None);
    assert_eq!(
        encode_model(&plain).unwrap(),
        encode_model(&traced).unwrap()
    );
    let r = &traced.fit_report.as_ref().unwrap()[0];
    assert_eq!(calls.load(Ordering::Relaxed), r.rounds_trained);
    assert_eq!(r.train_deviance.len(), r.rounds_trained as usize);
    assert_eq!(r.eval_deviance.len(), r.rounds_trained as usize);

    // History alone records the (free) stopping curve, not the training one.
    let mut history_only = base();
    history_only.fit_control.record_history = true;
    let lean = fit(history_only, None);
    let r = &lean.fit_report.as_ref().unwrap()[0];
    assert!(r.train_deviance.is_empty());
    assert_eq!(r.eval_deviance.len(), r.rounds_trained as usize);
}

#[test]
fn an_observer_can_stop_the_fit() {
    let mut config = base();
    config.validation_fraction = None;
    config.fit_control.observer = Some(RoundObserver(Arc::new(|event| event.round >= 4)));
    let model = fit(config, None);
    let r = &model.fit_report.as_ref().unwrap()[0];
    assert_eq!((r.trees_kept, r.reason), (4, StopReason::Callback));
}

#[test]
fn an_external_holdout_never_moves_the_intercept() {
    // Same holdout rows, marked internal vs external: only the external fit keeps their
    // targets out of the intercept re-anchor, so the two may differ; with re-anchoring off
    // they must agree exactly, since the rows only ever scored early stopping.
    let mask: Vec<bool> = (0..2000).map(|i| i % 5 == 0).collect();
    let mut internal = base();
    internal.boosters.reanchor = false;
    let mut external = internal.clone();
    external.fit_control.external_holdout = true;
    let a = fit(internal, Some(&mask));
    let b = fit(external, Some(&mask));
    assert_eq!(encode_model(&a).unwrap(), encode_model(&b).unwrap());
}
