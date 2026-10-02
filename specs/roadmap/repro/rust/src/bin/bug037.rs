// BUG-037 + BUG-038 harness: the doc's combined harness (bugs.md lines 2154-2355) with the
// AGBM/BUG-035 material stripped; SquaredError is used directly instead of `Recorded`.
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Mutex;
use t_boost_core::boosters::{BoosterConfig, EnsembleSpec};
use t_boost_core::constraints::{CredibilityFloor, InteractionPolicy};
use t_boost_core::data::{bin_columns, BinConfig};
use t_boost_core::engine::{Booster, Config, FitSpec};
use t_boost_core::loss::SquaredError;

static LAST_PANIC: Mutex<String> = Mutex::new(String::new());

fn main() {
    std::panic::set_hook(Box::new(|info| {
        let msg = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_default();
        let loc = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_default();
        *LAST_PANIC.lock().unwrap() = format!("{loc}: {msg}");
    }));
    let n = 200;
    let feature = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let y = (0..n)
        .map(|i| {
            let z = i as f32 / 20.;
            z.sin() * 5. + z
        })
        .collect::<Vec<_>>();
    let x = bin_columns(&[&feature], None, &BinConfig::default(), 0).unwrap();
    let mut spec = FitSpec {
        loss: &SquaredError,
        weight: None,
        exposure: None,
        monotone: Default::default(),
        interaction: InteractionPolicy::default(),
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed: 0,
    };
    let ym = (0..n).map(|i| (i % 3) as f32).collect::<Vec<_>>();
    let classes: Vec<String> = vec!["a".into(), "b".into(), "c".into()];

    println!("== BUG-038: multiclass invalid weights");
    for bagged in [false, true] {
        let booster = Booster::with_config(Config {
            n_trees: 1,
            boosters: BoosterConfig {
                ensemble: if bagged {
                    EnsembleSpec::OuterBag {
                        n_bags: 2,
                        bag_subsample: 0.8,
                        cell_refit: None,
                    }
                } else {
                    EnsembleSpec::Off
                },
                ..Default::default()
            },
            ..Default::default()
        });
        for kind in ["negative", "nan", "inf"] {
            let mut w = vec![1.; n];
            w[0] = match kind {
                "negative" => -10.,
                "nan" => f32::NAN,
                _ => f32::INFINITY,
            };
            let ws = FitSpec {
                weight: Some(&w),
                monotone: spec.monotone.clone(),
                interaction: spec.interaction.clone(),
                ..spec
            };
            let result = booster.fit_multiclass(&x, &ym, 3, &classes, &ws);
            println!(
                "multiclass bagged={bagged} weight={kind}: {:?}",
                result
                    .as_ref()
                    .map(|m| m.classes.iter().map(|x| x.f0).collect::<Vec<_>>())
                    .map_err(|e| format!("{e:?}"))
            );
            // Scalar comparison: same weights through Booster::fit with SquaredError.
            let result = booster.fit(&x, &y, &ws);
            println!(
                "   scalar   bagged={bagged} weight={kind}: {:?}",
                result.as_ref().map(|m| m.f0).map_err(|e| format!("{e:?}"))
            );
        }
    }

    println!("== BUG-037: short fixed_holdout mask with 2 bags");
    let mask = [false];
    spec.fixed_holdout = Some(&mask);
    for n_bags in [2u16, 1] {
        let booster = Booster::with_config(Config {
            n_trees: 1,
            boosters: BoosterConfig {
                ensemble: EnsembleSpec::OuterBag {
                    n_bags,
                    bag_subsample: 0.8,
                    cell_refit: None,
                },
                ..Default::default()
            },
            ..Default::default()
        });
        LAST_PANIC.lock().unwrap().clear();
        let result = catch_unwind(AssertUnwindSafe(|| booster.fit(&x, &y, &spec)));
        match result {
            Ok(r) => println!(
                "n_bags={n_bags} short fixed_holdout scalar panic=false result={:?}",
                r.map(|m| m.trees.len()).map_err(|e| format!("{e:?}"))
            ),
            Err(_) => println!(
                "n_bags={n_bags} short fixed_holdout scalar panic=true  {}",
                LAST_PANIC.lock().unwrap()
            ),
        }
        LAST_PANIC.lock().unwrap().clear();
        let result = catch_unwind(AssertUnwindSafe(|| {
            booster.fit_multiclass(&x, &ym, 3, &classes, &spec)
        }));
        match result {
            Ok(r) => println!(
                "n_bags={n_bags} short fixed_holdout multiclass panic=false result={:?}",
                r.map(|m| m.classes.len()).map_err(|e| format!("{e:?}"))
            ),
            Err(_) => println!(
                "n_bags={n_bags} short fixed_holdout multiclass panic=true  {}",
                LAST_PANIC.lock().unwrap()
            ),
        }
    }
    // Long mask control (201 entries) with 2 bags.
    let long_mask = vec![false; n + 1];
    spec.fixed_holdout = Some(&long_mask);
    let booster = Booster::with_config(Config {
        n_trees: 1,
        boosters: BoosterConfig {
            ensemble: EnsembleSpec::OuterBag {
                n_bags: 2,
                bag_subsample: 0.8,
                cell_refit: None,
            },
            ..Default::default()
        },
        ..Default::default()
    });
    LAST_PANIC.lock().unwrap().clear();
    let result = catch_unwind(AssertUnwindSafe(|| booster.fit(&x, &y, &spec)));
    match result {
        Ok(r) => println!(
            "n_bags=2 LONG fixed_holdout scalar panic=false result={:?}",
            r.map(|m| m.trees.len()).map_err(|e| format!("{e:?}"))
        ),
        Err(_) => println!(
            "n_bags=2 LONG fixed_holdout scalar panic=true  {}",
            LAST_PANIC.lock().unwrap()
        ),
    }
}
