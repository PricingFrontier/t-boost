// BUG-014 harness: zero-row dataset with outer bags.
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
    let empty: Vec<f32> = vec![];
    let x = match bin_columns(&[&empty[..]], None, &BinConfig::default(), 0) {
        Ok(x) => x,
        Err(e) => {
            println!("bin_columns on zero rows: Err({e:?}) -- repro cannot proceed");
            return;
        }
    };
    println!("bin_columns zero rows: n_rows={} n_features={}", x.n_rows, x.data.len());
    let y: Vec<f32> = vec![];
    let spec = FitSpec {
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
    let classes: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
    for (label, ens) in [
        ("Off", EnsembleSpec::Off),
        (
            "OuterBag n_bags=1",
            EnsembleSpec::OuterBag {
                n_bags: 1,
                bag_subsample: 0.8,
                cell_refit: None,
            },
        ),
        (
            "OuterBag n_bags=2",
            EnsembleSpec::OuterBag {
                n_bags: 2,
                bag_subsample: 0.8,
                cell_refit: None,
            },
        ),
        (
            "OuterBag n_bags=2 bag_subsample=1.0",
            EnsembleSpec::OuterBag {
                n_bags: 2,
                bag_subsample: 1.0,
                cell_refit: None,
            },
        ),
    ] {
        let booster = Booster::with_config(Config {
            boosters: BoosterConfig {
                ensemble: ens,
                ..Default::default()
            },
            ..Default::default()
        });
        LAST_PANIC.lock().unwrap().clear();
        let r = catch_unwind(AssertUnwindSafe(|| booster.fit(&x, &y, &spec)));
        match r {
            Ok(Ok(m)) => println!("{label} scalar: Ok(trees={}, f0={})", m.trees.len(), m.f0),
            Ok(Err(e)) => println!("{label} scalar: Err({e:?})"),
            Err(_) => println!("{label} scalar: PANIC {}", LAST_PANIC.lock().unwrap()),
        }
        LAST_PANIC.lock().unwrap().clear();
        let r = catch_unwind(AssertUnwindSafe(|| booster.fit_multiclass(&x, &y, 3, &classes, &spec)));
        match r {
            Ok(Ok(m)) => println!("{label} multiclass: Ok(classes={})", m.classes.len()),
            Ok(Err(e)) => println!("{label} multiclass: Err({e:?})"),
            Err(_) => println!("{label} multiclass: PANIC {}", LAST_PANIC.lock().unwrap()),
        }
    }
}
