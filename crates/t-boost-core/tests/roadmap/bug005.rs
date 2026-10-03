// BUG-005 harness: missing-vs-present signal with one distinct finite value.
use t_boost_core::constraints::{CredibilityFloor, InteractionPolicy};
use t_boost_core::data::{bin_columns, BinConfig};
use t_boost_core::engine::{Booster, Config, FitSpec, HistPrecision};
use t_boost_core::loss::SquaredError;

fn run(label: &str, x_col: &[f32], y: &[f32]) {
    let x = bin_columns(&[x_col], None, &BinConfig::default(), 0).unwrap();
    let g = &x.grids[0];
    println!(
        "{label}: grid borders={:?} n_bins={} missing_bin={}; bins(row0,row99)=({},{})",
        g.borders, g.n_bins, g.missing_bin, x.data[0][0], x.data[0][99]
    );
    for hp in [HistPrecision::FullF64, HistPrecision::QuantizedI32] {
        let cfg = Config {
            n_trees: 100,
            learning_rate: 0.5,
            lambda: 0.0,
            validation_fraction: None,
            hist_precision: hp,
            ..Default::default()
        };
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
        match Booster::with_config(cfg).fit(&x, y, &spec) {
            Ok(m) => {
                let p = m.predict(&x, None).unwrap();
                assert!(
                    p[0].abs() < 0.01 && (p[99] - 10.).abs() < 0.01,
                    "{label} {hp:?}: {p:?}"
                );
                println!(
                    "  {hp:?}: trees={} pred(row0)={} pred(row99)={} f0={}",
                    m.trees.len(),
                    p[0],
                    p[99],
                    m.f0
                );
            }
            Err(e) => panic!("valid {label} fixture failed under {hp:?}: {e}"),
        }
    }
}

#[test]
fn bug005_regression() {
    let y: Vec<f32> = (0..100).map(|i| if i < 50 { 0.0 } else { 10.0 }).collect();
    let x_nan: Vec<f32> = (0..100)
        .map(|i| if i < 50 { f32::NAN } else { 1.0 })
        .collect();
    run("missing-vs-1", &x_nan, &y);
    let x_zero: Vec<f32> = (0..100).map(|i| if i < 50 { 0.0 } else { 1.0 }).collect();
    run("control 0-vs-1", &x_zero, &y);
}
