// BUG-005 harness: missing-vs-present signal with one distinct finite value.
use t_boost_core::constraints::{CredibilityFloor, InteractionPolicy};
use t_boost_core::data::{bin_columns, BinConfig};
use t_boost_core::engine::{Booster, Config, FitSpec, HistPrecision};
use t_boost_core::loss::SquaredError;
use t_boost_core::{RefMeasure, ServeBinnedMatrix};

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

/// The pure missing-vs-present split (`bin_le = 0`) must also decompose: `explain` used to
/// reject it as having "no interior border", so every fit that learned one failed in the
/// table pipeline. The bank must reproduce the tree ensemble on every row.
#[test]
fn bug005_missing_split_decomposes_into_tables() {
    let n = 200;
    let x0: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { f32::NAN } else { 1.0 })
        .collect();
    let x1: Vec<f32> = (0..n).map(|i| (i % 10) as f32).collect();
    let y: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { 0.0 } else { 10.0 } + (i % 10) as f32)
        .collect();
    let x = bin_columns(&[&x0, &x1], None, &BinConfig::default(), 0).unwrap();
    let cfg = Config {
        n_trees: 50,
        learning_rate: 0.5,
        lambda: 0.0,
        validation_fraction: None,
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
    let model = Booster::with_config(cfg).fit(&x, &y, &spec).unwrap();
    assert!(
        model
            .trees
            .iter()
            .any(|(_, t)| t.splits.iter().any(|s| s.axis == 0 && s.bin_le == 0)),
        "fixture must learn a bin_le=0 split or this test proves nothing"
    );
    let serve = ServeBinnedMatrix(x.clone());
    let bank = model.explain(&serve, RefMeasure::default()).unwrap();
    let mut out = vec![0.0_f64; n];
    bank.score_binned(&model.schema.cat_encoders, &x, &mut out)
        .unwrap();
    for (r, got) in out.iter().enumerate() {
        let row: Vec<u8> = x.data.iter().map(|col| col[r]).collect();
        let want = model.ensemble_f64(&row).unwrap();
        assert!(
            (got - want).abs() < 1e-9,
            "row {r}: bank {got} != trees {want}"
        );
    }
}
