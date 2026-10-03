// BUG-006 harness: midpoint rounding merges adjacent float32 values.
use t_boost_core::constraints::{CredibilityFloor, InteractionPolicy};
use t_boost_core::data::grid::build_grid;
use t_boost_core::data::{bin, bin_columns, BinConfig, FeatureId};
use t_boost_core::engine::{Booster, Config, FitSpec, HistPrecision};
use t_boost_core::loss::SquaredError;

#[test]
fn bug006_regression() {
    let a = 16777218f32;
    let b = 16777220f32;
    println!(
        "a={a} b={b} distinct={} midpoint f64={} as f32={}",
        a != b,
        (f64::from(a) + f64::from(b)) / 2.0,
        ((f64::from(a) + f64::from(b)) / 2.0) as f32
    );
    let g = build_grid(&[a, b], None, &BinConfig::default(), 0, FeatureId(0)).unwrap();
    assert_ne!(bin(a, &g).unwrap(), bin(b, &g).unwrap());
    assert!(g.borders[0] >= a && g.borders[0] < b);
    println!("borders = {:?} n_bins={}", g.borders, g.n_bins);
    println!("bin({a}) = {}", bin(a, &g).unwrap());
    println!("bin({b}) = {}", bin(b, &g).unwrap());
    // Parity control: the other rounding parity (midpoint rounds down).
    let c = 16777216f32;
    let d = 16777218f32;
    let g2 = build_grid(&[c, d], None, &BinConfig::default(), 0, FeatureId(0)).unwrap();
    println!(
        "control [{c},{d}]: borders={:?} bin(c)={} bin(d)={}",
        g2.borders,
        bin(c, &g2).unwrap(),
        bin(d, &g2).unwrap()
    );

    // End-to-end: repeat each 50 times, two-group target.
    let col: Vec<f32> = (0..100).map(|i| if i < 50 { a } else { b }).collect();
    let y: Vec<f32> = (0..100).map(|i| if i < 50 { 0.0 } else { 10.0 }).collect();
    let x = bin_columns(&[&col], None, &BinConfig::default(), 0).unwrap();
    println!(
        "bin_columns grid borders={:?} n_bins={} bins(row0,row99)=({},{})",
        x.grids[0].borders, x.grids[0].n_bins, x.data[0][0], x.data[0][99]
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
        let m = Booster::with_config(cfg).fit(&x, &y, &spec).unwrap();
        let p = m.predict(&x, None).unwrap();
        assert!(p[0].abs() < 0.01 && (p[99] - 10.).abs() < 0.01);
        println!(
            "  {hp:?}: trees={} pred(row0)={} pred(row99)={}",
            m.trees.len(),
            p[0],
            p[99]
        );
    }
}
