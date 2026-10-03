use t_boost_core::boosters::{BoosterConfig, CellRefit, EnsembleSpec};
use t_boost_core::constraints::{CredibilityFloor, InteractionPolicy, MonoSign};
use t_boost_core::data::{bin_columns, BinConfig};
use t_boost_core::engine::{Booster, Config, FitSpec};
use t_boost_core::loss::SquaredError;
#[test]
fn bug048_regression() {
    let n = 4000;
    let xc = (0..n).map(|i| (i / 200) as f32).collect::<Vec<_>>();
    let zc = (0..n).map(|i| ((i / 10) % 20) as f32).collect::<Vec<_>>();
    let y = xc
        .iter()
        .zip(&zc)
        .map(|(&x, &z)| x * if z < 10. { 2. } else { -1. } + z * 0.4)
        .collect::<Vec<_>>();
    let x = bin_columns(&[&xc, &zc], None, &BinConfig::default(), 0).unwrap();
    let loss = SquaredError;
    let spec = FitSpec {
        loss: &loss,
        weight: None,
        exposure: None,
        monotone: [("f0".into(), MonoSign::Increasing)].into(),
        interaction: InteractionPolicy::default(),
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed: 0,
    };
    for base in [None, Some(0.01), Some(1.), Some(100.)] {
        let cfg = Config {
            n_trees: 10,
            learning_rate: 0.2,
            lambda: 1.,
            leaf_refine_steps: 0,
            boosters: BoosterConfig {
                ensemble: EnsembleSpec::OuterBag {
                    n_bags: 4,
                    bag_subsample: 0.8,
                    cell_refit: base.map(|base| CellRefit { base, gamma: 0. }),
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let fitted = Booster::with_config(cfg).fit(&x, &y, &spec);
        if base.is_some() {
            assert!(matches!(
                fitted,
                Err(t_boost_core::error::PbError::InvalidConfig { .. })
            ));
            continue;
        }
        let m = fitted.unwrap();
        let pred = m.predict(&x, None).unwrap();
        let min_delta = (0..n - 200)
            .map(|i| pred[i + 200] - pred[i])
            .fold(0., f32::min);
        assert!(
            min_delta >= -1e-5,
            "OOB correction broke monotonicity: {min_delta}"
        );
        println!("cellrefit={base:?} correction={} trees={} minimum_delta={min_delta} x0z15={} x19z15={}", m.correction.is_some(), m.trees.len(), pred[150], pred[3950]);
    }
}
