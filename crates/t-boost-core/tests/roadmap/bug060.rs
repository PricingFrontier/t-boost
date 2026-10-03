use t_boost_core::boosters::{BoosterConfig, EnsembleSpec};
use t_boost_core::constraints::{CredibilityFloor, InteractionPolicy};
use t_boost_core::data::{bin_columns, BinConfig, ServeBinnedMatrix};
use t_boost_core::engine::{Booster, Config, FitSpec};
use t_boost_core::explain::RefMeasure;
use t_boost_core::loss::SquaredError;
use t_boost_core::prune::{bag_oob_group_raw_sums, bag_score_variance_for_rows};
#[test]
fn bug060_regression() {
    let n = 20;
    let col = vec![1.; n];
    let x = bin_columns(&[&col], None, &BinConfig::default(), 0).unwrap();
    let loss = SquaredError;
    let spec = FitSpec {
        loss: &loss,
        weight: None,
        exposure: None,
        monotone: Default::default(),
        interaction: InteractionPolicy::default(),
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed: 0,
    };
    let booster = Booster::with_config(Config {
        n_trees: 1,
        boosters: BoosterConfig {
            ensemble: EnsembleSpec::OuterBag {
                n_bags: 2,
                bag_subsample: 0.5,
                cell_refit: None,
            },
            ..Default::default()
        },
        ..Default::default()
    });
    let mut y = vec![0.; n];
    let before = booster.fit(&x, &y, &spec).unwrap();
    let bags = before.bag_in_bag.as_ref().unwrap();
    let r = (0..n).find(|&r| bags[0][r] != bags[1][r]).unwrap();
    y[r] = 100.;
    let after = booster.fit(&x, &y, &spec).unwrap();
    for (name, m) in [("before", before), ("after", after)] {
        let evidence = bag_oob_group_raw_sums(
            &m,
            &ServeBinnedMatrix(x.clone()),
            RefMeasure::Uniform,
            None,
            &[],
        )
        .unwrap();
        let bags = m.bag_in_bag.as_ref().unwrap();
        let honest = (0..bags.len())
            .filter(|&b| !bags[b][r])
            .map(|b| {
                y.iter()
                    .enumerate()
                    .filter(|(i, _)| bags[b][*i])
                    .map(|(_, v)| *v as f64)
                    .sum::<f64>()
                    / bags[b].iter().filter(|v| **v).count() as f64
            })
            .sum::<f64>()
            / evidence.counts[r] as f64;
        let reported = evidence.full_sum[r] / f64::from(evidence.counts[r]);
        assert!(
            (reported - honest).abs() < 1e-6,
            "OOB intercept consulted held-out target: {reported} vs {honest}"
        );
        println!("{name}: row={r} trees={} bag_membership={:?} soup_f0={} oob_count={} reported_oob={} honest_oob={honest}", m.trees.len(), bags.iter().map(|b| b[r]).collect::<Vec<_>>(), m.f0, evidence.counts[r], evidence.full_sum[r] / evidence.counts[r] as f64);
        let variance = bag_score_variance_for_rows(
            &m,
            &ServeBinnedMatrix(x.clone()),
            RefMeasure::Uniform,
            None,
            None,
            &[r as u32],
        )
        .unwrap();
        assert_eq!(variance, vec![if name == "before" { 0.0 } else { 50.0 }]);
        println!("{name}: reconstructed bag score variance={variance:?}");
        println!(
            "{name}: bag sizes={:?}",
            bags.iter()
                .map(|b| b.iter().filter(|v| **v).count())
                .collect::<Vec<_>>()
        );
    }
}
