use std::collections::BTreeSet;
use std::sync::Mutex;
use t_boost_core::boosters::{BoosterConfig, CellRefit, EnsembleSpec};
use t_boost_core::constraints::{CredibilityFloor, InteractionPolicy};
use t_boost_core::data::{bin_columns, BinConfig};
use t_boost_core::engine::{Booster, Config, FitSpec};
use t_boost_core::loss::{GradHess, Link, Loss, Metric, ObjectiveTag, SquaredError};
use t_boost_core::PbError;
struct Recorded(Mutex<Vec<Vec<f32>>>);
impl Loss for Recorded {
    fn grad_hess(
        &self,
        y: &[f32],
        r: &[f32],
        w: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError> {
        SquaredError.grad_hess(y, r, w, out)
    }
    fn init_score(&self, y: &[f32], w: &[f32], o: Option<&[f32]>) -> Result<f64, PbError> {
        SquaredError.init_score(y, w, o)
    }
    fn link(&self) -> Link {
        Link::Identity
    }
    fn pred_from_raw(&self, r: f32) -> f32 {
        r
    }
    fn deviance(&self, y: &[f32], r: &[f32], w: &[f32]) -> Result<f32, PbError> {
        self.0.lock().unwrap().push(y.to_vec());
        SquaredError.deviance(y, r, w)
    }
    fn default_metric(&self) -> Metric {
        Metric::Rmse
    }
    fn objective_tag(&self) -> ObjectiveTag {
        SquaredError.objective_tag()
    }
}
#[test]
fn bug050_regression() {
    let n = 1000;
    let groups = (0..n).map(|i| (i / 10) as u32).collect::<Vec<_>>();
    let col = groups.iter().map(|&g| g as f32).collect::<Vec<_>>();
    let y = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let x = bin_columns(&[&col], None, &BinConfig::default(), 0).unwrap();
    let loss = Recorded(Mutex::new(vec![]));
    let spec = FitSpec {
        loss: &loss,
        weight: None,
        exposure: None,
        monotone: Default::default(),
        interaction: InteractionPolicy::default(),
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: Some(&groups),
        seed: 0,
    };
    let m = Booster::with_config(Config {
        n_trees: 5,
        learning_rate: 0.2,
        leaf_refine_steps: 0,
        boosters: BoosterConfig {
            ensemble: EnsembleSpec::OuterBag {
                n_bags: 4,
                bag_subsample: 0.8,
                cell_refit: Some(CellRefit {
                    base: 1.,
                    gamma: 0.,
                }),
            },
            ..Default::default()
        },
        ..Default::default()
    })
    .fit(&x, &y, &spec)
    .unwrap();
    let events = loss.0.lock().unwrap();
    let guard_rows = events
        .last()
        .unwrap()
        .iter()
        .map(|v| *v as usize)
        .collect::<BTreeSet<_>>();
    let bags = m.bag_in_bag.as_ref().unwrap();
    let covered = (0..n)
        .filter(|&r| bags.iter().any(|b| !b[r]))
        .collect::<BTreeSet<_>>();
    let fit_rows = covered
        .difference(&guard_rows)
        .copied()
        .collect::<BTreeSet<_>>();
    let guard_groups = guard_rows
        .iter()
        .map(|&r| groups[r])
        .collect::<BTreeSet<_>>();
    let fit_groups = fit_rows.iter().map(|&r| groups[r]).collect::<BTreeSet<_>>();
    let bags_split_group = bags
        .iter()
        .any(|b| (0..100).any(|g| (0..10).any(|i| b[g * 10 + i] != b[g * 10])));
    assert!(
        guard_groups.is_disjoint(&fit_groups),
        "cell-refit guard split declared groups"
    );
    assert!(!bags_split_group);
    println!("deviance_calls={} correction_kept={} covered_rows={} correction_fit_rows={} guard_rows={} guard_groups={} guard_groups_also_fit={} any_bag_splits_group={}", events.len(), m.correction.is_some(), covered.len(), fit_rows.len(), guard_rows.len(), guard_groups.len(), guard_groups.intersection(&fit_groups).count(), bags_split_group);
    let hash_guard = covered
        .iter()
        .filter(|&&r| ((r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) < 38)
        .copied()
        .collect::<BTreeSet<_>>();
    println!(
        "observed_guard_matches_production_hash={}",
        guard_rows == hash_guard
    );
    println!(
        "event sizes={:?}",
        events.iter().map(|e| e.len()).collect::<Vec<_>>()
    );
}
