use std::sync::Mutex;
use t_boost_core::boosters::{BoosterConfig, EnsembleSpec};
use t_boost_core::constraints::{CredibilityFloor, InteractionPolicy};
use t_boost_core::data::{bin_columns, BinConfig};
use t_boost_core::engine::{Booster, Config, FitSpec};
use t_boost_core::loss::{GradHess, Link, Loss, Metric, ObjectiveTag, SquaredError};
use t_boost_core::PbError;
struct Recorded(Mutex<Vec<(bool, Vec<f32>)>>);
impl Loss for Recorded {
    fn grad_hess(&self, y: &[f32], raw: &[f32], w: &[f32], out: &mut GradHess) -> Result<(), PbError> {
        SquaredError.grad_hess(y, raw, w, out)
    }
    fn init_score(&self, y: &[f32], w: &[f32], off: Option<&[f32]>) -> Result<f64, PbError> {
        self.0.lock().unwrap().push((true, y.to_vec()));
        SquaredError.init_score(y, w, off)
    }
    fn link(&self) -> Link { Link::Identity }
    fn pred_from_raw(&self, r: f32) -> f32 { r }
    fn deviance(&self, y: &[f32], r: &[f32], w: &[f32]) -> Result<f32, PbError> {
        self.0.lock().unwrap().push((false, y.to_vec()));
        SquaredError.deviance(y, r, w)
    }
    fn default_metric(&self) -> Metric { Metric::Rmse }
    fn objective_tag(&self) -> ObjectiveTag { SquaredError.objective_tag() }
}
fn main() {
    let feature = (0..100).map(|i| i as f32).collect::<Vec<_>>();
    let x = bin_columns(&[&feature], None, &BinConfig::default(), 0).unwrap();
    let loss = Recorded(Mutex::new(vec![]));
    let spec = FitSpec {
        loss: &loss, weight: None, exposure: None, monotone: Default::default(),
        interaction: InteractionPolicy::default(), credibility: CredibilityFloor::default(),
        fixed_holdout: None, bag_groups: None, seed: 0,
    };
    for fraction in [1., 0.8] {
        loss.0.lock().unwrap().clear();
        let cfg = Config {
            n_trees: 1, validation_fraction: Some(0.2), leaf_refine_steps: 0,
            boosters: BoosterConfig {
                ensemble: EnsembleSpec::OuterBag { n_bags: 2, bag_subsample: fraction, cell_refit: None },
                ..Default::default()
            },
            ..Default::default()
        };
        Booster::with_config(cfg).fit(&x, &feature, &spec).unwrap();
        let mut train = vec![];
        let mut awaiting = false;
        for (init, y) in loss.0.lock().unwrap().iter() {
            if *init {
                train = y.clone();
                awaiting = true;
            } else if awaiting {
                let overlapping = y.iter().filter(|r| train.contains(r)).copied().collect::<Vec<_>>();
                println!("bag_subsample={fraction} train_n={} holdout_n={} heldout_rows_also_in_train={} overlap_ids={overlapping:?}", train.len(), y.len(), overlapping.len());
                awaiting = false;
            }
        }
    }
}
