use std::sync::Mutex;
use t_boost_core::boosters::{BoosterConfig, NesterovSpec, RefitSpec};
use t_boost_core::constraints::{CredibilityFloor, InteractionPolicy};
use t_boost_core::data::{bin_columns, BinConfig};
use t_boost_core::engine::{Booster, Config, FitSpec};
use t_boost_core::loss::{GradHess, Link, Loss, Metric, ObjectiveTag, SquaredError};
use t_boost_core::PbError;
struct Recorded(Mutex<Vec<Vec<f32>>>);
impl Loss for Recorded {
    fn grad_hess(&self, y: &[f32], raw: &[f32], w: &[f32], out: &mut GradHess) -> Result<(), PbError> {
        self.0.lock().unwrap().push(raw.to_vec());
        SquaredError.grad_hess(y, raw, w, out)
    }
    fn init_score(&self, y: &[f32], w: &[f32], off: Option<&[f32]>) -> Result<f64, PbError> {
        SquaredError.init_score(y, w, off)
    }
    fn link(&self) -> Link { Link::Identity }
    fn pred_from_raw(&self, r: f32) -> f32 { r }
    fn deviance(&self, y: &[f32], r: &[f32], w: &[f32]) -> Result<f32, PbError> {
        SquaredError.deviance(y, r, w)
    }
    fn default_metric(&self) -> Metric { Metric::Rmse }
    fn objective_tag(&self) -> ObjectiveTag { SquaredError.objective_tag() }
}
fn main() {
    let n = 200;
    let feature = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let y = (0..n).map(|i| { let z = i as f32 / 20.; z.sin() * 5. + z }).collect::<Vec<_>>();
    let x = bin_columns(&[&feature], None, &BinConfig::default(), 0).unwrap();
    let loss = Recorded(Mutex::new(vec![]));
    let spec = FitSpec {
        loss: &loss, weight: None, exposure: None, monotone: Default::default(),
        interaction: InteractionPolicy::default(), credibility: CredibilityFloor::default(),
        fixed_holdout: None, bag_groups: None, seed: 0,
    };
    let cfg = Config {
        n_trees: 2, learning_rate: 0.3, lambda: 10., leaf_refine_steps: 0,
        boosters: BoosterConfig {
            refit_leaves: RefitSpec::Ridge { l2: 10., max_iter: 1, every_k_trees: Some(1) },
            nesterov: NesterovSpec::Agbm { momentum_correction: false },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut models = vec![];
    for nt in [1, 2, 3] {
        loss.0.lock().unwrap().clear();
        let m = Booster::with_config(Config { n_trees: nt, ..cfg.clone() }).fit(&x, &y, &spec).unwrap();
        println!("nt={nt} trees={} gh_calls={}", m.trees.len(), loss.0.lock().unwrap().len());
        models.push(m);
    }
    let actual = loss.0.lock().unwrap()[4].clone();
    let mut expected_model = models[1].clone();
    for (i, (a, _)) in expected_model.trees.iter_mut().enumerate() {
        let prev = models[0].trees.get(i).map_or(0., |v| v.0);
        *a = 1.5 * *a - 0.5 * prev;
    }
    let expected = expected_model.predict(&x, None).unwrap();
    let max_diff = actual.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0., f32::max);
    println!("AGBM + every-round ridge third-round gradient raw max difference from persisted alpha mixture: {max_diff}; first actual={} expected={}", actual[0], expected[0]);
    // extra diagnostics: alphas of the stored models
    for (i, m) in models.iter().enumerate() {
        println!("  model nt={} alphas={:?} f0={}", i + 1, m.trees.iter().map(|t| t.0).collect::<Vec<_>>(), m.f0);
    }
    let mut control_models = vec![];
    for nt in [1, 2, 3] {
        loss.0.lock().unwrap().clear();
        let mut c = cfg.clone();
        c.n_trees = nt;
        c.boosters.refit_leaves = RefitSpec::Off;
        control_models.push(Booster::with_config(c).fit(&x, &y, &spec).unwrap());
    }
    let actual = loss.0.lock().unwrap()[2].clone();
    let mut expected_model = control_models[1].clone();
    for (i, (a, _)) in expected_model.trees.iter_mut().enumerate() {
        let prev = control_models[0].trees.get(i).map_or(0., |v| v.0);
        *a = 1.5 * *a - 0.5 * prev;
    }
    let expected = expected_model.predict(&x, None).unwrap();
    println!("AGBM without ridge control max difference: {}", actual.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0., f32::max));
}
