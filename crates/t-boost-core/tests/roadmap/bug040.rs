use std::sync::Mutex;
use t_boost_core::{
    boosters::{BoosterConfig, DartSpec},
    constraints::{CredibilityFloor, InteractionPolicy},
    data::{bin_columns, BinConfig},
    engine::{Booster, Config, FitSpec},
    loss::{GradHess, Link, Loss, Metric, ObjectiveTag, SquaredError},
    PbError,
};

struct Recorded(Mutex<Vec<Vec<f32>>>);
impl Loss for Recorded {
    fn grad_hess(
        &self,
        y: &[f32],
        raw: &[f32],
        w: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError> {
        self.0.lock().unwrap().push(raw.to_vec());
        SquaredError.grad_hess(y, raw, w, out)
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
fn dart_gradients_see_retained_tree_scores_at_large_intercepts() {
    let a = (0..80).map(|i| ((i % 4) / 2) as f32).collect::<Vec<_>>();
    let b = (0..80).map(|i| (i % 2) as f32).collect::<Vec<_>>();
    let y = (0..80)
        .map(|i| 1e7 + [6., 2., 2., 0.][i % 4])
        .collect::<Vec<_>>();
    let x = bin_columns(&[&a, &b], None, &BinConfig::default(), 0).unwrap();
    // A positive rate selects the DART path; this seeded draw drops no trees,
    // isolating score accumulation from dropout's intentional regularization.
    for dart in [
        None,
        Some(DartSpec {
            drop_rate: 1e-30,
            normalize: true,
        }),
    ] {
        let loss = Recorded(Mutex::new(vec![]));
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
        let model = Booster::with_config(Config {
            n_trees: 100,
            learning_rate: 0.2,
            leaf_refine_steps: 0,
            validation_fraction: None,
            boosters: BoosterConfig {
                dart,
                ..Default::default()
            },
            ..Default::default()
        })
        .fit(&x, &y, &spec)
        .unwrap();
        for (round, actual) in loss.0.lock().unwrap().iter().enumerate() {
            let mut prefix = model.clone();
            prefix.trees.truncate(round);
            assert_eq!(
                *actual,
                prefix.predict(&x, None).unwrap(),
                "gradient round {round}"
            );
        }
        assert_eq!(model.predict(&x, None).unwrap(), y);
    }
}
