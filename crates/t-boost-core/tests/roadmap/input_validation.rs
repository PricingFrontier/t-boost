use t_boost_core::boosters::{BoosterConfig, EnsembleSpec};
use t_boost_core::data::{bin_columns, BinConfig};
use t_boost_core::engine::{Booster, Config, FitSpec};
use t_boost_core::loss::SquaredError;

fn spec() -> FitSpec<'static> {
    FitSpec {
        loss: &SquaredError,
        weight: None,
        exposure: None,
        monotone: Default::default(),
        interaction: Default::default(),
        credibility: Default::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed: 0,
    }
}

fn booster(bags: u16) -> Booster {
    Booster::with_config(Config {
        n_trees: 1,
        boosters: BoosterConfig {
            ensemble: EnsembleSpec::OuterBag {
                n_bags: bags,
                bag_subsample: 0.8,
                cell_refit: None,
            },
            ..Default::default()
        },
        ..Default::default()
    })
}

#[test]
fn bug014_empty_bagged_fit_returns_typed_error() {
    let x = bin_columns(&[&[]], None, &BinConfig::default(), 0).unwrap();
    for bags in [1, 2] {
        assert!(booster(bags).fit(&x, &[], &spec()).is_err());
        assert!(booster(bags)
            .fit_multiclass(&x, &[], 3, &["a".into(), "b".into(), "c".into()], &spec())
            .is_err());
    }
}

#[test]
fn bug037_bagged_fit_validates_fixed_holdout_length_before_indexing() {
    let col: Vec<f32> = (0..200).map(|i| i as f32).collect();
    let x = bin_columns(&[&col], None, &BinConfig::default(), 0).unwrap();
    let labels: Vec<f32> = (0..200).map(|i| (i % 3) as f32).collect();
    for length in [1, 201] {
        let mask = vec![false; length];
        let spec = FitSpec {
            fixed_holdout: Some(&mask),
            ..spec()
        };
        assert!(booster(2).fit(&x, &col, &spec).is_err());
        assert!(booster(2)
            .fit_multiclass(&x, &labels, 3, &["a".into(), "b".into(), "c".into()], &spec)
            .is_err());
    }
}

#[test]
fn bug038_multiclass_rejects_invalid_sample_weights() {
    let col: Vec<f32> = (0..200).map(|i| i as f32).collect();
    let x = bin_columns(&[&col], None, &BinConfig::default(), 0).unwrap();
    let labels: Vec<f32> = (0..200).map(|i| (i % 3) as f32).collect();
    for bags in [1, 2] {
        for invalid in [-10., f32::NAN, f32::INFINITY] {
            let mut weights = vec![1.; 200];
            weights[0] = invalid;
            let spec = FitSpec {
                weight: Some(&weights),
                ..spec()
            };
            assert!(
                booster(bags)
                    .fit_multiclass(&x, &labels, 3, &["a".into(), "b".into(), "c".into()], &spec)
                    .is_err(),
                "accepted invalid multiclass weight {invalid}"
            );
        }
    }
}

#[test]
fn bug049_zero_mass_validation_preserves_explicit_holdout_honesty() {
    let col: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let x = bin_columns(&[&col], None, &BinConfig::default(), 0).unwrap();
    let mut weights = vec![0.; 100];
    weights[17] = 1.;
    let labels: Vec<f32> = (0..100).map(|i| (i % 3) as f32).collect();
    let classes = ["a".into(), "b".into(), "c".into()];
    for bags in [1, 2] {
        for positive_held_out in [false, true] {
            let mut mask = vec![false; 100];
            mask[17] = positive_held_out;
            mask[18] = true;
            let spec = FitSpec {
                weight: Some(&weights),
                fixed_holdout: Some(&mask),
                ..spec()
            };
            let scalar = booster(bags).fit(&x, &col, &spec);
            let multi = booster(bags).fit_multiclass(&x, &labels, 3, &classes, &spec);
            assert_eq!(scalar.is_err(), positive_held_out);
            assert_eq!(multi.is_err(), positive_held_out);
        }
    }
}
