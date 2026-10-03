use t_boost_core::engine::{MultiClassModel, Split};
use t_boost_core::explain::{fixture_model, fixture_serve, FeatureSet, RefMeasure};
use t_boost_core::loss::LossId;
use t_boost_core::prune::prune_multiclass_to_keepset;
#[test]
fn bug061_regression() {
    let n = 100;
    let mut serve = fixture_serve();
    serve.0.data = vec![
        (0..n).map(|i| if i < 50 { 1 } else { 2 }).collect(),
        vec![1; n],
    ];
    serve.0.n_rows = n as u32;
    let classes = (0..3)
        .map(|k| {
            let mut m = fixture_model();
            m.schema.objective.loss = LossId::Softmax;
            let tree = &mut m.trees[0].1;
            tree.splits = vec![Split {
                axis: 0,
                bin_le: 1,
                missing_left: false,
            }];
            tree.depth = 1;
            tree.leaves = vec![0.; 8];
            if k == 0 {
                tree.leaves[1] = 20.;
            }
            if k == 1 {
                tree.leaves[0] = 20.;
            }
            m
        })
        .collect();
    let mc = MultiClassModel {
        classes,
        class_labels: vec!["a".into(), "b".into(), "c".into()],
        schema_version: 2,
        cell_refit: None,
    };
    mc.validate().unwrap();
    let labels = (0..n)
        .map(|i| {
            if i < 10 {
                0
            } else if i < 90 {
                1
            } else {
                2
            }
        })
        .collect::<Vec<_>>();
    let w = vec![1.; n];
    let tm = prune_multiclass_to_keepset(
        &mc,
        &serve,
        &labels,
        &w,
        RefMeasure::Uniform,
        &[FeatureSet::new(&[0])],
    )
    .unwrap();
    let pred = tm.predict_proba(&serve.0).unwrap();
    let masses = (0..3)
        .map(|k| (0..n).map(|r| pred[r * 3 + k] as f64).sum::<f64>())
        .collect::<Vec<_>>();
    for (predicted, actual) in masses.iter().zip([10., 80., 10.]) {
        assert!(
            (predicted - actual).abs() < 1e-3,
            "intercept did not converge: {masses:?}"
        );
    }
    println!("observed=[10,80,10] library IPF10 predicted={masses:?}");
    let raw = mc.predict_raw(&serve.0).unwrap();
    println!(
        "raw first row={:?} raw row 50={:?}",
        &raw[0..3],
        &raw[150..153]
    );
    let mut d = [0.; 3];
    for iter in 0..1000 {
        let mut masses = vec![0.; 3];
        for r in 0..n {
            let logits = (0..3)
                .map(|k| raw[r * 3 + k] as f64 + d[k])
                .collect::<Vec<_>>();
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps = logits.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
            let z = exps.iter().sum::<f64>();
            for k in 0..3 {
                masses[k] += exps[k] / z;
            }
        }
        if [10, 100, 999].contains(&iter) {
            println!("reference iter={iter} predicted={masses:?}");
        }
        for k in 0..3 {
            d[k] += ([10., 80., 10.][k] / masses[k]).ln();
        }
    }
}
