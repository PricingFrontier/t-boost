use t_boost_core::{
    explain::{fixture_model, fixture_serve, RefMeasure},
    joint::{rejoint, JointOptions},
    table_model::TableModel,
};

fn main() {
    let mut m = fixture_model();
    let mut x = fixture_serve();
    m.trees[0].1.leaves = vec![0., 1., 1., 2., 0., 0., 0., 0.];
    println!("validate {:?} serve {:?}", m.validate(), x.0.data);
    for correlated in [false, true] {
        if correlated {
            x.0.data[1] = x.0.data[0].clone();
        }
        let b = m.explain(&x, RefMeasure::Uniform).unwrap();
        let j = rejoint(&b, &JointOptions::default()).unwrap();
        let tm = TableModel::from_model_and_bank(&m, j.clone());
        let p = tm.score_raw(&x.0, None).unwrap();
        let mean = p.iter().map(|v| f64::from(*v)).sum::<f64>() / 4.;
        let variance = p
            .iter()
            .map(|v| (f64::from(*v) - mean).powi(2))
            .sum::<f64>()
            / 4.;
        println!(
            "joint correlated {correlated} preds {:?} model variance {variance} mains {:?} sumsobol {}",
            p,
            j.tables
                .iter()
                .map(|t| (t.u.clone(), t.variance))
                .collect::<Vec<_>>(),
            j.sobol().iter().map(|(_, s)| s).sum::<f64>()
        );
        let export = j
            .to_rating_export(
                m.link,
                &m.mode,
                &m.schema,
                &m.provenance,
                &m.schema.cat_encoders,
                None,
            )
            .unwrap();
        println!(
            "joint exported {:?}",
            export
                .tables
                .iter()
                .map(|t| (t.feature_set.clone(), t.variance, t.sobol))
                .collect::<Vec<_>>()
        );
    }
}
