use t_boost_core::{
    explain::{fixture_model, fixture_serve, RefMeasure},
    joint::{rejoint, JointOptions},
};

#[test]
fn joint_shares_require_and_use_aligned_rows() {
    let mut model = fixture_model();
    model.trees[0].1.leaves = vec![0., 1., 0.5, 1.5, 0., 0., 0., 0.];
    for correlation in [0, 1, -1] {
        let mut rows = fixture_serve();
        if correlation != 0 {
            rows.0.data[1] = rows.0.data[0]
                .iter()
                .map(|v| if correlation > 0 { *v } else { 3 - *v })
                .collect();
        }
        let product = model.explain(&rows, RefMeasure::Uniform).unwrap();
        let mut bank = rejoint(&product, &JointOptions::default()).unwrap();
        assert!(bank.sobol().is_empty());
        assert!(bank
            .to_rating_export(
                model.link,
                &model.mode,
                &model.schema,
                &model.provenance,
                &model.schema.cat_encoders,
                None
            )
            .is_err());
        let mass = [1., 2., 3., 4.];
        bank.measure_joint_variance(&model.schema.cat_encoders, &rows.0, Some(&mass))
            .unwrap();
        let mut scores = vec![0.; 4];
        bank.score_binned(&model.schema.cat_encoders, &rows.0, &mut scores)
            .unwrap();
        let mean = scores
            .iter()
            .zip(mass)
            .map(|(v, w)| v * f64::from(w))
            .sum::<f64>()
            / 10.;
        let variance = scores
            .iter()
            .zip(mass)
            .map(|(v, w)| (v - mean).powi(2) * f64::from(w))
            .sum::<f64>()
            / 10.;
        assert!((bank.joint_variance.unwrap() - variance).abs() < 1e-12);
        let export = bank
            .to_rating_export(
                model.link,
                &model.mode,
                &model.schema,
                &model.provenance,
                &model.schema.cat_encoders,
                None,
            )
            .unwrap();
        for table in export.tables {
            assert!((table.sobol - table.variance / variance).abs() < 1e-12);
        }
        let before = bank.sobol();
        bank.f0 += 1e12;
        bank.measure_joint_variance(&model.schema.cat_encoders, &rows.0, Some(&mass))
            .unwrap();
        assert_eq!(before, bank.sobol());
    }
}
