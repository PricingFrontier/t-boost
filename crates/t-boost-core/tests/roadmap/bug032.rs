use t_boost_core::explain::{fixture_model, fixture_serve, FeatureSet, RefMeasure, Tensor};
use t_boost_core::prune::retain_tables;
use t_boost_core::table_model::TableModel;

#[test]
fn bug032_recentring_banded_pair_preserves_axis_templates_for_new_mains() {
    let model = fixture_model();
    let bank = model
        .explain(&fixture_serve(), RefMeasure::Uniform)
        .unwrap();
    let mut bank = retain_tables(&bank, &[FeatureSet::new(&[0, 1])]);
    let table = &mut bank.tables[0];
    for axis in &mut table.axes {
        axis.band_of = Some(vec![0, 1, 1]);
        axis.cells = 2;
    }
    table.values = Tensor::from_vec(vec![2, 2], vec![1., -0.5, -0.5, 0.25]).unwrap();
    table.support = Tensor::from_vec(vec![2, 2], vec![0., 0., 0., 4.]).unwrap();
    table.variance = 0.25;
    TableModel::from_model_and_bank(&model, bank.clone())
        .validate()
        .unwrap();
    let recentered = bank.recompute_under(RefMeasure::Uniform).unwrap();
    TableModel::from_model_and_bank(&model, recentered.clone())
        .validate()
        .unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (bank.score(&[i, j]).unwrap() - recentered.score(&[i, j]).unwrap()).abs() < 1e-12
            );
        }
    }
    assert!(recentered.tables.iter().any(|t| t.u.order() == 1));
    for table in &recentered.tables {
        assert_eq!(table.support.values().iter().sum::<f64>(), 4.0);
        for axis in &table.axes {
            assert_eq!(axis.band_of, Some(vec![0, 1, 1]));
            assert_eq!(axis.cells, 2);
        }
    }
}
