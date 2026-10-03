use t_boost_core::{
    explain::{fixture_model, fixture_serve, RefMeasure},
    table_model::TableModel,
};

#[test]
fn bug031_regression() {
    let mut a = fixture_model();
    a.grids[0].borders = vec![0.0, 1.5];
    a.grids[0].n_bins = 4;
    a.trees[0].1.splits[0].bin_le = 2;
    let mut xa = fixture_serve();
    xa.0.grids = a.grids.clone();
    xa.0.data[0] = vec![2, 2, 3, 3];
    a.validate().unwrap();
    let ta = TableModel::from_model(&a, &xa, RefMeasure::Uniform).unwrap();
    ta.validate().unwrap();

    let mut b = a.clone();
    b.grids[0].borders = vec![1.5, 3.0];
    b.trees[0].1.splits[0].bin_le = 1;
    let mut xb = xa.clone();
    xb.0.grids = b.grids.clone();
    xb.0.data[0] = vec![1, 1, 2, 2];
    b.validate().unwrap();
    let tb = TableModel::from_model(&b, &xb, RefMeasure::Uniform).unwrap();
    tb.validate().unwrap();
    assert_eq!(ta.bank.merged_grids, tb.bank.merged_grids);
    println!("merged grids (shared): {:?}", tb.bank.merged_grids);
    println!("a.grids[0]: {:?}  b.grids[0]: {:?}", a.grids[0], b.grids[0]);
    let uncached = tb.score_raw(&xb.0, None).unwrap();
    let stale = tb.score_raw_with(&xb.0, None, Some(&ta.cell_maps().unwrap()));
    assert!(matches!(
        stale,
        Err(t_boost_core::PbError::InvalidInput { .. })
    ));
    assert_eq!(
        tb.score_raw_with(&xb.0, None, Some(&tb.cell_maps().unwrap()))
            .unwrap(),
        uncached
    );
    println!("correct: {:?}", tb.score_raw(&xb.0, None).unwrap());
    println!(
        "own cache: {:?}",
        tb.score_raw_with(&xb.0, None, Some(&tb.cell_maps().unwrap()))
            .unwrap()
    );
    println!(
        "tree model b predict: {:?}",
        b.predict(&xb.0, None).unwrap()
    );
}
