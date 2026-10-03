use t_boost_core::{
    explain::{fixture_model, fixture_serve, FeatureSet, RefMeasure},
    prune::retain_tables,
    table_model::TableModel,
};

fn main() {
    // Written by ../../python/bug030.py; run that first.
    let s = concat!(env!("CARGO_MANIFEST_DIR"), "/../out");
    let mut tm =
        TableModel::from_json(&std::fs::read_to_string(format!("{s}/tboost_review_banded.json")).unwrap())
            .unwrap();
    println!("tables before retain: {:?}", tm.bank.tables.iter().map(|t| (t.u.clone(), t.axes.iter().map(|a| a.band_of.is_some()).collect::<Vec<_>>())).collect::<Vec<_>>());
    tm.bank = retain_tables(&tm.bank, &[FeatureSet::new(&[0, 1])]);
    println!("tables after retain: {:?}", tm.bank.tables.iter().map(|t| t.u.clone()).collect::<Vec<_>>());
    println!("validate: {:?}", tm.validate());
    println!(
        "recenter banded: {:?}",
        tm.bank.recompute_under(RefMeasure::Uniform).map(|_| ())
    );
    // control: unbanded interaction-only bank
    let m = fixture_model();
    let x = fixture_serve();
    let b = m.explain(&x, RefMeasure::Uniform).unwrap();
    println!("fixture tables: {:?}", b.tables.iter().map(|t| t.u.clone()).collect::<Vec<_>>());
    let b2 = retain_tables(&b, &[FeatureSet::new(&[0, 1])]);
    let tm2 = TableModel::from_model_and_bank(&m, b2.clone());
    println!("unbanded validate: {:?}", tm2.validate());
    let r = b2.recompute_under(RefMeasure::Uniform);
    println!("unbanded recenter: {:?}", r.as_ref().map(|b| b.tables.iter().map(|t| t.u.clone()).collect::<Vec<_>>()));
}
