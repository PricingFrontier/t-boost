use t_boost_core::{
    explain::{fixture_model, fixture_serve, RefMeasure},
    table_model::TableModel,
};

fn main() {
    let base =
        TableModel::from_model(&fixture_model(), &fixture_serve(), RefMeasure::Uniform).unwrap();
    let mut doc: serde_json::Value = serde_json::from_str(&base.to_json().unwrap()).unwrap();
    println!(
        "tables[0]: u={} axes[0].raw={}",
        doc["model"]["bank"]["tables"][0]["u"], doc["model"]["bank"]["tables"][0]["axes"][0]["raw"]
    );
    // This main-effect table has axes[0].raw == 0. Change only its claimed support.
    doc["model"]["bank"]["tables"][0]["u"] = serde_json::json!([1]);
    let corrupt = TableModel::from_json(&doc.to_string()).unwrap();
    println!("validate {:?}", corrupt.validate());
    println!(
        "original {:?}, corrupted {:?}",
        base.bank.shap(&[1, 2]),
        corrupt.bank.shap(&[1, 2])
    );
    println!(
        "scores {:?}, {:?}",
        base.bank.score(&[1, 2]),
        corrupt.bank.score(&[1, 2])
    );
    let export = corrupt.bank.to_rating_export(
        corrupt.link,
        &corrupt.mode,
        &corrupt.schema,
        &corrupt.provenance,
        &corrupt.schema.cat_encoders,
        None,
    );
    println!("export {:?}", export.as_ref().map(|_| ()));
    if let Ok(e) = &export {
        for t in &e.tables {
            println!(
                "  export table feature_set={:?} names={:?} axes.raw={:?}",
                t.feature_set,
                t.feature_names,
                t.axes.iter().map(|a| (a.raw, a.name.clone())).collect::<Vec<_>>()
            );
        }
    }
    // Binary path: same corruption through bincode.
    let bytes = corrupt.to_bincode();
    println!("to_bincode of corrupt: {:?}", bytes.as_ref().map(|b| b.len()));
    if let Ok(b) = bytes {
        println!("from_bincode of corrupt: {:?}", TableModel::from_bincode(&b).map(|_| ()));
    }
    // Serving path: predictions identical?
    let x = fixture_serve();
    println!("base predict {:?}", base.predict_binned(&x.0, None).unwrap());
    println!("corrupt predict {:?}", corrupt.predict_binned(&x.0, None).unwrap());
}
