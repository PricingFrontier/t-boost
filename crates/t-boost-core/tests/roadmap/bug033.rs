use t_boost_core::{
    explain::{fixture_model, fixture_serve, RefMeasure},
    table_model::TableModel,
};

#[test]
fn bug033_regression() {
    let base =
        TableModel::from_model(&fixture_model(), &fixture_serve(), RefMeasure::Uniform).unwrap();
    let mut doc: serde_json::Value = serde_json::from_str(&base.to_json().unwrap()).unwrap();
    println!(
        "tables[0]: u={} axes[0].raw={}",
        doc["model"]["bank"]["tables"][0]["u"], doc["model"]["bank"]["tables"][0]["axes"][0]["raw"]
    );
    // This main-effect table has axes[0].raw == 0. Change only its claimed support.
    doc["model"]["bank"]["tables"][0]["u"] = serde_json::json!([1]);
    assert!(
        TableModel::from_json(&doc.to_string()).is_err(),
        "mismatched feature set accepted"
    );
}
