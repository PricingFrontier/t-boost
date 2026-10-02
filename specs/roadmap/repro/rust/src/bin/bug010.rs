use t_boost_core::explain::{fixture_model, fixture_serve, RefMeasure};
use t_boost_core::serialize::TablesDoc;
use t_boost_core::table_model::TableModel;

fn strip_band_of(doc: &mut serde_json::Value) {
    for key in ["tables", "factored"] {
        if let Some(arr) = doc["model"]["bank"][key].as_array_mut() {
            for t in arr {
                if let Some(axes) = t["axes"].as_array_mut() {
                    for ax in axes {
                        ax.as_object_mut().unwrap().remove("band_of");
                    }
                }
            }
        }
    }
}

fn main() {
    let tm = TableModel::from_model(&fixture_model(), &fixture_serve(), RefMeasure::Uniform).unwrap();
    let json = tm.to_json().unwrap();
    let mut doc: serde_json::Value = serde_json::from_str(&json).unwrap();
    println!(
        "fresh doc: schema_version={} model.schema_version={} tables={} factored={}",
        doc["schema_version"],
        doc["model"]["schema_version"],
        doc["model"]["bank"]["tables"].as_array().unwrap().len(),
        doc["model"]["bank"]["factored"].as_array().unwrap().len()
    );

    // Control A: strip band_of only (keep v7 stamps) -> serde default should load it.
    let mut a = doc.clone();
    strip_band_of(&mut a);
    println!("control A (band_of stripped, stamps 7): {:?}", TableModel::from_json(&a.to_string()).map(|m| m.schema_version));

    // Reproduction: pre-v7 shape -- stamps 2, band_of absent.
    doc["schema_version"] = serde_json::json!(2);
    doc["model"]["schema_version"] = serde_json::json!(2);
    strip_band_of(&mut doc);
    let r = TableModel::from_json(&doc.to_string());
    println!("pre-v7 JSON (stamps 2, no band_of): {:?}", r.as_ref().map(|m| m.schema_version));
    if let Err(e) = &r {
        println!("  error: {e}");
    }

    // Variant: only envelope stamp 2, model stamp 7 -- which gate fires first?
    let mut b: serde_json::Value = serde_json::from_str(&json).unwrap();
    b["schema_version"] = serde_json::json!(2);
    strip_band_of(&mut b);
    println!("envelope 2 / model 7: {:?}", TableModel::from_json(&b.to_string()).map(|_| ()));
    // Variant: envelope 7, model 2 -- the model-level gate (table_model.rs:108-120).
    let mut c: serde_json::Value = serde_json::from_str(&json).unwrap();
    c["model"]["schema_version"] = serde_json::json!(2);
    strip_band_of(&mut c);
    println!("envelope 7 / model 2: {:?}", TableModel::from_json(&c.to_string()).map(|_| ()));

    // Binary control: the same version-2 stamp in the bincode envelope.
    let mut m2 = tm.clone();
    m2.schema_version = 2;
    let bdoc = TablesDoc {
        kind: "t-boost-tables".into(),
        format_version: 1,
        schema_version: 2,
        model: m2,
    };
    let body = bincode::serde::encode_to_vec(&bdoc, bincode::config::standard()).unwrap();
    let mut bytes = b"TBTM".to_vec();
    bytes.extend_from_slice(&body);
    let rb = TableModel::from_bincode(&bytes);
    println!("binary with stamp 2 (v7 byte layout): {:?}", rb.as_ref().map(|_| ()));
    if let Err(e) = &rb {
        println!("  error: {e}");
    }
    // Binary round trip control at the current stamp.
    let ok = TableModel::from_bincode(&tm.to_bincode().unwrap()).map(|m| m.schema_version);
    println!("binary round trip (stamp 7): {ok:?}");
}
