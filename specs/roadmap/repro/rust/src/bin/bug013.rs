use t_boost_core::data::{AxisKind, AxisProvenance, BinnedMatrix, FeatureId, ServeBinnedMatrix};
use t_boost_core::engine::{ObliviousTree, Split};
use t_boost_core::explain::{fixture_model, RefMeasure};
use t_boost_core::table_model::TableModel;

fn main() {
    let mut model = fixture_model();
    let grid = model.grids[0].clone();
    model.grids = vec![grid.clone(), grid.clone(), grid.clone()];
    model.provenance.push(AxisProvenance { raw: FeatureId(2), kind: AxisKind::Numeric });
    model.schema.feature_names.push("x2".into());
    model.schema.feature_kinds.push(AxisKind::Numeric);
    let leaves = [0.125_f32, 1.25, 2.75, -3.875, 4.5, 5.125, 6.875, 8.625];
    model.trees = vec![(
        1.0,
        ObliviousTree {
            splits: (0..3)
                .map(|a| Split { axis: a, bin_le: 1, missing_left: false })
                .collect(),
            leaves: leaves.to_vec(),
            depth: 3,
        },
    )];
    model.validate().unwrap();
    // All eight finite-bin combinations {1,2}^3.
    let mut data = vec![Vec::new(); 3];
    for r in 0..8u8 {
        for a in 0..3 {
            data[a].push(1 + ((r >> a) & 1));
        }
    }
    let serve = ServeBinnedMatrix(BinnedMatrix {
        data,
        n_rows: 8,
        grids: model.grids.clone(),
        provenance: model.provenance.clone(),
    });
    let tm = TableModel::from_model(&model, &serve, RefMeasure::Uniform).unwrap();
    println!("validate (fresh): {:?}", tm.validate());
    println!("tables {} factored {}", tm.bank.tables.len(), tm.bank.factored.len());
    let doc: serde_json::Value = serde_json::from_str(&tm.to_json().unwrap()).unwrap();
    let p = &doc["model"]["bank"]["factored"][0]["boxes"][0]["p"];
    println!("before (boxes[0].p from JSON): {p}");
    let p0 = p[0].as_f64().unwrap();
    let before_pred = tm.predict_binned(&serve.0, None).unwrap();
    println!("before predictions: {before_pred:?}");

    // Byte-patch the first box corner in the bincode blob: varint length 8 then 8 LE f64 bytes.
    let mut bytes = tm.to_bincode().unwrap();
    let mut needle = vec![8u8];
    needle.extend_from_slice(&p0.to_le_bytes());
    let hits: Vec<usize> = (0..bytes.len() - needle.len())
        .filter(|&i| bytes[i..i + needle.len()] == needle[..])
        .collect();
    println!("needle (len-8 prefix + {p0} LE) occurrences at: {hits:?}");
    assert_eq!(hits.len(), 1, "needle must be unique");
    let off = hits[0] + 1;
    bytes[off..off + 8].copy_from_slice(&f64::NAN.to_le_bytes());

    // Control: untouched bytes round-trip.
    let clean = TableModel::from_bincode(&tm.to_bincode().unwrap());
    println!("control from_bincode (untouched): {:?}", clean.as_ref().map(|_| ()));
    println!("control validate: {:?}", clean.as_ref().unwrap().validate());

    let loaded = TableModel::from_bincode(&bytes);
    println!("from_bincode (NaN corner): {:?}", loaded.as_ref().map(|_| ()));
    let Ok(loaded) = loaded else { return };
    println!("validate (NaN corner): {:?}", loaded.validate());
    let doc2: serde_json::Value = serde_json::from_str(&loaded.to_json().unwrap_or_else(|e| format!("\"to_json error: {e}\""))).unwrap();
    println!("after (boxes[0].p from JSON; NaN renders as null): {}", doc2["model"]["bank"]["factored"][0]["boxes"][0]["p"]);
    println!("after predictions: {:?}", loaded.predict_binned(&serve.0, None));
    // Also the direct per-row bank score for cells (1,1,1) -> merged cells (1,1,1).
    println!("bank.score([1,1,1]) after: {:?}", loaded.bank.score(&[1, 1, 1]));
    println!("bank.score([2,2,2]) after: {:?}", loaded.bank.score(&[2, 2, 2]));
}
