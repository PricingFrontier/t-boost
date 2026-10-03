use t_boost_core::explain::{fixture_model, fixture_serve, RefMeasure};
use t_boost_core::loss::{Link, LossId};
use t_boost_core::table_model::TableModel;

#[test]
fn bug009_regression() {
    let mut model = fixture_model();
    model.f0 = -35.0;
    model.trees[0].0 = 20.0;
    model.link = Link::Log;
    model.schema.objective.link = Link::Log;
    model.schema.objective.loss = LossId::Poisson;
    model.validate().unwrap();
    let serve = fixture_serve();
    let tm = TableModel::from_model(&model, &serve, RefMeasure::Uniform).unwrap();
    println!("validate: {:?}", tm.validate());
    println!("bank f0 = {}", tm.bank.f0);

    // Row 1 of fixture_serve is bins (1,2).
    let row = 1usize;
    let tree_pred = model.predict(&serve.0, None).unwrap();
    let served = tm.predict_binned(&serve.0, None).unwrap();
    let raw = tm.score_raw(&serve.0, None).unwrap();
    println!("tree model predict (all rows): {tree_pred:?}");
    println!("table model predict (all rows): {served:?}");
    println!("table model raw score (all rows): {raw:?}");

    let cells = tm.column_cells(&serve.0).unwrap(); // [raw][row]
    let row_cells: Vec<u32> = cells.iter().map(|c| c[row]).collect();
    println!("row {row} merged cells: {row_cells:?}");

    let export = tm
        .bank
        .to_rating_export(
            tm.link,
            &tm.mode,
            &tm.schema,
            &tm.provenance,
            &tm.schema.cat_encoders,
            None,
        )
        .unwrap();
    println!("export f0 = {}", export.f0);
    let mut prod = 1.0_f64;
    let mut sum = 0.0_f64;
    for t in &export.tables {
        // row-major index over `shape`, coordinate on axis d = row's cell on axes[d].raw
        let mut idx = 0usize;
        for (d, ax) in t.axes.iter().enumerate() {
            idx = idx * t.shape[d] as usize + row_cells[ax.raw as usize] as usize;
        }
        let v = t.values[idx];
        let r = t.relativities.as_ref().unwrap()[idx];
        println!(
            "  table {:?} shape {:?} idx {idx}: value {v} relativity {r} exp(value) {}",
            t.feature_set,
            t.shape,
            v.exp()
        );
        println!("    all values: {:?}", t.values);
        prod *= r;
        sum += v;
    }
    assert!((export.f0.exp() * prod - f64::from(served[row])).abs() < 1e-4);
    for table in &export.tables {
        for (&value, &relativity) in table
            .values
            .iter()
            .zip(table.relativities.as_ref().unwrap())
        {
            assert!((relativity - value.exp()).abs() <= 1e-12 * value.exp());
        }
    }
    println!("served prediction row {row}            = {}", served[row]);
    println!(
        "exp(f0) * product(relativities)   = {}",
        export.f0.exp() * prod
    );
    println!(
        "exp(f0 + sum(values))             = {}",
        (export.f0 + sum).exp()
    );
    println!(
        "exp(clamp(f0 + sum(values)))      = {}",
        (export.f0 + sum).clamp(-30.0, 30.0).exp()
    );
}
