use t_boost_core::{
    data::{AxisKind, AxisProvenance, FeatureId},
    engine::Split,
    explain::{fixture_multichannel_model, fixture_multichannel_serve, RefMeasure},
    prune::retain_tables,
    table_model::TableModel,
};

fn cat_probe() {
    let mut m = fixture_multichannel_model();
    let mut x = fixture_multichannel_serve();
    for raw in 1..=3 {
        let axis = m.grids.len() as u32;
        m.grids.push(m.grids[0].clone());
        m.provenance.push(AxisProvenance {
            raw: FeatureId(raw),
            kind: AxisKind::Numeric,
        });
        m.schema.feature_names.push(format!("x{raw}"));
        m.schema.feature_kinds.push(AxisKind::Numeric);
        m.trees[0].1.splits.push(Split {
            axis,
            bin_le: 1,
            missing_left: false,
        });
        x.0.data.push(vec![1, 2, 1, 2]);
    }
    m.trees[0].1.depth = 5;
    m.trees[0].1.leaves = (0..32).map(|v| if v == 31 { 32. } else { 0. }).collect();
    m.schema_version = t_boost_core::serialize::SCHEMA_VERSION;
    x.0.grids = m.grids.clone();
    x.0.provenance = m.provenance.clone();
    println!("model validate {:?}", m.validate());
    let b = m.explain(&x, RefMeasure::Uniform).unwrap();
    let keep = b
        .factored
        .iter()
        .filter(|f| f.u.order() == 4)
        .map(|f| f.u.clone())
        .collect::<Vec<_>>();
    println!("keep {:?}", keep);
    let p = retain_tables(&b, &keep);
    for (name, bank) in [("full", b), ("pruned", p)] {
        println!(
            "{name}: dense{} factored{}",
            bank.tables.len(),
            bank.factored.len()
        );
        let before = TableModel::from_model_and_bank(&m, bank.clone());
        let r = before
            .recentred_bank(
                &x.0,
                Some(&[1., 2., 3., 10.]),
                RefMeasure::ExposureMarginals { floor: 0.01 },
            )
            .unwrap();
        let tm = TableModel::from_model_and_bank(&m, r.clone());
        println!("recenter validates {:?}", tm.validate());
        let mut maxdiff = 0f64;
        for c in 0..5 {
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        let cells = [c, i, j, k];
                        maxdiff = maxdiff
                            .max((bank.score(&cells).unwrap() - r.score(&cells).unwrap()).abs());
                    }
                }
            }
        }
        assert!(maxdiff < 1e-12);
        tm.validate().unwrap();
        println!("maxdiff {maxdiff}");
        let f = r
            .factored
            .iter()
            .find(|f| f.u.0.iter().map(|r| r.0).collect::<Vec<_>>() == vec![0, 1, 2])
            .unwrap();
        assert_eq!(
            f.axes[0].joint_channels,
            Some(vec![
                t_boost_core::cat::TsEncodingId(0),
                t_boost_core::cat::TsEncodingId(1)
            ])
        );
        let cells = [1, 1, 1, 1];
        let exported = f
            .export_boxes()
            .unwrap()
            .iter()
            .map(|b| {
                let mut corner = 0;
                for (d, raw) in f.u.0.iter().enumerate() {
                    let low = match &b.categorical_low_cells[d] {
                        Some(ids) => ids.contains(&cells[raw.0 as usize]),
                        None => 1.0 <= b.thresholds[d],
                    };
                    if low {
                        corner |= 1 << d;
                    }
                }
                b.octants[corner]
            })
            .sum::<f64>();
        println!(
            "effect{:?}: var {}; cat axis {:?}; native {} exported {}",
            f.u,
            f.variance,
            f.axes[0],
            f.eval(&cells).unwrap(),
            exported
        );
        assert!((f.eval(&cells).unwrap() - exported).abs() < 1e-12);
        let boxes = f.export_boxes().unwrap();
        println!(
            "  first box thresholds {:?} cat_low {:?}",
            boxes.first().map(|b| b.thresholds.clone()),
            boxes.first().map(|b| b.categorical_low_cells.clone())
        );
        // also the rating export path
        let export = r.to_rating_export(
            m.link,
            &m.mode,
            &m.schema,
            &m.provenance,
            &m.schema.cat_encoders,
            None,
        );
        match export {
            Ok(e) => {
                for ft in &e.factored {
                    if ft.feature_set.0.iter().map(|r| r.0).collect::<Vec<_>>() == vec![0, 1, 2] {
                        println!(
                            "  rating export factored {:?}: first box thresholds {:?} cat_low {:?}",
                            ft.feature_set,
                            ft.boxes.first().map(|b| b.thresholds.clone()),
                            ft.boxes.first().map(|b| b.categorical_low_cells.clone())
                        );
                    }
                }
            }
            Err(e) => panic!("recentered categorical export failed: {e}"),
        }
    }
}

#[test]
fn bug062_regression() {
    cat_probe();
}
