use t_boost_core::{
    data::{AxisKind, AxisProvenance, FeatureId},
    engine::Split,
    explain::{fixture_model, fixture_serve, OverflowPolicy, RefMeasure, TableBudget},
    table_model::TableModel,
};

fn main() {
    println!("--- whole-bank");
    let model = fixture_model();
    let x = fixture_serve();
    println!("fixture leaves {:?} depth {}", model.trees[0].1.leaves, model.trees[0].1.depth);
    for cap in [8, 9, 14, 15] {
        let got = model.explain_with_budget(
            &x,
            RefMeasure::Uniform,
            TableBudget {
                max_table_cells: 9,
                max_bank_cells: cap,
                on_overflow: OverflowPolicy::Error,
            },
        );
        println!(
            "cap {cap}: {:?}",
            got.map(|b| (b.tables.iter().map(|t| (t.u.clone(), t.values.len())).collect::<Vec<_>>(), b.tables.iter().map(|t| t.values.len()).sum::<usize>()))
        );
    }

    println!("--- factored/per-table");
    let mut m = fixture_model();
    m.grids.push(m.grids[0].clone());
    m.provenance.push(AxisProvenance {
        raw: FeatureId(2),
        kind: AxisKind::Numeric,
    });
    m.schema.feature_names.push("x2".into());
    m.schema.feature_kinds.push(AxisKind::Numeric);
    m.trees[0].1.depth = 3;
    m.trees[0].1.splits.push(Split {
        axis: 2,
        bin_le: 1,
        missing_left: false,
    });
    m.trees[0].1.leaves = vec![0., 0., 0., 0., 0., 0., 0., 8.];
    let mut x = fixture_serve();
    x.0.data.push(vec![1, 2, 1, 2]);
    x.0.grids = m.grids.clone();
    x.0.provenance = m.provenance.clone();
    println!("validate {:?}", m.validate());
    let b = m
        .explain_with_budget(
            &x,
            RefMeasure::Uniform,
            TableBudget {
                max_table_cells: 1,
                max_bank_cells: 1,
                on_overflow: OverflowPolicy::Factored,
            },
        )
        .unwrap();
    println!(
        "cells {:?} total {} factored {}",
        b.tables.iter().map(|t| (t.u.clone(), t.values.len(), t.values.is_sparse())).collect::<Vec<_>>(),
        b.tables.iter().map(|t| t.values.len()).sum::<usize>(),
        b.factored.len()
    );
    println!(
        "valid {:?}",
        TableModel::from_model_and_bank(&m, b).validate()
    );

    println!("--- sparse-density variant");
    let mut m = fixture_model();
    m.trees[0].1.leaves = vec![0., 0., 0., 9., 0., 0., 0., 0.];
    println!("validate {:?}", m.validate());
    let x = fixture_serve();
    let got = m.explain_with_budget(
        &x,
        RefMeasure::Uniform,
        TableBudget {
            max_table_cells: 1,
            max_bank_cells: 100,
            on_overflow: OverflowPolicy::SparseFallback { density_threshold: 0.2 },
        },
    );
    match got {
        Ok(b) => {
            for t in &b.tables {
                let vals = t.values.values();
                let nz = vals.iter().filter(|v| **v != 0.0).count();
                println!(
                    "table {:?}: shape {:?} sparse {} len {} nonzero {}/{} values {:?}",
                    t.u, t.values.shape(), t.values.is_sparse(), t.values.len(), nz, vals.len(), vals
                );
            }
            println!("factored {}", b.factored.len());
        }
        Err(e) => println!("err {e:?}"),
    }
}
