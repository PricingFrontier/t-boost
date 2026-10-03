use t_boost_core::{
    explain::{fixture_model, fixture_serve, RefMeasure},
    prune::retain_tables,
    table_model::TableModel,
};

fn support_probe() {
    let m = fixture_model();
    let x = fixture_serve();
    println!("serve data {:?}", x.0.data);
    let b = m.explain(&x, RefMeasure::Uniform).unwrap();
    let control = TableModel::from_model_and_bank(&m, b.clone())
        .recentred_bank(
            &x.0,
            Some(&[1., 2., 3., 10.]),
            RefMeasure::ExposureMarginals { floor: 0.01 },
        )
        .unwrap();
    println!(
        "support full-bank control {:?}",
        control
            .tables
            .iter()
            .filter(|t| t.u.order() == 1)
            .map(|t| (t.u.clone(), t.support.values().to_vec()))
            .collect::<Vec<_>>()
    );
    let keep = b
        .tables
        .iter()
        .filter(|t| t.u.order() == 2)
        .map(|t| t.u.clone())
        .collect::<Vec<_>>();
    let b = retain_tables(&b, &keep);
    let tm = TableModel::from_model_and_bank(&m, b);
    let mass = [1., 2., 3., 10.];
    let r = tm
        .recentred_bank(
            &x.0,
            Some(&mass),
            RefMeasure::ExposureMarginals { floor: 0.01 },
        )
        .unwrap();
    for t in &r.tables {
        assert!(
            (t.support.values().iter().sum::<f64>() - 16.).abs() < 1e-12,
            "recreated effect has wrong support: {:?}",
            t.u
        );
        if t.u.order() == 1 {
            let expected = control.tables.iter().find(|other| other.u == t.u).unwrap();
            assert_eq!(t.support.values(), expected.support.values());
        }
        println!(
            "support {:?}: values {:?} support {:?}",
            t.u,
            t.values.values(),
            t.support.values()
        );
    }
    let tm2 = TableModel::from_model_and_bank(&m, r.clone());
    let direct = tm2
        .recentred_bank(&x.0, Some(&mass), RefMeasure::default())
        .unwrap();
    let cached = r.recompute_under(RefMeasure::default()).unwrap();
    assert!((r.recompute_under(r.w.clone()).unwrap().f0 - r.f0).abs() < 1e-12);
    assert!((cached.f0 - direct.f0).abs() < 1e-12);
    println!(
        "same measure f0 {} -> {}",
        r.f0,
        r.recompute_under(r.w.clone()).unwrap().f0
    );
    println!("default cached f0 {} explicit f0 {}", cached.f0, direct.f0);
    for t in &cached.tables {
        println!(
            "cached {:?}: values {:?} support {:?}",
            t.u,
            t.values.values(),
            t.support.values()
        );
    }
    let mut maxchange = 0f64;
    for i in 0..3 {
        for j in 0..3 {
            maxchange = maxchange
                .max((cached.score(&[i, j]).unwrap() - direct.score(&[i, j]).unwrap()).abs());
        }
    }
    println!("function cached vs explicit diff {maxchange}");
}
#[test]
fn bug063_regression() {
    support_probe();
}

#[test]
fn bank_only_recentring_marginalizes_surviving_interaction_support() {
    let model = fixture_model();
    let rows = fixture_serve();
    let mass = [1., 2., 3., 10.];
    let bank = model
        .explain_weighted(&rows, RefMeasure::Uniform, &mass)
        .unwrap();
    let keep = bank
        .tables
        .iter()
        .filter(|t| t.u.order() == 2)
        .map(|t| t.u.clone())
        .collect::<Vec<_>>();
    let bank = retain_tables(&bank, &keep);
    let measure = RefMeasure::ExposureMarginals { floor: 0.01 };
    let cached = bank.recompute_under(measure.clone()).unwrap();
    let expected = TableModel::from_model_and_bank(&model, bank)
        .recentred_bank(&rows.0, Some(&mass), measure.clone())
        .unwrap();
    assert!((cached.f0 - expected.f0).abs() < 1e-12);
    for (table, reference) in cached.tables.iter().zip(&expected.tables) {
        assert_eq!(table.support.values(), reference.support.values());
        assert_eq!(table.values.values(), reference.values.values());
    }
    let model = TableModel::from_model_and_bank(&model, cached);
    let loaded = TableModel::from_bincode(&model.to_bincode().unwrap()).unwrap();
    let again = loaded.bank.recompute_under(measure).unwrap();
    assert!((again.f0 - expected.f0).abs() < 1e-12);
    for (table, reference) in again.tables.iter().zip(&expected.tables) {
        assert_eq!(table.support.values(), reference.support.values());
    }
}
