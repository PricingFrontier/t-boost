//! The DEPLOYED-BOX BUDGET, end to end on a real lifted fit.
//!
//! `prune.rs`'s unit tests pin the selector's arithmetic on synthetic costs. This file pins the
//! two things only a real fit can show:
//!
//!   1. a generous budget produces the SAME deployed model as no budget at all — the no-op
//!      guarantee that makes "budget off is byte-identical to order-lift HEAD" checkable, and
//!   2. a tight budget actually lands under it, on a bank whose boxes are the thing that grew.
//!
//! The fixture is the order lift's own: a 4-way sign product that only a depth-6 order-4 tree
//! can reach, which is precisely the shape whose deployed box count explodes.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use t_boost_core::explain::FeatureSet;
use t_boost_core::prune::{
    apply_box_budget, box_costs, box_variances, prune_model_to_keepset,
    prune_model_to_keepset_budgeted,
};
use t_boost_core::{
    bin_columns, BinConfig, Booster, Config, CredibilityFloor, FitSpec, InteractionPolicy, Model,
    MonotoneMap, RefMeasure, ServeBinnedMatrix, SquaredError,
};

fn xs(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 11) as f64 / (1_u64 << 53) as f64
}

/// Five features carrying a genuine 4-way term (the sign product of four of them) plus a
/// main-effect-only decoy — the `order_lift.rs` fixture, kept in step with it deliberately.
fn dataset(n: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut st = seed | 1;
    let mut cols: Vec<Vec<f32>> = (0..5).map(|_| Vec::with_capacity(n)).collect();
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let v: [f64; 5] = std::array::from_fn(|_| xs(&mut st) * 2.0 - 1.0);
        let quad = v.iter().take(4).map(|x| x.signum()).product::<f64>();
        let decoy = v.last().copied().unwrap_or(0.0);
        y.push((2.0 * quad + 0.7 * decoy + 0.1 * (xs(&mut st) - 0.5)) as f32);
        for (c, value) in cols.iter_mut().zip(&v) {
            c.push(*value as f32);
        }
    }
    (cols, y)
}

fn fit(max_depth: u8, max_order: u8) -> (Model, ServeBinnedMatrix, Vec<f32>, Vec<f32>) {
    let (cols, y) = dataset(2500, 0x0_4de5);
    let refs: Vec<&[f32]> = cols.iter().map(Vec::as_slice).collect();
    let x = bin_columns(&refs, None, &BinConfig::default(), 0).unwrap();
    let cfg = Config {
        n_trees: 40,
        learning_rate: 0.3,
        lambda: 1.0,
        interaction_gain_hurdle: 2.0,
        interaction_gain_hurdle_mode: t_boost_core::engine::InteractionGainHurdleMode::Adaptive,
        ..Config::default()
    };
    let spec = FitSpec {
        loss: &SquaredError,
        weight: None,
        exposure: None,
        monotone: MonotoneMap::new(),
        interaction: InteractionPolicy {
            max_order,
            max_depth,
            ..InteractionPolicy::default()
        },
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed: 0,
    };
    let model = Booster::with_config(cfg).fit(&x, &y, &spec).unwrap();
    let w = vec![1.0_f32; y.len()];
    (model, ServeBinnedMatrix(x), y, w)
}

fn deployed_boxes(tm: &t_boost_core::table_model::TableModel) -> usize {
    tm.bank.factored.iter().map(|ft| ft.n_boxes()).sum()
}

fn full_keepset(model: &Model, serve: &ServeBinnedMatrix) -> Vec<FeatureSet> {
    let bank = model.explain(serve, RefMeasure::default()).unwrap();
    bank.tables
        .iter()
        .map(|t| t.u.clone())
        .chain(bank.factored.iter().map(|ft| ft.u.clone()))
        .collect()
}

/// A budget at or above the bank's own box total deploys the SAME bytes as no budget. Stated
/// on the serialized document, not on a field-by-field walk, because the document is what
/// ships.
#[test]
fn a_generous_budget_deploys_the_same_bytes_as_no_budget() {
    let (model, serve, y, w) = fit(6, 4);
    let keep = full_keepset(&model, &serve);
    let plain = prune_model_to_keepset(
        &model,
        &serve,
        &y,
        &w,
        None,
        &SquaredError,
        RefMeasure::default(),
        false,
        &keep,
        false,
    )
    .unwrap();
    let boxes = deployed_boxes(&plain);
    assert!(
        boxes > 0,
        "the fixture must deploy factored boxes or this test proves nothing"
    );

    for max_boxes in [0_usize, boxes, boxes + 1, usize::MAX] {
        let (budgeted, report, _) = prune_model_to_keepset_budgeted(
            &model,
            &serve,
            &y,
            &w,
            None,
            &SquaredError,
            RefMeasure::default(),
            false,
            &keep,
            false,
            max_boxes,
            0,
            t_boost_core::prune::DEFAULT_TABLE_MIN_ARITY,
            &BTreeMap::new(),
        )
        .unwrap();
        assert!(!report.engaged, "max_boxes={max_boxes} must stay idle");
        assert_eq!(
            plain.to_json().unwrap(),
            budgeted.to_json().unwrap(),
            "max_boxes={max_boxes} must deploy an identical document"
        );
    }
}

/// A tight budget lands UNDER it, drops only factored mass, and leaves the dense tables alone
/// — the dense bank is governed by the cell firewall, not by this gate.
#[test]
fn a_tight_budget_lands_under_it_and_spares_the_dense_tables() {
    let (model, serve, y, w) = fit(6, 4);
    let keep = full_keepset(&model, &serve);
    let plain = prune_model_to_keepset(
        &model,
        &serve,
        &y,
        &w,
        None,
        &SquaredError,
        RefMeasure::default(),
        false,
        &keep,
        false,
    )
    .unwrap();
    let boxes = deployed_boxes(&plain);
    let dense_before = plain.bank.tables.len();

    for divisor in [2_usize, 4, 16] {
        let budget = boxes / divisor;
        let (budgeted, report, _) = prune_model_to_keepset_budgeted(
            &model,
            &serve,
            &y,
            &w,
            None,
            &SquaredError,
            RefMeasure::default(),
            false,
            &keep,
            false,
            budget,
            0,
            t_boost_core::prune::DEFAULT_TABLE_MIN_ARITY,
            &BTreeMap::new(),
        )
        .unwrap();
        assert!(report.engaged, "budget={budget}");
        let after = deployed_boxes(&budgeted);
        assert_eq!(after as u64, report.boxes_after);
        assert!(
            after <= budget,
            "budget={budget} but deployed {after} boxes"
        );
        assert_eq!(
            budgeted.bank.tables.len(),
            dense_before,
            "budget={budget} must not touch a dense table"
        );
        // Heredity: nothing deploys whose proper subset the budget removed.
        let gone: Vec<&FeatureSet> = report
            .dropped
            .iter()
            .chain(report.cascade_dropped.iter())
            .collect();
        for ft in &budgeted.bank.factored {
            for d in &gone {
                assert!(
                    !(d.order() < ft.u.order() && d.0.iter().all(|f| ft.u.contains(*f))),
                    "budget={budget}: deployed {:?} over dropped subset {:?}",
                    ft.u.0,
                    d.0
                );
            }
        }
    }
}

/// The costs the budget spends against are the DEPLOYED bank's own box counts — not an
/// estimate, and not a function of the keep-set. This is the invariant that lets the budget be
/// decided in one pass: retaining a subset of supports leaves every surviving effect's box
/// count exactly where it was.
#[test]
fn box_costs_are_a_property_of_the_model_not_of_the_keepset() {
    let (model, serve, _, _) = fit(6, 4);
    let bank = model.explain(&serve, RefMeasure::default()).unwrap();
    let costs = box_costs(&[&bank]);
    assert!(
        !costs.is_empty(),
        "the fixture must produce factored effects"
    );

    // Drop the most expensive support and re-cost: every survivor keeps its price.
    let dearest = costs
        .iter()
        .max_by_key(|(_, c)| **c)
        .map(|(u, _)| u.clone())
        .unwrap();
    let keep: Vec<FeatureSet> = bank
        .tables
        .iter()
        .map(|t| t.u.clone())
        .chain(bank.factored.iter().map(|ft| ft.u.clone()))
        .filter(|u| *u != dearest)
        .collect();
    let retained = t_boost_core::prune::retain_tables(&bank, &keep);
    let after = box_costs(&[&retained]);
    for (u, c) in &after {
        assert_eq!(
            costs.get(u),
            Some(c),
            "{:?} was re-priced by the prune",
            u.0
        );
    }
    assert!(!after.contains_key(&dearest));

    // And the selector agrees with the bank it was costed from.
    let evidence: BTreeMap<FeatureSet, f64> = BTreeMap::new();
    let all: Vec<FeatureSet> = bank
        .tables
        .iter()
        .map(|t| t.u.clone())
        .chain(bank.factored.iter().map(|ft| ft.u.clone()))
        .collect();
    let total: usize = costs.values().sum();
    let (_, rep) = apply_box_budget(&all, &costs, &box_variances(&[&bank]), total, &evidence);
    assert_eq!(rep.boxes_before, total as u64);
    assert!(!rep.engaged);
}
