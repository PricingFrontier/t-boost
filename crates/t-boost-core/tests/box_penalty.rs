//! The SELECTION-TIME box price (`PruneConfig::lambda_boxes`), end to end on a real fit.
//!
//! The av38 depth battery closed with one named gap: **selection cannot see bank size, so it
//! cannot refuse a diffuse config.** The backward prune walk optimizes held-out deviance
//! alone; the SE rule then breaks TIES toward fewer tables, but a rung that buys a real
//! deviance gain at twenty times the boxes is not a tie, so nothing in selection ever declines
//! it. The deployed-box budget (`box_budget.rs`) answers the same question at DEPLOY time,
//! after the choice has already been made. This is the selection-time analogue.
//!
//! Three properties, in the order they matter:
//!
//!   1. **Off is inert to the bit.** `lambda_boxes = 0.0` selects the same waypoint, and
//!      therefore ships the same bytes, as a build without the knob.
//!   2. **The path now carries `n_boxes` unconditionally**, so a lambda can be CALIBRATED from
//!      an unpenalized report instead of guessed.
//!   3. **A price actually buys parsimony**: raising it moves selection monotonically toward
//!      smaller banks, and a large enough price selects a waypoint carrying no boxes at all.
//!      Note what it does NOT do — collapse to the fewest TABLES. Once the box-bearing tables
//!      are gone the size term is flat and the remaining dense tables are chosen on fit, as
//!      they always were. The price is a brake on box diffusion; the SE rule is the general
//!      parsimony dial, and stacking a second one on it would make neither legible.

#![allow(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use t_boost_core::engine::MultiClassModel;
use t_boost_core::prune::{
    prune_bank, prune_multiclass_to_tables, HoldoutFold, PruneConfig, PruneReport,
};
use t_boost_core::{
    bin_columns, BinConfig, Booster, Config, CredibilityFloor, FitSpec, InteractionPolicy,
    MonotoneMap, RefMeasure, ServeBinnedMatrix, SquaredError, TableBank,
};

fn xs(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 11) as f64 / (1_u64 << 53) as f64
}

/// The order lift's own fixture: a genuine 4-way sign product plus a main-effect decoy, so a
/// depth-6 order-4 fit produces exactly the box-heavy bank the price is meant to shop in.
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

struct Fixture {
    bank: TableBank,
    x: ServeBinnedMatrix,
    y: Vec<f32>,
    w: Vec<f32>,
}

fn fixture() -> Fixture {
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
            max_order: 4,
            max_depth: 6,
            ..InteractionPolicy::default()
        },
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed: 0,
    };
    let model = Booster::with_config(cfg).fit(&x, &y, &spec).unwrap();
    let serve = ServeBinnedMatrix(x);
    let bank = model.explain(&serve, RefMeasure::default()).unwrap();
    let w = vec![1.0_f32; y.len()];
    Fixture {
        bank,
        x: serve,
        y,
        w,
    }
}

fn prune_at(f: &Fixture, lambda_boxes: f64) -> PruneReport {
    let folds = [HoldoutFold {
        x: &f.x.0,
        y: &f.y,
        w: &f.w,
        offset: None,
    }];
    let cfg = PruneConfig {
        se_rule: 0.0,
        lambda_boxes,
        lambda_tables: 0.0,
        table_price_min_arity: 3,
    };
    let (_bank, report) = prune_bank(
        &f.bank,
        &t_boost_core::cat::CatEncoderStore::new(),
        &folds,
        &SquaredError,
        &cfg,
    )
    .unwrap();
    report
}

/// **The inertness contract.** `lambda_boxes = 0.0` must select exactly what the unpenalized
/// selector selected, which is what makes the knob shippable default-off.
///
/// Compared on the selected KEEP-SET, not on a scalar summary: two different keep-sets can
/// share a table count and a deviance, so counting would not prove the claim. The default
/// `PruneConfig` is checked alongside the explicit zero, because "the default is inert" and
/// "zero is inert" are different statements and both need to hold.
#[test]
fn a_zero_price_selects_exactly_what_the_unpenalized_selector_selected() {
    let f = fixture();

    let explicit_zero = prune_at(&f, 0.0);
    let by_default = {
        let folds = [HoldoutFold {
            x: &f.x.0,
            y: &f.y,
            w: &f.w,
            offset: None,
        }];
        let cfg = PruneConfig {
            se_rule: 0.0,
            ..PruneConfig::default()
        };
        prune_bank(
            &f.bank,
            &t_boost_core::cat::CatEncoderStore::new(),
            &folds,
            &SquaredError,
            &cfg,
        )
        .unwrap()
        .1
    };

    assert_eq!(
        explicit_zero.kept, by_default.kept,
        "an explicit lambda_boxes=0.0 and the default must select the identical keep-set"
    );
    assert_eq!(explicit_zero.dropped, by_default.dropped);
    assert_eq!(
        PruneConfig::default().lambda_boxes,
        0.0,
        "the shipped default must be the inert value"
    );
}

/// The path reports `n_boxes` at every waypoint whether or not a price is armed — which is
/// what lets a caller CALIBRATE a lambda from a report it already has, instead of guessing at
/// a scale that is the dataset's, not the knob's.
///
/// Also pins the two structural properties a calibration relies on: waypoint 0 is the full
/// bank, and the box count is non-increasing along the backward walk (each step drops one
/// table, and a table's box cost is never negative).
#[test]
fn the_path_carries_box_counts_for_calibration_even_when_unpriced() {
    let f = fixture();
    let report = prune_at(&f, 0.0);

    let full_boxes: usize = f.bank.factored.iter().map(|ft| ft.n_boxes()).sum();
    assert!(
        full_boxes > 0,
        "the fixture must produce a factored (box-bearing) bank, or this file tests nothing"
    );
    assert_eq!(
        report.path[0].n_boxes as usize, full_boxes,
        "path point 0 must be the full bank's box count"
    );

    for w in report.path.windows(2) {
        assert!(
            w[1].n_boxes <= w[0].n_boxes,
            "box count must be non-increasing along the backward walk: {} -> {}",
            w[0].n_boxes,
            w[1].n_boxes
        );
    }
    assert_eq!(
        report.path.last().unwrap().n_boxes,
        0,
        "the intercept-only end of the path carries no boxes"
    );

    // The calibration recipes from the doc comment are computable from this report alone.
    let one_se_across_the_bank = report.path[0].se / f64::from(report.path[0].n_boxes.max(1));
    let one_percent =
        0.01 * report.path[0].mean_deviance / f64::from(report.path[0].n_boxes.max(1));
    assert!(one_se_across_the_bank.is_finite());
    assert!(one_percent.is_finite() && one_percent > 0.0);
}

/// **A price buys parsimony, monotonically.** Charging more per box may never select a LARGER
/// bank, and a price large enough to dominate the deviance scale collapses selection to the
/// smallest bank on the path.
///
/// This is the property that makes the knob a selection term rather than a decoration: it has
/// to be able to refuse a rung that the deviance alone would have taken. The ladder spans six
/// decades because the natural scale is the dataset's — which is precisely why the report
/// carries `n_boxes`, and why this test derives its top rung from the measured full-bank
/// deviance instead of hard-coding one.
#[test]
fn raising_the_price_moves_selection_monotonically_toward_a_smaller_bank() {
    let f = fixture();
    let baseline = prune_at(&f, 0.0);
    let full_boxes = f64::from(baseline.path[0].n_boxes.max(1));
    let dev = baseline.path[0].mean_deviance.abs().max(1e-12);

    // Prices expressed as "this fraction of the full-bank deviance, charged across the whole
    // bank" — dimensionless in the dataset's scale, so the ladder means the same thing on any
    // fixture.
    let ladder: Vec<f64> = [0.0, 1e-3, 1e-1, 1.0, 10.0]
        .iter()
        .map(|frac| frac * dev / full_boxes)
        .collect();

    let sizes: Vec<usize> = ladder.iter().map(|&l| prune_at(&f, l).kept.len()).collect();
    for w in sizes.windows(2) {
        assert!(
            w[1] <= w[0],
            "a higher box price selected a LARGER bank: {sizes:?} over prices {ladder:?}"
        );
    }
    assert!(
        *sizes.last().unwrap() < sizes[0],
        "a price of 10x the full-bank deviance must refuse rungs the unpriced selector \
         took: {sizes:?}"
    );

    // At an overwhelming price the selector lands on a ZERO-BOX waypoint — it has stopped
    // trading deviance for boxes entirely.
    //
    // Zero boxes, not fewest TABLES, and the difference is the whole design. Once every
    // box-bearing table is gone the size term is flat at 0, so the objective is back to being
    // pure deviance and the walk's remaining (dense, zero-cost) tables are chosen on fit
    // exactly as they always were. That is intended: the price is a brake on box diffusion,
    // not a general parsimony dial — the SE rule is already the general parsimony dial, and
    // stacking a second one on top of it would make neither legible.
    let extreme = prune_at(&f, 1e6 * dev / full_boxes);
    let extreme_boxes: usize = extreme
        .kept
        .iter()
        .filter_map(|u| f.bank.factored.iter().find(|ft| &ft.u == u))
        .map(|ft| ft.n_boxes())
        .sum();
    assert_eq!(
        extreme_boxes, 0,
        "an overwhelming price must select a waypoint carrying no priced boxes at all"
    );
    assert!(
        extreme.kept.len() < baseline.kept.len(),
        "the overwhelming-price bank ({}) must be smaller than the unpriced one ({})",
        extreme.kept.len(),
        baseline.kept.len()
    );
}

/// Dense tables cost zero boxes and are therefore never priced — the same convention the
/// deploy-time budget uses, so a selection-time price and a deploy-time cap can never disagree
/// about what a bank costs.
///
/// A main effect or a pair is what a filing READS; it is not what inflates it. Charging for
/// them would make the knob a general parsimony dial (the SE rule already is one) rather than
/// the box-diffusion brake it is meant to be.
#[test]
fn an_overwhelming_price_still_cannot_charge_a_dense_table() {
    let f = fixture();
    let dev = 1.0_f64;
    let extreme = prune_at(&f, 1e9 * dev);

    // Whatever survives an overwhelming price, its cost must be zero — i.e. it is dense.
    let costed: usize = extreme
        .kept
        .iter()
        .filter_map(|u| f.bank.factored.iter().find(|ft| &ft.u == u))
        .map(|ft| ft.n_boxes())
        .sum();
    assert_eq!(
        costed, 0,
        "an overwhelming price left {costed} priced boxes deployed; only zero-cost dense \
         tables should be able to survive it"
    );
}

// ---------------------------------------------------------------------------------
// The MULTICLASS path's box accounting.
// ---------------------------------------------------------------------------------

/// **The reported `n_boxes` must equal the boxes the returned model actually deploys** — on
/// the multiclass path, which is the one this battery's prize dataset (`fremotor_payfreq`,
/// 4 classes) really runs.
///
/// Everything above exercises the scalar `prune_bank`. The multiclass path is the more
/// intricate of the two and was flagged as untested in review: its id space is not "dense
/// tables then factored" but a `BTreeMap`-keyed UNION of supports across classes, and the box
/// cost of a support is SUMMED over every class's copy of it. An index misalignment between
/// `box_cost` and `ids` there would be invisible in any skill number and would quietly
/// mis-price every selection the knob is supposed to inform.
///
/// So this asserts the accounting against ground truth rather than against itself: find the
/// waypoint the selector actually chose (the one whose `n_tables` matches the keep-set), and
/// require its `n_boxes` to equal the box total of the `MultiClassTableModel` that came back.
/// If `box_cost` were indexed in a different order from `ids`, or decremented for the wrong
/// support, those two numbers would part company.
#[test]
fn multiclass_box_accounting_matches_the_model_that_ships() {
    // Three classes off one fitted model. The per-class banks are identical, which is exactly
    // what makes the SUMMED cost checkable: a support's cost must be 3x its per-class boxes,
    // so an accounting that forgot to sum across classes would land at 1x and fail below.
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
            max_order: 4,
            max_depth: 6,
            ..InteractionPolicy::default()
        },
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed: 0,
    };
    let model = Booster::with_config(cfg).fit(&x, &y, &spec).unwrap();
    let serve = ServeBinnedMatrix(x);
    let mc = MultiClassModel {
        classes: vec![model.clone(), model.clone(), model],
        class_labels: vec!["a".into(), "b".into(), "c".into()],
        schema_version: t_boost_core::SCHEMA_VERSION_UNLIFTED,
        cell_refit: None,
    };

    let n = serve.0.n_rows as usize;
    let labels: Vec<u32> = (0..n).map(|i| (i % 3) as u32).collect();
    let w = vec![1.0_f32; n];
    let sel: Vec<usize> = (0..n).collect();

    let (deployed, report) = prune_multiclass_to_tables(
        &mc,
        &serve,
        &labels,
        &w,
        RefMeasure::Uniform,
        &sel,
        2,
        &PruneConfig {
            se_rule: 0.0,
            lambda_boxes: 0.0,
            lambda_tables: 0.0,
            table_price_min_arity: 3,
        },
    )
    .unwrap();

    // The full bank's cost is the per-class total times the class count.
    let full_from_model: usize = mc
        .classes
        .iter()
        .map(|m| {
            m.explain(&serve, RefMeasure::Uniform)
                .unwrap()
                .factored
                .iter()
                .map(|ft| ft.n_boxes())
                .sum::<usize>()
        })
        .sum();
    assert!(
        full_from_model > 0,
        "the fixture must deploy boxes, or this test cannot detect a mis-count"
    );
    assert_eq!(
        report.path[0].n_boxes as usize, full_from_model,
        "path point 0 must be the SUMMED-over-classes box total of the full bank"
    );

    // Monotone non-increasing along the backward walk.
    for pair in report.path.windows(2) {
        assert!(
            pair[1].n_boxes <= pair[0].n_boxes,
            "multiclass box count must not grow along the walk: {} -> {}",
            pair[0].n_boxes,
            pair[1].n_boxes
        );
    }

    // GROUND TRUTH: the selected waypoint's reported count is what the shipped model carries.
    let shipped: usize = deployed
        .classes
        .iter()
        .map(|c| c.bank.factored.iter().map(|ft| ft.n_boxes()).sum::<usize>())
        .sum();
    let selected = report
        .path
        .iter()
        .find(|p| p.n_tables as usize == report.kept.len())
        .expect("the selected keep-set must correspond to a waypoint on the path");
    assert_eq!(
        selected.n_boxes as usize, shipped,
        "the reported n_boxes at the selected waypoint ({}) disagrees with the boxes the \
         returned model actually deploys ({shipped}) — box_cost is mis-indexed against ids",
        selected.n_boxes
    );
}
