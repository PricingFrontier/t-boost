//! The ORDER lift (`max_interaction_order` 1..=4) — exactness, structure and cost.
//!
//! Companion to `depth_lift.rs`, which moved the DEPTH cap. The two are orthogonal: depth is
//! how finely a tree resolves its features, order is how many distinct features it couples
//! (= how many axes an exported table has). This file's contract is the one that matters for
//! the product: **an order-4 fit is still EXACTLY decomposable**, its 4-way effects are
//! compact and evidence-backed, and nothing about an order-<=3 fit moved.
//!
//! Why order 4 is FACTORED and never dense (`a_four_way_effect_is_factored_and_small`): a
//! support's dense cube is the product of its raws' GLOBAL merged extents — every border any
//! tree placed on those features, including main-effect trees that legitimately want full
//! resolution. An order-4 support therefore pays for resolution it never asked for. But an
//! order-4 tree realizes only `k_0·k_1·k_2·k_3` REGIONS, so the effect is low-RANK, not
//! sparse: it is small in the factored representation and in no other.

#![allow(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_range_loop
)]

use proptest::prelude::*;
use t_boost_core::engine::{
    LEGACY_MAX_DEPTH, LEGACY_MAX_ORDER, MAX_ORDER, ORDER_LIFT_MAX_DEPTH, ORDER_LIFT_MAX_ORDER,
};
use t_boost_core::{
    assert_exact_decomposition, bin_columns, check_feature_budget, BinConfig, Booster, Config,
    CredibilityFloor, FitSpec, InteractionPolicy, Model, MonotoneMap, PbError, RefMeasure,
    ServeBinnedMatrix, SquaredError,
};

fn xs(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// The engine's `Config::default` is a NEUTRAL primitive — hurdle 0.0, Fixed mode. The
/// product default lives in `sklearn.py` (`_GAIN_HURDLE_DEFAULT = 2.0`, adaptive), and the
/// hurdle is the growth-side half of "a 4-way only where there is genuine signal", so a test
/// that means to exercise the product must say so. This helper is the PRODUCT recipe.
fn base_config(n_trees: u32) -> Config {
    Config {
        n_trees,
        learning_rate: 0.3,
        lambda: 1.0,
        interaction_gain_hurdle: 2.0,
        interaction_gain_hurdle_mode: t_boost_core::engine::InteractionGainHurdleMode::Adaptive,
        ..Config::default()
    }
}

/// The neutral engine default — no admission hurdle at all. Used only where a test needs to
/// see what the grower would do UNPRICED.
fn unpriced_config(n_trees: u32) -> Config {
    Config {
        n_trees,
        learning_rate: 0.3,
        lambda: 1.0,
        ..Config::default()
    }
}

fn spec_with(max_depth: u8, max_order: u8) -> FitSpec<'static> {
    FitSpec {
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
    }
}

/// Five features carrying a genuine 4-way term that NO lower-order structure explains: the
/// sign product of four of them. A greedy booster can only fit it by putting all four on one
/// tree, so a fixture without it would prove nothing about order 4 (the order-3 arm would
/// simply match). The fifth feature is a decoy — it enters only through a main effect, so
/// admitting it into a 4-way would be exactly the spurious behaviour the hurdle must stop.
fn dataset(n: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut s = seed | 1;
    let mut cols: Vec<Vec<f32>> = (0..5).map(|_| Vec::with_capacity(n)).collect();
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let v: Vec<f64> = (0..5).map(|_| xs(&mut s) * 4.0 - 2.0).collect();
        for (col, &x) in cols.iter_mut().zip(v.iter()) {
            col.push(x as f32);
        }
        let quad = v[0].signum() * v[1].signum() * v[2].signum() * v[3].signum();
        let target = 0.9 * v[0].tanh() - 0.6 * (1.5 * v[1]).sin() + 0.4 * v[4] + 2.2 * quad;
        y.push((target + 0.15 * (xs(&mut s) - 0.5)) as f32);
    }
    (cols, y)
}

fn binned(cols: &[Vec<f32>]) -> t_boost_core::BinnedMatrix {
    let refs: Vec<&[f32]> = cols.iter().map(Vec::as_slice).collect();
    bin_columns(&refs, None, &BinConfig::default(), 0).unwrap()
}

fn fit_at(max_depth: u8, max_order: u8, n: usize, n_trees: u32) -> (Model, ServeBinnedMatrix) {
    let (cols, y) = dataset(n, 0x0_4de5);
    let x = binned(&cols);
    let model = Booster::with_config(base_config(n_trees))
        .fit(&x, &y, &spec_with(max_depth, max_order))
        .unwrap();
    (model, ServeBinnedMatrix(x))
}

fn distinct_raws(model: &Model, tree: &t_boost_core::ObliviousTree) -> usize {
    let mut seen: Vec<u32> = Vec::new();
    for s in &tree.splits {
        let raw = model.provenance[s.axis as usize].raw.0;
        if !seen.contains(&raw) {
            seen.push(raw);
        }
    }
    seen.len()
}

// ---------------------------------------------------------------------------------
// 1. The contract: exactness at order 4.
// ---------------------------------------------------------------------------------

/// **THE non-negotiable.** All five I2 gates (Reconstruction, MassConservation, VarianceSum,
/// ThreeWayEqual, Purity) on real order-4 fits at `ExactnessMode::Exact`, across the whole
/// `(depth, order)` rectangle THIS LIFT ADMITTED. A fit is only admitted at `order` when
/// `depth >= order` — a tree needs one level per distinct feature.
///
/// Bounded at `ORDER_LIFT_MAX_*` rather than at `MAX_*`, and deliberately so. When the
/// high-order lift took the caps to 8/8 this loop silently went from 15 fits to 33, each of
/// the new ones an expensive depth-7/8 `explain`, and it became one of the slowest tests in
/// the suite — while proving nothing the order lift is responsible for. Orders 5-8 are
/// `tests/order_hi.rs`'s contract, and it tests them on a fixture that actually PLANTS the
/// order in question, which this one does not (its target is 4-way, so at order 5+ this loop
/// would only have shown the gates holding on a model that never used the extra order).
/// Each lift pays for its own rectangle.
#[test]
fn all_five_i2_gates_pass_at_every_lifted_order() {
    for max_depth in LEGACY_MAX_DEPTH..=ORDER_LIFT_MAX_DEPTH {
        for max_order in 1..=ORDER_LIFT_MAX_ORDER.min(max_depth) {
            let (model, x) = fit_at(max_depth as u8, max_order as u8, 1500, 25);
            assert!(matches!(model.mode, t_boost_core::ExactnessMode::Exact));
            check_feature_budget(&model).unwrap();
            let bank = model.explain(&x, RefMeasure::default()).unwrap();
            assert_exact_decomposition(&model, &bank, &x)
                .unwrap_or_else(|e| panic!("depth {max_depth} order {max_order}: {e}"));
            for t in &bank.tables {
                assert!(
                    t.u.order() <= max_order,
                    "depth {max_depth}: dense table order {} exceeds the cap {max_order}",
                    t.u.order()
                );
            }
            for ft in &bank.factored {
                assert!(
                    ft.u.order() <= max_order,
                    "depth {max_depth}: factored effect order {} exceeds the cap {max_order}",
                    ft.u.order()
                );
                assert!(ft.variance.is_finite() && ft.variance >= 0.0);
            }
        }
    }
}

/// A realized 4-way effect is kept in the COMPACT factored form, never densified — and the
/// bank is still exact. This is the representation claim stated as a test: if a 4-way ever
/// reached the dense path it would be the product of four global merged extents, which is
/// the multi-million-cell blow-up the factored form exists to avoid.
#[test]
fn a_four_way_effect_is_factored_and_small() {
    let (model, x) = fit_at(6, 4, 2500, 40);
    let has_order_four = model
        .trees
        .iter()
        .any(|(_, t)| distinct_raws(&model, t) == 4);
    assert!(
        has_order_four,
        "fixture must actually grow an order-4 tree (the 4-way sign product is only \
         reachable by putting all four features on one tree)"
    );

    let bank = model.explain(&x, RefMeasure::default()).unwrap();
    assert_exact_decomposition(&model, &bank, &x).unwrap();

    assert!(
        bank.tables.iter().all(|t| t.u.order() <= 2),
        "no order-3+ effect should ever be materialized as a dense cube"
    );
    let quads: Vec<_> = bank.factored.iter().filter(|f| f.u.order() == 4).collect();
    assert!(
        !quads.is_empty(),
        "the order-4 mass must appear as a factored effect"
    );
    for ft in &quads {
        assert_eq!(ft.axes.len(), 4);
        assert!(ft.variance.is_finite() && ft.variance >= 0.0);
    }
}

/// The order-4 shed lands one order down as BOXES, not as a dense order-3 cube. Stated
/// through its observable consequence: an order-3 support that no tree realized can still
/// appear as a factored effect, because a 4-way shed created it.
#[test]
fn the_order_four_shed_creates_factored_not_dense_sub_effects() {
    let (model, x) = fit_at(6, 4, 2500, 40);
    let bank = model.explain(&x, RefMeasure::default()).unwrap();
    let quad = bank
        .factored
        .iter()
        .find(|f| f.u.order() == 4)
        .expect("fixture must realize an order-4 effect");

    // Every 3-subset of a realized 4-way support must exist as a FACTORED effect (the shed
    // target), and must NOT exist as a dense table.
    for skip in 0..4usize {
        let sub: Vec<_> = quad
            .u
            .0
            .iter()
            .enumerate()
            .filter_map(|(i, f)| (i != skip).then_some(*f))
            .collect();
        assert!(
            bank.factored
                .iter()
                .any(|f| f.u.0.as_slice() == sub.as_slice()),
            "3-subset {sub:?} of the 4-way support must exist as a factored effect"
        );
        assert!(
            !bank
                .tables
                .iter()
                .any(|t| t.u.0.as_slice() == sub.as_slice()),
            "3-subset {sub:?} must not have been densified"
        );
    }
}

/// The FILING artifact: a 4-way effect must survive `to_rating_export` with its arity
/// intact, because that JSON is what a deployment (or a regulator) actually reads.
///
/// `FactoredBoxExport`'s `thresholds`/`missing_left`/`categorical_low_cells`/`octants` were
/// fixed-size `[_; 3]`/`[f64; 8]` arrays; they are now length-carrying, and this is the test
/// that says what those lengths must be. Measured on the fixture: a 4-way effect exports as
/// ~11 boxes of 4 thresholds and 16 corners each — i.e. a readable stack of case expressions,
/// which is the whole reason order 4 is factored rather than dense.
#[test]
fn a_four_way_effect_survives_the_rating_export_with_its_arity_intact() {
    let (model, x) = fit_at(4, 4, 2500, 40);
    let bank = model.explain(&x, RefMeasure::default()).unwrap();
    let export = bank
        .to_rating_export(
            model.link,
            &model.mode,
            &model.schema,
            &model.provenance,
            &model.schema.cat_encoders,
            None,
        )
        .unwrap();
    let quads: Vec<_> = export
        .factored
        .iter()
        .filter(|f| f.feature_names.len() == 4)
        .collect();
    assert!(
        !quads.is_empty(),
        "the fixture's 4-way effect must reach the rating export"
    );
    for f in &quads {
        assert_eq!(f.feature_set.order(), 4);
        assert!(!f.boxes.is_empty());
        for b in &f.boxes {
            assert_eq!(b.thresholds.len(), 4, "one threshold per support axis");
            assert_eq!(b.missing_left.len(), 4);
            assert_eq!(b.categorical_low_cells.len(), 4);
            assert_eq!(b.octants.len(), 16, "2^4 corners");
            assert!(b.octants.iter().all(|v| v.is_finite()));
        }
    }
}

// ---------------------------------------------------------------------------------
// 2. The product bar: 4-way only where there is genuine signal.
// ---------------------------------------------------------------------------------

/// On data whose highest true interaction is a PAIR, raising the cap to 4 must leave order-4
/// growth a rare exception rather than the norm — the growth-side half of "a 4-way effect
/// only where there is genuine signal".
///
/// **This is deliberately a RELATIVE bar, not `== 0`.** Two reasons, both measured. First,
/// growth is greedy and gain-driven: on a finite sample a spurious 4-way sometimes does clear
/// the (doubled) hurdle, and demanding zero here would be asserting something about sampling
/// noise, not about the mechanism. Second, growth is not where the absolute bar lives — the
/// av37 evidence-gated prune is, and it drops any table whose held-out per-fold gains cannot
/// clear `prune_drop_z`, with a heredity cascade that additionally requires all four of a
/// 4-way's 3-way subsets to have survived on their own evidence. That filter runs above this
/// crate, so the "~zero order-4 TABLES on an order-3-sufficient dataset" claim is measured in
/// the arena battery, not here.
///
/// Measured on this fixture at n=3000/60 trees: unpriced growth puts order 4 on essentially
/// every tree; the product hurdle leaves it on ~12%, and on the genuinely-4-way fixture the
/// same settings admit several times that. The ratio is the mechanism working.
#[test]
fn a_pairwise_only_target_admits_far_less_order_four_than_a_four_way_one() {
    let mut s = 0x9e37u64;
    let n = 3000;
    let mut cols: Vec<Vec<f32>> = (0..5).map(|_| Vec::with_capacity(n)).collect();
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let v: Vec<f64> = (0..5).map(|_| xs(&mut s) * 4.0 - 2.0).collect();
        for (col, &x) in cols.iter_mut().zip(v.iter()) {
            col.push(x as f32);
        }
        let target = 1.2 * v[0].tanh() - 0.7 * (1.5 * v[1]).sin() + 0.9 * v[0] * v[2] + 0.3 * v[3];
        y.push((target + 0.15 * (xs(&mut s) - 0.5)) as f32);
    }
    let n_trees = 60;
    let quad_share = |cols: &[Vec<f32>], y: &[f32]| -> f64 {
        let x = binned(cols);
        let model = Booster::with_config(base_config(n_trees))
            .fit(&x, y, &spec_with(6, 4))
            .unwrap();
        let n_quad = model
            .trees
            .iter()
            .filter(|(_, t)| distinct_raws(&model, t) == 4)
            .count();
        n_quad as f64 / model.trees.len().max(1) as f64
    };
    let pairwise = quad_share(&cols, &y);
    let (qcols, qy) = dataset(n, 0x0_4de5);
    let genuine = quad_share(&qcols, &qy);

    assert!(
        pairwise < 0.25,
        "a pairwise-only target put order 4 on {:.0}% of trees; the doubled 3->4 hurdle is \
         supposed to keep that a rare exception",
        pairwise * 100.0
    );
    assert!(
        genuine > 2.0 * pairwise,
        "a genuine 4-way target ({:.0}%) must admit order 4 far more readily than a \
         pairwise-only one ({:.0}%), or the hurdle is not discriminating — it is just a tax",
        genuine * 100.0,
        pairwise * 100.0
    );
}

/// The 3->4 admission hurdle is charged at TWICE the 2->3 rate. Observable form: with the
/// hurdle off, the 4-way fixture grows order-4 trees readily; with it at the product default
/// it grows strictly fewer.
#[test]
fn the_gain_hurdle_prices_the_fourth_feature_above_the_third() {
    let (cols, y) = dataset(3000, 0x0_4de5);
    let x = binned(&cols);
    let count_at = |hurdle: f32| -> usize {
        let mut cfg = base_config(60);
        cfg.interaction_gain_hurdle = hurdle;
        let model = Booster::with_config(cfg)
            .fit(&x, &y, &spec_with(6, 4))
            .unwrap();
        model
            .trees
            .iter()
            .filter(|(_, t)| distinct_raws(&model, t) == 4)
            .count()
    };
    assert_eq!(
        unpriced_config(1).interaction_gain_hurdle,
        0.0,
        "the neutral engine default must stay neutral"
    );
    let free = count_at(0.0);
    let priced = count_at(2.0);
    // Measured on this fixture at n=3000: unpriced admits order-4 on essentially every
    // tree; the product default admits a minority. The pairwise-only companion test
    // (`a_pairwise_only_target_admits_no_four_way_effect`) is the other half — there the
    // product default admits NONE.
    assert!(
        free > 0,
        "the fixture must grow order-4 trees when unpriced"
    );
    assert!(
        priced < free,
        "pricing the 4th feature must admit strictly fewer order-4 trees \
         (unpriced {free}, priced {priced})"
    );
}

/// `table_budget_order_shrink` is EXACTLY inert below order 4: the exponent is
/// `order - LEGACY_MAX_ORDER`, which is 0 for every support an order-<=3 fit can realize.
#[test]
fn the_order_shrink_is_inert_at_order_three() {
    let (cols, y) = dataset(1200, 7);
    let x = binned(&cols);
    let fit = |shrink: f32| {
        let mut spec = spec_with(6, 3);
        spec.interaction.table_budget_order_shrink = shrink;
        Booster::with_config(base_config(40))
            .fit(&x, &y, &spec)
            .unwrap()
    };
    let a = fit(1.0);
    let b = fit(8.0);
    assert_eq!(
        t_boost_core::encode_doc(&t_boost_core::ModelDoc::new(a)).unwrap(),
        t_boost_core::encode_doc(&t_boost_core::ModelDoc::new(b)).unwrap(),
        "the order shrink must not touch an order-3 fit"
    );
}

// ---------------------------------------------------------------------------------
// 3. Config validation.
// ---------------------------------------------------------------------------------

#[test]
fn an_order_above_the_structural_cap_is_refused() {
    let (cols, y) = dataset(400, 5);
    let x = binned(&cols);
    let err = Booster::with_config(base_config(5))
        .fit(&x, &y, &spec_with(6, MAX_ORDER as u8 + 1))
        .unwrap_err();
    assert!(matches!(err, PbError::InvalidConfig { .. }), "{err:?}");
}

/// `max_order > max_depth` is an UNREACHABLE request, not a tighter one — a tree needs one
/// level per distinct feature. Refuse it rather than silently behaving as `max_depth`.
#[test]
fn an_order_exceeding_the_depth_is_refused() {
    let (cols, y) = dataset(400, 5);
    let x = binned(&cols);
    let err = Booster::with_config(base_config(5))
        .fit(&x, &y, &spec_with(LEGACY_MAX_DEPTH as u8, 4))
        .unwrap_err();
    match err {
        PbError::InvalidConfig { what } => {
            assert!(what.contains("max_depth"), "{what}");
        }
        other => panic!("{other:?}"),
    }
}

// ---------------------------------------------------------------------------------
// 4. Determinism and I1 at order 4.
// ---------------------------------------------------------------------------------

#[test]
fn an_order_four_fit_is_byte_identical_across_thread_counts() {
    let (cols, y) = dataset(1500, 0xd06);
    let x = binned(&cols);
    let mut blobs: Vec<Vec<u8>> = Vec::new();
    for threads in [1usize, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let model = pool.install(|| {
            Booster::with_config(base_config(40))
                .fit(&x, &y, &spec_with(6, 4))
                .unwrap()
        });
        blobs.push(t_boost_core::encode_doc(&t_boost_core::ModelDoc::new(model)).unwrap());
    }
    assert_eq!(blobs[0], blobs[1], "order-4 fit is thread-count dependent");
}

/// The explained BANK is thread-count stable too — the factored cascade walks supports in
/// `BTreeSet` order and merges boxes in first-appearance order, so the box sequence (and
/// every sum built from it) is fixed.
#[test]
fn an_order_four_bank_is_byte_identical_across_thread_counts() {
    let (model, x) = fit_at(6, 4, 1500, 25);
    let mut docs: Vec<String> = Vec::new();
    for threads in [1usize, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let bank = pool.install(|| model.explain(&x, RefMeasure::default()).unwrap());
        docs.push(serde_json::to_string(&bank).unwrap());
    }
    assert_eq!(
        docs[0], docs[1],
        "order-4 explain is thread-count dependent"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// spec §07.207's invariant with BOTH outer bounds moved:
    /// `1 <= distinct(raw(splits)) <= min(depth, max_order)`, `depth <= max_depth`.
    #[test]
    fn i1_holds_on_randomized_fits_at_every_order(
        max_depth in (LEGACY_MAX_DEPTH as u8)..=(ORDER_LIFT_MAX_DEPTH as u8),
        order_pick in 1u8..=(ORDER_LIFT_MAX_ORDER as u8),
        seed in 1u64..10_000,
        n_trees in 5u32..25,
    ) {
        let max_order = order_pick.min(max_depth);
        let (cols, y) = dataset(900, seed);
        let x = binned(&cols);
        let model = Booster::with_config(base_config(n_trees))
            .fit(&x, &y, &spec_with(max_depth, max_order))
            .unwrap();
        for (_, tree) in &model.trees {
            let depth = usize::from(tree.depth);
            prop_assert_eq!(depth, tree.splits.len());
            prop_assert!(depth >= 1 && depth <= usize::from(max_depth));
            let distinct = distinct_raws(&model, tree);
            prop_assert!(distinct >= 1);
            prop_assert!(distinct <= depth);
            prop_assert!(distinct <= usize::from(max_order));
        }
        prop_assert!(check_feature_budget(&model).is_ok());
        prop_assert!(LEGACY_MAX_ORDER <= MAX_ORDER);
    }
}
