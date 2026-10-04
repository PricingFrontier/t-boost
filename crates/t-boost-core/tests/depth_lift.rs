//! The depth-lift gate (P-D1): trees deeper than 3 levels over `<= 3` DISTINCT raw
//! features.
//!
//! Interaction order is the count of distinct raw features, not the count of split
//! levels, so a depth-6 tree on 3 features is still exactly a `<=3rd`-order fANOVA
//! function (spec §01's theorem; §00's I1: "Reusing a raw feature is a valid lower-order
//! refinement, not an I1 violation"). Every test here exists to hold that claim to
//! account on REAL fits, not on hand-built trees:
//!
//! * the lift actually deepens trees, and `distinct_raws <= min(depth, 3)` at every depth
//! * all five I2 gates stay green under `ExactnessMode::Exact` at depth 4/5/6
//! * a lifted fit is byte-identical across thread counts
//! * the wire format round-trips, an UNLIFTED model still stamps the pre-lift version,
//!   and a lifted model stamped as unlifted fails closed
//! * monotonicity survives multi-level reuse (nested-threshold containment)
//! * the `(6,0,0)` single-feature tree — a 7-step main effect in one boosting round
//! * `ridge_refit_l2` at depth > 3 errors LOUDLY rather than allocating a 64x matrix
//! * path-A scoring through the `Arena` bank equals the canonical tree walk

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use proptest::prelude::*;
use t_boost_core::engine::{LEGACY_MAX_DEPTH, LEGACY_MAX_ORDER, MAX_DEPTH, ORDER_LIFT_MAX_DEPTH};
use t_boost_core::{
    assert_exact_decomposition, bin_columns, check_feature_budget, BinConfig, Booster, Config,
    CredibilityFloor, FitSpec, InteractionPolicy, Model, MonotoneMap, PbError, RefMeasure,
    ScoringBank, ServeBinnedMatrix, SquaredError, SCHEMA_VERSION_DEPTH_LIFTED,
    SCHEMA_VERSION_HIGH_ORDER, SCHEMA_VERSION_UNLIFTED,
};

/// A deterministic xorshift so the fixtures need no rng dependency.
fn xs(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn base_config(n_trees: u32) -> Config {
    Config {
        lambda_scale_invariant: false,
        n_trees,
        learning_rate: 0.3,
        lambda: 1.0,
        min_split_gain: 0.0,
        max_delta_step: None,
        max_delta_step_gated: Default::default(),
        sampling: Default::default(),
        hist_precision: Default::default(),
        l1_leaf: 0.0,
        colsample_bytree: 1.0,
        learning_rate_decay: 0.0,
        validation_fraction: None,
        early_stopping_rounds: 50,
        early_stopping_adaptive: None,
        early_stopping_min_delta: 0.0,
        interaction_gain_hurdle: 0.0,
        interaction_gain_hurdle_mode: t_boost_core::InteractionGainHurdleMode::Fixed,
        // NOTE: the Rust `Config::default()` has `leaf_refine_steps: 0`, but the PYTHON
        // estimator defaults it to 4 — so a Rust-only depth test that leaves this at 0 never
        // touches the leaf-refine layer that every real `TBoostRegressor(max_depth=4)` fit
        // goes through. `refine_and_smooth_paths_survive_every_depth` below turns it (and
        // path_smooth, and incremental_mu) ON precisely to close that gap.
        leaf_refine_steps: 0,
        leaf_refine_backtracks: 4,
        refine_closed_form_tier2: false,
        incremental_mu: false,
        boosters: Default::default(),
        fit_control: Default::default(),
    }
}

/// The PRODUCT default configuration, not the library default: leaf refinement on (the
/// Python estimator's `leaf_refine_steps = 4`), plus `path_smooth` and `incremental_mu`,
/// which pull in the credibility tally, the closed-form deviance layer, and the per-leaf
/// multiplier table. Every one of those carries per-leaf accumulators.
fn product_config(n_trees: u32, incremental_mu: bool) -> Config {
    Config {
        leaf_refine_steps: 4,
        refine_closed_form_tier2: true,
        incremental_mu,
        ..base_config(n_trees)
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

/// A smooth, interaction-bearing target on four features. Smooth main effects are what
/// reward *resolution* (many thresholds on one feature), which is exactly what the lift
/// is for — a piecewise-constant fixture would be saturated at depth 3 and prove nothing.
fn dataset(n: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut s = seed | 1;
    let mut cols: Vec<Vec<f32>> = (0..4).map(|_| Vec::with_capacity(n)).collect();
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let a = xs(&mut s) * 4.0 - 2.0;
        let b = xs(&mut s) * 4.0 - 2.0;
        let c = xs(&mut s) * 4.0 - 2.0;
        let d = xs(&mut s) * 4.0 - 2.0;
        cols[0].push(a as f32);
        cols[1].push(b as f32);
        cols[2].push(c as f32);
        cols[3].push(d as f32);
        let target = 1.3 * a.tanh() - 0.8 * (1.7 * b).sin() + 0.5 * a * c + 0.25 * d * d;
        y.push((target + 0.15 * (xs(&mut s) - 0.5)) as f32);
    }
    (cols, y)
}

fn binned(cols: &[Vec<f32>]) -> t_boost_core::BinnedMatrix {
    let refs: Vec<&[f32]> = cols.iter().map(Vec::as_slice).collect();
    bin_columns(&refs, None, &BinConfig::default(), 0).unwrap()
}

fn fit_at(max_depth: u8, max_order: u8, n: usize, n_trees: u32) -> (Model, ServeBinnedMatrix) {
    let (cols, y) = dataset(n, 0x5eed);
    let x = binned(&cols);
    let model = Booster::with_config(base_config(n_trees))
        .fit(&x, &y, &spec_with(max_depth, max_order))
        .unwrap();
    (model, ServeBinnedMatrix(x))
}

/// The distinct raw-feature count of a tree, via the model's own provenance.
fn distinct_raws(model: &Model, tree: &t_boost_core::engine::ObliviousTree) -> usize {
    let mut seen: Vec<u32> = Vec::new();
    for split in &tree.splits {
        let raw = model.provenance[split.axis as usize].raw.0;
        if !seen.contains(&raw) {
            seen.push(raw);
        }
    }
    seen.len()
}

// ---------------------------------------------------------------------------------
// 1. The lift does what it says: deeper trees, unchanged interaction order.
// ---------------------------------------------------------------------------------

#[test]
fn deeper_trees_are_grown_and_never_raise_interaction_order() {
    for max_depth in LEGACY_MAX_DEPTH..=MAX_DEPTH {
        let (model, _) = fit_at(max_depth as u8, 3, 3000, 40);
        let mut deepest = 0usize;
        for (_, tree) in &model.trees {
            let depth = usize::from(tree.depth);
            assert_eq!(depth, tree.splits.len(), "depth must equal splits.len()");
            assert!(
                (1..=max_depth).contains(&depth),
                "depth {depth} outside 1..={max_depth}"
            );
            // THE claim: order is distinct raw features, and the cap never moves.
            let distinct = distinct_raws(&model, tree);
            assert!(
                distinct <= depth.min(3),
                "distinct raws {distinct} exceeds min(depth {depth}, 3)"
            );
            assert!(
                tree.leaves.len() >= 1usize << depth,
                "leaf table shorter than 2^depth"
            );
            deepest = deepest.max(depth);
        }
        check_feature_budget(&model).unwrap();
        assert_eq!(
            deepest, max_depth,
            "max_depth={max_depth} must actually be reached on a smooth target"
        );
        // The lift is per-model on the wire: only a genuinely lifted fit costs a bump, and
        // the bump is the rung that FIRST admitted the shape. Three rungs are reachable from
        // these order-<=3 fixtures (the order lift's v4 is a strictly stronger claim they
        // never make):
        //
        //   depth <= 3  -> v2, byte-for-byte what a pre-lift build wrote
        //   depth <= 6  -> v3, the depth lift
        //   depth 7..8  -> v5, the high-order lift, which widened MAX_DEPTH past 6
        //
        // That last rung is why this assertion is written as a ladder rather than a boolean:
        // when `MAX_DEPTH` went 6 -> 8 the two-armed version silently claimed depth-8 content
        // was v3-readable, which an order-lift-era reader would have accepted and then failed
        // to validate. See `serialize::SCHEMA_VERSION`.
        let want = if max_depth > ORDER_LIFT_MAX_DEPTH {
            SCHEMA_VERSION_HIGH_ORDER
        } else if max_depth > LEGACY_MAX_DEPTH {
            SCHEMA_VERSION_DEPTH_LIFTED
        } else {
            SCHEMA_VERSION_UNLIFTED
        };
        assert_eq!(
            model.required_schema_version(),
            want,
            "max_depth={max_depth} must stamp the version that first admitted it"
        );
    }
}

/// A depth-6 tree restricted to ONE raw feature: the `(6,0,0)` row of the resolution
/// table — a 7-cell main effect fit in a single boosting round, where depth 3 gets 4.
#[test]
fn single_feature_depth_six_tree_is_an_order_one_refinement() {
    let (model, x) = fit_at(6, 1, 3000, 30);
    let mut deepest = 0usize;
    for (_, tree) in &model.trees {
        assert_eq!(
            distinct_raws(&model, tree),
            1,
            "max_order=1 must keep every tree order-1 however deep it grows"
        );
        deepest = deepest.max(usize::from(tree.depth));
    }
    assert!(
        deepest >= 4,
        "expected genuine order-1 refinement, got depth {deepest}"
    );
    let bank = model.explain(&x, RefMeasure::default()).unwrap();
    for table in &bank.tables {
        assert_eq!(
            table.u.order(),
            1,
            "an order-1 fit must export only main effects"
        );
    }
    assert_exact_decomposition(&model, &bank, &x).unwrap();
}

/// **The regression test for the widest class of depth bug there is.**
///
/// Every per-leaf accumulator outside the grower — the leaf-refine line search, the
/// closed-form Poisson/Gamma/Tweedie deviance layer, the `path_smooth` credibility tally,
/// and the `incremental_mu` multiplier table — was a fixed `[_; 8]` indexed by a membership
/// leaf id. Those ids now run to `2^depth - 1`, so at depth 4+ they either error out on the
/// first row that lands in leaf 8 or (worse) silently drop it. `leaf_refine_steps` defaults
/// to **4** in the Python estimator while `Config::default()` uses 0, so a Rust-only depth
/// test that takes the library default never executes any of this. Exercise it explicitly,
/// across the objectives that route through different refine implementations.
#[test]
fn refine_and_smooth_paths_survive_every_depth() {
    let (cols, y_raw) = dataset(2500, 0x9e3);
    let x = binned(&cols);
    // Strictly positive target for the log-link objectives.
    let y_pos: Vec<f32> = y_raw.iter().map(|v| v.abs() + 0.5).collect();

    for max_depth in LEGACY_MAX_DEPTH..=MAX_DEPTH {
        for incremental_mu in [false, true] {
            for (label, loss, y) in [
                (
                    "squared_error",
                    &SquaredError as &dyn t_boost_core::Loss,
                    &y_raw,
                ),
                (
                    "poisson",
                    &t_boost_core::Poisson as &dyn t_boost_core::Loss,
                    &y_pos,
                ),
                (
                    "gamma",
                    &t_boost_core::Gamma as &dyn t_boost_core::Loss,
                    &y_pos,
                ),
            ] {
                let spec = FitSpec {
                    loss,
                    weight: None,
                    exposure: None,
                    monotone: MonotoneMap::new(),
                    interaction: InteractionPolicy {
                        max_order: 3,
                        max_depth: max_depth as u8,
                        ..InteractionPolicy::default()
                    },
                    // path_smooth > 0 turns on the credibility tally AND the parent shrink,
                    // both of which carry per-leaf arrays.
                    credibility: CredibilityFloor {
                        path_smooth: 5.0,
                        ..CredibilityFloor::default()
                    },
                    fixed_holdout: None,
                    bag_groups: None,
                    seed: 0,
                };
                let model = Booster::with_config(product_config(25, incremental_mu))
                    .fit(&x, y, &spec)
                    .unwrap_or_else(|e| {
                        panic!("{label} depth {max_depth} incremental_mu={incremental_mu}: {e}")
                    });
                let deepest = model
                    .trees
                    .iter()
                    .map(|(_, t)| usize::from(t.depth))
                    .max()
                    .unwrap_or(0);
                assert_eq!(
                    deepest, max_depth,
                    "{label} depth {max_depth}: the refine path must run on GENUINELY deep \
                     trees, otherwise this test proves nothing"
                );
                model.validate().unwrap();
            }
        }
    }
}

/// Native-softmax multiclass at depth 6. Its refine loop is the only place in the crate
/// that indexes per-leaf accumulators RAW (`g_acc[k][leaf]`), so an out-of-range membership
/// there is a panic, not a typed error.
#[test]
fn multiclass_softmax_survives_the_lift() {
    let (cols, y_raw) = dataset(2000, 0x11c);
    let x = binned(&cols);
    // Three classes from the continuous target's terciles.
    let mut sorted = y_raw.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q1 = sorted[sorted.len() / 3];
    let q2 = sorted[2 * sorted.len() / 3];
    let labels: Vec<f32> = y_raw
        .iter()
        .map(|v| {
            if *v < q1 {
                0.0
            } else if *v < q2 {
                1.0
            } else {
                2.0
            }
        })
        .collect();

    for max_depth in [LEGACY_MAX_DEPTH, MAX_DEPTH] {
        let spec = FitSpec {
            loss: &SquaredError, // ignored by the softmax path; the class count drives it
            weight: None,
            exposure: None,
            monotone: MonotoneMap::new(),
            interaction: InteractionPolicy {
                max_order: 3,
                max_depth: max_depth as u8,
                ..InteractionPolicy::default()
            },
            credibility: CredibilityFloor {
                path_smooth: 5.0,
                ..CredibilityFloor::default()
            },
            fixed_holdout: None,
            bag_groups: None,
            seed: 0,
        };
        let model = Booster::with_config(product_config(20, false))
            .fit_multiclass(&x, &labels, 3, &["a".into(), "b".into(), "c".into()], &spec)
            .unwrap_or_else(|e| panic!("multiclass depth {max_depth}: {e}"));
        model.validate().unwrap();
        for class in &model.classes {
            for (_, tree) in &class.trees {
                assert!(usize::from(tree.depth) <= max_depth);
                assert_eq!(tree.leaves.len().max(8), tree.leaves.len());
            }
        }
    }
}

/// The leaf-slot count frames the bincode stream (no length prefix), so a short leaf vector
/// would emit a document that decodes every FOLLOWING tree from the wrong offset — silently.
/// Both construction gates must reject it.
#[test]
fn a_mis_sized_leaf_table_is_rejected_at_both_gates() {
    use t_boost_core::data::{AxisKind, AxisProvenance, FeatureId};
    use t_boost_core::engine::{ObliviousTree, Split};

    let provenance = vec![AxisProvenance {
        raw: FeatureId(0),
        kind: AxisKind::Numeric,
    }];
    let splits = vec![
        Split {
            axis: 0,
            bin_le: 1,
            missing_left: false,
        },
        Split {
            axis: 0,
            bin_le: 2,
            missing_left: false,
        },
        Split {
            axis: 0,
            bin_le: 3,
            missing_left: false,
        },
        Split {
            axis: 0,
            bin_le: 4,
            missing_left: false,
        },
    ];
    // depth 4 needs leaf_slots(4) == 16 slots; 8 is the pre-lift width.
    assert!(matches!(
        ObliviousTree::try_new(splits.clone(), vec![0.0; 8], &provenance),
        Err(PbError::ShapeMismatch { .. })
    ));
    assert!(ObliviousTree::try_new(splits.clone(), vec![0.0; 16], &provenance).is_ok());
    // And a depth-2 tree must keep the LEGACY 8, not 4 — that is what preserves the bytes.
    assert!(matches!(
        ObliviousTree::try_new(splits[..2].to_vec(), vec![0.0; 4], &provenance),
        Err(PbError::ShapeMismatch { .. })
    ));
    assert!(ObliviousTree::try_new(splits[..2].to_vec(), vec![0.0; 8], &provenance).is_ok());

    // Model::validate is the load gate and must agree with the constructor.
    let (mut model, _) = fit_at(4, 3, 800, 5);
    model.trees[0].1.leaves.truncate(8);
    assert!(matches!(
        model.validate(),
        Err(PbError::ShapeMismatch { .. })
    ));
}

/// **The credibility floor must not be vetoed by PHANTOM cells.**
///
/// The §07.3 floor rejects a candidate if ANY leaf's child falls under it — a symmetric
/// whole-level guarantee that is exactly right at depth 3, where every leaf is a real
/// cell. A LIFTED tree reuses a feature across levels, which makes a large fraction of its
/// `2^depth` leaf array structurally unreachable (a depth-6 `(2,2,2)` tree realizes 27 of
/// 64). Those phantoms hold zero rows, so before the empty-leaf exemption ANY non-zero
/// floor vetoed EVERY candidate at the first level that created one — and a lifted fit
/// terminated EARLIER than the depth-3 fit it was meant to refine. Measured, and strictly
/// worse. Pin the property: with a floor active, a deeper cap must still grow deeper trees.
#[test]
fn a_credibility_floor_does_not_veto_the_lift_through_phantom_cells() {
    let (cols, y) = dataset(4000, 0xf100);
    let x = binned(&cols);
    let mut deepest_by_cap = Vec::new();
    for max_depth in [LEGACY_MAX_DEPTH, 4, MAX_DEPTH] {
        let spec = FitSpec {
            loss: &SquaredError,
            weight: None,
            exposure: None,
            monotone: MonotoneMap::new(),
            interaction: InteractionPolicy {
                max_order: 3,
                max_depth: max_depth as u8,
                ..InteractionPolicy::default()
            },
            credibility: CredibilityFloor {
                // Deliberately non-trivial, and well below what 4000 rows can support.
                min_data_in_leaf: 12,
                ..CredibilityFloor::default()
            },
            fixed_holdout: None,
            bag_groups: None,
            seed: 0,
        };
        let model = Booster::with_config(base_config(30))
            .fit(&x, &y, &spec)
            .unwrap();
        let deepest = model
            .trees
            .iter()
            .map(|(_, t)| usize::from(t.depth))
            .max()
            .unwrap_or(0);
        deepest_by_cap.push((max_depth, deepest));
    }
    for (cap, deepest) in &deepest_by_cap {
        assert_eq!(
            deepest, cap,
            "with min_data_in_leaf=8, max_depth={cap} still reached only depth {deepest}; \
             the floor is being tripped by empty phantom cells ({deepest_by_cap:?})"
        );
    }
}

/// **A bagged lifted fit must not reject its own bags.**
///
/// The schema stamp is a per-model MINIMUM reader requirement derived from that model's
/// own contents (see `serialize::SCHEMA_VERSION`), so two bags of ONE lifted fit carry
/// different stamps exactly when one of them happened to terminate at depth <= 3 — which
/// is the common case on a small portfolio. `validate_soup_member` compared the stamps for
/// equality and the whole fit died with "model soup members must share grids, provenance,
/// link, objective, and schema version". Found on `brautocoll` (1.3k rows), where it killed
/// 30/30 splits; bagging is ON by default in the product recipe, so this made the lift
/// unusable on every small dataset.
#[test]
fn a_bagged_lifted_fit_tolerates_bags_of_differing_depth() {
    let (cols, y) = dataset(900, 0xba65);
    let x = binned(&cols);
    let mut config = base_config(40);
    config.boosters.ensemble = t_boost_core::EnsembleSpec::OuterBag {
        n_bags: 8,
        bag_subsample: 0.7,
        cell_refit: Default::default(),
    };
    let model = Booster::with_config(config)
        .fit(&x, &y, &spec_with(6, 3))
        .expect("a bagged depth-6 fit must not reject its own bags");
    model.validate().unwrap();

    // Per-bag depth reach. The production failure (brautocoll, 1.3k rows, 30/30 splits)
    // was a bag that stayed within the legacy depth sitting next to one that did not; this
    // fixture does not reliably reproduce that mix, so the assertion below is the fix's
    // actual CONTRACT rather than the reproduction: the soup's stamp is re-derived from
    // the MERGED tree list, never inherited from the first bag.
    let spans = model
        .bag_spans
        .as_ref()
        .expect("a bagged fit records its bag spans");
    assert!(spans.len() > 1, "fixture must actually bag");

    // The soup takes the MAX requirement over its members, never the first bag's.
    assert_eq!(model.schema_version, model.required_schema_version());
    assert_eq!(model.schema_version, SCHEMA_VERSION_DEPTH_LIFTED);

    // And it round-trips at that stamp.
    let doc = t_boost_core::ModelDoc::new(model.clone());
    let bytes = t_boost_core::encode_doc(&doc).unwrap();
    assert_eq!(t_boost_core::decode_doc(&bytes).unwrap().model, doc.model);
}

// ---------------------------------------------------------------------------------
// 2. The five I2 gates, at every lifted depth, on real fits.
// ---------------------------------------------------------------------------------

#[test]
#[ignore = "slow: run with `cargo test --release -- --ignored`"]
fn all_five_i2_gates_pass_at_every_lifted_depth() {
    for max_depth in LEGACY_MAX_DEPTH..=MAX_DEPTH {
        for max_order in 1..=(LEGACY_MAX_ORDER as u8) {
            let (model, x) = fit_at(max_depth as u8, max_order, 2500, 30);
            assert!(matches!(model.mode, t_boost_core::ExactnessMode::Exact));
            check_feature_budget(&model).unwrap();
            let bank = model.explain(&x, RefMeasure::default()).unwrap();
            // Reconstruction + MassConservation + VarianceSum + ThreeWayEqual + Purity.
            assert_exact_decomposition(&model, &bank, &x).unwrap_or_else(|e| {
                panic!("depth {max_depth} order {max_order}: {e}");
            });
            for table in &bank.tables {
                assert!(
                    table.u.order() <= usize::from(max_order),
                    "depth {max_depth}: exported table order {} exceeds the cap {max_order}",
                    table.u.order()
                );
            }
        }
    }
}

/// **P-D4: a lifted order-3 support stays FACTORED, and stays exact.**
///
/// P-D1 routed a reuse-bearing order-3 support to the dense cube, because a rank-1
/// `FactoredBox` cannot hold a `k0 x k1 x k2` product box. That measured 601,942-cell tables
/// on beMTPL16 (an 18x blow-up) and blew the 32M bank firewall outright on catelematic13 —
/// i.e. it broke the explainability contract and made order-3 x depth unusable. P-D4
/// decomposes a lifted tree into one rank-1 box per realized region tuple instead, so the
/// compact factored form is available at every depth.
#[test]
#[ignore = "slow: run with `cargo test --release -- --ignored`"]
fn a_lifted_order_three_support_stays_factored_and_exact() {
    let (model, x) = fit_at(6, 3, 3000, 60);
    let reuse_bearing = model.trees.iter().any(|(_, t)| {
        usize::from(t.depth) > distinct_raws(&model, t) && distinct_raws(&model, t) == 3
    });
    assert!(
        reuse_bearing,
        "fixture must actually produce an order-3 tree that reuses a feature"
    );

    let bank = model.explain(&x, RefMeasure::default()).unwrap();
    // All five I2 gates on the factored bank — the exactness claim, end to end.
    assert_exact_decomposition(&model, &bank, &x).unwrap();

    // The order-3 mass is FACTORED, not densified: no order-3 dense table exists, and the
    // factored list carries it instead.
    assert!(
        !bank.factored.is_empty(),
        "the lifted order-3 effect must be kept in factored form"
    );
    assert!(
        bank.tables.iter().all(|t| t.u.order() <= 2),
        "no order-3 effect should have been materialized as a dense cube"
    );

    // Every factored effect is order-3 and carries a finite variance (the value the Sobol
    // report and the VarianceSum gate read).
    for ft in &bank.factored {
        assert_eq!(ft.u.order(), 3);
        assert!(ft.variance.is_finite() && ft.variance >= 0.0);
    }
}

// ---------------------------------------------------------------------------------
// 3. Determinism.
// ---------------------------------------------------------------------------------

#[test]
fn depth_six_fit_is_byte_identical_across_thread_counts() {
    let (cols, y) = dataset(3000, 0xd06);
    let x = binned(&cols);
    let mut blobs: Vec<Vec<u8>> = Vec::new();
    for threads in [1usize, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let model = pool.install(|| {
            Booster::with_config(base_config(40))
                .fit(&x, &y, &spec_with(6, 3))
                .unwrap()
        });
        blobs.push(model.to_bincode().unwrap());
    }
    assert_eq!(
        blobs[0], blobs[1],
        "a depth-6 fit must be byte-identical at 1 and 8 threads"
    );
}

// ---------------------------------------------------------------------------------
// 4. The wire format.
// ---------------------------------------------------------------------------------

/// The leaf array is a fixed-length TUPLE with no length prefix, and a depth-`<=3` tree
/// keeps the legacy 8-slot width — so its bytes are exactly what the pre-lift
/// `leaves: [f32; 8]` field wrote. Pinned here by construction rather than by a checked-in
/// blob, so it fails the moment the encoding drifts.
#[test]
fn unlifted_tree_keeps_the_legacy_leaf_encoding() {
    let (model, _) = fit_at(3, 3, 1500, 5);
    for (_, tree) in &model.trees {
        assert_eq!(
            tree.leaves.len(),
            8,
            "a depth-<=3 tree must keep the legacy 8-slot leaf table (zero tail included)"
        );
        for (i, v) in tree.leaves.iter().enumerate() {
            if i >= 1usize << tree.depth {
                assert_eq!(*v, 0.0, "the legacy tail must stay zeroed");
            }
        }
    }
    let doc = t_boost_core::ModelDoc::new(model.clone());
    assert_eq!(
        doc.schema_version, SCHEMA_VERSION_UNLIFTED,
        "an unlifted model must be stamped at the pre-lift version so old readers keep working"
    );
    let bytes = t_boost_core::encode_doc(&doc).unwrap();
    let back = t_boost_core::decode_doc(&bytes).unwrap();
    assert_eq!(back.model, doc.model);
}

#[test]
fn lifted_model_round_trips_bincode_and_json_and_cannot_be_stamped_as_unlifted() {
    let (model, _) = fit_at(6, 3, 3000, 25);
    assert!(model.trees.iter().any(|(_, t)| usize::from(t.depth) > 3));
    for (_, tree) in &model.trees {
        if usize::from(tree.depth) > LEGACY_MAX_DEPTH {
            assert_eq!(tree.leaves.len(), 1usize << tree.depth);
        }
    }

    let doc = t_boost_core::ModelDoc::new(model.clone());
    assert_eq!(doc.schema_version, SCHEMA_VERSION_DEPTH_LIFTED);

    let bytes = t_boost_core::encode_doc(&doc).unwrap();
    assert_eq!(t_boost_core::decode_doc(&bytes).unwrap().model, doc.model);

    let json = t_boost_core::encode_doc_json(&doc).unwrap();
    assert_eq!(
        t_boost_core::decode_doc_json(&json).unwrap().model,
        doc.model
    );

    // Fail closed: a lifted model may not claim the pre-lift version, in either format.
    let mut forged = doc.clone();
    forged.schema_version = SCHEMA_VERSION_UNLIFTED;
    forged.model.schema_version = SCHEMA_VERSION_UNLIFTED;
    let forged_bytes = t_boost_core::encode_doc(&forged).unwrap();
    assert!(matches!(
        t_boost_core::decode_doc(&forged_bytes),
        Err(PbError::Serialization(_))
    ));
}

// ---------------------------------------------------------------------------------
// 5. Monotonicity under multi-level reuse.
// ---------------------------------------------------------------------------------

/// §07.5's reachability argument, exercised at depth 6.
///
/// When one axis is tested at several levels the induced nested thresholds make some leaf
/// patterns LOGICALLY unreachable (`bin<=2` at one level and `bin>5` at another: no finite
/// bin is both). `monotone_reachable` masks those out, and the surviving cousin pairs
/// compose transitively into the correct total order over the refined regions. If that
/// argument failed at depth > 3, the SERVED function would be non-monotone — which is what
/// this checks, end to end, on the actual scored surface.
#[test]
fn monotone_constraint_holds_under_multi_level_reuse() {
    let n = 2500usize;
    let mut seed = 0x9e37_79b9_u64;
    let mut c0 = Vec::with_capacity(n);
    let mut c1 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let a = xs(&mut seed) * 6.0 - 3.0;
        let b = xs(&mut seed) * 6.0 - 3.0;
        c0.push(a as f32);
        c1.push(b as f32);
        // Strictly increasing in feature 0, with curvature so reuse is genuinely rewarded.
        y.push((a.tanh() * 2.0 + 0.3 * a + 0.4 * b) as f32);
    }
    let cols = vec![c0, c1];
    let x = binned(&cols);

    let mut monotone = MonotoneMap::new();
    monotone.insert("f0".to_string(), t_boost_core::MonoSign::Increasing);
    let spec = FitSpec {
        loss: &SquaredError,
        weight: None,
        exposure: None,
        monotone,
        interaction: InteractionPolicy {
            max_order: 2,
            max_depth: 6,
            ..InteractionPolicy::default()
        },
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed: 0,
    };
    let model = Booster::with_config(base_config(60))
        .fit(&x, &y, &spec)
        .unwrap();

    let reused_constrained_axis = model
        .trees
        .iter()
        .any(|(_, t)| t.splits.iter().filter(|s| s.axis == 0).count() >= 2);
    assert!(
        reused_constrained_axis,
        "fixture must actually reuse the CONSTRAINED axis across levels"
    );

    // The served surface must be non-decreasing in feature 0 at every fixed level of
    // feature 1, over the full bin grid.
    let n_bins_0 = model.grids[0].n_bins;
    let n_bins_1 = model.grids[1].n_bins;
    for b1 in 1..n_bins_1 {
        let mut prev = f64::NEG_INFINITY;
        for b0 in 1..n_bins_0 {
            let v = model.ensemble_f64(&[b0 as u8, b1 as u8]).unwrap();
            assert!(
                v >= prev - 1e-6,
                "monotone violated at bin ({b0},{b1}): {v} < {prev}"
            );
            prev = v;
        }
    }
}

// ---------------------------------------------------------------------------------
// 6. Scoring-path equality (the Arena bank).
// ---------------------------------------------------------------------------------

#[test]
fn arena_scoring_bank_matches_the_canonical_tree_walk_at_depth_six() {
    let (model, x) = fit_at(6, 3, 2000, 30);
    let bank = ScoringBank::from_model(&model).unwrap();
    assert!(
        matches!(bank, ScoringBank::Arena { .. }),
        "a lifted model must take the side-arena bank, not the frozen 64-byte row"
    );
    for row in 0..x.0.n_rows as usize {
        let bins: Vec<u8> = x.0.data.iter().map(|c| c[row]).collect();
        let walk = model.score_trees_row(&bins, 0.0).unwrap();
        let packed = bank.score_row(&bins, model.f0).unwrap();
        assert_eq!(
            packed.to_bits(),
            walk.to_bits(),
            "arena scoring must be BIT-equal to the tree walk"
        );
    }
}

/// The default (unlifted) path must still take the frozen one-cache-line layout — the
/// depth knob may not perturb serving for models that do not use it.
#[test]
fn unlifted_model_still_takes_the_packed_bank() {
    let (model, _) = fit_at(3, 3, 1500, 20);
    assert!(matches!(
        ScoringBank::from_model(&model).unwrap(),
        ScoringBank::Packed { .. }
    ));
}

// ---------------------------------------------------------------------------------
// 7. Hard blocks.
// ---------------------------------------------------------------------------------

#[test]
fn ridge_refit_is_rejected_above_the_legacy_depth() {
    let (cols, y) = dataset(600, 7);
    let x = binned(&cols);
    let mut config = base_config(10);
    config.boosters.refit_leaves = t_boost_core::RefitSpec::Ridge {
        l2: 1.0,
        max_iter: 4,
        every_k_trees: Some(4),
    };
    let err = Booster::with_config(config)
        .fit(&x, &y, &spec_with(6, 3))
        .unwrap_err();
    match err {
        PbError::InvalidConfig { what } => {
            assert!(
                what.contains("ridge_refit_l2"),
                "unexpected message: {what}"
            );
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
fn out_of_range_max_depth_is_rejected() {
    let (cols, y) = dataset(400, 11);
    let x = binned(&cols);
    for bad in [0u8, 1, 2, (MAX_DEPTH + 1) as u8] {
        assert!(matches!(
            Booster::with_config(base_config(5)).fit(&x, &y, &spec_with(bad, 3)),
            Err(PbError::InvalidConfig { .. })
        ));
    }
}

// ---------------------------------------------------------------------------------
// 8. Proptest — I1 over randomized fits at every depth.
// ---------------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]

    /// spec §07.207's invariant, with the outer bound moved:
    /// `1 <= distinct(raw(splits)) <= depth <= max_depth`.
    #[test]
    fn i1_holds_on_randomized_fits_at_every_depth(
        max_depth in (LEGACY_MAX_DEPTH as u8)..=(MAX_DEPTH as u8),
        max_order in 1u8..=(LEGACY_MAX_ORDER as u8),
        seed in 1u64..10_000,
        n_trees in 5u32..25,
    ) {
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
    }
}
