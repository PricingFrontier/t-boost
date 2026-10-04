//! The MULTI-WAY TABLE budget and the selection-time table price (2026-08-27).
//!
//! Ralph moved the explainability bar: *"in terms of resolution, perfectly happy with as many
//! cells within tables to capture the correct resolution, it's just additional multi-way table
//! count that I'm keen on keeping to a minimum."* Cells are free; the count of >=3-way tables
//! is the scarce thing ("100 3-way tables is not explainable").
//!
//! The shipped `prune_box_budget` / `prune_lambda_boxes` price the OTHER quantity — resolution
//! — and bound table count only as a side effect of dropping supports binarily. These two knobs
//! price the bar directly. Both default to off and both must be inert to the bit when off,
//! because every dataset on the board that does not opt in has to stay byte-identical.
//!
//! What each test pins, in the order the properties matter:
//!
//!   1. **Off is inert**, for the budget and for the price, separately and together.
//!   2. **The cap is a cap**: the surviving bank carries at most `max_tables` supports at or
//!      above the floor, at every level, and the count is what the report says it is.
//!   3. **Below the floor is untouchable.** Mains and pairs are what a filing reads; no budget,
//!      however tight, may drop one. This is the property that distinguishes the table budget
//!      from "prune harder", and it is the reason the knob is not just a smaller `keep_budget`.
//!   4. **Resolution is free.** A support that survives keeps every box it had — the budget
//!      never trades cells for tables, which is the whole point of the new bar.
//!   5. **Heredity survives.** A cascade may only remove supports of higher arity than one
//!      already dropped, so it can never push the count back over the cap.
//!   6. **The path carries the arity histogram unconditionally**, so a budget or a price can be
//!      CALIBRATED from an unpenalized report rather than guessed — the same discipline
//!      `n_boxes` follows.

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

use std::collections::BTreeMap;
use t_boost_core::explain::FeatureSet;
use t_boost_core::prune::{
    apply_table_budget, box_variances, n_tables_at_or_above, prune_bank, table_variances,
    HoldoutFold, PruneConfig, PruneReport, DEFAULT_TABLE_MIN_ARITY,
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

fn fs(ids: &[u32]) -> FeatureSet {
    FeatureSet::new(ids)
}

// ---------------------------------------------------------------------------------------
// Unit-level: `apply_table_budget` against hand-built keep-sets, where the expected answer is
// arithmetic rather than a fit's opinion.
// ---------------------------------------------------------------------------------------

/// Mains, pairs and triples with a strictly ordered evidence ladder, so "keeps the best N" has
/// exactly one right answer and a wrong tie-break is visible rather than lucky.
fn ladder() -> (Vec<FeatureSet>, BTreeMap<FeatureSet, f64>) {
    let keep = vec![
        fs(&[0]),
        fs(&[1]),
        fs(&[2]),
        fs(&[0, 1]),
        fs(&[0, 2]),
        fs(&[1, 2]),
        fs(&[0, 1, 2]),
        fs(&[0, 1, 3]),
        fs(&[0, 2, 3]),
        fs(&[1, 2, 3]),
    ];
    let evidence: BTreeMap<FeatureSet, f64> = [
        (fs(&[0, 1, 2]), 4.0),
        (fs(&[0, 1, 3]), 3.0),
        (fs(&[0, 2, 3]), 2.0),
        (fs(&[1, 2, 3]), 1.0),
    ]
    .into_iter()
    .collect();
    (keep, evidence)
}

#[test]
fn a_disabled_or_satisfied_budget_is_a_verbatim_no_op() {
    let (keep, evidence) = ladder();
    let vars = BTreeMap::new();
    // 0 disables; 4 is exactly the incoming count; 5 and usize::MAX are above it. None may move
    // a single support, and none may report itself as engaged — "engaged" is what the sklearn
    // layer keys the report block on, so a false positive would change a report byte-for-byte
    // on a fit the budget never touched.
    for cap in [0_usize, 4, 5, usize::MAX] {
        let (out, rep) = apply_table_budget(&keep, &vars, cap, DEFAULT_TABLE_MIN_ARITY, &evidence);
        assert_eq!(out, keep, "cap={cap} must return the keep-set verbatim");
        assert!(!rep.engaged, "cap={cap} must stay idle");
        assert!(rep.dropped.is_empty() && rep.cascade_dropped.is_empty());
        assert_eq!(rep.tables_before, 4);
        assert_eq!(rep.tables_after, 4);
    }
}

#[test]
fn the_cap_is_a_cap_and_it_spends_on_evidence_first() {
    let (keep, evidence) = ladder();
    let vars = BTreeMap::new();
    for cap in [1_usize, 2, 3] {
        let (out, rep) = apply_table_budget(&keep, &vars, cap, DEFAULT_TABLE_MIN_ARITY, &evidence);
        assert!(rep.engaged, "cap={cap}");
        let n3 = out.iter().filter(|u| u.order() >= 3).count();
        assert_eq!(n3, cap, "cap={cap} must land exactly at the cap here");
        assert_eq!(rep.tables_after, n3 as u64);
        assert_eq!(
            n_tables_at_or_above(&rep.arity_after, DEFAULT_TABLE_MIN_ARITY),
            n3,
            "the report's histogram must agree with the keep-set it describes"
        );
        // Best evidence first: the survivors are the top `cap` of the 4/3/2/1 ladder.
        let want: Vec<FeatureSet> = [fs(&[0, 1, 2]), fs(&[0, 1, 3]), fs(&[0, 2, 3])]
            .into_iter()
            .take(cap)
            .collect();
        for u in &want {
            assert!(
                out.contains(u),
                "cap={cap} dropped a better-evidenced {u:?}"
            );
        }
    }
}

/// **Mains and pairs are untouchable.** The tightest possible cap must still deploy every
/// support below the floor. Without this the knob is just a harsher prune, and the bar Ralph
/// stated ("cells free, multi-way table COUNT minimal") would not be what it implements.
#[test]
fn nothing_below_the_arity_floor_can_ever_be_dropped() {
    let (keep, evidence) = ladder();
    let vars = BTreeMap::new();
    let (out, rep) = apply_table_budget(&keep, &vars, 1, DEFAULT_TABLE_MIN_ARITY, &evidence);
    for u in keep.iter().filter(|u| u.order() < 3) {
        assert!(out.contains(u), "the budget dropped {u:?}, below the floor");
    }
    assert_eq!(rep.arity_after[0], 3, "all 3 mains must survive");
    assert_eq!(rep.arity_after[1], 3, "all 3 pairs must survive");
    assert!(rep
        .dropped
        .iter()
        .chain(&rep.cascade_dropped)
        .all(|u| u.order() >= 3));
}

/// The floor is a knob, and moving it moves what is priced. At `min_arity = 2` the pairs join
/// the priced pool, and the mains still cannot be touched.
///
/// This case also pins the ONE way the cap can UNDERSHOOT, which is worth stating explicitly
/// rather than discovering in a battery. The greedy is best-evidence-first over the whole
/// priced pool, so at `cap = 2` it admits the two best-evidenced supports — both TRIPLES —
/// whose 2-way faces it has just dropped. Heredity then takes the triples too, and the priced
/// band empties. That is correct (a bank with `{0,1,2}` but no `{0,1}` is not a valid order
/// ideal) and it is deliberately NOT repaired by a re-admission pass: `apply_box_budget` makes
/// the same one-pass, never-re-admit promise, and matching it keeps the drop-set a pure
/// function of the inputs.
///
/// It also cannot happen in the configuration that ships. At `max_interaction_order = 3` with
/// the floor at 3, every priced support's faces are 2-way — below the floor, and therefore
/// undroppable — so the cascade has nothing to fire on. The undershoot needs an order-4+ bank
/// AND a floor below the top arity.
#[test]
fn the_arity_floor_selects_what_is_priced_and_heredity_may_undershoot_the_cap() {
    let (keep, evidence) = ladder();
    let vars = BTreeMap::new();
    let (out, rep) = apply_table_budget(&keep, &vars, 2, 2, &evidence);
    assert_eq!(rep.tables_before, 7, "3 pairs + 4 triples are priced at 2");
    assert_eq!(
        out.iter().filter(|u| u.order() == 1).count(),
        3,
        "mains held"
    );
    let priced_after = out.iter().filter(|u| u.order() >= 2).count();
    assert!(
        priced_after <= 2,
        "the cap is a CAP, undershoot included: {priced_after}"
    );
    assert_eq!(priced_after as u64, rep.tables_after);
    // The undershoot is heredity, not a lost support: every priced survivor of the greedy that
    // is missing from `out` must be recorded as a cascade drop.
    for u in &rep.cascade_dropped {
        assert!(u.order() >= 2, "cascade reached below the floor: {u:?}");
    }
    // Same fixture, floor at 3: the faces are all below the floor, so nothing cascades and the
    // cap is hit exactly. This is the shipping configuration.
    let (out3, rep3) = apply_table_budget(&keep, &vars, 2, 3, &evidence);
    assert!(rep3.cascade_dropped.is_empty());
    assert_eq!(out3.iter().filter(|u| u.order() >= 3).count(), 2);
}

/// Heredity: a surviving support may not properly contain a dropped one. The cascade can only
/// remove HIGHER arity than something already dropped, so it can never push the count back over
/// the cap — asserted here rather than argued.
#[test]
fn the_heredity_cascade_cannot_breach_the_cap() {
    // {0,1,2} has the best evidence but {0,1,2,3} contains it; drop {0,1,2} and the 4-way must
    // go with it even though its own evidence is the highest in the set.
    let keep = vec![
        fs(&[0]),
        fs(&[0, 1]),
        fs(&[0, 1, 2]),
        fs(&[0, 1, 3]),
        fs(&[0, 1, 2, 3]),
    ];
    let evidence: BTreeMap<FeatureSet, f64> = [
        (fs(&[0, 1, 2]), 1.0),
        (fs(&[0, 1, 3]), 5.0),
        (fs(&[0, 1, 2, 3]), 9.0),
    ]
    .into_iter()
    .collect();
    let vars = BTreeMap::new();
    let (out, rep) = apply_table_budget(&keep, &vars, 2, DEFAULT_TABLE_MIN_ARITY, &evidence);
    assert!(rep.engaged);
    assert!(!out.contains(&fs(&[0, 1, 2])), "{out:?}");
    assert!(
        !out.contains(&fs(&[0, 1, 2, 3])),
        "the 4-way survived without its 3-way face: {out:?}"
    );
    assert!(out.contains(&fs(&[0])) && out.contains(&fs(&[0, 1])));
    let n3 = out.iter().filter(|u| u.order() >= 3).count();
    assert!(n3 <= 2, "cap breached after cascade: {n3}");
    assert_eq!(rep.tables_after, n3 as u64);
    assert!(rep.cascade_dropped.contains(&fs(&[0, 1, 2, 3])));
}

/// Variance breaks evidence ties. On a wide bank most candidates are scored by no prune fold at
/// all and carry the same `0.0` gain; without this the admitted prefix would be raw feature-id
/// order, i.e. arbitrary.
#[test]
fn an_evidence_tie_breaks_on_purified_variance_then_on_ids() {
    let keep = vec![fs(&[0, 1, 2]), fs(&[0, 1, 3]), fs(&[0, 2, 3])];
    let evidence = BTreeMap::new(); // every candidate scores 0.0
    let vars: BTreeMap<FeatureSet, f64> = [(fs(&[0, 2, 3]), 9.0), (fs(&[0, 1, 3]), 5.0)]
        .into_iter()
        .collect();
    let (out, _) = apply_table_budget(&keep, &vars, 1, DEFAULT_TABLE_MIN_ARITY, &evidence);
    assert_eq!(out, vec![fs(&[0, 2, 3])], "variance must break the tie");

    // With no variance either, the order is the ids — total, seed-free, thread-count-free.
    let (out, _) = apply_table_budget(
        &keep,
        &BTreeMap::new(),
        1,
        DEFAULT_TABLE_MIN_ARITY,
        &evidence,
    );
    assert_eq!(out, vec![fs(&[0, 1, 2])]);
}

/// **Dense >=3-way tables are priced, so their variance must be visible to the tie-break.**
///
/// The box budget's `box_variances` skips dense effects deliberately — they cost zero boxes and
/// are never priced there. The table budget prices them, so it reads `table_variances` instead.
/// Reusing the box map would sort every dense candidate to the back of an evidence-tied field,
/// which on a wide bank is most of the field, and nothing in the type system would say so.
#[test]
fn the_table_budget_tie_break_can_see_a_dense_table_s_variance() {
    let bank = fixture().bank;
    let refs = [&bank];
    let tv = table_variances(&refs);
    let bv = box_variances(&refs);
    assert!(
        !bank.tables.is_empty(),
        "the fixture must deploy dense tables or this test proves nothing"
    );
    for t in &bank.tables {
        assert!(
            tv.contains_key(&t.u),
            "table_variances lost the dense support {:?}",
            t.u
        );
        assert!(
            !bv.contains_key(&t.u),
            "box_variances is supposed to skip dense supports; {:?} appeared",
            t.u
        );
    }
    for ft in &bank.factored {
        assert!(tv.contains_key(&ft.u), "lost the factored support");
    }
}

// ---------------------------------------------------------------------------------------
// End-to-end on a real fit: the selection-time PRICE, and the path's arity histogram.
// ---------------------------------------------------------------------------------------

/// A genuine 3-way sign product plus a main-effect decoy, fit deep enough to produce a bank with
/// real >=3-way structure for the price to shop in.
fn dataset(n: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut st = seed | 1;
    let mut cols: Vec<Vec<f32>> = (0..5).map(|_| Vec::with_capacity(n)).collect();
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let v: [f64; 5] = std::array::from_fn(|_| xs(&mut st) * 2.0 - 1.0);
        let tri = v.iter().take(3).map(|x| x.signum()).product::<f64>();
        let decoy = v.last().copied().unwrap_or(0.0);
        y.push((2.0 * tri + 0.7 * decoy + 0.1 * (xs(&mut st) - 0.5)) as f32);
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
    let (cols, y) = dataset(2500, 0x0_dc27);
    let refs: Vec<&[f32]> = cols.iter().map(Vec::as_slice).collect();
    let x = bin_columns(&refs, None, &BinConfig::default(), 0).unwrap();
    let cfg = Config {
        n_trees: 40,
        learning_rate: 0.3,
        lambda: 1.0,
        interaction_gain_hurdle: 0.5,
        interaction_gain_hurdle_mode: t_boost_core::engine::InteractionGainHurdleMode::Adaptive,
        ..Config::default()
    };
    let spec = FitSpec {
        loss: &SquaredError,
        weight: None,
        exposure: None,
        monotone: MonotoneMap::new(),
        interaction: InteractionPolicy {
            max_order: 3,
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

fn prune_at(f: &Fixture, cfg: PruneConfig) -> PruneReport {
    let folds = [HoldoutFold {
        x: &f.x.0,
        y: &f.y,
        w: &f.w,
        offset: None,
    }];
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

/// **The inertness contract**, on the KEEP-SET rather than on a scalar summary: two different
/// keep-sets can share a table count and a deviance, so counting would not prove the claim.
/// The default `PruneConfig` is checked alongside the explicit zero because "the default is
/// inert" and "zero is inert" are different statements and both must hold.
#[test]
fn a_zero_table_price_selects_exactly_what_the_unpenalized_selector_selected() {
    let f = fixture();
    let baseline = prune_at(&f, PruneConfig::default());
    for cfg in [
        PruneConfig {
            se_rule: 1.0,
            lambda_boxes: 0.0,
            lambda_tables: 0.0,
            table_price_min_arity: 3,
            prune_main_effects: false,
        },
        // A floor set to something exotic must ALSO be inert while the price is off — the floor
        // is only ever consulted behind the armed check, and that is worth pinning because it is
        // the one field whose default is not zero.
        PruneConfig {
            se_rule: 1.0,
            lambda_boxes: 0.0,
            lambda_tables: 0.0,
            table_price_min_arity: 1,
            prune_main_effects: false,
        },
        // A malformed price degrades to FREE, never to a NaN that would poison `total_cmp`.
        PruneConfig {
            se_rule: 1.0,
            lambda_boxes: 0.0,
            lambda_tables: f64::NAN,
            table_price_min_arity: 3,
            prune_main_effects: false,
        },
        PruneConfig {
            se_rule: 1.0,
            lambda_boxes: 0.0,
            lambda_tables: -1.0,
            table_price_min_arity: 3,
            prune_main_effects: false,
        },
    ] {
        let got = prune_at(&f, cfg);
        assert_eq!(
            got.kept, baseline.kept,
            "lambda_tables={} floor={} moved the selection",
            cfg.lambda_tables, cfg.table_price_min_arity
        );
    }
}

/// The path must carry the histogram whether or not a price is armed, so a budget can be
/// calibrated from an unpenalized run. It must also be a partition of `n_tables` — a histogram
/// that silently lost a bucket would calibrate a budget against the wrong number.
#[test]
fn the_path_carries_an_arity_histogram_for_calibration_even_when_unpriced() {
    let f = fixture();
    let report = prune_at(&f, PruneConfig::default());
    assert!(report.path.len() > 1, "need a real path to check");
    for p in &report.path {
        let total: u32 = p.n_tables_by_arity.iter().copied().sum();
        assert_eq!(
            total, p.n_tables,
            "the arity histogram must partition n_tables at every waypoint"
        );
    }
    // The walk only ever drops, so every bucket is monotone non-increasing along the path.
    for w in report.path.windows(2) {
        for k in 0..w[0].n_tables_by_arity.len() {
            assert!(
                w[1].n_tables_by_arity[k] <= w[0].n_tables_by_arity[k],
                "arity {} grew along a backward walk",
                k + 1
            );
        }
    }
    assert!(
        report.path[0].n_tables_by_arity[2] > 0,
        "the fixture must produce >=3-way tables or this file proves nothing"
    );
}

/// A price buys parsimony in the priced band, MONOTONICALLY — raising it must never increase
/// the count of tables at or above the floor. That is what makes it usable as a tuned dial.
///
/// What it deliberately does NOT assert is that a large enough price clears the band. It cannot,
/// and that limit is the whole reason `apply_table_budget` exists alongside it: the price only
/// re-picks a waypoint on an already-fixed deviance-greedy backward path, and that path drops
/// whichever table costs least deviance, not whichever is widest. Measured on a real deploy fit
/// (9 features, depth 6, order 3, 33 three-way tables unpriced): the price moves the count to 31
/// and then to 29 and SATURATES there, across four more decades of lambda. The deploy-time
/// budget on the same fit lands on 8, 4 or 2 exactly, on request. Anyone reaching for a table
/// knob to satisfy the explainability bar wants the budget; this is the term that lets SELECTION
/// see the count at all, which is the gap the av38 battery named.
#[test]
fn raising_the_table_price_moves_the_priced_count_monotonically_down_but_saturates() {
    let f = fixture();
    let base = prune_at(&f, PruneConfig::default());
    let full = &base.path[0];
    // Calibrate off the report exactly as the docs tell a caller to: a price in units of the
    // full bank's own held-out deviance, per priced table.
    let unit = full.mean_deviance / f64::from(full.n_tables.max(1));
    let unpriced = base.kept.iter().filter(|u| u.order() >= 3).count();
    assert!(
        unpriced > 0,
        "the fixture must have a priced band to shrink"
    );
    let mut last = usize::MAX;
    let mut smallest = unpriced;
    for mult in [0.0_f64, 0.01, 0.1, 1.0, 100.0, 10_000.0] {
        let cfg = PruneConfig {
            se_rule: 1.0,
            lambda_boxes: 0.0,
            lambda_tables: mult * unit,
            table_price_min_arity: 3,
            prune_main_effects: false,
        };
        let got = prune_at(&f, cfg);
        let n3 = got.kept.iter().filter(|u| u.order() >= 3).count();
        assert!(
            n3 <= last,
            "price {mult}x raised the >=3-way count {last} -> {n3}"
        );
        last = n3;
        smallest = smallest.min(n3);
    }
    assert!(
        smallest < unpriced,
        "the price never moved the priced band at all ({unpriced} throughout)"
    );
}
