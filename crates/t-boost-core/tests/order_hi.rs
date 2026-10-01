//! The HIGH-ORDER lift (`max_interaction_order` 5..=8, `max_depth` 7..=8).
//!
//! Companion to `order_lift.rs` (which took the order cap 3 → 4) and `depth_lift.rs` (which
//! took the depth cap 3 → 6). This file answers the two questions the earlier lifts left open,
//! and it is deliberate that they are different KINDS of question:
//!
//! 1. **Is the decomposition still exact at orders 5–8?** Mechanically, yes, and this file
//!    proves it on real fits: reconstruction is lossless to `1e-10` in `f64` on actual data
//!    rows, all five I2 gates pass, a planted order-`k` effect lands in exactly ONE factored
//!    table of arity `k`, and the whole thing round-trips the wire at arity 8.
//!
//! 2. **Does high order CONCENTRATE the signal, or diffuse it?** This is the campaign's real
//!    question and it is empirical — the arena battery answers it on real books. But one half
//!    of the answer is a THEOREM about the product contract, provable here: the prune keeps a
//!    downward-closed order ideal (heredity), so one surviving order-`k` table implies its
//!    entire subset lattice — `2^k − 1` tables, `Σ_{j≥4} C(k,j)` of them at order ≥ 4. That is
//!    6 at `k=5`, 22 at `k=6`, 64 at `k=7`, **163 at `k=8`**.
//!
//!    So "concentration" in the sense a reader wants — one big table instead of many small
//!    ones — is UNAVAILABLE at high order by construction, not by measurement. Order 5 fits a
//!    readable bank; order 6 spends a whole readable bank on one effect; orders 7 and 8 are
//!    expressible and not readable. `heredity_lattice_cost_is_combinatorial` states that in
//!    executable form so the claim cannot drift away from the code.

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

use t_boost_core::cat::CatEncoderStore;
use t_boost_core::engine::{
    InteractionGainHurdleMode, LEGACY_MAX_DEPTH, LEGACY_MAX_ORDER, MAX_DEPTH, MAX_ORDER,
    ORDER_LIFT_MAX_ORDER,
};
use t_boost_core::{
    assert_exact_decomposition, bin_columns, check_feature_budget, decode_doc, encode_doc,
    BinConfig, Booster, Config, CredibilityFloor, ExactnessMode, FeatureSet, FitSpec,
    InteractionPolicy, Model, ModelDoc, MonotoneMap, RefMeasure, ServeBinnedMatrix, SquaredError,
    TableBank, SCHEMA_VERSION_HIGH_ORDER, SCHEMA_VERSION_ORDER_LIFTED,
};

// ---------------------------------------------------------------------------------
// Fixture: a PLANTED order-k effect that a greedy booster can actually find.
// ---------------------------------------------------------------------------------

/// Fixture size. Large enough that a `2^8 = 256`-cell order-8 tree still puts ~30 rows in
/// every cell, which is what makes the planted variance recoverable at all at `k = 8`.
const N_ROWS: usize = 8000;
/// Boosting rounds. Enough for the shrunk fit to converge on the planted structure; more
/// would only sharpen numbers this file asserts loose bounds on.
const N_TREES: u32 = 40;
/// Planted order-`k` amplitude. The parity term's `w`-weighted variance is exactly `AMP²`
/// under balanced `±1` marginals, so `9.0` is the number every variance assertion is scaled
/// against.
const AMP: f64 = 3.0;

fn xs(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// `k` signal features plus two decoys, all BINARY (`±1`), carrying:
///
/// * a main effect on each signal feature, coefficient `c_i`, and
/// * one order-`k` PARITY term `amp · Π_{i<k} s_i`,
///
/// and nothing in between. Under the balanced `±1` marginals this is an exact fANOVA
/// decomposition already: the parity term's every proper-subset component is identically
/// zero, so the true order-`k` component is `amp · Π s_i` and the true components at orders
/// `2..k−1` are exactly `0`. That is what makes "the planted effect lands in ONE table of the
/// right arity" a checkable claim rather than a vibe.
///
/// # Why binary features, and why main effects alongside the parity
///
/// Two independent reasons, both about making the test test something.
///
/// Binary: a `k`-way effect's DENSE cube is the product of `k` global merged extents, so on a
/// 254-bin grid an order-8 support projects to `254^8` cells and the §07.4 readability prior
/// would (correctly) refuse to rank it. Two-valued features put every extent at 2, so the
/// cube is `2^8 = 256` and the prior is irrelevant — leaving the ARITY machinery as the only
/// thing under test, which is the point.
///
/// Main effects: a pure parity target is greedily UNFINDABLE. Every proper subset of it has
/// exactly zero marginal gain, so a greedy first split sees nothing and the fixture would
/// prove only that the booster cannot find parity — a true statement about greedy boosting
/// and a useless one about arity. The main effects are what walk the `k` features onto one
/// tree; once they are all on it, its `2^k` leaves fit the parity exactly and purification is
/// then free to place that mass wherever it belongs. This fixture is a test of the
/// DECOMPOSITION, not of the search.
///
/// `extra` decoy features carry weak main effects only. At `extra = 0` the order-`k` support
/// is UNAMBIGUOUS — there is exactly one way to choose `k` features from `k` — which is what
/// isolates the representation from the search. Raising it is how
/// `high_order_supports_proliferate_with_every_irrelevant_feature` measures the search.
fn planted(k: usize, extra: usize, n: usize, amp: f64, seed: u64) -> (Vec<Vec<f32>>, Vec<f32>) {
    let n_features = k + extra;
    let mut s = seed | 1;
    let mut cols: Vec<Vec<f32>> = (0..n_features).map(|_| Vec::with_capacity(n)).collect();
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let bits: Vec<f64> = (0..n_features)
            .map(|_| if xs(&mut s) < 0.5 { -1.0 } else { 1.0 })
            .collect();
        for (col, &b) in cols.iter_mut().zip(bits.iter()) {
            col.push(b as f32);
        }
        // Distinct, comfortably separated main-effect coefficients so the greedy level order
        // is stable and every signal feature is preferred to every decoy.
        let mut target = 0.0;
        for i in 0..k {
            target += (1.0 + 0.10 * (i as f64)) * bits[i];
        }
        for j in 0..extra {
            target += (0.05 - 0.02 * j as f64) * bits[k + j];
        }
        let mut parity = 1.0;
        for &b in bits.iter().take(k) {
            parity *= b;
        }
        target += amp * parity;
        y.push((target + 0.02 * (xs(&mut s) - 0.5)) as f32);
    }
    (cols, y)
}

fn binned(cols: &[Vec<f32>]) -> t_boost_core::BinnedMatrix {
    let refs: Vec<&[f32]> = cols.iter().map(Vec::as_slice).collect();
    bin_columns(&refs, None, &BinConfig::default(), 0).unwrap()
}

/// The NEUTRAL engine primitive: no admission hurdle. Deliberate, and worth saying why.
///
/// The product recipe's adaptive hurdle is an ORDER ESCALATOR — order `k` pays `2^(k−2)` × the
/// base, so at the product default of `2.0` an order-8 split must clear **128×** the level-1
/// gain. That is the readability contract working exactly as designed, and it means the
/// product recipe will essentially never grow an order-8 tree. Testing EXACTNESS through a
/// gate whose job is to refuse the thing under test would measure the gate, not the algebra.
/// The escalator gets its own test (`hurdle_escalates_by_a_uniform_doubling_per_order`).
fn unpriced(n_trees: u32) -> Config {
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

/// Fit the planted order-`k` fixture at `(depth, order) = (k, k)`.
///
/// `max_depth == max_order` is the right pairing at high order and the order lift measured
/// why: at depth `k` an order-`k` tree is rank-1 and contributes exactly ONE region box, while
/// a deeper tree contributes up to `Π kᵢ` boxes that do not merge across trees (swautoins went
/// 110 boxes at depth 4 to 5,643 at depth 6 for the same order). Binary features make that
/// pairing exact here — there is no second threshold to spend a deeper level on.
fn fit_planted(
    k: usize,
    extra: usize,
    n: usize,
    n_trees: u32,
    amp: f64,
) -> (Model, ServeBinnedMatrix) {
    let (cols, y) = planted(k, extra, n, amp, 0x0_11_de);
    let x = binned(&cols);
    let model = Booster::with_config(unpriced(n_trees))
        .fit(&x, &y, &spec_with(k as u8, k as u8))
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

/// Total `w`-weighted variance carried by effects of exactly `order`, dense and factored.
fn variance_at_order(bank: &TableBank, order: usize) -> f64 {
    bank.tables
        .iter()
        .filter(|t| t.u.order() == order)
        .map(|t| t.variance)
        .chain(
            bank.factored
                .iter()
                .filter(|f| f.u.order() == order)
                .map(|f| f.variance),
        )
        .sum()
}

// ---------------------------------------------------------------------------------
// 1. THE NON-NEGOTIABLE: exactness at every order 5..=8.
// ---------------------------------------------------------------------------------

/// All five I2 gates, on a REAL fit that actually realizes an order-`k` tree, for every
/// `k` in `5..=MAX_ORDER`.
///
/// `order_lift.rs::all_five_i2_gates_pass_at_every_lifted_order` already sweeps the whole
/// legal `(depth, order)` rectangle and therefore now covers 5–8 too — but it does so on a
/// fixture whose planted structure is only 4-way, so at order 5+ it proves the gates hold on
/// a model that never USED the extra order. This one plants the order the arm claims.
#[test]
fn all_five_i2_gates_pass_on_a_realized_order_five_to_eight_fit() {
    for k in (ORDER_LIFT_MAX_ORDER + 1)..=MAX_ORDER {
        let (model, x) = fit_planted(k, 0, N_ROWS, N_TREES, AMP);
        assert!(
            matches!(model.mode, ExactnessMode::Exact),
            "order {k}: a high-order fit must stay Exact"
        );
        check_feature_budget(&model).unwrap();
        assert!(
            model
                .trees
                .iter()
                .any(|(_, t)| distinct_raws(&model, t) == k),
            "order {k}: fixture must actually grow an order-{k} tree, or the gates below \
             prove nothing about arity {k}"
        );
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        assert_exact_decomposition(&model, &bank, &x)
            .unwrap_or_else(|e| panic!("order {k}: I2 gate failed: {e}"));
    }
}

/// **Lossless reconstruction, stated the way the product states it**: the sum of the tables
/// IS the model, to `1e-10`, on real data rows, in `f64` on both sides.
///
/// This is a stronger and more literal claim than the I2 `Reconstruction` gate, which sweeps
/// representative bin tuples against a tolerance that scales with tree count. Here every row
/// of the fitted matrix is scored twice — once by walking the ensemble
/// ([`Model::ensemble_f64`]) and once by summing the purified bank
/// ([`TableBank::score_binned`]) — and the two must agree to a hard absolute `1e-10`, which is
/// roughly five orders of magnitude tighter than `f32` epsilon and therefore a genuine
/// statement about the ALGEBRA rather than about rounding.
#[test]
fn sum_of_tables_equals_the_ensemble_to_1e_10_at_every_order_five_to_eight() {
    for k in (ORDER_LIFT_MAX_ORDER + 1)..=MAX_ORDER {
        let (model, x) = fit_planted(k, 0, N_ROWS, N_TREES, AMP);
        let bank = model.explain(&x, RefMeasure::default()).unwrap();

        let n_rows = x.0.n_rows as usize;
        let mut from_tables = vec![0.0_f64; n_rows];
        bank.score_binned(&CatEncoderStore::new(), &x.0, &mut from_tables)
            .unwrap();

        let mut worst = 0.0_f64;
        for row in 0..n_rows {
            let bins: Vec<u8> = x.0.data.iter().map(|c| c[row]).collect();
            let ensemble = model.ensemble_f64(&bins).unwrap();
            worst = worst.max((ensemble - from_tables[row]).abs());
        }
        assert!(
            worst <= 1e-10,
            "order {k}: worst |ensemble − Σ tables| over {n_rows} rows was {worst:e}, \
             above the 1e-10 losslessness bar"
        );
    }
}

// ---------------------------------------------------------------------------------
// 2. CONCENTRATION: the planted effect lands in ONE table of the right arity.
// ---------------------------------------------------------------------------------

/// **The central mechanical claim, and the campaign's question asked where it has a clean
/// answer**: when the order-`k` support is unambiguous, a planted order-`k` effect is
/// represented by exactly ONE factored table of arity `k`, carrying essentially all of the
/// planted variance.
///
/// The fixture is run with `extra = 0` — exactly `k` features — deliberately. There is then
/// only one way to choose `k` of them, so the SEARCH cannot be the thing being measured and
/// what remains under test is the REPRESENTATION: can a `k`-way signal live in a single
/// `k`-way table? It can, at every order through 8, and it does so while the reconstruction
/// stays lossless.
///
/// The intermediate orders (`2..k−1`) are not exactly zero, and the bound below says so
/// honestly. The planted function has no mass there — parity's every proper-subset component
/// vanishes under balanced `±1` marginals — so what lands at orders 2..k−1 is the FIT's own
/// error: a shrunk, finite-`n` greedy ensemble does not reproduce parity exactly, and the
/// residual is spread over the lattice by purification, which is the correct behaviour. It
/// measures ~15–25% of the top table's variance at these settings, well under the 35% bar,
/// and it shrinks with more rounds. What matters is the shape: order 1 and order `k` carry
/// the mass, the middle carries fit noise.
///
/// Note what this does NOT claim: that a REAL book's signal concentrates. Real signal is not
/// planted parity, and the arena battery is what answers that. This proves the representation
/// is CAPABLE of concentration — the precondition, not the conclusion.
#[test]
fn a_planted_k_way_effect_lands_in_exactly_one_table_of_arity_k() {
    for k in (ORDER_LIFT_MAX_ORDER + 1)..=MAX_ORDER {
        let (model, x) = fit_planted(k, 0, N_ROWS, N_TREES, AMP);
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        assert_exact_decomposition(&model, &bank, &x).unwrap();

        // Exactly one effect of arity k, and it is on the planted support {0..k−1}.
        let top: Vec<&FeatureSet> = bank
            .tables
            .iter()
            .map(|t| &t.u)
            .chain(bank.factored.iter().map(|f| &f.u))
            .filter(|u| u.order() == k)
            .collect();
        assert_eq!(
            top.len(),
            1,
            "order {k}: expected ONE arity-{k} effect, found {} — {:?}",
            top.len(),
            top
        );
        let planted_support: Vec<u32> = (0..k as u32).collect();
        let got: Vec<u32> = top[0].0.iter().map(|f| f.0).collect();
        assert_eq!(
            got, planted_support,
            "order {k}: the arity-{k} effect is on the wrong support"
        );

        // It is FACTORED, never a dense cube — the representation claim from the order lift,
        // which only gets more load-bearing as k grows.
        assert!(
            bank.tables.iter().all(|t| t.u.order() <= 2),
            "order {k}: no order-3+ effect may be materialized densely"
        );

        // The planted order-k mass is where it was planted.
        let top_var = variance_at_order(&bank, k);
        assert!(
            top_var > 0.7 * AMP * AMP,
            "order {k}: the arity-{k} table carries variance {top_var:.4}, far below the \
             planted {:.4} — the effect did not concentrate",
            AMP * AMP
        );
        // ...and the middle of the lattice carries only fit residual, not planted signal.
        let mid_var: f64 = (2..k).map(|m| variance_at_order(&bank, m)).sum();
        assert!(
            mid_var < 0.35 * top_var,
            "order {k}: orders 2..{} carry variance {mid_var:.4} against the arity-{k} \
             table's {top_var:.4} — the planted effect leaked into intermediate orders",
            k - 1
        );
    }
}

/// **The measured DIFFUSION, and the finding this file exists to record.**
///
/// The test above shows a `k`-way signal can live in one `k`-way table. This one shows what
/// it costs to have that option available, and the answer is that the cost is paid in TABLES
/// THAT CARRY NO SIGNAL AT ALL.
///
/// The mechanism is structural, not statistical: a depth-`k` oblivious tree must place `k`
/// distinct raw features, so once `max_depth == max_order == k` EVERY tree in the ensemble
/// emits some order-`k` support. With exactly `k` features available there is one such
/// support; with `k + e` there are `C(k+e, k)`, and boosting rounds fitting shrunken residual
/// noise wander across them. Each distinct subset any tree happens to pick becomes its own
/// table, plus its own subset lattice.
///
/// Measured here (`k = 6`, two irrelevant features added, unpriced): the arity-6 support count
/// goes 1 → 20-odd and the order-`>= 4` table count goes **22 → ~150**, a ~7x inflation, for
/// two decoys. At `k = 8` the same two decoys take it from 163 to ~800.
///
/// And the signal does not move: the planted support keeps essentially all of the order-`k`
/// variance in both arms. So the proliferating tables are EMPTY — which is the good news
/// (the evidence gate can drop them) and the bad news (nothing in the GROWER declines them,
/// so the bank is unreadable before the prune ever runs, and every dropped table still had to
/// be purified and scored first).
#[test]
fn high_order_supports_proliferate_with_every_irrelevant_feature() {
    let k = 6usize;

    let census = |extra: usize| -> (usize, usize, f64, f64) {
        let (model, x) = fit_planted(k, extra, N_ROWS, N_TREES, AMP);
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        let supports: Vec<(Vec<u32>, usize, f64)> = bank
            .tables
            .iter()
            .map(|t| (t.u.0.iter().map(|f| f.0).collect(), t.u.order(), t.variance))
            .chain(
                bank.factored
                    .iter()
                    .map(|f| (f.u.0.iter().map(|g| g.0).collect(), f.u.order(), f.variance)),
            )
            .collect();
        let planted: Vec<u32> = (0..k as u32).collect();
        let n_top = supports.iter().filter(|(_, o, _)| *o == k).count();
        let n_ge4 = supports
            .iter()
            .filter(|(_, o, _)| *o >= ORDER_LIFT_MAX_ORDER)
            .count();
        let v_planted: f64 = supports
            .iter()
            .filter(|(u, o, _)| *o == k && *u == planted)
            .map(|(_, _, v)| v)
            .sum();
        let v_top: f64 = supports
            .iter()
            .filter(|(_, o, _)| *o == k)
            .map(|(_, _, v)| v)
            .sum();
        (n_top, n_ge4, v_planted, v_top)
    };

    let (top_clean, ge4_clean, planted_clean, sum_clean) = census(0);
    let (top_noisy, ge4_noisy, planted_noisy, sum_noisy) = census(2);

    assert_eq!(
        top_clean, 1,
        "the unambiguous arm must realize one arity-{k} support"
    );
    assert!(
        top_noisy >= 10,
        "two decoys must visibly proliferate arity-{k} supports (got {top_noisy})"
    );
    assert!(
        ge4_noisy > 5 * ge4_clean,
        "two decoys took the order->=4 table count {ge4_clean} -> {ge4_noisy}; this test \
         exists to notice if that inflation ever stops happening"
    );

    // The proliferation is EMPTY: in both arms the planted support holds essentially all of
    // the order-k variance, so the extra supports are tables without signal.
    for (planted, sum, what) in [
        (planted_clean, sum_clean, "clean"),
        (planted_noisy, sum_noisy, "noisy"),
    ] {
        assert!(
            planted > 0.95 * sum,
            "{what} arm: the planted support holds {planted:.4} of the {sum:.4} total \
             arity-{k} variance — the SIGNAL diffused, not just the table count"
        );
    }
}

/// Decoy features DO enter high-order supports, and they carry no signal when they do.
///
/// This test asserted the opposite when it was written, on the assumption that a feature with
/// only a weak main effect would never be admitted to a `k`-way conjunction. That was wrong,
/// and the reason is worth keeping: admission is per-LEVEL and greedy, so at level `k` the
/// grower takes the best remaining axis whatever its provenance — there is no notion of "this
/// feature does not belong in an interaction" anywhere in the split finder. The hurdle prices
/// the TRANSITION, not the feature.
///
/// So the guarantee the product actually has is the weaker, sufficient one asserted here: a
/// decoy-bearing high-order support carries negligible purified variance, which is what lets
/// the evidence gate and the box budget remove it downstream. The grower is not the layer
/// that keeps decoys out; the prune is.
#[test]
fn decoy_bearing_high_order_supports_carry_no_signal() {
    let k = 6usize;
    let extra = 2usize;
    let (model, x) = fit_planted(k, extra, N_ROWS, N_TREES, AMP);
    let bank = model.explain(&x, RefMeasure::default()).unwrap();

    let decoys: Vec<u32> = (0..extra).map(|j| (k + j) as u32).collect();
    let mut decoy_bearing = 0usize;
    let mut decoy_variance = 0.0_f64;
    for (u, order, var) in bank
        .tables
        .iter()
        .map(|t| (&t.u, t.u.order(), t.variance))
        .chain(
            bank.factored
                .iter()
                .map(|f| (&f.u, f.u.order(), f.variance)),
        )
    {
        if order <= LEGACY_MAX_ORDER {
            continue;
        }
        if u.0.iter().any(|f| decoys.contains(&f.0)) {
            decoy_bearing += 1;
            decoy_variance += var;
        }
    }

    assert!(
        decoy_bearing > 0,
        "the grower is expected to admit decoys into high-order supports — if it has \
         stopped doing so, the reasoning in this test's doc comment needs revisiting"
    );
    let planted_var = variance_at_order(&bank, k);
    assert!(
        decoy_variance < 0.10 * planted_var,
        "decoy-bearing order->{LEGACY_MAX_ORDER} supports carry {decoy_variance:.4} of \
         variance against the planted {planted_var:.4}: they are no longer negligible, so \
         the downstream evidence gate can no longer be relied on to remove them"
    );
}

// ---------------------------------------------------------------------------------
// 3. HEREDITY: why "few high-order tables" is arithmetically unavailable.
// ---------------------------------------------------------------------------------

/// `Σ_{j=lo}^{k} C(k, j)`.
fn lattice_size(k: usize, lo: usize) -> usize {
    fn binom(n: usize, r: usize) -> usize {
        if r > n {
            return 0;
        }
        let mut acc = 1usize;
        for i in 0..r {
            acc = acc * (n - i) / (i + 1);
        }
        acc
    }
    (lo..=k).map(|j| binom(k, j)).sum()
}

/// **The readability theorem, in executable form.**
///
/// The prune keeps a downward-closed order ideal — a table survives only if every immediate
/// subset also earned its keep (`aggregate_prune_selection`'s cascade), and the deploy-time
/// box budget re-runs the same cascade. So a surviving order-`k` table is never one table: it
/// is the tip of its full subset lattice.
///
/// The consequence is arithmetic and it is the honest answer to "does native high order let
/// us ship FEW tables":
///
/// | k | tables implied (`2^k − 1`) | of them order ≥ 4 |
/// |---|---|---|
/// | 5 | 31 | **6** |
/// | 6 | 63 | **22** |
/// | 7 | 127 | **64** |
/// | 8 | 255 | **163** |
///
/// Against a "20–30 high-order tables a human can read" bar: order 5 costs 6 and fits several
/// times over; order 6 costs 22 and spends the entire allowance on a single effect; order 7
/// blows it by 2×; order 8 by 5–8×. **No amount of concentration in the signal changes this**
/// — it is the cost of the contract, not of the data. A campaign that hopes to buy readability
/// by raising the order is hoping against this table, and should be told so before it spends
/// the compute.
#[test]
fn heredity_lattice_cost_is_combinatorial() {
    assert_eq!((lattice_size(5, 1), lattice_size(5, 4)), (31, 6));
    assert_eq!((lattice_size(6, 1), lattice_size(6, 4)), (63, 22));
    assert_eq!((lattice_size(7, 1), lattice_size(7, 4)), (127, 64));
    assert_eq!((lattice_size(8, 1), lattice_size(8, 4)), (255, 163));

    // The readable-bank bar Ralph set for the order-hi campaign, applied to the table above.
    const READABLE_HIGH_ORDER_TABLES: usize = 30;
    assert!(lattice_size(5, 4) <= READABLE_HIGH_ORDER_TABLES / 4);
    assert!(lattice_size(6, 4) <= READABLE_HIGH_ORDER_TABLES);
    assert!(lattice_size(7, 4) > READABLE_HIGH_ORDER_TABLES);
    assert!(lattice_size(8, 4) > 5 * READABLE_HIGH_ORDER_TABLES);
}

/// And the lattice is not hypothetical — it is exactly, integer-for-integer, what a real fit
/// produces.
///
/// A bank that realizes an order-`k` effect carries every one of its `2^k − 1` sub-supports,
/// because the purification cascade SHEDS the effect's marginals down through every face on
/// its way to the intercept. This is the part a reader is most likely to doubt — "surely the
/// sub-tables are only created if they carry signal?" They are created unconditionally,
/// because the shed is what makes the decomposition exact. Whether they SURVIVE the prune is
/// a separate question; that they exist, are purified, and are scored is not.
///
/// The second assertion is the one worth the compute: on the unambiguous fixture the bank's
/// order-`>= 4` table count equals `Σ_{j=4}^{k} C(k,j)` **exactly** — 6, 22, 64, 163 for
/// `k = 5,6,7,8`. Theory and measurement agree to the integer, which is what licenses quoting
/// the combinatorial table as a prediction about real books rather than as an upper bound.
#[test]
fn a_realized_high_order_effect_materializes_its_whole_subset_lattice() {
    for k in (ORDER_LIFT_MAX_ORDER + 1)..=MAX_ORDER {
        let (model, x) = fit_planted(k, 0, N_ROWS, N_TREES, AMP);
        let bank = model.explain(&x, RefMeasure::default()).unwrap();

        let present: Vec<Vec<u32>> = bank
            .tables
            .iter()
            .map(|t| &t.u)
            .chain(bank.factored.iter().map(|f| &f.u))
            .map(|u| u.0.iter().map(|f| f.0).collect())
            .collect();

        // Every non-empty subset of {0..k−1} must be a realized support.
        for mask in 1u32..(1u32 << k) {
            let sub: Vec<u32> = (0..k as u32).filter(|i| mask >> i & 1 == 1).collect();
            assert!(
                present.contains(&sub),
                "order {k}: the effect's face {sub:?} is absent from the bank — the shed \
                 did not deposit it, which would make the decomposition inexact"
            );
        }
        assert_eq!(
            present.len(),
            lattice_size(k, 1),
            "order {k}: the bank should hold exactly the 2^{k} - 1 subset lattice"
        );

        // The readability bill, measured against the closed form.
        let n_high = present
            .iter()
            .filter(|u| u.len() >= ORDER_LIFT_MAX_ORDER)
            .count();
        assert_eq!(
            n_high,
            lattice_size(k, ORDER_LIFT_MAX_ORDER),
            "order {k}: order->={ORDER_LIFT_MAX_ORDER} table count must equal the \
             combinatorial prediction"
        );
    }
}

// ---------------------------------------------------------------------------------
// 4. The hurdle: a UNIFORM doubling at every n → n+1 transition.
// ---------------------------------------------------------------------------------

/// The interaction-gain hurdle must escalate by the SAME factor at every order transition,
/// all the way to `MAX_ORDER` — `3 → 4` charged exactly as `2 → 3` is relative to `1 → 2`, and
/// `7 → 8` likewise. The escalator is `2^(order − 2)`, so this is a monotone doubling with no
/// special case anywhere.
///
/// Measured through the FIT, not by reading the formula back: at a fixed base hurdle, raising
/// the order cap must never make the grower admit a HIGHER order more readily than a lower
/// one. Concretely, the realized order under the product recipe is non-increasing in the
/// hurdle, and the order at which admission stops moves down monotonically as the hurdle rises
/// — which is what "uniformly hurdled" means operationally.
#[test]
fn hurdle_escalates_by_a_uniform_doubling_per_order() {
    let k = 6;
    let (cols, y) = planted(k, 0, N_ROWS, AMP, 0x0_11_de);
    let x = binned(&cols);

    let realized_order = |hurdle: f32| -> usize {
        let cfg = Config {
            n_trees: 30,
            learning_rate: 0.3,
            lambda: 1.0,
            interaction_gain_hurdle: hurdle,
            interaction_gain_hurdle_mode: InteractionGainHurdleMode::Adaptive,
            ..Config::default()
        };
        let model = Booster::with_config(cfg)
            .fit(&x, &y, &spec_with(k as u8, k as u8))
            .unwrap();
        model
            .trees
            .iter()
            .map(|(_, t)| distinct_raws(&model, t))
            .max()
            .unwrap_or(0)
    };

    // A rising hurdle may never RAISE the order the grower is willing to reach.
    let ladder = [0.0_f32, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 32.0];
    let orders: Vec<usize> = ladder.iter().copied().map(realized_order).collect();
    for w in orders.windows(2) {
        assert!(
            w[1] <= w[0],
            "realized order is not monotone non-increasing in the hurdle: {orders:?} \
             over hurdles {ladder:?}"
        );
    }
    assert!(
        orders[0] > LEGACY_MAX_ORDER,
        "the unpriced arm must reach past the legacy order cap, or this test has nothing \
         to measure: {orders:?}"
    );
    assert!(
        *orders.last().unwrap() < orders[0],
        "a 32x base hurdle must refuse orders the unpriced arm accepted: {orders:?}"
    );
}

/// **The shipped hurdle REFUSES order 7 and 8 outright, and this is by design.**
///
/// The product default is `interaction_gain_hurdle = 2.0` in adaptive mode, and adaptive mode
/// is an ORDER ESCALATOR: order `k` pays `2^(k−2)` × the base against the tree's level-1 gain.
/// So the ladder a fresh feature must clear is 2.0× at order 2, 4× at 3, 8× at 4, 16× at 5,
/// 32× at 6, 64× at 7 and **128× at order 8**.
///
/// Nothing in a real book clears 128× its own strongest main effect. The consequence, measured
/// here on a fixture whose order-8 effect is genuine, planted, and unambiguous: under the
/// product recipe the grower does not reach order 8 at all — it caps out at the order lift's
/// own ceiling — and the planted variance is simply not fitted.
///
/// This is not a bug and the test does not treat it as one. It is the readability contract
/// doing exactly its job, and it is the single most important operational fact about the
/// high-order lift: **raising `MAX_ORDER` to 8 does not put order 7–8 into the product.** A
/// campaign that wants to measure high order must lower the hurdle deliberately, and by doing
/// so takes on the spurious-conjunction risk the hurdle was protecting it from.
#[test]
fn the_product_hurdle_refuses_the_top_orders_outright() {
    let k = MAX_ORDER;
    let (cols, y) = planted(k, 0, N_ROWS, AMP, 0x0_11_de);
    let x = binned(&cols);

    let realized = |hurdle: f32| -> usize {
        let cfg = Config {
            n_trees: N_TREES,
            learning_rate: 0.3,
            lambda: 1.0,
            interaction_gain_hurdle: hurdle,
            interaction_gain_hurdle_mode: InteractionGainHurdleMode::Adaptive,
            ..Config::default()
        };
        let model = Booster::with_config(cfg)
            .fit(&x, &y, &spec_with(k as u8, k as u8))
            .unwrap();
        model
            .trees
            .iter()
            .map(|(_, t)| distinct_raws(&model, t))
            .max()
            .unwrap_or(0)
    };

    // Unpriced, the planted order-8 structure is reachable...
    assert_eq!(
        realized(0.0),
        k,
        "unpriced, the grower must reach the planted order {k}"
    );
    // ...and under the SHIPPED product hurdle it is not.
    const PRODUCT_HURDLE: f32 = 2.0;
    let priced = realized(PRODUCT_HURDLE);
    assert!(
        priced < k,
        "the product hurdle ({PRODUCT_HURDLE}) admitted order {priced} == the cap {k}: the \
         2^(k-2) escalator is no longer refusing the top orders, which would put orders 7-8 \
         into the shipped recipe without anyone deciding to"
    );
    assert!(
        priced <= ORDER_LIFT_MAX_ORDER,
        "the product hurdle admitted order {priced}, past the order lift's ceiling \
         {ORDER_LIFT_MAX_ORDER} — the escalator has weakened"
    );
}

// ---------------------------------------------------------------------------------
// 5. The wire: arity 8 round-trips, and stamps a version that refuses an older reader.
// ---------------------------------------------------------------------------------

/// A model carrying an order-8 tree round-trips bincode and JSON exactly, and — the part that
/// matters for a filing — is STAMPED at the version whose reader can validate it.
///
/// The order lift already made a factored box length-carrying, so orders 5–8 need no encoding
/// change at all; the bytes an order-8 bank writes are framed identically to an order-4 one's.
/// That is precisely why the stamp is load-bearing rather than cosmetic here: it is the ONLY
/// thing standing between a `schema_version` 4 reader and a rating table whose arity it cannot
/// represent, because the decode itself would succeed.
#[test]
fn an_order_eight_model_round_trips_and_stamps_the_high_order_version() {
    let k = MAX_ORDER;
    let (model, _x) = fit_planted(k, 0, N_ROWS, N_TREES, AMP);
    assert!(
        model
            .trees
            .iter()
            .any(|(_, t)| distinct_raws(&model, t) == k),
        "fixture must realize an order-{k} tree for this to test anything"
    );

    assert_eq!(
        model.required_schema_version(),
        SCHEMA_VERSION_HIGH_ORDER,
        "an order-{k} model must claim the high-order reader version"
    );

    let doc = ModelDoc::new(model.clone());
    assert_eq!(doc.schema_version, SCHEMA_VERSION_HIGH_ORDER);

    let bytes = encode_doc(&doc).unwrap();
    let back = decode_doc(&bytes).unwrap();
    assert_eq!(
        back.model, model,
        "bincode round-trip must be exact at arity 8"
    );
    assert_eq!(
        encode_doc(&back).unwrap(),
        bytes,
        "re-encoding the decoded model must reproduce the same bytes"
    );

    let json = t_boost_core::encode_doc_json(&doc).unwrap();
    let from_json = t_boost_core::decode_doc_json(&json).unwrap();
    assert_eq!(
        from_json.model, model,
        "JSON round-trip must be exact at arity 8"
    );
}

/// **The stamp is read off the CONTENT, not off `bank.factored`.**
///
/// A regression test for a real gap, found in review. The first cut of
/// `required_tables_version` decided everything from `bank.factored`, reasoning that
/// `OverflowPolicy::Factored` (the default) routes every support of order `>= 3` there. Sound
/// for the default policy, wrong in general: `OverflowPolicy::Error` and `SparseFallback` are
/// both reachable from Python, and under either of them a high-order support whose merged cube
/// fits `max_table_cells` is materialized as a DENSE `EffectTable` with `factored` left empty.
///
/// This fixture is exactly that case — binary features give an order-`k` support a `2^k`-cell
/// cube, trivially under any budget — so the old rule stamped the most arity-surprising bank in
/// the codebase as `schema_version = 2`, and `to_rating_export` handed a filing consumer an
/// eight-axis table under a "version 2" label. That is the precise failure the export stamp
/// exists to prevent, so it is worth a test rather than a comment.
#[test]
fn a_dense_high_order_bank_still_claims_the_high_order_version() {
    use t_boost_core::{OverflowPolicy, TableBudget};

    let k = MAX_ORDER;
    let (model, x) = fit_planted(k, 0, N_ROWS, N_TREES, AMP);
    let bank = model
        .explain_with_budget(
            &x,
            RefMeasure::default(),
            TableBudget {
                on_overflow: OverflowPolicy::Error,
                ..TableBudget::default()
            },
        )
        .unwrap();

    assert!(
        bank.factored.is_empty(),
        "this fixture must exercise the DENSE path, or it is not testing the gap"
    );
    assert!(
        bank.tables
            .iter()
            .any(|t| t.u.order() > ORDER_LIFT_MAX_ORDER),
        "the dense bank must actually carry a high-order table"
    );
    assert_eq!(
        t_boost_core::serialize::content_tables_version(&bank),
        SCHEMA_VERSION_HIGH_ORDER,
        "a bank whose high-order effect is DENSE still needs the high-order reader"
    );

    // And the rating export — the artifact a filing is built from — carries that stamp.
    let export = bank
        .to_rating_export(
            model.link,
            &model.mode,
            &model.schema,
            &model.provenance,
            &CatEncoderStore::new(),
            None,
        )
        .unwrap();
    assert_eq!(
        export.schema_version, SCHEMA_VERSION_HIGH_ORDER,
        "the rating export must not understate the arity a consumer has to represent"
    );
}

/// The ladder's other rungs still hold: an order-4 / depth-6 model is stamped exactly where
/// the ORDER lift put it, not bumped along by the high-order lift.
///
/// This is the byte-identity contract in miniature. If widening the caps moved the stamp on
/// content that did not use the widening, every artifact from the order-lift era would need a
/// migration it does not in fact need.
#[test]
fn order_four_content_keeps_the_order_lift_stamp() {
    let (cols, y) = planted(4, 0, N_ROWS, AMP, 0x0_11_de);
    let x = binned(&cols);
    let model = Booster::with_config(unpriced(25))
        .fit(&x, &y, &spec_with(4, 4))
        .unwrap();
    assert!(
        model
            .trees
            .iter()
            .any(|(_, t)| distinct_raws(&model, t) == 4),
        "fixture must realize an order-4 tree"
    );
    assert_eq!(
        model.required_schema_version(),
        SCHEMA_VERSION_ORDER_LIFTED,
        "order-4 / depth-4 content must keep the stamp the ORDER lift gave it"
    );
}

// ---------------------------------------------------------------------------------
// 6. The default path is untouched.
// ---------------------------------------------------------------------------------

/// The caps moved; the DEFAULTS did not. A fit that asks for nothing gets depth 3, order 3,
/// and the pre-lift wire stamp — the single most important property of any lift in this
/// repository, and the one a widened `MAX_ORDER` is most likely to break by accident.
#[test]
fn defaults_are_untouched_by_the_high_order_lift() {
    let policy = InteractionPolicy::default();
    assert_eq!(usize::from(policy.max_depth), LEGACY_MAX_DEPTH);
    assert_eq!(usize::from(policy.max_order), LEGACY_MAX_ORDER);

    let (cols, y) = planted(4, 0, 3000, AMP, 0x0_11_de);
    let x = binned(&cols);
    let model = Booster::with_config(unpriced(25))
        .fit(&x, &y, &spec_with(policy.max_depth, policy.max_order))
        .unwrap();
    for (_, t) in &model.trees {
        assert!(usize::from(t.depth) <= LEGACY_MAX_DEPTH);
        assert!(distinct_raws(&model, t) <= LEGACY_MAX_ORDER);
    }
    assert_eq!(
        model.required_schema_version(),
        t_boost_core::SCHEMA_VERSION_UNLIFTED,
        "a default fit must still write exactly what a pre-lift build wrote"
    );
}

/// `max_order > max_depth` stays unreachable-by-construction and is refused with a
/// diagnostic, at every order the lift now admits — a tree needs one level per distinct raw
/// feature, so silently behaving as `max_depth` would give the caller a lower-order model than
/// they asked for and no way to notice.
#[test]
fn order_above_depth_is_refused_at_every_high_order() {
    for k in (ORDER_LIFT_MAX_ORDER + 1)..=MAX_ORDER {
        let (cols, y) = planted(4, 0, 800, 1.0, 7);
        let x = binned(&cols);
        let err = Booster::with_config(unpriced(5))
            .fit(&x, &y, &spec_with(k as u8 - 1, k as u8))
            .unwrap_err();
        assert!(
            matches!(err, t_boost_core::PbError::InvalidConfig { .. }),
            "order {k} at depth {}: expected InvalidConfig, got {err:?}",
            k - 1
        );
    }
    // ...and one past the cap is refused too, at a depth that would otherwise allow it.
    let (cols, y) = planted(4, 0, 800, 1.0, 7);
    let x = binned(&cols);
    let err = Booster::with_config(unpriced(5))
        .fit(&x, &y, &spec_with(MAX_DEPTH as u8, MAX_ORDER as u8 + 1))
        .unwrap_err();
    assert!(matches!(err, t_boost_core::PbError::InvalidConfig { .. }));
}
