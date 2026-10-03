//! Table pruning: greedily drop fANOVA tables to improve held-out generalization.
//!
//! Once a purified [`TableBank`] IS the model (`crate::table_model::TableModel`), a table is
//! removed simply by omitting it from the LUT-sum — the surviving purified tables are unchanged
//! (each still zero-`w`-mean), `f0` is invariant (all mass lives in the intercept), and the
//! pruned bank is a NEW exact model reconstructed against *its own* LUT-sum. This module chooses
//! *which* tables to drop.
//!
//! **Objective: improve held-out deviance.** Removing an overfit table lowers held-out deviance;
//! removing a real one raises it. So candidate drops are ranked by the proper validation loss they
//! produce, not by a marginal energy proxy. Main effects are sticky by default: whole-table pruning
//! may simplify interactions, but it should not erase a low-variance rating factor that carries rare
//! held-out signal.
//!
//! **Downward closure.** Only a *maximal* table (one with no kept proper superset) may be dropped,
//! so the keep-set stays a downward-closed order ideal (heredity): we never drop `{i,j}` while
//! keeping `{i,j,k}`. v1 is whole-table, no refit — a pure-noise table improves deviance when
//! simply omitted; recovering the partially-useful ones (refit) is a later step.

// Numeric index-arithmetic over jointly-built parallel Vecs (folds × tables × rows): the same
// tightened `indexing_slicing` deny lint that `explain.rs` scope-allows. All lengths are constructed
// together and every index is loop-derived from them, so the indexing cannot escape its buffer.
#![allow(
    clippy::indexing_slicing, // JUSTIFIED: pre-existing module-scoped debt (see comment above); burn-down to `.get()`/per-fn allows is incremental, not expanded here.
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use crate::data::{BinnedMatrix, ServeBinnedMatrix};
use crate::engine::{Model, MultiClassModel, MAX_ORDER};
use crate::error::PbError;
use crate::explain::{FeatureSet, RefMeasure, TableBank};
use crate::loss::{Link, Loss};
use crate::table_model::{MultiClassTableModel, TableModel};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::BTreeMap;

const STICKY_MAIN_EFFECTS: bool = true;

/// One held-out fold used to score candidate prunings (e.g. one bag's out-of-bag rows). Borrows
/// its data so the caller can carve folds from the training design without copying.
pub struct HoldoutFold<'a> {
    /// Already-binned design on the model's original grids.
    pub x: &'a BinnedMatrix,
    /// Targets, parallel to the fold rows.
    pub y: &'a [f32],
    /// Row weights, parallel to the fold rows.
    pub w: &'a [f32],
    /// Optional log-exposure offset (`ln(exposure)`) added to the raw score before the deviance, for
    /// log-link exposure models; `None` = no offset.
    pub offset: Option<&'a [f32]>,
}

/// Pruning knobs.
#[derive(Debug, Clone, Copy)]
pub struct PruneConfig {
    /// How many standard errors above the held-out-deviance minimum still count as "tied" — the
    /// selector then takes the simplest (fewest-tables) tied model. `1.0` is the classic 1-SE
    /// rule; `0.0` picks the exact minimum.
    pub se_rule: f64,
    /// SELECTION-time price of one deployed rank-1 region box, in held-out-deviance units.
    /// `0.0` (the default) is EXACTLY inert — see [`PruneConfig::objective`].
    ///
    /// # Why this knob exists
    ///
    /// The av38 depth battery closed with one named gap: *selection cannot see bank size, so it
    /// cannot refuse a diffuse config.* The backward walk optimizes held-out deviance alone and
    /// the SE rule then breaks TIES toward fewer tables — but a config that buys a real deviance
    /// gain at 22x the boxes is not a tie, so nothing in selection ever declines it. The
    /// deployed-box budget (`prune_box_budget`) answers the same question at DEPLOY time, after
    /// the choice has been made; this is its selection-time analogue, and the two compose.
    ///
    /// # Units, and how to pick a value
    ///
    /// Deliberately a raw scalar rather than something self-normalizing, because the honest
    /// statement is a PRICE: "I will give up this much held-out deviance per box." Deviance here
    /// is per unit weight, so the scale is the dataset's and a value has to be calibrated against
    /// it. The report carries everything needed — `path[0]` is the full bank's
    /// `(mean_deviance, se, n_boxes)` — so "one SE for the whole bank" is
    /// `lambda_boxes = path[0].se / path[0].n_boxes` and "1% of full deviance for the whole bank"
    /// is `0.01 * path[0].mean_deviance / path[0].n_boxes`.
    ///
    /// Dense tables cost ZERO boxes and are therefore never priced, exactly as in the deploy-time
    /// budget — a dense main effect or pair is what a filing reads, not what inflates it.
    pub lambda_boxes: f64,
    /// SELECTION-time price of one kept table of arity >= [`PruneConfig::table_price_min_arity`],
    /// in held-out-deviance units. `0.0` (the default) is EXACTLY inert.
    ///
    /// # Why this knob exists, and how it differs from `lambda_boxes`
    ///
    /// `lambda_boxes` prices RESOLUTION — how many rank-1 region boxes a filing carries. Ralph's
    /// 2026-08-27 explainability bar prices something else: *cells within a table are free; what
    /// must stay small is the COUNT of multi-way tables* ("100 3-way tables is not explainable").
    /// A box price answers the first question and only accidentally the second, because dropping
    /// boxes drops whole factored effects binarily. This is the direct term.
    ///
    /// # The structural limit, stated up front
    ///
    /// Like `lambda_boxes`, this re-picks a waypoint on an ALREADY-FIXED deviance-greedy backward
    /// path; it cannot reorder the walk, so it cannot preferentially surrender a high-arity table
    /// while keeping a low-arity one. Walking further back sheds mains and pairs too. For the
    /// explainability bar the deploy-time [`apply_table_budget`] is the sharper instrument — it
    /// touches ONLY supports at or above the arity floor. This term exists so selection can
    /// nonetheless SEE table count, which is what the av38 battery said was missing, and so a
    /// tuned search has a continuous dial rather than only a cliff.
    pub lambda_tables: f64,
    /// Lowest interaction order `lambda_tables` prices. Only consulted when `lambda_tables > 0`,
    /// so the default is inert regardless of its value. `3` is Ralph's bar: mains and pairs are
    /// what a filing reads, three-way tables are what inflates it.
    pub table_price_min_arity: u8,
}

impl Default for PruneConfig {
    fn default() -> Self {
        Self {
            se_rule: 1.0,
            lambda_boxes: 0.0,
            lambda_tables: 0.0,
            table_price_min_arity: DEFAULT_TABLE_MIN_ARITY,
        }
    }
}

/// The arity at or above which a table counts against the explainability bar. Mains and pairs
/// are what a filing reads; three-way tables are what inflates it (Ralph, 2026-08-27).
pub const DEFAULT_TABLE_MIN_ARITY: u8 = 3;

/// Kept-table counts by interaction order: index `k-1` = arity `k`, for `k` in `1..=MAX_ORDER`.
///
/// A fixed array rather than a map so a [`PrunePoint`] stays `Clone`-cheap and allocation-free —
/// a path is one point per table, and on the widest banks that is thousands of them.
pub type ArityCounts = [u32; MAX_ORDER];

/// The per-row effective mass (spec §08.7, `sample_weight · exposure`) a bank build reads
/// under `w_measure`, or `None` when the measure keeps the historical flat row count.
///
/// `w` is the loss weight and `offset` the log-exposure the log-link losses add to the raw
/// score, so `w · exp(offset)` is the exposure-weighted row mass; with no offset the loss
/// weight alone is the mass. Only [`RefMeasure::uses_row_mass`] measures read it — under
/// the legacy `ProductMarginals` every prune-path bank stays byte-identical to the boards
/// fitted before the exposure measure existed.
#[must_use]
pub fn measure_mass(w_measure: &RefMeasure, w: &[f32], offset: Option<&[f32]>) -> Option<Vec<f32>> {
    if !w_measure.uses_row_mass() {
        return None;
    }
    Some(match offset {
        Some(o) => w.iter().zip(o).map(|(&wi, &oi)| wi * oi.exp()).collect(),
        None => w.to_vec(),
    })
}

/// Keep `mass` only when `w_measure` reads it (see [`measure_mass`]).
#[must_use]
pub fn mass_for<'a>(w_measure: &RefMeasure, mass: Option<&'a [f32]>) -> Option<&'a [f32]> {
    if w_measure.uses_row_mass() {
        mass
    } else {
        None
    }
}

/// The arity histogram of a set of supports, in [`ArityCounts`] layout.
///
/// Orders outside `1..=MAX_ORDER` are dropped rather than clamped: a support of order 0 is not a
/// table and one above `MAX_ORDER` cannot exist (the engine's own cap), so silently folding
/// either into a real bucket would corrupt the count the explainability bar is read from.
#[must_use]
fn arity_counts_of<'a>(ids: impl IntoIterator<Item = &'a FeatureSet>) -> ArityCounts {
    let mut out: ArityCounts = [0; MAX_ORDER];
    for u in ids {
        let k = u.order();
        if (1..=MAX_ORDER).contains(&k) {
            out[k - 1] = out[k - 1].saturating_add(1);
        }
    }
    out
}

/// Drop one table of `u`'s arity from a running histogram.
///
/// Saturating, and silent on an out-of-range order for the same reason [`arity_counts_of`] skips
/// one: the histogram must stay a count, and a walk that dropped a table the histogram never
/// admitted must not underflow into `u32::MAX`.
fn decrement_arity(counts: &mut ArityCounts, u: &FeatureSet) {
    let k = u.order();
    if (1..=MAX_ORDER).contains(&k) {
        counts[k - 1] = counts[k - 1].saturating_sub(1);
    }
}

/// Tables of arity `>= min_arity` in an [`ArityCounts`] — the number Ralph's bar caps.
#[must_use]
pub fn n_tables_at_or_above(counts: &ArityCounts, min_arity: u8) -> usize {
    let floor = (min_arity as usize).clamp(1, MAX_ORDER);
    counts
        .iter()
        .skip(floor - 1)
        .copied()
        .fold(0_usize, |a, b| a.saturating_add(b as usize))
}

impl PruneConfig {
    /// Is the selection-time box price armed at all?
    ///
    /// Anything non-finite or `<= 0` is treated as OFF rather than rejected: this is a price,
    /// and a malformed price must degrade to "free", never to a NaN that would poison every
    /// `total_cmp` in the selector and make the chosen waypoint depend on iteration order.
    #[must_use]
    fn box_price_armed(&self) -> bool {
        self.lambda_boxes.is_finite() && self.lambda_boxes > 0.0
    }

    /// Is the selection-time table price armed at all? Same degrade-to-free discipline as
    /// [`PruneConfig::box_price_armed`], and for the same `total_cmp` reason.
    #[must_use]
    fn table_price_armed(&self) -> bool {
        self.lambda_tables.is_finite() && self.lambda_tables > 0.0
    }

    /// Kept tables at or above the price's arity floor, at this waypoint.
    ///
    /// The floor is clamped into `1..=MAX_ORDER` rather than rejected: a floor of 0 would price
    /// the whole bank (a table count, not a MULTI-WAY table count) and a floor above `MAX_ORDER`
    /// would price nothing. Both are configuration mistakes, and both must degrade to a defined
    /// count rather than panic inside a selector.
    #[must_use]
    fn n_priced_tables(&self, p: &PrunePoint) -> f64 {
        let floor = (self.table_price_min_arity as usize).clamp(1, MAX_ORDER);
        let n: u32 = p
            .n_tables_by_arity
            .iter()
            .skip(floor - 1)
            .copied()
            .fold(0, u32::saturating_add);
        f64::from(n)
    }

    /// The waypoint's selection objective: held-out deviance, plus the size term when a price
    /// is armed.
    ///
    /// **Returns `mean_deviance` UNTOUCHED when the price is off**, and that is the bit-identity
    /// contract rather than a nicety. An earlier cut of this computed
    /// `mean_deviance + box_penalty(..)` with `box_penalty` returning `0.0` when inert — which
    /// does NOT preserve the value, because the selector compares under `total_cmp`, and
    /// `total_cmp` distinguishes `-0.0` from `+0.0` while `-0.0 + 0.0` is `+0.0`. A deviance
    /// that underflowed to negative zero would have compared differently under a nominally
    /// inert lambda. The guard has to sit on the ADDITION, not inside the addend, so it lives
    /// here — the one place both prune paths call.
    /// The same guard now covers TWO addends. `(false, false)` still returns `mean_deviance`
    /// untouched — the `-0.0` contract above is about the addition existing at all, so the
    /// unarmed arm must remain a return, not a `+ 0.0 + 0.0`.
    #[must_use]
    fn objective(&self, p: &PrunePoint) -> f64 {
        match (self.box_price_armed(), self.table_price_armed()) {
            (false, false) => p.mean_deviance,
            (true, false) => p.mean_deviance + self.lambda_boxes * f64::from(p.n_boxes),
            (false, true) => p.mean_deviance + self.lambda_tables * self.n_priced_tables(p),
            (true, true) => {
                p.mean_deviance
                    + self.lambda_boxes * f64::from(p.n_boxes)
                    + self.lambda_tables * self.n_priced_tables(p)
            }
        }
    }
}

/// One waypoint on the backward path: the model after `n_tables` tables remain.
#[derive(Debug, Clone, Serialize)]
pub struct PrunePoint {
    /// Number of tables kept at this waypoint (fixed-width for the serialized report).
    pub n_tables: u32,
    /// Deployed rank-1 region BOXES kept at this waypoint — the size a filing actually carries,
    /// and what [`PruneConfig::lambda_boxes`] prices.
    ///
    /// Counted exactly the way the deploy-time box budget counts (see [`box_costs`]): a factored
    /// effect costs its `n_boxes()`, a dense table costs zero. Always populated, armed penalty or
    /// not, so a report from an unpenalized run can be used to CALIBRATE a lambda before
    /// spending one.
    pub n_boxes: u32,
    /// Kept-table counts by interaction order: index `k-1` holds the number of kept tables of
    /// arity `k`, for `k` in `1..=MAX_ORDER`. Sums to [`PrunePoint::n_tables`].
    ///
    /// This is the unit Ralph's explainability bar is written in — "100 3-way tables is not
    /// explainable" is a statement about `n_tables_by_arity[2]`, not about boxes or total tables.
    /// Counted DENSE AND FACTORED alike (unlike `n_boxes`, where a dense table costs nothing): a
    /// dense 3-way table is still a 3-way table in the filing. Always populated, armed price or
    /// not, so a report from an unpenalized run can CALIBRATE a price or a budget before one is
    /// spent — the same discipline `n_boxes` follows.
    pub n_tables_by_arity: ArityCounts,
    /// The table dropped to reach this waypoint (`None` for the full-model starting point).
    pub dropped: Option<FeatureSet>,
    /// Mean held-out deviance across folds (weighted, per-unit-weight).
    ///
    /// The RAW held-out figure, never the penalized one. The size term is applied at the
    /// comparison and deliberately not folded into the reported deviance, so the path stays a
    /// measurement and the penalty stays a decision.
    pub mean_deviance: f64,
    /// Standard error of `mean_deviance` across folds (`0` for a single fold).
    pub se: f64,
}

/// One table's initial held-out contribution score. `mean_gain > 0` means dropping the table from
/// the full bank worsens held-out deviance, so the table earns its keep.
#[derive(Debug, Clone, Serialize)]
pub struct PruneTableScore {
    /// Feature-set for the scored table.
    pub u: FeatureSet,
    /// Interaction order (`1` = main effect).
    pub order: u8,
    /// Held-out deviance after dropping this table minus full-model held-out deviance.
    pub mean_gain: f64,
    /// Standard error of `mean_gain` across held-out folds.
    pub se_gain: f64,
    /// Purified-table variance, retained as diagnostic metadata only.
    pub variance: f64,
    /// Whether this table is protected from whole-table deletion.
    pub sticky: bool,
    /// Whether the selected keep-set contains this table.
    pub selected: bool,
}

/// The outcome of a prune: which tables survived, the deviance path, and the improvement.
#[derive(Debug, Clone, Serialize)]
pub struct PruneReport {
    /// Feature-sets of the surviving tables.
    pub kept: Vec<FeatureSet>,
    /// Feature-sets dropped to reach the selected model, in drop order.
    pub dropped: Vec<FeatureSet>,
    /// Highest interaction order still present (`0` = intercept only, `1` = additive, up to `3`).
    pub effective_order: u8,
    /// The full backward path (index 0 = full model, last = intercept only).
    pub path: Vec<PrunePoint>,
    /// Per-table held-out contribution diagnostics.
    pub table_scores: Vec<PruneTableScore>,
    /// `full_mean_deviance − selected_mean_deviance` (positive ⇒ the prune improved held-out fit).
    pub delta_vs_full: f64,
}

/// A copy of `bank` retaining only tables whose feature-set is in `keep` (dense and factored drop
/// symmetrically). `f0` and `merged_grids` are preserved unchanged — a table integrates to zero
/// `w`-mass, so the intercept and the served mean are invariant under dropping tables.
#[must_use]
pub fn retain_tables(bank: &TableBank, keep: &[FeatureSet]) -> TableBank {
    let kept = |u: &FeatureSet| keep.iter().any(|k| k == u);
    TableBank {
        f0: bank.f0,
        tables: bank.tables.iter().filter(|t| kept(&t.u)).cloned().collect(),
        merged_grids: bank.merged_grids.clone(),
        w: bank.w.clone(),
        joint_variance: None,
        factored: bank
            .factored
            .iter()
            .filter(|ft| kept(&ft.u))
            .cloned()
            .collect(),
    }
}

/// `true` iff `sub` is a proper subset of `sup` (every member of `sub` is in `sup` and
/// `|sub| < |sup|`) — i.e. `sup` is a higher-order table containing `sub`.
fn is_proper_subset(sub: &FeatureSet, sup: &FeatureSet) -> bool {
    sub.order() < sup.order() && sub.0.iter().all(|f| sup.contains(*f))
}

// ---------------------------------------------------------------------------------------
// DEPLOYED-BOX BUDGET (2026-08-25)
//
// WHY IT EXISTS. The av37 evidence gate and its `keep_budget` count TABLES. That is the right
// unit for a dense bank and the wrong one for a factored one: a factored effect deploys one
// rank-1 REGION BOX per realized tree-region (merged across trees only where the per-axis
// masks agree exactly), and the rating export emits one row per box. Lifting `max_depth` from
// 3 to 6 leaves the table count almost still and multiplies the boxes-per-effect instead —
// measured on fremotor_payfreq split 0, the deployed order-3 effects go 101 -> 145 (1.4x)
// while their boxes-per-effect p50 goes 53 -> 817 (15x), for a bank of 7,358 -> 165,809 boxes.
// Selection had no term that could see that, so a depth champion won on skill while
// detonating the artifact.
//
// WHAT IT IS. A BANK-LEVEL budget on the deployed box total, spent greedily on the best
// evidence-per-box first. It is a filter over an ALREADY-CHOSEN keep-set, applied where the
// purified bank exists and its per-support box cost is therefore known exactly, so it is:
//
//   * EXACT — binary support dropping, exactly as the prune already does. No box is merged,
//     coarsened or approximated; the surviving bank is a purified bank of the same kind.
//   * MONOTONE / NO-OP — `max_boxes == 0`, or any budget at or above the bank's own box
//     total, returns the keep-set unchanged and therefore a bit-identical model. That is what
//     makes "budget off is byte-identical to order-lift HEAD" checkable rather than hopeful.
//   * FREE FOR DENSE BANKS — a dense table costs ZERO boxes, so a bank with no factored
//     effect (every `max_depth <= 3` order-<=2 fit, and any fit whose supports all went
//     dense) can never be touched at any budget.
//
// WHAT IT DOES NOT DO. It never re-admits. A support the greedy skipped stays out even if a
// later cascade frees room, and the cascade below only ever removes more. One pass, no
// fixed-point search, so the drop-set is a pure function of (keep, costs, evidence, budget).
// ---------------------------------------------------------------------------------------

/// What a deployed-box budget did to a keep-set. Serialized verbatim into the sklearn
/// estimator's `pruning_report_["box_budget"]`.
#[derive(Debug, Clone, Serialize)]
pub struct BoxBudgetReport {
    /// The budget in deployed boxes (`0` = disabled).
    pub max_boxes: u64,
    /// Deployed boxes the incoming keep-set carried.
    pub boxes_before: u64,
    /// Deployed boxes the outgoing keep-set carries.
    pub boxes_after: u64,
    /// Factored (box-carrying) effects before / after.
    pub effects_before: u64,
    /// Factored (box-carrying) effects before / after.
    pub effects_after: u64,
    /// `false` when the budget was disabled or already satisfied — then the keep-set is
    /// returned untouched and the deployed model is bit-identical to the unbudgeted one.
    pub engaged: bool,
    /// Supports the greedy could not afford, in the order it considered them (best
    /// evidence-per-box first, so the tail of this list is the weakest evidence).
    pub dropped: Vec<FeatureSet>,
    /// Supports removed afterwards to keep heredity: each properly contains a dropped one.
    pub cascade_dropped: Vec<FeatureSet>,
}

/// Per-support deployed BOX cost across one or more banks (one bank per class for a
/// multiclass fit — a filing reader faces every class's copy, so they add).
///
/// Dense tables contribute nothing: their size unit is cells, already governed by
/// `table_budget_cells` and the per-table firewall.
#[must_use]
pub fn box_costs(banks: &[&TableBank]) -> BTreeMap<FeatureSet, usize> {
    let mut out: BTreeMap<FeatureSet, usize> = BTreeMap::new();
    for bank in banks {
        for ft in &bank.factored {
            *out.entry(ft.u.clone()).or_insert(0) += ft.n_boxes();
        }
    }
    out
}

/// Per-support purified variance across the same banks — the SECONDARY rank key.
///
/// It is there because "no fold evidence" is the common case on a wide bank, not the exception
/// (measured: 74% of credit_g's candidates and the bulk of allstate_sev's are scored by no
/// prune fold at all), and every one of those supports carries the same `0.0` drop-gain. On
/// evidence alone the budget would then order thousands of effects by their raw feature ids —
/// i.e. arbitrarily. Variance is IN-SAMPLE and so can never be promoted over held-out
/// evidence, but as a tie-break "carries more purified mass per box" is a real statement and
/// feature-id order is not.
#[must_use]
pub fn box_variances(banks: &[&TableBank]) -> BTreeMap<FeatureSet, f64> {
    let mut out: BTreeMap<FeatureSet, f64> = BTreeMap::new();
    for bank in banks {
        for ft in &bank.factored {
            *out.entry(ft.u.clone()).or_insert(0.0) += ft.variance;
        }
    }
    out
}

/// Per-support purified variance across the same banks, DENSE EFFECTS INCLUDED — the
/// [`apply_table_budget`] tie-break.
///
/// Deliberately not [`box_variances`], which is the box budget's key and skips dense tables
/// because they cost zero boxes and are therefore never priced there. The TABLE budget prices
/// them: a dense 3-way table is a 3-way table in the filing whatever its storage. Reusing the
/// box map here would have silently sorted every dense candidate to the back of an
/// evidence-tied field, which on a wide bank is most of the field.
#[must_use]
pub fn table_variances(banks: &[&TableBank]) -> BTreeMap<FeatureSet, f64> {
    let mut out: BTreeMap<FeatureSet, f64> = BTreeMap::new();
    for bank in banks {
        for t in &bank.tables {
            *out.entry(t.u.clone()).or_insert(0.0) += t.variance;
        }
        for ft in &bank.factored {
            *out.entry(ft.u.clone()).or_insert(0.0) += ft.variance;
        }
    }
    out
}

/// Purified variance of every table in `model`'s full (unpruned) bank, built without the
/// ensemble-anchored verification gates — the ranking key of the ranked-path keep-set selector.
/// Same measure and per-row mass as the deployed bank.
pub fn full_bank_table_variances(
    model: &Model,
    serve: &ServeBinnedMatrix,
    w: &[f32],
    offset: Option<&[f32]>,
    w_measure: RefMeasure,
) -> Result<BTreeMap<FeatureSet, f64>, PbError> {
    let mass = measure_mass(&w_measure, w, offset);
    let bank = model.explain_bank_with_mass(
        serve,
        w_measure,
        crate::explain::TableBudget::default(),
        false,
        mass.as_deref(),
    )?;
    Ok(table_variances(&[&bank]))
}

/// Spend `max_boxes` deployed boxes on `keep`, best evidence-per-box first.
///
/// `evidence` is the per-support held-out drop-gain (higher = the held-out folds paid more to
/// lose it); a support absent from the map scores `0.0`. The rank key is, in order:
///
///   1. `evidence / boxes` — the marginal held-out deviance a box buys. Held-out, and primary.
///   2. `variance / boxes` — purified mass per box, from [`box_variances`]. In-sample, so it
///      only ever separates supports the held-out evidence could not — overwhelmingly, the
///      ones no prune fold ever scored.
///   3. the support's raw ids — a total, seed-free order, so the admitted prefix is a pure
///      function of the inputs and identical under any thread count.
///
/// Zero-cost supports (mains, pairs, any dense table) are ALWAYS kept: they are not what the
/// budget is protecting against, and dropping them would silently turn a readability knob into
/// a second selection rule.
///
/// Returns the surviving keep-set (sorted, deduplicated) and the report.
#[must_use]
pub fn apply_box_budget(
    keep: &[FeatureSet],
    costs: &BTreeMap<FeatureSet, usize>,
    variances: &BTreeMap<FeatureSet, f64>,
    max_boxes: usize,
    evidence: &BTreeMap<FeatureSet, f64>,
) -> (Vec<FeatureSet>, BoxBudgetReport) {
    let cost_of = |u: &FeatureSet| costs.get(u).copied().unwrap_or(0);
    let boxes_before: usize = keep.iter().map(cost_of).sum();
    let effects_before = keep.iter().filter(|u| cost_of(u) > 0).count();
    let idle = |engaged: bool| BoxBudgetReport {
        max_boxes: max_boxes as u64,
        boxes_before: boxes_before as u64,
        boxes_after: boxes_before as u64,
        effects_before: effects_before as u64,
        effects_after: effects_before as u64,
        engaged,
        dropped: Vec::new(),
        cascade_dropped: Vec::new(),
    };
    if max_boxes == 0 || boxes_before <= max_boxes {
        return (keep.to_vec(), idle(false));
    }

    let mut kept: std::collections::BTreeSet<FeatureSet> =
        keep.iter().filter(|u| cost_of(u) == 0).cloned().collect();
    let mut priced: Vec<(&FeatureSet, usize, f64, f64)> = keep
        .iter()
        .filter(|u| cost_of(u) > 0)
        .map(|u| {
            let c = cost_of(u);
            let g = evidence.get(u).copied().unwrap_or(0.0);
            let v = variances.get(u).copied().unwrap_or(0.0);
            // `c > 0` is guaranteed by the filter, so both densities are finite.
            (u, c, g / c as f64, v / c as f64)
        })
        .collect();
    priced.sort_by(|a, b| {
        b.2.total_cmp(&a.2)
            .then_with(|| b.3.total_cmp(&a.3))
            .then_with(|| a.0.cmp(b.0))
    });

    let mut spent = 0usize;
    let mut dropped: Vec<FeatureSet> = Vec::new();
    for (u, c, _, _) in &priced {
        // Keep scanning past an unaffordable support: a cheap, well-evidenced effect behind an
        // expensive one still deserves the room. (This is the standard density-greedy fill, so
        // the surviving set is not strictly monotone in the budget — a bigger budget can
        // swallow a large effect and squeeze out a small one. The report records exactly what
        // survived, so the frontier stays honest either way.)
        if spent + c <= max_boxes {
            spent += c;
            kept.insert((*u).clone());
        } else {
            dropped.push((*u).clone());
        }
    }

    // Heredity, relative to the keep-set we were handed: nothing may survive that properly
    // contains a support this budget removed. (An order-4 effect whose order-3 face was
    // dropped goes with it.) Cascade-removal only ever frees more boxes.
    let mut cascade_dropped: Vec<FeatureSet> = Vec::new();
    loop {
        let doomed: Vec<FeatureSet> = kept
            .iter()
            .filter(|u| {
                dropped
                    .iter()
                    .chain(cascade_dropped.iter())
                    .any(|d| is_proper_subset(d, u))
            })
            .cloned()
            .collect();
        if doomed.is_empty() {
            break;
        }
        for u in doomed {
            kept.remove(&u);
            cascade_dropped.push(u);
        }
    }

    let out: Vec<FeatureSet> = kept.into_iter().collect();
    let boxes_after: usize = out.iter().map(cost_of).sum();
    let effects_after = out.iter().filter(|u| cost_of(u) > 0).count();
    (
        out,
        BoxBudgetReport {
            max_boxes: max_boxes as u64,
            boxes_before: boxes_before as u64,
            boxes_after: boxes_after as u64,
            effects_before: effects_before as u64,
            effects_after: effects_after as u64,
            engaged: true,
            dropped,
            cascade_dropped,
        },
    )
}

// ---------------------------------------------------------------------------------------
// DEPLOYED MULTI-WAY TABLE BUDGET (2026-08-27)
//
// WHY IT EXISTS. Ralph moved the explainability bar: "in terms of resolution, perfectly happy
// with as many cells within tables to capture the correct resolution, it's just additional
// multi-way table COUNT that I'm keen on keeping to a minimum." Cells are free; the count of
// >=3-way tables is the thing a filing reader cannot absorb ("100 3-way tables is not
// explainable"). The deployed-BOX budget above prices the opposite quantity — resolution — and
// only bounds table count as a side effect of dropping supports binarily. Worse, it charges for
// exactly what Ralph now grants: on fremotor_payfreq the 20k-box rung bought its 132 three-way
// tables by starving them to ~129 boxes each, when the unbudgeted bank ran them at ~212.
//
// WHAT IT IS. A cap on the number of kept supports of arity >= `min_arity`, spent greedily on
// the best held-out evidence first. Same shape as `apply_box_budget` and the same guarantees:
//
//   * EXACT — binary support dropping. Surviving tables keep every box they had; the budget
//     never coarsens, merges or re-fits anything. That is the whole point: resolution is free.
//   * MONOTONE / NO-OP — `max_tables == 0`, or a cap at or above the incoming count, returns
//     the keep-set unchanged and therefore a bit-identical model.
//   * FREE BELOW THE FLOOR — a support of arity < `min_arity` is never counted and never
//     dropped. Mains and pairs are what a filing reads.
//
// WHY EVIDENCE AND NOT EVIDENCE-PER-COST. In the box budget the rank key is a DENSITY, because
// supports have different box costs. Here every priced support costs exactly one table, so the
// density collapses to the evidence itself — and that is the correct reading of the new bar: if
// I may print N three-way tables, print the N the held-out folds paid most to keep, whatever
// they cost in cells.
// ---------------------------------------------------------------------------------------

/// What a deployed multi-way table budget did to a keep-set. Serialized verbatim into the
/// sklearn estimator's `pruning_report_["table_budget"]`.
#[derive(Debug, Clone, Serialize)]
pub struct TableBudgetReport {
    /// The cap, in tables of arity `>= min_arity` (`0` = disabled).
    pub max_tables: u64,
    /// Lowest arity the cap counts.
    pub min_arity: u8,
    /// Tables at or above the floor that the incoming keep-set carried.
    pub tables_before: u64,
    /// Tables at or above the floor that the outgoing keep-set carries.
    pub tables_after: u64,
    /// Full arity histogram before / after, index `k-1` = arity `k` — so a report states the
    /// 2-way count Ralph asked to SEE alongside the >=3-way count the budget CAPS.
    pub arity_before: ArityCounts,
    /// Full arity histogram before / after, index `k-1` = arity `k`.
    pub arity_after: ArityCounts,
    /// `false` when the budget was disabled or already satisfied — then the keep-set is returned
    /// untouched and the deployed model is bit-identical to the unbudgeted one.
    pub engaged: bool,
    /// Supports the cap could not afford, in the order it considered them (best evidence first,
    /// so the tail of this list is the weakest evidence).
    pub dropped: Vec<FeatureSet>,
    /// Supports removed afterwards to keep heredity: each properly contains a dropped one.
    pub cascade_dropped: Vec<FeatureSet>,
}

/// Cap `keep` at `max_tables` supports of arity `>= min_arity`, best held-out evidence first.
///
/// `evidence` is the per-support held-out drop-gain (higher = the folds paid more to lose it); a
/// support absent from the map scores `0.0`, which ranks it below every positively-evidenced
/// support and above every harmful one. The rank key is, in order:
///
///   1. `evidence` — held-out, and primary.
///   2. `variance` — purified mass, from [`table_variances`] (dense effects included, unlike the
///      box budget's key). In-sample, so it only ever separates supports the held-out evidence
///      could not — on a wide bank that is most of them.
///   3. the support's raw ids — a total, seed-free order, so the admitted prefix is a pure
///      function of the inputs and identical under any thread count.
///
/// Returns the surviving keep-set (sorted, deduplicated) and the report.
#[must_use]
pub fn apply_table_budget(
    keep: &[FeatureSet],
    variances: &BTreeMap<FeatureSet, f64>,
    max_tables: usize,
    min_arity: u8,
    evidence: &BTreeMap<FeatureSet, f64>,
) -> (Vec<FeatureSet>, TableBudgetReport) {
    let floor = (min_arity as usize).clamp(1, MAX_ORDER);
    // Bounded ABOVE as well as below, so this predicate selects exactly the supports
    // `arity_counts_of` counted. The two disagreed for `order() > MAX_ORDER`: the histogram
    // drops those (they cannot exist, and folding one into a real bucket would corrupt the
    // count the bar is read from) while `>= floor` admitted them. `tables_before` would then
    // UNDER-count, so `tables_before <= max_tables` could hold with the real priced count over
    // the cap — the budget disengages and an over-cap bank ships, with a report claiming the cap
    // was met. Latent (the engine caps order at 8), and one bound away from not being.
    let priced_by = |u: &FeatureSet| (floor..=MAX_ORDER).contains(&u.order());
    let arity_before = arity_counts_of(keep.iter());
    // Read at the CLAMPED floor, so `priced.len() == tables_before` is an identity.
    let floor_u8 = floor as u8;
    let tables_before = n_tables_at_or_above(&arity_before, floor_u8);
    let idle = |engaged: bool| TableBudgetReport {
        max_tables: max_tables as u64,
        // The clamped floor, not the raw argument: a report has to describe what actually ran.
        min_arity: floor_u8,
        tables_before: tables_before as u64,
        tables_after: tables_before as u64,
        arity_before,
        arity_after: arity_before,
        engaged,
        dropped: Vec::new(),
        cascade_dropped: Vec::new(),
    };
    if max_tables == 0 || tables_before <= max_tables {
        return (keep.to_vec(), idle(false));
    }

    let mut kept: std::collections::BTreeSet<FeatureSet> =
        keep.iter().filter(|u| !priced_by(u)).cloned().collect();
    let mut priced: Vec<(&FeatureSet, f64, f64)> = keep
        .iter()
        .filter(|u| priced_by(u))
        .map(|u| {
            let g = evidence.get(u).copied().unwrap_or(0.0);
            let v = variances.get(u).copied().unwrap_or(0.0);
            (u, g, v)
        })
        .collect();
    priced.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| b.2.total_cmp(&a.2))
            .then_with(|| a.0.cmp(b.0))
    });

    // Every priced support costs exactly one, so the greedy is a prefix — no "keep scanning past
    // an unaffordable one" fill, and therefore (unlike the box budget) the surviving set IS
    // monotone in the cap.
    let mut dropped: Vec<FeatureSet> = Vec::new();
    for (i, (u, _, _)) in priced.iter().enumerate() {
        if i < max_tables {
            kept.insert((*u).clone());
        } else {
            dropped.push((*u).clone());
        }
    }

    // Heredity, relative to the keep-set we were handed: nothing may survive that properly
    // contains a support this budget removed. Cascade-removal only ever frees more room, and it
    // can only remove supports of HIGHER arity than something already dropped — so it can never
    // push the count back above the cap.
    let mut cascade_dropped: Vec<FeatureSet> = Vec::new();
    loop {
        let doomed: Vec<FeatureSet> = kept
            .iter()
            .filter(|u| {
                dropped
                    .iter()
                    .chain(cascade_dropped.iter())
                    .any(|d| is_proper_subset(d, u))
            })
            .cloned()
            .collect();
        if doomed.is_empty() {
            break;
        }
        for u in doomed {
            kept.remove(&u);
            cascade_dropped.push(u);
        }
    }

    let out: Vec<FeatureSet> = kept.into_iter().collect();
    let arity_after = arity_counts_of(out.iter());
    let tables_after = n_tables_at_or_above(&arity_after, floor_u8);
    (
        out,
        TableBudgetReport {
            max_tables: max_tables as u64,
            min_arity: floor_u8,
            tables_before: tables_before as u64,
            tables_after: tables_after as u64,
            arity_before,
            arity_after,
            engaged: true,
            dropped,
            cascade_dropped,
        },
    )
}

/// Re-state a [`BoxBudgetReport`]'s AFTER counts over the keep-set that actually deployed.
///
/// The box budget computes them from its own output, but the table budget runs afterwards and
/// removes more, so with both armed `boxes_after` / `effects_after` would overstate the deployed
/// bank. Report-only, but a calibration run reads exactly these numbers to pick the next lambda.
fn refresh_box_report_after(
    report: &mut BoxBudgetReport,
    deployed: &[FeatureSet],
    costs: &BTreeMap<FeatureSet, usize>,
) {
    let cost_of = |u: &FeatureSet| costs.get(u).copied().unwrap_or(0);
    report.boxes_after = deployed.iter().map(cost_of).sum::<usize>() as u64;
    report.effects_after = deployed.iter().filter(|u| cost_of(u) > 0).count() as u64;
}

/// Proper-subset DAG for the walks' maximality frontier: `counts[i]` = how many KEPT proper
/// supersets table i has (all tables start kept); `subsets_of[j]` = the tables j properly
/// contains, so a drop of j decrements exactly its subsets' counts. Maintained incrementally,
/// `counts[i] == 0` reproduces the former per-iteration `any(is_proper_subset)` rescan's
/// candidate set EXACTLY (same predicate, evaluated once up front) — the rescan was
/// O(tables²) bookkeeping per drop, which dominated walk time once deviance evaluations were
/// batched/lazy. One O(tables²) pass here replaces O(tables³) over a full walk.
/// Banks at or above this many tables run the lazy walk by default; below it the exhaustive
/// walk is bit-identical-cheap and the lazy sort/batch overhead is a net loss (battery-measured,
/// 2026-07-13). 1000 sits between the widest tied dataset (fremotor_prem, 596) and the narrowest
/// dataset where lazy pays (catelematic13, ~1.9k).
const LAZY_WALK_TABLE_THRESHOLD: usize = 1000;

/// The walk mode: lazy only for banks of `LAZY_WALK_TABLE_THRESHOLD`+ tables.
fn lazy_walk_engaged(n_tables: usize) -> bool {
    n_tables >= LAZY_WALK_TABLE_THRESHOLD
}

fn superset_frontier(ids: &[FeatureSet]) -> (Vec<u32>, Vec<Vec<u32>>) {
    let n = ids.len();
    let mut counts = vec![0_u32; n];
    let mut subsets_of: Vec<Vec<u32>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in 0..n {
            if j != i && is_proper_subset(&ids[i], &ids[j]) {
                counts[i] += 1;
                subsets_of[j].push(i as u32);
            }
        }
    }
    (counts, subsets_of)
}

/// Greedily prune `bank` to the held-out-deviance-optimal downward-closed keep-set.
///
/// Walks the backward path by repeatedly dropping the maximal table whose removal gives the best
/// held-out deviance, scores mean ± SE held-out deviance across `folds` at every waypoint, then
/// selects the simplest keep-set within `cfg.se_rule` SEs of the path minimum. Main effects are
/// sticky: they remain in the deployed model unless a caller explicitly omits them from a supplied
/// keep-set.
///
/// # Errors
/// [`PbError::ShapeMismatch`] if a fold's `y`/`w` length disagrees with its matrix; propagated
/// cell-map / table-eval / deviance errors. [`PbError::InvalidInput`] if `folds` is empty.
pub fn prune_bank(
    bank: &TableBank,
    cat_encoders: &crate::cat::CatEncoderStore,
    folds: &[HoldoutFold<'_>],
    loss: &dyn Loss,
    cfg: &PruneConfig,
) -> Result<(TableBank, PruneReport), PbError> {
    if folds.is_empty() {
        return Err(PbError::InvalidInput {
            what: "prune_bank requires at least one held-out fold".into(),
        });
    }

    // Combined table identity space: dense tables first, then factored triples.
    let ids: Vec<FeatureSet> = bank
        .tables
        .iter()
        .map(|t| t.u.clone())
        .chain(bank.factored.iter().map(|ft| ft.u.clone()))
        .collect();
    let variance: Vec<f64> = bank
        .tables
        .iter()
        .map(|t| t.variance)
        .chain(bank.factored.iter().map(|ft| ft.variance))
        .collect();
    let n_tables = ids.len();

    // Per-fold per-table row contributions, and a running raw score per fold (starts = full model).
    let t_contrib = std::time::Instant::now();
    let mut contrib: Vec<Vec<Vec<f64>>> = Vec::with_capacity(folds.len()); // [fold][table][row]
    let mut raw: Vec<Vec<f64>> = Vec::with_capacity(folds.len()); // [fold][row]
    let mut sum_w: Vec<f64> = Vec::with_capacity(folds.len());
    for fold in folds {
        let n_rows = fold.x.n_rows as usize;
        if fold.y.len() != n_rows || fold.w.len() != n_rows {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "prune fold: y={} w={} but matrix n_rows={n_rows}",
                    fold.y.len(),
                    fold.w.len()
                ),
            });
        }
        let maps = crate::scoring::build_cell_maps(&bank.merged_grids, cat_encoders, fold.x)?;
        let mut cells = vec![0u32; bank.merged_grids.len()];
        let mut fold_contrib = vec![vec![0.0_f64; n_rows]; n_tables];
        let mut fold_raw = vec![bank.f0; n_rows];
        for row in 0..n_rows {
            crate::scoring::fill_row_cells(fold.x, &maps, row, &mut cells)?;
            for (ti, t) in bank.tables.iter().enumerate() {
                let c = t.eval(&cells)?;
                fold_contrib[ti][row] = c;
                fold_raw[row] += c;
            }
            for (fi, ft) in bank.factored.iter().enumerate() {
                let c = ft.eval(&cells)?;
                fold_contrib[bank.tables.len() + fi][row] = c;
                fold_raw[row] += c;
            }
        }
        sum_w.push(fold.w.iter().map(|&w| f64::from(w)).sum());
        contrib.push(fold_contrib);
        raw.push(fold_raw);
    }
    prof_ms(t_contrib, "  contributions (fold scoring)");

    // Both closures also return the per-fold vector (not just its mean/SE) so the table-score
    // loop below can pair drop-fold against full-fold deviances fold-for-fold instead of
    // treating the two marginal SEs as independent (see se_gain below).
    let eval = |raw: &[Vec<f64>], loss: &dyn Loss| -> Result<(f64, f64, Vec<f64>), PbError> {
        let mut per_fold = Vec::with_capacity(folds.len());
        for (fi, fold) in folds.iter().enumerate() {
            let raw_f32: Vec<f32> = match fold.offset {
                Some(off) => raw[fi]
                    .iter()
                    .zip(off)
                    .map(|(&v, &o)| v as f32 + o)
                    .collect(),
                None => raw[fi].iter().map(|&v| v as f32).collect(),
            };
            let dev = f64::from(loss.deviance(fold.y, &raw_f32, fold.w)?);
            let denom = if sum_w[fi] > 0.0 { sum_w[fi] } else { 1.0 };
            per_fold.push(dev / denom);
        }
        let (mean, se) = mean_and_se(&per_fold);
        Ok((mean, se, per_fold))
    };
    let eval_without =
        |raw: &[Vec<f64>], table_idx: usize| -> Result<(f64, f64, Vec<f64>), PbError> {
            let mut per_fold = Vec::with_capacity(folds.len());
            for (fi, fold) in folds.iter().enumerate() {
                let raw_f32: Vec<f32> = match fold.offset {
                    Some(off) => raw[fi]
                        .iter()
                        .zip(&contrib[fi][table_idx])
                        .zip(off)
                        .map(|((&v, &c), &o)| (v - c) as f32 + o)
                        .collect(),
                    None => raw[fi]
                        .iter()
                        .zip(&contrib[fi][table_idx])
                        .map(|(&v, &c)| (v - c) as f32)
                        .collect(),
                };
                let dev = f64::from(loss.deviance(fold.y, &raw_f32, fold.w)?);
                let denom = if sum_w[fi] > 0.0 { sum_w[fi] } else { 1.0 };
                per_fold.push(dev / denom);
            }
            let (mean, se) = mean_and_se(&per_fold);
            Ok((mean, se, per_fold))
        };

    // Per-table deployed BOX cost, in the same index space as `ids`: dense tables come first and
    // cost nothing, then the factored effects, each costing its realized region-box count. This is
    // the identical convention `box_costs` uses for the deploy-time budget — one accounting, so a
    // selection-time price and a deploy-time cap can never disagree about what a bank costs.
    let box_cost: Vec<u32> = std::iter::repeat_n(0_u32, bank.tables.len())
        .chain(
            bank.factored
                .iter()
                .map(|ft| u32::try_from(ft.n_boxes()).unwrap_or(u32::MAX)),
        )
        .collect();
    let boxes_full: u32 = box_cost.iter().copied().fold(0_u32, u32::saturating_add);
    // Per-waypoint ARITY histogram, maintained by the same decrement discipline as `boxes_kept`.
    // Unlike the box cost, a DENSE table counts: it is a table in the filing whatever its
    // storage. `ids` is (dense .. factored), so one pass over it covers both.
    let mut arity_kept: ArityCounts = arity_counts_of(ids.iter());

    // Path point 0: the full model.
    let mut path: Vec<PrunePoint> = Vec::with_capacity(n_tables + 1);
    let (m0, s0, full_per_fold) = eval(&raw, loss)?;
    let mut boxes_kept = boxes_full;
    path.push(PrunePoint {
        n_tables: n_tables as u32,
        n_boxes: boxes_kept,
        n_tables_by_arity: arity_kept,
        dropped: None,
        mean_deviance: m0,
        se: s0,
    });

    // Per-table drop scores are independent reads of the shared `raw`/`contrib` state, so they
    // fan out on the ambient pool; `collect` preserves index order, so the vector is identical
    // to the former sequential loop's. Each table's exact full-model drop deviance rides along
    // to seed the lazy walk's cache (same value, not re-derived from mean_gain, so no fp drift).
    let scored: Vec<(PruneTableScore, f64)> = (0..n_tables)
        .into_par_iter()
        .map(|i| -> Result<(PruneTableScore, f64), PbError> {
            let (drop_m, _drop_s, drop_per_fold) = eval_without(&raw, i)?;
            // se_gain is the SE of the PAIRED per-fold difference, not sqrt(drop_s^2 + s0^2): the
            // drop-model and full-model deviances share the same folds and differ only by table i's
            // per-row contribution, so they are highly correlated — quadrature-summing their
            // marginal SEs as if independent grossly overstates the gain's uncertainty.
            let paired: Vec<f64> = drop_per_fold
                .iter()
                .zip(&full_per_fold)
                .map(|(&d, &f)| d - f)
                .collect();
            let (_, se_gain) = mean_and_se(&paired);
            Ok((
                PruneTableScore {
                    u: ids[i].clone(),
                    order: ids[i].order() as u8,
                    mean_gain: drop_m - m0,
                    se_gain,
                    variance: variance[i],
                    sticky: STICKY_MAIN_EFFECTS && ids[i].order() == 1,
                    selected: true,
                },
                drop_m,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut cached_m: Vec<f64> = scored.iter().map(|(_, m)| *m).collect();
    let mut table_scores: Vec<PruneTableScore> = scored.into_iter().map(|(ts, _)| ts).collect();

    // Greedy backward walk: drop the maximal table whose removal gives the best held-out deviance.
    //
    // The candidate evaluations each iteration are independent reads of the current `raw`, so
    // they fan out on the ambient pool (the O(candidates × held-rows) deviance scans dominated
    // wide fits: 135-430s per fold on homesite@100k). The winner is the lexicographic argmin
    // under (deviance `total_cmp`, FeatureSet order) — a total order over unique ids — so the
    // parallel reduction selects EXACTLY the sequential loop's table at every step and the
    // walk's drop sequence (hence the deployed model) is bit-identical.
    // Lazy (CELF-style) walk. Stale cached drop-deviances act as lower bounds
    // (near-monotonicity: dropping a table only hurts more as the model shrinks): candidates
    // re-evaluate in cached order, in FIXED-size batches, stopping once the best FRESH value
    // undercuts the next stale bound. Deterministic regardless of thread count (fixed sort
    // order, fixed batches, exact evals) and the winner's recorded deviance is always fresh —
    // but the SELECTION can differ from the exhaustive walk when a stale bound is optimistic.
    //
    // Default is AUTO by bank width (threshold: Ralph, 2026-07-13, from the 27-dataset paired
    // battery): below ~600 tables the lazy walk tied the exhaustive walk split-for-split on
    // 25/27 datasets (divergers beMTPL16 +0.0022% dev, fremotor_prem −0.0028% — noise) while its
    // sort/batch overhead actually cost ~6% wall; at 2k+ tables (homesite-class) it is ~2.7×
    // end-to-end. So: engage only where walks are wide enough to pay.
    let lazy_walk = lazy_walk_engaged(n_tables);
    const LAZY_BATCH: usize = 16;
    let arg_min = |a: (usize, f64, f64), b: (usize, f64, f64)| -> (usize, f64, f64) {
        match a.1.total_cmp(&b.1) {
            std::cmp::Ordering::Less => a,
            std::cmp::Ordering::Greater => b,
            std::cmp::Ordering::Equal => {
                if ids[a.0] < ids[b.0] {
                    a
                } else {
                    b
                }
            }
        }
    };
    let t_walk = std::time::Instant::now();
    let mut kept = vec![true; n_tables];
    let mut drops: Vec<FeatureSet> = Vec::with_capacity(n_tables);
    // Maximal ⇔ zero kept proper supersets — incrementally maintained (see `superset_frontier`);
    // identical candidate sets to the former per-iteration rescan.
    let (mut kept_superset_count, proper_subsets_of) = superset_frontier(&ids);
    loop {
        let candidates: Vec<usize> = (0..n_tables)
            .filter(|&i| {
                kept[i]
                    && !(STICKY_MAIN_EFFECTS && ids[i].order() == 1)
                    && kept_superset_count[i] == 0
            })
            .collect();
        let best: Option<(usize, f64, f64)> = if lazy_walk {
            let mut order = candidates.clone();
            order.sort_by(|&a, &b| {
                cached_m[a].total_cmp(&cached_m[b]).then_with(|| {
                    if ids[a] < ids[b] {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Greater
                    }
                })
            });
            let mut best: Option<(usize, f64, f64)> = None;
            let mut done = 0usize;
            while done < order.len() {
                let batch = &order[done..(done + LAZY_BATCH).min(order.len())];
                let evals: Vec<(usize, f64, f64)> = batch
                    .par_iter()
                    .map(|&i| -> Result<(usize, f64, f64), PbError> {
                        let (drop_m, drop_s, _) = eval_without(&raw, i)?;
                        Ok((i, drop_m, drop_s))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                for e in evals {
                    cached_m[e.0] = e.1;
                    best = Some(match best {
                        None => e,
                        Some(b) => arg_min(e, b),
                    });
                }
                done += batch.len();
                if let Some((_, bm, _)) = best {
                    if done >= order.len() || bm < cached_m[order[done]] {
                        break;
                    }
                }
            }
            best
        } else {
            candidates
                .par_iter()
                .map(|&i| -> Result<(usize, f64, f64), PbError> {
                    let (drop_m, drop_s, _) = eval_without(&raw, i)?;
                    Ok((i, drop_m, drop_s))
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .min_by(|a, b| {
                    a.1.total_cmp(&b.1).then_with(|| {
                        if ids[a.0] < ids[b.0] {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    })
                })
        };
        let Some((drop_idx, m, s)) = best else { break };
        kept[drop_idx] = false;
        for &i in &proper_subsets_of[drop_idx] {
            kept_superset_count[i as usize] -= 1; // exact by construction: one per kept superset
        }
        for fi in 0..folds.len() {
            for row in 0..raw[fi].len() {
                raw[fi][row] -= contrib[fi][drop_idx][row];
            }
        }
        drops.push(ids[drop_idx].clone());
        boxes_kept = boxes_kept.saturating_sub(box_cost[drop_idx]);
        decrement_arity(&mut arity_kept, &ids[drop_idx]);
        path.push(PrunePoint {
            n_tables: kept.iter().filter(|&&k| k).count() as u32,
            n_boxes: boxes_kept,
            n_tables_by_arity: arity_kept,
            dropped: Some(ids[drop_idx].clone()),
            mean_deviance: m,
            se: s,
        });
    }

    prof_ms(t_walk, "  greedy walk (deviance path)");

    // Select: simplest waypoint within `se_rule` SEs of the PENALIZED path minimum.
    //
    // The objective is `mean_deviance + lambda_boxes * n_boxes`, which is exactly
    // `mean_deviance` at the default `lambda_boxes = 0.0` (see `PruneConfig::objective`) — so
    // an unpenalized prune selects the same waypoint, bit for bit, as it did before this knob
    // existed. The band is still measured in SEs of the DEVIANCE, because the size term is a
    // stated price and not a random variable: widening the tie band by a deterministic quantity
    // would double-count it.
    let (min_i, min_obj) = path
        .iter()
        .enumerate()
        .map(|(i, p)| (i, cfg.objective(p)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .ok_or_else(|| PbError::Internal {
            what: "prune path unexpectedly empty".into(),
        })?;
    let threshold = min_obj + cfg.se_rule * path[min_i].se;
    // Fewest tables among the tied set (path is monotone in n_tables, so scan for the last within band).
    let sel_i = path
        .iter()
        .enumerate()
        .filter(|(_, p)| cfg.objective(p) <= threshold)
        .min_by_key(|(_, p)| p.n_tables)
        .map(|(i, _)| i)
        .unwrap_or(0);

    let n_drop = n_tables - path[sel_i].n_tables as usize;
    let dropped: Vec<FeatureSet> = drops[..n_drop].to_vec();
    let kept_ids: Vec<FeatureSet> = ids
        .iter()
        .filter(|u| !dropped.iter().any(|d| d == *u))
        .cloned()
        .collect();
    let effective_order = kept_ids.iter().map(|u| u.order() as u8).max().unwrap_or(0);
    for ts in &mut table_scores {
        ts.selected = kept_ids.iter().any(|u| u == &ts.u);
    }
    let report = PruneReport {
        kept: kept_ids.clone(),
        dropped,
        effective_order,
        delta_vs_full: path[0].mean_deviance - path[sel_i].mean_deviance,
        path,
        table_scores,
    };
    Ok((retain_tables(bank, &kept_ids), report))
}

/// Mean and standard error of the mean across folds. SE is `0` for a single fold.
fn mean_and_se(per_fold: &[f64]) -> (f64, f64) {
    let n = per_fold.len();
    let mean = per_fold.iter().sum::<f64>() / n as f64;
    if n < 2 {
        return (mean, 0.0);
    }
    let var = per_fold
        .iter()
        .map(|&v| (v - mean) * (v - mean))
        .sum::<f64>()
        / (n as f64 - 1.0);
    (mean, (var / n as f64).sqrt())
}

/// Emit a per-step prune timing to stderr when `TBOOST_PROFILE` is set (a no-op otherwise), mirroring
/// the fit-loop profiler. Pure measurement — never affects the pruned model or determinism.
fn prof_ms(t: std::time::Instant, label: &str) {
    if std::env::var_os("TBOOST_PROFILE").is_some() {
        eprintln!("[prune] {label}: {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
    }
}

/// Column-major row-subset of `x` (used to carve the held-out selection folds).
fn subset_binned(x: &BinnedMatrix, rows: &[usize]) -> Result<BinnedMatrix, PbError> {
    let mut data = Vec::with_capacity(x.data.len());
    for col in &x.data {
        let mut c = Vec::with_capacity(rows.len());
        for &r in rows {
            c.push(*col.get(r).ok_or_else(|| PbError::InvalidInput {
                what: format!("prune selection row {r} out of column range"),
            })?);
        }
        data.push(c);
    }
    Ok(BinnedMatrix {
        data,
        n_rows: rows.len() as u32,
        grids: x.grids.clone(),
        provenance: x.provenance.clone(),
    })
}

// -----------------------------------------------------------------------------------------
// PINNED-BANK FOLD FIDELITY (`prune_fold_fidelity`) — off by default, byte-identical when off.
//
// THE DEFECT. The av37 evidence gate (`aggregate_prune_selection`) judges every candidate
// support the DEPLOY bank realized, but the evidence it judges them on comes from
// `prune_n_folds` cheap fold refits that each SEARCH THEIR OWN STRUCTURE. A support the fold
// model never built contributes no `PruneTableScore`, so the aggregator sees `gain_n == 0`,
// imputes `mean_gain = 0.0`, and the legacy rule (`mean_gain > 0`) drops it — not on evidence
// but on ABSENCE. Measured on credit_g at the shipped b160 budget (30 splits x 2 seeds, 541
// candidates/fit): 438 per fit carry no fold evidence at all — order-2 105 of 175, order-3 332
// of 346.
//
// WHAT THIS IS. For each prune fold, after its own model is fit, the candidate supports its
// bank is MISSING are given per-fold values by a ridge on the purified cell basis
// (`cell_refit::fit_cell_correction` -> `explain::purify_correction` ->
// `explain::add_bank_assign`), fit ONLY on that fold's TRAIN rows (`fit_rows`; held-out rows
// carry IRLS weight 0, exactly as `rebalance_kept_cells` holds out its own slice). The fold
// bank then COVERS the candidate set, `prune_bank` scores a real drop-one gain for every
// candidate on the fold's held-out rows, and the aggregator gets `gain_n == k_folds` for all
// of them instead of 0. This is the "fold-refit fidelity" fix shape: structure from the
// deploy fit, VALUES re-estimated per fold, so the paired statistic exists everywhere.
//
// WHAT IT IS NOT. It is not the rejected "score the deploy bank post-hoc" design (see
// `_PRUNE_FOLD_BAGS_DEFAULT` in the sklearn wrapper), which imputes a drop-gain of exactly
// 0.0 for every fold that never built the table — a vector of zeros with mean 0 and SE 0 that
// the gate reads as "no evidence of harm => keep", i.e. no information at all. Here the
// augmented table carries a real surface fit on 1 - 1/k of the data and is scored on held-out
// rows it never saw, so a noise candidate measures a real negative gain with real scatter.
//
// LEAKAGE, STATED PLAINLY. The candidate SUPPORT SET was chosen by the full-data fit, which
// saw every fold's held-out rows; only the VALUES are honest. So the augmented evidence is
// selection-biased toward keep by roughly the 1/k share of the structure signal. That is why
// the ship decision is made on OUTER test skill (the battery), never on the inner evidence.
//
// COST. No extra boosting fit — the augmentation is one backfit solve per fold on top of the
// fold fit that already runs: 0.93-1.51x the deploy fit across the battery (credit_g 0.82s ->
// 1.24s; fremotor_prem at depth 6 / order 4, 800-table fold banks, 119s -> 123s). Contrast
// `prune_fold_bags` (the other fidelity design, measured on the `fold-fidelity` branch), which
// buys only PARTIAL coverage for 2.05-2.33x the whole fit.
//
// MEASURED, AND SHIPPED OFF. It closes the coverage defect completely (credit_g order-2
// 69.9/175.2 -> 175.2/175.2, order-3 13.4/345.7 -> 345.7/345.7) and the completed evidence
// then votes to DROP: -0.00376 (t=-3.38) against the shipped b160 arm, -0.00351 (t=-3.24) at
// MATCHED deployed size. The full numbers, the size-vs-evidence regression, the closure-pump
// mechanism and the order-4 limitation are in `_PRUNE_FOLD_FIDELITY_DEFAULT` (sklearn wrapper);
// the battery is insur-arena `scratchpad/fold_fidelity_battery`.

/// Ridge/design shape of the pinned-bank augmentation. Suite-wide (no per-dataset tuning).
#[derive(Debug, Clone, Copy)]
pub struct FoldFidelitySpec {
    /// Coarse cells per axis for an augmented table's ridge design (order >= 2). The
    /// augmented surface is deliberately COARSER than a boosted table: it exists to answer
    /// "does this support carry held-out signal on this fold", and at fold-train sizes a full
    /// merged grid on an order-3 support is pure noise. `cell_refit::CellRefitSpec::pair_cell_cap`.
    pub cell_cap: u32,
    /// Ridge prior in EQUIVALENT ROWS: the per-cell penalty is `ridge_rows * mean fit-row IRLS
    /// weight`, i.e. "shrink each cell as if this many average rows of zero residual were
    /// added". `CellRefitSpec::base` is an ABSOLUTE `col_w` penalty tuned for 100k-row
    /// datasets; used unscaled here it annihilates every cell on a 700-row fold and hands the
    /// gate a bank of exact zeros — the imputed-zero failure mode in disguise.
    pub ridge_rows: f64,
    /// Adaptive-ridge exponent (`CellRefitSpec::gamma`): high-signal terms are penalised less.
    pub gamma: f64,
    /// Skip a candidate whose MERGED cell count exceeds this: `correction_scaffold` allocates
    /// the correction at merged resolution, so a high-cardinality order-3 support can cost
    /// megabytes per table with hundreds of candidates in flight.
    pub max_merged_cells: usize,
}

impl Default for FoldFidelitySpec {
    fn default() -> Self {
        Self {
            cell_cap: 8,
            ridge_rows: 20.0,
            gamma: 2.0,
            max_merged_cells: 200_000,
        }
    }
}

/// Pin a fold model's bank to the deploy bank's candidate supports (see the module note above).
pub struct FoldFidelity<'a> {
    /// The candidate supports the SELECTION must judge — the deploy fit's realized supports,
    /// as raw feature ids. Order-1 entries are ignored (mains are sticky and never judged).
    pub candidate_supports: &'a [Vec<u32>],
    /// Per-`serve`-row mask of rows this fold may FIT on. Its held-out scoring rows MUST be
    /// `false` or the evidence is self-scored.
    pub fit_rows: &'a [bool],
    /// Ridge/design shape.
    pub spec: FoldFidelitySpec,
}

/// Report (under `TBOOST_PROFILE`) why a fold declined the candidate-bank augmentation. A
/// decline is never an error — the fold simply keeps the historical, structure-searched bank —
/// but it silently returns the fold to the coin-flip regime, so it must be visible.
fn ff_decline(why: &str) {
    if std::env::var_os("TBOOST_PROFILE").is_some() {
        eprintln!("[fold-fidelity] declined: {why}");
    }
}

/// Give `bank` a value-carrying table for every candidate support it is missing, estimated on
/// `ff.fit_rows` only. Returns how many supports were added (0 = the bank was already
/// complete, or a guard declined).
///
/// # Errors
/// Propagates the cell-refit solve, the correction purify, and the bank sum.
fn augment_bank_to_candidates(
    model: &Model,
    bank: &mut TableBank,
    serve: &ServeBinnedMatrix,
    y: &[f32],
    w: &[f32],
    offset: Option<&[f32]>,
    loss: &dyn Loss,
    w_measure: RefMeasure,
    ff: &FoldFidelity<'_>,
) -> Result<usize, PbError> {
    // Composing the augmentation on top of a fit-time cell correction is out of scope: the
    // correction is folded into the bank at decompose time, and a second `purify_correction`
    // over the same model would double-count it.
    if model.correction.is_some() {
        ff_decline("fit-time cell correction present");
        return Ok(0);
    }
    // Same multi-channel guard as `rebalance_kept_cells`: `correction_scaffold` reads its
    // `supports` entries as MODEL AXIS ids while the candidate supports (like `keep`) carry
    // raw FEATURE ids, which coincide only while every raw feature owns exactly one axis.
    if model.provenance.len() > crate::data::n_raw_features(&model.provenance) {
        ff_decline("multi-channel model (raw id != axis id)");
        return Ok(0);
    }
    let n_axes = model.provenance.len();
    // A factored effect lives in `bank.factored`, not `bank.tables`, while `add_bank_assign`
    // appends an unmatched delta table into `bank.tables` — so a candidate whose purify sweep
    // lands on a factored support would put the SAME support in the bank twice and
    // `prune_bank` would score it as two independent tables. `purify_correction` emits exactly
    // the DOWN-SET of the supports it is given, so it is enough to refuse a candidate that has
    // a factored support at or below it. (Refusing the whole bank instead — the first cut of
    // this guard — silently disabled the feature: on credit_g four of five prune folds carry a
    // factored order-3 effect, so fidelity engaged on ONE fold and the rest fell back to the
    // coin flip.)
    let factored_ids: Vec<Vec<u32>> = bank
        .factored
        .iter()
        .map(|ft| ft.u.0.iter().map(|f| f.0).collect::<Vec<u32>>())
        .collect();
    let realized: std::collections::BTreeSet<Vec<u32>> = bank
        .tables
        .iter()
        .map(|t| t.u.0.iter().map(|f| f.0).collect::<Vec<u32>>())
        .chain(factored_ids.iter().cloned())
        .collect();
    let merged_cells = |ids: &[u32]| -> Option<usize> {
        let mut prod = 1usize;
        for &a in ids {
            let g = bank.merged_grids.get(a as usize)?;
            prod = prod.checked_mul(usize::from(g.n_bins))?;
        }
        Some(prod)
    };
    let mut seen: std::collections::BTreeSet<Vec<u32>> = std::collections::BTreeSet::new();
    let mut missing: Vec<Vec<u32>> = Vec::new();
    for s in ff.candidate_supports {
        let mut ids = s.clone();
        ids.sort_unstable();
        ids.dedup();
        if ids.len() < 2 || ids.len() > MAX_ORDER {
            continue;
        }
        if ids.iter().any(|&a| a as usize >= n_axes) {
            continue;
        }
        if realized.contains(&ids) || !seen.insert(ids.clone()) {
            continue;
        }
        // See `factored_ids`: any factored support at or below this candidate would be
        // duplicated by the candidate's own purify sweep.
        if factored_ids
            .iter()
            .any(|f| f.iter().all(|a| ids.binary_search(a).is_ok()))
        {
            continue;
        }
        match merged_cells(&ids) {
            Some(c) if c <= ff.spec.max_merged_cells => missing.push(ids),
            _ => {}
        }
    }
    if missing.is_empty() {
        ff_decline("fold bank already covers every candidate");
        return Ok(0);
    }

    // Newton working residual z = -g/h and IRLS weight h at the CURRENT bank's raw score
    // (lossless against the ensemble, so this is the fold model's own prediction). Rows this
    // fold may not fit on carry weight 0, which `fit_cell_correction` compacts out exactly.
    let n = serve.0.n_rows as usize;
    if ff.fit_rows.len() != n {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "fold fidelity fit_rows len {} != serve rows {n}",
                ff.fit_rows.len()
            ),
        });
    }
    let mut base_raw = vec![0.0_f64; n];
    crate::scoring::score_bank_binned(bank, &model.schema.cat_encoders, &serve.0, &mut base_raw)?;
    let raw_f32: Vec<f32> = (0..n)
        .map(|r| {
            let o = offset.map_or(0.0_f32, |off| off.get(r).copied().unwrap_or(0.0));
            base_raw[r] as f32 + o
        })
        .collect();
    let mut gh = crate::loss::GradHess {
        g: vec![0.0_f32; n],
        h: vec![0.0_f32; n],
    };
    loss.grad_hess(y, &raw_f32, w, &mut gh)?;
    let mut residual = vec![0.0_f64; n];
    let mut weight = vec![0.0_f64; n];
    let mut w_sum = 0.0_f64;
    let mut w_n = 0_usize;
    for r in 0..n {
        if ff.fit_rows[r] {
            let h = f64::from(gh.h[r]).max(1e-12);
            residual[r] = -f64::from(gh.g[r]) / h;
            weight[r] = h;
            w_sum += h;
            w_n += 1;
        }
    }
    if w_n == 0 || w_sum <= 0.0 {
        ff_decline("no positive-weight fit rows");
        return Ok(0);
    }
    let spec = crate::cell_refit::CellRefitSpec {
        base: ff.spec.ridge_rows * (w_sum / w_n as f64),
        gamma: ff.spec.gamma,
        pair_cell_cap: ff.spec.cell_cap,
        ..crate::cell_refit::CellRefitSpec::default()
    };
    let corr = crate::cell_refit::fit_cell_correction(
        model,
        &serve.0.data,
        &residual,
        &weight,
        &missing,
        &spec,
    )?;
    let mass = measure_mass(&w_measure, w, offset);
    let delta = crate::explain::purify_correction(model, serve, w_measure, mass.as_deref(), &corr)?;
    crate::explain::add_bank_assign(bank, &delta)?;
    // `prune_bank` scores `tables` then `factored` in ONE identity space, so a support in both
    // would be scored (and budgeted, and voted on) twice. The candidate filter above is meant to
    // make that impossible; check it rather than trust it.
    let fac: std::collections::BTreeSet<&FeatureSet> = bank.factored.iter().map(|f| &f.u).collect();
    if let Some(dup) = bank.tables.iter().find(|t| fac.contains(&t.u)) {
        return Err(PbError::Internal {
            what: format!(
                "fold fidelity: support {:?} ended up both dense and factored",
                dup.u
            ),
        });
    }
    Ok(missing.len())
}

/// Prune a fitted `model` into a tables-only [`TableModel`], selecting the keep-set on a held-out set.
///
/// `sel_rows` index into `serve` (the training serve matrix) and MUST be rows the `model` did not
/// train on — the caller carves the train/select split (v1 orchestration lives at the Python layer).
/// The selection set is partitioned round-robin into `n_folds` sub-folds so the 1-SE rule has a
/// standard error. For a log or logit link the pruned `f0` is re-anchored so `Σwŷ == Σwy` over all
/// `serve` rows (an `f0`-only shift ⇒ exactness preserved).
///
/// # Errors
/// [`PbError`] from [`Model::explain`], the prune, or a malformed selection set / row index.
pub fn prune_model_to_tables(
    model: &Model,
    serve: &ServeBinnedMatrix,
    y: &[f32],
    w: &[f32],
    offset: Option<&[f32]>,
    loss: &dyn Loss,
    w_measure: RefMeasure,
    reanchor: bool,
    sel_rows: &[usize],
    n_folds: usize,
    cfg: &PruneConfig,
) -> Result<(TableModel, PruneReport), PbError> {
    prune_model_to_tables_pinned(
        model, serve, y, w, offset, loss, w_measure, reanchor, sel_rows, n_folds, cfg, None,
    )
}

/// [`prune_model_to_tables`] with optional PINNED-BANK FOLD FIDELITY: `fidelity = None` is
/// byte-for-byte the historical function; `Some(ff)` first gives the fold bank a value-carrying
/// table for every candidate support it is missing (fit on `ff.fit_rows` only) so the returned
/// report scores EVERY candidate instead of only the ones this fold's own structure search
/// happened to build. See the `FoldFidelity` note above.
///
/// # Errors
/// [`PbError`] from [`Model::explain`], the augmentation, the prune, or a malformed selection set.
pub fn prune_model_to_tables_pinned(
    model: &Model,
    serve: &ServeBinnedMatrix,
    y: &[f32],
    w: &[f32],
    offset: Option<&[f32]>,
    loss: &dyn Loss,
    w_measure: RefMeasure,
    reanchor: bool,
    sel_rows: &[usize],
    n_folds: usize,
    cfg: &PruneConfig,
    fidelity: Option<&FoldFidelity<'_>>,
) -> Result<(TableModel, PruneReport), PbError> {
    if sel_rows.is_empty() {
        return Err(PbError::InvalidInput {
            what: "prune_model_to_tables requires a non-empty held-out selection set".into(),
        });
    }
    // Ungated bank build: skip the ensemble-anchored gates (they re-prove the discarded ensemble and
    // dominate `explain`). The pruned bank is re-checked against its own LUT-sum instead.
    let t_explain = std::time::Instant::now();
    let mass = measure_mass(&w_measure, w, offset);
    let mut bank = model.explain_bank_with_mass(
        serve,
        w_measure.clone(),
        crate::explain::TableBudget::default(),
        false,
        mass.as_deref(),
    )?;
    prof_ms(t_explain, "explain (bank build, ungated)");

    // Pinned-bank fold fidelity: default OFF (`fidelity = None`), and then this is a no-op and
    // `bank` is exactly what it always was.
    if let Some(ff) = fidelity {
        let t_ff = std::time::Instant::now();
        let added =
            augment_bank_to_candidates(model, &mut bank, serve, y, w, offset, loss, w_measure, ff)?;
        prof_ms(t_ff, "fold fidelity (candidate-bank augment)");
        if std::env::var_os("TBOOST_PROFILE").is_some() {
            eprintln!(
                "[fold-fidelity] augmented {added} candidate supports (bank now {} tables)",
                bank.tables.len()
            );
        }
    }
    let bank = bank;

    // Partition the selection rows round-robin into >=1-row sub-folds (for the 1-SE standard error).
    let n_folds = n_folds.clamp(1, sel_rows.len());
    let mut fold_rows: Vec<Vec<usize>> = vec![Vec::new(); n_folds];
    for (i, &r) in sel_rows.iter().enumerate() {
        fold_rows[i % n_folds].push(r);
    }
    let mut fold_mats = Vec::with_capacity(n_folds);
    let mut fold_ys: Vec<Vec<f32>> = Vec::with_capacity(n_folds);
    let mut fold_ws: Vec<Vec<f32>> = Vec::with_capacity(n_folds);
    let mut fold_offsets: Vec<Option<Vec<f32>>> = Vec::with_capacity(n_folds);
    for rows in &fold_rows {
        fold_mats.push(subset_binned(&serve.0, rows)?);
        let mut fy = Vec::with_capacity(rows.len());
        let mut fw = Vec::with_capacity(rows.len());
        for &r in rows {
            fy.push(*y.get(r).ok_or_else(|| PbError::InvalidInput {
                what: format!("selection row {r} out of y range"),
            })?);
            fw.push(*w.get(r).ok_or_else(|| PbError::InvalidInput {
                what: format!("selection row {r} out of w range"),
            })?);
        }
        let fo = match offset {
            Some(off) => {
                let mut v = Vec::with_capacity(rows.len());
                for &r in rows {
                    v.push(*off.get(r).ok_or_else(|| PbError::InvalidInput {
                        what: format!("selection row {r} out of offset range"),
                    })?);
                }
                Some(v)
            }
            None => None,
        };
        fold_ys.push(fy);
        fold_ws.push(fw);
        fold_offsets.push(fo);
    }
    let folds: Vec<HoldoutFold<'_>> = (0..n_folds)
        .map(|i| HoldoutFold {
            x: &fold_mats[i],
            y: &fold_ys[i],
            w: &fold_ws[i],
            offset: fold_offsets[i].as_deref(),
        })
        .collect();

    let t_prune = std::time::Instant::now();
    let (mut pruned, report) = prune_bank(&bank, &model.schema.cat_encoders, &folds, loss, cfg)?;
    prof_ms(t_prune, "prune_bank (select)");

    // Re-anchor the pruned function (Log/Logit only, and only when the fit used reanchor):
    // dropping tables shifts the exposure-weighted aggregate, a first-order deviance penalty
    // either link realizes. Log keeps its closed-form `δ = ln(Σwy/Σwμ̂)` path (`reanchor_log_link`,
    // untouched by the Logit port below — bit-identical to before it). Logit has no closed form,
    // so it routes through the shared `boost::reanchor_delta` bisection (`reanchor_logit_link`).
    if reanchor {
        match loss.link() {
            Link::Log => {
                let t_re = std::time::Instant::now();
                reanchor_log_link(
                    &mut pruned,
                    &model.schema.cat_encoders,
                    serve,
                    y,
                    w,
                    offset,
                    loss,
                )?;
                prof_ms(t_re, "reanchor");
            }
            Link::Logit => {
                let t_re = std::time::Instant::now();
                reanchor_logit_link(&mut pruned, &model.schema.cat_encoders, serve, y, w, offset)?;
                prof_ms(t_re, "reanchor");
            }
            Link::Identity => {}
        }
    }

    let tm = TableModel::from_model_and_bank(model, pruned);
    tm.validate()?;
    Ok((tm, report))
}

/// Apply an already-chosen downward-closed `keep`-set to a (possibly differently-fit) `model`,
/// producing a pruned tables-only [`TableModel`] — no selection, just retain, re-anchor, and
/// optionally rebalance the fixed support. The sklearn wrapper selects this keep-set by aggregating
/// held-out fold reports, then applies it to the full-data fit. Retaining a downward-closed keep-set
/// from the full bank stays downward-closed; any support the full fit realized but the selection
/// didn't choose is dropped.
///
/// # Errors
/// [`PbError`] from [`Model::explain_bank`], the re-anchor, or [`TableModel::validate`].
pub fn prune_model_to_keepset(
    model: &Model,
    serve: &ServeBinnedMatrix,
    y: &[f32],
    w: &[f32],
    offset: Option<&[f32]>,
    loss: &dyn Loss,
    w_measure: RefMeasure,
    reanchor: bool,
    keep: &[FeatureSet],
    rebalance: bool,
) -> Result<TableModel, PbError> {
    let (tm, _, _) = prune_model_to_keepset_budgeted(
        model,
        serve,
        y,
        w,
        offset,
        loss,
        w_measure,
        reanchor,
        keep,
        rebalance,
        0,
        0,
        DEFAULT_TABLE_MIN_ARITY,
        &BTreeMap::new(),
    )?;
    Ok(tm)
}

/// [`prune_model_to_keepset`] with a DEPLOYED-BOX BUDGET applied to `keep` first (see
/// [`apply_box_budget`]). The budget is spent here, and only here, because this is where the
/// purified bank exists and each support's exact box cost is therefore known — the box count
/// of a support is a property of the model's own trees, not of the keep-set, so retaining a
/// budgeted subset is the same bank minus whole effects.
///
/// `max_boxes == 0` disables it and reproduces [`prune_model_to_keepset`] bit-for-bit.
///
/// # Errors
/// [`PbError`] from [`Model::explain_bank`], the re-anchor, or [`TableModel::validate`].
pub fn prune_model_to_keepset_budgeted(
    model: &Model,
    serve: &ServeBinnedMatrix,
    y: &[f32],
    w: &[f32],
    offset: Option<&[f32]>,
    loss: &dyn Loss,
    w_measure: RefMeasure,
    reanchor: bool,
    keep: &[FeatureSet],
    rebalance: bool,
    max_boxes: usize,
    max_tables: usize,
    table_min_arity: u8,
    evidence: &BTreeMap<FeatureSet, f64>,
) -> Result<(TableModel, BoxBudgetReport, TableBudgetReport), PbError> {
    let prof = std::env::var_os("TBOOST_PROFILE").is_some();
    let mass = measure_mass(&w_measure, w, offset);
    let bank = model.explain_bank_with_mass(
        serve,
        w_measure.clone(),
        crate::explain::TableBudget::default(),
        false,
        mass.as_deref(),
    )?;
    let box_cost_map = box_costs(&[&bank]);
    let (budgeted, mut box_report) = apply_box_budget(
        keep,
        &box_cost_map,
        &box_variances(&[&bank]),
        max_boxes,
        evidence,
    );
    // Composed, box budget FIRST: each is a no-op at 0, so an unbudgeted fit reaches
    // `retain_tables` with the keep-set it was handed, byte-for-byte as before. Order matters
    // only when both are armed, and this order is the honest one — the box budget is a
    // resolution cap and the table budget is a structure cap, so the structure cap should see
    // whatever survived the resolution cap and cap THAT, not the other way round.
    let (budgeted, table_report) = apply_table_budget(
        &budgeted,
        &table_variances(&[&bank]),
        max_tables,
        table_min_arity,
        evidence,
    );
    refresh_box_report_after(&mut box_report, &budgeted, &box_cost_map);
    // Below this line `keep` means the BUDGETED keep-set: the rebalance re-solve is over the
    // surviving support, exactly as it is over the selected support without a budget.
    let keep: &[FeatureSet] = &budgeted;
    let mut pruned = retain_tables(&bank, keep);

    // Score the pruned bank ONCE and reuse it for BOTH the re-anchor and (if requested) the
    // rebalance working residual.
    let n = serve.0.n_rows as usize;
    let mut base_raw = vec![0.0_f64; n];
    crate::scoring::score_bank_binned(
        &pruned,
        &model.schema.cat_encoders,
        &serve.0,
        &mut base_raw,
    )?;
    let do_reanchor = reanchor && matches!(loss.link(), Link::Log | Link::Logit);
    if do_reanchor {
        let shift = match loss.link() {
            Link::Log => reanchor_shift(&base_raw, y, w, offset, loss)?,
            Link::Logit => {
                let raw32: Vec<f32> = base_raw.iter().map(|&v| v as f32).collect();
                crate::engine::boost::reanchor_delta(Link::Logit, y, w, &raw32, offset)?
            }
            Link::Identity => 0.0,
        };
        pruned.f0 += shift;
        for v in base_raw.iter_mut() {
            *v += shift;
        }
    }
    if prof {
        eprintln!(
            "[prune] keepset+retain+score+reanchor (kept {}/{} tables)",
            keep.len(),
            bank.tables.len(),
        );
    }
    if rebalance {
        // Post-prune re-solve: recalibrate the surviving tables' cell values to the
        // deviance-optimum of the reduced structure (the keep-set is fixed — this only
        // touches the numbers, never re-selects). No-harm guarded on a held-out slice.
        let t = std::time::Instant::now();
        pruned = rebalance_kept_cells(
            model,
            pruned,
            &base_raw,
            keep,
            serve,
            y,
            w,
            offset,
            loss,
            w_measure,
            do_reanchor,
        )?;
        if prof {
            eprintln!("[prune] rebalance TOTAL {:.2}s", t.elapsed().as_secs_f64());
        }
    }
    let tm = TableModel::from_model_and_bank(model, pruned);
    tm.validate()?;
    Ok((tm, box_report, table_report))
}

/// Per-bag purified banks restricted to `keep`, on the SAME serve matrix the deployed bank
/// uses — the honest replicate values behind the outer-bag soup. Each bag's trees are
/// rescaled by `n_bags` (undoing the soup's `1/n_bags` member weight), so every returned
/// bank is that bag's standalone-scale effect surface; their `n_bags`-weighted mean
/// reproduces the soup bank's tables to floating-point tolerance (purify is linear in
/// trees at a FIXED weight/grid, Lengerich Cor. 2.2, spec §08.3 — see below for how that
/// precondition is actually met).
///
/// Every bag is purified on the SOUP's own merged grid and reference-measure weights
/// (built once from the full `model`, via [`crate::explain::MergedGrids::from_model`] and
/// [`Model::explain_bank_with_grids`]), not each bag's own bag-specific grid. This is what
/// makes the linearity property above hold: a bag whose trees realize only a SUBSET of the
/// soup's split borders still gets accumulated onto the full soup grid (a coarser leaf's
/// value is correctly replicated across every finer merged cell its region covers), so
/// every returned bank shares the deployed bank's table shapes and reference measure
/// exactly — no grid-refinement or measure-asymmetry gap to tolerate. A bag whose trees
/// realize none of a kept support simply omits that table — the caller must treat absence
/// as an all-zero replicate.
///
/// Requires the runtime bag partition recorded by the outer-bag soup (`Model::bag_spans`,
/// absent on single fits and deserialized models) and no §G1 cell correction (soup-level;
/// not attributable to a bag).
///
/// # Errors
/// [`PbError::InvalidInput`] if the model has no bag partition or carries a cell
/// correction; propagated bank-construction errors otherwise.
pub fn bag_banks_for_keepset(
    model: &Model,
    serve: &ServeBinnedMatrix,
    w_measure: RefMeasure,
    mass: Option<&[f32]>,
    keep: &[FeatureSet],
) -> Result<Vec<TableBank>, PbError> {
    let Some(spans) = model.bag_spans.as_ref() else {
        return Err(PbError::InvalidInput {
            what: "per-bag banks need the outer-bag soup's runtime bag partition \
                   (single fit or deserialized model?)"
                .into(),
        });
    };
    if model.correction.is_some() {
        return Err(PbError::InvalidInput {
            what: "per-bag banks cannot attribute a §G1 cell correction to a bag".into(),
        });
    }
    let soup_grids = crate::explain::MergedGrids::from_model(model)?;
    spans
        .iter()
        .enumerate()
        .map(|(bag, _)| {
            let bank = bag_bank(model, serve, &w_measure, mass, &soup_grids, bag, false)?;
            Ok(retain_tables(&bank, keep))
        })
        .collect()
}

/// The `bag`-th member's standalone-scale [`Model`] view: its own slice of the soup's `trees`
/// with every alpha multiplied by `n_bags` (undoing the soup's `1/n_bags` member weight), the
/// member's original `f0`, and the soup's shared grids/schema. Legacy models without
/// runtime member intercepts fall back to the soup's `f0`. The weighted mean over bags of
/// this view reproduces the soup itself.
///
/// `share_correction` decides what happens to a §G1 cell correction. It is a SOUP-level object
/// with no bag attribution, so:
///   * `false` (the replicate contract — [`bag_banks_for_keepset`], [`bag_raw_scores_for_rows`])
///     drops it, and those entry points refuse a corrected model outright rather than return
///     replicates that silently omit part of the model.
///   * `true` (the out-of-bag EVIDENCE path) gives every bag the WHOLE correction, unscaled.
///     That is the arithmetically right thing for the only quantity the evidence uses — the
///     mean over a row's jury — because purify is linear: mean_b purify(f0 + n_bags·trees_b +
///     corr) = f0 + purify(soup trees) + purify(corr), which is the soup's own bank. An
///     individual bag's bank is then NOT a coherent standalone model (trees at standalone
///     scale, correction at soup scale), which is exactly why the replicate API must not use it.
fn bag_member_model(
    model: &Model,
    spans: &[(u32, u32)],
    bag: usize,
    share_correction: bool,
) -> Result<Model, PbError> {
    let n_bags = spans.len() as f64;
    let &(start, end) = spans.get(bag).ok_or_else(|| PbError::Internal {
        what: format!("bag {bag} escapes {} bag spans", spans.len()),
    })?;
    let (s, e) = (start as usize, end as usize);
    let slice = model.trees.get(s..e).ok_or_else(|| PbError::Internal {
        what: format!(
            "bag span {start}..{end} escapes {} trees",
            model.trees.len()
        ),
    })?;
    let trees: Vec<_> = slice
        .iter()
        .map(|(alpha, tree)| ((f64::from(*alpha) * n_bags) as f32, tree.clone()))
        .collect();
    Ok(Model {
        f0: model
            .bag_intercepts
            .as_ref()
            .and_then(|values| values.get(bag))
            .copied()
            .unwrap_or(model.f0),
        trees,
        grids: model.grids.clone(),
        provenance: model.provenance.clone(),
        link: model.link,
        mode: model.mode.clone(),
        schema: model.schema.clone(),
        schema_version: model.schema_version,
        correction: if share_correction {
            model.correction.clone()
        } else {
            None
        },
        bag_spans: None,
        bag_intercepts: None,
        bag_in_bag: None,
        delta_step_gate: None,
    })
}

/// One bag's purified bank over ALL its realized supports, on the SOUP's merged grid and
/// reference measure (see [`bag_banks_for_keepset`] for why that shared grid is what makes the
/// mean-of-bags identity hold).
fn bag_bank(
    model: &Model,
    serve: &ServeBinnedMatrix,
    w_measure: &RefMeasure,
    mass: Option<&[f32]>,
    soup_grids: &crate::explain::MergedGrids,
    bag: usize,
    share_correction: bool,
) -> Result<TableBank, PbError> {
    let spans = model
        .bag_spans
        .as_ref()
        .ok_or_else(|| PbError::InvalidInput {
            what: "per-bag banks need the outer-bag soup's runtime bag partition \
               (single fit or deserialized model?)"
                .into(),
        })?;
    let bag_model = bag_member_model(model, spans, bag, share_correction)?;
    bag_model.explain_bank_with_grids_and_mass(
        serve,
        w_measure.clone(),
        crate::explain::TableBudget::default(),
        false,
        soup_grids,
        mass,
    )
}

/// Raw (link-scale) scores of every bag's keep-restricted bank on `rows` — the per-bag raw
/// scorer behind the prune guard's out-of-bag evidence, and the general "score this bank
/// restricted to a table keep-set on these rows" primitive.
///
/// `keep = None` scores the FULL bag bank (every realized table, which reproduces that bag's
/// tree ensemble losslessly); `Some(k)` scores only the tables whose support is in `k`, with
/// the bank intercept still included. Returns one `rows.len()`-long vector per bag, in bag
/// order. `rows` indexes the `serve` matrix.
///
/// Deterministic and thread-count independent: bags are built in order and each row writes
/// only its own slot.
///
/// # Errors
/// [`PbError::InvalidInput`] if the model has no bag partition, carries a §G1 cell correction,
/// or a row index escapes `serve`; propagated bank-construction/scoring errors otherwise.
pub fn bag_raw_scores_for_rows(
    model: &Model,
    serve: &ServeBinnedMatrix,
    w_measure: RefMeasure,
    mass: Option<&[f32]>,
    keep: Option<&[FeatureSet]>,
    rows: &[u32],
) -> Result<Vec<Vec<f64>>, PbError> {
    let Some(spans) = model.bag_spans.as_ref() else {
        return Err(PbError::InvalidInput {
            what: "per-bag raw scores need the outer-bag soup's runtime bag partition \
                   (single fit or deserialized model?)"
                .into(),
        });
    };
    if model.correction.is_some() {
        return Err(PbError::InvalidInput {
            what: "per-bag raw scores cannot attribute a §G1 cell correction to a bag".into(),
        });
    }
    let soup_grids = crate::explain::MergedGrids::from_model(model)?;
    let n_bags = spans.len();
    let mut out = Vec::with_capacity(n_bags);
    for bag in 0..n_bags {
        // Sequential over bags on purpose: one bank is live at a time, so peak memory stays at
        // a single bank instead of `n_bags` of them (a wide fit's bank is the big object here).
        // `explain_bank_with_grids` and `score_banks_rows` are each internally parallel.
        let full = bag_bank(model, serve, &w_measure, mass, &soup_grids, bag, false)?;
        let bank = match keep {
            Some(k) => retain_tables(&full, k),
            None => full,
        };
        let mut flat = vec![0.0_f64; rows.len()];
        crate::scoring::score_banks_rows(
            &[&bank],
            &model.schema.cat_encoders,
            &serve.0,
            rows,
            &mut flat,
        )?;
        out.push(flat);
    }
    Ok(out)
}

/// Per-row sample variance of the bag banks' scores, restricted to `keep` — the soup's noise at
/// each row (divide by the bag count for the variance of the soup's mean). Unlike
/// [`bag_raw_scores_for_rows`] this accepts a §G1 cell-corrected model: the correction is one
/// shared term added to every bag, so it cancels exactly in the between-bag spread.
///
/// # Errors
/// [`PbError::InvalidInput`] without the soup's runtime bag partition (fewer than two bags).
pub fn bag_score_variance_for_rows(
    model: &Model,
    serve: &ServeBinnedMatrix,
    w_measure: RefMeasure,
    mass: Option<&[f32]>,
    keep: Option<&[FeatureSet]>,
    rows: &[u32],
) -> Result<Vec<f64>, PbError> {
    let Some(spans) = model.bag_spans.as_ref() else {
        return Err(PbError::InvalidInput {
            what: "bag noise needs the outer-bag soup's runtime bag partition".into(),
        });
    };
    let n_bags = spans.len();
    if n_bags < 2 {
        return Err(PbError::InvalidInput {
            what: "bag noise needs at least two bags".into(),
        });
    }
    let soup_grids = crate::explain::MergedGrids::from_model(model)?;
    let mut sum = vec![0.0_f64; rows.len()];
    let mut sq = vec![0.0_f64; rows.len()];
    for bag in 0..n_bags {
        let full = bag_bank(model, serve, &w_measure, mass, &soup_grids, bag, false)?;
        let bank = match keep {
            Some(k) => retain_tables(&full, k),
            None => full,
        };
        let mut flat = vec![0.0_f64; rows.len()];
        crate::scoring::score_banks_rows(
            &[&bank],
            &model.schema.cat_encoders,
            &serve.0,
            rows,
            &mut flat,
        )?;
        for ((s, q), v) in sum.iter_mut().zip(sq.iter_mut()).zip(&flat) {
            *s += v;
            *q += v * v;
        }
    }
    let b = n_bags as f64;
    Ok(sum
        .iter()
        .zip(&sq)
        .map(|(s, q)| ((q - s * s / b) / (b - 1.0)).max(0.0))
        .collect())
}

/// The prune guard's out-of-bag evidence, accumulated in ONE pass over the per-bag banks.
///
/// For every fit row `r`, the bags that never trained on `r` (`!Model::bag_in_bag[b][r]`) are
/// its honest jury. This returns, per row, the sums over exactly that jury of
///   * `full_sum`: the FULL bag bank's raw score (`f0_b + Σ_all f_u^b(x_r)`) — the "no pruning
///     at all" arm, lossless against that bag's tree ensemble;
///   * `f0_sum`: the bag bank intercepts alone;
///   * `group_sums[g]`: the tables of table-group `g` ONLY (no intercept),
///
/// plus `counts[r]` = the jury size. A caller reconstructs any prefix union of the groups as
/// `(f0_sum[r] + Σ_{g ≤ k} group_sums[g][r]) / counts[r]`, which is why the guard can pass its
/// whole re-admission ladder (keep-set, then each doubling chunk, in rank order) as groups and
/// evaluate every rung from a single pass — table scores are ADDITIVE, so no rung needs its own
/// bank build. Rows with `counts[r] == 0` (no bag left them out) carry zeros and must be
/// dropped by the caller.
///
/// Why the mean over the jury is the right estimator: each bag bank is that bag's standalone-
/// scale effect surface, and the mean over ALL bags reproduces the soup's deployed bank
/// (purify is linear at a fixed grid/measure — see [`bag_banks_for_keepset`]). The jury mean is
/// therefore an unbiased, honest estimate of the deployed artifact's score at `r`, computed on
/// rows the contributing bags never saw. This is the same out-of-bag ensemble estimator the
/// §G1 cell refit's no-harm guard already uses on tree scores (`attach_cell_correction`).
///
/// # Errors
/// [`PbError::InvalidInput`] if the model has no bag partition or membership;
/// [`PbError::ShapeMismatch`] if `serve` is not the fit design (its row count
/// must equal the recorded membership length); propagated bank/scoring errors otherwise.
pub fn bag_oob_group_raw_sums(
    model: &Model,
    serve: &ServeBinnedMatrix,
    w_measure: RefMeasure,
    mass: Option<&[f32]>,
    groups: &[Vec<FeatureSet>],
) -> Result<BagOobGroupSums, PbError> {
    bag_oob_group_sums_with(model, serve, w_measure, mass, groups, true)
}

/// [`bag_oob_group_raw_sums`], optionally without the full-bank arm: `with_full = false` leaves
/// `full_sum` at zero and skips scoring each bag's full bank — half the scoring work — for a
/// caller that reads only the intercept and group sums (the ranked path). Every other output is
/// bit-identical to `with_full = true` (each bank is scored on its own).
///
/// # Errors
/// As [`bag_oob_group_raw_sums`].
pub fn bag_oob_group_sums_with(
    model: &Model,
    serve: &ServeBinnedMatrix,
    w_measure: RefMeasure,
    mass: Option<&[f32]>,
    groups: &[Vec<FeatureSet>],
    with_full: bool,
) -> Result<BagOobGroupSums, PbError> {
    let Some(spans) = model.bag_spans.as_ref() else {
        return Err(PbError::InvalidInput {
            what: "out-of-bag evidence needs the outer-bag soup's runtime bag partition \
                   (single fit or deserialized model?)"
                .into(),
        });
    };
    let Some(membership) = model.bag_in_bag.as_ref() else {
        return Err(PbError::InvalidInput {
            what: "out-of-bag evidence needs the outer-bag soup's runtime bag membership \
                   (single fit or deserialized model?)"
                .into(),
        });
    };
    if membership.len() != spans.len() {
        return Err(PbError::Internal {
            what: format!(
                "bag membership has {} bags but the partition has {}",
                membership.len(),
                spans.len()
            ),
        });
    }
    let n = serve.0.n_rows as usize;
    for (b, m) in membership.iter().enumerate() {
        if m.len() != n {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "out-of-bag evidence must be scored on the FIT design: bag {b} membership \
                     covers {} rows but serve has {n}",
                    m.len()
                ),
            });
        }
    }
    let n_groups = groups.len();
    let mut acc = BagOobGroupSums {
        full_sum: vec![0.0_f64; n],
        f0_sum: vec![0.0_f64; n],
        group_sums: vec![vec![0.0_f64; n]; n_groups],
        counts: vec![0_u32; n],
    };
    let soup_grids = crate::explain::MergedGrids::from_model(model)?;
    for (bag, in_bag) in membership.iter().enumerate() {
        let oob_rows: Vec<u32> = (0..n as u32).filter(|&r| !in_bag[r as usize]).collect();
        if oob_rows.is_empty() {
            continue;
        }
        // One bag's banks live at a time (see `bag_raw_scores_for_rows`) — peak memory is
        // this bag's full bank plus the group views of it. The groups a caller passes are
        // disjoint table sets, so those views together hold at most one more copy of the
        // bank's tables: ~2x one bank, never n_bags of them.
        // A §G1 cell correction is shared whole across the bags here — see
        // `bag_member_model` for why that is right for the jury mean and wrong for a replicate.
        let full = bag_bank(model, serve, &w_measure, mass, &soup_grids, bag, true)?;
        // Group banks are `retain_tables` views of `full` with the intercept zeroed, so a
        // group's score is a pure table sum and prefix unions add exactly.
        let group_banks: Vec<TableBank> = groups
            .iter()
            .map(|g| {
                let mut b = retain_tables(&full, g);
                b.f0 = 0.0;
                b
            })
            .collect();
        let lead = usize::from(with_full);
        let mut banks: Vec<&TableBank> = Vec::with_capacity(n_groups + lead);
        if with_full {
            banks.push(&full);
        }
        banks.extend(group_banks.iter());
        let width = banks.len();
        let mut flat = vec![0.0_f64; oob_rows.len() * width];
        crate::scoring::score_banks_rows(
            &banks,
            &model.schema.cat_encoders,
            &serve.0,
            &oob_rows,
            &mut flat,
        )?;
        let f0 = full.f0;
        for (i, &r) in oob_rows.iter().enumerate() {
            let r = r as usize;
            let chunk = flat
                .get(i * width..(i + 1) * width)
                .ok_or_else(|| PbError::Internal {
                    what: "out-of-bag score chunk escaped the flat buffer".into(),
                })?;
            acc.counts[r] += 1;
            acc.f0_sum[r] += f0;
            if with_full {
                acc.full_sum[r] += chunk[0];
            }
            for (g, slot) in acc.group_sums.iter_mut().enumerate() {
                slot[r] += chunk[g + lead];
            }
        }
    }
    Ok(acc)
}

/// Whether [`bag_oob_group_raw_sums`] can run on `model`: it needs the outer-bag soup's
/// runtime partition AND membership (so a single fit or a deserialized model is out). A §G1
/// cell correction is fine here — the evidence path shares it across the bags (see
/// `bag_member_model`) — even though the replicate APIs refuse it.
///
/// Callers use this to DECIDE, rather than catching the error, so a genuine failure inside the
/// evidence pass — a mismatched design, an allocation failure — still surfaces as an error
/// instead of being read as "this model has no out-of-bag rows".
#[must_use]
pub fn bag_oob_evidence_available(model: &Model) -> bool {
    model.bag_spans.is_some() && model.bag_in_bag.is_some()
}

/// Per-row out-of-bag accumulations returned by [`bag_oob_group_raw_sums`]. Every vector is
/// indexed by FIT row; divide by `counts` (skipping zeros) to get means.
#[derive(Debug, Clone, PartialEq)]
pub struct BagOobGroupSums {
    /// Σ over the row's out-of-bag bags of the FULL bag bank raw score (intercept included).
    pub full_sum: Vec<f64>,
    /// Σ over the row's out-of-bag bags of the bag bank intercepts alone.
    pub f0_sum: Vec<f64>,
    /// `group_sums[g][r]` = Σ over the row's out-of-bag bags of group `g`'s tables (no
    /// intercept), in the order the groups were passed.
    pub group_sums: Vec<Vec<f64>>,
    /// How many bags left each row out of bag. `0` ⇒ the row carries no evidence.
    pub counts: Vec<u32>,
}

/// Re-solve the surviving tables' cell values to the deviance-optimum of the reduced
/// (pruned) structure, holding the keep-set fixed. This is the option-2 "rebalance on
/// means": a single Newton (IRLS) step of the §G1 ridge cell-refit fit to the *pruned*
/// model's working residual over the kept main+pair supports.
///
/// The re-solve exploits that `purify` is **linear** at fixed weights, so
/// `purify(trees + δ) = base + purify(δ)`: it purifies the tiny correction `δ` alone
/// ([`crate::explain::purify_correction`]) — re-purified, so the 1-/2-/3-way order separation
/// is preserved — and adds it to the already-built deploy bank, instead of paying a second
/// (expensive, 8-bag) `explain_bank`. The base bank's raw score is passed in (`base_raw`,
/// already computed for the deploy re-anchor), and the corrected raw is reconstructed as
/// `base_raw + score(purify(δ))`, so no full-bank score is repeated. A deterministic ~15%
/// held-out slice never enters the fit and gates the result: the rebalance is adopted only if
/// it lowers held-out deviance, so it can only help or be neutralised — never regress.
#[allow(clippy::too_many_arguments)]
fn rebalance_kept_cells(
    model: &Model,
    base: TableBank,
    base_raw: &[f64],
    keep: &[FeatureSet],
    serve: &ServeBinnedMatrix,
    y: &[f32],
    w: &[f32],
    offset: Option<&[f32]>,
    loss: &dyn Loss,
    w_measure: RefMeasure,
    do_reanchor: bool,
) -> Result<TableBank, PbError> {
    // Composing a post-prune re-solve on top of a fit-time cell correction is out of
    // scope; leave the pruned model as-is rather than silently drop that correction.
    if model.correction.is_some() {
        return Ok(base);
    }
    // P1 multi-channel (bug #5): `correction_scaffold` (reached via `fit_cell_correction`
    // below) treats each of its `supports` entries as a MODEL AXIS id, but this function
    // builds those supports from `keep`'s raw FEATURE ids (`FeatureSet::0`) a little further
    // down — correct only under the pre-P1 green-spine invariant (raw id == axis id always).
    // Once ANY raw feature in the model owns more than one axis, every raw feature POSITIONED
    // AFTER it has axis index > raw id, so passing its raw id as an axis id there silently
    // resolves the WRONG axis — the earlier multi-channel feature's own extra channel, not
    // the intended raw feature — which is wrong even when that wrong axis is single-channel,
    // and is the exact crash bug #5 reported when that wrong axis happens to BE a joint one.
    //
    // The check must cover the WHOLE model, not just `keep`: a multi-channel raw feature that
    // ISN'T itself kept can still shift a LATER kept feature's axis index and trigger the same
    // misresolution — e.g. raw 0 (single-axis, axis 0), raw 1 (multi-channel, axes 1-2), raw 2
    // (single-axis, axis 3), `keep = [{raw 0}, {raw 2}]`: raw 1 is never kept, but
    // `supports` still contains raw 2's id (`2`), which `correction_scaffold` would resolve as
    // axis 2 — raw 1's own second channel, not raw 2 at all. Skipping only when a kept support
    // itself named a multi-channel feature would miss this case entirely.
    //
    // So: skip rebalance for the ENTIRE model (never per-feature — a partial rebalance would
    // need the very fix this skip avoids) whenever the model has ANY multi-channel raw feature
    // anywhere, falling back to the already-proven-lossless unrebalanced deploy. A
    // single-channel model (`provenance.len() == n_raw_features`, unaffected either way) always
    // takes the `false` branch here and rebalances exactly as before this fix existed.
    if model.provenance.len() > crate::data::n_raw_features(&model.provenance) {
        return Ok(base);
    }
    let prof = std::env::var_os("TBOOST_PROFILE").is_some();
    let n = serve.0.n_rows as usize;

    // (1) Newton working residual z = -g/h, IRLS weight h, at the pruned model's raw
    //     prediction (reused base_raw — no score). A deterministic ~15% slice is held out.
    let t = std::time::Instant::now();
    let raw_f32: Vec<f32> = (0..n)
        .map(|r| {
            let o = offset.map_or(0.0_f32, |off| off.get(r).copied().unwrap_or(0.0));
            base_raw[r] as f32 + o
        })
        .collect();
    let mut gh = crate::loss::GradHess {
        g: vec![0.0_f32; n],
        h: vec![0.0_f32; n],
    };
    loss.grad_hess(y, &raw_f32, w, &mut gh)?;
    let is_holdout = |r: usize| ((r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) < 38;
    let mut residual = vec![0.0_f64; n];
    let mut weight = vec![0.0_f64; n];
    for r in 0..n {
        if !is_holdout(r) {
            let h = f64::from(gh.h[r]).max(1e-12);
            residual[r] = -f64::from(gh.g[r]) / h;
            weight[r] = h;
        }
    }

    // (2) The correctable survivors: kept main + pair supports (order ≤ 2). The §G1
    //     cell-basis correction spans mains + pairs; kept triples ride along unchanged.
    let supports: Vec<Vec<u32>> = keep
        .iter()
        .filter(|u| u.order() <= 2)
        .map(|u| u.0.iter().map(|f| f.0).collect())
        .collect();
    if supports.is_empty() {
        return Ok(base);
    }
    if prof {
        eprintln!(
            "[rebalance]  (1) residual grad_hess {:.2}s (base_raw reused, {n} rows)",
            t.elapsed().as_secs_f64()
        );
    }

    // (3) Fit the ridge cell correction to the residual (held-out rows carry weight 0).
    let t = std::time::Instant::now();
    let spec = crate::cell_refit::CellRefitSpec::default();
    let corr = crate::cell_refit::fit_cell_correction(
        model,
        &serve.0.data,
        &residual,
        &weight,
        &supports,
        &spec,
    )?;
    if prof {
        eprintln!(
            "[rebalance]  (3) fit_cell_correction (ridge) {:.2}s ({} supports)",
            t.elapsed().as_secs_f64(),
            supports.len()
        );
    }

    // (4) Direct re-solve: purify δ ALONE (linear ⇒ no 8-bag re-explain), score the tiny δ
    //     bank, and reconstruct the corrected raw as base_raw + score(purify(δ)).
    let t = std::time::Instant::now();
    let mass = measure_mass(&w_measure, w, offset);
    let delta_bank =
        crate::explain::purify_correction(model, serve, w_measure, mass.as_deref(), &corr)?;
    let mut delta_raw = vec![0.0_f64; n];
    crate::scoring::score_bank_binned(
        &delta_bank,
        &model.schema.cat_encoders,
        &serve.0,
        &mut delta_raw,
    )?;
    let mut corr_raw: Vec<f64> = (0..n).map(|r| base_raw[r] + delta_raw[r]).collect();
    let shift = if do_reanchor {
        match loss.link() {
            Link::Log => reanchor_shift(&corr_raw, y, w, offset, loss)?,
            Link::Logit => {
                let raw32: Vec<f32> = corr_raw.iter().map(|&v| v as f32).collect();
                crate::engine::boost::reanchor_delta(Link::Logit, y, w, &raw32, offset)?
            }
            Link::Identity => 0.0,
        }
    } else {
        0.0
    };
    if shift != 0.0 {
        for v in corr_raw.iter_mut() {
            *v += shift;
        }
    }
    if prof {
        eprintln!(
            "[rebalance]  (4) purify(δ)+score+reanchor {:.2}s ({} δ-tables)  <-- no re-explain",
            t.elapsed().as_secs_f64(),
            delta_bank.tables.len()
        );
    }

    // (5) No-harm guard: adopt the rebalance only if it lowers deviance on the held-out slice
    //     it never saw. `corr_raw` == score(base ⊕ δ) exactly (scoring is linear), so no
    //     second full-bank score is needed to evaluate it.
    let t = std::time::Instant::now();
    let (mut hy, mut hb, mut hc, mut hw) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for r in 0..n {
        if is_holdout(r) {
            let o = offset.map_or(0.0_f32, |off| off.get(r).copied().unwrap_or(0.0));
            hy.push(y[r]);
            hb.push(base_raw[r] as f32 + o);
            hc.push(corr_raw[r] as f32 + o);
            hw.push(w[r]);
        }
    }
    let adopt = if hy.is_empty() {
        true // no held-out coverage → keep the rebalance
    } else {
        let base_dev = loss.deviance(&hy, &hb, &hw)?;
        let corr_dev = loss.deviance(&hy, &hc, &hw)?;
        if prof {
            eprintln!(
                "[rebalance]  (5) guard {:.2}s (base_dev={base_dev:.5} corr_dev={corr_dev:.5} -> {})",
                t.elapsed().as_secs_f64(),
                if corr_dev <= base_dev { "ADOPT" } else { "fall back" }
            );
        }
        corr_dev <= base_dev
    };
    if !adopt {
        return Ok(base);
    }
    // Materialise base ⊕ purify(δ) in place + apply the re-anchor shift.
    let mut corrected = base;
    crate::explain::add_bank_assign(&mut corrected, &delta_bank)?;
    corrected.f0 += shift;
    Ok(corrected)
}

/// Intercept-only multinomial re-anchor: the per-class shifts `delta` that make the softmax's
/// weighted predicted class mass reproduce the observed one, `Σ_i w_i·softmax(raw_i + delta)_k
/// = Σ_i w_i·1[y_i = k]`.
///
/// This is the K-class analogue of the single-output log-link `f0` re-anchor, solved by damped
/// Newton updates until every class's mass residual meets the convergence tolerance.
/// Degenerate inputs (zero total mass, or a class with no observed mass) get
/// an all-zero shift rather than a `ln(0)`.
///
/// `raw` is `[class][row]` on the link scale; `labels`/`w` are per row. `what` names the caller
/// in error text.
///
/// # Errors
/// [`PbError::InvalidInput`] if a label escapes `0..raw.len()`.
fn multiclass_intercept_shifts(
    raw: &[Vec<f64>],
    labels: &[u32],
    w: &[f32],
    what: &str,
) -> Result<Vec<f64>, PbError> {
    let n_classes = raw.len();
    let n = labels.len();
    // Weighted class masses Σw·1[y=k].
    let mut class_mass = vec![0.0_f64; n_classes];
    for (i, &l) in labels.iter().enumerate() {
        let li = l as usize;
        if li >= n_classes {
            return Err(PbError::InvalidInput {
                what: format!("{what}: label {l} >= n_classes {n_classes}"),
            });
        }
        class_mass[li] += f64::from(w[i]);
    }
    let total_mass: f64 = class_mass.iter().sum();
    let mut delta = vec![0.0_f64; n_classes];
    if total_mass > 0.0 && class_mass.iter().all(|&m| m > 0.0) {
        // Fix the last class's shift at zero to remove softmax's common-shift gauge.
        // A damped Newton solve avoids IPF's arbitrarily slow convergence near separation.
        let dimension = n_classes.saturating_sub(1);
        let moments = |shift: &[f64]| {
            let mut predicted = vec![0.0; n_classes];
            let mut hessian = vec![vec![0.0; dimension]; dimension];
            for i in 0..n {
                let mx = (0..n_classes)
                    .map(|k| raw[k][i] + shift[k])
                    .fold(f64::NEG_INFINITY, f64::max);
                let mut probabilities: Vec<f64> = (0..n_classes)
                    .map(|k| (raw[k][i] + shift[k] - mx).exp())
                    .collect();
                let z: f64 = probabilities.iter().sum();
                for probability in &mut probabilities {
                    *probability /= z;
                }
                let wi = f64::from(w[i]);
                for k in 0..n_classes {
                    predicted[k] += wi * probabilities[k];
                }
                for k in 0..dimension {
                    for j in 0..dimension {
                        hessian[k][j] +=
                            wi * probabilities[k] * (f64::from(k == j) - probabilities[j]);
                    }
                }
            }
            (predicted, hessian)
        };
        let error = |predicted: &[f64]| {
            predicted
                .iter()
                .zip(&class_mass)
                .map(|(p, y)| ((p - y) / total_mass).powi(2))
                .sum::<f64>()
        };
        for _ in 0..4096 {
            let (pred_mass, mut hessian) = moments(&delta);
            if pred_mass
                .iter()
                .zip(&class_mass)
                .all(|(p, y)| (p - y).abs() <= 1e-10 * total_mass)
            {
                return Ok(delta);
            }
            // Ridge stabilizes nearly singular Hessians, without changing the optimum.
            for k in 0..dimension {
                hessian[k][k] += 1e-12 * total_mass;
            }
            let mut lower = vec![vec![0.0; dimension]; dimension];
            for k in 0..dimension {
                for j in 0..=k {
                    let cross: f64 = (0..j).map(|r| lower[k][r] * lower[j][r]).sum();
                    lower[k][j] = if k == j {
                        (hessian[k][j] - cross).max(1e-15 * total_mass).sqrt()
                    } else {
                        (hessian[k][j] - cross) / lower[j][j]
                    };
                }
            }
            let mut step = vec![0.0; dimension];
            for k in 0..dimension {
                let cross: f64 = (0..k).map(|r| lower[k][r] * step[r]).sum();
                step[k] = (class_mass[k] - pred_mass[k] - cross) / lower[k][k];
            }
            for k in (0..dimension).rev() {
                let cross: f64 = ((k + 1)..dimension).map(|r| lower[r][k] * step[r]).sum();
                step[k] = (step[k] - cross) / lower[k][k];
            }
            let largest = step.iter().map(|v| v.abs()).fold(0.0, f64::max);
            let mut scale = if largest > 20.0 { 20.0 / largest } else { 1.0 };
            let current_error = error(&pred_mass);
            let mut accepted = false;
            for _ in 0..32 {
                let mut proposal = delta.clone();
                for k in 0..dimension {
                    proposal[k] += scale * step[k];
                }
                if error(&moments(&proposal).0) < current_error {
                    delta = proposal;
                    accepted = true;
                    break;
                }
                scale *= 0.5;
            }
            if !accepted {
                // IPF is also a descent method and supplies a conservative fallback.
                for k in 0..n_classes {
                    if pred_mass[k] > 0.0 {
                        delta[k] += (class_mass[k] / pred_mass[k]).ln();
                    }
                }
                let gauge = delta.last().copied().unwrap_or(0.0);
                for value in &mut delta {
                    *value -= gauge;
                }
            }
        }
        return Err(PbError::InvalidInput {
            what: format!("{what}: multiclass intercept reanchor did not converge"),
        });
    }
    Ok(delta)
}

/// Prune a native-softmax [`MultiClassModel`] into a tables-only [`MultiClassTableModel`] by dropping
/// a **shared** downward-closed keep-set from the `K` per-class banks, selected on held-out softmax
/// cross-entropy deviance. The class logits couple through softmax, so per-class independent selection
/// would be statistically wrong — the keep-set is chosen jointly over the union of feature-sets and
/// applied to every class. (The prune DOES re-anchor the per-class intercepts afterwards — see
/// the Re-anchor paragraph below. An earlier revision did not, and this line said so; it is kept
/// corrected rather than deleted because the stale claim outlived the code and was still being
/// read as "K>=3 has no re-anchor".)
///
/// `labels` are class indices (`0..K`) per serve row; `sel_rows` are held-out rows the model did not
/// train on.
///
/// Retain an already-selected keep-set on a (possibly bagged) multiclass model's per-class
/// banks — the multiclass analog of [`prune_model_to_keepset`]'s retain step — then re-anchor
/// the per-class intercepts. Used by the honest-selection + full-data-deploy flow: the keep-set
/// is chosen on a complement fit's held-out slice ([`prune_multiclass_to_tables`]), then applied
/// here to a fit that saw ALL rows. Retain-only for tables: a kept id this model never grew
/// simply contributes nothing.
///
/// Re-anchor: dropping tables shifts each class's logit mass, so the softmax class balance
/// drifts (the deployed universe — especially a bagged soup's — is wider than the selection
/// fit's, making the drift material at larger K). The multinomial analog of the binary
/// log-link `f0` re-anchor is an intercept-only IPF: `f0_k += ln(Σw·1[y=k] / Σw·p̂_k)`,
/// iterated until weighted class masses agree to a deterministic relative tolerance.
/// The updates are proportional fitting; nonconvergence returns a typed error.
///
/// # Errors
/// Propagates bank construction/scoring/validation failures. [`PbError::ShapeMismatch`] if
/// `labels`/`w` disagree with the serve row count.
pub fn prune_multiclass_to_keepset(
    mc: &MultiClassModel,
    serve: &ServeBinnedMatrix,
    labels: &[u32],
    w: &[f32],
    w_measure: RefMeasure,
    keep: &[FeatureSet],
) -> Result<MultiClassTableModel, PbError> {
    let (out, _, _) = prune_multiclass_to_keepset_budgeted(
        mc,
        serve,
        labels,
        w,
        w_measure,
        keep,
        0,
        0,
        DEFAULT_TABLE_MIN_ARITY,
        &BTreeMap::new(),
    )?;
    Ok(out)
}

/// The lossless tables-only form of a multiclass model: every class's full purified bank under
/// `w_measure`, with no table dropped and NO intercept re-anchor, so it predicts as the
/// ensemble does (within the §08 reconstruction tolerance). The multiclass twin of
/// [`TableModel::from_model`]; [`prune_multiclass_to_keepset`] is not, because its re-anchor
/// shifts the logits even when every table is kept.
///
/// # Errors
/// Propagates per-class bank construction and container validation failures.
pub fn multiclass_full_tables(
    mc: &MultiClassModel,
    serve: &ServeBinnedMatrix,
    w: &[f32],
    w_measure: RefMeasure,
) -> Result<MultiClassTableModel, PbError> {
    let n = serve.0.n_rows as usize;
    if w.len() != n {
        return Err(PbError::ShapeMismatch {
            what: format!("full tables: w {} != serve rows {n}", w.len()),
        });
    }
    let mass = measure_mass(&w_measure, w, None);
    let classes: Vec<TableModel> = mc
        .classes
        .par_iter()
        .map(|m| -> Result<TableModel, PbError> {
            let bank = m.explain_bank_with_mass(
                serve,
                w_measure.clone(),
                crate::explain::TableBudget::default(),
                false,
                mass.as_deref(),
            )?;
            Ok(TableModel::from_model_and_bank(m, bank))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let class_version = classes
        .iter()
        .map(|c| c.schema_version)
        .max()
        .unwrap_or(mc.schema_version);
    let out = MultiClassTableModel {
        classes,
        class_labels: mc.class_labels.clone(),
        schema_version: class_version,
    };
    out.validate()?;
    Ok(out)
}

/// [`prune_multiclass_to_keepset`] with a DEPLOYED-BOX BUDGET (see [`apply_box_budget`]).
///
/// The keep-set is shared across classes, so the budget is too: a support's cost is the sum of
/// its box counts over the K per-class banks, because a filing reader faces every class's copy
/// of it. `max_boxes == 0` disables the budget and reproduces
/// [`prune_multiclass_to_keepset`] bit-for-bit.
///
/// # Errors
/// As [`prune_multiclass_to_keepset`].
pub fn prune_multiclass_to_keepset_budgeted(
    mc: &MultiClassModel,
    serve: &ServeBinnedMatrix,
    labels: &[u32],
    w: &[f32],
    w_measure: RefMeasure,
    keep: &[FeatureSet],
    max_boxes: usize,
    max_tables: usize,
    table_min_arity: u8,
    evidence: &BTreeMap<FeatureSet, f64>,
) -> Result<(MultiClassTableModel, BoxBudgetReport, TableBudgetReport), PbError> {
    let n = serve.0.n_rows as usize;

    if labels.len() != n || w.len() != n {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "keepset reanchor: labels {} / w {} != serve rows {n}",
                labels.len(),
                w.len()
            ),
        });
    }
    let mass = measure_mass(&w_measure, w, None);
    let banks: Vec<TableBank> = mc
        .classes
        .par_iter()
        .map(|m| {
            m.explain_bank_with_mass(
                serve,
                w_measure.clone(),
                crate::explain::TableBudget::default(),
                false,
                mass.as_deref(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let bank_refs: Vec<&TableBank> = banks.iter().collect();
    let box_cost_map = box_costs(&bank_refs);
    let (budgeted, mut box_report) = apply_box_budget(
        keep,
        &box_cost_map,
        &box_variances(&bank_refs),
        max_boxes,
        evidence,
    );
    // Composed, box budget first — see the single-class twin for why that order.
    let (budgeted, table_report) = apply_table_budget(
        &budgeted,
        &table_variances(&bank_refs),
        max_tables,
        table_min_arity,
        evidence,
    );
    refresh_box_report_after(&mut box_report, &budgeted, &box_cost_map);
    let keep: &[FeatureSet] = &budgeted;
    let mut pruned_banks: Vec<TableBank> =
        banks.iter().map(|bank| retain_tables(bank, keep)).collect();
    // Per-class retained logits over the training rows (fold f0 in afterwards so the IPF can
    // adjust it without re-scoring).
    let raw: Vec<Vec<f64>> = pruned_banks
        .par_iter()
        .zip(mc.classes.par_iter())
        .map(|(bank, model)| -> Result<Vec<f64>, PbError> {
            let mut out = vec![0.0_f64; n];
            crate::scoring::score_bank_binned(
                bank,
                &model.schema.cat_encoders,
                &serve.0,
                &mut out,
            )?;
            Ok(out)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let delta = multiclass_intercept_shifts(&raw, labels, w, "keepset reanchor")?;
    for (bank, d) in pruned_banks.iter_mut().zip(&delta) {
        bank.f0 += d;
    }
    let classes: Vec<TableModel> = pruned_banks
        .into_iter()
        .zip(mc.classes.iter())
        .map(|(pruned, model)| TableModel::from_model_and_bank(model, pruned))
        .collect();
    // A pruned bank can only LOSE factored effects, never gain them, so this is at most the
    // source container's stamp — but take the max over the classes' own requirements rather
    // than trusting the source, mirroring `TableModel::from_model_and_bank`.
    let class_version = classes
        .iter()
        .map(|c| c.schema_version)
        .max()
        .unwrap_or(mc.schema_version);
    let out = MultiClassTableModel {
        classes,
        class_labels: mc.class_labels.clone(),
        schema_version: class_version,
    };
    out.validate()?;
    Ok((out, box_report, table_report))
}

/// # Errors
/// [`PbError`] from per-class [`Model::explain_bank`], the fold construction, the deviance, or
/// container validation. [`PbError::InvalidInput`] if `sel_rows` is empty.
pub fn prune_multiclass_to_tables(
    mc: &MultiClassModel,
    serve: &ServeBinnedMatrix,
    labels: &[u32],
    w: &[f32],
    w_measure: RefMeasure,
    sel_rows: &[usize],
    n_folds: usize,
    cfg: &PruneConfig,
) -> Result<(MultiClassTableModel, PruneReport), PbError> {
    if sel_rows.is_empty() {
        return Err(PbError::InvalidInput {
            what: "prune_multiclass_to_tables requires a non-empty held-out selection set".into(),
        });
    }
    let n_classes = mc.classes.len();
    let prof_on = std::env::var_os("TBOOST_PROFILE").is_some();
    let lap = |name: &str, t: &std::time::Instant| {
        if prof_on {
            eprintln!("[mc-walk] {name} {:.2}s", t.elapsed().as_secs_f64());
        }
    };
    let t_phase = std::time::Instant::now();

    // Per-class purified banks (ungated — same rationale as the scalar path).
    let mass = measure_mass(&w_measure, w, None);
    let banks: Vec<TableBank> = mc
        .classes
        .par_iter()
        .map(|m| {
            m.explain_bank_with_mass(
                serve,
                w_measure.clone(),
                crate::explain::TableBudget::default(),
                false,
                mass.as_deref(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    lap("banks", &t_phase);
    let t_phase = std::time::Instant::now();
    // Union of feature-sets across all classes, with a combined (summed) variance for the pre-order.
    let mut var_by_id: BTreeMap<FeatureSet, f64> = BTreeMap::new();
    for bank in &banks {
        for t in &bank.tables {
            *var_by_id.entry(t.u.clone()).or_insert(0.0) += t.variance;
        }
        for ft in &bank.factored {
            *var_by_id.entry(ft.u.clone()).or_insert(0.0) += ft.variance;
        }
    }
    let ids: Vec<FeatureSet> = var_by_id.keys().cloned().collect();
    let variance: Vec<f64> = ids.iter().map(|u| var_by_id[u]).collect();
    let n_ids = ids.len();
    let id_index: BTreeMap<FeatureSet, usize> = ids
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, u)| (u, i))
        .collect();

    // Fold partition (round-robin) of the held-out rows.
    let n_folds = n_folds.clamp(1, sel_rows.len());
    let mut fold_rows: Vec<Vec<usize>> = vec![Vec::new(); n_folds];
    for (i, &r) in sel_rows.iter().enumerate() {
        fold_rows[i % n_folds].push(r);
    }

    // Per fold: labels, weights, running per-class logits, and per-(id, class) row contributions.
    let mut fold_labels: Vec<Vec<u32>> = Vec::with_capacity(n_folds);
    let mut fold_ws: Vec<Vec<f32>> = Vec::with_capacity(n_folds);
    let mut f_logit: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_folds); // [fold][class][row]
    let mut contrib: Vec<Vec<Vec<Option<Vec<f64>>>>> = Vec::with_capacity(n_folds); // [fold][id][class]
    for rows in &fold_rows {
        let nrow = rows.len();
        let mut fl = Vec::with_capacity(nrow);
        let mut fw = Vec::with_capacity(nrow);
        for &r in rows {
            fl.push(*labels.get(r).ok_or_else(|| PbError::InvalidInput {
                what: format!("selection row {r} out of labels range"),
            })?);
            fw.push(*w.get(r).ok_or_else(|| PbError::InvalidInput {
                what: format!("selection row {r} out of w range"),
            })?);
        }
        let fold_x = subset_binned(&serve.0, rows)?;
        let mut flog: Vec<Vec<f64>> = Vec::with_capacity(n_classes);
        let mut fcontrib: Vec<Vec<Option<Vec<f64>>>> = (0..n_ids)
            .map(|_| (0..n_classes).map(|_| None).collect())
            .collect();
        for (ci, bank) in banks.iter().enumerate() {
            let class_model = mc.classes.get(ci).ok_or_else(|| PbError::Internal {
                what: format!("prune_multiclass_to_tables: class {ci} escaped mc.classes"),
            })?;
            let maps = crate::scoring::build_cell_maps(
                &bank.merged_grids,
                &class_model.schema.cat_encoders,
                &fold_x,
            )?;
            let mut cells = vec![0u32; bank.merged_grids.len()];
            let class_ids: Vec<FeatureSet> = bank
                .tables
                .iter()
                .map(|t| t.u.clone())
                .chain(bank.factored.iter().map(|ft| ft.u.clone()))
                .collect();
            let mut class_contrib: Vec<Vec<f64>> = vec![vec![0.0; nrow]; class_ids.len()];
            let mut logit = vec![bank.f0; nrow];
            for ri in 0..nrow {
                crate::scoring::fill_row_cells(&fold_x, &maps, ri, &mut cells)?;
                let mut ti = 0;
                for t in &bank.tables {
                    let c = t.eval(&cells)?;
                    class_contrib[ti][ri] = c;
                    logit[ri] += c;
                    ti += 1;
                }
                for ft in &bank.factored {
                    let c = ft.eval(&cells)?;
                    class_contrib[ti][ri] = c;
                    logit[ri] += c;
                    ti += 1;
                }
            }
            for (ti, u) in class_ids.iter().enumerate() {
                fcontrib[id_index[u]][ci] = Some(std::mem::take(&mut class_contrib[ti]));
            }
            flog.push(logit);
        }
        fold_labels.push(fl);
        fold_ws.push(fw);
        f_logit.push(flog);
        contrib.push(fcontrib);
    }

    lap("fold_contribs", &t_phase);
    let t_phase = std::time::Instant::now();
    // Both closures also return the per-fold vector (not just its mean/SE) so the table-score
    // loop below can pair drop-fold against full-fold deviances fold-for-fold instead of
    // treating the two marginal SEs as independent (see se_gain below).
    let eval = |f_logit: &[Vec<Vec<f64>>]| -> Result<(f64, f64, Vec<f64>), PbError> {
        // Folds are independent and collect back in fold order, preserving mean/SE arithmetic.
        let per_fold: Vec<f64> = (0..n_folds)
            .into_par_iter()
            .map(|fi| -> Result<f64, PbError> {
                let nrow = fold_ws[fi].len();
                let raw_cols: Vec<Vec<f32>> = (0..n_classes)
                    .map(|k| f_logit[fi][k].iter().map(|&v| v as f32).collect())
                    .collect();
                let rowset: Vec<usize> = (0..nrow).collect();
                let dev = crate::engine::boost::multiclass_deviance_for_rows(
                    &raw_cols,
                    &fold_labels[fi],
                    &fold_ws[fi],
                    &rowset,
                )?;
                // The native multiclass deviance is already a weighted mean.
                // Each fold contributes one mean to the equal-fold SE estimate.
                Ok(dev)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (mean, se) = mean_and_se(&per_fold);
        Ok((mean, se, per_fold))
    };
    let eval_without =
        |f_logit: &[Vec<Vec<f64>>], id_idx: usize| -> Result<(f64, f64, Vec<f64>), PbError> {
            let mut candidate = f_logit.to_vec();
            for fi in 0..n_folds {
                for ci in 0..n_classes {
                    if let Some(c) = &contrib[fi][id_idx][ci] {
                        for r in 0..candidate[fi][ci].len() {
                            candidate[fi][ci][r] -= c[r];
                        }
                    }
                }
            }
            eval(&candidate)
        };

    // Per-support deployed BOX cost, SUMMED ACROSS CLASSES — a filing reader faces every class's
    // copy of a support, so they add. Same convention as `box_costs`, which is what the
    // deploy-time budget spends; dense tables cost zero.
    let box_cost: Vec<u32> = {
        let mut acc = vec![0_u32; n_ids];
        for bank in &banks {
            for ft in &bank.factored {
                if let Some(&i) = id_index.get(&ft.u) {
                    acc[i] = acc[i].saturating_add(u32::try_from(ft.n_boxes()).unwrap_or(u32::MAX));
                }
            }
        }
        acc
    };
    let mut boxes_kept: u32 = box_cost.iter().copied().fold(0_u32, u32::saturating_add);
    // The keep-set is SHARED across classes, so a support is ONE entry here however many class
    // banks hold a copy of it. That is deliberately not the box convention above: boxes add
    // across classes because a filing prints every class's numbers, but the STRUCTURE a reader
    // has to understand — "these three features interact" — is stated once.
    let mut arity_kept: ArityCounts = arity_counts_of(ids.iter());

    let mut path: Vec<PrunePoint> = Vec::with_capacity(n_ids + 1);
    let (m0, s0, full_per_fold) = eval(&f_logit)?;
    path.push(PrunePoint {
        n_tables: n_ids as u32,
        n_boxes: boxes_kept,
        n_tables_by_arity: arity_kept,
        dropped: None,
        mean_deviance: m0,
        se: s0,
    });
    // Exact drop deviances ride along to seed the lazy walk's cache (mirrors the single-class
    // path — same value, not re-derived from mean_gain, so no fp drift).
    let scored: Vec<(PruneTableScore, f64)> = (0..n_ids)
        .into_par_iter()
        .map(|i| -> Result<(PruneTableScore, f64), PbError> {
            let (drop_m, _drop_s, drop_per_fold) = eval_without(&f_logit, i)?;
            // se_gain is the SE of the PAIRED per-fold difference — see the single-class path's
            // identical comment above.
            let paired: Vec<f64> = drop_per_fold
                .iter()
                .zip(&full_per_fold)
                .map(|(&d, &f)| d - f)
                .collect();
            let (_, se_gain) = mean_and_se(&paired);
            Ok((
                PruneTableScore {
                    u: ids[i].clone(),
                    order: ids[i].order() as u8,
                    mean_gain: drop_m - m0,
                    se_gain,
                    variance: variance[i],
                    sticky: STICKY_MAIN_EFFECTS && ids[i].order() == 1,
                    selected: true,
                },
                drop_m,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    lap("initial_scores", &t_phase);
    let t_phase = std::time::Instant::now();
    let mut cached_m: Vec<f64> = scored.iter().map(|(_, m)| *m).collect();
    let mut table_scores: Vec<PruneTableScore> = scored.into_iter().map(|(ts, _)| ts).collect();

    let mut kept = vec![true; n_ids];
    let mut drops: Vec<FeatureSet> = Vec::new();
    // Maximal ⇔ zero kept proper supersets — incrementally maintained (see `superset_frontier`);
    // identical candidate sets to the former per-iteration rescan.
    let (mut kept_superset_count, proper_subsets_of) = superset_frontier(&ids);
    // Same lazy (CELF-style) walk gate as the single-class path: exhaustive below the table
    // threshold (bit-identical to the historical walk), lazy above it — each multiclass
    // candidate eval is a full [fold][class][row] deviance pass, K× the binary cost, so wide
    // multiclass banks pay the most (K=8 prudential prune measured 1200s vs 66s unpruned).
    let lazy_walk = lazy_walk_engaged(n_ids);
    let arg_min = |a: (usize, f64, f64), b: (usize, f64, f64)| -> (usize, f64, f64) {
        match a.1.total_cmp(&b.1) {
            std::cmp::Ordering::Less => a,
            std::cmp::Ordering::Greater => b,
            std::cmp::Ordering::Equal => {
                if ids[a.0] < ids[b.0] {
                    a
                } else {
                    b
                }
            }
        }
    };
    loop {
        let candidates: Vec<usize> = (0..n_ids)
            .filter(|&i| {
                kept[i]
                    && !(STICKY_MAIN_EFFECTS && ids[i].order() == 1)
                    && kept_superset_count[i] == 0
            })
            .collect();
        let best: Option<(usize, f64, f64)> = if lazy_walk {
            let mut order = candidates.clone();
            order.sort_by(|&a, &b| {
                cached_m[a].total_cmp(&cached_m[b]).then_with(|| {
                    if ids[a] < ids[b] {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Greater
                    }
                })
            });
            let mut best: Option<(usize, f64, f64)> = None;
            let mut done = 0usize;
            while done < order.len() {
                let batch = &order[done..(done + 16).min(order.len())];
                let evals: Vec<(usize, f64, f64)> = batch
                    .par_iter()
                    .map(|&i| -> Result<(usize, f64, f64), PbError> {
                        let (drop_m, drop_s, _) = eval_without(&f_logit, i)?;
                        Ok((i, drop_m, drop_s))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                for e in evals {
                    cached_m[e.0] = e.1;
                    best = Some(match best {
                        None => e,
                        Some(b) => arg_min(e, b),
                    });
                }
                done += batch.len();
                if let Some((_, bm, _)) = best {
                    if done >= order.len() || bm < cached_m[order[done]] {
                        break;
                    }
                }
            }
            best
        } else {
            candidates
                .par_iter()
                .map(|&i| -> Result<(usize, f64, f64), PbError> {
                    let (drop_m, drop_s, _) = eval_without(&f_logit, i)?;
                    Ok((i, drop_m, drop_s))
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .min_by(|a, b| {
                    a.1.total_cmp(&b.1).then_with(|| {
                        if ids[a.0] < ids[b.0] {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        }
                    })
                })
        };
        let Some((drop_idx, m, s)) = best else { break };
        kept[drop_idx] = false;
        for &i in &proper_subsets_of[drop_idx] {
            kept_superset_count[i as usize] -= 1; // exact by construction: one per kept superset
        }
        f_logit
            .par_iter_mut()
            .zip(contrib.par_iter())
            .for_each(|(fold_logits, fold_contrib)| {
                fold_logits
                    .par_iter_mut()
                    .zip(fold_contrib[drop_idx].par_iter())
                    .for_each(|(class_logits, class_contrib)| {
                        if let Some(c) = class_contrib {
                            class_logits
                                .par_iter_mut()
                                .zip(c.par_iter())
                                .for_each(|(logit, contribution)| *logit -= *contribution);
                        }
                    });
            });
        drops.push(ids[drop_idx].clone());
        boxes_kept = boxes_kept.saturating_sub(box_cost[drop_idx]);
        decrement_arity(&mut arity_kept, &ids[drop_idx]);
        path.push(PrunePoint {
            n_tables: kept.iter().filter(|&&k| k).count() as u32,
            n_boxes: boxes_kept,
            n_tables_by_arity: arity_kept,
            dropped: Some(ids[drop_idx].clone()),
            mean_deviance: m,
            se: s,
        });
    }

    // Penalized selection — identical in form and in bytes-when-inert to the scalar path above.
    lap("walk", &t_phase);
    let t_phase = std::time::Instant::now();
    let (min_i, min_obj) = path
        .iter()
        .enumerate()
        .map(|(i, p)| (i, cfg.objective(p)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .ok_or_else(|| PbError::Internal {
            what: "multiclass prune path unexpectedly empty".into(),
        })?;
    let threshold = min_obj + cfg.se_rule * path[min_i].se;
    let sel_i = path
        .iter()
        .enumerate()
        .filter(|(_, p)| cfg.objective(p) <= threshold)
        .min_by_key(|(_, p)| p.n_tables)
        .map(|(i, _)| i)
        .unwrap_or(0);

    let n_drop = n_ids - path[sel_i].n_tables as usize;
    let dropped: Vec<FeatureSet> = drops[..n_drop].to_vec();
    let kept_ids: Vec<FeatureSet> = ids
        .iter()
        .filter(|u| !dropped.iter().any(|d| d == *u))
        .cloned()
        .collect();
    let effective_order = kept_ids.iter().map(|u| u.order() as u8).max().unwrap_or(0);
    for ts in &mut table_scores {
        ts.selected = kept_ids.iter().any(|u| u == &ts.u);
    }

    let classes: Vec<TableModel> = banks
        .par_iter()
        .zip(mc.classes.par_iter())
        .map(|(bank, model)| {
            let pruned = retain_tables(bank, &kept_ids);
            TableModel::from_model_and_bank(model, pruned)
        })
        .collect();
    // A pruned bank can only LOSE factored effects, never gain them, so this is at most the
    // source container's stamp — but take the max over the classes' own requirements rather
    // than trusting the source, mirroring `TableModel::from_model_and_bank`.
    let class_version = classes
        .iter()
        .map(|c| c.schema_version)
        .max()
        .unwrap_or(mc.schema_version);
    let mc_tables = MultiClassTableModel {
        classes,
        class_labels: mc.class_labels.clone(),
        schema_version: class_version,
    };
    mc_tables.validate()?;

    lap("retain", &t_phase);
    let report = PruneReport {
        kept: kept_ids,
        dropped,
        effective_order,
        delta_vs_full: path[0].mean_deviance - path[sel_i].mean_deviance,
        path,
        table_scores,
    };
    Ok((mc_tables, report))
}

/// Fold `δ = ln(Σwy / Σwμ̂)` into the pruned bank's `f0` (raw space), so the exposure-weighted mean
/// prediction matches the data mean again after pruning shifts the aggregate. `f0`-only ⇒ exact.
/// The f0 re-anchor shift `ln(Σwy / Σw·μ(raw))` for a set of already-computed raw predictions
/// — the aggregate-balance correction, split out so a caller that already holds `raw` (the
/// rebalance path) re-anchors without a second `score_bank_binned` pass.
fn reanchor_shift(
    raw: &[f64],
    y: &[f32],
    w: &[f32],
    offset: Option<&[f32]>,
    loss: &dyn Loss,
) -> Result<f64, PbError> {
    let (mut sum_wy, mut sum_wmu) = (0.0_f64, 0.0_f64);
    for r in 0..raw.len() {
        let wi = f64::from(*w.get(r).ok_or_else(|| PbError::InvalidInput {
            what: "reanchor w out of range".into(),
        })?);
        let yi = f64::from(*y.get(r).ok_or_else(|| PbError::InvalidInput {
            what: "reanchor y out of range".into(),
        })?);
        let o = offset.map_or(0.0_f32, |off| off.get(r).copied().unwrap_or(0.0));
        let mu = f64::from(loss.pred_from_raw(raw[r] as f32 + o));
        sum_wy += wi * yi;
        sum_wmu += wi * mu;
    }
    Ok(if sum_wmu > 0.0 && sum_wy > 0.0 {
        (sum_wy / sum_wmu).ln()
    } else {
        0.0
    })
}

fn reanchor_log_link(
    bank: &mut TableBank,
    cat_encoders: &crate::cat::CatEncoderStore,
    serve: &ServeBinnedMatrix,
    y: &[f32],
    w: &[f32],
    offset: Option<&[f32]>,
    loss: &dyn Loss,
) -> Result<(), PbError> {
    let n = serve.0.n_rows as usize;
    let mut raw = vec![0.0_f64; n];
    crate::scoring::score_bank_binned(bank, cat_encoders, &serve.0, &mut raw)?;
    bank.f0 += reanchor_shift(&raw, y, w, offset, loss)?;
    Ok(())
}

/// Logit-link analog of [`reanchor_log_link`]: `reanchor_shift`'s closed form `ln(Σwy/Σwμ̂)` only
/// holds for the log link (the exponential's multiplicative structure lets `δ` factor out of the
/// aggregate-balance equation) — a logit-link bank has no such closed form, so this instead goes
/// through the shared bisection solve in [`crate::engine::boost::reanchor_delta`], the same helper
/// the main fit loop's post-fit reanchor uses. `raw` (from [`crate::scoring::score_bank_binned`])
/// does not include `offset`, matching `reanchor_delta`'s contract.
fn reanchor_logit_link(
    bank: &mut TableBank,
    cat_encoders: &crate::cat::CatEncoderStore,
    serve: &ServeBinnedMatrix,
    y: &[f32],
    w: &[f32],
    offset: Option<&[f32]>,
) -> Result<(), PbError> {
    let n = serve.0.n_rows as usize;
    let mut raw = vec![0.0_f64; n];
    crate::scoring::score_bank_binned(bank, cat_encoders, &serve.0, &mut raw)?;
    let raw32: Vec<f32> = raw.iter().map(|&v| v as f32).collect();
    bank.f0 += crate::engine::boost::reanchor_delta(Link::Logit, y, w, &raw32, offset)?;
    Ok(())
}

// =============================================================================================
// Multiclass (K>=3) set-level prune no-harm guard — the softmax analogue of the single-output
// guard that `TBoostRegressor._fit_and_prune` runs in Python (sklearn.py, "Set-level prune
// no-harm guard"). It lives in Rust because the K>=3 fit-select-deploy flow is orchestrated
// entirely inside `fit_multiclass_pruned_owned`: that is the only place the deployed
// `MultiClassModel` (pre-keepset, with its bag partition and membership) and the selection
// report are both in hand.
//
// WHY IT EXISTS. The keep-set is chosen from PER-TABLE leave-one-out held-out gains, which are
// structurally blind to mass carried JOINTLY by correlated tables: each table's marginal
// drop-gain is ~0 when its siblings absorb the signal, so a whole correlated family can be
// dropped while the fold evidence says the total cost is ~0. On the single-output path that
// mis-selection was measured detonating a split by +23% test deviance. Nobody had ever
// measured it on a K>=3 fit, because no guard existed there.
//
// THE EVIDENCE. Out-of-bag, exactly as the ungrouped single-output path: for each fit row `r`
// the bags that never drew `r` are its honest jury, and the jury mean of the per-bag per-class
// banks is an unbiased estimate of the deployed per-class logit at `r` (purify is linear at a
// fixed grid/measure, so the mean over ALL bags reproduces the soup's bank). A K>=3 bag draws
// its rows ONCE and fits all K columns on that subset, so the K classes share one membership
// vector and therefore one jury per row — the softmax over the per-class jury means is a
// coherent probability vector, not K independently-sampled ones. `fit_multiclass_bagged`
// publishes that membership on every per-class soup for this reason.
//
// BOTH ARMS ARE RE-ANCHORED before they are measured (`multiclass_intercept_shifts`), for the
// same reason the single-output guard re-anchors: dropping most of the tables moves the
// empirical class balance a lot (purification centres under the REFERENCE measure, not the
// empirical one), and the shipped artifact carries that correction — `prune_multiclass_to_
// keepset` re-anchors the deployed banks. Measuring un-anchored arms would test a level
// artifact the shipped model does not have. `level_shift_*` reports the correction so the
// artifact stays visible.
//
// HONEST LIMITS, mirroring the single-output guard's:
//   * A ~2-bag jury is noisier than the deployed 8-bag soup, and that variance sits in BOTH
//     arms — the measured ratio is diluted toward 1, so the bias errs toward SILENCE.
//   * Neither arm carries the deploy re-anchor's *full* pipeline: they are raw jury-mean banks
//     plus the intercept IPF. There is no rebalance or graduation on the K>=3 path at all, so
//     unlike the single-output OOB path there is no selected-arm-only correction being omitted
//     here — the two arms are treated symmetrically.
//   * The SELECTION is not honest with respect to this evidence (the keep-set was voted on by
//     folds of the selection slice, and the deploy fit saw every row). The guard's job is to
//     catch a catastrophic set-level miss, not to be an unbiased generalization estimate.
// =============================================================================================

/// Per-class out-of-bag evidence for a bagged [`MultiClassModel`], plus the shared jury size.
#[derive(Debug, Clone)]
pub struct MulticlassOobEvidence {
    /// `per_class[k]` is class `k`'s [`BagOobGroupSums`] over the same rows and the same groups.
    pub per_class: Vec<BagOobGroupSums>,
    /// Jury size per fit row — identical across classes (one row draw per bag, K columns).
    pub counts: Vec<u32>,
}

/// Whether [`multiclass_bag_oob_group_raw_sums`] can run: every class must carry the outer-bag
/// soup's runtime partition AND membership (so a single-bag fit or a deserialized model is out).
#[must_use]
pub fn multiclass_bag_oob_evidence_available(mc: &MultiClassModel) -> bool {
    !mc.classes.is_empty() && mc.classes.iter().all(bag_oob_evidence_available)
}

/// [`bag_oob_group_raw_sums`] run once per class of a multiclass soup, over the SAME `groups`.
///
/// Classes are processed sequentially (each is itself internally parallel and holds one bag's
/// bank live at a time — see [`bag_raw_scores_for_rows`] — so a parallel outer loop would
/// multiply peak memory by K for no throughput the inner loop is not already taking).
///
/// # Errors
/// [`PbError::InvalidInput`] if the model has no classes or a class lacks the bag partition /
/// membership; [`PbError::Internal`] if the classes disagree about which rows are out of bag
/// (they share one draw, so a disagreement means the membership was not published together);
/// propagated bank/scoring errors otherwise.
pub fn multiclass_bag_oob_group_raw_sums(
    mc: &MultiClassModel,
    serve: &ServeBinnedMatrix,
    w_measure: RefMeasure,
    mass: Option<&[f32]>,
    groups: &[Vec<FeatureSet>],
) -> Result<MulticlassOobEvidence, PbError> {
    multiclass_bag_oob_group_sums_with(mc, serve, w_measure, mass, groups, true)
}

/// [`multiclass_bag_oob_group_raw_sums`] with the full-bank arm optional (see
/// [`bag_oob_group_sums_with`]).
///
/// # Errors
/// As [`multiclass_bag_oob_group_raw_sums`].
pub fn multiclass_bag_oob_group_sums_with(
    mc: &MultiClassModel,
    serve: &ServeBinnedMatrix,
    w_measure: RefMeasure,
    mass: Option<&[f32]>,
    groups: &[Vec<FeatureSet>],
    with_full: bool,
) -> Result<MulticlassOobEvidence, PbError> {
    if mc.classes.is_empty() {
        return Err(PbError::InvalidInput {
            what: "multiclass out-of-bag evidence needs at least one class".into(),
        });
    }
    // Classes are independent and collect in class order (deterministic). The per-bag loop
    // inside stays sequential to honour its one-bag-at-a-time memory contract, so the peak here
    // is K bags' banks rather than one — the same K-way peak the keep-set apply already has.
    let per_class: Vec<BagOobGroupSums> = mc
        .classes
        .par_iter()
        .map(|model| {
            bag_oob_group_sums_with(model, serve, w_measure.clone(), mass, groups, with_full)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let counts = per_class
        .first()
        .ok_or_else(|| PbError::Internal {
            what: "multiclass out-of-bag evidence lost its first class".into(),
        })?
        .counts
        .clone();
    for (k, ev) in per_class.iter().enumerate().skip(1) {
        if ev.counts != counts {
            return Err(PbError::Internal {
                what: format!(
                    "multiclass out-of-bag jury for class {k} disagrees with class 0: the K \
                     classes of a bag share one row draw, so their membership must match"
                ),
            });
        }
    }
    Ok(MulticlassOobEvidence { per_class, counts })
}

/// The report-ranked re-admission ladder: dropped tables ordered by `mean_gain` descending and
/// cut into DOUBLING chunks (the first chunk is at most `|keep|` tables, then the grown keep-set
/// is the next chunk's cap, and so on).
///
/// Fixing the ladder BEFORE any evidence is scored is what lets the guard price every rung from
/// a single pass over the per-bag banks: rung `k` is the prefix union of chunks `0..k`, and
/// table scores are additive. This is the same rule the single-output Python guard uses.
#[must_use]
pub fn guard_readmission_chunks(
    keep: &[FeatureSet],
    table_scores: &[PruneTableScore],
) -> Vec<Vec<FeatureSet>> {
    guard_readmission_chunks_with_growth(keep, table_scores, 1.0)
}

/// [`guard_readmission_chunks`] with a finer ladder: each rung grows the kept set by `growth`
/// times its current size (`1.0` is the doubling ladder above; `0.2` re-admits 20% more tables
/// per rung, so a guard that fires lands near the SMALLEST keep-set within its bar instead of
/// overshooting to the next power of two — the K>=3 guard uses this, since its selection can
/// under-keep by a factor of four on a 700-table bank and the doubling ladder then jumps
/// straight past the honest optimum, measured on fremotor_payfreq 2026-09-07).
#[must_use]
pub fn guard_readmission_chunks_with_growth(
    keep: &[FeatureSet],
    table_scores: &[PruneTableScore],
    growth: f64,
) -> Vec<Vec<FeatureSet>> {
    guard_readmission_chunks_ranked(keep, table_scores, growth, |t| t.mean_gain)
}

/// The K>=3 ladder: rungs ordered by the DEPLOY bank's purified table variance (largest first)
/// rather than by selection-time drop-gain. The CV selection's fold models are unbagged and
/// realize far fewer supports than the bagged deploy soup (fremotor_payfreq 2026-09-07: ~265
/// per fold against 698 deployed), so most dropped deploy tables carry NO fold evidence and a
/// gain-ranked ladder re-admits them last — after every table that merely looked good on a
/// fold. Variance is the deploy model's own signal size and the ordering the
/// price-of-simplicity probes found robust on every panel.
#[must_use]
pub fn guard_readmission_chunks_by_variance(
    keep: &[FeatureSet],
    table_scores: &[PruneTableScore],
    growth: f64,
) -> Vec<Vec<FeatureSet>> {
    guard_readmission_chunks_ranked(keep, table_scores, growth, |t| t.variance)
}

fn guard_readmission_chunks_ranked(
    keep: &[FeatureSet],
    table_scores: &[PruneTableScore],
    growth: f64,
    rank: impl Fn(&PruneTableScore) -> f64,
) -> Vec<Vec<FeatureSet>> {
    let kept: std::collections::BTreeSet<&FeatureSet> = keep.iter().collect();
    let mut ranked: Vec<&FeatureSet> = table_scores
        .iter()
        .map(|t| &t.u)
        .filter(|u| !kept.contains(*u))
        .collect();
    // Stable sort by mean_gain desc, keyed by position so equal gains keep report order (the
    // Python guard's `sorted(..., key=-mean_gain)` is likewise stable).
    let gain: BTreeMap<&FeatureSet, f64> = table_scores.iter().map(|t| (&t.u, rank(t))).collect();
    ranked.sort_by(|a, b| {
        gain[b]
            .partial_cmp(&gain[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.dedup();
    let mut chunks: Vec<Vec<FeatureSet>> = Vec::new();
    let mut grown = keep.len();
    let mut rest: &[&FeatureSet] = &ranked;
    while !rest.is_empty() {
        // `growth * grown`, rounded up, never below one table — with `growth = 1.0` this is the
        // historical doubling ladder byte for byte.
        let step = (growth * grown as f64).ceil().max(1.0) as usize;
        let take = step.clamp(1, rest.len());
        chunks.push(rest[..take].iter().map(|u| (*u).clone()).collect());
        rest = &rest[take..];
        grown += take;
    }
    chunks
}

/// Rung growth of the K>=3 guard's re-admission ladder (see
/// [`guard_readmission_chunks_with_growth`]): each rung re-admits 20% more tables.
pub const MULTICLASS_GUARD_LADDER_GROWTH: f64 = 0.2;

/// What the multiclass guard did, serialized into `pruning_report_["guard"]`.
#[derive(Debug, Clone, Serialize)]
pub struct MulticlassGuardReport {
    /// Whether the guard was asked to run at all.
    pub enabled: bool,
    /// Whether the breach test tripped and the keep-set grew.
    pub fired: bool,
    /// Evidence estimator actually used (`"oob"`), or `None` when skipped.
    pub evidence: Option<String>,
    /// Why no evidence was scored (`None` when the guard ran).
    pub skipped: Option<String>,
    /// Rows carrying at least one out-of-bag bag.
    pub oob_rows: u64,
    /// Fit rows every bag drew (no jury, dropped from the evidence).
    pub oob_rows_uncovered: u64,
    /// Mean jury size over the covered rows.
    pub oob_mean_jury: f64,
    /// Bags in the deployed soup.
    pub n_bags: u64,
    /// Relative-deviance tolerance the breach test used.
    pub tol: f64,
    /// Weighted mean multinomial deviance of the FULL arm — every candidate table id the
    /// selector could have kept — re-anchored. This is the breach threshold's base.
    pub dev_full: f64,
    /// Diagnostic: the same, for the deploy soup's WHOLE realized bank (which can carry
    /// supports the selection fit never grew, so no keep-set can reach it). A large
    /// `dev_bank` < `dev_full` gap means the deploy fit found structure the selector never
    /// voted on — worth knowing, never the guard's threshold.
    pub dev_bank: f64,
    /// Same, for the selected keep-set before any re-admission.
    pub dev_selected_initial: f64,
    /// Same, for the keep-set actually shipped.
    pub dev_selected_final: f64,
    /// Tables in the keep-set before / after the ladder.
    pub kept_initial: u64,
    /// Tables in the shipped keep-set.
    pub kept_final: u64,
    /// Re-admission chunks consumed (`0` ⇒ silent).
    pub steps: u64,
    /// Rungs the ladder could have offered.
    pub rungs_available: u64,
    /// Per-class intercept re-anchor applied to the SELECTED arm — the level artifact the guard
    /// would otherwise have been testing.
    pub level_shift_selected: Vec<f64>,
    /// Same, for the full-bank arm.
    pub level_shift_full: Vec<f64>,
    /// Weighted mean over evidence rows of the per-row standard deviation of the K class
    /// logits, for the shipped keep-set. Compare with `spread_full`: a ratio below 1 is the
    /// COMPRESSION a slope/temperature re-anchor would exist to undo.
    pub spread_selected: f64,
    /// Same, for the full-bank arm.
    pub spread_full: f64,
    /// The softmax analogue of the single-output post-prune SLOPE, MEASURED but never applied:
    /// the shared temperature `b` minimizing the honest-row deviance of `softmax(a_k(b) + b·F_k)`
    /// over the shipped bank's class logits, with the intercepts `a_k(b)` re-solved at each `b`
    /// (profile likelihood — the intercepts are the re-anchor the artifact already carries, so
    /// `b` is the only free scale). `b = 1` is the shipped model. See
    /// `slope_gain` for what it would be worth.
    ///
    /// READ IT AGAINST [`Self::slope_b_full`], NOT AGAINST 1. On the out-of-bag path the arm
    /// measured is a ~2-bag JURY MEAN, and averaging fewer models shrinks the score spread, so
    /// the ABSOLUTE temperature of any OOB arm is biased above 1 by the jury size alone —
    /// measured `b ≈ 1.12` on an arm whose spread ratio against the full bank is 0.9994, i.e.
    /// essentially all artifact. The single-output slope handles exactly this by measuring the
    /// keep-set's slope RELATIVE to the full bank on the same rows with the same estimator; the
    /// ratio `slope_b / slope_b_full` is that relative statistic here, and it is the number a
    /// decision may be taken on. On the carve path there is no jury, so both are absolute.
    pub slope_b: f64,
    /// The same profile temperature fitted to the FULL arm — the reference that cancels the
    /// out-of-bag jury-shrinkage bias out of [`Self::slope_b`] (see its note).
    pub slope_b_full: f64,
    /// Honest-row deviance at `slope_b` minus the deviance at `b = 1`, i.e. the (non-positive)
    /// improvement a temperature re-anchor could buy on the rows it was fitted on — an
    /// OPTIMISTIC bound, since `b` is chosen on exactly those rows. A `slope_b` within a
    /// fraction of a percent of 1 with a `slope_gain` in the 1e-5 range is the signature of a
    /// prune that does not compress the class-score scale, and therefore of a slope re-anchor
    /// with nothing to undo.
    pub slope_gain: f64,
    /// Wall-clock seconds spent building the per-class per-bag evidence.
    pub evidence_seconds: f64,
    /// Intercept-only (class-prior) deviance on the evidence rows — the skill scale's zero.
    pub dev_null: f64,
    /// Paired per-row standard error of the selected-minus-full loss on the evidence rows.
    pub gap_se: f64,
    /// The bar the guard actually used, in deviance units above `dev_full`.
    pub bar: f64,
}

impl MulticlassGuardReport {
    /// A guard report for a fit the guard could not measure: `enabled` as asked, nothing fired,
    /// and `why` recorded so the skip is visible in `pruning_report_["guard"]` rather than
    /// silently indistinguishable from a silent guard.
    #[must_use]
    pub fn skipped(enabled: bool, why: &str) -> Self {
        Self {
            enabled,
            fired: false,
            evidence: None,
            skipped: Some(why.to_string()),
            oob_rows: 0,
            oob_rows_uncovered: 0,
            oob_mean_jury: 0.0,
            n_bags: 0,
            tol: 0.0,
            dev_full: f64::NAN,
            dev_bank: f64::NAN,
            dev_selected_initial: f64::NAN,
            dev_selected_final: f64::NAN,
            kept_initial: 0,
            kept_final: 0,
            steps: 0,
            rungs_available: 0,
            level_shift_selected: Vec::new(),
            level_shift_full: Vec::new(),
            spread_selected: f64::NAN,
            spread_full: f64::NAN,
            slope_b: f64::NAN,
            slope_b_full: f64::NAN,
            slope_gain: f64::NAN,
            evidence_seconds: 0.0,
            dev_null: f64::NAN,
            gap_se: f64::NAN,
            bar: f64::NAN,
        }
    }
}

/// Weighted mean multinomial deviance (`-Σ w·ln p_y / Σ w`) over `raw` (`[class][row]`).
///
/// NOTE the divisor: [`crate::engine::boost::multiclass_deviance_for_rows`] ALREADY returns the
/// per-unit-weight mean, so nothing is divided again here or in pruning's fold evaluator.
fn multiclass_mean_deviance(raw: &[Vec<f64>], labels: &[u32], w: &[f32]) -> Result<f64, PbError> {
    let cols: Vec<Vec<f32>> = raw
        .iter()
        .map(|c| c.iter().map(|&v| v as f32).collect())
        .collect();
    let rows: Vec<usize> = (0..labels.len()).collect();
    crate::engine::boost::multiclass_deviance_for_rows(&cols, labels, w, &rows)
}

/// Weighted mean over rows of the per-row standard deviation of the K class logits — the scale
/// statistic the slope/temperature question turns on (see [`MulticlassGuardReport::spread_selected`]).
fn multiclass_mean_spread(raw: &[Vec<f64>], w: &[f32]) -> f64 {
    let n_classes = raw.len();
    if n_classes == 0 {
        return f64::NAN;
    }
    let n = raw[0].len();
    let mut acc = 0.0_f64;
    let mut mass = 0.0_f64;
    for i in 0..n {
        let mut m = 0.0_f64;
        for k in 0..n_classes {
            m += raw[k][i];
        }
        m /= n_classes as f64;
        let mut v = 0.0_f64;
        for k in 0..n_classes {
            let d = raw[k][i] - m;
            v += d * d;
        }
        let sd = (v / n_classes as f64).sqrt();
        let wi = f64::from(w[i]);
        acc += wi * sd;
        mass += wi;
    }
    if mass > 0.0 {
        acc / mass
    } else {
        f64::NAN
    }
}

/// Which honest rows the multiclass guard measures on.
#[derive(Debug, Clone, Copy)]
pub enum MulticlassGuardEvidence<'a> {
    /// Out-of-bag jury means over the deploy soup's per-class bag banks. Honest exactly when no
    /// group can straddle the in-bag/out-of-bag boundary — bags are drawn at ROW granularity,
    /// so this is for UNGROUPED fits (and degenerate all-singleton groupings). Costs the fit
    /// nothing: no carve, and at the shipped `n_bags=8` most rows get a jury.
    OutOfBag,
    /// The shared group-honest ES holdout the deploy fit already keeps out of every bag (mask
    /// over fit rows). The panel analogue: an out-of-bag row's panel-mates are almost surely in
    /// bag, which is the memorization the group carve exists to stop, so OOB is NOT honest
    /// there and the carve stays the evidence.
    Carve(&'a [bool]),
}

/// The evidence arms, materialized over the chosen rows: the full-bank arm and one arm per
/// ladder rung group, all on the link scale, `[class][evidence row]`.
pub struct GuardArms {
    /// Fit-row indices that carry evidence (a jury, or the carve).
    pub rows: Vec<usize>,
    /// Jury size per evidence row (all `1` on the carve path).
    pub counts: Vec<f64>,
    /// Full-model logits `[class][row]` (intercept + every table).
    pub full: Vec<Vec<f64>>,
    /// `group[g][class][row]` — group `g`'s tables ALONE (no intercept).
    pub group: Vec<Vec<Vec<f64>>>,
    /// Bank intercepts, `[class][row]`.
    pub f0: Vec<Vec<f64>>,
}

/// The ranked path's admission order and prefix grid (shared by every selector that walks it).
///
/// Interaction supports are ranked by `variance` (descending, ids break ties) and admitted subject to
/// heredity: a k-way support enters only once all its (k-1)-way subsets are in (mains are sticky,
/// so pairs are unconditioned); a deferred support enters as soon as it qualifies. Supports whose
/// subsets never enter are never admitted. Returns `(mains, sequence, sizes)` where `sizes` is a
/// geometric grid of prefix lengths over `sequence` including 0 and the full length.
#[must_use]
pub fn ranked_path_sequence(
    supports: &[(FeatureSet, f64)],
    steps: usize,
) -> (Vec<FeatureSet>, Vec<FeatureSet>, Vec<usize>) {
    let mut mains: Vec<FeatureSet> = supports
        .iter()
        .filter(|(u, _)| u.order() == 1)
        .map(|(u, _)| u.clone())
        .collect();
    mains.sort();
    mains.dedup();
    let mut inter: Vec<(FeatureSet, f64)> = supports
        .iter()
        .filter(|(u, _)| u.order() > 1)
        .cloned()
        .collect();
    inter.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    inter.dedup_by(|a, b| a.0 == b.0);
    let mut admitted: std::collections::BTreeSet<FeatureSet> = mains.iter().cloned().collect();
    let mut seq: Vec<FeatureSet> = Vec::new();
    let mut pending: Vec<FeatureSet> = Vec::new();
    let qualifies = |v: &FeatureSet, admitted: &std::collections::BTreeSet<FeatureSet>| -> bool {
        if v.order() <= 2 {
            return true;
        }
        let ids: Vec<u32> = v.0.iter().map(|f| f.0).collect();
        (0..ids.len()).all(|skip| {
            let sub: Vec<u32> = ids
                .iter()
                .enumerate()
                .filter_map(|(i, id)| (i != skip).then_some(*id))
                .collect();
            admitted.contains(&FeatureSet::new(&sub))
        })
    };
    for (u, _) in inter {
        pending.push(u);
        let mut moved = true;
        while moved {
            moved = false;
            let mut i = 0;
            while i < pending.len() {
                if qualifies(&pending[i], &admitted) {
                    let v = pending.remove(i);
                    admitted.insert(v.clone());
                    seq.push(v);
                    moved = true;
                } else {
                    i += 1;
                }
            }
        }
    }
    let n = seq.len();
    let steps = steps.max(2);
    let mut sizes: std::collections::BTreeSet<usize> = [0, n].into_iter().collect();
    if n >= 1 {
        let ln_n = (n as f64).ln();
        for s in 0..steps {
            let x = (ln_n * s as f64 / (steps - 1) as f64).exp();
            sizes.insert((x.round() as usize).min(n));
        }
    }
    (mains, seq, sizes.into_iter().collect())
}

/// What the multiclass ranked path chose, serialized into `pruning_report_["path"]`.
#[derive(Debug, Clone, Serialize)]
pub struct RankedPathReport {
    /// Prefix lengths over the admission sequence that were scored.
    pub sizes: Vec<u32>,
    /// Out-of-bag mean multinomial deviance at each prefix.
    pub oob_deviance: Vec<f64>,
    /// Index into `sizes` of the deployed prefix.
    pub best: u32,
    /// Fit rows that carried an out-of-bag jury.
    pub oob_rows: u64,
}

/// Multiclass ranked-path keep-set: the [`ranked_path_sequence`] prefixes scored in ONE pass on
/// the soup's per-class out-of-bag juries (softmax deviance), deploying the larger of the smallest
/// prefix capturing `fraction` of the out-of-bag improvement over the mains-only model and the
/// smallest prefix within `tolerance` of the best out-of-bag deviance (fraction 1.0 = the minimum). `None` when
/// the fit has no out-of-bag evidence or fewer than `min_rows` jury rows.
pub fn multiclass_ranked_path(
    mc: &MultiClassModel,
    serve: &ServeBinnedMatrix,
    labels: &[u32],
    w: &[f32],
    w_measure: RefMeasure,
    supports: &[(FeatureSet, f64)],
    steps: usize,
    fraction: f64,
    tolerance: f64,
    min_rows: usize,
) -> Result<Option<(Vec<FeatureSet>, RankedPathReport)>, PbError> {
    if !multiclass_bag_oob_evidence_available(mc) {
        return Ok(None);
    }
    let (mains, seq, sizes) = ranked_path_sequence(supports, steps);
    let mut groups: Vec<Vec<FeatureSet>> = vec![mains.clone()];
    for pair in sizes.windows(2) {
        groups.push(seq[pair[0]..pair[1]].to_vec());
    }
    let mass = measure_mass(&w_measure, w, None);
    // The path reads the intercept and group arms only, so the full-bank arm is not scored.
    let arms = guard_arms_oob_with(mc, serve, w_measure, mass.as_deref(), &groups, false)?;
    if arms.rows.len() < min_rows {
        return Ok(None);
    }
    let lab: Vec<u32> = arms.rows.iter().map(|&r| labels[r]).collect();
    let wv: Vec<f32> = arms.rows.iter().map(|&r| w[r]).collect();
    let mut raw: Vec<Vec<f64>> = arms.f0.clone();
    let mut devs: Vec<f64> = Vec::with_capacity(groups.len());
    for g in &arms.group {
        for (k, col) in raw.iter_mut().enumerate() {
            for (v, add) in col.iter_mut().zip(&g[k]) {
                *v += add;
            }
        }
        devs.push(multiclass_mean_deviance(&raw, &lab, &wv)?);
    }
    let best = devs
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);
    // Smallest prefix capturing `fraction` of the out-of-bag improvement over the mains-only model.
    let gain = devs[0] - devs[best];
    let best = if gain > 0.0 && fraction < 1.0 {
        let first = devs.first().copied().unwrap_or(0.0);
        let dmin = devs.get(best).copied().unwrap_or(first);
        let i_frac = devs
            .iter()
            .position(|&d| first - d >= fraction * gain)
            .unwrap_or(best);
        // ...and within `tolerance` of the best deviance: the larger of the two prefixes
        let i_tol = devs
            .iter()
            .position(|&d| d <= dmin * (1.0 + tolerance))
            .unwrap_or(best);
        i_frac.max(i_tol)
    } else {
        best
    };
    let mut keep = mains;
    keep.extend(seq[..sizes[best]].iter().cloned());
    Ok(Some((
        keep,
        RankedPathReport {
            sizes: sizes
                .iter()
                .map(|&s| u32::try_from(s).unwrap_or(u32::MAX))
                .collect(),
            oob_deviance: devs,
            best: u32::try_from(best).unwrap_or(u32::MAX),
            oob_rows: u64::try_from(arms.rows.len()).unwrap_or(u64::MAX),
        },
    )))
}

/// Out-of-bag arms: the jury mean of each class's per-bag banks (see the module note above).
pub fn guard_arms_oob(
    mc: &MultiClassModel,
    serve: &ServeBinnedMatrix,
    w_measure: RefMeasure,
    mass: Option<&[f32]>,
    groups: &[Vec<FeatureSet>],
) -> Result<GuardArms, PbError> {
    guard_arms_oob_with(mc, serve, w_measure, mass, groups, true)
}

/// [`guard_arms_oob`] with the full-bank arm optional: `with_full = false` leaves `full` at
/// zero (see [`bag_oob_group_sums_with`]) for a caller that never reads it.
fn guard_arms_oob_with(
    mc: &MultiClassModel,
    serve: &ServeBinnedMatrix,
    w_measure: RefMeasure,
    mass: Option<&[f32]>,
    groups: &[Vec<FeatureSet>],
    with_full: bool,
) -> Result<GuardArms, PbError> {
    let ev = multiclass_bag_oob_group_sums_with(mc, serve, w_measure, mass, groups, with_full)?;
    let rows: Vec<usize> = ev
        .counts
        .iter()
        .enumerate()
        .filter_map(|(r, &c)| (c > 0).then_some(r))
        .collect();
    let counts: Vec<f64> = rows.iter().map(|&r| f64::from(ev.counts[r])).collect();
    // Jury MEAN per class per evidence row, from the per-class jury sums.
    let mean_of = |src: Vec<&Vec<f64>>| -> Vec<Vec<f64>> {
        src.into_iter()
            .map(|column| {
                rows.iter()
                    .enumerate()
                    .map(|(i, &r)| column[r] / counts[i])
                    .collect()
            })
            .collect()
    };
    let full = mean_of(ev.per_class.iter().map(|p| &p.full_sum).collect());
    let f0 = mean_of(ev.per_class.iter().map(|p| &p.f0_sum).collect());
    let group: Vec<Vec<Vec<f64>>> = (0..groups.len())
        .map(|g| mean_of(ev.per_class.iter().map(|p| &p.group_sums[g]).collect()))
        .collect();
    Ok(GuardArms {
        rows,
        counts,
        full,
        group,
        f0,
    })
}

/// Carve arms: one purified bank per class (the deploy fit's own), scored on the held-out rows
/// whole and per ladder group. `retain_tables` is a cheap filter, so every rung is priced from
/// the SAME K banks — no rung needs its own bank build, exactly as the out-of-bag path's
/// additivity buys there.
pub fn guard_arms_carve(
    mc: &MultiClassModel,
    serve: &ServeBinnedMatrix,
    w_measure: RefMeasure,
    mass: Option<&[f32]>,
    groups: &[Vec<FeatureSet>],
    mask: &[bool],
) -> Result<GuardArms, PbError> {
    let n = serve.0.n_rows as usize;
    if mask.len() != n {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "guard carve mask covers {} rows but the deploy serve has {n}",
                mask.len()
            ),
        });
    }
    let rows: Vec<usize> = (0..n).filter(|&r| mask[r]).collect();
    let row_u32: Vec<u32> = rows.iter().map(|&r| r as u32).collect();
    let n_classes = mc.classes.len();
    // One bank per class, scored on the carve rows — classes in parallel, collected in class
    // order (deterministic; each class's numbers are its own).
    type ClassArm = (Vec<f64>, f64, Vec<Vec<f64>>);
    let per_class: Vec<ClassArm> = mc
        .classes
        .par_iter()
        .map(|model| -> Result<ClassArm, PbError> {
            let bank = model.explain_bank_with_mass(
                serve,
                w_measure.clone(),
                crate::explain::TableBudget::default(),
                false,
                mass,
            )?;
            let group_banks: Vec<TableBank> = groups
                .iter()
                .map(|g| {
                    let mut b = retain_tables(&bank, g);
                    b.f0 = 0.0;
                    b
                })
                .collect();
            let mut banks: Vec<&TableBank> = Vec::with_capacity(groups.len() + 1);
            banks.push(&bank);
            banks.extend(group_banks.iter());
            let width = banks.len();
            let mut flat = vec![0.0_f64; row_u32.len() * width];
            crate::scoring::score_banks_rows(
                &banks,
                &model.schema.cat_encoders,
                &serve.0,
                &row_u32,
                &mut flat,
            )?;
            let full_c: Vec<f64> = (0..rows.len()).map(|i| flat[i * width]).collect();
            let groups_c: Vec<Vec<f64>> = (0..groups.len())
                .map(|g| (0..rows.len()).map(|i| flat[i * width + g + 1]).collect())
                .collect();
            Ok((full_c, bank.f0, groups_c))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut full: Vec<Vec<f64>> = Vec::with_capacity(n_classes);
    let mut f0: Vec<Vec<f64>> = Vec::with_capacity(n_classes);
    let mut group: Vec<Vec<Vec<f64>>> = (0..groups.len()).map(|_| Vec::new()).collect();
    for (full_c, f0_c, groups_c) in per_class {
        full.push(full_c);
        f0.push(vec![f0_c; rows.len()]);
        for (g, slot) in group.iter_mut().enumerate() {
            slot.push(groups_c[g].clone());
        }
    }
    Ok(GuardArms {
        counts: vec![1.0; rows.len()],
        rows,
        full,
        group,
        f0,
    })
}

/// The shared-temperature profile likelihood: scan `b`, re-solving the per-class intercepts at
/// each candidate (so `b` is the only free scale, the intercepts being the re-anchor the shipped
/// artifact already carries), and return `(b*, dev(b*) - dev_at_one)`.
///
/// A coarse geometric bracket followed by a fixed golden-section refinement — deterministic, no
/// convergence test, no derivative of a function whose intercepts are themselves an inner solve.
/// `raw` is the UN-anchored `[class][row]` score of the bank in question.
///
/// # Errors
/// Propagated deviance/re-anchor failures.
fn multiclass_profile_temperature(
    raw: &[Vec<f64>],
    labels: &[u32],
    w: &[f32],
    dev_at_one: f64,
) -> Result<(f64, f64), PbError> {
    let dev_at = |b: f64| -> Result<f64, PbError> {
        let mut scaled: Vec<Vec<f64>> = raw
            .iter()
            .map(|c| c.iter().map(|&v| v * b).collect())
            .collect();
        let shift = multiclass_intercept_shifts(&scaled, labels, w, "slope probe")?;
        for (k, s) in shift.iter().enumerate() {
            for v in &mut scaled[k] {
                *v += s;
            }
        }
        multiclass_mean_deviance(&scaled, labels, w)
    };
    // Coarse bracket over a geometric grid spanning half to double the shipped scale.
    const GRID: [f64; 11] = [0.5, 0.65, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5, 2.0];
    let mut best = (1.0_f64, dev_at_one);
    for &b in &GRID {
        let d = dev_at(b)?;
        if d < best.1 {
            best = (b, d);
        }
    }
    // Golden-section refinement inside the bracketing grid cell (fixed 24 iterations: the
    // interval shrinks by 0.618 each time, so this pins b to ~1e-5 of the cell width).
    let idx = GRID.iter().position(|&g| g == best.0);
    let (mut lo, mut hi) = match idx {
        Some(i) => (
            GRID.get(i.saturating_sub(1)).copied().unwrap_or(GRID[0]),
            GRID.get(i + 1).copied().unwrap_or(2.0),
        ),
        None => (0.9, 1.1),
    };
    const INV_PHI: f64 = 0.618_033_988_749_894_9;
    let mut c = hi - (hi - lo) * INV_PHI;
    let mut d = lo + (hi - lo) * INV_PHI;
    let (mut fc, mut fd) = (dev_at(c)?, dev_at(d)?);
    for _ in 0..24 {
        if fc < fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - (hi - lo) * INV_PHI;
            fc = dev_at(c)?;
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + (hi - lo) * INV_PHI;
            fd = dev_at(d)?;
        }
    }
    let b = 0.5 * (lo + hi);
    let dev_b = dev_at(b)?;
    if dev_b < best.1 {
        best = (b, dev_b);
    }
    Ok((best.0, best.1 - dev_at_one))
}

/// Run the multiclass set-level no-harm guard and return the keep-set to ship.
///
/// `keep` is the honestly-selected keep-set; `table_scores` is the selection report's per-table
/// diagnostics (the re-admission ranking). The returned keep-set is either `keep` unchanged
/// (the silent case — the shipped artifact is then byte-identical to a guard-off fit) or `keep`
/// grown by whole ranked chunks until the selected bank's multinomial deviance on the honest
/// rows is within `tol` of the full bank's. Only the keep-set ever grows; binary table dropping
/// is preserved.
///
/// # Errors
/// Propagated evidence/bank/scoring failures. A model that cannot supply evidence is NOT an
/// error — it returns `keep` with a `skipped` report.
pub fn multiclass_prune_guard(
    mc: &MultiClassModel,
    serve: &ServeBinnedMatrix,
    labels: &[u32],
    w: &[f32],
    w_measure: RefMeasure,
    keep: &[FeatureSet],
    table_scores: &[PruneTableScore],
    evidence: MulticlassGuardEvidence<'_>,
    tol: f64,
    min_rows: usize,
    guard_z: f64,
    guard_floor: f64,
) -> Result<(Vec<FeatureSet>, MulticlassGuardReport), PbError> {
    // ASK, don't catch (see the single-output guard's note): a model with no bag partition or
    // membership is a legitimate "no evidence" outcome, but anything the evidence pass itself
    // raises is a real bug and must surface.
    if matches!(evidence, MulticlassGuardEvidence::OutOfBag)
        && !multiclass_bag_oob_evidence_available(mc)
    {
        return Ok((
            keep.to_vec(),
            MulticlassGuardReport::skipped(true, "no out-of-bag evidence (single-bag fit)"),
        ));
    }
    let chunks =
        guard_readmission_chunks_by_variance(keep, table_scores, MULTICLASS_GUARD_LADDER_GROWTH);
    // Group 0 is the FULL arm: every table id the selector had on the table (the keep-set plus
    // everything it dropped). It is NOT the deploy bag's whole realized bank — the deploy fit is
    // a second, full-data fit and can realize supports the selection fit never saw, which no
    // keep-set can ever name. Comparing against those would make the ladder structurally unable
    // to clear its own threshold. `dev_bank` below reports the whole-bank deviance anyway so
    // that structural gap stays visible instead of being quietly defined away.
    let mut all_ids: Vec<FeatureSet> = keep.to_vec();
    for chunk in &chunks {
        all_ids.extend(chunk.iter().cloned());
    }
    let mut groups: Vec<Vec<FeatureSet>> = Vec::with_capacity(chunks.len() + 2);
    groups.push(all_ids);
    groups.push(keep.to_vec());
    groups.extend(chunks.iter().cloned());

    let t_ev = std::time::Instant::now();
    let mass = measure_mass(&w_measure, w, None);
    let (arms, tag) = match evidence {
        MulticlassGuardEvidence::OutOfBag => (
            guard_arms_oob(mc, serve, w_measure, mass.as_deref(), &groups)?,
            "oob",
        ),
        MulticlassGuardEvidence::Carve(mask) => (
            guard_arms_carve(mc, serve, w_measure, mass.as_deref(), &groups, mask)?,
            "carve",
        ),
    };
    let evidence_seconds = t_ev.elapsed().as_secs_f64();

    let n_classes = mc.classes.len();
    let n_bags = mc
        .classes
        .first()
        .and_then(|m| m.bag_spans.as_ref())
        .map_or(0, Vec::len);
    let n_val = arms.rows.len();
    let mut report = MulticlassGuardReport::skipped(true, "");
    report.n_bags = n_bags as u64;
    report.tol = tol;
    report.oob_rows = n_val as u64;
    report.oob_rows_uncovered = ((serve.0.n_rows as usize) - n_val) as u64;
    report.kept_initial = keep.len() as u64;
    report.kept_final = keep.len() as u64;
    report.rungs_available = chunks.len() as u64;
    report.evidence_seconds = evidence_seconds;
    if n_val < min_rows {
        report.skipped = Some(format!("evidence rows {n_val} < {min_rows} [{tag}]"));
        return Ok((keep.to_vec(), report));
    }
    report.skipped = None;
    report.evidence = Some(tag.into());

    let ev_labels: Vec<u32> = arms.rows.iter().map(|&r| labels[r]).collect();
    let ev_w: Vec<f32> = arms.rows.iter().map(|&r| w[r]).collect();
    report.oob_mean_jury = arms.counts.iter().sum::<f64>() / (n_val as f64);

    // Full arm: the intercept plus group 0 (every candidate id). Rung 0 is the deployed
    // keep-set; each further rung adds one chunk. Every rung is the running PREFIX union from
    // group 1 on, so the intercept enters once and each group accumulates on top.
    let mut raw_full = arms.f0.clone();
    for k in 0..n_classes {
        for i in 0..n_val {
            raw_full[k][i] += arms.group[0][k][i];
        }
    }
    let mut rungs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(groups.len() - 1);
    let mut acc = arms.f0.clone();
    for g in 1..groups.len() {
        for k in 0..n_classes {
            for i in 0..n_val {
                acc[k][i] += arms.group[g][k][i];
            }
        }
        rungs.push(acc.clone());
    }
    // Diagnostic only: the deploy soup's WHOLE realized bank, which includes supports the
    // selection fit never grew and no keep-set can name. Never the breach threshold.
    let mut raw_bank = arms.full.clone();
    let shift_bank = multiclass_intercept_shifts(&raw_bank, &ev_labels, &ev_w, "guard bank arm")?;
    for (k, s) in shift_bank.iter().enumerate() {
        for v in &mut raw_bank[k] {
            *v += s;
        }
    }
    report.dev_bank = multiclass_mean_deviance(&raw_bank, &ev_labels, &ev_w)?;

    // Both arms carry the deploy re-anchor before they are measured: dropping most of the
    // tables moves the empirical class balance a lot (purification centres under the REFERENCE
    // measure, not the empirical one) and the shipped artifact carries that correction, so an
    // un-anchored comparison would test a level artifact the shipped model does not have.
    let shift_full = multiclass_intercept_shifts(&raw_full, &ev_labels, &ev_w, "guard full arm")?;
    for (k, s) in shift_full.iter().enumerate() {
        for v in &mut raw_full[k] {
            *v += s;
        }
    }
    report.level_shift_full = shift_full;
    report.dev_full = multiclass_mean_deviance(&raw_full, &ev_labels, &ev_w)?;
    report.spread_full = multiclass_mean_spread(&raw_full, &ev_w);

    // (deviance, per-class intercept shift, anchored raw logits) for rung `k`.
    type AnchoredRung = (f64, Vec<f64>, Vec<Vec<f64>>);
    let anchored_rung = |k: usize| -> Result<AnchoredRung, PbError> {
        let mut raw = rungs[k].clone();
        let shift = multiclass_intercept_shifts(&raw, &ev_labels, &ev_w, "guard selected arm")?;
        for (c, s) in shift.iter().enumerate() {
            for v in &mut raw[c] {
                *v += s;
            }
        }
        let dev = multiclass_mean_deviance(&raw, &ev_labels, &ev_w)?;
        Ok((dev, shift, raw))
    };

    let (mut dev_sel, mut shift_sel, mut raw_sel) = anchored_rung(0)?;
    report.dev_selected_initial = dev_sel;
    // THE BAR. The raw relative tolerance `tol` alone can never fire on a multiclass log-loss:
    // the loss is dominated by irreducible class entropy, so a prune that costs a quarter of the
    // model's whole skill over the GLM moves the raw deviance by ~0.2% (measured 2026-09-07:
    // pg16 0.16%, fremotor 0.26%, prudential 0.15% — against a 5% bar). So the bar is
    // SE-aware, the K>=3 reading of the single-output guard's downward tightening
    // (`_guard_tol_effective`): `bar = min(tol * dev_full, max(guard_z * SE(gap), guard_floor *
    // (dev_null - dev_full)))`, where `SE(gap)` is the paired per-row standard error of the
    // selected-minus-full loss on the evidence rows and `dev_null` the intercept-only loss there.
    // `guard_z = 0` restores the raw bar exactly.
    let raw_bar = report.dev_full * tol;
    let sw: f64 = ev_w
        .iter()
        .map(|&x| f64::from(x))
        .sum::<f64>()
        .max(f64::MIN_POSITIVE);
    let mut prior = vec![0.0_f64; n_classes];
    for (l, &wi) in ev_labels.iter().zip(&ev_w) {
        prior[*l as usize] += f64::from(wi);
    }
    let dev_null: f64 = ev_labels
        .iter()
        .zip(&ev_w)
        .map(|(l, &wi)| -f64::from(wi) * (prior[*l as usize] / sw).max(f64::MIN_POSITIVE).ln())
        .sum::<f64>()
        / sw;
    let floor_bar = guard_floor * (dev_null - report.dev_full).max(0.0);
    let row_loss = |raw: &[Vec<f64>], i: usize| -> f64 {
        let m = (0..n_classes)
            .map(|k| raw[k][i])
            .fold(f64::NEG_INFINITY, f64::max);
        let lse = (0..n_classes)
            .map(|k| (raw[k][i] - m).exp())
            .sum::<f64>()
            .ln()
            + m;
        lse - raw[ev_labels[i] as usize][i]
    };
    // The bar is evaluated PER RUNG: the paired SE of a rung's own gap, so a rung that has
    // nearly closed the gap is judged by its own (small) uncertainty, not by the wide SE of the
    // initial, badly under-kept arm.
    let gap_and_bar = |raw_arm: &[Vec<f64>]| -> (f64, f64, f64) {
        let d: Vec<f64> = (0..n_val)
            .map(|i| row_loss(raw_arm, i) - row_loss(&raw_full, i))
            .collect();
        let gap = d
            .iter()
            .zip(&ev_w)
            .map(|(di, &wi)| di * f64::from(wi))
            .sum::<f64>()
            / sw;
        let se = (d
            .iter()
            .zip(&ev_w)
            .map(|(di, &wi)| (f64::from(wi) * (di - gap)).powi(2))
            .sum::<f64>()
            / (sw * sw))
            .sqrt();
        let bar = if guard_z > 0.0 && se.is_finite() {
            raw_bar.min((guard_z * se).max(floor_bar))
        } else {
            raw_bar
        };
        (gap, se, bar)
    };
    report.dev_null = dev_null;
    let (mut gap, mut gap_se, mut bar) = gap_and_bar(&raw_sel);
    let mut steps = 0_usize;
    if gap.is_finite() && gap > bar {
        while gap > bar && steps < chunks.len() {
            steps += 1;
            let (d, s, r) = anchored_rung(steps)?;
            dev_sel = d;
            shift_sel = s;
            raw_sel = r;
            let (g2, se2, bar2) = gap_and_bar(&raw_sel);
            gap = g2;
            gap_se = se2;
            bar = bar2;
        }
        report.fired = true;
    }
    report.gap_se = gap_se;
    report.bar = bar;
    report.steps = steps as u64;
    report.dev_selected_final = dev_sel;
    report.level_shift_selected = shift_sel;
    report.spread_selected = multiclass_mean_spread(&raw_sel, &ev_w);
    // MEASURE the temperature, do not apply it. `rungs[final_k]` is the un-anchored score of
    // exactly the bank that is about to ship, which is the only array here on which the
    // question is honest (fitting a scale on an earlier rung and applying it to a bank the
    // ladder later grew is catastrophic, not merely wasteful — the single-output slope's own
    // note records a split where that took +0.0926 to +0.0091).
    let (b, gain) = multiclass_profile_temperature(&rungs[steps], &ev_labels, &ev_w, dev_sel)?;
    report.slope_b = b;
    report.slope_gain = gain;
    // The reference arm, fitted the same way on the same rows: `slope_b / slope_b_full` is the
    // jury-bias-free relative scale (see `slope_b`'s note). `raw_full` is already re-anchored,
    // which only shifts intercepts the profile re-solves anyway.
    report.slope_b_full =
        multiclass_profile_temperature(&raw_full, &ev_labels, &ev_w, report.dev_full)?.0;

    let mut shipped = keep.to_vec();
    for chunk in chunks.iter().take(steps) {
        shipped.extend(chunk.iter().cloned());
    }
    report.kept_final = shipped.len() as u64;
    Ok((shipped, report))
}

/// PROBE (2026-09-07): union of the realized supports across the K class banks, with the
/// per-support purified variance SUMMED over classes and the order — the candidate list the
/// K>=3 prune walks, exposed so a keep-set can be chosen outside the walk.
pub fn multiclass_realized_supports(
    mc: &MultiClassModel,
    serve: &ServeBinnedMatrix,
    w_measure: RefMeasure,
    w: &[f32],
) -> Result<Vec<(FeatureSet, f64)>, PbError> {
    let mass = measure_mass(&w_measure, w, None);
    let banks: Vec<TableBank> = mc
        .classes
        .par_iter()
        .map(|m| {
            m.explain_bank_with_mass(
                serve,
                w_measure.clone(),
                crate::explain::TableBudget::default(),
                false,
                mass.as_deref(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut var: BTreeMap<FeatureSet, f64> = BTreeMap::new();
    for bank in &banks {
        for t in &bank.tables {
            *var.entry(t.u.clone()).or_insert(0.0) += t.variance;
        }
        for ft in &bank.factored {
            *var.entry(ft.u.clone()).or_insert(0.0) += ft.variance;
        }
    }
    Ok(var.into_iter().collect())
}

/// PROBE (2026-09-07): the deploy soup's out-of-bag arms for arbitrary support groups —
/// per-row mean out-of-bag logits of the intercept, the full model, and each group, over the
/// rows that have a jury. Wraps [`guard_arms_oob`].
pub fn multiclass_oob_arms(
    mc: &MultiClassModel,
    serve: &ServeBinnedMatrix,
    w_measure: RefMeasure,
    w: &[f32],
    groups: &[Vec<FeatureSet>],
) -> Result<GuardArms, PbError> {
    if !multiclass_bag_oob_evidence_available(mc) {
        return Err(PbError::InvalidInput {
            what: "out-of-bag arms need an in-memory bagged multiclass fit (n_bags >= 2)".into(),
        });
    }
    let mass = measure_mass(&w_measure, w, None);
    guard_arms_oob(mc, serve, w_measure, mass.as_deref(), groups)
}

/// PROBE (2026-09-07): carve arms — the same shape as [`multiclass_oob_arms`] but evaluated on
/// the rows where `mask[i]` is true (a group-honest holdout every bag excluded), using the soup.
pub fn multiclass_carve_arms(
    mc: &MultiClassModel,
    serve: &ServeBinnedMatrix,
    w_measure: RefMeasure,
    w: &[f32],
    groups: &[Vec<FeatureSet>],
    mask: &[bool],
) -> Result<GuardArms, PbError> {
    let mass = measure_mass(&w_measure, w, None);
    guard_arms_carve(mc, serve, w_measure, mass.as_deref(), groups, mask)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
    use super::*;

    /// `measure_mass` hands the prune path `w · exp(offset)` ONLY under a measure that
    /// reads row mass; the legacy product measure keeps `None`, so its banks stay the flat
    /// row-count decomposition earlier boards were pruned on.
    #[test]
    fn measure_mass_is_exposure_weighted_only_under_the_exposure_measure() {
        let w = [1.0_f32, 2.0, 0.5];
        let offset = [0.0_f32, 1.0_f32.ln(), 4.0_f32.ln()];
        assert_eq!(
            measure_mass(&RefMeasure::default(), &w, Some(&offset)),
            None
        );
        assert_eq!(measure_mass(&RefMeasure::Uniform, &w, Some(&offset)), None);
        let m = RefMeasure::ExposureMarginals { floor: 1e-3 };
        let got = measure_mass(&m, &w, Some(&offset)).unwrap();
        for (g, e) in got.iter().zip([1.0_f32, 2.0, 2.0]) {
            assert!((g - e).abs() < 1e-6, "{got:?}");
        }
        assert_eq!(measure_mass(&m, &w, None).unwrap(), w.to_vec());
        assert_eq!(mass_for(&RefMeasure::default(), Some(&w)), None);
        assert_eq!(mass_for(&m, Some(&w)), Some(&w[..]));
    }
    use crate::constraints::{CredibilityFloor, InteractionPolicy, MonotoneMap};
    use crate::data::{AxisKind, AxisProvenance, BinnedMatrix, BorderGrid, FeatureId};
    use crate::engine::{
        Booster, Config, ExactnessMode, FitSpec, ModelSchema, ObliviousTree, Split,
    };
    use crate::explain::{
        fixture_model, fixture_multichannel_model, fixture_multichannel_serve, fixture_serve,
        RefMeasure,
    };
    use crate::loss::{Link, Logistic, LossId, ObjectiveTag, Poisson, SquaredError};

    fn score(u: &[u32], gain: f64) -> PruneTableScore {
        PruneTableScore {
            u: FeatureSet::new(u),
            order: u.len() as u8,
            mean_gain: gain,
            se_gain: 0.0,
            variance: 0.0,
            sticky: false,
            selected: false,
        }
    }

    #[test]
    fn readmission_chunks_are_gain_ranked_and_double() {
        // The ladder is fixed before any evidence is scored: dropped tables in mean_gain order,
        // cut into chunks capped by the GROWN keep-set size (1 kept -> 1, 2, 4, ...). This is
        // what lets the guard price every rung from one pass over the banks.
        let keep: Vec<FeatureSet> = vec![score(&[0], 0.0).u];
        let scores: Vec<PruneTableScore> = vec![
            score(&[0], 9.0), // already kept -> never a candidate
            score(&[1], 0.1),
            score(&[2], 0.5),
            score(&[3], 0.3),
            score(&[4], 0.4),
            score(&[5], 0.2),
            score(&[6], 0.6),
            score(&[7], 0.05),
        ];
        let chunks = guard_readmission_chunks(&keep, &scores);
        let ids: Vec<Vec<Vec<u32>>> = chunks
            .iter()
            .map(|c| {
                c.iter()
                    .map(|u| u.0.iter().map(|f| f.0).collect())
                    .collect()
            })
            .collect();
        // 7 candidates, |keep| = 1 -> chunk sizes 1, 2, 4 (the last is truncated by what is left).
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0].len(), 1);
        assert_eq!(ids[1].len(), 2);
        assert_eq!(ids[2].len(), 4);
        // Strictly gain-descending across the flattened ladder.
        let flat: Vec<u32> = ids.iter().flatten().map(|u| u[0]).collect();
        assert_eq!(flat, vec![6, 2, 4, 3, 5, 1, 7]);
    }

    #[test]
    fn readmission_chunks_are_empty_when_nothing_was_dropped() {
        let keep: Vec<FeatureSet> = vec![score(&[0], 0.0).u, score(&[1], 0.0).u];
        let scores = vec![score(&[0], 1.0), score(&[1], 2.0)];
        assert!(guard_readmission_chunks(&keep, &scores).is_empty());
    }

    #[test]
    fn intercept_shifts_reproduce_the_observed_class_mass() {
        // The re-anchor's defining property: after the shifts, the weighted predicted class mass
        // equals the observed one. 10 fixed IPF rounds, no convergence test — this asserts the
        // budget is actually enough on a badly-mis-levelled start. MEASURED convergence on this
        // deliberately hostile fixture (one class pinned 4 logits above the others, and a
        // 5-level weight pattern): ~1e-7 relative, not machine precision. That is the honest
        // headroom of the fixed budget, and it is three orders below anything a class balance
        // is read to.
        let n = 400;
        let labels: Vec<u32> = (0..n).map(|i| (i % 3) as u32).collect();
        let w: Vec<f32> = (0..n).map(|i| 1.0 + (i % 5) as f32).collect();
        // Class 2's logit is absurdly high: nothing about the start matches the observed mass.
        let raw: Vec<Vec<f64>> = vec![
            (0..n).map(|i| 0.01 * f64::from(i as u32)).collect(),
            vec![0.0; n as usize],
            vec![4.0; n as usize],
        ];
        let delta = multiclass_intercept_shifts(&raw, &labels, &w, "test").unwrap();
        let mut observed = [0.0_f64; 3];
        for (i, &l) in labels.iter().enumerate() {
            observed[l as usize] += f64::from(w[i]);
        }
        let mut predicted = [0.0_f64; 3];
        for i in 0..n as usize {
            let z: Vec<f64> = (0..3).map(|k| raw[k][i] + delta[k]).collect();
            let mx = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = z.iter().map(|v| (v - mx).exp()).collect();
            let denom: f64 = e.iter().sum();
            for k in 0..3 {
                predicted[k] += f64::from(w[i]) * e[k] / denom;
            }
        }
        for k in 0..3 {
            assert!(
                (predicted[k] - observed[k]).abs() / observed[k] < 1e-6,
                "class {k}: predicted {} vs observed {}",
                predicted[k],
                observed[k]
            );
        }
    }

    #[test]
    fn intercept_shifts_are_zero_when_a_class_is_unobserved() {
        // ln(0) guard: a class with no observed mass would make the IPF step -inf, so the whole
        // correction is declined rather than poisoning every class's intercept.
        let labels = vec![0_u32, 0, 1, 1];
        let w = vec![1.0_f32; 4];
        let raw = vec![vec![0.0; 4], vec![0.5; 4], vec![0.2; 4]];
        let delta = multiclass_intercept_shifts(&raw, &labels, &w, "test").unwrap();
        assert_eq!(delta, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn intercept_shifts_reject_an_out_of_range_label() {
        let raw = vec![vec![0.0; 2], vec![0.0; 2]];
        assert!(multiclass_intercept_shifts(&raw, &[0, 5], &[1.0, 1.0], "test").is_err());
    }

    #[test]
    fn lazy_walk_gate_auto_threshold() {
        // Strictly below the threshold stays exhaustive; at/above engages lazy.
        assert!(!lazy_walk_engaged(0));
        assert!(!lazy_walk_engaged(LAZY_WALK_TABLE_THRESHOLD - 1));
        assert!(lazy_walk_engaged(LAZY_WALK_TABLE_THRESHOLD));
        assert!(lazy_walk_engaged(5_246));
    }

    /// A hand-built 2-bag model whose bags realize DIFFERENT split borders on the same
    /// axis: bag 0's standalone tree splits at `bin_le=1` (only that border realized), bag
    /// 1's at `bin_le=2` — the exact case `bag_banks_for_keepset`'s fix targets, where
    /// `MergedGrids::from_model` on each bag ALONE would previously have derived a
    /// different (coarser) grid than the soup's. Trees are stored at `alpha =
    /// standalone/n_bags`, undone by `bag_banks_for_keepset`'s own `* n_bags` rescale.
    fn differing_borders_bagged_fixture() -> (Model, ServeBinnedMatrix) {
        let n_bags = 2.0_f32;
        let grid = BorderGrid {
            borders: vec![1.5, 2.5],
            n_bins: 4,
            missing_bin: 0,
        };
        // Leaf index = bit0 = (bin <= bin_le): leaves[0] = HIGH (bin > bin_le), leaves[1] =
        // LOW (bin <= bin_le).
        let bag0_tree = ObliviousTree {
            splits: vec![Split {
                axis: 0,
                bin_le: 1,
                missing_left: false,
            }],
            leaves: vec![10.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            depth: 1,
        };
        let bag1_tree = ObliviousTree {
            splits: vec![Split {
                axis: 0,
                bin_le: 2,
                missing_left: false,
            }],
            leaves: vec![20.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            depth: 1,
        };
        let schema = ModelSchema {
            feature_names: vec!["x0".into()],
            feature_kinds: vec![AxisKind::Numeric],
            cat_encoders: crate::cat::CatEncoderStore::new(),
            class_labels: None,
            objective: ObjectiveTag {
                link: Link::Identity,
                loss: LossId::SquaredError,
                tweedie_rho: None,
            },
        };
        let model = Model {
            f0: 0.0,
            trees: vec![(1.0 / n_bags, bag0_tree), (1.0 / n_bags, bag1_tree)],
            grids: vec![grid.clone()],
            provenance: vec![AxisProvenance {
                raw: FeatureId(0),
                kind: AxisKind::Numeric,
            }],
            link: Link::Identity,
            mode: ExactnessMode::Exact,
            schema,
            schema_version: crate::serialize::SCHEMA_VERSION_UNLIFTED,
            correction: None,
            bag_spans: Some(vec![(0, 1), (1, 2)]),
            bag_intercepts: None,
            bag_in_bag: None,
            delta_step_gate: None,
        };
        let serve = ServeBinnedMatrix(BinnedMatrix {
            data: vec![vec![1, 2, 3]],
            n_rows: 3,
            grids: vec![grid],
            provenance: model.provenance.clone(),
        });
        (model, serve)
    }

    #[test]
    fn bag_banks_linearity_reproduces_the_soup_bank_with_differing_realized_borders() {
        let (model, serve) = differing_borders_bagged_fixture();
        let n_bags = model.bag_spans.as_ref().unwrap().len() as f64;

        let soup_bank = model.explain(&serve, RefMeasure::Uniform).unwrap();
        let keep: Vec<FeatureSet> = soup_bank.tables.iter().map(|t| t.u.clone()).collect();
        assert!(!keep.is_empty(), "fixture must realize at least one table");
        let bags = bag_banks_for_keepset(&model, &serve, RefMeasure::Uniform, None, &keep).unwrap();
        assert_eq!(bags.len(), 2);

        // The n_bags-weighted mean of the per-bag banks must reproduce the soup bank's
        // tables to fp tolerance, table-for-table and cell-for-cell — purify's linearity
        // in trees at a fixed weight/grid, which only holds once every bag purifies on the
        // SAME (soup) grid instead of its own bag-specific one.
        for soup_table in &soup_bank.tables {
            let n_cells = soup_table.values.values().len();
            let mut mean_values = vec![0.0_f64; n_cells];
            for bag_bank in &bags {
                if let Some(bag_table) = bag_bank.tables.iter().find(|t| t.u == soup_table.u) {
                    for (m, &v) in mean_values.iter_mut().zip(bag_table.values.values().iter()) {
                        *m += v / n_bags;
                    }
                }
                // A bag whose trees realize none of this support omits the table entirely —
                // an all-zero replicate, contributing nothing to the mean (no-op here).
            }
            for (i, (&mean, &soup)) in mean_values
                .iter()
                .zip(soup_table.values.values().iter())
                .enumerate()
            {
                assert!(
                    (mean - soup).abs() < 1e-9,
                    "table {:?} cell {i}: bag-weighted mean {mean} != soup {soup}",
                    soup_table.u
                );
            }
        }
        let f0_mean: f64 = bags.iter().map(|b| b.f0 / n_bags).sum();
        assert!(
            (f0_mean - soup_bank.f0).abs() < 1e-9,
            "f0 mean {f0_mean} != soup f0 {}",
            soup_bank.f0
        );
    }

    #[test]
    fn bag_raw_scores_restrict_to_the_keepset_and_mean_to_the_soup() {
        // The per-bag raw scorer the prune guard runs on: `keep=None` is that bag's full bank
        // (lossless vs its trees), a keep-set restriction drops exactly the excluded tables,
        // and the mean over ALL bags reproduces the soup's own bank score row for row.
        let (model, serve) = differing_borders_bagged_fixture();
        let rows: Vec<u32> = (0..serve.0.n_rows).collect();
        let soup_bank = model.explain(&serve, RefMeasure::Uniform).unwrap();
        let mut soup_raw = vec![0.0_f64; rows.len()];
        crate::scoring::score_bank_binned(
            &soup_bank,
            &model.schema.cat_encoders,
            &serve.0,
            &mut soup_raw,
        )
        .unwrap();

        let full = bag_raw_scores_for_rows(&model, &serve, RefMeasure::Uniform, None, None, &rows)
            .unwrap();
        assert_eq!(full.len(), 2, "one score vector per bag, in bag order");
        for (r, &soup) in soup_raw.iter().enumerate() {
            let mean = full.iter().map(|b| b[r] / 2.0).sum::<f64>();
            assert!(
                (mean - soup).abs() < 1e-9,
                "row {r}: mean-of-bags {mean} != soup bank {soup}"
            );
        }
        // Empty keep-set ⇒ intercept only, so every row of a bag scores the same value and the
        // difference from the full bank is exactly that bag's table sum.
        let empty =
            bag_raw_scores_for_rows(&model, &serve, RefMeasure::Uniform, None, Some(&[]), &rows)
                .unwrap();
        for (bag, scores) in empty.iter().enumerate() {
            assert!(scores.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12));
            assert!(
                (full[bag][0] - scores[0]).abs() > 1e-9,
                "the fixture's tables must actually move the score"
            );
        }
        // A row subset scores the same values as the corresponding full-row slots.
        let subset =
            bag_raw_scores_for_rows(&model, &serve, RefMeasure::Uniform, None, None, &[2, 0])
                .unwrap();
        for bag in 0..2 {
            assert_eq!(subset[bag][0], full[bag][2]);
            assert_eq!(subset[bag][1], full[bag][0]);
        }
        // A model without the runtime bag partition cannot supply per-bag scores.
        let mut single = model.clone();
        single.bag_spans = None;
        assert!(
            bag_raw_scores_for_rows(&single, &serve, RefMeasure::Uniform, None, None, &rows)
                .is_err()
        );
    }

    #[test]
    fn bag_oob_evidence_shares_a_cell_correction_across_the_bags() {
        // A §G1 cell correction is a SOUP-level object with no bag attribution, so the
        // replicate APIs refuse it. The evidence path instead gives every bag the WHOLE
        // correction, which is the arithmetically right thing for the only quantity it uses:
        // purify is linear, so the mean over ALL bags of those banks is still the soup's own
        // bank — including the correction, exactly once. Without this the guard would vanish
        // on any `cell_refit_base` fit now that the ungrouped carve is gone.
        let (mut model, serve) = differing_borders_bagged_fixture();
        model.correction = Some(crate::engine::CorrectionBank {
            tables: vec![crate::engine::CorrectionTable {
                axes: vec![0],
                shape: vec![4],
                bin_to_cell: vec![vec![0, 1, 2, 3]],
                values: vec![0.5, -0.25, 0.75, -1.0],
            }],
        });
        model.bag_in_bag = Some(vec![vec![true, true, false], vec![false, true, true]]);
        assert!(
            bag_oob_evidence_available(&model),
            "a corrected model still has out-of-bag evidence"
        );
        assert!(
            bag_raw_scores_for_rows(&model, &serve, RefMeasure::Uniform, None, None, &[0]).is_err(),
            "the REPLICATE api must still refuse a correction it cannot attribute"
        );

        let soup_bank = model.explain(&serve, RefMeasure::Uniform).unwrap();
        let all: Vec<FeatureSet> = soup_bank.tables.iter().map(|t| t.u.clone()).collect();
        let mut soup_raw = vec![0.0_f64; 3];
        crate::scoring::score_bank_binned(
            &soup_bank,
            &model.schema.cat_encoders,
            &serve.0,
            &mut soup_raw,
        )
        .unwrap();

        // Give every row a jury of BOTH bags, so the jury mean IS the all-bags mean and must
        // reproduce the soup bank row for row.
        let mut all_oob = model.clone();
        all_oob.bag_in_bag = Some(vec![vec![false; 3], vec![false; 3]]);
        let sums =
            bag_oob_group_raw_sums(&all_oob, &serve, RefMeasure::Uniform, None, &[all]).unwrap();
        for r in 0..3 {
            assert_eq!(sums.counts[r], 2);
            let mean = sums.full_sum[r] / 2.0;
            assert!(
                (mean - soup_raw[r]).abs() < 1e-9,
                "row {r}: jury mean {mean} != corrected soup bank {}",
                soup_raw[r]
            );
        }
    }

    #[test]
    fn bag_oob_group_sums_are_additive_over_the_jury_and_thread_independent() {
        // The guard's one-pass evidence: only the bags that left a row OUT of bag may
        // contribute to it, the passed table GROUPS must sum (with the intercept) back to the
        // full bank score, and none of it may depend on the thread count.
        let (mut model, serve) = differing_borders_bagged_fixture();
        // bag 0 trained on rows {0,1}, bag 1 on rows {1,2}: row 0's jury is {bag 1}, row 2's
        // is {bag 0}, and row 1 has no jury at all (both bags saw it).
        model.bag_in_bag = Some(vec![vec![true, true, false], vec![false, true, true]]);
        let soup_bank = model.explain(&serve, RefMeasure::Uniform).unwrap();
        let all: Vec<FeatureSet> = soup_bank.tables.iter().map(|t| t.u.clone()).collect();
        assert!(!all.is_empty(), "fixture must realize at least one table");
        // Split the supports into two groups to exercise prefix reconstruction.
        let (head, tail) = all.split_at(1);
        let groups = vec![head.to_vec(), tail.to_vec()];

        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    bag_oob_group_raw_sums(&model, &serve, RefMeasure::Uniform, None, &groups)
                        .unwrap()
                })
        };
        let sums = run(1);
        assert_eq!(sums.counts, vec![1, 0, 1], "jury size per row");
        assert_eq!(sums.group_sums.len(), 2);
        // A row with no jury carries nothing at all.
        assert_eq!(sums.full_sum[1], 0.0);
        assert_eq!(sums.f0_sum[1], 0.0);
        // Full == intercept + every group, row by row (this is the prefix-union identity the
        // re-admission ladder rides on).
        for r in [0usize, 2] {
            let rebuilt = sums.f0_sum[r] + sums.group_sums.iter().map(|g| g[r]).sum::<f64>();
            assert!(
                (rebuilt - sums.full_sum[r]).abs() < 1e-9,
                "row {r}: groups + f0 = {rebuilt} != full {}",
                sums.full_sum[r]
            );
        }
        // The jury really is the OOB complement: row 0's value must be bag 1's own score.
        let per_bag =
            bag_raw_scores_for_rows(&model, &serve, RefMeasure::Uniform, None, None, &[0, 1, 2])
                .unwrap();
        assert!((sums.full_sum[0] - per_bag[1][0]).abs() < 1e-12);
        assert!((sums.full_sum[2] - per_bag[0][2]).abs() < 1e-12);
        assert_eq!(sums, run(4), "thread count cannot move the evidence");

        // Membership is required, and must cover the design being scored.
        let mut no_membership = model.clone();
        no_membership.bag_in_bag = None;
        assert!(
            bag_oob_group_raw_sums(&no_membership, &serve, RefMeasure::Uniform, None, &groups)
                .is_err()
        );
        let mut wrong_rows = model.clone();
        wrong_rows.bag_in_bag = Some(vec![vec![true, true], vec![false, true]]);
        assert!(
            bag_oob_group_raw_sums(&wrong_rows, &serve, RefMeasure::Uniform, None, &groups)
                .is_err(),
            "evidence must be refused on anything but the fit design"
        );
    }

    // ---------------------------------------------------------------------------------
    // DEPLOYED-BOX BUDGET
    // ---------------------------------------------------------------------------------

    fn fs(ids: &[u32]) -> FeatureSet {
        FeatureSet::new(ids)
    }

    /// keep, costs, evidence for the budget tests: three factored triples of very different
    /// size and evidence, plus two free (dense) supports that must never be touched.
    fn budget_fixture() -> (
        Vec<FeatureSet>,
        BTreeMap<FeatureSet, usize>,
        BTreeMap<FeatureSet, f64>,
    ) {
        let keep = vec![
            fs(&[0]),
            fs(&[0, 1]),
            fs(&[0, 1, 2]),
            fs(&[0, 1, 3]),
            fs(&[1, 2, 3]),
        ];
        let costs: BTreeMap<FeatureSet, usize> = [
            (fs(&[0, 1, 2]), 100_usize), // rich but ruinous: 0.001 gain per box
            (fs(&[0, 1, 3]), 10),        // the bargain: 0.005 per box
            (fs(&[1, 2, 3]), 40),        // 0.0005 per box
        ]
        .into_iter()
        .collect();
        let evidence: BTreeMap<FeatureSet, f64> = [
            (fs(&[0, 1, 2]), 0.1_f64),
            (fs(&[0, 1, 3]), 0.05),
            (fs(&[1, 2, 3]), 0.02),
        ]
        .into_iter()
        .collect();
        (keep, costs, evidence)
    }

    /// THE NO-OP GUARANTEE, which is what makes "budget off is byte-identical" checkable: a
    /// disabled budget and a budget the bank already satisfies both return the keep-set
    /// VERBATIM — same order, same members — so the caller's `retain_tables` sees exactly the
    /// bytes it would have seen.
    #[test]
    fn box_budget_is_a_verbatim_no_op_when_disabled_or_satisfied() {
        let (keep, costs, evidence) = budget_fixture();
        let vars: BTreeMap<FeatureSet, f64> = BTreeMap::new();
        for max_boxes in [0_usize, 150, 151, usize::MAX] {
            let (out, rep) = apply_box_budget(&keep, &costs, &vars, max_boxes, &evidence);
            assert_eq!(
                out, keep,
                "max_boxes={max_boxes} must not move the keep-set"
            );
            assert!(!rep.engaged, "max_boxes={max_boxes}");
            assert_eq!(rep.boxes_before, 150);
            assert_eq!(rep.boxes_after, 150);
            assert!(rep.dropped.is_empty() && rep.cascade_dropped.is_empty());
        }
        // 149 is the first budget the bank does NOT satisfy.
        let (_, rep) = apply_box_budget(&keep, &costs, &vars, 149, &evidence);
        assert!(rep.engaged);
    }

    /// The budget is spent on EVIDENCE PER BOX, not on evidence: the highest-gain effect is
    /// the first one dropped when it is also the most expensive. A budget of 50 buys the
    /// 10-box bargain (0.005/box) and the 40-box middle (0.0005/box) and refuses the 100-box
    /// headline (0.001/box) — total gain 0.07 against the 0.1 a gain-ranked spend would have
    /// bought with the whole budget on one effect.
    #[test]
    fn box_budget_spends_on_evidence_per_box_not_on_evidence() {
        let (keep, costs, evidence) = budget_fixture();
        let vars: BTreeMap<FeatureSet, f64> = BTreeMap::new();
        let (out, rep) = apply_box_budget(&keep, &costs, &vars, 50, &evidence);
        assert!(rep.engaged);
        assert_eq!(rep.boxes_after, 50);
        assert!(out.contains(&fs(&[0, 1, 3])) && out.contains(&fs(&[1, 2, 3])));
        assert!(!out.contains(&fs(&[0, 1, 2])));
        assert_eq!(rep.dropped, vec![fs(&[0, 1, 2])]);
        // Free supports are never a budget's business.
        assert!(out.contains(&fs(&[0])) && out.contains(&fs(&[0, 1])));
        assert_eq!(rep.effects_before, 3);
        assert_eq!(rep.effects_after, 2);
    }

    /// The greedy keeps SCANNING past an unaffordable support — a cheap, well-evidenced effect
    /// behind an expensive one still gets its room. At 45 the 100-box and 40-box effects are
    /// both refused but the 10-box bargain is not.
    #[test]
    fn box_budget_fills_behind_an_unaffordable_effect() {
        let (keep, costs, evidence) = budget_fixture();
        let vars: BTreeMap<FeatureSet, f64> = BTreeMap::new();
        let (out, rep) = apply_box_budget(&keep, &costs, &vars, 45, &evidence);
        assert_eq!(rep.boxes_after, 10);
        assert!(out.contains(&fs(&[0, 1, 3])));
        assert_eq!(rep.dropped, vec![fs(&[0, 1, 2]), fs(&[1, 2, 3])]);
    }

    /// Heredity survives the budget: nothing may deploy that properly contains a support the
    /// budget removed, so dropping the order-3 face takes the order-4 effect with it — even
    /// though the 4-way itself was affordable.
    #[test]
    fn box_budget_cascades_to_supersets_of_what_it_dropped() {
        let mut keep = vec![fs(&[0]), fs(&[0, 1]), fs(&[0, 1, 2]), fs(&[0, 1, 2, 3])];
        keep.sort();
        let costs: BTreeMap<FeatureSet, usize> =
            [(fs(&[0, 1, 2]), 100_usize), (fs(&[0, 1, 2, 3]), 5)]
                .into_iter()
                .collect();
        let evidence: BTreeMap<FeatureSet, f64> =
            [(fs(&[0, 1, 2]), 0.001_f64), (fs(&[0, 1, 2, 3]), 1.0)]
                .into_iter()
                .collect();
        let vars: BTreeMap<FeatureSet, f64> = BTreeMap::new();
        let (out, rep) = apply_box_budget(&keep, &costs, &vars, 50, &evidence);
        assert_eq!(rep.dropped, vec![fs(&[0, 1, 2])]);
        assert_eq!(rep.cascade_dropped, vec![fs(&[0, 1, 2, 3])]);
        assert_eq!(rep.boxes_after, 0);
        assert_eq!(out, vec![fs(&[0]), fs(&[0, 1])]);
    }

    /// Ties are broken by the support's raw ids, never by map/iteration order, so the admitted
    /// prefix is a pure function of the inputs. Two effects of identical density, one box of
    /// room short: the lower ids win, whichever order the caller hands them over in.
    #[test]
    fn box_budget_tie_break_is_the_feature_ids() {
        let costs: BTreeMap<FeatureSet, usize> = [(fs(&[0, 1, 2]), 10_usize), (fs(&[3, 4, 5]), 10)]
            .into_iter()
            .collect();
        let evidence: BTreeMap<FeatureSet, f64> =
            [(fs(&[0, 1, 2]), 0.5_f64), (fs(&[3, 4, 5]), 0.5)]
                .into_iter()
                .collect();
        let vars: BTreeMap<FeatureSet, f64> = BTreeMap::new();
        for keep in [
            vec![fs(&[0, 1, 2]), fs(&[3, 4, 5])],
            vec![fs(&[3, 4, 5]), fs(&[0, 1, 2])],
        ] {
            let (out, _) = apply_box_budget(&keep, &costs, &vars, 10, &evidence);
            assert_eq!(out, vec![fs(&[0, 1, 2])]);
        }
    }

    /// The SECONDARY key: where the held-out evidence cannot separate two supports — the
    /// common case on a wide bank, where most candidates are scored by no fold at all and so
    /// all carry drop-gain 0.0 — the budget spends on purified MASS per box, not on whichever
    /// feature happens to have the lower id. Same three unevidenced effects, three different
    /// variances, one effect's worth of room: the heaviest wins whatever order it arrives in.
    #[test]
    fn box_budget_breaks_an_evidence_tie_on_variance_per_box() {
        let keep = vec![fs(&[0, 1, 2]), fs(&[0, 1, 3]), fs(&[0, 2, 3])];
        let costs: BTreeMap<FeatureSet, usize> = keep.iter().map(|u| (u.clone(), 10)).collect();
        let vars: BTreeMap<FeatureSet, f64> = [
            (fs(&[0, 1, 2]), 0.1_f64),
            (fs(&[0, 1, 3]), 0.2),
            (fs(&[0, 2, 3]), 9.0),
        ]
        .into_iter()
        .collect();
        let evidence: BTreeMap<FeatureSet, f64> = BTreeMap::new();
        for order in [
            vec![fs(&[0, 1, 2]), fs(&[0, 1, 3]), fs(&[0, 2, 3])],
            vec![fs(&[0, 2, 3]), fs(&[0, 1, 3]), fs(&[0, 1, 2])],
        ] {
            let (out, _) = apply_box_budget(&order, &costs, &vars, 10, &evidence);
            assert_eq!(out, vec![fs(&[0, 2, 3])]);
        }
        // But it is only a TIE-break: real held-out evidence still outranks a heavier effect.
        let evidence: BTreeMap<FeatureSet, f64> =
            [(fs(&[0, 1, 2]), 1e-9_f64)].into_iter().collect();
        let (out, _) = apply_box_budget(&keep, &costs, &vars, 10, &evidence);
        assert_eq!(out, vec![fs(&[0, 1, 2])]);
    }

    /// An unevidenced support scores 0.0 — below every positively-evidenced one and above
    /// every harmful one. That ordering is the whole reason the budget can be pointed at a
    /// bank like allstate_sev's, where most candidates are scored by no fold at all.
    #[test]
    fn box_budget_ranks_unevidenced_between_useful_and_harmful() {
        let keep = vec![fs(&[0, 1, 2]), fs(&[0, 1, 3]), fs(&[0, 2, 3])];
        let costs: BTreeMap<FeatureSet, usize> = keep.iter().map(|u| (u.clone(), 10)).collect();
        let evidence: BTreeMap<FeatureSet, f64> =
            [(fs(&[0, 1, 2]), 0.5_f64), (fs(&[0, 2, 3]), -0.5)]
                .into_iter()
                .collect();
        let vars: BTreeMap<FeatureSet, f64> = BTreeMap::new();
        let (out, _) = apply_box_budget(&keep, &costs, &vars, 20, &evidence);
        assert_eq!(out, vec![fs(&[0, 1, 2]), fs(&[0, 1, 3])]);
    }

    // Held-out y == the full model's own predictions ⇒ the full model has zero deviance and any
    // drop strictly worsens it ⇒ the selector must keep every table.
    #[test]
    fn keeps_all_tables_when_every_table_earns_its_keep() {
        let model = fixture_model();
        let x = fixture_serve();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let y = model.predict(&x.0, None).unwrap();
        let w = vec![1.0_f32; y.len()];
        let folds = [HoldoutFold {
            x: &x.0,
            y: &y,
            w: &w,
            offset: None,
        }];

        let (pruned, report) = prune_bank(
            &bank,
            &crate::cat::CatEncoderStore::new(),
            &folds,
            &SquaredError,
            &PruneConfig::default(),
        )
        .unwrap();
        assert_eq!(report.dropped.len(), 0, "no table should be dropped");
        assert_eq!(pruned.tables.len(), bank.tables.len());
        assert_eq!(report.effective_order, 2, "the genuine pair is retained");
        assert!(report.delta_vs_full.abs() < 1e-9);
    }

    // Regression for se_gain: the OLD formula (sqrt(drop_s^2 + s0^2)) treats the drop-model
    // and full-model held-out deviances as independent, even though every fold's drop/full
    // pair differs only by one table's per-row contribution — so it grossly overstates
    // uncertainty. Shifting y by a per-fold additive constant `c_f` (baseline residual is
    // exactly zero: y := the model's own predictions) moves each fold's marginal deviance by
    // `c_f^2 + 2*c_f*mean(contrib)`, driving path[0].se up a lot; the drop-minus-full
    // difference per fold is `mean(contrib^2) + 2*c_f*mean(contrib)`, which shares only the
    // SAME small linear-in-`c_f` term (`mean(contrib)` is small but not exactly zero here,
    // since the merged grid's unsampled missing cell also carries part of the purified
    // table's zero-sum constraint) — so old-vs-new should differ by roughly an order of
    // magnitude: se_gain well below s0, never anywhere close to sqrt(drop_s^2 + s0^2) >= s0.
    #[test]
    fn se_gain_uses_the_paired_fold_difference_not_the_naive_quadrature_sum() {
        let model = fixture_model();
        let x = fixture_serve();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let y_base = model.predict(&x.0, None).unwrap();
        let w = vec![1.0_f32; y_base.len()];

        let shifts = [0.0_f32, 5.0, -3.0, 10.0];
        let ys: Vec<Vec<f32>> = shifts
            .iter()
            .map(|&c| y_base.iter().map(|&v| v + c).collect())
            .collect();
        let folds: Vec<HoldoutFold> = ys
            .iter()
            .map(|y| HoldoutFold {
                x: &x.0,
                y,
                w: &w,
                offset: None,
            })
            .collect();

        let (_, report) = prune_bank(
            &bank,
            &crate::cat::CatEncoderStore::new(),
            &folds,
            &SquaredError,
            &PruneConfig::default(),
        )
        .unwrap();
        let s0 = report.path[0].se;
        assert!(
            s0 > 1.0,
            "fixture must realize substantial marginal fold variance, got {s0}"
        );
        for score in &report.table_scores {
            assert!(
                score.se_gain < s0 * 0.5,
                "table {:?}: se_gain {} should be well below the marginal fold SE s0 = {s0} — \
                 the OLD quadrature-sum formula sqrt(drop_s^2 + s0^2) is always >= s0, so it \
                 could never have produced a value this small",
                score.u,
                score.se_gain
            );
        }
    }

    // Post-prune re-solve exactness: with y := the full model's own predictions and an explicit
    // full keep-set, the kept-model working residual is exactly zero ⇒ the §G1 re-solve
    // correction is ~0 ⇒ `rebalance=true` must reproduce the un-rebalanced base bit-for-bit
    // (up to fp noise). Exercises the whole path: grad_hess → fit_cell_correction → fold + purify
    // → retain → held-out guard.
    #[test]
    fn rebalance_is_a_noop_when_the_residual_is_zero() {
        let model = fixture_model();
        let x = fixture_serve();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let keep: Vec<FeatureSet> = bank.tables.iter().map(|t| t.u.clone()).collect();
        let y = model.predict(&x.0, None).unwrap();
        let w = vec![1.0_f32; y.len()];
        let loss = SquaredError;
        let base = prune_model_to_keepset(
            &model,
            &x,
            &y,
            &w,
            None,
            &loss,
            RefMeasure::Uniform,
            false,
            &keep,
            false,
        )
        .unwrap();
        let rebal = prune_model_to_keepset(
            &model,
            &x,
            &y,
            &w,
            None,
            &loss,
            RefMeasure::Uniform,
            false,
            &keep,
            true,
        )
        .unwrap();
        base.validate().unwrap();
        rebal.validate().unwrap();
        assert_eq!(
            keep.len(),
            base.bank.tables.len(),
            "the explicit keep-set keeps every table"
        );
        let n = x.0.n_rows as usize;
        let (mut bp, mut rp) = (vec![0.0_f64; n], vec![0.0_f64; n]);
        let empty_encoders = crate::cat::CatEncoderStore::new();
        crate::scoring::score_bank_binned(&base.bank, &empty_encoders, &x.0, &mut bp).unwrap();
        crate::scoring::score_bank_binned(&rebal.bank, &empty_encoders, &x.0, &mut rp).unwrap();
        for (a, b) in bp.iter().zip(&rp) {
            assert!(
                (a - b).abs() < 1e-4,
                "rebalance moved a zero-residual prediction: {a} vs {b}"
            );
        }
    }

    /// Fit a tiny model (6 trees, no fit-time reanchor) on the crate's canonical 4-row fixture
    /// matrix ([`fixture_serve`]) for a given loss — shared setup for the Log/Logit prune-reanchor
    /// balance tests below.
    fn fit_tiny_fixture_model(loss: &dyn Loss, y: &[f32]) -> Model {
        let x = fixture_serve().0;
        let cfg = Config {
            n_trees: 6,
            learning_rate: 0.3,
            ..Config::default()
        };
        let spec = FitSpec {
            loss,
            weight: None,
            exposure: None,
            monotone: MonotoneMap::default(),
            interaction: InteractionPolicy::default(),
            credibility: CredibilityFloor::default(),
            fixed_holdout: None,
            bag_groups: None,
            seed: 0,
        };
        Booster::with_config(cfg).fit(&x, y, &spec).unwrap()
    }

    /// Keep every realized table (no selection) and reanchor: the pruned model's response-scale,
    /// exposure-weighted mean prediction must match the observed weighted mean of `y` to tight
    /// tolerance — the balance guarantee `prune_model_to_tables`'s doc comment promises for both
    /// links (`Σwŷ == Σwy`), independent of how well the underlying trees fit.
    fn assert_prune_reanchor_balances(model: &Model, y: &[f32], loss: &dyn Loss) {
        let serve = fixture_serve();
        let bank = model
            .explain_bank(
                &serve,
                RefMeasure::Uniform,
                crate::explain::TableBudget::default(),
                false,
            )
            .unwrap();
        let keep: Vec<FeatureSet> = bank.tables.iter().map(|t| t.u.clone()).collect();
        let w = vec![1.0_f32; y.len()];
        let tm = prune_model_to_keepset(
            model,
            &serve,
            y,
            &w,
            None,
            loss,
            RefMeasure::Uniform,
            true, // reanchor
            &keep,
            false, // rebalance
        )
        .unwrap();
        let preds = tm.predict_binned(&serve.0, None).unwrap();
        let sum_w: f64 = w.iter().map(|&wi| f64::from(wi)).sum();
        let observed: f64 = y
            .iter()
            .zip(&w)
            .map(|(&yi, &wi)| f64::from(yi) * f64::from(wi))
            .sum();
        let predicted: f64 = preds
            .iter()
            .zip(&w)
            .map(|(&pi, &wi)| f64::from(pi) * f64::from(wi))
            .sum();
        assert!(
            (observed / sum_w - predicted / sum_w).abs() < 1e-4,
            "pruned+reanchored model should balance to the observed weighted mean: \
             observed={}, predicted={}",
            observed / sum_w,
            predicted / sum_w,
        );
    }

    // The new Logit port: `reanchor_shift`'s closed form is log-link-only, so
    // `prune_model_to_keepset` must route a logit-link loss through `boost::reanchor_delta`'s
    // bisection instead. Binary y with both classes present keeps the observed total strictly
    // inside (0, total_weight), satisfying `reanchor_delta`'s Logit precondition.
    #[test]
    fn prune_model_to_keepset_reanchors_logit_link_to_the_observed_rate() {
        let y = vec![1.0_f32, 1.0, 0.0, 0.0];
        let model = fit_tiny_fixture_model(&Logistic, &y);
        assert_eq!(
            model.link,
            Link::Logit,
            "a Logistic fit must deploy on the logit link"
        );
        assert_prune_reanchor_balances(&model, &y, &Logistic);
    }

    // Log-link regression check: widening the reanchor gate from `== Link::Log` to
    // `Log | Logit` must not disturb the pre-existing log-link path, which still routes through
    // the untouched `reanchor_shift`/`reanchor_log_link` closed form.
    #[test]
    fn prune_model_to_keepset_reanchors_log_link_to_the_observed_mean() {
        let y = vec![3.0_f32, 0.0, 1.0, 2.0];
        let model = fit_tiny_fixture_model(&Poisson, &y);
        assert_eq!(
            model.link,
            Link::Log,
            "a Poisson fit must deploy on the log link"
        );
        assert_prune_reanchor_balances(&model, &y, &Poisson);
    }

    /// P1 multi-channel hard gate (team-lead spec): `prune_model_to_keepset` — the SAME code
    /// path production `prune=True` triggers — on a 2-channel model, keeping every realized
    /// table (no actual dropping), must predict IDENTICALLY to the tree ensemble. This proves
    /// `scoring.rs`'s joint-aware `build_cell_maps`/`fill_row_cells`/`score_bank_binned` path
    /// reproduces the model exactly for a multi-channel categorical raw feature when reached
    /// through the FULL explain -> retain -> serve pipeline, not just `TableModel::from_model`
    /// directly (already covered by `table_model::tests::
    /// tables_model_validates_a_multichannel_bank`).
    #[test]
    fn prune_model_to_keepset_reproduces_a_multichannel_model_exactly() {
        let model = fixture_multichannel_model();
        let serve = fixture_multichannel_serve();
        let bank = model
            .explain_bank(
                &serve,
                RefMeasure::Uniform,
                crate::explain::TableBudget::default(),
                false,
            )
            .unwrap();
        let keep: Vec<FeatureSet> = bank.tables.iter().map(|t| t.u.clone()).collect();
        let n = serve.0.n_rows as usize;
        // Identity link (SquaredError): `prune_model_to_keepset`'s `do_reanchor` gate requires
        // Log/Logit, so `reanchor=false` below is already a no-op regardless of y/w — kept
        // simple rather than meaningful.
        let y = vec![0.0_f32; n];
        let w = vec![1.0_f32; n];
        let tm = prune_model_to_keepset(
            &model,
            &serve,
            &y,
            &w,
            None,
            &SquaredError,
            RefMeasure::Uniform,
            false, // reanchor
            &keep,
            false, // rebalance
        )
        .unwrap();

        let ens = model.predict(&serve.0, None).unwrap();
        let tab = tm.predict_binned(&serve.0, None).unwrap();
        assert_eq!(ens.len(), 4);
        for (e, t) in ens.iter().zip(&tab) {
            assert!((e - t).abs() < 1e-6, "ensemble {e} vs pruned-tables {t}");
        }
    }

    /// Bug #5's fix, at its narrowest possible grain: `rebalance_kept_cells` must return `base`
    /// completely UNTOUCHED for any model with a multi-channel raw feature, never reaching
    /// `fit_cell_correction`/`correction_scaffold` at all. Comparing only the RESULT against a
    /// `rebalance:false` run is not a reliable oracle for this on every fixture: the no-harm
    /// guard already refuses any correction that doesn't measurably improve held-out deviance,
    /// so a genuinely misrouted (wrong-axis) correction can independently get rejected on its
    /// own merits and produce the exact same output as a correctly-skipped one -- confirmed
    /// empirically on a harder fixture (see the t-boost-py `rebalance_skip_prevents_the_purely_
    /// silent_variant...` test, whose doc comment documents the profiling numbers). So this test
    /// makes reaching the buggy path unmistakable instead: `y`/`w` are passed one row LONGER
    /// than `serve`'s row count. If the skip fires (`Ok(base)` returned before `y`/`w` are ever
    /// read), the mismatched length is irrelevant and this succeeds trivially. If the skip is
    /// missing, execution reaches `loss.grad_hess(y, raw, w, ..)`, whose OWN shape check
    /// (`SquaredError::grad_hess`, `raw.len() != y.len()`) turns any reintroduction of the bug
    /// into a loud, unconditional `ShapeMismatch` error -- not a maybe-silent numeric wobble a
    /// data-dependent guard could mask.
    #[test]
    fn rebalance_kept_cells_skips_the_whole_model_when_any_raw_feature_is_multichannel() {
        let model = fixture_multichannel_model();
        let serve = fixture_multichannel_serve();
        assert!(
            model.provenance.len() > crate::data::n_raw_features(&model.provenance),
            "fixture must actually be multi-channel for this test to mean anything"
        );
        let bank = model
            .explain_bank(
                &serve,
                RefMeasure::Uniform,
                crate::explain::TableBudget::default(),
                false,
            )
            .unwrap();
        let keep: Vec<FeatureSet> = bank.tables.iter().map(|t| t.u.clone()).collect();
        assert!(!keep.is_empty(), "fixture must realize at least one table");
        let n = serve.0.n_rows as usize;
        let base = retain_tables(&bank, &keep);
        let mut base_raw = vec![0.0_f64; n];
        crate::scoring::score_bank_binned(
            &base,
            &model.schema.cat_encoders,
            &serve.0,
            &mut base_raw,
        )
        .unwrap();

        let bad_y = vec![0.0_f32; n + 1];
        let bad_w = vec![1.0_f32; n + 1];
        let out = rebalance_kept_cells(
            &model,
            base.clone(),
            &base_raw,
            &keep,
            &serve,
            &bad_y,
            &bad_w,
            None,
            &SquaredError,
            RefMeasure::Uniform,
            false,
        )
        .expect("multi-channel model: the skip must return before the mismatched y/w length is ever read");
        assert_eq!(
            out, base,
            "the skip must return `base` completely unchanged"
        );
    }

    // Held-out y is constant at the intercept ⇒ interactions only add overfit variance. Main
    // effects are sticky, so the selector prunes down to the main-effect backbone rather than
    // deleting every rating factor.
    #[test]
    fn prunes_interactions_but_keeps_sticky_mains_when_tables_overfit() {
        let model = fixture_model();
        let x = fixture_serve();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let n = x.0.n_rows as usize;
        let y = vec![bank.f0 as f32; n]; // held-out signal is pure intercept
        let w = vec![1.0_f32; n];
        let folds = [HoldoutFold {
            x: &x.0,
            y: &y,
            w: &w,
            offset: None,
        }];

        let (pruned, report) = prune_bank(
            &bank,
            &crate::cat::CatEncoderStore::new(),
            &folds,
            &SquaredError,
            &PruneConfig::default(),
        )
        .unwrap();
        assert!(
            pruned.tables.iter().all(|t| t.u.order() == 1),
            "only sticky mains should remain"
        );
        assert_eq!(pruned.factored.len(), 0);
        assert_eq!(report.effective_order, 1);
        assert!(report.kept.iter().all(|u| u.order() == 1));
        assert!(report.dropped.iter().all(|u| u.order() > 1));
        assert!(
            report.delta_vs_full > 0.0,
            "pruning overfit interactions must improve held-out deviance, got {}",
            report.delta_vs_full
        );
    }

    #[test]
    fn prune_bank_is_deterministic() {
        let model = fixture_model();
        let x = fixture_serve();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let n = x.0.n_rows as usize;
        let y = vec![bank.f0 as f32; n];
        let w = vec![1.0_f32; n];
        let folds = [HoldoutFold {
            x: &x.0,
            y: &y,
            w: &w,
            offset: None,
        }];
        let (p1, r1) = prune_bank(
            &bank,
            &crate::cat::CatEncoderStore::new(),
            &folds,
            &SquaredError,
            &PruneConfig::default(),
        )
        .unwrap();
        let (p2, r2) = prune_bank(
            &bank,
            &crate::cat::CatEncoderStore::new(),
            &folds,
            &SquaredError,
            &PruneConfig::default(),
        )
        .unwrap();
        assert_eq!(p1, p2, "pruned bank must be deterministic");
        assert_eq!(r1.kept, r2.kept);
        assert_eq!(r1.dropped, r2.dropped);
        // Downward closure: no kept table has a dropped proper subset.
        for kept in &r1.kept {
            for dropped in &r1.dropped {
                assert!(
                    !is_proper_subset(dropped, kept),
                    "kept {kept:?} has dropped subset {dropped:?} — heredity violated"
                );
            }
        }
    }

    #[test]
    fn prune_model_to_tables_keeps_all_on_genuine_signal() {
        let model = fixture_model();
        let x = fixture_serve();
        let y = model.predict(&x.0, None).unwrap();
        let w = vec![1.0_f32; y.len()];
        let sel: Vec<usize> = (0..y.len()).collect();
        let (tm, report) = prune_model_to_tables(
            &model,
            &x,
            &y,
            &w,
            None,
            &SquaredError,
            RefMeasure::Uniform,
            false,
            &sel,
            2,
            &PruneConfig::default(),
        )
        .unwrap();
        assert_eq!(report.dropped.len(), 0);
        // A no-op prune still predicts identically to the full ensemble.
        let ens = model.predict(&x.0, None).unwrap();
        let got = tm.predict_binned(&x.0, None).unwrap();
        for (e, g) in ens.iter().zip(&got) {
            assert!((e - g).abs() < 1e-5);
        }
    }

    #[test]
    fn fold_fidelity_none_is_the_historical_function() {
        // The safety claim in one assertion: the pinned entry point with `None` must produce the
        // SAME bank and the SAME report as the function it wraps, on the same inputs.
        let model = fixture_model();
        let x = fixture_serve();
        let y = model.predict(&x.0, None).unwrap();
        let w = vec![1.0_f32; y.len()];
        let sel: Vec<usize> = (0..y.len()).collect();
        let plain = prune_model_to_tables(
            &model,
            &x,
            &y,
            &w,
            None,
            &SquaredError,
            RefMeasure::Uniform,
            false,
            &sel,
            2,
            &PruneConfig::default(),
        )
        .unwrap();
        let pinned = prune_model_to_tables_pinned(
            &model,
            &x,
            &y,
            &w,
            None,
            &SquaredError,
            RefMeasure::Uniform,
            false,
            &sel,
            2,
            &PruneConfig::default(),
            None,
        )
        .unwrap();
        assert_eq!(plain.0.bank, pinned.0.bank);
        assert_eq!(plain.1.kept, pinned.1.kept);
        assert_eq!(plain.1.dropped, pinned.1.dropped);
    }

    #[test]
    fn fold_fidelity_is_a_no_op_when_the_bank_already_covers_the_candidates() {
        // The augmentation only ever ADDS supports the fold bank is missing. Handed a candidate
        // set the bank already covers (plus one naming an axis this model does not have — the
        // filter that keeps a stale/foreign support list from indexing off the end), it must
        // leave the bank exactly as `None` would.
        let model = fixture_model();
        let x = fixture_serve();
        let n = x.0.n_rows as usize;
        let y = model.predict(&x.0, None).unwrap();
        let w = vec![1.0_f32; n];
        let sel: Vec<usize> = (0..n).step_by(2).collect();
        let fit_rows: Vec<bool> = (0..n).map(|r| r % 2 == 1).collect();
        let supports = vec![vec![0_u32, 1], vec![0, 1, 9], vec![7, 8]];
        let ff = FoldFidelity {
            candidate_supports: &supports,
            fit_rows: &fit_rows,
            spec: FoldFidelitySpec::default(),
        };
        let plain = prune_model_to_tables_pinned(
            &model,
            &x,
            &y,
            &w,
            None,
            &SquaredError,
            RefMeasure::Uniform,
            false,
            &sel,
            2,
            &PruneConfig::default(),
            None,
        )
        .unwrap();
        let pinned = prune_model_to_tables_pinned(
            &model,
            &x,
            &y,
            &w,
            None,
            &SquaredError,
            RefMeasure::Uniform,
            false,
            &sel,
            2,
            &PruneConfig::default(),
            Some(&ff),
        )
        .unwrap();
        assert_eq!(plain.0.bank, pinned.0.bank);
        assert_eq!(plain.1.table_scores.len(), pinned.1.table_scores.len());
    }

    #[test]
    fn prune_model_to_tables_prunes_overfit_interactions_but_keeps_mains() {
        let model = fixture_model();
        let x = fixture_serve();
        let n = x.0.n_rows as usize;
        // Held-out signal is pure intercept ⇒ the model's tables only overfit it.
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let y = vec![bank.f0 as f32; n];
        let w = vec![1.0_f32; n];
        let sel: Vec<usize> = (0..n).collect();
        let (tm, report) = prune_model_to_tables(
            &model,
            &x,
            &y,
            &w,
            None,
            &SquaredError,
            RefMeasure::Uniform,
            false,
            &sel,
            2,
            &PruneConfig::default(),
        )
        .unwrap();
        assert_eq!(report.effective_order, 1);
        assert!(tm.bank.tables.iter().all(|t| t.u.order() == 1));
        assert!(report.kept.iter().all(|u| u.order() == 1));
        assert!(report.dropped.iter().all(|u| u.order() > 1));
        assert!(report.delta_vs_full > 0.0);
    }

    #[test]
    fn multiclass_prune_runs_and_yields_a_valid_simplex() {
        let model = fixture_model();
        let x = fixture_serve();
        let mc = MultiClassModel {
            classes: vec![model.clone(), model.clone()],
            class_labels: vec!["a".into(), "b".into()],
            schema_version: crate::serialize::SCHEMA_VERSION_UNLIFTED,
            cell_refit: None,
        };
        let n = x.0.n_rows as usize;
        let labels: Vec<u32> = (0..n).map(|i| (i % 2) as u32).collect();
        let w = vec![1.0_f32; n];
        let sel: Vec<usize> = (0..n).collect();
        let (mct, report) = prune_multiclass_to_tables(
            &mc,
            &x,
            &labels,
            &w,
            RefMeasure::Uniform,
            &sel,
            1,
            &PruneConfig::default(),
        )
        .unwrap();
        assert_eq!(mct.n_classes(), 2);
        mct.validate().unwrap();
        // Shared keep-set: both classes keep the same feature-sets.
        assert_eq!(
            mct.classes[0].bank.tables.len(),
            mct.classes[1].bank.tables.len()
        );
        // predict_proba rows are a valid simplex.
        let proba = mct.predict_proba(&x.0).unwrap();
        assert_eq!(proba.len(), n * 2);
        for p in proba.chunks(2) {
            assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        }
        let _ = report.effective_order;
    }

    #[test]
    fn multiclass_prune_is_byte_identical_across_thread_counts() {
        let model = fixture_model();
        let x = fixture_serve();
        let mc = MultiClassModel {
            classes: vec![model.clone(), model.clone(), model],
            class_labels: vec!["a".into(), "b".into(), "c".into()],
            schema_version: crate::serialize::SCHEMA_VERSION_UNLIFTED,
            cell_refit: None,
        };
        let n = x.0.n_rows as usize;
        let labels: Vec<u32> = (0..n).map(|i| (i % 3) as u32).collect();
        let w = vec![1.0_f32; n];
        let sel: Vec<usize> = (0..n).collect();
        let run = |threads: usize| -> (MultiClassTableModel, Vec<u8>) {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let (model, report) = prune_multiclass_to_tables(
                    &mc,
                    &x,
                    &labels,
                    &w,
                    RefMeasure::Uniform,
                    &sel,
                    2,
                    &PruneConfig::default(),
                )
                .unwrap();
                (model, serde_json::to_vec(&report).unwrap())
            })
        };
        let one = run(1);
        assert_eq!(one, run(2));
        assert_eq!(one, run(8));
    }
}
