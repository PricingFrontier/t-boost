//! The oblivious Newton split-finder (spec §06.2 / §06.4, milestone M1.4).
//!
//! A tree is grown one shared split per level (depth 1→3). At each level the level
//! histogram (over the admissible axes) is scanned for the `(axis, bin_le)` that
//! maximizes the SUMMED Newton gain across all current leaves; the reserved missing
//! bin is tried both sides and the better direction is recorded as the learned
//! `Split.missing_left`. The feature-budget guard (I1) limits each tree to at most
//! 3 distinct raw features, but later levels may reuse an already-selected raw feature
//! to refine lower-order main/pair surfaces. Growth terminates early only when no
//! admissible fresh or reused candidate clears the `min_split_gain` floor. Leaves are
//! the exact Newton step `w* = −G/(H+λ)·lr`
//! from FULL-PRECISION sums.
//!
//! The split scan is sequential (cheap — O(leaves·axes·bins), independent of row
//! count) with a deterministic first-wins argmax; the parallelism lives in the
//! histogram build (`engine::hist`). Determinism is therefore structural.

use crate::backend::{pb_seed, Stage};
use crate::constraints::{credibility_evidence_n, CredibilityEvidence, CredibilityFloor, MonoSign};
use crate::data::BinnedMatrix;
use crate::engine::hist::{
    build_histogram, build_quantized_histogram, quantize_grad_hess, subtract_sibling_into,
};
use crate::engine::{
    leaf_slots, low_bit, Hist, HistPrecision, InteractionGainHurdleMode, ObliviousTree,
    QuantGradHess, Split, LEGACY_MAX_DEPTH, LEGACY_MAX_ORDER, MAX_DEPTH, MAX_LEAVES, MAX_ORDER,
    MAX_SCAN_LEAVES,
};
use crate::error::PbError;
use crate::explain::FeatureSet;
use crate::loss::GradHess;
use std::collections::BTreeSet;

fn internal(what: &'static str) -> impl Fn() -> PbError {
    move || PbError::Internal { what: what.into() }
}

/// The split-finder's parameters (resolved from the §06 `Config` and the per-fit
/// `FitSpec`). Monotone bounds (§07.5) and credibility floors (§07.2/§07.6) are wired
/// through `monotone` and `credibility`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GrowConfig<'a> {
    /// L2 leaf regularizer `λ` (in `w* = −G/(H+λ)` and the gain).
    pub lambda: f64,
    /// L1 leaf regularizer used to soft-threshold aggregated gradients.
    pub l1_leaf: f64,
    /// Learning rate applied to each leaf value.
    pub lr: f64,
    /// `gamma` floor: a level terminates if the best gain is `<= min_split_gain`.
    pub min_split_gain: f64,
    /// Interaction-admission hurdle (soft heredity, §07.3 spirit): a split that would introduce a
    /// NEW raw feature, raising the tree's fANOVA order, is admitted only if its Newton gain clears
    /// the mode-specific hurdle against this tree's level-1 gain. (Keyed on PROJECTED ORDER, not on
    /// level — so it is unchanged by `max_depth`, and it never gates a reuse split at any depth,
    /// which is correct: a reused feature adds no interaction order.) When the hurdle is
    /// enabled, fresh features also compete with already-used raw features under the ranking score;
    /// reused/lower-order splits win exact ties. `0.0` restores pure greedy new-feature admission,
    /// with reuse only when no fresh feature is available.
    pub interaction_gain_hurdle: f64,
    /// Whole-tree interaction-order cap (`1..=3`); caps distinct raw features, not depth.
    pub max_order: u8,
    /// Whole-tree DEPTH cap (`3..=MAX_DEPTH`); caps split levels, not distinct features.
    /// Levels past the third are reuse-only for free — once `max_order` distinct raws are
    /// on the tree, `fresh_admissible` below is empty by construction.
    pub max_depth: u8,
    /// Leaf-stage `|w*|`-clamp resolved from `Config.max_delta_step` ∨ `Loss::max_delta_step()`
    /// (§05.6). `None` = uncapped; applied on the full-precision aggregated Newton step.
    pub max_delta_step: Option<f64>,
    /// Histogram precision for split search.
    pub hist_precision: HistPrecision,
    /// Base deterministic seed for quantized stochastic rounding.
    pub quant_seed: u64,
    /// Boosting round, used as the quantization re-seed coordinate.
    pub round: u32,
    /// Decaying deterministic split-score noise (§09.6). `0.0` is exactly inert.
    pub random_strength: f64,
    /// Optional whole-tree feature-group whitelist (§07): every realized tree support
    /// must be a subset of at least one group.
    pub groups: Option<&'a [FeatureSet]>,
    /// Optional per-axis monotone signs resolved at fit entry (§07).
    pub monotone: Option<&'a [Option<MonoSign>]>,
    /// Optional soft table-size admission prior (§07.3/§07.4). It changes only the
    /// ranking score; raw Newton gain still gates and is stored.
    pub table_budget_penalty: Option<TableBudgetPenalty>,
    /// The realized-extent accumulator `table_budget_penalty` reads (spec §07.4 stage 5),
    /// owned by the caller's round loop and updated as each tree commits. `None` keeps
    /// `TableBudgetPenalty::multiplier`'s legacy `grid.n_bins` behavior exactly (every
    /// caller not yet threaded onto a live round loop -- tests, and the not-yet-updated
    /// multiclass fit).
    pub realized_extent: Option<&'a RealizedExtent>,
    /// Per-leaf credibility floors + `path_smooth` (§07.2/§07.6). All-zero is exactly
    /// inert. The three hard floors veto a candidate whose level produces any
    /// under-supported cell; `path_smooth` shrinks final leaves toward their parent.
    pub credibility: CredibilityFloor,
    /// Which per-node quantity feeds `path_smooth`'s `Z = n/(n+k)` blend (§07.6, evidence
    /// generalization). Resolved once per fit from the loss objective
    /// (`engine::boost::credibility_evidence_for_loss`); `Count` (the default/legacy value)
    /// reproduces the pre-generalization behavior exactly. Read by `apply_path_smooth` and by
    /// `leaf_refine`'s companion shrink; inert (never read) whenever `credibility.path_smooth
    /// <= 0.0`.
    pub credibility_evidence: CredibilityEvidence,
    /// `true` iff every sample weight is exactly `1.0` (the engine sets this only when the
    /// caller supplied NO weights, so the weight vector is the materialized all-ones). It
    /// lets the histogram skip the per-row `Σw` accumulation and set `wsum = count` (which
    /// is bit-exact for unit weights: summing `1.0` `k<2^53` times is exact). Conservative:
    /// `false` whenever weights were provided, even if they happen to all be `1.0`.
    pub unit_weight: bool,
    /// Enable the level-2 FullF64 histogram-subtraction fast path (build the smaller sibling
    /// child, derive the larger from the level-1 parent). `true` in production; a kill-switch
    /// and the A/B reference for the equivalence tests (subtraction reproduces the full-build
    /// tree, with g/h differing only at ~1e-11). Inert unless FullF64 and depth reaches 2.
    pub hist_subtraction: bool,
}

/// A candidate level split with its summed Newton gain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Candidate {
    /// Global axis (feature column) index.
    pub axis: u32,
    /// Inclusive upper bin for the low (left) child.
    pub bin_le: u8,
    /// Learned missing-bin direction (bin 0 routes low when `true`).
    pub missing_left: bool,
    /// The summed Newton gain `½ Σ_leaf [G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)]`.
    pub gain: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ScoredCandidate {
    candidate: Candidate,
    score: f64,
}

type LevelCandidate = (ScoredCandidate, Hist, Vec<u32>);

fn choose_level_candidate(
    fresh: Option<LevelCandidate>,
    reused: Option<LevelCandidate>,
) -> Option<LevelCandidate> {
    match (fresh, reused) {
        (Some(fresh), Some(reused)) if fresh.0.score > reused.0.score => Some(fresh),
        (Some(_fresh), Some(reused)) => Some(reused),
        (Some(fresh), None) => Some(fresh),
        (None, Some(reused)) => Some(reused),
        (None, None) => None,
    }
}

/// Per-fit interaction-hurdle state passed into one tree grow. This is intentionally separate from
/// the serialized [`ObliviousTree`]: gains shape training admission only and are not model state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct InteractionHurdleState {
    mode: InteractionGainHurdleMode,
    first_split_gain_reference: Option<f64>,
}

impl InteractionHurdleState {
    pub(crate) fn new(
        mode: InteractionGainHurdleMode,
        first_split_gain_reference: Option<f64>,
    ) -> Self {
        Self {
            mode,
            first_split_gain_reference,
        }
    }

    fn effective_hurdle(self, base_hurdle: f64, level: usize, first_split_gain: f64) -> f64 {
        if base_hurdle <= 0.0 || level == 0 {
            return base_hurdle.max(0.0);
        }
        match self.mode {
            InteractionGainHurdleMode::Fixed => base_hurdle,
            InteractionGainHurdleMode::Adaptive => {
                // `level` here is the PROJECTED ORDER minus one (see the caller's
                // `hurdle_level`), so this multiplier is an ORDER escalator, not a depth
                // one: order 2 pays 1x, order 3 pays 2x, order 4 pays 4x, and so on.
                //
                // It used to be written `if level >= 2 { 2.0 } else { 1.0 }`, which is the
                // SAME function on `1..=2` — the only orders a pre-lift fit could project —
                // so the order-<=3 path is byte-identical. What changes is that the 3->4
                // transition is now charged twice what the 2->3 transition is, rather than
                // the same. That is the growth-side half of "a 4-way effect only where
                // there is genuine signal": the fourth feature must clear a hurdle twice as
                // tall as the third did, against the same level-1 reference gain.
                let order_multiplier = if level >= 1 {
                    2.0_f64.powi(i32::try_from(level.saturating_sub(1)).unwrap_or(i32::MAX))
                } else {
                    1.0
                };
                let reference = self
                    .first_split_gain_reference
                    .filter(|g| g.is_finite() && *g > 0.0)
                    .unwrap_or(first_split_gain);
                let pressure = if reference > 0.0 && first_split_gain.is_finite() {
                    let relative_main_gain = (first_split_gain / reference).clamp(0.0, 1.0);
                    0.25 + 0.75 * relative_main_gain.sqrt()
                } else {
                    1.0
                };
                base_hurdle * order_multiplier * pressure
            }
        }
    }

    fn admission_gain_floor(
        self,
        base_hurdle: f64,
        level: usize,
        first_split_gain: f64,
        parent_level_gain: Option<f64>,
    ) -> f64 {
        let main_floor =
            self.effective_hurdle(base_hurdle, level, first_split_gain) * first_split_gain;
        match self.mode {
            InteractionGainHurdleMode::Adaptive if level >= 2 => {
                let Some(parent_gain) = parent_level_gain.filter(|g| g.is_finite() && *g > 0.0)
                else {
                    return main_floor;
                };
                let parent_floor =
                    self.effective_hurdle(base_hurdle, 1, first_split_gain) * parent_gain;
                main_floor.max(parent_floor)
            }
            _ => main_floor,
        }
    }
}

impl Default for InteractionHurdleState {
    fn default() -> Self {
        Self::new(InteractionGainHurdleMode::Fixed, None)
    }
}

/// Internal tree-grow payload. The tree and leaf map feed prediction/refinement; the first split
/// gain feeds the next round's adaptive interaction hurdle.
#[derive(Debug)]
pub(crate) struct GrowResult {
    pub(crate) tree: ObliviousTree,
    pub(crate) leaf_of_row: Vec<u8>,
    pub(crate) first_split_gain: f64,
}

/// Per-raw-feature realized split-border accumulator (spec §07.4 stage 5). Owned by the
/// fit's round loop (`boost.rs`), one instance per model fit (a bagged fit's parallel bag
/// closures each own an independent instance -- never shared, so this introduces no
/// cross-bag/cross-thread state and the existing thread-count determinism is unaffected).
/// Tracks the distinct `bin_le` thresholds chosen as a split point for each raw feature,
/// across every tree committed SO FAR in this fit. `extent(raw) = borders.len() + 2` (the
/// missing cell plus the realized finite intervals) mirrors `MergedGrids::from_model`
/// (explain.rs) exactly, so [`TableBudgetPenalty::multiplier`] tracks the same storage
/// cost the eventual merged-grid table will realize, rather than the axis's full
/// theoretical resolution.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RealizedExtent {
    per_raw: Vec<BTreeSet<u8>>,
}

impl RealizedExtent {
    /// A fresh accumulator for a fit over `n_raw_features` raw features. Every feature
    /// starts at the minimum extent (2: the missing cell plus one undivided finite
    /// interval), matching a never-split axis's true realized state.
    pub(crate) fn new(n_raw_features: usize) -> Self {
        Self {
            per_raw: vec![BTreeSet::new(); n_raw_features],
        }
    }

    /// Record one committed tree's splits. Deterministic regardless of call order within
    /// a single accumulator: `BTreeSet` insertion is order-independent, and this is only
    /// ever called from one fit's own sequential round loop (never shared across the
    /// parallel bag closures), so thread count cannot affect the result.
    ///
    /// # Errors
    /// [`PbError::Internal`] if a split's axis, or its provenance-mapped raw feature id,
    /// escapes this accumulator's bounds (a build/shape bug).
    pub(crate) fn record_tree(
        &mut self,
        splits: &[Split],
        x: &BinnedMatrix,
    ) -> Result<(), PbError> {
        for split in splits {
            let prov = x
                .provenance
                .get(split.axis as usize)
                .ok_or_else(internal("realized-extent split axis provenance"))?;
            let raw = prov.raw.0 as usize;
            let set = self
                .per_raw
                .get_mut(raw)
                .ok_or_else(internal("realized-extent raw feature index"))?;
            set.insert(split.bin_le);
        }
        Ok(())
    }

    /// The realized extent for raw feature `raw`: `1 (missing) + n_finite_intervals`,
    /// where `n_finite_intervals = distinct_realized_borders + 1`. `2` for a feature never
    /// split so far.
    fn extent(&self, raw: u32) -> u64 {
        self.per_raw
            .get(raw as usize)
            .map_or(2, |set| set.len() as u64 + 2)
    }

    /// Is `bin` ALREADY a realized border of `raw`? (§6.4 refinement, P-D2.)
    ///
    /// A split landing on a border this fit has already committed adds **zero** cells to
    /// the eventual merged grid — `MergedGrids::from_model` unions border indices, so the
    /// second tree to choose `x <= 37` costs nothing that the first did not already pay.
    /// A split on a NEW border adds exactly one column. Charging that difference is what
    /// turns the table-budget prior from "how wide is this support" into "what does this
    /// split cost", and it is the mechanism that steers a lifted tree toward sharpening
    /// the fit on the grid it already has rather than growing the grid.
    ///
    /// The hot split scan uses the [`RealizedExtent::realized_bins`] bitset instead; this
    /// exact-semantics form is what the unit tests pin the bitset against.
    #[cfg(test)]
    fn contains(&self, raw: u32, bin: u8) -> bool {
        self.per_raw
            .get(raw as usize)
            .is_some_and(|set| set.contains(&bin))
    }

    /// The realized borders of `raw` as a 256-bit set, for O(1) per-candidate lookup
    /// inside the split scan (the scan is the hottest loop in the booster; a `BTreeSet`
    /// probe per candidate bin per axis is not affordable there).
    fn realized_bins(&self, raw: u32) -> [u64; 4] {
        let mut bits = [0u64; 4];
        if let Some(set) = self.per_raw.get(raw as usize) {
            for &b in set {
                let w = usize::from(b) >> 6;
                if let Some(slot) = bits.get_mut(w) {
                    *slot |= 1u64 << (u32::from(b) & 63);
                }
            }
        }
        bits
    }
}

/// `true` iff `bin` is set in a [`RealizedExtent::realized_bins`] bitset.
#[must_use]
fn realized_bit(bits: &[u64; 4], bin: u8) -> bool {
    let w = usize::from(bin) >> 6;
    bits.get(w)
        .is_some_and(|word| (word >> (u32::from(bin) & 63)) & 1 == 1)
}

/// Soft table-size admission prior (§07.3/§07.4).
///
/// This is a ranking-only multiplier:
/// `score = gain.max(0) * (budget(order) / max(budget(order), projected_cells))^beta`.
/// `beta = 0` is represented as `None` by [`TableBudgetPenalty::new`], which makes
/// the split finder exactly recover the unpenalized ordering and tie behavior.
///
/// `budget(order)` shrinks by `order_shrink` for each interaction order above
/// [`LEGACY_MAX_ORDER`] — see [`crate::InteractionPolicy::table_budget_order_shrink`].
/// The exponent is zero for every support a pre-lift fit could realize, so the shrink is
/// exactly inert at order `<= 3`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TableBudgetPenalty {
    beta: f64,
    budget_cells: u64,
    order_shrink: f64,
}

impl TableBudgetPenalty {
    pub(crate) fn new(beta: f64, budget_cells: u64, order_shrink: f64) -> Option<Self> {
        (beta > 0.0).then_some(Self {
            beta,
            budget_cells,
            // A non-finite or <=0 shrink is treated as "no shrink" rather than rejected:
            // this is a ranking prior, and a malformed prior must never be able to make a
            // budget of zero (which would drive every multiplier to 0 and flatten the
            // ranking entirely).
            order_shrink: if order_shrink.is_finite() && order_shrink >= 1.0 {
                order_shrink
            } else {
                1.0
            },
        })
    }

    /// The cell allowance a support of `order` distinct raw features is measured against.
    ///
    /// Floored at 1: a budget of 0 would make `score` zero for every candidate and erase
    /// the gain ordering the prior is only supposed to tilt.
    fn budget_for_order(self, order: usize) -> u64 {
        let over = order.saturating_sub(LEGACY_MAX_ORDER);
        if over == 0 || self.order_shrink <= 1.0 {
            return self.budget_cells;
        }
        let shrunk = (self.budget_cells as f64) / self.order_shrink.powi(over as i32);
        // `as u64` on a finite non-negative f64 saturates at 0, which `max(1)` lifts.
        (shrunk as u64).max(1)
    }

    /// `realized_extent`, when present, replaces each axis's full `grid.n_bins` with its
    /// per-raw-feature REALIZED extent so far (spec §07.4 stage 5) -- `None` (every
    /// existing caller that has not been threaded onto a live round loop, e.g. tests and
    /// the not-yet-updated multiclass fit) keeps the prior `grid.n_bins` behavior exactly.
    /// The ranking multiplier for a candidate on `candidate_axis`, as the pair
    /// `(if it lands on an ALREADY-realized border, if it opens a NEW one)`.
    ///
    /// `charge_new_border == false` reproduces the pre-P-D2 behavior exactly — both halves
    /// of the pair are the single value the old `multiplier` returned, so the caller's
    /// per-candidate choice is a no-op. It is `true` only for a LIFTED fit
    /// (`max_depth > 3`), which is what keeps the default path byte-identical: charging
    /// the new border changes the score of every FRESH candidate too (a never-split axis
    /// goes from extent 2 to 3, which is the honest count — one border makes two finite
    /// intervals plus the missing cell), and that would reorder depth-3 fits.
    fn multiplier_pair(
        self,
        x: &BinnedMatrix,
        used_axes: &[u32],
        candidate_axis: u32,
        realized_extent: Option<&RealizedExtent>,
        charge_new_border: bool,
    ) -> Result<(f64, f64), PbError> {
        let mut cells = 1u64;
        let mut candidate_raw_extent: Option<u64> = None;
        let mut cells_without_candidate_raw = 1u64;
        let candidate_raw = x
            .provenance
            .get(candidate_axis as usize)
            .ok_or_else(internal("budget-prior candidate axis provenance"))?
            .raw
            .0;
        // DISTINCT raw features only (the merged grid dedupes repeats) -> MAX_ORDER.
        let mut seen_raws: smallvec::SmallVec<[u32; MAX_ORDER]> = smallvec::SmallVec::new();
        for axis in used_axes
            .iter()
            .copied()
            .chain(std::iter::once(candidate_axis))
        {
            let raw = x
                .provenance
                .get(axis as usize)
                .ok_or_else(internal("budget-prior axis provenance"))?
                .raw
                .0;
            if seen_raws.contains(&raw) {
                continue;
            }
            seen_raws.push(raw);
            let extent = match realized_extent {
                Some(acc) => acc.extent(raw),
                None => {
                    let grid = x
                        .grids
                        .get(axis as usize)
                        .ok_or_else(internal("budget-prior axis grid"))?;
                    u64::from(grid.n_bins)
                }
            };
            let overflow = || PbError::Internal {
                what: "budget-prior projected cell count overflowed u64".into(),
            };
            // The CANDIDATE's own raw is the only one whose extent the candidate can move,
            // so keep the product of the others so the "+1 border" variant is one multiply
            // rather than a second pass.
            if candidate_raw == raw && candidate_raw_extent.is_none() {
                candidate_raw_extent = Some(extent);
            } else {
                cells_without_candidate_raw = cells_without_candidate_raw
                    .checked_mul(extent)
                    .ok_or_else(overflow)?;
            }
            cells = cells.checked_mul(extent).ok_or_else(overflow)?;
        }
        // The projected support is exactly `seen_raws` — the candidate is already folded in
        // above — so its length is the order this split would realize.
        let budget = self.budget_for_order(seen_raws.len());
        let score = |cells: u64| -> f64 {
            if cells <= budget {
                1.0
            } else {
                (budget as f64 / cells as f64).powf(self.beta)
            }
        };
        let existing = score(cells);
        if !charge_new_border {
            return Ok((existing, existing));
        }
        let cand_extent =
            candidate_raw_extent.ok_or_else(internal("budget-prior candidate raw"))?;
        let new_cells = cells_without_candidate_raw
            .checked_mul(cand_extent + 1)
            .ok_or_else(|| PbError::Internal {
                what: "budget-prior projected cell count overflowed u64".into(),
            })?;
        Ok((existing, score(new_cells)))
    }
}

/// Deterministic per-candidate split-score noise.
///
/// This is deliberately a ranking-only term: the raw Newton gain still gates
/// `min_split_gain` and is what gets stored on [`Candidate`]. The seed stream is
/// position-stable in `(seed, round, level, axis, bin, missing_left)`, so thread
/// count and scan partitioning cannot affect the selected candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SplitNoise {
    seed: u64,
    round: u32,
    level: usize,
    strength: f64,
}

impl SplitNoise {
    fn new(seed: u64, round: u32, level: usize, strength: f64) -> Option<Self> {
        (strength > 0.0).then_some(Self {
            seed,
            round,
            level,
            strength,
        })
    }

    fn adjustment(self, axis: u32, bin_le: usize, missing_left: bool) -> Result<f64, PbError> {
        let level = u64::try_from(self.level).map_err(|_| PbError::Internal {
            what: "split-noise level exceeded u64".into(),
        })?;
        let bin = u64::try_from(bin_le).map_err(|_| PbError::Internal {
            what: "split-noise bin exceeded u64".into(),
        })?;
        let salt = u64::from(axis).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ bin.wrapping_mul(0xBF58_476D_1CE4_E5B9)
            ^ level.wrapping_mul(0x94D0_49BB_1331_11EB)
            ^ u64::from(missing_left);
        let bits = pb_seed(self.seed ^ salt, self.round, Stage::SplitNoise as u32, axis);
        let mantissa = bits >> 11;
        const TWO_53: f64 = 9_007_199_254_740_992.0;
        let unit = mantissa as f64 / TWO_53;
        let centered = 2.0 * unit - 1.0;
        let decay = (f64::from(self.round) + 1.0).sqrt();
        Ok(centered * self.strength / decay)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MonotoneScan<'a> {
    level: usize,
    chosen: &'a [Option<MonoSign>],
    /// The actual `(axis, bin_le, missing_left)` of the `level` ancestor splits already
    /// committed in THIS tree (parallel in meaning to `chosen`, but carrying the real
    /// threshold structure `chosen`'s signs-only view discards). Feeds
    /// [`monotone_reachable`] so a same-axis nested-threshold candidate can tell a
    /// logically-unreachable cousin apart from a merely-empty, reachable one.
    ancestor_splits: &'a [Split],
    candidate_axis_signs: &'a [Option<MonoSign>],
    lr: f64,
    l1_leaf: f64,
    max_delta_step: Option<f64>,
}

/// Ranking-only split aids. These can choose among admissible raw-gain candidates,
/// but they never alter the stored [`Candidate::gain`] and never bypass
/// `min_split_gain`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RankingContext<'a> {
    monotone: Option<MonotoneScan<'a>>,
    noise: Option<SplitNoise>,
    /// Per-admissible-axis `(already-realized border, new border)` ranking multipliers.
    /// The two halves are equal unless the fit is lifted — see
    /// [`TableBudgetPenalty::multiplier_pair`].
    table_penalties: Option<&'a [(f64, f64)]>,
    /// Per-admissible-axis 256-bit set of borders this fit has ALREADY realized on that
    /// axis's raw feature, so the scan can pick between the two halves in O(1). `None`
    /// whenever the fit is not lifted.
    realized_bins: Option<&'a [[u64; 4]]>,
}

/// Twice the constrained quadratic improvement for a leaf.
///
/// Without L1/clamping this is `G²/(H+λ)`, matching the usual split-gain algebra.
/// When `l1_leaf` or `max_delta_step` is active, the split scan must rank the gain
/// the emitted leaf value can actually realize rather than the unconstrained Newton
/// optimum.
fn newton_term(g: f64, h: f64, lambda: f64, l1_leaf: f64, max_delta_step: Option<f64>) -> f64 {
    let denom = h + lambda;
    if denom > 0.0 {
        let g = soft_threshold(g, l1_leaf);
        let w = match max_delta_step {
            Some(d) => (-g / denom).clamp(-d, d),
            None => -g / denom,
        };
        (-2.0 * g * w - denom * w * w).max(0.0)
    } else {
        0.0
    }
}

fn soft_threshold(g: f64, l1_leaf: f64) -> f64 {
    if l1_leaf <= 0.0 {
        return g;
    }
    g.signum() * (g.abs() - l1_leaf).max(0.0)
}

fn newton_leaf(
    g: f64,
    h: f64,
    lambda: f64,
    l1_leaf: f64,
    lr: f64,
    max_delta_step: Option<f64>,
) -> f64 {
    let denom = h + lambda;
    let g = soft_threshold(g, l1_leaf);
    let w = if denom > 0.0 { -g / denom } else { 0.0 };
    let w = match max_delta_step {
        Some(d) => w.clamp(-d, d),
        None => w,
    };
    lr * w
}

/// Determine which of the `2^depth` (`depth <= 3`) leaf-index bit patterns are logically
/// REACHABLE by some row, given `depth` levels' `(axis, bin_le, missing_left)` (level
/// `i`'s bit is `1 << i`; bit=1 means "value <= bin_le", matching `low_bit`,
/// engine/mod.rs). A pattern is unreachable only when two or more levels test the SAME
/// axis and no bin value -- finite, or the single missing sentinel, which
/// deterministically routes to `missing_left` -- can satisfy every level's direction at
/// once (e.g. `bin<=2` at one level and `bin>5` at another on the SAME axis: no finite bin
/// is both). Different axes are independent dimensions and never create a contradiction; a
/// reused axis stays reachable via the missing route even when its finite-value interval
/// is empty (a single missing row realizes exactly the bit pattern where every level in
/// the group takes its own `missing_left` direction). A MERELY-empty (zero training
/// support) but reachable cell stays `true` here -- this function only ever removes cells
/// that no row, including a missing one, could occupy by construction.
///
/// # Errors
/// [`PbError::Internal`] if `levels.len() > MAX_DEPTH` (an oblivious tree's own depth cap).
pub(crate) fn monotone_reachable(
    levels: &[(u32, u8, bool)],
) -> Result<[bool; MAX_LEAVES], PbError> {
    let depth = levels.len();
    if depth > MAX_DEPTH {
        return Err(PbError::Internal {
            what: format!(
                "monotone reachability depth {depth} exceeds the oblivious-tree cap {MAX_DEPTH}"
            ),
        });
    }
    let n_leaves = 1usize << depth;
    let mut mask = [true; MAX_LEAVES];
    for idx in 0..n_leaves {
        for a in 0..depth {
            let axis_a = levels
                .get(a)
                .ok_or_else(internal("reachability level a"))?
                .0;
            let mut lower: Option<u8> = None;
            let mut upper: Option<u8> = None;
            let mut matches_missing_route = true;
            let mut group_size = 0usize;
            for b in 0..depth {
                let &(axis_b, t, missing_left) =
                    levels.get(b).ok_or_else(internal("reachability level b"))?;
                if axis_b != axis_a {
                    continue;
                }
                group_size += 1;
                let low = (idx >> b) & 1 == 1;
                if low != missing_left {
                    matches_missing_route = false;
                }
                if low {
                    upper = Some(upper.map_or(t, |u| u.min(t)));
                } else {
                    lower = Some(lower.map_or(t, |l| l.max(t)));
                }
            }
            if group_size < 2 || matches_missing_route {
                continue;
            }
            if let (Some(l), Some(u)) = (lower, upper) {
                if l >= u {
                    *mask
                        .get_mut(idx)
                        .ok_or_else(internal("reachability mask index"))? = false;
                    break;
                }
            }
        }
    }
    Ok(mask)
}

fn candidate_monotone_ok(
    values: &[f64],
    depth: usize,
    signs: &[Option<MonoSign>],
    reachable: &[bool],
) -> Result<bool, PbError> {
    let n_leaves = 1usize << depth;
    if values.len() != n_leaves || signs.len() != depth || reachable.len() != n_leaves {
        return Err(PbError::Internal {
            what: "monotone candidate shape mismatch".into(),
        });
    }
    for (level, sign) in signs.iter().enumerate() {
        let Some(sign) = sign else {
            continue;
        };
        if matches!(sign, MonoSign::None) {
            continue;
        }
        let bit = 1usize << level;
        for high in 0..n_leaves {
            if high & bit != 0 {
                continue;
            }
            let low = high | bit;
            let low_reachable = *reachable
                .get(low)
                .ok_or_else(internal("monotone reachable low"))?;
            let high_reachable = *reachable
                .get(high)
                .ok_or_else(internal("monotone reachable high"))?;
            if !low_reachable || !high_reachable {
                // A logically-unreachable cousin can never be served; excluding it from
                // the veto is the §07.5 fix -- a merely-empty (zero support) but
                // REACHABLE cousin still participates below, unchanged.
                continue;
            }
            let low_v = *values.get(low).ok_or_else(internal("monotone low"))?;
            let high_v = *values.get(high).ok_or_else(internal("monotone high"))?;
            let ok = match sign {
                MonoSign::Increasing => low_v <= high_v,
                MonoSign::Decreasing => low_v >= high_v,
                MonoSign::None => true,
            };
            if !ok {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Project a `2^depth`-leaf vector onto the monotone cone (§07.5) so every constrained level's
/// cousin pairs satisfy the required ordering, IN PLACE. The grow-time candidate filter
/// ([`candidate_monotone_ok`]) keeps the chosen STRUCTURE feasible, but any path that
/// RECOMPUTES leaves (MVS refit on full rows, the §09 fully-corrective ridge refit, or a
/// quantized-histogram round-off) can invert a cousin pair; applying this clamp at every
/// leaf-finalization site guarantees the monotone guarantee holds on the served model
/// (not just the freshly-grown one). Iterative cousin-pair pooling (POCS over the
/// half-space constraints) converges to a feasible point — the cone is non-empty (the
/// all-equal vector is always feasible). Leaf-VALUE only on a fixed structure, so I2 is
/// untouched. No-op when no constrained level is present.
pub(crate) fn clamp_monotone(
    leaves: &mut [f32],
    splits: &[Split],
    depth: usize,
    axis_signs: Option<&[Option<MonoSign>]>,
) -> Result<(), PbError> {
    // B3 (P-D0): these used to silently `.min(3)` the caller's depth, which at depth 4+
    // would have clamped only the low 3 bits of a 2^depth leaf array and silently left the
    // rest un-projected. Fail closed instead.
    if depth > MAX_DEPTH || splits.len() < depth || leaves.len() < (1usize << depth) {
        return Err(PbError::Internal {
            what: format!(
                "clamp_monotone shape: depth {depth} (cap {MAX_DEPTH}), {} splits, {} leaves",
                splits.len(),
                leaves.len()
            ),
        });
    }
    let Some(axis_signs) = axis_signs else {
        return Ok(());
    };
    // Resolve the per-level sign from each level's split axis; bail if none constrained.
    let mut level_sign: [Option<MonoSign>; MAX_DEPTH] = [None; MAX_DEPTH];
    let mut any = false;
    for (level, split) in splits.iter().enumerate().take(depth) {
        let s = axis_signs
            .get(split.axis as usize)
            .copied()
            .flatten()
            .filter(|s| matches!(s, MonoSign::Increasing | MonoSign::Decreasing));
        if s.is_some() {
            *level_sign
                .get_mut(level)
                .ok_or_else(internal("level sign"))? = s;
            any = true;
        }
    }
    if !any {
        return Ok(());
    }
    let d = depth;
    // Logically-unreachable cells (same-axis nested-threshold containment, e.g. a level-0
    // `x>5` ancestor combined with a level-1 `x<=2` candidate) are excluded from the clamp
    // below, the same way `candidate_monotone_ok` excludes them from the grow-time veto
    // (§07.5 fix): pooling a reachable leaf's value toward an unreachable phantom's forced
    // 0.0 has no statistical justification, since that cell can never be served.
    let mut reach_levels: [(u32, u8, bool); MAX_DEPTH] = [(0, 0, false); MAX_DEPTH];
    for (i, split) in splits.iter().enumerate().take(d) {
        *reach_levels
            .get_mut(i)
            .ok_or_else(internal("clamp reachability slot"))? =
            (split.axis, split.bin_le, split.missing_left);
    }
    let reach_slice = reach_levels
        .get(..d)
        .ok_or_else(internal("clamp reachability levels"))?;
    let reachable = monotone_reachable(reach_slice)?;
    let n_leaves = 1usize << d;
    // Sweep budget. This was a flat `64`, chosen when the cone had at most 8 leaves and 3
    // constrained levels. At `MAX_DEPTH = 8` a cone has 256 leaves and up to 8 constrained
    // levels — up to `8 * 128 = 1024` cousin pairs per sweep — so a flat 64 stopped being a
    // budget with headroom and became a coin flip. Worse, the old loop simply FELL OUT of it
    // and returned `Ok(())` carrying a still-non-monotone leaf vector: a monotone constraint
    // that silently stops holding is the worst failure this file can produce, because the
    // served model then breaks a promise the filing makes in writing.
    //
    // Two changes. The budget scales with the problem, and exhausting it is an ERROR rather
    // than a shrug.
    //
    // The floor at the old 64 is not decoration. `16 * n_leaves` alone is 32 sweeps at depth 1
    // and 64 at depth 2 — SMALLER than the constant it replaced, on shapes early split
    // termination produces routinely. Pairing a reduced budget with a newly-fatal exhaustion
    // would be a regression precisely where the change was advertised as strictly safer, so
    // the bound is `max(64, 16 * n_leaves)` and the claim "no fit that converged before can
    // newly fail" is true at EVERY depth rather than only at 3 and above.
    //
    // Termination is not in doubt in exact arithmetic: each pooling step replaces a violating
    // pair by its mean, the exact Euclidean projection onto that pair's half-space, and POCS
    // over finitely many closed convex half-spaces with non-empty intersection (the all-equal
    // vector is always feasible) converges with the sum of squared deviations non-increasing.
    // In `f32` that argument has a seam: when `lo_v` and `hi_v` are within an ULP,
    // `0.5 * (lo_v + hi_v)` can round back onto an endpoint, so a step can fail to decrease
    // the objective and boundary chatter is not provably impossible. That is why exhaustion
    // re-checks feasibility below instead of refusing outright — a leaf vector that is
    // monotone to within rounding is exactly what the old code shipped, and shipping it is
    // correct; shipping a GENUINELY violated constraint is not.
    let max_sweeps = 16usize.saturating_mul(n_leaves).max(64);
    let mut converged = false;
    for _iter in 0..max_sweeps {
        let mut changed = false;
        for (level, sign) in level_sign.iter().enumerate().take(d) {
            let Some(sign) = sign else {
                continue;
            };
            let bit = 1usize << level;
            for high in 0..n_leaves {
                if high & bit != 0 {
                    continue; // `high` = bit_level 0 = HIGH feature value
                }
                let low = high | bit; // `low` = bit_level 1 = LOW feature value
                let low_reachable = *reachable
                    .get(low)
                    .ok_or_else(internal("clamp reachable low"))?;
                let high_reachable = *reachable
                    .get(high)
                    .ok_or_else(internal("clamp reachable high"))?;
                if !low_reachable || !high_reachable {
                    continue;
                }
                let lo_v = *leaves.get(low).ok_or_else(internal("clamp low"))?;
                let hi_v = *leaves.get(high).ok_or_else(internal("clamp high"))?;
                // Increasing: low feature value ⇒ low response ⇒ lo_v <= hi_v.
                let violated = match sign {
                    MonoSign::Increasing => lo_v > hi_v,
                    MonoSign::Decreasing => lo_v < hi_v,
                    MonoSign::None => false,
                };
                if violated {
                    let avg = 0.5 * (lo_v + hi_v);
                    *leaves.get_mut(low).ok_or_else(internal("clamp low set"))? = avg;
                    *leaves
                        .get_mut(high)
                        .ok_or_else(internal("clamp high set"))? = avg;
                    changed = true;
                }
            }
        }
        if !changed {
            converged = true;
            break;
        }
    }
    if converged {
        return Ok(());
    }

    // Exhausted the budget. Decide on the LEAVES, not on the counter.
    //
    // The distinction matters because the two ways to get here are not equally bad. Genuine
    // non-convergence would be a broken monotone promise and must refuse. But `f32` chatter at
    // the cone boundary — where `0.5 * (lo + hi)` rounds back onto an endpoint and the pair
    // keeps re-flagging — leaves a vector that is monotone to within rounding, which is
    // exactly what the previous implementation shipped for years and is correct to ship.
    // Refusing on the counter alone would turn a benign, previously-invisible numerical
    // detail into a fit-killing error.
    //
    // The bar is a few ULPs of the leaf scale. `8 * eps * scale` is four doublings of the
    // worst single-step rounding, so it admits chatter and nothing that a reader of the served
    // table could ever notice, while a real inversion — which is a gain-driven quantity, not a
    // rounding one — sits orders of magnitude above it.
    let scale = leaves
        .iter()
        .take(n_leaves)
        .fold(1.0_f32, |acc, v| acc.max(v.abs()));
    let tol = 8.0 * f32::EPSILON * scale;
    let mut worst = 0.0_f32;
    for (level, sign) in level_sign.iter().enumerate().take(d) {
        let Some(sign) = sign else { continue };
        let bit = 1usize << level;
        for high in 0..n_leaves {
            if high & bit != 0 {
                continue;
            }
            let low = high | bit;
            let (Some(&lr), Some(&hr)) = (reachable.get(low), reachable.get(high)) else {
                continue;
            };
            if !lr || !hr {
                continue;
            }
            let lo_v = *leaves.get(low).ok_or_else(internal("clamp low recheck"))?;
            let hi_v = *leaves
                .get(high)
                .ok_or_else(internal("clamp high recheck"))?;
            let gap = match sign {
                MonoSign::Increasing => lo_v - hi_v,
                MonoSign::Decreasing => hi_v - lo_v,
                MonoSign::None => 0.0,
            };
            worst = worst.max(gap);
        }
    }
    if worst > tol {
        return Err(PbError::Internal {
            what: format!(
                "clamp_monotone did not reach the monotone cone in {max_sweeps} sweeps \
                 (depth {d}, {n_leaves} leaves): worst residual violation {worst:e} exceeds \
                 the {tol:e} rounding bar, so these leaves would break a declared monotone \
                 constraint on the served model"
            ),
        });
    }
    Ok(())
}

/// Scan one level's histogram for the best shared `(axis, bin_le, missing_left)`
/// split (spec §06.2). `axes[p]` is the global axis of histogram column `p`, and
/// `n_data_bins[p]` is that axis's data-bin count (candidate `bin_le ∈ 1..=ndb-1`).
/// Returns the best ranking-score candidate whose raw Newton gain clears
/// `min_split_gain`, or `None` (graceful early-termination). With no split noise,
/// the ranking score is the raw gain. Ties break deterministically: lowest axis,
/// then lowest `bin_le`, then `missing_left = false` (sequential first-wins,
/// strict `>`).
///
/// # Errors
/// [`PbError::Internal`] on an out-of-range histogram offset (a build/shape bug).
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn best_level_split(
    hist: &Hist,
    axes: &[u32],
    n_data_bins: &[usize],
    lambda: f64,
    l1_leaf: f64,
    max_delta_step: Option<f64>,
    min_split_gain: f64,
    ranking: RankingContext<'_>,
    credibility: &CredibilityFloor,
    exempt_empty_cells: bool,
) -> Result<Option<Candidate>, PbError> {
    Ok(best_level_split_scored(
        hist,
        axes,
        n_data_bins,
        lambda,
        l1_leaf,
        max_delta_step,
        min_split_gain,
        None,
        ranking,
        credibility,
        exempt_empty_cells,
    )?
    .map(|scored| scored.candidate))
}

#[allow(clippy::too_many_arguments)]
fn best_level_split_scored(
    hist: &Hist,
    axes: &[u32],
    n_data_bins: &[usize],
    lambda: f64,
    l1_leaf: f64,
    max_delta_step: Option<f64>,
    min_split_gain: f64,
    admission_gain_floor: Option<f64>,
    ranking: RankingContext<'_>,
    credibility: &CredibilityFloor,
    exempt_empty_cells: bool,
) -> Result<Option<ScoredCandidate>, PbError> {
    let nl = hist.n_leaves;
    let mut best: Option<ScoredCandidate> = None;
    // §07.3 step 2: per-cell credibility floors are a HARD reject. Only track per-cell
    // count/Σw support (alongside the always-present Σh) when a floor can actually bind,
    // so a default (inert) floor leaves the selected candidate byte-identical.
    let check_cred = !credibility.rejects_nothing();
    let min_data = u64::from(credibility.min_data_in_leaf);
    let min_hess = f64::from(credibility.min_sum_hessian_in_leaf);
    let min_wsum = f64::from(credibility.min_weight_sum_in_leaf);

    // Monotone-scan scratch, hoisted out of the innermost per-candidate loop below (perf
    // finding: two Vecs used to get heap-allocated per candidate, on every axis, whenever
    // ANY axis anywhere was monotone-constrained). Sized to the oblivious-tree depth cap
    // (<=3 levels, <=8 leaves). `scan.chosen` is invariant for the whole call, so it is
    // copied into the ancestor slots ONCE here; each candidate iteration below only
    // overwrites its own level's slot and the full `values` buffer, exactly as the old
    // per-candidate Vecs did -- every slot a read touches is freshly written on the SAME
    // iteration, so reuse cannot leak a stale value from a prior candidate.
    let mut monotone_signs: [Option<MonoSign>; MAX_DEPTH] = [None; MAX_DEPTH];
    let mut monotone_values: [f64; MAX_LEAVES] = [0.0; MAX_LEAVES];
    // The ancestor levels' actual `(axis, bin_le, missing_left)`, feeding
    // `monotone_reachable` so a same-axis nested-threshold candidate can exclude its
    // logically-unreachable cousins from both the veto below and the POCS clamp
    // (`clamp_monotone`, which computes its own copy from the committed tree). Also
    // invariant for the whole call; only the candidate's own slot is overwritten per
    // iteration.
    let mut monotone_levels: [(u32, u8, bool); MAX_DEPTH] = [(0, 0, false); MAX_DEPTH];
    if let Some(scan) = ranking.monotone {
        let ancestor = scan
            .chosen
            .get(..scan.level)
            .ok_or_else(internal("monotone ancestor signs"))?;
        monotone_signs
            .get_mut(..scan.level)
            .ok_or_else(internal("monotone signs capacity"))?
            .copy_from_slice(ancestor);
        for (slot, split) in monotone_levels
            .get_mut(..scan.level)
            .ok_or_else(internal("monotone levels capacity"))?
            .iter_mut()
            .zip(scan.ancestor_splits)
        {
            *slot = (split.axis, split.bin_le, split.missing_left);
        }
    }

    // B8 — hoist the reachability mask out of the per-candidate loop.
    //
    // `monotone_reachable` is O(2^d * d^2): ~72 ops at depth 3 but ~2.3k at depth 6, and it
    // used to run once per CANDIDATE BIN per axis. It is exactly hoistable for every
    // candidate whose axis does NOT already appear among the ancestor levels: such a
    // candidate is the only level testing its axis, so its group size is 1 and it can never
    // create the same-axis interval contradiction this function detects. The mask then
    // depends only on the ancestors, and extends to depth `scan.level + 1` as
    // `mask[idx] = ancestor_mask[idx & ((1 << scan.level) - 1)]` — the candidate's own bit
    // simply doesn't participate. A candidate that REUSES an ancestor axis (at most one
    // axis per level, and the whole point of the lift) still recomputes per candidate,
    // because there the interval intersection genuinely depends on its `bin_le`.
    let ancestor_reachable: Option<[bool; MAX_LEAVES]> = match ranking.monotone {
        Some(scan) => Some(monotone_reachable(
            monotone_levels
                .get(..scan.level)
                .ok_or_else(internal("monotone ancestor levels"))?,
        )?),
        None => None,
    };

    for p in 0..hist.n_axes {
        let axis = *axes.get(p).ok_or_else(internal("axis index"))?;
        let ndb = *n_data_bins
            .get(p)
            .ok_or_else(internal("n_data_bins index"))?;
        if ndb < 2 {
            continue; // need >=2 data bins for a non-trivial split
        }
        // Does a candidate on THIS axis reuse a level already on the tree? Invariant for
        // the whole axis, so it is resolved once here rather than per candidate bin (B8).
        let axis_reuses_ancestor = match ranking.monotone {
            Some(scan) => monotone_levels
                .get(..scan.level)
                .ok_or_else(internal("monotone ancestor levels for reuse check"))?
                .iter()
                .any(|&(a, _, _)| a == axis),
            None => false,
        };

        // Per-leaf totals (all bins) and the missing-bin (bin 0) mass.
        let mut total_g = smallvec::SmallVec::<[f64; MAX_SCAN_LEAVES]>::from_elem(0.0_f64, nl);
        let mut total_h = smallvec::SmallVec::<[f64; MAX_SCAN_LEAVES]>::from_elem(0.0_f64, nl);
        let mut miss_g = smallvec::SmallVec::<[f64; MAX_SCAN_LEAVES]>::from_elem(0.0_f64, nl);
        let mut miss_h = smallvec::SmallVec::<[f64; MAX_SCAN_LEAVES]>::from_elem(0.0_f64, nl);
        // Per-leaf count/Σw totals + missing mass for the credibility floor (only filled
        // when a floor can bind, so the inert path keeps its exact arithmetic).
        let mut total_c = smallvec::SmallVec::<[u64; MAX_SCAN_LEAVES]>::from_elem(0u64, nl);
        let mut total_w = smallvec::SmallVec::<[f64; MAX_SCAN_LEAVES]>::from_elem(0.0_f64, nl);
        let mut miss_c = smallvec::SmallVec::<[u64; MAX_SCAN_LEAVES]>::from_elem(0u64, nl);
        let mut miss_w = smallvec::SmallVec::<[f64; MAX_SCAN_LEAVES]>::from_elem(0.0_f64, nl);
        for leaf in 0..nl {
            let mut tg = 0.0_f64;
            let mut th = 0.0_f64;
            let mut tc = 0u64;
            let mut tw = 0.0_f64;
            for b in 0..hist.n_bins {
                let o = hist
                    .offset(leaf, p, b)
                    .ok_or_else(internal("scan offset"))?;
                tg += *hist.g.get(o).ok_or_else(internal("scan g"))?;
                th += *hist.h.get(o).ok_or_else(internal("scan h"))?;
                if check_cred {
                    tc += u64::from(*hist.count.get(o).ok_or_else(internal("scan count"))?);
                    tw += *hist.wsum.get(o).ok_or_else(internal("scan wsum"))?;
                }
            }
            let leaf_g = total_g.get_mut(leaf).ok_or_else(internal("total_g"))?;
            *leaf_g = tg;
            let leaf_h = total_h.get_mut(leaf).ok_or_else(internal("total_h"))?;
            *leaf_h = th;
            let o0 = hist
                .offset(leaf, p, 0)
                .ok_or_else(internal("miss offset"))?;
            *miss_g.get_mut(leaf).ok_or_else(internal("miss_g"))? =
                *hist.g.get(o0).ok_or_else(internal("miss g"))?;
            *miss_h.get_mut(leaf).ok_or_else(internal("miss_h"))? =
                *hist.h.get(o0).ok_or_else(internal("miss h"))?;
            if check_cred {
                *total_c.get_mut(leaf).ok_or_else(internal("total_c"))? = tc;
                *total_w.get_mut(leaf).ok_or_else(internal("total_w"))? = tw;
                *miss_c.get_mut(leaf).ok_or_else(internal("miss_c"))? =
                    u64::from(*hist.count.get(o0).ok_or_else(internal("miss count"))?);
                *miss_w.get_mut(leaf).ok_or_else(internal("miss_w"))? =
                    *hist.wsum.get(o0).ok_or_else(internal("miss wsum"))?;
            }
        }
        let parent: f64 = (0..nl)
            .map(|l| {
                newton_term(
                    *total_g.get(l).unwrap_or(&0.0),
                    *total_h.get(l).unwrap_or(&0.0),
                    lambda,
                    l1_leaf,
                    max_delta_step,
                )
            })
            .sum();

        // Prefix the data bins 1..=v as v advances; evaluate both missing directions.
        let mut data_l_g = smallvec::SmallVec::<[f64; MAX_SCAN_LEAVES]>::from_elem(0.0_f64, nl);
        let mut data_l_h = smallvec::SmallVec::<[f64; MAX_SCAN_LEAVES]>::from_elem(0.0_f64, nl);
        let mut data_l_c = smallvec::SmallVec::<[u64; MAX_SCAN_LEAVES]>::from_elem(0u64, nl);
        let mut data_l_w = smallvec::SmallVec::<[f64; MAX_SCAN_LEAVES]>::from_elem(0.0_f64, nl);
        for v in 1..ndb {
            for leaf in 0..nl {
                let o = hist
                    .offset(leaf, p, v)
                    .ok_or_else(internal("prefix offset"))?;
                *data_l_g.get_mut(leaf).ok_or_else(internal("data_l_g"))? +=
                    *hist.g.get(o).ok_or_else(internal("prefix g"))?;
                *data_l_h.get_mut(leaf).ok_or_else(internal("data_l_h"))? +=
                    *hist.h.get(o).ok_or_else(internal("prefix h"))?;
                if check_cred {
                    *data_l_c.get_mut(leaf).ok_or_else(internal("data_l_c"))? +=
                        u64::from(*hist.count.get(o).ok_or_else(internal("prefix count"))?);
                    *data_l_w.get_mut(leaf).ok_or_else(internal("data_l_w"))? +=
                        *hist.wsum.get(o).ok_or_else(internal("prefix wsum"))?;
                }
            }
            // NOTE: this is the AGGREGATE dual of the canonical `low_bit` rule —
            // `ml=true` routes the missing bin into the left sum exactly as
            // `low_bit(0, _, true) = true` routes a missing row left. The two
            // encodings (this set-partition over histogram bins vs. the per-row
            // `low_bit`) must stay in agreement; the grow→lookup round-trip proptest
            // guards that. A change to the routing rule must touch both.
            for &ml in &[false, true] {
                let mut acc = 0.0_f64;
                // Credibility (§07.3 step 2): a candidate is rejected if ANY of its child
                // cells across ALL current leaves falls under a floor — the symmetric
                // whole-level credibility guarantee.
                //
                // EXCEPT (when `exempt_empty_cells`) a child cell holding ZERO rows. A
                // floor exists to stop a THIN cell from being estimated and displayed; a
                // cell with no rows is not an estimate at all, it is the structural
                // artifact of a refinement.
                //
                // Under the depth lift this stops being a corner case and becomes the
                // dominant one. Refining an ALREADY-SPLIT axis at a nested threshold is
                // exactly the operation that sends every row of some leaf to one side —
                // that is what "refine" means — so the other child is empty by
                // construction. A depth-6 (2,2,2) tree realizes 27 of its 64 cells; the
                // other 37 are phantoms. Without this exemption ANY non-zero floor vetoes
                // EVERY candidate at the first level that creates one, and the lifted fit
                // terminates EARLIER than the depth-3 fit it was supposed to refine —
                // measured, and strictly worse than not lifting at all. That interaction
                // is what makes the memo's "the credibility floor makes the lift
                // self-throttling" too optimistic: unexempted, it does not throttle, it
                // stops.
                //
                // Same reasoning as the §07.5 fix that excludes logically-unreachable
                // cousins from the monotone veto. Scoped to `max_depth > 3` so the default
                // path is byte-for-byte unchanged (verified on the fingerprint battery).
                let mut credible = true;
                for leaf in 0..nl {
                    let dlg = *data_l_g.get(leaf).ok_or_else(internal("dlg"))?;
                    let dlh = *data_l_h.get(leaf).ok_or_else(internal("dlh"))?;
                    let (lg, lh) = if ml {
                        (
                            dlg + *miss_g.get(leaf).ok_or_else(internal("mg"))?,
                            dlh + *miss_h.get(leaf).ok_or_else(internal("mh"))?,
                        )
                    } else {
                        (dlg, dlh)
                    };
                    let tg = *total_g.get(leaf).ok_or_else(internal("tg"))?;
                    let th = *total_h.get(leaf).ok_or_else(internal("th"))?;
                    acc += newton_term(lg, lh, lambda, l1_leaf, max_delta_step)
                        + newton_term(tg - lg, th - lh, lambda, l1_leaf, max_delta_step);
                    if check_cred && credible {
                        let dlc = *data_l_c.get(leaf).ok_or_else(internal("dlc"))?;
                        let dlw = *data_l_w.get(leaf).ok_or_else(internal("dlw"))?;
                        let (cl, wl) = if ml {
                            (
                                dlc + *miss_c.get(leaf).ok_or_else(internal("mc"))?,
                                dlw + *miss_w.get(leaf).ok_or_else(internal("mw"))?,
                            )
                        } else {
                            (dlc, dlw)
                        };
                        let tc = *total_c.get(leaf).ok_or_else(internal("tc cred"))?;
                        let tw = *total_w.get(leaf).ok_or_else(internal("tw cred"))?;
                        // h_low = lh, h_high = th − lh (the two child cells of this leaf).
                        let ch = tc.saturating_sub(cl);
                        // A child holding ZERO rows is a structural refinement artifact,
                        // not a thin estimate (see the block comment above). Scoped to
                        // lifted fits so a `max_depth <= 3` fit keeps the pre-lift veto
                        // byte-for-byte; promoting the exemption to the reachability-based
                        // form (§07.5's `monotone_reachable`) and applying it at every
                        // depth is P-D2 work, gated on the benchmark.
                        let low_thin = !(exempt_empty_cells && cl == 0)
                            && (cl < min_data || lh < min_hess || wl < min_wsum);
                        let high_thin = !(exempt_empty_cells && ch == 0)
                            && (ch < min_data || th - lh < min_hess || tw - wl < min_wsum);
                        if low_thin || high_thin {
                            credible = false;
                        }
                    }
                }
                let gain = 0.5 * (acc - parent);
                if !gain.is_finite() || gain <= min_split_gain {
                    continue;
                }
                if admission_gain_floor.is_some_and(|floor| gain < floor) {
                    continue;
                }
                if check_cred && !credible {
                    continue; // hard credibility reject (§07.3 step 2)
                }
                if let Some(scan) = ranking.monotone {
                    *monotone_signs
                        .get_mut(scan.level)
                        .ok_or_else(internal("monotone signs candidate slot"))? = *scan
                        .candidate_axis_signs
                        .get(p)
                        .ok_or_else(internal("monotone candidate sign"))?;
                    let candidate_bin_le = u8::try_from(v).map_err(|_| PbError::Internal {
                        what: "monotone candidate bin_le exceeded u8".into(),
                    })?;
                    *monotone_levels
                        .get_mut(scan.level)
                        .ok_or_else(internal("monotone levels candidate slot"))? =
                        (axis, candidate_bin_le, ml);
                    let depth = scan.level + 1;
                    let n_leaves_scan = 1usize << depth;
                    for leaf in 0..nl {
                        let dlg = *data_l_g.get(leaf).ok_or_else(internal("dlg"))?;
                        let dlh = *data_l_h.get(leaf).ok_or_else(internal("dlh"))?;
                        let (lg, lh) = if ml {
                            (
                                dlg + *miss_g.get(leaf).ok_or_else(internal("mg"))?,
                                dlh + *miss_h.get(leaf).ok_or_else(internal("mh"))?,
                            )
                        } else {
                            (dlg, dlh)
                        };
                        let tg = *total_g.get(leaf).ok_or_else(internal("tg"))?;
                        let th = *total_h.get(leaf).ok_or_else(internal("th"))?;
                        let high = leaf;
                        let low = leaf | (1usize << scan.level);
                        *monotone_values
                            .get_mut(low)
                            .ok_or_else(internal("monotone low value"))? =
                            newton_leaf(lg, lh, lambda, scan.l1_leaf, scan.lr, scan.max_delta_step);
                        *monotone_values
                            .get_mut(high)
                            .ok_or_else(internal("monotone high value"))? = newton_leaf(
                            tg - lg,
                            th - lh,
                            lambda,
                            scan.l1_leaf,
                            scan.lr,
                            scan.max_delta_step,
                        );
                    }
                    let signs_slice = monotone_signs
                        .get(..depth)
                        .ok_or_else(internal("monotone signs slice"))?;
                    let values_slice = monotone_values
                        .get(..n_leaves_scan)
                        .ok_or_else(internal("monotone values slice"))?;
                    // See the B8 note above `ancestor_reachable`: recompute only when this
                    // candidate reuses one of the ancestor axes.
                    let reachable = if axis_reuses_ancestor {
                        let levels_slice = monotone_levels
                            .get(..depth)
                            .ok_or_else(internal("monotone levels slice"))?;
                        monotone_reachable(levels_slice)?
                    } else {
                        let anc = ancestor_reachable
                            .as_ref()
                            .ok_or_else(internal("hoisted ancestor reachability"))?;
                        let ancestor_mask = (1usize << scan.level) - 1;
                        let mut out = [true; MAX_LEAVES];
                        for (idx, slot) in out.iter_mut().enumerate().take(n_leaves_scan) {
                            *slot = *anc
                                .get(idx & ancestor_mask)
                                .ok_or_else(internal("hoisted reachability index"))?;
                        }
                        out
                    };
                    let reachable_slice = reachable
                        .get(..n_leaves_scan)
                        .ok_or_else(internal("monotone reachable slice"))?;
                    if !candidate_monotone_ok(values_slice, depth, signs_slice, reachable_slice)? {
                        continue;
                    }
                }
                // Strict `>` ⇒ the first candidate (lowest axis/bin_le, ml=false) wins
                // ties ⇒ deterministic argmax.
                let penalty = match ranking.table_penalties {
                    Some(penalties) => {
                        let (existing, new_border) = *penalties
                            .get(p)
                            .ok_or_else(internal("ranking penalty index"))?;
                        // Free when this candidate lands on a border the fit already
                        // realized on this raw feature (§6.4). The two halves are equal
                        // unless the fit is lifted, so this is inert by default.
                        match ranking.realized_bins {
                            Some(bins) => {
                                let bits =
                                    bins.get(p).ok_or_else(internal("realized-bin index"))?;
                                let bin = u8::try_from(v).map_err(|_| PbError::Internal {
                                    what: "candidate bin_le exceeded u8 in penalty lookup".into(),
                                })?;
                                if realized_bit(bits, bin) {
                                    existing
                                } else {
                                    new_border
                                }
                            }
                            None => existing,
                        }
                    }
                    None => 1.0,
                };
                let score = gain.max(0.0) * penalty
                    + match ranking.noise {
                        Some(split_noise) => split_noise.adjustment(axis, v, ml)?,
                        None => 0.0,
                    };
                let improves = match best {
                    Some(best) => score > best.score,
                    None => true,
                };
                if improves {
                    best = Some(ScoredCandidate {
                        candidate: Candidate {
                            axis,
                            bin_le: u8::try_from(v).map_err(|_| PbError::Internal {
                                what: "bin_le exceeded u8".into(),
                            })?,
                            missing_left: ml,
                            gain,
                        },
                        score,
                    });
                }
            }
        }
    }
    Ok(best)
}

/// Exact Newton leaf values from FULL-PRECISION sums (spec §06.4): `w* = −G/(H+λ)`,
/// scaled by `lr`. `leaf_of_row[r] ∈ 0..2^depth` is row `r`'s leaf; the unused tail
/// `leaves[2^depth..]` stays `0.0`. Sequential f64 fold ⇒ thread-count independent.
///
/// # Errors
/// [`PbError::Internal`] if a row's leaf id is out of range or an index escapes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn leaf_values(
    gh: &GradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    depth: usize,
    lambda: f64,
    l1_leaf: f64,
    lr: f64,
    max_delta_step: Option<f64>,
) -> Result<Vec<f32>, PbError> {
    // `depth == 0` (a single constant leaf) is a legal internal shape for this helper —
    // only the upper cap is structural. I1's `depth >= 1` is enforced at tree construction.
    if depth > MAX_DEPTH {
        return Err(PbError::Internal {
            what: format!("leaf_values depth {depth} exceeds the cap {MAX_DEPTH}"),
        });
    }
    let n_leaves = 1usize << depth;
    let mut g = vec![0.0_f64; n_leaves];
    let mut h = vec![0.0_f64; n_leaves];
    for &r in rows {
        let ru = r as usize;
        let leaf = usize::from(*leaf_of_row.get(ru).ok_or_else(internal("leaf map"))?);
        if leaf >= n_leaves {
            return Err(PbError::Internal {
                what: "leaf id out of range in leaf_values".into(),
            });
        }
        *g.get_mut(leaf).ok_or_else(internal("leaf g"))? +=
            f64::from(*gh.g.get(ru).ok_or_else(internal("gh.g"))?);
        *h.get_mut(leaf).ok_or_else(internal("leaf h"))? +=
            f64::from(*gh.h.get(ru).ok_or_else(internal("gh.h"))?);
    }
    // `leaf_slots`, not `n_leaves`: a depth-<3 tree keeps the legacy zeroed tail so its
    // wire bytes are unchanged (see `engine::leaf_slots`).
    let mut leaves = vec![0.0_f32; leaf_slots(depth)];
    for j in 0..n_leaves {
        let gj = soft_threshold(*g.get(j).ok_or_else(internal("g[j]"))?, l1_leaf);
        let hj = *h.get(j).ok_or_else(internal("h[j]"))?;
        let denom = hj + lambda;
        let w = if denom > 0.0 { -gj / denom } else { 0.0 };
        // §05.6 max_delta_step: clamp |w*| ≤ δ on the FULL-PRECISION aggregated step
        // (before lr), so the cap never perturbs the future quantized histogram.
        let w = match max_delta_step {
            Some(d) => w.clamp(-d, d),
            None => w,
        };
        let value = lr * w;
        if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
            return Err(PbError::InvalidInput {
                what: "Newton leaf value is not finite/representable as f32".into(),
            });
        }
        *leaves.get_mut(j).ok_or_else(internal("leaf slot"))? = value as f32;
    }
    Ok(leaves)
}

/// Per-leaf full-precision `(Σg, Σh, count)` aggregates over `rows` — the inputs to
/// `path_smooth`'s per-node Newton outputs. Mirrors the `leaf_values` fold but also keeps
/// the row counts. Sequential f64 fold ⇒ thread-count independent.
///
/// # Errors
/// [`PbError::Internal`] on an out-of-range leaf id or row index.
#[allow(clippy::type_complexity)]
fn leaf_aggregates(
    gh: &GradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    depth: usize,
) -> Result<([f64; MAX_LEAVES], [f64; MAX_LEAVES], [u64; MAX_LEAVES]), PbError> {
    // B3 (P-D0): was `1usize << depth.min(3)`, which at depth 4+ silently aggregated only
    // the low 3 bits of the leaf id into 8 slots. Fail closed instead.
    if depth > MAX_DEPTH {
        return Err(PbError::Internal {
            what: format!("leaf_aggregates depth {depth} exceeds the cap {MAX_DEPTH}"),
        });
    }
    let n_leaves = 1usize << depth;
    let mut g = [0.0_f64; MAX_LEAVES];
    let mut h = [0.0_f64; MAX_LEAVES];
    let mut c = [0u64; MAX_LEAVES];
    for &r in rows {
        let ru = r as usize;
        let leaf = usize::from(*leaf_of_row.get(ru).ok_or_else(internal("ps leaf map"))?);
        if leaf >= n_leaves {
            return Err(PbError::Internal {
                what: "leaf id out of range in leaf_aggregates".into(),
            });
        }
        *g.get_mut(leaf).ok_or_else(internal("ps g"))? +=
            f64::from(*gh.g.get(ru).ok_or_else(internal("ps gh.g"))?);
        *h.get_mut(leaf).ok_or_else(internal("ps h"))? +=
            f64::from(*gh.h.get(ru).ok_or_else(internal("ps gh.h"))?);
        *c.get_mut(leaf).ok_or_else(internal("ps c"))? += 1;
    }
    Ok((g, h, c))
}

/// Shrink each leaf toward its oblivious-tree parent node (§07.6 `path_smooth`):
/// `s[node] = (v[node]·n + s[parent]·ps) / (n + ps)`, recursively from the root (whose
/// parent contributes `s = 0`). Internal-node values `v` are the raw lr-scaled Newton step
/// on that node's rows; the depth-level (leaf) `v` are the (already monotone-clamped)
/// `leaves` passed in, so smoothing composes with the clamp. A node at level `L` is
/// identified by the first `L` split bits, and its parent drops the bit added at level
/// `L-1`. No-op when `ps <= 0`. VALUE-LEVEL on a fixed structure — depth, the ≤3-feature
/// support, and I2 are untouched.
///
/// `evidence` picks what `n` measures (§07.6 evidence generalization, `CredibilityEvidence`):
/// `Count` reads the exact node row count (the ONLY behavior before this generalization);
/// `Hessian` reads the node's Σh instead (already folded in this same loop from `leaf_h`, so
/// this costs no new per-row work) — Poisson/Tweedie's expected claim mass, which correctly
/// flags a zero-inflated cell as low-evidence in a way raw count cannot. The dispatch is
/// evaluated only when `path_smooth > 0` (the early return above), so `evidence` is never even
/// read on the default (inert) path.
///
/// # Errors
/// [`PbError::Internal`] on an index escape; [`PbError::InvalidInput`] if a smoothed leaf
/// is not finite/representable as `f32`.
#[allow(clippy::too_many_arguments)]
fn apply_path_smooth(
    leaves: &mut [f32],
    leaf_g: &[f64],
    leaf_h: &[f64],
    leaf_count: &[u64],
    depth: usize,
    lambda: f64,
    l1_leaf: f64,
    lr: f64,
    max_delta_step: Option<f64>,
    path_smooth: f64,
    evidence: CredibilityEvidence,
) -> Result<(), PbError> {
    if !(path_smooth.is_finite() && path_smooth > 0.0) {
        return Ok(());
    }
    // B3 (P-D0): was `depth.min(3)`, which at depth 4+ silently smoothed only the low 3
    // levels and left the deeper leaves at their raw Newton step. Fail closed instead.
    let n_leaves = 1usize << depth;
    if depth > MAX_DEPTH
        || leaves.len() < n_leaves
        || leaf_g.len() < n_leaves
        || leaf_h.len() < n_leaves
        || leaf_count.len() < n_leaves
    {
        return Err(PbError::Internal {
            what: format!("apply_path_smooth shape: depth {depth} (cap {MAX_DEPTH})"),
        });
    }
    // §07.6 evidence normalization (2026-07-23 amendment): the mean per-row hessian over ALL of
    // this tree's rows ("the root's aggregate", per-tree not a global-once constant) — used by
    // `credibility_evidence_n` to convert `Hessian` evidence from a raw mass into an "effective
    // row count" on `path_smooth`'s pseudo-count scale. Self-contained: summed from the SAME
    // `leaf_h`/`leaf_count` this function already receives (both are populated over exactly this
    // tree's rows, `leaf_aggregates`), so this costs no new fold or parameter. `count_total == 0`
    // is a genuinely empty tree (never happens in production — `apply_path_smooth` is only ever
    // called after a tree with `>=1` row grew) and falls back to `Count` evidence via
    // `credibility_evidence_n`'s own `h_bar <= 0.0` guard.
    let h_bar = {
        let h_total: f64 = leaf_h.iter().take(n_leaves).sum();
        let count_total: u64 = leaf_count.iter().take(n_leaves).sum();
        if count_total > 0 {
            h_total / (count_total as f64)
        } else {
            0.0
        }
    };
    // Smoothed outputs of the previous (shallower) level, indexed by that level's node id.
    let mut s_parent: Vec<f64> = Vec::new();
    let mut parent_is_virtual = true; // level 0's parent is the virtual `s = 0` root.
    for level in 0..=depth {
        let n_nodes = 1usize << level;
        let node_mask = n_nodes - 1;
        let mut s_cur = vec![0.0_f64; n_nodes];
        for j in 0..n_nodes {
            // Aggregate this node's rows from the leaves beneath it.
            let mut g = 0.0_f64;
            let mut h = 0.0_f64;
            let mut cnt = 0u64;
            for leaf in 0..n_leaves {
                if leaf & node_mask == j {
                    g += *leaf_g.get(leaf).ok_or_else(internal("ps node g"))?;
                    h += *leaf_h.get(leaf).ok_or_else(internal("ps node h"))?;
                    cnt += *leaf_count.get(leaf).ok_or_else(internal("ps node c"))?;
                }
            }
            let v = if level == depth {
                f64::from(*leaves.get(j).ok_or_else(internal("ps leaf v"))?)
            } else {
                newton_leaf(g, h, lambda, l1_leaf, lr, max_delta_step)
            };
            let parent_s = if parent_is_virtual {
                0.0
            } else {
                *s_parent
                    .get(j & (node_mask >> 1))
                    .ok_or_else(internal("ps parent s"))?
            };
            let n = credibility_evidence_n(evidence, h, cnt, h_bar);
            *s_cur.get_mut(j).ok_or_else(internal("ps s_cur"))? =
                (v * n + parent_s * path_smooth) / (n + path_smooth);
        }
        s_parent = s_cur;
        parent_is_virtual = false;
    }
    // `s_parent` now holds the depth-level (leaf) smoothed values.
    for j in 0..n_leaves {
        let val = *s_parent.get(j).ok_or_else(internal("ps out"))?;
        if !val.is_finite() || val < f64::from(f32::MIN) || val > f64::from(f32::MAX) {
            return Err(PbError::InvalidInput {
                what: "path_smooth leaf value is not finite/representable as f32".into(),
            });
        }
        *leaves.get_mut(j).ok_or_else(internal("ps write"))? = val as f32;
    }
    Ok(())
}

/// Refit an already-chosen tree structure from full-precision gradients over `rows`.
///
/// MVS uses a sampled row set only to choose the split structure; the leaf values are
/// then recomputed on all training rows so the final model remains a standard exact
/// constant-leaf ensemble.
///
/// # Errors
/// Propagates typed shape/index errors from row routing and the Newton leaf solve.
pub(crate) fn refit_tree_leaves(
    x: &BinnedMatrix,
    gh: &GradHess,
    rows: &[u32],
    tree: &mut ObliviousTree,
    cfg: &GrowConfig<'_>,
) -> Result<(), PbError> {
    let mut leaf_of_row: Vec<u8> = Hist::try_zeroed_vec(x.n_rows as usize, "refit leaf map")?;
    for &r in rows {
        let ru = r as usize;
        let mut leaf = 0u8;
        for (level, split) in tree.splits.iter().enumerate() {
            let col = x
                .data
                .get(split.axis as usize)
                .ok_or_else(internal("refit split col"))?;
            let bin = *col.get(ru).ok_or_else(internal("refit split bin"))?;
            leaf |= u8::from(low_bit(bin, split.bin_le, split.missing_left)) << level;
        }
        *leaf_of_row
            .get_mut(ru)
            .ok_or_else(internal("refit leaf slot"))? = leaf;
    }
    tree.leaves = leaf_values(
        gh,
        rows,
        &leaf_of_row,
        usize::from(tree.depth),
        cfg.lambda,
        cfg.l1_leaf,
        cfg.lr,
        cfg.max_delta_step,
    )?;
    // Re-enforce monotonicity: the structure was chosen on the sampled subset, but these
    // leaves were just recomputed on the FULL rows and could invert a cousin pair.
    let depth = usize::from(tree.depth);
    clamp_monotone(&mut tree.leaves, &tree.splits, depth, cfg.monotone)?;
    // path_smooth applies to the FINAL (full-row) leaves too, after the clamp (§07.6).
    if cfg.credibility.path_smooth > 0.0 {
        let (lg, lh, lc) = leaf_aggregates(gh, rows, &leaf_of_row, depth)?;
        apply_path_smooth(
            &mut tree.leaves,
            &lg,
            &lh,
            &lc,
            depth,
            cfg.lambda,
            cfg.l1_leaf,
            cfg.lr,
            cfg.max_delta_step,
            f64::from(cfg.credibility.path_smooth),
            cfg.credibility_evidence,
        )?;
        clamp_monotone(&mut tree.leaves, &tree.splits, depth, cfg.monotone)?;
    }
    Ok(())
}

/// Build a level-`L` (`L >= 1`) histogram via the **subtraction trick** instead of a full build:
/// accumulate only the SMALLER of each parent leaf's two children, then derive the larger by
/// subtracting from the retained level-`(L-1)` parent histogram (`prev_hist`, columns = `prev_admissible`
/// = A_{L-1}). It visits only ~half the rows. Every row's level-`L` leaf is already fixed in
/// `leaf_of_row` (bits `0..L-1`, values in `[0, 2^L)`) by the committed earlier splits, so the sibling
/// pairing — parent `p` in `[0, 2^(L-1))`, children `{p, p + 2^(L-1)}` — is known BEFORE this level's
/// split (no circular dependency). Accuracy moves only at ~1e-11 (g/h drift; count exact; wsum exact
/// under unit weights); determinism is preserved because `small_rows` is filtered in fixed row order
/// before the (chunk-deterministic) build. Works for BOTH precisions: `qgh = Some(..)` builds the
/// smaller children via the quantized integer accumulation, `None` via the FullF64 builder; the
/// subtraction itself runs on the (dequantized) f64 parent either way.
#[allow(clippy::too_many_arguments)]
fn build_subtracted_level(
    x: &BinnedMatrix,
    gh: &GradHess,
    qgh: Option<&QuantGradHess>,
    rows: &[u32],
    leaf_of_row: &[u8],
    level: usize,
    admissible: &[u32],
    weight: &[f32],
    unit_weight: bool,
    prev_hist: &Hist,
    prev_admissible: &[u32],
) -> Result<Hist, PbError> {
    let half = 1usize << (level - 1);
    let n_leaves = 1usize << level;
    // Step A — per-(level-L)-leaf row counts via one fixed-order pass (deterministic).
    let mut child_count = vec![0u64; n_leaves];
    for &r in rows {
        let leaf = usize::from(
            *leaf_of_row
                .get(r as usize)
                .ok_or_else(internal("subtraction leaf_of_row row"))?,
        );
        let c = child_count
            .get_mut(leaf)
            .ok_or_else(internal("subtraction child_count leaf"))?;
        *c = c
            .checked_add(1)
            .ok_or_else(internal("subtraction count overflow"))?;
    }
    // Pairing: parent p split into children {p, p+half}; SMALLER = fewer rows (tie -> lower index).
    let mut pairing: Vec<(usize, usize, usize)> = Vec::with_capacity(half);
    let mut is_smaller = vec![false; n_leaves];
    for p in 0..half {
        let c0 = p;
        let c1 = p + half;
        let cnt0 = *child_count
            .get(c0)
            .ok_or_else(internal("subtraction cnt0"))?;
        let cnt1 = *child_count
            .get(c1)
            .ok_or_else(internal("subtraction cnt1"))?;
        let (sm, lg) = if cnt1 < cnt0 { (c1, c0) } else { (c0, c1) };
        pairing.push((p, sm, lg));
        *is_smaller
            .get_mut(sm)
            .ok_or_else(internal("subtraction is_smaller set"))? = true;
    }
    // Step B — build ONLY the smaller children, filtering `rows` in fixed order (chunk boundaries,
    // and therefore the fixed-order chunk reduction, stay intact ⇒ thread-count independent).
    let mut small_rows: Vec<u32> = Vec::new();
    small_rows
        .try_reserve(rows.len())
        .map_err(|_| PbError::Internal {
            what: "subtraction small_rows allocation failed".into(),
        })?;
    for &r in rows {
        let leaf = usize::from(
            *leaf_of_row
                .get(r as usize)
                .ok_or_else(internal("subtraction small leaf"))?,
        );
        if *is_smaller
            .get(leaf)
            .ok_or_else(internal("subtraction is_smaller read"))?
        {
            small_rows.push(r);
        }
    }
    // Build the smaller children with the active precision; the larger are derived by subtraction
    // below (so neither path visits the larger children's rows).
    let mut hist = match qgh {
        Some(q) => build_quantized_histogram(
            x,
            q,
            &small_rows,
            leaf_of_row,
            n_leaves,
            admissible,
            weight,
            unit_weight,
        )?,
        None => build_histogram(
            x,
            gh,
            &small_rows,
            leaf_of_row,
            n_leaves,
            admissible,
            weight,
            unit_weight,
        )?,
    };
    // Step C — axis-position map A_L -> A_{L-1} (total: A_L ⊆ A_{L-1}, append-only `used_raws`).
    let mut axis_map: Vec<usize> = Vec::with_capacity(admissible.len());
    for &a in admissible {
        let pos = prev_admissible
            .iter()
            .position(|&pa| pa == a)
            .ok_or_else(internal("subtraction axis absent from parent admissible"))?;
        axis_map.push(pos);
    }
    // Step D — fill the larger children by subtracting the smaller from the parent leaf.
    subtract_sibling_into(&mut hist, prev_hist, &pairing, &axis_map)?;
    Ok(hist)
}

fn axes_are_subset(axes: &[u32], parent: &[u32]) -> bool {
    axes.iter().all(|axis| parent.iter().any(|p| p == axis))
}

#[allow(clippy::too_many_arguments)]
fn build_level_hist_for_axes(
    x: &BinnedMatrix,
    gh: &GradHess,
    qgh: Option<&QuantGradHess>,
    rows: &[u32],
    leaf_of_row: &[u8],
    level: usize,
    admissible: &[u32],
    cfg: &GrowConfig<'_>,
    weight: &[f32],
    hist_unit_weight: bool,
    prev_hist: Option<&Hist>,
    prev_admissible: Option<&[u32]>,
) -> Result<Hist, PbError> {
    if cfg.hist_subtraction && level >= 1 {
        if let (Some(ph), Some(pa)) = (prev_hist, prev_admissible) {
            if axes_are_subset(admissible, pa) {
                return build_subtracted_level(
                    x,
                    gh,
                    qgh,
                    rows,
                    leaf_of_row,
                    level,
                    admissible,
                    weight,
                    hist_unit_weight,
                    ph,
                    pa,
                );
            }
        }
    }
    match cfg.hist_precision {
        HistPrecision::FullF64 => build_histogram(
            x,
            gh,
            rows,
            leaf_of_row,
            1usize << level,
            admissible,
            weight,
            hist_unit_weight,
        ),
        HistPrecision::QuantizedI32 => build_quantized_histogram(
            x,
            qgh.ok_or_else(internal("QHIST qgh missing"))?,
            rows,
            leaf_of_row,
            1usize << level,
            admissible,
            weight,
            hist_unit_weight,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn scan_level_axes(
    x: &BinnedMatrix,
    gh: &GradHess,
    qgh: Option<&QuantGradHess>,
    rows: &[u32],
    leaf_of_row: &[u8],
    level: usize,
    admissible: &[u32],
    cfg: &GrowConfig<'_>,
    split_signs: &[Option<MonoSign>],
    ancestor_splits: &[Split],
    used_axes: &[u32],
    weight: &[f32],
    hist_unit_weight: bool,
    prev_hist: Option<&Hist>,
    prev_admissible: Option<&[u32]>,
    admission_gain_floor: Option<f64>,
) -> Result<Option<(ScoredCandidate, Hist)>, PbError> {
    if admissible.is_empty() {
        return Ok(None);
    }
    let n_data_bins: Vec<usize> = admissible
        .iter()
        .map(|&a| {
            x.grids
                .get(a as usize)
                .map_or(0, |grid| usize::from(grid.n_bins).saturating_sub(1))
        })
        .collect();
    // §6.4 (P-D2): charge only NEW borders. A reuse split landing on a border this fit has
    // already committed adds ZERO cells to the merged grid, so it should be free; one on a
    // new border adds exactly one column. Scoped to lifted fits — see `multiplier_pair`.
    let charge_new_border = usize::from(cfg.max_depth) > LEGACY_MAX_DEPTH;
    let ranking_penalties = match cfg.table_budget_penalty {
        Some(penalty) => Some(
            admissible
                .iter()
                .map(|&axis| {
                    penalty.multiplier_pair(
                        x,
                        used_axes,
                        axis,
                        cfg.realized_extent,
                        charge_new_border,
                    )
                })
                .collect::<Result<Vec<_>, PbError>>()?,
        ),
        None => None,
    };
    let realized_bins: Option<Vec<[u64; 4]>> = match (charge_new_border, cfg.realized_extent) {
        (true, Some(acc)) => Some(
            admissible
                .iter()
                .map(|&axis| {
                    let raw = x
                        .provenance
                        .get(axis as usize)
                        .ok_or_else(internal("realized-bin axis provenance"))?
                        .raw
                        .0;
                    Ok(acc.realized_bins(raw))
                })
                .collect::<Result<Vec<_>, PbError>>()?,
        ),
        _ => None,
    };
    let hist = crate::engine::boost::prof::timed("grow.hist_build", || {
        build_level_hist_for_axes(
            x,
            gh,
            qgh,
            rows,
            leaf_of_row,
            level,
            admissible,
            cfg,
            weight,
            hist_unit_weight,
            prev_hist,
            prev_admissible,
        )
    })?;
    let candidate_axis_signs: Vec<Option<MonoSign>> = match cfg.monotone {
        Some(signs) => admissible
            .iter()
            .map(|&axis| signs.get(axis as usize).copied().flatten())
            .collect(),
        None => Vec::new(),
    };
    let monotone_scan = cfg.monotone.map(|_| MonotoneScan {
        level,
        chosen: split_signs,
        ancestor_splits,
        candidate_axis_signs: &candidate_axis_signs,
        lr: cfg.lr,
        l1_leaf: cfg.l1_leaf,
        max_delta_step: cfg.max_delta_step,
    });
    let cand = crate::engine::boost::prof::timed("grow.split_find", || {
        best_level_split_scored(
            &hist,
            admissible,
            &n_data_bins,
            cfg.lambda,
            cfg.l1_leaf,
            cfg.max_delta_step,
            cfg.min_split_gain,
            admission_gain_floor,
            RankingContext {
                monotone: monotone_scan,
                noise: SplitNoise::new(cfg.quant_seed, cfg.round, level, cfg.random_strength),
                table_penalties: ranking_penalties.as_deref(),
                realized_bins: realized_bins.as_deref(),
            },
            &cfg.credibility,
            usize::from(cfg.max_depth) > LEGACY_MAX_DEPTH,
        )
    })?;
    Ok(cand.map(|c| (c, hist)))
}

/// Grow one depth-`1..=3` oblivious tree (spec §06.2/§06.6) over `rows`, scanning the
/// candidate `axes` (already column-sampled by the caller). Returns `None` when no
/// admissible candidate clears `min_split_gain` at the first level (a degenerate
/// no-split round — the boosting loop handles it). FLAG (spec §14 P2 signature
/// refinement): returns `Option<ObliviousTree>` so the no-split case is explicit
/// rather than a forced low-gain tree.
///
/// # Errors
/// [`PbError::Internal`] on an index/shape bug; [`Invariant::FeatureBudget`] if the
/// assembled tree violates I1 (cannot happen given the guard, but checked at
/// construction).
pub(crate) fn grow_oblivious_tree_with_leaf_map(
    x: &BinnedMatrix,
    gh: &GradHess,
    rows: &[u32],
    axes: &[u32],
    cfg: &GrowConfig<'_>,
    hurdle_state: InteractionHurdleState,
    weight: &[f32],
) -> Result<Option<GrowResult>, PbError> {
    let n_rows = x.n_rows as usize;
    let mut leaf_of_row: Vec<u8> = Hist::try_zeroed_vec(n_rows, "leaf assignment")?;
    let mut splits: Vec<Split> = Vec::new();
    let mut split_signs: Vec<Option<MonoSign>> = Vec::new();
    // `used_raws` is the DISTINCT raw-feature set (I1's order cap) -> MAX_ORDER.
    // `used_axes`/`level_gains` hold one entry PER LEVEL -> MAX_DEPTH, or they spill to
    // the heap on every lifted tree (B6).
    let mut used_raws: smallvec::SmallVec<[u32; MAX_ORDER]> = smallvec::SmallVec::new();
    let mut used_axes: smallvec::SmallVec<[u32; MAX_DEPTH]> = smallvec::SmallVec::new();
    let mut level_gains: smallvec::SmallVec<[f64; MAX_DEPTH]> = smallvec::SmallVec::new();
    let order_cap = usize::from(cfg.max_order).min(MAX_ORDER);
    // Retained level-1 (FullF64) histogram + its axis ids, so level 2 can build by subtraction
    // (the histogram-subtraction trick) instead of a full pass. Captured only on the FullF64 path.
    let mut prev_hist: Option<Hist> = None;
    let mut prev_admissible: Option<Vec<u32>> = None;
    // QHIST quantizes the (tree-constant) gradients/hessians ONCE here — not once per level — since
    // `gh` does not change within a tree. Bit-identical to the prior per-level re-quantization
    // (same `gh`, seed, round ⇒ same `QuantGradHess`), just computed a third as often.
    let qgh: Option<QuantGradHess> = match cfg.hist_precision {
        HistPrecision::QuantizedI32 => Some(quantize_grad_hess(gh, cfg.quant_seed, cfg.round)?),
        HistPrecision::FullF64 => None,
    };

    // Task-B: per-cell Σw is consumed ONLY by the credibility floor (`check_cred` in
    // `best_level_split`, gated on `!credibility.rejects_nothing()`). When no floor can bind, `wsum`
    // is never read — the split gain and leaf values use g/h only — so the per-row Σw accumulation is
    // dead work. Take the cheap `unit_weight` wsum=count fill instead (O(cells), not the per-row
    // O(rows) add). BIT-IDENTICAL: nothing on this path reads `wsum`, so its value is immaterial. When
    // a floor IS active, `rejects_nothing()` is false and the exact per-row Σw path is kept verbatim.
    let hist_unit_weight = cfg.unit_weight || cfg.credibility.rejects_nothing();

    // B1: the level loop. Levels beyond `max_order` can only refine an already-chosen raw
    // feature (see `fresh_admissible` below), so a deeper tree never raises fANOVA order.
    let max_depth = usize::from(cfg.max_depth);
    if !(LEGACY_MAX_DEPTH..=MAX_DEPTH).contains(&max_depth) {
        return Err(PbError::InvalidConfig {
            what: format!("max_depth must be in {LEGACY_MAX_DEPTH}..={MAX_DEPTH}, got {max_depth}"),
        });
    }
    for level in 0..max_depth {
        let prev_hist_ref = prev_hist.as_ref();
        let prev_admissible_ref = prev_admissible.as_deref();
        let fresh_admission_floor = if level >= 1 && cfg.interaction_gain_hurdle > 0.0 {
            level_gains.first().copied().map(|g0| {
                let projected_order = used_raws.len() + 1;
                let hurdle_level = projected_order.saturating_sub(1);
                hurdle_state.admission_gain_floor(
                    cfg.interaction_gain_hurdle,
                    hurdle_level,
                    g0,
                    level_gains.last().copied(),
                )
            })
        } else {
            None
        };
        // First try axes that would add a new raw feature, subject to the distinct-support cap.
        let fresh_admissible: Vec<u32> = if used_raws.len() < order_cap {
            axes.iter()
                .copied()
                .filter(|&a| fresh_axis_is_admissible(x, a, &used_raws, cfg.groups))
                .collect()
        } else {
            Vec::new()
        };
        let fresh_candidate = if fresh_admissible.is_empty() {
            None
        } else {
            scan_level_axes(
                x,
                gh,
                qgh.as_ref(),
                rows,
                &leaf_of_row,
                level,
                &fresh_admissible,
                cfg,
                &split_signs,
                &splits,
                &used_axes,
                weight,
                hist_unit_weight,
                prev_hist_ref,
                prev_admissible_ref,
                fresh_admission_floor,
            )?
        };
        let fresh_selected = fresh_candidate.map(|(scored, hist)| (scored, hist, fresh_admissible));

        // Reused axes are the lower-order alternative. They are not just a fallback after hurdle
        // failure: when the hurdle is enabled, they compete under the same ranking score and win
        // exact ties, keeping lower-order structure unless a fresh feature is strictly better.
        let skip_reuse_scan = cfg.interaction_gain_hurdle <= 0.0 && fresh_selected.is_some();
        let reuse_admissible: Vec<u32> = if !skip_reuse_scan && !used_raws.is_empty() {
            axes.iter()
                .copied()
                .filter(|&a| reused_axis_is_admissible(x, a, &used_raws))
                .collect()
        } else {
            Vec::new()
        };
        let reuse_selected = if reuse_admissible.is_empty() {
            None
        } else if let Some((cand, hist)) = scan_level_axes(
            x,
            gh,
            qgh.as_ref(),
            rows,
            &leaf_of_row,
            level,
            &reuse_admissible,
            cfg,
            &split_signs,
            &splits,
            &used_axes,
            weight,
            hist_unit_weight,
            prev_hist_ref,
            prev_admissible_ref,
            None,
        )? {
            Some((cand, hist, reuse_admissible))
        } else {
            None
        };

        let selected = choose_level_candidate(fresh_selected, reuse_selected);

        let Some((scored, hist, admissible)) = selected else {
            break; // graceful early-termination only when neither fresh nor reused axes can split
        };
        let cand = scored.candidate;
        level_gains.push(cand.gain);

        splits.push(Split {
            axis: cand.axis,
            bin_le: cand.bin_le,
            missing_left: cand.missing_left,
        });
        split_signs.push(match cfg.monotone {
            Some(signs) => signs.get(cand.axis as usize).copied().flatten(),
            None => None,
        });
        let raw = x
            .provenance
            .get(cand.axis as usize)
            .ok_or_else(internal("split axis provenance"))?
            .raw
            .0;
        if !used_raws.contains(&raw) {
            used_raws.push(raw);
        }
        used_axes.push(cand.axis);

        // Sample→leaf update: set this level's bit using the SAME canonical low_bit.
        let col = x
            .data
            .get(cand.axis as usize)
            .ok_or_else(internal("split col"))?;
        for &r in rows {
            let ru = r as usize;
            let bin = *col.get(ru).ok_or_else(internal("split bin"))?;
            let bit = u8::from(low_bit(bin, cand.bin_le, cand.missing_left)) << level;
            *leaf_of_row.get_mut(ru).ok_or_else(internal("leaf set"))? |= bit;
        }

        // Retain this level's histogram + axis ids as the parent for the NEXT level's subtraction
        // (level L → parent for level L+1). The LAST level (`level + 1 == max_depth`) has no
        // successor, so its histogram is never retained. Both precisions retain (the dequantized
        // QHIST hist is the parent its subtracted children derive from).
        if level + 1 < max_depth {
            prev_admissible = Some(admissible.clone());
            prev_hist = Some(hist);
        }
    }

    if splits.is_empty() {
        return Ok(None);
    }
    let depth = splits.len();
    let mut leaves = leaf_values(
        gh,
        rows,
        &leaf_of_row,
        depth,
        cfg.lambda,
        cfg.l1_leaf,
        cfg.lr,
        cfg.max_delta_step,
    )?;
    // Project onto the monotone cone — a no-op when the structure was grown feasible, but
    // a guard against quantized-histogram round-off inverting a cousin pair (§07.5).
    clamp_monotone(&mut leaves, &splits, depth, cfg.monotone)?;
    // §07.6 path_smooth: shrink each leaf toward its oblivious-tree parent, AFTER the
    // monotone clamp, then re-clamp so smoothing cannot cross a monotone bound.
    if cfg.credibility.path_smooth > 0.0 {
        let (lg, lh, lc) = leaf_aggregates(gh, rows, &leaf_of_row, depth)?;
        apply_path_smooth(
            &mut leaves,
            &lg,
            &lh,
            &lc,
            depth,
            cfg.lambda,
            cfg.l1_leaf,
            cfg.lr,
            cfg.max_delta_step,
            f64::from(cfg.credibility.path_smooth),
            cfg.credibility_evidence,
        )?;
        clamp_monotone(&mut leaves, &splits, depth, cfg.monotone)?;
    }
    // `leaf_of_row[r]` (set for every r in `rows` via the SAME canonical `low_bit` used by the
    // tree walk) is the per-row leaf partition the leaf-refine line search needs — returned so it
    // can be reused instead of re-walking the tree (byte-identical, cf. `gather_memberships`).
    let first_split_gain = *level_gains
        .first()
        .ok_or_else(internal("grown tree missing first split gain"))?;
    let tree = ObliviousTree::try_new(splits, leaves, &x.provenance)?;
    Ok(Some(GrowResult {
        tree,
        leaf_of_row,
        first_split_gain,
    }))
}

/// Test-only thin wrapper returning just the grown tree. Production calls
/// [`grow_oblivious_tree_with_leaf_map`] and reuses the per-row leaf map to skip the
/// leaf-refinement tree re-walk; unit tests that only assert on tree structure use this.
#[cfg(test)]
pub(crate) fn grow_oblivious_tree(
    x: &BinnedMatrix,
    gh: &GradHess,
    rows: &[u32],
    axes: &[u32],
    cfg: &GrowConfig<'_>,
    weight: &[f32],
) -> Result<Option<ObliviousTree>, PbError> {
    Ok(grow_oblivious_tree_with_leaf_map(
        x,
        gh,
        rows,
        axes,
        cfg,
        InteractionHurdleState::default(),
        weight,
    )?
    .map(|grown| grown.tree))
}

fn axis_raw_if_splittable(x: &BinnedMatrix, axis: u32) -> Option<u32> {
    let prov = x.provenance.get(axis as usize)?;
    // Degenerate-axis pre-filter: an axis with < 2 data bins (n_bins ≤ 2) has no candidate split
    // and is unconditionally skipped in `best_level_split` (ndb < 2). Excluding it here is
    // byte-identical to building-then-skipping it, but avoids the wasted O(rows) histogram build.
    let grid = x.grids.get(axis as usize)?;
    if usize::from(grid.n_bins).saturating_sub(1) < 2 {
        return None;
    }
    Some(prov.raw.0)
}

fn fresh_axis_is_admissible(
    x: &BinnedMatrix,
    axis: u32,
    used_raws: &[u32],
    groups: Option<&[FeatureSet]>,
) -> bool {
    let Some(raw) = axis_raw_if_splittable(x, axis) else {
        return false;
    };
    if used_raws.contains(&raw) {
        return false;
    }
    match groups {
        None => true,
        Some(groups) => groups.iter().any(|group| {
            group.contains(crate::data::FeatureId(raw))
                && used_raws
                    .iter()
                    .all(|&used| group.contains(crate::data::FeatureId(used)))
        }),
    }
}

fn reused_axis_is_admissible(x: &BinnedMatrix, axis: u32, used_raws: &[u32]) -> bool {
    axis_raw_if_splittable(x, axis).is_some_and(|raw| used_raws.contains(&raw))
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::float_cmp
    )]
    use super::*;
    use crate::cat::TsEncodingId;
    use crate::data::{AxisKind, AxisProvenance, BorderGrid, FeatureId};
    use crate::engine::Hist;
    use proptest::prelude::*;

    fn matrix(cols: Vec<Vec<u8>>, n_bins_each: &[u16]) -> BinnedMatrix {
        let n_rows = u32::try_from(cols.first().map_or(0, Vec::len)).unwrap();
        let grids = n_bins_each
            .iter()
            .map(|&nb| BorderGrid {
                borders: vec![0.0; usize::from(nb).saturating_sub(2)],
                n_bins: nb,
                missing_bin: 0,
            })
            .collect();
        let provenance = (0..u32::try_from(cols.len()).unwrap())
            .map(|i| AxisProvenance {
                raw: FeatureId(i),
                kind: AxisKind::Numeric,
            })
            .collect();
        BinnedMatrix {
            data: cols,
            n_rows,
            grids,
            provenance,
        }
    }

    fn gradhess(g: &[f32], h: &[f32]) -> GradHess {
        GradHess {
            g: g.to_vec(),
            h: h.to_vec(),
        }
    }

    #[test]
    fn l1_leaf_soft_thresholds_leaf_values_and_gain() {
        let gh = gradhess(&[-0.5, 0.5], &[1.0, 1.0]);
        let rows = [0u32, 1];
        let leaf_of_row = [0u8, 1];
        let no_l1 = leaf_values(&gh, &rows, &leaf_of_row, 1, 0.0, 0.0, 1.0, None).unwrap();
        let l1 = leaf_values(&gh, &rows, &leaf_of_row, 1, 0.0, 1.0, 1.0, None).unwrap();
        assert!(no_l1[0] > 0.0);
        assert!(no_l1[1] < 0.0);
        assert_eq!(l1[0], 0.0);
        assert_eq!(l1[1], 0.0);

        let mut hist = Hist::try_zeros(1, 1, 3).unwrap();
        let left = hist.offset(0, 0, 1).unwrap();
        hist.g[left] = -0.5;
        hist.h[left] = 1.0;
        hist.count[left] = 1;
        let right = hist.offset(0, 0, 2).unwrap();
        hist.g[right] = 0.5;
        hist.h[right] = 1.0;
        hist.count[right] = 1;
        assert!(best_level_split(
            &hist,
            &[0],
            &[2],
            0.0,
            0.0,
            None,
            0.0,
            RankingContext::default(),
            &CredibilityFloor::default(),
            false,
        )
        .unwrap()
        .is_some());
        assert!(best_level_split(
            &hist,
            &[0],
            &[2],
            0.0,
            1.0,
            None,
            0.0,
            RankingContext::default(),
            &CredibilityFloor::default(),
            false,
        )
        .unwrap()
        .is_none());
    }

    /// Hand-built 1-leaf histogram (1 axis, 3 bins: missing + 2 data) with a non-zero
    /// missing mass. The Newton gain and the learned missing direction match the
    /// closed form computed by hand.
    #[test]
    fn newton_gain_and_missing_direction_match_closed_form() {
        // bin0 (missing): g=2,h=1; bin1: g=4,h=2; bin2: g=-6,h=3. λ=1.
        let mut hist = Hist::try_zeros(1, 1, 3).unwrap();
        let set = |hist: &mut Hist, bin: usize, g: f64, h: f64, c: u32| {
            let o = hist.offset(0, 0, bin).unwrap();
            hist.g[o] = g;
            hist.h[o] = h;
            hist.count[o] = c;
        };
        set(&mut hist, 0, 2.0, 1.0, 1);
        set(&mut hist, 1, 4.0, 2.0, 1);
        set(&mut hist, 2, -6.0, 3.0, 1);

        let best = best_level_split(
            &hist,
            &[0],
            &[2],
            1.0,
            0.0,
            None,
            0.0,
            RankingContext::default(),
            &CredibilityFloor::default(),
            false,
        )
        .unwrap()
        .unwrap();
        // Only candidate is v=1. total_g=0,total_h=6 ⇒ parent=0²/7=0.
        //   ml=false: L=bin1 (4,2)→16/3; R=bin2+miss (-4,4)→16/5; gain=½(16/3+16/5)=4.2667.
        //   ml=true:  L=bin1+miss (6,3)→9;  R=bin2 (-6,3)→9;      gain=½(9+9)=9.
        // ml=true wins (missing routed LEFT with bin1).
        assert_eq!(best.axis, 0);
        assert_eq!(best.bin_le, 1);
        assert!(
            best.missing_left,
            "missing should be learned LEFT for higher gain"
        );
        assert!((best.gain - 9.0).abs() < 1e-9, "gain {} != 9", best.gain);
    }

    #[test]
    fn random_strength_does_not_rewrite_raw_gain() {
        // Same closed-form fixture as above, but with ranking noise enabled. Either
        // missing direction may win; the stored gain must remain that candidate's raw
        // Newton gain rather than the noisy score.
        let mut hist = Hist::try_zeros(1, 1, 3).unwrap();
        let set = |hist: &mut Hist, bin: usize, g: f64, h: f64| {
            let o = hist.offset(0, 0, bin).unwrap();
            hist.g[o] = g;
            hist.h[o] = h;
            hist.count[o] = 1;
        };
        set(&mut hist, 0, 2.0, 1.0);
        set(&mut hist, 1, 4.0, 2.0);
        set(&mut hist, 2, -6.0, 3.0);

        let noise = SplitNoise::new(123, 7, 0, 100.0);
        let best = best_level_split(
            &hist,
            &[0],
            &[2],
            1.0,
            0.0,
            None,
            0.0,
            RankingContext {
                noise,
                ..RankingContext::default()
            },
            &CredibilityFloor::default(),
            false,
        )
        .unwrap()
        .unwrap();
        let expected_raw = if best.missing_left {
            9.0
        } else {
            0.5 * (16.0 / 3.0 + 16.0 / 5.0)
        };
        assert!(
            (best.gain - expected_raw).abs() < 1e-9,
            "gain {} != raw {expected_raw}",
            best.gain
        );
    }

    #[test]
    fn table_budget_penalty_is_soft_and_preserves_raw_gain() {
        // Two one-threshold axes. Axis 0 has larger raw Newton gain (~10) but is
        // assigned an expensive-support multiplier; axis 1 has smaller raw gain (~9)
        // and wins only after the ranking prior. The stored gain remains axis 1's
        // raw Newton gain.
        let mut hist = Hist::try_zeros(1, 2, 3).unwrap();
        let set_axis = |hist: &mut Hist, axis: usize, a: f64| {
            let l = hist.offset(0, axis, 1).unwrap();
            let r = hist.offset(0, axis, 2).unwrap();
            hist.g[l] = a;
            hist.h[l] = 1.0;
            hist.count[l] = 1;
            hist.g[r] = -a;
            hist.h[r] = 1.0;
            hist.count[r] = 1;
        };
        set_axis(&mut hist, 0, (20.0_f64).sqrt());
        set_axis(&mut hist, 1, (18.0_f64).sqrt());

        let raw_best = best_level_split(
            &hist,
            &[0, 1],
            &[2, 2],
            1.0,
            0.0,
            None,
            0.0,
            RankingContext::default(),
            &CredibilityFloor::default(),
            false,
        )
        .unwrap()
        .unwrap();
        assert_eq!(raw_best.axis, 0);
        assert!((raw_best.gain - 10.0).abs() < 1.0e-9);

        let penalized = best_level_split(
            &hist,
            &[0, 1],
            &[2, 2],
            1.0,
            0.0,
            None,
            0.0,
            RankingContext {
                table_penalties: Some(&[(0.5, 0.5), (1.0, 1.0)]),
                realized_bins: None,
                ..RankingContext::default()
            },
            &CredibilityFloor::default(),
            false,
        )
        .unwrap()
        .unwrap();
        assert_eq!(penalized.axis, 1);
        assert!((penalized.gain - 9.0).abs() < 1.0e-9);
    }

    #[test]
    fn admission_floor_filters_before_ranking_score_wins() {
        // Axis 0 would win by ranking score, but its raw gain is below the hurdle floor. Axis 1
        // has a lower ranking score but clears the floor, so the scan must return axis 1 rather
        // than returning axis 0 and letting the caller reject the entire fresh scan.
        let mut hist = Hist::try_zeros(1, 2, 3).unwrap();
        let set_axis = |hist: &mut Hist, axis: usize, target_gain: f64| {
            let a = (2.0 * target_gain).sqrt();
            let l = hist.offset(0, axis, 1).unwrap();
            let r = hist.offset(0, axis, 2).unwrap();
            hist.g[l] = a;
            hist.h[l] = 1.0;
            hist.count[l] = 1;
            hist.g[r] = -a;
            hist.h[r] = 1.0;
            hist.count[r] = 1;
        };
        set_axis(&mut hist, 0, 7.0);
        set_axis(&mut hist, 1, 8.0);

        let raw_penalized = best_level_split_scored(
            &hist,
            &[0, 1],
            &[2, 2],
            1.0,
            0.0,
            None,
            0.0,
            None,
            RankingContext {
                table_penalties: Some(&[(1.0, 1.0), (0.5, 0.5)]),
                realized_bins: None,
                ..RankingContext::default()
            },
            &CredibilityFloor::default(),
            false,
        )
        .unwrap()
        .unwrap();
        assert_eq!(raw_penalized.candidate.axis, 0);

        let admitted = best_level_split_scored(
            &hist,
            &[0, 1],
            &[2, 2],
            1.0,
            0.0,
            None,
            0.0,
            Some(7.5),
            RankingContext {
                table_penalties: Some(&[(1.0, 1.0), (0.5, 0.5)]),
                realized_bins: None,
                ..RankingContext::default()
            },
            &CredibilityFloor::default(),
            false,
        )
        .unwrap()
        .unwrap();
        assert_eq!(admitted.candidate.axis, 1);
        assert!((admitted.candidate.gain - 8.0).abs() < 1.0e-9);
    }

    #[test]
    fn reused_refinement_wins_equal_ranking_score() {
        let hist = Hist::try_zeros(1, 1, 2).unwrap();
        let make = |axis, score| {
            (
                ScoredCandidate {
                    candidate: Candidate {
                        axis,
                        bin_le: 1,
                        missing_left: false,
                        gain: score,
                    },
                    score,
                },
                hist.clone(),
                vec![axis],
            )
        };
        let chosen = choose_level_candidate(Some(make(1, 3.0)), Some(make(0, 3.0))).unwrap();
        assert_eq!(chosen.0.candidate.axis, 0);
    }

    #[test]
    fn max_delta_step_is_reflected_in_split_gain() {
        let mut hist = Hist::try_zeros(1, 2, 3).unwrap();
        let set_axis = |hist: &mut Hist, axis: usize, g: f64, h: f64| {
            let l = hist.offset(0, axis, 1).unwrap();
            let r = hist.offset(0, axis, 2).unwrap();
            hist.g[l] = -g;
            hist.h[l] = h;
            hist.count[l] = 1;
            hist.g[r] = g;
            hist.h[r] = h;
            hist.count[r] = 1;
        };
        set_axis(&mut hist, 0, 100.0, 1.0);
        set_axis(&mut hist, 1, 150.0, 1000.0);

        let unclamped = best_level_split(
            &hist,
            &[0, 1],
            &[2, 2],
            0.0,
            0.0,
            None,
            0.0,
            RankingContext::default(),
            &CredibilityFloor::default(),
            false,
        )
        .unwrap()
        .unwrap();
        assert_eq!(unclamped.axis, 0);

        let clamped = best_level_split(
            &hist,
            &[0, 1],
            &[2, 2],
            0.0,
            0.0,
            Some(0.1),
            0.0,
            RankingContext::default(),
            &CredibilityFloor::default(),
            false,
        )
        .unwrap()
        .unwrap();
        assert_eq!(clamped.axis, 1);
        assert!((clamped.gain - 20.0).abs() < 1.0e-9);
    }

    #[test]
    fn table_budget_beta_zero_is_exactly_inert() {
        assert_eq!(TableBudgetPenalty::new(0.0, 1, 1.0), None);
        let x = matrix(vec![vec![1u8, 2]], &[3]);
        let penalty = TableBudgetPenalty::new(0.5, 3, 1.0).unwrap();
        let no_over_budget = penalty.multiplier_pair(&x, &[], 0, None, false).unwrap().0;
        assert_eq!(no_over_budget, 1.0);
    }

    /// The finding: two border-rich axes (50 bins each, well under the crate's 255-bin cap
    /// but plenty to demonstrate the effect at a tractable budget) with `used_axes=[0]` (an
    /// already-chosen ancestor) and `candidate_axis=1` (a fresh third-ish support). At
    /// `budget_cells=100`, `grid.n_bins`-based projection (`50*50=2500`) is 25x over budget
    /// from the very first candidate evaluated -- REGARDLESS of how much has actually been
    /// realized -- exactly the "penalized from tree 1 onward" failure scenario (spec review
    /// 2026-07-11, L1108-1119). A `RealizedExtent` that has genuinely committed only two
    /// borders on axis 0 (across prior trees) and none yet on the fresh axis 1 projects the
    /// TRUE realized table size instead (`4*2=8`), staying at or under budget and so
    /// admitting the candidate at the full, unpenalized ranking score.
    /// §6.4 (P-D2): a reuse split landing on an ALREADY-realized border adds zero cells to
    /// the merged grid, so it must be free; one on a NEW border must be charged exactly one
    /// column. And the whole refinement must be INERT unless the fit is lifted, or it
    /// reorders depth-3 fits (a never-split axis goes from extent 2 to 3).
    #[test]
    fn table_budget_charges_only_new_borders_and_only_when_lifted() {
        let x = matrix(vec![vec![1u8; 2], vec![1u8; 2]], &[50, 50]);
        // Budget 100, beta 1 so the multiplier is exactly `budget / cells` when over.
        let penalty = TableBudgetPenalty::new(1.0, 100, 1.0).unwrap();
        let mut realized = RealizedExtent::new(2);
        // Axis 0 already carries borders {5, 20} -> extent 4. Axis 1 is untouched.
        for bin_le in [5u8, 20] {
            realized
                .record_tree(
                    &[Split {
                        axis: 0,
                        bin_le,
                        missing_left: false,
                    }],
                    &x,
                )
                .unwrap();
        }
        assert!(realized.contains(0, 5));
        assert!(!realized.contains(0, 6));
        assert!(!realized.contains(1, 5));

        // A candidate REUSING axis 0 inside a support that also uses axis 1 (extent 2):
        // existing border -> 4*2 = 8 cells, new border -> 5*2 = 10. BOTH are under the
        // 100-cell budget, so both score 1.0 — the prior is inert below budget by design,
        // and charging the new border must not change that.
        let (existing, new_border) = penalty
            .multiplier_pair(&x, &[1], 0, Some(&realized), true)
            .unwrap();
        assert_eq!((existing, new_border), (1.0, 1.0));

        // Push the support over budget so both halves are in the penalized regime and the
        // exact ratio is checkable: 60 borders on axis 1 -> extent 62; 4*62 = 248 > 100.
        let mut wide = RealizedExtent::new(2);
        for bin_le in [5u8, 20] {
            wide.record_tree(
                &[Split {
                    axis: 0,
                    bin_le,
                    missing_left: false,
                }],
                &x,
            )
            .unwrap();
        }
        for bin_le in 1u8..=60 {
            wide.record_tree(
                &[Split {
                    axis: 1,
                    bin_le,
                    missing_left: false,
                }],
                &x,
            )
            .unwrap();
        }
        let (ex, nb) = penalty
            .multiplier_pair(&x, &[1], 0, Some(&wide), true)
            .unwrap();
        // existing: 4*62 = 248 -> 100/248; new: 5*62 = 310 -> 100/310. Over budget, a NEW
        // border ranks strictly below an already-realized one — the whole point.
        assert!(nb < ex, "existing {ex}, new {nb}");
        assert!((ex - 100.0 / 248.0).abs() < 1e-12, "existing {ex}");
        assert!((nb - 100.0 / 310.0).abs() < 1e-12, "new {nb}");

        // INERT when not lifted: both halves collapse to the pre-P-D2 single value.
        let (a, b) = penalty
            .multiplier_pair(&x, &[1], 0, Some(&wide), false)
            .unwrap();
        assert_eq!(a, b);
        assert!((a - 100.0 / 248.0).abs() < 1e-12);

        // The bitset the hot loop uses must agree with `contains` on every bin.
        let bits = wide.realized_bins(0);
        for bin in 0u8..=255 {
            assert_eq!(realized_bit(&bits, bin), wide.contains(0, bin), "bin {bin}");
        }
    }

    #[test]
    fn table_budget_penalty_uses_realized_extent_not_grid_resolution() {
        let x = matrix(vec![vec![1u8; 2], vec![1u8; 2]], &[50, 50]);
        let penalty = TableBudgetPenalty::new(0.5, 100, 1.0).unwrap();

        // Legacy behavior (no accumulator threaded in, e.g. multiclass / any caller not yet
        // wired to a live round loop): heavily over-budget from grid resolution alone.
        let legacy = penalty.multiplier_pair(&x, &[0], 1, None, false).unwrap().0;
        assert!(
            legacy < 0.25,
            "grid-resolution-based multiplier should be heavily penalized, got {legacy}"
        );

        // Fixed behavior: an accumulator reflecting two REALIZED borders on axis 0 (from
        // committed prior trees) and a never-split candidate axis 1.
        let mut realized = RealizedExtent::new(2);
        realized
            .record_tree(
                &[
                    Split {
                        axis: 0,
                        bin_le: 5,
                        missing_left: false,
                    },
                    Split {
                        axis: 0,
                        bin_le: 20,
                        missing_left: false,
                    },
                ],
                &x,
            )
            .unwrap();
        let fixed = penalty
            .multiplier_pair(&x, &[0], 1, Some(&realized), false)
            .unwrap()
            .0;
        assert_eq!(
            fixed, 1.0,
            "realized-extent-based multiplier must admit a support that is not actually \
             over budget yet"
        );

        // The accumulator is genuinely cumulative: recording more trees' borders on the
        // candidate axis grows ITS extent too (not reset), so a later, truly border-rich
        // support (many trees in) is correctly penalized once the realized table really is
        // that large -- this is not a blanket "always admit" bypass. 30 distinct borders on
        // axis 1 (extent 32) against axis 0's fixed extent 4 projects `4*32=128` cells,
        // over the 100-cell budget.
        for bin_le in 1u8..=30 {
            realized
                .record_tree(
                    &[Split {
                        axis: 1,
                        bin_le,
                        missing_left: false,
                    }],
                    &x,
                )
                .unwrap();
        }
        let grown = penalty
            .multiplier_pair(&x, &[0], 1, Some(&realized), false)
            .unwrap()
            .0;
        assert!(
            grown < fixed,
            "penalty must reassert itself once axis 1's realized extent actually grows, got {grown}"
        );
    }

    #[test]
    fn adaptive_interaction_hurdle_is_order_and_maturity_aware() {
        let fixed = InteractionHurdleState::new(InteractionGainHurdleMode::Fixed, Some(100.0));
        assert_eq!(fixed.effective_hurdle(5.0, 1, 100.0), 5.0);
        assert_eq!(fixed.effective_hurdle(5.0, 2, 100.0), 5.0);

        let adaptive =
            InteractionHurdleState::new(InteractionGainHurdleMode::Adaptive, Some(100.0));
        assert_eq!(adaptive.effective_hurdle(5.0, 1, 100.0), 5.0);
        assert_eq!(adaptive.effective_hurdle(5.0, 2, 100.0), 10.0);
        assert!((adaptive.effective_hurdle(5.0, 1, 10.0) - 2.4358541225631423).abs() < 1.0e-12);
        assert!((adaptive.effective_hurdle(5.0, 2, 10.0) - 4.8717082451262845).abs() < 1.0e-12);
        assert_eq!(adaptive.effective_hurdle(0.0, 2, 100.0), 0.0);

        // The ORDER lift. `level` here is the projected order minus one, so this argument
        // is 3 for a 3->4 transition. The multiplier was `if level >= 2 { 2.0 }`, which is
        // FLAT past order 3 — it would have charged the fourth distinct feature exactly
        // what the third cost. It is now `2^(order-2)`: order 2 pays 1x, order 3 pays 2x,
        // order 4 pays 4x, which is what makes the hurdle DISCRIMINATE at order 4 rather
        // than merely tax it (measured: at the product default a pairwise-only target
        // admits zero order-4 trees; a genuine 4-way target admits them readily).
        assert_eq!(adaptive.effective_hurdle(5.0, 3, 100.0), 20.0);
        // ... and the orders a pre-lift fit could project are UNCHANGED, which is what
        // keeps every order-<=3 fit byte-identical.
        assert_eq!(adaptive.effective_hurdle(5.0, 0, 100.0), 5.0);
        assert_eq!(adaptive.effective_hurdle(5.0, 1, 100.0), 5.0);
        assert_eq!(adaptive.effective_hurdle(5.0, 2, 100.0), 10.0);
    }

    /// The `table_budget_order_shrink` half of the order price: an order-4 support is
    /// measured against a SHRUNK cell allowance, and the shrink is exactly inert below
    /// order 4 (its exponent is `order - LEGACY_MAX_ORDER`).
    #[test]
    fn the_table_budget_allowance_shrinks_only_above_the_legacy_order() {
        let p = TableBudgetPenalty::new(0.5, 4096, 2.0).unwrap();
        assert_eq!(p.budget_for_order(1), 4096);
        assert_eq!(p.budget_for_order(2), 4096);
        assert_eq!(p.budget_for_order(LEGACY_MAX_ORDER), 4096);
        assert_eq!(p.budget_for_order(LEGACY_MAX_ORDER + 1), 2048);

        // `1.0` is exactly inert at every order; a malformed shrink is treated as inert
        // rather than rejected, so a bad prior can never drive the allowance to zero and
        // flatten the gain ordering entirely.
        for shrink in [1.0, 0.0, -3.0, f64::NAN] {
            let inert = TableBudgetPenalty::new(0.5, 4096, shrink).unwrap();
            assert_eq!(
                inert.budget_for_order(LEGACY_MAX_ORDER + 1),
                4096,
                "{shrink}"
            );
        }
        // The allowance never reaches zero even under an extreme shrink.
        let steep = TableBudgetPenalty::new(0.5, 4, 1.0e12).unwrap();
        assert!(steep.budget_for_order(MAX_ORDER) >= 1);
    }

    #[test]
    fn adaptive_third_raw_floor_uses_pair_evidence() {
        let adaptive =
            InteractionHurdleState::new(InteractionGainHurdleMode::Adaptive, Some(100.0));
        let first_split_gain = 10.0;
        let parent_level_gain = 100.0;
        let base_hurdle = 2.0;
        let pressure = 0.25 + 0.75 * f64::sqrt(first_split_gain / 100.0);
        let main_floor = base_hurdle * 2.0 * pressure * first_split_gain;
        let parent_floor = base_hurdle * pressure * parent_level_gain;
        assert!(parent_floor > main_floor);
        let floor = adaptive.admission_gain_floor(
            base_hurdle,
            2,
            first_split_gain,
            Some(parent_level_gain),
        );
        assert!((floor - parent_floor).abs() < 1.0e-12);

        let fixed = InteractionHurdleState::new(InteractionGainHurdleMode::Fixed, Some(100.0));
        assert_eq!(
            fixed.admission_gain_floor(base_hurdle, 2, first_split_gain, Some(parent_level_gain)),
            base_hurdle * first_split_gain
        );
    }

    #[test]
    fn below_min_split_gain_yields_no_candidate() {
        let mut hist = Hist::try_zeros(1, 1, 3).unwrap();
        let o = hist.offset(0, 0, 1).unwrap();
        hist.g[o] = 1.0;
        hist.h[o] = 1.0;
        // Single populated bin ⇒ any split is degenerate (gain 0); a positive floor rejects it.
        assert!(best_level_split(
            &hist,
            &[0],
            &[2],
            1.0,
            0.0,
            None,
            1.0,
            RankingContext::default(),
            &CredibilityFloor::default(),
            false,
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn leaf_values_match_per_leaf_newton_solve() {
        // 4 rows, 2 leaves. leaf0 rows 0,1 (g 1,2; h 1,1 ⇒ G=3,H=2);
        // leaf1 rows 2,3 (g -4,-1; h 1,1 ⇒ G=-5,H=2). λ=1, lr=0.1.
        let gh = gradhess(&[1.0, 2.0, -4.0, -1.0], &[1.0, 1.0, 1.0, 1.0]);
        let rows = [0u32, 1, 2, 3];
        let leaf_of_row = [0u8, 0, 1, 1];
        let leaves = leaf_values(&gh, &rows, &leaf_of_row, 1, 1.0, 0.0, 0.1, None).unwrap();
        // w*_0 = -3/(2+1) = -1 ⇒ leaf 0.1·-1 = -0.1; w*_1 = 5/3 ⇒ leaf 0.16667.
        assert!((leaves[0] - (-0.1)).abs() < 1e-6);
        assert!((leaves[1] - (5.0 / 3.0 * 0.1) as f32).abs() < 1e-6);
        // Unused tail (depth 1 ⇒ leaves 2..8) is zero.
        assert!(leaves[2..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn max_delta_step_caps_the_leaf_newton_step() {
        // A tiny hessian makes the Newton step w* = -G/(H+λ) explode; the §05.6 clamp
        // caps |w*| ≤ δ on the full-precision aggregate BEFORE the learning rate, so the
        // stored leaf is bounded by lr·δ. (This is the Poisson stability safeguard.)
        let gh = gradhess(&[-100.0], &[0.01]); // g=-100, h=0.01 ⇒ w* = 10000 uncapped
        let rows = [0u32];
        let leaf_of_row = [0u8];
        let uncapped = leaf_values(&gh, &rows, &leaf_of_row, 0, 0.0, 0.0, 0.1, None).unwrap();
        assert!(
            uncapped[0] > 100.0,
            "uncapped leaf should be huge, got {}",
            uncapped[0]
        );
        let capped = leaf_values(&gh, &rows, &leaf_of_row, 0, 0.0, 0.0, 0.1, Some(0.5)).unwrap();
        // |w*| clamped to 0.5 ⇒ leaf = 0.1·0.5 = 0.05.
        assert!(
            (capped[0] - 0.05).abs() < 1e-6,
            "clamped leaf should be 0.05, got {}",
            capped[0]
        );
    }

    #[test]
    fn unrepresentable_leaf_value_errors_instead_of_storing_inf() {
        let gh = gradhess(&[f32::MAX], &[f32::MIN_POSITIVE]);
        let rows = [0u32];
        let leaf_of_row = [0u8];
        assert!(matches!(
            leaf_values(&gh, &rows, &leaf_of_row, 0, 0.0, 0.0, 1.0, None),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn clamp_monotone_projects_inverted_leaves_onto_the_cone() {
        // depth-2, both levels Increasing. leaf idx = bit0 | bit1<<1; bit=1 ⇒ LOW feature
        // value (must have the LOWER response for Increasing). Start fully inverted.
        let splits = vec![
            Split {
                axis: 0,
                bin_le: 1,
                missing_left: false,
            },
            Split {
                axis: 1,
                bin_le: 1,
                missing_left: false,
            },
        ];
        // idx 0=(hi0,hi1) 1=(lo0,hi1) 2=(hi0,lo1) 3=(lo0,lo1): low-value leaves set HIGH.
        let mut leaves = [0.0_f32, 10.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0];
        let signs = [Some(MonoSign::Increasing), Some(MonoSign::Increasing)];
        clamp_monotone(&mut leaves, &splits, 2, Some(&signs)).unwrap();
        // Level 0 cousins (differ in bit0): w[bit0=1] <= w[bit0=0].
        assert!(leaves[1] <= leaves[0] + 1e-6);
        assert!(leaves[3] <= leaves[2] + 1e-6);
        // Level 1 cousins (differ in bit1): w[bit1=1] <= w[bit1=0].
        assert!(leaves[2] <= leaves[0] + 1e-6);
        assert!(leaves[3] <= leaves[1] + 1e-6);
        // Decreasing flips the required direction.
        let mut dec = [0.0_f32, -10.0, 0.0, -10.0, 0.0, 0.0, 0.0, 0.0];
        let dsigns = [Some(MonoSign::Decreasing), None];
        clamp_monotone(&mut dec, &splits, 2, Some(&dsigns)).unwrap();
        assert!(dec[1] >= dec[0] - 1e-6); // Decreasing: low-value leaf >= high-value leaf
        assert!(dec[3] >= dec[2] - 1e-6);
        // No signs ⇒ untouched.
        let mut none = [5.0_f32, 1.0, 9.0, 2.0, 0.0, 0.0, 0.0, 0.0];
        let before = none;
        clamp_monotone(&mut none, &splits, 2, None).unwrap();
        assert_eq!(none, before);
    }

    /// The finding's own example: level 0 tests `x>5` (bin_le=5), level 1 (a candidate
    /// reusing the SAME axis) tests `x<=2` (bin_le=2). idx bit0=level0, bit1=level1;
    /// bit=1 means "value<=bin_le" (`low_bit`, engine/mod.rs). idx=2 (bit0=0 i.e. x>5,
    /// bit1=1 i.e. x<=2) demands x>5 AND x<=2 simultaneously -- impossible for any finite
    /// bin, and no missing row reaches it either (missing_left=false at both levels routes
    /// a missing row to bit=0 at BOTH levels, i.e. idx=0, not idx=2). The other three
    /// combinations are all satisfiable intervals: idx=0 (x>5), idx=1 (2<x<=5), idx=3
    /// (x<=2).
    #[test]
    fn monotone_reachable_flags_only_the_same_axis_impossible_combination() {
        let levels = [(0_u32, 5_u8, false), (0_u32, 2_u8, false)];
        let mask = monotone_reachable(&levels).unwrap();
        assert_eq!(&mask[..4], &[true, true, false, true]);

        // A missing row DOES realize whichever combination matches both levels' own
        // missing_left -- here that's idx=0 (already reachable via x>5 too), but flipping
        // level 1's missing_left to route missing LOW makes idx=2 the exact missing-row
        // combination, so it becomes reachable again despite the same empty interval.
        let levels_missing_low = [(0_u32, 5_u8, false), (0_u32, 2_u8, true)];
        let mask2 = monotone_reachable(&levels_missing_low).unwrap();
        assert_eq!(&mask2[..4], &[true, true, true, true]);

        // Different axes never contradict, regardless of thresholds.
        let levels_diff_axes = [(0_u32, 5_u8, false), (1_u32, 2_u8, false)];
        let mask3 = monotone_reachable(&levels_diff_axes).unwrap();
        assert_eq!(&mask3[..4], &[true, true, true, true]);
    }

    /// (i) A refinement previously vetoed by a phantom cell is now accepted.
    /// `candidate_monotone_ok` is the exact function `best_level_split_scored`'s monotone
    /// block calls; this exercises it directly with the finding's own same-axis scenario.
    /// idx=2 is logically unreachable (see `monotone_reachable_flags_only_...` above) and
    /// its forced value is 0.0 (an empty histogram cell always Newton-solves to exactly 0),
    /// while idx=3 (a REAL, reachable cell) carries a genuine positive value: naive
    /// cousin-pairing at level 0 would demand `values[3] <= values[2]` (1.0 <= 0.0, false)
    /// and reject the whole candidate even though every REACHABLE pair is honestly
    /// monotone.
    #[test]
    fn candidate_monotone_ok_ignores_unreachable_same_axis_phantom() {
        let levels = [(0_u32, 5_u8, false), (0_u32, 2_u8, false)];
        let reachable = monotone_reachable(&levels).unwrap();
        let values = [5.0_f64, 3.0, 0.0, 1.0];
        let signs = [Some(MonoSign::Increasing), Some(MonoSign::Increasing)];
        assert!(candidate_monotone_ok(&values, 2, &signs, &reachable[..4]).unwrap());

        // Without reachability awareness (every cell treated as real), the SAME values are
        // rejected -- proving the phantom at idx=2 is what used to veto this candidate.
        let all_reachable = [true, true, true, true];
        assert!(!candidate_monotone_ok(&values, 2, &signs, &all_reachable).unwrap());
    }

    /// (ii) Clamp output over reachable cells matches the constrained projection computed
    /// by hand, and an unreachable phantom is excluded from the averaging entirely (not
    /// just left unmodified by coincidence -- its own value is never even read).
    ///
    /// Same same-axis scenario. idx0=1.0, idx1=5.0 start inverted for level 0's Increasing
    /// requirement (`values[1] <= values[0]`), so the reachable-only POCS clamp must
    /// average them to 3.0/3.0. idx3=2.0 is already <= the post-average idx1=3.0 (level 1's
    /// requirement), so it must be left EXACTLY at 2.0 -- the finding's failure scenario is
    /// idx3 getting spuriously halved by pooling with idx2's phantom 0.0; this pins that it
    /// no longer does. idx2 itself starts at 0.0 (what an empty cell's Newton solve always
    /// gives) and must come out untouched.
    #[test]
    fn clamp_monotone_excludes_unreachable_same_axis_phantom_from_pooling() {
        let splits = vec![
            Split {
                axis: 0,
                bin_le: 5,
                missing_left: false,
            },
            Split {
                axis: 0,
                bin_le: 2,
                missing_left: false,
            },
        ];
        let mut leaves = [1.0_f32, 5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0];
        let signs = [Some(MonoSign::Increasing), Some(MonoSign::Increasing)];
        clamp_monotone(&mut leaves, &splits, 2, Some(&signs)).unwrap();
        assert_eq!(leaves, [3.0, 3.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]);
    }

    /// A clean depth-2 fixture: two informative features ⇒ the engine grows a depth-2
    /// tree on two DISTINCT raw features (I1), with no early termination.
    fn cfg(lambda: f64, lr: f64, min_split_gain: f64, max_order: u8) -> GrowConfig<'static> {
        GrowConfig {
            lambda,
            l1_leaf: 0.0,
            lr,
            min_split_gain,
            interaction_gain_hurdle: 0.0,
            max_order,
            max_depth: LEGACY_MAX_DEPTH as u8,
            max_delta_step: None,
            hist_precision: HistPrecision::FullF64,
            quant_seed: 0,
            round: 0,
            random_strength: 0.0,
            groups: None,
            monotone: None,
            table_budget_penalty: None,
            realized_extent: None,
            credibility: CredibilityFloor::default(),
            credibility_evidence: CredibilityEvidence::Count,
            // Slow Σw path in grow tests (always correct for any weights); the dedicated
            // `unit_weight_*` hist tests pin the fast path's bit-identity directly.
            unit_weight: false,
            // Subtraction ON by default in grow tests too (matches production); the
            // equivalence tests flip it off to get the full-build reference.
            hist_subtraction: true,
        }
    }

    /// Unit weights of length `n` for `grow_oblivious_tree` calls that don't exercise
    /// the `min_weight_sum_in_leaf` floor.
    fn ones(n: usize) -> Vec<f32> {
        vec![1.0_f32; n]
    }

    /// A `GrowConfig` like `cfg(1.0, 0.1, 0.0, 3)` but carrying a credibility floor.
    fn cfg_cred(floor: CredibilityFloor) -> GrowConfig<'static> {
        GrowConfig {
            credibility: floor,
            ..cfg(1.0, 0.1, 0.0, 3)
        }
    }

    #[test]
    fn credibility_floors_reject_under_supported_children() {
        // 1 leaf, 1 axis, 3 bins. The only candidate (bin_le=1) splits into bin1 (well
        // supported) and bin2 (count 2, Σh 2, Σw 2) — a thin cell each floor can veto.
        let mut hist = Hist::try_zeros(1, 1, 3).unwrap();
        let set = |hist: &mut Hist, bin: usize, g: f64, h: f64, c: u32, w: f64| {
            let o = hist.offset(0, 0, bin).unwrap();
            hist.g[o] = g;
            hist.h[o] = h;
            hist.count[o] = c;
            hist.wsum[o] = w;
        };
        set(&mut hist, 0, 0.0, 0.0, 0, 0.0); // missing: empty
        set(&mut hist, 1, -5.0, 10.0, 10, 10.0);
        set(&mut hist, 2, 5.0, 2.0, 2, 2.0);
        let split = |floor: &CredibilityFloor| {
            best_level_split(
                &hist,
                &[0],
                &[2],
                1.0,
                0.0,
                None,
                0.0,
                RankingContext::default(),
                floor,
                false,
            )
            .unwrap()
        };
        // No floor ⇒ the informative split is found.
        assert!(split(&CredibilityFloor::default()).is_some());
        // Each floor independently vetoes the under-supported bin-2 child ⇒ no candidate.
        assert!(split(&CredibilityFloor {
            min_data_in_leaf: 5,
            ..CredibilityFloor::default()
        })
        .is_none());
        assert!(split(&CredibilityFloor {
            min_sum_hessian_in_leaf: 3.0,
            ..CredibilityFloor::default()
        })
        .is_none());
        assert!(split(&CredibilityFloor {
            min_weight_sum_in_leaf: 5.0,
            ..CredibilityFloor::default()
        })
        .is_none());
        // A floor at or below the thin cell's support still admits the split.
        assert!(split(&CredibilityFloor {
            min_data_in_leaf: 2,
            min_sum_hessian_in_leaf: 2.0,
            min_weight_sum_in_leaf: 2.0,
            ..CredibilityFloor::default()
        })
        .is_some());
    }

    /// PIN: bit-identity evidence for the monotone-scan allocation-hoisting change
    /// (perf finding, split.rs innermost-candidate-loop Vec allocations). This exact
    /// gain/bin_le/missing_left/score tuple was captured by running this test against the
    /// UNEDITED code, before the buffers in `best_level_split_scored`'s monotone block were
    /// hoisted out of the loop. If the hoisting changes so much as one bit of arithmetic,
    /// this test fails.
    #[test]
    fn monotone_scan_candidate_is_bit_identical_pin() {
        // Same hist as `credibility_floors_reject_under_supported_children`: bin1
        // (g=-5,h=10) vs bin2 (g=5,h=2) at bin_le=1. Root-level (nl=1) monotone scan,
        // Decreasing sign so the informative candidate is ACCEPTED (low_v=newton_leaf for
        // bin1's prefix >= high_v for the bin2 remainder), giving a real gain/score to pin.
        let mut hist = Hist::try_zeros(1, 1, 3).unwrap();
        let set = |hist: &mut Hist, bin: usize, g: f64, h: f64| {
            let o = hist.offset(0, 0, bin).unwrap();
            hist.g[o] = g;
            hist.h[o] = h;
        };
        set(&mut hist, 0, 0.0, 0.0);
        set(&mut hist, 1, -5.0, 10.0);
        set(&mut hist, 2, 5.0, 2.0);
        let signs = [Some(MonoSign::Decreasing)];
        let scan = MonotoneScan {
            level: 0,
            chosen: &[],
            ancestor_splits: &[],
            candidate_axis_signs: &signs,
            lr: 1.0,
            l1_leaf: 0.0,
            max_delta_step: None,
        };
        let got = best_level_split(
            &hist,
            &[0],
            &[2],
            1.0,
            0.0,
            None,
            0.0,
            RankingContext {
                monotone: Some(scan),
                noise: None,
                table_penalties: None,
                realized_bins: None,
            },
            &CredibilityFloor::default(),
            false,
        )
        .unwrap()
        .expect("Decreasing sign accepts the informative candidate");
        assert_eq!(got.axis, 0);
        assert_eq!(got.bin_le, 1);
        assert!(!got.missing_left);
        // gain = 0.5 * (newton_term(-5,10) + newton_term(5,2) - newton_term(0,12)), lambda=1.
        assert_eq!(got.gain.to_bits(), 4_617_656_699_751_553_334_u64);

        // The Increasing sign on the SAME hist rejects the same candidate outright (low_v <
        // high_v is required but the informative split has low_v > high_v here) -- pin the
        // rejection too, since candidate_monotone_ok's `continue` path also runs through the
        // hoisted buffers.
        let signs_inc = [Some(MonoSign::Increasing)];
        let scan_inc = MonotoneScan {
            candidate_axis_signs: &signs_inc,
            ..scan
        };
        let rejected = best_level_split(
            &hist,
            &[0],
            &[2],
            1.0,
            0.0,
            None,
            0.0,
            RankingContext {
                monotone: Some(scan_inc),
                noise: None,
                table_penalties: None,
                realized_bins: None,
            },
            &CredibilityFloor::default(),
            false,
        )
        .unwrap();
        assert!(rejected.is_none());
    }

    #[test]
    fn path_smooth_shrinks_leaves_toward_parent_node() {
        // depth 1: leaf0 raw value +1, leaf1 raw value −1, each n=10; root value 0.
        // λ=0, lr=1 ⇒ newton_leaf(g,h) = −g/h matches the leaf values below.
        let lg = [-10.0_f64, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let lh = [10.0_f64, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let lc = [10u64, 10, 0, 0, 0, 0, 0, 0];
        let base = [1.0_f32, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // ps = 0 ⇒ exact no-op.
        let mut a = base;
        apply_path_smooth(
            &mut a,
            &lg,
            &lh,
            &lc,
            1,
            0.0,
            0.0,
            1.0,
            None,
            0.0,
            CredibilityEvidence::Count,
        )
        .unwrap();
        assert_eq!(a, base);
        // ps = 2 ⇒ each leaf shrinks toward the root (0) by the factor n/(n+ps)=10/12.
        // lh == lc here, so Count and Hessian evidence agree — this test only pins the
        // blend arithmetic and the ps<=0 no-op; the next test pins evidence SELECTION.
        let mut s = base;
        apply_path_smooth(
            &mut s,
            &lg,
            &lh,
            &lc,
            1,
            0.0,
            0.0,
            1.0,
            None,
            2.0,
            CredibilityEvidence::Count,
        )
        .unwrap();
        assert!((s[0] - 10.0 / 12.0).abs() < 1e-6);
        assert!((s[1] + 10.0 / 12.0).abs() < 1e-6);
        let mut h = base;
        apply_path_smooth(
            &mut h,
            &lg,
            &lh,
            &lc,
            1,
            0.0,
            0.0,
            1.0,
            None,
            2.0,
            CredibilityEvidence::Hessian,
        )
        .unwrap();
        assert!((h[0] - 10.0 / 12.0).abs() < 1e-6);
        assert!((h[1] + 10.0 / 12.0).abs() < 1e-6);
    }

    /// §07.6 evidence generalization — the core new behavior: swapping `Count` for `Hessian`
    /// evidence can REVERSE which of two leaves is treated as "more credible" (shrinks less).
    /// Leaf0 mimics a zero-inflated cell: most of the tree's rows (count=90) but a low per-row
    /// claim mass averaging 0.1 each (Σh=9) — e.g. mostly-zero Poisson/Tweedie rows. Leaf1 is
    /// the opposite: few rows (count=10) but a high per-row mass averaging 9.1 each (Σh=91) —
    /// e.g. rows with substantial predicted frequency. `h_bar = (9+91)/(90+10) = 1.0` exactly
    /// (2026-07-23 amendment: evidence is normalized to EFFECTIVE ROWS `h/h_bar`, not raw Σh —
    /// see `credibility_evidence_n`, `constraints.rs`), so leaf0's effective-row evidence is
    /// exactly 9 and leaf1's is exactly 91, despite leaf0 having 9× the raw row COUNT. Root
    /// gradients are symmetric (Σg=0) so the internal (root) Newton value is exactly 0, making
    /// each leaf's smoothed value pure `v · n/(n+k)` with no parent offset — hand-verifiable.
    /// Under Count evidence leaf0 is the "confident" one (90 rows) and barely shrinks; under
    /// (normalized) Hessian evidence that flips: leaf0 shrinks HARD toward the parent because
    /// its rows carry little actual claim mass on average, which is the sparse-evidence
    /// behavior `CredibilityEvidence::Hessian` exists to capture.
    #[test]
    fn path_smooth_hessian_evidence_flips_which_leaf_is_credible() {
        let lg = [-5.0_f64, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // Σg = 0 ⇒ root v = 0.
        let lh = [9.0_f64, 91.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // leaf0 low mass, leaf1 high.
        let lc = [90u64, 10, 0, 0, 0, 0, 0, 0]; // leaf0 many rows, leaf1 few.
        let base = [1.0_f32, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let k = 10.0;

        let mut count_ev = base;
        apply_path_smooth(
            &mut count_ev,
            &lg,
            &lh,
            &lc,
            1,
            0.0,
            0.0,
            1.0,
            None,
            k,
            CredibilityEvidence::Count,
        )
        .unwrap();
        // Z0 = 90/100 = 0.9, Z1 = 10/20 = 0.5 — leaf0 (by count) is the confident one.
        assert!((f64::from(count_ev[0]) - 0.9).abs() < 1e-6);
        assert!((f64::from(count_ev[1]) + 0.5).abs() < 1e-6);

        let mut hess_ev = base;
        apply_path_smooth(
            &mut hess_ev,
            &lg,
            &lh,
            &lc,
            1,
            0.0,
            0.0,
            1.0,
            None,
            k,
            CredibilityEvidence::Hessian,
        )
        .unwrap();
        // h_bar = 100/100 = 1.0 ⇒ effective rows n_eff0=9/1=9, n_eff1=91/1=91.
        // Z0 = 9/19, Z1 = 91/101 — leaf1 (by normalized hessian mass) is now the confident one.
        assert!((f64::from(hess_ev[0]) - 9.0 / 19.0).abs() < 1e-6);
        assert!((f64::from(hess_ev[1]) + 91.0 / 101.0).abs() < 1e-6);

        // The headline claim: leaf0's sparse-evidence (low per-row Σh) blend sits STRICTLY
        // closer to the parent (0) under Hessian evidence than under Count evidence.
        assert!(
            hess_ev[0].abs() < count_ev[0].abs(),
            "hessian evidence must shrink the low-claim-mass leaf harder toward parent: \
             count={}, hessian={}",
            count_ev[0],
            hess_ev[0]
        );
    }

    /// §07.6 evidence normalization (2026-07-23 amendment), the requested pin: for a leaf
    /// holding EXACTLY an average per-row hessian share (`h_leaf = n_leaf · h_bar`), normalized
    /// `Hessian` evidence must equal `Count` evidence with the SAME `n_leaf` — at ANY hessian
    /// magnitude, not just `O(1)`. Uses `h_bar = 1000` (Tweedie/exposure-weighted scale, not
    /// the roughly-`O(1)` Poisson-unit-weight case) specifically to reproduce the `ohlsson_pp`
    /// smoke-test failure: the OLD (pre-amendment, un-normalized) raw-Σh evidence for leaf0
    /// here would have been `h=20000`, giving `Z=20000/20010≈0.9995` (saturated near 1
    /// regardless of leaf0 only having 20 rows) — exactly the reported bug.
    #[test]
    fn path_smooth_hessian_evidence_matches_count_for_average_hessian_at_large_scale() {
        let h_bar_target = 1000.0_f64;
        let n0 = 20u64;
        let n1 = 5u64;
        let lg = [-5.0_f64, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // Σg = 0 ⇒ root v = 0.
        let lh = [
            n0 as f64 * h_bar_target,
            n1 as f64 * h_bar_target,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let lc = [n0, n1, 0, 0, 0, 0, 0, 0];
        let base = [1.0_f32, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let k = 10.0;

        let mut count_ev = base;
        apply_path_smooth(
            &mut count_ev,
            &lg,
            &lh,
            &lc,
            1,
            0.0,
            0.0,
            1.0,
            None,
            k,
            CredibilityEvidence::Count,
        )
        .unwrap();
        let mut hess_ev = base;
        apply_path_smooth(
            &mut hess_ev,
            &lg,
            &lh,
            &lc,
            1,
            0.0,
            0.0,
            1.0,
            None,
            k,
            CredibilityEvidence::Hessian,
        )
        .unwrap();

        // h_bar = 25000/25 = 1000 exactly here, so n_eff = h/h_bar = n EXACTLY for both leaves
        // — Hessian evidence must reduce to Count evidence bit-for-bit, not just "close".
        assert_eq!(
            count_ev, hess_ev,
            "average-hessian leaves at Tweedie-scale h must match count evidence exactly"
        );
        // Sanity: a real, non-trivial shrink (not both collapsing to 0 or staying at base).
        assert!((f64::from(count_ev[0]) - 20.0 / 30.0).abs() < 1e-6);
        assert!((f64::from(count_ev[1]) + 5.0 / 15.0).abs() < 1e-6);
    }

    /// §07.6 bit-identity: the evidence CHOICE (`Count` vs `Hessian`) must never even be read
    /// when `path_smooth <= 0.0` — not merely "happen to agree". `lh`/`lc` here are deliberately
    /// wildly different (they would flip the shrink direction per the test above if evidence
    /// selection ran), so any output difference between the two evidence modes at `ps = 0.0`
    /// would prove the early-return guard is not actually gating the dispatch.
    #[test]
    fn apply_path_smooth_evidence_choice_is_unreachable_when_off() {
        let lg = [-10.0_f64, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let lh = [1.0_f64, 999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // wildly different from lc.
        let lc = [999u64, 1, 0, 0, 0, 0, 0, 0];
        let base = [1.0_f32, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let mut count_ev = base;
        apply_path_smooth(
            &mut count_ev,
            &lg,
            &lh,
            &lc,
            1,
            0.0,
            0.0,
            1.0,
            None,
            0.0,
            CredibilityEvidence::Count,
        )
        .unwrap();
        let mut hess_ev = base;
        apply_path_smooth(
            &mut hess_ev,
            &lg,
            &lh,
            &lc,
            1,
            0.0,
            0.0,
            1.0,
            None,
            0.0,
            CredibilityEvidence::Hessian,
        )
        .unwrap();
        assert_eq!(
            count_ev, base,
            "Count evidence must be a no-op at path_smooth=0"
        );
        assert_eq!(
            hess_ev, base,
            "Hessian evidence must be a no-op at path_smooth=0"
        );
        assert_eq!(count_ev, hess_ev);
    }

    #[test]
    fn grow_with_large_path_smooth_collapses_leaf_spread() {
        // Distinct per-quadrant gradients ⇒ an informative depth-2 tree.
        let c0: Vec<u8> = vec![1, 1, 1, 1, 2, 2, 2, 2];
        let c1: Vec<u8> = vec![1, 1, 2, 2, 1, 1, 2, 2];
        let x = matrix(vec![c0, c1], &[3, 3]);
        let g = [-3.0_f32, -3.0, 1.0, 1.0, 2.0, 2.0, 6.0, 6.0];
        let gh = gradhess(&g, &[1.0; 8]);
        let rows: Vec<u32> = (0..8).collect();
        let w = ones(x.n_rows as usize);
        let base = grow_oblivious_tree(&x, &gh, &rows, &[0, 1], &cfg(1.0, 0.1, 0.0, 3), &w)
            .unwrap()
            .expect("a tree");
        let smooth = grow_oblivious_tree(
            &x,
            &gh,
            &rows,
            &[0, 1],
            &cfg_cred(CredibilityFloor {
                path_smooth: 1e6,
                ..CredibilityFloor::default()
            }),
            &w,
        )
        .unwrap()
        .expect("a tree");
        // path_smooth is value-level: identical structure, but the leaves collapse toward
        // the (near-zero) parent/root under heavy smoothing.
        assert_eq!(base.splits, smooth.splits);
        let spread = |t: &ObliviousTree| {
            let lo = t.leaves.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = t.leaves.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            hi - lo
        };
        assert!(spread(&base) > 1e-3, "base tree must be informative");
        assert!(
            spread(&smooth) < spread(&base),
            "heavy path_smooth must compress the leaf spread: base {} vs smooth {}",
            spread(&base),
            spread(&smooth)
        );
    }

    #[test]
    fn grow_recovers_two_feature_structure() {
        // Target depends on (x0 bin, x1 bin); gradients pull leaves apart.
        let c0: Vec<u8> = vec![1, 1, 1, 1, 2, 2, 2, 2]; // axis0 splits rows 0-3 | 4-7
        let c1: Vec<u8> = vec![1, 1, 2, 2, 1, 1, 2, 2]; // axis1 splits within
        let x = matrix(vec![c0, c1], &[3, 3]);
        // Gradients: distinct per (x0,x1) quadrant so both splits have positive gain.
        let g = [-3.0_f32, -3.0, 1.0, 1.0, 2.0, 2.0, 6.0, 6.0];
        let gh = gradhess(&g, &[1.0; 8]);
        let rows: Vec<u32> = (0..8).collect();
        let tree = grow_oblivious_tree(
            &x,
            &gh,
            &rows,
            &[0, 1],
            &cfg(1.0, 1.0, 0.0, 3),
            &ones(x.n_rows as usize),
        )
        .unwrap()
        .expect("a tree should grow");
        assert!((1..=3).contains(&tree.depth));
        // Interaction order is distinct raw support; repeated levels are valid refinements.
        let mut raws: Vec<u32> = tree.splits.iter().map(|s| s.axis).collect();
        raws.sort_unstable();
        raws.dedup();
        assert!(!raws.is_empty());
        assert!(raws.len() <= usize::from(tree.depth));
    }

    #[test]
    fn grow_is_deterministic_across_thread_counts() {
        let n = 400usize;
        let c0: Vec<u8> = (0..n).map(|i| u8::try_from(i % 5 + 1).unwrap()).collect();
        let c1: Vec<u8> = (0..n).map(|i| u8::try_from(i % 7 + 1).unwrap()).collect();
        let c2: Vec<u8> = (0..n).map(|i| u8::try_from(i % 3 + 1).unwrap()).collect();
        let x = matrix(vec![c0, c1, c2], &[6, 8, 4]);
        let g: Vec<f32> = (0..n).map(|i| (i as f32 % 11.0) - 5.0).collect();
        let gh = gradhess(&g, &vec![1.0; n]);
        let rows: Vec<u32> = (0..n as u32).collect();
        let run = |nt: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                grow_oblivious_tree(
                    &x,
                    &gh,
                    &rows,
                    &[0, 1, 2],
                    &cfg(1.0, 0.1, 0.0, 3),
                    &ones(x.n_rows as usize),
                )
                .unwrap()
                .unwrap()
            })
        };
        let a = run(1);
        assert_eq!(a, run(2));
        assert_eq!(a, run(8));
    }

    #[test]
    fn grow_with_random_strength_is_deterministic_across_thread_counts() {
        let n = 360usize;
        let c0: Vec<u8> = (0..n).map(|i| u8::try_from(i % 6 + 1).unwrap()).collect();
        let c1: Vec<u8> = (0..n).map(|i| u8::try_from(i % 5 + 1).unwrap()).collect();
        let c2: Vec<u8> = (0..n).map(|i| u8::try_from(i % 4 + 1).unwrap()).collect();
        let x = matrix(vec![c0, c1, c2], &[7, 6, 5]);
        let g: Vec<f32> = (0..n).map(|i| (i as f32 % 13.0) - 6.0).collect();
        let gh = gradhess(&g, &vec![1.0; n]);
        let rows: Vec<u32> = (0..n as u32).collect();
        let mut cfg = cfg(1.0, 0.1, 0.0, 3);
        cfg.quant_seed = 99;
        cfg.round = 5;
        cfg.random_strength = 0.75;
        let run = |nt: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                grow_oblivious_tree(&x, &gh, &rows, &[0, 1, 2], &cfg, &ones(x.n_rows as usize))
                    .unwrap()
                    .unwrap()
            })
        };
        let a = run(1);
        assert_eq!(a, run(2));
        assert_eq!(a, run(8));
    }

    #[test]
    fn constant_gradient_gives_no_split() {
        // All gradients equal ⇒ no split improves the objective ⇒ None (no tree).
        let x = matrix(vec![vec![1u8, 2, 1, 2]], &[3]);
        let gh = gradhess(&[1.0, 1.0, 1.0, 1.0], &[1.0; 4]);
        let rows = [0u32, 1, 2, 3];
        let tree = grow_oblivious_tree(
            &x,
            &gh,
            &rows,
            &[0],
            &cfg(1.0, 0.1, 1e-9, 1),
            &ones(x.n_rows as usize),
        )
        .unwrap();
        assert!(tree.is_none(), "constant gradient must not split");
    }

    #[test]
    fn feature_budget_is_enforced_at_construction() {
        // One level per distinct raw feature, one MORE than the order cap allows.
        //
        // Written against the constants, so it stays an I1 violation wherever the caps land
        // — but be honest about WHICH clause refuses it. At `MAX_ORDER == MAX_DEPTH` (the
        // high-order lift) `MAX_ORDER + 1` levels exceed the DEPTH cap first, so this now
        // proves "over-budget shapes are refused at construction", not "the order cap is
        // enforced independently of depth". The order clause is unreachable in isolation;
        // `explain::tests::i1_order_clause_is_subsumed_at_equal_caps` asserts that fact and
        // will fire if the caps ever move apart again.
        let n = u32::try_from(MAX_ORDER).unwrap() + 1;
        let prov: Vec<AxisProvenance> = (0..n)
            .map(|i| AxisProvenance {
                raw: FeatureId(i),
                kind: AxisKind::Numeric,
            })
            .collect();
        let splits: Vec<Split> = (0..n)
            .map(|a| Split {
                axis: a,
                bin_le: 1,
                missing_left: false,
            })
            .collect();
        assert!(matches!(
            ObliviousTree::try_new(splits.clone(), vec![0.0; leaf_slots(n as usize)], &prov),
            Err(PbError::InvariantViolated {
                invariant: crate::Invariant::FeatureBudget
            })
        ));
        // ... and exactly MAX_ORDER of them is accepted.
        let at_cap: Vec<Split> = splits.iter().take(MAX_ORDER).copied().collect();
        assert!(ObliviousTree::try_new(at_cap, vec![0.0; leaf_slots(MAX_ORDER)], &prov).is_ok());

        // Repeated raw features are valid lower-order refinements.
        let prov2 = vec![
            AxisProvenance {
                raw: FeatureId(0),
                kind: AxisKind::Numeric,
            },
            AxisProvenance {
                raw: FeatureId(0),
                kind: AxisKind::Numeric,
            },
        ];
        let reused = vec![
            Split {
                axis: 0,
                bin_le: 1,
                missing_left: false,
            },
            Split {
                axis: 1,
                bin_le: 1,
                missing_left: false,
            },
        ];
        assert!(ObliviousTree::try_new(reused, vec![0.0; 8], &prov2).is_ok());
    }

    #[test]
    fn max_order_one_still_grows_depth_three_by_reusing_factor() {
        // max_order = 1 caps DISTINCT raw support, not depth: the tree can refine one feature
        // with multiple thresholds.
        let c0: Vec<u8> = vec![1, 1, 2, 2, 3, 3, 4, 4];
        let c1: Vec<u8> = vec![1, 2, 1, 2, 1, 2, 1, 2];
        let x = matrix(vec![c0, c1], &[5, 3]);
        let gh = gradhess(&[-4.0, -4.0, -1.0, -1.0, 1.0, 1.0, 4.0, 4.0], &[1.0; 8]);
        let rows = [0u32, 1, 2, 3, 4, 5, 6, 7];
        let tree = grow_oblivious_tree(
            &x,
            &gh,
            &rows,
            &[0, 1],
            &cfg(1.0, 0.1, 0.0, 1),
            &ones(x.n_rows as usize),
        )
        .unwrap()
        .unwrap();
        assert_eq!(tree.depth, 3);
        let mut raws: Vec<u32> = tree.splits.iter().map(|s| s.axis).collect();
        raws.sort_unstable();
        raws.dedup();
        assert_eq!(raws, vec![0]);
    }

    #[test]
    fn interaction_gain_hurdle_falls_back_to_used_factor() {
        // Axis 0 has several useful thresholds. Axis 1 has weaker conditional gain, so a high
        // hurdle rejects it and spends the second level refining axis 0 instead.
        let c0: Vec<u8> = vec![1, 1, 2, 2, 3, 3, 4, 4];
        let c1: Vec<u8> = vec![1, 2, 1, 2, 1, 2, 1, 2];
        let x = matrix(vec![c0, c1], &[5, 3]);
        let g = [-6.4, -6.4, -1.6, -1.6, 1.6, 1.6, 6.4, 6.4];
        let gh = gradhess(&g, &[1.0; 8]);
        let rows = [0u32, 1, 2, 3, 4, 5, 6, 7];

        let mut hurdled = cfg(1.0, 0.1, 0.0, 3);
        hurdled.interaction_gain_hurdle = 1.0e6;
        let refined_main = grow_oblivious_tree(&x, &gh, &rows, &[0, 1], &hurdled, &ones(8))
            .unwrap()
            .unwrap();
        assert_eq!(refined_main.depth, 3);
        assert!(refined_main.splits.iter().all(|s| s.axis == 0));
    }

    #[test]
    fn fresh_factor_must_beat_reused_refinement() {
        // Axis 0 carries a strong ordered shape with multiple useful thresholds. Axis 1 has a
        // weaker residual pattern. Once the hurdle policy is enabled, the second level should
        // refine axis 0 because that lower-order split has greater score than the fresh-factor split.
        let c0: Vec<u8> = vec![1, 1, 2, 2, 3, 3, 4, 4];
        let c1: Vec<u8> = vec![1, 2, 1, 2, 1, 2, 1, 2];
        let x = matrix(vec![c0, c1], &[5, 3]);
        let g = [-6.7, -6.1, -1.9, -1.3, 1.3, 1.9, 6.1, 6.7];
        let gh = gradhess(&g, &[1.0; 8]);
        let rows = [0u32, 1, 2, 3, 4, 5, 6, 7];

        let mut cfg = cfg(1.0, 0.1, 0.0, 3);
        cfg.interaction_gain_hurdle = 1.0e-9;
        let tree = grow_oblivious_tree(&x, &gh, &rows, &[0, 1], &cfg, &ones(8))
            .unwrap()
            .unwrap();
        assert_eq!(tree.depth, 3);
        assert_eq!(tree.splits[0].axis, 0);
        assert_eq!(tree.splits[1].axis, 0);
    }

    #[test]
    fn zero_hurdle_keeps_greedy_fresh_admission() {
        // Exact zero is the neutral engine primitive: if a fresh raw feature can split, it is
        // admitted greedily and reused-axis competition is skipped for legacy compatibility.
        let c0: Vec<u8> = vec![1, 1, 1, 1, 2, 2, 2, 2];
        let c1: Vec<u8> = vec![1, 2, 1, 2, 1, 2, 1, 2];
        let x = matrix(vec![c0, c1], &[3, 3]);
        let g = [-6.4, -1.6, -6.4, -1.6, 1.6, 6.4, 1.6, 6.4];
        let gh = gradhess(&g, &[1.0; 8]);
        let rows = [0u32, 1, 2, 3, 4, 5, 6, 7];

        let tree = grow_oblivious_tree(&x, &gh, &rows, &[0, 1], &cfg(1.0, 0.1, 0.0, 3), &ones(8))
            .unwrap()
            .unwrap();
        assert_eq!(tree.splits[0].axis, 0);
        assert_eq!(tree.splits[1].axis, 1);
    }

    #[test]
    fn adaptive_interaction_hurdle_relaxes_as_main_gain_matures() {
        let c0: Vec<u8> = vec![1, 1, 1, 1, 2, 2, 2, 2];
        let c1: Vec<u8> = vec![1, 2, 1, 2, 1, 2, 1, 2];
        let x = matrix(vec![c0, c1], &[3, 3]);
        let g = [-6.4, -1.6, -6.4, -1.6, 1.6, 6.4, 1.6, 6.4];
        let gh = gradhess(&g, &[1.0; 8]);
        let rows = [0u32, 1, 2, 3, 4, 5, 6, 7];
        let mut cfg = cfg(1.0, 0.1, 0.0, 3);
        cfg.interaction_gain_hurdle = 0.2;

        let early = grow_oblivious_tree_with_leaf_map(
            &x,
            &gh,
            &rows,
            &[0, 1],
            &cfg,
            InteractionHurdleState::new(InteractionGainHurdleMode::Adaptive, None),
            &ones(8),
        )
        .unwrap()
        .unwrap()
        .tree;
        assert_eq!(early.depth, 1);

        let mature = grow_oblivious_tree_with_leaf_map(
            &x,
            &gh,
            &rows,
            &[0, 1],
            &cfg,
            InteractionHurdleState::new(InteractionGainHurdleMode::Adaptive, Some(1.0e12)),
            &ones(8),
        )
        .unwrap()
        .unwrap()
        .tree;
        assert_eq!(mature.depth, 2);
    }

    #[test]
    fn group_whitelist_gates_candidate_axes() {
        // Axis 1 is the only informative axis; a group list that contains only raw 0
        // must therefore produce no tree, while a group containing raw 1 admits it.
        let c0: Vec<u8> = vec![1, 1, 1, 1];
        let c1: Vec<u8> = vec![1, 1, 2, 2];
        let x = matrix(vec![c0, c1], &[2, 3]);
        let gh = gradhess(&[-2.0, -2.0, 2.0, 2.0], &[1.0; 4]);
        let rows = [0u32, 1, 2, 3];

        let denied_groups = vec![FeatureSet::new(&[0])];
        let mut denied = cfg(1.0, 0.1, 0.0, 2);
        denied.groups = Some(&denied_groups);
        assert!(
            grow_oblivious_tree(&x, &gh, &rows, &[0, 1], &denied, &ones(x.n_rows as usize))
                .unwrap()
                .is_none()
        );

        let allowed_groups = vec![FeatureSet::new(&[1])];
        let mut allowed = cfg(1.0, 0.1, 0.0, 2);
        allowed.groups = Some(&allowed_groups);
        let tree = grow_oblivious_tree(&x, &gh, &rows, &[0, 1], &allowed, &ones(x.n_rows as usize))
            .unwrap()
            .unwrap();
        assert_eq!(tree.splits[0].axis, 1);
    }

    #[test]
    fn newton_gain_learns_missing_right() {
        // Mirror of the learned-LEFT test: with bin1/bin2 gradients swapped, routing
        // missing RIGHT (with bin2) is the higher-gain pairing, so missing_left=false.
        // bin0(missing): g=2,h=1; bin1: g=-6,h=3; bin2: g=4,h=2. λ=1, total (0,6).
        //   ml=false: L=bin1 (-6,3)→9; R=bin2+miss (6,3)→9; gain=½(18)=9.
        //   ml=true:  L=bin1+miss (-4,4)→3.2; R=bin2 (4,2)→5.333; gain=4.267.
        let mut hist = Hist::try_zeros(1, 1, 3).unwrap();
        let set = |hist: &mut Hist, bin: usize, g: f64, h: f64| {
            let o = hist.offset(0, 0, bin).unwrap();
            hist.g[o] = g;
            hist.h[o] = h;
            hist.count[o] = 1;
        };
        set(&mut hist, 0, 2.0, 1.0);
        set(&mut hist, 1, -6.0, 3.0);
        set(&mut hist, 2, 4.0, 2.0);
        let best = best_level_split(
            &hist,
            &[0],
            &[2],
            1.0,
            0.0,
            None,
            0.0,
            RankingContext::default(),
            &CredibilityFloor::default(),
            false,
        )
        .unwrap()
        .unwrap();
        assert!(!best.missing_left, "missing should be learned RIGHT here");
        assert!((best.gain - 9.0).abs() < 1e-9, "gain {} != 9", best.gain);
    }

    #[test]
    fn lookup_routes_missing_and_data_bins() {
        // The ONLY ObliviousTree::lookup test: a depth-1 tree, exercising data bins AND
        // the reserved missing bin (bin 0) under both learned directions. Leaf index
        // convention: low (bin <= bin_le) → bit 1, so leaves[1] is the LOW value.
        let prov = vec![AxisProvenance {
            raw: FeatureId(0),
            kind: AxisKind::Numeric,
        }];
        let leaves = [100.0_f32, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // idx0=high, idx1=low
        let left = ObliviousTree::try_new(
            vec![Split {
                axis: 0,
                bin_le: 1,
                missing_left: true,
            }],
            leaves.to_vec(),
            &prov,
        )
        .unwrap();
        assert_eq!(left.lookup(&[0]).unwrap(), 200.0); // missing → low (missing_left)
        assert_eq!(left.lookup(&[1]).unwrap(), 200.0); // bin 1 ≤ 1 → low
        assert_eq!(left.lookup(&[2]).unwrap(), 100.0); // bin 2 > 1 → high

        let right = ObliviousTree::try_new(
            vec![Split {
                axis: 0,
                bin_le: 1,
                missing_left: false,
            }],
            leaves.to_vec(),
            &prov,
        )
        .unwrap();
        assert_eq!(right.lookup(&[0]).unwrap(), 100.0); // missing → high
    }

    #[test]
    fn grow_recovers_three_feature_structure() {
        // 3 binary features, 8 rows = all (b0,b1,b2). Additive gradient with strictly
        // decreasing coefficients (4,2,1) ⇒ a full depth-3 tree on 3 distinct features.
        let c0 = vec![1u8, 1, 1, 1, 2, 2, 2, 2];
        let c1 = vec![1u8, 1, 2, 2, 1, 1, 2, 2];
        let c2 = vec![1u8, 2, 1, 2, 1, 2, 1, 2];
        let x = matrix(vec![c0.clone(), c1.clone(), c2.clone()], &[3, 3, 3]);
        let g: Vec<f32> = (0..8)
            .map(|r| {
                let b0 = f32::from(c0[r] - 1);
                let b1 = f32::from(c1[r] - 1);
                let b2 = f32::from(c2[r] - 1);
                4.0 * b0 + 2.0 * b1 + b2 - 3.5
            })
            .collect();
        let gh = gradhess(&g, &[1.0; 8]);
        let rows: Vec<u32> = (0..8).collect();
        // λ = 0: pure (non-negative) Newton gain, so every separating split is kept.
        // (With λ > 0 the small-variance 3rd split is correctly regularized away — the
        // L2 penalty adds λ to TWO child denominators vs one parent — which is why the
        // default-λ tree may stop at depth 2; that is correct behavior, not a defect.)
        let tree = grow_oblivious_tree(
            &x,
            &gh,
            &rows,
            &[0, 1, 2],
            &cfg(0.0, 0.1, 0.0, 3),
            &ones(x.n_rows as usize),
        )
        .unwrap()
        .expect("a depth-3 tree should grow");
        assert_eq!(tree.depth, 3, "all three features are informative");
        let mut raws: Vec<u32> = tree.splits.iter().map(|s| s.axis).collect();
        raws.sort_unstable();
        raws.dedup();
        assert_eq!(raws, vec![0, 1, 2]); // each level a distinct raw feature (I1)
    }

    /// A non-trivial depth-3 fixture for the histogram-subtraction tests: the 8-cell
    /// 3-binary-feature design replicated with deterministic per-row gradient noise (so the
    /// level-2 histograms carry real, varied mass), plus a weak 4th feature. The grower uses
    /// features 0,1,2 (coeffs 4,2,1), so at level 2 A_2 = {2,3} ⊂ A_1 = {1,2,3} with SHIFTED
    /// positions — exercising the axis-position remap.
    fn subtraction_fixture() -> (BinnedMatrix, GradHess, Vec<u32>) {
        let base0 = [1u8, 1, 1, 1, 2, 2, 2, 2];
        let base1 = [1u8, 1, 2, 2, 1, 1, 2, 2];
        let base2 = [1u8, 2, 1, 2, 1, 2, 1, 2];
        let reps = 80usize;
        let (mut c0, mut c1, mut c2, mut c3) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for r in 0..reps {
            for k in 0..8usize {
                c0.push(base0[k]);
                c1.push(base1[k]);
                c2.push(base2[k]);
                c3.push(1 + ((r * 8 + k) * 7 + 3) as u8 % 3); // weak 4-bin decoy in [1,3]
            }
        }
        let x = matrix(vec![c0.clone(), c1.clone(), c2.clone(), c3], &[3, 3, 3, 4]);
        let n = c0.len();
        let g: Vec<f32> = (0..n)
            .map(|i| {
                let b0 = f32::from(c0[i] - 1);
                let b1 = f32::from(c1[i] - 1);
                let b2 = f32::from(c2[i] - 1);
                4.0 * b0 + 2.0 * b1 + b2 - 3.5 + 0.05 * (((i * 13 + 7) % 11) as f32 - 5.0)
            })
            .collect();
        let gh = gradhess(&g, &vec![1.0_f32; n]);
        (x, gh, (0..n as u32).collect())
    }

    #[test]
    fn level2_subtraction_reproduces_full_build_tree() {
        // THE de-risking gate: grow the SAME data with the level-2 histogram-subtraction path
        // ON (default) and OFF (full build), assert the trees are byte-identical (splits, leaf
        // bits, depth). Subtraction perturbs g/h only at ~1e-11 — far below any non-tie split
        // gap — so the selected splits, and the gh-derived leaves, match exactly.
        let (x, gh, rows) = subtraction_fixture();
        let w = ones(x.n_rows as usize);
        let axes = [0u32, 1, 2, 3];
        let on = cfg(0.0, 0.1, 0.0, 3); // hist_subtraction = true
        let off = GrowConfig {
            hist_subtraction: false,
            ..cfg(0.0, 0.1, 0.0, 3)
        };
        let t_on = grow_oblivious_tree(&x, &gh, &rows, &axes, &on, &w)
            .unwrap()
            .expect("tree");
        let t_off = grow_oblivious_tree(&x, &gh, &rows, &axes, &off, &w)
            .unwrap()
            .expect("tree");
        assert!(
            t_on.depth >= 2,
            "fixture must reach depth 2 so the level-2 subtraction path engages (got {})",
            t_on.depth
        );
        assert_eq!(
            t_on, t_off,
            "level-2 subtraction must reproduce the full-build tree byte-for-byte"
        );
    }

    #[test]
    fn level2_subtraction_is_thread_count_independent() {
        // Subtraction must keep the §1 determinism GATE: same tree across 1/2/8 threads.
        let (x, gh, rows) = subtraction_fixture();
        let w = ones(x.n_rows as usize);
        let axes = [0u32, 1, 2, 3];
        let grow = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                grow_oblivious_tree(&x, &gh, &rows, &axes, &cfg(0.0, 0.1, 0.0, 3), &w)
                    .unwrap()
                    .expect("tree")
            })
        };
        let base = grow(1);
        assert_eq!(base, grow(2), "subtraction tree differs at 2 threads");
        assert_eq!(base, grow(8), "subtraction tree differs at 8 threads");
    }

    #[test]
    fn quantized_subtraction_reproduces_full_build_tree() {
        // QHIST now uses histogram subtraction at levels 1+2 (build only the smaller children by
        // quantized integer accumulation, derive the larger by subtracting from the retained parent).
        // The integer accumulation + dequantized-f64 subtraction perturbs g/h only at ~1e-11 — far
        // below any non-tie split gap — so the subtracted quantized grow selects the SAME splits and
        // gh-derived leaves as the full-build quantized grow (accuracy-neutral).
        let (x, gh, rows) = subtraction_fixture();
        let w = ones(x.n_rows as usize);
        let axes = [0u32, 1, 2, 3];
        let q_on = GrowConfig {
            hist_precision: HistPrecision::QuantizedI32,
            hist_subtraction: true,
            ..cfg(0.0, 0.1, 0.0, 3)
        };
        let q_off = GrowConfig {
            hist_subtraction: false,
            ..q_on.clone()
        };
        let t_on = grow_oblivious_tree(&x, &gh, &rows, &axes, &q_on, &w)
            .unwrap()
            .expect("tree");
        let t_off = grow_oblivious_tree(&x, &gh, &rows, &axes, &q_off, &w)
            .unwrap()
            .expect("tree");
        assert!(
            t_on.depth >= 2,
            "fixture must reach depth 2 so the quantized subtraction path engages (got {})",
            t_on.depth
        );
        assert_eq!(
            t_on, t_off,
            "quantized subtraction must reproduce the full-build quantized tree"
        );
    }

    #[test]
    fn grown_tree_passes_check_feature_budget() {
        // Run the NAMED I1 gate fn (explain::check_feature_budget) on a fitted tree.
        let c0 = vec![1u8, 1, 2, 2];
        let c1 = vec![1u8, 2, 1, 2];
        let x = matrix(vec![c0, c1], &[3, 3]);
        let gh = gradhess(&[-2.0, -1.0, 1.0, 2.0], &[1.0; 4]);
        let rows = [0u32, 1, 2, 3];
        let tree = grow_oblivious_tree(
            &x,
            &gh,
            &rows,
            &[0, 1],
            &cfg(1.0, 0.1, 0.0, 3),
            &ones(x.n_rows as usize),
        )
        .unwrap()
        .unwrap();
        let model = model_from(tree, &x);
        crate::explain::check_feature_budget(&model).expect("a grown tree must satisfy I1");
    }

    /// Wrap a grown tree in a minimal `Model` so the named I1 gate can run on it.
    fn model_from(tree: ObliviousTree, x: &BinnedMatrix) -> crate::engine::Model {
        use crate::cat::CatEncoderStore;
        use crate::engine::{ExactnessMode, ModelSchema};
        use crate::loss::{Link, LossId, ObjectiveTag};
        crate::engine::Model {
            f0: 0.0,
            trees: vec![(1.0, tree)],
            grids: x.grids.clone(),
            provenance: x.provenance.clone(),
            link: Link::Identity,
            mode: ExactnessMode::Exact,
            schema: ModelSchema {
                feature_names: Vec::new(),
                feature_kinds: Vec::new(),
                cat_encoders: CatEncoderStore::new(),
                class_labels: None,
                objective: ObjectiveTag {
                    link: Link::Identity,
                    loss: LossId::SquaredError,
                    tweedie_rho: None,
                },
            },
            schema_version: crate::serialize::SCHEMA_VERSION_UNLIFTED,
            correction: None,
            bag_spans: None,
            bag_in_bag: None,
            delta_step_gate: None,
        }
    }

    // ===================================================================
    // P1 multi-channel categoricals (design/multichannel-categoricals.md §4.2): two axes
    // sharing one raw feature id (distinct `TsEncodingId`s) must stay order-1 decomposable,
    // and the interaction-gain hurdle must never gate a second channel of an already-used
    // categorical the way it gates a genuinely fresh raw feature.
    // ===================================================================

    #[test]
    fn two_channel_categorical_stays_order_one_decomposable() {
        // Two channels of ONE categorical (e.g. mean-TS + count), stamped with the SAME raw
        // feature id and distinct encoder ids — exactly what the py binding now emits for
        // `cat_channels=["mean","count"]`. Data reused verbatim from the existing
        // `zero_hurdle_keeps_greedy_fresh_admission` fixture (there, axis 1 was a genuinely
        // different raw feature): gain computation never reads provenance, so relabeling
        // axis 1's raw to match axis 0 cannot change WHICH splits are chosen, only whether
        // they count as one raw feature or two.
        let c0: Vec<u8> = vec![1, 1, 1, 1, 2, 2, 2, 2];
        let c1: Vec<u8> = vec![1, 2, 1, 2, 1, 2, 1, 2];
        let mut x = matrix(vec![c0, c1], &[3, 3]);
        x.provenance = vec![
            AxisProvenance {
                raw: FeatureId(0),
                kind: AxisKind::CategoricalTS {
                    encoding: TsEncodingId(0),
                },
            },
            AxisProvenance {
                raw: FeatureId(0),
                kind: AxisKind::CategoricalTS {
                    encoding: TsEncodingId(1),
                },
            },
        ];
        let g = [-6.4, -1.6, -6.4, -1.6, 1.6, 6.4, 1.6, 6.4];
        let gh = gradhess(&g, &[1.0; 8]);
        let rows = [0u32, 1, 2, 3, 4, 5, 6, 7];

        let tree = grow_oblivious_tree(&x, &gh, &rows, &[0, 1], &cfg(1.0, 0.1, 0.0, 3), &ones(8))
            .unwrap()
            .unwrap();
        // Both channel axes are genuinely used...
        assert_eq!(tree.splits[0].axis, 0);
        assert_eq!(tree.splits[1].axis, 1);
        // ...yet the tree's distinct raw-feature support is exactly {0}: order-1, not order-2.
        // (`ObliviousTree::try_new` re-derives this from provenance at construction time and
        // would already have failed the I1 `FeatureBudget` invariant above if it disagreed.)
        let mut raws: Vec<u32> = tree
            .splits
            .iter()
            .map(|s| x.provenance[s.axis as usize].raw.0)
            .collect();
        raws.sort_unstable();
        raws.dedup();
        assert_eq!(raws, vec![0]);
        assert_eq!(tree.depth, 2);
    }

    #[test]
    fn interaction_gain_hurdle_admits_a_reused_categorical_channel_without_gating() {
        // The interaction-gain hurdle is applied ONLY to the fresh-axis scan
        // (`scan_level_axes` is called with a real `fresh_admission_floor` for fresh
        // candidates and `None` for reused ones, `grow_oblivious_tree_with_leaf_map` above) —
        // so whether a second channel of an already-used categorical ever faces that floor is
        // decided entirely by `fresh_axis_is_admissible`/`reused_axis_is_admissible`, the two
        // predicates that partition each level's candidate axes. This tests them directly
        // rather than relying on a hand-tuned gain race to exercise the same code path.
        //
        // axis 0 = channel 0 of a categorical (raw 0, encoder id 0), axis 1 = channel 1 of the
        // SAME categorical (raw 0, encoder id 1), axis 2 = a genuinely different raw feature
        // (raw 1). `used_raws = [0]` models the state right after axis 0's own split.
        let x = {
            let mut m = matrix(vec![vec![1u8; 4], vec![1u8; 4], vec![1u8; 4]], &[3, 3, 3]);
            m.provenance = vec![
                AxisProvenance {
                    raw: FeatureId(0),
                    kind: AxisKind::CategoricalTS {
                        encoding: TsEncodingId(0),
                    },
                },
                AxisProvenance {
                    raw: FeatureId(0),
                    kind: AxisKind::CategoricalTS {
                        encoding: TsEncodingId(1),
                    },
                },
                AxisProvenance {
                    raw: FeatureId(1),
                    kind: AxisKind::Numeric,
                },
            ];
            m
        };
        let used_raws = [0u32];

        // Channel 1 (axis 1): a REUSE of the already-used categorical. Eligible for the
        // reuse scan (never hurdle-gated), and NOT eligible as "fresh" (splitting on it can
        // never raise the tree's distinct-raw-feature order, so it must not compete for a
        // fresh admission slot either).
        assert!(reused_axis_is_admissible(&x, 1, &used_raws));
        assert!(!fresh_axis_is_admissible(&x, 1, &used_raws, None));

        // Axis 2: a genuinely different raw feature. Eligible as fresh (and therefore subject
        // to the interaction-gain hurdle), NOT eligible as a reuse.
        assert!(fresh_axis_is_admissible(&x, 2, &used_raws, None));
        assert!(!reused_axis_is_admissible(&x, 2, &used_raws));

        // End-to-end corroboration: `two_channel_categorical_stays_order_one_decomposable`
        // already grows a real (unhurdled) tree that DOES split on both channel 0 and channel
        // 1 within one order-1 tree — i.e. the eligibility this test pins is not merely
        // theoretical, it is what the grower actually exercises.
    }

    #[test]
    fn realized_extent_counts_a_multichannel_categorical_once_not_per_channel() {
        // Same 2-channel tree as `two_channel_categorical_stays_order_one_decomposable`.
        let c0: Vec<u8> = vec![1, 1, 1, 1, 2, 2, 2, 2];
        let c1: Vec<u8> = vec![1, 2, 1, 2, 1, 2, 1, 2];
        let mut x = matrix(vec![c0, c1], &[3, 3]);
        x.provenance = vec![
            AxisProvenance {
                raw: FeatureId(0),
                kind: AxisKind::CategoricalTS {
                    encoding: TsEncodingId(0),
                },
            },
            AxisProvenance {
                raw: FeatureId(0),
                kind: AxisKind::CategoricalTS {
                    encoding: TsEncodingId(1),
                },
            },
        ];
        let g = [-6.4, -1.6, -6.4, -1.6, 1.6, 6.4, 1.6, 6.4];
        let gh = gradhess(&g, &[1.0; 8]);
        let rows = [0u32, 1, 2, 3, 4, 5, 6, 7];
        let tree = grow_oblivious_tree(&x, &gh, &rows, &[0, 1], &cfg(1.0, 0.1, 0.0, 3), &ones(8))
            .unwrap()
            .unwrap();
        assert_eq!(tree.splits[0].axis, 0);
        assert_eq!(tree.splits[1].axis, 1); // two DIFFERENT axes, same raw feature

        // `RealizedExtent` is keyed by RAW feature (`prov.raw`), never by axis index. Sizing
        // the accumulator for exactly ONE raw feature and recording a tree whose splits
        // reference TWO axes (0 and 1) both mapped to that same raw proves the point two
        // ways: `record_tree` must succeed at all (had it indexed by axis instead of raw,
        // split 1's axis=1 would be out of bounds for a size-1 accumulator and error), and
        // the single pre-sized slot is the one that receives both thresholds.
        let mut extent = RealizedExtent::new(1);
        extent.record_tree(&tree.splits, &x).unwrap();
        assert_eq!(extent.per_raw.len(), 1);
        assert!(!extent.per_raw[0].is_empty());
    }

    proptest! {
        // I1 over RANDOM fitted trees + the grow→lookup round-trip (the canonical
        // low_bit must agree between the sample→leaf grower and ObliviousTree::lookup).
        // Columns include bin 0, so missing routing is exercised end-to-end.
        #[test]
        fn grown_trees_satisfy_i1_and_lookup_roundtrip(
            data in (1usize..4, 4usize..24).prop_flat_map(|(n_feat, n_rows)| {
                (
                    prop::collection::vec(prop::collection::vec(0u8..5u8, n_rows), n_feat),
                    prop::collection::vec(-10.0f32..10.0, n_rows),
                    prop::collection::vec(0.5f32..5.0, n_rows),
                )
            })
        ) {
            let (cols, g, h) = data;
            let n_feat = cols.len();
            let x = matrix(cols, &vec![5u16; n_feat]);
            let gh = gradhess(&g, &h);
            let rows: Vec<u32> = (0..x.n_rows).collect();
            let axes: Vec<u32> = (0..n_feat as u32).collect();
            let res = grow_oblivious_tree(&x, &gh, &rows, &axes, &cfg(1.0, 0.1, 0.0, 3), &ones(x.n_rows as usize));
            prop_assert!(res.is_ok());
            if let Some(tree) = res.unwrap() {
                // I1: depth 1..=3, splits.len()==depth, distinct raw features <= depth.
                prop_assert!((1..=3).contains(&tree.depth));
                prop_assert_eq!(tree.splits.len(), usize::from(tree.depth));
                let mut raws: Vec<u32> =
                    tree.splits.iter().map(|s| x.provenance[s.axis as usize].raw.0).collect();
                raws.sort_unstable();
                raws.dedup();
                prop_assert!(!raws.is_empty());
                prop_assert!(raws.len() <= usize::from(tree.depth));
                // grow→lookup round-trip: lookup folds to the SAME leaf low_bit assigns.
                for r in 0..x.n_rows as usize {
                    let row_bins: Vec<u8> = (0..n_feat).map(|f| x.data[f][r]).collect();
                    let mut idx = 0usize;
                    for (lvl, s) in tree.splits.iter().enumerate() {
                        let bit = usize::from(crate::engine::low_bit(
                            x.data[s.axis as usize][r],
                            s.bin_le,
                            s.missing_left,
                        ));
                        idx |= bit << lvl;
                    }
                    prop_assert_eq!(tree.lookup(&row_bins).unwrap(), tree.leaves[idx]);
                }
            }
        }
    }
}
