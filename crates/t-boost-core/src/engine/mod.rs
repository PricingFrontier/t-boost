//! The oblivious boosting engine (spec §2.3, §2.5, §2.6, §2.9 / §06). Owns the
//! trained-model types, the histogram accumulator, the split-finder, and the
//! boosting loop. Phase 1 (this milestone, M1.3) lands the full-precision histogram
//! engine ([`hist`]); the split-finder and `fit` loop land in M1.4/M1.5.

// Pre-existing index-arithmetic (Model::validate / score paths) flagged by the tightened
// `indexing_slicing` deny lint. Scope-allowed to keep CI green; TODO: convert to `.get()`.
#![allow(
    clippy::indexing_slicing, // JUSTIFIED: pre-existing module-scoped debt (see comment above); burn-down to `.get()`/per-fn allows is incremental, not expanded here.
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::derivable_impls
)]

use crate::cat::CatEncoderStore;
use crate::constraints::{CredibilityFloor, InteractionPolicy, MonotoneMap};
use crate::data::{AxisKind, AxisProvenance, BinnedMatrix, BorderGrid, TrainBinnedMatrix};
use crate::error::{Invariant, PbError};
use crate::loss::{GatedDeltaStep, Link, Loss, ObjectiveTag};
use crate::scoring::ScoringBank;
use crate::simd::{score_tile, CHUNK_ROWS};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

pub mod boost;
pub mod hist;
pub mod split;

/// The hard cap on an oblivious tree's DEPTH (number of shared level tests).
///
/// This is a *resolution* cap, not an interaction-order cap: the count of DISTINCT raw
/// features per tree is capped independently at [`MAX_ORDER`] (I1), so levels beyond the
/// `max_order`-th can only REUSE a raw feature already on the tree with a refined
/// threshold (spec §00 I1: "Reusing a raw feature is a valid lower-order refinement, not
/// an I1 violation"). A depth-`MAX_DEPTH` tree on `<= MAX_ORDER` raw features is therefore
/// still exactly a `<= MAX_ORDER`-order fANOVA function.
///
/// Ceiling rationale: `leaf_of_row` is `u8` and a leaf id packs ONE BIT PER LEVEL
/// (`leaf |= u8::from(low_bit(..)) << level`), so `depth <= 8` is the ABSOLUTE structural
/// ceiling — at depth 8 the id occupies bits `0..=7` exactly and depth 9 would silently
/// truncate. This constant now sits ON that ceiling (2026-08-26 `order-hi` lift, under
/// Ralph's explicit depth/order 5/6/7/8 authorization); it was `6` through the depth lift.
///
/// The phantom-leaf arithmetic that argued for 6 still holds and is still the real cost: a
/// depth-8 tree on `k` distinct raws realizes only `Π kᵢ` of 256 leaves, so most of the
/// leaf array is unreachable and `split::monotone_reachable` is what keeps the scan honest.
/// Depth beyond the order it serves buys RESOLUTION, never expressiveness.
///
/// This is the STRUCTURAL cap. The per-fit cap is the runtime
/// [`crate::InteractionPolicy::max_depth`] knob, which defaults to [`LEGACY_MAX_DEPTH`] —
/// raising it is opt-in, per the 2026-06-22 methodology decision (amended 2026-08-22)
/// that depth-3 remains the default.
pub const MAX_DEPTH: usize = 8;

/// The depth ceiling as of the DEPTH lift — the deepest tree an order-lift-era build
/// (`schema_version` 3/4) can decode AND validate.
///
/// Load-bearing on the wire exactly like [`LEGACY_MAX_DEPTH`]: the encoding is unchanged
/// (a tree's leaf count is recoverable in-band from `splits.len()`), so a deeper tree is a
/// pure reader-CAPABILITY claim and is stamped
/// [`crate::serialize::SCHEMA_VERSION_HIGH_ORDER`]. That makes an older build refuse it at
/// the version gate rather than at a confusing I1 error deep inside `Model::validate`.
pub const ORDER_LIFT_MAX_DEPTH: usize = 6;

/// The historical fixed depth, and the default value of the runtime `max_depth` knob.
///
/// Also the wire discriminator: a model all of whose trees are at most this deep encodes
/// identically to a pre-lift build and is stamped `schema_version = 2`.
pub const LEGACY_MAX_DEPTH: usize = 3;

/// `2^`[`MAX_DEPTH`] — the widest leaf array any oblivious tree can carry.
///
/// Most of it is phantom for a lifted tree (a depth-6 `(2,2,2)` tree realizes 27 of 64
/// cells); `split::monotone_reachable` models exactly which leaves a row can occupy.
pub const MAX_LEAVES: usize = 1usize << MAX_DEPTH;

/// The legacy leaf-array width — `2^`[`LEGACY_MAX_DEPTH`], the depth-3 tree's leaf count.
///
/// Load-bearing on the wire: a `schema_version <= 2` document encodes exactly this many
/// `f32` leaves per tree with no length prefix (bincode's positional encoding of the old
/// `leaves: [f32; 8]` field). See [`leaf_slots`] and `serialize::SCHEMA_VERSION`.
pub const LEGACY_LEAVES: usize = 1usize << LEGACY_MAX_DEPTH;

/// How many `f32` slots a depth-`depth` tree's leaf array occupies.
///
/// `2^depth` for a lifted tree, but never fewer than [`LEGACY_LEAVES`]: a depth-1/2/3
/// tree keeps the old 8-slot buffer with its zero tail, so its bincode encoding is
/// byte-for-byte what the pre-lift `[f32; 8]` field wrote. That is what lets the lift
/// ship WITHOUT invalidating existing `.bin` artifacts — the leaf count is recoverable
/// in-band from `splits.len()`, so the encoding needs no length prefix and no migration.
#[must_use]
pub const fn leaf_slots(depth: usize) -> usize {
    let exact = 1usize << depth;
    if exact < LEGACY_LEAVES {
        LEGACY_LEAVES
    } else {
        exact
    }
}

/// The widest per-level split-scan buffer: the leaf count of the DEEPEST level that is
/// still scanned, `2^(MAX_DEPTH-1)`. `best_level_split` allocates twelve `nl`-sized
/// accumulators per axis; sizing their SmallVec inline capacity here keeps a lifted fit
/// off the heap on the hottest loop in the booster (B6).
pub const MAX_SCAN_LEAVES: usize = MAX_LEAVES / 2;

/// The hard cap on a tree's DISTINCT raw-feature count — the fANOVA interaction order
/// (I1, spec §00/§01).
///
/// **This is a STRUCTURAL cap, not an explainability cap.** The product constraint that
/// matters is *exact lossless decomposition*, and that constraint is order-agnostic: a
/// tree on `k` distinct raw features collapses onto its `k`-axis merged-grid cube by
/// `leaf_index_for_tuple`, and `explain::purify` centres that cube axis by axis in
/// decreasing `|u|`. Both are written n-dimensionally (`Tensor` carries a
/// `shape: Vec<u32>`; `walk_extents` is an odometer of arbitrary rank), so the fANOVA
/// cascade at order 4 is the SAME algorithm, not a generalization of it.
///
/// What order 4 does cost is READABILITY, and that is priced elsewhere and deliberately:
/// the §07.4 table-budget prior charges the product of the realized per-axis extents, so
/// a fourth distinct feature multiplies a support's projected cell count rather than
/// adding to it; the interaction-gain hurdle doubles again at the 3→4 transition; and the
/// av37 evidence gate drops any table the held-out folds cannot pay for. A 4-way table
/// therefore appears only where the signal is large enough to clear all three.
///
/// Ceiling rationale: 8 is where the ARITHMETIC stops, not where the evidence does. Three
/// independent structures make 8 the natural wall, and they agree:
///
///  * a tree needs one level per distinct raw feature, so `max_order <= max_depth <=`
///    [`MAX_DEPTH`], which the `u8` leaf id pins at 8;
///  * a [`crate::explain::FactoredEffect`] box carries `2^k` purified corners, so an
///    order-8 box is 256 `f64` — the last order at which a single rank-1 region term is
///    still a small object;
///  * MANDATORY HEREDITY is the binding product constraint, and it is combinatorial. The
///    prune keeps a downward-closed order ideal, so ONE surviving order-`k` table drags in
///    its whole subset lattice: `2^k − 1` tables in total and `Σ_{j=4}^{k} C(k,j)` of them
///    at order `>= 4` — 6 for `k = 5`, 22 for `k = 6`, 64 for `k = 7`, **163 for `k = 8`**.
///    A readable bank ("a few dozen high-order tables") therefore admits order 5 freely,
///    order 6 for a single effect, and orders 7–8 not at all. That is a property of the
///    heredity contract, not of this constant — raising the constant does not buy a
///    readable order-8 table, it only makes one expressible.
///
/// The previous ceiling was 4 (the order lift). Ralph's 2026-08-26 authorization removed
/// the order cap as a product constraint outright — the constraint is exact decomposition,
/// mandatory pruning, and minimal signal-genuine survivors — so this is set at the
/// structural wall and the product bar is enforced where it belongs: the hurdle, the
/// shrunk table budget, the av37 evidence gate, the box budget, and heredity.
///
/// Exactness was never the binding constraint at any order and still is not: the fANOVA
/// purification cascade is the same algebra at 8 as at 3.
pub const MAX_ORDER: usize = 8;

/// The order ceiling as of the ORDER lift — the widest support an order-lift-era build
/// (`schema_version` 4) can decode AND validate.
///
/// Mirrors [`ORDER_LIFT_MAX_DEPTH`]. The order lift already made a factored box
/// length-carrying (`Vec<f64>` corners, `Vec<Vec<bool>>` masks), so orders 5–8 need NO
/// further encoding change — only the reader-capability claim.
pub const ORDER_LIFT_MAX_ORDER: usize = 4;

/// The historical fixed order cap, and the default value of the runtime `max_order` knob.
///
/// Also the wire discriminator: a model no tree of which uses more than this many distinct
/// raw features is stamped [`crate::serialize::SCHEMA_VERSION_UNLIFTED`] and is exactly
/// what a pre-lift build would have written. Mirrors [`LEGACY_MAX_DEPTH`].
pub const LEGACY_MAX_ORDER: usize = 3;

// ---------------------------------------------------------------------------------------
// The structural walls, asserted at COMPILE time.
//
// Until the high-order lift the crate carried none of these: the caps were prose in the doc
// comments above, and the depth cap in particular was one edit away from silently truncating
// every leaf id in the engine. Each assertion below names the exact mechanism that fails, so
// a future lift gets a compile error at the wall instead of a wrong rating table past it.
// ---------------------------------------------------------------------------------------

/// A leaf id is built as `leaf |= u8::from(low_bit(..)) << level` into a `u8` (see
/// `split::grow_oblivious_tree`'s `leaf_of_row`, and `scoring::ArenaTree::miss`, which packs
/// one missing-direction bit per level into a `u8` as well). `level` runs `0..depth`, so
/// depth 8 uses bits `0..=7` exactly and depth 9 shifts the top bit into oblivion — silently,
/// producing a tree that routes two distinct leaves to the same value.
const _: () = assert!(
    MAX_DEPTH <= 8,
    "MAX_DEPTH > 8 overflows the u8 leaf id (one bit per level) — widen leaf_of_row, \
     ArenaTree::miss and Model::lookup first"
);

/// A tree needs one shared level per DISTINCT raw feature, so an order above the depth cap is
/// unreachable by construction rather than merely disallowed. `i1_shape_ok` encodes the same
/// relation per-tree; this is the version that has to hold for the CONSTANTS.
const _: () = assert!(
    MAX_ORDER <= MAX_DEPTH,
    "MAX_ORDER > MAX_DEPTH is unreachable: an oblivious tree needs one level per distinct raw"
);

/// The wire ladder must stay ordered, or `required_schema_version` would stamp content below
/// the version that first admitted it and an older reader would accept what it cannot validate.
const _: () =
    assert!(LEGACY_MAX_DEPTH <= ORDER_LIFT_MAX_DEPTH && ORDER_LIFT_MAX_DEPTH <= MAX_DEPTH);
const _: () =
    assert!(LEGACY_MAX_ORDER <= ORDER_LIFT_MAX_ORDER && ORDER_LIFT_MAX_ORDER <= MAX_ORDER);

/// The SINGLE canonical I1 structural predicate (spec §2.5 / §3 / §13.2), written ONCE.
///
/// Before this consolidation the same predicate existed as four independent verbatim
/// copies (`ObliviousTree::try_new`, `Model::validate`, `explain::check_feature_budget`,
/// and the grow-time reachability guard). `scoring.rs`'s "THIRD, independent construction
/// ... MUST agree" hazard note applies with equal force here: four copies of one
/// invariant is four chances to drift. Every caller now routes through this.
///
/// Returns `true` iff `(depth, distinct_raw_features)` is a legal oblivious-tree shape:
/// `1 <= depth <= MAX_DEPTH` and `1 <= distinct <= min(depth, MAX_ORDER)`.
#[must_use]
pub const fn i1_shape_ok(depth: usize, distinct: usize) -> bool {
    i1_depth_ok(depth) && distinct >= 1 && distinct <= depth && distinct <= MAX_ORDER
}

/// [`i1_shape_ok`] restricted to the depth half of the predicate, for the callers that
/// check depth before they have counted distinct raw features.
#[must_use]
pub const fn i1_depth_ok(depth: usize) -> bool {
    depth >= 1 && depth <= MAX_DEPTH
}

/// How many DISTINCT raw features `splits` touches — the tree's fANOVA interaction order.
///
/// Deduped through `provenance`, not over `split.axis`: a multi-channel categorical maps
/// ONE raw feature onto several axes, so counting axes would over-state the order and, on
/// the wire-stamp path, would claim a model needs a newer reader than it does. A split
/// whose axis is absent from `provenance` is not counted; every caller of this either has
/// already range-checked its axes (`Model::validate`) or fails closed elsewhere.
#[must_use]
pub(crate) fn distinct_raw_count(splits: &[Split], provenance: &[AxisProvenance]) -> usize {
    let mut distinct: SmallVec<[u32; MAX_ORDER]> = SmallVec::new();
    for s in splits {
        if let Some(prov) = provenance.get(s.axis as usize) {
            if !distinct.contains(&prov.raw.0) {
                distinct.push(prov.raw.0);
            }
        }
    }
    distinct.len()
}

/// The SINGLE canonical missing low/left bit (spec §2.5 / §06.2, R-MISSING). The
/// reserved missing bin (bin 0) routes by its learned `missing_left`; every other
/// bin routes `bin <= bin_le`. Written ONCE and used identically at split evaluation,
/// the sample→leaf update ([`split::grow_oblivious_tree`]), tree scoring
/// ([`ObliviousTree::lookup`]), and table accumulation (§08) — agreement here is what
/// makes the tree, the purified tables, and the Shapley sum equal (I2 / ThreeWayEqual).
#[must_use]
pub(crate) fn low_bit(bin: u8, bin_le: u8, missing_left: bool) -> bool {
    if bin == 0 {
        missing_left
    } else {
        bin <= bin_le
    }
}

/// The Exact / Approximate firewall (spec §3). An `Exact` model passes all five
/// I2 checks and may export rating tables; any operation that cannot preserve them
/// flips the model to `Approximate { reason }` and refuses an `Exact` export. This
/// typed wall is the structural defense against death-by-a-thousand-cuts.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ExactnessMode {
    /// Passes all five invariant checks; exports an exact `TableBank`.
    #[default]
    Exact,
    /// Cannot pass the checks; `reason` explains why (e.g. nonlinear calibration).
    Approximate {
        /// Why the model is not exactly decomposable.
        reason: String,
    },
}

/// One shared level test of an oblivious tree (spec §2.5): `bin <= bin_le`.
///
/// `axis` is `u32` (fixed-width: serialized; `usize` would break cross-platform
/// byte-equality / the `wasm32` smoke build). `missing_left` is the explicit learned
/// default direction — the reserved missing bin (bin 0) routes left when `true`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Split {
    /// Index of the axis this level tests.
    pub axis: u32,
    /// Inclusive upper bin for the "low" (left) child.
    pub bin_le: u8,
    /// Learned default direction for the reserved missing bin (bin 0).
    pub missing_left: bool,
}

/// An oblivious tree (spec §2.5): one shared `(axis, bin_le)` test per level, at most
/// [`MAX_ORDER`] DISTINCT raw features, `2^depth` leaf values. Interaction order is the
/// number of distinct raw features, not the number of split levels: repeated levels
/// refine lower-order main/pair surfaces exactly, so a depth-6 tree on 3 raw features is
/// still exactly a 3rd-order fANOVA function.
///
/// `Serialize`/`Deserialize` are hand-written below rather than derived, so that a
/// depth-`<=3` tree's bytes are IDENTICAL to what the pre-lift `leaves: [f32; 8]` field
/// wrote. See [`leaf_slots`].
#[derive(Debug, Clone, PartialEq)]
pub struct ObliviousTree {
    /// `1..=`[`MAX_DEPTH`] level tests, in test order (bit 0 = level 0).
    pub splits: Vec<Split>,
    /// Leaf values; index = `Σ_l b_l << l`. Length is [`leaf_slots`]`(depth)`, so a
    /// depth-`<3` tree keeps a zeroed tail out to [`LEGACY_LEAVES`].
    pub leaves: Vec<f32>,
    /// `splits.len()` as `u8`, in `1..=`[`MAX_DEPTH`].
    pub depth: u8,
}

const OBLIVIOUS_TREE_FIELDS: &[&str] = &["splits", "leaves", "depth"];

/// Serializes `leaves` as a FIXED-LENGTH tuple, not a sequence.
///
/// bincode writes a length prefix for a `Vec` but nothing for a tuple, and the leaf count
/// is recoverable in-band from `splits.len()` (which precedes it), so the tuple encoding
/// reproduces the pre-lift `[f32; 8]` bytes exactly for a depth-`<=3` tree. Self-describing
/// formats (JSON) see the same array either way.
struct LeafTuple<'a>(&'a [f32]);

impl serde::Serialize for LeafTuple<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeTuple;
        let mut t = s.serialize_tuple(self.0.len())?;
        for v in self.0 {
            t.serialize_element(v)?;
        }
        t.end()
    }
}

/// `DeserializeSeed` for [`LeafTuple`]: reads exactly `self.0` `f32`s with no length
/// prefix, mirroring the serializer.
struct LeafTupleSeed(usize);

impl<'de> serde::de::DeserializeSeed<'de> for LeafTupleSeed {
    type Value = Vec<f32>;

    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<Vec<f32>, D::Error> {
        struct V(usize);
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = Vec<f32>;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{} oblivious-tree leaf values", self.0)
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut a: A,
            ) -> Result<Vec<f32>, A::Error> {
                let mut out = Vec::with_capacity(self.0);
                for i in 0..self.0 {
                    match a.next_element::<f32>()? {
                        Some(v) => out.push(v),
                        None => return Err(serde::de::Error::invalid_length(i, &self)),
                    }
                }
                Ok(out)
            }
        }
        d.deserialize_tuple(self.0, V(self.0))
    }
}

impl serde::Serialize for ObliviousTree {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("ObliviousTree", 3)?;
        st.serialize_field("splits", &self.splits)?;
        st.serialize_field("leaves", &LeafTuple(&self.leaves))?;
        st.serialize_field("depth", &self.depth)?;
        st.end()
    }
}

/// Shared tail of both deserialize paths: cross-check `depth`, I1's depth bound, and the
/// leaf-slot count, so a truncated or hand-edited document fails closed on load.
fn finish_oblivious_tree<E: serde::de::Error>(
    splits: Vec<Split>,
    leaves: Vec<f32>,
    depth: u8,
) -> Result<ObliviousTree, E> {
    if usize::from(depth) != splits.len() {
        return Err(E::custom(format!(
            "oblivious tree depth {depth} != splits len {}",
            splits.len()
        )));
    }
    if !i1_depth_ok(splits.len()) {
        return Err(E::custom(format!(
            "oblivious tree depth {} outside 1..={MAX_DEPTH}",
            splits.len()
        )));
    }
    let want = leaf_slots(splits.len());
    if leaves.len() != want {
        return Err(E::custom(format!(
            "oblivious tree at depth {} needs {want} leaf slots, got {}",
            splits.len(),
            leaves.len()
        )));
    }
    Ok(ObliviousTree {
        splits,
        leaves,
        depth,
    })
}

impl<'de> serde::Deserialize<'de> for ObliviousTree {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct TreeVisitor;
        impl<'de> serde::de::Visitor<'de> for TreeVisitor {
            type Value = ObliviousTree;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("struct ObliviousTree")
            }

            /// The bincode (positional, non-self-describing) path. `splits` precedes
            /// `leaves`, so the leaf count is known before the leaf field is read — which
            /// is exactly why no length prefix is needed on the wire, and why a
            /// `schema_version = 2` artifact written before the lift still loads here.
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut a: A,
            ) -> Result<ObliviousTree, A::Error> {
                let splits: Vec<Split> = a
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                if !i1_depth_ok(splits.len()) {
                    return Err(serde::de::Error::custom(format!(
                        "oblivious tree depth {} outside 1..={MAX_DEPTH}",
                        splits.len()
                    )));
                }
                let leaves = a
                    .next_element_seed(LeafTupleSeed(leaf_slots(splits.len())))?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                let depth: u8 = a
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
                finish_oblivious_tree(splits, leaves, depth)
            }

            /// The self-describing (JSON) path: fields may arrive in any order, and the
            /// leaf array delimits itself, so it reads as a plain sequence.
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut a: A,
            ) -> Result<ObliviousTree, A::Error> {
                let mut splits: Option<Vec<Split>> = None;
                let mut leaves: Option<Vec<f32>> = None;
                let mut depth: Option<u8> = None;
                while let Some(key) = a.next_key::<String>()? {
                    match key.as_str() {
                        "splits" => splits = Some(a.next_value()?),
                        "leaves" => leaves = Some(a.next_value()?),
                        "depth" => depth = Some(a.next_value()?),
                        other => {
                            return Err(serde::de::Error::unknown_field(
                                other,
                                OBLIVIOUS_TREE_FIELDS,
                            ))
                        }
                    }
                }
                finish_oblivious_tree(
                    splits.ok_or_else(|| serde::de::Error::missing_field("splits"))?,
                    leaves.ok_or_else(|| serde::de::Error::missing_field("leaves"))?,
                    depth.ok_or_else(|| serde::de::Error::missing_field("depth"))?,
                )
            }
        }
        d.deserialize_struct("ObliviousTree", OBLIVIOUS_TREE_FIELDS, TreeVisitor)
    }
}

impl ObliviousTree {
    /// Score one row given its per-axis bin ids, returning the leaf value (spec §2.5).
    ///
    /// Uses the SINGLE canonical missing low-bit:
    /// `low = if bin == 0 { missing_left } else { bin <= bin_le }`. This exact form
    /// is shared by split evaluation, the packed scoring kernel, and these gates —
    /// the basis of tree/table equality (I2).
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if a split's `axis` is absent from `row_bins`;
    /// [`PbError::Internal`] if the folded leaf index escapes `0..2^depth` (impossible —
    /// it is built from exactly `depth <= MAX_DEPTH` bits — but checked, not indexed raw).
    /// spec §13.132's no-panic bound is re-derived here as `idx = Σ_l b_l << l ∈ 0..2^depth`
    /// with `depth <= MAX_DEPTH`, superseding the depth-3-specific `idx ∈ 0..8`.
    pub fn lookup(&self, row_bins: &[u8]) -> Result<f32, PbError> {
        let mut idx = 0usize;
        for (level, split) in self.splits.iter().enumerate() {
            let bin = *row_bins
                .get(split.axis as usize)
                .ok_or_else(|| PbError::ShapeMismatch {
                    what: format!("row has no axis {} for tree lookup", split.axis),
                })?;
            let bit = usize::from(low_bit(bin, split.bin_le, split.missing_left));
            idx |= bit << level;
        }
        self.leaves
            .get(idx)
            .copied()
            .ok_or_else(|| PbError::Internal {
                what: format!(
                    "oblivious leaf index {idx} escaped 0..{}",
                    self.leaves.len()
                ),
            })
    }

    /// Construct a tree, enforcing I1 at the type boundary (spec §2.5 / §3): `depth`
    /// (= `splits.len()`) must be in `1..=3`, and the count of DISTINCT raw features
    /// across the splits (via `provenance`) must be in `1..=depth` and never exceed 3.
    /// Repeated raw features are valid lower-order refinements. `leaves[depth.pow2()..]`
    /// are the unused tail.
    ///
    /// # Errors
    /// [`Invariant::FeatureBudget`] (as [`PbError::InvariantViolated`]) if the depth
    /// or distinct-raw-feature budget is violated; [`PbError::Internal`] if a split
    /// names an axis absent from `provenance`.
    pub fn try_new(
        splits: Vec<Split>,
        leaves: Vec<f32>,
        provenance: &[AxisProvenance],
    ) -> Result<Self, PbError> {
        let depth = splits.len();
        if !i1_depth_ok(depth) {
            return Err(PbError::invariant(Invariant::FeatureBudget));
        }
        let mut distinct: SmallVec<[u32; MAX_ORDER]> = SmallVec::new();
        for s in &splits {
            let raw = provenance
                .get(s.axis as usize)
                .ok_or_else(|| PbError::Internal {
                    what: format!("split axis {} absent from provenance", s.axis),
                })?
                .raw
                .0;
            if !distinct.contains(&raw) {
                distinct.push(raw);
            }
        }
        if !i1_shape_ok(depth, distinct.len()) {
            return Err(PbError::invariant(Invariant::FeatureBudget));
        }
        // The leaf-slot count is load-bearing ON THE WIRE: `Serialize` writes `leaves.len()`
        // f32s with NO length prefix and `Deserialize` reads exactly `leaf_slots(depth)`, so a
        // short or long leaf vector would emit a MIS-FRAMED bincode stream that decodes every
        // following tree from the wrong offset — with no error at write time. Enforce it here
        // and in `Model::validate`, mirroring the check the decoder already performs on load.
        let want = leaf_slots(depth);
        if leaves.len() != want {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "tree at depth {depth} needs {want} leaf slots, got {}",
                    leaves.len()
                ),
            });
        }
        Ok(Self {
            splits,
            leaves,
            depth: depth as u8,
        })
    }
}

pub(crate) fn tree_split_columns<'a>(
    tree: &ObliviousTree,
    columns: &'a [Vec<u8>],
) -> Result<Vec<&'a [u8]>, PbError> {
    let mut out = Vec::with_capacity(tree.splits.len());
    for split in &tree.splits {
        out.push(
            columns
                .get(split.axis as usize)
                .ok_or_else(|| PbError::Internal {
                    what: "tree scorer: split axis out of range".into(),
                })?
                .as_slice(),
        );
    }
    Ok(out)
}

pub(crate) fn tree_leaf_index_for_row_with_columns(
    tree: &ObliviousTree,
    columns: &[&[u8]],
    row: usize,
) -> Result<usize, PbError> {
    if columns.len() != tree.splits.len() {
        return Err(PbError::Internal {
            what: "tree scorer: split column count mismatch".into(),
        });
    }
    let mut idx = 0usize;
    for (level, (split, col)) in tree.splits.iter().zip(columns).enumerate() {
        let bin = *col.get(row).ok_or_else(|| PbError::Internal {
            what: "tree scorer: row out of split column".into(),
        })?;
        idx |= usize::from(low_bit(bin, split.bin_le, split.missing_left)) << level;
    }
    if idx >= MAX_LEAVES {
        return Err(PbError::Internal {
            what: format!("tree scorer: leaf index {idx} escaped 0..{MAX_LEAVES}"),
        });
    }
    Ok(idx)
}

pub(crate) fn tree_value_for_row_with_columns(
    tree: &ObliviousTree,
    columns: &[&[u8]],
    row: usize,
) -> Result<f32, PbError> {
    let idx = tree_leaf_index_for_row_with_columns(tree, columns, row)?;
    tree.leaves
        .get(idx)
        .copied()
        .ok_or_else(|| PbError::Internal {
            what: format!(
                "tree scorer: leaf index {idx} escaped 0..{}",
                tree.leaves.len()
            ),
        })
}

/// The per-`(leaf, axis, bin)` gradient/hessian histogram accumulator (spec §06.3),
/// struct-of-arrays in `[leaf][axis][bin]` row-major order with a **uniform `n_bins`
/// stride** (the max grid bins over the built axes; shorter axes leave their high
/// bins zeroed). `count` stays `u32` (a bin holds at most `n_rows <= u32::MAX` rows).
///
/// The default path uses `f64` accumulators and earns determinism from a FIXED-ORDER fold
/// (feature-parallel, sequential within each axis — [`hist::build_histogram`]).
/// FLAG (spec reconciliation): §2.3/§06.3 specify `i64` accumulators, but that is the
/// *quantized* path. The M5-QHIST lever implements that integer-associative path
/// behind [`HistPrecision::QuantizedI32`], while [`HistPrecision::FullF64`] keeps
/// this `f64` accumulator as the default green-spine representation.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Hist {
    /// Per-cell gradient sums (`[leaf][axis][bin]`, row-major).
    pub g: Vec<f64>,
    /// Per-cell hessian sums.
    pub h: Vec<f64>,
    /// Per-cell sample-weight sums `Σw` (§07 credibility `min_weight_sum_in_leaf`).
    /// Accumulated in the same fixed-order f64 fold as `g`/`h` (thread-count independent).
    /// Unit when no `sample_weight` is supplied (so it then equals `count` as `f64`).
    pub wsum: Vec<f64>,
    /// Per-cell row counts.
    pub count: Vec<u32>,
    /// Number of leaves at this level (`2^depth`).
    pub n_leaves: usize,
    /// Number of axes (built features) in this histogram.
    pub n_axes: usize,
    /// Uniform per-axis bin stride (max grid bins over the built axes).
    pub n_bins: usize,
}

impl Hist {
    /// Try to allocate a zeroed histogram with the given shape. `g`/`h`/`count`
    /// each hold `n_leaves · n_axes · n_bins` cells.
    ///
    /// # Errors
    /// [`PbError::Internal`] if the shape arithmetic overflows or if the backing
    /// buffers cannot be reserved.
    pub fn try_zeros(n_leaves: usize, n_axes: usize, n_bins: usize) -> Result<Self, PbError> {
        let cells = Self::checked_cell_count(n_leaves, n_axes, n_bins)?;
        Ok(Self {
            g: Self::try_zeroed_vec(cells, "histogram g")?,
            h: Self::try_zeroed_vec(cells, "histogram h")?,
            wsum: Self::try_zeroed_vec(cells, "histogram wsum")?,
            count: Self::try_zeroed_vec(cells, "histogram count")?,
            n_leaves,
            n_axes,
            n_bins,
        })
    }

    /// The flat row-major offset of cell `(leaf, axis, bin)`, or `None` if any index
    /// is out of range (so callers stay panic-free without raw indexing).
    #[must_use]
    pub fn offset(&self, leaf: usize, axis: usize, bin: usize) -> Option<usize> {
        if leaf >= self.n_leaves || axis >= self.n_axes || bin >= self.n_bins {
            return None;
        }
        leaf.checked_mul(self.n_axes)?
            .checked_add(axis)?
            .checked_mul(self.n_bins)?
            .checked_add(bin)
    }

    /// `(n_leaves, n_axes, n_bins)` — the shape triple, for equality checks.
    #[must_use]
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.n_leaves, self.n_axes, self.n_bins)
    }

    /// Total number of cells (`n_leaves · n_axes · n_bins`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.g.len()
    }

    /// `true` if the histogram has no cells.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.g.is_empty()
    }

    pub(crate) fn checked_cell_count(
        n_leaves: usize,
        n_axes: usize,
        n_bins: usize,
    ) -> Result<usize, PbError> {
        n_leaves
            .checked_mul(n_axes)
            .and_then(|cells| cells.checked_mul(n_bins))
            .ok_or_else(|| PbError::Internal {
                what: "histogram shape overflows usize".into(),
            })
    }

    pub(crate) fn try_zeroed_vec<T>(cells: usize, what: &'static str) -> Result<Vec<T>, PbError>
    where
        T: Clone + Default,
    {
        let mut out = Vec::new();
        out.try_reserve_exact(cells)
            .map_err(|_| PbError::Internal {
                what: format!("{what} allocation failed"),
            })?;
        out.resize(cells, T::default());
        Ok(out)
    }
}

/// The scale factors mapping full-precision g/h onto quantized integers (spec §2.3),
/// used by the [`HistPrecision::QuantizedI32`] path.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GradScale {
    /// Multiplier applied to gradients before integer rounding.
    pub g_scale: f32,
    /// Multiplier applied to hessians before integer rounding.
    pub h_scale: f32,
}

/// Quantized integer g/h for associative, order-independent histogram sums (spec
/// §2.3). This is the M5-QHIST representation: split structure may be searched on
/// quantized histograms, while leaf values are refit from full-precision
/// [`crate::loss::GradHess`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct QuantGradHess {
    /// Quantized per-row gradients.
    pub g_q: Vec<i32>,
    /// Quantized per-row hessians.
    pub h_q: Vec<i32>,
    /// The scale factors used to quantize.
    pub scale: GradScale,
}

/// Histogram accumulator precision (§06/§11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistPrecision {
    /// Full-precision f64 histogram sums (v1 green-spine default).
    FullF64,
    /// Quantized i32 per-row gradients/hessians accumulated as i64, then
    /// dequantized for the existing split scanner. Leaves are still refit from full
    /// precision.
    QuantizedI32,
}

impl Default for HistPrecision {
    fn default() -> Self {
        Self::FullF64
    }
}

/// Model-level metadata so a `Model` can serve and export categoricals + classifiers
/// without the caller re-supplying anything (spec §2.6, R-SCHEMA). Serialized with
/// the `Model`; `schema_version` covers it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSchema {
    /// Human-readable feature names (parallel to provenance).
    pub feature_names: Vec<String>,
    /// Per-axis kinds (reuses `AxisKind`).
    pub feature_kinds: Vec<AxisKind>,
    /// Frozen full-data categorical encoders (owned by §04).
    pub cat_encoders: CatEncoderStore,
    /// Class labels for a classifier; `None` for regression.
    pub class_labels: Option<Vec<String>>,
    /// The trained objective (link + loss + Tweedie power).
    pub objective: ObjectiveTag,
}

/// One per-support raw (pre-purification) correction table at merged-cell resolution.
///
/// The §G1 fully-corrective cell-basis refit solves a per-cell additive delta on the
/// model's realized supports (mains + pairs), fit to the bagged out-of-bag residual on
/// the fANOVA cell basis. The delta is stored RAW (unpurified) at the merged-grid
/// resolution so it can be (a) added directly to the tree score at serve and (b) folded
/// into the raw effects before [`purify`](crate::explain) during decomposition. Because
/// purify is lossless, `trees + delta == purified(trees + delta)`, so G0 (the exact
/// `ensemble == bank.score` reconstruction) holds by construction and the delta is
/// re-purified into the canonical ≤order-d tables (its marginals flow to lower orders).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrectionTable {
    /// Model axis ids of this support, sorted ascending (length = order, 1 or 2).
    pub axes: Vec<u32>,
    /// Merged cells per axis (parallel to `axes`); `Π shape == values.len()`.
    pub shape: Vec<u32>,
    /// Per axis (parallel to `axes`): model bin → merged cell id. `bin_to_cell[k].len()`
    /// equals that axis's model `n_bins`; every entry is `< shape[k]`.
    pub bin_to_cell: Vec<Vec<u32>>,
    /// Raw additive delta, row-major over the merged cells (last axis fastest),
    /// length `Π shape`.
    pub values: Vec<f64>,
}

/// An optional cell-basis correction carried on a [`Model`]: a set of per-support raw
/// deltas added to the tree score at serve and folded into the decomposition before
/// purification. Empty `tables` is equivalent to `None`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrectionBank {
    /// One table per corrected realized support (mains + pairs).
    pub tables: Vec<CorrectionTable>,
}

/// The trained ensemble (spec §2.6): intercept + weighted oblivious trees, the
/// shared binning grids, provenance, the loss/link, an exactness flag, and the
/// serve/export schema.
///
/// Inference: `raw(x) = f0 + offset + Σ alpha_t · tree_t.lookup(x) + correction(x)`.
///
/// `PartialEq` is hand-written (see below), NOT derived: `bag_spans` is excluded so the
/// §10.7 round-trip contract (`decode(encode(m)) == m`) holds for bagged models too —
/// the field is `#[serde(skip)]` by design (runtime-only introspection aid, never part
/// of a model's persistent identity), and a derived `PartialEq` would compare it anyway.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    /// `link(weighted mean)` — a scalar intercept, never "tree 0".
    pub f0: f32,
    /// `(weight alpha, tree)` pairs; alphas allow DART/Nesterov/ensemble mixes.
    pub trees: Vec<(f32, ObliviousTree)>,
    /// Shared per-axis binning grids.
    pub grids: Vec<BorderGrid>,
    /// Per-axis provenance (maps each axis to its raw feature — drives I1).
    pub provenance: Vec<AxisProvenance>,
    /// The inverse-link family.
    pub link: Link,
    /// Exact / Approximate firewall state (§3).
    pub mode: ExactnessMode,
    /// Serve/export metadata (cats + classifier labels + objective).
    pub schema: ModelSchema,
    /// Monotone wire-version covering `Model` AND `schema`.
    pub schema_version: u32,
    /// Optional cell-basis correction (the §G1 adaptive fANOVA-cell refit). Added to the
    /// tree score at serve and folded into the raw effects before purify — G0-exact.
    #[serde(default)]
    pub correction: Option<CorrectionBank>,
    /// Runtime-only per-bag `trees` spans (`start..end`, in bag order) recorded by the
    /// outer-bag soup. Enables per-bag bank introspection (honest replicate values behind
    /// the averaged tables). Never serialized: `None` on single fits and loaded models.
    #[serde(skip)]
    pub bag_spans: Option<Vec<(u32, u32)>>,
    /// Runtime-only intercept of each standalone bag, in the order of `bag_spans`.
    /// Preserves honest out-of-bag predictions; omitted from wire identity.
    #[serde(skip)]
    pub bag_intercepts: Option<Vec<f32>>,
    /// Runtime-only per-bag IN-BAG row membership over the FIT rows (`bag_in_bag[b][r]` is
    /// true iff fit row `r` was drawn into bag `b`'s training sample, in the same bag order
    /// as [`Model::bag_spans`]), recorded by the outer-bag soup alongside the spans.
    ///
    /// Its complement is each bag's OUT-OF-BAG set — rows that bag never saw — which is the
    /// prune guard's zero-cost honest evidence (no carve): the guard scores the deployed
    /// keep-set against the full bank on rows that were out of bag, so the fit itself is
    /// untouched. The draw is deterministic in `(seed, bag index, n_rows, strata)` (see
    /// `subagging_rows_dispatch`), so this is the drawn membership itself, not a replay.
    ///
    /// Under a shared `fixed_holdout` the appended holdout rows count as IN bag (the bag
    /// early-stopped on them), so they never enter anyone's OOB set. Bootstrap multiplicity
    /// is not retained — membership is a set, which is all the OOB complement needs.
    ///
    /// Never serialized: `None` on single fits (`n_bags == 1`), and on loaded models.
    #[serde(skip)]
    pub bag_in_bag: Option<Vec<Vec<bool>>>,
    /// Runtime-only report from the [`Config::max_delta_step_gated`] rate-collapse detector
    /// (`None` when the gate was not armed for this fit, or on a loaded model). Like
    /// [`Self::bag_spans`] this is a `#[serde(skip)]` introspection aid excluded from
    /// `PartialEq`: it must NOT enter model identity, or "the gate stayed silent ⇒
    /// bit-identical model" would stop being checkable by comparing models.
    #[serde(skip)]
    pub delta_step_gate: Option<DeltaStepGateReport>,
    /// Runtime-only per-bag boosting report (round counts, stop reasons, optional history),
    /// one entry per bag in bag order. `None` on loaded models and on non-fit constructions.
    #[serde(skip)]
    pub fit_report: Option<Vec<BagFitReport>>,
}

impl PartialEq for Model {
    /// Structural equality over every field EXCEPT `bag_spans`/`bag_in_bag`/`delta_step_gate`
    /// (see the struct doc): all three are runtime-only introspection aids, never part of a
    /// model's identity.
    fn eq(&self, other: &Self) -> bool {
        self.f0 == other.f0
            && self.trees == other.trees
            && self.grids == other.grids
            && self.provenance == other.provenance
            && self.link == other.link
            && self.mode == other.mode
            && self.schema == other.schema
            && self.schema_version == other.schema_version
            && self.correction == other.correction
    }
}

impl Model {
    /// The MINIMUM `schema_version` a reader needs to decode this model correctly.
    ///
    /// A LADDER over the two structural caps, each rung the version that first admitted it:
    ///
    /// | tree shape | stamp |
    /// |---|---|
    /// | `depth <= 3`, `order <= 3` | [`crate::serialize::SCHEMA_VERSION_UNLIFTED`] (2) |
    /// | `depth <= 6`, `order <= 3` | [`crate::serialize::SCHEMA_VERSION_DEPTH_LIFTED`] (3) |
    /// | `depth <= 6`, `order <= 4` | [`crate::serialize::SCHEMA_VERSION_ORDER_LIFTED`] (4) |
    /// | `depth <= 8`, `order <= 8` | [`crate::serialize::SCHEMA_VERSION_HIGH_ORDER`] (5) |
    ///
    /// The strongest claim any tree makes wins. See [`crate::serialize::SCHEMA_VERSION`]
    /// for why the stamp is the required minimum rather than the newest version this build
    /// knows — that is what keeps an order-3/depth-3 fit byte-identical forever.
    #[must_use]
    pub fn required_schema_version(&self) -> u32 {
        // Every rung above 2 is the SAME encoding on a wider range — `splits` is a
        // length-prefixed `Vec` and the leaf count is recoverable in-band from
        // `splits.len()` — so each stamp is purely a reader-capability claim: the bytes
        // decode fine in an older build, but its I1 check would (correctly) refuse them,
        // and it must refuse LOUDLY at the version gate rather than at a confusing
        // invariant error deep in `validate`.
        let mut required = crate::serialize::SCHEMA_VERSION_UNLIFTED;
        for (_, tree) in &self.trees {
            let order = distinct_raw_count(&tree.splits, &self.provenance);
            let depth = usize::from(tree.depth);
            if order > ORDER_LIFT_MAX_ORDER || depth > ORDER_LIFT_MAX_DEPTH {
                // The top rung: nothing stronger exists, so short-circuit.
                return crate::serialize::SCHEMA_VERSION_HIGH_ORDER;
            }
            if order > LEGACY_MAX_ORDER {
                required = required.max(crate::serialize::SCHEMA_VERSION_ORDER_LIFTED);
            }
            if depth > LEGACY_MAX_DEPTH {
                required = required.max(crate::serialize::SCHEMA_VERSION_DEPTH_LIFTED);
            }
        }
        required
    }

    /// Validate model structure after construction or deserialization.
    ///
    /// This is the §10 load gate: it re-checks fixed-width schema consistency, split
    /// axis bounds, finite scalar payloads, and the I1 feature-budget shape before a
    /// decoded model can be scored or exported.
    ///
    /// # Errors
    /// [`PbError::Serialization`] for a schema-version mismatch;
    /// [`PbError::ShapeMismatch`] for inconsistent parallel metadata;
    /// [`PbError::InvalidInput`] for malformed grids or non-finite scalars;
    /// [`PbError::InvariantViolated`] for an I1 feature-budget violation.
    pub fn validate(&self) -> Result<(), PbError> {
        let required = self.required_schema_version();
        if self.schema_version > crate::serialize::SCHEMA_VERSION || self.schema_version < required
        {
            return Err(PbError::Serialization(format!(
                "model schema_version {} outside {}..={} for this model's contents",
                self.schema_version,
                required,
                crate::serialize::SCHEMA_VERSION
            )));
        }
        if !self.f0.is_finite() {
            return Err(PbError::InvalidInput {
                what: format!("model f0 must be finite, got {}", self.f0),
            });
        }
        if self.grids.len() != self.provenance.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "model grids len {} != provenance len {}",
                    self.grids.len(),
                    self.provenance.len()
                ),
            });
        }
        if self.schema.feature_names.len() != self.grids.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "schema feature_names len {} != grid count {}",
                    self.schema.feature_names.len(),
                    self.grids.len()
                ),
            });
        }
        if self.schema.feature_kinds.len() != self.grids.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "schema feature_kinds len {} != grid count {}",
                    self.schema.feature_kinds.len(),
                    self.grids.len()
                ),
            });
        }
        if self.schema.objective.link != self.link {
            return Err(PbError::InvalidInput {
                what: "schema objective link does not match model link".into(),
            });
        }
        for (axis, (prov, kind)) in self
            .provenance
            .iter()
            .zip(&self.schema.feature_kinds)
            .enumerate()
        {
            if prov.kind != *kind {
                return Err(PbError::ShapeMismatch {
                    what: format!(
                        "schema feature_kinds[{axis}] {:?} != provenance kind {:?}",
                        kind, prov.kind
                    ),
                });
            }
            if let AxisKind::CategoricalTS { encoding } = prov.kind {
                if self.schema.cat_encoders.get(encoding, prov.raw).is_err() {
                    return Err(PbError::InvalidInput {
                        what: format!(
                            "categorical axis {axis} references missing encoder {:?}/{:?}",
                            prov.raw, encoding
                        ),
                    });
                }
            }
        }
        for (axis, grid) in self.grids.iter().enumerate() {
            if grid.missing_bin != 0 {
                return Err(PbError::InvalidInput {
                    what: format!(
                        "grid {axis} missing_bin must be 0, got {}",
                        grid.missing_bin
                    ),
                });
            }
            if grid.n_bins == 0 || grid.n_bins > 255 {
                return Err(PbError::InvalidInput {
                    what: format!("grid {axis} n_bins must be in 1..=255, got {}", grid.n_bins),
                });
            }
            let expected_bins =
                u16::try_from(grid.borders.len().checked_add(2).ok_or_else(|| {
                    PbError::Internal {
                        what: "grid border count overflow".into(),
                    }
                })?)
                .map_err(|_| PbError::InvalidInput {
                    what: format!("grid {axis} has too many borders"),
                })?;
            if grid.n_bins != expected_bins && !(grid.n_bins == 1 && grid.borders.is_empty()) {
                return Err(PbError::InvalidInput {
                    what: format!(
                        "grid {axis} n_bins {} inconsistent with {} borders",
                        grid.n_bins,
                        grid.borders.len()
                    ),
                });
            }
            for (i, &border) in grid.borders.iter().enumerate() {
                if !border.is_finite() {
                    return Err(PbError::InvalidInput {
                        what: format!("grid {axis} border {i} must be finite"),
                    });
                }
            }
            for pair in grid.borders.windows(2) {
                if let [a, b] = pair {
                    if a >= b {
                        return Err(PbError::InvalidInput {
                            what: format!("grid {axis} borders must be strictly ascending"),
                        });
                    }
                }
            }
        }
        for (tree_idx, (alpha, tree)) in self.trees.iter().enumerate() {
            if !alpha.is_finite() {
                return Err(PbError::InvalidInput {
                    what: format!("tree {tree_idx} alpha must be finite, got {alpha}"),
                });
            }
            let depth = usize::from(tree.depth);
            if !i1_depth_ok(depth) || tree.splits.len() != depth {
                return Err(PbError::invariant(Invariant::FeatureBudget));
            }
            let mut distinct: SmallVec<[u32; MAX_ORDER]> = SmallVec::new();
            for split in &tree.splits {
                let axis = split.axis as usize;
                let prov = self
                    .provenance
                    .get(axis)
                    .ok_or_else(|| PbError::ShapeMismatch {
                        what: format!("tree {tree_idx} split axis {axis} absent from provenance"),
                    })?;
                let grid = self.grids.get(axis).ok_or_else(|| PbError::ShapeMismatch {
                    what: format!("tree {tree_idx} split axis {axis} absent from grids"),
                })?;
                if u16::from(split.bin_le) >= grid.n_bins {
                    return Err(PbError::InvalidInput {
                        what: format!(
                            "tree {tree_idx} split bin_le {} outside grid {axis} n_bins {}",
                            split.bin_le, grid.n_bins
                        ),
                    });
                }
                if !distinct.contains(&prov.raw.0) {
                    distinct.push(prov.raw.0);
                }
            }
            if !i1_shape_ok(depth, distinct.len()) {
                return Err(PbError::invariant(Invariant::FeatureBudget));
            }
            // See `ObliviousTree::try_new`: the leaf-slot count frames the wire.
            let want_leaves = leaf_slots(depth);
            if tree.leaves.len() != want_leaves {
                return Err(PbError::ShapeMismatch {
                    what: format!(
                        "tree {tree_idx} at depth {depth} needs {want_leaves} leaf slots, got {}",
                        tree.leaves.len()
                    ),
                });
            }
            for (leaf, &value) in tree.leaves.iter().enumerate() {
                if !value.is_finite() {
                    return Err(PbError::InvalidInput {
                        what: format!("tree {tree_idx} leaf {leaf} must be finite, got {value}"),
                    });
                }
            }
        }
        if let Some(bank) = &self.correction {
            for (t, table) in bank.tables.iter().enumerate() {
                let order = table.axes.len();
                if !(1..=MAX_ORDER).contains(&order) {
                    return Err(PbError::InvalidInput {
                        what: format!("correction table {t} order {order} not in 1..={MAX_ORDER}"),
                    });
                }
                if table.shape.len() != order || table.bin_to_cell.len() != order {
                    return Err(PbError::ShapeMismatch {
                        what: format!(
                            "correction table {t}: axes {order}, shape {}, bin_to_cell {}",
                            table.shape.len(),
                            table.bin_to_cell.len()
                        ),
                    });
                }
                for (k, &axis) in table.axes.iter().enumerate() {
                    if k > 0 && axis <= table.axes[k - 1] {
                        return Err(PbError::InvalidInput {
                            what: format!("correction table {t} axes not strictly ascending"),
                        });
                    }
                    let grid =
                        self.grids
                            .get(axis as usize)
                            .ok_or_else(|| PbError::ShapeMismatch {
                                what: format!("correction table {t} axis {axis} absent from grids"),
                            })?;
                    let map = &table.bin_to_cell[k];
                    if map.len() != usize::from(grid.n_bins) {
                        return Err(PbError::ShapeMismatch {
                            what: format!(
                                "correction table {t} axis {axis} bin_to_cell len {} != n_bins {}",
                                map.len(),
                                grid.n_bins
                            ),
                        });
                    }
                    let extent = table.shape[k];
                    if let Some(&bad) = map.iter().find(|&&c| c >= extent) {
                        return Err(PbError::InvalidInput {
                            what: format!(
                                "correction table {t} axis {axis} cell {bad} >= shape {extent}"
                            ),
                        });
                    }
                }
                let mut cells: usize = 1;
                for &s in &table.shape {
                    cells = cells
                        .checked_mul(s as usize)
                        .ok_or_else(|| PbError::Internal {
                            what: format!("correction table {t} shape product overflow"),
                        })?;
                }
                if table.values.len() != cells {
                    return Err(PbError::ShapeMismatch {
                        what: format!(
                            "correction table {t} values len {} != cell count {cells}",
                            table.values.len()
                        ),
                    });
                }
                if let Some((c, &v)) = table
                    .values
                    .iter()
                    .enumerate()
                    .find(|(_, v)| !v.is_finite())
                {
                    return Err(PbError::InvalidInput {
                        what: format!("correction table {t} value {c} must be finite, got {v}"),
                    });
                }
            }
        }
        Ok(())
    }

    /// The ensemble raw score for one row's bin ids, in full `f64`
    /// (`f0 + Σ alpha_t · tree_t.lookup(x)`), used by the §08 reconstruction gate.
    ///
    /// # Errors
    /// Propagates any [`ObliviousTree::lookup`] failure.
    pub fn ensemble_f64(&self, row_bins: &[u8]) -> Result<f64, PbError> {
        let mut acc = f64::from(self.f0);
        for (alpha, tree) in &self.trees {
            acc += f64::from(*alpha) * f64::from(tree.lookup(row_bins)?);
        }
        acc += self.correction_delta(row_bins)?;
        Ok(acc)
    }

    /// The cell-basis correction's additive contribution for one already-binned row
    /// (`0.0` when there is no correction). Sums each table's raw delta at the merged
    /// cell the row's model bins map to. This is the SAME quantity folded into the raw
    /// effects before purify, so predict and the decomposed bank stay byte-identical (G0).
    #[inline]
    pub fn correction_delta(&self, row_bins: &[u8]) -> Result<f64, PbError> {
        let Some(bank) = &self.correction else {
            return Ok(0.0);
        };
        let mut acc = 0.0_f64;
        for table in &bank.tables {
            let mut flat = 0usize;
            for (k, &axis) in table.axes.iter().enumerate() {
                let bin = *row_bins
                    .get(axis as usize)
                    .ok_or_else(|| PbError::ShapeMismatch {
                        what: format!("row has no axis {axis} for correction lookup"),
                    })?;
                let map = table.bin_to_cell.get(k).ok_or_else(|| PbError::Internal {
                    what: "correction bin_to_cell shorter than axes".into(),
                })?;
                let cell = *map.get(bin as usize).ok_or_else(|| PbError::InvalidInput {
                    what: format!(
                        "correction: model bin {bin} outside bin_to_cell for axis {axis}"
                    ),
                })?;
                let extent = *table.shape.get(k).ok_or_else(|| PbError::Internal {
                    what: "correction shape shorter than axes".into(),
                })?;
                flat = flat * extent as usize + cell as usize;
            }
            acc += *table.values.get(flat).ok_or_else(|| PbError::Internal {
                what: "correction flat index escaped values".into(),
            })?;
        }
        Ok(acc)
    }

    /// Raw score for one already-binned row, accumulated in `f64` and rounded once
    /// to the public `f32` scoring width (spec §10 path A).
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if `row_bins` does not match this model's width;
    /// plus propagated tree lookup errors.
    pub fn score_trees_row(&self, row_bins: &[u8], offset: f32) -> Result<f32, PbError> {
        if row_bins.len() != self.grids.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "row width {} != model grid count {}",
                    row_bins.len(),
                    self.grids.len()
                ),
            });
        }
        let mut acc = f64::from(self.f0) + f64::from(offset);
        for (alpha, tree) in &self.trees {
            acc += f64::from(*alpha) * f64::from(tree.lookup(row_bins)?);
        }
        acc += self.correction_delta(row_bins)?;
        Ok(acc as f32)
    }

    /// Batch raw scores over a column-major [`BinnedMatrix`] into `out`.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if matrix shape or grids/provenance do not match
    /// the model; [`PbError::InvalidInput`] if any bin escapes its grid.
    pub fn score_trees(
        &self,
        x: &BinnedMatrix,
        offset: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<(), PbError> {
        self.validate_binned_matrix(x)?;
        self.score_trees_prevalidated(x, offset, out)
    }

    /// Like [`Model::score_trees`] but SKIPS the per-call [`Model::validate_binned_matrix`] scan.
    /// The caller MUST have already validated `x` against a model with identical grids/provenance —
    /// e.g. [`MultiClassModel::predict_raw`] validates once against class 0, whose grids/provenance
    /// [`MultiClassModel::validate`] enforces equal to every class, then scores all K classes
    /// through this path (avoiding K−1 redundant `O(n·n_cols)` bin scans). Otherwise byte-identical.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] on out/offset length mismatch; propagated tree/correction errors.
    pub(crate) fn score_trees_prevalidated(
        &self,
        x: &BinnedMatrix,
        offset: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<(), PbError> {
        let n_rows = x.n_rows as usize;
        if out.len() != n_rows {
            return Err(PbError::ShapeMismatch {
                what: format!("out len {} != n_rows {n_rows}", out.len()),
            });
        }
        if let Some(off) = offset {
            if off.len() != n_rows {
                return Err(PbError::ShapeMismatch {
                    what: format!("offset len {} != n_rows {n_rows}", off.len()),
                });
            }
        }
        // Route through the packed row-parallel scorer (spec §11 / §10.2a). `ScoringBank` +
        // `score_tile` is proven bit-equal to this function's old per-row/per-tree-column serial
        // loop by `tests/scoring_determinism.rs`'s `model_predict_and_score_trees_are_thread_count_
        // stable_reference` (cross-checks `ScoringBank::Packed` against `Model::score_trees` row for
        // row), so delegating to it is a pure speedup: the old loop never used the ambient rayon
        // pool at all, leaving `n_jobs` a no-op for every tree-backed predict path. `score_row` never
        // adds `f0` itself (its documented contract: `offset` is used verbatim), so it is pre-folded
        // into a dense `combined_offset[r] = f0 + offset[r]` buffer — the exact same `self.f0 + off`
        // expression the old loop computed inline per row. Building the bank is O(trees) (it packs
        // each tree into a 64-byte row; no O(rows) work), so it is cheap to rebuild on every call —
        // this is a derived runtime view, never cached model state. `from_validated_model` (not
        // `from_model`) skips `Model::validate` — this function's own "prevalidated" contract already
        // requires the caller to have validated the model, and re-running that check on every call
        // (incl. once per class from `MultiClassModel::predict_raw`'s loop) was a measured fixed cost
        // dominating small-batch/single-row predicts (~83% of a 1-row call at 3000 trees).
        let bank = ScoringBank::from_validated_model(self)?;
        let combined_offset: Vec<f32> = (0..n_rows)
            .map(|r| self.f0 + offset.and_then(|o| o.get(r).copied()).unwrap_or(0.0))
            .collect();
        let rows: Vec<u32> = (0..u32::try_from(n_rows).map_err(|_| PbError::Internal {
            what: "score_trees_prevalidated: n_rows exceeds u32::MAX".into(),
        })?)
            .collect();
        // Fixed-order row-chunk parallelism (spec §11's blessed pattern, `CHUNK_ROWS = 4096`): each
        // chunk writes only its own disjoint `out` slice, so the result does not depend on how the
        // ambient/scoped rayon pool schedules chunks across threads — bit-identical at 1/2/8 threads
        // (pinned by the determinism test referenced above).
        out.par_chunks_mut(CHUNK_ROWS)
            .zip(rows.par_chunks(CHUNK_ROWS))
            .try_for_each(|(out_chunk, rows_chunk)| {
                score_tile(&bank, x, rows_chunk, Some(&combined_offset), out_chunk)
            })?;
        Ok(())
    }

    /// Raw score ONLY the rows in `rows`, writing `out[r]` for each (other `out` slots untouched).
    /// `out` must be length n_rows. Byte-identical to [`Model::score_trees`] at those rows. Used to
    /// compute the bagged out-of-bag residual, where only each bag's ~20% out-of-bag rows are read,
    /// so scoring all rows is wasted — this scores just the needed subset.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] on shape mismatch; propagated tree/correction lookup errors.
    pub fn score_trees_rows(
        &self,
        x: &BinnedMatrix,
        rows: &[u32],
        out: &mut [f32],
    ) -> Result<(), PbError> {
        self.validate_binned_matrix(x)?;
        let n_rows = x.n_rows as usize;
        if out.len() != n_rows {
            return Err(PbError::ShapeMismatch {
                what: format!("out len {} != n_rows {n_rows}", out.len()),
            });
        }
        let tree_columns: Vec<Vec<&[u8]>> = self
            .trees
            .iter()
            .map(|(_, tree)| tree_split_columns(tree, &x.data))
            .collect::<Result<_, _>>()?;
        // The correction (rare on this path — bag models carry none) is added per row via its
        // bins so `out` stays byte-identical to `score_trees` for the scored rows.
        let mut row_bins = vec![
            0u8;
            if self.correction.is_some() {
                x.data.len()
            } else {
                0
            }
        ];
        for &r in rows {
            let r = r as usize;
            let mut score = f64::from(self.f0);
            for ((alpha, tree), columns) in self.trees.iter().zip(&tree_columns) {
                score += f64::from(*alpha)
                    * f64::from(tree_value_for_row_with_columns(tree, columns, r)?);
            }
            if self.correction.is_some() {
                for (a, col) in x.data.iter().enumerate() {
                    row_bins[a] = *col.get(r).ok_or_else(|| PbError::Internal {
                        what: "score_trees_rows column shorter than n_rows".into(),
                    })?;
                }
                score += self.correction_delta(&row_bins)?;
            }
            *out.get_mut(r).ok_or_else(|| PbError::Internal {
                what: "score_trees_rows row escaped out".into(),
            })? = score as f32;
        }
        Ok(())
    }

    /// Response-space predictions from an already-binned design.
    ///
    /// # Errors
    /// Propagates [`Model::score_trees`] validation/scoring failures.
    pub fn predict_binned(
        &self,
        x: &BinnedMatrix,
        offset: Option<&[f32]>,
    ) -> Result<Vec<f32>, PbError> {
        let mut raw = vec![0.0_f32; x.n_rows as usize];
        self.score_trees(x, offset, &mut raw)?;
        for v in &mut raw {
            *v = inverse_link(self.link, *v);
        }
        Ok(raw)
    }

    /// Response-space predictions from a binned design. Alias for
    /// [`Model::predict_binned`] until raw-data ingest is exposed at this layer.
    ///
    /// # Errors
    /// Propagates [`Model::predict_binned`] failures.
    pub fn predict(&self, x: &BinnedMatrix, offset: Option<&[f32]>) -> Result<Vec<f32>, PbError> {
        self.predict_binned(x, offset)
    }

    fn validate_binned_matrix(&self, x: &BinnedMatrix) -> Result<(), PbError> {
        let n_rows = x.n_rows as usize;
        if x.data.len() != self.grids.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "matrix has {} columns, model has {} features",
                    x.data.len(),
                    self.grids.len()
                ),
            });
        }
        if x.grids != self.grids {
            return Err(PbError::ShapeMismatch {
                what: "matrix grids do not match model grids".into(),
            });
        }
        if x.provenance != self.provenance {
            return Err(PbError::ShapeMismatch {
                what: "matrix provenance does not match model provenance".into(),
            });
        }
        for (axis, col) in x.data.iter().enumerate() {
            if col.len() != n_rows {
                return Err(PbError::ShapeMismatch {
                    what: format!("column {axis} len {} != n_rows {n_rows}", col.len()),
                });
            }
            let grid = self.grids.get(axis).ok_or_else(|| PbError::Internal {
                what: "model grid disappeared during score validation".into(),
            })?;
            for (row, &bin) in col.iter().enumerate() {
                if u16::from(bin) >= grid.n_bins {
                    return Err(PbError::InvalidInput {
                        what: format!(
                            "column {axis} row {row} bin {bin} outside grid n_bins {}",
                            grid.n_bins
                        ),
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic
    )]

    use super::*;
    use crate::explain::{fixture_model, fixture_serve};

    /// A 2-table correction bank whose values (`0.1`, `0.2`) are classic decimal
    /// fractions that don't round-trip through `f32`: on row 0 of [`fixture_serve`]
    /// (tree score `6.0`), summing them in `f64` then casting once (correct) provably
    /// diverges from casting each table separately and folding the casts into the f32
    /// running score (the historical bug in `score_trees_prevalidated`) — verified
    /// empirically against the real fixture, not just asserted by construction, so this
    /// fixture cannot pass vacuously.
    fn two_table_correction() -> CorrectionBank {
        CorrectionBank {
            tables: vec![
                CorrectionTable {
                    axes: vec![0],
                    shape: vec![3],
                    bin_to_cell: vec![vec![0, 1, 2]],
                    values: vec![0.0, 0.1, 0.1],
                },
                CorrectionTable {
                    axes: vec![1],
                    shape: vec![3],
                    bin_to_cell: vec![vec![0, 1, 2]],
                    values: vec![0.0, 0.2, 0.2],
                },
            ],
        }
    }

    /// Pins spec §10.2/§10.7's tolerance-0 contract between the three path-A scorers on
    /// a corrected model: `score_trees_row` (the canonical per-row scalar path),
    /// `score_trees` (batch, via `score_trees_prevalidated`), and `score_trees_rows`
    /// (subset) must all compute bit-identical raw scores, including the cell-basis
    /// correction's contribution.
    #[test]
    fn score_trees_row_batch_and_rows_agree_bit_exactly_with_a_multi_table_correction() {
        let mut model = fixture_model();
        model.correction = Some(two_table_correction());
        model.validate().unwrap();
        let x = fixture_serve();

        let mut batch = vec![0.0_f32; x.0.n_rows as usize];
        model.score_trees(&x.0, None, &mut batch).unwrap();

        let all_rows: Vec<u32> = (0..x.0.n_rows).collect();
        let mut subset = vec![0.0_f32; x.0.n_rows as usize];
        model
            .score_trees_rows(&x.0, &all_rows, &mut subset)
            .unwrap();

        for row in 0..x.0.n_rows as usize {
            let bins: Vec<u8> = x.0.data.iter().map(|c| c[row]).collect();
            let per_row = model.score_trees_row(&bins, 0.0).unwrap();
            assert_eq!(
                batch[row].to_bits(),
                per_row.to_bits(),
                "row {row}: score_trees (batch) vs score_trees_row diverged"
            );
            assert_eq!(
                subset[row].to_bits(),
                per_row.to_bits(),
                "row {row}: score_trees_rows vs score_trees_row diverged"
            );
        }
    }

    /// Round-trip regression for the §10.7 promise `decode(encode(m)) == m`: `bag_spans` is
    /// `#[serde(skip)]` by design (see the struct doc) — a derived `PartialEq` used to compare
    /// it anyway, so any bagged model (`bag_spans: Some(_)`, the sklearn outer-bag default)
    /// failed this exact contract even though its wire bytes round-tripped losslessly. Pins
    /// both halves: equality holds under the hand-written `PartialEq`, AND the field is
    /// confirmed still absent from the wire (not silently started being serialized instead).
    #[test]
    fn bag_spans_is_excluded_from_round_trip_equality_but_not_serialized() {
        let mut model = fixture_model();
        model.bag_spans = Some(vec![(0, 1)]);

        let json = model.to_json().unwrap();
        let from_json = Model::from_json(&json).unwrap();
        assert_eq!(
            model, from_json,
            "JSON round-trip must equal the source model"
        );
        assert!(
            from_json.bag_spans.is_none(),
            "bag_spans is #[serde(skip)] by design — it must NOT survive the round-trip"
        );

        let bincode = model.to_bincode().unwrap();
        let from_bincode = Model::from_bincode(&bincode).unwrap();
        assert_eq!(
            model, from_bincode,
            "bincode round-trip must equal the source model"
        );
        assert!(from_bincode.bag_spans.is_none());
    }
}

/// A trained K-class softmax (multinomial) model — the native-softmax multiclass extension
/// (spec §05 / `design/multiclass-design.md`).
///
/// Native softmax boosting produces `K` additive raw score functions `F_0..F_{K-1}`, each an
/// exactly-decomposable scalar [`Model`] whose raw score is that class's logit. Probabilities are
/// `softmax(F_0(x), .., F_{K-1}(x))` — a response-layer transform applied OUTSIDE the additive
/// raw-score space, so every per-class [`Model`] decomposes into its own exact ≤3rd-order fANOVA
/// `TableBank` (the exactness invariant holds per class). The K classes are trained jointly:
/// each round's per-class gradients are computed from a softmax over all K current raw columns.
///
/// `PartialEq` is hand-written, NOT derived, for the same reason [`Model`]'s is: `cell_refit`
/// is `#[serde(skip)]` runtime introspection, so a derived comparison would break the §10.7
/// round-trip contract (`decode(encode(m)) == m`) for a refitted model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiClassModel {
    /// One scalar model per class; `classes[k]`'s raw score is class `k`'s logit `F_k`.
    pub classes: Vec<Model>,
    /// Class labels parallel to `classes` (length `K`).
    pub class_labels: Vec<String>,
    /// Wire-version covering the container.
    pub schema_version: u32,
    /// Runtime-only report from the §G1 K>=3 out-of-bag cell refit (`CellRefit`, i.e.
    /// `cell_refit_base`). `None` when the refit was never asked for. Never serialized, never
    /// part of model identity — the same contract as [`Model::delta_step_gate`].
    #[serde(skip)]
    pub cell_refit: Option<MultiClassCellRefitReport>,
}

/// What the §G1 K>=3 cell refit did on this fit — the diagnosable half of the design, so a
/// caller can tell "the guard declined" from "there was nothing reachable to correct".
/// Runtime-only: never serialized, never part of model identity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiClassCellRefitReport {
    /// The step length the ONE joint backtrack accepted on the true multinomial OOB loss.
    /// `0.0` means the guard declined the whole correction (all K classes dropped).
    pub lambda: f64,
    /// Realized order-<=2 supports the refit could correct.
    pub n_supports: usize,
    /// Realized supports EXCLUDED because they touch a P1 multi-channel raw feature and so
    /// cannot be a `CorrectionTable`. `n_supports / (n_supports + n_blocked)` is the reachable
    /// coverage — the honest denominator for reading a `lambda == 0` verdict.
    pub n_blocked: usize,
    /// Trial step lengths the joint loss refused (the halvings walked past).
    pub n_rejected: usize,
    /// Rows in the out-of-bag jury the backtrack scored on.
    pub guard_rows: usize,
}

impl MultiClassCellRefitReport {
    /// Did the joint backtrack decline the correction outright?
    #[must_use]
    pub fn declined(&self) -> bool {
        self.lambda <= 0.0
    }

    /// Share of realized supports the refit could actually reach, in `[0, 1]`. `1.0` when
    /// nothing was blocked (including the degenerate "no supports at all" case).
    #[must_use]
    pub fn reachable_coverage(&self) -> f64 {
        let total = self.n_supports + self.n_blocked;
        if total == 0 {
            1.0
        } else {
            self.n_supports as f64 / total as f64
        }
    }
}

impl PartialEq for MultiClassModel {
    /// Structural equality over every field EXCEPT `cell_refit` (runtime-only introspection,
    /// `#[serde(skip)]`, so it must not participate in a model's persistent identity).
    fn eq(&self, other: &Self) -> bool {
        self.classes == other.classes
            && self.class_labels == other.class_labels
            && self.schema_version == other.schema_version
    }
}

impl MultiClassModel {
    /// The MINIMUM `schema_version` a reader needs — the max over the per-class models.
    #[must_use]
    pub fn required_schema_version(&self) -> u32 {
        self.classes
            .iter()
            .map(Model::required_schema_version)
            .max()
            .unwrap_or(crate::serialize::SCHEMA_VERSION_UNLIFTED)
    }

    /// Number of classes `K`.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    /// Validate structure: `K >= 2`, labels parallel to classes, every sub-model valid and
    /// sharing identical grids/provenance, and the container `schema_version` current.
    ///
    /// # Errors
    /// [`PbError::Serialization`] for a schema-version mismatch; [`PbError::ShapeMismatch`] for
    /// inconsistent labels/grids/provenance; [`PbError::InvalidInput`] for `K < 2`; plus any
    /// propagated per-model [`Model::validate`] failure.
    pub fn validate(&self) -> Result<(), PbError> {
        let required = self.required_schema_version();
        if self.schema_version > crate::serialize::SCHEMA_VERSION || self.schema_version < required
        {
            return Err(PbError::Serialization(format!(
                "multiclass schema_version {} outside {}..={} for this model's contents",
                self.schema_version,
                required,
                crate::serialize::SCHEMA_VERSION
            )));
        }
        let k = self.classes.len();
        if k < 2 {
            return Err(PbError::InvalidInput {
                what: format!("multiclass model must have >= 2 classes, got {k}"),
            });
        }
        if self.class_labels.len() != k {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "multiclass class_labels len {} != class count {k}",
                    self.class_labels.len()
                ),
            });
        }
        let first = self.classes.first().ok_or_else(|| PbError::Internal {
            what: "multiclass has no classes after count check".into(),
        })?;
        first.validate()?;
        for (i, m) in self.classes.iter().enumerate().skip(1) {
            m.validate()?;
            if m.grids != first.grids {
                return Err(PbError::ShapeMismatch {
                    what: format!("multiclass class {i} grids differ from class 0"),
                });
            }
            if m.provenance != first.provenance {
                return Err(PbError::ShapeMismatch {
                    what: format!("multiclass class {i} provenance differ from class 0"),
                });
            }
        }
        Ok(())
    }

    /// Raw per-class logits for a binned design, row-major `n_rows × K`
    /// (row `r`, class `k` at flat index `r*K + k`).
    ///
    /// # Errors
    /// Propagates any per-class [`Model::score_trees`] failure; [`PbError::Internal`] on a size
    /// overflow or index escape.
    pub fn predict_raw(&self, x: &BinnedMatrix) -> Result<Vec<f32>, PbError> {
        let n = x.n_rows as usize;
        let k = self.classes.len();
        let cells = n.checked_mul(k).ok_or_else(|| PbError::Internal {
            what: "multiclass raw score size overflows usize".into(),
        })?;
        let mut out = vec![0.0_f32; cells];
        let mut col = vec![0.0_f32; n];
        // Validate the design ONCE against class 0 (all classes share identical grids/provenance,
        // enforced by `validate`), then score every class through the validation-skipping path —
        // avoiding K−1 redundant O(n·n_cols) bin scans + deep grid compares.
        if let Some(first) = self.classes.first() {
            first.validate_binned_matrix(x)?;
        }
        for (ci, model) in self.classes.iter().enumerate() {
            model.score_trees_prevalidated(x, None, &mut col)?;
            for (r, &v) in col.iter().enumerate() {
                let idx = r
                    .checked_mul(k)
                    .and_then(|b| b.checked_add(ci))
                    .ok_or_else(|| PbError::Internal {
                        what: "multiclass raw flat index overflow".into(),
                    })?;
                *out.get_mut(idx).ok_or_else(|| PbError::Internal {
                    what: "multiclass raw flat index escaped".into(),
                })? = v;
            }
        }
        Ok(out)
    }

    /// Class probabilities for a binned design, row-major `n_rows × K`; each row is a numerically
    /// stable softmax over the K logits (rows sum to 1).
    ///
    /// # Errors
    /// Propagates [`MultiClassModel::predict_raw`]; [`PbError::Internal`] on a row-slice escape.
    pub fn predict_proba(&self, x: &BinnedMatrix) -> Result<Vec<f32>, PbError> {
        let n = x.n_rows as usize;
        let k = self.classes.len();
        let mut raw = self.predict_raw(x)?;
        for r in 0..n {
            let base = r.checked_mul(k).ok_or_else(|| PbError::Internal {
                what: "multiclass proba base overflow".into(),
            })?;
            let end = base.checked_add(k).ok_or_else(|| PbError::Internal {
                what: "multiclass proba end overflow".into(),
            })?;
            let row = raw.get_mut(base..end).ok_or_else(|| PbError::Internal {
                what: "multiclass proba row escaped".into(),
            })?;
            softmax_in_place(row);
        }
        Ok(raw)
    }
}

/// Numerically stable in-place softmax over a row of logits (max-subtract, exponent clamped to
/// `[-30, 30]`). A non-finite max or an all-zero denominator degrades to the uniform distribution
/// rather than producing `NaN`. Deterministic: a fixed left-to-right fold.
pub(crate) fn softmax_in_place(row: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in row.iter() {
        if v > m {
            m = v;
        }
    }
    if !m.is_finite() {
        m = 0.0;
    }
    let mut denom = 0.0_f32;
    for v in row.iter_mut() {
        let e = (*v - m).clamp(-30.0, 30.0).exp();
        *v = e;
        denom += e;
    }
    let len = row.len();
    if denom > 0.0 && denom.is_finite() {
        for v in row.iter_mut() {
            *v /= denom;
        }
    } else if len > 0 {
        let u = 1.0 / len as f32;
        for v in row.iter_mut() {
            *v = u;
        }
    }
}

pub(crate) fn inverse_link(link: Link, raw: f32) -> f32 {
    match link {
        Link::Identity => raw,
        Link::Log => raw.clamp(-30.0, 30.0).exp(),
        Link::Logit => {
            if raw >= 0.0 {
                let z = (-raw).clamp(-30.0, 30.0).exp();
                1.0 / (1.0 + z)
            } else {
                let z = raw.clamp(-30.0, 30.0).exp();
                z / (1.0 + z)
            }
        }
    }
}

/// Row-sampling strategy for split search (§06 / M5-MVS).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sampling {
    /// Use every row for split search (the v1 green-spine default).
    Full,
    /// Minimal-variance-style probability-proportional-to-gradient row sampling.
    ///
    /// `rate` is the target sample fraction and `min_rows` is a lower bound on the
    /// selected row count. The final tree leaves are still refit from all rows.
    Mvs {
        /// Target row fraction, `0 < rate <= 1`.
        rate: f32,
        /// Minimum sampled row count.
        min_rows: u32,
    },
}

impl Default for Sampling {
    fn default() -> Self {
        Self::Full
    }
}

/// How the interaction-admission hurdle is applied during split growth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionGainHurdleMode {
    /// Use the configured scalar unchanged. This preserves the legacy split gate exactly.
    Fixed,
    /// Decay the hurdle as first-split gains fade, with stricter admission for 3-way interactions.
    Adaptive,
}

impl Default for InteractionGainHurdleMode {
    fn default() -> Self {
        Self::Fixed
    }
}

/// The optimizer configuration (spec §06.1). v1 green-spine subset: the full §06.1
/// knob set (`colsample_*`, LR schedule, early stopping) lands with its features. The
/// v1.5 levers are exposed here as row sampling via [`Sampling::Mvs`] and quantized
/// histograms via [`HistPrecision::QuantizedI32`]; §09 predictiveness knobs live in
/// [`crate::boosters::BoosterConfig`]; the §07 leaf-credibility floors + `path_smooth`
/// live on [`FitSpec::credibility`] (alongside the other §07 constraints). FLAG:
/// `Config` remains a subset of the full §06.1 type for now.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Number of boosting rounds (upper bound; growth also stops if a round can't split).
    pub n_trees: u32,
    /// Learning rate applied to each tree's leaf values.
    pub learning_rate: f32,
    /// L2 leaf regularizer `λ` (in `w* = −G/(H+λ)` and the gain).
    pub lambda: f32,
    /// L1 leaf regularizer applied by soft-thresholding aggregated gradients.
    pub l1_leaf: f32,
    /// `gamma` floor: a level terminates if the best gain is `<= min_split_gain`.
    pub min_split_gain: f32,
    /// Leaf-stage `|w*|`-clamp (LightGBM `max_delta_step`, §05.6/§06.4). `None` falls
    /// back to `Loss::max_delta_step()` (log-link objectives ⇒ `Some(0.7)`); a non-`None` value wins.
    /// Applied on the full-precision aggregated Newton step, never per-row `h`.
    pub max_delta_step: Option<f32>,
    /// Row sampler used for split search. [`Sampling::Full`] is the inert default.
    pub sampling: Sampling,
    /// Per-tree axis subsampling rate. `1.0` scans every axis.
    pub colsample_bytree: f32,
    /// Deterministic decay for per-round learning rate:
    /// `lr_t = learning_rate / (1 + learning_rate_decay * t)`.
    pub learning_rate_decay: f32,
    /// Optional deterministic internal validation fraction for early stopping.
    pub validation_fraction: Option<f32>,
    /// Patience in boosting rounds once `validation_fraction` is enabled.
    pub early_stopping_rounds: u32,
    /// Opt-in ADAPTIVE early-stopping patience. `None` (the default) keeps the fixed
    /// [`Self::early_stopping_rounds`] patience — bit-identical to the non-adaptive engine.
    /// `Some(ratio)` makes the effective patience grow with progress: at any point it is
    /// `clamp(ceil(ratio * best_round_so_far), EARLY_STOPPING_ADAPTIVE_FLOOR, early_stopping_rounds)`,
    /// so a fit that peaks early stops sooner (less trained-then-truncated overshoot) while a fit
    /// that keeps improving retains headroom up to the `early_stopping_rounds` cap. `ratio` must be
    /// finite and `> 0`. See [`Self::effective_patience`].
    pub early_stopping_adaptive: Option<f64>,
    /// Relative early-stopping improvement tolerance. A round becomes the new best — resetting the
    /// patience window AND advancing the deployed truncation point — only when the validation
    /// deviance beats the incumbent best by more than this fraction: `dev < best * (1 - min_delta)`.
    /// `0.0` (the default) is exactly the legacy any-improvement rule (`dev < best`), so even
    /// epsilon-level noise on the validation slice still resets patience; a small positive value
    /// (e.g. `1e-4`) ignores such noise so early stopping actually terminates on large low-signal
    /// data instead of running to the `n_trees` cap. Must be finite and in `[0.0, 1.0)`. See
    /// [`Self::is_material_improvement`].
    pub early_stopping_min_delta: f64,
    /// Interaction-admission hurdle (soft heredity). A level-2/3 split that would introduce a NEW
    /// raw feature, raising the tree's fANOVA order, must clear this scale-free hurdle against the
    /// tree's level-1 gain. When the hurdle is enabled, fresh candidates also compete with the best
    /// already-used raw feature under the split ranking score: lower-order refinement wins ties.
    /// `0.0` restores pure greedy new-feature admission, with reuse only when no fresh feature is
    /// available. Must be finite and `>= 0`.
    pub interaction_gain_hurdle: f32,
    /// Hurdle application mode. [`InteractionGainHurdleMode::Fixed`] keeps the scalar hurdle exactly
    /// as supplied. [`InteractionGainHurdleMode::Adaptive`] keeps the same single magnitude knob but
    /// decays the effective hurdle from its full early value toward a lenient late value as
    /// first-level/main-effect gains fade relative to a decayed-max reference; 3-way admission pays
    /// a fixed extra multiplier over 2-way admission and must also clear a parent-level/pair-evidence
    /// floor.
    pub interaction_gain_hurdle_mode: InteractionGainHurdleMode,
    /// Extra per-tree leaf Newton refinement steps after a structure is fixed.
    pub leaf_refine_steps: u8,
    /// Backtracking attempts per leaf-refinement step.
    pub leaf_refine_backtracks: u8,
    /// Tier-2 log-link closed-form leaf refinement (Poisson today). `true` (the default) derives
    /// each per-step leaf Newton delta straight from the per-leaf closed form `G_l`/`H_l`, dropping
    /// the remaining per-row `grad_hess` passes of the line search — a speed win for log-link fits
    /// with leaf refinement. Leaves may then drift ~1e-7 from the exact per-row path (f32 `grad_hess`
    /// vs f64 closed form; the per-row hessian floor is not applied), so it is NOT bit-identical to
    /// the generic per-row line search. `false` keeps Tier 1: the leaf UPDATES stay byte-identical to
    /// the generic path and only the accept-deviance is closed-form. Under either setting the
    /// exp-clamp guard still falls back to the exact generic per-row path when the ±30 clamp could
    /// bite. Inert for non-log-link losses and when [`Self::leaf_refine_steps`] `== 0`.
    pub refine_closed_form_tier2: bool,
    /// Incremental inverse-link cache `mu = exp(F)` for log-link losses (Poisson today). `false` (the
    /// default) recomputes `mu = exp(F)` from scratch in every round's `grad_hess` (an O(N) `exp`
    /// pass). `true` maintains `mu` MULTIPLICATIVELY — `mu *= e^{Δleaf}` in `update_raw` (≤8 `exp`s +
    /// an O(N) multiply), refreshed from a fresh `exp(F)` pass every `INCREMENTAL_MU_REFRESH_ROUNDS`
    /// rounds to bound floating-point drift — so `grad_hess` reads the cache instead of an `exp`. A
    /// speed win for log-link fits; leaves drift ~1e-9..1e-6 from the exact per-round path (bounded
    /// multiplicative accumulation between refreshes), so it is NOT bit-identical. Honored only on the
    /// plain boosting path (Poisson, no AGBM/DART/ridge-refit); any incompatible lever silently keeps
    /// the exact `exp(F)` path. Inert for non-log-link losses.
    pub incremental_mu: bool,
    /// Histogram precision used for split search.
    pub hist_precision: HistPrecision,
    /// Exactness-preserving predictiveness boosters (§09). Defaults are all inert.
    pub boosters: crate::boosters::BoosterConfig,
    /// Scale-invariant L2 leaf regularizer (prototype). `false` (the default) keeps [`Self::lambda`]
    /// literal in the Newton denominator `H+λ` everywhere it is consumed (leaf value AND split
    /// gain) — bit-identical to the pre-existing arithmetic. `true` rescales it once per round by
    /// that round's mean per-row hessian h̄ = Σh/N over the training rows (before any MVS
    /// subsample/reweight): `λ_eff = λ·h̄`. On exposure-weighted objectives (e.g. Poisson with
    /// per-row weights in the hundreds/thousands, `h = w·μ`), `H` dwarfs any `λ ∈ {0.1, 1, 10}`, so
    /// the raw denominator makes `lambda` a dead no-op exactly where regularization matters most;
    /// rescaling turns `λ` into a scale-free "pseudo-count of prior rows" that stays live
    /// regardless of the weight/exposure magnitude. `λ_eff` is resolved once per round and shared
    /// by every split-gain and leaf-value evaluation that round, so gains and values never disagree.
    /// Wired through the single-output fit path (`fit_single`, covering `squared_error`/`poisson`/
    /// `gamma`/`tweedie` and binary `logistic`, and everything `fit_outer_bag` bags). NOT yet wired
    /// into the `>2`-class multinomial round loop (`fit_multiclass_single`), which still resolves
    /// `H+λ` literally — inert there for now, same as an un-set flag.
    pub lambda_scale_invariant: bool,
    /// The in-fit rate-collapse gate on [`Self::max_delta_step`] (spec §05.6 addendum, av35).
    /// [`GatedStepPolicy::Objective`] — the default — defers to
    /// [`Loss::gated_max_delta_step`], which is `Some` for Tweedie ONLY (`1e-3`/`0.3`) and
    /// `None` for every other objective. An EXPLICIT [`Self::max_delta_step`] wins over all of
    /// this: a caller who named a cap gets exactly that cap for the whole fit, gate or no gate.
    /// See [`GatedDeltaStep`] for the signal, the engage semantics, and the evidence.
    pub max_delta_step_gated: GatedStepPolicy,
    /// Run-time controls that never change what a round computes: an external evaluation
    /// holdout, per-round history, a round observer. The default is inert (bit-identical).
    pub fit_control: FitControl,
}

/// Run-time controls of a single-output fit ([`Config::fit_control`]).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FitControl {
    /// The [`FitSpec::fixed_holdout`] rows are an external evaluation set, not fit rows: they
    /// only score early stopping. The holdout slope recalibration (`reanchor_slope`) is skipped
    /// and the intercept re-anchor uses the training rows only.
    pub external_holdout: bool,
    /// Record each round's stopping deviance in [`BagFitReport`] (and the training deviance
    /// too when an observer is set, which computes it).
    pub record_history: bool,
    /// Called after every boosting round; returning `true` stops this fit's boosting.
    pub observer: Option<RoundObserver>,
    /// This fit's bag index, reported in [`RoundEvent::bag`] (set by the outer bag).
    pub bag: u32,
}

/// One boosting round, as a [`RoundObserver`] sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoundEvent {
    /// Bag index (0 for an unbagged fit).
    pub bag: u32,
    /// Round number, 1-based.
    pub round: u32,
    /// The configured round ceiling.
    pub n_trees: u32,
    /// Mean deviance on the rows this bag trains on, after the round.
    pub train_deviance: f64,
    /// Mean deviance on the early-stopping rows after the round; `None` without them.
    pub eval_deviance: Option<f64>,
}

/// A per-round hook ([`FitControl::observer`]). Shared by every bag of a fit, so it may be
/// called concurrently; bags call it in round order each.
#[derive(Clone)]
pub struct RoundObserver(pub std::sync::Arc<dyn Fn(&RoundEvent) -> bool + Send + Sync>);

impl std::fmt::Debug for RoundObserver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RoundObserver")
    }
}

impl PartialEq for RoundObserver {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

/// Why a fit's boosting loop ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Ran every configured round.
    MaxTrees,
    /// Early-stopping patience ran out.
    EarlyStopping,
    /// No admissible split cleared the floors.
    NoSplit,
    /// A [`RoundObserver`] asked to stop.
    Callback,
}

impl StopReason {
    /// The stable name used in reports.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MaxTrees => "max_trees",
            Self::EarlyStopping => "early_stopping",
            Self::NoSplit => "no_split",
            Self::Callback => "callback",
        }
    }
}

/// How one bag's boosting went (runtime-only; see [`Model::fit_report`]).
#[derive(Debug, Clone, PartialEq)]
pub struct BagFitReport {
    /// Trees the bag kept (after any early-stopping truncation).
    pub trees_kept: u32,
    /// Rounds the bag trained.
    pub rounds_trained: u32,
    /// Why the loop ended.
    pub reason: StopReason,
    /// Per-round training deviance, when [`FitControl::record_history`] and an observer are set.
    pub train_deviance: Vec<f64>,
    /// Per-round stopping deviance, when [`FitControl::record_history`] is set and stopping
    /// rows exist.
    pub eval_deviance: Vec<f64>,
}

/// How [`Config::max_delta_step_gated`] resolves. Tri-state on purpose: `None`-as-"auto" and
/// `None`-as-"off" are different requests, and conflating them would make "disable the gate"
/// unexpressible without also disabling the objective's plain `max_delta_step`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum GatedStepPolicy {
    /// Use the objective's advertised gate ([`Loss::gated_max_delta_step`]). Tweedie ⇒ armed
    /// at `1e-3`/`0.3`; every other objective ⇒ no gate. The default.
    #[default]
    Objective,
    /// No gate: the resolved `max_delta_step` stands for the whole fit, whatever the fit does.
    Off,
    /// Caller-specified gate parameters, overriding the objective default (and arming the gate
    /// on objectives that ship none).
    On(GatedDeltaStep),
}

/// What the [`Config::max_delta_step_gated`] detector saw during a fit — the "in-fit
/// diagnosable" half of the design. Runtime-only: never serialized, never part of model
/// identity (see [`Model::delta_step_gate`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeltaStepGateReport {
    /// Did the cap engage during this fit (any bag, for a bagged model)?
    pub engaged: bool,
    /// The round the cap first engaged — EARLIEST across bags for a bagged model. `None` if
    /// the gate stayed silent.
    pub engaged_round: Option<u32>,
    /// How many of the model's fits (1, or one per bag) engaged the cap.
    pub bags_engaged: u32,
    /// How many fits the report covers (1, or the bag count).
    pub bags_total: u32,
    /// The most extreme `ln(μ_i / weighted-mean-rate)` the detector observed over every
    /// checked round, training row, and bag. `f64::INFINITY` if the gate never ran. This is
    /// the calibration signal: it says HOW FAR a silent fit was from tripping.
    pub min_log_rate_ratio: f64,
    /// The `ln(collapse_threshold)` this fit was compared against.
    pub log_threshold: f64,
    /// The `|w*|` clamp the gate installs on engage (already `min`-ed against the resolved
    /// `max_delta_step`, so it is the value actually used).
    pub capped_step: f64,
    /// Total per-round O(n) min scans performed, summed over bags — the overhead accounting.
    pub rounds_checked: u64,
}

impl DeltaStepGateReport {
    /// Fold a per-bag report into an accumulator (used by the outer-bag soup).
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            engaged: self.engaged || other.engaged,
            engaged_round: match (self.engaged_round, other.engaged_round) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            },
            bags_engaged: self.bags_engaged + other.bags_engaged,
            bags_total: self.bags_total + other.bags_total,
            min_log_rate_ratio: self.min_log_rate_ratio.min(other.min_log_rate_ratio),
            log_threshold: self.log_threshold,
            capped_step: self.capped_step,
            rounds_checked: self.rounds_checked + other.rounds_checked,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_trees: 1000,
            learning_rate: 0.05,
            lambda: 1.0,
            l1_leaf: 0.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Sampling::Full,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: true,
            incremental_mu: false,
            hist_precision: HistPrecision::FullF64,
            boosters: crate::boosters::BoosterConfig::default(),
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            fit_control: FitControl::default(),
        }
    }
}

/// Floor (minimum) for ADAPTIVE early-stopping patience: the effective patience never drops
/// below this many rounds even when `best_round_so_far` is tiny, so a fit that happens to peak at
/// its very first rounds still gets a real chance to recover before it is stopped. Only consulted
/// when [`Config::early_stopping_adaptive`] is `Some`.
pub(crate) const EARLY_STOPPING_ADAPTIVE_FLOOR: usize = 50;

impl Config {
    /// Effective early-stopping patience for the current best-scoring position (`best_round`,
    /// measured in whatever unit the caller compares against: tree count in `fit_single`, round
    /// index in the multiclass loop — the ratio is dimensionless, so either is consistent).
    ///
    /// With [`Self::early_stopping_adaptive`] `== None` this returns the fixed
    /// [`Self::early_stopping_rounds`] cap unchanged, so callers that swap a literal
    /// `early_stopping_rounds as usize` for this call are bit-identical. `Some(ratio)` returns
    /// `ceil(ratio * best_round)` clamped to `[EARLY_STOPPING_ADAPTIVE_FLOOR, early_stopping_rounds]`.
    /// The `early_stopping_rounds` cap always wins (a floor above the cap degrades to the cap
    /// rather than panicking), so the adaptive patience is never larger than the fixed one.
    pub(crate) fn effective_patience(&self, best_round: usize) -> usize {
        let cap = self.early_stopping_rounds as usize;
        match self.early_stopping_adaptive {
            None => cap,
            // `ratio` is validated finite and `> 0`, and `best_round` is a real count, so the
            // product is finite and non-negative; the `f64 -> usize` cast saturates defensively.
            // Clamp with `floor` capped to `cap` first, so `lo <= cap` always holds — a floor above
            // the cap degrades to the cap instead of tripping `clamp`'s inverted-bounds panic.
            Some(ratio) => {
                let scaled = (ratio * best_round as f64).ceil() as usize;
                let lo = EARLY_STOPPING_ADAPTIVE_FLOOR.min(cap);
                scaled.clamp(lo, cap)
            }
        }
    }

    /// Whether validation deviance `dev` is a MATERIAL improvement over the incumbent `best` under
    /// the [`Self::early_stopping_min_delta`] relative tolerance: `dev < best * (1 - min_delta)`.
    /// Both arguments are deviances (`>= 0`). With `min_delta == 0.0` this is exactly `dev < best` —
    /// the legacy any-improvement rule — so early stopping stays bit-identical when the tolerance is
    /// off. A positive tolerance requires beating `best` by more than that fraction, so epsilon-level
    /// noise on the validation slice no longer resets the patience window or advances the deployed
    /// truncation point. `best == 0.0` admits nothing (no `dev >= 0` is `< 0`).
    pub(crate) fn is_material_improvement(&self, best: f64, dev: f64) -> bool {
        dev < best * (1.0 - self.early_stopping_min_delta)
    }

    /// Validate the configuration.
    ///
    /// # Errors
    /// [`PbError::InvalidConfig`] if `n_trees == 0`, `learning_rate` is non-finite or
    /// `<= 0`, `lambda` is non-finite or `< 0`, or `min_split_gain` is non-finite or `< 0`.
    pub fn validate(&self) -> Result<(), PbError> {
        if self.n_trees == 0 {
            return Err(PbError::InvalidConfig {
                what: "n_trees must be > 0".into(),
            });
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(PbError::InvalidConfig {
                what: format!(
                    "learning_rate must be finite and > 0, got {}",
                    self.learning_rate
                ),
            });
        }
        if !self.lambda.is_finite() || self.lambda < 0.0 {
            return Err(PbError::InvalidConfig {
                what: format!("lambda must be finite and >= 0, got {}", self.lambda),
            });
        }
        if !self.l1_leaf.is_finite() || self.l1_leaf < 0.0 {
            return Err(PbError::InvalidConfig {
                what: format!("l1_leaf must be finite and >= 0, got {}", self.l1_leaf),
            });
        }
        if !self.min_split_gain.is_finite() || self.min_split_gain < 0.0 {
            return Err(PbError::InvalidConfig {
                what: format!(
                    "min_split_gain must be finite and >= 0, got {}",
                    self.min_split_gain
                ),
            });
        }
        if let Some(d) = self.max_delta_step {
            if !d.is_finite() || d <= 0.0 {
                return Err(PbError::InvalidConfig {
                    what: format!("max_delta_step must be finite and > 0 when set, got {d}"),
                });
            }
        }
        if let GatedStepPolicy::On(g) = self.max_delta_step_gated {
            g.validate()?;
        }
        match self.sampling {
            Sampling::Full => {}
            Sampling::Mvs { rate, min_rows } => {
                if !rate.is_finite() || rate <= 0.0 || rate > 1.0 {
                    return Err(PbError::InvalidConfig {
                        what: format!("MVS rate must be finite and in (0, 1], got {rate}"),
                    });
                }
                if min_rows == 0 {
                    return Err(PbError::InvalidConfig {
                        what: "MVS min_rows must be > 0".into(),
                    });
                }
            }
        }
        if !self.colsample_bytree.is_finite()
            || self.colsample_bytree <= 0.0
            || self.colsample_bytree > 1.0
        {
            return Err(PbError::InvalidConfig {
                what: format!(
                    "colsample_bytree must be finite and in (0, 1], got {}",
                    self.colsample_bytree
                ),
            });
        }
        if !self.learning_rate_decay.is_finite() || self.learning_rate_decay < 0.0 {
            return Err(PbError::InvalidConfig {
                what: format!(
                    "learning_rate_decay must be finite and >= 0, got {}",
                    self.learning_rate_decay
                ),
            });
        }
        if let Some(frac) = self.validation_fraction {
            if !frac.is_finite() || frac <= 0.0 || frac >= 1.0 {
                return Err(PbError::InvalidConfig {
                    what: format!("validation_fraction must be finite and in (0, 1), got {frac}"),
                });
            }
            if self.early_stopping_rounds == 0 {
                return Err(PbError::InvalidConfig {
                    what: "early_stopping_rounds must be > 0 when validation_fraction is set"
                        .into(),
                });
            }
        }
        if let Some(ratio) = self.early_stopping_adaptive {
            if !ratio.is_finite() || ratio <= 0.0 {
                return Err(PbError::InvalidConfig {
                    what: format!(
                        "early_stopping_adaptive must be finite and > 0 when set, got {ratio}"
                    ),
                });
            }
        }
        if !self.early_stopping_min_delta.is_finite()
            || self.early_stopping_min_delta < 0.0
            || self.early_stopping_min_delta >= 1.0
        {
            return Err(PbError::InvalidConfig {
                what: format!(
                    "early_stopping_min_delta must be finite and in [0.0, 1.0), got {}",
                    self.early_stopping_min_delta
                ),
            });
        }
        if !self.interaction_gain_hurdle.is_finite() || self.interaction_gain_hurdle < 0.0 {
            return Err(PbError::InvalidConfig {
                what: format!(
                    "interaction_gain_hurdle must be finite and >= 0, got {}",
                    self.interaction_gain_hurdle
                ),
            });
        }
        if self.leaf_refine_steps > 0 && self.leaf_refine_backtracks == 0 {
            return Err(PbError::InvalidConfig {
                what: "leaf_refine_backtracks must be > 0 when leaf_refine_steps is enabled".into(),
            });
        }
        self.boosters.validate()?;
        Ok(())
    }
}

/// The public estimator (spec §2.9). Builder-configured, `fit → Model`,
/// sklearn-mirrored in Python.
#[derive(Debug, Clone, Default)]
pub struct Booster {
    config: Config,
}

impl Booster {
    /// A fresh booster with the default [`Config`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: Config::default(),
        }
    }

    /// A booster with an explicit [`Config`].
    #[must_use]
    pub fn with_config(config: Config) -> Self {
        Self { config }
    }

    /// The booster's configuration.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Fit an ensemble (spec §06.6): `f0 = link(weighted mean)`, then per round a
    /// full-precision `grad_hess` pass → `grow_oblivious_tree` → `update_raw`, until
    /// `n_trees` rounds or a round cannot split. Emits an `Exact` [`Model`].
    ///
    /// # Errors
    /// [`PbError::InvalidConfig`] on a bad config; [`PbError::ShapeMismatch`] on a
    /// length mismatch; plus any propagated [`Loss`]/binning/grow error.
    pub fn fit(&self, x: &BinnedMatrix, y: &[f32], spec: &FitSpec) -> Result<Model, PbError> {
        boost::fit(&self.config, x, y, spec, &CatEncoderStore::new())
    }

    /// Fit from the explicit training-matrix role and persist its frozen
    /// categorical encoder store into the emitted [`ModelSchema`].
    ///
    /// This is the §03.2a/§04 audit seam: tree growth may use leakage-free
    /// categorical train bins, while the resulting model carries the full-data
    /// encoders needed to rebuild serve/audit bins after serialization.
    ///
    /// # Errors
    /// Same as [`Booster::fit`], plus [`PbError::InvalidInput`] if a categorical
    /// axis names an encoder absent from `cat_encoders`.
    pub fn fit_train(
        &self,
        x: &TrainBinnedMatrix,
        y: &[f32],
        spec: &FitSpec,
        cat_encoders: CatEncoderStore,
    ) -> Result<Model, PbError> {
        // Thread the encoders into the fit so every model built — including each OuterBag /
        // GreedySelect member validated inside `soup_models` — carries them, not just the
        // single-fit result (R-CATSERVE: the souped/member models also serve via these).
        let model = boost::fit(&self.config, &x.0, y, spec, &cat_encoders)?;
        model.validate()?;
        Ok(model)
    }

    /// Fit a native-softmax (multinomial) multiclass model from the explicit training-matrix role,
    /// persisting the frozen categorical encoders into every per-class [`ModelSchema`].
    ///
    /// `y` holds integer class labels in `0..n_classes` (as `f32`); `class_labels` are the
    /// human-readable labels parallel to the `n_classes` per-class models. See
    /// [`crate::engine::MultiClassModel`] and `design/multiclass-design.md`. The `spec.loss` field
    /// is ignored (the softmax gradient is coupled across classes and cannot be a scalar
    /// [`crate::loss::Loss`]); `spec`'s weight/exposure-free/seed/interaction/credibility/monotone
    /// fields are honored per the v1 support matrix.
    ///
    /// # Errors
    /// Same as the internal `boost::fit_multiclass`: [`PbError::InvalidConfig`] /
    /// [`PbError::InvalidInput`] / [`PbError::ShapeMismatch`] on bad config/labels, plus any
    /// propagated grow/update/validate error.
    pub fn fit_multiclass_train(
        &self,
        x: &TrainBinnedMatrix,
        y: &[f32],
        n_classes: usize,
        class_labels: &[String],
        spec: &FitSpec,
        cat_encoders: CatEncoderStore,
    ) -> Result<MultiClassModel, PbError> {
        let model = boost::fit_multiclass(
            &self.config,
            &x.0,
            y,
            n_classes,
            class_labels,
            spec,
            &cat_encoders,
        )?;
        model.validate()?;
        Ok(model)
    }

    /// Fit a native-softmax multiclass model from a numeric-only binned design (no persisted
    /// categorical encoders). See [`Booster::fit_multiclass_train`] for the categorical/train-role
    /// entry; the `spec.loss` field is ignored (softmax gradients are coupled across classes).
    ///
    /// # Errors
    /// Same as the internal `boost::fit_multiclass`.
    pub fn fit_multiclass(
        &self,
        x: &BinnedMatrix,
        y: &[f32],
        n_classes: usize,
        class_labels: &[String],
        spec: &FitSpec,
    ) -> Result<MultiClassModel, PbError> {
        boost::fit_multiclass(
            &self.config,
            x,
            y,
            n_classes,
            class_labels,
            spec,
            &CatEncoderStore::new(),
        )
    }
}

/// The per-fit specification (spec §2.9): objective + per-row data + constraints +
/// the deterministic seed.
pub struct FitSpec<'a> {
    /// The objective.
    pub loss: &'a dyn Loss,
    /// Optional per-row weights.
    pub weight: Option<&'a [f32]>,
    /// Optional per-row exposure (offset = `log(e)`; anchors base level = 1.000).
    pub exposure: Option<&'a [f32]>,
    /// Monotone constraints keyed by feature name.
    pub monotone: MonotoneMap,
    /// Interaction-order limit + optional group whitelist.
    pub interaction: InteractionPolicy,
    /// Per-leaf credibility floors + `path_smooth` (§07). Threaded here alongside the
    /// other §07 constraints (`monotone`, `interaction`); the default is exactly inert.
    pub credibility: CredibilityFloor,
    /// Precomputed validation-holdout mask (the "one honest holdout" contract for ordered
    /// target statistics, design/ordered-ts-early-stopping.md). `Some(mask)` overrides the
    /// internal `validation_fraction` carve: rows with `mask[i] == true` form the early-stop
    /// validation set and MUST be the same rows the categorical encoders were blinded to at
    /// binning time. `None` = legacy internal carve (bit-identical).
    pub fixed_holdout: Option<&'a [bool]>,
    /// Group id per row for GROUP-AWARE outer bagging (2026-09-07). When `Some`, every bag is a
    /// subsample of whole groups — drawn by the same deterministic stratified subagging over
    /// group indices, a group's stratum being the majority stratum of its rows — so no group
    /// straddles a bag's in-bag/out-of-bag boundary and the out-of-bag rows are honest evidence
    /// on panel data (a policy's other years never trained the jury that scores it). `None`
    /// draws rows, byte-identical to before.
    pub bag_groups: Option<&'a [u32]>,
    /// The deterministic base seed threaded through every randomized stage.
    pub seed: u64,
}
