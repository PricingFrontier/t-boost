//! Runtime-only scoring views (spec §10.2a / §11.9).
//!
//! [`ScoringBank`] is a load-derived view of a [`crate::Model`]: it packs each tree
//! into a cache-friendly row for prediction and carries a clone of the model's §G1
//! cell-basis correction, but it is never serialized and never owns independent model
//! semantics. Every scoring path is tested bit-equal to [`crate::Model::score_trees_row`]
//! (callers fold `f0` into `offset` themselves — see [`ScoringBank::score_row`]).

use crate::data::{BinnedMatrix, BorderGrid};
use crate::engine::{low_bit, CorrectionBank, Model, LEGACY_MAX_DEPTH, MAX_DEPTH};
use crate::error::PbError;
use crate::explain::TableBank;
use rayon::prelude::*;

/// A 64-byte runtime scoring row for one depth-`<=3` oblivious tree.
///
/// The leaves are byte-exact copies of the model leaves; `alpha` is kept separate so
/// DART/Nesterov/ensemble weights do not mutate leaf values in this derived view.
/// This type is never serialized.
///
/// This layout is frozen at spec §14.108's one-cache-line shape and is used ONLY for
/// models whose every tree is at most [`LEGACY_MAX_DEPTH`] deep — i.e. every model a
/// pre-lift build could have produced. A lifted model takes [`ArenaTree`] instead, so
/// nothing about the depth knob perturbs the default serving path.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PackedTree {
    /// Axis id per level. Valid only for models whose split axes fit in `u8`.
    pub feat: [u8; 3],
    /// Threshold bin per level.
    pub thresh: [u8; 3],
    /// Missing-left bits packed by level.
    pub miss: u8,
    /// Tree depth, `1..=3`.
    pub depth: u8,
    /// Tree multiplier.
    pub alpha: f32,
    /// Leaf lookup table.
    pub leaf: [f32; 8],
    _pad: [u8; 20],
}

impl PackedTree {
    fn score_row(&self, row: &[u8]) -> Result<f64, PbError> {
        let mut idx = 0usize;
        for level in 0..usize::from(self.depth) {
            let axis = *self.feat.get(level).ok_or_else(|| PbError::Internal {
                what: "packed tree level escaped feat array".into(),
            })? as usize;
            let thresh = *self.thresh.get(level).ok_or_else(|| PbError::Internal {
                what: "packed tree level escaped thresh array".into(),
            })?;
            let bin = *row.get(axis).ok_or_else(|| PbError::ShapeMismatch {
                what: format!("row has no axis {axis} for packed scoring"),
            })?;
            let missing_left = ((self.miss >> level) & 1) != 0;
            idx |= usize::from(low_bit(bin, thresh, missing_left)) << level;
        }
        let leaf = self
            .leaf
            .get(idx)
            .copied()
            .ok_or_else(|| PbError::Internal {
                what: format!("packed leaf index {idx} escaped 0..{}", self.leaf.len()),
            })?;
        Ok(f64::from(self.alpha) * f64::from(leaf))
    }
}

/// The [`ArenaTree`] target size — half a 64-byte cache line, so two descriptors stream per
/// line. Pinned by `arena_tree_is_half_a_cache_line`.
const ARENA_TREE_SIZE: usize = 32;

/// Explicit tail padding taking [`ArenaTree`]'s body to [`ARENA_TREE_SIZE`].
///
/// The body is `feat[MAX_DEPTH] + thresh[MAX_DEPTH] + miss(1) + depth(1) + align(2) +
/// alpha(4) + leaf_off(4)`. Computing the pad rather than writing it turns "the depth lift
/// silently doubled the scoring row" — which is precisely what a literal `[u8; 8]` did when
/// `MAX_DEPTH` went 6 → 8 — into a compile-time subtraction overflow.
const ARENA_TREE_PAD: usize = ARENA_TREE_SIZE - (2 * MAX_DEPTH + 1 + 1 + 2 + 4 + 4);

/// A 32-byte runtime scoring row for one LIFTED (depth > 3) oblivious tree.
///
/// `PackedTree` inlines its 8 leaves and is padded to exactly one 64-byte cache line
/// (spec §14.108). That layout does not survive the depth lift, and the high-order lift
/// makes the point overwhelming: at `MAX_DEPTH = 8` an inlined tree would be 8 feats +
/// 8 threshes + miss + depth + alpha + **256 leaves** = 1,046 bytes → seventeen cache
/// lines, wrecking §11's "stream the entire ScoringBank exactly once" bandwidth argument
/// outright.
///
/// So a lifted bank splits the leaf table into a side arena and keeps only a descriptor
/// here: `feat[MAX_DEPTH] + thresh[MAX_DEPTH] + miss + depth + (2 align) + alpha +
/// leaf_off` — 28 bytes of body at `MAX_DEPTH = 8`, sized and aligned to 32 → **two trees
/// per cache line, denser than today's one** (`arena_tree_is_half_a_cache_line` asserts it,
/// the way `packed_tree_is_one_cache_line` asserts the frozen row). Scoring touches exactly
/// one leaf per tree per row, so the arena read is a single random 4-byte access into a
/// `<= 1 KiB` block — the same access pattern the in-struct array already had, one
/// indirection out.
///
/// `MAX_DEPTH = 8` is the LAST depth at which this descriptor still fits 32 bytes (depth 9
/// would take the body to 30 and, with alignment, still fit — but the `u8` leaf id wall
/// stops it first either way). `_pad` is sized from the constant so the arithmetic is
/// stated in the type rather than left to the layout algorithm to arrange quietly.
///
/// A depth-`<=3` model keeps the [`ScoringBank::Packed`] layout VERBATIM; nothing about
/// this variant is reachable unless the fit actually used the lift.
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArenaTree {
    /// Axis id per level. Valid only for models whose split axes fit in `u8`.
    feat: [u8; MAX_DEPTH],
    /// Threshold bin per level.
    thresh: [u8; MAX_DEPTH],
    /// Missing-left bits packed by level (one bit per level; `MAX_DEPTH <= 8`).
    miss: u8,
    /// Tree depth, `1..=MAX_DEPTH`.
    depth: u8,
    /// Tree multiplier.
    alpha: f32,
    /// Start of this tree's `2^depth` leaf values in the bank's leaf arena.
    leaf_off: u32,
    // 8 + 8 + 1 + 1 + (2 align) + 4 + 4 = 28 bytes of body at MAX_DEPTH = 8; pad to the
    // 32-byte target so the size is stated in the type rather than left to layout. Derived
    // from the constant, not written as a literal, so a future depth move is a compile
    // error here (via `ARENA_TREE_PAD`) rather than a silent doubling to 64 bytes — which is
    // exactly what the 6 -> 8 lift would have done to the hand-written `[u8; 8]` this
    // replaced. Asserted by `arena_tree_is_half_a_cache_line`.
    _pad: [u8; ARENA_TREE_PAD],
}

impl ArenaTree {
    fn score_row(&self, row: &[u8], arena: &[f32]) -> Result<f64, PbError> {
        let mut idx = 0usize;
        for level in 0..usize::from(self.depth) {
            let axis = *self.feat.get(level).ok_or_else(|| PbError::Internal {
                what: "arena tree level escaped feat array".into(),
            })? as usize;
            let thresh = *self.thresh.get(level).ok_or_else(|| PbError::Internal {
                what: "arena tree level escaped thresh array".into(),
            })?;
            let bin = *row.get(axis).ok_or_else(|| PbError::ShapeMismatch {
                what: format!("row has no axis {axis} for arena scoring"),
            })?;
            let missing_left = ((self.miss >> level) & 1) != 0;
            idx |= usize::from(low_bit(bin, thresh, missing_left)) << level;
        }
        let leaf = *(self.leaf_off as usize)
            .checked_add(idx)
            .and_then(|at| arena.get(at))
            .ok_or_else(|| PbError::Internal {
                what: format!(
                    "arena leaf index {} escaped the leaf arena (len {})",
                    self.leaf_off as usize + idx,
                    arena.len()
                ),
            })?;
        Ok(f64::from(self.alpha) * f64::from(leaf))
    }
}

/// Runtime scoring row for models whose split axes do not fit in `u8`.
///
/// Fields stay private because this is a derived view, not a wire or modeling API.
#[derive(Debug, Clone, PartialEq)]
pub struct WideTree {
    feat: [u32; MAX_DEPTH],
    thresh: [u8; MAX_DEPTH],
    miss: u8,
    depth: u8,
    alpha: f32,
    /// Owned rather than arena-indexed: the wide variant is already the
    /// "cache layout forfeited" fallback, so there is nothing left to protect.
    leaf: Vec<f32>,
}

impl WideTree {
    fn score_row(&self, row: &[u8]) -> Result<f64, PbError> {
        let mut idx = 0usize;
        for level in 0..usize::from(self.depth) {
            let axis = *self.feat.get(level).ok_or_else(|| PbError::Internal {
                what: "wide tree level escaped feat array".into(),
            })? as usize;
            let thresh = *self.thresh.get(level).ok_or_else(|| PbError::Internal {
                what: "wide tree level escaped thresh array".into(),
            })?;
            let bin = *row.get(axis).ok_or_else(|| PbError::ShapeMismatch {
                what: format!("row has no axis {axis} for wide scoring"),
            })?;
            let missing_left = ((self.miss >> level) & 1) != 0;
            idx |= usize::from(low_bit(bin, thresh, missing_left)) << level;
        }
        let leaf = self
            .leaf
            .get(idx)
            .copied()
            .ok_or_else(|| PbError::Internal {
                what: "wide leaf index escaped the tree's leaf table".into(),
            })?;
        Ok(f64::from(self.alpha) * f64::from(leaf))
    }
}

/// Runtime-only path-A scoring view.
#[derive(Debug, Clone, PartialEq)]
pub enum ScoringBank {
    /// All split axes fit in `u8`, so each tree can use the compact 64-byte layout.
    Packed {
        /// Packed trees in stored model order.
        trees: Vec<PackedTree>,
        /// Clone of `Model.correction` (the §G1 cell-basis correction), if any.
        /// [`ScoringBank::score_row`] adds its delta after the tree sum, exactly
        /// mirroring [`crate::Model::score_trees_row`], so corrected models stay
        /// bit-equal through this view.
        correction: Option<CorrectionBank>,
    },
    /// At least one tree is deeper than [`LEGACY_MAX_DEPTH`] (the `max_depth` lift), so
    /// its `2^depth` leaves no longer fit the frozen one-cache-line row. Descriptors stay
    /// in `trees` (32 bytes each — DENSER than `Packed`) and the leaf tables live
    /// contiguously in `leaf_arena`. Same axis-width precondition as `Packed`.
    Arena {
        /// Arena-descriptor trees in stored model order.
        trees: Vec<ArenaTree>,
        /// Every tree's `2^depth` leaf values, concatenated in stored model order.
        leaf_arena: Vec<f32>,
        /// Clone of `Model.correction`; same role as the `Packed` variant's field.
        correction: Option<CorrectionBank>,
    },
    /// A model has at least one axis id above 255; keep `u32` axes instead of
    /// truncating. Correctness is identical, only the compact layout is forfeited.
    Wide {
        /// Wide-axis trees in stored model order.
        trees: Vec<WideTree>,
        /// Clone of `Model.correction`; same role as the `Packed` variant's field.
        correction: Option<CorrectionBank>,
    },
}

impl ScoringBank {
    /// Build a runtime scoring view from a model, after validating it.
    ///
    /// # Errors
    /// Propagates [`Model::validate`] failures.
    pub fn from_model(model: &Model) -> Result<Self, PbError> {
        model.validate()?;
        Self::from_validated_model(model)
    }

    /// Build a runtime scoring view from a model the CALLER has already validated (or knows to be
    /// valid by construction, e.g. straight out of `Booster::fit`) — skips the `Model::validate`
    /// pass `from_model` runs. `Model::validate` is O(features + trees), not O(rows), but re-running
    /// it on every single-row/small-batch predict call is still a fixed per-call cost a hot serving
    /// path shouldn't pay redundantly. Used by [`crate::engine::Model::score_trees_prevalidated`],
    /// whose own contract already requires the caller to have validated the model/matrix pairing.
    ///
    /// Building the bank itself is unconditional either way (same O(trees) pack, no O(rows) work) —
    /// only the validation pass is skipped. If `model` is in fact malformed, the packing loops below
    /// still use `.get(..).ok_or_else(..)` throughout (never an unchecked index), so this cannot
    /// panic; it can only produce a bank that silently reflects the malformed model, exactly as
    /// `score_trees_prevalidated`'s own no-matrix-revalidation contract already permits upstream of
    /// this call.
    ///
    /// # Errors
    /// [`PbError::Internal`] if a tree's depth/split count is inconsistent with its declared depth
    /// (the packing loops' own bounds checks); never [`Model::validate`]'s checks.
    pub(crate) fn from_validated_model(model: &Model) -> Result<Self, PbError> {
        let packed_ok = model
            .trees
            .iter()
            .flat_map(|(_, tree)| tree.splits.iter())
            .all(|split| u8::try_from(split.axis).is_ok());
        let correction = model.correction.clone();
        // The lift is opt-in AND per-model: a bank is `Arena` only when the fit actually
        // produced a tree deeper than the legacy cap, so every model a pre-lift build
        // could have written still takes the byte-identical `Packed` path.
        let lifted = model
            .trees
            .iter()
            .any(|(_, tree)| usize::from(tree.depth) > LEGACY_MAX_DEPTH);
        if packed_ok && lifted {
            let mut trees = Vec::with_capacity(model.trees.len());
            let mut leaf_arena: Vec<f32> = Vec::new();
            for (alpha, tree) in &model.trees {
                let mut feat = [0u8; MAX_DEPTH];
                let mut thresh = [0u8; MAX_DEPTH];
                let mut miss = 0u8;
                for (level, split) in tree.splits.iter().enumerate() {
                    let feat_slot = feat.get_mut(level).ok_or_else(|| PbError::Internal {
                        what: "tree depth escaped arena feat array".into(),
                    })?;
                    *feat_slot = u8::try_from(split.axis).map_err(|_| PbError::Internal {
                        what: "packed_ok accepted a non-u8 axis".into(),
                    })?;
                    let thresh_slot = thresh.get_mut(level).ok_or_else(|| PbError::Internal {
                        what: "tree depth escaped arena thresh array".into(),
                    })?;
                    *thresh_slot = split.bin_le;
                    if split.missing_left {
                        miss |= 1_u8 << level;
                    }
                }
                // `from_validated_model` promises never to panic even on a malformed model
                // (see its doc), and `tree.depth` is a raw `u8` off the wire — so bound it
                // before the shift rather than trusting `1usize << depth`.
                let depth = usize::from(tree.depth);
                if depth != tree.splits.len() || depth > MAX_DEPTH {
                    return Err(PbError::Internal {
                        what: format!(
                            "arena bank: tree depth {depth} inconsistent with {} splits (cap {MAX_DEPTH})",
                            tree.splits.len()
                        ),
                    });
                }
                let n_leaves = 1usize << depth;
                let reached = tree
                    .leaves
                    .get(..n_leaves)
                    .ok_or_else(|| PbError::Internal {
                        what: "tree leaf table shorter than 2^depth".into(),
                    })?;
                let leaf_off = u32::try_from(leaf_arena.len()).map_err(|_| PbError::Internal {
                    what: "scoring leaf arena exceeded u32 addressing".into(),
                })?;
                leaf_arena.extend_from_slice(reached);
                trees.push(ArenaTree {
                    feat,
                    thresh,
                    miss,
                    depth: tree.depth,
                    alpha: *alpha,
                    leaf_off,
                    _pad: [0; ARENA_TREE_PAD],
                });
            }
            return Ok(ScoringBank::Arena {
                trees,
                leaf_arena,
                correction,
            });
        }
        if packed_ok {
            let mut trees = Vec::with_capacity(model.trees.len());
            for (alpha, tree) in &model.trees {
                let mut feat = [0u8; 3];
                let mut thresh = [0u8; 3];
                let mut miss = 0u8;
                for (level, split) in tree.splits.iter().enumerate() {
                    let feat_slot = feat.get_mut(level).ok_or_else(|| PbError::Internal {
                        what: "tree depth escaped packed feat array".into(),
                    })?;
                    *feat_slot = u8::try_from(split.axis).map_err(|_| PbError::Internal {
                        what: "packed_ok accepted a non-u8 axis".into(),
                    })?;
                    let thresh_slot = thresh.get_mut(level).ok_or_else(|| PbError::Internal {
                        what: "tree depth escaped packed thresh array".into(),
                    })?;
                    *thresh_slot = split.bin_le;
                    if split.missing_left {
                        miss |= 1_u8 << level;
                    }
                }
                let mut leaf = [0.0_f32; 8];
                for (slot, v) in leaf.iter_mut().zip(tree.leaves.iter()) {
                    *slot = *v;
                }
                trees.push(PackedTree {
                    feat,
                    thresh,
                    miss,
                    depth: tree.depth,
                    alpha: *alpha,
                    leaf,
                    _pad: [0; 20],
                });
            }
            Ok(ScoringBank::Packed { trees, correction })
        } else {
            let mut trees = Vec::with_capacity(model.trees.len());
            for (alpha, tree) in &model.trees {
                let mut feat = [0u32; MAX_DEPTH];
                let mut thresh = [0u8; MAX_DEPTH];
                let mut miss = 0u8;
                for (level, split) in tree.splits.iter().enumerate() {
                    let feat_slot = feat.get_mut(level).ok_or_else(|| PbError::Internal {
                        what: "tree depth escaped wide feat array".into(),
                    })?;
                    *feat_slot = split.axis;
                    let thresh_slot = thresh.get_mut(level).ok_or_else(|| PbError::Internal {
                        what: "tree depth escaped wide thresh array".into(),
                    })?;
                    *thresh_slot = split.bin_le;
                    if split.missing_left {
                        miss |= 1_u8 << level;
                    }
                }
                trees.push(WideTree {
                    feat,
                    thresh,
                    miss,
                    depth: tree.depth,
                    alpha: *alpha,
                    leaf: tree.leaves.clone(),
                });
            }
            Ok(ScoringBank::Wide { trees, correction })
        }
    }

    /// Score one already-binned row in raw-score space: `offset + Σ tree(row) +
    /// correction(row)`.
    ///
    /// This bank never stores the model's intercept `f0` — unlike
    /// [`crate::Model::score_trees_row`], which starts its accumulator at `f0 + offset`.
    /// To reproduce it bit-for-bit, callers MUST fold `f0` into `offset` themselves
    /// (`offset = model.f0 + row_offset`); every test in this module calls it that way.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if the row lacks a referenced axis.
    pub fn score_row(&self, row: &[u8], offset: f32) -> Result<f32, PbError> {
        let mut acc = f64::from(offset);
        match self {
            ScoringBank::Packed { trees, .. } => {
                for tree in trees {
                    acc += tree.score_row(row)?;
                }
            }
            ScoringBank::Arena {
                trees, leaf_arena, ..
            } => {
                for tree in trees {
                    acc += tree.score_row(row, leaf_arena)?;
                }
            }
            ScoringBank::Wide { trees, .. } => {
                for tree in trees {
                    acc += tree.score_row(row)?;
                }
            }
        }
        acc += self.correction_delta(row)?;
        Ok(acc as f32)
    }

    /// The cell-basis correction's additive contribution for one already-binned row
    /// (`0.0` when this bank carries no correction). Mirrors
    /// [`crate::Model::correction_delta`] bit-for-bit — same table order, same
    /// flat-index arithmetic, same `f64` accumulation — so a bank built from a
    /// corrected model stays bit-equal to [`crate::Model::score_trees_row`] (spec
    /// §10.2a / G1).
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if `row` lacks a referenced axis;
    /// [`PbError::InvalidInput`] if a bin escapes `bin_to_cell`;
    /// [`PbError::Internal`] if correction metadata is inconsistent.
    #[inline]
    pub fn correction_delta(&self, row: &[u8]) -> Result<f64, PbError> {
        let correction = match self {
            ScoringBank::Packed { correction, .. }
            | ScoringBank::Arena { correction, .. }
            | ScoringBank::Wide { correction, .. } => correction,
        };
        let Some(bank) = correction else {
            return Ok(0.0);
        };
        let mut acc = 0.0_f64;
        for table in &bank.tables {
            let mut flat = 0usize;
            for (k, &axis) in table.axes.iter().enumerate() {
                let bin = *row
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

    /// Number of trees in the view.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            ScoringBank::Packed { trees, .. } => trees.len(),
            ScoringBank::Arena { trees, .. } => trees.len(),
            ScoringBank::Wide { trees, .. } => trees.len(),
        }
    }

    /// `true` if the view contains no trees.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone, PartialEq)]
struct FlatTable {
    axes: Vec<u32>,
    /// Per-axis merged-cell -> band map (banded tables); `None` = the axis is the merged grid.
    band_of: Vec<Option<Vec<u32>>>,
    shape: Vec<u32>,
    strides: Vec<usize>,
    values: Vec<f64>,
}

impl FlatTable {
    fn offset(&self, x_cells: &[u32]) -> Result<usize, PbError> {
        let mut off = 0usize;
        for (((&raw, &dim), &stride), band_of) in self
            .axes
            .iter()
            .zip(&self.shape)
            .zip(&self.strides)
            .zip(&self.band_of)
        {
            let cell = *x_cells
                .get(raw as usize)
                .ok_or_else(|| PbError::ShapeMismatch {
                    what: format!("x_cells missing raw feature {raw} for flat table scoring"),
                })?;
            let cell = match band_of {
                None => cell,
                Some(map) => *map
                    .get(cell as usize)
                    .ok_or_else(|| PbError::InvalidInput {
                        what: format!("raw feature {raw} cell {cell} outside its band map"),
                    })?,
            };
            if cell >= dim {
                return Err(PbError::InvalidInput {
                    what: format!("raw feature {raw} cell {cell} outside merged extent {dim}"),
                });
            }
            let cell = usize::try_from(cell).map_err(|_| PbError::Internal {
                what: "merged cell id exceeded usize".into(),
            })?;
            off = off
                .checked_add(cell.checked_mul(stride).ok_or_else(|| PbError::Internal {
                    what: "flat table offset multiplication overflowed".into(),
                })?)
                .ok_or_else(|| PbError::Internal {
                    what: "flat table offset addition overflowed".into(),
                })?;
        }
        Ok(off)
    }

    fn eval(&self, x_cells: &[u32]) -> Result<f64, PbError> {
        let off = self.offset(x_cells)?;
        self.values
            .get(off)
            .copied()
            .ok_or_else(|| PbError::Internal {
                what: "flat table offset escaped values arena".into(),
            })
    }
}

/// Runtime-only path-B table scorer.
///
/// This is the flattened single-arena view from spec §10.3: every DENSE table in a
/// [`TableBank`] is validated once, copied into row-major flat storage, and then rows
/// are scored by digitizing model bins into merged-grid cells exactly once per row.
/// Dense-only: [`TableScoringBank::from_bank`] rejects a bank that carries factored
/// order-3 tables rather than silently scoring short of them (see its docs). Like
/// [`ScoringBank`], it is never serialized and owns no independent model semantics.
#[derive(Debug, Clone, PartialEq)]
pub struct TableScoringBank {
    f0: f64,
    tables: Vec<FlatTable>,
    merged_grids: Vec<BorderGrid>,
    /// P1 multi-channel (design/multichannel-categoricals.md §4.3): needed to rebuild the
    /// flattened joint-cell space for a multi-channel categorical raw feature at
    /// [`Self::score_binned`] time (`merged_grids` alone carries only a placeholder for such a
    /// raw feature — see `explain::MergedGrids::border_grids`'s doc). Empty for a model with no
    /// categorical channels — the common case, and the only case before P1 existed.
    cat_encoders: crate::cat::CatEncoderStore,
}

impl TableScoringBank {
    /// Build a runtime table scorer from an exact [`TableBank`] and the frozen categorical
    /// encoders it was purified against (`model.schema.cat_encoders`/`TableModel.schema.
    /// cat_encoders` — needed only for a P1 multi-channel raw feature's flattened joint-cell
    /// space; pass [`crate::cat::CatEncoderStore::new`] for a model with no categoricals).
    ///
    /// Dense tables only: `bank.factored` (over-budget order-3 effects kept in
    /// per-tree-box form, §08.10) is NOT carried by this flat view, so a bank that has
    /// any is rejected rather than silently scored short. Score such a bank via
    /// [`TableBank::score`] directly, or rebuild it with `OverflowPolicy::Error` or a
    /// larger `max_table_cells` so no support needs factoring.
    ///
    /// # Errors
    /// [`PbError::InvalidConfig`] if `bank.factored` is non-empty;
    /// [`PbError::ShapeMismatch`] if a table's axes and tensor shape disagree;
    /// [`PbError::Internal`] if stride/product arithmetic overflows.
    pub fn from_bank(
        bank: &TableBank,
        cat_encoders: &crate::cat::CatEncoderStore,
    ) -> Result<Self, PbError> {
        if !bank.factored.is_empty() {
            return Err(PbError::InvalidConfig {
                what: format!(
                    "TableScoringBank::from_bank: bank carries {} factored high-order table(s); \
                     this flat view sums dense tables only and would silently drop their \
                     contribution — score via TableBank::score directly, or rebuild the bank \
                     with OverflowPolicy::Error or a larger max_table_cells",
                    bank.factored.len()
                ),
            });
        }
        let mut tables = Vec::with_capacity(bank.tables.len());
        for table in &bank.tables {
            let shape = table.values.shape_u32().to_vec();
            if shape.len() != table.axes.len() {
                return Err(PbError::ShapeMismatch {
                    what: format!(
                        "table order {} has {} axes but tensor rank {}",
                        table.u.order(),
                        table.axes.len(),
                        shape.len()
                    ),
                });
            }
            for (axis, &dim) in table.axes.iter().zip(&shape) {
                if axis.cells != dim {
                    return Err(PbError::ShapeMismatch {
                        what: format!(
                            "table axis raw {} cells {} != tensor dim {dim}",
                            axis.raw.0, axis.cells
                        ),
                    });
                }
            }
            let mut expected = 1usize;
            for &dim in &shape {
                expected = expected
                    .checked_mul(usize::try_from(dim).map_err(|_| PbError::Internal {
                        what: "tensor dim exceeded usize".into(),
                    })?)
                    .ok_or_else(|| PbError::Internal {
                        what: "flat table cell count overflowed".into(),
                    })?;
            }
            if expected != table.values.values().len() {
                return Err(PbError::ShapeMismatch {
                    what: format!(
                        "table values len {} != product(shape) {expected}",
                        table.values.values().len()
                    ),
                });
            }
            let mut strides = vec![1usize; shape.len()];
            let mut suffix = 1usize;
            for (slot, &dim) in strides.iter_mut().rev().zip(shape.iter().rev()) {
                *slot = suffix;
                suffix = suffix
                    .checked_mul(usize::try_from(dim).map_err(|_| PbError::Internal {
                        what: "tensor dim exceeded usize".into(),
                    })?)
                    .ok_or_else(|| PbError::Internal {
                        what: "flat table stride overflowed".into(),
                    })?;
            }
            tables.push(FlatTable {
                axes: table.axes.iter().map(|axis| axis.raw.0).collect(),
                band_of: table.axes.iter().map(|axis| axis.band_of.clone()).collect(),
                shape,
                strides,
                values: table.values.values().to_vec(),
            });
        }
        Ok(Self {
            f0: bank.f0,
            tables,
            merged_grids: bank.merged_grids.clone(),
            cat_encoders: cat_encoders.clone(),
        })
    }

    /// Score one merged-cell row in raw-score space.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if `x_cells` lacks a referenced raw feature;
    /// [`PbError::InvalidInput`] if a cell id exceeds a table extent.
    pub fn score_cells_row(&self, x_cells: &[u32]) -> Result<f64, PbError> {
        let mut acc = self.f0;
        for table in &self.tables {
            acc += table.eval(x_cells)?;
        }
        Ok(acc)
    }

    /// Score an already-binned matrix through the flat table arena in raw-score space.
    ///
    /// The input matrix must use the model's original grids; this scorer derives the
    /// merged-grid cell ids once per row and then reads the flat tables in fixed order.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] for width/length mismatches; [`PbError::InvalidInput`]
    /// if a model bin cannot be mapped to the bank's merged grid.
    pub fn score_binned(&self, x: &BinnedMatrix, out: &mut [f64]) -> Result<(), PbError> {
        let n_rows = x.n_rows as usize;
        if out.len() != n_rows {
            return Err(PbError::ShapeMismatch {
                what: format!("out len {} != n_rows {n_rows}", out.len()),
            });
        }
        let maps = build_cell_maps(&self.merged_grids, &self.cat_encoders, x)?;
        let n_cells = self.merged_grids.len();
        // Bit-identical to the serial loop (see `score_bank_binned`); parallelised over rows on
        // the ambient rayon pool.
        out.par_iter_mut().enumerate().try_for_each_init(
            || vec![0u32; n_cells],
            |cells, (row, dst)| -> Result<(), PbError> {
                fill_row_cells(x, &maps, row, cells)?;
                *dst = self.score_cells_row(cells)?;
                Ok(())
            },
        )?;
        Ok(())
    }
}

/// Per-raw-feature cell-computation strategy, built once per score call and reused across
/// every row. P1 multi-channel (design/multichannel-categoricals.md §4.3): a raw feature with
/// more than one categorical channel axis needs the flattened joint-cell machinery shared with
/// `explain.rs` ([`crate::cat::JointCatAxis`]) instead of the original single-axis bin->cell
/// map — this is a THIRD, independent construction of "model bins -> merged cell" (the other
/// two are `explain.rs`'s `MergedGrids` and, transitively, `fill_support`), so it MUST agree
/// with them cell-for-cell or a pruned/deployed `TableModel`'s predictions would silently
/// diverge from the tree ensemble it was built from; both now share `JointCatAxis::build`
/// verbatim rather than three independently-written implementations.
#[derive(Debug, Clone)]
pub(crate) enum RawCellMap {
    /// An ordinary raw feature (numeric, or today's single-channel categorical): its one model
    /// axis and a precomputed bin -> merged-cell map (`0..model.n_bins`), byte-identical to
    /// the original (pre-P1) `build_cell_maps` output.
    Single { axis: usize, map: Vec<u32> },
    /// A P1 multi-channel categorical raw feature's flattened joint per-level cell space.
    Joint(crate::cat::JointCatAxis),
}

/// Build one [`RawCellMap`] per raw feature (`x.provenance`-derived raw-feature count, which
/// may be smaller than `x.data.len()`/`x.grids.len()` once any raw feature owns more than one
/// axis — P1 multi-channel).
///
/// # Errors
/// [`PbError::ShapeMismatch`] if the matrix's data/grids/provenance lengths disagree, or the
/// table bank's merged-grid count disagrees with the matrix's raw feature count;
/// [`PbError::InvalidInput`] if a model bin cannot be mapped to the bank's merged grid;
/// propagates [`crate::cat::JointCatAxis::build`] errors for a multi-channel raw feature.
pub(crate) fn build_cell_maps(
    merged_grids: &[BorderGrid],
    cat_encoders: &crate::cat::CatEncoderStore,
    x: &BinnedMatrix,
) -> Result<Vec<RawCellMap>, PbError> {
    check_cell_map_matrix(x)?;
    build_cell_maps_for(merged_grids, cat_encoders, &x.grids, &x.provenance)
}

/// The per-matrix half of [`build_cell_maps`]: data, grids and provenance are all axis-indexed
/// and equal in length, and every column holds `n_rows` bins. Cheap (`O(n_axes)`), so a caller
/// scoring through cached maps ([`CellMaps`]) still runs it on every matrix.
fn check_cell_map_matrix(x: &BinnedMatrix) -> Result<(), PbError> {
    if x.data.len() != x.provenance.len() || x.grids.len() != x.provenance.len() {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "matrix has {} data column(s), {} grid(s), {} provenance entries — all three \
                 must be axis-indexed and equal",
                x.data.len(),
                x.grids.len(),
                x.provenance.len()
            ),
        });
    }
    let n_rows = x.n_rows as usize;
    for (axis, col) in x.data.iter().enumerate() {
        if col.len() != n_rows {
            return Err(PbError::ShapeMismatch {
                what: format!("matrix column {axis} len {} != n_rows {n_rows}", col.len()),
            });
        }
    }
    Ok(())
}

/// The model-only half of [`build_cell_maps`]: one [`RawCellMap`] per raw feature, from the
/// model's own axis `grids`/`provenance` (which every serve matrix must match exactly). Depends on
/// nothing row-specific, so it can be built once per model and reused across calls — see
/// [`CellMaps`]. Building it is the dominant fixed cost of a small predict (8 ms per call on a
/// high-cardinality multi-channel categorical, whose joint-cell tables are rebuilt here).
pub(crate) fn build_cell_maps_for(
    merged_grids: &[BorderGrid],
    cat_encoders: &crate::cat::CatEncoderStore,
    grids: &[BorderGrid],
    provenance: &[crate::data::AxisProvenance],
) -> Result<Vec<RawCellMap>, PbError> {
    use crate::data::{axes_for_raw, n_raw_features, FeatureId};
    if grids.len() != provenance.len() {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "{} grid(s) but {} provenance entries — both must be axis-indexed and equal",
                grids.len(),
                provenance.len()
            ),
        });
    }
    let n_raw = n_raw_features(provenance);
    if merged_grids.len() != n_raw {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "table bank has {} merged grid(s), matrix provenance implies {n_raw} raw \
                 feature(s)",
                merged_grids.len()
            ),
        });
    }
    let mut out = Vec::with_capacity(n_raw);
    for r in 0..n_raw {
        let raw = FeatureId(u32::try_from(r).map_err(|_| PbError::Internal {
            what: "raw feature index exceeded u32 building cell maps".into(),
        })?);
        let axes = axes_for_raw(provenance, raw);
        let merged = merged_grids.get(r).ok_or_else(|| PbError::Internal {
            what: format!("merged grid missing raw feature {r}"),
        })?;
        if axes.len() > 1 {
            let channels = crate::cat::channel_axes_for_raw(provenance, raw);
            let joint = crate::cat::JointCatAxis::build(raw, channels, grids, cat_encoders)?;
            out.push(RawCellMap::Joint(joint));
            continue;
        }
        let axis = *axes.first().ok_or_else(|| PbError::Internal {
            what: format!("raw feature {r} has no axis"),
        })?;
        let model = grids.get(axis).ok_or_else(|| PbError::Internal {
            what: format!("matrix missing grid for axis {axis}"),
        })?;
        if merged.missing_bin != 0 || model.missing_bin != 0 {
            return Err(PbError::InvalidInput {
                what: format!("axis {axis} missing_bin must be 0 for table scoring"),
            });
        }
        if merged.n_bins == 0 || model.n_bins == 0 {
            return Err(PbError::InvalidInput {
                what: format!("axis {axis} has zero bins"),
            });
        }
        let mut merged_border_index = Vec::with_capacity(merged.borders.len());
        for &border in &merged.borders {
            let pos = model
                .borders
                .iter()
                .position(|&candidate| candidate.to_bits() == border.to_bits())
                .ok_or_else(|| PbError::InvalidInput {
                    what: format!(
                        "merged border {border} for axis {axis} is absent from model grid"
                    ),
                })?;
            merged_border_index.push(pos);
        }
        let mut map = Vec::with_capacity(usize::from(model.n_bins));
        for bin in 0..model.n_bins {
            let cell = if bin == 0 {
                0usize
            } else {
                let threshold = i64::from(bin) - 2;
                merged_border_index
                    .iter()
                    .filter(|&&idx| (idx as i64) <= threshold)
                    .count()
                    + 1
            };
            if cell >= usize::from(merged.n_bins) {
                return Err(PbError::InvalidInput {
                    what: format!(
                        "axis {axis} model bin {bin} maps to merged cell {cell}, outside {}",
                        merged.n_bins
                    ),
                });
            }
            map.push(u32::try_from(cell).map_err(|_| PbError::Internal {
                what: "merged cell id exceeded u32".into(),
            })?);
        }
        out.push(RawCellMap::Single { axis, map });
    }
    Ok(out)
}

/// Fill `cells` (RAW-indexed, `cells.len() == maps.len()`) with row `row`'s merged cell on
/// every raw feature.
///
/// # Errors
/// [`PbError::Internal`] if a mapped axis escapes the matrix's columns;
/// [`PbError::InvalidInput`] if a bin (single-axis) or channel-bin tuple (joint) has no
/// entry in its map; propagates [`crate::cat::JointCatAxis::cell_for_channel_bins`] otherwise.
/// Every row's merged cell on every raw feature, column-major (`[raw][row]`): the same cells
/// [`fill_row_cells`] derives row by row, built one raw feature at a time (in parallel across
/// raw features) without a per-row allocation.
///
/// # Errors
/// As [`fill_row_cells`].
pub(crate) fn column_cells(
    x: &BinnedMatrix,
    maps: &[RawCellMap],
) -> Result<Vec<Vec<u32>>, PbError> {
    let n_rows = x.n_rows as usize;
    maps.par_iter()
        .enumerate()
        .map(|(r, map)| -> Result<Vec<u32>, PbError> {
            match map {
                RawCellMap::Single { axis, map } => {
                    let col = x
                        .data
                        .get(*axis)
                        .and_then(|c| c.get(..n_rows))
                        .ok_or_else(|| PbError::Internal {
                            what: "validated binned row escaped column".into(),
                        })?;
                    col.iter()
                        .map(|&bin| {
                            map.get(usize::from(bin)).copied().ok_or_else(|| {
                                PbError::InvalidInput {
                                    what: format!("raw feature {r} bin {bin} outside cell map"),
                                }
                            })
                        })
                        .collect()
                }
                RawCellMap::Joint(joint) => {
                    let cols: Vec<&[u8]> = joint
                        .channels
                        .iter()
                        .map(|ch| {
                            x.data
                                .get(ch.model_axis)
                                .and_then(|c| c.get(..n_rows))
                                .ok_or_else(|| PbError::Internal {
                                    what: "joint channel axis escaped matrix columns".into(),
                                })
                        })
                        .collect::<Result<_, _>>()?;
                    let mut bins = vec![0u8; cols.len()];
                    (0..n_rows)
                        .map(|row| {
                            for (b, col) in bins.iter_mut().zip(&cols) {
                                *b = *col.get(row).ok_or_else(|| PbError::Internal {
                                    what: "joint channel row escaped column".into(),
                                })?;
                            }
                            joint.cell_for_channel_bins(&bins)
                        })
                        .collect()
                }
            }
        })
        .collect()
}

pub(crate) fn fill_row_cells(
    x: &BinnedMatrix,
    maps: &[RawCellMap],
    row: usize,
    cells: &mut [u32],
) -> Result<(), PbError> {
    for (r, (map, cell_slot)) in maps.iter().zip(cells).enumerate() {
        *cell_slot = match map {
            RawCellMap::Single { axis, map } => {
                let col = x.data.get(*axis).ok_or_else(|| PbError::Internal {
                    what: "validated binned row escaped column".into(),
                })?;
                let bin = *col.get(row).ok_or_else(|| PbError::Internal {
                    what: "validated binned row escaped column".into(),
                })?;
                *map.get(usize::from(bin))
                    .ok_or_else(|| PbError::InvalidInput {
                        what: format!("raw feature {r} bin {bin} outside cell map"),
                    })?
            }
            RawCellMap::Joint(joint) => {
                let mut bins = Vec::with_capacity(joint.channels.len());
                for ch in &joint.channels {
                    let col = x.data.get(ch.model_axis).ok_or_else(|| PbError::Internal {
                        what: "joint channel axis escaped matrix columns".into(),
                    })?;
                    bins.push(*col.get(row).ok_or_else(|| PbError::Internal {
                        what: "joint channel row escaped column".into(),
                    })?);
                }
                joint.cell_for_channel_bins(&bins)?
            }
        };
    }
    Ok(())
}

/// Score an already-binned matrix (on the model's original grids) through a [`TableBank`]'s
/// LUT-sum in raw-score space, INCLUDING factored order-3 tables (unlike [`TableScoringBank`],
/// which sums only dense tables). Used by the tables-only `TableModel` serve path so its
/// predictions equal the original ensemble. `cat_encoders` (`TableModel.schema.cat_encoders`)
/// is needed only for a P1 multi-channel categorical raw feature's flattened joint-cell space.
///
/// # Errors
/// [`PbError::ShapeMismatch`] for width/length mismatch; [`PbError::InvalidInput`] if a model
/// bin cannot be mapped to the bank's merged grid.
pub(crate) fn score_bank_binned(
    bank: &TableBank,
    cat_encoders: &crate::cat::CatEncoderStore,
    x: &BinnedMatrix,
    out: &mut [f64],
) -> Result<(), PbError> {
    let n_rows = x.n_rows as usize;
    if out.len() != n_rows {
        return Err(PbError::ShapeMismatch {
            what: format!("out len {} != n_rows {n_rows}", out.len()),
        });
    }
    let maps = build_cell_maps(&bank.merged_grids, cat_encoders, x)?;
    score_bank_binned_with_maps(bank, &maps, x, out)
}

/// [`score_bank_binned`] through maps built beforehand by [`build_cell_maps_for`] for this bank
/// under `x`'s grids/provenance (the caller guarantees the pairing; [`CellMaps`] enforces it for
/// the public path). Bit-identical to [`score_bank_binned`].
///
/// # Errors
/// As [`score_bank_binned`].
pub(crate) fn score_bank_binned_with_maps(
    bank: &TableBank,
    maps: &[RawCellMap],
    x: &BinnedMatrix,
    out: &mut [f64],
) -> Result<(), PbError> {
    let n_rows = x.n_rows as usize;
    if out.len() != n_rows {
        return Err(PbError::ShapeMismatch {
            what: format!("out len {} != n_rows {n_rows}", out.len()),
        });
    }
    check_cell_map_matrix(x)?;
    let n_cells = bank.merged_grids.len();
    if maps.len() != n_cells {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "{} cell map(s) for a bank of {n_cells} raw feature(s)",
                maps.len()
            ),
        });
    }
    // Rows are scored in tiles, TABLE-major within a tile: every row's cells are derived first,
    // then each table is read for the whole tile before moving to the next. Each row still sums
    // `f0`, then the dense tables in order, then the factored effects in order — exactly the
    // order of `TableBank::score` — so the result is bit-identical to scoring row by row. What
    // changes is locality: row-major, a wide bank touches every table once per row, and on a
    // 7,466-table (942 MiB) bank each of those reads missed cache (~155 ns); table-major, a table
    // stays hot for the whole tile. Tiles are disjoint `out` chunks on the caller's rayon pool
    // (the prune body installs an n_jobs-sized pool), so the thread budget is honoured.
    let tile = tile_rows(bank, n_rows);
    let stride = n_cells.max(1);
    out.par_chunks_mut(tile)
        .enumerate()
        .try_for_each(|(chunk, dst)| -> Result<(), PbError> {
            let first = chunk * tile;
            let mut cells = vec![0u32; dst.len() * stride];
            for (i, row_cells) in cells.chunks_mut(stride).enumerate() {
                fill_row_cells(
                    x,
                    maps,
                    first + i,
                    row_cells.get_mut(..n_cells).unwrap_or_default(),
                )?;
            }
            dst.fill(bank.f0);
            for table in &bank.tables {
                for (acc, row_cells) in dst.iter_mut().zip(cells.chunks(stride)) {
                    *acc += table.eval(row_cells)?;
                }
            }
            for effect in &bank.factored {
                for (acc, row_cells) in dst.iter_mut().zip(cells.chunks(stride)) {
                    *acc += effect.eval(row_cells)?;
                }
            }
            Ok(())
        })?;
    Ok(())
}

/// Rows per scoring tile. At least [`rows_per_task`], so a small bank never splits a small batch
/// across threads; otherwise up to `TILE_MAX` rows, so a table read is shared by the whole tile,
/// but no more than half a thread's share of the batch, so a small batch on a wide bank (where
/// one row is ~1 ms of work) still spreads over the pool. Measured on the 7,466-table bank: 64-row
/// tiles 2.0x, 256 2.2x, 1024 2.0x, 4096 0.8x over row-at-a-time.
fn tile_rows(bank: &TableBank, n_rows: usize) -> usize {
    const TILE_MAX: usize = 256;
    let per_thread = n_rows / (2 * rayon::current_num_threads().max(1));
    rows_per_task(bank).max(per_thread.min(TILE_MAX)).max(1)
}

/// Rows per rayon task for the per-row bank scorer: enough that one task does about
/// `TASK_WORK` table reads. Without a floor rayon splits a small batch down to single rows and
/// wakes every thread for it — measured: 100 rows through a 12-table bank took 1.4 ms on a
/// 32-thread pool, 5x slower than on one thread. A wide bank (thousands of tables) still gets one
/// row per task, where each row is already ~1 ms of work.
fn rows_per_task(bank: &TableBank) -> usize {
    const TASK_WORK: usize = 16_384;
    let per_row = bank.tables.len() + bank.factored.len() + bank.merged_grids.len() + 1;
    (TASK_WORK / per_row).max(1)
}

/// Model-bin -> merged-cell maps for one [`TableBank`] under one model's axis grids, built once and
/// reused across predict calls instead of being rebuilt per call (see [`build_cell_maps_for`]).
/// Obtained from [`crate::TableModel::cell_maps`]; valid only for the model it was built from, and
/// the scorer refuses it when the bank's merged grids differ from the ones it was built against.
#[derive(Debug, Clone)]
pub struct CellMaps {
    pub(crate) maps: Vec<RawCellMap>,
    pub(crate) merged_grids: Vec<BorderGrid>,
    pub(crate) grids: Vec<BorderGrid>,
    pub(crate) provenance: Vec<crate::data::AxisProvenance>,
    pub(crate) cat_encoders: crate::cat::CatEncoderStore,
}

impl CellMaps {
    /// Build the maps for `bank` under the model axis `grids`/`provenance` and `cat_encoders`.
    ///
    /// # Errors
    /// As [`build_cell_maps_for`].
    pub(crate) fn build(
        bank: &TableBank,
        cat_encoders: &crate::cat::CatEncoderStore,
        grids: &[BorderGrid],
        provenance: &[crate::data::AxisProvenance],
    ) -> Result<Self, PbError> {
        Ok(Self {
            maps: build_cell_maps_for(&bank.merged_grids, cat_encoders, grids, provenance)?,
            merged_grids: bank.merged_grids.clone(),
            grids: grids.to_vec(),
            provenance: provenance.to_vec(),
            cat_encoders: cat_encoders.clone(),
        })
    }
}

/// Score SEVERAL banks that share one merged grid through a single row pass, on a row SUBSET.
///
/// The prune guard's out-of-bag evidence needs, per bag, the raw score of the full bag bank
/// AND of every keep-set/ladder-chunk restriction of it, on that bag's out-of-bag rows. All of
/// those are [`crate::prune::retain_tables`] views of the same bank, so they share
/// `merged_grids` exactly — building the model-bin -> merged-cell maps once and deriving each
/// row's cell vector once serves every one of them. Doing it per bank instead would repeat the
/// (identical) cell derivation `banks.len()` times over.
///
/// `out` is row-major over the SUBSET: `out[i * banks.len() + b]` is `banks[b]`'s score on
/// `rows[i]`. Each row writes only its own chunk, and the within-row/within-bank accumulation
/// order is fixed, so the result is bit-identical regardless of thread count.
///
/// # Errors
/// [`PbError::ShapeMismatch`] if `out` is not `rows.len() * banks.len()` long, if the banks
/// disagree on `merged_grids`, or on a width mismatch; [`PbError::InvalidInput`] if a row index
/// escapes `x` or a model bin cannot be mapped to the merged grid.
pub(crate) fn score_banks_rows(
    banks: &[&TableBank],
    cat_encoders: &crate::cat::CatEncoderStore,
    x: &BinnedMatrix,
    rows: &[u32],
    out: &mut [f64],
) -> Result<(), PbError> {
    if out.len() != rows.len().saturating_mul(banks.len()) {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "score_banks_rows out len {} != rows {} x banks {}",
                out.len(),
                rows.len(),
                banks.len()
            ),
        });
    }
    let Some(first) = banks.first() else {
        return Ok(());
    };
    for (b, bank) in banks.iter().enumerate() {
        if bank.merged_grids != first.merged_grids {
            return Err(PbError::ShapeMismatch {
                what: format!("score_banks_rows bank {b} does not share the first bank's grid"),
            });
        }
    }
    let n_rows = x.n_rows as usize;
    let maps = build_cell_maps(&first.merged_grids, cat_encoders, x)?;
    let n_cells = first.merged_grids.len();
    let width = banks.len();
    // Rows are scored in tiles, TABLE-major within a tile: the tile's cells are derived first,
    // then each bank's tables are read for the whole tile in turn. Every row still sums its
    // bank's `f0`, then the dense tables in order, then the factored effects in order — exactly
    // `TableBank::score` — so each output is bit-identical to scoring the row alone. What changes
    // is locality: row by row, a wide bank's thousands of tables are each touched once per row
    // (a cache miss apiece); table-major, a table stays hot for the whole tile.
    let per_row: usize = banks
        .iter()
        .map(|b| b.tables.len() + b.factored.len() + 1)
        .sum();
    let tile = banks_tile_rows(per_row, rows.len());
    let stride = n_cells.max(1);
    // Every factored effect packed once for the whole call (bit-identical evaluation, see
    // `PackedFactored`); an effect that cannot be packed is scored through its own `eval`.
    let packed: Vec<Vec<Option<crate::explain::PackedFactored>>> = banks
        .par_iter()
        .map(|bank| bank.factored.iter().map(|f| f.packed()).collect())
        .collect();
    // Dense tables likewise, as stride views over their own buffers (bit-identical lookups).
    let packed_tables: Vec<Vec<Option<crate::explain::PackedTable<'_>>>> = banks
        .iter()
        .map(|bank| bank.tables.iter().map(|t| t.packed()).collect())
        .collect();
    out.par_chunks_mut(tile * width)
        .zip(rows.par_chunks(tile))
        .try_for_each(|(dst, tile_rows)| -> Result<(), PbError> {
            let mut cells = vec![0u32; tile_rows.len() * stride];
            for (&row, row_cells) in tile_rows.iter().zip(cells.chunks_mut(stride)) {
                let row = row as usize;
                if row >= n_rows {
                    return Err(PbError::InvalidInput {
                        what: format!("score_banks_rows row {row} escapes {n_rows} rows"),
                    });
                }
                fill_row_cells(
                    x,
                    &maps,
                    row,
                    row_cells.get_mut(..n_cells).unwrap_or_default(),
                )?;
            }
            let mut acc = vec![0.0_f64; tile_rows.len()];
            for (((b, bank), packed_b), tables_b) in
                banks.iter().enumerate().zip(&packed).zip(&packed_tables)
            {
                acc.fill(bank.f0);
                for (table, pt) in bank.tables.iter().zip(tables_b) {
                    match pt {
                        Some(pt) => {
                            for (a, row_cells) in acc.iter_mut().zip(cells.chunks(stride)) {
                                *a += pt.eval(row_cells)?;
                            }
                        }
                        None => {
                            for (a, row_cells) in acc.iter_mut().zip(cells.chunks(stride)) {
                                *a += table.eval(row_cells)?;
                            }
                        }
                    }
                }
                for (effect, pk) in bank.factored.iter().zip(packed_b) {
                    match pk {
                        Some(pk) => {
                            for (a, row_cells) in acc.iter_mut().zip(cells.chunks(stride)) {
                                *a += pk.eval(row_cells)?;
                            }
                        }
                        None => {
                            for (a, row_cells) in acc.iter_mut().zip(cells.chunks(stride)) {
                                *a += effect.eval(row_cells)?;
                            }
                        }
                    }
                }
                for (slot, &a) in dst.iter_mut().skip(b).step_by(width).zip(&acc) {
                    *slot = a;
                }
            }
            Ok(())
        })?;
    Ok(())
}

/// Rows per tile for [`score_banks_rows`]: enough rows that one tile is about `TASK_WORK` table
/// reads (so a narrow bank is not split into single-row tasks), capped at `TILE_MAX` so a table
/// is shared by the whole tile while it stays in cache, and at half a thread's share of the rows
/// so a small batch still spreads over the pool.
fn banks_tile_rows(per_row: usize, n_rows: usize) -> usize {
    const TASK_WORK: usize = 16_384;
    const TILE_MAX: usize = 256;
    let floor = (TASK_WORK / per_row.max(1)).max(1);
    let per_thread = n_rows / (2 * rayon::current_num_threads().max(1));
    floor.max(per_thread.min(TILE_MAX)).max(1)
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
    use crate::data::{
        bin_columns, AxisKind, AxisProvenance, BinConfig, BorderGrid, FeatureId, ServeBinnedMatrix,
    };
    use crate::engine::{Booster, Config, CorrectionTable, FitSpec, ModelSchema, Split};
    use crate::explain::{fixture_model, fixture_serve, OverflowPolicy, RefMeasure, TableBudget};
    use crate::loss::SquaredError;

    #[test]
    fn packed_tree_is_one_cache_line() {
        assert_eq!(std::mem::size_of::<PackedTree>(), 64);
        assert_eq!(std::mem::align_of::<PackedTree>(), 64);
    }

    /// The whole justification for the side arena is that a LIFTED tree's descriptor is
    /// SMALLER than the frozen row, so §11's "stream the bank exactly once" argument gets
    /// stronger rather than 5x weaker. That claim is a layout fact, so assert it — a
    /// mis-sized `_pad` would silently make the arena buy nothing.
    #[test]
    fn arena_tree_is_half_a_cache_line() {
        assert_eq!(std::mem::size_of::<ArenaTree>(), 32);
        assert_eq!(std::mem::align_of::<ArenaTree>(), 32);
        assert!(std::mem::size_of::<ArenaTree>() * 2 == std::mem::size_of::<PackedTree>());
    }

    #[test]
    fn packed_scoring_matches_model_tree_walk() {
        let model = fixture_model();
        let bank = ScoringBank::from_model(&model).unwrap();
        assert!(matches!(bank, ScoringBank::Packed { .. }));
        let x = fixture_serve();
        for row in 0..x.0.n_rows as usize {
            let bins: Vec<u8> = x.0.data.iter().map(|c| c[row]).collect();
            let model_score = model.score_trees_row(&bins, 0.0).unwrap();
            let packed_score = bank.score_row(&bins, model.f0).unwrap();
            assert_eq!(packed_score.to_bits(), model_score.to_bits());
        }
    }

    #[test]
    fn missing_left_bit_is_preserved() {
        let mut model = fixture_model();
        model.trees[0].1.splits[0].missing_left = true;
        let bank = ScoringBank::from_model(&model).unwrap();
        let row = [0_u8, 1_u8];
        assert_eq!(
            bank.score_row(&row, model.f0).unwrap().to_bits(),
            model.score_trees_row(&row, 0.0).unwrap().to_bits()
        );
    }

    #[test]
    fn wide_axis_fallback_matches_model_tree_walk() {
        let mut model = fixture_model();
        let wide_axis = 300usize;
        while model.grids.len() <= wide_axis {
            model.grids.push(BorderGrid {
                borders: vec![1.5],
                n_bins: 3,
                missing_bin: 0,
            });
            let raw = u32::try_from(model.provenance.len()).unwrap();
            model.provenance.push(AxisProvenance {
                raw: FeatureId(raw),
                kind: AxisKind::Numeric,
            });
            model.schema.feature_names.push(format!("f{raw}"));
            model.schema.feature_kinds.push(AxisKind::Numeric);
        }
        model.trees[0].1.splits[0] = Split {
            axis: u32::try_from(wide_axis).unwrap(),
            bin_le: 1,
            missing_left: false,
        };
        model.schema = ModelSchema {
            feature_names: model.schema.feature_names.clone(),
            feature_kinds: model.schema.feature_kinds.clone(),
            cat_encoders: model.schema.cat_encoders.clone(),
            class_labels: None,
            objective: model.schema.objective.clone(),
        };
        let bank = ScoringBank::from_model(&model).unwrap();
        assert!(matches!(bank, ScoringBank::Wide { .. }));
        let mut row = vec![2_u8; model.grids.len()];
        row[wide_axis] = 1;
        row[1] = 1;
        assert_eq!(
            bank.score_row(&row, model.f0).unwrap().to_bits(),
            model.score_trees_row(&row, 0.0).unwrap().to_bits()
        );
    }

    #[test]
    fn flat_table_cells_match_table_bank_score_bit_exactly() {
        let model = fixture_model();
        let serve = fixture_serve();
        let bank = model.explain(&serve, RefMeasure::Uniform).unwrap();
        let flat = TableScoringBank::from_bank(&bank, &crate::cat::CatEncoderStore::new()).unwrap();
        let mut cells = vec![0_u32; bank.merged_grids.len()];
        for c0 in 0..bank.merged_grids[0].n_bins {
            for c1 in 0..bank.merged_grids[1].n_bins {
                cells[0] = u32::from(c0);
                cells[1] = u32::from(c1);
                assert_eq!(
                    flat.score_cells_row(&cells).unwrap().to_bits(),
                    bank.score(&cells).unwrap().to_bits()
                );
            }
        }
    }

    #[test]
    fn flat_table_binned_path_digitizes_once_and_matches_bank() {
        let model = fixture_model();
        let serve = fixture_serve();
        let bank = model
            .explain(&serve, RefMeasure::ProductMarginals { laplace: 1.0 })
            .unwrap();
        let flat = TableScoringBank::from_bank(&bank, &crate::cat::CatEncoderStore::new()).unwrap();
        let mut out = vec![0.0_f64; serve.0.n_rows as usize];
        flat.score_binned(&serve.0, &mut out).unwrap();

        let maps = build_cell_maps(
            &flat.merged_grids,
            &crate::cat::CatEncoderStore::new(),
            &serve.0,
        )
        .unwrap();
        let mut cells = vec![0_u32; flat.merged_grids.len()];
        for (row, score) in out.iter().enumerate() {
            fill_row_cells(&serve.0, &maps, row, &mut cells).unwrap();
            assert_eq!(
                score.to_bits(),
                bank.score(&cells).unwrap().to_bits(),
                "row {row}"
            );
        }
    }

    #[test]
    fn flat_table_rejects_grid_that_lost_a_merged_border() {
        let model = fixture_model();
        let serve = fixture_serve();
        let bank = model.explain(&serve, RefMeasure::Uniform).unwrap();
        let flat = TableScoringBank::from_bank(&bank, &crate::cat::CatEncoderStore::new()).unwrap();
        let mut bad = serve.0.clone();
        bad.grids[0].borders.clear();
        let mut out = vec![0.0_f64; bad.n_rows as usize];
        assert!(matches!(
            flat.score_binned(&bad, &mut out),
            Err(PbError::InvalidInput { .. })
        ));
    }

    /// A hand-built correction over the fixture model's realized pair (axes 0, 1) PLUS a
    /// main effect on axis 1 — TWO tables, the realistic G1 "mains + pairs" shape, not
    /// one. A single-table bank can't distinguish a per-table-cast-then-f32-sum bug from
    /// the correct f64-sum-then-cast-once accumulation (there is no summation order with
    /// only one term) — see `engine/mod.rs`'s
    /// `score_trees_row_batch_and_rows_agree_bit_exactly_with_a_multi_table_correction`,
    /// which pins exactly that class of bug for the canonical `Model` scorers. `bin_to_cell`
    /// is identity throughout (fixture grids have no merged/model bin gap).
    fn fixture_correction() -> CorrectionBank {
        CorrectionBank {
            tables: vec![
                CorrectionTable {
                    axes: vec![0, 1],
                    shape: vec![3, 3],
                    bin_to_cell: vec![vec![0, 1, 2], vec![0, 1, 2]],
                    values: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                },
                CorrectionTable {
                    axes: vec![1],
                    shape: vec![3],
                    bin_to_cell: vec![vec![0, 1, 2]],
                    values: vec![0.0, 0.1, 0.2],
                },
            ],
        }
    }

    #[test]
    fn scoring_bank_carries_correction_and_matches_model_score_trees_row() {
        let mut model = fixture_model();
        model.correction = Some(fixture_correction());
        let bank = ScoringBank::from_model(&model).unwrap();
        let x = fixture_serve();
        for row in 0..x.0.n_rows as usize {
            let bins: Vec<u8> = x.0.data.iter().map(|c| c[row]).collect();
            let model_score = model.score_trees_row(&bins, 0.0).unwrap();
            let packed_score = bank.score_row(&bins, model.f0).unwrap();
            assert_eq!(packed_score.to_bits(), model_score.to_bits(), "row {row}");
        }

        // The correction must actually move the score — otherwise this test could pass
        // with both sides silently ignoring it, which is the exact bug it guards against.
        let bins: Vec<u8> = x.0.data.iter().map(|c| c[0]).collect();
        let corrected = bank.score_row(&bins, model.f0).unwrap();
        let uncorrected_model = fixture_model();
        let uncorrected_bank = ScoringBank::from_model(&uncorrected_model).unwrap();
        let uncorrected = uncorrected_bank
            .score_row(&bins, uncorrected_model.f0)
            .unwrap();
        assert_ne!(corrected, uncorrected);
    }

    #[test]
    fn scoring_bank_correction_delta_is_zero_without_a_correction() {
        let model = fixture_model();
        assert!(model.correction.is_none());
        let bank = ScoringBank::from_model(&model).unwrap();
        let row = [1_u8, 1_u8];
        assert_eq!(bank.correction_delta(&row).unwrap(), 0.0);
    }

    // --- fixtures for a real over-budget fit (H2): `FactoredEffect` has no public
    // constructor and private fields, so the only way to obtain a bank with a non-empty
    // `factored` list is to fit a genuine order-3 interaction and explain it under
    // `OverflowPolicy::Factored`. Mirrors the pattern in `explain.rs`'s own tests.

    fn fit_spec(loss: &SquaredError) -> FitSpec<'_> {
        FitSpec {
            loss,
            weight: None,
            exposure: None,
            monotone: crate::constraints::MonotoneMap::new(),
            interaction: crate::constraints::InteractionPolicy::default(),
            credibility: crate::constraints::CredibilityFloor::default(),
            fixed_holdout: None,
            bag_groups: None,
            seed: 0,
        }
    }

    fn fit(cols: &[Vec<f32>], y: &[f32], cfg: Config) -> (Model, ServeBinnedMatrix) {
        let refs: Vec<&[f32]> = cols.iter().map(Vec::as_slice).collect();
        let x = bin_columns(&refs, None, &BinConfig::default(), 0).unwrap();
        let sqe = SquaredError;
        let model = Booster::with_config(cfg)
            .fit(&x, y, &fit_spec(&sqe))
            .unwrap();
        (model, ServeBinnedMatrix(x))
    }

    fn exact_cfg(n_trees: u32) -> Config {
        Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: crate::engine::GatedStepPolicy::Objective,
            n_trees,
            learning_rate: 1.0,
            lambda: 0.0,
            min_split_gain: 0.0,
            max_delta_step: None,
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
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        }
    }

    #[test]
    fn table_scoring_bank_rejects_factored_tables() {
        let n = 240usize;
        let cols: Vec<Vec<f32>> = (0..3)
            .map(|f| (0..n).map(|i| ((i * (f + 2)) % 6) as f32).collect())
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|i| cols[0][i] * cols[1][i] * cols[2][i] + cols[0][i] - cols[1][i])
            .collect();
        let (model, x) = fit(
            &cols,
            &y,
            Config {
                n_trees: 80,
                learning_rate: 0.3,
                lambda: 1.0,
                ..exact_cfg(80)
            },
        );
        // Under OverflowPolicy::Factored every realized order-3 support is kept factored
        // (explain.rs accumulate()), so a generous cell budget still yields a non-empty
        // `factored` list — no need to tune the budget down to force an overflow.
        let budget = TableBudget {
            max_table_cells: 2_000_000,
            max_bank_cells: 32_000_000,
            on_overflow: OverflowPolicy::Factored,
        };
        let bank = model
            .explain_with_budget(&x, RefMeasure::default(), budget)
            .unwrap();
        assert!(
            !bank.factored.is_empty(),
            "fixture must realize a factored order-3 support"
        );
        assert!(matches!(
            TableScoringBank::from_bank(&bank, &crate::cat::CatEncoderStore::new()),
            Err(PbError::InvalidConfig { .. })
        ));
    }
}
