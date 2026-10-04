//! The explainability engine (spec §2.7 / §08): the trained [`Model`] turned into the
//! [`TableBank`] that **is** the model in a second view, with the equality of the two
//! views enforced as a build gate (the five I2 checks).
//!
//! The pipeline is `accumulate → build_weights → purify → TableBank`, all on a
//! per-raw-feature **merged grid** (the sorted union of every split border realized on
//! that feature across the ensemble, plus an explicit missing cell at index 0 —
//! [`MergedGrids`], R-MERGEDCELL of §08.1). [`Model::explain`] runs the whole pipeline
//! and then the five build-blocking checks:
//!
//! 1. [`check_reconstruction`] — `F_ens == f0 + Σ_u f_u` at every merged-grid cell.
//! 2. `check_mass_conservation` — the purified tables hold zero net `w`-mass (it all
//!    lives in the intercept).
//! 3. `check_purity` — every axis-slice of every table has `w`-weighted mean zero
//!    (maps to [`Invariant::Decomposability`]).
//! 4. `check_variance_sum` — `σ²(F) == Σ_u σ²(f_u)` under product/uniform `w`.
//! 5. [`check_three_way_equal`] — tree-sum = table-sum = Shapley-sum.
//!
//! Plus the I1 [`check_feature_budget`]. The merged-grid missing cell honors each
//! tree's learned `Split.missing_left` exactly, via the SINGLE canonical
//! [`crate::engine`] `low_bit` routing rule — which is what makes tree-sum equal
//! table-sum (and so makes the gates pass rather than merely be asserted).
//!
//! v1 scope: the `ProductMarginals`/`Uniform` reference measures (single-pass purify, exact
//! variance-sum), and exact `Error`/`SparseFallback` table-budget policies. `Joint` is
//! rejected up front rather than silently mishandled.
//!
//! **P1 multi-channel** (design/multichannel-categoricals.md §4.3): a raw feature normally
//! maps to exactly one axis, but a categorical raw feature fit with `cat_channels` maps to
//! MULTIPLE axes (one per channel, e.g. mean-TS + count). [`MergedGrids`] flattens those into
//! ONE joint per-level cell space per raw feature (see [`crate::cat::JointCatAxis`]), so
//! `x_cells[raw]`-style per-raw addressing (this module's whole API) stays a single scalar and
//! needs no change anywhere downstream of [`MergedGrids::axis`]/[`MergedGrids::cells`]/
//! [`MergedGrids::axis_id`] — only how those three are BUILT and how a raw feature's
//! representative model bin(s) are recovered differs for a joint axis.

// Pre-existing merged-grid / purify index-arithmetic flagged by the tightened `indexing_slicing`
// deny lint (newer clippy). Scope-allowed to keep CI green; TODO: convert to `.get()` incrementally.
#![allow(
    clippy::indexing_slicing, // JUSTIFIED: pre-existing module-scoped debt (see comment above); burn-down to `.get()`/per-fn allows is incremental, not expanded here.
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::derivable_impls
)]

mod high_order_smoothing;
pub use high_order_smoothing::HighOrderSmoothingReport;

use crate::cat::{channel_axes_for_raw, ChannelAxis, JointCatAxis};
use crate::data::{AxisKind, BorderGrid, FeatureId, ServeBinnedMatrix};
use crate::engine::{i1_depth_ok, i1_shape_ok, low_bit, Model, LEGACY_MAX_ORDER, MAX_ORDER};
use crate::error::{Invariant, PbError};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

/// Hard ceiling on exhaustive joint-grid enumeration for the reconstruction /
/// variance / three-way checks. Below it the sweep is exhaustive (one interior point
/// per joint cell, spec §08.6); above it the checks sample a deterministic subset
/// (the release-mode behavior of §08.8). MassConservation never enumerates the joint
/// grid (it integrates exactly per tree); VarianceSum self-normalizes and widens its
/// tolerance when sampled; Reconstruction/ThreeWayEqual are per-point max checks, sound
/// under sampling. Most models stay under the cap and run exhaustively.
const JOINT_CAP: usize = 1 << 20;

/// Relative tolerance band for the SAMPLED VarianceSum estimator (a >`JOINT_CAP`-cell
/// joint grid): the self-normalized Monte-Carlo variance carries `O(1/√N)` sampling
/// error, so for such large models VarianceSum is a statistical certification, not a
/// bit-exact one. `5%` is comfortably above the realized sampling error at `N = JOINT_CAP`.
const SAMPLE_VAR_REL: f64 = 0.05;

#[cfg(test)]
thread_local! {
    /// Test-only override for the joint-grid exhaustion cap, so the sampling branch can
    /// be forced on a small model without fitting a multi-million-cell ensemble. `0`
    /// means "use [`JOINT_CAP`]". Thread-local, so parallel tests do not interfere.
    static TEST_JOINT_CAP: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The effective exhaustion cap ([`JOINT_CAP`], or a test override).
fn joint_cap() -> usize {
    #[cfg(test)]
    {
        let t = TEST_JOINT_CAP.with(std::cell::Cell::get);
        if t > 0 {
            return t;
        }
    }
    JOINT_CAP
}

// ===========================================================================
// §08.1 — Local aliases: the merged-grid axis and the effect tensor.
// ===========================================================================

/// A merged-grid axis (spec §08.1). `borders` is the sorted union of realized split
/// borders on one raw feature (the FINITE breakpoints); `cells` is the per-axis tensor
/// extent `== borders.len() + 2` — one EXPLICIT missing cell at index 0 PLUS the
/// `borders.len() + 1` finite half-open interval cells. The missing cell mirrors bin 0
/// of the underlying [`BorderGrid`] (§03) so a tree's learned `missing_left` routing is
/// representable losslessly rather than collapsed into the first finite interval.
///
/// **P1 multi-channel exception** (design/multichannel-categoricals.md §4.3): when
/// `joint_channels` is `Some`, this axis is a flattened joint per-level cell space over
/// MULTIPLE categorical channel axes of one raw feature, not a numeric border sequence —
/// `borders` is empty and carries no meaning, and `cells` is the joint cell count (bounded by
/// distinct levels, not a bin-count product; see [`crate::cat::JointCatAxis`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisId {
    /// The raw feature this axis decomposes (keyed off `AxisProvenance.raw`).
    pub raw: FeatureId,
    /// Sorted finite realized split borders (subset of the model grid's borders). Empty and
    /// meaningless when `joint_channels.is_some()`.
    pub borders: Vec<f32>,
    /// Per-axis tensor extent. `== borders.len() + 2` (missing cell + finite cells) when
    /// `joint_channels` is `None`; the joint cell count otherwise.
    pub cells: u32,
    /// `Some(channel encoder ids, sorted)` iff this is a P1 multi-channel joint axis; `None`
    /// (`#[serde(default)]`) for every axis before this field existed and every ordinary
    /// (numeric or single-channel-categorical) axis today — bit-identical.
    #[serde(default)]
    pub joint_channels: Option<Vec<crate::cat::TsEncodingId>>,
    /// Banding (2026-09-26): `Some(map)` when this table's axis is coarser than the merged grid —
    /// `map[merged_cell] = band`, and `cells` is the band count. Bands are contiguous runs of
    /// merged cells on ordinal axes, or groups of levels on per-level categorical grids; the
    /// missing cell is always its own band. `None` = the axis is the merged grid itself.
    /// A bare `Option` with `#[serde(default)]`, NOT `skip_serializing_if`: bincode is
    /// positional (see [`crate::serialize::SCHEMA_VERSION`] v7).
    #[serde(default)]
    pub band_of: Option<Vec<u32>>,
}

impl AxisId {
    /// Cell count of the merged grid this axis indexes (the band map's domain when banded).
    #[must_use]
    pub fn merged_cells(&self) -> u32 {
        self.band_of
            .as_ref()
            .map_or(self.cells, |m| u32::try_from(m.len()).unwrap_or(u32::MAX))
    }

    /// The finite borders of a banded ORDINAL axis whose bands are contiguous runs of merged
    /// cells (missing cell its own band): the merged border before each band's first cell.
    /// `None` when the axis is not banded, is a joint categorical axis, or its bands are not
    /// contiguous in merged-cell order.
    #[must_use]
    pub fn band_borders(&self) -> Option<Vec<f32>> {
        let map = self.band_of.as_ref()?;
        if self.joint_channels.is_some() || map.len() != self.borders.len() + 2 {
            return None;
        }
        let mut out = Vec::new();
        for c in 2..map.len() {
            if map[c] < map[c - 1] {
                return None;
            }
            if map[c] != map[c - 1] {
                out.push(self.borders[c - 2]);
            }
        }
        Some(out)
    }

    /// Tensor coordinate of a merged-grid cell on this axis (the band when banded).
    #[inline]
    #[must_use]
    pub fn coord(&self, merged_cell: u32) -> Option<usize> {
        match &self.band_of {
            None => Some(merged_cell as usize),
            Some(map) => map.get(merged_cell as usize).map(|&b| b as usize),
        }
    }
}

/// A dense row-major n-dimensional tensor of `f64` values (§08-local). Used for an
/// [`EffectTable`]'s purified `values` and its per-cell `support`. `f64` even though the
/// core trains in `f32`: purification accumulates many signed mass-moves and we want the
/// reconstruction residual at `f64` epsilon, not `f32` epsilon (§08.1).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Tensor {
    // Per-axis extents as fixed-width ints: this type is serialized inside an
    // EffectTable, and a serialized `usize` would differ between the host and the
    // wasm32 smoke build, breaking cross-platform byte-equality (spec §02.8).
    // Cell-count dimensions are tiny, so `u32` is ample.
    shape: Vec<u32>,
    data: TensorData,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum TensorData {
    Dense(Vec<f64>),
    Sparse(Vec<SparseEntry>),
}

impl Default for TensorData {
    fn default() -> Self {
        TensorData::Dense(Vec::new())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct SparseEntry {
    index: u64,
    value: f64,
}

fn checked_shape(shape: &[usize]) -> Result<(Vec<u32>, usize), PbError> {
    let mut shape_u32 = Vec::with_capacity(shape.len());
    let mut cells = 1usize;
    for (axis, &dim) in shape.iter().enumerate() {
        if dim == 0 {
            return Err(PbError::InvalidInput {
                what: format!("tensor axis {axis} has zero extent"),
            });
        }
        shape_u32.push(u32::try_from(dim).map_err(|_| PbError::InvalidInput {
            what: format!("tensor axis {axis} extent {dim} exceeds u32"),
        })?);
        cells = cells.checked_mul(dim).ok_or_else(|| PbError::Internal {
            what: "tensor shape overflows usize".into(),
        })?;
    }
    Ok((shape_u32, cells))
}

fn checked_shape_cells_u64(shape: &[u32]) -> Result<u64, PbError> {
    let mut cells = 1u64;
    for &dim in shape {
        if dim == 0 {
            return Err(PbError::InvalidInput {
                what: "tensor has a zero extent".into(),
            });
        }
        cells = cells
            .checked_mul(u64::from(dim))
            .ok_or_else(|| PbError::Internal {
                what: "tensor shape overflows u64".into(),
            })?;
    }
    Ok(cells)
}

fn filled_data(cells: usize, value: f64) -> Result<Vec<f64>, PbError> {
    let mut data = Vec::new();
    data.try_reserve_exact(cells)
        .map_err(|_| PbError::Internal {
            what: "tensor allocation failed".into(),
        })?;
    data.resize(cells, value);
    Ok(data)
}

impl Tensor {
    /// Try to build a zero tensor of the given per-axis extents. 0-D — an empty
    /// shape — is one scalar cell.
    ///
    /// # Errors
    /// [`PbError::InvalidInput`] if an extent is zero or exceeds `u32`;
    /// [`PbError::Internal`] if shape arithmetic overflows or allocation fails.
    pub fn try_zeros(shape: Vec<usize>) -> Result<Self, PbError> {
        let (shape, cells) = checked_shape(&shape)?;
        Ok(Self {
            data: TensorData::Dense(filled_data(cells, 0.0)?),
            shape,
        })
    }

    /// Try to build a sparse zero tensor of the given per-axis extents.
    ///
    /// # Errors
    /// [`PbError::InvalidInput`] if an extent is zero or exceeds `u32`;
    /// [`PbError::Internal`] if shape arithmetic overflows.
    pub fn try_sparse_zeros(shape: Vec<usize>) -> Result<Self, PbError> {
        let (shape, _) = checked_shape(&shape)?;
        Ok(Self {
            data: TensorData::Sparse(Vec::new()),
            shape,
        })
    }

    /// Build from an explicit row-major buffer.
    ///
    /// # Errors
    /// [`PbError::InvalidInput`] if an extent is zero or exceeds `u32`;
    /// [`PbError::Internal`] if shape arithmetic overflows;
    /// [`PbError::ShapeMismatch`] if `data.len()` does not equal the product of `shape`.
    pub fn from_vec(shape: Vec<usize>, data: Vec<f64>) -> Result<Self, PbError> {
        let (shape, n) = checked_shape(&shape)?;
        if data.len() != n {
            return Err(PbError::ShapeMismatch {
                what: format!("tensor data len {} != product(shape) {n}", data.len()),
            });
        }
        Ok(Self {
            shape,
            data: TensorData::Dense(data),
        })
    }

    /// The tensor's per-axis extents (as `usize` for indexing).
    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        self.shape.iter().map(|&d| d as usize).collect()
    }

    /// The tensor's per-axis extents as fixed-width serialized dimensions.
    #[must_use]
    pub fn shape_u32(&self) -> &[u32] {
        &self.shape
    }

    /// Total number of cells.
    #[must_use]
    pub fn len(&self) -> usize {
        checked_shape_cells_u64(&self.shape)
            .ok()
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(0)
    }

    /// `true` if the tensor has no cells.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The row-major values. Dense tensors borrow their backing slice; sparse tensors
    /// materialize an owned dense view for display/export compatibility. Infallible: a
    /// sparse-to-dense conversion failure (allocation under memory pressure) silently
    /// returns an EMPTY view rather than propagating the error — safe only for
    /// display-only call sites that can tolerate a degraded-but-not-wrong view. Anything
    /// that ships the result (an export, a rebase) MUST use [`Self::try_values`] instead.
    #[must_use]
    pub fn values(&self) -> Cow<'_, [f64]> {
        match &self.data {
            TensorData::Dense(data) => Cow::Borrowed(data),
            TensorData::Sparse(_) => Cow::Owned(self.dense_values().unwrap_or_default()),
        }
    }

    /// The row-major values, propagating a sparse-to-dense conversion failure instead of
    /// silently degrading to an empty view. Use this at any call site whose result is
    /// served or exported (the rating-export and rebase paths) — see [`Self::values`].
    ///
    /// # Errors
    /// [`PbError::Internal`] if the sparse-to-dense materialization fails (e.g. `try_reserve`
    /// under memory pressure).
    pub fn try_values(&self) -> Result<Cow<'_, [f64]>, PbError> {
        match &self.data {
            TensorData::Dense(data) => Ok(Cow::Borrowed(data)),
            TensorData::Sparse(_) => Ok(Cow::Owned(self.dense_values()?)),
        }
    }

    /// `true` if this tensor uses sparse backing storage.
    #[must_use]
    pub fn is_sparse(&self) -> bool {
        matches!(self.data, TensorData::Sparse(_))
    }

    /// The dense row-major backing, or `None` for a sparse tensor (or a dense buffer whose
    /// length disagrees with the shape). The explain hot loops stride over this directly;
    /// every such fast path visits and accumulates cells in the same order as the per-cell
    /// `at`/`add` walk it replaces, so the bits are unchanged.
    fn dense_slice(&self) -> Option<&[f64]> {
        match &self.data {
            TensorData::Dense(data) if data.len() == self.len() => Some(data),
            _ => None,
        }
    }

    /// Mutable twin of [`Self::dense_slice`].
    fn dense_slice_mut(&mut self) -> Option<&mut [f64]> {
        let cells = self.len();
        match &mut self.data {
            TensorData::Dense(data) if data.len() == cells => Some(data),
            _ => None,
        }
    }

    /// Number of stored (non-removed) entries in a sparse-backed tensor; `None` for dense
    /// backing. `set` only ever stores a genuinely nonzero cell (and removes an entry whose
    /// value becomes exactly 0.0), so this is the tensor's true occupancy, not an upper bound.
    fn sparse_nnz(&self) -> Option<usize> {
        match &self.data {
            TensorData::Dense(_) => None,
            TensorData::Sparse(entries) => Some(entries.len()),
        }
    }

    /// Add a constant to every cell. Used by rating-view re-basing, which moves an
    /// equal and opposite constant into the exported intercept and therefore preserves
    /// every reconstructed score exactly — so a sparse-to-dense conversion failure here
    /// MUST propagate rather than silently leave the table unshifted while the caller's
    /// intercept moves anyway (see `to_rating_export`).
    ///
    /// # Errors
    /// [`PbError::Internal`] if the sparse-to-dense materialization fails.
    pub fn add_scalar(&mut self, delta: f64) -> Result<(), PbError> {
        match &mut self.data {
            TensorData::Dense(data) => {
                for v in data {
                    *v += delta;
                }
            }
            TensorData::Sparse(_) => {
                let mut dense = self.dense_values()?;
                for v in &mut dense {
                    *v += delta;
                }
                self.data = TensorData::Dense(dense);
            }
        }
        Ok(())
    }

    fn offset(&self, coord: &[usize]) -> Option<u64> {
        if coord.len() != self.shape.len() {
            return None;
        }
        let mut off = 0u64;
        for (c, dim) in coord.iter().zip(self.shape.iter()) {
            let dim_u64 = u64::from(*dim);
            let c_u64 = u64::try_from(*c).ok()?;
            if c_u64 >= dim_u64 {
                return None;
            }
            off = off.checked_mul(dim_u64)?.checked_add(c_u64)?;
        }
        Some(off)
    }

    fn dense_values(&self) -> Result<Vec<f64>, PbError> {
        let cells_u64 = checked_shape_cells_u64(&self.shape)?;
        let cells = usize::try_from(cells_u64).map_err(|_| PbError::Internal {
            what: "tensor dense view exceeds usize".into(),
        })?;
        let mut dense = filled_data(cells, 0.0)?;
        match &self.data {
            TensorData::Dense(data) => {
                if data.len() != cells {
                    return Err(PbError::ShapeMismatch {
                        what: "tensor dense backing length does not match shape".into(),
                    });
                }
                dense = data.clone();
            }
            TensorData::Sparse(entries) => {
                for entry in entries {
                    let idx = usize::try_from(entry.index).map_err(|_| PbError::Internal {
                        what: "sparse tensor index exceeds usize".into(),
                    })?;
                    let slot = dense.get_mut(idx).ok_or_else(|| PbError::ShapeMismatch {
                        what: "sparse tensor index outside shape".into(),
                    })?;
                    *slot = entry.value;
                }
            }
        }
        Ok(dense)
    }

    /// Read the value at `coord`, or `None` if out of range / wrong rank.
    #[must_use]
    pub fn at(&self, coord: &[usize]) -> Option<f64> {
        let off = self.offset(coord)?;
        match &self.data {
            TensorData::Dense(data) => usize::try_from(off).ok().and_then(|o| data.get(o).copied()),
            TensorData::Sparse(entries) => entries
                .binary_search_by_key(&off, |entry| entry.index)
                .ok()
                .and_then(|pos| entries.get(pos).map(|entry| entry.value))
                .or(Some(0.0)),
        }
    }

    /// Write `value` at `coord`.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if `coord` is out of range or the wrong rank.
    pub fn set(&mut self, coord: &[usize], value: f64) -> Result<(), PbError> {
        let off = self.offset(coord).ok_or_else(|| PbError::ShapeMismatch {
            what: "tensor set coord out of range".into(),
        })?;
        match &mut self.data {
            TensorData::Dense(data) => {
                let off = usize::try_from(off).map_err(|_| PbError::Internal {
                    what: "tensor offset exceeds usize".into(),
                })?;
                let slot = data.get_mut(off).ok_or_else(|| PbError::Internal {
                    what: "tensor offset escaped buffer".into(),
                })?;
                *slot = value;
            }
            TensorData::Sparse(entries) => match entries.binary_search_by_key(&off, |e| e.index) {
                Ok(pos) => {
                    if value == 0.0 {
                        entries.remove(pos);
                    } else if let Some(entry) = entries.get_mut(pos) {
                        entry.value = value;
                    }
                }
                Err(pos) => {
                    if value != 0.0 {
                        entries.try_reserve(1).map_err(|_| PbError::Internal {
                            what: "sparse tensor allocation failed".into(),
                        })?;
                        entries.insert(pos, SparseEntry { index: off, value });
                    }
                }
            },
        }
        Ok(())
    }

    /// Add `delta` to the value at `coord`.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if `coord` is out of range or the wrong rank.
    pub fn add(&mut self, coord: &[usize], delta: f64) -> Result<(), PbError> {
        let value = self.at(coord).ok_or_else(|| PbError::ShapeMismatch {
            what: "tensor add coord out of range".into(),
        })?;
        self.set(coord, value + delta)
    }

    /// Deposit a rank-2 marginal broadcast: for every cell `(c0, c1)` add
    /// `m[low_a[c0] as usize][low_b[c1] as usize]`.
    ///
    /// This is BYTE-identical to looping [`Self::add`] over the cells in row-major order:
    /// each cell is written exactly once, so there is no cross-cell accumulation whose order
    /// could change the bits. It exists to skip the per-cell offset recomputation and bounds
    /// plumbing that dominate the factored-shed deposit (§08.10) — the dense inner loop is a
    /// plain element-wise `+=` over a contiguous row, which the compiler autovectorizes
    /// without altering any per-cell f64 result. See `add_marginal_pair_is_bit_identical_to_scalar`.
    ///
    /// # Errors
    /// [`PbError::Internal`] if the tensor is not rank 2 or a mask is shorter than its axis;
    /// [`PbError::ShapeMismatch`] if the dense backing length disagrees with the shape.
    fn add_marginal_pair(
        &mut self,
        m: &[[f64; 2]; 2],
        low_a: &[bool],
        low_b: &[bool],
    ) -> Result<(), PbError> {
        if self.shape.len() != 2 {
            return Err(PbError::Internal {
                what: "add_marginal_pair requires a rank-2 tensor".into(),
            });
        }
        let n0 = self.shape[0] as usize;
        let n1 = self.shape[1] as usize;
        if low_a.len() < n0 || low_b.len() < n1 {
            return Err(PbError::Internal {
                what: "add_marginal_pair mask shorter than tensor axis".into(),
            });
        }
        // Sparse tables keep the byte-identical scalar path (no dense backing to stride over).
        if self.is_sparse() {
            for (c0, &a) in low_a.iter().take(n0).enumerate() {
                let s0 = usize::from(a);
                for (c1, &b) in low_b.iter().take(n1).enumerate() {
                    self.add(&[c0, c1], m[s0][usize::from(b)])?;
                }
            }
            return Ok(());
        }
        let data = match &mut self.data {
            TensorData::Dense(data) => data,
            TensorData::Sparse(_) => {
                return Err(PbError::Internal {
                    what: "add_marginal_pair sparse fast path unreachable".into(),
                });
            }
        };
        if data.len() != n0 * n1 {
            return Err(PbError::ShapeMismatch {
                what: "add_marginal_pair dense backing length does not match shape".into(),
            });
        }
        // Precompute, once per call, the two candidate rows selected by `low_b` — one for each
        // value of `low_a[c0]`. These hold exact copies of `m`'s entries (no arithmetic), so the
        // values deposited are byte-for-byte the scalar path's.
        let mut row0 = vec![0.0_f64; n1];
        let mut row1 = vec![0.0_f64; n1];
        for ((r0, r1), &b) in row0.iter_mut().zip(row1.iter_mut()).zip(low_b.iter()) {
            let j = usize::from(b);
            *r0 = m[0][j];
            *r1 = m[1][j];
        }
        for (&a, row) in low_a.iter().take(n0).zip(data.chunks_mut(n1)) {
            let src = if a { &row1 } else { &row0 };
            for (d, &s) in row.iter_mut().zip(src.iter()) {
                *d += s;
            }
        }
        Ok(())
    }
}

/// A set of `0..=MAX_ORDER` distinct, sorted raw feature ids identifying one effect
/// (spec §2.7). The `SmallVec` inline capacity is `MAX_ORDER`, so no realized support
/// ever spills; it SERIALIZES as a length-prefixed sequence, so the inline width is a
/// pure stack-layout choice with no wire consequence.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct FeatureSet(
    #[serde(deserialize_with = "deserialize_feature_ids")] pub SmallVec<[FeatureId; MAX_ORDER]>,
);

// SmallVec reserves the entire untrusted sequence size hint. Serde's Vec visitor
// caps its initial reservation and grows only as elements are actually decoded.
// Both representations use the same length-prefixed sequence on the wire.
fn deserialize_feature_ids<'de, D>(
    deserializer: D,
) -> Result<SmallVec<[FeatureId; MAX_ORDER]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Vec::<FeatureId>::deserialize(deserializer).map(SmallVec::from_vec)
}

impl FeatureSet {
    /// Build a feature set from raw ids (caller ensures distinct/sorted).
    #[must_use]
    pub fn new(ids: &[u32]) -> Self {
        FeatureSet(ids.iter().map(|&i| FeatureId(i)).collect())
    }

    /// The interaction order `|u|` (1 = main effect, 2 = pairwise, 3 = triple, 4 = quad).
    #[must_use]
    pub fn order(&self) -> usize {
        self.0.len()
    }

    /// `true` if `f` is a member of this feature set.
    #[must_use]
    pub fn contains(&self, f: FeatureId) -> bool {
        self.0.contains(&f)
    }
}

/// Per-cell standard-error bands for bagged/averaged rating-table displays (§09.5).
/// This is display-only metadata: invariant checks and inference never read it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeBand {
    /// Per-cell standard error, parallel to [`EffectTable::values`].
    pub per_cell: Tensor,
}

/// One purified effect tensor for feature set `u`, on the merged grid (spec §2.7).
/// `support` and `se_band` are display metadata — excluded from the five invariant
/// checks and from inference (scoring reads `values` only).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectTable {
    /// The raw feature set this effect is over.
    pub u: FeatureSet,
    /// The merged-grid axes (parallel to `values`' dimensions).
    pub axes: Vec<AxisId>,
    /// The purified effect values (one cell per merged-grid cell).
    pub values: Tensor,
    /// Per-cell effective `w`-mass — exposure-weighted training-row count, spec §08.7
    /// (display-only; same extents as `values`). A flat row count (`Σ 1`) unless the
    /// bank was built from a per-row `mass` (e.g. [`Model::explain_weighted`]), in which
    /// case it is `Σ w·e` (sample weight times exposure) per cell instead.
    pub support: Tensor,
    /// Optional per-cell standard-error band, display-only.
    #[serde(default)]
    pub se_band: Option<SeBand>,
    /// `w`-weighted variance of this effect, `σ²(f_u)`.
    pub variance: f64,
}

impl EffectTable {
    /// Evaluate this effect at a row given its per-raw-feature merged-cell ids
    /// (`x_cells[raw] = cell`). The tensor coordinate is read off `axes[k].raw`.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if `x_cells` lacks one of this table's axes;
    /// [`PbError::Internal`] if the projected coordinate escapes the tensor.
    pub fn eval(&self, x_cells: &[u32]) -> Result<f64, PbError> {
        // Order <= MAX_ORDER, so a SmallVec never spills to the heap — this eval is the
        // per-row-per-table scoring hot path; the previous `Vec::with_capacity` allocated
        // on every call.
        let mut coord: SmallVec<[usize; MAX_ORDER]> = SmallVec::with_capacity(self.axes.len());
        for a in &self.axes {
            let cell = *x_cells
                .get(a.raw.0 as usize)
                .ok_or_else(|| PbError::ShapeMismatch {
                    what: format!("x_cells missing raw feature {} for table eval", a.raw.0),
                })?;
            coord.push(a.coord(cell).ok_or_else(|| PbError::Internal {
                what: "effect-table merged cell outside its band map".into(),
            })?);
        }
        self.values.at(&coord).ok_or_else(|| PbError::Internal {
            what: "effect-table coordinate out of range".into(),
        })
    }
}

/// The reference measure for purification (spec §2.7 / §08.4). Default = Laplace-
/// smoothed empirical product-of-marginals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RefMeasure {
    /// Product of per-axis Laplace-smoothed empirical marginals (DEFAULT; `laplace > 0`).
    /// Per cell `ŵ ∝ ŵ_unif + laplace · ŵ_emp` (§08.4), strictly positive (so empty
    /// merged cells never break zero-mean or single-pass convergence).
    ProductMarginals {
        /// Laplace smoothing weight on the empirical marginal.
        laplace: f32,
    },
    /// Uniform over realized cells (`ŵ ∝ 1` per cell).
    Uniform,
    /// Hooker hierarchical-orthogonality joint measure — a v1.5 fork (couples axes,
    /// breaks the variance-sum identity). Rejected by [`Model::explain`] in v1.
    Joint,
    /// Product of per-axis EXPOSURE marginals with a strictly-positive floor (2026-09-06).
    /// Per cell `ŵ ∝ share + floor / cells`, where `share` is the cell's share of the
    /// effective row mass (`Σ sample_weight · exposure`, spec §08.7) and the floor carries
    /// `floor` of that mass in total, spread evenly over the axis. Unlike
    /// [`RefMeasure::ProductMarginals`] — whose `laplace = 1` default gives the uniform
    /// component HALF the weight on every axis, so empty and thin cells dominate what
    /// purification moves down into the main effects — the floor exists only to keep the
    /// weights positive (zero-mean and single-pass purify never break on an empty cell).
    /// Still axis-factorized, so single-pass purify, the variance-sum identity and the
    /// equal-split SHAP identity all hold. Under this measure the prune, the per-bag banks
    /// and the export all read the fit-time effective mass rather than a flat row count.
    /// Appended AFTER `Joint` so every earlier variant keeps its bincode discriminant.
    ExposureMarginals {
        /// Total floor mass as a fraction of the empirical mass; finite and `> 0`.
        floor: f32,
    },
}

impl RefMeasure {
    /// Whether this measure reads the per-row effective mass (spec §08.7) when it is
    /// available. `ProductMarginals` keeps the historical flat row count on every path
    /// that never passed a mass (the prune, the per-bag banks), so the boards fitted
    /// before [`RefMeasure::ExposureMarginals`] existed stay byte-reproducible.
    #[must_use]
    pub fn uses_row_mass(&self) -> bool {
        matches!(self, RefMeasure::ExposureMarginals { .. })
    }
}

/// The per-axis weighting rule a validated [`RefMeasure`] resolves to (spec §08.4). One
/// place for the arithmetic so the three weight builders (serve matrix, raw-effect
/// support, bank support) cannot drift from each other.
#[derive(Debug, Clone, Copy)]
enum AxisRule {
    /// `ŵ ∝ 1` per cell.
    Uniform,
    /// `ŵ ∝ 1/cells + laplace · share` — the historical Laplace blend.
    Laplace(f64),
    /// `ŵ ∝ share + floor / cells` — the exposure marginal with a positivity floor.
    Floor(f64),
}

impl AxisRule {
    fn needs_counts(self) -> bool {
        !matches!(self, AxisRule::Uniform)
    }

    /// Un-normalized per-cell weights. `counts` is the per-cell mass (unused under
    /// `Uniform`) and `inv_n` the reciprocal of its total (`0.0` when that total is zero).
    /// The `Laplace` arm is the pre-existing expression, term for term, so the legacy
    /// measure stays bit-identical.
    fn raw_weights(self, counts: &[f64], inv_n: f64, cells: usize) -> Vec<f64> {
        let unif = 1.0_f64 / cells as f64;
        match self {
            AxisRule::Uniform => vec![unif; cells],
            AxisRule::Laplace(lap) => counts.iter().map(|c| unif + lap * (c * inv_n)).collect(),
            AxisRule::Floor(floor) => counts.iter().map(|c| c * inv_n + floor * unif).collect(),
        }
    }
}

/// Validate a [`RefMeasure`] and resolve its [`AxisRule`].
///
/// # Errors
/// [`PbError::InvalidConfig`] for `Joint` (not an axis-factorized measure) or a non-finite /
/// non-positive `laplace` or `floor`.
fn axis_rule(w: &RefMeasure) -> Result<AxisRule, PbError> {
    match w {
        RefMeasure::Uniform => Ok(AxisRule::Uniform),
        RefMeasure::ProductMarginals { laplace } => {
            if !laplace.is_finite() || *laplace <= 0.0 {
                return Err(PbError::InvalidConfig {
                    what: "ProductMarginals laplace must be finite and > 0".into(),
                });
            }
            Ok(AxisRule::Laplace(f64::from(*laplace)))
        }
        RefMeasure::ExposureMarginals { floor } => {
            if !floor.is_finite() || *floor <= 0.0 {
                return Err(PbError::InvalidConfig {
                    what: "ExposureMarginals floor must be finite and > 0".into(),
                });
            }
            Ok(AxisRule::Floor(f64::from(*floor)))
        }
        RefMeasure::Joint => Err(PbError::InvalidConfig {
            what: "Joint reference measure is a v1.5 fork; v1 supports ProductMarginals/\
                   ExposureMarginals/Uniform"
                .into(),
        }),
    }
}

impl Default for RefMeasure {
    fn default() -> Self {
        RefMeasure::ProductMarginals { laplace: 1.0 }
    }
}

/// The complete decomposition (spec §2.7): intercept + all purified tables on the
/// shared merged grid. `tables` is the lossless inference support; display pruning is a
/// view. `merged_grids` is indexed by raw feature id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableBank {
    /// The intercept term (the `w`-weighted mean of the ensemble).
    pub f0: f64,
    /// Every realized effect `u` of size `1..=3`, plus the lower-order tables the
    /// purification cascade generates (the down-set closure of realized supports).
    pub tables: Vec<EffectTable>,
    /// Per-raw-feature merged grid (sorted union of realized borders + missing cell).
    pub merged_grids: Vec<BorderGrid>,
    /// The reference measure stamped on the bank and every export.
    pub w: RefMeasure,
    /// Over-budget order-3 effects kept in factored (per-tree-box) form — never densified
    /// (§08.10). Empty for budget-fitting banks; each carries the same `f_u` a dense
    /// [`EffectTable`] would, for score/shap/sobol and the five I2 gates.
    #[serde(default)]
    pub factored: Vec<FactoredEffect>,
    /// Joint model variance measured on aligned rows, including covariance. Runtime only:
    /// after loading a joint bank, supply rows again before requesting variance shares.
    #[serde(skip)]
    pub joint_variance: Option<f64>,
}

/// Tolerances for the I2 checks (spec §13.1). `recon_tol` is the canonical
/// `4 · n_trees · f32::EPSILON`; the others track it (variance is squared-scale, so it
/// gets headroom).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExactTol {
    /// Per-cell reconstruction tolerance.
    pub recon_tol: f64,
    /// Mass-conservation tolerance.
    pub mass_tol: f64,
    /// Per-slice purity tolerance.
    pub purity_tol: f64,
    /// Variance-sum tolerance (squared-scale).
    pub var_tol: f64,
}

impl ExactTol {
    /// The tolerances for `model`: `4 · n_trees · (order / 3) · f32::EPSILON` (spec §13.1,
    /// widened by the realized interaction ORDER at the high-order lift).
    ///
    /// # Why order enters a tolerance that used to count only trees
    ///
    /// The §13.1 form `4 · n_trees · ε` prices the one error source a depth-3/order-3 fit
    /// had: summing `n_trees` `f32` contributions. The purification cascade's own error was
    /// negligible against it because the cascade was only three centring passes deep.
    ///
    /// At order `k` it is `k` passes deep — each `center_along` subtracts a weighted slice
    /// mean from every cell, and the shed cascades that residual down through
    /// `k → k−1 → … → 1` — while the factored variance path's `inner_walk` accumulates `4^k`
    /// terms per box PAIR instead of `4^3`. Both error sources grow with `k` and the old
    /// tolerance did not, so a genuine order-7/8 fit could fail `VarianceSum` /
    /// `ThreeWayEqual` on ACCUMULATED ROUNDING and be flipped to `Approximate` — refusing to
    /// export a rating table that is in fact exact. A false alarm on the most important gate
    /// in the crate is not harmless: it teaches a reader to distrust the gate.
    ///
    /// Linear in the realized order is the conservative reading (the pass count is linear;
    /// only the variance sweep is exponential, and `var_tol` already carries its own 16x for
    /// the squared scale). Crucially the factor is normalized by [`LEGACY_MAX_ORDER`], so an
    /// order-`<= 3` model gets `4 · n_trees · ε` EXACTLY as before — every pre-lift gate
    /// decision is unmoved — and only orders 4+ widen.
    #[must_use]
    pub fn for_model(model: &Model) -> Self {
        let n_trees = model.trees.len().max(1) as f64;
        let realized_order = model
            .trees
            .iter()
            .map(|(_, t)| crate::engine::distinct_raw_count(&t.splits, &model.provenance))
            .max()
            .unwrap_or(LEGACY_MAX_ORDER)
            .max(LEGACY_MAX_ORDER);
        let order_slack = (realized_order as f64) / (LEGACY_MAX_ORDER as f64);
        let base = 4.0 * n_trees * order_slack * f64::from(f32::EPSILON);
        ExactTol {
            recon_tol: base,
            mass_tol: base,
            purity_tol: base,
            var_tol: 16.0 * base,
        }
    }
}

/// Per-table and whole-bank cell budgets (spec §08.10, the memory firewall). Counted on
/// the realized merged (union) grid (R-TABLEBUDGET), checked at lazy allocation so an
/// over-budget table either fails before dense allocation or uses the explicit sparse
/// policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TableBudget {
    /// Per-[`EffectTable`] `Π cells_i` ceiling.
    pub max_table_cells: u64,
    /// `Σ` over all tables ceiling.
    pub max_bank_cells: u64,
    /// What to do when a table would exceed `max_table_cells`.
    pub on_overflow: OverflowPolicy,
}

impl Default for TableBudget {
    fn default() -> Self {
        Self {
            max_table_cells: 2_000_000,
            max_bank_cells: 32_000_000,
            on_overflow: OverflowPolicy::Factored,
        }
    }
}

/// The resolution when a table would exceed [`TableBudget::max_table_cells`] (§08.10).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OverflowPolicy {
    /// Hard error — refuse to build a bank that would exceed the budget
    /// ([`PbError::TableBudget`]). No silent truncation.
    Error,
    /// EXACT sparse-tensor storage for over-budget tables. This preserves the logical
    /// tensor shape and all I2 checks while avoiding dense allocation for cold cells.
    SparseFallback {
        /// Maximum realized nonzero occupancy allowed for sparse storage.
        density_threshold: f64,
    },
    /// Keep every effect of order `>= LEGACY_MAX_ORDER` in FACTORED rank-1-box form
    /// (§08.10) — exact, never densified. The default.
    ///
    /// Order 1 and 2 are bounded by `255` and `255^2` cells and stay dense, because that is
    /// what a human reads. Order 3 and above cannot: a support's dense cube is the product
    /// of its raws' GLOBAL merged extents, which include every border placed by main-effect
    /// trees that legitimately want full resolution, so a high-order support pays for
    /// resolution it never asked for. But a tree realizes only `Π k_d` REGIONS on its
    /// support, so the effect is low-RANK, not sparse — small in this representation and in
    /// no other. Each effect's order-`(k-1)` marginal mass is shed one order down (into the
    /// next factored effect above order 3, into the dense pair table at order 3), leaving
    /// the pure `k`-way residual factored.
    Factored,
}

/// The purification convergence mode (spec §08.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PurifyMode {
    /// Iterate to a fixpoint (for joint `w`, where slice masses couple across axes).
    ToFixpoint {
        /// Mass-move tolerance.
        tol: f64,
        /// Iteration cap before returning a non-convergence error.
        max_iter: u32,
    },
    /// One pass per axis — exact for axis-factorized `w` (product/uniform), §08.3.
    SinglePass,
}

// ===========================================================================
// §08.1 — The merged grids: per-raw-feature sorted union of realized borders.
// ===========================================================================

/// One raw feature's merged axis: EITHER the realized split borders of a single numeric/
/// categorical axis (a subset of the model grid's borders, recorded by both value and
/// model-border index) plus the cell maps, OR (P1 multi-channel) a flattened joint per-level
/// cell space over multiple categorical channel axes of the same raw feature.
#[derive(Debug, Clone)]
struct MergedAxis {
    axis: usize,
    borders: Vec<f32>,
    model_border_index: Vec<usize>,
    model_n_bins: u16,
    /// `Some` iff this raw feature has more than one categorical channel axis (P1
    /// multi-channel, design/multichannel-categoricals.md §4.3); `axis`/`borders`/
    /// `model_border_index`/`model_n_bins` above are then unused placeholders — every method
    /// branches on this field first. `None` (every axis before this field existed) is
    /// bit-identical to the original single-axis-only type.
    joint: Option<JointCatAxis>,
}

impl MergedAxis {
    fn cells(&self) -> usize {
        match &self.joint {
            Some(j) => j.n_cells as usize,
            None => self.borders.len() + 2,
        }
    }

    /// Merged cell → a representative model bin that routes identically under `low_bit`
    /// for every split on this feature (cell boundaries ⊇ split borders, §08.1).
    ///
    /// Single-axis ONLY — errors on a joint (P1 multi-channel) axis, which has no single
    /// representative bin (see [`Self::rep_model_bin_for_axis`]/[`Self::write_rep_bins`]).
    /// Callers reachable on a multi-channel model must use one of those instead; this keeps
    /// the handful of call sites outside this module's actively-maintained P1 scope (the §G1
    /// cell-basis correction/graduation subsystem, which documents its own "green-spine:
    /// raw == axis" assumption) failing LOUDLY rather than silently misrouting.
    fn rep_model_bin(&self, cell: usize) -> Result<u8, PbError> {
        if self.joint.is_some() {
            return Err(PbError::Internal {
                what: "rep_model_bin called on a P1 multi-channel joint axis; use \
                       rep_model_bin_for_axis or write_rep_bins instead"
                    .into(),
            });
        }
        let bin = match cell {
            0 => 0usize,
            1 => 1usize,
            c => {
                let k = *self
                    .model_border_index
                    .get(c - 2)
                    .ok_or_else(|| PbError::Internal {
                        what: "merged cell escaped border index".into(),
                    })?;
                k + 2
            }
        };
        u8::try_from(bin).map_err(|_| PbError::Internal {
            what: "representative model bin exceeded u8".into(),
        })
    }

    /// Model bin → its merged cell. Uses only the merged border indices (a model bin
    /// `b` sits above merged border index `k` iff `k <= b - 2`).
    ///
    /// Single-axis ONLY — see [`Self::rep_model_bin`]'s doc; joint axes use
    /// [`Self::cell_for_row_bins`]/[`Self::cell_for_every_row`] instead (a single model bin
    /// never determines a joint cell alone).
    fn model_bin_to_cell(&self, bin: u8) -> Result<u32, PbError> {
        if self.joint.is_some() {
            return Err(PbError::Internal {
                what: "model_bin_to_cell called on a P1 multi-channel joint axis; use \
                       cell_for_row_bins or cell_for_every_row instead"
                    .into(),
            });
        }
        if u16::from(bin) >= self.model_n_bins {
            return Err(PbError::InvalidInput {
                what: format!(
                    "model bin {bin} outside grid n_bins {} for axis {}",
                    self.model_n_bins, self.axis
                ),
            });
        }
        if bin == 0 {
            return Ok(0);
        }
        // Count merged borders strictly below bin `b` (i.e. model index <= b-2).
        let threshold = i64::from(bin) - 2;
        let below = self
            .model_border_index
            .iter()
            .filter(|&&k| (k as i64) <= threshold)
            .count();
        u32::try_from(below + 1).map_err(|_| PbError::Internal {
            what: "merged cell id exceeded u32".into(),
        })
    }

    /// [`Self::model_bin_to_cell`] precomputed for every model bin `0..model_n_bins`
    /// (<= 255), so a caller that looks up MANY rows on this axis (e.g. `fill_support`,
    /// `build_weights`) pays the O(#borders) linear scan once per bin instead of once
    /// per row. Byte-identical to calling `model_bin_to_cell` per row: same function,
    /// same inputs, just cached — this is caching, not a reimplementation.
    ///
    /// Single-axis ONLY — see [`Self::rep_model_bin`]'s doc.
    fn bin_to_cell_map(&self) -> Result<Vec<u32>, PbError> {
        if self.joint.is_some() {
            return Err(PbError::Internal {
                what: "bin_to_cell_map called on a P1 multi-channel joint axis; use \
                       cell_for_every_row instead"
                    .into(),
            });
        }
        (0..self.model_n_bins)
            .map(|bin| {
                let bin = u8::try_from(bin).map_err(|_| PbError::Internal {
                    what: "model_n_bins exceeded u8 building bin-to-cell map".into(),
                })?;
                self.model_bin_to_cell(bin)
            })
            .collect()
    }

    /// Representative model bin for `cell`, on a SPECIFIC model axis (P1 multi-channel:
    /// `model_axis` selects which channel; single-axis: `model_axis` must equal `self.axis`).
    /// Used by tree-split routing (`leaf_index_for_tuple`, `build_tree_box`), which addresses
    /// a raw feature's axes by MODEL AXIS (`split.axis`), never by raw feature alone once more
    /// than one axis can share a raw.
    ///
    /// # Errors
    /// [`PbError::Internal`] if `model_axis` doesn't match this axis (single-axis) or names
    /// none of this axis's channels (joint); propagates [`Self::rep_model_bin`] otherwise.
    fn rep_model_bin_for_axis(&self, cell: usize, model_axis: usize) -> Result<u8, PbError> {
        match &self.joint {
            Some(j) => j.rep_model_bin_for_axis(cell, model_axis),
            None => {
                if model_axis != self.axis {
                    return Err(PbError::Internal {
                        what: format!(
                            "rep_model_bin_for_axis: model axis {model_axis} != this axis's {}",
                            self.axis
                        ),
                    });
                }
                self.rep_model_bin(cell)
            }
        }
    }

    /// Write representative model bins for `cell` into `out` (indexed by MODEL AXIS, the same
    /// convention `rep_bins` uses throughout this module) — one entry for a single axis, one
    /// per channel for a joint axis.
    ///
    /// # Errors
    /// [`PbError::Internal`] if `self.axis` (single-axis) or a channel's model axis (joint)
    /// escapes `out`; propagates [`Self::rep_model_bin`] otherwise.
    fn write_rep_bins(&self, cell: usize, out: &mut [u8]) -> Result<(), PbError> {
        match &self.joint {
            Some(j) => j.write_rep_bins(cell, out),
            None => {
                let bin = self.rep_model_bin(cell)?;
                let slot = out.get_mut(self.axis).ok_or_else(|| PbError::Internal {
                    what: "write_rep_bins: axis escaped rep_bins".into(),
                })?;
                *slot = bin;
                Ok(())
            }
        }
    }

    /// Merged cell for EVERY row of `x` at once (single-axis: precomputes `bin_to_cell_map`
    /// once, O(model_n_bins), then an O(1) lookup per row — identical mechanics to the pre-P1
    /// caching `fill_support`/`build_weights` already did, just fused into one call; joint: no
    /// compact per-bin map is possible — see [`crate::cat::JointCatAxis`]'s doc — so this reads
    /// every channel's own column per row, O(n_rows · n_channels), unavoidable and still cheap
    /// since `n_channels` is small).
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if `self.axis` (single-axis) or a channel's model axis
    /// (joint) escapes `x`'s columns; propagates [`Self::bin_to_cell_map`]/
    /// [`JointCatAxis::cell_for_channel_bins`] otherwise.
    fn cell_for_every_row(&self, x: &ServeBinnedMatrix) -> Result<Vec<u32>, PbError> {
        match &self.joint {
            Some(j) => {
                let n_rows = x.0.n_rows as usize;
                let cols: Vec<&[u8]> = j
                    .channels
                    .iter()
                    .map(|c| {
                        x.0.data
                            .get(c.model_axis)
                            .map(Vec::as_slice)
                            .ok_or_else(|| PbError::ShapeMismatch {
                                what: format!(
                                    "serve matrix missing column {} for joint axis",
                                    c.model_axis
                                ),
                            })
                    })
                    .collect::<Result<_, _>>()?;
                let mut bins = vec![0u8; cols.len()];
                (0..n_rows)
                    .map(|row| {
                        for (slot, col) in bins.iter_mut().zip(&cols) {
                            *slot = *col.get(row).ok_or_else(|| PbError::Internal {
                                what: "joint axis row escaped column".into(),
                            })?;
                        }
                        j.cell_for_channel_bins(&bins)
                    })
                    .collect()
            }
            None => {
                let map = self.bin_to_cell_map()?;
                let col = x
                    .0
                    .data
                    .get(self.axis)
                    .ok_or_else(|| PbError::ShapeMismatch {
                        what: format!("serve matrix missing column {} for merged axis", self.axis),
                    })?;
                col.iter()
                    .map(|&bin| {
                        map.get(bin as usize)
                            .copied()
                            .ok_or_else(|| PbError::Internal {
                                what: "row escaped bin-to-cell map".into(),
                            })
                    })
                    .collect()
            }
        }
    }
}

/// The merged grids of a fitted model: per raw feature, EITHER the sorted union of every
/// realized split border (single axis) or a flattened joint per-level cell space (P1
/// multi-channel, `MergedAxis::joint`), with the §08.1 cell convention (cell 0 = missing).
#[derive(Debug, Clone)]
pub(crate) struct MergedGrids {
    per_raw: Vec<MergedAxis>,
    /// Total model axis count (`model.provenance.len()`) — DISTINCT from `per_raw.len()`
    /// (the raw feature count) once a raw feature can own more than one axis. Needed to size
    /// axis-indexed buffers (`rep_bins`) independently from raw-indexed ones (`x_cells`).
    n_axes: usize,
}

impl MergedGrids {
    /// Build the merged grids from a fitted `model`. Raw feature ids must still be dense
    /// `0..n_raw_features` (the green-spine invariant, relaxed from "exactly one axis per raw"
    /// to "one OR MORE axes per raw" — P1 multi-channel). Both [`AxisKind::Numeric`] and
    /// [`AxisKind::CategoricalTS`] axes are accepted — a categorical is a Target-Statistic-
    /// encoded ordinal axis carrying a normal [`BorderGrid`] (§04), so the merged-grid logic
    /// and the five I2 checks apply to it identically (the bank is audited on a
    /// `ServeBinnedMatrix` re-encoded through the frozen full-data encoders, R-CATSERVE). The
    /// reserved [`AxisKind::Missing`] axis is not a model feature and is rejected.
    ///
    /// # Errors
    /// [`PbError::InvalidConfig`] if a raw feature id is out of range, has no axis, or (P1)
    /// its channels' frozen level partitions disagree (see [`JointCatAxis::build`]);
    /// [`PbError::Internal`]/[`PbError::InvalidInput`] on malformed grids/splits.
    pub(crate) fn from_model(model: &Model) -> Result<Self, PbError> {
        let n_axes = model.provenance.len();
        // Raw ids must still be dense 0..n_raw; `axes_of_raw[r]` collects EVERY axis mapped to
        // raw `r` (was `Option<usize>`, one axis only, pre-P1).
        let n_raw = crate::data::n_raw_features(&model.provenance);
        let mut axes_of_raw: Vec<Vec<usize>> = vec![Vec::new(); n_raw];
        for (a, prov) in model.provenance.iter().enumerate() {
            if matches!(prov.kind, AxisKind::Missing) {
                return Err(PbError::InvalidConfig {
                    what: "explain does not support a standalone Missing axis kind".into(),
                });
            }
            let r = prov.raw.0 as usize;
            let slot = axes_of_raw
                .get_mut(r)
                .ok_or_else(|| PbError::InvalidConfig {
                    what: format!("raw feature id {r} out of range 0..{n_raw} (v1 explain)"),
                })?;
            slot.push(a);
        }
        for (r, axes) in axes_of_raw.iter().enumerate() {
            if axes.is_empty() {
                return Err(PbError::InvalidConfig {
                    what: format!("raw feature {r} has no axis"),
                });
            }
        }

        // Gather realized split borders per raw feature (dedup by model-border index) — only
        // meaningful for single-axis raw features; a joint raw feature's cell space comes from
        // its channels' frozen levels instead (JointCatAxis::build below), so its entry here
        // is built but simply never read.
        let mut border_indices: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n_raw];
        for (_, tree) in &model.trees {
            for split in &tree.splits {
                let a = split.axis as usize;
                let prov = model.provenance.get(a).ok_or_else(|| PbError::Internal {
                    what: format!("split axis {a} absent from provenance"),
                })?;
                let r = prov.raw.0 as usize;
                if split.bin_le == 0 {
                    // Pure missing-vs-present split (BUG-005): it realizes no finite border,
                    // and merged cell 0 is always the missing cell, so the cell space already
                    // separates exactly the rows it routes.
                    continue;
                }
                let bidx = usize::from(split.bin_le) - 1;
                let grid = model.grids.get(a).ok_or_else(|| PbError::Internal {
                    what: format!("split axis {a} absent from grids"),
                })?;
                if bidx >= grid.borders.len() {
                    return Err(PbError::Internal {
                        what: format!("split bin_le {} escapes grid borders", split.bin_le),
                    });
                }
                let set = border_indices.get_mut(r).ok_or_else(|| PbError::Internal {
                    what: "raw feature escaped border-index table".into(),
                })?;
                set.insert(bidx);
            }
        }

        let mut per_raw = Vec::with_capacity(n_raw);
        for (r, axes) in axes_of_raw.iter().enumerate() {
            if axes.len() == 1 {
                let axis = *axes.first().ok_or_else(|| PbError::Internal {
                    what: "single-axis raw feature lost its only axis".into(),
                })?;
                let grid = model.grids.get(axis).ok_or_else(|| PbError::Internal {
                    what: "merged-grid axis absent from model grids".into(),
                })?;
                if grid.n_bins == 0 {
                    return Err(PbError::InvalidInput {
                        what: format!("model grid {axis} has n_bins=0"),
                    });
                }
                let idxs = border_indices.get(r).ok_or_else(|| PbError::Internal {
                    what: "raw feature escaped border-index table".into(),
                })?;
                let mut model_border_index = Vec::with_capacity(idxs.len());
                let mut borders = Vec::with_capacity(idxs.len());
                for &k in idxs {
                    let b = *grid.borders.get(k).ok_or_else(|| PbError::Internal {
                        what: "merged border index escaped grid borders".into(),
                    })?;
                    model_border_index.push(k);
                    borders.push(b);
                }
                per_raw.push(MergedAxis {
                    axis,
                    borders,
                    model_border_index,
                    model_n_bins: grid.n_bins,
                    joint: None,
                });
            } else {
                // P1 multi-channel (>=2 axes for one raw feature): flatten into a joint
                // per-level cell space. `channel_axes_for_raw` re-derives the same axis set
                // from provenance (sorted by encoder id) rather than trusting `axes`'s
                // insertion order, so the channel order is canonical regardless of how the
                // model's axes happened to be laid out.
                let raw_id = FeatureId(u32::try_from(r).map_err(|_| PbError::Internal {
                    what: "raw feature index exceeded u32".into(),
                })?);
                let channels: Vec<ChannelAxis> = channel_axes_for_raw(&model.provenance, raw_id);
                let joint = JointCatAxis::build(
                    raw_id,
                    channels,
                    &model.grids,
                    &model.schema.cat_encoders,
                )?;
                per_raw.push(MergedAxis {
                    axis: 0,
                    borders: Vec::new(),
                    model_border_index: Vec::new(),
                    model_n_bins: 0,
                    joint: Some(joint),
                });
            }
        }
        Ok(MergedGrids { per_raw, n_axes })
    }

    /// Number of DISTINCT RAW FEATURES (was `n_features`, conflated with axis count before
    /// P1 multi-channel — a raw feature can now own more than one axis).
    fn n_raw_features(&self) -> usize {
        self.per_raw.len()
    }

    /// Total model axis count (`>= n_raw_features`, strictly greater once any raw feature has
    /// multiple channels). Sizes axis-indexed buffers (`rep_bins`), never raw-indexed ones.
    fn n_axes(&self) -> usize {
        self.n_axes
    }

    fn axis(&self, raw: FeatureId) -> Result<&MergedAxis, PbError> {
        self.per_raw
            .get(raw.0 as usize)
            .ok_or_else(|| PbError::Internal {
                what: format!("merged grid missing raw feature {}", raw.0),
            })
    }

    fn cells(&self, raw: FeatureId) -> Result<usize, PbError> {
        Ok(self.axis(raw)?.cells())
    }

    fn axis_id(&self, raw: FeatureId) -> Result<AxisId, PbError> {
        let ma = self.axis(raw)?;
        Ok(AxisId {
            raw,
            borders: ma.borders.clone(),
            cells: u32::try_from(ma.cells()).map_err(|_| PbError::Internal {
                what: "merged cell count exceeded u32".into(),
            })?,
            joint_channels: ma
                .joint
                .as_ref()
                .map(|j| j.channels.iter().map(|c| c.id).collect()),
            band_of: None,
        })
    }

    /// One [`BorderGrid`] per raw feature (the bank's shared merged grid). A joint (P1
    /// multi-channel) raw feature gets a PLACEHOLDER grid (`borders` empty, `n_bins` = the
    /// joint cell count) — real border-matching for such a raw feature is never done through
    /// this `BorderGrid` (see `scoring.rs`'s joint-aware `build_cell_maps` path), so this
    /// exists only to keep `TableBank.merged_grids` a stable, uniformly-shaped `Vec` other
    /// code can length-check against `n_raw_features`.
    fn border_grids(&self) -> Result<Vec<BorderGrid>, PbError> {
        let mut out = Vec::with_capacity(self.per_raw.len());
        for ma in &self.per_raw {
            let n_bins = u16::try_from(ma.cells()).map_err(|_| PbError::Internal {
                what: "merged grid n_bins exceeded u16".into(),
            })?;
            out.push(BorderGrid {
                borders: ma.borders.clone(),
                n_bins,
                missing_bin: 0,
            });
        }
        Ok(out)
    }
}

// ===========================================================================
// §08.2 — Accumulation: ensemble → raw tensors.
// ===========================================================================

/// One realized tree-support tensor before purification.
struct RawTable {
    u: FeatureSet,
    axes: Vec<AxisId>,
    values: Tensor,
}

/// The raw (pre-purify) bank: intercept + per-support tensors keyed by support.
struct RawBank {
    f0: f64,
    tables: BTreeMap<FeatureSet, RawTable>,
}

/// One pre-purification effect supplied by another exactness-preserving layer. `pub`: the
/// `t-boost-py` FFI crate feeds a post-graduation bank's own (mutated) table values back
/// through [`purify_raw_effects`] as "raw" effects to re-purify it — see that function's doc.
pub struct RawEffect {
    /// The effect support.
    pub u: FeatureSet,
    /// Raw, uncentered score-space values on the supplied merged grids.
    pub values: Tensor,
    /// Display-only effective `w`-mass (spec §08.7), same tensor shape as `values` — a
    /// row count when unweighted, `Σ w·e` per cell when derived from a per-row mass.
    /// Carried through unchanged by [`purify_raw_effects`] (§08.7's marginal is
    /// data-derived, not a function of the reference measure it purifies under).
    pub support: Tensor,
}

/// The distinct raw-feature support of a tree (sorted). Repeated split levels refine this
/// lower-order support; they do not create duplicate table axes.
fn tree_support(model: &Model, tree: &crate::engine::ObliviousTree) -> Result<FeatureSet, PbError> {
    let mut ids: SmallVec<[FeatureId; MAX_ORDER]> = SmallVec::new();
    for split in &tree.splits {
        let prov = model
            .provenance
            .get(split.axis as usize)
            .ok_or_else(|| PbError::Internal {
                what: format!("split axis {} absent from provenance", split.axis),
            })?;
        if !ids.contains(&prov.raw) {
            ids.push(prov.raw);
        }
    }
    ids.sort_unstable();
    Ok(FeatureSet(ids))
}

/// Walk every coordinate tuple of a tensor of `extents` (an in-place odometer; no
/// allocation of the full coordinate list). The empty-extents tensor is one cell.
fn walk_extents(
    extents: &[usize],
    mut f: impl FnMut(&[usize]) -> Result<(), PbError>,
) -> Result<(), PbError> {
    let mut idx = vec![0usize; extents.len()];
    loop {
        f(&idx)?;
        let mut k = extents.len();
        loop {
            if k == 0 {
                return Ok(());
            }
            k -= 1;
            let e = *extents.get(k).ok_or_else(|| PbError::Internal {
                what: "odometer axis escaped extents".into(),
            })?;
            let v = idx.get_mut(k).ok_or_else(|| PbError::Internal {
                what: "odometer index escaped buffer".into(),
            })?;
            *v += 1;
            if *v < e {
                break;
            }
            *v = 0;
        }
    }
}

/// Coordinate with position `p` dropped (no indexing).
fn drop_index(coord: &[usize], p: usize) -> Vec<usize> {
    coord
        .iter()
        .enumerate()
        .filter_map(|(i, &c)| if i == p { None } else { Some(c) })
        .collect()
}

/// `Π extents` as `u64`, or `PbError::Internal` on overflow.
fn product_u64(extents: &[usize]) -> Result<u64, PbError> {
    let mut acc = 1u64;
    for &e in extents {
        acc = acc.checked_mul(e as u64).ok_or_else(|| PbError::Internal {
            what: "merged tensor cell count overflowed u64".into(),
        })?;
    }
    Ok(acc)
}

/// `Π extents` as `u64`, SATURATING to `u64::MAX` on overflow. Used where only the comparison
/// "does this exceed the enumeration cap?" matters — never an exact count. A joint grid too
/// large to represent (e.g. a 130-feature model → `2^130` cells) must route to the sampling
/// branch, not hard-error, so the gate sweep stays bounded on wide models (§08.6/§08.8).
fn saturating_product_u64(extents: &[usize]) -> u64 {
    let mut acc = 1u64;
    for &e in extents {
        acc = acc.saturating_mul(e as u64);
    }
    acc
}

/// Compute the leaf index a tree assigns to a merged cell-tuple of its support, routing
/// each split via the SINGLE canonical `low_bit` on the cell's representative model bin.
fn leaf_index_for_tuple(
    model: &Model,
    tree: &crate::engine::ObliviousTree,
    grids: &MergedGrids,
    u_ids: &[FeatureId],
    tuple: &[usize],
) -> Result<usize, PbError> {
    let mut leaf_idx = 0usize;
    for (level, split) in tree.splits.iter().enumerate() {
        let prov = model
            .provenance
            .get(split.axis as usize)
            .ok_or_else(|| PbError::Internal {
                what: "split axis absent from provenance".into(),
            })?;
        let pos = u_ids
            .iter()
            .position(|r| *r == prov.raw)
            .ok_or_else(|| PbError::Internal {
                what: "split raw feature absent from tree support".into(),
            })?;
        let cell = *tuple.get(pos).ok_or_else(|| PbError::Internal {
            what: "tuple shorter than support".into(),
        })?;
        // Axis-aware: a P1 multi-channel raw feature's joint cell resolves to a DIFFERENT
        // representative bin per channel, and `split.axis` says which channel this specific
        // split is on.
        let rep_bin = grids
            .axis(prov.raw)?
            .rep_model_bin_for_axis(cell, split.axis as usize)?;
        let bit = usize::from(low_bit(rep_bin, split.bin_le, split.missing_left));
        leaf_idx |= bit << level;
    }
    Ok(leaf_idx)
}

/// Every tree's [`tree_support`], in tree order — computed once per bank build and shared by
/// [`accumulate_with_supports`] and the factored shed's per-support tree index.
fn tree_supports(model: &Model) -> Result<Vec<FeatureSet>, PbError> {
    model
        .trees
        .iter()
        .map(|(_, tree)| tree_support(model, tree))
        .collect()
}

/// Per support position `k`, the leaf-index bits each merged cell contributes:
/// `bits[k][cell]` ORs `low_bit(..) << level` over the tree's splits on `u_ids[k]`, so a cell
/// tuple's leaf index is the OR of its positions' entries — exactly the fold
/// [`leaf_index_for_tuple`] performs per tuple, with the routing hoisted out of the cell walk.
fn tree_cell_bits(
    model: &Model,
    tree: &crate::engine::ObliviousTree,
    grids: &MergedGrids,
    u_ids: &[FeatureId],
    extents: &[usize],
) -> Result<Vec<Vec<usize>>, PbError> {
    let mut bits: Vec<Vec<usize>> = extents.iter().map(|&e| vec![0usize; e]).collect();
    for (level, split) in tree.splits.iter().enumerate() {
        let prov = model
            .provenance
            .get(split.axis as usize)
            .ok_or_else(|| PbError::Internal {
                what: "split axis absent from provenance".into(),
            })?;
        let pos = u_ids
            .iter()
            .position(|r| *r == prov.raw)
            .ok_or_else(|| PbError::Internal {
                what: "split raw feature absent from tree support".into(),
            })?;
        let ma = grids.axis(prov.raw)?;
        let slot = bits.get_mut(pos).ok_or_else(|| PbError::Internal {
            what: "tuple shorter than support".into(),
        })?;
        for (cell, b) in slot.iter_mut().enumerate() {
            let rep_bin = ma.rep_model_bin_for_axis(cell, split.axis as usize)?;
            *b |= usize::from(low_bit(rep_bin, split.bin_le, split.missing_left)) << level;
        }
    }
    Ok(bits)
}

/// Add `alpha · leaf` to every cell of a dense row-major table, the leaf of each cell tuple
/// read through [`tree_cell_bits`]. Bit-identical to the per-cell walk in [`accumulate`]:
/// the same row-major visit order and the same single `+= alpha * leaf` per cell.
fn accumulate_dense_tree(
    data: &mut [f64],
    extents: &[usize],
    bits: &[Vec<usize>],
    leaves: &[f32],
    alpha: f64,
) -> Result<(), PbError> {
    let escaped = |leaf_idx: usize| PbError::Internal {
        what: format!(
            "oblivious leaf index {leaf_idx} escaped 0..{}",
            leaves.len()
        ),
    };
    let shape_err = || PbError::Internal {
        what: "dense accumulation shape mismatch".into(),
    };
    let (&n_last, lead) = extents.split_last().ok_or_else(shape_err)?;
    let (b_last, b_lead) = bits.split_last().ok_or_else(shape_err)?;
    let b_last = b_last.get(..n_last).ok_or_else(shape_err)?;
    if b_lead.len() != lead.len() || n_last == 0 {
        return Err(shape_err());
    }
    let mut idx = vec![0usize; lead.len()];
    for row in data.chunks_exact_mut(n_last) {
        let mut pre = 0usize;
        for (bk, &c) in b_lead.iter().zip(&idx) {
            pre |= *bk.get(c).ok_or_else(shape_err)?;
        }
        for (v, &bl) in row.iter_mut().zip(b_last) {
            let leaf_idx = pre | bl;
            let leaf = *leaves.get(leaf_idx).ok_or_else(|| escaped(leaf_idx))?;
            *v += alpha * f64::from(leaf);
        }
        for (slot, &e) in idx.iter_mut().zip(lead).rev() {
            *slot += 1;
            if *slot < e {
                break;
            }
            *slot = 0;
        }
    }
    Ok(())
}

/// Accumulate the ensemble into raw per-support tensors (spec §08.2). Each tree adds
/// `alpha · leaf` into every merged cell of its support; the table-budget firewall is
/// checked at lazy allocation.
#[cfg(test)]
fn accumulate(
    model: &Model,
    grids: &MergedGrids,
    budget: &TableBudget,
) -> Result<(RawBank, BTreeSet<FeatureSet>), PbError> {
    accumulate_with_supports(model, grids, budget, &tree_supports(model)?)
}

/// [`accumulate`] over precomputed [`tree_supports`] (one per tree, in tree order).
#[cfg(test)]
fn accumulate_with_supports(
    model: &Model,
    grids: &MergedGrids,
    budget: &TableBudget,
    supports: &[FeatureSet],
) -> Result<(RawBank, BTreeSet<FeatureSet>), PbError> {
    accumulate_with_plan(model, grids, budget, supports, None)
}

/// Reserve the entire decomposition's dense footprint before allocating any tensor.
fn table_allocation_plan(
    model: &Model,
    grids: &MergedGrids,
    budget: &TableBudget,
    supports: &[FeatureSet],
) -> Result<BTreeMap<FeatureSet, bool>, PbError> {
    let mut roots: BTreeSet<FeatureSet> = supports.iter().cloned().collect();
    if let Some(correction) = &model.correction {
        for table in &correction.tables {
            let mut ids = SmallVec::new();
            for axis in &table.axes {
                let raw = model
                    .provenance
                    .get(*axis as usize)
                    .ok_or_else(|| PbError::Internal {
                        what: "correction axis absent from provenance".into(),
                    })?
                    .raw;
                if !ids.contains(&raw) {
                    ids.push(raw);
                }
            }
            ids.sort_unstable();
            roots.insert(FeatureSet(ids));
        }
    }
    let mut closure = BTreeSet::new();
    for root in roots {
        for mask in 1..(1usize << root.order()) {
            let subset = FeatureSet(
                root.0
                    .iter()
                    .enumerate()
                    .filter_map(|(i, raw)| ((mask >> i) & 1 == 1).then_some(*raw))
                    .collect(),
            );
            if !matches!(budget.on_overflow, OverflowPolicy::Factored)
                || subset.order() < LEGACY_MAX_ORDER
            {
                closure.insert(subset);
            }
        }
    }
    let mut counts = BTreeMap::new();
    let mut total = 0u64;
    for support in closure {
        let extents = support
            .0
            .iter()
            .map(|r| grids.cells(*r))
            .collect::<Result<Vec<_>, _>>()?;
        let cells = product_u64(&extents)?;
        total = total.checked_add(cells).ok_or_else(|| PbError::Internal {
            what: "decomposition cell count overflowed".into(),
        })?;
        counts.insert(support, cells);
    }
    let sparse = matches!(budget.on_overflow, OverflowPolicy::SparseFallback { .. })
        && (total > budget.max_bank_cells
            || counts.values().any(|cells| *cells > budget.max_table_cells));
    if !sparse {
        for (support, cells) in &counts {
            if *cells > budget.max_table_cells {
                return Err(PbError::TableBudget {
                    what: format!("decomposition table {support:?}"),
                    cells: *cells,
                    budget: budget.max_table_cells,
                });
            }
        }
        if total > budget.max_bank_cells {
            return Err(PbError::TableBudget {
                what: "complete decomposition bank".into(),
                cells: total,
                budget: budget.max_bank_cells,
            });
        }
    }
    Ok(counts
        .into_keys()
        .map(|support| (support, sparse))
        .collect())
}

fn accumulate_with_plan(
    model: &Model,
    grids: &MergedGrids,
    budget: &TableBudget,
    supports: &[FeatureSet],
    plan: Option<&BTreeMap<FeatureSet, bool>>,
) -> Result<(RawBank, BTreeSet<FeatureSet>), PbError> {
    if supports.len() != model.trees.len() {
        return Err(PbError::Internal {
            what: "tree support list does not match the ensemble".into(),
        });
    }
    if let OverflowPolicy::SparseFallback { density_threshold } = budget.on_overflow {
        if !density_threshold.is_finite() || !(0.0..=1.0).contains(&density_threshold) {
            return Err(PbError::InvalidConfig {
                what: "SparseFallback density_threshold must be finite and in [0, 1]".into(),
            });
        }
    }
    let mut tables: BTreeMap<FeatureSet, RawTable> = BTreeMap::new();
    let mut bank_cells: u64 = 0;
    if let Some(plan) = plan {
        for (support, sparse) in plan {
            let axes = support
                .0
                .iter()
                .map(|raw| grids.axis_id(*raw))
                .collect::<Result<Vec<_>, _>>()?;
            let extents = support
                .0
                .iter()
                .map(|raw| grids.cells(*raw))
                .collect::<Result<Vec<_>, _>>()?;
            let values = if *sparse {
                Tensor::try_sparse_zeros(extents)?
            } else {
                Tensor::try_zeros(extents)?
            };
            tables.insert(
                support.clone(),
                RawTable {
                    u: support.clone(),
                    axes,
                    values,
                },
            );
        }
    }
    // Order-3 supports kept in compact factored form under OverflowPolicy::Factored — never
    // materialized as a dense cube here; the pipeline sheds each per tree into the shared lower
    // tables and keeps only the pure 3-way residual factored (§08.10).
    let mut factored_supports: BTreeSet<FeatureSet> = BTreeSet::new();
    for ((alpha, tree), u) in model.trees.iter().zip(supports) {
        if factored_supports.contains(u) {
            continue;
        }
        let u_ids: Vec<FeatureId> = u.0.iter().copied().collect();
        if !tables.contains_key(u) {
            let mut extents = Vec::with_capacity(u_ids.len());
            for r in &u_ids {
                extents.push(grids.cells(*r)?);
            }
            let table_cells = product_u64(&extents)?;
            // §08.10: under the Factored policy keep EVERY order-3 effect factored, not only those
            // whose dense cube exceeds `max_table_cells`. A depth-3 tree realizes just 2x2x2 regions,
            // so its dense merged cube (Π realized extents, e.g. 239x166x61) is a mostly-redundant
            // re-embedding of lower-order structure; hundreds of such mid-size cubes each stay under
            // the per-table cap yet sum past `max_bank_cells`. Shedding their order-<=2 mass into the
            // shared dense lower tables and keeping the pure 3-way residual factored is exact (same
            // I2 gates as the >budget path) and bounds the bank to the compact low-order tables.
            // Bounded ABOVE as well as below. The shed driver iterates exactly
            // `LEGACY_MAX_ORDER..=MAX_ORDER`, so claiming a support outside that band here
            // would make it vanish: no dense table, no factored effect, and
            // `check_factorable_order` never reached. `Model::validate`/`i1_shape_ok` cap
            // distinct raws at `MAX_ORDER` upstream, but `explain` does not re-validate, so
            // this is the difference between "unreachable" and "fails closed".
            if matches!(budget.on_overflow, OverflowPolicy::Factored)
                && (LEGACY_MAX_ORDER..=MAX_ORDER).contains(&u.order())
            {
                // P-D4: a LIFTED order-3 tree is a k0 x k1 x k2 product box, which no single
                // rank-1 `FactoredBox` can hold — P-D1 therefore routed such supports to the
                // dense cube, which measured 601,942-cell tables on beMTPL16 and blew the
                // 32M bank firewall outright on catelematic13. `build_tree_boxes` now
                // DECOMPOSES a lifted tree into one rank-1 box per realized region tuple, so
                // the factored representation is available at every depth and this special
                // case is gone.
                factored_supports.insert(u.clone());
                continue;
            }
            let values = if table_cells > budget.max_table_cells {
                match budget.on_overflow {
                    OverflowPolicy::Error => {
                        return Err(PbError::TableBudget {
                            what: format!("table {u:?}"),
                            cells: table_cells,
                            budget: budget.max_table_cells,
                        });
                    }
                    OverflowPolicy::SparseFallback { .. } => {
                        // Allocating the (empty) sparse tensor itself is O(1) — the walk below
                        // enforces R-TABLEBUDGET's fail-BEFORE-allocation promise incrementally,
                        // since an oblivious tree assigns a leaf to every point of its support
                        // cube (densely populating ~100% of the merged cells is the norm here,
                        // not the adversarial case) so waiting until after a full walk to check
                        // density would let the exact multi-hundred-MB spike the firewall exists
                        // to prevent happen first.
                        Tensor::try_sparse_zeros(extents.clone())?
                    }
                    OverflowPolicy::Factored => {
                        // Unreachable in practice: the branch above already claimed every
                        // support of order >= LEGACY_MAX_ORDER, and order 1/2 cubes are
                        // bounded by 255 / 255^2, well under any sane `max_table_cells`.
                        // Kept as a loud refusal rather than a silent densification if a
                        // caller ever sets a budget below 65k.
                        if u.order() < LEGACY_MAX_ORDER {
                            return Err(PbError::TableBudget {
                                what: format!(
                                    "an order-{} effect cannot be factored (only order \
                                     >= {LEGACY_MAX_ORDER} can); raise max_table_cells; {u:?}",
                                    u.order()
                                ),
                                cells: table_cells,
                                budget: budget.max_table_cells,
                            });
                        }
                        factored_supports.insert(u.clone());
                        continue;
                    }
                }
            } else {
                bank_cells =
                    bank_cells
                        .checked_add(table_cells)
                        .ok_or_else(|| PbError::Internal {
                            what: "bank cell count overflowed u64".into(),
                        })?;
                if bank_cells > budget.max_bank_cells {
                    return Err(PbError::TableBudget {
                        what: "bank".into(),
                        cells: bank_cells,
                        budget: budget.max_bank_cells,
                    });
                }
                Tensor::try_zeros(extents.clone())?
            };
            let mut axes = Vec::with_capacity(u_ids.len());
            for r in &u_ids {
                axes.push(grids.axis_id(*r)?);
            }
            tables.insert(
                u.clone(),
                RawTable {
                    u: u.clone(),
                    axes,
                    values,
                },
            );
        }
        let table = tables.get_mut(u).ok_or_else(|| PbError::Internal {
            what: "raw table vanished after insert".into(),
        })?;
        let extents = table.values.shape();
        let alpha = f64::from(*alpha);
        // SparseFallback occupancy cap, checked INSIDE the walk (not after it): fail as soon
        // as `entries.len()` would exceed `density_threshold * table_cells`, so a restrictive
        // threshold bails within one entry of the cap instead of after materializing the
        // whole (potentially ~100%-dense) cube. `density_threshold == 1.0` makes the cap equal
        // `table_cells` exactly, which `entries.len()` can never exceed — an explicit "accept
        // any density, just don't hard-fail" opt-in always succeeds, unbounded density and all
        // (spec §08's documented escape hatch; xtask's
        // adversarial-fixture regression test pins this). Dense-backed tables (the non-fallback
        // case) skip the check entirely — `is_sparse()` is false and `density_cap` is `None`.
        let density_cap = match (table.values.is_sparse(), budget.on_overflow) {
            (true, OverflowPolicy::SparseFallback { density_threshold }) => {
                let table_cells = product_u64(&extents)?;
                Some((density_threshold * table_cells as f64).floor() as u64)
            }
            _ => None,
        };
        // Dense tables (every table outside SparseFallback): route each axis once per tree
        // instead of once per cell tuple — see `tree_cell_bits`.
        if density_cap.is_none() && !extents.is_empty() {
            if let Some(data) = table.values.dense_slice_mut() {
                let bits = tree_cell_bits(model, tree, grids, &u_ids, &extents)?;
                accumulate_dense_tree(data, &extents, &bits, &tree.leaves, alpha)?;
                continue;
            }
        }
        walk_extents(&extents, |tuple| {
            let leaf_idx = leaf_index_for_tuple(model, tree, grids, &u_ids, tuple)?;
            let leaf = *tree.leaves.get(leaf_idx).ok_or_else(|| PbError::Internal {
                what: format!(
                    "oblivious leaf index {leaf_idx} escaped 0..{}",
                    tree.leaves.len()
                ),
            })?;
            table.values.add(tuple, alpha * f64::from(leaf))?;
            if let Some(cap) = density_cap {
                if let Some(nnz) = table.values.sparse_nnz() {
                    if nnz as u64 > cap {
                        return Err(PbError::TableBudget {
                            what: format!(
                                "table {u:?}: SparseFallback occupancy exceeded \
                                 density_threshold before accumulation finished \
                                 (fail-before-allocate)"
                            ),
                            cells: product_u64(&extents)?,
                            budget: budget.max_table_cells,
                        });
                    }
                }
            }
            Ok(())
        })?;
    }

    Ok((
        RawBank {
            f0: f64::from(model.f0),
            tables,
        },
        factored_supports,
    ))
}

// ===========================================================================
// §08.4 — The reference measure `w` (per-axis cell mass).
// ===========================================================================

/// Per-axis merged-cell `w`-mass, precomputed from the data and the chosen
/// [`RefMeasure`]. Indexed by raw feature; each inner vector sums to 1 and is strictly
/// positive (so zero-mean/convergence never break). Built over a [`ServeBinnedMatrix`]
/// (frozen encoders, R-CATSERVE) so the audited mass matches the deployed model.
#[derive(Debug, Clone)]
pub(crate) struct WeightCache {
    per_axis: Vec<Vec<f64>>,
    /// Stamped for provenance; the bank carries the canonical copy.
    kind: RefMeasure,
}

impl WeightCache {
    fn axis(&self, raw: FeatureId) -> Result<&[f64], PbError> {
        self.per_axis
            .get(raw.0 as usize)
            .map(Vec::as_slice)
            .ok_or_else(|| PbError::Internal {
                what: format!("weight cache missing raw feature {}", raw.0),
            })
    }
}

/// Build the per-axis cell weights (spec §08.4). `Joint` is rejected in v1 (it couples
/// axes and breaks the variance-sum identity).
///
/// `mass = None` counts a flat `1.0` per row (the historical/default behavior, BYTE-
/// IDENTICAL to pre-§08.7 weights); `Some(m)` sums `m[row]` per cell instead, so
/// `ProductMarginals`' empirical marginal becomes the same per-cell effective `w`-mass
/// `fill_support` accumulates into `support` (spec §08.7) — the reference measure and the
/// displayed support agree on what backs a cell. This is BY DESIGN not display-only:
/// `ProductMarginals` weights feed `purify`, so `values`/Sobol/SE bands shift under a
/// mass-derived measure. `Uniform` ignores `mass` entirely (its weights never depend on
/// data).
///
/// # Errors
/// [`PbError::InvalidConfig`] for `Joint` or non-finite/non-positive `laplace`;
/// [`PbError::ShapeMismatch`] if `mass`'s length disagrees with `x`'s row count;
/// [`PbError::InvalidInput`] if any mass entry is negative or non-finite; plus
/// propagated grid errors.
fn build_weights(
    x: &ServeBinnedMatrix,
    grids: &MergedGrids,
    w: &RefMeasure,
    mass: Option<&[f32]>,
) -> Result<WeightCache, PbError> {
    let rule = axis_rule(w)?;

    let n_rows = x.0.n_rows as usize;
    // Validate unconditionally (not gated on `laplace`): `Uniform` ignores `mass` in ITS
    // OWN weights, but a garbage mass would otherwise silently corrupt nothing here (no
    // error) while still reaching `fill_support` in the same `explain_bank_with_mass`
    // pass — better to fail fast at one place than let validity depend on `w`'s variant.
    validate_mass(mass, n_rows, "weight")?;
    // Total effective mass over all rows: every row lands in exactly one cell per axis,
    // so this is the same denominator regardless of axis. `mass = None` uses `n_rows as
    // f64` directly (not a summed `Σ 1.0`) so the unweighted path is the EXACT same
    // expression as before `mass` existed — no new floating-point summation on it.
    let total_mass = match mass {
        Some(m) => m.iter().map(|&v| f64::from(v)).sum(),
        None => n_rows as f64,
    };
    let mut per_axis = Vec::with_capacity(grids.n_raw_features());
    for ma in &grids.per_raw {
        let cells = ma.cells();
        let raw_w = if !rule.needs_counts() {
            rule.raw_weights(&[], 0.0, cells)
        } else {
            {
                // ŵ_emp is the per-cell mass share, Σ mass / total_mass (§08.4); the rule
                // decides how it blends with the uniform component.
                let mut counts = vec![0.0_f64; cells];
                // `cell_for_every_row` covers both the single-axis (O(model_n_bins) cached
                // map, O(1) per row) and P1 multi-channel joint (O(n_rows · n_channels), no
                // compact per-bin map possible) cases uniformly — see its doc.
                let row_cells = ma.cell_for_every_row(x)?;
                for (row, &cell) in row_cells.iter().enumerate() {
                    let slot = counts
                        .get_mut(cell as usize)
                        .ok_or_else(|| PbError::Internal {
                            what: "weight cell escaped counts".into(),
                        })?;
                    let m = match mass {
                        Some(m) => f64::from(*m.get(row).ok_or_else(|| PbError::Internal {
                            what: "weight mass row escaped column".into(),
                        })?),
                        None => 1.0_f64,
                    };
                    *slot += m;
                }
                let inv_n = if total_mass > 0.0 {
                    1.0 / total_mass
                } else {
                    0.0
                };
                rule.raw_weights(&counts, inv_n, cells)
            }
        };
        let total: f64 = raw_w.iter().sum();
        if total.is_nan() || total <= 0.0 {
            return Err(PbError::Internal {
                what: "reference-measure axis weights summed to zero".into(),
            });
        }
        per_axis.push(raw_w.iter().map(|x| x / total).collect());
    }
    Ok(WeightCache {
        per_axis,
        kind: w.clone(),
    })
}

// ===========================================================================
// §08.3 — Purification: the mass-moving cascade.
// ===========================================================================

/// Subtract the `axis_w`-weighted slice mean along position `p` from `values`,
/// returning that mean as a tensor over the remaining axes (the mass moved one order
/// down). For an order-1 table the returned tensor is 0-D (a scalar → the intercept).
#[cfg(test)]
fn center_along(values: &mut Tensor, p: usize, axis_w: &[f64]) -> Result<Tensor, PbError> {
    center_along_with_budget(values, p, axis_w, None)
}

fn check_sparse_budget(values: &Tensor, budget: Option<&TableBudget>) -> Result<(), PbError> {
    if let Some(TableBudget {
        on_overflow: OverflowPolicy::SparseFallback { density_threshold },
        max_table_cells,
        ..
    }) = budget
    {
        if let Some(nnz) = values.sparse_nnz() {
            if nnz as f64 > (density_threshold * values.len() as f64).floor() {
                return Err(PbError::TableBudget {
                    what: "sparse decomposition density".into(),
                    cells: values.len() as u64,
                    budget: *max_table_cells,
                });
            }
        }
    }
    Ok(())
}

fn center_along_with_budget(
    values: &mut Tensor,
    p: usize,
    axis_w: &[f64],
    budget: Option<&TableBudget>,
) -> Result<Tensor, PbError> {
    let extents = values.shape();
    let sub_extents = drop_index(&extents, p);
    let mut means = if values.is_sparse() {
        Tensor::try_sparse_zeros(sub_extents)?
    } else {
        Tensor::try_zeros(sub_extents)?
    };
    if let (Some(split), Some(data), Some(mdata)) = (
        axis_split(&extents, p),
        values.dense_slice_mut(),
        means.dense_slice_mut(),
    ) {
        center_dense(data, mdata, split, axis_w)?;
        return Ok(means);
    }
    // Pass 1: accumulate the weighted slice mean for each lower-order coordinate.
    walk_extents(&extents, |coord| {
        let cell_p = *coord.get(p).ok_or_else(|| PbError::Internal {
            what: "centering position escaped coord".into(),
        })?;
        let wp = *axis_w.get(cell_p).ok_or_else(|| PbError::Internal {
            what: "centering cell escaped axis weights".into(),
        })?;
        let v = values.at(coord).ok_or_else(|| PbError::Internal {
            what: "centering coord out of range".into(),
        })?;
        means.add(&drop_index(coord, p), wp * v)?;
        check_sparse_budget(&means, budget)
    })?;
    // Pass 2: subtract the mean from every cell of the slice.
    walk_extents(&extents, |coord| {
        let m = means
            .at(&drop_index(coord, p))
            .ok_or_else(|| PbError::Internal {
                what: "centering mean coord out of range".into(),
            })?;
        values.add(coord, -m)?;
        check_sparse_budget(values, budget)
    })?;
    Ok(means)
}

/// A row-major shape viewed around axis `p` as `(outer, mid, inner)`: the cell count before
/// `p`, the extent of `p`, and the cell count after it, so cell `(o, i, j)` sits at flat index
/// `(o * mid + i) * inner + j` and its slice mean over `p` at `o * inner + j`. `None` when `p`
/// is out of range, an extent is zero, or the product overflows.
fn axis_split(extents: &[usize], p: usize) -> Option<(usize, usize, usize)> {
    let mid = *extents.get(p)?;
    let outer = extents
        .get(..p)?
        .iter()
        .try_fold(1usize, |a, &e| a.checked_mul(e))?;
    let inner = extents
        .get(p + 1..)?
        .iter()
        .try_fold(1usize, |a, &e| a.checked_mul(e))?;
    let cells = outer.checked_mul(mid)?.checked_mul(inner)?;
    (cells > 0).then_some((outer, mid, inner))
}

/// Dense [`center_along`]. Bit-identical to the per-cell odometer walk: every mean cell
/// starts at `0.0` and receives `w[i] * v` for `i` ascending (the order a row-major walk
/// reaches them), then every cell receives `-mean`, so each f64 sees the same operations in
/// the same order. Only the per-cell offset arithmetic and `drop_index` allocations are gone.
fn center_dense(
    data: &mut [f64],
    means: &mut [f64],
    (outer, mid, inner): (usize, usize, usize),
    axis_w: &[f64],
) -> Result<(), PbError> {
    if data.len() != outer * mid * inner || means.len() != outer * inner || inner == 0 || mid == 0 {
        return Err(PbError::Internal {
            what: "centering buffers do not match their shape".into(),
        });
    }
    let w = axis_w.get(..mid).ok_or_else(|| PbError::Internal {
        what: "centering cell escaped axis weights".into(),
    })?;
    for (block, mrow) in data
        .chunks_exact(mid * inner)
        .zip(means.chunks_exact_mut(inner))
    {
        for (row, &wp) in block.chunks_exact(inner).zip(w) {
            for (m, &v) in mrow.iter_mut().zip(row) {
                *m += wp * v;
            }
        }
    }
    for (block, mrow) in data
        .chunks_exact_mut(mid * inner)
        .zip(means.chunks_exact(inner))
    {
        for row in block.chunks_exact_mut(inner) {
            for (v, &m) in row.iter_mut().zip(mrow) {
                *v += -m;
            }
        }
    }
    Ok(())
}

/// Tables centred per parallel batch in [`purify`] (bounds the means held at once).
const PURIFY_CHUNK: usize = 256;

/// `u` with the raw feature at position `p` removed.
fn support_without(u: &FeatureSet, p: usize) -> FeatureSet {
    FeatureSet(
        u.0.iter()
            .enumerate()
            .filter_map(|(i, &f)| if i == p { None } else { Some(f) })
            .collect(),
    )
}

/// Purify the raw bank into the canonical fANOVA tables (spec §08.3). Single pass per
/// axis in decreasing `|u|` (3→2→1→intercept) — exact for axis-factorized `w`. The
/// cascade lazily creates the lower-order tables it feeds.
fn purify(
    raw: RawBank,
    w: &WeightCache,
    grids: &MergedGrids,
    mode: PurifyMode,
) -> Result<TableBank, PbError> {
    purify_with_budget(raw, w, grids, mode, None)
}

fn purify_with_budget(
    raw: RawBank,
    w: &WeightCache,
    grids: &MergedGrids,
    mode: PurifyMode,
    budget: Option<&TableBudget>,
) -> Result<TableBank, PbError> {
    // Work map: support → (axes, values). Seed from the raw realized supports.
    let mut axes_of: BTreeMap<FeatureSet, Vec<AxisId>> = BTreeMap::new();
    let mut values_of: BTreeMap<FeatureSet, Tensor> = BTreeMap::new();
    for (u, rt) in raw.tables {
        axes_of.insert(u.clone(), rt.axes);
        values_of.insert(u, rt.values);
    }
    let mut f0 = raw.f0;

    let single_pass = matches!(mode, PurifyMode::SinglePass);
    if !single_pass {
        return Err(PbError::InvalidConfig {
            what: "v1 purify is SinglePass only (product/uniform w)".into(),
        });
    }

    // The fANOVA cascade is n-dimensional: `center_along` drops one axis of an
    // arbitrary-rank `Tensor` and `walk_extents` is an odometer of arbitrary rank, so the
    // ONLY thing that ever capped this at 3 was this loop bound. At order 4 the cascade
    // runs 4 -> 3 -> 2 -> 1 -> intercept and is exact by the same argument.
    for order in (1..=MAX_ORDER).rev() {
        let keys: Vec<FeatureSet> = values_of
            .keys()
            .filter(|u| u.order() == order)
            .cloned()
            .collect();
        // Nothing deposits into an order-`order` table while this order is being centred (the
        // cascade only feeds order `order - 1`), so each chunk of this order's tables is centred
        // in parallel and its means are then deposited in key order, `p` ascending — the same
        // operations on every f64, in the same order, as centring and depositing one table at a
        // time. Chunked so a level of large dense tables never holds all its means at once.
        for chunk in keys.chunks(PURIFY_CHUNK) {
            let mut level: Vec<(FeatureSet, Tensor)> = Vec::with_capacity(chunk.len());
            for u in chunk {
                let values = values_of.remove(u).ok_or_else(|| PbError::Internal {
                    what: "purify support vanished".into(),
                })?;
                level.push((u.clone(), values));
            }
            let means_of: Vec<Vec<Tensor>> = level
                .par_iter_mut()
                .map(|(u, values)| {
                    (0..u.order())
                        .map(|p| {
                            let r = *u.0.get(p).ok_or_else(|| PbError::Internal {
                                what: "purify axis position escaped support".into(),
                            })?;
                            center_along_with_budget(values, p, w.axis(r)?, budget)
                        })
                        .collect::<Result<Vec<_>, PbError>>()
                })
                .collect::<Result<_, _>>()?;
            for ((u, values), means_u) in level.into_iter().zip(means_of) {
                for (p, means) in means_u.into_iter().enumerate() {
                    if order == 1 {
                        let m = means.at(&[]).ok_or_else(|| PbError::Internal {
                            what: "order-1 mean is not a scalar".into(),
                        })?;
                        f0 += m;
                        continue;
                    }
                    let sub_u = support_without(&u, p);
                    // Ensure the lower-order table exists (lazy cascade allocation).
                    if !values_of.contains_key(&sub_u) {
                        let mut sub_axes = Vec::with_capacity(sub_u.order());
                        let mut sub_extents = Vec::with_capacity(sub_u.order());
                        for sr in &sub_u.0 {
                            sub_axes.push(grids.axis_id(*sr)?);
                            sub_extents.push(grids.cells(*sr)?);
                        }
                        axes_of.insert(sub_u.clone(), sub_axes);
                        values_of.insert(sub_u.clone(), Tensor::try_zeros(sub_extents)?);
                    }
                    let target = values_of.get_mut(&sub_u).ok_or_else(|| PbError::Internal {
                        what: "cascade target vanished".into(),
                    })?;
                    let sub_extents = target.shape();
                    // Dense cascade: one `+=` per cell, the same single add the walk makes.
                    if sub_extents == means.shape() {
                        if let (Some(t), Some(m)) = (target.dense_slice_mut(), means.dense_slice())
                        {
                            for (t, &m) in t.iter_mut().zip(m) {
                                *t += m;
                            }
                            continue;
                        }
                    }
                    walk_extents(&sub_extents, |coord| {
                        let m = means.at(coord).ok_or_else(|| PbError::Internal {
                            what: "cascade mean coord out of range".into(),
                        })?;
                        target.add(coord, m)?;
                        check_sparse_budget(target, budget)
                    })?;
                }
                values_of.insert(u, values);
            }
        }
    }

    // Build the EffectTables (variance cached; `support` is display-only metadata
    // filled separately by `fill_support` — it is not an fANOVA component, so purify,
    // which carries the algebra, stays independent of the data matrix). Each table's
    // variance is its own sequential fold, so the parallel map is bit-identical.
    let entries: Vec<(FeatureSet, Tensor)> = values_of.into_iter().collect();
    let tables: Vec<EffectTable> = entries
        .into_par_iter()
        .map(|(u, values)| {
            let axes = axes_of.get(&u).cloned().ok_or_else(|| PbError::Internal {
                what: "table lost its axes".into(),
            })?;
            let variance = table_variance(&u, &values, w)?;
            let support = if values.is_sparse() {
                Tensor::try_sparse_zeros(values.shape())?
            } else {
                Tensor::try_zeros(values.shape())?
            };
            Ok(EffectTable {
                u,
                axes,
                values,
                support,
                se_band: None,
                variance,
            })
        })
        .collect::<Result<_, PbError>>()?;

    Ok(TableBank {
        f0,
        tables,
        merged_grids: grids.border_grids()?,
        w: w.kind.clone(),
        factored: Vec::new(),
        joint_variance: None,
    })
}

/// `σ²(f_u)` under the product measure `w` (the table is pure, so this is `E_w[f²]`).
fn table_variance(u: &FeatureSet, values: &Tensor, w: &WeightCache) -> Result<f64, PbError> {
    let extents = values.shape();
    if let Some(data) = values.dense_slice() {
        if !extents.is_empty() && u.order() == extents.len() {
            let ws: Vec<&[f64]> = u.0.iter().map(|r| w.axis(*r)).collect::<Result<_, _>>()?;
            return dense_weighted_moments(data, &extents, &ws).map(|(m1, m2)| m2 - m1 * m1);
        }
    }
    let mut m1 = 0.0_f64;
    let mut m2 = 0.0_f64;
    walk_extents(&extents, |coord| {
        let mut wprod = 1.0_f64;
        for (k, &cell) in coord.iter().enumerate() {
            let r = *u.0.get(k).ok_or_else(|| PbError::Internal {
                what: "variance axis escaped support".into(),
            })?;
            let wc = *w.axis(r)?.get(cell).ok_or_else(|| PbError::Internal {
                what: "variance cell escaped axis weights".into(),
            })?;
            wprod *= wc;
        }
        let v = values.at(coord).ok_or_else(|| PbError::Internal {
            what: "variance coord out of range".into(),
        })?;
        m1 += wprod * v;
        m2 += wprod * v * v;
        Ok(())
    })?;
    Ok(m2 - m1 * m1)
}

/// `(Σ wprod·v, Σ wprod·v²)` over a dense row-major table, `wprod` the product of each axis's
/// weight at the cell. Bit-identical to [`table_variance`]'s per-cell walk: cells are visited
/// in the same row-major order and `wprod` is built left to right from `1.0` exactly as the
/// walk builds it (`((1·w₀)·w₁)·…`, the prefix held across the innermost axis).
fn dense_weighted_moments(
    data: &[f64],
    extents: &[usize],
    ws: &[&[f64]],
) -> Result<(f64, f64), PbError> {
    let escaped = || PbError::Internal {
        what: "variance cell escaped axis weights".into(),
    };
    let (&n_last, lead) = extents.split_last().ok_or_else(escaped)?;
    let (w_last, w_lead) = ws.split_last().ok_or_else(escaped)?;
    let w_last = w_last.get(..n_last).ok_or_else(escaped)?;
    if w_lead.len() != lead.len() || n_last == 0 {
        return Err(escaped());
    }
    let mut m1 = 0.0_f64;
    let mut m2 = 0.0_f64;
    let mut idx = vec![0usize; lead.len()];
    for row in data.chunks_exact(n_last) {
        let mut pre = 1.0_f64;
        for (wk, &c) in w_lead.iter().zip(&idx) {
            pre *= *wk.get(c).ok_or_else(escaped)?;
        }
        for (&v, &wl) in row.iter().zip(w_last) {
            let wprod = pre * wl;
            m1 += wprod * v;
            m2 += wprod * v * v;
        }
        // Advance the leading-axes odometer, last digit fastest.
        for (slot, &e) in idx.iter_mut().zip(lead).rev() {
            *slot += 1;
            if *slot < e {
                break;
            }
            *slot = 0;
        }
    }
    Ok((m1, m2))
}

// ===========================================================================
// §08.10 — Factored high-order effects (the over-budget escape hatch).
// ===========================================================================

/// One rank-1 purified contribution to a factored order-`k` effect: the `2^k` corner
/// values of a `step ⊗ … ⊗ step` term plus the `k` per-axis low-side masks over the merged
/// grid — never the dense union cube.
///
/// Both fields are length-carrying (`p.len() == 1 << low.len()`), which is what lets one
/// type serve order 3 and order 4. That is ALSO the wire change the order lift costs: the
/// old `[f64; 8]`/`[Vec<bool>; 3]` were fixed-size arrays, which bincode writes with no
/// length prefix, so a pre-order-lift `.bin` carrying a factored effect cannot be decoded
/// by this build (JSON is unaffected — a fixed array and a `Vec` render identically). The
/// `schema_version` ladder is what makes that a loud refusal instead of a mis-framed read.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct FactoredBox {
    /// Purified corner values; index `= Σ_d s_d << d`, `s_d = 1` on the LOW side of table
    /// axis `d` (matching [`ObliviousTree`] leaf routing). Centered along every axis under
    /// the global per-axis `w`, so it is exactly this term's order-`k` fANOVA component.
    p: Vec<f64>,
    /// Per-table-axis low-side mask over merged cells (`true` ⇒ cell is on the low side).
    low: Vec<Vec<bool>>,
}

/// One per-tree box of a factored order-3 effect in EXPORT form: the per-axis split
/// threshold (merged-grid border value) + missing-left routing, and the 8 purified octant
/// values (index `s0 | s1<<1 | s2<<2`, `s_d = 1` ⇒ the LOW side `x ≤ threshold`). The effect
/// is `Σ_box octant[side(x)]`, so a deployment can score it as a UNION of per-tree CASE
/// expressions without ever materializing the dense merged cube.
///
/// A numeric or single-channel-categorical axis is scalar-encoded (a categorical's own
/// target-statistic value), so its merged cells are linearly ordered and a `bin_le` split
/// always yields a CONTIGUOUS low-side prefix — `thresholds[d]` (the border at that prefix's
/// boundary) exactly and losslessly describes it, for either axis kind. A P1 multi-channel
/// joint categorical axis has no such scalar dimension at all (its cells are a per-level-tuple
/// combinatorial space — design/multichannel-categoricals.md §4.3), and its splits constrain
/// only ONE channel, so the low side is not generally contiguous in the joint cell
/// enumeration: `thresholds[d]` cannot describe it (bug #7). `categorical_low_cells[d]` carries
/// the explicit low-side cell set for that case instead, alongside (not replacing) the
/// existing threshold fields, so an already-working numeric/single-channel-categorical
/// consumer is untouched. A joint axis's cell IDs are the SAME numbering the raw feature's own
/// main-effect table already exports level labels against (Piece B, §4.4), so a consumer can
/// cross-reference that table's `levels` list for human-readable labels without this export
/// duplicating them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactoredBoxExport {
    /// Per-axis split threshold (the realized border value on each of the support's axes),
    /// one entry per support axis. Meaningless (a harmless `0.0` placeholder) at any index
    /// where `categorical_low_cells` is `Some` — check that field first.
    pub thresholds: Vec<f32>,
    /// Per-axis learned missing direction (the reserved missing cell routes low when true).
    pub missing_left: Vec<bool>,
    /// `Some(low_cells)` at axis index `d` iff that axis is a P1 multi-channel joint
    /// categorical: the box's low side is exactly these merged cell ids (no contiguous-prefix
    /// guarantee, unlike a scalar axis) — `thresholds[d]` is meaningless when this is `Some`.
    /// `None` (`#[serde(default)]`, so pre-bug-#7 export JSON still deserializes) for a numeric
    /// or single-channel-categorical axis, where `thresholds[d]` already exactly describes it.
    #[serde(default)]
    pub categorical_low_cells: Vec<Option<Vec<u32>>>,
    /// Corner values, `2^order` of them; expanded threshold terms sum to the purified effect; index `Σ_d s_d << d`, `s_d = 1` ⇒ the LOW
    /// side (either `x ≤ thresholds[d]` or `x`'s cell ∈ `categorical_low_cells[d]`, per the
    /// axis kind above). The name is historical — at order 3 these are octants.
    pub octants: Vec<f64>,
}

/// A factored order-`k` effect for support `u` (`k >= 3`): the sum of per-term purified
/// rank-1 boxes (`f_u = Σ_t p_t`, Lengerich Cor. 2.2 linearity), kept WITHOUT materializing
/// the dense `Π cells` merged cube. Exact prediction ([`eval`](Self::eval), `O(#boxes)`) and
/// exact `w`-weighted variance ([`compute_variance`](Self::compute_variance),
/// `Σ_{t,s}⟨p_t,p_s⟩_w` via a per-axis `2×2` weighted-mass contraction) — proven equal to the
/// dense purified table for any axis-factorized `w` (Uniform / ProductMarginals); only
/// non-product `Joint` would break the factorization (rejected in v1).
///
/// # Why order 4 is factored and not dense
///
/// This is not a size heuristic, it is the only representation that works. A support's
/// dense cube is the product of its raws' GLOBAL merged extents — every border any tree in
/// the model placed on those features, including the main-effect trees that legitimately
/// want full resolution. An order-4 support therefore pays for resolution it never asked
/// for: measured on a 6-feature synthetic at the lifted readability budget, the realized
/// 4-cubes ran 4.8M-23.7M cells against the 2M per-table firewall, and tightening the soft
/// budget did not shrink them monotonically — it just suppressed order 4 entirely. But an
/// order-4 tree realizes only `k_0·k_1·k_2·k_3` REGIONS (16 for a depth-4 tree), so the
/// effect is low-RANK, not sparse: it is small in exactly this representation and in no
/// other. The same argument is why §08.10 already keeps EVERY order-3 effect factored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactoredEffect {
    /// The support (table-dim order = sorted raw ids).
    pub u: FeatureSet,
    /// Merged-grid axes, parallel to the box dims.
    pub axes: Vec<AxisId>,
    /// Per-table-axis merged-cell weights (normalized, `Σ = 1`), shared by all boxes.
    per_axis_w: Vec<Vec<f64>>,
    /// One purified rank-1 box per realized region term.
    boxes: Vec<FactoredBox>,
    /// Cached `σ²(f_u)` under the bank's `w` (the proven `Σ_{t,s}⟨p_t,p_s⟩` closed form),
    /// so [`TableBank::sobol`]/`check_variance_sum` read it like an [`EffectTable`]'s.
    pub variance: f64,
}

/// One affine substitution for an original mask bit. The two rows map the
/// original [high, low] corner pair onto a threshold term's [high, low] pair.
struct NumericMaskExportTerm {
    threshold: f32,
    missing_left: bool,
    weights: [[f64; 2]; 2],
}

/// For finite cell c, m(c) = m(last) + sum_j (m(j)-m(j+1)) I(c <= j).
/// The missing cell is independent: assign its difference to one transition of
/// the required sign, or (for constant finite masks) to two equal thresholds
/// with opposite missing directions. This handles intervals, complements and
/// missing-only regions without changing the existing rating JSON schema.
fn numeric_mask_export_terms(
    axis: &AxisId,
    mask: &[bool],
) -> Result<Vec<NumericMaskExportTerm>, PbError> {
    if mask.len() != axis.borders.len() + 2 {
        return Err(PbError::ShapeMismatch {
            what: "numeric factored mask disagrees with its border grid".into(),
        });
    }
    let finite = &mask[1..];
    let nlow = finite.iter().take_while(|&&low| low).count();
    if nlow > 0 && nlow < finite.len() && finite[nlow..].iter().all(|&low| !low) {
        return Ok(vec![NumericMaskExportTerm {
            threshold: axis.borders[nlow - 1],
            missing_left: mask[0],
            weights: [[1.0, 0.0], [0.0, 1.0]],
        }]);
    }
    let last = f64::from(*finite.last().ok_or_else(|| PbError::ShapeMismatch {
        what: "numeric factored mask has no finite cell".into(),
    })?);
    // Both sides equal the original corner selected on the final finite interval.
    let pivot = axis.borders.first().copied().unwrap_or(0.0);
    let mut terms = vec![NumericMaskExportTerm {
        threshold: pivot,
        missing_left: false,
        weights: [[1.0 - last, last]; 2],
    }];
    let mut missing_delta = f64::from(mask[0]) - last;
    for (j, pair) in finite.windows(2).enumerate() {
        let delta = f64::from(pair[0]) - f64::from(pair[1]);
        if delta != 0.0 {
            let missing_left = missing_delta == delta;
            if missing_left {
                missing_delta = 0.0;
            }
            terms.push(NumericMaskExportTerm {
                threshold: axis.borders[j],
                missing_left,
                weights: [[0.0, 0.0], [-delta, delta]],
            });
        }
    }
    if missing_delta != 0.0 {
        // I_missing = I(x <= pivot, missing-left) - I(x <= pivot, missing-right).
        for (missing_left, delta) in [(true, missing_delta), (false, -missing_delta)] {
            terms.push(NumericMaskExportTerm {
                threshold: pivot,
                missing_left,
                weights: [[0.0, 0.0], [-delta, delta]],
            });
        }
    }
    Ok(terms)
}

/// A [`FactoredEffect`] laid out for scoring many rows: per axis, one bit per box for each
/// merged cell (the box's low-side mask), 64 boxes to a word, and every box's corner values in one
/// contiguous array. [`PackedFactored::eval`] adds the same `p_b[idx_b]` terms in the same box
/// order as [`FactoredEffect::eval`], so the value is bit-identical; it only stops chasing each
/// box's separately allocated masks.
pub(crate) struct PackedFactored {
    raws: SmallVec<[usize; MAX_ORDER]>,
    cells: SmallVec<[usize; MAX_ORDER]>,
    words: usize,
    bits: Vec<Vec<u64>>,
    corners: Vec<f64>,
    n_boxes: usize,
}

impl PackedFactored {
    /// Evaluate at one row's merged cells (indexed by raw feature, as for
    /// [`FactoredEffect::eval`]).
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if `x_cells` lacks one of the effect's raw features;
    /// [`PbError::Internal`] if a cell escapes an axis's masks.
    pub(crate) fn eval(&self, x_cells: &[u32]) -> Result<f64, PbError> {
        let mut rows: SmallVec<[&[u64]; MAX_ORDER]> = SmallVec::new();
        for ((&raw, &n_cells), axis_bits) in self.raws.iter().zip(&self.cells).zip(&self.bits) {
            let c = *x_cells.get(raw).ok_or_else(|| PbError::ShapeMismatch {
                what: format!("x_cells missing raw {raw} for factored eval"),
            })? as usize;
            if c >= n_cells {
                return Err(PbError::Internal {
                    what: "factored eval cell escaped low mask".into(),
                });
            }
            rows.push(
                axis_bits
                    .get(c * self.words..(c + 1) * self.words)
                    .ok_or_else(|| PbError::Internal {
                        what: "factored eval cell escaped low mask".into(),
                    })?,
            );
        }
        let stride = 1usize << self.raws.len();
        let mut acc = 0.0_f64;
        for (b, corners) in self
            .corners
            .chunks_exact(stride)
            .take(self.n_boxes)
            .enumerate()
        {
            let (w, sh) = (b / 64, b % 64);
            let mut idx = 0usize;
            for (d, row) in rows.iter().enumerate() {
                let word = *row.get(w).ok_or_else(|| PbError::Internal {
                    what: "factored eval box escaped its mask words".into(),
                })?;
                idx |= (((word >> sh) & 1) as usize) << d;
            }
            acc += *corners.get(idx).ok_or_else(|| PbError::Internal {
                what: "factored eval box index escaped the box's corner table".into(),
            })?;
        }
        Ok(acc)
    }
}

/// A dense [`EffectTable`] laid out for scoring many rows: each axis's raw feature, optional band
/// map and row-major stride, over the table's own value buffer. [`PackedTable::eval`] reads the
/// same cell [`EffectTable::eval`] reads (so the value is bit-identical), without rebuilding a
/// coordinate vector and re-checking the whole offset on every call.
pub(crate) struct PackedTable<'a> {
    raws: SmallVec<[usize; MAX_ORDER]>,
    dims: SmallVec<[usize; MAX_ORDER]>,
    strides: SmallVec<[usize; MAX_ORDER]>,
    bands: SmallVec<[Option<&'a [u32]>; MAX_ORDER]>,
    data: &'a [f64],
}

impl PackedTable<'_> {
    /// Evaluate at one row's merged cells (indexed by raw feature).
    ///
    /// # Errors
    /// As [`EffectTable::eval`].
    #[inline]
    pub(crate) fn eval(&self, x_cells: &[u32]) -> Result<f64, PbError> {
        let mut off = 0usize;
        for (((&raw, &dim), &stride), band) in self
            .raws
            .iter()
            .zip(&self.dims)
            .zip(&self.strides)
            .zip(&self.bands)
        {
            let cell = *x_cells.get(raw).ok_or_else(|| PbError::ShapeMismatch {
                what: format!("x_cells missing raw feature {raw} for table eval"),
            })?;
            let c = match band {
                None => cell as usize,
                Some(map) => *map.get(cell as usize).ok_or_else(|| PbError::Internal {
                    what: "effect-table merged cell outside its band map".into(),
                })? as usize,
            };
            if c >= dim {
                return Err(PbError::Internal {
                    what: "effect-table coordinate out of range".into(),
                });
            }
            off += c * stride;
        }
        self.data
            .get(off)
            .copied()
            .ok_or_else(|| PbError::Internal {
                what: "effect-table coordinate out of range".into(),
            })
    }
}

impl EffectTable {
    /// This table packed for batch scoring (see [`PackedTable`]), or `None` for a sparse table
    /// (then score through [`Self::eval`]).
    pub(crate) fn packed(&self) -> Option<PackedTable<'_>> {
        let data = self.values.dense_slice()?;
        let dims: SmallVec<[usize; MAX_ORDER]> = self.values.shape().into_iter().collect();
        if dims.len() != self.axes.len() {
            return None;
        }
        let mut strides: SmallVec<[usize; MAX_ORDER]> = SmallVec::from_elem(0, dims.len());
        let mut s = 1usize;
        for (st, &d) in strides.iter_mut().zip(&dims).rev() {
            *st = s;
            s = s.checked_mul(d)?;
        }
        Some(PackedTable {
            raws: self.axes.iter().map(|a| a.raw.0 as usize).collect(),
            dims,
            strides,
            bands: self.axes.iter().map(|a| a.band_of.as_deref()).collect(),
            data,
        })
    }
}

impl FactoredEffect {
    /// This effect packed for batch scoring (see [`PackedFactored`]), or `None` when its boxes
    /// are not uniformly shaped (then score through [`Self::eval`]).
    pub(crate) fn packed(&self) -> Option<PackedFactored> {
        let k = self.axes.len();
        let stride = 1usize.checked_shl(u32::try_from(k).ok()?)?;
        let n_boxes = self.boxes.len();
        let words = n_boxes.div_ceil(64).max(1);
        let mut cells: SmallVec<[usize; MAX_ORDER]> = SmallVec::new();
        let mut bits = Vec::with_capacity(k);
        for d in 0..k {
            let n_cells = self
                .boxes
                .first()
                .and_then(|b| b.low.get(d))
                .map_or(0, Vec::len);
            let mut axis_bits = vec![0u64; n_cells.checked_mul(words)?];
            for (b, bx) in self.boxes.iter().enumerate() {
                let mask = bx.low.get(d)?;
                if mask.len() != n_cells || bx.low.len() != k || bx.p.len() != stride {
                    return None;
                }
                for (c, &low) in mask.iter().enumerate() {
                    if low {
                        *axis_bits.get_mut(c * words + b / 64)? |= 1u64 << (b % 64);
                    }
                }
            }
            cells.push(n_cells);
            bits.push(axis_bits);
        }
        let mut corners = Vec::with_capacity(n_boxes * stride);
        for bx in &self.boxes {
            if bx.p.len() != stride {
                return None;
            }
            corners.extend_from_slice(&bx.p);
        }
        Some(PackedFactored {
            raws: self.axes.iter().map(|a| a.raw.0 as usize).collect(),
            cells,
            words,
            bits,
            corners,
            n_boxes,
        })
    }

    /// Every internal length agrees with the support's order and with the axes' cell counts.
    ///
    /// This is a LOAD gate, not a debug assert. The order lift changed the box encoding from
    /// fixed-size arrays to length-carrying `Vec`s (see `serialize::SCHEMA_VERSION` v4), and
    /// bincode is positional: a document written under the old encoding would be read with a
    /// value where a length prefix belongs. That almost always fails outright inside the
    /// decoder, but "almost always" is not a contract — this check is, and it is what turns a
    /// mis-framed read into a refusal rather than a silently wrong rating table.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if a box's corner count is not `2^order`, if its mask count
    /// is not the order, or if a mask is shorter than its axis's merged-cell count.
    pub fn validate_shape(&self) -> Result<(), PbError> {
        check_factorable_order(&self.u)?;
        let k = self.u.order();
        let mismatch = |what: String| PbError::ShapeMismatch { what };
        if self.axes.len() != k || self.per_axis_w.len() != k {
            return Err(mismatch(format!(
                "factored effect {:?}: {} axes / {} weight vectors for order {k}",
                self.u.0,
                self.axes.len(),
                self.per_axis_w.len()
            )));
        }
        let corners = 1usize << k;
        for (i, b) in self.boxes.iter().enumerate() {
            if b.p.iter().any(|value| !value.is_finite()) {
                return Err(PbError::InvalidInput {
                    what: format!(
                        "factored effect {:?} box {i}: nonfinite coefficient",
                        self.u
                    ),
                });
            }
            if b.p.len() != corners || b.low.len() != k {
                return Err(mismatch(format!(
                    "factored effect {:?} box {i}: {} corners / {} masks, expected \
                     {corners} / {k}",
                    self.u.0,
                    b.p.len(),
                    b.low.len()
                )));
            }
            for (d, (mask, axis)) in b.low.iter().zip(self.axes.iter()).enumerate() {
                if mask.len() < axis.cells as usize {
                    return Err(mismatch(format!(
                        "factored effect {:?} box {i} axis {d}: mask of {} for {} merged cells",
                        self.u.0,
                        mask.len(),
                        axis.cells
                    )));
                }
            }
        }
        for (d, (wd, axis)) in self.per_axis_w.iter().zip(self.axes.iter()).enumerate() {
            let total: f64 = wd.iter().sum();
            if wd.iter().any(|weight| !weight.is_finite() || *weight < 0.0)
                || !total.is_finite()
                || total <= 0.0
            {
                return Err(PbError::InvalidInput {
                    what: format!(
                        "factored effect {:?} axis {d}: invalid reference weights",
                        self.u
                    ),
                });
            }
            if wd.len() < axis.cells as usize {
                return Err(mismatch(format!(
                    "factored effect {:?} axis {d}: {} weights for {} merged cells",
                    self.u.0,
                    wd.len(),
                    axis.cells
                )));
            }
        }
        Ok(())
    }

    /// Build the factored order-`k` effect for support `u` from every tree whose support is
    /// exactly `u`, purifying each tree's box under the global per-axis weights `w`.
    ///
    /// The production pipeline uses [`factored_shed_into_raw`] instead, which does the same
    /// centering but ALSO deposits the shed marginals one order down. This form keeps the
    /// residual only, so it is the reference the shed's bit-identity tests compare against.
    #[cfg(test)]
    fn from_model(
        model: &Model,
        u: &FeatureSet,
        grids: &MergedGrids,
        w: &WeightCache,
    ) -> Result<Self, PbError> {
        check_factorable_order(u)?;
        let u_ids: Vec<FeatureId> = u.0.iter().copied().collect();
        let mut per_axis_w: Vec<Vec<f64>> = Vec::with_capacity(u_ids.len());
        for r in &u_ids {
            per_axis_w.push(w.axis(*r)?.to_vec());
        }
        let axes: Vec<AxisId> = u_ids
            .iter()
            .map(|r| grids.axis_id(*r))
            .collect::<Result<_, _>>()?;
        let mut raw_boxes = Vec::new();
        let mut any_lifted = false;
        for (alpha, tree) in &model.trees {
            if &tree_support(model, tree)? != u {
                continue;
            }
            let tb = build_tree_boxes(model, f64::from(*alpha), tree, &u_ids, grids)?;
            any_lifted |= tb.len() != 1;
            raw_boxes.extend(tb);
        }
        // Merge BEFORE purifying: centering is linear and depends only on the masks, so a
        // merged box purifies to the sum of its parts' purified boxes (see `merge_boxes`).
        let mut boxes = Vec::new();
        for (mut p, low) in merge_boxes(raw_boxes, any_lifted) {
            let (wlo, whi) = side_weights(&low, &per_axis_w)?;
            purify_box(&mut p, &wlo, &whi)?;
            boxes.push(FactoredBox { p, low });
        }
        let mut ft = FactoredEffect {
            u: u.clone(),
            axes,
            per_axis_w,
            boxes,
            variance: 0.0,
        };
        ft.variance = ft.compute_variance()?;
        Ok(ft)
    }

    /// `f_u(x_u)` at a row given its per-raw merged-cell ids — the sum of every box's
    /// octant value. `O(#trees)`, no dense cube.
    pub fn eval(&self, x_cells: &[u32]) -> Result<f64, PbError> {
        // Order <= MAX_ORDER ⇒ SmallVec never spills; avoids a heap alloc per row per
        // factored table.
        let cells: SmallVec<[usize; MAX_ORDER]> = self
            .axes
            .iter()
            .map(|a| {
                x_cells
                    .get(a.raw.0 as usize)
                    .copied()
                    .map(|c| c as usize)
                    .ok_or_else(|| PbError::ShapeMismatch {
                        what: format!("x_cells missing raw {} for factored eval", a.raw.0),
                    })
            })
            .collect::<Result<_, _>>()?;
        let mut acc = 0.0_f64;
        for b in &self.boxes {
            let mut idx = 0usize;
            for (d_bit, (mask, &cell)) in b.low.iter().zip(cells.iter()).enumerate() {
                let s = *mask.get(cell).ok_or_else(|| PbError::Internal {
                    what: "factored eval cell escaped low mask".into(),
                })?;
                idx |= usize::from(s) << d_bit;
            }
            acc += *b.p.get(idx).ok_or_else(|| PbError::Internal {
                what: "factored eval box index escaped the box's corner table".into(),
            })?;
        }
        Ok(acc)
    }

    /// The purified boxes as `(corner values, per-axis low-side masks over merged cells)` — for
    /// crate-internal passes (banding) that work on the boxes without a dense cube.
    pub(crate) fn box_parts(&self) -> Vec<(&[f64], &[Vec<bool>])> {
        self.boxes
            .iter()
            .map(|b| (b.p.as_slice(), b.low.as_slice()))
            .collect()
    }

    /// How many rank-1 region boxes this effect deploys — the DEPLOYED-SIZE unit of a factored
    /// effect, exactly as a dense table's is its cell count. It is the number a filing reader
    /// faces in native storage. [`export_boxes`](Self::export_boxes) can emit several
    /// threshold terms per region box when a numeric mask is not a low-side prefix.
    ///
    /// It is NOT bounded by the tree count. An unlifted tree contributes exactly one box, but a
    /// lifted one contributes up to `Π kᵢ` region boxes, and [`merge_boxes`] folds two of them
    /// together only when their per-axis masks agree EXACTLY — so an effect grown by many deep
    /// trees is large even though each of its trees is trivially small. That asymmetry is what
    /// the deployed-box budget (`crate::prune::apply_box_budget`) exists to price.
    #[must_use]
    pub fn n_boxes(&self) -> usize {
        self.boxes.len()
    }

    /// Boxes in export form (thresholds/cells + corners). Numeric region masks are
    /// expanded into a sum of threshold terms; individual exported terms need not be
    /// pure, but their sum is exactly the native purified effect. Ordinary single-split
    /// boxes retain their original payload and ordering.
    pub fn export_boxes(&self) -> Result<Vec<FactoredBoxExport>, PbError> {
        self.validate_shape()?;
        let k = self.axes.len();
        let mut out = Vec::with_capacity(self.boxes.len());
        for b in &self.boxes {
            let mut expanded = vec![FactoredBoxExport {
                thresholds: vec![0.0; k],
                missing_left: vec![false; k],
                categorical_low_cells: vec![None; k],
                octants: b.p.clone(),
            }];
            for (d, (axis, mask)) in self.axes.iter().zip(&b.low).enumerate() {
                if axis.joint_channels.is_some() {
                    let cells: Vec<u32> = mask
                        .iter()
                        .enumerate()
                        .filter(|(_, low)| **low)
                        .map(|(cell, _)| {
                            u32::try_from(cell).map_err(|_| PbError::Internal {
                                what: "factored box joint cell id exceeded u32".into(),
                            })
                        })
                        .collect::<Result<_, _>>()?;
                    for term in &mut expanded {
                        term.missing_left[d] = mask[0];
                        term.categorical_low_cells[d] = Some(cells.clone());
                    }
                    continue;
                }
                let transforms = numeric_mask_export_terms(axis, mask)?;
                let mut next = Vec::new();
                for term in expanded {
                    for transform in &transforms {
                        let mut value = term.clone();
                        value.thresholds[d] = transform.threshold;
                        value.missing_left[d] = transform.missing_left;
                        // Keep the historical prefix case byte-for-byte, including signed zero.
                        if transform.weights != [[1.0, 0.0], [0.0, 1.0]] {
                            let bit = 1usize << d;
                            for base in (0..term.octants.len()).filter(|i| i & bit == 0) {
                                let high = term.octants[base];
                                let low = term.octants[base | bit];
                                for side in 0..2 {
                                    let w = transform.weights[side];
                                    value.octants[base | (side << d)] = w[0] * high + w[1] * low;
                                }
                            }
                        }
                        next.push(value);
                    }
                }
                expanded = next;
            }
            out.extend(expanded);
        }
        Ok(out)
    }

    /// `σ²(f_u) = ⟨Σ_t p_t, Σ_t p_t⟩_w = Σ_{t,s}⟨p_t,p_s⟩_w`. Each pairwise inner product
    /// factorizes as a product over axes of `2×2` weighted-mass matrices (the joint low/high
    /// cell membership of the two trees), so no joint cube is ever built.
    ///
    /// The `2×2` joint mass on an axis depends ONLY on the pair of per-axis low-masks, and the
    /// support-`u` trees realize just a few distinct masks per axis (one split threshold each).
    /// So the masks are deduplicated per axis and each distinct mask-pair's joint mass is
    /// precomputed ONCE — turning the per-pair `O(merged_cells)` mass rescan into an `O(1)` table
    /// read, so the `O(n_boxes²)` sweep no longer re-walks the grid (the mass rescan was ~90% of
    /// deploy-time `explain` on the bagged soup). Byte-identical to the naive all-pairs form: each
    /// joint mass is the SAME per-cell weighted sum in the SAME cell order, and the pairwise
    /// accumulation order (diagonal, then doubled upper triangle) is unchanged.
    fn compute_variance(&self) -> Result<f64, PbError> {
        let n = self.boxes.len();
        if n == 0 {
            return Ok(0.0);
        }
        // Per axis: assign each box a distinct-mask id, and precompute the joint weighted-mass
        // 2×2 for every ORDERED pair of distinct masks (`mass_tab[d][a*k + b]`, a = i-side).
        let k_axes = self.per_axis_w.len();
        let mut mask_id: Vec<Vec<u32>> = vec![vec![0; n]; k_axes];
        let mut mass_tab: Vec<Vec<[[f64; 2]; 2]>> = vec![Vec::new(); k_axes];
        let mut kdim = vec![0usize; k_axes];
        for d in 0..k_axes {
            let wd = &self.per_axis_w[d];
            let mut uniq: Vec<&[bool]> = Vec::new();
            let mut lut: std::collections::HashMap<&[bool], u32> = std::collections::HashMap::new();
            for (i, b) in self.boxes.iter().enumerate() {
                let mask: &[bool] = &b.low[d];
                let id = *lut.entry(mask).or_insert_with(|| {
                    let id = uniq.len() as u32;
                    uniq.push(mask);
                    id
                });
                mask_id[d][i] = id;
            }
            let k = uniq.len();
            kdim[d] = k;
            // Same per-cell weighted sum, in the same cell order, as the naive `inner` mass build.
            let mut tab = vec![[[0.0_f64; 2]; 2]; k * k];
            for (a, &ma) in uniq.iter().enumerate() {
                for (b, &mb) in uniq.iter().enumerate() {
                    let m = &mut tab[a * k + b];
                    for ((&wc, &la), &lb) in wd.iter().zip(ma.iter()).zip(mb.iter()) {
                        m[usize::from(la)][usize::from(lb)] += wc;
                    }
                }
            }
            mass_tab[d] = tab;
        }
        let mut var = 0.0_f64;
        for i in 0..n {
            let bi = &self.boxes[i];
            var += self.inner_tabled(bi, bi, i, i, &mask_id, &mass_tab, &kdim);
            for j in (i + 1)..n {
                let bj = &self.boxes[j];
                var += 2.0 * self.inner_tabled(bi, bj, i, j, &mask_id, &mass_tab, &kdim);
            }
        }
        Ok(var)
    }

    /// `⟨p_i, p_j⟩_w` from the precomputed per-axis joint-mass tables (see [`Self::compute_variance`]).
    /// The `2×2×2` octant sum is byte-identical to the naive form — same term order, same
    /// `wgt != 0` skip, same left-associated multiply `((m0·m1)·m2)·p_i·p_j` — only the mass
    /// matrices are `O(1)` table reads. The per-axis `continue` on a zero partial product skips
    /// only terms whose `wgt` is exactly `0.0` (a zero mass times finite factors, so `0.0 · x ==
    /// 0.0`), which the naive `wgt != 0.0` guard also skips — so no surviving term is reordered.
    fn inner_tabled(
        &self,
        bi: &FactoredBox,
        bj: &FactoredBox,
        i: usize,
        j: usize,
        mask_id: &[Vec<u32>],
        mass_tab: &[Vec<[[f64; 2]; 2]>],
        kdim: &[usize],
    ) -> f64 {
        let k = self.per_axis_w.len();
        // The per-axis joint-mass matrices for THIS box pair, axis order preserved.
        let mut m: SmallVec<[&[[f64; 2]; 2]; MAX_ORDER]> = SmallVec::with_capacity(k);
        for d in 0..k {
            let (Some(ids), Some(tab), Some(&kd)) = (mask_id.get(d), mass_tab.get(d), kdim.get(d))
            else {
                return 0.0;
            };
            let (Some(&ai), Some(&aj)) = (ids.get(i), ids.get(j)) else {
                return 0.0;
            };
            match tab.get(ai as usize * kd + aj as usize) {
                Some(cell) => m.push(cell),
                None => return 0.0,
            }
        }
        let mut s = 0.0_f64;
        inner_walk(&m, &bi.p, &bj.p, 0, 1.0, 0, 0, &mut s);
        s
    }
}

/// The `2^{2k}` term sweep behind [`FactoredEffect::inner_tabled`], one axis per recursion
/// level: at axis `d` multiply the running partial by `m[d][a_i][a_j]`, and at the last axis
/// accumulate `wgt · p_i[idx_i] · p_j[idx_j]`.
///
/// Written recursively so it serves any arity, but deliberately shaped so that at `k = 3` it
/// visits terms in EXACTLY the order the previous hand-unrolled `a0i,a0j,a1i,a1j,a2i,a2j`
/// nest did, with the same left-associated product `((m0·m1)·m2)` and the same early-exit on
/// an exactly-zero partial. That matters because the resulting variance is serialized inside
/// the tables document, so a reassociation would move the bytes of every existing model
/// carrying an order-3 effect. The zero skip is safe rather than merely cheap: a zero partial
/// times finite factors is exactly `0.0`, which the terminal `wgt != 0.0` guard drops anyway.
fn inner_walk(
    m: &[&[[f64; 2]; 2]],
    pi: &[f64],
    pj: &[f64],
    d: usize,
    partial: f64,
    idx_i: usize,
    idx_j: usize,
    acc: &mut f64,
) {
    let Some(md) = m.get(d) else {
        return;
    };
    let last = d + 1 == m.len();
    for ai in 0..2usize {
        for aj in 0..2usize {
            let w = partial * md[ai][aj];
            let ii = idx_i | (ai << d);
            let jj = idx_j | (aj << d);
            if last {
                if w != 0.0 {
                    if let (Some(&vi), Some(&vj)) = (pi.get(ii), pj.get(jj)) {
                        *acc += w * vi * vj;
                    }
                }
            } else {
                if w == 0.0 {
                    continue;
                }
                inner_walk(m, pi, pj, d + 1, w, ii, jj, acc);
            }
        }
    }
}

/// Refuse a support that has no business being factored. The factored representation exists
/// for supports whose dense cube is a product of many global merged extents; order 1 and 2
/// are bounded by `255` and `255^2` and stay dense, where a human reads them as a table.
fn check_factorable_order(u: &FeatureSet) -> Result<(), PbError> {
    if !(LEGACY_MAX_ORDER..=MAX_ORDER).contains(&u.order()) {
        return Err(PbError::Internal {
            what: format!(
                "a factored effect needs an order in {LEGACY_MAX_ORDER}..={MAX_ORDER}, got {}",
                u.order()
            ),
        });
    }
    Ok(())
}

/// Per-axis low/high total `w`-mass for one tree's masks (`whi = 1 − wlo` since `w` sums
/// to 1 per axis, but both are summed explicitly for numerical symmetry).
fn side_weights(
    low: &[Vec<bool>],
    per_axis_w: &[Vec<f64>],
) -> Result<(Vec<f64>, Vec<f64>), PbError> {
    // The arity used to be carried by the type (`[_; 3]` against `[_; 3]`). With `Vec`s it
    // has to be checked: a short `per_axis_w` would leave the trailing axes at
    // `wlo = whi = 0`, and `center_box_capturing` — whose own shape check only compares
    // `p.len()` against `wlo.len()` — would then return a box UNCENTERED on those axes with
    // no error at all. Fail loudly instead.
    if low.len() != per_axis_w.len() {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "side_weights: {} masks against {} weight vectors",
                low.len(),
                per_axis_w.len()
            ),
        });
    }
    let mut wlo = vec![0.0_f64; low.len()];
    let mut whi = vec![0.0_f64; low.len()];
    for (((slot_lo, slot_hi), wd), mask) in wlo
        .iter_mut()
        .zip(whi.iter_mut())
        .zip(per_axis_w.iter())
        .zip(low.iter())
    {
        for (&wc, &is_low) in wd.iter().zip(mask.iter()) {
            if is_low {
                *slot_lo += wc;
            } else {
                *slot_hi += wc;
            }
        }
    }
    Ok((wlo, whi))
}

/// One rank-1 region box of a factored order-`k` effect, before it is purified: the `2^k`
/// corner values of a `step ⊗ … ⊗ step` term paired with the `k` per-axis indicator masks
/// over merged cells. A lifted (depth > order) tree decomposes into a SUM of these, one per
/// realized region tuple — see [`build_tree_boxes`].
type RankOneBox = (Vec<f64>, Vec<Vec<bool>>);

/// One support-`u` tree's contribution as a LIST of rank-1 octant boxes (table-dim side
/// order, `s_d = 1` ⇒ low) plus each box's per-axis low-side masks over the merged grid.
///
/// # Why a list, and why still `FactoredBox`
///
/// A depth-3 order-3 tree has exactly one split per support axis, so its contribution IS a
/// single `2×2×2` box — the historical case, and the FIRST branch below returns exactly the
/// box the pre-P-D4 `build_tree_box` did, byte for byte.
///
/// A depth-LIFTED tree reuses a raw feature across levels, so its contribution is a
/// `k₀×k₁×k₂` product box (a depth-6 `(2,2,2)` tree partitions its three axes into 3 regions
/// each = 27 cells), which no single rank-1 box can hold. The memo's plan (P-D4) was to
/// generalize `FactoredBox` itself. That is not available: `FactoredBox` is SERIALIZED inside
/// the tables document (`TableBank.factored`), its field order is `p` then `low`, and a
/// general box needs its shape BEFORE its payload — so no in-band discriminator exists and
/// any field-type change moves the bytes of every existing model carrying an order-3 effect
/// (2 of the 40 fingerprint-battery models do).
///
/// So instead of widening the box, the tree is DECOMPOSED into rank-1 boxes: one per region
/// tuple `(r₀, r₁, r₂)`, whose per-axis mask is that region's indicator over merged cells and
/// whose only non-zero octant is the all-low corner carrying the tree's value there. `low[d]`
/// was always an arbitrary `Vec<bool>` over merged cells — nothing required it to come from a
/// single split — so this is a legal, exact use of the existing type.
///
/// It is exact because everything downstream is LINEAR in the boxes: `f_u = Σ_t p_t`
/// (Lengerich Cor. 2.2), `purify_box`/`center_box_capturing` centre each box independently,
/// `factored_shed_into_raw` deposits per-box marginals additively, and `compute_variance` is
/// the bilinear `Σ_{i,j}⟨p_i,p_j⟩`. Splitting one tree's box into a sum of boxes changes none
/// of those sums. The five I2 gates are therefore INHERITED rather than re-derived — which is
/// the whole reason this branch was chosen over widening the type.
///
/// Cost: `k₀·k₁·k₂` boxes per lifted tree (≤ 28 in practice — a `(2,2,2)` depth-6 tree gives
/// 27, a `(6,0,0)` one gives 7·2·2). `compute_variance` is `O(n_boxes²)`, so the caller
/// MERGES boxes sharing a mask triple first (`merge_boxes`); lifted trees reuse borders
/// heavily — which the §6.4 new-border penalty actively encourages — so that collapses hard.
fn build_tree_boxes(
    model: &Model,
    alpha: f64,
    tree: &crate::engine::ObliviousTree,
    u_ids: &[FeatureId],
    grids: &MergedGrids,
) -> Result<Vec<RankOneBox>, PbError> {
    // Per support position: the tree levels that test it, and each level's cell mask.
    let k = u_ids.len();
    let mut level_masks: Vec<Vec<(usize, Vec<bool>)>> = vec![Vec::new(); k];
    for (level, split) in tree.splits.iter().enumerate() {
        let prov = model
            .provenance
            .get(split.axis as usize)
            .ok_or_else(|| PbError::Internal {
                what: "factored split axis absent from provenance".into(),
            })?;
        let d = u_ids
            .iter()
            .position(|r| *r == prov.raw)
            .ok_or_else(|| PbError::Internal {
                what: "factored split raw absent from support".into(),
            })?;
        let ma = grids.axis(prov.raw)?;
        let cells = ma.cells();
        let mut mask = Vec::with_capacity(cells);
        for c in 0..cells {
            // Axis-aware for the same reason as `leaf_index_for_tuple` — see its comment.
            // A P1 multi-channel raw feature resolves the specific CHANNEL this split tests.
            let rep_bin = ma.rep_model_bin_for_axis(c, split.axis as usize)?;
            mask.push(low_bit(rep_bin, split.bin_le, split.missing_left));
        }
        level_masks
            .get_mut(d)
            .ok_or_else(|| PbError::Internal {
                what: "factored mask slot escaped the support".into(),
            })?
            .push((level, mask));
    }
    for (d, lm) in level_masks.iter().enumerate() {
        if lm.is_empty() {
            return Err(PbError::Internal {
                what: format!("factored support axis {d} has no split in this tree"),
            });
        }
    }

    // --- the historical rank-1 case, reproduced EXACTLY (one split per support axis) ---
    if tree.splits.len() == k {
        let mut low: Vec<Vec<bool>> = vec![Vec::new(); k];
        let mut level_of: Vec<usize> = vec![0; k];
        for (d, lm) in level_masks.iter_mut().enumerate() {
            let (level, mask) = lm.pop().ok_or_else(|| PbError::Internal {
                what: "rank-1 factored axis lost its split".into(),
            })?;
            *low.get_mut(d).ok_or_else(|| PbError::Internal {
                what: "factored mask slot escaped the support".into(),
            })? = mask;
            *level_of.get_mut(d).ok_or_else(|| PbError::Internal {
                what: "factored level slot escaped the support".into(),
            })? = level;
        }
        // One corner per side tuple `s`, enumerated as the odometer `s_0` outermost — the
        // same visit order the previous hand-unrolled `s0/s1/s2` nest used at k = 3. Every
        // corner is ASSIGNED (never accumulated), so the order is a readability property
        // here rather than a bit-identity one, but keeping it costs nothing.
        let mut p = vec![0.0_f64; 1usize << k];
        for corner in 0..(1usize << k) {
            // `corner`'s bit d is `s_d`; the corresponding leaf sets bit `level_of[d]`.
            let mut leaf_idx = 0usize;
            for (d, &lv) in level_of.iter().enumerate() {
                leaf_idx |= ((corner >> d) & 1) << lv;
            }
            let leaf = *tree.leaves.get(leaf_idx).ok_or_else(|| PbError::Internal {
                what: "factored leaf index escaped the tree's leaf table".into(),
            })?;
            *p.get_mut(corner).ok_or_else(|| PbError::Internal {
                what: "factored box index escaped the corner table".into(),
            })? = alpha * f64::from(leaf);
        }
        return Ok(vec![(p, low)]);
    }

    // --- the lifted case: one rank-1 box per realized region tuple ---
    //
    // A cell's REGION on axis `d` is the bit pattern of its low-bits across the tree levels
    // that test `d`. Patterns are deduped in first-appearance order over cells, so only
    // REACHABLE regions are materialized (a depth-6 (2,2,2) tree realizes 3 of the 4 patterns
    // per axis — the nested-threshold phantom never appears on any cell) and the result is
    // deterministic. No ordering of cells is assumed, so this is correct for a joint
    // categorical axis too, where the low side is not a contiguous prefix.
    let mut region_of: Vec<Vec<usize>> = vec![Vec::new(); k];
    let mut region_pattern: Vec<Vec<u64>> = vec![Vec::new(); k];
    for d in 0..k {
        let lm = level_masks.get(d).ok_or_else(|| PbError::Internal {
            what: "region axis escaped the support".into(),
        })?;
        let cells = lm
            .first()
            .map(|(_, m)| m.len())
            .ok_or_else(|| PbError::Internal {
                what: "region axis has no mask".into(),
            })?;
        let mut of = Vec::with_capacity(cells);
        let mut pats: Vec<u64> = Vec::new();
        for c in 0..cells {
            let mut pat = 0u64;
            for (j, (_, mask)) in lm.iter().enumerate() {
                if *mask.get(c).ok_or_else(|| PbError::Internal {
                    what: "region mask cell escaped".into(),
                })? {
                    pat |= 1u64 << j;
                }
            }
            let r = match pats.iter().position(|p| *p == pat) {
                Some(r) => r,
                None => {
                    pats.push(pat);
                    pats.len() - 1
                }
            };
            of.push(r);
        }
        *region_of.get_mut(d).ok_or_else(|| PbError::Internal {
            what: "region_of slot escaped".into(),
        })? = of;
        *region_pattern.get_mut(d).ok_or_else(|| PbError::Internal {
            what: "region_pattern slot escaped".into(),
        })? = pats;
    }

    let kd: Vec<usize> = region_pattern.iter().map(Vec::len).collect();
    let n_tuples = kd
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| PbError::Internal {
            what: "lifted region tuple count overflowed usize".into(),
        })?;
    let mut out = Vec::with_capacity(n_tuples);
    // Odometer over region tuples with axis 0 as the OUTERMOST (slowest) digit — the same
    // visit order the previous `r0/r1/r2` nest produced at k = 3. `merge_boxes` keeps
    // first-appearance order, so this order is what fixes the downstream summation order.
    let mut r = vec![0usize; k];
    for _ in 0..n_tuples {
        // The tree's leaf for this region tuple: OR each axis's per-level bit into that
        // level's position. This is the SAME fold `leaf_index_for_tuple` does, just
        // evaluated once per region instead of once per merged cell.
        let mut leaf_idx = 0usize;
        for (d, &rd) in r.iter().enumerate() {
            let pat = *region_pattern
                .get(d)
                .and_then(|p| p.get(rd))
                .ok_or_else(|| PbError::Internal {
                    what: "region pattern escaped".into(),
                })?;
            let lm = level_masks.get(d).ok_or_else(|| PbError::Internal {
                what: "region level masks escaped".into(),
            })?;
            for (j, (level, _)) in lm.iter().enumerate() {
                if (pat >> j) & 1 == 1 {
                    leaf_idx |= 1usize << level;
                }
            }
        }
        let leaf = *tree.leaves.get(leaf_idx).ok_or_else(|| PbError::Internal {
            what: "lifted factored leaf index escaped the tree's leaf table".into(),
        })?;
        let value = alpha * f64::from(leaf);
        // An all-zero box contributes nothing to any downstream sum.
        if value != 0.0 {
            let mut low: Vec<Vec<bool>> = vec![Vec::new(); k];
            for (d, &rd) in r.iter().enumerate() {
                let of = region_of.get(d).ok_or_else(|| PbError::Internal {
                    what: "region_of escaped".into(),
                })?;
                *low.get_mut(d).ok_or_else(|| PbError::Internal {
                    what: "lifted mask slot escaped the support".into(),
                })? = of.iter().map(|g| *g == rd).collect();
            }
            let mut p = vec![0.0_f64; 1usize << k];
            // Only the all-low corner (every s_d = 1) is inside this region.
            let all_low = (1usize << k) - 1;
            *p.get_mut(all_low).ok_or_else(|| PbError::Internal {
                what: "lifted box corner escaped the corner table".into(),
            })? = value;
            out.push((p, low));
        }
        // Advance the odometer, LAST digit fastest.
        for d in (0..k).rev() {
            let (Some(slot), Some(&limit)) = (r.get_mut(d), kd.get(d)) else {
                break;
            };
            *slot += 1;
            if *slot < limit {
                break;
            }
            *slot = 0;
        }
    }
    Ok(out)
}

/// Sum boxes that share an identical mask triple.
///
/// Exact by linearity (`f_u = Σ_t p_t`), and the reason the decomposition above is
/// affordable: `compute_variance` is `O(n_boxes²)`, and a lifted tree contributes up to 28
/// boxes. Lifted trees on one support reuse borders heavily — which the §6.4 new-border
/// penalty actively encourages — so distinct mask triples are far fewer than
/// `n_trees · k₀k₁k₂`. First-appearance order is preserved, so the result is deterministic
/// and the pairwise accumulation order stays fixed.
fn merge_boxes(boxes: Vec<RankOneBox>, any_lifted: bool) -> Vec<RankOneBox> {
    // A support whose every tree is rank-1 keeps its boxes exactly as built, one per tree in
    // tree order. Merging there would fold two DIFFERENT trees' boxes that happen to share a
    // mask triple into one addend, changing the summation order of the purify/shed/variance
    // sums by a last bit — and those sums are serialized inside the tables document, so it
    // would move the bytes of every existing depth-3 model carrying an order-3 effect (2 of
    // the 40 fingerprint-battery models do). Merging is only ever needed because a LIFTED
    // tree contributes many boxes, so it is only ever applied when one did.
    if !any_lifted {
        return boxes;
    }
    // The mask triples are by far the largest thing here (three `Vec<bool>` over merged
    // cells, ~750 bytes each, times up to 28 boxes per lifted tree), so they are stored
    // EXACTLY ONCE — in `out`. The dedup index keys on a fingerprint and verifies the
    // candidate against `out` before merging, rather than owning a second copy of every
    // triple as a `HashMap` key. First-appearance order is unchanged, so the pairwise
    // accumulation order downstream is unchanged too.
    let mut out: Vec<RankOneBox> = Vec::new();
    let mut index: std::collections::HashMap<u64, smallvec::SmallVec<[usize; 1]>> =
        std::collections::HashMap::new();
    for (p, low) in boxes {
        let fp = mask_fingerprint(&low);
        let bucket = index.entry(fp).or_default();
        let hit = bucket
            .iter()
            .copied()
            .find(|&i| out.get(i).is_some_and(|(_, l)| *l == low));
        match hit {
            Some(i) => {
                if let Some(slot) = out.get_mut(i) {
                    for (a, b) in slot.0.iter_mut().zip(p.iter()) {
                        *a += *b;
                    }
                }
            }
            None => {
                bucket.push(out.len());
                out.push((p, low));
            }
        }
    }
    out
}

/// A 64-bit fingerprint of a box's mask tuple, for [`merge_boxes`]'s dedup index. Only ever
/// a bucket key — equality is always confirmed against the stored masks, so a collision
/// costs one comparison and can never merge two different boxes.
fn mask_fingerprint(low: &[Vec<bool>]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for mask in low {
        mask.hash(&mut h);
    }
    h.finish()
}

/// Center a `2×2×2` box along each table axis IN CASCADE ORDER (axis 0, then 1 on the
/// 0-centered box, then 2), returning the per-axis marginals shed downward: `marg[p]` is the
/// `2×2` weighted mean over the OTHER two axes (increasing position order) at step `p` —
/// exactly what the dense [`center_along`] deposits into the order-2 table `u\{p}`. Leaves
/// `p` as the triple-centered residual (the tree's order-3 fANOVA component).
fn center_box_capturing(p: &mut [f64], wlo: &[f64], whi: &[f64]) -> Result<Vec<Vec<f64>>, PbError> {
    let k = wlo.len();
    if whi.len() != k || p.len() != 1usize << k {
        return Err(PbError::Internal {
            what: format!(
                "center_box_capturing shape mismatch: {} corners for {k} axes",
                p.len()
            ),
        });
    }
    let sub_corners = 1usize << k.saturating_sub(1);
    let mut marg: Vec<Vec<f64>> = Vec::with_capacity(k);
    for d in 0..k {
        let (Some(&wl), Some(&wh)) = (wlo.get(d), whi.get(d)) else {
            return Err(PbError::Internal {
                what: "center axis weight escaped the support".into(),
            });
        };
        // The k-1 axes other than `d`, in increasing position order — the axis order of the
        // sub-support `u\{d}` this marginal is shed into.
        let others: SmallVec<[usize; MAX_ORDER]> = (0..k).filter(|&o| o != d).collect();
        let mut md = vec![0.0_f64; sub_corners];
        for (b, slot) in md.iter_mut().enumerate() {
            // `b` is the SUB-BOX corner index: bit t is the side of `others[t]`. Scatter it
            // back into this box's own corner numbering.
            let mut base = 0usize;
            for (t, &o) in others.iter().enumerate() {
                base |= ((b >> t) & 1) << o;
            }
            let i_hi = base; // s_d = 0 (high)
            let i_lo = base | (1 << d); // s_d = 1 (low)
            let hi = *p.get(i_hi).ok_or_else(|| PbError::Internal {
                what: "center hi index escaped the corner table".into(),
            })?;
            let lo = *p.get(i_lo).ok_or_else(|| PbError::Internal {
                what: "center lo index escaped the corner table".into(),
            })?;
            let mean = wh * hi + wl * lo;
            *slot = mean;
            *p.get_mut(i_hi).ok_or_else(|| PbError::Internal {
                what: "center hi write escaped the corner table".into(),
            })? -= mean;
            *p.get_mut(i_lo).ok_or_else(|| PbError::Internal {
                what: "center lo write escaped the corner table".into(),
            })? -= mean;
        }
        marg.push(md);
    }
    Ok(marg)
}

/// Center a rank-1 box along every axis, keeping only the residual; see
/// [`center_box_capturing`]. Test-only — the production shed keeps the marginals.
#[cfg(test)]
fn purify_box(p: &mut [f64], wlo: &[f64], whi: &[f64]) -> Result<(), PbError> {
    center_box_capturing(p, wlo, whi).map(|_| ())
}

/// Run the purification cascade for a factored support `u` WITHOUT the dense cube: per
/// realized region term, center its box along every axis in cascade order and shed each
/// axis's marginal one order down. By linearity (`Σ_t`) this deposits exactly what the dense
/// [`center_along`] would, so the rest of the cascade runs on the augmented lower structures
/// unchanged. Removes `u` from `raw_tables`; returns the fully-centered residual as a
/// [`FactoredEffect`].
///
/// # Where the shed mass lands
///
/// At order 3 the sub-supports are PAIRS, whose dense cubes are bounded by `255^2` and are
/// what a human actually reads, so the marginal is broadcast into the dense order-2 raw
/// table `u\{d}` (lazily created) exactly as before.
///
/// Above order 3 the sub-supports are themselves factored — densifying them is precisely the
/// blow-up §08.10 exists to avoid — so the marginal is handed down as a rank-1 BOX instead,
/// through `deposits`. That is not an approximation and not a special case: the marginal of a
/// `step ⊗ … ⊗ step` term along one axis IS a `step ⊗ … ⊗ step` term on the remaining axes,
/// with the same masks and `2^(k-1)` corner values. The caller drives supports in DESCENDING
/// order, so an order-3 effect has already collected every order-4 deposit before it centers.
///
/// `inherited` are the boxes shed down onto `u` from order `k+1`; they join `u`'s own trees'
/// boxes before the merge, which is exact for the same linearity reason.
#[cfg(test)]
fn factored_shed_into_raw(
    model: &Model,
    u: &FeatureSet,
    grids: &MergedGrids,
    w: &WeightCache,
    raw_tables: &mut BTreeMap<FeatureSet, RawTable>,
    inherited: Vec<RankOneBox>,
    deposits: &mut BTreeMap<FeatureSet, Vec<RankOneBox>>,
) -> Result<FactoredEffect, PbError> {
    let trees: Vec<usize> = tree_supports(model)?
        .iter()
        .enumerate()
        .filter(|(_, s)| *s == u)
        .map(|(i, _)| i)
        .collect();
    factored_shed_into_raw_for(model, u, &trees, grids, w, raw_tables, inherited, deposits)
}

/// [`factored_shed_into_raw`] given the indices of the trees whose support is `u`, in tree
/// order (the bank build indexes them once, instead of every shed rescanning the ensemble).
fn factored_shed_into_raw_for(
    model: &Model,
    u: &FeatureSet,
    trees: &[usize],
    grids: &MergedGrids,
    w: &WeightCache,
    raw_tables: &mut BTreeMap<FeatureSet, RawTable>,
    inherited: Vec<RankOneBox>,
    deposits: &mut BTreeMap<FeatureSet, Vec<RankOneBox>>,
) -> Result<FactoredEffect, PbError> {
    check_factorable_order(u)?;
    let k = u.order();
    let u_ids: Vec<FeatureId> = u.0.iter().copied().collect();
    let mut per_axis_w: Vec<Vec<f64>> = Vec::with_capacity(k);
    for r in &u_ids {
        per_axis_w.push(w.axis(*r)?.to_vec());
    }
    let axes: Vec<AxisId> = u_ids
        .iter()
        .map(|r| grids.axis_id(*r))
        .collect::<Result<_, _>>()?;
    let prof = std::env::var_os("TBOOST_PROFILE").is_some();
    let (mut scan_ns, mut build_ns) = (0u128, 0u128);
    let n_trees = trees.len();
    // Inherited boxes come first, in the caller's deterministic support order, so the merge
    // and every downstream sum see a fixed sequence. `any_lifted` is set when any inherited
    // box is present too: a support that exists only because a higher order shed onto it has
    // no tree of its own, and merging is what keeps its box count from compounding.
    let mut any_lifted = !inherited.is_empty();
    let mut raw_boxes = inherited;
    for &i in trees {
        // Timers gated on `prof` so a non-profiled run pays no `Instant` reads in this
        // per-tree loop (only an `Option` check).
        let t_scan = prof.then(std::time::Instant::now);
        let (alpha, tree) = model.trees.get(i).ok_or_else(|| PbError::Internal {
            what: format!("factored tree index {i} escaped the ensemble"),
        })?;
        // Guard the index's promise (cheap next to the box build).
        if &tree_support(model, tree)? != u {
            return Err(PbError::Internal {
                what: format!("tree {i} indexed under {u:?} realizes another support"),
            });
        }
        if let Some(t) = t_scan {
            scan_ns += t.elapsed().as_nanos();
        }
        let t_build = prof.then(std::time::Instant::now);
        let tb = build_tree_boxes(model, f64::from(*alpha), tree, &u_ids, grids)?;
        any_lifted |= tb.len() != 1;
        raw_boxes.extend(tb);
        if let Some(t) = t_build {
            build_ns += t.elapsed().as_nanos();
        }
    }
    let ft = shed_boxes_into_raw(
        u, per_axis_w, axes, raw_boxes, any_lifted, grids, raw_tables, deposits,
    )?;
    if prof {
        eprintln!(
            "[shed] u={:?} trees={n_trees} boxes={} | scan={:.1}ms build={:.1}ms",
            u.0,
            ft.boxes.len(),
            scan_ns as f64 / 1e6,
            build_ns as f64 / 1e6,
        );
    }
    Ok(ft)
}

/// The measure-dependent half of [`factored_shed_into_raw`]: merge `raw_boxes`, centre each
/// box under `per_axis_w`, and deposit the captured marginals one order down (densely at
/// order 3, as boxes above). Split out so that [`TableBank::recompute_under`] can re-shed an
/// EXISTING factored effect's boxes under a different axis-factorized measure without the
/// model: a box already centred under one product measure is still just a rank-1 function of
/// its `k` features, so centring it again under another measure captures exactly the
/// marginals the new measure assigns to lower orders, and the residual plus the deposits
/// still sum to the box. `any_lifted` follows [`merge_boxes`]'s contract.
#[allow(clippy::too_many_arguments)] // JUSTIFIED: one call site per driver; the tuple would only rename the arguments.
fn shed_boxes_into_raw(
    u: &FeatureSet,
    per_axis_w: Vec<Vec<f64>>,
    axes: Vec<AxisId>,
    raw_boxes: Vec<RankOneBox>,
    any_lifted: bool,
    grids: &MergedGrids,
    raw_tables: &mut BTreeMap<FeatureSet, RawTable>,
    deposits: &mut BTreeMap<FeatureSet, Vec<RankOneBox>>,
) -> Result<FactoredEffect, PbError> {
    let k = u.order();
    // Merge identical-mask boxes across the whole support before centring and shedding —
    // exact by linearity, and what keeps the lifted decomposition's `O(n_boxes²)` variance
    // sweep affordable (see `merge_boxes`).
    let mut boxes = Vec::new();
    for (mut p, low) in merge_boxes(raw_boxes, any_lifted) {
        let (wlo, whi) = side_weights(&low, &per_axis_w)?;
        let marg = center_box_capturing(&mut p, &wlo, &whi)?;
        for pos in 0..k {
            let m_pos = marg.get(pos).ok_or_else(|| PbError::Internal {
                what: "shed marginal axis escaped the support".into(),
            })?;
            let sub_u = support_without(u, pos);
            // The sub-support's axes are `u`'s minus position `pos`, in the same
            // increasing-position order `center_box_capturing` indexed `m_pos`'s corners by
            // and the same order `support_without` leaves `sub_u` in. Three orderings, one
            // convention.
            if k > LEGACY_MAX_ORDER {
                // Hand the marginal down as a box (see the doc comment). No dense cube is
                // created at any point on this branch. This is the ONLY branch that needs
                // the masks OWNED, so the clone lives here rather than above it — the
                // order-3 shed is the hot path (roughly 90% of deploy-time `explain` on a
                // bagged soup) and cloning k*(k-1) merged-cell masks per box to feed two
                // borrows would be a real cost there.
                let sub_low: Vec<Vec<bool>> = low
                    .iter()
                    .enumerate()
                    .filter(|&(d, _)| d != pos)
                    .map(|(_, mask)| mask.clone())
                    .collect();
                deposits
                    .entry(sub_u)
                    .or_default()
                    .push((m_pos.clone(), sub_low));
                continue;
            }
            // At order 3 the two remaining axes are whichever two are not `pos`, in
            // increasing order.
            //
            // FAIL CLOSED rather than fall through. The `_` arm below is correct ONLY for
            // `k == 3` (and `pos == 2`), and it is reachable only because the `k >
            // LEGACY_MAX_ORDER` branch above diverts every higher order to the box shed. If
            // anyone ever widens that branch's comparison — say to `ORDER_LIFT_MAX_ORDER` or
            // `MAX_ORDER`, both of which now name larger numbers — orders 4..8 would land
            // here and every `pos >= 2` would shed its marginal into the WRONG pair of axes,
            // silently, with no error anywhere. That is a wrong rating table produced by a
            // one-token edit, so state the precondition where it is relied on.
            if k != LEGACY_MAX_ORDER {
                return Err(PbError::Internal {
                    what: format!(
                        "the dense order-3 shed was reached at order {k}: the two-axis \
                         mapping below is only valid at order {LEGACY_MAX_ORDER} \
                         (higher orders shed as boxes)"
                    ),
                });
            }
            let (o0, o1) = match pos {
                0 => (1usize, 2usize),
                1 => (0usize, 2usize),
                _ => (0usize, 1usize),
            };
            if !raw_tables.contains_key(&sub_u) {
                let mut sub_axes = Vec::with_capacity(2);
                let mut sub_extents = Vec::with_capacity(2);
                for sr in &sub_u.0 {
                    sub_axes.push(grids.axis_id(*sr)?);
                    sub_extents.push(grids.cells(*sr)?);
                }
                raw_tables.insert(
                    sub_u.clone(),
                    RawTable {
                        u: sub_u.clone(),
                        axes: sub_axes,
                        values: Tensor::try_zeros(sub_extents)?,
                    },
                );
            }
            let target = raw_tables
                .get_mut(&sub_u)
                .ok_or_else(|| PbError::Internal {
                    what: "factored shed target vanished".into(),
                })?;
            let low_a = low.get(o0).ok_or_else(|| PbError::Internal {
                what: "shed mask a missing".into(),
            })?;
            let low_b = low.get(o1).ok_or_else(|| PbError::Internal {
                what: "shed mask b missing".into(),
            })?;
            // `m_pos` is indexed by the sub-box corner `s_a | s_b << 1`; `add_marginal_pair`
            // wants `m[s_a][s_b]`.
            let pair = [
                [
                    *m_pos.first().ok_or_else(internal_marg())?,
                    *m_pos.get(0b10).ok_or_else(internal_marg())?,
                ],
                [
                    *m_pos.get(0b01).ok_or_else(internal_marg())?,
                    *m_pos.get(0b11).ok_or_else(internal_marg())?,
                ],
            ];
            // Deposit `marg[pos]` broadcast over the order-2 table via the masks. Bit-identical
            // to the per-cell `walk_extents(|coord| add(coord, marg[..][..]))` it replaces (each
            // cell written once, row-major), but vectorized — see `Tensor::add_marginal_pair`.
            target.values.add_marginal_pair(&pair, low_a, low_b)?;
        }
        boxes.push(FactoredBox { p, low });
    }
    raw_tables.remove(u);
    // Variance (`compute_variance`, an O(n_boxes²) sweep) is deferred to a parallel pass across
    // supports in the caller — each triple's sweep reads only its own boxes, so it parallelizes
    // bit-identically. Left 0.0 here.
    Ok(FactoredEffect {
        u: u.clone(),
        axes,
        per_axis_w,
        boxes,
        variance: 0.0,
    })
}

/// The error a missing `2x2` marginal corner raises during the order-3 shed.
fn internal_marg() -> impl Fn() -> PbError {
    || PbError::Internal {
        what: "shed marginal corner escaped the 2x2 pair".into(),
    }
}

/// Validate a per-row effective mass slice against `n_rows` (spec §08.7's `w·e`: sample
/// weight times exposure). `None` is always valid (legacy row-count semantics: every row
/// counts as mass `1.0`); `Some(m)` must match `n_rows` and be finite/non-negative in
/// every entry — a negative or non-finite mass has no meaning as accumulated support or
/// as a `ProductMarginals` empirical count. `what` names the caller for the error text.
///
/// # Errors
/// [`PbError::ShapeMismatch`] if `m.len() != n_rows`; [`PbError::InvalidInput`] if any
/// entry is negative or non-finite.
fn validate_mass(mass: Option<&[f32]>, n_rows: usize, what: &str) -> Result<(), PbError> {
    let Some(m) = mass else {
        return Ok(());
    };
    if m.len() != n_rows {
        return Err(PbError::ShapeMismatch {
            what: format!("{what} mass len {} != n_rows {n_rows}", m.len()),
        });
    }
    if let Some((row, &bad)) = m
        .iter()
        .enumerate()
        .find(|(_, &v)| !v.is_finite() || v < 0.0)
    {
        return Err(PbError::InvalidInput {
            what: format!("{what} mass row {row} must be finite and >= 0, got {bad}"),
        });
    }
    Ok(())
}

/// Fill each table's per-cell `support` from `x` (display/credibility metadata, §08.7).
/// `mass = None` sums a flat row count per cell (the historical/default behavior);
/// `Some(m)` sums `m[row]` per cell instead — spec §08.7's "effective w-mass
/// (exposure-weighted training-row count)". Runs after purify (support is not an fANOVA
/// component, so it never enters the five exactness gates either way).
///
/// # Errors
/// Propagated grid/shape errors; [`PbError::ShapeMismatch`] if `mass`'s length
/// disagrees with `x`'s row count; [`PbError::InvalidInput`] if any mass entry is
/// negative or non-finite.
fn fill_support(
    bank: &mut TableBank,
    grids: &MergedGrids,
    x: &ServeBinnedMatrix,
    mass: Option<&[f32]>,
) -> Result<(), PbError> {
    let n_rows = x.0.n_rows as usize;
    validate_mass(mass, n_rows, "support")?;
    // Precompute every raw feature's PER-ROW merged cell ONCE (single-axis: an O(model_n_bins)
    // cached bin->cell map, then O(1) per row — mirrors scoring::build_cell_maps' shape; P1
    // multi-channel joint: O(n_rows · n_channels), no compact per-bin map is possible — see
    // `MergedAxis::cell_for_every_row`'s doc) instead of re-deriving it via a linear border
    // scan on every (table, row, axis) triple — the dominant cost on a many-table bagged soup
    // over a large serve matrix. Reused across every table a raw feature appears in (main
    // effect AND any interactions), so this is a single O(n_rows) pass per raw feature overall,
    // not per table — byte-identical to the per-row `model_bin_to_cell`/joint-cell calls it
    // replaces, just computed once and shared.
    let n_raw_features = grids.n_raw_features();
    let mut row_cells: Vec<Vec<u32>> = Vec::with_capacity(n_raw_features);
    for r in 0..n_raw_features {
        let raw = FeatureId(u32::try_from(r).map_err(|_| PbError::Internal {
            what: "raw feature id exceeded u32 building support maps".into(),
        })?);
        row_cells.push(grids.axis(raw)?.cell_for_every_row(x)?);
    }
    // Tables are independent and each one's cells accumulate rows in ascending order on
    // either path, so the parallel dense fill is bit-identical to the sequential walk.
    bank.tables
        .par_iter_mut()
        .try_for_each(|table| fill_table_support(table, &row_cells, n_rows, mass))
}

/// One table's [`fill_support`]: a dense flat-index accumulation when the support tensor is
/// dense (always, since it is rebuilt with `Tensor::try_zeros`), else the per-cell walk.
fn fill_table_support(
    table: &mut EffectTable,
    row_cells: &[Vec<u32>],
    n_rows: usize,
    mass: Option<&[f32]>,
) -> Result<(), PbError> {
    // Reset to a fresh zero tensor (purify's own initial state, `Tensor::try_zeros`) before
    // accumulating: `Tensor::add` only ever accumulates, so without this reset a second
    // `fill_support` call on the same bank (the `explain_weighted` re-derive path) would
    // double-count on top of whatever the first call already wrote, instead of replacing it.
    // Makes `fill_support` idempotent.
    table.support = Tensor::try_zeros(table.support.shape())?;
    let shape = table.support.shape();
    let mut axes: Vec<&[u32]> = Vec::with_capacity(table.u.order());
    for r in &table.u.0 {
        let cells = row_cells
            .get(r.0 as usize)
            .and_then(|c| c.get(..n_rows))
            .ok_or_else(|| PbError::Internal {
                what: "support row-cells missing raw feature".into(),
            })?;
        axes.push(cells);
    }
    if axes.len() == shape.len() {
        if let Some(data) = table.support.dense_slice_mut() {
            // Row-major strides of the support tensor.
            let mut strides = vec![0usize; shape.len()];
            let mut s = 1usize;
            for (st, &e) in strides.iter_mut().zip(&shape).rev() {
                *st = s;
                s = s.checked_mul(e).ok_or_else(|| PbError::Internal {
                    what: "support tensor size overflowed".into(),
                })?;
            }
            let out_of_range = || PbError::ShapeMismatch {
                what: "tensor add coord out of range".into(),
            };
            for row in 0..n_rows {
                let mut flat = 0usize;
                for ((cells, &st), &e) in axes.iter().zip(&strides).zip(&shape) {
                    let c = *cells.get(row).ok_or_else(|| PbError::Internal {
                        what: "support row escaped precomputed cells".into(),
                    })? as usize;
                    if c >= e {
                        return Err(out_of_range());
                    }
                    flat += c * st;
                }
                let m = match mass {
                    Some(m) => f64::from(*m.get(row).ok_or_else(|| PbError::Internal {
                        what: "support mass row escaped column".into(),
                    })?),
                    None => 1.0_f64,
                };
                *data.get_mut(flat).ok_or_else(out_of_range)? += m;
            }
            return Ok(());
        }
    }
    let mut coord = vec![0usize; table.u.order()];
    for row in 0..n_rows {
        for (k, cells) in axes.iter().enumerate() {
            let cell = *cells.get(row).ok_or_else(|| PbError::Internal {
                what: "support row escaped precomputed cells".into(),
            })?;
            let slot = coord.get_mut(k).ok_or_else(|| PbError::Internal {
                what: "support coord position escaped".into(),
            })?;
            *slot = cell as usize;
        }
        let m = match mass {
            Some(m) => f64::from(*m.get(row).ok_or_else(|| PbError::Internal {
                what: "support mass row escaped column".into(),
            })?),
            None => 1.0_f64,
        };
        table.support.add(&coord, m)?;
    }
    Ok(())
}

/// The pre-2026-09-27 sequential per-cell [`fill_support`] loop, kept as the bit-identity
/// reference for the parallel dense fill.
#[cfg(test)]
fn fill_support_sequential_reference(
    bank: &mut TableBank,
    row_cells: &[Vec<u32>],
    n_rows: usize,
    mass: Option<&[f32]>,
) -> Result<(), PbError> {
    for table in &mut bank.tables {
        // Reset to a fresh zero tensor (purify's own initial state, `Tensor::try_zeros`)
        // before accumulating: `Tensor::add` only ever accumulates, so without this reset
        // a second `fill_support` call on the same bank (the `explain_weighted` re-derive
        // path) would double-count on top of whatever the first call already wrote,
        // instead of replacing it. Makes `fill_support` idempotent.
        table.support = Tensor::try_zeros(table.support.shape())?;
        let mut coord = vec![0usize; table.u.order()];
        // Hoist each support axis's precomputed per-row cells out of the row loop — invariant
        // across rows for a given table.
        let mut axes = Vec::with_capacity(table.u.order());
        for r in &table.u.0 {
            let cells = row_cells
                .get(r.0 as usize)
                .ok_or_else(|| PbError::Internal {
                    what: "support row-cells missing raw feature".into(),
                })?;
            axes.push(cells.as_slice());
        }
        for row in 0..n_rows {
            for (k, cells) in axes.iter().enumerate() {
                let cell = *cells.get(row).ok_or_else(|| PbError::Internal {
                    what: "support row escaped precomputed cells".into(),
                })?;
                let slot = coord.get_mut(k).ok_or_else(|| PbError::Internal {
                    what: "support coord position escaped".into(),
                })?;
                *slot = cell as usize;
            }
            let m = match mass {
                Some(m) => f64::from(*m.get(row).ok_or_else(|| PbError::Internal {
                    what: "support mass row escaped column".into(),
                })?),
                None => 1.0_f64,
            };
            table.support.add(&coord, m)?;
        }
    }
    Ok(())
}

// ===========================================================================
// §08.6 — The five Invariant checks (build gates), at the real bank.
// ===========================================================================

/// The sorted distinct raw features appearing in any of the bank's tables.
fn bank_features(bank: &TableBank) -> Vec<FeatureId> {
    let mut set: BTreeSet<FeatureId> = BTreeSet::new();
    for t in &bank.tables {
        for r in &t.u.0 {
            set.insert(*r);
        }
    }
    set.into_iter().collect()
}

/// The sorted distinct raw features appearing in any tree split of the model.
fn model_features(model: &Model) -> Result<Vec<FeatureId>, PbError> {
    let mut set: BTreeSet<FeatureId> = BTreeSet::new();
    for (_, tree) in &model.trees {
        for split in &tree.splits {
            let prov =
                model
                    .provenance
                    .get(split.axis as usize)
                    .ok_or_else(|| PbError::Internal {
                        what: format!("split axis {} absent from provenance", split.axis),
                    })?;
            set.insert(prov.raw);
        }
    }
    Ok(set.into_iter().collect())
}

/// The joint-grid features a gate must inspect: every model-realized raw feature AND
/// every table feature. Using only table features would let a malformed bank hide a
/// missing table by shrinking the check domain.
fn gate_features(model: &Model, bank: &TableBank) -> Result<Vec<FeatureId>, PbError> {
    let mut set: BTreeSet<FeatureId> = BTreeSet::new();
    for f in model_features(model)? {
        set.insert(f);
    }
    for f in bank_features(bank) {
        set.insert(f);
    }
    Ok(set.into_iter().collect())
}

/// Visit interior points of the joint merged grid over `feats`. Exhaustive when the
/// product is `<= JOINT_CAP` (the §08.6 worst-case-per-cell sweep); otherwise a
/// deterministic sample (the §08.8 release behavior). Each visit gets `x_cells` (indexed
/// by raw feature) and `rep_bins` (indexed by model axis) for the same point. Returns
/// `true` iff the joint grid was SAMPLED (rather than exhausted) — the integral gates
/// (VarianceSum) self-normalize and widen their tolerance in that case; the per-point
/// gates (Reconstruction/ThreeWayEqual) are sound either way and ignore the flag.
fn enumerate_check_points(
    grids: &MergedGrids,
    feats: &[FeatureId],
    mut visit: impl FnMut(&[u32], &[u8]) -> Result<(), PbError>,
) -> Result<bool, PbError> {
    let n_raw_features = grids.n_raw_features();
    let n_axes = grids.n_axes();
    let mut extents = Vec::with_capacity(feats.len());
    for r in feats {
        extents.push(grids.cells(*r)?);
    }
    // SATURATE, don't error: on a wide model the joint grid (Π cells over all gate features)
    // can exceed u64 — that simply means it dwarfs the enumeration cap and must be SAMPLED.
    // (Erroring here was the high-dimensional-categorical decomposition bug, e.g. allstate's
    // 130 features overflowing u64 before the cap check could route to sampling.)
    let total = saturating_product_u64(&extents);

    let mut emit = |tuple: &[usize]| -> Result<(), PbError> {
        // `x_cells` indexed by RAW FEATURE (one scalar per raw feature, P1 multi-channel or
        // not); `rep_bins` indexed by MODEL AXIS (one entry per axis — up to K per raw feature
        // once a categorical has K channels), matching `Model::ensemble_f64`'s per-split
        // `row_bins.get(split.axis)` convention. These sizes DIVERGE once any raw feature owns
        // more than one axis, which is exactly why both are threaded through this function
        // (previously `n_features` conflated them, since raw==axis always held before P1).
        let mut x_cells = vec![0u32; n_raw_features];
        let mut rep_bins = vec![0u8; n_axes];
        for (k, r) in feats.iter().enumerate() {
            let cell = *tuple.get(k).ok_or_else(|| PbError::Internal {
                what: "check tuple shorter than feats".into(),
            })?;
            let ma = grids.axis(*r)?;
            *x_cells
                .get_mut(r.0 as usize)
                .ok_or_else(|| PbError::Internal {
                    what: "x_cells raw index escaped".into(),
                })? = cell as u32;
            ma.write_rep_bins(cell, &mut rep_bins)?;
        }
        visit(&x_cells, &rep_bins)
    };

    if total <= joint_cap() as u64 {
        walk_extents(&extents, &mut emit)?;
        Ok(false)
    } else {
        // Deterministic strided sample: mix the sample index per axis with a splitmix
        // step so the points spread across the space without RNG state.
        let mut tuple = vec![0usize; feats.len()];
        for s in 0..joint_cap() as u64 {
            for (k, &e) in extents.iter().enumerate() {
                let mut z = s
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((k as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z ^= z >> 27;
                *tuple.get_mut(k).ok_or_else(|| PbError::Internal {
                    what: "sample tuple position escaped".into(),
                })? = (z % e as u64) as usize;
            }
            emit(&tuple)?;
        }
        Ok(true)
    }
}

/// **Reconstruction (I2.1):** the ensemble equals `f0 + Σ_u f_u` at every merged-grid
/// cell (the missing cell included), within `recon_tol`.
///
/// # Errors
/// [`Invariant::Reconstruction`] if any cell's discrepancy exceeds tolerance.
pub fn check_reconstruction(model: &Model, bank: &TableBank) -> Result<(), PbError> {
    let tol = ExactTol::for_model(model).recon_tol;
    let grids = MergedGrids::from_model(model)?;
    let feats = gate_features(model, bank)?;
    enumerate_check_points(&grids, &feats, |x_cells, rep_bins| {
        let ens = model.ensemble_f64(rep_bins)?;
        let tab = bank.score(x_cells)?;
        if (ens - tab).abs() > tol {
            return Err(PbError::invariant(Invariant::Reconstruction));
        }
        Ok(())
    })?;
    Ok(())
}

/// The exact `w`-weighted mean of the ensemble, `E_w[F] = f0 + Σ_t alpha_t·E_w[T_t]`.
/// Computed SEPARABLY per tree over its own ≤3-feature support grid (integrating out the
/// other features, whose product weights sum to 1) — so it never materializes the joint
/// grid and stays exact regardless of the ensemble's total feature count. This is the
/// fix for the prior joint-grid sampler, which produced an un-normalized partial sum and
/// false-failed large exact models on MassConservation.
fn ensemble_w_mean(model: &Model, grids: &MergedGrids, w: &WeightCache) -> Result<f64, PbError> {
    let mut mass = f64::from(model.f0);
    for (alpha, tree) in &model.trees {
        let u = tree_support(model, tree)?;
        let u_ids: Vec<FeatureId> = u.0.iter().copied().collect();
        let mut extents = Vec::with_capacity(u_ids.len());
        for r in &u_ids {
            extents.push(grids.cells(*r)?);
        }
        let mut tree_int = 0.0_f64;
        walk_extents(&extents, |tuple| {
            let leaf_idx = leaf_index_for_tuple(model, tree, grids, &u_ids, tuple)?;
            let leaf = *tree.leaves.get(leaf_idx).ok_or_else(|| PbError::Internal {
                what: format!(
                    "mass: oblivious leaf index {leaf_idx} escaped 0..{}",
                    tree.leaves.len()
                ),
            })?;
            let mut wprod = 1.0_f64;
            for (k, r) in u_ids.iter().enumerate() {
                let cell = *tuple.get(k).ok_or_else(|| PbError::Internal {
                    what: "mass tuple shorter than support".into(),
                })?;
                wprod *= *w.axis(*r)?.get(cell).ok_or_else(|| PbError::Internal {
                    what: "mass cell escaped axis weights".into(),
                })?;
            }
            tree_int += wprod * f64::from(leaf);
            Ok(())
        })?;
        mass += f64::from(*alpha) * tree_int;
    }
    // The cell-basis correction is added to `ensemble_f64`, so its w-weighted mass must be
    // accounted here too — otherwise it lands only in `bank.f0` (via the purify cascade) and
    // mass conservation (I2.2: `E_w[F_ens] == f0`) fails by exactly `Σ_cells w·delta`.
    if let Some(corr) = &model.correction {
        for table in &corr.tables {
            let mut raws = Vec::with_capacity(table.axes.len());
            for &axis in &table.axes {
                raws.push(
                    model
                        .provenance
                        .get(axis as usize)
                        .ok_or_else(|| PbError::Internal {
                            what: format!("correction mass axis {axis} absent from provenance"),
                        })?
                        .raw,
                );
            }
            let extents: Vec<usize> = table.shape.iter().map(|&s| s as usize).collect();
            let mut flat = 0usize;
            walk_extents(&extents, |tuple| {
                let mut wprod = 1.0_f64;
                for (k, r) in raws.iter().enumerate() {
                    let cell = *tuple.get(k).ok_or_else(|| PbError::Internal {
                        what: "correction mass tuple shorter than support".into(),
                    })?;
                    wprod *= *w.axis(*r)?.get(cell).ok_or_else(|| PbError::Internal {
                        what: "correction mass cell escaped axis weights".into(),
                    })?;
                }
                let v = *table.values.get(flat).ok_or_else(|| PbError::Internal {
                    what: "correction mass flat index escaped values".into(),
                })?;
                flat += 1;
                mass += wprod * v;
                Ok(())
            })?;
        }
    }
    Ok(mass)
}

/// **MassConservation (I2.2):** all `w`-mass that survives purification sits in the
/// intercept — `E_w[F_ens] == f0` (every table integrates to zero). Computed EXACTLY via
/// the separable per-tree integral ([`ensemble_w_mean`]), so it is correct for arbitrarily
/// large ensembles (no joint-grid enumeration / sampling).
///
/// # Errors
/// [`Invariant::MassConservation`] if the `w`-weighted ensemble mean drifts from `f0`.
pub(crate) fn check_mass_conservation(
    model: &Model,
    bank: &TableBank,
    w: &WeightCache,
) -> Result<(), PbError> {
    let tol = ExactTol::for_model(model).mass_tol;
    let grids = MergedGrids::from_model(model)?;
    let mass = ensemble_w_mean(model, &grids, w)?;
    if (mass - bank.f0).abs() > tol {
        return Err(PbError::invariant(Invariant::MassConservation));
    }
    Ok(())
}

/// The product weight `Π_{i ∈ feats} w_i(x_i)` for a joint cell-tuple.
fn joint_weight(w: &WeightCache, feats: &[FeatureId], x_cells: &[u32]) -> Result<f64, PbError> {
    let mut wprod = 1.0_f64;
    for r in feats {
        let cell = *x_cells.get(r.0 as usize).ok_or_else(|| PbError::Internal {
            what: "joint-weight raw index escaped x_cells".into(),
        })? as usize;
        let wc = *w.axis(*r)?.get(cell).ok_or_else(|| PbError::Internal {
            what: "joint-weight cell escaped axis weights".into(),
        })?;
        wprod *= wc;
    }
    Ok(wprod)
}

/// **Purity (I2.3):** every axis-slice of every [`EffectTable`] has `w`-weighted mean
/// zero. Maps to [`Invariant::Decomposability`] (§2.8 has no separate `Purity`).
///
/// # Errors
/// [`Invariant::Decomposability`] if any conditional axis-slice mean is non-zero.
pub(crate) fn check_purity(
    model: &Model,
    bank: &TableBank,
    w: &WeightCache,
) -> Result<(), PbError> {
    let tol = ExactTol::for_model(model).purity_tol;
    // Every table is checked independently (a pass/fail per table), so the tables fan out.
    bank.tables
        .par_iter()
        .try_for_each(|table| -> Result<(), PbError> {
            let extents = table.values.shape();
            for p in 0..table.u.order() {
                let r = *table.u.0.get(p).ok_or_else(|| PbError::Internal {
                    what: "purity axis escaped support".into(),
                })?;
                let axis_w = w.axis(r)?;
                // Dense tables: the same slice means `center_along` would shed (each summed over
                // `p` ascending from 0.0, as the keyed walk below sums them), without its keys.
                if let (Some(data), Some((outer, mid, inner))) =
                    (table.values.dense_slice(), axis_split(&extents, p))
                {
                    let wp = axis_w.get(..mid).ok_or_else(|| PbError::Internal {
                        what: "purity cell escaped axis weights".into(),
                    })?;
                    let mut means = vec![0.0_f64; inner];
                    for block in data.chunks_exact(mid * inner).take(outer) {
                        means.iter_mut().for_each(|m| *m = 0.0);
                        for (row, &wi) in block.chunks_exact(inner).zip(wp) {
                            for (m, &v) in means.iter_mut().zip(row) {
                                *m += wi * v;
                            }
                        }
                        if means.iter().any(|m| m.abs() > tol) {
                            return Err(PbError::invariant(Invariant::Decomposability));
                        }
                    }
                    continue;
                }
                let mut means: BTreeMap<Vec<usize>, f64> = BTreeMap::new();
                walk_extents(&extents, |coord| {
                    let cell_p = *coord.get(p).ok_or_else(|| PbError::Internal {
                        what: "purity position escaped coord".into(),
                    })?;
                    let wp = *axis_w.get(cell_p).ok_or_else(|| PbError::Internal {
                        what: "purity cell escaped axis weights".into(),
                    })?;
                    let v = table.values.at(coord).ok_or_else(|| PbError::Internal {
                        what: "purity coord out of range".into(),
                    })?;
                    *means.entry(drop_index(coord, p)).or_insert(0.0) += wp * v;
                    Ok(())
                })?;
                for m in means.values() {
                    if m.abs() > tol {
                        return Err(PbError::invariant(Invariant::Decomposability));
                    }
                }
            }
            Ok(())
        })?;
    // Factored order-3 effects: each per-tree residual box is triple-centered by
    // construction, so re-centering it must shed ZERO marginals (every axis-slice already
    // has w-weighted mean 0). Σ_t of purified boxes is purified, so this certifies the
    // factored f_u's purity without densifying the merged cube.
    for ft in &bank.factored {
        for b in &ft.boxes {
            let (wlo, whi) = side_weights(&b.low, &ft.per_axis_w)?;
            let mut p = b.p.clone();
            let marg = center_box_capturing(&mut p, &wlo, &whi)?;
            for axis_marg in &marg {
                for &m in axis_marg {
                    if m.abs() > tol {
                        return Err(PbError::invariant(Invariant::Decomposability));
                    }
                }
            }
        }
    }
    Ok(())
}

/// **VarianceSum (I2.4):** `σ²(F) == Σ_u σ²(f_u)` under product/uniform `w`.
///
/// # Errors
/// [`Invariant::VarianceSum`] if total variance diverges from the sum of per-table
/// variances.
pub(crate) fn check_variance_sum(
    model: &Model,
    bank: &TableBank,
    w: &WeightCache,
) -> Result<(), PbError> {
    let centered_score = |bins: &[u8]| -> Result<f64, PbError> {
        let mut score = model.correction_delta(bins)?;
        for (alpha, tree) in &model.trees {
            score += f64::from(*alpha) * f64::from(tree.lookup(bins)?);
        }
        Ok(score)
    };
    let tol = ExactTol::for_model(model).var_tol;
    let grids = MergedGrids::from_model(model)?;
    let feats = gate_features(model, bank)?;
    let mut extents = Vec::with_capacity(feats.len());
    for r in &feats {
        extents.push(grids.cells(*r)?);
    }
    let total = saturating_product_u64(&extents);

    let (var_ens, sampled) = if total <= joint_cap() as u64 {
        // EXHAUSTIVE: exact w-weighted moments over the full joint grid. `wsum == 1` here (the
        // product of per-axis-normalized weights summed over the full grid), so the
        // self-normalization is a no-op and the integral stays bit-exact.
        let (mut m1, mut m2, mut wsum) = (0.0_f64, 0.0_f64, 0.0_f64);
        enumerate_check_points(&grids, &feats, |x_cells, rep_bins| {
            let wprod = joint_weight(w, &feats, x_cells)?;
            let e = centered_score(rep_bins)?;
            if wprod > 0.0 {
                let next_weight = wsum + wprod;
                let delta = e - m1;
                m1 += delta * wprod / next_weight;
                m2 += wprod * delta * (e - m1);
                wsum = next_weight;
            }
            Ok(())
        })?;
        if !wsum.is_finite() || wsum <= 0.0 {
            return Err(PbError::Internal {
                what: "variance check accumulated non-positive total weight".into(),
            });
        }
        (m2 / wsum, false)
    } else {
        // SAMPLED: draw each axis's cell from the REFERENCE MEASURE `w` itself (per-axis
        // inverse-CDF via a deterministic splitmix draw), then take an UNWEIGHTED average.
        // This keeps the effective sample size at N even in high dimensions. The former
        // uniform-sample + product-weight importance estimator collapsed on wide models —
        // a handful of points carried nearly all the weight — and false-failed VarianceSum
        // (e.g. allstate's ~40 realized features), even though the decomposition is exact
        // (reconstruction/mass/purity all hold). Sampling ∝ w removes the weight collapse.
        let mut cdfs: Vec<Vec<f64>> = Vec::with_capacity(feats.len());
        let mut totals: Vec<f64> = Vec::with_capacity(feats.len());
        for r in &feats {
            let aw = w.axis(*r)?;
            let mut cum = Vec::with_capacity(aw.len());
            let mut acc = 0.0_f64;
            for &x in aw {
                acc += x.max(0.0);
                cum.push(acc);
            }
            if !(acc.is_finite() && acc > 0.0) {
                return Err(PbError::Internal {
                    what: "variance sampler: axis has non-positive total weight".into(),
                });
            }
            cdfs.push(cum);
            totals.push(acc);
        }
        // `x_cells` raw-indexed, `rep_bins` axis-indexed — see `enumerate_check_points`'s doc
        // on why these sizes diverge once a raw feature can own more than one axis (P1).
        let mut x_cells = vec![0u32; grids.n_raw_features()];
        let mut rep_bins = vec![0u8; grids.n_axes()];
        let samples = joint_cap() as u64;
        let (mut m1, mut m2) = (0.0_f64, 0.0_f64);
        for s in 0..samples {
            for (k, r) in feats.iter().enumerate() {
                let mut z = s
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((k as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z ^= z >> 27;
                let unit = (z >> 11) as f64 / (1u64 << 53) as f64; // uniform in [0, 1)
                let u = unit * totals[k];
                let cell = cdfs[k].partition_point(|&c| c <= u).min(cdfs[k].len() - 1);
                let ma = grids.axis(*r)?;
                *x_cells
                    .get_mut(r.0 as usize)
                    .ok_or_else(|| PbError::Internal {
                        what: "variance sample x_cells index escaped".into(),
                    })? = cell as u32;
                ma.write_rep_bins(cell, &mut rep_bins)?;
            }
            let e = centered_score(&rep_bins)?;
            let delta = e - m1;
            m1 += delta / (s + 1) as f64;
            m2 += delta * (e - m1);
        }
        let nf = samples as f64;
        (m2 / nf, true)
    };
    let var_tables: f64 = bank.tables.iter().map(|t| t.variance).sum::<f64>()
        + bank.factored.iter().map(|ft| ft.variance).sum::<f64>();
    // Exact tolerance when exhaustive; a relative band when sampled (the sampled estimator
    // carries Monte-Carlo error, so for >JOINT_CAP-cell models VarianceSum is a STATISTICAL
    // certification, not bit-exact — documented FLAG; MassConservation stays exact).
    let eff_tol = if sampled {
        tol + SAMPLE_VAR_REL * (var_tables.abs() + var_ens.abs())
    } else {
        tol
    };
    if (var_ens - var_tables).abs() > eff_tol {
        return Err(PbError::invariant(Invariant::VarianceSum));
    }
    Ok(())
}

/// **ThreeWayEqual (I2.5):** tree-sum = table-sum = Shapley-sum at every cell, within
/// the derived float tolerance. The Shapley leg is an INDEPENDENT path (each `f_u`
/// split equally among its `|u|` features, summed per feature).
///
/// # Errors
/// [`Invariant::ThreeWayEqual`] if any of the three reconstructions disagree.
pub fn check_three_way_equal(model: &Model, bank: &TableBank) -> Result<(), PbError> {
    let tol = ExactTol::for_model(model).recon_tol;
    let grids = MergedGrids::from_model(model)?;
    let feats = gate_features(model, bank)?;
    enumerate_check_points(&grids, &feats, |x_cells, rep_bins| {
        let tree = model.ensemble_f64(rep_bins)?;
        let table = bank.score(x_cells)?;
        let shap = bank.f0 + bank.shap(x_cells)?.iter().sum::<f64>();
        if (tree - table).abs() > tol || (table - shap).abs() > tol {
            return Err(PbError::invariant(Invariant::ThreeWayEqual));
        }
        Ok(())
    })?;
    Ok(())
}

/// Run all five I2 checks against a real fitted model and its purified bank (spec
/// §13.1). Rebuilds the reference weights from `x` (a [`ServeBinnedMatrix`], R-CATSERVE)
/// and `bank.w`. `VarianceSum` is asserted under product/uniform `w` (the v1 measures).
///
/// # Errors
/// The first failing check's [`Invariant`], wrapped in [`PbError::InvariantViolated`];
/// or a propagated weight/grid error.
pub fn assert_exact_decomposition(
    model: &Model,
    bank: &TableBank,
    x: &ServeBinnedMatrix,
) -> Result<(), PbError> {
    if x.0.grids != model.grids || x.0.provenance != model.provenance {
        return Err(PbError::ShapeMismatch {
            what: "exactness serve grids do not match model".into(),
        });
    }
    let w = build_weights_from_support(bank, &bank.w)?;
    check_reconstruction(model, bank)?;
    check_mass_conservation(model, bank, &w)?;
    check_purity(model, bank, &w)?;
    check_variance_sum(model, bank, &w)?;
    check_three_way_equal(model, bank)?;
    Ok(())
}

/// **FeatureBudget (I1, spec §13.2):** every tree is depth `1..=3`, `splits.len() ==
/// depth`, and the count of DISTINCT raw features across its splits is in `1..=depth`
/// and never exceeds 3.
///
/// # Errors
/// [`Invariant::FeatureBudget`] for any tree that violates the depth-3 / ≤3-distinct
/// contract; [`PbError::Internal`] if a split names an axis absent from provenance.
pub fn check_feature_budget(model: &Model) -> Result<(), PbError> {
    for (_, tree) in &model.trees {
        let depth = usize::from(tree.depth);
        if !i1_depth_ok(depth) || tree.splits.len() != depth {
            return Err(PbError::invariant(Invariant::FeatureBudget));
        }
        let mut distinct: SmallVec<[u32; MAX_ORDER]> = SmallVec::new();
        for split in &tree.splits {
            let prov =
                model
                    .provenance
                    .get(split.axis as usize)
                    .ok_or_else(|| PbError::Internal {
                        what: format!("split axis {} absent from provenance", split.axis),
                    })?;
            let raw = prov.raw.0;
            if !distinct.contains(&raw) {
                distinct.push(raw);
            }
        }
        if !i1_shape_ok(depth, distinct.len()) {
            return Err(PbError::invariant(Invariant::FeatureBudget));
        }
    }
    Ok(())
}

// ===========================================================================
// §08.5 / §08.8 — Public API: Model::explain + TableBank reads.
// ===========================================================================

impl Model {
    /// Build the complete purified [`TableBank`] under `w` (spec §08.8). Takes a
    /// [`ServeBinnedMatrix`] (R-CATSERVE): the caller re-encodes raw categoricals
    /// through this model's frozen `schema.cat_encoders` — `explain` MUST NOT be handed a
    /// `TrainBinnedMatrix`. Runs all five build gates by default. An `Approximate` model
    /// refuses to export an `Exact` bank ([`PbError::ExactnessFirewall`]).
    ///
    /// # Errors
    /// [`PbError::ExactnessFirewall`] on an `Approximate` model; [`PbError::InvalidConfig`]
    /// for the unsupported-in-v1 `Joint` measure or a categorical axis; [`PbError::TableBudget`]
    /// if a table or the bank exceeds its cell budget; [`PbError::InvariantViolated`] if any
    /// of the five gates fail; plus propagated shape/grid errors.
    pub fn explain(&self, x: &ServeBinnedMatrix, w: RefMeasure) -> Result<TableBank, PbError> {
        self.explain_with_budget(x, w, TableBudget::default())
    }

    /// Build the complete purified [`TableBank`] with an explicit table budget
    /// (spec §08.10). This is the same exact pipeline as [`Model::explain`], but lets
    /// callers choose [`OverflowPolicy::SparseFallback`] for adversarial or large
    /// merged grids instead of the default hard-error policy.
    ///
    /// # Errors
    /// Same as [`Model::explain`], plus [`PbError::InvalidConfig`] if the sparse
    /// fallback threshold is malformed.
    pub fn explain_with_budget(
        &self,
        x: &ServeBinnedMatrix,
        w: RefMeasure,
        budget: TableBudget,
    ) -> Result<TableBank, PbError> {
        self.explain_bank(x, w, budget, true)
    }

    /// Like [`Model::explain`], but weights every table's `support` — AND, when `w` is
    /// [`RefMeasure::ProductMarginals`], the reference measure's empirical marginals — by
    /// `weight[row]` instead of a flat row count: spec §08.7's "effective `w`-mass
    /// (exposure-weighted training-row count)". `weight` is typically the same per-row
    /// sample/exposure weight (or `sample_weight * exposure`) the model was fit with.
    ///
    /// This is NOT a display-only variant of [`Model::explain`]: because `weight` also
    /// feeds `ProductMarginals` (via [`build_weights`]), table `values`, Sobol
    /// importances, and SE bands can differ from the unweighted bank — BY DESIGN, a
    /// different (exposure-correct) empirical reference measure is a different valid
    /// decomposition. `w = Uniform` is unaffected in `values` (its per-axis weights never
    /// depend on data); only `support` changes there.
    ///
    /// On an aggregated portfolio (one row = many policy-years), `support` under
    /// `explain` understates backing mass by the aggregation factor; `explain_weighted`
    /// with the per-row exposure fixes that (the motivating case: "a confident-looking
    /// 2.4x relativity backed by ~11 policies" should mean ~11 policy-years of mass, not
    /// 11 aggregated rows that are actually thousands).
    ///
    /// # Errors
    /// Same as [`Model::explain`], plus [`PbError::ShapeMismatch`] if `weight.len() !=
    /// x`'s row count, or [`PbError::InvalidInput`] if any weight is negative or
    /// non-finite.
    pub fn explain_weighted(
        &self,
        x: &ServeBinnedMatrix,
        w: RefMeasure,
        weight: &[f32],
    ) -> Result<TableBank, PbError> {
        self.explain_with_budget_weighted(x, w, TableBudget::default(), weight)
    }

    /// [`Model::explain_weighted`] with an explicit table budget — see
    /// [`Model::explain_with_budget`] for the budget knob, [`Model::explain_weighted`]
    /// for the weighting contract.
    ///
    /// # Errors
    /// Same as [`Model::explain_weighted`] and [`Model::explain_with_budget`].
    pub fn explain_with_budget_weighted(
        &self,
        x: &ServeBinnedMatrix,
        w: RefMeasure,
        budget: TableBudget,
        weight: &[f32],
    ) -> Result<TableBank, PbError> {
        self.explain_bank_with_mass(x, w, budget, true, Some(weight))
    }

    /// Return the realised fANOVA table supports without running the ensemble-anchored export
    /// gates. This is for selectors that only need the support lattice; callers that need table
    /// values and exactness certification should use [`Model::explain_with_budget`].
    ///
    /// # Errors
    /// Same data-shape / budget errors as [`Model::explain_with_budget`].
    pub fn table_supports(
        &self,
        x: &ServeBinnedMatrix,
        w: RefMeasure,
        budget: TableBudget,
    ) -> Result<Vec<FeatureSet>, PbError> {
        let bank = self.explain_bank(x, w, budget, false)?;
        let mut supports: BTreeSet<FeatureSet> = BTreeSet::new();
        supports.extend(bank.tables.iter().map(|t| t.u.clone()));
        supports.extend(bank.factored.iter().map(|t| t.u.clone()));
        Ok(supports.into_iter().collect())
    }

    /// Build the purified [`TableBank`], optionally skipping the four **ensemble-anchored** exactness
    /// gates (MassConservation, Reconstruction, VarianceSum, ThreeWayEqual). `run_gates = false` is
    /// used by the pruner (`crate::prune`): the model is already `Exact` and the pruned bank is
    /// re-checked against its OWN LUT-sum, so re-proving the *discarded* ensemble's exact
    /// decomposition here is redundant — and it dominates `explain` (≈35 of 37 s on `diamonds`). Only
    /// the cheap self-property check (Purity) still runs unconditionally; on its own it does NOT
    /// certify `f0` against the ensemble mean (that's what the skipped MassConservation gate does).
    ///
    /// Derives this model's OWN merged grid via [`MergedGrids::from_model`]. Thin wrapper over
    /// [`Model::explain_bank_with_grids_and_mass`] — see it for the shared-grid entry point multiple
    /// related models purify against (e.g. `crate::prune::bag_banks_for_keepset`'s per-bag
    /// banks, purified on the deployed soup's grid so purify's fixed-grid linearity holds).
    ///
    /// # Errors
    /// Same as [`Model::explain_bank_with_grids_and_mass`], plus propagated [`MergedGrids::from_model`]
    /// errors.
    pub(crate) fn explain_bank(
        &self,
        x: &ServeBinnedMatrix,
        w: RefMeasure,
        budget: TableBudget,
        run_gates: bool,
    ) -> Result<TableBank, PbError> {
        self.explain_bank_with_mass(x, w, budget, run_gates, None)
    }

    /// Like [`Model::explain_bank`], but threads a per-row effective `mass` through to
    /// [`Model::explain_bank_with_grids_and_mass`] — see it for the weighting contract.
    /// `mass = None` is IDENTICAL to [`Model::explain_bank`] (thin wrapper, zero behavior
    /// change: this is what [`Model::explain_bank`] itself now calls).
    ///
    /// # Errors
    /// Same as [`Model::explain_bank`], plus [`PbError::ShapeMismatch`] if `mass.len() !=
    /// x`'s row count, or [`PbError::InvalidInput`] if any mass entry is negative or
    /// non-finite.
    pub(crate) fn explain_bank_with_mass(
        &self,
        x: &ServeBinnedMatrix,
        w: RefMeasure,
        budget: TableBudget,
        run_gates: bool,
        mass: Option<&[f32]>,
    ) -> Result<TableBank, PbError> {
        // Preserve the pre-refactor check order: the Approximate firewall must fire before
        // grid derivation ever touches self.trees (an I1-violating tree — the sole reason a
        // model is Approximate — can also trip a less specific error inside from_model, so
        // checking mode first is what makes ExactnessFirewall the one callers see).
        if let crate::engine::ExactnessMode::Approximate { reason } = &self.mode {
            return Err(PbError::ExactnessFirewall(reason.clone()));
        }
        let grids = MergedGrids::from_model(self)?;
        self.explain_bank_with_grids_and_mass(x, w, budget, run_gates, &grids, mass)
    }

    /// The shared-grid bank build: like [`Model::explain_bank_with_mass`] but purified on the
    /// caller's `grids` (a soup/union grid at least as fine as this model realizes), and
    /// accepting a per-row effective `mass` (spec §08.7's `w·e`: sample weight times
    /// exposure) that feeds BOTH the `support` tensors ([`fill_support`]) AND, when `w` is
    /// [`RefMeasure::ProductMarginals`] or [`RefMeasure::ExposureMarginals`], the empirical
    /// reference-measure weights ([`build_weights`]) — the same per-row quantity drives both,
    /// so a table's displayed support and its purification measure agree on what "how much
    /// data backs this cell" means. `mass = None` keeps the pre-`mass` behavior: every
    /// table's support sums a flat row count and the marginal measures count rows
    /// (BYTE-IDENTICAL — the `None` arm of every mass match below is the exact expression
    /// this file used before `mass` existed).
    ///
    /// Because `mass` reaches [`build_weights`], the resulting bank's `values`/Sobol/SE
    /// bands can differ from the unweighted bank under `ProductMarginals` — BY DESIGN
    /// (spec §08.7): purification under a mass-derived reference measure is a different,
    /// exposure-correct decomposition, not a display-only change. `Uniform` is unaffected
    /// (its per-axis weights never depend on data, mass included).
    ///
    /// # Errors
    /// Same as [`Model::explain_bank_with_grids_and_mass`], plus [`PbError::ShapeMismatch`] if
    /// `mass.len() != x`'s row count, or [`PbError::InvalidInput`] if any mass entry is
    /// negative or non-finite.
    pub(crate) fn explain_bank_with_grids_and_mass(
        &self,
        x: &ServeBinnedMatrix,
        w: RefMeasure,
        budget: TableBudget,
        run_gates: bool,
        grids: &MergedGrids,
        mass: Option<&[f32]>,
    ) -> Result<TableBank, PbError> {
        if let crate::engine::ExactnessMode::Approximate { reason } = &self.mode {
            return Err(PbError::ExactnessFirewall(reason.clone()));
        }
        let n_features = self.provenance.len();
        if x.0.data.len() != n_features {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "serve matrix has {} columns, model has {n_features} features",
                    x.0.data.len()
                ),
            });
        }
        if x.0.grids.len() != n_features {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "serve matrix has {} grids, model has {n_features} features",
                    x.0.grids.len()
                ),
            });
        }
        if x.0.provenance.len() != n_features {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "serve matrix has {} provenance entries, model has {n_features} features",
                    x.0.provenance.len()
                ),
            });
        }
        if x.0.grids != self.grids {
            return Err(PbError::ShapeMismatch {
                what: "serve matrix grids do not match model grids".into(),
            });
        }
        if x.0.provenance != self.provenance {
            return Err(PbError::ShapeMismatch {
                what: "serve matrix provenance does not match model provenance".into(),
            });
        }
        let n_rows = x.0.n_rows as usize;
        for (a, col) in x.0.data.iter().enumerate() {
            if col.len() != n_rows {
                return Err(PbError::ShapeMismatch {
                    what: format!("serve column {a} len {} != n_rows {n_rows}", col.len()),
                });
            }
            let grid = self.grids.get(a).ok_or_else(|| PbError::Internal {
                what: "model grid disappeared during serve validation".into(),
            })?;
            for (row, &bin) in col.iter().enumerate() {
                if u16::from(bin) >= grid.n_bins {
                    return Err(PbError::InvalidInput {
                        what: format!(
                            "serve column {a} row {row} bin {bin} outside model grid n_bins {}",
                            grid.n_bins
                        ),
                    });
                }
            }
        }
        validate_mass(mass, n_rows, "explain")?;

        let prof = std::env::var_os("TBOOST_PROFILE").is_some();
        let mut _t = std::time::Instant::now();
        macro_rules! mark {
            ($label:literal) => {
                if prof {
                    eprintln!(
                        "[explain] {}: {:.1} ms",
                        $label,
                        _t.elapsed().as_secs_f64() * 1e3
                    );
                    _t = std::time::Instant::now();
                }
            };
        }

        let supports = tree_supports(self)?;
        let plan = table_allocation_plan(self, grids, &budget, &supports)?;
        let (mut raw, factored_supports) =
            accumulate_with_plan(self, grids, &budget, &supports, Some(&plan))?;
        // Each factored support's trees, in tree order, so a shed visits only its own trees
        // (the same trees in the same order as a full-ensemble scan would match).
        let mut trees_of: BTreeMap<&FeatureSet, Vec<usize>> = BTreeMap::new();
        for (i, u) in supports.iter().enumerate() {
            if factored_supports.contains(u) {
                trees_of.entry(u).or_default().push(i);
            }
        }
        mark!("accumulate");
        fold_correction_into_raw(self, grids, &mut raw)?;
        let weights = build_weights(x, grids, &w, mass)?;
        mark!("build_weights");
        // §08.10: shed each factored support per realized region term, keeping the residual
        // factored (no dense cube). Driven in DESCENDING order so an order-3 effect has
        // collected every order-4 marginal shed onto it before it centers — see
        // `factored_shed_into_raw`. No-op when nothing is factored.
        let mut factored = Vec::with_capacity(factored_supports.len());
        let mut deposits: BTreeMap<FeatureSet, Vec<RankOneBox>> = BTreeMap::new();
        for order in (LEGACY_MAX_ORDER..=MAX_ORDER).rev() {
            // Supports of this order that trees realized, PLUS any that exist only because a
            // higher order shed onto them. `BTreeSet` keeps the iteration order fixed, which
            // is what makes the box sequence — and every sum built from it — deterministic.
            let mut at_order: BTreeSet<FeatureSet> = factored_supports
                .iter()
                .filter(|u| u.order() == order)
                .cloned()
                .collect();
            at_order.extend(deposits.keys().filter(|u| u.order() == order).cloned());
            for u in at_order {
                let inherited = deposits.remove(&u).unwrap_or_default();
                factored.push(factored_shed_into_raw_for(
                    self,
                    &u,
                    trees_of.get(&u).map_or(&[][..], Vec::as_slice),
                    grids,
                    &weights,
                    &mut raw.tables,
                    inherited,
                    &mut deposits,
                )?);
            }
        }
        if let Some((u, _)) = deposits.into_iter().next() {
            return Err(PbError::Internal {
                what: format!("factored shed left an undelivered deposit for {u:?}"),
            });
        }
        mark!("factored_shed");
        // Each factored triple's variance is an independent O(n_boxes²) sweep over its own boxes
        // (no shared state), so compute them in parallel across supports — bit-identical (each
        // value comes from the same sequential fold) and deterministic at any thread count. On the
        // bagged deploy soup these sweeps are the bulk of `explain`; single-bag fold banks have
        // few/no factored triples, so this is a no-op there.
        factored
            .par_iter_mut()
            .try_for_each(|ft| -> Result<(), PbError> {
                ft.variance = ft.compute_variance()?;
                Ok(())
            })?;
        mark!("factored_variance");
        if run_gates {
            verify_raw_accumulation(self, &raw, &factored, grids)?;
            mark!("verify_raw");
        }
        let mut bank =
            purify_with_budget(raw, &weights, grids, PurifyMode::SinglePass, Some(&budget))?;
        bank.factored = factored;
        mark!("purify");
        fill_support(&mut bank, grids, x, mass)?;
        mark!("fill_support");

        // Purity (per-table zero-mean) is a self-property of the bank and cheap; always verify it as
        // the pruned model's well-formedness check.
        check_purity(self, &bank, &weights)?;
        mark!("check_purity");
        // The remaining gates are all ensemble-anchored: Reconstruction / VarianceSum / ThreeWayEqual
        // re-prove the ORIGINAL ensemble's exact decomposition, and MassConservation re-derives its
        // mean. All redundant for a pruner that discards the ensemble and re-checks its bank against
        // its own LUT-sum — `bank.f0 == E_w[F]` already holds by purify construction, so the re-anchor
        // stays correct.
        if run_gates {
            check_mass_conservation(self, &bank, &weights)?;
            mark!("check_mass_conservation");
            check_reconstruction(self, &bank)?;
            mark!("check_reconstruction");
            check_variance_sum(self, &bank, &weights)?;
            mark!("check_variance_sum");
            check_three_way_equal(self, &bank)?;
            mark!("check_three_way_equal");
        }
        Ok(bank)
    }
}

/// Fold a model's cell-basis correction (the §G1 adaptive fANOVA-cell refit) into the
/// raw per-support tensors, in place, BEFORE purify. Each correction table's raw delta is
/// added cell-for-cell on the merged grid; a zero raw table is inserted first when the
/// support was not a realized tree support (e.g. a pure main whose feature only appeared
/// inside higher-order trees). No-op when the model carries no correction.
///
/// This is the decompose half of the G0-exact symmetry: the SAME raw delta that
/// [`Model::correction_delta`] adds to the tree score is added here to the raw effects, so
/// `purify` re-expresses `trees + delta` losslessly and `bank.score == ensemble_f64`.
fn fold_correction_into_raw(
    model: &Model,
    grids: &MergedGrids,
    raw: &mut RawBank,
) -> Result<(), PbError> {
    let Some(bank) = &model.correction else {
        return Ok(());
    };
    fold_correction_bank_into_raw(model, grids, bank, raw)
}

/// The fold half above, but over an explicit `bank` rather than `model.correction`. Lets the
/// post-prune re-solve purify a stand-alone correction (`purify_correction`) without attaching
/// it to the model.
fn fold_correction_bank_into_raw(
    model: &Model,
    grids: &MergedGrids,
    bank: &crate::engine::CorrectionBank,
    raw: &mut RawBank,
) -> Result<(), PbError> {
    for table in &bank.tables {
        // Support = sorted raw feature ids of the corrected axes (green-spine: raw == axis).
        let mut u_ids: SmallVec<[FeatureId; MAX_ORDER]> = SmallVec::new();
        for &axis in &table.axes {
            let raw_id = model
                .provenance
                .get(axis as usize)
                .ok_or_else(|| PbError::Internal {
                    what: format!("correction axis {axis} absent from provenance"),
                })?
                .raw;
            if !u_ids.contains(&raw_id) {
                u_ids.push(raw_id);
            }
        }
        u_ids.sort_unstable();
        let u = FeatureSet(u_ids.clone());
        // The correction's merged shape must match this model's merged grid for the support.
        let mut extents = Vec::with_capacity(u_ids.len());
        for r in &u_ids {
            extents.push(grids.cells(*r)?);
        }
        if extents.len() != table.shape.len()
            || extents
                .iter()
                .zip(&table.shape)
                .any(|(&e, &s)| e != s as usize)
        {
            return Err(PbError::Internal {
                what: format!(
                    "correction shape {:?} != merged extents {extents:?} for {u:?}",
                    table.shape
                ),
            });
        }
        if !raw.tables.contains_key(&u) {
            let mut axes = Vec::with_capacity(u_ids.len());
            for r in &u_ids {
                axes.push(grids.axis_id(*r)?);
            }
            raw.tables.insert(
                u.clone(),
                RawTable {
                    u: u.clone(),
                    axes,
                    values: Tensor::try_zeros(extents.clone())?,
                },
            );
        }
        let rt = raw.tables.get_mut(&u).ok_or_else(|| PbError::Internal {
            what: "correction raw table vanished after insert".into(),
        })?;
        // Add the raw delta cell-for-cell. `walk_extents` enumerates the merged cells in the
        // same row-major (last-axis-fastest) order the `values` vector was built in.
        let mut flat = 0usize;
        walk_extents(&extents, |tuple| {
            let v = *table.values.get(flat).ok_or_else(|| PbError::Internal {
                what: "correction flat index escaped values during fold".into(),
            })?;
            flat += 1;
            rt.values.add(tuple, v)
        })?;
    }
    Ok(())
}

/// Purify a stand-alone §G1 cell correction `δ` into a fANOVA bank WITHOUT re-integrating the
/// model's trees. `purify` is linear in the raw effects at fixed weights, so
/// `purify(trees + δ) = purify(trees) + purify(δ)`; the post-prune re-solve exploits this to
/// recalibrate the kept tables by adding `purify(δ)` to the already-built deploy bank, paying a
/// cheap purify over `δ`'s handful of supports instead of a second (8-bag) `explain_bank`. `δ`
/// spans only mains + pairs, so there is no factored order-3 work. The returned bank is
/// **un-anchored** (`f0` is `δ`'s grand mean); the caller adds it to the deploy bank and
/// re-anchors once.
pub(crate) fn purify_correction(
    model: &Model,
    x: &ServeBinnedMatrix,
    w: RefMeasure,
    mass: Option<&[f32]>,
    correction: &crate::engine::CorrectionBank,
) -> Result<TableBank, PbError> {
    let grids = MergedGrids::from_model(model)?;
    let mut raw = RawBank {
        f0: 0.0,
        tables: BTreeMap::new(),
    };
    fold_correction_bank_into_raw(model, &grids, correction, &mut raw)?;
    // `mass` must be the same per-row mass the base bank was built under, or the δ bank
    // would be purified against a different measure than the tables it is added to.
    let weights = build_weights(x, &grids, &w, mass)?;
    purify(raw, &weights, &grids, PurifyMode::SinglePass)
}

/// In-place `base += delta`, cell-for-cell over matching supports (plus intercepts). `delta`
/// (a `purify_correction` result) shares `base`'s merged-grid cell layout, so every delta
/// table lands on a `base` table of identical shape; a delta support absent from `base` (should
/// not happen for a downward-closed keep-set) is appended verbatim. Scoring is linear, so the
/// summed bank scores exactly `base(x) + delta(x)`.
pub(crate) fn add_bank_assign(base: &mut TableBank, delta: &TableBank) -> Result<(), PbError> {
    let w = base.w.clone();
    let weights = build_weights_from_support(base, &w)?;
    base.f0 += delta.f0;
    let mut touched: Vec<FeatureSet> = Vec::with_capacity(delta.tables.len());
    for dt in &delta.tables {
        if let Some(bt) = base.tables.iter_mut().find(|t| t.u == dt.u) {
            if bt.values.len() != dt.values.len() {
                return Err(PbError::Internal {
                    what: format!(
                        "add_bank_assign shape mismatch for {:?}: {} vs {}",
                        dt.u,
                        bt.values.len(),
                        dt.values.len()
                    ),
                });
            }
            let extents = bt.values.shape();
            let dv = dt.values.values();
            let mut flat = 0usize;
            walk_extents(&extents, |tuple| {
                let v = *dv.get(flat).ok_or_else(|| PbError::Internal {
                    what: "add_bank_assign delta flat index escaped values".into(),
                })?;
                flat += 1;
                bt.values.add(tuple, v)
            })?;
            touched.push(dt.u.clone());
        } else {
            // `dt` (from `purify_correction`) already carries the correct variance for a
            // brand-new table: its served value IS δ alone (no prior `base` counterpart to
            // sum with), computed under the same `table_variance` this function recomputes
            // for the summed tables below.
            base.tables.push(dt.clone());
        }
    }
    // `base += delta` is a cell-wise sum of two PURE tables (delta is purify_correction's
    // output, itself already zero-mean under `w`), so the sum stays structurally pure — no
    // re-purify needed here, unlike graduation's GCV smoothing. But `EffectTable.variance`
    // (the cached w-weighted σ²(f_u) that `sobol()` and the rating export read) is a function
    // of `values`, which just changed, so it goes stale exactly like any other cache would.
    for table in &mut base.tables {
        if touched.contains(&table.u) {
            table.variance = table_variance(&table.u, &table.values, &weights)?;
        }
    }
    Ok(())
}

/// Build a zero-valued cell-basis correction scaffold for `supports` (each a sorted list
/// of model axis ids of size 1..=3), at the model's merged-grid resolution. Fills `shape`
/// (merged cells per axis) and `bin_to_cell` (model bin → merged cell, the exact forward
/// map predict uses); leaves `values` zeroed for the §G1 solver to fill. Centralises the
/// merged-grid logic so predict, the decompose fold, and the solver share one cell layout.
///
/// # Errors
/// [`PbError::Internal`] if an axis is absent from the model or a bin/cell exceeds `u8`/`u32`.
pub(crate) fn correction_scaffold(
    model: &Model,
    supports: &[Vec<u32>],
) -> Result<crate::engine::CorrectionBank, PbError> {
    let grids = MergedGrids::from_model(model)?;
    let mut tables = Vec::with_capacity(supports.len());
    for axes in supports {
        let mut shape = Vec::with_capacity(axes.len());
        let mut bin_to_cell = Vec::with_capacity(axes.len());
        for &axis in axes {
            let raw = model
                .provenance
                .get(axis as usize)
                .ok_or_else(|| PbError::Internal {
                    what: format!("correction scaffold axis {axis} absent from provenance"),
                })?
                .raw;
            let n_cells = grids.cells(raw)?;
            shape.push(u32::try_from(n_cells).map_err(|_| PbError::Internal {
                what: "merged cell count exceeded u32".into(),
            })?);
            let grid = model
                .grids
                .get(axis as usize)
                .ok_or_else(|| PbError::Internal {
                    what: format!("correction scaffold axis {axis} absent from grids"),
                })?;
            let axis_merged = grids.axis(raw)?;
            let mut map = Vec::with_capacity(usize::from(grid.n_bins));
            for bin in 0..grid.n_bins {
                let b = u8::try_from(bin).map_err(|_| PbError::Internal {
                    what: "model bin exceeded u8 in correction scaffold".into(),
                })?;
                map.push(axis_merged.model_bin_to_cell(b)?);
            }
            bin_to_cell.push(map);
        }
        let cells: usize = shape.iter().map(|&s| s as usize).product();
        tables.push(crate::engine::CorrectionTable {
            axes: axes.clone(),
            shape,
            bin_to_cell,
            values: vec![0.0; cells],
        });
    }
    Ok(crate::engine::CorrectionBank { tables })
}

/// The pre-purify exact-accumulation checkpoint (spec §08.2): `f0 + Σ_u T_raw[u](x) ==
/// F_ens(x)` identically at every realized-support cell, before any purification runs.
/// A failure is an accumulation bug, not an invariant violation, so it surfaces as
/// [`PbError::Internal`].
fn verify_raw_accumulation(
    model: &Model,
    raw: &RawBank,
    factored: &[FactoredEffect],
    grids: &MergedGrids,
) -> Result<(), PbError> {
    let tol = ExactTol::for_model(model).recon_tol;
    // Reuse the joint enumerator over the raw bank's realized features (+ factored supports).
    let mut set: BTreeSet<FeatureId> = model_features(model)?.into_iter().collect();
    for u in raw.tables.keys() {
        for r in &u.0 {
            set.insert(*r);
        }
    }
    for ft in factored {
        for r in &ft.u.0 {
            set.insert(*r);
        }
    }
    let feats: Vec<FeatureId> = set.into_iter().collect();
    enumerate_check_points(grids, &feats, |x_cells, rep_bins| {
        let ens = model.ensemble_f64(rep_bins)?;
        let mut acc = raw.f0;
        for rt in raw.tables.values() {
            let mut coord = Vec::with_capacity(rt.u.order());
            for a in &rt.axes {
                let cell = *x_cells
                    .get(a.raw.0 as usize)
                    .ok_or_else(|| PbError::Internal {
                        what: "raw checkpoint x_cells missing raw".into(),
                    })?;
                coord.push(cell as usize);
            }
            acc += rt.values.at(&coord).ok_or_else(|| PbError::Internal {
                what: "raw checkpoint coord out of range".into(),
            })?;
        }
        for ft in factored {
            acc += ft.eval(x_cells)?;
        }
        if (ens - acc).abs() > tol {
            return Err(PbError::Internal {
                what: "raw accumulation does not reconstruct the ensemble".into(),
            });
        }
        Ok(())
    })?;
    Ok(())
}

impl TableBank {
    /// `f0 + Σ_u f_u(x_u)` — the lossless LUT-sum score, equal to `F_ens` (spec §08.8).
    /// `x_cells[raw]` is the row's merged cell id on each raw feature.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`]/[`PbError::Internal`] if a table's axis is missing from
    /// `x_cells` or a coordinate escapes its tensor.
    pub fn score(&self, x_cells: &[u32]) -> Result<f64, PbError> {
        let mut acc = self.f0;
        for t in &self.tables {
            acc += t.eval(x_cells)?;
        }
        for ft in &self.factored {
            acc += ft.eval(x_cells)?;
        }
        Ok(acc)
    }

    /// `f0 + Σ_u f_u(x_u)` for every row of `x`, in `f64` — the LUT-sum score of the WHOLE
    /// bank, dense tables and factored effects alike.
    ///
    /// [`Self::score`] is the per-row form, but building its `x_cells` argument requires the
    /// bin→merged-cell maps, which are crate-internal; and [`crate::TableScoringBank`], the
    /// batch scorer that owns those maps, deliberately REFUSES a bank carrying factored
    /// effects rather than silently scoring it short. Between them a caller outside this
    /// crate had no way at all to evaluate a factored bank in full precision — which is
    /// awkward for a product whose central claim is "the tables ARE the model", and actively
    /// blocking above order 4, where every interaction effect is factored by construction.
    ///
    /// This is that missing path, and it is the one to compare against
    /// [`crate::Model::ensemble_f64`] when checking losslessness: both are `f64`, so the
    /// agreement is real rather than rounded to `f32` on the way out.
    ///
    /// `cat_encoders` are the frozen encoders the bank was purified against
    /// (`model.schema.cat_encoders`); pass [`crate::cat::CatEncoderStore::new`] for a model
    /// with no categoricals.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if `out.len() != x.n_rows`; propagates cell-map
    /// construction and per-table evaluation failures.
    pub fn score_binned(
        &self,
        cat_encoders: &crate::cat::CatEncoderStore,
        x: &crate::data::BinnedMatrix,
        out: &mut [f64],
    ) -> Result<(), PbError> {
        let n_rows = x.n_rows as usize;
        if out.len() != n_rows {
            return Err(PbError::ShapeMismatch {
                what: format!("score_binned out len {} != n_rows {n_rows}", out.len()),
            });
        }
        let maps = crate::scoring::build_cell_maps(&self.merged_grids, cat_encoders, x)?;
        let n_cells = self.merged_grids.len();
        let mut cells = vec![0u32; n_cells];
        for (row, slot) in out.iter_mut().enumerate() {
            crate::scoring::fill_row_cells(x, &maps, row, &mut cells)?;
            *slot = self.score(&cells)?;
        }
        Ok(())
    }

    /// Measure joint component and model variances on the same aligned rows and mass.
    /// The model variance includes covariance between effects; its intercept is excluded
    /// from accumulation to preserve precision under harmless intercept translations.
    ///
    /// # Errors
    /// Returns a typed input error for empty or zero-mass rows and propagates cell-map,
    /// shape, mass validation and effect evaluation errors.
    pub fn measure_joint_variance(
        &mut self,
        cat_encoders: &crate::cat::CatEncoderStore,
        x: &crate::data::BinnedMatrix,
        mass: Option<&[f32]>,
    ) -> Result<(), PbError> {
        let n_rows = x.n_rows as usize;
        validate_mass(mass, n_rows, "joint variance")?;
        let total = mass.map_or(n_rows as f64, |w| w.iter().map(|v| f64::from(*v)).sum());
        if total <= 0.0 || !total.is_finite() {
            return Err(PbError::InvalidInput {
                what: "joint variance needs positive finite row mass".into(),
            });
        }
        let maps = crate::scoring::build_cell_maps(&self.merged_grids, cat_encoders, x)?;
        let mut cells = vec![0; self.merged_grids.len()];
        let count = self.tables.len() + self.factored.len();
        let mut means = vec![0.0; count + 1];
        let mut m2 = vec![0.0; count + 1];
        let mut seen = 0.0;
        for row in 0..n_rows {
            let weight = mass.and_then(|w| w.get(row)).map_or(1.0, |w| f64::from(*w));
            if weight == 0.0 {
                continue;
            }
            crate::scoring::fill_row_cells(x, &maps, row, &mut cells)?;
            seen += weight;
            let mut sum = 0.0;
            let values = self
                .tables
                .iter()
                .map(|t| t.eval(&cells))
                .chain(self.factored.iter().map(|t| t.eval(&cells)));
            for (index, value) in values.enumerate() {
                let value = value?;
                sum += value;
                let mean = means.get_mut(index).ok_or_else(|| PbError::Internal {
                    what: "joint moment index escaped".into(),
                })?;
                let moment = m2.get_mut(index).ok_or_else(|| PbError::Internal {
                    what: "joint moment index escaped".into(),
                })?;
                let delta = value - *mean;
                *mean += weight / seen * delta;
                *moment += weight * delta * (value - *mean);
            }
            let mean = means.last_mut().ok_or_else(|| PbError::Internal {
                what: "joint total moment missing".into(),
            })?;
            let moment = m2.last_mut().ok_or_else(|| PbError::Internal {
                what: "joint total moment missing".into(),
            })?;
            let delta = sum - *mean;
            *mean += weight / seen * delta;
            *moment += weight * delta * (sum - *mean);
        }
        for (table, moment) in self.tables.iter_mut().zip(&m2) {
            table.variance = (moment / total).max(0.0);
        }
        for (table, moment) in self
            .factored
            .iter_mut()
            .zip(m2.iter().skip(self.tables.len()))
        {
            table.variance = (moment / total).max(0.0);
        }
        self.joint_variance = m2.last().map(|moment| (moment / total).max(0.0));
        Ok(())
    }

    /// Exact interventional Shapley values `φ_i(x) = Σ_{u ∋ i} f_u(x_u)/|u|` (spec §08.5),
    /// indexed by raw feature id. Sums to `score(x) − f0`. O(#tables) table reads, zero
    /// model calls.
    ///
    /// # Errors
    /// Propagates any [`EffectTable::eval`] failure.
    pub fn shap(&self, x_cells: &[u32]) -> Result<Vec<f64>, PbError> {
        let mut phi = vec![0.0_f64; self.merged_grids.len()];
        for t in &self.tables {
            let order = t.u.order().max(1) as f64;
            let share = t.eval(x_cells)? / order;
            for r in &t.u.0 {
                *phi.get_mut(r.0 as usize).ok_or_else(|| PbError::Internal {
                    what: "shap raw index escaped phi".into(),
                })? += share;
            }
        }
        for ft in &self.factored {
            let order = ft.u.order().max(1) as f64;
            let share = ft.eval(x_cells)? / order;
            for r in &ft.u.0 {
                *phi.get_mut(r.0 as usize).ok_or_else(|| PbError::Internal {
                    what: "shap raw index escaped phi".into(),
                })? += share;
            }
        }
        Ok(phi)
    }

    /// Sobol importances `S_u = σ²(f_u)/σ²(F)` from the cached table variances (spec
    /// §08.5), sorted descending. Under product/uniform `w` they sum to ~1. Joint
    /// shares require [`Self::measure_joint_variance`]; otherwise no shares are returned.
    #[must_use]
    pub fn sobol(&self) -> Vec<(FeatureSet, f64)> {
        let total: f64 = if self.w == RefMeasure::Joint {
            let Some(total) = self.joint_variance else {
                return Vec::new();
            };
            total
        } else {
            self.tables.iter().map(|t| t.variance).sum::<f64>()
                + self.factored.iter().map(|ft| ft.variance).sum::<f64>()
        };
        let mut out: Vec<(FeatureSet, f64)> = self
            .tables
            .iter()
            .map(|t| (t.u.clone(), t.variance))
            .chain(self.factored.iter().map(|ft| (ft.u.clone(), ft.variance)))
            .map(|(u, v)| {
                let s = if total > 0.0 { v / total } else { 0.0 };
                (u, s)
            })
            .collect();
        // Sobol-descending with the feature set as an explicit secondary key, so the
        // ranking is total and stable regardless of table insertion order.
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out
    }

    /// The reference measure stamped on this bank (spec §08.8).
    #[must_use]
    pub fn reference_measure(&self) -> &RefMeasure {
        &self.w
    }

    /// Recompute the tables under a different reference measure `w` without retraining
    /// (spec §08.8) — exactness-preserving: the sum is conserved (Lengerich Cor. 2.2), so
    /// the bank still reconstructs `F_ens` and the model stays `Exact`. Re-purifies the
    /// current tables (which sum to `F_ens`) under the new `w`, carrying the per-cell
    /// `support` over unchanged.
    ///
    /// FLAG (spec §08.8 reconciliation): the spec signature passes a `ServeBinnedMatrix`,
    /// but the bank's cached `support` IS the merged-grid empirical marginal already, so
    /// v1 derives `w` from it and needs no serve matrix (and so no model) here.
    ///
    /// # Errors
    /// [`PbError::InvalidConfig`] for the v1-unsupported `Joint` measure; plus propagated
    /// grid/purify errors. [`PbError::InvalidInput`] when pruning or banding removed
    /// the support needed to identify the requested marginal; use [`Self::recentre_on`]
    /// with aligned rows and mass in that case.
    pub fn recompute_under(&self, w: RefMeasure) -> Result<TableBank, PbError> {
        self.recompute_under_with(w, None)
    }

    /// The same function re-centred on other rows: every table's `support` refilled from
    /// `row_cells` (per raw feature, each row's merged cell) and `mass` (per row; `None` counts
    /// rows), then the bank re-purified under `w` with each axis's empirical marginal taken from
    /// those rows. The table sum on every cell — so every prediction — is unchanged; only how it
    /// is shared between tables moves, exactly as when the bank was first purified on the
    /// training rows.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if `row_cells`/`mass` disagree with the bank or each other;
    /// [`PbError::InvalidInput`] on a negative or non-finite mass; propagated purify errors.
    pub fn recentre_on(
        &self,
        row_cells: &[Vec<u32>],
        mass: Option<&[f32]>,
        w: RefMeasure,
    ) -> Result<TableBank, PbError> {
        let n_raw = self.merged_grids.len();
        if row_cells.len() != n_raw {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "{} cell columns for a bank of {n_raw} raw feature(s)",
                    row_cells.len()
                ),
            });
        }
        let n_rows = row_cells.first().map_or(0, Vec::len);
        validate_mass(mass, n_rows, "recentre")?;
        let row_mass =
            |row: usize| -> f64 { mass.and_then(|m| m.get(row)).map_or(1.0, |&v| f64::from(v)) };
        let mut marginals = Vec::with_capacity(n_raw);
        for (cells, grid) in row_cells.iter().zip(&self.merged_grids) {
            if cells.len() != n_rows {
                return Err(PbError::ShapeMismatch {
                    what: "recentre cell columns have different row counts".into(),
                });
            }
            let mut counts = vec![0.0_f64; usize::from(grid.n_bins)];
            for (row, &cell) in cells.iter().enumerate() {
                *counts
                    .get_mut(cell as usize)
                    .ok_or_else(|| PbError::ShapeMismatch {
                        what: "recentre cell outside its merged grid".into(),
                    })? += row_mass(row);
            }
            marginals.push(counts);
        }
        let mut base = self.recompute_under_with(w, Some(&marginals))?;
        for table in &mut base.tables {
            table.support = Tensor::try_zeros(table.values.shape())?;
            let mut coord = vec![0usize; table.axes.len()];
            for row in 0..n_rows {
                for (slot, axis) in coord.iter_mut().zip(&table.axes) {
                    let cell = row_cells
                        .get(axis.raw.0 as usize)
                        .and_then(|c| c.get(row))
                        .ok_or_else(|| PbError::Internal {
                            what: "recentre support lost a raw feature".into(),
                        })?;
                    *slot = axis.coord(*cell).ok_or_else(|| PbError::Internal {
                        what: "recentre cell outside its band map".into(),
                    })?;
                }
                table.support.add(&coord, row_mass(row))?;
            }
        }
        Ok(base)
    }

    /// [`Self::recompute_under`] with each axis's empirical marginal from `marginals` (see
    /// [`Self::recentre_on`]) instead of the main-effect supports.
    pub(crate) fn recompute_under_with(
        &self,
        w: RefMeasure,
        marginals: Option<&[Vec<f64>]>,
    ) -> Result<TableBank, PbError> {
        if self
            .tables
            .iter()
            .any(|t| t.axes.iter().any(|a| a.band_of.is_some()))
        {
            return crate::banding::repurify_bank_under_with(self, &w, marginals);
        }
        let grids = MergedGrids::from_border_grids(&self.merged_grids);
        let axis_templates: BTreeMap<FeatureId, AxisId> = self
            .tables
            .iter()
            .flat_map(|t| &t.axes)
            .chain(self.factored.iter().flat_map(|f| &f.axes))
            .map(|axis| (axis.raw, axis.clone()))
            .collect();
        let weights = build_weights_from_support_with(self, &w, marginals)?;
        // Seed a RawBank from the current (already-purified) dense tables: together with the
        // factored effects they sum to F_ens, a valid input to purify under the new measure.
        let mut tables: BTreeMap<FeatureSet, RawTable> = BTreeMap::new();
        for t in &self.tables {
            tables.insert(
                t.u.clone(),
                RawTable {
                    u: t.u.clone(),
                    axes: t.axes.clone(),
                    values: t.values.clone(),
                },
            );
        }
        // Factored (boxed) effects: re-shed every existing box under the NEW measure, in
        // descending order exactly as the model-side driver does (`explain_bank_with_grids_and_mass`),
        // so an order-3 effect collects every order-4 deposit before it centres. The boxes are
        // already merged, so no re-merge unless a higher order deposited onto the support.
        // Axes are copied from the stored effect (not re-derived from the self-describing
        // grids, whose joint-axis placeholders carry padded border values — bug #6).
        let mut factored = Vec::with_capacity(self.factored.len());
        let mut deposits: BTreeMap<FeatureSet, Vec<RankOneBox>> = BTreeMap::new();
        for order in (LEGACY_MAX_ORDER..=MAX_ORDER).rev() {
            let mut at_order: BTreeSet<FeatureSet> = self
                .factored
                .iter()
                .filter(|f| f.u.order() == order)
                .map(|f| f.u.clone())
                .collect();
            at_order.extend(deposits.keys().filter(|u| u.order() == order).cloned());
            for u in at_order {
                let inherited = deposits.remove(&u).unwrap_or_default();
                let any_lifted = !inherited.is_empty();
                let mut raw_boxes = inherited;
                let existing = self.factored.iter().find(|f| f.u == u);
                if let Some(f) = existing {
                    raw_boxes.extend(f.boxes.iter().map(|b| (b.p.clone(), b.low.clone())));
                }
                let mut per_axis_w = Vec::with_capacity(u.order());
                for r in &u.0 {
                    per_axis_w.push(weights.axis(*r)?.to_vec());
                }
                let axes: Vec<AxisId> = match existing {
                    Some(f) => f.axes.clone(),
                    None => {
                        u.0.iter()
                            .map(|r| {
                                axis_templates
                                    .get(r)
                                    .cloned()
                                    .map(Ok)
                                    .unwrap_or_else(|| grids.axis_id(*r))
                            })
                            .collect::<Result<_, _>>()?
                    }
                };
                factored.push(shed_boxes_into_raw(
                    &u,
                    per_axis_w,
                    axes,
                    raw_boxes,
                    any_lifted,
                    &grids,
                    &mut tables,
                    &mut deposits,
                )?);
            }
        }
        if let Some((u, _)) = deposits.into_iter().next() {
            return Err(PbError::Internal {
                what: format!("recompute_under left an undelivered deposit for {u:?}"),
            });
        }
        for ft in &mut factored {
            ft.variance = ft.compute_variance()?;
        }
        let raw = RawBank {
            f0: self.f0,
            tables,
        };
        let mut bank = purify(raw, &weights, &grids, PurifyMode::SinglePass)?;
        bank.factored = factored;
        // Support is data-derived (w-independent), so carry it over by support key.
        for t in &mut bank.tables {
            for axis in &mut t.axes {
                if let Some(template) = axis_templates.get(&axis.raw) {
                    *axis = template.clone();
                }
            }
            if let Some(support) = support_for_axes(&self.tables, &t.axes)? {
                t.support = support;
            }
        }
        // Same restoration `purify_raw_effects` applies, and for the same reason: `purify`
        // re-derives `merged_grids` via `grids.border_grids()`, which round-trips a joint
        // axis's padded-for-`cells()` placeholder (`from_border_grids`) back out as non-empty
        // borders — the wrong shape for storage (bug #6). `purify` never changes grid shape,
        // only cell values, so `self.merged_grids` is still exactly correct for every axis.
        bank.merged_grids = self.merged_grids.clone();
        Ok(bank)
    }
}

impl MergedGrids {
    /// Reconstruct merged grids from a bank's stored per-feature [`BorderGrid`]s (the
    /// self-describing grid: cell boundaries ARE the borders, so the model-border index
    /// is `0..borders.len()`). Used by [`TableBank::recompute_under`], where no model is
    /// in hand. Only the borders/cells are consulted downstream (purify + support-derived
    /// weights), so the indices need only be internally consistent.
    pub(crate) fn from_border_grids(grids: &[BorderGrid]) -> Self {
        // A P1 multi-channel raw feature's `BorderGrid` entry is a placeholder (empty
        // borders, `n_bins` = its true joint cell count — see `MergedGrids::border_grids`'s
        // doc), so the reconstructed `MergedAxis` here is never `joint: Some(..)` — it cannot
        // be, without a model to re-derive channels/encoders from. `cells()` (`borders.len() +
        // 2` when `joint` is `None`) would then silently return 2 instead of the true joint
        // cell count unless `borders` is padded to match here (bug #6): the padding VALUES are
        // never read for anything downstream of THIS reconstruction (purify's own body only
        // ever calls `.cells()` for shape, never reads border values; the only border-VALUE
        // readers anywhere in this module are factored-triple export, and factored effects
        // never reach `RawBank` — `recompute_under` explicitly rejects them and graduation's
        // `graduation_tables` never offers order-3 payloads), so only the correct LENGTH
        // matters here. The bank-level `merged_grids` this feeds into `purify()` gets restored
        // to the caller's own original (canonical, empty-borders) grids on the way back out —
        // see `purify_raw_effects`/`recompute_under` — so this padding never leaks into a
        // stored bank; it exists purely to size `purify()`'s internal tensors correctly.
        let per_raw: Vec<MergedAxis> = grids
            .iter()
            .enumerate()
            .map(|(r, g)| {
                let is_joint_placeholder = g.borders.is_empty() && g.n_bins > 2;
                let borders = if is_joint_placeholder {
                    (0..u32::from(g.n_bins) - 2).map(|i| i as f32).collect()
                } else {
                    g.borders.clone()
                };
                MergedAxis {
                    axis: r,
                    model_border_index: (0..borders.len()).collect(),
                    borders,
                    model_n_bins: g.n_bins,
                    joint: None,
                }
            })
            .collect();
        let n_axes = per_raw.len();
        MergedGrids { per_raw, n_axes }
    }
}

fn validate_bank_grid(grid: &BorderGrid, raw: usize) -> Result<(), PbError> {
    if grid.missing_bin != 0 {
        return Err(PbError::InvalidInput {
            what: format!("bank merged grid {raw} missing_bin must be 0"),
        });
    }
    // A P1 multi-channel joint categorical axis's `BorderGrid` is the placeholder
    // `MergedGrids::border_grids` documents: `borders` empty (a joint axis's cell space has
    // no single scalar dimension to derive real split thresholds from), `n_bins` = the true
    // joint cell count. This function has no model/provenance in hand (it validates a bare
    // `merged_grids: Vec<BorderGrid>`, the same "no model" seam `MergedGrids::from_border_grids`
    // serves — see its doc), so it cannot cross-check `is_joint` against provenance the way
    // `table_model.rs`'s `validate_merged_grid` does (that fix, bug #4); it infers the same
    // placeholder shape from the numbers instead, exactly as `from_border_grids` already does.
    // A genuine non-joint axis can never legitimately reach `n_bins > 2` with zero borders
    // (every realized border pushes `borders.len()` up by exactly one, so `n_bins <= 2` is the
    // only way to have zero of them there), so this is an unambiguous signature bug #6
    // surfaced: graduation is the first caller of this exact validate-then-reconstruct seam to
    // run on a joint axis (it's objective-gated: poisson/gamma + `graduate=True`), so the gap
    // was invisible on every earlier multi-channel gate, which used a non-graduating objective.
    let is_joint_placeholder = grid.borders.is_empty() && grid.n_bins > 2;
    if !is_joint_placeholder {
        let expected = grid
            .borders
            .len()
            .checked_add(2)
            .ok_or_else(|| PbError::Internal {
                what: "bank merged grid border count overflow".into(),
            })?;
        if usize::from(grid.n_bins) != expected {
            return Err(PbError::InvalidInput {
                what: format!(
                    "bank merged grid {raw} n_bins {} inconsistent with {} borders",
                    grid.n_bins,
                    grid.borders.len()
                ),
            });
        }
    }
    for (i, &border) in grid.borders.iter().enumerate() {
        if !border.is_finite() {
            return Err(PbError::InvalidInput {
                what: format!("bank merged grid {raw} border {i} must be finite"),
            });
        }
    }
    for pair in grid.borders.windows(2) {
        if let [a, b] = pair {
            if a >= b {
                return Err(PbError::InvalidInput {
                    what: format!("bank merged grid {raw} borders must be strictly ascending"),
                });
            }
        }
    }
    Ok(())
}

fn effect_extents(grids: &MergedGrids, u: &FeatureSet) -> Result<Vec<usize>, PbError> {
    let mut extents = Vec::with_capacity(u.order());
    for raw in &u.0 {
        extents.push(grids.cells(*raw)?);
    }
    Ok(extents)
}

fn validate_raw_effect(grids: &MergedGrids, effect: &RawEffect) -> Result<(), PbError> {
    if !(1..=MAX_ORDER).contains(&effect.u.order()) {
        return Err(PbError::InvalidInput {
            what: format!(
                "raw effect order {} outside 1..={MAX_ORDER}",
                effect.u.order()
            ),
        });
    }
    let extents = effect_extents(grids, &effect.u)?;
    if effect.values.shape() != extents {
        return Err(PbError::ShapeMismatch {
            what: "raw effect values shape does not match merged grid".into(),
        });
    }
    if effect.support.shape() != extents {
        return Err(PbError::ShapeMismatch {
            what: "raw effect support shape does not match merged grid".into(),
        });
    }
    Ok(())
}

fn marginal_counts_from_effect(
    effect: &RawEffect,
    raw: FeatureId,
    cells: usize,
) -> Result<Option<Vec<f64>>, PbError> {
    let Some(pos) = effect.u.0.iter().position(|r| *r == raw) else {
        return Ok(None);
    };
    let mut counts = vec![0.0_f64; cells];
    let extents = effect.support.shape();
    walk_extents(&extents, |coord| {
        let cell = *coord.get(pos).ok_or_else(|| PbError::Internal {
            what: "support marginal position escaped coordinate".into(),
        })?;
        let value = effect.support.at(coord).ok_or_else(|| PbError::Internal {
            what: "support marginal coordinate out of range".into(),
        })?;
        let slot = counts.get_mut(cell).ok_or_else(|| PbError::Internal {
            what: "support marginal cell escaped counts".into(),
        })?;
        *slot += value;
        Ok(())
    })?;
    let total: f64 = counts.iter().sum();
    if total > 0.0 {
        Ok(Some(counts))
    } else {
        Ok(None)
    }
}

fn axis_counts_from_effects(
    effects: &[RawEffect],
    raw: FeatureId,
    cells: usize,
) -> Result<Vec<f64>, PbError> {
    let mut candidates: Vec<&RawEffect> = effects.iter().filter(|e| e.u.contains(raw)).collect();
    candidates.sort_by_key(|e| e.u.order());
    for effect in candidates {
        if let Some(counts) = marginal_counts_from_effect(effect, raw, cells)? {
            return Ok(counts);
        }
    }
    Ok(vec![0.0_f64; cells])
}

fn build_weights_from_effect_support(
    grids: &[BorderGrid],
    effects: &[RawEffect],
    w: &RefMeasure,
) -> Result<WeightCache, PbError> {
    let rule = axis_rule(w)?;

    let mut per_axis = Vec::with_capacity(grids.len());
    for (raw, grid) in grids.iter().enumerate() {
        validate_bank_grid(grid, raw)?;
        let cells = usize::from(grid.n_bins);
        let raw_w = if !rule.needs_counts() {
            rule.raw_weights(&[], 0.0, cells)
        } else {
            let raw_id = FeatureId(u32::try_from(raw).map_err(|_| PbError::InvalidInput {
                what: "raw feature index exceeds u32".into(),
            })?);
            let counts = axis_counts_from_effects(effects, raw_id, cells)?;
            let n_total: f64 = counts.iter().sum();
            let inv_n = if n_total > 0.0 { 1.0 / n_total } else { 0.0 };
            rule.raw_weights(&counts, inv_n, cells)
        };
        let total: f64 = raw_w.iter().sum();
        if total.is_nan() || total <= 0.0 {
            return Err(PbError::Internal {
                what: "reference-measure axis weights summed to zero".into(),
            });
        }
        per_axis.push(raw_w.iter().map(|x| x / total).collect());
    }
    Ok(WeightCache {
        per_axis,
        kind: w.clone(),
    })
}

/// Recover subset mass only when the surviving tensor identifies its cells exactly.
/// In particular, never invent a within-band mass distribution from a compressed axis.
pub(crate) fn support_for_axes(
    tables: &[EffectTable],
    axes: &[AxisId],
) -> Result<Option<Tensor>, PbError> {
    let source = tables
        .iter()
        .filter(|table| {
            axes.iter().all(|axis| {
                table.axes.iter().any(|a| {
                    a.raw == axis.raw && a.cells == axis.cells && a.band_of == axis.band_of
                })
            })
        })
        .min_by_key(|table| table.u.order());
    let Some(source) = source else {
        return Ok(None);
    };
    let positions = axes
        .iter()
        .map(|axis| {
            source
                .axes
                .iter()
                .position(|a| a.raw == axis.raw)
                .ok_or_else(|| PbError::Internal {
                    what: "support projection lost an axis".into(),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut support = Tensor::try_zeros(axes.iter().map(|axis| axis.cells as usize).collect())?;
    walk_extents(&source.support.shape(), |coord| {
        let subset = positions
            .iter()
            .map(|&p| {
                coord.get(p).copied().ok_or_else(|| PbError::Internal {
                    what: "support projection coordinate escaped".into(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mass = source.support.at(coord).ok_or_else(|| PbError::Internal {
            what: "support projection source escaped".into(),
        })?;
        support.add(&subset, mass)
    })?;
    Ok(Some(support))
}

fn support_for_subset(
    support_by_u: &BTreeMap<FeatureSet, Tensor>,
    target: &FeatureSet,
    grids: &MergedGrids,
) -> Result<Option<Tensor>, PbError> {
    let mut candidates: Vec<(&FeatureSet, &Tensor)> = support_by_u
        .iter()
        .filter(|(u, _)| target.0.iter().all(|raw| u.contains(*raw)))
        .collect();
    candidates.sort_by_key(|(u, _)| u.order());

    if let Some((source_u, support)) = candidates.into_iter().next() {
        if source_u == target {
            return Ok(Some(support.clone()));
        }
        let mut positions = Vec::with_capacity(target.order());
        for raw in &target.0 {
            let pos = source_u
                .0
                .iter()
                .position(|source_raw| source_raw == raw)
                .ok_or_else(|| PbError::Internal {
                    what: "support subset search lost a raw feature".into(),
                })?;
            positions.push(pos);
        }
        let extents = effect_extents(grids, target)?;
        let mut out = Tensor::try_zeros(extents)?;
        let source_extents = support.shape();
        walk_extents(&source_extents, |coord| {
            let mut target_coord = Vec::with_capacity(target.order());
            for pos in &positions {
                target_coord.push(*coord.get(*pos).ok_or_else(|| PbError::Internal {
                    what: "support subset coordinate escaped source".into(),
                })?);
            }
            let v = support.at(coord).ok_or_else(|| PbError::Internal {
                what: "support subset source coordinate out of range".into(),
            })?;
            out.add(&target_coord, v)
        })?;
        return Ok(Some(out));
    }
    Ok(None)
}

/// Purify raw score-space effects on an explicit merged grid, carrying support
/// metadata through for callers that build exact banks without a [`Model`]. `pub`: also the
/// re-purification seam for post-graduation banks (`t-boost-py`'s `apply_graduation_updates`)
/// — GCV smoothing mutates a table's own values without touching its neighbors' cascade
/// relationships, so feeding the (now impure) CURRENT table values back through here as if they
/// were raw is exactly equivalent to purifying the original raw effects (purify is linear), and
/// is the only reconstruction available once a `TableModel` has already dropped its trees.
pub fn purify_raw_effects(
    f0: f64,
    merged_grids: Vec<BorderGrid>,
    w: RefMeasure,
    effects: Vec<RawEffect>,
) -> Result<TableBank, PbError> {
    for (raw, grid) in merged_grids.iter().enumerate() {
        validate_bank_grid(grid, raw)?;
    }
    let grids = MergedGrids::from_border_grids(&merged_grids);
    let weights = build_weights_from_effect_support(&merged_grids, &effects, &w)?;
    let mut tables = BTreeMap::new();
    let mut support_by_u = BTreeMap::new();
    for effect in effects {
        validate_raw_effect(&grids, &effect)?;
        let mut axes = Vec::with_capacity(effect.u.order());
        for raw in &effect.u.0 {
            axes.push(grids.axis_id(*raw)?);
        }
        if tables
            .insert(
                effect.u.clone(),
                RawTable {
                    u: effect.u.clone(),
                    axes,
                    values: effect.values,
                },
            )
            .is_some()
        {
            return Err(PbError::InvalidInput {
                what: "duplicate raw effect support".into(),
            });
        }
        support_by_u.insert(effect.u, effect.support);
    }
    let raw = RawBank { f0, tables };
    let mut bank = purify(raw, &weights, &grids, PurifyMode::SinglePass)?;
    for table in &mut bank.tables {
        if let Some(support) = support_for_subset(&support_by_u, &table.u, &grids)? {
            table.support = support;
        }
    }
    // `purify` re-derives its OWN `merged_grids` via `grids.border_grids()`, which round-trips
    // a joint axis's padded-for-`cells()` placeholder (see `from_border_grids`) back out as
    // non-empty borders — the wrong shape for storage (bug #6). Restore the caller's own,
    // already-validated, canonical grids instead: `purify` never changes grid SHAPE, only cell
    // VALUES, so this is exactly the input this function was given, for every axis kind.
    bank.merged_grids = merged_grids;
    Ok(bank)
}

/// Build the per-axis cell weights from a bank's cached `support` tensors — the
/// merged-grid empirical marginal, identical to what [`build_weights`] computes from the
/// serve matrix (both count the same rows into the same cells). Lets `recompute_under`
/// change `w` without a serve matrix or the model.
/// The bank's per-raw-axis measure weights over merged cells (what `purify` uses), rebuilt
/// from the bank alone — for crate-internal passes (banding) that re-purify a deployed bank.
/// `marginals` (per raw feature, per merged cell), when given, replaces the main-effect supports
/// as each axis's empirical counts — see [`TableBank::recentre_on`].
pub(crate) fn bank_axis_measure_with(
    bank: &TableBank,
    marginals: Option<&[Vec<f64>]>,
) -> Result<Vec<Vec<f64>>, PbError> {
    Ok(build_weights_from_support_with(bank, &bank.w, marginals)?.per_axis)
}

fn build_weights_from_support(bank: &TableBank, w: &RefMeasure) -> Result<WeightCache, PbError> {
    build_weights_from_support_with(bank, w, None)
}

/// [`build_weights_from_support`], with `marginals` (per raw feature, one mass per merged cell)
/// replacing the main-effect supports as each axis's empirical counts when given.
fn build_weights_from_support_with(
    bank: &TableBank,
    w: &RefMeasure,
    marginals: Option<&[Vec<f64>]>,
) -> Result<WeightCache, PbError> {
    let rule = axis_rule(w)?;

    let grids = MergedGrids::from_border_grids(&bank.merged_grids);
    let mut per_axis = Vec::with_capacity(bank.merged_grids.len());
    for (r, g) in bank.merged_grids.iter().enumerate() {
        let cells = usize::from(g.n_bins);
        if cells == 0 {
            return Err(PbError::InvalidInput {
                what: format!("bank merged grid {r} has n_bins=0"),
            });
        }
        let raw_w = if !rule.needs_counts() {
            rule.raw_weights(&[], 0.0, cells)
        } else if let Some(marginals) = marginals {
            let counts = marginals.get(r).ok_or_else(|| PbError::ShapeMismatch {
                what: format!("no marginal for raw feature {r}"),
            })?;
            if counts.len() != cells {
                return Err(PbError::ShapeMismatch {
                    what: format!(
                        "raw feature {r} marginal has {} cells, grid {cells}",
                        counts.len()
                    ),
                });
            }
            let n_total: f64 = counts.iter().sum();
            let inv_n = if n_total > 0.0 { 1.0 / n_total } else { 0.0 };
            rule.raw_weights(counts, inv_n, cells)
        } else {
            {
                // The main-effect support tensor for {r} is the per-cell effective
                // w-mass (spec §08.7) — a flat row count when the bank was built
                // unweighted, Σ w·e otherwise; either way it's already the right
                // numerator for an empirical marginal, so this is agnostic to which.
                let axis = grids.axis_id(FeatureId(r as u32))?;
                let support = support_for_axes(&bank.tables, &[axis])?;
                let mut counts = vec![0.0_f64; cells];
                let mut n_total = 0.0_f64;
                if let Some(support) = support {
                    for c in 0..cells {
                        let v = support.at(&[c]).ok_or_else(|| PbError::Internal {
                            what: "support cell out of range for recompute weights".into(),
                        })?;
                        *counts.get_mut(c).ok_or_else(|| PbError::Internal {
                            what: "recompute weight cell escaped counts".into(),
                        })? = v;
                        n_total += v;
                    }
                } else if bank
                    .tables
                    .iter()
                    .any(|t| t.u.contains(FeatureId(r as u32)))
                    || bank
                        .factored
                        .iter()
                        .any(|t| t.u.contains(FeatureId(r as u32)))
                {
                    return Err(PbError::InvalidInput {
                        what: "stored support cannot recover this marginal; supply aligned rows and mass for recentering".into(),
                    });
                }
                let inv_n = if n_total > 0.0 { 1.0 / n_total } else { 0.0 };
                rule.raw_weights(&counts, inv_n, cells)
            }
        };
        let total: f64 = raw_w.iter().sum();
        if total.is_nan() || total <= 0.0 {
            return Err(PbError::Internal {
                what: "reference-measure axis weights summed to zero".into(),
            });
        }
        per_axis.push(raw_w.iter().map(|x| x / total).collect());
    }
    Ok(WeightCache {
        per_axis,
        kind: w.clone(),
    })
}

// ===========================================================================
// Hand-built fixtures (doc-hidden public; shared by unit + integration tests).
// ===========================================================================

fn fixture_grid() -> BorderGrid {
    BorderGrid {
        borders: vec![1.5],
        n_bins: 3,
        missing_bin: 0,
    }
}

fn fixture_schema() -> crate::engine::ModelSchema {
    use crate::cat::CatEncoderStore;
    use crate::loss::{Link, LossId, ObjectiveTag};
    crate::engine::ModelSchema {
        feature_names: vec!["x0".into(), "x1".into()],
        feature_kinds: vec![
            crate::data::AxisKind::Numeric,
            crate::data::AxisKind::Numeric,
        ],
        cat_encoders: CatEncoderStore::new(),
        class_labels: None,
        objective: ObjectiveTag {
            link: Link::Identity,
            loss: LossId::SquaredError,
            tweedie_rho: None,
        },
    }
}

/// A tiny exact model whose single depth-2 tree realizes
/// `g(1,1)=6, g(1,2)=2, g(2,1)=2, g(2,2)=0` (a genuine pairwise interaction).
#[doc(hidden)]
#[must_use]
pub fn fixture_model() -> Model {
    use crate::data::{AxisKind, AxisProvenance, FeatureId};
    use crate::engine::{ExactnessMode, ObliviousTree, Split};
    use crate::loss::Link;

    // Leaf index = bit0 | bit1<<1, bit = (bin <= bin_le). With bin_le = 1 and bins
    // {1,2}: bin1 → bit 1, bin2 → bit 0. So (2,2)→0→g=0, (1,2)→1→g=2, (2,1)→2→g=2,
    // (1,1)→3→g=6.
    let leaves = [0.0, 2.0, 2.0, 6.0, 0.0, 0.0, 0.0, 0.0];
    let tree = ObliviousTree {
        splits: vec![
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
        ],
        leaves: leaves.to_vec(),
        depth: 2,
    };
    Model {
        f0: 0.0,
        trees: vec![(1.0, tree)],
        grids: vec![fixture_grid(), fixture_grid()],
        provenance: vec![
            AxisProvenance {
                raw: FeatureId(0),
                kind: AxisKind::Numeric,
            },
            AxisProvenance {
                raw: FeatureId(1),
                kind: AxisKind::Numeric,
            },
        ],
        link: Link::Identity,
        mode: ExactnessMode::Exact,
        schema: fixture_schema(),
        schema_version: crate::serialize::SCHEMA_VERSION_UNLIFTED,
        correction: None,
        bag_spans: None,
        bag_intercepts: None,
        bag_in_bag: None,
        delta_step_gate: None,
        fit_report: None,
    }
}

/// A 2-axis serve matrix exercising all four `(b0, b1) ∈ {1,2}²` cells of
/// [`fixture_model`] (so the empirical product marginals are balanced).
#[doc(hidden)]
#[must_use]
pub fn fixture_serve() -> ServeBinnedMatrix {
    use crate::data::BinnedMatrix;
    ServeBinnedMatrix(BinnedMatrix {
        data: vec![vec![1, 1, 2, 2], vec![1, 2, 1, 2]],
        n_rows: 4,
        grids: vec![fixture_grid(), fixture_grid()],
        provenance: fixture_model().provenance,
    })
}

/// P1 multi-channel (design/multichannel-categoricals.md §4.3): the SAME genuine pairwise
/// interaction pattern as [`fixture_model`] (`g(1,1)=6, g(1,2)=2, g(2,1)=2, g(2,2)=0` — NOT
/// additively separable: an additive model would force `g(1,1) = g(1,2)+g(2,1)-g(2,2) = 4`,
/// not `6`), but both axes are CHANNELS OF THE SAME RAW FEATURE (`raw = FeatureId(0)`, encoder
/// ids 0 and 1) instead of two different raw features. This is the exact scenario the first
/// (rejected) collapse design got wrong: summing two independently-purified single-channel
/// tables can never reproduce a non-zero residual interaction, because neither table alone
/// carries it. The flattened joint-cell design must reproduce it exactly, since the "interaction"
/// here isn't a second raw feature at all — it's the true recorded behavior of ONE order-1 effect.
#[doc(hidden)]
#[must_use]
pub fn fixture_multichannel_model() -> Model {
    use crate::cat::{CatEncoder, CatEncoderStore, CatLevel, CatTarget, TsConfig, TsEncodingId};
    use crate::data::{AxisKind, AxisProvenance, FeatureId};
    use crate::engine::{ExactnessMode, ObliviousTree, Split};
    use crate::loss::Link;

    let leaves = [0.0, 2.0, 2.0, 6.0, 0.0, 0.0, 0.0, 0.0];
    let tree = ObliviousTree {
        splits: vec![
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
        ],
        leaves: leaves.to_vec(),
        depth: 2,
    };
    // FOUR levels, one per (mean_bin, count_bin) combination in {1,2}x{1,2} — a joint axis
    // only recognizes (bin0, bin1, ...) tuples that some FROZEN LEVEL actually produces, so
    // exercising all four corners of this hand-built tree requires one level per corner (level
    // name encodes which corner: first letter -> mean bin, second letter -> count bin). Encoded
    // VALUES don't matter here (only the frozen bins/splits do) — they exist so the encoder
    // store/schema are internally consistent, matching what a real 2-channel fit would freeze.
    let level = |label: &str, bin: u8, encoding: f32| CatLevel {
        label: label.to_owned(),
        members: vec![label.to_owned()],
        encoding,
        bin,
        weight: 1.0,
    };
    let mean_enc = CatEncoder {
        raw: FeatureId(0),
        id: TsEncodingId(0),
        levels: vec![
            level("aa", 1, 1.0),
            level("ab", 1, 1.1),
            level("ba", 2, 2.0),
            level("bb", 2, 2.1),
        ],
        base: 0.0,
        config: TsConfig::default(),
    };
    let count_enc = CatEncoder {
        raw: FeatureId(0),
        id: TsEncodingId(1),
        // `JointCatAxis::build` re-derives each level's bin from `encode_label` binned
        // against the shared `fixture_grid()` border (1.5), NOT from the `bin` field below
        // (which only matters for Fisher-bin bookkeeping elsewhere) — so these VALUES, not
        // the `bin` argument, are what fix each level's count channel bin: <=1.5 -> bin 1
        // (second letter 'a'), >1.5 -> bin 2 (second letter 'b').
        levels: vec![
            level("aa", 1, 0.1),
            level("ab", 2, 2.0),
            level("ba", 1, 0.2),
            level("bb", 2, 2.1),
        ],
        base: 0.0,
        config: TsConfig {
            target: CatTarget::Count,
            ..TsConfig::default()
        },
    };
    Model {
        f0: 0.0,
        trees: vec![(1.0, tree)],
        grids: vec![fixture_grid(), fixture_grid()],
        provenance: vec![
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
        ],
        link: Link::Identity,
        mode: ExactnessMode::Exact,
        schema: crate::engine::ModelSchema {
            feature_names: vec!["cat0_mean".into(), "cat0_count".into()],
            feature_kinds: vec![
                AxisKind::CategoricalTS {
                    encoding: TsEncodingId(0),
                },
                AxisKind::CategoricalTS {
                    encoding: TsEncodingId(1),
                },
            ],
            cat_encoders: CatEncoderStore::from_encoders(vec![mean_enc, count_enc]),
            class_labels: None,
            objective: crate::loss::ObjectiveTag {
                link: Link::Identity,
                loss: crate::loss::LossId::SquaredError,
                tweedie_rho: None,
            },
        },
        schema_version: crate::serialize::SCHEMA_VERSION_UNLIFTED,
        correction: None,
        bag_spans: None,
        bag_intercepts: None,
        bag_in_bag: None,
        delta_step_gate: None,
        fit_report: None,
    }
}

/// A 2-axis serve matrix exercising all four `(b0, b1) ∈ {1,2}²` cells of
/// [`fixture_multichannel_model`], mirroring [`fixture_serve`].
#[doc(hidden)]
#[must_use]
pub fn fixture_multichannel_serve() -> ServeBinnedMatrix {
    use crate::data::BinnedMatrix;
    ServeBinnedMatrix(BinnedMatrix {
        data: vec![vec![1, 1, 2, 2], vec![1, 2, 1, 2]],
        n_rows: 4,
        grids: vec![fixture_grid(), fixture_grid()],
        provenance: fixture_multichannel_model().provenance,
    })
}

/// A model whose tree spans `MAX_ORDER + 1` DISTINCT raw features — an I1 violation.
///
/// **What this fixture proves has changed twice, and the honest statement matters.** Before
/// the depth lift it was a depth violation. After the depth lift (`MAX_DEPTH = 6` >
/// `MAX_ORDER = 4`) it was cleanly an ORDER violation at a legal depth, which is the sharper
/// claim. After the high-order lift `MAX_ORDER == MAX_DEPTH == 8`, so `MAX_ORDER + 1 = 9`
/// distinct raws needs 9 levels and trips the DEPTH clause of `i1_shape_ok` first — the two
/// clauses now coincide and no fixture can separate them (see
/// `i1_order_clause_is_subsumed_at_equal_caps`, which asserts that coincidence rather than
/// letting it rot silently).
///
/// It remains a correct negative for `check_feature_budget`, which is all it is used for:
/// the tree IS over budget, `check_feature_budget` must refuse it, and the fixture stays
/// written against the constants so it stays over budget wherever the caps land.
#[doc(hidden)]
#[must_use]
pub fn fixture_over_budget_model() -> Model {
    use crate::data::{AxisKind, AxisProvenance, FeatureId};
    use crate::engine::{ExactnessMode, ObliviousTree, Split};
    use crate::loss::Link;

    // One distinct raw feature more than the ORDER cap allows, one level each. Written
    // against `MAX_ORDER` so the fixture stays over budget when the cap moves.
    let n_raw = u32::try_from(crate::engine::MAX_ORDER).unwrap_or(u32::MAX) + 1;
    let depth = n_raw as usize;
    let splits = (0..n_raw)
        .map(|a| Split {
            axis: a,
            bin_le: 1,
            missing_left: false,
        })
        .collect();
    let provenance = (0..n_raw)
        .map(|a| AxisProvenance {
            raw: FeatureId(a),
            kind: AxisKind::Numeric,
        })
        .collect();
    let tree = ObliviousTree {
        splits,
        // The right `leaf_slots`, so the fixture is malformed ONLY in the way it is
        // meant to be (an over-budget DISTINCT-raw count, not a mis-framed leaf array).
        leaves: vec![0.0; crate::engine::leaf_slots(depth)],
        depth: depth as u8,
    };
    Model {
        f0: 0.0,
        trees: vec![(1.0, tree)],
        grids: vec![],
        provenance,
        link: Link::Identity,
        mode: ExactnessMode::Approximate {
            reason: "deliberately over-budget fixture".into(),
        },
        schema: fixture_schema(),
        schema_version: crate::serialize::SCHEMA_VERSION_UNLIFTED,
        correction: None,
        bag_spans: None,
        bag_intercepts: None,
        bag_in_bag: None,
        delta_step_gate: None,
        fit_report: None,
    }
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

    /// `ExposureMarginals` per-axis weights are `share + floor/cells`, normalized — the
    /// empirical marginal with a floor, NOT the half-uniform Laplace blend. Same fixture and
    /// mass as `build_weights_product_marginals_uses_hand_computed_mass_not_row_count`.
    #[test]
    fn exposure_marginals_weights_are_share_plus_floor() {
        let model = fixture_model();
        let x = fixture_serve();
        let grids = MergedGrids::from_model(&model).unwrap();
        let mass = [10.0_f32, 20.0, 30.0, 40.0];
        let floor = 1e-3_f32;
        let w = build_weights(
            &x,
            &grids,
            &RefMeasure::ExposureMarginals { floor },
            Some(&mass),
        )
        .unwrap();
        let fl = f64::from(floor);
        let unif = 1.0_f64 / 3.0;
        for (axis, cell_mass) in [(0usize, [0.0_f64, 30.0, 70.0]), (1, [0.0, 40.0, 60.0])] {
            let raw: Vec<f64> = cell_mass.iter().map(|m| m / 100.0 + fl * unif).collect();
            let total: f64 = raw.iter().sum();
            let expected: Vec<f64> = raw.iter().map(|v| v / total).collect();
            let got = w.axis(FeatureId(axis as u32)).unwrap();
            for (g, e) in got.iter().zip(&expected) {
                assert!(
                    (g - e).abs() < 1e-15,
                    "axis {axis}: {got:?} vs {expected:?}"
                );
            }
            // The empty missing cell carries ONLY the floor: a thousandth of the mass shared
            // over three cells, not half the axis as under `laplace = 1`.
            assert!(
                got[0] < 1e-3,
                "missing cell weight {} should be ~floor",
                got[0]
            );
        }
    }

    /// The exposure measure is axis-factorized, so the full five-gate `explain` passes
    /// under it (reconstruction, mass conservation, purity, variance sum, three-way), both
    /// unweighted and with a per-row mass.
    #[test]
    fn exposure_marginals_bank_passes_the_five_gates() {
        let (model, x) = moderate_model();
        let w = RefMeasure::ExposureMarginals { floor: 1e-3 };
        let bank = model.explain(&x, w.clone()).unwrap();
        assert_eq!(bank.w, w);
        let n = x.0.n_rows as usize;
        let mass: Vec<f32> = (0..n).map(|r| 0.5 + (r % 7) as f32).collect();
        let weighted = model.explain_weighted(&x, w, &mass).unwrap();
        assert_eq!(weighted.tables.len(), bank.tables.len());
    }

    /// A non-positive or non-finite floor is rejected up front, like a bad `laplace`.
    #[test]
    fn exposure_marginals_rejects_a_non_positive_floor() {
        let model = fixture_model();
        let x = fixture_serve();
        for floor in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
            let err = model
                .explain(&x, RefMeasure::ExposureMarginals { floor })
                .unwrap_err();
            assert!(
                matches!(err, PbError::InvalidConfig { .. }),
                "{floor}: {err:?}"
            );
        }
    }

    /// `recompute_under` re-expresses a legacy bank under the exposure measure without
    /// changing the function: the score agrees on every serve row.
    #[test]
    fn recompute_under_exposure_marginals_preserves_the_score() {
        let (model, x) = moderate_model();
        let legacy = model.explain(&x, RefMeasure::default()).unwrap();
        let re = legacy
            .recompute_under(RefMeasure::ExposureMarginals { floor: 1e-3 })
            .unwrap();
        assert!(matches!(re.w, RefMeasure::ExposureMarginals { .. }));
        let n = x.0.n_rows as usize;
        let mut a = vec![0.0_f64; n];
        let mut b = vec![0.0_f64; n];
        legacy
            .score_binned(&model.schema.cat_encoders, &x.0, &mut a)
            .unwrap();
        re.score_binned(&model.schema.cat_encoders, &x.0, &mut b)
            .unwrap();
        for (p, q) in a.iter().zip(&b) {
            assert!((p - q).abs() < 1e-9, "{p} vs {q}");
        }
    }
    use crate::data::{bin_columns, BinConfig};
    use crate::engine::{Booster, Config, FitSpec};
    use crate::loss::SquaredError;
    use proptest::prelude::*;

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

    fn moderate_model() -> (Model, ServeBinnedMatrix) {
        // A 4-feature additive model with a non-trivial joint grid (each feature
        // realizes several borders), so the integral gates have something to integrate.
        let n = 120usize;
        let cols: Vec<Vec<f32>> = (0..4)
            .map(|f| (0..n).map(|i| ((i * 7 + f * 3) % 11) as f32).collect())
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                cols.iter()
                    .enumerate()
                    .map(|(f, c)| c[i] * (1.0 + f as f32))
                    .sum()
            })
            .collect();
        fit(
            &cols,
            &y,
            Config {
                n_trees: 40,
                learning_rate: 0.3,
                lambda: 1.0,
                ..exact_cfg(40)
            },
        )
    }

    #[test]
    fn bin_to_cell_map_matches_model_bin_to_cell_per_bin() {
        // Merged grid coarser than the model's own (only model border index 2 realized),
        // so several model bins collapse into a shared merged cell — the case the cache
        // actually needs to get right, not a trivial 1:1 mapping.
        let ma = MergedAxis {
            axis: 0,
            borders: vec![2.0],
            model_border_index: vec![2],
            model_n_bins: 7,
            joint: None,
        };
        let map = ma.bin_to_cell_map().unwrap();
        assert_eq!(
            map,
            vec![0, 1, 1, 1, 2, 2, 2],
            "hand-derived expected cells"
        );
        for bin in 0..ma.model_n_bins {
            assert_eq!(
                map[bin as usize],
                ma.model_bin_to_cell(bin as u8).unwrap(),
                "bin {bin}: cached map diverged from the direct per-bin computation"
            );
        }
    }

    /// Regression for the `fill_support` bin-to-cell-map cache (a performance fix, spec
    /// §08.7 display metadata): its output must stay byte-identical to the UNCACHED
    /// per-row `MergedAxis::model_bin_to_cell` walk it replaces. Uses `moderate_model`'s
    /// non-trivial multi-border grid so the cache actually collapses several model bins
    /// into shared merged cells on at least one axis, not a 1:1 mapping.
    #[test]
    fn fill_support_matches_the_uncached_per_row_bin_to_cell_walk() {
        let (model, x) = moderate_model();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();

        let grids = MergedGrids::from_model(&model).unwrap();
        let n_rows = x.0.n_rows as usize;
        for table in &bank.tables {
            let mut expected = Tensor::try_zeros(table.support.shape()).unwrap();
            let mut coord = vec![0usize; table.u.order()];
            for row in 0..n_rows {
                for (k, r) in table.u.0.iter().enumerate() {
                    let ma = grids.axis(*r).unwrap();
                    let bin = x.0.data[ma.axis][row];
                    // The pre-fix, uncached per-row computation — the reference this test
                    // pins the cache against.
                    coord[k] = ma.model_bin_to_cell(bin).unwrap() as usize;
                }
                expected.add(&coord, 1.0).unwrap();
            }
            assert_eq!(
                expected.values(),
                table.support.values(),
                "table {:?}: cached fill_support diverged from the uncached per-row walk",
                table.u
            );
        }
    }

    /// Uniform `weight = [1.0; n]` must reproduce `explain`'s flat-row-count support
    /// exactly (the row-count behavior is the weight=1 special case of the general
    /// exposure-weighted definition, not a different one) — AND leave every table's
    /// `values` bit-identical (only `support`, display metadata, is allowed to move).
    #[test]
    fn explain_weighted_with_unit_weight_matches_explain_exactly() {
        let model = fixture_model();
        let x = fixture_serve();
        let plain = model.explain(&x, RefMeasure::Uniform).unwrap();
        let unit_weight = vec![1.0_f32; x.0.n_rows as usize];
        let weighted = model
            .explain_weighted(&x, RefMeasure::Uniform, &unit_weight)
            .unwrap();
        assert_eq!(plain.tables.len(), weighted.tables.len());
        for (p, w) in plain.tables.iter().zip(&weighted.tables) {
            assert_eq!(p.u, w.u);
            assert_eq!(
                p.values.values(),
                w.values.values(),
                "table {:?}: values must be untouched by explain_weighted",
                p.u
            );
            assert_eq!(
                p.support.values(),
                w.support.values(),
                "table {:?}: unit weight must reproduce the flat row count",
                p.u
            );
        }
    }

    /// Hand-verified non-uniform weights on `fixture_serve`'s 4 rows (bins (1,1), (1,2),
    /// (2,1), (2,2), weights 10/20/30/40): support must sum WEIGHT, not row count, per
    /// merged cell — spec §08.7's exposure-weighted effective mass.
    #[test]
    fn explain_weighted_sums_weight_not_row_count_per_cell() {
        let model = fixture_model();
        let x = fixture_serve();
        let weight = [10.0_f32, 20.0, 30.0, 40.0];
        let bank = model
            .explain_weighted(&x, RefMeasure::Uniform, &weight)
            .unwrap();

        let main0 = bank
            .tables
            .iter()
            .find(|t| t.u == FeatureSet::new(&[0]))
            .expect("main0 table");
        assert_eq!(main0.support.at(&[1]), Some(30.0)); // rows 0,1: 10+20
        assert_eq!(main0.support.at(&[2]), Some(70.0)); // rows 2,3: 30+40

        let main1 = bank
            .tables
            .iter()
            .find(|t| t.u == FeatureSet::new(&[1]))
            .expect("main1 table");
        assert_eq!(main1.support.at(&[1]), Some(40.0)); // rows 0,2: 10+30
        assert_eq!(main1.support.at(&[2]), Some(60.0)); // rows 1,3: 20+40

        let pair = bank
            .tables
            .iter()
            .find(|t| t.u == FeatureSet::new(&[0, 1]))
            .expect("pair table");
        assert_eq!(pair.support.at(&[1, 1]), Some(10.0)); // row 0 alone
        assert_eq!(pair.support.at(&[1, 2]), Some(20.0)); // row 1 alone
        assert_eq!(pair.support.at(&[2, 1]), Some(30.0)); // row 2 alone
        assert_eq!(pair.support.at(&[2, 2]), Some(40.0)); // row 3 alone
    }

    #[test]
    fn explain_weighted_rejects_mismatched_length_and_bad_weights() {
        let model = fixture_model();
        let x = fixture_serve();
        assert!(matches!(
            model.explain_weighted(&x, RefMeasure::Uniform, &[1.0, 1.0, 1.0]),
            Err(PbError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            model.explain_weighted(&x, RefMeasure::Uniform, &[1.0, -1.0, 1.0, 1.0]),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            model.explain_weighted(&x, RefMeasure::Uniform, &[1.0, f32::NAN, 1.0, 1.0]),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            model.explain_weighted(&x, RefMeasure::Uniform, &[1.0, f32::INFINITY, 1.0, 1.0]),
            Err(PbError::InvalidInput { .. })
        ));
    }

    /// Chunk A (bin-to-cell map cache) + Chunk B (`mass = None` regression gate) for
    /// `build_weights`, combined: its cached, mass-less path must be BYTE-IDENTICAL to an
    /// independently hand-rolled reference that (a) walks bins via the UNCACHED per-row
    /// `MergedAxis::model_bin_to_cell` (not `bin_to_cell_map`) and (b) accumulates a flat
    /// `+= 1.0` per row — the exact pre-`mass` `ProductMarginals` arithmetic. Uses
    /// `moderate_model`'s real multi-border grid, both `RefMeasure` kinds.
    #[test]
    fn build_weights_matches_uncached_hand_rolled_reference_when_unweighted() {
        let (model, x) = moderate_model();
        let grids = MergedGrids::from_model(&model).unwrap();
        let n_rows = x.0.n_rows as usize;
        for w_kind in [
            RefMeasure::Uniform,
            RefMeasure::default(),
            RefMeasure::ExposureMarginals { floor: 1e-3 },
        ] {
            let weights = build_weights(&x, &grids, &w_kind, None).unwrap();
            for r in 0..grids.n_raw_features() {
                let raw = FeatureId(r as u32);
                let ma = grids.axis(raw).unwrap();
                let cells = ma.cells();
                // Mirror build_weights' FULL algorithm, not just the per-branch raw_w: it
                // always normalizes raw_w by its own summed total afterward — including
                // the Uniform branch, where that sum is only float-close to 1.0 (a
                // hand-rolled `1/cells` alone, skipping this step, is a DIFFERENT number).
                let raw_w: Vec<f64> = match &w_kind {
                    RefMeasure::Uniform => vec![1.0_f64 / cells as f64; cells],
                    RefMeasure::ProductMarginals { laplace } => {
                        let lap = f64::from(*laplace);
                        let mut counts = vec![0.0_f64; cells];
                        for &bin in &x.0.data[ma.axis] {
                            let cell = ma.model_bin_to_cell(bin).unwrap() as usize;
                            counts[cell] += 1.0;
                        }
                        let unif = 1.0_f64 / cells as f64;
                        let inv_n = 1.0 / n_rows as f64;
                        counts.iter().map(|c| unif + lap * (c * inv_n)).collect()
                    }
                    RefMeasure::ExposureMarginals { floor } => {
                        let fl = f64::from(*floor);
                        let mut counts = vec![0.0_f64; cells];
                        for &bin in &x.0.data[ma.axis] {
                            let cell = ma.model_bin_to_cell(bin).unwrap() as usize;
                            counts[cell] += 1.0;
                        }
                        let unif = 1.0_f64 / cells as f64;
                        let inv_n = 1.0 / n_rows as f64;
                        counts.iter().map(|c| c * inv_n + fl * unif).collect()
                    }
                    RefMeasure::Joint => panic!("Joint rejected by build_weights"),
                };
                let total: f64 = raw_w.iter().sum();
                let expected: Vec<f64> = raw_w.iter().map(|v| v / total).collect();
                assert_eq!(
                    weights.axis(raw).unwrap(),
                    expected.as_slice(),
                    "{w_kind:?} axis {r}: cached build_weights diverged from the hand-rolled \
                     uncached reference"
                );
            }
        }
    }

    /// Hand-derived reference for spec §08.7's `Σ w·e` mass reaching `build_weights`'
    /// `ProductMarginals` empirical marginal — not just `fill_support`'s support tensor,
    /// which `explain_weighted_sums_weight_not_row_count_per_cell` already pins.
    /// `fixture_serve`'s 4 rows are `(bin0,bin1)` = `(1,1),(1,2),(2,1),(2,2)` with mass
    /// `[10,20,30,40]` and `laplace = 1` (`RefMeasure::default()`): axis0 cell1 mass =
    /// 10+20=30, cell2 mass = 30+40=70; axis1 cell1 mass = 10+30=40, cell2 mass =
    /// 20+40=60; total mass = 100; 3 cells (missing + 2 finite) so `unif = 1/3`;
    /// `raw_w[c] = unif + 1.0·(mass[c]/100)`, normalized by their sum (exactly `1 + lap =
    /// 2`: `cells·unif + lap·Σmass/total_mass = 1 + lap·1`).
    #[test]
    fn build_weights_product_marginals_uses_hand_computed_mass_not_row_count() {
        let model = fixture_model();
        let x = fixture_serve();
        let grids = MergedGrids::from_model(&model).unwrap();
        let mass = [10.0_f32, 20.0, 30.0, 40.0];
        let w = build_weights(&x, &grids, &RefMeasure::default(), Some(&mass)).unwrap();

        let unif = 1.0_f64 / 3.0;
        let axis0 = w.axis(FeatureId(0)).unwrap();
        assert!(
            (axis0[0] - unif / 2.0).abs() < 1e-12,
            "axis0 missing cell: {axis0:?}"
        );
        assert!(
            (axis0[1] - (unif + 0.30) / 2.0).abs() < 1e-12,
            "axis0 cell1: {axis0:?}"
        );
        assert!(
            (axis0[2] - (unif + 0.70) / 2.0).abs() < 1e-12,
            "axis0 cell2: {axis0:?}"
        );

        let axis1 = w.axis(FeatureId(1)).unwrap();
        assert!(
            (axis1[0] - unif / 2.0).abs() < 1e-12,
            "axis1 missing cell: {axis1:?}"
        );
        assert!(
            (axis1[1] - (unif + 0.40) / 2.0).abs() < 1e-12,
            "axis1 cell1: {axis1:?}"
        );
        assert!(
            (axis1[2] - (unif + 0.60) / 2.0).abs() < 1e-12,
            "axis1 cell2: {axis1:?}"
        );
    }

    /// `ProductMarginals` weights derive from `mass` (via `build_weights`), so
    /// `explain_weighted` is NOT a display-only variant of `explain`: `values` differ
    /// from the unweighted bank BY DESIGN (spec §08.7) — a skewed mass reweights which
    /// cells dominate the zero-mean centering. Uses the same skewed
    /// `[10,20,30,40]` mass as the hand-computed `build_weights` fixture above, so a
    /// regression here likely means mass stopped reaching `build_weights` from
    /// `explain_bank_with_grids_and_mass`.
    #[test]
    fn explain_weighted_product_marginals_values_shift_by_design() {
        let model = fixture_model();
        let x = fixture_serve();
        let flat = model.explain(&x, RefMeasure::default()).unwrap();
        let mass = [10.0_f32, 20.0, 30.0, 40.0];
        let weighted = model
            .explain_weighted(&x, RefMeasure::default(), &mass)
            .unwrap();
        let pair_flat = flat
            .tables
            .iter()
            .find(|t| t.u == FeatureSet::new(&[0, 1]))
            .expect("pair table");
        let pair_weighted = weighted
            .tables
            .iter()
            .find(|t| t.u == FeatureSet::new(&[0, 1]))
            .expect("pair table");
        assert_ne!(
            pair_flat.values.values(),
            pair_weighted.values.values(),
            "ProductMarginals values must shift under skewed mass — if this trips, mass \
             stopped reaching build_weights"
        );
    }

    /// End-to-end: a real fitted model's weighted export still passes all five I2 gates
    /// (Purity/MassConservation/Reconstruction/VarianceSum/ThreeWayEqual) under a skewed
    /// per-row mass, both `RefMeasure` kinds. `explain_bank_with_grids_and_mass` runs
    /// them unconditionally when `run_gates = true` (which `explain_weighted` always
    /// passes), so `Ok(_)` here IS the certification — a mass-derived reference measure
    /// is still a valid one for purify's exactness properties.
    #[test]
    fn explain_weighted_gates_pass_on_a_real_fitted_model() {
        let (model, x) = moderate_model();
        let n = x.0.n_rows as usize;
        let mass: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let bank = model.explain_weighted(&x, RefMeasure::default(), &mass);
        assert!(
            bank.is_ok(),
            "weighted gates (ProductMarginals) failed: {bank:?}"
        );
        let bank_uniform = model.explain_weighted(&x, RefMeasure::Uniform, &mass);
        assert!(
            bank_uniform.is_ok(),
            "weighted gates (Uniform) failed: {bank_uniform:?}"
        );
    }

    /// Stage 1 (§08.10): the FACTORED order-3 effect reproduces the dense purified table's
    /// variance AND per-cell values bit-for-bit, under BOTH Uniform and the default
    /// ProductMarginals measure — without ever building the dense union cube.
    #[test]
    fn factored_triple_matches_dense_table() {
        // 3 features with a strong 3-way interaction so the booster realizes {0,1,2} trees.
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
        let grids = MergedGrids::from_model(&model).unwrap();
        let n_features = model.provenance.len();
        for w_kind in [RefMeasure::Uniform, RefMeasure::default()] {
            let weights = build_weights(&x, &grids, &w_kind, None).unwrap();
            // Dense order-3 reference: the default Factored policy now always factors order-3, so use
            // the Error policy (identical dense path when under budget) to keep the order-3 table dense.
            let dense_ref = TableBudget {
                max_table_cells: 2_000_000,
                max_bank_cells: 32_000_000,
                on_overflow: OverflowPolicy::Error,
            };
            let bank = model
                .explain_with_budget(&x, w_kind.clone(), dense_ref)
                .unwrap();
            let t3 = bank
                .tables
                .iter()
                .find(|t| t.u.order() == 3)
                .expect("model should realize an order-3 table");
            let ft = FactoredEffect::from_model(&model, &t3.u, &grids, &weights).unwrap();
            let vf = ft.variance;
            assert!(
                (vf - t3.variance).abs() <= 1e-9 * (1.0 + t3.variance.abs()),
                "{w_kind:?}: factored var {vf} != dense var {}",
                t3.variance
            );
            // per-cell values over the full 3-way merged grid
            let cn: Vec<usize> = t3.u.0.iter().map(|r| grids.cells(*r).unwrap()).collect();
            let mut x_cells = vec![0u32; n_features];
            let mut max_abs = 0.0_f64;
            for c0 in 0..cn[0] {
                for c1 in 0..cn[1] {
                    for c2 in 0..cn[2] {
                        x_cells[t3.u.0[0].0 as usize] = c0 as u32;
                        x_cells[t3.u.0[1].0 as usize] = c1 as u32;
                        x_cells[t3.u.0[2].0 as usize] = c2 as u32;
                        let fe = ft.eval(&x_cells).unwrap();
                        let de = t3.eval(&x_cells).unwrap();
                        max_abs = max_abs.max((fe - de).abs());
                    }
                }
            }
            assert!(
                max_abs <= 1e-9,
                "{w_kind:?}: factored eval max abs diff {max_abs}"
            );
        }
    }

    /// The dedup+precompute `compute_variance` must be BYTE-identical to the naive all-pairs
    /// form (the factored-triple variance orders the prune drop-set, so a single-bit drift could
    /// flip which tables survive). Build a real multi-box triple and compare `f64::to_bits`.
    #[test]
    fn compute_variance_tabled_is_bit_identical_to_naive() {
        // Naive reference = the pre-optimization algorithm, inlined verbatim.
        fn naive_inner(per_axis_w: &[Vec<f64>], bi: &FactoredBox, bj: &FactoredBox) -> f64 {
            let mut mass: Vec<[[f64; 2]; 2]> = Vec::with_capacity(3);
            for ((wd, li), lj) in per_axis_w.iter().zip(bi.low.iter()).zip(bj.low.iter()) {
                let mut m = [[0.0_f64; 2]; 2];
                for ((&wc, &a_i), &a_j) in wd.iter().zip(li.iter()).zip(lj.iter()) {
                    m[usize::from(a_i)][usize::from(a_j)] += wc;
                }
                mass.push(m);
            }
            let (m0, m1, m2) = (mass[0], mass[1], mass[2]);
            let g = |m: &[[f64; 2]; 2], a: usize, b: usize| m[a][b];
            let pat = |bx: &FactoredBox, a0: usize, a1: usize, a2: usize| {
                bx.p[a0 | (a1 << 1) | (a2 << 2)]
            };
            let mut s = 0.0_f64;
            for a0i in 0..2 {
                for a0j in 0..2 {
                    for a1i in 0..2 {
                        for a1j in 0..2 {
                            for a2i in 0..2 {
                                for a2j in 0..2 {
                                    let wgt =
                                        g(&m0, a0i, a0j) * g(&m1, a1i, a1j) * g(&m2, a2i, a2j);
                                    if wgt != 0.0 {
                                        s += wgt * pat(bi, a0i, a1i, a2i) * pat(bj, a0j, a1j, a2j);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            s
        }

        let n = 240usize;
        let cols: Vec<Vec<f32>> = (0..3)
            .map(|f| (0..n).map(|i| ((i * (f + 2)) % 7) as f32).collect())
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|i| cols[0][i] * cols[1][i] * cols[2][i] + cols[0][i] - cols[1][i])
            .collect();
        let (model, x) = fit(
            &cols,
            &y,
            Config {
                n_trees: 120,
                learning_rate: 0.3,
                lambda: 1.0,
                ..exact_cfg(120)
            },
        );
        let grids = MergedGrids::from_model(&model).unwrap();
        for w_kind in [RefMeasure::Uniform, RefMeasure::default()] {
            let weights = build_weights(&x, &grids, &w_kind, None).unwrap();
            // Find the order-3 support the booster realized and build its factored triple.
            let dense_ref = TableBudget {
                max_table_cells: 2_000_000,
                max_bank_cells: 32_000_000,
                on_overflow: OverflowPolicy::Error,
            };
            let bank = model
                .explain_with_budget(&x, w_kind.clone(), dense_ref)
                .unwrap();
            let u = bank
                .tables
                .iter()
                .find(|t| t.u.order() == 3)
                .expect("model should realize an order-3 table")
                .u
                .clone();
            let ft = FactoredEffect::from_model(&model, &u, &grids, &weights).unwrap();
            assert!(
                ft.boxes.len() >= 2,
                "need multiple boxes to exercise the sweep"
            );
            // Reference variance via the naive all-pairs fold, in the SAME accumulation order.
            let mut naive_var = 0.0_f64;
            for (i, bi) in ft.boxes.iter().enumerate() {
                naive_var += naive_inner(&ft.per_axis_w, bi, bi);
                for bj in ft.boxes.iter().skip(i + 1) {
                    naive_var += 2.0 * naive_inner(&ft.per_axis_w, bi, bj);
                }
            }
            assert_eq!(
                ft.variance.to_bits(),
                naive_var.to_bits(),
                "{w_kind:?}: tabled variance {} != naive {naive_var} (bit drift)",
                ft.variance
            );
        }
    }

    /// The factored-shed deposit primitive `Tensor::add_marginal_pair` must be BYTE-identical
    /// to the per-cell scalar walk it replaces (`walk_extents(|c| add(c, marg[..][..]))`).
    /// A single-bit drift here would perturb every lower table the over-budget triples shed
    /// into, and hence every downstream purified score. Compare `f64::to_bits` over an
    /// exhaustive grid of shapes, masks, non-zero bases, and repeated accumulation.
    #[test]
    fn add_marginal_pair_is_bit_identical_to_scalar() {
        // Reference = the exact scalar path: add `m[low_a[c0]][low_b[c1]]` into each cell in
        // row-major order via `Tensor::add`.
        fn scalar_pair(t: &mut Tensor, m: &[[f64; 2]; 2], low_a: &[bool], low_b: &[bool]) {
            let sh = t.shape();
            for c0 in 0..sh[0] {
                let s0 = usize::from(low_a[c0]);
                for c1 in 0..sh[1] {
                    t.add(&[c0, c1], m[s0][usize::from(low_b[c1])]).unwrap();
                }
            }
        }
        // Deterministic "ugly" mask/base generators so a swapped axis or wrong `m` entry is
        // caught (distinct `m` entries, non-square shapes, mixed masks, non-zero starting cells).
        let mask = |len: usize, seed: usize| -> Vec<bool> {
            (0..len)
                .map(|i| (i * 2654435761 + seed * 40503) % 3 != 0)
                .collect()
        };
        // Non-square shapes catch an o0/o1 (axis) swap; squares are covered too.
        let shapes = [
            (5usize, 7usize),
            (7, 5),
            (1, 6),
            (6, 1),
            (3, 3),
            (8, 5),
            (4, 4),
        ];
        // Distinct entries so a wrong (s0,s1) selection shows up; include a zero for the sparse case.
        let ms: [[[f64; 2]; 2]; 3] = [
            [[1.25, -2.5], [4.75, -8.125]],
            [[0.1, 0.2], [0.4, 0.8]],
            [[0.0, 3.5], [-6.25, 9.0]],
        ];
        for &(n0, n1) in &shapes {
            for (mi, m) in ms.iter().enumerate() {
                let low_a = mask(n0, mi + 1);
                let low_b = mask(n1, mi + 7);
                // Non-zero base so the deposit adds onto existing content (the shared-table case).
                let base: Vec<f64> = (0..n0 * n1)
                    .map(|k| (k as f64) * 0.6180339887 - 3.0)
                    .collect();
                // Dense: three repeated deposits (simulating multiple boxes into one table).
                let mut fast = Tensor::from_vec(vec![n0, n1], base.clone()).unwrap();
                let mut refr = Tensor::from_vec(vec![n0, n1], base.clone()).unwrap();
                for _ in 0..3 {
                    fast.add_marginal_pair(m, &low_a, &low_b).unwrap();
                    scalar_pair(&mut refr, m, &low_a, &low_b);
                }
                let vf = fast.values();
                let vr = refr.values();
                assert_eq!(vf.len(), vr.len());
                for (k, (a, b)) in vf.iter().zip(vr.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "dense {n0}x{n1} m#{mi} cell {k}: {a} != {b} (bit drift)"
                    );
                }
                // Sparse backing takes the scalar fallback; confirm it too matches bit-for-bit.
                let mut fast_s = Tensor::try_sparse_zeros(vec![n0, n1]).unwrap();
                let mut ref_s = Tensor::try_sparse_zeros(vec![n0, n1]).unwrap();
                fast_s.add_marginal_pair(m, &low_a, &low_b).unwrap();
                scalar_pair(&mut ref_s, m, &low_a, &low_b);
                let vfs = fast_s.values();
                let vrs = ref_s.values();
                for (a, b) in vfs.iter().zip(vrs.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "sparse {n0}x{n1} m#{mi} bit drift"
                    );
                }
            }
        }
    }

    /// End-to-end oracle for the deposit swap: on a REAL multi-triple model whose two
    /// over-budget 3-ways ({0,1,2} and {0,1,3}) both shed into the shared {0,1} pair, the
    /// production shed (`factored_shed_into_raw`, now using `add_marginal_pair`) must leave
    /// EVERY lower table byte-identical to a scalar-deposit shed driven from the same boxes in
    /// the same order. This exercises multi-box AND cross-support accumulation into one table.
    #[test]
    fn factored_shed_deposit_is_bit_identical_to_scalar_walk() {
        // Scalar oracle: same box construction / centering as production (shared code paths),
        // forking ONLY at the deposit, which it does with the per-cell `Tensor::add` walk.
        fn shed_via_scalar(
            model: &Model,
            u: &FeatureSet,
            grids: &MergedGrids,
            w: &WeightCache,
            raw_tables: &mut BTreeMap<FeatureSet, RawTable>,
        ) {
            let u_ids: Vec<FeatureId> = u.0.iter().copied().collect();
            let per_axis_w: Vec<Vec<f64>> =
                u_ids.iter().map(|r| w.axis(*r).unwrap().to_vec()).collect();
            for (alpha, tree) in &model.trees {
                if &tree_support(model, tree).unwrap() != u {
                    continue;
                }
                for (mut p, low) in
                    build_tree_boxes(model, f64::from(*alpha), tree, &u_ids, grids).unwrap()
                {
                    let (wlo, whi) = side_weights(&low, &per_axis_w).unwrap();
                    let marg = center_box_capturing(&mut p, &wlo, &whi).unwrap();
                    for pos in 0..3usize {
                        let m_pos = &marg[pos];
                        let others: [usize; 2] = match pos {
                            0 => [1, 2],
                            1 => [0, 2],
                            _ => [0, 1],
                        };
                        let (o0, o1) = (others[0], others[1]);
                        let sub_u = support_without(u, pos);
                        if !raw_tables.contains_key(&sub_u) {
                            let mut sub_axes = Vec::with_capacity(2);
                            let mut sub_extents = Vec::with_capacity(2);
                            for sr in &sub_u.0 {
                                sub_axes.push(grids.axis_id(*sr).unwrap());
                                sub_extents.push(grids.cells(*sr).unwrap());
                            }
                            raw_tables.insert(
                                sub_u.clone(),
                                RawTable {
                                    u: sub_u.clone(),
                                    axes: sub_axes,
                                    values: Tensor::try_zeros(sub_extents).unwrap(),
                                },
                            );
                        }
                        let target = raw_tables.get_mut(&sub_u).unwrap();
                        let low_a = &low[o0];
                        let low_b = &low[o1];
                        let sh = target.values.shape();
                        for c0 in 0..sh[0] {
                            let s0 = usize::from(low_a[c0]);
                            for c1 in 0..sh[1] {
                                let s1 = usize::from(low_b[c1]);
                                target.values.add(&[c0, c1], m_pos[s0 | (s1 << 1)]).unwrap();
                            }
                        }
                    }
                }
            }
            raw_tables.remove(u);
        }

        let n = 6usize.pow(4);
        let cols: Vec<Vec<f32>> = (0..4)
            .map(|f| {
                (0..n)
                    .map(|i| ((i / 6usize.pow(f as u32)) % 6) as f32)
                    .collect()
            })
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|i| cols[0][i] * cols[1][i] * cols[2][i] - cols[0][i] * cols[1][i] * cols[3][i])
            .collect();
        let (model, x) = fit(
            &cols,
            &y,
            Config {
                n_trees: 120,
                learning_rate: 0.3,
                lambda: 1.0,
                ..exact_cfg(120)
            },
        );
        let grids = MergedGrids::from_model(&model).unwrap();
        let weights = build_weights(&x, &grids, &RefMeasure::default(), None).unwrap();

        // Budget just above the largest order-2 forces exactly the 3-ways to factor.
        let dense = model
            .explain_with_budget(
                &x,
                RefMeasure::default(),
                TableBudget {
                    max_table_cells: 2_000_000,
                    max_bank_cells: 32_000_000,
                    on_overflow: OverflowPolicy::Error,
                },
            )
            .unwrap();
        let max_2way = dense
            .tables
            .iter()
            .filter(|t| t.u.order() == 2)
            .map(|t| t.values.values().len())
            .max()
            .unwrap_or(1);
        let budget = TableBudget {
            max_table_cells: (max_2way + 1) as u64,
            max_bank_cells: 32_000_000,
            on_overflow: OverflowPolicy::Factored,
        };
        let (raw, factored_supports) = accumulate(&model, &grids, &budget).unwrap();
        assert!(
            factored_supports.len() >= 2,
            "need >=2 factored triples to exercise a shared lower table, got {}",
            factored_supports.len()
        );
        // Both triples must share a pair, so the {0,1} table receives sheds from both supports.
        let shares_pair = factored_supports.iter().enumerate().any(|(i, a)| {
            factored_supports
                .iter()
                .skip(i + 1)
                .any(|b| a.0.iter().filter(|r| b.0.contains(r)).count() == 2)
        });
        assert!(shares_pair, "expected two factored triples sharing a pair");

        // Shed every triple both ways, in the SAME support/box/pos order, into clones.
        // (`RawTable` is not `Clone`; deep-copy its fields to keep the change test-local.)
        let clone_tables = |t: &BTreeMap<FeatureSet, RawTable>| -> BTreeMap<FeatureSet, RawTable> {
            t.iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        RawTable {
                            u: v.u.clone(),
                            axes: v.axes.clone(),
                            values: v.values.clone(),
                        },
                    )
                })
                .collect()
        };
        let mut tables_fast = clone_tables(&raw.tables);
        let mut tables_ref = clone_tables(&raw.tables);
        for u in &factored_supports {
            factored_shed_into_raw(
                &model,
                u,
                &grids,
                &weights,
                &mut tables_fast,
                Vec::new(),
                &mut BTreeMap::new(),
            )
            .unwrap();
            shed_via_scalar(&model, u, &grids, &weights, &mut tables_ref);
        }

        assert_eq!(
            tables_fast.keys().collect::<Vec<_>>(),
            tables_ref.keys().collect::<Vec<_>>(),
            "fast and scalar sheds must produce the same set of tables"
        );
        let mut checked_cells = 0usize;
        for (k, tf) in &tables_fast {
            let tr = tables_ref.get(k).unwrap();
            let vf = tf.values.values();
            let vr = tr.values.values();
            assert_eq!(vf.len(), vr.len(), "table {k:?} cell count");
            for (cell, (a, b)) in vf.iter().zip(vr.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "table {k:?} cell {cell}: fast {a} != scalar {b} (bit drift)"
                );
                checked_cells += 1;
            }
        }
        assert!(checked_cells > 0, "no cells compared");
    }

    /// Stage 2 (§08.10): shedding the over-budget 3-way per tree into the RAW lower tables,
    /// then running the normal order-2->1 cascade, reproduces the dense bank EXACTLY — every
    /// lower table cell-for-cell, f0, and the 3-way residual (variance + per-cell eval).
    #[test]
    fn factored_shed_reproduces_dense_bank() {
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
        let grids = MergedGrids::from_model(&model).unwrap();
        for w_kind in [RefMeasure::Uniform, RefMeasure::default()] {
            let weights = build_weights(&x, &grids, &w_kind, None).unwrap();
            // Dense reference via Error policy (default Factored now always factors order-3).
            let dense_ref = TableBudget {
                max_table_cells: 2_000_000,
                max_bank_cells: 32_000_000,
                on_overflow: OverflowPolicy::Error,
            };
            let bank_dense = model
                .explain_with_budget(&x, w_kind.clone(), dense_ref)
                .unwrap();
            let u3 = bank_dense
                .tables
                .iter()
                .find(|t| t.u.order() == 3)
                .expect("order-3 table")
                .u
                .clone();
            // The 3-way must be fed by >1 tree so the variance cross-terms Σ_{t≠s}⟨p_t,p_s⟩
            // and the multi-tree shed are actually exercised (not an incidental single box).
            let n_u3_trees = model
                .trees
                .iter()
                .filter(|(_, t)| tree_support(&model, t).is_ok_and(|s| s == u3))
                .count();
            assert!(
                n_u3_trees > 1,
                "{w_kind:?}: need >1 tree feeding the 3-way support"
            );

            // Factored path: accumulate raw, shed the 3-way per tree into the raw lower
            // tables, drop the 3-way raw table, then run the standard order-2->1 cascade.
            let (raw, _) = accumulate(&model, &grids, &TableBudget::default()).unwrap();
            let RawBank { f0, mut tables } = raw;
            let ft = factored_shed_into_raw(
                &model,
                &u3,
                &grids,
                &weights,
                &mut tables,
                Vec::new(),
                &mut BTreeMap::new(),
            )
            .unwrap();
            assert!(!tables.contains_key(&u3), "3-way raw table must be removed");
            let bank_fac = purify(
                RawBank { f0, tables },
                &weights,
                &grids,
                PurifyMode::SinglePass,
            )
            .unwrap();

            assert!(
                (bank_fac.f0 - bank_dense.f0).abs() <= 1e-9 * (1.0 + bank_dense.f0.abs()),
                "{w_kind:?}: f0 {} != {}",
                bank_fac.f0,
                bank_dense.f0
            );
            assert_eq!(
                bank_fac.tables.len(),
                bank_dense.tables.len() - 1,
                "{w_kind:?}: factored bank should hold every dense table except the 3-way"
            );
            for td in &bank_dense.tables {
                if td.u == u3 {
                    // `factored_shed_into_raw` now defers variance to the caller's parallel pass
                    // (leaving it 0.0), so compute it explicitly for this direct-call test.
                    let vf = ft.compute_variance().unwrap();
                    assert!(
                        (vf - td.variance).abs() <= 1e-9 * (1.0 + td.variance.abs()),
                        "{w_kind:?}: factored 3-way var {vf} != dense {}",
                        td.variance
                    );
                    let cn: Vec<usize> = td.u.0.iter().map(|r| grids.cells(*r).unwrap()).collect();
                    let mut x_cells = vec![0u32; model.provenance.len()];
                    let mut max_abs = 0.0_f64;
                    for c0 in 0..cn[0] {
                        for c1 in 0..cn[1] {
                            for c2 in 0..cn[2] {
                                x_cells[td.u.0[0].0 as usize] = c0 as u32;
                                x_cells[td.u.0[1].0 as usize] = c1 as u32;
                                x_cells[td.u.0[2].0 as usize] = c2 as u32;
                                let d =
                                    (ft.eval(&x_cells).unwrap() - td.eval(&x_cells).unwrap()).abs();
                                max_abs = max_abs.max(d);
                            }
                        }
                    }
                    assert!(
                        max_abs <= 1e-9,
                        "{w_kind:?}: factored 3-way eval diff {max_abs}"
                    );
                    continue;
                }
                let tf = bank_fac
                    .tables
                    .iter()
                    .find(|t| t.u == td.u)
                    .unwrap_or_else(|| panic!("{w_kind:?}: factored bank missing {:?}", td.u));
                assert!(
                    (tf.variance - td.variance).abs() <= 1e-9 * (1.0 + td.variance.abs()),
                    "{w_kind:?}: table {:?} var {} != {}",
                    td.u,
                    tf.variance,
                    td.variance
                );
                let vf = tf.values.values();
                let vd = td.values.values();
                assert_eq!(
                    vf.len(),
                    vd.len(),
                    "{w_kind:?}: table {:?} cell count",
                    td.u
                );
                let mut max_abs = 0.0_f64;
                for (a, b) in vf.iter().zip(vd.iter()) {
                    max_abs = max_abs.max((a - b).abs());
                }
                assert!(
                    max_abs <= 1e-9,
                    "{w_kind:?}: table {:?} cell diff {max_abs}",
                    td.u
                );
            }
        }
    }

    /// Stage 3 (§08.10) END-TO-END: a model whose 3-way table exceeds the cell budget now
    /// produces a VALID bank via the factored path instead of `PbError::TableBudget`. The
    /// `Ok(_)` itself means all five I2 gates passed on the factored bank; we also check it
    /// equals the dense bank (score everywhere + total variance) and is structurally factored.
    #[test]
    fn explain_factors_over_budget_three_way_and_passes_all_gates() {
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
        let grids = MergedGrids::from_model(&model).unwrap();
        let n_features = model.provenance.len();

        // Dense reference (huge budget, hard-error policy — the old behavior).
        let dense_budget = TableBudget {
            max_table_cells: 2_000_000,
            max_bank_cells: 32_000_000,
            on_overflow: OverflowPolicy::Error,
        };
        let bank_dense = model
            .explain_with_budget(&x, RefMeasure::default(), dense_budget)
            .unwrap();
        let u3 = bank_dense
            .tables
            .iter()
            .find(|t| t.u.order() == 3)
            .expect("order-3 table")
            .u
            .clone();

        // Budget that forces ONLY the 3-way over: just above the largest order-2 table.
        let max_2way = bank_dense
            .tables
            .iter()
            .filter(|t| t.u.order() == 2)
            .map(|t| t.values.values().len())
            .max()
            .unwrap_or(1);
        let cells3: usize = u3.0.iter().map(|r| grids.cells(*r).unwrap()).product();
        assert!(
            cells3 > max_2way + 1,
            "3-way ({cells3}) must exceed the budget ({max_2way})"
        );
        let budget = TableBudget {
            max_table_cells: (max_2way + 1) as u64,
            max_bank_cells: 32_000_000,
            on_overflow: OverflowPolicy::Factored,
        };

        // The Ok here means accumulate -> shed -> purify -> ALL FIVE I2 GATES succeeded.
        let bank_fac = model
            .explain_with_budget(&x, RefMeasure::default(), budget)
            .expect("factored explain must pass all five exactness gates");

        // Structurally factored, not dense.
        assert_eq!(
            bank_fac.factored.len(),
            1,
            "exactly the one 3-way is factored"
        );
        assert_eq!(bank_fac.factored[0].u, u3);
        assert!(
            bank_fac.tables.iter().all(|t| t.u.order() < 3),
            "no dense order-3 table should remain"
        );

        // Score equals the dense bank at every merged cell of the 3 features.
        let cells_per: Vec<usize> = (0..n_features)
            .map(|r| grids.cells(FeatureId(r as u32)).unwrap())
            .collect();
        let mut x_cells = vec![0u32; n_features];
        let mut max_score_diff = 0.0_f64;
        for c0 in 0..cells_per[0] {
            for c1 in 0..cells_per[1] {
                for c2 in 0..cells_per[2] {
                    x_cells[0] = c0 as u32;
                    x_cells[1] = c1 as u32;
                    x_cells[2] = c2 as u32;
                    let d = (bank_fac.score(&x_cells).unwrap()
                        - bank_dense.score(&x_cells).unwrap())
                    .abs();
                    max_score_diff = max_score_diff.max(d);
                }
            }
        }
        assert!(
            max_score_diff <= 1e-9,
            "factored score diverges from dense by {max_score_diff}"
        );

        // Total variance (dense tables + factored) matches the dense bank.
        let var_fac: f64 = bank_fac.tables.iter().map(|t| t.variance).sum::<f64>()
            + bank_fac.factored.iter().map(|f| f.variance).sum::<f64>();
        let var_dense: f64 = bank_dense.tables.iter().map(|t| t.variance).sum();
        assert!(
            (var_fac - var_dense).abs() <= 1e-9 * (1.0 + var_dense.abs()),
            "factored total variance {var_fac} != dense {var_dense}"
        );
    }

    /// A factored high-order effect is a SUM of per-tree purified boxes, not a dense table,
    /// so the rebase-a-cell-into-f0 operation is undefined for it. A rating basis that names
    /// a factored support must error loudly (a deployer trying to "anchor" a 3-way), while a
    /// basis naming a still-dense order-2 support in the same factored bank rebases normally.
    #[test]
    fn rating_basis_refuses_to_rebase_factored_effect() {
        use crate::serialize::{RatingBasis, RatingReference};

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
        // Budget just above the largest order-2 table forces ONLY the 3-way to factor.
        let bank_dense = model
            .explain_with_budget(&x, RefMeasure::default(), TableBudget::default())
            .unwrap();
        let max_2way = bank_dense
            .tables
            .iter()
            .filter(|t| t.u.order() == 2)
            .map(|t| t.values.values().len())
            .max()
            .unwrap();
        let bank = model
            .explain_with_budget(
                &x,
                RefMeasure::default(),
                TableBudget {
                    max_table_cells: (max_2way + 1) as u64,
                    max_bank_cells: 32_000_000,
                    on_overflow: OverflowPolicy::Factored,
                },
            )
            .unwrap();
        assert_eq!(bank.factored.len(), 1, "exactly one 3-way is factored");

        // A basis naming the factored 3-way support is rejected.
        let factored_set: Vec<u32> = bank.factored[0].u.0.iter().map(|f| f.0).collect();
        let bad = RatingBasis {
            reference: vec![RatingReference {
                feature_set: factored_set,
                coord: vec![0, 0, 0],
            }],
        };
        assert!(matches!(
            bank.to_rating_export(
                model.link,
                &model.mode,
                &model.schema,
                &model.provenance,
                &model.schema.cat_encoders,
                Some(&bad)
            ),
            Err(PbError::InvalidConfig { .. })
        ));

        // A basis naming a still-dense order-2 support in the SAME bank rebases fine.
        let pair = bank
            .tables
            .iter()
            .find(|t| t.u.order() == 2)
            .expect("a dense order-2 table survives")
            .u
            .clone();
        let ok = RatingBasis {
            reference: vec![RatingReference {
                feature_set: pair.0.iter().map(|f| f.0).collect(),
                coord: vec![0, 0],
            }],
        };
        bank.to_rating_export(
            model.link,
            &model.mode,
            &model.schema,
            &model.provenance,
            &model.schema.cat_encoders,
            Some(&ok),
        )
        .unwrap();
    }

    /// Stage 3 (§08.10): MULTIPLE over-budget triples sharing a pair. A non-separable
    /// `x0·x1·x2 − x0·x1·x3` signal forces both {0,1,2} and {0,1,3}, so the shared {0,1}
    /// table receives sheds from BOTH — the case a single-triple test cannot exercise.
    /// All five gates still pass and the bank stays dense-equivalent.
    #[test]
    fn explain_factors_multiple_over_budget_three_ways_sharing_a_pair() {
        let n = 6usize.pow(4);
        let cols: Vec<Vec<f32>> = (0..4)
            .map(|f| {
                (0..n)
                    .map(|i| ((i / 6usize.pow(f as u32)) % 6) as f32)
                    .collect()
            })
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|i| cols[0][i] * cols[1][i] * cols[2][i] - cols[0][i] * cols[1][i] * cols[3][i])
            .collect();
        let (model, x) = fit(
            &cols,
            &y,
            Config {
                n_trees: 120,
                learning_rate: 0.3,
                lambda: 1.0,
                ..exact_cfg(120)
            },
        );
        let grids = MergedGrids::from_model(&model).unwrap();
        let n_features = model.provenance.len();

        let dense_budget = TableBudget {
            max_table_cells: 2_000_000,
            max_bank_cells: 32_000_000,
            on_overflow: OverflowPolicy::Error,
        };
        let bank_dense = model
            .explain_with_budget(&x, RefMeasure::default(), dense_budget)
            .unwrap();
        let n_triples = bank_dense
            .tables
            .iter()
            .filter(|t| t.u.order() == 3)
            .count();
        assert!(
            n_triples >= 2,
            "need >=2 realized 3-way supports, got {n_triples}"
        );
        let shares_a_pair = {
            let triples: Vec<&FeatureSet> = bank_dense
                .tables
                .iter()
                .filter(|t| t.u.order() == 3)
                .map(|t| &t.u)
                .collect();
            triples.iter().enumerate().any(|(i, a)| {
                triples
                    .iter()
                    .skip(i + 1)
                    .any(|b| a.0.iter().filter(|r| b.0.contains(r)).count() == 2)
            })
        };
        assert!(shares_a_pair, "expected two triples sharing a pair");

        let max_2way = bank_dense
            .tables
            .iter()
            .filter(|t| t.u.order() == 2)
            .map(|t| t.values.values().len())
            .max()
            .unwrap_or(1);
        let budget = TableBudget {
            max_table_cells: (max_2way + 1) as u64,
            max_bank_cells: 32_000_000,
            on_overflow: OverflowPolicy::Factored,
        };
        let bank_fac = model
            .explain_with_budget(&x, RefMeasure::default(), budget)
            .expect("multi-triple factored explain must pass all five gates");
        assert_eq!(
            bank_fac.factored.len(),
            n_triples,
            "every triple should be factored"
        );
        assert!(bank_fac.tables.iter().all(|t| t.u.order() < 3));

        let cn: Vec<usize> = (0..n_features)
            .map(|r| grids.cells(FeatureId(r as u32)).unwrap())
            .collect();
        let mut x_cells = vec![0u32; n_features];
        let mut maxd = 0.0_f64;
        for c0 in 0..cn[0] {
            for c1 in 0..cn[1] {
                for c2 in 0..cn[2] {
                    for c3 in 0..cn[3] {
                        x_cells[0] = c0 as u32;
                        x_cells[1] = c1 as u32;
                        x_cells[2] = c2 as u32;
                        x_cells[3] = c3 as u32;
                        let d = (bank_fac.score(&x_cells).unwrap()
                            - bank_dense.score(&x_cells).unwrap())
                        .abs();
                        maxd = maxd.max(d);
                    }
                }
            }
        }
        assert!(
            maxd <= 1e-9,
            "multi-triple factored score diverges by {maxd}"
        );
    }

    /// Stage 5 (§08.10): the rating export emits factored effects as per-tree boxes, and
    /// evaluating those PUBLISHED boxes (threshold routing, the deployment/SQL form) exactly
    /// reproduces the in-bank factored eval — i.e. the export is a complete, lossless artifact.
    #[test]
    fn numeric_mask_export_preserves_all_cell_patterns_and_mixed_axes() {
        // Every possible mask over four finite cells plus missing, including an
        // interior interval, high-side region, missing-only, and constant masks.
        for pattern in 0u32..32 {
            let axes: Vec<AxisId> = (0..3)
                .map(|raw| AxisId {
                    raw: FeatureId(raw),
                    borders: vec![-1.0, 0.0, 1.0],
                    cells: 5,
                    joint_channels: None,
                    band_of: None,
                })
                .collect();
            let masks: Vec<Vec<bool>> = [pattern, (pattern * 7 + 3) % 32, (pattern * 13 + 5) % 32]
                .iter()
                .map(|bits| (0..5).map(|c| bits & (1 << c) != 0).collect())
                .collect();
            let effect = FactoredEffect {
                u: FeatureSet((0..3).map(FeatureId).collect()),
                axes,
                per_axis_w: vec![vec![0.2; 5]; 3],
                boxes: vec![FactoredBox {
                    p: vec![-2.0, 1.5, 0.0, 4.0, 2.5, -3.0, 0.5, 7.0],
                    low: masks,
                }],
                variance: 0.0,
            };
            let export = effect.export_boxes().unwrap();
            let values = [f32::NAN, -2.0, -0.5, 0.5, 2.0];
            for a in 0..5u32 {
                for b in 0..5u32 {
                    for c in 0..5u32 {
                        let cells = [a, b, c];
                        let expected = effect.eval(&cells).unwrap();
                        let actual: f64 = export
                            .iter()
                            .map(|term| {
                                let mut corner = 0;
                                for (d, &cell) in cells.iter().enumerate() {
                                    let low = if cell == 0 {
                                        term.missing_left[d]
                                    } else {
                                        values[cell as usize] <= term.thresholds[d]
                                    };
                                    corner |= usize::from(low) << d;
                                }
                                term.octants[corner]
                            })
                            .sum();
                        assert_eq!(actual, expected, "mask {pattern}, cells {cells:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn numeric_mask_export_preserves_the_original_prefix_payload() {
        let axis = AxisId {
            raw: FeatureId(0),
            borders: vec![-1.0, 0.0, 1.0],
            cells: 5,
            joint_channels: None,
            band_of: None,
        };
        for missing in [false, true] {
            let terms =
                numeric_mask_export_terms(&axis, &[missing, true, true, false, false]).unwrap();
            assert_eq!(terms.len(), 1);
            assert_eq!(terms[0].threshold, 0.0);
            assert_eq!(terms[0].missing_left, missing);
            assert_eq!(terms[0].weights, [[1.0, 0.0], [0.0, 1.0]]);
        }
    }

    #[test]
    fn factored_rating_export_roundtrips_to_bank_eval() {
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
        let grids = MergedGrids::from_model(&model).unwrap();
        let dense = model
            .explain_with_budget(
                &x,
                RefMeasure::default(),
                TableBudget {
                    max_table_cells: 2_000_000,
                    max_bank_cells: 32_000_000,
                    on_overflow: OverflowPolicy::Error,
                },
            )
            .unwrap();
        let max_2way = dense
            .tables
            .iter()
            .filter(|t| t.u.order() == 2)
            .map(|t| t.values.values().len())
            .max()
            .unwrap_or(1);
        let bank = model
            .explain_with_budget(
                &x,
                RefMeasure::default(),
                TableBudget {
                    max_table_cells: (max_2way + 1) as u64,
                    max_bank_cells: 32_000_000,
                    on_overflow: OverflowPolicy::Factored,
                },
            )
            .unwrap();
        assert!(!bank.factored.is_empty());

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
        assert_eq!(export.factored.len(), bank.factored.len());
        assert!(
            export.tables.iter().all(|t| t.feature_set.order() < 3),
            "no dense order-3 table should be exported"
        );

        // Evaluate each EXPORTED factored effect (threshold routing in cell space) and confirm
        // it matches the in-bank eval at every merged cell of its support.
        for (ft, rf) in bank.factored.iter().zip(export.factored.iter()) {
            assert_eq!(rf.feature_set, ft.u);
            let cn: Vec<usize> = ft.u.0.iter().map(|r| grids.cells(*r).unwrap()).collect();
            let mut x_cells = vec![0u32; model.provenance.len()];
            for c0 in 0..cn[0] {
                for c1 in 0..cn[1] {
                    for c2 in 0..cn[2] {
                        let cells = [c0, c1, c2];
                        x_cells[ft.u.0[0].0 as usize] = c0 as u32;
                        x_cells[ft.u.0[1].0 as usize] = c1 as u32;
                        x_cells[ft.u.0[2].0 as usize] = c2 as u32;
                        let mut from_export = 0.0_f64;
                        for bx in &rf.boxes {
                            let mut idx = 0usize;
                            for (d, ((&cell, axis), (&thr, &miss))) in cells
                                .iter()
                                .zip(ft.axes.iter())
                                .zip(bx.thresholds.iter().zip(bx.missing_left.iter()))
                                .enumerate()
                            {
                                let low = if cell == 0 {
                                    miss
                                } else {
                                    // finite cell is low iff cell <= j+1, where borders[j] == thr
                                    let j = axis.borders.iter().position(|&b| b == thr).unwrap();
                                    cell <= j + 1
                                };
                                idx |= usize::from(low) << d;
                            }
                            from_export += bx.octants[idx];
                        }
                        let from_bank = ft.eval(&x_cells).unwrap();
                        assert!(
                            (from_export - from_bank).abs() <= 1e-12,
                            "exported box eval {from_export} != bank eval {from_bank}"
                        );
                    }
                }
            }
        }
    }

    /// F5 (§08.10 deployment): the COMPLETE rating export — intercept + every dense table +
    /// every factored effect together — is a self-sufficient scoring artifact. A scorer that
    /// reads ONLY the export (no bank, no model) reproduces `bank.score` at every cell, and the
    /// reconstruction gate already equates `bank.score` to the ensemble. This closes the chain
    /// the per-component tests only cover in pieces: dense values are copied verbatim and the
    /// factored boxes are checked alone, but their COMPOSITION (f0 + dense + factored summed at
    /// one input) is exercised here. The factored boxes ship raw-value thresholds only, so the
    /// standalone scorer recovers each feature's borders from the dense tables that name it —
    /// proving the export is closed under scoring without re-consulting the model.
    #[test]
    fn rating_export_is_a_complete_standalone_scorer() {
        use std::collections::HashMap;
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
        let grids = MergedGrids::from_model(&model).unwrap();
        let n_features = model.provenance.len();

        let dense = model
            .explain_with_budget(&x, RefMeasure::default(), TableBudget::default())
            .unwrap();
        let max_2way = dense
            .tables
            .iter()
            .filter(|t| t.u.order() == 2)
            .map(|t| t.values.values().len())
            .max()
            .unwrap();
        let bank = model
            .explain_with_budget(
                &x,
                RefMeasure::default(),
                TableBudget {
                    max_table_cells: (max_2way + 1) as u64,
                    max_bank_cells: 32_000_000,
                    on_overflow: OverflowPolicy::Factored,
                },
            )
            .unwrap();
        assert!(
            !bank.factored.is_empty(),
            "the 3-way must factor so composition includes a factored term"
        );

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

        // Recover per-feature borders from the dense tables (factored boxes carry only
        // raw-value thresholds). Every feature appears in at least its main-effect table.
        let mut borders_of: HashMap<u32, Vec<f32>> = HashMap::new();
        for t in &export.tables {
            for a in &t.axes {
                borders_of.entry(a.raw).or_insert_with(|| a.borders.clone());
            }
        }

        // A scorer that consults ONLY `export` (+ the borders gathered from it).
        let score_from_export = |x_cells: &[u32]| -> f64 {
            let mut s = export.f0;
            for t in &export.tables {
                let mut idx = 0usize;
                for (d, a) in t.axes.iter().enumerate() {
                    idx = idx * t.shape[d] as usize + x_cells[a.raw as usize] as usize;
                }
                s += t.values[idx];
            }
            for rf in &export.factored {
                for bx in &rf.boxes {
                    let mut oct = 0usize;
                    for (d, raw) in rf.feature_set.0.iter().enumerate() {
                        let cell = x_cells[raw.0 as usize] as usize;
                        let low = if cell == 0 {
                            bx.missing_left[d]
                        } else {
                            let bs = &borders_of[&raw.0];
                            let j = bs.iter().position(|&b| b == bx.thresholds[d]).unwrap();
                            cell <= j + 1
                        };
                        oct |= usize::from(low) << d;
                    }
                    s += bx.octants[oct];
                }
            }
            s
        };

        let cells_per: Vec<usize> = (0..n_features)
            .map(|r| grids.cells(FeatureId(r as u32)).unwrap())
            .collect();
        let mut x_cells = vec![0u32; n_features];
        let mut max_diff = 0.0_f64;
        for c0 in 0..cells_per[0] {
            for c1 in 0..cells_per[1] {
                for c2 in 0..cells_per[2] {
                    x_cells[0] = c0 as u32;
                    x_cells[1] = c1 as u32;
                    x_cells[2] = c2 as u32;
                    let d = (score_from_export(&x_cells) - bank.score(&x_cells).unwrap()).abs();
                    max_diff = max_diff.max(d);
                }
            }
        }
        assert!(
            max_diff <= 1e-12,
            "standalone export scorer diverges from bank by {max_diff}"
        );
    }

    /// F4 (§08.10 + §09.5): an OuterBag model (a tree-soup of all bagged members) whose 3-way
    /// support exceeds the cell budget still produces a VALID factored bank — bagging composes
    /// with the factored path, so the user-facing bagging at competitive fidelity stays
    /// exactly decomposable. (average_banks/recompute_under are internal, not on this path.)
    #[test]
    fn outer_bag_over_budget_three_way_stays_factored_and_exact() {
        use crate::boosters::{BoosterConfig, EnsembleSpec};
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
                boosters: BoosterConfig {
                    ensemble: EnsembleSpec::OuterBag {
                        n_bags: 2,
                        bag_subsample: 1.0,
                        cell_refit: None,
                    },
                    ..Default::default()
                },
                ..exact_cfg(80)
            },
        );
        let grids = MergedGrids::from_model(&model).unwrap();
        let dense = model
            .explain_with_budget(
                &x,
                RefMeasure::default(),
                TableBudget {
                    max_table_cells: 2_000_000,
                    max_bank_cells: 32_000_000,
                    on_overflow: OverflowPolicy::Error,
                },
            )
            .unwrap();
        let u3 = dense
            .tables
            .iter()
            .find(|t| t.u.order() == 3)
            .expect("bagged soup should realize a 3-way")
            .u
            .clone();
        let max_2way = dense
            .tables
            .iter()
            .filter(|t| t.u.order() == 2)
            .map(|t| t.values.values().len())
            .max()
            .unwrap_or(1);
        let cells3: usize = u3.0.iter().map(|r| grids.cells(*r).unwrap()).product();
        assert!(
            cells3 > max_2way + 1,
            "3-way ({cells3}) must exceed the budget"
        );
        // The Ok here = all five I2 gates passed on the BAGGED soup's factored bank.
        let bank = model
            .explain_with_budget(
                &x,
                RefMeasure::default(),
                TableBudget {
                    max_table_cells: (max_2way + 1) as u64,
                    max_bank_cells: 32_000_000,
                    on_overflow: OverflowPolicy::Factored,
                },
            )
            .expect("OuterBag + factored must pass all five exactness gates");
        assert!(
            !bank.factored.is_empty(),
            "the over-budget 3-way should be factored"
        );
        assert!(bank.tables.iter().all(|t| t.u.order() < 3));
    }

    /// MassConservation fix: `ensemble_w_mean` (the new exact per-tree integral) equals
    /// the exhaustive joint-grid `Σ w·F_ens` — so mass is exact with NO joint enumeration.
    #[test]
    fn ensemble_w_mean_equals_exhaustive_joint_integral() {
        let (model, x) = moderate_model();
        let grids = MergedGrids::from_model(&model).unwrap();
        let w = build_weights(&x, &grids, &RefMeasure::default(), None).unwrap();
        let per_tree = ensemble_w_mean(&model, &grids, &w).unwrap();
        // Independent exhaustive joint-grid integral over all realized features.
        let feats =
            gate_features(&model, &model.explain(&x, RefMeasure::default()).unwrap()).unwrap();
        let mut joint = 0.0_f64;
        enumerate_check_points(&grids, &feats, |x_cells, rep_bins| {
            joint += joint_weight(&w, &feats, x_cells)? * model.ensemble_f64(rep_bins)?;
            Ok(())
        })
        .unwrap();
        assert!(
            (per_tree - joint).abs() < 1e-9 * (1.0 + joint.abs()),
            "per-tree mass {per_tree} != exhaustive joint mass {joint}"
        );
    }

    /// REGRESSION (high-dimensional decomposition): a model wide enough that the joint-grid
    /// cell count `Π(cells)` exceeds `u64` (here 70 binary-cell axes => `2^70`) must SAMPLE the
    /// gate sweep, not hard-error. This was the allstate (130-feature) `tables()` crash —
    /// `enumerate_check_points` computed the product as `u64` and overflowed BEFORE the cap
    /// check could route it to sampling. The fix (`saturating_product_u64`) caps at `u64::MAX`.
    #[test]
    fn wide_model_joint_grid_samples_instead_of_overflowing() {
        let n = 70usize;
        let per_raw: Vec<MergedAxis> = (0..n)
            .map(|i| MergedAxis {
                axis: i,
                borders: vec![], // cells() = 2 per axis => 2^70 total, overflows u64
                model_border_index: vec![],
                model_n_bins: 2,
                joint: None,
            })
            .collect();
        let n_axes = per_raw.len();
        let grids = MergedGrids { per_raw, n_axes };
        let feats: Vec<FeatureId> = (0..n as u32).map(FeatureId).collect();

        TEST_JOINT_CAP.with(|c| c.set(64));
        let mut visited = 0usize;
        let sampled = enumerate_check_points(&grids, &feats, |_x_cells, _rep_bins| {
            visited += 1;
            Ok(())
        });
        TEST_JOINT_CAP.with(|c| c.set(0));

        let sampled = sampled.expect("a >u64 joint grid must sample, not overflow-error");
        assert!(
            sampled,
            "a 2^70-cell joint grid must take the SAMPLING branch"
        );
        assert_eq!(visited, 64, "sampling emits exactly joint_cap points");
    }

    /// The integral gates stay correct when the joint grid is SAMPLED (the >JOINT_CAP
    /// path, forced here with a tiny test cap on a small model). MassConservation is
    /// exact (never samples); the self-normalized sampled mean still recovers f0 (the
    /// un-normalized partial sum — the prior bug — would be a tiny fraction of f0).
    #[test]
    fn integral_gates_correct_under_forced_sampling() {
        let (model, x) = moderate_model();
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        let grids = MergedGrids::from_model(&model).unwrap();
        let feats = gate_features(&model, &bank).unwrap();
        let w = build_weights(&x, &grids, &RefMeasure::default(), None).unwrap();

        // Force the SAMPLED branch on this small model.
        TEST_JOINT_CAP.with(|c| c.set(64));

        // Mass is exact (per-tree, ignores the cap); Reconstruction/ThreeWayEqual are
        // per-point and sound under sampling — all pass.
        check_mass_conservation(&model, &bank, &w).unwrap();
        check_reconstruction(&model, &bank).unwrap();
        check_three_way_equal(&model, &bank).unwrap();
        // VarianceSum under sampling now draws ∝ w (per-axis inverse-CDF) and averages
        // unweighted, so the estimator stays accurate instead of collapsing on the product
        // weight — it must certify this exact model, not false-fail.
        check_variance_sum(&model, &bank, &w).unwrap();

        // Self-normalization recovers the true mean from the partial sample. Replicate
        // the estimator: the UN-normalized m1 would be ≈ (Σ_sampled wprod)·f0, a small
        // fraction; the self-normalized m1/wsum ≈ f0.
        let (mut m1, mut wsum) = (0.0_f64, 0.0_f64);
        let sampled = enumerate_check_points(&grids, &feats, |x_cells, rep_bins| {
            let wp = joint_weight(&w, &feats, x_cells)?;
            m1 += wp * model.ensemble_f64(rep_bins)?;
            wsum += wp;
            Ok(())
        })
        .unwrap();
        assert!(sampled, "the tiny cap must force sampling");
        assert!(
            wsum < 0.99,
            "a partial sample must under-cover the weight, got {wsum}"
        );
        let self_norm_mean = m1 / wsum;
        assert!(
            (self_norm_mean - bank.f0).abs() < 0.25 * (1.0 + bank.f0.abs()),
            "self-normalized mean {self_norm_mean} should recover f0 {}",
            bank.f0
        );

        // Negatives are still caught under sampling.
        let mut bad_mass = bank.clone();
        bad_mass.f0 += 10.0;
        assert!(matches!(
            check_mass_conservation(&model, &bad_mass, &w),
            Err(PbError::InvariantViolated {
                invariant: Invariant::MassConservation
            })
        ));
        let mut bad_var = bank.clone();
        if let Some(t) = bad_var.tables.first_mut() {
            t.variance += 1000.0;
        }
        assert!(matches!(
            check_variance_sum(&model, &bad_var, &w),
            Err(PbError::InvariantViolated {
                invariant: Invariant::VarianceSum
            })
        ));

        TEST_JOINT_CAP.with(|c| c.set(0));
    }

    fn binary_conditional_mean(
        leaves: &[f64; 8],
        p_low: [f64; 3],
        mask: usize,
        leaf: usize,
    ) -> f64 {
        let mut out = 0.0_f64;
        for (z, &leaf_value) in leaves.iter().enumerate() {
            if (z & mask) != (leaf & mask) {
                continue;
            }
            let mut weight = 1.0_f64;
            for (bit, &p) in p_low.iter().enumerate() {
                if (mask & (1usize << bit)) == 0 {
                    weight *= if (z & (1usize << bit)) != 0 {
                        p
                    } else {
                        1.0 - p
                    };
                }
            }
            out += leaf_value * weight;
        }
        out
    }

    fn binary_fanova_effects(leaves: &[f64; 8], p_low: [f64; 3]) -> [[f64; 8]; 8] {
        let mut effects = [[0.0_f64; 8]; 8];
        for mask in 0usize..8 {
            for leaf in 0usize..8 {
                let mut value = binary_conditional_mean(leaves, p_low, mask, leaf);
                let mut sub = mask;
                while sub > 0 {
                    sub = (sub - 1) & mask;
                    if sub != mask {
                        value -= effects[sub][leaf];
                    }
                }
                effects[mask][leaf] = value;
            }
        }
        effects
    }

    #[test]
    fn fixture_explain_passes_all_gates() {
        let m = fixture_model();
        let x = fixture_serve();
        for w in [RefMeasure::Uniform, RefMeasure::default()] {
            let bank = m.explain(&x, w.clone()).unwrap();
            assert_exact_decomposition(&m, &bank, &x).unwrap();
            check_feature_budget(&m).unwrap();
        }
    }

    /// P1 multi-channel: a HAND-BUILT model (not left to boosting's greedy search, which may
    /// or may not choose to combine channels) whose single tree splits on BOTH channels of one
    /// raw feature, realizing a genuine non-additively-separable pattern
    /// (`g(1,1)=6 != g(1,2)+g(2,1)-g(2,2)=4`) — the exact case the rejected "sum of two
    /// independently-purified tables" design would get wrong. The flattened joint-cell design
    /// must reproduce it exactly: all five I2 gates pass, `check_feature_budget` (I1) sees this
    /// as order-1 (both splits are the SAME raw feature), and the single collapsed table's
    /// value at each level equals the tree's actual (residual-inclusive) leaf value, not an
    /// additive approximation of it.
    #[test]
    fn fixture_multichannel_explain_passes_all_gates_with_a_genuine_interaction() {
        let m = fixture_multichannel_model();
        let x = fixture_multichannel_serve();
        check_feature_budget(&m).unwrap(); // order-1: both splits are raw feature 0
        for w in [RefMeasure::Uniform, RefMeasure::default()] {
            let bank = m
                .explain(&x, w.clone())
                .unwrap_or_else(|e| panic!("multichannel explain({w:?}) failed: {e:?}"));
            assert_exact_decomposition(&m, &bank, &x)
                .unwrap_or_else(|e| panic!("multichannel I2 gates failed under {w:?}: {e:?}"));
            // Exactly ONE table for raw feature 0 (order-1, collapsed) — never one per channel.
            let raw0_tables: Vec<_> = bank
                .tables
                .iter()
                .filter(|t| t.u.0.iter().any(|f| f.0 == 0))
                .collect();
            assert_eq!(
                raw0_tables.len(),
                1,
                "must collapse to one table, not one per channel"
            );
            assert_eq!(raw0_tables[0].u.order(), 1);
            // The lossless LUT-sum reproduces the GENUINE interaction at every level — in
            // particular level "aa" (mean_bin=1, count_bin=1) scores 6.0, which an additive
            // sum of separate channel tables cannot produce (it would give 4:
            // g(ab)+g(ba)-g(bb) = 2+2-0). `x_cells` is RAW-indexed (one raw feature here, with
            // two channels folded into its own joint cell) — one element, the joint cell id;
            // insertion order in `mean_enc.levels` above ("aa","ab","ba","bb") fixes joint
            // cells 1..4 respectively (cell 0 is the reserved, never-visited missing cell).
            for (joint_cell, want) in [(1u32, 6.0), (2, 2.0), (3, 2.0), (4, 0.0)] {
                let got = bank.score(&[joint_cell]).unwrap();
                assert!(
                    (got - want).abs() < 1e-6,
                    "joint cell {joint_cell}: got {got}, want {want} (residual interaction must \
                     survive the collapse)"
                );
            }
        }
    }

    #[test]
    fn fixture_uniform_bank_reconstructs_and_centers_intercept() {
        // The merged grid carries the explicit missing cell (cell 0), so the Uniform
        // intercept is the mean over ALL 9 cells of g (incl. the missing-routed leaf),
        // not the 4 finite corners: g sums to 0+2+0 + 2+6+2 + 0+2+0 = 14 ⇒ f0 = 14/9.
        let m = fixture_model();
        let x = fixture_serve();
        let bank = m.explain(&x, RefMeasure::Uniform).unwrap();
        assert!((bank.f0 - 14.0 / 9.0).abs() < 1e-9, "f0 = {}", bank.f0);
        // The bank reconstructs the ensemble at every finite corner (lossless LUT-sum).
        for (cells, want) in [([1, 1], 6.0), ([1, 2], 2.0), ([2, 1], 2.0), ([2, 2], 0.0)] {
            assert!((bank.score(&cells).unwrap() - want).abs() < 1e-6);
        }
        // A genuine pairwise interaction table is realized.
        assert!(bank.tables.iter().any(|t| t.u.order() == 2));
    }

    #[test]
    fn negative_reconstruction_is_caught() {
        let m = fixture_model();
        let x = fixture_serve();
        let mut bank = m.explain(&x, RefMeasure::Uniform).unwrap();
        // Perturb a main-effect finite cell: tables no longer reconstruct the ensemble.
        let main = bank
            .tables
            .iter_mut()
            .find(|t| t.u.order() == 1)
            .expect("a main effect");
        main.values.add(&[1], 1.0).unwrap();
        assert!(matches!(
            check_reconstruction(&m, &bank),
            Err(PbError::InvariantViolated {
                invariant: Invariant::Reconstruction
            })
        ));
    }

    #[test]
    fn reconstruction_enumerates_model_features_even_if_bank_omits_tables() {
        let m = fixture_model();
        let x = fixture_serve();
        let mut bank = m.explain(&x, RefMeasure::Uniform).unwrap();
        // Old bug shape: if the check domain came only from `bank.tables`, clearing
        // tables and setting f0 to the missing/missing score made reconstruction pass
        // vacuously at one point. The gate must inspect the model's realized features.
        bank.f0 = 0.0;
        bank.tables.clear();
        assert!(matches!(
            check_reconstruction(&m, &bank),
            Err(PbError::InvariantViolated {
                invariant: Invariant::Reconstruction
            })
        ));
    }

    #[test]
    fn negative_three_way_is_caught() {
        let m = fixture_model();
        let x = fixture_serve();
        let mut bank = m.explain(&x, RefMeasure::Uniform).unwrap();
        bank.tables[0].values.add(&[2], 0.75).unwrap();
        assert!(matches!(
            check_three_way_equal(&m, &bank),
            Err(PbError::InvariantViolated {
                invariant: Invariant::ThreeWayEqual
            })
        ));
    }

    #[test]
    fn negative_purity_is_caught() {
        let m = fixture_model();
        let x = fixture_serve();
        let w = {
            let grids = MergedGrids::from_model(&m).unwrap();
            build_weights(&x, &grids, &RefMeasure::Uniform, None).unwrap()
        };
        let mut bank = m.explain(&x, RefMeasure::Uniform).unwrap();
        // Add a constant to a whole b0=1 slice of the pairwise table → its conditional
        // mean is no longer zero (a residual main effect left in the 2-way).
        let pair = bank
            .tables
            .iter_mut()
            .find(|t| t.u.order() == 2)
            .expect("a pairwise table");
        pair.values.add(&[1, 1], 1.0).unwrap();
        pair.values.add(&[1, 2], 1.0).unwrap();
        pair.values.add(&[1, 0], 1.0).unwrap();
        assert!(matches!(
            check_purity(&m, &bank, &w),
            Err(PbError::InvariantViolated {
                invariant: Invariant::Decomposability
            })
        ));
    }

    #[test]
    fn repurify_raw_effects_restores_purity_and_recomputes_variance_after_a_mutation() {
        // Models the H3 fix: graduation-style smoothing mutates a table's own values without
        // touching its neighbors' cascade relationships, breaking Purity and leaving
        // `variance` stale (it is a function of `values`, cached at purification time). Feeding
        // the mutated bank's CURRENT table values back through `purify_raw_effects` as if they
        // were raw — the exact seam `apply_graduation_updates` uses — must restore Purity
        // exactly and recompute every table's variance from the mutated values, not the
        // pre-mutation ones.
        let m = fixture_model();
        let x = fixture_serve();
        let grids = MergedGrids::from_model(&m).unwrap();
        let w = build_weights(&x, &grids, &RefMeasure::Uniform, None).unwrap();
        let mut bank = m.explain(&x, RefMeasure::Uniform).unwrap();

        let pair_u = bank
            .tables
            .iter()
            .find(|t| t.u.order() == 2)
            .expect("a pairwise table")
            .u
            .clone();
        let stale_variance = bank.tables.iter().find(|t| t.u == pair_u).unwrap().variance;

        // Same mutation shape as `negative_purity_is_caught`: break the pairwise table's
        // conditional zero-mean, simulating GCV ridge smoothing.
        {
            let pair = bank.tables.iter_mut().find(|t| t.u == pair_u).unwrap();
            pair.values.add(&[1, 1], 1.0).unwrap();
            pair.values.add(&[1, 2], 1.0).unwrap();
            pair.values.add(&[1, 0], 1.0).unwrap();
        }
        assert!(
            check_purity(&m, &bank, &w).is_err(),
            "the mutation must actually break purity, or this test proves nothing"
        );

        let effects: Vec<RawEffect> = bank
            .tables
            .iter()
            .map(|t| RawEffect {
                u: t.u.clone(),
                values: t.values.clone(),
                support: t.support.clone(),
            })
            .collect();
        let factored = bank.factored.clone();
        let mut repurified =
            purify_raw_effects(bank.f0, bank.merged_grids.clone(), bank.w.clone(), effects)
                .unwrap();
        repurified.factored = factored;

        check_purity(&m, &repurified, &w).expect("re-purify must restore Purity");
        // Per-axis w-weighted means are ~0 on every table (what check_purity itself asserts,
        // spelled out here for the "purity of a re-purified bank" requirement directly).
        for table in &repurified.tables {
            let extents = table.values.shape();
            for p in 0..table.u.order() {
                let r = table.u.0[p];
                let axis_w = w.axis(r).unwrap();
                let mut means: BTreeMap<Vec<usize>, f64> = BTreeMap::new();
                walk_extents(&extents, |coord| {
                    let wp = axis_w[coord[p]];
                    let v = table.values.at(coord).unwrap();
                    *means.entry(drop_index(coord, p)).or_insert(0.0) += wp * v;
                    Ok(())
                })
                .unwrap();
                for m in means.values() {
                    assert!(m.abs() < 1e-9, "table {:?} axis {p} mean {m}", table.u);
                }
            }
        }

        let final_table = repurified.tables.iter().find(|t| t.u == pair_u).unwrap();
        assert_ne!(
            final_table.variance, stale_variance,
            "variance must be recomputed, not left at the pre-mutation value"
        );
        let independently_recomputed = table_variance(&pair_u, &final_table.values, &w).unwrap();
        assert!(
            (independently_recomputed - final_table.variance).abs() < 1e-9,
            "cached variance must match table_variance on the FINAL values"
        );
    }

    /// Bug #6: `purify_raw_effects` (the seam `apply_graduation_updates` re-purifies through)
    /// crashed on a P1 multi-channel joint categorical axis with "bank merged grid n_bins ..
    /// inconsistent with .. borders", because `validate_bank_grid` assumed every axis is
    /// numeric-shaped (`n_bins == borders.len() + 2`) — true for a real split-border axis, but
    /// not for a joint axis's placeholder grid (`borders` empty, `n_bins` = the true joint cell
    /// count). It only surfaced now because graduation is objective-gated (poisson/gamma +
    /// `graduate=True`), so every earlier multi-channel gate (fit on a non-graduating
    /// objective) never exercised this exact reconstruction path.
    #[test]
    fn purify_raw_effects_handles_a_joint_categorical_axis() {
        let m = fixture_multichannel_model();
        let x = fixture_multichannel_serve();
        let bank = m
            .explain_bank(&x, RefMeasure::Uniform, TableBudget::default(), false)
            .unwrap();

        // Confirm the fixture actually has a joint placeholder grid (raw feature 0, the
        // fixture's only raw feature) -- or this test proves nothing.
        assert_eq!(bank.merged_grids.len(), 1);
        let joint_grid = &bank.merged_grids[0];
        assert!(
            joint_grid.borders.is_empty() && joint_grid.n_bins > 2,
            "fixture must produce a joint placeholder grid (borders empty, n_bins = joint \
             cells), got borders={:?} n_bins={}",
            joint_grid.borders,
            joint_grid.n_bins
        );

        let effects: Vec<RawEffect> = bank
            .tables
            .iter()
            .map(|t| RawEffect {
                u: t.u.clone(),
                values: t.values.clone(),
                support: t.support.clone(),
            })
            .collect();
        let repurified =
            purify_raw_effects(bank.f0, bank.merged_grids.clone(), bank.w.clone(), effects)
                .expect("must not crash on a joint categorical axis (bug #6)");

        // The fix must not leak `from_border_grids`'s internal cells()-sizing padding into the
        // OUTPUT bank: the re-purified bank's stored grid must stay the canonical placeholder
        // shape, bit-for-bit identical to the input (`purify` changes cell values, never grid
        // shape).
        assert_eq!(
            repurified.merged_grids, bank.merged_grids,
            "re-purify must not alter the joint axis's stored grid shape"
        );

        // And the re-purified bank must still reproduce the ensemble exactly.
        let n = x.0.n_rows as usize;
        let mut scored = vec![0.0_f64; n];
        crate::scoring::score_bank_binned(&repurified, &m.schema.cat_encoders, &x.0, &mut scored)
            .unwrap();
        let ens = m.predict(&x.0, None).unwrap();
        for (s, e) in scored.iter().zip(&ens) {
            assert!(
                (*s - f64::from(*e)).abs() < 1e-6,
                "re-purified joint-axis bank must still reproduce the ensemble: {s} vs {e}"
            );
        }
    }

    #[test]
    fn add_bank_assign_recomputes_variance_for_summed_tables() {
        // add_bank_assign (the rebalance path, default-on for pruning) mutates matched tables'
        // VALUES cell-for-cell but must not leave their cached `variance` stale.
        let m = fixture_model();
        let x = fixture_serve();
        let grids = MergedGrids::from_model(&m).unwrap();
        let w = build_weights(&x, &grids, &RefMeasure::Uniform, None).unwrap();
        let mut base = m.explain(&x, RefMeasure::Uniform).unwrap();

        let main_u = base
            .tables
            .iter()
            .find(|t| t.u.order() == 1)
            .expect("a main-effect table")
            .u
            .clone();
        let stale_variance = base.tables.iter().find(|t| t.u == main_u).unwrap().variance;

        // A small correction over the SAME support, shaped exactly like a `purify_correction`
        // result (already pure and correctly-varianced on its own).
        let main_table = base.tables.iter().find(|t| t.u == main_u).unwrap();
        let shape = main_table.values.shape();
        let mut bumped = main_table.values.clone();
        bumped.add(&[0], 0.7).unwrap();
        let delta_effects = vec![RawEffect {
            u: main_u.clone(),
            values: bumped,
            support: Tensor::try_zeros(shape).unwrap(),
        }];
        let delta = purify_raw_effects(
            0.0,
            base.merged_grids.clone(),
            base.w.clone(),
            delta_effects,
        )
        .unwrap();

        add_bank_assign(&mut base, &delta).unwrap();

        let summed = base.tables.iter().find(|t| t.u == main_u).unwrap();
        let recomputed = table_variance(&main_u, &summed.values, &w).unwrap();
        assert!(
            (summed.variance - recomputed).abs() < 1e-9,
            "cached variance must match table_variance on the summed values"
        );
        assert_ne!(
            summed.variance, stale_variance,
            "variance must actually change after the add, not stay at the pre-add value"
        );
    }

    #[test]
    fn over_budget_model_violates_feature_budget() {
        assert!(matches!(
            check_feature_budget(&fixture_over_budget_model()),
            Err(PbError::InvariantViolated {
                invariant: Invariant::FeatureBudget
            })
        ));
    }

    #[test]
    fn approximate_model_refuses_to_export() {
        let m = fixture_over_budget_model();
        let x = fixture_serve();
        assert!(matches!(
            m.explain(&x, RefMeasure::Uniform),
            Err(PbError::ExactnessFirewall(_))
        ));
    }

    #[test]
    fn joint_measure_is_rejected_in_v1() {
        let m = fixture_model();
        let x = fixture_serve();
        assert!(matches!(
            m.explain(&x, RefMeasure::Joint),
            Err(PbError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn nonfinite_laplace_and_malformed_serve_matrix_are_rejected() {
        let m = fixture_model();
        let x = fixture_serve();
        assert!(matches!(
            m.explain(
                &x,
                RefMeasure::ProductMarginals {
                    laplace: f32::INFINITY
                }
            ),
            Err(PbError::InvalidConfig { .. })
        ));

        let mut bad = x.clone();
        bad.0.data[0][0] = u8::MAX;
        assert!(matches!(
            m.explain(&bad, RefMeasure::Uniform),
            Err(PbError::InvalidInput { .. })
        ));

        let mut bad = x;
        bad.0.grids[0].borders.clear();
        assert!(matches!(
            m.explain(&bad, RefMeasure::Uniform),
            Err(PbError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn tensor_rejects_unrepresentable_shapes_without_panicking() {
        assert!(matches!(
            Tensor::try_zeros(vec![0]),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            Tensor::try_zeros(vec![u32::MAX as usize + 1]),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            Tensor::from_vec(vec![0], Vec::new()),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn sparse_tensor_preserves_logical_dense_values() {
        let mut tensor = Tensor::try_sparse_zeros(vec![2, 3]).unwrap();
        assert!(tensor.is_sparse());
        assert_eq!(tensor.len(), 6);
        assert_eq!(tensor.values().as_ref(), &[0.0; 6]);

        tensor.add(&[1, 2], 1.5).unwrap();
        tensor.add(&[0, 1], -2.0).unwrap();
        tensor.add(&[1, 2], -1.5).unwrap();

        assert_eq!(tensor.at(&[1, 2]), Some(0.0));
        assert_eq!(tensor.at(&[0, 1]), Some(-2.0));
        assert_eq!(tensor.values().as_ref(), &[0.0, -2.0, 0.0, 0.0, 0.0, 0.0]);

        tensor.add_scalar(0.25).unwrap();
        assert!(!tensor.is_sparse());
        assert_eq!(
            tensor.values().as_ref(),
            &[0.25, -1.75, 0.25, 0.25, 0.25, 0.25]
        );
    }

    #[test]
    fn try_values_matches_values_on_the_happy_path() {
        // try_values/values must agree whenever the sparse-to-dense conversion succeeds —
        // try_values only differs in how it reports FAILURE (propagate vs. silently degrade).
        let mut tensor = Tensor::try_sparse_zeros(vec![2, 2]).unwrap();
        tensor.add(&[0, 1], 3.5).unwrap();
        assert_eq!(
            tensor.try_values().unwrap().as_ref(),
            tensor.values().as_ref()
        );

        let dense = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(
            dense.try_values().unwrap().as_ref(),
            dense.values().as_ref()
        );
    }

    #[test]
    fn g3_real_fit_two_feature_additive() {
        // Gate G3: a real fitted model, explained, passes all five checks.
        let n = 64usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 3.0 { 10.0 } else { 20.0 };
                let b = if x1[i] <= 2.0 { 5.0 } else { 0.0 };
                a + b
            })
            .collect();
        let (model, x) = fit(&[x0, x1], &y, exact_cfg(30));
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        assert_exact_decomposition(&model, &bank, &x).unwrap();
        // Sobol importances exist and sum to ~1 (product w).
        let sobol = bank.sobol();
        let s: f64 = sobol.iter().map(|(_, v)| v).sum();
        assert!((s - 1.0).abs() < 1e-6, "sobol sum {s}");
    }

    #[test]
    fn g3_real_fit_three_feature() {
        let n = 64usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 2 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| ((i / 2) % 2 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| ((i / 4) % 2 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 1.0 { 0.0 } else { 4.0 };
                let b = if x1[i] <= 1.0 { 0.0 } else { 2.0 };
                let c = if x2[i] <= 1.0 { 0.0 } else { 1.0 };
                a + b + c - 3.5
            })
            .collect();
        let (model, x) = fit(&[x0, x1, x2], &y, exact_cfg(40));
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        assert_exact_decomposition(&model, &bank, &x).unwrap();
    }

    #[test]
    fn correction_preserves_g0_and_shifts_prediction() {
        // A 2-feature target WITH a {0,1} interaction, so the model realizes the pair.
        let n = 64usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| ((i / 4) % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 2.0 { 10.0 } else { 20.0 };
                let b = if x1[i] <= 2.0 { 5.0 } else { 0.0 };
                let ab = if x0[i] <= 2.0 && x1[i] <= 2.0 {
                    7.0
                } else {
                    0.0
                };
                a + b + ab
            })
            .collect();
        let (mut model, x) = fit(&[x0, x1], &y, exact_cfg(40));

        // Baseline decomposition passes the five gates.
        let base_bank = model.explain(&x, RefMeasure::default()).unwrap();
        assert_exact_decomposition(&model, &base_bank, &x).unwrap();
        let row = [1u8, 1u8];
        let base_score = model.ensemble_f64(&row).unwrap();

        // Attach a raw cell-basis correction: a main on axis 0 AND the {0,1} pair.
        let mut corr = correction_scaffold(&model, &[vec![0], vec![0, 1]]).unwrap();
        for (c, v) in corr.tables[0].values.iter_mut().enumerate() {
            *v = 0.25 + 0.1 * c as f64;
        }
        for (c, v) in corr.tables[1].values.iter_mut().enumerate() {
            *v = -0.2 + 0.05 * c as f64;
        }
        model.correction = Some(corr);
        model.validate().unwrap();

        // Prediction moves by EXACTLY the correction delta at `row`.
        let corr_score = model.ensemble_f64(&row).unwrap();
        let delta = model.correction_delta(&row).unwrap();
        assert!(delta.abs() > 1e-9, "correction must be nonzero");
        assert!((corr_score - base_score - delta).abs() < 1e-12);

        // G0 WITH the correction present: explain re-runs all five exactness gates internally
        // (reconstruction ties ensemble_f64 == bank.score), and they must still pass.
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        assert_exact_decomposition(&model, &bank, &x).unwrap();
        // The correction's marginal flowed into the decomposition (a {0} or {0,1} table exists).
        assert!(!bank.tables.is_empty());
    }

    #[test]
    fn empty_model_explains_to_intercept_only() {
        // A constant target → no trees → the bank is just f0 (== mean), gates trivial.
        let (model, x) = fit(
            &[vec![1.0, 2.0, 3.0, 4.0]],
            &[5.0, 5.0, 5.0, 5.0],
            exact_cfg(10),
        );
        assert!(model.trees.is_empty());
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        assert!(bank.tables.is_empty());
        assert!((bank.f0 - 5.0).abs() < 1e-5);
        assert_exact_decomposition(&model, &bank, &x).unwrap();
    }

    #[test]
    fn shap_sums_to_score_minus_f0() {
        let n = 48usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 3 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| if x0[i] <= 2.0 { -3.0 } else { 4.0 } + x1[i] * 0.5)
            .collect();
        let (model, x) = fit(&[x0, x1], &y, exact_cfg(25));
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        // At every realized cell tuple, Σ φ_i == score − f0.
        let g = MergedGrids::from_model(&model).unwrap();
        for c0 in 0..g.cells(FeatureId(0)).unwrap() {
            for c1 in 0..g.cells(FeatureId(1)).unwrap() {
                let cells = [c0 as u32, c1 as u32];
                let phi: f64 = bank.shap(&cells).unwrap().iter().sum();
                let score = bank.score(&cells).unwrap();
                assert!((phi - (score - bank.f0)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn missing_left_routing_is_honored_end_to_end() {
        // The §08 load-bearing claim: the merged-grid missing cell (cell 0) honors a
        // tree's learned `missing_left` exactly and is NOT collapsed into a finite
        // interval, so tree-sum == table-sum even for rows missing on a split axis. A
        // depth-1 tree with `missing_left: true` routes missing to the LOW leaf; the
        // serve data includes a genuine bin-0 (missing) row.
        use crate::data::{AxisKind, AxisProvenance, BinnedMatrix};
        use crate::engine::{ExactnessMode, ModelSchema, ObliviousTree, Split};
        use crate::loss::{Link, LossId, ObjectiveTag};
        // leaves: idx0 (high, bin2 side) = 3; idx1 (low, bin1 + missing) = 7.
        let tree = ObliviousTree {
            splits: vec![Split {
                axis: 0,
                bin_le: 1,
                missing_left: true,
            }],
            leaves: vec![3.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            depth: 1,
        };
        let model = Model {
            f0: 0.0,
            trees: vec![(1.0, tree)],
            grids: vec![fixture_grid()],
            provenance: vec![AxisProvenance {
                raw: FeatureId(0),
                kind: AxisKind::Numeric,
            }],
            link: Link::Identity,
            mode: ExactnessMode::Exact,
            schema: ModelSchema {
                feature_names: vec!["x0".into()],
                feature_kinds: vec![AxisKind::Numeric],
                cat_encoders: crate::cat::CatEncoderStore::new(),
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
            bag_intercepts: None,
            bag_in_bag: None,
            delta_step_gate: None,
            fit_report: None,
        };
        let x = ServeBinnedMatrix(BinnedMatrix {
            data: vec![vec![0, 1, 2, 1, 2]], // a genuine missing (bin 0) row at index 0
            n_rows: 5,
            grids: vec![fixture_grid()],
            provenance: model.provenance.clone(),
        });
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        assert_exact_decomposition(&model, &bank, &x).unwrap();
        // The missing cell reconstructs the missing-ROUTED leaf (7), exactly as the model
        // scores a missing row — never the bin-2/high leaf (3).
        assert!((bank.score(&[0]).unwrap() - model.ensemble_f64(&[0]).unwrap()).abs() < 1e-6);
        assert!((model.ensemble_f64(&[0]).unwrap() - 7.0).abs() < 1e-6);
        // ...and the missing cell (7) differs from the bin-2 finite cell (3): proof the
        // missing routing is not silently collapsed into the first/last finite interval.
        assert!((bank.score(&[2]).unwrap() - 3.0).abs() < 1e-6);
        assert!((bank.score(&[1]).unwrap() - 7.0).abs() < 1e-6);
    }

    #[test]
    fn repeated_split_levels_accumulate_as_lower_order_support() {
        use crate::data::{AxisKind, AxisProvenance, BinnedMatrix};
        use crate::engine::{ExactnessMode, ModelSchema, ObliviousTree, Split};
        use crate::loss::{Link, LossId, ObjectiveTag};

        let grid = BorderGrid {
            borders: vec![1.5, 2.5],
            n_bins: 4,
            missing_bin: 0,
        };
        let provenance = vec![AxisProvenance {
            raw: FeatureId(0),
            kind: AxisKind::Numeric,
        }];
        let tree = ObliviousTree::try_new(
            vec![
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
            ],
            // bin3/missing -> idx0 = 30, bin2 -> idx2 = 20, bin1 -> idx3 = 10.
            vec![30.0, 0.0, 20.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            &provenance,
        )
        .unwrap();
        let model = Model {
            f0: 0.0,
            trees: vec![(1.0, tree)],
            grids: vec![grid.clone()],
            provenance: provenance.clone(),
            link: Link::Identity,
            mode: ExactnessMode::Exact,
            schema: ModelSchema {
                feature_names: vec!["x0".into()],
                feature_kinds: vec![AxisKind::Numeric],
                cat_encoders: crate::cat::CatEncoderStore::new(),
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
            bag_intercepts: None,
            bag_in_bag: None,
            delta_step_gate: None,
            fit_report: None,
        };
        let x = ServeBinnedMatrix(BinnedMatrix {
            data: vec![vec![0, 1, 2, 3]],
            n_rows: 4,
            grids: vec![grid],
            provenance,
        });

        check_feature_budget(&model).unwrap();
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        assert_eq!(bank.tables.len(), 1);
        assert_eq!(bank.tables[0].u.order(), 1);
        assert_exact_decomposition(&model, &bank, &x).unwrap();
        for bin in [0_u8, 1, 2, 3] {
            assert!(
                (bank.score(&[u32::from(bin)]).unwrap() - model.ensemble_f64(&[bin]).unwrap())
                    .abs()
                    < 1e-6
            );
        }
    }

    /// **The P-D4 exactness proof, on the hazard P-D0 first fenced off.**
    ///
    /// P-D0 made `build_tree_box` REJECT a tree that reuses a raw feature across levels,
    /// because indexing `low`/`level_of` by support position clobbers a slot and yields a
    /// silently wrong box. P-D4 removes the need for that fence: `build_tree_boxes`
    /// DECOMPOSES a lifted tree into one rank-1 box per realized region tuple. The contract
    /// is now that the decomposition REPRODUCES the tree exactly, so assert that directly —
    /// at every merged cell tuple, `Σ_boxes octant[side(x)]` must equal `alpha * leaf(x)`,
    /// which is what `leaf_index_for_tuple` (the dense path) computes.
    #[test]
    fn lifted_tree_box_decomposition_reproduces_the_tree_exactly() {
        use crate::data::{AxisKind, AxisProvenance};
        use crate::engine::{ExactnessMode, ModelSchema, ObliviousTree, Split};
        use crate::loss::{Link, LossId, ObjectiveTag};

        let grid = BorderGrid {
            borders: vec![1.5, 2.5, 3.5, 4.5],
            n_bins: 6,
            missing_bin: 0,
        };
        let provenance: Vec<AxisProvenance> = (0..3u32)
            .map(|a| AxisProvenance {
                raw: FeatureId(a),
                kind: AxisKind::Numeric,
            })
            .collect();
        // Depth 5, order 3: raw 0 tested THREE times (nested thresholds 1/3/4), raws 1 and 2
        // once each. Regions: 4 x 2 x 2 on axis 0/1/2 -> the rank-1 box cannot hold it.
        let splits = vec![
            Split {
                axis: 0,
                bin_le: 1,
                missing_left: false,
            },
            Split {
                axis: 1,
                bin_le: 2,
                missing_left: true,
            },
            Split {
                axis: 0,
                bin_le: 3,
                missing_left: false,
            },
            Split {
                axis: 2,
                bin_le: 2,
                missing_left: false,
            },
            Split {
                axis: 0,
                bin_le: 4,
                missing_left: true,
            },
        ];
        let mut leaves = vec![0.0_f32; 32];
        for (i, v) in leaves.iter_mut().enumerate() {
            *v = ((i * 37 % 23) as f32) - 11.0;
        }
        let tree = ObliviousTree::try_new(splits, leaves, &provenance).unwrap();
        let alpha = 0.75_f32;
        let model = Model {
            f0: 0.0,
            trees: vec![(alpha, tree.clone())],
            grids: vec![grid.clone(), grid.clone(), grid],
            provenance: provenance.clone(),
            link: Link::Identity,
            mode: ExactnessMode::Exact,
            schema: ModelSchema {
                feature_names: vec!["x0".into(), "x1".into(), "x2".into()],
                feature_kinds: vec![AxisKind::Numeric; 3],
                cat_encoders: crate::cat::CatEncoderStore::new(),
                class_labels: None,
                objective: ObjectiveTag {
                    link: Link::Identity,
                    loss: LossId::SquaredError,
                    tweedie_rho: None,
                },
            },
            schema_version: crate::serialize::SCHEMA_VERSION,
            correction: None,
            bag_spans: None,
            bag_intercepts: None,
            bag_in_bag: None,
            delta_step_gate: None,
            fit_report: None,
        };
        let grids = MergedGrids::from_model(&model).unwrap();
        let u_ids = [FeatureId(0), FeatureId(1), FeatureId(2)];
        let boxes = build_tree_boxes(&model, f64::from(alpha), &tree, &u_ids, &grids).unwrap();
        assert!(
            boxes.len() > 1,
            "a reuse-bearing tree must decompose into MORE than one rank-1 box"
        );

        // Exactness at every merged cell tuple, against the dense path's own leaf lookup.
        let n0 = grids.cells(FeatureId(0)).unwrap();
        let n1 = grids.cells(FeatureId(1)).unwrap();
        let n2 = grids.cells(FeatureId(2)).unwrap();
        for c0 in 0..n0 {
            for c1 in 0..n1 {
                for c2 in 0..n2 {
                    let tuple = [c0, c1, c2];
                    let leaf_idx =
                        leaf_index_for_tuple(&model, &tree, &grids, &u_ids, &tuple).unwrap();
                    let want = f64::from(alpha) * f64::from(tree.leaves[leaf_idx]);
                    let mut got = 0.0_f64;
                    for (p, low) in &boxes {
                        let mut idx = 0usize;
                        for (d, &c) in tuple.iter().enumerate() {
                            idx |= usize::from(low[d][c]) << d;
                        }
                        got += p[idx];
                    }
                    assert!(
                        (got - want).abs() < 1e-9,
                        "cell {tuple:?}: decomposition {got} != tree {want}"
                    );
                }
            }
        }

        // And the RANK-1 case still returns exactly ONE box — that is what keeps a depth-3
        // model's `FactoredBox` bytes (which ARE serialized, inside the tables document)
        // identical to a pre-lift build's.
        let rank1_splits = vec![
            Split {
                axis: 0,
                bin_le: 1,
                missing_left: false,
            },
            Split {
                axis: 1,
                bin_le: 2,
                missing_left: true,
            },
            Split {
                axis: 2,
                bin_le: 2,
                missing_left: false,
            },
        ];
        let rank1 = ObliviousTree::try_new(rank1_splits, vec![0.5_f32; 8], &provenance).unwrap();
        let one = build_tree_boxes(&model, 1.0, &rank1, &u_ids, &grids).unwrap();
        assert_eq!(one.len(), 1);
    }

    /// P-D0: the four duplicated I1 predicates are now one function; pin its truth table
    /// (including the `distinct <= depth` clause the depth lift must NOT relax).
    #[test]
    fn i1_shape_predicate_truth_table() {
        use crate::engine::{i1_depth_ok, i1_shape_ok, MAX_DEPTH, MAX_ORDER};

        assert!(!i1_depth_ok(0));
        assert!(i1_depth_ok(1));
        assert!(i1_depth_ok(MAX_DEPTH));
        assert!(!i1_depth_ok(MAX_DEPTH + 1));

        // depth 0 and "no distinct raw features" are always illegal.
        assert!(!i1_shape_ok(0, 0));
        assert!(!i1_shape_ok(1, 0));
        // distinct may never exceed depth (a tree cannot test 2 features in 1 level) ...
        assert!(!i1_shape_ok(1, 2));
        // ... nor the frozen order cap, however deep the tree gets.
        assert!(!i1_shape_ok(MAX_DEPTH, MAX_ORDER + 1));
        // Reuse is legal at every depth: distinct < depth is the whole point of the lift.
        for depth in 1..=MAX_DEPTH {
            for distinct in 1..=depth.min(MAX_ORDER) {
                assert!(
                    i1_shape_ok(depth, distinct),
                    "depth {depth} distinct {distinct}"
                );
            }
        }
    }

    /// The high-order lift set `MAX_ORDER == MAX_DEPTH`, which SUBSUMES the order clause of
    /// `i1_shape_ok`: `distinct <= depth <= MAX_DEPTH == MAX_ORDER` already implies
    /// `distinct <= MAX_ORDER`, so no `(depth, distinct)` pair can violate the order cap
    /// without violating the depth cap too.
    ///
    /// This is worth an explicit test rather than a comment, because two negative fixtures
    /// (`fixture_over_budget_model` and `split::feature_budget_is_enforced_at_construction`)
    /// still SAY they prove an order violation and now in fact prove a depth one. They keep
    /// passing either way, which is exactly how a test quietly stops testing its own claim.
    ///
    /// If a later lift moves the caps apart again — most plausibly by holding order below
    /// depth — this assertion fires and points at the two fixtures that then become
    /// meaningful again.
    #[test]
    fn i1_order_clause_is_subsumed_at_equal_caps() {
        use crate::engine::{i1_shape_ok, MAX_DEPTH, MAX_ORDER};

        assert_eq!(
            MAX_ORDER, MAX_DEPTH,
            "the caps have moved apart; the order clause of i1_shape_ok is independently \
             testable again — restore the ORDER-violation claim in fixture_over_budget_model \
             and in split::feature_budget_is_enforced_at_construction"
        );
        // The concrete consequence: there is no legal depth at which the order cap bites.
        for depth in 1..=MAX_DEPTH {
            assert!(
                !i1_shape_ok(depth, MAX_ORDER + 1),
                "depth {depth}: an over-order shape must still be refused"
            );
            // ...and the refusal is the depth/`distinct <= depth` clause doing the work,
            // since `MAX_ORDER + 1 > MAX_DEPTH >= depth` for every depth in range.
            assert!(MAX_ORDER + 1 > depth);
        }
    }

    #[test]
    fn wht8_coefficients_match_purified_single_tree_tables() {
        use crate::cat::CatEncoderStore;
        use crate::constraints::wht8_uniform;
        use crate::data::{AxisKind, AxisProvenance};
        use crate::engine::{ExactnessMode, ModelSchema, ObliviousTree, Split};
        use crate::loss::{Link, LossId, ObjectiveTag};

        let leaves = [0.0, 1.0, 3.0, 4.0, 8.0, -2.0, 5.0, 11.0];
        let tree = ObliviousTree {
            splits: vec![
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
                Split {
                    axis: 2,
                    bin_le: 1,
                    missing_left: false,
                },
            ],
            leaves: leaves.to_vec(),
            depth: 3,
        };
        let provenance: Vec<AxisProvenance> = (0..3u32)
            .map(|raw| AxisProvenance {
                raw: FeatureId(raw),
                kind: AxisKind::Numeric,
            })
            .collect();
        let model = Model {
            f0: 0.0,
            trees: vec![(1.0, tree)],
            grids: vec![fixture_grid(), fixture_grid(), fixture_grid()],
            provenance: provenance.clone(),
            link: Link::Identity,
            mode: ExactnessMode::Exact,
            schema: ModelSchema {
                feature_names: vec!["x0".into(), "x1".into(), "x2".into()],
                feature_kinds: vec![AxisKind::Numeric; 3],
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
            bag_intercepts: None,
            bag_in_bag: None,
            delta_step_gate: None,
            fit_report: None,
        };
        let serve = ServeBinnedMatrix(crate::data::BinnedMatrix {
            data: vec![
                vec![2, 1, 2, 1, 2, 1, 2, 1],
                vec![2, 2, 1, 1, 2, 2, 1, 1],
                vec![2, 2, 2, 2, 1, 1, 1, 1],
            ],
            n_rows: 8,
            grids: model.grids.clone(),
            provenance,
        });
        // Dense reference via Error policy (default Factored now always factors order-3), so the
        // order-3 (mask 7) purified table is materialized densely for the per-cell WHT comparison.
        let dense_ref = TableBudget {
            max_table_cells: 2_000_000,
            max_bank_cells: 32_000_000,
            on_overflow: OverflowPolicy::Error,
        };
        let bank = model
            .explain_with_budget(&serve, RefMeasure::Uniform, dense_ref)
            .unwrap();
        let leaves_f64 = leaves.map(f64::from);
        let coeffs = wht8_uniform(leaves_f64).coeffs;
        let half_effects = binary_fanova_effects(&leaves_f64, [0.5; 3]);
        for (mask, values) in half_effects.iter().enumerate() {
            for (leaf, &effect) in values.iter().enumerate() {
                let sign = if ((mask & leaf).count_ones() & 1) == 0 {
                    1.0
                } else {
                    -1.0
                };
                assert!((effect - sign * coeffs[mask]).abs() < 1.0e-10);
            }
        }

        // Under the §08 Uniform measure the missing cell is a third cell. With
        // `missing_left=false`, missing routes with the high side, so the induced
        // binary cut weights are P(low)=1/3 and P(high)=2/3 on every axis.
        let effects = binary_fanova_effects(&leaves_f64, [1.0 / 3.0; 3]);
        assert!((bank.f0 - effects[0][0]).abs() < 1.0e-10);

        for (mask, effect_values) in effects.iter().enumerate().skip(1) {
            let ids: Vec<u32> = (0..3)
                .filter(|bit| (mask & (1usize << bit)) != 0)
                .map(|bit| bit as u32)
                .collect();
            let u = FeatureSet::new(&ids);
            let table = bank
                .tables
                .iter()
                .find(|table| table.u == u)
                .expect("purified table for WHT mask");
            for (leaf, &expected) in effect_values.iter().enumerate() {
                let coord: Vec<usize> = ids
                    .iter()
                    .map(|id| {
                        if (leaf & (1usize << usize::try_from(*id).unwrap())) != 0 {
                            1
                        } else {
                            2
                        }
                    })
                    .collect();
                let got = table.values.at(&coord).unwrap();
                assert!(
                    (got - expected).abs() < 1.0e-9,
                    "mask {mask:03b} leaf {leaf:03b}: got {got}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn recompute_under_is_exactness_preserving() {
        let n = 48usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| if x0[i] <= 2.0 { 1.0 } else { -2.0 } + if x1[i] <= 2.0 { 0.5 } else { 1.5 })
            .collect();
        let (model, x) = fit(&[x0, x1], &y, exact_cfg(30));
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        // Recompute under Uniform: still reconstructs the ensemble (sum conserved).
        let re = bank.recompute_under(RefMeasure::Uniform).unwrap();
        check_reconstruction(&model, &re).unwrap();
        check_three_way_equal(&model, &re).unwrap();
        // The new bank's stamped measure reflects the recompute.
        assert!(matches!(re.reference_measure(), RefMeasure::Uniform));
    }

    #[test]
    fn table_budget_error_trips_on_tiny_ceiling() {
        // A real triple, with a 1-cell budget → the firewall fires before allocation.
        let n = 32usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 2 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| ((i / 2) % 2 + 1) as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (x0[i] + x1[i]) * 0.3).collect();
        let (model, _) = fit(&[x0, x1], &y, exact_cfg(10));
        let grids = MergedGrids::from_model(&model).unwrap();
        let budget = TableBudget {
            max_table_cells: 1,
            max_bank_cells: 1,
            on_overflow: OverflowPolicy::Error,
        };
        assert!(matches!(
            accumulate(&model, &grids, &budget),
            Err(PbError::TableBudget { .. })
        ));
    }

    #[test]
    fn sparse_fallback_is_exact_and_serializable_at_permissive_density_threshold() {
        // density_threshold=1.0 is a deliberate "accept any density, just don't hard-fail"
        // opt-in (spec §08's documented escape hatch). An oblivious
        // tree densely populates ~100% of its support cube, so 1.0 is the ONLY threshold under
        // which SparseFallback can succeed for a tree-realized support — but it MUST still
        // succeed there, and the resulting bank must still be exact (not silently degrade to
        // dropped values just because the backing is sparse). xtask's
        // `adversarial_artifact_is_seed_stable_and_uses_sparse_fallback` exercises the
        // identical scenario end to end via the release-preflight artifact; this is the
        // core-level pin for the same contract.
        let model = fixture_model();
        let x = fixture_serve();
        let grids = MergedGrids::from_model(&model).unwrap();
        let budget = TableBudget {
            max_table_cells: 1,
            max_bank_cells: 1,
            on_overflow: OverflowPolicy::SparseFallback {
                density_threshold: 1.0,
            },
        };
        let (raw, _) = accumulate(&model, &grids, &budget).unwrap();
        assert!(raw.tables.values().any(|table| table.values.is_sparse()));
        verify_raw_accumulation(&model, &raw, &[], &grids).unwrap();

        let weights = build_weights(&x, &grids, &RefMeasure::Uniform, None).unwrap();
        let mut bank = purify(raw, &weights, &grids, PurifyMode::SinglePass).unwrap();
        fill_support(&mut bank, &grids, &x, None).unwrap();
        assert!(bank.tables.iter().any(|table| table.values.is_sparse()));

        check_reconstruction(&model, &bank).unwrap();
        check_mass_conservation(&model, &bank, &weights).unwrap();
        check_purity(&model, &bank, &weights).unwrap();
        check_variance_sum(&model, &bank, &weights).unwrap();
        check_three_way_equal(&model, &bank).unwrap();

        let encoded = bincode::serde::encode_to_vec(&bank, bincode::config::standard()).unwrap();
        let (decoded, consumed): (TableBank, usize) =
            bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded, bank);
        check_reconstruction(&model, &decoded).unwrap();
    }

    #[test]
    fn sparse_fallback_fails_before_allocate_under_a_restrictive_density_threshold() {
        // R-TABLEBUDGET: an over-budget tree-realized support under a RESTRICTIVE
        // density_threshold must fail before materializing the (near-100%-dense, for a
        // tree-realized support) sparse tensor — not after walking the whole ensemble into
        // it. The occupancy cap is enforced INSIDE the walk (`accumulate`'s per-cell check
        // against `density_threshold * table_cells`), so a restrictive threshold like this
        // (spec default is 0.05) must still fail fast rather than "succeed" the way it used
        // to at any threshold (the bug this closes).
        let model = fixture_model();
        let x = fixture_serve();
        let budget = TableBudget {
            max_table_cells: 1,
            max_bank_cells: 1,
            on_overflow: OverflowPolicy::SparseFallback {
                density_threshold: 0.05,
            },
        };
        assert!(matches!(
            model.explain_with_budget(&x, RefMeasure::Uniform, budget),
            Err(PbError::TableBudget { .. })
        ));
    }

    #[test]
    fn sparse_fallback_rejects_invalid_density_threshold() {
        let model = fixture_model();
        let grids = MergedGrids::from_model(&model).unwrap();
        for density_threshold in [f64::NAN, -0.1, 1.1] {
            let budget = TableBudget {
                on_overflow: OverflowPolicy::SparseFallback { density_threshold },
                ..TableBudget::default()
            };
            assert!(matches!(
                accumulate(&model, &grids, &budget),
                Err(PbError::InvalidConfig { .. })
            ));
        }
    }

    #[test]
    fn sparse_fallback_rejects_tables_above_density_threshold() {
        let model = fixture_model();
        let grids = MergedGrids::from_model(&model).unwrap();
        let budget = TableBudget {
            max_table_cells: 1,
            max_bank_cells: 1,
            on_overflow: OverflowPolicy::SparseFallback {
                density_threshold: 0.0,
            },
        };
        assert!(matches!(
            accumulate(&model, &grids, &budget),
            Err(PbError::TableBudget { .. })
        ));
    }

    // --- Purification identity proptests (spec §08.9) -----------------------

    /// A small random raw bank: one pairwise table over a 2×3 merged grid (cells incl.
    /// the missing cell), plus the uniform weight cache, for the purify identities.
    fn small_raw(values: &[f64]) -> (RawBank, WeightCache, MergedGrids) {
        // Two raw features with cells {2, 3}: feature 0 has no realized border (missing
        // + one finite cell); feature 1 has one realized border (missing + two finite).
        let per_raw = vec![
            MergedAxis {
                axis: 0,
                borders: vec![],
                model_border_index: vec![],
                model_n_bins: 2,
                joint: None,
            },
            MergedAxis {
                axis: 1,
                borders: vec![1.5],
                model_border_index: vec![0],
                model_n_bins: 3,
                joint: None,
            },
        ];
        let grids = MergedGrids { per_raw, n_axes: 2 };
        let axes = vec![
            grids.axis_id(FeatureId(0)).unwrap(),
            grids.axis_id(FeatureId(1)).unwrap(),
        ];
        let mut tens = Tensor::try_zeros(vec![2, 3]).unwrap();
        for (i, &v) in values.iter().enumerate() {
            let c = [i / 3, i % 3];
            tens.set(&c, v).unwrap();
        }
        let mut tables = BTreeMap::new();
        let u = FeatureSet::new(&[0, 1]);
        tables.insert(
            u.clone(),
            RawTable {
                u,
                axes,
                values: tens,
            },
        );
        let raw = RawBank { f0: 0.0, tables };
        let w = WeightCache {
            per_axis: vec![vec![0.5, 0.5], vec![1.0 / 3.0; 3]],
            kind: RefMeasure::Uniform,
        };
        (raw, w, grids)
    }

    fn purified_values(values: &[f64]) -> TableBank {
        let (raw, w, grids) = small_raw(values);
        purify(raw, &w, &grids, PurifyMode::SinglePass).unwrap()
    }

    fn bank_full(bank: &TableBank) -> (f64, BTreeMap<FeatureSet, Vec<f64>>) {
        let mut m = BTreeMap::new();
        for t in &bank.tables {
            let mut v = Vec::new();
            let ext = t.values.shape();
            walk_extents(&ext, |c| {
                v.push(t.values.at(c).unwrap());
                Ok(())
            })
            .unwrap();
            m.insert(t.u.clone(), v);
        }
        (bank.f0, m)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Purity: every purified bank passes the slice-mean-zero check under its w.
        #[test]
        fn prop_purify_is_pure(values in prop::collection::vec(-5.0f64..5.0, 6)) {
            let bank = purified_values(&values);
            let m = fixture_model();
            let (_, w, _) = small_raw(&values);
            prop_assert!(check_purity(&m, &bank, &w).is_ok());
        }

        /// Idempotence: purify(purify(T)) == purify(T) (already-pure stays put).
        #[test]
        fn prop_purify_idempotent(values in prop::collection::vec(-5.0f64..5.0, 6)) {
            let once = purified_values(&values);
            let twice = once.recompute_under(RefMeasure::Uniform).unwrap();
            let (f0a, a) = bank_full(&once);
            let (f0b, b) = bank_full(&twice);
            prop_assert!((f0a - f0b).abs() < 1e-9);
            for (u, va) in &a {
                let vb = b.get(u).unwrap();
                for (x, y) in va.iter().zip(vb) {
                    prop_assert!((x - y).abs() < 1e-9, "u={:?} {} vs {}", u, x, y);
                }
            }
        }

        /// Linearity: purify(αA) == α·purify(A) (cellwise, including the intercept).
        #[test]
        fn prop_purify_linear(
            values in prop::collection::vec(-5.0f64..5.0, 6),
            alpha in -3.0f64..3.0,
        ) {
            let base = purified_values(&values);
            let scaled_in: Vec<f64> = values.iter().map(|v| alpha * v).collect();
            let scaled = purified_values(&scaled_in);
            let (f0a, a) = bank_full(&base);
            let (f0b, b) = bank_full(&scaled);
            prop_assert!((alpha * f0a - f0b).abs() < 1e-7);
            for (u, va) in &a {
                let vb = b.get(u).unwrap();
                for (x, y) in va.iter().zip(vb) {
                    prop_assert!((alpha * x - y).abs() < 1e-7);
                }
            }
        }
    }

    // ---- 2026-09-27 fit-speed fast paths: each pinned bit-for-bit to the per-cell walk it
    // replaced (the walks below are verbatim copies of the pre-change code).

    fn stream(seed: u64, n: usize) -> Vec<f64> {
        let mut z = seed;
        (0..n)
            .map(|_| {
                z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut x = z;
                x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                x ^= x >> 31;
                (x >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
            })
            .collect()
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn center_along_walk(values: &mut Tensor, p: usize, axis_w: &[f64]) -> Tensor {
        let extents = values.shape();
        let sub_extents = drop_index(&extents, p);
        let mut means = Tensor::try_zeros(sub_extents).unwrap();
        walk_extents(&extents, |coord| {
            let wp = axis_w[coord[p]];
            let v = values.at(coord).unwrap();
            means.add(&drop_index(coord, p), wp * v)
        })
        .unwrap();
        walk_extents(&extents, |coord| {
            let m = means.at(&drop_index(coord, p)).unwrap();
            values.add(coord, -m)
        })
        .unwrap();
        means
    }

    #[test]
    fn dense_center_along_is_bit_identical_to_the_cell_walk() {
        for (s, shape) in [vec![9], vec![5, 7], vec![4, 3, 6], vec![2, 3, 2, 5]]
            .into_iter()
            .enumerate()
        {
            for p in 0..shape.len() {
                let n: usize = shape.iter().product();
                let data = stream(17 + s as u64, n);
                let w: Vec<f64> = stream(99 + p as u64, shape[p])
                    .iter()
                    .map(|x| x.abs() + 0.05)
                    .collect();
                let mut fast = Tensor::from_vec(shape.clone(), data.clone()).unwrap();
                let mut walk = Tensor::from_vec(shape.clone(), data).unwrap();
                let m_fast = center_along(&mut fast, p, &w).unwrap();
                let m_walk = center_along_walk(&mut walk, p, &w);
                assert_eq!(
                    bits(&m_fast.values()),
                    bits(&m_walk.values()),
                    "{shape:?} p={p}"
                );
                assert_eq!(
                    bits(&fast.values()),
                    bits(&walk.values()),
                    "{shape:?} p={p}"
                );
            }
        }
    }

    #[test]
    fn dense_table_variance_is_bit_identical_to_the_cell_walk() {
        for (s, shape) in [vec![9], vec![5, 7], vec![4, 3, 6]].into_iter().enumerate() {
            let order = shape.len();
            let n: usize = shape.iter().product();
            let values = Tensor::from_vec(shape.clone(), stream(5 + s as u64, n)).unwrap();
            let per_axis: Vec<Vec<f64>> = shape
                .iter()
                .enumerate()
                .map(|(d, &e)| {
                    let raw: Vec<f64> = stream(40 + d as u64, e)
                        .iter()
                        .map(|x| x.abs() + 0.1)
                        .collect();
                    let t: f64 = raw.iter().sum();
                    raw.iter().map(|x| x / t).collect()
                })
                .collect();
            let u = FeatureSet((0..order as u32).map(FeatureId).collect());
            let w = WeightCache {
                per_axis: per_axis.clone(),
                kind: RefMeasure::Uniform,
            };
            let fast = table_variance(&u, &values, &w).unwrap();
            let (mut m1, mut m2) = (0.0_f64, 0.0_f64);
            walk_extents(&shape, |coord| {
                let mut wprod = 1.0_f64;
                for (k, &cell) in coord.iter().enumerate() {
                    wprod *= per_axis[k][cell];
                }
                let v = values.at(coord).unwrap();
                m1 += wprod * v;
                m2 += wprod * v * v;
                Ok(())
            })
            .unwrap();
            assert_eq!(fast.to_bits(), (m2 - m1 * m1).to_bits(), "{shape:?}");
        }
    }

    #[test]
    fn parallel_fill_support_matches_the_sequential_reference() {
        let (model, x) = moderate_model();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let grids = MergedGrids::from_model(&model).unwrap();
        let n_rows = x.0.n_rows as usize;
        let row_cells: Vec<Vec<u32>> = (0..grids.n_raw_features())
            .map(|r| {
                grids
                    .axis(FeatureId(r as u32))
                    .unwrap()
                    .cell_for_every_row(&x)
                    .unwrap()
            })
            .collect();
        let mass: Vec<f32> = stream(3, n_rows)
            .iter()
            .map(|v| (v.abs() + 0.5) as f32)
            .collect();
        for m in [None, Some(mass.as_slice())] {
            let mut fast = bank.clone();
            let mut seq = bank.clone();
            fill_support(&mut fast, &grids, &x, m).unwrap();
            fill_support_sequential_reference(&mut seq, &row_cells, n_rows, m).unwrap();
            for (a, b) in fast.tables.iter().zip(&seq.tables) {
                assert_eq!(
                    bits(&a.support.values()),
                    bits(&b.support.values()),
                    "{:?}",
                    a.u
                );
            }
        }
    }

    #[test]
    fn dense_accumulate_matches_the_leaf_walk() {
        let (model, _x) = moderate_model();
        let grids = MergedGrids::from_model(&model).unwrap();
        let (raw, factored) = accumulate(&model, &grids, &TableBudget::default()).unwrap();
        let mut expected: BTreeMap<FeatureSet, Tensor> = BTreeMap::new();
        for (alpha, tree) in &model.trees {
            let u = tree_support(&model, tree).unwrap();
            if factored.contains(&u) {
                continue; // order-3 supports stay factored; only dense tables accumulate
            }
            let u_ids: Vec<FeatureId> = u.0.iter().copied().collect();
            let extents: Vec<usize> = u_ids.iter().map(|r| grids.cells(*r).unwrap()).collect();
            let t = expected
                .entry(u)
                .or_insert_with(|| Tensor::try_zeros(extents.clone()).unwrap());
            let alpha = f64::from(*alpha);
            walk_extents(&extents, |tuple| {
                let leaf_idx = leaf_index_for_tuple(&model, tree, &grids, &u_ids, tuple)?;
                t.add(tuple, alpha * f64::from(tree.leaves[leaf_idx]))
            })
            .unwrap();
        }
        assert_eq!(raw.tables.len(), expected.len());
        for (u, t) in &expected {
            assert_eq!(
                bits(&raw.tables[u].values.values()),
                bits(&t.values()),
                "{u:?}"
            );
        }
    }

    #[test]
    fn purify_is_bit_identical_across_thread_counts() {
        let (model, x) = moderate_model();
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| model.explain(&x, RefMeasure::Uniform).unwrap())
        };
        let (a, b) = (run(1), run(6));
        assert_eq!(a.f0.to_bits(), b.f0.to_bits());
        for (ta, tb) in a.tables.iter().zip(&b.tables) {
            assert_eq!(ta.u, tb.u);
            assert_eq!(
                bits(&ta.values.values()),
                bits(&tb.values.values()),
                "{:?}",
                ta.u
            );
            assert_eq!(ta.variance.to_bits(), tb.variance.to_bits());
        }
    }

    #[test]
    fn packed_factored_eval_is_bit_identical_to_eval() {
        let (model, x) = moderate_model();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        assert!(
            !bank.factored.is_empty(),
            "fixture should realize an order-3 support"
        );
        let n_raw = bank.merged_grids.len();
        for ft in &bank.factored {
            let pk = ft.packed().expect("uniform boxes pack");
            let extents: Vec<usize> = ft.axes.iter().map(|a| a.cells as usize).collect();
            let mut x_cells = vec![0u32; n_raw];
            walk_extents(&extents, |coord| {
                for (a, &c) in ft.axes.iter().zip(coord) {
                    x_cells[a.raw.0 as usize] = c as u32;
                }
                let (fast, slow) = (pk.eval(&x_cells).unwrap(), ft.eval(&x_cells).unwrap());
                assert_eq!(fast.to_bits(), slow.to_bits(), "{:?} at {coord:?}", ft.u);
                Ok(())
            })
            .unwrap();
        }
    }

    #[test]
    fn packed_table_eval_is_bit_identical_to_eval() {
        let (model, x) = moderate_model();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let n_raw = bank.merged_grids.len();
        for t in &bank.tables {
            let pt = t.packed().expect("dense tables pack");
            let extents: Vec<usize> = t.axes.iter().map(|a| a.merged_cells() as usize).collect();
            let mut x_cells = vec![0u32; n_raw];
            walk_extents(&extents, |coord| {
                for (a, &c) in t.axes.iter().zip(coord) {
                    x_cells[a.raw.0 as usize] = c as u32;
                }
                let (fast, slow) = (pt.eval(&x_cells).unwrap(), t.eval(&x_cells).unwrap());
                assert_eq!(fast.to_bits(), slow.to_bits(), "{:?} at {coord:?}", t.u);
                Ok(())
            })
            .unwrap();
        }
    }
}
