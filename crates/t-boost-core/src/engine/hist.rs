//! The full-precision histogram engine (spec §06.3, milestone M1.3).
//!
//! v1 accumulates full-precision [`GradHess`] into `f64` [`Hist`] cells. The build is
//! **feature-parallel** and, for large row sets, **row-chunk parallel** within each
//! axis: chunks accumulate sequentially, then reduce in deterministic chunk order into
//! a disjoint region of the `[leaf][axis][bin]` tensor. So the result is a FIXED-ORDER
//! fold — byte-identical regardless of thread count (the §1 determinism `[GATE]`) —
//! while narrow/small data keeps the lower-overhead sequential axis path.
//!
//! The subtraction trick (`Hist_child_larger = Hist_parent − Hist_child_smaller`, see
//! `subtract_sibling_into`) is integer-exact for `count`; for `g`/`h` it is bit-exact
//! for well-conditioned gradients (verified to ~1e-11 even at hundreds of thousands of
//! rows) but CAN drift under catastrophic magnitude spread (float non-associativity), so
//! it matches a direct build only within a tolerance. The unconditionally-associative,
//! exact-by-construction version is the quantized integer path
//! (`build_quantized_histogram`).

use crate::backend::{pb_seed, Stage};
use crate::data::BinnedMatrix;
use crate::engine::{GradScale, Hist, QuantGradHess};
use crate::error::PbError;
use crate::loss::GradHess;
use rayon::prelude::*;

const ROW_PAR_MIN_ROWS: usize = 32_768;
const ROW_PAR_CHUNK_ROWS: usize = 8_192;

fn oob(what: &'static str) -> impl Fn() -> PbError {
    move || PbError::Internal { what: what.into() }
}

fn offset2(
    first: usize,
    second: usize,
    stride: usize,
    what: &'static str,
) -> Result<usize, PbError> {
    first
        .checked_mul(stride)
        .and_then(|base| base.checked_add(second))
        .ok_or_else(oob(what))
}

fn offset3(
    first: usize,
    second: usize,
    third: usize,
    second_stride: usize,
    third_stride: usize,
    what: &'static str,
) -> Result<usize, PbError> {
    first
        .checked_mul(second_stride)
        .and_then(|base| base.checked_add(second))
        .and_then(|base| base.checked_mul(third_stride))
        .and_then(|base| base.checked_add(third))
        .ok_or_else(oob(what))
}

/// Build the per-level `[leaf][axis][bin]` histogram from full-precision `gh`
/// (spec §06.3). `rows` are the row ids in scope (the subsample; in fixed order),
/// `leaf_of_row[r]` is the leaf id (`0..n_leaves`) of row `r`, and `axes` are the
/// (sampled) feature columns to build. The bin stride is uniform — the max grid
/// `n_bins` over `axes` — so shorter-grid axes leave their high bins zeroed.
///
/// Feature-parallel + fixed-order row-chunk reduction ⇒ thread-count independent.
///
/// # Errors
/// [`PbError::Internal`] if any `axis`, row id, leaf id, or bin id is out of range
/// (the engine builds these so it is a bug, surfaced as a typed error not a panic).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_histogram(
    x: &BinnedMatrix,
    gh: &GradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    n_leaves: usize,
    axes: &[u32],
    weight: &[f32],
    unit_weight: bool,
) -> Result<Hist, PbError> {
    let mut max_bins = 0usize;
    for &a in axes {
        let grid = x
            .grids
            .get(a as usize)
            .ok_or_else(oob("axis has no grid"))?;
        max_bins = max_bins.max(usize::from(grid.n_bins));
    }
    let n_axes = axes.len();

    // Axes build in BLOCKS (disjoint output regions, one task per block): one row scan feeds the
    // whole block's tables, reading the shared per-row streams once per block instead of once per
    // axis. Bit-identical to the former one-task-per-axis build for any block width (see
    // `accumulate_axes_blocked`); `par_chunks` + order-preserving collect + flatten keep the
    // per-axis sub-histograms in axis order regardless of scheduling.
    let subs: Result<Vec<Vec<AxisHist>>, PbError> = axes
        .par_chunks(AXIS_BLOCK)
        .map(|block| {
            accumulate_axes_block_chunked(
                x,
                gh,
                rows,
                leaf_of_row,
                n_leaves,
                block,
                weight,
                unit_weight,
            )
        })
        .collect();
    let subs: Vec<AxisHist> = subs?.into_iter().flatten().collect();

    // Assemble the disjoint per-axis sub-histograms into the flat tensor, in axis order —
    // deterministic regardless of how rayon scheduled the per-axis tasks. Each `sub` was built at
    // its OWN (jagged) n_bins stride, so copy its `axis_bins` cells per leaf into the uniform
    // max_bins-strided tensor; the high cells (`axis_bins..max_bins`) stay zero, exactly as before.
    let mut hist = Hist::try_zeros(n_leaves, n_axes, max_bins)?;
    for (axis_pos, sub) in subs.iter().enumerate() {
        let axis = *axes.get(axis_pos).ok_or_else(oob("axis pos escaped"))?;
        let axis_bins = usize::from(
            x.grids
                .get(axis as usize)
                .ok_or_else(oob("axis has no grid"))?
                .n_bins,
        );
        for leaf in 0..n_leaves {
            for bin in 0..axis_bins {
                let src = offset2(leaf, bin, axis_bins, "axis histogram offset overflow")?;
                let dst = offset3(
                    leaf,
                    axis_pos,
                    bin,
                    n_axes,
                    max_bins,
                    "histogram assembly offset overflow",
                )?;
                let cell = sub.ghc.get(src).ok_or_else(oob("sub ghc"))?;
                *hist.g.get_mut(dst).ok_or_else(oob("assemble g"))? = cell.g;
                *hist.h.get_mut(dst).ok_or_else(oob("assemble h"))? = cell.h;
                *hist.wsum.get_mut(dst).ok_or_else(oob("assemble wsum"))? =
                    *sub.wsum.get(src).ok_or_else(oob("sub wsum"))?;
                *hist.count.get_mut(dst).ok_or_else(oob("assemble count"))? = cell.count;
            }
        }
    }
    Ok(hist)
}

/// Quantize full-precision gradients/hessians for the integer histogram path
/// (§06/§11 M5-QHIST).
///
/// Scale factors map the maximum absolute value to the i32 range. FLAG (spec
/// reconciliation): M5-QHIST asks for stochastic rounding AND a `< 0.5 / scale` error
/// bound; adjacent stochastic rounding can miss by nearly one step, so this path uses
/// nearest rounding with deterministic randomized tie-breaking keyed by the frozen
/// `Stage::Quantize` stream. The result is deterministic by row position, independent
/// of thread count, and bounded by half a quantization step.
///
/// # Errors
/// [`PbError::ShapeMismatch`] if `g`/`h` lengths differ; [`PbError::InvalidInput`] if
/// any value is non-finite or if a row index exceeds the frozen re-seed coordinate.
pub fn quantize_grad_hess(gh: &GradHess, seed: u64, round: u32) -> Result<QuantGradHess, PbError> {
    if gh.g.len() != gh.h.len() {
        return Err(PbError::ShapeMismatch {
            what: format!("GradHess g len {} != h len {}", gh.g.len(), gh.h.len()),
        });
    }
    let mut max_g = 0.0_f64;
    let mut max_h = 0.0_f64;
    for (i, (&g, &h)) in gh.g.iter().zip(&gh.h).enumerate() {
        if !g.is_finite() || !h.is_finite() {
            return Err(PbError::InvalidInput {
                what: format!("GradHess row {i} must be finite before quantization"),
            });
        }
        max_g = max_g.max(f64::from(g).abs());
        max_h = max_h.max(f64::from(h).abs());
    }
    let g_scale = scale_for(max_g)?;
    let h_scale = scale_for(max_h)?;
    let mut g_q = Vec::with_capacity(gh.g.len());
    let mut h_q = Vec::with_capacity(gh.h.len());
    for (i, (&g, &h)) in gh.g.iter().zip(&gh.h).enumerate() {
        let base = u32::try_from(i).map_err(|_| PbError::InvalidInput {
            what: "quantization supports at most u32::MAX rows".into(),
        })?;
        let block_g = base.checked_mul(2).ok_or_else(|| PbError::InvalidInput {
            what: "quantization row coordinate overflowed".into(),
        })?;
        let block_h = block_g
            .checked_add(1)
            .ok_or_else(|| PbError::InvalidInput {
                what: "quantization row coordinate overflowed".into(),
            })?;
        g_q.push(stochastic_round(
            f64::from(g) * f64::from(g_scale),
            seed,
            round,
            block_g,
        )?);
        h_q.push(stochastic_round(
            f64::from(h) * f64::from(h_scale),
            seed,
            round,
            block_h,
        )?);
    }
    Ok(QuantGradHess {
        g_q,
        h_q,
        scale: GradScale { g_scale, h_scale },
    })
}

fn scale_for(max_abs: f64) -> Result<f32, PbError> {
    // Map |max| to half the i32 range (|q| <= i32::MAX/2 ≈ 1.07e9). Fine-grained quantization keeps
    // QHIST accuracy-neutral (Δacc <= 0.01% on the suite; the coarser i16 range was measured to cost
    // miami −0.23%). The per-row q sums accumulate into i64 AoS cells (see `QHotCell`), which never
    // overflow for realistic n.
    let scale = if max_abs > 0.0 {
        (f64::from(i32::MAX) * 0.5) / max_abs
    } else {
        1.0
    };
    // A round whose max |g|/|h| is subnormal-tiny but nonzero (a near-converged fit) can push
    // this f64 scale past f32::MAX, so the `as f32` cast below would overflow to +inf and the
    // `Err` branch would abort the whole fit — for a graceful-stop situation, not an invalid
    // input (the max_abs == 0 branch above already treats the analogous all-zero case
    // gracefully). Clamp to f32::MAX instead: every |g|/|h| in this round is <= max_abs by
    // construction, so the clamped (finite) scale still maps them to valid, bounded i32
    // quanta — just not the full i32::MAX/2 dynamic range for this pathological edge case.
    // A no-op for every normal-range
    // max_abs (byte-identical to before).
    let out = scale.min(f64::from(f32::MAX)) as f32;
    if out.is_finite() && out > 0.0 {
        Ok(out)
    } else {
        Err(PbError::InvalidInput {
            what: format!("quantization scale is not finite: {scale}"),
        })
    }
}

fn stochastic_round(value: f64, seed: u64, round: u32, block: u32) -> Result<i32, PbError> {
    let lo = value.floor();
    let frac = value - lo;
    // The frozen `Quantize` stream is consulted ONLY at an exact tie (`frac == 0.5`), so compute
    // its hash lazily — `pb_seed` is a pure function of the coordinates, so deferring it to the
    // (rare) tie branch is bit-identical to computing it every row, but skips the per-row hash on
    // the overwhelming majority of rows (the hot quantization pass).
    let rounded = if frac > 0.5 {
        lo + 1.0
    } else if frac < 0.5 {
        lo
    } else {
        let bits = pb_seed(seed, round, Stage::Quantize as u32, block);
        let unit = ((bits >> 11) as f64 + 1.0) / ((1_u64 << 53) as f64 + 1.0);
        if unit < 0.5 {
            lo + 1.0
        } else {
            lo
        }
    };
    if rounded < f64::from(i32::MIN) || rounded > f64::from(i32::MAX) {
        return Err(PbError::InvalidInput {
            what: format!("quantized value {rounded} escaped i32"),
        });
    }
    Ok(rounded as i32)
}

/// Build a histogram via quantized integer accumulation, then dequantize into the
/// canonical [`Hist`] view for the existing split scanner.
///
/// # Errors
/// Propagates quantization and shape/index errors.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_quantized_histogram(
    x: &BinnedMatrix,
    qgh: &QuantGradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    n_leaves: usize,
    axes: &[u32],
    weight: &[f32],
    unit_weight: bool,
) -> Result<Hist, PbError> {
    let mut max_bins = 0usize;
    for &a in axes {
        let grid = x
            .grids
            .get(a as usize)
            .ok_or_else(oob("axis has no grid"))?;
        max_bins = max_bins.max(usize::from(grid.n_bins));
    }
    let n_axes = axes.len();
    let subs: Result<Vec<AxisQHist>, PbError> = axes
        .par_iter()
        .map(|&a| {
            accumulate_axis_quantized(
                x,
                qgh,
                rows,
                leaf_of_row,
                n_leaves,
                a,
                max_bins,
                weight,
                unit_weight,
            )
        })
        .collect();
    let subs = subs?;
    let mut hist = Hist::try_zeros(n_leaves, n_axes, max_bins)?;
    let inv_g = 1.0_f64 / f64::from(qgh.scale.g_scale);
    let inv_h = 1.0_f64 / f64::from(qgh.scale.h_scale);
    for (axis_pos, sub) in subs.iter().enumerate() {
        for leaf in 0..n_leaves {
            for bin in 0..max_bins {
                let src = offset2(leaf, bin, max_bins, "quant axis histogram offset overflow")?;
                let dst = offset3(
                    leaf,
                    axis_pos,
                    bin,
                    n_axes,
                    max_bins,
                    "quant histogram assembly offset overflow",
                )?;
                *hist.g.get_mut(dst).ok_or_else(oob("assemble quant g"))? =
                    *sub.g.get(src).ok_or_else(oob("sub quant g"))? as f64 * inv_g;
                *hist.h.get_mut(dst).ok_or_else(oob("assemble quant h"))? =
                    *sub.h.get(src).ok_or_else(oob("sub quant h"))? as f64 * inv_h;
                *hist
                    .wsum
                    .get_mut(dst)
                    .ok_or_else(oob("assemble quant wsum"))? =
                    *sub.wsum.get(src).ok_or_else(oob("sub quant wsum"))?;
                *hist
                    .count
                    .get_mut(dst)
                    .ok_or_else(oob("assemble quant count"))? =
                    *sub.count.get(src).ok_or_else(oob("sub quant count"))?;
            }
        }
    }
    Ok(hist)
}

/// One `[leaf][bin]` cell's gradient/hessian/count, packed so the per-row scatter touches a
/// single cache line and needs one bounds check instead of three (the hot accumulation path).
#[derive(Clone, Copy, Default)]
struct GhcCell {
    g: f64,
    h: f64,
    count: u32,
}

/// One axis's `[leaf][bin]` sub-histogram (stride `max_bins`). `g`/`h`/`count` are packed into
/// one [`GhcCell`] per cell (array-of-structs) so each accumulated row is a single bounds-checked
/// write to one cache line; `wsum` stays a separate array, touched only for non-unit weights.
struct AxisHist {
    ghc: Vec<GhcCell>,
    wsum: Vec<f64>,
}

struct AxisQHist {
    g: Vec<i64>,
    h: Vec<i64>,
    /// Sample-weight sums stay full-precision `f64` — weights are not gradients, so they
    /// are never quantized (the credibility floor reads exact Σw on both hist paths).
    wsum: Vec<f64>,
    count: Vec<u32>,
}

impl AxisQHist {
    fn try_zeros(n_leaves: usize, max_bins: usize) -> Result<Self, PbError> {
        let size = Hist::checked_cell_count(n_leaves, 1, max_bins)?;
        Ok(Self {
            g: Hist::try_zeroed_vec(size, "axis quant histogram g")?,
            h: Hist::try_zeroed_vec(size, "axis quant histogram h")?,
            wsum: Hist::try_zeroed_vec(size, "axis quant histogram wsum")?,
            count: Hist::try_zeroed_vec(size, "axis quant histogram count")?,
        })
    }
}

/// AoS hot cell for the quantized per-row scatter: the quantized `g`/`h` sums in `i64`, packed with
/// `count` into ONE cell (mirrors the FullF64 `GhcCell`), so the per-row scatter is a single
/// bounds-checked write to one cache line — vs the previous SoA `AxisQHist` that touched 4 separate
/// arrays (4 cache lines) per row, the root cause of QHIST's slowness. `i64` holds any realistic
/// `Σ q` (n·i32::MAX/2 ≪ i64::MAX), so the plain `+=` never overflows.
#[derive(Clone, Copy, Default)]
struct QHotCell {
    g: i64,
    h: i64,
    count: u32,
}

/// One axis's per-chunk quantized sub-histogram (stride `max_bins`): `g`/`h`/`count` packed into one
/// [`QHotCell`] per cell; `wsum` stays a separate f64 array, touched only for non-unit weights
/// (mirrors [`AxisHist`]).
struct AxisQHistHot {
    ghc: Vec<QHotCell>,
    wsum: Vec<f64>,
}

impl AxisQHistHot {
    fn try_zeros(n_leaves: usize, max_bins: usize) -> Result<Self, PbError> {
        let size = Hist::checked_cell_count(n_leaves, 1, max_bins)?;
        Ok(Self {
            ghc: Hist::try_zeroed_vec(size, "axis quant hot ghc")?,
            wsum: Hist::try_zeroed_vec(size, "axis quant hot wsum")?,
        })
    }
}

impl AxisHist {
    fn try_zeros(n_leaves: usize, max_bins: usize) -> Result<Self, PbError> {
        let size = Hist::checked_cell_count(n_leaves, 1, max_bins)?;
        Ok(Self {
            ghc: Hist::try_zeroed_vec(size, "axis histogram ghc")?,
            wsum: Hist::try_zeroed_vec(size, "axis histogram wsum")?,
        })
    }
}

/// Axis-block width for [`build_histogram`]'s blocked builder. The per-axis builder re-streamed
/// the shared per-row arrays (`gh.g`, `gh.h`, `leaf_of_row`, `weight`) once PER AXIS — A-fold
/// redundant traffic that dominated wide-data `hist_build` (only the 1-byte bin column is
/// inherently per-axis). One row scan feeding a block of per-axis tables reads them once per
/// BLOCK; 8 tables of ≤256 bins stay L1/L2-resident.
///
/// ANY block size is bit-identical (see `accumulate_axes_blocked`), so the width is purely a
/// bandwidth/utilization trade, NOT a determinism knob.
const AXIS_BLOCK: usize = 8;

/// The cells of one `(axis, leaf)` over the whole `u8` bin range, so a bin id indexes them with no
/// bounds check. Only `0..axis_bins` can be reached from valid input; the tail stays zero.
type BinCells = [GhcCell; 256];
/// Per-bin `Σw` of one `(axis, leaf)`, laid out like [`BinCells`].
type BinWsum = [f64; 256];

/// Blocked multi-axis accumulation: one pass over `rows` scatters into each of the block's
/// per-axis histograms.
///
/// Bit-identical to running `accumulate_axis_sequential` per axis, for EVERY block size and
/// partitioning: cells are per-axis disjoint, and nesting the axis loop INSIDE the row loop
/// leaves each `(axis, leaf, bin)` cell's f64 addition sequence exactly the caller's row order —
/// regrouping axes regroups whole cells, never a cell's summation order. The hoisted per-row
/// `f64::from` widenings are position-invariant, so the added values are identical too.
///
/// The scatter writes into full-width [`BinCells`] rows (no per-update bounds or overflow checks;
/// the axis loop unrolled for full [`AXIS_BLOCK`] blocks, and the leaf map not read at all when
/// there is one leaf), then compacts each axis to its own `axis_bins` stride. A bin id past
/// `axis_bins` — which the engine never produces — surfaces as the same error the per-row check
/// raised, from the non-empty tail.
#[allow(clippy::too_many_arguments)]
fn accumulate_axes_blocked(
    x: &BinnedMatrix,
    gh: &GradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    n_leaves: usize,
    axes: &[u32],
    weight: &[f32],
    unit_weight: bool,
) -> Result<Vec<AxisHist>, PbError> {
    let mut cols: Vec<&[u8]> = Vec::with_capacity(axes.len());
    let mut bins: Vec<usize> = Vec::with_capacity(axes.len());
    for &a in axes {
        cols.push(
            x.data
                .get(a as usize)
                .ok_or_else(oob("axis has no column"))?,
        );
        bins.push(usize::from(
            x.grids
                .get(a as usize)
                .ok_or_else(oob("axis has no grid"))?
                .n_bins,
        ));
    }
    let mut tabs: Vec<Vec<BinCells>> = axes
        .iter()
        .map(|_| try_bin_rows(n_leaves, [GhcCell::default(); 256]))
        .collect::<Result<_, _>>()?;
    let mut wtabs: Vec<Vec<BinWsum>> = if unit_weight {
        Vec::new()
    } else {
        axes.iter()
            .map(|_| try_bin_rows(n_leaves, [0.0; 256]))
            .collect::<Result<_, _>>()?
    };
    let rows_in = RowsIn {
        gh,
        rows,
        leaf_of_row,
        n_leaves,
        weight,
    };
    match (unit_weight, n_leaves == 1) {
        (true, true) => scatter_block::<true, true>(&rows_in, &cols, &mut tabs, &mut wtabs)?,
        (true, false) => scatter_block::<true, false>(&rows_in, &cols, &mut tabs, &mut wtabs)?,
        (false, true) => scatter_block::<false, true>(&rows_in, &cols, &mut tabs, &mut wtabs)?,
        (false, false) => scatter_block::<false, false>(&rows_in, &cols, &mut tabs, &mut wtabs)?,
    }
    let mut outs: Vec<AxisHist> = Vec::with_capacity(axes.len());
    for (k, (tab, &axis_bins)) in tabs.iter().zip(&bins).enumerate() {
        let mut out = AxisHist::try_zeros(n_leaves, axis_bins)?;
        for (leaf, cells) in tab.iter().enumerate() {
            if cells.iter().skip(axis_bins).any(|c| c.count != 0) {
                return Err(PbError::Internal {
                    what: "bin id out of histogram range".into(),
                });
            }
            let lo = leaf * axis_bins;
            out.ghc
                .get_mut(lo..lo + axis_bins)
                .ok_or_else(oob("compact ghc"))?
                .copy_from_slice(cells.get(..axis_bins).ok_or_else(oob("bin cells"))?);
            if !unit_weight {
                let w = wtabs
                    .get(k)
                    .and_then(|t| t.get(leaf))
                    .ok_or_else(oob("bin wsum"))?;
                out.wsum
                    .get_mut(lo..lo + axis_bins)
                    .ok_or_else(oob("compact wsum"))?
                    .copy_from_slice(w.get(..axis_bins).ok_or_else(oob("bin wsum"))?);
            }
        }
        if unit_weight {
            // wsum == count for unit weights (Σ 1.0 == count, exact in f64) — per axis, exactly as
            // the per-axis builder fills it.
            for (w, c) in out.wsum.iter_mut().zip(&out.ghc) {
                *w = f64::from(c.count);
            }
        }
        outs.push(out);
    }
    Ok(outs)
}

/// `n_leaves` zeroed full-width bin rows, or the overflow/allocation error `Hist` sizing gives.
fn try_bin_rows<T: Copy>(n_leaves: usize, zero: T) -> Result<Vec<T>, PbError> {
    Hist::checked_cell_count(n_leaves, 1, 256)?;
    let mut v: Vec<T> = Vec::new();
    v.try_reserve_exact(n_leaves)
        .map_err(|_| PbError::Internal {
            what: "histogram bin rows allocation failed".into(),
        })?;
    v.resize(n_leaves, zero);
    Ok(v)
}

/// The per-row inputs of one scatter pass.
struct RowsIn<'a> {
    gh: &'a GradHess,
    rows: &'a [u32],
    leaf_of_row: &'a [u8],
    n_leaves: usize,
    weight: &'a [f32],
}

/// Scatter `rows` into a block's [`BinCells`] tables: full [`AXIS_BLOCK`]-wide blocks take the
/// unrolled [`scatter_rows`], a narrower tail block the same loop over a slice.
fn scatter_block<const UNIT: bool, const ONE: bool>(
    rows_in: &RowsIn<'_>,
    cols: &[&[u8]],
    tabs: &mut [Vec<BinCells>],
    wtabs: &mut [Vec<BinWsum>],
) -> Result<(), PbError> {
    if let (Ok(c), Ok(t)) = (
        <&[&[u8]; AXIS_BLOCK]>::try_from(cols),
        <&mut [Vec<BinCells>; AXIS_BLOCK]>::try_from(&mut *tabs),
    ) {
        return scatter_rows::<AXIS_BLOCK, UNIT, ONE>(rows_in, c, t, wtabs);
    }
    scatter_rows_dyn::<UNIT, ONE>(rows_in, cols, tabs, wtabs)
}

/// The leaf, `g`, `h` and weight of row `ru`. With one leaf the leaf id is not branched on per row:
/// it is OR-ed into `stray`, which the caller checks once after the pass.
#[inline(always)]
fn row_values<const UNIT: bool, const ONE: bool>(
    rows_in: &RowsIn<'_>,
    ru: usize,
    stray: &mut u8,
) -> Result<(usize, f64, f64, f64), PbError> {
    let leaf = if ONE {
        *stray |= *rows_in
            .leaf_of_row
            .get(ru)
            .ok_or_else(oob("row out of leaf map"))?;
        0
    } else {
        let leaf = usize::from(
            *rows_in
                .leaf_of_row
                .get(ru)
                .ok_or_else(oob("row out of leaf map"))?,
        );
        if leaf >= rows_in.n_leaves {
            return Err(PbError::Internal {
                what: "leaf id out of histogram range".into(),
            });
        }
        leaf
    };
    let g = f64::from(*rows_in.gh.g.get(ru).ok_or_else(oob("gh.g"))?);
    let h = f64::from(*rows_in.gh.h.get(ru).ok_or_else(oob("gh.h"))?);
    let w = if UNIT {
        0.0
    } else {
        f64::from(*rows_in.weight.get(ru).ok_or_else(oob("weight"))?)
    };
    Ok((leaf, g, h, w))
}

/// Add one row to one axis's cell: the `(leaf, bin)` lookups cannot miss (`leaf < n_leaves`, and
/// a `u8` bin indexes a 256-cell row).
#[inline(always)]
fn add_row<const UNIT: bool>(
    tab: &mut [BinCells],
    wtab: Option<&mut Vec<BinWsum>>,
    leaf: usize,
    bin: u8,
    g: f64,
    h: f64,
    w: f64,
) {
    if let Some(cell) = tab
        .get_mut(leaf)
        .and_then(|cells| cells.get_mut(usize::from(bin)))
    {
        cell.g += g;
        cell.h += h;
        // count <= rows.len() <= u32::MAX: never wraps
        cell.count = cell.count.wrapping_add(1);
    }
    if !UNIT {
        if let Some(ws) = wtab
            .and_then(|t| t.get_mut(leaf))
            .and_then(|cells| cells.get_mut(usize::from(bin)))
        {
            *ws += w;
        }
    }
}

fn scatter_rows<const N: usize, const UNIT: bool, const ONE: bool>(
    rows_in: &RowsIn<'_>,
    cols: &[&[u8]; N],
    tabs: &mut [Vec<BinCells>; N],
    wtabs: &mut [Vec<BinWsum>],
) -> Result<(), PbError> {
    let mut stray = 0u8;
    for &r in rows_in.rows {
        let ru = r as usize;
        let (leaf, g, h, w) = row_values::<UNIT, ONE>(rows_in, ru, &mut stray)?;
        let mut wt = wtabs.iter_mut();
        for (col, tab) in cols.iter().zip(tabs.iter_mut()) {
            let bin = *col.get(ru).ok_or_else(oob("row out of column"))?;
            add_row::<UNIT>(tab, if UNIT { None } else { wt.next() }, leaf, bin, g, h, w);
        }
    }
    if stray != 0 {
        return Err(PbError::Internal {
            what: "leaf id out of histogram range".into(),
        });
    }
    Ok(())
}

fn scatter_rows_dyn<const UNIT: bool, const ONE: bool>(
    rows_in: &RowsIn<'_>,
    cols: &[&[u8]],
    tabs: &mut [Vec<BinCells>],
    wtabs: &mut [Vec<BinWsum>],
) -> Result<(), PbError> {
    let mut stray = 0u8;
    for &r in rows_in.rows {
        let ru = r as usize;
        let (leaf, g, h, w) = row_values::<UNIT, ONE>(rows_in, ru, &mut stray)?;
        let mut wt = wtabs.iter_mut();
        for (col, tab) in cols.iter().zip(tabs.iter_mut()) {
            let bin = *col.get(ru).ok_or_else(oob("row out of column"))?;
            add_row::<UNIT>(tab, if UNIT { None } else { wt.next() }, leaf, bin, g, h, w);
        }
    }
    if stray != 0 {
        return Err(PbError::Internal {
            what: "leaf id out of histogram range".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
/// The row-scatter [`accumulate_axes_blocked`] replaced (2026-09-28), kept as its bit-identity
/// ORACLE: per-axis tables at the axis's own stride, every access checked.
///
/// Bit-identical to running `accumulate_axis_sequential` per axis, for EVERY block size and
/// partitioning: cells are per-axis disjoint, and nesting the axis loop INSIDE the row loop
/// leaves each `(axis, leaf, bin)` cell's f64 addition sequence exactly the caller's row order —
/// regrouping axes regroups whole cells, never a cell's summation order. The hoisted per-row
/// `f64::from` widenings are position-invariant, so the added values are identical too.
#[allow(clippy::too_many_arguments)]
fn accumulate_axes_blocked_reference(
    x: &BinnedMatrix,
    gh: &GradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    n_leaves: usize,
    axes: &[u32],
    weight: &[f32],
    unit_weight: bool,
) -> Result<Vec<AxisHist>, PbError> {
    let mut cols: Vec<&[u8]> = Vec::with_capacity(axes.len());
    let mut bins: Vec<usize> = Vec::with_capacity(axes.len());
    let mut outs: Vec<AxisHist> = Vec::with_capacity(axes.len());
    for &a in axes {
        cols.push(
            x.data
                .get(a as usize)
                .ok_or_else(oob("axis has no column"))?,
        );
        let axis_bins = usize::from(
            x.grids
                .get(a as usize)
                .ok_or_else(oob("axis has no grid"))?
                .n_bins,
        );
        bins.push(axis_bins);
        outs.push(AxisHist::try_zeros(n_leaves, axis_bins)?);
    }
    for &r in rows {
        let ru = r as usize;
        let leaf = usize::from(*leaf_of_row.get(ru).ok_or_else(oob("row out of leaf map"))?);
        if leaf >= n_leaves {
            return Err(PbError::Internal {
                what: "leaf id out of histogram range".into(),
            });
        }
        // Shared per-row loads, hoisted out of the axis loop — the whole point of blocking.
        let g = f64::from(*gh.g.get(ru).ok_or_else(oob("gh.g"))?);
        let h = f64::from(*gh.h.get(ru).ok_or_else(oob("gh.h"))?);
        let w = if unit_weight {
            0.0
        } else {
            f64::from(*weight.get(ru).ok_or_else(oob("weight"))?)
        };
        for ((col, &axis_bins), out) in cols.iter().zip(&bins).zip(&mut outs) {
            let bin = usize::from(*col.get(ru).ok_or_else(oob("row out of column"))?);
            if bin >= axis_bins {
                return Err(PbError::Internal {
                    what: "bin id out of histogram range".into(),
                });
            }
            let idx = offset2(leaf, bin, axis_bins, "axis histogram row offset overflow")?;
            let cell = out.ghc.get_mut(idx).ok_or_else(oob("ghc cell"))?;
            cell.g += g;
            cell.h += h;
            cell.count = cell
                .count
                .checked_add(1)
                .ok_or_else(oob("bin count overflow"))?;
            if !unit_weight {
                *out.wsum.get_mut(idx).ok_or_else(oob("wsum cell"))? += w;
            }
        }
    }
    if unit_weight {
        // wsum == count for unit weights (Σ 1.0 == count, exact in f64) — per axis, exactly as
        // the per-axis builder fills it.
        for out in &mut outs {
            for (w, c) in out.wsum.iter_mut().zip(&out.ghc) {
                *w = f64::from(c.count);
            }
        }
    }
    Ok(outs)
}

/// Row-chunk parallel wrapper for [`accumulate_axes_blocked`], mirroring [`accumulate_axis`]'s
/// chunking exactly: same `ROW_PAR_MIN_ROWS` gate (and the same saturated-axis prototype skip),
/// same fixed `ROW_PAR_CHUNK_ROWS` boundaries, per-axis partials reduced in CHUNK ORDER — so each
/// axis's chunked fold is bit-identical to the per-axis builder's.
#[allow(clippy::too_many_arguments)]
fn accumulate_axes_block_chunked(
    x: &BinnedMatrix,
    gh: &GradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    n_leaves: usize,
    axes: &[u32],
    weight: &[f32],
    unit_weight: bool,
) -> Result<Vec<AxisHist>, PbError> {
    let skip_chunking = rows.len() < ROW_PAR_MIN_ROWS;
    if skip_chunking {
        return accumulate_axes_blocked(
            x,
            gh,
            rows,
            leaf_of_row,
            n_leaves,
            axes,
            weight,
            unit_weight,
        );
    }
    let chunks: Result<Vec<Vec<AxisHist>>, PbError> = rows
        .par_chunks(ROW_PAR_CHUNK_ROWS)
        .with_min_len(crate::sched::min_chunks_per_task())
        .map(|chunk| {
            accumulate_axes_blocked(
                x,
                gh,
                chunk,
                leaf_of_row,
                n_leaves,
                axes,
                weight,
                unit_weight,
            )
        })
        .collect();
    let chunks = chunks?;
    let mut outs: Vec<AxisHist> = Vec::with_capacity(axes.len());
    for (k, &a) in axes.iter().enumerate() {
        let axis_bins = usize::from(
            x.grids
                .get(a as usize)
                .ok_or_else(oob("axis has no grid"))?
                .n_bins,
        );
        let mut out = AxisHist::try_zeros(n_leaves, axis_bins)?;
        for chunk in &chunks {
            add_axis_hist(
                &mut out,
                chunk.get(k).ok_or_else(oob("chunk axis escaped"))?,
            )?;
        }
        outs.push(out);
    }
    Ok(outs)
}

/// Per-axis reference builder — the ORACLE `accumulate_axes_block_chunked` is proven
/// bit-identical against (`blocked_axis_build_is_bit_identical_to_per_axis`). The engine
/// only ever calls the blocked builder, so this is test-only code.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn accumulate_axis(
    x: &BinnedMatrix,
    gh: &GradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    n_leaves: usize,
    axis: u32,
    weight: &[f32],
    unit_weight: bool,
) -> Result<AxisHist, PbError> {
    let skip_chunking = rows.len() < ROW_PAR_MIN_ROWS;
    if skip_chunking {
        return accumulate_axis_sequential(
            x,
            gh,
            rows,
            leaf_of_row,
            n_leaves,
            axis,
            weight,
            unit_weight,
        );
    }
    let chunks: Result<Vec<AxisHist>, PbError> = rows
        .par_chunks(ROW_PAR_CHUNK_ROWS)
        .with_min_len(crate::sched::min_chunks_per_task())
        .map(|chunk| {
            accumulate_axis_sequential(
                x,
                gh,
                chunk,
                leaf_of_row,
                n_leaves,
                axis,
                weight,
                unit_weight,
            )
        })
        .collect();
    let chunks = chunks?;
    // Reduce at the same per-axis (jagged) stride the chunks were built at.
    let axis_bins = usize::from(
        x.grids
            .get(axis as usize)
            .ok_or_else(oob("axis has no grid"))?
            .n_bins,
    );
    let mut out = AxisHist::try_zeros(n_leaves, axis_bins)?;
    for chunk in &chunks {
        add_axis_hist(&mut out, chunk)?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn accumulate_axis_sequential(
    x: &BinnedMatrix,
    gh: &GradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    n_leaves: usize,
    axis: u32,
    weight: &[f32],
    unit_weight: bool,
) -> Result<AxisHist, PbError> {
    let col = x
        .data
        .get(axis as usize)
        .ok_or_else(oob("axis has no column"))?;
    // Per-axis (jagged) bin stride: size this axis's intermediate to ITS OWN grid `n_bins`, not the
    // global `max_bins`. The per-row scatter then writes into a smaller, more L1-resident array for
    // low-cardinality axes, and the zero/reduce passes shrink to O(n_leaves·axis_bins). Byte-
    // identical: no row has `bin >= axis_bins`, so the high cells the uniform-stride build left zero
    // are exactly the ones omitted here, and `build_histogram` re-expands to the uniform max_bins
    // stride. Deterministic: stride is a pure function of the (frozen) grid, not the thread count.
    let axis_bins = usize::from(
        x.grids
            .get(axis as usize)
            .ok_or_else(oob("axis has no grid"))?
            .n_bins,
    );
    let mut out = AxisHist::try_zeros(n_leaves, axis_bins)?;
    // Sequential over rows in their given (fixed) order ⇒ deterministic f64 fold. When
    // `unit_weight`, the per-row weight read + `Σw` add are skipped (the loop-invariant branch
    // is hoisted by LLVM into two loop variants), and `wsum` is set from `count` afterwards —
    // bit-exact, since for unit weights `Σ 1.0` over a bin equals its (integer, <2^53) count.
    for &r in rows {
        let ru = r as usize;
        let bin = usize::from(*col.get(ru).ok_or_else(oob("row out of column"))?);
        let leaf = usize::from(*leaf_of_row.get(ru).ok_or_else(oob("row out of leaf map"))?);
        if leaf >= n_leaves || bin >= axis_bins {
            return Err(PbError::Internal {
                what: "leaf or bin id out of histogram range".into(),
            });
        }
        let idx = offset2(leaf, bin, axis_bins, "axis histogram row offset overflow")?;
        // One bounds-checked write to the packed g/h/count cell (one cache line).
        let cell = out.ghc.get_mut(idx).ok_or_else(oob("ghc cell"))?;
        cell.g += f64::from(*gh.g.get(ru).ok_or_else(oob("gh.g"))?);
        cell.h += f64::from(*gh.h.get(ru).ok_or_else(oob("gh.h"))?);
        // count <= rows.len() <= n_rows <= u32::MAX, so this never actually overflows;
        // checked_add keeps it panic-free under overflow-checks regardless.
        cell.count = cell
            .count
            .checked_add(1)
            .ok_or_else(oob("bin count overflow"))?;
        if !unit_weight {
            *out.wsum.get_mut(idx).ok_or_else(oob("wsum cell"))? +=
                f64::from(*weight.get(ru).ok_or_else(oob("weight"))?);
        }
    }
    if unit_weight {
        // wsum == count for unit weights (Σ 1.0 == count, exact in f64). Per-chunk this gives
        // each chunk's count; `add_axis_hist` then sums them, matching the per-row Σw exactly.
        for (w, c) in out.wsum.iter_mut().zip(&out.ghc) {
            *w = f64::from(c.count);
        }
    }
    Ok(out)
}

fn add_axis_hist(dst: &mut AxisHist, src: &AxisHist) -> Result<(), PbError> {
    if dst.ghc.len() != src.ghc.len() || dst.wsum.len() != src.wsum.len() {
        return Err(PbError::Internal {
            what: "axis histogram chunk shape mismatch".into(),
        });
    }
    for (d, s) in dst.ghc.iter_mut().zip(&src.ghc) {
        d.g += s.g;
        d.h += s.h;
        d.count = d
            .count
            .checked_add(s.count)
            .ok_or_else(oob("axis histogram chunk count overflow"))?;
    }
    for (d, s) in dst.wsum.iter_mut().zip(&src.wsum) {
        *d += *s;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn accumulate_axis_quantized(
    x: &BinnedMatrix,
    qgh: &QuantGradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    n_leaves: usize,
    axis: u32,
    max_bins: usize,
    weight: &[f32],
    unit_weight: bool,
) -> Result<AxisQHist, PbError> {
    // The dense AoS hot path accumulates into i64 cells (see `QHotCell`) then reduces per-chunk into
    // this SoA `AxisQHist` via `add_qhot_into_qhist` — the fix for the old 4-cache-line SoA scatter.
    let mut out = AxisQHist::try_zeros(n_leaves, max_bins)?;
    if rows.len() < ROW_PAR_MIN_ROWS {
        let hot = accumulate_axis_quantized_sequential(
            x,
            qgh,
            rows,
            leaf_of_row,
            n_leaves,
            axis,
            max_bins,
            weight,
            unit_weight,
        )?;
        add_qhot_into_qhist(&mut out, &hot)?;
        return Ok(out);
    }
    let chunks: Result<Vec<AxisQHistHot>, PbError> = rows
        .par_chunks(ROW_PAR_CHUNK_ROWS)
        .with_min_len(crate::sched::min_chunks_per_task())
        .map(|chunk| {
            accumulate_axis_quantized_sequential(
                x,
                qgh,
                chunk,
                leaf_of_row,
                n_leaves,
                axis,
                max_bins,
                weight,
                unit_weight,
            )
        })
        .collect();
    let chunks = chunks?;
    for chunk in &chunks {
        add_qhot_into_qhist(&mut out, chunk)?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn accumulate_axis_quantized_sequential(
    x: &BinnedMatrix,
    qgh: &QuantGradHess,
    rows: &[u32],
    leaf_of_row: &[u8],
    n_leaves: usize,
    axis: u32,
    max_bins: usize,
    weight: &[f32],
    unit_weight: bool,
) -> Result<AxisQHistHot, PbError> {
    let col = x
        .data
        .get(axis as usize)
        .ok_or_else(oob("axis has no column"))?;
    let mut out = AxisQHistHot::try_zeros(n_leaves, max_bins)?;
    for &r in rows {
        let ru = r as usize;
        let bin = usize::from(*col.get(ru).ok_or_else(oob("quant row out of column"))?);
        let leaf = usize::from(
            *leaf_of_row
                .get(ru)
                .ok_or_else(oob("quant row out of leaf map"))?,
        );
        if leaf >= n_leaves || bin >= max_bins {
            return Err(PbError::Internal {
                what: "leaf or bin id out of quant histogram range".into(),
            });
        }
        let idx = offset2(
            leaf,
            bin,
            max_bins,
            "quant axis histogram row offset overflow",
        )?;
        // One bounds-checked write to the packed g/h/count cell (one cache line); the i64 sums hold
        // any realistic Σ q without overflow.
        let cell = out.ghc.get_mut(idx).ok_or_else(oob("quant ghc cell"))?;
        cell.g += i64::from(*qgh.g_q.get(ru).ok_or_else(oob("qgh.g"))?);
        cell.h += i64::from(*qgh.h_q.get(ru).ok_or_else(oob("qgh.h"))?);
        cell.count = cell
            .count
            .checked_add(1)
            .ok_or_else(oob("quant bin count overflow"))?;
        if !unit_weight {
            *out.wsum.get_mut(idx).ok_or_else(oob("quant wsum cell"))? +=
                f64::from(*weight.get(ru).ok_or_else(oob("quant weight"))?);
        }
    }
    if unit_weight {
        // wsum == count for unit weights (Σ 1.0 == count, exact in f64); the per-chunk reduction then
        // sums per-chunk counts, matching the per-row Σw exactly (cf. the FullF64 path).
        for (w, c) in out.wsum.iter_mut().zip(&out.ghc) {
            *w = f64::from(c.count);
        }
    }
    Ok(out)
}

/// Reduce a per-chunk AoS hot histogram into the SoA `AxisQHist` accumulator (chunk reduction in
/// fixed order ⇒ thread-count independent; integer addition is exact + associative).
fn add_qhot_into_qhist(dst: &mut AxisQHist, src: &AxisQHistHot) -> Result<(), PbError> {
    if dst.g.len() != src.ghc.len() || dst.wsum.len() != src.wsum.len() {
        return Err(PbError::Internal {
            what: "axis quant histogram chunk shape mismatch".into(),
        });
    }
    for (i, cell) in src.ghc.iter().enumerate() {
        *dst.g.get_mut(i).ok_or_else(oob("qhist g cell"))? += cell.g;
        *dst.h.get_mut(i).ok_or_else(oob("qhist h cell"))? += cell.h;
        let c = dst.count.get_mut(i).ok_or_else(oob("qhist count cell"))?;
        *c = c
            .checked_add(cell.count)
            .ok_or_else(oob("axis quant histogram chunk count overflow"))?;
    }
    for (d, s) in dst.wsum.iter_mut().zip(&src.wsum) {
        *d += *s;
    }
    Ok(())
}

/// Fill each LARGER sibling-child leaf of a level-`L` histogram by subtracting the already-built
/// SMALLER child from its level-`(L-1)` parent leaf — the histogram-subtraction trick wired to the
/// oblivious grower. `child` has `2·parent.n_leaves` leaves (each parent leaf split into a smaller +
/// larger child); `pairing[i] = (parent_leaf, smaller_child_leaf, larger_child_leaf)`; `axis_map[a2]`
/// is the column position in `parent` of `child`'s axis `a2` (the parent's axis set is a superset, so
/// every child axis maps). On entry the smaller-child leaves are populated and the larger are zero;
/// on return `child[larger] = parent[p] − child[smaller]` cellwise (g/h/wsum plain f64, count
/// `checked_sub`). Because the SMALLER child is the subtrahend, `larger` is the bigger remainder, so
/// the f64 subtraction is NOT catastrophic cancellation (drift stays ~1e-11 for well-conditioned
/// gradients); `count` is integer-exact and, under unit weights, `wsum == count` stays exact. Used
/// only on the FullF64 path; bin range `0..child.n_bins ≤ parent.n_bins` (a dropped axis may have
/// held the max stride), and every access goes through [`Hist::offset`] so strides are honored.
pub(crate) fn subtract_sibling_into(
    child: &mut Hist,
    parent: &Hist,
    pairing: &[(usize, usize, usize)],
    axis_map: &[usize],
) -> Result<(), PbError> {
    if child.n_leaves != 2 * parent.n_leaves {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "subtract_sibling_into: child n_leaves {} != 2·parent {}",
                child.n_leaves, parent.n_leaves
            ),
        });
    }
    if child.n_axes != axis_map.len() {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "subtract_sibling_into: child n_axes {} != axis_map len {}",
                child.n_axes,
                axis_map.len()
            ),
        });
    }
    if child.n_bins > parent.n_bins {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "subtract_sibling_into: child n_bins {} > parent n_bins {}",
                child.n_bins, parent.n_bins
            ),
        });
    }
    for &(p, sm, lg) in pairing {
        for (a2, &a1) in axis_map.iter().enumerate() {
            for b in 0..child.n_bins {
                let po = parent
                    .offset(p, a1, b)
                    .ok_or_else(oob("subtract_sibling parent offset"))?;
                let so = child
                    .offset(sm, a2, b)
                    .ok_or_else(oob("subtract_sibling smaller offset"))?;
                let lo = child
                    .offset(lg, a2, b)
                    .ok_or_else(oob("subtract_sibling larger offset"))?;
                // Read parent (p) and smaller (sm) first, then write larger (lg) — lg differs
                // from sm so there is no aliasing; copies keep the borrows sequential.
                let pg = *parent
                    .g
                    .get(po)
                    .ok_or_else(oob("subtract_sibling parent g"))?;
                let ph = *parent
                    .h
                    .get(po)
                    .ok_or_else(oob("subtract_sibling parent h"))?;
                let pw = *parent
                    .wsum
                    .get(po)
                    .ok_or_else(oob("subtract_sibling parent wsum"))?;
                let pc = *parent
                    .count
                    .get(po)
                    .ok_or_else(oob("subtract_sibling parent count"))?;
                let sg = *child
                    .g
                    .get(so)
                    .ok_or_else(oob("subtract_sibling small g"))?;
                let sh = *child
                    .h
                    .get(so)
                    .ok_or_else(oob("subtract_sibling small h"))?;
                let sw = *child
                    .wsum
                    .get(so)
                    .ok_or_else(oob("subtract_sibling small wsum"))?;
                let sc = *child
                    .count
                    .get(so)
                    .ok_or_else(oob("subtract_sibling small count"))?;
                *child
                    .g
                    .get_mut(lo)
                    .ok_or_else(oob("subtract_sibling g cell"))? = pg - sg;
                *child
                    .h
                    .get_mut(lo)
                    .ok_or_else(oob("subtract_sibling h cell"))? = ph - sh;
                *child
                    .wsum
                    .get_mut(lo)
                    .ok_or_else(oob("subtract_sibling wsum cell"))? = pw - sw;
                *child
                    .count
                    .get_mut(lo)
                    .ok_or_else(oob("subtract_sibling count cell"))? = pc
                    .checked_sub(sc)
                    .ok_or_else(oob("subtract_sibling count underflow"))?;
            }
        }
    }
    Ok(())
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
    use crate::data::{AxisKind, AxisProvenance, BorderGrid, FeatureId};
    use crate::engine::Hist;

    /// Build a BinnedMatrix from pre-binned columns + each axis's `n_bins` (only the
    /// `n_bins` matters to the histogram; borders are unread here).
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

    /// The blocked multi-axis builder must be BIT-identical to the per-axis builder — per cell,
    /// per field, on both the sequential (< ROW_PAR_MIN_ROWS) and chunked paths, with unit and
    /// non-unit weights, magnitude-spread gradients, and a block-width-indivisible axis count.
    #[test]
    fn blocked_build_matches_per_axis_builder_bitwise() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state >> 33
        };
        let bins: Vec<u16> = vec![2, 3, 5, 17, 255, 7, 31, 64, 9, 4, 11, 128, 6];
        let n_axes = bins.len();
        for n in [1_000_usize, ROW_PAR_MIN_ROWS + 12_345] {
            let cols: Vec<Vec<u8>> = bins
                .iter()
                .map(|&nb| (0..n).map(|_| (next() % u64::from(nb)) as u8).collect())
                .collect();
            let x = matrix(cols, &bins);
            let g: Vec<f32> = (0..n)
                .map(|_| ((next() % 2001) as f32 - 1000.0) * 10f32.powi((next() % 7) as i32 - 3))
                .collect();
            let h: Vec<f32> = (0..n)
                .map(|_| ((next() % 1000) as f32 + 1.0) * 10f32.powi((next() % 5) as i32 - 2))
                .collect();
            let gh = gradhess(&g, &h);
            let w: Vec<f32> = (0..n)
                .map(|_| 0.25 + (next() % 1000) as f32 / 100.0)
                .collect();
            let leaf_of_row: Vec<u8> = (0..n).map(|_| (next() % 4) as u8).collect();
            let n_leaves = 4usize;
            // Full row set AND a strided subset (the builders take arbitrary row lists).
            let all_rows: Vec<u32> = (0..n as u32).collect();
            let subset: Vec<u32> = (0..n as u32).step_by(3).collect();
            let axes: Vec<u32> = (0..n_axes as u32).collect();
            for rows in [&all_rows, &subset] {
                for unit_weight in [true, false] {
                    let blocked = accumulate_axes_block_chunked(
                        &x,
                        &gh,
                        rows,
                        &leaf_of_row,
                        n_leaves,
                        &axes,
                        &w,
                        unit_weight,
                    )
                    .unwrap();
                    assert_eq!(blocked.len(), n_axes);
                    for (k, &a) in axes.iter().enumerate() {
                        let reference = accumulate_axis(
                            &x,
                            &gh,
                            rows,
                            &leaf_of_row,
                            n_leaves,
                            a,
                            &w,
                            unit_weight,
                        )
                        .unwrap();
                        let b = &blocked[k];
                        assert_eq!(b.ghc.len(), reference.ghc.len(), "axis {a} cell count");
                        for (i, (bc, rc)) in b.ghc.iter().zip(&reference.ghc).enumerate() {
                            assert_eq!(bc.g.to_bits(), rc.g.to_bits(), "axis {a} cell {i} g");
                            assert_eq!(bc.h.to_bits(), rc.h.to_bits(), "axis {a} cell {i} h");
                            assert_eq!(bc.count, rc.count, "axis {a} cell {i} count");
                        }
                        for (i, (bw, rw)) in b.wsum.iter().zip(&reference.wsum).enumerate() {
                            assert_eq!(bw.to_bits(), rw.to_bits(), "axis {a} cell {i} wsum");
                        }
                    }
                }
            }
        }
    }

    /// The one-leaf scatter (the root level: the leaf map is not read) and full 256-bin axes match
    /// the per-axis oracle bit for bit, sequential and chunked, unit and real weights.
    #[test]
    fn one_leaf_and_full_width_axes_match_the_per_axis_builder() {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state >> 33
        };
        let bins: Vec<u16> = vec![256, 2, 255, 3, 256, 40, 7, 256, 5, 200];
        for n in [700_usize, ROW_PAR_MIN_ROWS + 4_321] {
            let cols: Vec<Vec<u8>> = bins
                .iter()
                .map(|&nb| (0..n).map(|_| (next() % u64::from(nb)) as u8).collect())
                .collect();
            let x = matrix(cols, &bins);
            let g: Vec<f32> = (0..n)
                .map(|_| (next() % 2001) as f32 / 7.0 - 140.0)
                .collect();
            let h: Vec<f32> = (0..n)
                .map(|_| (next() % 999) as f32 / 13.0 + 0.01)
                .collect();
            let gh = gradhess(&g, &h);
            let w: Vec<f32> = (0..n).map(|_| 0.5 + (next() % 100) as f32 / 9.0).collect();
            let zeros = vec![0u8; n];
            let rows: Vec<u32> = (0..n as u32).filter(|r| r % 5 != 2).collect();
            let axes: Vec<u32> = (0..bins.len() as u32).collect();
            for unit_weight in [true, false] {
                let blocked = accumulate_axes_block_chunked(
                    &x,
                    &gh,
                    &rows,
                    &zeros,
                    1,
                    &axes,
                    &w,
                    unit_weight,
                )
                .unwrap();
                for (k, &a) in axes.iter().enumerate() {
                    let reference =
                        accumulate_axis(&x, &gh, &rows, &zeros, 1, a, &w, unit_weight).unwrap();
                    for (bc, rc) in blocked[k].ghc.iter().zip(&reference.ghc) {
                        assert_eq!(bc.g.to_bits(), rc.g.to_bits());
                        assert_eq!(bc.h.to_bits(), rc.h.to_bits());
                        assert_eq!(bc.count, rc.count);
                    }
                    for (bw, rw) in blocked[k].wsum.iter().zip(&reference.wsum) {
                        assert_eq!(bw.to_bits(), rw.to_bits());
                    }
                }
            }
        }
    }

    /// A bin id past the axis's grid is still an error (the engine never produces one).
    #[test]
    fn out_of_grid_bin_is_still_an_error() {
        let x = matrix(vec![vec![0, 1, 5, 2]], &[4]);
        let gh = gradhess(&[1.0; 4], &[1.0; 4]);
        let rows: Vec<u32> = (0..4).collect();
        for leaves in [vec![0u8; 4], vec![0, 1, 1, 0]] {
            let n_leaves = usize::from(*leaves.iter().max().unwrap()) + 1;
            assert!(
                build_histogram(&x, &gh, &rows, &leaves, n_leaves, &[0], &[1.0; 4], true).is_err()
            );
        }
    }

    /// `cargo test --release -p t-boost-core --lib bench_row_scatter -- --ignored --nocapture`:
    /// single-thread ns per (row, axis) update, old scatter vs new, on catelematic13-like data.
    #[test]
    #[ignore]
    fn bench_row_scatter() {
        let mut state = 0x5DEE_CE66_D1CE_4E5B_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state >> 33
        };
        let n_rows = 80_000usize;
        let n_axes = 72usize;
        let bins: Vec<u16> = (0..n_axes)
            .map(|k| match k % 4 {
                0 => 2 + (next() % 3) as u16,
                1 => 10 + (next() % 30) as u16,
                _ => 200 + (next() % 56) as u16,
            })
            .collect();
        let cols: Vec<Vec<u8>> = bins
            .iter()
            .map(|&nb| {
                (0..n_rows)
                    .map(|_| (next() % u64::from(nb)) as u8)
                    .collect()
            })
            .collect();
        let x = matrix(cols, &bins);
        let g: Vec<f32> = (0..n_rows)
            .map(|_| (next() % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let h: Vec<f32> = (0..n_rows)
            .map(|_| (next() % 999) as f32 / 1000.0 + 0.01)
            .collect();
        let gh = gradhess(&g, &h);
        let w = vec![1.0_f32; n_rows];
        let all: Vec<u32> = (0..n_rows as u32).filter(|_| next() % 100 < 72).collect();
        let half: Vec<u32> = all.iter().copied().filter(|_| next() % 2 == 0).collect();
        let axes: Vec<u32> = (0..n_axes as u32).collect();
        let scen: Vec<(&str, &Vec<u32>, usize)> = vec![
            ("level0", &all, 1),
            ("level1", &half, 2),
            ("level2", &half, 4),
        ];
        for (name, rows, n_leaves) in scen {
            let leaf: Vec<u8> = (0..n_rows)
                .map(|_| (next() % n_leaves as u64) as u8)
                .collect();
            let leaf = if n_leaves == 1 {
                vec![0u8; n_rows]
            } else {
                leaf
            };
            let reps = 20;
            let updates = (rows.len() * n_axes * reps) as f64;
            let mut best = [f64::INFINITY; 2];
            for _ in 0..3 {
                for (v, slot) in best.iter_mut().enumerate() {
                    let t0 = std::time::Instant::now();
                    for _ in 0..reps {
                        for block in axes.chunks(AXIS_BLOCK) {
                            let out = if v == 0 {
                                accumulate_axes_blocked_reference(
                                    &x, &gh, rows, &leaf, n_leaves, block, &w, true,
                                )
                            } else {
                                accumulate_axes_blocked(
                                    &x, &gh, rows, &leaf, n_leaves, block, &w, true,
                                )
                            };
                            std::hint::black_box(out.unwrap());
                        }
                    }
                    *slot = slot.min(t0.elapsed().as_secs_f64() * 1e9 / updates);
                }
            }
            println!(
                "{name}: rows {} leaves {n_leaves}: old {:.2} ns/update, new {:.2} ns/update ({:.2}x)",
                rows.len(),
                best[0],
                best[1],
                best[0] / best[1]
            );
        }
    }

    /// Most existing tests don't exercise weighted Σw, so they build with unit weights
    /// (then `wsum == count` as `f64`). The dedicated `wsum_*` tests below pass real
    /// weights to `build_histogram`/`build_quantized_histogram` directly.
    fn build_hist(
        x: &BinnedMatrix,
        gh: &GradHess,
        rows: &[u32],
        leaf_of_row: &[u8],
        n_leaves: usize,
        axes: &[u32],
    ) -> Result<Hist, PbError> {
        build_histogram(
            x,
            gh,
            rows,
            leaf_of_row,
            n_leaves,
            axes,
            &vec![1.0_f32; gh.g.len()],
            false,
        )
    }

    fn cell(hist: &Hist, leaf: usize, axis: usize, bin: usize) -> (f64, f64, u32) {
        let o = hist.offset(leaf, axis, bin).unwrap();
        (hist.g[o], hist.h[o], hist.count[o])
    }

    #[test]
    fn scale_for_normal_max_abs_is_unaffected() {
        // Normal-range values must be byte-identical to the pre-fix formula (no
        // behavior change): the `f32::MAX` clamp is a no-op whenever the f64 scale
        // doesn't overflow f32 on cast.
        let scale = scale_for(2.0).unwrap();
        let expected = ((f64::from(i32::MAX) * 0.5) / 2.0) as f32;
        assert_eq!(scale, expected);
    }

    #[test]
    fn scale_for_subnormal_max_abs_degrades_gracefully_instead_of_erroring() {
        // max_abs below ~3.2e-30 previously pushed the f64 scale past f32::MAX, so the
        // `as f32` cast overflowed to +inf and this returned Err, aborting the whole
        // fit for what is really a near-converged, graceful-stop situation. It must now
        // return a finite, positive, in-range scale instead.
        let tiny = 1e-33_f64;
        let scale =
            scale_for(tiny).expect("subnormal-tiny max_abs must degrade gracefully, not error");
        assert!(scale.is_finite());
        assert!(scale > 0.0);
        assert_eq!(scale, f32::MAX);
        // Every |g|/|h| in this round is <= max_abs by construction, so it still maps
        // to a bounded, valid i32 quantum at the clamped scale.
        let quantized = tiny * f64::from(scale);
        assert!(quantized.is_finite());
        assert!(quantized.abs() < f64::from(i32::MAX));
    }

    #[test]
    fn build_uses_uniform_stride_and_sums_per_axis_bin() {
        // axis0 has 3 bins, axis1 has 4 bins ⇒ uniform stride max_bins = 4.
        let cols = vec![vec![1u8, 2, 1, 2], vec![1u8, 1, 2, 3]];
        let x = matrix(cols, &[3, 4]);
        let gh = gradhess(&[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0, 1.0, 1.0]);
        let rows: Vec<u32> = (0..4).collect();
        let leaf_of_row = vec![0u8; 4];
        let hist = build_hist(&x, &gh, &rows, &leaf_of_row, 1, &[0, 1]).unwrap();

        assert_eq!(hist.shape(), (1, 2, 4)); // uniform stride 4
                                             // axis0: bin1 = rows 0,2 (g 1+3=4, count 2); bin2 = rows 1,3 (g 2+4=6, count 2).
        assert_eq!(cell(&hist, 0, 0, 1), (4.0, 2.0, 2));
        assert_eq!(cell(&hist, 0, 0, 2), (6.0, 2.0, 2));
        // axis0's high bin (3, beyond its 3-bin grid) and missing bin are zero.
        assert_eq!(cell(&hist, 0, 0, 0), (0.0, 0.0, 0));
        assert_eq!(cell(&hist, 0, 0, 3), (0.0, 0.0, 0));
        // axis1: bin1 = rows 0,1 (3, count 2); bin2 = row 2 (3, count 1); bin3 = row 3 (4, count 1).
        assert_eq!(cell(&hist, 0, 1, 1), (3.0, 2.0, 2));
        assert_eq!(cell(&hist, 0, 1, 2), (3.0, 1.0, 1));
        assert_eq!(cell(&hist, 0, 1, 3), (4.0, 1.0, 1));
    }

    #[test]
    fn build_partitions_rows_by_leaf() {
        let cols = vec![vec![1u8, 1, 1, 1]]; // all in bin 1
        let x = matrix(cols, &[3]);
        let gh = gradhess(&[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0, 1.0, 1.0]);
        let rows: Vec<u32> = (0..4).collect();
        let leaf_of_row = vec![0u8, 0, 1, 1]; // rows 0,1 -> leaf 0; rows 2,3 -> leaf 1
        let hist = build_hist(&x, &gh, &rows, &leaf_of_row, 2, &[0]).unwrap();
        assert_eq!(cell(&hist, 0, 0, 1), (3.0, 2.0, 2)); // 1+2
        assert_eq!(cell(&hist, 1, 0, 1), (7.0, 2.0, 2)); // 3+4
    }

    fn fixture() -> (BinnedMatrix, GradHess, Vec<u32>, Vec<u8>) {
        let n = 200usize;
        let c0: Vec<u8> = (0..n).map(|i| ((i % 5) + 1) as u8).collect();
        let c1: Vec<u8> = (0..n).map(|i| ((i % 9) + 1) as u8).collect();
        let c2: Vec<u8> = (0..n).map(|i| ((i * 7 % 11) + 1) as u8).collect();
        let x = matrix(vec![c0, c1, c2], &[6, 10, 12]);
        let g: Vec<f32> = (0..n).map(|i| (i as f32 % 13.0) - 6.0).collect();
        let h: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32 % 3.0)).collect();
        let gh = gradhess(&g, &h);
        let rows: Vec<u32> = (0..n as u32).collect();
        let leaf_of_row: Vec<u8> = (0..n).map(|i| (i % 4) as u8).collect();
        (x, gh, rows, leaf_of_row)
    }

    #[test]
    fn histogram_is_byte_identical_across_thread_counts() {
        let (x, gh, rows, leaf_of_row) = fixture();
        let axes = [0u32, 1, 2];
        let run = |nt: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| build_hist(&x, &gh, &rows, &leaf_of_row, 4, &axes).unwrap())
        };
        let h1 = run(1);
        let h2 = run(2);
        let h8 = run(8);
        // Byte-identical f64 cells across thread counts (fixed-order fold).
        let bits = |h: &Hist| -> (Vec<u64>, Vec<u64>, Vec<u32>) {
            (
                h.g.iter().map(|v| v.to_bits()).collect(),
                h.h.iter().map(|v| v.to_bits()).collect(),
                h.count.clone(),
            )
        };
        assert_eq!(bits(&h1), bits(&h2));
        assert_eq!(bits(&h1), bits(&h8));
    }

    #[test]
    fn row_parallel_histogram_is_byte_identical_across_thread_counts() {
        let n = ROW_PAR_MIN_ROWS + 1024;
        let c0: Vec<u8> = (0..n).map(|i| ((i % 13) + 1) as u8).collect();
        let c1: Vec<u8> = (0..n).map(|i| ((i * 7 % 17) + 1) as u8).collect();
        let x = matrix(vec![c0, c1], &[14, 18]);
        let g: Vec<f32> = (0..n).map(|i| ((i % 19) as f32 - 9.0) * 0.125).collect();
        let h: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32).collect();
        let gh = gradhess(&g, &h);
        let rows: Vec<u32> = (0..n as u32).collect();
        let leaf_of_row: Vec<u8> = (0..n).map(|i| (i % 4) as u8).collect();
        let axes = [0u32, 1];
        let run = |nt: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| build_hist(&x, &gh, &rows, &leaf_of_row, 4, &axes).unwrap())
        };
        let bits = |h: &Hist| -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u32>) {
            (
                h.g.iter().map(|v| v.to_bits()).collect(),
                h.h.iter().map(|v| v.to_bits()).collect(),
                h.wsum.iter().map(|v| v.to_bits()).collect(),
                h.count.clone(),
            )
        };
        let h1 = run(1);
        assert_eq!(bits(&h1), bits(&run(2)));
        assert_eq!(bits(&h1), bits(&run(8)));
    }

    #[test]
    fn assembly_layout_distinguishes_leaf_from_axis() {
        // n_leaves=2 AND n_axes=2: a leaf/axis transposition in the flat offset would
        // swap the off-diagonal cells (leaf1,axis0) <-> (leaf0,axis1). Distinct values
        // at those cells catch it (the build's `dst` and `Hist::offset` must agree).
        let c0 = vec![1u8, 1, 1, 1]; // axis0: all bin 1
        let c1 = vec![2u8, 2, 2, 2]; // axis1: all bin 2
        let x = matrix(vec![c0, c1], &[3, 3]);
        let gh = gradhess(&[10.0, 20.0, 100.0, 200.0], &[1.0, 1.0, 1.0, 1.0]);
        let rows: Vec<u32> = (0..4).collect();
        let leaf_of_row = vec![0u8, 0, 1, 1]; // leaf0: rows 0,1 (Σg 30); leaf1: rows 2,3 (Σg 300)
        let hist = build_hist(&x, &gh, &rows, &leaf_of_row, 2, &[0, 1]).unwrap();

        assert_eq!(cell(&hist, 0, 0, 1), (30.0, 2.0, 2)); // leaf0, axis0
        assert_eq!(cell(&hist, 1, 0, 1), (300.0, 2.0, 2)); // leaf1, axis0  (off-diagonal)
        assert_eq!(cell(&hist, 0, 1, 2), (30.0, 2.0, 2)); // leaf0, axis1  (off-diagonal)
        assert_eq!(cell(&hist, 1, 1, 2), (300.0, 2.0, 2)); // leaf1, axis1
                                                           // The cells a transposition would read instead are empty in the correct layout.
        assert_eq!(cell(&hist, 1, 0, 2), (0.0, 0.0, 0));
        assert_eq!(cell(&hist, 0, 1, 1), (0.0, 0.0, 0));
    }

    #[test]
    fn oversized_hist_shapes_error_without_overflowing() {
        assert!(matches!(
            Hist::try_zeros(usize::MAX, 2, 2),
            Err(PbError::Internal { .. })
        ));

        let x = matrix(vec![Vec::new()], &[3]);
        let gh = gradhess(&[], &[]);
        assert!(matches!(
            build_hist(&x, &gh, &[], &[], usize::MAX, &[0]),
            Err(PbError::Internal { .. })
        ));
    }

    #[test]
    fn malformed_hist_offset_overflow_returns_none() {
        let hist = Hist {
            n_leaves: usize::MAX,
            n_axes: usize::MAX,
            n_bins: usize::MAX,
            ..Hist::default()
        };
        assert_eq!(hist.offset(2, 0, 0), None);
    }

    #[test]
    fn out_of_range_inputs_return_internal_not_panic() {
        let x = matrix(vec![vec![1u8, 1]], &[3]);
        let gh = gradhess(&[1.0, 2.0], &[1.0, 1.0]);
        let rows = [0u32, 1];
        // Axis with no grid/column.
        assert!(matches!(
            build_hist(&x, &gh, &rows, &[0, 0], 1, &[5]),
            Err(PbError::Internal { .. })
        ));
        // A leaf_of_row entry >= n_leaves.
        assert!(matches!(
            build_hist(&x, &gh, &rows, &[0, 3], 1, &[0]),
            Err(PbError::Internal { .. })
        ));
        // A bin id >= max_bins (malformed column value 5 against a 3-bin grid).
        let bad = matrix(vec![vec![5u8, 1]], &[3]);
        assert!(matches!(
            build_hist(&bad, &gh, &rows, &[0, 0], 1, &[0]),
            Err(PbError::Internal { .. })
        ));
        // A row id outside the column.
        assert!(matches!(
            build_hist(&x, &gh, &[0, 9], &[0, 0], 1, &[0]),
            Err(PbError::Internal { .. })
        ));
    }

    #[test]
    fn subtract_sibling_into_matches_hand_computed_larger_child() {
        // parent: 2 leaves × 2 axes × 3 bins; child: 4 leaves × 1 axis (axis_map=[1], so the
        // child's only axis is the parent's axis 1 — a NON-identity remap, the A_2⊂A_1 case)
        // × 3 bins. Smaller children (leaves 0,1) are populated; the larger (2,3) are derived.
        let mut parent = Hist::try_zeros(2, 2, 3).unwrap();
        let mut child = Hist::try_zeros(4, 1, 3).unwrap();
        let set = |hh: &mut Hist, leaf: usize, axis: usize, bin: usize, g: f64, c: u32| {
            let o = hh.offset(leaf, axis, bin).unwrap();
            hh.g[o] = g;
            hh.h[o] = g * 0.1;
            hh.wsum[o] = f64::from(c);
            hh.count[o] = c;
        };
        // parent axis 1 (axis 0 left zero — unused by the map): leaf 0 then leaf 1.
        set(&mut parent, 0, 1, 0, 10.0, 10);
        set(&mut parent, 0, 1, 1, 20.0, 20);
        set(&mut parent, 0, 1, 2, 30.0, 30);
        set(&mut parent, 1, 1, 0, 5.0, 5);
        set(&mut parent, 1, 1, 1, 15.0, 15);
        set(&mut parent, 1, 1, 2, 25.0, 25);
        // smaller children on child axis 0: leaf 0 (smaller of parent 0), leaf 1 (of parent 1).
        set(&mut child, 0, 0, 0, 3.0, 3);
        set(&mut child, 0, 0, 1, 8.0, 8);
        set(&mut child, 0, 0, 2, 12.0, 12);
        set(&mut child, 1, 0, 0, 2.0, 2);
        set(&mut child, 1, 0, 1, 5.0, 5);
        set(&mut child, 1, 0, 2, 10.0, 10);
        // parent p → (smaller, larger): 0 → (0, 2); 1 → (1, 3).
        subtract_sibling_into(&mut child, &parent, &[(0, 0, 2), (1, 1, 3)], &[1]).unwrap();
        let chk = |leaf: usize, bin: usize, g: f64, c: u32| {
            let o = child.offset(leaf, 0, bin).unwrap();
            assert!((child.g[o] - g).abs() < 1e-12, "g leaf {leaf} bin {bin}");
            assert!(
                (child.wsum[o] - f64::from(c)).abs() < 1e-12,
                "wsum leaf {leaf} bin {bin}"
            );
            assert_eq!(child.count[o], c, "count leaf {leaf} bin {bin}");
        };
        // larger children = parent − smaller.
        chk(2, 0, 7.0, 7);
        chk(2, 1, 12.0, 12);
        chk(2, 2, 18.0, 18);
        chk(3, 0, 3.0, 3);
        chk(3, 1, 10.0, 10);
        chk(3, 2, 15.0, 15);
        // smaller children untouched.
        chk(0, 0, 3.0, 3);
        chk(1, 2, 10.0, 10);
    }

    #[test]
    fn subtract_sibling_into_count_underflow_is_internal() {
        let mut parent = Hist::try_zeros(1, 1, 1).unwrap();
        parent.count[0] = 2;
        let mut child = Hist::try_zeros(2, 1, 1).unwrap();
        let o = child.offset(0, 0, 0).unwrap();
        child.count[o] = 5; // smaller (5) > parent leaf (2) ⇒ underflow on the larger child
        assert!(matches!(
            subtract_sibling_into(&mut child, &parent, &[(0, 0, 1)], &[0]),
            Err(PbError::Internal { .. })
        ));
    }

    #[test]
    fn subtract_sibling_into_shape_mismatch_errors() {
        // child.n_leaves must be 2× parent.
        let parent = Hist::try_zeros(2, 1, 1).unwrap();
        let mut child = Hist::try_zeros(2, 1, 1).unwrap();
        assert!(matches!(
            subtract_sibling_into(&mut child, &parent, &[(0, 0, 1)], &[0]),
            Err(PbError::ShapeMismatch { .. })
        ));
        // axis_map.len() must equal child.n_axes.
        let parent2 = Hist::try_zeros(2, 2, 1).unwrap();
        let mut child2 = Hist::try_zeros(4, 2, 1).unwrap();
        assert!(matches!(
            subtract_sibling_into(&mut child2, &parent2, &[(0, 0, 2)], &[0]),
            Err(PbError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn counts_are_bounded_by_rows_and_total_matches() {
        // Every cell's count <= n_rows, and the total count over a single axis equals
        // the number of accumulated rows (the u32 count can never exceed n_rows).
        let (x, gh, rows, leaf_of_row) = fixture();
        let hist = build_hist(&x, &gh, &rows, &leaf_of_row, 4, &[0]).unwrap();
        let total: u64 = hist.count.iter().map(|&c| u64::from(c)).sum();
        assert_eq!(total, rows.len() as u64);
        assert!(hist
            .count
            .iter()
            .all(|&c| u64::from(c) <= u64::from(x.n_rows)));
    }

    #[test]
    fn quantize_grad_hess_is_deterministic_and_half_step_bounded() {
        let gh = gradhess(&[0.0, 0.125, -0.75, 2.5, -3.25], &[1.0, 0.5, 2.0, 4.0, 8.0]);
        let q1 = quantize_grad_hess(&gh, 77, 3).unwrap();
        let q2 = quantize_grad_hess(&gh, 77, 3).unwrap();
        assert_eq!(q1, q2);
        let g_step = 1.0_f64 / f64::from(q1.scale.g_scale);
        let h_step = 1.0_f64 / f64::from(q1.scale.h_scale);
        for ((&g, &gq), (&h, &hq)) in gh.g.iter().zip(&q1.g_q).zip(gh.h.iter().zip(&q1.h_q)) {
            let dg = (f64::from(gq) * g_step - f64::from(g)).abs();
            let dh = (f64::from(hq) * h_step - f64::from(h)).abs();
            assert!(dg <= 0.5 * g_step + f64::EPSILON, "dg={dg}, step={g_step}");
            assert!(dh <= 0.5 * h_step + f64::EPSILON, "dh={dh}, step={h_step}");
        }
    }

    #[test]
    fn quantized_histogram_is_thread_count_independent() {
        let (x, gh, rows, leaf_of_row) = fixture();
        let qgh = quantize_grad_hess(&gh, 99, 4).unwrap();
        let build = |threads: usize| -> Hist {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                build_quantized_histogram(
                    &x,
                    &qgh,
                    &rows,
                    &leaf_of_row,
                    4,
                    &[0, 1],
                    &vec![1.0_f32; gh.g.len()],
                    false,
                )
                .unwrap()
            })
        };
        let h1 = build(1);
        assert_eq!(h1, build(2));
        assert_eq!(h1, build(8));
    }

    #[test]
    fn wsum_tracks_weighted_mass_per_cell() {
        // Two cells (bin 1 / bin 2) on one axis; weights differ from counts so Σw must be
        // distinct from `count` (the credibility `min_weight_sum_in_leaf` path).
        let cols = vec![vec![1u8, 2, 1, 2]];
        let x = matrix(cols, &[3]);
        let gh = gradhess(&[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0, 1.0, 1.0]);
        let weight = [0.5_f32, 2.0, 1.5, 4.0];
        let rows: Vec<u32> = (0..4).collect();
        let one_leaf = vec![0u8; 4];
        let hist = build_histogram(&x, &gh, &rows, &one_leaf, 1, &[0], &weight, false).unwrap();
        let o1 = hist.offset(0, 0, 1).unwrap();
        let o2 = hist.offset(0, 0, 2).unwrap();
        // bin1 = rows 0,2 (w 0.5+1.5=2.0); bin2 = rows 1,3 (w 2.0+4.0=6.0).
        assert!((hist.wsum[o1] - 2.0).abs() < 1e-12);
        assert!((hist.wsum[o2] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn unit_weights_make_wsum_equal_count() {
        let (x, gh, rows, leaf_of_row) = fixture();
        let hist = build_hist(&x, &gh, &rows, &leaf_of_row, 4, &[0, 1]).unwrap();
        for (w, &c) in hist.wsum.iter().zip(&hist.count) {
            assert!((w - f64::from(c)).abs() < 1e-12);
        }
    }

    #[test]
    fn unit_weight_fast_path_is_bit_identical_to_full_sigma_w() {
        // The `unit_weight=true` fast path (skip per-row Σw, set wsum=count) must produce a
        // histogram bit-for-bit equal to the full Σw path fed all-ones — the byte-identity
        // the engine relies on whenever the caller supplied no weights. Exercise both the
        // sequential and the row-chunk-parallel branches (rows >= ROW_PAR_MIN_ROWS).
        let (xs, gh_s, rows_s, leaf_s) = fixture();
        let ones_s = vec![1.0_f32; gh_s.g.len()];
        let big_n = ROW_PAR_MIN_ROWS + 257;
        let cols: Vec<Vec<u8>> = vec![
            (0..big_n).map(|i| (i % 7) as u8 + 1).collect(),
            (0..big_n).map(|i| (i % 5) as u8 + 1).collect(),
        ];
        let xb = matrix(cols, &[8, 6]);
        let gh_b = gradhess(
            &(0..big_n)
                .map(|i| (i % 11) as f32 - 5.0)
                .collect::<Vec<_>>(),
            &(0..big_n).map(|i| 1.0 + (i % 3) as f32).collect::<Vec<_>>(),
        );
        let rows_b: Vec<u32> = (0..big_n as u32).collect();
        let leaf_b = vec![0u8; big_n];
        let ones_b = vec![1.0_f32; big_n];
        for (x, gh, rows, leaf, ones, nl, axes) in [
            (
                &xs,
                &gh_s,
                &rows_s,
                &leaf_s,
                &ones_s,
                4usize,
                &[0u32, 1][..],
            ),
            (
                &xb,
                &gh_b,
                &rows_b,
                &leaf_b,
                &ones_b,
                1usize,
                &[0u32, 1][..],
            ),
        ] {
            let slow = build_histogram(x, gh, rows, leaf, nl, axes, ones, false).unwrap();
            let fast = build_histogram(x, gh, rows, leaf, nl, axes, ones, true).unwrap();
            assert_eq!(slow.g.len(), fast.g.len());
            for i in 0..slow.g.len() {
                assert_eq!(slow.g[i].to_bits(), fast.g[i].to_bits(), "g cell {i}");
                assert_eq!(slow.h[i].to_bits(), fast.h[i].to_bits(), "h cell {i}");
                assert_eq!(
                    slow.wsum[i].to_bits(),
                    fast.wsum[i].to_bits(),
                    "wsum cell {i}"
                );
                assert_eq!(slow.count[i], fast.count[i], "count cell {i}");
            }
        }
    }
}
