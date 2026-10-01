//! §G1 fully-corrective cell-basis refit: a per-term adaptive ridge on the purified
//! fANOVA *cell* basis, fit to the bagged out-of-bag residual.
//!
//! t-boost grows the right supports but a shared greedy budget under-fits weak interaction
//! *surfaces* (the depth-2 pair-fitting gap vs EBM). This module re-fits the cell values
//! of the realized mains + pairs jointly — a totally-corrective step on the basis the
//! decomposition actually lives in (cells, not leaves) — with an adaptive ridge that gives
//! high-signal terms more freedom. The solution is returned as a [`CorrectionBank`] of raw
//! per-merged-cell deltas; attaching it to the model keeps G0 exact (see
//! [`crate::engine::Model::correction_delta`] and the decompose-side fold in
//! [`crate::explain`]).
//!
//! The design column count can be large (one column per merged cell, summed over every
//! corrected support), so the joint ridge is solved with a diagonally-preconditioned
//! conjugate gradient that never materialises `XᵀX` — the existing dense refit
//! (`fully_corrective_refit`) would be infeasible at cell width.
//!
//! Performance: the two mat-vecs dominate the solve. They have DISJOINT per-thread outputs
//! (`matvec` over rows, `rmatvec` over the support-partitioned column windows), so they are
//! parallelised with no floating-point reduction — bit-identical to the sequential result,
//! hence byte-deterministic across thread counts. The solve also (a) fits only nonzero-weight
//! rows (held-out/uncovered rows contribute exactly 0), (b) shares the byte-identical `b` and
//! `col_w` across both solves, (c) runs a cheap capped flat init (its per-term magnitudes are
//! all that feed the reweighting), and (d) warm-starts the adaptive solve from it.

// The G1 cell-refit solver is dense index-arithmetic over the sparse cell basis (hot loops like
// `v[off + active_u[base + i]]`); a newer clippy tightened the workspace `indexing_slicing = deny`
// to flag these. Scope-allowed here (bounds are structurally guaranteed by the design build) to keep
// CI green without rewriting the tuned solver. TODO: convert to `.get()` incrementally.
#![allow(
    clippy::indexing_slicing, // JUSTIFIED: pre-existing module-scoped debt (see comment above); burn-down to `.get()`/per-fn allows is incremental, not expanded here.
    clippy::manual_slice_fill,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::derivable_impls
)]

use crate::engine::{CorrectionBank, Model};
use crate::error::PbError;
use crate::explain::correction_scaffold;
use rayon::prelude::*;

/// Hyper-parameters for the cell-basis refit. Suite-wide (no per-dataset tuning).
#[derive(Debug, Clone, Copy)]
pub struct CellRefitSpec {
    /// Base ridge penalty `α` on the cell coefficients.
    pub base: f64,
    /// Adaptive exponent `γ`: per-term penalty is `base / (s_term^γ normalised)²`, so
    /// high-signal terms (large `s_term`) are penalised less. `γ = 0` is a flat ridge.
    pub gamma: f64,
    /// Maximum coarse cells per axis for ORDER-2 (pair) corrections; mains keep full merged
    /// resolution. The full merged pair grid (realized borders of both features) over-fits
    /// noisy high-cardinality surfaces — acute for categoricals — so pairs are solved on a
    /// coarsened grid (contiguous groups of merged cells), matching the prototype's coarse
    /// pair grids. The δ stays constant WITHIN merged cells, so the correction is still
    /// stored at merged resolution and G0 is untouched.
    pub pair_cell_cap: u32,
    /// Sweep cap for the CHEAP flat init. Its per-cell values are discarded — only the per-term
    /// RMS magnitudes feed the adaptive reweighting — so a rough solve suffices.
    pub flat_cg_iters: usize,
    /// Sweep cap for the accurate adaptive solve.
    pub cg_iters: usize,
    /// Relative per-sweep coefficient-change tolerance (max|Δ| / max|δ|). The solved δ is
    /// globally rescaled by the held-out no-harm guard, so sub-noise precision is wasted.
    pub cg_tol: f64,
    /// Maximum rows used to FIT the correction. The correction is a coarse, over-determined surface
    /// (≤ `pair_cell_cap`² coeffs per pair), so above this a deterministic strided subsample of the
    /// (sorted) nonzero-weight rows gives a near-identical fit while bounding `Design.active`
    /// (supports × rows × u16) and the solve — the two structures that scale with row count. Set far
    /// above every realistic dataset's OOB-covered row count, so ordinary fits are NEVER subsampled
    /// (byte-identical); only very large datasets (10M+ rows) hit it.
    pub max_fit_rows: usize,
}

impl Default for CellRefitSpec {
    fn default() -> Self {
        // base≈4000, γ≈2 — the held-out-validated prototype plateau; pair grid capped at the
        // prototype's 24 cells/axis. Backfitting converges in far fewer sweeps than CG needs
        // iterations, so the sweep caps are small (30 flat / 100 adaptive); tol 1e-4.
        CellRefitSpec {
            base: 4000.0,
            gamma: 2.0,
            pair_cell_cap: 24,
            flat_cg_iters: 30,
            cg_iters: 100,
            cg_tol: 1e-4,
            // ~9x the widest current dataset's OOB fit rows (allstate ≈ 106k), so nothing in the
            // suite is ever subsampled; only 10M-row-scale datasets hit the cap.
            max_fit_rows: 1_000_000,
        }
    }
}

/// Block size (rows) for the row-parallel `matvec`: sized so each thread's output slice stays
/// L2-resident across its passes over the supports.
const MATVEC_BLOCK: usize = 4096;

/// The sparse cell-indicator design, stored implicitly: every row activates exactly one
/// COARSE cell per corrected support. `active[u][i]` is the SUPPORT-LOCAL coarse cell id
/// (u16 — halves the index bandwidth the mat-vecs stream) of the `i`-th fit row on support
/// `u`; the global column is `col_offset[u] + active[u][i]`.
struct Design {
    /// `active[u][i]` = support-local coarse cell id (`< term_cols[u]`) of fit row `i`.
    active: Vec<Vec<u16>>,
    /// Total coarse column count (Σ coarse cells over supports).
    n_cols: usize,
    /// `col_term[c]` = the support index owning coarse column `c` (for per-term reweighting).
    col_term: Vec<u32>,
    /// First global coarse column id of each support (prefix sum of per-support coarse cells).
    col_offset: Vec<u32>,
    /// Coarse column count of each support (`col_offset[u+1] − col_offset[u]`); the width of
    /// support `u`'s disjoint column window, used to partition `rmatvec` output in parallel.
    term_cols: Vec<u32>,
    /// `coarse_shape[u][k]` = coarse cell count of support `u`'s axis `k` (parallel to its axes).
    coarse_shape: Vec<Vec<u32>>,
    /// `groups[u][k][merged_cell]` = the coarse group id a merged cell maps to, for expanding
    /// the solved δ back to merged resolution when filling the [`CorrectionBank`].
    groups: Vec<Vec<Vec<u32>>>,
}

/// Bijective 64-bit scramble (splitmix64 finalizer): deterministic, tie-free, and uncorrelated with
/// input order. Used to subsample the cell-refit fit rows on huge datasets without biasing on sorted
/// or periodic row order.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Map each of `merged_m` merged cells to one of `min(merged_m, cap)` contiguous coarse
/// groups (`cap` ≥ 1). Returns the per-merged-cell group ids and the coarse group count.
fn coarse_groups(merged_m: u32, cap: u32) -> (Vec<u32>, u32) {
    let k = merged_m.min(cap.max(1)).max(1);
    let groups: Vec<u32> = (0..merged_m)
        .map(|c| ((u64::from(c) * u64::from(k)) / u64::from(merged_m.max(1))) as u32)
        .collect();
    (groups, k)
}

impl Design {
    /// `out = X v` (length n_rows). Parallel over disjoint row blocks; each `out[r]` is summed
    /// over supports in fixed order, so the result is bit-identical to the sequential loop
    /// regardless of thread count (no cross-thread reduction).
    fn matvec(&self, v: &[f64], out: &mut [f64]) {
        out.par_chunks_mut(MATVEC_BLOCK)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let base = ci * MATVEC_BLOCK;
                for o in chunk.iter_mut() {
                    *o = 0.0;
                }
                for (u, active_u) in self.active.iter().enumerate() {
                    let off = self.col_offset[u] as usize;
                    for (i, o) in chunk.iter_mut().enumerate() {
                        *o += v[off + active_u[base + i] as usize];
                    }
                }
            });
    }

    /// `out = Xᵀ y` (length n_cols). Parallel over supports: each support owns a DISJOINT
    /// column window `[col_offset[u], col_offset[u]+term_cols[u])`, so scatters never race and
    /// each column accumulates its rows in fixed order — bit-identical to the sequential loop.
    fn rmatvec(&self, y: &[f64], out: &mut [f64]) {
        // Partition `out` into per-support column windows (disjoint mutable sub-slices).
        let mut windows: Vec<&mut [f64]> = Vec::with_capacity(self.active.len());
        let mut rest = &mut out[..];
        for &w in &self.term_cols {
            let (head, tail) = rest.split_at_mut(w as usize);
            windows.push(head);
            rest = tail;
        }
        windows.into_par_iter().enumerate().for_each(|(u, win)| {
            for o in win.iter_mut() {
                *o = 0.0;
            }
            for (r, &c) in self.active[u].iter().enumerate() {
                win[c as usize] += y[r];
            }
        });
    }
}

/// Build the implicit COARSE design for `bank`'s supports over the fit rows `rows` (the
/// nonzero-weight subset). `data[axis][row]` is the model bin id; each axis's merged cells are
/// grouped into ≤`pair_cell_cap` coarse cells for ORDER-2 supports (mains keep full merged
/// resolution). Stores support-local coarse cell ids.
fn build_design(
    bank: &CorrectionBank,
    data: &[Vec<u8>],
    rows: &[usize],
    pair_cell_cap: u32,
) -> Result<Design, PbError> {
    let n_terms = bank.tables.len();
    let n_solve = rows.len();
    let mut col_offset = Vec::with_capacity(n_terms);
    let mut term_cols = Vec::with_capacity(n_terms);
    let mut coarse_shape: Vec<Vec<u32>> = Vec::with_capacity(n_terms);
    let mut groups: Vec<Vec<Vec<u32>>> = Vec::with_capacity(n_terms);
    let mut n_cols: usize = 0;
    for table in &bank.tables {
        let order = table.axes.len();
        let mut ks = Vec::with_capacity(order);
        let mut grps = Vec::with_capacity(order);
        let mut cols: usize = 1;
        for k in 0..order {
            let merged_m = table.shape[k];
            let cap = if order <= 1 { merged_m } else { pair_cell_cap };
            let (g, kk) = coarse_groups(merged_m, cap);
            ks.push(kk);
            grps.push(g);
            cols = cols
                .checked_mul(kk as usize)
                .ok_or_else(|| PbError::Internal {
                    what: "cell-refit coarse term column count overflow".into(),
                })?;
        }
        // Support-local cell ids are u16: guard the per-support coarse width. Pairs are
        // ≤ pair_cell_cap² (576 at the default), mains ≤ merged cells ≤ n_bins.
        if cols > usize::from(u16::MAX) {
            return Err(PbError::Internal {
                what: format!("cell-refit support coarse width {cols} exceeds u16"),
            });
        }
        col_offset.push(u32::try_from(n_cols).map_err(|_| PbError::Internal {
            what: "cell-refit column offset exceeded u32".into(),
        })?);
        term_cols.push(cols as u32);
        n_cols = n_cols.checked_add(cols).ok_or_else(|| PbError::Internal {
            what: "cell-refit column count overflow".into(),
        })?;
        coarse_shape.push(ks);
        groups.push(grps);
    }
    let mut col_term = vec![0u32; n_cols];
    for (t, &off) in col_offset.iter().enumerate() {
        let start = off as usize;
        let end = start + term_cols[t] as usize;
        for slot in &mut col_term[start..end] {
            *slot = t as u32;
        }
    }
    let mut active: Vec<Vec<u16>> = Vec::with_capacity(n_terms);
    for (t, table) in bank.tables.iter().enumerate() {
        let mut col = vec![0u16; n_solve];
        for (k, &axis) in table.axes.iter().enumerate() {
            let column = data
                .get(axis as usize)
                .ok_or_else(|| PbError::ShapeMismatch {
                    what: format!("cell-refit: data has no axis {axis}"),
                })?;
            let map = &table.bin_to_cell[k];
            let grp = &groups[t][k];
            let kk = coarse_shape[t][k];
            for (i, slot) in col.iter_mut().enumerate() {
                let row = rows[i];
                let bin = *column.get(row).ok_or_else(|| PbError::ShapeMismatch {
                    what: format!("cell-refit: axis {axis} column shorter than rows"),
                })? as usize;
                let merged_cell = *map.get(bin).ok_or_else(|| PbError::InvalidInput {
                    what: format!("cell-refit: bin {bin} outside bin_to_cell axis {axis}"),
                })? as usize;
                let coarse = *grp.get(merged_cell).ok_or_else(|| PbError::Internal {
                    what: "cell-refit: merged cell escaped coarse groups".into(),
                })?;
                // Row-major fold stays < the support's coarse width (≤ u16::MAX, guarded above).
                *slot = (u32::from(*slot) * kk + coarse) as u16;
            }
        }
        active.push(col);
    }
    Ok(Design {
        active,
        n_cols,
        col_term,
        col_offset,
        term_cols,
        coarse_shape,
        groups,
    })
}

/// Solve `(XᵀWX + diag(lambda)) δ = b` by diagonally-preconditioned conjugate gradient, where
/// `b = XᵀW r` and `col_w = XᵀW` (the diagonal) are precomputed and shared across solves.
/// `w` is the per-row IRLS weight; `x0` an optional warm start. Deterministic: the mat-vecs are
/// bit-identical to sequential and all vector reductions are sequential.
///
/// Retained as a fallback; `fit_cell_correction` now uses [`solve_ridge_backfit`].
#[allow(clippy::too_many_arguments, dead_code)]
fn solve_ridge_cg(
    design: &Design,
    w: &[f64],
    b: &[f64],
    col_w: &[f64],
    lambda: &[f64],
    x0: Option<&[f64]>,
    max_iters: usize,
    tol: f64,
) -> Vec<f64> {
    let n_rows = w.len();
    let n_cols = b.len();

    let minv: Vec<f64> = (0..n_cols)
        .map(|c| {
            let d = col_w[c] + lambda[c];
            if d > 0.0 {
                1.0 / d
            } else {
                0.0
            }
        })
        .collect();

    // A·v = Xᵀ W X v + lambda ⊙ v (scratch buffers reused across iterations).
    let mut xv = vec![0.0_f64; n_rows];
    let mut wxv = vec![0.0_f64; n_rows];
    let apply_a = |v: &[f64], out: &mut [f64], xv: &mut [f64], wxv: &mut [f64]| {
        design.matvec(v, xv);
        for i in 0..n_rows {
            wxv[i] = w[i] * xv[i];
        }
        design.rmatvec(wxv, out);
        for c in 0..n_cols {
            out[c] += lambda[c] * v[c];
        }
    };

    let mut x = match x0 {
        Some(x0) => x0.to_vec(),
        None => vec![0.0_f64; n_cols],
    };
    // r0 = b − A·x0 (= b when x0 is zero).
    let mut resid = vec![0.0_f64; n_cols];
    if x0.is_some() {
        apply_a(&x, &mut resid, &mut xv, &mut wxv);
        for c in 0..n_cols {
            resid[c] = b[c] - resid[c];
        }
    } else {
        resid.copy_from_slice(b);
    }
    let mut z: Vec<f64> = (0..n_cols).map(|c| minv[c] * resid[c]).collect();
    let mut p = z.clone();
    let mut ap = vec![0.0_f64; n_cols];

    let b_norm = dot(b, b).sqrt().max(1e-30);
    let mut rz = dot(&resid, &z);
    for _ in 0..max_iters {
        if dot(&resid, &resid).sqrt() / b_norm <= tol {
            break;
        }
        apply_a(&p, &mut ap, &mut xv, &mut wxv);
        let pap = dot(&p, &ap);
        if pap <= 0.0 {
            break;
        }
        let alpha = rz / pap;
        for c in 0..n_cols {
            x[c] += alpha * p[c];
            resid[c] -= alpha * ap[c];
        }
        for c in 0..n_cols {
            z[c] = minv[c] * resid[c];
        }
        let rz_new = dot(&resid, &z);
        let beta = rz_new / rz;
        for c in 0..n_cols {
            p[c] = z[c] + beta * p[c];
        }
        rz = rz_new;
    }
    x
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Solve the SAME ridge `(XᵀWX + diag(lambda)) δ = XᵀW r_target` by block Gauss-Seidel
/// (backfitting) over supports. Within a support each row activates exactly one cell, so the
/// support's block of `XᵀWX` is DIAGONAL and its coordinate update is closed-form (a shrunk
/// weighted cell-mean) — no inner solve. One sweep costs one mat-vec pair but takes an EXACT
/// block step per support, so on the purified near-orthogonal basis it converges in far fewer
/// sweeps than CG needs iterations. Converges to the SAME minimizer, so it changes speed, not
/// the correction. Deterministic: supports swept in fixed order, each cell/residual reduction
/// in fixed row order (sequential — Gauss-Seidel couples supports within a sweep).
fn solve_ridge_backfit(
    design: &Design,
    w: &[f64],
    r_target: &[f64],
    col_w: &[f64],
    lambda: &[f64],
    x0: Option<&[f64]>,
    max_sweeps: usize,
    tol: f64,
) -> Vec<f64> {
    let n_rows = w.len();
    let n_cols = design.n_cols;
    let mut delta = match x0 {
        Some(x0) => x0.to_vec(),
        None => vec![0.0_f64; n_cols],
    };
    // Running residual res = r_target − X δ (maintained incrementally across supports).
    let mut res = r_target.to_vec();
    if x0.is_some() {
        let mut f = vec![0.0_f64; n_rows];
        design.matvec(&delta, &mut f);
        for i in 0..n_rows {
            res[i] -= f[i];
        }
    }
    let max_cells = design
        .term_cols
        .iter()
        .map(|&t| t as usize)
        .max()
        .unwrap_or(0);
    let mut s_buf = vec![0.0_f64; max_cells];
    let mut d_buf = vec![0.0_f64; max_cells];
    let mut d_prev = vec![0.0_f64; max_cells];
    // The support whose Δ has been solved but not yet taken out of `res` (see below).
    let mut pending: Option<usize> = None;
    for _sweep in 0..max_sweeps {
        let mut max_change = 0.0_f64;
        let mut max_coef = 1e-30_f64;
        for (u, active_u) in design.active.iter().enumerate() {
            let off = design.col_offset[u] as usize;
            let ncell = design.term_cols[u] as usize;
            // s[c] = Σ_{rows in cell c} w_r · res_r (the support's gradient against res).
            let s = &mut s_buf[..ncell];
            for v in s.iter_mut() {
                *v = 0.0;
            }
            // The previous support's residual update (Gauss-Seidel: this support sees it) is
            // applied in the same pass that reads `res` here: per row, exactly the subtraction
            // and then the read the two separate passes made, and every cell still sums its rows
            // in row order — bit-identical, with one pass over the rows instead of two.
            match pending.take() {
                Some(p) => {
                    let active_p = &design.active[p];
                    for ((i, &c), &cp) in active_u.iter().enumerate().zip(active_p) {
                        res[i] -= d_prev[cp as usize];
                        s[c as usize] += w[i] * res[i];
                    }
                }
                None => {
                    for (i, &c) in active_u.iter().enumerate() {
                        s[c as usize] += w[i] * res[i];
                    }
                }
            }
            // Exact diagonal-block ridge update; Δ = new − old.
            let d = &mut d_buf[..ncell];
            for (cc, dcc) in d.iter_mut().enumerate() {
                let c = off + cc;
                let old = delta[c];
                let cw = col_w[c];
                let den = cw + lambda[c];
                let new = if den > 0.0 {
                    (s[cc] + old * cw) / den
                } else {
                    0.0
                };
                *dcc = new - old;
                delta[c] = new;
                max_change = max_change.max(dcc.abs());
                max_coef = max_coef.max(new.abs());
            }
            // This support's change reaches `res` in the next support's pass (or the next sweep).
            std::mem::swap(&mut d_buf, &mut d_prev);
            pending = Some(u);
        }
        if max_change <= tol * max_coef {
            break;
        }
    }
    delta
}

/// Fit the §G1 cell-basis correction for `supports` (each a sorted list of model axis ids,
/// order 1..=2 — mains and pairs; triples are left to the trees). `data[axis][row]` is the
/// binned training matrix, `residual` the bagged OOB working residual, `sample_weight` the
/// per-row IRLS weight (1.0 for squared error; 0.0 excludes a row). Returns a [`CorrectionBank`]
/// of raw per-merged-cell deltas ready to attach to `model`.
///
/// # Errors
/// [`PbError::InvalidConfig`] if `spec.max_fit_rows == 0`; propagates scaffold / shape failures.
pub fn fit_cell_correction(
    model: &Model,
    data: &[Vec<u8>],
    residual: &[f64],
    sample_weight: &[f64],
    supports: &[Vec<u32>],
    spec: &CellRefitSpec,
) -> Result<CorrectionBank, PbError> {
    // `CellRefitSpec` is `pub` with `pub` fields (re-exported at the crate root), so a direct
    // core-crate caller can construct `max_fit_rows: 0` — 0 has no "unlimited" meaning here (it
    // would underflow the `cap - 1` subsample cap below), so reject it loudly instead of either
    // panicking or silently clamping to a surprising row count.
    if spec.max_fit_rows == 0 {
        return Err(PbError::InvalidConfig {
            what: "CellRefitSpec::max_fit_rows must be >= 1 (0 is not an 'unlimited' sentinel; \
                   omit the cell-refit correction entirely to skip it)"
                .into(),
        });
    }
    let n_rows = residual.len();
    if sample_weight.len() != n_rows {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "cell-refit sample_weight len {} != residual len {n_rows}",
                sample_weight.len()
            ),
        });
    }
    let mut bank = correction_scaffold(model, supports)?;
    if bank.tables.is_empty() {
        return Ok(bank);
    }

    // Fit only nonzero-weight rows. Held-out/uncovered rows carry weight 0 and contribute
    // EXACTLY 0 to `b` and `col_w`, so compacting them out is lossless (same δ, cheaper).
    let mut rows: Vec<usize> = (0..n_rows).filter(|&r| sample_weight[r] > 0.0).collect();
    if rows.is_empty() {
        return Ok(bank);
    }
    // Bound the fit rows for very large datasets (see [`CellRefitSpec::max_fit_rows`]). Keep the
    // `cap` rows with the smallest bijective scramble of their row id — a HASH-based subsample
    // (a strided pick would bias on sorted/periodic row order, e.g. date-sorted data), uncorrelated
    // with row structure and tie-free (splitmix64 is a bijection). Re-sort by row id so the solve
    // keeps ascending/sequential access. Pure function of the row set ⇒ byte-deterministic. Below
    // the cap — every ordinary dataset — `rows` is untouched, so the fit is byte-identical.
    if rows.len() > spec.max_fit_rows {
        let cap = spec.max_fit_rows;
        let mut keyed: Vec<(u64, usize)> =
            rows.iter().map(|&r| (splitmix64(r as u64), r)).collect();
        keyed.select_nth_unstable_by_key(cap - 1, |&(h, _)| h);
        keyed.truncate(cap);
        rows = keyed.into_iter().map(|(_, r)| r).collect();
        rows.sort_unstable();
    }
    let w_c: Vec<f64> = rows.iter().map(|&r| sample_weight[r]).collect();
    let r_c: Vec<f64> = rows.iter().map(|&r| residual[r]).collect();

    let design = build_design(&bank, data, &rows, spec.pair_cell_cap)?;
    let n_cols = design.n_cols;

    // col_w = Xᵀ W (the per-support diagonal block), shared by both backfit solves.
    let mut col_w = vec![0.0_f64; n_cols];
    design.rmatvec(&w_c, &mut col_w);

    // Cheap flat ridge init (λ=base): only its per-term RMS magnitudes feed the reweighting.
    let flat = vec![spec.base; n_cols];
    let delta0 = solve_ridge_backfit(
        &design,
        &w_c,
        &r_c,
        &col_w,
        &flat,
        None,
        spec.flat_cg_iters,
        spec.cg_tol,
    );

    // Per-term signal s_term = sqrt(mean δ0² over the term's columns); adaptive weight
    // w_term = s_term^γ normalised to mean 1; per-column penalty λ_col = base / w_term².
    let n_terms = bank.tables.len();
    let mut sumsq = vec![0.0_f64; n_terms];
    let mut counts = vec![0.0_f64; n_terms];
    for c in 0..n_cols {
        let t = design.col_term[c] as usize;
        sumsq[t] += delta0[c] * delta0[c];
        counts[t] += 1.0;
    }
    let mut w_term: Vec<f64> = (0..n_terms)
        .map(|t| {
            let s = if counts[t] > 0.0 {
                (sumsq[t] / counts[t]).sqrt()
            } else {
                0.0
            };
            s.powf(spec.gamma)
        })
        .collect();
    let mean_w = w_term.iter().sum::<f64>() / n_terms.max(1) as f64;
    if mean_w > 0.0 {
        for w in &mut w_term {
            *w /= mean_w;
        }
    } else {
        for w in &mut w_term {
            *w = 1.0;
        }
    }
    // Guard tiny weights so λ stays finite; floor relative to the mean (=1).
    const W_FLOOR: f64 = 1e-3;
    let lambda: Vec<f64> = (0..n_cols)
        .map(|c| {
            let wt = w_term[design.col_term[c] as usize].max(W_FLOOR);
            spec.base / (wt * wt)
        })
        .collect();

    let delta = if spec.gamma == 0.0 {
        delta0
    } else {
        // Warm-start the accurate adaptive backfit from the flat solution (same X,W,r; only λ
        // changed), so it starts near the answer and needs far fewer sweeps.
        solve_ridge_backfit(
            &design,
            &w_c,
            &r_c,
            &col_w,
            &lambda,
            Some(&delta0),
            spec.cg_iters,
            spec.cg_tol,
        )
    };

    // Expand the solved COARSE δ to the bank's merged-resolution values: each merged cell
    // takes its coarse group's δ (constant within coarse groups ⊇ merged cells, so G0 holds).
    for (t, table) in bank.tables.iter_mut().enumerate() {
        let coarse_off = design.col_offset[t] as usize;
        let ks = &design.coarse_shape[t];
        let grps = &design.groups[t];
        let merged_shape: Vec<usize> = table.shape.iter().map(|&s| s as usize).collect();
        let order = merged_shape.len();
        let mut cells = vec![0usize; order];
        for merged_flat in 0..table.values.len() {
            // Decode merged_flat (row-major, last axis fastest) into per-axis merged cells.
            let mut rem = merged_flat;
            for k in (0..order).rev() {
                let e = merged_shape[k];
                cells[k] = rem % e;
                rem /= e;
            }
            // Map each merged cell to its coarse group, refold to the coarse column.
            let mut coarse_flat = 0usize;
            for k in 0..order {
                let g = grps[k][cells[k]] as usize;
                coarse_flat = coarse_flat * ks[k] as usize + g;
            }
            table.values[merged_flat] = delta[coarse_off + coarse_flat];
        }
    }
    Ok(bank)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing, clippy::float_cmp)]
    use super::*;

    // Build a tiny 2-feature model via the explain fixture so we have real grids/provenance.
    fn small_model() -> Model {
        crate::explain::fixture_model()
    }

    #[test]
    fn cg_recovers_a_known_main_effect() {
        // Single main on axis 0; residual is a pure step function of axis-0 bin → the ridge
        // (light penalty) should recover a delta that reduces the residual sharply.
        let model = small_model();
        let n = 200usize;
        // axis 0 cycles bins {1,2}; axis 1 bin 1 (irrelevant here).
        let a0: Vec<u8> = (0..n).map(|i| if i % 2 == 0 { 1 } else { 2 }).collect();
        let a1: Vec<u8> = vec![1u8; n];
        let data = vec![a0.clone(), a1];
        // Target residual: +3 on bin 1, -3 on bin 2.
        let resid: Vec<f64> = a0
            .iter()
            .map(|&b| if b == 1 { 3.0 } else { -3.0 })
            .collect();
        let w = vec![1.0_f64; n];
        let spec = CellRefitSpec {
            base: 1.0,
            gamma: 0.0,
            pair_cell_cap: 24,
            flat_cg_iters: 200,
            cg_iters: 200,
            cg_tol: 1e-12,
            max_fit_rows: 1_000_000,
        };
        let bank = fit_cell_correction(&model, &data, &resid, &w, &[vec![0]], &spec).unwrap();
        // Predict the correction at bins 1 and 2 and check it reduces the residual a lot.
        let rows: Vec<usize> = (0..n).collect();
        let design = build_design(&bank, &data, &rows, 24).unwrap();
        let delta_flat: Vec<f64> = bank.tables[0].values.clone();
        let mut pred = vec![0.0_f64; n];
        design.matvec(&delta_flat, &mut pred);
        let before: f64 = resid.iter().map(|r| r * r).sum();
        let after: f64 = resid.iter().zip(&pred).map(|(r, p)| (r - p).powi(2)).sum();
        assert!(
            after < 0.05 * before,
            "residual not reduced: before {before} after {after}"
        );
    }

    #[test]
    fn adaptive_gamma_shrinks_a_zero_signal_term_more() {
        // Two mains: axis 0 carries signal, axis 1 carries none. With γ>0 the no-signal
        // term's coefficients should be shrunk much closer to zero than under a flat ridge.
        let model = small_model();
        let n = 400usize;
        let a0: Vec<u8> = (0..n).map(|i| if i % 2 == 0 { 1 } else { 2 }).collect();
        let a1: Vec<u8> = (0..n)
            .map(|i| if (i / 2) % 2 == 0 { 1 } else { 2 })
            .collect();
        let data = vec![a0.clone(), a1.clone()];
        // axis 0 carries a STRONG signal, axis 1 a WEAK one — both nonzero so adaptive γ has
        // something to differentially shrink (a perfectly-zero term can't be "shrunk more").
        let resid: Vec<f64> = (0..n)
            .map(|i| {
                let s0 = if a0[i] == 1 { 2.0 } else { -2.0 };
                let s1 = if a1[i] == 1 { 0.4 } else { -0.4 };
                s0 + s1
            })
            .collect();
        let w = vec![1.0_f64; n];
        let supports = vec![vec![0u32], vec![1u32]];

        let flat_spec = CellRefitSpec {
            base: 50.0,
            gamma: 0.0,
            pair_cell_cap: 24,
            flat_cg_iters: 300,
            cg_iters: 300,
            cg_tol: 1e-12,
            max_fit_rows: 1_000_000,
        };
        let adapt_spec = CellRefitSpec {
            gamma: 2.0,
            ..flat_spec
        };
        let flat = fit_cell_correction(&model, &data, &resid, &w, &supports, &flat_spec).unwrap();
        let adapt = fit_cell_correction(&model, &data, &resid, &w, &supports, &adapt_spec).unwrap();

        let norm = |bank: &CorrectionBank, term: usize| -> f64 {
            bank.tables[term]
                .values
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
        };
        // axis-1 (no signal) shrinks relative to axis-0 more under adaptive than flat.
        let flat_ratio = norm(&flat, 1) / norm(&flat, 0).max(1e-12);
        let adapt_ratio = norm(&adapt, 1) / norm(&adapt, 0).max(1e-12);
        assert!(
            adapt_ratio < flat_ratio,
            "adaptive should shrink the zero-signal term more: flat {flat_ratio} adapt {adapt_ratio}"
        );
    }

    #[test]
    fn max_fit_rows_zero_is_a_typed_error_not_a_panic() {
        // `CellRefitSpec` is `pub` with `pub` fields, so a direct core-crate caller can
        // construct `max_fit_rows: 0`. Before the fix this underflowed `cap - 1` inside
        // the subsample-cap branch and panicked; it must now return a typed
        // `InvalidConfig` error instead.
        let model = small_model();
        let n = 4usize;
        let a0: Vec<u8> = vec![1, 2, 1, 2];
        let data = vec![a0];
        let resid = vec![1.0, -1.0, 1.0, -1.0];
        let w = vec![1.0_f64; n];
        let spec = CellRefitSpec {
            base: 1.0,
            gamma: 0.0,
            pair_cell_cap: 24,
            flat_cg_iters: 10,
            cg_iters: 10,
            cg_tol: 1e-12,
            max_fit_rows: 0,
        };
        let err = fit_cell_correction(&model, &data, &resid, &w, &[vec![0]], &spec).unwrap_err();
        assert!(
            matches!(err, PbError::InvalidConfig { .. }),
            "expected InvalidConfig, got {err:?}"
        );
    }

    #[test]
    fn fit_row_cap_recovers_the_effect_and_stays_deterministic() {
        // The fit-row cap subsamples the fit rows for very large datasets. Verify it (a) still
        // recovers the effect (the correction is a coarse over-determined surface, so it barely needs
        // the extra rows) and (b) is byte-deterministic (the strided subsample is a pure function of
        // the row set — no RNG).
        let model = small_model();
        let n = 2000usize;
        let a0: Vec<u8> = (0..n).map(|i| if i % 2 == 0 { 1 } else { 2 }).collect();
        let a1 = vec![1u8; n];
        let data = vec![a0.clone(), a1];
        let resid: Vec<f64> = a0
            .iter()
            .map(|&b| if b == 1 { 3.0 } else { -3.0 })
            .collect();
        let w = vec![1.0_f64; n];
        let spec = CellRefitSpec {
            base: 1.0,
            gamma: 0.0,
            pair_cell_cap: 24,
            flat_cg_iters: 200,
            cg_iters: 200,
            cg_tol: 1e-12,
            max_fit_rows: 500, // << n, so the strided subsample fires
        };
        let bank1 = fit_cell_correction(&model, &data, &resid, &w, &[vec![0]], &spec).unwrap();
        let bank2 = fit_cell_correction(&model, &data, &resid, &w, &[vec![0]], &spec).unwrap();
        // Deterministic: identical banks bit-for-bit across runs.
        assert_eq!(bank1.tables.len(), bank2.tables.len());
        for (t1, t2) in bank1.tables.iter().zip(&bank2.tables) {
            for (v1, v2) in t1.values.iter().zip(&t2.values) {
                assert_eq!(
                    v1.to_bits(),
                    v2.to_bits(),
                    "capped fit is not deterministic"
                );
            }
        }
        // Still recovers the ±3 step (a 500-row subsample fits the 2-cell main near-perfectly).
        let rows: Vec<usize> = (0..n).collect();
        let design = build_design(&bank1, &data, &rows, 24).unwrap();
        let delta_flat: Vec<f64> = bank1.tables[0].values.clone();
        let mut pred = vec![0.0_f64; n];
        design.matvec(&delta_flat, &mut pred);
        let before: f64 = resid.iter().map(|r| r * r).sum();
        let after: f64 = resid.iter().zip(&pred).map(|(r, p)| (r - p).powi(2)).sum();
        assert!(
            after < 0.1 * before,
            "capped fit did not recover the effect: before {before} after {after}"
        );
    }

    /// The pre-2026-09-27 backfit (two passes per support), the reference the fused pass is
    /// pinned to bit-for-bit.
    #[allow(clippy::too_many_arguments)]
    fn backfit_two_pass(
        design: &Design,
        w: &[f64],
        r_target: &[f64],
        col_w: &[f64],
        lambda: &[f64],
        x0: Option<&[f64]>,
        max_sweeps: usize,
        tol: f64,
    ) -> Vec<f64> {
        let n_rows = w.len();
        let mut delta = match x0 {
            Some(x0) => x0.to_vec(),
            None => vec![0.0_f64; design.n_cols],
        };
        let mut res = r_target.to_vec();
        if x0.is_some() {
            let mut f = vec![0.0_f64; n_rows];
            design.matvec(&delta, &mut f);
            for i in 0..n_rows {
                res[i] -= f[i];
            }
        }
        for _sweep in 0..max_sweeps {
            let mut max_change = 0.0_f64;
            let mut max_coef = 1e-30_f64;
            for (u, active_u) in design.active.iter().enumerate() {
                let off = design.col_offset[u] as usize;
                let ncell = design.term_cols[u] as usize;
                let mut s = vec![0.0_f64; ncell];
                for (i, &c) in active_u.iter().enumerate() {
                    s[c as usize] += w[i] * res[i];
                }
                let mut d = vec![0.0_f64; ncell];
                for (cc, dcc) in d.iter_mut().enumerate() {
                    let c = off + cc;
                    let old = delta[c];
                    let cw = col_w[c];
                    let den = cw + lambda[c];
                    let new = if den > 0.0 {
                        (s[cc] + old * cw) / den
                    } else {
                        0.0
                    };
                    *dcc = new - old;
                    delta[c] = new;
                    max_change = max_change.max(dcc.abs());
                    max_coef = max_coef.max(new.abs());
                }
                for (i, &c) in active_u.iter().enumerate() {
                    res[i] -= d[c as usize];
                }
            }
            if max_change <= tol * max_coef {
                break;
            }
        }
        delta
    }

    #[test]
    fn fused_backfit_is_bit_identical_to_the_two_pass_sweep() {
        let mut state = 0x1234_5678_9abc_def1_u64;
        let mut next = || {
            state = splitmix64(state);
            state
        };
        let n = 3000usize;
        let term_cols: Vec<u32> = vec![1, 7, 24, 3, 16, 2, 9];
        let mut col_offset = Vec::new();
        let mut col_term = Vec::new();
        let mut off = 0u32;
        for (u, &k) in term_cols.iter().enumerate() {
            col_offset.push(off);
            for _ in 0..k {
                col_term.push(u as u32);
            }
            off += k;
        }
        let active: Vec<Vec<u16>> = term_cols
            .iter()
            .map(|&k| (0..n).map(|_| (next() % u64::from(k)) as u16).collect())
            .collect();
        let design = Design {
            active,
            n_cols: off as usize,
            col_term,
            col_offset,
            term_cols: term_cols.clone(),
            coarse_shape: term_cols.iter().map(|&k| vec![k]).collect(),
            groups: term_cols.iter().map(|&k| vec![(0..k).collect()]).collect(),
        };
        let unit = |x: u64| (x >> 11) as f64 / (1u64 << 53) as f64;
        let w: Vec<f64> = (0..n).map(|_| 0.1 + unit(next())).collect();
        let r: Vec<f64> = (0..n).map(|_| unit(next()) * 4.0 - 2.0).collect();
        let mut col_w = vec![0.0_f64; design.n_cols];
        design.rmatvec(&w, &mut col_w);
        let lambda: Vec<f64> = (0..design.n_cols).map(|_| 0.5 + unit(next())).collect();
        let x0: Vec<f64> = (0..design.n_cols).map(|_| unit(next()) - 0.5).collect();
        for (sweeps, tol) in [(1, 0.0), (3, 0.0), (30, 1e-4), (100, 1e-12)] {
            for start in [None, Some(x0.as_slice())] {
                let fused =
                    solve_ridge_backfit(&design, &w, &r, &col_w, &lambda, start, sweeps, tol);
                let two = backfit_two_pass(&design, &w, &r, &col_w, &lambda, start, sweeps, tol);
                let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&fused), bits(&two), "sweeps={sweeps} tol={tol}");
            }
        }
    }
}
