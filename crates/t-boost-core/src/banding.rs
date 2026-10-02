//! Banding: condense every interaction table of a deployed bank into a small product grid of
//! bands, with the change to predictions held to a stated fraction of the model's own noise.
//!
//! Pipeline (all on TRAINING ROWS — the fine merged-grid cube is never built):
//!
//! 1. **Ordering.** Every axis is ordinal. Numeric and single-channel categorical axes are
//!    already ordered by their encoded value; a per-level (joint) categorical axis is put in order
//!    by the model's mean training score per level, so contiguous bands are groups of similar
//!    levels. The missing cell is always its own band.
//! 2. **Master cut lists** per feature: agglomerative merging of adjacent cells, cheapest merge
//!    first, where the cost of merging two adjacent groups is the curvature-weighted variance it
//!    destroys, summed over every interaction table containing the feature (so every table cuts
//!    the feature at the same, nested, places). Weights: data curvature per row plus a floor
//!    carried by pseudo-rows drawn from the purification measure.
//! 3. **Resolution** per table and per axis: a greedy that repeatedly buys the most fidelity per
//!    added cell, until the cross-fitted curvature-weighted mean squared move (bands and values
//!    from half A of a row subsample, move measured on half B, and vice versa) is within
//!    `(tolerance · σ)²`, σ = the soup's bag noise. A k-way table's axis is never finer than the
//!    same axis of a (k-1)-way table it contains.
//! 4. **Values**: data-weighted block means (curvature plus the measure floor, the floor part in
//!    closed form), then exact re-purification (marginals pushed down, highest order first, the
//!    lower-order table's bands refined where a pushed piece needs it).
//! 5. **Distil**: mains and pairs refit (Gauss-Seidel, ridge-anchored to the block means) to the
//!    UNBANDED model's training scores — never to y — then re-purified.
//!
//! Measured (arena, 31 datasets, tolerance 0.75): 3-way cells 274M -> 84k, largest 3-way 10.8k,
//! pooled test deviance +0.015%. See `target/banding/r5_prune/PIPELINE.md`.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap as StdHashMap};
use std::hash::BuildHasherDefault;

/// Deterministic-order hash map: the floating-point sums below iterate these, and banding must
/// be bit-reproducible across processes (spec §13.4), which a per-process random hasher breaks.
type HashMap<K, V> = StdHashMap<K, V, BuildHasherDefault<DefaultHasher>>;

use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::Serialize;

use crate::error::PbError;
use crate::explain::{bank_axis_measure_with, AxisId, EffectTable, FeatureSet, TableBank, Tensor};

/// Cut counts per axis the greedy steps through (`usize::MAX` = full merged resolution).
const SCHEDULE: [usize; 15] = [
    1,
    2,
    3,
    5,
    7,
    11,
    15,
    23,
    31,
    47,
    63,
    95,
    127,
    191,
    usize::MAX,
];

/// Banding knobs.
#[derive(Debug, Clone)]
pub struct BandingConfig {
    /// Fidelity tolerance `c`: cross-fitted RMS move <= `c · σ`.
    pub tolerance: f64,
    /// Row subsample the fidelity greedy is scored on.
    pub max_select_rows: usize,
    /// Measure floor: share of the curvature mass spread over the purification measure.
    pub floor: f64,
    /// Pseudo-rows carrying the floor in the selection stages.
    pub pseudo_rows: usize,
    /// Distil sweeps (mains and pairs refit to the unbanded scores).
    pub distill_sweeps: usize,
    /// Distil ridge strength toward the block means (in units of the floor mass).
    pub distill_lambda: f64,
    /// Seed for the row subsample and the pseudo-rows.
    pub seed: u64,
}

impl Default for BandingConfig {
    fn default() -> Self {
        Self {
            tolerance: 0.75,
            max_select_rows: 60_000,
            floor: 0.01,
            pseudo_rows: 20_000,
            distill_sweeps: 40,
            distill_lambda: 1.0,
            seed: 0,
        }
    }
}

/// Per-row training inputs to [`band_bank`].
pub struct BandingRows<'a> {
    /// `cells[raw][row]`: merged-grid cell of every training row on every raw feature.
    pub cells: &'a [Vec<u32>],
    /// Loss curvature per row at the deployed score (the fidelity and value weights).
    pub h: &'a [f64],
    /// Effective mass per row (`w · exposure`), for the banded tables' `support`.
    pub mass: &'a [f64],
    /// The unbanded bank's raw score per row (the distil target).
    pub target: &'a [f64],
    /// Noise scale σ (curvature-weighted RMS of the soup's per-row bag standard error).
    pub sigma: f64,
    /// Cap on the curvature-weighted mean squared move (`Σ h δ² / Σ h`) from the deviance side:
    /// `deviance_cap · D̄ · Σ w / Σ h`, since the expected deviance increase of a move δ is
    /// `Σ h δ² / Σ w` when `h` is the NLL curvature and the deviance is twice the NLL.
    /// `f64::INFINITY` = no cap.
    pub mse_cap: f64,
}

/// What banding did, serialized into `pruning_report_["banding"]`.
#[derive(Debug, Clone, Serialize)]
pub struct BandingReport {
    /// Fidelity tolerance used.
    pub tolerance: f64,
    /// The binding budget: the smaller of `noise_budget` and `deviance_budget`.
    pub budget: f64,
    /// `(tolerance · σ)²` — the move allowed relative to the model's own noise.
    pub noise_budget: f64,
    /// The move whose expected deviance cost is `deviance_cap` of the model's own deviance.
    pub deviance_budget: f64,
    /// Cross-fitted MSE of the selected banding (per-table sum, what the greedy allocates with).
    pub selected_mse: f64,
    /// Cross-fitted MSE of the whole banding (tables' moves summed per row) — held to `budget`.
    pub combined_mse: f64,
    /// The Lagrange multiplier (move per added cell) the calibration settled on.
    pub allocation_budget: f64,
    /// Combined-move evaluations the calibration took.
    pub calibration_rounds: u32,
    /// Interaction tables banded.
    pub tables: u32,
    /// Cells per order before banding (dense merged-grid equivalent).
    pub cells_before: BTreeMap<u8, u64>,
    /// Cells per order after banding.
    pub cells_after: BTreeMap<u8, u64>,
    /// Largest banded table.
    pub largest_after: u64,
    /// Raw features whose per-level categorical axis was reordered before banding.
    pub reordered: Vec<u32>,
    /// Why the bank was left unbanded, when it was.
    pub skipped: Option<String>,
}

// ------------------------------------------------------------------------------------------
// Checked access (the no-panic deny set: every out-of-range index is a typed internal error).

fn oob(what: &'static str) -> PbError {
    PbError::Internal {
        what: format!("banding: {what} index out of range"),
    }
}

#[inline]
fn ix<T: Copy>(s: &[T], i: usize, what: &'static str) -> Result<T, PbError> {
    s.get(i).copied().ok_or_else(|| oob(what))
}

#[inline]
fn ix_ref<'a, T>(s: &'a [T], i: usize, what: &'static str) -> Result<&'a T, PbError> {
    s.get(i).ok_or_else(|| oob(what))
}

#[inline]
fn ix_mut<'a, T>(s: &'a mut [T], i: usize, what: &'static str) -> Result<&'a mut T, PbError> {
    s.get_mut(i).ok_or_else(|| oob(what))
}

/// Merged cell of row `i` on raw feature `r` of a `[raw][row]` column store.
#[inline]
fn cell_at(cells: &[Vec<u32>], r: usize, i: usize) -> Result<usize, PbError> {
    cells
        .get(r)
        .and_then(|c| c.get(i))
        .map(|&c| c as usize)
        .ok_or_else(|| oob("row cell"))
}

/// Cut count of schedule state `s`.
#[inline]
fn sched(s: u8) -> Result<usize, PbError> {
    ix(&SCHEDULE, s as usize, "schedule state")
}

/// Advance a row-major odometer `coord` over the extents `ext` (last axis fastest).
#[inline]
fn step(coord: &mut [usize], ext: &[usize]) {
    for (c, &e) in coord.iter_mut().zip(ext).rev() {
        *c += 1;
        if *c < e {
            return;
        }
        *c = 0;
    }
}

// ------------------------------------------------------------------------------------------
// Interaction tables as row-evaluable sources.

enum Src<'a> {
    Dense(&'a EffectTable),
    Boxes(Vec<(&'a [f64], &'a [Vec<bool>])>),
}

struct Inter<'a> {
    u: FeatureSet,
    raws: Vec<usize>,
    n: Vec<usize>,
    axes: Vec<AxisId>,
    src: Src<'a>,
}

impl Inter<'_> {
    fn order(&self) -> usize {
        self.raws.len()
    }

    /// The table at one merged-cell tuple.
    fn at(&self, cell: &[usize]) -> Result<f64, PbError> {
        match &self.src {
            Src::Dense(t) => Ok(t.values.at(cell).unwrap_or(0.0)),
            Src::Boxes(boxes) => {
                let mut acc = 0.0;
                for (p, low) in boxes {
                    let mut idx = 0usize;
                    for (d, (&c, lowd)) in cell.iter().zip(low.iter()).enumerate() {
                        if ix(lowd, c, "box mask")? {
                            idx |= 1 << d;
                        }
                    }
                    acc += ix(p, idx, "box corner")?;
                }
                Ok(acc)
            }
        }
    }

    /// Σ over each band-grid cell of P(c)·f(c), P = the product of `m` (normalized per axis).
    fn prior_band_sums(
        &self,
        band: &[Vec<u32>],
        nb: &[usize],
        m: &[Vec<f64>],
    ) -> Result<Vec<f64>, PbError> {
        let total: usize = nb.iter().product();
        let mut out = vec![0.0; total];
        match &self.src {
            Src::Dense(t) => {
                let vals = t.values.values();
                let k = self.order();
                let mut coord = vec![0usize; k];
                for v in vals.iter() {
                    let mut p = 1.0;
                    let mut flat = 0usize;
                    for (((&c, md), bd), &nbd) in coord.iter().zip(m).zip(band).zip(nb) {
                        p *= ix(md, c, "prior measure")?;
                        flat = flat * nbd + ix(bd, c, "prior band")? as usize;
                    }
                    *ix_mut(&mut out, flat, "prior block")? += p * v;
                    step(&mut coord, &self.n);
                }
            }
            Src::Boxes(boxes) => {
                let k = self.order();
                for (p, low) in boxes {
                    // per axis: measure mass of each band on the low / high side of this box
                    let mut side: Vec<[Vec<f64>; 2]> = Vec::with_capacity(k);
                    for ((((lowd, bd), md), &nbd), &nd) in
                        low.iter().zip(band).zip(m).zip(nb).zip(&self.n)
                    {
                        let mut lo = vec![0.0; nbd];
                        let mut hi = vec![0.0; nbd];
                        for c in 0..nd {
                            let b = ix(bd, c, "prior band")? as usize;
                            let mc = ix(md, c, "prior measure")?;
                            if ix(lowd, c, "box mask")? {
                                *ix_mut(&mut lo, b, "prior band mass")? += mc;
                            } else {
                                *ix_mut(&mut hi, b, "prior band mass")? += mc;
                            }
                        }
                        side.push([hi, lo]);
                    }
                    for (corner, &pv) in p.iter().enumerate() {
                        if pv == 0.0 {
                            continue;
                        }
                        let mut coord = vec![0usize; k];
                        for slot in out.iter_mut() {
                            let mut w = pv;
                            for (d, (sd, &c)) in side.iter().zip(&coord).enumerate() {
                                let half = ix_ref(sd, (corner >> d) & 1, "box side")?;
                                w *= ix(half, c, "box side mass")?;
                            }
                            *slot += w;
                            step(&mut coord, nb);
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

// ------------------------------------------------------------------------------------------
// Rows (data + pseudo) as a column store over raw features.

struct RowSet {
    cells: Vec<Vec<u32>>, // [raw][row]
    w: Vec<f64>,          // weight per row (curvature, or the floor's share for pseudo-rows)
}

impl RowSet {
    fn len(&self) -> usize {
        self.w.len()
    }
}

fn eval_rows(t: &Inter<'_>, rows: &RowSet) -> Result<Vec<f32>, PbError> {
    let mut cell = vec![0usize; t.order()];
    (0..rows.len())
        .map(|i| {
            for (slot, &r) in cell.iter_mut().zip(&t.raws) {
                *slot = cell_at(&rows.cells, r, i)?;
            }
            Ok(t.at(&cell)? as f32)
        })
        .collect()
}

// ------------------------------------------------------------------------------------------
// Master cut lists.

/// One master position's merge statistics as a KEY-SORTED list of `(key, Σw, Σw·g)`, with
/// `key = (table << 40) | cross cell` (the cell of every other axis of the table, mixed radix).
type Groups = Vec<(u64, f64, f64)>;

/// Cross-cell spaces up to this size accumulate in a dense buffer; larger ones sort instead.
const DENSE_CROSS_MAX: u64 = 1 << 20;

/// A merge whose smaller side is this many times smaller than the other looks its keys up by
/// binary search instead of walking both lists.
const LOOKUP_RATIO: usize = 16;

/// Cut positions (boundary before position `p`, `p in 2..n`) for one raw feature, most important
/// first. Position 0 is the missing cell (never merged). Rows: the selection rows passing `keep`
/// (curvature weights) plus every pseudo-row (weight `pse_w`).
///
/// Bottom-up: every position starts as its own group and the adjacent pair whose merge adds the
/// least within-cell squared error — summed over every table on this raw and every cell of the
/// table's other axes — merges first; the boundaries removed, last first, are the ranking.
#[allow(clippy::too_many_arguments)]
fn master_cuts(
    r: usize,
    n: usize,
    pos: &[u32],
    tabs: &[(usize, &Inter<'_>)],
    g_sel: &[Vec<f32>],
    g_pse: &[Vec<f32>],
    sel: &RowSet,
    pse: &RowSet,
    keep: &(dyn Fn(usize) -> bool + Sync),
    pse_w: f64,
) -> Result<Vec<u32>, PbError> {
    if n <= 2 {
        return Ok(Vec::new());
    }
    let groups = master_groups(r, n, pos, tabs, g_sel, g_pse, sel, pse, keep, pse_w)?;
    merge_order(groups)
}

/// Every position's [`Groups`]. Each `(Σw, Σw·g)` adds its rows in arrival order — selection
/// rows (ascending, passing `keep`), then pseudo-rows — starting from `0.0`, the order a per-row
/// running sum gives. Tables are taken in `tabs` order and each position's list is sorted by key.
#[allow(clippy::too_many_arguments)]
fn master_groups(
    r: usize,
    n: usize,
    pos: &[u32],
    tabs: &[(usize, &Inter<'_>)],
    g_sel: &[Vec<f32>],
    g_pse: &[Vec<f32>],
    sel: &RowSet,
    pse: &RowSet,
    keep: &(dyn Fn(usize) -> bool + Sync),
    pse_w: f64,
) -> Result<Vec<Groups>, PbError> {
    // The rows' positions along `r` are the same for every table: bucket the contributing rows
    // by position once (a stable counting sort), keeping arrival order inside each bucket. A row
    // ref below `n_sel` is a selection row, the rest are pseudo-rows.
    let n_sel = sel.len();
    let mut arrivals: Vec<(usize, u32)> = Vec::with_capacity(n_sel + pse.len());
    let as_ref = |i: usize| u32::try_from(i).map_err(|_| oob("master row"));
    for i in 0..n_sel {
        if keep(i) {
            let q = ix(pos, cell_at(&sel.cells, r, i)?, "master position")? as usize;
            if q != 0 {
                arrivals.push((q, as_ref(i)?));
            }
        }
    }
    for j in 0..pse.len() {
        let q = ix(pos, cell_at(&pse.cells, r, j)?, "master position")? as usize;
        if q != 0 {
            arrivals.push((q, as_ref(n_sel + j)?));
        }
    }
    let mut start = vec![0_usize; n + 1];
    for &(q, _) in &arrivals {
        if q >= n {
            return Err(oob("master group"));
        }
        *ix_mut(&mut start, q + 1, "master bucket")? += 1;
    }
    for q in 1..=n {
        let prev = ix(&start, q - 1, "master bucket")?;
        *ix_mut(&mut start, q, "master bucket")? += prev;
    }
    let mut fill = start.clone();
    let mut order = vec![0_u32; arrivals.len()];
    for &(q, row) in &arrivals {
        let at = ix_mut(&mut fill, q, "master bucket")?;
        *ix_mut(&mut order, *at, "master order")? = row;
        *at += 1;
    }
    drop(arrivals);

    let mut groups: Vec<Groups> = vec![Vec::new(); n];
    let mut acc: Vec<(f64, f64)> = Vec::new();
    let mut seen: Vec<bool> = Vec::new();
    let mut touched: Vec<u32> = Vec::new();
    let mut sparse: Vec<(u64, f64, f64)> = Vec::new();
    let mut other: Vec<(usize, u64)> = Vec::new();
    for &(ti, t) in tabs {
        let d = t.raws.iter().position(|&x| x == r).unwrap_or(0);
        other.clear();
        other.extend(
            t.raws
                .iter()
                .zip(&t.n)
                .enumerate()
                .filter(|&(e, _)| e != d)
                .map(|(_, (&rr, &ne))| (rr, ne as u64)),
        );
        let n_cross = other
            .iter()
            .fold(1_u64, |acc, &(_, ne)| acc.saturating_mul(ne));
        let tag = (ti as u64) << 40;
        let gs = ix_ref(g_sel, ti, "selection values")?;
        let gp = ix_ref(g_pse, ti, "pseudo values")?;
        // (cross cell, weight, value) of one bucketed row
        let row_at = |row: u32| -> Result<(u64, f64, f32), PbError> {
            let row = row as usize;
            let (rows, i, w, g) = match row.checked_sub(n_sel) {
                None => (
                    sel,
                    row,
                    ix(&sel.w, row, "selection weight")?,
                    ix(gs, row, "selection value")?,
                ),
                Some(j) => (pse, j, pse_w, ix(gp, j, "pseudo value")?),
            };
            let mut key: u64 = 0;
            for &(rr, ne) in &other {
                key = key * ne + cell_at(&rows.cells, rr, i)? as u64;
            }
            Ok((key, w, g))
        };
        let dense = n_cross <= DENSE_CROSS_MAX;
        if dense {
            let len = usize::try_from(n_cross).map_err(|_| oob("cross cells"))?;
            if acc.len() < len {
                acc.resize(len, (0.0, 0.0));
                seen.resize(len, false);
            }
        }
        for q in 1..n {
            let lo = ix(&start, q, "master bucket")?;
            let hi = ix(&start, q + 1, "master bucket")?;
            let bucket = order.get(lo..hi).ok_or_else(|| oob("master bucket"))?;
            if bucket.is_empty() {
                continue;
            }
            let grp = ix_mut(&mut groups, q, "master group")?;
            if dense {
                for &row in bucket {
                    let (key, w, g) = row_at(row)?;
                    let k = key as usize;
                    let flag = ix_mut(&mut seen, k, "cross cell")?;
                    if !*flag {
                        *flag = true;
                        touched.push(u32::try_from(k).map_err(|_| oob("cross cell"))?);
                    }
                    let e = ix_mut(&mut acc, k, "cross cell")?;
                    e.0 += w;
                    e.1 += w * f64::from(g);
                }
                touched.sort_unstable();
                for &k in &touched {
                    let k = k as usize;
                    let e = ix_mut(&mut acc, k, "cross cell")?;
                    grp.push((tag | k as u64, e.0, e.1));
                    *e = (0.0, 0.0);
                    *ix_mut(&mut seen, k, "cross cell")? = false;
                }
                touched.clear();
            } else {
                sparse.clear();
                for &row in bucket {
                    let (key, w, g) = row_at(row)?;
                    sparse.push((key, w, w * f64::from(g)));
                }
                sparse.sort_by_key(|e| e.0); // stable: equal keys keep arrival order
                let mut it = sparse.iter().peekable();
                while let Some(&(key, w, wg)) = it.next() {
                    let (mut s0, mut s1) = (0.0 + w, 0.0 + wg);
                    while let Some(&(_, w2, wg2)) = it.next_if(|e| e.0 == key) {
                        s0 += w2;
                        s1 += wg2;
                    }
                    grp.push((tag | key, s0, s1));
                }
            }
        }
    }
    // Tables arrive in ascending index (so ascending key) from the caller; sort if not.
    for grp in &mut groups {
        if grp.iter().zip(grp.iter().skip(1)).any(|(a, b)| a.0 >= b.0) {
            grp.sort_by_key(|e| e.0);
        }
    }
    Ok(groups)
}

/// First index at or after `lo` in the key-sorted `g` whose key is `>= key` (`g.len()` if none).
fn key_at_or_after(g: &[(u64, f64, f64)], lo: usize, key: u64) -> usize {
    g.get(lo..)
        .map_or(g.len(), |rest| lo + rest.partition_point(|e| e.0 < key))
}

/// Within-cell squared-error increase of merging two positions' groups: over the keys present in
/// both, `x1²/x0 + y1²/y0 − (x1+y1)²/(x0+y0)` (the smaller group's entry is `x`), summed in key
/// order.
fn merge_cost(a: &Groups, b: &Groups) -> f64 {
    let (small, big) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let term = |x0: f64, x1: f64, y0: f64, y1: f64| {
        let t0 = x0 + y0;
        x1 * x1 / x0 + y1 * y1 / y0 - (x1 + y1) * (x1 + y1) / t0
    };
    let mut c = 0.0;
    if small.len().saturating_mul(LOOKUP_RATIO) < big.len() {
        let mut lo = 0;
        for &(k, x0, x1) in small {
            lo = key_at_or_after(big, lo, k);
            if let Some(&(bk, y0, y1)) = big.get(lo) {
                if bk == k {
                    if x0 > 0.0 && y0 > 0.0 {
                        c += term(x0, x1, y0, y1);
                    }
                    lo += 1;
                }
            }
        }
    } else {
        let (mut i, mut j) = (0, 0);
        while let (Some(&(ka, x0, x1)), Some(&(kb, y0, y1))) = (small.get(i), big.get(j)) {
            match ka.cmp(&kb) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    if x0 > 0.0 && y0 > 0.0 {
                        c += term(x0, x1, y0, y1);
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
    }
    c
}

/// Fold group `b` into group `a` (the left group of the merged pair): a key in both becomes
/// `(a0 + b0, a1 + b1)`, a key only in `b` enters as `(0.0 + b0, 0.0 + b1)` — the values a running
/// per-key sum gives. `scratch` is reused output space.
fn merge_groups(a: &mut Groups, b: Groups, scratch: &mut Groups) {
    // Fast path: every key of the smaller side already present in the larger one — update the
    // larger list in place (IEEE addition commutes, so either side may hold the sum).
    let (small, big_is_a) = if b.len() <= a.len() {
        (&b, true)
    } else {
        (&*a, false)
    };
    if small.len().saturating_mul(LOOKUP_RATIO) < if big_is_a { a.len() } else { b.len() } {
        let big: &Groups = if big_is_a { a } else { &b };
        let mut at = Vec::with_capacity(small.len());
        let mut lo = 0;
        let mut all = true;
        for &(k, _, _) in small {
            lo = key_at_or_after(big, lo, k);
            if big.get(lo).map(|e| e.0) == Some(k) {
                at.push(lo);
                lo += 1;
            } else {
                all = false;
                break;
            }
        }
        if all {
            if big_is_a {
                for (&p, &(_, y0, y1)) in at.iter().zip(&b) {
                    if let Some(e) = a.get_mut(p) {
                        e.1 += y0;
                        e.2 += y1;
                    }
                }
            } else {
                let mut b = b;
                for (&p, &(_, x0, x1)) in at.iter().zip(a.iter()) {
                    if let Some(e) = b.get_mut(p) {
                        e.1 += x0;
                        e.2 += x1;
                    }
                }
                *a = b;
            }
            return;
        }
    }
    scratch.clear();
    scratch.reserve(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    loop {
        match (a.get(i), b.get(j)) {
            (Some(&(ka, x0, x1)), Some(&(kb, y0, y1))) => match ka.cmp(&kb) {
                std::cmp::Ordering::Less => {
                    scratch.push((ka, x0, x1));
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    scratch.push((kb, 0.0 + y0, 0.0 + y1));
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    scratch.push((ka, x0 + y0, x1 + y1));
                    i += 1;
                    j += 1;
                }
            },
            (Some(&e), None) => {
                scratch.push(e);
                i += 1;
            }
            (None, Some(&(kb, y0, y1))) => {
                scratch.push((kb, 0.0 + y0, 0.0 + y1));
                j += 1;
            }
            (None, None) => break,
        }
    }
    std::mem::swap(a, scratch);
}

/// Greedy bottom-up merge of adjacent groups (positions `1..n`), cheapest [`merge_cost`] first
/// (the first of equal costs); returns the removed boundaries, most important first.
fn merge_order(mut groups: Vec<Groups>) -> Result<Vec<u32>, PbError> {
    let n = groups.len();
    let pair_cost = |groups: &[Groups], a: usize, b: usize| -> Result<f64, PbError> {
        Ok(merge_cost(
            ix_ref(groups, a, "master group")?,
            ix_ref(groups, b, "master group")?,
        ))
    };
    // alive groups are positions 1..n in order; group i's first position = start[i]
    let mut alive: Vec<usize> = (1..n).collect();
    let mut costs: Vec<f64> = alive
        .iter()
        .zip(alive.iter().skip(1))
        .map(|(&a, &b)| pair_cost(&groups, a, b))
        .collect::<Result<_, _>>()?;
    let mut removed: Vec<u32> = Vec::with_capacity(n);
    let mut scratch: Groups = Vec::new();
    while alive.len() > 1 {
        let (i, _) = costs
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        let a = ix(&alive, i, "alive group")?;
        let b = ix(&alive, i + 1, "alive group")?;
        let gb = std::mem::take(ix_mut(&mut groups, b, "master group")?);
        merge_groups(ix_mut(&mut groups, a, "master group")?, gb, &mut scratch);
        removed.push(b as u32); // the boundary before group b's first position
        alive.remove(i + 1);
        costs.remove(i);
        if i > 0 {
            let c = pair_cost(
                &groups,
                ix(&alive, i - 1, "alive group")?,
                ix(&alive, i, "alive group")?,
            )?;
            *ix_mut(&mut costs, i - 1, "merge cost")? = c;
        }
        if i < alive.len() - 1 {
            let c = pair_cost(
                &groups,
                ix(&alive, i, "alive group")?,
                ix(&alive, i + 1, "alive group")?,
            )?;
            *ix_mut(&mut costs, i, "merge cost")? = c;
        }
    }
    removed.reverse();
    Ok(removed)
}

/// Row-major band-grid index of the first `n` rows of `cells` (`[raw][row]`) under per-axis
/// labels `lab` (merged cell -> band) with band counts `nb`: `f = f * nb_d + lab_d[cell_d]`, axis
/// by axis — the same integer fold a per-row pass computes, run one column at a time.
fn flat_band_index(
    raws: &[usize],
    lab: &[Vec<u32>],
    nb: &[usize],
    cells: &[Vec<u32>],
    n: usize,
) -> Result<Vec<usize>, PbError> {
    let mut f = vec![0usize; n];
    for ((&r, labd), &nbd) in raws.iter().zip(lab).zip(nb) {
        let col = cells
            .get(r)
            .and_then(|c| c.get(..n))
            .ok_or_else(|| oob("row cell"))?;
        for (fi, &c) in f.iter_mut().zip(col) {
            let b = *labd.get(c as usize).ok_or_else(|| oob("band label"))?;
            *fi = *fi * nbd + b as usize;
        }
    }
    Ok(f)
}

/// Band per merged cell from the first `l` master cuts (all cuts => full resolution).
fn labels(cuts: &[u32], l: usize, pos: &[u32]) -> Result<(Vec<u32>, usize), PbError> {
    let n = pos.len();
    let mut chosen: Vec<u32> = cuts.iter().take(l.min(cuts.len())).copied().collect();
    chosen.sort_unstable();
    // band of position q: 0 for q = 0, else 1 + #chosen cuts <= q
    let mut band_of_pos = vec![0u32; n];
    let mut b = 1u32;
    let mut j = 0usize;
    for (q, slot) in band_of_pos.iter_mut().enumerate().skip(1) {
        while let Some(&cj) = chosen.get(j) {
            if cj as usize > q {
                break;
            }
            if cj as usize == q && q > 1 {
                b += 1;
            }
            j += 1;
        }
        *slot = b;
    }
    let nb = if n > 1 { b as usize + 1 } else { 1 };
    let lab = pos
        .iter()
        .map(|&q| ix(&band_of_pos, q as usize, "label position"))
        .collect::<Result<_, _>>()?;
    Ok((lab, nb))
}

// ------------------------------------------------------------------------------------------
// Fidelity of one banding (cross-fitted over two halves of the selection rows).

/// Reusable block accumulator for [`Fidelity::moves_with`]: per block `(Σw, Σw·g)` plus a
/// generation stamp, so a pass touches only the blocks its rows land in instead of allocating and
/// zeroing the whole band grid. A block untouched in the current generation reads as absent.
#[derive(Default)]
struct BlockScratch {
    sums: Vec<(f64, f64)>,
    stamp: Vec<u32>,
    /// A block's floor prior mean, once computed in this pass (see `prior_stamp`).
    prior: Vec<f64>,
    prior_stamp: Vec<u32>,
    gen: u32,
}

impl BlockScratch {
    /// Start a pass over a band grid of `total` blocks.
    fn begin(&mut self, total: usize) {
        if self.sums.len() < total {
            self.sums.resize(total, (0.0, 0.0));
            self.stamp.resize(total, 0);
            self.prior.resize(total, 0.0);
            self.prior_stamp.resize(total, 0);
        }
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.prior_stamp.iter_mut().for_each(|s| *s = 0);
            self.gen = 1;
        }
    }

    /// The prior mean cached for block `f` in this pass, if any.
    #[inline]
    fn cached_prior(&self, f: usize) -> Option<f64> {
        match (self.prior_stamp.get(f), self.prior.get(f)) {
            (Some(&st), Some(&v)) if st == self.gen => Some(v),
            _ => None,
        }
    }

    #[inline]
    fn cache_prior(&mut self, f: usize, v: f64) -> Result<(), PbError> {
        *self
            .prior_stamp
            .get_mut(f)
            .ok_or_else(|| oob("band prior"))? = self.gen;
        *self.prior.get_mut(f).ok_or_else(|| oob("band prior"))? = v;
        Ok(())
    }

    #[inline]
    fn add(&mut self, f: usize, w: f64, g: f64) -> Result<(), PbError> {
        let st = self.stamp.get_mut(f).ok_or_else(|| oob("band sum"))?;
        let e = self.sums.get_mut(f).ok_or_else(|| oob("band sum"))?;
        if *st != self.gen {
            *st = self.gen;
            *e = (0.0, 0.0);
        }
        e.0 += w;
        e.1 += w * g;
        Ok(())
    }

    #[inline]
    fn get(&self, f: usize) -> Option<(f64, f64)> {
        match (self.stamp.get(f), self.sums.get(f)) {
            (Some(&st), Some(&e)) if st == self.gen => Some(e),
            _ => None,
        }
    }
}

/// Per axis of one table under one labelling: each band's `[lo, hi)` position range and its
/// merged cells in ascending order ([`Fidelity::band_index`]).
struct BandIndex {
    ranges: Vec<Vec<(usize, usize)>>,
    cells: Vec<Vec<Vec<usize>>>,
}

struct Fidelity<'a> {
    tabs: &'a [Inter<'a>],
    g_sel: &'a [Vec<f32>],
    g_pse: &'a [Vec<f32>],
    sel: &'a RowSet,
    pse: &'a RowSet,
    half: &'a [u8],    // 0/1 per selection row
    lam_pse: [f64; 2], // floor weight per pseudo-row, per half's mass
    masters: &'a [Vec<Vec<u32>>; 2],
    pos: &'a [Vec<u32>],
    m: &'a [Vec<f64>],
    hsum: f64,
    /// Per table, lazily: per-box per-axis prefix sums of the measure on each side of the box's
    /// mask, in the axis's band (position) order — so a block's exact floor sum factorises.
    prefix: Vec<std::sync::OnceLock<Option<BoxPrefix>>>,
}

/// `side[b][d][s][q]` = Σ over the first `q` positions of axis `d` of `m(c)·[box b puts c on side
/// s]` (s = 1 is the LOW side); `pm[d][q]` = Σ of `m` over the first `q` positions.
struct BoxPrefix {
    side: Vec<Vec<[Vec<f64>; 2]>>,
    pm: Vec<Vec<f64>>,
}

impl Fidelity<'_> {
    fn box_prefix(&self, t: usize) -> Result<Option<&BoxPrefix>, PbError> {
        let tab = ix_ref(self.tabs, t, "table")?;
        let cell = ix_ref(&self.prefix, t, "prefix cache")?;
        if let Some(v) = cell.get() {
            return Ok(v.as_ref());
        }
        let built = match &tab.src {
            Src::Dense(_) => None,
            Src::Boxes(boxes) => {
                // merged cell at each position, per axis
                let mut order: Vec<Vec<usize>> = Vec::with_capacity(tab.order());
                for (&r, &nd) in tab.raws.iter().zip(&tab.n) {
                    let pos = ix_ref(self.pos, r, "axis order")?;
                    let mut at = vec![0usize; nd];
                    for (c, &q) in pos.iter().enumerate().take(nd) {
                        *ix_mut(&mut at, q as usize, "position")? = c;
                    }
                    order.push(at);
                }
                let mm: Vec<&Vec<f64>> = tab
                    .raws
                    .iter()
                    .map(|&r| ix_ref(self.m, r, "measure"))
                    .collect::<Result<_, _>>()?;
                let mut pm = Vec::with_capacity(tab.order());
                for (at, md) in order.iter().zip(&mm) {
                    let mut acc = vec![0.0; at.len() + 1];
                    for (q, &c) in at.iter().enumerate() {
                        *ix_mut(&mut acc, q + 1, "prefix")? =
                            ix(&acc, q, "prefix")? + ix(md, c, "measure")?;
                    }
                    pm.push(acc);
                }
                let mut side = Vec::with_capacity(boxes.len());
                for (_, low) in boxes {
                    let mut per = Vec::with_capacity(tab.order());
                    for ((at, md), lowd) in order.iter().zip(&mm).zip(low.iter()) {
                        let mut hi = vec![0.0; at.len() + 1];
                        let mut lo = vec![0.0; at.len() + 1];
                        for (q, &c) in at.iter().enumerate() {
                            let w = ix(md, c, "measure")?;
                            let is_low = ix(lowd, c, "box mask")?;
                            *ix_mut(&mut lo, q + 1, "prefix")? =
                                ix(&lo, q, "prefix")? + if is_low { w } else { 0.0 };
                            *ix_mut(&mut hi, q + 1, "prefix")? =
                                ix(&hi, q, "prefix")? + if is_low { 0.0 } else { w };
                        }
                        per.push([hi, lo]);
                    }
                    side.push(per);
                }
                Some(BoxPrefix { side, pm })
            }
        };
        let _ = cell.set(built);
        Ok(cell.get().and_then(Option::as_ref))
    }

    fn bands(
        &self,
        t: usize,
        state: &[u8],
        k: usize,
    ) -> Result<(Vec<Vec<u32>>, Vec<usize>), PbError> {
        let tab = ix_ref(self.tabs, t, "table")?;
        let master = ix_ref(self.masters.as_slice(), k, "half")?;
        let mut lab = Vec::with_capacity(tab.order());
        let mut nb = Vec::with_capacity(tab.order());
        for (&r, &s) in tab.raws.iter().zip(state) {
            let (l, n) = labels(
                ix_ref(master, r, "master cuts")?,
                sched(s)?,
                ix_ref(self.pos, r, "axis order")?,
            )?;
            lab.push(l);
            nb.push(n);
        }
        Ok((lab, nb))
    }

    /// Cross-fitted move of every selection row under this banding of table `t`: a row in half
    /// `1-k` moves by (block mean from half `k`) minus the table's own value. `scratch` is the
    /// caller's reusable block accumulator (see [`BlockScratch`]).
    fn moves_with(
        &self,
        t: usize,
        state: &[u8],
        scratch: &mut BlockScratch,
    ) -> Result<Vec<f64>, PbError> {
        let tab = ix_ref(self.tabs, t, "table")?;
        let g_sel = ix_ref(self.g_sel, t, "selection values")?;
        let g_pse = ix_ref(self.g_pse, t, "pseudo values")?;
        let mut out = vec![0.0; self.sel.len()];
        for (k, &lam) in self.lam_pse.iter().enumerate() {
            let (lab, nb) = self.bands(t, state, k)?;
            // band index of every selection and pseudo row, once — built one axis at a time
            // (`f = f * nb_d + label_d`, the same integer fold as a per-row pass)
            let fl_sel = flat_band_index(&tab.raws, &lab, &nb, &self.sel.cells, self.sel.len())?;
            let fl_pse = flat_band_index(&tab.raws, &lab, &nb, &self.pse.cells, self.pse.len())?;
            // Block sums in row order (half k's selection rows, then every pseudo-row): a dense
            // stamped scratch while the band grid is small enough, a map beyond. Either way each
            // touched block starts at (0, 0) and sums its rows in order; an untouched block reads
            // as absent, which the zeroed array's (0, 0) did too (both fail `s0 > 0`).
            let total: usize = nb.iter().product();
            let dense = total <= (1 << 22);
            let mut msum: HashMap<usize, (f64, f64)> = HashMap::default();
            if dense {
                scratch.begin(total);
            }
            for ((&h, &w), (&f, &g)) in self
                .half
                .iter()
                .zip(&self.sel.w)
                .zip(fl_sel.iter().zip(g_sel))
            {
                if h as usize == k {
                    if dense {
                        scratch.add(f, w, f64::from(g))?;
                    } else {
                        let e = msum.entry(f).or_insert((0.0, 0.0));
                        e.0 += w;
                        e.1 += w * f64::from(g);
                    }
                }
            }
            for (&f, &g) in fl_pse.iter().zip(g_pse) {
                if dense {
                    scratch.add(f, lam, f64::from(g))?;
                } else {
                    let e = msum.entry(f).or_insert((0.0, 0.0));
                    e.0 += lam;
                    e.1 += lam * f64::from(g);
                }
            }
            // rows of the other half landing in a block half k never saw: the exact floor mean
            let mut empty: HashMap<usize, f64> = HashMap::default();
            let mut index: Option<BandIndex> = None;
            for (i, slot) in out.iter_mut().enumerate() {
                if ix(self.half, i, "half")? as usize == k {
                    continue;
                }
                let f = ix(&fl_sel, i, "band index")?;
                let got = if dense {
                    scratch.get(f)
                } else {
                    msum.get(&f).copied()
                };
                let bm = match got {
                    Some((s0, s1)) if s0 > 0.0 => s1 / s0,
                    _ => {
                        let cached = if dense {
                            scratch.cached_prior(f)
                        } else {
                            empty.get(&f).copied()
                        };
                        if let Some(v) = cached {
                            v
                        } else {
                            if index.is_none() {
                                index = Some(self.band_index(t, &lab)?);
                            }
                            let bi = index.as_ref().ok_or_else(|| oob("band index"))?;
                            let v = self.block_prior_mean(t, bi, &nb, f)?;
                            if dense {
                                scratch.cache_prior(f, v)?;
                            } else {
                                empty.insert(f, v);
                            }
                            v
                        }
                    }
                };
                *slot = bm - f64::from(ix(g_sel, i, "selection value")?);
            }
        }
        Ok(out)
    }

    fn mse_with(&self, t: usize, state: &[u8], scratch: &mut BlockScratch) -> Result<f64, PbError> {
        let mv = self.moves_with(t, state, scratch)?;
        let tot: f64 = mv.iter().zip(&self.sel.w).map(|(m, w)| w * m * m).sum();
        Ok(tot / self.hsum)
    }

    /// Per axis of table `t`, each band's position range and its merged cells (ascending) under
    /// the labels `lab` — what every empty-block prior mean of one `moves` pass looks up.
    fn band_index(&self, t: usize, lab: &[Vec<u32>]) -> Result<BandIndex, PbError> {
        let tab = ix_ref(self.tabs, t, "table")?;
        let mut ranges = Vec::with_capacity(tab.order());
        let mut cells = Vec::with_capacity(tab.order());
        for ((&r, labd), &nd) in tab.raws.iter().zip(lab).zip(&tab.n) {
            let pos = ix_ref(self.pos, r, "axis order")?;
            let nbands = labd.iter().map(|&b| b as usize + 1).max().unwrap_or(0);
            let mut rg = vec![(usize::MAX, 0usize); nbands];
            for (&b, &q) in labd.iter().zip(pos) {
                let e = ix_mut(&mut rg, b as usize, "band range")?;
                e.0 = e.0.min(q as usize);
                e.1 = e.1.max(q as usize + 1);
            }
            let mut members: Vec<Vec<usize>> = vec![Vec::new(); nbands];
            for (c, &b) in labd.iter().take(nd).enumerate() {
                ix_mut(&mut members, b as usize, "band cells")?.push(c);
            }
            ranges.push(rg);
            cells.push(members);
        }
        Ok(BandIndex { ranges, cells })
    }

    /// Cross-fitted mean squared move of the WHOLE banding (every table's move summed per row) —
    /// what the budget is really about; the per-table sum the greedy allocates with ignores the
    /// covariance between tables, which overlapping tables make large.
    fn combined_mse(&self, states: &[Vec<u8>]) -> Result<f64, PbError> {
        // Tables in chunks: each chunk's moves are computed in parallel, then added to the running
        // total in table order — the same per-row sum as materializing every table's moves at
        // once, without holding `n_tables x n_rows` of them (3.6 GB on a 7,466-table bank).
        let mut tot = vec![0.0; self.sel.len()];
        for (ci, chunk) in states.chunks(COMBINED_CHUNK).enumerate() {
            let per: Vec<Vec<f64>> = chunk
                .par_iter()
                .enumerate()
                .map_init(BlockScratch::default, |scratch, (i, st)| {
                    self.moves_with(ci * COMBINED_CHUNK + i, st, scratch)
                })
                .collect::<Result<_, _>>()?;
            for mv in &per {
                for (a, m) in tot.iter_mut().zip(mv) {
                    *a += m;
                }
            }
        }
        Ok(tot
            .iter()
            .zip(&self.sel.w)
            .map(|(m, w)| w * m * m)
            .sum::<f64>()
            / self.hsum)
    }

    /// Exact floor-weighted mean of the table over one band-grid block (`index` from
    /// [`Self::band_index`] under the same labels).
    fn block_prior_mean(
        &self,
        t: usize,
        index: &BandIndex,
        nb: &[usize],
        flat: usize,
    ) -> Result<f64, PbError> {
        let tab = ix_ref(self.tabs, t, "table")?;
        let k = tab.order();
        let mut want = vec![0usize; k];
        let mut f = flat;
        for (w, &nbd) in want.iter_mut().zip(nb).rev() {
            *w = f % nbd;
            f /= nbd;
        }
        if let (Src::Boxes(boxes), Some(bp)) = (&tab.src, self.box_prefix(t)?) {
            // the block is a contiguous position range per axis: the floor sum factorises
            let mut rng = Vec::with_capacity(k);
            for (rd, &wd) in index.ranges.iter().zip(&want) {
                let (lo, hi) = rd.get(wd).copied().unwrap_or((usize::MAX, 0));
                if lo >= hi {
                    return Ok(0.0);
                }
                rng.push((lo, hi));
            }
            let mut den = 1.0;
            for (pmd, &(lo, hi)) in bp.pm.iter().zip(&rng) {
                den *= ix(pmd, hi, "prefix")? - ix(pmd, lo, "prefix")?;
            }
            let mut num = 0.0;
            for ((p, _), per) in boxes.iter().zip(&bp.side) {
                for (corner, &pv) in p.iter().enumerate() {
                    if pv == 0.0 {
                        continue;
                    }
                    let mut w = pv;
                    for (d, (sides, &(lo, hi))) in per.iter().zip(&rng).enumerate() {
                        let sd = ix_ref(sides.as_slice(), (corner >> d) & 1, "box side")?;
                        w *= ix(sd, hi, "prefix")? - ix(sd, lo, "prefix")?;
                    }
                    num += w;
                }
            }
            return Ok(if den > 0.0 { num / den } else { 0.0 });
        }
        // dense tables: restrict every axis to the block's cells, then sum P·f over the sub-grid
        let empty_band: Vec<usize> = Vec::new();
        let sub: Vec<&Vec<usize>> = index
            .cells
            .iter()
            .zip(&want)
            .map(|(cd, &wd)| cd.get(wd).unwrap_or(&empty_band))
            .collect();
        let mm: Vec<&Vec<f64>> = tab
            .raws
            .iter()
            .map(|&r| ix_ref(self.m, r, "measure"))
            .collect::<Result<_, _>>()?;
        let mut num = 0.0;
        let mut den = 0.0;
        let mut idx = vec![0usize; k];
        if sub.iter().any(|s| s.is_empty()) {
            return Ok(0.0);
        }
        // A dense source's value at a cell tuple is its row-major buffer entry (what `at` reads).
        let dense_vals: Option<&[f64]> = match &tab.src {
            Src::Dense(et) if !et.values.is_sparse() && et.values.shape() == tab.n => {
                Some(match et.values.values() {
                    std::borrow::Cow::Borrowed(v) => v,
                    std::borrow::Cow::Owned(_) => return Err(oob("dense band source")),
                })
            }
            _ => None,
        };
        let mut cell = vec![0usize; k];
        loop {
            for ((slot, s), &j) in cell.iter_mut().zip(&sub).zip(&idx) {
                *slot = ix(s, j, "block cell")?;
            }
            let mut p = 1.0_f64;
            for (md, &c) in mm.iter().zip(&cell) {
                p *= ix(md, c, "measure")?;
            }
            let v = match dense_vals {
                Some(vals) => {
                    let mut off = 0usize;
                    for (&c, &nd) in cell.iter().zip(&tab.n) {
                        off = off * nd + c;
                    }
                    vals.get(off).copied().unwrap_or(0.0)
                }
                None => tab.at(&cell)?,
            };
            num += p * v;
            den += p;
            let mut d = k;
            loop {
                if d == 0 {
                    return Ok(if den > 0.0 { num / den } else { 0.0 });
                }
                d -= 1;
                let len = ix_ref(&sub, d, "block axis")?.len();
                let id = ix_mut(&mut idx, d, "block axis")?;
                *id += 1;
                if *id < len {
                    break;
                }
                *id = 0;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------
// Banded tables, purification and distil.

#[derive(Clone)]
struct Banded {
    u: FeatureSet,
    raws: Vec<usize>,
    band: Vec<Vec<u32>>, // per axis: merged cell -> band
    nb: Vec<usize>,
    vals: Vec<f64>, // row-major over nb
}

impl Banded {
    fn flat_of_cells(&self, cells: &[Vec<u32>], i: usize) -> Result<usize, PbError> {
        let mut f = 0usize;
        for ((&r, bd), &nbd) in self.raws.iter().zip(&self.band).zip(&self.nb) {
            f = f * nbd + ix(bd, cell_at(cells, r, i)?, "band map")? as usize;
        }
        Ok(f)
    }
}

/// Band masses (normalized measure summed over each band), per axis.
fn band_mass(m: &[f64], band: &[u32], nb: usize) -> Result<Vec<f64>, PbError> {
    let tot: f64 = m.iter().sum();
    let mut out = vec![0.0; nb];
    for (&b, &mc) in band.iter().zip(m) {
        *ix_mut(&mut out, b as usize, "band mass")? += mc / tot;
    }
    Ok(out)
}

/// Common refinement of two partitions of the same merged axis, bands renumbered in merged order.
fn refine(a: &[u32], b: &[u32]) -> (Vec<u32>, usize) {
    let mut ids: HashMap<(u32, u32), u32> = HashMap::default();
    let mut out = Vec::with_capacity(a.len());
    for (&x, &y) in a.iter().zip(b) {
        let next = ids.len() as u32;
        out.push(*ids.entry((x, y)).or_insert(next));
    }
    let n = ids.len();
    (out, n)
}

fn refines(a: &[u32], b: &[u32]) -> bool {
    let mut mp: HashMap<u32, u32> = HashMap::default();
    a.iter()
        .zip(b)
        .all(|(&x, &y)| *mp.entry(x).or_insert(y) == y)
}

/// Re-express `t` on finer partitions `new_band` (each refining `t.band[d]`).
fn regrid(t: &Banded, new_band: Vec<Vec<u32>>, new_nb: Vec<usize>) -> Result<Banded, PbError> {
    let k = t.raws.len();
    let maps: Vec<Vec<usize>> = new_band
        .iter()
        .zip(&new_nb)
        .zip(&t.band)
        .take(k)
        .map(|((nbd, &nnb), tbd)| {
            let mut mp = vec![0usize; nnb];
            for (&nbnd, &tb) in nbd.iter().zip(tbd) {
                *ix_mut(&mut mp, nbnd as usize, "regrid map")? = tb as usize;
            }
            Ok(mp)
        })
        .collect::<Result<_, PbError>>()?;
    let total: usize = new_nb.iter().product();
    let mut vals = vec![0.0; total];
    let mut coord = vec![0usize; k];
    for slot in &mut vals {
        let mut f = 0usize;
        for ((&tnb, mp), &c) in t.nb.iter().zip(&maps).zip(&coord) {
            f = f * tnb + ix(mp, c, "regrid map")?;
        }
        *slot = ix(&t.vals, f, "regrid value")?;
        step(&mut coord, &new_nb);
    }
    Ok(Banded {
        u: t.u.clone(),
        raws: t.raws.clone(),
        band: new_band,
        nb: new_nb,
        vals,
    })
}

/// `target += piece` (piece on its own partitions of the same axes), refining `target` if needed.
fn add_piece(
    target: Banded,
    piece_band: &[Vec<u32>],
    piece_nb: &[usize],
    piece: &[f64],
) -> Result<Banded, PbError> {
    let k = target.raws.len();
    let mut t = target;
    if t.band
        .iter()
        .zip(piece_band)
        .take(k)
        .any(|(tb, pb)| !refines(tb, pb))
    {
        let mut nb = Vec::with_capacity(k);
        let mut band = Vec::with_capacity(k);
        for ((tb, pb), &tn) in t.band.iter().zip(piece_band).zip(&t.nb).take(k) {
            if refines(tb, pb) {
                band.push(tb.clone());
                nb.push(tn);
            } else {
                let (b, n) = refine(tb, pb);
                band.push(b);
                nb.push(n);
            }
        }
        t = regrid(&t, band, nb)?;
    }
    let maps: Vec<Vec<usize>> = t
        .band
        .iter()
        .zip(&t.nb)
        .zip(piece_band)
        .take(k)
        .map(|((tb, &tn), pb)| {
            let mut mp = vec![0usize; tn];
            for (&b, &pbc) in tb.iter().zip(pb) {
                *ix_mut(&mut mp, b as usize, "piece map")? = pbc as usize;
            }
            Ok(mp)
        })
        .collect::<Result<_, PbError>>()?;
    let mut coord = vec![0usize; k];
    for v in &mut t.vals {
        let mut f = 0usize;
        for ((&pn, mp), &c) in piece_nb.iter().zip(&maps).zip(&coord) {
            f = f * pn + ix(mp, c, "piece map")?;
        }
        *v += ix(piece, f, "piece value")?;
        step(&mut coord, &t.nb);
    }
    Ok(t)
}

/// Weighted mean of row-major `vals` (extents `nb`) over axis `d` under per-band masses `w`,
/// subtracted in place; returns the means, row-major over the remaining axes. Viewing the
/// shape as `(outer, nb[d], inner)`, each mean sums `w[i] * v` for `i` ascending from `0.0` and
/// every cell then has its mean subtracted — the same per-f64 operations, in the same order, as
/// an odometer walk over the cells.
fn center_axis(vals: &mut [f64], nb: &[usize], d: usize, w: &[f64]) -> Result<Vec<f64>, PbError> {
    let mid = ix(nb, d, "band count")?;
    let outer: usize = nb
        .get(..d)
        .ok_or_else(|| oob("band axis"))?
        .iter()
        .product();
    let inner: usize = nb
        .get(d + 1..)
        .ok_or_else(|| oob("band axis"))?
        .iter()
        .product();
    if mid == 0 || inner == 0 || vals.len() != outer * mid * inner {
        return Err(oob("band table shape"));
    }
    let w = w.get(..mid).ok_or_else(|| oob("axis mass"))?;
    let mut mean = vec![0.0; (outer * inner).max(1)];
    for (block, mrow) in vals
        .chunks_exact(mid * inner)
        .zip(mean.chunks_exact_mut(inner))
    {
        for (row, &wd) in block.chunks_exact(inner).zip(w) {
            for (m, &v) in mrow.iter_mut().zip(row) {
                *m += wd * v;
            }
        }
    }
    for (block, mrow) in vals
        .chunks_exact_mut(mid * inner)
        .zip(mean.chunks_exact(inner))
    {
        for row in block.chunks_exact_mut(inner) {
            for (v, &m) in row.iter_mut().zip(mrow) {
                *v -= m;
            }
        }
    }
    Ok(mean)
}

/// Exact purification of the banded bank under the product measure, highest order first:
/// each table is made zero-mean along every axis and the means pushed one order down (the order-1
/// means into `f0`).
fn repurify(
    bank: &mut BTreeMap<FeatureSet, Banded>,
    f0: &mut f64,
    m: &[Vec<f64>],
) -> Result<(), PbError> {
    let max_order = bank.keys().map(FeatureSet::order).max().unwrap_or(0);
    for order in (1..=max_order).rev() {
        let keys: Vec<FeatureSet> = bank
            .keys()
            .filter(|u| u.order() == order)
            .cloned()
            .collect();
        for u in keys {
            let Some(mut t) = bank.remove(&u) else {
                continue;
            };
            let k = t.raws.len();
            for d in 0..k {
                let w = band_mass(
                    ix_ref(m, ix(&t.raws, d, "raw")?, "measure")?,
                    ix_ref(&t.band, d, "band")?,
                    ix(&t.nb, d, "band count")?,
                )?;
                // mean over axis d
                let rest_nb: Vec<usize> =
                    t.nb.iter()
                        .enumerate()
                        .filter(|&(e, _)| e != d)
                        .map(|(_, &x)| x)
                        .collect();
                let mean = center_axis(&mut t.vals, &t.nb, d, &w)?;
                if k == 1 {
                    *f0 += ix(&mean, 0, "axis mean")?;
                    continue;
                }
                let rest_ids: Vec<u32> = t
                    .raws
                    .iter()
                    .enumerate()
                    .filter(|&(e, _)| e != d)
                    .map(|(_, &r)| r as u32)
                    .collect();
                let rest_u = FeatureSet::new(&rest_ids);
                let rest_band: Vec<Vec<u32>> = t
                    .band
                    .iter()
                    .enumerate()
                    .filter(|&(e, _)| e != d)
                    .map(|(_, b)| b.clone())
                    .collect();
                let tgt = bank.remove(&rest_u).unwrap_or_else(|| Banded {
                    u: rest_u.clone(),
                    raws: t
                        .raws
                        .iter()
                        .enumerate()
                        .filter(|&(e, _)| e != d)
                        .map(|(_, &r)| r)
                        .collect(),
                    band: rest_band.iter().map(|b| vec![0u32; b.len()]).collect(),
                    nb: vec![1; k - 1],
                    vals: vec![0.0],
                });
                bank.insert(rest_u, add_piece(tgt, &rest_band, &rest_nb, &mean)?);
            }
            bank.insert(u, t);
        }
    }
    Ok(())
}

/// Gauss-Seidel refit of the order-1 and order-2 tables to `target` on the training rows,
/// ridge-anchored to their current values with weight `lam · floor · Σh · band mass`.
fn distill(
    bank: &mut BTreeMap<FeatureSet, Banded>,
    f0: &mut f64,
    rows: &BandingRows<'_>,
    m: &[Vec<f64>],
    cfg: &BandingConfig,
) -> Result<(), PbError> {
    let n = rows.h.len();
    let hs: f64 = rows.h.iter().sum();
    let keys: Vec<FeatureSet> = bank.keys().cloned().collect();
    let missing = || PbError::Internal {
        what: "banding: distil key missing from the bank".into(),
    };
    let tables: Vec<&Banded> = keys
        .iter()
        .map(|u| bank.get(u).ok_or_else(missing))
        .collect::<Result<_, _>>()?;
    // Each table's band index per row (the `flat_of_cells` fold, a column at a time), tables in
    // parallel — independent integer work.
    let idx: Vec<Vec<u32>> = tables
        .par_iter()
        .map(|t| {
            Ok(flat_band_index(&t.raws, &t.band, &t.nb, rows.cells, n)?
                .into_iter()
                .map(|f| f as u32)
                .collect())
        })
        .collect::<Result<_, PbError>>()?;
    // Every row's prediction sums the tables in key order; row chunks are independent, so the
    // parallel fill adds the same values to each row in the same order.
    let mut pred = vec![*f0; n];
    pred.par_chunks_mut(DISTIL_ROW_CHUNK)
        .enumerate()
        .try_for_each(|(ci, chunk)| -> Result<(), PbError> {
            let lo = ci * DISTIL_ROW_CHUNK;
            for (t, ixs) in tables.iter().zip(&idx) {
                let ixs = ixs
                    .get(lo..lo + chunk.len())
                    .ok_or_else(|| oob("distil index"))?;
                for (p, &c) in chunk.iter_mut().zip(ixs) {
                    *p += ix(&t.vals, c as usize, "distil value")?;
                }
            }
            Ok(())
        })?;
    let prior: Vec<Vec<f64>> = tables
        .par_iter()
        .map(|t| {
            let per: Vec<Vec<f64>> = t
                .raws
                .iter()
                .zip(&t.band)
                .zip(&t.nb)
                .map(|((&r, bd), &nbd)| band_mass(ix_ref(m, r, "measure")?, bd, nbd))
                .collect::<Result<_, _>>()?;
            let mut out = vec![1.0; t.vals.len()];
            let mut coord = vec![0usize; t.raws.len()];
            for slot in &mut out {
                for (pd, &c) in per.iter().zip(&coord) {
                    *slot *= ix(pd, c, "band mass")?;
                }
                step(&mut coord, &t.nb);
            }
            Ok(out
                .iter()
                .map(|p| cfg.distill_lambda * cfg.floor * hs * p)
                .collect())
        })
        .collect::<Result<_, PbError>>()?;
    let anchor: Vec<Vec<f64>> = tables.iter().map(|t| t.vals.clone()).collect();
    let wd: Vec<Vec<f64>> = tables
        .par_iter()
        .zip(&idx)
        .map(|(t, ixs)| {
            let mut w = vec![0.0; t.vals.len()];
            for (&c, &h) in ixs.iter().zip(rows.h) {
                *ix_mut(&mut w, c as usize, "distil weight")? += h;
            }
            Ok(w)
        })
        .collect::<Result<_, PbError>>()?;
    drop(tables);
    let mut order: Vec<(usize, usize)> = keys
        .iter()
        .enumerate()
        .filter(|(_, u)| u.order() <= 2)
        .map(|(j, u)| (j, u.order()))
        .collect();
    order.sort_by_key(|&(_, o)| std::cmp::Reverse(o));
    let prof = std::env::var_os("TBOOST_PROFILE").is_some();
    let (mut ns_num, mut ns_d0) = (0u128, 0u128);
    let t_sweeps = std::time::Instant::now();
    // Gauss-Seidel sweeps. Each table's update to the predictions is DEFERRED into the next
    // table's accumulation pass: one sequential pass per table applies the pending update to a
    // row and then adds that row to the table's cell sums. Per row that is exactly the order of
    // the two separate passes (every row's update lands before the row is read, and each cell
    // still sums its rows in row order), so the result is bit-identical — with one pass over the
    // rows per table instead of two, and no thread-pool round trip per table.
    let mut carry: Option<(usize, Vec<f64>)> = None; // (table, new - old per cell) not yet applied
    let mut carry_d0: Option<f64> = None; // the previous sweep's intercept step, not yet applied
    for _ in 0..cfg.distill_sweeps {
        for &(j, _) in &order {
            let key = ix_ref(&keys, j, "distil key")?;
            let idx_j = ix_ref(&idx, j, "distil index")?;
            let wd_j = ix_ref(&wd, j, "distil weight")?;
            let prior_j = ix_ref(&prior, j, "distil prior")?;
            let anchor_j = ix_ref(&anchor, j, "distil anchor")?;
            let t0 = prof.then(std::time::Instant::now);
            let t = bank.get_mut(key).ok_or_else(missing)?;
            let mut num = vec![0.0; t.vals.len()];
            let tv = &t.vals;
            let mut add = |c: u32, h: f64, tg: f64, p: f64| -> Result<(), PbError> {
                let c = c as usize;
                *ix_mut(&mut num, c, "distil numerator")? +=
                    h * (tg - p + ix(tv, c, "distil value")?);
                Ok(())
            };
            let rows_iter = idx_j
                .iter()
                .zip(rows.h)
                .zip(rows.target)
                .zip(pred.iter_mut());
            match (carry.take(), carry_d0.take()) {
                (Some((jp, delta)), _) => {
                    let idx_p = ix_ref(&idx, jp, "distil index")?;
                    for ((((&c, &h), &tg), p), &cp) in rows_iter.zip(idx_p) {
                        *p += ix(&delta, cp as usize, "distil value")?;
                        add(c, h, tg, *p)?;
                    }
                }
                (None, Some(d0)) => {
                    for (((&c, &h), &tg), p) in rows_iter {
                        *p += d0;
                        add(c, h, tg, *p)?;
                    }
                }
                (None, None) => {
                    for (((&c, &h), &tg), p) in rows_iter {
                        add(c, h, tg, *p)?;
                    }
                }
            }
            let new: Vec<f64> = t
                .vals
                .iter()
                .zip(&num)
                .zip(wd_j)
                .zip(prior_j)
                .zip(anchor_j)
                .map(|((((&tv, &nc), &wc), &pc), &ac)| {
                    let den = wc + pc;
                    if den > 0.0 {
                        (nc + pc * ac) / den
                    } else {
                        tv
                    }
                })
                .collect();
            // Per-row update `+= new - old`, applied by the next pass (or before `d0` below).
            let delta: Vec<f64> = new.iter().zip(&t.vals).map(|(&a, &b)| a - b).collect();
            carry = Some((j, delta));
            t.vals = new;
            if let Some(t0) = t0 {
                ns_num += t0.elapsed().as_nanos();
            }
        }
        let t0 = prof.then(std::time::Instant::now);
        // With no order <= 2 table to carry it, the previous intercept step lands here.
        if let Some(d0p) = carry_d0.take() {
            for p in &mut pred {
                *p += d0p;
            }
        }
        if let Some((jp, delta)) = carry.take() {
            let idx_p = ix_ref(&idx, jp, "distil index")?;
            for (p, &cp) in pred.iter_mut().zip(idx_p) {
                *p += ix(&delta, cp as usize, "distil value")?;
            }
        }
        let d0: f64 = rows
            .h
            .iter()
            .zip(rows.target)
            .zip(&pred)
            .map(|((&h, &tg), &p)| h * (tg - p))
            .sum::<f64>()
            / hs;
        *f0 += d0;
        carry_d0 = Some(d0);
        if let Some(t0) = t0 {
            ns_d0 += t0.elapsed().as_nanos();
        }
    }
    if prof {
        eprintln!(
            "[banding-distil] {} sweeps x {} tables: {:.1}s | table passes {:.1}s intercept {:.1}s",
            cfg.distill_sweeps,
            order.len(),
            t_sweeps.elapsed().as_secs_f64(),
            ns_num as f64 / 1e9,
            ns_d0 as f64 / 1e9,
        );
    }
    Ok(())
}

/// Tables per parallel batch in [`Fidelity::combined_mse`].
const COMBINED_CHUNK: usize = 512;

/// Rows per parallel task in [`distill`]'s row-wise passes.
const DISTIL_ROW_CHUNK: usize = 16_384;

/// Normalized per-raw-axis measure of `bank` (what purification weights by).
fn normalized_measure(bank: &TableBank) -> Result<Vec<Vec<f64>>, PbError> {
    normalized_measure_with(bank, None)
}

fn normalized_measure_with(
    bank: &TableBank,
    marginals: Option<&[Vec<f64>]>,
) -> Result<Vec<Vec<f64>>, PbError> {
    Ok(bank_axis_measure_with(bank, marginals)?
        .iter()
        .map(|w| {
            let s: f64 = w.iter().sum();
            w.iter().map(|x| x / s).collect()
        })
        .collect())
}

/// Exact band-aware re-purification of a (possibly banded) all-dense bank — the counterpart of
/// `purify_raw_effects` for banks whose tables live on band grids (e.g. after graduation smooths
/// banded tables). A table whose grid must be refined to take a pushed piece keeps its support
/// split onto the finer grid by copy (display-only).
///
/// # Errors
/// [`PbError::InvalidInput`] if the bank still carries factored effects.
pub fn repurify_bank(bank: &TableBank) -> Result<TableBank, PbError> {
    repurify_bank_under(bank, &bank.w)
}

/// [`repurify_bank`] under another reference measure `w` (the ledger changes, the predictions do
/// not): the band-aware counterpart of [`TableBank::recompute_under`].
///
/// # Errors
/// [`PbError::InvalidInput`] if the bank still carries factored effects; measure errors.
pub fn repurify_bank_under(
    bank: &TableBank,
    w: &crate::explain::RefMeasure,
) -> Result<TableBank, PbError> {
    repurify_bank_under_with(bank, w, None)
}

/// [`repurify_bank_under`] with each axis's empirical marginal from `marginals` instead of the
/// main-effect supports (see [`TableBank::recentre_on`]).
pub(crate) fn repurify_bank_under_with(
    bank: &TableBank,
    w: &crate::explain::RefMeasure,
    marginals: Option<&[Vec<f64>]>,
) -> Result<TableBank, PbError> {
    if !bank.factored.is_empty() {
        return Err(PbError::InvalidInput {
            what: "repurify_bank needs an all-dense bank".into(),
        });
    }
    let mut probe = bank.clone();
    probe.w = w.clone();
    let m = normalized_measure_with(&probe, marginals)?;
    let ncell: Vec<usize> = m.iter().map(Vec::len).collect();
    let mut work: BTreeMap<FeatureSet, Banded> = BTreeMap::new();
    let mut support: BTreeMap<FeatureSet, Banded> = BTreeMap::new();
    let mut tpl: BTreeMap<FeatureSet, Vec<AxisId>> = BTreeMap::new();
    for t in &bank.tables {
        let raws: Vec<usize> = t.u.0.iter().map(|f| f.0 as usize).collect();
        let band: Vec<Vec<u32>> = t
            .axes
            .iter()
            .zip(&raws)
            .map(|(a, &r)| match &a.band_of {
                Some(b) => Ok(b.clone()),
                None => Ok((0..ix(&ncell, r, "cell count")? as u32).collect()),
            })
            .collect::<Result<_, PbError>>()?;
        let nb: Vec<usize> = t.axes.iter().map(|a| a.cells as usize).collect();
        let b = Banded {
            u: t.u.clone(),
            raws,
            band,
            nb,
            vals: t.values.values().into_owned(),
        };
        support.insert(
            t.u.clone(),
            Banded {
                vals: t.support.values().into_owned(),
                ..b.clone()
            },
        );
        tpl.insert(t.u.clone(), t.axes.clone());
        work.insert(t.u.clone(), b);
    }
    let mut f0 = bank.f0;
    repurify(&mut work, &mut f0, &m)?;
    let mut out = bank.clone();
    out.f0 = f0;
    out.w = w.clone();
    out.tables = work
        .into_values()
        .map(|b| {
            let axes0 = tpl.get(&b.u).cloned().unwrap_or_default();
            let sup = match support.get(&b.u) {
                Some(s) => {
                    if s.nb == b.nb && s.band == b.band {
                        s.vals.clone()
                    } else {
                        regrid(s, b.band.clone(), b.nb.clone())?.vals
                    }
                }
                None => vec![0.0; b.vals.len()],
            };
            emit_table(b, &axes0, &ncell, &m, sup)
        })
        .collect::<Result<_, _>>()?;
    Ok(out)
}

/// A banded table as an [`EffectTable`] (band maps on its axes, variance under the measure).
fn emit_table(
    b: Banded,
    axes_tpl: &[AxisId],
    ncell: &[usize],
    m: &[Vec<f64>],
    support: Vec<f64>,
) -> Result<EffectTable, PbError> {
    let k = b.raws.len();
    let mut axes = Vec::with_capacity(k);
    for (d, ((&r, bd), &nbd)) in b.raws.iter().zip(&b.band).zip(&b.nb).enumerate() {
        let mut a = axes_tpl.get(d).cloned().ok_or_else(|| PbError::Internal {
            what: format!("banding: no axis template for raw {r}"),
        })?;
        let nc = ix(ncell, r, "cell count")?;
        let identity = nbd == nc && bd.iter().enumerate().all(|(c, &x)| x as usize == c);
        if identity {
            a.band_of = None;
            a.cells = nc as u32;
        } else {
            a.band_of = Some(bd.clone());
            a.cells = nbd as u32;
        }
        axes.push(a);
    }
    let per: Vec<Vec<f64>> = b
        .raws
        .iter()
        .zip(&b.band)
        .zip(&b.nb)
        .map(|((&r, bd), &nbd)| band_mass(ix_ref(m, r, "measure")?, bd, nbd))
        .collect::<Result<_, _>>()?;
    let mut variance = 0.0;
    let mut coord = vec![0usize; k];
    for v in &b.vals {
        let p: f64 = per
            .iter()
            .zip(&coord)
            .map(|(pd, &c)| ix(pd, c, "band mass"))
            .product::<Result<f64, _>>()?;
        variance += p * v * v;
        step(&mut coord, &b.nb);
    }
    Ok(EffectTable {
        u: b.u,
        axes,
        values: Tensor::from_vec(b.nb.clone(), b.vals)?,
        support: Tensor::from_vec(b.nb, support)?,
        se_band: None,
        variance,
    })
}

// ------------------------------------------------------------------------------------------
// The stage.

/// Band every interaction table of `bank` (see the module doc). Mains are kept on the merged grid
/// (and refit by the distil). Returns the banded bank (every interaction dense on its band grid,
/// no factored effects) and the report.
///
/// # Errors
/// [`PbError::ShapeMismatch`] if the row inputs disagree with each other or with the bank.
pub fn band_bank(
    bank: &TableBank,
    rows: &BandingRows<'_>,
    cfg: &BandingConfig,
) -> Result<(TableBank, BandingReport), PbError> {
    let n = rows.h.len();
    if rows.mass.len() != n || rows.target.len() != n || rows.cells.iter().any(|c| c.len() != n) {
        return Err(PbError::ShapeMismatch {
            what: "banding row inputs have inconsistent lengths".into(),
        });
    }
    let n_raw = bank.merged_grids.len();
    if rows.cells.len() != n_raw {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "banding cells has {} raw features, bank {n_raw}",
                rows.cells.len()
            ),
        });
    }
    let m = normalized_measure(bank)?;
    let ncell: Vec<usize> = m.iter().map(Vec::len).collect();
    let ncells_of = |u: &FeatureSet| -> Result<Vec<usize>, PbError> {
        u.0.iter()
            .map(|f| ix(&ncell, f.0 as usize, "cell count"))
            .collect()
    };

    // interaction sources
    let mut inters: Vec<Inter<'_>> = Vec::new();
    for t in bank.tables.iter().filter(|t| t.u.order() >= 2) {
        inters.push(Inter {
            u: t.u.clone(),
            raws: t.u.0.iter().map(|f| f.0 as usize).collect(),
            n: ncells_of(&t.u)?,
            axes: t.axes.clone(),
            src: Src::Dense(t),
        });
    }
    for ft in &bank.factored {
        inters.push(Inter {
            u: ft.u.clone(),
            raws: ft.u.0.iter().map(|f| f.0 as usize).collect(),
            n: ncells_of(&ft.u)?,
            axes: ft.axes.clone(),
            src: Src::Boxes(ft.box_parts()),
        });
    }
    let mut report = BandingReport {
        tolerance: cfg.tolerance,
        budget: (cfg.tolerance * rows.sigma).powi(2).min(rows.mse_cap),
        noise_budget: (cfg.tolerance * rows.sigma).powi(2),
        deviance_budget: rows.mse_cap,
        selected_mse: 0.0,
        combined_mse: 0.0,
        allocation_budget: 0.0,
        calibration_rounds: 0,
        tables: u32::try_from(inters.len()).unwrap_or(u32::MAX),
        cells_before: BTreeMap::new(),
        cells_after: BTreeMap::new(),
        largest_after: 0,
        reordered: Vec::new(),
        skipped: None,
    };
    for t in &inters {
        *report.cells_before.entry(t.order() as u8).or_insert(0) +=
            t.n.iter().product::<usize>() as u64;
    }
    if inters.is_empty() {
        return Ok((bank.clone(), report));
    }

    // axis order: identity, except per-level (joint) categorical axes ordered by mean target
    let mut axis_tpl: Vec<Option<AxisId>> = vec![None; n_raw];
    for t in &inters {
        for (&r, a) in t.raws.iter().zip(&t.axes) {
            ix_mut(&mut axis_tpl, r, "axis template")?.get_or_insert_with(|| a.clone());
        }
    }
    let first_raw = |t: &EffectTable| -> Result<usize, PbError> {
        t.u.0
            .first()
            .map(|f| f.0 as usize)
            .ok_or_else(|| oob("main effect raw"))
    };
    for t in bank.tables.iter().filter(|t| t.u.order() == 1) {
        let r = first_raw(t)?;
        let slot = ix_mut(&mut axis_tpl, r, "axis template")?;
        if slot.is_none() {
            *slot = Some(
                t.axes
                    .first()
                    .cloned()
                    .ok_or_else(|| oob("main effect axis"))?,
            );
        }
    }
    let pos: Vec<Vec<u32>> = (0..n_raw)
        .map(|r| -> Result<Vec<u32>, PbError> {
            let nr = ix(&ncell, r, "cell count")?;
            let joint = ix_ref(&axis_tpl, r, "axis template")?
                .as_ref()
                .is_some_and(|a| a.joint_channels.is_some());
            if !joint || nr <= 3 {
                return Ok((0..nr as u32).collect());
            }
            let mut sw = vec![0.0; nr];
            let mut s1 = vec![0.0; nr];
            let col = ix_ref(rows.cells, r, "row cells")?;
            for ((&c, &ms), &tg) in col.iter().zip(rows.mass).zip(rows.target) {
                let c = c as usize;
                *ix_mut(&mut sw, c, "level mass")? += ms;
                *ix_mut(&mut s1, c, "level sum")? += ms * tg;
            }
            // (level, mean target) for every non-missing level; a stable sort on the key
            let mut order: Vec<(usize, f64)> = sw
                .iter()
                .zip(&s1)
                .map(|(&w, &s)| if w > 0.0 { s / w } else { f64::INFINITY })
                .enumerate()
                .skip(1)
                .collect();
            order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut p = vec![0u32; nr];
            for (q, &(c, _)) in order.iter().enumerate() {
                *ix_mut(&mut p, c, "level position")? = (q + 1) as u32;
            }
            Ok(p)
        })
        .collect::<Result<_, _>>()?;
    for (r, pr) in pos.iter().enumerate() {
        if pr.iter().enumerate().any(|(c, &q)| q as usize != c) {
            report.reordered.push(r as u32);
        }
    }

    // selection rows (subsample, two halves) and pseudo-rows from the measure
    let mut rng = rand_pcg::Pcg64::seed_from_u64(cfg.seed);
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        perm.swap(i, j);
    }
    perm.truncate(cfg.max_select_rows.min(n));
    let sel = RowSet {
        cells: rows
            .cells
            .iter()
            .map(|col| {
                perm.iter()
                    .map(|&i| ix(col, i, "row cell"))
                    .collect::<Result<Vec<u32>, _>>()
            })
            .collect::<Result<_, _>>()?,
        w: perm
            .iter()
            .map(|&i| ix(rows.h, i, "row curvature"))
            .collect::<Result<_, _>>()?,
    };
    let nsel = sel.len();
    let half: Vec<u8> = (0..nsel).map(|i| u8::from(i >= nsel / 2)).collect();
    let cdf: Vec<Vec<f64>> = m
        .iter()
        .map(|w| {
            let mut acc = 0.0;
            w.iter()
                .map(|x| {
                    acc += x;
                    acc
                })
                .collect()
        })
        .collect();
    let npse = cfg.pseudo_rows;
    let pse_cells: Vec<Vec<u32>> = (0..n_raw)
        .map(|r| -> Result<Vec<u32>, PbError> {
            let cd = ix_ref(&cdf, r, "measure cdf")?;
            let nc = ix(&ncell, r, "cell count")?;
            Ok((0..npse)
                .map(|_| {
                    let u: f64 = rng.gen();
                    cd.partition_point(|&c| c < u).min(nc - 1) as u32
                })
                .collect())
        })
        .collect::<Result<_, _>>()?;
    let hs_sel: f64 = sel.w.iter().sum();
    let half_mass = |h: u8| -> f64 {
        half.iter()
            .zip(&sel.w)
            .filter(|&(&hi, _)| hi == h)
            .map(|(_, &w)| w)
            .sum::<f64>()
    };
    let hs_half = [half_mass(0), half_mass(1)];
    let pse = RowSet {
        cells: pse_cells,
        w: vec![cfg.floor * hs_sel / npse.max(1) as f64; npse],
    };

    // table values at the selection rows and the pseudo-rows
    let prof = std::env::var_os("TBOOST_PROFILE").is_some();
    let mut t_lap = std::time::Instant::now();
    let mut lap = |label: &str| {
        if prof {
            eprintln!("[banding] {label}: {:.2}s", t_lap.elapsed().as_secs_f64());
        }
        t_lap = std::time::Instant::now();
    };
    let g_sel: Vec<Vec<f32>> = inters
        .par_iter()
        .map(|t| eval_rows(t, &sel))
        .collect::<Result<_, _>>()?;
    let g_pse: Vec<Vec<f32>> = inters
        .par_iter()
        .map(|t| eval_rows(t, &pse))
        .collect::<Result<_, _>>()?;

    // master cuts: halves (for the cross-fit) and all selection rows (for the deployed bands)
    let raws_used: Vec<usize> = {
        let mut v: Vec<usize> = inters.iter().flat_map(|t| t.raws.iter().copied()).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    lap("setup + row values");
    // Master cuts for the two cross-fit halves and for all selection rows: three independent
    // passes over every (raw feature, labelling) pair, run as ONE parallel batch — the largest
    // raws (most tables) first — so the few raws that dominate the work overlap instead of each
    // pass waiting on its own slowest raw. Each (labelling, raw) result is exactly what the
    // per-pass loop computed; they are reassembled in raw order.
    let tables_of: Vec<Vec<(usize, &Inter<'_>)>> = (0..n_raw)
        .map(|r| {
            inters
                .iter()
                .enumerate()
                .filter(|(_, t)| t.raws.contains(&r))
                .collect()
        })
        .collect();
    let keep_half0 = |i: usize| half.get(i) == Some(&0);
    let keep_half1 = |i: usize| half.get(i) == Some(&1);
    let keep_all = |_: usize| true;
    let passes: [(&(dyn Fn(usize) -> bool + Sync), f64); 3] = [
        (&keep_half0, hs_half[0]),
        (&keep_half1, hs_half[1]),
        (&keep_all, hs_sel),
    ];
    let mut jobs: Vec<(usize, usize)> = (0..passes.len())
        .flat_map(|p| raws_used.iter().map(move |&r| (p, r)))
        .collect();
    jobs.sort_by_key(|&(p, r)| {
        let work = tables_of.get(r).map_or(0, Vec::len);
        (std::cmp::Reverse(work), p, r)
    });
    // Workers pull jobs in that order from a shared counter (longest-first list scheduling).
    let next = std::sync::atomic::AtomicUsize::new(0);
    let workers = rayon::current_num_threads().max(1);
    type CutJob = ((usize, usize), Vec<u32>);
    let done: Vec<Vec<CutJob>> = (0..workers)
        .into_par_iter()
        .with_max_len(1)
        .map(|_| -> Result<Vec<CutJob>, PbError> {
            let mut out = Vec::new();
            loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(&(p, r)) = jobs.get(i) else {
                    return Ok(out);
                };
                let (keep, lam) = *passes.get(p).ok_or_else(|| oob("cut pass"))?;
                let pse_w = cfg.floor * lam / npse.max(1) as f64;
                let cuts = master_cuts(
                    r,
                    ix(&ncell, r, "cell count")?,
                    ix_ref(&pos, r, "axis order")?,
                    ix_ref(&tables_of, r, "raw tables")?,
                    &g_sel,
                    &g_pse,
                    &sel,
                    &pse,
                    keep,
                    pse_w,
                )?;
                out.push(((p, r), cuts));
            }
        })
        .collect::<Result<_, _>>()?;
    let mut by_pass: Vec<Vec<Vec<u32>>> = vec![vec![Vec::new(); n_raw]; passes.len()];
    for ((p, r), cuts) in done.into_iter().flatten() {
        *by_pass
            .get_mut(p)
            .and_then(|v| v.get_mut(r))
            .ok_or_else(|| oob("cut pass"))? = cuts;
    }
    let master_all = by_pass.pop().ok_or_else(|| oob("cut pass"))?;
    let half1 = by_pass.pop().ok_or_else(|| oob("cut pass"))?;
    let half0 = by_pass.pop().ok_or_else(|| oob("cut pass"))?;
    let masters_half = [half0, half1];
    lap("master cuts");

    // greedy per-axis resolution under the fidelity budget
    let fid = Fidelity {
        tabs: &inters,
        g_sel: &g_sel,
        g_pse: &g_pse,
        sel: &sel,
        pse: &pse,
        half: &half,
        lam_pse: [
            cfg.floor * hs_half[0] / npse.max(1) as f64,
            cfg.floor * hs_half[1] / npse.max(1) as f64,
        ],
        masters: &masters_half,
        pos: &pos,
        m: &m,
        hsum: hs_sel,
        prefix: (0..inters.len())
            .map(|_| std::sync::OnceLock::new())
            .collect(),
    };
    let nt = inters.len();
    let sub: Vec<Vec<usize>> = inters
        .iter()
        .map(|ta| {
            inters
                .iter()
                .enumerate()
                .filter(|(_, tb)| {
                    tb.order() < ta.order() && tb.raws.iter().all(|r| ta.raws.contains(r))
                })
                .map(|(b, _)| b)
                .collect()
        })
        .collect();
    let mut sup: Vec<Vec<usize>> = vec![Vec::new(); nt];
    for (a, subs) in sub.iter().enumerate() {
        for &b in subs {
            ix_mut(&mut sup, b, "super tables")?.push(a);
        }
    }
    lap("sub-table lattice");
    let n_states = SCHEDULE.len() as u8;
    let cells_of = |t: usize, st: &[u8]| -> Result<f64, PbError> {
        let tab = ix_ref(&inters, t, "table")?;
        tab.n
            .iter()
            .zip(st)
            .map(|(&nd, &s)| Ok((sched(s)?.saturating_add(2)).min(nd) as f64))
            .product::<Result<f64, PbError>>()
    };
    // (1) Each table's own frontier, independently and in parallel: from the coarsest banding,
    // repeatedly take the one-axis refinement buying the most fidelity per added cell, until full
    // resolution or until the table's move is negligible against the budget.
    // a table whose move is below a tenth of an even share of the budget cannot matter
    let negligible = report.budget / (10.0 * nt.max(1) as f64);
    type Pts = Vec<(Vec<u8>, f64, f64)>;
    // Extend a table's frontier from its last point — each step the one-axis refinement buying
    // the most fidelity per added cell — until its move is negligible (complete) or its cells
    // pass `cap` (may be deepened later). Returns whether the frontier is complete.
    let extend = |t: usize, pts: &mut Pts, cap: f64| -> Result<bool, PbError> {
        let tab = ix_ref(&inters, t, "table")?;
        let mut scratch = BlockScratch::default();
        if pts.is_empty() {
            let st = vec![0u8; tab.order()];
            let mse = fid.mse_with(t, &st, &mut scratch)?;
            let c = cells_of(t, &st)?;
            pts.push((st, mse, c));
        }
        loop {
            let (st, mse, cells) = pts.last().cloned().ok_or_else(|| PbError::Internal {
                what: "banding: empty frontier".into(),
            })?;
            if mse <= negligible {
                return Ok(true);
            }
            if cells > cap {
                return Ok(false);
            }
            let mut best: Option<(f64, Vec<u8>, f64)> = None;
            for (d, (&sd, &nd)) in st.iter().zip(&tab.n).enumerate() {
                if sd + 1 >= n_states || sched(sd)? >= nd {
                    continue;
                }
                let mut new = st.clone();
                *ix_mut(&mut new, d, "state axis")? += 1;
                let m2 = fid.mse_with(t, &new, &mut scratch)?;
                let dc = (cells_of(t, &new)? - cells).max(1.0);
                let g = (mse - m2) / dc;
                let better = match &best {
                    None => true,
                    Some(b) => g > b.0,
                };
                if better {
                    best = Some((g, new, m2));
                }
            }
            let Some((_, new, m2)) = best else {
                return Ok(true);
            };
            let c = cells_of(t, &new)?;
            pts.push((new, m2, c));
        }
    };
    // (2) One Lagrange multiplier for all tables: each takes the frontier point minimising
    // mse + λ·cells; then a k-way table's axes are never finer than those of the (k-1)-way
    // tables it contains (sub-tables are bumped, which only lowers their move). Also returns,
    // per table, whether it took the LAST point of its frontier.
    let pick = |fronts: &[Pts], lam: f64| -> Result<(Vec<Vec<u8>>, Vec<bool>), PbError> {
        let mut at_end = Vec::with_capacity(fronts.len());
        let mut state: Vec<Vec<u8>> = Vec::with_capacity(fronts.len());
        for f in fronts {
            let best = f
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    (a.1 .1 + lam * a.1 .2)
                        .partial_cmp(&(b.1 .1 + lam * b.1 .2))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, p)| (i, p.0.clone()))
                .unwrap_or_default();
            at_end.push(best.0 + 1 == f.len());
            state.push(best.1);
        }
        let mut order: Vec<usize> = (0..nt).collect();
        order.sort_by_key(|&t| std::cmp::Reverse(inters.get(t).map_or(0, Inter::order)));
        for t in order {
            let tab = ix_ref(&inters, t, "table")?;
            let st = ix_ref(&state, t, "state")?.clone();
            for &p in ix_ref(&sub, t, "sub tables")? {
                let praws = ix_ref(&inters, p, "table")?.raws.clone();
                let ps = ix_mut(&mut state, p, "state")?;
                for (dp, rp) in praws.iter().enumerate() {
                    if let Some(du) = tab.raws.iter().position(|x| x == rp) {
                        let nv = ix(&st, du, "state axis")?;
                        let slot = ix_mut(ps, dp, "state axis")?;
                        if *slot < nv {
                            *slot = nv;
                        }
                    }
                }
            }
        }
        Ok((state, at_end))
    };
    // (3) Bisect λ (in log space) on the TRUE combined cross-fitted move — every table's move
    // summed per row — so overlapping tables cannot jointly exceed the budget. `None` when even
    // every table's finest point is over budget.
    type Cal = (Vec<Vec<u8>>, Vec<bool>, f64, f64, u32);
    let calibrate = |fronts: &[Pts]| -> Result<Option<Cal>, PbError> {
        let max_gain = fronts
            .iter()
            .flat_map(|f| {
                f.iter()
                    .zip(f.iter().skip(1))
                    .map(|(a, b)| (a.1 - b.1) / (b.2 - a.2).max(1.0))
            })
            .fold(0.0_f64, f64::max)
            .max(f64::MIN_POSITIVE);
        let (mut lo, mut hi) = (max_gain * 1e-12, max_gain * 2.0); // lo: finest, hi: coarsest
        let (mut state, mut at_end) = pick(fronts, lo)?;
        let mut combined = fid.combined_mse(&state)?;
        let mut rounds = 1u32;
        if combined > report.budget {
            return Ok(None);
        }
        let (coarse, c_end) = pick(fronts, hi)?;
        let c_hi = fid.combined_mse(&coarse)?;
        rounds += 1;
        if c_hi <= report.budget {
            return Ok(Some((coarse, c_end, c_hi, hi, rounds)));
        }
        for _ in 0..24 {
            let mid = (lo * hi).sqrt();
            let (st, ae) = pick(fronts, mid)?;
            let c = fid.combined_mse(&st)?;
            rounds += 1;
            if c <= report.budget {
                lo = mid;
                state = st;
                at_end = ae;
                combined = c;
            } else {
                hi = mid;
            }
            if hi / lo < 1.05 {
                break;
            }
        }
        Ok(Some((state, at_end, combined, lo, rounds)))
    };
    // Iterative deepening: frontiers only as deep as the calibration actually uses.
    let mut cap = 4096.0_f64;
    let mut fronts: Vec<Pts> = vec![Vec::new(); nt];
    let mut done: Vec<bool> = fronts
        .par_iter_mut()
        .enumerate()
        .map(|(t, f)| extend(t, f, cap))
        .collect::<Result<_, _>>()?;
    lap("frontiers");
    let mut total_rounds = 0u32;
    let (state, combined, lam_used) = loop {
        let t_cal = std::time::Instant::now();
        let cal = calibrate(&fronts)?;
        if prof {
            eprintln!(
                "[banding]   calibrate over {} frontier points: {:.2}s",
                fronts.iter().map(Vec::len).sum::<usize>(),
                t_cal.elapsed().as_secs_f64()
            );
        }
        // tables to deepen: over budget even at the finest reached, every unfinished frontier;
        // otherwise the unfinished frontiers whose chosen point is their current end.
        let need: Vec<usize> = match &cal {
            None => (0..nt)
                .filter(|&t| !done.get(t).copied().unwrap_or(true))
                .collect(),
            Some((_, at_end, _, _, _)) => (0..nt)
                .filter(|&t| {
                    at_end.get(t).copied().unwrap_or(false) && !done.get(t).copied().unwrap_or(true)
                })
                .collect(),
        };
        if let Some((_, _, _, _, r)) = &cal {
            total_rounds += r;
        }
        if need.is_empty() || cap > 1e9 {
            match cal {
                Some((st, _, c, lam, _)) => break (st, c, lam),
                None => {
                    // The contract cannot be met even at every table's finest frontier point:
                    // the bank is too diffuse to band within the budget, so it ships unbanded.
                    let (st, _) = pick(&fronts, 0.0)?;
                    let c = fid.combined_mse(&st)?;
                    report.combined_mse = c;
                    report.calibration_rounds = total_rounds;
                    report.skipped = Some(format!(
                        "banding cannot meet its fidelity budget ({:.3e} > {:.3e}) even at \
                         the finest resolution; tables left unbanded",
                        c, report.budget
                    ));
                    return Ok((bank.clone(), report));
                }
            }
        }
        cap *= 8.0;
        let t_ext = std::time::Instant::now();
        let deeper: Vec<(usize, Pts, bool)> = need
            .par_iter()
            .map(|&t| {
                let mut f = fronts.get(t).cloned().unwrap_or_default();
                let complete = extend(t, &mut f, cap)?;
                Ok((t, f, complete))
            })
            .collect::<Result<_, PbError>>()?;
        for (t, f, complete) in deeper {
            *ix_mut(&mut fronts, t, "frontier")? = f;
            *ix_mut(&mut done, t, "frontier done")? = complete;
        }
        if prof {
            eprintln!(
                "[banding]   deepened {} frontiers to cap {cap}: {:.2}s",
                need.len(),
                t_ext.elapsed().as_secs_f64()
            );
        }
    };
    report.combined_mse = combined;
    report.allocation_budget = lam_used;
    report.calibration_rounds = total_rounds;
    let mut scratch = BlockScratch::default();
    report.selected_mse = state
        .iter()
        .enumerate()
        .map(|(t, st)| fid.mse_with(t, st, &mut scratch))
        .sum::<Result<f64, PbError>>()?;

    // deployed bands (master cuts from all selection rows) and block-mean values on ALL rows
    let hs_all: f64 = rows.h.iter().sum();
    let lam = cfg.floor * hs_all;
    let all = RowSet {
        cells: rows.cells.to_vec(),
        w: rows.h.to_vec(),
    };
    lap("calibration");
    let banded: Vec<Banded> = (0..nt)
        .into_par_iter()
        .map(|t| -> Result<Banded, PbError> {
            let tab = ix_ref(&inters, t, "table")?;
            let st = ix_ref(&state, t, "state")?;
            let mut band = Vec::with_capacity(tab.order());
            let mut nb = Vec::with_capacity(tab.order());
            for (&r, &s) in tab.raws.iter().zip(st) {
                let (l, k) = labels(
                    ix_ref(&master_all, r, "master cuts")?,
                    sched(s)?,
                    ix_ref(&pos, r, "axis order")?,
                )?;
                band.push(l);
                nb.push(k);
            }
            let mm: Vec<Vec<f64>> = tab
                .raws
                .iter()
                .map(|&r| ix_ref(&m, r, "measure").cloned())
                .collect::<Result<_, _>>()?;
            let q = tab.prior_band_sums(&band, &nb, &mm)?;
            let pmass: Vec<Vec<f64>> = mm
                .iter()
                .zip(&band)
                .zip(&nb)
                .map(|((md, bd), &nbd)| band_mass(md, bd, nbd))
                .collect::<Result<_, _>>()?;
            let g = eval_rows(tab, &all)?;
            let total: usize = nb.iter().product();
            let mut s0 = vec![0.0; total];
            let mut s1 = vec![0.0; total];
            let bt = Banded {
                u: tab.u.clone(),
                raws: tab.raws.clone(),
                band: band.clone(),
                nb: nb.clone(),
                vals: Vec::new(),
            };
            for (i, (&h, &gi)) in rows.h.iter().zip(&g).enumerate() {
                let f = bt.flat_of_cells(rows.cells, i)?;
                *ix_mut(&mut s0, f, "block mass")? += h;
                *ix_mut(&mut s1, f, "block sum")? += h * f64::from(gi);
            }
            let mut coord = vec![0usize; tab.order()];
            let vals: Vec<f64> = s0
                .iter()
                .zip(&s1)
                .zip(&q)
                .map(|((&a, &b), &qf)| {
                    let p: f64 = pmass
                        .iter()
                        .zip(&coord)
                        .map(|(pd, &c)| ix(pd, c, "band mass"))
                        .product::<Result<f64, _>>()?;
                    step(&mut coord, &nb);
                    let den = a + lam * p;
                    Ok(if den > 0.0 { (b + lam * qf) / den } else { 0.0 })
                })
                .collect::<Result<_, PbError>>()?;
            Ok(Banded { vals, ..bt })
        })
        .collect::<Result<_, _>>()?;

    // assemble (mains on the merged grid), re-purify, distil, re-purify
    let mut f0 = bank.f0;
    let mut work: BTreeMap<FeatureSet, Banded> = BTreeMap::new();
    for t in bank.tables.iter().filter(|t| t.u.order() == 1) {
        let r = first_raw(t)?;
        let nr = ix(&ncell, r, "cell count")?;
        work.insert(
            t.u.clone(),
            Banded {
                u: t.u.clone(),
                raws: vec![r],
                band: vec![(0..nr as u32).collect()],
                nb: vec![nr],
                vals: t.values.values().into_owned(),
            },
        );
    }
    for b in banded {
        work.insert(b.u.clone(), b);
    }
    lap("block means");
    repurify(&mut work, &mut f0, &m)?;
    distill(&mut work, &mut f0, rows, &m, cfg)?;
    repurify(&mut work, &mut f0, &m)?;
    lap("purify + distil");

    // emit
    let mut out = bank.clone();
    out.f0 = f0;
    out.factored.clear();
    let mut tables: Vec<EffectTable> = Vec::with_capacity(work.len());
    for (_, b) in work {
        let k = b.raws.len();
        let tpl: Vec<AxisId> = b
            .raws
            .iter()
            .map(|&r| axis_tpl.get(r).cloned().flatten())
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| PbError::Internal {
                what: "banding: missing axis template".into(),
            })?;
        let mut support = vec![0.0; b.vals.len()];
        for (i, &ms) in rows.mass.iter().enumerate() {
            *ix_mut(&mut support, b.flat_of_cells(rows.cells, i)?, "support")? += ms;
        }
        if k >= 2 {
            *report.cells_after.entry(k as u8).or_insert(0) += b.vals.len() as u64;
            report.largest_after = report.largest_after.max(b.vals.len() as u64);
        }
        tables.push(emit_table(b, &tpl, &ncell, &m, support)?);
    }
    out.tables = tables;
    Ok((out, report))
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::float_cmp,
        clippy::needless_range_loop
    )]
    use super::*;
    use rand::Rng;

    type RefGroups = HashMap<u64, (f64, f64)>;

    /// The hash-map master cuts this module replaced (2026-09-27), verbatim: the group build.
    #[allow(clippy::too_many_arguments)]
    fn reference_groups(
        r: usize,
        n: usize,
        pos: &[u32],
        tabs: &[(usize, &Inter<'_>)],
        g_sel: &[Vec<f32>],
        g_pse: &[Vec<f32>],
        sel: &RowSet,
        pse: &RowSet,
        keep: &(dyn Fn(usize) -> bool + Sync),
        pse_w: f64,
    ) -> Result<Vec<RefGroups>, PbError> {
        let mut groups: Vec<RefGroups> = vec![HashMap::default(); n];
        for &(ti, t) in tabs {
            let d = t.raws.iter().position(|&x| x == r).unwrap_or(0);
            let mut add = |rows: &RowSet, i: usize, w: f64, g: f32| -> Result<(), PbError> {
                let q = ix(pos, cell_at(&rows.cells, r, i)?, "master position")? as usize;
                if q == 0 {
                    return Ok(());
                }
                let mut key: u64 = 0;
                for (e, (&rr, &ne)) in t.raws.iter().zip(&t.n).enumerate() {
                    if e != d {
                        key = key * ne as u64 + cell_at(&rows.cells, rr, i)? as u64;
                    }
                }
                key |= (ti as u64) << 40;
                let e = ix_mut(&mut groups, q, "master group")?
                    .entry(key)
                    .or_insert((0.0, 0.0));
                e.0 += w;
                e.1 += w * f64::from(g);
                Ok(())
            };
            let gs = ix_ref(g_sel, ti, "selection values")?;
            for i in 0..sel.len() {
                if keep(i) {
                    add(sel, i, sel.w[i], gs[i])?;
                }
            }
            let gp = ix_ref(g_pse, ti, "pseudo values")?;
            for j in 0..pse.len() {
                add(pse, j, pse_w, gp[j])?;
            }
        }
        Ok(groups)
    }

    /// The replaced merge cost: summed in the smaller map's iteration order.
    fn reference_cost(a: &RefGroups, b: &RefGroups) -> f64 {
        let (small, big) = if a.len() <= b.len() { (a, b) } else { (b, a) };
        let mut c = 0.0;
        for (k, &(x0, x1)) in small {
            if let Some(&(y0, y1)) = big.get(k) {
                let t0 = x0 + y0;
                if x0 > 0.0 && y0 > 0.0 {
                    c += x1 * x1 / x0 + y1 * y1 / y0 - (x1 + y1) * (x1 + y1) / t0;
                }
            }
        }
        c
    }

    /// The replaced greedy merge loop.
    fn reference_order(mut groups: Vec<RefGroups>) -> Vec<u32> {
        let n = groups.len();
        let mut alive: Vec<usize> = (1..n).collect();
        let mut costs: Vec<f64> = alive
            .iter()
            .zip(alive.iter().skip(1))
            .map(|(&a, &b)| reference_cost(&groups[a], &groups[b]))
            .collect();
        let mut removed = Vec::new();
        while alive.len() > 1 {
            let (i, _) = costs
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));
            let (a, b) = (alive[i], alive[i + 1]);
            let gb = std::mem::take(&mut groups[b]);
            for (k, (y0, y1)) in gb {
                let e = groups[a].entry(k).or_insert((0.0, 0.0));
                e.0 += y0;
                e.1 += y1;
            }
            removed.push(b as u32);
            alive.remove(i + 1);
            costs.remove(i);
            if i > 0 {
                costs[i - 1] = reference_cost(&groups[alive[i - 1]], &groups[alive[i]]);
            }
            if i < alive.len() - 1 {
                costs[i] = reference_cost(&groups[alive[i]], &groups[alive[i + 1]]);
            }
        }
        removed.reverse();
        removed
    }

    fn sorted(g: &RefGroups) -> Groups {
        let mut v: Groups = g.iter().map(|(&k, &(s0, s1))| (k, s0, s1)).collect();
        v.sort_by_key(|e| e.0);
        v
    }

    fn same_bits(a: &Groups, b: &Groups) -> bool {
        a.len() == b.len()
            && a.iter().zip(b).all(|(x, y)| {
                x.0 == y.0 && x.1.to_bits() == y.1.to_bits() && x.2.to_bits() == y.2.to_bits()
            })
    }

    /// Random rows over `ncell.len()` raws (cell 0 = missing), with every main, some pairs and
    /// triples, and one triple whose declared extents force the sparse (sorting) accumulator.
    struct Case {
        ncell: Vec<usize>,
        tables: Vec<Inter<'static>>,
        sel: RowSet,
        pse: RowSet,
        g_sel: Vec<Vec<f32>>,
        g_pse: Vec<Vec<f32>>,
    }

    fn case(seed: u64) -> Case {
        let mut rng = rand_pcg::Pcg64::seed_from_u64(seed);
        let n_raw = 5;
        let ncell: Vec<usize> = (0..n_raw).map(|_| rng.gen_range(3..10)).collect();
        let inter = |raws: Vec<usize>, n: Vec<usize>| Inter {
            u: FeatureSet::new(&raws.iter().map(|&r| r as u32).collect::<Vec<_>>()),
            raws,
            n,
            axes: Vec::new(),
            src: Src::Boxes(Vec::new()),
        };
        let mut tables = Vec::new();
        for r in 0..n_raw {
            tables.push(inter(vec![r], vec![ncell[r]]));
        }
        for &(a, b) in &[(0, 1), (0, 3), (1, 2), (2, 4), (3, 4)] {
            tables.push(inter(vec![a, b], vec![ncell[a], ncell[b]]));
        }
        for &(a, b, c) in &[(0, 1, 2), (1, 3, 4), (0, 2, 4)] {
            tables.push(inter(vec![a, b, c], vec![ncell[a], ncell[b], ncell[c]]));
        }
        // declared extents past DENSE_CROSS_MAX: the key radix only needs cells < extent
        tables.push(inter(vec![0, 3, 4], vec![ncell[0], 1500, 1500]));
        let rows = |n: usize, rng: &mut rand_pcg::Pcg64| -> Vec<Vec<u32>> {
            ncell
                .iter()
                .map(|&nc| (0..n).map(|_| rng.gen_range(0..nc) as u32).collect())
                .collect()
        };
        let n_sel = 700;
        let n_pse = 90;
        let sel_cells = rows(n_sel, &mut rng);
        let pse_cells = rows(n_pse, &mut rng);
        let w: Vec<f64> = (0..n_sel)
            .map(|_| {
                if rng.gen_bool(0.05) {
                    0.0
                } else {
                    rng.gen_range(0.01..2.0)
                }
            })
            .collect();
        let g = |n: usize, rng: &mut rand_pcg::Pcg64| -> Vec<Vec<f32>> {
            (0..tables.len())
                .map(|_| (0..n).map(|_| rng.gen_range(-1.0_f32..1.0)).collect())
                .collect()
        };
        let g_sel = g(n_sel, &mut rng);
        let g_pse = g(n_pse, &mut rng);
        Case {
            ncell: ncell.clone(),
            sel: RowSet {
                cells: sel_cells,
                w,
            },
            pse: RowSet {
                cells: pse_cells,
                w: vec![1.0; n_pse],
            },
            g_sel,
            g_pse,
            tables,
        }
    }

    /// Each position's `(Σw, Σw·g)` is summed in row-arrival order by both builds, so the sorted
    /// lists hold exactly the hash maps' entries, bit for bit — dense and sparse accumulators.
    #[test]
    fn sorted_master_groups_hold_the_hash_map_sums_bit_for_bit() {
        for seed in 0..12 {
            let c = case(seed);
            let keep = |i: usize| i % 3 != 1;
            for r in 0..c.ncell.len() {
                let tabs: Vec<(usize, &Inter<'_>)> = c
                    .tables
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| t.raws.contains(&r))
                    .collect();
                let pos: Vec<u32> = (0..c.ncell[r] as u32).collect();
                let n = c.ncell[r];
                let new = master_groups(
                    r, n, &pos, &tabs, &c.g_sel, &c.g_pse, &c.sel, &c.pse, &keep, 0.37,
                )
                .unwrap();
                let old = reference_groups(
                    r, n, &pos, &tabs, &c.g_sel, &c.g_pse, &c.sel, &c.pse, &keep, 0.37,
                )
                .unwrap();
                assert_eq!(new.len(), old.len());
                for (q, (a, b)) in new.iter().zip(&old).enumerate() {
                    assert!(same_bits(a, &sorted(b)), "seed {seed} raw {r} position {q}");
                }
            }
        }
    }

    /// Only the ORDER of each merge cost's sum changed (key order, not hash order): the costs
    /// agree to rounding, and on these cases the cut rankings are identical.
    #[test]
    fn sorted_master_cuts_rank_like_the_hash_maps() {
        for seed in 0..12 {
            let c = case(seed);
            let keep = |i: usize| i % 4 != 0;
            for r in 0..c.ncell.len() {
                let tabs: Vec<(usize, &Inter<'_>)> = c
                    .tables
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| t.raws.contains(&r))
                    .collect();
                let pos: Vec<u32> = (0..c.ncell[r] as u32).collect();
                let n = c.ncell[r];
                let new = master_groups(
                    r, n, &pos, &tabs, &c.g_sel, &c.g_pse, &c.sel, &c.pse, &keep, 0.21,
                )
                .unwrap();
                let old = reference_groups(
                    r, n, &pos, &tabs, &c.g_sel, &c.g_pse, &c.sel, &c.pse, &keep, 0.21,
                )
                .unwrap();
                for q in 1..n.saturating_sub(1) {
                    let (a, b) = (
                        merge_cost(&new[q], &new[q + 1]),
                        reference_cost(&old[q], &old[q + 1]),
                    );
                    assert!(
                        (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1e-300),
                        "{a} vs {b}"
                    );
                }
                assert_eq!(
                    merge_order(new).unwrap(),
                    reference_order(old),
                    "seed {seed} raw {r}"
                );
            }
        }
    }

    /// Folding one group into another matches a running per-key sum bit for bit, on the in-place
    /// fast paths (either side a subset of the other) and on the general merge.
    #[test]
    fn merge_groups_matches_a_running_per_key_sum() {
        let mut rng = rand_pcg::Pcg64::seed_from_u64(7);
        let random_group = |keys: &[u64], rng: &mut rand_pcg::Pcg64| -> Groups {
            let mut v: Groups = keys
                .iter()
                .map(|&k| (k, rng.gen_range(0.0..3.0), rng.gen_range(-2.0..2.0)))
                .collect();
            v.sort_by_key(|e| e.0);
            v
        };
        let all: Vec<u64> = (0..400).map(|k| k * 7 + 3).collect();
        let subset: Vec<u64> = all.iter().copied().step_by(37).collect();
        let offset: Vec<u64> = (0..300).map(|k| k * 5).collect();
        let mut scratch = Vec::new();
        for (ka, kb) in [
            (&all, &subset),
            (&subset, &all),
            (&all, &offset),
            (&subset, &offset),
        ] {
            let a = random_group(ka, &mut rng);
            let b = random_group(kb, &mut rng);
            let mut want: RefGroups = HashMap::default();
            for &(k, x0, x1) in &a {
                want.insert(k, (x0, x1));
            }
            for &(k, y0, y1) in &b {
                let e = want.entry(k).or_insert((0.0, 0.0));
                e.0 += y0;
                e.1 += y1;
            }
            let mut got = a.clone();
            merge_groups(&mut got, b, &mut scratch);
            assert!(same_bits(&got, &sorted(&want)));
        }
    }
}
