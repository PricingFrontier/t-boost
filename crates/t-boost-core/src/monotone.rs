//! Monotone repair of a deployed table bank (2026-10-14).
//!
//! The trees honour `monotone_constraints` (split vetoes plus `clamp_monotone`), so the full
//! purified bank is monotone in every constrained feature. The post-fit stages that drop or
//! rewrite tables — selection, rebalance, banding, graduation, the budgets — do not know about
//! the constraint, and a pure interaction cannot be monotone on its own: only the FAMILY of a
//! constrained feature `j` (its main effect plus every interaction containing `j`) is. This
//! module runs last and projects each family back onto the constraint, changing as little as it
//! can, in phases that each finish for every constrained feature before the next starts (a
//! clip made for one feature can move another feature's steps through a table they share):
//!
//! 1. every constrained main effect is made isotonic (weighted PAVA). That is necessary: under a
//!    product reference measure a pure interaction averages to zero over its partners, so the
//!    main effect is the partner-average of its family, and a monotone family has a monotone
//!    main effect;
//! 2. where an interaction still steps the wrong way by more than the main effect's slope can
//!    absorb, its slices along `j` are clipped (shifted PAVA) against floors that share the
//!    slope between the interactions in proportion to their need; a table that holds several
//!    constrained features alternates the projections (Dykstra). Factored effects are never
//!    clipped: their need is taken from the slope first;
//! 3. the clipped tables and their down-closure are re-purified, which moves no family's steps
//!    along its own feature;
//! 4. every constrained main effect is projected onto the interaction ENVELOPE — the smallest
//!    step its interactions take at each boundary over every combination of the other features
//!    (missing cells included) — which closes whatever is left;
//! 5. main effects are re-centred and their variances recomputed; the caller re-anchors.
//!
//! A bank whose every family already passes is returned untouched. Only the numeric missing
//! cell (cell 0) is exempt from the constraint: the trees route it freely.
//!
//! Measured (freMTPL2, 2026-10-14, recorded in `specs/roadmap/monotone-pruning.md`): on the
//! default pipeline the repaired model's test deviance moved -0.004% and no main effect moved by
//! more than 4%. Correcting only the main effects cost up to +24.8% when banding had put large
//! swings into the interactions, which is why step 2 exists.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::data::{BinnedMatrix, FeatureId};
use crate::error::PbError;
use crate::explain::{AxisId, EffectTable, FactoredEffect, FeatureSet, TableBank};
use crate::loss::{Link, Loss};
use crate::table_model::TableModel;

/// Share of the curvature mass spread over the purification measure as a positive floor under
/// every cell's projection weight, so an empty cell still has a weight (banding's `floor`, for
/// the same reason).
const WEIGHT_FLOOR: f64 = 0.01;
/// Table lookups one component's exact envelope may cost; beyond it the sound per-table bound
/// (the sum of each table's own minimum step) is used instead.
const EXACT_WORK_CAP: u64 = 50_000_000;
/// Rounds of alternating projections for a table that holds two or more constrained features.
const SHARED_ROUNDS: usize = 8;
/// A link-scale step above `-STEP_TOL` counts as monotone: it is the `f64` rounding of
/// purification, far below anything a rating table prints.
const STEP_TOL: f64 = 1e-9;

/// One constrained raw feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MonotoneSign {
    /// Raw feature id (native order).
    pub feature: u32,
    /// `true` for non-decreasing, `false` for non-increasing.
    pub increasing: bool,
}

/// The training rows the projections are weighted by.
pub struct RepairRows<'a> {
    /// `cells[raw][row]`: merged-grid cell of every training row on every raw feature.
    pub cells: &'a [Vec<u32>],
    /// Loss curvature per row at the deployed score.
    pub h: &'a [f64],
}

/// What the repair did, serialized into `pruning_report_["monotone"]`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MonotoneReport {
    /// One entry per constrained feature.
    pub features: Vec<FeatureReport>,
    /// `true` when every family already passed and the bank was returned untouched.
    pub certified: bool,
}

/// One constrained feature's repair.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FeatureReport {
    /// Raw feature id.
    pub feature: u32,
    /// `1` increasing, `-1` decreasing.
    pub sign: i8,
    /// Supports of every deployed table containing the feature (empty: the model does not use it).
    pub tables: Vec<Vec<u32>>,
    /// `"exact"` when every boundary's envelope was enumerated over every combination of the
    /// other features; `"bound"` when a component was too large and the sound per-table bound was
    /// used (it can repair a family that was already monotone, never miss one that was not).
    pub certificate: String,
    /// The family's most negative step before the repair (link scale, sign applied; `>= 0` is
    /// monotone).
    pub worst_step_before: f64,
    /// The same after the repair.
    pub worst_step_after: f64,
    /// Largest change to any cell of the main effect (link scale).
    pub main_change: f64,
    /// Interaction tables whose slices were clipped.
    pub clipped: Vec<Vec<u32>>,
    /// Largest change to any cell of a clipped table, before re-purification (link scale).
    pub clip_change: f64,
}

// ---------------------------------------------------------------------------------------------
// Projections.

/// Weighted least-squares projection of `v` onto `{sign * (v[k+1] - v[k]) >= floor[k]}`, where a
/// `None` floor leaves that step free. `w` must be positive. Exact: the free steps split `v` into
/// independent segments, and within a segment the substitution `u[k] = sign * v[k] - C[k]`,
/// `C[k] = Σ_{i<k} floor[i]`, turns the constraint into plain isotonicity of `u`, which PAVA
/// solves exactly; the substitution is a translation, so it preserves the weighted distance.
///
/// # Errors
/// [`PbError::ShapeMismatch`] if the lengths disagree; [`PbError::InvalidInput`] for a
/// non-positive or non-finite weight.
pub fn shifted_pava(
    v: &[f64],
    w: &[f64],
    floor: &[Option<f64>],
    increasing: bool,
) -> Result<Vec<f64>, PbError> {
    let n = v.len();
    if w.len() != n || floor.len() != n.saturating_sub(1) {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "shifted_pava: {} values, {} weights, {} floors",
                n,
                w.len(),
                floor.len()
            ),
        });
    }
    if w.iter().any(|&x| !(x.is_finite() && x > 0.0)) {
        return Err(PbError::InvalidInput {
            what: "shifted_pava weights must be finite and positive".into(),
        });
    }
    let s = if increasing { 1.0 } else { -1.0 };
    let mut out = Vec::with_capacity(n);
    let mut start = 0usize;
    while start < n {
        // The segment runs to the first free step (or the end).
        let mut end = start;
        while let Some(Some(_)) = floor.get(end) {
            end += 1;
        }
        // end is the last index of the segment: v[start..=end]
        let seg_v = v.get(start..=end).ok_or_else(internal("pava segment"))?;
        let seg_w = w.get(start..=end).ok_or_else(internal("pava weights"))?;
        let mut shift = Vec::with_capacity(seg_v.len());
        let mut acc = 0.0_f64;
        for k in 0..seg_v.len() {
            shift.push(acc);
            if let Some(Some(f)) = floor.get(start + k) {
                acc += f;
            }
        }
        let u: Vec<f64> = seg_v.iter().zip(&shift).map(|(&x, &c)| s * x - c).collect();
        let fitted = pava(&u, seg_w);
        out.extend(fitted.iter().zip(&shift).map(|(&x, &c)| s * (x + c)));
        start = end + 1;
    }
    Ok(out)
}

/// Weighted isotonic (non-decreasing) regression by pool-adjacent-violators.
fn pava(y: &[f64], w: &[f64]) -> Vec<f64> {
    // Each block: (weighted mean, total weight, length).
    let mut blocks: Vec<(f64, f64, usize)> = Vec::with_capacity(y.len());
    for (&yi, &wi) in y.iter().zip(w) {
        blocks.push((yi, wi, 1));
        while blocks.len() > 1 {
            let (Some(&(m2, w2, c2)), Some(&(m1, w1, c1))) =
                (blocks.last(), blocks.get(blocks.len() - 2))
            else {
                break;
            };
            if m1 <= m2 {
                break;
            }
            blocks.truncate(blocks.len() - 2);
            let wt = w1 + w2;
            blocks.push(((m1 * w1 + m2 * w2) / wt, wt, c1 + c2));
        }
    }
    let mut out = Vec::with_capacity(y.len());
    for (m, _, c) in blocks {
        out.extend(std::iter::repeat_n(m, c));
    }
    out
}

fn internal(what: &str) -> impl FnOnce() -> PbError + '_ {
    move || PbError::Internal {
        what: format!("monotone repair: {what}"),
    }
}

// ---------------------------------------------------------------------------------------------
// Geometry.

/// Merged-cell count of raw feature `raw`.
fn n_cells(bank: &TableBank, raw: u32) -> Result<usize, PbError> {
    bank.merged_grids
        .get(raw as usize)
        .map(|g| usize::from(g.n_bins))
        .ok_or_else(internal("raw feature outside the merged grids"))
}

/// Merged cell -> coordinate along `axis` (its band when banded).
fn coord_map(axis: &AxisId, merged: usize) -> Result<Vec<usize>, PbError> {
    (0..merged)
        .map(|c| {
            let cell = u32::try_from(c).map_err(|_| internal("merged cell exceeds u32")())?;
            axis.coord(cell)
                .ok_or_else(internal("merged cell outside the band map"))
        })
        .collect()
}

/// The number of lines through a factored effect's partner cells (the product of `dims`), or
/// `None` when it overflows a `usize`.
fn line_count(dims: &[usize]) -> Option<usize> {
    dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

/// Row-major strides of `shape`.
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut out = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        let next = out.get(d + 1).copied().unwrap_or(1);
        let dim = shape.get(d + 1).copied().unwrap_or(1);
        if let Some(slot) = out.get_mut(d) {
            *slot = next.saturating_mul(dim);
        }
    }
    out
}

/// Flat offsets of every line along axis `p` (the cells with coordinate 0 on `p`), in row-major
/// order of the other coordinates.
fn line_bases(shape: &[usize], p: usize) -> Result<Vec<usize>, PbError> {
    let st = strides(shape);
    let sp = *st.get(p).ok_or_else(internal("line axis"))?;
    let dp = *shape.get(p).ok_or_else(internal("line axis extent"))?;
    let total: usize = shape.iter().product();
    Ok((0..total).filter(|&f| (f / sp) % dp.max(1) == 0).collect())
}

/// The per-raw purification measure, normalized to sum to one on every axis.
fn axis_measure(bank: &TableBank) -> Result<Vec<Vec<f64>>, PbError> {
    Ok(crate::explain::bank_axis_measure_with(bank, None)?
        .into_iter()
        .map(|w| {
            let s: f64 = w.iter().sum();
            w.into_iter()
                .map(|x| if s > 0.0 { x / s } else { 0.0 })
                .collect()
        })
        .collect())
}

/// The measure of every cell of a dense table: the product over its axes of the axis measure
/// summed into the axis's bands.
fn table_measure(t: &EffectTable, mu: &[Vec<f64>], bank: &TableBank) -> Result<Vec<f64>, PbError> {
    let shape = t.values.shape();
    let mut per_axis = Vec::with_capacity(t.axes.len());
    for (a, &dim) in t.axes.iter().zip(&shape) {
        let raw_mu = mu
            .get(a.raw.0 as usize)
            .ok_or_else(internal("measure axis"))?;
        let map = coord_map(a, n_cells(bank, a.raw.0)?)?;
        let mut m = vec![0.0_f64; dim];
        for (c, &coord) in map.iter().enumerate() {
            *m.get_mut(coord)
                .ok_or_else(internal("band outside the table"))? +=
                raw_mu.get(c).copied().unwrap_or(0.0);
        }
        per_axis.push(m);
    }
    let total: usize = shape.iter().product();
    let st = strides(&shape);
    (0..total)
        .map(|f| {
            let mut p = 1.0_f64;
            for ((m, &s), &dim) in per_axis.iter().zip(&st).zip(&shape) {
                p *= m.get((f / s) % dim.max(1)).copied().unwrap_or(0.0);
            }
            Ok(p)
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// A constrained feature's family.

/// A floor per boundary of a constrained feature (`None`: that step is free).
type Floors = Vec<Option<f64>>;

/// One interaction of a family, as its steps along the constrained feature.
struct Steps {
    /// Partner raw features, in the table's axis order.
    partners: Vec<u32>,
    /// Merged cell -> own coordinate, per partner.
    maps: Vec<Vec<usize>>,
    /// Own extent per partner (row-major order of the lines).
    dims: Vec<usize>,
    /// `steps[b * n_lines + line]`: `sign * (value(b+2) - value(b+1))` on every line, when the
    /// steps are known cell by cell (`None`: a factored effect too large to evaluate).
    steps: Option<Vec<f64>>,
    /// A sound lower bound of the step at every boundary (the exact minimum over the lines when
    /// `steps` is known).
    floor: Vec<f64>,
    /// Lookups one evaluation of this table costs during enumeration.
    cost: u64,
}

impl Steps {
    fn n_lines(&self) -> usize {
        self.dims.iter().product()
    }
}

struct Family {
    feature: u32,
    sign: f64,
    increasing: bool,
    /// Index of the main effect in `bank.tables`.
    main: usize,
    /// Indices of the dense interactions in `bank.tables`.
    dense: Vec<usize>,
    /// Indices of the factored interactions in `bank.factored`.
    factored: Vec<usize>,
    /// Finite merged cells `1..n` give `n - 2` boundaries.
    nb: usize,
}

/// The steps of a dense interaction `t` along `fam`'s feature.
fn dense_steps(t: &EffectTable, fam: &Family, bank: &TableBank) -> Result<Steps, PbError> {
    let p =
        t.u.0
            .iter()
            .position(|f| f.0 == fam.feature)
            .ok_or_else(internal("family table without its feature"))?;
    let shape = t.values.shape();
    let vals = t.values.try_values()?;
    let st = strides(&shape);
    let sp = *st.get(p).ok_or_else(internal("step axis"))?;
    let axis = t.axes.get(p).ok_or_else(internal("step axis id"))?;
    let jmap = coord_map(axis, n_cells(bank, fam.feature)?)?;
    let bases = line_bases(&shape, p)?;
    let n_lines = bases.len();
    let mut steps = vec![0.0_f64; fam.nb.saturating_mul(n_lines)];
    let mut floor = vec![f64::INFINITY; fam.nb];
    for b in 0..fam.nb {
        let lo = *jmap.get(b + 1).ok_or_else(internal("step low cell"))?;
        let hi = *jmap.get(b + 2).ok_or_else(internal("step high cell"))?;
        let fl = floor.get_mut(b).ok_or_else(internal("step floor"))?;
        for (line, &base) in bases.iter().enumerate() {
            let d = if lo == hi {
                0.0
            } else {
                let vh = vals
                    .get(base + hi * sp)
                    .ok_or_else(internal("step high value"))?;
                let vl = vals
                    .get(base + lo * sp)
                    .ok_or_else(internal("step low value"))?;
                fam.sign * (vh - vl)
            };
            *steps
                .get_mut(b * n_lines + line)
                .ok_or_else(internal("step slot"))? = d;
            *fl = fl.min(d);
        }
    }
    let mut partners = Vec::new();
    let mut maps = Vec::new();
    let mut dims = Vec::new();
    for (d, (a, &dim)) in t.axes.iter().zip(&shape).enumerate() {
        if d == p {
            continue;
        }
        partners.push(a.raw.0);
        maps.push(coord_map(a, n_cells(bank, a.raw.0)?)?);
        dims.push(dim);
    }
    Ok(Steps {
        partners,
        maps,
        dims,
        steps: Some(steps),
        floor: floor
            .into_iter()
            .map(|f| if f.is_finite() { f } else { 0.0 })
            .collect(),
        cost: 1,
    })
}

/// The steps of a factored interaction: exact cell by cell when that is affordable, otherwise a
/// sound per-box bound (each box's worst step over the sides its other axes can take, summed).
fn factored_steps(
    fe: &FactoredEffect,
    fam: &Family,
    bank: &TableBank,
    cap: u64,
) -> Result<Steps, PbError> {
    let p =
        fe.u.0
            .iter()
            .position(|f| f.0 == fam.feature)
            .ok_or_else(internal("factored family table without its feature"))?;
    let mut partners = Vec::new();
    let mut maps = Vec::new();
    let mut dims = Vec::new();
    for (d, a) in fe.axes.iter().enumerate() {
        if d == p {
            continue;
        }
        let n = n_cells(bank, a.raw.0)?;
        partners.push(a.raw.0);
        maps.push((0..n).collect::<Vec<usize>>());
        dims.push(n);
    }
    let boxes = u64::try_from(fe.n_boxes()).unwrap_or(u64::MAX).max(1);
    // A line count that does not even fit a `usize` is past any cap: take the per-box bound.
    let n_lines = line_count(&dims);
    let work = n_lines.map_or(u64::MAX, |n| {
        (fam.nb as u64)
            .saturating_mul(u64::try_from(n).unwrap_or(u64::MAX))
            .saturating_mul(boxes)
            .saturating_mul(2)
    });
    if let Some(n_lines) = n_lines.filter(|_| work <= cap) {
        let n_raw = bank.merged_grids.len();
        let mut x = vec![0u32; n_raw];
        let mut steps = vec![0.0_f64; fam.nb.saturating_mul(n_lines)];
        let mut floor = vec![f64::INFINITY; fam.nb];
        let st = strides(&dims);
        for line in 0..n_lines {
            for (&raw, (&s, &dim)) in partners.iter().zip(st.iter().zip(&dims)) {
                *x.get_mut(raw as usize)
                    .ok_or_else(internal("factored partner"))? =
                    u32::try_from((line / s) % dim.max(1)).unwrap_or(u32::MAX);
            }
            for b in 0..fam.nb {
                let jslot = x
                    .get_mut(fam.feature as usize)
                    .ok_or_else(internal("factored j"))?;
                *jslot = u32::try_from(b + 1).unwrap_or(u32::MAX);
                let lo = fe.eval(&x)?;
                let jslot = x
                    .get_mut(fam.feature as usize)
                    .ok_or_else(internal("factored j"))?;
                *jslot = u32::try_from(b + 2).unwrap_or(u32::MAX);
                let hi = fe.eval(&x)?;
                let d = fam.sign * (hi - lo);
                *steps
                    .get_mut(b * n_lines + line)
                    .ok_or_else(internal("factored slot"))? = d;
                let fl = floor.get_mut(b).ok_or_else(internal("factored floor"))?;
                *fl = fl.min(d);
            }
        }
        return Ok(Steps {
            partners,
            maps,
            dims,
            steps: Some(steps),
            floor: floor
                .into_iter()
                .map(|f| if f.is_finite() { f } else { 0.0 })
                .collect(),
            cost: 1,
        });
    }
    // Per-box bound: a box's value is a corner of its low/high sides; along the constrained axis
    // it changes only where its own split falls, and the worst it can then do is the worst corner
    // difference over the sides its other axes can actually take.
    let mut floor = vec![0.0_f64; fam.nb];
    for (corners, low) in fe.box_parts() {
        let jl = low.get(p).ok_or_else(internal("box constrained axis"))?;
        let sides: Vec<Vec<bool>> = low
            .iter()
            .enumerate()
            .filter(|&(d, _)| d != p)
            .map(|(_, m)| {
                let mut s = Vec::new();
                if m.iter().any(|&l| !l) {
                    s.push(false);
                }
                if m.iter().any(|&l| l) {
                    s.push(true);
                }
                s
            })
            .collect();
        let others: Vec<usize> = (0..low.len()).filter(|&d| d != p).collect();
        for (b, fl) in floor.iter_mut().enumerate() {
            let slo = *jl.get(b + 1).ok_or_else(internal("box low side"))?;
            let shi = *jl.get(b + 2).ok_or_else(internal("box high side"))?;
            if slo == shi {
                continue;
            }
            let mut worst = f64::INFINITY;
            let combos: usize = sides.iter().map(Vec::len).product();
            for combo in 0..combos {
                let mut idx_lo = usize::from(slo) << p;
                let mut idx_hi = usize::from(shi) << p;
                let mut rem = combo;
                for (side, &d) in sides.iter().zip(&others) {
                    let pick = *side
                        .get(rem % side.len().max(1))
                        .ok_or_else(internal("side"))?;
                    rem /= side.len().max(1);
                    idx_lo |= usize::from(pick) << d;
                    idx_hi |= usize::from(pick) << d;
                }
                let vh = corners.get(idx_hi).ok_or_else(internal("box corner"))?;
                let vl = corners.get(idx_lo).ok_or_else(internal("box corner"))?;
                worst = worst.min(fam.sign * (vh - vl));
            }
            if worst.is_finite() {
                *fl += worst;
            }
        }
    }
    Ok(Steps {
        partners,
        maps,
        dims,
        steps: None,
        floor,
        cost: boxes,
    })
}

/// The interaction envelope of a family: per boundary, the smallest total step its interactions
/// take over every combination of the other features. Returns it and whether it is exact.
fn envelope(fam: &Family, tables: &[Steps], cap: u64) -> Result<(Vec<f64>, bool), PbError> {
    let mut env = vec![0.0_f64; fam.nb];
    let mut exact = true;
    // Components: tables that share a partner must be enumerated together.
    let mut comp_of: Vec<usize> = (0..tables.len()).collect();
    for i in 0..tables.len() {
        for k in 0..i {
            let share = tables
                .get(i)
                .zip(tables.get(k))
                .is_some_and(|(a, b)| a.partners.iter().any(|p| b.partners.contains(p)));
            if share {
                let (ci, ck) = (comp_of.get(i).copied(), comp_of.get(k).copied());
                if let (Some(ci), Some(ck)) = (ci, ck) {
                    let (from, to) = (ci.max(ck), ci.min(ck));
                    for c in comp_of.iter_mut() {
                        if *c == from {
                            *c = to;
                        }
                    }
                }
            }
        }
    }
    let comps: BTreeSet<usize> = comp_of.iter().copied().collect();
    for comp in comps {
        let members: Vec<&Steps> = tables
            .iter()
            .zip(&comp_of)
            .filter(|&(_, &c)| c == comp)
            .map(|(t, _)| t)
            .collect();
        let partners: BTreeSet<u32> = members
            .iter()
            .flat_map(|t| t.partners.iter().copied())
            .collect();
        let partners: Vec<u32> = partners.into_iter().collect();
        let all_known = members.iter().all(|t| t.steps.is_some());
        if members.len() == 1 && all_known {
            // A single table reaches every own coordinate from some merged cell, so its own
            // minimum over the lines IS the exact envelope.
            if let Some(t) = members.first() {
                for (e, f) in env.iter_mut().zip(&t.floor) {
                    *e += f;
                }
            }
            continue;
        }
        // Exact enumeration over the merged cells of every partner, if affordable.
        let mut configs: u64 = 1;
        let mut part_cells = Vec::with_capacity(partners.len());
        for &r in &partners {
            let m = members
                .iter()
                .find_map(|t| {
                    t.partners
                        .iter()
                        .position(|&q| q == r)
                        .and_then(|i| t.maps.get(i))
                })
                .map(Vec::len)
                .ok_or_else(internal("component partner"))?;
            part_cells.push(m);
            configs = configs.saturating_mul(m as u64);
        }
        let cost: u64 = members.iter().map(|t| t.cost).sum();
        let work = configs.saturating_mul(fam.nb as u64).saturating_mul(cost);
        if !all_known || work > cap {
            exact = false;
            for t in &members {
                for (e, f) in env.iter_mut().zip(&t.floor) {
                    *e += f;
                }
            }
            continue;
        }
        // Per member: (position of each own partner in `partners`, its map, its stride).
        let layout: Vec<Vec<(usize, &Vec<usize>, usize)>> = members
            .iter()
            .map(|t| {
                let st = strides(&t.dims);
                t.partners
                    .iter()
                    .zip(&t.maps)
                    .zip(st)
                    .map(|((&r, map), s)| {
                        let pos = partners
                            .iter()
                            .position(|&q| q == r)
                            .ok_or_else(internal("cfg"))?;
                        Ok((pos, map, s))
                    })
                    .collect::<Result<Vec<_>, PbError>>()
            })
            .collect::<Result<_, _>>()?;
        let mut best = vec![f64::INFINITY; fam.nb];
        let mut cfg = vec![0usize; partners.len()];
        let mut lines = vec![0usize; members.len()];
        loop {
            // Each member's own line for this configuration.
            for (slot, lay) in lines.iter_mut().zip(&layout) {
                let mut line = 0usize;
                for &(pos, map, s) in lay {
                    let cell = *cfg.get(pos).ok_or_else(internal("cfg cell"))?;
                    line += map.get(cell).ok_or_else(internal("cfg coord"))? * s;
                }
                *slot = line;
            }
            for (b, bst) in best.iter_mut().enumerate() {
                let mut total = 0.0_f64;
                for (t, &line) in members.iter().zip(&lines) {
                    let steps = t.steps.as_ref().ok_or_else(internal("known steps"))?;
                    total += steps
                        .get(b * t.n_lines() + line)
                        .ok_or_else(internal("enum step"))?;
                }
                *bst = bst.min(total);
            }
            // Odometer over the partners' merged cells.
            let mut d = 0usize;
            while let Some(c) = cfg.get_mut(d) {
                *c += 1;
                if *c < part_cells.get(d).copied().unwrap_or(0) {
                    break;
                }
                *c = 0;
                d += 1;
            }
            if d >= cfg.len() {
                break;
            }
        }
        for (e, b) in env.iter_mut().zip(&best) {
            *e += if b.is_finite() { *b } else { 0.0 };
        }
    }
    Ok((env, exact))
}

/// The family's main-effect steps along its finite cells.
fn main_steps(t: &EffectTable, fam: &Family) -> Result<Vec<f64>, PbError> {
    let v = t.values.try_values()?;
    (0..fam.nb)
        .map(|b| {
            let lo = v.get(b + 1).ok_or_else(internal("main low"))?;
            let hi = v.get(b + 2).ok_or_else(internal("main high"))?;
            Ok(fam.sign * (hi - lo))
        })
        .collect()
}

/// Per-cell projection weights of a dense table: curvature mass of the rows that land in each
/// cell, plus [`WEIGHT_FLOOR`] of the total spread by the purification measure.
fn cell_weights(
    t: &EffectTable,
    bank: &TableBank,
    mu: &[Vec<f64>],
    rows: &RepairRows<'_>,
    h_total: f64,
) -> Result<Vec<f64>, PbError> {
    let shape = t.values.shape();
    let st = strides(&shape);
    let maps: Vec<Vec<usize>> = t
        .axes
        .iter()
        .map(|a| coord_map(a, n_cells(bank, a.raw.0)?))
        .collect::<Result<_, _>>()?;
    let cols: Vec<&Vec<u32>> = t
        .axes
        .iter()
        .map(|a| {
            rows.cells
                .get(a.raw.0 as usize)
                .ok_or_else(internal("row cells"))
        })
        .collect::<Result<_, _>>()?;
    let mut w: Vec<f64> = table_measure(t, mu, bank)?
        .into_iter()
        .map(|m| WEIGHT_FLOOR * h_total * m)
        .collect();
    for (r, &h) in rows.h.iter().enumerate() {
        let mut flat = 0usize;
        for ((col, map), s) in cols.iter().zip(&maps).zip(&st) {
            let cell = *col.get(r).ok_or_else(internal("row cell"))? as usize;
            flat += map.get(cell).ok_or_else(internal("row coord"))? * s;
        }
        *w.get_mut(flat).ok_or_else(internal("row weight"))? += h;
    }
    // A cell no measure or row reaches still needs a positive weight.
    let tiny = w
        .iter()
        .copied()
        .filter(|&x| x > 0.0)
        .fold(f64::INFINITY, f64::min);
    let tiny = if tiny.is_finite() { tiny * 1e-6 } else { 1.0 };
    Ok(w.into_iter()
        .map(|x| if x > 0.0 { x } else { tiny })
        .collect())
}

/// Project a dense table's `vals` along `fam`'s feature onto the per-boundary `floors`.
fn project_along(
    t: &EffectTable,
    vals: &[f64],
    w: &[f64],
    fam: &Family,
    floors: &[Option<f64>],
    bank: &TableBank,
) -> Result<Vec<f64>, PbError> {
    let p =
        t.u.0
            .iter()
            .position(|f| f.0 == fam.feature)
            .ok_or_else(internal("projected table without its feature"))?;
    let shape = t.values.shape();
    let sp = *strides(&shape)
        .get(p)
        .ok_or_else(internal("projection axis"))?;
    let axis = t.axes.get(p).ok_or_else(internal("projection axis id"))?;
    let jmap = coord_map(axis, n_cells(bank, fam.feature)?)?;
    // The finite coordinates in order (bands are contiguous runs; the missing cell is its own),
    // and the floor on each edge between consecutive ones.
    let mut coords: Vec<usize> = Vec::new();
    let mut edge_floor: Vec<Option<f64>> = Vec::new();
    for c in 1..jmap.len() {
        let coord = *jmap.get(c).ok_or_else(internal("finite coord"))?;
        if coords.last() != Some(&coord) {
            if !coords.is_empty() {
                // The edge into `coord` is crossed by merged boundary b = c - 2.
                edge_floor.push(floors.get(c - 2).copied().flatten());
            }
            coords.push(coord);
        }
    }
    let mut out = vals.to_vec();
    for base in line_bases(&shape, p)? {
        let idx: Vec<usize> = coords.iter().map(|&k| base + k * sp).collect();
        let v: Vec<f64> = idx
            .iter()
            .map(|&i| vals.get(i).copied().ok_or_else(internal("line value")))
            .collect::<Result<_, _>>()?;
        let lw: Vec<f64> = idx
            .iter()
            .map(|&i| w.get(i).copied().ok_or_else(internal("line weight")))
            .collect::<Result<_, _>>()?;
        let fitted = shifted_pava(&v, &lw, &edge_floor, fam.increasing)?;
        for (&i, x) in idx.iter().zip(fitted) {
            *out.get_mut(i).ok_or_else(internal("line write"))? = x;
        }
    }
    Ok(out)
}

/// Every step array of `fam`'s interactions in `bank`.
fn family_steps(fam: &Family, bank: &TableBank, cap: u64) -> Result<Vec<Steps>, PbError> {
    let mut out = Vec::with_capacity(fam.dense.len() + fam.factored.len());
    for &i in &fam.dense {
        out.push(dense_steps(
            bank.tables
                .get(i)
                .ok_or_else(internal("dense family table"))?,
            fam,
            bank,
        )?);
    }
    for &i in &fam.factored {
        out.push(factored_steps(
            bank.factored
                .get(i)
                .ok_or_else(internal("factored family table"))?,
            fam,
            bank,
            cap,
        )?);
    }
    Ok(out)
}

/// `(worst step, exact)` of a family: the most negative `main step + envelope` over its
/// boundaries (`f64::INFINITY` when it has none).
fn family_worst(fam: &Family, bank: &TableBank, cap: u64) -> Result<(f64, bool), PbError> {
    let steps = family_steps(fam, bank, cap)?;
    let (env, exact) = envelope(fam, &steps, cap)?;
    let main = bank
        .tables
        .get(fam.main)
        .ok_or_else(internal("main table"))?;
    let ms = main_steps(main, fam)?;
    let worst = ms
        .iter()
        .zip(&env)
        .map(|(m, e)| m + e)
        .fold(f64::INFINITY, f64::min);
    Ok((worst, exact))
}

/// Re-purify the `modified` dense tables together with every table below them, in place. A
/// marginal moved into a lower-order table never changes a family's steps along its own feature:
/// either the lower table is still in the family, or the moved piece is constant along it.
fn repurify_modified(out: &mut TableBank, modified: &BTreeSet<usize>) -> Result<(), PbError> {
    let tops: Vec<FeatureSet> = modified
        .iter()
        .map(|&i| {
            out.tables
                .get(i)
                .map(|t| t.u.clone())
                .ok_or_else(internal("modified table"))
        })
        .collect::<Result<_, _>>()?;
    let below = |u: &FeatureSet| tops.iter().any(|m| u.0.iter().all(|f| m.contains(*f)));
    let sub = TableBank {
        f0: 0.0,
        tables: out.tables.iter().filter(|t| below(&t.u)).cloned().collect(),
        merged_grids: out.merged_grids.clone(),
        w: out.w.clone(),
        factored: Vec::new(),
        joint_variance: None,
    };
    let pure = crate::banding::repurify_bank(&sub)?;
    out.f0 += pure.f0;
    for t in pure.tables {
        match out.tables.iter_mut().find(|o| o.u == t.u) {
            Some(slot) => {
                // The display-only error band still describes the bag spread when the grid kept
                // its shape (see the module doc of the report's `se_band` decision).
                let se = if slot.values.shape() == t.values.shape() {
                    slot.se_band.take()
                } else {
                    None
                };
                *slot = t;
                slot.se_band = se;
            }
            None => out.tables.push(t),
        }
    }
    Ok(())
}

/// Project every constrained feature's family of `bank` back onto its sign (see the module
/// doc). Returns the repaired bank and the report; a bank whose families all pass comes back
/// untouched with `certified == true`.
///
/// # Errors
/// [`PbError::Internal`] if a constrained feature has interactions but no main effect, a banded
/// main effect, or a family that the repair could not make monotone; propagated errors otherwise.
pub fn repair_bank(
    bank: &TableBank,
    signs: &[MonotoneSign],
    rows: &RepairRows<'_>,
) -> Result<(TableBank, MonotoneReport), PbError> {
    repair_bank_with_cap(bank, signs, rows, EXACT_WORK_CAP)
}

/// [`repair_bank`] with the exact-envelope work cap as a parameter (the tests lower it to reach
/// the bound paths).
fn repair_bank_with_cap(
    bank: &TableBank,
    signs: &[MonotoneSign],
    rows: &RepairRows<'_>,
    cap: u64,
) -> Result<(TableBank, MonotoneReport), PbError> {
    let mut fams: Vec<Family> = Vec::new();
    let mut reports: Vec<FeatureReport> = Vec::new();
    for sg in signs {
        let fid = FeatureId(sg.feature);
        let mut tables: Vec<Vec<u32>> = bank
            .tables
            .iter()
            .map(|t| &t.u)
            .chain(bank.factored.iter().map(|f| &f.u))
            .filter(|u| u.contains(fid))
            .map(|u| u.0.iter().map(|f| f.0).collect())
            .collect();
        tables.sort();
        let main = bank
            .tables
            .iter()
            .position(|t| t.u.order() == 1 && t.u.contains(fid));
        let dense: Vec<usize> = (0..bank.tables.len())
            .filter(|&i| {
                bank.tables
                    .get(i)
                    .is_some_and(|t| t.u.order() > 1 && t.u.contains(fid))
            })
            .collect();
        let factored: Vec<usize> = (0..bank.factored.len())
            .filter(|&i| bank.factored.get(i).is_some_and(|t| t.u.contains(fid)))
            .collect();
        let report = FeatureReport {
            feature: sg.feature,
            sign: if sg.increasing { 1 } else { -1 },
            tables,
            certificate: "exact".into(),
            worst_step_before: 0.0,
            worst_step_after: 0.0,
            main_change: 0.0,
            clipped: Vec::new(),
            clip_change: 0.0,
        };
        let Some(main) = main else {
            if !dense.is_empty() || !factored.is_empty() {
                return Err(internal(
                    "a constrained feature has interactions but no main effect",
                )());
            }
            reports.push(report); // unused by the model: trivially monotone
            continue;
        };
        let main_t = bank.tables.get(main).ok_or_else(internal("main table"))?;
        if main_t.axes.first().is_some_and(|a| a.band_of.is_some()) {
            return Err(internal("a constrained main effect is banded")());
        }
        fams.push(Family {
            feature: sg.feature,
            sign: if sg.increasing { 1.0 } else { -1.0 },
            increasing: sg.increasing,
            main,
            dense,
            factored,
            nb: n_cells(bank, sg.feature)?.saturating_sub(2),
        });
        reports.push(report);
    }
    let report_of = |reports: &mut Vec<FeatureReport>, feature: u32| -> Result<usize, PbError> {
        reports
            .iter()
            .position(|r| r.feature == feature)
            .ok_or_else(internal("feature report"))
    };

    // Certificate: a bank whose every family passes is returned untouched.
    let mut all_pass = true;
    for fam in &fams {
        let (worst, exact) = family_worst(fam, bank, cap)?;
        let r = report_of(&mut reports, fam.feature)?;
        let rep = reports.get_mut(r).ok_or_else(internal("report"))?;
        rep.worst_step_before = if worst.is_finite() { worst } else { 0.0 };
        rep.worst_step_after = rep.worst_step_before;
        if !exact {
            rep.certificate = "bound".into();
        }
        all_pass &= worst >= -STEP_TOL;
    }
    if all_pass {
        return Ok((
            bank.clone(),
            MonotoneReport {
                features: reports,
                certified: true,
            },
        ));
    }

    let mu = axis_measure(bank)?;
    let h_total: f64 = rows.h.iter().sum();
    let mut out = bank.clone();
    let old_mains: Vec<Vec<f64>> = fams
        .iter()
        .map(|f| {
            out.tables
                .get(f.main)
                .ok_or_else(internal("main"))
                .and_then(|t| Ok(t.values.try_values()?.into_owned()))
        })
        .collect::<Result<_, _>>()?;
    let main_w: Vec<Vec<f64>> = fams
        .iter()
        .map(|f| {
            cell_weights(
                out.tables.get(f.main).ok_or_else(internal("main"))?,
                &out,
                &mu,
                rows,
                h_total,
            )
        })
        .collect::<Result<_, _>>()?;

    // Phase 1: every constrained main effect isotonic.
    for (fam, w) in fams.iter().zip(&main_w) {
        let t = out.tables.get_mut(fam.main).ok_or_else(internal("main"))?;
        let mut v = t.values.try_values()?.into_owned();
        let (Some(fin), Some(fw)) = (v.get(1..), w.get(1..)) else {
            continue;
        };
        let fitted = shifted_pava(fin, fw, &vec![Some(0.0); fam.nb], fam.increasing)?;
        for (slot, x) in v.iter_mut().skip(1).zip(fitted) {
            *slot = x;
        }
        t.values = crate::explain::Tensor::from_vec(t.values.shape(), v)?;
    }

    // Phase 2a: floors, sharing each main effect's slope between its interactions by need.
    let mut floors: BTreeMap<usize, Vec<(usize, Floors)>> = BTreeMap::new();
    for (fi, fam) in fams.iter().enumerate() {
        let steps = family_steps(fam, &out, cap)?;
        let (env, _) = envelope(fam, &steps, cap)?;
        let main = out.tables.get(fam.main).ok_or_else(internal("main"))?;
        let beta = main_steps(main, fam)?;
        let n_dense = fam.dense.len();
        let need = |s: &Steps, b: usize| s.floor.get(b).map_or(0.0, |&f| (-f).max(0.0));
        for (k, &ti) in fam.dense.iter().enumerate() {
            let mut fl = vec![None; fam.nb];
            let mut any = false;
            for (b, slot) in fl.iter_mut().enumerate() {
                let (bb, e) = (
                    beta.get(b).copied().unwrap_or(0.0),
                    env.get(b).copied().unwrap_or(0.0),
                );
                if bb + e >= -STEP_TOL {
                    continue;
                }
                let nf: f64 = steps.iter().skip(n_dense).map(|s| need(s, b)).sum();
                let nd: f64 = steps.iter().take(n_dense).map(|s| need(s, b)).sum();
                let mine = steps.get(k).map_or(0.0, |s| need(s, b));
                if nd > 0.0 && mine > 0.0 {
                    let budget = (bb - nf).max(0.0);
                    *slot = Some(-mine * budget / nd);
                    any = true;
                }
            }
            if any {
                floors.entry(ti).or_default().push((fi, fl));
            }
        }
    }

    // Phase 2b: clip, alternating the projections on a table shared by several features.
    let mut modified: BTreeSet<usize> = BTreeSet::new();
    for (&ti, specs) in &floors {
        let t = out.tables.get(ti).ok_or_else(internal("clipped table"))?;
        let w = cell_weights(t, &out, &mu, rows, h_total)?;
        let start = t.values.try_values()?.into_owned();
        let mut x = start.clone();
        let mut inc = vec![vec![0.0_f64; x.len()]; specs.len()];
        for _ in 0..if specs.len() > 1 { SHARED_ROUNDS } else { 1 } {
            let mut moved = 0.0_f64;
            for ((fi, fl), inc_i) in specs.iter().zip(inc.iter_mut()) {
                let fam = fams.get(*fi).ok_or_else(internal("clip family"))?;
                let z: Vec<f64> = x.iter().zip(inc_i.iter()).map(|(a, b)| a + b).collect();
                let y = project_along(t, &z, &w, fam, fl, &out)?;
                for ((d, zi), yi) in inc_i.iter_mut().zip(&z).zip(&y) {
                    *d = zi - yi;
                }
                moved = x
                    .iter()
                    .zip(&y)
                    .map(|(a, b)| (a - b).abs())
                    .fold(moved, f64::max);
                x = y;
            }
            if moved <= 1e-12 {
                break;
            }
        }
        let change = start
            .iter()
            .zip(&x)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let support: Vec<u32> = t.u.0.iter().map(|f| f.0).collect();
        for (fi, _) in specs {
            let fam = fams.get(*fi).ok_or_else(internal("clip family"))?;
            let r = report_of(&mut reports, fam.feature)?;
            let rep = reports.get_mut(r).ok_or_else(internal("report"))?;
            rep.clipped.push(support.clone());
            rep.clip_change = rep.clip_change.max(change);
        }
        let shape = t.values.shape();
        let t = out
            .tables
            .get_mut(ti)
            .ok_or_else(internal("clipped table"))?;
        t.values = crate::explain::Tensor::from_vec(shape, x)?;
        modified.insert(ti);
    }

    // Phase 3: re-purify what was clipped (and everything below it).
    if !modified.is_empty() {
        repurify_modified(&mut out, &modified)?;
    }

    // Phase 4: project every main effect onto its final interaction envelope.
    for (fam, w) in fams.iter().zip(&main_w) {
        let steps = family_steps(fam, &out, cap)?;
        let (env, _) = envelope(fam, &steps, cap)?;
        let t = out.tables.get_mut(fam.main).ok_or_else(internal("main"))?;
        let mut v = t.values.try_values()?.into_owned();
        let (Some(fin), Some(fw)) = (v.get(1..), w.get(1..)) else {
            continue;
        };
        let fl: Vec<Option<f64>> = env.iter().map(|&e| Some(-e)).collect();
        let fitted = shifted_pava(fin, fw, &fl, fam.increasing)?;
        for (slot, x) in v.iter_mut().skip(1).zip(fitted) {
            *slot = x;
        }
        t.values = crate::explain::Tensor::from_vec(t.values.shape(), v)?;
    }

    // Phase 5: re-centre the main effects under the measure and recompute their variances.
    for (fam, old) in fams.iter().zip(&old_mains) {
        let raw_mu = mu
            .get(fam.feature as usize)
            .ok_or_else(internal("main measure"))?;
        let t = out.tables.get_mut(fam.main).ok_or_else(internal("main"))?;
        let mut v = t.values.try_values()?.into_owned();
        let mean: f64 = v.iter().zip(raw_mu).map(|(x, m)| x * m).sum();
        for x in v.iter_mut() {
            *x -= mean;
        }
        t.variance = v.iter().zip(raw_mu).map(|(x, m)| m * x * x).sum();
        let change = v
            .iter()
            .zip(old)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        t.values = crate::explain::Tensor::from_vec(t.values.shape(), v)?;
        out.f0 += mean;
        let r = report_of(&mut reports, fam.feature)?;
        reports
            .get_mut(r)
            .ok_or_else(internal("report"))?
            .main_change = change;
    }

    // Verify: every family must now pass its certificate.
    for fam in &fams {
        let (worst, _) = family_worst(fam, &out, cap)?;
        if worst < -STEP_TOL {
            return Err(PbError::Internal {
                what: format!(
                    "monotone repair left feature {} stepping the wrong way by {worst:e}",
                    fam.feature
                ),
            });
        }
        let r = report_of(&mut reports, fam.feature)?;
        reports
            .get_mut(r)
            .ok_or_else(internal("report"))?
            .worst_step_after = if worst.is_finite() { worst } else { 0.0 };
    }
    Ok((
        out,
        MonotoneReport {
            features: reports,
            certified: false,
        },
    ))
}

/// [`repair_bank`] on a deployed [`TableModel`] trained on `x` (serve-binned), with the intercept
/// re-anchored on the training rows afterwards when `reanchor` is set (log and logit links, the
/// same rule as `prune::prune_model_to_keepset`). A certified bank returns the model unchanged.
///
/// # Errors
/// Propagated from [`repair_bank`], scoring, the re-anchor and [`TableModel::validate`].
#[allow(clippy::too_many_arguments)] // JUSTIFIED: one argument per training input (§12.2).
pub fn enforce_monotone(
    model: &TableModel,
    x: &BinnedMatrix,
    y: &[f32],
    w: &[f32],
    offset: Option<&[f32]>,
    loss: &dyn Loss,
    signs: &[MonotoneSign],
    h: &[f64],
    reanchor: bool,
) -> Result<(TableModel, MonotoneReport), PbError> {
    let cells = model.column_cells(x)?;
    let (mut bank, report) = repair_bank(&model.bank, signs, &RepairRows { cells: &cells, h })?;
    if report.certified {
        return Ok((model.clone(), report));
    }
    if reanchor {
        let n = x.n_rows as usize;
        let mut raw = vec![0.0_f64; n];
        crate::scoring::score_bank_binned(&bank, &model.schema.cat_encoders, x, &mut raw)?;
        match loss.link() {
            Link::Log => bank.f0 += crate::prune::reanchor_shift(&raw, y, w, offset, loss)?,
            Link::Logit => {
                let raw32: Vec<f32> = raw.iter().map(|&v| v as f32).collect();
                bank.f0 += crate::engine::boost::reanchor_delta(Link::Logit, y, w, &raw32, offset)?;
            }
            Link::Identity => {}
        }
    }
    let mut tm = model.clone();
    tm.bank = bank;
    tm.validate()?;
    Ok((tm, report))
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
    use crate::constraints::{CredibilityFloor, InteractionPolicy, MonoSign, MonotoneMap};
    use crate::data::{bin_columns, BinConfig, ServeBinnedMatrix};
    use crate::engine::{Booster, Config, FitSpec};
    use crate::explain::{RefMeasure, TableBudget, Tensor};
    use crate::loss::SquaredError;

    fn xs(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    struct Fixture {
        bank: TableBank,
        cells: Vec<Vec<u32>>,
        h: Vec<f64>,
    }

    /// A squared-error fit of `f` on small grids (so brute force can sweep every cell), with
    /// `signs` enforced by the trees, and its full purified bank.
    fn fit(
        n_feat: usize,
        f: impl Fn(&[f64]) -> f64,
        signs: &[(usize, bool)],
        order: u8,
        budget: TableBudget,
    ) -> Fixture {
        let n = 3000;
        let mut s = 0x5eed_u64;
        let mut cols: Vec<Vec<f32>> = vec![Vec::with_capacity(n); n_feat];
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let x: Vec<f64> = (0..n_feat).map(|_| xs(&mut s) * 4.0 - 2.0).collect();
            for (c, v) in cols.iter_mut().zip(&x) {
                c.push(*v as f32);
            }
            y.push((f(&x) + 0.3 * (xs(&mut s) - 0.5)) as f32);
        }
        let refs: Vec<&[f32]> = cols.iter().map(Vec::as_slice).collect();
        let cfg_bins = BinConfig {
            max_bin: 12,
            ..BinConfig::default()
        };
        let x = bin_columns(&refs, None, &cfg_bins, 0).unwrap();
        let mut monotone = MonotoneMap::new();
        for &(i, inc) in signs {
            let sign = if inc {
                MonoSign::Increasing
            } else {
                MonoSign::Decreasing
            };
            monotone.insert(format!("f{i}"), sign);
        }
        let spec = FitSpec {
            loss: &SquaredError,
            weight: None,
            exposure: None,
            monotone,
            interaction: InteractionPolicy {
                max_order: order,
                ..InteractionPolicy::default()
            },
            credibility: CredibilityFloor::default(),
            fixed_holdout: None,
            bag_groups: None,
            seed: 0,
        };
        let cfg = Config {
            n_trees: 120,
            learning_rate: 0.3,
            validation_fraction: None,
            interaction_gain_hurdle: 0.0,
            ..Config::default()
        };
        let model = Booster::with_config(cfg).fit(&x, &y, &spec).unwrap();
        let serve = ServeBinnedMatrix(x);
        let bank = model
            .explain_bank(&serve, RefMeasure::default(), budget, false)
            .unwrap();
        let tm = TableModel::from_model_and_bank(&model, bank.clone());
        let cells = tm.column_cells(&serve.0).unwrap();
        Fixture {
            bank,
            cells,
            h: vec![1.0; n],
        }
    }

    fn value(bank: &TableBank, x: &[u32]) -> f64 {
        bank.f0
            + bank.tables.iter().map(|t| t.eval(x).unwrap()).sum::<f64>()
            + bank
                .factored
                .iter()
                .map(|t| t.eval(x).unwrap())
                .sum::<f64>()
    }

    /// The most negative step along `j` (sign applied) over EVERY combination of the other
    /// features' merged cells, missing cells included.
    fn worst_step(bank: &TableBank, j: u32, increasing: bool) -> f64 {
        let dims: Vec<usize> = bank
            .merged_grids
            .iter()
            .map(|g| usize::from(g.n_bins))
            .collect();
        let s = if increasing { 1.0 } else { -1.0 };
        let others: Vec<usize> = (0..dims.len()).filter(|&r| r != j as usize).collect();
        let mut x = vec![0u32; dims.len()];
        let mut worst = f64::INFINITY;
        let total: usize = others.iter().map(|&r| dims[r]).product();
        for mut code in 0..total {
            for &r in &others {
                x[r] = (code % dims[r]) as u32;
                code /= dims[r];
            }
            let mut prev = None;
            for c in 1..dims[j as usize] {
                x[j as usize] = c as u32;
                let v = value(bank, &x);
                if let Some(p) = prev {
                    worst = worst.min(s * (v - p));
                }
                prev = Some(v);
            }
        }
        worst
    }

    /// Every table is pure along every axis under the measure, and its stored variance matches.
    fn assert_pure(bank: &TableBank) {
        let mu = axis_measure(bank).unwrap();
        for t in &bank.tables {
            let shape = t.values.shape();
            let v = t.values.try_values().unwrap();
            let st = strides(&shape);
            for (p, a) in t.axes.iter().enumerate() {
                let map = coord_map(a, n_cells(bank, a.raw.0).unwrap()).unwrap();
                let mut m = vec![0.0; shape[p]];
                for (c, &k) in map.iter().enumerate() {
                    m[k] += mu[a.raw.0 as usize][c];
                }
                for base in line_bases(&shape, p).unwrap() {
                    let mean: f64 = (0..shape[p]).map(|k| m[k] * v[base + k * st[p]]).sum();
                    assert!(
                        mean.abs() < 1e-9,
                        "table {:?} impure along axis {p}: {mean:e}",
                        t.u
                    );
                }
            }
            let cm = table_measure(t, &mu, bank).unwrap();
            let var: f64 = cm.iter().zip(v.iter()).map(|(m, x)| m * x * x).sum();
            assert!(
                (var - t.variance).abs() <= 1e-9 * (1.0 + var),
                "stale variance on {:?}",
                t.u
            );
        }
    }

    fn sign(feature: u32, increasing: bool) -> MonotoneSign {
        MonotoneSign {
            feature,
            increasing,
        }
    }

    fn find(bank: &TableBank, u: &[u32]) -> usize {
        bank.tables
            .iter()
            .position(|t| t.u == FeatureSet::new(u))
            .unwrap()
    }

    /// Add a pure (centred, rank-1) sign-alternating pattern to a dense table, the way a
    /// value-rewriting stage leaves a pure but non-monotone interaction. Factored effects make
    /// `repurify_bank` unusable on the whole bank, so the pattern is built pure instead.
    fn perturb(bank: &mut TableBank, u: &[u32], size: f64) {
        let mu = axis_measure(bank).unwrap();
        let i = find(bank, u);
        let t = &mut bank.tables[i];
        let shape = t.values.shape();
        let pat: Vec<Vec<f64>> = t
            .axes
            .iter()
            .zip(&shape)
            .map(|(a, &d)| {
                let mut m = vec![0.0; d];
                for (c, &w) in mu[a.raw.0 as usize].iter().enumerate() {
                    m[a.coord(c as u32).unwrap()] += w;
                }
                let alt: Vec<f64> = (0..d)
                    .map(|k| if k % 2 == 0 { 1.0 } else { -1.0 })
                    .collect();
                let mean: f64 = alt.iter().zip(&m).map(|(x, w)| x * w).sum();
                alt.into_iter().map(|x| x - mean).collect()
            })
            .collect();
        let st = strides(&shape);
        let mut v = t.values.try_values().unwrap().into_owned();
        for (f, x) in v.iter_mut().enumerate() {
            let mut p = size;
            for (d, pd) in pat.iter().enumerate() {
                p *= pd[(f / st[d]) % shape[d]];
            }
            *x += p;
        }
        t.values = Tensor::from_vec(shape, v).unwrap();
    }

    fn brute_isotonic(y: &[f64], w: &[f64]) -> Vec<f64> {
        // x_i = max_{a<=i} min_{b>=i} weighted mean of y[a..=b]
        let n = y.len();
        (0..n)
            .map(|i| {
                (0..=i)
                    .map(|a| {
                        (i..n)
                            .map(|b| {
                                let ws: f64 = w[a..=b].iter().sum();
                                y[a..=b]
                                    .iter()
                                    .zip(&w[a..=b])
                                    .map(|(y, w)| y * w)
                                    .sum::<f64>()
                                    / ws
                            })
                            .fold(f64::INFINITY, f64::min)
                    })
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect()
    }

    #[test]
    fn pava_matches_the_minmax_formula() {
        let mut s = 7u64;
        for n in 1..9 {
            for _ in 0..40 {
                let y: Vec<f64> = (0..n).map(|_| xs(&mut s) * 4.0 - 2.0).collect();
                let w: Vec<f64> = (0..n).map(|_| 0.1 + xs(&mut s)).collect();
                let got = pava(&y, &w);
                let want = brute_isotonic(&y, &w);
                for (g, e) in got.iter().zip(&want) {
                    assert!((g - e).abs() < 1e-12, "{got:?} vs {want:?}");
                }
            }
        }
    }

    #[test]
    fn shifted_pava_meets_its_floors_and_leaves_a_feasible_input_alone() {
        let mut s = 11u64;
        for _ in 0..200 {
            let n = 2 + (xs(&mut s) * 7.0) as usize;
            let v: Vec<f64> = (0..n).map(|_| xs(&mut s) * 4.0 - 2.0).collect();
            let w: Vec<f64> = (0..n).map(|_| 0.1 + xs(&mut s)).collect();
            let fl: Vec<Option<f64>> = (0..n - 1)
                .map(|_| (xs(&mut s) < 0.7).then(|| xs(&mut s) - 0.7))
                .collect();
            for inc in [true, false] {
                let sg = if inc { 1.0 } else { -1.0 };
                let out = shifted_pava(&v, &w, &fl, inc).unwrap();
                for (k, f) in fl.iter().enumerate() {
                    if let Some(f) = f {
                        assert!(sg * (out[k + 1] - out[k]) >= f - 1e-12);
                    }
                }
                // Idempotent: a feasible input comes back unchanged.
                let again = shifted_pava(&out, &w, &fl, inc).unwrap();
                for (a, b) in out.iter().zip(&again) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn shifted_pava_refuses_bad_inputs() {
        assert!(shifted_pava(&[1.0, 2.0], &[1.0], &[None], true).is_err());
        assert!(shifted_pava(&[1.0, 2.0], &[1.0, 0.0], &[None], true).is_err());
        assert!(shifted_pava(&[1.0, 2.0], &[1.0, 1.0], &[], true).is_err());
    }

    fn pair_target(x: &[f64]) -> f64 {
        2.0 * x[0].tanh() + 0.8 * x[0] * x[1] + 0.6 * (1.5 * x[1]).sin() + 0.3 * x[2]
    }

    #[test]
    fn a_monotone_full_bank_is_certified_and_returned_untouched() {
        let fx = fit(3, pair_target, &[(0, true)], 3, TableBudget::default());
        assert!(
            worst_step(&fx.bank, 0, true) >= -1e-9,
            "the trees must enforce the sign"
        );
        let rows = RepairRows {
            cells: &fx.cells,
            h: &fx.h,
        };
        let (out, rep) = repair_bank(&fx.bank, &[sign(0, true)], &rows).unwrap();
        assert!(rep.certified);
        assert_eq!(out, fx.bank);
    }

    #[test]
    fn repair_restores_a_family_broken_the_way_the_pipeline_breaks_it() {
        for inc in [true, false] {
            let f = |x: &[f64]| if inc { pair_target(x) } else { -pair_target(x) };
            let mut fx = fit(3, f, &[(0, inc)], 3, TableBudget::default());
            // Drop the 3-way table, perturb a pair, and put a dip in the main effect.
            fx.bank.tables.retain(|t| t.u.order() < 3);
            fx.bank.factored.clear();
            perturb(&mut fx.bank, &[0, 1], 0.4);
            let m = find(&fx.bank, &[0]);
            let mut v = fx.bank.tables[m].values.try_values().unwrap().into_owned();
            v[4] += if inc { -0.5 } else { 0.5 };
            fx.bank.tables[m].values = Tensor::from_vec(vec![v.len()], v).unwrap();
            assert!(
                worst_step(&fx.bank, 0, inc) < -0.1,
                "the fixture must be broken"
            );
            let rows = RepairRows {
                cells: &fx.cells,
                h: &fx.h,
            };
            let (out, rep) = repair_bank(&fx.bank, &[sign(0, inc)], &rows).unwrap();
            assert!(!rep.certified);
            assert!(
                worst_step(&out, 0, inc) >= -1e-9,
                "still not monotone ({inc})"
            );
            assert!(rep.features[0].worst_step_after >= -1e-9);
            assert!(
                !rep.features[0].clipped.is_empty(),
                "the pair needed clipping"
            );
            assert_pure(&out);
            // Deterministic.
            let (again, rep2) = repair_bank(&fx.bank, &[sign(0, inc)], &rows).unwrap();
            assert_eq!(out, again);
            assert_eq!(rep, rep2);
        }
    }

    #[test]
    fn a_clipped_table_keeps_its_error_band() {
        // The display-only `se_band` describes the bag spread of a table; re-purification drops
        // it, and the repair puts it back on every table whose grid kept its shape.
        let mut fx = fit(3, pair_target, &[(0, true)], 2, TableBudget::default());
        perturb(&mut fx.bank, &[0, 1], 0.4);
        let i = find(&fx.bank, &[0, 1]);
        let shape = fx.bank.tables[i].values.shape();
        let n: usize = shape.iter().product();
        let band = crate::explain::SeBand {
            per_cell: Tensor::from_vec(shape, (0..n).map(|k| 0.01 * k as f64).collect()).unwrap(),
        };
        fx.bank.tables[i].se_band = Some(band.clone());
        let rows = RepairRows {
            cells: &fx.cells,
            h: &fx.h,
        };
        let (out, rep) = repair_bank(&fx.bank, &[sign(0, true)], &rows).unwrap();
        assert!(
            rep.features[0].clipped.contains(&vec![0, 1]),
            "the pair must be clipped"
        );
        assert_eq!(out.tables[find(&out, &[0, 1])].se_band, Some(band));
    }

    #[test]
    fn two_constrained_features_sharing_a_table_both_end_monotone() {
        let f = |x: &[f64]| 2.0 * x[0].tanh() - 1.5 * x[1].tanh() + 0.7 * x[0] * x[1] + 0.2 * x[2];
        let mut fx = fit(3, f, &[(0, true), (1, false)], 2, TableBudget::default());
        perturb(&mut fx.bank, &[0, 1], 0.5);
        assert!(worst_step(&fx.bank, 0, true) < -0.1 || worst_step(&fx.bank, 1, false) < -0.1);
        let rows = RepairRows {
            cells: &fx.cells,
            h: &fx.h,
        };
        let signs = [sign(0, true), sign(1, false)];
        let (out, rep) = repair_bank(&fx.bank, &signs, &rows).unwrap();
        assert!(worst_step(&out, 0, true) >= -1e-9);
        assert!(worst_step(&out, 1, false) >= -1e-9);
        // The shared pair was clipped for BOTH features: the alternating projections ran.
        for f in &rep.features {
            assert!(
                f.clipped.contains(&vec![0, 1]),
                "feature {} did not clip the pair",
                f.feature
            );
        }
        assert_pure(&out);
    }

    #[test]
    fn three_constrained_features_with_shared_three_way_tables() {
        let f = |x: &[f64]| x[0].tanh() + x[1].tanh() - x[2].tanh() + 0.5 * x[0] * x[1] * x[2];
        let mut fx = fit(
            3,
            f,
            &[(0, true), (1, true), (2, false)],
            3,
            TableBudget::default(),
        );
        for u in [[0u32, 1], [1, 2], [0, 2]] {
            if fx.bank.tables.iter().any(|t| t.u == FeatureSet::new(&u)) {
                perturb(&mut fx.bank, &u, 0.3);
            }
        }
        let rows = RepairRows {
            cells: &fx.cells,
            h: &fx.h,
        };
        let signs = [sign(0, true), sign(1, true), sign(2, false)];
        assert!(
            signs
                .iter()
                .any(|s| worst_step(&fx.bank, s.feature, s.increasing) < -0.05),
            "the fixture must be broken"
        );
        let (out, _) = repair_bank(&fx.bank, &signs, &rows).unwrap();
        for s in signs {
            assert!(worst_step(&out, s.feature, s.increasing) >= -1e-9, "{s:?}");
        }
        assert_pure(&out);
    }

    #[test]
    fn the_exact_envelope_matches_brute_force_and_the_bound_sits_below_it() {
        let mut fx = fit(3, pair_target, &[(0, true)], 3, TableBudget::default());
        perturb(&mut fx.bank, &[0, 1], 0.3);
        let fam = Family {
            feature: 0,
            sign: 1.0,
            increasing: true,
            main: find(&fx.bank, &[0]),
            dense: (0..fx.bank.tables.len())
                .filter(|&i| {
                    fx.bank.tables[i].u.order() > 1 && fx.bank.tables[i].u.contains(FeatureId(0))
                })
                .collect(),
            factored: (0..fx.bank.factored.len())
                .filter(|&i| fx.bank.factored[i].u.contains(FeatureId(0)))
                .collect(),
            nb: n_cells(&fx.bank, 0).unwrap() - 2,
        };
        let steps = family_steps(&fam, &fx.bank, u64::MAX).unwrap();
        let (exact, is_exact) = envelope(&fam, &steps, u64::MAX).unwrap();
        let (bound, is_bound) = envelope(&fam, &steps, 0).unwrap();
        assert!(is_exact);
        // The family's worst step by brute force equals main step + exact envelope.
        let main = main_steps(&fx.bank.tables[fam.main], &fam).unwrap();
        let worst = main
            .iter()
            .zip(&exact)
            .map(|(m, e)| m + e)
            .fold(f64::INFINITY, f64::min);
        assert!((worst - worst_step(&fx.bank, 0, true)).abs() < 1e-9);
        let has_three_way =
            !fam.factored.is_empty() || fam.dense.iter().any(|&i| fx.bank.tables[i].u.order() == 3);
        assert!(
            has_three_way,
            "fixture needs a component of more than one table"
        );
        assert!(!is_bound);
        for (b, e) in bound.iter().zip(&exact) {
            assert!(b <= &(e + 1e-12));
        }
    }

    #[test]
    fn factored_effects_enter_the_envelope_and_are_never_clipped() {
        // The decomposition keeps order-3 effects factored (banding is what densifies them).
        let mut fx = fit(3, pair_target, &[(0, true)], 3, TableBudget::default());
        assert!(
            !fx.bank.factored.is_empty(),
            "fixture must carry a factored effect"
        );
        perturb(&mut fx.bank, &[0, 1], 0.4);
        perturb(&mut fx.bank, &[0, 2], 0.4);
        assert!(
            worst_step(&fx.bank, 0, true) < -0.1,
            "the fixture must be broken"
        );
        let rows = RepairRows {
            cells: &fx.cells,
            h: &fx.h,
        };
        for cap in [u64::MAX, 0] {
            let (out, rep) = repair_bank_with_cap(&fx.bank, &[sign(0, true)], &rows, cap).unwrap();
            assert!(worst_step(&out, 0, true) >= -1e-9, "cap {cap}");
            assert_eq!(
                out.factored, fx.bank.factored,
                "factored effects are never clipped"
            );
            if cap == 0 {
                assert_eq!(rep.features[0].certificate, "bound");
            }
        }
    }

    #[test]
    fn an_unused_constrained_feature_is_reported_and_skipped() {
        let fx = fit(
            3,
            |x: &[f64]| 2.0 * x[0].tanh(),
            &[(0, true)],
            2,
            TableBudget::default(),
        );
        let rows = RepairRows {
            cells: &fx.cells,
            h: &fx.h,
        };
        // Feature 2 carries no signal; drop any table it has to make it unused.
        let mut bank = fx.bank.clone();
        bank.tables.retain(|t| !t.u.contains(FeatureId(2)));
        let (_, rep) = repair_bank(&bank, &[sign(2, true)], &rows).unwrap();
        assert!(rep.certified);
        assert!(rep.features[0].tables.is_empty());
    }

    #[test]
    fn a_line_count_past_usize_takes_the_bounded_path() {
        // An order-eight effect with seven 1,024-cell partners: 2^70 lines. The product used to
        // overflow (a trap in every profile) before the cost check could pick the per-box bound.
        assert_eq!(line_count(&[1024; 7]), None);
        assert_eq!(line_count(&[3, 4, 5]), Some(60));
        assert_eq!(line_count(&[]), Some(1));
    }
}
