//! Model-report aggregates: the per-row passes behind the Python model report
//! (`pricing_report` / `html_report`), kept native so a report on millions of rows costs about
//! as much as scoring them.
//!
//! * [`cell_sums`] — actual-versus-expected sums per merged-grid cell for every raw feature, in
//!   parallel over features.
//! * [`ranking`] — the Lorenz (concentration) Gini, the perfect-ranking Gini it is normalised
//!   by, equal-mass lift bands and a sampled Lorenz curve, from one sort.
//! * [`cell_quantiles`] — the weighted mean and quantiles of a per-row value (each feature's
//!   Shapley value) within each merged-grid cell, for every raw feature in parallel.
//!
//! All are deterministic: every sum runs in a fixed row order (rows sorted by value, ties by
//! row index), independent of the thread count.

use crate::error::PbError;
use rayon::prelude::*;

/// Sums over the rows of each cell of one raw feature's merged grid.
#[derive(Debug, Clone, PartialEq)]
pub struct CellSums {
    /// `Σ w·y` per cell.
    pub actual: Vec<f64>,
    /// `Σ w·prediction` per cell.
    pub expected: Vec<f64>,
    /// `Σ w·exposure` per cell.
    pub mass: Vec<f64>,
    /// Number of rows per cell.
    pub rows: Vec<u64>,
    /// `Σ w·variance` per cell (zero when no variance was given).
    pub variance: Vec<f64>,
}

/// The per-row inputs of [`cell_sums`]: target, prediction, weight, exposure and, optionally,
/// the per-row variance (dispersion × variance function) used for the cells' z-scores.
#[derive(Debug, Clone, Copy)]
pub struct RowValues<'a> {
    /// The target per row.
    pub y: &'a [f64],
    /// The expected value per row.
    pub pred: &'a [f64],
    /// The sample weight per row.
    pub weight: &'a [f64],
    /// The exposure per row.
    pub exposure: &'a [f64],
    /// The variance per row, or `None`.
    pub variance: Option<&'a [f64]>,
}

/// Per-cell sums for every raw feature. `cells` is row-major `n_rows × n_cells.len()` (the
/// layout `TableModel::row_cells` produces, flattened); `n_cells[j]` is feature `j`'s cell count.
///
/// # Errors
/// [`PbError::ShapeMismatch`] if the arrays disagree in length; [`PbError::InvalidInput`] if a
/// cell index is outside its feature's grid.
pub fn cell_sums(
    cells: &[u32],
    n_cells: &[u32],
    rows: RowValues<'_>,
) -> Result<Vec<CellSums>, PbError> {
    let n_features = n_cells.len();
    let n = rows.y.len();
    let lengths_ok = rows.pred.len() == n
        && rows.weight.len() == n
        && rows.exposure.len() == n
        && rows.variance.is_none_or(|v| v.len() == n)
        && Some(cells.len()) == n.checked_mul(n_features);
    if !lengths_ok {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "cell_sums: {} cell indices for {n} rows x {n_features} features, and per-row \
                 arrays of {}/{}/{}/{} values",
                cells.len(),
                rows.pred.len(),
                rows.weight.len(),
                rows.exposure.len(),
                rows.variance.map_or(n, <[f64]>::len),
            ),
        });
    }
    n_cells
        .par_iter()
        .enumerate()
        .map(|(j, &k)| feature_sums(cells, n_features, j, k as usize, rows))
        .collect()
}

fn feature_sums(
    cells: &[u32],
    n_features: usize,
    feature: usize,
    k: usize,
    rows: RowValues<'_>,
) -> Result<CellSums, PbError> {
    let mut out = CellSums {
        actual: vec![0.0; k],
        expected: vec![0.0; k],
        mass: vec![0.0; k],
        rows: vec![0; k],
        variance: vec![0.0; k],
    };
    let column = cells.iter().skip(feature).step_by(n_features.max(1));
    let values = rows
        .y
        .iter()
        .zip(rows.pred)
        .zip(rows.weight.iter().zip(rows.exposure));
    for (i, (&cell, ((&y, &p), (&w, &e)))) in column.zip(values).enumerate() {
        let c = cell as usize;
        let bad = || PbError::InvalidInput {
            what: format!("cell {c} of feature {feature} is outside its {k}-cell grid"),
        };
        *out.actual.get_mut(c).ok_or_else(bad)? += w * y;
        *out.expected.get_mut(c).ok_or_else(bad)? += w * p;
        *out.mass.get_mut(c).ok_or_else(bad)? += w * e;
        *out.rows.get_mut(c).ok_or_else(bad)? += 1;
        if let Some(var) = rows.variance {
            let v = var.get(i).copied().unwrap_or(0.0);
            *out.variance.get_mut(c).ok_or_else(bad)? += w * v;
        }
    }
    Ok(out)
}

/// Weighted statistics of a per-row value within each cell of one raw feature's merged grid.
#[derive(Debug, Clone, PartialEq)]
pub struct CellQuantiles {
    /// `Σ weight` per cell, over the rows counted (a finite value and a positive weight).
    pub weight: Vec<f64>,
    /// The weighted mean per cell (`NaN` for a cell with no weight).
    pub mean: Vec<f64>,
    /// The weighted quantiles per cell, row-major `cells × qs.len()` (`NaN` for an empty cell).
    pub quantiles: Vec<f64>,
}

/// Per-cell weighted means and quantiles of a per-row value, for every raw feature. `cells` and
/// `values` are row-major `n_rows × n_cells.len()` (one value per row and feature, e.g. each
/// feature's Shapley value); `weight` holds one weight per row. Quantile `q` of a cell is the
/// smallest value whose cumulative weight, values ascending and ties in row order, reaches `q`
/// of the cell's total weight: the weighted lower quantile (`q = 0` is the minimum, `q = 1` the
/// maximum). Rows with a non-positive or non-finite weight, or a non-finite value, are skipped.
///
/// # Errors
/// [`PbError::ShapeMismatch`] if the arrays disagree in length; [`PbError::InvalidInput`] if a
/// cell index is outside its feature's grid or a quantile is outside `[0, 1]`.
pub fn cell_quantiles(
    cells: &[u32],
    n_cells: &[u32],
    values: &[f64],
    weight: &[f64],
    qs: &[f64],
) -> Result<Vec<CellQuantiles>, PbError> {
    let n_features = n_cells.len();
    let expected = weight.len().checked_mul(n_features);
    if expected != Some(cells.len()) || expected != Some(values.len()) {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "cell_quantiles: {} cell indices and {} values for {} rows x {n_features} features",
                cells.len(),
                values.len(),
                weight.len(),
            ),
        });
    }
    if let Some(q) = qs.iter().find(|q| !(0.0..=1.0).contains(*q)) {
        return Err(PbError::InvalidInput {
            what: format!("cell_quantiles: quantile {q} is outside [0, 1]"),
        });
    }
    n_cells
        .par_iter()
        .enumerate()
        .map(|(j, &k)| feature_quantiles(cells, values, weight, (n_features, j), k as usize, qs))
        .collect()
}

fn feature_quantiles(
    cells: &[u32],
    values: &[f64],
    weight: &[f64],
    (n_features, feature): (usize, usize),
    k: usize,
    qs: &[f64],
) -> Result<CellQuantiles, PbError> {
    let stride = n_features.max(1);
    let column = cells
        .iter()
        .skip(feature)
        .step_by(stride)
        .zip(values.iter().skip(feature).step_by(stride))
        .zip(weight);
    let mut rows: Vec<(u32, f64, f64)> = Vec::new();
    for ((&cell, &v), &w) in column {
        if cell as usize >= k {
            return Err(PbError::InvalidInput {
                what: format!("cell {cell} of feature {feature} is outside its {k}-cell grid"),
            });
        }
        if w > 0.0 && w.is_finite() && v.is_finite() {
            rows.push((cell, v, w));
        }
    }
    // By cell, then value; the sort is stable, so tied values keep their row order.
    rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let n_q = qs.len();
    let mut out = CellQuantiles {
        weight: vec![0.0; k],
        mean: vec![f64::NAN; k],
        quantiles: vec![f64::NAN; k.saturating_mul(n_q)],
    };
    let mut cumulative: Vec<f64> = Vec::new();
    for run in rows.chunk_by(|a, b| a.0 == b.0) {
        let Some(&(cell, _, _)) = run.first() else {
            continue;
        };
        let cell = cell as usize;
        cumulative.clear();
        let (mut total, mut sum) = (0.0_f64, 0.0_f64);
        for &(_, v, w) in run {
            total += w;
            sum += w * v;
            cumulative.push(total);
        }
        if let Some(slot) = out.weight.get_mut(cell) {
            *slot = total;
        }
        if let Some(slot) = out.mean.get_mut(cell) {
            *slot = sum / total;
        }
        let base = cell.saturating_mul(n_q);
        for (i, &q) in qs.iter().enumerate() {
            let target = q * total;
            let at = cumulative
                .partition_point(|&c| c < target)
                .min(run.len().saturating_sub(1));
            if let (Some(slot), Some(&(_, v, _))) =
                (out.quantiles.get_mut(base.saturating_add(i)), run.get(at))
            {
                *slot = v;
            }
        }
    }
    Ok(out)
}

/// One equal-mass band of ascending predicted rate.
#[derive(Debug, Clone, PartialEq)]
pub struct LiftBand {
    /// Rows in the band.
    pub rows: u64,
    /// `Σ mass` in the band.
    pub mass: f64,
    /// `Σ mass·actual rate` (the band's actual total).
    pub actual: f64,
    /// `Σ mass·predicted rate` (the band's predicted total).
    pub predicted: f64,
    /// Smallest predicted rate in the band.
    pub rate_min: f64,
    /// Largest predicted rate in the band.
    pub rate_max: f64,
    /// Sum of the rows' variance of the actual total (0 when no variance was given), for the
    /// band's A/E standard error.
    pub variance: f64,
}

/// The ranking statistics of predicted rates against actual rates under a row mass.
#[derive(Debug, Clone, PartialEq)]
pub struct Ranking {
    /// Rows used (finite rates and a positive, finite mass).
    pub rows: u64,
    /// Concentration Gini of the actual rate ranked by the predicted rate (0 when degenerate).
    pub gini: f64,
    /// The same Gini for the perfect ranking (by the actual rate itself).
    pub gini_perfect: f64,
    /// Non-empty equal-mass bands, ascending predicted rate.
    pub lift: Vec<LiftBand>,
    /// The Lorenz curve's cumulative actual share at `points` evenly spaced mass shares
    /// (descending predicted rate); empty when degenerate.
    pub lorenz: Vec<f64>,
}

/// One run of rows sharing a ranking value.
struct Group {
    value: f64,
    rows: u64,
    mass: f64,
    /// `Σ mass·actual rate`.
    actual: f64,
    /// `Σ mass·max(actual rate, 0)`, the Lorenz numerator.
    actual_pos: f64,
    /// `Σ mass·predicted rate`.
    predicted: f64,
    /// `Σ variance` of the actual total.
    variance: f64,
}

/// The per-row inputs of [`ranking`] besides the sort key.
#[derive(Debug, Clone, Copy)]
struct RankRows<'a> {
    rate: &'a [f64],
    y_rate: &'a [f64],
    mass: &'a [f64],
    variance: Option<&'a [f64]>,
}

/// Rows (indices into `keep`) sorted ascending by `key`, ties by row index, grouped by value.
fn groups(keep: &[usize], key: &[f64], rows: RankRows<'_>) -> Vec<Group> {
    let RankRows {
        rate,
        y_rate,
        mass,
        variance,
    } = rows;
    let value = |i: usize| key.get(i).copied().unwrap_or(f64::NAN);
    let mut order = keep.to_vec();
    order.par_sort_unstable_by(|&a, &b| value(a).total_cmp(&value(b)).then(a.cmp(&b)));
    let mut out: Vec<Group> = Vec::new();
    for i in order {
        let v = value(i);
        let m = mass.get(i).copied().unwrap_or(0.0);
        let y = y_rate.get(i).copied().unwrap_or(0.0);
        let r = rate.get(i).copied().unwrap_or(0.0);
        let var = variance.and_then(|vs| vs.get(i).copied()).unwrap_or(0.0);
        match out.last_mut() {
            Some(g) if g.value.total_cmp(&v).is_eq() => {
                g.rows += 1;
                g.mass += m;
                g.actual += m * y;
                g.actual_pos += m * y.max(0.0);
                g.predicted += m * r;
                g.variance += var;
            }
            _ => out.push(Group {
                value: v,
                rows: 1,
                mass: m,
                actual: m * y,
                actual_pos: m * y.max(0.0),
                predicted: m * r,
                variance: var,
            }),
        }
    }
    out
}

/// The concentration Gini over `groups` taken in DESCENDING value order, and the Lorenz
/// polyline's vertices `(x, y)` (excluding the origin). Mirrors `t_boost.metrics`'
/// `concentration_gini`: a tie group is one straight segment, trapezoid rule, `2·area − 1`.
fn lorenz(groups: &[Group]) -> Option<(f64, Vec<(f64, f64)>)> {
    let total_m: f64 = groups.iter().map(|g| g.mass).sum();
    let total_y: f64 = groups.iter().map(|g| g.actual_pos).sum();
    if total_m <= 0.0 || total_y <= 0.0 {
        return None;
    }
    let (mut cx, mut cy, mut area) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut vertices = Vec::with_capacity(groups.len());
    for g in groups.iter().rev() {
        let (nx, ny) = (cx + g.mass / total_m, cy + g.actual_pos / total_y);
        area += (nx - cx) * (ny + cy) * 0.5;
        vertices.push((nx, ny));
        (cx, cy) = (nx, ny);
    }
    let gini = 2.0 * area - 1.0;
    Some((if gini.is_finite() { gini } else { 0.0 }, vertices))
}

/// The polyline through the origin and `vertices`, sampled at `points` evenly spaced x.
fn sample(vertices: &[(f64, f64)], points: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(points);
    let mut prev = (0.0_f64, 0.0_f64);
    let mut it = vertices.iter().copied().peekable();
    let denom = points.saturating_sub(1).max(1) as f64;
    for p in 0..points {
        let x = p as f64 / denom;
        while let Some(&(vx, vy)) = it.peek() {
            if vx < x {
                prev = (vx, vy);
                it.next();
            } else {
                break;
            }
        }
        let y = match it.peek() {
            Some(&(vx, vy)) if vx > prev.0 => prev.1 + (vy - prev.1) * (x - prev.0) / (vx - prev.0),
            Some(&(_, vy)) => vy,
            None => prev.1,
        };
        out.push(y);
    }
    out
}

/// Ranking statistics of `rate` (predicted) against `y_rate` (actual) under `mass`. Rows with a
/// non-finite rate or a mass that is not positive and finite are left out. `bands` equal-mass
/// lift bands are formed in ascending predicted rate, a tie group going whole to the band of
/// its cumulative-mass midpoint (so fewer bands come back when there are fewer distinct
/// predictions); the Lorenz curve is sampled at `points` mass shares. `variance`, when given,
/// is each row's variance of its actual total (`mass·actual rate`), summed per band.
///
/// # Errors
/// [`PbError::ShapeMismatch`] if the arrays disagree in length.
pub fn ranking(
    rate: &[f64],
    y_rate: &[f64],
    mass: &[f64],
    variance: Option<&[f64]>,
    bands: usize,
    points: usize,
) -> Result<Ranking, PbError> {
    if y_rate.len() != rate.len()
        || mass.len() != rate.len()
        || variance.is_some_and(|v| v.len() != rate.len())
    {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "ranking: {} rates, {} actual rates, {} masses, {} variances",
                rate.len(),
                y_rate.len(),
                mass.len(),
                variance.map_or(rate.len(), <[f64]>::len)
            ),
        });
    }
    let keep: Vec<usize> = rate
        .iter()
        .zip(y_rate)
        .zip(mass)
        .enumerate()
        .filter(|(_, ((r, y), m))| r.is_finite() && y.is_finite() && m.is_finite() && **m > 0.0)
        .map(|(i, _)| i)
        .collect();
    let rows = RankRows {
        rate,
        y_rate,
        mass,
        variance,
    };
    let by_rate = groups(&keep, rate, rows);
    let (gini, lorenz_curve) = match lorenz(&by_rate) {
        Some((g, vertices)) => (g, sample(&vertices, points)),
        None => (0.0, Vec::new()),
    };
    let gini_perfect = lorenz(&groups(&keep, y_rate, rows)).map_or(0.0, |(g, _)| g);
    let total: f64 = by_rate.iter().map(|g| g.mass).sum();
    let mut lift: Vec<LiftBand> = Vec::new();
    let mut last_band = usize::MAX;
    let mut cum = 0.0_f64;
    for g in &by_rate {
        let mid = cum + g.mass / 2.0;
        cum += g.mass;
        let band = if total > 0.0 {
            ((mid / total * bands as f64) as usize).min(bands.saturating_sub(1))
        } else {
            0
        };
        match lift.last_mut() {
            Some(b) if band == last_band => {
                b.rows += g.rows;
                b.mass += g.mass;
                b.actual += g.actual;
                b.predicted += g.predicted;
                b.rate_max = g.value;
                b.variance += g.variance;
            }
            _ => lift.push(LiftBand {
                rows: g.rows,
                mass: g.mass,
                actual: g.actual,
                predicted: g.predicted,
                rate_min: g.value,
                rate_max: g.value,
                variance: g.variance,
            }),
        }
        last_band = band;
    }
    Ok(Ranking {
        rows: keep.len() as u64,
        gini,
        gini_perfect,
        lift,
        lorenz: lorenz_curve,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_sums_aggregate_by_cell_and_reject_out_of_grid_cells() {
        let cells = [0_u32, 1, 1, 1, 1, 0];
        let y = [1.0, 2.0, 3.0];
        let p = [0.5, 1.0, 1.5];
        let w = [1.0, 2.0, 1.0];
        let e = [1.0, 1.0, 0.5];
        let rows = RowValues {
            y: &y,
            pred: &p,
            weight: &w,
            exposure: &e,
            variance: None,
        };
        let out = cell_sums(&cells, &[2, 2], rows).unwrap_or_default();
        assert_eq!(out.len(), 2);
        assert_eq!(out.first().map(|c| c.actual.clone()), Some(vec![1.0, 7.0]));
        assert_eq!(out.get(1).map(|c| c.rows.clone()), Some(vec![1, 2]));
        assert_eq!(out.get(1).map(|c| c.mass.clone()), Some(vec![0.5, 3.0]));
        assert!(cell_sums(&cells, &[1, 2], rows).is_err());
    }

    #[test]
    fn ranking_groups_ties_and_matches_a_hand_gini() {
        // Two tied predictions: they must share a band, and the Gini treats them as one
        // segment. Perfect ranking of y = [0, 1, 1, 2] with unit mass: area by hand.
        let rate = [1.0, 2.0, 2.0, 3.0];
        let y = [0.0, 1.0, 1.0, 2.0];
        let m = [1.0; 4];
        let var = [0.5, 1.0, 1.0, 2.0];
        let r = ranking(&rate, &y, &m, Some(&var), 4, 5).unwrap_or_else(|_| Ranking {
            rows: 0,
            gini: f64::NAN,
            gini_perfect: f64::NAN,
            lift: Vec::new(),
            lorenz: Vec::new(),
        });
        assert_eq!(r.rows, 4);
        assert_eq!(r.lift.len(), 3);
        assert_eq!(r.lift.get(1).map(|b| b.rows), Some(2));
        assert_eq!(r.lift.get(1).map(|b| b.variance), Some(2.0));
        // descending: (0.25, 0.5), (0.75, 1.0), (1.0, 1.0) -> area 0.0625+0.375+0.25
        assert!((r.gini - (2.0 * 0.6875 - 1.0)).abs() < 1e-12);
        assert!((r.gini_perfect - r.gini).abs() < 1e-12);
        assert_eq!(r.lorenz.len(), 5);
        assert!((r.lorenz.last().copied().unwrap_or(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cell_quantiles_weight_each_cell_and_reject_bad_input() {
        // Two features over four rows; feature 0 puts rows 0, 2, 3 in cell 1 with values
        // 3, 1, 2 and weights 1, 1, 2 (sorted: 1 (w 1), 2 (w 2), 3 (w 1)), row 1 alone in cell 0.
        let cells = [1_u32, 0, 0, 0, 1, 0, 1, 0];
        let values = [3.0, 9.0, 5.0, 9.0, 1.0, 9.0, 2.0, f64::NAN];
        let weight = [1.0, 1.0, 1.0, 2.0];
        let out = cell_quantiles(&cells, &[2, 1], &values, &weight, &[0.0, 0.5, 0.9, 1.0])
            .unwrap_or_default();
        let first = out.first().cloned();
        assert_eq!(
            first.as_ref().map(|c| c.weight.clone()),
            Some(vec![1.0, 4.0])
        );
        assert_eq!(first.as_ref().map(|c| c.mean.clone()), Some(vec![5.0, 2.0]));
        // Cumulative weights 1, 3, 4 of 4: q 0 -> 1, q 0.5 (2) -> 2, q 0.9 (3.6) -> 3, q 1 -> 3.
        assert_eq!(
            first.map(|c| c.quantiles),
            Some(vec![5.0, 5.0, 5.0, 5.0, 1.0, 2.0, 3.0, 3.0])
        );
        // Feature 1: the NaN value of row 3 is skipped.
        assert_eq!(out.get(1).map(|c| c.weight.clone()), Some(vec![3.0]));
        assert!(cell_quantiles(&cells, &[1, 1], &values, &weight, &[0.5]).is_err());
        assert!(cell_quantiles(&cells, &[2, 1], &values, &weight, &[1.5]).is_err());
        assert!(cell_quantiles(&cells, &[2, 1], &values, &weight[..3], &[0.5]).is_err());
    }

    #[test]
    fn ranking_drops_rows_without_mass() {
        let r = ranking(&[1.0, 2.0], &[1.0, 1.0], &[0.0, 1.0], None, 10, 3);
        assert_eq!(r.map(|r| r.rows).unwrap_or(0), 1);
    }
}
