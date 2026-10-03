//! Per-feature border construction (spec §03.3). Borders are built independently per
//! feature, so the [`crate::data::bin::bin_columns`] loop is rayon-parallel and
//! order-independent — bit-reproducible across thread counts (§1 `[GATE]`).
//!
//! Determinism + permutation-invariance: the grid is a pure function of the value
//! multiset, the weights, and `(seed, feat)`. The subsample (when a feature exceeds
//! `subsample_for_binning`) is drawn from the VALUE-SORTED finite values via a
//! per-feature re-seeded `Pcg64`, so permuting input rows cannot change the grid.

use crate::backend::{pb_rng, Stage};
use crate::data::{BinConfig, BorderGrid, FeatureId};
use crate::error::PbError;
use rand::RngCore;

/// Build one feature's frozen [`BorderGrid`] (spec §03.3).
///
/// Midpoint borders when the feature has at most `max_bin` distinct finite values
/// (exact, one bin per real value); otherwise equal-count quantile borders over a
/// seeded subsample, with ties deduped so `borders` is strictly ascending. An
/// all-missing column yields the degenerate single-data-bin grid (`n_bins = 2`).
///
/// # Errors
/// [`PbError::InvalidConfig`] if `cfg` is invalid; [`PbError::ShapeMismatch`] if
/// `weight` length differs from `col`; [`PbError::InvalidInput`] if the total finite
/// weight is non-positive; [`PbError::Internal`] for an impossible index/cast.
pub fn build_grid(
    col: &[f32],
    weight: Option<&[f32]>,
    cfg: &BinConfig,
    seed: u64,
    feat: FeatureId,
) -> Result<BorderGrid, PbError> {
    cfg.validate()?;
    if let Some(w) = weight {
        if w.len() != col.len() {
            return Err(PbError::ShapeMismatch {
                what: format!("weight len {} != column len {}", w.len(), col.len()),
            });
        }
        if w.iter().any(|value| !value.is_finite() || *value < 0.0)
            || !w.iter().any(|value| *value > 0.0)
        {
            return Err(PbError::InvalidInput {
                what: "sample weights must be finite, nonnegative, and have positive total mass"
                    .into(),
            });
        }
    }

    // Collect finite (value, weight) pairs — NaN and ±inf are excluded from border
    // construction (inf would distort quantiles; bin() handles inf by clamping).
    let mut fw: Vec<(f32, f64)> = Vec::new();
    for (i, &v) in col.iter().enumerate() {
        if v.is_finite() {
            let wt = match weight {
                Some(w) => {
                    let raw = *w.get(i).ok_or_else(internal("weight index"))?;
                    if !raw.is_finite() || raw < 0.0 {
                        return Err(PbError::InvalidInput {
                            what: "sample weights must be finite and >= 0".into(),
                        });
                    }
                    f64::from(raw)
                }
                None => 1.0,
            };
            fw.push((v, wt));
        }
    }
    if fw.iter().all(|(_, weight)| *weight == 0.0) {
        // All-missing axis: no interior borders ⇒ 1 degenerate data bin + missing.
        return Ok(BorderGrid {
            borders: Vec::new(),
            n_bins: 2,
            missing_bin: 0,
        });
    }
    // Total order over BOTH value and weight: tied values are ordered deterministically
    // regardless of input row order, so the position-based subsample (and therefore the
    // grid) is a pure function of the (value, weight) multiset — permutation-invariant
    // even with per-row weights (§03.3 / the §1 determinism [GATE]).
    fw.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));

    let distinct = distinct_values(&fw);
    let max_bin = usize::from(cfg.max_bin);

    let mut borders: Vec<f32> = if distinct.len() <= max_bin {
        // dedup even on the midpoint path: for distinct f32 values ~1 ULP apart, two
        // f64 midpoints can round to the same f32, so dedup is what guarantees the
        // strictly-ascending / no-duplicate BorderGrid invariant (§03.2).
        dedup_ascending(midpoint_borders(&distinct))
    } else if u64::try_from(fw.len()).unwrap_or(u64::MAX) <= u64::from(cfg.subsample_for_binning) {
        // Whole (already-sorted) column IS the sample — borrow it directly instead of cloning (it
        // stays owned for collapse_rare_bins below). Byte-identical: quantile_borders only reads it.
        quantile_borders(&fw, cfg.max_bin)?
    } else {
        let sample = subsample_sorted(&fw, cfg.subsample_for_binning, seed, feat)?;
        if sample.iter().any(|(_, weight)| *weight > 0.0) {
            quantile_borders(&sample, cfg.max_bin)?
        } else {
            // A valid weighted feature can have all of its mass outside the
            // seeded binning sample. Use its full ledger in that case.
            quantile_borders(&fw, cfg.max_bin)?
        }
    };

    if cfg.min_data_per_bin > 0 {
        if !borders.is_empty() {
            borders = collapse_rare_bins(&fw, &borders, cfg.min_data_per_bin)?;
        }
    } else {
        // Reclaim bin budget the quantile builder's point-mass dedup left unused (a no-op when the
        // budget is already full). On heavy-point-mass features (a magic-size grid like diamonds
        // carat) many uniform-quantile probes collapse onto one border, leaving the continuous
        // regions under-resolved; refilling sharpens the main-effect shape. Order-independent
        // (G0-preserving). Skipped when a rare-bin floor is active (a refill split could create a
        // sub-`min` bin and break collapse_rare_bins's guarantee).
        borders = refill_borders(borders, &fw, max_bin);
    }

    let n_bins = u16::try_from(borders.len() + 2).map_err(|_| PbError::Internal {
        what: "n_bins exceeded u16".into(),
    })?;
    Ok(BorderGrid {
        borders,
        n_bins,
        missing_bin: 0,
    })
}

fn internal(what: &'static str) -> impl Fn() -> PbError {
    move || PbError::Internal { what: what.into() }
}

/// Sorted-ascending distinct values of an already-value-sorted `(value, weight)` slice.
fn distinct_values(sorted_fw: &[(f32, f64)]) -> Vec<f32> {
    let mut distinct: Vec<f32> = Vec::new();
    for &(v, _) in sorted_fw {
        if distinct.last() != Some(&v) {
            distinct.push(v);
        }
    }
    distinct
}

/// Midpoint borders between consecutive sorted distinct values (exact splits, one
/// bin per real value). Strictly ascending because `distinct` is strictly increasing.
fn midpoint_borders(distinct: &[f32]) -> Vec<f32> {
    distinct
        .windows(2)
        .filter_map(|w| match w {
            [a, b] => Some(separating_border(*a, *b)),
            _ => None,
        })
        .collect()
}

fn separating_border(a: f32, b: f32) -> f32 {
    let midpoint = ((f64::from(a) + f64::from(b)) * 0.5) as f32;
    // Borders are upper-inclusive: rounding to b would merge the two values.
    if midpoint >= b {
        a
    } else {
        midpoint
    }
}

/// Interior quantile probabilities `linspace(0,1,max_bin+1)[1..max_bin]` — the
/// `max_bin - 1` evenly-spaced interior ranks.
fn interior_quantile_probs(max_bin: u8) -> Vec<f64> {
    let m = f64::from(max_bin);
    (1..u16::from(max_bin)).map(|k| f64::from(k) / m).collect()
}

/// Equal-count quantile borders over a value-sorted weighted sample, deduped to
/// strictly ascending (spec §03.3).
fn quantile_borders(sorted_sample: &[(f32, f64)], max_bin: u8) -> Result<Vec<f32>, PbError> {
    let mut cum = Vec::with_capacity(sorted_sample.len());
    let mut vals = Vec::with_capacity(sorted_sample.len());
    let mut acc = 0.0_f64;
    for &(v, w) in sorted_sample {
        acc += w;
        cum.push(acc);
        vals.push(f64::from(v));
    }
    let total = acc;
    if total <= 0.0 {
        return Err(PbError::InvalidInput {
            what: "non-positive total weight in binning subsample".into(),
        });
    }
    let qs = interior_quantile_probs(max_bin);
    let raw = weighted_quantiles(&vals, &cum, total, &qs)?;
    Ok(dedup_ascending(raw))
}

/// Weighted quantiles via the averaged-inverted-CDF convention (sklearn-compatible),
/// computed in `f64` and stored `f32`. `vals` is value-sorted; `cum[i]` is the
/// cumulative weight through `vals[i]`.
fn weighted_quantiles(
    vals: &[f64],
    cum: &[f64],
    total: f64,
    qs: &[f64],
) -> Result<Vec<f32>, PbError> {
    let last = vals.len().saturating_sub(1);
    let mut out = Vec::with_capacity(qs.len());
    for &q in qs {
        let target = q * total;
        // Inverted CDF: smallest i with cum[i] >= target. By construction every
        // j < i has cum[j] < target strictly.
        let i = cum.partition_point(|&c| c < target).min(last);
        let hi = *vals.get(i).ok_or_else(internal("quantile index"))?;
        // averaged_inverted_cdf: when the CDF reaches `target` EXACTLY at vals[i]
        // (cum[i] == target), the quantile straddles vals[i] and vals[i+1] — average
        // them. Otherwise (cum[i] > target) the inverted-CDF value vals[i] stands.
        let v = if i < last && *cum.get(i).ok_or_else(internal("quantile cum"))? == target {
            f64::from(separating_border(
                hi as f32,
                *vals.get(i + 1).ok_or_else(internal("quantile next"))? as f32,
            ))
        } else {
            hi
        };
        out.push(v as f32);
    }
    Ok(out)
}

/// Sort (total order) and drop duplicates, yielding strictly-ascending borders.
fn dedup_ascending(mut v: Vec<f32>) -> Vec<f32> {
    v.sort_by(f32::total_cmp);
    v.dedup();
    v
}

/// Reclaim bin budget the quantile builder's point-mass `dedup` left unused. When the interior
/// borders fall short of the `max_bin - 1` budget — because heavy point-masses collapsed many
/// uniform-quantile probes into a single border — fill the rest by greedily splitting the
/// highest-row-mass SPLITTABLE interval at the distinct-value boundary nearest its weighted-mass
/// median, until the budget is met or no interval can be split. Resolves the continuous regions
/// instead of leaving budget unused. Deterministic (pure function of `fw`; densest-first then nearest
/// boundary) and G0-preserving (binning never changes the fANOVA order).
fn refill_borders(borders: Vec<f32>, fw: &[(f32, f64)], max_bin: usize) -> Vec<f32> {
    let target = max_bin.saturating_sub(1);
    let n = fw.len();
    if borders.len() >= target || n < 2 {
        return borders;
    }
    let val = |i: usize| fw.get(i).map(|p| p.0);
    // prefix[i] = Σ weight of fw[0..i].
    let prefix: Vec<f64> = std::iter::once(0.0_f64)
        .chain(fw.iter().scan(0.0_f64, |acc, &(_, w)| {
            *acc += w;
            Some(*acc)
        }))
        .collect();
    let psum = |i: usize| prefix.get(i).copied().unwrap_or(0.0);
    // fw index bounds of the current intervals (value-sorted; border b ⇒ first index with value >
    // b) — upper-inclusive, matching `bin()`'s rule (v == border ⇒ the LOWER bin, data/bin.rs's
    // `partition_point(|&b| b < v)`). Using `v < b` here previously counted rows tied exactly at a
    // border into the interval ABOVE it, so a heavy point mass sitting on a border (e.g. a
    // 90%-zeros column with border 0.0) got misattributed and the refill heuristic wasted budget
    // splitting an already-resolved tied run instead of the actual continuous tail.
    let mut bounds: Vec<usize> = Vec::with_capacity(borders.len() + 2);
    bounds.push(0);
    for &b in &borders {
        bounds.push(fw.partition_point(|&(v, _)| v <= b));
    }
    bounds.push(n);
    bounds.sort_unstable();
    bounds.dedup();
    let mut intervals: Vec<(usize, usize)> = bounds
        .windows(2)
        .filter_map(|w| match w {
            &[a, b] => Some((a, b)),
            _ => None,
        })
        .collect();

    let mut out = borders;
    while out.len() < target {
        // Densest interval that still holds >= 2 distinct values.
        let mut best: Option<(usize, usize, usize)> = None; // (k, lo, hi)
        let mut best_mass = f64::NEG_INFINITY;
        for (k, &(lo, hi)) in intervals.iter().enumerate() {
            if hi - lo < 2 || val(lo) == val(hi - 1) {
                continue;
            }
            let mass = psum(hi) - psum(lo);
            if mass > best_mass {
                best_mass = mass;
                best = Some((k, lo, hi));
            }
        }
        let Some((k, lo, hi)) = best else { break };
        let half = psum(lo) + (psum(hi) - psum(lo)) * 0.5;
        let mut m = lo + 1;
        while m + 1 < hi && psum(m) < half {
            m += 1;
        }
        let Some(split) = nearest_distinct_boundary(fw, lo, hi, m) else {
            break;
        };
        let (Some(a), Some(b)) = (val(split - 1), val(split)) else {
            break;
        };
        out.push(separating_border(a, b));
        if let Some(slot) = intervals.get_mut(k) {
            *slot = (lo, split);
        }
        intervals.insert(k + 1, (split, hi));
    }
    dedup_ascending(out)
}

/// Nearest index `j` in `(lo, hi)` with `fw[j-1] != fw[j]` (a valid split boundary), searching
/// outward from `m`. `None` only if `fw[lo..hi]` is a single value (caller pre-checks splittable).
fn nearest_distinct_boundary(fw: &[(f32, f64)], lo: usize, hi: usize, m: usize) -> Option<usize> {
    let val = |i: usize| fw.get(i).map(|p| p.0);
    let m = m.clamp(lo + 1, hi.saturating_sub(1));
    for d in 0..(hi - lo) {
        let up = m + d;
        if up < hi && val(up - 1) != val(up) {
            return Some(up);
        }
        if let Some(dn) = m.checked_sub(d) {
            if dn > lo && val(dn - 1) != val(dn) {
                return Some(dn);
            }
        }
    }
    None
}

/// Draw `k` rows from a VALUE-SORTED `(value, weight)` slice without replacement,
/// using the per-feature re-seeded `Pcg64`, and return them re-sorted by value. The
/// draw operates on positions in the sorted array, so the result depends only on the
/// value multiset and `(seed, feat)` — never on input row order (permutation-invariant).
fn subsample_sorted(
    sorted_fw: &[(f32, f64)],
    k: u32,
    seed: u64,
    feat: FeatureId,
) -> Result<Vec<(f32, f64)>, PbError> {
    let len = sorted_fw.len();
    let k = usize::try_from(k).unwrap_or(usize::MAX).min(len);
    let mut rng = pb_rng(seed, 0, Stage::Binning, feat.0);
    let mut idx: Vec<usize> = (0..len).collect();
    // Partial Fisher–Yates: select k positions. Modulo is computed in u64 so the
    // index is identical on 32- and 64-bit targets (the wasm32 wire/determinism guard).
    for i in 0..k {
        let range = u64::try_from(len - i).unwrap_or(1).max(1);
        let offset = usize::try_from(rng.next_u64() % range).unwrap_or(0);
        idx.swap(i, i + offset);
    }
    let mut chosen: Vec<usize> = idx.into_iter().take(k).collect();
    chosen.sort_unstable();
    let mut out = Vec::with_capacity(chosen.len());
    for i in chosen {
        out.push(*sorted_fw.get(i).ok_or_else(internal("subsample index"))?);
    }
    Ok(out)
}

/// Greedy left-to-right merge: keep a border only when the bin accumulated since the
/// last kept border has `>= min` rows AND the remaining rows can still form one more
/// `>= min` bin (spec §03.1; default `min_data_per_bin = 0` ⇒ never called).
fn collapse_rare_bins(
    sorted_fw: &[(f32, f64)],
    borders: &[f32],
    min: u32,
) -> Result<Vec<f32>, PbError> {
    let n_data = borders.len() + 1;
    let mut counts = vec![0u64; n_data];
    for &(v, _) in sorted_fw {
        let b = borders.partition_point(|&x| x < v);
        *counts.get_mut(b).ok_or_else(internal("collapse count"))? += 1;
    }
    let total: u64 = counts.iter().sum();
    let minw = u64::from(min);
    let mut kept = Vec::new();
    let (mut acc, mut seen) = (0_u64, 0_u64);
    for k in 0..n_data {
        let c = *counts.get(k).ok_or_else(internal("collapse bin"))?;
        acc += c;
        seen += c;
        if k < borders.len() {
            let remaining = total - seen;
            if acc >= minw && remaining >= minw {
                kept.push(*borders.get(k).ok_or_else(internal("collapse border"))?);
                acc = 0;
            }
        }
    }
    Ok(kept)
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

    fn cfg(max_bin: u8, subsample: u32) -> BinConfig {
        BinConfig {
            max_bin,
            subsample_for_binning: subsample,
            min_data_per_bin: 0,
            border_family: crate::data::BorderFamily::EqualCount,
        }
    }

    fn is_strictly_ascending(b: &[f32]) -> bool {
        b.windows(2).all(|w| w[0] < w[1])
    }

    #[test]
    fn midpoint_branch_is_exact_and_ascending() {
        let col = [1.0, 2.0, 3.0, 2.0, 1.0];
        let g = build_grid(&col, None, &cfg(254, 200_000), 0, FeatureId(0)).unwrap();
        // 3 distinct ⇒ 2 midpoint borders at 1.5 and 2.5.
        assert_eq!(g.borders, vec![1.5, 2.5]);
        assert_eq!(g.n_bins, 4); // 3 data + missing
        assert!(is_strictly_ascending(&g.borders));
    }

    #[test]
    fn bug006_adjacent_floats_separate_in_midpoints_quantiles_and_refills() {
        for (a, b) in [
            (16_777_218., 16_777_220.),
            (16_777_216., 16_777_218.),
            (-16_777_220., -16_777_218.),
            (f32::from_bits(1), f32::from_bits(2)),
            (-f32::from_bits(2), -f32::from_bits(1)),
            (f32::from_bits(f32::MAX.to_bits() - 1), f32::MAX),
        ] {
            for border in [
                midpoint_borders(&[a, b])[0],
                refill_borders(vec![], &[(a, 1.), (b, 1.)], 2)[0],
                weighted_quantiles(&[f64::from(a), f64::from(b)], &[1., 2.], 2., &[0.5]).unwrap()
                    [0],
            ] {
                assert!(a <= border && border < b, "{a} <= {border} < {b}");
            }
        }
    }

    #[test]
    fn all_missing_is_degenerate_single_bin() {
        let col = [f32::NAN, f32::NAN];
        let g = build_grid(&col, None, &cfg(254, 200_000), 0, FeatureId(0)).unwrap();
        assert!(g.borders.is_empty());
        assert_eq!(g.n_bins, 2);
        assert_eq!(g.missing_bin, 0);
    }

    #[test]
    fn quantile_branch_respects_max_bin_and_dedups() {
        // 100 distinct values, max_bin 8 ⇒ quantile path, <= 7 borders, strictly ascending.
        let col: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let g = build_grid(&col, None, &cfg(8, 200_000), 7, FeatureId(2)).unwrap();
        assert!(g.borders.len() <= 7, "got {} borders", g.borders.len());
        assert!(is_strictly_ascending(&g.borders));
        assert!(g.n_bins as usize == g.borders.len() + 2);
    }

    #[test]
    fn skewed_ties_dedup_to_strictly_ascending() {
        // Mostly zeros ⇒ many quantiles collapse to 0.0; must remain strictly ascending.
        let mut col = vec![0.0_f32; 90];
        col.extend((1..=10).map(|i| i as f32));
        let g = build_grid(&col, None, &cfg(16, 200_000), 1, FeatureId(0)).unwrap();
        assert!(is_strictly_ascending(&g.borders));
    }

    #[test]
    fn refill_attributes_point_mass_ties_to_the_lower_interval_not_the_upper() {
        // 90 rows tied exactly at the existing border (0.0), then a resolved continuous
        // tail 1.0..=10.0 (one row each). Before the fix, `v < b` counted the 90 tied
        // rows into the interval ABOVE the border, collapsing the border's separating
        // effect entirely and wasting the one refill slot re-splitting the
        // already-resolved zero/non-zero boundary (a redundant border at 0.5) instead
        // of the actual unresolved tail. After the fix (`v <= b`, matching `bin()`'s
        // upper-inclusive rule), the tied rows stay in the lower interval, the zero-run
        // is correctly recognized as a single repeated value (unsplittable), and the
        // one refill slot goes to the tail (a border at 5.5, bisecting it).
        let mut fw: Vec<(f32, f64)> = (0..90).map(|_| (0.0_f32, 1.0_f64)).collect();
        fw.extend((1..=10).map(|v| (v as f32, 1.0_f64)));
        let borders = vec![0.0_f32];
        let out = refill_borders(borders, &fw, 3); // target = max_bin - 1 = 2
        assert_eq!(
            out,
            vec![0.0, 5.5],
            "expected the tail to be split, not the zero tie: {out:?}"
        );
    }

    #[test]
    fn grid_is_permutation_invariant_and_deterministic() {
        let col: Vec<f32> = (0..500).map(|i| (i % 37) as f32 * 0.5).collect();
        let mut shuffled = col.clone();
        shuffled.reverse();
        // Small subsample forces the RNG path; must still be permutation-invariant.
        let c = cfg(8, 50);
        let a = build_grid(&col, None, &c, 123, FeatureId(5)).unwrap();
        let b = build_grid(&shuffled, None, &c, 123, FeatureId(5)).unwrap();
        assert_eq!(a, b, "grid must depend only on the value multiset");
        // And bit-stable across repeated calls.
        let a2 = build_grid(&col, None, &c, 123, FeatureId(5)).unwrap();
        assert_eq!(a, a2);
    }

    #[test]
    fn rare_bin_collapse_guarantees_min_rows_per_bin() {
        use crate::data::bin::bin;
        // 10 distinct values, each 1 row; min 4 ⇒ every surviving bin must hold >= 4.
        let col: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let mut c = cfg(254, 200_000);
        c.min_data_per_bin = 4;
        let g = build_grid(&col, None, &c, 0, FeatureId(0)).unwrap();
        assert!(g.borders.len() <= 2, "got {} borders", g.borders.len());
        assert!(is_strictly_ascending(&g.borders));
        // The real guarantee: no surviving bin is under-full (counts, not just count).
        let mut counts = vec![0u32; g.n_data_bins()];
        for &v in &col {
            counts[usize::from(bin(v, &g).unwrap()) - 1] += 1;
        }
        assert!(
            counts.iter().all(|&c| c >= 4),
            "every bin must hold >= min_data_per_bin rows, got {counts:?}"
        );
    }

    #[test]
    fn averaged_inverted_cdf_branch_is_reachable_unweighted() {
        // max_bin 2 ⇒ quantile path with a single interior prob q=0.5. For values
        // {0,1,2,3} the CDF reaches 0.5 exactly at value 1 (cum=2 == target=2), so
        // averaged_inverted_cdf averages 1 and 2 ⇒ border 1.5.
        let col = [0.0_f32, 1.0, 2.0, 3.0];
        let g = build_grid(&col, None, &cfg(2, 200_000), 0, FeatureId(0)).unwrap();
        assert_eq!(g.borders, vec![1.5]);
    }

    #[test]
    fn weighted_quantiles_differ_from_unweighted() {
        // Same values, but weight 3 on value 0 shifts the median border down to 0.5
        // (cum after 0 is 3 == target 3 ⇒ average 0 and 1).
        let col = [0.0_f32, 1.0, 2.0, 3.0];
        let w = [3.0_f32, 1.0, 1.0, 1.0];
        let g = build_grid(&col, Some(&w), &cfg(2, 200_000), 0, FeatureId(0)).unwrap();
        assert_eq!(g.borders, vec![0.5]);
        // ... and it genuinely differs from the unweighted grid.
        let u = build_grid(&col, None, &cfg(2, 200_000), 0, FeatureId(0)).unwrap();
        assert_ne!(g.borders, u.borders);
    }

    #[test]
    fn all_zero_weight_on_quantile_path_errors() {
        let col = [0.0_f32, 1.0, 2.0, 3.0]; // 4 distinct > max_bin 2 ⇒ quantile path
        let w = [0.0_f32; 4];
        assert!(matches!(
            build_grid(&col, Some(&w), &cfg(2, 200_000), 0, FeatureId(0)),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn negative_or_nonfinite_weight_errors() {
        let col = [0.0_f32, 1.0];
        assert!(matches!(
            build_grid(
                &col,
                Some(&[-1.0, 1.0]),
                &cfg(254, 200_000),
                0,
                FeatureId(0)
            ),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            build_grid(
                &col,
                Some(&[f32::NAN, 1.0]),
                &cfg(254, 200_000),
                0,
                FeatureId(0)
            ),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn weighted_grid_is_permutation_invariant() {
        // Tied values carrying DISTINCT weights, column > subsample ⇒ RNG path. The
        // (value, weight) tie-break is what makes this permutation-invariant.
        let col: Vec<f32> = (0..200).map(|i| (i % 20) as f32).collect();
        let w: Vec<f32> = (0..200).map(|i| 1.0 + (i % 7) as f32).collect();
        let c = cfg(8, 30);
        let a = build_grid(&col, Some(&w), &c, 99, FeatureId(4)).unwrap();
        // Reverse BOTH in lockstep: same (value, weight) multiset, different row order.
        let col_r: Vec<f32> = col.iter().rev().copied().collect();
        let w_r: Vec<f32> = w.iter().rev().copied().collect();
        let b = build_grid(&col_r, Some(&w_r), &c, 99, FeatureId(4)).unwrap();
        assert_eq!(a, b, "weighted grid must be permutation-invariant");
    }

    #[test]
    fn build_grid_midpoint_branch_at_max_bin_ceiling() {
        use crate::data::bin::bin;
        // Exactly 254 distinct values with max_bin 254 ⇒ midpoint branch at the ceiling.
        let col: Vec<f32> = (0..254).map(|i| i as f32).collect();
        let g = build_grid(&col, None, &cfg(254, 200_000), 0, FeatureId(0)).unwrap();
        assert_eq!(g.borders.len(), 253, "midpoint borders = distinct - 1");
        assert_eq!(g.n_bins, 255);
        assert!(is_strictly_ascending(&g.borders));
        // Every value bins in 1..=254 with no panic/overflow.
        for &v in &col {
            let b = bin(v, &g).unwrap();
            assert!((1..=254).contains(&b));
        }
    }

    #[test]
    fn build_grid_quantile_branch_caps_at_max_bin() {
        // 255 distinct values, max_bin 254 ⇒ quantile branch, borders.len() <= 253.
        let col: Vec<f32> = (0..255).map(|i| i as f32).collect();
        let g = build_grid(&col, None, &cfg(254, 200_000), 0, FeatureId(0)).unwrap();
        assert!(g.borders.len() <= 253, "got {} borders", g.borders.len());
        assert!(g.n_bins <= 255);
        assert!(is_strictly_ascending(&g.borders));
    }

    #[test]
    fn max_bin_255_is_rejected() {
        let col = [1.0_f32, 2.0];
        assert!(matches!(
            build_grid(&col, None, &cfg(255, 200_000), 0, FeatureId(0)),
            Err(PbError::InvalidConfig { .. })
        ));
    }
}
