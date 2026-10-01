//! Property tests for binning (spec §03.10/§13.3): borders strictly ascending; `bin`
//! non-decreasing on finite values; every binned value in `0..n_bins`; ONLY NaN maps
//! to bin 0 (±inf/extremes clamp to the first/last finite bin); no panics; the grid
//! depends only on the value multiset (permutation-invariant).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp
)]

use proptest::prelude::*;
use t_boost_core::{bin, build_grid, BinConfig, BorderFamily, FeatureId};

fn small_cfg() -> BinConfig {
    // Small max_bin + subsample forces both the quantile and the RNG-subsample paths
    // under modest proptest column sizes.
    BinConfig {
        max_bin: 8,
        subsample_for_binning: 16,
        min_data_per_bin: 0,
        border_family: BorderFamily::EqualCount,
    }
}

proptest! {
    #[test]
    fn borders_are_strictly_ascending(col in prop::collection::vec(any::<f32>(), 1..256)) {
        let g = build_grid(&col, None, &small_cfg(), 7, FeatureId(0)).unwrap();
        for w in g.borders.windows(2) {
            prop_assert!(w[0] < w[1], "borders must be strictly ascending: {:?}", g.borders);
        }
        // Cardinality invariant: borders.len() <= max_bin - 1, n_bins == borders.len()+2.
        prop_assert!(g.borders.len() <= 7);
        prop_assert_eq!(usize::from(g.n_bins), g.borders.len() + 2);
    }

    #[test]
    fn every_value_bins_in_range_and_only_nan_is_missing(
        col in prop::collection::vec(any::<f32>(), 1..256)
    ) {
        let g = build_grid(&col, None, &small_cfg(), 3, FeatureId(0)).unwrap();
        for &v in &col {
            let b = bin(v, &g).unwrap();
            prop_assert!(u16::from(b) < g.n_bins, "bin {b} out of range (n_bins {})", g.n_bins);
            if v.is_nan() {
                prop_assert_eq!(b, 0, "NaN must map to the missing bin 0");
            } else {
                // Finite AND ±inf are non-missing: they clamp into 1..=n_data_bins.
                prop_assert!(b >= 1, "non-NaN value {v} must not map to missing bin 0");
            }
        }
    }

    #[test]
    fn bin_is_non_decreasing_on_finite(
        col in prop::collection::vec(-1.0e6f32..1.0e6f32, 2..256)
    ) {
        let g = build_grid(&col, None, &small_cfg(), 11, FeatureId(1)).unwrap();
        let mut finite: Vec<f32> = col.iter().copied().filter(|v| v.is_finite()).collect();
        finite.sort_by(f32::total_cmp);
        let mut prev = 0u8;
        for v in finite {
            let b = bin(v, &g).unwrap();
            prop_assert!(b >= prev, "bin must be non-decreasing in value");
            prev = b;
        }
    }

    #[test]
    fn grid_is_permutation_invariant(
        col in prop::collection::vec(any::<f32>(), 1..256),
        seed in any::<u64>(),
    ) {
        let mut reversed = col.clone();
        reversed.reverse();
        let a = build_grid(&col, None, &small_cfg(), seed, FeatureId(5)).unwrap();
        let b = build_grid(&reversed, None, &small_cfg(), seed, FeatureId(5)).unwrap();
        prop_assert_eq!(a, b, "grid must depend only on the value multiset, not row order");
    }

    // The WEIGHTED path must be permutation-invariant too (the (value, weight)
    // tie-break is what guarantees it). Pairs are reversed in lockstep, so the
    // (value, weight) multiset is unchanged but the row order is not. Weights are
    // strictly positive so the quantile total is never zero.
    #[test]
    fn weighted_grid_is_permutation_invariant(
        pairs in prop::collection::vec((any::<f32>(), 0.1f32..100.0f32), 1..256),
        seed in any::<u64>(),
    ) {
        let col: Vec<f32> = pairs.iter().map(|p| p.0).collect();
        let w: Vec<f32> = pairs.iter().map(|p| p.1).collect();
        let col_r: Vec<f32> = col.iter().rev().copied().collect();
        let w_r: Vec<f32> = w.iter().rev().copied().collect();
        let a = build_grid(&col, Some(&w), &small_cfg(), seed, FeatureId(6)).unwrap();
        let b = build_grid(&col_r, Some(&w_r), &small_cfg(), seed, FeatureId(6)).unwrap();
        prop_assert_eq!(a, b, "weighted grid must depend only on the (value,weight) multiset");
    }
}
