//! The frozen deterministic re-seeding mixer (`pb_seed`/`pb_rng`, spec §02.3b) and
//! the [`Stage`] coordinate its streams are indexed by.
//!
//! Every randomized stage in the library — binning subsample, row/column sampling,
//! stochastic rounding, bagging, categorical folds, split noise, DART, holdout — draws
//! from `pb_rng(base, round, stage, block)`. Because the stream is a pure function of
//! the work-unit coordinates, draws are position-stable and **independent of thread
//! count**, which is what makes the §1 determinism `[GATE]` hold.

use rand::SeedableRng;
use rand_pcg::Pcg64;

/// Frozen deterministic re-seeding (spec §02.3b). `splitmix64` is the standard
/// 64-bit mixer; this exact function is part of the determinism `[GATE]` contract
/// and MUST NOT change without a schema/repro bump.
///
/// The per-`(round, stage, block)` stream is a pure function of the base seed and
/// the work-unit coordinates, so draws are position-stable and **independent of
/// thread count**. Downstream: `Pcg64::seed_from_u64(pb_seed(base, round, stage,
/// block))`. The `wrapping_mul`/`>>`/`^` here are the documented exception to the
/// integer-overflow trap — wrapping is intentional in the mixer.
#[must_use]
pub fn pb_seed(base: u64, round: u32, stage: u32, block: u32) -> u64 {
    let mut z = base
        ^ u64::from(round).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ u64::from(stage).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ u64::from(block).wrapping_mul(0x94D0_49BB_1331_11EB);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The randomized stage a re-seed belongs to — the `stage` coordinate of
/// [`pb_seed`] (spec §1/§02.3b). The discriminants are **frozen** (part of the
/// determinism `[GATE]` contract): never renumber an existing stage, only append.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Per-feature binning subsample (§03.3).
    Binning = 1,
    /// Row/feature subsampling & MVS (§06.7).
    Sample = 2,
    /// Stochastic rounding for quantized histograms (§06/§11, v1.5).
    Quantize = 3,
    /// Bagged ensemble selection (§09.6).
    Bagging = 4,
    /// Categorical target-statistic permutations/folds (§04.3).
    Categorical = 5,
    /// Split-score random-strength tie/regularization noise (§09.6).
    SplitNoise = 6,
    /// DART tree-dropout masks (§09.6).
    Dart = 7,
    /// Per-tree column subsampling masks (§06.5).
    Cols = 8,
    /// Internal validation holdout carving for early stopping (§06.6).
    Holdout = 9,
}

/// Construct the per-work-unit [`Pcg64`] from the frozen [`pb_seed`] mixer
/// (spec §02.3b). The single canonical way the library obtains a stream, so every
/// randomized stage is position-stable and thread-count-independent.
#[must_use]
pub fn pb_rng(base: u64, round: u32, stage: Stage, block: u32) -> Pcg64 {
    Pcg64::seed_from_u64(pb_seed(base, round, stage as u32, block))
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

    /// Known-vector test pinning the frozen mixer so the determinism RNG can never
    /// silently drift. The all-zero input is provable by hand (`z` stays `0` through
    /// every step), so it anchors the implementation; the remaining vectors are
    /// frozen outputs of THIS `splitmix64` — regenerate ONLY with a documented
    /// schema/repro bump.
    #[test]
    fn pb_seed_is_frozen() {
        // Pure function of the coordinates: same inputs ⇒ same output, always.
        assert_eq!(pb_seed(7, 3, 2, 9), pb_seed(7, 3, 2, 9));

        // All-zero input mixes to zero (0 ⊕ 0 = 0; every multiply is of 0).
        assert_eq!(pb_seed(0, 0, 0, 0), 0);

        // Distinct coordinates yield distinct streams (no trivial collisions).
        assert_ne!(pb_seed(1, 0, 0, 0), pb_seed(0, 0, 0, 0));
        assert_ne!(pb_seed(0, 1, 0, 0), pb_seed(0, 0, 0, 0));
        assert_ne!(pb_seed(0, 0, 1, 0), pb_seed(0, 0, 0, 0));
        assert_ne!(pb_seed(0, 0, 0, 1), pb_seed(0, 0, 0, 0));

        // Frozen reference vectors (outputs of this exact mixer).
        assert_eq!(pb_seed(1, 0, 0, 0), 6_238_072_747_940_578_789);
        assert_eq!(pb_seed(0, 1, 0, 0), 16_294_208_416_658_607_535);
        assert_eq!(pb_seed(42, 1, 2, 3), 1_962_896_480_199_194_022);
    }
}
