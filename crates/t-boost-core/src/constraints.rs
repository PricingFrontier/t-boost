//! Interaction selection & monotone constraints (spec §2.9 / §07). This module owns
//! the serialized interaction policy, name-keyed monotone constraints, and the
//! order-3 Walsh-Hadamard primitive used as an independent oracle for tree-local
//! interaction strength. The online screening accumulator and soft heredity/FAST/Sobol
//! admission prior build on these pieces.

use crate::error::PbError;
use crate::explain::FeatureSet;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A monotonicity direction for one feature (spec §07).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MonoSign {
    /// The response must be non-decreasing in this feature.
    Increasing,
    /// The response must be non-increasing in this feature.
    Decreasing,
    /// No monotone constraint.
    None,
}

/// Monotone constraints keyed by feature NAME, never positional (spec §2.9 / §07).
/// A `BTreeMap` for deterministic iteration order (it can be serialized as part of a
/// fit record).
pub type MonotoneMap = BTreeMap<String, MonoSign>;

/// The whole-tree interaction constraint plus the optional feature-group whitelist
/// (spec §2.9 / §07). `groups` (when `Some`) restricts each tree's distinct-raw
/// support to lie within one declared group; `None` = unconstrained.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InteractionPolicy {
    /// Maximum interaction order, in `1..=`[`crate::engine::MAX_ORDER`]; default
    /// [`crate::engine::LEGACY_MAX_ORDER`] (3).
    ///
    /// Caps DISTINCT raw features per tree — the fANOVA order, and therefore the number
    /// of axes an exported table may have. Paired with, and orthogonal to,
    /// [`InteractionPolicy::max_depth`]: order is how many features an effect couples,
    /// depth is how finely it resolves them. `max_depth` must be at least `max_order`,
    /// since a tree needs one level per distinct feature.
    ///
    /// Raising it above 3 is opt-in. Exactness is not the constraint — the purification
    /// cascade is n-dimensional and a 4-way effect decomposes losslessly by the same
    /// algorithm (see [`crate::engine::MAX_ORDER`]) — READABILITY is, so an order-4
    /// support is charged against a budget shrunk by
    /// [`InteractionPolicy::table_budget_order_shrink`] per order above 3, and a split
    /// that would raise a tree's order past 3 must clear a doubled gain hurdle.
    #[serde(default = "default_max_order")]
    pub max_order: u8,
    /// Maximum tree DEPTH (shared level tests), in `3..=`[`crate::engine::MAX_DEPTH`] (8);
    /// default `3`.
    ///
    /// A RESOLUTION cap, not a capacity-in-order cap. Once `max_order` distinct raw
    /// features are on a tree the split-finder's fresh-admission list is empty by
    /// construction, so levels beyond the third can only REUSE a feature already on the
    /// tree with a refined threshold — a valid lower-order refinement per I1
    /// (spec §00: "Reusing a raw feature is a valid lower-order refinement"). A depth-6
    /// tree on 3 raw features is therefore still exactly a 3rd-order fANOVA function:
    /// tables get finer grids, never more axes.
    ///
    /// What it buys is estimation, not expressiveness: a depth-3 ENSEMBLE already spans
    /// the whole `<=3rd`-order space. A deeper tree fits its finer partition JOINTLY, in
    /// one Newton step under one shrinkage and one `lambda`, where the depth-3 ensemble
    /// reaches the same function only through a sequence of independently-shrunk 8-cell
    /// fits.
    ///
    /// **Default stays 3** (the 2026-06-22 methodology decision). Raising it is opt-in
    /// and, at the product layer, couples a non-zero `min_data_in_leaf` floor — cells
    /// hold fewer rows at depth 6 and the credibility floor is inert by default.
    #[serde(default = "default_max_depth")]
    pub max_depth: u8,
    /// Allowed co-occurrence groups; `None` = unconstrained.
    pub groups: Option<Vec<FeatureSet>>,
    /// Soft table-size prior exponent (§07.3/§07.4). A value of `0.0` is exactly
    /// inert; positive values down-rank supports whose projected table cells exceed
    /// [`InteractionPolicy::table_budget_cells`], but never hard-reject them.
    #[serde(default = "default_table_budget_beta")]
    pub table_budget_beta: f32,
    /// Cell budget used by the soft table-size prior. This is an admission score
    /// prior, separate from the hard [`crate::TableBudget`] allocation firewall.
    #[serde(default = "default_table_budget_cells")]
    pub table_budget_cells: u64,
    /// How much [`InteractionPolicy::table_budget_cells`] SHRINKS per interaction order
    /// above [`crate::engine::LEGACY_MAX_ORDER`]; default `2.0`. `1.0` is exactly inert.
    ///
    /// The soft prior charges `Π extent(raw)` over a support's distinct raws, so a fourth
    /// feature already multiplies the projected cell count rather than adding to it. This
    /// shrink is the SEPARATE, deliberate statement that a 4-way table of `n` cells is
    /// harder to read than a 3-way table of `n` cells — you page through it in 3-D slices
    /// — so it should be allowed fewer of them. At the lifted default budget of 4096 an
    /// order-3 support is measured against 4096 cells (about `16^3`) and an order-4
    /// support against 2048 (about `6.7^4`).
    ///
    /// Exactly inert at order `<= LEGACY_MAX_ORDER`: the exponent is
    /// `order.saturating_sub(LEGACY_MAX_ORDER)`, which is `0` for every support a
    /// pre-lift fit could realize, so the depth-3/order-3 default path is byte-identical.
    #[serde(default = "default_table_budget_order_shrink")]
    pub table_budget_order_shrink: f32,
}

impl Default for InteractionPolicy {
    fn default() -> Self {
        Self {
            max_order: default_max_order(),
            max_depth: default_max_depth(),
            groups: None,
            table_budget_beta: default_table_budget_beta(),
            table_budget_cells: default_table_budget_cells(),
            table_budget_order_shrink: default_table_budget_order_shrink(),
        }
    }
}

fn default_max_depth() -> u8 {
    crate::engine::LEGACY_MAX_DEPTH as u8
}

fn default_max_order() -> u8 {
    crate::engine::LEGACY_MAX_ORDER as u8
}

fn default_table_budget_beta() -> f32 {
    0.5
}

fn default_table_budget_cells() -> u64 {
    2_000_000
}

fn default_table_budget_order_shrink() -> f32 {
    2.0
}

/// Per-leaf credibility floors (spec §07.2 / §07.6). §07 OWNS these; they shape *which*
/// shared levels may fire and stabilize thin/low-exposure leaves. All-zero (the default)
/// is exactly inert — no candidate is rejected and no leaf value is shrunk, so a fit with
/// the default floor is byte-identical to one with floors disabled.
///
/// The first three are HARD per-candidate rejects evaluated across **all cells of the
/// shared level** (one under-supported child cell vetoes the candidate — the symmetric
/// credibility guarantee actuaries expect): `min_data_in_leaf` on the exact binned row
/// count, `min_sum_hessian_in_leaf` on the per-cell Σh, `min_weight_sum_in_leaf` on the
/// per-cell Σw (e.g. exposure, stable under a log link). These are DISTINCT from §03's
/// grid-build `min_data_per_bin` (rare-bin merge at binning time). `path_smooth` (0 = off)
/// shrinks each fitted leaf toward its oblivious-tree parent node, applied **after** the
/// monotone clamp and then re-clamped (§07.6); it is value-level only, so structure, the
/// ≤3-feature property, and exact decomposability are untouched.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CredibilityFloor {
    /// Minimum exact binned row count per cell (`0` = off).
    pub min_data_in_leaf: u32,
    /// Minimum Σh (curvature mass) per cell (`0.0` = off).
    pub min_sum_hessian_in_leaf: f32,
    /// Minimum Σw (e.g. exposure mass) per cell (`0.0` = off).
    pub min_weight_sum_in_leaf: f32,
    /// Parent-shrinkage strength (`0.0` = off). Larger ⇒ more shrinkage toward the parent.
    pub path_smooth: f32,
}

impl Default for CredibilityFloor {
    fn default() -> Self {
        Self {
            min_data_in_leaf: 0,
            min_sum_hessian_in_leaf: 0.0,
            min_weight_sum_in_leaf: 0.0,
            path_smooth: 0.0,
        }
    }
}

impl CredibilityFloor {
    /// Validate the floor.
    ///
    /// # Errors
    /// [`PbError::InvalidConfig`] if any float floor is non-finite or negative.
    pub fn validate(&self) -> Result<(), PbError> {
        for (name, value) in [
            ("min_sum_hessian_in_leaf", self.min_sum_hessian_in_leaf),
            ("min_weight_sum_in_leaf", self.min_weight_sum_in_leaf),
            ("path_smooth", self.path_smooth),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(PbError::InvalidConfig {
                    what: format!("CredibilityFloor.{name} must be finite and >= 0, got {value}"),
                });
            }
        }
        Ok(())
    }

    /// `true` if none of the three hard floors can reject a candidate (the fast path:
    /// no per-cell support accounting is needed). `path_smooth` is a value-level clamp,
    /// not a candidate reject, so it does not affect this.
    #[must_use]
    pub fn rejects_nothing(&self) -> bool {
        self.min_data_in_leaf == 0
            && self.min_sum_hessian_in_leaf <= 0.0
            && self.min_weight_sum_in_leaf <= 0.0
    }
}

/// §07.6 `path_smooth` credibility EVIDENCE: which per-node quantity feeds the Bühlmann blend
/// `Z = n/(n+k)` in `apply_path_smooth` (`engine/split.rs`) and its `leaf_refine` companion
/// shrink (`engine/boost.rs`). Resolved automatically, once per fit, from the loss objective
/// (`engine::boost::credibility_evidence_for_loss`) — never user-set, never persisted. It lives
/// only on the engine-internal `GrowConfig`, not on `CredibilityFloor`/`FitSpec`/`ModelSchema`,
/// so generalizing the evidence measure has no serialization or schema-version footprint.
///
/// - `Count`: raw exact row count (the ONLY behavior before this evidence generalization, and
///   still the right one for Gamma/SquaredError/Logistic/Softmax — GLM theory says their
///   *expected* Fisher information collapses to plain exposure/count, and Gamma's *observed*
///   hessian is outlier-inflated by heavy claims, the wrong direction for a credibility weight).
/// - `Hessian`: the per-node Σh (hessian/curvature mass), used for Poisson and Tweedie — for a
///   log-link frequency/pure-premium objective this is the (observed) claim mass, which
///   correctly flags a zero-inflated cell (many exposure rows, few actual claims) as low-evidence
///   in a way raw row count cannot. Consumed only through [`credibility_evidence_n`], which
///   normalizes the raw Σh to an "effective row count" — see that function's doc for why the
///   raw mass alone is NOT usable as `path_smooth`'s pseudo-count `n` (2026-07-23 amendment).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum CredibilityEvidence {
    /// Raw exact binned row count (`leaf_aggregates`'s `count` / `apply_path_smooth`'s `cnt`).
    #[default]
    Count,
    /// Per-node Σh (`leaf_aggregates`'s `h` / `apply_path_smooth`'s locally-folded `h`),
    /// normalized to effective rows by [`credibility_evidence_n`] before use.
    Hessian,
}

/// §07.6 evidence-to-Bühlmann-count conversion — the single dispatch shared by grow-time
/// `apply_path_smooth` (`engine/split.rs`) and the `leaf_refine` companion shrink
/// (`engine/boost.rs`), so both stages apply the IDENTICAL normalization and "agree" on what a
/// leaf's evidence is worth.
///
/// `Count` evidence is unchanged: the raw row count. `Hessian` evidence divides the per-node/
/// leaf Σh (`h`) by `h_bar` — the tree's own mean per-row hessian (Σh over ALL of this tree's
/// rows, divided by that row count; "the root's aggregate", per-TREE not a global-once
/// constant) — converting a hessian MASS into an "effective row count" on the same scale
/// `path_smooth`'s pseudo-count `k` was calibrated against.
///
/// This normalization is NOT optional: `path_smooth` (§07.6) was calibrated as a pseudo-COUNT
/// (a handful to a few dozen "prior rows"). Poisson's per-row hessian happens to be `O(1)` on
/// unit-weight data, so raw Σh and row count are roughly the same scale there and the un-
/// normalized version looked fine in early testing — but on exposure-weighted objectives
/// (Tweedie especially, `h ∝ w·μ^{2-ρ}`) per-row hessian routinely runs into the
/// hundreds-to-thousands. Feeding that raw mass straight into `Z = n/(n+k)` makes `Z ≈ 1`
/// EVERYWHERE (the "+k" pseudo-count becomes negligible next to a mass that large), silently
/// neutering `path_smooth` for every leaf regardless of how thin it actually is. Found via the
/// `ohlsson_pp` (zero-inflated Tweedie) smoke test, 2026-07-23: the default `path_smooth=10`
/// build scored identically to `path_smooth=0` (38.657 vs 38.659), instead of matching the
/// measured count-evidence-era score of 38.533.
///
/// Falls back to `Count` whenever `h_bar` isn't usable (`<= 0.0` or non-finite — e.g. a
/// degenerate all-zero-hessian tree) rather than dividing by a broken denominator.
#[must_use]
pub(crate) fn credibility_evidence_n(
    evidence: CredibilityEvidence,
    h: f64,
    count: u64,
    h_bar: f64,
) -> f64 {
    if evidence == CredibilityEvidence::Hessian && h_bar.is_finite() && h_bar > 0.0 {
        h / h_bar
    } else {
        count as f64
    }
}

/// Uniform 8-leaf Walsh-Hadamard / Möbius coefficients for one depth-3 oblivious
/// leaf vector (§07.4a).
///
/// Coefficients are indexed by a bitmask over split levels: `0b000` is the constant
/// term, `0b001/010/100` are main effects, `0b011/101/110` are pairs, and `0b111`
/// is the pure triple interaction. The transform is orthonormal up to the standard
/// `1/8` averaging factor, so [`inverse_wht8_uniform`] reconstructs the original leaf
/// vector exactly up to floating-point roundoff.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wht8 {
    /// Coefficients in mask order.
    pub coeffs: [f64; 8],
}

/// Compute uniform Walsh-Hadamard coefficients for an 8-leaf vector.
#[must_use]
pub fn wht8_uniform(leaves: [f64; 8]) -> Wht8 {
    let mut coeffs = [0.0_f64; 8];
    for (mask, slot) in coeffs.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for (leaf, value) in leaves.iter().enumerate() {
            acc += sign(mask, leaf) * value;
        }
        *slot = acc / 8.0;
    }
    Wht8 { coeffs }
}

/// Reconstruct an 8-leaf vector from uniform Walsh-Hadamard coefficients.
#[must_use]
pub fn inverse_wht8_uniform(wht: Wht8) -> [f64; 8] {
    let mut leaves = [0.0_f64; 8];
    for (leaf, slot) in leaves.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for (mask, coeff) in wht.coeffs.iter().enumerate() {
            acc += sign(mask, leaf) * coeff;
        }
        *slot = acc;
    }
    leaves
}

fn sign(mask: usize, leaf: usize) -> f64 {
    if ((mask & leaf).count_ones() & 1) == 0 {
        1.0
    } else {
        -1.0
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::indexing_slicing, clippy::float_cmp)]

    use super::*;

    #[test]
    fn credibility_floor_default_is_inert_and_validates() {
        let d = CredibilityFloor::default();
        assert!(d.rejects_nothing());
        assert!(d.validate().is_ok());
        // Any positive hard floor means the split-finder must track per-cell support.
        assert!(!CredibilityFloor {
            min_data_in_leaf: 1,
            ..CredibilityFloor::default()
        }
        .rejects_nothing());
        // path_smooth alone is a value-level clamp, not a candidate reject.
        assert!(CredibilityFloor {
            path_smooth: 2.0,
            ..CredibilityFloor::default()
        }
        .rejects_nothing());
        // Negative / non-finite floors are rejected.
        assert!(CredibilityFloor {
            min_sum_hessian_in_leaf: -1.0,
            ..CredibilityFloor::default()
        }
        .validate()
        .is_err());
        assert!(CredibilityFloor {
            path_smooth: f32::NAN,
            ..CredibilityFloor::default()
        }
        .validate()
        .is_err());
    }

    /// §07.6 evidence normalization edge case (2026-07-23 amendment, explicitly requested):
    /// `h_bar <= 0.0` or non-finite must fall back to `Count` evidence rather than dividing by
    /// a broken denominator (an all-zero-hessian degenerate tree, or any other pathological
    /// case that could zero or NaN the mean).
    #[test]
    fn credibility_evidence_n_falls_back_to_count_on_bad_h_bar() {
        for bad_h_bar in [0.0_f64, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let n = credibility_evidence_n(CredibilityEvidence::Hessian, 12345.0, 7, bad_h_bar);
            assert_eq!(
                n, 7.0,
                "h_bar={bad_h_bar} must fall back to count evidence, got n={n}"
            );
        }
        // A genuinely usable h_bar does NOT fall back.
        let n = credibility_evidence_n(CredibilityEvidence::Hessian, 100.0, 7, 10.0);
        assert!(
            (n - 10.0).abs() < 1e-9,
            "h=100, h_bar=10 ⇒ n_eff=10, got {n}"
        );
        // Count evidence never reads h/h_bar at all -- even a broken h_bar is irrelevant.
        let n = credibility_evidence_n(CredibilityEvidence::Count, f64::NAN, 7, f64::NAN);
        assert_eq!(n, 7.0);
    }

    #[test]
    fn wht8_round_trips_leaf_values() {
        let leaves = [1.0, -2.0, 3.5, 4.0, -1.0, 0.25, 8.0, -3.0];
        let got = inverse_wht8_uniform(wht8_uniform(leaves));
        for (a, b) in got.iter().zip(leaves) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn wht8_names_constant_main_pair_and_triple_masks() {
        let leaves = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0];
        let coeffs = wht8_uniform(leaves).coeffs;
        assert_eq!(coeffs[0], leaves.iter().sum::<f64>() / 8.0);
        // The single extra bump at leaf 0b111 is visible in the pure triple mask.
        assert!(coeffs[0b111].abs() > 0.0);
        // Main and pair masks are finite and stored in the documented mask positions.
        for coeff in coeffs.iter().take(0b110 + 1).skip(0b001) {
            assert!(coeff.is_finite());
        }
    }
}
