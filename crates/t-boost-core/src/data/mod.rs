//! Data model & binning (spec §03, OWNED HERE). Raw user `f32` columns become the
//! frozen, persisted `u8` representation every later stage reads.
//!
//! The load-bearing decision is the **one global per-feature border grid**: computed
//! once at fit, persisted in the `Model`, and reused bit-identically at fit, predict,
//! and table-export time. This is the algebraic precondition for I2 — every tree
//! shares the same axes, so every tree's 8-cell tensor is exactly representable on
//! the merged grid and `purify(Σ trees) = Σ purify(trees)` (§08).
//!
//! This module owns: [`BorderGrid`], [`BinnedMatrix`], [`AxisKind`]/[`AxisProvenance`],
//! [`BinConfig`]; the border-construction algorithm ([`grid::build_grid`]), the
//! `f32 → u8` map ([`bin::bin`], [`bin::bin_columns`]), the [`TrainBinnedMatrix`] vs
//! [`ServeBinnedMatrix`] audit-on-serve seam, and exposure/offset plumbing
//! ([`exposure::compute_offset`]).

use crate::cat::TsEncodingId;
use serde::{Deserialize, Serialize};

pub mod bin;
pub mod exposure;
pub mod grid;

pub use bin::{
    bin, bin_columns, bin_serve_columns, bin_serve_columns_coded, bin_serve_columns_with,
    bin_train_columns, bin_train_columns_with_holdout, CatServeMaps, CategoricalColumn,
    FittedBinnedData, NumericColumn, ServeCategoricalCodes, ServeCategoricalColumn,
};
pub use exposure::compute_offset;
pub use grid::build_grid;

/// Index into the user's original feature columns — the RAW underlying feature
/// (spec §2.1). `u32` because it appears in serialized provenance (no `usize` on
/// the wire, §02.8). `Ord + Hash` so the ≤3-distinct-raw-feature budget (I1) can be
/// counted over a set of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FeatureId(pub u32);

/// What kind of model column an axis is (spec §2.1). Provenance (below) tracks the
/// RAW feature behind an axis so I1 is enforced on distinct raw features, never on
/// encoded columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AxisKind {
    /// A raw numeric feature, binned by quantile/midpoint borders.
    Numeric,
    /// A categorical feature encoded as a Target-Statistic ordinal axis.
    CategoricalTS {
        /// Which frozen TS encoding produced this axis.
        encoding: TsEncodingId,
    },
    /// The reserved missing-indicator axis (bin 0 semantics).
    Missing,
}

/// Tracks the RAW feature behind an axis (spec §2.1) so the ≤3-feature invariant
/// (I1) is enforced on DISTINCT raw features, not on encoded columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxisProvenance {
    /// The raw feature this axis was derived from.
    pub raw: FeatureId,
    /// How the axis was produced from that raw feature.
    pub kind: AxisKind,
}

/// The count of DISTINCT raw feature ids referenced by `provenance` (`1 + the largest raw id`,
/// `0` for empty provenance) — raw ids must be dense `0..n_raw_features` (the green-spine
/// invariant). Before P1 multi-channel categoricals (design/multichannel-categoricals.md), this
/// always equaled `provenance.len()` (exactly one axis per raw feature); a categorical raw
/// feature fit with multiple channels now owns more than one axis, so the two counts diverge —
/// this is the raw-feature count specifically, never the axis count. Shared by
/// `explain::MergedGrids::from_model` and `TableModel::validate` so both agree on it by
/// construction rather than by two independently-written computations.
#[must_use]
pub fn n_raw_features(provenance: &[AxisProvenance]) -> usize {
    provenance
        .iter()
        .map(|p| p.raw.0 as usize)
        .max()
        .map_or(0, |m| m + 1)
}

/// Every model axis mapped to raw feature `raw`, in provenance (model axis) order — ANY axis
/// kind, unlike `crate::cat::channel_axes_for_raw` (categorical channels only, sorted by
/// encoder id). A raw feature with exactly one axis (numeric or single-channel categorical)
/// has exactly one entry here; more than one entry means a P1 multi-channel categorical raw
/// feature (design/multichannel-categoricals.md), and every such axis is `CategoricalTS` by
/// construction (numeric and categorical axes never share a raw id).
#[must_use]
pub fn axes_for_raw(provenance: &[AxisProvenance], raw: FeatureId) -> Vec<usize> {
    provenance
        .iter()
        .enumerate()
        .filter_map(|(a, p)| (p.raw == raw).then_some(a))
        .collect()
}

/// One feature's border grid (spec §2.2/§03.2). Bin 0 is the reserved MISSING bin.
///
/// **Canonical cardinality (R-BINS).** `borders[k]` is the upper border of data bin
/// `k+1` (strictly ascending, no duplicates). A finite value `v` lands in data bin
/// `1 + (count of borders strictly below v)`, so data bins are `1..=n_data_bins`
/// where `n_data_bins = borders.len() + 1`, and `n_bins = n_data_bins + 1` (data +
/// missing). Constraint: `borders.len() <= max_bin - 1` (default `max_bin = 254`),
/// so `n_bins <= 255` (fits `u8`, values `0..=254`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BorderGrid {
    /// Strictly-ascending upper borders; `borders[k]` is the top of data bin `k+1`.
    /// Length is `n_data_bins - 1`.
    pub borders: Vec<f32>,
    /// Total bin count including the missing bin (`borders.len() + 2`); `<= 255`.
    pub n_bins: u16,
    /// The reserved missing bin id (always `0` in v1).
    pub missing_bin: u8,
}

impl BorderGrid {
    /// The number of data bins (`borders.len() + 1`); missing bin excluded.
    #[must_use]
    pub fn n_data_bins(&self) -> usize {
        self.borders.len() + 1
    }
}

/// Column-major, pre-binned design matrix (spec §2.2). `data[f]` is column `f` as
/// `u8` bin ids. `f32 → u8` binning happens once at ingest; the grid persists in the
/// `Model`. This is the base type; [`TrainBinnedMatrix`]/[`ServeBinnedMatrix`] are
/// its two audit roles (§03.2a).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinnedMatrix {
    /// `[n_features][n_rows]`, column-major; each entry is a `u8` bin id.
    pub data: Vec<Vec<u8>>,
    /// Row count. Fixed-width `u32`: serialized, and `usize` would break
    /// cross-platform byte-equality / the `wasm32` smoke build (§02.8).
    pub n_rows: u32,
    /// Per-feature border grids (parallel to `data`).
    pub grids: Vec<BorderGrid>,
    /// Per-axis provenance (parallel to `data`).
    pub provenance: Vec<AxisProvenance>,
}

/// Fitting-time matrix (spec §03.2a). Numeric columns are binned through the frozen
/// global grids; categorical TS columns carry the **leakage-free** (out-of-fold)
/// encodings used ONLY to grow trees — they are noisy by design and MUST NEVER be
/// accumulated into the `TableBank`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainBinnedMatrix(pub BinnedMatrix);

/// Prediction-time matrix (spec §03.2a). Numeric columns are binned through the SAME
/// frozen grids (numeric binning is fold-independent — Train == Serve there);
/// categorical TS columns are re-encoded through the frozen full-data
/// `CatEncoderStore`. This is the matrix the served model AND the audited
/// `TableBank` are evaluated on (§08).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServeBinnedMatrix(pub BinnedMatrix);

/// The interior-border family used by [`grid::build_grid`] (spec §03.11).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BorderFamily {
    /// Equal-count (equal-frequency) quantile borders. The only family implemented
    /// in v1, and the default.
    EqualCount,
    /// Hessian-weighted (equal Newton-loss-mass) borders. A v1.5 fork (§03.11).
    /// Named here to match the §03.2 canonical type, but **not yet implemented** —
    /// [`BinConfig::validate`] rejects it with `InvalidConfig` so a caller can never
    /// silently receive equal-count grids when they asked for Hessian-weighted ones.
    HessianWeighted,
}

/// Binning configuration (spec §03.2, OWNED HERE; the §06 `Config` references it,
/// never redefines it).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinConfig {
    /// Max bin id budget; default 254. Caps `borders.len() <= max_bin - 1`, so
    /// `n_data_bins <= max_bin` and `n_bins <= max_bin + 1` (bin 0 = missing).
    pub max_bin: u8,
    /// Subsample size for quantile border construction; default `200_000`.
    pub subsample_for_binning: u32,
    /// Numeric bins below this row count merge into their lower neighbour; default
    /// `0` (off).
    pub min_data_per_bin: u32,
    /// Which interior-border family to use; default [`BorderFamily::EqualCount`].
    pub border_family: BorderFamily,
}

impl Default for BinConfig {
    fn default() -> Self {
        Self {
            max_bin: 254,
            subsample_for_binning: 200_000,
            min_data_per_bin: 0,
            border_family: BorderFamily::EqualCount,
        }
    }
}

impl BinConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// [`crate::PbError::InvalidConfig`] if `max_bin` is outside `2..=254`,
    /// `subsample_for_binning == 0`, or `border_family` names an unimplemented family
    /// ([`BorderFamily::HessianWeighted`] is a v1.5 fork — rejected so it can never
    /// silently fall through to equal-count borders).
    pub fn validate(&self) -> Result<(), crate::PbError> {
        if self.max_bin < 2 {
            return Err(crate::PbError::InvalidConfig {
                what: format!("max_bin must be >= 2, got {}", self.max_bin),
            });
        }
        // max_bin == 255 would realize n_bins == 256 and bin id 255, breaking the
        // R-BINS invariant (n_bins <= 255, ids 0..=254, "255 free; never 256").
        if self.max_bin > 254 {
            return Err(crate::PbError::InvalidConfig {
                what: format!("max_bin must be <= 254, got {}", self.max_bin),
            });
        }
        if self.subsample_for_binning == 0 {
            return Err(crate::PbError::InvalidConfig {
                what: "subsample_for_binning must be > 0".into(),
            });
        }
        // Fail fast on an accepted-but-unimplemented border family: v1 builds only
        // EqualCount borders (§03.11), and build_grid does not branch on this field,
        // so accepting HessianWeighted would silently return equal-count grids.
        if matches!(self.border_family, BorderFamily::HessianWeighted) {
            return Err(crate::PbError::InvalidConfig {
                what: "BorderFamily::HessianWeighted is a v1.5 fork (§03.11); v1 supports only EqualCount".into(),
            });
        }
        Ok(())
    }
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
    use crate::PbError;

    #[test]
    fn default_equal_count_config_validates() {
        assert!(BinConfig::default().validate().is_ok());
        assert_eq!(BinConfig::default().border_family, BorderFamily::EqualCount);
    }

    #[test]
    fn hessian_weighted_is_rejected_in_v1() {
        // The variant exists (matches the §03.2 canonical type) but v1 must NOT
        // silently fall back to equal-count borders — it fails fast instead.
        let cfg = BinConfig {
            border_family: BorderFamily::HessianWeighted,
            ..BinConfig::default()
        };
        assert!(matches!(cfg.validate(), Err(PbError::InvalidConfig { .. })));
        // build_grid / bin_columns both call validate() first, so the rejection
        // propagates through the whole binning entrypoint.
        let g = grid::build_grid(&[1.0, 2.0, 3.0], None, &cfg, 0, FeatureId(0));
        assert!(matches!(g, Err(PbError::InvalidConfig { .. })));
    }

    #[test]
    fn max_bin_must_be_in_2_254() {
        for bad in [0u8, 1, 255] {
            let cfg = BinConfig {
                max_bin: bad,
                ..BinConfig::default()
            };
            assert!(
                matches!(cfg.validate(), Err(PbError::InvalidConfig { .. })),
                "max_bin={bad} should be rejected"
            );
        }
        let ok = BinConfig {
            max_bin: 254,
            ..BinConfig::default()
        };
        assert!(ok.validate().is_ok());
    }
}
