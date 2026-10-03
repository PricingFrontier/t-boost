//! The tables-only served model (`TableModel`): the purified [`TableBank`] IS the model.
//!
//! A normal [`Model`] serves by summing its oblivious trees; its [`TableBank`] is derived
//! on demand by [`Model::explain`] purely as an *explanation*. A `TableModel` inverts that:
//! it drops the trees and makes the purified bank the deployable artifact, scored by the
//! lossless LUT-sum (`f0 + Σ_u f_u`). Because purify is lossless (the G0 reconstruction
//! gate, §08), an *un-pruned* `TableModel` predicts identically to the ensemble it came from
//! — it is a re-representation, not an approximation.
//!
//! Its reason to exist is **table pruning** (`crate::prune`): once the bank is the model, a
//! table is removed simply by omitting it from the LUT-sum, and the pruned bank is a NEW
//! exact model reconstructed against *its own* LUT-sum (never the discarded ensemble).
//!
//! v1 scope: single-output (regression + binary) only; numeric + categorical axes served
//! through the same frozen `schema.cat_encoders`/`provenance` as [`Model`].

use crate::data::{AxisProvenance, BinnedMatrix, BorderGrid, ServeBinnedMatrix};
use crate::engine::{inverse_link, ExactnessMode, Model, ModelSchema};
use crate::error::PbError;
use crate::explain::{AxisId, RefMeasure, TableBank, Tensor};
use crate::loss::Link;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A tables-only served model: a purified [`TableBank`] plus exactly the metadata a
/// [`Model`] needs to bin raw inputs and map to response space — everything except the
/// trees. The served raw score is `bank.score(x)` (`= f0 + Σ_u f_u`), so **do not** add
/// `model.f0` separately: `bank.f0` already equals `E_w[F_ens]` (it absorbs `model.f0`
/// plus the folded main-effect marginal mass; enforced by the MassConservation gate).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableModel {
    /// The purified (and possibly pruned) fANOVA tables — the model itself.
    pub bank: TableBank,
    /// Shared per-axis binning grids (the ORIGINAL model grids), for raw ingest + validation.
    pub grids: Vec<BorderGrid>,
    /// Per-axis provenance (maps each axis to its raw feature).
    pub provenance: Vec<AxisProvenance>,
    /// The inverse-link family.
    pub link: Link,
    /// Exact / Approximate firewall state. A pruned tables-only bank is `Exact` against its
    /// own LUT-sum.
    pub mode: ExactnessMode,
    /// Serve/export metadata (feature names + categorical encoders + classifier labels + objective).
    pub schema: ModelSchema,
    /// Monotone wire-version covering `TableModel` and `schema`.
    pub schema_version: u32,
}

impl TableModel {
    /// Build a tables-only model from a fitted [`Model`] and the serve matrix + reference
    /// measure the bank is purified under (the same inputs as [`Model::explain`]). The
    /// resulting model predicts identically to `model` (within the §08 reconstruction
    /// tolerance) until it is pruned.
    ///
    /// # Errors
    /// Propagates [`Model::explain`] (firewall, budget, or gate failures).
    pub fn from_model(
        model: &Model,
        x: &ServeBinnedMatrix,
        w: RefMeasure,
    ) -> Result<Self, PbError> {
        let bank = model.explain(x, w)?;
        Ok(Self::from_model_and_bank(model, bank))
    }

    /// Wrap an already-built (possibly pruned) [`TableBank`] with `model`'s serve metadata.
    ///
    /// The stamp is the MAX of what the source model claimed and what this bank's own bytes
    /// require. Those can differ: a depth-3 order-3 tree model is stamped
    /// `SCHEMA_VERSION_UNLIFTED`, but its purified bank may carry a factored effect, whose
    /// length-carrying box encoding needs a v4 reader (see `serialize::SCHEMA_VERSION`).
    #[must_use]
    pub fn from_model_and_bank(model: &Model, bank: TableBank) -> Self {
        let schema_version = model
            .schema_version
            .max(crate::serialize::required_tables_version(&bank));
        Self {
            bank,
            grids: model.grids.clone(),
            provenance: model.provenance.clone(),
            link: model.link,
            mode: model.mode.clone(),
            schema: model.schema.clone(),
            schema_version,
        }
    }

    /// Number of served features.
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.grids.len()
    }

    /// Validate structure after construction or deserialization — the §10 load gate for a
    /// tables-only model. Checks fixed-width schema consistency, grid well-formedness (both
    /// the serve grids and the bank's merged grids), and every table's cell finiteness and
    /// shape/sparse-backing soundness; the full five-gate exactness was established when the
    /// bank was built/pruned.
    ///
    /// # Errors
    /// [`PbError::Serialization`] for a schema-version mismatch; [`PbError::ShapeMismatch`]
    /// for inconsistent parallel metadata or tensor/axis shape corruption;
    /// [`PbError::InvalidInput`] for a non-finite intercept/cell/border or a malformed grid.
    pub fn validate(&self) -> Result<(), PbError> {
        // A tables-only model carries NO trees, so the DEPTH lift can never raise the
        // version its contents require — but the ORDER lift can, through the factored box
        // encoding, so the floor is content-derived. Accept the whole supported band above
        // that floor so a document written by a newer build still loads.
        let required = crate::serialize::required_tables_version(&self.bank);
        if self.schema_version > crate::serialize::SCHEMA_VERSION || self.schema_version < required
        {
            return Err(PbError::Serialization(format!(
                "tables model schema_version {} outside {}..={}",
                self.schema_version,
                required,
                crate::serialize::SCHEMA_VERSION
            )));
        }
        if !self.bank.f0.is_finite() {
            return Err(PbError::InvalidInput {
                what: format!("tables model bank.f0 must be finite, got {}", self.bank.f0),
            });
        }
        if self.grids.len() != self.provenance.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "tables model grids len {} != provenance len {}",
                    self.grids.len(),
                    self.provenance.len()
                ),
            });
        }
        // `merged_grids` is per RAW FEATURE (its own doc: "per-raw-feature merged grid"),
        // `self.grids` is per AXIS — these coincided before P1 multi-channel categoricals
        // (exactly one axis per raw feature always held) but diverge once a categorical raw
        // feature owns more than one channel axis, so this checks against the raw-feature
        // count, not `self.grids.len()`.
        let n_raw = crate::data::n_raw_features(&self.provenance);
        if self.bank.merged_grids.len() != n_raw {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "tables model bank merged_grids len {} != raw feature count {}",
                    self.bank.merged_grids.len(),
                    n_raw
                ),
            });
        }
        if self.schema.feature_names.len() != self.grids.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "tables model feature_names len {} != feature count {}",
                    self.schema.feature_names.len(),
                    self.grids.len()
                ),
            });
        }
        if self.schema.feature_kinds.len() != self.grids.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "tables model feature_kinds len {} != feature count {}",
                    self.schema.feature_kinds.len(),
                    self.grids.len()
                ),
            });
        }
        // Grid well-formedness: mirrors Model::validate's grid loop (engine/mod.rs) —
        // duplicated rather than shared since this is the only tables-only load gate and
        // the tree-model module is not a dependency of this one. `self.grids` is per AXIS —
        // every entry is an ordinary single-channel grid, joint or not (a channel's own model
        // axis always has a real numeric `BorderGrid`), so it keeps the strict check.
        for (axis, grid) in self.grids.iter().enumerate() {
            validate_border_grid("serve", axis, grid)?;
        }
        // `self.bank.merged_grids` is per RAW FEATURE: a P1 multi-channel raw feature's entry
        // is a placeholder (empty borders, `n_bins` = its joint cell count — see
        // `MergedGrids::border_grids`'s doc) rather than a real numeric grid, so it needs the
        // joint-tolerant check instead of the strict one — cross-checked against `provenance`
        // (never inferred from the numbers alone, which could coincidentally match the
        // pattern for a genuinely malformed ordinary grid).
        for (r, grid) in self.bank.merged_grids.iter().enumerate() {
            let raw = crate::data::FeatureId(u32::try_from(r).map_err(|_| PbError::Internal {
                what: "raw feature index exceeded u32 validating merged grids".into(),
            })?);
            let is_joint = crate::cat::channel_axes_for_raw(&self.provenance, raw).len() > 1;
            validate_merged_grid("bank merged", r, grid, is_joint)?;
        }
        // Every table: finite cells/variance, axes parallel to the feature set, tensor shape
        // consistent with the declared axes, and (for sparse backing) that per-cell lookup
        // agrees with the full materialization -- the §10 gap this hardens (spec review
        // 2026-07-11): NaN/Inf cells and out-of-range/unsorted sparse entries previously
        // deserialized cleanly and served silently wrong predictions.
        for (idx, table) in self.bank.tables.iter().enumerate() {
            let what = format!("table {idx} {:?}", table.u);
            if !table.variance.is_finite() {
                return Err(PbError::InvalidInput {
                    what: format!("{what} variance must be finite, got {}", table.variance),
                });
            }
            if table.axes.len() != table.u.order() {
                return Err(PbError::ShapeMismatch {
                    what: format!(
                        "{what} has {} axes but feature_set order {}",
                        table.axes.len(),
                        table.u.order()
                    ),
                });
            }
            validate_effect_identity(&what, &table.u, &table.axes, self.bank.merged_grids.len())?;
            for (axis_idx, axis) in table.axes.iter().enumerate() {
                validate_axis(&what, axis_idx, axis)?;
            }
            let axis_cells: Vec<u32> = table.axes.iter().map(|a| a.cells).collect();
            if table.values.shape_u32() != axis_cells.as_slice() {
                return Err(PbError::ShapeMismatch {
                    what: format!(
                        "{what} values tensor shape {:?} != axes cells {axis_cells:?}",
                        table.values.shape_u32()
                    ),
                });
            }
            if table.support.shape_u32() != table.values.shape_u32() {
                return Err(PbError::ShapeMismatch {
                    what: format!("{what} support tensor shape != values tensor shape"),
                });
            }
            validate_tensor(&format!("{what} values"), &table.values)?;
            validate_tensor(&format!("{what} support"), &table.support)?;
            if let Some(band) = &table.se_band {
                if band.per_cell.shape_u32() != table.values.shape_u32() {
                    return Err(PbError::ShapeMismatch {
                        what: format!("{what} se_band tensor shape != values tensor shape"),
                    });
                }
                validate_tensor(&format!("{what} se_band"), &band.per_cell)?;
            }
        }
        // Factored order-3 effects (over-budget triples kept unbaked, §08.10). Only
        // `u`/`axes`/`variance` are public on `FactoredEffect` -- its per-tree boxes are an
        // explain.rs implementation detail this load gate cannot reach from here; check what
        // IS visible rather than skipping the field outright.
        for (idx, triple) in self.bank.factored.iter().enumerate() {
            let what = format!("factored {idx} {:?}", triple.u);
            if !triple.variance.is_finite() {
                return Err(PbError::InvalidInput {
                    what: format!("{what} variance must be finite, got {}", triple.variance),
                });
            }
            if triple.axes.len() != triple.u.order() {
                return Err(PbError::ShapeMismatch {
                    what: format!(
                        "{what} has {} axes but feature_set order {}",
                        triple.axes.len(),
                        triple.u.order()
                    ),
                });
            }
            validate_effect_identity(&what, &triple.u, &triple.axes, self.bank.merged_grids.len())?;
            for (axis_idx, axis) in triple.axes.iter().enumerate() {
                validate_axis(&what, axis_idx, axis)?;
            }
            // The box-shape gate: see `FactoredEffect::validate_shape` for why this is a
            // load contract and not a debug assert.
            triple.validate_shape()?;
        }
        Ok(())
    }

    /// Guard an incoming already-binned matrix against this model's grids/provenance
    /// (mirrors [`Model`]'s serve-time validation), so the merged-cell mapping is sound.
    fn validate_binned_matrix(&self, x: &BinnedMatrix) -> Result<(), PbError> {
        if x.data.len() != self.grids.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "matrix has {} columns, tables model has {} features",
                    x.data.len(),
                    self.grids.len()
                ),
            });
        }
        if x.grids != self.grids {
            return Err(PbError::ShapeMismatch {
                what: "matrix grids do not match tables model grids".into(),
            });
        }
        if x.provenance != self.provenance {
            return Err(PbError::ShapeMismatch {
                what: "matrix provenance does not match tables model provenance".into(),
            });
        }
        Ok(())
    }

    /// [`TableModel::row_cells`] column-major: `out[raw][row]` is the merged cell of `row` on
    /// raw feature `raw` — the same cells, built a raw feature at a time (in parallel) instead of
    /// one allocated row at a time. What banding consumes.
    ///
    /// # Errors
    /// Propagates the cell-map build and the per-row cell fill.
    pub fn column_cells(&self, x: &BinnedMatrix) -> Result<Vec<Vec<u32>>, PbError> {
        let maps =
            crate::scoring::build_cell_maps(&self.bank.merged_grids, &self.schema.cat_encoders, x)?;
        crate::scoring::column_cells(x, &maps)
    }

    /// Merged-grid cell index of every row on every raw feature: one `n_raw`-long `u32` row
    /// per data row, in raw-feature order — exactly the cells [`TableModel::score_raw`]
    /// looks up. The aggregation key for an actual-versus-expected by rating-factor level
    /// (2026-09-06).
    ///
    /// # Errors
    /// Propagates the cell-map build and the per-row cell fill.
    pub fn row_cells(&self, x: &BinnedMatrix) -> Result<Vec<Vec<u32>>, PbError> {
        let maps =
            crate::scoring::build_cell_maps(&self.bank.merged_grids, &self.schema.cat_encoders, x)?;
        let n_raw = self.bank.merged_grids.len();
        let n_rows = x.n_rows as usize;
        let mut out = Vec::with_capacity(n_rows);
        for row in 0..n_rows {
            let mut cells = vec![0u32; n_raw];
            crate::scoring::fill_row_cells(x, &maps, row, &mut cells)?;
            out.push(cells);
        }
        Ok(out)
    }

    /// This model's bank re-centred on the rows of `x`, weighted by `mass` (per row; `None`
    /// counts rows), under `w` — see [`TableBank::recentre_on`]. The same function, so the same
    /// predictions; only how it is shared between tables changes.
    ///
    /// # Errors
    /// Propagates the cell lookup and [`TableBank::recentre_on`].
    pub fn recentred_bank(
        &self,
        x: &BinnedMatrix,
        mass: Option<&[f32]>,
        w: RefMeasure,
    ) -> Result<TableBank, PbError> {
        self.validate_binned_matrix(x)?;
        let cells = self.column_cells(x)?;
        self.bank.recentre_on(&cells, mass, w)
    }

    /// The raw feature ids of every effect, in the order [`TableModel::effect_contributions`]
    /// reports them: the dense tables, then the factored effects.
    #[must_use]
    pub fn effect_feature_sets(&self) -> Vec<Vec<u32>> {
        self.bank
            .tables
            .iter()
            .map(|t| &t.u)
            .chain(self.bank.factored.iter().map(|ft| &ft.u))
            .map(|u| u.0.iter().map(|f| f.0).collect())
            .collect()
    }

    /// Per-row value of every effect `f_u(x_u)`, row-major (`n_rows × n_effects`, effects in
    /// [`TableModel::effect_feature_sets`] order). `bank.f0` plus a row's values, summed in this
    /// order, is exactly the `f64` LUT-sum [`TableModel::score_raw`] rounds to `f32`: the same
    /// cells, tables and summation order as `scoring::score_bank_binned_with_maps`.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] on a width mismatch or an output too large to address;
    /// propagated cell-map and per-effect evaluation errors.
    pub fn effect_contributions(&self, x: &BinnedMatrix) -> Result<Vec<f64>, PbError> {
        self.validate_binned_matrix(x)?;
        let maps =
            crate::scoring::build_cell_maps(&self.bank.merged_grids, &self.schema.cat_encoders, x)?;
        let n_tables = self.bank.tables.len();
        let n_effects = n_tables + self.bank.factored.len();
        let n_cells = self.bank.merged_grids.len();
        let n_rows = x.n_rows as usize;
        let len = n_rows
            .checked_mul(n_effects)
            .ok_or_else(|| PbError::ShapeMismatch {
                what: format!("{n_rows} rows x {n_effects} effects overflows the output"),
            })?;
        let mut out = vec![0.0_f64; len];
        if n_effects == 0 {
            return Ok(out);
        }
        out.par_chunks_mut(n_effects).enumerate().try_for_each(
            |(row, dst)| -> Result<(), PbError> {
                let mut cells = vec![0u32; n_cells];
                crate::scoring::fill_row_cells(x, &maps, row, &mut cells)?;
                for (slot, table) in dst.iter_mut().zip(&self.bank.tables) {
                    *slot = table.eval(&cells)?;
                }
                for (slot, effect) in dst.iter_mut().skip(n_tables).zip(&self.bank.factored) {
                    *slot = effect.eval(&cells)?;
                }
                Ok(())
            },
        )?;
        Ok(out)
    }

    /// Raw (link-space) scores for an already-binned design, via the LUT-sum (`bank.score`,
    /// which includes any factored order-3 tables). `offset` is added before the inverse
    /// link, exactly as in [`Model`].
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] on width/length mismatch; propagated cell-map/score errors.
    pub fn score_raw(&self, x: &BinnedMatrix, offset: Option<&[f32]>) -> Result<Vec<f32>, PbError> {
        self.score_raw_with(x, offset, None)
    }

    /// The model-bin -> merged-cell maps [`TableModel::score_raw_with`] needs, built once so a
    /// serving caller can reuse them across calls rather than rebuilding them per call (the
    /// dominant fixed cost of a small predict). Depends only on this model.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`]/[`PbError::InvalidInput`] if the bank's merged grids cannot be
    /// mapped from the model grids (a malformed model).
    pub fn cell_maps(&self) -> Result<crate::scoring::CellMaps, PbError> {
        crate::scoring::CellMaps::build(
            &self.bank,
            &self.schema.cat_encoders,
            &self.grids,
            &self.provenance,
        )
    }

    /// [`TableModel::score_raw`] through maps from [`TableModel::cell_maps`] (`None` builds them
    /// for this call). Bit-identical to [`TableModel::score_raw`] either way.
    ///
    /// # Errors
    /// As [`TableModel::score_raw`]; [`PbError::InvalidInput`] if `maps` was built for a model
    /// with different merged grids.
    pub fn score_raw_with(
        &self,
        x: &BinnedMatrix,
        offset: Option<&[f32]>,
        maps: Option<&crate::scoring::CellMaps>,
    ) -> Result<Vec<f32>, PbError> {
        self.validate_binned_matrix(x)?;
        let n_rows = x.n_rows as usize;
        if let Some(off) = offset {
            if off.len() != n_rows {
                return Err(PbError::ShapeMismatch {
                    what: format!("offset len {} != n_rows {n_rows}", off.len()),
                });
            }
        }
        let mut raw64 = vec![0.0_f64; n_rows];
        match maps {
            Some(m) => {
                if m.merged_grids != self.bank.merged_grids
                    || m.grids != self.grids
                    || m.provenance != self.provenance
                    || m.cat_encoders != self.schema.cat_encoders
                {
                    return Err(PbError::InvalidInput {
                        what: "cell maps were built for different grids, provenance, or categorical encoders".into(),
                    });
                }
                crate::scoring::score_bank_binned_with_maps(&self.bank, &m.maps, x, &mut raw64)?;
            }
            None => crate::scoring::score_bank_binned(
                &self.bank,
                &self.schema.cat_encoders,
                x,
                &mut raw64,
            )?,
        }
        let mut out = Vec::with_capacity(n_rows);
        for (r, v) in raw64.iter().enumerate() {
            let off = offset.and_then(|o| o.get(r).copied()).unwrap_or(0.0);
            out.push(*v as f32 + off);
        }
        Ok(out)
    }

    /// Response-space predictions for an already-binned design.
    ///
    /// # Errors
    /// Propagates [`TableModel::score_raw`] failures.
    pub fn predict_binned(
        &self,
        x: &BinnedMatrix,
        offset: Option<&[f32]>,
    ) -> Result<Vec<f32>, PbError> {
        self.predict_binned_with(x, offset, None)
    }

    /// [`TableModel::predict_binned`] through maps from [`TableModel::cell_maps`].
    ///
    /// # Errors
    /// As [`TableModel::score_raw_with`].
    pub fn predict_binned_with(
        &self,
        x: &BinnedMatrix,
        offset: Option<&[f32]>,
        maps: Option<&crate::scoring::CellMaps>,
    ) -> Result<Vec<f32>, PbError> {
        let mut raw = self.score_raw_with(x, offset, maps)?;
        for v in &mut raw {
            *v = inverse_link(self.link, *v);
        }
        Ok(raw)
    }
}

/// Validate one [`BorderGrid`]'s well-formedness: `missing_bin == 0`, `n_bins` in range and
/// consistent with `borders.len()`, every border finite, and borders strictly ascending.
/// Mirrors `Model::validate`'s grid loop (engine/mod.rs) field-for-field.
fn validate_effect_identity(
    what: &str,
    u: &crate::explain::FeatureSet,
    axes: &[AxisId],
    n_raw: usize,
) -> Result<(), PbError> {
    let sorted = u.0.windows(2).all(|pair| matches!(pair, [a, b] if a < b));
    if !sorted
        || u.0
            .iter()
            .zip(axes)
            .any(|(raw, axis)| *raw != axis.raw || raw.0 as usize >= n_raw)
    {
        return Err(PbError::ShapeMismatch {
            what: format!("{what} feature set does not match its ordered raw axes"),
        });
    }
    Ok(())
}

fn validate_border_grid(what: &str, axis: usize, grid: &BorderGrid) -> Result<(), PbError> {
    if grid.missing_bin != 0 {
        return Err(PbError::InvalidInput {
            what: format!(
                "{what} grid {axis} missing_bin must be 0, got {}",
                grid.missing_bin
            ),
        });
    }
    if grid.n_bins == 0 || grid.n_bins > 255 {
        return Err(PbError::InvalidInput {
            what: format!(
                "{what} grid {axis} n_bins must be in 1..=255, got {}",
                grid.n_bins
            ),
        });
    }
    let expected_bins =
        u16::try_from(
            grid.borders
                .len()
                .checked_add(2)
                .ok_or_else(|| PbError::Internal {
                    what: "grid border count overflow".into(),
                })?,
        )
        .map_err(|_| PbError::InvalidInput {
            what: format!("{what} grid {axis} has too many borders"),
        })?;
    if grid.n_bins != expected_bins && !(grid.n_bins == 1 && grid.borders.is_empty()) {
        return Err(PbError::InvalidInput {
            what: format!(
                "{what} grid {axis} n_bins {} inconsistent with {} borders",
                grid.n_bins,
                grid.borders.len()
            ),
        });
    }
    for (i, &border) in grid.borders.iter().enumerate() {
        if !border.is_finite() {
            return Err(PbError::InvalidInput {
                what: format!("{what} grid {axis} border {i} must be finite"),
            });
        }
    }
    for pair in grid.borders.windows(2) {
        if let [a, b] = pair {
            if a >= b {
                return Err(PbError::InvalidInput {
                    what: format!("{what} grid {axis} borders must be strictly ascending"),
                });
            }
        }
    }
    Ok(())
}

/// Validate one `bank.merged_grids` entry (per RAW FEATURE, unlike [`validate_border_grid`]'s
/// per-AXIS grids). `is_joint` (cross-checked against `provenance`, never inferred from the
/// numbers alone) selects which shape is expected: an ORDINARY raw feature's entry is a real
/// numeric grid and gets the exact same check `validate_border_grid` runs (including its
/// `<= 255` bound, which is load-bearing there: a REAL axis's per-row bin is stored as `u8` in
/// `BinnedMatrix::data`); a P1 multi-channel raw feature's entry is the placeholder
/// `MergedGrids::border_grids` produces for it (empty borders, `n_bins` = its joint cell
/// count, §4.3) — no numeric-border checks apply, and no `u8`-storage bound applies either: a
/// joint cell id is never stored per-row as a bin (it is looked up on demand via
/// `JointCatAxis::cell_for_channel_bins`, then carried in the raw-indexed `x_cells: Vec<u32>`
/// — see `explain.rs`/`scoring.rs` — never a per-axis `u8` array), so the ONLY real constraint
/// is `BorderGrid::n_bins`'s own field width (`u16`, i.e. `<= 65535`, `u16::try_from` already
/// enforces this cleanly at construction in `MergedGrids::border_grids`). A high-cardinality
/// categorical's joint cell count is bounded by its distinct (post-rare-pooling) level count
/// (§4.3's compactness note), which can exceed 255 even though EACH channel's own bins stay
/// within the single-axis 254-bin budget (`assign_fisher_bins`) — that was the actual
/// overflow bug this comment/check update fixes (bug #4): the previous `<= 255` bound here was
/// copied from the single-axis case without the storage justification that makes it necessary
/// there, and rejected a perfectly valid, lossless joint axis.
fn validate_merged_grid(
    what: &str,
    raw: usize,
    grid: &BorderGrid,
    is_joint: bool,
) -> Result<(), PbError> {
    if !is_joint {
        return validate_border_grid(what, raw, grid);
    }
    if grid.missing_bin != 0 {
        return Err(PbError::InvalidInput {
            what: format!(
                "{what} joint grid {raw} missing_bin must be 0, got {}",
                grid.missing_bin
            ),
        });
    }
    if grid.n_bins < 2 {
        return Err(PbError::InvalidInput {
            what: format!(
                "{what} joint grid {raw} n_bins must be >= 2, got {}",
                grid.n_bins
            ),
        });
    }
    if !grid.borders.is_empty() {
        return Err(PbError::InvalidInput {
            what: format!(
                "{what} joint grid {raw} must have empty borders (P1 multi-channel \
                 placeholder), got {} border(s)",
                grid.borders.len()
            ),
        });
    }
    Ok(())
}

/// Validate one table [`AxisId`]'s borders (finite, strictly ascending) and that `cells`
/// agrees with `borders.len() + 2` (missing cell + finite cells, per its own doc contract) —
/// UNLESS `joint_channels` is `Some` (P1 multi-channel), in which case `borders` must be empty
/// and `cells` is simply required to be a valid (non-zero) joint cell count instead (see
/// [`crate::explain::AxisId`]'s doc).
fn validate_axis(what: &str, axis_idx: usize, axis: &AxisId) -> Result<(), PbError> {
    if let Some(map) = &axis.band_of {
        validate_band_map(what, axis_idx, axis, map)?;
    }
    if axis.joint_channels.is_some() {
        if !axis.borders.is_empty() {
            return Err(PbError::InvalidInput {
                what: format!(
                    "{what} axis {axis_idx} is P1 multi-channel but has {} non-empty border(s)",
                    axis.borders.len()
                ),
            });
        }
        if axis.cells == 0 {
            return Err(PbError::InvalidInput {
                what: format!("{what} axis {axis_idx} joint cell count must be > 0"),
            });
        }
        return Ok(());
    }
    for (i, &border) in axis.borders.iter().enumerate() {
        if !border.is_finite() {
            return Err(PbError::InvalidInput {
                what: format!("{what} axis {axis_idx} border {i} must be finite"),
            });
        }
    }
    for pair in axis.borders.windows(2) {
        if let [a, b] = pair {
            if a >= b {
                return Err(PbError::InvalidInput {
                    what: format!("{what} axis {axis_idx} borders must be strictly ascending"),
                });
            }
        }
    }
    let expected = u32::try_from(axis.borders.len())
        .ok()
        .and_then(|n| n.checked_add(2))
        .ok_or_else(|| PbError::Internal {
            what: "axis border count overflow".into(),
        })?;
    // A banded axis keeps the merged grid's borders (the map's domain) and counts bands.
    let merged = axis
        .band_of
        .as_ref()
        .map_or(axis.cells, |m| u32::try_from(m.len()).unwrap_or(u32::MAX));
    if merged != expected {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "{what} axis {axis_idx} cells {} inconsistent with {} borders",
                merged,
                axis.borders.len()
            ),
        });
    }
    Ok(())
}

/// A band map must cover every band `0..cells` and nothing else, and (on an ordinal axis) keep
/// the missing cell 0 as a band of its own.
fn validate_band_map(
    what: &str,
    axis_idx: usize,
    axis: &AxisId,
    map: &[u32],
) -> Result<(), PbError> {
    if map.is_empty() || axis.cells == 0 {
        return Err(PbError::InvalidInput {
            what: format!("{what} axis {axis_idx} band map must be non-empty"),
        });
    }
    let mut used = vec![false; axis.cells as usize];
    for &b in map {
        let slot = used
            .get_mut(b as usize)
            .ok_or_else(|| PbError::InvalidInput {
                what: format!(
                    "{what} axis {axis_idx} band {b} >= band count {}",
                    axis.cells
                ),
            })?;
        *slot = true;
    }
    if used.iter().any(|u| !u) {
        return Err(PbError::InvalidInput {
            what: format!("{what} axis {axis_idx} band map leaves a band unused"),
        });
    }
    if axis.joint_channels.is_none()
        && map
            .first()
            .is_some_and(|&m0| map.iter().skip(1).any(|&b| b == m0))
    {
        return Err(PbError::InvalidInput {
            what: format!("{what} axis {axis_idx} missing cell must be its own band"),
        });
    }
    Ok(())
}

/// Validate one tensor's cell values (all finite) and, for sparse backing, that per-cell
/// lookup (`Tensor::at`, binary search) agrees with the full materialization (`Tensor::values`,
/// an order-independent linear write) at every cell. The two diverge exactly when sparse
/// entries are out of range, duplicated, or unsorted -- which silently breaks `at`'s binary
/// search (returning a wrong value or the sparse-zero default) while `values` stays correct
/// -- so any disagreement proves corruption on the same lookup `EffectTable::eval` uses to
/// score. Dense tensors skip the per-cell pass: `values()` already reflects their backing
/// directly, so the length check below is the whole story.
fn validate_tensor(what: &str, t: &Tensor) -> Result<(), PbError> {
    let vals = t.values();
    if vals.len() != t.len() {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "{what} data length {} != shape product {}",
                vals.len(),
                t.len()
            ),
        });
    }
    for (i, &v) in vals.iter().enumerate() {
        if !v.is_finite() {
            return Err(PbError::InvalidInput {
                what: format!("{what} cell {i} must be finite, got {v}"),
            });
        }
    }
    if t.is_sparse() {
        let shape = t.shape();
        for (flat, &want) in vals.iter().enumerate() {
            let mut coord = vec![0usize; shape.len()];
            let mut rem = flat;
            for (c, &dim) in coord.iter_mut().zip(shape.iter()).rev() {
                *c = rem % dim;
                rem /= dim;
            }
            let got = t.at(&coord).ok_or_else(|| PbError::Internal {
                what: format!("{what} cell {flat} coordinate out of range"),
            })?;
            if got != want {
                return Err(PbError::InvalidInput {
                    what: format!(
                        "{what} cell {flat} sparse lookup ({got}) disagrees with materialized \
                         value ({want}); corrupt or unsorted sparse entries"
                    ),
                });
            }
        }
    }
    Ok(())
}

/// A tables-only K-class model: `K` per-class [`TableModel`] logits combined by softmax — the pruned
/// analogue of [`crate::engine::MultiClassModel`]. Each `classes[k]` is that class's pruned bank; the
/// probabilities are `softmax(F_0(x), .., F_{K-1}(x))`, applied at this response layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiClassTableModel {
    /// Per-class tables-only logit models (all sharing identical grids/provenance).
    pub classes: Vec<TableModel>,
    /// Class labels, parallel to `classes`.
    pub class_labels: Vec<String>,
    /// Monotone wire-version covering the container.
    pub schema_version: u32,
}

impl MultiClassTableModel {
    /// Number of classes `K`.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    /// Validate structure: `K >= 2`, labels parallel to classes, every sub-model valid and sharing
    /// identical grids/provenance, and the container `schema_version` current.
    ///
    /// # Errors
    /// [`PbError::Serialization`] on a schema-version mismatch; [`PbError::ShapeMismatch`] for
    /// inconsistent labels/grids/provenance; [`PbError::InvalidInput`] for `K < 2`; plus per-model
    /// validation failures.
    pub fn validate(&self) -> Result<(), PbError> {
        // Per-class banks each carry their own required version (the ORDER lift's factored
        // box encoding); the container takes the MAX, never the first class's.
        let required = self
            .classes
            .iter()
            .map(|c| crate::serialize::required_tables_version(&c.bank))
            .max()
            .unwrap_or(crate::serialize::SCHEMA_VERSION_UNLIFTED);
        if self.schema_version > crate::serialize::SCHEMA_VERSION || self.schema_version < required
        {
            return Err(PbError::Serialization(format!(
                "multiclass tables schema_version {} outside {}..={}",
                self.schema_version,
                required,
                crate::serialize::SCHEMA_VERSION
            )));
        }
        let k = self.classes.len();
        if k < 2 {
            return Err(PbError::InvalidInput {
                what: format!("multiclass tables model must have >= 2 classes, got {k}"),
            });
        }
        if self.class_labels.len() != k {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "multiclass tables class_labels len {} != class count {k}",
                    self.class_labels.len()
                ),
            });
        }
        let first = self.classes.first().ok_or_else(|| PbError::Internal {
            what: "multiclass tables has no classes after count check".into(),
        })?;
        first.validate()?;
        for (i, m) in self.classes.iter().enumerate().skip(1) {
            m.validate()?;
            if m.grids != first.grids {
                return Err(PbError::ShapeMismatch {
                    what: format!("multiclass tables class {i} grids differ from class 0"),
                });
            }
            if m.provenance != first.provenance {
                return Err(PbError::ShapeMismatch {
                    what: format!("multiclass tables class {i} provenance differ from class 0"),
                });
            }
        }
        Ok(())
    }

    /// Raw per-class logits for a binned design, row-major `n_rows × K` (row `r`, class `k` at flat
    /// index `r*K + k`).
    ///
    /// # Errors
    /// Propagates any per-class [`TableModel::score_raw`] failure; [`PbError::Internal`] on overflow.
    pub fn predict_raw(&self, x: &BinnedMatrix) -> Result<Vec<f32>, PbError> {
        self.predict_raw_with(x, None)
    }

    /// Per-class [`TableModel::cell_maps`], parallel to `classes`, for reuse across calls.
    ///
    /// # Errors
    /// As [`TableModel::cell_maps`].
    pub fn cell_maps(&self) -> Result<Vec<crate::scoring::CellMaps>, PbError> {
        self.classes.iter().map(TableModel::cell_maps).collect()
    }

    /// [`MultiClassTableModel::predict_raw`] through per-class maps from
    /// [`MultiClassTableModel::cell_maps`] (`None` builds them for this call).
    ///
    /// # Errors
    /// As [`MultiClassTableModel::predict_raw`]; [`PbError::ShapeMismatch`] if `maps` is not one
    /// entry per class.
    pub fn predict_raw_with(
        &self,
        x: &BinnedMatrix,
        maps: Option<&[crate::scoring::CellMaps]>,
    ) -> Result<Vec<f32>, PbError> {
        let n = x.n_rows as usize;
        let k = self.classes.len();
        if let Some(m) = maps {
            if m.len() != k {
                return Err(PbError::ShapeMismatch {
                    what: format!("{} cell map set(s) for {k} class(es)", m.len()),
                });
            }
        }
        let cells = n.checked_mul(k).ok_or_else(|| PbError::Internal {
            what: "multiclass tables raw size overflows usize".into(),
        })?;
        let mut out = vec![0.0_f32; cells];
        for (ci, tm) in self.classes.iter().enumerate() {
            let col = tm.score_raw_with(x, None, maps.and_then(|m| m.get(ci)))?;
            for (r, &v) in col.iter().enumerate() {
                let idx = r
                    .checked_mul(k)
                    .and_then(|b| b.checked_add(ci))
                    .ok_or_else(|| PbError::Internal {
                        what: "multiclass tables raw flat index overflow".into(),
                    })?;
                *out.get_mut(idx).ok_or_else(|| PbError::Internal {
                    what: "multiclass tables raw flat index escaped".into(),
                })? = v;
            }
        }
        Ok(out)
    }

    /// Class probabilities for a binned design, row-major `n_rows × K`; each row is a stable softmax.
    ///
    /// # Errors
    /// Propagates [`MultiClassTableModel::predict_raw`]; [`PbError::Internal`] on a row escape.
    pub fn predict_proba(&self, x: &BinnedMatrix) -> Result<Vec<f32>, PbError> {
        self.predict_proba_with(x, None)
    }

    /// [`MultiClassTableModel::predict_proba`] through per-class maps from
    /// [`MultiClassTableModel::cell_maps`].
    ///
    /// # Errors
    /// As [`MultiClassTableModel::predict_raw_with`].
    pub fn predict_proba_with(
        &self,
        x: &BinnedMatrix,
        maps: Option<&[crate::scoring::CellMaps]>,
    ) -> Result<Vec<f32>, PbError> {
        let n = x.n_rows as usize;
        let k = self.classes.len();
        let mut raw = self.predict_raw_with(x, maps)?;
        for r in 0..n {
            let base = r.checked_mul(k).ok_or_else(|| PbError::Internal {
                what: "multiclass tables proba base overflow".into(),
            })?;
            let end = base.checked_add(k).ok_or_else(|| PbError::Internal {
                what: "multiclass tables proba end overflow".into(),
            })?;
            let row = raw.get_mut(base..end).ok_or_else(|| PbError::Internal {
                what: "multiclass tables proba row escaped".into(),
            })?;
            crate::engine::softmax_in_place(row);
        }
        Ok(raw)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]
    use super::*;
    use crate::explain::{
        fixture_model, fixture_multichannel_model, fixture_multichannel_serve, fixture_serve,
    };

    /// P1 multi-channel: `TableModel::from_model` on a hand-built model with a genuine
    /// cross-channel interaction (see `fixture_multichannel_model`'s doc) builds a bank whose
    /// `validate()` accepts the joint (empty-borders, `n_bins` = joint cell count) merged-grid
    /// and `AxisId` shapes rather than rejecting them as malformed ordinary axes.
    #[test]
    fn tables_model_validates_a_multichannel_bank() {
        let model = fixture_multichannel_model();
        let x = fixture_multichannel_serve();
        let tm = TableModel::from_model(&model, &x, RefMeasure::Uniform).unwrap();
        tm.validate().unwrap();

        // The raw feature's table carries the joint marker, and merged_grids is raw-indexed
        // (length 1: one raw feature, even though the model has 2 axes).
        assert_eq!(tm.bank.merged_grids.len(), 1);
        let raw0_table = tm
            .bank
            .tables
            .iter()
            .find(|t| t.u.0.iter().any(|f| f.0 == 0))
            .unwrap();
        assert_eq!(raw0_table.axes.len(), 1);
        let raw0_axis = raw0_table.axes.first().unwrap();
        assert!(raw0_axis.joint_channels.is_some());
        assert!(raw0_axis.borders.is_empty());

        // The core hard-gate claim: an un-pruned TableModel's predictions, computed through
        // scoring.rs's joint-aware build_cell_maps/fill_row_cells/score_bank_binned path,
        // EXACTLY match the tree ensemble's own predictions — including for the genuine
        // cross-channel interaction (g(1,1)=6, not the additive 4 a naive per-channel sum
        // would give).
        let ens = model.predict(&x.0, None).unwrap();
        let tab = tm.predict_binned(&x.0, None).unwrap();
        assert_eq!(ens.len(), 4);
        assert_eq!(tab.len(), 4);
        for (e, t) in ens.iter().zip(&tab) {
            assert!((e - t).abs() < 1e-6, "ensemble {e} vs tables {t}");
        }
    }

    /// `f0` plus a row's effect contributions, summed in reported order, is bit-for-bit the
    /// float64 score `score_raw` rounds to float32 — for ordinary and multi-channel banks.
    #[test]
    fn effect_contributions_reproduce_the_served_score_exactly() {
        for (model, x) in [
            (fixture_model(), fixture_serve()),
            (fixture_multichannel_model(), fixture_multichannel_serve()),
        ] {
            let tm = TableModel::from_model(&model, &x, RefMeasure::Uniform).unwrap();
            let sets = tm.effect_feature_sets();
            assert_eq!(sets.len(), tm.bank.tables.len() + tm.bank.factored.len());
            let values = tm.effect_contributions(&x.0).unwrap();
            let raw = tm.score_raw(&x.0, None).unwrap();
            assert_eq!(values.len(), raw.len() * sets.len());
            for (row, served) in values.chunks(sets.len()).zip(&raw) {
                let sum = row.iter().fold(tm.bank.f0, |acc, v| acc + v);
                assert_eq!(sum as f32, *served);
            }
        }
    }

    /// Re-centring on rows never changes the function: every row scores the same. On the very
    /// rows the bank was purified on (flat count, same measure) it reproduces the stored ledger;
    /// a skewed per-row mass moves the supports.
    #[test]
    fn recentring_keeps_the_function_and_reproduces_the_stored_ledger() {
        for (model, x) in [
            (fixture_model(), fixture_serve()),
            (fixture_multichannel_model(), fixture_multichannel_serve()),
        ] {
            let tm = TableModel::from_model(&model, &x, RefMeasure::Uniform).unwrap();
            let cells = tm.row_cells(&x.0).unwrap();
            let same = tm.recentred_bank(&x.0, None, RefMeasure::Uniform).unwrap();
            for (a, b) in tm.bank.tables.iter().zip(&same.tables) {
                assert_eq!(a.u, b.u);
                for (va, vb) in a.values.values().iter().zip(b.values.values().iter()) {
                    assert!((va - vb).abs() < 1e-12, "{va} vs {vb}");
                }
                assert_eq!(a.support, b.support);
            }
            let n = x.0.n_rows as usize;
            let skew: Vec<f32> = (0..n).map(|r| if r % 2 == 0 { 9.0 } else { 1.0 }).collect();
            let moved = tm
                .recentred_bank(
                    &x.0,
                    Some(&skew),
                    RefMeasure::ExposureMarginals { floor: 1e-3 },
                )
                .unwrap();
            assert_ne!(moved.tables, tm.bank.tables);
            for row in &cells {
                let (a, b) = (tm.bank.score(row).unwrap(), moved.score(row).unwrap());
                assert!(
                    (a - b).abs() < 1e-9,
                    "re-centring moved a score: {a} vs {b}"
                );
            }
            assert!(tm
                .recentred_bank(&x.0, skew.get(1..), RefMeasure::Uniform)
                .is_err());
        }
    }

    #[test]
    fn tables_model_predicts_identically_to_ensemble() {
        let model = fixture_model();
        let x = fixture_serve();
        let tm = TableModel::from_model(&model, &x, RefMeasure::Uniform).unwrap();

        let ens = model.predict(&x.0, None).unwrap();
        let tab = tm.predict_binned(&x.0, None).unwrap();
        assert_eq!(ens.len(), tab.len());
        for (e, t) in ens.iter().zip(&tab) {
            assert!((e - t).abs() < 1e-5, "ensemble {e} vs tables {t}");
        }
    }

    #[test]
    fn tables_model_bincode_round_trips() {
        let model = fixture_model();
        let x = fixture_serve();
        let tm = TableModel::from_model(&model, &x, RefMeasure::Uniform).unwrap();

        let bytes = tm.to_bincode().unwrap();
        let back = TableModel::from_bincode(&bytes).unwrap();
        assert_eq!(tm, back);

        let json = tm.to_json().unwrap();
        let back_json = TableModel::from_json(&json).unwrap();
        assert_eq!(tm, back_json);
    }

    /// A NaN table cell used to deserialize and `validate()` cleanly (only `bank.f0` and four
    /// length equalities were checked) and then serve silently wrong predictions. Every field
    /// touched here is public, so the corruption is injected directly rather than through
    /// fragile hand-built bytes/JSON.
    #[test]
    fn validate_catches_non_finite_table_cell() {
        let model = fixture_model();
        let x = fixture_serve();
        let mut tm = TableModel::from_model(&model, &x, RefMeasure::Uniform).unwrap();
        tm.validate().unwrap(); // sanity: the freshly-built model is valid

        let table = tm.bank.tables.get_mut(0).unwrap();
        let coord = vec![0usize; table.values.shape().len()];
        table.values.set(&coord, f64::NAN).unwrap();

        assert!(matches!(tm.validate(), Err(PbError::InvalidInput { .. })));
    }

    /// A support tensor is validated the same way as `values` (the shared `validate_tensor`
    /// helper) -- this pins that it isn't skipped.
    #[test]
    fn validate_catches_non_finite_support_cell() {
        let model = fixture_model();
        let x = fixture_serve();
        let mut tm = TableModel::from_model(&model, &x, RefMeasure::Uniform).unwrap();

        let table = tm.bank.tables.get_mut(0).unwrap();
        let coord = vec![0usize; table.support.shape().len()];
        table.support.set(&coord, f64::INFINITY).unwrap();

        assert!(matches!(tm.validate(), Err(PbError::InvalidInput { .. })));
    }

    /// An axis whose declared `cells` disagrees with the values tensor's actual per-axis
    /// extent is exactly the "shape corruption" the fix sketch calls out.
    #[test]
    fn validate_catches_axis_cells_shape_mismatch() {
        let model = fixture_model();
        let x = fixture_serve();
        let mut tm = TableModel::from_model(&model, &x, RefMeasure::Uniform).unwrap();

        let table = tm.bank.tables.get_mut(0).unwrap();
        let axis = table.axes.get_mut(0).unwrap();
        axis.cells += 1;

        assert!(matches!(tm.validate(), Err(PbError::ShapeMismatch { .. })));
    }

    /// The serve grids get the same well-formedness gate `Model::validate` applies to its own
    /// grids (previously `TableModel::validate` never looked past their length).
    #[test]
    fn validate_catches_malformed_serve_grid() {
        let model = fixture_model();
        let x = fixture_serve();
        let mut tm = TableModel::from_model(&model, &x, RefMeasure::Uniform).unwrap();

        let grid = tm.grids.get_mut(0).unwrap();
        grid.missing_bin = 1;

        assert!(matches!(tm.validate(), Err(PbError::InvalidInput { .. })));
    }

    /// The bank's own merged grids are a separate `Vec<BorderGrid>` from `self.grids` and need
    /// their own check.
    #[test]
    fn validate_catches_non_finite_merged_grid_border() {
        let model = fixture_model();
        let x = fixture_serve();
        let mut tm = TableModel::from_model(&model, &x, RefMeasure::Uniform).unwrap();

        let grid = tm
            .bank
            .merged_grids
            .iter_mut()
            .find(|g| !g.borders.is_empty())
            .unwrap();
        let border = grid.borders.get_mut(0).unwrap();
        *border = f32::NAN;

        assert!(matches!(tm.validate(), Err(PbError::InvalidInput { .. })));
    }

    /// The checks above pin `validate()`'s own logic against direct struct mutation; these
    /// pin that `TableModel::from_json`/`from_bincode` actually WIRE that logic into the
    /// public load path -- corrupting real serialized text/bytes (not the in-memory struct)
    /// and asserting the load call itself now fails, the way a corrupted artifact from disk
    /// would be encountered. JSON has no representation for NaN/Infinity (`serde_json`
    /// refuses to encode them), so the non-finite-cell case is covered via bincode instead
    /// (below); shape corruption is representable in both and is covered here in JSON.
    #[test]
    fn corrupted_json_with_truncated_values_fails_to_load() {
        let model = fixture_model();
        let x = fixture_serve();
        let tm = TableModel::from_model(&model, &x, RefMeasure::Uniform).unwrap();
        let json = tm.to_json().unwrap();
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();

        let dense = value
            .pointer_mut("/model/bank/tables/0/values/data/Dense")
            .and_then(serde_json::Value::as_array_mut)
            .unwrap();
        assert!(
            dense.len() > 1,
            "fixture table too small for a truncation test"
        );
        dense.pop();

        let corrupted = serde_json::to_string(&value).unwrap();
        assert!(matches!(
            TableModel::from_json(&corrupted),
            Err(PbError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn corrupted_json_with_mismatched_axis_shape_fails_to_load() {
        let model = fixture_model();
        let x = fixture_serve();
        let tm = TableModel::from_model(&model, &x, RefMeasure::Uniform).unwrap();
        let json = tm.to_json().unwrap();
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();

        let cells = value
            .pointer_mut("/model/bank/tables/0/axes/0/cells")
            .unwrap();
        let bumped = cells.as_u64().unwrap() + 1;
        *cells = serde_json::json!(bumped);

        let corrupted = serde_json::to_string(&value).unwrap();
        assert!(matches!(
            TableModel::from_json(&corrupted),
            Err(PbError::ShapeMismatch { .. })
        ));
    }

    /// Serving through maps built once (`cell_maps`) is bit-identical to rebuilding them per call,
    /// for the single-output and the multiclass model, and maps from another model are refused.
    #[test]
    fn cached_cell_maps_score_bit_identically() {
        let model = fixture_model();
        let x = fixture_serve();
        let tm = TableModel::from_model(&model, &x, RefMeasure::Uniform).unwrap();
        let maps = tm.cell_maps().unwrap();
        let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        assert_eq!(
            bits(tm.score_raw(&x.0, None).unwrap()),
            bits(tm.score_raw_with(&x.0, None, Some(&maps)).unwrap())
        );
        assert_eq!(
            bits(tm.predict_binned(&x.0, None).unwrap()),
            bits(tm.predict_binned_with(&x.0, None, Some(&maps)).unwrap())
        );

        let mc = MultiClassTableModel {
            classes: vec![tm.clone(), tm.clone()],
            class_labels: vec!["a".into(), "b".into()],
            schema_version: tm.schema_version,
        };
        let mc_maps = mc.cell_maps().unwrap();
        assert_eq!(
            bits(mc.predict_proba(&x.0).unwrap()),
            bits(mc.predict_proba_with(&x.0, Some(&mc_maps)).unwrap())
        );
        assert!(mc.predict_raw_with(&x.0, mc_maps.get(..1)).is_err());

        // Maps recorded against different merged grids (another model's) are refused.
        let mut stale = maps.clone();
        stale.merged_grids.first_mut().unwrap().n_bins += 1;
        assert!(tm.score_raw_with(&x.0, None, Some(&stale)).is_err());
    }

    #[test]
    fn multiclass_tables_model_round_trips_and_softmaxes() {
        let model = fixture_model();
        let x = fixture_serve();
        let tm = TableModel::from_model(&model, &x, RefMeasure::Uniform).unwrap();
        let mc = MultiClassTableModel {
            classes: vec![tm.clone(), tm.clone()],
            class_labels: vec!["a".into(), "b".into()],
            schema_version: tm.schema_version,
        };
        mc.validate().unwrap();
        // Two identical logits ⇒ every row's softmax is ~[0.5, 0.5].
        let proba = mc.predict_proba(&x.0).unwrap();
        assert_eq!(proba.len(), (x.0.n_rows as usize) * 2);
        for p in proba.chunks(2) {
            assert!(p.iter().all(|&v| (v - 0.5).abs() < 1e-5));
        }
        let bytes = mc.to_bincode().unwrap();
        assert!(crate::serialize::is_multiclass_tables_bytes(&bytes));
        assert_eq!(mc, MultiClassTableModel::from_bincode(&bytes).unwrap());
        assert_eq!(
            mc,
            MultiClassTableModel::from_json(&mc.to_json().unwrap()).unwrap()
        );
    }

    #[test]
    fn tables_model_magic_distinguishes_from_plain_model() {
        use crate::serialize::{is_multiclass_bytes, is_tables_bytes};
        let model = fixture_model();
        let x = fixture_serve();
        let tm = TableModel::from_model(&model, &x, RefMeasure::Uniform).unwrap();

        let tm_bytes = tm.to_bincode().unwrap();
        assert!(
            is_tables_bytes(&tm_bytes),
            "tables model must carry TBTM magic"
        );
        assert!(!is_multiclass_bytes(&tm_bytes));
        // A plain tree Model blob is not a tables-only blob.
        assert!(!is_tables_bytes(&model.to_bincode().unwrap()));
    }
}
