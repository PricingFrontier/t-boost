//! The `f32 → u8` map (spec §03.4) and the column binner (§03.4 hot path). The hot
//! map is a branchless, panic-free binary search: `partition_point` does the work,
//! and the single `+ 1` is bounded by the cardinality invariant, so no indexing can
//! go out of bounds and no arithmetic can overflow.

use crate::cat::{fit_cat_encoder, CatEncoderStore, CatFitSpec, TsConfig, TsEncodingId};
use crate::data::grid::build_grid;
use crate::data::{AxisKind, AxisProvenance, BinConfig, BinnedMatrix, BorderGrid, FeatureId};
use crate::data::{ServeBinnedMatrix, TrainBinnedMatrix};
use crate::error::PbError;
use rayon::prelude::*;
use std::collections::HashMap;

/// One raw numeric column plus its stable raw-feature id.
#[derive(Debug, Clone, Copy)]
pub struct NumericColumn<'a> {
    /// Raw feature id carried into [`AxisProvenance`].
    pub raw: FeatureId,
    /// Column values, one per row.
    pub values: &'a [f32],
}

/// One raw categorical column to fit as a target-statistic axis.
#[derive(Debug, Clone, Copy)]
pub struct CategoricalColumn<'a> {
    /// Raw feature id carried into [`AxisProvenance`].
    pub raw: FeatureId,
    /// Encoder id stamped into [`AxisKind::CategoricalTS`].
    pub id: TsEncodingId,
    /// Per-row labels.
    pub levels: &'a [String],
    /// Target-statistic configuration.
    pub config: &'a TsConfig,
}

/// One raw categorical column to re-encode for serving/auditing.
#[derive(Debug, Clone, Copy)]
pub struct ServeCategoricalColumn<'a> {
    /// Raw feature id.
    pub raw: FeatureId,
    /// Encoder id to resolve in the model's [`CatEncoderStore`].
    pub id: TsEncodingId,
    /// Per-row labels.
    pub levels: &'a [String],
}

/// One raw categorical column to re-encode for serving, as per-row codes into its distinct labels:
/// row `r`'s label is `labels[codes[r]]`. Labels may repeat; each is encoded independently.
#[derive(Debug, Clone, Copy)]
pub struct ServeCategoricalCodes<'a> {
    /// Raw feature id.
    pub raw: FeatureId,
    /// Encoder id to resolve in the model's [`CatEncoderStore`].
    pub id: TsEncodingId,
    /// Per-row index into `labels`.
    pub codes: &'a [u32],
    /// The distinct labels the codes index.
    pub labels: &'a [String],
}

/// Output of fitting mixed numeric/categorical columns.
#[derive(Debug, Clone, PartialEq)]
pub struct FittedBinnedData {
    /// Leakage-free training matrix used to grow trees.
    pub train: TrainBinnedMatrix,
    /// Full-data serve/audit matrix for the same rows.
    pub serve: ServeBinnedMatrix,
    /// Frozen categorical encoders to persist in [`crate::ModelSchema`].
    pub cat_encoders: CatEncoderStore,
}

/// Map a finite-or-missing value to its bin id (spec §03.4).
///
/// `NaN` (and Arrow nulls, mapped to `NaN` at ingest) is the ONLY missing case and
/// returns bin `0`. Every finite value — including out-of-range and `±inf` (which are
/// non-NaN) — clamps to the first/last finite data bin via `partition_point`:
/// `bin(v) = 1 + (count of borders strictly below v)`.
///
/// # Errors
/// [`PbError::Internal`] only on the impossible case `bin index > u8` (the
/// cardinality invariant `borders.len() <= 253` rules it out); never panics.
pub fn bin(v: f32, g: &BorderGrid) -> Result<u8, PbError> {
    if v.is_nan() {
        return Ok(g.missing_bin); // == 0
    }
    // k in 0..=borders.len() (<= 253 by the R-BINS cardinality invariant).
    let k = g.borders.partition_point(|&b| b < v);
    // data bin = k + 1, in 1..=n_data_bins <= 254. The `+ 1` cannot overflow usize
    // (k <= 253). u8::try_from never fails here, but maps the impossible case to a
    // typed error rather than panicking under overflow-checks.
    u8::try_from(k + 1).map_err(|_| PbError::Internal {
        what: "bin index exceeded u8 (borders.len() > 253?)".into(),
    })
}

/// Bin a set of column-major raw `f32` feature columns into a [`BinnedMatrix`]
/// (spec §03.4). Each column gets its own frozen [`BorderGrid`] (built once) and is
/// then mapped to `u8` bin ids; the per-feature work is rayon-parallel and
/// order-independent, so the result is byte-identical across thread counts (§1 `[GATE]`).
/// Every axis is `AxisKind::Numeric` with `raw = FeatureId(column index)`.
///
/// # Errors
/// [`PbError::InvalidConfig`] if `cfg` is invalid; [`PbError::ShapeMismatch`] if the
/// columns or `weight` disagree on length; [`PbError::InvalidInput`] if there are
/// more than `u32::MAX` rows; plus any [`build_grid`] error.
pub fn bin_columns(
    columns: &[&[f32]],
    weight: Option<&[f32]>,
    cfg: &BinConfig,
    seed: u64,
) -> Result<BinnedMatrix, PbError> {
    cfg.validate()?;
    let n_rows = columns.first().map_or(0, |c| c.len());
    for c in columns {
        if c.len() != n_rows {
            return Err(PbError::ShapeMismatch {
                what: "feature columns have unequal lengths".into(),
            });
        }
    }
    if let Some(w) = weight {
        if w.len() != n_rows {
            return Err(PbError::ShapeMismatch {
                what: format!("weight len {} != n_rows {n_rows}", w.len()),
            });
        }
    }
    let n_rows_u32 = u32::try_from(n_rows).map_err(|_| PbError::InvalidInput {
        what: "more than u32::MAX rows is out of scope for v1".into(),
    })?;

    // Per-feature: build grid + bin the column. Order-preserving collect ⇒ the
    // assembled matrix is independent of how rayon scheduled the features.
    let per_feature: Result<Vec<(BorderGrid, Vec<u8>)>, PbError> = columns
        .par_iter()
        .enumerate()
        .map(|(f, &col)| {
            let feat = FeatureId(u32::try_from(f).map_err(|_| PbError::Internal {
                what: "feature index exceeded u32".into(),
            })?);
            let grid = build_grid(col, weight, cfg, seed, feat)?;
            let mut binned = Vec::with_capacity(col.len());
            for &v in col {
                binned.push(bin(v, &grid)?);
            }
            Ok((grid, binned))
        })
        .collect();
    let per_feature = per_feature?;

    let mut data = Vec::with_capacity(per_feature.len());
    let mut grids = Vec::with_capacity(per_feature.len());
    let mut provenance = Vec::with_capacity(per_feature.len());
    for (f, (grid, binned)) in per_feature.into_iter().enumerate() {
        let raw = FeatureId(u32::try_from(f).map_err(|_| PbError::Internal {
            what: "feature index exceeded u32".into(),
        })?);
        data.push(binned);
        grids.push(grid);
        provenance.push(AxisProvenance {
            raw,
            kind: AxisKind::Numeric,
        });
    }
    Ok(BinnedMatrix {
        data,
        n_rows: n_rows_u32,
        grids,
        provenance,
    })
}

/// Fit and bin mixed numeric/categorical training columns.
///
/// Numeric columns use the ordinary §03 border builder. Categorical columns fit a
/// frozen full-data [`CatEncoderStore`] and produce two aligned matrices: the
/// training matrix bins leakage-free row encodings, while the serve matrix bins the
/// frozen full-data encodings that will be used after serialization.
///
/// # Errors
/// [`PbError::InvalidConfig`] for invalid bin/target-stat configs;
/// [`PbError::ShapeMismatch`] for column length mismatches; [`PbError::InvalidInput`]
/// for invalid row values; plus any propagated binning/encoder error.
pub fn bin_train_columns(
    numeric: &[NumericColumn<'_>],
    categorical: &[CategoricalColumn<'_>],
    y: &[f32],
    weight: Option<&[f32]>,
    exposure: Option<&[f32]>,
    cfg: &BinConfig,
    seed: u64,
) -> Result<FittedBinnedData, PbError> {
    bin_train_columns_with_holdout(numeric, categorical, y, weight, exposure, cfg, seed, None)
}

/// [`bin_train_columns`] with an optional validation-holdout mask (the "one honest holdout"
/// design, `design/ordered-ts-early-stopping.md`): when `holdout` is `Some(mask)` every
/// categorical encoder (rare-level collapse, EB smoothing, full/serve map, per-row leakage
/// scheme) is fit on the `!mask` train-proper rows ONLY; train-proper rows keep their
/// leakage-free per-row encodings and holdout rows are encoded through the serve map —
/// exactly what unseen test rows get — so an early-stopping signal computed on the holdout
/// is blind to its own targets AND test-representative. Numeric grids are target-free and
/// stay fit on all rows. `None` is byte-identical to [`bin_train_columns`].
///
/// # Errors
/// As [`bin_train_columns`], plus [`PbError::ShapeMismatch`] if `holdout` length differs
/// from `n_rows` and [`PbError::InvalidInput`] if the mask leaves no train-proper rows.
#[allow(clippy::too_many_arguments)]
pub fn bin_train_columns_with_holdout(
    numeric: &[NumericColumn<'_>],
    categorical: &[CategoricalColumn<'_>],
    y: &[f32],
    weight: Option<&[f32]>,
    exposure: Option<&[f32]>,
    cfg: &BinConfig,
    seed: u64,
    holdout: Option<&[bool]>,
) -> Result<FittedBinnedData, PbError> {
    cfg.validate()?;
    if let Some(mask) = holdout {
        if mask.len() != y.len() {
            return Err(PbError::ShapeMismatch {
                what: format!("holdout mask len {} != n_rows {}", mask.len(), y.len()),
            });
        }
        if mask.iter().all(|&h| h) {
            return Err(PbError::InvalidInput {
                what: "holdout mask leaves no train-proper rows".into(),
            });
        }
    }
    validate_optional_row_data(y.len(), weight, exposure)?;
    let n_rows_u32 = u32::try_from(y.len()).map_err(|_| PbError::InvalidInput {
        what: "more than u32::MAX rows is out of scope for v1".into(),
    })?;

    let mut train_data = Vec::with_capacity(numeric.len() + categorical.len());
    let mut serve_data = Vec::with_capacity(numeric.len() + categorical.len());
    let mut grids = Vec::with_capacity(numeric.len() + categorical.len());
    let mut provenance = Vec::with_capacity(numeric.len() + categorical.len());

    // Per-feature fit+bin is independent, so build numeric grids and the (dominant, on
    // high-cardinality data) categorical TS encoders in parallel. Order-preserving collects ⇒
    // byte-identical to the serial build (each encoder is deterministic in its own seed stream).
    let numeric_out: Result<Vec<(BorderGrid, Vec<u8>)>, PbError> = numeric
        .par_iter()
        .map(|col| {
            if col.values.len() != y.len() {
                return Err(PbError::ShapeMismatch {
                    what: format!(
                        "numeric raw {:?} len {} != n_rows {}",
                        col.raw,
                        col.values.len(),
                        y.len()
                    ),
                });
            }
            let grid = build_grid(col.values, weight, cfg, seed, col.raw)?;
            let binned = bin_values(col.values, &grid)?;
            Ok((grid, binned))
        })
        .collect();
    let numeric_out = numeric_out?;

    // Categorical shape + (raw, id) uniqueness are checked up front (a parallel encode cannot
    // do the incremental "already seen" check); same first-duplicate-wins semantics as before.
    for (i, col) in categorical.iter().enumerate() {
        if col.levels.len() != y.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "categorical raw {:?} len {} != n_rows {}",
                    col.raw,
                    col.levels.len(),
                    y.len()
                ),
            });
        }
        if categorical
            .iter()
            .take(i)
            .any(|prev| prev.raw == col.raw && prev.id == col.id)
        {
            return Err(PbError::InvalidConfig {
                what: format!(
                    "duplicate categorical encoder {:?} for raw {:?}",
                    col.id, col.raw
                ),
            });
        }
    }
    type CatOut = (crate::cat::CatEncoder, BorderGrid, Vec<u8>, Vec<u8>);
    let cat_out: Result<Vec<CatOut>, PbError> = categorical
        .par_iter()
        .map(|col| {
            let (encoder, train_encoded) = match holdout {
                None => fit_cat_encoder(
                    col.levels,
                    y,
                    CatFitSpec {
                        raw: col.raw,
                        id: col.id,
                        weight,
                        exposure,
                        config: col.config,
                        seed,
                    },
                )?,
                // One-honest-holdout path: the encoder (collapse, smoothing, serve map, per-row
                // leakage scheme) sees train-proper rows only; holdout rows are encoded through
                // the serve map exactly like unseen test rows.
                Some(mask) => {
                    let mut lv: Vec<String> = Vec::new();
                    let mut ys: Vec<f32> = Vec::new();
                    let mut ws: Vec<f32> = Vec::new();
                    let mut es: Vec<f32> = Vec::new();
                    for ((level, &hold), &yi) in col.levels.iter().zip(mask).zip(y) {
                        if hold {
                            continue;
                        }
                        lv.push(level.clone());
                        ys.push(yi);
                    }
                    if let Some(w) = weight {
                        for (&hold, &wi) in mask.iter().zip(w) {
                            if !hold {
                                ws.push(wi);
                            }
                        }
                    }
                    if let Some(e) = exposure {
                        for (&hold, &ei) in mask.iter().zip(e) {
                            if !hold {
                                es.push(ei);
                            }
                        }
                    }
                    let (enc, tr_sub) = fit_cat_encoder(
                        &lv,
                        &ys,
                        CatFitSpec {
                            raw: col.raw,
                            id: col.id,
                            weight: weight.map(|_| ws.as_slice()),
                            exposure: exposure.map(|_| es.as_slice()),
                            config: col.config,
                            seed,
                        },
                    )?;
                    let serve_map = enc.encoding_map();
                    let mut full = Vec::with_capacity(col.levels.len());
                    let mut it = tr_sub.iter();
                    for (level, &hold) in col.levels.iter().zip(mask) {
                        if hold {
                            full.push(serve_map.get(level.as_str()).copied().unwrap_or(enc.base));
                        } else {
                            full.push(*it.next().ok_or_else(|| PbError::Internal {
                                what: "holdout scatter exhausted train encodings".into(),
                            })?);
                        }
                    }
                    (enc, full)
                }
            };
            let grid = encoder.border_grid()?;
            let train_bins = bin_values(&train_encoded, &grid)?;
            let serve_map = encoder.encoding_map();
            let serve_encoded: Vec<f32> = col
                .levels
                .iter()
                .map(|level| {
                    serve_map
                        .get(level.as_str())
                        .copied()
                        .unwrap_or(encoder.base)
                })
                .collect();
            let serve_bins = bin_values(&serve_encoded, &grid)?;
            Ok((encoder, grid, train_bins, serve_bins))
        })
        .collect();
    let cat_out = cat_out?;

    // Assemble in axis order: numeric features first, then categorical (matches the serial build).
    let mut encoders = Vec::with_capacity(categorical.len());
    for (col, (grid, binned)) in numeric.iter().zip(numeric_out) {
        train_data.push(binned.clone());
        serve_data.push(binned);
        grids.push(grid);
        provenance.push(AxisProvenance {
            raw: col.raw,
            kind: AxisKind::Numeric,
        });
    }
    for (col, (encoder, grid, train_bins, serve_bins)) in categorical.iter().zip(cat_out) {
        train_data.push(train_bins);
        serve_data.push(serve_bins);
        grids.push(grid);
        provenance.push(AxisProvenance {
            raw: col.raw,
            kind: AxisKind::CategoricalTS { encoding: col.id },
        });
        encoders.push(encoder);
    }

    let cat_encoders = CatEncoderStore::from_encoders(encoders);
    Ok(FittedBinnedData {
        train: TrainBinnedMatrix(BinnedMatrix {
            data: train_data,
            n_rows: n_rows_u32,
            grids: grids.clone(),
            provenance: provenance.clone(),
        }),
        serve: ServeBinnedMatrix(BinnedMatrix {
            data: serve_data,
            n_rows: n_rows_u32,
            grids,
            provenance,
        }),
        cat_encoders,
    })
}

/// Re-bin mixed raw serve columns against a fitted model's grids/provenance.
///
/// Numeric values are binned through the persisted grids. Categorical labels are
/// encoded through the frozen [`CatEncoderStore`], with unseen labels mapping to the
/// encoder base value.
///
/// # Errors
/// [`PbError::ShapeMismatch`] if inputs are missing or row counts differ;
/// [`PbError::Internal`] if a categorical provenance references a missing encoder;
/// plus any propagated binning error.
pub fn bin_serve_columns(
    numeric: &[NumericColumn<'_>],
    categorical: &[ServeCategoricalColumn<'_>],
    grids: &[BorderGrid],
    provenance: &[AxisProvenance],
    cat_encoders: &CatEncoderStore,
) -> Result<ServeBinnedMatrix, PbError> {
    bin_serve_columns_with(numeric, categorical, grids, provenance, cat_encoders, None)
}

/// Per-axis categorical label -> encoding maps for [`bin_serve_columns_with`], built once per
/// model so a serving caller does not rebuild a hash map over every level on every call (0.6 ms a
/// call on a model with a 3,830-level categorical). Each map is exactly
/// [`crate::cat::CatEncoder::encoding_map`], owned.
#[derive(Debug, Clone, Default)]
pub struct CatServeMaps {
    by_axis: Vec<Option<HashMap<String, f32>>>,
}

impl CatServeMaps {
    /// Build the maps for every categorical axis of `provenance`.
    ///
    /// # Errors
    /// [`PbError::Internal`] if a categorical provenance references a missing encoder.
    pub fn build(
        provenance: &[AxisProvenance],
        cat_encoders: &CatEncoderStore,
    ) -> Result<Self, PbError> {
        let by_axis = provenance
            .iter()
            .map(|prov| match prov.kind {
                AxisKind::CategoricalTS { encoding } => Ok(Some(
                    cat_encoders
                        .get(encoding, prov.raw)?
                        .encoding_map()
                        .into_iter()
                        .map(|(label, enc)| (label.to_owned(), enc))
                        .collect(),
                )),
                AxisKind::Numeric | AxisKind::Missing => Ok(None),
            })
            .collect::<Result<_, PbError>>()?;
        Ok(Self { by_axis })
    }
}

/// [`bin_serve_columns`] with optional per-model categorical maps from [`CatServeMaps::build`]
/// (`None` builds each map for this call). Byte-identical output either way.
///
/// # Errors
/// As [`bin_serve_columns`]; [`PbError::ShapeMismatch`] if `cat_maps` was built for a different
/// axis count.
pub fn bin_serve_columns_with(
    numeric: &[NumericColumn<'_>],
    categorical: &[ServeCategoricalColumn<'_>],
    grids: &[BorderGrid],
    provenance: &[AxisProvenance],
    cat_encoders: &CatEncoderStore,
    cat_maps: Option<&CatServeMaps>,
) -> Result<ServeBinnedMatrix, PbError> {
    bin_serve_impl(
        numeric,
        ServeCats::Labels(categorical),
        grids,
        provenance,
        cat_encoders,
        cat_maps,
    )
}

/// [`bin_serve_columns_with`] for categorical columns given as per-row codes into their distinct
/// labels ([`ServeCategoricalCodes`]). Byte-identical to passing `labels[codes[r]]` per row, but
/// each distinct label is encoded and binned once instead of once per row, and no per-row string
/// ever crosses from the caller.
///
/// # Errors
/// As [`bin_serve_columns_with`]; [`PbError::InvalidInput`] if a code is outside its labels.
pub fn bin_serve_columns_coded(
    numeric: &[NumericColumn<'_>],
    categorical: &[ServeCategoricalCodes<'_>],
    grids: &[BorderGrid],
    provenance: &[AxisProvenance],
    cat_encoders: &CatEncoderStore,
    cat_maps: Option<&CatServeMaps>,
) -> Result<ServeBinnedMatrix, PbError> {
    bin_serve_impl(
        numeric,
        ServeCats::Codes(categorical),
        grids,
        provenance,
        cat_encoders,
        cat_maps,
    )
}

/// The categorical half of a serve call, as labels or as codes.
#[derive(Clone, Copy)]
enum ServeCats<'a> {
    Labels(&'a [ServeCategoricalColumn<'a>]),
    Codes(&'a [ServeCategoricalCodes<'a>]),
}

fn bin_serve_impl(
    numeric: &[NumericColumn<'_>],
    cats: ServeCats<'_>,
    grids: &[BorderGrid],
    provenance: &[AxisProvenance],
    cat_encoders: &CatEncoderStore,
    cat_maps: Option<&CatServeMaps>,
) -> Result<ServeBinnedMatrix, PbError> {
    if let Some(m) = cat_maps {
        if m.by_axis.len() != provenance.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "categorical serve maps cover {} axes, provenance has {}",
                    m.by_axis.len(),
                    provenance.len()
                ),
            });
        }
    }
    if grids.len() != provenance.len() {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "serve grids len {} != provenance len {}",
                grids.len(),
                provenance.len()
            ),
        });
    }
    let n_rows = match cats {
        ServeCats::Labels(cols) => infer_serve_n_rows(numeric, cols)?,
        ServeCats::Codes(cols) => infer_coded_n_rows(numeric, cols)?,
    };
    // Axes are independent and each yields its own column, so binning them in parallel is
    // byte-identical to the serial loop; below `PAR_MIN_ROWS` the thread wake-up costs more than
    // the binning (a 1M-row, 14-column batch spent 47 ms here serially).
    const PAR_MIN_ROWS: usize = 16_384;
    let bin_axis = |axis: usize,
                    prov: &AxisProvenance,
                    grid: &BorderGrid|
     -> Result<Vec<u8>, PbError> {
        match prov.kind {
            AxisKind::Numeric => {
                let values = find_numeric_column(numeric, prov.raw)?;
                if values.len() != n_rows {
                    return Err(PbError::ShapeMismatch {
                        what: format!(
                            "numeric raw {:?} len {} != n_rows {n_rows}",
                            prov.raw,
                            values.len()
                        ),
                    });
                }
                bin_values(values, grid)
            }
            AxisKind::CategoricalTS { encoding } => {
                let encoder = cat_encoders.get(encoding, prov.raw)?;
                let cached = cat_maps
                    .and_then(|m| m.by_axis.get(axis))
                    .and_then(Option::as_ref);
                let fresh = if cached.is_none() {
                    Some(encoder.encoding_map())
                } else {
                    None
                };
                let encode = |label: &str| -> f32 {
                    match (cached, &fresh) {
                        (Some(map), _) => map.get(label).copied(),
                        (None, Some(map)) => map.get(label).copied(),
                        (None, None) => None,
                    }
                    .unwrap_or(encoder.base)
                };
                let check_len = |len: usize| -> Result<(), PbError> {
                    if len == n_rows {
                        return Ok(());
                    }
                    Err(PbError::ShapeMismatch {
                        what: format!(
                            "categorical raw {:?}/{:?} len {len} != n_rows {n_rows}",
                            prov.raw, encoding
                        ),
                    })
                };
                match cats {
                    ServeCats::Labels(cols) => {
                        let levels = find_categorical_column(cols, prov.raw, encoding)?;
                        check_len(levels.len())?;
                        let encoded: Vec<f32> =
                            levels.iter().map(|level| encode(level.as_str())).collect();
                        bin_values(&encoded, grid)
                    }
                    ServeCats::Codes(cols) => {
                        let col = find_categorical_codes(cols, prov.raw, encoding)?;
                        check_len(col.codes.len())?;
                        // Encoding and binning are pure functions of the label, so binning each
                        // distinct label once and gathering by code is identical to encoding and
                        // binning every row — without a hash lookup per row.
                        let label_bins = col
                            .labels
                            .iter()
                            .map(|label| bin(encode(label.as_str()), grid))
                            .collect::<Result<Vec<u8>, _>>()?;
                        col.codes
                            .iter()
                            .map(|&code| {
                                label_bins.get(code as usize).copied().ok_or_else(|| {
                                    PbError::InvalidInput {
                                        what: format!(
                                        "categorical raw {:?} code {code} outside its {} labels",
                                        prov.raw,
                                        label_bins.len()
                                    ),
                                    }
                                })
                            })
                            .collect()
                    }
                }
            }
            AxisKind::Missing => Err(PbError::InvalidInput {
                what: format!("serve axis {axis} is a standalone Missing axis; unsupported in v1"),
            }),
        }
    };
    let data: Vec<Vec<u8>> = if n_rows >= PAR_MIN_ROWS {
        provenance
            .par_iter()
            .zip(grids.par_iter())
            .enumerate()
            .map(|(axis, (prov, grid))| bin_axis(axis, prov, grid))
            .collect::<Result<_, _>>()?
    } else {
        provenance
            .iter()
            .zip(grids)
            .enumerate()
            .map(|(axis, (prov, grid))| bin_axis(axis, prov, grid))
            .collect::<Result<_, _>>()?
    };
    Ok(ServeBinnedMatrix(BinnedMatrix {
        data,
        n_rows: u32::try_from(n_rows).map_err(|_| PbError::InvalidInput {
            what: "more than u32::MAX rows is out of scope for v1".into(),
        })?,
        grids: grids.to_vec(),
        provenance: provenance.to_vec(),
    }))
}

fn validate_optional_row_data(
    n_rows: usize,
    weight: Option<&[f32]>,
    exposure: Option<&[f32]>,
) -> Result<(), PbError> {
    if let Some(w) = weight {
        if w.len() != n_rows {
            return Err(PbError::ShapeMismatch {
                what: format!("weight len {} != n_rows {n_rows}", w.len()),
            });
        }
    }
    if let Some(e) = exposure {
        if e.len() != n_rows {
            return Err(PbError::ShapeMismatch {
                what: format!("exposure len {} != n_rows {n_rows}", e.len()),
            });
        }
    }
    Ok(())
}

fn bin_values(values: &[f32], grid: &BorderGrid) -> Result<Vec<u8>, PbError> {
    let mut out = Vec::with_capacity(values.len());
    for &value in values {
        out.push(bin(value, grid)?);
    }
    Ok(out)
}

fn infer_serve_n_rows(
    numeric: &[NumericColumn<'_>],
    categorical: &[ServeCategoricalColumn<'_>],
) -> Result<usize, PbError> {
    let mut n_rows = None;
    for col in numeric {
        n_rows = Some(match n_rows {
            Some(n) if n != col.values.len() => {
                return Err(PbError::ShapeMismatch {
                    what: "serve numeric columns have unequal lengths".into(),
                });
            }
            Some(n) => n,
            None => col.values.len(),
        });
    }
    for col in categorical {
        n_rows = Some(match n_rows {
            Some(n) if n != col.levels.len() => {
                return Err(PbError::ShapeMismatch {
                    what: "serve categorical columns have unequal lengths".into(),
                });
            }
            Some(n) => n,
            None => col.levels.len(),
        });
    }
    Ok(n_rows.unwrap_or(0))
}

fn find_numeric_column<'a>(
    columns: &'a [NumericColumn<'a>],
    raw: FeatureId,
) -> Result<&'a [f32], PbError> {
    let mut found = None;
    for col in columns {
        if col.raw == raw {
            if found.is_some() {
                return Err(PbError::ShapeMismatch {
                    what: format!("duplicate numeric raw column {raw:?}"),
                });
            }
            found = Some(col.values);
        }
    }
    found.ok_or_else(|| PbError::ShapeMismatch {
        what: format!("missing numeric raw column {raw:?}"),
    })
}

fn find_categorical_codes<'a>(
    columns: &'a [ServeCategoricalCodes<'a>],
    raw: FeatureId,
    id: TsEncodingId,
) -> Result<&'a ServeCategoricalCodes<'a>, PbError> {
    let mut found = None;
    for col in columns {
        if col.raw == raw && col.id == id {
            if found.is_some() {
                return Err(PbError::ShapeMismatch {
                    what: format!("duplicate categorical raw/id column {raw:?}/{id:?}"),
                });
            }
            found = Some(col);
        }
    }
    found.ok_or_else(|| PbError::ShapeMismatch {
        what: format!("missing categorical raw/id column {raw:?}/{id:?}"),
    })
}

fn infer_coded_n_rows(
    numeric: &[NumericColumn<'_>],
    categorical: &[ServeCategoricalCodes<'_>],
) -> Result<usize, PbError> {
    let lens = numeric
        .iter()
        .map(|c| c.values.len())
        .chain(categorical.iter().map(|c| c.codes.len()));
    let mut n_rows = None;
    for len in lens {
        match n_rows {
            Some(n) if n != len => {
                return Err(PbError::ShapeMismatch {
                    what: "serve columns have unequal lengths".into(),
                });
            }
            _ => n_rows = Some(len),
        }
    }
    Ok(n_rows.unwrap_or(0))
}

fn find_categorical_column<'a>(
    columns: &'a [ServeCategoricalColumn<'a>],
    raw: FeatureId,
    id: TsEncodingId,
) -> Result<&'a [String], PbError> {
    let mut found = None;
    for col in columns {
        if col.raw == raw && col.id == id {
            if found.is_some() {
                return Err(PbError::ShapeMismatch {
                    what: format!("duplicate categorical raw/id column {raw:?}/{id:?}"),
                });
            }
            found = Some(col.levels);
        }
    }
    found.ok_or_else(|| PbError::ShapeMismatch {
        what: format!("missing categorical raw/id column {raw:?}/{id:?}"),
    })
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
    use crate::cat::{LeakageScheme, Smooth, TsConfig, TsEncodingId};

    fn grid(borders: Vec<f32>) -> BorderGrid {
        let n_bins = u16::try_from(borders.len() + 2).unwrap();
        BorderGrid {
            borders,
            n_bins,
            missing_bin: 0,
        }
    }

    #[test]
    fn bin_semantics_nan_inf_and_borders() {
        let g = grid(vec![1.5, 2.5]); // 3 data bins: (-inf,1.5]→1, (1.5,2.5]→2, (2.5,inf)→3
        assert_eq!(bin(f32::NAN, &g).unwrap(), 0); // only NaN is missing
        assert_eq!(bin(1.0, &g).unwrap(), 1);
        assert_eq!(bin(1.5, &g).unwrap(), 1); // upper-inclusive
        assert_eq!(bin(2.0, &g).unwrap(), 2);
        assert_eq!(bin(3.0, &g).unwrap(), 3);
        // ±inf are non-NaN ⇒ clamp to first/last finite bin, NOT missing.
        assert_eq!(bin(f32::NEG_INFINITY, &g).unwrap(), 1);
        assert_eq!(bin(f32::INFINITY, &g).unwrap(), 3);
    }

    #[test]
    fn bin_is_non_decreasing_in_value() {
        let g = grid(vec![0.0, 1.0, 2.0, 3.0]);
        let mut prev = 0;
        for step in 0..=40 {
            let v = -1.0 + step as f32 * 0.1;
            let b = bin(v, &g).unwrap();
            assert!(b >= prev, "bin must be non-decreasing");
            prev = b;
        }
    }

    #[test]
    fn max_cardinality_grid_never_overflows_u8() {
        // 253 borders ⇒ 254 data bins ⇒ max bin id 254 (fits u8, no panic).
        let borders: Vec<f32> = (0..253).map(|i| i as f32).collect();
        let g = grid(borders);
        assert_eq!(g.n_bins, 255);
        // A value above the top border lands in the 254th data bin.
        assert_eq!(bin(1.0e9, &g).unwrap(), 254);
        assert_eq!(bin(f32::INFINITY, &g).unwrap(), 254);
        // Every finite probe stays in 1..=254, missing stays 0.
        for step in 0..300 {
            let b = bin(step as f32 - 10.0, &g).unwrap();
            assert!((1..=254).contains(&b));
        }
        assert_eq!(bin(f32::NAN, &g).unwrap(), 0);
    }

    #[test]
    fn bin_columns_populates_provenance_and_shapes() {
        let c0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let c1: Vec<f32> = vec![10.0, 10.0, 20.0, f32::NAN];
        let cols: Vec<&[f32]> = vec![&c0, &c1];
        let m = bin_columns(&cols, None, &BinConfig::default(), 0).unwrap();
        assert_eq!(m.n_rows, 4);
        assert_eq!(m.data.len(), 2);
        assert_eq!(m.grids.len(), 2);
        assert_eq!(m.provenance[0].raw, FeatureId(0));
        assert_eq!(m.provenance[1].raw, FeatureId(1));
        assert!(matches!(m.provenance[0].kind, AxisKind::Numeric));
        // The NaN in c1 must bin to 0 (missing).
        assert_eq!(m.data[1][3], 0);
    }

    #[test]
    fn bin_columns_rejects_unequal_columns() {
        let c0: Vec<f32> = vec![1.0, 2.0];
        let c1: Vec<f32> = vec![1.0];
        let cols: Vec<&[f32]> = vec![&c0, &c1];
        assert!(matches!(
            bin_columns(&cols, None, &BinConfig::default(), 0),
            Err(PbError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn mixed_train_and_serve_constructors_preserve_categorical_schema() {
        let numeric = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let levels = vec!["low", "high", "low", "high", "mid", "mid"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = [1.0_f32, 10.0, 2.0, 12.0, 5.0, 6.0];
        let ts = TsConfig {
            leakage: LeakageScheme::KFold { k: 3 },
            smooth: Smooth::Fixed { m: 0.0 },
            min_data_per_group: 0.0,
            ..TsConfig::default()
        };
        let fitted = bin_train_columns(
            &[NumericColumn {
                raw: FeatureId(0),
                values: &numeric,
            }],
            &[CategoricalColumn {
                raw: FeatureId(1),
                id: TsEncodingId(0),
                levels: &levels,
                config: &ts,
            }],
            &y,
            None,
            None,
            &BinConfig::default(),
            11,
        )
        .unwrap();

        assert_eq!(fitted.cat_encoders.len(), 1);
        assert_eq!(fitted.train.0.data.len(), 2);
        assert_eq!(fitted.serve.0.data.len(), 2);
        assert_eq!(fitted.train.0.data[0], fitted.serve.0.data[0]);
        assert!(matches!(
            fitted.serve.0.provenance[1].kind,
            AxisKind::CategoricalTS {
                encoding: TsEncodingId(0)
            }
        ));

        let new_numeric = vec![1.0_f32, 2.0, 3.0];
        let new_levels = vec!["high", "unseen", "low"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let served = bin_serve_columns(
            &[NumericColumn {
                raw: FeatureId(0),
                values: &new_numeric,
            }],
            &[ServeCategoricalColumn {
                raw: FeatureId(1),
                id: TsEncodingId(0),
                levels: &new_levels,
            }],
            &fitted.serve.0.grids,
            &fitted.serve.0.provenance,
            &fitted.cat_encoders,
        )
        .unwrap();
        assert_eq!(served.0.n_rows, 3);
        let enc = fitted
            .cat_encoders
            .get(TsEncodingId(0), FeatureId(1))
            .unwrap();
        let cat_grid = enc.border_grid().unwrap();
        let unseen_bin = bin(enc.base, &cat_grid).unwrap();
        assert_eq!(served.0.data[1][1], unseen_bin);

        // Codes into distinct labels (labels may repeat) and cached categorical maps bin exactly
        // like one label per row.
        let labels = ["low", "high", "unseen", "high"]
            .map(str::to_owned)
            .to_vec();
        let numeric = [NumericColumn {
            raw: FeatureId(0),
            values: &new_numeric,
        }];
        let coded_col = |codes: &'static [u32]| ServeCategoricalCodes {
            raw: FeatureId(1),
            id: TsEncodingId(0),
            codes,
            labels: &labels,
        };
        let maps = CatServeMaps::build(&fitted.serve.0.provenance, &fitted.cat_encoders).unwrap();
        for cat_maps in [None, Some(&maps)] {
            let coded = bin_serve_columns_coded(
                &numeric,
                &[coded_col(&[3, 2, 0])], // high, unseen, low
                &fitted.serve.0.grids,
                &fitted.serve.0.provenance,
                &fitted.cat_encoders,
                cat_maps,
            )
            .unwrap();
            assert_eq!(coded.0.data, served.0.data);
            let labelled = bin_serve_columns_with(
                &numeric,
                &[ServeCategoricalColumn {
                    raw: FeatureId(1),
                    id: TsEncodingId(0),
                    levels: &new_levels,
                }],
                &fitted.serve.0.grids,
                &fitted.serve.0.provenance,
                &fitted.cat_encoders,
                cat_maps,
            )
            .unwrap();
            assert_eq!(labelled.0.data, served.0.data);
        }
        // A code outside its labels is refused, never clamped.
        assert!(bin_serve_columns_coded(
            &numeric,
            &[coded_col(&[3, 9, 0])],
            &fitted.serve.0.grids,
            &fitted.serve.0.provenance,
            &fitted.cat_encoders,
            None,
        )
        .is_err());
    }

    /// P1 multi-channel (design/multichannel-categoricals.md §4.2): two `CategoricalColumn`s
    /// sharing one `raw` feature with distinct `id`s (a mean-TS channel plus a target-free
    /// count channel) fit and serve cleanly through the existing plumbing — no bin.rs change
    /// was needed for this, only a channel that computes a value without a target.
    #[test]
    fn same_raw_multiple_encoder_ids_fit_and_serve_both_channels() {
        let levels = vec!["low", "high", "low", "high", "mid", "mid"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = [1.0_f32, 10.0, 2.0, 12.0, 5.0, 6.0];
        let mean_ts = TsConfig {
            leakage: LeakageScheme::KFold { k: 3 },
            smooth: Smooth::Fixed { m: 0.0 },
            min_data_per_group: 0.0,
            ..TsConfig::default()
        };
        let count_ts = TsConfig {
            target: crate::cat::CatTarget::Count,
            min_data_per_group: 0.0,
            ..TsConfig::default()
        };
        let fitted = bin_train_columns(
            &[],
            &[
                CategoricalColumn {
                    raw: FeatureId(0),
                    id: TsEncodingId(0),
                    levels: &levels,
                    config: &mean_ts,
                },
                CategoricalColumn {
                    raw: FeatureId(0),
                    id: TsEncodingId(1),
                    levels: &levels,
                    config: &count_ts,
                },
            ],
            &y,
            None,
            None,
            &BinConfig::default(),
            11,
        )
        .unwrap();

        // Both channels fit, both share raw 0, and both carry distinct encoder ids.
        assert_eq!(fitted.cat_encoders.len(), 2);
        assert_eq!(fitted.train.0.data.len(), 2);
        assert_eq!(fitted.serve.0.data.len(), 2);
        for (axis, want_id) in [(0usize, TsEncodingId(0)), (1, TsEncodingId(1))] {
            assert_eq!(fitted.serve.0.provenance[axis].raw, FeatureId(0));
            assert!(matches!(
                fitted.serve.0.provenance[axis].kind,
                AxisKind::CategoricalTS { encoding } if encoding == want_id
            ));
        }

        // Serving re-encodes BOTH channels through their own frozen maps.
        let new_levels = vec!["high", "unseen", "low"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let served = bin_serve_columns(
            &[],
            &[
                ServeCategoricalColumn {
                    raw: FeatureId(0),
                    id: TsEncodingId(0),
                    levels: &new_levels,
                },
                ServeCategoricalColumn {
                    raw: FeatureId(0),
                    id: TsEncodingId(1),
                    levels: &new_levels,
                },
            ],
            &fitted.serve.0.grids,
            &fitted.serve.0.provenance,
            &fitted.cat_encoders,
        )
        .unwrap();
        assert_eq!(served.0.n_rows, 3);
        // The count channel's unseen-level bin matches its own (base=0.0) grid, independent of
        // the mean channel's own unseen bin.
        let count_enc = fitted
            .cat_encoders
            .get(TsEncodingId(1), FeatureId(0))
            .unwrap();
        assert_eq!(count_enc.base, 0.0);
        let count_grid = count_enc.border_grid().unwrap();
        let unseen_count_bin = bin(count_enc.base, &count_grid).unwrap();
        assert_eq!(served.0.data[1][1], unseen_count_bin);
    }
}
