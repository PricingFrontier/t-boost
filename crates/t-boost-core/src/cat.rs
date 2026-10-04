//! Categorical handling (spec §04). This module owns the frozen Target-Statistic
//! encoder schema that is serialized with a [`crate::Model`]. The leakage-free fit
//! algorithms land on top of these types; the store and lookup semantics are already
//! deterministic and total.

// A wide public categorical-fit signature trips clippy::too_many_arguments (a style lint promoted
// to an error by CI's `-D warnings`); scope-allowed for this module.
#![allow(clippy::too_many_arguments, clippy::derivable_impls)]

use crate::data::{bin::bin as bin_value, AxisKind, AxisProvenance, BorderGrid, FeatureId};
use crate::error::PbError;
use crate::{pb_seed, Stage};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

mod training_prior;
use training_prior::TrainingPriorIndex;

const MAX_CAT_BINS: usize = 254;
const MAX_AUTO_SMOOTH: f64 = 1_000_000.0;
const MIN_AUTO_VARIANCE: f64 = 1.0e-12;
// pub(crate): serialize.rs's Piece B label export (design/multichannel-categoricals.md §4.4)
// needs to detect the reserved rare bucket by this exact sentinel so it can show a clear
// synthetic display label instead -- a pure Rust-visibility widening, no wire-format change
// (the stored label is still this same string either way).
pub(crate) const RARE_LEVEL_LABEL: &str = "__t_boost_rare__";

type RareMembers = BTreeMap<String, Vec<String>>;

/// Identifier for one categorical Target-Statistic encoding (spec §04). Resolves to
/// a concrete [`CatEncoder`] in the [`CatEncoderStore`].
///
/// Append-only and fixed-width (`u8`): it is serialized inside
/// [`crate::data::AxisKind::CategoricalTS`], so it must never be a platform-width int.
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct TsEncodingId(pub u8);

/// Leakage-avoidance scheme used to produce training-time categorical encodings
/// (spec §04.3). Serve-time encodings always use the frozen full-data map stored in
/// [`CatEncoder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeakageScheme {
    /// Ordered target statistics over `n_perms` seeded permutations.
    Ordered {
        /// Number of deterministic permutations.
        n_perms: u32,
    },
    /// K-fold cross-fit target statistics.
    KFold {
        /// Number of folds.
        k: u32,
    },
    /// Leave-one-out target statistics: each row sees its level's full statistics with
    /// ITSELF removed. Deterministic and knob-free, but NOT the default: although it is the
    /// nominal cross-fit variance floor, it reintroduces a per-row dependence on the row's
    /// own target (`enc ≈ (Σ − yᵣ)/(n−1)`) that depth-3 trees partially invert — measured
    /// WORSE than KFold on both MTPL tasks (the classic LOO target-encoding pathology).
    LeaveOneOut,
}

impl Default for LeakageScheme {
    fn default() -> Self {
        // K-fold cross-fit: each row's encoding comes from OTHER folds, so (unlike LOO) rows
        // in the same fold share a held-out value and the per-row self-dependence is broken.
        // Empirically the lowest-variance / best-accuracy leakage-free scheme on MTPL
        // (beats the old `Ordered{1}` default and LOO on both frequency and severity).
        Self::KFold { k: 5 }
    }
}

/// Smoothing rule for target-statistic shrinkage (spec §04.3).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Smooth {
    /// Fixed pseudo-count `m` in `(n_c·mean_c + m·base)/(n_c + m)`.
    Fixed {
        /// Pseudo-count strength.
        m: f32,
    },
    /// Estimate `m` from the data: credible-set credibility for 3+-level features (tau^2 from
    /// levels clearing 3 sigma of their own sampling noise, null-corrected; no credible signal
    /// means full shrink), and the within/between variance ratio for binary features (their
    /// partition is encoding-invariant, so the legacy path is kept bit-for-bit).
    Auto,
}

impl Default for Smooth {
    fn default() -> Self {
        Self::Fixed { m: 20.0 }
    }
}

/// Target transform used by the categorical target-statistic encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CatTarget {
    /// Encode the exposure-weighted mean target, matching the original TS path.
    Mean,
    /// Encode the weighted mean of `log(y)`, useful for positive severity targets.
    LogMean,
    /// Target-FREE level exposure share (P1 multi-channel spec §3): `log1p(Σw_level /
    /// Σw_total × N)`, frozen at fit with NO cross-fit/leakage/smoothing machinery — nothing
    /// in [`fit_cat_encoder`]'s `Count` path reads `y`. Shares the same rare-level pooling
    /// (`min_data_per_group`) as the target channels so a rare bucket gets one representative
    /// share value; `base = 0.0` for unseen levels (maximally rare, `log1p(0)`). Intended as a
    /// SECOND axis alongside `Mean`/`LogMean` on the same raw feature (distinct
    /// [`CatEncoder::id`], same [`CatEncoder::raw`]) — a "count/rarity" companion to the
    /// existing mean-TS channel, never a drop-in replacement for it.
    Count,
    /// P3 multi-channel (design/multichannel-categoricals.md §8 P3): the one-vs-rest
    /// **per-class frequency** of `class` within a level — `E[1[y == class] | level]`, smoothed
    /// and cross-fit by the SAME machinery `Mean` uses (nothing here is a new leakage surface;
    /// the only difference from `Mean` is the row target transform `y -> 1[y == class]`).
    ///
    /// Exists because the K>=3 softmax path bins categoricals ONCE against the integer class
    /// labels, so a plain `Mean` encoder computes `E[label | level]` — an ORDINAL statistic over
    /// nominal classes (`design/multiclass-design.md` §9a §5a, the documented multiclass
    /// categorical weakness). A set of K `ClassFreq` channels on one raw feature carries the
    /// full per-level class distribution instead, losing nothing to the label ordering.
    ///
    /// Rare-level pooling uses `Σ(w·e)` — byte-for-byte the same denominator as `Mean` and
    /// `Count` (see [`categorical_row_terms`]) — which is load-bearing: every channel of one raw
    /// feature MUST partition levels into the same rare bucket or [`JointCatAxis::build`]
    /// rejects the set.
    ClassFreq {
        /// Class index this channel encodes the frequency of, matching the multiclass label
        /// encoding (`0..n_classes`) the softmax path fits against.
        class: u32,
    },
}

impl Default for CatTarget {
    fn default() -> Self {
        Self::Mean
    }
}

/// Configuration for one target-statistic encoder (spec §04.3/§04.12).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TsConfig {
    /// Leakage-avoidance scheme for training rows.
    pub leakage: LeakageScheme,
    /// Shrinkage rule.
    pub smooth: Smooth,
    /// Target transform used before shrinkage.
    pub target: CatTarget,
    /// Number of target borders used by the encoder's target pre-binning.
    pub target_borders: u32,
    /// Low-cardinality DIRECT threshold: a feature with 3..=this many distinct
    /// (post-rare-pooling) levels bypasses cross-fit encodings and credibility smoothing — each
    /// level keeps its raw full-fit target statistic, so it gets its own bin ordered by that
    /// statistic, and the trees fit the level values directly. At low cardinality the encoding
    /// carries order-only information (leaf values are refit by boosting), and the cross-fit
    /// noise plus per-feature Auto-credibility measurably cost both deviance and ranking
    /// (2026-07-10 battery: swautoins/norauto/euhealth improve; the rest are neutral).
    /// BINARY features are exempt regardless of the threshold: with 2 levels the partition is
    /// encoding-invariant, so bypassing buys nothing and only perturbs early-stopping paths
    /// (measured: pure per-split noise, materially harmful on one Tweedie case). `0` disables
    /// the bypass. Serialized as `one_hot_max_size` (the field's historical name — wire
    /// compatibility with stored models).
    #[serde(rename = "one_hot_max_size")]
    pub direct_max_levels: u32,
    /// Weighted-count floor below which levels collapse into the rare bucket.
    pub min_data_per_group: f32,
}

impl Default for TsConfig {
    fn default() -> Self {
        Self {
            leakage: LeakageScheme::default(),
            smooth: Smooth::default(),
            target: CatTarget::default(),
            target_borders: 16,
            direct_max_levels: 16,
            min_data_per_group: 10.0,
        }
    }
}

impl TsConfig {
    /// Validate categorical encoder configuration.
    ///
    /// # Errors
    /// [`PbError::InvalidConfig`] if a count/smoothing parameter is outside its
    /// finite domain.
    pub fn validate(&self) -> Result<(), PbError> {
        match self.leakage {
            LeakageScheme::Ordered { n_perms } => {
                if n_perms == 0 {
                    return Err(PbError::InvalidConfig {
                        what: "Ordered target statistics require n_perms > 0".into(),
                    });
                }
            }
            LeakageScheme::KFold { k } => {
                if k < 2 {
                    return Err(PbError::InvalidConfig {
                        what: format!("KFold target statistics require k >= 2, got {k}"),
                    });
                }
            }
            LeakageScheme::LeaveOneOut => {}
        }
        if self.target_borders == 0 {
            return Err(PbError::InvalidConfig {
                what: "target_borders must be > 0".into(),
            });
        }
        if !self.min_data_per_group.is_finite() || self.min_data_per_group < 0.0 {
            return Err(PbError::InvalidConfig {
                what: format!(
                    "min_data_per_group must be finite and >= 0, got {}",
                    self.min_data_per_group
                ),
            });
        }
        if let Smooth::Fixed { m } = self.smooth {
            if !m.is_finite() || m < 0.0 {
                return Err(PbError::InvalidConfig {
                    what: format!("Smooth::Fixed m must be finite and >= 0, got {m}"),
                });
            }
        }
        Ok(())
    }
}

/// One frozen categorical level in serve/export order (spec §04.4/§04.12).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatLevel {
    /// Human-readable level label. The reserved rare bucket is represented as its own
    /// label when rare-level collapse is active.
    pub label: String,
    /// Original labels represented by this row. Non-rare rows contain exactly their
    /// own label; the rare row contains all collapsed training labels.
    pub members: Vec<String>,
    /// Full-data shrunken target-statistic value used at serve time.
    pub encoding: f32,
    /// Ordinal bin id emitted by the Fisher-sorted categorical axis.
    pub bin: u8,
    /// Effective weighted count behind this level.
    pub weight: f32,
}

/// One frozen, full-data categorical encoder (spec §04). Training-time encodings are
/// leakage-free views; this stored encoder is the serve/export map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatEncoder {
    /// Raw feature this encoder belongs to.
    pub raw: FeatureId,
    /// Encoder id, unique within [`CatEncoderStore`] for a raw feature.
    pub id: TsEncodingId,
    /// Frozen levels in deterministic serve/export order.
    pub levels: Vec<CatLevel>,
    /// Base value for unseen levels at serve time.
    pub base: f32,
    /// Configuration that produced the encoder.
    pub config: TsConfig,
}

/// The smallest `f32` strictly greater than `x` (standard IEEE-754 `nextafter`
/// bit-stepping; MSRV 1.85 predates the stable `f32::next_up`, stabilized in 1.86).
/// `NaN`/`+inf` pass through unchanged; `+0.0`/`-0.0` both step to the smallest
/// positive subnormal.
fn next_up_f32(x: f32) -> f32 {
    if x.is_nan() || x == f32::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f32::from_bits(1);
    }
    let bits = x.to_bits();
    f32::from_bits(if x > 0.0 { bits + 1 } else { bits - 1 })
}

/// The largest `f32` strictly less than `x`; `next_down(x) == -next_up(-x)`.
fn next_down_f32(x: f32) -> f32 {
    -next_up_f32(-x)
}

impl CatEncoder {
    /// Serve-time target-statistic value for `label`; unseen levels map to
    /// [`CatEncoder::base`] (§04.8).
    #[must_use]
    pub fn encode_label(&self, label: &str) -> f32 {
        self.levels
            .iter()
            .find(|level| level.label == label || level.members.iter().any(|m| m == label))
            .map_or(self.base, |level| level.encoding)
    }

    /// Build a `label → encoding` lookup map (each level's own label AND its `members`) for batch
    /// serve-time encoding — O(1) per label vs [`encode_label`]'s O(#levels) linear scan. Labels and
    /// members are disjoint across levels, so `map.get(label)` returns exactly what `encode_label`'s
    /// `find` would; unseen labels still fall back to [`CatEncoder::base`] at the call site. The map
    /// is a local lookup (never iterated for output order, never serialized).
    #[must_use]
    pub fn encoding_map(&self) -> std::collections::HashMap<&str, f32> {
        let mut map = std::collections::HashMap::with_capacity(self.levels.len());
        for level in &self.levels {
            map.insert(level.label.as_str(), level.encoding);
            for member in &level.members {
                map.insert(member.as_str(), level.encoding);
            }
        }
        map
    }

    /// Reconstruct the ordinal Fisher [`BorderGrid`] for this encoder.
    ///
    /// The grid is derived from the stored level encodings and their frozen bin ids,
    /// so it round-trips with the model schema without persisting duplicate border
    /// state. Unseen levels are encoded as [`CatEncoder::base`] and then binned
    /// against this same grid.
    ///
    /// # Errors
    /// [`PbError::InvalidInput`] if the stored encoder has non-finite encodings or
    /// inconsistent bin ordering; [`PbError::Internal`] on impossible width casts.
    pub fn border_grid(&self) -> Result<BorderGrid, PbError> {
        let mut levels = self.levels.clone();
        levels.sort_by(|a, b| {
            a.encoding
                .total_cmp(&b.encoding)
                .then_with(|| a.label.cmp(&b.label))
        });

        let mut borders = Vec::new();
        let mut prev: Option<(u8, f32)> = None;
        for level in &levels {
            if !level.encoding.is_finite() {
                return Err(PbError::InvalidInput {
                    what: format!(
                        "categorical encoder {:?}/{:?} has non-finite encoding for `{}`",
                        self.raw, self.id, level.label
                    ),
                });
            }
            if level.bin == 0 {
                return Err(PbError::InvalidInput {
                    what: format!(
                        "categorical encoder {:?}/{:?} assigned reserved bin 0 to `{}`",
                        self.raw, self.id, level.label
                    ),
                });
            }
            if let Some((prev_bin, prev_encoding)) = prev {
                if level.bin < prev_bin {
                    return Err(PbError::InvalidInput {
                        what: "categorical encoder bins must be non-decreasing in encoding order"
                            .into(),
                    });
                }
                if level.bin > prev_bin {
                    if level.encoding <= prev_encoding {
                        return Err(PbError::InvalidInput {
                            what: "categorical encoder split tied encodings across bins".into(),
                        });
                    }
                    let mut border =
                        ((f64::from(prev_encoding) + f64::from(level.encoding)) / 2.0) as f32;
                    if !border.is_finite() {
                        return Err(PbError::InvalidInput {
                            what: "categorical Fisher border is not finite".into(),
                        });
                    }
                    // The f64 midpoint's cast to f32 can round exactly onto the UPPER
                    // encoding when the two are exactly 1 ULP apart (near-tied shrunken
                    // means on high-cardinality columns): bin() counts borders strictly
                    // below v, so border == level.encoding would silently merge this level
                    // into the previous bin, diverging from the bin id already assigned at
                    // Fisher-sort time. Clamping down to the largest f32 < level.encoding
                    // always still separates the pair correctly, even in the 1-ULP case:
                    // bin()'s "strictly below" rule only needs border == prev_encoding to
                    // route prev_encoding low and level.encoding high, so this is exact,
                    // not an approximation — there is no case a border can fail to separate.
                    if border >= level.encoding {
                        border = next_down_f32(level.encoding);
                    }
                    if borders.last().is_some_and(|last| *last >= border) {
                        return Err(PbError::InvalidInput {
                            what: "categorical Fisher borders must be strictly ascending".into(),
                        });
                    }
                    borders.push(border);
                }
            }
            prev = Some((level.bin, level.encoding));
        }
        let n_bins =
            u16::try_from(
                borders
                    .len()
                    .checked_add(2)
                    .ok_or_else(|| PbError::Internal {
                        what: "categorical border count overflow".into(),
                    })?,
            )
            .map_err(|_| PbError::Internal {
                what: "categorical n_bins exceeded u16".into(),
            })?;
        Ok(BorderGrid {
            borders,
            n_bins,
            missing_bin: 0,
        })
    }
}

/// Per-fit categorical encoder specification (spec §04.3).
pub struct CatFitSpec<'a> {
    /// Raw feature this encoder belongs to.
    pub raw: FeatureId,
    /// Encoder id to stamp into [`crate::data::AxisKind::CategoricalTS`].
    pub id: TsEncodingId,
    /// Optional per-row weights; absent means all ones.
    pub weight: Option<&'a [f32]>,
    /// Optional exposure values `e_i`; absent means `e_i = 1`.
    pub exposure: Option<&'a [f32]>,
    /// Encoder configuration.
    pub config: &'a TsConfig,
    /// Deterministic base seed.
    pub seed: u64,
}

/// The frozen `TsEncodingId → CatEncoder` table backing serve/export (spec §2.6 /
/// §04, R-SCHEMA). `explain()` and `TableBank` accumulation re-encode raw
/// categoricals through THESE (never the noisy train-time encoders).
///
/// Backed by a `Vec` (not `HashMap`): serialized state must have deterministic
/// iteration order (the `check-no-hashmap-serialized` gate).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CatEncoderStore {
    encoders: Vec<CatEncoder>,
}

impl CatEncoderStore {
    /// An empty store (no categorical axes). Used by purely-numeric models.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a store from encoders already in deterministic order.
    #[must_use]
    pub fn from_encoders(encoders: Vec<CatEncoder>) -> Self {
        Self { encoders }
    }

    /// The frozen encoders in deterministic order.
    #[must_use]
    pub fn encoders(&self) -> &[CatEncoder] {
        &self.encoders
    }

    /// Every training label each categorical raw feature knows, in serve/export order: each
    /// level's members (a rare bucket contributes the labels it pooled, never its reserved
    /// label). A feature with several encoders (multi-channel) lists each label once. A label
    /// absent from its feature's list scores at the encoder base — an unseen level.
    #[must_use]
    pub fn known_labels(&self) -> BTreeMap<u32, Vec<String>> {
        let mut out: BTreeMap<u32, Vec<String>> = BTreeMap::new();
        let mut seen: BTreeMap<u32, BTreeSet<&str>> = BTreeMap::new();
        for encoder in &self.encoders {
            let labels = out.entry(encoder.raw.0).or_default();
            let known = seen.entry(encoder.raw.0).or_default();
            for level in &encoder.levels {
                let own = std::slice::from_ref(&level.label);
                let members = if level.members.is_empty() {
                    own
                } else {
                    &level.members
                };
                for label in members {
                    if label != RARE_LEVEL_LABEL && known.insert(label.as_str()) {
                        labels.push(label.clone());
                    }
                }
            }
        }
        out
    }

    /// `true` if no categorical encoders are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.encoders.is_empty()
    }

    /// Number of registered encoders.
    #[must_use]
    pub fn len(&self) -> usize {
        self.encoders.len()
    }

    /// Look up one encoder by `(id, raw)` without panicking.
    ///
    /// # Errors
    /// [`PbError::Internal`] if no matching encoder is present. A model that names a
    /// missing encoder is internally inconsistent, not malformed user data.
    pub fn get(&self, id: TsEncodingId, raw: FeatureId) -> Result<&CatEncoder, PbError> {
        self.encoders
            .iter()
            .find(|enc| enc.id == id && enc.raw == raw)
            .ok_or_else(|| PbError::Internal {
                what: format!("categorical encoder {id:?} for raw feature {raw:?} not found"),
            })
    }
}

// ===========================================================================
// P1 multi-channel (design/multichannel-categoricals.md §4.3): the joint per-level cell
// space for a raw feature with more than one categorical channel axis. Lives here (not in
// explain.rs/scoring.rs) so both can build/consume the IDENTICAL mapping from the SAME
// source of truth (the frozen `CatEncoderStore`) — the exactness guarantee that a served,
// pruned tables-only model reproduces the tree ensemble depends on there being exactly one
// implementation of "row bins -> joint cell", not two independently-derived ones.
// ===========================================================================

/// One categorical channel axis of a (possibly multi-channel) raw feature: its encoder id and
/// the model axis (column index into a `BinnedMatrix`/`Model::provenance`) it was bound to at
/// fit time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChannelAxis {
    /// The encoder id this channel was fit under.
    pub id: TsEncodingId,
    /// The model axis (column index) carrying this channel.
    pub model_axis: usize,
}

/// Every categorical channel axis for raw feature `raw`, sorted by [`TsEncodingId`] ascending —
/// the canonical, deterministic per-raw channel order the rest of the P1 multi-channel design
/// relies on. Empty for a purely numeric raw feature; a single entry for today's ordinary
/// (single-channel) categorical; `>= 2` entries only for a `cat_channels`-enabled fit.
#[must_use]
pub(crate) fn channel_axes_for_raw(
    provenance: &[AxisProvenance],
    raw: FeatureId,
) -> Vec<ChannelAxis> {
    let mut out: Vec<ChannelAxis> = provenance
        .iter()
        .enumerate()
        .filter_map(|(axis, prov)| {
            if prov.raw != raw {
                return None;
            }
            match prov.kind {
                AxisKind::CategoricalTS { encoding } => Some(ChannelAxis {
                    id: encoding,
                    model_axis: axis,
                }),
                _ => None,
            }
        })
        .collect();
    out.sort_by_key(|c| c.id);
    out
}

/// The joint per-level cell mapping for a multi-channel categorical raw feature (P1
/// multi-channel spec §4.3): the flattened combination of every channel's own model bin,
/// treated as ONE scalar cell per raw feature — so `x_cells[raw]`-style per-raw addressing
/// (explain.rs, scoring.rs) needs no change, only how that one scalar gets computed. Built
/// ONCE from the frozen [`CatEncoderStore`] and shared verbatim by explain.rs's table
/// construction and scoring.rs's served prediction, so the two can never independently drift
/// apart on which cell a row belongs to.
///
/// Cell `0` is reserved for missing (mirrors the single-channel `BorderGrid`/merged-axis
/// convention): every channel encodes the SAME raw label, so a row's categorical value being
/// missing routes it to bin 0 on every channel identically, never a mix. Cells `1..=n_cells-1`
/// are one per DISTINCT `(bin_c0, bin_c1, ...)` signature that actually occurs across the
/// channels' frozen levels, PLUS (if not already among them) one more for the shared
/// "genuinely unseen" fallback tuple every channel produces via its own `base` — bounded by
/// the number of distinct (post-rare-pooling) levels plus one (every seen-level signature is
/// the image of a deterministic function of "which level", never a bin-count product — spec
/// §4.3's compactness note; the unseen tuple is the one further addition, needed so a level
/// absent from THIS fit — e.g. a CV fold's held rows scored against that fold's own encoder —
/// resolves to a well-defined cell instead of erroring). Two levels (seen or the unseen
/// fallback) that happen to share every channel's bin share one cell — correct: the model
/// genuinely cannot distinguish them (spec §6 edge case), and forcing them apart would invent
/// a distinction the fitted trees don't make.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct JointCatAxis {
    /// Channel axes, sorted by encoder id (parallel to internal per-channel bookkeeping).
    pub channels: Vec<ChannelAxis>,
    /// Full per-channel-bin tuple (parallel to `channels`) -> joint cell, `1..=n_cells-1`.
    tuple_to_cell: BTreeMap<Vec<u8>, u32>,
    /// Inverse of `tuple_to_cell`: joint cell -> a representative per-channel-bin tuple
    /// (parallel to `channels`). Index `0` is an unused placeholder — missing is handled
    /// structurally by every consumer, never looked up here.
    cell_to_tuple: Vec<Vec<u8>>,
    /// Total cells including the reserved missing cell (`== cell_to_tuple.len()`).
    pub n_cells: u32,
    /// Canonical (lowest-id) channel's own level label -> its joint cell, in that channel's
    /// frozen (Fisher-sorted) order — computed once here, alongside `tuple_to_cell`, so Piece B's
    /// label export (design/multichannel-categoricals.md §4.4, serialize.rs) never needs a
    /// second, independent encode+bin walk that could drift from the cell assignment actually
    /// used. Every canonical level appears exactly once; levels sharing a cell (identical
    /// signature on every channel) correctly repeat that cell id, never invent a new one.
    pub(crate) level_cells: Vec<(String, u32)>,
}

/// `label -> encoding` over `enc`'s levels (own label, then pooled members), keeping the FIRST
/// level that claims a label — the level [`CatEncoder::encode_label`]'s `find` returns.
fn first_match_encodings(enc: &CatEncoder) -> std::collections::HashMap<&str, f32> {
    let mut map = std::collections::HashMap::with_capacity(enc.levels.len());
    for level in &enc.levels {
        map.entry(level.label.as_str()).or_insert(level.encoding);
        for member in &level.members {
            map.entry(member.as_str()).or_insert(level.encoding);
        }
    }
    map
}

impl JointCatAxis {
    /// Build the joint axis for raw feature `raw` from its channel axes (already located via
    /// [`channel_axes_for_raw`]), the model's per-axis grids (for each channel's own
    /// `BorderGrid`), and the frozen encoder store.
    ///
    /// # Errors
    /// [`PbError::Internal`] if fewer than 2 channels are supplied (the caller's bug — a
    /// single-channel raw feature should never reach this constructor), a channel's model
    /// axis has no grid, or a bin computation escapes `u32`; [`PbError::InvalidConfig`] if the
    /// channels' frozen level sets disagree — a paired mean+count fit must partition levels
    /// identically (see [`fit_count_encoder`]'s doc); this is the defensive check that catches
    /// it if that invariant is ever violated, e.g. by a caller bypassing the py binding's
    /// automatic config sharing. Propagates [`CatEncoderStore::get`] / [`crate::data::bin::bin`]
    /// errors otherwise.
    pub(crate) fn build(
        raw: FeatureId,
        channels: Vec<ChannelAxis>,
        grids: &[BorderGrid],
        cat_encoders: &CatEncoderStore,
    ) -> Result<Self, PbError> {
        if channels.len() < 2 {
            return Err(PbError::Internal {
                what: format!(
                    "joint categorical axis for raw {} needs >= 2 channels, got {}",
                    raw.0,
                    channels.len()
                ),
            });
        }
        let mut encoders = Vec::with_capacity(channels.len());
        for ch in &channels {
            if grids.get(ch.model_axis).is_none() {
                return Err(PbError::Internal {
                    what: format!(
                        "joint axis channel {:?} model axis {} missing grid",
                        ch.id, ch.model_axis
                    ),
                });
            }
            encoders.push(cat_encoders.get(ch.id, raw)?);
        }

        // Consistency check: every channel must partition levels identically (same label
        // set) — load-bearing per `fit_count_encoder`'s doc, checked here defensively rather
        // than assumed, since this type is the one place multi-channel correctness hinges on
        // it.
        let canonical = *encoders.first().ok_or_else(|| PbError::Internal {
            what: "joint axis has no channel encoders after population".into(),
        })?;
        let canonical_id = channels
            .first()
            .ok_or_else(|| PbError::Internal {
                what: "joint axis has no channels after population".into(),
            })?
            .id;
        let canonical_labels: BTreeSet<&str> =
            canonical.levels.iter().map(|l| l.label.as_str()).collect();
        for (ch, enc) in channels.iter().zip(&encoders).skip(1) {
            let labels: BTreeSet<&str> = enc.levels.iter().map(|l| l.label.as_str()).collect();
            if labels != canonical_labels {
                return Err(PbError::InvalidConfig {
                    what: format!(
                        "raw feature {} channels {:?} and {:?} disagree on their level \
                         partition (rare-pooling must match across every channel of one raw \
                         feature): {} vs {} distinct levels",
                        raw.0,
                        canonical_id,
                        ch.id,
                        canonical_labels.len(),
                        labels.len()
                    ),
                });
            }
        }

        // Canonical level order: the lowest-id channel's own frozen (Fisher-sorted) level
        // list — deterministic, and every level is guaranteed present on every channel by the
        // check above.
        let mut tuple_to_cell: BTreeMap<Vec<u8>, u32> = BTreeMap::new();
        let mut cell_to_tuple: Vec<Vec<u8>> = vec![Vec::new()]; // index 0: unused placeholder
        let mut level_cells: Vec<(String, u32)> = Vec::with_capacity(canonical.levels.len());
        // Per-channel label -> encoding, first match kept: exactly `encode_label`'s linear `find`
        // (a level's own label or one of its pooled members, earliest level first), without the
        // O(#levels) scan per level that made this build quadratic in the cardinality.
        let lookups: Vec<std::collections::HashMap<&str, f32>> = encoders
            .iter()
            .map(|enc| first_match_encodings(enc))
            .collect();
        for level in &canonical.levels {
            let mut tuple = Vec::with_capacity(channels.len());
            for ((ch, enc), lookup) in channels.iter().zip(&encoders).zip(&lookups) {
                let grid = grids.get(ch.model_axis).ok_or_else(|| PbError::Internal {
                    what: "joint axis channel model axis missing grid (second pass)".into(),
                })?;
                let value = lookup
                    .get(level.label.as_str())
                    .copied()
                    .unwrap_or(enc.base);
                tuple.push(bin_value(value, grid)?);
            }
            let cell = match tuple_to_cell.entry(tuple.clone()) {
                std::collections::btree_map::Entry::Vacant(e) => {
                    let next_cell =
                        u32::try_from(cell_to_tuple.len()).map_err(|_| PbError::Internal {
                            what: "joint categorical axis cell count exceeded u32".into(),
                        })?;
                    cell_to_tuple.push(tuple);
                    e.insert(next_cell);
                    next_cell
                }
                std::collections::btree_map::Entry::Occupied(e) => *e.get(),
            };
            level_cells.push((level.label.clone(), cell));
        }

        // The unseen-level fallback (design/multichannel-categoricals.md follow-on: CV-pruning
        // scores a fold model against rows outside its own fold-train, where a high-card
        // level can be genuinely absent from that fold's frozen encoder). Every channel
        // independently falls back to its OWN `base` for any label absent from its frozen
        // levels (`CatEncoder::encode_label`'s documented fallback), and every channel of one
        // raw feature shares the IDENTICAL frozen level partition (enforced above), so a label
        // unseen to one channel is unseen to ALL of them simultaneously, always producing this
        // SAME base tuple — never a partial mix of "unseen on one channel, seen on another".
        // `base` is always a validated-finite statistic (never `NaN`), so this tuple can never
        // collide with the reserved all-missing cell 0 either. Register it now (reusing an
        // existing seen-level cell if the tuple happens to coincide, exactly like any other
        // level) so `cell_for_channel_bins` resolves a genuinely unseen level to a well-defined
        // cell instead of erroring — matching single-channel semantics, where `base` simply
        // bins into whatever bin its own encoded value naturally falls into.
        let mut base_tuple = Vec::with_capacity(channels.len());
        for (ch, enc) in channels.iter().zip(&encoders) {
            let grid = grids.get(ch.model_axis).ok_or_else(|| PbError::Internal {
                what: "joint axis channel model axis missing grid (base pass)".into(),
            })?;
            base_tuple.push(bin_value(enc.base, grid)?);
        }
        if let std::collections::btree_map::Entry::Vacant(e) =
            tuple_to_cell.entry(base_tuple.clone())
        {
            let next_cell = u32::try_from(cell_to_tuple.len()).map_err(|_| PbError::Internal {
                what: "joint categorical axis cell count exceeded u32".into(),
            })?;
            cell_to_tuple.push(base_tuple);
            e.insert(next_cell);
        }

        let n_cells = u32::try_from(cell_to_tuple.len()).map_err(|_| PbError::Internal {
            what: "joint categorical axis cell count exceeded u32".into(),
        })?;

        Ok(Self {
            channels,
            tuple_to_cell,
            cell_to_tuple,
            n_cells,
            level_cells,
        })
    }

    /// Joint cell for a row given its bin on EVERY channel (parallel to `self.channels`,
    /// channel-id-sorted order — the same order [`channel_axes_for_raw`] produces). Cell `0`
    /// (missing) iff every channel's bin is `0`.
    ///
    /// # Errors
    /// [`PbError::ShapeMismatch`] if `channel_bins.len() != self.channels.len()`;
    /// [`PbError::InvalidInput`] if the channels disagree on missingness (only reachable if a
    /// caller's grids/provenance are mismatched — every channel encodes the same raw label, so
    /// a frozen model never produces this), or the tuple was never seen at fit time (an unseen
    /// LEVEL always maps to `base` on every channel, which the frozen encoders guarantee IS a
    /// seen tuple, so reaching this branch at serve time means a channel's grid disagrees with
    /// what it was fit against).
    pub(crate) fn cell_for_channel_bins(&self, channel_bins: &[u8]) -> Result<u32, PbError> {
        if channel_bins.len() != self.channels.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "joint categorical axis: {} channel bins supplied, {} channels expected",
                    channel_bins.len(),
                    self.channels.len()
                ),
            });
        }
        let n_missing = channel_bins.iter().filter(|&&b| b == 0).count();
        if n_missing == channel_bins.len() {
            return Ok(0);
        }
        if n_missing > 0 {
            return Err(PbError::InvalidInput {
                what: "joint categorical axis: channels disagree on missingness for one row".into(),
            });
        }
        self.tuple_to_cell
            .get(channel_bins)
            .copied()
            .ok_or_else(|| PbError::InvalidInput {
                what: format!(
                    "joint categorical axis: bin combination {channel_bins:?} was never seen \
                     at fit time"
                ),
            })
    }

    /// Representative model bin for `cell` on the channel whose model axis is `model_axis`
    /// (tree-split routing addresses a raw feature's channels by model axis, e.g.
    /// `split.axis`, not by position within `self.channels`).
    ///
    /// # Errors
    /// [`PbError::Internal`] if `model_axis` names none of this axis's channels, or `cell`
    /// escapes `cell_to_tuple`.
    pub(crate) fn rep_model_bin_for_axis(
        &self,
        cell: usize,
        model_axis: usize,
    ) -> Result<u8, PbError> {
        if cell == 0 {
            return Ok(0);
        }
        let pos = self
            .channels
            .iter()
            .position(|c| c.model_axis == model_axis)
            .ok_or_else(|| PbError::Internal {
                what: format!("joint categorical axis: model axis {model_axis} is not a channel"),
            })?;
        let tuple = self
            .cell_to_tuple
            .get(cell)
            .ok_or_else(|| PbError::Internal {
                what: "joint categorical axis: cell escaped cell_to_tuple".into(),
            })?;
        tuple.get(pos).copied().ok_or_else(|| PbError::Internal {
            what: "joint categorical axis: channel position escaped tuple".into(),
        })
    }

    /// Representative per-channel bins for `cell`, written into `out` at each channel's own
    /// model axis position — used to reconstruct one full row (every axis at once) for gate
    /// checking.
    ///
    /// # Errors
    /// [`PbError::Internal`] if `cell` escapes `cell_to_tuple` or a channel's model axis
    /// escapes `out`.
    pub(crate) fn write_rep_bins(&self, cell: usize, out: &mut [u8]) -> Result<(), PbError> {
        if cell == 0 {
            for ch in &self.channels {
                let slot = out
                    .get_mut(ch.model_axis)
                    .ok_or_else(|| PbError::Internal {
                        what: "joint categorical axis: channel model axis escaped rep_bins".into(),
                    })?;
                *slot = 0;
            }
            return Ok(());
        }
        let tuple = self
            .cell_to_tuple
            .get(cell)
            .ok_or_else(|| PbError::Internal {
                what: "joint categorical axis: cell escaped cell_to_tuple".into(),
            })?
            .clone();
        for (ch, bin) in self.channels.iter().zip(tuple) {
            let slot = out
                .get_mut(ch.model_axis)
                .ok_or_else(|| PbError::Internal {
                    what: "joint categorical axis: channel model axis escaped rep_bins".into(),
                })?;
            *slot = bin;
        }
        Ok(())
    }
}

/// Exposure-weighted base rate for categorical target statistics (§04.3/§03.7):
/// `p = Σ w_i y_i / Σ w_i e_i`, with `e_i = 1` when `exposure` is absent.
///
/// # Errors
/// [`PbError::ShapeMismatch`] on length mismatch; [`PbError::InvalidInput`] on
/// non-finite labels/weights/exposures, negative weights, non-positive exposure, or
/// a zero denominator.
pub fn exposure_weighted_base_rate(
    y: &[f32],
    weight: &[f32],
    exposure: Option<&[f32]>,
) -> Result<f32, PbError> {
    if weight.len() != y.len() {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "categorical base rate: y={}, weight={}",
                y.len(),
                weight.len()
            ),
        });
    }
    if let Some(e) = exposure {
        if e.len() != y.len() {
            return Err(PbError::ShapeMismatch {
                what: format!("categorical base rate: y={}, exposure={}", y.len(), e.len()),
            });
        }
    }
    let mut sum_wy = 0.0_f64;
    let mut sum_we = 0.0_f64;
    for (i, (&yi, &wi)) in y.iter().zip(weight).enumerate() {
        if !yi.is_finite() {
            return Err(PbError::InvalidInput {
                what: format!("categorical y[{i}] must be finite, got {yi}"),
            });
        }
        if !wi.is_finite() || wi < 0.0 {
            return Err(PbError::InvalidInput {
                what: format!("categorical weight[{i}] must be finite and >= 0, got {wi}"),
            });
        }
        let e = match exposure {
            Some(ex) => {
                let ei = *ex.get(i).ok_or_else(|| PbError::Internal {
                    what: "validated exposure lost a row".into(),
                })?;
                if !ei.is_finite() || ei <= 0.0 {
                    return Err(PbError::InvalidInput {
                        what: format!("categorical exposure[{i}] must be finite and > 0, got {ei}"),
                    });
                }
                f64::from(ei)
            }
            None => 1.0,
        };
        let w = f64::from(wi);
        sum_wy += w * f64::from(yi);
        sum_we += w * e;
    }
    if sum_we <= 0.0 {
        return Err(PbError::InvalidInput {
            what: "categorical base rate denominator Σw·e must be > 0".into(),
        });
    }
    let out = (sum_wy / sum_we) as f32;
    if out.is_finite() {
        Ok(out)
    } else {
        Err(PbError::InvalidInput {
            what: format!(
                "categorical base rate is not representable as f32: {}",
                sum_wy / sum_we
            ),
        })
    }
}

/// Closed-form target-statistic shrinkage (§04.3):
/// `(sum_wy + m·base) / (sum_w + m)`.
///
/// # Errors
/// [`PbError::InvalidInput`] if inputs are non-finite or the denominator is zero;
/// [`PbError::InvalidConfig`] for [`Smooth::Auto`], whose variance estimator belongs
/// to the full encoder-fit path.
pub fn shrunken_encoding(
    sum_wy: f64,
    sum_w: f64,
    base: f32,
    smooth: Smooth,
) -> Result<f32, PbError> {
    if !sum_wy.is_finite() || !sum_w.is_finite() || sum_w < 0.0 || !base.is_finite() {
        return Err(PbError::InvalidInput {
            what: format!(
                "invalid categorical shrinkage inputs: sum_wy={sum_wy}, sum_w={sum_w}, base={base}"
            ),
        });
    }
    let m = match smooth {
        Smooth::Fixed { m } => {
            if !m.is_finite() || m < 0.0 {
                return Err(PbError::InvalidConfig {
                    what: format!("Smooth::Fixed m must be finite and >= 0, got {m}"),
                });
            }
            f64::from(m)
        }
        Smooth::Auto => {
            return Err(PbError::InvalidConfig {
                what: "Smooth::Auto requires the full encoder-fit variance estimator".into(),
            });
        }
    };
    let denom = sum_w + m;
    if denom <= 0.0 {
        return Err(PbError::InvalidInput {
            what: "categorical shrinkage denominator sum_w + m must be > 0".into(),
        });
    }
    let out = ((sum_wy + m * f64::from(base)) / denom) as f32;
    if out.is_finite() {
        Ok(out)
    } else {
        Err(PbError::InvalidInput {
            what: "categorical shrinkage output is not finite".into(),
        })
    }
}

/// Fit a categorical target-statistic encoder (§04.3), returning the frozen
/// full-data serve encoder plus leakage-free per-row training encodings.
///
/// The full-data encoder is deterministic by level label order. Training encodings
/// use the configured leakage scheme: ordered prefix statistics never include the
/// current row, and K-fold encodings exclude the row's whole fold.
///
/// # Errors
/// [`PbError::ShapeMismatch`] on length mismatch; [`PbError::InvalidInput`] on bad
/// row values; [`PbError::InvalidConfig`] on unsupported/invalid config values.
pub fn fit_cat_encoder(
    levels: &[String],
    y: &[f32],
    spec: CatFitSpec<'_>,
) -> Result<(CatEncoder, Vec<f32>), PbError> {
    spec.config.validate()?;
    if levels.len() != y.len() {
        return Err(PbError::ShapeMismatch {
            what: format!("categorical fit: levels={}, y={}", levels.len(), y.len()),
        });
    }
    let weights;
    let w = match spec.weight {
        Some(w) => {
            if w.len() != y.len() {
                return Err(PbError::ShapeMismatch {
                    what: format!("categorical fit: y={}, weight={}", y.len(), w.len()),
                });
            }
            w
        }
        None => {
            weights = vec![1.0_f32; y.len()];
            &weights
        }
    };
    if spec.config.target == CatTarget::Count {
        // Target-free channel (P1 multi-channel spec §3): ignores `y`/leakage/smoothing
        // entirely, so it takes a fully separate path rather than threading a dummy branch
        // through the target-statistic machinery below (which all assume a target). `exposure`
        // IS threaded through — not for the count value itself, but so rare-level pooling
        // matches a paired Mean channel's exactly; see `fit_count_encoder`'s doc.
        return fit_count_encoder(levels, w, spec.exposure, spec.raw, spec.id, spec.config);
    }
    let row_terms = categorical_row_terms(y, w, spec.exposure, spec.config.target)?;
    let (fit_levels, members) =
        collapse_rare_levels(levels, &row_terms, spec.config.min_data_per_group)?;
    // Intern the collapsed per-row labels into integer ids ONCE, so the dominant full-data + k-fold
    // encoders index dense Vecs by id instead of string-keyed BTreeMaps per row (the high-card
    // binning cost). Byte-identical (every reduction stays in row order; Fisher re-sorts by label).
    let (fit_ids, id_labels) = intern_levels(&fit_levels)?;
    let base = categorical_base_from_terms(&row_terms, spec.config.target)?;
    // Low-cardinality direct bypass (see `TsConfig::direct_max_levels`): raw unsmoothed level
    // statistics for both the frozen serve map and the training rows. Binary features are
    // exempt — their partition is encoding-invariant, so the bypass would only re-roll
    // early-stopping paths.
    let direct = id_labels.len() >= 3 && id_labels.len() <= spec.config.direct_max_levels as usize;
    let smooth = if direct {
        Smooth::Fixed { m: 0.0 }
    } else {
        resolve_smooth(&fit_levels, &row_terms, base, spec.config.smooth)?
    };
    let mut resolved_config = spec.config.clone();
    resolved_config.smooth = smooth;
    let full = full_data_encoder(
        spec.raw,
        spec.id,
        &fit_ids,
        &id_labels,
        &row_terms,
        base,
        &resolved_config,
        &members,
    )?;
    let train = if direct {
        // Training rows reuse the frozen full-fit statistics (no cross-fitting): with so few
        // levels the encoding is order-only information, and the cross-fit per-row noise
        // outweighs the self-row leakage it prevents. Aggregation mirrors `full_data_encoder`
        // exactly, so train values are identical to the serve map's.
        let mut agg = vec![CatRowTerm::default(); id_labels.len()];
        for (&lid, term) in fit_ids.iter().zip(&row_terms) {
            let entry = agg.get_mut(lid as usize).ok_or_else(|| PbError::Internal {
                what: "direct-bypass encoder level id escaped".into(),
            })?;
            entry.sum_y += term.sum_y;
            entry.denom += term.denom;
        }
        let mut value_of_id = Vec::with_capacity(id_labels.len());
        for term in &agg {
            // Mirror full_data_encoder's zero-weight-level fallback so the direct-bypass train
            // rows stay identical to the serve map even when a level's total weight is zero.
            value_of_id.push(if term.denom > 0.0 {
                shrunken_encoding(term.sum_y, term.denom, base, smooth)?
            } else {
                base
            });
        }
        fit_ids
            .iter()
            .map(|&lid| {
                value_of_id
                    .get(lid as usize)
                    .copied()
                    .ok_or_else(|| PbError::Internal {
                        what: "direct-bypass train encoding id escaped".into(),
                    })
            })
            .collect::<Result<Vec<f32>, PbError>>()?
    } else {
        match spec.config.leakage {
            // Ordered / LOO are non-default and keep the per-row string path (lower priority).
            LeakageScheme::Ordered { n_perms } => ordered_training_encodings(
                &fit_levels,
                &row_terms,
                base,
                spec.config.smooth,
                spec.seed,
                n_perms,
            )?,
            LeakageScheme::KFold { k } => kfold_training_encodings(
                &fit_ids,
                id_labels.len(),
                &row_terms,
                base,
                spec.config.smooth,
                spec.seed,
                k,
            )?,
            LeakageScheme::LeaveOneOut => {
                loo_training_encodings(&fit_levels, &row_terms, base, spec.config.smooth)?
            }
        }
    };
    Ok((full, train))
}

/// Distinct level count AFTER rare-level pooling (`min_data_per_group`, §04.3), without fitting
/// a full target-statistic encoder — the count-channel cardinality gate (`cat_count_min_levels`,
/// the py binding's `CategoricalColumn` assembly) needs this to decide whether a categorical
/// feature is high-enough-cardinality to warrant a second (count) channel BEFORE committing to
/// fit it, ideally without paying for `fit_cat_encoder`'s Fisher-sort/base/smoothing/training-
/// encoding work just to read off a level count.
///
/// Target-free by construction: [`collapse_rare_levels`] pools purely on each level's
/// weight/exposure denominator (`CatRowTerm::denom`, from [`categorical_row_terms`]), never on
/// `y`'s value — see either function's body — so this needs no real target at all. `target`
/// still selects which denominator convention to replicate (`Mean`'s includes exposure,
/// `LogMean`'s does not; see `categorical_row_terms`), matching whatever the real encoder for
/// this feature would use, so the count reported here is EXACTLY what `fit_cat_encoder` would
/// later produce for the same feature under the same config — not an approximation of it.
///
/// # Errors
/// [`PbError::ShapeMismatch`] if `weight` is present and its length disagrees with `levels`;
/// propagates [`categorical_row_terms`]/[`collapse_rare_levels`] otherwise (e.g. a reserved
/// `RARE_LEVEL_LABEL` collision in the raw data).
pub fn post_pooling_level_count(
    levels: &[String],
    weight: Option<&[f32]>,
    exposure: Option<&[f32]>,
    target: CatTarget,
    min_data_per_group: f32,
) -> Result<usize, PbError> {
    let owned_weights;
    let w: &[f32] = match weight {
        Some(w) => {
            if w.len() != levels.len() {
                return Err(PbError::ShapeMismatch {
                    what: format!(
                        "post_pooling_level_count: levels={}, weight={}",
                        levels.len(),
                        w.len()
                    ),
                });
            }
            w
        }
        None => {
            owned_weights = vec![1.0_f32; levels.len()];
            &owned_weights
        }
    };
    // A dummy strictly-positive `y` is safe here: rarity depends only on `term.denom`, never
    // `term.sum_y` (see this function's doc); `1.0` also satisfies `CatTarget::LogMean`'s
    // `y > 0` row validation, so one dummy value works for either target.
    let dummy_y = vec![1.0_f32; levels.len()];
    let row_terms = categorical_row_terms(&dummy_y, w, exposure, target)?;
    let (_, members) = collapse_rare_levels(levels, &row_terms, min_data_per_group)?;
    Ok(members.len())
}

/// Fit the target-free "count/rarity" channel (P1 multi-channel spec §3): per level,
/// `log1p(Σw_level / Σw_total × N)` where `N` is the total row count — frozen at fit time, no
/// target/smoothing involved. Reuses [`collapse_rare_levels`]'s rare-pooling contract and
/// [`assign_fisher_bins`]'s ordinal binning, so the resulting [`CatEncoder`] has the identical
/// shape as a Mean/LogMean encoder; `base = 0.0` for unseen levels (maximally rare, `log1p(0)`).
/// Training encodings equal the frozen serve value for every row of a level — target-free, so
/// there is no leakage to avoid and no fold/permutation structure to introduce.
///
/// **Rare-pooling uses `Σ(w·e)`, matching [`categorical_row_terms`]'s `CatTarget::Mean` denom
/// exactly** (`exposure` defaults to `1.0` per row when absent) — NOT the `Σw`-alone share
/// formula below. This is load-bearing for the P1 multi-channel design (§4.3): when this
/// channel is paired with a Mean channel over the SAME raw feature, both must partition levels
/// into the SAME rare bucket, or the two channels' frozen level sets diverge and the joint
/// per-level cell space (explain.rs) has no single coherent level to key on. Sharing the exact
/// pooling denom (not just the same `min_data_per_group` threshold) guarantees this whenever an
/// exposure-weighted fit pairs the two — the scenario this feature specifically targets
/// (frequency models). The share VALUE itself still uses `Σw` alone (weight, not exposure) once
/// pooling has already grouped the rows — rarity is a row/policy-count notion, not an
/// exposure-years one, matching the spec's literal `Σw_level/Σw_total` formula.
///
/// # Errors
/// [`PbError::ShapeMismatch`] if `exposure` is present and its length disagrees with `levels`;
/// [`PbError::InvalidInput`] on a non-finite/negative weight, a non-finite/non-positive
/// exposure, if every row has zero weight (no exposure to measure a share against), or a
/// level's computed value is non-finite; [`PbError::Internal`] on an interned-id bookkeeping
/// bug propagated from [`collapse_rare_levels`]/[`intern_levels`]/[`assign_fisher_bins`], or
/// raised directly if a level id somehow escapes this function's own aggregation.
fn fit_count_encoder(
    levels: &[String],
    weight: &[f32],
    exposure: Option<&[f32]>,
    raw: FeatureId,
    id: TsEncodingId,
    config: &TsConfig,
) -> Result<(CatEncoder, Vec<f32>), PbError> {
    if let Some(ex) = exposure {
        if ex.len() != levels.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "categorical count fit: levels={}, exposure={}",
                    levels.len(),
                    ex.len()
                ),
            });
        }
    }
    let n_rows = levels.len() as f64;
    // Pooling term: denom = w·e (mirrors `categorical_row_terms`'s Mean-target denom exactly),
    // used ONLY to decide which levels collapse into the rare bucket (see doc above).
    let mut pooling_rows: Vec<CatRowTerm> = Vec::with_capacity(weight.len());
    for (i, &wi) in weight.iter().enumerate() {
        if !wi.is_finite() || wi < 0.0 {
            return Err(PbError::InvalidInput {
                what: format!("invalid categorical count row {i}: weight={wi}"),
            });
        }
        let e = match exposure {
            Some(ex) => {
                let ei = *ex.get(i).ok_or_else(|| PbError::Internal {
                    what: "validated exposure lost a row".into(),
                })?;
                if !ei.is_finite() || ei <= 0.0 {
                    return Err(PbError::InvalidInput {
                        what: format!(
                            "categorical count exposure[{i}] must be finite and > 0, got {ei}"
                        ),
                    });
                }
                f64::from(ei)
            }
            None => 1.0,
        };
        pooling_rows.push(CatRowTerm {
            sum_y: 0.0,
            denom: f64::from(wi) * e,
        });
    }
    let (fit_levels, members) =
        collapse_rare_levels(levels, &pooling_rows, config.min_data_per_group)?;
    let (fit_ids, id_labels) = intern_levels(&fit_levels)?;

    // Share value: Σw alone (NOT Σw·e) over the already-pooled levels — see doc above.
    let mut level_weight = vec![0.0_f64; id_labels.len()];
    let mut total_weight = 0.0_f64;
    for (&lid, &wi) in fit_ids.iter().zip(weight) {
        let entry = level_weight
            .get_mut(lid as usize)
            .ok_or_else(|| PbError::Internal {
                what: "count-channel level id escaped".into(),
            })?;
        let w = f64::from(wi);
        *entry += w;
        total_weight += w;
    }
    if total_weight <= 0.0 {
        return Err(PbError::InvalidInput {
            what: "categorical count channel requires positive total weight".into(),
        });
    }

    let mut value_of_id = Vec::with_capacity(id_labels.len());
    let mut out_levels = Vec::with_capacity(id_labels.len());
    for (lid, &lw) in level_weight.iter().enumerate() {
        let label = *id_labels.get(lid).ok_or_else(|| PbError::Internal {
            what: "count-channel id label escaped".into(),
        })?;
        let share = lw / total_weight;
        let encoding = (share * n_rows).ln_1p() as f32;
        if !encoding.is_finite() {
            return Err(PbError::InvalidInput {
                what: format!(
                    "categorical count channel produced a non-finite value for `{label}`"
                ),
            });
        }
        value_of_id.push(encoding);
        out_levels.push(CatLevel {
            members: members
                .get(label)
                .cloned()
                .unwrap_or_else(|| vec![label.to_owned()]),
            label: label.to_owned(),
            encoding,
            bin: 0,
            weight: lw as f32,
        });
    }
    assign_fisher_bins(&mut out_levels)?;

    let train = fit_ids
        .iter()
        .map(|&lid| {
            value_of_id
                .get(lid as usize)
                .copied()
                .ok_or_else(|| PbError::Internal {
                    what: "count-channel train encoding id escaped".into(),
                })
        })
        .collect::<Result<Vec<f32>, PbError>>()?;

    Ok((
        CatEncoder {
            raw,
            id,
            levels: out_levels,
            base: 0.0,
            config: config.clone(),
        },
        train,
    ))
}

/// Intern per-row (collapsed) labels into integer ids in FIRST-SEEN / ROW order, returning the
/// per-row id vector and the id→label table. The `HashMap` is a local lookup only (never iterated,
/// never serialized) so the result is deterministic by row order. Lets the full-data + k-fold
/// encoders index dense `Vec`s by id — byte-identical to the string-keyed path.
fn intern_levels(levels: &[String]) -> Result<(Vec<u32>, Vec<&str>), PbError> {
    let mut id_of: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    let mut id_labels: Vec<&str> = Vec::new();
    let mut ids: Vec<u32> = Vec::with_capacity(levels.len());
    for label in levels {
        let id = match id_of.get(label.as_str()) {
            Some(&id) => id,
            None => {
                let id = u32::try_from(id_labels.len()).map_err(|_| PbError::InvalidInput {
                    what: "categorical fit supports at most u32::MAX distinct levels".into(),
                })?;
                id_labels.push(label.as_str());
                id_of.insert(label.as_str(), id);
                id
            }
        };
        ids.push(id);
    }
    Ok((ids, id_labels))
}

#[derive(Debug, Clone, Copy, Default)]
struct CatRowTerm {
    sum_y: f64,
    denom: f64,
}

fn categorical_row_terms(
    y: &[f32],
    weight: &[f32],
    exposure: Option<&[f32]>,
    target: CatTarget,
) -> Result<Vec<CatRowTerm>, PbError> {
    if let Some(ex) = exposure {
        if ex.len() != y.len() {
            return Err(PbError::ShapeMismatch {
                what: format!("categorical fit: y={}, exposure={}", y.len(), ex.len()),
            });
        }
    }
    let mut out = Vec::with_capacity(y.len());
    for (i, (&yi, &wi)) in y.iter().zip(weight).enumerate() {
        if !yi.is_finite() || !wi.is_finite() || wi < 0.0 {
            return Err(PbError::InvalidInput {
                what: format!("invalid categorical row {i}: y={yi}, weight={wi}"),
            });
        }
        let e = match exposure {
            Some(ex) => {
                let ei = *ex.get(i).ok_or_else(|| PbError::Internal {
                    what: "validated exposure lost a row".into(),
                })?;
                if !ei.is_finite() || ei <= 0.0 {
                    return Err(PbError::InvalidInput {
                        what: format!("categorical exposure[{i}] must be finite and > 0, got {ei}"),
                    });
                }
                f64::from(ei)
            }
            None => 1.0,
        };
        let w = f64::from(wi);
        let (sum_y, denom) = match target {
            CatTarget::Mean => (w * f64::from(yi), w * e),
            CatTarget::LogMean => {
                if yi <= 0.0 {
                    return Err(PbError::InvalidInput {
                        what: format!("cat_target='log_mean' requires y[{i}] > 0, got {yi}"),
                    });
                }
                (w * f64::from(yi).ln(), w)
            }
            // P3 per-class frequency channel: the ONLY difference from `Mean` is the row
            // target transform `y -> 1[y == class]`. The denominator stays `w·e` so this
            // channel's rare-level pooling is identical to a paired `Mean`/`Count` channel's
            // (required by `JointCatAxis::build`). Labels arrive as exact small integers in
            // `f32` (`multiclass_labels` rejects anything else upstream), so the equality
            // compare against `class as f32` is exact, not a float-tolerance question.
            CatTarget::ClassFreq { class } => {
                let hit = f64::from(yi) == f64::from(class);
                (w * if hit { 1.0 } else { 0.0 }, w * e)
            }
            CatTarget::Count => {
                // `fit_cat_encoder` intercepts `CatTarget::Count` before it ever reaches this
                // target-statistic row-term path (the count channel is target-free and has its
                // own `fit_count_encoder`); reaching here is an internal routing bug, not a
                // reachable user input.
                return Err(PbError::Internal {
                    what: "categorical_row_terms called with target-free CatTarget::Count".into(),
                });
            }
        };
        out.push(CatRowTerm { sum_y, denom });
    }
    Ok(out)
}

fn categorical_base_from_terms(rows: &[CatRowTerm], target: CatTarget) -> Result<f32, PbError> {
    let mut sum_y = 0.0_f64;
    let mut denom = 0.0_f64;
    for term in rows {
        sum_y += term.sum_y;
        denom += term.denom;
    }
    if denom <= 0.0 {
        let label = match target {
            CatTarget::Mean => "categorical base rate denominator Σw·e",
            CatTarget::LogMean => "categorical log-mean denominator Σw",
            CatTarget::ClassFreq { .. } => "categorical class-frequency denominator Σw·e",
            // Unreachable in practice (see `categorical_row_terms`'s `Count` arm) — kept as a
            // descriptive label rather than an early error so this match stays a pure,
            // total string choice like its two siblings.
            CatTarget::Count => "categorical count channel denominator Σw",
        };
        return Err(PbError::InvalidInput {
            what: format!("{label} must be > 0"),
        });
    }
    let out = (sum_y / denom) as f32;
    if out.is_finite() {
        Ok(out)
    } else {
        Err(PbError::InvalidInput {
            what: format!(
                "categorical base target statistic is not representable as f32: {}",
                sum_y / denom
            ),
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn full_data_encoder(
    raw: FeatureId,
    id: TsEncodingId,
    fit_ids: &[u32],
    id_labels: &[&str],
    rows: &[CatRowTerm],
    base: f32,
    config: &TsConfig,
    members: &BTreeMap<String, Vec<String>>,
) -> Result<CatEncoder, PbError> {
    // Aggregate (sum_y, denom) per level id by walking rows in order — same row order, hence the
    // same f64 sums, as the old string-keyed BTreeMap. `out` is built in id order, but
    // `assign_fisher_bins` re-sorts by (encoding, label) (labels distinct ⇒ total order), so the
    // final encoder is byte-identical regardless of the build order.
    let mut agg = vec![CatRowTerm::default(); id_labels.len()];
    for (&lid, term) in fit_ids.iter().zip(rows) {
        let entry = agg.get_mut(lid as usize).ok_or_else(|| PbError::Internal {
            what: "full-data encoder level id escaped".into(),
        })?;
        entry.sum_y += term.sum_y;
        entry.denom += term.denom;
    }
    let mut out = Vec::with_capacity(id_labels.len());
    for (lid, term) in agg.iter().enumerate() {
        let label = *id_labels.get(lid).ok_or_else(|| PbError::Internal {
            what: "full-data encoder id label escaped".into(),
        })?;
        // A level can aggregate to zero total weight (every row weight 0, e.g. an excluded
        // code); fall back to `base` instead of handing shrunken_encoding a zero denominator,
        // mirroring the kfold/loo/ordered cross-fit guards below.
        let encoding = if term.denom > 0.0 {
            shrunken_encoding(term.sum_y, term.denom, base, config.smooth)?
        } else {
            base
        };
        out.push(CatLevel {
            members: members
                .get(label)
                .cloned()
                .unwrap_or_else(|| vec![label.to_owned()]),
            label: label.to_owned(),
            encoding,
            bin: 0,
            weight: term.denom as f32,
        });
    }
    assign_fisher_bins(&mut out)?;
    Ok(CatEncoder {
        raw,
        id,
        levels: out,
        base,
        config: config.clone(),
    })
}

fn collapse_rare_levels(
    levels: &[String],
    rows: &[CatRowTerm],
    min_data_per_group: f32,
) -> Result<(Vec<String>, RareMembers), PbError> {
    let mut agg: BTreeMap<&str, f64> = BTreeMap::new();
    for (label, term) in levels.iter().zip(rows) {
        let entry = agg.entry(label.as_str()).or_default();
        *entry += term.denom;
    }

    if min_data_per_group <= 0.0 {
        let mut members = BTreeMap::new();
        for label in agg.keys() {
            members.insert((*label).to_owned(), vec![(*label).to_owned()]);
        }
        return Ok((levels.to_vec(), members));
    }

    if agg.contains_key(RARE_LEVEL_LABEL) {
        return Err(PbError::InvalidInput {
            what: format!("categorical label `{RARE_LEVEL_LABEL}` is reserved for rare buckets"),
        });
    }

    let min = f64::from(min_data_per_group);
    let mut rare_members = Vec::new();
    let mut members = BTreeMap::new();
    for (label, denom) in &agg {
        if *denom < min {
            rare_members.push((*label).to_owned());
        } else {
            members.insert((*label).to_owned(), vec![(*label).to_owned()]);
        }
    }
    if !rare_members.is_empty() {
        members.insert(RARE_LEVEL_LABEL.to_owned(), rare_members.clone());
    }

    // Membership test against a set (O(1)/row) instead of `rare_members.iter().any()` (O(rare)/row,
    // i.e. O(n·rare) total on high-cardinality columns). Byte-identical: same partition into rare
    // (→ RARE_LEVEL_LABEL) vs kept (→ label).
    let rare_set: std::collections::HashSet<&str> =
        rare_members.iter().map(String::as_str).collect();
    let collapsed = levels
        .iter()
        .map(|label| {
            if rare_set.contains(label.as_str()) {
                RARE_LEVEL_LABEL.to_owned()
            } else {
                label.clone()
            }
        })
        .collect();
    Ok((collapsed, members))
}

fn resolve_smooth(
    levels: &[String],
    rows: &[CatRowTerm],
    base: f32,
    smooth: Smooth,
) -> Result<Smooth, PbError> {
    match smooth {
        Smooth::Fixed { m } => Ok(Smooth::Fixed { m }),
        Smooth::Auto => {
            let m = auto_smooth_strength(levels, rows, base)?;
            Ok(Smooth::Fixed { m })
        }
    }
}

/// Per-level second moments for the Auto-m estimator: level weights/means (positive-weight
/// levels only, label order) and the row-level `within` dispersion (the per-unit-weight noise
/// variance scale sigma^2, so Var(level mean) ~ within/w). `between` vs base is not tracked here:
/// the credible-set estimator re-derives each level's deviation from `mean`/`base` directly
/// (its z-scores are per-level, not pooled), and its degenerate branch no longer consults a
/// pooled between-variance (see `auto_smooth_strength`).
struct AutoMoments {
    w: Vec<f64>,
    mean: Vec<f64>,
    within: f64,
}

fn auto_moments(levels: &[String], rows: &[CatRowTerm]) -> Result<AutoMoments, PbError> {
    let mut agg: BTreeMap<&str, CatRowTerm> = BTreeMap::new();
    for (label, term) in levels.iter().zip(rows) {
        let entry = agg.entry(label.as_str()).or_default();
        entry.sum_y += term.sum_y;
        entry.denom += term.denom;
    }
    let mut w = Vec::with_capacity(agg.len());
    let mut mean = Vec::with_capacity(agg.len());
    let mut total = 0.0_f64;
    for term in agg.values() {
        if term.denom > 0.0 {
            total += term.denom;
            w.push(term.denom);
            mean.push(term.sum_y / term.denom);
        }
    }
    if total <= 0.0 {
        return Err(PbError::InvalidInput {
            what: "Smooth::Auto requires positive categorical weight".into(),
        });
    }
    let mut within = 0.0_f64;
    for (label, term) in levels.iter().zip(rows) {
        if term.denom > 0.0 {
            let level = agg.get(label.as_str()).ok_or_else(|| PbError::Internal {
                what: "auto-smooth level missing from aggregate".into(),
            })?;
            let level_mean = level.sum_y / level.denom;
            let row_rate = term.sum_y / term.denom;
            let diff = row_rate - level_mean;
            within += term.denom * diff * diff;
        }
    }
    // Degrees-of-freedom correction: each of the L positive-weight levels' own mean is
    // estimated from the same rows the deviations are measured against, so the raw sum
    // has expectation sigma^2 * (total - L), not sigma^2 * total (the standard ANOVA
    // within-group dof loss, one per level). Dividing by `total` alone underestimates
    // sigma^2 by a factor of total / (total - L) — material on high-cardinality columns
    // with small levels, where the credible-set estimator (`auto_smooth_strength`) then
    // inflates z^2 by that same factor and under-shrinks noise it was designed to catch.
    let l_effective = w.len() as f64;
    within /= (total - l_effective).max(MIN_AUTO_VARIANCE);
    Ok(AutoMoments { w, mean, within })
}

fn finite_m(m: f64) -> Result<f32, PbError> {
    let out = m as f32;
    if out.is_finite() && out >= 0.0 {
        Ok(out)
    } else {
        Err(PbError::InvalidInput {
            what: "Smooth::Auto produced a non-finite shrinkage strength".into(),
        })
    }
}

/// z^2 threshold for the credible-set estimator (3 sigma).
const CRED_Z2_THRESHOLD: f64 = 9.0;
/// E[(z^2 - 1) * 1{z^2 > 9}] under N(0,1): the expected per-level null leakage above the
/// threshold, subtracted (times L) so an all-null feature resolves to tau^2 <= 0 -> MAX.
const CRED_NULL_KAPPA: f64 = 0.026_60;

/// Credible-set Auto-m (the shipped `Smooth::Auto` estimator since the 2026-07-11 redesign):
/// estimate tau^2 only from levels whose deviation clears 3 sigma of their own sampling noise
/// (z_l^2 = w_l (mean_l-base)^2 / s2 > 9), null-corrected by the Gaussian truncation
/// expectation, mass-weighted within the credible set:
/// `tau2 = s2 * (sum_S (z^2-1) - kappa*L) / sum_S w`. No credible signal -> MAX (full shrink).
///
/// Replaces the pooled variance ratio, whose mass-weighted between-variance is diluted by
/// junk-level mass on high-cardinality cats (m 5-15x too big; well-populated minority levels
/// over-shrunk). 9-split gate: freMTPL2freq dev -0.062% (8/9) gini +0.0011, brvehins1 dev
/// -0.093% (5/9) gini +0.0052, beMTPL16 statistical tie with better per-level calibration on
/// all three cats; binary-only fits bit-identical via the exemption below. Falsified
/// alternatives (Buhlmann-Straub noise subtraction, global Q gate, fixed-point MSE m) in
/// speed_accuracy_work.md.
fn auto_smooth_strength(levels: &[String], rows: &[CatRowTerm], base: f32) -> Result<f32, PbError> {
    let mo = auto_moments(levels, rows)?;
    // BINARY features keep the variance-ratio estimator: with 2 levels the partition is
    // encoding-invariant, so the m choice cannot improve placement — it only re-rolls
    // cross-fit noise paths (measured on ohlsson_sev: 2/9 wins, +0.31% dev under the
    // credible-set m). Mirrors the direct-bypass binary exemption.
    if mo.w.len() <= 2 {
        return variance_ratio_smooth_strength(levels, rows, base);
    }
    let s2 = mo.within;
    if s2 <= MIN_AUTO_VARIANCE {
        // No row-level noise scale: the level means are exact, so there is nothing to shrink
        // toward base regardless of `between`. Matches the legacy ratio (cat.rs
        // variance_ratio_smooth_strength): within/between -> 0 whenever between >
        // MIN_AUTO_VARIANCE, and its 0.0 default when both moments are degenerate.
        return finite_m(0.0);
    }
    let l_count = mo.w.len() as f64;
    let mut zsum = 0.0_f64;
    let mut wsum = 0.0_f64;
    for (&w, &mean) in mo.w.iter().zip(&mo.mean) {
        let diff = mean - f64::from(base);
        let z2 = w * diff * diff / s2;
        if z2 > CRED_Z2_THRESHOLD {
            zsum += z2 - 1.0;
            wsum += w;
        }
    }
    let num = s2 * (zsum - CRED_NULL_KAPPA * l_count);
    if wsum <= 0.0 || num <= 0.0 {
        return finite_m(MAX_AUTO_SMOOTH);
    }
    let tau2 = num / wsum;
    if tau2 <= MIN_AUTO_VARIANCE {
        return finite_m(MAX_AUTO_SMOOTH);
    }
    finite_m((s2 / tau2).min(MAX_AUTO_SMOOTH))
}

/// The pre-2026-07-11 `Smooth::Auto` estimator: pooled within/between variance ratio.
/// Retained as the BINARY path of [`auto_smooth_strength`] (see the exemption there); no
/// longer used for 3+-level features, where junk-mass dilution of `between` over-shrinks
/// well-populated minority levels.
fn variance_ratio_smooth_strength(
    levels: &[String],
    rows: &[CatRowTerm],
    base: f32,
) -> Result<f32, PbError> {
    let mut agg: BTreeMap<&str, CatRowTerm> = BTreeMap::new();
    for (label, term) in levels.iter().zip(rows) {
        let entry = agg.entry(label.as_str()).or_default();
        entry.sum_y += term.sum_y;
        entry.denom += term.denom;
    }
    let mut total = 0.0_f64;
    let mut between = 0.0_f64;
    for term in agg.values() {
        if term.denom > 0.0 {
            let mean = term.sum_y / term.denom;
            let diff = mean - f64::from(base);
            between += term.denom * diff * diff;
            total += term.denom;
        }
    }
    if total <= 0.0 {
        return Err(PbError::InvalidInput {
            what: "Smooth::Auto requires positive categorical weight".into(),
        });
    }
    between /= total;

    let mut within = 0.0_f64;
    for (label, term) in levels.iter().zip(rows) {
        if term.denom > 0.0 {
            let level = agg.get(label.as_str()).ok_or_else(|| PbError::Internal {
                what: "auto-smooth level missing from aggregate".into(),
            })?;
            let level_mean = level.sum_y / level.denom;
            let row_rate = term.sum_y / term.denom;
            let diff = row_rate - level_mean;
            within += term.denom * diff * diff;
        }
    }
    within /= total;

    let m = if between <= MIN_AUTO_VARIANCE {
        if within <= MIN_AUTO_VARIANCE {
            0.0
        } else {
            MAX_AUTO_SMOOTH
        }
    } else {
        (within / between).min(MAX_AUTO_SMOOTH)
    };
    let out = m as f32;
    if out.is_finite() && out >= 0.0 {
        Ok(out)
    } else {
        Err(PbError::InvalidInput {
            what: "Smooth::Auto produced a non-finite shrinkage strength".into(),
        })
    }
}

fn assign_fisher_bins(levels: &mut [CatLevel]) -> Result<(), PbError> {
    levels.sort_by(|a, b| {
        a.encoding
            .total_cmp(&b.encoding)
            .then_with(|| a.label.cmp(&b.label))
    });
    let mut distinct = Vec::<f32>::new();
    for level in levels.iter() {
        if !level.encoding.is_finite() {
            return Err(PbError::InvalidInput {
                what: format!(
                    "categorical level `{}` has non-finite encoding",
                    level.label
                ),
            });
        }
        if distinct.last() != Some(&level.encoding) {
            distinct.push(level.encoding);
        }
    }
    let n_distinct = distinct.len();
    if n_distinct == 0 {
        return Ok(());
    }
    let n_bins = n_distinct.clamp(1, MAX_CAT_BINS);
    let mut rank = 0usize;
    let mut prev_encoding: Option<f32> = None;
    for level in levels.iter_mut() {
        if let Some(prev) = prev_encoding {
            if level.encoding != prev {
                rank = rank.checked_add(1).ok_or_else(|| PbError::Internal {
                    what: "categorical rank overflow".into(),
                })?;
            }
        }
        // `rank` can reach `n_distinct - 1` and `n_distinct` can be up to `u32::MAX` (interned
        // level count), so on a 32-bit target `rank * n_bins` can overflow `usize` — checked,
        // like the `rank` increment above.
        let scaled = rank.checked_mul(n_bins).ok_or_else(|| PbError::Internal {
            what: "categorical bin rank*n_bins overflow".into(),
        })?;
        let bin = 1 + (scaled / n_distinct);
        level.bin = u8::try_from(bin).map_err(|_| PbError::Internal {
            what: "categorical Fisher bin exceeded u8".into(),
        })?;
        prev_encoding = Some(level.encoding);
    }
    Ok(())
}

/// Sufficient moments for a training-only categorical prior and Auto strength.
#[derive(Clone, Copy, Default)]
struct TrainingMoment {
    weight: f64,
    mean: f64,
    within: f64,
}

impl TrainingMoment {
    fn merge(self, other: Self) -> Self {
        let weight = self.weight + other.weight;
        if weight <= 0.0 {
            return Self::default();
        }
        let delta = other.mean - self.mean;
        Self {
            weight,
            mean: self.mean + delta * other.weight / weight,
            within: self.within
                + other.within
                + delta * delta * self.weight * other.weight / weight,
        }
    }

    fn add(&mut self, row: CatRowTerm) {
        if row.denom > 0.0 {
            *self = self.merge(Self {
                weight: row.denom,
                mean: row.sum_y / row.denom,
                within: 0.0,
            });
        }
    }
}

fn training_prior(moments: &[TrainingMoment], smooth: Smooth) -> Result<(f32, Smooth), PbError> {
    let positive: Vec<_> = moments.iter().filter(|m| m.weight > 0.0).collect();
    let global = positive
        .iter()
        .fold(TrainingMoment::default(), |all, m| all.merge(**m));
    if global.weight <= 0.0 {
        // No allowed target yet (e.g. the first ordered row). A fixed neutral
        // target-scale prior is the only value independent of withheld targets.
        return Ok((0.0, Smooth::Fixed { m: 0.0 }));
    }
    let base = global.mean as f32;
    if matches!(smooth, Smooth::Fixed { .. }) {
        return Ok((base, smooth));
    }
    let sum_within: f64 = positive.iter().map(|m| m.within).sum();
    let strength = if positive.len() <= 2 {
        let within = sum_within / global.weight;
        let between = positive
            .iter()
            .map(|m| m.weight * (m.mean - f64::from(base)).powi(2))
            .sum::<f64>()
            / global.weight;
        if between <= MIN_AUTO_VARIANCE {
            if within <= MIN_AUTO_VARIANCE {
                0.0
            } else {
                MAX_AUTO_SMOOTH
            }
        } else {
            (within / between).min(MAX_AUTO_SMOOTH)
        }
    } else {
        let within = sum_within / (global.weight - positive.len() as f64).max(MIN_AUTO_VARIANCE);
        if within <= MIN_AUTO_VARIANCE {
            0.0
        } else {
            let (mut zsum, mut wsum) = (0.0, 0.0);
            for moment in &positive {
                let z2 = moment.weight * (moment.mean - f64::from(base)).powi(2) / within;
                if z2 > CRED_Z2_THRESHOLD {
                    zsum += z2 - 1.0;
                    wsum += moment.weight;
                }
            }
            let numerator = within * (zsum - CRED_NULL_KAPPA * positive.len() as f64);
            if wsum <= 0.0 || numerator / wsum <= MIN_AUTO_VARIANCE {
                MAX_AUTO_SMOOTH
            } else {
                (within * wsum / numerator).min(MAX_AUTO_SMOOTH)
            }
        }
    };
    Ok((
        base,
        Smooth::Fixed {
            m: finite_m(strength)?,
        },
    ))
}

fn training_value(moment: TrainingMoment, base: f32, smooth: Smooth) -> Result<f32, PbError> {
    if moment.weight <= 0.0 {
        Ok(base)
    } else {
        shrunken_encoding(moment.mean * moment.weight, moment.weight, base, smooth)
    }
}

fn ordered_training_encodings(
    levels: &[String],
    rows: &[CatRowTerm],
    _base: f32,
    smooth: Smooth,
    seed: u64,
    n_perms: u32,
) -> Result<Vec<f32>, PbError> {
    if n_perms == 0 {
        return Err(PbError::InvalidConfig {
            what: "Ordered target statistics require n_perms > 0".into(),
        });
    }
    let (ids, labels) = intern_levels(levels)?;
    let mut out = vec![0.0_f64; rows.len()];
    for perm in 0..n_perms {
        let mut order: Vec<_> = (0..rows.len())
            .map(|row| {
                let id = u32::try_from(row).map_err(|_| PbError::InvalidInput {
                    what: "too many categorical rows".into(),
                })?;
                Ok((pb_seed(seed, perm, Stage::Categorical as u32, id), row))
            })
            .collect::<Result<_, PbError>>()?;
        order.sort_unstable();
        let mut prefix = vec![TrainingMoment::default(); labels.len()];
        let mut prior = TrainingPriorIndex::new(labels.len())?;
        for (_, row) in order {
            let id = *ids.get(row).ok_or_else(|| PbError::Internal {
                what: "ordered id escaped".into(),
            })? as usize;
            let (base, local_smooth) = prior.prior(smooth)?;
            let moment = prefix.get_mut(id).ok_or_else(|| PbError::Internal {
                what: "ordered moment escaped".into(),
            })?;
            let encoded = training_value(*moment, base, local_smooth)?;
            *out.get_mut(row).ok_or_else(|| PbError::Internal {
                what: "ordered output escaped".into(),
            })? += f64::from(encoded);
            moment.add(*rows.get(row).ok_or_else(|| PbError::Internal {
                what: "ordered term escaped".into(),
            })?);
            prior.set(id, *moment)?;
        }
    }
    Ok(out
        .into_iter()
        .map(|value| (value / f64::from(n_perms)) as f32)
        .collect())
}

fn loo_training_encodings(
    levels: &[String],
    rows: &[CatRowTerm],
    _base: f32,
    smooth: Smooth,
) -> Result<Vec<f32>, PbError> {
    let (ids, labels) = intern_levels(levels)?;
    let mut all = vec![TrainingMoment::default(); labels.len()];
    let mut prefixes = Vec::with_capacity(rows.len());
    for (&id, &term) in ids.iter().zip(rows) {
        let moment = all.get_mut(id as usize).ok_or_else(|| PbError::Internal {
            what: "LOO moment escaped".into(),
        })?;
        prefixes.push(*moment);
        moment.add(term);
    }
    let mut suffix = vec![TrainingMoment::default(); labels.len()];
    let mut excluded = vec![TrainingMoment::default(); rows.len()];
    for (row, (&id, &term)) in ids.iter().zip(rows).enumerate().rev() {
        let moment = suffix
            .get_mut(id as usize)
            .ok_or_else(|| PbError::Internal {
                what: "LOO suffix escaped".into(),
            })?;
        *excluded.get_mut(row).ok_or_else(|| PbError::Internal {
            what: "LOO excluded escaped".into(),
        })? = prefixes
            .get(row)
            .copied()
            .unwrap_or_default()
            .merge(*moment);
        moment.add(term);
    }
    let mut prior = TrainingPriorIndex::new(labels.len())?;
    for (id, moment) in all.iter().enumerate() {
        prior.set(id, *moment)?;
    }
    let mut out = Vec::with_capacity(rows.len());
    for (row, &id) in ids.iter().enumerate() {
        let held = excluded.get(row).copied().unwrap_or_default();
        let saved = *all.get(id as usize).ok_or_else(|| PbError::Internal {
            what: "LOO level escaped".into(),
        })?;
        prior.set(id as usize, held)?;
        let (base, local_smooth) = prior.prior(smooth)?;
        out.push(training_value(held, base, local_smooth)?);
        prior.set(id as usize, saved)?;
    }
    Ok(out)
}

fn kfold_training_encodings(
    fit_ids: &[u32],
    n_ids: usize,
    rows: &[CatRowTerm],
    _base: f32,
    smooth: Smooth,
    seed: u64,
    k: u32,
) -> Result<Vec<f32>, PbError> {
    if k < 2 {
        return Err(PbError::InvalidConfig {
            what: "KFold target statistics require k >= 2".into(),
        });
    }
    let folds: Vec<_> = (0..rows.len())
        .map(|row| {
            let id = u32::try_from(row).map_err(|_| PbError::InvalidInput {
                what: "too many categorical rows".into(),
            })?;
            Ok((pb_seed(seed, 0, Stage::Categorical as u32, id) % u64::from(k)) as u32)
        })
        .collect::<Result<_, PbError>>()?;
    let mut out = vec![0.0; rows.len()];
    // Compute each complement directly: no subtraction of held-out targets,
    // and no held-out targets in either the prior or automatic strength.
    for fold in 0..k {
        let mut moments = vec![TrainingMoment::default(); n_ids];
        for ((&id, &term), &row_fold) in fit_ids.iter().zip(rows).zip(&folds) {
            if row_fold != fold {
                moments
                    .get_mut(id as usize)
                    .ok_or_else(|| PbError::Internal {
                        what: "KFold moment escaped".into(),
                    })?
                    .add(term);
            }
        }
        let (base, local_smooth) = training_prior(&moments, smooth)?;
        let values = moments
            .into_iter()
            .map(|moment| training_value(moment, base, local_smooth))
            .collect::<Result<Vec<_>, _>>()?;
        for (row, (&id, &row_fold)) in fit_ids.iter().zip(&folds).enumerate() {
            if row_fold == fold {
                *out.get_mut(row).ok_or_else(|| PbError::Internal {
                    what: "KFold output escaped".into(),
                })? = *values.get(id as usize).ok_or_else(|| PbError::Internal {
                    what: "KFold value escaped".into(),
                })?;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

    use super::*;

    /// The joint axis's per-channel lookup returns exactly `encode_label`: a level's own label,
    /// a pooled member, the FIRST claiming level on a duplicate, and `base` for an unseen label.
    #[test]
    fn first_match_encodings_agree_with_encode_label() {
        let level = |label: &str, members: &[&str], encoding: f32| CatLevel {
            label: label.into(),
            members: members.iter().map(|m| (*m).to_string()).collect(),
            encoding,
            bin: 0,
            weight: 1.0,
        };
        let enc = CatEncoder {
            raw: FeatureId(0),
            id: TsEncodingId(0),
            levels: vec![
                level("a", &["a"], 1.5),
                level("__rare__", &["r1", "r2", "b"], -0.25),
                level("b", &["b"], 7.0),
                level("r2", &["r2"], 3.0),
            ],
            base: 0.125,
            config: TsConfig::default(),
        };
        let map = first_match_encodings(&enc);
        for label in ["a", "__rare__", "r1", "r2", "b", "unseen", ""] {
            let got = map.get(label).copied().unwrap_or(enc.base);
            assert_eq!(
                got.to_bits(),
                enc.encode_label(label).to_bits(),
                "label {label:?}"
            );
        }
    }

    fn encoder(id: u8, raw: u32, label: &str) -> CatEncoder {
        CatEncoder {
            raw: FeatureId(raw),
            id: TsEncodingId(id),
            levels: vec![CatLevel {
                label: label.into(),
                members: vec![label.into()],
                encoding: 1.25,
                bin: 1,
                weight: 10.0,
            }],
            base: 0.5,
            config: TsConfig::default(),
        }
    }

    #[test]
    fn store_lookup_is_total_and_raw_aware() {
        let store = CatEncoderStore::from_encoders(vec![encoder(0, 1, "a"), encoder(0, 2, "b")]);
        assert_eq!(store.len(), 2);
        assert_eq!(
            store.get(TsEncodingId(0), FeatureId(1)).unwrap().levels[0].label,
            "a"
        );
        assert_eq!(
            store.get(TsEncodingId(0), FeatureId(2)).unwrap().levels[0].label,
            "b"
        );
        assert!(matches!(
            store.get(TsEncodingId(1), FeatureId(1)),
            Err(PbError::Internal { .. })
        ));
    }

    #[test]
    fn non_empty_store_round_trips_byte_stably() {
        let store = CatEncoderStore::from_encoders(vec![encoder(0, 1, "a"), encoder(1, 1, "rare")]);
        let cfg = bincode::config::standard();
        let a = bincode::serde::encode_to_vec(&store, cfg).unwrap();
        let b = bincode::serde::encode_to_vec(&store, cfg).unwrap();
        assert_eq!(a, b);
        let (decoded, len): (CatEncoderStore, usize) =
            bincode::serde::decode_from_slice(&a, cfg).unwrap();
        assert_eq!(len, a.len());
        assert_eq!(decoded, store);
    }

    #[test]
    fn ts_config_validates_fail_closed() {
        assert!(TsConfig::default().validate().is_ok());
        assert!(matches!(
            TsConfig {
                leakage: LeakageScheme::Ordered { n_perms: 0 },
                ..TsConfig::default()
            }
            .validate(),
            Err(PbError::InvalidConfig { .. })
        ));
        assert!(matches!(
            TsConfig {
                leakage: LeakageScheme::KFold { k: 1 },
                ..TsConfig::default()
            }
            .validate(),
            Err(PbError::InvalidConfig { .. })
        ));
        assert!(matches!(
            TsConfig {
                smooth: Smooth::Fixed { m: f32::NAN },
                ..TsConfig::default()
            }
            .validate(),
            Err(PbError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn base_rate_matches_exposure_weighted_closed_form() {
        let y = [2.0_f32, 8.0, 10.0];
        let w = [1.0_f32, 2.0, 1.0];
        let e = [1.0_f32, 2.0, 4.0];
        let got = exposure_weighted_base_rate(&y, &w, Some(&e)).unwrap();
        let want = (1.0 * 2.0 + 2.0 * 8.0 + 1.0 * 10.0) / (1.0 * 1.0 + 2.0 * 2.0 + 1.0 * 4.0);
        assert!((got - want).abs() < 1e-6);
        assert!(matches!(
            exposure_weighted_base_rate(&y, &[0.0, 0.0, 0.0], None),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn shrinkage_matches_closed_form_and_auto_fails_closed() {
        let got = shrunken_encoding(30.0, 3.0, 5.0, Smooth::Fixed { m: 2.0 }).unwrap();
        let want = (30.0 + 2.0 * 5.0) / (3.0 + 2.0);
        assert!((got - want).abs() < 1e-6);
        assert_eq!(
            shrunken_encoding(0.0, 0.0, 7.0, Smooth::Fixed { m: 4.0 }).unwrap(),
            7.0
        );
        assert!(matches!(
            shrunken_encoding(1.0, 1.0, 0.0, Smooth::Auto),
            Err(PbError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn smooth_auto_resolves_to_closed_form_variance_ratio() {
        // BINARY fixture: Auto resolves through the binary exemption, i.e. the legacy
        // within/between variance ratio (3+-level features use the credible-set estimator).
        let levels = vec!["a", "a", "b", "b"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = [0.0_f32, 2.0, 4.0, 8.0];
        let cfg = TsConfig {
            leakage: LeakageScheme::KFold { k: 2 },
            smooth: Smooth::Auto,
            min_data_per_group: 0.0,
            direct_max_levels: 0, // this test exercises Auto resolution — keep the bypass out
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(5),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 3,
        };
        let (enc, train) = fit_cat_encoder(&levels, &y, spec).unwrap();
        assert!(train.iter().all(|v| v.is_finite()));
        let resolved_m = match enc.config.smooth {
            Smooth::Fixed { m } => m,
            Smooth::Auto => f32::NAN,
        };
        assert!((resolved_m - 0.4).abs() < 1e-6);
        assert!((enc.encode_label("a") - (2.0 + 0.4 * 3.5) / 2.4).abs() < 1e-6);
        assert!((enc.encode_label("b") - (12.0 + 0.4 * 3.5) / 2.4).abs() < 1e-6);
    }

    /// Junk-heavy mixture used by the Auto-m estimator tests: two 400-row null levels exactly
    /// at base (the junk mass) plus two 10-row signal levels at +/-4, with +/-1 row noise
    /// everywhere so within == 1 and base == 1 exactly.
    fn junk_heavy_fixture() -> (Vec<String>, Vec<f32>) {
        let mut levels = Vec::new();
        let mut y = Vec::new();
        for (label, n, mean) in [
            ("j0", 400usize, 1.0f32),
            ("j1", 400, 1.0),
            ("s0", 10, 5.0),
            ("s1", 10, -3.0),
        ] {
            for i in 0..n {
                levels.push(label.to_owned());
                y.push(if i % 2 == 0 { mean - 1.0 } else { mean + 1.0 });
            }
        }
        (levels, y)
    }

    #[test]
    fn auto_m_cred9_matches_closed_form_and_shrinks_junk_heavy_features_less() {
        let (levels, y) = junk_heavy_fixture();
        let w = vec![1.0_f32; y.len()];
        let rows = categorical_row_terms(&y, &w, None, CatTarget::Mean).unwrap();
        let base = categorical_base_from_terms(&rows, CatTarget::Mean).unwrap();
        assert!((base - 1.0).abs() < 1e-6);
        // Variance ratio is junk-mass-diluted: within/between = 1 / (320/820).
        let curr = variance_ratio_smooth_strength(&levels, &rows, base).unwrap();
        assert!((curr - 820.0 / 320.0).abs() < 1e-4, "curr={curr}");
        // SS_within = 820 (every row is exactly +-1 from its level mean). The dof-corrected
        // s2 = SS_within / (N - L) = 820 / (820 - 4), NOT / 820 (auto_moments). Signal
        // levels: z^2 = 10*16/s2 each (junk z^2 = 0 exactly), so tau2 = s2*(2*(z2-1) -
        // kappa*4)/20 and m = s2/tau2.
        let s2 = 820.0_f64 / (820.0 - 4.0);
        let z2_signal = 10.0 * 16.0 / s2;
        let zsum = 2.0 * (z2_signal - 1.0);
        let tau2 = s2 * (zsum - 4.0 * CRED_NULL_KAPPA) / 20.0;
        let want = s2 / tau2;
        let cred = auto_smooth_strength(&levels, &rows, base).unwrap();
        assert!(
            (f64::from(cred) - want).abs() < 1e-6,
            "cred={cred} want={want}"
        );
        assert!(cred < curr, "cred={cred} curr={curr}");
        // Deterministic.
        assert_eq!(cred, auto_smooth_strength(&levels, &rows, base).unwrap());

        // Pure-null feature (3+ levels, all exactly at base, with row noise) -> MAX
        // (full shrink), unlike the variance ratio which returns a finite noise-scaled m.
        let levels_null: Vec<String> = ["a", "a", "b", "b", "c", "c"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let y_null = [0.0_f32, 2.0, 0.0, 2.0, 0.0, 2.0];
        let w_null = vec![1.0_f32; 6];
        let rows_null = categorical_row_terms(&y_null, &w_null, None, CatTarget::Mean).unwrap();
        let base_null = categorical_base_from_terms(&rows_null, CatTarget::Mean).unwrap();
        let cred_null = auto_smooth_strength(&levels_null, &rows_null, base_null).unwrap();
        assert_eq!(f64::from(cred_null), MAX_AUTO_SMOOTH);
    }

    #[test]
    fn auto_m_cred9_keeps_the_variance_ratio_for_binary_features() {
        // 2 levels with real means, uneven weights, row noise: the partition is
        // encoding-invariant, so the credible-set arm must return the shipped variance-ratio
        // m bit-for-bit (no noise-path re-rolls on binary-only datasets).
        let levels: Vec<String> = ["a", "a", "a", "a", "b", "b"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let y = [0.0_f32, 2.0, 1.0, 3.0, 5.0, 9.0];
        let w = vec![1.0_f32; 6];
        let rows = categorical_row_terms(&y, &w, None, CatTarget::Mean).unwrap();
        let base = categorical_base_from_terms(&rows, CatTarget::Mean).unwrap();
        let curr = variance_ratio_smooth_strength(&levels, &rows, base).unwrap();
        let cred = auto_smooth_strength(&levels, &rows, base).unwrap();
        assert!(curr > 0.0 && curr < MAX_AUTO_SMOOTH as f32);
        assert_eq!(curr.to_bits(), cred.to_bits());
    }

    #[test]
    fn auto_m_cred9_zero_within_variance_and_real_signal_means_no_shrink() {
        // 3+ levels, y exactly deterministic per level (every row of a level shares the same
        // y): within == 0 exactly while between > 0 (real signal). Zero row-level noise means
        // the level means are EXACT, so Auto-m must resolve to ~0 (no shrink) like the legacy
        // variance-ratio estimator's within/between -> 0 in this regime -- not MAX_AUTO_SMOOTH,
        // which was the pre-fix (inverted) behavior of the degenerate branch.
        let levels: Vec<String> = ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let y = [10.0_f32, 10.0, 10.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0];
        let w = vec![1.0_f32; y.len()];
        let rows = categorical_row_terms(&y, &w, None, CatTarget::Mean).unwrap();
        let base = categorical_base_from_terms(&rows, CatTarget::Mean).unwrap();
        assert!((base - 20.0).abs() < 1e-6);

        let mo = auto_moments(&levels, &rows).unwrap();
        assert!(mo.within <= MIN_AUTO_VARIANCE, "within={}", mo.within);
        // Real between-level signal by construction (level means 10/20/30 vs base 20),
        // confirmed below: the estimator must not collapse them toward base.
        assert!(mo.mean.iter().any(|&m| (m - f64::from(base)).abs() > 1.0));

        let m = auto_smooth_strength(&levels, &rows, base).unwrap();
        assert!(
            m.abs() < 1e-6,
            "m={m}, want ~=0 (exact level means, no shrink)"
        );

        // End to end: fit_cat_encoder under Smooth::Auto (bypass disabled) must freeze the raw
        // level means, not collapse every level toward base.
        let cfg = TsConfig {
            leakage: LeakageScheme::KFold { k: 3 },
            smooth: Smooth::Auto,
            min_data_per_group: 0.0,
            direct_max_levels: 0, // keep the low-cardinality bypass out; exercise Auto directly
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(11),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 5,
        };
        let (enc, _train) = fit_cat_encoder(&levels, &y, spec).unwrap();
        assert!((enc.encode_label("a") - 10.0).abs() < 1e-4);
        assert!((enc.encode_label("b") - 20.0).abs() < 1e-4);
        assert!((enc.encode_label("c") - 30.0).abs() < 1e-4);
    }

    #[test]
    fn auto_moments_within_uses_dof_corrected_denominator() {
        // 2 levels x 3 unit-weight rows each. SS_within = Sum (y_i - level_mean)^2:
        // level "a" (y=1,2,3, mean=2): 1+0+1 = 2; level "b" (y=10,12,14, mean=12): 4+0+4 = 8;
        // total SS_within = 10. The dof-corrected pooled variance is the standard unbiased
        // ANOVA within-group estimator SS_within / (N - L) = 10 / (6 - 2) = 2.5. Dividing by
        // the raw total mass alone (the pre-fix bug) would instead give 10 / 6 ~= 1.667 —
        // biased low by exactly N / (N - L) = 1.5x, so this pins the fix, not just a
        // plausible-looking number.
        let levels: Vec<String> = ["a", "a", "a", "b", "b", "b"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let rows: Vec<CatRowTerm> = [1.0_f64, 2.0, 3.0, 10.0, 12.0, 14.0]
            .iter()
            .map(|&sum_y| CatRowTerm { sum_y, denom: 1.0 })
            .collect();
        let mo = auto_moments(&levels, &rows).unwrap();
        assert!(
            (mo.within - 2.5).abs() < 1e-9,
            "within should be the dof-corrected 2.5, got {} (a pre-fix run would give ~1.667)",
            mo.within
        );
    }

    #[test]
    fn fit_cat_encoder_is_deterministic_and_freezes_full_data_map() {
        let levels = vec!["b", "a", "b", "c", "a", "c"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = [2.0_f32, 1.0, 4.0, 8.0, 3.0, 10.0];
        let cfg = TsConfig {
            leakage: LeakageScheme::Ordered { n_perms: 3 },
            smooth: Smooth::Fixed { m: 2.0 },
            min_data_per_group: 0.0,
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(2),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 99,
        };
        let (enc_a, train_a) = fit_cat_encoder(&levels, &y, spec).unwrap();
        let spec = CatFitSpec {
            raw: FeatureId(2),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 99,
        };
        let (enc_b, train_b) = fit_cat_encoder(&levels, &y, spec).unwrap();
        assert_eq!(enc_a, enc_b);
        assert_eq!(train_a, train_b);
        assert_eq!(enc_a.base, y.iter().sum::<f32>() / y.len() as f32);
        assert_eq!(
            enc_a
                .levels
                .iter()
                .map(|l| l.label.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
        assert_eq!(
            enc_a.levels.iter().map(|l| l.bin).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(enc_a.encode_label("unseen"), enc_a.base);
        assert!(train_a.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn zero_weight_level_falls_back_to_base_under_direct_bypass() {
        // Level "a"'s rows are ALL weight 0 (a standard exclusion idiom), so it aggregates to
        // (sum_y=0, denom=0). 3 post-collapse levels with the default direct_max_levels forces
        // Smooth::Fixed{m:0}, which used to hand shrunken_encoding a zero denominator and
        // hard-error the whole fit; it must now fall back to `base` for that level, in both the
        // frozen serve map and the direct-bypass training rows.
        let levels: Vec<String> = ["a", "a", "b", "b", "b", "c", "c", "c"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let y = [1.0_f32, 1.0, 4.0, 6.0, 5.0, 9.0, 11.0, 10.0];
        let w = [0.0_f32, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let cfg = TsConfig {
            min_data_per_group: 0.0, // no rare-level collapse: keep "a" as its own level
            ..TsConfig::default()    // direct_max_levels=16 default; 3 levels ⇒ bypass engages
        };
        let spec = CatFitSpec {
            raw: FeatureId(13),
            id: TsEncodingId(0),
            weight: Some(&w),
            exposure: None,
            config: &cfg,
            seed: 17,
        };
        let (enc, train) = fit_cat_encoder(&levels, &y, spec).unwrap();
        assert!(matches!(enc.config.smooth, Smooth::Fixed { m } if m == 0.0));
        assert_eq!(enc.encode_label("a"), enc.base);
        assert!(train.iter().all(|v| v.is_finite()));
        assert_eq!(train[0], enc.base);
        assert_eq!(train[1], enc.base);

        // A genuinely invalid global state -- EVERY row weight 0 -- must still hard-error
        // (guarded upstream by categorical_base_from_terms, untouched by the per-level fallback).
        let w_all_zero = [0.0_f32; 8];
        let spec_all_zero = CatFitSpec {
            raw: FeatureId(13),
            id: TsEncodingId(0),
            weight: Some(&w_all_zero),
            exposure: None,
            config: &cfg,
            seed: 17,
        };
        assert!(matches!(
            fit_cat_encoder(&levels, &y, spec_all_zero),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn log_mean_target_encodes_positive_targets_on_log_scale() {
        let levels = vec!["low", "low", "high", "high"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = [2.0_f32, 8.0, 20.0, 80.0];
        let weight = [1.0_f32, 3.0, 2.0, 2.0];
        let cfg = TsConfig {
            leakage: LeakageScheme::KFold { k: 2 },
            smooth: Smooth::Fixed { m: 0.0 },
            target: CatTarget::LogMean,
            min_data_per_group: 0.0,
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(2),
            id: TsEncodingId(0),
            weight: Some(&weight),
            exposure: None,
            config: &cfg,
            seed: 99,
        };
        let (enc, train) = fit_cat_encoder(&levels, &y, spec).unwrap();
        let low = (2.0_f32.ln() + 3.0 * 8.0_f32.ln()) / 4.0;
        let high = (2.0 * 20.0_f32.ln() + 2.0 * 80.0_f32.ln()) / 4.0;
        assert!((enc.encode_label("low") - low).abs() < 1e-6);
        assert!((enc.encode_label("high") - high).abs() < 1e-6);
        assert_eq!(enc.config.target, CatTarget::LogMean);
        assert!(train.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn log_mean_target_rejects_non_positive_targets() {
        let levels = vec!["a".to_owned(), "b".to_owned()];
        let y = [1.0_f32, 0.0];
        let cfg = TsConfig {
            target: CatTarget::LogMean,
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(2),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 99,
        };
        assert!(matches!(
            fit_cat_encoder(&levels, &y, spec),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn direct_bypass_trains_on_the_frozen_full_fit_statistics() {
        // 3 levels ≤ direct_max_levels ⇒ bypass: every train row's encoding equals its level's
        // raw full-fit statistic (no cross-fit noise), and the configured smoothing — set here
        // strong enough to crush everything toward base — is ignored.
        let levels = vec!["a", "b", "c", "a", "b", "c"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = [1.0_f32, 10.0, 100.0, 3.0, 12.0, 104.0];
        let cfg = TsConfig {
            leakage: LeakageScheme::KFold { k: 2 },
            smooth: Smooth::Fixed { m: 1.0e6 },
            min_data_per_group: 0.0,
            direct_max_levels: 8,
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(0),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 11,
        };
        let (enc, train) = fit_cat_encoder(&levels, &y, spec).unwrap();
        for (label, t) in levels.iter().zip(&train) {
            let served = enc.encode_label(label);
            assert!(
                (t - served).abs() < 1e-6,
                "{label}: train {t} vs serve {served}"
            );
        }
        // Raw level means survive unshrunk: a=2.0, c=102.0 (base ≈ 38.3 would dominate if the
        // Fixed{1e6} smoothing had applied).
        assert!((enc.encode_label("a") - 2.0).abs() < 1e-3);
        assert!((enc.encode_label("c") - 102.0).abs() < 1e-3);
    }

    #[test]
    fn direct_bypass_exempts_binary_features() {
        // 2 levels ≤ direct_max_levels, but the bypass must NOT engage: a binary partition is
        // encoding-invariant, so bypassing would only perturb early-stopping paths. Cross-fit
        // training encodings (which differ from the frozen map) prove the TS path stayed active.
        let levels = vec!["a", "b", "a", "b", "a", "b"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = [1.0_f32, 10.0, 3.0, 12.0, 2.0, 14.0];
        let cfg = TsConfig {
            leakage: LeakageScheme::KFold { k: 2 },
            smooth: Smooth::Fixed { m: 0.0 },
            min_data_per_group: 0.0,
            direct_max_levels: 16,
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(0),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 11,
        };
        let (enc, train) = fit_cat_encoder(&levels, &y, spec).unwrap();
        let any_differs = levels
            .iter()
            .zip(&train)
            .any(|(label, t)| (t - enc.encode_label(label)).abs() > 1e-6);
        assert!(
            any_differs,
            "binary feature unexpectedly took the direct bypass"
        );
    }

    #[test]
    fn direct_bypass_disabled_keeps_cross_fit_training_encodings() {
        // Same data with the gate off (direct_max_levels=0): k-fold cross-fit training encodings
        // must differ from the frozen serve map on at least one row — the bypass is opt-gated,
        // not a silent global behavior change.
        let levels = vec!["a", "b", "c", "a", "b", "c"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = [1.0_f32, 10.0, 100.0, 3.0, 12.0, 104.0];
        let cfg = TsConfig {
            leakage: LeakageScheme::KFold { k: 2 },
            smooth: Smooth::Fixed { m: 0.0 },
            min_data_per_group: 0.0,
            direct_max_levels: 0,
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(0),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 11,
        };
        let (enc, train) = fit_cat_encoder(&levels, &y, spec).unwrap();
        let any_differs = levels
            .iter()
            .zip(&train)
            .any(|(label, t)| (t - enc.encode_label(label)).abs() > 1e-6);
        assert!(
            any_differs,
            "k-fold training encodings unexpectedly matched the frozen map"
        );
    }

    #[test]
    fn full_data_encoder_uses_fisher_sorted_ordinal_order() {
        let levels = vec!["z", "a", "m", "z", "a", "m"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = [30.0_f32, 1.0, 10.0, 40.0, 2.0, 12.0];
        let cfg = TsConfig {
            leakage: LeakageScheme::KFold { k: 2 },
            smooth: Smooth::Fixed { m: 0.0 },
            min_data_per_group: 0.0,
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(7),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 4,
        };
        let (enc, _) = fit_cat_encoder(&levels, &y, spec).unwrap();
        assert_eq!(
            enc.levels
                .iter()
                .map(|l| l.label.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "m", "z"]
        );
        assert_eq!(enc.encode_label("a"), 1.5);
        assert_eq!(enc.encode_label("m"), 11.0);
        assert_eq!(enc.encode_label("z"), 35.0);
        assert_eq!(
            enc.levels.iter().map(|l| l.bin).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn rare_levels_collapse_before_fisher_ordering_but_unseen_uses_base() {
        let levels = vec!["rare_a", "common", "common", "rare_b", "common"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = [100.0_f32, 1.0, 3.0, 200.0, 5.0];
        let cfg = TsConfig {
            leakage: LeakageScheme::KFold { k: 2 },
            smooth: Smooth::Fixed { m: 0.0 },
            min_data_per_group: 2.0,
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(8),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 5,
        };
        let (enc, train) = fit_cat_encoder(&levels, &y, spec).unwrap();
        assert_eq!(
            enc.levels
                .iter()
                .map(|level| level.label.as_str())
                .collect::<Vec<_>>(),
            vec!["common", RARE_LEVEL_LABEL]
        );
        let rare = enc
            .levels
            .iter()
            .find(|level| level.label == RARE_LEVEL_LABEL)
            .unwrap();
        assert_eq!(
            rare.members,
            vec!["rare_a".to_string(), "rare_b".to_string()]
        );
        assert_eq!(enc.encode_label("rare_a"), rare.encoding);
        assert_eq!(enc.encode_label("rare_b"), rare.encoding);
        assert_eq!(enc.encode_label("brand_new"), enc.base);
        assert_eq!(train.len(), levels.len());
        assert!(train.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn high_cardinality_levels_share_the_254_bin_budget() {
        let n = 300usize;
        let levels = (0..n).map(|i| format!("l{i:03}")).collect::<Vec<_>>();
        let y = (0..n).map(|i| i as f32).collect::<Vec<_>>();
        let cfg = TsConfig {
            leakage: LeakageScheme::KFold { k: 5 },
            smooth: Smooth::Fixed { m: 0.0 },
            min_data_per_group: 0.0,
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(3),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 10,
        };
        let (enc, _) = fit_cat_encoder(&levels, &y, spec).unwrap();
        assert_eq!(enc.levels.len(), n);
        assert_eq!(enc.levels.first().unwrap().bin, 1);
        assert_eq!(enc.levels.last().unwrap().bin, 254);
        assert!(enc.levels.windows(2).all(|w| w[0].bin <= w[1].bin));
    }

    #[test]
    fn tied_encodings_share_bins_and_reconstruct_a_grid() {
        let levels = vec!["b", "a", "d", "c"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = [1.0_f32, 1.0, 1.0, 1.0];
        let cfg = TsConfig {
            leakage: LeakageScheme::KFold { k: 2 },
            smooth: Smooth::Fixed { m: 0.0 },
            min_data_per_group: 0.0,
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(9),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 1,
        };
        let (enc, _) = fit_cat_encoder(&levels, &y, spec).unwrap();
        assert!(enc.levels.iter().all(|level| level.bin == 1));
        let grid = enc.border_grid().unwrap();
        assert!(grid.borders.is_empty());
        assert_eq!(grid.n_bins, 2);
    }

    #[test]
    fn next_down_f32_steps_by_exactly_one_ulp() {
        assert_eq!(next_down_f32(1.0_f32).to_bits(), 1.0_f32.to_bits() - 1);
        assert_eq!(next_down_f32(-1.0_f32).to_bits(), (-1.0_f32).to_bits() + 1);
        assert_eq!(next_down_f32(0.0_f32), -f32::from_bits(1));
        assert_eq!(next_down_f32(-0.0_f32), -f32::from_bits(1));
        assert!(next_down_f32(1.0_f32) < 1.0_f32);
        assert!(next_up_f32(next_down_f32(2.5_f32)) == 2.5_f32);
        assert!(next_up_f32(f32::NAN).is_nan());
        assert_eq!(next_up_f32(f32::INFINITY), f32::INFINITY);
    }

    #[test]
    fn border_grid_separates_levels_one_ulp_apart_despite_f64_midpoint_rounding_up() {
        // A concrete adjacent pair (found empirically, not asserted by construction) whose
        // f64 midpoint rounds, via ties-to-even, exactly onto the UPPER encoding — the
        // silent-merge direction of the bug. Pre-fix, border_grid() would have pushed a
        // border == level.encoding, so bin()'s "strictly below" rule would route BOTH
        // encodings into the lower bin, contradicting their distinct stored `bin` ids.
        let prev_encoding = 0.300_000_04_f32;
        let level_encoding = 0.300_000_07_f32;
        assert_eq!(
            level_encoding.to_bits(),
            prev_encoding.to_bits() + 1,
            "fixture must stay exactly 1 ULP apart"
        );
        let midpoint = ((f64::from(prev_encoding) + f64::from(level_encoding)) / 2.0) as f32;
        assert_eq!(
            midpoint, level_encoding,
            "fixture must reproduce the buggy round-up-to-upper-encoding case"
        );

        let enc = CatEncoder {
            raw: FeatureId(1),
            id: TsEncodingId(0),
            levels: vec![
                CatLevel {
                    label: "a".into(),
                    members: vec!["a".into()],
                    encoding: prev_encoding,
                    bin: 1,
                    weight: 10.0,
                },
                CatLevel {
                    label: "b".into(),
                    members: vec!["b".into()],
                    encoding: level_encoding,
                    bin: 2,
                    weight: 10.0,
                },
            ],
            base: 0.5,
            config: TsConfig::default(),
        };
        let grid = enc.border_grid().unwrap();
        for level in &enc.levels {
            assert_eq!(
                crate::data::bin::bin(level.encoding, &grid).unwrap(),
                level.bin,
                "level `{}` must bin to its stored bin id",
                level.label
            );
        }
    }

    #[test]
    fn kfold_training_encoding_does_not_consult_own_target() {
        let levels = vec!["a".to_string(); 30];
        let y = (0..30).map(|i| i as f32).collect::<Vec<_>>();
        let cfg = TsConfig {
            leakage: LeakageScheme::KFold { k: 5 },
            smooth: Smooth::Fixed { m: 0.0 },
            direct_max_levels: 0, // this test exercises k-fold cross-fitting — keep the bypass out
            ..TsConfig::default()
        };
        let spec = CatFitSpec {
            raw: FeatureId(0),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 123,
        };
        let (_, train_a) = fit_cat_encoder(&levels, &y, spec).unwrap();
        let mut changed = y.clone();
        changed[0] += 10_000.0;
        let spec = CatFitSpec {
            raw: FeatureId(0),
            id: TsEncodingId(0),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 123,
        };
        let (_, train_b) = fit_cat_encoder(&levels, &changed, spec).unwrap();
        assert_eq!(train_a[0], train_b[0]);
    }

    #[test]
    fn leave_one_out_excludes_each_rows_own_target() {
        // Two rows in one level; m=0 ⇒ no shrinkage, so each row's LOO encoding is exactly
        // the OTHER row's target (its own is excluded) — leakage-free by construction.
        let levels = vec!["a".to_string(), "a".to_string()];
        let rows = vec![
            CatRowTerm {
                sum_y: 1.0,
                denom: 1.0,
            },
            CatRowTerm {
                sum_y: 3.0,
                denom: 1.0,
            },
        ];
        let enc = loo_training_encodings(&levels, &rows, 2.0, Smooth::Fixed { m: 0.0 }).unwrap();
        assert!(
            (enc[0] - 3.0).abs() < 1e-6,
            "row 0 sees only row 1's target"
        );
        assert!(
            (enc[1] - 1.0).abs() < 1e-6,
            "row 1 sees only row 0's target"
        );
        // With no allowed observations, even the prior must be target independent.
        let solo = loo_training_encodings(
            &["b".to_string()],
            &[CatRowTerm {
                sum_y: 5.0,
                denom: 1.0,
            }],
            2.0,
            Smooth::Fixed { m: 0.0 },
        )
        .unwrap();
        assert_eq!(solo[0], 0.0, "singleton ⇒ neutral prior");
    }

    #[test]
    fn default_leakage_scheme_is_kfold() {
        assert!(matches!(
            LeakageScheme::default(),
            LeakageScheme::KFold { k: 5 }
        ));
    }

    // ===================================================================
    // P1 multi-channel: the target-free `CatTarget::Count` channel.
    // ===================================================================

    fn count_cfg(min_data_per_group: f32) -> TsConfig {
        TsConfig {
            target: CatTarget::Count,
            min_data_per_group,
            direct_max_levels: 0, // irrelevant to Count, but keep it out of the way
            ..TsConfig::default()
        }
    }

    #[test]
    fn count_channel_matches_closed_form_log1p_share() {
        // 5 rows, unit weight: level "a" gets 2/5 share, level "b" gets 3/5 share.
        let levels = vec!["a", "a", "b", "b", "b"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = [0.0_f32; 5]; // unread by the count channel
        let cfg = count_cfg(0.0);
        let spec = CatFitSpec {
            raw: FeatureId(7),
            id: TsEncodingId(1),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 0,
        };
        let (enc, train) = fit_cat_encoder(&levels, &y, spec).unwrap();
        let want_a = (2.0_f64 / 5.0 * 5.0).ln_1p() as f32; // log1p(2.0)
        let want_b = (3.0_f64 / 5.0 * 5.0).ln_1p() as f32; // log1p(3.0)
        assert!((enc.encode_label("a") - want_a).abs() < 1e-6);
        assert!((enc.encode_label("b") - want_b).abs() < 1e-6);
        assert_eq!(enc.raw, FeatureId(7));
        assert_eq!(enc.id, TsEncodingId(1));
        assert_eq!(train, vec![want_a, want_a, want_b, want_b, want_b]);
    }

    #[test]
    fn count_channel_base_is_zero_for_unseen_levels() {
        let levels = vec!["x", "x", "y"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = [1.0_f32, 2.0, 3.0]; // unread
        let cfg = count_cfg(0.0);
        let spec = CatFitSpec {
            raw: FeatureId(0),
            id: TsEncodingId(1),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 0,
        };
        let (enc, _train) = fit_cat_encoder(&levels, &y, spec).unwrap();
        assert_eq!(enc.base, 0.0);
        assert_eq!(enc.encode_label("never_seen"), 0.0);
    }

    #[test]
    fn count_channel_ignores_the_target_entirely() {
        // Same levels/weights, two wildly different `y` arrays ⇒ identical count encoder:
        // the channel is target-free by construction, not merely "shrunk to look similar".
        let levels = vec!["a", "b", "a", "c", "b", "c"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y_a = [0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let y_b = [1.0_f32, -50.0, 9.0, 1e6, -3.5, 0.25];
        let cfg = count_cfg(0.0);
        let spec_a = CatFitSpec {
            raw: FeatureId(3),
            id: TsEncodingId(1),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 42,
        };
        let spec_b = CatFitSpec {
            raw: FeatureId(3),
            id: TsEncodingId(1),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 42,
        };
        let (enc_a, train_a) = fit_cat_encoder(&levels, &y_a, spec_a).unwrap();
        let (enc_b, train_b) = fit_cat_encoder(&levels, &y_b, spec_b).unwrap();
        assert_eq!(enc_a, enc_b);
        assert_eq!(train_a, train_b);
    }

    // ---- P3 per-class frequency channels (design/multichannel-categoricals.md §8) ----

    /// A `ClassFreq` config for class `k` with smoothing and the low-cardinality direct bypass
    /// both disabled, so the fitted level values are the RAW per-level class frequencies and
    /// can be checked against a closed form.
    fn class_freq_cfg(class: u32, min_data_per_group: f32) -> TsConfig {
        TsConfig {
            target: CatTarget::ClassFreq { class },
            smooth: Smooth::Fixed { m: 0.0 },
            leakage: LeakageScheme::KFold { k: 2 },
            direct_max_levels: 0,
            min_data_per_group,
            ..TsConfig::default()
        }
    }

    /// 4 levels x a deliberately UNEVEN class mix, so no two levels share a class profile and
    /// the label-mean statistic is a genuinely lossy summary of them.
    fn class_freq_fixture() -> (Vec<String>, Vec<f32>) {
        let mut levels = Vec::new();
        let mut y = Vec::new();
        // level, then that level's class sequence.
        for (label, classes) in [
            ("a", vec![0.0_f32, 0.0, 2.0, 2.0]), // 50% class 0, 0% class 1, 50% class 2
            ("b", vec![1.0, 1.0, 1.0, 1.0]),     // 100% class 1
            ("c", vec![0.0, 1.0, 2.0, 0.0]),     // 50/25/25
            ("d", vec![2.0, 2.0, 2.0, 1.0]),     // 0/25/75
        ] {
            for c in classes {
                levels.push(label.to_string());
                y.push(c);
            }
        }
        (levels, y)
    }

    #[test]
    fn class_freq_channel_encodes_the_per_level_one_vs_rest_frequency() {
        let (levels, y) = class_freq_fixture();
        let cfg = class_freq_cfg(2, 0.0);
        let (enc, _train) = fit_cat_encoder(
            &levels,
            &y,
            CatFitSpec {
                raw: FeatureId(4),
                id: TsEncodingId(4),
                weight: None,
                exposure: None,
                config: &cfg,
                seed: 0,
            },
        )
        .unwrap();
        assert_eq!(enc.raw, FeatureId(4));
        assert_eq!(enc.id, TsEncodingId(4));
        for (label, want) in [("a", 0.5_f32), ("b", 0.0), ("c", 0.25), ("d", 0.75)] {
            assert!(
                (enc.encode_label(label) - want).abs() < 1e-6,
                "class-2 frequency for `{label}`: got {}, want {want}",
                enc.encode_label(label)
            );
        }
        // Unseen levels fall back to the overall class-2 rate (6 of 16 rows), exactly as the
        // Mean channel's base does for its own statistic.
        assert!(
            (enc.base - 6.0 / 16.0).abs() < 1e-6,
            "base should be the marginal class-2 rate, got {}",
            enc.base
        );
    }

    /// The K channels of one feature form the level's full class DISTRIBUTION: unsmoothed,
    /// their per-level values sum to 1. This is the property the ordinal label-mean channel
    /// cannot express — it only carries `Σ_k k·p_k`.
    #[test]
    fn class_freq_channels_sum_to_one_per_level() {
        let (levels, y) = class_freq_fixture();
        let mut per_level = std::collections::BTreeMap::<&str, f32>::new();
        for class in 0..3_u32 {
            let cfg = class_freq_cfg(class, 0.0);
            let (enc, _) = fit_cat_encoder(
                &levels,
                &y,
                CatFitSpec {
                    raw: FeatureId(0),
                    id: TsEncodingId(u8::try_from(2 + class).unwrap()),
                    weight: None,
                    exposure: None,
                    config: &cfg,
                    seed: 0,
                },
            )
            .unwrap();
            for label in ["a", "b", "c", "d"] {
                *per_level.entry(label).or_default() += enc.encode_label(label);
            }
        }
        for (label, total) in per_level {
            assert!(
                (total - 1.0).abs() < 1e-6,
                "class frequencies for `{label}` sum to {total}, not 1"
            );
        }
    }

    /// The load-bearing invariant for the joint per-level collapse: every channel of one raw
    /// feature must partition levels into the SAME rare bucket. `ClassFreq` shares `Mean`'s and
    /// `Count`'s `Σ(w·e)` pooling denominator, so a mixed mean + count + per-class channel set
    /// agrees on the frozen level set even when pooling actually fires.
    #[test]
    fn class_freq_pools_rare_levels_identically_to_the_mean_and_count_channels() {
        // "rare" carries 3 rows (below the floor of 10); "common"/"other" clear it.
        let mut levels = vec!["rare".to_string(); 3];
        levels.extend(std::iter::repeat_n("common".to_string(), 20));
        levels.extend(std::iter::repeat_n("other".to_string(), 15));
        let y: Vec<f32> = (0..levels.len()).map(|i| (i % 3) as f32).collect();

        let frozen_labels = |cfg: &TsConfig, id: u8| -> Vec<String> {
            let (enc, _) = fit_cat_encoder(
                &levels,
                &y,
                CatFitSpec {
                    raw: FeatureId(0),
                    id: TsEncodingId(id),
                    weight: None,
                    exposure: None,
                    config: cfg,
                    seed: 0,
                },
            )
            .unwrap();
            let mut out: Vec<String> = enc.levels.iter().map(|l| l.label.clone()).collect();
            out.sort();
            out
        };

        let mean_cfg = TsConfig {
            target: CatTarget::Mean,
            smooth: Smooth::Fixed { m: 0.0 },
            direct_max_levels: 0,
            min_data_per_group: 10.0,
            ..TsConfig::default()
        };
        let want = frozen_labels(&mean_cfg, 0);
        assert!(
            want.iter().any(|l| l == RARE_LEVEL_LABEL),
            "fixture must actually trigger rare pooling"
        );
        assert_eq!(want, frozen_labels(&count_cfg(10.0), 1));
        for class in 0..3_u32 {
            assert_eq!(
                want,
                frozen_labels(
                    &class_freq_cfg(class, 10.0),
                    u8::try_from(2 + class).unwrap()
                ),
                "class {class} channel pooled levels differently from the mean channel"
            );
        }
    }

    /// `ClassFreq` is a genuine target statistic (unlike `Count`): its k-fold cross-fit
    /// training encodings are leakage-free, so a row's own label does not appear in its own
    /// encoding — the training values differ from the frozen full-data serve map.
    #[test]
    fn class_freq_training_encodings_are_cross_fit_not_the_serve_map() {
        let (levels, y) = class_freq_fixture();
        let cfg = class_freq_cfg(1, 0.0);
        let (enc, train) = fit_cat_encoder(
            &levels,
            &y,
            CatFitSpec {
                raw: FeatureId(0),
                id: TsEncodingId(3),
                weight: None,
                exposure: None,
                config: &cfg,
                seed: 5,
            },
        )
        .unwrap();
        let serve: Vec<f32> = levels.iter().map(|l| enc.encode_label(l)).collect();
        assert_ne!(
            train, serve,
            "k-fold cross-fit training encodings must not equal the full-data serve map"
        );
    }

    /// The defect P3 exists to fix, stated as a test: the ordinal `Mean` channel maps two
    /// levels with COMPLETELY different class distributions to the same encoding (hence the
    /// same bin, hence indistinguishable to every tree), while the per-class channels separate
    /// them.
    #[test]
    fn class_freq_separates_levels_the_ordinal_label_mean_collapses() {
        // "p" is all class 1; "q" is half class 0, half class 2. Both have label mean 1.0.
        let mut levels = vec!["p".to_string(); 20];
        levels.extend(std::iter::repeat_n("q".to_string(), 20));
        let mut y = vec![1.0_f32; 20];
        y.extend(std::iter::repeat_n(0.0_f32, 10));
        y.extend(std::iter::repeat_n(2.0_f32, 10));

        let fit = |cfg: &TsConfig| -> CatEncoder {
            fit_cat_encoder(
                &levels,
                &y,
                CatFitSpec {
                    raw: FeatureId(0),
                    id: TsEncodingId(0),
                    weight: None,
                    exposure: None,
                    config: cfg,
                    seed: 0,
                },
            )
            .unwrap()
            .0
        };

        let mean = fit(&TsConfig {
            target: CatTarget::Mean,
            smooth: Smooth::Fixed { m: 0.0 },
            direct_max_levels: 0,
            min_data_per_group: 0.0,
            ..TsConfig::default()
        });
        assert!(
            (mean.encode_label("p") - mean.encode_label("q")).abs() < 1e-6,
            "the fixture's whole point: the label-mean statistic cannot tell `p` from `q`"
        );

        let cls0 = fit(&class_freq_cfg(0, 0.0));
        assert!(
            (cls0.encode_label("p") - cls0.encode_label("q")).abs() > 0.4,
            "the class-0 frequency channel must separate them (0.0 vs 0.5)"
        );
    }

    #[test]
    fn count_channel_reuses_rare_level_pooling() {
        // "rare" carries weight 1 (below the floor of 10), "common" carries weight 20 ⇒
        // "rare" collapses into the reserved rare bucket, which must still get a representative
        // share value from its (here, sole) pooled member's aggregate weight.
        let mut levels = vec!["rare".to_string()];
        levels.extend(std::iter::repeat_n("common".to_string(), 20));
        let y = vec![0.0_f32; levels.len()]; // unread
        let cfg = count_cfg(10.0);
        let spec = CatFitSpec {
            raw: FeatureId(0),
            id: TsEncodingId(1),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 0,
        };
        let (enc, _train) = fit_cat_encoder(&levels, &y, spec).unwrap();
        // "rare" is no longer its own level; it is folded into the reserved bucket.
        assert!(enc.levels.iter().all(|l| l.label != "rare"));
        let bucket = enc
            .levels
            .iter()
            .find(|l| l.label == RARE_LEVEL_LABEL)
            .expect("rare bucket must be present");
        assert_eq!(bucket.members, vec!["rare".to_string()]);
        let n = levels.len() as f64;
        let want = (1.0_f64 / n * n).ln_1p() as f32; // log1p(1.0) — the pooled bucket's own share
        assert!((bucket.encoding - want).abs() < 1e-6);
        // Unseen labels (e.g. the original "rare" string is gone as its own level) still fall
        // back through `encode_label`'s member-lookup, not through `base`.
        assert!((enc.encode_label("rare") - want).abs() < 1e-6);
    }

    #[test]
    fn count_channel_is_deterministic() {
        let levels = vec!["a", "b", "a", "a", "c", "b"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let y = vec![0.0_f32; levels.len()];
        let cfg = count_cfg(0.0);
        let make_spec = || CatFitSpec {
            raw: FeatureId(1),
            id: TsEncodingId(1),
            weight: None,
            exposure: None,
            config: &cfg,
            seed: 7,
        };
        let (enc_a, train_a) = fit_cat_encoder(&levels, &y, make_spec()).unwrap();
        let (enc_b, train_b) = fit_cat_encoder(&levels, &y, make_spec()).unwrap();
        assert_eq!(enc_a, enc_b);
        assert_eq!(train_a, train_b);
    }

    #[test]
    fn count_channel_respects_per_row_weight() {
        // Weighted share, not a plain row-count share: level "a" carries weight 1+1=2 of a
        // total 2+3=5 (unchanged from the unweighted fixture above), but via non-unit weights.
        let levels = vec!["a", "a", "b", "b", "b"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let weight = [0.5_f32, 0.5, 1.0, 1.0, 1.0];
        let y = [0.0_f32; 5];
        let cfg = count_cfg(0.0);
        let spec = CatFitSpec {
            raw: FeatureId(0),
            id: TsEncodingId(1),
            weight: Some(&weight),
            exposure: None,
            config: &cfg,
            seed: 0,
        };
        let (enc, _train) = fit_cat_encoder(&levels, &y, spec).unwrap();
        let want_a = (1.0_f64 / 4.0 * 5.0).ln_1p() as f32; // Σw_a=1, Σw_total=4, N=5 rows
        let want_b = (3.0_f64 / 4.0 * 5.0).ln_1p() as f32;
        assert!((enc.encode_label("a") - want_a).abs() < 1e-6);
        assert!((enc.encode_label("b") - want_b).abs() < 1e-6);
    }

    #[test]
    fn count_channel_rare_pooling_matches_mean_channel_under_exposure() {
        // Level "a": 1 row, weight=20, exposure=0.1 -> Sigma-w=20 (clears a threshold of 5
        // alone) but Sigma-w.e=2 (does NOT clear it). If the count channel pooled by Sigma-w
        // alone, it would keep "a" as its own level while the mean channel (which always uses
        // Sigma-w.e, `categorical_row_terms`) pools it into the rare bucket -- the two channels
        // would then disagree about what "a raw feature's set of levels" even is, which breaks
        // the P1 joint per-level cell space this channel pairing exists to support.
        let mut levels = vec!["a".to_string()];
        let mut weight = vec![20.0_f32];
        let mut exposure = vec![0.1_f32];
        for _ in 0..20 {
            levels.push("common".to_string());
            weight.push(1.0);
            exposure.push(1.0);
        }
        let y: Vec<f32> = levels
            .iter()
            .map(|l| if l == "a" { 5.0 } else { 1.0 })
            .collect();

        let mean_cfg = TsConfig {
            leakage: LeakageScheme::KFold { k: 2 },
            smooth: Smooth::Fixed { m: 0.0 },
            min_data_per_group: 5.0,
            direct_max_levels: 0,
            ..TsConfig::default()
        };
        let mean_spec = CatFitSpec {
            raw: FeatureId(0),
            id: TsEncodingId(0),
            weight: Some(&weight),
            exposure: Some(&exposure),
            config: &mean_cfg,
            seed: 0,
        };
        let (mean_enc, _) = fit_cat_encoder(&levels, &y, mean_spec).unwrap();

        let count_cfg = count_cfg(5.0);
        let count_spec = CatFitSpec {
            raw: FeatureId(0),
            id: TsEncodingId(1),
            weight: Some(&weight),
            exposure: Some(&exposure),
            config: &count_cfg,
            seed: 0,
        };
        let (count_enc, _) = fit_cat_encoder(&levels, &y, count_spec).unwrap();

        let mean_labels: std::collections::BTreeSet<&str> =
            mean_enc.levels.iter().map(|l| l.label.as_str()).collect();
        let count_labels: std::collections::BTreeSet<&str> =
            count_enc.levels.iter().map(|l| l.label.as_str()).collect();
        assert!(!mean_labels.contains("a"), "mean channel should pool `a`");
        assert!(
            !count_labels.contains("a"),
            "count channel should pool `a` to match the mean channel"
        );
        assert_eq!(
            mean_labels, count_labels,
            "paired channels must partition levels identically"
        );
    }

    // ===================================================================
    // `post_pooling_level_count`: the `cat_count_min_levels` cardinality gate's cheap
    // pre-fit level count (count-channel follow-on, design/multichannel-categoricals.md).
    // ===================================================================

    #[test]
    fn post_pooling_level_count_matches_fit_cat_encoder_level_count() {
        // One rare level (weight 1, below the floor of 10) pooled among four common levels
        // (weight 20 each): the gate's cheap pre-check must agree EXACTLY with what a real
        // `fit_cat_encoder` fit would produce (4 kept + 1 rare bucket = 5), for both `Mean`
        // (exposure-weighted denom) and `LogMean` (unweighted-by-exposure denom) targets --
        // proving the shared-denom-convention argument in the function's own doc, not just
        // asserting it.
        let mut levels = vec!["rare".to_string()];
        for label in ["a", "b", "c", "d"] {
            levels.extend(std::iter::repeat_n(label.to_string(), 20));
        }
        let weight = vec![1.0_f32; levels.len()];
        let exposure = vec![0.5_f32; levels.len()]; // deliberately != 1 to exercise Mean's e-term
        let y: Vec<f32> = levels.iter().map(|_| 3.0_f32).collect(); // valid for LogMean too

        for target in [CatTarget::Mean, CatTarget::LogMean] {
            let got =
                post_pooling_level_count(&levels, Some(&weight), Some(&exposure), target, 10.0)
                    .unwrap();
            let cfg = TsConfig {
                target,
                min_data_per_group: 10.0,
                direct_max_levels: 0,
                ..TsConfig::default()
            };
            let spec = CatFitSpec {
                raw: FeatureId(0),
                id: TsEncodingId(0),
                weight: Some(&weight),
                exposure: Some(&exposure),
                config: &cfg,
                seed: 0,
            };
            let (enc, _) = fit_cat_encoder(&levels, &y, spec).unwrap();
            assert_eq!(
                got,
                enc.levels.len(),
                "post_pooling_level_count must match the real encoder's level count for {target:?}"
            );
            assert_eq!(got, 5, "4 kept levels + 1 rare bucket for {target:?}");
        }
    }

    #[test]
    fn post_pooling_level_count_zero_threshold_keeps_every_level_distinct() {
        // `min_data_per_group=0.0` disables pooling entirely (mirrors `collapse_rare_levels`'s
        // own early-return) -- every distinct raw label is its own level, however rare.
        let levels = vec!["a", "a", "b", "c", "c", "c"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let got = post_pooling_level_count(&levels, None, None, CatTarget::Mean, 0.0).unwrap();
        assert_eq!(got, 3);
    }

    #[test]
    fn post_pooling_level_count_defaults_weight_to_unit_when_absent() {
        // No `weight` supplied: every row counts as 1, matching `fit_cat_encoder`'s own
        // `None` -> all-ones default (see its `w` resolution).
        let mut levels = vec!["rare".to_string()];
        levels.extend(std::iter::repeat_n("common".to_string(), 20));
        let got = post_pooling_level_count(&levels, None, None, CatTarget::Mean, 10.0).unwrap();
        assert_eq!(
            got, 2,
            "\"rare\" (weight 1) pools below the floor of 10; \"common\" survives"
        );
    }

    #[test]
    fn post_pooling_level_count_rejects_mismatched_weight_length() {
        let levels = vec!["a".to_string(), "b".to_string()];
        let weight = [1.0_f32]; // one short
        assert!(matches!(
            post_pooling_level_count(&levels, Some(&weight), None, CatTarget::Mean, 10.0),
            Err(PbError::ShapeMismatch { .. })
        ));
    }

    // ===================================================================
    // P1 multi-channel: `ChannelAxis`/`JointCatAxis` (the shared joint-cell builder consumed
    // by explain.rs and scoring.rs).
    // ===================================================================

    fn joint_fixture_encoder(raw: u32, id: u8, levels: &[(&str, f32)]) -> (CatEncoder, BorderGrid) {
        let mut out: Vec<CatLevel> = levels
            .iter()
            .map(|&(label, encoding)| CatLevel {
                label: label.to_owned(),
                members: vec![label.to_owned()],
                encoding,
                bin: 0,
                weight: 1.0,
            })
            .collect();
        assign_fisher_bins(&mut out).unwrap();
        let enc = CatEncoder {
            raw: FeatureId(raw),
            id: TsEncodingId(id),
            levels: out,
            base: 0.0,
            config: TsConfig::default(),
        };
        let grid = enc.border_grid().unwrap();
        (enc, grid)
    }

    #[test]
    fn channel_axes_for_raw_finds_only_matching_categorical_axes_sorted_by_id() {
        let provenance = vec![
            AxisProvenance {
                raw: FeatureId(0),
                kind: AxisKind::Numeric,
            },
            AxisProvenance {
                raw: FeatureId(1),
                kind: AxisKind::CategoricalTS {
                    encoding: TsEncodingId(1),
                },
            },
            AxisProvenance {
                raw: FeatureId(1),
                kind: AxisKind::CategoricalTS {
                    encoding: TsEncodingId(0),
                },
            },
            AxisProvenance {
                raw: FeatureId(2),
                kind: AxisKind::CategoricalTS {
                    encoding: TsEncodingId(0),
                },
            },
        ];
        let channels = channel_axes_for_raw(&provenance, FeatureId(1));
        assert_eq!(
            channels,
            vec![
                ChannelAxis {
                    id: TsEncodingId(0),
                    model_axis: 2
                },
                ChannelAxis {
                    id: TsEncodingId(1),
                    model_axis: 1
                },
            ],
            "must be sorted by encoder id, not by model axis position"
        );
        assert_eq!(channel_axes_for_raw(&provenance, FeatureId(2)).len(), 1);
        assert!(channel_axes_for_raw(&provenance, FeatureId(99)).is_empty());
    }

    #[test]
    fn joint_cat_axis_unseen_level_resolves_to_a_valid_cell_not_an_error() {
        // The 3rd switchover bug (real-data CV-pruning): a fold's frozen encoder can be
        // scored against a level absent from ITS OWN fit (e.g. a CV fold's held rows). That
        // level encodes to `base` on every channel (`CatEncoder::encode_label`'s documented
        // fallback) -- built here with `base` chosen so the resulting tuple is GENUINELY
        // novel: it bins like "high" on the mean channel (100.0 > 11.0) but like "low" on
        // the count channel (0.05 < 0.2), a combination none of the 3 seen levels produce
        // (each seen level's own mean-bin and count-bin come from the SAME level, never a
        // cross-level mix).
        let level = |label: &str, encoding: f32| CatLevel {
            label: label.to_owned(),
            members: vec![label.to_owned()],
            encoding,
            bin: 0,
            weight: 1.0,
        };
        let mut mean_levels = vec![level("low", 1.0), level("mid", 5.0), level("high", 11.0)];
        assign_fisher_bins(&mut mean_levels).unwrap();
        let mean_enc = CatEncoder {
            raw: FeatureId(0),
            id: TsEncodingId(0),
            levels: mean_levels,
            base: 100.0,
            config: TsConfig::default(),
        };
        let mean_grid = mean_enc.border_grid().unwrap();

        let mut count_levels = vec![level("low", 0.2), level("mid", 0.5), level("high", 0.9)];
        assign_fisher_bins(&mut count_levels).unwrap();
        let count_enc = CatEncoder {
            raw: FeatureId(0),
            id: TsEncodingId(1),
            levels: count_levels,
            base: 0.05,
            config: TsConfig {
                target: CatTarget::Count,
                ..TsConfig::default()
            },
        };
        let count_grid = count_enc.border_grid().unwrap();

        let store = CatEncoderStore::from_encoders(vec![mean_enc, count_enc]);
        let grids = vec![mean_grid, count_grid];
        let channels = vec![
            ChannelAxis {
                id: TsEncodingId(0),
                model_axis: 0,
            },
            ChannelAxis {
                id: TsEncodingId(1),
                model_axis: 1,
            },
        ];
        let joint = JointCatAxis::build(FeatureId(0), channels, &grids, &store).unwrap();

        // 1 missing + 3 seen levels + 1 unseen/base cell = 5 total -- proves the unseen
        // tuple is genuinely novel, not a coincidental match with any seen level (which
        // would leave n_cells at 4 and make this test's proof toothless).
        assert_eq!(joint.n_cells, 5);

        let mean_map = store.get(TsEncodingId(0), FeatureId(0)).unwrap();
        let count_map = store.get(TsEncodingId(1), FeatureId(0)).unwrap();
        // An unseen label -- one that never appeared at fit time for THIS encoder --
        // encodes to base on both channels.
        let unseen_mean_bin = bin_value(mean_map.encode_label("never_seen"), &grids[0]).unwrap();
        let unseen_count_bin = bin_value(count_map.encode_label("never_seen"), &grids[1]).unwrap();
        let cell = joint
            .cell_for_channel_bins(&[unseen_mean_bin, unseen_count_bin])
            .expect("an unseen level's base tuple must resolve to a valid cell, not an error");
        assert_ne!(
            cell, 0,
            "the unseen-level cell must not be confused with the reserved missing cell"
        );
        for label in ["low", "mid", "high"] {
            let mb = bin_value(mean_map.encode_label(label), &grids[0]).unwrap();
            let cb = bin_value(count_map.encode_label(label), &grids[1]).unwrap();
            let seen_cell = joint.cell_for_channel_bins(&[mb, cb]).unwrap();
            assert_ne!(
                cell, seen_cell,
                "the unseen cell must not collide with level `{label}`'s own cell"
            );
        }
    }

    #[test]
    fn joint_cat_axis_basic_build_and_lookup_roundtrip() {
        // Two channels of raw feature 0, three distinct levels with distinct values on BOTH
        // channels, so each level should land in its own joint cell.
        let (mean_enc, mean_grid) =
            joint_fixture_encoder(0, 0, &[("low", 1.0), ("mid", 5.0), ("high", 11.0)]);
        let (count_enc, count_grid) =
            joint_fixture_encoder(0, 1, &[("low", 0.2), ("mid", 0.5), ("high", 0.9)]);
        let store = CatEncoderStore::from_encoders(vec![mean_enc, count_enc]);
        let grids = vec![mean_grid, count_grid];
        let channels = vec![
            ChannelAxis {
                id: TsEncodingId(0),
                model_axis: 0,
            },
            ChannelAxis {
                id: TsEncodingId(1),
                model_axis: 1,
            },
        ];
        let joint = JointCatAxis::build(FeatureId(0), channels, &grids, &store).unwrap();

        // 1 missing cell + 3 distinct levels.
        assert_eq!(joint.n_cells, 4);

        // Every level maps to its OWN cell, and round-trips through write_rep_bins back to a
        // tuple that resolves to the SAME cell.
        let mean_map = store.get(TsEncodingId(0), FeatureId(0)).unwrap();
        let count_map = store.get(TsEncodingId(1), FeatureId(0)).unwrap();
        let mut seen_cells = std::collections::BTreeSet::new();
        let mut expected_level_cells = Vec::new();
        for label in ["low", "mid", "high"] {
            let mean_bin = bin_value(mean_map.encode_label(label), &grids[0]).unwrap();
            let count_bin = bin_value(count_map.encode_label(label), &grids[1]).unwrap();
            let cell = joint.cell_for_channel_bins(&[mean_bin, count_bin]).unwrap();
            assert!(cell >= 1 && cell < joint.n_cells, "cell={cell}");
            assert!(seen_cells.insert(cell), "level `{label}` reused a cell");
            expected_level_cells.push((label.to_string(), cell));

            let mut rep = vec![0u8; 2];
            joint.write_rep_bins(cell as usize, &mut rep).unwrap();
            assert_eq!(rep, vec![mean_bin, count_bin]);
            assert_eq!(
                joint.rep_model_bin_for_axis(cell as usize, 0).unwrap(),
                mean_bin
            );
            assert_eq!(
                joint.rep_model_bin_for_axis(cell as usize, 1).unwrap(),
                count_bin
            );
        }
        assert_eq!(seen_cells.len(), 3);
        // `level_cells` (Piece B's label-export source) must agree with the independently
        // recomputed cell for every level, in the canonical channel's own (insertion) order.
        assert_eq!(joint.level_cells, expected_level_cells);

        // Missing on every channel -> the reserved cell 0.
        assert_eq!(joint.cell_for_channel_bins(&[0, 0]).unwrap(), 0);
        // Channels disagreeing on missingness is a consistency error, not a silent cell.
        let mean_bin_low = bin_value(mean_map.encode_label("low"), &grids[0]).unwrap();
        assert!(matches!(
            joint.cell_for_channel_bins(&[0, mean_bin_low]),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn joint_cat_axis_levels_sharing_every_channel_bin_share_one_cell() {
        // "b" and "c" have IDENTICAL values on both channels -> the model cannot distinguish
        // them, so they must share one joint cell (spec §6 edge case), not two.
        let (mean_enc, mean_grid) =
            joint_fixture_encoder(0, 0, &[("a", 1.0), ("b", 5.0), ("c", 5.0)]);
        let (count_enc, count_grid) =
            joint_fixture_encoder(0, 1, &[("a", 0.1), ("b", 0.7), ("c", 0.7)]);
        let store = CatEncoderStore::from_encoders(vec![mean_enc, count_enc]);
        let grids = vec![mean_grid, count_grid];
        let channels = vec![
            ChannelAxis {
                id: TsEncodingId(0),
                model_axis: 0,
            },
            ChannelAxis {
                id: TsEncodingId(1),
                model_axis: 1,
            },
        ];
        let joint = JointCatAxis::build(FeatureId(0), channels, &grids, &store).unwrap();
        // 1 missing + "a" alone + {"b","c"} sharing one cell = 3 total.
        assert_eq!(joint.n_cells, 3);

        let mean_map = store.get(TsEncodingId(0), FeatureId(0)).unwrap();
        let count_map = store.get(TsEncodingId(1), FeatureId(0)).unwrap();
        let cell_of = |label: &str| {
            let mb = bin_value(mean_map.encode_label(label), &grids[0]).unwrap();
            let cb = bin_value(count_map.encode_label(label), &grids[1]).unwrap();
            joint.cell_for_channel_bins(&[mb, cb]).unwrap()
        };
        assert_ne!(cell_of("a"), cell_of("b"));
        assert_eq!(cell_of("b"), cell_of("c"));

        // `level_cells` must list ALL THREE levels (never collapse "b"/"c" down to one entry,
        // per the team lead's Piece B clarification: every level appears, sharing levels just
        // repeat the same cell id) in canonical (mean channel) insertion order.
        assert_eq!(
            joint.level_cells,
            vec![
                ("a".to_string(), cell_of("a")),
                ("b".to_string(), cell_of("b")),
                ("c".to_string(), cell_of("c")),
            ]
        );
    }

    #[test]
    fn joint_cat_axis_rejects_mismatched_channel_level_partitions() {
        // Channel 0 has levels {a,b}; channel 1 has {a,c} — a paired mean+count fit must never
        // produce this (see `fit_count_encoder`'s doc), but this defensive check must catch it
        // if it ever happens rather than silently building a nonsensical joint axis.
        let (enc0, grid0) = joint_fixture_encoder(0, 0, &[("a", 1.0), ("b", 2.0)]);
        let (enc1, grid1) = joint_fixture_encoder(0, 1, &[("a", 0.1), ("c", 0.2)]);
        let store = CatEncoderStore::from_encoders(vec![enc0, enc1]);
        let grids = vec![grid0, grid1];
        let channels = vec![
            ChannelAxis {
                id: TsEncodingId(0),
                model_axis: 0,
            },
            ChannelAxis {
                id: TsEncodingId(1),
                model_axis: 1,
            },
        ];
        assert!(matches!(
            JointCatAxis::build(FeatureId(0), channels, &grids, &store),
            Err(PbError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn joint_cat_axis_requires_at_least_two_channels() {
        let (enc0, grid0) = joint_fixture_encoder(0, 0, &[("a", 1.0), ("b", 2.0)]);
        let store = CatEncoderStore::from_encoders(vec![enc0]);
        let grids = vec![grid0];
        let channels = vec![ChannelAxis {
            id: TsEncodingId(0),
            model_axis: 0,
        }];
        assert!(matches!(
            JointCatAxis::build(FeatureId(0), channels, &grids, &store),
            Err(PbError::Internal { .. })
        ));
    }
}
