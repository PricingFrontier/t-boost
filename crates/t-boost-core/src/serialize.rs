//! Inference & serialization (spec §2.6, §02.8 / §10): the versioned model wire
//! envelope, JSON/bincode helpers, validation-on-load, and the rating-table export
//! artifact.
//!
//! The binary path uses bincode 2.x's `encode_to_vec`/`decode_from_slice` with the
//! config **frozen** to `bincode::config::standard()`. `ModelDoc` is a plain nested
//! struct (NOT `#[serde(flatten)]`, which is incompatible with the non-self-describing
//! bincode round-trip).

use crate::cat::{channel_axes_for_raw, CatEncoderStore, JointCatAxis, RARE_LEVEL_LABEL};
use crate::data::bin::bin as bin_value;
use crate::data::{axes_for_raw, AxisProvenance, BorderGrid, FeatureId};
use crate::engine::{ExactnessMode, Model, ModelSchema, MultiClassModel};
use crate::error::PbError;
use crate::explain::{AxisId, FactoredBoxExport, FeatureSet, RefMeasure, TableBank};
use crate::loss::{Link, ObjectiveTag};
use crate::table_model::{MultiClassTableModel, TableModel};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A raw feature's representative model axis for schema lookups (`feature_names`/
/// `feature_kinds`, both axis-indexed — parallel to `provenance`). An ordinary raw feature
/// has exactly one axis, always correct here; a P1 multi-channel raw feature has more than
/// one, and this returns its FIRST (lowest model-axis-index) channel — the schema name shown
/// for the collapsed export is that channel's, a placeholder until Piece B (per-level label
/// export) gives multi-channel categoricals their own proper display name.
///
/// # Errors
/// [`PbError::ShapeMismatch`] if `raw` has no axis in `provenance` at all (a malformed model).
pub fn representative_axis_for_raw(
    provenance: &[AxisProvenance],
    raw: FeatureId,
) -> Result<usize, PbError> {
    axes_for_raw(provenance, raw)
        .first()
        .copied()
        .ok_or_else(|| PbError::ShapeMismatch {
            what: format!("raw feature {} has no axis in provenance", raw.0),
        })
}

/// The on-disk container format version (the envelope around `Model`). Bumped on any
/// wire-incompatible change to the *framing*; a load of a newer `format_version` is
/// rejected.
pub const FORMAT_VERSION: u32 = 1;

/// The NEWEST `Model`/`schema` wire-schema version this build emits or understands
/// (spec §2.6). A single monotone `u32` bumped on any wire-incompatible change.
///
/// v2: added the optional `Model::correction` cell-basis refit field (§G1). bincode is
/// positional, so appending a field is wire-breaking for older readers.
/// v3: oblivious trees may be deeper than [`crate::engine::LEGACY_MAX_DEPTH`] (the
/// `max_depth` lift), so a tree's leaf table may hold up to `2^MAX_DEPTH` values.
/// v4: the ORDER lift. Two independent contents move to it:
///   (a) a tree may use more than [`crate::engine::LEGACY_MAX_ORDER`] distinct raw
///       features. Like the depth lift this is the SAME encoding on a wider range —
///       `splits` is length-prefixed — so it is a reader-capability claim only.
///   (b) a factored effect's boxes became length-carrying (`Vec<f64>` corners,
///       `Vec<Vec<bool>>` masks) so one type can hold an order-3 and an order-4 effect.
///       This one is a genuine ENCODING change: the old `[f64; 8]`/`[Vec<bool>; 3]` were
///       fixed-size arrays, which bincode writes with no length prefix. **A `.bin` written
///       by a pre-order-lift build and carrying a factored effect cannot be read by this
///       build, and vice versa.** JSON is unaffected (a fixed array and a `Vec` render
///       identically), and a bank with no factored effect is unchanged in both.
/// v5: the HIGH-ORDER lift (`order-hi`, 2026-08-26) — `MAX_DEPTH` 6→8 and `MAX_ORDER` 4→8.
///     Purely a reader-capability claim on BOTH halves, and that is the whole point of the
///     v4 (b) work: because a factored box is ALREADY length-carrying, an order-5..8 effect
///     serializes through exactly the same code path as an order-4 one, with a longer
///     `p`/`low`. **No encoding changed at v5.** A v5 document is bytes a v4 reader can
///     frame correctly and must nonetheless refuse, because its own `MAX_ORDER`/`MAX_DEPTH`
///     cannot validate the content — which is precisely what the version gate is for.
///
/// **The stamp is the MINIMUM reader version the content requires, not this constant.**
/// A tree's leaf count is recoverable in-band from `splits.len()`, so v2 and v3 are the SAME
/// encoding restricted to different depth ranges, (a) above extends that to v4, and v5
/// extends it again: content that used no lift is stamped [`SCHEMA_VERSION_UNLIFTED`] and is
/// byte-for-byte what a pre-lift build wrote, which is why no knob invalidates artifacts
/// that did not use it. Only content that actually used a lift is stamped higher, and only
/// that content is (correctly, loudly) rejected by an older reader.
/// v6: the EXPOSURE-MARGINAL reference measure (2026-09-06) —
///     [`crate::explain::RefMeasure::ExposureMarginals`], a new enum variant appended after
///     `Joint`. bincode writes an enum as its discriminant, so every pre-existing variant
///     keeps its bytes and a bank purified under one of them is still stamped as before; only
///     a bank whose `w` IS the new variant carries a discriminant a v5 reader has never seen,
///     and only that bank takes the v6 stamp. No tree-model bytes changed at v6.
/// v7: BANDED tables (2026-09-26) — [`crate::explain::AxisId`] gained `band_of` (merged cell
///     -> band). An ENCODING change for every tables-only document: bincode is positional, so
///     every table axis now carries the option's tag, and **a tables `.bin` written before v7
///     cannot be read by this build, nor a v7 one by an older build.** JSON is unaffected in the
///     reading direction (`#[serde(default)]` loads a pre-v7 document). Tree-model bytes are
///     unchanged. Every tables-only document is therefore stamped at least v7.
pub const SCHEMA_VERSION: u32 = 7;

/// The minimum reader version for any tables-only document (see [`SCHEMA_VERSION`] v7).
pub const SCHEMA_VERSION_BANDED_TABLES: u32 = 7;

/// The minimum reader version for a bank purified under
/// [`crate::explain::RefMeasure::ExposureMarginals`] (see [`SCHEMA_VERSION`] v6).
pub const SCHEMA_VERSION_EXPOSURE_MEASURE: u32 = 6;

/// The minimum reader version for a tree deeper than [`crate::engine::LEGACY_MAX_DEPTH`]
/// (but within [`crate::engine::ORDER_LIFT_MAX_DEPTH`], and of order `<= 3`).
pub const SCHEMA_VERSION_DEPTH_LIFTED: u32 = 3;

/// The minimum reader version for content the ORDER lift first admitted: a tree of order 4,
/// or any bank carrying a factored effect (whose box encoding v4 changed).
pub const SCHEMA_VERSION_ORDER_LIFTED: u32 = 4;

/// The minimum reader version for content the HIGH-ORDER lift first admitted: a tree deeper
/// than [`crate::engine::ORDER_LIFT_MAX_DEPTH`], or of order above
/// [`crate::engine::ORDER_LIFT_MAX_ORDER`], or a bank carrying a factored effect of order
/// above [`crate::engine::ORDER_LIFT_MAX_ORDER`].
pub const SCHEMA_VERSION_HIGH_ORDER: u32 = 5;

/// The minimum reader version a purified bank's BYTES require.
///
/// Three rungs, and only the middle one is about an encoding:
///
///  * a bank with no factored effect and no interaction of order above
///    [`crate::engine::LEGACY_MAX_ORDER`] is [`SCHEMA_VERSION_UNLIFTED`] — it writes exactly
///    the bytes a pre-lift build wrote and still loads in one;
///  * order-4 content, or ANY factored effect, is [`SCHEMA_VERSION_ORDER_LIFTED`]: the
///    length-carrying box encoding is what v4 introduced (see [`SCHEMA_VERSION`] (b));
///  * an order-5..8 effect is [`SCHEMA_VERSION_HIGH_ORDER`]. Its bytes frame identically
///    under v4, so the stamp is the ONLY thing standing between a v4 reader and a rating
///    table whose arity it cannot represent. That makes this rung load-bearing rather than
///    cosmetic.
///
/// # Why both lists are scanned, not just `factored`
///
/// The first cut of this keyed the whole decision on `bank.factored`, on the reasoning that
/// [`crate::OverflowPolicy::Factored`] routes every support of order `>= 3` there. That
/// reasoning is sound for the DEFAULT policy and wrong in general: `OverflowPolicy::Error`
/// and `SparseFallback` are reachable from Python (`parse_table_budget`, and thence
/// `tables_json` / `rating_export`), and under either of them a high-order support whose
/// merged cube happens to fit `max_table_cells` is materialized as a DENSE `EffectTable`
/// with `factored` left empty. Eight low-cardinality axes give a `3^8 ≈ 6.5k`-cell cube,
/// comfortably under the 2M default — so the bank that most needs the stamp would have been
/// stamped `2`, and [`crate::explain::TableBank::to_rating_export`] would have handed a
/// filing consumer an eight-axis table under a "version 2" label. That is precisely the
/// failure the rating-export stamp exists to prevent, so the arity is read off the CONTENT
/// wherever the content lives.
#[must_use]
pub fn required_tables_version(bank: &crate::explain::TableBank) -> u32 {
    // v7 changed the axis encoding of EVERY tables document (see `SCHEMA_VERSION`), so it
    // floors every content rung below.
    content_tables_version(bank).max(SCHEMA_VERSION_BANDED_TABLES)
}

/// The content rungs of [`required_tables_version`] before the v7 encoding floor: what the
/// bank's CONTENT (measure, arity, factored boxes) would require on its own.
#[must_use]
pub fn content_tables_version(bank: &crate::explain::TableBank) -> u32 {
    // The measure stamp outranks every content rung below it: an `ExposureMarginals` bank is
    // unreadable by any pre-v6 build whatever its arity (the enum discriminant is new).
    if matches!(bank.w, crate::explain::RefMeasure::ExposureMarginals { .. }) {
        return SCHEMA_VERSION_EXPOSURE_MEASURE;
    }
    let max_order = bank
        .tables
        .iter()
        .map(|t| t.u.order())
        .chain(bank.factored.iter().map(|f| f.u.order()))
        .max()
        .unwrap_or(0);
    if max_order > crate::engine::ORDER_LIFT_MAX_ORDER {
        return SCHEMA_VERSION_HIGH_ORDER;
    }
    // A factored effect is a v4 ENCODING regardless of its arity, so it takes the rung even
    // at order 3 — which is what the pre-existing `!factored.is_empty()` rule already said.
    if max_order > crate::engine::LEGACY_MAX_ORDER || !bank.factored.is_empty() {
        return SCHEMA_VERSION_ORDER_LIFTED;
    }
    SCHEMA_VERSION_UNLIFTED
}

/// The stamp for a model that uses nothing newer than the pre-lift wire schema — i.e.
/// every tables-only document, and every tree model whose trees are all at most
/// [`crate::engine::LEGACY_MAX_DEPTH`] deep. See [`SCHEMA_VERSION`].
pub const SCHEMA_VERSION_UNLIFTED: u32 = 2;

/// Shared schema-version gate: the stamp must be one this build understands AND at least
/// the minimum the document's own contents require, so a lifted model cannot masquerade
/// as an unlifted one (a hand-edited or corrupt stamp fails closed rather than silently
/// mis-loading).
fn check_schema_version(stamped: u32, required: u32, what: &str) -> Result<(), PbError> {
    if !(SCHEMA_VERSION_UNLIFTED..=SCHEMA_VERSION).contains(&stamped) {
        return Err(PbError::Serialization(format!(
            "unsupported {what} schema_version {stamped} (this build supports \
             {SCHEMA_VERSION_UNLIFTED}..={SCHEMA_VERSION}); {VERSION_MISMATCH_GUIDANCE}"
        )));
    }
    if stamped < required {
        return Err(PbError::Serialization(format!(
            "{what} schema_version {stamped} is below the {required} its contents require \
             (a tree deeper than the legacy cap needs schema_version {SCHEMA_VERSION})"
        )));
    }
    Ok(())
}

/// The serialized envelope (spec §02.8): a plain nested `{ format_version,
/// schema_version, model }`. No `#[serde(flatten)]` — it does not round-trip through
/// non-self-describing bincode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelDoc {
    /// The container framing version (see [`FORMAT_VERSION`]).
    pub format_version: u32,
    /// The model/schema wire version (see [`SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// The wrapped trained model.
    pub model: Model,
}

impl ModelDoc {
    /// Wrap `model` in a MINIMUM-version envelope (see [`SCHEMA_VERSION`]): an unlifted
    /// model is stamped [`SCHEMA_VERSION_UNLIFTED`] so its bytes stay loadable by
    /// pre-lift readers and byte-identical to what they would have written.
    #[must_use]
    pub fn new(mut model: Model) -> Self {
        // The envelope and the model carry independent copies of the stamp and
        // `migrate_schema_json` fails closed if they drift, so normalize both here.
        let schema_version = model.required_schema_version();
        model.schema_version = schema_version;
        Self {
            format_version: FORMAT_VERSION,
            schema_version,
            model,
        }
    }
}

fn serialization_error(e: impl ToString) -> PbError {
    PbError::Serialization(e.to_string())
}

/// Version-mismatch remedy shared across this module's error messages (README
/// "Serialization compatibility", spec §02.8): a JSON [`ModelDoc`] from a known older
/// `schema_version` migrates forward automatically on load (see [`migrate_schema_json`]);
/// bincode requires an EXACT `schema_version`/`format_version` match, because its
/// positional, non-self-describing wire format cannot distinguish an old schema from a
/// corrupt one -- there is no field-name self-description to skip a since-added field by.
/// `format_version` itself has no historical shims yet (nothing has bumped it), so it
/// stays exact-match on both formats. When neither format can resolve a mismatch, the
/// remedy is the same either way.
const VERSION_MISMATCH_GUIDANCE: &str =
    "pin the t-boost version that produced this artifact, or re-export it (refit or \
     re-save) under this build";

fn validate_doc(doc: &ModelDoc) -> Result<(), PbError> {
    if doc.format_version != FORMAT_VERSION {
        return Err(PbError::Serialization(format!(
            "unsupported format_version {} (this build supports exactly {FORMAT_VERSION}); {VERSION_MISMATCH_GUIDANCE}",
            doc.format_version
        )));
    }
    // The envelope and the model carry INDEPENDENT copies of the stamp. Before the lift both
    // had to equal `SCHEMA_VERSION`, which forced them equal; the band check alone would let a
    // document whose two stamps have drifted apart (corruption, not a genuine older artifact)
    // load clean. `migrate_schema_json` already fails closed on that for JSON — do the same here.
    if doc.schema_version != doc.model.schema_version {
        return Err(PbError::Serialization(format!(
            "envelope schema_version {} disagrees with model schema_version {}",
            doc.schema_version, doc.model.schema_version
        )));
    }
    check_schema_version(
        doc.schema_version,
        doc.model.required_schema_version(),
        "model",
    )?;
    doc.model.validate()
}

/// Encode a [`ModelDoc`] to the fast binary format (bincode 2.x, frozen
/// `config::standard()`).
///
/// # Errors
/// [`PbError::Serialization`] if encoding fails.
pub fn encode_doc(doc: &ModelDoc) -> Result<Vec<u8>, PbError> {
    bincode::serde::encode_to_vec(doc, bincode::config::standard()).map_err(serialization_error)
}

/// Decode a [`ModelDoc`] from the fast binary format, rejecting a newer
/// `format_version` (spec §02.8 version gate).
///
/// # Errors
/// [`PbError::Serialization`] if decoding fails or the framing version is unknown.
pub fn decode_doc(bytes: &[u8]) -> Result<ModelDoc, PbError> {
    let (doc, len): (ModelDoc, usize) =
        bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .map_err(serialization_error)?;
    if len != bytes.len() {
        return Err(PbError::Serialization(format!(
            "trailing bytes after ModelDoc: decoded {len}, input {}",
            bytes.len()
        )));
    }
    validate_doc(&doc)?;
    Ok(doc)
}

/// Encode a [`ModelDoc`] to the canonical pretty JSON format.
///
/// # Errors
/// [`PbError::Serialization`] if encoding fails.
pub fn encode_doc_json(doc: &ModelDoc) -> Result<String, PbError> {
    serde_json::to_string_pretty(doc).map_err(serialization_error)
}

/// Decode a [`ModelDoc`] from canonical JSON, migrating a known older `schema_version`
/// forward to current (see [`migrate_schema_json`]) before validating.
///
/// # Errors
/// [`PbError::Serialization`] if decoding fails or versions are unsupported; plus
/// model validation errors.
pub fn decode_doc_json(s: &str) -> Result<ModelDoc, PbError> {
    let mut doc: ModelDoc = serde_json::from_str(s).map_err(serialization_error)?;
    migrate_schema_json(&mut doc)?;
    validate_doc(&doc)?;
    Ok(doc)
}

/// Upgrade `doc` in place from its recorded `schema_version` to [`SCHEMA_VERSION`],
/// applying each historical shim in order and re-stamping the version on both the
/// envelope and the wrapped `Model` (which carries an independent copy, re-checked by
/// [`Model::validate`]). Each shim step first confirms the envelope and model versions
/// still agree, so a document where they've drifted apart (corruption, not a genuine
/// older artifact) fails closed instead of being silently "corrected".
///
/// JSON-only (spec §02.8's From-based migration contract applies to the self-describing
/// envelope): bincode's positional, non-self-describing wire format can't tell an
/// old-schema document apart from a corrupt one, so [`decode_doc`] keeps the strict
/// same-version gate via [`validate_doc`] and never calls this.
///
/// # Errors
/// [`PbError::Serialization`] if `doc.schema_version` is newer than this build, has no
/// registered migration shim, or disagrees with the model's own `schema_version`.
fn migrate_schema_json(doc: &mut ModelDoc) -> Result<(), PbError> {
    if doc.schema_version > SCHEMA_VERSION {
        return Err(PbError::Serialization(format!(
            "unsupported schema_version {} (this build supports exactly {SCHEMA_VERSION}); {VERSION_MISMATCH_GUIDANCE}",
            doc.schema_version
        )));
    }
    let target = doc.model.required_schema_version();
    while doc.schema_version < target {
        if doc.model.schema_version != doc.schema_version {
            return Err(PbError::Serialization(format!(
                "envelope schema_version {} disagrees with model schema_version {}",
                doc.schema_version, doc.model.schema_version
            )));
        }
        doc.schema_version = match doc.schema_version {
            // v1 -> v2: `Model.correction` (§G1 cell-basis refit) was added with
            // `#[serde(default)]`, so a v1 document already deserializes with
            // `correction: None` -- only the version stamp needs to move.
            1 => 2,
            v => {
                return Err(PbError::Serialization(format!(
                    "no migration path registered from schema_version {v} to {target}; {VERSION_MISMATCH_GUIDANCE}"
                )))
            }
        };
        doc.model.schema_version = doc.schema_version;
    }
    Ok(())
}

/// Migrate a JSON [`ModelDoc`] value from `(from_format_version, from_schema_version)` to
/// the current model.
///
/// There are no historical `format_version`s before [`FORMAT_VERSION`] yet, so the only
/// accepted format migration today is the identity migration; `schema_version` is routed
/// through [`migrate_schema_json`], the same shim chain [`decode_doc_json`] uses. The
/// explicit facade is still useful: a caller's declared source version is cross-checked
/// against the document's own (so a mis-declared source fails closed rather than
/// migrating the wrong thing), and future version shims have a single public entry point.
///
/// # Errors
/// [`PbError::Serialization`] if `from_format_version`/`from_schema_version` is newer
/// than this build, has no registered migration path, disagrees with the document's own
/// versions, or if the migrated model fails validation.
pub fn migrate(
    value: serde_json::Value,
    from_format_version: u32,
    from_schema_version: u32,
) -> Result<Model, PbError> {
    if from_format_version > FORMAT_VERSION {
        return Err(PbError::Serialization(format!(
            "cannot migrate future format_version {from_format_version}; this build supports {FORMAT_VERSION}"
        )));
    }
    if from_format_version != FORMAT_VERSION {
        return Err(PbError::Serialization(format!(
            "no migration path registered from format_version {from_format_version} to {FORMAT_VERSION}"
        )));
    }
    let mut doc: ModelDoc = serde_json::from_value(value).map_err(serialization_error)?;
    if doc.format_version != from_format_version {
        return Err(PbError::Serialization(format!(
            "document format_version {} disagrees with requested migration source {from_format_version}",
            doc.format_version
        )));
    }
    if doc.schema_version != from_schema_version {
        return Err(PbError::Serialization(format!(
            "document schema_version {} disagrees with requested migration source {from_schema_version}",
            doc.schema_version
        )));
    }
    migrate_schema_json(&mut doc)?;
    validate_doc(&doc)?;
    Ok(doc.model)
}

/// Encode a bare [`Model`] (wrapped in a current-version [`ModelDoc`]) to binary.
///
/// # Errors
/// [`PbError::Serialization`] if encoding fails.
pub fn encode_model(model: &Model) -> Result<Vec<u8>, PbError> {
    model.validate()?;
    encode_doc(&ModelDoc::new(model.clone()))
}

/// Decode a bare [`Model`] from binary, applying the version gate.
///
/// # Errors
/// [`PbError::Serialization`] if decoding fails or a version is unsupported.
pub fn decode_model(bytes: &[u8]) -> Result<Model, PbError> {
    Ok(decode_doc(bytes)?.model)
}

/// Encode a bare [`Model`] to canonical JSON.
///
/// # Errors
/// [`PbError::Serialization`] if encoding fails; plus validation errors.
pub fn encode_model_json(model: &Model) -> Result<String, PbError> {
    model.validate()?;
    encode_doc_json(&ModelDoc::new(model.clone()))
}

/// Decode a bare [`Model`] from canonical JSON.
///
/// # Errors
/// [`PbError::Serialization`] if decoding fails or versions are unsupported; plus
/// model validation errors.
pub fn decode_model_json(s: &str) -> Result<Model, PbError> {
    Ok(decode_doc_json(s)?.model)
}

/// The container framing version for a [`MultiClassModel`] envelope (native-softmax multiclass).
pub const MULTICLASS_FORMAT_VERSION: u32 = 1;

/// Discriminator recorded in a [`MultiClassDoc`] so a reader can tell a multiclass envelope from a
/// single-model one before decoding (esp. on the JSON path).
pub const MULTICLASS_KIND: &str = "t-boost-multiclass";

/// Four-byte magic prefixing the BINARY multiclass envelope, so a byte reader can distinguish it
/// from a single-[`Model`] blob without a trial decode.
const MULTICLASS_MAGIC: &[u8; 4] = b"TBMC";

/// Serialized envelope for a [`MultiClassModel`]. Distinct from [`ModelDoc`] so single-output
/// models keep their exact wire format; each contained per-class [`Model`] uses the unchanged
/// `Model` wire schema (no [`SCHEMA_VERSION`] bump).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiClassDoc {
    /// Fixed discriminator [`MULTICLASS_KIND`].
    pub kind: String,
    /// Container framing version (see [`MULTICLASS_FORMAT_VERSION`]).
    pub format_version: u32,
    /// The per-class model/schema wire version (see [`SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// The wrapped trained multiclass model.
    pub model: MultiClassModel,
}

impl MultiClassDoc {
    /// Wrap `model` in a current-version envelope.
    #[must_use]
    pub fn new(mut model: MultiClassModel) -> Self {
        let schema_version = model.required_schema_version();
        model.schema_version = schema_version;
        for class in &mut model.classes {
            class.schema_version = schema_version;
        }
        Self {
            kind: MULTICLASS_KIND.to_string(),
            format_version: MULTICLASS_FORMAT_VERSION,
            schema_version,
            model,
        }
    }
}

fn validate_multiclass_doc(doc: &MultiClassDoc) -> Result<(), PbError> {
    if doc.kind != MULTICLASS_KIND {
        return Err(PbError::Serialization(format!(
            "not a multiclass envelope: kind {:?} != {MULTICLASS_KIND:?}",
            doc.kind
        )));
    }
    if doc.format_version != MULTICLASS_FORMAT_VERSION {
        return Err(PbError::Serialization(format!(
            "unsupported multiclass format_version {} (this build supports exactly {MULTICLASS_FORMAT_VERSION})",
            doc.format_version
        )));
    }
    if doc.schema_version != doc.model.schema_version {
        return Err(PbError::Serialization(format!(
            "multiclass envelope schema_version {} disagrees with model schema_version {}",
            doc.schema_version, doc.model.schema_version
        )));
    }
    check_schema_version(
        doc.schema_version,
        doc.model.required_schema_version(),
        "multiclass",
    )?;
    doc.model.validate()
}

/// Encode a [`MultiClassModel`] to the fast binary format, prefixed with [`MULTICLASS_MAGIC`] so a
/// reader can distinguish it from a single-[`Model`] blob.
///
/// # Errors
/// [`PbError::Serialization`] if encoding fails; plus model validation errors.
pub fn encode_multiclass(model: &MultiClassModel) -> Result<Vec<u8>, PbError> {
    model.validate()?;
    let doc = MultiClassDoc::new(model.clone());
    let body = bincode::serde::encode_to_vec(&doc, bincode::config::standard())
        .map_err(serialization_error)?;
    let mut out = Vec::with_capacity(MULTICLASS_MAGIC.len().saturating_add(body.len()));
    out.extend_from_slice(MULTICLASS_MAGIC);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decode a [`MultiClassModel`] from the magic-prefixed binary format, applying the version gate.
///
/// # Errors
/// [`PbError::Serialization`] if the magic is absent, decoding fails, a version is unsupported, or
/// trailing bytes remain; plus model validation errors.
pub fn decode_multiclass(bytes: &[u8]) -> Result<MultiClassModel, PbError> {
    let body = bytes.strip_prefix(MULTICLASS_MAGIC).ok_or_else(|| {
        PbError::Serialization("not a t-boost multiclass model (missing magic prefix)".into())
    })?;
    let (doc, len): (MultiClassDoc, usize) =
        bincode::serde::decode_from_slice(body, bincode::config::standard())
            .map_err(serialization_error)?;
    if len != body.len() {
        return Err(PbError::Serialization(format!(
            "trailing bytes after MultiClassDoc: decoded {len}, body {}",
            body.len()
        )));
    }
    validate_multiclass_doc(&doc)?;
    Ok(doc.model)
}

/// `true` iff `bytes` begins with the multiclass magic prefix (a single-[`Model`] blob does not).
#[must_use]
pub fn is_multiclass_bytes(bytes: &[u8]) -> bool {
    bytes.starts_with(MULTICLASS_MAGIC)
}

/// Encode a [`MultiClassModel`] to canonical pretty JSON (carries the [`MULTICLASS_KIND`]
/// discriminator).
///
/// # Errors
/// [`PbError::Serialization`] if encoding fails; plus model validation errors.
pub fn encode_multiclass_json(model: &MultiClassModel) -> Result<String, PbError> {
    model.validate()?;
    serde_json::to_string_pretty(&MultiClassDoc::new(model.clone())).map_err(serialization_error)
}

/// Decode a [`MultiClassModel`] from canonical JSON and validate it.
///
/// # Errors
/// [`PbError::Serialization`] if decoding fails or a version/kind is unsupported; plus model
/// validation errors.
pub fn decode_multiclass_json(s: &str) -> Result<MultiClassModel, PbError> {
    let doc: MultiClassDoc = serde_json::from_str(s).map_err(serialization_error)?;
    validate_multiclass_doc(&doc)?;
    Ok(doc.model)
}

/// The container framing version for a [`TableModel`] envelope (tables-only served model).
pub const TABLES_FORMAT_VERSION: u32 = 1;

/// Discriminator recorded in a [`TablesDoc`] so a reader can tell a tables-only envelope from a
/// single-`Model` or multiclass one before decoding (esp. on the JSON path).
pub const TABLES_KIND: &str = "t-boost-tables";

/// Four-byte magic prefixing the BINARY tables-only envelope, so a byte reader can distinguish it
/// from a single-[`Model`] blob (no magic) or a multiclass one ([`MULTICLASS_MAGIC`]).
const TABLES_MAGIC: &[u8; 4] = b"TBTM";

/// Serialized envelope for a [`TableModel`]. Distinct from [`ModelDoc`] so single-output tree
/// models keep their exact wire format; the contained bank uses the unchanged [`TableBank`] wire
/// representation (no [`SCHEMA_VERSION`] bump).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TablesDoc {
    /// Fixed discriminator [`TABLES_KIND`].
    pub kind: String,
    /// Container framing version (see [`TABLES_FORMAT_VERSION`]).
    pub format_version: u32,
    /// The model/schema wire version (see [`SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// The wrapped tables-only model.
    pub model: TableModel,
}

impl TablesDoc {
    /// Wrap `model` in a current-version envelope.
    #[must_use]
    pub fn new(model: TableModel) -> Self {
        Self {
            kind: TABLES_KIND.to_string(),
            format_version: TABLES_FORMAT_VERSION,
            // Tables-only documents carry no trees, so the DEPTH lift can never move their
            // required version — but the ORDER lift can, through the factored box encoding.
            schema_version: required_tables_version(&model.bank),
            model,
        }
    }
}

fn validate_tables_doc(doc: &TablesDoc) -> Result<(), PbError> {
    if doc.kind != TABLES_KIND {
        return Err(PbError::Serialization(format!(
            "not a tables-only envelope: kind {:?} != {TABLES_KIND:?}",
            doc.kind
        )));
    }
    if doc.format_version != TABLES_FORMAT_VERSION {
        return Err(PbError::Serialization(format!(
            "unsupported tables format_version {} (this build supports exactly {TABLES_FORMAT_VERSION})",
            doc.format_version
        )));
    }
    check_schema_version(
        doc.schema_version,
        required_tables_version(&doc.model.bank),
        "tables",
    )?;
    doc.model.validate()
}

/// Encode a [`TableModel`] to the fast binary format, prefixed with the tables magic so a reader
/// can distinguish it from a single-[`Model`] or multiclass blob.
///
/// # Errors
/// [`PbError::Serialization`] if encoding fails; plus model validation errors.
pub fn encode_tables(model: &TableModel) -> Result<Vec<u8>, PbError> {
    model.validate()?;
    let doc = TablesDoc::new(model.clone());
    let body = bincode::serde::encode_to_vec(&doc, bincode::config::standard())
        .map_err(serialization_error)?;
    let mut out = Vec::with_capacity(TABLES_MAGIC.len().saturating_add(body.len()));
    out.extend_from_slice(TABLES_MAGIC);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decode a [`TableModel`] from the magic-prefixed binary format, applying the version gate.
///
/// # Errors
/// [`PbError::Serialization`] if the magic is absent, decoding fails, a version is unsupported, or
/// trailing bytes remain; plus model validation errors.
pub fn decode_tables(bytes: &[u8]) -> Result<TableModel, PbError> {
    let body = bytes.strip_prefix(TABLES_MAGIC).ok_or_else(|| {
        PbError::Serialization("not a t-boost tables-only model (missing magic prefix)".into())
    })?;
    let (doc, len): (TablesDoc, usize) =
        bincode::serde::decode_from_slice(body, bincode::config::standard())
            .map_err(serialization_error)?;
    if len != body.len() {
        return Err(PbError::Serialization(format!(
            "trailing bytes after TablesDoc: decoded {len}, body {}",
            body.len()
        )));
    }
    validate_tables_doc(&doc)?;
    Ok(doc.model)
}

/// `true` iff `bytes` begins with the tables-only magic prefix.
#[must_use]
pub fn is_tables_bytes(bytes: &[u8]) -> bool {
    bytes.starts_with(TABLES_MAGIC)
}

/// Encode a [`TableModel`] to canonical pretty JSON (carries the [`TABLES_KIND`] discriminator).
///
/// # Errors
/// [`PbError::Serialization`] if encoding fails; plus model validation errors.
pub fn encode_tables_json(model: &TableModel) -> Result<String, PbError> {
    model.validate()?;
    serde_json::to_string_pretty(&TablesDoc::new(model.clone())).map_err(serialization_error)
}

/// Decode a [`TableModel`] from canonical JSON and validate it.
///
/// # Errors
/// [`PbError::Serialization`] if decoding fails or a version/kind is unsupported; plus model
/// validation errors.
pub fn decode_tables_json(s: &str) -> Result<TableModel, PbError> {
    let doc: TablesDoc = serde_json::from_str(s).map_err(serialization_error)?;
    validate_tables_doc(&doc)?;
    Ok(doc.model)
}

impl TableModel {
    /// Serialize to the compact magic-prefixed bincode format.
    ///
    /// # Errors
    /// [`PbError::Serialization`] if encoding fails; plus validation errors.
    pub fn to_bincode(&self) -> Result<Vec<u8>, PbError> {
        encode_tables(self)
    }

    /// Deserialize from the compact magic-prefixed bincode format.
    ///
    /// # Errors
    /// [`PbError::Serialization`] if decoding fails or versions are unsupported; plus validation.
    pub fn from_bincode(bytes: &[u8]) -> Result<Self, PbError> {
        decode_tables(bytes)
    }

    /// Serialize to canonical JSON.
    ///
    /// # Errors
    /// [`PbError::Serialization`] if encoding fails; plus validation errors.
    pub fn to_json(&self) -> Result<String, PbError> {
        encode_tables_json(self)
    }

    /// Deserialize from canonical JSON.
    ///
    /// # Errors
    /// [`PbError::Serialization`] if decoding fails or versions are unsupported; plus validation.
    pub fn from_json(s: &str) -> Result<Self, PbError> {
        decode_tables_json(s)
    }
}

/// Container framing version for a [`MultiClassTableModel`] envelope (pruned multiclass tables model).
pub const MULTICLASS_TABLES_FORMAT_VERSION: u32 = 1;

/// Discriminator recorded in a [`MultiClassTablesDoc`].
pub const MULTICLASS_TABLES_KIND: &str = "t-boost-multiclass-tables";

/// Four-byte magic prefixing the BINARY multiclass-tables envelope.
const MULTICLASS_TABLES_MAGIC: &[u8; 4] = b"TBMT";

/// Serialized envelope for a [`MultiClassTableModel`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiClassTablesDoc {
    /// Fixed discriminator [`MULTICLASS_TABLES_KIND`].
    pub kind: String,
    /// Container framing version (see [`MULTICLASS_TABLES_FORMAT_VERSION`]).
    pub format_version: u32,
    /// The model/schema wire version (see [`SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// The wrapped tables-only multiclass model.
    pub model: MultiClassTableModel,
}

impl MultiClassTablesDoc {
    /// Wrap `model` in a current-version envelope.
    #[must_use]
    pub fn new(model: MultiClassTableModel) -> Self {
        Self {
            kind: MULTICLASS_TABLES_KIND.to_string(),
            format_version: MULTICLASS_TABLES_FORMAT_VERSION,
            // The container takes the MAX over its per-class banks, never the first's.
            schema_version: model
                .classes
                .iter()
                .map(|c| required_tables_version(&c.bank))
                .max()
                .unwrap_or(SCHEMA_VERSION_UNLIFTED),
            model,
        }
    }
}

fn validate_multiclass_tables_doc(doc: &MultiClassTablesDoc) -> Result<(), PbError> {
    if doc.kind != MULTICLASS_TABLES_KIND {
        return Err(PbError::Serialization(format!(
            "not a multiclass-tables envelope: kind {:?} != {MULTICLASS_TABLES_KIND:?}",
            doc.kind
        )));
    }
    if doc.format_version != MULTICLASS_TABLES_FORMAT_VERSION {
        return Err(PbError::Serialization(format!(
            "unsupported multiclass-tables format_version {} (this build supports {MULTICLASS_TABLES_FORMAT_VERSION})",
            doc.format_version
        )));
    }
    check_schema_version(
        doc.schema_version,
        doc.model
            .classes
            .iter()
            .map(|c| required_tables_version(&c.bank))
            .max()
            .unwrap_or(SCHEMA_VERSION_UNLIFTED),
        "tables",
    )?;
    doc.model.validate()
}

/// Encode a [`MultiClassTableModel`] to the magic-prefixed binary format.
///
/// # Errors
/// [`PbError::Serialization`] if encoding fails; plus validation errors.
pub fn encode_multiclass_tables(model: &MultiClassTableModel) -> Result<Vec<u8>, PbError> {
    model.validate()?;
    let doc = MultiClassTablesDoc::new(model.clone());
    let body = bincode::serde::encode_to_vec(&doc, bincode::config::standard())
        .map_err(serialization_error)?;
    let mut out = Vec::with_capacity(MULTICLASS_TABLES_MAGIC.len().saturating_add(body.len()));
    out.extend_from_slice(MULTICLASS_TABLES_MAGIC);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Decode a [`MultiClassTableModel`] from the magic-prefixed binary format.
///
/// # Errors
/// [`PbError::Serialization`] if the magic is absent, decoding fails, a version is unsupported, or
/// trailing bytes remain; plus validation errors.
pub fn decode_multiclass_tables(bytes: &[u8]) -> Result<MultiClassTableModel, PbError> {
    let body = bytes.strip_prefix(MULTICLASS_TABLES_MAGIC).ok_or_else(|| {
        PbError::Serialization(
            "not a t-boost multiclass-tables model (missing magic prefix)".into(),
        )
    })?;
    let (doc, len): (MultiClassTablesDoc, usize) =
        bincode::serde::decode_from_slice(body, bincode::config::standard())
            .map_err(serialization_error)?;
    if len != body.len() {
        return Err(PbError::Serialization(format!(
            "trailing bytes after MultiClassTablesDoc: decoded {len}, body {}",
            body.len()
        )));
    }
    validate_multiclass_tables_doc(&doc)?;
    Ok(doc.model)
}

/// `true` iff `bytes` begins with the multiclass-tables magic prefix.
#[must_use]
pub fn is_multiclass_tables_bytes(bytes: &[u8]) -> bool {
    bytes.starts_with(MULTICLASS_TABLES_MAGIC)
}

/// Encode a [`MultiClassTableModel`] to canonical pretty JSON.
///
/// # Errors
/// [`PbError::Serialization`] if encoding fails; plus validation errors.
pub fn encode_multiclass_tables_json(model: &MultiClassTableModel) -> Result<String, PbError> {
    model.validate()?;
    serde_json::to_string_pretty(&MultiClassTablesDoc::new(model.clone()))
        .map_err(serialization_error)
}

/// Decode a [`MultiClassTableModel`] from canonical JSON.
///
/// # Errors
/// [`PbError::Serialization`] if decoding fails or a version/kind is unsupported; plus validation.
pub fn decode_multiclass_tables_json(s: &str) -> Result<MultiClassTableModel, PbError> {
    let doc: MultiClassTablesDoc = serde_json::from_str(s).map_err(serialization_error)?;
    validate_multiclass_tables_doc(&doc)?;
    Ok(doc.model)
}

impl MultiClassTableModel {
    /// Serialize to the compact magic-prefixed bincode format.
    ///
    /// # Errors
    /// [`PbError::Serialization`] if encoding fails; plus validation errors.
    pub fn to_bincode(&self) -> Result<Vec<u8>, PbError> {
        encode_multiclass_tables(self)
    }

    /// Deserialize from the compact magic-prefixed bincode format.
    ///
    /// # Errors
    /// [`PbError::Serialization`] if decoding fails or versions are unsupported; plus validation.
    pub fn from_bincode(bytes: &[u8]) -> Result<Self, PbError> {
        decode_multiclass_tables(bytes)
    }

    /// Serialize to canonical JSON.
    ///
    /// # Errors
    /// [`PbError::Serialization`] if encoding fails; plus validation errors.
    pub fn to_json(&self) -> Result<String, PbError> {
        encode_multiclass_tables_json(self)
    }

    /// Deserialize from canonical JSON.
    ///
    /// # Errors
    /// [`PbError::Serialization`] if decoding fails or versions are unsupported; plus validation.
    pub fn from_json(s: &str) -> Result<Self, PbError> {
        decode_multiclass_tables_json(s)
    }
}

impl Model {
    /// Serialize this model to the compact same-version bincode cache format.
    ///
    /// # Errors
    /// [`PbError::Serialization`] if encoding fails; plus validation errors.
    pub fn to_bincode(&self) -> Result<Vec<u8>, PbError> {
        encode_model(self)
    }

    /// Deserialize a model from the compact same-version bincode cache format.
    ///
    /// # Errors
    /// [`PbError::Serialization`] if decoding fails or versions are unsupported; plus
    /// model validation errors.
    pub fn from_bincode(bytes: &[u8]) -> Result<Self, PbError> {
        decode_model(bytes)
    }

    /// Serialize this model to canonical pretty JSON.
    ///
    /// # Errors
    /// [`PbError::Serialization`] if encoding fails; plus validation errors.
    pub fn to_json(&self) -> Result<String, PbError> {
        encode_model_json(self)
    }

    /// Deserialize a model from canonical JSON.
    ///
    /// # Errors
    /// [`PbError::Serialization`] if decoding fails or versions are unsupported; plus
    /// model validation errors.
    pub fn from_json(s: &str) -> Result<Self, PbError> {
        decode_model_json(s)
    }
}

/// One reference-cell selector: a table support (sorted distinct raw feature ids) and
/// the merged-cell coordinate within that table that should read as neutral.
///
/// A flat struct of `Vec<u32>` fields (rather than a `FeatureSet`-keyed map) so the
/// enclosing [`RatingBasis`] round-trips through JSON — a `FeatureSet` (a sequence) is
/// not a valid JSON object key, so the former `BTreeMap<FeatureSet, _>` could not be
/// authored as `basis_json` through the public Python API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RatingReference {
    /// The table support, as sorted distinct raw feature ids.
    pub feature_set: Vec<u32>,
    /// The reference coordinate within that table (one merged-cell id per axis).
    pub coord: Vec<u32>,
}

/// Optional reference-cell selector for rating-view exports. Each entry identifies a
/// table support and the coordinate within that table that should read as neutral
/// (`0.0` in score space, `1.000` as a log-link relativity). The shifted mass is
/// folded into the exported intercept, so reconstructed scores are unchanged.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RatingBasis {
    /// Per-table reference coordinates. JSON-representable (a sequence of entries, not a
    /// non-string-keyed map).
    pub reference: Vec<RatingReference>,
}

impl RatingBasis {
    /// The reference coordinate for a table support `u`, matched by its sorted raw ids.
    #[must_use]
    pub fn coord_for(&self, u: &FeatureSet) -> Option<&[u32]> {
        self.reference.iter().find_map(|entry| {
            let matches = entry.feature_set.len() == u.0.len()
                && entry
                    .feature_set
                    .iter()
                    .zip(u.0.iter())
                    .all(|(&f, raw)| f == raw.0);
            if matches {
                Some(entry.coord.as_slice())
            } else {
                None
            }
        })
    }
}

/// One post-rare-pooling categorical level's export entry (Piece B, design/multichannel-
/// categoricals.md §4.4): a label paired with the cell it lands in, so a consumer can join
/// `label -> cell -> RatingTable::values[cell]` (or `relativities[cell]`) to read the level's
/// actual rating. Levels that share every channel's bin (P1 multi-channel, spec §6) correctly
/// each get their own entry pointing at the SAME cell — never collapsed away, since the
/// mapping itself (which labels exist) is separate information from how many DISTINCT cells
/// back it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatLevelExport {
    /// Human-readable level label. The reserved rare bucket is shown as `"<rare>"` here
    /// (export-only substitution — the stored model keeps its internal sentinel label).
    pub label: String,
    /// The axis cell this level lands in (indexes the same cell space as `AxisExport::cells`
    /// / the table's `values`/`relativities`/`support`).
    pub cell: u32,
}

/// One exported rating-table axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxisExport {
    /// Raw feature id.
    pub raw: u32,
    /// Human-readable feature name from [`ModelSchema`].
    pub name: String,
    /// Merged-grid finite borders.
    pub borders: Vec<f32>,
    /// Number of cells including the explicit missing cell.
    pub cells: u32,
    /// Per-level label -> cell mapping (Piece B, design/multichannel-categoricals.md §4.4),
    /// `Some` for a categorical axis (single- or multi-channel), `None` for a numeric one. A
    /// bare `Option`, NOT `skip_serializing_if` (see [`RatingTable::se_band`]'s doc for why —
    /// bincode is positional); `#[serde(default)]` keeps older JSON (from before this field
    /// existed) loading.
    #[serde(default)]
    pub levels: Option<Vec<CatLevelExport>>,
}

/// One exported purified rating table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RatingTable {
    /// Raw feature set this table represents.
    pub feature_set: FeatureSet,
    /// Feature names parallel to [`RatingTable::feature_set`].
    pub feature_names: Vec<String>,
    /// Axis metadata parallel to tensor dimensions.
    pub axes: Vec<AxisExport>,
    /// Tensor shape as fixed-width dimensions.
    pub shape: Vec<u32>,
    /// Score-space table values in dense row-major order.
    pub values: Vec<f64>,
    /// Log-link relativities (`exp(value)`) when applicable; `None` otherwise.
    pub relativities: Option<Vec<f64>>,
    /// Per-cell support counts, display-only.
    pub support: Vec<f64>,
    /// Optional per-cell standard-error band, display-only. A bare `Option`, NOT
    /// `skip_serializing_if`: bincode is positional/non-self-describing (see the module
    /// doc's `#[serde(flatten)]` ban), so omitting the field when `None` desyncs every
    /// later field for a bincode reader. `#[serde(default)]` keeps older JSON (from before
    /// this field existed) loading.
    #[serde(default)]
    pub se_band: Option<Vec<f64>>,
    /// Cached table variance.
    pub variance: f64,
    /// Sobol share under the bank's reference measure.
    pub sobol: f64,
}

/// The rating-table export artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RatingExport {
    /// Export wire format version.
    pub format_version: u32,
    /// Model/schema wire version.
    pub schema_version: u32,
    /// Exactness mode carried by the source model.
    pub mode: ExactnessMode,
    /// The trained inverse-link family.
    pub link: Link,
    /// The trained objective tag.
    pub objective: ObjectiveTag,
    /// Exported intercept, possibly shifted by [`RatingBasis`] rebasing.
    pub f0: f64,
    /// Reference measure used for purification.
    pub reference_measure: RefMeasure,
    /// Sobol-sorted purified tables.
    pub tables: Vec<RatingTable>,
    /// Factored over-budget order-3 effects (§08.10); empty for budget-fitting banks.
    #[serde(default)]
    pub factored: Vec<RatingFactored>,
}

/// One exported factored order-3 effect (the over-budget escape hatch, §08.10): a sum of
/// per-tree boxes kept WITHOUT a dense cube. Score it as `Σ_box octant[side(x)]`, e.g. a
/// UNION of per-tree CASE expressions in SQL.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RatingFactored {
    /// Raw feature set (order 3).
    pub feature_set: FeatureSet,
    /// Feature names parallel to `feature_set`.
    pub feature_names: Vec<String>,
    /// Per-tree boxes (each: per-axis threshold + missing routing + 8 purified octants).
    pub boxes: Vec<FactoredBoxExport>,
    /// Cached effect variance `σ²(f_u)`.
    pub variance: f64,
    /// Sobol share under the bank's reference measure.
    pub sobol: f64,
}

/// Reserved rare-bucket display label for exported categorical levels (Piece B). The stored
/// model keeps [`RARE_LEVEL_LABEL`] (an internal sentinel) forever; only the EXPORT substitutes
/// this human-readable synthetic label, so model load/save is completely untouched.
const RARE_LEVEL_EXPORT_LABEL: &str = "<rare>";

fn cat_level_display_label(raw_label: &str) -> String {
    if raw_label == RARE_LEVEL_LABEL {
        RARE_LEVEL_EXPORT_LABEL.to_string()
    } else {
        raw_label.to_string()
    }
}

/// Per-level label -> cell mapping for one exported axis (Piece B, design/multichannel-
/// categoricals.md §4.4): `None` for a purely numeric raw feature (no categorical channel at
/// all). Single-channel: each of the one channel's frozen levels, binned against the AXIS's own
/// EXPORTED merged grid (`axis.borders`/`axis.cells` — the tree-gathered grid the table was
/// actually built against, NOT the encoder's full native grid — see explain.rs's `MergedAxis`
/// doc for why single-axis cells are the gathered-split subset), so a level's reported
/// cell always matches which cell of the EXPORTED table it actually falls in. Multi-channel:
/// rebuilds the IDENTICAL `JointCatAxis` `explain.rs`'s `MergedGrids::from_model` built at fit
/// time (same construction function, same frozen encoders, each channel's own NATIVE
/// `border_grid()` — a joint axis's cell space is over every frozen level regardless of which
/// splits any tree happened to use, unlike the single-axis gathered grid; see
/// `JointCatAxis::build`'s own doc) and reads off its `level_cells`, so this can never
/// independently drift from the cell assignment the real exported table actually uses — a
/// cross-check (`joint.n_cells == axis.cells`) catches it loudly if it ever did.
///
/// # Errors
/// Propagates [`CatEncoderStore::get`]/[`JointCatAxis::build`]/[`bin_value`] errors;
/// [`PbError::Internal`] if the axis's cell count overflows `u16` building the single-channel
/// lookup grid, or if a rebuilt joint axis's cell count disagrees with the exported table's own.
fn cat_level_export(
    axis: &AxisId,
    provenance: &[AxisProvenance],
    cat_encoders: &CatEncoderStore,
) -> Result<Option<Vec<CatLevelExport>>, PbError> {
    let channels = channel_axes_for_raw(provenance, axis.raw);
    if channels.is_empty() {
        return Ok(None);
    }
    if channels.len() == 1 {
        let ch = channels.first().ok_or_else(|| PbError::Internal {
            what: "single categorical channel vanished after length check".into(),
        })?;
        let enc = cat_encoders.get(ch.id, axis.raw)?;
        let grid = BorderGrid {
            borders: axis.borders.clone(),
            n_bins: u16::try_from(axis.merged_cells()).map_err(|_| PbError::Internal {
                what: "axis cell count exceeded u16 building level export".into(),
            })?,
            missing_bin: 0,
        };
        let mut out = Vec::with_capacity(enc.levels.len());
        for level in &enc.levels {
            let cell = bin_value(level.encoding, &grid)?;
            let cell = axis_band(axis, u32::from(cell))?;
            out.push(CatLevelExport {
                label: cat_level_display_label(&level.label),
                cell,
            });
        }
        Ok(Some(out))
    } else {
        let mut grids: Vec<BorderGrid> = Vec::new();
        for ch in &channels {
            let native_grid = cat_encoders.get(ch.id, axis.raw)?.border_grid()?;
            if grids.len() <= ch.model_axis {
                grids.resize(
                    ch.model_axis + 1,
                    BorderGrid {
                        borders: Vec::new(),
                        n_bins: 1,
                        missing_bin: 0,
                    },
                );
            }
            if let Some(slot) = grids.get_mut(ch.model_axis) {
                *slot = native_grid;
            }
        }
        let joint = JointCatAxis::build(axis.raw, channels, &grids, cat_encoders)?;
        if joint.n_cells != axis.merged_cells() {
            return Err(PbError::Internal {
                what: format!(
                    "rebuilt joint categorical axis has {} cells but the exported table has {} \
                     for raw feature {}",
                    joint.n_cells,
                    axis.merged_cells(),
                    axis.raw.0
                ),
            });
        }
        Ok(Some(
            joint
                .level_cells
                .iter()
                .map(|(label, cell)| {
                    Ok(CatLevelExport {
                        label: cat_level_display_label(label),
                        cell: axis_band(axis, *cell)?,
                    })
                })
                .collect::<Result<Vec<_>, PbError>>()?,
        ))
    }
}

/// A merged-grid cell's exported cell on `axis`: its band when the table is banded.
fn axis_band(axis: &AxisId, cell: u32) -> Result<u32, PbError> {
    axis.coord(cell)
        .and_then(|c| u32::try_from(c).ok())
        .ok_or_else(|| PbError::Internal {
            what: format!(
                "merged cell {cell} outside the band map of raw {}",
                axis.raw.0
            ),
        })
}

impl TableBank {
    /// Export this exact bank as a rating-table artifact. `provenance` (`model.provenance`/
    /// `TableModel.provenance`) resolves each raw feature's schema name — needed because
    /// `schema.feature_names` is axis-indexed, and a P1 multi-channel raw feature's axis
    /// index no longer equals its raw id (see [`representative_axis_for_raw`]). `cat_encoders`
    /// (`model.schema.cat_encoders`/`TableModel.schema.cat_encoders`) sources every categorical
    /// axis's per-level label -> cell mapping (Piece B, [`AxisExport::levels`]); ignored for a
    /// purely numeric model, so an empty [`CatEncoderStore`] is always a safe argument then.
    ///
    /// # Errors
    /// [`PbError::ExactnessFirewall`] if `mode` is approximate; [`PbError::ShapeMismatch`]
    /// if schema/table metadata is inconsistent; [`PbError::Internal`] if a reference
    /// coordinate escapes a tensor or a categorical axis's rebuilt cell space disagrees with
    /// the exported table's own (see [`cat_level_export`]).
    pub fn to_rating_export(
        &self,
        link: Link,
        mode: &ExactnessMode,
        schema: &ModelSchema,
        provenance: &[AxisProvenance],
        cat_encoders: &CatEncoderStore,
        basis: Option<&RatingBasis>,
    ) -> Result<RatingExport, PbError> {
        if let ExactnessMode::Approximate { reason } = mode {
            return Err(PbError::ExactnessFirewall(reason.clone()));
        }
        // Validate rating-basis references up front: rebasing only shifts a DENSE table <-> f0,
        // so every reference must name a realized dense support. Referencing a factored
        // high-order effect or an unknown support is an error, not a silent no-op.
        if let Some(b) = basis {
            for entry in &b.reference {
                let same = |u: &FeatureSet| {
                    entry.feature_set.len() == u.0.len()
                        && entry
                            .feature_set
                            .iter()
                            .zip(u.0.iter())
                            .all(|(&f, raw)| f == raw.0)
                };
                if self.factored.iter().any(|ft| same(&ft.u)) {
                    return Err(PbError::InvalidConfig {
                        what: format!(
                            "rating basis cannot rebase factored high-order effect {:?}",
                            entry.feature_set
                        ),
                    });
                }
                if !self.tables.iter().any(|t| same(&t.u)) {
                    return Err(PbError::InvalidConfig {
                        what: format!(
                            "rating basis references support {:?} not present in the bank",
                            entry.feature_set
                        ),
                    });
                }
            }
        }
        let sobol: BTreeMap<FeatureSet, f64> = self.sobol().into_iter().collect();
        let mut f0 = self.f0;
        let mut tables = Vec::with_capacity(self.tables.len());
        for table in &self.tables {
            let mut values = table.values.clone();
            if let Some(coord) = basis.and_then(|b| b.coord_for(&table.u)) {
                if coord.len() != table.u.order() {
                    return Err(PbError::ShapeMismatch {
                        what: format!(
                            "rating basis for order-{} table has {} coordinates",
                            table.u.order(),
                            coord.len()
                        ),
                    });
                }
                let coord_usize: Vec<usize> = coord
                    .iter()
                    .zip(&table.axes)
                    .map(|(&c, axis)| axis.coord(c).unwrap_or(usize::MAX))
                    .collect();
                let shift = values.at(&coord_usize).ok_or_else(|| PbError::Internal {
                    what: "rating basis coordinate escaped table".into(),
                })?;
                // Propagate a sparse-to-dense conversion failure instead of shifting f0 while
                // the table itself silently stays unshifted (the exact bug this closes).
                values.add_scalar(-shift)?;
                f0 += shift;
            }
            let mut feature_names = Vec::with_capacity(table.u.order());
            for raw in &table.u.0 {
                let schema_axis = representative_axis_for_raw(provenance, *raw)?;
                let name = schema
                    .feature_names
                    .get(schema_axis)
                    .ok_or_else(|| PbError::ShapeMismatch {
                        what: format!("schema missing feature name for raw {}", raw.0),
                    })?
                    .clone();
                feature_names.push(name);
            }
            let mut axes = Vec::with_capacity(table.axes.len());
            for axis in &table.axes {
                let schema_axis = representative_axis_for_raw(provenance, axis.raw)?;
                let name = schema
                    .feature_names
                    .get(schema_axis)
                    .ok_or_else(|| PbError::ShapeMismatch {
                        what: format!("schema missing feature name for raw {}", axis.raw.0),
                    })?
                    .clone();
                let levels = cat_level_export(axis, provenance, cat_encoders)?;
                // A banded ordinal axis exports its band borders (cells == borders + 2 as ever);
                // a banded categorical axis keeps its level -> band mapping in `levels`.
                axes.push(AxisExport {
                    raw: axis.raw.0,
                    name,
                    borders: axis.band_borders().unwrap_or_else(|| axis.borders.clone()),
                    cells: axis.cells,
                    levels,
                });
            }
            // The rating export's whole contract is "the number deployed is the number
            // audited" (§1) — every tensor read here is `try_values()`, propagating a
            // sparse-to-dense conversion failure instead of silently shipping an empty
            // `values`/`support`/`se_band` against a `shape` that still promises N cells.
            let values_vec = values.try_values()?;
            let relativities = if link == Link::Log {
                Some(
                    values_vec
                        .iter()
                        .map(|v| v.clamp(-30.0, 30.0).exp())
                        .collect(),
                )
            } else {
                None
            };
            let se_band = table
                .se_band
                .as_ref()
                .map(|band| band.per_cell.try_values().map(|v| v.to_vec()))
                .transpose()?;
            tables.push(RatingTable {
                feature_set: table.u.clone(),
                feature_names,
                axes,
                shape: values.shape_u32().to_vec(),
                values: values_vec.to_vec(),
                relativities,
                support: table.support.try_values()?.to_vec(),
                se_band,
                variance: table.variance,
                sobol: *sobol.get(&table.u).unwrap_or(&0.0),
            });
        }
        // Sobol-descending, with the feature set as an explicit secondary key so the
        // export order is total and stable regardless of the input table order.
        tables.sort_by(|a, b| {
            b.sobol
                .total_cmp(&a.sobol)
                .then_with(|| a.feature_set.cmp(&b.feature_set))
        });
        // Factored over-budget order-3 effects (§08.10): emit per-tree boxes (not rebased —
        // the factored residual is already purified) so the export reconstructs F_ens.
        let mut factored = Vec::with_capacity(self.factored.len());
        for ft in &self.factored {
            let mut feature_names = Vec::with_capacity(ft.u.order());
            for raw in &ft.u.0 {
                // Bug #8: `schema.feature_names` is AXIS-indexed (parallel to `provenance`),
                // never raw-feature-indexed — `.get(raw.0 as usize)` only happened to be
                // correct under the pre-P1 green-spine invariant (raw id == axis id always).
                // Once any EARLIER raw feature owns more than one axis (P1 multi-channel), a
                // later raw feature's own id can numerically collide with that earlier
                // feature's own extra (e.g. count) channel axis, misattributing this raw
                // feature's name to a DIFFERENT raw feature's synthetic channel-suffixed
                // schema name instead of its own — the same raw-id-as-axis-id conflation bug
                // class as bug #5, surfacing here in the naming code instead of scoring.
                // `representative_axis_for_raw` is the same resolution the main-effect table
                // loop just above already uses correctly for exactly this reason.
                let schema_axis = representative_axis_for_raw(provenance, *raw)?;
                let name = schema
                    .feature_names
                    .get(schema_axis)
                    .ok_or_else(|| PbError::ShapeMismatch {
                        what: format!("schema missing feature name for raw {}", raw.0),
                    })?
                    .clone();
                feature_names.push(name);
            }
            factored.push(RatingFactored {
                feature_set: ft.u.clone(),
                feature_names,
                boxes: ft.export_boxes()?,
                variance: ft.variance,
                sobol: *sobol.get(&ft.u).unwrap_or(&0.0),
            });
        }
        Ok(RatingExport {
            format_version: FORMAT_VERSION,
            // The stamp is what the CONTENT requires, not a constant. This used to be a
            // hard-wired `SCHEMA_VERSION_UNLIFTED`, which was defensible while every
            // factored effect had 8 octants and any reader could hold one; it stopped being
            // defensible at the high-order lift, where `boxes[i].octants` can be 256 long.
            // A consumer that reads `schema_version` off the export and finds `2` would have
            // no warning at all that the box arity exceeds what it can represent — and the
            // rating export is the artifact a FILING is built from, so a silent arity
            // surprise here is the most expensive one in the codebase.
            schema_version: content_tables_version(self), // the export format did not change at v7
            mode: mode.clone(),
            link,
            objective: schema.objective.clone(),
            f0,
            reference_measure: self.w.clone(),
            tables,
            factored,
        })
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

    /// A bank purified under the exposure measure carries a bincode discriminant no pre-v6
    /// reader knows, so it is stamped v6 whatever its arity; the legacy measure stays at
    /// the content rung. The enum itself round-trips through JSON by name.
    #[test]
    fn exposure_measure_bank_is_stamped_v6() {
        use crate::explain::{fixture_model, fixture_serve, RefMeasure};
        let model = fixture_model();
        let x = fixture_serve();
        let legacy = model.explain(&x, RefMeasure::default()).unwrap();
        assert_eq!(content_tables_version(&legacy), SCHEMA_VERSION_UNLIFTED);
        let w = RefMeasure::ExposureMarginals { floor: 1e-3 };
        let bank = model.explain(&x, w.clone()).unwrap();
        assert_eq!(
            content_tables_version(&bank),
            SCHEMA_VERSION_EXPOSURE_MEASURE
        );
        // v7 floors every tables document (the banded-axis encoding).
        assert_eq!(
            required_tables_version(&legacy),
            SCHEMA_VERSION_BANDED_TABLES
        );
        assert_eq!(required_tables_version(&bank), SCHEMA_VERSION);
        let json = serde_json::to_string(&w).unwrap();
        assert!(json.contains("ExposureMarginals"), "{json}");
        let back: RefMeasure = serde_json::from_str(&json).unwrap();
        assert_eq!(back, w);
    }
    use crate::engine::ExactnessMode;
    use crate::explain::{fixture_serve, RefMeasure};

    /// P1 multi-channel (design/multichannel-categoricals.md §4.3, Piece A): the rating
    /// export collapses a multi-channel raw feature to exactly ONE `AxisExport` (not one per
    /// channel) — the interpretability contract's "no per-channel columns" requirement,
    /// checked at the actual export boundary rather than only at the internal `AxisId` level
    /// (`table_model::tests::tables_model_validates_a_multichannel_bank` already checks that).
    /// This also pins the fix to `to_rating_export`'s feature-name lookup: `schema.
    /// feature_names` is axis-indexed, so indexing it by RAW id (as the pre-P1 code did)
    /// silently reads the wrong name once a raw feature's axis index no longer equals its raw
    /// id — exactly what a multi-channel categorical causes.
    #[test]
    fn rating_export_collapses_a_multichannel_raw_feature_to_one_axis() {
        let model = crate::explain::fixture_multichannel_model();
        let x = crate::explain::fixture_multichannel_serve();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let export = bank
            .to_rating_export(
                model.link,
                &model.mode,
                &model.schema,
                &model.provenance,
                &model.schema.cat_encoders,
                None,
            )
            .unwrap();

        let raw0_tables: Vec<_> = export
            .tables
            .iter()
            .filter(|t| t.feature_set.0.iter().any(|f| f.0 == 0))
            .collect();
        assert_eq!(
            raw0_tables.len(),
            1,
            "one table for raw feature 0, not one per channel"
        );
        let table = raw0_tables[0];
        assert_eq!(table.axes.len(), 1, "one AxisExport, not one per channel");
        assert_eq!(table.axes[0].raw, 0);
        // The representative (first-channel, id 0) name from the fixture's schema — the
        // pre-fix code would have indexed `feature_names` by raw id (0) instead of by this
        // axis's actual position, which happens to also be 0 here (this fixture's raw feature
        // IS raw 0), so this alone would not have caught the bug; see the dedicated
        // regression test below for a case where raw id and axis index diverge.
        assert_eq!(table.axes[0].name, "cat0_mean");
    }

    /// Dedicated regression for the bug `rating_export_collapses_a_multichannel_raw_feature_
    /// to_one_axis` cannot catch by itself: a raw feature positioned AFTER a multi-channel one
    /// has an axis index strictly greater than its raw id (raw 0 owns axes 0-1, raw 1 owns
    /// axis 2), so indexing `schema.feature_names` (axis-indexed) by raw id directly — the
    /// pre-fix code's approach — reads raw 1's name from slot 1, which is actually raw 0's
    /// SECOND channel's name, not raw 1's.
    #[test]
    fn rating_export_names_a_raw_feature_after_a_multichannel_one_correctly() {
        use crate::cat::{
            CatEncoder, CatEncoderStore, CatLevel, CatTarget, TsConfig, TsEncodingId,
        };
        use crate::data::{
            AxisKind, AxisProvenance, BinnedMatrix, BorderGrid, FeatureId, ServeBinnedMatrix,
        };
        use crate::engine::{ObliviousTree, Split};
        use crate::loss::{Link, LossId, ObjectiveTag};

        let grid = BorderGrid {
            borders: vec![1.5],
            n_bins: 3,
            missing_bin: 0,
        };
        // Depth-2 tree: level 0 splits raw 0's channel (axis 0), level 1 splits raw 1 (axis 2).
        let tree = ObliviousTree {
            splits: vec![
                Split {
                    axis: 0,
                    bin_le: 1,
                    missing_left: false,
                },
                Split {
                    axis: 2,
                    bin_le: 1,
                    missing_left: false,
                },
            ],
            leaves: vec![0.0, 2.0, 2.0, 6.0, 0.0, 0.0, 0.0, 0.0],
            depth: 2,
        };
        // `bin` must independently agree with `encoding` binned against `grid`'s border 1.5
        // (<=1.5 -> bin 1, >1.5 -> bin 2): `JointCatAxis::build` at explain time re-derives bins
        // from `encoding` + `model.grids` and ignores `.bin` entirely, but `CatEncoder::
        // border_grid()` (used to rebuild the joint axis for Piece B's label export) reads
        // `.bin` directly — a wrong `.bin` here is invisible to the former and silently wrong
        // to the latter, so both must be set consistently, not just the encoding.
        let level = |label: &str, encoding: f32, bin: u8| CatLevel {
            label: label.to_owned(),
            members: vec![label.to_owned()],
            encoding,
            bin,
            weight: 1.0,
        };
        let mean_enc = CatEncoder {
            raw: FeatureId(0),
            id: TsEncodingId(0),
            levels: vec![level("a", 1.0, 1), level("b", 2.0, 2)],
            base: 0.0,
            config: TsConfig::default(),
        };
        let count_enc = CatEncoder {
            raw: FeatureId(0),
            id: TsEncodingId(1),
            // Mirrors mean_enc's own bin split (<=1.5 -> bin 1, >1.5 -> bin 2) so "a"/"b" map to
            // joint tuples (1,1)/(2,2) — this test only needs the tuples the serve matrix below
            // actually uses to be valid, not every one of the four possible combinations.
            levels: vec![level("a", 1.0, 1), level("b", 2.0, 2)],
            base: 0.0,
            config: TsConfig {
                target: CatTarget::Count,
                ..TsConfig::default()
            },
        };
        let model = crate::engine::Model {
            f0: 0.0,
            trees: vec![(1.0, tree)],
            grids: vec![grid.clone(), grid.clone(), grid.clone()],
            provenance: vec![
                AxisProvenance {
                    raw: FeatureId(0),
                    kind: AxisKind::CategoricalTS {
                        encoding: TsEncodingId(0),
                    },
                },
                AxisProvenance {
                    raw: FeatureId(0),
                    kind: AxisKind::CategoricalTS {
                        encoding: TsEncodingId(1),
                    },
                },
                AxisProvenance {
                    raw: FeatureId(1),
                    kind: AxisKind::Numeric,
                },
            ],
            link: Link::Identity,
            mode: ExactnessMode::Exact,
            schema: crate::engine::ModelSchema {
                feature_names: vec!["cat0_mean".into(), "cat0_count".into(), "num1".into()],
                feature_kinds: vec![
                    AxisKind::CategoricalTS {
                        encoding: TsEncodingId(0),
                    },
                    AxisKind::CategoricalTS {
                        encoding: TsEncodingId(1),
                    },
                    AxisKind::Numeric,
                ],
                cat_encoders: CatEncoderStore::from_encoders(vec![mean_enc, count_enc]),
                class_labels: None,
                objective: ObjectiveTag {
                    link: Link::Identity,
                    loss: LossId::SquaredError,
                    tweedie_rho: None,
                },
            },
            schema_version: crate::serialize::SCHEMA_VERSION_UNLIFTED,
            correction: None,
            bag_spans: None,
            bag_in_bag: None,
            delta_step_gate: None,
        };
        // Categorical columns both follow the SAME (1,1)="a" / (2,2)="b" pattern (the only two
        // joint tuples this fixture's frozen levels produce); the numeric column (raw 1) varies
        // independently to realize the tree's second split.
        let x = ServeBinnedMatrix(BinnedMatrix {
            data: vec![vec![1, 1, 2, 2], vec![1, 1, 2, 2], vec![1, 2, 1, 2]],
            n_rows: 4,
            grids: vec![grid.clone(), grid.clone(), grid],
            provenance: model.provenance.clone(),
        });

        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let export = bank
            .to_rating_export(
                model.link,
                &model.mode,
                &model.schema,
                &model.provenance,
                &model.schema.cat_encoders,
                None,
            )
            .unwrap();

        let raw1_table = export
            .tables
            .iter()
            .find(|t| t.feature_set.0.iter().any(|f| f.0 == 1))
            .expect("raw feature 1 should have a realized table");
        assert_eq!(raw1_table.axes.len(), 1);
        assert_eq!(raw1_table.axes[0].raw, 1);
        // The bug this pins: pre-fix code indexed `feature_names` by raw id (1), reading
        // "cat0_count" (slot 1, actually raw 0's second channel) instead of "num1" (slot 2,
        // raw 1's actual axis).
        assert_eq!(raw1_table.axes[0].name, "num1");
        assert_eq!(raw1_table.feature_names, vec!["num1".to_string()]);
    }

    /// (a) Piece B: a single-channel categorical axis's export lists every level with the
    /// cell it actually lands in, and that cell's exported value reconstructs (via the
    /// mass-conservation `f0 + table.values[cell]` identity, I2's own reconstruction gate)
    /// the ensemble's own prediction for a row bearing that level — not just "some cell",
    /// the CORRECT one.
    #[test]
    fn rating_export_single_channel_categorical_lists_every_level_with_correct_cell() {
        use crate::cat::{CatEncoder, CatEncoderStore, CatLevel, TsConfig, TsEncodingId};
        use crate::data::{
            AxisKind, AxisProvenance, BinnedMatrix, BorderGrid, FeatureId, ServeBinnedMatrix,
        };
        use crate::engine::{ObliviousTree, Split};
        use crate::loss::{Link, LossId, ObjectiveTag};

        let grid = BorderGrid {
            borders: vec![1.5, 2.5],
            n_bins: 4,
            missing_bin: 0,
        };
        // Two splits on the SAME axis — a legitimate lower-order refinement (engine::mod's
        // `ObliviousTree` doc: "repeated raw features are valid") — fully resolve all three
        // bins, so every level gets its own distinguishable exported cell rather than a
        // coarser gathered grid.
        let tree = ObliviousTree {
            splits: vec![
                Split {
                    axis: 0,
                    bin_le: 1,
                    missing_left: false,
                },
                Split {
                    axis: 0,
                    bin_le: 2,
                    missing_left: false,
                },
            ],
            // idx = b0 | b1<<1 (ObliviousTree::lookup): bin=1 -> both splits low -> idx 3;
            // bin=2 -> low only on the second split -> idx 2; bin=3/missing -> high on both
            // -> idx 0.
            leaves: vec![100.0, 0.0, 200.0, 300.0, 0.0, 0.0, 0.0, 0.0],
            depth: 2,
        };
        let level = |label: &str, bin: u8, encoding: f32| CatLevel {
            label: label.to_owned(),
            members: vec![label.to_owned()],
            encoding,
            bin,
            weight: 1.0,
        };
        let enc = CatEncoder {
            raw: FeatureId(0),
            id: TsEncodingId(0),
            levels: vec![level("x", 1, 1.0), level("y", 2, 2.0), level("z", 3, 3.0)],
            base: 0.0,
            config: TsConfig::default(),
        };
        let model = crate::engine::Model {
            f0: 0.0,
            trees: vec![(1.0, tree)],
            grids: vec![grid.clone()],
            provenance: vec![AxisProvenance {
                raw: FeatureId(0),
                kind: AxisKind::CategoricalTS {
                    encoding: TsEncodingId(0),
                },
            }],
            link: Link::Identity,
            mode: ExactnessMode::Exact,
            schema: crate::engine::ModelSchema {
                feature_names: vec!["cat0".into()],
                feature_kinds: vec![AxisKind::CategoricalTS {
                    encoding: TsEncodingId(0),
                }],
                cat_encoders: CatEncoderStore::from_encoders(vec![enc]),
                class_labels: None,
                objective: ObjectiveTag {
                    link: Link::Identity,
                    loss: LossId::SquaredError,
                    tweedie_rho: None,
                },
            },
            schema_version: crate::serialize::SCHEMA_VERSION_UNLIFTED,
            correction: None,
            bag_spans: None,
            bag_in_bag: None,
            delta_step_gate: None,
        };
        let x = ServeBinnedMatrix(BinnedMatrix {
            data: vec![vec![1, 2, 3]],
            n_rows: 3,
            grids: vec![grid],
            provenance: model.provenance.clone(),
        });
        let preds = model.predict(&x.0, None).unwrap();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let export = bank
            .to_rating_export(
                model.link,
                &model.mode,
                &model.schema,
                &model.provenance,
                &model.schema.cat_encoders,
                None,
            )
            .unwrap();
        let table = export
            .tables
            .iter()
            .find(|t| t.feature_set.0.iter().any(|f| f.0 == 0))
            .unwrap();
        let axis = &table.axes[0];
        let levels = axis
            .levels
            .as_ref()
            .expect("categorical axis must export levels");
        assert_eq!(levels.len(), 3, "one export entry per level");

        let cell_of = |label: &str| {
            levels
                .iter()
                .find(|l| l.label == label)
                .unwrap_or_else(|| panic!("level `{label}` missing from export"))
                .cell
        };
        let (cx, cy, cz) = (cell_of("x"), cell_of("y"), cell_of("z"));
        assert_ne!(cx, cy);
        assert_ne!(cy, cz);
        assert_ne!(cx, cz);

        for (row, label) in [(0usize, "x"), (1, "y"), (2, "z")] {
            let cell = cell_of(label) as usize;
            let reconstructed = export.f0 + table.values[cell];
            assert!(
                (f64::from(preds[row]) - reconstructed).abs() < 1e-6,
                "level `{label}`: ensemble predicted {}, export cell {cell} reconstructs to {reconstructed}",
                preds[row]
            );
        }
    }

    /// (b) Piece B: a multi-channel export lists EVERY level (never dropping one whose
    /// signature happens to coincide with another's), levels that share every channel's bin
    /// correctly share the SAME cell (not forced apart — the team lead's explicit
    /// clarification, spec §6), remains exactly one table/one `AxisExport` for the raw
    /// feature, and no entry is named after a channel.
    #[test]
    fn rating_export_multichannel_lists_every_level_sharing_cells_correctly() {
        use crate::cat::{
            CatEncoder, CatEncoderStore, CatLevel, CatTarget, TsConfig, TsEncodingId,
        };
        use crate::data::{
            AxisKind, AxisProvenance, BinnedMatrix, BorderGrid, FeatureId, ServeBinnedMatrix,
        };
        use crate::engine::{ObliviousTree, Split};
        use crate::loss::{Link, LossId, ObjectiveTag};

        let grid = BorderGrid {
            borders: vec![1.5],
            n_bins: 3,
            missing_bin: 0,
        };
        // The same proven tree as `fixture_multichannel_model` (explain.rs): 2 splits, one
        // per channel axis of the SAME raw feature, with a genuine (non-additive) interaction.
        let tree = ObliviousTree {
            splits: vec![
                Split {
                    axis: 0,
                    bin_le: 1,
                    missing_left: false,
                },
                Split {
                    axis: 1,
                    bin_le: 1,
                    missing_left: false,
                },
            ],
            leaves: vec![0.0, 2.0, 2.0, 6.0, 0.0, 0.0, 0.0, 0.0],
            depth: 2,
        };
        let level = |label: &str, bin: u8, encoding: f32| CatLevel {
            label: label.to_owned(),
            members: vec![label.to_owned()],
            encoding,
            bin,
            weight: 1.0,
        };
        let mean_enc = CatEncoder {
            raw: FeatureId(0),
            id: TsEncodingId(0),
            levels: vec![
                level("aa", 1, 1.0),
                level("ab", 1, 1.1),
                level("ba", 2, 2.0),
                level("bb", 2, 2.1),
                // "bb2" shares mean_enc's encoding EXACTLY with "bb" -> same mean bin.
                level("bb2", 2, 2.1),
            ],
            base: 0.0,
            config: TsConfig::default(),
        };
        let count_enc = CatEncoder {
            raw: FeatureId(0),
            id: TsEncodingId(1),
            levels: vec![
                level("aa", 1, 0.1),
                level("ab", 2, 2.0),
                level("ba", 1, 0.2),
                level("bb", 2, 2.1),
                // ... and shares count_enc's encoding EXACTLY with "bb" too -> same count
                // bin -> "bb"/"bb2" are indistinguishable on EVERY channel, so they must
                // share one joint cell (not be forced apart into two).
                level("bb2", 2, 2.1),
            ],
            base: 0.0,
            config: TsConfig {
                target: CatTarget::Count,
                ..TsConfig::default()
            },
        };
        let model = crate::engine::Model {
            f0: 0.0,
            trees: vec![(1.0, tree)],
            grids: vec![grid.clone(), grid.clone()],
            provenance: vec![
                AxisProvenance {
                    raw: FeatureId(0),
                    kind: AxisKind::CategoricalTS {
                        encoding: TsEncodingId(0),
                    },
                },
                AxisProvenance {
                    raw: FeatureId(0),
                    kind: AxisKind::CategoricalTS {
                        encoding: TsEncodingId(1),
                    },
                },
            ],
            link: Link::Identity,
            mode: ExactnessMode::Exact,
            schema: crate::engine::ModelSchema {
                feature_names: vec!["cat0_mean".into(), "cat0_count".into()],
                feature_kinds: vec![
                    AxisKind::CategoricalTS {
                        encoding: TsEncodingId(0),
                    },
                    AxisKind::CategoricalTS {
                        encoding: TsEncodingId(1),
                    },
                ],
                cat_encoders: CatEncoderStore::from_encoders(vec![mean_enc, count_enc]),
                class_labels: None,
                objective: ObjectiveTag {
                    link: Link::Identity,
                    loss: LossId::SquaredError,
                    tweedie_rho: None,
                },
            },
            schema_version: crate::serialize::SCHEMA_VERSION_UNLIFTED,
            correction: None,
            bag_spans: None,
            bag_in_bag: None,
            delta_step_gate: None,
        };
        let x = ServeBinnedMatrix(BinnedMatrix {
            data: vec![vec![1, 1, 2, 2, 2], vec![1, 2, 1, 2, 2]],
            n_rows: 5,
            grids: vec![grid.clone(), grid],
            provenance: model.provenance.clone(),
        });
        let preds = model.predict(&x.0, None).unwrap();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let export = bank
            .to_rating_export(
                model.link,
                &model.mode,
                &model.schema,
                &model.provenance,
                &model.schema.cat_encoders,
                None,
            )
            .unwrap();

        assert_eq!(
            export.tables.len(),
            1,
            "one table for the whole model (a single raw feature)"
        );
        let table = &export.tables[0];
        assert_eq!(table.axes.len(), 1, "one AxisExport, not one per channel");
        let axis = &table.axes[0];
        let levels = axis
            .levels
            .as_ref()
            .expect("categorical axis must export levels");
        assert_eq!(
            levels.len(),
            5,
            "every level listed, including the one that shares a cell"
        );

        let cell_of = |label: &str| levels.iter().find(|l| l.label == label).unwrap().cell;
        let (c_aa, c_ab, c_ba, c_bb, c_bb2) = (
            cell_of("aa"),
            cell_of("ab"),
            cell_of("ba"),
            cell_of("bb"),
            cell_of("bb2"),
        );
        assert_eq!(
            c_bb, c_bb2,
            "\"bb\"/\"bb2\" share every channel bin -> must share one cell"
        );
        let distinct: std::collections::BTreeSet<u32> =
            [c_aa, c_ab, c_ba, c_bb].into_iter().collect();
        assert_eq!(
            distinct.len(),
            4,
            "aa/ab/ba/bb must all land in distinct cells"
        );

        for (row, label) in [(0usize, "aa"), (1, "ab"), (2, "ba"), (3, "bb"), (4, "bb2")] {
            let cell = cell_of(label) as usize;
            let reconstructed = export.f0 + table.values[cell];
            assert!(
                (f64::from(preds[row]) - reconstructed).abs() < 1e-6,
                "level `{label}`: model predicted {}, label->cell->value reconstructs to {reconstructed}",
                preds[row]
            );
        }
    }

    /// (c) Piece B: the reserved rare bucket appears in the export as ONE entry with a clear
    /// synthetic display label (`"<rare>"`), never the internal storage sentinel and never
    /// silently dropped.
    #[test]
    fn rating_export_shows_the_rare_bucket_as_one_labeled_entry() {
        use crate::cat::{
            CatEncoder, CatEncoderStore, CatLevel, TsConfig, TsEncodingId, RARE_LEVEL_LABEL,
        };
        use crate::data::{
            AxisKind, AxisProvenance, BinnedMatrix, BorderGrid, FeatureId, ServeBinnedMatrix,
        };
        use crate::engine::{ObliviousTree, Split};
        use crate::loss::{Link, LossId, ObjectiveTag};

        let grid = BorderGrid {
            borders: vec![1.5],
            n_bins: 3,
            missing_bin: 0,
        };
        let tree = ObliviousTree {
            splits: vec![Split {
                axis: 0,
                bin_le: 1,
                missing_left: false,
            }],
            leaves: vec![10.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            depth: 1,
        };
        let common = CatLevel {
            label: "common".into(),
            members: vec!["common".into()],
            encoding: 1.0,
            bin: 1,
            weight: 20.0,
        };
        let rare_bucket = CatLevel {
            label: RARE_LEVEL_LABEL.to_string(),
            members: vec!["r1".into(), "r2".into()],
            encoding: 2.0,
            bin: 2,
            weight: 2.0,
        };
        let enc = CatEncoder {
            raw: FeatureId(0),
            id: TsEncodingId(0),
            levels: vec![common, rare_bucket],
            base: 0.0,
            config: TsConfig::default(),
        };
        let model = crate::engine::Model {
            f0: 0.0,
            trees: vec![(1.0, tree)],
            grids: vec![grid.clone()],
            provenance: vec![AxisProvenance {
                raw: FeatureId(0),
                kind: AxisKind::CategoricalTS {
                    encoding: TsEncodingId(0),
                },
            }],
            link: Link::Identity,
            mode: ExactnessMode::Exact,
            schema: crate::engine::ModelSchema {
                feature_names: vec!["cat0".into()],
                feature_kinds: vec![AxisKind::CategoricalTS {
                    encoding: TsEncodingId(0),
                }],
                cat_encoders: CatEncoderStore::from_encoders(vec![enc]),
                class_labels: None,
                objective: ObjectiveTag {
                    link: Link::Identity,
                    loss: LossId::SquaredError,
                    tweedie_rho: None,
                },
            },
            schema_version: crate::serialize::SCHEMA_VERSION_UNLIFTED,
            correction: None,
            bag_spans: None,
            bag_in_bag: None,
            delta_step_gate: None,
        };
        let x = ServeBinnedMatrix(BinnedMatrix {
            data: vec![vec![1, 2]],
            n_rows: 2,
            grids: vec![grid],
            provenance: model.provenance.clone(),
        });
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let export = bank
            .to_rating_export(
                model.link,
                &model.mode,
                &model.schema,
                &model.provenance,
                &model.schema.cat_encoders,
                None,
            )
            .unwrap();
        let table = &export.tables[0];
        let levels = table.axes[0]
            .levels
            .as_ref()
            .expect("categorical axis must export levels");
        assert_eq!(levels.len(), 2);
        assert!(levels.iter().any(|l| l.label == "common"));
        // The reserved internal sentinel must never leak into the export.
        assert!(!levels.iter().any(|l| l.label == RARE_LEVEL_LABEL));
        assert!(levels.iter().any(|l| l.label == "<rare>"));
    }

    /// (d) Piece B round-trip: label -> cell -> value equals the model's ACTUAL per-level
    /// contribution on the same non-additive fixture Stage B's own exactness proof uses (a
    /// tree splitting BOTH channels, so the true value has a residual interaction a naive
    /// per-channel sum would get wrong: `g(1,1)=6`, not the additive
    /// `g(1,2)+g(2,1)-g(2,2)=2+2-0=4`).
    #[test]
    fn rating_export_round_trips_label_to_cell_to_value_on_a_genuine_interaction() {
        let model = crate::explain::fixture_multichannel_model();
        let x = crate::explain::fixture_multichannel_serve();
        let preds = model.predict(&x.0, None).unwrap();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let export = bank
            .to_rating_export(
                model.link,
                &model.mode,
                &model.schema,
                &model.provenance,
                &model.schema.cat_encoders,
                None,
            )
            .unwrap();
        let table = &export.tables[0]; // the fixture's only raw feature
        let levels = table.axes[0]
            .levels
            .as_ref()
            .expect("categorical axis must export levels");

        // `fixture_multichannel_serve`'s rows are, in column-major (mean, count) order,
        // [1,1,2,2] / [1,2,1,2] -> row 0 = "aa", row 1 = "ab", row 2 = "ba", row 3 = "bb".
        let cell_of = |label: &str| levels.iter().find(|l| l.label == label).unwrap().cell;
        for (row, label) in [(0usize, "aa"), (1, "ab"), (2, "ba"), (3, "bb")] {
            let cell = cell_of(label) as usize;
            let reconstructed = export.f0 + table.values[cell];
            assert!(
                (f64::from(preds[row]) - reconstructed).abs() < 1e-6,
                "level `{label}`: model predicted {}, label->cell->value reconstructs to {reconstructed}",
                preds[row]
            );
        }
        // Pin the specific non-additive value the naive per-channel-sum design would get
        // wrong (an additive model would force g(1,1) = 2+2-0 = 4, not 6).
        let reconstructed_aa = export.f0 + table.values[cell_of("aa") as usize];
        assert!((reconstructed_aa - 6.0).abs() < 1e-6);
    }

    #[test]
    fn bincode_round_trip_is_bit_identical() {
        let model = crate::explain::fixture_model();
        let bytes = encode_model(&model).unwrap();
        let back = decode_model(&bytes).unwrap();
        assert_eq!(model, back);
        // Re-encoding the decoded model reproduces the exact bytes.
        assert_eq!(bytes, encode_model(&back).unwrap());
    }

    #[test]
    fn json_round_trip_matches_and_model_methods_work() {
        let doc = ModelDoc::new(crate::explain::fixture_model());
        let json = encode_doc_json(&doc).unwrap();
        let back = decode_doc_json(&json).unwrap();
        assert_eq!(doc, back);

        let model_json = doc.model.to_json().unwrap();
        assert_eq!(Model::from_json(&model_json).unwrap(), doc.model);
        let model_bytes = doc.model.to_bincode().unwrap();
        assert_eq!(Model::from_bincode(&model_bytes).unwrap(), doc.model);
    }

    #[test]
    fn migrate_identity_loads_current_version_and_revalidates() {
        // The fixture is unlifted (depth <= LEGACY_MAX_DEPTH), so it is stamped at the
        // MINIMUM version its contents require -- see `SCHEMA_VERSION`'s doc.
        let doc = ModelDoc::new(crate::explain::fixture_model());
        assert_eq!(doc.schema_version, SCHEMA_VERSION_UNLIFTED);
        let stamped = doc.schema_version;
        let value = serde_json::to_value(&doc).unwrap();
        assert_eq!(migrate(value, FORMAT_VERSION, stamped).unwrap(), doc.model);

        let value = serde_json::to_value(&doc).unwrap();
        assert!(matches!(
            migrate(value, FORMAT_VERSION + 1, stamped),
            Err(PbError::Serialization(_))
        ));

        let mut mismatched = serde_json::to_value(&doc).unwrap();
        mismatched["format_version"] = serde_json::json!(FORMAT_VERSION + 1);
        assert!(matches!(
            migrate(mismatched, FORMAT_VERSION, stamped),
            Err(PbError::Serialization(_))
        ));

        // A caller declaring the wrong schema_version fails closed rather than migrating
        // the wrong thing, mirroring the format_version cross-check above.
        let value = serde_json::to_value(&doc).unwrap();
        assert!(matches!(
            migrate(value, FORMAT_VERSION, SCHEMA_VERSION + 1),
            Err(PbError::Serialization(_))
        ));
    }

    /// A v1 JSON document (pre-d3c1140, before `Model.correction` existed). It deserializes
    /// losslessly into the current `Model` struct (`correction` is `#[serde(default)]`, so
    /// it fills as `None`) and spec §02.8's From-based migration contract applies to this
    /// self-describing JSON envelope, so it must MIGRATE forward, not be rejected.
    fn v1_schema_fixture() -> (ModelDoc, serde_json::Value) {
        let doc = ModelDoc::new(crate::explain::fixture_model());
        let mut value = serde_json::to_value(&doc).unwrap();
        value["schema_version"] = serde_json::json!(1);
        value["model"]["schema_version"] = serde_json::json!(1);
        value["model"].as_object_mut().unwrap().remove("correction");
        (doc, value)
    }

    #[test]
    fn schema_v1_json_migrates_via_decode_doc_json() {
        let (doc, value) = v1_schema_fixture();
        let json = serde_json::to_string(&value).unwrap();

        let migrated = decode_doc_json(&json).unwrap();
        // Migration targets the version the CONTENTS require, not the newest this build
        // knows: an unlifted model stays at `SCHEMA_VERSION_UNLIFTED` so it remains
        // loadable by (and byte-identical to) a pre-lift reader.
        assert_eq!(migrated.schema_version, SCHEMA_VERSION_UNLIFTED);
        assert_eq!(migrated.model.schema_version, SCHEMA_VERSION_UNLIFTED);
        assert_eq!(migrated.model.correction, None);
        // The v1 -> v2 shim is identity-plus-version-stamp: every other field is untouched.
        assert_eq!(migrated.model, doc.model);
    }

    #[test]
    fn schema_v1_json_migrates_via_migrate_facade() {
        let (doc, value) = v1_schema_fixture();
        let model = migrate(value, FORMAT_VERSION, 1).unwrap();
        assert_eq!(model, doc.model);
    }

    #[test]
    fn schema_v1_bincode_is_still_rejected() {
        // bincode is positional/non-self-describing: only the JSON entry points above get a
        // migration path (migrate_schema_json's doc explains why -- spec §02.8's From-based
        // contract is scoped to the self-describing envelope).
        let mut doc = ModelDoc::new(crate::explain::fixture_model());
        doc.schema_version = 1;
        doc.model.schema_version = 1;
        let bytes = encode_doc(&doc).unwrap();
        let err = decode_doc(&bytes).unwrap_err();
        assert!(matches!(err, PbError::Serialization(_)));
        assert!(
            err.to_string()
                .contains("unsupported model schema_version 1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn schema_newer_than_build_is_rejected_via_json() {
        let doc = ModelDoc::new(crate::explain::fixture_model());
        let mut value = serde_json::to_value(&doc).unwrap();
        value["schema_version"] = serde_json::json!(SCHEMA_VERSION + 1);
        value["model"]["schema_version"] = serde_json::json!(SCHEMA_VERSION + 1);
        let json = serde_json::to_string(&value).unwrap();
        assert!(matches!(
            decode_doc_json(&json),
            Err(PbError::Serialization(_))
        ));
    }

    #[test]
    fn unregistered_schema_version_has_no_migration_path() {
        // schema_version 0 has never existed; there is no shim for it (unlike 1, which DOES
        // migrate above), and it must not be silently treated as "close enough".
        let doc = ModelDoc::new(crate::explain::fixture_model());
        let mut value = serde_json::to_value(&doc).unwrap();
        value["schema_version"] = serde_json::json!(0);
        value["model"]["schema_version"] = serde_json::json!(0);
        let json = serde_json::to_string(&value).unwrap();
        let err = decode_doc_json(&json).unwrap_err();
        assert!(matches!(err, PbError::Serialization(_)));
        let msg = err.to_string();
        assert!(
            msg.contains("no migration path registered from schema_version 0"),
            "msg={msg}"
        );
        assert!(
            msg.contains("pin the t-boost version") || msg.contains("re-export"),
            "msg={msg}"
        );
    }

    #[test]
    fn schema_metadata_round_trips_through_json_and_bincode() {
        let mut model = crate::explain::fixture_model();
        model.schema.feature_names = vec!["territory".into(), "age_band".into()];
        model.schema.class_labels = Some(vec!["low".into(), "high".into()]);

        let json = model.to_json().unwrap();
        let from_json = Model::from_json(&json).unwrap();
        assert_eq!(from_json.schema.feature_names, model.schema.feature_names);
        assert_eq!(from_json.schema.class_labels, model.schema.class_labels);

        let bytes = model.to_bincode().unwrap();
        let from_bytes = Model::from_bincode(&bytes).unwrap();
        assert_eq!(from_bytes.schema.feature_names, model.schema.feature_names);
        assert_eq!(from_bytes.schema.class_labels, model.schema.class_labels);
    }

    #[test]
    fn bumped_format_version_is_rejected() {
        let mut doc = ModelDoc::new(crate::explain::fixture_model());
        doc.format_version = FORMAT_VERSION + 1;
        let bytes = encode_doc(&doc).unwrap();
        assert!(matches!(decode_doc(&bytes), Err(PbError::Serialization(_))));
    }

    #[test]
    fn bumped_schema_version_is_rejected() {
        let mut doc = ModelDoc::new(crate::explain::fixture_model());
        doc.schema_version = SCHEMA_VERSION + 1;
        let bytes = encode_doc(&doc).unwrap();
        assert!(matches!(decode_doc(&bytes), Err(PbError::Serialization(_))));
    }

    #[test]
    fn trailing_bincode_bytes_are_rejected() {
        let mut bytes = encode_model(&crate::explain::fixture_model()).unwrap();
        bytes.push(0);
        assert!(matches!(
            decode_model(&bytes),
            Err(PbError::Serialization(_))
        ));
    }

    #[test]
    fn decode_revalidates_tree_axes() {
        let mut doc = ModelDoc::new(crate::explain::fixture_model());
        doc.model.trees[0].1.splits[0].axis = 99;
        let bytes = encode_doc(&doc).unwrap();
        assert!(decode_doc(&bytes).is_err());
    }

    #[test]
    fn decode_revalidates_schema_lengths() {
        let mut doc = ModelDoc::new(crate::explain::fixture_model());
        doc.model.schema.feature_kinds.pop();
        let json = encode_doc_json(&doc).unwrap();
        assert!(decode_doc_json(&json).is_err());
    }

    #[test]
    fn path_a_scoring_matches_ensemble_and_predicts_response() {
        let model = crate::explain::fixture_model();
        let x = fixture_serve();
        let mut out = vec![0.0_f32; x.0.n_rows as usize];
        model.score_trees(&x.0, None, &mut out).unwrap();
        for (row, &score) in out.iter().enumerate() {
            let bins: Vec<u8> = x.0.data.iter().map(|c| c[row]).collect();
            assert_eq!(
                score.to_bits(),
                (model.ensemble_f64(&bins).unwrap() as f32).to_bits()
            );
        }
        assert_eq!(model.predict_binned(&x.0, None).unwrap(), out);
        assert_eq!(model.predict(&x.0, None).unwrap(), out);
    }

    #[test]
    fn rating_export_pure_and_rebased_forms_are_exact() {
        let model = crate::explain::fixture_model();
        let x = fixture_serve();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let pure = bank
            .to_rating_export(
                model.link,
                &model.mode,
                &model.schema,
                &model.provenance,
                &model.schema.cat_encoders,
                None,
            )
            .unwrap();
        assert_eq!(pure.mode, ExactnessMode::Exact);
        assert_eq!(pure.tables.len(), bank.tables.len());

        let first = bank.tables[0].clone();
        let coord = vec![0_u32; first.u.order()];
        let shift = first.values.at(&vec![0_usize; first.u.order()]).unwrap();
        let basis = RatingBasis {
            reference: vec![RatingReference {
                feature_set: first.u.0.iter().map(|f| f.0).collect(),
                coord,
            }],
        };
        // A NON-EMPTY RatingBasis must round-trip through JSON (the public basis_json path).
        let json = serde_json::to_string(&basis).unwrap();
        let basis: RatingBasis = serde_json::from_str(&json).unwrap();
        let rebased = bank
            .to_rating_export(
                model.link,
                &model.mode,
                &model.schema,
                &model.provenance,
                &model.schema.cat_encoders,
                Some(&basis),
            )
            .unwrap();
        assert!((rebased.f0 - (bank.f0 + shift)).abs() < 1e-12);
        let table = rebased
            .tables
            .iter()
            .find(|t| t.feature_set == first.u)
            .unwrap();
        assert!(table.values[0].abs() < 1e-12);
    }

    /// `RatingTable.se_band` used to carry `skip_serializing_if = "Option::is_none"`, which
    /// omits the field's bytes entirely when `None`. bincode is positional/non-self-describing
    /// (module doc), so the decoder still expects an Option discriminant there -- every field
    /// after `se_band` would desync. `se_band` is `None` in the common (non-bagged) case, so
    /// this fixture (a plain single-fit `explain()` bank, never `average_banks`/
    /// `attach_se_bands`) already exercises exactly that case.
    #[test]
    fn rating_export_round_trips_through_bincode_with_se_band_none() {
        let model = crate::explain::fixture_model();
        let x = fixture_serve();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let pure = bank
            .to_rating_export(
                model.link,
                &model.mode,
                &model.schema,
                &model.provenance,
                &model.schema.cat_encoders,
                None,
            )
            .unwrap();
        assert!(!pure.tables.is_empty());
        assert!(pure.tables.iter().all(|t| t.se_band.is_none()));

        let bytes = bincode::serde::encode_to_vec(&pure, bincode::config::standard()).unwrap();
        let (back, len): (RatingExport, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(len, bytes.len());
        assert_eq!(back, pure);
    }

    /// The `None` case above is the one that was actually broken; this pins the `Some(..)`
    /// case round-trips too (every field is a plain `Option` now -- no reason it wouldn't,
    /// but the whole point of a wire-format contract is not to assume).
    #[test]
    fn rating_export_round_trips_through_bincode_with_se_band_populated() {
        let model = crate::explain::fixture_model();
        let x = fixture_serve();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let mut pure = bank
            .to_rating_export(
                model.link,
                &model.mode,
                &model.schema,
                &model.provenance,
                &model.schema.cat_encoders,
                None,
            )
            .unwrap();
        assert!(!pure.tables.is_empty());
        let n = pure.tables[0].values.len();
        pure.tables[0].se_band = Some(vec![0.25; n]);

        let bytes = bincode::serde::encode_to_vec(&pure, bincode::config::standard()).unwrap();
        let (back, len): (RatingExport, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(len, bytes.len());
        assert_eq!(back, pure);
        assert!(back.tables[0].se_band.is_some());
    }

    /// End-to-end negative test for the TableModel load gate (companion to the direct-
    /// mutation unit tests in table_model.rs, which pin `validate()`'s own logic): builds
    /// real corrupted bytes -- bypassing `to_bincode()`'s own validate-before-encode gate by
    /// constructing the `TablesDoc` envelope directly, exactly as an externally-corrupted or
    /// bit-flipped artifact would arrive -- and asserts the public `decode_tables` load path
    /// rejects it. JSON cannot represent NaN at all (`serde_json` refuses to encode it), so
    /// this non-finite-cell case is bincode-only; shape corruption is covered via JSON in
    /// table_model.rs's `corrupted_json_with_*` tests.
    #[test]
    fn corrupted_tables_bincode_with_nan_cell_fails_to_load() {
        let model = crate::explain::fixture_model();
        let x = fixture_serve();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let mut tm = TableModel::from_model_and_bank(&model, bank);
        let table = tm.bank.tables.get_mut(0).unwrap();
        let coord = vec![0usize; table.values.shape().len()];
        table.values.set(&coord, f64::NAN).unwrap();

        let doc = TablesDoc::new(tm);
        let body = bincode::serde::encode_to_vec(&doc, bincode::config::standard()).unwrap();
        let mut bytes = Vec::with_capacity(TABLES_MAGIC.len() + body.len());
        bytes.extend_from_slice(TABLES_MAGIC);
        bytes.extend_from_slice(&body);

        assert!(matches!(
            decode_tables(&bytes),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn rating_export_refuses_approximate_mode() {
        let model = crate::explain::fixture_model();
        let x = fixture_serve();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let mode = ExactnessMode::Approximate {
            reason: "test".into(),
        };
        assert!(matches!(
            bank.to_rating_export(
                model.link,
                &mode,
                &model.schema,
                &model.provenance,
                &model.schema.cat_encoders,
                None
            ),
            Err(PbError::ExactnessFirewall(_))
        ));
    }

    /// A rating basis may only rebase a DENSE support that the bank actually realized.
    /// Naming a support the bank never produced is a deployer typo and must error loudly,
    /// not be silently ignored (which would ship an un-rebased table under a basis label).
    #[test]
    fn rating_basis_rejects_unknown_support() {
        let model = crate::explain::fixture_model();
        let x = fixture_serve();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();

        // A reference to a realized dense support rebases successfully.
        let present: Vec<u32> = bank.tables[0].u.0.iter().map(|f| f.0).collect();
        let order = bank.tables[0].u.order();
        let ok = RatingBasis {
            reference: vec![RatingReference {
                feature_set: present,
                coord: vec![0_u32; order],
            }],
        };
        bank.to_rating_export(
            model.link,
            &model.mode,
            &model.schema,
            &model.provenance,
            &model.schema.cat_encoders,
            Some(&ok),
        )
        .unwrap();

        // A reference to a support absent from the bank is a hard error, not a silent no-op.
        let absent = (model.schema.feature_names.len() as u32) + 7;
        let bad = RatingBasis {
            reference: vec![RatingReference {
                feature_set: vec![absent],
                coord: vec![0],
            }],
        };
        assert!(matches!(
            bank.to_rating_export(
                model.link,
                &model.mode,
                &model.schema,
                &model.provenance,
                &model.schema.cat_encoders,
                Some(&bad)
            ),
            Err(PbError::InvalidConfig { .. })
        ));
    }
}
