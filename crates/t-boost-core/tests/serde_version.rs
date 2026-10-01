//! Serde round-trip + `schema_version`/`format_version` gate (spec §02.10(4), plan
//! F8). The bytes deployed must equal the bytes audited (DECOMPOSABLE), and a blob
//! from a newer wire version must be rejected — not silently mis-decoded.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use t_boost_core::explain::fixture_model;
use t_boost_core::{
    decode_doc, decode_model, encode_doc, encode_model, ModelDoc, PbError, FORMAT_VERSION,
    SCHEMA_VERSION, SCHEMA_VERSION_UNLIFTED,
};

#[test]
fn bincode_round_trip_is_bit_identical() {
    let model = fixture_model();
    let bytes = encode_model(&model).unwrap();
    let back = decode_model(&bytes).unwrap();
    assert_eq!(model, back);
    assert_eq!(
        bytes,
        encode_model(&back).unwrap(),
        "round-trip is not stable"
    );
}

#[test]
fn bumped_versions_are_rejected() {
    let model = fixture_model();

    let mut doc = ModelDoc::new(model.clone());
    doc.format_version = FORMAT_VERSION + 1;
    let bytes = encode_doc(&doc).unwrap();
    assert!(
        matches!(decode_doc(&bytes), Err(PbError::Serialization(_))),
        "a newer format_version must be rejected"
    );

    let mut doc = ModelDoc::new(model);
    doc.schema_version = SCHEMA_VERSION + 1;
    let bytes = encode_doc(&doc).unwrap();
    assert!(
        matches!(decode_doc(&bytes), Err(PbError::Serialization(_))),
        "a newer schema_version must be rejected"
    );
}

#[test]
fn current_version_doc_loads() {
    let doc = ModelDoc::new(fixture_model());
    let bytes = encode_doc(&doc).unwrap();
    let back = decode_doc(&bytes).unwrap();
    assert_eq!(doc, back);
    assert_eq!(back.format_version, FORMAT_VERSION);
    // The stamp is the MINIMUM a reader needs; this fixture is unlifted (depth <= 3), so
    // it is written at the pre-lift version and stays loadable by a pre-lift build.
    assert_eq!(back.schema_version, SCHEMA_VERSION_UNLIFTED);
    const _: () = assert!(SCHEMA_VERSION_UNLIFTED <= SCHEMA_VERSION);
}
