#![no_main]

use libfuzzer_sys::fuzz_target;
use t_boost_core::serialize::{
    decode_multiclass_tables, decode_multiclass_tables_json, decode_tables, decode_tables_json,
};
use t_boost_core::{decode_doc, decode_doc_json, decode_multiclass, decode_multiclass_json, Model};

// Exercises all four deserialization entry points (spec §10 / §02.8) on the SAME raw
// input, undispatched: the bare ModelDoc envelope has no magic prefix, while the three
// newer envelopes (multiclass/tables/multiclass-tables) each self-select via their own
// 4-byte magic check inside the decoder, so no manual dispatch is needed here. Before
// this, only decode_doc/decode_doc_json/Model::from_json were fuzzed — the three newer
// magic-prefixed decoders (TBMC/TBTM/TBMT), which decode deeply nested TableBank/
// Tensor/sparse structures from untrusted bytes, had zero fuzz coverage.
// fuzz_deserialize.dict seeds the
// three magic prefixes so libFuzzer's mutator can clear each decoder's initial gate
// instead of relying on chance to guess 4 specific bytes.
fuzz_target!(|data: &[u8]| {
    let _ = decode_doc(data);
    let _ = decode_multiclass(data);
    let _ = decode_tables(data);
    let _ = decode_multiclass_tables(data);
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = decode_doc_json(s);
        let _ = decode_multiclass_json(s);
        let _ = decode_tables_json(s);
        let _ = decode_multiclass_tables_json(s);
        let _ = Model::from_json(s);
    }
});
