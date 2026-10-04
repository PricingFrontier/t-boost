//! # t-boost-core
//!
//! A depth-3 **oblivious** gradient boosting machine that is *exactly* decomposable
//! into low-order functional-ANOVA tables — the "rating tables" — without sacrificing
//! speed or accuracy. Every tree is a symmetric tree with one shared
//! `(feature, threshold)` test per level; later levels may reuse a feature to refine
//! lower-order surfaces rather than add an axis.
//!
//! Depth and order are **defaults, not limits**, and they are orthogonal knobs:
//! [`InteractionPolicy::max_depth`] (`3..=`[`engine::MAX_DEPTH`], default 3) sets how
//! finely a tree resolves its features, [`InteractionPolicy::max_order`]
//! (`1..=`[`engine::MAX_ORDER`], default 3) sets how many distinct raw features an effect
//! may couple — i.e. how many axes an exported table has. Both default to the historical 3,
//! and a fit that leaves them there is byte-identical to a pre-lift build's.
//!
//! **Exactness is not what the order cap protects.** The purification cascade is
//! n-dimensional (an arbitrary-rank `Tensor` centred by an arbitrary-rank odometer), so an
//! order-8 effect decomposes losslessly by the same algebra an order-3 one does, and the
//! five I2 gates are inherited rather than re-derived. What high order costs is
//! READABILITY, and that is priced explicitly: a doubling interaction-gain hurdle at every
//! `n → n+1` transition, a table-cell budget shrunk per order, the evidence-gated prune,
//! and mandatory heredity — which is combinatorial, and is the real wall (one surviving
//! order-`k` table implies its whole `2^k − 1` subset lattice).
//!
//! This crate is **pure Rust** (no Python; verified by the `NoPyo3` CI gate) and is
//! built to a high engineering standard enforced *structurally*:
//!
//! * `#![forbid(unsafe_code)]` on the whole core (SIMD arrives via safe wrappers).
//! * The no-panic gate — `unwrap`/`expect`/`panic`/`unreachable`/`indexing_slicing`
//!   are denied — so the single [`PbError`] enum is the only way to surface failure.
//! * `#![deny(missing_docs)]` — every public item is documented.
//! * The five lossless I2 invariants + the I1 feature budget are *real,
//!   build-blocking checks* ([`explain`]), live from the first commit.
//!
//! Module layout is 1:1 with the spec's section-ownership map (§4).
#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod backend;
pub mod banding;
pub mod boosters;
pub mod cat;
pub mod cell_refit;
pub mod constraints;
pub mod data;
pub mod engine;
pub mod error;
pub mod explain;
pub mod joint;
pub mod loss;
pub mod prune;
pub(crate) mod sched;
pub mod scoring;
pub mod serialize;
pub mod simd;
pub mod table_model;

// --- Canonical re-exports (spec §2 single source of truth). ---------------

pub use error::{Invariant, PbError};

pub use backend::{pb_rng, pb_seed, Stage};

pub use data::{
    bin, bin_columns, bin_serve_columns, bin_serve_columns_coded, bin_serve_columns_with,
    bin_train_columns, build_grid, compute_offset, AxisKind, AxisProvenance, BinConfig,
    BinnedMatrix, BorderFamily, BorderGrid, CatServeMaps, CategoricalColumn, FeatureId,
    FittedBinnedData, NumericColumn, ServeBinnedMatrix, ServeCategoricalCodes,
    ServeCategoricalColumn, TrainBinnedMatrix,
};

pub use cat::{
    exposure_weighted_base_rate, fit_cat_encoder, shrunken_encoding, CatEncoder, CatEncoderStore,
    CatFitSpec, CatLevel, CatTarget, LeakageScheme, Smooth, TsConfig, TsEncodingId,
};

pub use loss::{
    Gamma, GatedDeltaStep, GradHess, Link, Logistic, Loss, LossId, Metric, ObjectiveTag, Poisson,
    SquaredError, Tweedie, TWEEDIE_GATED_DELTA_STEP,
};

pub use engine::{
    BagFitReport, Booster, Config, DeltaStepGateReport, ExactnessMode, FitControl, FitSpec,
    GatedStepPolicy, GradScale, Hist, HistPrecision, InteractionGainHurdleMode, Model, ModelSchema,
    MultiClassModel, ObliviousTree, QuantGradHess, RoundEvent, RoundObserver, Sampling, Split,
    StopReason,
};

pub use constraints::{
    inverse_wht8_uniform, wht8_uniform, CredibilityFloor, InteractionPolicy, MonoSign, MonotoneMap,
    Wht8,
};

pub use explain::{
    assert_exact_decomposition, check_feature_budget, AxisId, EffectTable, ExactTol, FeatureSet,
    OverflowPolicy, PurifyMode, RefMeasure, SeBand, TableBank, TableBudget, Tensor,
};

pub use boosters::{
    average_banks, BoosterConfig, DartSpec, EnsembleSpec, HpGrid, NesterovSpec, RefitSpec,
};

pub use serialize::{
    decode_doc, decode_doc_json, decode_model, decode_model_json, decode_multiclass,
    decode_multiclass_json, encode_doc, encode_doc_json, encode_model, encode_model_json,
    encode_multiclass, encode_multiclass_json, is_multiclass_bytes, migrate, AxisExport, ModelDoc,
    MultiClassDoc, RatingBasis, RatingExport, RatingReference, RatingTable, FORMAT_VERSION,
    MULTICLASS_FORMAT_VERSION, MULTICLASS_KIND, SCHEMA_VERSION, SCHEMA_VERSION_DEPTH_LIFTED,
    SCHEMA_VERSION_HIGH_ORDER, SCHEMA_VERSION_ORDER_LIFTED, SCHEMA_VERSION_UNLIFTED,
};

pub use scoring::{CellMaps, PackedTree, ScoringBank, TableScoringBank};

pub use simd::{score_tile, CHUNK_ROWS};
