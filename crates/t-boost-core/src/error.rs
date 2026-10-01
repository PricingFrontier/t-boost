//! The typed-error firewall — `PbError` + `Invariant` (spec §2.8 / §02.4, OWNED HERE).
//!
//! Every fallible public function in the workspace returns `Result<T, PbError>`;
//! no public signature uses `Box<dyn Error>` (enforced by the `check-no-box-dyn`
//! grep-gate). `PbError` is the single mechanism that keeps "no panics in library
//! code" a build `[GATE]` rather than a convention: the no-panic clippy deny set
//! (`unwrap`/`expect`/`panic`/`unreachable`/`indexing_slicing`) leaves this enum as
//! the only way to surface a failure.

use core::fmt;

/// The one error type for the whole workspace (spec §2.8). Defined here as its
/// canonical home; no other module may redefine it.
///
/// Each variant carries just enough context to be actionable without allocating a
/// backtrace. `InvariantViolated` is the bridge to the I1/I2 contract: a broken
/// lossless property surfaces here, never as a panic.
///
/// # Examples
/// ```
/// use t_boost_core::PbError;
/// // A binning call given a non-finite border would return:
/// let e = PbError::InvalidInput { what: "border must be finite".into() };
/// assert!(e.to_string().contains("invalid input"));
/// ```
#[derive(thiserror::Error, Debug)]
pub enum PbError {
    /// Caller-supplied data is malformed (e.g. a non-finite feature value, an
    /// out-of-domain label for the objective). `what` describes the offending input.
    #[error("invalid input: {what}")]
    InvalidInput {
        /// Human-readable description of the malformed input.
        what: String,
    },

    /// A buffer or column had the wrong element type for the requested operation.
    /// `expected` names the type that was required.
    #[error("dtype mismatch: expected {expected}")]
    DtypeMismatch {
        /// The dtype the operation required (a `'static` label, never allocated).
        expected: &'static str,
    },

    /// Two arrays/axes that had to agree on length or rank did not. `what`
    /// identifies which dimensions disagreed.
    #[error("shape mismatch: {what}")]
    ShapeMismatch {
        /// Description of the mismatched shapes.
        what: String,
    },

    /// A configuration value is outside its legal range (e.g. `max_bin > 254`,
    /// `max_order` outside `1..=MAX_ORDER`). `what` describes the invalid setting.
    #[error("invalid config: {what}")]
    InvalidConfig {
        /// Description of the invalid configuration.
        what: String,
    },

    /// One of the I1/I2 lossless properties failed its `[GATE]` check. The
    /// `invariant` field names exactly which property broke.
    #[error("invariant violated: {invariant}")]
    InvariantViolated {
        /// Which lossless invariant was violated (see [`Invariant`]).
        invariant: Invariant,
    },

    /// An operation that cannot preserve exact decomposability was attempted on an
    /// `Exact` model (the typed firewall of §3). The string explains the breach.
    #[error("exactness firewall: {0}")]
    ExactnessFirewall(String),

    /// A `TableBank` table (or the whole bank) would exceed its cell budget (§08.10).
    /// The memory firewall: rather than silently truncate or coarsen a table — which
    /// would break Reconstruction — the build fails loudly. `what` names the offending
    /// support, `cells` is its projected `Π cells_i`, and `budget` is the ceiling hit.
    #[error("table budget exceeded: {what} would materialize {cells} cells (budget {budget})")]
    TableBudget {
        /// Which support (or "bank") overflowed.
        what: String,
        /// The projected cell count that exceeded the budget.
        cells: u64,
        /// The ceiling that was exceeded.
        budget: u64,
    },

    /// Serialization or deserialization failed, including a `schema_version` /
    /// `format_version` mismatch on load.
    #[error("serialization: {0}")]
    Serialization(String),

    /// An internal invariant of the implementation was violated — a bug. Used by
    /// fail-closed legacy seams and by `?`-propagated "this cannot happen" guards
    /// (so the no-panic gate is honored instead of `unreachable!`).
    #[error("internal bug: {what}")]
    Internal {
        /// Description of the internal failure.
        what: String,
    },
}

/// The six lossless properties the library upholds as build-blocking `[GATE]`s
/// (spec §3). I1 is `FeatureBudget`; the other five are the I2 decomposability
/// checks. Each maps to exactly one [`PbError::InvariantViolated`].
///
/// `Copy + Eq` so a check can name the broken property cheaply and tests can assert
/// on the exact variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Invariant {
    /// I1 — every tree is depth `1..=MAX_DEPTH` with `1..=min(depth, MAX_ORDER)` distinct
    /// raw features. (Historically `1..=3` on both counts; the depth and order lifts widened
    /// the range without changing the predicate — see `engine::i1_shape_ok`, the single
    /// canonical copy.)
    FeatureBudget,
    /// The ensemble is expressible as a constant plus ≤3rd-order tables — i.e. every
    /// table is PURE (each axis-slice has `w`-weighted mean zero, no lower-order mass
    /// left in a higher-order table). This is the variant the I2 purity check returns.
    Decomposability,
    /// Total signed `w`-mass is conserved across purification.
    MassConservation,
    /// Ensemble equals intercept + sum of purified tables, cell-by-cell within tol.
    Reconstruction,
    /// Total variance equals the sum of per-table variances (under product/uniform `w`).
    VarianceSum,
    /// Tree-sum = table-sum = Shapley-sum agree (exact n-Shapley ≤order-3).
    ThreeWayEqual,
}

impl fmt::Display for Invariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Invariant::FeatureBudget => "FeatureBudget (I1: depth-3, ≤3 distinct raw features)",
            Invariant::Decomposability => {
                "Decomposability (purity: every table slice has w-mean zero)"
            }
            Invariant::MassConservation => "MassConservation (signed w-mass conserved)",
            Invariant::Reconstruction => "Reconstruction (ensemble == intercept + Σ tables)",
            Invariant::VarianceSum => "VarianceSum (σ²(F) == Σ σ²(f_u))",
            Invariant::ThreeWayEqual => "ThreeWayEqual (tree == table == Shapley sum)",
        };
        f.write_str(name)
    }
}

impl PbError {
    /// Construct an [`PbError::InvariantViolated`] for `invariant`. A tiny helper so
    /// the five `explain` checks read as `return Err(PbError::invariant(..))`.
    #[must_use]
    pub fn invariant(invariant: Invariant) -> Self {
        PbError::InvariantViolated { invariant }
    }
}
