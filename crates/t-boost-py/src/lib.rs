//! PyO3 bindings for t-boost (spec §12).
//!
//! `#![allow(unsafe_code)]` is required here (and is the single justified,
//! encapsulated exception to the core's `forbid`): the pyo3/numpy procedural macros
//! expand to `unsafe`. The pure Rust core carries `#![forbid(unsafe_code)]`; this crate
//! remains a thin FFI adapter and owns no model math.
#![allow(unsafe_code)]

use numpy::{
    IntoPyArray, PyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2, PyReadwriteArray1, PyUntypedArrayMethods,
};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyDict, PyTuple, PyType};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, OnceLock};
use t_boost_core::boosters::{
    BoosterConfig, CellRefit, DartSpec, EnsembleSpec, NesterovSpec, RefitSpec,
};
use t_boost_core::cat::{
    post_pooling_level_count, CatTarget, LeakageScheme, Smooth, TsConfig, TsEncodingId,
};
use t_boost_core::constraints::{CredibilityFloor, InteractionPolicy, MonoSign, MonotoneMap};
use t_boost_core::data::{
    bin, bin_columns, bin_serve_columns, bin_serve_columns_coded, bin_serve_columns_with,
    bin_train_columns_with_holdout, AxisKind, AxisProvenance, BinConfig, BinnedMatrix,
    CatServeMaps, CategoricalColumn, FeatureId, NumericColumn, ServeBinnedMatrix,
    ServeCategoricalCodes, ServeCategoricalColumn,
};
use t_boost_core::engine::{
    Booster, Config, FitSpec, GatedStepPolicy, HistPrecision, InteractionGainHurdleMode, Model,
    MultiClassModel, Sampling,
};
use t_boost_core::error::{Invariant, PbError};
use t_boost_core::explain::{
    purify_raw_effects, FeatureSet, OverflowPolicy, RawEffect, RefMeasure, TableBank, TableBudget,
    Tensor,
};
use t_boost_core::loss::{
    Gamma, GatedDeltaStep, Link, Logistic, Loss, LossId, Poisson, SquaredError, Tweedie,
};
use t_boost_core::prune::{
    bag_banks_for_keepset, bag_oob_evidence_available, bag_oob_group_sums_with,
    bag_raw_scores_for_rows, multiclass_bag_oob_evidence_available, multiclass_carve_arms,
    multiclass_full_tables, multiclass_oob_arms, multiclass_prune_guard, multiclass_ranked_path,
    multiclass_realized_supports, prune_model_to_keepset, prune_model_to_tables,
    prune_model_to_tables_pinned, prune_multiclass_to_keepset,
    prune_multiclass_to_keepset_budgeted, prune_multiclass_to_tables, MulticlassGuardEvidence,
    MulticlassGuardReport, PruneConfig, PruneReport, PruneTableScore, DEFAULT_TABLE_MIN_ARITY,
};
use t_boost_core::serialize::{
    decode_multiclass, decode_multiclass_json, encode_multiclass, encode_multiclass_json,
    RatingBasis,
};
use t_boost_core::table_model::{MultiClassTableModel, TableModel};
use t_boost_core::CellMaps;

create_exception!(
    _t_boost,
    TBoostError,
    PyException,
    "Base t-boost exception."
);
create_exception!(
    _t_boost,
    InvariantError,
    TBoostError,
    "A lossless decomposability invariant failed."
);
create_exception!(
    _t_boost,
    ExactnessError,
    TBoostError,
    "An exact-only operation was attempted on an approximate model."
);
create_exception!(
    _t_boost,
    SerializationError,
    TBoostError,
    "Model serialization or deserialization failed."
);
create_exception!(
    _t_boost,
    InternalError,
    TBoostError,
    "An internal t-boost implementation invariant failed."
);

/// Build (once, cached) a class that is both `TBoostError` and `builtin_base` via genuine
/// Python multiple inheritance: `type(name, (TBoostError, builtin_base), {"__doc__": doc})`
/// — exactly what `class Name(TBoostError, builtin_base): pass` would produce. This backs
/// [`t_boost_value_error_type`]/[`t_boost_type_error_type`] (spec §12.7): `except
/// ValueError`/`except TypeError` (the spec-promised type) AND `except
/// t_boost.TBoostError` (the pre-existing catch-all) both match the SAME raised instance.
/// No raw FFI — `type` is just called like any other Python callable through the safe API
/// (`create_exception!`/`PyErr::new_type` only support a single base, so they can't do this).
fn dual_exception_type<'py>(
    py: Python<'py>,
    cell: &'static PyOnceLock<Py<PyType>>,
    name: &str,
    builtin_base: Bound<'py, PyType>,
    doc: &str,
) -> PyResult<Bound<'py, PyType>> {
    cell.get_or_try_init(py, || -> PyResult<Py<PyType>> {
        let bases = PyTuple::new(py, [py.get_type::<TBoostError>(), builtin_base])?;
        let namespace = PyDict::new(py);
        namespace.set_item("__doc__", doc)?;
        let ty = py
            .get_type::<PyType>()
            .call1((name, bases, namespace))?
            .cast_into::<PyType>()?;
        Ok(ty.unbind())
    })
    .map(|ty| ty.bind(py).clone())
}

fn t_boost_value_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    static CELL: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    dual_exception_type(
        py,
        &CELL,
        "TBoostValueError",
        py.get_type::<PyValueError>(),
        "Invalid input, shape mismatch, or invalid config (spec \u{a7}12.7). Both a \
         ValueError (catch with `except ValueError`) and a t_boost.TBoostError (catch \
         with `except t_boost.TBoostError`).",
    )
}

fn t_boost_type_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    static CELL: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    dual_exception_type(
        py,
        &CELL,
        "TBoostTypeError",
        py.get_type::<PyTypeError>(),
        "A dtype mismatch (spec \u{a7}12.7). Both a TypeError and a t_boost.TBoostError.",
    )
}

/// Raise [`t_boost_value_error_type`]. Attaches to the interpreter internally so `py_err`'s
/// external signature — `fn(PbError) -> PyErr`, called as `.map_err(py_err)` at ~100 sites —
/// never had to change; every call site already runs with the GIL held (this only ever fires
/// after `py.detach(..)` has returned and reattached, never from inside a detached closure),
/// so this is a cheap, reentrant attach, not a fresh acquisition.
fn t_boost_value_error(message: String) -> PyErr {
    Python::attach(|py| match t_boost_value_error_type(py) {
        Ok(ty) => PyErr::from_type(ty, message),
        Err(err) => err,
    })
}

/// Raise [`t_boost_type_error_type`]; see [`t_boost_value_error`].
fn t_boost_type_error(message: String) -> PyErr {
    Python::attach(|py| match t_boost_type_error_type(py) {
        Ok(ty) => PyErr::from_type(ty, message),
        Err(err) => err,
    })
}

#[derive(Debug, Clone)]
enum Objective {
    SquaredError,
    Logistic,
    Poisson,
    Gamma,
    Tweedie { rho: f32 },
}

enum LossChoice {
    SquaredError(SquaredError),
    Logistic(Logistic),
    Poisson(Poisson),
    Gamma(Gamma),
    Tweedie(Tweedie),
}

impl LossChoice {
    fn as_loss(&self) -> &dyn Loss {
        match self {
            LossChoice::SquaredError(loss) => loss,
            LossChoice::Logistic(loss) => loss,
            LossChoice::Poisson(loss) => loss,
            LossChoice::Gamma(loss) => loss,
            LossChoice::Tweedie(loss) => loss,
        }
    }
}

impl Objective {
    fn parse(name: Option<String>, tweedie_rho: f32) -> Result<Self, PbError> {
        let normalized = name
            .unwrap_or_else(|| "squared_error".to_owned())
            .replace('-', "_")
            .to_ascii_lowercase();
        match normalized.as_str() {
            "squared_error" | "squarederror" | "l2" | "regression" => Ok(Self::SquaredError),
            "logistic" | "binary_logloss" | "log_loss" | "classifier" => Ok(Self::Logistic),
            "poisson" => Ok(Self::Poisson),
            "gamma" => Ok(Self::Gamma),
            "tweedie" => {
                Tweedie::new(tweedie_rho)?;
                Ok(Self::Tweedie { rho: tweedie_rho })
            }
            other => Err(PbError::InvalidConfig {
                what: format!("unknown objective `{other}`"),
            }),
        }
    }

    fn instantiate(&self) -> Result<LossChoice, PbError> {
        match self {
            Objective::SquaredError => Ok(LossChoice::SquaredError(SquaredError)),
            Objective::Logistic => Ok(LossChoice::Logistic(Logistic)),
            Objective::Poisson => Ok(LossChoice::Poisson(Poisson)),
            Objective::Gamma => Ok(LossChoice::Gamma(Gamma)),
            Objective::Tweedie { rho } => Ok(LossChoice::Tweedie(Tweedie::new(*rho)?)),
        }
    }
}

/// Low-level Python booster wrapper.
#[pyclass(name = "_Booster", skip_from_py_object)]
#[derive(Clone)]
struct PyBooster {
    config: Config,
    bin_config: BinConfig,
    objective: Objective,
    credibility: CredibilityFloor,
    interaction: InteractionPolicy,
    cat_config: TsConfig,
    // P1 multi-channel (design/multichannel-categoricals.md): `Some(cfg)` when `cat_channels`
    // requested the target-free count channel — `cfg` mirrors `cat_config`'s
    // `min_data_per_group` (same rare-pooling) with `target: CatTarget::Count`. `None` (the
    // default) emits no second axis at all — bit-identical to before this field existed.
    cat_count_config: Option<TsConfig>,
    // Count-channel cardinality gate (follow-on to P1 multi-channel): the count axis is only
    // emitted for a categorical feature whose post-rare-pooling distinct level count is >=
    // this threshold (the benchmark win this channel targets was measured on high-cardinality
    // categoricals specifically). Inert when `cat_count_config` is `None`. `0` disables the
    // gate (every categorical gets the count channel whenever `cat_channels` requests it,
    // matching the pre-gate behavior).
    cat_count_min_levels: u32,
    // P3 per-class-channel cardinality gate: the per-class axes are only emitted for a
    // categorical whose post-rare-pooling distinct level count is >= this. Defaults to 3, the
    // encoding-invariance floor: below 3 levels the axis partition is encoding-INVARIANT (a
    // 2-level feature admits exactly one split however its levels are ordered), so per-class
    // channels could only add axes without adding representable structure — the same argument
    // `TsConfig::direct_max_levels` already makes to exempt binary features. Raising it
    // restricts the channels to high-cardinality features the way `cat_count_min_levels` does
    // for the count channel. Inert unless `cat_channels.class_freq`.
    cat_class_freq_min_levels: u32,
    // P3 multi-channel (design/multichannel-categoricals.md §8): the resolved `cat_channels`
    // plan. `CatChannelPlan::default()` (mean-only) is bit-identical to before this field
    // existed; `class_freq` is honored ONLY on the K>=3 softmax path (`fit_multiclass_owned`),
    // where the label-mean TS it replaces is the documented ordinal-encoding weakness.
    cat_channels: CatChannelPlan,
    // Template for the P3 per-class frequency channels: `cat_config` with its `target` slot
    // left at the mean placeholder, cloned per class at fit time with
    // `CatTarget::ClassFreq { class }` stamped in (the class count is only known there).
    // `None` unless `cat_channels.class_freq`.
    cat_class_freq_config: Option<TsConfig>,
    seed: u64,
    n_jobs: Option<usize>,
    fit_pool_width: Option<usize>,
}

/// Resolve the `max_delta_step_gated` kwarg into [`GatedStepPolicy`] (spec §05.6 addendum).
///
/// The knob is deliberately TRI-STATE, because "use the objective's gate" and "no gate" are
/// different requests:
///
/// * `None` (the default) / `True` / `"auto"` / `"objective"` ⇒ [`GatedStepPolicy::Objective`]
///   — the objective decides. Tweedie ships a gate (`0.01` / `0.3`); every other objective
///   ships none, so this is inert for them.
/// * `False` / `"off"` / `"none"` ⇒ [`GatedStepPolicy::Off`] — never gate.
/// * `(collapse_threshold, capped_step)` (any 2-sequence) or
///   `{"collapse_threshold": ..., "capped_step": ...}` ⇒ [`GatedStepPolicy::On`] — explicit
///   parameters, which also ARM the gate on objectives that ship none.
///
/// Note that an explicit `max_delta_step` outranks all of these at fit time: the engine never
/// gates a caller-named cap.
fn parse_gated_step_policy(value: Option<&Bound<'_, PyAny>>) -> PyResult<GatedStepPolicy> {
    let Some(v) = value else {
        return Ok(GatedStepPolicy::Objective);
    };
    if v.is_none() {
        return Ok(GatedStepPolicy::Objective);
    }
    // `bool` before any numeric/sequence probing: Python's bool is an int subclass.
    if let Ok(b) = v.extract::<bool>() {
        return Ok(if b {
            GatedStepPolicy::Objective
        } else {
            GatedStepPolicy::Off
        });
    }
    if let Ok(s) = v.extract::<String>() {
        return match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "objective" | "default" | "on" => Ok(GatedStepPolicy::Objective),
            "off" | "none" | "disabled" => Ok(GatedStepPolicy::Off),
            other => Err(PyValueError::new_err(format!(
                "max_delta_step_gated: unknown mode {other:?} (expected 'auto' or 'off', a \
                 (collapse_threshold, capped_step) pair, a dict, True/False, or None)"
            ))),
        };
    }
    let (threshold, step) = if let Ok(d) = v.cast::<PyDict>() {
        let get = |k: &str| -> PyResult<f32> {
            d.get_item(k)?
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "max_delta_step_gated dict is missing required key {k:?}"
                    ))
                })?
                .extract::<f32>()
        };
        (get("collapse_threshold")?, get("capped_step")?)
    } else if let Ok(pair) = v.extract::<Vec<f32>>() {
        match pair.as_slice() {
            [a, b] => (*a, *b),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "max_delta_step_gated sequence must be (collapse_threshold, capped_step), \
                     got {} element(s)",
                    pair.len()
                )))
            }
        }
    } else {
        return Err(PyTypeError::new_err(
            "max_delta_step_gated must be None, a bool, 'auto'/'off', a \
             (collapse_threshold, capped_step) pair, or a dict with those keys",
        ));
    };
    let gate = GatedDeltaStep {
        collapse_threshold: threshold,
        capped_step: step,
    };
    gate.validate().map_err(py_err)?;
    Ok(GatedStepPolicy::On(gate))
}

/// Resolve the `hist_precision` kwarg into the core enum.
fn parse_hist_precision(name: Option<&str>) -> Result<HistPrecision, PbError> {
    match name.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("full") | Some("f64") | Some("fullf64") => Ok(HistPrecision::FullF64),
        Some("quantized") | Some("qhist") | Some("i32") | Some("quantizedi32") => {
            Ok(HistPrecision::QuantizedI32)
        }
        Some(other) => Err(PbError::InvalidConfig {
            what: format!("hist_precision must be 'full' or 'quantized', got {other:?}"),
        }),
    }
}

/// Resolve the `interaction_gain_hurdle_mode` kwarg into the core enum.
fn parse_interaction_gain_hurdle_mode(
    name: Option<&str>,
) -> Result<InteractionGainHurdleMode, PbError> {
    match name.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("fixed") | Some("static") => Ok(InteractionGainHurdleMode::Fixed),
        Some("adaptive") | Some("auto") => Ok(InteractionGainHurdleMode::Adaptive),
        Some(other) => Err(PbError::InvalidConfig {
            what: format!(
                "interaction_gain_hurdle_mode must be 'fixed' or 'adaptive', got {other:?}"
            ),
        }),
    }
}

/// Resolve the `subsample` kwarg into a row-sampling strategy (`None`/`1.0` ⇒ full rows;
/// `0 < r < 1` ⇒ MVS at that rate, with `mvs_min_rows` as the floor).
fn parse_sampling(subsample: Option<f32>, mvs_min_rows: u32) -> Sampling {
    match subsample {
        None => Sampling::Full,
        Some(rate) if rate >= 1.0 => Sampling::Full,
        Some(rate) => Sampling::Mvs {
            rate,
            min_rows: mvs_min_rows.max(1),
        },
    }
}

fn parse_cat_target(name: Option<&str>) -> Result<CatTarget, PbError> {
    match name.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("mean") | Some("rate") => Ok(CatTarget::Mean),
        Some("log_mean") | Some("logmean") | Some("log") => Ok(CatTarget::LogMean),
        Some(other) => Err(PbError::InvalidConfig {
            what: format!("cat_target must be 'mean' or 'log_mean', got {other:?}"),
        }),
    }
}

/// Which categorical channels one fit emits per native categorical column, resolved from the
/// `cat_channels` kwarg. Every field `false` except `mean` is today's (default) behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CatChannelPlan {
    /// The mean/log-mean target-statistic channel (`TsEncodingId(0)`), i.e. every fit before
    /// multi-channel existed.
    mean: bool,
    /// P1 target-free count/rarity channel (`TsEncodingId(1)`), cardinality-gated by
    /// `cat_count_min_levels`.
    count: bool,
    /// P3 per-class frequency channels (`TsEncodingId(2 + k)` for class `k`), MULTICLASS ONLY.
    class_freq: bool,
}

impl Default for CatChannelPlan {
    fn default() -> Self {
        Self {
            mean: true,
            count: false,
            class_freq: false,
        }
    }
}

impl CatChannelPlan {
    /// `true` when the plan is exactly the pre-multi-channel default, so the fit path can keep
    /// its bit-identical single-axis shape without inspecting individual fields.
    fn is_mean_only(self) -> bool {
        self == Self::default()
    }
}

/// Resolve the `cat_channels` kwarg (P1 multi-channel spec §4.5, P3 §8): `None` keeps today's
/// mean-only behavior (the default, bit-identical); a list is a forgiving set of channel names.
///
/// - `"count"` turns on the P1 target-free count/rarity channel as an EXTRA axis alongside the
///   mean-TS channel on every native categorical column.
/// - `"class_freq"` turns on the P3 per-class frequency channels (K>=3 softmax fits only; see
///   `CatTarget::ClassFreq`). Per the design's P3 wording ("**replace** the ordinal label-mean
///   encoding with per-class frequency channels") this SUPPRESSES the mean channel — the thing
///   it is replacing — unless `"mean"` is ALSO listed explicitly, which keeps both.
/// - `"mean"` is otherwise implied and ignored.
///
/// # Errors
/// [`PbError::InvalidConfig`] if a list entry is not `"mean"`/`"count"`/`"class_freq"`.
fn parse_cat_channels(channels: Option<&[String]>) -> Result<CatChannelPlan, PbError> {
    let Some(names) = channels else {
        return Ok(CatChannelPlan::default());
    };
    let mut mean_explicit = false;
    let mut plan = CatChannelPlan {
        mean: false,
        count: false,
        class_freq: false,
    };
    for name in names {
        match name.trim().to_ascii_lowercase().as_str() {
            "mean" => mean_explicit = true,
            "count" => plan.count = true,
            "class_freq" | "classfreq" | "class" => plan.class_freq = true,
            other => {
                return Err(PbError::InvalidConfig {
                    what: format!(
                        "cat_channels entries must be 'mean', 'count' or 'class_freq', got \
                         {other:?}"
                    ),
                })
            }
        }
    }
    // Mean stays on unless class_freq explicitly replaces it (design P3), and an empty/`["mean"]`
    // list is exactly the default.
    plan.mean = mean_explicit || !plan.class_freq;
    Ok(plan)
}

/// The channel axes to emit for ONE native categorical column, as ascending [`TsEncodingId`]s.
///
/// The single source of truth for channel admission — both `CategoricalColumn` assembly sites
/// (`fit_model_ambient`, `fit_multiclass_owned`) and the `feature_names` expansion consume this
/// same list, so the axis layout and the names can never disagree.
///
/// - Mean (`0`) whenever the plan keeps it, AND whenever the per-class channels that would have
///   replaced it are not actually emitted for this feature (wrong path, or gated out by
///   cardinality) — a categorical must never lose its target statistic, nor end up with zero
///   axes and silently vanish from the design.
/// - Count (`1`) when requested AND the feature clears `cat_count_min_levels`.
/// - Per-class frequency (`2 + k`, one per class) when requested AND `n_classes` is `Some`
///   (i.e. the K>=3 softmax path — the only place the ordinal label-mean defect exists) AND the
///   feature clears `cat_class_freq_min_levels` (default 3, the encoding-invariance floor).
///
/// The post-pooling level count is computed at most ONCE per feature and shared by both gates.
///
/// # Errors
/// Propagates [`post_pooling_level_count`]; [`PbError::InvalidConfig`] if the per-class channels
/// were requested for more than [`MAX_CLASS_FREQ_CLASSES`] classes (the `u8` encoder-id space).
#[allow(clippy::too_many_arguments)] // JUSTIFIED: each argument is an independent gate input.
fn categorical_channel_ids(
    levels: &[String],
    weight: Option<&[f32]>,
    exposure: Option<&[f32]>,
    plan: CatChannelPlan,
    cat_config: &TsConfig,
    cat_count_min_levels: u32,
    cat_class_freq_min_levels: u32,
    n_classes: Option<usize>,
) -> Result<Vec<TsEncodingId>, PbError> {
    let class_freq_wanted = plan.class_freq && n_classes.is_some();
    if plan.is_mean_only() || !(plan.count || class_freq_wanted) {
        // Fast path: the default plan (and any plan whose extra channels are all inert here)
        // never pays for the level count, so a mean-only fit is untouched by this function.
        return Ok(vec![TsEncodingId(MEAN_CHANNEL_ID)]);
    }
    let n_levels = post_pooling_level_count(
        levels,
        weight,
        exposure,
        cat_config.target,
        cat_config.min_data_per_group,
    )?;
    // Whether the per-class channels actually land on THIS feature. Load-bearing for the line
    // below: `plan.mean == false` only ever means "class_freq REPLACES the mean channel", so the
    // replacement must be conditioned on the replacement actually happening — per feature, not
    // once for the whole fit. Otherwise a feature the class gate skips (or any feature at all on
    // the single-output path, where `class_freq` is inert) would silently lose its target
    // statistic and be left with the count channel alone, or with nothing.
    let emit_class_freq = class_freq_wanted && n_levels >= cat_class_freq_min_levels as usize;
    let emit_mean = plan.mean || !emit_class_freq;

    let mut ids = Vec::new();
    if emit_mean {
        ids.push(TsEncodingId(MEAN_CHANNEL_ID));
    }
    if plan.count && n_levels >= cat_count_min_levels as usize {
        ids.push(TsEncodingId(COUNT_CHANNEL_ID));
    }
    if emit_class_freq {
        let k = n_classes.unwrap_or(0);
        if k > MAX_CLASS_FREQ_CLASSES {
            return Err(PbError::InvalidConfig {
                what: format!(
                    "cat_channels=['class_freq'] supports at most {MAX_CLASS_FREQ_CLASSES} \
                     classes (encoder ids are u8), got {k}"
                ),
            });
        }
        for class in 0..k {
            let id =
                u8::try_from(usize::from(CLASS_FREQ_CHANNEL_ID_BASE) + class).map_err(|_| {
                    PbError::Internal {
                        what: "class-frequency encoder id exceeded u8 after the class-count check"
                            .into(),
                    }
                })?;
            ids.push(TsEncodingId(id));
        }
    }
    debug_assert!(
        !ids.is_empty(),
        "`emit_mean` is true whenever the class channels are absent, so this cannot be empty"
    );
    if ids.is_empty() {
        // Unreachable given `emit_mean` above; kept as a total guard so a future channel edit
        // can never silently drop a categorical out of the design.
        ids.push(TsEncodingId(MEAN_CHANNEL_ID));
    }
    Ok(ids)
}

/// The [`TsConfig`] backing one emitted channel id, resolved against the fit's own configs.
/// Mirrors [`categorical_channel_ids`]'s id assignment — the two are only ever used together.
///
/// # Errors
/// [`PbError::Internal`] if an id has no config (only reachable if the two functions above
/// drifted apart).
fn cat_channel_config<'a>(
    id: TsEncodingId,
    cat_config: &'a TsConfig,
    count_config: Option<&'a TsConfig>,
    class_freq_configs: &'a [TsConfig],
) -> Result<&'a TsConfig, PbError> {
    match id.0 {
        MEAN_CHANNEL_ID => Ok(cat_config),
        COUNT_CHANNEL_ID => count_config.ok_or_else(|| PbError::Internal {
            what: "count channel admitted without a count config".into(),
        }),
        other => class_freq_configs
            .get(usize::from(other) - usize::from(CLASS_FREQ_CHANNEL_ID_BASE))
            .ok_or_else(|| PbError::Internal {
                what: format!("class-frequency channel {other} has no config"),
            }),
    }
}

/// Expand a caller-supplied `feature_names` list — always ONE NAME PER INPUT COLUMN
/// (`n_numeric` numeric names, then one name per RAW categorical feature; this is what
/// `sklearn.py`'s `_split_columns` builds, and is unaffected by `cat_channels`, since it
/// doesn't know which categoricals get a count channel) — into an AXIS-indexed list matching
/// the actual binned matrix (`model.schema.feature_names`'s own convention): one entry per
/// numeric axis, then for each categorical raw feature, its own name on its FIRST channel axis
/// followed by a derived `"{name}#..."` entry for every ADDITIONAL channel axis beyond it.
///
/// The base name goes on the first axis (never a suffixed one) because
/// `serialize::representative_axis_for_raw` reads exactly that axis's name for the collapsed
/// per-raw-feature rating table — so the exported table keeps the user's own feature name
/// whichever channel happens to be emitted first.
///
/// `cat_channel_ids[j]` is raw categorical feature `j`'s emitted encoder ids, ascending, in the
/// SAME order the `CategoricalColumn` assembly produced them, so this can never disagree with
/// the actual column layout — it consumes that same per-raw-feature plan rather than
/// re-deriving admission independently. Suffixes come from the ids themselves
/// ([`cat_channel_name_suffix`]).
///
/// # Errors
/// [`PbError::ShapeMismatch`] if `names.len() != n_numeric + cat_channel_ids.len()` (the
/// caller supplied the wrong number of INPUT column names — a real user error, reported
/// against the input shape they actually control, not the internal axis count they can't see);
/// [`PbError::Internal`] if a raw feature emitted no channel axis at all.
fn expand_feature_names_for_cat_channels(
    names: Vec<String>,
    n_numeric: usize,
    cat_channel_ids: &[Vec<TsEncodingId>],
) -> Result<Vec<String>, PbError> {
    let expected = n_numeric + cat_channel_ids.len();
    if names.len() != expected {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "feature_names len {} != n_input_features {} ({n_numeric} numeric + {} \
                 categorical)",
                names.len(),
                expected,
                cat_channel_ids.len()
            ),
        });
    }
    let capacity = n_numeric + cat_channel_ids.iter().map(Vec::len).sum::<usize>();
    let mut out = Vec::with_capacity(capacity);
    let mut names = names.into_iter();
    for _ in 0..n_numeric {
        out.push(names.next().ok_or_else(|| PbError::Internal {
            what: "feature_names exhausted before the numeric prefix ended".into(),
        })?);
    }
    for ids in cat_channel_ids {
        let base = names.next().ok_or_else(|| PbError::Internal {
            what: "feature_names exhausted walking categorical raw features".into(),
        })?;
        // `split_first` doubles as the "a raw feature must own at least one axis" check; the
        // first channel's own id is unused because that axis takes the base name verbatim.
        let (_first, rest) = ids.split_first().ok_or_else(|| PbError::Internal {
            what: "categorical raw feature emitted no channel axis".into(),
        })?;
        // Derive the extra names from `base` (a borrow) before moving `base` itself, so all of
        // them land in axis order.
        let extra: Vec<String> = rest
            .iter()
            .map(|id| format!("{base}{}", cat_channel_name_suffix(*id)))
            .collect();
        out.push(base);
        out.extend(extra);
    }
    Ok(out)
}

/// Expand an INPUT-indexed monotone map to the AXIS-indexed schema the fit actually runs
/// against — the exact analogue of [`expand_feature_names_for_cat_channels`], and required for
/// the same reason.
///
/// `build_monotone_map` keys signs `f{i}` by INPUT column (`n_numeric + n_cat`), because that is
/// the shape the caller supplies. `resolve_monotone` (engine/boost.rs) resolves those names
/// against `BinnedMatrix.data`, which carries one axis per admitted CHANNEL. The two agree only
/// while every categorical emits exactly one axis; as soon as any extra channel is admitted
/// (`cat_channels = ["mean", "count"]`, the benchmark's own recommended setting) every input
/// index at or after the first expanded categorical is SHIFTED. Nothing errored, because
/// `resolve_monotone` only rejects an out-of-range axis and a shifted index is still in range —
/// so a constraint was silently dropped or aimed at another feature's channel.
///
/// SEMANTICS, stated because it is a choice and not a derivation: a sign on a categorical is
/// applied to EVERY axis that raw feature emits, not just its first. The alternative — constrain
/// the mean channel only — leaves the other channels free to break the very monotonicity the
/// caller asked for, and a monotone constraint exists for filing, where a silently partial
/// guarantee is worse than a slightly strong one. A numeric column always owns exactly one axis,
/// so numeric constraints are unaffected either way.
///
/// # Errors
/// [`PbError::Internal`] if a raw categorical emitted no channel axis at all (the same invariant
/// `expand_feature_names_for_cat_channels` checks).
fn expand_monotone_map_for_cat_channels(
    map: &MonotoneMap,
    n_numeric: usize,
    cat_channel_ids: &[Vec<TsEncodingId>],
) -> Result<MonotoneMap, PbError> {
    if map.is_empty() {
        return Ok(map.clone());
    }
    let mut out = MonotoneMap::new();
    // Numeric axes are the identity prefix: `_split_columns` puts them first and the sign vector
    // is remapped to match before it ever reaches here.
    for i in 0..n_numeric {
        if let Some(sign) = map.get(&format!("f{i}")) {
            out.insert(format!("f{i}"), *sign);
        }
    }
    let mut axis = n_numeric;
    for (j, ids) in cat_channel_ids.iter().enumerate() {
        if ids.is_empty() {
            return Err(PbError::Internal {
                what: "categorical raw feature emitted no channel axis".into(),
            });
        }
        if let Some(sign) = map.get(&format!("f{}", n_numeric + j)) {
            for k in 0..ids.len() {
                out.insert(format!("f{}", axis + k), *sign);
            }
        }
        axis += ids.len();
    }
    Ok(out)
}

/// The synthetic `feature_names` suffix for a NON-first channel axis of a categorical raw
/// feature: `#mean` for the mean-TS channel, `#count` for the P1 count/rarity channel, and
/// `#cls{k}` for the P3 per-class frequency channel of class `k`. Kept as a total function of
/// the encoder id (never a positional guess) so the naming survives any channel subset — e.g.
/// a `class_freq`-only fit, where the mean channel is absent entirely.
fn cat_channel_name_suffix(id: TsEncodingId) -> String {
    match id.0 {
        MEAN_CHANNEL_ID => "#mean".to_string(),
        COUNT_CHANNEL_ID => "#count".to_string(),
        other => format!(
            "#cls{}",
            u32::from(other) - u32::from(CLASS_FREQ_CHANNEL_ID_BASE)
        ),
    }
}

/// `TsEncodingId` of the mean/log-mean target-statistic channel — the one and only channel
/// before multi-channel existed, so its id is pinned at 0 for wire compatibility.
const MEAN_CHANNEL_ID: u8 = 0;
/// `TsEncodingId` of the P1 target-free count/rarity channel.
const COUNT_CHANNEL_ID: u8 = 1;
/// First `TsEncodingId` of the P3 per-class frequency channels: class `k` gets
/// `CLASS_FREQ_CHANNEL_ID_BASE + k`, leaving ids 0/1 to the mean and count channels so an
/// existing model's ids never shift meaning.
const CLASS_FREQ_CHANNEL_ID_BASE: u8 = 2;
/// Largest class count the P3 per-class channels can address: ids are `u8` and start at
/// [`CLASS_FREQ_CHANNEL_ID_BASE`], so `2 + K - 1 <= u8::MAX`.
const MAX_CLASS_FREQ_CLASSES: usize =
    (u8::MAX as usize) + 1 - (CLASS_FREQ_CHANNEL_ID_BASE as usize);

#[cfg(test)]
mod monotone_axis_expansion_tests {
    #![allow(clippy::unwrap_used)]
    use super::{build_monotone_map, expand_monotone_map_for_cat_channels};
    use t_boost_core::cat::TsEncodingId;
    use t_boost_core::constraints::MonoSign;

    fn ids(v: &[u8]) -> Vec<TsEncodingId> {
        v.iter().map(|&i| TsEncodingId(i)).collect()
    }

    /// One axis per categorical: expansion is the identity. This is the pre-channel world and
    /// every fit that admits no extra channel must stay byte-identical.
    #[test]
    fn single_channel_is_identity() {
        // inputs: a(numeric), c1, c2  ->  constraint on c2 == input index 2
        let map = build_monotone_map(Some(&[0, 0, -1]), 3).unwrap();
        let axes = vec![ids(&[0]), ids(&[0])];
        let out = expand_monotone_map_for_cat_channels(&map, 1, &axes).unwrap();
        assert_eq!(out, map);
    }

    /// The regression: with a count channel admitted, input index 2 (`c2`) must reach the axes
    /// c2 owns (4 and 5), NOT axis 2 — which is `c1#count`.
    #[test]
    fn count_channel_shifts_categorical_constraints() {
        let map = build_monotone_map(Some(&[0, 0, -1]), 3).unwrap();
        // axes: 0 a | 1 c1 | 2 c1#count | 3 c2 | 4 c2#count
        let axes = vec![ids(&[0, 1]), ids(&[0, 1])];
        let out = expand_monotone_map_for_cat_channels(&map, 1, &axes).unwrap();
        assert_eq!(out.get("f3"), Some(&MonoSign::Decreasing));
        assert_eq!(out.get("f4"), Some(&MonoSign::Decreasing));
        assert_eq!(out.get("f2"), None, "must not land on c1's count axis");
        assert_eq!(out.get("f1"), None, "must not land on c1");
        assert_eq!(out.len(), 2);
    }

    /// Numeric constraints occupy the identity prefix and never move, whatever the categoricals
    /// expand to. This is why a numeric-only probe could not detect the defect.
    #[test]
    fn numeric_prefix_is_never_shifted() {
        let map = build_monotone_map(Some(&[1, -1, 0, 0]), 4).unwrap();
        let axes = vec![ids(&[0, 1]), ids(&[0, 1, 2, 3])];
        let out = expand_monotone_map_for_cat_channels(&map, 2, &axes).unwrap();
        assert_eq!(out.get("f0"), Some(&MonoSign::Increasing));
        assert_eq!(out.get("f1"), Some(&MonoSign::Decreasing));
        assert_eq!(out.len(), 2);
    }

    /// A sign on a categorical reaches EVERY axis that feature emits (see the helper's doc for
    /// why the partial alternative was rejected).
    #[test]
    fn sign_reaches_every_channel_of_its_own_feature() {
        let map = build_monotone_map(Some(&[0, 1]), 2).unwrap();
        let axes = vec![ids(&[0, 1, 2, 3])]; // mean + count + 2 class-freq axes
        let out = expand_monotone_map_for_cat_channels(&map, 1, &axes).unwrap();
        for f in ["f1", "f2", "f3", "f4"] {
            assert_eq!(out.get(f), Some(&MonoSign::Increasing), "{f}");
        }
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn empty_map_short_circuits() {
        let map = build_monotone_map(None, 3).unwrap();
        let axes = vec![ids(&[0, 1]), ids(&[0, 1])];
        assert!(expand_monotone_map_for_cat_channels(&map, 1, &axes)
            .unwrap()
            .is_empty());
    }
}

#[cfg(test)]
mod cat_count_gate_tests {
    #![allow(clippy::unwrap_used, clippy::indexing_slicing)]
    use super::{
        categorical_channel_ids, CatChannelPlan, CLASS_FREQ_CHANNEL_ID_BASE, COUNT_CHANNEL_ID,
        MEAN_CHANNEL_ID,
    };
    use t_boost_core::cat::{CatTarget, Smooth, TsConfig, TsEncodingId};

    fn levels_of(n: usize, label: &str) -> Vec<String> {
        std::iter::repeat_n(label.to_string(), n).collect()
    }

    fn ts(min_data_per_group: f32) -> TsConfig {
        TsConfig {
            target: CatTarget::Mean,
            smooth: Smooth::Fixed { m: 0.0 },
            min_data_per_group,
            ..TsConfig::default()
        }
    }

    fn count_plan() -> CatChannelPlan {
        CatChannelPlan {
            mean: true,
            count: true,
            class_freq: false,
        }
    }

    fn class_freq_plan(keep_mean: bool) -> CatChannelPlan {
        CatChannelPlan {
            mean: keep_mean,
            count: false,
            class_freq: true,
        }
    }

    /// `true` iff the emitted channel set includes the P1 count axis — the admission decision
    /// the `CategoricalColumn` assembly actually reads, now that `categorical_channel_ids` is
    /// its single source of truth.
    fn count_admitted(levels: &[String], min_data_per_group: f32, min_levels: u32) -> bool {
        categorical_channel_ids(
            levels,
            None,
            None,
            count_plan(),
            &ts(min_data_per_group),
            min_levels,
            3,
            None,
        )
        .unwrap()
        .contains(&TsEncodingId(COUNT_CHANNEL_ID))
    }

    /// The team lead's own acceptance scenario: a high-cardinality categorical (>= the
    /// default threshold of 20 distinct levels, no pooling since every level clears the
    /// floor) is admitted for the count channel; a low-cardinality one (a handful of
    /// levels) is not — mirroring "the high-card feature gets 2 axes, the low-card gets 1"
    /// at the level of the actual admission decision the CategoricalColumn assembly reads.
    #[test]
    fn high_cardinality_feature_is_admitted_low_cardinality_is_not() {
        // 25 distinct levels, 20 rows each -- comfortably above both the level-count
        // threshold (20) and the rare-pooling floor (10), so every level survives pooling
        // and the post-pooling count is exactly 25.
        let mut high_card = Vec::new();
        for i in 0..25 {
            high_card.extend(levels_of(20, &format!("level_{i}")));
        }
        assert!(count_admitted(&high_card, 10.0, 20));

        // 3 distinct levels, well above the rare-pooling floor individually, but far below
        // the cardinality gate's threshold of 20.
        let mut low_card = Vec::new();
        for label in ["a", "b", "c"] {
            low_card.extend(levels_of(20, label));
        }
        assert!(!count_admitted(&low_card, 10.0, 20));
    }

    #[test]
    fn threshold_zero_admits_every_cardinality() {
        let low_card = levels_of(20, "only_level");
        assert!(count_admitted(&low_card, 10.0, 0));
    }

    #[test]
    fn admission_uses_post_pooling_cardinality_not_raw_distinct_count() {
        // 30 distinct RAW labels, but each appears only once (weight 1, below a floor of
        // 10) -- every one of them pools into the single reserved rare bucket, so the
        // POST-pooling count is 1, not 30. A gate that (incorrectly) counted raw distinct
        // labels would wrongly admit this feature at a threshold of 20.
        let raw_distinct: Vec<String> = (0..30).map(|i| format!("rare_{i}")).collect();
        assert!(!count_admitted(&raw_distinct, 10.0, 20));
        // The same data with pooling disabled (`min_data_per_group=0.0`) keeps all 30 as
        // their own levels, clearing the threshold.
        assert!(count_admitted(&raw_distinct, 0.0, 20));
    }

    /// The default plan emits exactly the one mean axis, for any data — the bit-identity
    /// contract for every fit that never asks for a channel.
    #[test]
    fn default_plan_is_always_exactly_the_mean_axis() {
        let mut high_card = Vec::new();
        for i in 0..25 {
            high_card.extend(levels_of(20, &format!("level_{i}")));
        }
        for n_classes in [None, Some(4)] {
            let ids = categorical_channel_ids(
                &high_card,
                None,
                None,
                CatChannelPlan::default(),
                &ts(10.0),
                20,
                3,
                n_classes,
            )
            .unwrap();
            assert_eq!(ids, vec![TsEncodingId(MEAN_CHANNEL_ID)]);
        }
    }

    /// P3: `class_freq` emits one axis per class, ids `2..2+K`, and REPLACES the mean axis
    /// unless the caller kept it explicitly.
    #[test]
    fn class_freq_emits_one_axis_per_class_and_replaces_mean() {
        let mut high_card = Vec::new();
        for i in 0..25 {
            high_card.extend(levels_of(20, &format!("level_{i}")));
        }
        let replaced = categorical_channel_ids(
            &high_card,
            None,
            None,
            class_freq_plan(false),
            &ts(10.0),
            20,
            3,
            Some(4),
        )
        .unwrap();
        assert_eq!(
            replaced,
            (0..4)
                .map(|k| TsEncodingId(CLASS_FREQ_CHANNEL_ID_BASE + k))
                .collect::<Vec<_>>()
        );

        let kept = categorical_channel_ids(
            &high_card,
            None,
            None,
            class_freq_plan(true),
            &ts(10.0),
            20,
            3,
            Some(4),
        )
        .unwrap();
        assert_eq!(kept.first(), Some(&TsEncodingId(MEAN_CHANNEL_ID)));
        assert_eq!(kept.len(), 5);
        // Ascending ids: the canonical per-raw channel order `channel_axes_for_raw` re-derives.
        assert!(kept.windows(2).all(|w| w[0].0 < w[1].0));
    }

    /// P3 is multiclass-only: on the single-output path (`n_classes = None`) the same plan
    /// falls back to the mean axis rather than emitting nothing.
    #[test]
    fn class_freq_is_inert_without_a_class_count() {
        let mut high_card = Vec::new();
        for i in 0..25 {
            high_card.extend(levels_of(20, &format!("level_{i}")));
        }
        let ids = categorical_channel_ids(
            &high_card,
            None,
            None,
            class_freq_plan(false),
            &ts(10.0),
            20,
            3,
            None,
        )
        .unwrap();
        assert_eq!(ids, vec![TsEncodingId(MEAN_CHANNEL_ID)]);
    }

    /// P3 cardinality floor: a 2-level (post-pooling) categorical is encoding-INVARIANT, so it
    /// gets no per-class axes — and, since `class_freq` replaced the mean channel, it must
    /// still fall back to one mean axis rather than vanishing from the design.
    #[test]
    fn class_freq_below_the_level_floor_falls_back_to_the_mean_axis() {
        let mut binary_cat = Vec::new();
        for label in ["a", "b"] {
            binary_cat.extend(levels_of(20, label));
        }
        let ids = categorical_channel_ids(
            &binary_cat,
            None,
            None,
            class_freq_plan(false),
            &ts(10.0),
            20,
            3,
            Some(4),
        )
        .unwrap();
        assert_eq!(ids, vec![TsEncodingId(MEAN_CHANNEL_ID)]);
    }

    /// `cat_class_freq_min_levels` restricts the per-class channels to high-cardinality
    /// features exactly the way `cat_count_min_levels` does the count channel: a 10-level
    /// feature clears the default floor of 3 but not a raised one, and then falls back to the
    /// mean axis it would otherwise have replaced.
    #[test]
    fn class_freq_min_levels_gates_by_cardinality() {
        let mut mid_card = Vec::new();
        for i in 0..10 {
            mid_card.extend(levels_of(20, &format!("level_{i}")));
        }
        let admitted = categorical_channel_ids(
            &mid_card,
            None,
            None,
            class_freq_plan(false),
            &ts(10.0),
            20,
            3,
            Some(3),
        )
        .unwrap();
        assert_eq!(admitted.len(), 3, "10 levels clears the default floor of 3");

        let gated = categorical_channel_ids(
            &mid_card,
            None,
            None,
            class_freq_plan(false),
            &ts(10.0),
            20,
            20,
            Some(3),
        )
        .unwrap();
        assert_eq!(gated, vec![TsEncodingId(MEAN_CHANNEL_ID)]);
    }

    /// The mean channel is only ever REPLACED where the per-class channels actually land.
    ///
    /// Three ways `class_freq` can fail to reach a feature — wrong fit path, cardinality gate,
    /// or both — and in every one of them the feature must keep a target statistic. The
    /// regression this pins: `["count","class_freq"]` on the single-output path used to emit
    /// the COUNT channel alone for any feature clearing the count gate, silently deleting the
    /// mean-TS from a binary/regression fit that merely passed a multiclass-shaped knob.
    #[test]
    fn mean_survives_wherever_the_class_channels_do_not_land() {
        let mut high_card = Vec::new();
        for i in 0..25 {
            high_card.extend(levels_of(20, &format!("level_{i}")));
        }
        let plan = CatChannelPlan {
            mean: false,
            count: true,
            class_freq: true,
        };

        // (a) Single-output path: class_freq is inert, so this is exactly ["mean","count"].
        let single =
            categorical_channel_ids(&high_card, None, None, plan, &ts(10.0), 20, 3, None).unwrap();
        assert_eq!(
            single,
            vec![
                TsEncodingId(MEAN_CHANNEL_ID),
                TsEncodingId(COUNT_CHANNEL_ID)
            ],
            "the count channel must never be a feature's ONLY axis"
        );

        // (b) Multiclass, but the class gate skips this feature while the count gate admits it.
        let gated =
            categorical_channel_ids(&high_card, None, None, plan, &ts(10.0), 20, 100, Some(3))
                .unwrap();
        assert_eq!(
            gated,
            vec![
                TsEncodingId(MEAN_CHANNEL_ID),
                TsEncodingId(COUNT_CHANNEL_ID)
            ]
        );

        // (c) Both gates admit: the replacement really happens, so no mean axis.
        let replaced =
            categorical_channel_ids(&high_card, None, None, plan, &ts(10.0), 20, 3, Some(3))
                .unwrap();
        assert_eq!(replaced.first(), Some(&TsEncodingId(COUNT_CHANNEL_ID)));
        assert_eq!(replaced.len(), 4);
    }

    /// The `u8` encoder-id space is a hard bound on K, reported as a config error rather than
    /// silently wrapping into another channel's id.
    #[test]
    fn too_many_classes_is_a_config_error() {
        let mut high_card = Vec::new();
        for i in 0..25 {
            high_card.extend(levels_of(20, &format!("level_{i}")));
        }
        let err = categorical_channel_ids(
            &high_card,
            None,
            None,
            class_freq_plan(false),
            &ts(10.0),
            20,
            3,
            Some(300),
        );
        assert!(matches!(
            err,
            Err(t_boost_core::error::PbError::InvalidConfig { .. })
        ));
    }
}

/// The switchover blocker: `cat_channels=["mean","count"]` fit through the ACTUAL
/// `fit_model_ambient`/`fit_multiclass_owned` path (not just the encoder/hand-built-`Booster`
/// paths `cat.rs`'s own tests cover) used to crash with `feature_names len N != n_features M`
/// whenever any categorical's count channel was admitted, because `sklearn.py`'s
/// `axis_names` is always sized to the INPUT columns, never the resulting axis count. See
/// `expand_feature_names_for_cat_channels`'s doc for the fix.
#[cfg(test)]
mod cat_channels_fit_path_tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic
    )]
    use super::*;
    use t_boost_core::explain::RefMeasure;

    /// A `PyBooster` with `cat_channels=["mean","count"]` and a low
    /// `cat_min_data_per_group`/`cat_count_min_levels` so a small synthetic fixture can
    /// exercise real admission without needing thousands of rows. Built via a plain struct
    /// literal (NOT `PyBooster::new`, which returns `PyResult` and pulls in pyo3-ffi symbols
    /// this crate's `extension-module` build intentionally leaves unresolved outside an
    /// embedding Python interpreter — confirmed empirically: a `cargo test` calling `::new`
    /// fails to LINK, not just run, with `undefined symbol: PyErr_Restore` etc.) — every
    /// field here is exactly what `PyBooster::new`'s own defaults would set, so this
    /// constructs the same state `TBoostRegressor(cat_channels=["mean","count"],
    /// cat_count_min_levels=...)` would on the Python side.
    fn booster_with_count_channel(cat_count_min_levels: u32) -> PyBooster {
        booster_with_count_channel_and_floor(cat_count_min_levels, 5.0)
    }

    /// Like `booster_with_count_channel`, with an explicit `cat_min_data_per_group` floor —
    /// needed by fixtures whose fold-train subsets have too few rows per level for the 5.0
    /// default (which would rare-pool everything into one bucket and mask the very unseen-
    /// level scenario under test).
    fn booster_with_count_channel_and_floor(
        cat_count_min_levels: u32,
        cat_min_data_per_group: f32,
    ) -> PyBooster {
        PyBooster {
            // A small ensemble: this test is about the schema-naming fix, not fit quality, and
            // `Config::default()`'s 1000 trees on a 48-row/4-axis fixture can grow deep enough
            // to hit unrelated over-budget "factored effect" edge cases.
            config: Config {
                n_trees: 20,
                ..Config::default()
            },
            bin_config: BinConfig::default(),
            objective: Objective::SquaredError,
            credibility: CredibilityFloor::default(),
            // Order-1 only: this test is about the axis-count/schema-naming fix (a per-
            // feature concern), not the interaction/factored-effects machinery, which has
            // its own dedicated coverage elsewhere.
            interaction: InteractionPolicy {
                max_order: 1,
                ..InteractionPolicy::default()
            },
            cat_config: TsConfig {
                min_data_per_group: cat_min_data_per_group,
                ..TsConfig::default()
            },
            cat_count_config: Some(TsConfig {
                target: CatTarget::Count,
                min_data_per_group: cat_min_data_per_group,
                ..TsConfig::default()
            }),
            cat_count_min_levels,
            cat_class_freq_min_levels: 3,
            cat_channels: CatChannelPlan {
                mean: true,
                count: true,
                class_freq: false,
            },
            cat_class_freq_config: None,
            seed: 0,
            n_jobs: None,
            fit_pool_width: None,
        }
    }

    /// The team lead's exact regression scenario: a mix of a high-cardinality categorical
    /// (admitted for the count channel -> 2 axes) and a low-cardinality one (mean-only -> 1
    /// axis), fit with caller-supplied `feature_names` sized to the 3 INPUT columns (1
    /// numeric + 2 categorical) — never the resulting 4 axes — exactly what `sklearn.py`'s
    /// `axis_names` sends. Before the fix this raised `feature_names len 3 != n_features 4`.
    /// Also proves predict succeeds/is finite on the same layout, and that the collapsed
    /// rating export shows one table per RAW feature with no synthetic "#count" name leaking
    /// in anywhere — the two things the team lead asked to have verified, not assumed.
    #[test]
    fn cat_channels_fit_predict_and_export_survive_a_real_admitted_count_channel() {
        let n = 48;
        let numeric: Vec<f32> = (0..n).map(|i| i as f32).collect();
        // 8 distinct levels, 6 rows each -- clears both the rare-pooling floor (5.0) and the
        // cardinality gate (threshold 5) -> admitted for the count channel -> 2 axes.
        let cat_high: Vec<String> = (0..n).map(|i| format!("h{}", i % 8)).collect();
        // 3 distinct levels, 16 rows each -- clears the rare-pooling floor but NOT the
        // cardinality gate -> mean-only -> 1 axis.
        let cat_low: Vec<String> = (0..n).map(|i| format!("l{}", i % 3)).collect();
        // A deliberately NON-monotonic, per-level target so the booster actually has an
        // incentive to split on `cat_high` (a target that were merely a function of the row
        // index, like `i % 5`, correlates just as well or better with the NUMERIC feature,
        // and the booster would never touch `cat_high` at all -- defeating the point of this
        // fixture). `gcd(8, 3) == 1`, so this pattern carries no information for `cat_low` or
        // (being non-monotonic) for the numeric feature either.
        const LEVEL_TARGET: [f32; 8] = [0.0, 50.0, 5.0, 45.0, 10.0, 40.0, 15.0, 35.0];
        let y: Vec<f32> = (0..n).map(|i| LEVEL_TARGET[i as usize % 8]).collect();

        let state = booster_with_count_channel(5);
        let monotone_map = build_monotone_map(None, 3).unwrap();
        let feature_names = Some(vec![
            "num_a".to_string(),
            "cat_high".to_string(),
            "cat_low".to_string(),
        ]);
        let cat_x = vec![cat_high, cat_low];
        let model = fit_model_ambient(
            &state,
            std::slice::from_ref(&numeric),
            &y,
            None,
            None,
            feature_names,
            None,
            &monotone_map,
            Some(&cat_x),
            None,
        )
        .expect(
            "fit must succeed: this is the exact case that used to raise `feature_names len 3 \
             != n_features 4`",
        );

        // 1 numeric axis + 2 axes for the admitted high-card categorical + 1 axis for the
        // mean-only low-card categorical = 4 axes total.
        assert_eq!(model.provenance.len(), 4);
        assert_eq!(model.schema.feature_names.len(), 4);
        assert_eq!(model.schema.feature_names[0], "num_a");
        assert_eq!(model.schema.feature_names[1], "cat_high");
        assert_eq!(model.schema.feature_names[2], "cat_high#count");
        assert_eq!(model.schema.feature_names[3], "cat_low");
        model.validate().unwrap();

        // Predict must succeed and be finite (the serve path is already model-provenance-
        // driven, unaffected by this fix, but confirmed end-to-end anyway).
        let binned = serve_binned_for_model(&model, &[numeric], Some(&cat_x)).unwrap();
        let preds = model.predict(&binned, None).unwrap();
        assert_eq!(preds.len(), n as usize);
        assert!(
            preds.iter().all(|p| p.is_finite()),
            "every prediction must be finite"
        );

        // The collapsed rating export must show exactly one table/one axis per RAW feature
        // (never a 4th "raw feature" for the count channel), and NO synthetic "#count" name
        // may leak into it anywhere.
        let serve = ServeBinnedMatrix(binned);
        let bank = model.explain(&serve, RefMeasure::Uniform).unwrap();
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
        let raw_feature_ids: std::collections::BTreeSet<u32> = export
            .tables
            .iter()
            .flat_map(|t| t.feature_set.0.iter().map(|f| f.0))
            .collect();
        assert!(
            raw_feature_ids.iter().all(|&r| r < 3),
            "only 3 distinct raw features exist (num_a=0, cat_high=1, cat_low=2); a 4th would \
             mean the count channel leaked out as its own feature"
        );
        for table in &export.tables {
            for name in &table.feature_names {
                assert!(
                    !name.contains("#count"),
                    "table feature name `{name}` leaked a synthetic count-channel name"
                );
            }
            for axis in &table.axes {
                assert!(
                    !axis.name.contains("#count"),
                    "axis name `{}` leaked a synthetic count-channel name",
                    axis.name
                );
            }
        }
        let cat_high_tables: Vec<_> = export
            .tables
            .iter()
            .filter(|t| t.feature_set.0.iter().any(|f| f.0 == 1))
            .collect();
        assert_eq!(
            cat_high_tables.len(),
            1,
            "one table for the admitted-count raw feature, not one per channel"
        );
        assert_eq!(
            cat_high_tables[0].axes.len(),
            1,
            "one axis, not one per channel"
        );
    }

    /// The multiclass mirror of the test above: `fit_multiclass_owned`'s schema-name
    /// expansion was restructured differently (threaded out through the fit-pool closure's
    /// own return value, a tuple, rather than an outer-variable mutation) since `run_fit_pool`
    /// may run it on a freshly built rayon pool — worth its own direct check rather than
    /// trusting it compiles.
    #[test]
    fn cat_channels_multiclass_fit_survives_a_real_admitted_count_channel() {
        let n = 48;
        let numeric: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let cat_high: Vec<String> = (0..n).map(|i| format!("h{}", i % 8)).collect();
        let cat_low: Vec<String> = (0..n).map(|i| format!("l{}", i % 3)).collect();
        // 3-class label, deterministic per `cat_high` level (0,3,6 -> class 0; 1,4,7 -> class
        // 1; 2,5 -> class 2) so the booster has a real incentive to split on it.
        const LEVEL_CLASS: [f32; 8] = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0];
        let y: Vec<f32> = (0..n).map(|i| LEVEL_CLASS[i as usize % 8]).collect();

        let state = booster_with_count_channel(5);
        let feature_names = Some(vec![
            "num_a".to_string(),
            "cat_high".to_string(),
            "cat_low".to_string(),
        ]);
        let cat_x = vec![cat_high, cat_low];
        let model = fit_multiclass_owned_bagged(
            state,
            vec![numeric.clone()],
            y,
            3,
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            None,
            feature_names,
            None,
            Some(cat_x.clone()),
            None,
            None,
        )
        .expect("multiclass fit must succeed: the same feature_names-length bug applied here too");

        assert_eq!(model.classes.len(), 3);
        for class_model in &model.classes {
            assert_eq!(class_model.provenance.len(), 4);
            assert_eq!(class_model.schema.feature_names.len(), 4);
            assert_eq!(class_model.schema.feature_names[0], "num_a");
            assert_eq!(class_model.schema.feature_names[1], "cat_high");
            assert_eq!(class_model.schema.feature_names[2], "cat_high#count");
            assert_eq!(class_model.schema.feature_names[3], "cat_low");
            class_model.validate().unwrap();
        }

        let first = &model.classes[0];
        let binned = serve_binned_for_model(first, &[numeric], Some(&cat_x)).unwrap();
        let preds = first.predict(&binned, None).unwrap();
        assert!(preds.iter().all(|p| p.is_finite()));
    }

    /// A `PyBooster` configured exactly as `TBoostClassifier(cat_channels=["class_freq"])`
    /// would build it: the mean channel REPLACED by one per-class frequency channel per class
    /// (P3, design/multichannel-categoricals.md §8). `keep_mean` mirrors the
    /// `["mean","class_freq"]` spelling, which keeps both.
    fn booster_with_class_freq_channels(keep_mean: bool) -> PyBooster {
        let cat_config = TsConfig {
            min_data_per_group: 2.0,
            ..TsConfig::default()
        };
        PyBooster {
            config: Config {
                n_trees: 20,
                ..Config::default()
            },
            bin_config: BinConfig::default(),
            objective: Objective::SquaredError,
            credibility: CredibilityFloor::default(),
            interaction: InteractionPolicy {
                max_order: 1,
                ..InteractionPolicy::default()
            },
            cat_class_freq_config: Some(cat_config.clone()),
            cat_config,
            cat_count_config: None,
            cat_count_min_levels: 20,
            cat_class_freq_min_levels: 3,
            cat_channels: CatChannelPlan {
                mean: keep_mean,
                count: false,
                class_freq: true,
            },
            seed: 0,
            n_jobs: None,
            fit_pool_width: None,
        }
    }

    /// P3 end-to-end through the REAL multiclass fit path: `cat_channels=["class_freq"]` on a
    /// 3-class fixture emits one categorical axis PER CLASS for the high-cardinality feature,
    /// names them without colliding, fits, predicts finite logits, and — the load-bearing
    /// part — still decomposes EXACTLY into one table per RAW feature per class (never one per
    /// channel), i.e. the P1 lossless joint collapse generalizes past two channels.
    #[test]
    fn class_freq_multiclass_fit_collapses_k_channels_to_one_table_per_raw_feature() {
        let n = 60;
        let numeric: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
        let cat_high: Vec<String> = (0..n).map(|i| format!("h{}", i % 10)).collect();
        // 2 levels: below the level floor of 3, so this one falls back to a single mean axis
        // — the low-cardinality control, in the same fit.
        let cat_low: Vec<String> = (0..n).map(|i| format!("l{}", i % 2)).collect();
        // Deliberately NON-ordinal class structure: the 10 levels' class assignment has no
        // monotone relationship to the level index, so a label-mean TS is a lossy summary of
        // it. (This test asserts structure/exactness, not accuracy.)
        const LEVEL_CLASS: [f32; 10] = [2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0];
        let y: Vec<f32> = (0..n).map(|i| LEVEL_CLASS[i as usize % 10]).collect();

        let state = booster_with_class_freq_channels(false);
        let cat_x = vec![cat_high, cat_low];
        let model = fit_multiclass_owned_bagged(
            state,
            vec![numeric.clone()],
            y,
            3,
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            None,
            Some(vec![
                "num_a".to_string(),
                "cat_high".to_string(),
                "cat_low".to_string(),
            ]),
            None,
            Some(cat_x.clone()),
            None,
            None,
        )
        .expect("multiclass class_freq fit must succeed");

        assert_eq!(model.classes.len(), 3);
        for class_model in &model.classes {
            assert_eq!(
                class_model.schema.feature_names,
                vec![
                    "num_a".to_string(),
                    "cat_high".to_string(),
                    "cat_high#cls1".to_string(),
                    "cat_high#cls2".to_string(),
                    "cat_low".to_string(),
                ],
                "first axis of each raw feature keeps the user's own name; the rest are suffixed"
            );
            assert_eq!(class_model.provenance.len(), 5);
            class_model.validate().unwrap();
        }

        // Three encoders for cat_high (raw 1), one per class; cat_low (raw 2) fell back to mean.
        let first = &model.classes[0];
        for class in 0..3_u8 {
            first
                .schema
                .cat_encoders
                .get(
                    TsEncodingId(CLASS_FREQ_CHANNEL_ID_BASE + class),
                    FeatureId(1),
                )
                .unwrap_or_else(|e| panic!("cat_high class-{class} encoder missing: {e}"));
        }
        first
            .schema
            .cat_encoders
            .get(TsEncodingId(MEAN_CHANNEL_ID), FeatureId(2))
            .expect("the 2-level control keeps its single mean axis");

        let binned = serve_binned_for_model(first, &[numeric], Some(&cat_x)).unwrap();
        let preds = first.predict(&binned, None).unwrap();
        assert!(preds.iter().all(|p| p.is_finite()));

        // The five invariant checks, per class, on a model whose categorical carries 3 channels.
        let serve = ServeBinnedMatrix(binned);
        for class_model in &model.classes {
            let bank = class_model.explain(&serve, RefMeasure::Uniform).unwrap();
            t_boost_core::explain::assert_exact_decomposition(class_model, &bank, &serve).unwrap();
            let high_tables: Vec<_> = bank
                .tables
                .iter()
                .filter(|t| t.u.contains(FeatureId(1)))
                .collect();
            assert_eq!(
                high_tables.len(),
                1,
                "one table for the 3-channel raw feature, not one per channel"
            );
            assert_eq!(
                high_tables[0].axes.len(),
                1,
                "one axis, not one per channel"
            );
        }
    }

    /// `["mean","class_freq"]` keeps the label-mean axis alongside the per-class ones, and the
    /// emitted ids stay ascending (`channel_axes_for_raw`'s canonical order).
    #[test]
    fn class_freq_can_keep_the_mean_channel_alongside() {
        let n = 60;
        let numeric: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
        let cat_high: Vec<String> = (0..n).map(|i| format!("h{}", i % 10)).collect();
        const LEVEL_CLASS: [f32; 10] = [2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0];
        let y: Vec<f32> = (0..n).map(|i| LEVEL_CLASS[i as usize % 10]).collect();
        let model = fit_multiclass_owned_bagged(
            booster_with_class_freq_channels(true),
            vec![numeric],
            y,
            3,
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            None,
            Some(vec!["num_a".to_string(), "cat_high".to_string()]),
            None,
            Some(vec![cat_high]),
            None,
            None,
        )
        .expect("mean + class_freq fit must succeed");
        assert_eq!(
            model.classes[0].schema.feature_names,
            vec![
                "num_a".to_string(),
                "cat_high".to_string(),
                "cat_high#cls0".to_string(),
                "cat_high#cls1".to_string(),
                "cat_high#cls2".to_string(),
            ]
        );
    }

    /// The 3rd switchover bug: `fit_prune_selection`'s underlying K-fold CV
    /// (`fit_prune_reports_owned`) fits each fold's model on the fold's OWN train subset,
    /// then scores it against the FULL data (`serve_binned_for_model(&fold_model, &columns,
    /// ...)`, not just that fold's held rows) via `prune_model_to_tables`. On a high-card
    /// categorical, a level can appear in a fold's held rows but be entirely ABSENT from that
    /// fold's train rows — genuinely unseen to that fold's frozen encoder. This builds a
    /// fixture with EXACTLY that property: level "h0" lands only on rows assigned to fold 0,
    /// so fold 0's train subset (fold_of == 1) never sees it at all.
    fn fold_of_and_cat_high_with_one_level_unseen_by_fold_zeros_train() -> (Vec<i64>, Vec<String>) {
        // "h0" is EXCLUSIVELY fold_of=0 -- unseen to fold 0's train (fold_of==1). "h1".."h7"
        // are EXCLUSIVELY fold_of=1, with DELIBERATELY VARIED row counts (2,3,4,5,6,7,8) so
        // their count-channel encodings actually differ from each other (equal counts would
        // give every level the IDENTICAL count encoding, collapsing the count channel to one
        // bin and making it impossible for an unseen level's base tuple to miss every seen
        // level's tuple -- the mistake in this fixture's first draft).
        let mut fold_of = Vec::new();
        let mut cat_high = Vec::new();
        for _ in 0..4 {
            cat_high.push("h0".to_string());
            fold_of.push(0_i64);
        }
        for (level, count) in (1..8).zip([2, 3, 4, 5, 6, 7, 8]) {
            for _ in 0..count {
                cat_high.push(format!("h{level}"));
                fold_of.push(1_i64);
            }
        }
        (fold_of, cat_high)
    }

    /// Builds the shared fixture (numeric distractor + the `cat_high` design above + a
    /// non-monotonic per-level target, same reasoning as the earlier fit test: a target that
    /// were merely a function of row order would let the booster ignore `cat_high` entirely,
    /// defeating the point of exercising its `JointCatAxis`).
    fn cv_pruning_fixture() -> (Vec<f32>, Vec<String>, Vec<f32>, Vec<i64>) {
        let (fold_of, cat_high) = fold_of_and_cat_high_with_one_level_unseen_by_fold_zeros_train();
        let n = cat_high.len();
        let numeric: Vec<f32> = (0..n).map(|i| i as f32).collect();
        const LEVEL_TARGET: [f32; 8] = [0.0, 50.0, 5.0, 45.0, 10.0, 40.0, 15.0, 35.0];
        let y: Vec<f32> = cat_high
            .iter()
            .map(|label| {
                let idx: usize = label
                    .strip_prefix('h')
                    .and_then(|s| s.parse().ok())
                    .expect("well-formed h<N> label");
                LEVEL_TARGET[idx]
            })
            .collect();
        (numeric, cat_high, y, fold_of)
    }

    /// `fit_prune_reports_owned` (the function underlying `PyBooster::fit_prune_selection`,
    /// the actual CV-pruning path) with `cat_channels=["mean","count"]`, on a fixture where
    /// fold 0's train subset never sees level "h0" at all (it's exclusively in fold 0's held
    /// rows) — the exact scenario the team lead's real-data traceback hit.
    ///
    /// CONFIRMED REPRODUCED PRE-FIX: this exact fixture, run against `fit_prune_reports_owned`
    /// before any fix to `JointCatAxis::build`, returned
    /// `Err(InvalidInput { what: "joint categorical axis: bin combination [4, 1] was never \
    /// seen at fit time" })` — the same error family/shape as the team lead's reported
    /// `[188, 1]` (different bin values, same root cause, same code path). The first draft of
    /// this fixture (equal row-counts per level) did NOT reproduce it: with every level
    /// equally represented, the count channel's encoding is IDENTICAL for every level
    /// (`log1p(share * N)` is the same when every level's share is equal), collapsing the
    /// count channel to a single bin and making it impossible for an unseen level's base
    /// tuple to miss every seen level's tuple. Deliberately varying `cat_high`'s per-level
    /// row counts (below) gives the count channel genuine multi-bin structure and reliably
    /// reproduces the crash.
    #[test]
    fn cv_pruning_survives_a_level_unseen_by_one_folds_train() {
        let (numeric, cat_high, y, fold_of) = cv_pruning_fixture();
        // A floor of 1.0 (not the shared 5.0 default): fold 0's train has as few as 2 rows
        // for one level (h1) -- a 5.0 floor would rare-pool it away, defeating the deliberate
        // count-channel variation this fixture depends on.
        let state = booster_with_count_channel_and_floor(5, 1.0);
        let monotone_map = build_monotone_map(None, 2).unwrap();
        let feature_names = Some(vec!["num_a".to_string(), "cat_high".to_string()]);
        let reports = fit_prune_reports_owned(
            state,
            vec![numeric],
            y,
            fold_of,
            2,
            None,
            None,
            feature_names,
            None,
            monotone_map,
            Some(vec![cat_high]),
            RefMeasure::Uniform,
            0.0,
            0.0,
            0.0,
            DEFAULT_TABLE_MIN_ARITY,
            1,
            false,
            None,
            // pinned-bank fold fidelity off: these gates pin the HISTORICAL selection path.
            None,
        )
        .expect(
            "CV-pruning must succeed once an unseen level resolves to the base cell instead \
             of erroring",
        );
        assert_eq!(reports.len(), 2, "one report slot per fold");
        assert!(
            reports.iter().any(Option::is_some),
            "at least one fold must have produced a real report"
        );
    }

    /// The mandatory "unseen levels get the base relativity" check, verified directly rather
    /// than inferred: fits a model on ONLY the levels `cv_pruning_fixture` puts in fold 0's
    /// train (h1..h7, never h0), then predicts on TWO DIFFERENT labels neither of which was
    /// ever seen by this encoder -- "h0" (the fixture's own held-out level) and a wholly
    /// fictional label. Both are genuinely unseen, so both must encode to the SAME `base` on
    /// every channel and therefore predict IDENTICALLY -- the direct, checkable consequence
    /// of "an unseen level gets the base relativity" (if it got anything else, or got
    /// something that depended on WHICH unseen label it was, this equality would fail).
    #[test]
    fn unseen_levels_get_the_base_relativity_regardless_of_which_label() {
        let (fold_of, cat_high) = fold_of_and_cat_high_with_one_level_unseen_by_fold_zeros_train();
        // Keep ONLY fold 0's train rows (fold_of == 1): h1..h7, never h0.
        let train_idx: Vec<usize> = fold_of
            .iter()
            .enumerate()
            .filter_map(|(i, &f)| (f == 1).then_some(i))
            .collect();
        let cat_high_train: Vec<String> = train_idx.iter().map(|&i| cat_high[i].clone()).collect();
        let n = cat_high_train.len();
        let numeric_train: Vec<f32> = (0..n).map(|i| i as f32).collect();
        const LEVEL_TARGET: [f32; 8] = [0.0, 50.0, 5.0, 45.0, 10.0, 40.0, 15.0, 35.0];
        let y_train: Vec<f32> = cat_high_train
            .iter()
            .map(|label| {
                let idx: usize = label
                    .strip_prefix('h')
                    .and_then(|s| s.parse().ok())
                    .expect("well-formed h<N> label");
                LEVEL_TARGET[idx]
            })
            .collect();

        let state = booster_with_count_channel_and_floor(5, 1.0);
        let monotone_map = build_monotone_map(None, 1).unwrap();
        let model = fit_model_ambient(
            &state,
            &[numeric_train],
            &y_train,
            None,
            None,
            None,
            None,
            &monotone_map,
            Some(&[cat_high_train]),
            None,
        )
        .unwrap();

        // Two rows, same numeric value, two DIFFERENT labels -- neither ever seen by this
        // encoder (it was fit on h1..h7 only).
        let cat_x = vec![vec![
            "h0".to_string(),
            "totally_fictional_label".to_string(),
        ]];
        let binned = serve_binned_for_model(&model, &[vec![0.0, 0.0]], Some(&cat_x)).unwrap();
        let preds = model.predict(&binned, None).unwrap();
        assert_eq!(preds.len(), 2);
        assert!(preds.iter().all(|p| p.is_finite()));
        assert_eq!(
            preds[0], preds[1],
            "two different genuinely-unseen labels must predict identically (both resolve to \
             the SAME base cell) -- an unseen level's relativity must not depend on which \
             unseen label it happens to be"
        );
    }

    /// A VehModel-like synthetic fixture: a single high-cardinality categorical (350 levels,
    /// varied row-counts so both channels have real multi-bin structure) plus a per-level
    /// target whose sign/slope FLIPS depending on rarity, so a tree genuinely needs BOTH "which
    /// level, roughly" (mean) AND "is it rare" (count) to route correctly -- this is what
    /// makes the resulting joint axis genuinely non-separable (see the investigation numbers
    /// in the test below), not just structurally joint.
    ///
    /// Investigation history (bug #4): the first fixture draft used `i % 50` for the
    /// level-dependent term, whose period shares a factor with the row-count cycle's period
    /// of 15 (`LCM(50,15)=150`), making levels 150 apart exact duplicates and capping cells at
    /// ~150 regardless of level count -- switched to a term STRICTLY monotonic in `i`. The
    /// second draft used small (1..15) row-counts and a purely additive target, which the
    /// booster satisfied via the count channel ALONE in 300/300 trees (separable) -- larger,
    /// more stable row-counts (10..150) plus the sign-flipping interaction above were needed to
    /// get a genuinely non-separable fit.
    fn overflow_fixture() -> (Vec<f32>, Vec<String>, Vec<f32>) {
        let n_levels = 350;
        let mut cat_high = Vec::new();
        let mut y = Vec::new();
        for i in 0..n_levels {
            let count = 10 * (1 + (i % 15));
            for _ in 0..count {
                cat_high.push(format!("lvl{i}"));
                // Scaled down from an earlier draft (i*5.0 / 1000.0-i*2.0, reaching ~1700):
                // the codebase's own reconstruction tolerance (4*n_trees*f32::EPSILON) is an
                // ABSOLUTE bound, so it implicitly assumes a "normal" response scale -- at
                // ~1700 the accumulated f32 rounding across 300 trees exceeded it (a scale
                // problem with the fixture, not evidence against the fix).
                let y_val = if count <= 30 {
                    i as f32 * 0.05
                } else {
                    10.0 - (i as f32 * 0.02)
                };
                y.push(y_val);
            }
        }
        let n = cat_high.len();
        let numeric: Vec<f32> = vec![0.0; n];
        (numeric, cat_high, y)
    }

    /// A smaller multi-channel high-card fixture for the bug #6 coverage gap: every earlier
    /// multi-channel test in this module fit under `SquaredError` (Identity link), so NONE of
    /// them ever exercised graduation (poisson/gamma + `graduate=True` only) or prune-time
    /// reanchor (`do_reanchor = reanchor && Log|Logit` -- Identity never qualifies). `gamma`
    /// picks the target's sign convention: Poisson needs `y >= 0` (count-like), Gamma needs
    /// `y > 0` strictly.
    fn poisson_or_gamma_multichannel_fixture(gamma: bool) -> (Vec<f32>, Vec<String>, Vec<f32>) {
        let n_levels = 30;
        let mut cat_high = Vec::new();
        let mut y = Vec::new();
        for i in 0..n_levels {
            let count = 8 * (1 + (i % 5)); // varied per-level row counts, admits the count channel
            for _ in 0..count {
                cat_high.push(format!("lvl{i}"));
                let base = (i % 6) as f32; // a real, level-varying signal to fit and graduate
                y.push(if gamma { 1.0 + base } else { base });
            }
        }
        let n = cat_high.len();
        let numeric: Vec<f32> = (0..n).map(|r| (r % 3) as f32).collect();
        (numeric, cat_high, y)
    }

    /// Bug #4 (the switchover blocker): a joint categorical axis for a very-high-cardinality
    /// feature can enumerate more than 255 cells (bounded by distinct post-rare-pooling level
    /// count, per spec §4.3 -- NOT by the 254-bin-per-channel budget `assign_fisher_bins`
    /// enforces on EACH channel individually), which `TableModel`'s validation used to reject
    /// outright with "joint grid n_bins must be in 2..=255, got N".
    ///
    /// INVESTIGATION NUMBERS (measured on this exact fixture, confirming the numbers before
    /// the fix per the team lead's instruction):
    /// - (a) SEPARABILITY: NOT separable. 257 of 300 fitted trees split on BOTH the mean and
    ///   count channels of this one raw feature (43 split mean-only, 0 split count-only, 0
    ///   split neither) -- a genuine, substantial cross-channel interaction, not an
    ///   unnecessary joint axis. Option 1 (represent as separate per-channel tables) does not
    ///   apply here, and — since the count channel's whole design purpose is to be combined
    ///   with mean specifically for high-cardinality/rare levels (design/multichannel-
    ///   categoricals.md, Stage A) — is unlikely to apply to the general high-card+count-
    ///   channel case this bug actually occurs in.
    /// - (b) DISTINCT VALUES: 347 of 351 cells have bit-distinct relativities -- essentially
    ///   no coincidental merging available. Option 2 (merge cells with bit-identical values)
    ///   would only reduce 351 -> 347, nowhere near the <=255 target.
    /// - FIX USED: option 3, WIDEN. Traced every consumer of a joint axis's cell count
    ///   (`x_cells: Vec<u32>`, `JointCatAxis::n_cells: u32`, `AxisId::cells: u32`,
    ///   `BorderGrid::n_bins: u16`) and found NONE of them actually require `<= 255` for the
    ///   joint case — that bound was only ever enforced by one validation check
    ///   (`table_model.rs`'s `validate_merged_grid`), which had copied the SINGLE-AXIS bound
    ///   (genuinely load-bearing there: a real axis's bin is `u8`-stored per row) into the
    ///   joint-axis branch, where no such storage constraint exists (a joint cell id is never
    ///   stored per-row; it's computed on demand via `JointCatAxis::cell_for_channel_bins` and
    ///   carried in the raw-indexed `x_cells: Vec<u32>`). Relaxed that one check to the
    ///   joint-axis's own natural field-width limit (`BorderGrid::n_bins: u16`, already
    ///   guarded by a clean `u16::try_from` at construction) — no other code changes needed.
    #[test]
    fn joint_axis_overflow_is_fixed_losslessly() {
        let (numeric, cat_high, y) = overflow_fixture();

        let mut state = booster_with_count_channel_and_floor(20, 1.0);
        state.config.n_trees = 300;
        let monotone_map = build_monotone_map(None, 1).unwrap();
        let model = fit_model_ambient(
            &state,
            std::slice::from_ref(&numeric),
            &y,
            None,
            None,
            None,
            None,
            &monotone_map,
            Some(std::slice::from_ref(&cat_high)),
            None,
        )
        .unwrap();
        assert_eq!(model.provenance.len(), 3, "1 numeric + mean + count axes");

        // (a) Separability check, kept as a live assertion (not just a one-time measurement):
        // this fixture must keep exercising a genuine cross-channel interaction, or the "not
        // separable" investigation finding above would go stale.
        let mean_axis = model
            .provenance
            .iter()
            .position(|p| {
                p.raw.0 == 1
                    && matches!(p.kind, AxisKind::CategoricalTS { encoding } if encoding.0 == 0)
            })
            .expect("mean channel axis must exist");
        let count_axis = model
            .provenance
            .iter()
            .position(|p| {
                p.raw.0 == 1
                    && matches!(p.kind, AxisKind::CategoricalTS { encoding } if encoding.0 == 1)
            })
            .expect("count channel axis must exist");
        let both_channel_trees = model
            .trees
            .iter()
            .filter(|(_, tree)| {
                let has_mean = tree.splits.iter().any(|s| s.axis as usize == mean_axis);
                let has_count = tree.splits.iter().any(|s| s.axis as usize == count_axis);
                has_mean && has_count
            })
            .count();
        assert!(
            both_channel_trees > 0,
            "fixture must exercise a genuine cross-channel interaction (found 0 trees using \
             both channels) -- otherwise this test would only be proving the SEPARABLE case"
        );

        let binned = serve_binned_for_model(&model, &[numeric], Some(&[cat_high])).unwrap();
        let serve = ServeBinnedMatrix(binned);
        let bank = model.explain(&serve, RefMeasure::Uniform).unwrap();
        let table = bank
            .tables
            .iter()
            .find(|t| t.u.0.iter().any(|f| f.0 == 1))
            .expect("raw feature 1 (the categorical) must have a realized table");
        let cells = table.axes.first().map(|a| a.cells).unwrap();
        assert!(
            cells > 255,
            "fixture must actually exercise the overflow (got {cells} cells, need > 255)"
        );

        // Hard gate 1: all five I2 exactness gates, on the overflowing axis itself.
        t_boost_core::explain::assert_exact_decomposition(&model, &bank, &serve).unwrap();

        // Hard gate 2 (the actual reported crash site): TableModel construction + validation
        // must now succeed where it used to reject with "n_bins must be in 2..=255".
        let tm = TableModel::from_model(&model, &serve, RefMeasure::Uniform).unwrap();
        tm.validate().unwrap();

        // Hard gate 3: the losslessness proof -- an UNPRUNED tables-only model must predict
        // identically to the tree ensemble on the overflowing feature. Tolerance is the
        // codebase's own derived reconstruction tolerance (`4 * n_trees * f32::EPSILON`, spec
        // §13.1), not an arbitrary constant -- a hardcoded `1e-6` is too tight for a 300-tree
        // ensemble's accumulated f32 rounding and would fail even on an exact reconstruction.
        let tol = t_boost_core::explain::ExactTol::for_model(&model).recon_tol;
        let ensemble_preds = model.predict(&serve.0, None).unwrap();
        let tables_preds = tm.predict_binned(&serve.0, None).unwrap();
        assert_eq!(ensemble_preds.len(), tables_preds.len());
        for (e, t) in ensemble_preds.iter().zip(&tables_preds) {
            assert!(
                (f64::from(*e) - f64::from(*t)).abs() < tol,
                "unpruned tables-only predict must reproduce the ensemble: {e} vs {t} (tol={tol})"
            );
        }

        // Hard gate 4: the actual CV-pruning path (`fit_prune_selection`'s underlying
        // function) must also succeed on this overflowing feature, end to end. Rebuilds the
        // fixture fresh (a new, independent `(numeric, cat_high, y)`) since the ones above
        // were consumed/cloned into the earlier calls.
        let (numeric2, cat_high2, y2) = overflow_fixture();
        let n2 = y2.len();
        let monotone_map2 = build_monotone_map(None, 1).unwrap();
        let fold_of: Vec<i64> = (0..n2).map(|i| (i % 3) as i64).collect();
        let reports = fit_prune_reports_owned(
            state,
            vec![numeric2],
            y2,
            fold_of,
            3,
            None,
            None,
            None,
            None,
            monotone_map2,
            Some(vec![cat_high2]),
            RefMeasure::Uniform,
            0.0,
            0.0,
            0.0,
            DEFAULT_TABLE_MIN_ARITY,
            1,
            false,
            None,
            // pinned-bank fold fidelity off: these gates pin the HISTORICAL selection path.
            None,
        )
        .expect(
            "fit_prune_selection's underlying CV-pruning path must succeed on the >255-cell \
             feature",
        );
        assert!(reports.iter().any(Option::is_some));
    }

    /// Bug #5, and the strategy change that followed it: rather than fixing single crashes
    /// one at a time, this drives the WHOLE deploy lifecycle on one multi-channel high-card
    /// (>255 would-be joint cells, non-separable) fixture, so no un-migrated path can survive
    /// silently: fit -> CV prune-select -> apply_keepset (deploy, `rebalance: true` -- the
    /// sklearn.py DEFAULT, `prune_rebalance=True`) -> predict -> tables -> serialize save ->
    /// load -> predict-after-load -> shap.
    ///
    /// `rebalance: true` here (previously `false`, see the git history for that draft) now
    /// exercises `rebalance_kept_cells`'s multi-channel SKIP (bug #5's follow-up fix): a
    /// separate subsystem (`cell_refit::fit_cell_correction` -> `correction_scaffold`) has its
    /// own independent row-to-cell representation with no notion of a joint axis, plus an
    /// unrelated raw-id-vs-axis-id conflation bug only latent pre-P1 -- rather than growing
    /// joint support through that whole ridge-regression pipeline (out of scope, a redesign,
    /// per the team lead), `rebalance_kept_cells` now detects any multi-channel raw feature
    /// ANYWHERE in the model and skips rebalance for the entire model, falling back to the
    /// already-proven-lossless unrebalanced deploy — which is exactly what this test's
    /// `rebalance: true` request now silently (and correctly) becomes.
    #[test]
    fn full_deploy_lifecycle_survives_a_nonseparable_overflowing_joint_axis() {
        let (numeric, cat_high, y) = overflow_fixture();

        let mut state = booster_with_count_channel_and_floor(20, 1.0);
        state.config.n_trees = 300;
        let monotone_map = build_monotone_map(None, 1).unwrap();
        let feature_names = Some(vec!["num_a".to_string(), "veh_model".to_string()]);
        let model = fit_model_ambient(
            &state,
            std::slice::from_ref(&numeric),
            &y,
            None,
            None,
            feature_names,
            None,
            &monotone_map,
            Some(std::slice::from_ref(&cat_high)),
            None,
        )
        .unwrap();
        assert_eq!(model.provenance.len(), 3);

        // Keep the fixture honest: overflow + non-separability, exactly like bug #4.
        let mean_axis = model
            .provenance
            .iter()
            .position(|p| {
                p.raw.0 == 1
                    && matches!(p.kind, AxisKind::CategoricalTS { encoding } if encoding.0 == 0)
            })
            .unwrap();
        let count_axis = model
            .provenance
            .iter()
            .position(|p| {
                p.raw.0 == 1
                    && matches!(p.kind, AxisKind::CategoricalTS { encoding } if encoding.0 == 1)
            })
            .unwrap();
        let both_channel_trees = model
            .trees
            .iter()
            .filter(|(_, tree)| {
                let has_mean = tree.splits.iter().any(|s| s.axis as usize == mean_axis);
                let has_count = tree.splits.iter().any(|s| s.axis as usize == count_axis);
                has_mean && has_count
            })
            .count();
        assert!(both_channel_trees > 0, "fixture must stay non-separable");

        // STEP 1: CV prune-select (`fit_prune_selection`'s underlying function).
        let (numeric2, cat_high2, y2) = overflow_fixture();
        let n2 = y2.len();
        let monotone_map2 = build_monotone_map(None, 1).unwrap();
        let fold_of: Vec<i64> = (0..n2).map(|i| (i % 3) as i64).collect();
        let reports = fit_prune_reports_owned(
            state.clone(),
            vec![numeric2],
            y2,
            fold_of,
            3,
            None,
            None,
            None,
            None,
            monotone_map2,
            Some(vec![cat_high2]),
            RefMeasure::Uniform,
            0.0,
            0.0,
            0.0,
            DEFAULT_TABLE_MIN_ARITY,
            1,
            false,
            None,
            // pinned-bank fold fidelity off: these gates pin the HISTORICAL selection path.
            None,
        )
        .expect("CV prune-select must succeed on the overflowing feature");
        assert!(reports.iter().any(Option::is_some));

        // STEP 2: apply_keepset (deploy) -- keep every realized support, `rebalance: false`
        // (see the KNOWN GAP doc above).
        let binned = serve_binned_for_model(
            &model,
            std::slice::from_ref(&numeric),
            Some(std::slice::from_ref(&cat_high)),
        )
        .unwrap();
        let serve = ServeBinnedMatrix(binned);
        let bank = model.explain(&serve, RefMeasure::Uniform).unwrap();
        let cells = bank
            .tables
            .iter()
            .find(|t| t.u.0.iter().any(|f| f.0 == 1))
            .and_then(|t| t.axes.first())
            .map(|a| a.cells)
            .unwrap();
        assert!(
            cells > 255,
            "fixture must actually overflow (got {cells} cells)"
        );
        t_boost_core::explain::assert_exact_decomposition(&model, &bank, &serve).unwrap();

        let keep: Vec<FeatureSet> = bank.tables.iter().map(|t| t.u.clone()).collect();
        let w = vec![1.0_f32; n2];
        let tm = t_boost_core::prune::prune_model_to_keepset(
            &model,
            &serve,
            &y,
            &w,
            None,
            &SquaredError,
            RefMeasure::Uniform,
            false, // reanchor
            &keep,
            true, // rebalance -- exercises the multi-channel skip end to end (bug #5 follow-up)
        )
        .expect("apply_keepset (prune_model_to_keepset) must succeed on the overflowing feature");
        tm.validate().unwrap();

        // STEP 3: predict, losslessly.
        let tol = t_boost_core::explain::ExactTol::for_model(&model).recon_tol;
        let ensemble_preds = model.predict(&serve.0, None).unwrap();
        let tm_preds_before = tm.predict_binned(&serve.0, None).unwrap();
        assert_eq!(ensemble_preds.len(), tm_preds_before.len());
        for (e, t) in ensemble_preds.iter().zip(&tm_preds_before) {
            assert!((f64::from(*e) - f64::from(*t)).abs() < tol);
        }

        // STEP 4: tables() -- level labels present, one table per raw feature, no #count leak.
        let export = tm
            .bank
            .to_rating_export(
                tm.link,
                &tm.mode,
                &tm.schema,
                &tm.provenance,
                &tm.schema.cat_encoders,
                None,
            )
            .unwrap();
        let cat_tables: Vec<_> = export
            .tables
            .iter()
            .filter(|t| t.feature_set.0.iter().any(|f| f.0 == 1))
            .collect();
        assert_eq!(
            cat_tables.len(),
            1,
            "one table for the categorical, not one per channel"
        );
        assert_eq!(cat_tables[0].axes.len(), 1, "one axis, not one per channel");
        let levels = cat_tables[0].axes[0]
            .levels
            .as_ref()
            .expect("categorical axis must export level labels");
        assert!(
            levels.len() > 255,
            "level labels must be present for every one of the >255 cells"
        );
        for name in &cat_tables[0].feature_names {
            assert!(
                !name.contains("#count"),
                "no synthetic count-channel name may leak: {name}"
            );
        }
        for axis in &cat_tables[0].axes {
            assert!(!axis.name.contains("#count"));
        }

        // STEP 5: serialize save -> load -> predict-after-load EXACTLY equals predict-before.
        let bytes = t_boost_core::serialize::encode_tables(&tm).unwrap();
        let tm_loaded = t_boost_core::serialize::decode_tables(&bytes).unwrap();
        let tm_preds_after = tm_loaded.predict_binned(&serve.0, None).unwrap();
        assert_eq!(
            tm_preds_before, tm_preds_after,
            "predict after a save/load round-trip must be EXACTLY (bit-for-bit) identical"
        );
        // Same check through the JSON round-trip.
        let json = t_boost_core::serialize::encode_tables_json(&tm).unwrap();
        let tm_loaded_json = t_boost_core::serialize::decode_tables_json(&json).unwrap();
        let tm_preds_after_json = tm_loaded_json.predict_binned(&serve.0, None).unwrap();
        assert_eq!(tm_preds_before, tm_preds_after_json);

        // STEP 6: shap (not Python-exposed, but part of the core API -- confirm it works on
        // the overflowing joint axis directly).
        let raw_feature_count = t_boost_core::data::n_raw_features(&model.provenance);
        for cell in [0u32, 1, cells - 1] {
            let mut x_cells = vec![0u32; raw_feature_count];
            if let Some(slot) = x_cells.get_mut(1) {
                *slot = cell;
            }
            let phi = tm.bank.shap(&x_cells).unwrap();
            assert_eq!(phi.len(), raw_feature_count);
            assert!(phi.iter().all(|v| v.is_finite()));
        }
    }

    /// Proves the multi-channel skip is correctly SCOPED: a single-channel (no `cat_channels`
    /// at all) model with `rebalance: true` must still actually rebalance -- no regression to
    /// the normal, pre-bug-#5 path. Fits a genuine `x0 * x1` interaction, then prunes down to
    /// ONLY the two main-effect (order-1) tables, dropping the pair entirely -- this leaves a
    /// clear, consistent residual (the discarded interaction's own marginal projection) that a
    /// per-cell ridge correction can partially recover, so `rebalance: true`'s no-harm guard
    /// should adopt the correction and the deployed table VALUES must differ from the
    /// `rebalance: false` bank.
    #[test]
    fn single_channel_rebalance_is_unaffected_by_the_multichannel_skip() {
        let n = 200;
        let x0: Vec<f32> = (0..n).map(|i| (i % 10) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| ((i / 10) % 10) as f32).collect();
        let y: Vec<f32> = x0.iter().zip(&x1).map(|(&a, &b)| a * b).collect();

        let state = PyBooster {
            // Deliberately UNDERFIT (few trees): an underfit model's bias is systematic (the
            // same shortfall pattern everywhere), not noise-shaped, so a ridge correction
            // learned on the non-held-out rows should generalize to the held-out slice too --
            // a fully-converged fit's residual (tried first, at n_trees=200) was too small/
            // sample-specific for the no-harm guard to adopt on held-out data.
            config: Config {
                n_trees: 8,
                learning_rate: 0.3,
                ..Config::default()
            },
            bin_config: BinConfig::default(),
            objective: Objective::SquaredError,
            credibility: CredibilityFloor::default(),
            interaction: InteractionPolicy {
                max_order: 2,
                ..InteractionPolicy::default()
            },
            cat_config: TsConfig::default(),
            cat_count_config: None,
            cat_count_min_levels: 20,
            cat_class_freq_min_levels: 3,
            cat_channels: CatChannelPlan::default(),
            cat_class_freq_config: None,
            seed: 0,
            n_jobs: None,
            fit_pool_width: None,
        };
        let monotone_map = build_monotone_map(None, 2).unwrap();
        let model = fit_model_ambient(
            &state,
            &[x0.clone(), x1.clone()],
            &y,
            None,
            None,
            None,
            None,
            &monotone_map,
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            model.provenance.len(),
            2,
            "no cat_channels -> no multi-channel axes at all"
        );

        let binned = bin_columns(&[&x0, &x1], None, &state.bin_config, state.seed).unwrap();
        let serve = ServeBinnedMatrix(binned);
        let bank = model.explain(&serve, RefMeasure::Uniform).unwrap();
        assert!(
            bank.tables.iter().any(|t| t.u.order() == 2),
            "fixture must realize a genuine pair effect to prune away"
        );
        // Keep ONLY the two order-1 main effects -- drop the pair entirely.
        let keep: Vec<FeatureSet> = bank
            .tables
            .iter()
            .map(|t| t.u.clone())
            .filter(|u| u.order() == 1)
            .collect();
        assert_eq!(keep.len(), 2);

        let w = vec![1.0_f32; n as usize];
        let tm_off = t_boost_core::prune::prune_model_to_keepset(
            &model,
            &serve,
            &y,
            &w,
            None,
            &SquaredError,
            RefMeasure::Uniform,
            false,
            &keep,
            false, // rebalance off -- the baseline
        )
        .unwrap();
        let tm_on = t_boost_core::prune::prune_model_to_keepset(
            &model,
            &serve,
            &y,
            &w,
            None,
            &SquaredError,
            RefMeasure::Uniform,
            false,
            &keep,
            true, // rebalance on -- must actually run for a single-channel model
        )
        .unwrap();

        let values_off: Vec<f64> = tm_off
            .bank
            .tables
            .iter()
            .flat_map(|t| t.values.try_values().unwrap().into_owned())
            .collect();
        let values_on: Vec<f64> = tm_on
            .bank
            .tables
            .iter()
            .flat_map(|t| t.values.try_values().unwrap().into_owned())
            .collect();
        assert_ne!(
            values_off, values_on,
            "rebalance:true must actually change the deployed table values for a \
             single-channel model (dropping a genuine interaction leaves recoverable residual \
             structure) -- if these are equal, rebalance silently didn't run, which would be a \
             regression from the multi-channel skip, not a correct no-harm rejection"
        );
    }

    /// THE regression test for the green-spine bug the skip closes: a multi-channel raw
    /// feature (raw 0, axes 0-1, mean+count admitted) FOLLOWED by another raw feature (raw 1,
    /// axis 2 -- a single-channel categorical, deliberately kept low-cardinality so it is NOT
    /// admitted for a count channel of its own). `fit_model_ambient` numbers numeric columns
    /// before categorical ones, so with NO numeric columns here, `cat_x`'s ORDER alone
    /// controls which categorical is raw 0 vs raw 1 -- `cat_high` (admitted) is passed first,
    /// giving it axes 0-1, so `cat_low` (raw 1, not admitted) ends up at axis 2, strictly
    /// AFTER the multi-channel feature -- exactly the green-spine failure mode. If
    /// `correction_scaffold`'s raw-id-as-axis-id bug ever ran, it would resolve raw 1's
    /// supposed "axis 1" as raw 0's OWN second (count) channel instead of raw 1's real axis
    /// (2) -- silently building the correction against the wrong column entirely, which
    /// cannot possibly reconstruct the ensemble's actual prediction. `rebalance: true` here
    /// must be silently skipped (this model has a multi-channel raw feature, so the
    /// WHOLE-MODEL skip applies) -- if it were not skipped, or skipped only by checking raw 0
    /// itself rather than the whole model, this test would fail on the buggy path.
    #[test]
    fn rebalance_skip_prevents_the_green_spine_misscore_for_a_feature_after_multichannel() {
        let n = 80;
        let cat_high: Vec<String> = (0..n).map(|i| format!("h{}", i % 8)).collect();
        // 3 distinct levels -- below the admission threshold (5) below, so this stays
        // single-channel, at whatever axis it ends up positioned at.
        let cat_low: Vec<String> = (0..n).map(|i| format!("l{}", i % 3)).collect();
        // y depends on BOTH raw features (so both survive pruning as their own order-1 kept
        // support) but additively, so this is purely about the axis-id bug, not about an
        // interaction rebalance might legitimately need to recover.
        const HIGH_TARGET: [f32; 8] = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0];
        const LOW_TARGET: [f32; 3] = [0.0, 10.0, 20.0];
        let y: Vec<f32> = (0..n)
            .map(|i| HIGH_TARGET[i % 8] + LOW_TARGET[i % 3])
            .collect();

        let state = booster_with_count_channel_and_floor(5, 1.0);
        let monotone_map = build_monotone_map(None, 2).unwrap();
        let model = fit_model_ambient(
            &state,
            &[],
            &y,
            None,
            None,
            None,
            None,
            &monotone_map,
            Some(&[cat_high.clone(), cat_low.clone()]),
            None,
        )
        .unwrap();
        // raw 0 = cat_high, mean+count (axes 0-1); raw 1 = cat_low, single-channel (axis 2) --
        // confirm the actual layout matches what this test needs (a multi-channel raw feature
        // strictly followed by another one) via the exact same check the fix's own skip uses.
        assert!(
            model.provenance.len() > t_boost_core::data::n_raw_features(&model.provenance),
            "fixture must have a genuine multi-channel raw feature"
        );
        assert_eq!(
            model.provenance.len(),
            3,
            "cat_high: 2 axes, cat_low: 1 axis"
        );

        let binned = serve_binned_for_model(&model, &[], Some(&[cat_high, cat_low])).unwrap();
        let serve = ServeBinnedMatrix(binned);
        let bank = model.explain(&serve, RefMeasure::Uniform).unwrap();
        let keep: Vec<FeatureSet> = bank.tables.iter().map(|t| t.u.clone()).collect();
        assert_eq!(
            keep.len(),
            2,
            "both raw features must survive as their own order-1 effect"
        );
        let w = vec![1.0_f32; n];
        let tm = t_boost_core::prune::prune_model_to_keepset(
            &model,
            &serve,
            &y,
            &w,
            None,
            &SquaredError,
            RefMeasure::Uniform,
            false,
            &keep,
            true, // rebalance requested -- must be silently skipped, not silently wrong
        )
        .expect("must not crash");

        let tol = t_boost_core::explain::ExactTol::for_model(&model).recon_tol;
        let ensemble_preds = model.predict(&serve.0, None).unwrap();
        let tm_preds = tm.predict_binned(&serve.0, None).unwrap();
        for (e, t) in ensemble_preds.iter().zip(&tm_preds) {
            assert!(
                (f64::from(*e) - f64::from(*t)).abs() < tol,
                "deployed predict must equal the ensemble ({e} vs {t}) -- it CANNOT if the \
                 raw-id-as-axis-id bug ran and built a correction against the wrong axis"
            );
        }
    }

    /// The PURELY SILENT variant of the test above, and the reason the fix checks the WHOLE
    /// model rather than just `keep`'s own contents. With only 2 raw features (the test
    /// above), the misinterpreted axis id always lands back on some part of the SAME
    /// multi-channel feature (still joint, so the guard still fires loudly) -- confirmed by
    /// temporarily disabling the fix and re-running that test with `keep` restricted to raw 1
    /// alone: it STILL crashed via the guard, not a silent wrong number, because raw 1's id
    /// (1) resolves to axis 1, which belongs to raw 0 (the multi-channel feature) either way.
    /// A TRULY silent misscore (no crash, just a wrong number) needs the misinterpreted axis
    /// to land on a DIFFERENT single-channel raw feature -- which needs a multi-channel
    /// feature followed by at least TWO more single-channel features, only the LATER of which
    /// is kept: raw 0 (numeric), raw 1 (multi-channel, axes 1-2), raw 2 (single-channel, axis
    /// 3), raw 3 (single-channel, axis 4) -- with ONLY raw 3 kept (raw 1 never appears in
    /// `supports` at all), raw 3's id (3) misinterpreted as axis 3 resolves to raw 2 --
    /// single-channel, so the guard never fires.
    ///
    /// ORACLE, take 2: an earlier draft compared this `rebalance:true` bank directly against a
    /// `rebalance:false` run on the same keep-set, expecting `assert_eq!`. That does NOT
    /// reliably discriminate on this fixture: re-running it with the fix disabled and
    /// `TBOOST_PROFILE=1` shows the misrouted correction (built against raw 2's REAL cells --
    /// axis 3 really is raw 2's column in `serve`, so the ridge fit itself is internally
    /// consistent, just aimed at the wrong raw feature) gets REJECTED by the no-harm guard on
    /// its own merits: `base_dev=20407.55078 corr_dev=20415.32617 -> fall back`. So the two
    /// banks come back identical EITHER WAY here -- the guard's independent data-dependent
    /// adopt/reject decision can mask whether the whole-model skip fired at all, which makes a
    /// plain value comparison meaningless (a prior version of this comment claimed the values
    /// came back different under the same conditions; that was not reproducible and has been
    /// corrected after re-verifying directly).
    ///
    /// So instead: `y`/`w` are passed ONE ROW LONGER than `serve`'s row count. With
    /// `reanchor=false`, `prune_model_to_keepset` never reads `y`/`w` at all except inside
    /// `rebalance_kept_cells` (reached only when `rebalance=true`) -- so if the whole-model
    /// skip fires (returning `base` before `y`/`w` are ever touched), the mismatched length is
    /// irrelevant and this succeeds. If the skip is missing, execution reaches
    /// `loss.grad_hess(y, ..)`, whose OWN shape check (`y.len() != raw.len()`) turns that
    /// reintroduction into an immediate, unconditional `ShapeMismatch` error: a hard failure
    /// the no-harm guard's adoption dynamics cannot mask, unlike a plain value comparison.
    #[test]
    fn rebalance_skip_prevents_the_purely_silent_variant_two_features_after_multichannel() {
        let n = 288;
        let numeric: Vec<f32> = (0..n).map(|i| (i % 4) as f32).collect();
        let cat1: Vec<String> = (0..n).map(|i| format!("h{}", i % 8)).collect();
        // cat2 cycles on `i % 3`, cat3 on `(i / 3) % 3` -- independent periods (not both `i %
        // 3`, which would make them perfectly collinear and let the booster capture the
        // whole target via cat2 alone, never touching cat3 at all -- the first draft's bug,
        // confirmed empirically: raw 3 never appeared as a realized table).
        let cat2: Vec<String> = (0..n).map(|i| format!("l{}", i % 3)).collect();
        let cat3: Vec<String> = (0..n).map(|i| format!("m{}", (i / 3) % 3)).collect();
        const CAT2_TARGET: [f32; 3] = [0.0, 1.0, 2.0];
        const CAT3_TARGET: [f32; 3] = [0.0, 100.0, 200.0]; // scaled apart from cat2's own
        let y: Vec<f32> = (0..n)
            .map(|i| CAT2_TARGET[i % 3] + CAT3_TARGET[(i / 3) % 3])
            .collect();

        let state = booster_with_count_channel_and_floor(5, 1.0);
        let monotone_map = build_monotone_map(None, 4).unwrap();
        let model = fit_model_ambient(
            &state,
            std::slice::from_ref(&numeric),
            &y,
            None,
            None,
            None,
            None,
            &monotone_map,
            Some(&[cat1.clone(), cat2.clone(), cat3.clone()]),
            None,
        )
        .unwrap();
        // raw 0 = numeric (axis 0); raw 1 = cat1, mean+count (axes 1-2); raw 2 = cat2, single
        // (axis 3); raw 3 = cat3, single (axis 4).
        assert_eq!(model.provenance.len(), 5);
        assert!(model.provenance.len() > t_boost_core::data::n_raw_features(&model.provenance));

        let binned = serve_binned_for_model(
            &model,
            std::slice::from_ref(&numeric),
            Some(&[cat1, cat2, cat3]),
        )
        .unwrap();
        let serve = ServeBinnedMatrix(binned);
        let bank = model.explain(&serve, RefMeasure::Uniform).unwrap();
        // Keep ONLY raw 3's own order-1 table -- raw 0, raw 1, and raw 2 are all pruned away,
        // so raw 1 (the actual multi-channel feature) never appears in `supports` at all; if
        // the fix checked only `keep`'s own contents for multi-channel-ness, this case would
        // slip through the skip entirely.
        let keep: Vec<FeatureSet> = bank
            .tables
            .iter()
            .map(|t| t.u.clone())
            .filter(|u| u.order() == 1 && u.0.first().map(|f| f.0) == Some(3))
            .collect();
        assert_eq!(
            keep.len(),
            1,
            "raw 3 must have survived as its own order-1 effect"
        );

        // Tripwire: y/w one row longer than serve's row count -- see the doc comment above.
        let bad_y = vec![0.0_f32; n + 1];
        let bad_w = vec![1.0_f32; n + 1];
        t_boost_core::prune::prune_model_to_keepset(
            &model,
            &serve,
            &bad_y,
            &bad_w,
            None,
            &SquaredError,
            RefMeasure::Uniform,
            false,
            &keep,
            true, // rebalance requested -- must be skipped (whole-model, not per-feature)
                  // before bad_y/bad_w are ever read, since raw 1 alone isn't even in `keep`
        )
        .expect(
            "multi-channel model: the whole-model skip must return before rebalance_kept_cells \
             ever reads the mismatched-length y/w via grad_hess",
        );
    }

    /// Shared body for the bug #6 test-coverage-gap fix: a poisson/gamma multi-channel
    /// high-card fixture driving prune-time reanchor (Log-link only -- NEW coverage for a
    /// joint axis, since every earlier multi-channel test in this module fit under
    /// `SquaredError`/Identity, where `do_reanchor` never engages) and `apply_graduation`
    /// (poisson/gamma + `graduate=True` only -- bug #6's own crash site). The update handed
    /// to `apply_graduation_updates` is a hand-perturbation of the joint categorical's OWN
    /// table (not an empty no-op), so `purify_raw_effects` genuinely reconstructs a joint
    /// axis, exactly the path that crashed with "bank merged grid n_bins .. inconsistent with
    /// .. borders" before the fix.
    fn graduation_survives_a_multichannel_high_card_fixture(gamma: bool) {
        let (numeric, cat_high, y) = poisson_or_gamma_multichannel_fixture(gamma);
        let n = y.len();
        let w = vec![1.0_f32; n];

        let mut state = booster_with_count_channel(5);
        state.objective = if gamma {
            Objective::Gamma
        } else {
            Objective::Poisson
        };
        state.config.n_trees = 40;
        // Fit-time reanchor (a plain intercept fold, `f0 += reanchor_delta(...)`, entirely
        // independent of prune-time reanchor): cheap to enable here too, for coverage alongside
        // the prune-time one below. `reanchor_slope` (affine tree-alpha rescaling) is NOT
        // exercised here -- it needs a validation holdout to engage at all, and both mechanisms
        // operate purely on `trees`/`f0` (verified by reading engine/boost.rs:1082-1128: neither
        // ever touches a grid/axis/joint-categorical representation), so they cannot be affected
        // by the axis-representation bug class bugs #5/#6 both were -- this line is belt and
        // braces, not closing a suspected gap.
        state.config.boosters.reanchor = true;
        let monotone_map = build_monotone_map(None, 1).unwrap();
        let model = fit_model_ambient(
            &state,
            std::slice::from_ref(&numeric),
            &y,
            None,
            None,
            None,
            None,
            &monotone_map,
            Some(std::slice::from_ref(&cat_high)),
            None,
        )
        .unwrap();
        assert_eq!(
            model.provenance.len(),
            3,
            "numeric + cat mean/count channels"
        );
        assert!(
            model.provenance.len() > t_boost_core::data::n_raw_features(&model.provenance),
            "fixture must actually be multi-channel"
        );
        assert_eq!(
            model.link,
            t_boost_core::loss::Link::Log,
            "poisson/gamma must deploy on the log link -- the only link prune-time reanchor \
             engages on (do_reanchor = reanchor && Log|Logit)"
        );

        let binned = serve_binned_for_model(
            &model,
            std::slice::from_ref(&numeric),
            Some(std::slice::from_ref(&cat_high)),
        )
        .unwrap();
        let serve = ServeBinnedMatrix(binned);
        let bank = model.explain(&serve, RefMeasure::Uniform).unwrap();
        let keep: Vec<FeatureSet> = bank.tables.iter().map(|t| t.u.clone()).collect();

        let table_model: t_boost_core::table_model::TableModel = if gamma {
            t_boost_core::prune::prune_model_to_keepset(
                &model,
                &serve,
                &y,
                &w,
                None,
                &Gamma,
                RefMeasure::Uniform,
                true, // reanchor -- prune-time, Log-link-eligible: new coverage on a joint axis
                &keep,
                true, // rebalance -- already covered by bug #5; included for end-to-end realism
            )
        } else {
            t_boost_core::prune::prune_model_to_keepset(
                &model,
                &serve,
                &y,
                &w,
                None,
                &Poisson,
                RefMeasure::Uniform,
                true,
                &keep,
                true,
            )
        }
        .expect("prune-time reanchor+rebalance must not crash on a joint axis under the log link");

        // Hand-perturb the joint categorical's own main-effect table -- exercises the EXACT
        // purify_raw_effects reconstruction bug #6 fixed, not an empty no-op.
        let cat_u = FeatureSet::new(&[1]);
        let (table_idx, joint_table) = table_model
            .bank
            .tables
            .iter()
            .enumerate()
            .find(|(_, t)| t.u == cat_u)
            .expect("the joint categorical's own main-effect table must survive prune+keepset");
        let mut values = joint_table.values.try_values().unwrap().into_owned();
        for v in &mut values {
            *v += 0.01;
        }
        let updates = vec![(table_idx, values)];

        let columns = vec![numeric.clone()];
        let cat_x = Some(vec![cat_high.clone()]);
        let (graduated, _adopted) = apply_graduation_updates(
            &table_model,
            columns,
            y.clone(),
            w.clone(),
            None,
            cat_x,
            updates,
            None,
        )
        .expect("apply_graduation_updates must not crash on a joint categorical axis (bug #6)");

        // Whichever model resulted (graduated, or the safe no-harm-guard fallback), predict
        // must stay finite -- `_adopted`'s own value is the no-harm guard's business, not
        // this test's; either outcome is a valid, lossless-or-rejected result.
        let preds = graduated.predict_binned(&serve.0, None).unwrap();
        assert!(
            preds.iter().all(|p| p.is_finite()),
            "post-graduation predict must stay finite on a joint categorical axis"
        );
    }

    #[test]
    fn graduation_survives_a_poisson_multichannel_high_card_fixture() {
        graduation_survives_a_multichannel_high_card_fixture(false);
    }

    #[test]
    fn graduation_survives_a_gamma_multichannel_high_card_fixture() {
        graduation_survives_a_multichannel_high_card_fixture(true);
    }

    /// Prune-time reanchor's OTHER gated link (`do_reanchor = reanchor && (Log | Logit)`):
    /// the poisson/gamma tests above cover Log; this covers Logit, not yet exercised on a
    /// multi-channel model by any test in this module. Asserts the balance guarantee
    /// `prune_model_to_keepset`'s own doc comment promises for either link: the reanchored
    /// model's weighted mean prediction matches the observed weighted mean of y.
    #[test]
    fn prune_time_reanchor_balances_a_logit_link_multichannel_model() {
        let n_levels = 20;
        let mut cat_high = Vec::new();
        let mut y = Vec::new();
        for i in 0..n_levels {
            let count = 8 * (1 + (i % 4));
            for r in 0..count {
                cat_high.push(format!("lvl{i}"));
                y.push(if (i + r) % 3 == 0 { 1.0_f32 } else { 0.0_f32 });
            }
        }
        let n = cat_high.len();
        let numeric: Vec<f32> = (0..n).map(|r| (r % 3) as f32).collect();
        let w = vec![1.0_f32; n];

        let mut state = booster_with_count_channel(5);
        state.objective = Objective::Logistic;
        state.config.n_trees = 40;
        let monotone_map = build_monotone_map(None, 1).unwrap();
        let model = fit_model_ambient(
            &state,
            std::slice::from_ref(&numeric),
            &y,
            None,
            None,
            None,
            None,
            &monotone_map,
            Some(std::slice::from_ref(&cat_high)),
            None,
        )
        .unwrap();
        assert_eq!(model.provenance.len(), 3);
        assert!(
            model.provenance.len() > t_boost_core::data::n_raw_features(&model.provenance),
            "fixture must actually be multi-channel"
        );
        assert_eq!(model.link, t_boost_core::loss::Link::Logit);

        let binned = serve_binned_for_model(
            &model,
            std::slice::from_ref(&numeric),
            Some(std::slice::from_ref(&cat_high)),
        )
        .unwrap();
        let serve = ServeBinnedMatrix(binned);
        let bank = model.explain(&serve, RefMeasure::Uniform).unwrap();
        let keep: Vec<FeatureSet> = bank.tables.iter().map(|t| t.u.clone()).collect();

        let table_model = t_boost_core::prune::prune_model_to_keepset(
            &model,
            &serve,
            &y,
            &w,
            None,
            &Logistic,
            RefMeasure::Uniform,
            true, // reanchor -- logit-link-eligible
            &keep,
            false, // rebalance -- already covered by bug #5's tests; keep this test focused
        )
        .expect("logit-link prune-time reanchor must not crash on a joint axis");

        let preds = table_model.predict_binned(&serve.0, None).unwrap();
        let sum_w: f64 = w.iter().map(|&wi| f64::from(wi)).sum();
        let observed: f64 = y
            .iter()
            .zip(&w)
            .map(|(&yi, &wi)| f64::from(yi) * f64::from(wi))
            .sum();
        let predicted: f64 = preds
            .iter()
            .zip(&w)
            .map(|(&pi, &wi)| f64::from(pi) * f64::from(wi))
            .sum();
        assert!(
            (observed / sum_w - predicted / sum_w).abs() < 1e-4,
            "logit-link reanchor on a joint axis must balance: observed={}, predicted={}",
            observed / sum_w,
            predicted / sum_w,
        );
    }

    /// Bug #7 repro: a GENUINE over-budget order-3 interaction involving a joint multi-channel
    /// categorical, forced into `export_boxes`'s factored path via a deliberately tiny
    /// `TableBudget` (no need for thousands of real rows/levels -- the budget is an explicit
    /// caller parameter, not hardcoded, so a small fixture reaches the exact same code path a
    /// real over-budget high-card interaction would). `numeric0`/`numeric1` are independent
    /// (different periods) so the booster cannot collapse the interaction onto just one of
    /// them; `y` is a genuine 3-way product so a real order-3 split combining all three axes is
    /// the only way to fit it well.
    #[test]
    fn export_boxes_handles_a_genuine_overbudget_order3_joint_cat_interaction() {
        let n = 400;
        let numeric0: Vec<f32> = (0..n).map(|i| (i % 4) as f32).collect();
        let numeric1: Vec<f32> = (0..n).map(|i| ((i / 4) % 4) as f32).collect();
        let cat_high: Vec<String> = (0..n).map(|i| format!("lvl{}", i % 20)).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = numeric0[i] - 1.5;
                let b = numeric1[i] - 1.5;
                let level_signal = (i % 20) as f32 - 9.5;
                // A pure a*b*level_signal product washes out the categorical's OWN marginal
                // target-statistic encoding (E[a*b] ~ 0 over independent numeric0/numeric1, so
                // the mean-encoding per level carries no signal, and the booster never finds it
                // worth splitting on at all -- confirmed empirically: the first draft's tree
                // support histogram was 100% {numeric0, numeric1}, never touching the
                // categorical). Adding an additive main effect for level_signal gives the
                // categorical's own encoding a real, undiluted signal to fit, while the
                // interaction term still requires a genuine order-3 split to fully capture.
                level_signal + 2.0 * a * b * level_signal
            })
            .collect();

        let mut state = booster_with_count_channel(5);
        state.config.n_trees = 200;
        state.interaction = InteractionPolicy {
            max_order: 3,
            ..InteractionPolicy::default()
        };
        let monotone_map = build_monotone_map(None, 2).unwrap();
        let model = fit_model_ambient(
            &state,
            &[numeric0.clone(), numeric1.clone()],
            &y,
            None,
            None,
            None,
            None,
            &monotone_map,
            Some(std::slice::from_ref(&cat_high)),
            None,
        )
        .unwrap();
        // raw 0 = numeric0, raw 1 = numeric1, raw 2 = cat_high (mean+count -> axes 2-3).
        assert_eq!(
            model.provenance.len(),
            4,
            "2 numeric + cat mean/count channels"
        );
        assert!(
            model.provenance.len() > t_boost_core::data::n_raw_features(&model.provenance),
            "fixture must actually be multi-channel"
        );
        let order3_trees = model
            .trees
            .iter()
            .filter(|(_, tree)| {
                let raws: std::collections::BTreeSet<u32> = tree
                    .splits
                    .iter()
                    .map(|s| model.provenance[s.axis as usize].raw.0)
                    .collect();
                raws.len() == 3
            })
            .count();
        assert!(
            order3_trees > 0,
            "fixture must realize a genuine order-3 (3 distinct raw feature) tree, or this \
             test proves nothing about the factored path"
        );

        let binned = serve_binned_for_model(
            &model,
            &[numeric0, numeric1],
            Some(std::slice::from_ref(&cat_high)),
        )
        .unwrap();
        let serve = ServeBinnedMatrix(binned);
        // Deliberately tiny budget: forces the genuine order-3 support over budget without
        // needing thousands of real rows/levels -- explain_with_budget is the exact same
        // pipeline `Model::explain`/`.tables()` use, just with an explicit budget parameter.
        let tiny_budget = TableBudget {
            max_table_cells: 50,
            max_bank_cells: 10_000,
            on_overflow: OverflowPolicy::Factored,
        };
        let bank = model
            .explain_with_budget(&serve, RefMeasure::Uniform, tiny_budget)
            .unwrap();
        assert!(
            !bank.factored.is_empty(),
            "the order-3 support must actually go factored, or this test proves nothing"
        );
        assert!(
            bank.factored.iter().any(|ft| ft.u.order() == 3),
            "a factored triple must include the joint categorical's raw feature"
        );

        let export = bank
            .to_rating_export(
                model.link,
                &model.mode,
                &model.schema,
                &model.provenance,
                &model.schema.cat_encoders,
                None,
            )
            .expect("to_rating_export must not crash on a joint categorical axis in a factored order-3 box (bug #7)");
        assert!(
            !export.factored.is_empty(),
            "the factored order-3 effect must be exported"
        );
        let rf = export
            .factored
            .iter()
            .find(|rf| !rf.boxes.is_empty())
            .expect("the factored effect must export at least one box");

        // `axes`/`feature_set` are sorted by raw id (documented on `FactoredEffect.u`), so the
        // joint categorical (raw 2) is axis index 2.
        assert_eq!(rf.feature_set.0.len(), 3);
        let cat_axis_idx = rf
            .feature_set
            .0
            .iter()
            .position(|f| f.0 == 2)
            .expect("the joint categorical (raw 2) must be one of the factored support's axes");

        // team-lead's ask: the joint axis must be represented by CELLS, not a threshold.
        for b in &rf.boxes {
            assert!(
                b.categorical_low_cells[cat_axis_idx].is_some(),
                "the joint categorical axis must be represented by its low-side cell set, not \
                 a numeric threshold (bug #7)"
            );
        }

        // The collapsed main-effect (order-1) table for the same categorical must still be
        // present and one-relativity-per-level (untouched by this fix, but a real regression
        // risk if the factored path's changes ever leaked into the dense-table path).
        let main_table = export
            .tables
            .iter()
            .find(|t| t.feature_set.order() == 1 && t.feature_set.contains(FeatureId(2)))
            .expect("the joint categorical's own main-effect table must survive");
        assert_eq!(
            main_table.axes.len(),
            1,
            "one axis for the collapsed categorical"
        );
        let levels = main_table.axes[0]
            .levels
            .as_ref()
            .expect("the categorical's axis must export level labels");
        assert!(
            !levels.is_empty(),
            "one relativity per level -- level list must be non-empty"
        );

        // Reconstruction check: the export's own data, replayed independently of the internal
        // `FactoredEffect`/`FactoredBox` this fix never touches, must reproduce `ft.eval` exactly.
        // Numeric axes' low side is a value comparison (`x <= threshold`); cell 1 is always on
        // the low side of ANY realized split (`bin_le >= 1` always -- a split at bin_le=0 has
        // no interior border, engine/mod.rs's own invariant), so fixing both numeric axes at
        // cell 1 isolates the categorical axis's own low/high pattern for direct comparison.
        let ft = bank
            .factored
            .iter()
            .find(|ft| ft.u.order() == 3)
            .expect("the order-3 factored triple must be in the bank");
        let cat_axis = &ft.axes[cat_axis_idx];
        let mut checked_low = 0usize;
        let mut checked_high = 0usize;
        for c2 in 0..cat_axis.cells {
            let mut x_cells = vec![0u32; model.provenance.len()];
            for (d, axis) in ft.axes.iter().enumerate() {
                x_cells[axis.raw.0 as usize] = if d == cat_axis_idx { c2 } else { 1 };
            }
            let ground_truth = ft.eval(&x_cells).unwrap();
            let mut replayed = 0.0_f64;
            for b in &rf.boxes {
                let low_here = b.categorical_low_cells[cat_axis_idx]
                    .as_ref()
                    .unwrap()
                    .contains(&c2);
                if low_here {
                    checked_low += 1;
                } else {
                    checked_high += 1;
                }
                // Both other axes fixed at cell 1 (low, per the reasoning above): idx bit 0/1
                // both set; bit for the categorical axis set iff its cell is low here.
                let idx = 0b011 | (usize::from(low_here) << cat_axis_idx);
                replayed += b.octants[idx];
            }
            assert!(
                (ground_truth - replayed).abs() < 1e-9,
                "cell {c2}: export replay {replayed} != ft.eval {ground_truth} -- the exported \
                 categorical_low_cells must reproduce the model's own factored effect exactly"
            );
        }
        assert!(
            checked_low > 0 && checked_high > 0,
            "the categorical axis must realize both a low and a high side across its cells, or \
             this reconstruction check never exercised the split at all"
        );
    }

    /// Team-lead's other ask alongside bug #7: confirm a SINGLE-channel categorical axis in a
    /// factored order-3 box is handled correctly too (it relies on the threshold/encoding-
    /// borders path, untouched by this fix). No existing test anywhere in the crate exercised
    /// `export_boxes`/`FactoredBoxExport` with ANY categorical axis before this session (only
    /// `factored_rating_export_roundtrips_to_bank_eval`, a purely numeric fixture) -- this
    /// closes that gap with the same exhaustive per-cell reconstruction method as the joint-cat
    /// test above, just via the threshold path instead of `categorical_low_cells`.
    #[test]
    fn export_boxes_handles_a_single_channel_categorical_in_an_order3_box() {
        let n = 400;
        let numeric0: Vec<f32> = (0..n).map(|i| (i % 4) as f32).collect();
        let numeric1: Vec<f32> = (0..n).map(|i| ((i / 4) % 4) as f32).collect();
        // Only 4 levels -- well below cat_count_min_levels (20 below), so this stays
        // single-channel (mean-only), unlike the joint-cat test's 20-level fixture.
        let cat_low: Vec<String> = (0..n).map(|i| format!("lvl{}", i % 4)).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = numeric0[i] - 1.5;
                let b = numeric1[i] - 1.5;
                let level_signal = (i % 4) as f32 - 1.5;
                level_signal + 2.0 * a * b * level_signal
            })
            .collect();

        let mut state = booster_with_count_channel(20);
        state.config.n_trees = 200;
        state.interaction = InteractionPolicy {
            max_order: 3,
            ..InteractionPolicy::default()
        };
        let monotone_map = build_monotone_map(None, 2).unwrap();
        let model = fit_model_ambient(
            &state,
            &[numeric0.clone(), numeric1.clone()],
            &y,
            None,
            None,
            None,
            None,
            &monotone_map,
            Some(std::slice::from_ref(&cat_low)),
            None,
        )
        .unwrap();
        // raw 0 = numeric0, raw 1 = numeric1, raw 2 = cat_low (single-channel, mean only).
        assert_eq!(
            model.provenance.len(),
            3,
            "single-channel: no count-channel axis added"
        );
        assert_eq!(
            model.provenance.len(),
            t_boost_core::data::n_raw_features(&model.provenance),
            "fixture must stay single-channel, or this test proves nothing about that path"
        );
        let order3_trees = model
            .trees
            .iter()
            .filter(|(_, tree)| {
                let raws: std::collections::BTreeSet<u32> = tree
                    .splits
                    .iter()
                    .map(|s| model.provenance[s.axis as usize].raw.0)
                    .collect();
                raws.len() == 3
            })
            .count();
        assert!(
            order3_trees > 0,
            "fixture must realize a genuine order-3 tree, or this test proves nothing"
        );

        let binned = serve_binned_for_model(
            &model,
            &[numeric0, numeric1],
            Some(std::slice::from_ref(&cat_low)),
        )
        .unwrap();
        let serve = ServeBinnedMatrix(binned);
        let tiny_budget = TableBudget {
            max_table_cells: 50,
            max_bank_cells: 10_000,
            on_overflow: OverflowPolicy::Factored,
        };
        let bank = model
            .explain_with_budget(&serve, RefMeasure::Uniform, tiny_budget)
            .unwrap();
        assert!(!bank.factored.is_empty());

        let export = bank
            .to_rating_export(
                model.link,
                &model.mode,
                &model.schema,
                &model.provenance,
                &model.schema.cat_encoders,
                None,
            )
            .expect("to_rating_export must not crash on a single-channel categorical axis in a factored order-3 box");
        let rf = export
            .factored
            .iter()
            .find(|rf| !rf.boxes.is_empty())
            .expect("the factored effect must export at least one box");
        assert_eq!(rf.feature_set.0.len(), 3);
        let cat_axis_idx = rf
            .feature_set
            .0
            .iter()
            .position(|f| f.0 == 2)
            .expect("the single-channel categorical (raw 2) must be one of the axes");
        for b in &rf.boxes {
            assert!(
                b.categorical_low_cells[cat_axis_idx].is_none(),
                "a single-channel categorical is scalar-encoded, so it must keep using the \
                 threshold representation, not categorical_low_cells"
            );
        }

        let ft = bank
            .factored
            .iter()
            .find(|ft| ft.u.order() == 3)
            .expect("the order-3 factored triple must be in the bank");
        let cat_axis = &ft.axes[cat_axis_idx];
        let mut checked_low = 0usize;
        let mut checked_high = 0usize;
        for c2 in 0..cat_axis.cells {
            let mut x_cells = vec![0u32; model.provenance.len()];
            for (d, axis) in ft.axes.iter().enumerate() {
                x_cells[axis.raw.0 as usize] = if d == cat_axis_idx { c2 } else { 1 };
            }
            let ground_truth = ft.eval(&x_cells).unwrap();
            let mut replayed = 0.0_f64;
            for b in &rf.boxes {
                // Threshold path: cell 0 is the reserved missing cell, governed by the
                // SEPARATE `missing_left` flag, not the threshold (the finite_low count
                // `export_boxes` derives the threshold from explicitly skips index 0 -- see
                // explain.rs). Cells 1+ are finite and scalar-ordered: `axis.borders` (the
                // bank's own, not the export's) locates which border the exported threshold
                // names, then "cell <= j + 1" is the same contiguous-prefix rule
                // `export_boxes` itself applied -- mirroring the existing
                // `factored_rating_export_roundtrips_to_bank_eval` test's own reconstruction
                // approach, just for a categorical axis instead of numeric.
                let low_here = if c2 == 0 {
                    b.missing_left[cat_axis_idx]
                } else {
                    let j = cat_axis
                        .borders
                        .iter()
                        .position(|&border| (border - b.thresholds[cat_axis_idx]).abs() < 1e-6)
                        .expect("exported threshold must name a real border on this axis");
                    (c2 as usize) <= j + 1
                };
                if low_here {
                    checked_low += 1;
                } else {
                    checked_high += 1;
                }
                let idx = 0b011 | (usize::from(low_here) << cat_axis_idx);
                replayed += b.octants[idx];
            }
            assert!(
                (ground_truth - replayed).abs() < 1e-9,
                "cell {c2}: export replay {replayed} != ft.eval {ground_truth} -- the exported \
                 threshold must reproduce the model's own factored effect exactly"
            );
        }
        assert!(checked_low > 0 && checked_high > 0);
    }

    /// Bug #8 repro: a genuine order-3 factored interaction spanning {numeric, cat_a
    /// (multi-channel, admitted), cat_b (positioned AFTER cat_a)}. `cat_a`'s own raw id is
    /// LESS than its own count-channel axis index (mean channel axis 1, count channel axis 2,
    /// since numeric occupies axis 0) -- so `cat_b`'s raw id (2) numerically COINCIDES with
    /// `cat_a`'s own count-channel axis index, exactly the scenario team-lead's real beMTPL16
    /// finding needs: the buggy `schema.feature_names.get(raw.0)` (serialize.rs's factored
    /// loop, unlike the main-effect loop just above it, which correctly resolves through
    /// `representative_axis_for_raw`) misattributes `cat_b`'s OWN interaction-support entry to
    /// `cat_a`'s synthetic count-channel name instead.
    #[test]
    fn to_rating_export_factored_feature_names_never_leak_a_synthetic_channel_suffix() {
        let n = 400;
        let numeric0: Vec<f32> = (0..n).map(|i| (i % 4) as f32).collect();
        let cat_a: Vec<String> = (0..n).map(|i| format!("a{}", i % 20)).collect();
        let cat_b: Vec<String> = (0..n).map(|i| format!("b{}", (i / 4) % 4)).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let num_signal = numeric0[i] - 1.5;
                let cat_b_signal = ((i / 4) % 4) as f32 - 1.5;
                let cat_a_signal = (i % 20) as f32 - 9.5;
                cat_a_signal + cat_b_signal + 2.0 * num_signal * cat_b_signal * cat_a_signal
            })
            .collect();

        let mut state = booster_with_count_channel(5);
        state.config.n_trees = 200;
        state.interaction = InteractionPolicy {
            max_order: 3,
            ..InteractionPolicy::default()
        };
        let monotone_map = build_monotone_map(None, 1).unwrap();
        let feature_names = Some(vec![
            "numeric0".to_string(),
            "cat_a".to_string(),
            "cat_b".to_string(),
        ]);
        let model = fit_model_ambient(
            &state,
            std::slice::from_ref(&numeric0),
            &y,
            None,
            None,
            feature_names,
            None,
            &monotone_map,
            Some(&[cat_a.clone(), cat_b.clone()]),
            None,
        )
        .unwrap();
        // raw 0 = numeric0 (axis 0); raw 1 = cat_a, admitted (axes 1 mean, 2 count); raw 2 =
        // cat_b, single-channel (axis 3) -- cat_b's raw id (2) coincides with cat_a's own
        // count-channel axis index, the exact collision this bug needs.
        assert_eq!(model.provenance.len(), 4);
        assert!(model.provenance.len() > t_boost_core::data::n_raw_features(&model.provenance));

        let order3_trees = model
            .trees
            .iter()
            .filter(|(_, tree)| {
                let raws: std::collections::BTreeSet<u32> = tree
                    .splits
                    .iter()
                    .map(|s| model.provenance[s.axis as usize].raw.0)
                    .collect();
                raws.len() == 3
            })
            .count();
        assert!(
            order3_trees > 0,
            "fixture must realize a genuine order-3 (3 distinct raw feature) tree, or this \
             test proves nothing"
        );

        let binned = serve_binned_for_model(
            &model,
            std::slice::from_ref(&numeric0),
            Some(&[cat_a, cat_b]),
        )
        .unwrap();
        let serve = ServeBinnedMatrix(binned);
        let tiny_budget = TableBudget {
            max_table_cells: 50,
            max_bank_cells: 10_000,
            on_overflow: OverflowPolicy::Factored,
        };
        let bank = model
            .explain_with_budget(&serve, RefMeasure::Uniform, tiny_budget)
            .unwrap();
        assert!(!bank.factored.is_empty());

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

        // The permanent regression assertion team-lead asked for: NO synthetic channel suffix
        // anywhere in the export -- main tables' feature_names/axis names, AND factored
        // interactions' feature_names alike.
        let mut leaks: Vec<String> = Vec::new();
        for t in &export.tables {
            for name in &t.feature_names {
                if name.contains('#') {
                    leaks.push(format!("table {:?} feature_names: {name:?}", t.feature_set));
                }
            }
            for axis in &t.axes {
                if axis.name.contains('#') {
                    leaks.push(format!(
                        "table {:?} axis name: {:?}",
                        t.feature_set, axis.name
                    ));
                }
            }
        }
        for rf in &export.factored {
            for name in &rf.feature_names {
                if name.contains('#') {
                    leaks.push(format!(
                        "factored {:?} feature_names: {name:?}",
                        rf.feature_set
                    ));
                }
            }
        }
        assert!(
            leaks.is_empty(),
            "to_rating_export must never expose a synthetic channel-suffixed name (bug #8): {leaks:?}"
        );

        // Positive check, not just absence: the factored interaction's feature_names must be
        // the CORRECT raw-feature names, in the same order as feature_set (raw id order) --
        // not merely suffix-free by coincidence.
        let rf = export
            .factored
            .iter()
            .find(|rf| rf.feature_set.0.len() == 3)
            .expect("the order-3 factored interaction must be exported");
        assert_eq!(
            rf.feature_names,
            vec![
                "numeric0".to_string(),
                "cat_a".to_string(),
                "cat_b".to_string()
            ],
            "each factored axis must be named after its OWN raw feature, not a differently- \
             positioned raw feature's synthetic channel name"
        );
    }
}

fn parse_cat_leakage(name: Option<&str>, n_perms: u32, k: u32) -> Result<LeakageScheme, PbError> {
    match name.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        // Default to K-fold cross-fit: lowest-variance leakage-free encoding measured on MTPL
        // (beats both the old Ordered{1} default and leave-one-out on frequency and severity).
        None | Some("kfold") | Some("k_fold") | Some("crossfit") | Some("cross_fit") => {
            Ok(LeakageScheme::KFold { k })
        }
        Some("ordered") => Ok(LeakageScheme::Ordered { n_perms }),
        Some("loo") | Some("leave_one_out") | Some("leaveoneout") => Ok(LeakageScheme::LeaveOneOut),
        Some(other) => Err(PbError::InvalidConfig {
            what: format!("cat_leakage must be 'kfold', 'ordered', or 'loo', got {other:?}"),
        }),
    }
}

fn feature_set_from_raw_ids(ids: &[u32]) -> FeatureSet {
    let mut ids = ids.to_vec();
    ids.sort_unstable();
    ids.dedup();
    FeatureSet::new(&ids)
}

fn feature_set_ids(u: &FeatureSet) -> Vec<u32> {
    u.0.iter().map(|f| f.0).collect()
}

fn add_downward_closure(
    selected: &mut BTreeSet<FeatureSet>,
    full_supports: &BTreeSet<FeatureSet>,
) -> BTreeSet<FeatureSet> {
    let before = selected.clone();
    for fs in before.iter() {
        let ids = feature_set_ids(fs);
        if ids.len() <= 1 {
            continue;
        }
        let end = 1usize << ids.len();
        for mask in 1..(end - 1) {
            let mut sub = Vec::new();
            for (i, &id) in ids.iter().enumerate() {
                if (mask & (1usize << i)) != 0 {
                    sub.push(id);
                }
            }
            let sub_fs = feature_set_from_raw_ids(&sub);
            if full_supports.contains(&sub_fs) {
                selected.insert(sub_fs);
            }
        }
    }
    selected.difference(&before).cloned().collect()
}

/// Mean and standard error of the mean of `xs` (sample SD, `ddof = 1`). `None` when fewer than
/// two observations exist — one fold is a point estimate with no measurable spread, not a
/// zero-SE certainty.
fn mean_se_of(xs: &[f64]) -> (f64, Option<f64>) {
    let n = xs.len();
    if n == 0 {
        return (0.0, None);
    }
    let m = xs.iter().sum::<f64>() / n as f64;
    if n < 2 {
        return (m, None);
    }
    let var = xs.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n as f64 - 1.0);
    (m, Some((var / n as f64).sqrt()))
}

fn aggregate_prune_selection(
    full_supports_raw: Vec<Vec<u32>>,
    reports: &[Option<PruneReport>],
    min_stability: f64,
    // Minimum mean held-out drop-gain a table must clear to be kept on the legacy rule. `0.0`
    // (the shipped default) is the historical `mean_gain > 0.0` test, bit-for-bit. It was a
    // literal here AND a literal `0.0` in the report below, so the report named a threshold the
    // caller could not set. Raising it is the threshold-parameterised counterpart to
    // `prune_table_budget`'s count cap: drop every table whose held-out evidence is weaker than
    // this, rather than keeping the best-ranked N.
    min_mean_gain: f64,
    // --- EVIDENCE-GATED DROPPING (av37) -------------------------------------------------
    // `drop_z = None` reproduces the av36 aggregator byte-for-byte. Set, it inverts the
    // WEAK-EVIDENCE DEFAULT: a table with paired per-fold evidence is dropped only when its
    // mean drop-gain clears `drop_z` standard errors of that mean over the CV folds. See the
    // long note at the selection loop for why, and for what `keep_budget` is protecting.
    drop_z: Option<f64>,
    keep_budget: usize,
    // Provenance only: which fold fidelity produced the evidence below. Recorded so a stored
    // selection says how its keep-set was chosen; the aggregation itself is unchanged.
    fold_fidelity: bool,
) -> Result<(Vec<Vec<u32>>, String), PbError> {
    let full_supports: BTreeSet<FeatureSet> = full_supports_raw
        .iter()
        .map(|ids| feature_set_from_raw_ids(ids))
        .filter(|u| u.order() > 0)
        .collect();
    let full_supports_vec: Vec<FeatureSet> = full_supports.iter().cloned().collect();
    let n_reports = reports.iter().filter(|r| r.is_some()).count().max(1);
    let n_reports_f = n_reports as f64;

    let mut kept_counts: BTreeMap<FeatureSet, usize> = BTreeMap::new();
    let mut gain_sum: BTreeMap<FeatureSet, f64> = BTreeMap::new();
    let mut gain_n: BTreeMap<FeatureSet, usize> = BTreeMap::new();
    let mut positive_counts: BTreeMap<FeatureSet, usize> = BTreeMap::new();
    // Per-table PAIRED per-fold drop-gains, in fold order. `gain_sum`/`gain_n` alone cannot
    // answer "is this table's mean gain distinguishable from zero?" — the spread is the whole
    // question the av37 gate asks, so the raw values are retained. Fold order is the reports'
    // own Vec order, so this is deterministic under any thread count.
    let mut gain_vals: BTreeMap<FeatureSet, Vec<f64>> = BTreeMap::new();
    let mut fold_score_rows = Vec::new();

    for report in reports.iter().flatten() {
        let fold_kept: BTreeSet<FeatureSet> = report.kept.iter().cloned().collect();
        for fs in fold_kept {
            if fs.order() > 0 {
                *kept_counts.entry(fs).or_insert(0) += 1;
            }
        }
        for ts in &report.table_scores {
            if ts.u.order() == 0 {
                continue;
            }
            *gain_sum.entry(ts.u.clone()).or_insert(0.0) += ts.mean_gain;
            *gain_n.entry(ts.u.clone()).or_insert(0) += 1;
            gain_vals
                .entry(ts.u.clone())
                .or_default()
                .push(ts.mean_gain);
            if ts.mean_gain > 0.0 {
                *positive_counts.entry(ts.u.clone()).or_insert(0) += 1;
            }
            fold_score_rows.push(serde_json::json!({
                "u": feature_set_ids(&ts.u),
                "order": ts.order,
                "mean_gain": ts.mean_gain,
                "se_gain": ts.se_gain,
                "variance": ts.variance,
                "fold_selected": ts.selected,
                "sticky": ts.sticky,
            }));
        }
    }

    let mut selected: BTreeSet<FeatureSet> = full_supports
        .iter()
        .filter(|u| u.order() == 1)
        .cloned()
        .collect();
    // Candidate admissions the EVIDENCE gate wants but the legacy rule refuses. `(mean_gain,
    // ids, u)`: the budget below spends on the highest mean gain first, with the feature-set
    // ids as a total, seed-free tie-break, so the admitted prefix is a pure function of the
    // fold scores. Same ranking the guard's re-admission ladder uses, for the same reason.
    let mut evidence_extra: Vec<(f64, Vec<u32>, FeatureSet)> = Vec::new();
    let mut evidence_rows: Vec<serde_json::Value> = Vec::new();
    for fs in &full_supports_vec {
        if fs.order() == 1 {
            continue;
        }
        let kept_rate = *kept_counts.get(fs).unwrap_or(&0) as f64 / n_reports_f;
        let gn = (*gain_n.get(fs).unwrap_or(&0)).max(1);
        let mean_gain = *gain_sum.get(fs).unwrap_or(&0.0) / gn as f64;
        let positive_rate = *positive_counts.get(fs).unwrap_or(&0) as f64 / gn as f64;
        // Gate on measured held-out gain: a table that hurts CV deviance on average is never
        // kept on stability alone (downward closure may still re-add it to honor heredity).
        let legacy_keep = mean_gain > min_mean_gain
            && (kept_rate >= min_stability || positive_rate >= min_stability);
        if legacy_keep {
            selected.insert(fs.clone());
        }
        // --- EVIDENCE-GATED DROPPING (av37) ---------------------------------------------
        // The rule above DROPS ON WEAK EVIDENCE: `mean_gain > 0` is a coin flip on the sign of
        // a fold-mean whose own scatter dwarfs it, and `positive_rate >= 0.5` is satisfied by a
        // SINGLE fold that happened to score positive. Measured (trio forensic, 90 deploy fits
        // per dataset on identical data, seed the only difference): swautoins keeps 11.0 +/- 1.9
        // of its 14 candidate tables and drops each of its four 3-way tables 40-64 times out of
        // 90; the damage is monotone in how many were thrown away (skill delta vs an unpruned
        // fit: +0.1176 at 7 kept, +0.0090 at 11, -0.0001 at 14).
        //
        // So the DEFAULT is inverted here: with paired per-fold evidence in hand, a table is
        // dropped only when its mean drop-gain clears `drop_z` standard errors of that mean over
        // the folds. No evidence => keep. Strong keep-evidence => keep. Strong DROP-evidence =>
        // drop, binary, exactly as before. Because the legacy rule already requires
        // `mean_gain > 0` to keep and this one requires `mean < -z*se` to drop, the gate is
        // MONOTONE: it can only ever keep MORE tables than av36, never fewer. That is what makes
        // the two guarantees below checkable rather than hopeful.
        //
        // TWO DELIBERATE LIMITS, both there to protect deployed-bank size — a 100-table 3-way
        // bank is not an explainable rating structure, whatever it scores:
        //
        //  (1) FEWER THAN TWO SCORING FOLDS => the legacy verdict stands. A table absent from a
        //      fold's bank was never built by a model trained on 80% of the data; that is a
        //      structural fact, not an ambiguous measurement, and there is no paired spread to
        //      test. On the large banks this is the common case (measured: 74% of credit_g's
        //      302 candidates and the bulk of allstate_sev's are scored by no fold at all), and
        //      it is exactly why those banks do not balloon under this gate.
        //  (2) THE EXPLAINABILITY BUDGET `keep_budget`. Ambiguous admissions are ranked by
        //      mean_gain and admitted only while the pre-cascade keep-set stays within
        //      `max(keep_budget, legacy_keep_set_size)`. Where the legacy bank is already larger
        //      than the budget the gate is a NO-OP by construction, so big-set behavior is
        //      unchanged; where the whole candidate bank is small enough to be read by a human
        //      the gate is unconstrained and every ambiguous table comes back.
        if let Some(z) = drop_z {
            let gains = gain_vals.get(fs).map(Vec::as_slice).unwrap_or(&[]);
            let (m, se) = mean_se_of(gains);
            let evidence_keep = match se {
                // `se == 0.0` (every fold measured the identical gain) leaves the bar at zero,
                // which is the right limit: perfectly reproduced harm IS significant harm.
                // JUSTIFIED: `!(m < bar)` is deliberate — a NaN mean keeps the table.
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                Some(se) => !(m < -z * se),
                None => legacy_keep,
            };
            let t_stat = match se {
                Some(se) if se > 0.0 => m / se,
                Some(_) => f64::INFINITY * m.signum(),
                None => f64::NAN,
            };
            evidence_rows.push(serde_json::json!({
                "u": feature_set_ids(fs),
                "gain_n": gains.len(),
                "mean_gain": m,
                "se_gain": se,
                "t": if t_stat.is_finite() { serde_json::json!(t_stat) } else { serde_json::Value::Null },
                "legacy_keep": legacy_keep,
                "evidence_keep": evidence_keep,
            }));
            if evidence_keep && !legacy_keep {
                evidence_extra.push((mean_gain, feature_set_ids(fs), fs.clone()));
            }
        }
    }
    // Rank the ambiguous admissions and spend the budget. `total_cmp` gives a total order on the
    // f64 key (no NaN surprises); the feature-set ids break ties, so the admitted prefix is a
    // pure function of the fold scores — identical under any thread count or fold ordering.
    evidence_extra.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let legacy_selected = selected.len();
    let budget = keep_budget.max(legacy_selected);
    let mut evidence_admitted: Vec<Vec<u32>> = Vec::new();
    for (_, ids, fs) in &evidence_extra {
        if selected.len() >= budget {
            break;
        }
        if selected.insert(fs.clone()) {
            evidence_admitted.push(ids.clone());
        }
    }
    let evidence_budget_bound = drop_z.is_some() && evidence_admitted.len() < evidence_extra.len();

    // Heredity by cascade-removal, not closure-addition: a higher-order table stays only if every
    // immediate subset also earned its keep — otherwise a kept triple re-admits (via closure)
    // exactly the negative-gain pairs the gate removed. Mains are sticky, so the cascade only
    // travels upward from gated-out interactions.
    let mut cascade_removed: Vec<FeatureSet> = Vec::new();
    loop {
        let doomed: Vec<FeatureSet> = selected
            .iter()
            .filter(|fs| {
                let ids = feature_set_ids(fs);
                ids.len() > 1
                    && (0..ids.len()).any(|skip| {
                        let sub: Vec<u32> = ids
                            .iter()
                            .enumerate()
                            .filter_map(|(i, id)| (i != skip).then_some(*id))
                            .collect();
                        sub.len() > 1 && !selected.contains(&feature_set_from_raw_ids(&sub))
                    })
            })
            .cloned()
            .collect();
        if doomed.is_empty() {
            break;
        }
        for fs in doomed {
            selected.remove(&fs);
            cascade_removed.push(fs);
        }
    }

    let closure_added = add_downward_closure(&mut selected, &full_supports);
    let keep: Vec<Vec<u32>> = selected.iter().map(feature_set_ids).collect();
    let summary_rows: Vec<_> = full_supports_vec
        .iter()
        .map(|fs| {
            let gn = (*gain_n.get(fs).unwrap_or(&0)).max(1);
            serde_json::json!({
                "u": feature_set_ids(fs),
                "order": fs.order(),
                "kept_rate": *kept_counts.get(fs).unwrap_or(&0) as f64 / n_reports_f,
                "positive_rate": *positive_counts.get(fs).unwrap_or(&0) as f64 / gn as f64,
                "mean_gain": *gain_sum.get(fs).unwrap_or(&0.0) / gn as f64,
                "selected": selected.contains(fs),
                "sticky": fs.order() == 1,
            })
        })
        .collect();
    let effective_order = keep.iter().map(Vec::len).max().unwrap_or(0);
    let report = serde_json::json!({
        "kept": keep,
        "effective_order": effective_order,
        "cv_folds": reports.iter().filter(|r| r.is_some()).count(),
        "main_effect_policy": "sticky",
        "selector": "heldout_contribution_stability",
        "min_stability": min_stability,
        "min_mean_gain": min_mean_gain,
        "table_scores": summary_rows,
        "fold_table_scores": fold_score_rows,
        "closure_added": closure_added.iter().map(feature_set_ids).collect::<Vec<_>>(),
        "cascade_removed": cascade_removed.iter().map(feature_set_ids).collect::<Vec<_>>(),
        // av37 evidence gate. `drop_z: null` is the av36 aggregator, and then every field below
        // is inert (no admissions, no budget pressure) — the report shape stays stable either way.
        "drop_z": drop_z,
        "keep_budget": keep_budget,
        "fold_fidelity": fold_fidelity,
        "legacy_selected": legacy_selected,
        "evidence_budget": budget,
        "evidence_candidates": evidence_extra.len(),
        "evidence_admitted": evidence_admitted,
        "evidence_budget_bound": evidence_budget_bound,
        "evidence_scores": evidence_rows,
    });
    let report_json = serde_json::to_string(&report).map_err(|err| {
        PbError::Serialization(format!("could not serialize prune selection report: {err}"))
    })?;
    Ok((selected.iter().map(feature_set_ids).collect(), report_json))
}

#[allow(clippy::too_many_arguments)]
fn fit_prune_reports_owned(
    state: PyBooster,
    columns: Vec<Vec<f32>>,
    y_vec: Vec<f32>,
    fold_of: Vec<i64>,
    k_folds: usize,
    weight_vec: Option<Vec<f32>>,
    exposure_vec: Option<Vec<f32>>,
    feature_names: Option<Vec<String>>,
    class_labels: Option<Vec<String>>,
    monotone_map: MonotoneMap,
    cat_x: Option<Vec<Vec<String>>>,
    w_measure: RefMeasure,
    se_rule: f64,
    lambda_boxes: f64,
    lambda_tables: f64,
    table_min_arity: u8,
    n_folds: usize,
    reanchor: bool,
    // Caller-supplied honest ES holdout over ALL rows (e.g. a GROUP-aware carve for panel
    // data): gathered per fold to the fold's own train subset below and threaded into that
    // fold's nested fit exactly like the top-level `fit()`'s `es_holdout`, so a fold's own
    // internal early-stopping carve is ALSO group-honest instead of leaking near-duplicate rows
    // of the same entity across its train-proper/validation split. `None` is byte-identical to
    // before this parameter existed (each fold model derives its own row-level carve).
    es_holdout: Option<Vec<bool>>,
    // PINNED-BANK FOLD FIDELITY (`prune_fold_fidelity`). `None` is byte-identical to before
    // this parameter existed: each fold report scores only the supports that fold's own
    // structure search built, and the aggregator's `gain_n == 0` candidates fall back to the
    // legacy verdict. `Some(supports)` gives every fold bank a value-carrying table for each
    // candidate support it is MISSING, fit on that fold's TRAIN rows only, so the paired
    // per-fold statistic exists for the whole candidate set. See `prune::FoldFidelity`.
    fold_fidelity: Option<Vec<Vec<u32>>>,
) -> Result<Vec<Option<PruneReport>>, PbError> {
    let n = y_vec.len();
    let w_full: Vec<f32> = match &weight_vec {
        Some(w) => w.clone(),
        None => vec![1.0_f32; n],
    };
    let offset_full: Option<Vec<f32>> = exposure_vec
        .as_ref()
        .map(|e| e.iter().map(|&v| v.ln()).collect());
    let fold_task = |k: usize| -> Result<Option<PruneReport>, PbError> {
        let k_i = k as i64;
        let mut held: Vec<usize> = Vec::new();
        let mut train: Vec<usize> = Vec::new();
        for (i, &f) in fold_of.iter().enumerate() {
            if f == k_i {
                held.push(i);
            } else {
                train.push(i);
            }
        }
        if held.is_empty() || train.is_empty() {
            return Ok(None);
        }
        let train_cols: Vec<Vec<f32>> = columns
            .iter()
            .map(|c| gather_f32(c, &train))
            .collect::<Result<_, _>>()?;
        let train_y = gather_f32(&y_vec, &train)?;
        let train_w = match &weight_vec {
            Some(w) => Some(gather_f32(w, &train)?),
            None => None,
        };
        let train_e = match &exposure_vec {
            Some(e) => Some(gather_f32(e, &train)?),
            None => None,
        };
        let train_cat = match &cat_x {
            Some(cats) => Some(
                cats.iter()
                    .map(|col| gather_str(col, &train))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            None => None,
        };
        let train_es_holdout: Option<Vec<bool>> = match &es_holdout {
            Some(mask) => Some(gather_bool(mask, &train)?),
            None => None,
        };
        let fold_model = fit_model_ambient(
            &state,
            &train_cols,
            &train_y,
            train_w.as_deref(),
            train_e.as_deref(),
            feature_names.clone(),
            class_labels.clone(),
            &monotone_map,
            train_cat.as_deref(),
            train_es_holdout.as_deref(),
        )?;
        let binned = serve_binned_for_model(&fold_model, &columns, cat_x.as_deref())?;
        let serve = ServeBinnedMatrix(binned);
        let loss = loss_choice_from_tag(&fold_model.schema.objective)?;
        let cfg = PruneConfig {
            se_rule,
            lambda_boxes,
            lambda_tables,
            table_price_min_arity: table_min_arity,
        };
        // The fidelity augmentation may fit ONLY on this fold's train side: `held` is exactly
        // the rows its evidence is scored on, so they carry IRLS weight 0 in the solve.
        let fit_rows: Option<Vec<bool>> = fold_fidelity.as_ref().map(|_| {
            let mut mask = vec![true; n];
            for &r in &held {
                if let Some(slot) = mask.get_mut(r) {
                    *slot = false;
                }
            }
            mask
        });
        let ff = match (&fold_fidelity, &fit_rows) {
            (Some(supports), Some(mask)) => Some(t_boost_core::prune::FoldFidelity {
                candidate_supports: supports.as_slice(),
                fit_rows: mask.as_slice(),
                spec: t_boost_core::prune::FoldFidelitySpec::default(),
            }),
            _ => None,
        };
        let (_tm, report) = prune_model_to_tables_pinned(
            &fold_model,
            &serve,
            &y_vec,
            &w_full,
            offset_full.as_deref(),
            loss.as_loss(),
            w_measure.clone(),
            reanchor,
            &held,
            n_folds,
            &cfg,
            ff.as_ref(),
        )?;
        Ok(Some(report))
    };
    let run_all = || -> Result<Vec<Option<PruneReport>>, PbError> {
        (0..k_folds).into_par_iter().map(fold_task).collect()
    };
    run_fit_pool(state.n_jobs, state.fit_pool_width, run_all)
}

#[pymethods]
impl PyBooster {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        n_trees=1000,
        learning_rate=0.05,
        lambda_=1.0,
        lambda_scale_invariant=false,
        l1_leaf=0.0,
        min_split_gain=0.0,
        max_delta_step=None,
        max_delta_step_gated=None,
        max_bin=254,
        objective=None,
        tweedie_rho=1.5,
        min_data_in_leaf=0,
        min_sum_hessian_in_leaf=0.0,
        min_weight_sum_in_leaf=0.0,
        path_smooth=0.0,
        subsample=None,
        colsample_bytree=1.0,
        learning_rate_decay=0.0,
        validation_fraction=None,
        early_stopping_rounds=50,
        early_stopping_adaptive=None,
        early_stopping_min_delta=0.0,
        interaction_gain_hurdle=0.0,
        interaction_gain_hurdle_mode=None,
        leaf_refine_steps=0,
        leaf_refine_backtracks=4,
        refine_closed_form_tier2=true,
        incremental_mu=false,
        mvs_min_rows=1,
        hist_precision=None,
        n_bags=0,
        bag_subsample=1.0,
        ridge_refit_l2=None,
        ridge_refit_max_iter=5,
        nesterov=false,
        dart_drop_rate=None,
        random_strength=0.0,
        reanchor=false,
        reanchor_slope=false,
        max_interaction_order=3,
        max_depth=3,
        table_budget_cells=2_000_000,
        table_budget_order_shrink=2.0,
        cat_smooth=None,
        cat_target=None,
        cat_leakage=None,
        cat_n_perms=1,
        cat_k=5,
        cat_min_data_per_group=10.0,
        cat_direct_max_levels=16,
        cat_channels=None,
        cat_count_min_levels=20,
        cat_class_freq_min_levels=3,
        cell_refit_base=None,
        cell_refit_gamma=2.0,
        seed=0,
        n_jobs=None,
        fit_pool_width=None
    ))]
    fn new(
        n_trees: u32,
        learning_rate: f32,
        lambda_: f32,
        lambda_scale_invariant: bool,
        l1_leaf: f32,
        min_split_gain: f32,
        max_delta_step: Option<f32>,
        max_delta_step_gated: Option<Bound<'_, PyAny>>,
        max_bin: u8,
        objective: Option<String>,
        tweedie_rho: f32,
        min_data_in_leaf: u32,
        min_sum_hessian_in_leaf: f32,
        min_weight_sum_in_leaf: f32,
        path_smooth: f32,
        subsample: Option<f32>,
        colsample_bytree: f32,
        learning_rate_decay: f32,
        validation_fraction: Option<f32>,
        early_stopping_rounds: u32,
        early_stopping_adaptive: Option<f64>,
        early_stopping_min_delta: f64,
        interaction_gain_hurdle: f32,
        interaction_gain_hurdle_mode: Option<String>,
        leaf_refine_steps: u8,
        leaf_refine_backtracks: u8,
        refine_closed_form_tier2: bool,
        incremental_mu: bool,
        mvs_min_rows: u32,
        hist_precision: Option<String>,
        n_bags: u16,
        bag_subsample: f32,
        ridge_refit_l2: Option<f32>,
        ridge_refit_max_iter: u8,
        nesterov: bool,
        dart_drop_rate: Option<f32>,
        random_strength: f32,
        reanchor: bool,
        reanchor_slope: bool,
        max_interaction_order: u8,
        max_depth: u8,
        table_budget_cells: u64,
        table_budget_order_shrink: f32,
        cat_smooth: Option<f32>,
        cat_target: Option<String>,
        cat_leakage: Option<String>,
        cat_n_perms: u32,
        cat_k: u32,
        cat_min_data_per_group: f32,
        cat_direct_max_levels: u32,
        cat_channels: Option<Vec<String>>,
        cat_count_min_levels: u32,
        cat_class_freq_min_levels: u32,
        cell_refit_base: Option<f64>,
        cell_refit_gamma: f64,
        seed: u64,
        n_jobs: Option<usize>,
        fit_pool_width: Option<usize>,
    ) -> PyResult<Self> {
        let cell_refit = cell_refit_base.map(|base| CellRefit {
            base,
            gamma: cell_refit_gamma,
        });
        let ensemble = if n_bags == 0 {
            // The OOB fANOVA cell refit fits on each bag's out-of-bag rows, so it has no
            // meaning without a bag partition: without this check `cell_refit` is silently
            // dropped in the `Off` arm below and the caller gets an ordinary fit with no
            // diagnostic (g1-cell-refit-shipped is the lever that closed the particulate
            // near-tie, so users set it directly on the native API and need to know it no-op'd).
            if cell_refit.is_some() {
                return Err(py_err(PbError::InvalidConfig {
                    what: "cell_refit_base requires n_bags >= 1 (OOB cell refit needs bag \
                           partitions)"
                        .into(),
                }));
            }
            EnsembleSpec::Off
        } else {
            EnsembleSpec::OuterBag {
                n_bags,
                bag_subsample,
                cell_refit,
            }
        };
        let refit_leaves = match ridge_refit_l2 {
            None => RefitSpec::Off,
            Some(l2) => RefitSpec::Ridge {
                l2,
                max_iter: ridge_refit_max_iter,
                every_k_trees: None,
            },
        };
        if nesterov {
            // AGBM look-ahead currently diverges (the §09.4 momentum-correction step is not
            // implemented), so refuse it loudly rather than silently produce a blown-up model.
            return Err(py_err(PbError::InvalidConfig {
                what: "nesterov/AGBM acceleration is experimental and currently unstable \
                       (it diverges; the momentum-correction step is not yet implemented) — \
                       it is not supported in this release"
                    .into(),
            }));
        }
        let nesterov = NesterovSpec::Off;
        let dart = dart_drop_rate.map(|drop_rate| DartSpec {
            drop_rate,
            normalize: true,
        });
        let boosters = BoosterConfig {
            refit_leaves,
            nesterov,
            ensemble,
            dart,
            random_strength,
            reanchor,
            reanchor_slope,
        };
        let config = Config {
            n_trees,
            learning_rate,
            lambda: lambda_,
            lambda_scale_invariant,
            l1_leaf,
            min_split_gain,
            max_delta_step,
            max_delta_step_gated: parse_gated_step_policy(max_delta_step_gated.as_ref())?,
            sampling: parse_sampling(subsample, mvs_min_rows),
            colsample_bytree,
            learning_rate_decay,
            validation_fraction,
            early_stopping_rounds,
            early_stopping_adaptive,
            early_stopping_min_delta,
            interaction_gain_hurdle,
            interaction_gain_hurdle_mode: parse_interaction_gain_hurdle_mode(
                interaction_gain_hurdle_mode.as_deref(),
            )
            .map_err(py_err)?,
            leaf_refine_steps,
            leaf_refine_backtracks,
            refine_closed_form_tier2,
            incremental_mu,
            hist_precision: parse_hist_precision(hist_precision.as_deref()).map_err(py_err)?,
            boosters,
        };
        config.validate().map_err(py_err)?;
        let bin_config = BinConfig {
            max_bin,
            ..BinConfig::default()
        };
        bin_config.validate().map_err(py_err)?;
        let credibility = CredibilityFloor {
            min_data_in_leaf,
            min_sum_hessian_in_leaf,
            min_weight_sum_in_leaf,
            path_smooth,
        };
        credibility.validate().map_err(py_err)?;
        let interaction = InteractionPolicy {
            max_order: max_interaction_order,
            max_depth,
            table_budget_cells,
            table_budget_order_shrink,
            ..InteractionPolicy::default()
        };
        // Categorical target-statistic config: default to empirical-Bayes Auto smoothing
        // (spec §04), overridable by a fixed pseudo-count via `cat_smooth`.
        let cat_config = TsConfig {
            leakage: parse_cat_leakage(cat_leakage.as_deref(), cat_n_perms, cat_k)
                .map_err(py_err)?,
            smooth: match cat_smooth {
                None => Smooth::Auto,
                Some(m) => Smooth::Fixed { m },
            },
            target: parse_cat_target(cat_target.as_deref()).map_err(py_err)?,
            min_data_per_group: cat_min_data_per_group,
            direct_max_levels: cat_direct_max_levels,
            ..TsConfig::default()
        };
        cat_config.validate().map_err(py_err)?;
        // P1 multi-channel (design/multichannel-categoricals.md §4.5): the count channel
        // mirrors `cat_config`'s rare-pooling floor (same `min_data_per_group`, so both
        // channels of one categorical partition rare levels identically) but is otherwise
        // target-free — `leakage`/`smooth`/`direct_max_levels` are inert for `CatTarget::Count`
        // (see `fit_count_encoder`), so leaving them at `cat_config`'s values is harmless.
        let cat_channel_plan = parse_cat_channels(cat_channels.as_deref()).map_err(py_err)?;
        let cat_count_config = if cat_channel_plan.count {
            let count_config = TsConfig {
                target: CatTarget::Count,
                ..cat_config.clone()
            };
            count_config.validate().map_err(py_err)?;
            Some(count_config)
        } else {
            None
        };
        // P3 per-class frequency channels: same rare-pooling floor and same cross-fit/smoothing
        // machinery as the mean channel (it IS a mean, of the one-vs-rest class indicator), so
        // the template is `cat_config` verbatim — only `target` is restamped per class at fit
        // time, once the class count is known.
        let cat_class_freq_config = if cat_channel_plan.class_freq {
            Some(cat_config.clone())
        } else {
            None
        };
        let objective = Objective::parse(objective, tweedie_rho).map_err(py_err)?;
        if matches!(n_jobs, Some(0)) {
            return Err(py_err(PbError::InvalidConfig {
                what: "n_jobs must be >= 1 when set".into(),
            }));
        }
        if matches!(fit_pool_width, Some(0)) {
            return Err(py_err(PbError::InvalidConfig {
                what: "fit_pool_width must be >= 1 when set".into(),
            }));
        }
        Ok(Self {
            config,
            bin_config,
            objective,
            interaction,
            credibility,
            cat_config,
            cat_count_config,
            cat_count_min_levels,
            cat_class_freq_min_levels,
            cat_channels: cat_channel_plan,
            cat_class_freq_config,
            seed,
            n_jobs,
            fit_pool_width,
        })
    }

    #[pyo3(signature = (x, y, weight=None, exposure=None, feature_names=None, class_labels=None, monotone=None, cat_x=None, es_holdout=None, bag_groups=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        y: PyReadonlyArray1<'_, f32>,
        weight: Option<PyReadonlyArray1<'_, f32>>,
        exposure: Option<PyReadonlyArray1<'_, f32>>,
        feature_names: Option<Vec<String>>,
        class_labels: Option<Vec<String>>,
        monotone: Option<Vec<i8>>,
        // Native categorical columns as per-row string labels: `cat_x[j]` is one column,
        // appended after the numeric columns of `x` (raw ids assigned sequentially).
        cat_x: Option<Vec<Vec<String>>>,
        // Caller-supplied honest ES holdout (e.g. a GROUP-aware carve for panel data, where the
        // default row-level carve leaks near-duplicate rows into validation and defeats early
        // stopping — the single-output analog of `fit_multiclass`'s `es_holdout`). `None` keeps
        // the engine's own row-level carve, byte-identical to before this argument existed.
        es_holdout: Option<Vec<bool>>,
        bag_groups: Option<PyReadonlyArray1<'_, u32>>,
    ) -> PyResult<PyModel> {
        cap_global_pool_once(self.n_jobs);
        let bag_groups = bag_groups
            .map(|g| array1_to_vec_u32(g, "bag_groups"))
            .transpose()?;
        let columns = raw_columns_from_array(x)?;
        if columns.len() + cat_x.as_ref().map_or(0, Vec::len) == 0 {
            return Err(py_err(PbError::InvalidInput {
                what: "x must contain at least one feature (numeric or categorical)".into(),
            }));
        }
        let y = array1_to_vec(y, "y")?;
        let weight = weight.map(|w| array1_to_vec(w, "weight")).transpose()?;
        let exposure = exposure.map(|e| array1_to_vec(e, "exposure")).transpose()?;
        let state = self.clone();
        let model = py
            .detach(move || {
                fit_owned(
                    state,
                    columns,
                    y,
                    weight,
                    exposure,
                    feature_names,
                    class_labels,
                    monotone,
                    cat_x,
                    es_holdout,
                    bag_groups,
                )
            })
            .map_err(py_err)?;
        Ok(PyModel {
            model: Arc::new(model),
        })
    }

    /// Fit a native-softmax (multinomial) multiclass model. `y` holds integer class labels in
    /// `0..n_classes`; `class_labels` are the human-readable labels. Returns a `_MultiClassModel`.
    /// `self.objective` is ignored (softmax is implied); exposure is not applicable and is not
    /// accepted here.
    #[pyo3(signature = (x, y, n_classes, class_labels, weight=None, feature_names=None, monotone=None, cat_x=None, es_holdout=None, bag_groups=None))]
    #[allow(clippy::too_many_arguments)]
    fn fit_multiclass(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        y: PyReadonlyArray1<'_, f32>,
        n_classes: usize,
        class_labels: Vec<String>,
        weight: Option<PyReadonlyArray1<'_, f32>>,
        feature_names: Option<Vec<String>>,
        monotone: Option<Vec<i8>>,
        cat_x: Option<Vec<Vec<String>>>,
        es_holdout: Option<Vec<bool>>,
        bag_groups: Option<PyReadonlyArray1<'_, u32>>,
    ) -> PyResult<PyMultiClassModel> {
        cap_global_pool_once(self.n_jobs);
        let bag_groups = bag_groups
            .map(|g| array1_to_vec_u32(g, "bag_groups"))
            .transpose()?;
        let columns = raw_columns_from_array(x)?;
        if columns.len() + cat_x.as_ref().map_or(0, Vec::len) == 0 {
            return Err(py_err(PbError::InvalidInput {
                what: "x must contain at least one feature (numeric or categorical)".into(),
            }));
        }
        let y = array1_to_vec(y, "y")?;
        let weight = weight.map(|w| array1_to_vec(w, "weight")).transpose()?;
        let state = self.clone();
        let model = py
            .detach(move || {
                fit_multiclass_owned_bagged(
                    state,
                    columns,
                    y,
                    n_classes,
                    class_labels,
                    weight,
                    feature_names,
                    monotone,
                    cat_x,
                    es_holdout,
                    bag_groups,
                )
            })
            .map_err(py_err)?;
        Ok(PyMultiClassModel {
            model: Arc::new(model),
        })
    }

    /// Fit native-softmax on the complement of `sel_rows`, then prune the per-class banks with a
    /// shared keep-set selected on those held-out rows. The full design crosses the Python boundary
    /// once; row gathering, fitting, full-data serving, and pruning run on one bounded Rayon pool.
    #[pyo3(signature = (x, y, n_classes, class_labels, sel_rows, weight=None, feature_names=None, monotone=None, cat_x=None, ref_measure=None, laplace=1.0, measure_floor=0.001, se_rule=0.0, lambda_boxes=0.0, n_folds=5, es_holdout=None, deploy_es_holdout=None, prune_guard=false, prune_guard_tol=0.05, prune_guard_min_rows=500, guard_oob_honest=true, sel_bags=1, box_budget=0, lambda_tables=0.0, table_budget=0, table_min_arity=3, fold_of=None, k_folds=0, fold_es_holdout=None, fold_es_patience=None, min_stability=0.5, min_mean_gain=0.0, drop_z=None, keep_budget=0, guard_z=0.0, guard_floor=0.0, bag_groups=None, ranked_path=false, path_steps=32, path_fraction=1.0, band_tolerance=None, band_deviance_cap=0.001, path_tolerance=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn fit_multiclass_pruned(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        y: PyReadonlyArray1<'_, f32>,
        n_classes: usize,
        class_labels: Vec<String>,
        sel_rows: Vec<usize>,
        weight: Option<PyReadonlyArray1<'_, f32>>,
        feature_names: Option<Vec<String>>,
        monotone: Option<Vec<i8>>,
        cat_x: Option<Vec<Vec<String>>>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
        se_rule: f64,
        lambda_boxes: f64,
        n_folds: usize,
        // Honest ES holdout for the SELECTION fit, expressed over the COMPLEMENT of `sel_rows`
        // in ascending original-row order (the gather order used below).
        es_holdout: Option<Vec<bool>>,
        // Honest ES holdout for the full-data DEPLOY fit, expressed over ALL rows.
        deploy_es_holdout: Option<Vec<bool>>,
        // Set-level no-harm guard on the DEPLOYED per-class banks, with out-of-bag evidence
        // (see `prune::multiclass_prune_guard`). OFF by default: unlike the single-output path
        // this one is new, so a K>=3 fit that does not ask for it stays byte-identical.
        prune_guard: bool,
        prune_guard_tol: f64,
        prune_guard_min_rows: usize,
        // Whether OUT-OF-BAG rows are honest for this fit — true iff no group can straddle the
        // in-bag/out-of-bag boundary (ungrouped, or an all-singleton grouping). Only the caller
        // knows the grouping, so it decides; false routes the guard to the shared carve.
        guard_oob_honest: bool,
        // Bag count of the SELECTION fit (the K>=3 analogue of the single-output path's
        // `prune_fold_bags`). 1 = the shipped single, unbagged selection fit, byte-identical to
        // every earlier release. See `fit_multiclass_pruned_owned` for why the knob exists.
        sel_bags: u16,
        // Deployed-box budget for the shared keep-set (`0` disables; see `apply_box_budget`).
        // The evidence it spends on is this path's own `PruneReport.table_scores`, so there is
        // nothing for the caller to supply.
        box_budget: usize,
        // Selection-time price per kept table of arity >= `table_min_arity` (`0.0` disables).
        lambda_tables: f64,
        // Deployed >=`table_min_arity`-way TABLE cap for the shared keep-set (`0` disables; see
        // `apply_table_budget`). Spends the same evidence the box budget does.
        table_budget: usize,
        // Lowest arity the table price and the table budget count. Ralph's explainability bar
        // is 3: mains and pairs are what a filing reads.
        table_min_arity: u8,
        // CV selection (2026-09-07 default): fold id per row in `0..k_folds`, or None for the
        // legacy single-split walk on `sel_rows`.
        fold_of: Option<PyReadonlyArray1<'_, i64>>,
        k_folds: usize,
        // Shared ES holdout for the FOLD fits, over all rows (grouped panels); None = internal.
        fold_es_holdout: Option<Vec<bool>>,
        // Early-stopping patience override for the fold fits (the single-output path's 250).
        fold_es_patience: Option<u32>,
        // `aggregate_prune_selection` knobs — the single-output path's own.
        min_stability: f64,
        min_mean_gain: f64,
        drop_z: Option<f64>,
        keep_budget: usize,
        // SE-aware guard bar: `min(tol*dev_full, max(guard_z*SE, guard_floor*(null-full)))`.
        guard_z: f64,
        guard_floor: f64,
        bag_groups: Option<PyReadonlyArray1<'_, u32>>,
        // Ranked-path selector (see `multiclass_ranked_path`): replaces the fold CV and the guard.
        ranked_path: bool,
        path_steps: usize,
        path_fraction: f64,
        band_tolerance: Option<f64>,
        band_deviance_cap: f64,
        path_tolerance: f64,
    ) -> PyResult<(PyMultiClassTableModel, String)> {
        cap_global_pool_once(self.n_jobs);
        let bag_groups = bag_groups
            .map(|g| array1_to_vec_u32(g, "bag_groups"))
            .transpose()?;
        let fold_of: Option<Vec<i64>> = fold_of
            .map(|f| {
                f.as_slice().map(<[i64]>::to_vec).map_err(|e| {
                    PyValueError::new_err(format!("fold_of must be contiguous int64: {e}"))
                })
            })
            .transpose()?;
        let columns = raw_columns_from_array(x)?;
        if columns.len() + cat_x.as_ref().map_or(0, Vec::len) == 0 {
            return Err(py_err(PbError::InvalidInput {
                what: "x must contain at least one feature (numeric or categorical)".into(),
            }));
        }
        let y = array1_to_vec(y, "y")?;
        let weight = weight.map(|w| array1_to_vec(w, "weight")).transpose()?;
        let state = self.clone();
        let w_measure = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let (model, report_json) = py
            .detach(move || {
                fit_multiclass_pruned_owned(
                    state,
                    columns,
                    y,
                    n_classes,
                    class_labels,
                    sel_rows,
                    weight,
                    feature_names,
                    monotone,
                    cat_x,
                    w_measure,
                    se_rule,
                    lambda_boxes,
                    n_folds,
                    es_holdout,
                    deploy_es_holdout,
                    prune_guard,
                    prune_guard_tol,
                    prune_guard_min_rows,
                    guard_oob_honest,
                    sel_bags,
                    box_budget,
                    lambda_tables,
                    table_budget,
                    table_min_arity,
                    fold_of,
                    k_folds,
                    fold_es_holdout,
                    fold_es_patience,
                    min_stability,
                    min_mean_gain,
                    drop_z,
                    keep_budget,
                    guard_z,
                    guard_floor,
                    bag_groups,
                    ranked_path,
                    path_steps,
                    path_fraction,
                    band_tolerance,
                    band_deviance_cap,
                    path_tolerance,
                )
            })
            .map_err(py_err)?;
        Ok((
            PyMultiClassTableModel {
                model: Arc::new(model),
                serve: Default::default(),
            },
            report_json,
        ))
    }

    /// Deploy-path prune-fold phase (v2 pruning step 2), in ONE call on ONE rayon pool.
    ///
    /// For each of `k_folds` honest-CV folds, fit a single-bag model on the K-1 train rows and
    /// score its contribution-ranked prune path on the held-out fold, returning that fold's
    /// `PruneReport` JSON. The Python caller aggregates selected supports and held-out contribution
    /// diagnostics across folds, then applies one stable keep-set to the full-data fit. The K folds
    /// are independent computations run as parallel tasks inside one `n_jobs`-sized pool — each
    /// fold's fit/serve/prune runs its own par_iters on that ambient pool (no nested pools), so total
    /// live worker threads never exceed `n_jobs`. Fold models are seeded by `state.seed` on data
    /// carved by `fold_of`, every engine reduction is order-preserving, and the full design is
    /// marshalled ONCE here and sliced per fold instead of re-marshalled.
    ///
    /// `x`/`y`/`weight`/`exposure` are the FULL design (weight/exposure `None` ⇒ unit-weight fit,
    /// matching the loop). `fold_of[i]` is row `i`'s fold in `0..k_folds` (the caller's RNG draws
    /// it). The held-out prune scoring's `w_full`/`offset_full` are derived internally (`weight`
    /// or ones; `ln(exposure)`), exactly as the loop did. Returns one JSON per fold in fold order;
    /// an empty string marks a fold with an empty train or held set (the caller skips it, as the
    /// loop's `continue` did).
    #[pyo3(signature = (x, y, fold_of, k_folds, weight=None, exposure=None, feature_names=None, class_labels=None, monotone=None, cat_x=None, ref_measure=None, laplace=1.0, measure_floor=0.001, se_rule=0.0, lambda_boxes=0.0, n_folds=1, reanchor=true, es_holdout=None, lambda_tables=0.0, table_min_arity=3))]
    #[allow(clippy::too_many_arguments)]
    fn fit_prune_folds(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        y: PyReadonlyArray1<'_, f32>,
        fold_of: PyReadonlyArray1<'_, i64>,
        k_folds: usize,
        weight: Option<PyReadonlyArray1<'_, f32>>,
        exposure: Option<PyReadonlyArray1<'_, f32>>,
        feature_names: Option<Vec<String>>,
        class_labels: Option<Vec<String>>,
        monotone: Option<Vec<i8>>,
        cat_x: Option<Vec<Vec<String>>>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
        se_rule: f64,
        lambda_boxes: f64,
        n_folds: usize,
        reanchor: bool,
        // Caller-supplied honest ES holdout over ALL rows (e.g. a GROUP-aware carve for panel
        // data), gathered per fold to that fold's train subset — see `fit_prune_reports_owned`.
        es_holdout: Option<Vec<bool>>,
        // Selection-time price per kept table of arity >= `table_min_arity` (`0.0` disables).
        lambda_tables: f64,
        // Lowest arity the table price counts. Ralph's explainability bar is 3.
        table_min_arity: u8,
    ) -> PyResult<Vec<String>> {
        let columns = raw_columns_from_array(x)?;
        let y_vec = array1_to_vec(y, "y")?;
        let fold_of: Vec<i64> = fold_of
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("fold_of must be contiguous int64: {e}")))?
            .to_vec();
        if fold_of.len() != y_vec.len() {
            return Err(py_err(PbError::ShapeMismatch {
                what: format!("fold_of len {} != n_rows {}", fold_of.len(), y_vec.len()),
            }));
        }
        // An out-of-range fold id (e.g. a sentinel -1, or an off-by-one k) matches no `k` in
        // `fold_task`'s `f == k_i` test, so that row would train on EVERY fold and be held out
        // by NONE — silently biasing the CV rather than raising, since the row simply never
        // participates in held-out scoring instead of erroring.
        if fold_of.iter().any(|&f| f < 0 || f >= k_folds as i64) {
            return Err(py_err(PbError::InvalidInput {
                what: format!(
                    "fold_of values must be in 0..k_folds ({k_folds}); an out-of-range value \
                     matches no fold id and would silently train on every fold, held out by none"
                ),
            }));
        }
        if let Some(mask) = es_holdout.as_deref() {
            if mask.len() != y_vec.len() {
                return Err(py_err(PbError::ShapeMismatch {
                    what: format!("es_holdout len {} != n_rows {}", mask.len(), y_vec.len()),
                }));
            }
        }
        let weight_vec = weight.map(|w| array1_to_vec(w, "weight")).transpose()?;
        let exposure_vec = exposure.map(|e| array1_to_vec(e, "exposure")).transpose()?;
        let n_numeric = columns.len();
        let n_cat = cat_x.as_ref().map_or(0, Vec::len);
        if n_numeric + n_cat == 0 {
            return Err(py_err(PbError::InvalidInput {
                what: "x must contain at least one feature (numeric or categorical)".into(),
            }));
        }
        let monotone_map =
            build_monotone_map(monotone.as_deref(), n_numeric + n_cat).map_err(py_err)?;
        // LOO stays gated (own-target mean-shift pathology). Ordered is allowed here: fold fits
        // route through the same `fit_model_ambient` as a top-level fit, so whenever
        // validation_fraction is set they DO carve the one-honest-holdout split (blinding the
        // encoders to it) exactly like the deploy fit — not a weaker per-row-only exclusion.
        // Fold fits steer support selection, not deployed values; the deploy fit itself is
        // fully blinded regardless (design/ordered-ts-early-stopping.md).
        if cat_x.is_some()
            && self.config.validation_fraction.is_some()
            && matches!(self.cat_config.leakage, LeakageScheme::LeaveOneOut)
        {
            return Err(py_err(PbError::InvalidConfig {
                what: "validation_fraction with native categoricals is not supported under \
                       cat_leakage='loo' (own-target mean-shift pathology); use 'kfold' \
                       (default) or 'ordered', or an external validation split"
                    .into(),
            }));
        }
        let w_measure = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let state = self.clone();
        let reports = py
            .detach(move || {
                fit_prune_reports_owned(
                    state,
                    columns,
                    y_vec,
                    fold_of,
                    k_folds,
                    weight_vec,
                    exposure_vec,
                    feature_names,
                    class_labels,
                    monotone_map,
                    cat_x,
                    w_measure,
                    se_rule,
                    lambda_boxes,
                    lambda_tables,
                    table_min_arity,
                    n_folds,
                    reanchor,
                    es_holdout,
                    // `fit_prune_folds` returns the RAW per-fold reports for a Python-side
                    // aggregator that has no candidate bank to pin against; fidelity is a
                    // `fit_prune_selection`-only feature.
                    None,
                )
            })
            .map_err(py_err)?;
        let mut out = Vec::with_capacity(reports.len());
        for report in reports {
            out.push(match report {
                Some(report) => serde_json::to_string(&report).map_err(|e| {
                    py_err(PbError::Serialization(format!(
                        "could not serialize PruneReport: {e}"
                    )))
                })?,
                None => String::new(),
            });
        }
        Ok(out)
    }

    /// Like [`PyBooster::fit_prune_folds`], but aggregate the contribution/stability reports in
    /// Rust and return the final keep-set plus report JSON. This avoids per-fold JSON crossing into
    /// Python on the scalar sklearn prune path.
    #[pyo3(signature = (x, y, fold_of, k_folds, full_supports, weight=None, exposure=None, feature_names=None, class_labels=None, monotone=None, cat_x=None, ref_measure=None, laplace=1.0, measure_floor=0.001, se_rule=0.0, lambda_boxes=0.0, n_folds=1, reanchor=true, min_stability=0.5, min_mean_gain=0.0, es_holdout=None, drop_z=None, keep_budget=0, lambda_tables=0.0, table_min_arity=3, fold_fidelity=false))]
    #[allow(clippy::too_many_arguments)]
    fn fit_prune_selection(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        y: PyReadonlyArray1<'_, f32>,
        fold_of: PyReadonlyArray1<'_, i64>,
        k_folds: usize,
        full_supports: Vec<Vec<u32>>,
        weight: Option<PyReadonlyArray1<'_, f32>>,
        exposure: Option<PyReadonlyArray1<'_, f32>>,
        feature_names: Option<Vec<String>>,
        class_labels: Option<Vec<String>>,
        monotone: Option<Vec<i8>>,
        cat_x: Option<Vec<Vec<String>>>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
        se_rule: f64,
        lambda_boxes: f64,
        n_folds: usize,
        reanchor: bool,
        min_stability: f64,
        // Minimum mean held-out drop-gain for a legacy keep (`0.0` = the historical rule).
        min_mean_gain: f64,
        // Caller-supplied honest ES holdout over ALL rows (e.g. a GROUP-aware carve for panel
        // data), gathered per fold to that fold's train subset — see `fit_prune_reports_owned`.
        es_holdout: Option<Vec<bool>>,
        // av37 evidence gate — `None` reproduces the av36 selection byte-for-byte.
        drop_z: Option<f64>,
        keep_budget: usize,
        // Selection-time price per kept table of arity >= `table_min_arity` (`0.0` disables).
        lambda_tables: f64,
        // Lowest arity the table price counts. Ralph's explainability bar is 3.
        table_min_arity: u8,
        // PINNED-BANK FOLD FIDELITY. `false` (the default) is byte-identical to before this
        // parameter existed. `true` pins each fold bank to `full_supports` — see
        // `prune::FoldFidelity` and `_PRUNE_FOLD_FIDELITY_DEFAULT` in the sklearn wrapper.
        fold_fidelity: bool,
    ) -> PyResult<(Vec<Vec<u32>>, String)> {
        let columns = raw_columns_from_array(x)?;
        let y_vec = array1_to_vec(y, "y")?;
        let fold_of: Vec<i64> = fold_of
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("fold_of must be contiguous int64: {e}")))?
            .to_vec();
        if fold_of.len() != y_vec.len() {
            return Err(py_err(PbError::ShapeMismatch {
                what: format!("fold_of len {} != n_rows {}", fold_of.len(), y_vec.len()),
            }));
        }
        // An out-of-range fold id (e.g. a sentinel -1, or an off-by-one k) matches no `k` in
        // `fold_task`'s `f == k_i` test, so that row would train on EVERY fold and be held out
        // by NONE — silently biasing the CV rather than raising, since the row simply never
        // participates in held-out scoring instead of erroring.
        if fold_of.iter().any(|&f| f < 0 || f >= k_folds as i64) {
            return Err(py_err(PbError::InvalidInput {
                what: format!(
                    "fold_of values must be in 0..k_folds ({k_folds}); an out-of-range value \
                     matches no fold id and would silently train on every fold, held out by none"
                ),
            }));
        }
        if let Some(mask) = es_holdout.as_deref() {
            if mask.len() != y_vec.len() {
                return Err(py_err(PbError::ShapeMismatch {
                    what: format!("es_holdout len {} != n_rows {}", mask.len(), y_vec.len()),
                }));
            }
        }
        let weight_vec = weight.map(|w| array1_to_vec(w, "weight")).transpose()?;
        let exposure_vec = exposure.map(|e| array1_to_vec(e, "exposure")).transpose()?;
        let n_numeric = columns.len();
        let n_cat = cat_x.as_ref().map_or(0, Vec::len);
        if n_numeric + n_cat == 0 {
            return Err(py_err(PbError::InvalidInput {
                what: "x must contain at least one feature (numeric or categorical)".into(),
            }));
        }
        let monotone_map =
            build_monotone_map(monotone.as_deref(), n_numeric + n_cat).map_err(py_err)?;
        if cat_x.is_some()
            && self.config.validation_fraction.is_some()
            && matches!(self.cat_config.leakage, LeakageScheme::LeaveOneOut)
        {
            return Err(py_err(PbError::InvalidConfig {
                what: "validation_fraction with native categoricals is not supported under \
                       cat_leakage='loo' (own-target mean-shift pathology); use 'kfold' \
                       (default) or 'ordered', or an external validation split"
                    .into(),
            }));
        }
        let w_measure = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let state = self.clone();
        py.detach(move || {
            let reports = fit_prune_reports_owned(
                state,
                columns,
                y_vec,
                fold_of,
                k_folds,
                weight_vec,
                exposure_vec,
                feature_names,
                class_labels,
                monotone_map,
                cat_x,
                w_measure,
                se_rule,
                lambda_boxes,
                lambda_tables,
                table_min_arity,
                n_folds,
                reanchor,
                es_holdout,
                fold_fidelity.then(|| full_supports.clone()),
            )?;
            aggregate_prune_selection(
                full_supports,
                &reports,
                min_stability,
                min_mean_gain,
                drop_z,
                keep_budget,
                fold_fidelity,
            )
        })
        .map_err(py_err)
    }
}

/// Low-level Python model wrapper.
#[pyclass(name = "_Model", skip_from_py_object)]
#[derive(Clone)]
struct PyModel {
    model: Arc<Model>,
}

#[pymethods]
impl PyModel {
    /// The `max_delta_step_gated` rate-collapse detector's report for the fit that produced
    /// this model, or `None` when no gate was armed (or the model was loaded from disk — the
    /// report is runtime-only and never serialized).
    ///
    /// Keys: `engaged`, `engaged_round`, `bags_engaged`, `bags_total`, `min_log_rate_ratio`
    /// (the most extreme `ln(mu / weighted mean rate)` seen — the calibration signal: how far
    /// a SILENT fit was from tripping), `log_threshold`, `capped_step`, `rounds_checked`.
    #[getter]
    fn delta_step_gate<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(r) = self.model.delta_step_gate else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item("engaged", r.engaged)?;
        d.set_item("engaged_round", r.engaged_round)?;
        d.set_item("bags_engaged", r.bags_engaged)?;
        d.set_item("bags_total", r.bags_total)?;
        d.set_item("min_log_rate_ratio", r.min_log_rate_ratio)?;
        d.set_item("log_threshold", r.log_threshold)?;
        d.set_item("capped_step", r.capped_step)?;
        d.set_item("rounds_checked", r.rounds_checked)?;
        Ok(Some(d))
    }

    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        let model = Model::from_json(s).map_err(py_err)?;
        Ok(Self {
            model: Arc::new(model),
        })
    }

    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
        let model = Model::from_bincode(bytes).map_err(py_err)?;
        Ok(Self {
            model: Arc::new(model),
        })
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.model.grids.len()
    }

    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.model.schema.feature_names.clone()
    }

    #[getter]
    fn class_labels(&self) -> Option<Vec<String>> {
        self.model.schema.class_labels.clone()
    }

    /// Number of trees retained in the fitted model. With internal early stopping this is the
    /// best-validation tree count (the model is truncated back to it), summed across bags for an
    /// outer-bag ensemble. Exposed read-only so the deploy/benchmark path can quantify how far a
    /// fit converged relative to its `n_trees` cap and its early-stopping patience.
    #[getter]
    fn n_trees(&self) -> usize {
        self.model.trees.len()
    }

    #[pyo3(signature = (x, out=None, cat_x=None, n_jobs=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        out: Option<Bound<'py, PyArray1<f32>>>,
        cat_x: Option<Vec<Vec<String>>>,
        n_jobs: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let columns = raw_columns_from_array(x)?;
        let model = Arc::clone(&self.model);
        let pred = py
            .detach(move || {
                run_on_pool(n_jobs, || {
                    let binned = serve_binned_for_model(&model, &columns, cat_x.as_deref())?;
                    model.predict_binned(&binned, None)
                })
            })
            .map_err(py_err)?;
        write_or_return_array1(py, pred, out)
    }

    #[pyo3(signature = (x, out=None, cat_x=None, n_jobs=None))]
    fn predict_raw<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        out: Option<Bound<'py, PyArray1<f32>>>,
        cat_x: Option<Vec<Vec<String>>>,
        n_jobs: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let columns = raw_columns_from_array(x)?;
        let model = Arc::clone(&self.model);
        let raw = py
            .detach(move || {
                run_on_pool(n_jobs, || {
                    let binned = serve_binned_for_model(&model, &columns, cat_x.as_deref())?;
                    let mut scores = vec![0.0_f32; binned.n_rows as usize];
                    model.score_trees(&binned, None, &mut scores)?;
                    Ok::<Vec<f32>, PbError>(scores)
                })
            })
            .map_err(py_err)?;
        write_or_return_array1(py, raw, out)
    }

    #[pyo3(signature = (x, cat_x=None, n_jobs=None))]
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        cat_x: Option<Vec<Vec<String>>>,
        n_jobs: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if self.model.link != t_boost_core::loss::Link::Logit {
            return Err(PyTypeError::new_err(
                "predict_proba is only available for logit-link models",
            ));
        }
        let columns = raw_columns_from_array(x)?;
        let model = Arc::clone(&self.model);
        let pred = py
            .detach(move || {
                run_on_pool(n_jobs, || {
                    let binned = serve_binned_for_model(&model, &columns, cat_x.as_deref())?;
                    model.predict_binned(&binned, None)
                })
            })
            .map_err(py_err)?;
        if pred.is_empty() {
            // sklearn contract: predict_proba returns (n_samples, n_classes); for an
            // empty design that is (0, 2), not the (0, 0) that from_vec2 of [] yields.
            return Ok(numpy::PyArray2::<f32>::zeros(py, [0usize, 2], false));
        }
        let mut rows = Vec::with_capacity(pred.len());
        for p1 in pred {
            rows.push(vec![1.0 - p1, p1]);
        }
        PyArray::from_vec2(py, &rows).map_err(|err| {
            InternalError::new_err(format!("could not allocate probability array: {err}"))
        })
    }

    #[pyo3(signature = (x, ref_measure=None, laplace=1.0, measure_floor=0.001, cat_x=None, overflow=None, weight=None, exposure=None))]
    #[allow(clippy::too_many_arguments)]
    fn explain(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
        cat_x: Option<Vec<Vec<String>>>,
        overflow: Option<String>,
        weight: Option<PyReadonlyArray1<'_, f32>>,
        exposure: Option<PyReadonlyArray1<'_, f32>>,
    ) -> PyResult<PyTableBank> {
        let columns = raw_columns_from_array(x)?;
        let model = Arc::clone(&self.model);
        let w = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let budget = parse_table_budget(overflow.as_deref()).map_err(py_err)?;
        let combined_weight = combined_explain_weight(weight, exposure)?;
        let bank = py
            .detach(move || {
                let binned = serve_binned_for_model(&model, &columns, cat_x.as_deref())?;
                let serve = t_boost_core::data::ServeBinnedMatrix(binned);
                match combined_weight {
                    Some(mass) => model.explain_with_budget_weighted(&serve, w, budget, &mass),
                    None => model.explain_with_budget(&serve, w, budget),
                }
            })
            .map_err(py_err)?;
        Ok(PyTableBank { bank })
    }

    #[pyo3(signature = (x, ref_measure=None, laplace=1.0, measure_floor=0.001, basis_json=None, cat_x=None, overflow=None, weight=None, exposure=None))]
    #[allow(clippy::too_many_arguments)]
    fn tables(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
        basis_json: Option<&str>,
        cat_x: Option<Vec<Vec<String>>>,
        overflow: Option<String>,
        weight: Option<PyReadonlyArray1<'_, f32>>,
        exposure: Option<PyReadonlyArray1<'_, f32>>,
    ) -> PyResult<String> {
        let columns = raw_columns_from_array(x)?;
        let model = Arc::clone(&self.model);
        let w = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let basis = parse_rating_basis(basis_json)?;
        let budget = parse_table_budget(overflow.as_deref()).map_err(py_err)?;
        let combined_weight = combined_explain_weight(weight, exposure)?;
        py.detach(move || {
            let binned = serve_binned_for_model(&model, &columns, cat_x.as_deref())?;
            let serve = t_boost_core::data::ServeBinnedMatrix(binned);
            // The joint ledger is export-only: purify under the exposure measure first, then
            // re-express orders one and two under the joint support (`joint::rejoint`).
            let (w, joint) = if w == RefMeasure::Joint {
                (
                    RefMeasure::ExposureMarginals {
                        floor: measure_floor,
                    },
                    true,
                )
            } else {
                (w, false)
            };
            let mut bank = match &combined_weight {
                Some(mass) => model.explain_with_budget_weighted(&serve, w, budget, mass)?,
                None => model.explain_with_budget(&serve, w, budget)?,
            };
            if joint {
                bank = t_boost_core::joint::rejoint(
                    &bank,
                    &t_boost_core::joint::JointOptions::default(),
                )?;
                bank.measure_joint_variance(
                    &model.schema.cat_encoders,
                    &serve.0,
                    combined_weight.as_deref(),
                )?;
            }
            let export = bank.to_rating_export(
                model.link,
                &model.mode,
                &model.schema,
                &model.provenance,
                &model.schema.cat_encoders,
                basis.as_ref(),
            )?;
            serde_json::to_string_pretty(&export).map_err(|err| {
                PbError::Serialization(format!("could not serialize RatingExport JSON: {err}"))
            })
        })
        .map_err(py_err)
    }

    /// Purified variance of every table in the full (unpruned) bank, as `[(support, variance)]`,
    /// built without the verification gates `explain` runs — the ranked-path selector's ranking.
    #[pyo3(signature = (x, weight, ref_measure=None, laplace=1.0, measure_floor=0.001, cat_x=None, exposure=None, n_jobs=None))]
    #[allow(clippy::too_many_arguments)] // JUSTIFIED: pyo3 surface mirrors apply_keepset's keyword signature (§12.2).
    fn table_variances(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        weight: PyReadonlyArray1<'_, f32>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
        cat_x: Option<Vec<Vec<String>>>,
        exposure: Option<PyReadonlyArray1<'_, f32>>,
        n_jobs: Option<usize>,
    ) -> PyResult<Vec<(Vec<u32>, f64)>> {
        let columns = raw_columns_from_array(x)?;
        let w_vec = weight
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("weight must be contiguous: {e}")))?
            .to_vec();
        let offset_vec: Option<Vec<f32>> = match exposure {
            Some(e) => Some(
                e.as_slice()
                    .map_err(|err| {
                        PyValueError::new_err(format!("exposure must be contiguous: {err}"))
                    })?
                    .iter()
                    .map(|&v| v.ln())
                    .collect(),
            ),
            None => None,
        };
        let model = Arc::clone(&self.model);
        let w_measure = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let vars = py
            .detach(move || {
                let binned = serve_binned_for_model(&model, &columns, cat_x.as_deref())?;
                let serve = ServeBinnedMatrix(binned);
                let run = || {
                    t_boost_core::prune::full_bank_table_variances(
                        &model,
                        &serve,
                        &w_vec,
                        offset_vec.as_deref(),
                        w_measure,
                    )
                };
                require_nonzero_n_jobs(n_jobs)?;
                match n_jobs {
                    Some(nj) => rayon::ThreadPoolBuilder::new()
                        .num_threads(nj)
                        .build()
                        .map_err(|err| PbError::InvalidConfig {
                            what: format!("could not build rayon pool: {err}"),
                        })?
                        .install(run),
                    None => run(),
                }
            })
            .map_err(py_err)?;
        Ok(vars
            .into_iter()
            .map(|(u, v)| (u.0.iter().map(|f| f.0).collect(), v))
            .collect())
    }

    #[pyo3(signature = (x, ref_measure=None, laplace=1.0, measure_floor=0.001, cat_x=None, overflow=None))]
    #[allow(clippy::too_many_arguments)]
    fn table_supports(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
        cat_x: Option<Vec<Vec<String>>>,
        overflow: Option<String>,
    ) -> PyResult<Vec<Vec<u32>>> {
        let columns = raw_columns_from_array(x)?;
        let model = Arc::clone(&self.model);
        let w = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let budget = parse_table_budget(overflow.as_deref()).map_err(py_err)?;
        py.detach(move || {
            let binned = serve_binned_for_model(&model, &columns, cat_x.as_deref())?;
            let supports =
                model.table_supports(&t_boost_core::data::ServeBinnedMatrix(binned), w, budget)?;
            Ok::<Vec<Vec<u32>>, PbError>(
                supports
                    .into_iter()
                    .map(|u| u.0.into_iter().map(|f| f.0).collect())
                    .collect(),
            )
        })
        .map_err(py_err)
    }

    fn to_json(&self) -> PyResult<String> {
        self.model.to_json().map_err(py_err)
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.model.to_bincode().map_err(py_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Prune this model into a tables-only [`PyTableModel`] by dropping fANOVA tables that don't
    /// improve held-out deviance. `sel_rows` index the rows this model did NOT train on (the caller
    /// carves the train/select split). Returns `(model, prune_report_json)`.
    #[pyo3(signature = (x, y, weight, sel_rows, ref_measure=None, laplace=1.0, measure_floor=0.001, se_rule=1.0, lambda_boxes=0.0, n_folds=5, reanchor=true, exposure=None, cat_x=None))]
    #[allow(clippy::too_many_arguments)]
    fn prune_to_tables(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        y: PyReadonlyArray1<'_, f32>,
        weight: PyReadonlyArray1<'_, f32>,
        sel_rows: Vec<usize>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
        se_rule: f64,
        lambda_boxes: f64,
        n_folds: usize,
        reanchor: bool,
        exposure: Option<PyReadonlyArray1<'_, f32>>,
        cat_x: Option<Vec<Vec<String>>>,
    ) -> PyResult<(PyTableModel, String)> {
        let columns = raw_columns_from_array(x)?;
        let y_vec = y
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("y must be contiguous: {e}")))?
            .to_vec();
        let w_vec = weight
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("weight must be contiguous: {e}")))?
            .to_vec();
        // Exposure enters the log-link selection/reanchor as a log-offset added to the raw score.
        let offset_vec: Option<Vec<f32>> = match exposure {
            Some(e) => Some(
                e.as_slice()
                    .map_err(|err| {
                        PyValueError::new_err(format!("exposure must be contiguous: {err}"))
                    })?
                    .iter()
                    .map(|&v| v.ln())
                    .collect(),
            ),
            None => None,
        };
        let model = Arc::clone(&self.model);
        let w_measure = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let (tm, report_json) = py
            .detach(move || {
                let binned = serve_binned_for_model(&model, &columns, cat_x.as_deref())?;
                let serve = ServeBinnedMatrix(binned);
                let loss = loss_choice_from_tag(&model.schema.objective)?;
                let cfg = PruneConfig {
                    se_rule,
                    lambda_boxes,
                    lambda_tables: 0.0,
                    table_price_min_arity: DEFAULT_TABLE_MIN_ARITY,
                };
                let (tm, report) = prune_model_to_tables(
                    &model,
                    &serve,
                    &y_vec,
                    &w_vec,
                    offset_vec.as_deref(),
                    loss.as_loss(),
                    w_measure,
                    reanchor,
                    &sel_rows,
                    n_folds,
                    &cfg,
                )?;
                let report_json = serde_json::to_string(&report).map_err(|e| {
                    PbError::Serialization(format!("could not serialize PruneReport: {e}"))
                })?;
                Ok::<(TableModel, String), PbError>((tm, report_json))
            })
            .map_err(py_err)?;
        Ok((
            PyTableModel {
                model: Arc::new(tm),
                serve: Default::default(),
            },
            report_json,
        ))
    }

    /// Apply an already-selected keep-set (list of raw-feature-id lists) to THIS model, producing a
    /// tables-only [`PyTableModel`] — no selection, just retain/re-anchor/optional rebalance.
    #[pyo3(signature = (x, y, weight, keep, ref_measure=None, laplace=1.0, measure_floor=0.001, reanchor=true, rebalance=false, n_jobs=None, exposure=None, cat_x=None))]
    #[allow(clippy::too_many_arguments)]
    fn apply_keepset(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        y: PyReadonlyArray1<'_, f32>,
        weight: PyReadonlyArray1<'_, f32>,
        keep: Vec<Vec<u32>>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
        reanchor: bool,
        rebalance: bool,
        n_jobs: Option<usize>,
        exposure: Option<PyReadonlyArray1<'_, f32>>,
        cat_x: Option<Vec<Vec<String>>>,
    ) -> PyResult<PyTableModel> {
        let columns = raw_columns_from_array(x)?;
        let y_vec = y
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("y must be contiguous: {e}")))?
            .to_vec();
        let w_vec = weight
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("weight must be contiguous: {e}")))?
            .to_vec();
        let offset_vec: Option<Vec<f32>> = match exposure {
            Some(e) => Some(
                e.as_slice()
                    .map_err(|err| {
                        PyValueError::new_err(format!("exposure must be contiguous: {err}"))
                    })?
                    .iter()
                    .map(|&v| v.ln())
                    .collect(),
            ),
            None => None,
        };
        let keepset: Vec<FeatureSet> = keep
            .iter()
            .map(|ids| feature_set_from_raw_ids(ids))
            .collect();
        let model = Arc::clone(&self.model);
        let w_measure = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let tm = py
            .detach(move || {
                let binned = serve_binned_for_model(&model, &columns, cat_x.as_deref())?;
                let serve = ServeBinnedMatrix(binned);
                let loss = loss_choice_from_tag(&model.schema.objective)?;
                let run = || {
                    prune_model_to_keepset(
                        &model,
                        &serve,
                        &y_vec,
                        &w_vec,
                        offset_vec.as_deref(),
                        loss.as_loss(),
                        w_measure,
                        reanchor,
                        &keepset,
                        rebalance,
                    )
                };
                require_nonzero_n_jobs(n_jobs)?;
                match n_jobs {
                    Some(nj) => rayon::ThreadPoolBuilder::new()
                        .num_threads(nj)
                        .build()
                        .map_err(|err| PbError::InvalidConfig {
                            what: format!("could not build rayon pool: {err}"),
                        })?
                        .install(run),
                    None => run(),
                }
            })
            .map_err(py_err)?;
        Ok(PyTableModel {
            model: Arc::new(tm),
            serve: Default::default(),
        })
    }

    /// [`apply_keepset`](Self::apply_keepset) with a DEPLOYED-BOX BUDGET, returning
    /// `(table_model, box_budget_report_json)`.
    ///
    /// `box_budget` is the total rank-1 boxes the deployed bank may carry (`0` disables it and
    /// makes this method bit-identical to `apply_keepset`). `box_rank` supplies the held-out
    /// drop-gain evidence the budget spends on — `(raw_ids, mean_gain)` pairs; a support absent
    /// from it scores `0.0`. Dense tables cost no boxes and are never dropped.
    ///
    /// Kept SEPARATE from `apply_keepset` rather than folded into it so the unbudgeted call —
    /// the one every `max_depth <= 3` fit and every budget-off fit makes — keeps its exact
    /// signature and return type.
    #[pyo3(signature = (x, y, weight, keep, box_budget, box_rank=None, ref_measure=None, laplace=1.0, measure_floor=0.001, reanchor=true, rebalance=false, n_jobs=None, exposure=None, cat_x=None, table_budget=0, table_min_arity=3))]
    #[allow(clippy::too_many_arguments)] // JUSTIFIED: pyo3 surface mirrors apply_keepset's keyword signature (§12.2) plus the budget args.
    fn apply_keepset_budgeted(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        y: PyReadonlyArray1<'_, f32>,
        weight: PyReadonlyArray1<'_, f32>,
        keep: Vec<Vec<u32>>,
        box_budget: usize,
        box_rank: Option<Vec<(Vec<u32>, f64)>>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
        reanchor: bool,
        rebalance: bool,
        n_jobs: Option<usize>,
        exposure: Option<PyReadonlyArray1<'_, f32>>,
        cat_x: Option<Vec<Vec<String>>>,
        table_budget: usize,
        table_min_arity: u8,
    ) -> PyResult<(PyTableModel, String, String)> {
        let columns = raw_columns_from_array(x)?;
        let y_vec = y
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("y must be contiguous: {e}")))?
            .to_vec();
        let w_vec = weight
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("weight must be contiguous: {e}")))?
            .to_vec();
        let offset_vec: Option<Vec<f32>> = match exposure {
            Some(e) => Some(
                e.as_slice()
                    .map_err(|err| {
                        PyValueError::new_err(format!("exposure must be contiguous: {err}"))
                    })?
                    .iter()
                    .map(|&v| v.ln())
                    .collect(),
            ),
            None => None,
        };
        let keepset: Vec<FeatureSet> = keep
            .iter()
            .map(|ids| feature_set_from_raw_ids(ids))
            .collect();
        let evidence: BTreeMap<FeatureSet, f64> = box_rank
            .unwrap_or_default()
            .into_iter()
            .map(|(ids, g)| (feature_set_from_raw_ids(&ids), g))
            .collect();
        let model = Arc::clone(&self.model);
        let w_measure = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let (tm, report, table_report) = py
            .detach(move || {
                let binned = serve_binned_for_model(&model, &columns, cat_x.as_deref())?;
                let serve = ServeBinnedMatrix(binned);
                let loss = loss_choice_from_tag(&model.schema.objective)?;
                let run = || {
                    t_boost_core::prune::prune_model_to_keepset_budgeted(
                        &model,
                        &serve,
                        &y_vec,
                        &w_vec,
                        offset_vec.as_deref(),
                        loss.as_loss(),
                        w_measure,
                        reanchor,
                        &keepset,
                        rebalance,
                        box_budget,
                        table_budget,
                        table_min_arity,
                        &evidence,
                    )
                };
                require_nonzero_n_jobs(n_jobs)?;
                match n_jobs {
                    Some(nj) => rayon::ThreadPoolBuilder::new()
                        .num_threads(nj)
                        .build()
                        .map_err(|err| PbError::InvalidConfig {
                            what: format!("could not build rayon pool: {err}"),
                        })?
                        .install(run),
                    None => run(),
                }
            })
            .map_err(py_err)?;
        let report_json = serde_json::to_string(&report)
            .map_err(|err| py_err(PbError::Serialization(format!("box budget report: {err}"))))?;
        let table_report_json = serde_json::to_string(&table_report).map_err(|err| {
            py_err(PbError::Serialization(format!(
                "table budget report: {err}"
            )))
        })?;
        Ok((
            PyTableModel {
                model: Arc::new(tm),
                serve: Default::default(),
            },
            report_json,
            table_report_json,
        ))
    }

    /// Per-bag purified banks restricted to `keep`, as JSON strings (one per bag, in bag
    /// order) — the honest replicate values behind the outer-bag soup, for per-cell
    /// uncertainty (SE bands / credibility). Only available on a freshly-fitted bagged
    /// model (the bag partition is runtime-only). A bag that realizes none of a kept
    /// support omits that table: treat absence as an all-zero replicate.
    #[pyo3(signature = (x, keep, ref_measure=None, laplace=1.0, measure_floor=0.001, cat_x=None, n_jobs=None, weight=None, exposure=None))]
    #[allow(clippy::too_many_arguments)] // JUSTIFIED: pyo3 surface mirrors apply_keepset's keyword signature (§12.2); py+self push it to 8.
    fn bag_bank_jsons(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        keep: Vec<Vec<u32>>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
        cat_x: Option<Vec<Vec<String>>>,
        n_jobs: Option<usize>,
        weight: Option<PyReadonlyArray1<'_, f32>>,
        exposure: Option<PyReadonlyArray1<'_, f32>>,
    ) -> PyResult<Vec<String>> {
        let columns = raw_columns_from_array(x)?;
        let keepset: Vec<FeatureSet> = keep
            .iter()
            .map(|ids| feature_set_from_raw_ids(ids))
            .collect();
        let model = Arc::clone(&self.model);
        let w_measure = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let mass = measure_mass_py(&w_measure, weight, exposure)?;
        let banks = py
            .detach(move || {
                let binned = serve_binned_for_model(&model, &columns, cat_x.as_deref())?;
                let serve = ServeBinnedMatrix(binned);
                let run =
                    || bag_banks_for_keepset(&model, &serve, w_measure, mass.as_deref(), &keepset);
                require_nonzero_n_jobs(n_jobs)?;
                match n_jobs {
                    Some(nj) => rayon::ThreadPoolBuilder::new()
                        .num_threads(nj)
                        .build()
                        .map_err(|err| PbError::InvalidConfig {
                            what: format!("could not build rayon pool: {err}"),
                        })?
                        .install(run),
                    None => run(),
                }
            })
            .map_err(py_err)?;
        banks
            .iter()
            .map(|b| {
                serde_json::to_string(b).map_err(|e| {
                    py_err(PbError::Serialization(format!(
                        "could not serialize per-bag bank: {e}"
                    )))
                })
            })
            .collect()
    }

    /// Per-bag IN-BAG row membership over the FIT rows as a `(n_bags, n_fit_rows)` bool
    /// array: `mask[b, r]` is true iff fit row `r` was drawn into bag `b`'s training sample.
    /// Its complement is each bag's out-of-bag set. Only available on a freshly-fitted
    /// bagged model (the membership is runtime-only, like the bag partition).
    ///
    /// The draw is deterministic in `(seed, bag index, n_rows, strata)`, so this is the
    /// membership the engine actually drew — not a Python-side replay of the sampler.
    fn bag_in_bag_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<bool>>> {
        let Some(membership) = self.model.bag_in_bag.as_ref() else {
            return Err(py_err(PbError::InvalidInput {
                what: "bag membership needs the outer-bag soup's runtime partition \
                       (single fit or deserialized model?)"
                    .into(),
            }));
        };
        PyArray::from_vec2(py, membership).map_err(|err| {
            InternalError::new_err(format!("could not allocate bag membership array: {err}"))
        })
    }

    /// Whether `bag_oob_group_raw` can run on this model: it needs the outer-bag soup's
    /// runtime bag partition AND membership (so a single fit or a wire round-trip is out) and
    /// no §G1 cell correction. Ask this instead of catching the error, so a genuine failure in
    /// the evidence pass is not misread as "no out-of-bag rows".
    fn bag_oob_available(&self) -> bool {
        bag_oob_evidence_available(&self.model)
    }

    /// Per-row variance of the bag banks' scores restricted to `keep` (the soup's noise; divide by
    /// the bag count for the variance of the soup's mean). Valid on a §G1 cell-corrected soup,
    /// where the shared correction cancels in the spread. Returns `(variance, n_bags)`.
    #[pyo3(signature = (x, keep=None, ref_measure=None, laplace=1.0, measure_floor=0.001, cat_x=None, n_jobs=None, weight=None, exposure=None))]
    #[allow(clippy::too_many_arguments)] // JUSTIFIED: mirrors bag_raw_scores' keyword signature (§12.2).
    fn bag_score_variance<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        keep: Option<Vec<Vec<u32>>>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
        cat_x: Option<Vec<Vec<String>>>,
        n_jobs: Option<usize>,
        weight: Option<PyReadonlyArray1<'_, f32>>,
        exposure: Option<PyReadonlyArray1<'_, f32>>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, usize)> {
        let columns = raw_columns_from_array(x)?;
        let keepset: Option<Vec<FeatureSet>> = keep
            .as_ref()
            .map(|k| k.iter().map(|ids| feature_set_from_raw_ids(ids)).collect());
        let model = Arc::clone(&self.model);
        let n_bags = model.bag_spans.as_ref().map_or(0, Vec::len);
        let w_measure = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let mass = measure_mass_py(&w_measure, weight, exposure)?;
        let var = py
            .detach(move || {
                run_on_pool(n_jobs, || {
                    let binned = serve_binned_for_model(&model, &columns, cat_x.as_deref())?;
                    let rows: Vec<u32> = (0..binned.n_rows).collect();
                    let serve = ServeBinnedMatrix(binned);
                    t_boost_core::prune::bag_score_variance_for_rows(
                        &model,
                        &serve,
                        w_measure,
                        mass.as_deref(),
                        keepset.as_deref(),
                        &rows,
                    )
                })
            })
            .map_err(py_err)?;
        Ok((var.into_pyarray(py), n_bags))
    }

    /// Raw (link-scale) scores of every bag's bank, restricted to `keep`, on `rows` of `x` —
    /// a `(n_bags, len(rows))` float64 array in bag order.
    ///
    /// `keep=None` scores each bag's FULL bank (every realized table). `rows=None` scores
    /// every row of `x`. The mean over ALL bags reproduces the soup's own bank score, so a
    /// mean over just the bags that left a row OUT of bag is an honest estimate of the
    /// deployed score at that row (see `bag_oob_group_raw`).
    #[pyo3(signature = (x, keep=None, rows=None, ref_measure=None, laplace=1.0, measure_floor=0.001, cat_x=None, n_jobs=None, weight=None, exposure=None))]
    #[allow(clippy::too_many_arguments)] // JUSTIFIED: pyo3 surface mirrors bag_bank_jsons' keyword signature (§12.2).
    fn bag_raw_scores<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        keep: Option<Vec<Vec<u32>>>,
        rows: Option<Vec<u32>>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
        cat_x: Option<Vec<Vec<String>>>,
        n_jobs: Option<usize>,
        weight: Option<PyReadonlyArray1<'_, f32>>,
        exposure: Option<PyReadonlyArray1<'_, f32>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let columns = raw_columns_from_array(x)?;
        let keepset: Option<Vec<FeatureSet>> = keep
            .as_ref()
            .map(|k| k.iter().map(|ids| feature_set_from_raw_ids(ids)).collect());
        let model = Arc::clone(&self.model);
        let w_measure = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let mass = measure_mass_py(&w_measure, weight, exposure)?;
        let scores = py
            .detach(move || {
                run_on_pool(n_jobs, || {
                    let binned = serve_binned_for_model(&model, &columns, cat_x.as_deref())?;
                    let all: Vec<u32>;
                    let rows = match rows.as_deref() {
                        Some(r) => r,
                        None => {
                            all = (0..binned.n_rows).collect();
                            &all
                        }
                    };
                    let serve = ServeBinnedMatrix(binned);
                    bag_raw_scores_for_rows(
                        &model,
                        &serve,
                        w_measure,
                        mass.as_deref(),
                        keepset.as_deref(),
                        rows,
                    )
                })
            })
            .map_err(py_err)?;
        PyArray::from_vec2(py, &scores).map_err(|err| {
            InternalError::new_err(format!("could not allocate bag score array: {err}"))
        })
    }

    /// The prune guard's out-of-bag evidence in ONE pass: for each fit row, the sums over the
    /// bags that never trained on it of (a) the full bag bank's raw score, (b) the bag bank
    /// intercepts, and (c) each passed table GROUP's tables alone, plus the count of such bags.
    ///
    /// `groups` is a list of table groups, each a list of supports (raw feature id lists).
    /// Because table scores are additive, any PREFIX union of the groups is
    /// `(f0_sum + Σ_{g ≤ k} group_sums[g]) / counts` — which is what lets the guard evaluate
    /// its whole re-admission ladder (keep-set first, then each doubling chunk in rank order)
    /// without rebuilding a single bank per rung.
    ///
    /// `x` MUST be the fit design — same rows, SAME ORDER — since the recorded membership is
    /// positional. Only the row COUNT is checked (a permuted design cannot be detected and
    /// would silently mis-attribute the evidence). Returns `(full_sum, f0_sum, group_sums,
    /// counts)` with `group_sums` shaped `(len(groups), n_rows)`.
    ///
    /// Check `bag_oob_available()` first: this raises when the model has no bag partition or
    /// membership (a single fit, a deserialized model) or carries a §G1 cell correction, and a
    /// caller that catches the error cannot tell those apart from a real failure.
    ///
    /// `with_full=False` skips scoring each bag's full bank (half the scoring work) and returns
    /// `full_sum` as zeros; the other three arrays are bit-identical either way.
    #[pyo3(signature = (x, groups, ref_measure=None, laplace=1.0, measure_floor=0.001, cat_x=None, n_jobs=None, weight=None, exposure=None, with_full=true))]
    #[allow(clippy::too_many_arguments)] // JUSTIFIED: pyo3 surface mirrors bag_bank_jsons' keyword signature (§12.2).
    fn bag_oob_group_raw<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        groups: Vec<Vec<Vec<u32>>>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
        cat_x: Option<Vec<Vec<String>>>,
        n_jobs: Option<usize>,
        weight: Option<PyReadonlyArray1<'_, f32>>,
        exposure: Option<PyReadonlyArray1<'_, f32>>,
        with_full: bool,
    ) -> PyResult<BagOobGroupArrays<'py>> {
        let columns = raw_columns_from_array(x)?;
        let groups: Vec<Vec<FeatureSet>> = groups
            .iter()
            .map(|g| g.iter().map(|ids| feature_set_from_raw_ids(ids)).collect())
            .collect();
        let model = Arc::clone(&self.model);
        let w_measure = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let mass = measure_mass_py(&w_measure, weight, exposure)?;
        let sums = py
            .detach(move || {
                run_on_pool(n_jobs, || {
                    let binned = serve_binned_for_model(&model, &columns, cat_x.as_deref())?;
                    let serve = ServeBinnedMatrix(binned);
                    bag_oob_group_sums_with(
                        &model,
                        &serve,
                        w_measure,
                        mass.as_deref(),
                        &groups,
                        with_full,
                    )
                })
            })
            .map_err(py_err)?;
        let group_sums = PyArray::from_vec2(py, &sums.group_sums).map_err(|err| {
            InternalError::new_err(format!("could not allocate group-sum array: {err}"))
        })?;
        Ok((
            sums.full_sum.into_pyarray(py),
            sums.f0_sum.into_pyarray(py),
            group_sums,
            sums.counts.into_pyarray(py),
        ))
    }
}

/// `(full_sum, f0_sum, group_sums, counts)` — the return of [`PyModel::bag_oob_group_raw`].
type BagOobGroupArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<u32>>,
);

/// Low-level Python tables-only served model: a pruned/purified fANOVA bank that IS the model,
/// scored by the lossless LUT-sum (no trees). Produced by [`PyModel::prune_to_tables`].
type GraduationTableInput = (usize, Vec<u32>, Vec<usize>, Vec<f64>, Vec<f64>, Vec<bool>);

#[pyclass(name = "_TableModel", skip_from_py_object)]
#[derive(Clone)]
struct PyTableModel {
    model: Arc<TableModel>,
    /// Per-model serve maps, built on the first predict and shared by clones (see `table_serve`).
    serve: Arc<OnceLock<TableServe>>,
}

#[pymethods]
impl PyTableModel {
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        Ok(Self {
            model: Arc::new(TableModel::from_json(s).map_err(py_err)?),
            serve: Default::default(),
        })
    }

    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
        Ok(Self {
            model: Arc::new(TableModel::from_bincode(bytes).map_err(py_err)?),
            serve: Default::default(),
        })
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.model.grids.len()
    }

    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.model.schema.feature_names.clone()
    }

    #[getter]
    fn class_labels(&self) -> Option<Vec<String>> {
        self.model.schema.class_labels.clone()
    }

    /// Number of tables in the deployed bank, dense and factored — what counting the `tables`
    /// and `factored` entries of `to_json()` gives, without serializing the bank.
    fn deployed_table_count(&self) -> usize {
        self.model.bank.tables.len() + self.model.bank.factored.len()
    }

    /// Deployed tables and cells by interaction order: `{"banks": 1, "tables": {order: n},
    /// "cells": {order: n}}` — the census that walking `to_json()`'s `tables` and `factored`
    /// entries gives, without serializing the bank (see [`deployed_census_dict`]).
    fn deployed_census<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        deployed_census_dict(py, &[&self.model.bank])
    }

    /// The feature sets of the tables actually present in the deployed bank (sorted raw ids,
    /// order >= 1). Cheap — no data pass. Lets callers reconcile the prune-selection report's
    /// keep-set with what survived purification into the shipped model.
    fn deployed_supports(&self) -> Vec<Vec<u32>> {
        self.model
            .bank
            .tables
            .iter()
            .map(|t| &t.u)
            .chain(self.model.bank.factored.iter().map(|t| &t.u))
            .filter(|u| u.order() > 0)
            .map(feature_set_ids)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// Dense order-1/2 table payloads needed by the Python LAPACK graduation solver. This avoids
    /// serializing and parsing the complete model merely to inspect a handful of table tensors.
    #[pyo3(signature = (max_cells=2500))]
    fn graduation_tables(&self, max_cells: usize) -> PyResult<Vec<GraduationTableInput>> {
        let mut out = Vec::new();
        for (index, table) in self.model.bank.tables.iter().enumerate() {
            let shape = table.values.shape();
            if table.values.is_sparse()
                || table.support.is_sparse()
                || !(1..=2).contains(&shape.len())
                || table.values.len() > max_cells
            {
                continue;
            }
            if table.support.shape() != shape {
                return Err(py_err(PbError::ShapeMismatch {
                    what: format!("graduation table {index} support shape differs from values"),
                }));
            }
            let mut axis_categorical = Vec::with_capacity(table.axes.len());
            for axis in &table.axes {
                let provenance = self
                    .model
                    .provenance
                    .iter()
                    .find(|provenance| provenance.raw == axis.raw)
                    .ok_or_else(|| {
                        py_err(PbError::Internal {
                            what: format!(
                                "graduation table {index} axis raw {} has no provenance",
                                axis.raw.0
                            ),
                        })
                    })?;
                axis_categorical.push(!matches!(provenance.kind, AxisKind::Numeric));
            }
            out.push((
                index,
                feature_set_ids(&table.u),
                shape,
                table.values.values().into_owned(),
                table.support.values().into_owned(),
                axis_categorical,
            ));
        }
        Ok(out)
    }

    /// Apply dense graduation and optional reference-preserving factored diffusion together.
    #[pyo3(signature = (x, y, weight, updates, alpha, exposure=None, cat_x=None, n_jobs=None, box_budget=0))]
    #[allow(clippy::too_many_arguments)]
    fn apply_high_order_graduation(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        y: PyReadonlyArray1<'_, f32>,
        weight: PyReadonlyArray1<'_, f32>,
        updates: Vec<(usize, Vec<f64>)>,
        alpha: f64,
        exposure: Option<PyReadonlyArray1<'_, f32>>,
        cat_x: Option<Vec<Vec<String>>>,
        n_jobs: Option<usize>,
        box_budget: usize,
    ) -> PyResult<(PyTableModel, bool, String)> {
        let columns = raw_columns_from_array(x)?;
        let y = array1_to_vec(y, "y")?;
        let weight = array1_to_vec(weight, "weight")?;
        let exposure = exposure.map(|v| array1_to_vec(v, "exposure")).transpose()?;
        let model = Arc::clone(&self.model);
        let (graduated, adopted, report) = py
            .detach(move || {
                run_on_pool(n_jobs, || {
                    apply_graduation_updates_impl(
                        &model, columns, y, weight, exposure, cat_x, updates, None, alpha,
                        box_budget,
                    )
                })
            })
            .map_err(py_err)?;
        let report =
            serde_json::to_string(&report).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            PyTableModel {
                model: Arc::new(graduated),
                serve: Default::default(),
            },
            adopted,
            report,
        ))
    }

    /// Apply LAPACK-smoothed dense table values, re-anchor a log-link model, and enforce the
    /// weighted train-deviance no-harm guard. The design is marshalled and binned only once.
    #[pyo3(signature = (x, y, weight, updates, exposure=None, cat_x=None, n_jobs=None, eval_rows=None))]
    #[allow(clippy::too_many_arguments)]
    fn apply_graduation(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        y: PyReadonlyArray1<'_, f32>,
        weight: PyReadonlyArray1<'_, f32>,
        updates: Vec<(usize, Vec<f64>)>,
        exposure: Option<PyReadonlyArray1<'_, f32>>,
        cat_x: Option<Vec<Vec<String>>>,
        n_jobs: Option<usize>,
        eval_rows: Option<Vec<u32>>,
    ) -> PyResult<(PyTableModel, bool)> {
        let columns = raw_columns_from_array(x)?;
        let y = array1_to_vec(y, "y")?;
        let weight = array1_to_vec(weight, "weight")?;
        let exposure = exposure
            .map(|values| array1_to_vec(values, "exposure"))
            .transpose()?;
        let model = Arc::clone(&self.model);
        let (graduated, adopted) = py
            .detach(move || {
                run_on_pool(n_jobs, || {
                    apply_graduation_updates(
                        &model, columns, y, weight, exposure, cat_x, updates, eval_rows,
                    )
                })
            })
            .map_err(py_err)?;
        Ok((
            PyTableModel {
                model: Arc::new(graduated),
                serve: Default::default(),
            },
            adopted,
        ))
    }

    #[pyo3(signature = (x, out=None, cat_x=None, cat_codes=None, n_jobs=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        out: Option<Bound<'py, PyArray1<f32>>>,
        cat_x: Option<Vec<Vec<String>>>,
        cat_codes: Option<Vec<(PyReadonlyArray1<'_, u32>, Vec<String>)>>,
        n_jobs: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let columns = raw_columns_from_array(x)?;
        let cats = serve_cats(cat_x, cat_codes)?;
        let model = Arc::clone(&self.model);
        let serve = Arc::clone(&self.serve);
        let pred = py
            .detach(move || {
                run_on_pool(n_jobs, || {
                    let ts = table_serve(&model, &serve)?;
                    let binned = serve_binned_tables_any(&model, columns, cats, Some(&ts.cats))?;
                    model.predict_binned_with(&binned, None, Some(&ts.cells))
                })
            })
            .map_err(py_err)?;
        write_or_return_array1(py, pred, out)
    }

    #[pyo3(signature = (x, out=None, cat_x=None, cat_codes=None, n_jobs=None))]
    fn predict_raw<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        out: Option<Bound<'py, PyArray1<f32>>>,
        cat_x: Option<Vec<Vec<String>>>,
        cat_codes: Option<Vec<(PyReadonlyArray1<'_, u32>, Vec<String>)>>,
        n_jobs: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let columns = raw_columns_from_array(x)?;
        let cats = serve_cats(cat_x, cat_codes)?;
        let model = Arc::clone(&self.model);
        let serve = Arc::clone(&self.serve);
        let raw = py
            .detach(move || {
                run_on_pool(n_jobs, || {
                    let ts = table_serve(&model, &serve)?;
                    let binned = serve_binned_tables_any(&model, columns, cats, Some(&ts.cats))?;
                    model.score_raw_with(&binned, None, Some(&ts.cells))
                })
            })
            .map_err(py_err)?;
        write_or_return_array1(py, raw, out)
    }

    /// The exact additive decomposition of `predict_raw`: `(f0, values, feature_sets)` with
    /// `values` a `(n_rows, n_effects)` float64 array of each deployed effect's value per row and
    /// `feature_sets` each effect's raw feature ids. `f0 + values.sum(axis=1)` (summed left to
    /// right) is the float64 score `predict_raw` rounds to float32.
    #[pyo3(signature = (x, cat_x=None, cat_codes=None, n_jobs=None))]
    #[allow(clippy::type_complexity)] // JUSTIFIED: a plain (scalar, array, list) Python tuple.
    fn effect_contributions<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        cat_x: Option<Vec<Vec<String>>>,
        cat_codes: Option<Vec<(PyReadonlyArray1<'_, u32>, Vec<String>)>>,
        n_jobs: Option<usize>,
    ) -> PyResult<(f64, Bound<'py, PyArray2<f64>>, Vec<Vec<u32>>)> {
        let columns = raw_columns_from_array(x)?;
        let cats = serve_cats(cat_x, cat_codes)?;
        let model = Arc::clone(&self.model);
        let serve = Arc::clone(&self.serve);
        let (values, n_rows) = py
            .detach(move || {
                run_on_pool(n_jobs, || {
                    let ts = table_serve(&model, &serve)?;
                    let binned = serve_binned_tables_any(&model, columns, cats, Some(&ts.cats))?;
                    Ok((model.effect_contributions(&binned)?, binned.n_rows as usize))
                })
            })
            .map_err(py_err)?;
        let feature_sets = self.model.effect_feature_sets();
        let values = values
            .into_pyarray(py)
            .reshape([n_rows, feature_sets.len()])
            .map_err(|err| {
                InternalError::new_err(format!("could not reshape contribution array: {err}"))
            })?;
        Ok((self.model.bank.f0, values, feature_sets))
    }

    /// Sobol share `σ²(f_u)/σ²(F)` of every deployed effect under the bank's reference measure,
    /// as `(raw feature ids, share)` sorted by share descending. Read from the cached table
    /// variances: no data needed.
    fn sobol(&self) -> Vec<(Vec<u32>, f64)> {
        self.model
            .bank
            .sobol()
            .into_iter()
            .map(|(u, s)| (u.0.iter().map(|f| f.0).collect(), s))
            .collect()
    }

    #[pyo3(signature = (x, cat_x=None, cat_codes=None, n_jobs=None))]
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        cat_x: Option<Vec<Vec<String>>>,
        cat_codes: Option<Vec<(PyReadonlyArray1<'_, u32>, Vec<String>)>>,
        n_jobs: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if self.model.link != t_boost_core::loss::Link::Logit {
            return Err(PyTypeError::new_err(
                "predict_proba is only available for logit-link models",
            ));
        }
        let columns = raw_columns_from_array(x)?;
        let cats = serve_cats(cat_x, cat_codes)?;
        let model = Arc::clone(&self.model);
        let serve = Arc::clone(&self.serve);
        let pred = py
            .detach(move || {
                run_on_pool(n_jobs, || {
                    let ts = table_serve(&model, &serve)?;
                    let binned = serve_binned_tables_any(&model, columns, cats, Some(&ts.cats))?;
                    model.predict_binned_with(&binned, None, Some(&ts.cells))
                })
            })
            .map_err(py_err)?;
        if pred.is_empty() {
            return Ok(numpy::PyArray2::<f32>::zeros(py, [0usize, 2], false));
        }
        let mut rows = Vec::with_capacity(pred.len());
        for p1 in pred {
            rows.push(vec![1.0 - p1, p1]);
        }
        PyArray::from_vec2(py, &rows).map_err(|err| {
            InternalError::new_err(format!("could not allocate probability array: {err}"))
        })
    }

    /// Merged-grid cell index of every row on every raw feature — a `(n_rows, n_raw_features)`
    /// uint32 array in raw-feature order, the very cells `predict` reads. What an
    /// actual-versus-expected by rating-factor level aggregates over (2026-09-06).
    #[pyo3(signature = (x, cat_x=None))]
    fn cell_indices<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        cat_x: Option<Vec<Vec<String>>>,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        if x.shape().first() == Some(&0) {
            return Ok(PyArray2::<u32>::zeros(
                py,
                [0, self.model.bank.merged_grids.len()],
                false,
            ));
        }
        let columns = raw_columns_from_array(x)?;
        let model = Arc::clone(&self.model);
        let rows = py
            .detach(move || {
                let binned = serve_binned_for_tables(&model, columns, cat_x, None)?;
                model.row_cells(&binned)
            })
            .map_err(py_err)?;
        PyArray::from_vec2(py, &rows).map_err(|err| {
            InternalError::new_err(format!("could not allocate cell-index array: {err}"))
        })
    }

    /// Band every interaction table of this deployed model (see `t_boost_core::banding`): each
    /// becomes a small product grid of bands with the change to predictions held within
    /// `(tolerance · sigma)²` (cross-fitted, curvature-weighted mean squared move on the training
    /// rows). `x`/`cat_x` are the TRAINING design; `h` the loss curvature per row at this
    /// model's score, `mass` the effective row mass (`w · exposure`), `target` this model's raw
    /// score per row, `sigma` the soup's bag noise. Returns `(banded model, report_json)`.
    #[pyo3(signature = (x, h, mass, target, sigma, tolerance=0.75, cat_x=None, max_select_rows=60000, pseudo_rows=20000, seed=0, n_jobs=None, mse_cap=f64::INFINITY))]
    #[allow(clippy::too_many_arguments)] // JUSTIFIED: one knob per banding input (§12.2).
    fn band(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        h: PyReadonlyArray1<'_, f64>,
        mass: PyReadonlyArray1<'_, f64>,
        target: PyReadonlyArray1<'_, f64>,
        sigma: f64,
        tolerance: f64,
        cat_x: Option<Vec<Vec<String>>>,
        max_select_rows: usize,
        pseudo_rows: usize,
        seed: u64,
        n_jobs: Option<usize>,
        mse_cap: f64,
    ) -> PyResult<(PyTableModel, String)> {
        let columns = raw_columns_from_array(x)?;
        let slice = |a: PyReadonlyArray1<'_, f64>, what: &str| -> PyResult<Vec<f64>> {
            a.as_slice().map(<[f64]>::to_vec).map_err(|e| {
                PyValueError::new_err(format!("{what} must be contiguous float64: {e}"))
            })
        };
        let (h, mass, target) = (
            slice(h, "h")?,
            slice(mass, "mass")?,
            slice(target, "target")?,
        );
        let model = Arc::clone(&self.model);
        let cfg = t_boost_core::banding::BandingConfig {
            tolerance,
            max_select_rows,
            pseudo_rows,
            seed,
            ..Default::default()
        };
        let (tm, report) = py
            .detach(move || {
                let run = || -> Result<(TableModel, String), PbError> {
                    let binned = serve_binned_for_tables(&model, columns, cat_x, None)?;
                    let cells = model.column_cells(&binned)?;
                    let rows = t_boost_core::banding::BandingRows {
                        cells: &cells,
                        h: &h,
                        mass: &mass,
                        target: &target,
                        sigma,
                        mse_cap,
                    };
                    let (bank, report) =
                        t_boost_core::banding::band_bank(&model.bank, &rows, &cfg)?;
                    let mut tm = (*model).clone();
                    tm.bank = bank;
                    tm.validate()?;
                    let json = serde_json::to_string(&report).map_err(|err| {
                        PbError::Serialization(format!("could not serialize banding report: {err}"))
                    })?;
                    Ok((tm, json))
                };
                require_nonzero_n_jobs(n_jobs)?;
                match n_jobs {
                    Some(nj) => rayon::ThreadPoolBuilder::new()
                        .num_threads(nj)
                        .build()
                        .map_err(|err| PbError::InvalidConfig {
                            what: format!("could not build rayon pool: {err}"),
                        })?
                        .install(run),
                    None => run(),
                }
            })
            .map_err(py_err)?;
        Ok((
            PyTableModel {
                model: Arc::new(tm),
                serve: Default::default(),
            },
            report,
        ))
    }

    /// One display name per raw feature, in the raw-feature order `cell_indices` uses.
    fn raw_feature_names(&self) -> PyResult<Vec<String>> {
        let m = &self.model;
        (0..m.bank.merged_grids.len())
            .map(|raw| {
                let axis = t_boost_core::serialize::representative_axis_for_raw(
                    &m.provenance,
                    t_boost_core::data::FeatureId(raw as u32),
                )
                .map_err(py_err)?;
                m.schema.feature_names.get(axis).cloned().ok_or_else(|| {
                    py_err(PbError::ShapeMismatch {
                        what: format!("schema missing feature name for raw {raw}"),
                    })
                })
            })
            .collect()
    }

    #[pyo3(signature = (x=None, ref_measure=None, laplace=1.0, measure_floor=0.001, basis_json=None, cat_x=None, overflow=None, weight=None, exposure=None))]
    #[allow(clippy::too_many_arguments, unused_variables)]
    fn tables(
        &self,
        x: Option<PyReadonlyArray2<'_, f32>>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
        basis_json: Option<&str>,
        cat_x: Option<Vec<Vec<String>>>,
        overflow: Option<String>,
        weight: Option<PyReadonlyArray1<'_, f32>>,
        exposure: Option<PyReadonlyArray1<'_, f32>>,
    ) -> PyResult<String> {
        // The bank was purified on the training rows at fit. A call-time `weight`/`exposure`
        // re-centres it on the rows of `x` (`TableBank::recentre_on`), and a `ref_measure`
        // other than the bank's re-expresses it under that measure; either way the table sum
        // on every cell, so every prediction, is unchanged. Neither keeps the deployed ledger.
        let basis = parse_rating_basis(basis_json)?;
        let m = &self.model;
        let requested = ref_measure
            .map(|name| parse_ref_measure(Some(name), laplace, measure_floor))
            .transpose()
            .map_err(py_err)?;
        let mass = combined_explain_weight(weight, exposure)?;
        let binned = match x {
            Some(x) => Some(
                serve_binned_for_tables(m, raw_columns_from_array(x)?, cat_x, None)
                    .map_err(py_err)?,
            ),
            _ => None,
        };
        let bank = export_bank(
            m,
            binned.as_ref(),
            mass.as_deref(),
            requested.as_ref(),
            measure_floor,
        )
        .map_err(py_err)?;
        bank.to_rating_export(
            m.link,
            &m.mode,
            &m.schema,
            &m.provenance,
            &m.schema.cat_encoders,
            basis.as_ref(),
        )
        .and_then(|export| {
            serde_json::to_string_pretty(&export).map_err(|err| {
                PbError::Serialization(format!("could not serialize RatingExport JSON: {err}"))
            })
        })
        .map_err(py_err)
    }

    fn to_json(&self) -> PyResult<String> {
        self.model.to_json().map_err(py_err)
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.model.to_bincode().map_err(py_err)?;
        Ok(PyBytes::new(py, &bytes))
    }
}

/// Reshape a row-major `n × K` flat vector into an `(n, K)` numpy array (empty ⇒ `(0, K)`).
fn multiclass_array2(
    py: Python<'_>,
    flat: Vec<f32>,
    k: usize,
) -> PyResult<Bound<'_, PyArray2<f32>>> {
    if k == 0 || flat.is_empty() {
        return Ok(numpy::PyArray2::<f32>::zeros(py, [0usize, k], false));
    }
    // Zero-copy: move the row-major n*K buffer straight into numpy and reshape (a view), matching
    // the single-output path's into_pyarray — no intermediate per-row Vecs, no second copy.
    let n = flat.len() / k;
    flat.into_pyarray(py)
        .reshape([n, k])
        .map_err(|err| InternalError::new_err(format!("could not reshape multiclass array: {err}")))
}

/// Low-level Python native-softmax multiclass model wrapper (`K` per-class exact logits).
#[pyclass(name = "_MultiClassModel", skip_from_py_object)]
#[derive(Clone)]
struct PyMultiClassModel {
    model: Arc<MultiClassModel>,
}

#[pymethods]
impl PyMultiClassModel {
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        Ok(Self {
            model: Arc::new(decode_multiclass_json(s).map_err(py_err)?),
        })
    }

    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
        Ok(Self {
            model: Arc::new(decode_multiclass(bytes).map_err(py_err)?),
        })
    }

    /// The §G1 K>=3 out-of-bag cell refit's report for the fit that produced this model, or
    /// `None` when the refit was never asked for (or the model was loaded from disk — the
    /// report is runtime-only and never serialized, exactly like `_Model.delta_step_gate`).
    ///
    /// Keys: `lambda` (the joint backtrack's accepted step; `0.0` = the guard declined the whole
    /// correction), `declined`, `n_supports`/`n_blocked`/`reachable_coverage` (how much of the
    /// realized order-<=2 structure the refit could reach — the honest denominator for reading a
    /// `lambda == 0` verdict on a categorical-heavy set), `n_rejected`, `guard_rows`.
    #[getter]
    fn cell_refit_report<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(r) = self.model.cell_refit else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        d.set_item("lambda", r.lambda)?;
        d.set_item("declined", r.declined())?;
        d.set_item("n_supports", r.n_supports)?;
        d.set_item("n_blocked", r.n_blocked)?;
        d.set_item("reachable_coverage", r.reachable_coverage())?;
        d.set_item("n_rejected", r.n_rejected)?;
        d.set_item("guard_rows", r.guard_rows)?;
        Ok(Some(d))
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.model.classes.first().map_or(0, |m| m.grids.len())
    }

    #[getter]
    fn n_classes(&self) -> usize {
        self.model.n_classes()
    }

    #[getter]
    fn class_labels(&self) -> Vec<String> {
        self.model.class_labels.clone()
    }

    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.model
            .classes
            .first()
            .map_or_else(Vec::new, |m| m.schema.feature_names.clone())
    }

    #[pyo3(signature = (x, cat_x=None, n_jobs=None))]
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        cat_x: Option<Vec<Vec<String>>>,
        n_jobs: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let columns = raw_columns_from_array(x)?;
        let model = Arc::clone(&self.model);
        let (flat, k) = py
            .detach(move || {
                run_on_pool(n_jobs, || {
                    let first = model.classes.first().ok_or_else(|| PbError::Internal {
                        what: "multiclass model has no classes".into(),
                    })?;
                    let binned = serve_binned_for_model(first, &columns, cat_x.as_deref())?;
                    let proba = model.predict_proba(&binned)?;
                    Ok::<(Vec<f32>, usize), PbError>((proba, model.n_classes()))
                })
            })
            .map_err(py_err)?;
        multiclass_array2(py, flat, k)
    }

    #[pyo3(signature = (x, cat_x=None, n_jobs=None))]
    fn predict_raw<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        cat_x: Option<Vec<Vec<String>>>,
        n_jobs: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let columns = raw_columns_from_array(x)?;
        let model = Arc::clone(&self.model);
        let (flat, k) = py
            .detach(move || {
                run_on_pool(n_jobs, || {
                    let first = model.classes.first().ok_or_else(|| PbError::Internal {
                        what: "multiclass model has no classes".into(),
                    })?;
                    let binned = serve_binned_for_model(first, &columns, cat_x.as_deref())?;
                    let raw = model.predict_raw(&binned)?;
                    Ok::<(Vec<f32>, usize), PbError>((raw, model.n_classes()))
                })
            })
            .map_err(py_err)?;
        multiclass_array2(py, flat, k)
    }

    /// Per-class rating tables as a JSON object keyed by class label (each value the per-class
    /// [`Model`]'s rating export). Every per-class logit decomposes exactly (§ exactness per class).
    /// PROBE (2026-09-07): realized supports (union over classes) as `(raw ids, summed variance)`.
    #[pyo3(signature = (x, weight=None, cat_x=None, ref_measure=None, laplace=1.0, measure_floor=0.001))]
    #[allow(clippy::too_many_arguments)]
    fn mc_supports(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        weight: Option<PyReadonlyArray1<'_, f32>>,
        cat_x: Option<Vec<Vec<String>>>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
    ) -> PyResult<Vec<(Vec<u32>, f64)>> {
        let columns = raw_columns_from_array(x)?;
        let n = columns.first().map_or(0, Vec::len);
        let w_vec: Vec<f32> = match weight {
            Some(w) => w
                .as_slice()
                .map_err(|e| PyValueError::new_err(format!("weight must be contiguous: {e}")))?
                .to_vec(),
            None => vec![1.0; n],
        };
        let model = Arc::clone(&self.model);
        let w_measure = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let out = py
            .detach(move || {
                let first = model.classes.first().ok_or_else(|| PbError::Internal {
                    what: "multiclass model has no classes".into(),
                })?;
                let binned = serve_binned_for_model(first, &columns, cat_x.as_deref())?;
                let serve = ServeBinnedMatrix(binned);
                let v = multiclass_realized_supports(&model, &serve, w_measure, &w_vec)?;
                Ok::<_, PbError>(
                    v.into_iter()
                        .map(|(u, var)| (feature_set_ids(&u), var))
                        .collect::<Vec<_>>(),
                )
            })
            .map_err(py_err)?;
        Ok(out)
    }

    /// The lossless tables-only form of this model: every class's full bank purified on the
    /// training design `x` under the measure, no table dropped, no intercept re-anchor (see
    /// `multiclass_full_tables`). What an unpruned multiclass fit deploys.
    #[pyo3(signature = (x, weight, cat_x=None, ref_measure=None, laplace=1.0, measure_floor=0.001))]
    #[allow(clippy::too_many_arguments)] // JUSTIFIED: the design plus the measure's knobs.
    fn to_tables(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        weight: PyReadonlyArray1<'_, f32>,
        cat_x: Option<Vec<Vec<String>>>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
    ) -> PyResult<PyMultiClassTableModel> {
        let columns = raw_columns_from_array(x)?;
        let w_vec = weight
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("weight must be contiguous: {e}")))?
            .to_vec();
        let model = Arc::clone(&self.model);
        let w_measure = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let tm = py
            .detach(move || {
                let first = model.classes.first().ok_or_else(|| PbError::Internal {
                    what: "multiclass model has no classes".into(),
                })?;
                let binned = serve_binned_for_model(first, &columns, cat_x.as_deref())?;
                multiclass_full_tables(&model, &ServeBinnedMatrix(binned), &w_vec, w_measure)
            })
            .map_err(py_err)?;
        Ok(PyMultiClassTableModel {
            model: Arc::new(tm),
            serve: Default::default(),
        })
    }

    /// PROBE (2026-09-07): retain an arbitrary keep-set on this model (the K>=3 deploy step).
    #[pyo3(signature = (x, y, weight, keep, cat_x=None, ref_measure=None, laplace=1.0, measure_floor=0.001))]
    #[allow(clippy::too_many_arguments)]
    fn mc_apply_keepset(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        y: PyReadonlyArray1<'_, f32>,
        weight: PyReadonlyArray1<'_, f32>,
        keep: Vec<Vec<u32>>,
        cat_x: Option<Vec<Vec<String>>>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
    ) -> PyResult<PyMultiClassTableModel> {
        let columns = raw_columns_from_array(x)?;
        let n_classes = self.model.n_classes();
        let labels: Vec<u32> = y
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("y must be contiguous: {e}")))?
            .iter()
            .map(|&l| {
                if !l.is_finite() || l < 0.0 || l.fract() != 0.0 || l as usize >= n_classes {
                    Err(PyValueError::new_err(format!(
                        "label {l} not in 0..{n_classes}"
                    )))
                } else {
                    Ok(l as u32)
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        let w_vec = weight
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("weight must be contiguous: {e}")))?
            .to_vec();
        let keepset: Vec<FeatureSet> = keep
            .iter()
            .map(|ids| feature_set_from_raw_ids(ids))
            .collect();
        let model = Arc::clone(&self.model);
        let w_measure = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let tm = py
            .detach(move || {
                let first = model.classes.first().ok_or_else(|| PbError::Internal {
                    what: "multiclass model has no classes".into(),
                })?;
                let binned = serve_binned_for_model(first, &columns, cat_x.as_deref())?;
                let serve = ServeBinnedMatrix(binned);
                prune_multiclass_to_keepset(&model, &serve, &labels, &w_vec, w_measure, &keepset)
            })
            .map_err(py_err)?;
        Ok(PyMultiClassTableModel {
            model: Arc::new(tm),
            serve: Default::default(),
        })
    }

    /// PROBE (2026-09-07): carve arms on `mask` rows — `(rows, f0[K,n], full[K,n], group[G,K,n])`.
    #[pyo3(signature = (x, weight, groups, mask, cat_x=None, ref_measure=None, laplace=1.0, measure_floor=0.001))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn mc_carve_arms<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        weight: PyReadonlyArray1<'_, f32>,
        groups: Vec<Vec<Vec<u32>>>,
        mask: Vec<bool>,
        cat_x: Option<Vec<Vec<String>>>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
    ) -> PyResult<(
        Vec<usize>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray3<f64>>,
    )> {
        let columns = raw_columns_from_array(x)?;
        let w_vec = weight
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("weight must be contiguous: {e}")))?
            .to_vec();
        let groups_fs: Vec<Vec<FeatureSet>> = groups
            .iter()
            .map(|g| g.iter().map(|ids| feature_set_from_raw_ids(ids)).collect())
            .collect();
        let model = Arc::clone(&self.model);
        let w_measure = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let arms = py
            .detach(move || {
                let first = model.classes.first().ok_or_else(|| PbError::Internal {
                    what: "multiclass model has no classes".into(),
                })?;
                let binned = serve_binned_for_model(first, &columns, cat_x.as_deref())?;
                let serve = ServeBinnedMatrix(binned);
                multiclass_carve_arms(&model, &serve, w_measure, &w_vec, &groups_fs, &mask)
            })
            .map_err(py_err)?;
        let f0 = PyArray2::from_vec2(py, &arms.f0)
            .map_err(|e| PyValueError::new_err(format!("f0 shape: {e}")))?;
        let full = PyArray2::from_vec2(py, &arms.full)
            .map_err(|e| PyValueError::new_err(format!("full shape: {e}")))?;
        let group = PyArray3::from_vec3(py, &arms.group)
            .map_err(|e| PyValueError::new_err(format!("group shape: {e}")))?;
        Ok((arms.rows, f0, full, group))
    }

    /// PROBE (2026-09-07): out-of-bag arms — `(rows, f0[K,n], full[K,n], group[G,K,n])`.
    #[pyo3(signature = (x, weight, groups, cat_x=None, ref_measure=None, laplace=1.0, measure_floor=0.001))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn mc_oob_arms<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        weight: PyReadonlyArray1<'_, f32>,
        groups: Vec<Vec<Vec<u32>>>,
        cat_x: Option<Vec<Vec<String>>>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
    ) -> PyResult<(
        Vec<usize>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray3<f64>>,
    )> {
        let columns = raw_columns_from_array(x)?;
        let w_vec = weight
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("weight must be contiguous: {e}")))?
            .to_vec();
        let groups_fs: Vec<Vec<FeatureSet>> = groups
            .iter()
            .map(|g| g.iter().map(|ids| feature_set_from_raw_ids(ids)).collect())
            .collect();
        let model = Arc::clone(&self.model);
        let w_measure = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let arms = py
            .detach(move || {
                let first = model.classes.first().ok_or_else(|| PbError::Internal {
                    what: "multiclass model has no classes".into(),
                })?;
                let binned = serve_binned_for_model(first, &columns, cat_x.as_deref())?;
                let serve = ServeBinnedMatrix(binned);
                multiclass_oob_arms(&model, &serve, w_measure, &w_vec, &groups_fs)
            })
            .map_err(py_err)?;
        let f0 = PyArray2::from_vec2(py, &arms.f0)
            .map_err(|e| PyValueError::new_err(format!("f0 shape: {e}")))?;
        let full = PyArray2::from_vec2(py, &arms.full)
            .map_err(|e| PyValueError::new_err(format!("full shape: {e}")))?;
        let group = PyArray3::from_vec3(py, &arms.group)
            .map_err(|e| PyValueError::new_err(format!("group shape: {e}")))?;
        Ok((arms.rows, f0, full, group))
    }

    #[pyo3(signature = (x, ref_measure=None, laplace=1.0, measure_floor=0.001, basis_json=None, cat_x=None, overflow=None, weight=None, exposure=None))]
    #[allow(clippy::too_many_arguments)]
    fn tables(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
        basis_json: Option<&str>,
        cat_x: Option<Vec<Vec<String>>>,
        overflow: Option<String>,
        weight: Option<PyReadonlyArray1<'_, f32>>,
        exposure: Option<PyReadonlyArray1<'_, f32>>,
    ) -> PyResult<String> {
        let columns = raw_columns_from_array(x)?;
        let model = Arc::clone(&self.model);
        let w = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let basis = parse_rating_basis(basis_json)?;
        let budget = parse_table_budget(overflow.as_deref()).map_err(py_err)?;
        let combined_weight = combined_explain_weight(weight, exposure)?;
        py.detach(move || {
            let first = model.classes.first().ok_or_else(|| PbError::Internal {
                what: "multiclass model has no classes".into(),
            })?;
            let binned = serve_binned_for_model(first, &columns, cat_x.as_deref())?;
            let serve = t_boost_core::data::ServeBinnedMatrix(binned);
            let mut map = serde_json::Map::new();
            let (w, joint) = if w == RefMeasure::Joint {
                (
                    RefMeasure::ExposureMarginals {
                        floor: measure_floor,
                    },
                    true,
                )
            } else {
                (w, false)
            };
            for (label, m) in model.class_labels.iter().zip(&model.classes) {
                let mut bank = match &combined_weight {
                    Some(mass) => {
                        m.explain_with_budget_weighted(&serve, w.clone(), budget, mass)?
                    }
                    None => m.explain_with_budget(&serve, w.clone(), budget)?,
                };
                if joint {
                    bank = t_boost_core::joint::rejoint(
                        &bank,
                        &t_boost_core::joint::JointOptions::default(),
                    )?;
                    bank.measure_joint_variance(
                        &m.schema.cat_encoders,
                        &serve.0,
                        combined_weight.as_deref(),
                    )?;
                }
                let export = bank.to_rating_export(
                    m.link,
                    &m.mode,
                    &m.schema,
                    &m.provenance,
                    &m.schema.cat_encoders,
                    basis.as_ref(),
                )?;
                let value = serde_json::to_value(&export).map_err(|err| {
                    PbError::Serialization(format!("rating export to JSON value: {err}"))
                })?;
                map.insert(label.clone(), value);
            }
            serde_json::to_string_pretty(&serde_json::Value::Object(map)).map_err(|err| {
                PbError::Serialization(format!("could not serialize multiclass tables JSON: {err}"))
            })
        })
        .map_err(py_err)
    }

    /// Prune the K per-class banks into a tables-only [`PyMultiClassTableModel`] with a shared
    /// keep-set selected on held-out softmax cross-entropy deviance. `labels` are class indices; the
    /// caller carves `sel_rows` (held out from the fit). Returns `(model, prune_report_json)`.
    #[pyo3(signature = (x, labels, weight, sel_rows, ref_measure=None, laplace=1.0, measure_floor=0.001, se_rule=0.0, lambda_boxes=0.0, n_folds=5, cat_x=None, n_jobs=None))]
    #[allow(clippy::too_many_arguments)]
    fn prune_to_tables(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        labels: PyReadonlyArray1<'_, u32>,
        weight: PyReadonlyArray1<'_, f32>,
        sel_rows: Vec<usize>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
        se_rule: f64,
        lambda_boxes: f64,
        n_folds: usize,
        cat_x: Option<Vec<Vec<String>>>,
        n_jobs: Option<usize>,
    ) -> PyResult<(PyMultiClassTableModel, String)> {
        cap_global_pool_once(n_jobs);
        let columns = raw_columns_from_array(x)?;
        let labels_vec = labels
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("labels must be contiguous: {e}")))?
            .to_vec();
        let w_vec = weight
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("weight must be contiguous: {e}")))?
            .to_vec();
        let model = Arc::clone(&self.model);
        let w_measure = parse_ref_measure(ref_measure, laplace, measure_floor).map_err(py_err)?;
        let (mct, report_json) = py
            .detach(move || {
                run_on_pool(n_jobs, || {
                    let first = model.classes.first().ok_or_else(|| PbError::Internal {
                        what: "multiclass model has no classes".into(),
                    })?;
                    let binned = serve_binned_for_model(first, &columns, cat_x.as_deref())?;
                    let serve = ServeBinnedMatrix(binned);
                    let cfg = PruneConfig {
                        se_rule,
                        lambda_boxes,
                        lambda_tables: 0.0,
                        table_price_min_arity: DEFAULT_TABLE_MIN_ARITY,
                    };
                    let (mct, report) = prune_multiclass_to_tables(
                        &model,
                        &serve,
                        &labels_vec,
                        &w_vec,
                        w_measure,
                        &sel_rows,
                        n_folds,
                        &cfg,
                    )?;
                    let rj = serde_json::to_string(&report).map_err(|e| {
                        PbError::Serialization(format!("could not serialize PruneReport: {e}"))
                    })?;
                    Ok::<(MultiClassTableModel, String), PbError>((mct, rj))
                })
            })
            .map_err(py_err)?;
        Ok((
            PyMultiClassTableModel {
                model: Arc::new(mct),
                serve: Default::default(),
            },
            report_json,
        ))
    }

    fn to_json(&self) -> PyResult<String> {
        encode_multiclass_json(&self.model).map_err(py_err)
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = encode_multiclass(&self.model).map_err(py_err)?;
        Ok(PyBytes::new(py, &bytes))
    }
}

/// Low-level Python tables-only K-class model: K per-class pruned banks combined by softmax. Produced
/// by [`PyMultiClassModel::prune_to_tables`].
#[pyclass(name = "_MultiClassTableModel", skip_from_py_object)]
#[derive(Clone)]
struct PyMultiClassTableModel {
    model: Arc<MultiClassTableModel>,
    /// Per-class serve maps, built on the first predict (see `multiclass_table_serve`).
    serve: Arc<OnceLock<MultiClassTableServe>>,
}

#[pymethods]
impl PyMultiClassTableModel {
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        Ok(Self {
            model: Arc::new(MultiClassTableModel::from_json(s).map_err(py_err)?),
            serve: Default::default(),
        })
    }

    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
        Ok(Self {
            model: Arc::new(MultiClassTableModel::from_bincode(bytes).map_err(py_err)?),
            serve: Default::default(),
        })
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.model.classes.first().map_or(0, |m| m.grids.len())
    }

    #[getter]
    fn n_classes(&self) -> usize {
        self.model.n_classes()
    }

    #[getter]
    fn class_labels(&self) -> Vec<String> {
        self.model.class_labels.clone()
    }

    /// Tables across every class bank, dense and factored (see `_TableModel.deployed_table_count`).
    fn deployed_table_count(&self) -> usize {
        self.model
            .classes
            .iter()
            .map(|c| c.bank.tables.len() + c.bank.factored.len())
            .sum()
    }

    /// Tables and cells by interaction order summed over every class bank, with `"banks"` the
    /// number of class banks (see `_TableModel.deployed_census`).
    fn deployed_census<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let banks: Vec<&TableBank> = self.model.classes.iter().map(|c| &c.bank).collect();
        deployed_census_dict(py, &banks)
    }

    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.model
            .classes
            .first()
            .map_or_else(Vec::new, |m| m.schema.feature_names.clone())
    }

    #[pyo3(signature = (x, cat_x=None, cat_codes=None, n_jobs=None))]
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        cat_x: Option<Vec<Vec<String>>>,
        cat_codes: Option<Vec<(PyReadonlyArray1<'_, u32>, Vec<String>)>>,
        n_jobs: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let columns = raw_columns_from_array(x)?;
        let cats = serve_cats(cat_x, cat_codes)?;
        let model = Arc::clone(&self.model);
        let serve = Arc::clone(&self.serve);
        let (flat, k) = py
            .detach(move || {
                run_on_pool(n_jobs, || {
                    let first = model.classes.first().ok_or_else(|| PbError::Internal {
                        what: "multiclass tables model has no classes".into(),
                    })?;
                    let ts = multiclass_table_serve(&model, &serve)?;
                    let binned = serve_binned_tables_any(first, columns, cats, Some(&ts.cats))?;
                    let proba = model.predict_proba_with(&binned, Some(&ts.cells))?;
                    Ok::<(Vec<f32>, usize), PbError>((proba, model.n_classes()))
                })
            })
            .map_err(py_err)?;
        multiclass_array2(py, flat, k)
    }

    #[pyo3(signature = (x, cat_x=None, cat_codes=None, n_jobs=None))]
    fn predict_raw<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        cat_x: Option<Vec<Vec<String>>>,
        cat_codes: Option<Vec<(PyReadonlyArray1<'_, u32>, Vec<String>)>>,
        n_jobs: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let columns = raw_columns_from_array(x)?;
        let cats = serve_cats(cat_x, cat_codes)?;
        let model = Arc::clone(&self.model);
        let serve = Arc::clone(&self.serve);
        let (flat, k) = py
            .detach(move || {
                run_on_pool(n_jobs, || {
                    let first = model.classes.first().ok_or_else(|| PbError::Internal {
                        what: "multiclass tables model has no classes".into(),
                    })?;
                    let ts = multiclass_table_serve(&model, &serve)?;
                    let binned = serve_binned_tables_any(first, columns, cats, Some(&ts.cats))?;
                    let raw = model.predict_raw_with(&binned, Some(&ts.cells))?;
                    Ok::<(Vec<f32>, usize), PbError>((raw, model.n_classes()))
                })
            })
            .map_err(py_err)?;
        multiclass_array2(py, flat, k)
    }

    /// `_TableModel.effect_contributions` for every class logit: one `(f0, values,
    /// feature_sets)` per class, in `class_labels` order.
    #[pyo3(signature = (x, cat_x=None, cat_codes=None, n_jobs=None))]
    #[allow(clippy::type_complexity)] // JUSTIFIED: a plain list of (scalar, array, list) tuples.
    fn effect_contributions<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f32>,
        cat_x: Option<Vec<Vec<String>>>,
        cat_codes: Option<Vec<(PyReadonlyArray1<'_, u32>, Vec<String>)>>,
        n_jobs: Option<usize>,
    ) -> PyResult<Vec<(f64, Bound<'py, PyArray2<f64>>, Vec<Vec<u32>>)>> {
        let columns = raw_columns_from_array(x)?;
        let cats = serve_cats(cat_x, cat_codes)?;
        let model = Arc::clone(&self.model);
        let serve = Arc::clone(&self.serve);
        let (per_class, n_rows) = py
            .detach(move || {
                run_on_pool(n_jobs, || {
                    let first = model.classes.first().ok_or_else(|| PbError::Internal {
                        what: "multiclass tables model has no classes".into(),
                    })?;
                    let ts = multiclass_table_serve(&model, &serve)?;
                    let binned = serve_binned_tables_any(first, columns, cats, Some(&ts.cats))?;
                    let per_class = model
                        .classes
                        .iter()
                        .map(|tm| tm.effect_contributions(&binned))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok((per_class, binned.n_rows as usize))
                })
            })
            .map_err(py_err)?;
        per_class
            .into_iter()
            .zip(&self.model.classes)
            .map(|(values, tm)| {
                let feature_sets = tm.effect_feature_sets();
                let values = values
                    .into_pyarray(py)
                    .reshape([n_rows, feature_sets.len()])
                    .map_err(|err| {
                        InternalError::new_err(format!(
                            "could not reshape contribution array: {err}"
                        ))
                    })?;
                Ok((tm.bank.f0, values, feature_sets))
            })
            .collect()
    }

    /// `_TableModel.sobol` for every class logit, in `class_labels` order.
    fn sobol(&self) -> Vec<Vec<(Vec<u32>, f64)>> {
        self.model
            .classes
            .iter()
            .map(|tm| {
                tm.bank
                    .sobol()
                    .into_iter()
                    .map(|(u, s)| (u.0.iter().map(|f| f.0).collect(), s))
                    .collect()
            })
            .collect()
    }

    /// Per-class rating tables as a JSON object keyed by class label (each the stored pruned bank's
    /// export). `x`/`ref_measure`/`weight`/`exposure` are ignored — the banks are frozen.
    #[pyo3(signature = (x=None, ref_measure=None, laplace=1.0, measure_floor=0.001, basis_json=None, cat_x=None, overflow=None, weight=None, exposure=None))]
    #[allow(clippy::too_many_arguments, unused_variables)]
    fn tables(
        &self,
        x: Option<PyReadonlyArray2<'_, f32>>,
        ref_measure: Option<String>,
        laplace: f32,
        measure_floor: f32,
        basis_json: Option<&str>,
        cat_x: Option<Vec<Vec<String>>>,
        overflow: Option<String>,
        weight: Option<PyReadonlyArray1<'_, f32>>,
        exposure: Option<PyReadonlyArray1<'_, f32>>,
    ) -> PyResult<String> {
        let basis = parse_rating_basis(basis_json)?;
        let m = &self.model;
        // Same contract as the scalar `_TableModel.tables`, class by class: a call-time
        // `weight`/`exposure` re-centres each class's bank on the rows of `x`, and `ref_measure`
        // re-expresses it under another ledger, without touching predictions.
        let requested = ref_measure
            .map(|name| parse_ref_measure(Some(name), laplace, measure_floor))
            .transpose()
            .map_err(py_err)?;
        let mass = combined_explain_weight(weight, exposure)?;
        let binned = match (x, m.classes.first()) {
            (Some(x), Some(first)) => Some(
                serve_binned_for_tables(first, raw_columns_from_array(x)?, cat_x, None)
                    .map_err(py_err)?,
            ),
            _ => None,
        };
        let mut map = serde_json::Map::new();
        for (label, tm) in m.class_labels.iter().zip(&m.classes) {
            let bank = export_bank(
                tm,
                binned.as_ref(),
                mass.as_deref(),
                requested.as_ref(),
                measure_floor,
            )
            .map_err(py_err)?;
            let export = bank
                .to_rating_export(
                    tm.link,
                    &tm.mode,
                    &tm.schema,
                    &tm.provenance,
                    &tm.schema.cat_encoders,
                    basis.as_ref(),
                )
                .map_err(py_err)?;
            let value = serde_json::to_value(&export).map_err(|err| {
                InternalError::new_err(format!("rating export to JSON value: {err}"))
            })?;
            map.insert(label.clone(), value);
        }
        serde_json::to_string_pretty(&serde_json::Value::Object(map))
            .map_err(|err| InternalError::new_err(format!("multiclass tables JSON: {err}")))
    }

    fn to_json(&self) -> PyResult<String> {
        self.model.to_json().map_err(py_err)
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.model.to_bincode().map_err(py_err)?;
        Ok(PyBytes::new(py, &bytes))
    }
}

/// Low-level Python table-bank wrapper.
#[pyclass(name = "_TableBank", skip_from_py_object)]
#[derive(Clone)]
struct PyTableBank {
    bank: TableBank,
}

#[pymethods]
impl PyTableBank {
    #[getter]
    fn f0(&self) -> f64 {
        self.bank.f0
    }

    #[getter]
    fn n_tables(&self) -> usize {
        // Total effect tables in the decomposition, including factored over-budget order-3
        // effects (§08.10) — which are real f_u terms, just stored per-tree, not densified.
        self.bank.tables.len() + self.bank.factored.len()
    }

    fn score_cells(&self, cells: Vec<u32>) -> PyResult<f64> {
        self.bank.score(&cells).map_err(py_err)
    }

    fn sobol(&self) -> Vec<(Vec<u32>, f64)> {
        self.bank
            .sobol()
            .into_iter()
            .map(|(u, s)| (u.0.iter().map(|f| f.0).collect(), s))
            .collect()
    }
}

/// Build a name-keyed [`MonotoneMap`] from a positional sign vector (`-1`/`0`/`+1` per
/// feature). Keyed `f{i}` to match the fit-time schema names (the user's `feature_names`
/// are applied to the schema only AFTER fit, so monotone resolution runs against `f{i}`).
fn build_monotone_map(signs: Option<&[i8]>, n_features: usize) -> Result<MonotoneMap, PbError> {
    let mut map = MonotoneMap::new();
    let Some(signs) = signs else {
        return Ok(map);
    };
    if signs.len() != n_features {
        return Err(PbError::ShapeMismatch {
            what: format!("monotone len {} != n_features {n_features}", signs.len()),
        });
    }
    for (i, &s) in signs.iter().enumerate() {
        let sign = match s {
            0 => continue,
            1 => MonoSign::Increasing,
            -1 => MonoSign::Decreasing,
            other => {
                return Err(PbError::InvalidConfig {
                    what: format!("monotone[{i}] must be -1, 0, or 1, got {other}"),
                })
            }
        };
        map.insert(format!("f{i}"), sign);
    }
    Ok(map)
}

/// Gather `src[idx[0]], src[idx[1]], …` for a fold's row-subset. `idx` is built by scanning the
/// fold assignment over `0..n` and every array passed here has `n` rows, so an out-of-range index
/// is an internal invariant break — surfaced as an error rather than panicking (the workspace
/// denies `indexing_slicing`).
fn gather_f32(src: &[f32], idx: &[usize]) -> Result<Vec<f32>, PbError> {
    idx.iter()
        .map(|&i| {
            src.get(i).copied().ok_or_else(|| PbError::Internal {
                what: format!("fold gather row {i} out of range (len {})", src.len()),
            })
        })
        .collect()
}

/// [`gather_f32`] for categorical string levels.
fn gather_str(src: &[String], idx: &[usize]) -> Result<Vec<String>, PbError> {
    idx.iter()
        .map(|&i| {
            src.get(i).cloned().ok_or_else(|| PbError::Internal {
                what: format!("fold gather row {i} out of range (len {})", src.len()),
            })
        })
        .collect()
}

/// `groups[r]` for each row in `rows`, as a shape error instead of a panic on a short array.
fn gather_groups(groups: &[u32], rows: &[usize]) -> Result<Vec<u32>, PbError> {
    rows.iter()
        .map(|&r| {
            groups
                .get(r)
                .copied()
                .ok_or_else(|| PbError::ShapeMismatch {
                    what: format!("bag_groups len {} has no row {r}", groups.len()),
                })
        })
        .collect()
}

/// [`gather_f32`] for a boolean mask (e.g. a caller-supplied `es_holdout` gathered to one
/// prune fold's train subset).
fn gather_bool(src: &[bool], idx: &[usize]) -> Result<Vec<bool>, PbError> {
    idx.iter()
        .map(|&i| {
            src.get(i).copied().ok_or_else(|| PbError::Internal {
                what: format!("fold gather row {i} out of range (len {})", src.len()),
            })
        })
        .collect()
}

/// Run `op` on a rayon pool of `n_jobs` threads, or directly (ambient/global pool) when `None`.
/// Lets predict/score honour the fitted estimator's thread budget instead of the score par_iter
/// grabbing every core off the global pool (which oversubscribes a box running many estimators).
/// Cap rayon's GLOBAL pool to `n_jobs` once per process.
///
/// Compute runs on the per-call local pools built by `run_on_pool` (and the prune/fit paths),
/// but stray global-pool use — e.g. a `rayon::current_num_threads()` probe outside an
/// `install`, or any bare `par_iter` — lazily spawns rayon's *default* global pool, sized to
/// ALL logical cores. Its idle worker threads spin-wait. One process is harmless; but when a
/// caller runs many fits as concurrent processes (a benchmark harness with N workers), that is
/// N × (cores) spinning threads oversubscribing the box — observed as load ~180 with only ~2
/// useful cores per worker. Sizing the global pool to the caller's `n_jobs` budget removes the
/// storm and lets `RAYON_NUM_THREADS` be irrelevant.
///
/// Bit-identical: thread count never changes results (see the core's
/// `histogram_is_byte_identical_across_thread_counts`). Best-effort and idempotent — guarded by
/// a `Once`, and if the global pool was already built `build_global` returns `Err`, which we
/// intentionally ignore (leave whatever exists in place).
fn cap_global_pool_once(n_jobs: Option<usize>) {
    let _ = worker_process_id();
    use std::sync::Once;
    static GLOBAL_POOL_CAP: Once = Once::new();
    if let Some(nj) = n_jobs {
        if nj >= 1 {
            GLOBAL_POOL_CAP.call_once(|| {
                let _ = rayon::ThreadPoolBuilder::new()
                    .num_threads(nj)
                    .build_global();
            });
        }
    }
}

/// Scheduling-only: the scored output is pool-independent (disjoint per-row writes, fixed
/// within-row reduction order), so the `Some`/`None` paths are bit-identical.
fn run_on_pool<R: Send>(
    n_jobs: Option<usize>,
    op: impl FnOnce() -> Result<R, PbError> + Send,
) -> Result<R, PbError> {
    require_nonzero_n_jobs(n_jobs)?;
    // A fork inherits both Rayon workers and mutex bookkeeping, without their
    // threads. Refuse before touching either the local cache or global pool.
    if worker_process_id() != std::process::id() {
        return Err(PbError::InvalidInput {
            what: "native worker state was inherited through fork; use multiprocessing's spawn or forkserver start method".into(),
        });
    }
    cap_global_pool_once(n_jobs);
    match n_jobs {
        Some(nj) => serve_pool(nj)?.install(op),
        None => op(),
    }
}

fn worker_process_id() -> u32 {
    static PID: OnceLock<u32> = OnceLock::new();
    *PID.get_or_init(std::process::id)
}

/// The scoped pool [`run_on_pool`] installs for an explicit `n_jobs`, built once per width and
/// reused. Building a pool spawns (and later joins) every thread, so building one per call cost
/// ~60 us per thread on every predict: a 1-row predict at `n_jobs=16` took 1.0 ms, 16x its scoring.
/// Thread count never changes results (the determinism guarantee), so sharing a pool between calls
/// is a pure scheduling choice. The idle threads of each width used stay parked for the process.
fn serve_pool(width: usize) -> Result<Arc<rayon::ThreadPool>, PbError> {
    static POOLS: OnceLock<std::sync::Mutex<BTreeMap<usize, Arc<rayon::ThreadPool>>>> =
        OnceLock::new();
    let mut pools = POOLS
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| PbError::Internal {
            what: "serve pool cache mutex poisoned".into(),
        })?;
    if let Some(pool) = pools.get(&width) {
        return Ok(Arc::clone(pool));
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(width)
            .build()
            .map_err(|err| PbError::InvalidConfig {
                what: format!("could not build rayon pool: {err}"),
            })?,
    );
    pools.insert(width, Arc::clone(&pool));
    Ok(pool)
}

/// Build/install the LOCAL scoped pool a single fit runs its own par_iters on, sized to the
/// scheduling heuristic `fit_pool_width` (`sklearn.py`'s `_default_fit_pool`) when the caller set
/// one, else the caller's own `n_jobs`; ambient (no pool of its own) when both are `None`.
///
/// Deliberately DOES NOT touch [`cap_global_pool_once`] — that must keep seeing the caller's TRUE
/// `n_jobs` (callers already pass it separately, e.g. `PyBooster::fit`'s `cap_global_pool_once(
/// self.n_jobs)`), never `fit_pool_width`. A default-config fit (`n_jobs=None`,
/// `fit_pool_width=Some(heuristic)`) narrows only THIS fit's own local pool; the process-global
/// pool stays uncapped (or capped to the user's real budget), so a later `n_jobs=None` serve call
/// (predict/tables/apply_graduation, all riding `run_on_pool(None)` -> the ambient/global pool)
/// still gets full width instead of being permanently pinned at the fit's narrower heuristic — the
/// regression this function was added to fix (a heuristic width of e.g. 12 was reaching
/// `cap_global_pool_once` and pinning the process-global pool at 12 for its whole lifetime).
///
/// Composability note: builds its own scoped pool whenever `fit_pool_width.or(n_jobs)` is `Some`,
/// even when this call happens to be nested inside an already-installed outer pool — rayon
/// supports nested `install` (the outer pool's thread just blocks on the inner one), so this can
/// make an inner fit narrower or wider than its enclosing pool. Accepted trade-off: every current
/// caller in this crate resets its own state's `n_jobs`/`fit_pool_width` to `None` before
/// recursing into a nested fit (see `fit_multiclass_pruned_owned`), so double pool-nesting does
/// not actually happen on any path here today, and the GIL makes genuine concurrent nested fits
/// from the Python surface rare regardless.
fn run_fit_pool<R: Send>(
    n_jobs: Option<usize>,
    fit_pool_width: Option<usize>,
    op: impl FnOnce() -> Result<R, PbError> + Send,
) -> Result<R, PbError> {
    require_nonzero_n_jobs(n_jobs)?;
    require_nonzero_n_jobs(fit_pool_width)?;
    match fit_pool_width.or(n_jobs) {
        Some(width) => rayon::ThreadPoolBuilder::new()
            .num_threads(width)
            .build()
            .map_err(|err| PbError::InvalidConfig {
                what: format!("could not build rayon pool: {err}"),
            })?
            .install(op),
        None => op(),
    }
}

/// `rayon::ThreadPoolBuilder::num_threads(0)` means "use the default (all logical cores)", the
/// opposite of what a caller probing for "disable parallelism" would expect — and the exact
/// oversubscription [`run_on_pool`]'s ambient-pool `None` path exists to avoid. The `_Booster`
/// constructor already rejects `n_jobs=0`; every per-call `n_jobs` parameter must match.
fn require_nonzero_n_jobs(n_jobs: Option<usize>) -> Result<(), PbError> {
    if matches!(n_jobs, Some(0)) {
        return Err(PbError::InvalidConfig {
            what: "n_jobs must be >= 1 when set".into(),
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn fit_owned(
    state: PyBooster,
    columns: Vec<Vec<f32>>,
    y: Vec<f32>,
    weight: Option<Vec<f32>>,
    exposure: Option<Vec<f32>>,
    feature_names: Option<Vec<String>>,
    class_labels: Option<Vec<String>>,
    monotone: Option<Vec<i8>>,
    cat_x: Option<Vec<Vec<String>>>,
    es_holdout: Option<Vec<bool>>,
    bag_groups: Option<Vec<u32>>,
) -> Result<Model, PbError> {
    let n_numeric = columns.len();
    let n_cat = cat_x.as_ref().map_or(0, Vec::len);
    let monotone_map = build_monotone_map(monotone.as_deref(), n_numeric + n_cat)?;
    if let Some(mask) = es_holdout.as_deref() {
        if mask.len() != y.len() {
            return Err(PbError::ShapeMismatch {
                what: format!("es_holdout len {} != y len {}", mask.len(), y.len()),
            });
        }
    }
    // Internal early-stopping needs the validation fold's categorical encodings to exclude
    // each val row's own target. KFold cross-fit (the default) produces exactly that — OOF
    // per-row training encodings (cat.rs `kfold_training_encodings` excludes the row's whole
    // fold) — and the val fold is carved by index from that already-encoded matrix, so it is
    // leakage-free. Ordered/LeaveOneOut have subtler profiles (the LOO target-encoding
    // pathology), so keep them gated with internal early stopping for now.
    if cat_x.is_some()
        && state.config.validation_fraction.is_some()
        && matches!(state.cat_config.leakage, LeakageScheme::LeaveOneOut)
    {
        return Err(PbError::InvalidConfig {
            what: "validation_fraction with native categoricals is not supported under \
                   cat_leakage='loo' (own-target mean-shift pathology); use 'kfold' (default) \
                   or 'ordered' (one-honest-holdout), or an external validation split"
                .into(),
        });
    }
    // One rayon pool for the whole fit; the fit body (`fit_model_ambient`) runs its own par_iters
    // on this ambient pool, so total worker threads stay at the resolved local width. Sized to
    // `state.fit_pool_width` (the scheduling heuristic) when set, else `state.n_jobs` — see
    // `run_fit_pool`'s doc for why the two are kept separate.
    let run = || {
        fit_model_ambient_bagged(
            &state,
            &columns,
            &y,
            weight.as_deref(),
            exposure.as_deref(),
            feature_names,
            class_labels,
            &monotone_map,
            cat_x.as_deref(),
            es_holdout.as_deref(),
            bag_groups.as_deref(),
        )
    };
    run_fit_pool(state.n_jobs, state.fit_pool_width, run)
}

/// Fit a single-output [`Model`] on the **ambient** rayon pool (no pool of its own): bin the
/// design (numeric grids, or native-categorical grids+encoders), run the booster, then stamp
/// feature names / class labels and validate. The caller owns the thread budget — either
/// [`fit_owned`] wrapping this in one `n_jobs` pool, or [`PyBooster::fit_prune_folds`] running
/// several of these as parallel tasks inside one shared pool. Byte-identical to the old inline
/// body of `fit_owned` (same bin/fit/validate order); only the pool ownership moved out.
#[allow(clippy::too_many_arguments)]
fn fit_model_ambient(
    state: &PyBooster,
    columns: &[Vec<f32>],
    y: &[f32],
    weight: Option<&[f32]>,
    exposure: Option<&[f32]>,
    feature_names: Option<Vec<String>>,
    class_labels: Option<Vec<String>>,
    monotone_map: &MonotoneMap,
    cat_x: Option<&[Vec<String>]>,
    // Caller-supplied honest ES holdout (e.g. the group-aware carve for panel data threaded
    // down from `fit()`/`fit_prune_reports_owned`'s per-fold gather). When `Some`, it wins over
    // and skips the internal ordered-cat-leakage derivation below — it already IS the shared
    // mask both the boost loop and (when cat_x is present) the encoder-blinding binning step
    // need. `None` is byte-identical to before this parameter existed.
    es_holdout: Option<&[bool]>,
) -> Result<Model, PbError> {
    fit_model_ambient_bagged(
        state,
        columns,
        y,
        weight,
        exposure,
        feature_names,
        class_labels,
        monotone_map,
        cat_x,
        es_holdout,
        None,
    )
}

/// [`fit_model_ambient`] with GROUP-AWARE outer bags: `bag_groups` (group id per row) makes
/// every bag a subsample of whole groups (`FitSpec::bag_groups`), so the out-of-bag rows the
/// cell refit and the prune guard read are honest on panel data. `None` is the row draw.
#[allow(clippy::too_many_arguments)]
fn fit_model_ambient_bagged(
    state: &PyBooster,
    columns: &[Vec<f32>],
    y: &[f32],
    weight: Option<&[f32]>,
    exposure: Option<&[f32]>,
    feature_names: Option<Vec<String>>,
    class_labels: Option<Vec<String>>,
    monotone_map: &MonotoneMap,
    cat_x: Option<&[Vec<String>]>,
    es_holdout: Option<&[bool]>,
    bag_groups: Option<&[u32]>,
) -> Result<Model, PbError> {
    let n_numeric = columns.len();
    let loss = state.objective.instantiate()?;
    // One-honest-holdout (design/ordered-ts-early-stopping.md): ordered target statistics with
    // internal early stopping carve the validation slice FIRST so the encoders can be blinded
    // to it; the same deterministic mask drives the boost loop via `FitSpec::fixed_holdout`.
    let honest_holdout: Option<Vec<bool>> = if let Some(mask) = es_holdout {
        if mask.len() != y.len() {
            return Err(PbError::ShapeMismatch {
                what: format!("es_holdout len {} != y len {}", mask.len(), y.len()),
            });
        }
        Some(mask.to_vec())
    } else if cat_x.is_some()
        && matches!(state.cat_config.leakage, LeakageScheme::Ordered { .. })
        && state.config.validation_fraction.is_some()
    {
        let n_rows = u32::try_from(y.len()).map_err(|_| PbError::InvalidInput {
            what: "more than u32::MAX rows is out of scope for v1".into(),
        })?;
        // This mask flows to BOTH `FitSpec::fixed_holdout` (below) and the encoder-blinding
        // binning step via one shared variable, which is what makes it "honest" — but passing
        // `fixed_holdout` bypasses `fit_single`'s own automatic ES stratification (it only
        // derives `strata` when `fixed_holdout` is None), so we must reproduce that rule here
        // ourselves via the SAME shared helper `fit_single` uses (`es_strata_for_loss`), or this
        // branch would silently regress to an unstratified ES holdout for exactly the objectives
        // that most need it (Logistic, and — since 2026-07-20 — Poisson/Tweedie on zero-dominated
        // data). Every other objective keeps the original, unstratified carve (`strata: None`).
        let strata: Option<Vec<u32>> =
            t_boost_core::engine::boost::es_strata_for_loss(loss.as_loss().objective_tag().loss, y);
        t_boost_core::engine::boost::holdout_mask(
            n_rows,
            state.config.validation_fraction,
            state.seed,
            strata.as_deref(),
        )?
    } else {
        None
    };
    let spec = FitSpec {
        loss: loss.as_loss(),
        weight,
        exposure,
        monotone: monotone_map.clone(),
        interaction: state.interaction.clone(),
        credibility: state.credibility,
        fixed_holdout: honest_holdout.as_deref(),
        bag_groups,
        seed: state.seed,
    };
    // `Some(v)` after the match iff `cat_x` was present — `v[j]` is raw categorical feature
    // `j`'s emitted channel ids, in raw-feature order, needed to expand a caller-supplied
    // `feature_names` (always one name per INPUT column, never per axis) to match once any
    // extra channel is admitted (see `expand_feature_names_for_cat_channels`).
    let mut cat_axes_per_raw: Option<Vec<Vec<TsEncodingId>>> = None;
    let mut model = match cat_x {
        None => {
            let refs: Vec<&[f32]> = columns.iter().map(Vec::as_slice).collect();
            let x = bin_columns(&refs, weight, &state.bin_config, state.seed)?;
            Booster::with_config(state.config.clone()).fit(&x, y, &spec)?
        }
        // Native categorical path: numeric columns keep raw ids `0..n_numeric`, each
        // categorical column gets a sequential raw id after them, so the serve-time
        // `bin_serve_columns` re-aligns by raw without any extra index bookkeeping.
        Some(cats) => {
            let numeric = columns
                .iter()
                .enumerate()
                .map(|(i, values)| {
                    Ok::<_, PbError>(NumericColumn {
                        raw: FeatureId(raw_id(i)?),
                        values,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let cat_cols = cats
                .iter()
                .enumerate()
                .map(|(j, levels)| {
                    let raw = FeatureId(raw_id(n_numeric + j)?);
                    // P1 multi-channel: extra axes (same raw, distinct encoder ids) when
                    // `cat_channels` requested them — `None` (the default) emits exactly the
                    // one mean-TS column as before. `n_classes: None` keeps the P3 per-class
                    // channels out of the single-output path, where a label-mean TS is a
                    // genuine target mean, not an ordinal-over-nominal artifact.
                    let ids = categorical_channel_ids(
                        levels,
                        weight,
                        exposure,
                        state.cat_channels,
                        &state.cat_config,
                        state.cat_count_min_levels,
                        state.cat_class_freq_min_levels,
                        None,
                    )?;
                    let cols = ids
                        .iter()
                        .map(|&id| {
                            Ok::<_, PbError>(CategoricalColumn {
                                raw,
                                id,
                                levels,
                                config: cat_channel_config(
                                    id,
                                    &state.cat_config,
                                    state.cat_count_config.as_ref(),
                                    &[],
                                )?,
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok::<_, PbError>((ids, cols))
                })
                .collect::<Result<Vec<(Vec<TsEncodingId>, Vec<_>)>, _>>()?;
            let axes: Vec<Vec<TsEncodingId>> =
                cat_cols.iter().map(|(ids, _)| ids.clone()).collect();
            // The monotone map is keyed by INPUT column but resolved against the AXIS array, so
            // it must be expanded with the SAME per-raw channel counts the assembly just
            // produced — exactly as `feature_names` is below. Without this every constraint at
            // or after the first multi-channel categorical is silently misaimed.
            let cat_spec = FitSpec {
                monotone: expand_monotone_map_for_cat_channels(&spec.monotone, n_numeric, &axes)?,
                ..spec
            };
            cat_axes_per_raw = Some(axes);
            let categorical = cat_cols
                .into_iter()
                .flat_map(|(_, cols)| cols)
                .collect::<Vec<_>>();
            let fitted = bin_train_columns_with_holdout(
                &numeric,
                &categorical,
                y,
                weight,
                exposure,
                &state.bin_config,
                state.seed,
                honest_holdout.as_deref(),
            )?;
            Booster::with_config(state.config.clone()).fit_train(
                &fitted.train,
                y,
                &cat_spec,
                fitted.cat_encoders,
            )?
        }
    };
    if let Some(names) = feature_names {
        // `names` is one entry per INPUT column (never per axis) — expand it to match the
        // actual axis-indexed schema whenever any categorical feature's count channel was
        // admitted, using the SAME per-raw-feature axis counts the assembly above just
        // produced (never re-derived, so this can't disagree with the real column layout).
        let names = match &cat_axes_per_raw {
            Some(axes) => expand_feature_names_for_cat_channels(names, n_numeric, axes)?,
            None => names,
        };
        if names.len() != model.schema.feature_names.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "feature_names len {} != n_features {}",
                    names.len(),
                    model.schema.feature_names.len()
                ),
            });
        }
        model.schema.feature_names = names;
    }
    model.schema.class_labels = class_labels;
    model.validate()?;
    Ok(model)
}

/// K>=3 owned fit. `bag_groups` (group id per row) makes every outer bag a subsample of whole
/// groups (`FitSpec::bag_groups`; see [`fit_model_ambient_bagged`]); `None` draws rows.
#[allow(clippy::too_many_arguments)]
fn fit_multiclass_owned_bagged(
    state: PyBooster,
    columns: Vec<Vec<f32>>,
    y: Vec<f32>,
    n_classes: usize,
    class_labels: Vec<String>,
    weight: Option<Vec<f32>>,
    feature_names: Option<Vec<String>>,
    monotone: Option<Vec<i8>>,
    cat_x: Option<Vec<Vec<String>>>,
    es_holdout: Option<Vec<bool>>,
    bag_groups: Option<Vec<u32>>,
) -> Result<MultiClassModel, PbError> {
    let n_numeric = columns.len();
    let n_cat = cat_x.as_ref().map_or(0, Vec::len);
    let monotone_map = build_monotone_map(monotone.as_deref(), n_numeric + n_cat)?;
    if let Some(mask) = es_holdout.as_deref() {
        if mask.len() != y.len() {
            return Err(PbError::ShapeMismatch {
                what: format!("es_holdout len {} != y len {}", mask.len(), y.len()),
            });
        }
    }
    if cat_x.is_some()
        && state.config.validation_fraction.is_some()
        && !matches!(state.cat_config.leakage, LeakageScheme::KFold { .. })
    {
        return Err(PbError::InvalidConfig {
            what: "multiclass validation_fraction with native categoricals requires \
                   cat_leakage='kfold' (the default); the ordered one-honest-holdout path is \
                   single-output-only in v1 — use 'kfold' or an external validation split"
                .into(),
        });
    }
    let honest_holdout = if es_holdout.is_some() {
        es_holdout
    } else if cat_x.is_some() && state.config.validation_fraction.is_some() {
        let labels = y
            .iter()
            .map(|&value| {
                if !value.is_finite()
                    || value < 0.0
                    || value.fract() != 0.0
                    || value as usize >= n_classes
                {
                    return Err(PbError::InvalidInput {
                        what: "invalid multiclass target index".into(),
                    });
                }
                Ok(value as u32)
            })
            .collect::<Result<Vec<_>, PbError>>()?;
        t_boost_core::engine::boost::holdout_mask(
            u32::try_from(y.len()).map_err(|_| PbError::InvalidInput {
                what: "more than u32::MAX rows is out of scope for v1".into(),
            })?,
            state.config.validation_fraction,
            state.seed,
            Some(&labels),
        )?
    } else {
        None
    };
    let admission_weight = match (&honest_holdout, &weight) {
        (Some(mask), Some(w)) => {
            if w.len() != mask.len() {
                return Err(PbError::ShapeMismatch {
                    what: "weight and holdout lengths differ".into(),
                });
            }
            Some(
                w.iter()
                    .zip(mask)
                    .filter_map(|(&w, &held)| (!held).then_some(w))
                    .collect::<Vec<_>>(),
            )
        }
        _ => None,
    };
    // See `fit_model_ambient`'s identical comment: `Some(n)` iff `cat_x` was present, `n[j]` =
    // raw categorical feature `j`'s axis count, needed to expand a caller-supplied
    // `feature_names` once any feature's count channel is admitted. Threaded out through the
    // closure's own return value (not an outer-variable mutation) since `run_fit_pool` may run
    // `run` on a freshly built rayon pool via `ThreadPool::install`.
    type RunOutput = (MultiClassModel, Option<Vec<Vec<TsEncodingId>>>);
    let run = || -> Result<RunOutput, PbError> {
        // The softmax gradient is coupled across classes and cannot be a scalar `Loss`; the engine
        // ignores `spec.loss` on the multiclass path. A dummy keeps the shared `FitSpec` shape.
        let dummy = SquaredError;
        let spec = FitSpec {
            loss: &dummy,
            weight: weight.as_deref(),
            exposure: None,
            monotone: monotone_map.clone(),
            interaction: state.interaction.clone(),
            credibility: state.credibility,
            // Caller-supplied honest ES holdout (e.g. a GROUP-aware carve for panel data, where
            // the default row-level carve leaks near-duplicate rows into validation and defeats
            // early stopping — measured running to the 4000-tree cap on grouped multiclass).
            // None keeps the engine's own row-level carve, byte-identical to before this arg.
            fixed_holdout: honest_holdout.as_deref(),
            bag_groups: bag_groups.as_deref(),
            seed: state.seed,
        };
        match &cat_x {
            None => {
                let refs: Vec<&[f32]> = columns.iter().map(Vec::as_slice).collect();
                let x = bin_columns(&refs, weight.as_deref(), &state.bin_config, state.seed)?;
                let model = Booster::with_config(state.config.clone()).fit_multiclass(
                    &x,
                    &y,
                    n_classes,
                    &class_labels,
                    &spec,
                )?;
                Ok((model, None))
            }
            Some(cats) => {
                let numeric = columns
                    .iter()
                    .enumerate()
                    .map(|(i, values)| {
                        Ok::<_, PbError>(NumericColumn {
                            raw: FeatureId(raw_id(i)?),
                            values,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                // P3 per-class frequency channels (design/multichannel-categoricals.md §8):
                // one `CatTarget::ClassFreq { class }` config per class, built once and shared
                // by every categorical column. Empty unless `cat_channels` asked for them, in
                // which case `categorical_channel_ids` emits no id that would index this.
                let class_freq_configs: Vec<TsConfig> = match &state.cat_class_freq_config {
                    None => Vec::new(),
                    Some(template) => (0..n_classes)
                        .map(|class| {
                            let cfg = TsConfig {
                                target: CatTarget::ClassFreq {
                                    class: u32::try_from(class).map_err(|_| PbError::Internal {
                                        what: "class index exceeded u32".into(),
                                    })?,
                                },
                                ..template.clone()
                            };
                            cfg.validate()?;
                            Ok::<_, PbError>(cfg)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                };
                let cat_cols = cats
                    .iter()
                    .enumerate()
                    .map(|(j, levels)| {
                        let raw = FeatureId(raw_id(n_numeric + j)?);
                        // P1/P3 multi-channel: mirrors the single-output `fit_owned` path
                        // above, plus the per-class channels (`n_classes` is `Some` here — this
                        // IS the softmax path the ordinal label-mean defect lives on). No
                        // exposure on the multiclass path (matches this function's
                        // `FitSpec { exposure: None, .. }` a few lines up).
                        let permitted_levels = honest_holdout.as_ref().map(|mask| {
                            levels
                                .iter()
                                .zip(mask)
                                .filter_map(|(level, &held)| (!held).then_some(level.clone()))
                                .collect::<Vec<_>>()
                        });
                        let ids = categorical_channel_ids(
                            permitted_levels.as_deref().unwrap_or(levels),
                            admission_weight.as_deref().or(weight.as_deref()),
                            None,
                            state.cat_channels,
                            &state.cat_config,
                            state.cat_count_min_levels,
                            state.cat_class_freq_min_levels,
                            Some(n_classes),
                        )?;
                        let cols = ids
                            .iter()
                            .map(|&id| {
                                Ok::<_, PbError>(CategoricalColumn {
                                    raw,
                                    id,
                                    levels,
                                    config: cat_channel_config(
                                        id,
                                        &state.cat_config,
                                        state.cat_count_config.as_ref(),
                                        &class_freq_configs,
                                    )?,
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok::<_, PbError>((ids, cols))
                    })
                    .collect::<Result<Vec<(Vec<TsEncodingId>, Vec<_>)>, _>>()?;
                let cat_axes_per_raw: Vec<Vec<TsEncodingId>> =
                    cat_cols.iter().map(|(ids, _)| ids.clone()).collect();
                // Same input-index -> axis-index expansion as the single-output path above.
                let cat_spec = FitSpec {
                    monotone: expand_monotone_map_for_cat_channels(
                        &spec.monotone,
                        n_numeric,
                        &cat_axes_per_raw,
                    )?,
                    ..spec
                };
                let categorical = cat_cols
                    .into_iter()
                    .flat_map(|(_, cols)| cols)
                    .collect::<Vec<_>>();
                let fitted = bin_train_columns_with_holdout(
                    &numeric,
                    &categorical,
                    &y,
                    weight.as_deref(),
                    None,
                    &state.bin_config,
                    state.seed,
                    honest_holdout.as_deref(),
                )?;
                let model = Booster::with_config(state.config.clone()).fit_multiclass_train(
                    &fitted.train,
                    &y,
                    n_classes,
                    &class_labels,
                    &cat_spec,
                    fitted.cat_encoders,
                )?;
                Ok((model, Some(cat_axes_per_raw)))
            }
        }
    };
    let (mut model, cat_axes_per_raw) = run_fit_pool(state.n_jobs, state.fit_pool_width, run)?;
    if let Some(names) = feature_names {
        // Expand ONCE (all classes share the same design/axis layout), then clone per class —
        // mirrors the pre-existing `names.clone()` per-class pattern below.
        let names = match &cat_axes_per_raw {
            Some(axes) => expand_feature_names_for_cat_channels(names, n_numeric, axes)?,
            None => names,
        };
        for m in &mut model.classes {
            if names.len() != m.schema.feature_names.len() {
                return Err(PbError::ShapeMismatch {
                    what: format!(
                        "feature_names len {} != n_features {}",
                        names.len(),
                        m.schema.feature_names.len()
                    ),
                });
            }
            m.schema.feature_names = names.clone();
        }
    }
    model.validate()?;
    Ok(model)
}

/// Dev-only stage timer for the K>=3 prune pipeline: prints `[mc-prune] <stage> <secs>` to
/// stderr when `TBOOST_PROFILE` is set, mirroring the core's `prof` breakdown for one fit.
fn mc_stage<T>(name: &str, f: impl FnOnce() -> T) -> T {
    let on = std::env::var_os("TBOOST_PROFILE").is_some();
    let t = std::time::Instant::now();
    let out = f();
    if on {
        eprintln!("[mc-prune] {name} {:.2}s", t.elapsed().as_secs_f64());
    }
    out
}

/// K>=3 fit-and-prune. Two selection regimes share one deploy/guard/apply tail:
///
/// * **CV selection** (`fold_of` given; the default since 2026-09-07): the single-output prune's
///   design ported to K>=3. The DEPLOY soup is fitted first and its realized supports are the
///   candidates; `k_folds` fold models (unbagged unless `sel_bags > 1`, tree cap 2x the deploy's
///   per-bag count, no cell refit) are each scored by `prune_multiclass_to_tables` on THEIR
///   held-out fold, and [`aggregate_prune_selection`] votes the keep-set across folds with the
///   `drop_z` evidence gate and `keep_budget` — every training row is honest evidence exactly
///   once, and the paired per-fold gains give each table a real standard error.
/// * **Legacy split selection** (`fold_of == None`): one selection fit on the complement of
///   `sel_rows`, walked on that slice. Kept for A/B and for the historical artifacts; it is the
///   path measured to drop tables on noise (see `design/multiclass-design.md` §9g).
///
/// The guard then measures the shipped keep-set against the full deploy bank on honest rows
/// with an SE-aware bar (`guard_z`, `guard_floor`), and the keep-set is retained on the deploy
/// banks with the box/table budgets.
#[allow(clippy::too_many_arguments)]
fn fit_multiclass_pruned_owned(
    mut state: PyBooster,
    columns: Vec<Vec<f32>>,
    y: Vec<f32>,
    n_classes: usize,
    class_labels: Vec<String>,
    sel_rows: Vec<usize>,
    weight: Option<Vec<f32>>,
    feature_names: Option<Vec<String>>,
    monotone: Option<Vec<i8>>,
    cat_x: Option<Vec<Vec<String>>>,
    w_measure: RefMeasure,
    se_rule: f64,
    lambda_boxes: f64,
    n_folds: usize,
    es_holdout: Option<Vec<bool>>,
    deploy_es_holdout: Option<Vec<bool>>,
    prune_guard: bool,
    prune_guard_tol: f64,
    prune_guard_min_rows: usize,
    guard_oob_honest: bool,
    sel_bags: u16,
    box_budget: usize,
    lambda_tables: f64,
    table_budget: usize,
    table_min_arity: u8,
    fold_of: Option<Vec<i64>>,
    k_folds: usize,
    fold_es_holdout: Option<Vec<bool>>,
    fold_es_patience: Option<u32>,
    min_stability: f64,
    min_mean_gain: f64,
    drop_z: Option<f64>,
    keep_budget: usize,
    guard_z: f64,
    guard_floor: f64,
    bag_groups: Option<Vec<u32>>,
    ranked_path: bool,
    path_steps: usize,
    path_fraction: f64,
    band_tolerance: Option<f64>,
    band_deviance_cap: f64,
    path_tolerance: f64,
) -> Result<(MultiClassTableModel, String), PbError> {
    let n = y.len();
    if n == 0 {
        return Err(PbError::InvalidInput {
            what: "multiclass fit-and-prune requires at least one row".into(),
        });
    }
    for (axis, column) in columns.iter().enumerate() {
        if column.len() != n {
            return Err(PbError::ShapeMismatch {
                what: format!("x column {axis} len {} != y len {n}", column.len()),
            });
        }
    }
    if let Some(w) = weight.as_deref() {
        if w.len() != n {
            return Err(PbError::ShapeMismatch {
                what: format!("weight len {} != y len {n}", w.len()),
            });
        }
    }
    if let Some(cats) = cat_x.as_deref() {
        for (axis, column) in cats.iter().enumerate() {
            if column.len() != n {
                return Err(PbError::ShapeMismatch {
                    what: format!(
                        "categorical column {axis} len {} != y len {n}",
                        column.len()
                    ),
                });
            }
        }
    }
    let cv = fold_of.is_some();
    if let Some(f) = fold_of.as_deref() {
        if f.len() != n {
            return Err(PbError::ShapeMismatch {
                what: format!("fold_of len {} != n_rows {n}", f.len()),
            });
        }
        if k_folds < 2 {
            return Err(PbError::InvalidInput {
                what: format!("k_folds must be >= 2 for CV selection, got {k_folds}"),
            });
        }
        if f.iter().any(|&v| v < 0 || v >= k_folds as i64) {
            return Err(PbError::InvalidInput {
                what: format!(
                    "fold_of values must be in 0..k_folds ({k_folds}); an out-of-range value \
                     matches no fold id and would silently train on every fold, held out by none"
                ),
            });
        }
        if let Some(mask) = fold_es_holdout.as_deref() {
            if mask.len() != n {
                return Err(PbError::ShapeMismatch {
                    what: format!("fold_es_holdout len {} != n_rows {n}", mask.len()),
                });
            }
        }
    } else if sel_rows.is_empty() || sel_rows.len() >= n {
        return Err(PbError::InvalidInput {
            what: format!(
                "selection rows must be non-empty and leave training data (got {} of {n})",
                sel_rows.len()
            ),
        });
    }
    let mut is_selected = vec![false; n];
    for &row in &sel_rows {
        let selected = is_selected
            .get_mut(row)
            .ok_or_else(|| PbError::InvalidInput {
                what: format!("selection row {row} out of range for {n} rows"),
            })?;
        if *selected {
            return Err(PbError::InvalidInput {
                what: format!("selection row {row} appears more than once"),
            });
        }
        *selected = true;
    }
    let labels: Vec<u32> = y
        .iter()
        .enumerate()
        .map(|(row, &label)| {
            if !label.is_finite()
                || label < 0.0
                || label.fract() != 0.0
                || label as usize >= n_classes
            {
                return Err(PbError::InvalidInput {
                    what: format!(
                        "multiclass label[{row}] must be an integer in 0..{n_classes}, got {label}"
                    ),
                });
            }
            Ok(label as u32)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let n_jobs = state.n_jobs;
    let fit_pool_width = state.fit_pool_width;
    state.n_jobs = None;
    state.fit_pool_width = None;
    run_fit_pool(n_jobs, fit_pool_width, move || {
        let full_weight = weight.clone().unwrap_or_else(|| vec![1.0_f32; n]);
        let cfg = PruneConfig {
            se_rule,
            lambda_boxes,
            lambda_tables,
            table_price_min_arity: table_min_arity,
        };
        // The selection fits share the deploy recipe minus its ensemble: unbagged (or
        // `sel_bags` bags) and without the out-of-bag cell refit, which needs a bag partition.
        let selection_state = |base: &PyBooster| -> PyBooster {
            let mut st = base.clone();
            st.config.boosters.ensemble = if sel_bags <= 1 {
                EnsembleSpec::Off
            } else {
                EnsembleSpec::OuterBag {
                    n_bags: sel_bags,
                    bag_subsample: match &base.config.boosters.ensemble {
                        EnsembleSpec::OuterBag { bag_subsample, .. } => *bag_subsample,
                        _ => 1.0,
                    },
                    cell_refit: None,
                }
            };
            st
        };
        let deploy_es_holdout_mask = deploy_es_holdout.clone();
        let fit_deploy = |st: PyBooster| -> Result<MultiClassModel, PbError> {
            fit_multiclass_owned_bagged(
                st,
                columns.clone(),
                y.clone(),
                n_classes,
                class_labels.clone(),
                weight.clone(),
                feature_names.clone(),
                monotone.clone(),
                cat_x.clone(),
                deploy_es_holdout.clone(),
                bag_groups.clone(),
            )
        };
        let mut ranked_used = false;
        let (mut report, deploy_model, selection_json): (
            PruneReport,
            MultiClassModel,
            Option<serde_json::Value>,
        ) = if ranked_path {
            // RANKED PATH: one deploy fit, supports ranked by purified variance under heredity,
            // prefixes scored on the per-class out-of-bag juries, the minimum deployed. No fold
            // CV, no guard (see `multiclass_ranked_path`).
            let deploy_model = mc_stage("deploy_fit", || fit_deploy(state.clone()))?;
            let deploy_first = deploy_model
                .classes
                .first()
                .ok_or_else(|| PbError::Internal {
                    what: "multiclass deploy fit produced no classes".into(),
                })?;
            let serve = ServeBinnedMatrix(serve_binned_for_model(
                deploy_first,
                &columns,
                cat_x.as_deref(),
            )?);
            let supports = mc_stage("deploy_supports", || {
                multiclass_realized_supports(&deploy_model, &serve, w_measure.clone(), &full_weight)
            })?;
            let sel = mc_stage("ranked_path", || {
                multiclass_ranked_path(
                    &deploy_model,
                    &serve,
                    &labels,
                    &full_weight,
                    w_measure.clone(),
                    &supports,
                    path_steps,
                    path_fraction,
                    path_tolerance,
                    prune_guard_min_rows,
                )
            })?;
            drop(serve);
            let (keep, path_json) = match sel {
                Some((keep, path)) => (
                    keep,
                    serde_json::to_value(&path).map_err(|err| {
                        PbError::Serialization(format!("could not serialize path report: {err}"))
                    })?,
                ),
                None => (
                    supports.iter().map(|(u, _)| u.clone()).collect(),
                    serde_json::json!({ "skipped": "no out-of-bag evidence: full bank kept" }),
                ),
            };
            ranked_used = true;
            let kept_set: std::collections::BTreeSet<FeatureSet> = keep.iter().cloned().collect();
            let table_scores: Vec<PruneTableScore> = supports
                .iter()
                .map(|(u, v)| PruneTableScore {
                    u: u.clone(),
                    order: u8::try_from(u.order()).unwrap_or(u8::MAX),
                    mean_gain: *v,
                    se_gain: 0.0,
                    variance: *v,
                    sticky: u.order() == 1,
                    selected: kept_set.contains(u),
                })
                .collect();
            let effective_order =
                u8::try_from(keep.iter().map(FeatureSet::order).max().unwrap_or(0))
                    .unwrap_or(u8::MAX);
            (
                PruneReport {
                    kept: keep,
                    dropped: Vec::new(),
                    effective_order,
                    path: Vec::new(),
                    table_scores,
                    delta_vs_full: 0.0,
                },
                deploy_model,
                Some(path_json),
            )
        } else if let Some(fold_of) = fold_of.as_deref() {
            // ---- CV selection: deploy first, its realized supports are the candidates.
            let deploy_model = mc_stage("deploy_fit", || fit_deploy(state.clone()))?;
            let deploy_first = deploy_model
                .classes
                .first()
                .ok_or_else(|| PbError::Internal {
                    what: "multiclass deploy fit produced no classes".into(),
                })?;
            let deploy_serve = ServeBinnedMatrix(serve_binned_for_model(
                deploy_first,
                &columns,
                cat_x.as_deref(),
            )?);
            let full_supports = mc_stage("deploy_supports", || {
                multiclass_realized_supports(
                    &deploy_model,
                    &deploy_serve,
                    w_measure.clone(),
                    &full_weight,
                )
            })?;
            drop(deploy_serve);
            let full_ids: Vec<Vec<u32>> = full_supports
                .iter()
                .map(|(u, _)| feature_set_ids(u))
                .collect();
            let n_bags = match &state.config.boosters.ensemble {
                EnsembleSpec::OuterBag { n_bags, .. } => usize::from(*n_bags).max(1),
                _ => 1,
            };
            let trees_per_bag = deploy_first.trees.len().div_ceil(n_bags).max(1);
            let mut fold_state = selection_state(&state);
            fold_state.config.n_trees = fold_state
                .config
                .n_trees
                .min(u32::try_from((2 * trees_per_bag).max(64)).unwrap_or(u32::MAX));
            if let Some(p) = fold_es_patience {
                fold_state.config.early_stopping_rounds = p;
            }
            let fold_task = |k: usize| -> Result<Option<PruneReport>, PbError> {
                let k_i = k as i64;
                let mut held: Vec<usize> = Vec::new();
                let mut train: Vec<usize> = Vec::new();
                for (i, &f) in fold_of.iter().enumerate() {
                    if f == k_i {
                        held.push(i);
                    } else {
                        train.push(i);
                    }
                }
                if held.is_empty() || train.is_empty() {
                    return Ok(None);
                }
                let train_cols: Vec<Vec<f32>> = columns
                    .iter()
                    .map(|c| gather_f32(c, &train))
                    .collect::<Result<_, _>>()?;
                let train_y = gather_f32(&y, &train)?;
                let train_w = weight
                    .as_deref()
                    .map(|w| gather_f32(w, &train))
                    .transpose()?;
                let train_cat = cat_x
                    .as_deref()
                    .map(|cats| {
                        cats.iter()
                            .map(|col| gather_str(col, &train))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .transpose()?;
                let train_es = fold_es_holdout
                    .as_deref()
                    .map(|mask| gather_bool(mask, &train))
                    .transpose()?;
                let train_groups = bag_groups
                    .as_deref()
                    .map(|g| gather_groups(g, &train))
                    .transpose()?;
                let fold_model = mc_stage("fold_fit", || {
                    fit_multiclass_owned_bagged(
                        fold_state.clone(),
                        train_cols,
                        train_y,
                        n_classes,
                        class_labels.clone(),
                        train_w,
                        feature_names.clone(),
                        monotone.clone(),
                        train_cat,
                        train_es,
                        train_groups,
                    )
                })?;
                let fold_first = fold_model
                    .classes
                    .first()
                    .ok_or_else(|| PbError::Internal {
                        what: "multiclass fold fit produced no classes".into(),
                    })?;
                let serve = ServeBinnedMatrix(serve_binned_for_model(
                    fold_first,
                    &columns,
                    cat_x.as_deref(),
                )?);
                let (_tm, report) = mc_stage("fold_walk", || {
                    prune_multiclass_to_tables(
                        &fold_model,
                        &serve,
                        &labels,
                        &full_weight,
                        w_measure.clone(),
                        &held,
                        n_folds,
                        &cfg,
                    )
                })?;
                Ok(Some(report))
            };
            let reports: Vec<Option<PruneReport>> = mc_stage("folds_total", || {
                (0..k_folds)
                    .into_par_iter()
                    .map(fold_task)
                    .collect::<Result<_, _>>()
            })?;
            let (keep_ids, sel_json) = aggregate_prune_selection(
                full_ids,
                &reports,
                min_stability,
                min_mean_gain,
                drop_z,
                keep_budget,
                false,
            )?;
            let kept: Vec<FeatureSet> = keep_ids
                .iter()
                .map(|ids| feature_set_from_raw_ids(ids))
                .collect();
            let kept_set: BTreeSet<FeatureSet> = kept.iter().cloned().collect();
            // Per-support fold evidence, summarised for the guard's re-admission ranking and the
            // deploy report: mean gain across the folds that realized the support, with the SE
            // of that mean (the paired-fold standard error), and the deploy bank's variance.
            let mut gains: BTreeMap<FeatureSet, Vec<f64>> = BTreeMap::new();
            for r in reports.iter().flatten() {
                for ts in &r.table_scores {
                    gains.entry(ts.u.clone()).or_default().push(ts.mean_gain);
                }
            }
            let table_scores: Vec<PruneTableScore> = full_supports
                .iter()
                .map(|(u, variance)| {
                    let (m, se) = mean_se_of(gains.get(u).map(Vec::as_slice).unwrap_or(&[]));
                    PruneTableScore {
                        u: u.clone(),
                        order: u.order() as u8,
                        mean_gain: m,
                        se_gain: se.unwrap_or(f64::NAN),
                        variance: *variance,
                        sticky: u.order() == 1,
                        selected: kept_set.contains(u),
                    }
                })
                .collect();
            let dropped: Vec<FeatureSet> = full_supports
                .iter()
                .map(|(u, _)| u.clone())
                .filter(|u| !kept_set.contains(u))
                .collect();
            let effective_order = kept.iter().map(|u| u.order() as u8).max().unwrap_or(0);
            let report = PruneReport {
                kept,
                dropped,
                effective_order,
                path: Vec::new(),
                table_scores,
                delta_vs_full: f64::NAN,
            };
            let sel_value: serde_json::Value = serde_json::from_str(&sel_json).map_err(|err| {
                PbError::Serialization(format!("could not parse selection report: {err}"))
            })?;
            (report, deploy_model, Some(sel_value))
        } else {
            // ---- Legacy split selection.
            let train_rows: Vec<usize> = is_selected
                .iter()
                .enumerate()
                .filter_map(|(row, &selected)| (!selected).then_some(row))
                .collect();
            let train_columns: Vec<Vec<f32>> = columns
                .par_iter()
                .map(|column| gather_f32(column, &train_rows))
                .collect::<Result<Vec<_>, _>>()?;
            let train_y = gather_f32(&y, &train_rows)?;
            let train_weight = weight
                .as_deref()
                .map(|w| gather_f32(w, &train_rows))
                .transpose()?;
            let train_cat_x = cat_x
                .as_deref()
                .map(|cats| {
                    cats.par_iter()
                        .map(|column| gather_str(column, &train_rows))
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?;
            if let Some(mask) = es_holdout.as_deref() {
                if mask.len() != train_rows.len() {
                    return Err(PbError::ShapeMismatch {
                        what: format!(
                            "es_holdout len {} != complement rows {} (mask is over the complement \
                             of sel_rows, in ascending original-row order)",
                            mask.len(),
                            train_rows.len()
                        ),
                    });
                }
            }
            let sel_groups = bag_groups
                .as_deref()
                .map(|g| gather_groups(g, &train_rows))
                .transpose()?;
            let sel_model = fit_multiclass_owned_bagged(
                selection_state(&state),
                train_columns,
                train_y,
                n_classes,
                class_labels.clone(),
                train_weight,
                feature_names.clone(),
                monotone.clone(),
                train_cat_x,
                es_holdout.clone(),
                sel_groups,
            )?;
            let sel_first = sel_model.classes.first().ok_or_else(|| PbError::Internal {
                what: "multiclass fit-and-prune produced no classes".into(),
            })?;
            let sel_serve = ServeBinnedMatrix(serve_binned_for_model(
                sel_first,
                &columns,
                cat_x.as_deref(),
            )?);
            let (_sel_tables, report) = prune_multiclass_to_tables(
                &sel_model,
                &sel_serve,
                &labels,
                &full_weight,
                w_measure.clone(),
                &sel_rows,
                n_folds,
                &cfg,
            )?;
            drop(sel_serve);
            drop(sel_model);
            let deploy_model = fit_deploy(state.clone())?;
            (report, deploy_model, None)
        };
        let deploy_first = deploy_model
            .classes
            .first()
            .ok_or_else(|| PbError::Internal {
                what: "multiclass deploy fit produced no classes".into(),
            })?;
        let deploy_binned = serve_binned_for_model(deploy_first, &columns, cat_x.as_deref())?;
        let deploy_serve = ServeBinnedMatrix(deploy_binned);
        let (shipped_keep, guard) = if prune_guard && !ranked_used {
            let ev = if guard_oob_honest {
                Some(MulticlassGuardEvidence::OutOfBag)
            } else {
                deploy_es_holdout_mask
                    .as_deref()
                    .map(MulticlassGuardEvidence::Carve)
            };
            match ev {
                Some(ev) => {
                    let (k, g) = mc_stage("guard", || {
                        multiclass_prune_guard(
                            &deploy_model,
                            &deploy_serve,
                            &labels,
                            &full_weight,
                            w_measure.clone(),
                            &report.kept,
                            &report.table_scores,
                            ev,
                            prune_guard_tol,
                            prune_guard_min_rows,
                            guard_z,
                            guard_floor,
                        )
                    })?;
                    (k, Some(g))
                }
                None => (
                    report.kept.clone(),
                    Some(MulticlassGuardReport::skipped(
                        true,
                        "no honest evidence: a panel fit with validation_fraction unset",
                    )),
                ),
            }
        } else {
            (report.kept.clone(), None)
        };
        let evidence: BTreeMap<FeatureSet, f64> = report
            .table_scores
            .iter()
            .map(|ts| (ts.u.clone(), ts.mean_gain))
            .collect();
        let w_measure_band = w_measure.clone();
        let (table_model, box_report, table_report) = mc_stage("keepset_apply", || {
            prune_multiclass_to_keepset_budgeted(
                &deploy_model,
                &deploy_serve,
                &labels,
                &full_weight,
                w_measure,
                &shipped_keep,
                box_budget,
                table_budget,
                table_min_arity,
                &evidence,
            )
        })?;
        // BANDING (see `band_multiclass_table_model`): after the keep-set, before the report.
        let (table_model, banding_reports) = match band_tolerance {
            Some(tol) if multiclass_bag_oob_evidence_available(&deploy_model) => {
                let (tm, reps) = mc_stage("banding", || {
                    band_multiclass_table_model(
                        &table_model,
                        &deploy_model,
                        &deploy_serve,
                        &columns,
                        cat_x.as_deref(),
                        &full_weight,
                        w_measure_band.clone(),
                        &shipped_keep,
                        tol,
                        &labels,
                        band_deviance_cap,
                    )
                })?;
                (tm, Some(reps))
            }
            _ => (table_model, None),
        };
        report.kept = shipped_keep;
        let mut report_value = serde_json::to_value(&report).map_err(|err| {
            PbError::Serialization(format!("could not serialize PruneReport: {err}"))
        })?;
        if let Some(obj) = report_value.as_object_mut() {
            obj.insert(
                "selector".into(),
                serde_json::Value::String(if ranked_used {
                    "ranked_path".into()
                } else if cv {
                    "cv_fold_vote".into()
                } else {
                    "single_split_walk".into()
                }),
            );
            obj.insert(
                "cv_folds".into(),
                serde_json::json!(if cv { k_folds } else { 0 }),
            );
            if let Some(sel) = selection_json {
                obj.insert("selection".into(), sel);
            }
            let mut g = match &guard {
                Some(g) => serde_json::to_value(g).map_err(|err| {
                    PbError::Serialization(format!("could not serialize guard report: {err}"))
                })?,
                None => serde_json::json!({ "enabled": false, "fired": false }),
            };
            if (box_report.engaged || table_report.engaged) && guard.is_some() {
                if let Some(obj) = g.as_object_mut() {
                    obj.insert("budget_blind".into(), serde_json::Value::Bool(true));
                }
            }
            obj.insert("guard".into(), g);
            if let Some(reps) = &banding_reports {
                obj.insert(
                    "banding".into(),
                    serde_json::to_value(reps).map_err(|err| {
                        PbError::Serialization(format!("could not serialize banding report: {err}"))
                    })?,
                );
            }
            if let Some(cr) = deploy_model.cell_refit {
                obj.insert(
                    "cell_refit".into(),
                    serde_json::json!({
                        "lambda": cr.lambda,
                        "declined": cr.declined(),
                        "n_supports": cr.n_supports,
                        "n_blocked": cr.n_blocked,
                        "reachable_coverage": cr.reachable_coverage(),
                        "n_rejected": cr.n_rejected,
                        "guard_rows": cr.guard_rows,
                    }),
                );
            }
            if box_budget > 0 {
                obj.insert(
                    "box_budget".into(),
                    serde_json::to_value(&box_report).map_err(|err| {
                        PbError::Serialization(format!(
                            "could not serialize BoxBudgetReport: {err}"
                        ))
                    })?,
                );
            }
            if table_budget > 0 {
                obj.insert(
                    "table_budget".into(),
                    serde_json::to_value(&table_report).map_err(|err| {
                        PbError::Serialization(format!(
                            "could not serialize TableBudgetReport: {err}"
                        ))
                    })?,
                );
            }
        }
        let report_json = serde_json::to_string(&report_value).map_err(|err| {
            PbError::Serialization(format!("could not serialize PruneReport: {err}"))
        })?;
        Ok((table_model, report_json))
    })
}

fn raw_columns_from_array(x: PyReadonlyArray2<'_, f32>) -> PyResult<Vec<Vec<f32>>> {
    let shape = x.shape();
    let n_rows = *shape
        .first()
        .ok_or_else(|| PyTypeError::new_err("x must be a two-dimensional float32 array"))?;
    let n_features = *shape
        .get(1)
        .ok_or_else(|| PyTypeError::new_err("x must be a two-dimensional float32 array"))?;
    // 0 numeric columns is legal when native categorical columns are supplied (all-categorical
    // input); the total-feature (numeric + categorical) check lives in the core fit/serve.
    if n_features == 0 {
        return Ok(Vec::new());
    }
    // 0 rows: nothing to chunk (and `chunks_exact(0)` would panic below), so short-circuit to
    // n_features empty columns regardless of layout.
    if n_rows == 0 {
        return Ok((0..n_features).map(|_| Vec::new()).collect());
    }
    // F-contiguous (column-major) input needs no transpose: each of the n_features columns is
    // already a contiguous run in the underlying buffer, so splitting the flat buffer into
    // n_features row-count-sized chunks and copying each is a straight memcpy per column — the
    // "costs nothing on ingest" case spec §12.3 promises, and it accepts the layout at all
    // (previously rejected with a TypeError). C-contiguous (row-major, the numpy/pandas default)
    // still needs an actual transpose; that path is unchanged below.
    if x.is_fortran_contiguous() {
        let view = x.as_array();
        let flat = view.as_slice_memory_order().ok_or_else(|| {
            PyTypeError::new_err("x must be a contiguous numpy.ndarray with dtype float32")
        })?;
        return Ok(flat.chunks_exact(n_rows).map(<[f32]>::to_vec).collect());
    }
    let slice = x.as_slice().map_err(|_| {
        PyTypeError::new_err("x must be a C- or F-contiguous numpy.ndarray with dtype float32")
    })?;
    let mut columns: Vec<Vec<f32>> = (0..n_features)
        .map(|_| Vec::with_capacity(n_rows))
        .collect();
    for row in slice.chunks_exact(n_features) {
        for (feature, &value) in row.iter().enumerate() {
            let col = columns.get_mut(feature).ok_or_else(|| {
                PyTypeError::new_err("x row width changed while marshaling the array")
            })?;
            col.push(value);
        }
    }
    Ok(columns)
}

fn array1_to_vec_u32(a: PyReadonlyArray1<'_, u32>, what: &str) -> PyResult<Vec<u32>> {
    a.as_slice()
        .map(<[u32]>::to_vec)
        .map_err(|e| PyValueError::new_err(format!("{what} must be contiguous uint32: {e}")))
}

fn array1_to_vec(x: PyReadonlyArray1<'_, f32>, name: &str) -> PyResult<Vec<f32>> {
    // Non-contiguous 1-D input is a caller-fixable `ValueError` (spec §12.3), not a dtype
    // problem — `TypeError` is reserved for actual dtype mismatches (§12.7).
    let slice = x.as_slice().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be a C-contiguous numpy.ndarray with dtype float32"
        ))
    })?;
    Ok(slice.to_vec())
}

/// Combine an optional per-row sample weight and exposure into the single effective row-mass
/// `weight` that [`Model::explain_weighted`]/[`Model::explain_with_budget_weighted`] (spec
/// §08.7) expect: elementwise product when both are given, pass-through when only one is,
/// `None` (the plain unweighted `explain`) when neither is — so an explain/tables call with no
/// weight or exposure is byte-identical to before this existed. Length/finiteness validation of
/// the combined vector against `x`'s row count is the core's own job (`PbError::ShapeMismatch`/
/// `InvalidInput` surfaced by `explain_weighted` itself), not duplicated here.
/// The per-row mass a bank build under `w_measure` reads (spec §08.7): `weight · exposure`
/// when the measure uses row mass, `None` otherwise — the Python-side twin of
/// `prune::measure_mass` for bindings that receive the weight and exposure separately
/// rather than a loss weight and a log offset.
fn measure_mass_py(
    w_measure: &RefMeasure,
    weight: Option<PyReadonlyArray1<'_, f32>>,
    exposure: Option<PyReadonlyArray1<'_, f32>>,
) -> PyResult<Option<Vec<f32>>> {
    if !w_measure.uses_row_mass() {
        return Ok(None);
    }
    combined_explain_weight(weight, exposure)
}

/// The bank a tables-only model's `tables()` exports: the stored bank, re-centred on the rows of
/// `binned` when a per-row `mass` (call-time weight x exposure) was passed, then re-expressed
/// under `requested` when that names another measure. None of this changes a prediction.
fn export_bank<'a>(
    tm: &'a TableModel,
    binned: Option<&t_boost_core::data::BinnedMatrix>,
    mass: Option<&[f32]>,
    requested: Option<&RefMeasure>,
    measure_floor: f32,
) -> Result<std::borrow::Cow<'a, t_boost_core::explain::TableBank>, PbError> {
    use std::borrow::Cow;
    let rejoint = |bank: &t_boost_core::explain::TableBank| {
        t_boost_core::joint::rejoint(bank, &t_boost_core::joint::JointOptions::default())
    };
    if requested == Some(&RefMeasure::Joint)
        || (requested.is_none() && tm.bank.w == RefMeasure::Joint)
    {
        let binned = binned.ok_or_else(|| PbError::InvalidInput {
            what: "a joint export requires aligned rows".into(),
        })?;
        let base = tm.recentred_bank(
            binned,
            mass,
            RefMeasure::ExposureMarginals {
                floor: measure_floor,
            },
        )?;
        if mass.is_none()
            && tm.bank.tables.iter().any(|table| {
                base.tables
                    .iter()
                    .find(|t| t.u == table.u)
                    .is_none_or(|t| t.support != table.support)
            })
        {
            return Err(PbError::InvalidInput {
                what: "joint export rows do not reproduce stored support; supply explicit weight/exposure (or unit weights for a new unweighted reference)".into(),
            });
        }
        let mut bank = rejoint(&base)?;
        bank.measure_joint_variance(&tm.schema.cat_encoders, binned, mass)?;
        return Ok(Cow::Owned(bank));
    }
    if let Some(mass) = mass {
        let binned = binned.ok_or_else(|| PbError::InvalidInput {
            what: "a call-time weight or exposure needs the rows it weights".into(),
        })?;
        // The joint ledger is export-only: re-centre under the exposure measure, then rejoint.
        let base_w = match requested {
            None => tm.bank.w.clone(),
            Some(RefMeasure::Joint) => RefMeasure::ExposureMarginals {
                floor: measure_floor,
            },
            Some(w) => w.clone(),
        };
        let bank = tm.recentred_bank(binned, Some(mass), base_w)?;
        return Ok(Cow::Owned(match requested {
            Some(RefMeasure::Joint) => rejoint(&bank)?,
            _ => bank,
        }));
    }
    Ok(match requested {
        None => Cow::Borrowed(&tm.bank),
        Some(w) if *w == tm.bank.w => Cow::Borrowed(&tm.bank),
        Some(RefMeasure::Joint) => Cow::Owned(rejoint(&tm.bank)?),
        Some(w) => Cow::Owned(tm.bank.recompute_under(w.clone())?),
    })
}

fn combined_explain_weight(
    weight: Option<PyReadonlyArray1<'_, f32>>,
    exposure: Option<PyReadonlyArray1<'_, f32>>,
) -> PyResult<Option<Vec<f32>>> {
    let w = weight.map(|a| array1_to_vec(a, "weight")).transpose()?;
    let e = exposure.map(|a| array1_to_vec(a, "exposure")).transpose()?;
    Ok(match (w, e) {
        (Some(w), Some(e)) => {
            if w.len() != e.len() {
                return Err(py_err(PbError::ShapeMismatch {
                    what: format!("weight len {} != exposure len {}", w.len(), e.len()),
                }));
            }
            Some(w.iter().zip(&e).map(|(&wi, &ei)| wi * ei).collect())
        }
        (Some(w), None) => Some(w),
        (None, Some(e)) => Some(e),
        (None, None) => None,
    })
}

fn write_or_return_array1<'py>(
    py: Python<'py>,
    values: Vec<f32>,
    out: Option<Bound<'py, PyArray1<f32>>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let Some(out) = out else {
        return Ok(values.into_pyarray(py));
    };
    {
        // `try_readwrite` (not `readwrite`) so a read-only / borrowed numpy array returns
        // a typed Python error rather than panicking across the FFI boundary.
        let mut borrowed: PyReadwriteArray1<'_, f32> = out
            .try_readwrite()
            .map_err(|_| PyTypeError::new_err("out must be a writable contiguous float32 array"))?;
        let out_slice = borrowed
            .as_slice_mut()
            .map_err(|_| PyTypeError::new_err("out must be a contiguous writable float32 array"))?;
        if out_slice.len() != values.len() {
            return Err(py_err(PbError::ShapeMismatch {
                what: format!(
                    "out len {} != prediction len {}",
                    out_slice.len(),
                    values.len()
                ),
            }));
        }
        out_slice.copy_from_slice(&values);
    }
    Ok(out)
}

/// Convert a feature position into a raw [`FeatureId`], guarding the `u32` cast.
fn raw_id(i: usize) -> Result<u32, PbError> {
    u32::try_from(i).map_err(|_| PbError::InvalidInput {
        what: "more than u32::MAX features is out of scope for v1".into(),
    })
}

/// The distinct [`TsEncodingId`]s a fitted model's own provenance records for raw feature
/// `raw` (sorted, deduped). P1 multi-channel (design/multichannel-categoricals.md §4.4):
/// serving re-encodes EVERY channel a model was actually fit with, derived entirely from the
/// model's own provenance rather than any external "how many channels" hint — a single-channel
/// (legacy) model yields exactly `[TsEncodingId(0)]`, unchanged from before this existed.
fn categorical_encoding_ids_for_raw(
    provenance: &[AxisProvenance],
    raw: FeatureId,
) -> Vec<TsEncodingId> {
    let mut ids: Vec<TsEncodingId> = provenance
        .iter()
        .filter_map(|p| match p.kind {
            AxisKind::CategoricalTS { encoding } if p.raw == raw => Some(encoding),
            _ => None,
        })
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Rebuild a serve [`BinnedMatrix`] for prediction/explanation. The numeric-only fast
/// path bins through the model grids positionally; the categorical path re-encodes string
/// labels through the frozen [`crate::ModelSchema`] encoders via [`bin_serve_columns`]
/// (matched by raw id, so numeric `0..n_numeric` then sequential categorical ids align
/// with how `fit` laid the axes out).
fn serve_binned_for_model(
    model: &Model,
    numeric_cols: &[Vec<f32>],
    cat_x: Option<&[Vec<String>]>,
) -> Result<BinnedMatrix, PbError> {
    let Some(cats) = cat_x else {
        return binned_for_model(model, numeric_cols);
    };
    let n_numeric = numeric_cols.len();
    let numeric = numeric_cols
        .iter()
        .enumerate()
        .map(|(i, values)| {
            Ok::<_, PbError>(NumericColumn {
                raw: FeatureId(raw_id(i)?),
                values,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let categorical = cats
        .iter()
        .enumerate()
        .map(|(j, levels)| {
            let raw = FeatureId(raw_id(n_numeric + j)?);
            // P1 multi-channel: serve every channel the MODEL actually recorded (never a
            // caller-supplied hint), so this is unchanged for legacy single-channel models.
            Ok::<_, PbError>(
                categorical_encoding_ids_for_raw(&model.provenance, raw)
                    .into_iter()
                    .map(|id| ServeCategoricalColumn { raw, id, levels })
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Result<Vec<Vec<_>>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let serve = bin_serve_columns(
        &numeric,
        &categorical,
        &model.grids,
        &model.provenance,
        &model.schema.cat_encoders,
    )?;
    Ok(serve.0)
}

/// Reject the numeric-only serve fast path when the model/table-model carries native
/// categorical axes (`AxisKind::CategoricalTS`). Those axes' grids live in
/// target-statistic space (e.g. `0.02..0.31`); binning raw positional columns through
/// them (as `binned_for_model`/`serve_binned_for_tables` do when `cat_x` is omitted)
/// produces finite but silently wrong bins rather than an error. Caught once here,
/// O(n_axes), zero cost for numeric-only models.
fn require_cat_x_for_categorical_axes(provenance: &[AxisProvenance]) -> Result<(), PbError> {
    let n_cat = provenance
        .iter()
        .filter(|p| matches!(p.kind, AxisKind::CategoricalTS { .. }))
        .count();
    if n_cat > 0 {
        return Err(PbError::InvalidInput {
            what: format!(
                "model was fit with {n_cat} native categorical column(s); pass cat_x with \
                 the raw category labels"
            ),
        });
    }
    Ok(())
}

fn binned_for_model(model: &Model, columns: &[Vec<f32>]) -> Result<BinnedMatrix, PbError> {
    require_cat_x_for_categorical_axes(&model.provenance)?;
    if columns.len() != model.grids.len() {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "x has {} columns, model has {} features",
                columns.len(),
                model.grids.len()
            ),
        });
    }
    let n_rows = columns.first().map_or(0usize, Vec::len);
    let mut data = Vec::with_capacity(columns.len());
    for (axis, (col, grid)) in columns.iter().zip(&model.grids).enumerate() {
        if col.len() != n_rows {
            return Err(PbError::ShapeMismatch {
                what: format!("x column {axis} len {} != n_rows {n_rows}", col.len()),
            });
        }
        let mut bins = Vec::with_capacity(n_rows);
        for &value in col {
            bins.push(bin(value, grid)?);
        }
        data.push(bins);
    }
    Ok(BinnedMatrix {
        data,
        n_rows: u32::try_from(n_rows).map_err(|_| PbError::InvalidInput {
            what: "more than u32::MAX rows is out of scope".into(),
        })?,
        grids: model.grids.clone(),
        provenance: model.provenance.clone(),
    })
}

/// Build the concrete loss for a fitted model from its recorded objective tag (pruning scores
/// held-out deviance through it). Multiclass (softmax) is rejected — pruning is single-output only.
fn loss_choice_from_tag(tag: &t_boost_core::loss::ObjectiveTag) -> Result<LossChoice, PbError> {
    Ok(match tag.loss {
        LossId::SquaredError => LossChoice::SquaredError(SquaredError),
        LossId::Logistic => LossChoice::Logistic(Logistic),
        LossId::Poisson => LossChoice::Poisson(Poisson),
        LossId::Gamma => LossChoice::Gamma(Gamma),
        LossId::Tweedie => LossChoice::Tweedie(Tweedie::new(tag.tweedie_rho.unwrap_or(1.5))?),
        LossId::Softmax => {
            return Err(PbError::InvalidConfig {
                what: "pruning is not supported for multiclass (softmax) models".into(),
            })
        }
    })
}

/// Bin raw serve columns through a [`TableModel`]'s frozen grids/encoders (mirrors
/// [`serve_binned_for_model`] but for the tables-only served model — same fields, no trees).
/// Band every class bank of a multiclass tables model (see `t_boost_core::banding`). Each class is
/// banded on its own softmax curvature `w · p_k (1 - p_k)`, its own logits as the distil target and
/// its own bag noise (the deploy soup's class-`k` bags restricted to the keep-set).
#[allow(clippy::too_many_arguments)] // JUSTIFIED: the multiclass deploy stage's own inputs (§12.2).
fn band_multiclass_table_model(
    tm: &MultiClassTableModel,
    mc: &MultiClassModel,
    serve: &ServeBinnedMatrix,
    columns: &[Vec<f32>],
    cat_x: Option<&[Vec<String>]>,
    w: &[f32],
    w_measure: RefMeasure,
    keep: &[FeatureSet],
    tolerance: f64,
    labels: &[u32],
    deviance_cap: f64,
) -> Result<
    (
        MultiClassTableModel,
        Vec<t_boost_core::banding::BandingReport>,
    ),
    PbError,
> {
    let first = tm.classes.first().ok_or_else(|| PbError::Internal {
        what: "multiclass tables model has no classes".into(),
    })?;
    let binned = serve_binned_for_tables(
        first,
        columns.to_vec(),
        cat_x.map(<[Vec<String>]>::to_vec),
        None,
    )?;
    let n = w.len();
    // Each class bank has its OWN merged grid (the classes share raw grids, not merged ones), so
    // the rows' merged cells are taken per class.
    let class_cells =
        |c: &TableModel| -> Result<Vec<Vec<u32>>, PbError> { c.column_cells(&binned) };
    let logits: Vec<Vec<f64>> = mc_stage("banding.logits", || {
        tm.classes
            .iter()
            .map(|c| {
                c.score_raw(&binned, None)
                    .map(|v| v.iter().map(|&x| f64::from(x)).collect())
            })
            .collect::<Result<_, _>>()
    })?;
    let k_classes = logits.len();
    // softmax per row, accumulated class by class in the same order as a per-row loop
    let mut mx = vec![f64::NEG_INFINITY; n];
    for lk in &logits {
        for (m, &x) in mx.iter_mut().zip(lk) {
            *m = m.max(x);
        }
    }
    let mut z = vec![-0.0_f64; n];
    for lk in &logits {
        for ((zi, &x), &m) in z.iter_mut().zip(lk).zip(&mx) {
            *zi += (x - m).exp();
        }
    }
    let prob: Vec<Vec<f64>> = logits
        .iter()
        .map(|lk| {
            lk.iter()
                .zip(&mx)
                .zip(&z)
                .map(|((&x, &m), &zi)| (x - m).exp() / zi)
                .collect()
        })
        .collect();
    let mass: Vec<f64> = w.iter().map(|&x| f64::from(x)).collect();
    // Mean multinomial deviance (twice the NLL) at the deployed logits: the deviance cap's scale.
    let ws: f64 = mass.iter().sum();
    let mut dev = 0.0;
    for (i, (&l, &mi)) in labels.iter().zip(&mass).enumerate() {
        let p = prob
            .get(l as usize)
            .and_then(|pk| pk.get(i))
            .copied()
            .unwrap_or(1.0)
            .max(1e-15);
        dev -= 2.0 * mi * p.ln();
    }
    let dev_mean = dev / ws.max(f64::MIN_POSITIVE);
    let all_rows: Vec<u32> = (0..u32::try_from(n).unwrap_or(u32::MAX)).collect();
    // The bag banks behind the noise estimate are purified under the same row mass as the
    // multiclass prune's and keep-set's banks (`measure_mass`: the weights, under a measure that
    // reads row mass).
    let bag_mass = t_boost_core::prune::measure_mass(&w_measure, w, None);
    let cfg = t_boost_core::banding::BandingConfig {
        tolerance,
        ..Default::default()
    };
    let no_class = |k: usize| PbError::Internal {
        what: format!("multiclass banding: no class {k}"),
    };
    let mut out = tm.clone();
    let mut reports = Vec::with_capacity(k_classes);
    for (k, (pk, lk)) in prob.iter().zip(&logits).enumerate() {
        let h: Vec<f64> = mass
            .iter()
            .zip(pk)
            .map(|(&mi, &p)| mi * p * (1.0 - p))
            .collect();
        let mck = mc.classes.get(k).ok_or_else(|| no_class(k))?;
        let tmk = tm.classes.get(k).ok_or_else(|| no_class(k))?;
        let var = mc_stage("banding.bag_variance", || {
            t_boost_core::prune::bag_score_variance_for_rows(
                mck,
                serve,
                w_measure.clone(),
                bag_mass.as_deref(),
                Some(keep),
                &all_rows,
            )
        })?;
        let b = mck.bag_spans.as_ref().map_or(2, Vec::len).max(2) as f64;
        let hs: f64 = h.iter().sum();
        let num: f64 = h.iter().zip(&var).map(|(&hi, &vi)| hi * vi / b).sum();
        let cells = mc_stage("banding.class_cells", || class_cells(tmk))?;
        let rows = t_boost_core::banding::BandingRows {
            cells: &cells,
            h: &h,
            mass: &mass,
            target: lk,
            sigma: (num / hs.max(f64::MIN_POSITIVE)).sqrt(),
            // the class's share of the deviance cap (expected increase Σ h δ² / Σ w per class)
            mse_cap: deviance_cap * dev_mean * ws / (hs.max(f64::MIN_POSITIVE) * k_classes as f64),
        };
        let (bank, rep) = t_boost_core::banding::band_bank(&tmk.bank, &rows, &cfg)?;
        out.classes.get_mut(k).ok_or_else(|| no_class(k))?.bank = bank;
        reports.push(rep);
    }
    out.validate()?;
    Ok((out, reports))
}

/// Per-model serve state for a [`PyTableModel`]: the model-bin -> merged-cell maps and the
/// categorical label maps, which depend only on the model and were rebuilt on every predict (the
/// dominant fixed cost of a small batch: 8.8 ms a call on brvehins1). Built on the first predict
/// and never serialized; the model behind the `Arc` is immutable, so it cannot go stale.
struct TableServe {
    cells: CellMaps,
    cats: CatServeMaps,
}

/// Per-class counterpart of [`TableServe`] for a [`PyMultiClassTableModel`] (every class shares
/// the first class's grids/provenance, so one set of categorical maps serves all of them).
struct MultiClassTableServe {
    cells: Vec<CellMaps>,
    cats: CatServeMaps,
}

fn table_serve<'a>(
    model: &TableModel,
    cache: &'a OnceLock<TableServe>,
) -> Result<&'a TableServe, PbError> {
    if let Some(ts) = cache.get() {
        return Ok(ts);
    }
    let built = TableServe {
        cells: model.cell_maps()?,
        cats: CatServeMaps::build(&model.provenance, &model.schema.cat_encoders)?,
    };
    // A concurrent first call may have won the race; either build is identical.
    Ok(cache.get_or_init(|| built))
}

fn multiclass_table_serve<'a>(
    model: &MultiClassTableModel,
    cache: &'a OnceLock<MultiClassTableServe>,
) -> Result<&'a MultiClassTableServe, PbError> {
    if let Some(ts) = cache.get() {
        return Ok(ts);
    }
    let first = model.classes.first().ok_or_else(|| PbError::Internal {
        what: "multiclass tables model has no classes".into(),
    })?;
    let built = MultiClassTableServe {
        cells: model.cell_maps()?,
        cats: CatServeMaps::build(&first.provenance, &first.schema.cat_encoders)?,
    };
    Ok(cache.get_or_init(|| built))
}

/// The categorical half of a table-model predict call: string labels per row (`cat_x`), or codes
/// into distinct labels (`cat_codes`, the fast path — see `bin_serve_columns_coded`).
enum ServeCatsInput {
    Labels(Option<Vec<Vec<String>>>),
    Codes(Vec<(Vec<u32>, Vec<String>)>),
}

fn serve_cats(
    cat_x: Option<Vec<Vec<String>>>,
    cat_codes: Option<Vec<(PyReadonlyArray1<'_, u32>, Vec<String>)>>,
) -> PyResult<ServeCatsInput> {
    match (cat_x, cat_codes) {
        (Some(_), Some(_)) => Err(PyValueError::new_err(
            "pass categorical columns as cat_x or as cat_codes, not both",
        )),
        (labels, None) => Ok(ServeCatsInput::Labels(labels)),
        (None, Some(coded)) => coded
            .into_iter()
            .map(|(codes, labels)| Ok((array1_to_vec_u32(codes, "cat_codes")?, labels)))
            .collect::<PyResult<Vec<_>>>()
            .map(ServeCatsInput::Codes),
    }
}

fn serve_binned_tables_any(
    tm: &TableModel,
    numeric_cols: Vec<Vec<f32>>,
    cats: ServeCatsInput,
    cat_maps: Option<&CatServeMaps>,
) -> Result<BinnedMatrix, PbError> {
    let coded = match cats {
        ServeCatsInput::Labels(labels) => {
            return serve_binned_for_tables(tm, numeric_cols, labels, cat_maps)
        }
        ServeCatsInput::Codes(coded) => coded,
    };
    let n_numeric = numeric_cols.len();
    let numeric = numeric_cols
        .iter()
        .enumerate()
        .map(|(i, values)| {
            Ok::<_, PbError>(NumericColumn {
                raw: FeatureId(raw_id(i)?),
                values,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut categorical = Vec::new();
    for (j, (codes, labels)) in coded.iter().enumerate() {
        let raw = FeatureId(raw_id(n_numeric + j)?);
        // Every channel the model recorded for this raw feature, as in `serve_binned_for_tables`.
        for id in categorical_encoding_ids_for_raw(&tm.provenance, raw) {
            categorical.push(ServeCategoricalCodes {
                raw,
                id,
                codes,
                labels,
            });
        }
    }
    let serve = bin_serve_columns_coded(
        &numeric,
        &categorical,
        &tm.grids,
        &tm.provenance,
        &tm.schema.cat_encoders,
        cat_maps,
    )?;
    Ok(serve.0)
}

/// Row count from which serve binning runs its columns in parallel (see `bin_serve_columns_with`).
const SERVE_PAR_MIN_ROWS: usize = 16_384;

fn serve_binned_for_tables(
    tm: &TableModel,
    numeric_cols: Vec<Vec<f32>>,
    cat_x: Option<Vec<Vec<String>>>,
    cat_maps: Option<&CatServeMaps>,
) -> Result<BinnedMatrix, PbError> {
    let Some(cats) = cat_x else {
        require_cat_x_for_categorical_axes(&tm.provenance)?;
        if numeric_cols.len() != tm.grids.len() {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "x has {} columns, tables model has {} features",
                    numeric_cols.len(),
                    tm.grids.len()
                ),
            });
        }
        let n_rows = numeric_cols.first().map_or(0usize, Vec::len);
        let bin_axis = |axis: usize,
                        col: &Vec<f32>,
                        grid: &t_boost_core::data::BorderGrid|
         -> Result<Vec<u8>, PbError> {
            if col.len() != n_rows {
                return Err(PbError::ShapeMismatch {
                    what: format!("x column {axis} len {} != n_rows {n_rows}", col.len()),
                });
            }
            col.iter().map(|&value| bin(value, grid)).collect()
        };
        // Columns are independent, so binning them in parallel is byte-identical to the serial
        // loop; small batches stay serial (see `bin_serve_columns_with`).
        let data: Vec<Vec<u8>> = if n_rows >= SERVE_PAR_MIN_ROWS {
            numeric_cols
                .par_iter()
                .zip(tm.grids.par_iter())
                .enumerate()
                .map(|(axis, (col, grid))| bin_axis(axis, col, grid))
                .collect::<Result<_, _>>()?
        } else {
            numeric_cols
                .iter()
                .zip(&tm.grids)
                .enumerate()
                .map(|(axis, (col, grid))| bin_axis(axis, col, grid))
                .collect::<Result<_, _>>()?
        };
        return Ok(BinnedMatrix {
            data,
            n_rows: u32::try_from(n_rows).map_err(|_| PbError::InvalidInput {
                what: "more than u32::MAX rows is out of scope".into(),
            })?,
            grids: tm.grids.clone(),
            provenance: tm.provenance.clone(),
        });
    };
    let n_numeric = numeric_cols.len();
    let numeric = numeric_cols
        .iter()
        .enumerate()
        .map(|(i, values)| {
            Ok::<_, PbError>(NumericColumn {
                raw: FeatureId(raw_id(i)?),
                values,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let categorical = cats
        .iter()
        .enumerate()
        .map(|(j, levels)| {
            let raw = FeatureId(raw_id(n_numeric + j)?);
            // P1 multi-channel: mirrors `serve_binned_for_model` above.
            Ok::<_, PbError>(
                categorical_encoding_ids_for_raw(&tm.provenance, raw)
                    .into_iter()
                    .map(|id| ServeCategoricalColumn { raw, id, levels })
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Result<Vec<Vec<_>>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let serve = bin_serve_columns_with(
        &numeric,
        &categorical,
        &tm.grids,
        &tm.provenance,
        &tm.schema.cat_encoders,
        cat_maps,
    )?;
    Ok(serve.0)
}

fn graduation_deviance_term(
    loss: LossId,
    tweedie_rho: Option<f32>,
    y: f32,
    pred: f32,
    weight: f32,
    exposure: f32,
) -> f64 {
    let y = f64::from(y);
    let mu = (f64::from(pred) * f64::from(exposure)).max(1e-12);
    let per = match loss {
        LossId::Poisson => {
            let log_term = if y > 0.0 { y * (y / mu).ln() } else { 0.0 };
            2.0 * (log_term - (y - mu))
        }
        LossId::Gamma => {
            let ratio = if y > 0.0 { y / mu } else { 1.0 };
            2.0 * (ratio - 1.0 - ratio.ln())
        }
        LossId::Tweedie => {
            let rho = f64::from(tweedie_rho.unwrap_or(1.5));
            let positive_y = y.max(0.0);
            2.0 * (if positive_y > 0.0 {
                positive_y.powf(2.0 - rho) / ((1.0 - rho) * (2.0 - rho))
            } else {
                0.0
            } - positive_y * mu.powf(1.0 - rho) / (1.0 - rho)
                + mu.powf(2.0 - rho) / (2.0 - rho))
        }
        // Binomial deviance for the logit link (2026-09-06): graduation's no-harm gate
        // must judge a classifier on its own loss, not on a squared-error proxy.
        LossId::Logistic => {
            let p = mu.clamp(1e-12, 1.0 - 1e-12);
            let yy = y.clamp(0.0, 1.0);
            let mut d = 0.0_f64;
            if yy > 0.0 {
                d += yy * (yy / p).ln();
            }
            if yy < 1.0 {
                d += (1.0 - yy) * ((1.0 - yy) / (1.0 - p)).ln();
            }
            2.0 * d
        }
        LossId::SquaredError | LossId::Softmax => (y - mu) * (y - mu),
    };
    f64::from(weight) * per
}

fn graduation_weighted_deviance(
    model: &TableModel,
    y: &[f32],
    pred: &[f32],
    weight: &[f32],
    exposure: Option<&[f32]>,
) -> Result<f64, PbError> {
    let n = y.len();
    if pred.len() != n || weight.len() != n || exposure.is_some_and(|values| values.len() != n) {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "graduation deviance lengths y={n}, pred={}, weight={}, exposure={}",
                pred.len(),
                weight.len(),
                exposure.map_or(n, <[f32]>::len)
            ),
        });
    }
    let objective = &model.schema.objective;
    let terms: Vec<f64> = match exposure {
        Some(exposure) => y
            .par_iter()
            .zip(pred.par_iter())
            .zip(weight.par_iter())
            .zip(exposure.par_iter())
            .map(|(((&y, &pred), &weight), &exposure)| {
                graduation_deviance_term(
                    objective.loss,
                    objective.tweedie_rho,
                    y,
                    pred,
                    weight,
                    exposure,
                )
            })
            .collect(),
        None => y
            .par_iter()
            .zip(pred.par_iter())
            .zip(weight.par_iter())
            .map(|((&y, &pred), &weight)| {
                graduation_deviance_term(
                    objective.loss,
                    objective.tweedie_rho,
                    y,
                    pred,
                    weight,
                    1.0,
                )
            })
            .collect(),
    };
    let mut total = 0.0_f64;
    for term in terms {
        total += term;
    }
    Ok(total)
}

/// Deployed tables and cells by interaction order over `banks`, as a Python dict
/// `{"banks": len(banks), "tables": {order: n}, "cells": {order: n}}` — what walking
/// `to_json()`'s `tables` and `factored` entries counts, without serializing a bank. A factored
/// (box) table counts as a table but has no cell grid, so an order appears under `cells` only
/// when a dense or sparse table of that order does; its cells are the table's full extent.
fn deployed_census_dict<'py>(
    py: Python<'py>,
    banks: &[&TableBank],
) -> PyResult<Bound<'py, PyDict>> {
    let mut tables: BTreeMap<usize, u64> = BTreeMap::new();
    let mut cells: BTreeMap<usize, u64> = BTreeMap::new();
    for bank in banks {
        for t in &bank.tables {
            *tables.entry(t.u.order()).or_insert(0) += 1;
            let shape = t.values.shape_u32();
            if !shape.is_empty() {
                let n = shape
                    .iter()
                    .fold(1_u64, |acc, &d| acc.saturating_mul(u64::from(d)));
                let c = cells.entry(t.u.order()).or_insert(0);
                *c = c.saturating_add(n);
            }
        }
        for f in &bank.factored {
            *tables.entry(f.u.order()).or_insert(0) += 1;
        }
    }
    let d = PyDict::new(py);
    d.set_item("banks", banks.len())?;
    d.set_item("tables", tables)?;
    d.set_item("cells", cells)?;
    Ok(d)
}

/// `eval_rows`, when given, restricts the no-harm comparison to those rows of the design —
/// the caller passes the rows the bags left OUT of bag, so the acceptance is honest rather
/// than a training-loss brake — and tightens the slack from the legacy 1% (a training loss
/// always rises under smoothing) to a tenth of a percent.
/// The log-link intercept rebalance still runs on every row: it is the balance step, not the
/// gate. `None` is the pre-2026-09-06 behaviour, byte for byte.
#[allow(clippy::too_many_arguments)] // JUSTIFIED: one binding, one core call; the tuple would only rename them.
fn apply_graduation_updates(
    model: &TableModel,
    columns: Vec<Vec<f32>>,
    y: Vec<f32>,
    weight: Vec<f32>,
    exposure: Option<Vec<f32>>,
    cat_x: Option<Vec<Vec<String>>>,
    updates: Vec<(usize, Vec<f64>)>,
    eval_rows: Option<Vec<u32>>,
) -> Result<(TableModel, bool), PbError> {
    let (model, adopted, _) = apply_graduation_updates_impl(
        model, columns, y, weight, exposure, cat_x, updates, eval_rows, 0.0, 0,
    )?;
    Ok((model, adopted))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn apply_graduation_updates_impl(
    model: &TableModel,
    columns: Vec<Vec<f32>>,
    y: Vec<f32>,
    weight: Vec<f32>,
    exposure: Option<Vec<f32>>,
    cat_x: Option<Vec<Vec<String>>>,
    updates: Vec<(usize, Vec<f64>)>,
    eval_rows: Option<Vec<u32>>,
    high_order_alpha: f64,
    box_budget: usize,
) -> Result<
    (
        TableModel,
        bool,
        Vec<t_boost_core::explain::HighOrderSmoothingReport>,
    ),
    PbError,
> {
    let numeric_raws: Vec<u32> = model
        .provenance
        .iter()
        .filter(|p| matches!(p.kind, AxisKind::Numeric))
        .map(|p| p.raw.0)
        .collect();
    let existing_boxes: usize = model.bank.factored.iter().map(|f| f.n_boxes()).sum();
    let added_budget = if box_budget == 0 {
        4096
    } else {
        4096.min(box_budget.saturating_sub(existing_boxes))
    };
    let (bank, report) = model.bank.smooth_high_order(
        &numeric_raws,
        high_order_alpha,
        1024,
        added_budget,
        2_000_000,
    )?;
    if updates.is_empty() && !report.iter().any(|r| r.applied) {
        return Ok((model.clone(), true, report));
    }
    if let Some(rows) = eval_rows.as_deref() {
        if rows.is_empty() {
            return Err(PbError::InvalidInput {
                what: "graduation eval_rows must not be empty".into(),
            });
        }
        if let Some(&bad) = rows.iter().find(|&&r| r as usize >= y.len()) {
            return Err(PbError::InvalidInput {
                what: format!(
                    "graduation eval row {bad} is out of range for {} rows",
                    y.len()
                ),
            });
        }
    }
    if y.len() != weight.len()
        || exposure
            .as_ref()
            .is_some_and(|values| values.len() != y.len())
    {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "graduation input lengths y={}, weight={}, exposure={}",
                y.len(),
                weight.len(),
                exposure.as_ref().map_or(y.len(), Vec::len)
            ),
        });
    }

    let mut graduated = model.clone();
    graduated.bank = bank;
    let dense_changed = !updates.is_empty();
    let mut seen = BTreeSet::new();
    for (index, values) in updates {
        if !seen.insert(index) {
            return Err(PbError::InvalidInput {
                what: format!("graduation table {index} was updated more than once"),
            });
        }
        let table = graduated
            .bank
            .tables
            .get_mut(index)
            .ok_or_else(|| PbError::InvalidInput {
                what: format!("graduation table index {index} out of range"),
            })?;
        if table.values.is_sparse() {
            return Err(PbError::InvalidInput {
                what: format!("graduation table {index} is sparse"),
            });
        }
        if values.len() != table.values.len() || values.iter().any(|value| !value.is_finite()) {
            return Err(PbError::InvalidInput {
                what: format!(
                    "graduation table {index} received {} finite cells, expected {}",
                    values.len(),
                    table.values.len()
                ),
            });
        }
        table.values = Tensor::from_vec(table.values.shape(), values)?;
    }

    if dense_changed {
        // Re-purify: GCV smoothing preserves each table's own support-weighted grand mean, not
        // the per-axis-slice w-weighted zero means Purity requires, so a smoothed table can
        // silently absorb lower-order mass (e.g. a smoothed 2-D interaction table absorbing part
        // of a main effect). Feed every CURRENT (post-smoothing) table back through the canonical
        // purify cascade as if it were raw — purify is linear, so this is exactly equivalent to
        // purifying the original raw effects (Lengerich Cor 2.2), and it is the only
        // reconstruction available at all once a TableModel has already dropped its trees.
        // `purify` also recomputes every table's `variance` internally (`table_variance`),
        // closing the stale-Sobol half of the same finding for free. Factored effects have
        // already undergone optional reference-preserving diffusion and remain pure.
        let banded = graduated
            .bank
            .tables
            .iter()
            .any(|t| t.axes.iter().any(|a| a.band_of.is_some()));
        if banded {
            // Banded tables live on band grids: re-purify band-aware (exactly equivalent).
            graduated.bank = t_boost_core::banding::repurify_bank(&graduated.bank)?;
        } else {
            let effects: Vec<RawEffect> = graduated
                .bank
                .tables
                .iter()
                .map(|table| RawEffect {
                    u: table.u.clone(),
                    values: table.values.clone(),
                    support: table.support.clone(),
                })
                .collect();
            let factored = std::mem::take(&mut graduated.bank.factored);
            let mut repurified = purify_raw_effects(
                graduated.bank.f0,
                graduated.bank.merged_grids.clone(),
                graduated.bank.w.clone(),
                effects,
            )?;
            repurified.factored = factored;
            graduated.bank = repurified;
        }
    }

    let binned = serve_binned_for_tables(model, columns, cat_x, None)?;
    if binned.n_rows as usize != y.len() {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "graduation design has {} rows but y has {}",
                binned.n_rows,
                y.len()
            ),
        });
    }

    if model.link == Link::Log {
        let pred = graduated.predict_binned(&binned, None)?;
        let mut numerator = 0.0_f64;
        let mut denominator = 0.0_f64;
        match exposure.as_deref() {
            Some(exposure) => {
                for (((&y, &weight), &pred), &exposure) in y
                    .iter()
                    .zip(weight.iter())
                    .zip(pred.iter())
                    .zip(exposure.iter())
                {
                    let w = f64::from(weight);
                    numerator += w * f64::from(y);
                    denominator += w * f64::from(pred) * f64::from(exposure);
                }
            }
            None => {
                for ((&y, &weight), &pred) in y.iter().zip(weight.iter()).zip(pred.iter()) {
                    let w = f64::from(weight);
                    numerator += w * f64::from(y);
                    denominator += w * f64::from(pred);
                }
            }
        }
        if numerator > 0.0 && denominator > 0.0 {
            graduated.bank.f0 += (numerator / denominator).ln();
        }
    }

    let raw_pred = model.predict_binned(&binned, None)?;
    let graduated_pred = graduated.predict_binned(&binned, None)?;
    let (raw_deviance, graduated_deviance, slack) = match eval_rows.as_deref() {
        Some(rows) => {
            let take = |v: &[f32]| -> Vec<f32> {
                rows.iter()
                    .filter_map(|&r| v.get(r as usize).copied())
                    .collect()
            };
            let y_e = take(&y);
            let w_e = take(&weight);
            let e_e = exposure.as_deref().map(take);
            let raw_e = take(&raw_pred);
            let grad_e = take(&graduated_pred);
            (
                graduation_weighted_deviance(model, &y_e, &raw_e, &w_e, e_e.as_deref())?,
                graduation_weighted_deviance(model, &y_e, &grad_e, &w_e, e_e.as_deref())?,
                // A tenth of a percent: on the InsurArena skill scale (1 − D/D_GLM) a slack
                // of half a percent of the deviance was worth ~0.005 skill, and the gate let
                // exactly that much harm through on a zero-heavy Tweedie book (ohlsson_pp).
                1.001,
            )
        }
        None => (
            graduation_weighted_deviance(model, &y, &raw_pred, &weight, exposure.as_deref())?,
            graduation_weighted_deviance(model, &y, &graduated_pred, &weight, exposure.as_deref())?,
            1.01,
        ),
    };
    if !graduated_deviance.is_finite() || graduated_deviance > raw_deviance * slack {
        return Ok((model.clone(), false, report));
    }
    graduated.validate()?;
    Ok((graduated, true, report))
}

fn parse_ref_measure(
    name: Option<String>,
    laplace: f32,
    measure_floor: f32,
) -> Result<RefMeasure, PbError> {
    let normalized = name
        .unwrap_or_else(|| "product_marginals".to_owned())
        .replace('-', "_")
        .to_ascii_lowercase();
    match normalized.as_str() {
        "product" | "product_marginals" => Ok(RefMeasure::ProductMarginals { laplace }),
        "uniform" => Ok(RefMeasure::Uniform),
        // Exposure-weighted marginals with a positivity floor (2026-09-06). `None` still
        // resolves to the legacy `product_marginals` at THIS layer; the estimator resolves
        // its own default and always passes an explicit name.
        "exposure" | "exposure_marginals" => Ok(RefMeasure::ExposureMarginals {
            floor: measure_floor,
        }),
        // Export-only (2026-09-06): the joint-exposure hierarchical-orthogonality ledger,
        // produced by `joint::rejoint` from a product-measure bank. Never a fit-time measure.
        "joint" => Ok(RefMeasure::Joint),
        other => Err(PbError::InvalidConfig {
            what: format!("unknown reference measure `{other}`"),
        }),
    }
}

/// Resolve the table-budget overflow policy for `explain`/`tables` (spec §08.10).
///
/// Defaults to `Factored` (§08.10): when a converged model's merged grid pushes an order-3
/// table over the dense `max_table_cells` budget, the over-budget effect is kept exactly as a
/// factored per-tree-box sum rather than materializing the dense cube — no hard failure at
/// competitive tree counts. `"error"` opts into the loud, fast fail-fast policy instead (an
/// immediate, actionable `PbError::TableBudget`); `"sparse"` opts into the EXACT
/// `SparseFallback` storage — currently a slow path (accumulation still walks the dense
/// extent), so it stays opt-in until the occupancy-driven walk lands. All three policies are
/// exactness-preserving.
fn parse_table_budget(overflow: Option<&str>) -> Result<TableBudget, PbError> {
    let on_overflow = match overflow.map(str::to_ascii_lowercase).as_deref() {
        // DEFAULT: factor over-budget order-3 effects (exact, §08.10) — the decomposition no
        // longer hard-fails at competitive tree counts. `error`/`sparse` remain opt-in.
        None | Some("factored") | Some("factor") => OverflowPolicy::Factored,
        Some("error") | Some("hard") => OverflowPolicy::Error,
        Some("sparse") | Some("sparse_fallback") => OverflowPolicy::SparseFallback {
            density_threshold: 0.05,
        },
        Some(other) => {
            return Err(PbError::InvalidConfig {
                what: format!("overflow must be 'factored', 'error', or 'sparse', got `{other}`"),
            })
        }
    };
    Ok(TableBudget {
        on_overflow,
        ..TableBudget::default()
    })
}

fn parse_rating_basis(value: Option<&str>) -> PyResult<Option<RatingBasis>> {
    value
        .map(|s| {
            serde_json::from_str::<RatingBasis>(s).map_err(|err| {
                py_err(PbError::Serialization(format!(
                    "could not parse RatingBasis JSON: {err}"
                )))
            })
        })
        .transpose()
}

/// The single funnel from `PbError` to `PyErr` (spec §12.7's mapping table). `InvalidInput` /
/// `ShapeMismatch` / `InvalidConfig` surface as [`t_boost_value_error_type`] and
/// `DtypeMismatch` as [`t_boost_type_error_type`] — each simultaneously the spec-promised
/// builtin (`ValueError`/`TypeError`, matching sklearn/numpy convention, catchable with no
/// `t_boost` import) AND `t_boost.TBoostError` (so existing `except TBoostError`
/// handlers keep working). The four `t_boost`-only exceptions (Invariant/Exactness/
/// Serialization/Internal) stay single-inheritance because they name a firewall or a bug
/// class, not an ordinary bad-input mistake, and have no builtin equivalent to dual-inherit.
/// `TableBudget` (not in the spec table) keeps its prior `TBoostError` fallback.
fn py_err(err: PbError) -> PyErr {
    match err {
        PbError::InvalidInput { what } => t_boost_value_error(format!("invalid input: {what}")),
        PbError::ShapeMismatch { what } => t_boost_value_error(format!("shape mismatch: {what}")),
        PbError::InvalidConfig { what } => t_boost_value_error(format!("invalid config: {what}")),
        PbError::DtypeMismatch { expected } => {
            t_boost_type_error(format!("dtype mismatch: expected {expected}"))
        }
        PbError::InvariantViolated { invariant } => {
            InvariantError::new_err(invariant_message(invariant))
        }
        PbError::ExactnessFirewall(reason) => ExactnessError::new_err(reason),
        PbError::Serialization(message) => SerializationError::new_err(message),
        PbError::Internal { what } => InternalError::new_err(what),
        other => TBoostError::new_err(other.to_string()),
    }
}

fn invariant_message(invariant: Invariant) -> String {
    invariant.to_string()
}

/// The compiled extension module `t_boost._t_boost` (§02.7).
#[pymodule]
fn _t_boost(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("TBoostError", py.get_type::<TBoostError>())?;
    m.add("InvariantError", py.get_type::<InvariantError>())?;
    m.add("ExactnessError", py.get_type::<ExactnessError>())?;
    m.add("SerializationError", py.get_type::<SerializationError>())?;
    m.add("InternalError", py.get_type::<InternalError>())?;
    m.add("TBoostValueError", t_boost_value_error_type(py)?)?;
    m.add("TBoostTypeError", t_boost_type_error_type(py)?)?;
    m.add_class::<PyBooster>()?;
    m.add_class::<PyModel>()?;
    m.add_class::<PyMultiClassModel>()?;
    m.add_class::<PyTableBank>()?;
    m.add_class::<PyTableModel>()?;
    m.add_class::<PyMultiClassTableModel>()?;
    // Timing tripwire: lets Python callers detect an unoptimized debug build (typically
    // 5-30x slower) before trusting any benchmark numbers.
    m.add(
        "BUILD_PROFILE",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;
    Ok(())
}
