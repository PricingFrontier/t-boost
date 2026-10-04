//! The boosting loop (spec §06.6, milestone M1.5) — the Phase-2 capstone that ties
//! the objective (§05), binning (§03), histogram engine (§06.3) and split-finder
//! (§06.2) into `Booster::fit`.
//!
//! `f0 = link(weighted mean)` (the fANOVA intercept), then each round: one
//! full-precision `grad_hess` pass w.r.t. the current raw score → `grow_oblivious_tree`
//! → `update_raw`, until `n_trees` rounds or a round cannot split (graceful stop).
//! Every tree carries `alpha = 1.0` on the green path. Optional §09 boosters may
//! rewrite only leaf scalars / alphas / the intercept; the emitted model remains
//! `ExactnessMode::Exact`.
//!
//! v1 simplifications (the green spine): `subsample = 1.0` (all rows), no column
//! sampling, full-precision histograms, single Newton leaf step, early stopping off.
//! Determinism is structural: the loop is sequential round-by-round, the only
//! parallelism (the histogram build) is a fixed-order fold, so the trained `Model` is
//! byte-identical across thread counts.
//!
//! Train/serve precision: the loop accumulates `raw` in `f32` (the §05 `grad_hess`
//! contract is `raw: &[f32]`); the f64 `ensemble_f64` / §08 table path agrees with it
//! within ~`4·n_trees·f32::EPSILON·magnitude` — exactly the tolerance the §08
//! Reconstruction gate is sized for. The per-tree leaf values are bit-identical
//! between the two scorers (same `low_bit` routing); only the accumulation width differs.

// Pre-existing index-arithmetic in the ensemble/refit helpers flagged by the tightened
// `indexing_slicing` deny lint. Scope-allowed to keep CI green; TODO: convert to `.get()`.
#![allow(
    clippy::indexing_slicing, // JUSTIFIED: pre-existing module-scoped debt (see comment above); burn-down to `.get()`/per-fn allows is incremental, not expanded here.
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::derivable_impls
)]

use crate::backend::{pb_rng, pb_seed, Stage};
use crate::boosters::{CellRefit, DartSpec, EnsembleSpec, HpGrid, NesterovSpec, RefitSpec};
use crate::cat::CatEncoderStore;
use crate::constraints::{credibility_evidence_n, CredibilityEvidence, MonoSign};
use crate::data::{compute_offset, BinnedMatrix};
use crate::engine::split::{
    clamp_monotone, grow_oblivious_tree_with_leaf_map, refit_tree_leaves, GrowConfig, GrowResult,
    InteractionHurdleState, RealizedExtent, TableBudgetPenalty,
};
use crate::engine::{
    softmax_in_place, tree_leaf_index_for_row_with_columns, tree_split_columns,
    tree_value_for_row_with_columns, BagFitReport, Config, CorrectionBank, DeltaStepGateReport,
    ExactnessMode, FitSpec, GatedStepPolicy, Model, ModelSchema, MultiClassCellRefitReport,
    MultiClassModel, ObliviousTree, RoundEvent, Sampling, StopReason, LEGACY_MAX_DEPTH, MAX_LEAVES,
};
use crate::error::PbError;
use crate::loss::{GradHess, Link, Loss, LossId, ObjectiveTag};
use crate::serialize::SCHEMA_VERSION_UNLIFTED;
use rand::RngCore;
use rayon::prelude::*;
use std::borrow::Cow;
use std::collections::BTreeSet;

fn invalid_config(what: &'static str) -> PbError {
    PbError::InvalidConfig { what: what.into() }
}

fn invalid_input(what: String) -> PbError {
    PbError::InvalidInput { what }
}

const REFIT_ACCEPT_TOL: f64 = 1.0e-10;
const REFIT_MAX_BACKTRACKS: usize = 32;
const REFIT_CHOLESKY_JITTERS: [f64; 4] = [0.0, 1.0e-12, 1.0e-10, 1.0e-8];

fn validate_fit_spec(config: &Config, spec: &FitSpec<'_>) -> Result<(), PbError> {
    if !spec.monotone.is_empty()
        && matches!(
            config.boosters.ensemble,
            EnsembleSpec::OuterBag {
                cell_refit: Some(_),
                ..
            }
        )
    {
        return Err(PbError::InvalidConfig {
            what: "cell_refit is incompatible with monotone constraints".into(),
        });
    }
    let max_order = usize::from(spec.interaction.max_order);
    if !(1..=crate::engine::MAX_ORDER).contains(&max_order) {
        return Err(PbError::InvalidConfig {
            what: format!(
                "interaction.max_order must be in 1..={}, got {max_order}",
                crate::engine::MAX_ORDER
            ),
        });
    }
    let max_depth = usize::from(spec.interaction.max_depth);
    if !(LEGACY_MAX_DEPTH..=crate::engine::MAX_DEPTH).contains(&max_depth) {
        return Err(PbError::InvalidConfig {
            what: format!(
                "interaction.max_depth must be in {LEGACY_MAX_DEPTH}..={}, got {max_depth}",
                crate::engine::MAX_DEPTH
            ),
        });
    }
    // A tree needs one level per DISTINCT feature, so `max_order > max_depth` is not a
    // tighter constraint — it is an unreachable one, and it would silently behave as
    // `max_order = max_depth` with no diagnostic. Refuse it: the caller asked for an
    // interaction order this tree shape cannot express, and the fix (raise `max_depth`)
    // is not something to guess on their behalf.
    if max_order > max_depth {
        return Err(PbError::InvalidConfig {
            what: format!(
                "interaction.max_order {max_order} exceeds interaction.max_depth \
                 {max_depth}: an oblivious tree needs one level per distinct raw feature, \
                 so order {max_order} is unreachable at depth {max_depth}"
            ),
        });
    }
    // C4 — the fully-corrective ridge refit is dimensioned to a FIXED stride of 8 leaf
    // columns per tree (`n_cols = n_trees * 8`, `refit_col = tree_idx * 8 + leaf`). At
    // depth 6 the normal matrix grows 64x (500 trees: 32000^2 cells, ~8 GB) AND becomes
    // badly rank-deficient, since 58% of those columns are phantom leaves standing on
    // zero rows. Fail LOUDLY rather than silently mis-solve or exhaust memory. The fix
    // (a per-tree prefix-sum column offset plus dropping unreachable-leaf columns via the
    // existing `monotone_reachable` mask) is tracked as P-D4.
    if max_depth > LEGACY_MAX_DEPTH
        && matches!(config.boosters.refit_leaves, RefitSpec::Ridge { .. })
    {
        return Err(PbError::InvalidConfig {
            what: format!(
                "ridge_refit_l2 is not supported with max_depth {max_depth} > {LEGACY_MAX_DEPTH}: \
                 the fully-corrective refit assumes a fixed {LEGACY_LEAVES}-leaf-column stride \
                 per tree (the design would grow {grow}x and be rank-deficient, since a \
                 depth-{max_depth} tree on k distinct raws realizes only prod(k_i) of its \
                 {slots} leaves). Set ridge_refit_l2=None or max_depth={LEGACY_MAX_DEPTH}.",
                grow = crate::engine::leaf_slots(max_depth) / crate::engine::LEGACY_LEAVES,
                slots = crate::engine::leaf_slots(max_depth),
                LEGACY_LEAVES = crate::engine::LEGACY_LEAVES
            ),
        });
    }
    if let Some(groups) = &spec.interaction.groups {
        if groups.is_empty() {
            return Err(invalid_config(
                "interaction groups must be non-empty when set",
            ));
        }
        for group in groups {
            if group.order() == 0 || group.order() > usize::from(spec.interaction.max_order) {
                return Err(PbError::InvalidConfig {
                    what: format!(
                        "interaction group order must be in 1..={}, got {}",
                        spec.interaction.max_order,
                        group.order()
                    ),
                });
            }
        }
    }
    if !spec.interaction.table_budget_beta.is_finite() || spec.interaction.table_budget_beta < 0.0 {
        return Err(PbError::InvalidConfig {
            what: format!(
                "interaction.table_budget_beta must be finite and >= 0, got {}",
                spec.interaction.table_budget_beta
            ),
        });
    }
    if spec.interaction.table_budget_cells == 0 {
        return Err(PbError::InvalidConfig {
            what: "interaction.table_budget_cells must be > 0".into(),
        });
    }
    spec.credibility.validate()?;
    Ok(())
}

fn resolve_monotone(
    spec: &FitSpec<'_>,
    n_features: usize,
) -> Result<Vec<Option<MonoSign>>, PbError> {
    let mut out = vec![None; n_features];
    for (name, sign) in &spec.monotone {
        let Some(stripped) = name.strip_prefix('f') else {
            return Err(PbError::InvalidConfig {
                what: format!("unknown monotone feature `{name}`; expected default name f{{axis}}"),
            });
        };
        let axis = stripped
            .parse::<usize>()
            .map_err(|_| PbError::InvalidConfig {
                what: format!("unknown monotone feature `{name}`; expected default name f{{axis}}"),
            })?;
        if axis >= n_features {
            return Err(PbError::InvalidConfig {
                what: format!("monotone feature `{name}` is outside {n_features} model axes"),
            });
        }
        if !matches!(sign, MonoSign::None) {
            let slot = out.get_mut(axis).ok_or_else(|| PbError::Internal {
                what: "monotone axis escaped bounds".into(),
            })?;
            *slot = Some(*sign);
        }
    }
    Ok(out)
}

fn validate_binned_matrix(x: &BinnedMatrix) -> Result<(), PbError> {
    let n = x.n_rows as usize;
    let n_features = x.data.len();
    if x.grids.len() != n_features {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "BinnedMatrix grids len {} != data len {n_features}",
                x.grids.len()
            ),
        });
    }
    if x.provenance.len() != n_features {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "BinnedMatrix provenance len {} != data len {n_features}",
                x.provenance.len()
            ),
        });
    }

    for (axis, col) in x.data.iter().enumerate() {
        if col.len() != n {
            return Err(PbError::ShapeMismatch {
                what: format!("BinnedMatrix column {axis} len {} != n_rows {n}", col.len()),
            });
        }
        let grid = x.grids.get(axis).ok_or_else(|| PbError::Internal {
            what: "grid disappeared during BinnedMatrix validation".into(),
        })?;
        if grid.missing_bin != 0 {
            return Err(invalid_input(format!(
                "BinnedMatrix grid {axis} missing_bin must be 0, got {}",
                grid.missing_bin
            )));
        }
        if grid.n_bins == 0 || grid.n_bins > 255 {
            return Err(invalid_input(format!(
                "BinnedMatrix grid {axis} n_bins must be in 1..=255, got {}",
                grid.n_bins
            )));
        }
        let expected_bins = if grid.n_bins == 1 {
            if !grid.borders.is_empty() {
                return Err(invalid_input(format!(
                    "BinnedMatrix grid {axis} has n_bins=1 but {} borders",
                    grid.borders.len()
                )));
            }
            1usize
        } else {
            grid.borders
                .len()
                .checked_add(2)
                .ok_or_else(|| PbError::Internal {
                    what: "BinnedMatrix border count overflow".into(),
                })?
        };
        if expected_bins != usize::from(grid.n_bins) {
            return Err(invalid_input(format!(
                "BinnedMatrix grid {axis} n_bins {} != borders.len()+2 ({expected_bins})",
                grid.n_bins
            )));
        }
        for (i, &border) in grid.borders.iter().enumerate() {
            if !border.is_finite() {
                return Err(invalid_input(format!(
                    "BinnedMatrix grid {axis} border {i} must be finite"
                )));
            }
        }
        for pair in grid.borders.windows(2) {
            if let [a, b] = pair {
                if a >= b {
                    return Err(invalid_input(format!(
                        "BinnedMatrix grid {axis} borders must be strictly ascending"
                    )));
                }
            }
        }
        for (row, &bin) in col.iter().enumerate() {
            if u16::from(bin) >= grid.n_bins {
                return Err(invalid_input(format!(
                    "BinnedMatrix column {axis} row {row} bin {bin} outside grid n_bins {}",
                    grid.n_bins
                )));
            }
        }
    }
    Ok(())
}

/// Fit an ensemble (spec §06.6). See [`crate::engine::Booster::fit`].
///
/// # Errors
/// [`PbError::InvalidConfig`] on a bad config; [`PbError::ShapeMismatch`] on a
/// `y`/`weight` length mismatch; plus any propagated `Loss`/binning/grow error.
pub(crate) fn fit(
    config: &Config,
    x: &BinnedMatrix,
    y: &[f32],
    spec: &FitSpec,
    cat_encoders: &CatEncoderStore,
) -> Result<Model, PbError> {
    match &config.boosters.ensemble {
        EnsembleSpec::Off => fit_single(config, x, y, spec, cat_encoders),
        EnsembleSpec::OuterBag {
            n_bags,
            bag_subsample,
            cell_refit,
        } => fit_outer_bag(
            config,
            x,
            y,
            spec,
            *n_bags,
            *bag_subsample,
            *cell_refit,
            cat_encoders,
        ),
        EnsembleSpec::GreedySelect {
            library_size,
            hp_grid,
            selection_bags,
            seed_top_n,
        } => fit_greedy_select(
            config,
            x,
            y,
            spec,
            GreedyParams {
                library_size: *library_size,
                hp_grid,
                selection_bags: *selection_bags,
                seed_top_n: *seed_top_n,
            },
            cat_encoders,
        ),
    }
}

/// Dev-only fit profiler (gated by the `TBOOST_PROFILE` env var; a no-op otherwise so it never
/// affects production timing or determinism). Accumulates wall-time per named fit phase on the
/// calling (main) thread — parallel work inside a phase is captured at its main-thread call site —
/// and prints a breakdown to stderr at the end of `fit_single`. Pure measurement: no model effect.
pub(crate) mod prof {
    use std::cell::RefCell;
    use std::time::{Duration, Instant};

    thread_local! {
        static ENABLED: bool = std::env::var_os("TBOOST_PROFILE").is_some();
        static SPANS: RefCell<Vec<(&'static str, Duration)>> = const { RefCell::new(Vec::new()) };
    }

    pub(crate) fn enabled() -> bool {
        ENABLED.with(|e| *e)
    }
    fn add(name: &'static str, d: Duration) {
        SPANS.with(|s| {
            let mut v = s.borrow_mut();
            if let Some(e) = v.iter_mut().find(|(n, _)| *n == name) {
                e.1 += d;
            } else {
                v.push((name, d));
            }
        });
    }
    /// Time `f`, accumulating its wall-time under `name` (only when enabled). Returns `f`'s value.
    #[inline]
    pub(crate) fn timed<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
        if !enabled() {
            return f();
        }
        let t = Instant::now();
        let r = f();
        add(name, t.elapsed());
        r
    }
    pub(crate) fn reset() {
        SPANS.with(|s| s.borrow_mut().clear());
    }
    /// Current resident set size and its peak (high-water mark) in MB, from `/proc/self/status`.
    /// RSS is process-wide physical memory (includes the Python host + shared libs); the DELTAS
    /// between phase snapshots are what attribute memory to library steps.
    pub(crate) fn rss_mb() -> Option<(f64, f64)> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let (mut rss, mut hwm) = (None, None);
        for line in status.lines() {
            let mut it = line.split_whitespace();
            match it.next() {
                Some("VmRSS:") => rss = it.next().and_then(|v| v.parse::<f64>().ok()),
                Some("VmHWM:") => hwm = it.next().and_then(|v| v.parse::<f64>().ok()),
                _ => {}
            }
        }
        Some((rss? / 1024.0, hwm? / 1024.0))
    }
    /// Emit an RSS snapshot (gated by `TBOOST_PROFILE`) for a fit-lifecycle boundary.
    pub(crate) fn mem(label: &str) {
        if !enabled() {
            return;
        }
        if let Some((rss, hwm)) = rss_mb() {
            eprintln!("[mem] {label:<22} rss {rss:>6.0} MB   peak {hwm:>6.0} MB");
        }
    }
    pub(crate) fn report() {
        if !enabled() {
            return;
        }
        SPANS.with(|s| {
            let v = s.borrow();
            // Top-level spans (no '.') sum to wall-time; nested 'parent.child' spans are subsets.
            let top: f64 = v
                .iter()
                .filter(|(n, _)| !n.contains('.'))
                .map(|(_, d)| d.as_secs_f64())
                .sum();
            eprintln!(
                "[t-boost fit profile] top-level phases {top:.2}s (nested '.' spans are subsets):"
            );
            let mut rows: Vec<_> = v.iter().collect();
            rows.sort_by_key(|r| std::cmp::Reverse(r.1));
            for (n, d) in rows {
                let sec = d.as_secs_f64();
                let pct = if n.contains('.') {
                    String::new()
                } else {
                    format!("{:5.1}%", 100.0 * sec / top.max(1e-9))
                };
                eprintln!("  {n:24} {sec:8.3}s  {pct}");
            }
        });
    }
}

/// Rounds between forced refreshes of the incremental inverse-link cache (`Config::incremental_mu`):
/// `mu` is re-derived from a fresh `exp(F)` pass this often to reset the drift that the
/// round-to-round multiplicative update accumulates. Small enough that the f64 drift stays ~1e-12,
/// large enough that the amortized `exp` cost is ~1/64 of the every-round pass it replaces.
const INCREMENTAL_MU_REFRESH_ROUNDS: u32 = 64;

/// Class solves the K>=3 cell refit runs at once (see `attach_multiclass_cell_correction`): each
/// holds one `supports x fit_rows` design, so this caps the designs live together.
const MC_REFIT_CLASS_WAVE: usize = 8;

/// Best-round restore state for early-stop truncation (see the `best_snapshot` comment in
/// `fit_single`): the cheap `Alphas` form suffices when only alphas can be mutated after the
/// best round (AGBM, DART); `Trees` (a full clone) is only used when leaves can ALSO be
/// mutated (every-k-trees Ridge refit).
enum BestSnapshot {
    Alphas(Vec<f32>),
    Trees(Vec<(f32, ObliviousTree)>),
}

/// §07.6 `path_smooth` evidence generalization: which per-node quantity should feed the
/// Bühlmann credibility blend `Z = n/(n+k)`, resolved once per fit from the trained loss.
///
/// `Hessian` (per-node Σh, expressed as effective rows Σh/h̄) for Poisson and Tweedie: both are
/// log-link claim-count/pure-premium objectives whose expected (Fisher) information tracks
/// predicted claim mass, so a cell with many exposure rows but few actual claims (zero-inflated)
/// is correctly flagged as low-evidence — something raw row count cannot see. This is not just
/// theory: on the current engine (leaf-refine erosion fix in place) a 2026-07-23 SAME-BUILD
/// control on ohlsson_pp measured path_smooth's effect at −0.09% under hessian evidence but
/// +0.99% (actively HARMFUL) under count — count over-shrinks the zero-heavy Tweedie leaves that
/// hold plenty of rows but little claim evidence. (A pre-refine-fix screen had shown count −0.33%;
/// that was a cross-build artifact — the old refine step accidentally clawed back count's
/// over-shrinkage. Do not trust it.) `Count` (raw row count) for everything else: Gamma's
/// *observed* hessian is outlier-inflated (a thin cell with one huge loss would look MORE
/// credible, not less) and its *expected* information collapses to exposure/count anyway;
/// SquaredError/Logistic/Softmax have no analogous claim-mass concept.
fn credibility_evidence_for_loss(loss_id: crate::loss::LossId) -> CredibilityEvidence {
    match loss_id {
        crate::loss::LossId::Poisson | crate::loss::LossId::Tweedie => CredibilityEvidence::Hessian,
        crate::loss::LossId::SquaredError
        | crate::loss::LossId::Logistic
        | crate::loss::LossId::Gamma
        | crate::loss::LossId::Softmax => CredibilityEvidence::Count,
    }
}

/// A resolved, armed rate-collapse gate: the raw-space comparison point and the clamp to
/// install on engage. See [`crate::loss::GatedDeltaStep`] for the mechanism and semantics.
#[derive(Debug, Clone, Copy)]
struct ArmedDeltaStepGate {
    /// `ln(collapse_threshold)`.
    log_threshold: f64,
    /// `min(resolved max_delta_step, capped_step)` — the gate only ever tightens.
    capped_step: f64,
}

/// Resolve [`Config::max_delta_step_gated`] into an armed gate, or `None` for "no gate".
///
/// Precedence, highest first:
///  1. An EXPLICIT `config.max_delta_step` ⇒ never gate. The caller named a cap; honoring it
///     for the whole fit is what "explicit wins over everything" means, and it is also the
///     escape hatch a tuned champion uses to opt out.
///  2. [`GatedStepPolicy::Off`] ⇒ never gate.
///  3. [`GatedStepPolicy::On(g)`] ⇒ gate with `g`.
///  4. [`GatedStepPolicy::Objective`] (the default) ⇒ [`Loss::gated_max_delta_step`], which is
///     `Some` for Tweedie ONLY.
///
/// A gate whose `capped_step` does not actually tighten the resolved clamp still resolves to
/// `Some`, but engaging it is then a no-op on the arithmetic (`min` keeps the resolved value),
/// so the fit stays bit-identical and the report still records what was seen.
fn resolve_delta_step_gate(
    config: &Config,
    loss: &dyn Loss,
    resolved_max_delta_step: Option<f64>,
) -> Option<ArmedDeltaStepGate> {
    if config.max_delta_step.is_some() {
        return None;
    }
    let g = match config.max_delta_step_gated {
        GatedStepPolicy::Off => return None,
        GatedStepPolicy::On(g) => g,
        GatedStepPolicy::Objective => loss.gated_max_delta_step()?,
    };
    let capped = f64::from(g.capped_step);
    Some(ArmedDeltaStepGate {
        log_threshold: g.log_threshold(),
        capped_step: match resolved_max_delta_step {
            Some(d) => d.min(capped),
            None => capped,
        },
    })
}

/// Per-fit state of the rate-collapse gate: monotone `engaged` flag plus the diagnostic
/// accumulators. Inert (and zero-cost per round) when `gate` is `None`.
#[derive(Debug, Clone, Copy)]
struct DeltaStepGateState {
    gate: Option<ArmedDeltaStepGate>,
    engaged_round: Option<u32>,
    min_log_rate_ratio: f64,
    rounds_checked: u64,
}

impl DeltaStepGateState {
    fn new(gate: Option<ArmedDeltaStepGate>) -> Self {
        Self {
            gate,
            engaged_round: None,
            min_log_rate_ratio: f64::INFINITY,
            rounds_checked: 0,
        }
    }

    /// Round-`t` check, run at the TOP of the round against the model's real accumulated raw
    /// scores (`raw`, never the AGBM/DART lookahead).
    ///
    /// Returns `true` on the round the gate FIRST engages, so the caller can install the
    /// tighter clamp once. Costs nothing once engaged: the scan stops for the rest of the fit
    /// (the gate is monotone, so a second observation could not change the outcome).
    ///
    /// Determinism: a sequential f64 `min` over `train_rows` in fixed index order. Same value
    /// on 1 thread and on 20; a function of `(data, config, seed)` and nothing else.
    fn observe(
        &mut self,
        t: u32,
        raw: &[f32],
        offset: Option<&[f32]>,
        f0: f64,
        train_rows: &[u32],
    ) -> bool {
        let Some(gate) = self.gate else { return false };
        if self.engaged_round.is_some() {
            return false;
        }
        // s_i = F_i − offset_i − f0 = ln(μ_i / weighted-mean-rate). At t = 0 this is 0 for
        // every row (`raw` is seeded to `f0 + offset`), so a non-collapsing fit can never
        // trip on its first round.
        let mut min_s = f64::INFINITY;
        for &r in train_rows {
            let i = r as usize;
            let Some(&fi) = raw.get(i) else { continue };
            let oi = offset.map_or(0.0, |o| o.get(i).copied().unwrap_or(0.0));
            let s = f64::from(fi) - f64::from(oi) - f0;
            if s < min_s {
                min_s = s;
            }
        }
        self.rounds_checked += 1;
        if min_s < self.min_log_rate_ratio {
            self.min_log_rate_ratio = min_s;
        }
        if min_s < gate.log_threshold {
            self.engaged_round = Some(t);
            return true;
        }
        false
    }

    /// The fit's report, or `None` when no gate was armed.
    fn report(&self) -> Option<DeltaStepGateReport> {
        let gate = self.gate?;
        Some(DeltaStepGateReport {
            engaged: self.engaged_round.is_some(),
            engaged_round: self.engaged_round,
            bags_engaged: u32::from(self.engaged_round.is_some()),
            bags_total: 1,
            min_log_rate_ratio: self.min_log_rate_ratio,
            log_threshold: gate.log_threshold,
            capped_step: gate.capped_step,
            rounds_checked: self.rounds_checked,
        })
    }
}

fn fit_single(
    config: &Config,
    x: &BinnedMatrix,
    y: &[f32],
    spec: &FitSpec,
    cat_encoders: &CatEncoderStore,
) -> Result<Model, PbError> {
    config.validate()?;
    validate_fit_spec(config, spec)?;
    validate_binned_matrix(x)?;
    let n = x.n_rows as usize;
    if y.len() != n {
        return Err(PbError::ShapeMismatch {
            what: format!("y len {} != n_rows {n}", y.len()),
        });
    }

    // Weights default to all-ones; `ones_storage` backs that borrow for the fn body.
    let ones_storage: Vec<f32>;
    let weight: &[f32] = match spec.weight {
        Some(w) => {
            if w.len() != n {
                return Err(PbError::ShapeMismatch {
                    what: format!("weight len {} != n_rows {n}", w.len()),
                });
            }
            w
        }
        None => {
            ones_storage = vec![1.0_f32; n];
            &ones_storage
        }
    };

    // Exposure → per-row offset (§03.7); folded into the raw score, not into binning.
    let offset: Option<Vec<f32>> = match spec.exposure {
        Some(e) => Some(compute_offset(e, n)?),
        None => None,
    };

    let n_features = x.data.len();
    let monotone = resolve_monotone(spec, n_features)?;
    let monotone_ref = if monotone.iter().any(Option::is_some) {
        Some(monotone.as_slice())
    } else {
        None
    };
    let axes: Vec<u32> = (0..u32::try_from(n_features).map_err(|_| PbError::Internal {
        what: "more than u32::MAX features".into(),
    })?)
        .collect();
    // ES-stratify fix: rank the internal early-stopping holdout WITHIN each stratum (see
    // es_strata_for_loss's doc for which objectives stratify on what) so a rare positive class
    // (Logistic) or the positive-event rows (Poisson/Tweedie, on high-zero-mass data) aren't
    // starved of holdout representation. Every other objective is unaffected (strata stays None,
    // and carve_validation_rows_stratified(..., None) is byte-identical to the old
    // carve_validation_rows). `fixed_holdout`-driven carves (the ordered-TS "one honest holdout"
    // contract, or a caller-supplied group-honest carve) never reach this — they take the
    // `Some(mask)` arm below.
    let strata: Option<Vec<u32>> = es_strata_for_loss(spec.loss.objective_tag().loss, y);
    let (mut train_rows, mut validation_rows) = match spec.fixed_holdout {
        Some(mask) => split_rows_by_mask(x.n_rows, mask)?,
        None => carve_validation_rows_stratified(
            x.n_rows,
            config.validation_fraction,
            spec.seed,
            strata.as_deref(),
        )?,
    };
    ensure_validation_mass(
        &mut train_rows,
        &mut validation_rows,
        weight,
        spec.fixed_holdout.is_some(),
    )?;
    // §H6: `RefitProblem::train_rows` wants `usize` (to reuse `deviance_for_rows`); computed
    // once here since it does not change across rounds.
    let train_rows_usize: Vec<usize> = train_rows.iter().map(|&r| r as usize).collect();

    // f0 = link(weighted mean) in f64 (the exact fANOVA intercept); down-cast once.
    // Intercept honesty: with a validation carve (internal or fixed_holdout) the weighted
    // mean is taken over TRAIN rows only — the full-array mean leaked the holdout's target
    // level into every early-stopping evaluation through f0. Without a carve, train_rows is
    // every row and the full-slice call below is byte-identical to the previous behavior.
    let f0 = if train_rows.len() != n {
        let gather = |src: &[f32]| -> Result<Vec<f32>, PbError> {
            train_rows
                .iter()
                .map(|&r| {
                    src.get(r as usize)
                        .copied()
                        .ok_or_else(|| PbError::Internal {
                            what: "train row escaped fit input arrays".into(),
                        })
                })
                .collect()
        };
        let train_y = gather(y)?;
        let train_w = gather(weight)?;
        let train_off = offset.as_deref().map(gather).transpose()?;
        spec.loss
            .init_score(&train_y, &train_w, train_off.as_deref())?
    } else {
        spec.loss.init_score(y, weight, offset.as_deref())?
    };
    let f0_f32 = f0 as f32;
    let mut raw = vec![f0_f32; n];
    if let Some(off) = &offset {
        for (r, o) in raw.iter_mut().zip(off) {
            *r += o;
        }
    }
    // Resolve the leaf-stage |w*|-clamp (§05.6): an explicit Config value wins, else fall
    // back to the loss's advertised cap (Poisson ⇒ Some(0.7)).
    let max_delta_step = config
        .max_delta_step
        .or_else(|| spec.loss.max_delta_step())
        .map(f64::from);
    // §05.6 addendum (av35): arm the in-fit rate-collapse gate. `None` here ⇒ every line below
    // that mentions the gate is dead and this fit is bit-identical to the pre-gate engine.
    let gate = resolve_delta_step_gate(config, spec.loss, max_delta_step);
    let mut gate_state = DeltaStepGateState::new(gate);
    let mut grow_cfg = GrowConfig {
        lambda: f64::from(config.lambda),
        l1_leaf: f64::from(config.l1_leaf),
        lr: f64::from(config.learning_rate),
        min_split_gain: f64::from(config.min_split_gain),
        interaction_gain_hurdle: f64::from(config.interaction_gain_hurdle),
        max_order: spec.interaction.max_order,
        max_depth: spec.interaction.max_depth,
        max_delta_step,
        hist_precision: config.hist_precision,
        quant_seed: spec.seed,
        round: 0,
        random_strength: f64::from(config.boosters.random_strength),
        groups: spec.interaction.groups.as_deref(),
        monotone: monotone_ref,
        table_budget_penalty: TableBudgetPenalty::new(
            f64::from(spec.interaction.table_budget_beta),
            spec.interaction.table_budget_cells,
            f64::from(spec.interaction.table_budget_order_shrink),
        ),
        // Set fresh on each round's cloned config below (`round_grow_cfg`/`correction_cfg`),
        // never here: `grow_cfg` itself lives across the whole round loop (cloned every
        // iteration), so a borrow baked in at this template would stay alive for the entire
        // loop and block the mutable access `RealizedExtent::record_tree` needs after each
        // round.
        realized_extent: None,
        credibility: spec.credibility,
        credibility_evidence: credibility_evidence_for_loss(spec.loss.objective_tag().loss),
        // No caller weights ⇒ `weight` is the materialized all-ones, so the histogram can
        // set `wsum = count` instead of summing 1.0 per row (bit-exact). Provided weights
        // (even if all 1.0) keep the full Σw path — conservative and always correct.
        // An explicit all-ones weight is the unit weight (see fit_multiclass_single): the
        // histogram skips its weighted sum, byte-identical since a sum of 1.0s IS the count.
        unit_weight: spec.weight.is_none(),
        // Level-2 histogram subtraction on by default (FullF64); ~half the level-2 row visits,
        // accuracy moves only at ~1e-11. A kill-switch if a near-tie split ever proves sensitive.
        hist_subtraction: true,
    };
    // Owned by THIS fit's round loop (spec §07.4 stage 5): the per-raw-feature realized
    // split-border accumulator `table_budget_penalty` reads instead of the axis's full
    // `grid.n_bins`. One instance per `fit_single` call -- bagging fits each bag through an
    // independent call (see `fit_outer_bag`), so this never crosses a bag/thread boundary.
    let mut realized_extent = RealizedExtent::new(n_features);

    let precise_append = matches!(config.boosters.nesterov, NesterovSpec::Off)
        && config
            .boosters
            .dart
            .as_ref()
            .is_none_or(|dart| dart.drop_rate == 0.0);
    let mut precise_raw: Vec<f64> = if precise_append {
        (0..n)
            .map(|row| base_raw(offset.as_deref(), f0_f32, row))
            .collect::<Result<_, _>>()?
    } else {
        Vec::new()
    };
    let mut trees: Vec<(f32, ObliviousTree)> = Vec::new();
    let mut gh = GradHess::default();
    let mut last_refit_tree_count = 0usize;
    let mut prev_alphas: Vec<f32> = Vec::new();
    let mut interaction_gain_reference: Option<f64> = None;
    // Reusable gather scratch for the early-stopping deviance: the round loop below re-evaluates
    // deviance_for_rows_scratch every round against this SAME validation_rows, so hoisting the
    // three gather buffers here (instead of letting deviance_for_rows allocate fresh ones per
    // call) amortizes their allocation to once per fit instead of once per round.
    let mut es_y_sub: Vec<f32> = Vec::new();
    let mut es_raw_sub: Vec<f32> = Vec::new();
    let mut es_weight_sub: Vec<f32> = Vec::new();
    let mut best_validation_deviance = match validation_rows.as_deref() {
        Some(val_rows) => Some(deviance_for_rows_scratch(
            spec.loss,
            y,
            &raw,
            weight,
            val_rows,
            &mut es_y_sub,
            &mut es_raw_sub,
            &mut es_weight_sub,
        )?),
        None => None,
    };
    let mut best_validation_tree_count = 0usize;
    let dart_active = config
        .boosters
        .dart
        .as_ref()
        .is_some_and(|d| d.drop_rate > 0.0);
    // Best-model snapshot for truncation restore: AGBM (lookahead alpha mix) and DART
    // (dropped-tree rescale) mutate only ALPHAS on rounds after the best one (set_tree_alphas /
    // apply_dart_normalization never touch a tree's leaves); every-k-trees Ridge refit mutates
    // only LEAVES (write_leaf_theta never touches alpha). Either way, `trees.truncate(..)` alone
    // would keep the retained prefix's FINAL-round mutated state, not the state that actually
    // produced `best_validation_deviance`. `alphas_may_mutate` picks the cheap `Vec<f32>` form
    // (paired with `set_tree_alphas` on restore) when only alphas are at risk; the full tree
    // clone is reserved for when leaves can ALSO be dirtied. On the default plain-boosting path
    // neither is true, so no snapshot is ever taken.
    let leaves_may_mutate = matches!(
        config.boosters.refit_leaves,
        RefitSpec::Ridge {
            every_k_trees: Some(_),
            ..
        }
    );
    let alphas_may_mutate =
        matches!(config.boosters.nesterov, NesterovSpec::Agbm { .. }) || dart_active;
    let mut best_snapshot: Option<BestSnapshot> = None;

    // Task-A incremental inverse-link cache: maintain `mu = exp(F)` MULTIPLICATIVELY across rounds
    // (in `update_raw`) so each round's `grad_hess` reads the cache instead of an O(N) `exp(F)` pass.
    // Only on the plain log-link boosting path: AGBM/DART/ridge-refit recompute `raw` wholesale
    // mid-loop, which would desync `mu`, so any of them forces the exact `exp(F)` path (silent, since
    // this is a speed-only knob). `mu` mirrors `raw` over ALL rows; it is seeded here and refreshed
    // from a fresh `exp(F)` pass every `INCREMENTAL_MU_REFRESH_ROUNDS` rounds to bound drift.
    let use_inc_mu = config.incremental_mu
        && spec.loss.supports_incremental_mu()
        && matches!(config.boosters.nesterov, NesterovSpec::Off)
        && !dart_active
        && matches!(config.boosters.refit_leaves, RefitSpec::Off);
    let mut mu: Vec<f64> = Vec::new();
    if use_inc_mu {
        mu.try_reserve_exact(n).map_err(|_| PbError::Internal {
            what: "incremental mu buffer allocation failed".into(),
        })?;
        mu.resize(n, 0.0);
        spec.loss.refresh_mu(&raw, &mut mu)?;
    }
    let mut rounds_since_mu_refresh: u32 = 0;

    // Leaf-refine's `y`/`weight` subset gather (`refine_tree_leaves_after_grow`'s `y_sub`/`w_sub`)
    // reads only `y`/`weight` at `rows`, and every call below passes `&train_rows` — fixed once
    // above, never resampled per round — so the gathered values are identical on every round.
    // Precompute them ONCE here rather than re-copying `train_rows.len()` elements out of `y`/
    // `weight` on every round (twice a round when the AGBM momentum-correction refine also fires).
    // Gated to `Sampling::Full` only: that is the mode this reasoning is verified for; any other
    // `Sampling` variant keeps `refine_tree_leaves_after_grow`'s own per-round gather unchanged, so
    // a future sampling mode that DID vary the refine row set could never silently read stale data
    // here. `leaf_refine_steps == 0` skips it too (refine returns before ever gathering anything).
    let leaf_refine_hoisted_y_w: Option<(Vec<f32>, Vec<f32>)> =
        if matches!(config.sampling, Sampling::Full) && config.leaf_refine_steps > 0 {
            Some((
                gather_rows(y, &train_rows)?,
                gather_rows(weight, &train_rows)?,
            ))
        } else {
            None
        };
    let leaf_refine_hoisted_y_w_ref: Option<(&[f32], &[f32])> = leaf_refine_hoisted_y_w
        .as_ref()
        .map(|(y_sub, w_sub)| (y_sub.as_slice(), w_sub.as_slice()));

    // Run-time controls (R2/R3): the per-round history and observer read deviances only, so a
    // fit without them takes none of the branches below and stays bit-identical. The stopping
    // deviance is computed for early stopping anyway; the training deviance is an extra O(n)
    // pass per round (~10-20% of a fit), so only an observer pays for it.
    let control = &config.fit_control;
    let track_rounds = control.record_history || control.observer.is_some();
    let track_train = control.observer.is_some();
    let mut history_train: Vec<f64> = Vec::new();
    let mut history_eval: Vec<f64> = Vec::new();
    let (mut tr_y_sub, mut tr_raw_sub, mut tr_weight_sub) = (Vec::new(), Vec::new(), Vec::new());
    let mut rounds_trained: u32 = 0;
    let mut stop_reason = StopReason::MaxTrees;
    // Reported deviances are means: the summed deviance over each row set's total weight.
    let mass_of = |rows: &[usize]| -> f64 {
        rows.iter()
            .filter_map(|&r| weight.get(r))
            .map(|&w| f64::from(w))
            .sum::<f64>()
            .max(f64::MIN_POSITIVE)
    };
    let (train_mass, eval_mass) = if track_rounds {
        (
            mass_of(&train_rows_usize),
            validation_rows.as_deref().map_or(1.0, mass_of),
        )
    } else {
        (1.0, 1.0)
    };

    prof::reset();
    for t in 0..config.n_trees {
        // §05.6 addendum (av35) rate-collapse gate. Read `raw` — the model's REAL accumulated
        // state after round t−1 — before AGBM/DART build this round's `fit_raw` lookahead, so
        // the decision is about the ensemble that exists, not a transient mixing artifact.
        // Mutating the loop-lifetime `grow_cfg` (rather than the per-round clone) is what makes
        // the engage monotone by construction and carries it into BOTH the main
        // `round_grow_cfg` and the AGBM `correction_cfg` cloned from it below.
        if gate_state.observe(t, &raw, offset.as_deref(), f0, &train_rows) {
            if let Some(gate) = gate_state.gate {
                grow_cfg.max_delta_step = Some(gate.capped_step);
            }
        }
        // Per-round deterministic re-seed — the seam for MVS/subsampling (M5-QHIST,
        // v1.5).
        let _round_rng = pb_rng(spec.seed, t, Stage::Sample, 0);
        let agbm = match config.boosters.nesterov {
            NesterovSpec::Off => None,
            NesterovSpec::Agbm {
                momentum_correction,
            } => Some((agbm_beta(t), momentum_correction)),
        };
        // `current_alphas`/`prev_alphas` only ever feed the AGBM lookahead mix (below) and the
        // no-split restore (the `None` arm further down); collecting them is wasted O(t)
        // alloc+copy on every other path (DART, plain), so skip it there — `Vec::new()` doesn't
        // allocate, and the empty placeholder is never read when `agbm` is `None` all fit long
        // (`config.boosters.nesterov` is fixed per fit, so this gate is the same every round).
        let current_alphas = if agbm.is_some() {
            collect_tree_alphas(&trees)?
        } else {
            Vec::new()
        };
        let mut fit_raw = raw.clone();
        if let Some((beta, _)) = agbm {
            let lookahead_alphas = combine_alphas(&current_alphas, &prev_alphas, beta)?;
            // Persist the mixed coefficients for the next round, serialization,
            // and early-stopping snapshots.
            set_tree_alphas(&mut trees, &lookahead_alphas)?;
            // Mixing rounded score caches loses small updates at large intercepts.
            // Reconstruct the lookahead from the actual leaves and mixed alphas.
            fit_raw = raw_from_tree_alphas(f0_f32, offset.as_deref(), x, &trees)?;
        }
        let dart_cfg = config
            .boosters
            .dart
            .as_ref()
            .filter(|dart| dart.drop_rate > 0.0);
        let dart_drops = if agbm.is_none() {
            dart_drop_mask(dart_cfg, spec.seed, t, trees.len())?
        } else {
            Vec::new()
        };
        if dart_drops.iter().any(|dropped| *dropped) {
            fit_raw =
                raw_from_tree_alphas_kept(f0_f32, offset.as_deref(), x, &trees, Some(&dart_drops))?;
        }
        // Bound the incremental cache's drift: re-derive `mu = exp(F)` from a fresh pass every
        // `INCREMENTAL_MU_REFRESH_ROUNDS`. `fit_raw == raw` on this gated path (no AGBM/DART), and
        // `mu` tracks `raw`, so this re-aligns the cache to the exact value grad_hess would read.
        if use_inc_mu && rounds_since_mu_refresh >= INCREMENTAL_MU_REFRESH_ROUNDS {
            spec.loss.refresh_mu(&fit_raw, &mut mu)?;
            rounds_since_mu_refresh = 0;
        }
        prof::timed("grad_hess", || {
            // Incremental-mu path: read the maintained `mu = exp(fit_raw)` cache, skipping the O(N)
            // exp. Else, after round 0, a loss whose hessian is round-invariant (SquaredError:
            // h = w·max(floor)) refills ONLY the gradient and reuses the established `gh.h` —
            // bit-identical, fewer stores. Every other loss recomputes both via the full pass.
            if use_inc_mu {
                spec.loss.grad_hess_from_mu(y, &mu, weight, &mut gh)
            } else if t == 0 || spec.loss.hessian_depends_on_raw() {
                spec.loss.grad_hess(y, &fit_raw, weight, &mut gh)
            } else {
                spec.loss
                    .fill_grad_reusing_hessian(y, &fit_raw, weight, &mut gh)
            }
        })?;
        let (sampled_rows, mvs_reweight) =
            sample_rows(&config.sampling, &gh, spec.seed, t, &train_rows)?;
        // §06.5: MVS's sampled (g, h) are a gradient-biased sample of the population, not an
        // unbiased estimator, unless reweighted by 1/p_i first. `scaled_gh` is a fresh clone
        // (`gh` itself must stay exact — it is reused unmodified below for the full-data leaf
        // refit); `None` on the default `Sampling::Full` path, so no clone happens there.
        let scaled_gh = match &mvs_reweight {
            Some(mult) => Some(reweighted_gh(&gh, &sampled_rows, mult)?),
            None => None,
        };
        let grow_gh = scaled_gh.as_ref().unwrap_or(&gh);
        let round_axes = sample_axes(&axes, config.colsample_bytree, spec.seed, t)?;
        let mut round_grow_cfg = grow_cfg.clone();
        round_grow_cfg.round = t;
        round_grow_cfg.lr =
            learning_rate_for_round(config.learning_rate, config.learning_rate_decay, t);
        round_grow_cfg.realized_extent = Some(&realized_extent);
        round_grow_cfg.lambda = round_lambda(
            round_grow_cfg.lambda,
            config.lambda_scale_invariant,
            &gh,
            &train_rows,
        )?;
        let grown = prof::timed("grow_tree", || {
            grow_oblivious_tree_with_leaf_map(
                x,
                grow_gh,
                &sampled_rows,
                &round_axes,
                &round_grow_cfg,
                InteractionHurdleState::new(
                    config.interaction_gain_hurdle_mode,
                    interaction_gain_reference,
                ),
                weight,
            )
        })?;
        match grown {
            Some(grown) => {
                let first_split_gain = grown.first_split_gain;
                let mut tree = grown.tree;
                let leaf_of_row = grown.leaf_of_row;
                // grow's `leaf_of_row` is the per-row leaf partition over `sampled_rows`; it equals
                // the partition over `train_rows` exactly when grow saw the full set (no subsample),
                // in which case the line search can reuse it instead of re-walking the tree.
                let full_sample = sampled_rows.len() == train_rows.len();
                if !full_sample {
                    refit_tree_leaves(x, &gh, &train_rows, &mut tree, &round_grow_cfg)?;
                }
                prof::timed("leaf_refine", || {
                    refine_tree_leaves_after_grow(
                        config,
                        spec.loss,
                        y,
                        weight,
                        &fit_raw,
                        x,
                        &train_rows,
                        monotone_ref,
                        &mut tree,
                        &round_grow_cfg,
                        full_sample.then_some(leaf_of_row.as_slice()),
                        leaf_refine_hoisted_y_w_ref,
                    )
                })?;
                if let Some(dart) = dart_cfg {
                    let new_alpha = apply_dart_normalization(&mut trees, &dart_drops, dart)?;
                    // Supply the intermediate score to an optional leaf refit.
                    // The round boundary below reconstructs the retained ensemble
                    // in f64 before validation or the next gradient.
                    raw = raw_plus_dart_round(fit_raw, x, &trees, &dart_drops, &tree, new_alpha)?;
                    realized_extent.record_tree(&tree.splits, x)?;
                    trees.push((new_alpha, tree));
                } else {
                    prev_alphas = current_alphas;
                    raw = fit_raw;
                    // `raw` spans ALL rows (incl. any validation rows); grow's `leaf_of_row` covers
                    // every one of them only when grow saw the full set, so reuse it for the
                    // tree-walk-free update just then. Otherwise update_raw re-walks (unchanged).
                    // Reuse grow's leaf map for the train rows and re-walk ONLY the validation
                    // holdout (grow saw every train row when `full_sample`); else fall back to the
                    // covers-all fast path or a full walk. All branches update each row exactly once,
                    // byte-identically to a full re-walk.
                    let covers_all_rows = sampled_rows.len() == x.n_rows as usize;
                    // On the incremental-mu path, `mu` rides the SAME leaf application as `raw` here
                    // (lockstep), keeping `mu == exp(raw)` for the next round's grad_hess.
                    let mu_arg: Option<&mut [f64]> = if use_inc_mu {
                        Some(mu.as_mut_slice())
                    } else {
                        None
                    };
                    prof::timed("update_raw", || match validation_rows.as_deref() {
                        Some(val) if full_sample => update_raw_split(
                            &mut raw,
                            mu_arg,
                            x,
                            &tree,
                            &leaf_of_row,
                            &sampled_rows,
                            val,
                        ),
                        _ => update_raw(
                            &mut raw,
                            mu_arg,
                            x,
                            &tree,
                            covers_all_rows.then_some(leaf_of_row.as_slice()),
                        ),
                    })?;
                    if use_inc_mu {
                        rounds_since_mu_refresh += 1;
                    }
                    realized_extent.record_tree(&tree.splits, x)?;
                    if precise_append {
                        let columns = tree_split_columns(&tree, &x.data)?;
                        for (row, (precise, value)) in
                            precise_raw.iter_mut().zip(&mut raw).enumerate()
                        {
                            *precise +=
                                f64::from(tree_value_for_row_with_columns(&tree, &columns, row)?);
                            *value = *precise as f32;
                        }
                        if use_inc_mu && raw.iter().any(|value| value.abs() >= 30.0) {
                            spec.loss.refresh_mu(&raw, &mut mu)?;
                        }
                    }
                    trees.push((1.0, tree));
                }
                update_interaction_gain_reference(
                    &mut interaction_gain_reference,
                    first_split_gain,
                );
                if should_refit_after_round(&config.boosters.refit_leaves, trees.len())? {
                    let problem = RefitProblem {
                        spec,
                        x,
                        y,
                        weight,
                        offset: offset.as_deref(),
                        f0: f0_f32,
                        monotone: monotone_ref,
                        train_rows: &train_rows_usize,
                    };
                    fully_corrective_refit(
                        &config.boosters.refit_leaves,
                        &problem,
                        &mut trees,
                        &mut raw,
                    )?;
                    if precise_append {
                        precise_raw = raw64_from_tree_alphas_kept(
                            f0_f32,
                            offset.as_deref(),
                            x,
                            &trees,
                            None,
                        )?;
                        for (value, precise) in raw.iter_mut().zip(&precise_raw) {
                            *value = *precise as f32;
                        }
                    }
                    last_refit_tree_count = trees.len();
                }
                if matches!(
                    config.boosters.nesterov,
                    NesterovSpec::Agbm {
                        momentum_correction: true
                    }
                ) {
                    raw = raw_from_tree_alphas(f0_f32, offset.as_deref(), x, &trees)?;
                    spec.loss.grad_hess(y, &raw, weight, &mut gh)?;
                    let (correction_rows, correction_reweight) =
                        sample_rows(&config.sampling, &gh, spec.seed, t, &train_rows)?;
                    // §06.5, same as the main grow call above: reweight the sampled (g, h) by
                    // 1/p_i for split selection; `gh` itself stays exact for the full-data leaf
                    // refit that follows.
                    let correction_scaled_gh = match &correction_reweight {
                        Some(mult) => Some(reweighted_gh(&gh, &correction_rows, mult)?),
                        None => None,
                    };
                    let correction_grow_gh = correction_scaled_gh.as_ref().unwrap_or(&gh);
                    let mut correction_cfg = grow_cfg.clone();
                    correction_cfg.round = t;
                    correction_cfg.lr = learning_rate_for_round(
                        config.learning_rate,
                        config.learning_rate_decay,
                        t,
                    );
                    correction_cfg.realized_extent = Some(&realized_extent);
                    correction_cfg.lambda = round_lambda(
                        correction_cfg.lambda,
                        config.lambda_scale_invariant,
                        &gh,
                        &train_rows,
                    )?;
                    if let Some(correction_grown) = grow_oblivious_tree_with_leaf_map(
                        x,
                        correction_grow_gh,
                        &correction_rows,
                        &round_axes,
                        &correction_cfg,
                        InteractionHurdleState::new(
                            config.interaction_gain_hurdle_mode,
                            interaction_gain_reference,
                        ),
                        weight,
                    )? {
                        let correction_first_split_gain = correction_grown.first_split_gain;
                        let corr_leaf_of_row = correction_grown.leaf_of_row;
                        let mut correction = correction_grown.tree;
                        let corr_full = correction_rows.len() == train_rows.len();
                        if !corr_full {
                            refit_tree_leaves(
                                x,
                                &gh,
                                &train_rows,
                                &mut correction,
                                &correction_cfg,
                            )?;
                        }
                        refine_tree_leaves_after_grow(
                            config,
                            spec.loss,
                            y,
                            weight,
                            &raw,
                            x,
                            &train_rows,
                            monotone_ref,
                            &mut correction,
                            &correction_cfg,
                            corr_full.then_some(corr_leaf_of_row.as_slice()),
                            leaf_refine_hoisted_y_w_ref,
                        )?;
                        let corr_covers_all = correction_rows.len() == x.n_rows as usize;
                        // AGBM correction path: incompatible with `use_inc_mu` (which requires
                        // Nesterov Off), so the cache is never live here — exact update, `mu = None`.
                        update_raw(
                            &mut raw,
                            None,
                            x,
                            &correction,
                            corr_covers_all.then_some(corr_leaf_of_row.as_slice()),
                        )?;
                        realized_extent.record_tree(&correction.splits, x)?;
                        trees.push((1.0, correction));
                        update_interaction_gain_reference(
                            &mut interaction_gain_reference,
                            correction_first_split_gain,
                        );
                        if should_refit_after_round(&config.boosters.refit_leaves, trees.len())? {
                            let problem = RefitProblem {
                                spec,
                                x,
                                y,
                                weight,
                                offset: offset.as_deref(),
                                f0: f0_f32,
                                monotone: monotone_ref,
                                train_rows: &train_rows_usize,
                            };
                            fully_corrective_refit(
                                &config.boosters.refit_leaves,
                                &problem,
                                &mut trees,
                                &mut raw,
                            )?;
                            last_refit_tree_count = trees.len();
                        }
                    }
                }
                if !precise_append {
                    // DART/AGBM mutate coefficients and ridge can mutate leaves.
                    // The next gradient and this round's validation must see the
                    // same accumulated score as the retained ensemble.
                    raw = raw_from_tree_alphas(f0_f32, offset.as_deref(), x, &trees)?;
                }
                rounds_trained = rounds_trained.saturating_add(1);
                let mut eval_deviance = None;
                let mut patience_spent = false;
                if let Some(val_rows) = validation_rows.as_deref() {
                    let deviance = prof::timed("earlystop_eval", || {
                        deviance_for_rows_scratch(
                            spec.loss,
                            y,
                            &raw,
                            weight,
                            val_rows,
                            &mut es_y_sub,
                            &mut es_raw_sub,
                            &mut es_weight_sub,
                        )
                    })?;
                    eval_deviance = Some(deviance);
                    let improved = match best_validation_deviance {
                        Some(best) => config.is_material_improvement(best, deviance),
                        None => true,
                    };
                    if improved {
                        best_validation_deviance = Some(deviance);
                        best_validation_tree_count = trees.len();
                        if leaves_may_mutate {
                            best_snapshot = Some(BestSnapshot::Trees(trees.clone()));
                        } else if alphas_may_mutate {
                            best_snapshot =
                                Some(BestSnapshot::Alphas(collect_tree_alphas(&trees)?));
                        }
                    } else if trees.len().saturating_sub(best_validation_tree_count)
                        >= config.effective_patience(best_validation_tree_count)
                    {
                        patience_spent = true;
                    }
                }
                if track_rounds {
                    let train_deviance = if track_train {
                        deviance_for_rows_scratch(
                            spec.loss,
                            y,
                            &raw,
                            weight,
                            &train_rows_usize,
                            &mut tr_y_sub,
                            &mut tr_raw_sub,
                            &mut tr_weight_sub,
                        )? / train_mass
                    } else {
                        f64::NAN
                    };
                    let eval_deviance = eval_deviance.map(|d| d / eval_mass);
                    if control.record_history {
                        if track_train {
                            history_train.push(train_deviance);
                        }
                        history_eval.extend(eval_deviance);
                    }
                    if let Some(observer) = &control.observer {
                        let event = RoundEvent {
                            bag: control.bag,
                            round: rounds_trained,
                            n_trees: config.n_trees,
                            train_deviance,
                            eval_deviance,
                        };
                        if (observer.0)(&event) {
                            stop_reason = StopReason::Callback;
                            break;
                        }
                    }
                }
                if patience_spent {
                    stop_reason = StopReason::EarlyStopping;
                    break;
                }
            }
            // No admissible split clears the floor (e.g. converged / constant target):
            // stop early with what we have — a valid (possibly empty) Exact model.
            None => {
                if agbm.is_some() {
                    set_tree_alphas(&mut trees, &current_alphas)?;
                }
                stop_reason = StopReason::NoSplit;
                break;
            }
        }
    }
    if validation_rows.is_some() && best_validation_tree_count < trees.len() {
        // Restore the exact tree state (alphas, and leaves when a mutator can dirty those too)
        // that produced `best_validation_deviance`; a plain truncate is only correct because on
        // the plain path trees are append-only (no booster above ever rewrites an earlier tree's
        // alpha/leaves).
        match best_snapshot.take() {
            Some(BestSnapshot::Trees(snapshot)) => trees = snapshot,
            Some(BestSnapshot::Alphas(alphas)) => {
                trees.truncate(best_validation_tree_count);
                set_tree_alphas(&mut trees, &alphas)?;
            }
            None => trees.truncate(best_validation_tree_count),
        }
        // Any mid-loop refit (if one ran) was solved jointly with the pre-truncation, LONGER tree
        // set, so its leaf values are stale for this shorter/restored prefix. Reset to the same
        // sentinel used when no refit has happened yet, so `should_refit_at_end` below
        // unconditionally re-solves the kept prefix whenever `RefitSpec::Ridge` is configured
        // (idempotent, and hence harmless, in the rare case the stale solve already matches).
        last_refit_tree_count = 0;
        raw = raw_from_tree_alphas(f0_f32, offset.as_deref(), x, &trees)?;
    }
    if should_refit_at_end(
        &config.boosters.refit_leaves,
        trees.len(),
        last_refit_tree_count,
    ) {
        let problem = RefitProblem {
            spec,
            x,
            y,
            weight,
            offset: offset.as_deref(),
            f0: f0_f32,
            monotone: monotone_ref,
            train_rows: &train_rows_usize,
        };
        fully_corrective_refit(
            &config.boosters.refit_leaves,
            &problem,
            &mut trees,
            &mut raw,
        )?;
    }

    let mut f0_eff = f64::from(f0_f32);
    // An external evaluation holdout only scores early stopping: it never fits the slope.
    if config.boosters.reanchor_slope && !control.external_holdout {
        if let Some(val_rows) = validation_rows.as_deref() {
            if let Some((a, b)) =
                fit_affine_reanchor(spec.loss, y, &raw, weight, offset.as_deref(), val_rows)?
            {
                for (alpha, _) in trees.iter_mut() {
                    let scaled = f64::from(*alpha) * a;
                    if !scaled.is_finite()
                        || scaled < f64::from(f32::MIN)
                        || scaled > f64::from(f32::MAX)
                    {
                        return Err(PbError::InvalidInput {
                            what: "affine-reanchored tree weight is not representable as f32"
                                .into(),
                        });
                    }
                    *alpha = scaled as f32;
                }
                // raw = offset + F_model, so the corrected score is offset + a·F_model + b.
                match offset.as_deref() {
                    Some(off) => {
                        for (r, o) in raw.iter_mut().zip(off) {
                            *r = (f64::from(*o) + a * (f64::from(*r) - f64::from(*o)) + b) as f32;
                        }
                    }
                    None => {
                        for r in raw.iter_mut() {
                            *r = (a * f64::from(*r) + b) as f32;
                        }
                    }
                }
                f0_eff = a * f0_eff + b;
                if prof::enabled() {
                    eprintln!(
                        "[affine] a={a:.4} b={b:+.5} ({} holdout rows)",
                        val_rows.len()
                    );
                }
            }
        }
    }
    let f0_model = {
        let shifted = if config.boosters.reanchor && control.external_holdout {
            // The intercept re-anchors on the training rows only: an external evaluation
            // holdout's targets must not reach the model.
            let train_y = gather_rows(y, &train_rows)?;
            let train_w = gather_rows(weight, &train_rows)?;
            let train_raw = gather_rows(&raw, &train_rows)?;
            f0_eff + reanchor_delta(spec.loss.link(), &train_y, &train_w, &train_raw, None)?
        } else if config.boosters.reanchor {
            f0_eff + reanchor_delta(spec.loss.link(), y, weight, &raw, None)?
        } else {
            f0_eff
        };
        if !shifted.is_finite() || shifted < f64::from(f32::MIN) || shifted > f64::from(f32::MAX) {
            return Err(PbError::InvalidInput {
                what: "reanchored intercept is not representable as f32".into(),
            });
        }
        shifted as f32
    };

    prof::report();
    let schema = ModelSchema {
        feature_names: (0..n_features).map(|i| format!("f{i}")).collect(),
        feature_kinds: x.provenance.iter().map(|p| p.kind).collect(),
        // Carry the (full-data) categorical encoders so every model this builds — including
        // each OuterBag/GreedySelect member — validates and serves correctly, not just the
        // single-fit path stamped by `Booster::fit_train`.
        cat_encoders: cat_encoders.clone(),
        class_labels: None,
        objective: spec.loss.objective_tag(),
    };
    let trees_kept = trees.len();
    let mut model = Model {
        f0: f0_model,
        trees,
        grids: x.grids.clone(),
        provenance: x.provenance.clone(),
        link: spec.loss.link(),
        mode: ExactnessMode::Exact,
        schema,
        schema_version: SCHEMA_VERSION_UNLIFTED,
        correction: None,
        bag_spans: None,
        bag_intercepts: None,
        bag_in_bag: None,
        delta_step_gate: gate_state.report(),
        fit_report: Some(vec![BagFitReport {
            trees_kept: u32::try_from(trees_kept).unwrap_or(u32::MAX),
            rounds_trained,
            reason: stop_reason,
            train_deviance: history_train,
            eval_deviance: history_eval,
        }]),
    };
    // The stamp is the MINIMUM a reader needs: `SCHEMA_VERSION_UNLIFTED` unless this fit
    // actually grew a tree past the legacy depth cap (see `serialize::SCHEMA_VERSION`).
    model.schema_version = model.required_schema_version();
    Ok(model)
}

/// Native-softmax (multinomial) multiclass fit — see `design/multiclass-design.md`.
///
/// Trains `n_classes` additive raw score functions JOINTLY: each round computes per-class softmax
/// gradients from a single softmax over all K current raw columns, grows one oblivious tree per
/// class, and updates that class's column. Returns K exactly-decomposable scalar [`Model`]s (each
/// a class logit, [`Link::Identity`]) wrapped in a [`MultiClassModel`]; probabilities are
/// `softmax(F_0..F_{K-1})`, applied at the container response layer (outside the additive raw
/// space) so exactness holds per class.
///
/// `y` holds integer class labels in `0..n_classes` as `f32` (validated integral + in range).
/// `class_labels` are the human-readable labels (length `n_classes`). This entry dispatches on
/// `config.boosters.ensemble`: `Off` fits the single base path below; `OuterBag` fits `n_bags`
/// subagged multiclass models and soups them PER CLASS (see [`fit_multiclass_bagged`]), the same
/// variance-reduction architecture the single-output path ships; `GreedySelect` is
/// single-output-only and errors. The base path does NOT apply the
/// DART/Nesterov/fully-corrective-refit/reanchor levers. It DOES honor
/// `lambda`, `l1_leaf`, learning rate (+decay), `min_split_gain`, `max_delta_step`,
/// `colsample_bytree`, interaction order, `random_strength`, histogram precision, monotone
/// constraints, credibility floors, internal-validation early stopping, and — as of
/// 2026-08-23 — §06.5 MVS
/// row sampling ([`mc_sample_rows`]: one joint-gradient-norm row draw per round, shared by the
/// round's K class trees, with the leaves re-solved on all train rows). Deterministic:
/// rounds remain sequential; independent class work is stored in class-index order and
/// validation folds are fixed-order, so the result is byte-identical across thread counts.
///
/// # Errors
/// [`PbError::InvalidConfig`]/[`PbError::InvalidInput`]/[`PbError::ShapeMismatch`] on bad config,
/// labels, or class-label count; plus any propagated grow/update/validate error.
pub(crate) fn fit_multiclass(
    config: &Config,
    x: &BinnedMatrix,
    y: &[f32],
    n_classes: usize,
    class_labels: &[String],
    spec: &FitSpec,
    cat_encoders: &CatEncoderStore,
) -> Result<MultiClassModel, PbError> {
    validate_ensemble_fit_inputs(config, x, y, spec)?;
    match &config.boosters.ensemble {
        EnsembleSpec::Off => {
            fit_multiclass_single(config, x, y, n_classes, class_labels, spec, cat_encoders)
        }
        EnsembleSpec::OuterBag {
            n_bags,
            bag_subsample,
            cell_refit,
        } => fit_multiclass_bagged(
            config,
            x,
            y,
            n_classes,
            class_labels,
            spec,
            cat_encoders,
            *n_bags,
            *bag_subsample,
            *cell_refit,
        ),
        EnsembleSpec::GreedySelect { .. } => Err(PbError::InvalidConfig {
            what: "greedy-select ensembles are single-output-only; multiclass (K>=3) supports \
                   Off or OuterBag"
                .into(),
        }),
    }
}

/// `n_bags` subagged multiclass fits, souped PER CLASS with `alpha = 1/n_bags` — the exact
/// [`fit_outer_bag`] architecture applied to the K-column softmax path. Bags share the binned
/// matrix and differ only in their row subsets; each bag's fit is the deterministic
/// [`fit_multiclass_single`] seeded by its index, bags collect in bag order, and
/// [`soup_models`] folds each class's members in that fixed order — byte-identical across
/// thread counts. With `spec.fixed_holdout` set (e.g. the group-honest carve for panel data),
/// every bag trains on subagged train-proper rows plus the SHARED holdout appended, early-
/// stopping against that one honest slice — bags never leak holdout rows into training.
///
/// `cell_refit` (`Some` only when the caller asked for the §G1 refit) additionally has every bag
/// score its OUT-OF-BAG rows on all K class columns, and hands the K jury means to
/// [`attach_multiclass_cell_correction`] — the K-class analogue of [`fit_outer_bag`]'s
/// `attach_cell_correction`. With `cell_refit == None` nothing extra is computed or scored and
/// the fit is byte-identical to the pre-2026-08-25 path.
fn fit_multiclass_bagged(
    config: &Config,
    x: &BinnedMatrix,
    y: &[f32],
    n_classes: usize,
    class_labels: &[String],
    spec: &FitSpec,
    cat_encoders: &CatEncoderStore,
    n_bags: u16,
    bag_subsample: f32,
    cell_refit: Option<CellRefit>,
) -> Result<MultiClassModel, PbError> {
    let base_config = ensemble_base_config(config);
    // A single bag has no out-of-bag complement, so there is no honest evidence to refit
    // against. The scalar path gets this from `Config::validate` (which the bagged K>=3 path
    // never reaches for the OuterBag block, because every bag fit validates the DERIVED
    // ensemble-Off config); re-state it here verbatim so both paths refuse identically instead
    // of K>=3 silently dropping the correction the caller asked for.
    if n_bags < 2 && cell_refit.is_some() {
        return Err(PbError::InvalidConfig {
            what: "OuterBag cell_refit requires n_bags >= 2 (needs an OOB residual)".into(),
        });
    }
    if n_bags == 1 {
        return fit_multiclass_single(
            &base_config,
            x,
            y,
            n_classes,
            class_labels,
            spec,
            cat_encoders,
        );
    }
    if n_bags == 0 {
        return Err(PbError::InvalidConfig {
            what: "multiclass outer bag requires n_bags >= 1".into(),
        });
    }
    let n_rows = x.n_rows as usize;
    let alpha = 1.0_f64 / f64::from(n_bags);
    // Same subagging semantics as fit_outer_bag: >= 1 is a full-size bootstrap WITH replacement,
    // < 1 a without-replacement subag of round(f * n_rows) rows (leak-free vs per-bag carves).
    let subsample = bag_subsample < 1.0;
    let sample_len =
        (((n_rows as f64) * f64::from(bag_subsample)).round() as usize).clamp(1, n_rows);
    // Every bag stratifies on the K-way class label (always available for Softmax) — the
    // multiclass analog of fit_outer_bag's `es_strata_for_loss` gate, but unconditional since
    // there is no "no strata" case for a K-class fit.
    let strata = multiclass_labels(y, n_classes)?;
    // Only the §G1 path needs each bag's out-of-bag class columns; without it nothing is scored.
    let collect_oob = cell_refit.is_some();
    type McBagMember = (MultiClassModel, Vec<bool>, Option<Vec<Vec<f32>>>);
    let bag_members: Vec<McBagMember> = (0..usize::from(n_bags))
        .into_par_iter()
        .map(|bag| -> Result<McBagMember, PbError> {
            let bag_round = u32::try_from(bag).map_err(|_| PbError::Internal {
                what: "multiclass bag index exceeded u32".into(),
            })?;
            // Mirrors fit_outer_bag's two arms: a caller-fixed holdout is SHARED (bags train on
            // subagged train-proper rows + the appended holdout, masked so the fit early-stops
            // on it); otherwise each bag's own (stratified) carve runs inside the single fit.
            let (mut rows, mut bag_mask): (Vec<u32>, Option<Vec<bool>>) = match spec.fixed_holdout {
                Some(mask) => {
                    let train_idx: Vec<u32> =
                        (0..n_rows as u32).filter(|&r| !mask[r as usize]).collect();
                    let bag_len = ((train_idx.len() as f64) * (sample_len as f64) / (n_rows as f64))
                        .ceil()
                        .max(1.0) as usize;
                    let picked = if let Some(g) = spec.bag_groups {
                        let train_groups = gather_strata(g, &train_idx)?;
                        let train_strata = gather_strata(&strata, &train_idx)?;
                        group_subagging_rows(
                            spec.seed,
                            bag_round,
                            &train_groups,
                            Some(&train_strata),
                            bag_len,
                        )?
                    } else if subsample {
                        let train_strata = gather_strata(&strata, &train_idx)?;
                        subagging_rows_dispatch(
                            spec.seed,
                            bag_round,
                            train_idx.len(),
                            Some(&train_strata),
                            bag_len,
                        )?
                    } else {
                        // Bootstrap (with-replacement) draws are left unstratified: see
                        // fit_outer_bag's matching comment.
                        bootstrap_rows(spec.seed, bag_round, train_idx.len(), train_idx.len())?
                    };
                    let mut rows: Vec<u32> =
                        picked.into_iter().map(|i| train_idx[i as usize]).collect();
                    let n_train_rows = rows.len();
                    rows.extend((0..n_rows as u32).filter(|&r| mask[r as usize]));
                    let mut m = vec![false; rows.len()];
                    for slot in m.iter_mut().skip(n_train_rows) {
                        *slot = true;
                    }
                    (rows, Some(m))
                }
                None => {
                    let rows = if let Some(g) = spec.bag_groups {
                        group_subagging_rows(spec.seed, bag_round, g, Some(&strata), sample_len)?
                    } else if subsample {
                        subagging_rows_dispatch(
                            spec.seed,
                            bag_round,
                            n_rows,
                            Some(&strata),
                            sample_len,
                        )?
                    } else {
                        // Bootstrap (with-replacement) draws are left unstratified: see
                        // fit_outer_bag's matching comment.
                        bootstrap_rows(spec.seed, bag_round, n_rows, n_rows)?
                    };
                    (rows, None)
                }
            };
            if bag_mask.is_none() && !subsample {
                bag_mask = bootstrap_holdout_mask(
                    &rows,
                    config.validation_fraction,
                    spec.seed,
                    bag_round,
                    Some(strata.as_slice()),
                )?;
            }
            ensure_bag_training_mass(&mut rows, &mut bag_mask, spec)?;
            let data = row_subset(x, y, spec.weight, None, &rows)?;
            let bag_seed = pb_seed(spec.seed, bag_round, Stage::Sample as u32, 0);
            let bag_spec = FitSpec {
                loss: spec.loss,
                weight: data.weight.as_deref(),
                exposure: None,
                monotone: spec.monotone.clone(),
                interaction: spec.interaction.clone(),
                credibility: spec.credibility,
                fixed_holdout: bag_mask.as_deref(),
                bag_groups: None,
                seed: bag_seed,
            };
            let bag_model = fit_multiclass_single(
                &base_config,
                &data.x,
                &data.y,
                n_classes,
                class_labels,
                &bag_spec,
                cat_encoders,
            )?;
            // Membership over the FIT rows, recorded for every bagged K>=3 fit — the exact
            // mirror of `fit_outer_bag`'s block (see its comment): `Model::bag_in_bag` is what
            // lets the post-fit multiclass prune guard use each bag's OUT-OF-BAG complement as
            // honest evidence without carving anything out of the fit. The K classes of a bag
            // share ONE row draw (the bag subsets rows once, then fits all K columns jointly),
            // so the same membership vector is published on every per-class soup below.
            let mut in_bag = vec![false; n_rows];
            for &r in &rows {
                if let Some(slot) = in_bag.get_mut(r as usize) {
                    *slot = true;
                }
            }
            // §G1 evidence: this bag's K raw class columns at the rows it never drew. Scored
            // only over `!in_bag` (the other ~80% would be computed and discarded), exactly as
            // `fit_outer_bag`'s `collect_oob` block does for the scalar path. Bag models carry
            // no correction of their own, so `score_trees_rows` here is pure tree score.
            let oob_cols = if collect_oob {
                let oob_rows: Vec<u32> = (0..n_rows as u32)
                    .filter(|&r| !in_bag[r as usize])
                    .collect();
                let mut cols: Vec<Vec<f32>> = Vec::with_capacity(n_classes);
                for k in 0..n_classes {
                    let missing = || PbError::Internal {
                        what: "multiclass bag member missing a class model for OOB scoring".into(),
                    };
                    let class_model = bag_model.classes.get(k).ok_or_else(missing)?;
                    let mut preds = vec![0.0_f32; n_rows];
                    class_model.score_trees_rows(x, &oob_rows, &mut preds)?;
                    cols.push(preds);
                }
                Some(cols)
            } else {
                None
            };
            Ok((bag_model, in_bag, oob_cols))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let bag_in_bag: Vec<Vec<bool>> = bag_members.iter().map(|(_, m, _)| m.clone()).collect();
    let bag_models: Vec<&MultiClassModel> = bag_members.iter().map(|(m, _, _)| m).collect();
    // Per-class soup in class order; members fold in bag order inside soup_models.
    let mut classes = Vec::with_capacity(n_classes);
    for k in 0..n_classes {
        let members: Vec<WeightedModel> = bag_models
            .iter()
            .map(|m| -> Result<WeightedModel, PbError> {
                Ok(WeightedModel {
                    alpha,
                    model: m
                        .classes
                        .get(k)
                        .ok_or_else(|| PbError::Internal {
                            what: "multiclass bag member missing a class model".into(),
                        })?
                        .clone(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut souped = soup_models(&members)?;
        // Bags were collected IN BAG ORDER above, matching `soup_models`' `bag_spans` — and the
        // draw is shared across classes, so every class publishes the SAME membership. Both
        // fields are `#[serde(skip)]` runtime metadata excluded from `Model`'s `PartialEq`, so
        // this cannot move a fitted artifact, only what the guard can ask of it.
        souped.bag_in_bag = Some(bag_in_bag.clone());
        classes.push(souped);
    }
    let mut cell_refit_report: Option<MultiClassCellRefitReport> = None;
    if let Some(cr) = cell_refit {
        // Bag order is fixed (the parallel map collects in index order) and the accumulation
        // below is sequential, so the jury means — and everything downstream of them — are
        // byte-identical across thread counts.
        let oob_cols: Vec<&Vec<Vec<f32>>> = bag_members
            .iter()
            .map(|(_, _, o)| {
                o.as_ref().ok_or_else(|| PbError::Internal {
                    what: "multiclass cell refit asked for out-of-bag columns that were not \
                           collected"
                        .into(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (corrected, report) = attach_multiclass_cell_correction(
            classes,
            x,
            &strata,
            spec.weight,
            spec.bag_groups,
            n_classes,
            &bag_in_bag,
            &oob_cols,
            cr,
        )?;
        classes = corrected;
        cell_refit_report = Some(report);
    }
    let mut out = MultiClassModel {
        classes,
        class_labels: class_labels.to_vec(),
        schema_version: SCHEMA_VERSION_UNLIFTED,
        cell_refit: cell_refit_report,
    };
    out.schema_version = out.required_schema_version();
    out.validate()?;
    Ok(out)
}

/// The K-class §G1 cell-basis correction: K decoupled diagonal-Hessian cell solves on the
/// out-of-bag softmax working residual, accepted or shrunk by ONE JOINT BACKTRACK on the true
/// multinomial loss. The multiclass analogue of [`attach_cell_correction`].
///
/// # Why decoupled solves plus a joint backtrack, and not a joint K×K solve
///
/// The per-row softmax Hessian block is `H_i = diag(p_i) − p_i p_iᵀ` (times `w_i`): PSD, rank
/// `K−1`, null space `span(1)` — the gauge freedom that makes a common shift across classes a
/// softmax no-op. Its diagonal is `H_i[k,k] = p_ik(1−p_ik)`, and
/// `Σ_{j≠k} |H_i[k,j]| = Σ_{j≠k} p_ij p_ik = p_ik(1−p_ik) = H_i[k,k]`, so `H_i` is EXACTLY
/// weakly diagonally dominant. Two consequences, and they are the whole design:
///
///  1. Dropping the off-diagonal (the Jacobi/decoupled split) keeps the DIRECTION nearly right —
///     the neglected mass is spread across `K−1` off-diagonals, none dominant.
///  2. It gets the MAGNITUDE wrong in one direction only: the decoupled step systematically
///     OVERSHOOTS (the classic `(K−1)/K` factor in Friedman's multiclass GBM), worst at small K.
///
/// Measured on the geometry this function actually solves (a cell-indicator design, 6k rows,
/// two supports × 12 cells, 5 seeds; probe script in git history before e7d02a0): cosine similarity between
/// the decoupled direction and the FULL joint Newton direction is 0.97–0.997 at K ∈ {3,…,12},
/// while `‖d_dec‖/‖d_joint‖` runs 1.04–1.57 (largest at K=3, and growing as the ridge weakens).
/// With the backtrack, the decoupled step reaches 99–104% of the joint step's held-out loss
/// reduction. WITHOUT it, at a weak ridge, λ=1 is measurably worse than the backtracked optimum
/// (K=3, ridge 0.2: 0.0700 vs 0.0752 loss reduction) — i.e. the backtrack is exactly the piece
/// whose absence made the K≥3 per-round Jacobi leaf refine harmful. Cost: `K` solves of the
/// scalar shape, vs a `K·n_cells`-unknown system whose matvec is `K²` per row.
///
/// So: solve per class, then choose ONE global `λ` minimising the FULL softmax deviance of
/// `oob_raw + λ·δ` on a held-out slice, with all K corrections applied together. `λ = 0` drops
/// the whole correction (all K classes), which is the honest "no multiclass signal" answer.
///
/// Determinism: the jury accumulation, the per-class solves (class-index order) and the λ search
/// are all sequential / fixed-order; `fit_cell_correction`'s own parallelism has disjoint outputs.
#[allow(clippy::too_many_arguments)]
fn attach_multiclass_cell_correction(
    mut classes: Vec<Model>,
    x: &BinnedMatrix,
    labels: &[u32],
    weight: Option<&[f32]>,
    bag_groups: Option<&[u32]>,
    n_classes: usize,
    bag_in_bag: &[Vec<bool>],
    oob_cols: &[&Vec<Vec<f32>>],
    cr: CellRefit,
) -> Result<(Vec<Model>, MultiClassCellRefitReport), PbError> {
    let n_rows = x.n_rows as usize;
    if classes.len() != n_classes {
        return Err(PbError::Internal {
            what: format!(
                "multiclass cell refit got {} class models for {n_classes} classes",
                classes.len()
            ),
        });
    }
    // Out-of-bag raw per class: mean over the bags that did NOT train on the row. The K classes
    // of a bag share ONE row draw, so every class has the same jury on every row — the softmax
    // over the K jury means is a coherent probability vector (the same property the K>=3 prune
    // guard relies on).
    let mut oob_cnt = vec![0_u32; n_rows];
    let mut oob_sum: Vec<Vec<f64>> = vec![vec![0.0_f64; n_rows]; n_classes];
    for (bag, in_bag) in bag_in_bag.iter().enumerate() {
        let cols = oob_cols.get(bag).ok_or_else(|| PbError::Internal {
            what: "multiclass cell refit: bag has no out-of-bag columns".into(),
        })?;
        for r in 0..n_rows {
            if in_bag.get(r).copied().unwrap_or(true) {
                continue;
            }
            oob_cnt[r] += 1;
            for k in 0..n_classes {
                let col = cols.get(k).ok_or_else(|| PbError::Internal {
                    what: "multiclass cell refit: bag out-of-bag column class escaped".into(),
                })?;
                oob_sum[k][r] += f64::from(*col.get(r).ok_or_else(|| PbError::Internal {
                    what: "multiclass cell refit: out-of-bag column shorter than n_rows".into(),
                })?);
            }
        }
    }
    // Uncovered rows fall back to the per-class intercept — any in-domain raw works, since they
    // carry weight 0 in the solve and are excluded from the guard slice (mirrors the scalar path).
    // There is no exposure term: K>=3 rejects exposure (softmax has no offset).
    let mut oob_raw: Vec<Vec<f32>> = Vec::with_capacity(n_classes);
    for (k, model) in classes.iter().enumerate() {
        let mut col = vec![0.0_f32; n_rows];
        for r in 0..n_rows {
            col[r] = if oob_cnt[r] > 0 {
                (oob_sum[k][r] / f64::from(oob_cnt[r])) as f32
            } else {
                model.f0
            };
        }
        oob_raw.push(col);
    }
    let sample_w: Vec<f32> = match weight {
        Some(w) => w.to_vec(),
        None => vec![1.0_f32; n_rows],
    };
    // Softmax at the jury means: the K-column probability the corrections are linearised around.
    let mut probs: Vec<f32> = Vec::new();
    multiclass_softmax(&oob_raw, n_classes, n_rows, &mut probs)?;
    // The SAME deterministic ~15% slice the scalar path holds out, so a K=2-vs-K>=3 comparison
    // of the guard is like for like.
    let is_holdout = |r: usize| {
        let identity = bag_groups
            .and_then(|groups| groups.get(r))
            .copied()
            .map_or(r as u64, u64::from);
        (identity.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) < 38
    };

    // (1) K decoupled diagonal-Hessian solves. Class k's working residual is the softmax
    // Newton residual z = -g/h with g = w(p_k - 1[y=k]) and h = w·p_k(1-p_k) — the K-class
    // instance of the scalar path's `z = -g/h`, IRLS weight `h`, so `fit_cell_correction` is
    // reused verbatim (it is loss-agnostic: it only ever sees (residual, weight)).
    let mut banks: Vec<Option<CorrectionBank>> = vec![None; n_classes];
    let refit_spec = crate::cell_refit::CellRefitSpec {
        base: cr.base,
        gamma: cr.gamma,
        ..Default::default()
    };
    let prof = std::env::var_os("TBOOST_PROFILE").is_some();
    let t_solve = std::time::Instant::now();
    let mut n_supports = 0usize;
    let mut n_blocked = 0usize;
    // The K class solves are independent (each reads only the shared jury softmax and writes its
    // own bank), and each solve's backfit sweeps are sequential, so the classes fan out over the
    // pool — `MC_REFIT_CLASS_WAVE` at a time, which bounds how many `supports x fit_rows` designs
    // are live at once. Every class computes exactly what the sequential loop computed.
    let class_ids: Vec<usize> = (0..n_classes).collect();
    for wave in class_ids.chunks(MC_REFIT_CLASS_WAVE) {
        type ClassSolve = (usize, usize, usize, Option<CorrectionBank>);
        let solved: Vec<ClassSolve> = wave
            .par_iter()
            .map(|&k| -> Result<ClassSolve, PbError> {
                let model = classes.get(k).ok_or_else(|| PbError::Internal {
                    what: "multiclass cell refit class escaped the soup".into(),
                })?;
                let mut gh = crate::loss::GradHess {
                    g: vec![0.0_f32; n_rows],
                    h: vec![0.0_f32; n_rows],
                };
                fill_grad_hess_from_probs(&probs, labels, &sample_w, k, n_classes, &mut gh)?;
                let mut residual = vec![0.0_f64; n_rows];
                let mut cell_w = vec![0.0_f64; n_rows];
                for r in 0..n_rows {
                    if oob_cnt[r] > 0 && !is_holdout(r) {
                        let h = f64::from(gh.h[r]).max(1e-12);
                        residual[r] = -f64::from(gh.g[r]) / h;
                        cell_w[r] = h;
                    }
                }
                let supports = correctable_supports(model);
                // Counted unconditionally (they used to be `if prof`): they are the
                // reachable-coverage half of `MultiClassCellRefitReport`, which the fit report
                // now carries on every fit. Both are structural scans of the already-built bank.
                let blocked = realized_supports_blocked_by_channels(model);
                let bank = if supports.is_empty() {
                    None
                } else {
                    Some(crate::cell_refit::fit_cell_correction(
                        model,
                        &x.data,
                        &residual,
                        &cell_w,
                        &supports,
                        &refit_spec,
                    )?)
                };
                Ok((k, supports.len(), blocked, bank))
            })
            .collect::<Result<_, _>>()?;
        for (k, n_sup, blocked, bank) in solved {
            n_supports += n_sup;
            n_blocked += blocked;
            if let Some(slot) = banks.get_mut(k) {
                *slot = bank;
            }
        }
    }
    let solve_s = t_solve.elapsed().as_secs_f64();
    if banks.iter().all(Option::is_none) {
        // Nothing reachable to correct: lambda 0 (declined), no jury scored.
        return Ok((
            classes,
            MultiClassCellRefitReport {
                lambda: 0.0,
                n_supports,
                n_blocked,
                n_rejected: 0,
                guard_rows: 0,
            },
        ));
    }
    for (model, bank) in classes.iter_mut().zip(banks) {
        model.correction = bank;
    }

    // (2) THE JOINT BACKTRACK. One global λ over ALL K corrections, scored by the true
    // multinomial deviance of the held-out out-of-bag slice — never K independent per-class
    // guards, which is what would let one class's over-large step ride in on another's gain.
    // The grid is step-halving from 1 (the Newton step) down to 0 (decline), widened below 0.25
    // relative to the scalar path's 5-point grid because the decoupled direction's known failure
    // mode is OVERSHOOT: the optimum sits below 1 exactly when the ridge is weak.
    let t_guard = std::time::Instant::now();
    let mut hrows: Vec<usize> = Vec::new();
    for r in 0..n_rows {
        if oob_cnt[r] > 0 && is_holdout(r) {
            hrows.push(r);
        }
    }
    let mut lambda = 1.0_f32;
    let mut n_rejected = 0_usize;
    if !hrows.is_empty() {
        let mut row_bins = vec![0u8; x.data.len()];
        let mut hd: Vec<Vec<f32>> = vec![vec![0.0_f32; hrows.len()]; n_classes];
        for (i, &r) in hrows.iter().enumerate() {
            for (a, col) in x.data.iter().enumerate() {
                if let Some(&bin) = col.get(r) {
                    row_bins[a] = bin;
                }
            }
            for (k, model) in classes.iter().enumerate() {
                hd[k][i] = model.correction_delta(&row_bins)? as f32;
            }
        }
        let hlabels: Vec<u32> = hrows
            .iter()
            .map(|&r| {
                labels.get(r).copied().ok_or_else(|| PbError::Internal {
                    what: "multiclass cell refit guard row escaped labels".into(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let hw: Vec<f32> = hrows.iter().map(|&r| sample_w[r]).collect();
        let hraw: Vec<Vec<f32>> = (0..n_classes)
            .map(|k| hrows.iter().map(|&r| oob_raw[k][r]).collect())
            .collect();
        let rowset: Vec<usize> = (0..hrows.len()).collect();
        let mut cand: Vec<Vec<f32>> = vec![vec![0.0_f32; hrows.len()]; n_classes];
        let mut best_dev = f64::INFINITY;
        let mut best = 0.0_f32;
        const LAMBDAS: [f32; 8] = [0.0, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0];
        for &lam in &LAMBDAS {
            for k in 0..n_classes {
                for i in 0..hrows.len() {
                    cand[k][i] = hraw[k][i] + lam * hd[k][i];
                }
            }
            let dev = multiclass_deviance_for_rows(&cand, &hlabels, &hw, &rowset)?;
            if dev < best_dev {
                best_dev = dev;
                best = lam;
            }
        }
        // "Rejected" = every trial step the joint loss refused, i.e. the halvings walked past.
        n_rejected = LAMBDAS.iter().filter(|&&l| l > best).count();
        lambda = best;
    }
    if lambda <= 0.0 {
        for model in &mut classes {
            model.correction = None;
        }
    } else if lambda < 1.0 {
        for model in &mut classes {
            if let Some(bank) = &mut model.correction {
                for table in &mut bank.tables {
                    for v in &mut table.values {
                        *v *= f64::from(lambda);
                    }
                }
            }
        }
    }
    if prof {
        // `blocked` counts the realized order-<=2 supports that touch a P1 multi-channel raw
        // feature and therefore cannot be a `CorrectionTable` (see `correctable_supports`) —
        // the honest denominator for reading a lambda=0 verdict on a categorical-heavy set.
        eprintln!(
            "[mc cell_refit] {n_classes} class solves {solve_s:.2}s | \
             supports {n_supports} corrected, {n_blocked} blocked by P1 channels | \
             joint backtrack (lambda={lambda}, {n_rejected} steps rejected, {} guard rows) {:.2}s",
            hrows.len(),
            t_guard.elapsed().as_secs_f64(),
        );
    }
    for model in &mut classes {
        model.validate()?;
    }
    let report = MultiClassCellRefitReport {
        lambda: f64::from(lambda),
        n_supports,
        n_blocked,
        n_rejected,
        guard_rows: hrows.len(),
    };
    Ok((classes, report))
}

/// Float `y` (Softmax's label encoding: each value a non-negative integer `< n_classes`) as `u32`
/// class labels. Shared by [`fit_multiclass_single`] (needs them for the round's softmax
/// grad/hess) and [`fit_multiclass_bagged`] (needs them again, pre-bag, as the stratification
/// key for [`stratified_subagging_rows`] — an unweighted bag draw can otherwise miss a rare
/// class's rows entirely, the same failure `es_strata_for_loss` fixes for the ES holdout carve).
fn multiclass_labels(y: &[f32], n_classes: usize) -> Result<Vec<u32>, PbError> {
    let mut labels: Vec<u32> = Vec::with_capacity(y.len());
    for (i, &yi) in y.iter().enumerate() {
        if !yi.is_finite() || yi < 0.0 || yi.fract() != 0.0 {
            return Err(invalid_input(format!(
                "multiclass label[{i}] must be a non-negative integer, got {yi}"
            )));
        }
        let li = yi as u32;
        if li as usize >= n_classes {
            return Err(invalid_input(format!(
                "multiclass label[{i}] = {li} >= n_classes {n_classes}"
            )));
        }
        labels.push(li);
    }
    Ok(labels)
}

/// The single (unbagged) native-softmax fit — see [`fit_multiclass`] for the dispatch and the
/// honored-lever inventory. Deterministic and byte-identical across thread counts.
fn fit_multiclass_single(
    config: &Config,
    x: &BinnedMatrix,
    y: &[f32],
    n_classes: usize,
    class_labels: &[String],
    spec: &FitSpec,
    cat_encoders: &CatEncoderStore,
) -> Result<MultiClassModel, PbError> {
    config.validate()?;
    validate_fit_spec(config, spec)?;
    validate_binned_matrix(x)?;
    prof::reset(); // dev-only stage profile, mirroring fit_single (TBOOST_PROFILE-gated no-op)
    let n = x.n_rows as usize;
    if y.len() != n {
        return Err(PbError::ShapeMismatch {
            what: format!("y len {} != n_rows {n}", y.len()),
        });
    }
    if n_classes < 2 {
        return Err(invalid_input(format!(
            "multiclass requires >= 2 classes, got {n_classes}"
        )));
    }
    if class_labels.len() != n_classes {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "class_labels len {} != n_classes {n_classes}",
                class_labels.len()
            ),
        });
    }

    // Integer labels in `0..n_classes`.
    let labels = multiclass_labels(y, n_classes)?;

    let ones_storage: Vec<f32>;
    let weight: &[f32] = match spec.weight {
        Some(w) => {
            if w.len() != n {
                return Err(PbError::ShapeMismatch {
                    what: format!("weight len {} != n_rows {n}", w.len()),
                });
            }
            w
        }
        None => {
            ones_storage = vec![1.0_f32; n];
            &ones_storage
        }
    };

    let n_features = x.data.len();
    let monotone = resolve_monotone(spec, n_features)?;
    let monotone_ref = if monotone.iter().any(Option::is_some) {
        Some(monotone.as_slice())
    } else {
        None
    };
    let axes: Vec<u32> = (0..u32::try_from(n_features).map_err(|_| PbError::Internal {
        what: "more than u32::MAX features".into(),
    })?)
        .collect();
    let (mut train_rows, mut validation_rows) = match spec.fixed_holdout {
        Some(mask) => split_rows_by_mask(x.n_rows, mask)?,
        // ES-stratify: rank the internal early-stopping holdout WITHIN each class (the
        // class labels ARE the strata), mirroring fit_single's Logistic rule — a rare
        // class is never starved of holdout representation.
        None => carve_validation_rows_stratified(
            x.n_rows,
            config.validation_fraction,
            spec.seed,
            Some(&labels),
        )?,
    };
    ensure_validation_mass(
        &mut train_rows,
        &mut validation_rows,
        weight,
        spec.fixed_holdout.is_some(),
    )?;
    let max_delta_step = config.max_delta_step.map(f64::from);

    // Per-class init `f0_k = ln(prior_k)`; K raw columns seeded to `f0_k`.
    // Intercept honesty (mirrors fit_single): with a validation carve the class priors are
    // computed over TRAIN rows only, so the holdout's class mix never leaks into the
    // early-stopping baseline through f0. Without a carve this is byte-identical.
    let f0 = if train_rows.len() != n {
        let (train_labels, train_weight): (Vec<u32>, Vec<f32>) = train_rows
            .iter()
            .map(|&r| -> Result<(u32, f32), PbError> {
                let l = labels
                    .get(r as usize)
                    .copied()
                    .ok_or_else(|| PbError::Internal {
                        what: "train row escaped multiclass labels".into(),
                    })?;
                let w = weight
                    .get(r as usize)
                    .copied()
                    .ok_or_else(|| PbError::Internal {
                        what: "train row escaped multiclass weights".into(),
                    })?;
                Ok((l, w))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        multiclass_init(&train_labels, &train_weight, n_classes)?
    } else {
        multiclass_init(&labels, weight, n_classes)?
    };
    let mut raw: Vec<Vec<f32>> = f0.iter().map(|&f| vec![f as f32; n]).collect();
    let mut trees: Vec<Vec<(f32, ObliviousTree)>> = (0..n_classes).map(|_| Vec::new()).collect();

    let grow_cfg = GrowConfig {
        lambda: f64::from(config.lambda),
        l1_leaf: f64::from(config.l1_leaf),
        lr: f64::from(config.learning_rate),
        min_split_gain: f64::from(config.min_split_gain),
        interaction_gain_hurdle: f64::from(config.interaction_gain_hurdle),
        max_order: spec.interaction.max_order,
        max_depth: spec.interaction.max_depth,
        max_delta_step,
        hist_precision: config.hist_precision,
        quant_seed: spec.seed,
        round: 0,
        random_strength: f64::from(config.boosters.random_strength),
        groups: spec.interaction.groups.as_deref(),
        monotone: monotone_ref,
        table_budget_penalty: TableBudgetPenalty::new(
            f64::from(spec.interaction.table_budget_beta),
            spec.interaction.table_budget_cells,
            f64::from(spec.interaction.table_budget_order_shrink),
        ),
        // Not yet threaded onto a live round loop here (multiclass WIP): `None` keeps
        // `TableBudgetPenalty::multiplier`'s prior `grid.n_bins` behavior exactly.
        realized_extent: None,
        credibility: spec.credibility,
        // Native softmax has no Poisson/Tweedie-style claim-mass concept — always Count
        // (`credibility_evidence_for_loss`'s `LossId::Softmax` arm agrees; hardcoded here
        // since this path never reads `spec.loss`, it fits a fixed per-class softmax).
        credibility_evidence: CredibilityEvidence::Count,
        // An explicit all-ones weight is the unit weight: the histogram then skips its weighted
        // sum entirely (`wsum` is the count, exactly what a sum of 1.0s is), byte-identical.
        unit_weight: spec.weight.is_none(),
        hist_subtraction: true,
    };

    // One reusable buffer per class lets classes grow concurrently without allocating 2*n*K
    // gradient cells every round.
    let mut grad_hess_by_class: Vec<GradHess> =
        (0..n_classes).map(|_| GradHess::default()).collect();
    // Reused across rounds: the round's `n*K` softmax probabilities, resized in place so there is
    // no per-round allocation/zero-fill (every cell is overwritten before it is read).
    let mut probs: Vec<f32> = Vec::new();
    let mut deviance_scratch: Vec<f64> = Vec::new();
    let mut best_deviance = match validation_rows.as_deref() {
        Some(val) => Some(multiclass_deviance_for_rows_with_scratch(
            &raw,
            &labels,
            weight,
            val,
            &mut deviance_scratch,
        )?),
        None => None,
    };
    let mut best_counts: Vec<usize> = vec![0; n_classes];
    let mut best_round_seen = 0u32;
    let mut interaction_gain_references: Vec<Option<f64>> = vec![None; n_classes];

    for t in 0..config.n_trees {
        let mut round_cfg = grow_cfg.clone();
        round_cfg.round = t;
        round_cfg.lr = learning_rate_for_round(config.learning_rate, config.learning_rate_decay, t);
        let round_axes = sample_axes(&axes, config.colsample_bytree, spec.seed, t)?;
        // Softmax probabilities for ALL classes, from the round-start raw columns. Every class's
        // gradient this round is w.r.t. this same state (standard multinomial GBM), so all class
        // trees can be grown independently on the same bounded Rayon pool.
        prof::timed("mc_softmax", || {
            multiclass_softmax(&raw, n_classes, n, &mut probs)
        })?;
        // Gradients FIRST, for every class, then the row sample, then growth. The single-output
        // path can do all three in one statement because it has one (g, h) column; the K-class
        // path needs all K filled before it can score a row, because MVS's sampling weight is a
        // property of the ROW ACROSS CLASSES, not of any one class (see `mc_sample_rows`).
        prof::timed("mc_grad", || {
            grad_hess_by_class
                .par_iter_mut()
                .enumerate()
                .try_for_each(|(k, gh)| {
                    fill_grad_hess_from_probs(&probs, &labels, weight, k, n_classes, gh)
                })
        })?;
        // §06.5 MVS, K-class form. ONE row draw per round, shared by all K trees: the classes
        // are coupled through the softmax and grow against the same round-start state, so
        // sampling them independently would give the K columns of one round disjoint views of
        // the data and break that shared-state contract. The per-row weight is the JOINT
        // gradient/hessian norm over classes (`mc_sample_rows`), which is the natural K-class
        // reading of the scalar `sqrt(g^2 + h^2)`.
        let (sampled_rows, mvs_reweight) = mc_sample_rows(
            &config.sampling,
            &grad_hess_by_class,
            spec.seed,
            t,
            &train_rows,
        )?;
        let full_sample = mvs_reweight.is_none();
        // The sampled (g, h) are a gradient-biased sample of the population, so SPLIT SELECTION
        // must see them reweighted by 1/p_i. The originals stay exact — they are what the
        // full-data leaf refit below re-solves against.
        let scaled_by_class: Option<Vec<GradHess>> = match &mvs_reweight {
            Some(mult) => Some(
                grad_hess_by_class
                    .iter()
                    .map(|gh| reweighted_gh(gh, &sampled_rows, mult))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            None => None,
        };
        let grown_by_class: Vec<Option<GrowResult>> = prof::timed("mc_grow", || {
            grad_hess_by_class
                .par_iter()
                .enumerate()
                .map(|(k, gh)| -> Result<Option<GrowResult>, PbError> {
                    let grow_gh = match &scaled_by_class {
                        Some(scaled) => scaled.get(k).ok_or_else(|| PbError::Internal {
                            what: "multiclass MVS reweighted class escaped".into(),
                        })?,
                        None => gh,
                    };
                    let reference =
                        *interaction_gain_references
                            .get(k)
                            .ok_or_else(|| PbError::Internal {
                                what: "multiclass hurdle reference class escaped".into(),
                            })?;
                    let mut grown = grow_oblivious_tree_with_leaf_map(
                        x,
                        grow_gh,
                        &sampled_rows,
                        &round_axes,
                        &round_cfg,
                        InteractionHurdleState::new(config.interaction_gain_hurdle_mode, reference),
                        weight,
                    )?;
                    // MVS samples the STRUCTURE only: the leaf values are re-solved on ALL train
                    // rows against the exact (unreweighted) gradients, exactly as the single-output
                    // path does. `leaf_of_row` is then stale (it partitions `sampled_rows`), which
                    // is why the raw update below re-walks instead of reusing it.
                    if !full_sample {
                        if let Some(g) = grown.as_mut() {
                            refit_tree_leaves(x, gh, &train_rows, &mut g.tree, &round_cfg)?;
                        }
                    }
                    Ok(grown)
                })
                .collect::<Result<Vec<_>, _>>()
        })?;

        // Each result mutates only its own class column/bank/reference. Rayon zips indexed
        // iterators, so class k always receives result k regardless of scheduling.
        let grew_by_class: Vec<bool> = prof::timed("mc_update_raw", || {
            raw.par_iter_mut()
                .zip(trees.par_iter_mut())
                .zip(interaction_gain_references.par_iter_mut())
                .zip(grown_by_class.into_par_iter())
                .map(
                    |(((raw_k, trees_k), gain_reference), grown)| -> Result<bool, PbError> {
                        let Some(grown) = grown else {
                            return Ok(false);
                        };
                        let first_split_gain = grown.first_split_gain;
                        let tree = grown.tree;
                        let leaf_of_row = grown.leaf_of_row;
                        // grow saw exactly `train_rows`, so reuse its leaf map for those (a plain
                        // index-add) and re-walk ONLY the validation holdout — byte-identical to a full
                        // re-walk, mirroring fit_single's update_raw_split fast path. With no validation,
                        // `train_rows` is every row, so the map covers all of `raw_k`.
                        match (full_sample, validation_rows.as_deref()) {
                            (true, Some(val)) => update_raw_split(
                                raw_k,
                                None,
                                x,
                                &tree,
                                &leaf_of_row,
                                &train_rows,
                                val,
                            )?,
                            (true, None) => {
                                update_raw(raw_k, None, x, &tree, Some(leaf_of_row.as_slice()))?
                            }
                            // Under MVS `leaf_of_row` partitions the SAMPLED rows, not `train_rows`,
                            // and the leaves were re-solved after it was built — a full re-walk is
                            // the only correct update.
                            (false, _) => update_raw(raw_k, None, x, &tree, None)?,
                        }
                        trees_k.push((1.0, tree));
                        update_interaction_gain_reference(gain_reference, first_split_gain);
                        Ok(true)
                    },
                )
                .collect::<Result<Vec<_>, _>>()
        })?;
        let any_grew = grew_by_class.into_iter().any(|grew| grew);
        // No class could split (converged / constant) — stop with what we have.
        if !any_grew {
            break;
        }
        if let Some(val) = validation_rows.as_deref() {
            let dev = prof::timed("mc_earlystop_eval", || {
                multiclass_deviance_for_rows_with_scratch(
                    &raw,
                    &labels,
                    weight,
                    val,
                    &mut deviance_scratch,
                )
            })?;
            let improved = match best_deviance {
                Some(b) => config.is_material_improvement(b, dev),
                None => true,
            };
            if improved {
                best_deviance = Some(dev);
                for (k, tk) in trees.iter().enumerate() {
                    *best_counts.get_mut(k).ok_or_else(|| PbError::Internal {
                        what: "multiclass best_counts class escaped".into(),
                    })? = tk.len();
                }
                best_round_seen = t;
            } else if (t as usize).saturating_sub(best_round_seen as usize)
                >= config.effective_patience(best_round_seen as usize)
            {
                break;
            }
        }
    }

    if prof::enabled() {
        let trained = trees.first().map_or(0, Vec::len);
        eprintln!(
            "[mc-es] trained_rounds={trained} best_round={} kept_rounds={}",
            best_round_seen + 1,
            best_counts.first().copied().unwrap_or(trained)
        );
    }
    // Truncate each class to the best round when early stopping is active.
    if validation_rows.is_some() {
        for (k, tk) in trees.iter_mut().enumerate() {
            let keep = *best_counts.get(k).ok_or_else(|| PbError::Internal {
                what: "multiclass best_counts read escaped".into(),
            })?;
            tk.truncate(keep);
        }
    }

    // Assemble one scalar Model per class (raw score = class logit; Identity link).
    let feature_names: Vec<String> = (0..n_features).map(|i| format!("f{i}")).collect();
    let feature_kinds: Vec<crate::data::AxisKind> = x.provenance.iter().map(|p| p.kind).collect();
    let mut classes = Vec::with_capacity(n_classes);
    for (k, tk) in trees.into_iter().enumerate() {
        let f0_k = *f0.get(k).ok_or_else(|| PbError::Internal {
            what: "multiclass f0 class escaped".into(),
        })? as f32;
        let schema = ModelSchema {
            feature_names: feature_names.clone(),
            feature_kinds: feature_kinds.clone(),
            cat_encoders: cat_encoders.clone(),
            class_labels: None,
            objective: ObjectiveTag {
                link: Link::Identity,
                loss: LossId::Softmax,
                tweedie_rho: None,
            },
        };
        let mut model = Model {
            f0: f0_k,
            trees: tk,
            grids: x.grids.clone(),
            provenance: x.provenance.clone(),
            link: Link::Identity,
            mode: ExactnessMode::Exact,
            schema,
            schema_version: SCHEMA_VERSION_UNLIFTED,
            correction: None,
            bag_spans: None,
            bag_intercepts: None,
            bag_in_bag: None,
            delta_step_gate: None,
            fit_report: None,
        };
        model.schema_version = model.required_schema_version();
        model.validate()?;
        classes.push(model);
    }
    let mut out = MultiClassModel {
        classes,
        class_labels: class_labels.to_vec(),
        schema_version: SCHEMA_VERSION_UNLIFTED,
        // The unbagged base path has no out-of-bag complement, so the §G1 refit never runs here.
        cell_refit: None,
    };
    out.schema_version = out.required_schema_version();
    out.validate()?;
    prof::report(); // one stage breakdown per multiclass fit, mirroring fit_single
    Ok(out)
}

/// Per-class softmax init `f0_k = ln(max(prior_k, eps))` with `prior_k = Σw·1[y=k] / Σw`.
/// Fixed-order f64 fold (thread-count independent).
fn multiclass_init(labels: &[u32], weight: &[f32], n_classes: usize) -> Result<Vec<f64>, PbError> {
    let mut sums = vec![0.0_f64; n_classes];
    let mut total = 0.0_f64;
    for (i, &l) in labels.iter().enumerate() {
        let w = f64::from(*weight.get(i).ok_or_else(|| PbError::Internal {
            what: "multiclass init weight row escaped".into(),
        })?);
        total += w;
        *sums.get_mut(l as usize).ok_or_else(|| PbError::Internal {
            what: "multiclass init label escaped".into(),
        })? += w;
    }
    if total <= 0.0 {
        return Err(invalid_input(
            "multiclass init: non-positive total weight".into(),
        ));
    }
    const EPS: f64 = 1e-12;
    Ok(sums
        .into_iter()
        .map(|s| (s / total).max(EPS).ln())
        .collect())
}

/// Fill `probs` with the row-major `n × K` softmax probabilities from the K raw columns (reuses
/// the stable [`softmax_in_place`]). `probs` is resized in place (a no-op at steady `n,K`), so the
/// caller reuses one buffer across rounds with no per-round allocation. Deterministic per-row.
fn multiclass_softmax(
    raw: &[Vec<f32>],
    n_classes: usize,
    n: usize,
    probs: &mut Vec<f32>,
) -> Result<(), PbError> {
    if raw.len() != n_classes {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "multiclass softmax has {} raw columns, expected {n_classes}",
                raw.len()
            ),
        });
    }
    if let Some((class, column)) = raw.iter().enumerate().find(|(_, column)| column.len() != n) {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "multiclass softmax raw class {class} has {} rows, expected {n}",
                column.len()
            ),
        });
    }
    let cells = n.checked_mul(n_classes).ok_or_else(|| PbError::Internal {
        what: "multiclass softmax size overflows usize".into(),
    })?;
    // Every cell is overwritten below before it is read, so the resize's fill value is never
    // observed; after round 0 this is a no-op (capacity retained).
    probs.resize(cells, 0.0);
    // Fixed row-block chunks (objective-parity 2026-07-16): the former one-task-PER-ROW split
    // (~K writes + one softmax per task) is the finest-grained parallel site in the crate — the
    // P1.1 small-task disease at its worst. Blocks of MC_ROW_CHUNK rows keep the identical
    // per-row ops on disjoint slices (byte-identical at any thread count / sched knob value).
    const MC_ROW_CHUNK: usize = 4_096;
    probs
        .par_chunks_mut(n_classes.saturating_mul(MC_ROW_CHUNK))
        .with_min_len(crate::sched::min_chunks_per_task())
        .enumerate()
        .for_each(|(ci, block)| {
            for (j, row) in block.chunks_exact_mut(n_classes).enumerate() {
                let i = ci * MC_ROW_CHUNK + j;
                for (dst, raw_class) in row.iter_mut().zip(raw.iter()) {
                    *dst = raw_class[i];
                }
                softmax_in_place(row);
            }
        });
    Ok(())
}

/// Fill `out` with class-`k` softmax `(g, h)` from the precomputed `n x K` probabilities:
/// `g = w(p_k - 1[y=k])`, `h = (w * p_k(1-p_k)).max(1e-16)`. Deterministic row map.
fn fill_grad_hess_from_probs(
    probs: &[f32],
    labels: &[u32],
    weight: &[f32],
    k: usize,
    n_classes: usize,
    out: &mut GradHess,
) -> Result<(), PbError> {
    let n = labels.len();
    if weight.len() != n {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "multiclass gradient weight len {} != label len {n}",
                weight.len()
            ),
        });
    }
    if k >= n_classes {
        return Err(PbError::Internal {
            what: format!("multiclass gradient class {k} >= n_classes {n_classes}"),
        });
    }
    let cells = n.checked_mul(n_classes).ok_or_else(|| PbError::Internal {
        what: "multiclass gradient size overflows usize".into(),
    })?;
    if probs.len() != cells {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "multiclass gradient probability len {} != n*K {cells}",
                probs.len()
            ),
        });
    }
    out.g.resize(n, 0.0);
    out.h.resize(n, 0.0);
    const FLOOR: f32 = 1e-16;
    // Fixed chunks (objective-parity 2026-07-16): was one task PER ELEMENT. Identical per-row
    // ops on disjoint slices — byte-identical at any thread count / sched knob value.
    const MC_GH_CHUNK: usize = 8_192;
    out.g
        .par_chunks_mut(MC_GH_CHUNK)
        .zip(out.h.par_chunks_mut(MC_GH_CHUNK))
        .with_min_len(crate::sched::min_chunks_per_task())
        .enumerate()
        .for_each(|(ci, (gc, hc))| {
            let base = ci * MC_GH_CHUNK;
            for (j, (g, h)) in gc.iter_mut().zip(hc.iter_mut()).enumerate() {
                let i = base + j;
                let pk = probs[i * n_classes + k];
                let w = weight[i];
                let target = if labels[i] as usize == k {
                    1.0_f32
                } else {
                    0.0_f32
                };
                *g = w * (pk - target);
                *h = (w * pk * (1.0 - pk)).max(FLOOR);
            }
        });
    Ok(())
}

/// Parallel per-row softmax loss followed by the original fixed-order f64 fold. The scratch buffer
/// is reused by the fit loop, avoiding a validation-sized allocation every boosting round.
fn multiclass_deviance_for_rows_with_scratch(
    raw: &[Vec<f32>],
    labels: &[u32],
    weight: &[f32],
    rows: &[usize],
    row_losses: &mut Vec<f64>,
) -> Result<f64, PbError> {
    let n_classes = raw.len();
    if n_classes == 0 {
        return Err(invalid_input(
            "multiclass deviance requires at least one raw class".into(),
        ));
    }
    if weight.len() != labels.len() {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "multiclass deviance weight len {} != label len {}",
                weight.len(),
                labels.len()
            ),
        });
    }
    if let Some((class, column)) = raw
        .iter()
        .enumerate()
        .find(|(_, column)| column.len() != labels.len())
    {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "multiclass deviance raw class {class} has {} rows, expected {}",
                column.len(),
                labels.len()
            ),
        });
    }

    row_losses.resize(rows.len(), 0.0);
    // Fixed chunks (objective-parity 2026-07-16): was one task PER ROW. The per-row loss is a
    // pure map (the f64 fold below is separate and unchanged) — byte-identical at any thread
    // count / sched knob value.
    const MC_LOSS_CHUNK: usize = 8_192;
    let per_row = |loss: &mut f64, r: usize| -> Result<(), PbError> {
        let mut m = f32::NEG_INFINITY;
        for raw_class in raw {
            let v = *raw_class.get(r).ok_or_else(|| PbError::Internal {
                what: "multiclass deviance raw row escaped".into(),
            })?;
            if v > m {
                m = v;
            }
        }
        if !m.is_finite() {
            m = 0.0;
        }
        let mut denom = 0.0_f64;
        for raw_class in raw {
            let v = *raw_class.get(r).ok_or_else(|| PbError::Internal {
                what: "multiclass deviance raw row escaped".into(),
            })?;
            denom += f64::from((v - m).clamp(-30.0, 30.0).exp());
        }
        let label = *labels.get(r).ok_or_else(|| PbError::Internal {
            what: "multiclass deviance label escaped".into(),
        })? as usize;
        let zl = *raw
            .get(label)
            .ok_or_else(|| PbError::Internal {
                what: "multiclass deviance label class escaped".into(),
            })?
            .get(r)
            .ok_or_else(|| PbError::Internal {
                what: "multiclass deviance label row escaped".into(),
            })?;
        let num = f64::from((zl - m).clamp(-30.0, 30.0).exp());
        let p = (num / denom.max(1e-30)).max(1e-30);
        let w = f64::from(*weight.get(r).ok_or_else(|| PbError::Internal {
            what: "multiclass deviance weight escaped".into(),
        })?);
        *loss = w * (-p.ln());
        Ok(())
    };
    row_losses
        .par_chunks_mut(MC_LOSS_CHUNK)
        .zip(rows.par_chunks(MC_LOSS_CHUNK))
        .with_min_len(crate::sched::min_chunks_per_task())
        .try_for_each(|(lc, rc)| -> Result<(), PbError> {
            for (loss, &r) in lc.iter_mut().zip(rc) {
                per_row(loss, r)?;
            }
            Ok(())
        })?;

    // Keep the accumulation order identical across every thread count (and to the former serial
    // implementation); only the independent per-row score calculation runs in parallel.
    let (mut acc, mut sw) = (0.0_f64, 0.0_f64);
    for (&r, &loss) in rows.iter().zip(row_losses.iter()) {
        acc += loss;
        sw += f64::from(weight[r]);
    }
    if sw <= 0.0 {
        return Err(invalid_input(
            "multiclass deviance: non-positive total weight over eval rows".into(),
        ));
    }
    Ok(acc / sw)
}

/// Weighted mean softmax cross-entropy `-sum(w*ln p_y) / sum(w)` over `rows` (the early-stopping
/// metric). Stable per-row softmax; f64 fold in `rows` order (thread-count independent).
pub(crate) fn multiclass_deviance_for_rows(
    raw: &[Vec<f32>],
    labels: &[u32],
    weight: &[f32],
    rows: &[usize],
) -> Result<f64, PbError> {
    let mut row_losses = Vec::new();
    multiclass_deviance_for_rows_with_scratch(raw, labels, weight, rows, &mut row_losses)
}

const ENSEMBLE_WEIGHT_TOL: f64 = 1.0e-6;

struct OwnedFitData {
    x: BinnedMatrix,
    y: Vec<f32>,
    weight: Option<Vec<f32>>,
    exposure: Option<Vec<f32>>,
}

struct WeightedModel {
    alpha: f64,
    model: Model,
}

struct LibraryMember {
    model: Model,
    holdout_raw: Vec<f32>,
    deviance: f64,
}

struct GreedyParams<'a> {
    library_size: u16,
    hp_grid: &'a HpGrid,
    selection_bags: u16,
    seed_top_n: u8,
}

#[derive(Clone, Copy)]
struct HpChoice {
    max_bin: u16,
    lambda: f32,
    learning_rate: f32,
    n_trees: u32,
    max_order: u8,
    random_strength: f32,
}

fn ensemble_base_config(config: &Config) -> Config {
    let mut base = config.clone();
    base.boosters.ensemble = EnsembleSpec::Off;
    base
}

fn validate_ensemble_fit_inputs(
    config: &Config,
    x: &BinnedMatrix,
    y: &[f32],
    spec: &FitSpec,
) -> Result<(), PbError> {
    config.validate()?;
    validate_fit_spec(config, spec)?;
    validate_binned_matrix(x)?;
    let n = x.n_rows as usize;
    if n == 0 {
        return Err(invalid_input("fit requires at least one row".into()));
    }
    if let Some(mask) = spec.fixed_holdout {
        if mask.len() != n {
            return Err(PbError::ShapeMismatch {
                what: format!("fixed_holdout len {} != n_rows {n}", mask.len()),
            });
        }
    }
    if let Some(g) = spec.bag_groups {
        if g.len() != n {
            return Err(PbError::ShapeMismatch {
                what: format!("bag_groups len {} != n_rows {n}", g.len()),
            });
        }
    }
    if y.len() != n {
        return Err(PbError::ShapeMismatch {
            what: format!("y len {} != n_rows {n}", y.len()),
        });
    }
    if let Some(weight) = spec.weight {
        if weight.len() != n {
            return Err(PbError::ShapeMismatch {
                what: format!("weight len {} != n_rows {n}", weight.len()),
            });
        }
        if weight
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
            || !weight.iter().any(|value| *value > 0.0)
        {
            return Err(invalid_input(
                "sample weights must be finite, nonnegative, and have positive total mass".into(),
            ));
        }
    }
    if let Some(exposure) = spec.exposure {
        if exposure.len() != n {
            return Err(PbError::ShapeMismatch {
                what: format!("exposure len {} != n_rows {n}", exposure.len()),
            });
        }
    }
    Ok(())
}

/// One bag's out-of-bag contribution for the §G1 cell-basis refit: which rows the bag
/// trained on (OOB = complement) and the bag's raw predictions on EVERY training row.
struct BagOob {
    in_bag: Vec<bool>,
    preds: Vec<f32>,
}

/// Carve source observations rather than bootstrap positions, so copies stay together.
fn bootstrap_holdout_mask(
    rows: &[u32],
    fraction: Option<f32>,
    seed: u64,
    bag: u32,
    strata: Option<&[u32]>,
) -> Result<Option<Vec<bool>>, PbError> {
    if fraction.is_none() {
        return Ok(None);
    }
    let sources: Vec<u32> = rows
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if sources.len() < 2 {
        return Ok(None);
    }
    let local_strata = strata
        .map(|values| gather_strata(values, &sources))
        .transpose()?;
    let Some(mask) = holdout_mask(
        sources.len() as u32,
        fraction,
        pb_seed(seed, bag, Stage::Sample as u32, 0),
        local_strata.as_deref(),
    )?
    else {
        return Ok(None);
    };
    Ok(Some(
        rows.iter()
            .map(|source| {
                sources
                    .binary_search(source)
                    .ok()
                    .and_then(|index| mask.get(index))
                    .copied()
                    .unwrap_or(false)
            })
            .collect(),
    ))
}

/// Deterministically retain positive training mass when a sparse weight ledger's
/// positive observations were omitted by a bag draw. Preserve declared group membership.
fn ensure_bag_training_mass(
    rows: &mut Vec<u32>,
    mask: &mut Option<Vec<bool>>,
    spec: &FitSpec<'_>,
) -> Result<(), PbError> {
    let Some(weights) = spec.weight else {
        return Ok(());
    };
    if rows.iter().enumerate().any(|(i, row)| {
        !mask
            .as_ref()
            .and_then(|m| m.get(i))
            .copied()
            .unwrap_or(false)
            && weights.get(*row as usize).is_some_and(|w| *w > 0.0)
    }) {
        return Ok(());
    }
    let positive = weights
        .iter()
        .enumerate()
        .find(|(row, weight)| {
            **weight > 0.0
                && !spec
                    .fixed_holdout
                    .and_then(|m| m.get(*row))
                    .copied()
                    .unwrap_or(false)
        })
        .map(|(row, _)| row)
        .ok_or_else(|| invalid_input("bag has no eligible positive training mass".into()))?;
    let group = spec.bag_groups.and_then(|g| g.get(positive)).copied();
    let retained = |row: usize| {
        row == positive
            || group.is_some_and(|id| spec.bag_groups.and_then(|g| g.get(row)) == Some(&id))
    };
    // If the source was carved into an automatic holdout, promote all its copies
    // together rather than putting a newly appended copy into training alone.
    if let Some(mask) = mask.as_mut() {
        for (row, held) in rows.iter().zip(mask.iter_mut()) {
            if retained(*row as usize) {
                *held = false;
            }
        }
    }
    for row in 0..weights.len() {
        if retained(row)
            && !rows.contains(&(row as u32))
            && !spec
                .fixed_holdout
                .and_then(|m| m.get(row))
                .copied()
                .unwrap_or(false)
        {
            rows.push(row as u32);
            if let Some(mask) = mask.as_mut() {
                mask.push(false);
            }
        }
    }
    Ok(())
}

fn fit_outer_bag(
    config: &Config,
    x: &BinnedMatrix,
    y: &[f32],
    spec: &FitSpec,
    n_bags: u16,
    bag_subsample: f32,
    cell_refit: Option<CellRefit>,
    cat_encoders: &CatEncoderStore,
) -> Result<Model, PbError> {
    validate_ensemble_fit_inputs(config, x, y, spec)?;
    let base_config = ensemble_base_config(config);
    if n_bags == 1 {
        return fit_single(&base_config, x, y, spec, cat_encoders);
    }

    let n_rows = x.n_rows as usize;
    let alpha = 1.0_f64 / f64::from(n_bags);
    // bag_subsample >= 1 ⇒ full-size bootstrap WITH replacement (classic bagging); < 1 ⇒
    // without-replacement subagging of round(f * n_rows) rows (faster, more diverse, leak-free).
    let subsample = bag_subsample < 1.0;
    let sample_len =
        (((n_rows as f64) * f64::from(bag_subsample)).round() as usize).clamp(1, n_rows);
    let collect_oob = cell_refit.is_some();
    // ES-stratify-style bagging fix: every bag draws from the SAME per-loss strata, computed
    // ONCE here rather than per-bag inside the (parallel) loop below — `None` for
    // Gamma/SquaredError (see `es_strata_for_loss`), which keeps those objectives' bags on the
    // byte-identical `subagging_rows_dispatch` fallthrough.
    let strata: Option<Vec<u32>> = es_strata_for_loss(spec.loss.objective_tag().loss, y);
    // Dev-only phase timers (gated by TBOOST_PROFILE; no-op otherwise).
    let prof = std::env::var_os("TBOOST_PROFILE").is_some();
    if prof {
        let feat = x.data.len();
        let shared_mb = (n_rows * feat) as f64 / 1e6;
        let bag_mb = (sample_len * feat) as f64 / 1e6;
        eprintln!(
            "[mem] structures: shared X {shared_mb:.0} MB ({n_rows} rows x {feat} feats, u8) | per-bag X ~{bag_mb:.0} MB x {n_bags} bags run concurrently",
        );
        prof::mem("fit start");
    }
    let oob_predict_ns = std::sync::atomic::AtomicU64::new(0);
    let t_bags = std::time::Instant::now();
    // Bags are independent (each seeded by its index) and collected IN BAG ORDER, so fitting them
    // concurrently is byte-identical to the sequential fit regardless of thread count; soup_models
    // then folds the members in that fixed order. Nests with fit_single's own rayon parallelism on the
    // shared pool (work-stealing), which is what recovers speed in the small-bag regime that
    // per-bag row subsampling cannot.
    let members: Vec<(WeightedModel, Option<BagOob>, Vec<bool>)> = (0..usize::from(n_bags))
        .into_par_iter()
        .map(
            |bag| -> Result<(WeightedModel, Option<BagOob>, Vec<bool>), PbError> {
                let bag_round = u32::try_from(bag).map_err(|_| PbError::Internal {
                    what: "OuterBag bag index exceeded u32".into(),
                })?;
                // One-honest-holdout mode (ordered TS + internal ES): every bag draws its
                // training rows from train-proper only and appends the SHARED holdout, which the
                // encoders were blinded to at binning time; the bag-local mask marks those rows so
                // fit_single early-stops on them instead of carving its own (leaky) slice.
                let (mut rows, mut bag_mask): (Vec<u32>, Option<Vec<bool>>) = match spec
                    .fixed_holdout
                {
                    Some(mask) => {
                        let train_idx: Vec<u32> =
                            (0..n_rows as u32).filter(|&r| !mask[r as usize]).collect();
                        // Same subagging fraction as the legacy path (sample_len / n_rows),
                        // applied to the train-proper pool.
                        let bag_len = ((train_idx.len() as f64) * (sample_len as f64)
                            / (n_rows as f64))
                            .ceil()
                            .max(1.0) as usize;
                        let picked = if let Some(g) = spec.bag_groups {
                            let train_groups = gather_strata(g, &train_idx)?;
                            let train_strata = match &strata {
                                Some(s) => Some(gather_strata(s, &train_idx)?),
                                None => None,
                            };
                            group_subagging_rows(
                                spec.seed,
                                bag_round,
                                &train_groups,
                                train_strata.as_deref(),
                                bag_len,
                            )?
                        } else if subsample {
                            let train_strata = match &strata {
                                Some(s) => Some(gather_strata(s, &train_idx)?),
                                None => None,
                            };
                            subagging_rows_dispatch(
                                spec.seed,
                                bag_round,
                                train_idx.len(),
                                train_strata.as_deref(),
                                bag_len,
                            )?
                        } else {
                            // Bootstrap (with-replacement) draws are left unstratified: the shipped
                            // recipe uses subagging (bag_subsample < 1.0), and containing the fix to
                            // that path keeps the stratified sampler's blast radius small.
                            bootstrap_rows(spec.seed, bag_round, train_idx.len(), train_idx.len())?
                        };
                        let mut rows: Vec<u32> =
                            picked.into_iter().map(|i| train_idx[i as usize]).collect();
                        let n_train_rows = rows.len();
                        rows.extend((0..n_rows as u32).filter(|&r| mask[r as usize]));
                        let mut m = vec![false; rows.len()];
                        for slot in m.iter_mut().skip(n_train_rows) {
                            *slot = true;
                        }
                        (rows, Some(m))
                    }
                    None => {
                        let rows = if let Some(g) = spec.bag_groups {
                            group_subagging_rows(
                                spec.seed,
                                bag_round,
                                g,
                                strata.as_deref(),
                                sample_len,
                            )?
                        } else if subsample {
                            subagging_rows_dispatch(
                                spec.seed,
                                bag_round,
                                n_rows,
                                strata.as_deref(),
                                sample_len,
                            )?
                        } else {
                            // Bootstrap (with-replacement) draws are left unstratified: see the
                            // comment in the `Some(mask)` arm above.
                            bootstrap_rows(spec.seed, bag_round, n_rows, n_rows)?
                        };
                        (rows, None)
                    }
                };
                if bag_mask.is_none() && !subsample {
                    bag_mask = bootstrap_holdout_mask(
                        &rows,
                        config.validation_fraction,
                        spec.seed,
                        bag_round,
                        strata.as_deref(),
                    )?;
                }
                ensure_bag_training_mass(&mut rows, &mut bag_mask, spec)?;
                let data = row_subset(x, y, spec.weight, spec.exposure, &rows)?;
                let bag_seed = pb_seed(spec.seed, bag_round, Stage::Sample as u32, 0);
                let bag_spec = FitSpec {
                    loss: spec.loss,
                    weight: data.weight.as_deref(),
                    exposure: data.exposure.as_deref(),
                    monotone: spec.monotone.clone(),
                    interaction: spec.interaction.clone(),
                    credibility: spec.credibility,
                    fixed_holdout: bag_mask.as_deref(),
                    bag_groups: None,
                    seed: bag_seed,
                };
                let mut bag_config = base_config.clone();
                bag_config.fit_control.bag = bag_round;
                let model = fit_single(&bag_config, &data.x, &data.y, &bag_spec, cat_encoders)?;
                // Membership over the FIT rows, recorded for EVERY bagged fit (not just the
                // cell-refit path that used to build it): `Model::bag_in_bag` publishes it so the
                // post-fit prune guard can use each bag's OUT-OF-BAG complement as honest evidence
                // without carving anything out of the fit. One bool per fit row per bag — 3 MB at
                // 400k rows x 8 bags, against a bag design matrix that is orders larger.
                let mut in_bag = vec![false; n_rows];
                for &r in &rows {
                    if let Some(slot) = in_bag.get_mut(r as usize) {
                        *slot = true;
                    }
                }
                let oob = if collect_oob {
                    let in_bag = in_bag.clone();
                    // Only out-of-bag rows are read from `preds` (attach reads `!in_bag`), so score
                    // just those — the other ~80% would be computed and discarded.
                    let oob_rows: Vec<u32> = (0..n_rows as u32)
                        .filter(|&r| !in_bag[r as usize])
                        .collect();
                    let mut preds = vec![0.0_f32; n_rows];
                    if prof {
                        let t = std::time::Instant::now();
                        model.score_trees_rows(x, &oob_rows, &mut preds)?;
                        oob_predict_ns.fetch_add(
                            t.elapsed().as_nanos() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    } else {
                        model.score_trees_rows(x, &oob_rows, &mut preds)?;
                    }
                    Some(BagOob { in_bag, preds })
                } else {
                    None
                };
                Ok((WeightedModel { alpha, model }, oob, in_bag))
            },
        )
        .collect::<Result<Vec<_>, _>>()?;

    let bags_s = t_bags.elapsed().as_secs_f64();
    prof::mem("after bag_loop");
    let mut weighted = Vec::with_capacity(members.len());
    let mut oobs = Vec::with_capacity(members.len());
    let mut bag_in_bag = Vec::with_capacity(members.len());
    for (wm, oob, in_bag) in members {
        weighted.push(wm);
        if let Some(o) = oob {
            oobs.push(o);
        }
        bag_in_bag.push(in_bag);
    }
    let t_soup = std::time::Instant::now();
    let mut model = soup_models(&weighted)?;
    // Bags were collected IN BAG ORDER above, matching `soup_models`' `bag_spans`.
    model.bag_in_bag = Some(bag_in_bag);
    let soup_s = t_soup.elapsed().as_secs_f64();
    match cell_refit {
        Some(cr) => {
            let t_attach = std::time::Instant::now();
            let corrected = attach_cell_correction(model, x, y, spec, &oobs, cr)?;
            prof::mem("after cell_refit");
            if prof {
                let oob_s = oob_predict_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;
                eprintln!(
                    "[outer-bag] bag_loop {bags_s:.2}s (OOB-predict cpu-sum {oob_s:.2}s, overlaps fit) | soup {soup_s:.2}s | cell_refit attach {:.2}s",
                    t_attach.elapsed().as_secs_f64(),
                );
            }
            Ok(corrected)
        }
        None => {
            if prof {
                eprintln!("[outer-bag] bag_loop {bags_s:.2}s | soup {soup_s:.2}s | no cell_refit");
            }
            Ok(model)
        }
    }
}

/// The realized supports (order 1..=2) the §G1 refit corrects: every main that appears in a
/// tree, plus every realized pair (each order-2 subset of any tree's support). Order-3
/// supports are left untouched so the depth-3 edge is preserved. Axis ids == raw ids under
/// the green spine, which is what `correction_scaffold` and the decompose fold expect.
///
/// P1 MULTI-CHANNEL: an axis whose raw feature owns MORE THAN ONE model axis
/// (`design/multichannel-categoricals.md`) is excluded. `correction_scaffold` builds each
/// table's `bin_to_cell` from `MergedAxis::model_bin_to_cell`, which a JOINT axis rejects by
/// design — one model bin never determines a joint merged cell on its own — so such a support
/// cannot be represented as a `CorrectionTable` at all. Before this filter the refit hit that
/// rejection as a hard `Internal` error on any `cat_channels` model; now those supports simply
/// go uncorrected and the rest of the bank is refit as usual. This is the same "skip rather
/// than misroute" call `prune::rebalance_on_means` already makes for the post-prune re-solve,
/// narrowed to the individual support instead of the whole model (here the axis ids come from
/// the trees, so there is no raw-vs-axis id confusion to defend against — only the joint
/// axis's own inability to carry a per-bin cell map).
fn correctable_supports(model: &Model) -> Vec<Vec<u32>> {
    use std::collections::BTreeSet;
    let multichannel = |axis: u32| -> bool {
        model
            .provenance
            .get(axis as usize)
            .is_some_and(|p| crate::data::axes_for_raw(&model.provenance, p.raw).len() > 1)
    };
    let mut mains: BTreeSet<u32> = BTreeSet::new();
    let mut pairs: BTreeSet<(u32, u32)> = BTreeSet::new();
    for (_, tree) in &model.trees {
        let mut axes: Vec<u32> = Vec::new();
        for s in &tree.splits {
            if !axes.contains(&s.axis) && !multichannel(s.axis) {
                axes.push(s.axis);
            }
        }
        for &a in &axes {
            mains.insert(a);
        }
        for i in 0..axes.len() {
            for j in (i + 1)..axes.len() {
                let (lo, hi) = if axes[i] < axes[j] {
                    (axes[i], axes[j])
                } else {
                    (axes[j], axes[i])
                };
                pairs.insert((lo, hi));
            }
        }
    }
    let mut supports: Vec<Vec<u32>> = Vec::with_capacity(mains.len() + pairs.len());
    for a in mains {
        supports.push(vec![a]);
    }
    for (a, b) in pairs {
        supports.push(vec![a, b]);
    }
    supports
}

/// How many realized order-<=2 supports [`correctable_supports`] had to drop because they touch
/// a P1 multi-channel raw feature. Diagnostic only (`TBOOST_PROFILE`): it is the denominator
/// that says whether a declined correction means "no signal" or "could not reach the surfaces".
fn realized_supports_blocked_by_channels(model: &Model) -> usize {
    use std::collections::BTreeSet;
    let n_raw = crate::data::n_raw_features(&model.provenance);
    if model.provenance.len() == n_raw {
        return 0; // no multi-channel raw feature anywhere; nothing can be blocked
    }
    let mut all_mains: BTreeSet<u32> = BTreeSet::new();
    let mut all_pairs: BTreeSet<(u32, u32)> = BTreeSet::new();
    for (_, tree) in &model.trees {
        let mut axes: Vec<u32> = Vec::new();
        for s in &tree.splits {
            if !axes.contains(&s.axis) {
                axes.push(s.axis);
            }
        }
        for &a in &axes {
            all_mains.insert(a);
        }
        for i in 0..axes.len() {
            for j in (i + 1)..axes.len() {
                let (lo, hi) = if axes[i] < axes[j] {
                    (axes[i], axes[j])
                } else {
                    (axes[j], axes[i])
                };
                all_pairs.insert((lo, hi));
            }
        }
    }
    (all_mains.len() + all_pairs.len()) - correctable_supports(model).len()
}

/// Fit the §G1 cell-basis correction to the bagged out-of-bag residual and attach it to the
/// soup. Out-of-bag = honest cross-fit: each row's residual uses only the bags that did not
/// train on it, so the correction tightens the soup's interaction surfaces without the
/// in-sample over-fit that broke earlier totally-corrective attempts. Determinism: the OOB
/// accumulation, the Newton residual, and the CG solve are all sequential / fixed-order.
fn attach_cell_correction(
    mut model: Model,
    x: &BinnedMatrix,
    y: &[f32],
    spec: &FitSpec,
    oobs: &[BagOob],
    cr: CellRefit,
) -> Result<Model, PbError> {
    let n_rows = x.n_rows as usize;
    // Out-of-bag raw prediction per row: mean over bags that did NOT train on the row.
    let mut oob_sum = vec![0.0_f64; n_rows];
    let mut oob_cnt = vec![0u32; n_rows];
    for bag in oobs {
        for r in 0..n_rows {
            if !bag.in_bag[r] {
                oob_sum[r] += f64::from(bag.preds[r]);
                oob_cnt[r] += 1;
            }
        }
    }
    // Rows with no OOB coverage fall back to the intercept f0 — any in-domain raw works, since
    // they carry weight 0 below and their grad_hess is discarded. (Previously this was a full
    // soup prediction over every row, a wasted whole-ensemble pass that dominated the attach on
    // large, many-tree models like particulate.)
    let mut oob_raw = vec![0.0_f32; n_rows];
    for r in 0..n_rows {
        oob_raw[r] = if oob_cnt[r] > 0 {
            (oob_sum[r] / f64::from(oob_cnt[r])) as f32
        } else {
            model.f0
        };
    }
    // Each bag trained with `spec.exposure` folded into its raw (§03.7, mirroring fit_single's
    // `raw = f0 + offset`), but `score_trees_rows` never adds it back, so `oob_raw` is currently
    // f0 + Σ alpha·tree with the offset missing. Add it once, here, so both the residual pass
    // below AND the no-harm guard (which reads `hraw` straight from `oob_raw`) evaluate at the
    // same raw the bags were fit against — otherwise the OOB residual carries a spurious
    // ln(exposure) pattern that the correction bank would absorb as bogus cell structure.
    if let Some(e) = spec.exposure {
        let offset = compute_offset(e, n_rows)?;
        for (r, o) in oob_raw.iter_mut().zip(&offset) {
            *r += o;
        }
    }
    // Newton working residual z = -g/h with IRLS weight h, evaluated at the OOB prediction.
    // (Squared error → z = y − raw, h = 1; logistic → z = (y − p)/(p(1−p)), h = p(1−p).)
    let sample_w: Vec<f32> = match spec.weight {
        Some(w) => w.to_vec(),
        None => vec![1.0_f32; n_rows],
    };
    let mut gh = crate::loss::GradHess {
        g: vec![0.0_f32; n_rows],
        h: vec![0.0_f32; n_rows],
    };
    spec.loss.grad_hess(y, &oob_raw, &sample_w, &mut gh)?;
    // A deterministic ~15% slice held OUT of the correction fit, used below to choose a global
    // shrinkage so the correction can only help on honest data (the no-harm guard).
    let is_holdout = |r: usize| {
        let identity = spec
            .bag_groups
            .and_then(|groups| groups.get(r))
            .copied()
            .map_or(r as u64, u64::from);
        (identity.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) < 38
    };
    let mut residual = vec![0.0_f64; n_rows];
    let mut weight = vec![0.0_f64; n_rows];
    for r in 0..n_rows {
        if oob_cnt[r] > 0 && !is_holdout(r) {
            let h = f64::from(gh.h[r]).max(1e-12);
            residual[r] = -f64::from(gh.g[r]) / h;
            weight[r] = h;
        }
    }
    let supports = correctable_supports(&model);
    if supports.is_empty() {
        return Ok(model);
    }
    let refit_spec = crate::cell_refit::CellRefitSpec {
        base: cr.base,
        gamma: cr.gamma,
        ..Default::default()
    };
    let prof = std::env::var_os("TBOOST_PROFILE").is_some();
    if prof {
        let n_solve = weight
            .iter()
            .filter(|&&w| w > 0.0)
            .count()
            .min(refit_spec.max_fit_rows);
        let design_mb = (supports.len() * n_solve * 2) as f64 / 1e6;
        eprintln!(
            "[mem] cell_refit design: active ~{design_mb:.0} MB ({} supports x {n_solve} fit rows, u16)",
            supports.len(),
        );
    }
    let t_solve = std::time::Instant::now();
    let bank = crate::cell_refit::fit_cell_correction(
        &model,
        &x.data,
        &residual,
        &weight,
        &supports,
        &refit_spec,
    )?;
    let solve_s = t_solve.elapsed().as_secs_f64();
    model.correction = Some(bank);
    let t_guard = std::time::Instant::now();
    // No-harm guard: choose a global shrinkage λ ∈ [0,1] minimising held-out deviance of
    // `oob_raw + λ·correction` on the OOB-covered held-out slice. λ=0 (the correction does not
    // generalise — e.g. noisy high-cardinality categorical pairs) drops it entirely; λ=1 keeps
    // it at full strength (signal that generalises, e.g. particulate's time-of-day pairs). This
    // is the prototype's held-out config selection, built into the fit, so the refit can only
    // help or be neutralised — never regress a winner.
    let mut hy: Vec<f32> = Vec::new();
    let mut hraw: Vec<f32> = Vec::new();
    let mut hw: Vec<f32> = Vec::new();
    let mut hd: Vec<f32> = Vec::new();
    let mut row_bins = vec![0u8; x.data.len()];
    for r in 0..n_rows {
        if oob_cnt[r] > 0 && is_holdout(r) {
            for (a, col) in x.data.iter().enumerate() {
                if let Some(&bin) = col.get(r) {
                    row_bins[a] = bin;
                }
            }
            hy.push(y[r]);
            hraw.push(oob_raw[r]);
            hw.push(sample_w[r]);
            hd.push(model.correction_delta(&row_bins)? as f32);
        }
    }
    // No held-out coverage → cannot evaluate → keep the correction at full strength.
    let mut best_lambda = 1.0_f32;
    if !hy.is_empty() {
        best_lambda = 0.0;
        let mut best_dev = f32::INFINITY;
        let mut raw_buf = vec![0.0_f32; hy.len()];
        for &lam in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            for (i, slot) in raw_buf.iter_mut().enumerate() {
                *slot = hraw[i] + lam * hd[i];
            }
            let dev = spec.loss.deviance(&hy, &raw_buf, &hw)?;
            if dev < best_dev {
                best_dev = dev;
                best_lambda = lam;
            }
        }
    }
    if best_lambda <= 0.0 {
        model.correction = None;
    } else if best_lambda < 1.0 {
        if let Some(bank) = &mut model.correction {
            for table in &mut bank.tables {
                for v in &mut table.values {
                    *v *= f64::from(best_lambda);
                }
            }
        }
    }
    if prof {
        eprintln!(
            "[cell_refit] solve(design+CG) {solve_s:.2}s | guard(held-out λ={best_lambda}) {:.2}s | {} supports",
            t_guard.elapsed().as_secs_f64(),
            supports.len(),
        );
    }
    model.validate()?;
    Ok(model)
}

fn fit_greedy_select(
    config: &Config,
    x: &BinnedMatrix,
    y: &[f32],
    spec: &FitSpec,
    params: GreedyParams<'_>,
    cat_encoders: &CatEncoderStore,
) -> Result<Model, PbError> {
    validate_ensemble_fit_inputs(config, x, y, spec)?;
    let base_config = ensemble_base_config(config);
    // One-honest-holdout mode (ordered TS + internal ES): a caller-supplied fixed_holdout mask
    // names the ONLY rows the categorical TS encoders were blinded to at binning time
    // (engine/mod.rs's FitSpec::fixed_holdout doc). Mirroring fit_outer_bag, those rows become
    // BOTH the GreedySelect selection holdout below AND every member's own internal
    // early-stopping set (member_spec.fixed_holdout below), never a fresh random carve that
    // could put a TS-leaky slice into either member training or member/selection validation.
    // Without a mask, the existing seeded 80/20 holdout_split is used instead.
    let (train_rows, holdout_rows) = match spec.fixed_holdout {
        Some(mask) => holdout_rows_from_mask(x.n_rows, mask)?,
        None => holdout_split(spec.seed, x.n_rows as usize)?,
    };
    let holdout = row_subset(x, y, spec.weight, spec.exposure, &holdout_rows)?;
    let holdout_weight = effective_weight(&holdout);
    // Members mirror fit_outer_bag when a fixed_holdout mask is present: fit_single is handed
    // the FULL x/y with that SAME mask (no train-only subsetting needed — unlike fit_outer_bag's
    // bags, members never subsample rows), so its own internal train/validation split is
    // byte-identical to (train_rows, holdout_rows) above, holdout rows never enter tree growth
    // (fit_single restricts growth/refit to its train rows internally), and they still serve as
    // that member's own honest early-stopping set. Without a mask, each member instead trains on
    // the train-only 80% (`train`) with fit_single carving its own internal random carve.
    let train: Option<OwnedFitData> = match spec.fixed_holdout {
        Some(_) => None,
        None => Some(row_subset(x, y, spec.weight, spec.exposure, &train_rows)?),
    };
    type MemberFitInputs<'a> = (
        &'a BinnedMatrix,
        &'a [f32],
        Option<&'a [f32]>,
        Option<&'a [f32]>,
    );
    let (member_x, member_y, member_weight, member_exposure): MemberFitInputs<'_> = match &train {
        Some(t) => (&t.x, &t.y, t.weight.as_deref(), t.exposure.as_deref()),
        None => (x, y, spec.weight, spec.exposure),
    };
    // Members train WITH the exposure offset (member_spec.exposure = train.exposure below), but
    // `raw_predictions` scores via `score_trees(.., None, ..)`, so it never adds it back. Every
    // member shares the SAME holdout rows, so the offset is one common additive vector; compute
    // it once here and fold it into each member's holdout_raw before it is used for anything —
    // the library deviance below, and every deviance_for_rows/mix_raw call inside
    // greedy_selection_weights, which only ever reads holdout_raw back out of LibraryMember.
    // (`mix_raw`'s running means are convex combinations, so a common per-row offset survives
    // the greedy mixing exactly, and is safe to add just this once.)
    let holdout_offset: Option<Vec<f32>> = match holdout.exposure.as_deref() {
        Some(e) => Some(compute_offset(e, holdout.y.len())?),
        None => None,
    };

    let library_size = usize::from(params.library_size);
    let mut library = Vec::new();
    library
        .try_reserve_exact(library_size)
        .map_err(|_| PbError::Internal {
            what: "GreedySelect library allocation failed".into(),
        })?;
    for ordinal in 0..params.library_size {
        let choice = hp_choice_at(params.hp_grid, usize::from(ordinal))?;
        let mut member_config = base_config.clone();
        member_config.lambda = choice.lambda;
        member_config.learning_rate = choice.learning_rate;
        member_config.n_trees = choice.n_trees;
        member_config.boosters.random_strength = choice.random_strength;
        member_config.validate()?;

        let mut interaction = spec.interaction.clone();
        interaction.max_order = choice.max_order;
        // FLAG (M4/M5 seam): this core entrypoint consumes a frozen BinnedMatrix, so
        // HpGrid::max_bins cannot rebuild raw-data grids here. It is still validated
        // by BoosterConfig and folded into the deterministic member seed; raw callers
        // can materialize distinct grids before crossing this binned seam.
        let hp_block = u32::from(choice.max_bin)
            .checked_add(u32::from(ordinal))
            .ok_or_else(|| PbError::Internal {
                what: "GreedySelect HP seed block overflow".into(),
            })?;
        let member_seed = pb_seed(
            spec.seed,
            u32::from(ordinal),
            Stage::Sample as u32,
            hp_block,
        );
        let member_spec = FitSpec {
            loss: spec.loss,
            weight: member_weight,
            exposure: member_exposure,
            monotone: spec.monotone.clone(),
            interaction,
            credibility: spec.credibility,
            fixed_holdout: spec.fixed_holdout,
            bag_groups: spec.bag_groups,
            seed: member_seed,
        };
        let model = fit_single(
            &member_config,
            member_x,
            member_y,
            &member_spec,
            cat_encoders,
        )?;
        let mut holdout_raw = raw_predictions(&model, &holdout.x)?;
        if let Some(off) = &holdout_offset {
            for (r, o) in holdout_raw.iter_mut().zip(off) {
                *r += o;
            }
        }
        let deviance = f64::from(
            spec.loss
                .deviance(&holdout.y, &holdout_raw, &holdout_weight)?,
        );
        library.push(LibraryMember {
            model,
            holdout_raw,
            deviance,
        });
    }

    let weights = greedy_selection_weights(
        &library,
        &holdout.y,
        &holdout_weight,
        spec.loss,
        spec.seed,
        usize::from(params.selection_bags),
        usize::from(params.seed_top_n),
    )?;
    let mut members = Vec::new();
    members
        .try_reserve_exact(library.len())
        .map_err(|_| PbError::Internal {
            what: "GreedySelect soup allocation failed".into(),
        })?;
    for (alpha, member) in weights.into_iter().zip(library) {
        if alpha > 0.0 {
            members.push(WeightedModel {
                alpha,
                model: member.model,
            });
        }
    }
    soup_models(&members)
}

fn row_subset(
    x: &BinnedMatrix,
    y: &[f32],
    weight: Option<&[f32]>,
    exposure: Option<&[f32]>,
    rows: &[u32],
) -> Result<OwnedFitData, PbError> {
    let n_rows = u32::try_from(rows.len()).map_err(|_| PbError::InvalidInput {
        what: "row subset has more than u32::MAX rows".into(),
    })?;
    let mut data = Vec::new();
    data.try_reserve_exact(x.data.len())
        .map_err(|_| PbError::Internal {
            what: "row subset column allocation failed".into(),
        })?;
    for col in &x.data {
        let mut out = Vec::new();
        out.try_reserve_exact(rows.len())
            .map_err(|_| PbError::Internal {
                what: "row subset data allocation failed".into(),
            })?;
        for &row in rows {
            let idx = row as usize;
            out.push(*col.get(idx).ok_or_else(|| PbError::Internal {
                what: "row subset escaped binned column".into(),
            })?);
        }
        data.push(out);
    }
    Ok(OwnedFitData {
        x: BinnedMatrix {
            data,
            n_rows,
            grids: x.grids.clone(),
            provenance: x.provenance.clone(),
        },
        y: gather_f32("y", y, rows)?,
        weight: match weight {
            Some(values) => Some(gather_f32("weight", values, rows)?),
            None => None,
        },
        exposure: match exposure {
            Some(values) => Some(gather_f32("exposure", values, rows)?),
            None => None,
        },
    })
}

fn gather_f32(label: &'static str, values: &[f32], rows: &[u32]) -> Result<Vec<f32>, PbError> {
    let mut out = Vec::new();
    out.try_reserve_exact(rows.len())
        .map_err(|_| PbError::Internal {
            what: format!("{label} subset allocation failed"),
        })?;
    for &row in rows {
        out.push(*values.get(row as usize).ok_or_else(|| PbError::Internal {
            what: format!("{label} subset escaped source rows"),
        })?);
    }
    Ok(out)
}

fn bootstrap_rows(
    seed: u64,
    round: u32,
    n_rows: usize,
    sample_len: usize,
) -> Result<Vec<u32>, PbError> {
    if n_rows == 0 {
        return Ok(Vec::new());
    }
    let n_u64 = u64::try_from(n_rows).map_err(|_| PbError::InvalidInput {
        what: "bootstrap supports at most u64::MAX rows".into(),
    })?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(sample_len)
        .map_err(|_| PbError::Internal {
            what: "bootstrap row allocation failed".into(),
        })?;
    for i in 0..sample_len {
        let block = u32::try_from(i).map_err(|_| PbError::InvalidInput {
            what: "bootstrap supports at most u32::MAX sampled rows".into(),
        })?;
        let row = pb_seed(seed, round, Stage::Sample as u32, block) % n_u64;
        rows.push(u32::try_from(row).map_err(|_| PbError::Internal {
            what: "bootstrap row exceeded u32".into(),
        })?);
    }
    Ok(rows)
}

/// Draw `k` DISTINCT row indices from `[0, n_rows)` without replacement (subagging), deterministically
/// via the per-bag `Pcg64` stream and a partial Fisher–Yates, returned sorted ascending for
/// cache-friendly `row_subset`. Unlike `bootstrap_rows` (with replacement), no original row appears
/// twice in a bag — which removes the train/val overlap leak in each bag's early-stop carve and
/// injects more cross-bag diversity. Thread-independent ⇒ byte-deterministic.
fn subagging_rows(seed: u64, round: u32, n_rows: usize, k: usize) -> Result<Vec<u32>, PbError> {
    let k = k.min(n_rows);
    let n32 = u32::try_from(n_rows).map_err(|_| PbError::InvalidInput {
        what: "subagging supports at most u32::MAX rows".into(),
    })?;
    let mut idx: Vec<u32> = Vec::new();
    idx.try_reserve_exact(n_rows)
        .map_err(|_| PbError::Internal {
            what: "subagging index allocation failed".into(),
        })?;
    idx.extend(0..n32);
    let mut rng = pb_rng(seed, round, Stage::Sample, 0);
    for i in 0..k {
        let range = u64::try_from(n_rows - i).unwrap_or(1).max(1);
        let offset = usize::try_from(rng.next_u64() % range).unwrap_or(0);
        idx.swap(i, i + offset);
    }
    idx.truncate(k);
    idx.sort_unstable();
    Ok(idx)
}

/// Exact-`k` largest-remainder apportionment of `k` draws across groups of sizes `group_lens`
/// (parallel, same order in and out — [`stratified_subagging_rows`] supplies them in ascending
/// `BTreeMap`-key order, which is what makes the tie-break below a deterministic function of the
/// stratum keys rather than of iteration/sort-stability accidents). Each group's base share is
/// `floor(k * group_len / n_total)`; the `k - Σfloor` leftover units go to the groups with the
/// largest fractional remainder, ties broken by earliest position in `group_lens`. Every entry is
/// clamped to `[0, group_len]`, and the output ALWAYS sums to EXACTLY `k` (`k` is silently
/// clamped to `n_total = Σgroup_lens` first, mirroring `subagging_rows`' `k.min(n_rows)`) —
/// verified by `stratified_apportionment_always_sums_to_exactly_k` across tiny/singleton/near-`n`
/// group shapes, not just argued from the floor/remainder algebra.
fn stratified_apportionment(group_lens: &[usize], k: usize) -> Vec<usize> {
    let n_total: usize = group_lens.iter().sum();
    if n_total == 0 {
        return vec![0; group_lens.len()];
    }
    let k = k.min(n_total);
    // (group len, base share) in ORIGINAL `group_lens` order — kept index-aligned throughout so
    // the final `Vec` needs no re-sort to restore that order.
    let mut items: Vec<(usize, usize)> = Vec::with_capacity(group_lens.len());
    let mut fracs: Vec<f64> = Vec::with_capacity(group_lens.len());
    let mut used = 0usize;
    for &len in group_lens {
        let raw = (k as f64) * (len as f64) / (n_total as f64);
        let floor = (raw.floor() as usize).min(len);
        used += floor;
        fracs.push(raw - floor as f64);
        items.push((len, floor));
    }
    let mut remaining = k.saturating_sub(used);
    // Positions into `items`/`fracs`, ordered by descending fractional remainder, ties broken
    // by ascending position (== ascending stratum key at the caller) — sorting this small index
    // list (not `items` itself) is what lets the redistribution loop below mutate `items` in
    // place while still walking it in largest-remainder order.
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.sort_by(|&a, &b| {
        let fa = fracs.get(a).copied().unwrap_or(0.0);
        let fb = fracs.get(b).copied().unwrap_or(0.0);
        fb.total_cmp(&fa).then_with(|| a.cmp(&b))
    });
    // Sweeps `order` repeatedly until every leftover unit is placed. One pass always suffices
    // when `k <= n_total` (enforced above): `remaining == Σfracs < count of groups`, so the
    // first `remaining` entries in `order` still have spare capacity. The outer loop does not
    // rely on that proof — it keeps sweeping until nothing more can be placed — so it stays
    // exact even if a future caller relaxes the `k <= n_total` precondition.
    while remaining > 0 {
        let mut progressed = false;
        for &pos in &order {
            if remaining == 0 {
                break;
            }
            if let Some(item) = items.get_mut(pos) {
                if item.1 < item.0 {
                    item.1 += 1;
                    remaining -= 1;
                    progressed = true;
                }
            }
        }
        if !progressed {
            break;
        }
    }
    items.into_iter().map(|(_, count)| count).collect()
}

/// Row indices in `[0, strata.len())` drawn WITHOUT replacement (subagging), stratified so every
/// distinct value in `strata` keeps its proportional share of the `k` draw. The bagging analog of
/// [`holdout_mask`]'s stratified branch: an unstratified [`subagging_rows`] can draw a bag with
/// few or no rows from a rare event (measured: an 876-row/8-positive set, 8 bags at
/// `bag_subsample=0.8`, produced per-bag link-scale balance from 0.10 to 1.54 across splits),
/// which collapses that bag's fitted level before the soup averages it back out.
///
/// Groups rows by stratum (`BTreeMap`, so iteration — and every tie-break, here and in
/// [`stratified_apportionment`] — is in ascending stratum-key order), apportions `k` across
/// groups, then draws each group's share with the SAME partial Fisher–Yates `subagging_rows`
/// uses, seeded independently per stratum via `pb_rng(seed, round, Stage::Bagging, stratum_key)`
/// — the FIRST use of the `Bagging` stage (`backend.rs`): defined and documented since the stage
/// enumeration was written, never constructed until now. Concatenated and sorted ascending,
/// matching `subagging_rows`' output contract for `row_subset`.
fn stratified_subagging_rows(
    seed: u64,
    round: u32,
    strata: &[u32],
    k: usize,
) -> Result<Vec<u32>, PbError> {
    let n_rows = strata.len();
    let k = k.min(n_rows);
    let mut groups: std::collections::BTreeMap<u32, Vec<u32>> = std::collections::BTreeMap::new();
    for (row, &s) in strata.iter().enumerate() {
        let row_u32 = u32::try_from(row).map_err(|_| PbError::InvalidInput {
            what: "stratified subagging supports at most u32::MAX rows".into(),
        })?;
        groups.entry(s).or_default().push(row_u32);
    }
    let group_lens: Vec<usize> = groups.values().map(Vec::len).collect();
    let counts = stratified_apportionment(&group_lens, k);
    let mut idx: Vec<u32> = Vec::new();
    idx.try_reserve_exact(k).map_err(|_| PbError::Internal {
        what: "stratified subagging row allocation failed".into(),
    })?;
    for ((&key, rows), &take) in groups.iter().zip(&counts) {
        let mut pool = rows.clone();
        let m = pool.len();
        let mut rng = pb_rng(seed, round, Stage::Bagging, key);
        for i in 0..take {
            let range = u64::try_from(m - i).unwrap_or(1).max(1);
            let offset = usize::try_from(rng.next_u64() % range).unwrap_or(0);
            pool.swap(i, i + offset);
        }
        pool.truncate(take);
        idx.extend(pool);
    }
    idx.sort_unstable();
    Ok(idx)
}

/// `strata: None` ⇒ [`subagging_rows`] UNCHANGED (byte-identical: same seed/round/n_rows/k,
/// verified by `subagging_rows_dispatch_with_no_strata_is_byte_identical_to_subagging_rows`) —
/// the fallthrough every Gamma/SquaredError/plain-regression bag takes, since
/// [`es_strata_for_loss`] returns `None` for those objectives. `strata: Some(s)` ⇒
/// [`stratified_subagging_rows`]. The single call site [`fit_outer_bag`] and
/// [`fit_multiclass_bagged`]'s four draw points all route through, so the strata gate lives in
/// exactly one place.
/// GROUP-AWARE subagging (2026-09-07): draw `sample_rows` worth of WHOLE groups. `groups[i]` is
/// row `i`'s group id over the index space being drawn (so a caller drawing among a holdout's
/// complement passes the complement's gathered ids); `strata`, when given, is per row and a
/// group's stratum is the majority stratum of its rows (ties to the smaller id), so a rare
/// class keeps its share of GROUPS in every bag as `stratified_subagging_rows` keeps its share
/// of rows. The number of groups drawn is `sample_rows / n_rows` of the distinct groups
/// (rounded, at least one); the returned row indices are ascending. Deterministic in
/// `(seed, round, groups, strata, sample_rows)`; the group index space is first-appearance
/// order, so a relabelled but identically partitioned `groups` draws the identical bag.
fn group_subagging_rows(
    seed: u64,
    round: u32,
    groups: &[u32],
    strata: Option<&[u32]>,
    sample_rows: usize,
) -> Result<Vec<u32>, PbError> {
    let n_rows = groups.len();
    if n_rows == 0 {
        return Ok(Vec::new());
    }
    if let Some(s) = strata {
        if s.len() != n_rows {
            return Err(PbError::ShapeMismatch {
                what: format!("bag strata len {} != groups len {n_rows}", s.len()),
            });
        }
    }
    let mut dense: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut rows_of: Vec<Vec<u32>> = Vec::new();
    for (i, &g) in groups.iter().enumerate() {
        let row = u32::try_from(i).map_err(|_| PbError::InvalidInput {
            what: "group subagging supports at most u32::MAX rows".into(),
        })?;
        let next = rows_of.len() as u32;
        let id = *dense.entry(g).or_insert(next);
        if id as usize == rows_of.len() {
            rows_of.push(Vec::new());
        }
        rows_of[id as usize].push(row);
    }
    let n_groups = rows_of.len();
    let group_strata: Option<Vec<u32>> = strata.map(|s| {
        rows_of
            .iter()
            .map(|rows| {
                let mut counts: std::collections::BTreeMap<u32, usize> =
                    std::collections::BTreeMap::new();
                for &r in rows {
                    *counts.entry(s[r as usize]).or_insert(0) += 1;
                }
                // Majority stratum; BTreeMap order + strict `>` make ties go to the smaller id.
                let mut best = (0u32, 0usize);
                for (&k, &c) in &counts {
                    if c > best.1 {
                        best = (k, c);
                    }
                }
                best.0
            })
            .collect()
    });
    let k_groups = (((n_groups as f64) * (sample_rows as f64) / (n_rows as f64)).round() as usize)
        .clamp(1, n_groups);
    let picked = subagging_rows_dispatch(seed, round, n_groups, group_strata.as_deref(), k_groups)?;
    let mut rows: Vec<u32> = picked
        .iter()
        .flat_map(|&g| rows_of[g as usize].iter().copied())
        .collect();
    rows.sort_unstable();
    Ok(rows)
}

fn subagging_rows_dispatch(
    seed: u64,
    round: u32,
    n_rows: usize,
    strata: Option<&[u32]>,
    k: usize,
) -> Result<Vec<u32>, PbError> {
    match strata {
        Some(s) => {
            if s.len() != n_rows {
                return Err(PbError::ShapeMismatch {
                    what: format!(
                        "stratified subagging strata len {} != n_rows {n_rows}",
                        s.len()
                    ),
                });
            }
            stratified_subagging_rows(seed, round, s, k)
        }
        None => subagging_rows(seed, round, n_rows, k),
    }
}

/// Re-index `strata` (parallel to the FULL row range `0..n_rows`) down to just `train_idx`'s
/// rows, in `train_idx`'s order — what a `fixed_holdout` bag's stratified draw needs, since it
/// samples over the train-proper pool `0..train_idx.len()`, not the full row range (mirrors
/// `gather_f32`'s row-subset-with-bounds-check shape).
fn gather_strata(strata: &[u32], train_idx: &[u32]) -> Result<Vec<u32>, PbError> {
    train_idx
        .iter()
        .map(|&r| {
            strata
                .get(r as usize)
                .copied()
                .ok_or_else(|| PbError::Internal {
                    what: "train_idx row escaped strata".into(),
                })
        })
        .collect()
}

fn holdout_split(seed: u64, n_rows: usize) -> Result<(Vec<u32>, Vec<u32>), PbError> {
    if n_rows < 2 {
        return Err(PbError::InvalidInput {
            what: "GreedySelect requires at least two rows for a held-out deviance split".into(),
        });
    }
    let mut keyed = Vec::new();
    keyed
        .try_reserve_exact(n_rows)
        .map_err(|_| PbError::Internal {
            what: "holdout split allocation failed".into(),
        })?;
    for row in 0..n_rows {
        let row_u32 = u32::try_from(row).map_err(|_| PbError::InvalidInput {
            what: "GreedySelect supports at most u32::MAX rows".into(),
        })?;
        keyed.push((pb_seed(seed, 0, Stage::Sample as u32, row_u32), row_u32));
    }
    keyed.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let holdout_len = (n_rows / 5).max(1).min(n_rows - 1);
    let mut holdout: Vec<u32> = keyed
        .iter()
        .take(holdout_len)
        .map(|(_, row)| *row)
        .collect();
    let mut train: Vec<u32> = keyed
        .iter()
        .skip(holdout_len)
        .map(|(_, row)| *row)
        .collect();
    holdout.sort_unstable();
    train.sort_unstable();
    Ok((train, holdout))
}

/// The GreedySelect (train rows, holdout rows) split for a caller-supplied `fixed_holdout` mask,
/// in the same `(Vec<u32>, Vec<u32>)` shape `holdout_split` returns. Errors if the mask names no
/// holdout rows at all — GreedySelect always needs a non-empty selection holdout, unlike
/// `fit_single`'s own `split_rows_by_mask` use, where an empty validation set just disables early
/// stopping.
fn holdout_rows_from_mask(n_rows: u32, mask: &[bool]) -> Result<(Vec<u32>, Vec<u32>), PbError> {
    let (train_rows, validation_rows) = split_rows_by_mask(n_rows, mask)?;
    let validation_rows = validation_rows.ok_or_else(|| PbError::InvalidInput {
        what: "GreedySelect requires at least one fixed_holdout row for a held-out deviance split"
            .into(),
    })?;
    let mut holdout_rows = Vec::new();
    holdout_rows
        .try_reserve_exact(validation_rows.len())
        .map_err(|_| PbError::Internal {
            what: "GreedySelect fixed_holdout allocation failed".into(),
        })?;
    for row in validation_rows {
        holdout_rows.push(u32::try_from(row).map_err(|_| PbError::Internal {
            what: "fixed_holdout row index exceeded u32".into(),
        })?);
    }
    Ok((train_rows, holdout_rows))
}

fn effective_weight(data: &OwnedFitData) -> Vec<f32> {
    data.weight
        .clone()
        .unwrap_or_else(|| vec![1.0_f32; data.y.len()])
}

fn hp_choice_at(grid: &HpGrid, ordinal: usize) -> Result<HpChoice, PbError> {
    fn take<T: Copy>(values: &[T], cursor: &mut usize) -> Result<T, PbError> {
        if values.is_empty() {
            return Err(PbError::InvalidConfig {
                what: "HpGrid candidate lists must be non-empty".into(),
            });
        }
        let idx = *cursor % values.len();
        *cursor /= values.len();
        values.get(idx).copied().ok_or_else(|| PbError::Internal {
            what: "HpGrid index escaped candidate list".into(),
        })
    }

    let mut cursor = ordinal;
    Ok(HpChoice {
        max_bin: take(&grid.max_bins, &mut cursor)?,
        lambda: take(&grid.lambdas, &mut cursor)?,
        learning_rate: take(&grid.learning_rates, &mut cursor)?,
        n_trees: take(&grid.n_trees, &mut cursor)?,
        max_order: take(&grid.max_interaction_orders, &mut cursor)?,
        random_strength: take(&grid.random_strengths, &mut cursor)?,
    })
}

fn raw_predictions(model: &Model, x: &BinnedMatrix) -> Result<Vec<f32>, PbError> {
    let mut raw = vec![0.0_f32; x.n_rows as usize];
    model.score_trees(x, None, &mut raw)?;
    Ok(raw)
}

fn greedy_selection_weights(
    library: &[LibraryMember],
    y: &[f32],
    weight: &[f32],
    loss: &dyn crate::loss::Loss,
    seed: u64,
    selection_bags: usize,
    seed_top_n: usize,
) -> Result<Vec<f64>, PbError> {
    if library.is_empty() {
        return Err(PbError::InvalidConfig {
            what: "GreedySelect requires at least one library member".into(),
        });
    }
    if selection_bags == 0 || seed_top_n == 0 || seed_top_n > library.len() {
        return Err(PbError::InvalidConfig {
            what: "GreedySelect selection_bags and seed_top_n are inconsistent".into(),
        });
    }
    let mut order: Vec<usize> = (0..library.len()).collect();
    order.sort_by(|&a, &b| {
        library
            .get(a)
            .map(|m| m.deviance)
            .unwrap_or(f64::INFINITY)
            .total_cmp(&library.get(b).map(|m| m.deviance).unwrap_or(f64::INFINITY))
            .then_with(|| a.cmp(&b))
    });
    let mut totals = vec![0.0_f64; library.len()];
    for bag in 0..selection_bags {
        let eval = bootstrap_indices(
            seed,
            u32::try_from(bag).map_err(|_| PbError::InvalidInput {
                what: "GreedySelect supports at most u32::MAX selection bags".into(),
            })?,
            y.len(),
        )?;
        let mut counts = vec![0u32; library.len()];
        let mut best_seed = *order.first().ok_or_else(|| PbError::Internal {
            what: "GreedySelect empty ordering".into(),
        })?;
        let mut best_loss = f64::INFINITY;
        for &candidate in order.iter().take(seed_top_n) {
            let score = deviance_for_rows(
                loss,
                y,
                &library_member(library, candidate)?.holdout_raw,
                weight,
                &eval,
            )?;
            if score < best_loss || (score == best_loss && candidate < best_seed) {
                best_loss = score;
                best_seed = candidate;
            }
        }
        let mut current = library_member(library, best_seed)?.holdout_raw.clone();
        let slot = counts.get_mut(best_seed).ok_or_else(|| PbError::Internal {
            what: "GreedySelect seed escaped counts".into(),
        })?;
        *slot = slot.checked_add(1).ok_or_else(|| PbError::Internal {
            what: "GreedySelect count overflow".into(),
        })?;
        for step in 1..library.len() {
            let denom = (step + 1) as f32;
            let prior = step as f32;
            let mut best_candidate = 0usize;
            let mut best_score = f64::INFINITY;
            let mut best_raw = Vec::new();
            for candidate in 0..library.len() {
                let cand_raw = &library_member(library, candidate)?.holdout_raw;
                let mixed = mix_raw(&current, prior, cand_raw, denom)?;
                let score = deviance_for_rows(loss, y, &mixed, weight, &eval)?;
                if score < best_score || (score == best_score && candidate < best_candidate) {
                    best_score = score;
                    best_candidate = candidate;
                    best_raw = mixed;
                }
            }
            current = best_raw;
            let slot = counts
                .get_mut(best_candidate)
                .ok_or_else(|| PbError::Internal {
                    what: "GreedySelect candidate escaped counts".into(),
                })?;
            *slot = slot.checked_add(1).ok_or_else(|| PbError::Internal {
                what: "GreedySelect count overflow".into(),
            })?;
        }
        let denom = library.len() as f64;
        for (total, count) in totals.iter_mut().zip(counts) {
            *total += f64::from(count) / denom;
        }
    }
    let bags = selection_bags as f64;
    for total in &mut totals {
        *total /= bags;
    }
    Ok(totals)
}

fn library_member(library: &[LibraryMember], idx: usize) -> Result<&LibraryMember, PbError> {
    library.get(idx).ok_or_else(|| PbError::Internal {
        what: "GreedySelect library index escaped".into(),
    })
}

fn bootstrap_indices(seed: u64, round: u32, n: usize) -> Result<Vec<usize>, PbError> {
    if n == 0 {
        return Err(PbError::InvalidInput {
            what: "GreedySelect holdout set must be non-empty".into(),
        });
    }
    let n_u64 = u64::try_from(n).map_err(|_| PbError::InvalidInput {
        what: "GreedySelect holdout size exceeded u64".into(),
    })?;
    let mut out = Vec::new();
    out.try_reserve_exact(n).map_err(|_| PbError::Internal {
        what: "GreedySelect bootstrap allocation failed".into(),
    })?;
    for i in 0..n {
        let block = u32::try_from(i).map_err(|_| PbError::InvalidInput {
            what: "GreedySelect bootstrap supports at most u32::MAX rows".into(),
        })?;
        let idx = pb_seed(seed, round, Stage::Sample as u32, block) % n_u64;
        out.push(usize::try_from(idx).map_err(|_| PbError::Internal {
            what: "GreedySelect bootstrap index exceeded usize".into(),
        })?);
    }
    Ok(out)
}

fn mix_raw(left: &[f32], left_scale: f32, right: &[f32], denom: f32) -> Result<Vec<f32>, PbError> {
    if left.len() != right.len() {
        return Err(PbError::ShapeMismatch {
            what: "GreedySelect raw vectors have different lengths".into(),
        });
    }
    let mut out = Vec::new();
    out.try_reserve_exact(left.len())
        .map_err(|_| PbError::Internal {
            what: "GreedySelect raw mix allocation failed".into(),
        })?;
    for (&a, &b) in left.iter().zip(right) {
        let value = (left_scale * a + b) / denom;
        if !value.is_finite() {
            return Err(PbError::InvalidInput {
                what: "GreedySelect mixed raw score is not finite".into(),
            });
        }
        out.push(value);
    }
    Ok(out)
}

/// Fit the honest affine link-scale recalibration `F' = b + a·F_model` on held-out rows by damped
/// Newton on this fit's own loss, from the identity start `(a, b) = (1, 0)`.
///
/// Motivation: shrinkage + bulk-dominated early stopping can leave the fitted score scale
/// uniformly compressed toward the mean (extremes under-separated) even when the aggregate level
/// is right. A two-parameter family can only express level (`b`) and scale (`a`), so — unlike a
/// tail-weighted stopping metric — the zero-heavy bulk cannot buy deviance by flattening the tail
/// (see speed_accuracy_work.md, 2026-07-10). Fitted on the internal validation holdout, so it is
/// honest to the training rows; `a > 0` preserves ranking and monotone constraints exactly, and
/// scaling purified tables by `a` preserves the fANOVA decomposition.
///
/// The fit is EVIDENCE-GATED: the Newton objective carries ridge priors `a ~ N(1, 0.1²)` and
/// (pivot-centered) level `~ N(0, 0.1²)` in holdout-deviance units, and the result is applied only
/// when its holdout deviance gain clears the penalty. Large holdouts don't feel the priors; thin
/// or uninformative ones (small portfolios, claim-only severity data) shrink to (near-)identity —
/// the unpenalized fit measurably hurt exactly those cases (2026-07-10 battery, 9 datasets).
///
/// Returns `None` (identity) when there is no holdout signal, the Newton system is degenerate,
/// the gated fit does not improve the penalized holdout objective, or the correction is
/// immaterial.
fn fit_affine_reanchor(
    loss: &dyn crate::loss::Loss,
    y: &[f32],
    raw: &[f32],
    weight: &[f32],
    offset: Option<&[f32]>,
    rows: &[usize],
) -> Result<Option<(f64, f64)>, PbError> {
    const MAX_ITER: usize = 25;
    const A_MIN: f64 = 0.25;
    const A_MAX: f64 = 4.0;
    const B_BOUND: f64 = 20.0;
    // Ridge evidence gate: prior scales for the slope (around 1) and the pivot-centered level
    // (around 0), as penalties in holdout-deviance units. 0.1 ⇒ a genuine ~5-10% scale compression
    // needs only a modest deviance gain to apply, while a spurious fit on a few hundred holdout
    // rows cannot clear it.
    const PRIOR_SCALE_A: f64 = 0.1;
    const PRIOR_SCALE_B: f64 = 0.1;
    if rows.is_empty() {
        return Ok(None);
    }
    let n = rows.len();
    let mut y_s = Vec::new();
    let mut w_s = Vec::new();
    let mut f_s: Vec<f64> = Vec::new();
    let mut off_s: Vec<f64> = Vec::new();
    for (buf, what) in [
        (&mut y_s, "affine reanchor y"),
        (&mut w_s, "affine reanchor weight"),
    ] {
        buf.try_reserve_exact(n).map_err(|_| PbError::Internal {
            what: format!("{what} allocation failed"),
        })?;
    }
    for (buf, what) in [
        (&mut f_s, "affine reanchor score"),
        (&mut off_s, "affine reanchor offset"),
    ] {
        buf.try_reserve_exact(n).map_err(|_| PbError::Internal {
            what: format!("{what} allocation failed"),
        })?;
    }
    for &r in rows {
        y_s.push(y[r]);
        w_s.push(weight[r]);
        let o = offset.map_or(0.0, |o| f64::from(o[r]));
        off_s.push(o);
        f_s.push(f64::from(raw[r]) - o);
    }
    let mut eta = vec![0.0_f32; n];
    for (e, &r) in eta.iter_mut().zip(rows) {
        *e = raw[r]; // identity scores are the raw values themselves
    }
    let dev0 = f64::from(loss.deviance(&y_s, &eta, &w_s)?);
    let mut gh = GradHess::default();
    // Information-weighted pivot: center the slope so `a` is pure scale and `b_c` pure level —
    // decoupled and well-conditioned, and the quantities the priors are meant to act on.
    loss.grad_hess(&y_s, &eta, &w_s, &mut gh)?;
    let mut h_sum = 0.0_f64;
    let mut hf_sum = 0.0_f64;
    for i in 0..n {
        let h = f64::from(gh.h[i]);
        h_sum += h;
        hf_sum += h * f_s[i];
    }
    if !(h_sum.is_finite() && h_sum > 0.0) {
        return Ok(None);
    }
    let fbar = hf_sum / h_sum;
    let f_c: Vec<f64> = f_s.iter().map(|f| f - fbar).collect();
    let lam_a = 1.0 / (PRIOR_SCALE_A * PRIOR_SCALE_A);
    let lam_b = 1.0 / (PRIOR_SCALE_B * PRIOR_SCALE_B);
    let pen = |a: f64, b_c: f64| (a - 1.0) * (a - 1.0) * lam_a + b_c * b_c * lam_b;
    let eval = |a: f64, b_c: f64, eta: &mut Vec<f32>| -> Result<f64, PbError> {
        for i in 0..n {
            eta[i] = (off_s[i] + fbar + b_c + a * f_c[i]) as f32;
        }
        Ok(f64::from(loss.deviance(&y_s, eta, &w_s)?))
    };
    let (mut a, mut b_c) = (1.0_f64, 0.0_f64);
    let mut obj = dev0; // identity objective: dev0 + pen(1, 0) = dev0
    for _ in 0..MAX_ITER {
        // `eta` holds the scores of the current accepted (a, b_c).
        loss.grad_hess(&y_s, &eta, &w_s, &mut gh)?;
        let (mut gb, mut ga) = (0.0_f64, 0.0_f64);
        let (mut hbb, mut hba, mut haa) = (0.0_f64, 0.0_f64, 0.0_f64);
        for i in 0..n {
            let g = f64::from(gh.g[i]);
            let h = f64::from(gh.h[i]);
            let f = f_c[i];
            gb += g;
            ga += g * f;
            hbb += h;
            hba += h * f;
            haa += h * f * f;
        }
        ga += 2.0 * (a - 1.0) * lam_a;
        gb += 2.0 * b_c * lam_b;
        haa += 2.0 * lam_a;
        hbb += 2.0 * lam_b;
        let det = hbb * haa - hba * hba;
        if !det.is_finite() || det <= 1e-12 * hbb.max(1.0) {
            break;
        }
        let mut db = (-gb * haa + ga * hba) / det;
        let mut da = (-ga * hbb + gb * hba) / det;
        if !db.is_finite() || !da.is_finite() {
            break;
        }
        // Damped step: halve until the penalized holdout objective does not get worse.
        let mut stepped = false;
        for _ in 0..8 {
            let cand = eval(a + da, b_c + db, &mut eta)? + pen(a + da, b_c + db);
            if cand.is_finite() && cand <= obj {
                a += da;
                b_c += db;
                obj = cand;
                stepped = true;
                break;
            }
            da *= 0.5;
            db *= 0.5;
        }
        if !stepped || da.abs() + db.abs() < 1e-9 {
            break;
        }
    }
    a = a.clamp(A_MIN, A_MAX);
    let b = b_c + fbar * (1.0 - a); // back to the caller's F' = b + a·F parameterization
    if !a.is_finite() || !b.is_finite() || b.abs() > B_BOUND {
        return Ok(None);
    }
    // The gate itself: apply only when the holdout deviance gain clears the prior penalty
    // (recomputed because clamping may have moved the point).
    if eval(a, b_c, &mut eta)? + pen(a, b_c) > dev0 {
        return Ok(None);
    }
    if (a - 1.0).abs() < 1e-3 && b.abs() < 1e-4 {
        return Ok(None); // immaterial — skip the tree-scaling work entirely
    }
    Ok(Some((a, b)))
}

fn deviance_for_rows(
    loss: &dyn crate::loss::Loss,
    y: &[f32],
    raw: &[f32],
    weight: &[f32],
    rows: &[usize],
) -> Result<f64, PbError> {
    let mut y_sub = Vec::new();
    let mut raw_sub = Vec::new();
    let mut weight_sub = Vec::new();
    deviance_for_rows_scratch(
        loss,
        y,
        raw,
        weight,
        rows,
        &mut y_sub,
        &mut raw_sub,
        &mut weight_sub,
    )
}

/// Like [`deviance_for_rows`], but gathers into CALLER-OWNED scratch buffers instead of
/// allocating three fresh `Vec`s every call. `fit_single`'s round loop calls this every round
/// against the SAME (unchanging) `rows` — the early-stopping holdout — so hoisting the buffers
/// once and reusing them here (`clear()` keeps their capacity; `try_reserve_exact` is then a
/// no-op after the first call) turns every round after the first into zero new allocations.
/// Byte-identical result to `deviance_for_rows` for the same inputs.
fn deviance_for_rows_scratch(
    loss: &dyn crate::loss::Loss,
    y: &[f32],
    raw: &[f32],
    weight: &[f32],
    rows: &[usize],
    y_sub: &mut Vec<f32>,
    raw_sub: &mut Vec<f32>,
    weight_sub: &mut Vec<f32>,
) -> Result<f64, PbError> {
    y_sub.clear();
    raw_sub.clear();
    weight_sub.clear();
    y_sub
        .try_reserve_exact(rows.len())
        .map_err(|_| PbError::Internal {
            what: "early-stop y eval allocation failed".into(),
        })?;
    raw_sub
        .try_reserve_exact(rows.len())
        .map_err(|_| PbError::Internal {
            what: "early-stop raw eval allocation failed".into(),
        })?;
    weight_sub
        .try_reserve_exact(rows.len())
        .map_err(|_| PbError::Internal {
            what: "early-stop weight eval allocation failed".into(),
        })?;
    for &row in rows {
        y_sub.push(*y.get(row).ok_or_else(|| PbError::Internal {
            what: "early-stop y eval row escaped".into(),
        })?);
        raw_sub.push(*raw.get(row).ok_or_else(|| PbError::Internal {
            what: "early-stop raw eval row escaped".into(),
        })?);
        weight_sub.push(*weight.get(row).ok_or_else(|| PbError::Internal {
            what: "early-stop weight eval row escaped".into(),
        })?);
    }
    Ok(f64::from(loss.deviance(y_sub, raw_sub, weight_sub)?))
}

fn soup_models(members: &[WeightedModel]) -> Result<Model, PbError> {
    let first = members.first().ok_or_else(|| PbError::InvalidConfig {
        what: "model soup requires at least one member".into(),
    })?;
    let mut alpha_sum = 0.0_f64;
    let mut f0 = 0.0_f64;
    let mut trees: Vec<(f32, ObliviousTree)> = Vec::new();
    let mut bag_spans: Vec<(u32, u32)> = Vec::with_capacity(members.len());
    for (idx, member) in members.iter().enumerate() {
        if !member.alpha.is_finite() || member.alpha < 0.0 {
            return Err(PbError::InvalidConfig {
                what: format!("model soup member {idx} alpha must be finite and >= 0"),
            });
        }
        validate_soup_member(&first.model, &member.model)?;
        alpha_sum += member.alpha;
        f0 += member.alpha * f64::from(member.model.f0);
        trees
            .try_reserve(member.model.trees.len())
            .map_err(|_| PbError::Internal {
                what: "model soup tree allocation failed".into(),
            })?;
        let span_start = u32::try_from(trees.len()).map_err(|_| PbError::Internal {
            what: "model soup tree count exceeded u32".into(),
        })?;
        for (tree_alpha, tree) in &member.model.trees {
            let scaled = member.alpha * f64::from(*tree_alpha);
            if scaled != 0.0 {
                if !scaled.is_finite()
                    || scaled < f64::from(f32::MIN)
                    || scaled > f64::from(f32::MAX)
                {
                    return Err(PbError::InvalidInput {
                        what: "model soup tree alpha is not representable as f32".into(),
                    });
                }
                trees.push((scaled as f32, tree.clone()));
            }
        }
        let span_end = u32::try_from(trees.len()).map_err(|_| PbError::Internal {
            what: "model soup tree count exceeded u32".into(),
        })?;
        bag_spans.push((span_start, span_end));
    }
    if (alpha_sum - 1.0).abs() > ENSEMBLE_WEIGHT_TOL {
        return Err(PbError::InvalidConfig {
            what: format!("model soup alphas must sum to 1.0, got {alpha_sum}"),
        });
    }
    if !f0.is_finite() || f0 < f64::from(f32::MIN) || f0 > f64::from(f32::MAX) {
        return Err(PbError::InvalidInput {
            what: "model soup intercept is not representable as f32".into(),
        });
    }
    let mut model = Model {
        f0: f0 as f32,
        trees,
        grids: first.model.grids.clone(),
        provenance: first.model.provenance.clone(),
        link: first.model.link,
        mode: ExactnessMode::Exact,
        schema: first.model.schema.clone(),
        // The soup's stamp is re-derived from the MERGED tree list below, not inherited
        // from the first bag: with min-version stamping (see `serialize::SCHEMA_VERSION`)
        // two bags of the same fit can legitimately carry different stamps, because a bag
        // that happened to terminate at depth <= 3 requires only the pre-lift schema while
        // one that grew deeper requires the newer one. The soup needs whichever is higher.
        schema_version: SCHEMA_VERSION_UNLIFTED,
        correction: None,
        bag_spans: Some(bag_spans),
        bag_intercepts: Some(members.iter().map(|m| m.model.f0).collect()),
        // Filled in by `fit_outer_bag` (the only caller with a row partition to record).
        bag_in_bag: None,
        // Each bag ran its own independent gate (its own bootstrap/subagged rows ⇒ its own
        // collapse behavior), so the soup's report is the fold of the members': engaged if ANY
        // bag engaged, earliest engaging round, and the extreme rate ratio over all of them.
        delta_step_gate: members
            .iter()
            .filter_map(|m| m.model.delta_step_gate)
            .reduce(DeltaStepGateReport::merge),
        // Each bag's own report, in bag order (`None` if any member carries none).
        fit_report: members
            .iter()
            .map(|m| m.model.fit_report.clone())
            .collect::<Option<Vec<_>>>()
            .map(|reports| reports.into_iter().flatten().collect()),
    };
    model.schema_version = model.required_schema_version();
    model.validate()?;
    Ok(model)
}

fn validate_soup_member(reference: &Model, member: &Model) -> Result<(), PbError> {
    if !matches!(member.mode, ExactnessMode::Exact) {
        return Err(PbError::ExactnessFirewall(
            "model soup accepts only Exact members".into(),
        ));
    }
    // NOTE the schema_version is deliberately NOT compared. It is a per-model MINIMUM
    // reader requirement derived from the model's own contents, not a property of the fit
    // (see `serialize::SCHEMA_VERSION`), so two bags of one lifted fit differ exactly when
    // one of them happened to terminate shallow. `soup_models` takes the max over members.
    // Everything below IS a genuine shape contract between members.
    if member.grids != reference.grids
        || member.provenance != reference.provenance
        || member.link != reference.link
        || member.schema.objective != reference.schema.objective
    {
        return Err(PbError::ShapeMismatch {
            what: "model soup members must share grids, provenance, link, and objective".into(),
        });
    }
    member.validate()
}

fn should_refit_after_round(refit: &RefitSpec, n_trees: usize) -> Result<bool, PbError> {
    match refit {
        RefitSpec::Ridge {
            every_k_trees: Some(k),
            ..
        } => {
            let k = usize::try_from(*k).map_err(|_| PbError::Internal {
                what: "refit every_k_trees exceeded usize".into(),
            })?;
            Ok(n_trees > 0 && n_trees % k == 0)
        }
        _ => Ok(false),
    }
}

fn should_refit_at_end(refit: &RefitSpec, n_trees: usize, last_refit_tree_count: usize) -> bool {
    matches!(refit, RefitSpec::Ridge { .. }) && n_trees > 0 && last_refit_tree_count != n_trees
}

fn agbm_beta(round: u32) -> f32 {
    let theta = 2.0_f32 / (round as f32 + 2.0);
    1.0 - theta
}

fn collect_tree_alphas(trees: &[(f32, ObliviousTree)]) -> Result<Vec<f32>, PbError> {
    let mut out = Vec::new();
    out.try_reserve_exact(trees.len())
        .map_err(|_| PbError::Internal {
            what: "AGBM alpha allocation failed".into(),
        })?;
    for (alpha, _) in trees {
        if !alpha.is_finite() {
            return Err(PbError::InvalidInput {
                what: "AGBM tree alpha must be finite".into(),
            });
        }
        out.push(*alpha);
    }
    Ok(out)
}

fn combine_alphas(current: &[f32], previous: &[f32], beta: f32) -> Result<Vec<f32>, PbError> {
    if previous.len() > current.len() {
        return Err(PbError::Internal {
            what: "AGBM previous alpha vector longer than current".into(),
        });
    }
    let mut out = Vec::new();
    out.try_reserve_exact(current.len())
        .map_err(|_| PbError::Internal {
            what: "AGBM combined alpha allocation failed".into(),
        })?;
    for (idx, &alpha) in current.iter().enumerate() {
        let prev = previous.get(idx).copied().unwrap_or(0.0);
        let value = (1.0 + beta) * alpha - beta * prev;
        if !value.is_finite() {
            return Err(PbError::InvalidInput {
                what: "AGBM combined alpha is not finite".into(),
            });
        }
        out.push(value);
    }
    Ok(out)
}

fn set_tree_alphas(trees: &mut [(f32, ObliviousTree)], alphas: &[f32]) -> Result<(), PbError> {
    if trees.len() != alphas.len() {
        return Err(PbError::ShapeMismatch {
            what: "AGBM alpha vector length does not match tree count".into(),
        });
    }
    for ((alpha_slot, _), &alpha) in trees.iter_mut().zip(alphas) {
        if !alpha.is_finite() {
            return Err(PbError::InvalidInput {
                what: "AGBM alpha must be finite".into(),
            });
        }
        *alpha_slot = alpha;
    }
    Ok(())
}

fn raw_from_tree_alphas(
    f0: f32,
    offset: Option<&[f32]>,
    x: &BinnedMatrix,
    trees: &[(f32, ObliviousTree)],
) -> Result<Vec<f32>, PbError> {
    raw_from_tree_alphas_kept(f0, offset, x, trees, None)
}

fn raw_from_tree_alphas_kept(
    f0: f32,
    offset: Option<&[f32]>,
    x: &BinnedMatrix,
    trees: &[(f32, ObliviousTree)],
    drops: Option<&[bool]>,
) -> Result<Vec<f32>, PbError> {
    Ok(raw64_from_tree_alphas_kept(f0, offset, x, trees, drops)?
        .into_iter()
        .map(|value| value as f32)
        .collect())
}

fn raw64_from_tree_alphas_kept(
    f0: f32,
    offset: Option<&[f32]>,
    x: &BinnedMatrix,
    trees: &[(f32, ObliviousTree)],
    drops: Option<&[bool]>,
) -> Result<Vec<f64>, PbError> {
    let n_rows = x.n_rows as usize;
    let mut out: Vec<f64> = crate::engine::Hist::try_zeroed_vec(n_rows, "ensemble raw")?;
    let tree_columns: Vec<Vec<&[u8]>> = trees
        .iter()
        .map(|(_, tree)| tree_split_columns(tree, &x.data))
        .collect::<Result<_, _>>()?;
    for row in 0..n_rows {
        let mut score = base_raw(offset, f0, row)?;
        for (index, ((alpha, tree), columns)) in trees.iter().zip(&tree_columns).enumerate() {
            if drops
                .and_then(|mask| mask.get(index))
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            score +=
                f64::from(*alpha) * f64::from(tree_value_for_row_with_columns(tree, columns, row)?);
        }
        if !score.is_finite() || score < f64::from(f32::MIN) || score > f64::from(f32::MAX) {
            return Err(PbError::InvalidInput {
                what: "AGBM raw score is not finite/representable as f32".into(),
            });
        }
        *out.get_mut(row).ok_or_else(|| PbError::Internal {
            what: "AGBM raw write escaped".into(),
        })? = score;
    }
    Ok(out)
}

/// §C3 incremental AGBM lookahead raw — an O(n) alternative to a `raw_from_tree_alphas` O(n·T)
/// re-walk. Derivation: `combine_alphas` sets `lookahead[i] = (1+beta)*current[i] - beta*previous[i]`
/// (§ combine_alphas). `raw_from_tree_alphas(f0, offset, x, trees)` with `trees` at `lookahead` is
/// therefore, by linearity of the per-row sum over trees,
///   f0+offset + Σ_i lookahead[i]*tree_i = (1+beta)·(f0+offset + Σ_i current[i]*tree_i)
///                                          - beta·(f0+offset + Σ_i previous[i]*tree_i)
/// (the extra `f0+offset` term introduced by distributing `(1+beta)` and `-beta` across it cancels:
/// `(1+beta) - beta == 1`). `raw` is kept in sync with the trees' actual stored alphas at every
/// round boundary — i.e. `raw == f0+offset+Σ_i current[i]*tree_i` exactly at the top of each round
/// (an invariant maintained by construction: every place `raw` is written — the plain `update_raw`
/// path, `raw_plus_dart_round`, and this function's own caller — writes the ensemble's actual score
/// at its actual current alphas). `raw_prev` is maintained to be one round-boundary behind `raw` in
/// exactly the same way (see the capture at the `prev_alphas = current_alphas` call site), so
/// `raw_prev == f0+offset+Σ_i previous[i]*tree_i`. Substituting both collapses the walk to:
///     fit_raw[row] = (1+beta)*raw[row] - beta*raw_prev[row]
/// Pinned against a full `raw_from_tree_alphas` re-walk by
/// `agbm_incremental_lookahead_matches_full_realphas_walk` (test module, below). Reorders the f32
/// accumulation vs the old full walk (a sum over 2 terms instead of over T tree contributions), so
/// AGBM's fitted numbers move relative to the prior implementation — see `NesterovSpec::Agbm`'s doc;
/// this function does not claim bit-identity with `raw_from_tree_alphas`, only algebraic equivalence.
#[cfg(test)]
fn agbm_lookahead_raw(raw: &[f32], raw_prev: &[f32], beta: f32) -> Result<Vec<f32>, PbError> {
    if raw.len() != raw_prev.len() {
        return Err(PbError::ShapeMismatch {
            what: "AGBM raw_prev length does not match raw".into(),
        });
    }
    let beta = f64::from(beta);
    let mut out = Vec::new();
    out.try_reserve_exact(raw.len())
        .map_err(|_| PbError::Internal {
            what: "AGBM lookahead raw allocation failed".into(),
        })?;
    for (current, prev) in raw.iter().zip(raw_prev) {
        let score = (1.0 + beta) * f64::from(*current) - beta * f64::from(*prev);
        if !score.is_finite() || score < f64::from(f32::MIN) || score > f64::from(f32::MAX) {
            return Err(PbError::InvalidInput {
                what: "AGBM lookahead raw is not finite/representable as f32".into(),
            });
        }
        out.push(score as f32);
    }
    Ok(out)
}

fn dart_drop_mask(
    dart: Option<&DartSpec>,
    seed: u64,
    round: u32,
    n_trees: usize,
) -> Result<Vec<bool>, PbError> {
    let Some(dart) = dart else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    out.try_reserve_exact(n_trees)
        .map_err(|_| PbError::Internal {
            what: "DART mask allocation failed".into(),
        })?;
    for tree_idx in 0..n_trees {
        let block = u32::try_from(tree_idx).map_err(|_| PbError::InvalidInput {
            what: "DART supports at most u32::MAX trees".into(),
        })?;
        let bits = pb_seed(seed, round, Stage::Dart as u32, block);
        let unit = ((bits >> 11) as f64 + 1.0) / ((1_u64 << 53) as f64 + 1.0);
        out.push(unit < f64::from(dart.drop_rate));
    }
    Ok(out)
}

#[cfg(test)]
fn raw_minus_dropped(
    raw: &[f32],
    x: &BinnedMatrix,
    trees: &[(f32, ObliviousTree)],
    drops: &[bool],
) -> Result<Vec<f32>, PbError> {
    if trees.len() != drops.len() {
        return Err(PbError::ShapeMismatch {
            what: "DART drop mask length does not match tree count".into(),
        });
    }
    let mut out = Vec::new();
    out.try_reserve_exact(raw.len())
        .map_err(|_| PbError::Internal {
            what: "DART raw allocation failed".into(),
        })?;
    out.extend_from_slice(raw);
    let dropped_trees: Vec<(f32, &ObliviousTree, Vec<&[u8]>)> = trees
        .iter()
        .zip(drops)
        .filter(|(_, dropped)| **dropped)
        .map(|((alpha, tree), _)| Ok((*alpha, tree, tree_split_columns(tree, &x.data)?)))
        .collect::<Result<_, PbError>>()?;
    for row in 0..out.len() {
        let mut score = f64::from(*out.get(row).ok_or_else(|| PbError::Internal {
            what: "DART raw row escaped".into(),
        })?);
        for (alpha, tree, columns) in &dropped_trees {
            score -=
                f64::from(*alpha) * f64::from(tree_value_for_row_with_columns(tree, columns, row)?);
        }
        if !score.is_finite() || score < f64::from(f32::MIN) || score > f64::from(f32::MAX) {
            return Err(PbError::InvalidInput {
                what: "DART dropout raw is not finite/representable as f32".into(),
            });
        }
        *out.get_mut(row).ok_or_else(|| PbError::Internal {
            what: "DART raw write escaped".into(),
        })? = score as f32;
    }
    Ok(out)
}

/// §C3 incremental DART post-round raw reconstruction. `raw_acc` (`fit_raw`) already excludes
/// the dropped trees' OLD contribution (`raw_minus_dropped`, above, run earlier this round);
/// `apply_dart_normalization` has already rescaled `trees[i].alpha` for each dropped `i` to its
/// POST-normalization value by the time this is called. Adding the dropped trees' contribution
/// back at that (already-current) alpha, plus the new tree's own contribution at `new_alpha`, in
/// one walk touching only `#dropped + 1` trees, reconstructs the same raw `raw_from_tree_alphas`
/// would — its O(n·T) full re-walk of every tree in the ensemble — in O(n·(#dropped+1)) instead.
/// Takes `raw_acc` by value and mutates it in place (it is the round's scratch `fit_raw`, not
/// read again after this call), so no extra allocation is needed beyond the caller's existing
/// clone. Changes DART's f32 accumulation order vs the full re-walk — see `DartSpec`'s doc.
fn raw_plus_dart_round(
    mut raw_acc: Vec<f32>,
    x: &BinnedMatrix,
    trees: &[(f32, ObliviousTree)],
    drops: &[bool],
    new_tree: &ObliviousTree,
    new_alpha: f32,
) -> Result<Vec<f32>, PbError> {
    if trees.len() != drops.len() {
        return Err(PbError::ShapeMismatch {
            what: "DART drop mask length does not match tree count".into(),
        });
    }
    let mut walk: Vec<(f32, &ObliviousTree, Vec<&[u8]>)> = trees
        .iter()
        .zip(drops)
        .filter(|(_, dropped)| **dropped)
        .map(|((alpha, tree), _)| Ok((*alpha, tree, tree_split_columns(tree, &x.data)?)))
        .collect::<Result<_, PbError>>()?;
    walk.push((new_alpha, new_tree, tree_split_columns(new_tree, &x.data)?));
    for row in 0..raw_acc.len() {
        let mut score = f64::from(*raw_acc.get(row).ok_or_else(|| PbError::Internal {
            what: "DART incremental raw row escaped".into(),
        })?);
        for (alpha, tree, columns) in &walk {
            score +=
                f64::from(*alpha) * f64::from(tree_value_for_row_with_columns(tree, columns, row)?);
        }
        if !score.is_finite() || score < f64::from(f32::MIN) || score > f64::from(f32::MAX) {
            return Err(PbError::InvalidInput {
                what: "DART incremental raw is not finite/representable as f32".into(),
            });
        }
        *raw_acc.get_mut(row).ok_or_else(|| PbError::Internal {
            what: "DART incremental raw write escaped".into(),
        })? = score as f32;
    }
    Ok(raw_acc)
}

fn apply_dart_normalization(
    trees: &mut [(f32, ObliviousTree)],
    drops: &[bool],
    dart: &DartSpec,
) -> Result<f32, PbError> {
    if trees.len() != drops.len() {
        return Err(PbError::ShapeMismatch {
            what: "DART normalization mask length does not match tree count".into(),
        });
    }
    if !dart.normalize {
        return Ok(1.0);
    }
    let dropped = drops.iter().filter(|drop| **drop).count();
    if dropped == 0 {
        return Ok(1.0);
    }
    let denom = dropped.checked_add(1).ok_or_else(|| PbError::Internal {
        what: "DART dropped count overflow".into(),
    })?;
    let dropped_scale = dropped as f32 / denom as f32;
    let new_alpha = 1.0_f32 / denom as f32;
    for ((alpha, _), drop) in trees.iter_mut().zip(drops) {
        if *drop {
            *alpha *= dropped_scale;
            if !alpha.is_finite() {
                return Err(PbError::InvalidInput {
                    what: "DART normalized alpha is not finite".into(),
                });
            }
        }
    }
    Ok(new_alpha)
}

struct RefitProblem<'a, 's> {
    spec: &'a FitSpec<'s>,
    x: &'a BinnedMatrix,
    y: &'a [f32],
    weight: &'a [f32],
    offset: Option<&'a [f32]>,
    f0: f32,
    /// Resolved per-axis monotone signs (§07.5); the ridge solve is unconstrained, so the
    /// final leaves are projected onto the monotone cone and `raw` re-derived to match.
    monotone: Option<&'a [Option<MonoSign>]>,
    /// Rows the refit is allowed to FIT on (excludes the early-stopping validation carve, if
    /// any). `refit_normal_equations`'s accumulation and the backtracking accept/reject deviance
    /// are both restricted to these — otherwise the jointly-solved leaf values would be fit
    /// partly to the validation rows' own targets. `raw` is still refreshed over every row after
    /// each accepted step (that is scoring, not fitting) — see `raw_from_theta`.
    train_rows: &'a [usize],
}

fn fully_corrective_refit(
    refit: &RefitSpec,
    problem: &RefitProblem<'_, '_>,
    trees: &mut [(f32, ObliviousTree)],
    raw: &mut [f32],
) -> Result<(), PbError> {
    let RefitSpec::Ridge { l2, max_iter, .. } = refit else {
        return Ok(());
    };
    if trees.is_empty() {
        return Ok(());
    }
    let memberships = leaf_memberships(problem.x, trees)?;
    let n_trees = trees.len();
    let n_cols = n_trees.checked_mul(8).ok_or_else(|| PbError::Internal {
        what: "refit column count overflow".into(),
    })?;
    let mut gh = GradHess::default();
    for _ in 0..*max_iter {
        problem
            .spec
            .loss
            .grad_hess(problem.y, raw, problem.weight, &mut gh)?;
        let (normal, rhs) = refit_normal_equations(
            problem,
            &gh,
            raw,
            trees,
            &memberships,
            n_cols,
            f64::from(*l2),
        )?;
        let target = solve_refit_system(&normal, &rhs, n_cols)?;
        let current = collect_leaf_theta(trees, n_cols)?;
        // Both the baseline and every backtracking candidate are scored on `train_rows` ONLY
        // (§H6): the accept/reject step-size decision must not be shaped by the validation
        // rows' own targets, or the leakage `refit_normal_equations` avoids on the fit side
        // would simply re-enter through the line search.
        let current_deviance = deviance_for_rows(
            problem.spec.loss,
            problem.y,
            raw,
            problem.weight,
            problem.train_rows,
        )?;
        let mut step = 1.0_f64;
        let mut accepted: Option<(Vec<f64>, Vec<f32>, f64)> = None;
        for _ in 0..REFIT_MAX_BACKTRACKS {
            let candidate_theta = interpolate_theta(&current, &target, step)?;
            let candidate_raw = raw_from_theta(
                problem.offset,
                problem.f0,
                trees,
                &memberships,
                &candidate_theta,
            )?;
            let deviance = deviance_for_rows(
                problem.spec.loss,
                problem.y,
                &candidate_raw,
                problem.weight,
                problem.train_rows,
            )?;
            if deviance.is_finite()
                && deviance <= current_deviance + REFIT_ACCEPT_TOL * (1.0 + current_deviance.abs())
            {
                accepted = Some((candidate_theta, candidate_raw, deviance));
                break;
            }
            step *= 0.5;
        }
        let Some((theta, candidate_raw, accepted_deviance)) = accepted else {
            break;
        };
        write_leaf_theta(trees, &theta)?;
        if raw.len() != candidate_raw.len() {
            return Err(PbError::Internal {
                what: "refit raw length changed".into(),
            });
        }
        for (dst, src) in raw.iter_mut().zip(candidate_raw) {
            *dst = src;
        }
        if (current_deviance - accepted_deviance).abs()
            <= REFIT_ACCEPT_TOL * (1.0 + current_deviance.abs())
        {
            break;
        }
    }
    // The ridge solve is UNCONSTRAINED, so a monotone constraint can be inverted by the
    // refit. Project each tree's leaves back onto the monotone cone (§07.5) and re-derive
    // `raw` from the clamped leaves so the served model and the next round's gradients
    // stay consistent (the projection trades a little deviance for the guarantee).
    if let Some(signs) = problem.monotone {
        let mut clamped = false;
        for (_, tree) in trees.iter_mut() {
            let before = tree.leaves.clone();
            clamp_monotone(
                &mut tree.leaves,
                &tree.splits,
                usize::from(tree.depth),
                Some(signs),
            )?;
            if tree.leaves != before {
                clamped = true;
            }
        }
        if clamped {
            let theta = collect_leaf_theta(trees, n_cols)?;
            let new_raw = raw_from_theta(problem.offset, problem.f0, trees, &memberships, &theta)?;
            if raw.len() != new_raw.len() {
                return Err(PbError::Internal {
                    what: "refit raw length changed after monotone clamp".into(),
                });
            }
            raw.copy_from_slice(&new_raw);
        }
    }
    Ok(())
}

fn leaf_memberships(x: &BinnedMatrix, trees: &[(f32, ObliviousTree)]) -> Result<Vec<u8>, PbError> {
    let n_rows = x.n_rows as usize;
    let n_trees = trees.len();
    let cells = n_rows
        .checked_mul(n_trees)
        .ok_or_else(|| PbError::Internal {
            what: "leaf membership shape overflow".into(),
        })?;
    let mut memberships: Vec<u8> = crate::engine::Hist::try_zeroed_vec(cells, "leaf membership")?;
    let tree_columns: Vec<Vec<&[u8]>> = trees
        .iter()
        .map(|(_, tree)| tree_split_columns(tree, &x.data))
        .collect::<Result<_, _>>()?;
    for row in 0..n_rows {
        for (tree_idx, ((_, tree), columns)) in trees.iter().zip(&tree_columns).enumerate() {
            let leaf = u8::try_from(tree_leaf_index_for_row_with_columns(tree, columns, row)?)
                .map_err(|_| PbError::Internal {
                    what: "leaf index exceeded u8".into(),
                })?;
            let slot = membership_offset(row, tree_idx, n_trees)?;
            *memberships.get_mut(slot).ok_or_else(|| PbError::Internal {
                what: "leaf membership offset escaped".into(),
            })? = leaf;
        }
    }
    Ok(memberships)
}

fn membership_offset(row: usize, tree_idx: usize, n_trees: usize) -> Result<usize, PbError> {
    row.checked_mul(n_trees)
        .and_then(|o| o.checked_add(tree_idx))
        .ok_or_else(|| PbError::Internal {
            what: "leaf membership offset overflow".into(),
        })
}

fn refit_col(tree_idx: usize, leaf: usize) -> Result<usize, PbError> {
    tree_idx
        .checked_mul(8)
        .and_then(|o| o.checked_add(leaf))
        .ok_or_else(|| PbError::Internal {
            what: "refit column offset overflow".into(),
        })
}

fn leaf_from_membership(
    memberships: &[u8],
    row: usize,
    tree_idx: usize,
    n_trees: usize,
) -> Result<usize, PbError> {
    let offset = membership_offset(row, tree_idx, n_trees)?;
    let leaf = usize::from(*memberships.get(offset).ok_or_else(|| PbError::Internal {
        what: "leaf membership lookup escaped".into(),
    })?);
    // Bounded against the LEGACY leaf count, not `MAX_LEAVES`, and that is deliberate.
    //
    // Every consumer of this value indexes the ridge design through `refit_col`, whose column
    // stride is a hard-coded `8` (`LEGACY_LEAVES`). The fit-time guard above refuses
    // `ridge_refit_l2` with `max_depth > LEGACY_MAX_DEPTH`, so a wider leaf id cannot arrive
    // here today — but `MAX_LEAVES` is now 256, so bounding against it would let a leaf id of
    // 8..255 through the moment that guard is relaxed, and `refit_col` would then ALIAS
    // columns across trees rather than fail. Silent aliasing in a normal-equations solve is
    // undetectable downstream: it produces a plausible refit of the wrong model.
    //
    // Bound at the stride the caller actually assumes. When P-D4's prefix-sum offset lands,
    // this bound moves with it, in the same commit, by construction.
    if leaf >= crate::engine::LEGACY_LEAVES {
        return Err(PbError::Internal {
            what: format!(
                "leaf membership value {leaf} escaped 0..{} — the ridge refit's fixed \
                 {}-column-per-tree stride cannot address it",
                crate::engine::LEGACY_LEAVES,
                crate::engine::LEGACY_LEAVES
            ),
        });
    }
    Ok(leaf)
}

fn base_raw(offset: Option<&[f32]>, f0: f32, row: usize) -> Result<f64, PbError> {
    let mut out = f64::from(f0);
    if let Some(off) = offset {
        out += f64::from(*off.get(row).ok_or_else(|| PbError::Internal {
            what: "refit offset row escaped".into(),
        })?);
    }
    Ok(out)
}

fn refit_normal_equations(
    problem: &RefitProblem<'_, '_>,
    gh: &GradHess,
    raw: &[f32],
    trees: &[(f32, ObliviousTree)],
    memberships: &[u8],
    n_cols: usize,
    l2: f64,
) -> Result<(Vec<f64>, Vec<f64>), PbError> {
    let n_rows = raw.len();
    if gh.g.len() != n_rows || gh.h.len() != n_rows {
        return Err(PbError::ShapeMismatch {
            what: "refit GradHess length does not match raw".into(),
        });
    }
    let cells = n_cols
        .checked_mul(n_cols)
        .ok_or_else(|| PbError::Internal {
            what: "refit normal matrix shape overflow".into(),
        })?;
    let mut normal: Vec<f64> = crate::engine::Hist::try_zeroed_vec(cells, "refit normal matrix")?;
    let mut rhs: Vec<f64> = crate::engine::Hist::try_zeroed_vec(n_cols, "refit rhs")?;
    let n_trees = trees.len();
    // §H6: accumulate the normal equations over `train_rows` ONLY — the early-stopping
    // validation carve (if any) must never shape the jointly-solved leaf values.
    for &row in problem.train_rows {
        let g = f64::from(*gh.g.get(row).ok_or_else(|| PbError::Internal {
            what: "refit gradient row escaped".into(),
        })?);
        let h = f64::from(*gh.h.get(row).ok_or_else(|| PbError::Internal {
            what: "refit hessian row escaped".into(),
        })?);
        if !g.is_finite() || !h.is_finite() {
            return Err(PbError::InvalidInput {
                what: "refit gradients must be finite".into(),
            });
        }
        if h <= 0.0 {
            continue;
        }
        let z_centered = f64::from(*raw.get(row).ok_or_else(|| PbError::Internal {
            what: "refit raw row escaped".into(),
        })?) - g / h
            - base_raw(problem.offset, problem.f0, row)?;
        for (a_idx, (alpha_a, _)) in trees.iter().enumerate() {
            let alpha_a = f64::from(*alpha_a);
            if !alpha_a.is_finite() {
                return Err(PbError::InvalidInput {
                    what: "refit tree alpha must be finite".into(),
                });
            }
            let col_a = refit_col(
                a_idx,
                leaf_from_membership(memberships, row, a_idx, n_trees)?,
            )?;
            add_vec(&mut rhs, col_a, h * alpha_a * z_centered)?;
            for (b_idx, (alpha_b, _)) in trees.iter().enumerate() {
                let alpha_b = f64::from(*alpha_b);
                if !alpha_b.is_finite() {
                    return Err(PbError::InvalidInput {
                        what: "refit tree alpha must be finite".into(),
                    });
                }
                let col_b = refit_col(
                    b_idx,
                    leaf_from_membership(memberships, row, b_idx, n_trees)?,
                )?;
                add_matrix(&mut normal, n_cols, col_a, col_b, h * alpha_a * alpha_b)?;
            }
        }
    }
    for col in 0..n_cols {
        add_matrix(&mut normal, n_cols, col, col, l2)?;
    }
    Ok((normal, rhs))
}

fn raw_from_theta(
    offset: Option<&[f32]>,
    f0: f32,
    trees: &[(f32, ObliviousTree)],
    memberships: &[u8],
    theta: &[f64],
) -> Result<Vec<f32>, PbError> {
    let n_trees = trees.len();
    let n_rows = if n_trees == 0 {
        0
    } else {
        memberships
            .len()
            .checked_div(n_trees)
            .ok_or_else(|| PbError::Internal {
                what: "refit membership row count overflow".into(),
            })?
    };
    let mut out: Vec<f32> = crate::engine::Hist::try_zeroed_vec(n_rows, "refit raw")?;
    for row in 0..n_rows {
        let mut score = base_raw(offset, f0, row)?;
        for (tree_idx, (alpha, _)) in trees.iter().enumerate() {
            let leaf = leaf_from_membership(memberships, row, tree_idx, n_trees)?;
            let col = refit_col(tree_idx, leaf)?;
            score += f64::from(*alpha)
                * *theta.get(col).ok_or_else(|| PbError::Internal {
                    what: "refit theta lookup escaped".into(),
                })?;
        }
        if !score.is_finite() || score < f64::from(f32::MIN) || score > f64::from(f32::MAX) {
            return Err(PbError::InvalidInput {
                what: "refit raw score is not finite/representable as f32".into(),
            });
        }
        *out.get_mut(row).ok_or_else(|| PbError::Internal {
            what: "refit raw write escaped".into(),
        })? = score as f32;
    }
    Ok(out)
}

fn collect_leaf_theta(trees: &[(f32, ObliviousTree)], n_cols: usize) -> Result<Vec<f64>, PbError> {
    let mut theta: Vec<f64> = Vec::new();
    theta
        .try_reserve_exact(n_cols)
        .map_err(|_| PbError::Internal {
            what: "refit theta allocation failed".into(),
        })?;
    for (_, tree) in trees {
        for &leaf in &tree.leaves {
            theta.push(f64::from(leaf));
        }
    }
    if theta.len() != n_cols {
        return Err(PbError::Internal {
            what: "refit theta length mismatch".into(),
        });
    }
    Ok(theta)
}

fn interpolate_theta(current: &[f64], target: &[f64], step: f64) -> Result<Vec<f64>, PbError> {
    if current.len() != target.len() {
        return Err(PbError::ShapeMismatch {
            what: "refit theta interpolation length mismatch".into(),
        });
    }
    let mut out: Vec<f64> = Vec::new();
    out.try_reserve_exact(current.len())
        .map_err(|_| PbError::Internal {
            what: "refit theta interpolation allocation failed".into(),
        })?;
    for (&a, &b) in current.iter().zip(target) {
        out.push(a + step * (b - a));
    }
    Ok(out)
}

fn write_leaf_theta(trees: &mut [(f32, ObliviousTree)], theta: &[f64]) -> Result<(), PbError> {
    for (tree_idx, (_, tree)) in trees.iter_mut().enumerate() {
        for leaf in 0..8usize {
            let col = refit_col(tree_idx, leaf)?;
            let value = *theta.get(col).ok_or_else(|| PbError::Internal {
                what: "refit theta write lookup escaped".into(),
            })?;
            if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
                return Err(PbError::InvalidInput {
                    what: "refit leaf value is not finite/representable as f32".into(),
                });
            }
            *tree.leaves.get_mut(leaf).ok_or_else(|| PbError::Internal {
                what: "refit leaf write escaped".into(),
            })? = value as f32;
        }
    }
    Ok(())
}

fn matrix_offset(n: usize, row: usize, col: usize) -> Result<usize, PbError> {
    if row >= n || col >= n {
        return Err(PbError::Internal {
            what: "matrix coordinate out of range".into(),
        });
    }
    row.checked_mul(n)
        .and_then(|o| o.checked_add(col))
        .ok_or_else(|| PbError::Internal {
            what: "matrix offset overflow".into(),
        })
}

fn matrix_get(a: &[f64], n: usize, row: usize, col: usize) -> Result<f64, PbError> {
    let offset = matrix_offset(n, row, col)?;
    a.get(offset).copied().ok_or_else(|| PbError::Internal {
        what: "matrix lookup escaped".into(),
    })
}

fn matrix_set(a: &mut [f64], n: usize, row: usize, col: usize, value: f64) -> Result<(), PbError> {
    let offset = matrix_offset(n, row, col)?;
    *a.get_mut(offset).ok_or_else(|| PbError::Internal {
        what: "matrix write escaped".into(),
    })? = value;
    Ok(())
}

fn add_matrix(a: &mut [f64], n: usize, row: usize, col: usize, delta: f64) -> Result<(), PbError> {
    let offset = matrix_offset(n, row, col)?;
    let slot = a.get_mut(offset).ok_or_else(|| PbError::Internal {
        what: "matrix add escaped".into(),
    })?;
    *slot += delta;
    Ok(())
}

fn add_vec(v: &mut [f64], idx: usize, delta: f64) -> Result<(), PbError> {
    let slot = v.get_mut(idx).ok_or_else(|| PbError::Internal {
        what: "vector add escaped".into(),
    })?;
    *slot += delta;
    Ok(())
}

fn clone_f64_slice(input: &[f64], what: &'static str) -> Result<Vec<f64>, PbError> {
    let mut out = Vec::new();
    out.try_reserve_exact(input.len())
        .map_err(|_| PbError::Internal {
            what: format!("{what} allocation failed"),
        })?;
    out.extend_from_slice(input);
    Ok(out)
}

fn solve_refit_system(normal: &[f64], rhs: &[f64], n: usize) -> Result<Vec<f64>, PbError> {
    if rhs.len() != n {
        return Err(PbError::ShapeMismatch {
            what: "refit rhs length mismatch".into(),
        });
    }
    if normal.len()
        != n.checked_mul(n).ok_or_else(|| PbError::Internal {
            what: "refit solve shape overflow".into(),
        })?
    {
        return Err(PbError::ShapeMismatch {
            what: "refit normal matrix length mismatch".into(),
        });
    }
    let mut diag_scale = 1.0_f64;
    for i in 0..n {
        diag_scale = diag_scale.max(matrix_get(normal, n, i, i)?.abs());
    }
    for jitter in REFIT_CHOLESKY_JITTERS {
        let mut a = clone_f64_slice(normal, "refit solve matrix")?;
        if jitter > 0.0 {
            for i in 0..n {
                add_matrix(&mut a, n, i, i, jitter * diag_scale)?;
            }
        }
        if let Some(solution) = cholesky_solve(&a, rhs, n)? {
            return Ok(solution);
        }
    }
    Err(PbError::InvalidInput {
        what: "refit normal equations are not positive definite".into(),
    })
}

fn cholesky_solve(a: &[f64], rhs: &[f64], n: usize) -> Result<Option<Vec<f64>>, PbError> {
    let cells = n.checked_mul(n).ok_or_else(|| PbError::Internal {
        what: "Cholesky matrix shape overflow".into(),
    })?;
    let mut l: Vec<f64> = crate::engine::Hist::try_zeroed_vec(cells, "Cholesky factor")?;
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix_get(a, n, i, j)?;
            for k in 0..j {
                sum -= matrix_get(&l, n, i, k)? * matrix_get(&l, n, j, k)?;
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return Ok(None);
                }
                matrix_set(&mut l, n, i, j, sum.sqrt())?;
            } else {
                let diag = matrix_get(&l, n, j, j)?;
                if diag <= 0.0 || !diag.is_finite() {
                    return Ok(None);
                }
                matrix_set(&mut l, n, i, j, sum / diag)?;
            }
        }
    }

    let mut y: Vec<f64> = crate::engine::Hist::try_zeroed_vec(n, "Cholesky forward solve")?;
    for i in 0..n {
        let mut sum = *rhs.get(i).ok_or_else(|| PbError::Internal {
            what: "Cholesky rhs lookup escaped".into(),
        })?;
        for k in 0..i {
            sum -= matrix_get(&l, n, i, k)?
                * *y.get(k).ok_or_else(|| PbError::Internal {
                    what: "Cholesky y lookup escaped".into(),
                })?;
        }
        let diag = matrix_get(&l, n, i, i)?;
        if diag <= 0.0 || !diag.is_finite() {
            return Ok(None);
        }
        *y.get_mut(i).ok_or_else(|| PbError::Internal {
            what: "Cholesky y write escaped".into(),
        })? = sum / diag;
    }

    let mut x: Vec<f64> = crate::engine::Hist::try_zeroed_vec(n, "Cholesky back solve")?;
    for i in (0..n).rev() {
        let mut sum = *y.get(i).ok_or_else(|| PbError::Internal {
            what: "Cholesky y back lookup escaped".into(),
        })?;
        for k in (i + 1)..n {
            sum -= matrix_get(&l, n, k, i)?
                * *x.get(k).ok_or_else(|| PbError::Internal {
                    what: "Cholesky x lookup escaped".into(),
                })?;
        }
        let diag = matrix_get(&l, n, i, i)?;
        if diag <= 0.0 || !diag.is_finite() {
            return Ok(None);
        }
        *x.get_mut(i).ok_or_else(|| PbError::Internal {
            what: "Cholesky x write escaped".into(),
        })? = sum / diag;
    }
    Ok(Some(x))
}

fn inverse_link_f64(link: Link, raw: f64) -> f64 {
    match link {
        Link::Identity => raw,
        Link::Log => raw.clamp(-30.0, 30.0).exp(),
        Link::Logit => {
            if raw >= 0.0 {
                let z = (-raw).clamp(-30.0, 30.0).exp();
                1.0 / (1.0 + z)
            } else {
                let z = raw.clamp(-30.0, 30.0).exp();
                z / (1.0 + z)
            }
        }
    }
}

fn weighted_response_total(link: Link, raw: &[f32], offset: Option<&[f32]>, weight: &[f32]) -> f64 {
    let mut total = 0.0_f64;
    for (i, (&v, &w)) in raw.iter().zip(weight).enumerate() {
        let o = offset.map_or(0.0, |off| f64::from(off[i]));
        total += f64::from(w) * inverse_link_f64(link, f64::from(v) + o);
    }
    total
}

fn weighted_observed_total(y: &[f32], weight: &[f32]) -> f64 {
    y.iter()
        .zip(weight)
        .map(|(&yi, &wi)| f64::from(wi) * f64::from(yi))
        .sum()
}

/// Link-generic post-hoc intercept shift `δ` solving the aggregate-balance equation
/// `Σw·inverse_link(raw[i] + offset[i] + δ) == Σwy` (spec §06.6's reanchor). `raw` is the
/// model's raw score EXCLUDING `offset` — every branch folds `offset[i]` in per row before
/// applying the link, so a caller whose `raw` already includes the offset MUST pass `None`
/// here (folding it twice would double-count). `Link::Log` has a closed form (the exponential's
/// multiplicative structure lets `δ` factor out); `Link::Logit` has none, so it bisects.
///
/// `pub(crate)`: also called from [`crate::prune`]'s post-prune Logit reanchor, which has no
/// closed form of its own to fall back on (see `prune::reanchor_logit_link`).
pub(crate) fn reanchor_delta(
    link: Link,
    y: &[f32],
    weight: &[f32],
    raw: &[f32],
    offset: Option<&[f32]>,
) -> Result<f64, PbError> {
    let sum_w: f64 = weight.iter().map(|&w| f64::from(w)).sum();
    if sum_w <= 0.0 || !sum_w.is_finite() {
        return Err(PbError::InvalidInput {
            what: "reanchor requires positive finite total weight".into(),
        });
    }
    let observed = weighted_observed_total(y, weight);
    if !observed.is_finite() {
        return Err(PbError::InvalidInput {
            what: "reanchor observed total is not finite".into(),
        });
    }
    match link {
        Link::Identity => {
            let predicted = raw
                .iter()
                .zip(weight)
                .enumerate()
                .map(|(i, (&ri, &wi))| {
                    let o = offset.map_or(0.0, |off| f64::from(off[i]));
                    f64::from(wi) * (f64::from(ri) + o)
                })
                .sum::<f64>();
            Ok((observed - predicted) / sum_w)
        }
        Link::Log => {
            if observed <= 0.0 {
                return Err(PbError::InvalidInput {
                    what: "log-link reanchor requires positive observed total".into(),
                });
            }
            let predicted = weighted_response_total(link, raw, offset, weight);
            if predicted <= 0.0 || !predicted.is_finite() {
                return Err(PbError::InvalidInput {
                    what: "log-link reanchor predicted total must be positive and finite".into(),
                });
            }
            Ok((observed / predicted).ln())
        }
        Link::Logit => {
            if observed <= 0.0 || observed >= sum_w {
                return Err(PbError::InvalidInput {
                    what: "logit-link reanchor requires observed positives strictly inside (0, total_weight)".into(),
                });
            }
            let mut lo = -60.0_f64;
            let mut hi = 60.0_f64;
            for _ in 0..96 {
                let mid = 0.5 * (lo + hi);
                let mut predicted = 0.0_f64;
                for (i, (&ri, &wi)) in raw.iter().zip(weight).enumerate() {
                    let o = offset.map_or(0.0, |off| f64::from(off[i]));
                    predicted += f64::from(wi) * inverse_link_f64(link, f64::from(ri) + o + mid);
                }
                if predicted < observed {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            Ok(0.5 * (lo + hi))
        }
    }
}

/// Sample rows for a growth round; for [`Sampling::Mvs`] also returns the §06.5 1/p_i
/// importance-reweighting multiplier for each returned row (`None` for [`Sampling::Full`], where
/// every row is included with p_i = 1). Callers MUST scale that round's `(g, h)` at each sampled
/// row by its multiplier (see `reweighted_gh`) before using them for SPLIT SELECTION — the raw
/// gradients/hessians are otherwise a gradient-biased sample of the population, not an unbiased
/// estimator of the full-data gradient and Hessian totals.
///
/// Returns `Cow` rather than an owned `Vec`: `Sampling::Full` (and the MVS `k == n` fallback)
/// borrow `all_rows` directly instead of copying it — this is the row list `fit_single`'s round
/// loop re-derives every round, and on the default (no MVS) path it never changes, so cloning it
/// per round is pure waste. Only an actual MVS sub-sample owns a freshly built `Vec`.
type SampledRows<'a> = (std::borrow::Cow<'a, [u32]>, Option<Vec<f64>>);

fn sample_rows<'a>(
    sampling: &Sampling,
    gh: &GradHess,
    seed: u64,
    round: u32,
    all_rows: &'a [u32],
) -> Result<SampledRows<'a>, PbError> {
    sample_rows_by_norm(sampling, seed, round, all_rows, |row| {
        let g = f64::from(*gh.g.get(row as usize).ok_or_else(|| PbError::Internal {
            what: "MVS row escaped gradients".into(),
        })?);
        let h = f64::from(*gh.h.get(row as usize).ok_or_else(|| PbError::Internal {
            what: "MVS row escaped hessians".into(),
        })?);
        Ok(g.hypot(h))
    })
}

/// One shared draw for all class trees, using the joint gradient/Hessian norm.
fn mc_sample_rows<'a>(
    sampling: &Sampling,
    gh_by_class: &[GradHess],
    seed: u64,
    round: u32,
    all_rows: &'a [u32],
) -> Result<SampledRows<'a>, PbError> {
    if gh_by_class.is_empty() {
        return Ok((std::borrow::Cow::Borrowed(all_rows), None));
    }
    sample_rows_by_norm(sampling, seed, round, all_rows, |row| {
        let mut squared = 0.;
        for gh in gh_by_class {
            let g = f64::from(*gh.g.get(row as usize).ok_or_else(|| PbError::Internal {
                what: "multiclass MVS row escaped gradients".into(),
            })?);
            let h = f64::from(*gh.h.get(row as usize).ok_or_else(|| PbError::Internal {
                what: "multiclass MVS row escaped hessians".into(),
            })?);
            squared += g * g + h * h;
        }
        Ok(squared.sqrt())
    })
}

fn sample_rows_by_norm<'a>(
    sampling: &Sampling,
    seed: u64,
    round: u32,
    all_rows: &'a [u32],
    norm: impl Fn(u32) -> Result<f64, PbError>,
) -> Result<SampledRows<'a>, PbError> {
    let Sampling::Mvs { rate, min_rows } = *sampling else {
        return Ok((std::borrow::Cow::Borrowed(all_rows), None));
    };
    let n = all_rows.len();
    if n == 0 {
        return Ok((std::borrow::Cow::Borrowed(all_rows), None));
    }
    let min_rows = usize::try_from(min_rows).map_err(|_| PbError::Internal {
        what: "MVS min_rows exceeded usize".into(),
    })?;
    let k = (((n as f64) * f64::from(rate)).ceil() as usize)
        .max(min_rows)
        .min(n)
        .max(1);
    if k == n {
        return Ok((std::borrow::Cow::Borrowed(all_rows), None));
    }
    let mut scores = all_rows
        .iter()
        .map(|&row| {
            let score = norm(row)?;
            if !score.is_finite() {
                return Err(PbError::InvalidInput {
                    what: "MVS requires finite gradient norms".into(),
                });
            }
            Ok((row, score.max(1e-12)))
        })
        .collect::<Result<Vec<_>, PbError>>()?;
    scores.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    // Solve sum min(1, s_i / mu) = k. Summing the small scores first avoids
    // subtracting a dominant score from the total and losing the remaining mass.
    let mut prefix = 0.;
    let mut mu = 0.;
    for (pos, &(_, score)) in scores.iter().enumerate() {
        prefix += score;
        let count = pos + 1;
        if count > n - k {
            let candidate = prefix / (count - (n - k)) as f64;
            if scores.get(count).is_none_or(|next| next.1 >= candidate) {
                mu = candidate;
                break;
            }
        }
    }
    if !mu.is_finite() || mu <= 0. {
        return Err(PbError::Internal {
            what: "MVS inclusion threshold is invalid".into(),
        });
    }
    scores.sort_unstable_by_key(|&(row, _)| row);
    // A random offset through cumulative inclusion probabilities is a fixed-size
    // systematic sample. Each interval has length p_i, so its inclusion probability
    // is exactly p_i and 1/p_i corrects both gradient and Hessian totals.
    let bits = pb_seed(seed, round, Stage::Sample as u32, 0);
    let mut next = (bits >> 12) as f64 / (1_u64 << 52) as f64;
    let mut cumulative = 0.;
    let mut rows = Vec::with_capacity(k);
    let mut reweight = Vec::with_capacity(k);
    for (pos, &(row, score)) in scores.iter().enumerate() {
        let p = (score / mu).min(1.);
        cumulative = if pos + 1 == n {
            k as f64
        } else {
            cumulative + p
        };
        if next < cumulative && rows.len() < k {
            rows.push(row);
            reweight.push(1. / p);
            next += 1.;
        }
    }
    if rows.len() != k {
        return Err(PbError::Internal {
            what: "MVS draw missed its fixed sample size".into(),
        });
    }
    Ok((std::borrow::Cow::Owned(rows), Some(reweight)))
}

/// Scale `gh`'s gradient/hessian at each of `rows` by the parallel `reweight` multiplier (§06.5
/// MVS 1/p_i importance correction), returning a fresh clone. The caller's original `gh` — reused
/// unmodified for the full-data leaf refit right after growth (`refit_tree_leaves`), which must
/// stay an exact, unbiased full pass — is left untouched.
fn reweighted_gh(gh: &GradHess, rows: &[u32], reweight: &[f64]) -> Result<GradHess, PbError> {
    if rows.len() != reweight.len() {
        return Err(PbError::Internal {
            what: "MVS reweight length does not match sampled rows".into(),
        });
    }
    let mut scaled = gh.clone();
    for (&row, &mult) in rows.iter().zip(reweight) {
        let idx = row as usize;
        let g = scaled.g.get_mut(idx).ok_or_else(|| PbError::Internal {
            what: "MVS reweight row escaped gradients".into(),
        })?;
        *g = (f64::from(*g) * mult) as f32;
        let h = scaled.h.get_mut(idx).ok_or_else(|| PbError::Internal {
            what: "MVS reweight row escaped hessians".into(),
        })?;
        *h = (f64::from(*h) * mult) as f32;
    }
    Ok(scaled)
}

/// ES-stratify rule (§ES-runaway fix #2, 2026-07-20): the single source of truth for which
/// objectives get a stratified internal ES holdout, shared by every call site that needs to
/// derive `y`'s strata — [`fit_single`]'s own automatic carve AND the Python binding's ordered-TS
/// "one honest holdout" precompute (crates/t-boost-py/src/lib.rs `fit_model_ambient`), which
/// must reproduce this rule bit-for-bit rather than let the two derivations drift apart.
///
/// `Logistic`: stratum is `1` iff `y >= 0.5` (hard 0/1 labels and Logistic's supported soft/
/// probabilistic labels alike) — the original ES-stratify fix, so a rare positive class isn't
/// starved of holdout representation.
///
/// `Poisson`/`Tweedie`: stratum is `1` iff `y > 0.0`. On high-zero-mass frequency/pure-premium
/// data (aggregated low-frequency counts, or Tweedie pure premium where most rows have no
/// claim) an unstratified thin holdout (`validation_fraction` ~0.1) can land almost entirely on
/// zero rows — and a zero row's deviance is monotone-decreasing in the fitted mean, so held-out
/// deviance then "improves" as the mean shrinks regardless of genuine skill, and early stopping
/// never fires (measured: runs to the tree cap on zero-dominated aggregated data — the same
/// catastrophic class as the grouped-panel carve leak, different mechanism).
///
/// `Gamma`/`SquaredError`/`Softmax`: `None` (unstratified), unaffected by construction — Gamma is
/// `y > 0` by construction (no zero mass to stratify against), `SquaredError` has no zero-mass
/// semantics, and `Softmax` (multiclass) is already class-stratified by its own caller
/// (`fit_multiclass`'s `Some(&labels)`), not through this function.
pub fn es_strata_for_loss(loss: LossId, y: &[f32]) -> Option<Vec<u32>> {
    match loss {
        LossId::Logistic => Some(y.iter().map(|&v| u32::from(v >= 0.5)).collect()),
        LossId::Poisson | LossId::Tweedie => Some(y.iter().map(|&v| u32::from(v > 0.0)).collect()),
        LossId::SquaredError | LossId::Gamma | LossId::Softmax => None,
    }
}

/// Deterministic validation-holdout mask: pure in `(n_rows, validation_fraction, seed)` when
/// `strata` is `None`, so binning-time encoders (`bin_train_columns_with_holdout`) and the boost
/// loop derive the SAME carve independently — the "one honest holdout" contract
/// (design/ordered-ts-early-stopping.md). That contract only ever calls this with `strata: None`
/// (via `spec.fixed_holdout`, computed once and passed in explicitly — this function is never
/// invoked a second time in that path), so the stratified branch below cannot desync it.
///
/// `strata: Some(labels)` (ES-stratify fix, see [`es_strata_for_loss`] for who supplies which
/// labels): ranks rows for selection WITHIN each distinct label value independently, instead of
/// one global ranking, so every class/stratum gets its own ~`frac` share of the holdout — a
/// global ranking can otherwise starve a rare stratum down to near-zero holdout representation,
/// making `best_iteration` noise-driven (or, worse, never converge at all). Reuses the SAME
/// per-row hash (`Stage::Holdout`, unchanged — the hash-stage enumeration is frozen,
/// append-only) as the unstratified path; only the grouping-before-ranking differs, so this is
/// purely additive.
pub fn holdout_mask(
    n_rows: u32,
    validation_fraction: Option<f32>,
    seed: u64,
    strata: Option<&[u32]>,
) -> Result<Option<Vec<bool>>, PbError> {
    let Some(frac) = validation_fraction else {
        return Ok(None);
    };
    if n_rows < 2 {
        return Err(PbError::InvalidConfig {
            what: "validation_fraction requires at least two rows".into(),
        });
    }
    let n = n_rows as usize;
    let Some(strata) = strata else {
        let holdout = ((n as f64) * f64::from(frac)).ceil() as usize;
        let holdout = holdout.clamp(1, n - 1);
        let mut keyed: Vec<(u64, u32)> = Vec::with_capacity(n);
        for row in 0..n_rows {
            keyed.push((pb_seed(seed, 0, Stage::Holdout as u32, row), row));
        }
        keyed.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut is_holdout = vec![false; n];
        for &(_, row) in keyed.iter().take(holdout) {
            *is_holdout
                .get_mut(row as usize)
                .ok_or_else(|| PbError::Internal {
                    what: "holdout row escaped mask".into(),
                })? = true;
        }
        return Ok(Some(is_holdout));
    };
    if strata.len() != n {
        return Err(PbError::ShapeMismatch {
            what: format!("holdout strata len {} != n_rows {n}", strata.len()),
        });
    }
    let mut groups: std::collections::BTreeMap<u32, Vec<u32>> = std::collections::BTreeMap::new();
    for row in 0..n_rows {
        let label = *strata.get(row as usize).ok_or_else(|| PbError::Internal {
            what: "holdout stratum row escaped".into(),
        })?;
        groups.entry(label).or_default().push(row);
    }
    let mut is_holdout = vec![false; n];
    for rows in groups.values() {
        let m = rows.len();
        // A singleton stratum can't supply both a holdout AND a train row from one example;
        // keeping it in train (rather than losing the class's only training example) mirrors
        // sklearn's stratified-split behavior on classes too rare to split.
        if m < 2 {
            continue;
        }
        let take = (((m as f64) * f64::from(frac)).ceil() as usize).clamp(1, m - 1);
        let mut keyed: Vec<(u64, u32)> = rows
            .iter()
            .map(|&row| (pb_seed(seed, 0, Stage::Holdout as u32, row), row))
            .collect();
        keyed.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for &(_, row) in keyed.iter().take(take) {
            *is_holdout
                .get_mut(row as usize)
                .ok_or_else(|| PbError::Internal {
                    what: "stratified holdout row escaped mask".into(),
                })? = true;
        }
    }
    Ok(Some(is_holdout))
}

// Zero-mass validation has no defined deviance. Disable early stopping in that case.
// If an automatic carve takes every positive-weight row, abandon that carve entirely;
// an explicit holdout must remain excluded from training, even when fitting is impossible.
fn ensure_validation_mass(
    train: &mut Vec<u32>,
    validation: &mut Option<Vec<usize>>,
    weight: &[f32],
    fixed_holdout: bool,
) -> Result<(), PbError> {
    let has_training_mass = train
        .iter()
        .any(|&r| weight.get(r as usize).is_some_and(|&w| w > 0.0));
    if !has_training_mass {
        if fixed_holdout {
            return Err(PbError::InvalidInput {
                what: "training partition has no positive sample weight".into(),
            });
        }
        if let Some(rows) = validation.take() {
            for row in rows {
                train.push(u32::try_from(row).map_err(|_| PbError::Internal {
                    what: "validation row exceeds u32".into(),
                })?);
            }
            train.sort_unstable();
        }
    } else if validation.as_ref().is_some_and(|rows| {
        !rows
            .iter()
            .any(|&r| weight.get(r).is_some_and(|&w| w > 0.0))
    }) {
        *validation = None;
    }
    Ok(())
}

/// Split rows by a precomputed holdout mask into (train rows, validation rows) with the same
/// shapes `carve_validation_rows_stratified` produces. An all-false mask degrades to "no validation".
fn split_rows_by_mask(
    n_rows: u32,
    mask: &[bool],
) -> Result<(Vec<u32>, Option<Vec<usize>>), PbError> {
    if mask.len() != n_rows as usize {
        return Err(PbError::ShapeMismatch {
            what: format!("fixed_holdout len {} != n_rows {}", mask.len(), n_rows),
        });
    }
    let mut train_rows = Vec::new();
    let mut validation_rows = Vec::new();
    for row in 0..n_rows {
        if mask[row as usize] {
            validation_rows.push(row as usize);
        } else {
            train_rows.push(row);
        }
    }
    if validation_rows.is_empty() {
        return Ok((train_rows, None));
    }
    Ok((train_rows, Some(validation_rows)))
}

/// ES-stratify fix: `strata: None` is the original unstratified carve (the pre-fix code moved
/// into `holdout_mask`'s early-return arm, byte-identical); `strata: Some(labels)` ranks the holdout
/// selection within each class independently (see `holdout_mask`'s doc) instead of one global
/// ranking. Callers passing `Some`: `fit_single` for Logistic (it has `y` in scope for the
/// top-level fit AND every OuterBag/GreedySelect member fit, so one call site covers every
/// binary carve), and `fit_multiclass` with its per-row class `labels`. Every other objective
/// passes `None` and is bit-identical to the pre-fix carve.
fn carve_validation_rows_stratified(
    n_rows: u32,
    validation_fraction: Option<f32>,
    seed: u64,
    strata: Option<&[u32]>,
) -> Result<(Vec<u32>, Option<Vec<usize>>), PbError> {
    let all_rows: Vec<u32> = (0..n_rows).collect();
    let Some(is_holdout) = holdout_mask(n_rows, validation_fraction, seed, strata)? else {
        return Ok((all_rows, None));
    };
    let mut train_rows = Vec::new();
    let mut validation_rows = Vec::new();
    for row in 0..n_rows {
        if *is_holdout
            .get(row as usize)
            .ok_or_else(|| PbError::Internal {
                what: "holdout row escaped final mask".into(),
            })?
        {
            validation_rows.push(row as usize);
        } else {
            train_rows.push(row);
        }
    }
    Ok((train_rows, Some(validation_rows)))
}

fn sample_axes(axes: &[u32], rate: f32, seed: u64, round: u32) -> Result<Vec<u32>, PbError> {
    if axes.is_empty() || rate >= 1.0 {
        return Ok(axes.to_vec());
    }
    let n = axes.len();
    let k = ((n as f64) * f64::from(rate)).ceil() as usize;
    let k = k.clamp(1, n);
    let mut keyed = Vec::with_capacity(n);
    for &axis in axes {
        keyed.push((pb_seed(seed, round, Stage::Cols as u32, axis), axis));
    }
    keyed.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut sampled: Vec<u32> = keyed.into_iter().take(k).map(|(_, axis)| axis).collect();
    sampled.sort_unstable();
    Ok(sampled)
}

fn learning_rate_for_round(base: f32, decay: f32, round: u32) -> f64 {
    f64::from(base) / (1.0 + f64::from(decay) * f64::from(round))
}

/// This round's effective L2 leaf regularizer for `Config::lambda_scale_invariant`. `false` (the
/// default) returns `base_lambda` untouched — no floating-point operation runs — so the OFF path
/// is bit-identical to the legacy `H+λ` denominator downstream (every split-gain and leaf-value
/// Newton solve reads `GrowConfig::lambda`, so setting it once here reaches both). `true` rescales
/// by this round's mean per-row hessian h̄ = Σh/N over `rows` — the EXACT, pre-MVS/pre-reweight
/// `gh` — giving `λ_eff = λ·h̄`: a scale-free "pseudo-count of prior rows" that stays live
/// regardless of per-row weight/exposure magnitude (see the field doc for the motivating case).
///
/// # Errors
/// [`PbError::Internal`] if a row index escapes `gh.h`.
fn round_lambda(
    base_lambda: f64,
    scale_invariant: bool,
    gh: &GradHess,
    rows: &[u32],
) -> Result<f64, PbError> {
    if !scale_invariant || rows.is_empty() {
        return Ok(base_lambda);
    }
    let mut sum_h = 0.0_f64;
    for &r in rows {
        sum_h += f64::from(*gh.h.get(r as usize).ok_or_else(|| PbError::Internal {
            what: "round_lambda: row escaped gh.h".into(),
        })?);
    }
    Ok(base_lambda * (sum_h / rows.len() as f64))
}

const INTERACTION_GAIN_REFERENCE_DECAY: f64 = 0.999;

fn update_interaction_gain_reference(reference: &mut Option<f64>, gain: f64) {
    if gain.is_finite() && gain > 0.0 {
        *reference = Some(reference.map_or(gain, |current| {
            (current * INTERACTION_GAIN_REFERENCE_DECAY).max(gain)
        }));
    }
}

#[allow(clippy::too_many_arguments)]
fn refine_tree_leaves_after_grow(
    config: &Config,
    loss: &dyn crate::loss::Loss,
    y: &[f32],
    weight: &[f32],
    base_raw: &[f32],
    x: &BinnedMatrix,
    rows: &[u32],
    monotone: Option<&[Option<MonoSign>]>,
    tree: &mut ObliviousTree,
    grow_cfg: &GrowConfig<'_>,
    // grow's per-row leaf map (absolute-indexed), passed `Some` only when grow saw exactly `rows`
    // (no subsample). When present it is gathered in `rows` order to skip the tree re-walk —
    // byte-identical, since grow set it with the SAME canonical `low_bit` the walk uses.
    precomputed_leaf_of_row: Option<&[u8]>,
    // Caller-precomputed `(gather_rows(y, rows), gather_rows(weight, rows))`, passed `Some` only
    // when the caller has proven `rows`/`y`/`weight` round-invariant for this fit (currently:
    // `Sampling::Full`, where every call site passes the same fixed `train_rows`). Reusing it here
    // skips this function's own O(rows) gather — bit-identical, since it is the exact same gather
    // over the exact same `rows` the caller would otherwise redo.
    hoisted_y_w: Option<(&[f32], &[f32])>,
) -> Result<(), PbError> {
    if config.leaf_refine_steps == 0 || rows.is_empty() {
        return Ok(());
    }
    let n_rows = x.n_rows as usize;
    if base_raw.len() != n_rows {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "leaf refinement base_raw len {} != n_rows {n_rows}",
                base_raw.len()
            ),
        });
    }
    let n_leaves = 1usize << usize::from(tree.depth);
    let memberships = prof::timed("refine.members", || -> Result<Vec<u8>, PbError> {
        match precomputed_leaf_of_row {
            // grow already assigned each row its leaf via the SAME canonical `low_bit` the tree
            // walk uses, so `leaf_of_row[rows[i]]` is bit-identical to a re-walk — gather it in
            // `rows` order and skip the per-row walk. (Caller guarantees grow saw exactly `rows`.)
            Some(leaf_of_row) => gather_memberships(leaf_of_row, rows, n_leaves),
            None => {
                let columns = tree_split_columns(tree, &x.data)?;
                tree_memberships_for_rows(tree, &columns, rows, n_leaves)
            }
        }
    })?;
    // SquaredError's half-deviance is EXACTLY the separable 8-D quadratic `D_l(v) = C_l + v·B_l +
    // ½·v²·H_l` (`B_l = Σ_{rows∈l} w(base−y)`, `H_l = Σw`, `C_l = ½Σw(base−y)²`, all CONSTANT across
    // steps). So the multi-step damped-Newton refine + backtrack collapse to O(8) per step — one
    // O(rows) aggregate of `B_l/H_l/C_l` up front, then the per-step aggregate gradient is the exact
    // recurrence `G_l = B_l + H_l·v_l` and every trial deviance is the closed form — eliminating the
    // O(rows) grad_hess / deviance re-folds (`refine.grad_hess`, `refine.backtrack_eval`). Iterates
    // are the same damped-Newton steps as the per-row path; accuracy-neutral (~1e-11, the summation
    // groups differently). Only SE is quadratic — log-link keeps the per-row path below.
    if matches!(loss.objective_tag().loss, crate::loss::LossId::SquaredError) {
        return refine_tree_leaves_se_quadratic(
            config,
            loss,
            y,
            weight,
            base_raw,
            rows,
            monotone,
            tree,
            grow_cfg,
            &memberships,
            n_leaves,
        );
    }
    // The line search reads y/weight and the base score ONLY at `rows`, and only the 8 leaf
    // VALUES change per trial. Gather those three into DENSE per-tree buffers ONCE (constant
    // across every step + backtrack), so each trial is a single contiguous fill + the
    // vectorized `deviance` over contiguous slices — no per-trial scatter-gather, no
    // allocation. `base_sub[i] == base_raw[rows[i]]`.
    // `y_sub`/`w_sub` are round-invariant whenever `hoisted_y_w` is `Some` (see the caller-side
    // gate on `hoisted_y_w`'s doc above) — reuse them instead of re-gathering; `Cow` keeps every
    // downstream read (`&y_sub`, `.len()`, ...) identical whether borrowed or freshly gathered.
    // `base_sub` always changes round to round (it reads `base_raw`, this round's fit raw score),
    // so it is never a hoist candidate and is still gathered here every call.
    let (y_sub, w_sub): (Cow<'_, [f32]>, Cow<'_, [f32]>) = match hoisted_y_w {
        Some((y_pre, w_pre)) => (Cow::Borrowed(y_pre), Cow::Borrowed(w_pre)),
        None => (
            Cow::Owned(gather_rows(y, rows)?),
            Cow::Owned(gather_rows(weight, rows)?),
        ),
    };
    let base_sub = gather_rows(base_raw, rows)?;
    // Log-link (Poisson) closed-form fast path: the per-trial O(rows) exp+ln deviance re-folds of
    // the generic line search below collapse to an O(#leaves) closed form (`refine_leaves_loglink_
    // closed`). It handles the refine outright when it applies, and DECLINES (→ the generic per-row
    // path) for unsupported losses or when the ±30 exp clamp could bite. A test-only toggle forces
    // the decline so the closed and generic paths can be A/B-compared for leaf-selection identity.
    if !loglink_closed_disabled()
        && refine_leaves_loglink_closed(
            config,
            loss,
            &y_sub,
            &w_sub,
            &base_sub,
            &memberships,
            monotone,
            tree,
            grow_cfg,
            n_leaves,
        )?
    {
        return Ok(());
    }
    // Reused dense subset-raw scratch (no per-trial alloc) — the SINGLE source of truth for the raw
    // at `rows`. The backtrack refills it each trial; on accept it already holds the accepted leaves'
    // raw, so the next step's grad_hess reads it directly. grad_hess is pointwise, so evaluating it
    // over (y_sub, trial_raw_sub, w_sub) gives bit-identical (g,h) to the old full-length grad_hess
    // read at `rows` — and is O(rows) not O(n), with no full `raw` clone and no per-accept scatter.
    let mut trial_raw_sub = base_sub.clone();
    fill_leaf_raw_contiguous(&mut trial_raw_sub, &base_sub, &memberships, &tree.leaves)?;
    let mut gh = GradHess::default();
    // ALWAYS fuse init_dev with step-0's grad_hess into ONE pass over the dense subset
    // (`grad_hess_and_deviance` shares the link σ/exp): the returned deviance is bit-identical to the
    // standalone `deviance`, and `gh` is then step-0's gradient (bit-identical to a separate
    // grad_hess). On a validation split (`rows ⊊ 0..n`) this drops the previously-separate step-0
    // grad_hess pass — a pure byte-identical reduction (the full-sample path already did this).
    let init_dev = prof::timed("refine.init_dev", || {
        loss.grad_hess_and_deviance(&y_sub, &trial_raw_sub, &w_sub, &mut gh)
    })?;
    let mut best_deviance = f64::from(init_dev);
    // §07.6 leaf_refine companion fix: membership (hence row count) is fixed across every step
    // of this call (only leaf VALUES move), so tally once. Gated on `path_smooth > 0` — the
    // (default) inert path never pays for this tally.
    let credibility_counts = if grow_cfg.credibility.path_smooth > 0.0 {
        leaf_row_counts(&memberships, n_leaves)?
    } else {
        [0u64; MAX_LEAVES]
    };

    // Losses with a one-pass `grad_hess_and_deviance` evaluate each backtrack trial with it: when
    // the trial is accepted its (g, h) are exactly the next step's (same raw, pointwise and
    // bit-identical to `grad_hess`), so that step skips its own O(rows) gradient pass. The fused
    // deviance is bit-identical to `deviance` (same clamped terms, same chunked fold).
    let fused_trials = matches!(
        loss.objective_tag().loss,
        crate::loss::LossId::Logistic | crate::loss::LossId::Poisson | crate::loss::LossId::Tweedie
    );
    let mut gh_trial = GradHess::default();
    // `gh` already holds the gradient at the current `trial_raw_sub` (step 0: the fused init pass).
    let mut gh_current = true;
    for _step in 0..config.leaf_refine_steps {
        // Otherwise recompute the gradient over the dense subset (O(rows)); `gh` is subset-indexed.
        if !gh_current {
            prof::timed("refine.grad_hess", || {
                loss.grad_hess(&y_sub, &trial_raw_sub, &w_sub, &mut gh)
            })?;
        }
        gh_current = false;
        let mut g = [0.0_f64; MAX_LEAVES];
        let mut h = [0.0_f64; MAX_LEAVES];
        prof::timed("refine.aggregate", || -> Result<(), PbError> {
            // `gh` is subset-indexed (i ↔ rows[i]); fold each into its leaf in the SAME rows order as
            // before ⇒ bit-identical per-leaf sums.
            for (i, &leaf_u8) in memberships.iter().enumerate() {
                let leaf = usize::from(leaf_u8);
                *g.get_mut(leaf).ok_or_else(|| PbError::Internal {
                    what: "leaf refinement g leaf escaped".into(),
                })? += f64::from(*gh.g.get(i).ok_or_else(|| PbError::Internal {
                    what: "leaf refinement gradient row escaped".into(),
                })?);
                *h.get_mut(leaf).ok_or_else(|| PbError::Internal {
                    what: "leaf refinement h leaf escaped".into(),
                })? += f64::from(*gh.h.get(i).ok_or_else(|| PbError::Internal {
                    what: "leaf refinement hessian row escaped".into(),
                })?);
            }
            Ok(())
        })?;
        // §07.6 evidence normalization: this step's mean per-row hessian over the tree's rows.
        // Recomputed each step since `h` is fresh each step (the round's residual moves as
        // trial leaves are accepted); gated on `path_smooth > 0` like the count tally above.
        let h_bar = if grow_cfg.credibility.path_smooth > 0.0 {
            refine_h_bar(&h, n_leaves, memberships.len())
        } else {
            0.0
        };

        let mut delta = [0.0_f32; MAX_LEAVES];
        let mut any_delta = false;
        for leaf in 0..n_leaves {
            let step = incremental_leaf_delta(
                *g.get(leaf).ok_or_else(|| PbError::Internal {
                    what: "leaf refinement g lookup escaped".into(),
                })?,
                *h.get(leaf).ok_or_else(|| PbError::Internal {
                    what: "leaf refinement h lookup escaped".into(),
                })?,
                grow_cfg.lambda,
                grow_cfg.l1_leaf,
                grow_cfg.max_delta_step,
                grow_cfg.lr,
            )?;
            let credibility_n = credibility_evidence_n(
                grow_cfg.credibility_evidence,
                *h.get(leaf).ok_or_else(|| PbError::Internal {
                    what: "leaf refinement h lookup escaped".into(),
                })?,
                *credibility_counts
                    .get(leaf)
                    .ok_or_else(|| PbError::Internal {
                        what: "leaf refinement credibility count lookup escaped".into(),
                    })?,
                h_bar,
            );
            let step = shrink_refine_delta(step, credibility_n, grow_cfg.credibility.path_smooth);
            if step.abs() > 1.0e-7 {
                any_delta = true;
            }
            *delta.get_mut(leaf).ok_or_else(|| PbError::Internal {
                what: "leaf refinement delta lookup escaped".into(),
            })? = step;
        }
        if !any_delta {
            break;
        }

        let mut accepted = false;
        let mut scale = 1.0_f32;
        for _ in 0..config.leaf_refine_backtracks {
            let mut trial_leaves = tree.leaves.clone();
            for (leaf_value, delta_value) in trial_leaves.iter_mut().zip(delta.iter()) {
                let value = f64::from(*leaf_value) + f64::from(scale * *delta_value);
                if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX)
                {
                    return Err(PbError::InvalidInput {
                        what: "leaf refinement value is not finite/representable".into(),
                    });
                }
                *leaf_value = value as f32;
            }
            clamp_monotone(
                &mut trial_leaves,
                &tree.splits,
                usize::from(tree.depth),
                monotone,
            )?;
            let deviance = prof::timed("refine.backtrack_eval", || -> Result<f64, PbError> {
                // Fill the dense subset raw from the fixed base + the 8 trial leaf values, then
                // the vectorized `deviance` over contiguous (y_sub, trial_raw_sub, w_sub) — the
                // same value the old `apply_membership_leaves` + gather-`deviance` produced.
                fill_leaf_raw_contiguous(
                    &mut trial_raw_sub,
                    &base_sub,
                    &memberships,
                    &trial_leaves,
                )?;
                Ok(f64::from(if fused_trials {
                    loss.grad_hess_and_deviance(&y_sub, &trial_raw_sub, &w_sub, &mut gh_trial)?
                } else {
                    loss.deviance(&y_sub, &trial_raw_sub, &w_sub)?
                }))
            })?;
            if deviance < best_deviance {
                tree.leaves = trial_leaves;
                // `trial_raw_sub` already holds this accepted trial's raw at `rows` (the backtrack
                // just filled it), so it IS the next step's grad_hess input — no extra scatter, no
                // full `raw` to maintain; under `fused_trials` its gradient is already in hand too.
                if fused_trials {
                    std::mem::swap(&mut gh, &mut gh_trial);
                    gh_current = true;
                }
                best_deviance = deviance;
                accepted = true;
                break;
            }
            scale *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    Ok(())
}

/// Closed-form-DEVIANCE leaf refinement for SquaredError (Identity link). SE's half-deviance is the
/// separable 8-D quadratic `D_l(v) = C_l + v·B_l + ½·v²·H_l` (`B_l = Σ_{rows∈l} w·(base−y)`,
/// `H_l = Σw`, `C_l = ½Σw·(base−y)²`, all CONSTANT), so the O(rows) deviance re-folds collapse to
/// O(8). The per-row `grad_hess` + aggregate that produce the leaf UPDATES are KEPT VERBATIM (same
/// f32 path as the generic `refine_tree_leaves_after_grow`), so the leaves — hence the model, scores
/// and early-stop trajectory — are byte-identical; only `refine.init_dev`/`refine.backtrack_eval`
/// turn from an O(rows) fold into the O(8) closed form. The closed-form value is f32-cast EXACTLY as
/// `Loss::deviance`, so the accept comparison matches the per-row deviance save for a vanishingly
/// rare f32-boundary straddle. Deterministic: the coefficient fold is one fixed-order sequential pass.
#[allow(clippy::too_many_arguments)]
fn refine_tree_leaves_se_quadratic(
    config: &Config,
    loss: &dyn crate::loss::Loss,
    y: &[f32],
    weight: &[f32],
    base_raw: &[f32],
    rows: &[u32],
    monotone: Option<&[Option<MonoSign>]>,
    tree: &mut ObliviousTree,
    grow_cfg: &GrowConfig<'_>,
    memberships: &[u8],
    n_leaves: usize,
) -> Result<(), PbError> {
    // Per-leaf quadratic coefficients of SE's half-deviance, from the FIXED base score (one pass).
    let mut coef_b = [0.0_f64; MAX_LEAVES];
    let mut coef_h = [0.0_f64; MAX_LEAVES];
    let mut coef_c = [0.0_f64; MAX_LEAVES];
    for (&row, &leaf_u8) in rows.iter().zip(memberships) {
        let ru = row as usize;
        let leaf = usize::from(leaf_u8);
        let wi = f64::from(*weight.get(ru).ok_or_else(|| PbError::Internal {
            what: "se refine weight row escaped".into(),
        })?);
        let ri = f64::from(*base_raw.get(ru).ok_or_else(|| PbError::Internal {
            what: "se refine base row escaped".into(),
        })?) - f64::from(*y.get(ru).ok_or_else(|| PbError::Internal {
            what: "se refine y row escaped".into(),
        })?);
        *coef_b.get_mut(leaf).ok_or_else(|| PbError::Internal {
            what: "se refine B leaf escaped".into(),
        })? += wi * ri;
        *coef_h.get_mut(leaf).ok_or_else(|| PbError::Internal {
            what: "se refine H leaf escaped".into(),
        })? += wi;
        *coef_c.get_mut(leaf).ok_or_else(|| PbError::Internal {
            what: "se refine C leaf escaped".into(),
        })? += 0.5 * wi * ri * ri;
    }
    // Closed-form half-deviance `Σ_l (C_l + v_l·B_l + ½·v_l²·H_l)`, f32-cast EXACTLY as
    // `SquaredError::deviance` (`finish_deviance(0.5·acc)`) then widened — so the accept comparison
    // matches the per-row deviance bit-for-bit save for a rare f32-boundary straddle.
    let deviance_of = |leaves: &[f32]| -> f64 {
        let mut d = 0.0_f64;
        for leaf in 0..n_leaves {
            let v = f64::from(leaves.get(leaf).copied().unwrap_or(0.0));
            d += coef_c.get(leaf).copied().unwrap_or(0.0)
                + v * coef_b.get(leaf).copied().unwrap_or(0.0)
                + 0.5 * v * v * coef_h.get(leaf).copied().unwrap_or(0.0);
        }
        f64::from(d as f32)
    };
    // `raw` kept valid at `rows` for the per-step grad_hess (leaf updates come from the SAME f32
    // aggregate the generic path uses ⇒ byte-identical leaves).
    let mut raw = base_raw.to_vec();
    apply_membership_leaves(&mut raw, base_raw, rows, memberships, &tree.leaves)?;
    let mut gh = GradHess::default();
    let mut best_deviance = prof::timed("refine.init_dev", || {
        Ok::<f64, PbError>(deviance_of(&tree.leaves))
    })?;
    // §07.6 leaf_refine companion fix: membership (hence row count) is fixed across every step
    // of this call. Gated on `path_smooth > 0` — the (default) inert path never pays for this.
    let credibility_counts = if grow_cfg.credibility.path_smooth > 0.0 {
        leaf_row_counts(memberships, n_leaves)?
    } else {
        [0u64; MAX_LEAVES]
    };

    for _ in 0..config.leaf_refine_steps {
        // FUSED grad_hess + per-leaf aggregate in ONE rows-order pass (no materialized gradient
        // vector). The SquaredError override computes each row's f32 (g,h) inline and folds it into
        // the per-leaf f64 sums — bit-identical to the old grad_hess-then-aggregate, and the leaves
        // (hence the whole model) stay byte-identical. Only the deviance below is closed-form.
        let (g, h) = prof::timed("refine.grad_hess", || {
            loss.grad_hess_aggregate(y, &raw, weight, rows, memberships, &mut gh)
        })?;
        // §07.6 evidence normalization (always inert here in practice: this closed form only
        // ever runs for SquaredError, which stays on Count evidence per
        // `credibility_evidence_for_loss` — computed anyway for structural parity with the
        // other 3 refine call sites, which DO need it).
        let h_bar = if grow_cfg.credibility.path_smooth > 0.0 {
            refine_h_bar(&h, n_leaves, memberships.len())
        } else {
            0.0
        };
        let mut delta = [0.0_f32; MAX_LEAVES];
        let mut any_delta = false;
        for leaf in 0..n_leaves {
            let h_leaf = *h.get(leaf).ok_or_else(|| PbError::Internal {
                what: "se refine h lookup escaped".into(),
            })?;
            let step = incremental_leaf_delta(
                *g.get(leaf).ok_or_else(|| PbError::Internal {
                    what: "se refine g lookup escaped".into(),
                })?,
                h_leaf,
                grow_cfg.lambda,
                grow_cfg.l1_leaf,
                grow_cfg.max_delta_step,
                grow_cfg.lr,
            )?;
            let credibility_n = credibility_evidence_n(
                grow_cfg.credibility_evidence,
                h_leaf,
                *credibility_counts
                    .get(leaf)
                    .ok_or_else(|| PbError::Internal {
                        what: "se refine credibility count lookup escaped".into(),
                    })?,
                h_bar,
            );
            let step = shrink_refine_delta(step, credibility_n, grow_cfg.credibility.path_smooth);
            if step.abs() > 1.0e-7 {
                any_delta = true;
            }
            *delta.get_mut(leaf).ok_or_else(|| PbError::Internal {
                what: "se refine delta escaped".into(),
            })? = step;
        }
        if !any_delta {
            break;
        }

        let mut accepted = false;
        let mut scale = 1.0_f32;
        for _ in 0..config.leaf_refine_backtracks {
            let mut trial_leaves = tree.leaves.clone();
            for (leaf_value, delta_value) in trial_leaves.iter_mut().zip(delta.iter()) {
                let value = f64::from(*leaf_value) + f64::from(scale * *delta_value);
                if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX)
                {
                    return Err(PbError::InvalidInput {
                        what: "leaf refinement value is not finite/representable".into(),
                    });
                }
                *leaf_value = value as f32;
            }
            clamp_monotone(
                &mut trial_leaves,
                &tree.splits,
                usize::from(tree.depth),
                monotone,
            )?;
            let deviance = prof::timed("refine.backtrack_eval", || {
                Ok::<f64, PbError>(deviance_of(&trial_leaves))
            })?;
            if deviance < best_deviance {
                tree.leaves = trial_leaves;
                // Reflect the accepted leaves into `raw` at `rows` for the next step's grad_hess.
                apply_membership_leaves(&mut raw, base_raw, rows, memberships, &tree.leaves)?;
                best_deviance = deviance;
                accepted = true;
                break;
            }
            scale *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    Ok(())
}

/// The §05.4 exp-clamp bound (`loss::EXP_CLAMP`, ±30) widened to `f64` — the log-link closed form
/// falls back to the per-row path before the factored exponent `base + v` could reach it.
const EXP_CLAMP_F64: f64 = crate::loss::EXP_CLAMP as f64;

/// Per-leaf closed-form coefficients of the Poisson weighted deviance for a trial leaf vector `v`,
/// with `F_i = base_i + v_{leaf(i)}` and `μ_i = eb_i·e^{v_l}` (`eb_i = exp(base_i)`):
///
/// `D(v) = K − Σ_l v_l·A_l + Σ_l e^{v_l}·M_l`, where
/// `M_l = Σ_{i∈l} 2 w_i eb_i`, `A_l = Σ_{i∈l} 2 w_i y_i`,
/// `K = Σ_i 2 w_i (y_i ln y_i − y_i base_i − y_i)` (the `y=0` term uses `y ln y := 0`, matching
/// `Poisson::deviance`'s `y>0` branch). Derived by substituting `ln μ_i = base_i + v_l` and
/// `μ_i = eb_i e^{v_l}` into the unit deviance `2 w (y ln(y/μ) − (y − μ))`.
///
/// Also returns `max_i |base_i|` (for the exp-clamp fall-back guard) and `Σ w_i` (to reject the
/// all-zero-weight domain like `Poisson::deviance`). One contiguous fold in `rows` order; the same
/// per-row domain guards as `Poisson::deviance`, validated once (they are trial-invariant). All
/// accumulators are `f64` in fixed subset order — byte-deterministic, no rayon.
struct PoissonLeafCoeffs {
    /// `M_l = Σ_{i∈l} 2 w_i exp(base_i)` (Poisson) / `Σ 2 w_i y_i exp(−base_i)` (Gamma).
    m: [f64; MAX_LEAVES],
    /// `A_l = Σ_{i∈l} 2 w_i y_i` (Poisson) / `−Σ 2 w_i` (Gamma — the sign folds the +v·W term
    /// into the shared `− v·A` template).
    a: [f64; MAX_LEAVES],
    /// `K = Σ_i 2 w_i (y_i ln y_i − y_i base_i − y_i)` (Poisson) /
    /// `Σ 2 w_i (base_i − ln y_i − 1)` (Gamma).
    k: f64,
    /// Exponent sign `s` in the shared template `D(v) = K + Σ_l (e^{s·v_l}·M_l − v_l·A_l)`:
    /// +1 (Poisson) or −1 (Gamma). Multiplying by +1.0 is IEEE-exact, so the Poisson path is
    /// bit-identical to the pre-Gamma code.
    exp_sign: f64,
    /// `max_i |base_i|` over the subset — the exp-clamp fall-back guard reads this.
    max_abs_base: f32,
    /// `Σ_i w_i` — rejects the all-zero-weight domain, as `Poisson::deviance` does.
    sum_w: f64,
}

/// Fixed row-chunk size for the parallel coefficient fold. Chunk boundaries and the
/// chunk-order combine are independent of the thread count, so the result is deterministic
/// for a given dataset regardless of `n_jobs` — but it is NOT bit-identical to the serial
/// single-fold (the f64 sums are grouped per chunk; measured drift ~1e-12 relative, same
/// class as the tier-2 refinement drift, ratified by Ralph 2026-07-16 with the 4-dataset
/// no-harm battery).
const LEAF_COEFF_CHUNK_ROWS: usize = 8_192;

/// Minimum subset size for the PARALLEL coefficient fold; below it the serial fold runs
/// (exact historical bytes, no fork-join). See the threshold rationale at the call site.
const LEAF_COEFF_PAR_MIN_ROWS: usize = 262_144;

/// Two-exponential per-leaf coefficients for the TWEEDIE closed-form deviance (ρ ∈ (1,2)):
/// with `μ = e^{base+v}`, `p1 = 1−ρ < 0`, `p2 = 2−ρ > 0`, the weighted deviance factors as
/// `D(v) = K + Σ_l (e^{p1·v_l}·M1_l + e^{p2·v_l}·M2_l)` with
/// `M1_l = −Σ 2 w y e^{p1·base}/p1 ≥ 0`, `M2_l = Σ 2 w e^{p2·base}/p2 > 0`,
/// `K = Σ 2 w y^{2−ρ}/(p1·p2)` (0 at y = 0 — the compound-Poisson mass point).
struct TweedieLeafCoeffs {
    m1: [f64; MAX_LEAVES],
    m2: [f64; MAX_LEAVES],
    k: f64,
    max_abs_base: f32,
    sum_w: f64,
    p1: f64,
    p2: f64,
}

fn tweedie_leaf_coeffs_serial(
    rho: f32,
    y_sub: &[f32],
    w_sub: &[f32],
    base_sub: &[f32],
    memberships: &[u8],
) -> Result<TweedieLeafCoeffs, PbError> {
    if y_sub.len() != w_sub.len()
        || y_sub.len() != base_sub.len()
        || y_sub.len() != memberships.len()
    {
        return Err(PbError::Internal {
            what: "tweedie closed-form coeff length mismatch".into(),
        });
    }
    let p1 = 1.0 - f64::from(rho);
    let p2 = 2.0 - f64::from(rho);
    let mut m1 = [0.0_f64; MAX_LEAVES];
    let mut m2 = [0.0_f64; MAX_LEAVES];
    let mut k = 0.0_f64;
    let mut max_abs_base = 0.0_f32;
    let mut sum_w = 0.0_f64;
    for (((&yi, &wi), &bi), &leaf_u8) in y_sub.iter().zip(w_sub).zip(base_sub).zip(memberships) {
        // Domain guards mirror `Tweedie::deviance` (y >= 0 — zeros are in-domain).
        if !yi.is_finite() || yi < 0.0 {
            return Err(PbError::InvalidInput {
                what: format!("tweedie: y must be finite and >= 0, got {yi}"),
            });
        }
        if !bi.is_finite() {
            return Err(PbError::InvalidInput {
                what: format!("tweedie: base raw must be finite, got {bi}"),
            });
        }
        if !wi.is_finite() || wi < 0.0 {
            return Err(PbError::InvalidInput {
                what: format!("tweedie: weight must be finite and >= 0, got {wi}"),
            });
        }
        let leaf = usize::from(leaf_u8);
        let w = f64::from(wi);
        let yy = f64::from(yi);
        let b = f64::from(bi);
        // |p1|, |p2| < 1 and the caller's ±30 guard bounds |base + v|, so these f64 exps are
        // safe; they are fuller-precision than the per-row f32 `clamp_exp(p·F)`, re-aligned by
        // the final f32 cast in the closed deviance (same argument as the Poisson/Gamma folds).
        let e1 = (p1 * b).exp();
        let e2 = (p2 * b).exp();
        *m1.get_mut(leaf).ok_or_else(|| PbError::Internal {
            what: "tweedie closed-form M1 leaf escaped".into(),
        })? -= 2.0 * w * yy * e1 / p1;
        *m2.get_mut(leaf).ok_or_else(|| PbError::Internal {
            what: "tweedie closed-form M2 leaf escaped".into(),
        })? += 2.0 * w * e2 / p2;
        let y_term = if yy > 0.0 { (p2 * yy.ln()).exp() } else { 0.0 };
        k += 2.0 * w * y_term / (p1 * p2);
        max_abs_base = max_abs_base.max(bi.abs());
        sum_w += w;
    }
    Ok(TweedieLeafCoeffs {
        m1,
        m2,
        k,
        max_abs_base,
        sum_w,
        p1,
        p2,
    })
}

/// Tweedie analog of [`chunked_leaf_coeffs`] — same fixed-chunk threshold/combine pattern
/// (serial below `LEAF_COEFF_PAR_MIN_ROWS`, chunk-order f64 combine above, exact max),
/// duplicated rather than trait-abstracted because the two coeff shapes differ.
fn tweedie_leaf_coeffs(
    rho: f32,
    y_sub: &[f32],
    w_sub: &[f32],
    base_sub: &[f32],
    memberships: &[u8],
) -> Result<TweedieLeafCoeffs, PbError> {
    if y_sub.len() < LEAF_COEFF_PAR_MIN_ROWS {
        return tweedie_leaf_coeffs_serial(rho, y_sub, w_sub, base_sub, memberships);
    }
    let parts: Result<Vec<TweedieLeafCoeffs>, PbError> = y_sub
        .par_chunks(LEAF_COEFF_CHUNK_ROWS)
        .zip(w_sub.par_chunks(LEAF_COEFF_CHUNK_ROWS))
        .zip(base_sub.par_chunks(LEAF_COEFF_CHUNK_ROWS))
        .zip(memberships.par_chunks(LEAF_COEFF_CHUNK_ROWS))
        .with_min_len(crate::sched::min_chunks_per_task())
        .map(|(((yc, wc), bc), mc)| tweedie_leaf_coeffs_serial(rho, yc, wc, bc, mc))
        .collect();
    let mut out = TweedieLeafCoeffs {
        m1: [0.0; MAX_LEAVES],
        m2: [0.0; MAX_LEAVES],
        k: 0.0,
        max_abs_base: 0.0,
        sum_w: 0.0,
        p1: 1.0 - f64::from(rho),
        p2: 2.0 - f64::from(rho),
    };
    for part in parts? {
        for (acc, p) in out.m1.iter_mut().zip(part.m1.iter()) {
            *acc += *p;
        }
        for (acc, p) in out.m2.iter_mut().zip(part.m2.iter()) {
            *acc += *p;
        }
        out.k += part.k;
        out.max_abs_base = out.max_abs_base.max(part.max_abs_base);
        out.sum_w += part.sum_w;
    }
    Ok(out)
}

/// `D(v)` from Tweedie coefficients, f32-cast on the same grid as `Tweedie::deviance`.
fn tweedie_closed_deviance(coeffs: &TweedieLeafCoeffs, leaves: &[f32], n_leaves: usize) -> f64 {
    let mut d = coeffs.k;
    for leaf in 0..n_leaves {
        let v = f64::from(leaves.get(leaf).copied().unwrap_or(0.0));
        let m1 = coeffs.m1.get(leaf).copied().unwrap_or(0.0);
        let m2 = coeffs.m2.get(leaf).copied().unwrap_or(0.0);
        d += (coeffs.p1 * v).exp() * m1 + (coeffs.p2 * v).exp() * m2;
    }
    f64::from(d as f32)
}

/// The closed-form refine's coefficient carrier: one-exponential template (Poisson s=+1,
/// Gamma s=−1) or the Tweedie two-exponential form. Guards and the deviance evals dispatch
/// through this; the tier-2 Newton template exists only for the one-exp (Poisson) shape.
enum ClosedCoeffs {
    OneExp(PoissonLeafCoeffs),
    TwoExp(TweedieLeafCoeffs),
}

impl ClosedCoeffs {
    fn sum_w(&self) -> f64 {
        match self {
            ClosedCoeffs::OneExp(c) => c.sum_w,
            ClosedCoeffs::TwoExp(c) => c.sum_w,
        }
    }
    fn max_abs_base(&self) -> f32 {
        match self {
            ClosedCoeffs::OneExp(c) => c.max_abs_base,
            ClosedCoeffs::TwoExp(c) => c.max_abs_base,
        }
    }
    fn deviance(&self, leaves: &[f32], n_leaves: usize) -> f64 {
        match self {
            ClosedCoeffs::OneExp(c) => loglink_closed_deviance(c, leaves, n_leaves),
            ClosedCoeffs::TwoExp(c) => tweedie_closed_deviance(c, leaves, n_leaves),
        }
    }
}

fn poisson_leaf_coeffs(
    y_sub: &[f32],
    w_sub: &[f32],
    base_sub: &[f32],
    memberships: &[u8],
) -> Result<PoissonLeafCoeffs, PbError> {
    chunked_leaf_coeffs(
        y_sub,
        w_sub,
        base_sub,
        memberships,
        poisson_leaf_coeffs_serial,
    )
}

fn gamma_leaf_coeffs(
    y_sub: &[f32],
    w_sub: &[f32],
    base_sub: &[f32],
    memberships: &[u8],
) -> Result<PoissonLeafCoeffs, PbError> {
    chunked_leaf_coeffs(
        y_sub,
        w_sub,
        base_sub,
        memberships,
        gamma_leaf_coeffs_serial,
    )
}

/// Shared serial-below-threshold / fixed-chunk-parallel-above dispatch for the per-leaf
/// coefficient folds (P2). Objective-independent: the combine (per-leaf adds, k/sum_w adds,
/// max of max_abs_base, exp_sign carry-through) is the same for every log-link fold.
fn chunked_leaf_coeffs<F>(
    y_sub: &[f32],
    w_sub: &[f32],
    base_sub: &[f32],
    memberships: &[u8],
    serial: F,
) -> Result<PoissonLeafCoeffs, PbError>
where
    F: Fn(&[f32], &[f32], &[f32], &[u8]) -> Result<PoissonLeafCoeffs, PbError> + Sync,
{
    if y_sub.len() != w_sub.len()
        || y_sub.len() != base_sub.len()
        || y_sub.len() != memberships.len()
    {
        return Err(PbError::Internal {
            what: "closed-form coeff length mismatch".into(),
        });
    }
    // The exp pass is the compute-bound heart of leaf_refine at real tree counts (measured
    // 5.7-7.4s of a 6.8-8.8s leaf_refine on a 1-bag 406k ES fit; the parallel fold halves it).
    // Below the threshold the serial fold runs unchanged — small subsets keep their exact
    // historical bytes AND avoid the fork-join tax that made a 200k/300-tree probe ~10% SLOWER
    // when this parallelized unconditionally (P1.1's small-task disease). The threshold is a
    // fixed CONSTANT — deliberately independent of `sched::min_chunks_per_task`, whose contract
    // is that it can never change a computed value; a threshold reading it would let the env
    // knob flip serial-vs-parallel and thus the (ratified, ~1e-12) fold grouping.
    if y_sub.len() < LEAF_COEFF_PAR_MIN_ROWS {
        return serial(y_sub, w_sub, base_sub, memberships);
    }
    let parts: Result<Vec<PoissonLeafCoeffs>, PbError> = y_sub
        .par_chunks(LEAF_COEFF_CHUNK_ROWS)
        .zip(w_sub.par_chunks(LEAF_COEFF_CHUNK_ROWS))
        .zip(base_sub.par_chunks(LEAF_COEFF_CHUNK_ROWS))
        .zip(memberships.par_chunks(LEAF_COEFF_CHUNK_ROWS))
        .with_min_len(crate::sched::min_chunks_per_task())
        .map(|(((yc, wc), bc), mc)| serial(yc, wc, bc, mc))
        .collect();
    // Combine in CHUNK ORDER: f64 adds grouped per chunk (the declared drift), max is exact.
    let mut out = PoissonLeafCoeffs {
        m: [0.0; MAX_LEAVES],
        a: [0.0; MAX_LEAVES],
        k: 0.0,
        max_abs_base: 0.0,
        sum_w: 0.0,
        exp_sign: 0.0, // overwritten below from the (identical) per-chunk value
    };
    for part in parts? {
        out.exp_sign = part.exp_sign;
        for (acc, p) in out.m.iter_mut().zip(part.m.iter()) {
            *acc += *p;
        }
        for (acc, p) in out.a.iter_mut().zip(part.a.iter()) {
            *acc += *p;
        }
        out.k += part.k;
        out.max_abs_base = out.max_abs_base.max(part.max_abs_base);
        out.sum_w += part.sum_w;
    }
    Ok(out)
}

fn poisson_leaf_coeffs_serial(
    y_sub: &[f32],
    w_sub: &[f32],
    base_sub: &[f32],
    memberships: &[u8],
) -> Result<PoissonLeafCoeffs, PbError> {
    let mut m = [0.0_f64; MAX_LEAVES];
    let mut a = [0.0_f64; MAX_LEAVES];
    let mut k = 0.0_f64;
    let mut max_abs_base = 0.0_f32;
    let mut sum_w = 0.0_f64;
    for (((&yi, &wi), &bi), &leaf_u8) in y_sub.iter().zip(w_sub).zip(base_sub).zip(memberships) {
        // Domain guards mirror `Poisson::{deviance, grad_hess_and_deviance}` (trial-invariant, so
        // validated once here rather than on every per-trial deviance call the generic path makes).
        if !yi.is_finite() || yi < 0.0 {
            return Err(PbError::InvalidInput {
                what: format!("poisson: y must be finite and >= 0, got {yi}"),
            });
        }
        if !bi.is_finite() {
            return Err(PbError::InvalidInput {
                what: format!("poisson: base raw must be finite, got {bi}"),
            });
        }
        if !wi.is_finite() || wi < 0.0 {
            return Err(PbError::InvalidInput {
                what: format!("poisson: weight must be finite and >= 0, got {wi}"),
            });
        }
        let leaf = usize::from(leaf_u8);
        let w = f64::from(wi);
        let yy = f64::from(yi);
        let b = f64::from(bi);
        // eb = exp(base): the caller's guard ensures |base| ≤ 30 before these coeffs are consumed,
        // so this f64 exp neither overflows nor needs `clamp_exp`; it is a fuller-precision exp than
        // the per-row f32 `clamp_exp` (they agree to ~1e-7, re-aligned by the final f32 deviance
        // cast in `loglink_closed_deviance`).
        let eb = b.exp();
        *m.get_mut(leaf).ok_or_else(|| PbError::Internal {
            what: "poisson closed-form M leaf escaped".into(),
        })? += 2.0 * w * eb;
        *a.get_mut(leaf).ok_or_else(|| PbError::Internal {
            what: "poisson closed-form A leaf escaped".into(),
        })? += 2.0 * w * yy;
        let yln = if yy > 0.0 { yy * yy.ln() } else { 0.0 };
        k += 2.0 * w * (yln - yy * b - yy);
        max_abs_base = max_abs_base.max(bi.abs());
        sum_w += w;
    }
    Ok(PoissonLeafCoeffs {
        m,
        a,
        k,
        max_abs_base,
        sum_w,
        exp_sign: 1.0,
    })
}

/// Gamma analog of [`poisson_leaf_coeffs_serial`]. With `μ = exp(F)`, `F = base + v_l`, the
/// weighted Gamma deviance `2 Σ w (y/μ − 1 − ln(y/μ))` factors per leaf as
/// `D(v) = K + Σ_l (e^{−v_l}·M_l − v_l·A_l)` with `M_l = Σ 2 w y e^{−base}`, `A_l = −Σ 2 w`,
/// `K = Σ 2 w (base − ln y − 1)` — the SAME template as Poisson with `exp_sign = −1`.
/// Domain guards mirror `Gamma::{deviance, init_score}` (y strictly positive).
fn gamma_leaf_coeffs_serial(
    y_sub: &[f32],
    w_sub: &[f32],
    base_sub: &[f32],
    memberships: &[u8],
) -> Result<PoissonLeafCoeffs, PbError> {
    if y_sub.len() != w_sub.len()
        || y_sub.len() != base_sub.len()
        || y_sub.len() != memberships.len()
    {
        return Err(PbError::Internal {
            what: "gamma closed-form coeff length mismatch".into(),
        });
    }
    let mut m = [0.0_f64; MAX_LEAVES];
    let mut a = [0.0_f64; MAX_LEAVES];
    let mut k = 0.0_f64;
    let mut max_abs_base = 0.0_f32;
    let mut sum_w = 0.0_f64;
    for (((&yi, &wi), &bi), &leaf_u8) in y_sub.iter().zip(w_sub).zip(base_sub).zip(memberships) {
        if !yi.is_finite() || yi <= 0.0 {
            return Err(PbError::InvalidInput {
                what: format!("gamma: y must be finite and > 0, got {yi}"),
            });
        }
        if !bi.is_finite() {
            return Err(PbError::InvalidInput {
                what: format!("gamma: base raw must be finite, got {bi}"),
            });
        }
        if !wi.is_finite() || wi < 0.0 {
            return Err(PbError::InvalidInput {
                what: format!("gamma: weight must be finite and >= 0, got {wi}"),
            });
        }
        let leaf = usize::from(leaf_u8);
        let w = f64::from(wi);
        let yy = f64::from(yi);
        let b = f64::from(bi);
        // e^{−base}: the caller's ±30 guard bounds |base + v|, so the f64 exp is safe and
        // fuller-precision than the per-row f32 `clamp_exp`, re-aligned by the final f32 cast
        // in `loglink_closed_deviance` (same argument as the Poisson fold).
        let emb = (-b).exp();
        *m.get_mut(leaf).ok_or_else(|| PbError::Internal {
            what: "gamma closed-form M leaf escaped".into(),
        })? += 2.0 * w * yy * emb;
        *a.get_mut(leaf).ok_or_else(|| PbError::Internal {
            what: "gamma closed-form A leaf escaped".into(),
        })? -= 2.0 * w;
        k += 2.0 * w * (b - yy.ln() - 1.0);
        max_abs_base = max_abs_base.max(bi.abs());
        sum_w += w;
    }
    Ok(PoissonLeafCoeffs {
        m,
        a,
        k,
        max_abs_base,
        sum_w,
        exp_sign: -1.0,
    })
}

/// The Poisson weighted deviance of trial leaf vector `leaves`, in O(#leaves), from precomputed
/// [`PoissonLeafCoeffs`]: `D = K + Σ_l (e^{v_l}·M_l − v_l·A_l)`. f32-cast EXACTLY as
/// `Poisson::deviance` (`finish_deviance`: `acc as f32`) then widened, so the accept comparison
/// sits on the same f32 grid as the per-row deviance — matching it save for a vanishingly rare
/// f32-boundary straddle. The per-leaf accumulation is a fixed `0..n_leaves` order (deterministic).
fn loglink_closed_deviance(coeffs: &PoissonLeafCoeffs, leaves: &[f32], n_leaves: usize) -> f64 {
    let s = coeffs.exp_sign;
    let mut d = coeffs.k;
    for leaf in 0..n_leaves {
        let v = f64::from(leaves.get(leaf).copied().unwrap_or(0.0));
        let m = coeffs.m.get(leaf).copied().unwrap_or(0.0);
        let a = coeffs.a.get(leaf).copied().unwrap_or(0.0);
        // `s * v` with s = +1.0 is IEEE-exact, so the Poisson path computes byte-identically
        // to the pre-Gamma `v.exp() * m - v * a`.
        d += (s * v).exp() * m - v * a;
    }
    f64::from(d as f32)
}

/// Closed-form-DEVIANCE leaf refinement for LOG-LINK losses (Poisson today; Gamma/Tweedie once
/// their per-leaf closed forms are derived + tested). The log-link deviance factors as
/// `D(v) = K − Σ_l v_l·A_l + Σ_l e^{v_l}·M_l` ([`poisson_leaf_coeffs`]), so the per-trial O(rows)
/// exp+ln deviance re-folds of the generic line search (`refine_tree_leaves_after_grow`) collapse
/// to O(#leaves). Precompute cost: ONE exp pass (`eb_i = exp(base_i)`) + one contiguous fold.
///
/// TIER 1 ([`Config::refine_closed_form_tier2`] = false): the per-step `grad_hess` + per-leaf
/// aggregate that produce the leaf UPDATES are KEPT VERBATIM (same f32 path as the generic refine),
/// so the proposed deltas — hence the accepted leaves, the model, and the early-stop trajectory —
/// are byte-identical; only `refine.init_dev` / `refine.backtrack_eval` turn from an O(rows)
/// transcendental fold into the O(#leaves) closed form. The closed deviance is f32-cast exactly as
/// `Poisson::deviance`, so each accept comparison matches the per-row deviance save for a rare
/// f32-boundary straddle (the documented SquaredError-path exactness class).
///
/// TIER 2 ([`Config::refine_closed_form_tier2`] = true, the default): also derive the per-step deltas
/// from the closed-form per-leaf `G_l = e^{v_l}·M_l/2 − A_l/2` and `H_l = e^{v_l}·M_l/2` (verified
/// against Poisson `grad_hess`: `g = w(μ−y)`, `h = w·μ`; note `H_l = Σ_{i∈l} w·μ` omits the per-row
/// hessian floor), eliminating the remaining per-row passes. Leaves may then drift ~1e-7, so it is
/// NOT bit-identical — set the field to `false` to keep the byte-identical Tier-1 path.
///
/// Returns `Ok(true)` when it fully handled the refine; `Ok(false)` when it DECLINES — the loss has
/// no closed form here, the step is uncapped (can't bound `v`), or `max|base| + v_budget` could
/// reach the ±30 exp clamp so the factoring `exp(base+v) = eb·e^v` would diverge from the clamped
/// per-row path — in which case the caller runs the exact generic per-row line search. On a decline
/// `tree` is untouched (all guards precede any mutation).
#[allow(clippy::too_many_arguments)]
fn refine_leaves_loglink_closed(
    config: &Config,
    loss: &dyn crate::loss::Loss,
    y_sub: &[f32],
    w_sub: &[f32],
    base_sub: &[f32],
    memberships: &[u8],
    monotone: Option<&[Option<MonoSign>]>,
    tree: &mut ObliviousTree,
    grow_cfg: &GrowConfig<'_>,
    n_leaves: usize,
) -> Result<bool, PbError> {
    // Only losses whose per-leaf closed form is derived AND tested here: Poisson + Gamma +
    // Tweedie (2026-07-16, objective-parity campaign). Tweedie needs its ρ from the tag.
    let tag = loss.objective_tag();
    let loss_id = tag.loss;
    if !matches!(
        loss_id,
        crate::loss::LossId::Poisson | crate::loss::LossId::Gamma | crate::loss::LossId::Tweedie
    ) {
        return Ok(false);
    }
    // A Tweedie tag without ρ cannot build coefficients — decline to the generic path.
    if matches!(loss_id, crate::loss::LossId::Tweedie) && tag.tweedie_rho.is_none() {
        return Ok(false);
    }
    // Bound |v_l| over the WHOLE refine: `incremental_leaf_delta` clamps each Newton step to
    // ±max_delta_step then scales by lr, and the monotone projection only pulls a leaf toward an
    // existing sibling — so |v_l| ≤ max|initial leaf| + steps·|lr·max_delta_step|. An uncapped step
    // cannot be bounded ⇒ decline.
    let Some(dstep) = grow_cfg.max_delta_step else {
        return Ok(false);
    };
    let max_abs_leaf = tree
        .leaves
        .iter()
        .take(n_leaves)
        .fold(0.0_f32, |acc, &v| acc.max(v.abs()));
    let v_budget = f64::from(max_abs_leaf)
        + f64::from(config.leaf_refine_steps) * grow_cfg.lr.abs() * dstep.abs();

    let coeffs = prof::timed("refine.coeff_fold", || -> Result<ClosedCoeffs, PbError> {
        match loss_id {
            crate::loss::LossId::Gamma => Ok(ClosedCoeffs::OneExp(gamma_leaf_coeffs(
                y_sub,
                w_sub,
                base_sub,
                memberships,
            )?)),
            crate::loss::LossId::Tweedie => {
                let rho = tag.tweedie_rho.ok_or_else(|| PbError::Internal {
                    what: "tweedie closed refine reached without rho".into(),
                })?;
                Ok(ClosedCoeffs::TwoExp(tweedie_leaf_coeffs(
                    rho,
                    y_sub,
                    w_sub,
                    base_sub,
                    memberships,
                )?))
            }
            _ => Ok(ClosedCoeffs::OneExp(poisson_leaf_coeffs(
                y_sub,
                w_sub,
                base_sub,
                memberships,
            )?)),
        }
    })?;
    if coeffs.sum_w() <= 0.0 {
        return Err(PbError::InvalidInput {
            what: match loss_id {
                crate::loss::LossId::Gamma => "gamma deviance: all-zero weights".into(),
                crate::loss::LossId::Tweedie => "tweedie deviance: all-zero weights".into(),
                _ => "poisson deviance: all-zero weights".into(),
            },
        });
    }
    // Exp-clamp guard: if the factored exponent `base + v` could reach the ±30 clamp for any trial,
    // the factored deviance would diverge from the clamped per-row deviance — decline to the generic
    // per-row path (which applies the clamp row by row). O(1): uses the precomputed max|base|.
    if f64::from(coeffs.max_abs_base()) + v_budget > EXP_CLAMP_F64 {
        return Ok(false);
    }

    // Dense subset-raw scratch for the KEPT per-row grad_hess (Tier 1): `trial_raw_sub[i] =
    // base_sub[i] + tree.leaves[memberships[i]]`, refilled from the CURRENT leaves at the top of
    // each step so grad_hess reads exactly the accepted leaves — bit-identical to the generic path's
    // per-step raw state. Tier 2 never reads or writes it (no per-row pass at all), so materializing
    // `base_sub.to_vec()` there was a dead O(rows) alloc+copy every call under the default
    // (`refine_closed_form_tier2 = true`) config; only allocate it for Tier 1.
    // Poisson AND Gamma take Tier 2 under the default `refine_closed_form_tier2 = true` (Gamma
    // ratified by Ralph 2026-07-19 after its multi-split no-harm battery: 6 real severity datasets
    // × k=5 paired, held-out gamma deviance shows NO systematic effect — 23 worse / 23 better,
    // sign-test p=1.0, no dataset significantly worse, the largest (allstate_sev) slightly better;
    // the ~1e-7-per-leaf drift is an unbiased ±0.1..0.6% per-fold wobble, not a bias. Wall:
    // leaf_refine −80% (the per-step grad_hess + per-leaf aggregate passes are eliminated —
    // deltas come straight from the closed form), −24% single-bag / −41% fremotor_prem full fit.
    // NOT bit-identical to the pre-Tier-2 output — set `refine_closed_form_tier2 = false` for the
    // byte-identical Tier-1 path. Tweedie stays Tier 1 regardless (its TwoExp coeffs never match
    // the OneExp Tier-2 branch below).
    let tier2 = config.refine_closed_form_tier2
        && matches!(
            loss_id,
            crate::loss::LossId::Poisson | crate::loss::LossId::Gamma
        );
    let mut trial_raw_sub = if tier2 { Vec::new() } else { base_sub.to_vec() };
    let mut gh = GradHess::default();

    let mut best_deviance = prof::timed("refine.init_dev", || {
        coeffs.deviance(&tree.leaves, n_leaves)
    });
    // §07.6 leaf_refine companion fix: membership (hence row count) is fixed across every step
    // of this call, in BOTH tiers. Gated on `path_smooth > 0` — the (default) inert path never
    // pays for this tally.
    let credibility_counts = if grow_cfg.credibility.path_smooth > 0.0 {
        leaf_row_counts(memberships, n_leaves)?
    } else {
        [0u64; MAX_LEAVES]
    };

    for _ in 0..config.leaf_refine_steps {
        let (g, h) = if let (true, ClosedCoeffs::OneExp(ce)) = (tier2, &coeffs) {
            // Tier 2: per-leaf Newton quantities straight from the closed form (no per-row pass),
            // via the signed template: dD/dv_l = s·e^{s·v_l}·M_l − A_l, so
            // G_l = s·e^{s·v}·M_l/2 − A_l/2 and H_l = e^{s·v}·M_l/2 (d²D/dv² = s²·e^{s·v}·M = H·2).
            // Poisson (s=+1): G = e^v·M/2 − A/2 = Σ w(μ−y), H = Σ w·μ — byte-identical to the
            // pre-Gamma code (×+1.0 is IEEE-exact). Gamma (s=−1, A=−W): G = (W − e^{−v}·B)/2 =
            // Σ w(1 − y·e^{−F}), H = e^{−v}·B/2 = Σ w·y·e^{−F} — matching `Gamma::grad_hess`.
            let sg = ce.exp_sign;
            let mut g = [0.0_f64; MAX_LEAVES];
            let mut h = [0.0_f64; MAX_LEAVES];
            for leaf in 0..n_leaves {
                let v = f64::from(*tree.leaves.get(leaf).ok_or_else(|| PbError::Internal {
                    what: "closed-form leaf value escaped".into(),
                })?);
                let ev_m_half = 0.5 * (sg * v).exp() * ce.m.get(leaf).copied().unwrap_or(0.0);
                *g.get_mut(leaf).ok_or_else(|| PbError::Internal {
                    what: "closed-form g leaf escaped".into(),
                })? = sg * ev_m_half - 0.5 * ce.a.get(leaf).copied().unwrap_or(0.0);
                *h.get_mut(leaf).ok_or_else(|| PbError::Internal {
                    what: "closed-form h leaf escaped".into(),
                })? = ev_m_half;
            }
            (g, h)
        } else {
            // Tier 1: KEEP the generic path's per-step grad_hess + per-leaf aggregate VERBATIM, so
            // the proposed deltas are bit-identical. Refill the dense raw from the CURRENT leaves,
            // then the (subset) grad_hess and the SAME rows-order fold into per-leaf f64 sums.
            fill_leaf_raw_contiguous(&mut trial_raw_sub, base_sub, memberships, &tree.leaves)?;
            prof::timed("refine.grad_hess", || {
                loss.grad_hess(y_sub, &trial_raw_sub, w_sub, &mut gh)
            })?;
            let mut g = [0.0_f64; MAX_LEAVES];
            let mut h = [0.0_f64; MAX_LEAVES];
            prof::timed("refine.aggregate", || -> Result<(), PbError> {
                for (i, &leaf_u8) in memberships.iter().enumerate() {
                    let leaf = usize::from(leaf_u8);
                    *g.get_mut(leaf).ok_or_else(|| PbError::Internal {
                        what: "leaf refinement g leaf escaped".into(),
                    })? += f64::from(*gh.g.get(i).ok_or_else(|| PbError::Internal {
                        what: "leaf refinement gradient row escaped".into(),
                    })?);
                    *h.get_mut(leaf).ok_or_else(|| PbError::Internal {
                        what: "leaf refinement h leaf escaped".into(),
                    })? += f64::from(*gh.h.get(i).ok_or_else(|| PbError::Internal {
                        what: "leaf refinement hessian row escaped".into(),
                    })?);
                }
                Ok(())
            })?;
            (g, h)
        };
        // §07.6 evidence normalization: this step's mean per-row hessian (Poisson/Tweedie stay
        // on Hessian evidence through BOTH the Tier-2 closed form and the Tier-1 fallback above,
        // since `h` means the same Σh either way — see the Tier-2 comment). Recomputed each
        // step; gated on `path_smooth > 0` like the count tally above.
        let h_bar = if grow_cfg.credibility.path_smooth > 0.0 {
            refine_h_bar(&h, n_leaves, memberships.len())
        } else {
            0.0
        };

        let mut delta = [0.0_f32; MAX_LEAVES];
        let mut any_delta = false;
        for leaf in 0..n_leaves {
            let h_leaf = *h.get(leaf).ok_or_else(|| PbError::Internal {
                what: "leaf refinement h lookup escaped".into(),
            })?;
            let step = incremental_leaf_delta(
                *g.get(leaf).ok_or_else(|| PbError::Internal {
                    what: "leaf refinement g lookup escaped".into(),
                })?,
                h_leaf,
                grow_cfg.lambda,
                grow_cfg.l1_leaf,
                grow_cfg.max_delta_step,
                grow_cfg.lr,
            )?;
            let credibility_n = credibility_evidence_n(
                grow_cfg.credibility_evidence,
                h_leaf,
                *credibility_counts
                    .get(leaf)
                    .ok_or_else(|| PbError::Internal {
                        what: "leaf refinement credibility count lookup escaped".into(),
                    })?,
                h_bar,
            );
            let step = shrink_refine_delta(step, credibility_n, grow_cfg.credibility.path_smooth);
            if step.abs() > 1.0e-7 {
                any_delta = true;
            }
            *delta.get_mut(leaf).ok_or_else(|| PbError::Internal {
                what: "leaf refinement delta lookup escaped".into(),
            })? = step;
        }
        if !any_delta {
            break;
        }

        let mut accepted = false;
        let mut scale = 1.0_f32;
        for _ in 0..config.leaf_refine_backtracks {
            let mut trial_leaves = tree.leaves.clone();
            for (leaf_value, delta_value) in trial_leaves.iter_mut().zip(delta.iter()) {
                let value = f64::from(*leaf_value) + f64::from(scale * *delta_value);
                if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX)
                {
                    return Err(PbError::InvalidInput {
                        what: "leaf refinement value is not finite/representable".into(),
                    });
                }
                *leaf_value = value as f32;
            }
            clamp_monotone(
                &mut trial_leaves,
                &tree.splits,
                usize::from(tree.depth),
                monotone,
            )?;
            let deviance = prof::timed("refine.backtrack_eval", || {
                coeffs.deviance(&trial_leaves, n_leaves)
            });
            if deviance < best_deviance {
                tree.leaves = trial_leaves;
                best_deviance = deviance;
                accepted = true;
                break;
            }
            scale *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    Ok(true)
}

// Test-only kill-switch for the log-link closed-form fast path: when set on the current thread,
// `refine_tree_leaves_after_grow` forces the generic per-row line search, so a fit with the closed
// form ON vs OFF can be compared for byte-identical leaves. Always `false` outside tests (the
// closed path is always attempted in production).
#[cfg(test)]
thread_local! {
    static DISABLE_LOGLINK_CLOSED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn loglink_closed_disabled() -> bool {
    DISABLE_LOGLINK_CLOSED.with(std::cell::Cell::get)
}

#[cfg(not(test))]
#[inline(always)]
fn loglink_closed_disabled() -> bool {
    false
}

/// Run `f` with the log-link closed-form fast path DISABLED on this thread (test A/B helper).
#[cfg(test)]
fn with_loglink_closed_disabled<T>(f: impl FnOnce() -> T) -> T {
    DISABLE_LOGLINK_CLOSED.with(|c| c.set(true));
    let out = f();
    DISABLE_LOGLINK_CLOSED.with(|c| c.set(false));
    out
}

fn incremental_leaf_delta(
    g: f64,
    h: f64,
    lambda: f64,
    l1_leaf: f64,
    max_delta_step: Option<f64>,
    lr: f64,
) -> Result<f32, PbError> {
    let denom = h + lambda;
    let g = if l1_leaf <= 0.0 {
        g
    } else {
        g.signum() * (g.abs() - l1_leaf).max(0.0)
    };
    let step = if denom > 0.0 { -g / denom } else { 0.0 };
    // §05.6: `max_delta_step` clamps the full-precision Newton step BEFORE the learning rate.
    let step = match max_delta_step {
        Some(d) => step.clamp(-d, d),
        None => step,
    };
    // Match `leaf_values` (split.rs): the refinement step is `lr`-scaled, so multi-step
    // Newton refines the leaf WITHIN its shrinkage budget instead of driving it to the
    // `lr = 1.0` optimum — the latter silently un-shrinks every leaf and overfits (the
    // Armijo guard cannot catch it because the un-shrunk step still lowers TRAIN deviance).
    let step = lr * step;
    if !step.is_finite() || step < f64::from(f32::MIN) || step > f64::from(f32::MAX) {
        return Err(PbError::InvalidInput {
            what: "leaf refinement step is not finite/representable".into(),
        });
    }
    Ok(step as f32)
}

/// §07.6 `leaf_refine`-erosion companion fix. `incremental_leaf_delta` above computes a plain,
/// credibility-unaware Newton step: it walks a leaf toward its OWN unshrunk deviance optimum
/// with no pull toward the parent, silently eroding whatever `apply_path_smooth` (`split.rs`)
/// blended in at grow time (see the design note this implements, `agentH_cred_lambda.md` §2).
///
/// Unlike grow time, the refine call sites only ever see the flat per-leaf `(g, h, count)`
/// arrays for the CURRENT round — the oblivious tree's internal-node parent-chain values
/// `apply_path_smooth` recurses over (§07.6) are not reconstructed here, so a full re-blend
/// against the actual parent is not available at this call site. Instead this shrinks the
/// INCREMENT itself by the same Bühlmann factor `Z = n/(n+k)`: every refine step moves the leaf
/// only a `Z` fraction of the distance it otherwise would, keeping it close to its
/// grow-time-blended value across the field's realistic (small) `leaf_refine_steps` budgets.
/// This is an approximation, not an exact invariant — scaling every step by a FIXED `Z` does
/// not change the fixed point an unbounded number of steps would converge to (still the
/// unshrunk optimum), it only damps the rate of approach, so it "preserves the grow-time blend
/// in expectation" for a bounded step budget rather than exactly for any budget.
///
/// No-op (bit-identical: returns `step` untouched, not merely numerically close) whenever
/// `path_smooth <= 0.0` — the shared inert-by-default contract every `path_smooth` call site
/// upholds.
#[inline]
fn shrink_refine_delta(step: f32, credibility_n: f64, path_smooth: f32) -> f32 {
    if path_smooth > 0.0 {
        let z = credibility_n / (credibility_n + f64::from(path_smooth));
        (f64::from(step) * z) as f32
    } else {
        step
    }
}

/// §07.6 evidence normalization (2026-07-23 amendment): the mean per-row hessian over ALL of
/// this refine call's rows, feeding `credibility_evidence_n`'s `h_bar` so `Hessian` evidence is
/// an "effective row count" on `path_smooth`'s pseudo-count scale, not a raw (possibly
/// hundreds-to-thousands-per-row, e.g. exposure-weighted Tweedie) mass — see
/// `credibility_evidence_n`'s doc (`constraints.rs`) for why this is mandatory. Built from the
/// SAME per-leaf `h` sums the step already computed for the Newton step itself (no new fold) and
/// `total_rows` (the call site's own `memberships.len()`/`rows.len()`, already in scope).
#[inline]
fn refine_h_bar(h: &[f64], n_leaves: usize, total_rows: usize) -> f64 {
    if total_rows == 0 {
        return 0.0;
    }
    let h_total: f64 = h.iter().take(n_leaves).sum();
    h_total / (total_rows as f64)
}

/// Per-leaf row counts tallied once from a FIXED membership assignment (leaf refine only ever
/// moves leaf VALUES, never structure, so membership — hence these counts — is invariant across
/// every step of one refine call). Only called when `path_smooth > 0`; the leaf-refine call
/// sites gate this so the tally costs nothing on the (default) inert path.
#[inline]
fn leaf_row_counts(memberships: &[u8], n_leaves: usize) -> Result<[u64; MAX_LEAVES], PbError> {
    let mut count = [0u64; MAX_LEAVES];
    for &leaf_u8 in memberships {
        let leaf = usize::from(leaf_u8);
        // Was a silent `if leaf < n_leaves` skip. At depth <= 3 that was unreachable, but a
        // lifted tree's leaf ids run to 2^depth-1, so a stale membership would have been
        // dropped from the tally and quietly understated the credibility evidence. Fail closed.
        if leaf >= n_leaves {
            return Err(PbError::Internal {
                what: format!("leaf membership {leaf} outside 0..{n_leaves} in leaf_row_counts"),
            });
        }
        *count.get_mut(leaf).ok_or_else(|| PbError::Internal {
            what: "leaf_row_counts index escaped the leaf array".into(),
        })? += 1;
    }
    Ok(count)
}

/// Gather grow's absolute-indexed `leaf_of_row` into the compact `rows`-ordered membership vector
/// the line search consumes. grow sets `leaf_of_row[r]` with the SAME canonical `low_bit` the tree
/// walk applies, so `leaf_of_row[rows[i]]` is bit-identical to `tree_memberships_for_rows(...)[i]`
/// — but free of the per-row tree re-walk. Sound only when grow saw exactly `rows` (no subsample);
/// the caller gates on `sampled_rows.len() == train_rows.len()`. The `< n_leaves` guard catches a
/// stale/under-populated map (e.g. a row grow never visited).
fn gather_memberships(
    leaf_of_row: &[u8],
    rows: &[u32],
    n_leaves: usize,
) -> Result<Vec<u8>, PbError> {
    let mut out = Vec::new();
    out.try_reserve_exact(rows.len())
        .map_err(|_| PbError::Internal {
            what: "leaf refinement membership gather allocation failed".into(),
        })?;
    for &r in rows {
        let leaf = *leaf_of_row
            .get(r as usize)
            .ok_or_else(|| PbError::Internal {
                what: "leaf refinement precomputed membership row escaped".into(),
            })?;
        if usize::from(leaf) >= n_leaves {
            return Err(PbError::Internal {
                what: "leaf refinement precomputed membership escaped depth".into(),
            });
        }
        out.push(leaf);
    }
    Ok(out)
}

fn tree_memberships_for_rows(
    tree: &ObliviousTree,
    columns: &[&[u8]],
    rows: &[u32],
    n_leaves: usize,
) -> Result<Vec<u8>, PbError> {
    let mut out = Vec::with_capacity(rows.len());
    for &row in rows {
        let leaf = tree_leaf_index_for_row_with_columns(tree, columns, row as usize)?;
        if leaf >= n_leaves {
            return Err(PbError::Internal {
                what: "leaf refinement membership escaped depth".into(),
            });
        }
        out.push(u8::try_from(leaf).map_err(|_| PbError::Internal {
            what: "leaf refinement membership exceeded u8".into(),
        })?);
    }
    Ok(out)
}

/// Tree-walk reconstruction of `raw = base + leaf(row)`. Superseded in production by the
/// membership-based [`apply_membership_leaves`] (no per-row walk); retained as the independent
/// reference the equality test checks the fast path against.
#[cfg(test)]
fn raw_with_tree_leaves(
    base_raw: &[f32],
    tree: &ObliviousTree,
    columns: &[&[u8]],
    leaves: &[f32],
) -> Result<Vec<f32>, PbError> {
    let mut out = Vec::with_capacity(base_raw.len());
    for (row, &base) in base_raw.iter().enumerate() {
        let leaf = tree_leaf_index_for_row_with_columns(tree, columns, row)?;
        let value = f64::from(base)
            + f64::from(*leaves.get(leaf).ok_or_else(|| PbError::Internal {
                what: "leaf refinement leaf lookup escaped".into(),
            })?);
        if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
            return Err(PbError::InvalidInput {
                what: "leaf refinement raw is not finite/representable".into(),
            });
        }
        out.push(value as f32);
    }
    Ok(out)
}

/// Overwrite `out[rows[i]] = base_raw[rows[i]] + leaves[memberships[i]]` using the precomputed leaf
/// memberships — NO per-row tree re-walk. A tree's contribution to `raw` is exactly its leaf value, so
/// for the leaf-refinement line search (whose only per-trial change is the 8 leaf VALUES) this produces
/// values bit-identical to a full tree-walk reconstruction on `rows`, far cheaper. Entries outside `rows` are
/// left untouched (they are never read: the line search evaluates deviance only on `rows`).
fn apply_membership_leaves(
    out: &mut [f32],
    base_raw: &[f32],
    rows: &[u32],
    memberships: &[u8],
    leaves: &[f32],
) -> Result<(), PbError> {
    for (&r, &leaf) in rows.iter().zip(memberships) {
        let ru = r as usize;
        let base = *base_raw.get(ru).ok_or_else(|| PbError::Internal {
            what: "leaf refinement base_raw row escaped".into(),
        })?;
        let lv = *leaves
            .get(usize::from(leaf))
            .ok_or_else(|| PbError::Internal {
                what: "leaf refinement membership leaf escaped".into(),
            })?;
        let value = f64::from(base) + f64::from(lv);
        if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
            return Err(PbError::InvalidInput {
                what: "leaf refinement raw is not finite/representable".into(),
            });
        }
        *out.get_mut(ru).ok_or_else(|| PbError::Internal {
            what: "leaf refinement out row escaped".into(),
        })? = value as f32;
    }
    Ok(())
}

/// Gather `src[rows[i]]` into a fresh DENSE buffer (subset in `rows` order), bounds-checked
/// and `try_reserve`d. Used once per tree by the leaf-refine line search to lift the
/// trial-invariant slices (y, weight, base raw) out of the per-trial deviance.
fn gather_rows(src: &[f32], rows: &[u32]) -> Result<Vec<f32>, PbError> {
    let mut out = Vec::new();
    out.try_reserve_exact(rows.len())
        .map_err(|_| PbError::Internal {
            what: "leaf refinement subset gather allocation failed".into(),
        })?;
    for &r in rows {
        out.push(*src.get(r as usize).ok_or_else(|| PbError::Internal {
            what: "leaf refinement subset gather row escaped".into(),
        })?);
    }
    Ok(out)
}

/// Dense (contiguous) twin of [`apply_membership_leaves`]: write
/// `out_sub[i] = base_sub[i] + leaves[memberships[i]]` over a packed subset buffer (no
/// scatter), with the SAME `f64` add + finite/representable check. When
/// `base_sub[i] == base_raw[rows[i]]` the result is **bit-identical** to gathering
/// `apply_membership_leaves`'s output over `rows`, so feeding it to `deviance` reproduces
/// the former `apply_membership_leaves` + `deviance_for_rows` value exactly — while keeping
/// the deviance fold over contiguous slices (vectorizable), unlike a scattered direct-index fold.
fn fill_leaf_raw_contiguous(
    out_sub: &mut [f32],
    base_sub: &[f32],
    memberships: &[u8],
    leaves: &[f32],
) -> Result<(), PbError> {
    if out_sub.len() != base_sub.len() || out_sub.len() != memberships.len() {
        return Err(PbError::Internal {
            what: "leaf refinement contiguous fill length mismatch".into(),
        });
    }
    for ((slot, &base), &leaf) in out_sub.iter_mut().zip(base_sub).zip(memberships) {
        let lv = *leaves
            .get(usize::from(leaf))
            .ok_or_else(|| PbError::Internal {
                what: "leaf refinement membership leaf escaped".into(),
            })?;
        let value = f64::from(base) + f64::from(lv);
        if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
            return Err(PbError::InvalidInput {
                what: "leaf refinement raw is not finite/representable".into(),
            });
        }
        *slot = value as f32;
    }
    Ok(())
}

/// Add a freshly-grown tree's contribution to every row's raw score (spec §06.6
/// sample→leaf update). Scores ALL rows (not just the round's subsample) so the next
/// round's gradients are correct everywhere. Panic-free; uses the canonical low bit.
/// The ≤8 per-leaf inverse-link multipliers `e^{leaf_value}` for the incremental-mu update — a leaf
/// delta `Δ` on `raw` maps to `mu *= e^{Δ}`. Computed ONCE per tree (8 `exp`s) so the row loop is a
/// single multiply. Returns all-ones (and skips the `exp`s) when `want` is false (the exact path).
fn leaf_multipliers(tree: &ObliviousTree, want: bool) -> [f64; MAX_LEAVES] {
    // `zip` stops at the shorter of the two, and `tree.leaves` is `leaf_slots(depth) >= 2^depth`
    // long, so every REACHED leaf is covered; the unreached tail keeps the neutral 1.0.
    let mut m = [1.0_f64; MAX_LEAVES];
    if want {
        for (slot, &lv) in m.iter_mut().zip(tree.leaves.iter()) {
            *slot = f64::from(lv).exp();
        }
    }
    m
}

fn update_raw(
    raw: &mut [f32],
    // Optional log-link inverse-link cache maintained in LOCKSTEP with `raw` (Task-A incremental mu,
    // gated on `Config::incremental_mu`): `mu[r] *= e^{Δ}` for the SAME per-row raw delta `Δ` added to
    // `raw[r]`, through the IDENTICAL control flow, so the cache tracks `exp(raw[r])`. `None` on the
    // exact path — the raw arithmetic is then byte-for-byte the original (the incremental-mu-off
    // bit-compare reference).
    mut mu: Option<&mut [f64]>,
    x: &BinnedMatrix,
    tree: &ObliviousTree,
    // grow's per-row leaf map; `Some` only when grow saw EVERY row of `raw` (full sample, no
    // validation split — the caller gates on `sampled_rows.len() == x.n_rows`). Reused to add
    // `tree.leaves[leaf_of_row[r]]` directly — byte-identical to the walk (grow set the map with
    // the SAME canonical `low_bit` the walk applies, and refinement changed only leaf VALUES, not
    // memberships) — skipping the per-row tree re-walk.
    precomputed_leaf_of_row: Option<&[u8]>,
) -> Result<(), PbError> {
    let leaf_mult = leaf_multipliers(tree, mu.is_some());
    if let Some(leaf_of_row) = precomputed_leaf_of_row {
        for (r, slot) in raw.iter_mut().enumerate() {
            let leaf = *leaf_of_row.get(r).ok_or_else(|| PbError::Internal {
                what: "update_raw precomputed membership row escaped".into(),
            })?;
            let previous = *slot;
            *slot += *tree
                .leaves
                .get(usize::from(leaf))
                .ok_or_else(|| PbError::Internal {
                    what: "update_raw precomputed leaf escaped".into(),
                })?;
            if let Some(mu) = mu.as_deref_mut() {
                *mu.get_mut(r).ok_or_else(|| PbError::Internal {
                    what: "update_raw mu row escaped".into(),
                })? *= *leaf_mult
                    .get(usize::from(leaf))
                    .ok_or_else(|| PbError::Internal {
                        what: "incremental-mu leaf id escaped the multiplier table".into(),
                    })?;
                if previous.abs() >= 30.0 || slot.abs() >= 30.0 {
                    if let Some(entry) = mu.get_mut(r) {
                        *entry = f64::from(slot.clamp(-30.0, 30.0).exp());
                    }
                }
            }
        }
        return Ok(());
    }
    let columns = tree_split_columns(tree, &x.data)?;
    for (r, slot) in raw.iter_mut().enumerate() {
        let v = tree_value_for_row_with_columns(tree, &columns, r)?;
        let previous = *slot;
        *slot += v;
        if let Some(mu) = mu.as_deref_mut() {
            *mu.get_mut(r).ok_or_else(|| PbError::Internal {
                what: "update_raw mu walk row escaped".into(),
            })? *= f64::from(v).exp();
            if previous.abs() >= 30.0 || slot.abs() >= 30.0 {
                if let Some(entry) = mu.get_mut(r) {
                    *entry = f64::from(slot.clamp(-30.0, 30.0).exp());
                }
            }
        }
    }
    Ok(())
}

/// Update `raw` for the validation-carve case: grow's `leaf_of_row` is indexed by ABSOLUTE row and
/// is valid exactly for `covered_rows` (the sampled/train rows grow saw), so apply the map to those
/// and re-walk ONLY `walk_rows` (the validation holdout) — instead of re-walking every row. Byte-
/// identical to a full re-walk: `covered_rows` and `walk_rows` partition `0..raw.len()`, so each row
/// is updated exactly once, and grow's map matches the walk bit-for-bit (same canonical `low_bit`;
/// refinement changed only leaf VALUES, not memberships — pinned by
/// `update_raw_leaf_map_matches_tree_walk_bit_for_bit`).
fn update_raw_split(
    raw: &mut [f32],
    // Incremental-mu companion (see [`update_raw`]): `mu[r] *= e^{Δ}` in lockstep over BOTH the
    // covered-row leaf-map fill and the validation walk. `None` on the exact path.
    mut mu: Option<&mut [f64]>,
    x: &BinnedMatrix,
    tree: &ObliviousTree,
    leaf_of_row: &[u8],
    covered_rows: &[u32],
    walk_rows: &[usize],
) -> Result<(), PbError> {
    let leaf_mult = leaf_multipliers(tree, mu.is_some());
    for &r in covered_rows {
        let leaf = *leaf_of_row
            .get(r as usize)
            .ok_or_else(|| PbError::Internal {
                what: "update_raw_split covered row escaped membership map".into(),
            })?;
        let slot = raw.get_mut(r as usize).ok_or_else(|| PbError::Internal {
            what: "update_raw_split covered row escaped raw".into(),
        })?;
        let previous = *slot;
        *slot += *tree
            .leaves
            .get(usize::from(leaf))
            .ok_or_else(|| PbError::Internal {
                what: "update_raw_split leaf escaped".into(),
            })?;
        if let Some(mu) = mu.as_deref_mut() {
            *mu.get_mut(r as usize).ok_or_else(|| PbError::Internal {
                what: "update_raw_split covered mu row escaped".into(),
            })? *= *leaf_mult
                .get(usize::from(leaf))
                .ok_or_else(|| PbError::Internal {
                    what: "incremental-mu leaf id escaped the multiplier table".into(),
                })?;
            if previous.abs() >= 30.0 || slot.abs() >= 30.0 {
                if let Some(entry) = mu.get_mut(r as usize) {
                    *entry = f64::from(slot.clamp(-30.0, 30.0).exp());
                }
            }
        }
    }
    if !walk_rows.is_empty() {
        let columns = tree_split_columns(tree, &x.data)?;
        for &r in walk_rows {
            let v = tree_value_for_row_with_columns(tree, &columns, r)?;
            let slot = raw.get_mut(r).ok_or_else(|| PbError::Internal {
                what: "update_raw_split walk row escaped raw".into(),
            })?;
            let previous = *slot;
            *slot += v;
            if let Some(mu) = mu.as_deref_mut() {
                *mu.get_mut(r).ok_or_else(|| PbError::Internal {
                    what: "update_raw_split walk mu row escaped".into(),
                })? *= f64::from(v).exp();
                if previous.abs() >= 30.0 || slot.abs() >= 30.0 {
                    if let Some(entry) = mu.get_mut(r) {
                        *entry = f64::from(slot.clamp(-30.0, 30.0).exp());
                    }
                }
            }
        }
    }
    Ok(())
}

/// Score one row against one tree by column-major reads, folding the leaf index with
/// the SAME canonical `low_bit` rule as [`ObliviousTree::lookup`] and the grower.
#[cfg(test)]
fn tree_value_for_row(tree: &ObliviousTree, x: &BinnedMatrix, r: usize) -> Result<f32, PbError> {
    let columns = tree_split_columns(tree, &x.data)?;
    tree_value_for_row_with_columns(tree, &columns, r)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::float_cmp
    )]
    use super::*;
    use crate::boosters::{BoosterConfig, DartSpec, EnsembleSpec, HpGrid, NesterovSpec, RefitSpec};
    use crate::cat::{CatEncoderStore, LeakageScheme, Smooth, TsConfig, TsEncodingId};
    use crate::constraints::{CredibilityEvidence, CredibilityFloor, MonoSign};
    use crate::data::{
        bin_columns, bin_train_columns, BinConfig, CategoricalColumn, NumericColumn,
    };
    use crate::engine::{Booster, HistPrecision};
    use crate::explain::{assert_exact_decomposition, FeatureSet, RefMeasure};
    use crate::loss::{Gamma, Logistic, Loss, LossId, Poisson, SquaredError, Tweedie};

    fn spec<'a>(loss: &'a dyn Loss) -> FitSpec<'a> {
        FitSpec {
            loss,
            weight: None,
            exposure: None,
            monotone: crate::constraints::MonotoneMap::new(),
            interaction: crate::constraints::InteractionPolicy::default(),
            credibility: crate::constraints::CredibilityFloor::default(),
            fixed_holdout: None,
            bag_groups: None,
            seed: 0,
        }
    }

    fn binned(cols: &[Vec<f32>]) -> BinnedMatrix {
        let refs: Vec<&[f32]> = cols.iter().map(Vec::as_slice).collect();
        bin_columns(&refs, None, &BinConfig::default(), 0).unwrap()
    }

    fn predict(model: &Model, x: &BinnedMatrix, row: usize) -> f64 {
        let bins: Vec<u8> = x.data.iter().map(|c| c[row]).collect();
        model.ensemble_f64(&bins).unwrap()
    }

    #[test]
    fn affine_reanchor_recovers_a_known_scale_compression() {
        // y = exp(t) exactly ⇒ Poisson deviance over η is minimized pointwise at η = t, so a raw
        // score of 0.5·t (a uniformly compressed scale) must be corrected by a ≈ 2, b ≈ 0. The
        // holdout is large enough that the ridge evidence gate shrinks the slope only slightly.
        let n = 20_000;
        let t: Vec<f32> = (0..n).map(|i| -3.0 + 4.0 * i as f32 / n as f32).collect();
        let y: Vec<f32> = t.iter().map(|v| v.exp()).collect();
        let raw: Vec<f32> = t.iter().map(|v| 0.5 * v).collect();
        let w = vec![1.0_f32; t.len()];
        let rows: Vec<usize> = (0..t.len()).collect();
        let (a, b) = fit_affine_reanchor(&Poisson, &y, &raw, &w, None, &rows)
            .unwrap()
            .expect("a 2x compression on a large holdout must clear the evidence gate");
        assert!((1.85..=2.01).contains(&a), "a={a}");
        assert!(b.abs() < 0.1, "b={b}");
    }

    #[test]
    fn affine_reanchor_gates_a_spurious_fit_on_a_thin_holdout() {
        // 60 rows whose score pattern (period 2) is exactly orthogonal to the target pattern
        // (period 3). The unpenalized MLE would flatten the meaningless ±0.2 scores (a → 0), but
        // its small deviance gain cannot pay the ridge penalty of moving that far, so the gated
        // fit stays at (or shrinks hard toward) identity.
        let raw: Vec<f32> = (0..60)
            .map(|i| if i % 2 == 0 { 0.2 } else { -0.2 })
            .collect();
        let y: Vec<f32> = (0..60).map(|i| [0.0_f32, 2.0, 1.0][i % 3]).collect();
        let w = vec![1.0_f32; 60];
        let rows: Vec<usize> = (0..60).collect();
        let fit = fit_affine_reanchor(&Poisson, &y, &raw, &w, None, &rows).unwrap();
        if let Some((a, b)) = fit {
            assert!(
                (a - 1.0).abs() < 0.05,
                "gate failed to shrink a spurious slope: a={a}"
            );
            assert!(b.abs() < 0.1, "b={b}");
        } // None is equally acceptable — the gate collapsed it to identity outright.
    }

    #[test]
    fn affine_reanchor_is_identity_on_a_calibrated_score() {
        let t: Vec<f32> = (0..200).map(|i| -3.0 + 0.02 * i as f32).collect();
        let y: Vec<f32> = t.iter().map(|v| v.exp()).collect();
        let w = vec![1.0_f32; t.len()];
        let rows: Vec<usize> = (0..t.len()).collect();
        let fit = fit_affine_reanchor(&Poisson, &y, &t, &w, None, &rows).unwrap();
        if let Some((a, b)) = fit {
            assert!((a - 1.0).abs() < 1e-3 && b.abs() < 1e-3, "a={a} b={b}");
        } // None (immaterial correction) is the expected outcome on a calibrated score.
    }

    /// The leaf-refinement line search reconstructs `raw` from precomputed leaf MEMBERSHIPS
    /// (`apply_membership_leaves`) instead of re-walking the tree each trial. That fast path must
    /// be BIT-IDENTICAL to the tree-walk reconstruction (`raw_with_tree_leaves`) on the rows — the
    /// invariant that makes the speed optimization exactness-preserving.
    #[test]
    fn membership_leaf_fill_matches_tree_walk_bit_for_bit() {
        let n = 240usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 3) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| x0[i] + 2.0 * x1[i] - x2[i] + x0[i] * x1[i])
            .collect();
        let x = binned(&[x0, x1, x2]);
        let sqe = SquaredError;
        let model = Booster::with_config(Config {
            n_trees: 6,
            learning_rate: 0.5,
            lambda: 1.0,
            ..Config::default()
        })
        .fit(&x, &y, &spec(&sqe))
        .unwrap();
        let tree = &model.trees[0].1;
        let columns = tree_split_columns(tree, &x.data).unwrap();
        let rows: Vec<u32> = (0..n as u32).collect();
        let n_leaves = 1usize << usize::from(tree.depth);
        let memberships = tree_memberships_for_rows(tree, &columns, &rows, n_leaves).unwrap();
        let base_raw = vec![0.37_f32; n];
        let walk = raw_with_tree_leaves(&base_raw, tree, &columns, &tree.leaves).unwrap();
        let mut fill = vec![0.0_f32; n];
        apply_membership_leaves(&mut fill, &base_raw, &rows, &memberships, &tree.leaves).unwrap();
        for r in 0..n {
            assert_eq!(
                walk[r].to_bits(),
                fill[r].to_bits(),
                "row {r}: tree-walk {} != membership-fill {}",
                walk[r],
                fill[r]
            );
        }

        // The dense (contiguous) leaf-refine fill must equal the scattered
        // `apply_membership_leaves` output gathered over a SUBSET of rows, bit-for-bit — the
        // invariant the leaf-refine line search relies on for byte-stable predictions. Use a
        // reordered subset with repeats and altered leaf values (a refinement trial).
        let sub_rows: Vec<u32> = vec![5, 0, 123, 123, 239, 17, 88, 0, 200, 9];
        let sub_members = tree_memberships_for_rows(tree, &columns, &sub_rows, n_leaves).unwrap();
        let base_sub = gather_rows(&base_raw, &sub_rows).unwrap();
        let trial_leaves = {
            let mut l = tree.leaves.clone();
            for (k, v) in l.iter_mut().enumerate() {
                *v = *v + 0.013 * (k as f32) - 0.05;
            }
            l
        };
        let mut full = base_raw.clone();
        apply_membership_leaves(&mut full, &base_raw, &sub_rows, &sub_members, &trial_leaves)
            .unwrap();
        let mut dense = base_sub.clone();
        fill_leaf_raw_contiguous(&mut dense, &base_sub, &sub_members, &trial_leaves).unwrap();
        for (i, &r) in sub_rows.iter().enumerate() {
            assert_eq!(
                full[r as usize].to_bits(),
                dense[i].to_bits(),
                "subset row {r}: scatter {} != dense {}",
                full[r as usize],
                dense[i]
            );
        }
    }

    /// WIN #10 invariant: grow returns its per-row leaf map (`leaf_of_row`); the leaf-refine line
    /// search reuses it instead of re-walking the tree. Gathering that map over `rows` must be
    /// BIT-IDENTICAL to `tree_memberships_for_rows` (the walk) — both assign leaves via the SAME
    /// canonical `low_bit`. If they ever diverged, the reused-membership fast path would silently
    /// refine the wrong leaves and break exact decomposition.
    #[test]
    fn grow_leaf_map_matches_tree_walk_memberships_bit_for_bit() {
        let n = 240usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 3) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| x0[i] + 2.0 * x1[i] - x2[i] + x0[i] * x1[i])
            .collect();
        let x = binned(&[x0, x1, x2]);
        let weight = vec![1.0_f32; n];
        let raw = vec![0.0_f32; n];
        let sqe = SquaredError;
        let mut gh = GradHess::default();
        sqe.grad_hess(&y, &raw, &weight, &mut gh).unwrap();
        let rows: Vec<u32> = (0..n as u32).collect();
        let cfg = GrowConfig {
            lambda: 1.0,
            l1_leaf: 0.0,
            lr: 0.5,
            min_split_gain: 0.0,
            interaction_gain_hurdle: 0.0,
            max_order: 3,
            max_depth: LEGACY_MAX_DEPTH as u8,
            max_delta_step: None,
            hist_precision: HistPrecision::FullF64,
            quant_seed: 0,
            round: 0,
            random_strength: 0.0,
            groups: None,
            monotone: None,
            table_budget_penalty: None,
            realized_extent: None,
            credibility: CredibilityFloor::default(),
            credibility_evidence: CredibilityEvidence::Count,
            unit_weight: true,
            hist_subtraction: true,
        };
        let grown = grow_oblivious_tree_with_leaf_map(
            &x,
            &gh,
            &rows,
            &[0, 1, 2],
            &cfg,
            InteractionHurdleState::default(),
            &weight,
        )
        .unwrap()
        .expect("a tree");
        let tree = grown.tree;
        let leaf_of_row = grown.leaf_of_row;
        let n_leaves = 1usize << usize::from(tree.depth);
        let columns = tree_split_columns(&tree, &x.data).unwrap();
        // Full rows: gather of grow's map == tree-walk memberships, element for element.
        let walk = tree_memberships_for_rows(&tree, &columns, &rows, n_leaves).unwrap();
        let gathered = gather_memberships(&leaf_of_row, &rows, n_leaves).unwrap();
        assert_eq!(walk, gathered, "grow leaf map != tree-walk memberships");
        // Reordered subset with repeats (the line search passes arbitrary `rows`).
        let sub: Vec<u32> = vec![5, 0, 123, 239, 17, 88, 200, 9, 0, 123];
        let walk_sub = tree_memberships_for_rows(&tree, &columns, &sub, n_leaves).unwrap();
        let gathered_sub = gather_memberships(&leaf_of_row, &sub, n_leaves).unwrap();
        assert_eq!(walk_sub, gathered_sub, "subset: grow leaf map != tree-walk");
    }

    /// WIN #11 invariant: `update_raw` fed grow's `leaf_of_row` must add EXACTLY the same per-row
    /// contribution as the tree walk (`tree_value_for_row_with_columns`) — both index `tree.leaves`
    /// by the canonical-`low_bit` leaf, so the tree-walk-free update is bit-identical. Pinned here so
    /// it can never drift the accumulated `raw` (hence predictions).
    #[test]
    fn update_raw_leaf_map_matches_tree_walk_bit_for_bit() {
        let n = 240usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 3) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| x0[i] + 2.0 * x1[i] - x2[i] + x0[i] * x1[i])
            .collect();
        let x = binned(&[x0, x1, x2]);
        let weight = vec![1.0_f32; n];
        let raw0 = vec![0.0_f32; n];
        let sqe = SquaredError;
        let mut gh = GradHess::default();
        sqe.grad_hess(&y, &raw0, &weight, &mut gh).unwrap();
        let rows: Vec<u32> = (0..n as u32).collect();
        let cfg = GrowConfig {
            lambda: 1.0,
            l1_leaf: 0.0,
            lr: 0.5,
            min_split_gain: 0.0,
            interaction_gain_hurdle: 0.0,
            max_order: 3,
            max_depth: LEGACY_MAX_DEPTH as u8,
            max_delta_step: None,
            hist_precision: HistPrecision::FullF64,
            quant_seed: 0,
            round: 0,
            random_strength: 0.0,
            groups: None,
            monotone: None,
            table_budget_penalty: None,
            realized_extent: None,
            credibility: CredibilityFloor::default(),
            credibility_evidence: CredibilityEvidence::Count,
            unit_weight: true,
            hist_subtraction: true,
        };
        let grown = grow_oblivious_tree_with_leaf_map(
            &x,
            &gh,
            &rows,
            &[0, 1, 2],
            &cfg,
            InteractionHurdleState::default(),
            &weight,
        )
        .unwrap()
        .expect("a tree");
        let tree = grown.tree;
        let leaf_of_row = grown.leaf_of_row;
        // Non-trivial base raw so the ADD (not just the leaf value) is exercised.
        let base: Vec<f32> = (0..n).map(|i| 0.1 * i as f32 - 3.0).collect();
        let mut raw_walk = base.clone();
        update_raw(&mut raw_walk, None, &x, &tree, None).unwrap();
        let mut raw_map = base.clone();
        update_raw(&mut raw_map, None, &x, &tree, Some(&leaf_of_row)).unwrap();
        for r in 0..n {
            assert_eq!(
                raw_walk[r].to_bits(),
                raw_map[r].to_bits(),
                "row {r}: tree-walk update {} != leaf-map update {}",
                raw_walk[r],
                raw_map[r]
            );
        }
    }

    #[test]
    fn update_raw_split_matches_full_walk_bit_for_bit() {
        // The validation-carve fast path (`update_raw_split`) must match a full re-walk bit-for-bit.
        // Covered rows are the EVENS and walk rows the ODDS — non-contiguous so `covered_rows[i] != i`,
        // which catches absolute-vs-position indexing of `leaf_of_row` (a bug the full-sample tests
        // where `sampled_rows[i] == i` would miss).
        let n = 240usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 3) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| x0[i] + 2.0 * x1[i] - x2[i] + x0[i] * x1[i])
            .collect();
        let x = binned(&[x0, x1, x2]);
        let weight = vec![1.0_f32; n];
        let raw0 = vec![0.0_f32; n];
        let sqe = SquaredError;
        let mut gh = GradHess::default();
        sqe.grad_hess(&y, &raw0, &weight, &mut gh).unwrap();
        let rows: Vec<u32> = (0..n as u32).collect();
        let cfg = GrowConfig {
            lambda: 1.0,
            l1_leaf: 0.0,
            lr: 0.5,
            min_split_gain: 0.0,
            interaction_gain_hurdle: 0.0,
            max_order: 3,
            max_depth: LEGACY_MAX_DEPTH as u8,
            max_delta_step: None,
            hist_precision: HistPrecision::FullF64,
            quant_seed: 0,
            round: 0,
            random_strength: 0.0,
            groups: None,
            monotone: None,
            table_budget_penalty: None,
            realized_extent: None,
            credibility: CredibilityFloor::default(),
            credibility_evidence: CredibilityEvidence::Count,
            unit_weight: true,
            hist_subtraction: true,
        };
        let grown = grow_oblivious_tree_with_leaf_map(
            &x,
            &gh,
            &rows,
            &[0, 1, 2],
            &cfg,
            InteractionHurdleState::default(),
            &weight,
        )
        .unwrap()
        .expect("a tree");
        let tree = grown.tree;
        let leaf_of_row = grown.leaf_of_row;
        let base: Vec<f32> = (0..n).map(|i| 0.1 * i as f32 - 3.0).collect();
        let covered: Vec<u32> = (0..n as u32).filter(|r| r % 2 == 0).collect();
        let walk: Vec<usize> = (0..n).filter(|r| r % 2 == 1).collect();
        let mut raw_walk = base.clone();
        update_raw(&mut raw_walk, None, &x, &tree, None).unwrap();
        let mut raw_split = base.clone();
        update_raw_split(
            &mut raw_split,
            None,
            &x,
            &tree,
            &leaf_of_row,
            &covered,
            &walk,
        )
        .unwrap();
        for r in 0..n {
            assert_eq!(
                raw_walk[r].to_bits(),
                raw_split[r].to_bits(),
                "row {r}: full walk {} != split update {}",
                raw_walk[r],
                raw_split[r]
            );
        }
    }

    /// Gate G2 (exact): an additive piecewise-constant target on 2 features is
    /// recovered to float tolerance. With λ=0, lr=1 the first depth-2 tree fits the 4
    /// regions exactly (each leaf = its region's value), so recovery is essentially bit-exact.
    #[test]
    fn g2_recovers_piecewise_constant_target_exactly() {
        let n = 60usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 3.0 { 10.0 } else { 20.0 };
                let b = if x1[i] <= 2.0 { 5.0 } else { 0.0 };
                a + b
            })
            .collect();
        let x = binned(&[x0, x1]);
        let booster = Booster::with_config(Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 20,
            learning_rate: 1.0,
            lambda: 0.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        });
        let sqe = SquaredError;
        let model = booster.fit(&x, &y, &spec(&sqe)).unwrap();
        assert_eq!(model.mode, ExactnessMode::Exact);
        for (i, &yi) in y.iter().enumerate() {
            let pred = predict(&model, &x, i);
            assert!(
                (pred - f64::from(yi)).abs() < 1e-3,
                "row {i}: pred {pred} != y {yi}"
            );
        }
    }

    /// Gate G2 (regularized convergence): with λ=1 the iterative loop converges to the
    /// target over many shrunken trees (exercises multi-round boosting, not 1-tree exactness).
    #[test]
    fn g2_converges_under_regularization() {
        let n = 80usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 3 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| if x0[i] <= 2.0 { -3.0 } else { 4.0 } + if x1[i] <= 1.0 { 1.0 } else { -1.0 })
            .collect();
        let x = binned(&[x0, x1]);
        let booster = Booster::with_config(Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 300,
            learning_rate: 0.3,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        });
        let sqe = SquaredError;
        let model = booster.fit(&x, &y, &spec(&sqe)).unwrap();
        let max_err = (0..n)
            .map(|i| (predict(&model, &x, i) - f64::from(y[i])).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 0.05, "did not converge: max_err {max_err}");
    }

    #[test]
    fn fitted_model_is_byte_identical_across_thread_counts() {
        let n = 300usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 7 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 3 + 1) as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (i as f32 % 11.0) - 5.0).collect();
        let x = binned(&[x0, x1, x2]);
        let bytes = |nt: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let booster = Booster::with_config(Config {
                    lambda_scale_invariant: false,
                    max_delta_step_gated: GatedStepPolicy::Objective,
                    n_trees: 40,
                    learning_rate: 0.3,
                    lambda: 1.0,
                    min_split_gain: 0.0,
                    max_delta_step: None,
                    sampling: Default::default(),
                    hist_precision: Default::default(),
                    l1_leaf: 0.0,
                    colsample_bytree: 1.0,
                    learning_rate_decay: 0.0,
                    validation_fraction: None,
                    early_stopping_rounds: 50,
                    early_stopping_adaptive: None,
                    early_stopping_min_delta: 0.0,
                    interaction_gain_hurdle: 0.0,
                    interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
                    leaf_refine_steps: 0,
                    leaf_refine_backtracks: 4,
                    refine_closed_form_tier2: false,
                    incremental_mu: false,
                    boosters: Default::default(),
                    fit_control: Default::default(),
                });
                let sqe = SquaredError;
                let model = booster.fit(&x, &y, &spec(&sqe)).unwrap();
                crate::serialize::encode_model(&model).unwrap()
            })
        };
        let b1 = bytes(1);
        assert!(!b1.is_empty());
        assert_eq!(b1, bytes(2));
        assert_eq!(b1, bytes(8));
    }

    /// A border-rich fixture (spec review 2026-07-11, L1108-1119, W3k fix): two axes with
    /// ~60 realized bins each, fit with the table-budget admission prior ACTIVE
    /// (`table_budget_beta=0.5`) against a budget tight enough (`table_budget_cells=500`)
    /// that the OLD `grid.n_bins`-based projection (`~60*60=3600`, 7x over budget) would
    /// have meaningfully suppressed candidate ranking scores from the very first tree --
    /// nothing has been realized yet at that point, so the fixed, accumulator-based
    /// projection (extent 2 per never-split axis, `2*2=4`, nowhere near the budget) must
    /// leave tree 1 IDENTICAL to a fully unpenalized (`beta=0`) reference fit. A handful of
    /// small trees keeps every one of them "tree 1" in the sense that matters here: no axis
    /// accumulates enough realized borders within this short a fit to legitimately earn a
    /// penalty, so the two configurations must match byte-for-byte all the way through.
    #[test]
    fn table_budget_penalty_does_not_over_penalize_realized_extent_from_tree_one() {
        let n = 3000usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 60) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| ((i / 60) % 50) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = x0[i];
                let b = x1[i];
                // Additive-plus-interaction signal: both axes carry real, separable gain.
                a * 0.1 + b * 0.15 + if a > 30.0 && b > 25.0 { 5.0 } else { 0.0 }
            })
            .collect();
        let x = binned(&[x0, x1]);
        // Sanity: the fixture is genuinely border-rich -- otherwise this test would pass
        // trivially without ever exercising the over-budget path either way.
        assert!(
            x.grids[0].n_bins > 20 && x.grids[1].n_bins > 20,
            "fixture must realize enough bins to matter: {} / {}",
            x.grids[0].n_bins,
            x.grids[1].n_bins
        );

        let sqe = SquaredError;
        let fit = |beta: f32| {
            let mut s = spec(&sqe);
            s.interaction = crate::constraints::InteractionPolicy {
                table_budget_beta: beta,
                table_budget_cells: 500,
                ..crate::constraints::InteractionPolicy::default()
            };
            let config = Config {
                n_trees: 5,
                learning_rate: 0.3,
                ..Config::default()
            };
            let booster = Booster::with_config(config);
            let model = booster.fit(&x, &y, &s).unwrap();
            crate::serialize::encode_model(&model).unwrap()
        };
        let unpenalized = fit(0.0);
        let penalized = fit(0.5);
        assert!(!unpenalized.is_empty());
        assert_eq!(
            unpenalized, penalized,
            "the realized-extent-based penalty must not diverge from the unpenalized \
             reference this early in the fit"
        );
    }

    /// Thread-count determinism (the §1 GATE) with the realized-extent accumulator actively
    /// growing across rounds: `RealizedExtent::record_tree` is called from `fit_single`'s own
    /// sequential round loop (never shared across bags/threads), so the fitted bytes must
    /// match regardless of the pool's thread count, exactly like the unpenalized case above.
    #[test]
    fn table_budget_penalty_fit_is_byte_identical_across_thread_counts() {
        let n = 2000usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 60) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| ((i / 40) % 45) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 17) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| x0[i] * 0.1 + x1[i] * 0.2 - x2[i] * 0.05)
            .collect();
        let x = binned(&[x0, x1, x2]);
        let bytes = |nt: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let sqe = SquaredError;
                let mut s = spec(&sqe);
                s.interaction = crate::constraints::InteractionPolicy {
                    table_budget_beta: 0.5,
                    table_budget_cells: 500,
                    ..crate::constraints::InteractionPolicy::default()
                };
                let config = Config {
                    n_trees: 25,
                    learning_rate: 0.3,
                    ..Config::default()
                };
                let booster = Booster::with_config(config);
                let model = booster.fit(&x, &y, &s).unwrap();
                crate::serialize::encode_model(&model).unwrap()
            })
        };
        let b1 = bytes(1);
        assert!(!b1.is_empty());
        assert_eq!(b1, bytes(2));
        assert_eq!(b1, bytes(8));
    }

    #[test]
    fn multiclass_softmax_fits_separable_three_classes_and_proba_is_a_simplex() {
        // Three classes cleanly separated by x0's region; x1 is noise. Native softmax should
        // drive train accuracy near-perfect, and predict_proba rows must be a probability simplex.
        let n = 300usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 30) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
        let y: Vec<f32> = x0
            .iter()
            .map(|&v| {
                if v < 10.0 {
                    0.0
                } else if v < 20.0 {
                    1.0
                } else {
                    2.0
                }
            })
            .collect();
        let xb = binned(&[x0, x1]);
        let config = Config {
            n_trees: 80,
            learning_rate: 0.3,
            ..Config::default()
        };
        let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let sqe = SquaredError; // ignored by fit_multiclass (softmax gradient is coupled)
        let model = fit_multiclass(
            &config,
            &xb,
            &y,
            3,
            &labels,
            &spec(&sqe),
            &CatEncoderStore::new(),
        )
        .unwrap();
        assert_eq!(model.n_classes(), 3);
        assert_eq!(model.class_labels, labels);
        let proba = model.predict_proba(&xb).unwrap();
        assert_eq!(proba.len(), n * 3);
        let mut correct = 0usize;
        for r in 0..n {
            let row = &proba[r * 3..r * 3 + 3];
            let s: f32 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "row {r} proba sums to {s}, not 1");
            for &p in row {
                assert!((0.0..=1.0).contains(&p), "proba {p} out of [0,1]");
            }
            let best = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap();
            if best as f32 == y[r] {
                correct += 1;
            }
        }
        assert!(
            correct as f64 / n as f64 > 0.95,
            "train accuracy {correct}/{n} too low"
        );
    }

    #[test]
    fn multiclass_per_class_models_are_exactly_decomposable() {
        // Each per-class logit F_k is an ordinary scalar Model, so it must independently pass the
        // five exactness invariants — K exact fANOVA banks, the guarantee the design preserves.
        let n = 240usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 12) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (i % 3) as f32).collect();
        let xb = binned(&[x0, x1]);
        let config = Config {
            n_trees: 25,
            learning_rate: 0.3,
            ..Config::default()
        };
        let labels = vec!["0".to_string(), "1".to_string(), "2".to_string()];
        let sqe = SquaredError;
        let model = fit_multiclass(
            &config,
            &xb,
            &y,
            3,
            &labels,
            &spec(&sqe),
            &CatEncoderStore::new(),
        )
        .unwrap();
        let serve = crate::data::ServeBinnedMatrix(xb.clone());
        for m in &model.classes {
            assert_eq!(m.link, Link::Identity);
            assert_eq!(m.mode, ExactnessMode::Exact);
            assert_eq!(m.schema.objective.loss, LossId::Softmax);
            let bank = m.explain(&serve, RefMeasure::default()).unwrap();
            assert_exact_decomposition(m, &bank, &serve).unwrap();
        }
    }

    /// A bagged K-class fit with a signal the base loop under-fits, for the §G1 tests below.
    /// Two categorical-ish axes whose INTERACTION carries the class signal is exactly the
    /// surface the cell refit exists to tighten.
    fn multiclass_cellrefit_fixture() -> (BinnedMatrix, Vec<f32>, Config) {
        let n = 3_000usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 11) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| ((i / 11) % 7) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| ((i / 77) % 5) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| (((i % 11) * ((i / 11) % 7) + (i / 77) % 5) % 3) as f32)
            .collect();
        let xb = binned(&[x0, x1, x2]);
        let config = Config {
            n_trees: 30,
            learning_rate: 0.3,
            boosters: BoosterConfig {
                ensemble: EnsembleSpec::OuterBag {
                    n_bags: 4,
                    bag_subsample: 0.8,
                    cell_refit: None,
                },
                ..BoosterConfig::default()
            },
            ..Config::default()
        };
        (xb, y, config)
    }

    fn with_cell_refit(config: &Config, cr: Option<CellRefit>) -> Config {
        let mut c = config.clone();
        if let EnsembleSpec::OuterBag { cell_refit, .. } = &mut c.boosters.ensemble {
            *cell_refit = cr;
        }
        c
    }

    #[test]
    fn multiclass_cell_refit_moves_the_model_and_stays_exactly_decomposable() {
        // The K>=3 §G1 correction deposits into the PER-CLASS banks as ordinary table entries,
        // so the corrected per-class logit must still pass all five exactness invariants — that
        // is the "exact by construction" claim, verified rather than asserted.
        let (xb, y, base) = multiclass_cellrefit_fixture();
        let labels = vec!["0".to_string(), "1".to_string(), "2".to_string()];
        let sqe = SquaredError;
        let off = fit_multiclass(
            &with_cell_refit(&base, None),
            &xb,
            &y,
            3,
            &labels,
            &spec(&sqe),
            &CatEncoderStore::new(),
        )
        .unwrap();
        let on = fit_multiclass(
            &with_cell_refit(
                &base,
                Some(CellRefit {
                    base: 50.0,
                    gamma: 2.0,
                }),
            ),
            &xb,
            &y,
            3,
            &labels,
            &spec(&sqe),
            &CatEncoderStore::new(),
        )
        .unwrap();
        assert!(
            on.classes.iter().any(|m| m.correction.is_some()),
            "the joint backtrack declined every class on a surface built for the refit"
        );
        let serve = crate::data::ServeBinnedMatrix(xb.clone());
        for m in &on.classes {
            let bank = m.explain(&serve, RefMeasure::default()).unwrap();
            assert_exact_decomposition(m, &bank, &serve).unwrap();
        }
        let a = off.predict_raw(&xb).unwrap();
        let b = on.predict_raw(&xb).unwrap();
        let moved = a
            .iter()
            .zip(&b)
            .map(|(u, v)| (u - v).abs())
            .fold(0.0_f32, f32::max);
        assert!(moved > 0.0, "the accepted correction changed no logit");

        // av39 reporting surface: the refit reports what it did, and reporting is not identity.
        assert!(
            off.cell_refit.is_none(),
            "a fit that never asked for the refit must carry no report"
        );
        let r = on.cell_refit.expect("a refitted fit must carry its report");
        assert!(
            !r.declined(),
            "lambda {} declined an accepted fit",
            r.lambda
        );
        assert!(r.lambda > 0.0 && r.lambda <= 1.0, "lambda {}", r.lambda);
        assert!(r.n_supports > 0, "no correctable supports were counted");
        assert!(r.guard_rows > 0, "the backtrack scored no jury rows");
        assert!(
            (0.0..=1.0).contains(&r.reachable_coverage()),
            "coverage {}",
            r.reachable_coverage()
        );
        // `cell_refit` is `#[serde(skip)]` runtime metadata excluded from `PartialEq`, so two
        // models differing ONLY in their report still compare equal (the §10.7 round-trip
        // contract).
        let mut stripped = on.clone();
        stripped.cell_refit = None;
        assert_eq!(
            on, stripped,
            "the report must not be part of model identity"
        );
    }

    #[test]
    fn multiclass_cell_refit_off_is_byte_identical_and_on_is_thread_stable() {
        // Standing gate: K>=3 with the feature OFF must be bit-for-bit the pre-feature fit, and
        // the corrected fit must be byte-identical across thread counts (the jury accumulation,
        // the per-class solves and the λ search are all fixed-order).
        let (xb, y, base) = multiclass_cellrefit_fixture();
        let labels = vec!["0".to_string(), "1".to_string(), "2".to_string()];
        let sqe = SquaredError;
        let fit = |cr: Option<CellRefit>, threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                fit_multiclass(
                    &with_cell_refit(&base, cr),
                    &xb,
                    &y,
                    3,
                    &labels,
                    &spec(&sqe),
                    &CatEncoderStore::new(),
                )
                .unwrap()
            })
        };
        let cr = Some(CellRefit {
            base: 50.0,
            gamma: 2.0,
        });
        let off1 = fit(None, 1);
        let off8 = fit(None, 8);
        assert_eq!(off1.classes, off8.classes);
        let on1 = fit(cr, 1);
        let on2 = fit(cr, 2);
        let on8 = fit(cr, 8);
        assert_eq!(on1.classes, on2.classes);
        assert_eq!(on1.classes, on8.classes);
    }

    #[test]
    fn multiclass_cell_refit_declines_when_there_is_nothing_to_correct() {
        // Pure noise: the decoupled per-class solves fit sampling noise on the out-of-bag jury,
        // the JOINT backtrack sees no held-out gain, and λ=0 drops the whole K-class correction.
        // This is the "no multiclass signal" answer the design must be able to give.
        let n = 1_200usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 13) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| ((i * 7) % 9) as f32).collect();
        // Labels independent of x (cycling with a period coprime to both axes' periods).
        let y: Vec<f32> = (0..n).map(|i| ((i * 5) % 3) as f32).collect();
        let xb = binned(&[x0, x1]);
        let config = Config {
            n_trees: 8,
            learning_rate: 0.05,
            boosters: BoosterConfig {
                ensemble: EnsembleSpec::OuterBag {
                    n_bags: 4,
                    bag_subsample: 0.8,
                    // A huge ridge makes every per-class δ tiny; the guard still has to be the
                    // thing that decides, and on a null surface it must decline outright.
                    cell_refit: Some(CellRefit {
                        base: 1.0e12,
                        gamma: 2.0,
                    }),
                },
                ..BoosterConfig::default()
            },
            ..Config::default()
        };
        let labels = vec!["0".to_string(), "1".to_string(), "2".to_string()];
        let sqe = SquaredError;
        let m = fit_multiclass(
            &config,
            &xb,
            &y,
            3,
            &labels,
            &spec(&sqe),
            &CatEncoderStore::new(),
        )
        .unwrap();
        assert!(
            m.classes.iter().all(|c| c.correction.is_none()),
            "a null surface must leave no correction attached"
        );
    }

    #[test]
    fn multiclass_cell_refit_at_one_bag_refuses() {
        // One bag has no out-of-bag complement, so there is no honest evidence. The K>=3 path
        // must REFUSE with the scalar path's own message rather than silently dropping the
        // correction the caller asked for.
        let (xb, y, base) = multiclass_cellrefit_fixture();
        let mut cfg = base.clone();
        cfg.boosters.ensemble = EnsembleSpec::OuterBag {
            n_bags: 1,
            bag_subsample: 0.8,
            cell_refit: Some(CellRefit {
                base: 50.0,
                gamma: 2.0,
            }),
        };
        let labels = vec!["0".to_string(), "1".to_string(), "2".to_string()];
        let sqe = SquaredError;
        let err = fit_multiclass(
            &cfg,
            &xb,
            &y,
            3,
            &labels,
            &spec(&sqe),
            &CatEncoderStore::new(),
        )
        .unwrap_err();
        assert!(
            format!("{err}").contains("cell_refit requires n_bags >= 2"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn multiclass_fit_is_byte_identical_across_thread_counts() {
        let n = 320usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 9 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (i % 4) as f32).collect();
        let xb = binned(&[x0, x1, x2]);
        let config = Config {
            n_trees: 40,
            learning_rate: 0.3,
            ..Config::default()
        };
        let labels: Vec<String> = (0..4).map(|k| k.to_string()).collect();
        let run = |nt: usize| -> MultiClassModel {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let sqe = SquaredError;
                fit_multiclass(
                    &config,
                    &xb,
                    &y,
                    4,
                    &labels,
                    &spec(&sqe),
                    &CatEncoderStore::new(),
                )
                .unwrap()
            })
        };
        let m1 = run(1);
        assert_eq!(m1, run(2));
        assert_eq!(m1, run(8));
    }

    #[test]
    fn interaction_gain_reference_is_decayed_max() {
        let mut reference = None;
        update_interaction_gain_reference(&mut reference, 100.0);
        assert_eq!(reference, Some(100.0));
        update_interaction_gain_reference(&mut reference, 1.0);
        assert_eq!(reference, Some(100.0 * INTERACTION_GAIN_REFERENCE_DECAY));
        update_interaction_gain_reference(&mut reference, 200.0);
        assert_eq!(reference, Some(200.0));
    }

    /// The ADAPTIVE-patience knob is a pure policy over `early_stopping_rounds`: `None` must return
    /// the fixed cap unchanged for EVERY position (so the swapped fit-loop comparison is
    /// bit-identical), and `Some(ratio)` must clamp `ceil(ratio * best)` into `[floor, cap]` with the
    /// cap always winning (an inverted floor>cap must degrade to the cap, never panic).
    #[test]
    fn adaptive_effective_patience_is_clamped_ceiling_or_the_fixed_cap() {
        use crate::engine::EARLY_STOPPING_ADAPTIVE_FLOOR as FLOOR;
        // None ⇒ exactly the fixed patience, for many best-round positions (the "multiple seeds").
        let fixed = Config {
            early_stopping_rounds: 500,
            early_stopping_adaptive: None,
            ..Config::default()
        };
        for best in [0usize, 1, 54, 499, 500, 10_000] {
            assert_eq!(fixed.effective_patience(best), 500);
        }
        // Some(1.5): ceil(1.5 * best), clamped to [FLOOR, 500].
        let adaptive = Config {
            early_stopping_rounds: 500,
            early_stopping_adaptive: Some(1.5),
            ..Config::default()
        };
        assert_eq!(adaptive.effective_patience(0), FLOOR); // ceil(0) -> floor
        assert_eq!(adaptive.effective_patience(10), FLOOR); // ceil(15) -> floor
        assert_eq!(adaptive.effective_patience(33), FLOOR); // ceil(49.5)=50 == floor
        assert_eq!(adaptive.effective_patience(34), 51); // ceil(51) just clears the floor
        assert_eq!(adaptive.effective_patience(100), 150);
        assert_eq!(adaptive.effective_patience(333), 500); // ceil(499.5)=500 == cap
        assert_eq!(adaptive.effective_patience(334), 500); // ceil(501)=501 -> cap wins
        assert_eq!(adaptive.effective_patience(10_000), 500); // cap
                                                              // A sub-floor ratio never drops below the floor.
        let small_ratio = Config {
            early_stopping_rounds: 500,
            early_stopping_adaptive: Some(0.4),
            ..Config::default()
        };
        assert_eq!(small_ratio.effective_patience(54), FLOOR); // ceil(21.6)=22 -> floor
                                                               // Degenerate: floor above the cap ⇒ the cap wins, no inverted-clamp panic.
        let tiny_cap = Config {
            early_stopping_rounds: 20,
            early_stopping_adaptive: Some(1.5),
            ..Config::default()
        };
        assert_eq!(tiny_cap.effective_patience(1), 20);
        assert_eq!(tiny_cap.effective_patience(10_000), 20);
    }

    /// Replays the exact `fit_single` early-stop rule (tree-count units, `>=` comparison, threshold
    /// from `effective_patience`) over a controlled V-shaped validation curve. `None` (large fixed
    /// patience) trains strictly longer than `Some(1.5)`, yet BOTH lock onto the same best round — so
    /// the truncated model is identical while the adaptive fit discards far fewer trained rounds.
    #[test]
    fn adaptive_patience_stops_sooner_but_keeps_the_same_best_round() {
        let best_at = 30usize;
        // One tree per round; strictly decreasing to a single minimum at `best_at`, then rising.
        let deviances: Vec<f64> = (1..=200)
            .map(|r| (r as f64 - best_at as f64).abs())
            .collect();
        // Faithful replay of the fit_single stop rule; returns (trained_rounds, best_round).
        let simulate = |config: &Config| -> (usize, usize) {
            let mut best = f64::INFINITY;
            let mut best_round = 0usize;
            for (idx, &dev) in deviances.iter().enumerate() {
                let trees_len = idx + 1;
                if dev < best {
                    best = dev;
                    best_round = trees_len;
                } else if trees_len.saturating_sub(best_round)
                    >= config.effective_patience(best_round)
                {
                    return (trees_len, best_round);
                }
            }
            (deviances.len(), best_round)
        };
        let fixed = Config {
            early_stopping_rounds: 100,
            early_stopping_adaptive: None,
            ..Config::default()
        };
        let adaptive = Config {
            early_stopping_rounds: 500,
            early_stopping_adaptive: Some(1.5),
            ..Config::default()
        };
        let (fixed_trained, fixed_best) = simulate(&fixed);
        let (adaptive_trained, adaptive_best) = simulate(&adaptive);
        // Both truncate to the true minimum...
        assert_eq!(fixed_best, best_at);
        assert_eq!(adaptive_best, best_at);
        // ...but the adaptive fit stops far sooner: fixed patience 100 ⇒ best+100; adaptive
        // ceil(1.5*30)=45 -> floored to 50 ⇒ best+50.
        assert_eq!(fixed_trained, best_at + 100);
        assert_eq!(adaptive_trained, best_at + 50);
        assert!(adaptive_trained < fixed_trained);
    }

    /// End-to-end (real `fit_single`): when the adaptive ratio is large enough that
    /// `effective_patience` saturates to the `early_stopping_rounds` cap at every position, the
    /// `Some` code path is BYTE-IDENTICAL to `None`. With `early_stopping_rounds == FLOOR` the floor
    /// can never lift patience above the cap, so `Some(1e9)` and `None` share the same patience for
    /// every best round — a hard equality that exercises the adaptive branch on the real fit loop.
    #[test]
    fn adaptive_matches_fixed_byte_for_byte_when_patience_saturates_to_the_cap() {
        let n = 240usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 11) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let noise = |i: usize| -> f32 {
            let v = ((i as f32) * 12.9898).sin() * 43_758.547;
            v - v.floor()
        };
        let y: Vec<f32> = (0..n)
            .map(|i| 0.6 * x0[i] - 0.4 * x1[i] + 2.5 * (noise(i) - 0.5))
            .collect();
        let x = binned(&[x0, x1, x2]);
        // early_stopping_rounds == FLOOR ⇒ effective_patience is exactly the cap for ANY best round
        // (max(_, FLOOR).min(cap) == cap), so Some(huge) == None at every step.
        let base = Config {
            n_trees: 300,
            learning_rate: 0.4,
            lambda: 0.5,
            validation_fraction: Some(0.25),
            early_stopping_rounds: crate::engine::EARLY_STOPPING_ADAPTIVE_FLOOR as u32,
            ..Config::default()
        };
        let sqe = SquaredError;
        let fit = |cfg: &Config| {
            Booster::with_config(cfg.clone())
                .fit(&x, &y, &spec(&sqe))
                .unwrap()
        };
        let m_none = fit(&base);
        let mut saturated = base.clone();
        saturated.early_stopping_adaptive = Some(1e9);
        let m_saturated = fit(&saturated);
        assert_eq!(
            m_none.trees.len(),
            m_saturated.trees.len(),
            "saturated-adaptive kept a different tree count than fixed"
        );
        for i in 0..n {
            assert_eq!(
                predict(&m_none, &x, i).to_bits(),
                predict(&m_saturated, &x, i).to_bits(),
                "row {i}: saturated-adaptive diverged from fixed patience",
            );
        }
    }

    /// End-to-end (real `fit_single`): a real `Some(1.5)` fit against fixed patience. A smaller
    /// adaptive patience can only stop at or before the fixed-patience stop, so it never RETAINS
    /// more trees than the fixed fit (its best round over a shorter trained prefix is <= the fixed
    /// one). And whenever the two land on the same best round, the deployed models must be
    /// byte-identical — the adaptive knob changes only trained-then-discarded rounds, not the model.
    #[test]
    fn adaptive_never_keeps_more_trees_and_agrees_when_best_round_matches() {
        let n = 240usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 11) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let noise = |i: usize| -> f32 {
            let v = ((i as f32) * 12.9898).sin() * 43_758.547;
            v - v.floor()
        };
        let y: Vec<f32> = (0..n)
            .map(|i| 0.6 * x0[i] - 0.4 * x1[i] + 2.5 * (noise(i) - 0.5))
            .collect();
        let x = binned(&[x0, x1, x2]);
        let base = Config {
            n_trees: 300,
            learning_rate: 0.4,
            lambda: 0.5,
            validation_fraction: Some(0.25),
            early_stopping_rounds: 200,
            ..Config::default()
        };
        let sqe = SquaredError;
        let fit = |cfg: &Config| {
            Booster::with_config(cfg.clone())
                .fit(&x, &y, &spec(&sqe))
                .unwrap()
        };
        let m_fixed = fit(&base);
        let mut adaptive_cfg = base.clone();
        adaptive_cfg.early_stopping_adaptive = Some(1.5);
        let m_adaptive = fit(&adaptive_cfg);
        // A smaller patience never retains MORE trees than the fixed fit.
        assert!(
            m_adaptive.trees.len() <= m_fixed.trees.len(),
            "adaptive kept {} > fixed {}",
            m_adaptive.trees.len(),
            m_fixed.trees.len()
        );
        // When the best round is the same, the deployed models are byte-identical.
        if m_adaptive.trees.len() == m_fixed.trees.len() {
            for i in 0..n {
                assert_eq!(
                    predict(&m_fixed, &x, i).to_bits(),
                    predict(&m_adaptive, &x, i).to_bits(),
                    "row {i}: same best round but predictions diverge",
                );
            }
        }
    }

    /// The relative early-stopping improvement predicate: `min_delta == 0.0` is the exact legacy
    /// `dev < best` (strict decrease only — equality is not an improvement); a positive tolerance
    /// requires clearing `best * (1 - min_delta)`; and a zero incumbent admits nothing (deviances are
    /// `>= 0`, so no `dev` is `< 0`).
    #[test]
    fn is_material_improvement_honors_the_relative_tolerance() {
        // min_delta == 0.0 ⇒ any strict decrease improves; equality / increase do not.
        let legacy = Config {
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            ..Config::default()
        };
        assert!(legacy.is_material_improvement(1.0, 0.999_999));
        assert!(!legacy.is_material_improvement(1.0, 1.0));
        assert!(!legacy.is_material_improvement(1.0, 1.000_001));
        // min_delta == 0.1 ⇒ must beat `best` by MORE than 10% relative: dev < best * 0.9.
        let tol = Config {
            early_stopping_min_delta: 0.1,
            interaction_gain_hurdle: 0.0,
            ..Config::default()
        };
        assert!(tol.is_material_improvement(1.0, 0.89)); // 11% better clears the bar
        assert!(!tol.is_material_improvement(1.0, 0.9)); // exactly 10% does NOT (strict `<`)
        assert!(!tol.is_material_improvement(1.0, 0.95)); // 5% noise is ignored
                                                          // best == 0.0 ⇒ nothing improves, under any tolerance (no deviance `>= 0` is `< 0`).
        assert!(!legacy.is_material_improvement(0.0, 0.0));
        assert!(!tol.is_material_improvement(0.0, 0.0));
    }

    /// `early_stopping_min_delta` must be finite and in `[0.0, 1.0)`: the legacy `0.0` and an interior
    /// `0.5` validate, while a negative value, `>= 1.0`, or NaN are rejected (mirrors the
    /// `early_stopping_adaptive` validation style).
    #[test]
    fn config_rejects_out_of_range_early_stopping_min_delta() {
        let with = |d: f64| -> Config {
            Config {
                early_stopping_min_delta: d,
                interaction_gain_hurdle: 0.0,
                ..Config::default()
            }
        };
        assert!(with(0.0).validate().is_ok());
        assert!(with(0.5).validate().is_ok());
        assert!(with(-1e-9).validate().is_err());
        assert!(with(1.0).validate().is_err());
        assert!(with(1.5).validate().is_err());
        assert!(with(f64::NAN).validate().is_err());
    }

    #[test]
    fn config_rejects_out_of_range_interaction_gain_hurdle() {
        let with = |h: f32| -> Config {
            Config {
                interaction_gain_hurdle: h,
                ..Config::default()
            }
        };
        assert!(with(0.0).validate().is_ok());
        assert!(with(1.0).validate().is_ok());
        assert!(with(5.0).validate().is_ok());
        assert!(with(-1e-3).validate().is_err());
        assert!(with(f32::NAN).validate().is_err());
        assert!(with(f32::INFINITY).validate().is_err());
    }

    /// End-to-end (real `fit_single`): a large `early_stopping_min_delta` stops earlier — keeps fewer
    /// trees — than the legacy `0.0` tolerance under otherwise identical params, because epsilon-level
    /// validation improvements no longer advance the best round. On this clean low-noise signal the
    /// validation deviance keeps creeping down every round, so the legacy `dev < best` rule almost
    /// never resets patience (trains to the cap), while a 50%-relative tolerance treats those tiny
    /// late gains as noise and stops soon after the early material drop. The `0.0` count is the
    /// pre-change any-improvement baseline, derived here by fitting rather than hardcoded.
    #[test]
    fn min_delta_stops_earlier_than_legacy_any_improvement() {
        let n = 400usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 13) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| 1.3 * x0[i] - 0.7 * x1[i] + 0.4 * x2[i])
            .collect();
        let x = binned(&[x0, x1, x2]);
        let base = Config {
            n_trees: 120,
            learning_rate: 0.05,
            validation_fraction: Some(0.25),
            early_stopping_rounds: 10,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            ..Config::default()
        };
        let sqe = SquaredError;
        let fit_len = |cfg: &Config| -> usize {
            Booster::with_config(cfg.clone())
                .fit(&x, &y, &spec(&sqe))
                .unwrap()
                .trees
                .len()
        };
        let trees_zero = fit_len(&base); // legacy any-improvement baseline (derived, not hardcoded)
        let mut big = base.clone();
        big.early_stopping_min_delta = 0.5; // require a >50% relative val-deviance drop to reset patience
        let trees_big = fit_len(&big);
        assert!(
            trees_big < trees_zero,
            "min_delta=0.5 kept {trees_big} trees, expected strictly fewer than legacy {trees_zero}",
        );
    }

    /// End-to-end multiclass (real `fit_multiclass`): with a validation carve and a huge
    /// `early_stopping_min_delta`, the softmax early-stopping loop treats nearly every round's tiny
    /// cross-entropy gain as immaterial, so the best round stops advancing and the fit terminates
    /// (each class truncated) well before the `n_trees` cap.
    #[test]
    fn multiclass_min_delta_stops_before_the_n_trees_cap() {
        let n = 300usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 9 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (i % 3) as f32).collect();
        let xb = binned(&[x0, x1, x2]);
        let n_trees_cap = 200u32;
        let config = Config {
            n_trees: n_trees_cap,
            learning_rate: 0.3,
            validation_fraction: Some(0.25),
            early_stopping_rounds: 5,
            early_stopping_min_delta: 0.5, // require a >50% relative deviance drop to reset patience
            ..Config::default()
        };
        let labels: Vec<String> = (0..3).map(|k| k.to_string()).collect();
        let sqe = SquaredError;
        let model = fit_multiclass(
            &config,
            &xb,
            &y,
            3,
            &labels,
            &spec(&sqe),
            &CatEncoderStore::new(),
        )
        .unwrap();
        let kept = model.classes[0].trees.len();
        assert!(
            (kept as u32) < n_trees_cap,
            "multiclass kept {kept} trees, expected an early stop before the {n_trees_cap} cap",
        );
    }

    #[test]
    fn multiclass_fit_with_validation_is_byte_identical_across_thread_counts() {
        // Exercises the update_raw_split + softmax-cross-entropy early-stopping path (validation
        // carve ⇒ covers_all false), which must stay thread-count-independent like the no-val path.
        let n = 360usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 9 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (i % 3) as f32).collect();
        let xb = binned(&[x0, x1, x2]);
        let config = Config {
            n_trees: 80,
            learning_rate: 0.3,
            validation_fraction: Some(0.2),
            early_stopping_rounds: 5,
            early_stopping_adaptive: None,
            ..Config::default()
        };
        let labels: Vec<String> = (0..3).map(|k| k.to_string()).collect();
        let run = |nt: usize| -> MultiClassModel {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let sqe = SquaredError;
                fit_multiclass(
                    &config,
                    &xb,
                    &y,
                    3,
                    &labels,
                    &spec(&sqe),
                    &CatEncoderStore::new(),
                )
                .unwrap()
            })
        };
        let m1 = run(1);
        assert_eq!(m1, run(2));
        assert_eq!(m1, run(8));
        // Sanity: still a valid simplex after the update_raw_split scoring path.
        let proba = m1.predict_proba(&xb).unwrap();
        for row in proba.chunks(3) {
            let s: f32 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "row proba sums to {s}");
        }
    }

    #[test]
    fn multiclass_model_round_trips_through_bincode_and_json() {
        let n = 150usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 10) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 4) as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (i % 3) as f32).collect();
        let xb = binned(&[x0, x1]);
        let config = Config {
            n_trees: 20,
            learning_rate: 0.3,
            ..Config::default()
        };
        let labels = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let sqe = SquaredError;
        let model = fit_multiclass(
            &config,
            &xb,
            &y,
            3,
            &labels,
            &spec(&sqe),
            &CatEncoderStore::new(),
        )
        .unwrap();
        let proba = model.predict_proba(&xb).unwrap();

        // Binary round-trip (magic-prefixed) preserves labels and predictions bit-for-bit.
        let bytes = crate::serialize::encode_multiclass(&model).unwrap();
        assert!(crate::serialize::is_multiclass_bytes(&bytes));
        let back = crate::serialize::decode_multiclass(&bytes).unwrap();
        assert_eq!(back.class_labels, labels);
        assert_eq!(back.predict_proba(&xb).unwrap(), proba);

        // JSON round-trip.
        let json = crate::serialize::encode_multiclass_json(&model).unwrap();
        let back_json = crate::serialize::decode_multiclass_json(&json).unwrap();
        assert_eq!(back_json.predict_proba(&xb).unwrap(), proba);

        // A single-model blob must NOT be mistaken for a multiclass envelope, and vice-versa.
        let single = crate::serialize::encode_model(&model.classes[0]).unwrap();
        assert!(!crate::serialize::is_multiclass_bytes(&single));
        assert!(crate::serialize::decode_multiclass(&single).is_err());
    }

    #[test]
    fn random_strength_fit_is_byte_identical_across_thread_counts() {
        let n = 240usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                (if x0[i] <= 3.0 { -2.0 } else { 3.0 })
                    + if x1[i] <= 2.0 { 1.5 } else { -1.0 }
                    + (i % 7) as f32 * 0.05
            })
            .collect();
        let x = binned(&[x0, x1, x2]);
        let cfg = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 35,
            learning_rate: 0.25,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig {
                random_strength: 0.35,
                ..BoosterConfig::default()
            },
            fit_control: Default::default(),
        };
        let sqe = SquaredError;
        let bytes = |nt: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let model = Booster::with_config(cfg.clone())
                    .fit(&x, &y, &spec(&sqe))
                    .unwrap();
                let serve = crate::data::ServeBinnedMatrix(x.clone());
                let bank = model.explain(&serve, RefMeasure::default()).unwrap();
                assert_exact_decomposition(&model, &bank, &serve).unwrap();
                crate::serialize::encode_model(&model).unwrap()
            })
        };
        let b1 = bytes(1);
        assert!(!b1.is_empty());
        assert_eq!(b1, bytes(2));
        assert_eq!(b1, bytes(8));
    }

    /// Regression for the leaf-refinement lr bug: the incremental Newton step MUST be
    /// `lr`-scaled like a grown leaf (`leaf_values` uses `lr·w`). Without it the step is the
    /// full `lr = 1.0` optimum, which un-shrinks every leaf and overfits (the Armijo guard
    /// cannot catch it — the un-shrunk step still lowers train deviance).
    #[test]
    fn leaf_refine_delta_is_learning_rate_scaled() {
        let full = incremental_leaf_delta(10.0, 10.0, 0.0, 0.0, None, 1.0).unwrap();
        let shrunk = incremental_leaf_delta(10.0, 10.0, 0.0, 0.0, None, 0.1).unwrap();
        assert!(
            (full - (-1.0)).abs() < 1e-6,
            "lr=1 ⇒ full Newton step −G/(H+λ)"
        );
        assert!(
            (shrunk - (-0.1)).abs() < 1e-6,
            "lr=0.1 ⇒ exactly 0.1× the full step"
        );
    }

    /// §07.6 bit-identity: `shrink_refine_delta` (the `leaf_refine`-erosion companion fix) must
    /// return `step` UNTOUCHED — not merely numerically close — whenever `path_smooth <= 0.0`.
    /// Uses `.to_bits()` (stricter than `==`, which would treat two differently-signed zeros or
    /// NaN payloads as equal/unequal in ways that don't actually pin bit-identity) across a
    /// spread of `step`/`credibility_n` values, including edge cases (`n = 0`, `n = ∞`) that
    /// would visibly change the result if the `path_smooth > 0.0` branch executed at all.
    #[test]
    fn shrink_refine_delta_is_bit_identical_to_step_when_path_smooth_is_off() {
        for step in [0.0_f32, -0.0, 1.5, -3.25, f32::MIN_POSITIVE, 1e30, -1e-30] {
            for n in [0.0_f64, 1.0, 1e9, f64::INFINITY] {
                for path_smooth in [0.0_f32, -0.0, -5.0, f32::NEG_INFINITY] {
                    assert_eq!(
                        shrink_refine_delta(step, n, path_smooth).to_bits(),
                        step.to_bits(),
                        "step={step} n={n} path_smooth={path_smooth}"
                    );
                }
            }
        }
    }

    /// §07.6 bit-identity: `shrink_refine_delta` DOES change `step` once `path_smooth > 0.0` —
    /// pins the companion test above is not vacuously true (e.g. a stray typo'd condition that
    /// always takes the `else` branch would pass the inert test too).
    ///
    /// Also pins the 2026-07-23 evidence-normalization amendment's headline semantic,
    /// end-to-end through `credibility_evidence_n` + `shrink_refine_delta` together: a leaf
    /// holding EXACTLY an average per-row hessian share (`h = n · h_bar`) must shrink IDENTICALLY
    /// whether it goes through `Hessian` evidence (normalized to effective rows) or plain
    /// `Count` evidence with the same `n` — at a large (Tweedie-scale) `h_bar = 1000`, not just
    /// `O(1)`, since that scale mismatch is exactly what the `ohlsson_pp` smoke test caught (the
    /// un-normalized version saturates `Z` near 1 regardless of the true row count).
    #[test]
    fn shrink_refine_delta_shrinks_by_the_buhlmann_factor_when_path_smooth_is_on() {
        // n=10, k=10 ⇒ Z=0.5.
        assert!((shrink_refine_delta(2.0, 10.0, 10.0) - 1.0).abs() < 1e-6);
        // n=0 (zero evidence) ⇒ Z=0 -- refine contributes nothing, full deferral.
        assert!(shrink_refine_delta(2.0, 0.0, 10.0).abs() < 1e-6);

        // Normalized-semantics pin (the team's requested amendment test): n=20 rows, average
        // per-row hessian h_bar=1000 (Tweedie-scale) ⇒ this leaf's Σh = 20_000.
        let n = 20u64;
        let h_bar = 1000.0_f64;
        let h_avg = n as f64 * h_bar; // Σh for a leaf at EXACTLY the tree's average per-row h.
        let path_smooth = 10.0_f32;
        let step = 3.0_f32;

        let n_hessian = credibility_evidence_n(CredibilityEvidence::Hessian, h_avg, n, h_bar);
        let n_count = credibility_evidence_n(CredibilityEvidence::Count, h_avg, n, h_bar);
        assert!(
            (n_hessian - n as f64).abs() < 1e-9,
            "average-hessian leaf must normalize to n_eff == n exactly, got {n_hessian}"
        );
        assert_eq!(
            n_hessian, n_count,
            "average-hessian leaf's effective n must equal count evidence's n exactly"
        );

        let shrunk_hessian = shrink_refine_delta(step, n_hessian, path_smooth);
        let shrunk_count = shrink_refine_delta(step, n_count, path_smooth);
        assert_eq!(
            shrunk_hessian, shrunk_count,
            "normalized Hessian evidence and Count evidence must shrink an average leaf identically"
        );

        // Contrast: feeding the RAW (un-normalized) Σh straight into shrink_refine_delta -- the
        // pre-amendment bug -- saturates Z near 1 regardless of n=20, confirming normalization
        // is load-bearing here, not cosmetic (this is the ohlsson_pp failure mode in miniature).
        let shrunk_if_unnormalized = shrink_refine_delta(step, h_avg, path_smooth);
        assert!(
            (shrunk_if_unnormalized - step).abs() < 1e-2,
            "sanity: raw evidence at Tweedie scale must be nearly un-shrunk (Z≈1): got {shrunk_if_unnormalized}, step={step}"
        );
        assert!(
            (shrunk_hessian - shrunk_if_unnormalized).abs() > 0.5,
            "normalized and un-normalized evidence must diverge sharply at Tweedie-scale h_bar: \
             normalized={shrunk_hessian}, unnormalized={shrunk_if_unnormalized}"
        );
    }

    /// §07.2/§07.6: credibility floors are a candidate mask and `path_smooth` is a
    /// value-level clamp on a fixed oblivious structure, so a model fit with both stays
    /// `Exact` and decomposes — and `path_smooth` measurably changes the served leaves.
    #[test]
    fn credibility_floors_and_path_smooth_stay_exact_and_decompose() {
        let n = 160usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                (if x0[i] <= 3.0 { -2.0 } else { 2.5 })
                    + if x1[i] <= 2.0 { 1.25 } else { -0.75 }
                    + if x2[i] <= 2.0 { 0.5 } else { -0.25 }
            })
            .collect();
        let x = binned(&[x0, x1, x2]);
        let cfg = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 25,
            learning_rate: 0.2,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        };
        let sqe = SquaredError;
        let fit = |floor: CredibilityFloor| -> Model {
            let mut s = spec(&sqe);
            s.credibility = floor;
            let model = Booster::with_config(cfg.clone()).fit(&x, &y, &s).unwrap();
            assert_eq!(model.mode, ExactnessMode::Exact);
            let serve = crate::data::ServeBinnedMatrix(x.clone());
            let bank = model.explain(&serve, RefMeasure::default()).unwrap();
            assert_exact_decomposition(&model, &bank, &serve).unwrap();
            model
        };
        let plain = fit(CredibilityFloor::default());
        let floored = fit(CredibilityFloor {
            min_data_in_leaf: 4,
            min_sum_hessian_in_leaf: 1.0,
            min_weight_sum_in_leaf: 4.0,
            path_smooth: 1.5,
        });
        // Both produced real ensembles, and path_smooth shifted at least one prediction.
        let differs =
            (0..n).any(|i| (predict(&plain, &x, i) - predict(&floored, &x, i)).abs() > 1e-6);
        assert!(differs, "path_smooth must change the served leaves");
    }

    /// §07.6 `leaf_refine`-erosion regression (the mandatory companion fix,
    /// `agentH_cred_lambda.md` §2): `incremental_leaf_delta` re-Newtons leaf values with no
    /// awareness of `path_smooth`, so without the fix, enough refine steps walk a shrunk leaf
    /// back toward its own unshrunk per-leaf optimum, silently defeating the shrinkage the
    /// instant `leaf_refine_steps > 0`. `n_trees = 1` isolates exactly ONE grow+refine cycle so
    /// the comparison is not muddied by later rounds. Compares leaf spread (max − min leaf
    /// value) across the 2×2 of {path_smooth off/on} × {no refine / many refine steps} on the
    /// SAME data:
    /// - `A` = shrunk, no refine (grow-time-only shrinkage, the reference "how shrunk").
    /// - `B` = unshrunk, with refine (refine's natural target with NO credibility pull at all —
    ///   the erosion DESTINATION an unfixed shrunk-refine would drift toward).
    /// - `C` = shrunk, with refine (the fit under test).
    ///
    /// The regression this guards: `C` drifting from `A` toward `B` merely because more Newton
    /// steps ran. With the fix, `C` must stay much closer to `A` than to `B`.
    #[test]
    fn leaf_refine_does_not_erode_path_smooth_shrinkage() {
        let n = 200usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                (if x0[i] <= 3.0 { -4.0 } else { 5.0 })
                    + if x1[i] <= 2.0 { 2.0 } else { -1.5 }
                    + if x2[i] <= 2.0 { 1.0 } else { -0.5 }
            })
            .collect();
        let x = binned(&[x0, x1, x2]);
        let sqe = SquaredError;
        let leaf_spread = |model: &Model| -> f32 {
            let (_, tree) = model.trees.first().expect("a tree");
            let lo = tree.leaves.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = tree
                .leaves
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            hi - lo
        };
        let fit = |path_smooth: f32, leaf_refine_steps: u8| -> Model {
            let cfg = Config {
                lambda_scale_invariant: false,
                max_delta_step_gated: GatedStepPolicy::Objective,
                n_trees: 1,
                learning_rate: 1.0,
                lambda: 1.0,
                min_split_gain: 0.0,
                max_delta_step: None,
                sampling: Default::default(),
                hist_precision: Default::default(),
                l1_leaf: 0.0,
                colsample_bytree: 1.0,
                learning_rate_decay: 0.0,
                validation_fraction: None,
                early_stopping_rounds: 50,
                early_stopping_adaptive: None,
                early_stopping_min_delta: 0.0,
                interaction_gain_hurdle: 0.0,
                interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
                leaf_refine_steps,
                leaf_refine_backtracks: 8,
                refine_closed_form_tier2: false,
                incremental_mu: false,
                boosters: Default::default(),
                fit_control: Default::default(),
            };
            let mut s = spec(&sqe);
            s.credibility = CredibilityFloor {
                path_smooth,
                ..CredibilityFloor::default()
            };
            Booster::with_config(cfg).fit(&x, &y, &s).unwrap()
        };

        // path_smooth = 500 makes these ~25-row leaves "thin" relative to the pseudo-count k
        // (Z0 = n/(n+k) is small), which is exactly the regime `shrink_refine_delta`'s per-step
        // damping protects well: SquaredError's Newton step reaches its exact (unshrunk)
        // optimum in ONE step regardless of the starting point (h is round/value-invariant),
        // so scaling that step by a FIXED Z every iteration decays the remaining shrinkage
        // geometrically as `(1-Z0)^(steps+1)` -- slow when Z0 is small (thin leaves), fast when
        // Z0 is large (confident leaves, which need the protection less). Verified empirically:
        // reverting `shrink_refine_delta` to a no-op collapses this SAME comparison to
        // drift ≈ 0.96-1.0 regardless of path_smooth, confirming this test does detect the
        // erosion regression rather than passing vacuously.
        let a_shrunk_no_refine = leaf_spread(&fit(500.0, 0));
        let b_unshrunk_with_refine = leaf_spread(&fit(0.0, 5));
        let c_shrunk_with_refine = leaf_spread(&fit(500.0, 5));

        let erosion_span = (b_unshrunk_with_refine - a_shrunk_no_refine).abs();
        assert!(
            erosion_span > 1e-2,
            "fixture must separate the shrunk and unshrunk-with-refine regimes: A={a_shrunk_no_refine}, B={b_unshrunk_with_refine}"
        );
        let drift_fraction = (c_shrunk_with_refine - a_shrunk_no_refine).abs() / erosion_span;
        assert!(
            drift_fraction < 0.5,
            "leaf_refine eroded path_smooth's shrinkage: A(shrunk,no-refine)={a_shrunk_no_refine}, \
             B(unshrunk,with-refine)={b_unshrunk_with_refine}, C(shrunk,with-refine)={c_shrunk_with_refine} \
             drifted {drift_fraction:.2} of the way from A to B (must stay under 0.5)"
        );
    }

    /// §07.6 end-to-end, through the PRODUCTION-DEFAULT closed form: Poisson + a severely
    /// zero-inflated fixture (half the rows never claim) exercises BOTH changes together via
    /// `refine_leaves_loglink_closed`'s Tier-2 path (`refine_closed_form_tier2 = true`, the
    /// shipped default; Poisson's own `max_delta_step = Some(0.7)` lets Tier-2 engage without
    /// setting anything explicitly) — `credibility_evidence_for_loss` auto-selects Hessian
    /// evidence for Poisson, and `shrink_refine_delta` protects that shrinkage through refine.
    /// Same A/B/C drift comparison as the SquaredError regression test above.
    #[test]
    fn poisson_zero_inflated_leaf_refine_does_not_erode_path_smooth_shrinkage() {
        let n = 200usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                if x0[i] <= 3.0 {
                    0.0 // zero-inflated segment: never claims.
                } else {
                    5.0 + if x1[i] <= 2.0 { 2.0 } else { 0.0 }
                        + if x2[i] <= 2.0 { 1.0 } else { 0.0 }
                }
            })
            .collect();
        let x = binned(&[x0, x1, x2]);
        let poisson = Poisson;
        let leaf_spread = |model: &Model| -> f32 {
            let (_, tree) = model.trees.first().expect("a tree");
            let lo = tree.leaves.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = tree
                .leaves
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            hi - lo
        };
        let fit = |path_smooth: f32, leaf_refine_steps: u8| -> Model {
            let cfg = Config {
                lambda_scale_invariant: false,
                max_delta_step_gated: GatedStepPolicy::Objective,
                n_trees: 1,
                learning_rate: 1.0,
                lambda: 1.0,
                min_split_gain: 0.0,
                max_delta_step: None,
                sampling: Default::default(),
                hist_precision: Default::default(),
                l1_leaf: 0.0,
                colsample_bytree: 1.0,
                learning_rate_decay: 0.0,
                validation_fraction: None,
                early_stopping_rounds: 50,
                early_stopping_adaptive: None,
                early_stopping_min_delta: 0.0,
                interaction_gain_hurdle: 0.0,
                interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
                leaf_refine_steps,
                leaf_refine_backtracks: 8,
                refine_closed_form_tier2: true,
                incremental_mu: false,
                boosters: Default::default(),
                fit_control: Default::default(),
            };
            let mut s = spec(&poisson);
            s.credibility = CredibilityFloor {
                path_smooth,
                ..CredibilityFloor::default()
            };
            Booster::with_config(cfg).fit(&x, &y, &s).unwrap()
        };

        // Same rationale as the SquaredError regression test above (large path_smooth relative
        // to leaf evidence ⇒ small Z0 ⇒ slow per-step erosion decay). Verified empirically
        // (2026-07-23, after the evidence-normalization amendment): reverting
        // `shrink_refine_delta` to a no-op raises drift at these SAME parameters from ≈0.19 to
        // ≈0.75-0.88, confirming this test still detects the erosion regression on the
        // production-default Tier-2 closed-form path post-normalization.
        let a_shrunk_no_refine = leaf_spread(&fit(500.0, 0));
        let b_unshrunk_with_refine = leaf_spread(&fit(0.0, 5));
        let c_shrunk_with_refine = leaf_spread(&fit(500.0, 5));

        let erosion_span = (b_unshrunk_with_refine - a_shrunk_no_refine).abs();
        assert!(
            erosion_span > 1e-2,
            "fixture must separate the shrunk and unshrunk-with-refine regimes: A={a_shrunk_no_refine}, B={b_unshrunk_with_refine}"
        );
        let drift_fraction = (c_shrunk_with_refine - a_shrunk_no_refine).abs() / erosion_span;
        assert!(
            drift_fraction < 0.5,
            "leaf_refine eroded path_smooth's Hessian-evidence shrinkage: A(shrunk,no-refine)={a_shrunk_no_refine}, \
             B(unshrunk,with-refine)={b_unshrunk_with_refine}, C(shrunk,with-refine)={c_shrunk_with_refine} \
             drifted {drift_fraction:.2} of the way from A to B (must stay under 0.5)"
        );
    }

    /// Task-A `incremental_mu`: the multiplicative `mu = exp(F)` cache maintained across rounds (see
    /// [`INCREMENTAL_MU_REFRESH_ROUNDS`]) tracks the exact per-round `exp(F)` closely enough that a
    /// full WEIGHTED Poisson fit with the cache ON matches the exact (OFF) fit — same tree COUNT and
    /// predictions within a small relative drift. NOT bit-identical: the cache carries bounded f64
    /// multiplicative drift between refreshes, so `grad_hess`'s `(g, h)` differ from the exact pass at
    /// the ~1e-7 level, shifting leaf VALUES (not, on this well-separated fixture, the split STRUCTURE
    /// — hence the equal tree count). The fit spans `n_trees > INCREMENTAL_MU_REFRESH_ROUNDS`, so at
    /// least one mid-fit refresh is exercised.
    #[test]
    fn incremental_mu_matches_exact_poisson_fit_within_drift() {
        let n = 400usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        // Non-negative integer counts with clear feature structure + a deterministic wiggle (keeps
        // the fit finding admissible splits well past the refresh boundary).
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let base = (x0[i] as i32)
                    + 2 * (x1[i] as i32)
                    + (x2[i] as i32)
                    + ((i * 7 + i % 13) % 6) as i32;
                base.max(0) as f32
            })
            .collect();
        // Varying positive sample weights — the deploy path always carries exposure weights, and Σw
        // folds into g/h, so the cache must stay faithful under non-unit weights.
        let w: Vec<f32> = (0..n).map(|i| 0.5 + (i % 4) as f32 * 0.4).collect();
        let x = binned(&[x0, x1, x2]);
        let cfg = |incremental_mu: bool| Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 80,
            learning_rate: 0.1,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 4,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: true,
            incremental_mu,
            boosters: Default::default(),
            fit_control: Default::default(),
        };
        let poisson = Poisson;
        let fit = |inc: bool| -> Model {
            let mut s = spec(&poisson);
            s.weight = Some(&w);
            Booster::with_config(cfg(inc)).fit(&x, &y, &s).unwrap()
        };
        let exact = fit(false);
        let cached = fit(true);
        assert!(
            exact.trees.len() > usize::try_from(INCREMENTAL_MU_REFRESH_ROUNDS).unwrap(),
            "fixture should train past a refresh boundary (got {} trees)",
            exact.trees.len()
        );
        assert_eq!(
            exact.trees.len(),
            cached.trees.len(),
            "incremental_mu must not change the tree count on a structurally-stable fixture"
        );
        let mut max_rel = 0.0_f64;
        for i in 0..n {
            let pe = f64::from(poisson.pred_from_raw(predict(&exact, &x, i) as f32));
            let pc = f64::from(poisson.pred_from_raw(predict(&cached, &x, i) as f32));
            max_rel = max_rel.max((pe - pc).abs() / pe.abs().max(1e-9));
        }
        // Empirically ~1e-7 on this fixture; the bound documents the exactness class (drift, not zero).
        assert!(
            max_rel < 1e-5,
            "incremental_mu prediction drift {max_rel:.3e} exceeds tolerance"
        );
    }

    #[test]
    fn ridge_refit_improves_deviance_and_preserves_exact_determinism() {
        let n = 180usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                (if x0[i] <= 3.0 { -2.0 } else { 2.5 })
                    + if x1[i] <= 2.0 { 1.25 } else { -0.75 }
                    + if x2[i] <= 2.0 { 0.5 } else { -0.25 }
                    + (i % 9) as f32 * 0.02
            })
            .collect();
        let x = binned(&[x0, x1, x2]);
        let base_cfg = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 8,
            learning_rate: 0.12,
            lambda: 2.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        };
        let refit_cfg = Config {
            boosters: BoosterConfig {
                refit_leaves: RefitSpec::Ridge {
                    l2: 1.0e-3,
                    max_iter: 3,
                    every_k_trees: None,
                },
                ..BoosterConfig::default()
            },
            ..base_cfg.clone()
        };
        let sqe = SquaredError;
        let base = Booster::with_config(base_cfg)
            .fit(&x, &y, &spec(&sqe))
            .unwrap();
        let refit = Booster::with_config(refit_cfg.clone())
            .fit(&x, &y, &spec(&sqe))
            .unwrap();
        let w = vec![1.0_f32; y.len()];
        let base_pred = base.predict_binned(&x, None).unwrap();
        let refit_pred = refit.predict_binned(&x, None).unwrap();
        let base_dev = sqe.deviance(&y, &base_pred, &w).unwrap();
        let refit_dev = sqe.deviance(&y, &refit_pred, &w).unwrap();
        assert!(
            refit_dev < base_dev,
            "refit deviance {refit_dev} should improve base {base_dev}"
        );

        let bytes = |nt: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let model = Booster::with_config(refit_cfg.clone())
                    .fit(&x, &y, &spec(&sqe))
                    .unwrap();
                let serve = crate::data::ServeBinnedMatrix(x.clone());
                let bank = model.explain(&serve, RefMeasure::default()).unwrap();
                assert_exact_decomposition(&model, &bank, &serve).unwrap();
                crate::serialize::encode_model(&model).unwrap()
            })
        };
        let b1 = bytes(1);
        assert_eq!(b1, bytes(2));
        assert_eq!(b1, bytes(8));
    }

    #[test]
    fn ridge_every_k_refit_early_stop_truncation_still_gets_a_final_refit() {
        // Truncation-restore + last_refit_tree_count reset, combined: with RefitSpec::Ridge {
        // every_k_trees: Some(k) }, the last mid-loop refit almost always lands AFTER the best
        // round (patience keeps training past it), so its leaf values are jointly solved with a
        // LONGER tree set than the truncated/restored prefix keeps. Post-fix, truncation resets
        // last_refit_tree_count so should_refit_at_end unconditionally re-solves the kept prefix.
        // As with the AGBM case, the exported model must then match a fresh fit capped at exactly
        // the best round's tree count (which, being un-early-stopped, naturally ends with its own
        // correct final refit).
        let n = 320usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 9) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 6) as f32).collect();
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut noise = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1_u64 << 53) as f64) as f32 - 0.5
        };
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 4.0 { 1.4 } else { -1.0 };
                let b = if x1[i] <= 2.0 { 0.6 } else { -0.4 };
                a + b + noise() * 1.2
            })
            .collect();
        let x = binned(&[x0, x1]);
        let mask: Vec<bool> = (0..n).map(|i| i % 5 == 0).collect(); // fixed, deterministic 20% holdout
        let sqe = SquaredError;
        let mut fit_spec = spec(&sqe);
        fit_spec.fixed_holdout = Some(&mask);

        let cfg = |n_trees: u32| Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees,
            learning_rate: 0.15,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None, // fixed_holdout drives the split instead
            early_stopping_rounds: 8,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig {
                refit_leaves: RefitSpec::Ridge {
                    l2: 1.0,
                    max_iter: 3,
                    every_k_trees: Some(7),
                },
                ..BoosterConfig::default()
            },
            fit_control: Default::default(),
        };

        let full = Booster::with_config(cfg(90))
            .fit(&x, &y, &fit_spec)
            .unwrap();
        let best_count = full.trees.len();
        assert!(
            best_count > 0 && best_count < 90,
            "test needs the best round to land strictly before the n_trees cap, got {best_count}"
        );

        let control = Booster::with_config(cfg(best_count as u32))
            .fit(&x, &y, &fit_spec)
            .unwrap();
        assert_eq!(
            control.trees.len(),
            best_count,
            "control run must not itself stop before its own n_trees cap"
        );
        assert_eq!(
            crate::serialize::encode_model(&full).unwrap(),
            crate::serialize::encode_model(&control).unwrap(),
            "truncated Ridge/every_k_trees model must match a fresh fit stopped exactly at the \
             best round (i.e. it must have received its own final refit)"
        );
    }

    #[test]
    fn ridge_refit_is_near_noop_on_exact_fit() {
        let n = 64usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 2 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| ((i / 2) % 2 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                (if x0[i] <= 1.0 { -2.0 } else { 3.0 }) + if x1[i] <= 1.0 { 0.5 } else { -1.5 }
            })
            .collect();
        let x = binned(&[x0, x1]);
        let base_cfg = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 4,
            learning_rate: 1.0,
            lambda: 0.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        };
        let refit_cfg = Config {
            boosters: BoosterConfig {
                refit_leaves: RefitSpec::Ridge {
                    l2: 0.0,
                    max_iter: 2,
                    every_k_trees: Some(2),
                },
                ..BoosterConfig::default()
            },
            ..base_cfg.clone()
        };
        let sqe = SquaredError;
        let base = Booster::with_config(base_cfg)
            .fit(&x, &y, &spec(&sqe))
            .unwrap();
        let refit = Booster::with_config(refit_cfg)
            .fit(&x, &y, &spec(&sqe))
            .unwrap();
        let base_pred = base.predict_binned(&x, None).unwrap();
        let refit_pred = refit.predict_binned(&x, None).unwrap();
        for (a, b) in base_pred.iter().zip(refit_pred) {
            assert!(
                (f64::from(*a) - f64::from(b)).abs() < 1.0e-5,
                "refit moved an exact score: {a} vs {b}"
            );
        }
    }

    #[test]
    fn fully_corrective_refit_ignores_rows_outside_train_rows() {
        // §H6: fully_corrective_refit's normal equations AND its backtracking accept/reject
        // deviance must only ever read `train_rows`. Isolates this from the surrounding boost
        // loop / early-stopping bookkeeping (which could confound a full-fit comparison via
        // early-stop timing) by calling the refit directly: two scenarios share EVERYTHING (x,
        // trees, train rows' y) except the y of rows OUTSIDE train_rows, which differ wildly.
        // If the refit is train-only, the solved leaf values must be byte-identical.
        let n = 100usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 8) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let seed_y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 3.0 { 2.0 } else { -1.5 };
                let b = if x1[i] <= 2.0 { 0.7 } else { -0.4 };
                a + b + (i % 7) as f32 * 0.05
            })
            .collect();
        let x = binned(&[x0, x1]);
        let sqe = SquaredError;
        let model = Booster::with_config(Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 6,
            learning_rate: 0.3,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig::default(),
            fit_control: Default::default(),
        })
        .fit(&x, &seed_y, &spec(&sqe))
        .unwrap();
        assert!(!model.trees.is_empty(), "need real trees to refit");

        let train_rows: Vec<usize> = (0..80).collect(); // rows 80..100 are the "validation" carve
        let weight = vec![1.0_f32; n];
        let mut raw = vec![0.0_f32; n];
        model.score_trees(&x, None, &mut raw).unwrap();

        let y_a = seed_y.clone();
        let mut y_b = seed_y.clone();
        for slot in y_b.iter_mut().skip(80) {
            *slot = 12345.0; // wildly different target on the excluded rows only
        }

        let fit_spec = spec(&sqe);
        let refit_spec = RefitSpec::Ridge {
            l2: 1.0,
            max_iter: 5,
            every_k_trees: None,
        };
        let run = |y: &[f32]| -> Vec<(f32, ObliviousTree)> {
            let mut trees = model.trees.clone();
            let mut raw = raw.clone();
            let problem = RefitProblem {
                spec: &fit_spec,
                x: &x,
                y,
                weight: &weight,
                offset: None,
                f0: model.f0,
                monotone: None,
                train_rows: &train_rows,
            };
            fully_corrective_refit(&refit_spec, &problem, &mut trees, &mut raw).unwrap();
            trees
        };
        assert_eq!(
            run(&y_a),
            run(&y_b),
            "refit must be blind to y outside train_rows"
        );
    }

    #[test]
    fn agbm_fit_is_alpha_folded_exact_and_thread_deterministic() {
        let n = 220usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 7 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                (if x0[i] <= 3.0 { -1.5 } else { 2.0 })
                    + if x1[i] <= 2.0 { 1.0 } else { -0.75 }
                    + (i % 11) as f32 * 0.03
            })
            .collect();
        let x = binned(&[x0, x1, x2]);
        let cfg = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 30,
            learning_rate: 0.18,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig {
                nesterov: NesterovSpec::Agbm {
                    momentum_correction: false,
                },
                ..BoosterConfig::default()
            },
            fit_control: Default::default(),
        };
        let sqe = SquaredError;
        let bytes = |nt: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let model = Booster::with_config(cfg.clone())
                    .fit(&x, &y, &spec(&sqe))
                    .unwrap();
                assert!(
                    model
                        .trees
                        .iter()
                        .any(|(alpha, _)| (*alpha - 1.0).abs() > 1.0e-6),
                    "AGBM should fold non-unit alphas into the plain model"
                );
                let serve = crate::data::ServeBinnedMatrix(x.clone());
                let bank = model.explain(&serve, RefMeasure::default()).unwrap();
                assert_exact_decomposition(&model, &bank, &serve).unwrap();
                crate::serialize::encode_model(&model).unwrap()
            })
        };
        let b1 = bytes(1);
        assert_eq!(b1, bytes(2));
        assert_eq!(b1, bytes(8));
    }

    #[test]
    fn agbm_momentum_correction_stays_exact() {
        let n = 120usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| (if x0[i] <= 3.0 { 0.0 } else { 2.0 }) + (i % 5) as f32 * 0.1)
            .collect();
        let x = binned(&[x0, x1]);
        let sqe = SquaredError;
        let model = Booster::with_config(Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 8,
            learning_rate: 0.2,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig {
                nesterov: NesterovSpec::Agbm {
                    momentum_correction: true,
                },
                ..BoosterConfig::default()
            },
            fit_control: Default::default(),
        })
        .fit(&x, &y, &spec(&sqe))
        .unwrap();
        assert!(
            model.trees.len() > 8,
            "momentum correction should append correction trees when splits remain"
        );
        let serve = crate::data::ServeBinnedMatrix(x);
        let bank = model.explain(&serve, RefMeasure::default()).unwrap();
        assert_exact_decomposition(&model, &bank, &serve).unwrap();
    }

    #[test]
    fn agbm_early_stop_truncation_restores_the_best_round_exactly() {
        // Truncation-restore: AGBM overwrites EVERY tree's alpha with the lookahead mix every
        // round, including rounds AFTER the best one (patience keeps training past it). Post-fix,
        // the exported (truncated) model must be byte-identical to a FRESH fit stopped exactly at
        // the best round's tree count — proving the retained trees carry the alphas that actually
        // produced the recorded best validation deviance, not whatever the final patience round
        // last mixed them to.
        let n = 300usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 9) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 6) as f32).collect();
        // Deterministic pseudo-noise (xorshift64) so validation deviance genuinely fluctuates
        // round to round — the peak is very unlikely to land on the very last trained round.
        let mut state = 0x243F_6A88_85A3_08D3_u64;
        let mut noise = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1_u64 << 53) as f64) as f32 - 0.5
        };
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 4.0 { 1.6 } else { -1.1 };
                let b = if x1[i] <= 2.0 { 0.7 } else { -0.5 };
                a + b + noise() * 1.4
            })
            .collect();
        let x = binned(&[x0, x1]);
        let mask: Vec<bool> = (0..n).map(|i| i % 5 == 0).collect(); // fixed, deterministic 20% holdout
        let sqe = SquaredError;
        let mut fit_spec = spec(&sqe);
        fit_spec.fixed_holdout = Some(&mask);

        let cfg = |n_trees: u32| Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees,
            learning_rate: 0.2,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None, // fixed_holdout drives the split instead
            early_stopping_rounds: 8,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig {
                nesterov: NesterovSpec::Agbm {
                    momentum_correction: false,
                },
                ..BoosterConfig::default()
            },
            fit_control: Default::default(),
        };

        let full = Booster::with_config(cfg(80))
            .fit(&x, &y, &fit_spec)
            .unwrap();
        let best_count = full.trees.len();
        assert!(
            best_count > 0 && best_count < 80,
            "test needs the best round to land strictly before the n_trees cap, got {best_count}"
        );

        // A fresh fit capped at EXACTLY best_count trees never trains past its own best round
        // (patience's gap only starts growing the round AFTER an improvement), so it needs no
        // restore at all — its output is definitionally "the state that produced the best
        // recorded deviance" for `full` to be compared against.
        let control = Booster::with_config(cfg(best_count as u32))
            .fit(&x, &y, &fit_spec)
            .unwrap();
        assert_eq!(
            control.trees.len(),
            best_count,
            "control run must not itself stop before its own n_trees cap"
        );
        assert_eq!(
            crate::serialize::encode_model(&full).unwrap(),
            crate::serialize::encode_model(&control).unwrap(),
            "truncated AGBM model must match a fresh fit stopped exactly at the best round"
        );
    }

    #[test]
    fn dart_drop_rate_zero_is_byte_identical_to_default() {
        let (x0, x1, y) = additive_2feat(96);
        let x = binned(&[x0, x1]);
        let base_cfg = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 25,
            learning_rate: 0.2,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        };
        let dart_cfg = Config {
            boosters: BoosterConfig {
                dart: Some(DartSpec {
                    drop_rate: 0.0,
                    normalize: true,
                }),
                ..BoosterConfig::default()
            },
            ..base_cfg.clone()
        };
        let sqe = SquaredError;
        let base = Booster::with_config(base_cfg)
            .fit(&x, &y, &spec(&sqe))
            .unwrap();
        let dart = Booster::with_config(dart_cfg)
            .fit(&x, &y, &spec(&sqe))
            .unwrap();
        assert_eq!(
            crate::serialize::encode_model(&base).unwrap(),
            crate::serialize::encode_model(&dart).unwrap()
        );
    }

    #[test]
    fn dart_normalized_fit_is_exact_and_thread_deterministic() {
        let n = 220usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 7 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                (if x0[i] <= 3.0 { -1.0 } else { 2.25 })
                    + if x1[i] <= 2.0 { 1.0 } else { -0.5 }
                    + (i % 13) as f32 * 0.02
            })
            .collect();
        let x = binned(&[x0, x1, x2]);
        let cfg = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 35,
            learning_rate: 0.2,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig {
                dart: Some(DartSpec {
                    drop_rate: 0.45,
                    normalize: true,
                }),
                ..BoosterConfig::default()
            },
            fit_control: Default::default(),
        };
        let sqe = SquaredError;
        let bytes = |nt: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let model = Booster::with_config(cfg.clone())
                    .fit(&x, &y, &spec(&sqe))
                    .unwrap();
                assert!(
                    model
                        .trees
                        .iter()
                        .any(|(alpha, _)| (*alpha - 1.0).abs() > 1.0e-6),
                    "DART normalization should fold non-unit alphas into the model"
                );
                let serve = crate::data::ServeBinnedMatrix(x.clone());
                let bank = model.explain(&serve, RefMeasure::default()).unwrap();
                assert_exact_decomposition(&model, &bank, &serve).unwrap();
                crate::serialize::encode_model(&model).unwrap()
            })
        };
        let b1 = bytes(1);
        assert_eq!(b1, bytes(2));
        assert_eq!(b1, bytes(8));
    }

    #[test]
    fn agbm_incremental_lookahead_matches_full_realphas_walk() {
        // §C3 algebra check: `agbm_lookahead_raw`'s O(n) blend of `raw`/`raw_prev` must match a
        // full `raw_from_tree_alphas` O(n·T) re-walk at the lookahead alphas, at every round of a
        // multi-round trajectory (growing tree count, the real `agbm_beta` schedule, a nonzero
        // per-row offset so the f0+offset cancellation in the derivation is actually exercised).
        // This directly pins the ALGEBRA (not bit-identity with the old implementation, which is
        // not claimed — see `NesterovSpec::Agbm`'s doc).
        let n = 12usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 4) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 3) as f32).collect();
        let x = binned(&[x0, x1]);
        let f0 = 0.3_f32;
        let offset: Vec<f32> = (0..n).map(|i| 0.05 * (i as f32 - 6.0)).collect();

        // A small pool of distinguishable, genuinely-splitting trees, reused as a growing prefix
        // each round (exactly how `fit_single`'s `trees` only ever grows by appending).
        let pool: Vec<ObliviousTree> = (0..5)
            .map(|k| ObliviousTree {
                splits: vec![crate::engine::Split {
                    axis: (k % 2) as u32,
                    bin_le: (k % 2 + 1) as u8,
                    missing_left: k % 2 == 0,
                }],
                leaves: {
                    let mut l = [0.0_f32; 8];
                    for (i, v) in l.iter_mut().enumerate() {
                        *v = 0.1 * (k as f32 + 1.0) * (i as f32 - 3.5);
                    }
                    l.to_vec()
                },
                depth: 1,
            })
            .collect();
        let trees_at = |alphas: &[f32]| -> Vec<(f32, ObliviousTree)> {
            pool.iter()
                .take(alphas.len())
                .zip(alphas)
                .map(|(tree, &a)| (a, tree.clone()))
                .collect()
        };

        let mut current: Vec<f32> = Vec::new();
        let mut previous: Vec<f32> = Vec::new();
        for t in 0..pool.len() {
            let beta = agbm_beta(t as u32);

            // The two invariants `agbm_lookahead_raw`'s derivation relies on: `raw` is the walk at
            // the CURRENT (pre-lookahead) alphas, `raw_prev` the walk at last round's alphas.
            let raw = raw_from_tree_alphas(f0, Some(&offset), &x, &trees_at(&current)).unwrap();
            let raw_prev =
                raw_from_tree_alphas(f0, Some(&offset), &x, &trees_at(&previous)).unwrap();

            let lookahead = combine_alphas(&current, &previous, beta).unwrap();
            let old_fit_raw =
                raw_from_tree_alphas(f0, Some(&offset), &x, &trees_at(&lookahead)).unwrap();
            let new_fit_raw = agbm_lookahead_raw(&raw, &raw_prev, beta).unwrap();

            assert_eq!(old_fit_raw.len(), new_fit_raw.len());
            for (row, (&o, &nv)) in old_fit_raw.iter().zip(&new_fit_raw).enumerate() {
                assert!(
                    (f64::from(o) - f64::from(nv)).abs() < 1.0e-4,
                    "round {t} row {row}: full-rewalk={o} incremental={nv}"
                );
            }

            // Advance exactly as `fit_single` does: `previous` becomes this round's pre-lookahead
            // `current`, `current` becomes the lookahead mix with the new tree joining at alpha 1.0.
            previous = current;
            current = lookahead;
            current.push(1.0);
        }
    }

    #[test]
    fn dart_incremental_reconstruction_matches_full_realphas_walk() {
        // §C3 algebra check: `raw_plus_dart_round`'s O(n·(#dropped+1)) reconstruction must match a
        // full `raw_from_tree_alphas` O(n·T) re-walk at every round of a multi-round trajectory —
        // growing tree count, varying drop patterns (including a zero-dropped round, exercising
        // `apply_dart_normalization`'s early-return path), a nonzero per-row offset.
        let n = 12usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 4) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 3) as f32).collect();
        let x = binned(&[x0, x1]);
        let f0 = -0.2_f32;
        let offset: Vec<f32> = (0..n).map(|i| 0.03 * (i as f32 - 5.0)).collect();

        let pool: Vec<ObliviousTree> = (0..6)
            .map(|k| ObliviousTree {
                splits: vec![crate::engine::Split {
                    axis: (k % 2) as u32,
                    bin_le: (k % 2 + 1) as u8,
                    missing_left: k % 2 == 1,
                }],
                leaves: {
                    let mut l = [0.0_f32; 8];
                    for (i, v) in l.iter_mut().enumerate() {
                        *v = 0.07 * (k as f32 + 1.0) * (i as f32 - 3.0);
                    }
                    l.to_vec()
                },
                depth: 1,
            })
            .collect();
        let drop_patterns: Vec<Vec<bool>> = vec![
            vec![],
            vec![false],
            vec![true, false],
            vec![false, false, false],
            vec![true, false, true, false],
            vec![false, true, false, false, true],
        ];
        let dart = DartSpec {
            drop_rate: 0.4,
            normalize: true,
        };

        let mut trees: Vec<(f32, ObliviousTree)> = Vec::new();
        for (t, new_tree) in pool.iter().enumerate() {
            let drops = &drop_patterns[t];
            assert_eq!(
                drops.len(),
                trees.len(),
                "fixture drop pattern round {t} length"
            );

            let raw = raw_from_tree_alphas(f0, Some(&offset), &x, &trees).unwrap();
            let fit_raw = raw_minus_dropped(&raw, &x, &trees, drops).unwrap();

            // `apply_dart_normalization` mutates `trees` in place, exactly as the real call site
            // does; both the reference re-walk and the incremental reconstruction below read the
            // SAME post-normalization `trees`.
            let new_alpha = apply_dart_normalization(&mut trees, drops, &dart).unwrap();

            let mut trees_with_new = trees.clone();
            trees_with_new.push((new_alpha, new_tree.clone()));
            let old_raw = raw_from_tree_alphas(f0, Some(&offset), &x, &trees_with_new).unwrap();

            let new_raw =
                raw_plus_dart_round(fit_raw, &x, &trees, drops, new_tree, new_alpha).unwrap();

            assert_eq!(old_raw.len(), new_raw.len());
            for (row, (&o, &nv)) in old_raw.iter().zip(&new_raw).enumerate() {
                assert!(
                    (f64::from(o) - f64::from(nv)).abs() < 1.0e-4,
                    "round {t} row {row}: full-rewalk={o} incremental={nv}"
                );
            }

            trees.push((new_alpha, new_tree.clone()));
        }
    }

    #[test]
    fn outer_bag_model_soup_stays_exact_and_thread_deterministic() {
        let n = 180usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 9 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 7 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                (if x0[i] <= 4.0 { 1.5 } else { -2.0 })
                    + if x1[i] <= 3.0 { 2.0 } else { -0.5 }
                    + if x2[i] <= 2.0 { 0.75 } else { -0.25 }
                    + (i % 11) as f32 * 0.03
            })
            .collect();
        let x = binned(&[x0, x1, x2]);
        let cfg = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 25,
            learning_rate: 0.25,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig {
                ensemble: EnsembleSpec::OuterBag {
                    n_bags: 3,
                    bag_subsample: 1.0,
                    cell_refit: None,
                },
                ..BoosterConfig::default()
            },
            fit_control: Default::default(),
        };
        let sqe = SquaredError;
        let bytes = |nt: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let model = Booster::with_config(cfg.clone())
                    .fit(&x, &y, &spec(&sqe))
                    .unwrap();
                assert!(
                    model
                        .trees
                        .iter()
                        .any(|(alpha, _)| (*alpha - 1.0).abs() > 1.0e-6),
                    "OuterBag should fold convex member weights into tree alphas"
                );
                let serve = crate::data::ServeBinnedMatrix(x.clone());
                let bank = model.explain(&serve, RefMeasure::default()).unwrap();
                assert_exact_decomposition(&model, &bank, &serve).unwrap();
                crate::serialize::encode_model(&model).unwrap()
            })
        };
        let b1 = bytes(1);
        assert_eq!(b1, bytes(2));
        assert_eq!(b1, bytes(8));
    }

    #[test]
    fn stratified_apportionment_always_sums_to_exactly_k() {
        // Exact-k invariant across tiny/singleton/near-n group shapes — the property the
        // sampler above leans on, checked independently of any RNG draw.
        let cases: &[(&[usize], usize)] = &[
            (&[96, 4], 80),           // rare-event shape: 4 positives / 100 rows, 0.8 subsample
            (&[1, 99], 1),            // singleton stratum, k=1
            (&[1, 99], 80),           // singleton stratum, k close to n
            (&[1, 1, 1, 1, 1], 3),    // every stratum a singleton
            (&[50, 50], 100),         // k == n_total (full-size draw)
            (&[7], 5),                // a single stratum
            (&[3, 3, 4], 0),          // k == 0
            (&[10, 10, 10], 1),       // tiny k spread across equal groups
            (&[2, 3, 5, 7, 11], 200), // k > n_total: must clamp, not panic or over-fill
        ];
        for &(lens, k) in cases {
            let counts = stratified_apportionment(lens, k);
            assert_eq!(counts.len(), lens.len(), "lens={lens:?} k={k}");
            let n_total: usize = lens.iter().sum();
            let total: usize = counts.iter().sum();
            assert_eq!(
                total,
                k.min(n_total),
                "lens={lens:?} k={k}: counts={counts:?} did not sum to k.min(n_total)"
            );
            for (&count, &len) in counts.iter().zip(lens) {
                assert!(
                    count <= len,
                    "lens={lens:?} k={k}: count {count} exceeds group len {len}"
                );
            }
        }
    }

    #[test]
    fn stratified_subagging_never_starves_the_rare_stratum() {
        // ~100 rows, 4 positives (stratum 1), n_bags=8, bag_subsample=0.8 — floor(4 * 0.8) = 3,
        // so every bag's draw must retain at least 3 of the 4 positives. The bagging analog of
        // `holdout_stratification_prevents_minority_class_starvation`.
        let n = 100usize;
        let strata: Vec<u32> = (0..n as u32).map(|i| u32::from(i < 4)).collect();
        let sample_len = ((n as f64) * 0.8_f64).round() as usize;
        for bag in 0..8u32 {
            let rows = stratified_subagging_rows(12345, bag, &strata, sample_len).unwrap();
            assert_eq!(rows.len(), sample_len, "bag {bag}: draw size");
            assert!(
                rows.windows(2).all(|w| w[0] < w[1]),
                "bag {bag}: rows must be strictly ascending (sorted, no dupes): {rows:?}"
            );
            let positives = rows.iter().filter(|&&r| strata[r as usize] == 1).count();
            assert!(
                positives >= 3,
                "bag {bag}: only {positives} of 4 positives retained (rows={rows:?})"
            );
        }
    }

    #[test]
    fn subagging_rows_dispatch_with_no_strata_is_byte_identical_to_subagging_rows() {
        // Gamma/SquaredError bagged fits carry `strata = None` end-to-end (`es_strata_for_loss`
        // returns `None` for both) — the dispatch's `None` arm must call the UNCHANGED original
        // sampler with the same arguments, not a re-implementation that happens to agree.
        for seed in [0u64, 1, 42, 999] {
            for round in 0u32..5 {
                for (n_rows, k) in [(0usize, 0usize), (1, 1), (10, 3), (100, 80), (7, 20)] {
                    let direct = subagging_rows(seed, round, n_rows, k).unwrap();
                    let dispatched = subagging_rows_dispatch(seed, round, n_rows, None, k).unwrap();
                    assert_eq!(
                        direct, dispatched,
                        "seed={seed} round={round} n_rows={n_rows} k={k}"
                    );
                }
            }
        }
    }

    #[test]
    fn multiclass_bagged_stratifies_on_class_label_and_stays_thread_deterministic() {
        // fit_multiclass_bagged had no prior test coverage; this exercises n_bags > 1 with
        // bag_subsample < 1.0 (each bag now stratifies on the K-way label via
        // `multiclass_labels`) and checks the soup is still thread-count independent.
        let n = 240usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 30) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
        let y: Vec<f32> = x0
            .iter()
            .map(|&v| {
                if v < 10.0 {
                    0.0
                } else if v < 20.0 {
                    1.0
                } else {
                    2.0
                }
            })
            .collect();
        let xb = binned(&[x0, x1]);
        let config = Config {
            n_trees: 20,
            learning_rate: 0.3,
            boosters: BoosterConfig {
                ensemble: EnsembleSpec::OuterBag {
                    n_bags: 3,
                    bag_subsample: 0.8,
                    cell_refit: None,
                },
                ..BoosterConfig::default()
            },
            ..Config::default()
        };
        let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let sqe = SquaredError; // ignored by fit_multiclass (softmax gradient is coupled)
        let bytes = |nt: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let model = fit_multiclass(
                    &config,
                    &xb,
                    &y,
                    3,
                    &labels,
                    &spec(&sqe),
                    &CatEncoderStore::new(),
                )
                .unwrap();
                assert_eq!(model.n_classes(), 3);
                let proba = model.predict_proba(&xb).unwrap();
                for r in 0..n {
                    let row = &proba[r * 3..r * 3 + 3];
                    let s: f32 = row.iter().sum();
                    assert!((s - 1.0).abs() < 1e-3, "row {r} proba sums to {s}, not 1");
                }
                crate::serialize::encode_multiclass(&model).unwrap()
            })
        };
        let b1 = bytes(1);
        assert_eq!(b1, bytes(2));
        assert_eq!(b1, bytes(8));
    }

    #[test]
    fn outer_bag_soup_records_contiguous_bag_spans_and_single_fit_does_not() {
        // The runtime bag partition behind per-bag bank introspection: one span per bag,
        // contiguous, jointly covering every soup tree. Single fits carry no partition,
        // and the spans never survive the wire (serde-skip).
        let n = 180usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 9 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| (if x0[i] <= 4.0 { 1.5 } else { -2.0 }) + (i % 11) as f32 * 0.03)
            .collect();
        let x = binned(&[x0]);
        let mut cfg = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 20,
            learning_rate: 0.25,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig {
                ensemble: EnsembleSpec::OuterBag {
                    n_bags: 3,
                    bag_subsample: 1.0,
                    cell_refit: None,
                },
                ..BoosterConfig::default()
            },
            fit_control: Default::default(),
        };
        let sqe = SquaredError;
        let model = Booster::with_config(cfg.clone())
            .fit(&x, &y, &spec(&sqe))
            .unwrap();
        let spans = model
            .bag_spans
            .as_ref()
            .expect("outer-bag soup records bag spans");
        assert_eq!(spans.len(), 3, "one span per bag");
        assert_eq!(spans[0].0, 0, "spans start at the first tree");
        for w in spans.windows(2) {
            assert_eq!(w[0].1, w[1].0, "spans are contiguous in bag order");
        }
        assert_eq!(
            spans.last().unwrap().1 as usize,
            model.trees.len(),
            "spans jointly cover every soup tree"
        );
        // Round-tripping the wire drops the runtime partition (serde-skip).
        let decoded =
            crate::serialize::decode_model(&crate::serialize::encode_model(&model).unwrap())
                .unwrap();
        assert!(
            decoded.bag_spans.is_none(),
            "bag spans must not survive the wire"
        );
        // A single (un-bagged) fit records no partition.
        cfg.boosters = BoosterConfig::default();
        let single = Booster::with_config(cfg).fit(&x, &y, &spec(&sqe)).unwrap();
        assert!(
            single.bag_spans.is_none(),
            "single fits carry no bag partition"
        );
        assert!(
            single.bag_in_bag.is_none(),
            "single fits carry no bag membership either"
        );
    }

    #[test]
    fn outer_bag_soup_records_the_drawn_membership_deterministically() {
        // The prune guard's zero-cost honest evidence is each bag's OUT-of-bag complement, so
        // the recorded membership must (a) exist per bag over the fit rows, (b) BE the draw
        // `subagging_rows_dispatch` made rather than an approximation of it, and (c) not
        // depend on the thread count the bags happened to run on.
        let n = 300usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 9 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| (if x0[i] <= 4.0 { 1.5 } else { -2.0 }) + (i % 11) as f32 * 0.03)
            .collect();
        let x = binned(&[x0]);
        let n_bags = 4u16;
        let bag_subsample = 0.7f32;
        let cfg = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 12,
            learning_rate: 0.25,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig {
                ensemble: EnsembleSpec::OuterBag {
                    n_bags,
                    bag_subsample,
                    cell_refit: None,
                },
                ..BoosterConfig::default()
            },
            fit_control: Default::default(),
        };
        let sqe = SquaredError;
        let fit_with = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    Booster::with_config(cfg.clone())
                        .fit(&x, &y, &spec(&sqe))
                        .unwrap()
                })
        };
        let model = fit_with(1);
        let membership = model
            .bag_in_bag
            .as_ref()
            .expect("outer-bag soup records bag membership");
        assert_eq!(membership.len(), usize::from(n_bags), "one mask per bag");
        let sample_len = ((n as f64) * f64::from(bag_subsample)).round() as usize;
        let fit_spec = spec(&sqe);
        for (bag, mask) in membership.iter().enumerate() {
            assert_eq!(mask.len(), n, "membership covers every fit row");
            // (b): the mask IS the sampler's draw. `subagging_rows_dispatch` is what
            // `fit_outer_bag` calls; SquaredError has no ES strata (`es_strata_for_loss`),
            // so the unstratified branch is the one under test here.
            let drawn = subagging_rows_dispatch(
                fit_spec.seed,
                u32::try_from(bag).unwrap(),
                n,
                None,
                sample_len,
            )
            .unwrap();
            let mut expected = vec![false; n];
            for &r in &drawn {
                expected[r as usize] = true;
            }
            assert_eq!(mask, &expected, "bag {bag} membership is the drawn sample");
            let in_bag = mask.iter().filter(|&&m| m).count();
            assert_eq!(
                in_bag, sample_len,
                "bag {bag} drew round(f*n) distinct rows"
            );
            assert!(in_bag < n, "bag {bag} must leave rows out of bag");
        }
        // (c): thread count cannot move it.
        assert_eq!(model.bag_in_bag, fit_with(4).bag_in_bag);
        // Runtime-only, exactly like the partition: never on the wire.
        let decoded =
            crate::serialize::decode_model(&crate::serialize::encode_model(&model).unwrap())
                .unwrap();
        assert!(
            decoded.bag_in_bag.is_none(),
            "bag membership must not survive the wire"
        );
    }

    #[test]
    fn cell_refit_outer_bag_is_g0_exact_and_thread_deterministic() {
        // A 2-feature target with a genuine {0,1} interaction, fit with bagging + the §G1
        // OOB cell-refit. The attached correction must keep G0 exact and be byte-identical
        // across thread counts (the OOB accumulation and the CG solve are deterministic).
        let n = 240usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 8 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| ((i / 8) % 6 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 4.0 { 1.5 } else { -2.0 };
                let b = if x1[i] <= 3.0 { 1.0 } else { -0.5 };
                let ab = if x0[i] <= 4.0 && x1[i] <= 3.0 {
                    2.0
                } else {
                    0.0
                };
                a + b + ab + (i % 13) as f32 * 0.02
            })
            .collect();
        let x = binned(&[x0, x1]);
        let cfg = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 30,
            learning_rate: 0.25,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig {
                ensemble: EnsembleSpec::OuterBag {
                    n_bags: 4,
                    bag_subsample: 0.8,
                    cell_refit: Some(CellRefit {
                        base: 50.0,
                        gamma: 2.0,
                    }),
                },
                ..BoosterConfig::default()
            },
            fit_control: Default::default(),
        };
        let sqe = SquaredError;
        let bytes = |nt: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let model = Booster::with_config(cfg.clone())
                    .fit(&x, &y, &spec(&sqe))
                    .unwrap();
                // The no-harm guard may keep, shrink, or drop the correction depending on the
                // held-out fit; either way the bag→OOB→refit→guard pipeline must be byte-
                // deterministic and the (possibly corrected) model must stay G0-exact.
                let serve = crate::data::ServeBinnedMatrix(x.clone());
                let bank = model.explain(&serve, RefMeasure::default()).unwrap();
                assert_exact_decomposition(&model, &bank, &serve).unwrap();
                crate::serialize::encode_model(&model).unwrap()
            })
        };
        let b1 = bytes(1);
        assert_eq!(
            b1,
            bytes(2),
            "cell-refit must be byte-identical across threads"
        );
        assert_eq!(b1, bytes(8));
    }

    #[test]
    fn cell_correction_oob_residual_is_offset_correct() {
        // Isolates the H5 offset bug from the surrounding boosting loop by constructing
        // attach_cell_correction's inputs directly. `preds` stands in for whatever a bag
        // happened to learn (score_trees_rows: f0 + Σ alpha·tree, no offset); `y` is defined so
        // that `preds + ln(exposure)` is the EXACT Poisson optimum at every row (the same f32
        // arithmetic attach_cell_correction itself uses). With the offset correctly folded into
        // the OOB residual basis, that residual is then exactly zero everywhere, so the
        // cell-basis correction must fit ~0 deltas no matter how strongly the (fabricated)
        // exposure varies with x0 — pre-fix, the same construction leaves a large, x0-shaped
        // `1 - e[r]` residual that the correction bank happily fits.
        let n = 96usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 8) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 6) as f32).collect();
        // A real fit purely to get non-trivial splits on both axes (and their pair), so
        // correctable_supports is non-empty; its own target is irrelevant beyond that — only
        // the resulting f0/trees are used (as a stand-in bag) below.
        let seed_y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 3.0 { 1.0 } else { -1.0 };
                let b = if x1[i] <= 2.0 { 0.6 } else { -0.4 };
                let ab = if x0[i] <= 3.0 && x1[i] <= 2.0 {
                    0.8
                } else {
                    0.0
                };
                a + b + ab
            })
            .collect();
        let e: Vec<f32> = x0.iter().map(|&v| 0.3 + 0.5 * v).collect(); // 0.3..=3.8
        let x = binned(&[x0, x1]);

        let sqe = SquaredError;
        let model = Booster::with_config(Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 12,
            learning_rate: 0.3,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig::default(),
            fit_control: Default::default(),
        })
        .fit(&x, &seed_y, &spec(&sqe))
        .unwrap();
        assert!(
            !model.trees.is_empty(),
            "need real splits so correctable_supports is non-empty"
        );

        let all_rows: Vec<u32> = (0..n as u32).collect();
        let mut preds = vec![0.0_f32; n];
        model.score_trees_rows(&x, &all_rows, &mut preds).unwrap();
        let offset: Vec<f32> = e.iter().map(|v| v.ln()).collect();
        let y: Vec<f32> = preds
            .iter()
            .zip(&offset)
            .map(|(&p, &o)| (p + o).exp())
            .collect();

        let poisson = Poisson;
        let mut fit_spec = spec(&poisson);
        fit_spec.exposure = Some(&e);
        let oobs = vec![BagOob {
            in_bag: vec![false; n], // every row OOB ⇒ oob_raw == preds exactly (pre-offset)
            preds: preds.clone(),
        }];
        let cr = CellRefit {
            base: 50.0,
            gamma: 2.0,
        };
        let corrected = attach_cell_correction(model, &x, &y, &fit_spec, &oobs, cr).unwrap();
        match &corrected.correction {
            None => {} // the no-harm guard (or the CG solve itself) dropped it — also correct
            Some(bank) => {
                let max_delta = bank
                    .tables
                    .iter()
                    .flat_map(|t| t.values.iter())
                    .fold(0.0_f64, |m, &v| m.max(v.abs()));
                assert!(
                    max_delta < 1e-6,
                    "offset-correct OOB residual is exactly zero by construction; the \
                     correction must not fit a phantom exposure pattern, got max|delta|={max_delta}"
                );
            }
        }
    }

    #[test]
    fn outer_bag_single_member_is_byte_identical_to_inert_fit() {
        let n = 96usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| (if x0[i] <= 3.0 { 2.0 } else { -1.0 }) + x1[i] * 0.2)
            .collect();
        let x = binned(&[x0, x1]);
        let sqe = SquaredError;
        let base_cfg = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 20,
            learning_rate: 0.2,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig::default(),
            fit_control: Default::default(),
        };
        let mut bag_cfg = base_cfg.clone();
        bag_cfg.boosters.ensemble = EnsembleSpec::OuterBag {
            n_bags: 1,
            bag_subsample: 1.0,
            cell_refit: None,
        };
        let base = Booster::with_config(base_cfg)
            .fit(&x, &y, &spec(&sqe))
            .unwrap();
        let bag = Booster::with_config(bag_cfg)
            .fit(&x, &y, &spec(&sqe))
            .unwrap();
        assert_eq!(
            crate::serialize::encode_model(&base).unwrap(),
            crate::serialize::encode_model(&bag).unwrap()
        );
    }

    #[test]
    fn greedy_select_uses_deviance_and_stays_exact_deterministic() {
        let n = 150usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 10 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 5.0 { 0.8 } else { 3.2 };
                let b = if x1[i] <= 3.0 { 0.4 } else { 1.1 };
                let c = if x2[i] <= 2.0 { 0.2 } else { 0.7 };
                a + b + c
            })
            .collect();
        let x = binned(&[x0, x1, x2]);
        let cfg = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 12,
            learning_rate: 0.2,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig {
                ensemble: EnsembleSpec::GreedySelect {
                    library_size: 4,
                    hp_grid: HpGrid {
                        max_bins: vec![32],
                        lambdas: vec![0.0, 1.0],
                        learning_rates: vec![0.15, 0.25],
                        n_trees: vec![6],
                        max_interaction_orders: vec![2],
                        random_strengths: vec![0.0],
                    },
                    selection_bags: 3,
                    seed_top_n: 2,
                },
                ..BoosterConfig::default()
            },
            fit_control: Default::default(),
        };
        let poisson = Poisson;
        let bytes = |nt: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let model = Booster::with_config(cfg.clone())
                    .fit(&x, &y, &spec(&poisson))
                    .unwrap();
                assert_eq!(model.schema.objective.loss, LossId::Poisson);
                assert!(!model.trees.is_empty());
                let serve = crate::data::ServeBinnedMatrix(x.clone());
                let bank = model.explain(&serve, RefMeasure::default()).unwrap();
                assert_exact_decomposition(&model, &bank, &serve).unwrap();
                let pred = model.predict_binned(&x, None).unwrap();
                assert!(pred.iter().all(|v| v.is_finite() && *v > 0.0));
                crate::serialize::encode_model(&model).unwrap()
            })
        };
        let b1 = bytes(1);
        assert_eq!(b1, bytes(2));
        assert_eq!(b1, bytes(8));
    }

    #[test]
    fn greedy_select_holdout_deviance_includes_the_exposure_offset() {
        // Deterministic plumbing check: replays fit_greedy_select's own algorithm by hand,
        // calling the SAME private helpers (holdout_split / row_subset / fit_single /
        // greedy_selection_weights / soup_models) with the holdout offset added explicitly, and
        // asserts the real entrypoint agrees byte-for-byte. This pins that fit_greedy_select's
        // holdout evaluation is offset-inclusive without hard-coding which library member
        // "should" win — pre-fix, `actual` (computed without the offset) diverges from this
        // offset-correct `expected` replay because the offset materially shifts every holdout
        // row's raw score (exposure spans roughly a 20x range here, correlated with x0).
        let n = 160usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 8) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let raw_true: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 3.0 { 0.7 } else { -0.5 };
                let b = if x1[i] <= 2.0 { 0.3 } else { -0.2 };
                a + b
            })
            .collect();
        // Exposure is a strong, deterministic function of x0 (the routine pricing case the
        // finding calls out: exposure correlated with a rating feature), spanning ~0.2..=4.4.
        let e: Vec<f32> = x0.iter().map(|&v| 0.2 + 0.6 * v).collect();
        let y: Vec<f32> = raw_true
            .iter()
            .zip(&e)
            .map(|(&f, &ei)| (f + ei.ln()).exp())
            .collect();
        let x = binned(&[x0, x1]);

        let poisson = Poisson;
        let mut fit_spec = spec(&poisson);
        fit_spec.exposure = Some(&e);
        fit_spec.seed = 7;

        let hp_grid = HpGrid {
            max_bins: vec![32],
            lambdas: vec![0.0, 1.0],
            learning_rates: vec![0.2],
            n_trees: vec![5],
            max_interaction_orders: vec![1],
            random_strengths: vec![0.0],
        };
        let library_size: u16 = 2;
        let selection_bags: u16 = 1;
        let seed_top_n: u8 = 2;
        let config = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 5,
            learning_rate: 0.2,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig::default(),
            fit_control: Default::default(),
        };
        let cat_encoders = CatEncoderStore::new();
        let params = GreedyParams {
            library_size,
            hp_grid: &hp_grid,
            selection_bags,
            seed_top_n,
        };
        let actual = fit_greedy_select(&config, &x, &y, &fit_spec, params, &cat_encoders).unwrap();

        // --- independent replay: same private helpers, offset added explicitly ---
        let base_config = ensemble_base_config(&config);
        let (train_rows, holdout_rows) = holdout_split(fit_spec.seed, x.n_rows as usize).unwrap();
        let train = row_subset(&x, &y, fit_spec.weight, fit_spec.exposure, &train_rows).unwrap();
        let holdout =
            row_subset(&x, &y, fit_spec.weight, fit_spec.exposure, &holdout_rows).unwrap();
        let holdout_weight = effective_weight(&holdout);
        let holdout_offset =
            compute_offset(holdout.exposure.as_deref().unwrap(), holdout.y.len()).unwrap();

        let mut library = Vec::new();
        for ordinal in 0..library_size {
            let choice = hp_choice_at(&hp_grid, usize::from(ordinal)).unwrap();
            let mut member_config = base_config.clone();
            member_config.lambda = choice.lambda;
            member_config.learning_rate = choice.learning_rate;
            member_config.n_trees = choice.n_trees;
            member_config.boosters.random_strength = choice.random_strength;
            let mut interaction = fit_spec.interaction.clone();
            interaction.max_order = choice.max_order;
            let hp_block = u32::from(choice.max_bin) + u32::from(ordinal);
            let member_seed = pb_seed(
                fit_spec.seed,
                u32::from(ordinal),
                Stage::Sample as u32,
                hp_block,
            );
            let member_spec = FitSpec {
                loss: fit_spec.loss,
                weight: train.weight.as_deref(),
                exposure: train.exposure.as_deref(),
                monotone: fit_spec.monotone.clone(),
                interaction,
                credibility: fit_spec.credibility,
                fixed_holdout: None,
                bag_groups: None,
                seed: member_seed,
            };
            let model = fit_single(
                &member_config,
                &train.x,
                &train.y,
                &member_spec,
                &cat_encoders,
            )
            .unwrap();
            let mut holdout_raw = raw_predictions(&model, &holdout.x).unwrap();
            for (r, o) in holdout_raw.iter_mut().zip(&holdout_offset) {
                *r += o;
            }
            let deviance = f64::from(
                fit_spec
                    .loss
                    .deviance(&holdout.y, &holdout_raw, &holdout_weight)
                    .unwrap(),
            );
            library.push(LibraryMember {
                model,
                holdout_raw,
                deviance,
            });
        }
        let weights = greedy_selection_weights(
            &library,
            &holdout.y,
            &holdout_weight,
            fit_spec.loss,
            fit_spec.seed,
            usize::from(selection_bags),
            usize::from(seed_top_n),
        )
        .unwrap();
        let mut members = Vec::new();
        for (alpha, member) in weights.into_iter().zip(library) {
            if alpha > 0.0 {
                members.push(WeightedModel {
                    alpha,
                    model: member.model,
                });
            }
        }
        let expected = soup_models(&members).unwrap();

        assert_eq!(
            crate::serialize::encode_model(&actual).unwrap(),
            crate::serialize::encode_model(&expected).unwrap(),
            "fit_greedy_select's holdout evaluation must match an independently offset-corrected replay"
        );
    }

    #[test]
    fn greedy_select_holdout_matches_the_fixed_holdout_mask() {
        // Deterministic plumbing check, mirroring
        // greedy_select_holdout_deviance_includes_the_exposure_offset: replays fit_greedy_select's
        // own algorithm by hand, calling the SAME private helpers (holdout_rows_from_mask /
        // row_subset / fit_single / greedy_selection_weights / soup_models) with the caller's
        // fixed_holdout mask threaded through exactly as the fix does, and asserts the real
        // entrypoint agrees byte-for-byte. Pins that a caller-supplied mask becomes the
        // GreedySelect selection holdout (not holdout_split's own random 80/20 carve) and is
        // passed straight through as every member's own fixed_holdout — pre-fix, `actual`
        // (computed from holdout_split's unrelated random carve, with `fixed_holdout: None`
        // hardcoded for members) diverges from this mask-correct `expected` replay whenever the
        // mask doesn't happen to coincide with that random carve, which is effectively always.
        let n = 160usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 8) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 3.0 { 0.7 } else { -0.5 };
                let b = if x1[i] <= 2.0 { 0.3 } else { -0.2 };
                a + b + (i % 11) as f32 * 0.03
            })
            .collect();
        let x = binned(&[x0, x1]);
        let mask: Vec<bool> = (0..n).map(|i| i % 5 == 0).collect(); // 32 holdout rows

        let sqe = SquaredError;
        let mut fit_spec = spec(&sqe);
        fit_spec.fixed_holdout = Some(&mask);
        fit_spec.seed = 11;

        let hp_grid = HpGrid {
            max_bins: vec![32],
            lambdas: vec![0.0, 1.0],
            learning_rates: vec![0.2],
            n_trees: vec![5],
            max_interaction_orders: vec![1],
            random_strengths: vec![0.0],
        };
        let library_size: u16 = 2;
        let selection_bags: u16 = 1;
        let seed_top_n: u8 = 2;
        let config = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 5,
            learning_rate: 0.2,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig::default(),
            fit_control: Default::default(),
        };
        let cat_encoders = CatEncoderStore::new();
        let params = GreedyParams {
            library_size,
            hp_grid: &hp_grid,
            selection_bags,
            seed_top_n,
        };
        let actual = fit_greedy_select(&config, &x, &y, &fit_spec, params, &cat_encoders).unwrap();

        // --- independent replay: same private helpers, the mask threaded through by hand ---
        let (_, holdout_rows) = holdout_rows_from_mask(x.n_rows, &mask).unwrap();
        let holdout =
            row_subset(&x, &y, fit_spec.weight, fit_spec.exposure, &holdout_rows).unwrap();
        let holdout_weight = effective_weight(&holdout);

        let mut library = Vec::new();
        for ordinal in 0..library_size {
            let choice = hp_choice_at(&hp_grid, usize::from(ordinal)).unwrap();
            let mut member_config = config.clone();
            member_config.lambda = choice.lambda;
            member_config.learning_rate = choice.learning_rate;
            member_config.n_trees = choice.n_trees;
            member_config.boosters.random_strength = choice.random_strength;
            let mut interaction = fit_spec.interaction.clone();
            interaction.max_order = choice.max_order;
            let hp_block = u32::from(choice.max_bin) + u32::from(ordinal);
            let member_seed = pb_seed(
                fit_spec.seed,
                u32::from(ordinal),
                Stage::Sample as u32,
                hp_block,
            );
            let member_spec = FitSpec {
                loss: fit_spec.loss,
                weight: fit_spec.weight,
                exposure: fit_spec.exposure,
                monotone: fit_spec.monotone.clone(),
                interaction,
                credibility: fit_spec.credibility,
                fixed_holdout: Some(&mask),
                bag_groups: None,
                seed: member_seed,
            };
            let model = fit_single(&member_config, &x, &y, &member_spec, &cat_encoders).unwrap();
            let holdout_raw = raw_predictions(&model, &holdout.x).unwrap();
            let deviance = f64::from(
                fit_spec
                    .loss
                    .deviance(&holdout.y, &holdout_raw, &holdout_weight)
                    .unwrap(),
            );
            library.push(LibraryMember {
                model,
                holdout_raw,
                deviance,
            });
        }
        let weights = greedy_selection_weights(
            &library,
            &holdout.y,
            &holdout_weight,
            fit_spec.loss,
            fit_spec.seed,
            usize::from(selection_bags),
            usize::from(seed_top_n),
        )
        .unwrap();
        let mut members = Vec::new();
        for (alpha, member) in weights.into_iter().zip(library) {
            if alpha > 0.0 {
                members.push(WeightedModel {
                    alpha,
                    model: member.model,
                });
            }
        }
        let expected = soup_models(&members).unwrap();

        assert_eq!(
            crate::serialize::encode_model(&actual).unwrap(),
            crate::serialize::encode_model(&expected).unwrap(),
            "fit_greedy_select must use fixed_holdout as its selection holdout and thread it \
             through to every member, matching an independent mask-correct replay"
        );
    }

    #[test]
    fn greedy_select_fixed_holdout_and_exposure_offset_compose_without_double_adding() {
        // Combines the two independent GreedySelect fixes (offset in holdout evaluation,
        // fixed_holdout as the selection holdout) on the SAME fit: exposure is present AND a
        // fixed_holdout mask is present. The replay independently reconstructs BOTH the mask
        // threading and the "add the holdout offset exactly once" step from first principles
        // (not by calling a shared helper the production code might also mis-call), so if either
        // fix double-counted the offset when the other is also active, or dropped it, this
        // diverges from the real entrypoint's output.
        let n = 160usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 8) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let raw_true: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 3.0 { 0.7 } else { -0.5 };
                let b = if x1[i] <= 2.0 { 0.3 } else { -0.2 };
                a + b
            })
            .collect();
        let e: Vec<f32> = x0.iter().map(|&v| 0.2 + 0.6 * v).collect(); // 0.2..=4.4
        let y: Vec<f32> = raw_true
            .iter()
            .zip(&e)
            .map(|(&f, &ei)| (f + ei.ln()).exp())
            .collect();
        let x = binned(&[x0, x1]);
        let mask: Vec<bool> = (0..n).map(|i| i % 5 == 0).collect();

        let poisson = Poisson;
        let mut fit_spec = spec(&poisson);
        fit_spec.exposure = Some(&e);
        fit_spec.fixed_holdout = Some(&mask);
        fit_spec.seed = 5;

        let hp_grid = HpGrid {
            max_bins: vec![32],
            lambdas: vec![0.0, 1.0],
            learning_rates: vec![0.2],
            n_trees: vec![5],
            max_interaction_orders: vec![1],
            random_strengths: vec![0.0],
        };
        let library_size: u16 = 2;
        let selection_bags: u16 = 1;
        let seed_top_n: u8 = 2;
        let config = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 5,
            learning_rate: 0.2,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig::default(),
            fit_control: Default::default(),
        };
        let cat_encoders = CatEncoderStore::new();
        let params = GreedyParams {
            library_size,
            hp_grid: &hp_grid,
            selection_bags,
            seed_top_n,
        };
        let actual = fit_greedy_select(&config, &x, &y, &fit_spec, params, &cat_encoders).unwrap();

        // --- independent replay: mask threading + offset addition, each done exactly once ---
        let (_, holdout_rows) = holdout_rows_from_mask(x.n_rows, &mask).unwrap();
        let holdout =
            row_subset(&x, &y, fit_spec.weight, fit_spec.exposure, &holdout_rows).unwrap();
        let holdout_weight = effective_weight(&holdout);
        let holdout_offset =
            compute_offset(holdout.exposure.as_deref().unwrap(), holdout.y.len()).unwrap();

        let mut library = Vec::new();
        for ordinal in 0..library_size {
            let choice = hp_choice_at(&hp_grid, usize::from(ordinal)).unwrap();
            let mut member_config = config.clone();
            member_config.lambda = choice.lambda;
            member_config.learning_rate = choice.learning_rate;
            member_config.n_trees = choice.n_trees;
            member_config.boosters.random_strength = choice.random_strength;
            let mut interaction = fit_spec.interaction.clone();
            interaction.max_order = choice.max_order;
            let hp_block = u32::from(choice.max_bin) + u32::from(ordinal);
            let member_seed = pb_seed(
                fit_spec.seed,
                u32::from(ordinal),
                Stage::Sample as u32,
                hp_block,
            );
            let member_spec = FitSpec {
                loss: fit_spec.loss,
                weight: fit_spec.weight,
                exposure: fit_spec.exposure,
                monotone: fit_spec.monotone.clone(),
                interaction,
                credibility: fit_spec.credibility,
                fixed_holdout: Some(&mask),
                bag_groups: None,
                seed: member_seed,
            };
            let model = fit_single(&member_config, &x, &y, &member_spec, &cat_encoders).unwrap();
            let mut holdout_raw = raw_predictions(&model, &holdout.x).unwrap();
            for (r, o) in holdout_raw.iter_mut().zip(&holdout_offset) {
                *r += o;
            }
            let deviance = f64::from(
                fit_spec
                    .loss
                    .deviance(&holdout.y, &holdout_raw, &holdout_weight)
                    .unwrap(),
            );
            library.push(LibraryMember {
                model,
                holdout_raw,
                deviance,
            });
        }
        let weights = greedy_selection_weights(
            &library,
            &holdout.y,
            &holdout_weight,
            fit_spec.loss,
            fit_spec.seed,
            usize::from(selection_bags),
            usize::from(seed_top_n),
        )
        .unwrap();
        let mut members = Vec::new();
        for (alpha, member) in weights.into_iter().zip(library) {
            if alpha > 0.0 {
                members.push(WeightedModel {
                    alpha,
                    model: member.model,
                });
            }
        }
        let expected = soup_models(&members).unwrap();

        assert_eq!(
            crate::serialize::encode_model(&actual).unwrap(),
            crate::serialize::encode_model(&expected).unwrap(),
            "fixed_holdout and the exposure-offset fix must compose without double-adding or \
             dropping the offset"
        );
    }

    #[test]
    fn holdout_stratification_prevents_minority_class_starvation() {
        // ES-stratify fix: a 90/10 imbalanced label set (n=100, 10 minority rows) with a 10%
        // validation_fraction. The UNSTRATIFIED carve (holdout_mask's pre-existing global-ranking
        // path) can, for some seeds, place ZERO of the 10 minority rows in the holdout — a real
        // hypergeometric-tail event at this ratio, not a contrived one, and exactly the failure
        // the finding describes (best_iteration decided on a holdout with no positives). The
        // STRATIFIED carve (same data, same seeds) must never do that.
        let n: u32 = 100;
        let labels: Vec<u32> = (0..n).map(|i| u32::from(i < 10)).collect(); // 10 minority (label 1)
        let frac = 0.1_f32;

        let starved_seed = (0..200u64).find(|&seed| {
            let mask = holdout_mask(n, Some(frac), seed, None).unwrap().unwrap();
            (0..n)
                .filter(|&r| labels[r as usize] == 1 && mask[r as usize])
                .count()
                == 0
        });
        let seed = starved_seed.expect(
            "expected at least one seed in 0..200 where the unstratified carve excludes the \
             minority class entirely — if this fails, the fixture no longer reproduces the \
             finding's failure mode and needs re-tuning, not deletion",
        );

        for seed in std::iter::once(seed).chain(0..30u64) {
            let stratified = holdout_mask(n, Some(frac), seed, Some(&labels))
                .unwrap()
                .unwrap();
            let minority_in_holdout = (0..n)
                .filter(|&r| labels[r as usize] == 1 && stratified[r as usize])
                .count();
            let majority_in_holdout = (0..n)
                .filter(|&r| labels[r as usize] == 0 && stratified[r as usize])
                .count();
            assert!(
                minority_in_holdout > 0,
                "seed {seed}: stratified holdout excluded the minority class entirely"
            );
            assert!(
                majority_in_holdout > 0,
                "seed {seed}: stratified holdout excluded the majority class entirely"
            );
        }
    }

    #[test]
    fn unstratified_carve_is_untouched_by_the_stratify_fix() {
        // Non-Logistic / regressor carve sanity: strata=None (the pre-fix code, moved verbatim
        // into holdout_mask's early-return arm — byte-identity to the old wrapper was pinned
        // during the transition before the wrapper was removed) still produces a deterministic,
        // exhaustive train/holdout partition at the requested fraction.
        for seed in 0..10u64 {
            let (train, holdout) =
                carve_validation_rows_stratified(137, Some(0.2), seed, None).unwrap();
            let holdout = holdout.expect("fraction requested => carve exists");
            assert_eq!(train.len() + holdout.len(), 137, "seed {seed}");
            assert!(
                !holdout.is_empty() && holdout.len() < 137 / 2,
                "seed {seed}"
            );
            let again = carve_validation_rows_stratified(137, Some(0.2), seed, None).unwrap();
            assert_eq!((train, Some(holdout)), again, "seed {seed}");
        }

        // End-to-end: a SquaredError fit with validation_fraction stays deterministic (strata is
        // None for any non-Logistic loss, so fit_single's new carve call degrades to the old one).
        let (x0, x1, y) = additive_2feat(120);
        let x = binned(&[x0, x1]);
        let cfg = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 10,
            learning_rate: 0.2,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: Some(0.2),
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig::default(),
            fit_control: Default::default(),
        };
        let sqe = SquaredError;
        let a = Booster::with_config(cfg.clone())
            .fit(&x, &y, &spec(&sqe))
            .unwrap();
        let b = Booster::with_config(cfg).fit(&x, &y, &spec(&sqe)).unwrap();
        assert_eq!(
            crate::serialize::encode_model(&a).unwrap(),
            crate::serialize::encode_model(&b).unwrap()
        );
    }

    #[test]
    fn f0_intercept_is_fit_on_train_rows_only_under_a_validation_carve() {
        // Intercept honesty: the holdout's target level must not leak into f0. The mask holds
        // out rows whose y is wildly different (1000.0 vs ~0..14), so a full-array mean would
        // shift f0 by hundreds — far beyond f32 noise — while the honest train-only mean stays
        // at the train level. Without a carve, f0 must equal the full-array mean exactly
        // (byte-identical legacy path).
        let n = 120usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 8) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let mask: Vec<bool> = (0..n).map(|i| i >= 96).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                if mask[i] {
                    1000.0
                } else {
                    (x0[i] + x1[i]).max(0.0)
                }
            })
            .collect();
        let x = binned(&[x0, x1]);
        let sqe = SquaredError;
        let ones = vec![1.0_f32; n];

        let mut holdout_spec = spec(&sqe);
        holdout_spec.fixed_holdout = Some(&mask);
        let model = Booster::with_config(Config {
            n_trees: 5,
            ..Config::default()
        })
        .fit(&x, &y, &holdout_spec)
        .unwrap();

        let train_y: Vec<f32> = y[..96].to_vec();
        let expected_train_f0 = sqe.init_score(&train_y, &ones[..96], None).unwrap() as f32;
        let leaky_full_f0 = sqe.init_score(&y, &ones, None).unwrap() as f32;
        assert_eq!(model.f0, expected_train_f0);
        assert!(
            (model.f0 - leaky_full_f0).abs() > 100.0,
            "fixture must make the leak visible: train f0 {} vs full-array f0 {leaky_full_f0}",
            model.f0
        );

        // No carve => the legacy full-slice call, bit-for-bit.
        let no_holdout = Booster::with_config(Config {
            n_trees: 5,
            ..Config::default()
        })
        .fit(&x, &y, &spec(&sqe))
        .unwrap();
        assert_eq!(no_holdout.f0.to_bits(), leaky_full_f0.to_bits());
    }

    #[test]
    fn multiclass_f0_priors_are_fit_on_train_rows_only_under_a_validation_carve() {
        // Same honesty contract for the per-class priors: the holdout deliberately over-samples
        // class 2 (holds out half its rows), so full-array priors differ measurably from the
        // honest train-only priors that must seed the raw columns.
        let n = 120usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 8) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let labels: Vec<u32> = (0..n).map(|i| (i % 3) as u32).collect();
        let y: Vec<f32> = labels.iter().map(|&l| l as f32).collect();
        // Hold out every second class-2 row (20 of its 40 rows), nothing else.
        let mask: Vec<bool> = (0..n).map(|i| labels[i] == 2 && (i / 3) % 2 == 0).collect();
        let x = binned(&[x0, x1]);
        let ones = vec![1.0_f32; n];
        let class_labels: Vec<String> = ["a", "b", "c"].iter().map(|s| (*s).to_owned()).collect();

        let mut holdout_spec = spec(&Logistic);
        holdout_spec.fixed_holdout = Some(&mask);
        let mc = fit_multiclass(
            &Config {
                n_trees: 5,
                ..Config::default()
            },
            &x,
            &y,
            3,
            &class_labels,
            &holdout_spec,
            &CatEncoderStore::default(),
        )
        .unwrap();

        let (train_labels, train_w): (Vec<u32>, Vec<f32>) = (0..n)
            .filter(|&i| !mask[i])
            .map(|i| (labels[i], ones[i]))
            .unzip();
        let expected = multiclass_init(&train_labels, &train_w, 3).unwrap();
        let leaky = multiclass_init(&labels, &ones, 3).unwrap();
        for k in 0..3 {
            assert_eq!(mc.classes[k].f0.to_bits(), (expected[k] as f32).to_bits());
        }
        assert!(
            (expected[2] - leaky[2]).abs() > 1e-3,
            "fixture must make the class-2 prior leak visible: train {} vs full {}",
            expected[2],
            leaky[2]
        );
    }

    #[test]
    fn logistic_fit_with_imbalanced_validation_holdout_stays_finite_and_exact() {
        // Wiring smoke test: fit_single derives `strata` from `y >= 0.5` for LossId::Logistic
        // ONLY and threads it into carve_validation_rows_stratified — exercised here through the
        // full Booster::fit path (not just holdout_mask in isolation) on an imbalanced label set,
        // the exact shape the finding's failure scenario describes.
        let n = 200usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 8) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        // ~10% positive, correlated with x so there is real signal to early-stop on.
        let y: Vec<f32> = (0..n)
            .map(|i| if x0[i] <= 1.0 && i % 3 == 0 { 1.0 } else { 0.0 })
            .collect();
        assert!(
            y.iter().filter(|&&v| v > 0.0).count() >= 2,
            "fixture needs at least a couple of positives to be a meaningful imbalance test"
        );
        let x = binned(&[x0, x1]);
        let logistic = Logistic;
        let model = Booster::with_config(Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 15,
            learning_rate: 0.2,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: Some(0.2),
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig::default(),
            fit_control: Default::default(),
        })
        .fit(&x, &y, &spec(&logistic))
        .unwrap();
        assert_eq!(model.mode, ExactnessMode::Exact);
        let preds = model.predict_binned(&x, None).unwrap();
        assert!(preds
            .iter()
            .all(|p| p.is_finite() && (0.0..=1.0).contains(p)));
    }

    #[test]
    fn es_strata_for_loss_matches_the_documented_rule_per_objective() {
        // Pure-function pin for the ES-runaway fix #2 (2026-07-20): Logistic stratifies on
        // y >= 0.5, Poisson/Tweedie on y > 0.0, and Gamma/SquaredError/Softmax stay
        // unstratified (None) — the exact table in es_strata_for_loss's doc.
        let y = vec![0.0, 0.3, 0.6, 1.0, 2.5];
        assert_eq!(
            es_strata_for_loss(LossId::Logistic, &y),
            Some(vec![0, 0, 1, 1, 1])
        );
        assert_eq!(
            es_strata_for_loss(LossId::Poisson, &y),
            Some(vec![0, 1, 1, 1, 1])
        );
        assert_eq!(
            es_strata_for_loss(LossId::Tweedie, &y),
            Some(vec![0, 1, 1, 1, 1])
        );
        assert_eq!(es_strata_for_loss(LossId::Gamma, &y), None);
        assert_eq!(es_strata_for_loss(LossId::SquaredError, &y), None);
        assert_eq!(es_strata_for_loss(LossId::Softmax, &y), None);
    }

    #[test]
    fn poisson_fit_with_zero_dominated_validation_holdout_stays_finite_and_exact() {
        // Wiring smoke test (ES-runaway fix #2, 2026-07-20): fit_single derives `strata` from
        // y > 0.0 for Poisson (mirroring the pre-existing Logistic wiring test above) on a
        // zero-dominated fixture — the exact shape that starves an unstratified thin holdout of
        // positive-event rows and can leave early stopping with no genuine signal to stop on.
        let n = 300usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 8) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        // ~7% positive counts, correlated with x so there is real signal to early-stop on.
        let y: Vec<f32> = (0..n)
            .map(|i| if x0[i] <= 1.0 && i % 3 == 0 { 2.0 } else { 0.0 })
            .collect();
        assert!(
            y.iter().filter(|&&v| v > 0.0).count() >= 2,
            "fixture needs at least a couple of positives to be a meaningful zero-dominance test"
        );
        let x = binned(&[x0, x1]);
        let poisson = Poisson;
        let model = Booster::with_config(Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 15,
            learning_rate: 0.2,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: Some(0.1),
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig::default(),
            fit_control: Default::default(),
        })
        .fit(&x, &y, &spec(&poisson))
        .unwrap();
        assert_eq!(model.mode, ExactnessMode::Exact);
        let preds = model.predict_binned(&x, None).unwrap();
        assert!(preds.iter().all(|p| p.is_finite() && *p >= 0.0));
    }

    #[test]
    fn greedy_select_requires_a_holdout_row() {
        let x = binned(&[vec![1.0]]);
        let y = vec![1.0];
        let cfg = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 2,
            learning_rate: 0.2,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig {
                ensemble: EnsembleSpec::GreedySelect {
                    library_size: 1,
                    hp_grid: HpGrid {
                        max_bins: vec![32],
                        lambdas: vec![1.0],
                        learning_rates: vec![0.2],
                        n_trees: vec![2],
                        max_interaction_orders: vec![1],
                        random_strengths: vec![0.0],
                    },
                    selection_bags: 1,
                    seed_top_n: 1,
                },
                ..BoosterConfig::default()
            },
            fit_control: Default::default(),
        };
        let sqe = SquaredError;
        assert!(matches!(
            Booster::with_config(cfg).fit(&x, &y, &spec(&sqe)),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn realized_split_borders_are_a_subset_of_the_grid() {
        // Every split's bin_le indexes a real interior border of the persisted grid
        // (§03.5): bin_le ∈ 1..=borders.len(). This is the I2 precondition.
        let n = 100usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 8) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 6) as f32 * 2.0).collect();
        let y: Vec<f32> = (0..n).map(|i| (i as f32 % 9.0) - 4.0).collect();
        let x = binned(&[x0, x1]);
        let booster = Booster::with_config(Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 50,
            learning_rate: 0.3,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        });
        let sqe = SquaredError;
        let model = booster.fit(&x, &y, &spec(&sqe)).unwrap();
        for (_, tree) in &model.trees {
            for split in &tree.splits {
                let grid = &model.grids[split.axis as usize];
                assert!(
                    split.bin_le >= 1 && usize::from(split.bin_le) <= grid.borders.len(),
                    "bin_le {} outside 1..={} for axis {}",
                    split.bin_le,
                    grid.borders.len(),
                    split.axis
                );
            }
        }
    }

    #[test]
    fn degenerate_inputs_give_valid_finite_exact_models() {
        let sqe = SquaredError;
        let booster = Booster::with_config(Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 10,
            learning_rate: 0.5,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        });

        // (1) Constant target ⇒ no split ⇒ 0 trees ⇒ every prediction == f0 == mean.
        let x = binned(&[vec![1.0, 2.0, 3.0, 4.0]]);
        let model = booster.fit(&x, &[5.0, 5.0, 5.0, 5.0], &spec(&sqe)).unwrap();
        assert!(model.trees.is_empty());
        assert_eq!(model.mode, ExactnessMode::Exact);
        for i in 0..4 {
            assert!((predict(&model, &x, i) - 5.0).abs() < 1e-6);
        }

        // (2) Single row.
        let x1 = binned(&[vec![1.0]]);
        let m1 = booster.fit(&x1, &[7.0], &spec(&sqe)).unwrap();
        assert!((predict(&m1, &x1, 0) - 7.0).abs() < 1e-6);

        // (3) An all-missing (all-NaN) column alongside an informative one: the
        // missing axis is never split (degenerate grid), the model stays finite/Exact.
        let informative: Vec<f32> = (0..20).map(|i| (i % 4) as f32).collect();
        let all_missing: Vec<f32> = vec![f32::NAN; 20];
        let yv: Vec<f32> = (0..20)
            .map(|i| if i % 4 < 2 { -1.0 } else { 2.0 })
            .collect();
        let x3 = binned(&[informative, all_missing]);
        let m3 = booster.fit(&x3, &yv, &spec(&sqe)).unwrap();
        assert_eq!(m3.mode, ExactnessMode::Exact);
        for i in 0..20 {
            assert!(predict(&m3, &x3, i).is_finite());
        }
        // No split ever lands on the all-missing axis (axis 1).
        for (_, tree) in &m3.trees {
            assert!(tree.splits.iter().all(|s| s.axis != 1));
        }
    }

    #[test]
    fn bad_config_and_shape_errors() {
        let sqe = SquaredError;
        let x = binned(&[vec![1.0, 2.0]]);
        // n_trees = 0.
        let bad = Booster::with_config(Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 0,
            learning_rate: 0.1,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        });
        assert!(matches!(
            bad.fit(&x, &[1.0, 2.0], &spec(&sqe)),
            Err(PbError::InvalidConfig { .. })
        ));
        let bad = Booster::with_config(Config {
            boosters: BoosterConfig {
                random_strength: -1.0,
                ..BoosterConfig::default()
            },
            ..Config::default()
        });
        assert!(matches!(
            bad.fit(&x, &[1.0, 2.0], &spec(&sqe)),
            Err(PbError::InvalidConfig { .. })
        ));
        // y length mismatch.
        let ok = Booster::new();
        assert!(matches!(
            ok.fit(&x, &[1.0], &spec(&sqe)),
            Err(PbError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn unsupported_future_fit_spec_knobs_are_rejected() {
        let sqe = SquaredError;
        let x = binned(&[vec![1.0, 2.0]]);
        let y = [1.0_f32, 2.0];
        let booster = Booster::new();

        let mut s = spec(&sqe);
        s.interaction.max_order = 0;
        assert!(matches!(
            booster.fit(&x, &y, &s),
            Err(PbError::InvalidConfig { .. })
        ));

        // `max_order` above the spec's `max_depth` (3) is unreachable by construction and
        // must be refused with a diagnostic, not silently clamped. Written against the
        // constant rather than a literal `4`: at the high-order lift `max_order = 4` is a
        // perfectly legal ORDER, and what makes this spec invalid is that its depth cannot
        // express it. `MAX_ORDER + 1` additionally exceeds the order cap itself, so this
        // case stays invalid for at least one reason at every future cap.
        let mut s = spec(&sqe);
        s.interaction.max_order = crate::engine::MAX_ORDER as u8 + 1;
        assert!(matches!(
            booster.fit(&x, &y, &s),
            Err(PbError::InvalidConfig { .. })
        ));

        // ...and the depth relation alone is enough: a legal order that the spec's depth
        // cannot reach is refused too.
        let mut s = spec(&sqe);
        s.interaction.max_order = crate::engine::LEGACY_MAX_ORDER as u8 + 1;
        assert!(usize::from(s.interaction.max_depth) < usize::from(s.interaction.max_order));
        assert!(matches!(
            booster.fit(&x, &y, &s),
            Err(PbError::InvalidConfig { .. })
        ));

        let mut s = spec(&sqe);
        s.interaction.groups = Some(Vec::new());
        assert!(matches!(
            booster.fit(&x, &y, &s),
            Err(PbError::InvalidConfig { .. })
        ));

        let mut s = spec(&sqe);
        s.interaction.max_order = 1;
        s.interaction.groups = Some(vec![FeatureSet::new(&[0, 1])]);
        assert!(matches!(
            booster.fit(&x, &y, &s),
            Err(PbError::InvalidConfig { .. })
        ));

        let mut s = spec(&sqe);
        s.interaction.groups = Some(vec![FeatureSet::new(&[0])]);
        assert!(booster.fit(&x, &y, &s).is_ok());

        let mut s = spec(&sqe);
        s.interaction.table_budget_beta = f32::NAN;
        assert!(matches!(
            booster.fit(&x, &y, &s),
            Err(PbError::InvalidConfig { .. })
        ));

        let mut s = spec(&sqe);
        s.interaction.table_budget_cells = 0;
        assert!(matches!(
            booster.fit(&x, &y, &s),
            Err(PbError::InvalidConfig { .. })
        ));

        let mut s = spec(&sqe);
        s.monotone.insert("f0".into(), MonoSign::Increasing);
        assert!(booster.fit(&x, &y, &s).is_ok());

        let mut s = spec(&sqe);
        s.monotone
            .insert("unknown_feature".into(), MonoSign::Increasing);
        assert!(matches!(
            booster.fit(&x, &y, &s),
            Err(PbError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn monotone_constraint_admits_compatible_splits_and_rejects_opposite_direction() {
        let x = binned(&[vec![1.0, 1.0, 2.0, 2.0]]);
        let booster = Booster::with_config(Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 1,
            learning_rate: 1.0,
            lambda: 0.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        });
        let sqe = SquaredError;

        let mut inc = spec(&sqe);
        inc.monotone.insert("f0".into(), MonoSign::Increasing);
        let increasing = [0.0_f32, 0.0, 10.0, 10.0];
        let model = booster.fit(&x, &increasing, &inc).unwrap();
        assert_eq!(model.trees.len(), 1);
        assert!(predict(&model, &x, 0) <= predict(&model, &x, 2));

        let anti_monotone = [10.0_f32, 10.0, 0.0, 0.0];
        let model = booster.fit(&x, &anti_monotone, &inc).unwrap();
        assert!(
            model.trees.is_empty(),
            "anti-monotone split should terminate gracefully"
        );
        assert_eq!(predict(&model, &x, 0), predict(&model, &x, 2));
    }

    #[test]
    fn monotone_holds_under_ridge_refit() {
        // The §09 fully-corrective ridge refit re-solves leaves UNCONSTRAINED; the §07.5
        // clamp at the end of the refit must keep the served total score monotone.
        let sqe = SquaredError;
        let vals: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let x = binned(&[vals]);
        let y = [0.0_f32, 2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 9.0]; // increasing with local dips
        let mut sp = spec(&sqe);
        sp.monotone.insert("f0".into(), MonoSign::Increasing);
        let model = Booster::with_config(Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 30,
            learning_rate: 0.5,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Sampling::Full,
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: BoosterConfig {
                refit_leaves: RefitSpec::Ridge {
                    l2: 0.1,
                    max_iter: 4,
                    every_k_trees: None,
                },
                ..Default::default()
            },
            fit_control: Default::default(),
        })
        .fit(&x, &y, &sp)
        .unwrap();
        let preds: Vec<f64> = (0..8).map(|i| predict(&model, &x, i)).collect();
        for i in 1..8 {
            assert!(
                preds[i - 1] <= preds[i] + 1e-4,
                "ridge refit broke monotonicity: {preds:?}"
            );
        }
    }

    #[test]
    fn monotone_holds_under_mvs_sampling() {
        // MVS grows structure on a sampled subset then refits leaves on ALL rows; the
        // §07.5 clamp in refit_tree_leaves must keep the served total score monotone.
        let sqe = SquaredError;
        let n = 40usize;
        let vals: Vec<f32> = (0..n).map(|i| (i % 8 + 1) as f32).collect();
        let x = binned(&[vals]);
        let y: Vec<f32> = (0..n)
            .map(|i| (i % 8) as f32 + if i % 3 == 0 { 2.0 } else { 0.0 })
            .collect();
        let mut sp = spec(&sqe);
        sp.monotone.insert("f0".into(), MonoSign::Increasing);
        let model = Booster::with_config(Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 30,
            learning_rate: 0.5,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Sampling::Mvs {
                rate: 0.5,
                min_rows: 4,
            },
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        })
        .fit(&x, &y, &sp)
        .unwrap();
        let mut by_level: Vec<(u8, f64)> = (0..n)
            .map(|i| (x.data[0][i], predict(&model, &x, i)))
            .collect();
        by_level.sort_by_key(|&(b, _)| b);
        for w in by_level.windows(2) {
            assert!(
                w[0].1 <= w[1].1 + 1e-4,
                "MVS refit broke monotonicity: {by_level:?}"
            );
        }
    }

    #[test]
    fn malformed_binned_matrix_errors_at_fit_boundary() {
        let sqe = SquaredError;
        let x = binned(&[vec![1.0, 2.0, 3.0]]);
        let y = [1.0_f32, 2.0, 3.0];
        let booster = Booster::new();

        let mut bad = x.clone();
        bad.data[0].push(1);
        assert!(matches!(
            booster.fit(&bad, &y, &spec(&sqe)),
            Err(PbError::ShapeMismatch { .. })
        ));

        let mut bad = x.clone();
        bad.grids.pop();
        assert!(matches!(
            booster.fit(&bad, &y, &spec(&sqe)),
            Err(PbError::ShapeMismatch { .. })
        ));

        let mut bad = x.clone();
        bad.provenance.pop();
        assert!(matches!(
            booster.fit(&bad, &y, &spec(&sqe)),
            Err(PbError::ShapeMismatch { .. })
        ));

        let mut bad = x.clone();
        bad.grids[0].n_bins = 0;
        assert!(matches!(
            booster.fit(&bad, &y, &spec(&sqe)),
            Err(PbError::InvalidInput { .. })
        ));

        let mut bad = x.clone();
        bad.grids[0].borders.clear();
        assert!(matches!(
            booster.fit(&bad, &y, &spec(&sqe)),
            Err(PbError::InvalidInput { .. })
        ));

        let mut bad = x;
        bad.data[0][0] = u8::MAX;
        assert!(matches!(
            booster.fit(&bad, &y, &spec(&sqe)),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn fit_train_persists_categorical_encoder_store() {
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
                raw: crate::data::FeatureId(0),
                values: &numeric,
            }],
            &[CategoricalColumn {
                raw: crate::data::FeatureId(1),
                id: TsEncodingId(0),
                levels: &levels,
                config: &ts,
            }],
            &y,
            None,
            None,
            &BinConfig::default(),
            12,
        )
        .unwrap();
        let sqe = SquaredError;
        let booster = Booster::with_config(Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 3,
            learning_rate: 1.0,
            lambda: 0.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        });
        let model = booster
            .fit_train(&fitted.train, &y, &spec(&sqe), fitted.cat_encoders.clone())
            .unwrap();
        assert_eq!(model.schema.cat_encoders.len(), 1);
        model.validate().unwrap();

        assert!(matches!(
            booster.fit_train(&fitted.train, &y, &spec(&sqe), CatEncoderStore::new()),
            Err(PbError::InvalidInput { .. })
        ));
    }

    fn additive_2feat(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let x0: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 3.0 { 10.0 } else { 20.0 };
                let b = if x1[i] <= 2.0 { 5.0 } else { 0.0 };
                a + b
            })
            .collect();
        (x0, x1, y)
    }

    #[test]
    fn per_tree_scorers_agree_bit_exactly() {
        // The column-major update scorer (tree_value_for_row) and the row-vector
        // ObliviousTree::lookup MUST produce bit-identical PER-TREE leaf values — both
        // fold the leaf index via the canonical low_bit. This is the structural
        // invariant that makes "the model scores what it trained on" hold; the f32
        // train sum and the f64 ensemble sum then differ only by accumulation WIDTH.
        let (x0, x1, y) = additive_2feat(60);
        let x = binned(&[x0, x1]);
        let sqe = SquaredError;
        let model = Booster::with_config(Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 30,
            learning_rate: 0.3,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        })
        .fit(&x, &y, &spec(&sqe))
        .unwrap();
        for r in 0..x.n_rows as usize {
            let bins: Vec<u8> = x.data.iter().map(|c| c[r]).collect();
            for (_, tree) in &model.trees {
                assert_eq!(
                    tree_value_for_row(tree, &x, r).unwrap(),
                    tree.lookup(&bins).unwrap()
                );
            }
        }
    }

    #[test]
    fn f32_train_raw_matches_f64_ensemble_within_reconstruction_tol() {
        // Training optimizes an f32-accumulated raw (the §05 `grad_hess` takes
        // `raw: &[f32]`); ensemble_f64 / the §08 tables accumulate in f64. The two
        // agree within ~4·n_trees·f32::EPSILON·magnitude — exactly the tolerance the
        // §08 Reconstruction gate is sized for, NOT a routing/structural bug.
        let (x0, x1, y) = additive_2feat(80);
        let x = binned(&[x0, x1]);
        let sqe = SquaredError;
        let model = Booster::with_config(Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 200,
            learning_rate: 0.3,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        })
        .fit(&x, &y, &spec(&sqe))
        .unwrap();
        let n_trees = model.trees.len() as f64;
        for r in 0..x.n_rows as usize {
            let mut raw_f32: f32 = model.f0;
            for (_, tree) in &model.trees {
                raw_f32 += tree_value_for_row(tree, &x, r).unwrap();
            }
            let bins: Vec<u8> = x.data.iter().map(|c| c[r]).collect();
            let ens = model.ensemble_f64(&bins).unwrap();
            let tol = 4.0 * n_trees * f64::from(f32::EPSILON) * (1.0 + ens.abs());
            assert!(
                (f64::from(raw_f32) - ens).abs() <= tol,
                "raw_f32 {raw_f32} vs ensemble_f64 {ens} exceeds recon tol {tol}"
            );
        }
    }

    #[test]
    fn weighted_fit_recovers_target() {
        // Weights scale (g,h) and the init mean; an exactly-representable target is
        // still recovered (each region's WEIGHTED mean equals its constant). λ=0,lr=1.
        let (x0, x1, y) = additive_2feat(60);
        let x = binned(&[x0, x1]);
        let w: Vec<f32> = (0..y.len()).map(|i| 0.5 + (i % 4) as f32).collect();
        let sqe = SquaredError;
        let mut s = spec(&sqe);
        s.weight = Some(&w);
        let model = Booster::with_config(Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 20,
            learning_rate: 1.0,
            lambda: 0.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        })
        .fit(&x, &y, &s)
        .unwrap();
        for (i, &yi) in y.iter().enumerate() {
            assert!((predict(&model, &x, i) - f64::from(yi)).abs() < 1e-3);
        }
    }

    #[test]
    fn exposure_fit_produces_finite_exact_model() {
        // Smoke test of the offset path: exposure → offset = ln(e) folded into raw,
        // offset-aware init_score. A valid finite Exact model results.
        let (x0, x1, y) = additive_2feat(40);
        let x = binned(&[x0, x1]);
        let e: Vec<f32> = (0..y.len()).map(|i| 1.0 + (i % 3) as f32 * 0.5).collect();
        let sqe = SquaredError;
        let mut s = spec(&sqe);
        s.exposure = Some(&e);
        let model = Booster::with_config(Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 15,
            learning_rate: 0.3,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        })
        .fit(&x, &y, &s)
        .unwrap();
        assert_eq!(model.mode, ExactnessMode::Exact);
        for i in 0..x.n_rows as usize {
            assert!(predict(&model, &x, i).is_finite());
        }
    }

    #[test]
    fn reanchor_shifts_only_intercept_and_matches_response_total() {
        let n = 96usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 8 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (1 + i % 7) as f32).collect();
        let x = binned(&[x0, x1]);
        let base_cfg = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 8,
            learning_rate: 0.4,
            lambda: 2.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        };
        let anchored_cfg = Config {
            boosters: BoosterConfig {
                reanchor: true,
                ..BoosterConfig::default()
            },
            ..base_cfg.clone()
        };
        let base = Booster::with_config(base_cfg)
            .fit(&x, &y, &spec(&Poisson))
            .unwrap();
        let anchored = Booster::with_config(anchored_cfg)
            .fit(&x, &y, &spec(&Poisson))
            .unwrap();

        assert_eq!(base.trees, anchored.trees);
        assert_ne!(base.f0, anchored.f0);
        let observed: f64 = y.iter().map(|&yi| f64::from(yi)).sum();
        let predicted: f64 = anchored
            .predict_binned(&x, None)
            .unwrap()
            .iter()
            .map(|&yi| f64::from(yi))
            .sum();
        assert!((predicted - observed).abs() < 1.0e-3);

        let serve = crate::data::ServeBinnedMatrix(x);
        let bank = anchored.explain(&serve, RefMeasure::default()).unwrap();
        assert_exact_decomposition(&anchored, &bank, &serve).unwrap();
    }

    #[test]
    fn g2_recovers_three_feature_target() {
        // Order-3: an additive piecewise-constant target on 3 features, recovered
        // exactly with λ=0 (a full depth-3 tree captures the 8 regions).
        let n = 64usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 2 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| ((i / 2) % 2 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| ((i / 4) % 2 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 1.0 { 0.0 } else { 4.0 };
                let b = if x1[i] <= 1.0 { 0.0 } else { 2.0 };
                let c = if x2[i] <= 1.0 { 0.0 } else { 1.0 };
                a + b + c - 3.5
            })
            .collect();
        let x = binned(&[x0, x1, x2]);
        let sqe = SquaredError;
        let model = Booster::with_config(Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 20,
            learning_rate: 1.0,
            lambda: 0.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Default::default(),
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        })
        .fit(&x, &y, &spec(&sqe))
        .unwrap();
        for (i, &yi) in y.iter().enumerate() {
            assert!(
                (predict(&model, &x, i) - f64::from(yi)).abs() < 1e-3,
                "row {i}: {} != {yi}",
                predict(&model, &x, i)
            );
        }
    }

    #[test]
    fn mvs_sampler_is_deterministic_and_gradient_weighted() {
        let rows: Vec<u32> = (0..20).collect();
        let gh = GradHess {
            g: (0..20)
                .map(|i| if i < 5 { 20.0 - i as f32 } else { 0.1 })
                .collect(),
            h: vec![1.0; 20],
        };
        let sampling = Sampling::Mvs {
            rate: 0.25,
            min_rows: 4,
        };
        let (rows_a, reweight_a) = sample_rows(&sampling, &gh, 123, 7, &rows).unwrap();
        let (rows_b, reweight_b) = sample_rows(&sampling, &gh, 123, 7, &rows).unwrap();
        assert_eq!(rows_a, rows_b);
        assert_eq!(reweight_a, reweight_b);
        assert_eq!(rows_a.len(), 5);
        let sampled_mean_g: f32 =
            rows_a.iter().map(|&r| gh.g[r as usize].abs()).sum::<f32>() / rows_a.len() as f32;
        let population_mean_g: f32 = gh.g.iter().map(|g| g.abs()).sum::<f32>() / gh.g.len() as f32;
        assert!(sampled_mean_g > population_mean_g);
        // §06.5: 1/p_i >= 1 everywhere, and — since p_i = min(1, s_i/mu) is monotone in the
        // row's own weight s_i — a higher-gradient sampled row must never be reweighted MORE
        // than a lower-gradient one (the exact closed-form values are pinned separately by
        // `mvs_reweight_matches_the_closed_form_and_is_absent_under_full_sampling`).
        let reweight = reweight_a.expect("Sampling::Mvs always returns a reweight vector");
        assert_eq!(reweight.len(), rows_a.len());
        for (&r, &mult) in rows_a.iter().zip(&reweight) {
            assert!(mult >= 1.0, "1/p_i must be >= 1, got {mult} for row {r}");
        }
        let by_weight_desc: Vec<(f32, f64)> = {
            let mut v: Vec<(f32, f64)> = rows_a
                .iter()
                .zip(&reweight)
                .map(|(&r, &mult)| (gh.g[r as usize].abs(), mult))
                .collect();
            v.sort_by(|a, b| b.0.total_cmp(&a.0));
            v
        };
        for w in by_weight_desc.windows(2) {
            assert!(
                w[0].1 <= w[1].1 + 1e-12,
                "a higher-gradient row must not be reweighted more than a lower-gradient one: {w:?}"
            );
        }
    }

    #[test]
    fn bug001_scalar_and_multiclass_mvs_estimate_population_totals() {
        let gh = GradHess {
            g: vec![100., 0., 0.],
            h: vec![1.; 3],
        };
        let sampling = Sampling::Mvs {
            rate: 0.5,
            min_rows: 1,
        };
        let rows = [0, 1, 2];
        for multiclass in [false, true] {
            let mut total_h = 0.;
            let mut total_g = 0.;
            for seed in 0..10_000 {
                let (selected, weights) = if multiclass {
                    mc_sample_rows(&sampling, std::slice::from_ref(&gh), seed, 0, &rows)
                } else {
                    sample_rows(&sampling, &gh, seed, 0, &rows)
                }
                .unwrap();
                let weights = weights.unwrap();
                for (&row, &weight) in selected.iter().zip(&weights) {
                    total_h += f64::from(gh.h[row as usize]) * weight;
                    total_g += f64::from(gh.g[row as usize]) * weight;
                }
            }
            assert!(
                (total_h / 10_000. - 3.).abs() < 0.1,
                "biased Hessian estimate: {} (multiclass={multiclass})",
                total_h / 10_000.
            );
            assert!((total_g / 10_000. - 100.).abs() < 1.);
        }
    }

    #[test]
    fn mvs_reweight_matches_the_closed_form_and_is_absent_under_full_sampling() {
        // Sampling::Full needs no reweighting: every row is included with p_i = 1.
        let rows: Vec<u32> = (0..10).collect();
        let gh_full = GradHess {
            g: vec![1.0; 10],
            h: vec![1.0; 10],
        };
        let (full_rows, full_reweight) =
            sample_rows(&Sampling::Full, &gh_full, 1, 0, &rows).unwrap();
        assert_eq!(full_rows, rows);
        assert!(full_reweight.is_none());

        // This fixture has no saturated rows, so the solution of sum p_i = k is mu = Σs/k.
        // §06.5: p_i = min(1, s_i/mu). Recompute
        // mu and each selected row's expected 1/p_i independently from the SAME per-row weights
        // sample_rows derives from `gh`, and assert an exact match — deterministic plumbing, not
        // a statistical property.
        let g: Vec<f32> = (0..30).map(|i| 1.0 + (i % 7) as f32 * 3.0).collect();
        let h = vec![1.0_f32; 30];
        let gh = GradHess {
            g: g.clone(),
            h: h.clone(),
        };
        let rows: Vec<u32> = (0..30).collect();
        let sampling = Sampling::Mvs {
            rate: 0.4,
            min_rows: 1,
        };
        let (sampled, reweight) = sample_rows(&sampling, &gh, 42, 3, &rows).unwrap();
        let reweight = reweight.expect("Sampling::Mvs always returns a reweight vector");
        assert_eq!(sampled.len(), reweight.len());
        assert!(
            sampled.len() < rows.len(),
            "test must exercise the sub-sampled branch"
        );
        let k = sampled.len() as f64;
        let total_weight: f64 = g
            .iter()
            .zip(&h)
            .map(|(&gi, &hi)| {
                (f64::from(gi) * f64::from(gi) + f64::from(hi) * f64::from(hi))
                    .sqrt()
                    .max(1e-12)
            })
            .sum();
        let mu = total_weight / k;
        for (&row, &mult) in sampled.iter().zip(&reweight) {
            let gi = f64::from(g[row as usize]);
            let hi = f64::from(h[row as usize]);
            let s = (gi * gi + hi * hi).sqrt().max(1e-12);
            let expected = (mu / s).max(1.0);
            assert!(
                (mult - expected).abs() < 1e-9 * expected.max(1.0),
                "row {row}: expected 1/p_i={expected}, got {mult}"
            );
            assert!(mult >= 1.0, "1/p_i must be >= 1, got {mult} for row {row}");
        }
    }

    #[test]
    fn mvs_config_validation_is_fail_closed() {
        for sampling in [
            Sampling::Mvs {
                rate: 0.0,
                min_rows: 1,
            },
            Sampling::Mvs {
                rate: 1.1,
                min_rows: 1,
            },
            Sampling::Mvs {
                rate: 0.5,
                min_rows: 0,
            },
        ] {
            let cfg = Config {
                sampling,
                ..Config::default()
            };
            assert!(matches!(cfg.validate(), Err(PbError::InvalidConfig { .. })));
        }
    }

    #[test]
    fn mvs_fit_stays_exact_and_thread_deterministic() {
        let n = 180usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 9 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 7 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 4.0 { 2.0 } else { -1.0 };
                let b = if x1[i] <= 3.0 { 1.5 } else { -0.5 };
                let c = if x2[i] <= 2.0 { 0.75 } else { -0.25 };
                a + b + c
            })
            .collect();
        let x = binned(&[x0, x1, x2]);
        let cfg = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 40,
            learning_rate: 0.3,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Sampling::Mvs {
                rate: 0.6,
                min_rows: 40,
            },
            hist_precision: Default::default(),
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        };
        let sqe = SquaredError;
        let bytes = |nt: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let model = Booster::with_config(cfg.clone())
                    .fit(&x, &y, &spec(&sqe))
                    .unwrap();
                let serve = crate::data::ServeBinnedMatrix(x.clone());
                let bank = model.explain(&serve, RefMeasure::default()).unwrap();
                assert_exact_decomposition(&model, &bank, &serve).unwrap();
                crate::serialize::encode_model(&model).unwrap()
            })
        };
        let b1 = bytes(1);
        assert_eq!(b1, bytes(2));
        assert_eq!(b1, bytes(8));
    }

    #[test]
    fn new_accuracy_knobs_stay_exact_and_thread_deterministic() {
        let n = 220usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 9 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 7 + 1) as f32).collect();
        let x2: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let x3: Vec<f32> = (0..n).map(|i| (i % 4 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 4.0 { 2.5 } else { -1.0 };
                let b = if x1[i] <= 3.0 { 1.25 } else { -0.75 };
                let c = if x2[i] <= 2.0 { 0.8 } else { -0.2 };
                a + b + c + (i % 11) as f32 * 0.01
            })
            .collect();
        let x = binned(&[x0, x1, x2, x3]);
        let cfg = Config {
            n_trees: 45,
            learning_rate: 0.25,
            lambda: 1.0,
            l1_leaf: 0.02,
            colsample_bytree: 0.6,
            learning_rate_decay: 0.05,
            validation_fraction: Some(0.2),
            early_stopping_rounds: 4,
            early_stopping_adaptive: None,
            ..Config::default()
        };
        let sqe = SquaredError;
        let bytes = |nt: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let model = Booster::with_config(cfg.clone())
                    .fit(&x, &y, &spec(&sqe))
                    .unwrap();
                assert_eq!(model.mode, ExactnessMode::Exact);
                assert!(!model.trees.is_empty());
                assert!(model.trees.len() <= cfg.n_trees as usize);
                let serve = crate::data::ServeBinnedMatrix(x.clone());
                let bank = model.explain(&serve, RefMeasure::default()).unwrap();
                assert_exact_decomposition(&model, &bank, &serve).unwrap();
                crate::serialize::encode_model(&model).unwrap()
            })
        };
        let b1 = bytes(1);
        assert_eq!(b1, bytes(2));
        assert_eq!(b1, bytes(8));
    }

    #[test]
    fn leaf_refinement_stays_exact_and_does_not_increase_deviance() {
        let n = 180usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 8 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                let a = if x0[i] <= 4.0 { 0.8 } else { 1.6 };
                let b = if x1[i] <= 2.0 { 1.2 } else { 0.7 };
                a * b + (i % 7) as f32 * 0.03
            })
            .collect();
        let x = binned(&[x0, x1]);
        let base_cfg = Config {
            n_trees: 12,
            learning_rate: 0.3,
            lambda: 1.0,
            ..Config::default()
        };
        let refined_cfg = Config {
            leaf_refine_steps: 2,
            leaf_refine_backtracks: 4,
            ..base_cfg.clone()
        };
        let gamma = Gamma;
        let base = Booster::with_config(base_cfg)
            .fit(&x, &y, &spec(&gamma))
            .unwrap();
        let refined = Booster::with_config(refined_cfg)
            .fit(&x, &y, &spec(&gamma))
            .unwrap();
        let mut base_raw = vec![0.0_f32; n];
        let mut refined_raw = vec![0.0_f32; n];
        base.score_trees(&x, None, &mut base_raw).unwrap();
        refined.score_trees(&x, None, &mut refined_raw).unwrap();
        let w = vec![1.0_f32; n];
        let base_dev = gamma.deviance(&y, &base_raw, &w).unwrap();
        let refined_dev = gamma.deviance(&y, &refined_raw, &w).unwrap();
        assert!(
            refined_dev <= base_dev + 1.0e-6,
            "leaf refinement worsened deviance: {refined_dev} > {base_dev}"
        );

        let serve = crate::data::ServeBinnedMatrix(x);
        let bank = refined.explain(&serve, RefMeasure::default()).unwrap();
        assert_exact_decomposition(&refined, &bank, &serve).unwrap();
    }

    #[test]
    fn new_accuracy_config_validation_is_fail_closed() {
        for cfg in [
            Config {
                l1_leaf: -1.0,
                ..Config::default()
            },
            Config {
                colsample_bytree: 0.0,
                ..Config::default()
            },
            Config {
                colsample_bytree: 1.1,
                ..Config::default()
            },
            Config {
                learning_rate_decay: -0.1,
                ..Config::default()
            },
            Config {
                validation_fraction: Some(0.0),
                ..Config::default()
            },
            Config {
                validation_fraction: Some(1.0),
                ..Config::default()
            },
            Config {
                validation_fraction: Some(0.2),
                early_stopping_rounds: 0,
                early_stopping_adaptive: None,
                ..Config::default()
            },
            Config {
                leaf_refine_steps: 1,
                leaf_refine_backtracks: 0,
                ..Config::default()
            },
        ] {
            assert!(matches!(cfg.validate(), Err(PbError::InvalidConfig { .. })));
        }
    }

    #[test]
    fn quantized_hist_fit_stays_exact_and_thread_deterministic() {
        let n = 160usize;
        let x0: Vec<f32> = (0..n).map(|i| (i % 8 + 1) as f32).collect();
        let x1: Vec<f32> = (0..n).map(|i| (i % 6 + 1) as f32).collect();
        let y: Vec<f32> = (0..n)
            .map(|i| {
                (if x0[i] <= 4.0 { -2.0 } else { 3.0 }) + if x1[i] <= 3.0 { 1.0 } else { -1.0 }
            })
            .collect();
        let x = binned(&[x0, x1]);
        let cfg = Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: GatedStepPolicy::Objective,
            n_trees: 30,
            learning_rate: 0.3,
            lambda: 1.0,
            min_split_gain: 0.0,
            max_delta_step: None,
            sampling: Sampling::Full,
            hist_precision: HistPrecision::QuantizedI32,
            l1_leaf: 0.0,
            colsample_bytree: 1.0,
            learning_rate_decay: 0.0,
            validation_fraction: None,
            early_stopping_rounds: 50,
            early_stopping_adaptive: None,
            early_stopping_min_delta: 0.0,
            interaction_gain_hurdle: 0.0,
            interaction_gain_hurdle_mode: crate::engine::InteractionGainHurdleMode::Fixed,
            leaf_refine_steps: 0,
            leaf_refine_backtracks: 4,
            refine_closed_form_tier2: false,
            incremental_mu: false,
            boosters: Default::default(),
            fit_control: Default::default(),
        };
        let sqe = SquaredError;
        let bytes = |nt: usize| -> Vec<u8> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| {
                let model = Booster::with_config(cfg.clone())
                    .fit(&x, &y, &spec(&sqe))
                    .unwrap();
                let serve = crate::data::ServeBinnedMatrix(x.clone());
                let bank = model.explain(&serve, RefMeasure::Uniform).unwrap();
                assert_exact_decomposition(&model, &bank, &serve).unwrap();
                crate::serialize::encode_model(&model).unwrap()
            })
        };
        let b1 = bytes(1);
        assert_eq!(b1, bytes(2));
        assert_eq!(b1, bytes(8));
    }

    /// Deterministic xorshift64* — reproducible fixture noise in `[0, 1)` (test-only; production
    /// randomness is the named `Pcg64` stream).
    fn xs01(state: &mut u64) -> f64 {
        let mut x = *state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        *state = x;
        ((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn test_grow_cfg(lambda: f64, lr: f64, max_delta_step: Option<f64>) -> GrowConfig<'static> {
        GrowConfig {
            lambda,
            l1_leaf: 0.0,
            lr,
            min_split_gain: 0.0,
            interaction_gain_hurdle: 0.0,
            max_order: 3,
            max_depth: LEGACY_MAX_DEPTH as u8,
            max_delta_step,
            hist_precision: HistPrecision::FullF64,
            quant_seed: 0,
            round: 0,
            random_strength: 0.0,
            groups: None,
            monotone: None,
            table_budget_penalty: None,
            realized_extent: None,
            credibility: CredibilityFloor::default(),
            credibility_evidence: CredibilityEvidence::Count,
            unit_weight: false,
            hist_subtraction: true,
        }
    }

    /// (P2, 2026-07-16, Ralph-ratified drift) The parallel chunked coefficient fold:
    /// (a) deterministic across thread-pool widths — fixed `LEAF_COEFF_CHUNK_ROWS` boundaries
    /// and a chunk-order combine make the result a pure function of the data;
    /// (b) tracks the serial single-fold to ≤1e-9 relative (measured ~1e-12; the f64 sums are
    /// merely regrouped per chunk) with `max_abs_base` EXACT (max is associative);
    /// (c) below two chunks the serial path runs — bit-identical to the historical fold.
    #[test]
    fn poisson_leaf_coeffs_parallel_fold_is_thread_invariant_with_bounded_drift() {
        let n = LEAF_COEFF_PAR_MIN_ROWS + 12_345;
        let mut s = 7u64;
        let mut y = vec![0.0_f32; n];
        let mut w = vec![0.0_f32; n];
        let mut base = vec![0.0_f32; n];
        let mut mem = vec![0u8; n];
        for i in 0..n {
            y[i] = if xs01(&mut s) < 0.3 {
                0.0
            } else {
                (0.2 + xs01(&mut s) * 5.0) as f32
            };
            w[i] = (0.1 + xs01(&mut s) * 3.0) as f32;
            base[i] = ((xs01(&mut s) - 0.5) * 5.0) as f32;
            mem[i] = ((xs01(&mut s) * 8.0) as usize % 8) as u8;
        }
        let run = |nt: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| poisson_leaf_coeffs(&y, &w, &base, &mem).unwrap())
        };
        let bits = |c: &PoissonLeafCoeffs| -> (Vec<u64>, Vec<u64>, u64, u64, u32) {
            (
                c.m.iter().map(|v| v.to_bits()).collect(),
                c.a.iter().map(|v| v.to_bits()).collect(),
                c.k.to_bits(),
                c.sum_w.to_bits(),
                c.max_abs_base.to_bits(),
            )
        };
        // (a) thread-count invariance, bit-for-bit
        assert_eq!(bits(&run(1)), bits(&run(8)));

        // (b) bounded drift vs the serial single-fold
        let par = run(8);
        let ser = poisson_leaf_coeffs_serial(&y, &w, &base, &mem).unwrap();
        let rel = |p: f64, q: f64| (p - q).abs() / q.abs().max(1e-12);
        for l in 0..8 {
            assert!(rel(par.m[l], ser.m[l]) < 1e-9, "m[{l}] drift too large");
            assert!(rel(par.a[l], ser.a[l]) < 1e-9, "a[{l}] drift too large");
        }
        assert!(rel(par.k, ser.k) < 1e-9);
        assert!(rel(par.sum_w, ser.sum_w) < 1e-9);
        assert_eq!(par.max_abs_base.to_bits(), ser.max_abs_base.to_bits());

        // (c) below-threshold subsets: the parallel entry point IS the serial fold, bit-for-bit
        let small = LEAF_COEFF_PAR_MIN_ROWS - 1;
        let via_entry =
            poisson_leaf_coeffs(&y[..small], &w[..small], &base[..small], &mem[..small]).unwrap();
        let via_serial =
            poisson_leaf_coeffs_serial(&y[..small], &w[..small], &base[..small], &mem[..small])
                .unwrap();
        assert_eq!(bits(&via_entry), bits(&via_serial));
    }

    /// (a) The factored per-leaf coefficients reproduce the exact Poisson weighted deviance. The
    /// closed form `D = K + Σ_l (e^{v_l}·M_l − v_l·A_l)` must equal a straight f64 per-row deviance
    /// fold to ≤1e-10 relative on random WEIGHTED fixtures that include `y=0` rows and populate all
    /// 8 leaves — the correctness of the coefficient algebra. Separately, the SHIPPED f32-cast
    /// `loglink_closed_deviance` must track `Poisson::deviance` (the per-row f32 exp path) to f32
    /// precision — the shared grid that keeps Tier-1 accept decisions matching the generic path.
    #[test]
    fn poisson_closed_form_deviance_matches_f64_reference() {
        for seed in 0..48u64 {
            let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            let n = 160usize + (seed as usize % 64);
            let n_leaves = 8usize;
            let mut y = vec![0.0_f32; n];
            let mut w = vec![0.0_f32; n];
            let mut base = vec![0.0_f32; n];
            let mut mem = vec![0u8; n];
            for i in 0..n {
                // ~30% exact zeros; else small positive counts/frequencies.
                y[i] = if xs01(&mut s) < 0.3 {
                    0.0
                } else {
                    (0.2 + xs01(&mut s) * 5.0) as f32
                };
                w[i] = (0.1 + xs01(&mut s) * 3.0) as f32; // exposure-like positive weights
                base[i] = ((xs01(&mut s) - 0.5) * 5.0) as f32; // base in ~[-2.5, 2.5]
                mem[i] = ((xs01(&mut s) * n_leaves as f64) as usize % n_leaves) as u8;
            }
            let leaves: [f32; 8] = std::array::from_fn(|_| ((xs01(&mut s) - 0.5) * 0.9) as f32);

            let coeffs = poisson_leaf_coeffs(&y, &w, &base, &mem).unwrap();

            // f64 reference: the exact per-row Poisson deviance at F_i = base_i + leaves[mem_i].
            let mut d_ref = 0.0_f64;
            for i in 0..n {
                let f = f64::from(base[i]) + f64::from(leaves[usize::from(mem[i])]);
                let mu = f.exp();
                let yy = f64::from(y[i]);
                let term = if yy > 0.0 { yy * (yy / mu).ln() } else { 0.0 };
                d_ref += f64::from(w[i]) * 2.0 * (term - (yy - mu));
            }
            // Pre-cast closed form (factored coeffs) — the algebra under test.
            let mut d_closed = coeffs.k;
            for l in 0..n_leaves {
                let v = f64::from(leaves[l]);
                d_closed += v.exp() * coeffs.m[l] - v * coeffs.a[l];
            }
            let rel = (d_closed - d_ref).abs() / d_ref.abs().max(1e-12);
            assert!(
                rel < 1e-10,
                "seed {seed}: closed {d_closed} vs f64 ref {d_ref}, rel {rel:e}"
            );

            // Shipped f32-cast form vs the per-row f32 `Poisson::deviance`: same f32 grid.
            let mut raw = vec![0.0_f32; n];
            for i in 0..n {
                raw[i] = (f64::from(base[i]) + f64::from(leaves[usize::from(mem[i])])) as f32;
            }
            let d_perrow = f64::from(Poisson.deviance(&y, &raw, &w).unwrap());
            let d_ship = loglink_closed_deviance(&coeffs, &leaves, n_leaves);
            let rel_f32 = (d_ship - d_perrow).abs() / d_perrow.abs().max(1e-12);
            assert!(
                rel_f32 < 1e-4,
                "seed {seed}: shipped {d_ship} vs per-row {d_perrow}, rel {rel_f32:e}"
            );
        }
    }

    /// (b) Tier-1 leaf-selection identity: the SAME Poisson model fit with the log-link closed-form
    /// refine ON (production) vs OFF (forced generic per-row line search) must serialize BIT-FOR-BIT
    /// identically. Tier 1 keeps the leaf UPDATES byte-identical and only closes the deviance, so the
    /// accepted leaves — hence the whole ensemble — match. Swept over learning rates / refine steps /
    /// tree counts (many small trees ⇒ the "randomized small trees, multiple seeds" property), and
    /// once with exposure weights to exercise the weighted closed form.
    #[test]
    fn poisson_closed_refine_matches_generic_leaves_bit_for_bit() {
        for (idx, (lr, steps, n_trees, weighted)) in [
            (0.3f32, 2u8, 30u32, false),
            (0.1, 4, 40, false),
            (0.5, 1, 24, false),
            (0.2, 3, 36, true),
        ]
        .into_iter()
        .enumerate()
        {
            let n = 240usize;
            let x0: Vec<f32> = (0..n).map(|i| (i % 9 + 1) as f32).collect();
            let x1: Vec<f32> = (0..n).map(|i| (i % 7 + 1) as f32).collect();
            let x2: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
            // Non-negative count-like target with structure and ~20% exact zeros (Poisson domain).
            let y: Vec<f32> = (0..n)
                .map(|i| {
                    if i % 5 == 0 {
                        0.0
                    } else {
                        let a = if x0[i] <= 4.0 { 0.4 } else { 1.3 };
                        let b = if x1[i] <= 3.0 { 1.5 } else { 0.8 };
                        a * b + (i % 4) as f32 * 0.2
                    }
                })
                .collect();
            let x = binned(&[x0, x1, x2]);
            let cfg = Config {
                n_trees,
                learning_rate: lr,
                lambda: 1.0,
                leaf_refine_steps: steps,
                leaf_refine_backtracks: 4,
                // Tier 1: the closed form keeps the leaf UPDATES byte-identical to the generic
                // per-row path — that is exactly what this test asserts. (The default is Tier 2,
                // whose ~1e-7 delta drift is covered by `poisson_closed_refine_tier2_*` below.)
                refine_closed_form_tier2: false,
                incremental_mu: false,
                ..Config::default()
            };
            let poisson = Poisson;
            let wts: Vec<f32> = (0..n).map(|i| 0.2 + (i % 7) as f32 * 0.3).collect();
            let fit = || {
                let s = FitSpec {
                    loss: &poisson,
                    weight: if weighted { Some(&wts) } else { None },
                    exposure: None,
                    monotone: crate::constraints::MonotoneMap::new(),
                    interaction: crate::constraints::InteractionPolicy::default(),
                    credibility: crate::constraints::CredibilityFloor::default(),
                    fixed_holdout: None,
                    bag_groups: None,
                    seed: idx as u64,
                };
                crate::serialize::encode_model(
                    &Booster::with_config(cfg.clone()).fit(&x, &y, &s).unwrap(),
                )
                .unwrap()
            };
            let closed = fit(); // Tier 1: closed-form refine active, leaf updates byte-identical
            let generic = with_loglink_closed_disabled(fit); // forced generic per-row path
            assert_eq!(
                closed, generic,
                "config #{idx} (lr={lr}, steps={steps}, weighted={weighted}): closed-form refine \
                 diverged from the generic per-row line search"
            );
        }
    }

    /// (b2) Tier-2 leaf-selection tolerance: the SAME Poisson fit with the DEFAULT Tier-2 closed-form
    /// refine (per-step Newton deltas taken straight from the closed form) tracks the exact generic
    /// per-row line search to a small tolerance — NOT bit-for-bit (the f32 `grad_hess` vs f64
    /// closed-form deltas drift ~1e-7 per step and compound through boosting), but close enough that
    /// the shipped default is a faithful accelerator. The byte-identity guarantee is Tier 1's job
    /// (test (b)); here we bound the drift of the scored raw predictions.
    #[test]
    fn poisson_closed_refine_tier2_matches_generic_within_tolerance() {
        for (idx, (lr, steps, n_trees, weighted)) in [
            (0.3f32, 2u8, 30u32, false),
            (0.1, 4, 40, false),
            (0.5, 1, 24, false),
            (0.2, 3, 36, true),
        ]
        .into_iter()
        .enumerate()
        {
            let n = 240usize;
            let x0: Vec<f32> = (0..n).map(|i| (i % 9 + 1) as f32).collect();
            let x1: Vec<f32> = (0..n).map(|i| (i % 7 + 1) as f32).collect();
            let x2: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
            let y: Vec<f32> = (0..n)
                .map(|i| {
                    if i % 5 == 0 {
                        0.0
                    } else {
                        let a = if x0[i] <= 4.0 { 0.4 } else { 1.3 };
                        let b = if x1[i] <= 3.0 { 1.5 } else { 0.8 };
                        a * b + (i % 4) as f32 * 0.2
                    }
                })
                .collect();
            let x = binned(&[x0, x1, x2]);
            let cfg = Config {
                n_trees,
                learning_rate: lr,
                lambda: 1.0,
                leaf_refine_steps: steps,
                leaf_refine_backtracks: 4,
                refine_closed_form_tier2: true, // the shipped default
                ..Config::default()
            };
            let poisson = Poisson;
            let wts: Vec<f32> = (0..n).map(|i| 0.2 + (i % 7) as f32 * 0.3).collect();
            let fit = || {
                let s = FitSpec {
                    loss: &poisson,
                    weight: if weighted { Some(&wts) } else { None },
                    exposure: None,
                    monotone: crate::constraints::MonotoneMap::new(),
                    interaction: crate::constraints::InteractionPolicy::default(),
                    credibility: crate::constraints::CredibilityFloor::default(),
                    fixed_holdout: None,
                    bag_groups: None,
                    seed: idx as u64,
                };
                Booster::with_config(cfg.clone()).fit(&x, &y, &s).unwrap()
            };
            let tier2 = fit(); // production default: Tier-2 closed-form deltas
            let generic = with_loglink_closed_disabled(fit); // exact generic per-row reference
            let mut raw_t2 = vec![0.0_f32; n];
            let mut raw_gen = vec![0.0_f32; n];
            tier2.score_trees(&x, None, &mut raw_t2).unwrap();
            generic.score_trees(&x, None, &mut raw_gen).unwrap();
            let max_abs = raw_t2
                .iter()
                .zip(&raw_gen)
                .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_abs < 1e-3,
                "config #{idx} (lr={lr}, steps={steps}, weighted={weighted}): Tier-2 closed-form \
                 refine drifted {max_abs:e} from the generic per-row path (raw score)"
            );
        }
    }

    /// (a-gamma) The Gamma factored coefficients reproduce the exact weighted Gamma deviance:
    /// `D = K + Σ_l (e^{−v_l}·M_l − v_l·A_l)` (A_l = −Σ2w) must equal a straight f64 per-row fold
    /// of `2w(r − 1 − ln r)`, r = y/μ, μ = e^{base+v}, to ≤1e-10 relative on random strictly
    /// positive weighted fixtures — and the SHIPPED f32-cast `loglink_closed_deviance` must track
    /// `Gamma::deviance` (the per-row f32 clamp_exp path) on the same f32 grid.
    #[test]
    fn gamma_closed_form_deviance_matches_f64_reference() {
        for seed in 0..48u64 {
            let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(11);
            let n = 160usize + (seed as usize % 64);
            let n_leaves = 8usize;
            let mut y = vec![0.0_f32; n];
            let mut w = vec![0.0_f32; n];
            let mut base = vec![0.0_f32; n];
            let mut mem = vec![0u8; n];
            for i in 0..n {
                y[i] = (0.05 + xs01(&mut s) * 8.0) as f32; // strictly positive (Gamma domain)
                w[i] = (0.1 + xs01(&mut s) * 3.0) as f32;
                base[i] = ((xs01(&mut s) - 0.5) * 5.0) as f32;
                mem[i] = ((xs01(&mut s) * n_leaves as f64) as usize % n_leaves) as u8;
            }
            let leaves: [f32; 8] = std::array::from_fn(|_| ((xs01(&mut s) - 0.5) * 0.9) as f32);

            let coeffs = gamma_leaf_coeffs(&y, &w, &base, &mem).unwrap();
            assert_eq!(coeffs.exp_sign, -1.0);

            // f64 reference: exact per-row Gamma deviance at F_i = base_i + leaves[mem_i].
            let mut d_ref = 0.0_f64;
            for i in 0..n {
                let f = f64::from(base[i]) + f64::from(leaves[usize::from(mem[i])]);
                let r = f64::from(y[i]) / f.exp();
                d_ref += f64::from(w[i]) * 2.0 * (r - 1.0 - r.ln());
            }
            // Pre-cast closed form (factored coeffs) — the algebra under test.
            let mut d_closed = coeffs.k;
            for l in 0..n_leaves {
                let v = f64::from(leaves[l]);
                d_closed += (-v).exp() * coeffs.m[l] - v * coeffs.a[l];
            }
            let rel = (d_closed - d_ref).abs() / d_ref.abs().max(1e-12);
            assert!(
                rel < 1e-10,
                "seed {seed}: closed {d_closed} vs f64 ref {d_ref}, rel {rel:e}"
            );

            // Shipped f32-cast form vs the per-row f32 `Gamma::deviance`: same f32 grid.
            let mut raw = vec![0.0_f32; n];
            for i in 0..n {
                raw[i] = (f64::from(base[i]) + f64::from(leaves[usize::from(mem[i])])) as f32;
            }
            let d_perrow = f64::from(Gamma.deviance(&y, &raw, &w).unwrap());
            let d_ship = loglink_closed_deviance(&coeffs, &leaves, n_leaves);
            let rel_f32 = (d_ship - d_perrow).abs() / d_perrow.abs().max(1e-12);
            assert!(
                rel_f32 < 1e-4,
                "seed {seed}: shipped {d_ship} vs per-row {d_perrow}, rel {rel_f32:e}"
            );
        }
    }

    /// (a2-gamma) The gamma coefficient fold through the shared chunked wrapper: thread-count
    /// invariant above the P2 threshold (fixed boundaries, chunk-order combine, exp_sign carried),
    /// bounded drift vs the serial single-fold.
    #[test]
    fn gamma_leaf_coeffs_parallel_fold_is_thread_invariant() {
        let n = LEAF_COEFF_PAR_MIN_ROWS + 1000;
        let mut s = 13u64;
        let mut y = vec![0.0_f32; n];
        let mut w = vec![0.0_f32; n];
        let mut base = vec![0.0_f32; n];
        let mut mem = vec![0u8; n];
        for i in 0..n {
            y[i] = (0.05 + xs01(&mut s) * 8.0) as f32;
            w[i] = (0.1 + xs01(&mut s) * 3.0) as f32;
            base[i] = ((xs01(&mut s) - 0.5) * 5.0) as f32;
            mem[i] = ((xs01(&mut s) * 8.0) as usize % 8) as u8;
        }
        let run = |nt: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap();
            pool.install(|| gamma_leaf_coeffs(&y, &w, &base, &mem).unwrap())
        };
        let c1 = run(1);
        let c8 = run(8);
        assert_eq!(c1.exp_sign, -1.0);
        assert_eq!(
            c1.m.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            c8.m.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(c1.k.to_bits(), c8.k.to_bits());
        let ser = gamma_leaf_coeffs_serial(&y, &w, &base, &mem).unwrap();
        let rel = |p: f64, q: f64| (p - q).abs() / q.abs().max(1e-12);
        for l in 0..8 {
            assert!(rel(c8.m[l], ser.m[l]) < 1e-9);
            assert!(rel(c8.a[l], ser.a[l]) < 1e-9);
        }
        assert!(rel(c8.k, ser.k) < 1e-9);
    }

    /// (b-gamma) Tier-1 leaf-selection identity for GAMMA: closed-form refine ON vs forced generic
    /// per-row line search must serialize bit-for-bit — same guarantee as the Poisson pair, on the
    /// strictly positive severity domain.
    #[test]
    fn gamma_closed_refine_matches_generic_leaves_bit_for_bit() {
        for (idx, (lr, steps, n_trees, weighted)) in [
            (0.3f32, 2u8, 30u32, false),
            (0.1, 4, 40, false),
            (0.5, 1, 24, false),
            (0.2, 3, 36, true),
        ]
        .into_iter()
        .enumerate()
        {
            let n = 240usize;
            let x0: Vec<f32> = (0..n).map(|i| (i % 9 + 1) as f32).collect();
            let x1: Vec<f32> = (0..n).map(|i| (i % 7 + 1) as f32).collect();
            let x2: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
            // Strictly positive severity-like target with structure (Gamma domain).
            let y: Vec<f32> = (0..n)
                .map(|i| {
                    let a = if x0[i] <= 4.0 { 0.6 } else { 1.7 };
                    let b = if x1[i] <= 3.0 { 1.4 } else { 0.9 };
                    0.05 + a * b + (i % 4) as f32 * 0.2
                })
                .collect();
            let x = binned(&[x0, x1, x2]);
            let cfg = Config {
                n_trees,
                learning_rate: lr,
                lambda: 1.0,
                leaf_refine_steps: steps,
                leaf_refine_backtracks: 4,
                refine_closed_form_tier2: false, // Tier 1: byte-identity is the claim under test
                incremental_mu: false,
                ..Config::default()
            };
            let gamma = Gamma;
            let wts: Vec<f32> = (0..n).map(|i| 0.2 + (i % 7) as f32 * 0.3).collect();
            let fit = || {
                let s = FitSpec {
                    loss: &gamma,
                    weight: if weighted { Some(&wts) } else { None },
                    exposure: None,
                    monotone: crate::constraints::MonotoneMap::new(),
                    interaction: crate::constraints::InteractionPolicy::default(),
                    credibility: crate::constraints::CredibilityFloor::default(),
                    fixed_holdout: None,
                    bag_groups: None,
                    seed: idx as u64,
                };
                crate::serialize::encode_model(
                    &Booster::with_config(cfg.clone()).fit(&x, &y, &s).unwrap(),
                )
                .unwrap()
            };
            let closed = fit();
            let generic = with_loglink_closed_disabled(fit);
            assert_eq!(
                closed, generic,
                "config #{idx} (lr={lr}, steps={steps}, weighted={weighted}): GAMMA closed-form \
                 refine diverged from the generic per-row line search"
            );
        }
    }

    /// (b2-gamma) Tier-2 leaf-selection tolerance for GAMMA (ratified by Ralph 2026-07-19 after its
    /// own multi-split no-harm battery — 6 real severity datasets × k=5 paired, held-out deviance
    /// unbiased, 23 worse/23 better, sign-test p=1.0). Under the default `refine_closed_form_tier2
    /// = true`, a gamma fit takes the closed-form Newton deltas and tracks the exact generic per-row
    /// line search to a small tolerance — NOT bit-for-bit (f32 `grad_hess` vs f64 closed-form deltas
    /// drift ~1e-7 per step and compound through boosting). Byte-identity is Tier 1's job (the
    /// `..._matches_generic_leaves_bit_for_bit` test with tier2=false); here we bound the drift of
    /// the scored raw predictions, mirroring `poisson_closed_refine_tier2_matches_generic_within_tolerance`.
    #[test]
    fn gamma_closed_refine_tier2_matches_generic_within_tolerance() {
        for (idx, (lr, steps, n_trees, weighted)) in [
            (0.3f32, 2u8, 30u32, false),
            (0.1, 4, 40, false),
            (0.2, 3, 36, true),
        ]
        .into_iter()
        .enumerate()
        {
            let n = 240usize;
            let x0: Vec<f32> = (0..n).map(|i| (i % 9 + 1) as f32).collect();
            let x1: Vec<f32> = (0..n).map(|i| (i % 7 + 1) as f32).collect();
            let x2: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
            let y: Vec<f32> = (0..n)
                .map(|i| {
                    let a = if x0[i] <= 4.0 { 0.6 } else { 1.7 };
                    let b = if x1[i] <= 3.0 { 1.4 } else { 0.9 };
                    0.05 + a * b + (i % 4) as f32 * 0.2
                })
                .collect();
            let x = binned(&[x0, x1, x2]);
            let cfg = Config {
                n_trees,
                learning_rate: lr,
                lambda: 1.0,
                leaf_refine_steps: steps,
                leaf_refine_backtracks: 4,
                refine_closed_form_tier2: true, // the shipped default (Gamma now takes Tier 2)
                ..Config::default()
            };
            let gamma = Gamma;
            let wts: Vec<f32> = (0..n).map(|i| 0.2 + (i % 7) as f32 * 0.3).collect();
            let fit = || {
                let s = FitSpec {
                    loss: &gamma,
                    weight: if weighted { Some(&wts) } else { None },
                    exposure: None,
                    monotone: crate::constraints::MonotoneMap::new(),
                    interaction: crate::constraints::InteractionPolicy::default(),
                    credibility: crate::constraints::CredibilityFloor::default(),
                    fixed_holdout: None,
                    bag_groups: None,
                    seed: idx as u64,
                };
                Booster::with_config(cfg.clone()).fit(&x, &y, &s).unwrap()
            };
            let tier2 = fit(); // production default: Tier-2 closed-form deltas
            let generic = with_loglink_closed_disabled(fit); // exact generic per-row reference
            let mut raw_t2 = vec![0.0_f32; n];
            let mut raw_gen = vec![0.0_f32; n];
            tier2.score_trees(&x, None, &mut raw_t2).unwrap();
            generic.score_trees(&x, None, &mut raw_gen).unwrap();
            let max_abs = raw_t2
                .iter()
                .zip(&raw_gen)
                .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_abs < 1e-3,
                "config #{idx} (lr={lr}, steps={steps}, weighted={weighted}): GAMMA Tier-2 closed-form \
                 refine drifted {max_abs:e} from the generic per-row path (raw score)"
            );
        }
    }

    /// (a-tweedie) The Tweedie two-exponential coefficients reproduce the exact weighted
    /// deviance: `D = K + Σ_l (e^{p1·v}·M1_l + e^{p2·v}·M2_l)` must equal the straight f64
    /// per-row fold to ≤1e-10 relative (zeros included — the compound-Poisson mass point),
    /// and the shipped f32-cast form must track `Tweedie::deviance` on the f32 grid.
    #[test]
    fn tweedie_closed_form_deviance_matches_f64_reference() {
        for (rho, seed_base) in [(1.5f32, 0u64), (1.3, 100), (1.8, 200)] {
            for seed in seed_base..seed_base + 16 {
                let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(23);
                let n = 160usize + (seed as usize % 64);
                let n_leaves = 8usize;
                let mut y = vec![0.0_f32; n];
                let mut w = vec![0.0_f32; n];
                let mut base = vec![0.0_f32; n];
                let mut mem = vec![0u8; n];
                for i in 0..n {
                    y[i] = if xs01(&mut s) < 0.3 {
                        0.0
                    } else {
                        (0.2 + xs01(&mut s) * 5.0) as f32
                    };
                    w[i] = (0.1 + xs01(&mut s) * 3.0) as f32;
                    base[i] = ((xs01(&mut s) - 0.5) * 5.0) as f32;
                    mem[i] = ((xs01(&mut s) * n_leaves as f64) as usize % n_leaves) as u8;
                }
                let leaves: [f32; 8] = std::array::from_fn(|_| ((xs01(&mut s) - 0.5) * 0.9) as f32);

                let coeffs = tweedie_leaf_coeffs(rho, &y, &w, &base, &mem).unwrap();

                let p1 = 1.0 - f64::from(rho);
                let p2 = 2.0 - f64::from(rho);
                let mut d_ref = 0.0_f64;
                for i in 0..n {
                    let f = f64::from(base[i]) + f64::from(leaves[usize::from(mem[i])]);
                    let yy = f64::from(y[i]);
                    let y_term = if yy > 0.0 { (p2 * yy.ln()).exp() } else { 0.0 };
                    let d = y_term / (p1 * p2) - yy * (p1 * f).exp() / p1 + (p2 * f).exp() / p2;
                    d_ref += f64::from(w[i]) * 2.0 * d;
                }
                let mut d_closed = coeffs.k;
                for l in 0..n_leaves {
                    let v = f64::from(leaves[l]);
                    d_closed += (p1 * v).exp() * coeffs.m1[l] + (p2 * v).exp() * coeffs.m2[l];
                }
                let rel = (d_closed - d_ref).abs() / d_ref.abs().max(1e-12);
                assert!(
                    rel < 1e-10,
                    "rho {rho} seed {seed}: closed {d_closed} vs f64 ref {d_ref}, rel {rel:e}"
                );

                let mut raw = vec![0.0_f32; n];
                for i in 0..n {
                    raw[i] = (f64::from(base[i]) + f64::from(leaves[usize::from(mem[i])])) as f32;
                }
                let tw = Tweedie::new(rho).unwrap();
                let d_perrow = f64::from(tw.deviance(&y, &raw, &w).unwrap());
                let d_ship = tweedie_closed_deviance(&coeffs, &leaves, n_leaves);
                let rel_f32 = (d_ship - d_perrow).abs() / d_perrow.abs().max(1e-12);
                assert!(
                    rel_f32 < 1e-4,
                    "rho {rho} seed {seed}: shipped {d_ship} vs per-row {d_perrow}, rel {rel_f32:e}"
                );
            }
        }
    }

    /// (b-tweedie) Tier-1 leaf-selection identity for TWEEDIE: closed-form refine ON vs the
    /// forced generic per-row line search must serialize bit-for-bit (Tweedie pins Tier 1 via
    /// the same gate as Gamma, so this also covers the tier-2 default config).
    #[test]
    fn tweedie_closed_refine_matches_generic_leaves_bit_for_bit() {
        for (idx, (lr, steps, n_trees, weighted)) in [
            (0.3f32, 2u8, 30u32, false),
            (0.1, 4, 40, false),
            (0.2, 3, 36, true),
        ]
        .into_iter()
        .enumerate()
        {
            let n = 240usize;
            let x0: Vec<f32> = (0..n).map(|i| (i % 9 + 1) as f32).collect();
            let x1: Vec<f32> = (0..n).map(|i| (i % 7 + 1) as f32).collect();
            let x2: Vec<f32> = (0..n).map(|i| (i % 5 + 1) as f32).collect();
            // Non-negative pure-premium-like target with ~20% exact zeros (Tweedie domain).
            let y: Vec<f32> = (0..n)
                .map(|i| {
                    if i % 5 == 0 {
                        0.0
                    } else {
                        let a = if x0[i] <= 4.0 { 0.4 } else { 1.3 };
                        let b = if x1[i] <= 3.0 { 1.5 } else { 0.8 };
                        a * b + (i % 4) as f32 * 0.2
                    }
                })
                .collect();
            let x = binned(&[x0, x1, x2]);
            let cfg = Config {
                n_trees,
                learning_rate: lr,
                lambda: 1.0,
                leaf_refine_steps: steps,
                leaf_refine_backtracks: 4,
                refine_closed_form_tier2: true, // shipped default; Tweedie pins Tier 1 anyway
                ..Config::default()
            };
            let tweedie = Tweedie::new(1.5).unwrap();
            let wts: Vec<f32> = (0..n).map(|i| 0.2 + (i % 7) as f32 * 0.3).collect();
            let fit = || {
                let s = FitSpec {
                    loss: &tweedie,
                    weight: if weighted { Some(&wts) } else { None },
                    exposure: None,
                    monotone: crate::constraints::MonotoneMap::new(),
                    interaction: crate::constraints::InteractionPolicy::default(),
                    credibility: crate::constraints::CredibilityFloor::default(),
                    fixed_holdout: None,
                    bag_groups: None,
                    seed: idx as u64,
                };
                crate::serialize::encode_model(
                    &Booster::with_config(cfg.clone()).fit(&x, &y, &s).unwrap(),
                )
                .unwrap()
            };
            let closed = fit();
            let generic = with_loglink_closed_disabled(fit);
            assert_eq!(
                closed, generic,
                "config #{idx} (lr={lr}, steps={steps}, weighted={weighted}): TWEEDIE closed-form \
                 refine diverged from the generic per-row line search"
            );
        }
    }

    /// (c) The exp-clamp fall-back: a base score near the ±30 clamp must make the closed form DECLINE
    /// (return `Ok(false)`, tree untouched) so the caller runs the clamped per-row path; a small base
    /// is handled (`Ok(true)`) and actually refines the leaves. Exercised directly on hand-built
    /// dense buffers + a real `GrowConfig`.
    #[test]
    fn poisson_closed_refine_declines_when_exp_clamp_could_bite() {
        let n = 64usize;
        let n_leaves = 8usize;
        let y: Vec<f32> = (0..n)
            .map(|i| if i % 4 == 0 { 0.0 } else { (i % 3 + 1) as f32 })
            .collect();
        let w: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.3).collect();
        let mem: Vec<u8> = (0..n).map(|i| (i % n_leaves) as u8).collect();
        let grow_cfg = test_grow_cfg(1.0, 0.3, Some(0.7));
        let cfg = Config {
            leaf_refine_steps: 3,
            leaf_refine_backtracks: 4,
            ..Config::default()
        };
        let poisson = Poisson;
        let mk_tree = || ObliviousTree {
            splits: vec![
                crate::engine::Split {
                    axis: 0,
                    bin_le: 0,
                    missing_left: false,
                };
                3
            ],
            leaves: vec![0.05, -0.03, 0.1, -0.08, 0.02, 0.0, -0.05, 0.07],
            depth: 3,
        };

        // Safe base ⇒ handled, and the refine moves at least one leaf.
        let base_safe = vec![0.4_f32; n];
        let mut t = mk_tree();
        let handled = refine_leaves_loglink_closed(
            &cfg, &poisson, &y, &w, &base_safe, &mem, None, &mut t, &grow_cfg, n_leaves,
        )
        .unwrap();
        assert!(handled, "safe base must be handled by the closed form");
        assert_ne!(
            t.leaves,
            mk_tree().leaves,
            "closed-form refine should have moved a leaf"
        );

        // Base near the ±30 clamp ⇒ decline; the tree is left untouched for the generic path.
        let base_hot = vec![29.9_f32; n];
        let mut t2 = mk_tree();
        let handled_hot = refine_leaves_loglink_closed(
            &cfg, &poisson, &y, &w, &base_hot, &mem, None, &mut t2, &grow_cfg, n_leaves,
        )
        .unwrap();
        assert!(
            !handled_hot,
            "hot base must decline to the generic per-row path"
        );
        assert_eq!(
            t2.leaves,
            mk_tree().leaves,
            "a declined closed-form refine must not mutate the tree"
        );
    }
}
