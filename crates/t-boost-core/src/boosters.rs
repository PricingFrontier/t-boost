//! Predictiveness boosters (spec §09).
//!
//! This module owns exactness-preserving booster levers: each lever may touch only
//! leaf scalars, tree alphas, the intercept, or convex averages of already-exact table
//! banks. The first production primitive is [`average_banks`], the §09.5 ensemble
//! on-ramp: it aligns members on a lossless union grid, averages in score space, and
//! re-runs the same purification cascade used by [`crate::engine::Model::explain`].

use crate::data::BorderGrid;
use crate::error::PbError;
use crate::explain::{
    purify_raw_effects, FeatureSet, RawEffect, RefMeasure, SeBand, TableBank, TableBudget, Tensor,
};
use std::collections::{BTreeMap, BTreeSet};

const WEIGHT_SUM_TOL: f64 = 1.0e-6;

/// Predictiveness-booster knobs. Defaults are all-off / identity (§09.1).
#[derive(Clone, Debug, PartialEq)]
pub struct BoosterConfig {
    /// Fully-corrective leaf refit configuration.
    pub refit_leaves: RefitSpec,
    /// Function-space acceleration configuration.
    pub nesterov: NesterovSpec,
    /// Exact table-bank ensemble configuration.
    pub ensemble: EnsembleSpec,
    /// Dropout-additive-regression-tree configuration.
    pub dart: Option<DartSpec>,
    /// Deterministic split-score noise strength. `0.0` is inert.
    pub random_strength: f32,
    /// Whether to fold a global mean re-anchor into the intercept.
    pub reanchor: bool,
    /// Whether to fit an honest affine link-scale recalibration `F' = b + a·F` on the fit's own
    /// internal validation holdout after early stopping, folding `a` into the tree weights and
    /// `(a, b)` into the intercept. Corrects the uniform score-scale compression that shrinkage +
    /// bulk-dominated early stopping leave behind (extremes under-separated), without touching
    /// ranking or the fANOVA decomposition. Evidence-gated: the fit is ridge-shrunk toward
    /// identity and applied only when its holdout deviance gain clears the prior penalty, so thin
    /// or uninformative holdouts stay (near-)identity. Inert without a validation holdout.
    pub reanchor_slope: bool,
}

impl Default for BoosterConfig {
    fn default() -> Self {
        Self {
            refit_leaves: RefitSpec::Off,
            nesterov: NesterovSpec::Off,
            ensemble: EnsembleSpec::Off,
            dart: None,
            random_strength: 0.0,
            reanchor: false,
            reanchor_slope: false,
        }
    }
}

impl BoosterConfig {
    /// Validate booster knobs without touching data.
    ///
    /// # Errors
    /// [`PbError::InvalidConfig`] if a non-inert knob has an invalid numeric parameter.
    pub fn validate(&self) -> Result<(), PbError> {
        self.refit_leaves.validate()?;
        self.nesterov.validate()?;
        self.ensemble.validate()?;
        if let Some(dart) = &self.dart {
            dart.validate()?;
        }
        if self.dart.is_some() && !matches!(self.nesterov, NesterovSpec::Off) {
            return Err(PbError::InvalidConfig {
                what: "DART and Nesterov/AGBM cannot be combined in the v1.5 booster pipeline"
                    .into(),
            });
        }
        if !self.random_strength.is_finite() || self.random_strength < 0.0 {
            return Err(PbError::InvalidConfig {
                what: format!(
                    "random_strength must be finite and >= 0, got {}",
                    self.random_strength
                ),
            });
        }
        Ok(())
    }
}

/// Fully-corrective leaf-refit configuration (§09.3).
#[derive(Clone, Debug, PartialEq)]
pub enum RefitSpec {
    /// Do not refit leaves after stagewise training.
    Off,
    /// Ridge-regularized IRLS over frozen tree leaf memberships.
    Ridge {
        /// L2 regularization on leaf values.
        l2: f32,
        /// Maximum Newton/IRLS passes.
        max_iter: u8,
        /// Refit every `k` trees when set; otherwise refit once at the end.
        every_k_trees: Option<u32>,
    },
}

impl RefitSpec {
    fn validate(&self) -> Result<(), PbError> {
        match self {
            RefitSpec::Off => Ok(()),
            RefitSpec::Ridge {
                l2,
                max_iter,
                every_k_trees,
            } => {
                if !l2.is_finite() || *l2 < 0.0 {
                    return Err(PbError::InvalidConfig {
                        what: format!("refit l2 must be finite and >= 0, got {l2}"),
                    });
                }
                if *max_iter == 0 {
                    return Err(PbError::InvalidConfig {
                        what: "refit max_iter must be > 0".into(),
                    });
                }
                if matches!(every_k_trees, Some(0)) {
                    return Err(PbError::InvalidConfig {
                        what: "refit every_k_trees must be > 0 when set".into(),
                    });
                }
                Ok(())
            }
        }
    }
}

/// Nesterov/accelerated-GBM configuration (§09.4).
#[derive(Clone, Debug, PartialEq)]
pub enum NesterovSpec {
    /// Plain MART boosting.
    Off,
    /// AGBM-style look-ahead boosting with optional momentum correction.
    ///
    /// Lookahead and round-boundary scores are reconstructed from the stored leaves
    /// and coefficients in f64, then rounded once. This costs O(n·T) per round and
    /// preserves small updates at large intercepts; fitted numbers can differ from
    /// the former incremental f32 score-cache implementation.
    Agbm {
        /// Whether to fit the second correction tree per accelerated step.
        momentum_correction: bool,
    },
}

impl NesterovSpec {
    fn validate(&self) -> Result<(), PbError> {
        Ok(())
    }
}

/// §G1 cell-basis refit configuration: an adaptive ridge on the purified fANOVA *cell*
/// basis, fit to the bagged out-of-bag residual and attached to the bagged soup as an
/// exact correction. Suite-wide params (no per-dataset tuning); `None` disables it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellRefit {
    /// Base ridge penalty `α` on the cell coefficients.
    pub base: f64,
    /// Adaptive exponent `γ` (`0` = flat ridge; high-signal terms penalised less).
    pub gamma: f64,
}

/// Exact ensemble-averaging configuration (§09.5).
#[derive(Clone, Debug, PartialEq)]
pub enum EnsembleSpec {
    /// Do not average multiple banks.
    Off,
    /// Train one hyperparameter setting across independent bags and average banks.
    OuterBag {
        /// Number of bags to train and average.
        n_bags: u16,
        /// Per-bag row sampling fraction. `>= 1.0` ⇒ full-size bootstrap WITH replacement (classic
        /// bagging). `0 < f < 1.0` ⇒ without-replacement subsample of `round(f * n_rows)` rows
        /// (subagging): faster per bag, more cross-bag diversity, and no within-bag train/val leak.
        bag_subsample: f32,
        /// §G1 cell-basis refit on the bagged soup, fit to the OOB residual (`None` = off).
        /// Requires real bagging (`n_bags >= 2`) so an out-of-bag residual exists.
        cell_refit: Option<CellRefit>,
    },
    /// Hyperparameter-diverse library with bagged greedy selection.
    GreedySelect {
        /// Number of library members.
        library_size: u16,
        /// Hyperparameter grid used to generate library members.
        hp_grid: HpGrid,
        /// Number of bootstrap replicates used by selection.
        selection_bags: u16,
        /// Number of top single models used to seed greedy selection.
        seed_top_n: u8,
    },
}

impl EnsembleSpec {
    fn validate(&self) -> Result<(), PbError> {
        match self {
            EnsembleSpec::Off => Ok(()),
            EnsembleSpec::OuterBag {
                n_bags,
                bag_subsample,
                cell_refit,
            } => {
                if *n_bags == 0 {
                    return Err(PbError::InvalidConfig {
                        what: "OuterBag n_bags must be > 0".into(),
                    });
                }
                if !(bag_subsample.is_finite() && *bag_subsample > 0.0) {
                    return Err(PbError::InvalidConfig {
                        what: "OuterBag bag_subsample must be finite and > 0".into(),
                    });
                }
                if let Some(cr) = cell_refit {
                    if !(cr.base.is_finite() && cr.base > 0.0) {
                        return Err(PbError::InvalidConfig {
                            what: "OuterBag cell_refit base must be finite and > 0".into(),
                        });
                    }
                    if !(cr.gamma.is_finite() && cr.gamma >= 0.0) {
                        return Err(PbError::InvalidConfig {
                            what: "OuterBag cell_refit gamma must be finite and >= 0".into(),
                        });
                    }
                    if *n_bags < 2 {
                        return Err(PbError::InvalidConfig {
                            what:
                                "OuterBag cell_refit requires n_bags >= 2 (needs an OOB residual)"
                                    .into(),
                        });
                    }
                }
                Ok(())
            }
            EnsembleSpec::GreedySelect {
                library_size,
                hp_grid,
                selection_bags,
                seed_top_n,
            } => {
                if *library_size == 0 {
                    return Err(PbError::InvalidConfig {
                        what: "GreedySelect library_size must be > 0".into(),
                    });
                }
                if *selection_bags == 0 {
                    return Err(PbError::InvalidConfig {
                        what: "GreedySelect selection_bags must be > 0".into(),
                    });
                }
                if *seed_top_n == 0 || u16::from(*seed_top_n) > *library_size {
                    return Err(PbError::InvalidConfig {
                        what: "GreedySelect seed_top_n must be in 1..=library_size".into(),
                    });
                }
                hp_grid.validate()
            }
        }
    }
}

/// Hyperparameter grid for §09.5 greedy ensemble selection.
#[derive(Clone, Debug, PartialEq)]
pub struct HpGrid {
    /// Candidate `max_bin` values.
    pub max_bins: Vec<u16>,
    /// Candidate L2 leaf regularizers.
    pub lambdas: Vec<f32>,
    /// Candidate learning rates.
    pub learning_rates: Vec<f32>,
    /// Candidate boosting-round counts.
    pub n_trees: Vec<u32>,
    /// Candidate maximum interaction orders.
    pub max_interaction_orders: Vec<u8>,
    /// Candidate deterministic split-score noise strengths.
    pub random_strengths: Vec<f32>,
}

impl Default for HpGrid {
    fn default() -> Self {
        Self {
            max_bins: vec![64, 128, 254],
            lambdas: vec![0.0, 1.0, 10.0],
            learning_rates: vec![0.03, 0.05, 0.1],
            n_trees: vec![250, 500, 1000],
            max_interaction_orders: vec![2, 3],
            random_strengths: vec![0.0],
        }
    }
}

impl HpGrid {
    fn validate(&self) -> Result<(), PbError> {
        if self.max_bins.is_empty()
            || self.lambdas.is_empty()
            || self.learning_rates.is_empty()
            || self.n_trees.is_empty()
            || self.max_interaction_orders.is_empty()
            || self.random_strengths.is_empty()
        {
            return Err(PbError::InvalidConfig {
                what: "HpGrid candidate lists must be non-empty".into(),
            });
        }
        for &max_bin in &self.max_bins {
            if !(2..=254).contains(&max_bin) {
                return Err(PbError::InvalidConfig {
                    what: format!("HpGrid max_bin must be in 2..=254, got {max_bin}"),
                });
            }
        }
        for &lambda in &self.lambdas {
            if !lambda.is_finite() || lambda < 0.0 {
                return Err(PbError::InvalidConfig {
                    what: format!("HpGrid lambda must be finite and >= 0, got {lambda}"),
                });
            }
        }
        for &learning_rate in &self.learning_rates {
            if !learning_rate.is_finite() || learning_rate <= 0.0 {
                return Err(PbError::InvalidConfig {
                    what: format!(
                        "HpGrid learning_rate must be finite and > 0, got {learning_rate}"
                    ),
                });
            }
        }
        for &n_trees in &self.n_trees {
            if n_trees == 0 {
                return Err(PbError::InvalidConfig {
                    what: "HpGrid n_trees must be > 0".into(),
                });
            }
        }
        for &order in &self.max_interaction_orders {
            if !(1..=crate::engine::MAX_ORDER).contains(&usize::from(order)) {
                return Err(PbError::InvalidConfig {
                    what: format!(
                        "HpGrid max_interaction_order must be in 1..={}, got {order}",
                        crate::engine::MAX_ORDER
                    ),
                });
            }
        }
        for &strength in &self.random_strengths {
            if !strength.is_finite() || strength < 0.0 {
                return Err(PbError::InvalidConfig {
                    what: format!("HpGrid random_strength must be finite and >= 0, got {strength}"),
                });
            }
        }
        Ok(())
    }
}

/// DART tree-dropout configuration (§09.6).
///
/// Dropout and round-boundary scores are reconstructed from retained trees in f64,
/// then rounded once. This costs O(n·T) per round and preserves small updates at
/// large intercepts; fitted numbers can differ from the former incremental score cache.
#[derive(Clone, Debug, PartialEq)]
pub struct DartSpec {
    /// Probability of dropping a prior tree for the current round.
    pub drop_rate: f32,
    /// Whether to fold standard DART normalization into tree alphas.
    pub normalize: bool,
}

impl DartSpec {
    fn validate(&self) -> Result<(), PbError> {
        if !self.drop_rate.is_finite() || !(0.0..1.0).contains(&self.drop_rate) {
            return Err(PbError::InvalidConfig {
                what: format!(
                    "DART drop_rate must be finite and in [0, 1), got {}",
                    self.drop_rate
                ),
            });
        }
        Ok(())
    }
}

struct AccumEffect {
    values: Tensor,
    support: Tensor,
}

fn walk_extents(
    extents: &[usize],
    mut f: impl FnMut(&[usize]) -> Result<(), PbError>,
) -> Result<(), PbError> {
    let mut idx = vec![0usize; extents.len()];
    loop {
        f(&idx)?;
        let mut k = extents.len();
        loop {
            if k == 0 {
                return Ok(());
            }
            k -= 1;
            let e = *extents.get(k).ok_or_else(|| PbError::Internal {
                what: "odometer axis escaped extents".into(),
            })?;
            let v = idx.get_mut(k).ok_or_else(|| PbError::Internal {
                what: "odometer index escaped buffer".into(),
            })?;
            *v += 1;
            if *v < e {
                break;
            }
            *v = 0;
        }
    }
}

fn validate_grid(grid: &BorderGrid, raw: usize) -> Result<(), PbError> {
    if grid.missing_bin != 0 {
        return Err(PbError::InvalidInput {
            what: format!("bank grid {raw} missing_bin must be 0"),
        });
    }
    let expected = grid
        .borders
        .len()
        .checked_add(2)
        .ok_or_else(|| PbError::Internal {
            what: "bank grid border count overflow".into(),
        })?;
    if usize::from(grid.n_bins) != expected {
        return Err(PbError::InvalidInput {
            what: format!(
                "bank grid {raw} n_bins {} inconsistent with {} borders",
                grid.n_bins,
                grid.borders.len()
            ),
        });
    }
    for (i, &border) in grid.borders.iter().enumerate() {
        if !border.is_finite() {
            return Err(PbError::InvalidInput {
                what: format!("bank grid {raw} border {i} must be finite"),
            });
        }
    }
    for pair in grid.borders.windows(2) {
        if let [a, b] = pair {
            if a >= b {
                return Err(PbError::InvalidInput {
                    what: format!("bank grid {raw} borders must be strictly ascending"),
                });
            }
        }
    }
    Ok(())
}

fn validate_bank(
    bank: &TableBank,
    w: &RefMeasure,
    expected_features: usize,
) -> Result<(), PbError> {
    if bank.reference_measure() != w {
        return Err(PbError::InvalidConfig {
            what: "average_banks requires every member to use the requested RefMeasure".into(),
        });
    }
    if bank.merged_grids.len() != expected_features {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "member has {} merged grids, expected {expected_features}",
                bank.merged_grids.len()
            ),
        });
    }
    if !bank.f0.is_finite() {
        return Err(PbError::InvalidInput {
            what: "member bank f0 must be finite".into(),
        });
    }
    for (raw, grid) in bank.merged_grids.iter().enumerate() {
        validate_grid(grid, raw)?;
    }
    for table in &bank.tables {
        if !(1..=crate::engine::MAX_ORDER).contains(&table.u.order()) {
            return Err(PbError::InvalidInput {
                what: format!(
                    "effect order {} outside 1..={}",
                    table.u.order(),
                    crate::engine::MAX_ORDER
                ),
            });
        }
        if table.axes.len() != table.u.order() {
            return Err(PbError::ShapeMismatch {
                what: "effect axes length does not match support order".into(),
            });
        }
        for (axis, raw) in table.axes.iter().zip(&table.u.0) {
            if axis.raw != *raw {
                return Err(PbError::ShapeMismatch {
                    what: "effect axes are not parallel to support raw ids".into(),
                });
            }
        }
        let extents = table_extents(&bank.merged_grids, &table.u)?;
        if table.values.shape() != extents || table.support.shape() != extents {
            return Err(PbError::ShapeMismatch {
                what: "effect tensor shape does not match its bank merged grid".into(),
            });
        }
        for &value in table.values.values().iter() {
            if !value.is_finite() {
                return Err(PbError::InvalidInput {
                    what: "effect table value must be finite".into(),
                });
            }
        }
        for &support in table.support.values().iter() {
            if !support.is_finite() || support < 0.0 {
                return Err(PbError::InvalidInput {
                    what: "effect table support must be finite and >= 0".into(),
                });
            }
        }
    }
    Ok(())
}

fn table_extents(grids: &[BorderGrid], u: &FeatureSet) -> Result<Vec<usize>, PbError> {
    let mut extents = Vec::with_capacity(u.order());
    for raw in &u.0 {
        let grid = grids
            .get(raw.0 as usize)
            .ok_or_else(|| PbError::ShapeMismatch {
                what: format!("merged grid missing raw feature {}", raw.0),
            })?;
        extents.push(usize::from(grid.n_bins));
    }
    Ok(extents)
}

fn product_cells(extents: &[usize]) -> Result<u64, PbError> {
    let mut acc = 1u64;
    for &extent in extents {
        let e = u64::try_from(extent).map_err(|_| PbError::Internal {
            what: "tensor extent exceeds u64".into(),
        })?;
        acc = acc.checked_mul(e).ok_or_else(|| PbError::Internal {
            what: "tensor cell count overflow".into(),
        })?;
    }
    Ok(acc)
}

fn union_grids(members: &[(f32, TableBank)]) -> Result<Vec<BorderGrid>, PbError> {
    let first = members.first().ok_or_else(|| PbError::InvalidConfig {
        what: "average_banks requires at least one member".into(),
    })?;
    let n_features = first.1.merged_grids.len();
    let mut out = Vec::with_capacity(n_features);
    for raw in 0..n_features {
        let mut borders = Vec::new();
        for (_, bank) in members {
            let grid = bank
                .merged_grids
                .get(raw)
                .ok_or_else(|| PbError::ShapeMismatch {
                    what: format!("member missing merged grid {raw}"),
                })?;
            borders.extend(grid.borders.iter().copied());
        }
        borders.sort_by(|a, b| a.total_cmp(b));
        borders.dedup_by(|a, b| *a == *b);
        let n_bins =
            u16::try_from(
                borders
                    .len()
                    .checked_add(2)
                    .ok_or_else(|| PbError::Internal {
                        what: "union grid border count overflow".into(),
                    })?,
            )
            .map_err(|_| PbError::TableBudget {
                what: format!("union grid {raw}"),
                cells: borders.len() as u64 + 2,
                budget: u64::from(u16::MAX),
            })?;
        out.push(BorderGrid {
            borders,
            n_bins,
            missing_bin: 0,
        });
    }
    Ok(out)
}

fn map_union_cell_to_source(
    union_grid: &BorderGrid,
    source_grid: &BorderGrid,
    cell: usize,
) -> Result<usize, PbError> {
    match cell {
        0 => Ok(0),
        1 => Ok(1),
        _ => {
            let lower_idx = cell.checked_sub(2).ok_or_else(|| PbError::Internal {
                what: "union cell underflow".into(),
            })?;
            let lower = *union_grid
                .borders
                .get(lower_idx)
                .ok_or_else(|| PbError::Internal {
                    what: "union cell escaped border list".into(),
                })?;
            let below_or_equal = source_grid.borders.iter().filter(|&&b| b <= lower).count();
            let mapped = below_or_equal
                .checked_add(1)
                .ok_or_else(|| PbError::Internal {
                    what: "source cell overflow".into(),
                })?;
            if mapped >= usize::from(source_grid.n_bins) {
                return Err(PbError::Internal {
                    what: "mapped source cell escaped grid".into(),
                });
            }
            Ok(mapped)
        }
    }
}

/// Per source cell, how many union cells map to it (the refinement fanout) — used to
/// split a member's per-cell support count across the union sub-cells it covers, so the
/// averaged-bank support conserves the source count instead of multiplying it.
fn source_fanout(union_grid: &BorderGrid, source_grid: &BorderGrid) -> Result<Vec<usize>, PbError> {
    let mut fanout = vec![0usize; usize::from(source_grid.n_bins)];
    for cell in 0..usize::from(union_grid.n_bins) {
        let s = map_union_cell_to_source(union_grid, source_grid, cell)?;
        *fanout.get_mut(s).ok_or_else(|| PbError::Internal {
            what: "fanout source cell escaped".into(),
        })? += 1;
    }
    Ok(fanout)
}

/// Per-axis [`source_fanout`] for every axis of support `u`, one `Vec<usize>` (indexed
/// by source cell) per axis. Shared by every accumulation that projects a member bank
/// onto a union grid ([`average_banks`]'s main pass and [`mapped_raw_effects`]) so they
/// cannot silently diverge on how a source cell's support is split again.
fn support_fanouts(
    union_grids: &[BorderGrid],
    source_grids: &[BorderGrid],
    u: &FeatureSet,
) -> Result<Vec<Vec<usize>>, PbError> {
    let mut fanouts = Vec::with_capacity(u.order());
    for raw in &u.0 {
        let union_grid = union_grids
            .get(raw.0 as usize)
            .ok_or_else(|| PbError::Internal {
                what: "union grid missing raw feature for fanout".into(),
            })?;
        let source_grid =
            source_grids
                .get(raw.0 as usize)
                .ok_or_else(|| PbError::ShapeMismatch {
                    what: format!("source bank missing raw feature {} for fanout", raw.0),
                })?;
        fanouts.push(source_fanout(union_grid, source_grid)?);
    }
    Ok(fanouts)
}

/// `Π` over axes of the per-axis fanout at source coordinate `src` (from
/// [`support_fanouts`]) — the number of union sub-cells jointly covering this source
/// cell, i.e. the divisor a per-cell support count must be SPLIT by (never replicated)
/// when projected onto the union grid, so `Σ` support is conserved instead of multiplied
/// by the refinement fanout (this is what distorts the `ProductMarginals` reference
/// measure derived from support, `build_weights_from_effect_support`, when gotten wrong).
fn fanout_at(fanouts: &[Vec<usize>], src: &[usize]) -> Result<f64, PbError> {
    let mut k = 1usize;
    for (pos, &s) in src.iter().enumerate() {
        let f = fanouts
            .get(pos)
            .and_then(|fv| fv.get(s))
            .copied()
            .ok_or_else(|| PbError::Internal {
                what: "fanout lookup escaped".into(),
            })?;
        k = k.saturating_mul(f);
    }
    Ok(k.max(1) as f64)
}

fn source_coord(
    union_grids: &[BorderGrid],
    source_grids: &[BorderGrid],
    u: &FeatureSet,
    target_coord: &[usize],
) -> Result<Vec<usize>, PbError> {
    let mut coord = Vec::with_capacity(u.order());
    for (pos, raw) in u.0.iter().enumerate() {
        let cell = *target_coord.get(pos).ok_or_else(|| PbError::Internal {
            what: "target coordinate shorter than support".into(),
        })?;
        let union_grid = union_grids
            .get(raw.0 as usize)
            .ok_or_else(|| PbError::Internal {
                what: "union grid missing raw feature".into(),
            })?;
        let source_grid =
            source_grids
                .get(raw.0 as usize)
                .ok_or_else(|| PbError::ShapeMismatch {
                    what: format!("source bank missing raw feature {}", raw.0),
                })?;
        coord.push(map_union_cell_to_source(union_grid, source_grid, cell)?);
    }
    Ok(coord)
}

fn collect_supports(members: &[(f32, TableBank)]) -> BTreeSet<FeatureSet> {
    let mut supports = BTreeSet::new();
    for (_, bank) in members {
        for table in &bank.tables {
            supports.insert(table.u.clone());
        }
    }
    supports
}

fn mapped_raw_effects(
    bank: &TableBank,
    union_grids: &[BorderGrid],
    scale: f64,
) -> Result<Vec<RawEffect>, PbError> {
    let mut out = Vec::with_capacity(bank.tables.len());
    for table in &bank.tables {
        let extents = table_extents(union_grids, &table.u)?;
        let mut values = Tensor::try_zeros(extents.clone())?;
        let mut support = Tensor::try_zeros(extents.clone())?;
        // Same fanout SPLIT as average_banks' main accumulation (source_fanout /
        // support_fanouts doc): a source cell's support is a ROW COUNT and must be
        // divided across the union sub-cells it covers, not replicated into each, or the
        // ProductMarginals reference measure this bank (and attach_se_bands' per-member
        // purification) derives from support is distorted by the refinement fanout.
        let fanouts = support_fanouts(union_grids, &bank.merged_grids, &table.u)?;
        walk_extents(&extents, |target_coord| {
            let src = source_coord(union_grids, &bank.merged_grids, &table.u, target_coord)?;
            let value = table.values.at(&src).ok_or_else(|| PbError::Internal {
                what: "member value coordinate out of range".into(),
            })?;
            let count = table.support.at(&src).ok_or_else(|| PbError::Internal {
                what: "member support coordinate out of range".into(),
            })?;
            let k = fanout_at(&fanouts, &src)?;
            values.add(target_coord, scale * value)?;
            support.add(target_coord, scale * count / k)?;
            Ok(())
        })?;
        out.push(RawEffect {
            u: table.u.clone(),
            values,
            support,
        });
    }
    Ok(out)
}

fn table_value(bank: &TableBank, u: &FeatureSet, coord: &[usize]) -> Result<f64, PbError> {
    let Some(table) = bank.tables.iter().find(|table| &table.u == u) else {
        return Ok(0.0);
    };
    table.values.at(coord).ok_or_else(|| PbError::Internal {
        what: "SE-band coordinate out of range".into(),
    })
}

fn attach_se_bands(
    bank: &mut TableBank,
    members: &[(f32, TableBank)],
    union_grids: &[BorderGrid],
    w: &RefMeasure,
) -> Result<(), PbError> {
    let positive: Vec<(f64, &TableBank)> = members
        .iter()
        .filter_map(|(alpha, member)| {
            if *alpha > 0.0 {
                Some((f64::from(*alpha), member))
            } else {
                None
            }
        })
        .collect();
    if positive.len() < 2 {
        return Ok(());
    }
    let mut projected = Vec::with_capacity(positive.len());
    for (alpha, member) in &positive {
        let effects = mapped_raw_effects(member, union_grids, 1.0)?;
        let member_bank = purify_raw_effects(member.f0, union_grids.to_vec(), w.clone(), effects)?;
        projected.push((*alpha, member_bank));
    }
    let sum_alpha_sq: f64 = projected.iter().map(|(alpha, _)| alpha * alpha).sum();
    let denom = 1.0 - sum_alpha_sq;
    if denom <= 0.0 {
        return Ok(());
    }
    for table in &mut bank.tables {
        let extents = table.values.shape();
        let mut band = Tensor::try_zeros(extents.clone())?;
        walk_extents(&extents, |coord| {
            let mean = table.values.at(coord).ok_or_else(|| PbError::Internal {
                what: "SE-band mean coordinate out of range".into(),
            })?;
            let mut ss = 0.0_f64;
            for (alpha, member_bank) in &projected {
                let value = table_value(member_bank, &table.u, coord)?;
                let delta = value - mean;
                ss += alpha * delta * delta;
            }
            let sample_var = (ss / denom).max(0.0);
            let se = sample_var.sqrt() * sum_alpha_sq.sqrt();
            band.set(coord, se)
        })?;
        table.se_band = Some(SeBand { per_cell: band });
    }
    Ok(())
}

/// Average exact table banks on their lossless union grid (§09.5).
///
/// The operation is intentionally narrow: all member banks must already be purified
/// under the same [`RefMeasure`], weights must form a convex combination, and averaging
/// happens in raw score space before one final purification pass. Missing supports are
/// treated as zero tensors, so the output bank carries the union of realized supports.
/// For [`RefMeasure::ProductMarginals`], union-grid marginals are derived from the
/// member banks' cached support tensors; callers that need full-data empirical
/// marginals on a newly refined union grid should construct member banks from that
/// common support source before averaging.
///
/// `budget` bounds the union-grid tables this call allocates; `None` uses
/// [`TableBudget::default`]. The union grid's tables are never smaller than the members'
/// own — a member built densely under an enlarged `max_table_cells` needs that same (or
/// larger) budget passed here too, or its union-grid table can still overflow the
/// default even though every member held it densely.
///
/// # Errors
/// [`PbError::InvalidConfig`] for empty input, non-convex weights, reference-measure
/// mismatch, or v1-unsupported [`RefMeasure::Joint`]; [`PbError::ShapeMismatch`] for
/// incompatible bank shapes; [`PbError::TableBudget`] when the dense union grid would
/// breach `budget` (or the default, if `None`); plus propagated tensor allocation/shape
/// errors.
pub fn average_banks(
    members: &[(f32, TableBank)],
    w: &RefMeasure,
    budget: Option<TableBudget>,
) -> Result<TableBank, PbError> {
    if members.is_empty() {
        return Err(PbError::InvalidConfig {
            what: "average_banks requires at least one member".into(),
        });
    }
    if matches!(w, RefMeasure::Joint) {
        return Err(PbError::InvalidConfig {
            what: "Joint reference measure is a v1.5 fork; v1 supports ProductMarginals/Uniform"
                .into(),
        });
    }

    let mut alpha_sum = 0.0_f64;
    for (idx, (alpha, _)) in members.iter().enumerate() {
        if !alpha.is_finite() || *alpha < 0.0 {
            return Err(PbError::InvalidConfig {
                what: format!("member {idx} alpha must be finite and >= 0, got {alpha}"),
            });
        }
        alpha_sum += f64::from(*alpha);
    }
    if (alpha_sum - 1.0).abs() > WEIGHT_SUM_TOL {
        return Err(PbError::InvalidConfig {
            what: format!("member alphas must sum to 1.0, got {alpha_sum}"),
        });
    }
    // §08.10: a member with factored high-order tables would have its order-3 effect
    // SILENTLY DROPPED by the tables-only accumulation below. Reject it loudly until
    // factored-aware averaging exists (Stage 5), mirroring to_rating_export/recompute_under.
    for (idx, (_, bank)) in members.iter().enumerate() {
        if !bank.factored.is_empty() {
            return Err(PbError::InvalidConfig {
                what: format!(
                    "average_banks member {idx} carries factored high-order tables; \
                     factored-aware averaging is not yet implemented (Stage 5) — build members \
                     with OverflowPolicy::Error or a larger max_table_cells, and pass that same \
                     budget as average_banks' `budget` argument"
                ),
            });
        }
    }

    let n_features = members
        .first()
        .ok_or_else(|| PbError::InvalidConfig {
            what: "average_banks requires at least one member".into(),
        })?
        .1
        .merged_grids
        .len();
    for (_, bank) in members {
        validate_bank(bank, w, n_features)?;
    }

    let union_grids = union_grids(members)?;
    for (raw, grid) in union_grids.iter().enumerate() {
        validate_grid(grid, raw)?;
    }

    let budget = budget.unwrap_or_default();
    let supports = collect_supports(members);
    let mut bank_cells = 0u64;
    let mut effects: BTreeMap<FeatureSet, AccumEffect> = BTreeMap::new();
    for u in supports {
        let extents = table_extents(&union_grids, &u)?;
        let table_cells = product_cells(&extents)?;
        if table_cells > budget.max_table_cells {
            return Err(PbError::TableBudget {
                what: format!("average_banks table {u:?}"),
                cells: table_cells,
                budget: budget.max_table_cells,
            });
        }
        bank_cells = bank_cells
            .checked_add(table_cells)
            .ok_or_else(|| PbError::Internal {
                what: "average_banks bank cell count overflow".into(),
            })?;
        if bank_cells > budget.max_bank_cells {
            return Err(PbError::TableBudget {
                what: "average_banks bank".into(),
                cells: bank_cells,
                budget: budget.max_bank_cells,
            });
        }
        effects.insert(
            u,
            AccumEffect {
                values: Tensor::try_zeros(extents.clone())?,
                support: Tensor::try_zeros(extents)?,
            },
        );
    }

    let mut f0 = 0.0_f64;
    for (alpha, bank) in members {
        let alpha64 = f64::from(*alpha);
        f0 += alpha64 * bank.f0;
        for table in &bank.tables {
            let target = effects.get_mut(&table.u).ok_or_else(|| PbError::Internal {
                what: "average_banks support missing from accumulator".into(),
            })?;
            let extents = target.values.shape();
            // The source support is a ROW COUNT, so it must be SPLIT across the union
            // sub-cells it covers (Σ conserved), not replicated into each — replication
            // would multiply the realized support by the fanout and distort the
            // ProductMarginals reference measure derived from it
            // (build_weights_from_effect_support). See `support_fanouts`/`fanout_at`,
            // shared with `mapped_raw_effects` so the two accumulations cannot diverge.
            let fanouts = support_fanouts(&union_grids, &bank.merged_grids, &table.u)?;
            walk_extents(&extents, |target_coord| {
                let src = source_coord(&union_grids, &bank.merged_grids, &table.u, target_coord)?;
                let value = table.values.at(&src).ok_or_else(|| PbError::Internal {
                    what: "member value coordinate out of range".into(),
                })?;
                let support = table.support.at(&src).ok_or_else(|| PbError::Internal {
                    what: "member support coordinate out of range".into(),
                })?;
                let k = fanout_at(&fanouts, &src)?;
                // The cell VALUE is constant on the refinement (correct to replicate);
                // the SUPPORT count is split evenly across the K covering union sub-cells.
                target.values.add(target_coord, alpha64 * value)?;
                target.support.add(target_coord, alpha64 * support / k)?;
                Ok(())
            })?;
        }
    }

    let raw_effects = effects
        .into_iter()
        .map(|(u, effect)| RawEffect {
            u,
            values: effect.values,
            support: effect.support,
        })
        .collect();
    let mut bank = purify_raw_effects(f0, union_grids.clone(), w.clone(), raw_effects)?;
    attach_se_bands(&mut bank, members, &union_grids, w)?;
    Ok(bank)
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
    use crate::data::FeatureId;
    use crate::explain::{fixture_model, fixture_serve};

    fn table(bank: &TableBank, u: &[u32]) -> crate::explain::EffectTable {
        let key = FeatureSet::new(u);
        bank.tables
            .iter()
            .find(|t| t.u == key)
            .expect("table")
            .clone()
    }

    #[test]
    fn inert_config_validates() {
        BoosterConfig::default().validate().unwrap();
    }

    #[test]
    fn booster_config_rejects_invalid_knobs() {
        let mut cfg = BoosterConfig {
            random_strength: -1.0,
            ..BoosterConfig::default()
        };
        assert!(matches!(cfg.validate(), Err(PbError::InvalidConfig { .. })));
        cfg.random_strength = 0.0;
        cfg.ensemble = EnsembleSpec::OuterBag {
            n_bags: 0,
            bag_subsample: 1.0,
            cell_refit: None,
        };
        assert!(matches!(cfg.validate(), Err(PbError::InvalidConfig { .. })));
        cfg.ensemble = EnsembleSpec::Off;
        cfg.dart = Some(DartSpec {
            drop_rate: 0.1,
            normalize: true,
        });
        cfg.nesterov = NesterovSpec::Agbm {
            momentum_correction: false,
        };
        assert!(matches!(cfg.validate(), Err(PbError::InvalidConfig { .. })));
    }

    #[test]
    fn average_identical_banks_is_identity() {
        let model = fixture_model();
        let x = fixture_serve();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        let avg = average_banks(
            &[(0.25, bank.clone()), (0.75, bank.clone())],
            &RefMeasure::Uniform,
            None,
        )
        .unwrap();

        assert!((avg.f0 - bank.f0).abs() < 1.0e-10);
        for c0 in 0..3 {
            for c1 in 0..3 {
                let cells = [c0, c1];
                let got = avg.score(&cells).unwrap();
                let want = bank.score(&cells).unwrap();
                assert!((got - want).abs() < 1.0e-10, "{cells:?}: {got} vs {want}");
            }
        }
        assert_eq!(avg.reference_measure(), &RefMeasure::Uniform);
        for table in &avg.tables {
            let band = table
                .se_band
                .as_ref()
                .expect("averaged banks carry SE bands");
            assert!(band.per_cell.values().iter().all(|se| se.abs() < 1.0e-12));
        }
    }

    #[test]
    fn average_identical_product_marginal_banks_is_identity() {
        let model = fixture_model();
        let x = fixture_serve();
        let bank = model.explain(&x, RefMeasure::default()).unwrap();
        let avg = average_banks(
            &[(0.4, bank.clone()), (0.6, bank.clone())],
            &RefMeasure::default(),
            None,
        )
        .unwrap();

        assert_eq!(avg.reference_measure(), &RefMeasure::default());
        for c0 in 0..3 {
            for c1 in 0..3 {
                let cells = [c0, c1];
                let got = avg.score(&cells).unwrap();
                let want = bank.score(&cells).unwrap();
                assert!((got - want).abs() < 1.0e-6, "{cells:?}: {got} vs {want}");
            }
        }
    }

    fn fit_main_effect_bank(vals: &[f32]) -> TableBank {
        use crate::constraints::{CredibilityFloor, InteractionPolicy, MonotoneMap};
        use crate::data::{bin_columns, BinConfig, ServeBinnedMatrix};
        use crate::engine::{Booster, Config, FitSpec};
        use crate::explain::RefMeasure;
        use crate::loss::SquaredError;
        let cols: Vec<&[f32]> = vec![vals];
        let x = bin_columns(&cols, None, &BinConfig::default(), 0).unwrap();
        let sqe = SquaredError;
        let spec = FitSpec {
            loss: &sqe,
            weight: None,
            exposure: None,
            monotone: MonotoneMap::new(),
            interaction: InteractionPolicy::default(),
            credibility: CredibilityFloor::default(),
            fixed_holdout: None,
            bag_groups: None,
            seed: 0,
        };
        let model = Booster::with_config(Config {
            lambda_scale_invariant: false,
            max_delta_step_gated: crate::engine::GatedStepPolicy::Objective,
            n_trees: 15,
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
        })
        .fit(&x, vals, &spec)
        .unwrap();
        model
            .explain(&ServeBinnedMatrix(x), RefMeasure::default())
            .unwrap()
    }

    #[test]
    fn average_banks_conserves_support_across_differing_grids() {
        // Two single-feature models whose realized borders DIFFER, so the averaged union
        // grid refines each member. A member's per-cell support is a ROW COUNT and must be
        // SPLIT across the union sub-cells it covers (Σ conserved), not replicated.
        let a: Vec<f32> = (1..=10).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=10).map(|i| (i * 2) as f32).collect();
        let ba = fit_main_effect_bank(&a);
        let bb = fit_main_effect_bank(&b);
        assert_ne!(
            ba.merged_grids, bb.merged_grids,
            "members must differ in grid for this test to exercise the refinement split"
        );
        let avg = average_banks(
            &[(0.5, ba.clone()), (0.5, bb.clone())],
            &RefMeasure::default(),
            None,
        )
        .unwrap();

        // Total support over the main-effect table is conserved: each member's table sums
        // to its row count (10), so the 0.5/0.5 average sums to 10 — NOT 10·fanout.
        let member_total = |bank: &TableBank| -> f64 {
            bank.tables
                .iter()
                .find(|t| t.u.order() == 1)
                .map(|t| t.support.values().iter().sum::<f64>())
                .unwrap_or(0.0)
        };
        let want = 0.5 * member_total(&ba) + 0.5 * member_total(&bb);
        let got: f64 = avg
            .tables
            .iter()
            .find(|t| t.u.order() == 1)
            .unwrap()
            .support
            .values()
            .iter()
            .sum();
        assert!(
            (got - want).abs() < 1e-6,
            "averaged support {got} should conserve the member total {want} (replication would inflate it)"
        );
    }

    #[test]
    fn average_rejects_nonconvex_or_mismatched_members() {
        let model = fixture_model();
        let x = fixture_serve();
        let bank = model.explain(&x, RefMeasure::Uniform).unwrap();
        assert!(matches!(
            average_banks(&[(0.5, bank.clone())], &RefMeasure::Uniform, None),
            Err(PbError::InvalidConfig { .. })
        ));
        assert!(matches!(
            average_banks(
                &[(-1.0, bank.clone()), (2.0, bank.clone())],
                &RefMeasure::Uniform,
                None,
            ),
            Err(PbError::InvalidConfig { .. })
        ));
        assert!(matches!(
            average_banks(&[(1.0, bank)], &RefMeasure::default(), None),
            Err(PbError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn union_grid_mapping_represents_coarse_member_losslessly() {
        let model = fixture_model();
        let x = fixture_serve();
        let coarse = model.explain(&x, RefMeasure::Uniform).unwrap();
        let mut fine = coarse.clone();
        fine.merged_grids[0].borders = vec![1.0, 1.5, 2.0];
        fine.merged_grids[0].n_bins = 5;
        for table in &mut fine.tables {
            if table.u.contains(FeatureId(0)) {
                let old_values = table.values.clone();
                let old_support = table.support.clone();
                let shape = table.values.shape();
                let mut new_shape = shape.clone();
                if let Some(first) = new_shape.get_mut(0) {
                    *first = 5;
                }
                table.values = Tensor::try_zeros(new_shape.clone()).unwrap();
                table.support = Tensor::try_zeros(new_shape).unwrap();
                let extents = table.values.shape();
                walk_extents(&extents, |coord| {
                    let mut src = coord.to_vec();
                    let mapped = match coord[0] {
                        0 => 0,
                        1 | 2 => 1,
                        3 | 4 => 2,
                        _ => panic!("unexpected coord"),
                    };
                    src[0] = mapped;
                    table.values.set(coord, old_values.at(&src).unwrap())?;
                    table.support.set(coord, old_support.at(&src).unwrap())?;
                    Ok(())
                })
                .unwrap();
            }
        }

        let avg = average_banks(
            &[(0.5, coarse.clone()), (0.5, fine.clone())],
            &RefMeasure::Uniform,
            None,
        )
        .unwrap();
        assert_eq!(avg.merged_grids[0].borders, vec![1.0, 1.5, 2.0]);
        assert_eq!(avg.merged_grids[1], coarse.merged_grids[1]);

        for c0 in 0..5 {
            let coarse_c0 = match c0 {
                0 => 0,
                1 | 2 => 1,
                3 | 4 => 2,
                _ => panic!("unexpected coord"),
            };
            for c1 in 0..3 {
                let got = avg.score(&[c0, c1]).unwrap();
                let want = coarse.score(&[coarse_c0, c1]).unwrap();
                assert!((got - want).abs() < 1.0e-10, "({c0},{c1}): {got} vs {want}");
            }
        }

        // Re-purification still leaves slice means centered on the finer grid.
        let main0 = table(&avg, &[0]);
        let mean0: f64 = main0.values.values().iter().sum::<f64>() / main0.values.len() as f64;
        assert!(mean0.abs() < 1.0e-10);
        assert!(main0.se_band.is_some());
    }

    #[test]
    fn attach_se_bands_are_zero_for_score_identical_members_with_uneven_fanout() {
        // Regression test for the mapped_raw_effects fanout bug (attach_se_bands used to
        // REPLICATE a member's support across the union sub-cells it covers instead of
        // splitting it, unlike the main accumulation). Build two members that represent
        // the IDENTICAL function — `fine` is `coarse` with axis 0's bin 1 losslessly
        // split into two sub-bins, bin 2 left whole — under RefMeasure::ProductMarginals
        // (the default), where purification weights are derived from support. The
        // refinement is deliberately ASYMMETRIC (fanout 2 on bin 1, fanout 1 on bin 2):
        // a uniform fanout across every cell would rescale the induced weights by a
        // constant that normalization cancels either way, hiding a replicate-vs-split
        // bug; an uneven fanout does not cancel, so this reliably exercises the fix.
        let model = fixture_model();
        let x = fixture_serve();
        let coarse = model.explain(&x, RefMeasure::default()).unwrap();

        let mut fine = coarse.clone();
        fine.merged_grids[0].borders = vec![1.0, 1.5];
        fine.merged_grids[0].n_bins = 4;
        for table in &mut fine.tables {
            if table.u.contains(FeatureId(0)) {
                let old_values = table.values.clone();
                let old_support = table.support.clone();
                let shape = table.values.shape();
                let mut new_shape = shape.clone();
                if let Some(first) = new_shape.get_mut(0) {
                    *first = 4;
                }
                table.values = Tensor::try_zeros(new_shape.clone()).unwrap();
                table.support = Tensor::try_zeros(new_shape).unwrap();
                let extents = table.values.shape();
                walk_extents(&extents, |coord| {
                    // Coarse bin 1 ("<=1.5") splits into fine bins 1,2 (fanout 2, support
                    // halved each); coarse bin 2 (">1.5") stays fine bin 3 whole (fanout 1).
                    let mut src = coord.to_vec();
                    let (mapped, k) = match coord[0] {
                        0 => (0, 1.0),
                        1 | 2 => (1, 2.0),
                        3 => (2, 1.0),
                        _ => panic!("unexpected coord"),
                    };
                    src[0] = mapped;
                    table.values.set(coord, old_values.at(&src).unwrap())?;
                    table
                        .support
                        .set(coord, old_support.at(&src).unwrap() / k)?;
                    Ok(())
                })
                .unwrap();
            }
        }

        let avg = average_banks(
            &[(0.5, coarse.clone()), (0.5, fine.clone())],
            &RefMeasure::default(),
            None,
        )
        .unwrap();

        for table in &avg.tables {
            let band = table
                .se_band
                .as_ref()
                .expect("averaged banks carry SE bands");
            for se in band.per_cell.values().iter() {
                assert!(
                    se.abs() < 1.0e-9,
                    "table {:?}: SE {se} should be ~0 for score-identical members (grids differ only in axis-0 resolution)",
                    table.u
                );
            }
        }
    }
}
