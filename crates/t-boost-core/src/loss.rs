//! Objectives & the `Loss` trait (spec §2.4 / §05). The trait + companion types are
//! frozen here; the full v1 objective set ships: `SquaredError` (Identity),
//! `Logistic` (Logit), and the log-link frequency/severity objectives `Poisson`,
//! `Gamma`, `Tweedie { rho }`.
//!
//! The trait is fully orthogonal to tree shape, so it cannot touch I1/I2 — swapping
//! objectives never creates a >3-feature coupling or a non-constant leaf (§05.8). Every
//! method that can fail returns `Result<_, PbError>` (R-LOSSFALLIBLE): a fallible
//! objective maps its failure onto `PbError` rather than panicking.
//!
//! **Numerics (§05.4).** Log-link rows use `mu = exp(F)`; powers are emitted as
//! `exp(k·F)` (never `powf`), with the **exponent clamped to `[-30, 30]`**
//! ([`clamp_exp`]) so a runaway score saturates to a finite `mu` rather than `inf`/`NaN`.
//! The per-row hessian is floored at [`Loss::hessian_floor`] (`1e-16`, a NaN-guard).
//! `init_score`/`deviance` fold per-row `f32` terms into an `f64` accumulator in fixed
//! index order, so the scalar reductions are thread-count-independent (§05.9 #7).
//!
//! **Hot-loop / fail-fast (FLAG, spec §05.2 reconciliation).** §05.2 specifies a `grad_hess`
//! whose only failure is `ShapeMismatch` (saturation + the floor make the kernel total).
//! This crate makes `grad_hess` **stricter** — every loss rejects non-finite `y`/`raw`
//! and negative/non-finite `weight` per row, and rejects any non-finite computed `g/h`,
//! with a typed `InvalidInput` (the fail-fast house style established by the Phase-2
//! `SquaredError` hardening). Domain (sign) checks still live in `init_score`/`deviance`
//! per §05.3a; this adds a finiteness guard so a bug upstream surfaces as a typed error
//! rather than `NaN`/`inf` in the histogram.

use crate::engine::MAX_LEAVES;
use crate::error::PbError;
use itertools::izip;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Link-argument floor for `init_score` (§05.3): a valid-but-extreme weighted mean is
/// clamped here so an all-zero Poisson target yields a finite very-negative `f0`, not
/// `-inf`. A genuinely out-of-domain input is a typed `Err`, not a clamp (§05.3a).
const EPS_INIT: f64 = 1e-12;

/// The §05.4 exponent clamp range: `exp(k·F)` is evaluated as `exp(clamp(k·F, ±30))`,
/// well inside `f32` `exp` range, so the result is always finite. `pub(crate)` so the
/// leaf-refine closed form (engine::boost) can gate its exp-factoring on the same bound.
pub(crate) const EXP_CLAMP: f32 = 30.0;

fn invalid_input(what: String) -> PbError {
    PbError::InvalidInput { what }
}

/// `exp(x)` with the exponent clamped to `[-30, 30]` (§05.4): degrades a runaway score
/// to a finite saturated value rather than `inf`/`NaN`. The single power primitive for
/// every log-link loss — there is no `powf` on any objective path.
fn clamp_exp(x: f32) -> f32 {
    x.clamp(-EXP_CLAMP, EXP_CLAMP).exp()
}

/// Branch-stable logistic sigmoid `σ(F)` (§05.3): `F ≥ 0 → 1/(1+e^{−F})`, else
/// `e^F/(1+e^F)`. Never overflows; saturates to a finite `0`/`1` at large `|F|`.
fn stable_sigmoid(f: f32) -> f32 {
    if f >= 0.0 {
        let z = clamp_exp(-f);
        1.0 / (1.0 + z)
    } else {
        let z = clamp_exp(f);
        z / (1.0 + z)
    }
}

fn require_finite(obj: &str, label: &str, i: usize, v: f32) -> Result<(), PbError> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(invalid_input(format!(
            "{obj} {label}[{i}] must be finite, got {v}"
        )))
    }
}

fn require_weight(obj: &str, i: usize, w: f32) -> Result<(), PbError> {
    if w.is_finite() && w >= 0.0 {
        Ok(())
    } else {
        Err(invalid_input(format!(
            "{obj} weight[{i}] must be finite and >= 0, got {w}"
        )))
    }
}

fn ensure_finite_grad_hess(obj: &str, i: usize, g: f32, h: f32) -> Result<(f32, f32), PbError> {
    if g.is_finite() && h.is_finite() {
        Ok((g, h))
    } else {
        Err(invalid_input(format!(
            "{obj} grad_hess row {i} produced non-finite g/h: g={g}, h={h}"
        )))
    }
}

fn finish_deviance(obj: &str, acc: f64) -> Result<f32, PbError> {
    let out = acc as f32;
    if acc.is_finite() && out.is_finite() {
        Ok(out)
    } else {
        Err(invalid_input(format!(
            "{obj} deviance is not finite/representable as f32: {acc}"
        )))
    }
}

/// Row chunk size for the chunked-parallel deviance fold of compute-bound (log-link) objectives.
const PAR_DEVIANCE_CHUNK: usize = 8_192;

/// Row-parallel `(g, h)` fill for compute-bound (log-link) objectives. `out.{g,h}[i]` is an
/// independent function of row `i`, so this is a **map, not a reduction** — bit-identical to the
/// sequential loop regardless of thread count (no drift). `per_row(y, raw, weight)` returns the
/// UNCHECKED `(g, h)`; the per-row finite guards run here. Below the chunk size a sequential fold
/// runs. Squared-error keeps its sequential kernel.
fn fill_grad_hess_parallel<F>(
    obj: &'static str,
    out: &mut GradHess,
    y: &[f32],
    raw: &[f32],
    weight: &[f32],
    per_row: F,
) -> Result<(), PbError>
where
    F: Fn(f32, f32, f32) -> (f32, f32) + Sync,
{
    let n = y.len();
    // No `clear()`: the kernel below overwrites all `n` entries, so resizing to `n` is a no-op when
    // `out` is reused at the same length (the common case across boosting rounds) — this skips a
    // redundant per-call zero-fill. Byte-identical (every element is written before it is read).
    out.g.resize(n, 0.0);
    out.h.resize(n, 0.0);
    let body = |base: usize, gc: &mut [f32], hc: &mut [f32], yc: &[f32], rc: &[f32], wc: &[f32]| {
        for (k, ((gi, hi), ((&yi, &fi), &wi))) in gc
            .iter_mut()
            .zip(hc.iter_mut())
            .zip(yc.iter().zip(rc).zip(wc))
            .enumerate()
        {
            let i = base + k;
            require_finite(obj, "y", i, yi)?;
            require_finite(obj, "raw", i, fi)?;
            require_weight(obj, i, wi)?;
            let (g, h) = per_row(yi, fi, wi);
            let (g, h) = ensure_finite_grad_hess(obj, i, g, h)?;
            *gi = g;
            *hi = h;
        }
        Ok::<(), PbError>(())
    };
    if n < PAR_DEVIANCE_CHUNK {
        return body(0, &mut out.g, &mut out.h, y, raw, weight);
    }
    out.g
        .par_chunks_mut(PAR_DEVIANCE_CHUNK)
        .zip(out.h.par_chunks_mut(PAR_DEVIANCE_CHUNK))
        .zip(y.par_chunks(PAR_DEVIANCE_CHUNK))
        .zip(raw.par_chunks(PAR_DEVIANCE_CHUNK))
        .zip(weight.par_chunks(PAR_DEVIANCE_CHUNK))
        .with_min_len(crate::sched::min_chunks_per_task())
        .enumerate()
        .try_for_each(|(ci, ((((gc, hc), yc), rc), wc))| {
            body(ci * PAR_DEVIANCE_CHUNK, gc, hc, yc, rc, wc)
        })
}

/// Row-parallel `(g, h)` fill from a PRECOMPUTED inverse-link cache `mu[i] ≈ exp(F_i)` — the
/// incremental-mu path ([`Loss::grad_hess_from_mu`], gated on [`crate::engine::Config::incremental_mu`]).
/// Identical structure to [`fill_grad_hess_parallel`] (an independent per-row MAP ⇒ bit-identical
/// regardless of thread count), but reads the maintained `mu` (f64) instead of recomputing `exp(F)`
/// per row. `per_row(y, mu, weight)` returns the UNCHECKED `(g, h)` from the cached `mu`; the per-row
/// finite guards run here (on `mu`, not `raw`). NOT bit-identical to [`fill_grad_hess_parallel`]: the
/// cache carries the bounded multiplicative drift of the incremental update (see `incremental_mu`).
fn fill_grad_hess_from_mu_parallel<F>(
    obj: &'static str,
    out: &mut GradHess,
    y: &[f32],
    mu: &[f64],
    weight: &[f32],
    per_row: F,
) -> Result<(), PbError>
where
    F: Fn(f32, f64, f32) -> (f32, f32) + Sync,
{
    let n = y.len();
    out.g.resize(n, 0.0);
    out.h.resize(n, 0.0);
    let body = |base: usize, gc: &mut [f32], hc: &mut [f32], yc: &[f32], mc: &[f64], wc: &[f32]| {
        for (k, ((gi, hi), ((&yi, &mui), &wi))) in gc
            .iter_mut()
            .zip(hc.iter_mut())
            .zip(yc.iter().zip(mc).zip(wc))
            .enumerate()
        {
            let i = base + k;
            require_finite(obj, "y", i, yi)?;
            if !mui.is_finite() {
                return Err(invalid_input(format!(
                    "{obj} mu[{i}] must be finite, got {mui}"
                )));
            }
            require_weight(obj, i, wi)?;
            let (g, h) = per_row(yi, mui, wi);
            let (g, h) = ensure_finite_grad_hess(obj, i, g, h)?;
            *gi = g;
            *hi = h;
        }
        Ok::<(), PbError>(())
    };
    if n < PAR_DEVIANCE_CHUNK {
        return body(0, &mut out.g, &mut out.h, y, mu, weight);
    }
    out.g
        .par_chunks_mut(PAR_DEVIANCE_CHUNK)
        .zip(out.h.par_chunks_mut(PAR_DEVIANCE_CHUNK))
        .zip(y.par_chunks(PAR_DEVIANCE_CHUNK))
        .zip(mu.par_chunks(PAR_DEVIANCE_CHUNK))
        .zip(weight.par_chunks(PAR_DEVIANCE_CHUNK))
        .with_min_len(crate::sched::min_chunks_per_task())
        .enumerate()
        .try_for_each(|(ci, ((((gc, hc), yc), mc), wc))| {
            body(ci * PAR_DEVIANCE_CHUNK, gc, hc, yc, mc, wc)
        })
}

/// Chunked-parallel `(Σw, acc)` deviance fold for COMPUTE-bound (log-link) objectives. The rows are
/// split into FIXED-size chunks; each chunk folds sequentially into a local `(Σw, acc)`, and the
/// chunk partials are then combined in CHUNK ORDER. So the result is **thread-count-independent**
/// (the §05.9 #7 determinism gate holds — chunk boundaries and combination order do not depend on
/// the thread count), differing from a single linear fold only by ~1e-11 (chunked-summation
/// non-associativity). `term(i, y, raw, weight)` returns the per-row `(w_contrib, dev_contrib)`
/// after running the objective's domain guards. Below the chunk size a sequential fold runs (no
/// rayon overhead). Squared-error keeps its own sequential fold — its per-row term is a single
/// multiply (memory-bandwidth bound) where parallelism does not pay (cf. the reverted grad_hess
/// parallelization); the log-link `exp`/`ln` IS compute-bound, so it overlaps across cores. Like
/// the histogram-subtraction path this trades exact bit-reproducibility for speed at the ~1e-11
/// level — accuracy-neutral (it only perturbs the leaf-refine line search at an exact near-tie).
fn parallel_deviance_fold<F>(
    y: &[f32],
    raw: &[f32],
    weight: &[f32],
    term: F,
) -> Result<(f64, f64), PbError>
where
    F: Fn(usize, f32, f32, f32) -> Result<(f64, f64), PbError> + Sync,
{
    let fold_chunk =
        |base: usize, yc: &[f32], rc: &[f32], wc: &[f32]| -> Result<(f64, f64), PbError> {
            let (mut sw, mut acc) = (0.0_f64, 0.0_f64);
            for (k, ((&yi, &fi), &wi)) in yc.iter().zip(rc).zip(wc).enumerate() {
                let (w, d) = term(base + k, yi, fi, wi)?;
                sw += w;
                acc += d;
            }
            Ok((sw, acc))
        };
    if y.len() < PAR_DEVIANCE_CHUNK {
        return fold_chunk(0, y, raw, weight);
    }
    let partials: Result<Vec<(f64, f64)>, PbError> = y
        .par_chunks(PAR_DEVIANCE_CHUNK)
        .zip(raw.par_chunks(PAR_DEVIANCE_CHUNK))
        .zip(weight.par_chunks(PAR_DEVIANCE_CHUNK))
        .with_min_len(crate::sched::min_chunks_per_task())
        .enumerate()
        .map(|(ci, ((yc, rc), wc))| fold_chunk(ci * PAR_DEVIANCE_CHUNK, yc, rc, wc))
        .collect();
    // Combine partials in CHUNK ORDER (sequential over the order-preserving collected Vec).
    let (mut sum_w, mut acc) = (0.0_f64, 0.0_f64);
    for (csw, ca) in partials? {
        sum_w += csw;
        acc += ca;
    }
    Ok((sum_w, acc))
}

/// Fused single-pass `(g, h)` MAP + `(Σw, deviance)` FOLD for compute-bound (log-link) objectives,
/// so a caller needing BOTH at the same `raw` (the leaf-refine line-search baseline `init_dev`)
/// computes the link transcendental ONCE per row instead of in two separate passes. `per_row`
/// returns `(g, h, w_contrib, dev_contrib)` with the UNCHECKED `(g, h)` (the per-row finite/domain
/// guards run inside it); the `ensure_finite_grad_hess` guard runs here, exactly as in
/// [`fill_grad_hess_parallel`]. The g/h writes are an order-independent map (bit-identical to
/// `fill_grad_hess_parallel`); the deviance folds per FIXED-size chunk and combines partials in
/// CHUNK ORDER with the SAME `PAR_DEVIANCE_CHUNK` (bit-identical to [`parallel_deviance_fold`]) ⇒
/// the result is byte-for-byte the separate `grad_hess` then `deviance`, just one σ/exp cheaper.
/// Returns `(Σw, acc)`; the caller applies the all-zero-weight check + `finish_deviance` (matching
/// `deviance`). `out.{g,h}` are resized to `n`; the caller guards lengths first.
fn fill_grad_hess_and_fold_deviance<F>(
    obj: &'static str,
    out: &mut GradHess,
    y: &[f32],
    raw: &[f32],
    weight: &[f32],
    per_row: F,
) -> Result<(f64, f64), PbError>
where
    F: Fn(usize, f32, f32, f32) -> Result<(f32, f32, f64, f64), PbError> + Sync,
{
    let n = y.len();
    // No `clear()`: the kernel below overwrites all `n` entries, so resizing to `n` is a no-op when
    // `out` is reused at the same length (the common case across boosting rounds) — this skips a
    // redundant per-call zero-fill. Byte-identical (every element is written before it is read).
    out.g.resize(n, 0.0);
    out.h.resize(n, 0.0);
    let body = |base: usize,
                gc: &mut [f32],
                hc: &mut [f32],
                yc: &[f32],
                rc: &[f32],
                wc: &[f32]|
     -> Result<(f64, f64), PbError> {
        let (mut sw, mut acc) = (0.0_f64, 0.0_f64);
        for (k, ((gi, hi), ((&yi, &fi), &wi))) in gc
            .iter_mut()
            .zip(hc.iter_mut())
            .zip(yc.iter().zip(rc).zip(wc))
            .enumerate()
        {
            let i = base + k;
            let (g, h, w, d) = per_row(i, yi, fi, wi)?;
            let (g, h) = ensure_finite_grad_hess(obj, i, g, h)?;
            *gi = g;
            *hi = h;
            sw += w;
            acc += d;
        }
        Ok((sw, acc))
    };
    if n < PAR_DEVIANCE_CHUNK {
        return body(0, &mut out.g, &mut out.h, y, raw, weight);
    }
    let partials: Result<Vec<(f64, f64)>, PbError> = out
        .g
        .par_chunks_mut(PAR_DEVIANCE_CHUNK)
        .zip(out.h.par_chunks_mut(PAR_DEVIANCE_CHUNK))
        .zip(y.par_chunks(PAR_DEVIANCE_CHUNK))
        .zip(raw.par_chunks(PAR_DEVIANCE_CHUNK))
        .zip(weight.par_chunks(PAR_DEVIANCE_CHUNK))
        .with_min_len(crate::sched::min_chunks_per_task())
        .enumerate()
        .map(|(ci, ((((gc, hc), yc), rc), wc))| body(ci * PAR_DEVIANCE_CHUNK, gc, hc, yc, rc, wc))
        .collect();
    // Combine partials in CHUNK ORDER (matches `parallel_deviance_fold`).
    let (mut sum_w, mut acc) = (0.0_f64, 0.0_f64);
    for (csw, ca) in partials? {
        sum_w += csw;
        acc += ca;
    }
    Ok((sum_w, acc))
}

/// Shared entry length-guard for the three slice methods (§05.2): the one in-method
/// failure for the closed-form losses, checked once before any hot loop.
fn require_equal_len(
    obj: &str,
    method: &str,
    n: usize,
    lens: &[(&str, usize)],
) -> Result<(), PbError> {
    for &(name, len) in lens {
        if len != n {
            return Err(PbError::ShapeMismatch {
                what: format!("{obj} {method}: y={n}, {name}={len}"),
            });
        }
    }
    Ok(())
}

/// The GLM log-link intercept: `f0 = log(Σ w y e^(1-p) / Σ w e^(2-p))` with
/// `e = exp(offset)` (or `e = 1` when there is no offset), the ratio floored to
/// [`EPS_INIT`]. `validate_y` enforces the per-row domain (Poisson/Tweedie `y ≥ 0`,
/// Gamma `y > 0`). Rejects all-zero weights and non-positive exposure (`Σ w e ≤ 0`).
fn log_link_init(
    obj: &str,
    power: f64,
    y: &[f32],
    weight: &[f32],
    offset: Option<&[f32]>,
    validate_y: impl Fn(usize, f32) -> Result<(), PbError>,
) -> Result<f64, PbError> {
    let n = y.len();
    require_equal_len(obj, "init_score", n, &[("weight", weight.len())])?;
    let (mut sum_wy, mut sum_we) = (0.0_f64, 0.0_f64);
    match offset {
        Some(off) => {
            require_equal_len(obj, "init_score", n, &[("offset", off.len())])?;
            for (i, ((&yi, &wi), &oi)) in y.iter().zip(weight).zip(off).enumerate() {
                require_finite(obj, "y", i, yi)?;
                require_weight(obj, i, wi)?;
                require_finite(obj, "offset", i, oi)?;
                validate_y(i, yi)?;
                let w = f64::from(wi);
                sum_wy += w * f64::from(yi) * ((1.0 - power) * f64::from(oi)).exp();
                sum_we += w * ((2.0 - power) * f64::from(oi)).exp();
            }
        }
        None => {
            for (i, (&yi, &wi)) in y.iter().zip(weight).enumerate() {
                require_finite(obj, "y", i, yi)?;
                require_weight(obj, i, wi)?;
                validate_y(i, yi)?;
                let w = f64::from(wi);
                sum_wy += w * f64::from(yi);
                sum_we += w; // e_i = 1
            }
        }
    }
    if !sum_wy.is_finite() || !sum_we.is_finite() || sum_we <= 0.0 {
        return Err(invalid_input(format!(
            "{obj} init_score: non-positive Σ w·e (all-zero weights or non-positive exposure)"
        )));
    }
    let ratio = (sum_wy / sum_we).max(EPS_INIT);
    Ok(ratio.ln())
}

/// Per-row first/second derivatives of the loss w.r.t. the raw score `F`
/// (spec §2.3). Full precision; leaves are always refit from these exact values.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct GradHess {
    /// Per-row gradient `∂L/∂F`.
    pub g: Vec<f32>,
    /// Per-row hessian `∂²L/∂F²` (floored at `Loss::hessian_floor`).
    pub h: Vec<f32>,
}

/// The inverse-link family of an objective (spec §2.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Link {
    /// `pred = raw` (regression on the natural scale).
    Identity,
    /// `pred = exp(raw)` (Poisson / Gamma / Tweedie mean).
    Log,
    /// `pred = 1/(1+exp(-raw))` (binary probability).
    Logit,
}

/// The early-stopping / evaluation metric an objective reports (spec §2.4; this is
/// §05's canonical form). Deviance-based by default — never RMSE on
/// Poisson/Gamma/Tweedie.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Metric {
    /// Root mean squared error (squared-error objective only).
    Rmse,
    /// Binary log-loss (logistic objective).
    LogLoss,
    /// Poisson deviance.
    PoissonDeviance,
    /// Gamma deviance.
    GammaDeviance,
    /// Tweedie deviance with power parameter `rho`.
    TweedieDeviance {
        /// Tweedie variance power, `1 < rho < 2`.
        rho: f32,
    },
}

/// Stable identifier for a concrete objective, recorded in [`ObjectiveTag`] so a
/// loaded `Model` can reproduce its link + loss (spec §05).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossId {
    /// `SquaredError` (identity link).
    SquaredError,
    /// `Logistic` (logit link).
    Logistic,
    /// `Poisson` (log link).
    Poisson,
    /// `Gamma` (log link).
    Gamma,
    /// `Tweedie` (log link; power in [`ObjectiveTag::tweedie_rho`]).
    Tweedie,
    /// Native softmax (multinomial multiclass). Tags each per-class scalar model whose raw score
    /// is that class's logit; the joint softmax lives on [`crate::engine::MultiClassModel`], so the
    /// per-class model itself carries [`Link::Identity`] (its raw score IS the logit).
    Softmax,
}

/// The trained objective, recorded on the model so export / `predict_proba` can
/// reproduce link + loss without the caller re-supplying anything (spec §2.6, R-SCHEMA).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveTag {
    /// The inverse-link family.
    pub link: Link,
    /// Which concrete objective was trained.
    pub loss: LossId,
    /// Tweedie power, present iff `loss == LossId::Tweedie`.
    pub tweedie_rho: Option<f32>,
}

/// A loss/objective (spec §2.4). Fully orthogonal to tree shape (I1/I2 untouched).
///
/// `grad_hess` is one pass w.r.t. the raw score `F` (after the exposure offset).
/// `init_score`/`deviance` are fallible (R-LOSSFALLIBLE): invalid domains (Gamma
/// `y<=0`, all-zero weights, bad exposure, user-loss failures) return a typed
/// `PbError` rather than panicking or emitting `NaN`.
pub trait Loss: Send + Sync {
    /// One full-precision gradient/hessian pass into `out`. Never panics: a fallible
    /// objective maps its failure onto `PbError`.
    fn grad_hess(
        &self,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError>;

    /// `link(weighted mean)` — the mandatory `boost_from_average` intercept `f0`.
    /// Full-width `f64` (the exact fANOVA intercept). Fallible on invalid domains.
    fn init_score(&self, y: &[f32], weight: &[f32], offset: Option<&[f32]>)
        -> Result<f64, PbError>;

    /// The inverse-link family of this objective.
    fn link(&self) -> Link;

    /// Inverse link applied to a single raw score (`exp(k·F)`, not `powf`).
    fn pred_from_raw(&self, raw: f32) -> f32;

    /// Strictly-proper deviance for early stopping (NOT RMSE on Poisson/Gamma/Tweedie).
    /// `f64` fold, reported in `f32`. Same invalid-domain typed errors as `init_score`.
    fn deviance(&self, y: &[f32], raw: &[f32], weight: &[f32]) -> Result<f32, PbError>;

    /// One pass computing BOTH the grad/hess (into `out`) AND the deviance — for a caller that
    /// needs both at the SAME `raw` (the leaf-refine line-search baseline) and would otherwise pay
    /// the link transcendental (σ/exp) twice. The returned deviance is bit-identical to
    /// `self.deviance(y, raw, weight)` and `out` is bit-identical to `self.grad_hess(y, raw,
    /// weight, out)`. The default is exactly that unfused pair (correct, no sharing); compute-bound
    /// log-link objectives override it to share the transcendental in a single fused pass.
    fn grad_hess_and_deviance(
        &self,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<f32, PbError> {
        self.grad_hess(y, raw, weight, out)?;
        self.deviance(y, raw, weight)
    }

    /// Fused grad/hess + per-leaf aggregate over `rows` (`memberships[i]` = leaf id of `rows[i]`,
    /// `< 8`): returns `(Σ g, Σ h)` per leaf as f64 sums folded in `rows` order. Bit-identical to
    /// calling `grad_hess` then folding `out.{g,h}[rows[i]]` into per-leaf sums, but lets a loss skip
    /// materializing the full-length gradient vector — the leaf-refine line search re-reads it
    /// immediately, so the round-trip is pure waste. The default IS exactly that two-pass form (using
    /// `scratch` as the gradient buffer). Deterministic: the fold is `rows`-order sequential.
    fn grad_hess_aggregate(
        &self,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        rows: &[u32],
        memberships: &[u8],
        scratch: &mut GradHess,
    ) -> Result<([f64; MAX_LEAVES], [f64; MAX_LEAVES]), PbError> {
        self.grad_hess(y, raw, weight, scratch)?;
        let mut g = [0.0_f64; MAX_LEAVES];
        let mut h = [0.0_f64; MAX_LEAVES];
        for (&row, &leaf) in rows.iter().zip(memberships) {
            let ru = row as usize;
            let l = usize::from(leaf);
            *g.get_mut(l).ok_or_else(|| PbError::Internal {
                what: "grad_hess_aggregate leaf id escaped".into(),
            })? += f64::from(*scratch.g.get(ru).ok_or_else(|| PbError::Internal {
                what: "grad_hess_aggregate gradient row escaped".into(),
            })?);
            *h.get_mut(l).ok_or_else(|| PbError::Internal {
                what: "grad_hess_aggregate leaf id escaped".into(),
            })? += f64::from(*scratch.h.get(ru).ok_or_else(|| PbError::Internal {
                what: "grad_hess_aggregate hessian row escaped".into(),
            })?);
        }
        Ok((g, h))
    }

    /// Whether the per-row hessian depends on the current `raw` score (so must be recomputed every
    /// boosting round). Default `true` (safe). `SquaredError` returns `false`: its hessian
    /// `h = w·max(floor)` is raw-independent and the weights are round-invariant, so the boosting
    /// loop fills `h` once (round 0) and reuses it via [`fill_grad_reusing_hessian`].
    fn hessian_depends_on_raw(&self) -> bool {
        true
    }

    /// Refill ONLY the gradient column `out.g`, leaving `out.h` (the round-invariant hessian from a
    /// prior full [`grad_hess`]) untouched — for the boosting loop when `!hessian_depends_on_raw()`.
    /// The default IS a full `grad_hess`, so a loss that does not opt in stays correct; an opting-in
    /// loss must keep `out.g` bit-identical to `grad_hess`'s. `out.h` must already be length `y.len()`
    /// (a prior full pass established it).
    fn fill_grad_reusing_hessian(
        &self,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError> {
        self.grad_hess(y, raw, weight, out)
    }

    /// Whether this objective supports the incremental inverse-link cache `mu = exp(F)`
    /// ([`crate::engine::Config::incremental_mu`]): a log-link loss whose `(g, h)` is a pointwise
    /// function of `mu`, so the boosting loop can maintain `mu` MULTIPLICATIVELY (`mu *= e^{Δleaf}`)
    /// across rounds instead of recomputing `exp(F)` every round. Default `false` (the loss stays on
    /// the exact per-round `exp(F)` path). `Poisson` overrides to `true`.
    fn supports_incremental_mu(&self) -> bool {
        false
    }

    /// Recompute the inverse-link cache in place, `mu[i] = pred_from_raw(raw[i])` — a fresh `exp(F)`
    /// pass that RE-ALIGNS the cache to the exact score, resetting the incremental path's accumulated
    /// drift. Only meaningful when [`Self::supports_incremental_mu`]; the default errors.
    /// `mu.len() == raw.len()` (caller-sized). The produced value is EXACTLY what [`Self::grad_hess`]
    /// reads, so `grad_hess_from_mu` on a just-refreshed cache is bit-identical to `grad_hess`.
    fn refresh_mu(&self, _raw: &[f32], _mu: &mut [f64]) -> Result<(), PbError> {
        Err(PbError::Internal {
            what: "refresh_mu called on a loss without incremental-mu support".into(),
        })
    }

    /// `(g, h)` from the maintained inverse-link cache `mu` (see [`Self::supports_incremental_mu`]),
    /// skipping the per-row `exp(F)` of [`Self::grad_hess`]. `out.{g,h}` match `grad_hess` EXCEPT for
    /// `mu`'s bounded multiplicative drift; the cache clamp reproduces [`clamp_exp`]'s ±30 saturation
    /// (monotone: `clamp(exp(F), e^{-30}, e^{30}) == exp(clamp(F, ±30))`). The default errors.
    fn grad_hess_from_mu(
        &self,
        _y: &[f32],
        _mu: &[f64],
        _weight: &[f32],
        _out: &mut GradHess,
    ) -> Result<(), PbError> {
        Err(PbError::Internal {
            what: "grad_hess_from_mu called on a loss without incremental-mu support".into(),
        })
    }

    /// The objective's natural early-stopping metric (deviance by default).
    fn default_metric(&self) -> Metric;

    /// The trained-objective tag (link + loss id + Tweedie power) recorded in
    /// `ModelSchema.objective` so a loaded model can reproduce link + loss for
    /// export / `predict_proba` (R-SCHEMA). FLAG (spec §2.4/§05 trait addition): the
    /// canonical trait did not list this; the engine needs it to populate the schema.
    fn objective_tag(&self) -> ObjectiveTag;

    /// Lower clamp on the per-row hessian (numerical floor ε); default `1e-16`
    /// (NaN-guard only — stability is `λ` + `max_delta_step`'s job).
    fn hessian_floor(&self) -> f32 {
        1e-16
    }

    /// Per-objective default leaf-stage `|w*|`-clamp. `None` = uncapped; spec §05.3's
    /// canonical loss table caps the LOG-LINK objectives (Poisson, Gamma, Tweedie) at
    /// `Some(0.7)` — one shared failure mode, a near-zero log-link Newton denominator:
    /// Poisson's `h = w·μ` explodes the step as `μ → 0`, and Gamma/Tweedie's vanishing
    /// `h` under heavy-tailed `y ≫ μ` (`g ∼ −w·y·μ^{1−ρ}`) destabilizes small
    /// aggregated-exposure portfolios. Widened from Poisson-only to all log links as a
    /// confirmed deliberate decision (2026-07-11 review remediation); spec §05.6 records
    /// it. `Config.max_delta_step` falls back to this when unset and overrides it when
    /// set (including back to uncapped).
    fn max_delta_step(&self) -> Option<f32> {
        None
    }

    /// Per-objective default for the GATED tighter leaf-step cap (spec §05.6 addendum,
    /// av35). `None` (the trait default) = this objective ships no gate at all and keeps
    /// its plain [`Loss::max_delta_step`] for the whole fit — bit-identical to before the
    /// gate existed. `Some(g)` arms the in-fit collapse detector described on
    /// [`GatedDeltaStep`].
    ///
    /// Only Tweedie overrides this. Gamma was MEASURED HARMFUL under a blanket `0.3`
    /// (freclaimdam −0.0105 skill, t=−2.97) and Poisson is untested, so both deliberately
    /// keep the plain `0.7`; see the [`GatedDeltaStep`] doc for the evidence.
    fn gated_max_delta_step(&self) -> Option<GatedDeltaStep> {
        None
    }
}

/// Parameters of the GATED tighter leaf-step cap (spec §05.6 addendum, av35).
///
/// # The failure mode
///
/// On log-link Tweedie fits over SMALL AGGREGATED-EXPOSURE portfolios, boosting can drive a
/// small subgroup's predicted rate `μ = exp(F)` toward zero. The `ρ ≈ 1.7` unit deviance's
/// `−y·μ^{1−ρ}` term then diverges like `μ^{−0.7}`, so a single high-`y` test row in that
/// subgroup can carry ~24% of the whole test deviance. Measured on `ohlsson_pp` (mean rate
/// ≈ 274): the collapsing splits push min `μ` down to 0.4–0.9 — a rate ratio of ~1.5e-3 to
/// 3.3e-3 — and the top-1 row's deviance share reaches 5–10%.
///
/// # Why a gate rather than a tighter default
///
/// A blanket `max_delta_step = 0.3` fixes the collapse (+0.0325 mean skill on the affected
/// `ohlsson_pp` splits, seed-sd −30%) but CLIPS USEFUL LEAF STEPS elsewhere: it costs
/// `swautoins_pp`'s `learning_rate = 0.1` tuned champion −0.0043 (sign-test p = 0.043), a
/// portfolio whose min `μ` never drops below ~50. The dose response is a threshold, not a
/// ramp — `0.5` is inert, `0.3` is effective — so the cap has to be spent only where the
/// collapse actually happens.
///
/// # The signal
///
/// Log link makes the collapse test exact and exp-free. The engine's init score is
/// `f0 = ln(weighted mean rate)` (see `log_link_init`), and each row's accumulated raw
/// score is `F_i = f0 + offset_i + Σ_t α_t · tree_t(x_i)`. So
///
/// ```text
/// s_i := F_i − offset_i − f0 = ln( μ_i / weighted-mean-rate )
/// ```
///
/// is exactly the log of the row's predicted rate RELATIVE TO THE PORTFOLIO MEAN — scale
/// free, and computable as two f64 subtractions per row. The gate engages when
///
/// ```text
/// min over TRAIN rows of s_i  <  ln(collapse_threshold)
/// ```
///
/// i.e. when some training row's predicted rate has fallen below
/// `collapse_threshold × mean rate`. The min is taken over `train_rows` (the ES carve's
/// holdout is excluded — the gate is a training-state diagnostic) in fixed row order, so it
/// is a function of `(data, config, seed)` alone: no wall clock, no environment, no thread
/// dependence.
///
/// # Engage semantics (MONOTONE, MID-FIT)
///
/// The check runs at the TOP of each boosting round `t`, against `raw` as it stands after
/// round `t−1`'s tree was added (never the AGBM/DART lookahead — the gate reads the model's
/// real accumulated state). If the condition holds, the cap engages FOR ROUND `t` AND EVERY
/// SUBSEQUENT ROUND of that fit; trees `0..t` are already grown and are left untouched. Once
/// engaged it never disengages, and the check stops running (the per-round O(n) min is paid
/// only until the gate fires, or for the whole fit if it never does).
///
/// At `t = 0` every `s_i` is 0 by construction, so a fit that never collapses is
/// BIT-IDENTICAL to one built without the gate. Under outer bagging each bag is an
/// independent `fit_single` with its own gate, so bags can engage at different rounds (or
/// not at all) — which is what "the collapse is a property of this bag's data" means.
///
/// Engaging applies `min(resolved max_delta_step, capped_step)`, so the gate can only ever
/// TIGHTEN the clamp, never loosen it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GatedDeltaStep {
    /// Ratio of predicted rate to weighted mean rate below which the collapse is declared.
    /// Must be finite and in `(0, 1)`. The shipped Tweedie value is `1e-3`.
    pub collapse_threshold: f32,
    /// The tighter `|w*|` clamp applied from the engaging round onward. Must be finite and
    /// `> 0`. The shipped Tweedie value is `0.3`.
    pub capped_step: f32,
}

impl GatedDeltaStep {
    /// `ln(collapse_threshold)` — the raw-space comparison point the engine actually uses.
    #[must_use]
    pub fn log_threshold(&self) -> f64 {
        f64::from(self.collapse_threshold).ln()
    }

    /// Reject a nonsensical gate at config-validation time.
    ///
    /// # Errors
    /// [`PbError::InvalidConfig`] when `collapse_threshold` is not finite in `(0, 1)` or
    /// `capped_step` is not finite and `> 0`.
    pub fn validate(&self) -> Result<(), PbError> {
        if !self.collapse_threshold.is_finite()
            || self.collapse_threshold <= 0.0
            || self.collapse_threshold >= 1.0
        {
            return Err(PbError::InvalidConfig {
                what: format!(
                    "max_delta_step_gated collapse_threshold must be finite and in (0, 1), got {}",
                    self.collapse_threshold
                ),
            });
        }
        if !self.capped_step.is_finite() || self.capped_step <= 0.0 {
            return Err(PbError::InvalidConfig {
                what: format!(
                    "max_delta_step_gated capped_step must be finite and > 0, got {}",
                    self.capped_step
                ),
            });
        }
        Ok(())
    }
}

/// The Tweedie gated-cap default: engage `max_delta_step = 0.3` once a training row's
/// predicted rate falls below **0.1%** of the weighted mean rate.
///
/// Both numbers are the battery's measured operating point, not round numbers.
/// `capped_step = 0.3` is the dose the blanket study established (`0.5` inert, `0.3`
/// effective). `collapse_threshold = 1e-3` was chosen from the measured lower tail of
/// `min_i ln(mu_i / mean rate)` over every train row and round of a fit:
///
/// | set                         | worst observed | fires at 1e-3? |
/// |-----------------------------|---------------:|:---------------|
/// | `swautoins_pp` (both hp arms) | −3.35        | never (3.6 nat margin) |
/// | `euhealth_pp`               | −4.97          | never          |
/// | `ausprivauto_pp`            | −12.9          | sporadically, per bag |
/// | `ohlsson_pp` (the mechanism)| −10.4 … −36.9  | ALWAYS, every fit |
///
/// A 1e-2 threshold also separates `swautoins_pp` cleanly, but engages ~17 rounds earlier on
/// `ohlsson_pp` and measured WORSE there (+0.0022, t = +0.64, and it destabilized the stable
/// split 2) than 1e-3 (+0.0050, t = +2.71, W-L 8-1, split 2 untouched). 3e-2 scored higher
/// still (+0.0096) but leaves only 0.16 nats of margin over `swautoins_pp`'s worst
/// observation — not a threshold anyone should stand on. See [`GatedDeltaStep`].
pub const TWEEDIE_GATED_DELTA_STEP: GatedDeltaStep = GatedDeltaStep {
    collapse_threshold: 1e-3,
    capped_step: 0.3,
};

/// Squared-error regression (spec §05.3): Identity link, `g = w·(F − y)`, `h = w·1`,
/// `init_score = weighted mean of y` (offset-aware), half-deviance metric. The v1
/// green-spine objective — chosen first because reconstruction is testable before any
/// nonlinear link. `weight[i]` scales row `i`'s `(g, h)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SquaredError;

impl Loss for SquaredError {
    fn grad_hess(
        &self,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError> {
        let n = y.len();
        if raw.len() != n || weight.len() != n {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "grad_hess: y={n}, raw={}, weight={}",
                    raw.len(),
                    weight.len()
                ),
            });
        }
        // No `clear()`: the loop overwrites all `n` entries, so resizing to `n` is a no-op when
        // `out` is reused at the same length — skips a redundant per-call zero-fill (byte-identical).
        out.g.resize(n, 0.0);
        out.h.resize(n, 0.0);
        let floor = self.hessian_floor();
        // izip! ⇒ no indexing in the hot loop (§05 hot-loop policy). g = w(F−y);
        // h = w·1, floored so the postcondition out.h[i] >= floor holds even at w=0.
        for (i, (gi, hi, &yi, &fi, &wi)) in
            izip!(&mut out.g, &mut out.h, y, raw, weight).enumerate()
        {
            require_finite("squared-error", "y", i, yi)?;
            require_finite("squared-error", "raw", i, fi)?;
            require_weight("squared-error", i, wi)?;
            let (g, h) =
                ensure_finite_grad_hess("squared-error", i, wi * (fi - yi), wi.max(floor))?;
            *gi = g;
            *hi = h;
        }
        Ok(())
    }

    fn grad_hess_aggregate(
        &self,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        rows: &[u32],
        memberships: &[u8],
        _scratch: &mut GradHess,
    ) -> Result<([f64; MAX_LEAVES], [f64; MAX_LEAVES]), PbError> {
        // FUSED SE kernel + per-leaf aggregate in ONE `rows`-order pass: each row's f32 `(g, h)` is
        // EXACTLY what `grad_hess` computes (`g = w(F−y)`, `h = w.max(floor)`, same finite checks and
        // same f32 rounding via `ensure_finite_grad_hess`), folded straight into the per-leaf f64
        // sums — bit-identical to grad_hess-then-aggregate but WITHOUT materializing the full-length
        // gradient vector (which the leaf-refine line search would re-read at once). Visits only
        // `rows` (the train subset), not all n.
        let floor = self.hessian_floor();
        let mut g = [0.0_f64; MAX_LEAVES];
        let mut h = [0.0_f64; MAX_LEAVES];
        for (&row, &leaf) in rows.iter().zip(memberships) {
            let ru = row as usize;
            let l = usize::from(leaf);
            let yi = *y.get(ru).ok_or_else(|| PbError::Internal {
                what: "se grad_hess_aggregate y row escaped".into(),
            })?;
            let fi = *raw.get(ru).ok_or_else(|| PbError::Internal {
                what: "se grad_hess_aggregate raw row escaped".into(),
            })?;
            let wi = *weight.get(ru).ok_or_else(|| PbError::Internal {
                what: "se grad_hess_aggregate weight row escaped".into(),
            })?;
            require_finite("squared-error", "y", ru, yi)?;
            require_finite("squared-error", "raw", ru, fi)?;
            require_weight("squared-error", ru, wi)?;
            let (gi, hi) =
                ensure_finite_grad_hess("squared-error", ru, wi * (fi - yi), wi.max(floor))?;
            *g.get_mut(l).ok_or_else(|| PbError::Internal {
                what: "se grad_hess_aggregate g leaf escaped".into(),
            })? += f64::from(gi);
            *h.get_mut(l).ok_or_else(|| PbError::Internal {
                what: "se grad_hess_aggregate h leaf escaped".into(),
            })? += f64::from(hi);
        }
        Ok((g, h))
    }

    fn hessian_depends_on_raw(&self) -> bool {
        // h = w·max(floor) is independent of `raw`, and weights are round-invariant ⇒ the hessian
        // column is bit-identical every boosting round.
        false
    }

    fn fill_grad_reusing_hessian(
        &self,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError> {
        let n = y.len();
        if raw.len() != n || weight.len() != n {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "grad(reuse-h): y={n}, raw={}, weight={}",
                    raw.len(),
                    weight.len()
                ),
            });
        }
        // Only valid once a full pass has established `out.h`; otherwise fall back to the full kernel.
        if out.h.len() != n {
            return self.grad_hess(y, raw, weight, out);
        }
        out.g.resize(n, 0.0);
        // g is computed EXACTLY as in `grad_hess` (g = w·(F−y), same finite checks/rounding); h is
        // left as the prior full pass set it (= w·max(floor), unchanged this round) ⇒ bit-identical
        // (g,h), at the cost of only the gradient store instead of g + the redundant hessian store.
        for (i, (gi, (&yi, (&fi, &wi)))) in out
            .g
            .iter_mut()
            .zip(y.iter().zip(raw.iter().zip(weight)))
            .enumerate()
        {
            require_finite("squared-error", "y", i, yi)?;
            require_finite("squared-error", "raw", i, fi)?;
            require_weight("squared-error", i, wi)?;
            let g = wi * (fi - yi);
            if !g.is_finite() {
                return Err(invalid_input(format!(
                    "squared-error grad row {i} produced non-finite g: g={g}"
                )));
            }
            *gi = g;
        }
        Ok(())
    }

    fn init_score(
        &self,
        y: &[f32],
        weight: &[f32],
        offset: Option<&[f32]>,
    ) -> Result<f64, PbError> {
        let n = y.len();
        if weight.len() != n {
            return Err(PbError::ShapeMismatch {
                what: format!("init_score: y={n}, weight={}", weight.len()),
            });
        }
        // f64 fixed-order (sequential, index-order) fold ⇒ thread-count-independent.
        // Identity link ⇒ link(weighted mean) = weighted mean. With an offset, the
        // best initial constant is the weighted mean of (y − offset), since the
        // initial raw score is f0 + offset (offset-aware Identity form, §03.7/§05.5).
        let (mut sum_w, mut sum_wy) = (0.0_f64, 0.0_f64);
        match offset {
            Some(off) => {
                if off.len() != n {
                    return Err(PbError::ShapeMismatch {
                        what: format!("init_score: y={n}, offset={}", off.len()),
                    });
                }
                for (i, ((&yi, &wi), &oi)) in y.iter().zip(weight).zip(off).enumerate() {
                    require_finite("squared-error", "y", i, yi)?;
                    require_weight("squared-error", i, wi)?;
                    require_finite("squared-error", "offset", i, oi)?;
                    sum_w += f64::from(wi);
                    sum_wy += f64::from(wi) * (f64::from(yi) - f64::from(oi));
                }
            }
            None => {
                for (i, (&yi, &wi)) in y.iter().zip(weight).enumerate() {
                    require_finite("squared-error", "y", i, yi)?;
                    require_weight("squared-error", i, wi)?;
                    sum_w += f64::from(wi);
                    sum_wy += f64::from(wi) * f64::from(yi);
                }
            }
        }
        if sum_w <= 0.0 {
            return Err(PbError::InvalidInput {
                what: "squared-error init_score: all-zero (or non-positive) weights".into(),
            });
        }
        Ok(sum_wy / sum_w)
    }

    fn link(&self) -> Link {
        Link::Identity
    }

    fn pred_from_raw(&self, raw: f32) -> f32 {
        raw // Identity inverse link
    }

    fn deviance(&self, y: &[f32], raw: &[f32], weight: &[f32]) -> Result<f32, PbError> {
        let n = y.len();
        if raw.len() != n || weight.len() != n {
            return Err(PbError::ShapeMismatch {
                what: format!(
                    "deviance: y={n}, raw={}, weight={}",
                    raw.len(),
                    weight.len()
                ),
            });
        }
        // Half-deviance = ½ Σ w (raw − y)²  (= ½ MSE·Σw). f64 fold, reported f32.
        let (mut sum_w, mut acc) = (0.0_f64, 0.0_f64);
        for (i, ((&yi, &fi), &wi)) in y.iter().zip(raw).zip(weight).enumerate() {
            require_finite("squared-error", "y", i, yi)?;
            require_finite("squared-error", "raw", i, fi)?;
            require_weight("squared-error", i, wi)?;
            sum_w += f64::from(wi);
            let r = f64::from(fi) - f64::from(yi);
            acc += f64::from(wi) * r * r;
        }
        if sum_w <= 0.0 {
            return Err(PbError::InvalidInput {
                what: "squared-error deviance: all-zero (or non-positive) weights".into(),
            });
        }
        finish_deviance("squared-error", 0.5 * acc)
    }

    fn default_metric(&self) -> Metric {
        Metric::Rmse
    }

    fn objective_tag(&self) -> ObjectiveTag {
        ObjectiveTag {
            link: Link::Identity,
            loss: LossId::SquaredError,
            tweedie_rho: None,
        }
    }
}

/// Binary logistic regression (spec §05.3): Logit link, `g = w·(σ(F) − y)`,
/// `h = w·σ(F)(1−σ(F))` (floored — the corner where `σ` saturates), `init_score =
/// log(p̄/(1−p̄))`, log-loss deviance. `σ` is the branch-stable sigmoid ([`stable_sigmoid`]).
/// Accepts soft labels `y ∈ [0, 1]`, not only `{0, 1}`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Logistic;

impl Loss for Logistic {
    fn grad_hess(
        &self,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError> {
        let n = y.len();
        require_equal_len(
            "logistic",
            "grad_hess",
            n,
            &[("raw", raw.len()), ("weight", weight.len())],
        )?;
        // Compute-bound (sigmoid per row) ⇒ row-parallel map, bit-identical to the sequential
        // kernel (no fold, no drift): g = w(σ−y), h = w·σ(1−σ) floored.
        let floor = self.hessian_floor();
        fill_grad_hess_parallel("logistic", out, y, raw, weight, |yi, fi, wi| {
            let s = stable_sigmoid(fi);
            (wi * (s - yi), (wi * s * (1.0 - s)).max(floor))
        })
    }

    fn init_score(
        &self,
        y: &[f32],
        weight: &[f32],
        offset: Option<&[f32]>,
    ) -> Result<f64, PbError> {
        let n = y.len();
        require_equal_len("logistic", "init_score", n, &[("weight", weight.len())])?;
        // The offset (if any) is a logit-space base margin the engine folds into `raw`;
        // it is validated for well-formedness but does NOT enter f0 (exposure is a
        // log-link concept, §05.5). f0 is the boost-from-average seed log(p̄/(1−p̄)).
        if let Some(off) = offset {
            require_equal_len("logistic", "init_score", n, &[("offset", off.len())])?;
            for (i, &oi) in off.iter().enumerate() {
                require_finite("logistic", "offset", i, oi)?;
            }
        }
        let (mut sum_w, mut sum_wy) = (0.0_f64, 0.0_f64);
        for (i, (&yi, &wi)) in y.iter().zip(weight).enumerate() {
            require_finite("logistic", "y", i, yi)?;
            require_weight("logistic", i, wi)?;
            if !(0.0..=1.0).contains(&yi) {
                return Err(invalid_input(format!(
                    "logistic: y[{i}] must be in [0, 1], got {yi}"
                )));
            }
            sum_w += f64::from(wi);
            sum_wy += f64::from(wi) * f64::from(yi);
        }
        if sum_w <= 0.0 {
            return Err(invalid_input(
                "logistic init_score: all-zero weights".into(),
            ));
        }
        if sum_wy <= 0.0 || sum_wy >= sum_w {
            return Err(invalid_input(
                "logistic: no positive (or no negative) class under w".into(),
            ));
        }
        let p = (sum_wy / sum_w).clamp(EPS_INIT, 1.0 - EPS_INIT);
        Ok((p / (1.0 - p)).ln())
    }

    fn link(&self) -> Link {
        Link::Logit
    }

    fn pred_from_raw(&self, raw: f32) -> f32 {
        stable_sigmoid(raw)
    }

    fn deviance(&self, y: &[f32], raw: &[f32], weight: &[f32]) -> Result<f32, PbError> {
        let n = y.len();
        require_equal_len(
            "logistic",
            "deviance",
            n,
            &[("raw", raw.len()), ("weight", weight.len())],
        )?;
        // Binomial unit deviance: 2 Σ w [ y ln(y/p) + (1−y) ln((1−y)/(1−p)) ]. Compute-bound
        // (sigmoid + two logs per row) ⇒ chunked-parallel fold (thread-count-independent).
        let (sum_w, acc) = parallel_deviance_fold(y, raw, weight, |i, yi, fi, wi| {
            require_finite("logistic", "y", i, yi)?;
            require_finite("logistic", "raw", i, fi)?;
            require_weight("logistic", i, wi)?;
            if !(0.0..=1.0).contains(&yi) {
                return Err(invalid_input(format!(
                    "logistic: y[{i}] must be in [0, 1], got {yi}"
                )));
            }
            let p = f64::from(stable_sigmoid(fi)).clamp(EPS_INIT, 1.0 - EPS_INIT);
            let yy = f64::from(yi);
            let t1 = if yy > 0.0 { yy * (yy / p).ln() } else { 0.0 };
            let omy = 1.0 - yy;
            let t2 = if omy > 0.0 {
                omy * (omy / (1.0 - p)).ln()
            } else {
                0.0
            };
            Ok((f64::from(wi), f64::from(wi) * 2.0 * (t1 + t2)))
        })?;
        if sum_w <= 0.0 {
            return Err(invalid_input("logistic deviance: all-zero weights".into()));
        }
        finish_deviance("logistic", acc)
    }

    fn grad_hess_and_deviance(
        &self,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<f32, PbError> {
        require_equal_len(
            "logistic",
            "grad_hess_and_deviance",
            y.len(),
            &[("raw", raw.len()), ("weight", weight.len())],
        )?;
        // One σ(F) per row, reused for BOTH (g, h) and the binomial deviance — bit-identical to
        // the separate `grad_hess` (same `stable_sigmoid(fi)` ⇒ same g/h) and `deviance` (same
        // clamped `p`, same chunked fold).
        let floor = self.hessian_floor();
        let (sum_w, acc) =
            fill_grad_hess_and_fold_deviance("logistic", out, y, raw, weight, |i, yi, fi, wi| {
                require_finite("logistic", "y", i, yi)?;
                require_finite("logistic", "raw", i, fi)?;
                require_weight("logistic", i, wi)?;
                if !(0.0..=1.0).contains(&yi) {
                    return Err(invalid_input(format!(
                        "logistic: y[{i}] must be in [0, 1], got {yi}"
                    )));
                }
                let s = stable_sigmoid(fi);
                let g = wi * (s - yi);
                let h = (wi * s * (1.0 - s)).max(floor);
                let p = f64::from(s).clamp(EPS_INIT, 1.0 - EPS_INIT);
                let yy = f64::from(yi);
                let t1 = if yy > 0.0 { yy * (yy / p).ln() } else { 0.0 };
                let omy = 1.0 - yy;
                let t2 = if omy > 0.0 {
                    omy * (omy / (1.0 - p)).ln()
                } else {
                    0.0
                };
                Ok((g, h, f64::from(wi), f64::from(wi) * 2.0 * (t1 + t2)))
            })?;
        if sum_w <= 0.0 {
            return Err(invalid_input("logistic deviance: all-zero weights".into()));
        }
        finish_deviance("logistic", acc)
    }

    fn default_metric(&self) -> Metric {
        Metric::LogLoss
    }

    fn objective_tag(&self) -> ObjectiveTag {
        ObjectiveTag {
            link: Link::Logit,
            loss: LossId::Logistic,
            tweedie_rho: None,
        }
    }
}

/// Poisson regression for counts/frequencies (spec §05.3): Log link, `μ = exp(F)`,
/// `g = w·(μ − y)`, `h = w·μ`, `init_score = log(p̄)` (exposure-weighted form §05.5),
/// Poisson deviance. `max_delta_step = Some(0.7)` — the leaf-step cap that keeps the
/// explosive `h = exp(F)` Newton step stable (§05.6).
#[derive(Debug, Clone, Copy, Default)]
pub struct Poisson;

impl Loss for Poisson {
    fn grad_hess(
        &self,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError> {
        let n = y.len();
        require_equal_len(
            "poisson",
            "grad_hess",
            n,
            &[("raw", raw.len()), ("weight", weight.len())],
        )?;
        // Compute-bound (exp per row) ⇒ row-parallel map, bit-identical: μ=exp(F), g=w(μ−y), h=w·μ.
        let floor = self.hessian_floor();
        fill_grad_hess_parallel("poisson", out, y, raw, weight, |yi, fi, wi| {
            let mu = clamp_exp(fi);
            (wi * (mu - yi), (wi * mu).max(floor))
        })
    }

    fn supports_incremental_mu(&self) -> bool {
        true
    }

    fn refresh_mu(&self, raw: &[f32], mu: &mut [f64]) -> Result<(), PbError> {
        require_equal_len("poisson", "refresh_mu", raw.len(), &[("mu", mu.len())])?;
        // mu = clamp_exp(F) — EXACTLY the value `grad_hess` reads per row, so a refresh re-aligns the
        // cache to the exact per-round pass (bit-identical g/h at a refresh round). Sequential map.
        for (m, &fi) in mu.iter_mut().zip(raw) {
            *m = f64::from(clamp_exp(fi));
        }
        Ok(())
    }

    fn grad_hess_from_mu(
        &self,
        y: &[f32],
        mu: &[f64],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError> {
        let n = y.len();
        require_equal_len(
            "poisson",
            "grad_hess_from_mu",
            n,
            &[("mu", mu.len()), ("weight", weight.len())],
        )?;
        let floor = self.hessian_floor();
        // Reproduce `clamp_exp`'s ±30 saturation on the CACHED mu: exp is monotone, so
        // clamp(exp(F), e^{-30}, e^{30}) == exp(clamp(F, ±30)) == clamp_exp(F). The interior
        // (|F| < 30, the log-link norm) is unclamped. Then the SAME f32 (g, h) arithmetic as
        // `grad_hess`, so only the `mu` value differs from the exact pass (the drift).
        let hi = f64::from(EXP_CLAMP.exp());
        let lo = f64::from((-EXP_CLAMP).exp());
        fill_grad_hess_from_mu_parallel("poisson", out, y, mu, weight, |yi, mui, wi| {
            let mu = mui.clamp(lo, hi) as f32;
            (wi * (mu - yi), (wi * mu).max(floor))
        })
    }

    fn init_score(
        &self,
        y: &[f32],
        weight: &[f32],
        offset: Option<&[f32]>,
    ) -> Result<f64, PbError> {
        log_link_init("poisson", 1.0, y, weight, offset, |i, yi| {
            if yi < 0.0 {
                Err(invalid_input(format!(
                    "poisson: y[{i}] must be >= 0, got {yi}"
                )))
            } else {
                Ok(())
            }
        })
    }

    fn link(&self) -> Link {
        Link::Log
    }

    fn pred_from_raw(&self, raw: f32) -> f32 {
        clamp_exp(raw)
    }

    fn deviance(&self, y: &[f32], raw: &[f32], weight: &[f32]) -> Result<f32, PbError> {
        let n = y.len();
        require_equal_len(
            "poisson",
            "deviance",
            n,
            &[("raw", raw.len()), ("weight", weight.len())],
        )?;
        // Poisson unit deviance: 2 Σ w [ y ln(y/μ) − (y − μ) ]  (y ln(y/μ) → 0 at y=0).
        // Compute-bound (exp + log per row) ⇒ chunked-parallel fold (thread-count-independent).
        let (sum_w, acc) = parallel_deviance_fold(y, raw, weight, |i, yi, fi, wi| {
            require_finite("poisson", "y", i, yi)?;
            require_finite("poisson", "raw", i, fi)?;
            require_weight("poisson", i, wi)?;
            if yi < 0.0 {
                return Err(invalid_input(format!(
                    "poisson: y[{i}] must be >= 0, got {yi}"
                )));
            }
            let mu = f64::from(clamp_exp(fi));
            let yy = f64::from(yi);
            let term = if yy > 0.0 { yy * (yy / mu).ln() } else { 0.0 };
            Ok((f64::from(wi), f64::from(wi) * 2.0 * (term - (yy - mu))))
        })?;
        if sum_w <= 0.0 {
            return Err(invalid_input("poisson deviance: all-zero weights".into()));
        }
        finish_deviance("poisson", acc)
    }

    fn grad_hess_and_deviance(
        &self,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<f32, PbError> {
        require_equal_len(
            "poisson",
            "grad_hess_and_deviance",
            y.len(),
            &[("raw", raw.len()), ("weight", weight.len())],
        )?;
        // One μ=exp(F) per row, reused for (g, h) and the Poisson deviance.
        let floor = self.hessian_floor();
        let (sum_w, acc) =
            fill_grad_hess_and_fold_deviance("poisson", out, y, raw, weight, |i, yi, fi, wi| {
                require_finite("poisson", "y", i, yi)?;
                require_finite("poisson", "raw", i, fi)?;
                require_weight("poisson", i, wi)?;
                if yi < 0.0 {
                    return Err(invalid_input(format!(
                        "poisson: y[{i}] must be >= 0, got {yi}"
                    )));
                }
                let mu = clamp_exp(fi);
                let g = wi * (mu - yi);
                let h = (wi * mu).max(floor);
                let muf = f64::from(mu);
                let yy = f64::from(yi);
                let term = if yy > 0.0 { yy * (yy / muf).ln() } else { 0.0 };
                Ok((
                    g,
                    h,
                    f64::from(wi),
                    f64::from(wi) * 2.0 * (term - (yy - muf)),
                ))
            })?;
        if sum_w <= 0.0 {
            return Err(invalid_input("poisson deviance: all-zero weights".into()));
        }
        finish_deviance("poisson", acc)
    }

    fn default_metric(&self) -> Metric {
        Metric::PoissonDeviance
    }

    fn objective_tag(&self) -> ObjectiveTag {
        ObjectiveTag {
            link: Link::Log,
            loss: LossId::Poisson,
            tweedie_rho: None,
        }
    }

    fn max_delta_step(&self) -> Option<f32> {
        Some(0.7)
    }
}

/// Gamma regression for positive severities (spec §05.3): Log link, `μ = exp(F)`,
/// `g = w·(1 − y·e^{−F})`, `h = w·y·e^{−F}` (floored at the `y → 0` corner),
/// `init_score = log(p̄)` (exposure form §05.5), Gamma deviance. Strict domain `y > 0`
/// (the Gamma support excludes `0`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Gamma;

impl Loss for Gamma {
    fn grad_hess(
        &self,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError> {
        let n = y.len();
        require_equal_len(
            "gamma",
            "grad_hess",
            n,
            &[("raw", raw.len()), ("weight", weight.len())],
        )?;
        // Compute-bound (exp per row) ⇒ row-parallel map, bit-identical: t=y·e^{−F}, g=w(1−t), h=w·t.
        let floor = self.hessian_floor();
        fill_grad_hess_parallel("gamma", out, y, raw, weight, |yi, fi, wi| {
            let em = clamp_exp(-fi); // e^{−F} = y/μ factor base
            let t = yi * em; // y·e^{−F}
            (wi * (1.0 - t), (wi * t).max(floor))
        })
    }

    fn init_score(
        &self,
        y: &[f32],
        weight: &[f32],
        offset: Option<&[f32]>,
    ) -> Result<f64, PbError> {
        log_link_init("gamma", 2.0, y, weight, offset, |i, yi| {
            if yi <= 0.0 {
                Err(invalid_input(format!(
                    "gamma: y[{i}] must be > 0, got {yi}"
                )))
            } else {
                Ok(())
            }
        })
    }

    fn link(&self) -> Link {
        Link::Log
    }

    fn pred_from_raw(&self, raw: f32) -> f32 {
        clamp_exp(raw)
    }

    fn deviance(&self, y: &[f32], raw: &[f32], weight: &[f32]) -> Result<f32, PbError> {
        let n = y.len();
        require_equal_len(
            "gamma",
            "deviance",
            n,
            &[("raw", raw.len()), ("weight", weight.len())],
        )?;
        // Gamma unit deviance: 2 Σ w [ (y−μ)/μ − ln(y/μ) ] = 2 Σ w [ r − 1 − ln r ], r=y/μ.
        // Compute-bound (exp + log per row) ⇒ chunked-parallel fold (thread-count-independent).
        let (sum_w, acc) = parallel_deviance_fold(y, raw, weight, |i, yi, fi, wi| {
            require_finite("gamma", "y", i, yi)?;
            require_finite("gamma", "raw", i, fi)?;
            require_weight("gamma", i, wi)?;
            if yi <= 0.0 {
                return Err(invalid_input(format!(
                    "gamma: y[{i}] must be > 0, got {yi}"
                )));
            }
            let mu = f64::from(clamp_exp(fi));
            let r = f64::from(yi) / mu;
            Ok((f64::from(wi), f64::from(wi) * 2.0 * (r - 1.0 - r.ln())))
        })?;
        if sum_w <= 0.0 {
            return Err(invalid_input("gamma deviance: all-zero weights".into()));
        }
        finish_deviance("gamma", acc)
    }

    fn default_metric(&self) -> Metric {
        Metric::GammaDeviance
    }

    fn objective_tag(&self) -> ObjectiveTag {
        ObjectiveTag {
            link: Link::Log,
            loss: LossId::Gamma,
            tweedie_rho: None,
        }
    }

    // Log-link leaf-step cap (spec §05.3/§05.6): Gamma's Newton denominator vanishes under
    // heavy-tailed `y ≫ μ`, so it shares Poisson's stabilizing default.
    fn max_delta_step(&self) -> Option<f32> {
        Some(0.7)
    }
}

/// Tweedie compound-Poisson–Gamma regression (spec §05.3): Log link, power `ρ ∈ (1, 2)`.
/// `g = w·(−y·e^{(1−ρ)F} + e^{(2−ρ)F})`, `h = w·(−y(1−ρ)e^{(1−ρ)F} + (2−ρ)e^{(2−ρ)F})`
/// (floored), `init_score = log(p̄)` (exposure form §05.5), Tweedie deviance. `ρ` is
/// validated `∈ (1, 2)` exclusive at construction ([`Tweedie::new`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tweedie {
    rho: f32,
}

impl Default for Tweedie {
    fn default() -> Self {
        Tweedie { rho: 1.5 }
    }
}

impl Tweedie {
    /// Construct a Tweedie objective with variance power `rho`.
    ///
    /// # Errors
    /// [`PbError::InvalidConfig`] if `rho` is not finite or not in `(1, 2)` exclusive
    /// (distinct from the data-domain [`PbError::InvalidInput`] of §05.3a).
    pub fn new(rho: f32) -> Result<Self, PbError> {
        if !rho.is_finite() || rho <= 1.0 || rho >= 2.0 {
            return Err(PbError::InvalidConfig {
                what: format!("Tweedie rho must be in (1, 2) exclusive, got {rho}"),
            });
        }
        Ok(Tweedie { rho })
    }

    /// The variance power `ρ`.
    #[must_use]
    pub fn rho(&self) -> f32 {
        self.rho
    }
}

impl Loss for Tweedie {
    fn grad_hess(
        &self,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<(), PbError> {
        let n = y.len();
        require_equal_len(
            "tweedie",
            "grad_hess",
            n,
            &[("raw", raw.len()), ("weight", weight.len())],
        )?;
        // Compute-bound (two exps per row) ⇒ row-parallel map, bit-identical to the sequential
        // kernel (no fold, no drift).
        let floor = self.hessian_floor();
        let p1 = 1.0 - self.rho; // (1−ρ) < 0
        let p2 = 2.0 - self.rho; // (2−ρ) > 0
        fill_grad_hess_parallel("tweedie", out, y, raw, weight, |yi, fi, wi| {
            let a = clamp_exp(p1 * fi); // e^{(1−ρ)F} = μ^{1−ρ}
            let b = clamp_exp(p2 * fi); // e^{(2−ρ)F} = μ^{2−ρ}
            let g = wi * (-yi * a + b);
            // h = w[ −y(1−ρ)e^{(1−ρ)F} + (2−ρ)e^{(2−ρ)F} ] (the §05.3 form; here
            // `-yi * p1` with p1=(1−ρ)<0 is the equivalent y(ρ−1)·a ≥ 0), ≥ 0 for y ≥ 0.
            let h = (wi * (-yi * p1 * a + p2 * b)).max(floor);
            (g, h)
        })
    }

    fn init_score(
        &self,
        y: &[f32],
        weight: &[f32],
        offset: Option<&[f32]>,
    ) -> Result<f64, PbError> {
        log_link_init(
            "tweedie",
            f64::from(self.rho),
            y,
            weight,
            offset,
            |i, yi| {
                if yi < 0.0 {
                    Err(invalid_input(format!(
                        "tweedie: y[{i}] must be >= 0, got {yi}"
                    )))
                } else {
                    Ok(())
                }
            },
        )
    }

    fn link(&self) -> Link {
        Link::Log
    }

    fn pred_from_raw(&self, raw: f32) -> f32 {
        clamp_exp(raw)
    }

    fn deviance(&self, y: &[f32], raw: &[f32], weight: &[f32]) -> Result<f32, PbError> {
        let n = y.len();
        require_equal_len(
            "tweedie",
            "deviance",
            n,
            &[("raw", raw.len()), ("weight", weight.len())],
        )?;
        // Tweedie unit deviance:
        //   2 Σ w [ y^{2−ρ}/((1−ρ)(2−ρ)) − y·μ^{1−ρ}/(1−ρ) + μ^{2−ρ}/(2−ρ) ].
        let p1 = 1.0 - f64::from(self.rho);
        let p2 = 2.0 - f64::from(self.rho);
        let p1f = 1.0 - self.rho;
        let p2f = 2.0 - self.rho;
        // Compute-bound (two exps + a log per row) ⇒ chunked-parallel fold (thread-independent).
        let (sum_w, acc) = parallel_deviance_fold(y, raw, weight, |i, yi, fi, wi| {
            require_finite("tweedie", "y", i, yi)?;
            require_finite("tweedie", "raw", i, fi)?;
            require_weight("tweedie", i, wi)?;
            if yi < 0.0 {
                return Err(invalid_input(format!(
                    "tweedie: y[{i}] must be >= 0, got {yi}"
                )));
            }
            let yy = f64::from(yi);
            // y^{2−ρ} via exp((2−ρ)·ln y) (no powf); 0 at y = 0.
            let y_term = if yy > 0.0 { (p2 * yy.ln()).exp() } else { 0.0 };
            let mu_p1 = f64::from(clamp_exp(p1f * fi)); // μ^{1−ρ}
            let mu_p2 = f64::from(clamp_exp(p2f * fi)); // μ^{2−ρ}
            let d = y_term / (p1 * p2) - yy * mu_p1 / p1 + mu_p2 / p2;
            Ok((f64::from(wi), f64::from(wi) * 2.0 * d))
        })?;
        if sum_w <= 0.0 {
            return Err(invalid_input("tweedie deviance: all-zero weights".into()));
        }
        finish_deviance("tweedie", acc)
    }

    fn grad_hess_and_deviance(
        &self,
        y: &[f32],
        raw: &[f32],
        weight: &[f32],
        out: &mut GradHess,
    ) -> Result<f32, PbError> {
        require_equal_len(
            "tweedie",
            "grad_hess_and_deviance",
            y.len(),
            &[("raw", raw.len()), ("weight", weight.len())],
        )?;
        // The two F-exps (μ^{1−ρ}, μ^{2−ρ}) per row are reused for (g, h) AND the deviance — the
        // only extra deviance work is the y-only `y^{2−ρ}` term. Bit-identical to the separate
        // kernels (same `clamp_exp(p·fi)` ⇒ same a/b ⇒ same g/h and same mu_p1/mu_p2).
        let floor = self.hessian_floor();
        let p1f = 1.0 - self.rho; // (1−ρ) < 0, f32 (matches grad_hess + the exp arg)
        let p2f = 2.0 - self.rho; // (2−ρ) > 0, f32
        let p1d = 1.0 - f64::from(self.rho); // f64 (matches the deviance d-formula)
        let p2d = 2.0 - f64::from(self.rho);
        let (sum_w, acc) =
            fill_grad_hess_and_fold_deviance("tweedie", out, y, raw, weight, |i, yi, fi, wi| {
                require_finite("tweedie", "y", i, yi)?;
                require_finite("tweedie", "raw", i, fi)?;
                require_weight("tweedie", i, wi)?;
                if yi < 0.0 {
                    return Err(invalid_input(format!(
                        "tweedie: y[{i}] must be >= 0, got {yi}"
                    )));
                }
                let a = clamp_exp(p1f * fi); // e^{(1−ρ)F} = μ^{1−ρ}
                let b = clamp_exp(p2f * fi); // e^{(2−ρ)F} = μ^{2−ρ}
                let g = wi * (-yi * a + b);
                let h = (wi * (-yi * p1f * a + p2f * b)).max(floor);
                let yy = f64::from(yi);
                let y_term = if yy > 0.0 { (p2d * yy.ln()).exp() } else { 0.0 };
                let mu_p1 = f64::from(a);
                let mu_p2 = f64::from(b);
                let d = y_term / (p1d * p2d) - yy * mu_p1 / p1d + mu_p2 / p2d;
                Ok((g, h, f64::from(wi), f64::from(wi) * 2.0 * d))
            })?;
        if sum_w <= 0.0 {
            return Err(invalid_input("tweedie deviance: all-zero weights".into()));
        }
        finish_deviance("tweedie", acc)
    }

    fn default_metric(&self) -> Metric {
        Metric::TweedieDeviance { rho: self.rho }
    }

    fn objective_tag(&self) -> ObjectiveTag {
        ObjectiveTag {
            link: Link::Log,
            loss: LossId::Tweedie,
            tweedie_rho: Some(self.rho),
        }
    }

    // Log-link leaf-step cap (spec §05.3/§05.6): uncapped Tweedie destabilizes small
    // aggregated-exposure portfolios under heavy-tailed `y ≫ μ` (`g ∼ −w·y·μ^{1−ρ}`), so it
    // shares Poisson's stabilizing default.
    fn max_delta_step(&self) -> Option<f32> {
        Some(0.7)
    }

    // §05.6 addendum (av35): 0.7 is the right STANDING cap, but it does not stop the
    // rate-collapse blowup this objective is uniquely prone to (see `GatedDeltaStep`). The
    // gate spends a 0.3 cap only on the fits that actually collapse; a fit that never
    // trips the detector is bit-identical to the ungated engine.
    fn gated_max_delta_step(&self) -> Option<GatedDeltaStep> {
        Some(TWEEDIE_GATED_DELTA_STEP)
    }
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
    use proptest::prelude::*;

    #[test]
    fn log_link_objectives_default_to_the_stabilizing_leaf_step_cap() {
        // Spec §05.3's canonical loss table caps every log-link objective at 0.7 (one
        // shared failure mode: a near-zero log-link Newton denominator) — widened from
        // Poisson-only as a confirmed deliberate decision, 2026-07-11 review remediation;
        // see the `max_delta_step` trait-default doc and spec §05.6.
        // `Config.max_delta_step` falls back to these per-objective defaults when unset.
        assert_eq!(Poisson.max_delta_step(), Some(0.7));
        assert_eq!(Gamma.max_delta_step(), Some(0.7));
        assert_eq!(Tweedie::new(1.7).unwrap().max_delta_step(), Some(0.7));
        assert_eq!(SquaredError.max_delta_step(), None);
        assert_eq!(Logistic.max_delta_step(), None);
    }

    #[test]
    fn grad_hess_matches_closed_form() {
        let sqe = SquaredError;
        let y = [2.0_f32, 4.0];
        let raw = [3.0_f32, 3.0];
        let w = [1.0_f32, 2.0];
        let mut gh = GradHess::default();
        sqe.grad_hess(&y, &raw, &w, &mut gh).unwrap();
        // g = w(F−y): [1·(3−2), 2·(3−4)] = [1, −2]; h = w: [1, 2].
        assert_eq!(gh.g, vec![1.0, -2.0]);
        assert_eq!(gh.h, vec![1.0, 2.0]);
    }

    /// WIN #12: the fused `grad_hess_and_deviance` must be BYTE-IDENTICAL to the separate
    /// `grad_hess` then `deviance` — same `out.{g,h}` bits and same returned deviance bits — for
    /// every objective (the σ/exp-sharing overrides AND the default). `n > PAR_DEVIANCE_CHUNK`
    /// exercises the parallel chunked path, so the chunk-order deviance fold must match too.
    #[test]
    fn fused_grad_hess_and_deviance_is_bit_identical_to_separate() {
        let n = 20_000usize; // > PAR_DEVIANCE_CHUNK (8192) ⇒ multiple chunks
        let raw: Vec<f32> = (0..n).map(|i| ((i % 41) as f32 - 20.0) * 0.13).collect();
        let weight: Vec<f32> = (0..n).map(|i| 0.5 + (i % 7) as f32).collect();
        let check = |loss: &dyn Loss, y: &[f32], label: &str| {
            let mut gh_sep = GradHess::default();
            loss.grad_hess(y, &raw, &weight, &mut gh_sep).unwrap();
            let dev_sep = loss.deviance(y, &raw, &weight).unwrap();
            let mut gh_fused = GradHess::default();
            let dev_fused = loss
                .grad_hess_and_deviance(y, &raw, &weight, &mut gh_fused)
                .unwrap();
            assert_eq!(
                dev_sep.to_bits(),
                dev_fused.to_bits(),
                "{label}: deviance bits differ"
            );
            for i in 0..n {
                assert_eq!(
                    gh_sep.g[i].to_bits(),
                    gh_fused.g[i].to_bits(),
                    "{label}: g[{i}] differs"
                );
                assert_eq!(
                    gh_sep.h[i].to_bits(),
                    gh_fused.h[i].to_bits(),
                    "{label}: h[{i}] differs"
                );
            }
        };
        // SquaredError + Gamma use the default (unfused) impl; Logistic/Poisson/Tweedie override it.
        let y_sqe: Vec<f32> = (0..n).map(|i| (i % 13) as f32 - 6.0).collect();
        check(&SquaredError, &y_sqe, "squared_error");
        let y_log: Vec<f32> = (0..n).map(|i| ((i % 5) as f32) / 4.0).collect(); // y ∈ [0,1]
        check(&Logistic, &y_log, "logistic");
        let y_pois: Vec<f32> = (0..n).map(|i| (i % 4) as f32).collect(); // y ≥ 0 (incl. 0)
        check(&Poisson, &y_pois, "poisson");
        let y_gam: Vec<f32> = (0..n).map(|i| 0.25 + (i % 6) as f32).collect(); // y > 0
        check(&Gamma, &y_gam, "gamma");
        let tw = Tweedie::new(1.5).unwrap();
        let y_tw: Vec<f32> = (0..n).map(|i| (i % 3) as f32 * 0.7).collect(); // y ≥ 0
        check(&tw, &y_tw, "tweedie");
    }

    #[test]
    fn hessian_is_floored_for_zero_weight_rows() {
        let sqe = SquaredError;
        let mut gh = GradHess::default();
        sqe.grad_hess(&[1.0], &[0.0], &[0.0], &mut gh).unwrap();
        assert_eq!(gh.g, vec![0.0]); // 0·(0−1) = 0
        assert!(gh.h[0] >= sqe.hessian_floor() && gh.h[0] > 0.0);
    }

    #[test]
    fn init_score_is_the_weighted_mean() {
        let sqe = SquaredError;
        assert!(
            (sqe.init_score(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0], None)
                .unwrap()
                - 2.0)
                .abs()
                < 1e-9
        );
        // weight 0 on the middle row ⇒ mean of {1, 3}.
        assert!(
            (sqe.init_score(&[1.0, 2.0, 3.0], &[1.0, 0.0, 1.0], None)
                .unwrap()
                - 2.0)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn init_score_offset_aware_identity_form() {
        let sqe = SquaredError;
        // f0 = weighted mean of (y − offset) = mean([3−1, 5−1]) = mean([2, 4]) = 3.
        let f0 = sqe
            .init_score(&[3.0, 5.0], &[1.0, 1.0], Some(&[1.0, 1.0]))
            .unwrap();
        assert!((f0 - 3.0).abs() < 1e-9);
    }

    #[test]
    fn init_score_satisfies_first_order_condition() {
        // Σ w·g(y, f0) ≈ 0 at the intercept, with and without offset.
        let sqe = SquaredError;
        let y = [1.0_f32, 5.0, 2.0, 8.0];
        let w = [1.0_f32, 2.0, 0.5, 1.0];

        let f0 = sqe.init_score(&y, &w, None).unwrap() as f32;
        let raw: Vec<f32> = vec![f0; y.len()];
        let mut gh = GradHess::default();
        sqe.grad_hess(&y, &raw, &w, &mut gh).unwrap();
        let sum_g: f64 = gh.g.iter().map(|&g| f64::from(g)).sum();
        assert!(sum_g.abs() < 1e-4, "Σ w·g should vanish at f0, got {sum_g}");

        let off = [0.5_f32, -1.0, 2.0, 0.0];
        let f0o = sqe.init_score(&y, &w, Some(&off)).unwrap() as f32;
        let raw_o: Vec<f32> = off.iter().map(|&o| f0o + o).collect();
        sqe.grad_hess(&y, &raw_o, &w, &mut gh).unwrap();
        let sum_go: f64 = gh.g.iter().map(|&g| f64::from(g)).sum();
        assert!(
            sum_go.abs() < 1e-3,
            "Σ w·g should vanish at f0+offset, got {sum_go}"
        );
    }

    #[test]
    fn deviance_is_zero_at_perfect_fit_and_nonneg() {
        let sqe = SquaredError;
        let y = [1.0_f32, 2.0, 3.0];
        assert_eq!(sqe.deviance(&y, &y, &[1.0, 1.0, 1.0]).unwrap(), 0.0);
        let d = sqe
            .deviance(&y, &[1.5, 2.0, 2.0], &[1.0, 1.0, 1.0])
            .unwrap();
        assert!(d > 0.0);
        // ½·(0.25 + 0 + 1.0) = 0.625
        assert!((d - 0.625).abs() < 1e-6);
    }

    #[test]
    fn all_zero_weights_error_on_init_and_deviance() {
        let sqe = SquaredError;
        assert!(matches!(
            sqe.init_score(&[1.0, 2.0], &[0.0, 0.0], None),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            sqe.deviance(&[1.0, 2.0], &[1.0, 1.0], &[0.0, 0.0]),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn nonfinite_or_negative_inputs_are_invalid_input() {
        let sqe = SquaredError;
        let mut gh = GradHess::default();
        assert!(matches!(
            sqe.grad_hess(&[f32::NAN], &[1.0], &[1.0], &mut gh),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            sqe.grad_hess(&[1.0], &[f32::INFINITY], &[1.0], &mut gh),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            sqe.grad_hess(&[1.0], &[1.0], &[-1.0], &mut gh),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            sqe.init_score(&[1.0], &[1.0], Some(&[f32::NAN])),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            sqe.deviance(&[1.0], &[1.0], &[f32::INFINITY]),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn nonfinite_grad_hess_and_metric_outputs_are_invalid_input() {
        let sqe = SquaredError;
        let mut gh = GradHess::default();
        // Finite inputs can still overflow f32 arithmetic; never let inf enter Hist.
        assert!(matches!(
            sqe.grad_hess(&[-f32::MAX], &[f32::MAX], &[1.0], &mut gh),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            Poisson.grad_hess(&[0.0], &[30.0], &[f32::MAX], &mut gh),
            Err(PbError::InvalidInput { .. })
        ));
        // Deviance is returned as f32; unrepresentable finite f64 reductions are errors.
        assert!(matches!(
            sqe.deviance(&[-f32::MAX], &[f32::MAX], &[f32::MAX]),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn unequal_lengths_are_shape_mismatch_not_panic() {
        let sqe = SquaredError;
        let mut gh = GradHess::default();
        assert!(matches!(
            sqe.grad_hess(&[1.0, 2.0], &[1.0], &[1.0, 1.0], &mut gh),
            Err(PbError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            sqe.init_score(&[1.0, 2.0], &[1.0], None),
            Err(PbError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            sqe.deviance(&[1.0], &[1.0], &[1.0, 1.0]),
            Err(PbError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn empty_and_single_row_do_not_panic() {
        let sqe = SquaredError;
        let mut gh = GradHess::default();
        // n = 0: grad_hess Ok with empty buffers; init_score/deviance error (Σw = 0).
        sqe.grad_hess(&[], &[], &[], &mut gh).unwrap();
        assert!(gh.g.is_empty() && gh.h.is_empty());
        assert!(sqe.init_score(&[], &[], None).is_err());
        // n = 1.
        sqe.grad_hess(&[2.0], &[5.0], &[1.0], &mut gh).unwrap();
        assert_eq!(gh.g, vec![3.0]);
        assert_eq!(gh.h, vec![1.0]);
    }

    #[test]
    fn init_score_and_deviance_are_thread_count_independent() {
        // The folds are sequential index-order f64 reductions, so thread-count cannot
        // affect the result by construction; this asserts the §05.9#7 gate holds and
        // guards against a future parallelization regressing it.
        let sqe = SquaredError;
        let y: Vec<f32> = (0..5000).map(|i| (i % 17) as f32 * 0.5).collect();
        let raw: Vec<f32> = (0..5000).map(|i| (i % 13) as f32).collect();
        let w: Vec<f32> = (0..5000).map(|i| 1.0 + (i % 3) as f32).collect();
        let run = |n: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap();
            pool.install(|| {
                (
                    sqe.init_score(&y, &w, None).unwrap().to_bits(),
                    sqe.deviance(&y, &raw, &w).unwrap().to_bits(),
                )
            })
        };
        let a = run(1);
        assert_eq!(a, run(2));
        assert_eq!(a, run(8));
    }

    proptest! {
        // Finite-difference oracle (the primary §05.9 check). For the per-row weighted
        // half-loss L(F) = ½ w (F − y)², the central difference equals g exactly and
        // the second difference equals h exactly (it's quadratic), up to f32 rounding.
        #[test]
        fn grad_hess_matches_finite_difference(
            y in -100.0f32..100.0,
            f in -100.0f32..100.0,
            w in 0.0f32..10.0,
        ) {
            let sqe = SquaredError;
            let mut gh = GradHess::default();
            sqe.grad_hess(&[y], &[f], &[w], &mut gh).unwrap();
            let l = |ff: f64| 0.5 * f64::from(w) * (ff - f64::from(y)).powi(2);
            let delta = 0.5_f64;
            let fd_g = (l(f64::from(f) + delta) - l(f64::from(f) - delta)) / (2.0 * delta);
            let fd_h = (l(f64::from(f) + delta) - 2.0 * l(f64::from(f)) + l(f64::from(f) - delta))
                / (delta * delta);
            prop_assert!((f64::from(gh.g[0]) - fd_g).abs() <= 1e-2 + 1e-3 * fd_g.abs());
            prop_assert!((f64::from(gh.h[0]) - fd_h).abs() <= 1e-3 + 1e-3 * fd_h.abs());
        }

        // The DoD-named oracle: finite-difference the SHIPPED `deviance` w.r.t. `raw`
        // and assert it matches `grad_hess`'s g (and the 2nd difference matches h).
        // This is what ties the gradient the engine boosts on to the metric that
        // actually ships — a sign/scale disagreement between them fails here. Values
        // are kept in [-10, 10] so the f32 deviance return doesn't lose the small
        // second-difference to catastrophic cancellation.
        #[test]
        fn shipped_deviance_gradient_matches_grad_hess(
            y in -10.0f32..10.0,
            f in -10.0f32..10.0,
            w in 0.01f32..10.0,
        ) {
            let sqe = SquaredError;
            let mut gh = GradHess::default();
            sqe.grad_hess(&[y], &[f], &[w], &mut gh).unwrap();
            let dev = |ff: f32| -> f64 { f64::from(sqe.deviance(&[y], &[ff], &[w]).unwrap()) };
            let d = 1.0_f32;
            let fd_g = (dev(f + d) - dev(f - d)) / (2.0 * f64::from(d));
            let fd_h = (dev(f + d) - 2.0 * dev(f) + dev(f - d)) / f64::from(d * d);
            prop_assert!(
                (f64::from(gh.g[0]) - fd_g).abs() <= 1e-2 + 1e-3 * fd_g.abs(),
                "deviance gradient {fd_g} != grad_hess g {}", gh.g[0]
            );
            prop_assert!(
                (f64::from(gh.h[0]) - fd_h).abs() <= 1e-2 + 1e-2 * fd_h.abs(),
                "deviance curvature {fd_h} != grad_hess h {}", gh.h[0]
            );
        }
    }

    // ===================================================================
    // Phase 4 objectives: Logistic / Poisson / Gamma / Tweedie.
    // ===================================================================

    /// Central-difference oracle (§05.9 #2): `grad_hess` g/h equal the 1st/2nd central
    /// differences of the per-row loss `L(F)` (`L` includes the row weight). `F` is kept
    /// in a modest range by callers so `exp(F)` does not blow up the finite difference.
    fn assert_fd(loss: &dyn Loss, y: f32, f: f32, w: f32, l: impl Fn(f64) -> f64) {
        let mut gh = GradHess::default();
        loss.grad_hess(&[y], &[f], &[w], &mut gh).unwrap();
        let (g, h) = (f64::from(gh.g[0]), f64::from(gh.h[0]));
        let d = 1e-3_f64;
        let ff = f64::from(f);
        let fd_g = (l(ff + d) - l(ff - d)) / (2.0 * d);
        let fd_h = (l(ff + d) - 2.0 * l(ff) + l(ff - d)) / (d * d);
        assert!(
            (g - fd_g).abs() <= 1e-3 + 1e-3 * fd_g.abs(),
            "g {g} vs central-diff {fd_g}"
        );
        assert!(
            (h - fd_h).abs() <= 1e-2 + 1e-2 * fd_h.abs(),
            "h {h} vs central-diff {fd_h}"
        );
    }

    #[test]
    fn logistic_poisson_gamma_tweedie_closed_form() {
        let mut gh = GradHess::default();
        // Logistic y=1,F=0: σ=0.5 ⇒ g=−0.5, h=0.25.
        Logistic.grad_hess(&[1.0], &[0.0], &[1.0], &mut gh).unwrap();
        assert!((gh.g[0] + 0.5).abs() < 1e-6 && (gh.h[0] - 0.25).abs() < 1e-6);
        // Poisson y=2,F=0: μ=1 ⇒ g=−1, h=1.
        Poisson.grad_hess(&[2.0], &[0.0], &[1.0], &mut gh).unwrap();
        assert!((gh.g[0] + 1.0).abs() < 1e-6 && (gh.h[0] - 1.0).abs() < 1e-6);
        // Gamma y=2,F=0: e^{−F}=1 ⇒ g=1−2=−1, h=2.
        Gamma.grad_hess(&[2.0], &[0.0], &[1.0], &mut gh).unwrap();
        assert!((gh.g[0] + 1.0).abs() < 1e-6 && (gh.h[0] - 2.0).abs() < 1e-6);
        // Tweedie ρ=1.5, y=2,F=0: a=b=1 ⇒ g=−2+1=−1, h=2·0.5·1+0.5·1=1.5.
        let tw = Tweedie::new(1.5).unwrap();
        tw.grad_hess(&[2.0], &[0.0], &[1.0], &mut gh).unwrap();
        assert!((gh.g[0] + 1.0).abs() < 1e-6 && (gh.h[0] - 1.5).abs() < 1e-6);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(96))]

        #[test]
        fn logistic_grad_hess_matches_finite_difference(
            y in 0.0f32..=1.0, f in -6.0f32..6.0, w in 0.1f32..5.0,
        ) {
            // L = softplus(F) − y·F, weighted.
            let (y64, w64) = (f64::from(y), f64::from(w));
            assert_fd(&Logistic, y, f, w, |ff| w64 * ((1.0 + ff.exp()).ln() - y64 * ff));
        }

        #[test]
        fn poisson_grad_hess_matches_finite_difference(
            y in 0.0f32..50.0, f in -3.0f32..3.0, w in 0.1f32..5.0,
        ) {
            // L = exp(F) − y·F.
            let (y64, w64) = (f64::from(y), f64::from(w));
            assert_fd(&Poisson, y, f, w, |ff| w64 * (ff.exp() - y64 * ff));
        }

        #[test]
        fn gamma_grad_hess_matches_finite_difference(
            y in 0.01f32..50.0, f in -3.0f32..3.0, w in 0.1f32..5.0,
        ) {
            // L = y·exp(−F) + F.
            let (y64, w64) = (f64::from(y), f64::from(w));
            assert_fd(&Gamma, y, f, w, |ff| w64 * (y64 * (-ff).exp() + ff));
        }

        #[test]
        fn tweedie_grad_hess_matches_finite_difference(
            y in 0.0f32..30.0, f in -3.0f32..3.0, w in 0.1f32..5.0, rho in 1.05f32..1.95,
        ) {
            // L = −y·exp((1−ρ)F)/(1−ρ) + exp((2−ρ)F)/(2−ρ).
            let tw = Tweedie::new(rho).unwrap();
            let (y64, w64) = (f64::from(y), f64::from(w));
            let (p1, p2) = (1.0 - f64::from(rho), 2.0 - f64::from(rho));
            assert_fd(&tw, y, f, w, |ff| {
                w64 * (-y64 * (p1 * ff).exp() / p1 + (p2 * ff).exp() / p2)
            });
        }
    }

    #[test]
    fn init_score_first_order_condition_per_objective() {
        // Σ w·g(y, f0) ≈ 0 at the boost-from-average intercept, for every objective.
        let w = [1.0_f32, 2.0, 0.5, 1.5, 1.0];
        let cases: Vec<(&dyn Loss, Vec<f32>)> = vec![
            (&Logistic, vec![1.0, 0.0, 1.0, 0.0, 1.0]),
            (&Poisson, vec![0.0, 3.0, 1.0, 5.0, 2.0]),
            (&Gamma, vec![1.0, 3.0, 0.5, 5.0, 2.0]),
        ];
        let tw = Tweedie::new(1.5).unwrap();
        let mut all = cases;
        all.push((&tw, vec![0.0, 3.0, 1.0, 5.0, 2.0]));
        for (loss, y) in all {
            let f0 = loss.init_score(&y, &w, None).unwrap() as f32;
            let raw = vec![f0; y.len()];
            let mut gh = GradHess::default();
            loss.grad_hess(&y, &raw, &w, &mut gh).unwrap();
            let sum_g: f64 = gh.g.iter().map(|&g| f64::from(g)).sum();
            assert!(
                sum_g.abs() < 1e-3,
                "Σ w·g should vanish at f0 for {:?}, got {sum_g}",
                loss.objective_tag().loss
            );
        }
    }

    #[test]
    fn poisson_exposure_init_first_order_and_base_level() {
        // The exposure-weighted Poisson intercept (§05.5): f0 = log(Σwy / Σwe). With a
        // flat exposure e=1 (offset=0) the base level is e⁰ = 1.000, i.e. exp(f0) = p̄.
        let y = [0.0_f32, 2.0, 1.0, 3.0];
        let w = [1.0_f32, 1.0, 1.0, 1.0];
        let off = [0.0_f32; 4]; // e_i = 1
        let f0 = Poisson.init_score(&y, &w, Some(&off)).unwrap();
        let pbar = (0.0 + 2.0 + 1.0 + 3.0) / 4.0_f64;
        assert!(
            (f0.exp() - pbar).abs() < 1e-9,
            "exp(f0) {} != p̄ {pbar}",
            f0.exp()
        );
        // First-order condition with non-flat exposure (Poisson form is exact).
        let e = [1.0_f32, 2.0, 0.5, 1.5];
        let off2: Vec<f32> = e.iter().map(|v| v.ln()).collect();
        let f0b = Poisson.init_score(&y, &w, Some(&off2)).unwrap() as f32;
        let raw: Vec<f32> = off2.iter().map(|&o| f0b + o).collect();
        let mut gh = GradHess::default();
        Poisson.grad_hess(&y, &raw, &w, &mut gh).unwrap();
        let sum_g: f64 = gh.g.iter().map(|&g| f64::from(g)).sum();
        assert!(
            sum_g.abs() < 1e-3,
            "exposure-weighted Σ w·g should vanish, got {sum_g}"
        );
    }

    #[test]
    fn log_link_deviance_is_zero_at_perfect_fit_and_nonneg() {
        // Log-link: μ = y at raw = ln(y) (y>0). Logistic: p = y at raw = logit(y).
        let yp = [1.0_f32, 2.0, 3.0];
        let w = [1.0_f32, 1.0, 1.0];
        let lnln: Vec<f32> = yp.iter().map(|v| v.ln()).collect();
        for loss in [&Poisson as &dyn Loss, &Gamma, &Tweedie::new(1.5).unwrap()] {
            let d0 = loss.deviance(&yp, &lnln, &w).unwrap();
            assert!(d0.abs() < 1e-4, "deviance at μ=y should be ~0, got {d0}");
            let bad: Vec<f32> = yp.iter().map(|v| v.ln() + 0.5).collect();
            assert!(loss.deviance(&yp, &bad, &w).unwrap() > 0.0);
        }
        // Logistic with soft labels: p = y at raw = logit(y).
        let ys = [0.2_f32, 0.5, 0.8];
        let logit: Vec<f32> = ys.iter().map(|&v| (v / (1.0 - v)).ln()).collect();
        let d0 = Logistic.deviance(&ys, &logit, &w).unwrap();
        assert!(
            d0.abs() < 1e-4,
            "logistic deviance at p=y should be ~0, got {d0}"
        );
        assert!(Logistic.deviance(&ys, &[0.0, 0.0, 0.0], &w).unwrap() > 0.0);
    }

    #[test]
    fn domain_errors_per_objective() {
        let w = [1.0_f32, 1.0];
        // Logistic: y outside [0,1]; single-class; all-zero weights.
        assert!(matches!(
            Logistic.init_score(&[0.0, 2.0], &w, None),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            Logistic.init_score(&[0.0, 0.0], &w, None),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            Logistic.init_score(&[1.0, 1.0], &w, None),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            Logistic.init_score(&[0.0, 1.0], &[0.0, 0.0], None),
            Err(PbError::InvalidInput { .. })
        ));
        // Poisson: y<0; all-zero weights. (All-zero y is CLAMPED, not rejected.)
        assert!(matches!(
            Poisson.init_score(&[-1.0, 1.0], &w, None),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(Poisson
            .init_score(&[0.0, 0.0], &w, None)
            .unwrap()
            .is_finite());
        // Gamma: strict y>0 (a single 0 rejects); all-zero weights.
        assert!(matches!(
            Gamma.init_score(&[0.0, 1.0], &w, None),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            Gamma.deviance(&[0.0, 1.0], &[0.0, 0.0], &w),
            Err(PbError::InvalidInput { .. })
        ));
        // Tweedie: y<0; bad rho is InvalidConfig (construction), distinct from data domain.
        let tw = Tweedie::new(1.5).unwrap();
        assert!(matches!(
            tw.init_score(&[-1.0, 1.0], &w, None),
            Err(PbError::InvalidInput { .. })
        ));
        for bad in [1.0f32, 2.0, 0.5, 2.5, f32::NAN] {
            assert!(
                matches!(Tweedie::new(bad), Err(PbError::InvalidConfig { .. })),
                "rho {bad}"
            );
        }
        // Non-positive exposure via a −inf offset is rejected (not −inf into f0).
        assert!(matches!(
            Poisson.init_score(&[1.0, 2.0], &w, Some(&[f32::NEG_INFINITY, 0.0])),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn saturation_and_corners_stay_finite_and_floored() {
        let mut gh = GradHess::default();
        // Logistic F=±40: σ saturates, h floored, all finite.
        Logistic
            .grad_hess(&[1.0, 0.0], &[40.0, -40.0], &[1.0, 1.0], &mut gh)
            .unwrap();
        assert!(
            gh.g.iter().all(|g| g.is_finite())
                && gh
                    .h
                    .iter()
                    .all(|&h| h.is_finite() && h >= Logistic.hessian_floor())
        );
        // Poisson huge F: μ clamped, finite.
        Poisson
            .grad_hess(&[1.0], &[1000.0], &[1.0], &mut gh)
            .unwrap();
        assert!(gh.g[0].is_finite() && gh.h[0].is_finite());
        // Gamma y=0 corner: t=0 ⇒ g=w, h floored.
        Gamma.grad_hess(&[0.0], &[5.0], &[1.0], &mut gh).unwrap();
        assert!(
            (gh.g[0] - 1.0).abs() < 1e-6 && gh.h[0] >= Gamma.hessian_floor() && gh.h[0].is_finite()
        );
        // Tweedie y=0: h = w·(2−ρ)·e^{(2−ρ)F} ≥ 0, finite.
        let tw = Tweedie::new(1.5).unwrap();
        tw.grad_hess(&[0.0], &[3.0], &[1.0], &mut gh).unwrap();
        assert!(gh.g[0].is_finite() && gh.h[0].is_finite() && gh.h[0] >= 0.0);
    }

    #[test]
    fn no_powf_on_the_shipped_loss_path() {
        // §05.4: powers are exp(k·F), never powf/powi on the objective path. Build the
        // needles from pieces so this test's own source does not contain the literals.
        let src = include_str!("loss.rs");
        let shipped = src.split("#[cfg(test)]").next().unwrap_or(src);
        let pf = [".", "pow", "f", "("].concat();
        let pi = [".", "pow", "i", "("].concat();
        assert!(
            !shipped.contains(&pf),
            "shipped loss code must not call powf"
        );
        assert!(
            !shipped.contains(&pi),
            "shipped loss code must not call powi"
        );
    }

    #[test]
    fn log_link_init_and_deviance_are_thread_count_independent() {
        // The f64 folds are sequential index-order, so they are thread-count invariant;
        // this guards a future parallelization from regressing the §05.9 #7 gate.
        let y: Vec<f32> = (0..4000).map(|i| (i % 7) as f32).collect();
        let raw: Vec<f32> = (0..4000).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
        let w: Vec<f32> = (0..4000).map(|i| 1.0 + (i % 3) as f32).collect();
        let run = |n: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap();
            pool.install(|| {
                (
                    Poisson.init_score(&y, &w, None).unwrap().to_bits(),
                    Poisson.deviance(&y, &raw, &w).unwrap().to_bits(),
                )
            })
        };
        let a = run(1);
        assert_eq!(a, run(2));
        assert_eq!(a, run(8));
    }

    #[test]
    fn log_link_deviance_parallel_path_is_thread_count_independent() {
        // The log-link deviance is now a CHUNKED-parallel fold (fixed chunk boundaries combined in
        // chunk order). Exercise the PARALLEL path (n > PAR_DEVIANCE_CHUNK) and pin the f32 result
        // bit-for-bit across 1/2/8 threads — the §05.9 #7 gate the leaf-refine line search relies on.
        let n = PAR_DEVIANCE_CHUNK * 3 + 91;
        let raw: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let w: Vec<f32> = (0..n).map(|i| 1.0 + (i % 4) as f32).collect();
        let y_count: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect(); // >= 0
        let y_pos: Vec<f32> = (0..n).map(|i| 0.25 + (i % 5) as f32).collect(); // > 0
        let y_bin: Vec<f32> = (0..n).map(|i| (i % 2) as f32).collect(); // {0,1}
        let cases: Vec<(&str, &dyn Loss, &[f32])> = vec![
            ("logistic", &Logistic, &y_bin),
            ("poisson", &Poisson, &y_count),
            ("gamma", &Gamma, &y_pos),
        ];
        for (name, loss, yy) in cases {
            let run = |threads: usize| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                pool.install(|| loss.deviance(yy, &raw, &w).unwrap().to_bits())
            };
            let base = run(1);
            assert_eq!(base, run(2), "{name} deviance differs at 2 threads");
            assert_eq!(base, run(8), "{name} deviance differs at 8 threads");
        }
        let tw = Tweedie::new(1.5).unwrap();
        let run_tw = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| tw.deviance(&y_count, &raw, &w).unwrap().to_bits())
        };
        let base = run_tw(1);
        assert_eq!(base, run_tw(2), "tweedie deviance differs at 2 threads");
        assert_eq!(base, run_tw(8), "tweedie deviance differs at 8 threads");
    }

    #[test]
    fn log_link_grad_hess_parallel_path_is_bit_identical_across_thread_counts() {
        // The log-link grad_hess is a row-parallel MAP (independent per-row writes) ⇒ bit-identical
        // to the sequential loop regardless of thread count (no fold, no drift). Exercise the
        // parallel path (n > PAR_DEVIANCE_CHUNK) and pin every g/h cell across 1/2/8 threads.
        let n = PAR_DEVIANCE_CHUNK * 3 + 53;
        let raw: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let w: Vec<f32> = (0..n).map(|i| 1.0 + (i % 4) as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.5).collect();
        let y_pos: Vec<f32> = (0..n).map(|i| 0.25 + (i % 5) as f32).collect();
        let losses: Vec<(&str, Box<dyn Loss>)> = vec![
            ("logistic", Box::new(Logistic)),
            ("poisson", Box::new(Poisson)),
            ("gamma", Box::new(Gamma)),
            ("tweedie", Box::new(Tweedie::new(1.5).unwrap())),
        ];
        for (name, loss) in &losses {
            let yy: &[f32] = if *name == "gamma" { &y_pos } else { &y };
            let run = |threads: usize| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                pool.install(|| {
                    let mut gh = GradHess::default();
                    loss.grad_hess(yy, &raw, &w, &mut gh).unwrap();
                    let g: Vec<u32> = gh.g.iter().map(|v| v.to_bits()).collect();
                    let h: Vec<u32> = gh.h.iter().map(|v| v.to_bits()).collect();
                    (g, h)
                })
            };
            let base = run(1);
            assert_eq!(base, run(2), "{name} grad_hess differs at 2 threads");
            assert_eq!(base, run(8), "{name} grad_hess differs at 8 threads");
        }
    }
}
