// BUG-001 harness. `sample_rows` and the `SampledRows` alias are copied VERBATIM from
// crates/t-boost-core/src/engine/boost.rs lines 5347-5407 (private items); pb_seed / Stage /
// Sampling / GradHess / PbError are the crate's public exports.
use t_boost_core::engine::Sampling;
use t_boost_core::loss::GradHess;
use t_boost_core::{pb_seed, PbError, Stage};

type SampledRows<'a> = (std::borrow::Cow<'a, [u32]>, Option<Vec<f64>>);

fn sample_rows<'a>(
    sampling: &Sampling,
    gh: &GradHess,
    seed: u64,
    round: u32,
    all_rows: &'a [u32],
) -> Result<SampledRows<'a>, PbError> {
    match *sampling {
        Sampling::Full => Ok((std::borrow::Cow::Borrowed(all_rows), None)),
        Sampling::Mvs { rate, min_rows } => {
            let n = all_rows.len();
            if n == 0 {
                return Ok((std::borrow::Cow::Borrowed(all_rows), None));
            }
            let target = ((n as f64) * f64::from(rate)).ceil() as usize;
            let min_rows = usize::try_from(min_rows).map_err(|_| PbError::Internal {
                what: "MVS min_rows exceeded usize".into(),
            })?;
            let k = target.max(min_rows).min(n).max(1);
            if k == n {
                // Every row included ⇒ p_i = 1 everywhere, no reweighting needed.
                return Ok((std::borrow::Cow::Borrowed(all_rows), None));
            }
            let mut keyed: Vec<(f64, u32, f64)> = Vec::with_capacity(n);
            let mut total_weight = 0.0_f64;
            for (pos, &row) in all_rows.iter().enumerate() {
                let ru = row as usize;
                let g = f64::from(*gh.g.get(ru).ok_or_else(|| PbError::Internal {
                    what: "MVS row escaped gradients".into(),
                })?);
                let h = f64::from(*gh.h.get(ru).ok_or_else(|| PbError::Internal {
                    what: "MVS row escaped hessians".into(),
                })?);
                let s = (g * g + h * h).sqrt().max(1e-12);
                total_weight += s;
                let block = u32::try_from(pos).map_err(|_| PbError::InvalidInput {
                    what: "MVS sampling supports at most u32::MAX rows".into(),
                })?;
                let bits = pb_seed(seed, round, Stage::Sample as u32, block);
                let unit = ((bits >> 11) as f64 + 1.0) / ((1_u64 << 53) as f64 + 1.0);
                // Efraimidis-Spirakis PPS-without-replacement key. Larger is better
                // (`ln(unit)` is negative; dividing by a larger gradient weight moves
                // it closer to zero).
                keyed.push((unit.ln() / s, row, s));
            }
            keyed.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            keyed.truncate(k);
            // Restore ascending row-id order (row ids are unique ⇒ unstable sort is fine).
            keyed.sort_unstable_by_key(|&(_, row, _)| row);
            // §06.5: p_i = min(1, s_i/μ), the size-biased inclusion probability, with the
            // threshold μ = W/k chosen so Σ p_i ≈ k (W = Σ s_i over the population sampled
            // from). 1/p_i = max(1, μ/s_i): rows whose own weight already clears the threshold
            // (would be selected regardless) are left unscaled; the rest are upweighted to
            // compensate for their lower selection chance, keeping the sampled Newton gain an
            // unbiased estimator of the full-data gain.
            let mu = total_weight.max(f64::EPSILON) / (k as f64);
            let reweight: Vec<f64> = keyed.iter().map(|&(_, _, s)| (mu / s).max(1.0)).collect();
            let rows: Vec<u32> = keyed.into_iter().map(|(_, row, _)| row).collect();
            Ok((std::borrow::Cow::Owned(rows), Some(reweight)))
        }
    }
}

fn main() {
    let gh = GradHess {
        g: vec![100.0, 0.0, 0.0],
        h: vec![1.0, 1.0, 1.0],
    };
    let all_rows: Vec<u32> = vec![0, 1, 2];
    let sampling = Sampling::Mvs {
        rate: 0.5,
        min_rows: 1,
    };
    let (rows, rw) = sample_rows(&sampling, &gh, 0, 0, &all_rows).unwrap();
    let rw = rw.unwrap();
    let est_h: f64 = rows
        .iter()
        .zip(&rw)
        .map(|(&r, &m)| f64::from(gh.h[r as usize]) * m)
        .sum();
    println!("seed 0: rows = {:?}", rows);
    println!("multipliers = {:?}", rw);
    println!("estimated Hessian = {est_h} (population Hessian = 3)");

    // Claimed closed-form inclusion probabilities p_i = min(1, s_i/mu): do they sum to k=2?
    let s: Vec<f64> = gh
        .g
        .iter()
        .zip(&gh.h)
        .map(|(&g, &h)| (f64::from(g) * f64::from(g) + f64::from(h) * f64::from(h)).sqrt())
        .collect();
    let w: f64 = s.iter().sum();
    let mu = w / 2.0;
    let p: Vec<f64> = s.iter().map(|&si| (si / mu).min(1.0)).collect();
    println!(
        "closed-form p_i = {:?}, sum = {} (comment claims ≈ k = 2)",
        p,
        p.iter().sum::<f64>()
    );

    for (label, sweep_seed) in [("seed sweep (round=0)", true), ("round sweep (seed=0)", false)] {
        let mut counts = [0usize; 3];
        let mut sum_h = 0.0_f64;
        let mut sum_g = 0.0_f64;
        let n_trials = 10_000u64;
        for t in 0..n_trials {
            let (seed, round) = if sweep_seed { (t, 0u32) } else { (0u64, t as u32) };
            let (rows, rw) = sample_rows(&sampling, &gh, seed, round, &all_rows).unwrap();
            let rw = rw.unwrap();
            for (&r, &m) in rows.iter().zip(&rw) {
                counts[r as usize] += 1;
                sum_h += f64::from(gh.h[r as usize]) * m;
                sum_g += f64::from(gh.g[r as usize]) * m;
            }
        }
        println!(
            "{label}: inclusion counts = {:?}, mean estimated Hessian = {}, mean estimated gradient = {} (population g = 100)",
            counts,
            sum_h / n_trials as f64,
            sum_g / n_trials as f64
        );
        println!(
            "  empirical inclusion freq = [{:.4}, {:.4}, {:.4}]  vs claimed p_i = [{:.4}, {:.4}, {:.4}]",
            counts[0] as f64 / n_trials as f64,
            counts[1] as f64 / n_trials as f64,
            counts[2] as f64 / n_trials as f64,
            p[0],
            p[1],
            p[2]
        );
    }
}
