//! Export-time re-decomposition of a purified [`TableBank`] from the product reference
//! measure into Hooker's hierarchically-orthogonal decomposition under the joint exposure
//! measure, for orders one and two only.
//!
//! # Background and Motivation
//!
//! Under the product reference measure (e.g. [`RefMeasure::ProductMarginals`]), a pair table's
//! slice means are taken over the other feature's marginal distribution. This integrates over
//! feature combinations that may never occur in reality (e.g. driver age 18 with a 30-year licence).
//! Under the joint measure, the main effect at age 18 averages only over the licence lengths
//! that 18-year-olds actually have.
//!
//! This re-decomposition never changes the mathematical function that the bank computes:
//! the cell-wise sum of all tables plus the intercept is preserved to floating-point rounding.
//! Only the allocation of shared mass between pair tables and their constituent main effects
//! is altered.
//!
//! Dense tables of order 3 and above, as well as high-order factored effects in
//! [`TableBank::factored`], are left unchanged. The resulting bank is thus a hybrid:
//! decomposed under the joint measure at orders 1 and 2, and under the product measure above.

use crate::data::FeatureId;
use crate::error::{Invariant, PbError};
use crate::explain::{RefMeasure, TableBank, Tensor};
use std::collections::BTreeMap;

/// Options for [`rejoint`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointOptions {
    /// Ridge shrinkage of the projected main-effect pieces toward zero (the product-measure
    /// solution), in units of the data term (the joint weights sum to one). Default 1e-3.
    pub ridge: f64,
    /// Positivity floor: each joint cell weight gets `floor * m_i(c_i) * m_j(c_j)` added
    /// (product of normalized marginals), so the projection is well-posed on cells with no
    /// joint support. Default 1e-6.
    pub floor: f64,
}

impl Default for JointOptions {
    fn default() -> Self {
        Self {
            ridge: 1e-3,
            floor: 1e-6,
        }
    }
}

/// Re-express `bank` under the joint exposure measure. See module docs.
///
/// # Errors
/// [`PbError::InvalidConfig`] if options are non-finite or negative;
/// [`PbError::InvalidInput`] if a pair table's feature has no order-1 table in the bank;
/// [`PbError::ShapeMismatch`] if table dimensions disagree with marginal extents;
/// [`PbError::Internal`] if projection is singular or tensor manipulation fails.
pub fn rejoint(bank: &TableBank, opts: &JointOptions) -> Result<TableBank, PbError> {
    if !opts.ridge.is_finite() || opts.ridge < 0.0 {
        return Err(PbError::InvalidConfig {
            what: "JointOptions ridge must be finite and >= 0".into(),
        });
    }
    if !opts.floor.is_finite() || opts.floor < 0.0 {
        return Err(PbError::InvalidConfig {
            what: "JointOptions floor must be finite and >= 0".into(),
        });
    }

    if bank
        .tables
        .iter()
        .any(|t| t.axes.iter().any(|a| a.band_of.is_some()))
    {
        return Err(PbError::InvalidInput {
            what: "the joint-measure ledger does not yet support banded tables; export under a \
                   product measure, or fit with band_tolerance=None"
                .into(),
        });
    }

    // Step 1: Marginal mass.
    // Collect per-feature main-effect index in bank.tables and normalized marginal mass m_i.
    let mut main_effects: BTreeMap<FeatureId, (usize, Vec<f64>)> = BTreeMap::new();
    for (idx, table) in bank.tables.iter().enumerate() {
        if table.u.order() == 1 {
            let feat = *table.u.0.first().ok_or_else(|| PbError::Internal {
                what: "order-1 table has empty feature set".into(),
            })?;
            let shape = table.support.shape();
            let n_cells = *shape.first().ok_or_else(|| PbError::Internal {
                what: "order-1 table has empty shape".into(),
            })?;
            let mut m_i = Vec::with_capacity(n_cells);
            let mut t_i = 0.0_f64;
            for c in 0..n_cells {
                let mass = table.support.at(&[c]).ok_or_else(|| PbError::Internal {
                    what: format!("missing support at cell {c} for feature {feat:?}"),
                })?;
                t_i += mass;
                m_i.push(mass);
            }
            if t_i == 0.0 || !t_i.is_finite() {
                let unif = 1.0_f64 / (n_cells as f64);
                m_i.fill(unif);
            } else {
                for slot in &mut m_i {
                    *slot /= t_i;
                }
            }
            main_effects.insert(feat, (idx, m_i));
        }
    }

    // Downward closure check: every feature in every order-2 table must have an order-1 table.
    for table in &bank.tables {
        if table.u.order() == 2 {
            for feat in &table.u.0 {
                if !main_effects.contains_key(feat) {
                    return Err(PbError::InvalidInput {
                        what: format!(
                            "order-2 table {:?} contains feature {:?} with no order-1 table in bank",
                            table.u, feat
                        ),
                    });
                }
            }
        }
    }

    // Clone tables to mutate in stored order.
    let mut tables = bank.tables.clone();
    let mut f0 = bank.f0;

    // Step 2: For every order-2 table u = {i, j} (i < j as stored).
    for pair_idx in 0..tables.len() {
        let is_pair = tables
            .get(pair_idx)
            .map(|t| t.u.order() == 2)
            .unwrap_or(false);
        if !is_pair {
            continue;
        }

        let (feat_i, feat_j, u_repr) = {
            let table = tables.get(pair_idx).ok_or_else(|| PbError::Internal {
                what: "pair table index".into(),
            })?;
            let fi = *table.u.0.first().ok_or_else(|| PbError::Internal {
                what: "pair table axis 0".into(),
            })?;
            let fj = *table.u.0.get(1).ok_or_else(|| PbError::Internal {
                what: "pair table axis 1".into(),
            })?;
            (fi, fj, table.u.clone())
        };

        let (main_idx_i, m_i) =
            main_effects
                .get(&feat_i)
                .cloned()
                .ok_or_else(|| PbError::InvalidInput {
                    what: format!("missing main effect for feature {feat_i:?}"),
                })?;
        let (main_idx_j, m_j) =
            main_effects
                .get(&feat_j)
                .cloned()
                .ok_or_else(|| PbError::InvalidInput {
                    what: format!("missing main effect for feature {feat_j:?}"),
                })?;

        let (n_i, n_j, a, b, r_values, r_variance, c_shift) = {
            let table = tables.get(pair_idx).ok_or_else(|| PbError::Internal {
                what: "pair table index".into(),
            })?;
            let shape = table.values.shape();
            let ni = *shape.first().ok_or_else(|| PbError::Internal {
                what: "pair table axis 0 extent".into(),
            })?;
            let nj = *shape.get(1).ok_or_else(|| PbError::Internal {
                what: "pair table axis 1 extent".into(),
            })?;

            if m_i.len() != ni || m_j.len() != nj {
                return Err(PbError::ShapeMismatch {
                    what: format!(
                        "table {u_repr:?} shape ({ni}, {nj}) does not match marginal extents ({}, {})",
                        m_i.len(),
                        m_j.len()
                    ),
                });
            }

            // Total support T = Σ S
            let mut t_sum = 0.0_f64;
            for ci in 0..ni {
                for cj in 0..nj {
                    let s_val = table
                        .support
                        .at(&[ci, cj])
                        .ok_or_else(|| PbError::Internal {
                            what: "pair table support coord out of range".into(),
                        })?;
                    t_sum += s_val;
                }
            }

            let cell_count = ni.checked_mul(nj).ok_or_else(|| PbError::Internal {
                what: "cell count overflow".into(),
            })?;
            let mut s = Vec::with_capacity(cell_count);

            if t_sum == 0.0 || !t_sum.is_finite() {
                for ci in 0..ni {
                    let mi = *m_i.get(ci).ok_or_else(|| PbError::Internal {
                        what: "m_i index".into(),
                    })?;
                    for cj in 0..nj {
                        let mj = *m_j.get(cj).ok_or_else(|| PbError::Internal {
                            what: "m_j index".into(),
                        })?;
                        s.push(mi * mj);
                    }
                }
            } else {
                for ci in 0..ni {
                    let mi = *m_i.get(ci).ok_or_else(|| PbError::Internal {
                        what: "m_i index".into(),
                    })?;
                    for cj in 0..nj {
                        let mj = *m_j.get(cj).ok_or_else(|| PbError::Internal {
                            what: "m_j index".into(),
                        })?;
                        let s_val =
                            table
                                .support
                                .at(&[ci, cj])
                                .ok_or_else(|| PbError::Internal {
                                    what: "pair table support coord out of range".into(),
                                })?;
                        s.push(s_val / t_sum + opts.floor * mi * mj);
                    }
                }
            }

            // Renormalize s to sum to one
            let s_total: f64 = s.iter().copied().sum();
            if s_total == 0.0 || !s_total.is_finite() {
                return Err(PbError::Internal {
                    what: format!("joint weights sum to zero for table {u_repr:?}"),
                });
            }
            for slot in &mut s {
                *slot /= s_total;
            }

            // Unknowns are the cells that carry marginal mass. A cell no row occupies on an
            // axis has nothing to move into it and would make the system singular (its row
            // and column are zero); it keeps `a = 0` / `b = 0` and the pair residual keeps
            // whatever value the product ledger left there.
            let act_i: Vec<usize> = (0..ni)
                .filter(|&c| m_i.get(c).is_some_and(|&m| m > 0.0))
                .collect();
            let act_j: Vec<usize> = (0..nj)
                .filter(|&c| m_j.get(c).is_some_and(|&m| m > 0.0))
                .collect();
            let na = act_i.len();
            let nb = act_j.len();
            // Assemble the KKT system: unknowns `[a (na) | b (nb) | c | λ_a | λ_b]`, where `c`
            // is the pair's joint-weighted constant. The projection target is span{1, V_i,
            // V_j}: under the product measure the pair had zero mean, but under the joint
            // measure it need not, and that constant belongs to the intercept, not to a
            // zero-mean main effect (without it the mains cannot absorb a level shift).
            let kkt_n = na
                .checked_add(nb)
                .and_then(|v| v.checked_add(3))
                .ok_or_else(|| PbError::Internal {
                    what: "KKT size overflow".into(),
                })?;
            let ic = na + nb;
            let ila = ic + 1;
            let ilb = ic + 2;
            let mut m_mat = vec![vec![0.0_f64; kkt_n]; kkt_n];
            let mut rhs = vec![0.0_f64; kkt_n];
            let set =
                |mat: &mut Vec<Vec<f64>>, r: usize, c: usize, v: f64| -> Result<(), PbError> {
                    let slot = mat
                        .get_mut(r)
                        .and_then(|row| row.get_mut(c))
                        .ok_or_else(|| PbError::Internal {
                            what: "matrix access".into(),
                        })?;
                    *slot = v;
                    Ok(())
                };
            let at_a = |ci: usize| -> Result<f64, PbError> {
                m_i.get(ci).copied().ok_or_else(|| PbError::Internal {
                    what: "m_i index".into(),
                })
            };
            let at_b = |cj: usize| -> Result<f64, PbError> {
                m_j.get(cj).copied().ok_or_else(|| PbError::Internal {
                    what: "m_j index".into(),
                })
            };
            let a_val = |ci: usize, cj: usize| -> Result<f64, PbError> {
                table.values.at(&[ci, cj]).ok_or_else(|| PbError::Internal {
                    what: "A coord".into(),
                })
            };
            let mut s_all = 0.0_f64;
            let mut rhs_c = 0.0_f64;
            // Rows for a(ci), ci active.
            for (ia, &ci) in act_i.iter().enumerate() {
                let mut sum_s = 0.0_f64;
                let mut rhs_ci = 0.0_f64;
                for (jb, &cj) in act_j.iter().enumerate() {
                    let s_ij = get_s(&s, nj, ci, cj)?;
                    sum_s += s_ij;
                    rhs_ci += s_ij * a_val(ci, cj)?;
                    set(&mut m_mat, ia, na + jb, s_ij)?;
                }
                set(&mut m_mat, ia, ia, sum_s + opts.ridge * at_a(ci)?)?;
                set(&mut m_mat, ia, ic, sum_s)?;
                set(&mut m_mat, ic, ia, sum_s)?;
                set(&mut m_mat, ia, ila, at_a(ci)?)?;
                set(&mut m_mat, ila, ia, at_a(ci)?)?;
                s_all += sum_s;
                rhs_c += rhs_ci;
                *rhs.get_mut(ia).ok_or_else(|| PbError::Internal {
                    what: "rhs access".into(),
                })? = rhs_ci;
            }
            // Rows for b(cj), cj active.
            for (jb, &cj) in act_j.iter().enumerate() {
                let r_idx = na + jb;
                let mut sum_s = 0.0_f64;
                let mut rhs_cj = 0.0_f64;
                for (ia, &ci) in act_i.iter().enumerate() {
                    let s_ij = get_s(&s, nj, ci, cj)?;
                    sum_s += s_ij;
                    rhs_cj += s_ij * a_val(ci, cj)?;
                    set(&mut m_mat, r_idx, ia, s_ij)?;
                }
                set(&mut m_mat, r_idx, r_idx, sum_s + opts.ridge * at_b(cj)?)?;
                set(&mut m_mat, r_idx, ic, sum_s)?;
                set(&mut m_mat, ic, r_idx, sum_s)?;
                set(&mut m_mat, r_idx, ilb, at_b(cj)?)?;
                set(&mut m_mat, ilb, r_idx, at_b(cj)?)?;
                *rhs.get_mut(r_idx).ok_or_else(|| PbError::Internal {
                    what: "rhs access".into(),
                })? = rhs_cj;
            }
            // Row for the constant: Σ s (A − c − a − b) = 0.
            set(&mut m_mat, ic, ic, s_all)?;
            *rhs.get_mut(ic).ok_or_else(|| PbError::Internal {
                what: "rhs access".into(),
            })? = rhs_c;

            // Solve via Gaussian elimination with partial pivoting
            let x_sol = solve_gaussian_elimination(&mut m_mat, &mut rhs, &u_repr)?;
            let c_shift = *x_sol.get(ic).ok_or_else(|| PbError::Internal {
                what: "solution index".into(),
            })?;

            let mut a = vec![0.0_f64; ni];
            for (ia, &ci) in act_i.iter().enumerate() {
                *a.get_mut(ci).ok_or_else(|| PbError::Internal {
                    what: "a index".into(),
                })? = *x_sol.get(ia).ok_or_else(|| PbError::Internal {
                    what: "solution index".into(),
                })?;
            }
            let mut b = vec![0.0_f64; nj];
            for (jb, &cj) in act_j.iter().enumerate() {
                *b.get_mut(cj).ok_or_else(|| PbError::Internal {
                    what: "b index".into(),
                })? = *x_sol.get(na + jb).ok_or_else(|| PbError::Internal {
                    what: "solution index".into(),
                })?;
            }

            // Compute residual R(ci, cj) = A(ci, cj) - a(ci) - b(cj)
            let mut r_values = Vec::with_capacity(cell_count);
            let mut r_variance = 0.0_f64;
            for ci in 0..ni {
                let a_ci = *a.get(ci).ok_or_else(|| PbError::Internal {
                    what: "a index".into(),
                })?;
                for cj in 0..nj {
                    let b_cj = *b.get(cj).ok_or_else(|| PbError::Internal {
                        what: "b index".into(),
                    })?;
                    let orig_a = table
                        .values
                        .at(&[ci, cj])
                        .ok_or_else(|| PbError::Internal {
                            what: "A coord".into(),
                        })?;
                    let r_val = orig_a - c_shift - a_ci - b_cj;
                    r_values.push(r_val);
                    let s_val = get_s(&s, nj, ci, cj)?;
                    r_variance += s_val * r_val * r_val;
                }
            }

            (ni, nj, a, b, r_values, r_variance, c_shift)
        };
        f0 += c_shift;

        // Update pair table: new values, se_band = None, new variance
        let pair_table = tables.get_mut(pair_idx).ok_or_else(|| PbError::Internal {
            what: "pair table index".into(),
        })?;
        pair_table.values = Tensor::from_vec(vec![n_i, n_j], r_values)?;
        pair_table.se_band = None;
        pair_table.variance = r_variance;

        // Add a into main table i
        let main_table_i = tables
            .get_mut(main_idx_i)
            .ok_or_else(|| PbError::Internal {
                what: "main table i".into(),
            })?;
        for (ci, &delta) in a.iter().enumerate() {
            main_table_i.values.add(&[ci], delta)?;
        }

        // Add b into main table j
        let main_table_j = tables
            .get_mut(main_idx_j)
            .ok_or_else(|| PbError::Internal {
                what: "main table j".into(),
            })?;
        for (cj, &delta) in b.iter().enumerate() {
            main_table_j.values.add(&[cj], delta)?;
        }
    }

    // Step 3: After all pairs: for every main table i,
    // μ = Σ_c m_i(c) f_i(c); subtract μ from every cell and add μ to f0.
    // Recompute variance = Σ_c m_i(c) f_i(c)², set se_band = None.
    for &(main_idx, ref m_i) in main_effects.values() {
        let main_table = tables.get_mut(main_idx).ok_or_else(|| PbError::Internal {
            what: "main table".into(),
        })?;
        let mut mu = 0.0_f64;
        for (c, &mi) in m_i.iter().enumerate() {
            let f_val = main_table
                .values
                .at(&[c])
                .ok_or_else(|| PbError::Internal {
                    what: "main table val".into(),
                })?;
            mu += mi * f_val;
        }
        f0 += mu;
        let mut new_var = 0.0_f64;
        for (c, &mi) in m_i.iter().enumerate() {
            let f_val = main_table
                .values
                .at(&[c])
                .ok_or_else(|| PbError::Internal {
                    what: "main table val".into(),
                })?;
            let centered = f_val - mu;
            main_table.values.set(&[c], centered)?;
            new_var += mi * centered * centered;
        }
        main_table.variance = new_var;
        main_table.se_band = None;
    }

    // Steps 4, 5, 6: Keep order >= 3 and factored unchanged; stamp w = RefMeasure::Joint.
    Ok(TableBank {
        f0,
        tables,
        merged_grids: bank.merged_grids.clone(),
        w: RefMeasure::Joint,
        factored: bank.factored.clone(),
        joint_variance: None,
    })
}

/// Conditional-purity check for a joint bank: every order-2 table averages to zero along
/// each axis under the JOINT support conditional on the other axis's cell, and every
/// order-1 table averages to zero under its marginal mass. `tol` is absolute.
///
/// # Errors
/// [`PbError::InvalidInput`] if `tol` is not finite or negative;
/// [`PbError::InvariantViolated`] ([`Invariant::Decomposability`]) if purity is violated;
/// [`PbError::Internal`] if table tensor dimensions escape or are malformed.
pub fn check_conditional_purity(bank: &TableBank, tol: f64) -> Result<(), PbError> {
    if !tol.is_finite() || tol < 0.0 {
        return Err(PbError::InvalidInput {
            what: "tolerance must be finite and non-negative".into(),
        });
    }

    for table in &bank.tables {
        if table.u.order() == 1 {
            let shape = table.values.shape();
            let n_cells = *shape.first().ok_or_else(|| PbError::Internal {
                what: "order-1 table has empty shape".into(),
            })?;
            let mut total_mass = 0.0_f64;
            for c in 0..n_cells {
                let mass = table.support.at(&[c]).ok_or_else(|| PbError::Internal {
                    what: "order-1 table support coord out of range".into(),
                })?;
                total_mass += mass;
            }
            let mut mean = 0.0_f64;
            if total_mass == 0.0 || !total_mass.is_finite() {
                for c in 0..n_cells {
                    let v = table.values.at(&[c]).ok_or_else(|| PbError::Internal {
                        what: "order-1 table values coord out of range".into(),
                    })?;
                    mean += v;
                }
                mean /= n_cells as f64;
            } else {
                for c in 0..n_cells {
                    let mass = table.support.at(&[c]).ok_or_else(|| PbError::Internal {
                        what: "order-1 table support coord out of range".into(),
                    })?;
                    let v = table.values.at(&[c]).ok_or_else(|| PbError::Internal {
                        what: "order-1 table values coord out of range".into(),
                    })?;
                    mean += (mass / total_mass) * v;
                }
            }
            if mean.abs() > tol {
                return Err(PbError::invariant(Invariant::Decomposability));
            }
        } else if table.u.order() == 2 {
            let shape = table.values.shape();
            let n_i = *shape.first().ok_or_else(|| PbError::Internal {
                what: "order-2 table axis 0 extent".into(),
            })?;
            let n_j = *shape.get(1).ok_or_else(|| PbError::Internal {
                what: "order-2 table axis 1 extent".into(),
            })?;

            // Along axis 0: conditional on cell c_j of axis 1
            for cj in 0..n_j {
                let mut slice_mass = 0.0_f64;
                for ci in 0..n_i {
                    let s_val = table
                        .support
                        .at(&[ci, cj])
                        .ok_or_else(|| PbError::Internal {
                            what: "order-2 support coord".into(),
                        })?;
                    slice_mass += s_val;
                }
                if slice_mass > 0.0 && slice_mass.is_finite() {
                    let mut cond_mean = 0.0_f64;
                    for ci in 0..n_i {
                        let s_val =
                            table
                                .support
                                .at(&[ci, cj])
                                .ok_or_else(|| PbError::Internal {
                                    what: "order-2 support coord".into(),
                                })?;
                        let v_val =
                            table
                                .values
                                .at(&[ci, cj])
                                .ok_or_else(|| PbError::Internal {
                                    what: "order-2 values coord".into(),
                                })?;
                        cond_mean += (s_val / slice_mass) * v_val;
                    }
                    if cond_mean.abs() > tol {
                        return Err(PbError::invariant(Invariant::Decomposability));
                    }
                }
            }

            // Along axis 1: conditional on cell c_i of axis 0
            for ci in 0..n_i {
                let mut slice_mass = 0.0_f64;
                for cj in 0..n_j {
                    let s_val = table
                        .support
                        .at(&[ci, cj])
                        .ok_or_else(|| PbError::Internal {
                            what: "order-2 support coord".into(),
                        })?;
                    slice_mass += s_val;
                }
                if slice_mass > 0.0 && slice_mass.is_finite() {
                    let mut cond_mean = 0.0_f64;
                    for cj in 0..n_j {
                        let s_val =
                            table
                                .support
                                .at(&[ci, cj])
                                .ok_or_else(|| PbError::Internal {
                                    what: "order-2 support coord".into(),
                                })?;
                        let v_val =
                            table
                                .values
                                .at(&[ci, cj])
                                .ok_or_else(|| PbError::Internal {
                                    what: "order-2 values coord".into(),
                                })?;
                        cond_mean += (s_val / slice_mass) * v_val;
                    }
                    if cond_mean.abs() > tol {
                        return Err(PbError::invariant(Invariant::Decomposability));
                    }
                }
            }
        }
    }
    Ok(())
}

fn get_s(s: &[f64], n_j: usize, ci: usize, cj: usize) -> Result<f64, PbError> {
    let idx = ci
        .checked_mul(n_j)
        .and_then(|v| v.checked_add(cj))
        .ok_or_else(|| PbError::Internal {
            what: "cell coordinate index overflow".into(),
        })?;
    s.get(idx).copied().ok_or_else(|| PbError::Internal {
        what: "cell coordinate index out of bounds".into(),
    })
}

fn solve_gaussian_elimination(
    m: &mut [Vec<f64>],
    rhs: &mut [f64],
    u: &crate::explain::FeatureSet,
) -> Result<Vec<f64>, PbError> {
    let n = m.len();
    if rhs.len() != n {
        return Err(PbError::ShapeMismatch {
            what: "matrix and rhs dimension mismatch".into(),
        });
    }

    // Maximum absolute entry of the initial matrix
    let mut max_abs = 0.0_f64;
    for row in m.iter() {
        for &val in row {
            let a = val.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
    }
    let threshold = 1e-14 * max_abs;

    // Elimination
    for k in 0..n {
        let mut pivot_row = k;
        let mut max_pivot = m
            .get(k)
            .and_then(|r| r.get(k))
            .copied()
            .ok_or_else(|| PbError::Internal {
                what: "matrix access".into(),
            })?
            .abs();

        // Deterministic partial pivoting: strictly greater ensures lowest index on ties
        for i in (k + 1)..n {
            let candidate = m
                .get(i)
                .and_then(|r| r.get(k))
                .copied()
                .ok_or_else(|| PbError::Internal {
                    what: "matrix access".into(),
                })?
                .abs();
            if candidate > max_pivot {
                max_pivot = candidate;
                pivot_row = i;
            }
        }

        if max_abs == 0.0 || max_pivot < threshold || !max_pivot.is_finite() {
            return Err(PbError::Internal {
                what: format!("joint projection singular for table {u:?}"),
            });
        }

        if pivot_row != k {
            m.swap(k, pivot_row);
            rhs.swap(k, pivot_row);
        }

        let pivot_val = *m
            .get(k)
            .and_then(|r| r.get(k))
            .ok_or_else(|| PbError::Internal {
                what: "pivot access".into(),
            })?;
        let rhs_k = *rhs.get(k).ok_or_else(|| PbError::Internal {
            what: "rhs access".into(),
        })?;
        let row_k = m.get(k).cloned().ok_or_else(|| PbError::Internal {
            what: "row k access".into(),
        })?;

        for (i, row_i) in m.iter_mut().enumerate().skip(k + 1) {
            let entry_ik = *row_i.get(k).ok_or_else(|| PbError::Internal {
                what: "entry ik".into(),
            })?;
            let factor = entry_ik / pivot_val;
            if let Some(slot) = row_i.get_mut(k) {
                *slot = 0.0;
            }
            for (j, slot) in row_i.iter_mut().enumerate().skip(k + 1) {
                let val_kj = *row_k.get(j).ok_or_else(|| PbError::Internal {
                    what: "val kj".into(),
                })?;
                *slot -= factor * val_kj;
            }
            let rhs_i = rhs.get_mut(i).ok_or_else(|| PbError::Internal {
                what: "rhs i".into(),
            })?;
            *rhs_i -= factor * rhs_k;
        }
    }

    // Back-substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let row_i = m.get(i).ok_or_else(|| PbError::Internal {
            what: "row i".into(),
        })?;
        let mut sum = *rhs.get(i).ok_or_else(|| PbError::Internal {
            what: "rhs i".into(),
        })?;
        for (j, &slot_xj) in x.iter().enumerate().skip(i + 1) {
            let a_ij = *row_i.get(j).ok_or_else(|| PbError::Internal {
                what: "row i col j".into(),
            })?;
            sum -= a_ij * slot_xj;
        }
        let pivot = *row_i.get(i).ok_or_else(|| PbError::Internal {
            what: "pivot i".into(),
        })?;
        let xi = sum / pivot;
        if !xi.is_finite() {
            return Err(PbError::Internal {
                what: format!("joint projection singular for table {u:?}"),
            });
        }
        let slot_xi = x
            .get_mut(i)
            .ok_or_else(|| PbError::Internal { what: "x i".into() })?;
        *slot_xi = xi;
    }

    Ok(x)
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
    use crate::explain::{fixture_model, fixture_serve, Tensor};

    /// Every joint grid cell of the fixture's two features, as `x_cells`.
    fn all_cells(bank: &TableBank) -> Vec<Vec<u32>> {
        let n0 = u32::from(bank.merged_grids[0].n_bins);
        let n1 = u32::from(bank.merged_grids[1].n_bins);
        let mut out = Vec::new();
        for a in 0..n0 {
            for b in 0..n1 {
                out.push(vec![a, b]);
            }
        }
        out
    }

    fn product_bank() -> TableBank {
        let model = fixture_model();
        let x = fixture_serve();
        let n = x.0.n_rows as usize;
        // A per-row mass that makes the joint support unbalanced, so the joint and product
        // ledgers actually differ.
        let mass: Vec<f32> = (0..n).map(|r| 1.0 + 3.0 * (r % 2) as f32).collect();
        model
            .explain_weighted(&x, RefMeasure::ExposureMarginals { floor: 1e-3 }, &mass)
            .unwrap()
    }

    #[test]
    fn rejoint_preserves_the_table_sum_on_every_cell() {
        let bank = product_bank();
        assert!(
            bank.tables.iter().any(|t| t.u.order() == 2),
            "fixture needs a pair"
        );
        let joint = rejoint(&bank, &JointOptions::default()).unwrap();
        assert_eq!(joint.w, RefMeasure::Joint);
        assert_eq!(joint.tables.len(), bank.tables.len());
        for cells in all_cells(&bank) {
            let a = bank.score(&cells).unwrap();
            let b = joint.score(&cells).unwrap();
            assert!((a - b).abs() < 1e-9, "{cells:?}: {a} vs {b}");
        }
    }

    #[test]
    fn rejoint_output_is_conditionally_pure() {
        let bank = product_bank();
        let joint = rejoint(
            &bank,
            &JointOptions {
                ridge: 0.0,
                floor: 1e-9,
            },
        )
        .unwrap();
        check_conditional_purity(&joint, 1e-7).unwrap();
        // With the ridge on, the pair residual is no longer exactly conditionally pure along
        // the ridge direction, but the mains still are under their marginal mass.
        let ridged = rejoint(&bank, &JointOptions::default()).unwrap();
        for t in ridged.tables.iter().filter(|t| t.u.order() == 1) {
            let n = t.values.shape()[0];
            let tot: f64 = (0..n).map(|c| t.support.at(&[c]).unwrap()).sum();
            let mean: f64 = (0..n)
                .map(|c| t.support.at(&[c]).unwrap() / tot * t.values.at(&[c]).unwrap())
                .sum();
            assert!(mean.abs() < 1e-9, "{mean}");
        }
    }

    /// Hand-built case. Two features with 3 cells each (cell 0 = missing). Support lives
    /// ONLY on the diagonal of the finite cells: (1,1) and (2,2). The pair table holds
    /// `A(ci,cj) = g(ci)·h(cj)` with `g = h = (0, +1, −1)`, which is pure under a uniform
    /// product measure (every row and column of the finite block sums to zero), yet on the
    /// diagonal support `h(cj) = g(ci)` so it is really the main effect `g(ci)² = +1` on
    /// both supported cells. The joint projection must find the additive fit
    /// `a(ci) + b(cj)` that reproduces `+1` on the two supported cells and is centred under
    /// each marginal (mass 1/2 on cells 1 and 2): `a = b = (·, +1/2, +1/2)` shifted to mean
    /// zero gives `a = b = (·, 0, 0)` and the constant `+1` ends up in `f0`. So the pair
    /// residual vanishes on the supported cells and the intercept rises by exactly one.
    #[test]
    fn joint_measure_moves_a_correlated_main_effect_out_of_the_pair() {
        let mut bank = product_bank();
        let pair = bank.tables.iter().position(|t| t.u.order() == 2).unwrap();
        let n0 = bank.tables[pair].values.shape()[0];
        let n1 = bank.tables[pair].values.shape()[1];
        assert_eq!(
            (n0, n1),
            (3, 3),
            "fixture pair grid is 3x3 (missing + 2 finite)"
        );
        let g = [0.0_f64, 1.0, -1.0];
        let mut a = Vec::new();
        let mut s = Vec::new();
        for ci in 0..3 {
            for cj in 0..3 {
                a.push(g[ci] * g[cj]);
                s.push(if ci == cj && ci > 0 { 10.0 } else { 0.0 });
            }
        }
        bank.tables[pair].values = Tensor::from_vec(vec![3, 3], a).unwrap();
        bank.tables[pair].support = Tensor::from_vec(vec![3, 3], s).unwrap();
        for t in bank.tables.iter_mut().filter(|t| t.u.order() == 1) {
            t.values = Tensor::from_vec(vec![3], vec![0.0; 3]).unwrap();
            t.support = Tensor::from_vec(vec![3], vec![0.0, 10.0, 10.0]).unwrap();
        }
        let f0_before = bank.f0;
        let joint = rejoint(
            &bank,
            &JointOptions {
                ridge: 0.0,
                floor: 1e-9,
            },
        )
        .unwrap();
        let r = &joint.tables[pair].values;
        assert!(r.at(&[1, 1]).unwrap().abs() < 1e-6, "{:?}", r.values());
        assert!(r.at(&[2, 2]).unwrap().abs() < 1e-6, "{:?}", r.values());
        assert!(
            (joint.f0 - f0_before - 1.0).abs() < 1e-6,
            "f0 {} vs {}",
            joint.f0,
            f0_before
        );
        for cells in all_cells(&bank) {
            let p = bank.score(&cells).unwrap();
            let q = joint.score(&cells).unwrap();
            assert!((p - q).abs() < 1e-9, "{cells:?}: {p} vs {q}");
        }
    }

    #[test]
    fn singular_without_ridge_reports_internal_error_not_panic() {
        let mut bank = product_bank();
        let pair = bank.tables.iter().position(|t| t.u.order() == 2).unwrap();
        // Diagonal-only support with no floor and no ridge: the bipartite support graph is
        // disconnected, so `(a, b) = (t, -t, -t, t)` is a null direction of the projection.
        let mut sup = vec![0.0_f64; 9];
        sup[4] = 10.0;
        sup[8] = 10.0;
        bank.tables[pair].support = Tensor::from_vec(vec![3, 3], sup).unwrap();
        for t in bank.tables.iter_mut().filter(|t| t.u.order() == 1) {
            t.support = Tensor::from_vec(vec![3], vec![0.0, 10.0, 10.0]).unwrap();
        }
        let err = rejoint(
            &bank,
            &JointOptions {
                ridge: 0.0,
                floor: 0.0,
            },
        )
        .unwrap_err();
        assert!(matches!(err, PbError::Internal { .. }), "{err:?}");
    }

    #[test]
    fn rejoint_is_deterministic() {
        let bank = product_bank();
        let a = rejoint(&bank, &JointOptions::default()).unwrap();
        let b = rejoint(&bank, &JointOptions::default()).unwrap();
        for (ta, tb) in a.tables.iter().zip(&b.tables) {
            let va = ta.values.values();
            let vb = tb.values.values();
            assert!(va
                .iter()
                .zip(vb.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()));
        }
        assert_eq!(a.f0.to_bits(), b.f0.to_bits());
    }
}
