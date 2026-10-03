//! Reference-preserving numeric neighbour diffusion on factored interactions.
use super::*;

/// Outcome for one factored interaction considered for optional graduation.
#[derive(Debug, Clone, Serialize)]
pub struct HighOrderSmoothingReport {
    /// Sorted raw feature identifiers.
    pub features: Vec<u32>,
    /// Index in the source bank's factored effects.
    pub factored_idx: u64,
    /// Requested diffusion strength.
    pub alpha: f64,
    /// Number of boxes before smoothing.
    pub boxes_before: u64,
    /// Number of boxes after smoothing (unchanged on a skip).
    pub boxes_after: u64,
    /// Whether this effect was modified; holdout adoption is a separate decision.
    pub applied: bool,
    /// Reason for leaving this effect unchanged.
    pub skipped: Option<String>,
}

impl TableBank {
    /// Smooth numeric directions of order-3..8 factored effects without a joint tensor.
    ///
    /// For adjacent finite cells i,j use c=min(w_i,w_j)/2 and the reversible
    /// diffusion L: (Lf)_i=c/w_i*(f_i-f_j). Apply I-alpha*mean_a(L_a), with
    /// alpha in [0,1]. Each axis operator preserves constants and its reference
    /// mean, so a pure interaction stays pure without moving mass to lower orders.
    /// Missing cell 0 has no edges; categorical axes have no operator. Distances
    /// are in cell order, not physical feature units. This is a first-difference
    /// smoother, not the dense tables' second-difference WH/GCV smoother.
    ///
    /// Corrections are pairs of centred singleton-mask boxes. Work and storage
    /// depend on mask boundaries, not the product of axis extents. Whole effects
    /// exceeding any expansion budget are skipped atomically in stored order.
    /// The result is a different predictor; validation must decide adoption.
    ///
    /// # Errors
    /// Returns an error for invalid strength, malformed boxes or nonpositive weights.
    pub fn smooth_high_order(
        &self,
        numeric_raws: &[u32],
        alpha: f64,
        max_boxes_per_effect: usize,
        max_added_boxes: usize,
        max_added_mask_cells: usize,
    ) -> Result<(Self, Vec<HighOrderSmoothingReport>), PbError> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(PbError::InvalidInput {
                what: "higher-order smoothing alpha must be finite and in [0,1]".into(),
            });
        }
        let mut out = self.clone();
        let mut report = Vec::new();
        if alpha == 0.0 {
            return Ok((out, report));
        }
        if matches!(self.w, RefMeasure::Joint) {
            return Err(PbError::InvalidInput {
                what: "higher-order smoothing requires a product reference measure".into(),
            });
        }
        let mut boxes_left = max_added_boxes;
        let mut masks_left = max_added_mask_cells;
        for (index, effect) in self.factored.iter().enumerate() {
            if !(3..=8).contains(&effect.u.order()) {
                return Err(PbError::InvalidInput {
                    what: "higher-order smoothing expects factored order 3 through 8".into(),
                });
            }
            effect.validate_shape()?;
            let mut entry = HighOrderSmoothingReport {
                features: effect.u.0.iter().map(|r| r.0).collect(),
                factored_idx: index as u64,
                alpha,
                boxes_before: effect.boxes.len() as u64,
                boxes_after: effect.boxes.len() as u64,
                applied: false,
                skipped: None,
            };
            let axes: Vec<usize> = effect
                .axes
                .iter()
                .enumerate()
                .filter(|(_, a)| numeric_raws.contains(&a.raw.0) && a.cells >= 3)
                .map(|(d, _)| d)
                .collect();
            if axes.is_empty() {
                entry.skipped = Some("no numeric axis with two finite cells".into());
                report.push(entry);
                continue;
            }
            for (axis, weights) in effect.axes.iter().zip(&effect.per_axis_w) {
                let mass: f64 = weights.iter().sum();
                if weights.len() != axis.cells as usize
                    || weights.iter().any(|w| !w.is_finite() || *w <= 0.0)
                    || (mass - 1.0).abs() > 1e-10
                {
                    return Err(PbError::InvalidInput {
                        what: "higher-order smoothing requires normalized positive product weights"
                            .into(),
                    });
                }
            }
            let mut needed = 0usize;
            let mut mask_cells = 0usize;
            for b in &effect.boxes {
                let size: usize = b.low.iter().map(Vec::len).sum();
                for &d in &axes {
                    for i in 1..effect.axes[d].cells as usize - 1 {
                        if b.low[d][i] != b.low[d][i + 1] {
                            needed = needed.saturating_add(2);
                            mask_cells = mask_cells.saturating_add(size.saturating_mul(2));
                        }
                    }
                }
            }
            if needed == 0 {
                entry.skipped = Some("no finite numeric mask boundaries".into());
            } else if needed.saturating_add(effect.boxes.len()) > max_boxes_per_effect
                || needed > boxes_left
                || mask_cells > masks_left
            {
                entry.skipped = Some("factored smoothing expansion budget".into());
            } else {
                let mut smoothed = effect.clone();
                let step = alpha / axes.len() as f64;
                for b in &effect.boxes {
                    for &d in &axes {
                        let bit = 1usize << d;
                        let weights = &effect.per_axis_w[d];
                        for i in 1..effect.axes[d].cells as usize - 1 {
                            let j = i + 1;
                            if b.low[d][i] == b.low[d][j] {
                                continue;
                            }
                            let wi = weights[i];
                            let wj = weights[j];
                            let conductance = wi.min(wj) * 0.5;
                            let mut left = FactoredBox {
                                p: vec![0.0; b.p.len()],
                                low: b.low.clone(),
                            };
                            let mut right = left.clone();
                            left.low[d].fill(false);
                            left.low[d][i] = true;
                            right.low[d].fill(false);
                            right.low[d][j] = true;
                            for corner in 0..b.p.len() {
                                if corner & bit != 0 {
                                    continue;
                                }
                                let at_i = corner | if b.low[d][i] { bit } else { 0 };
                                let at_j = corner | if b.low[d][j] { bit } else { 0 };
                                let difference = b.p[at_j] - b.p[at_i];
                                let mass = step * conductance * difference;
                                // The high sides cancel exactly outside the endpoints.
                                left.p[corner] = -mass;
                                left.p[corner | bit] =
                                    step * (conductance / wi) * difference - mass;
                                right.p[corner] = mass;
                                right.p[corner | bit] =
                                    -step * (conductance / wj) * difference + mass;
                            }
                            smoothed.boxes.push(left);
                            smoothed.boxes.push(right);
                        }
                    }
                }
                if smoothed
                    .boxes
                    .iter()
                    .any(|b| b.p.iter().any(|v| !v.is_finite()))
                {
                    return Err(PbError::InvalidInput {
                        what: "nonfinite higher-order smoothing coefficient".into(),
                    });
                }
                smoothed.variance = smoothed.compute_variance()?;
                if !smoothed.variance.is_finite() {
                    return Err(PbError::InvalidInput {
                        what: "nonfinite smoothed variance".into(),
                    });
                }
                entry.applied = true;
                entry.boxes_after = smoothed.boxes.len() as u64;
                out.factored[index] = smoothed;
                boxes_left -= needed;
                masks_left -= mask_cells;
            }
            report.push(entry);
        }
        Ok((out, report))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
    use super::*;

    fn fixture(order: usize) -> TableBank {
        let weights = vec![0.1, 0.15, 0.3, 0.45];
        let masks: Vec<Vec<bool>> = (0..order)
            .map(|d| {
                if d % 2 == 0 {
                    vec![true, false, true, false]
                } else {
                    vec![false, true, true, false]
                }
            })
            .collect();
        let p = (0..1usize << order)
            .map(|corner| {
                (0..order)
                    .map(|d| {
                        let mass: f64 = weights
                            .iter()
                            .zip(&masks[d])
                            .filter_map(|(w, low)| low.then_some(w))
                            .sum();
                        (if corner & (1 << d) != 0 { 1.0 } else { 0.0 }) - mass
                    })
                    .product::<f64>()
                    * 3.0
            })
            .collect();
        let mut effect = FactoredEffect {
            u: FeatureSet((0..order).map(|d| FeatureId(d as u32)).collect()),
            axes: (0..order)
                .map(|d| AxisId {
                    raw: FeatureId(d as u32),
                    borders: vec![0.0, 1.0],
                    cells: 4,
                    joint_channels: None,
                    band_of: None,
                })
                .collect(),
            per_axis_w: vec![weights; order],
            boxes: vec![FactoredBox { p, low: masks }],
            variance: 0.0,
        };
        effect.variance = effect.compute_variance().expect("variance");
        TableBank {
            f0: 2.0,
            tables: vec![],
            merged_grids: vec![],
            w: RefMeasure::Uniform,
            factored: vec![effect],
            joint_variance: None,
        }
    }

    fn coordinates(mut index: usize, order: usize) -> Vec<u32> {
        (0..order)
            .map(|_| {
                let c = (index % 4) as u32;
                index /= 4;
                c
            })
            .collect()
    }

    #[test]
    fn matches_dense_diffusion_preserves_purity_and_reduces_roughness() {
        for order in [3, 4, 8] {
            let bank = fixture(order);
            let original = serde_json::to_string(&bank).expect("serialize");
            // Last axis is categorical context: it must not receive a diffusion operator.
            let numeric: Vec<u32> = (0..order as u32 - 1).collect();
            let (out, report) = bank
                .smooth_high_order(&numeric, 0.7, 512, 4096, 2_000_000)
                .expect("smooth");
            assert!(report[0].applied);
            assert_eq!(bank.f0, out.f0);
            assert_eq!(original, serde_json::to_string(&bank).expect("serialize"));
            let f = &bank.factored[0];
            let g = &out.factored[0];
            let mut variance = 0.0;
            let mut rough_before = 0.0;
            let mut rough_after = 0.0;
            for index in 0..4usize.pow(order as u32) {
                let cells = coordinates(index, order);
                let before = f.eval(&cells).expect("eval");
                let after = g.eval(&cells).expect("eval");
                let mut expected = before;
                let mass: f64 = cells
                    .iter()
                    .enumerate()
                    .map(|(d, &c)| f.per_axis_w[d][c as usize])
                    .product();
                variance += mass * after * after;
                for &d in &numeric {
                    let d = d as usize;
                    let i = cells[d] as usize;
                    if i == 0 {
                        continue;
                    }
                    for j in [i.checked_sub(1), (i + 1 < 4).then_some(i + 1)]
                        .into_iter()
                        .flatten()
                    {
                        if j == 0 {
                            continue;
                        }
                        let mut neighbour = cells.clone();
                        neighbour[d] = j as u32;
                        let conductance = f.per_axis_w[d][i].min(f.per_axis_w[d][j]) * 0.5;
                        let difference = f.eval(&neighbour).expect("eval") - before;
                        expected += 0.7 / numeric.len() as f64 * conductance / f.per_axis_w[d][i]
                            * difference;
                        rough_before +=
                            mass * conductance / f.per_axis_w[d][i] * difference.powi(2);
                        rough_after += mass * conductance / f.per_axis_w[d][i]
                            * (g.eval(&neighbour).expect("eval") - after).powi(2);
                    }
                }
                assert!((after - expected).abs() < 1e-12, "order {order}, {cells:?}");
                // Every marginal slice must still have zero reference mean.
                for d in 0..order {
                    if cells[d] != 0 {
                        continue;
                    }
                    let mean: f64 = (0..4)
                        .map(|j| {
                            let mut point = cells.clone();
                            point[d] = j as u32;
                            f.per_axis_w[d][j] * g.eval(&point).expect("eval")
                        })
                        .sum();
                    assert!(mean.abs() < 1e-12);
                }
            }
            assert!((variance - g.variance).abs() < 1e-12);
            assert!(g.variance < f.variance);
            assert!(rough_after < rough_before);
            let roundtrip: TableBank =
                serde_json::from_str(&serde_json::to_string(&out).expect("serialize"))
                    .expect("deserialize");
            assert_eq!(out, roundtrip);
        }
    }

    #[test]
    fn noops_budgets_and_missing_cell_are_explicit() {
        let bank = fixture(3);
        let (zero, report) = bank
            .smooth_high_order(&[0, 1, 2], 0.0, 0, 0, 0)
            .expect("zero");
        assert_eq!(zero, bank);
        assert!(report.is_empty());
        for (numeric, boxes, added, masks) in [
            (vec![], 512, 4096, 2_000_000),
            (vec![0], 1, 4096, 2_000_000),
            (vec![0], 512, 0, 2_000_000),
            (vec![0], 512, 4096, 0),
        ] {
            let (out, report) = bank
                .smooth_high_order(&numeric, 0.5, boxes, added, masks)
                .expect("skip");
            assert_eq!(out, bank);
            assert!(!report[0].applied);
            assert!(report[0].skipped.is_some());
        }
        let (out, _) = bank
            .smooth_high_order(&[0], 1.0, 512, 4096, 2_000_000)
            .expect("smooth");
        for j in 0..4 {
            for k in 0..4 {
                let cells = [0, j, k];
                assert!(
                    (out.factored[0].eval(&cells).expect("eval")
                        - bank.factored[0].eval(&cells).expect("eval"))
                    .abs()
                        < 1e-14
                );
            }
        }
        for alpha in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
            assert!(bank
                .smooth_high_order(&[0], alpha, 512, 4096, 2_000_000)
                .is_err());
        }
    }
}
