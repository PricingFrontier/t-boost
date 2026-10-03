//! Dynamic moments for honest ordered/LOO priors. A deterministic treap ordered by
//! category mean keeps the global moments current without rescanning every level.
//! Auto smoothing queries the credible tails, pruning subtrees whose mean/weight
//! bounds put all their levels on the same side of the credibility threshold.

use super::{
    finite_m, Smooth, TrainingMoment, CRED_NULL_KAPPA, CRED_Z2_THRESHOLD, MAX_AUTO_SMOOTH,
    MIN_AUTO_VARIANCE,
};
use crate::{pb_seed, PbError, Stage};

#[derive(Clone, Copy, Default)]
struct Summary {
    means: TrainingMoment,
    within: f64,
    count: usize,
    min_mean: f64,
    max_mean: f64,
    min_weight: f64,
    max_weight: f64,
}

impl Summary {
    fn leaf(moment: TrainingMoment) -> Self {
        if moment.weight <= 0.0 {
            return Self::default();
        }
        Self {
            means: TrainingMoment {
                within: 0.0,
                ..moment
            },
            within: moment.within,
            count: 1,
            min_mean: moment.mean,
            max_mean: moment.mean,
            min_weight: moment.weight,
            max_weight: moment.weight,
        }
    }

    fn merge(self, other: Self) -> Self {
        if self.count == 0 {
            return other;
        }
        if other.count == 0 {
            return self;
        }
        Self {
            means: self.means.merge(other.means),
            within: self.within + other.within,
            count: self.count + other.count,
            min_mean: self.min_mean.min(other.min_mean),
            max_mean: self.max_mean.max(other.max_mean),
            min_weight: self.min_weight.min(other.min_weight),
            max_weight: self.max_weight.max(other.max_weight),
        }
    }

    fn squared_distance(self, base: f64) -> f64 {
        self.means.within + self.means.weight * (self.means.mean - base).powi(2)
    }
}

struct Node {
    moment: TrainingMoment,
    summary: Summary,
    left: Option<usize>,
    right: Option<usize>,
    priority: u64,
}

pub(super) struct TrainingPriorIndex {
    nodes: Vec<Node>,
    root: Option<usize>,
}

impl TrainingPriorIndex {
    pub(super) fn new(count: usize) -> Result<Self, PbError> {
        let nodes = (0..count)
            .map(|id| {
                let id = u32::try_from(id).map_err(|_| PbError::InvalidInput {
                    what: "too many categorical levels".into(),
                })?;
                Ok(Node {
                    moment: TrainingMoment::default(),
                    summary: Summary::default(),
                    left: None,
                    right: None,
                    priority: pb_seed(0, id, Stage::Categorical as u32, 0),
                })
            })
            .collect::<Result<_, PbError>>()?;
        Ok(Self { nodes, root: None })
    }

    fn node(&self, id: usize) -> Result<&Node, PbError> {
        self.nodes.get(id).ok_or_else(|| PbError::Internal {
            what: "prior index escaped".into(),
        })
    }

    fn node_mut(&mut self, id: usize) -> Result<&mut Node, PbError> {
        self.nodes.get_mut(id).ok_or_else(|| PbError::Internal {
            what: "prior index escaped".into(),
        })
    }

    fn summary(&self, root: Option<usize>) -> Result<Summary, PbError> {
        root.map(|id| self.node(id).map(|node| node.summary))
            .transpose()
            .map(Option::unwrap_or_default)
    }

    fn refresh(&mut self, id: usize) -> Result<(), PbError> {
        let node = self.node(id)?;
        let summary = self
            .summary(node.left)?
            .merge(Summary::leaf(node.moment))
            .merge(self.summary(node.right)?);
        self.node_mut(id)?.summary = summary;
        Ok(())
    }

    fn less(&self, a: usize, b: usize) -> Result<bool, PbError> {
        Ok(self
            .node(a)?
            .moment
            .mean
            .total_cmp(&self.node(b)?.moment.mean)
            .then(a.cmp(&b))
            .is_lt())
    }

    fn merge(
        &mut self,
        left: Option<usize>,
        right: Option<usize>,
    ) -> Result<Option<usize>, PbError> {
        let (Some(a), Some(b)) = (left, right) else {
            return Ok(left.or(right));
        };
        if (self.node(a)?.priority, a) < (self.node(b)?.priority, b) {
            let child = self.merge(self.node(a)?.right, right)?;
            self.node_mut(a)?.right = child;
            self.refresh(a)?;
            Ok(Some(a))
        } else {
            let child = self.merge(left, self.node(b)?.left)?;
            self.node_mut(b)?.left = child;
            self.refresh(b)?;
            Ok(Some(b))
        }
    }

    fn split(
        &mut self,
        root: Option<usize>,
        key: usize,
    ) -> Result<(Option<usize>, Option<usize>), PbError> {
        let Some(id) = root else {
            return Ok((None, None));
        };
        if self.less(id, key)? {
            let (left, right) = self.split(self.node(id)?.right, key)?;
            self.node_mut(id)?.right = left;
            self.refresh(id)?;
            Ok((Some(id), right))
        } else {
            let (left, right) = self.split(self.node(id)?.left, key)?;
            self.node_mut(id)?.left = right;
            self.refresh(id)?;
            Ok((left, Some(id)))
        }
    }

    fn remove(&mut self, root: Option<usize>, key: usize) -> Result<Option<usize>, PbError> {
        let Some(id) = root else {
            return Ok(None);
        };
        if id == key {
            return self.merge(self.node(id)?.left, self.node(id)?.right);
        }
        if self.less(key, id)? {
            let child = self.remove(self.node(id)?.left, key)?;
            self.node_mut(id)?.left = child;
        } else {
            let child = self.remove(self.node(id)?.right, key)?;
            self.node_mut(id)?.right = child;
        }
        self.refresh(id)?;
        Ok(Some(id))
    }

    pub(super) fn set(&mut self, id: usize, moment: TrainingMoment) -> Result<(), PbError> {
        if self.node(id)?.moment.weight > 0.0 {
            self.root = self.remove(self.root, id)?;
        }
        let node = self.node_mut(id)?;
        node.moment = moment;
        node.left = None;
        node.right = None;
        node.summary = Summary::leaf(moment);
        if moment.weight > 0.0 {
            let (left, right) = self.split(self.root, id)?;
            let left = self.merge(left, Some(id))?;
            self.root = self.merge(left, right)?;
        }
        Ok(())
    }

    fn credible(
        &self,
        root: Option<usize>,
        base: f64,
        threshold: f64,
        visits: &mut usize,
    ) -> Result<Summary, PbError> {
        let Some(id) = root else {
            return Ok(Summary::default());
        };
        *visits += 1;
        let node = self.node(id)?;
        let s = node.summary;
        let nearest = (s.min_mean - base).max(base - s.max_mean).max(0.0);
        let furthest = (s.min_mean - base).abs().max((s.max_mean - base).abs());
        if s.max_weight * furthest.powi(2) <= threshold {
            return Ok(Summary::default());
        }
        if s.min_weight * nearest.powi(2) > threshold {
            return Ok(s);
        }
        let own = if node.moment.weight * (node.moment.mean - base).powi(2) > threshold {
            Summary::leaf(node.moment)
        } else {
            Summary::default()
        };
        Ok(self
            .credible(node.left, base, threshold, visits)?
            .merge(own)
            .merge(self.credible(node.right, base, threshold, visits)?))
    }

    pub(super) fn prior(&self, smooth: Smooth) -> Result<(f32, Smooth), PbError> {
        let s = self.summary(self.root)?;
        if s.count == 0 {
            return Ok((0.0, Smooth::Fixed { m: 0.0 }));
        }
        let base = s.means.mean as f32;
        if matches!(smooth, Smooth::Fixed { .. }) {
            return Ok((base, smooth));
        }
        let strength = if s.count <= 2 {
            let within = s.within / s.means.weight;
            let between = s.squared_distance(f64::from(base)) / s.means.weight;
            if between <= MIN_AUTO_VARIANCE {
                if within <= MIN_AUTO_VARIANCE {
                    0.0
                } else {
                    MAX_AUTO_SMOOTH
                }
            } else {
                (within / between).min(MAX_AUTO_SMOOTH)
            }
        } else {
            let within = s.within / (s.means.weight - s.count as f64).max(MIN_AUTO_VARIANCE);
            if within <= MIN_AUTO_VARIANCE {
                0.0
            } else {
                let credible = self.credible(
                    self.root,
                    f64::from(base),
                    CRED_Z2_THRESHOLD * within,
                    &mut 0,
                )?;
                let numerator = credible.squared_distance(f64::from(base))
                    - within * (credible.count as f64 + CRED_NULL_KAPPA * s.count as f64);
                if credible.means.weight <= 0.0
                    || numerator / credible.means.weight <= MIN_AUTO_VARIANCE
                {
                    MAX_AUTO_SMOOTH
                } else {
                    (within * credible.means.weight / numerator).min(MAX_AUTO_SMOOTH)
                }
            }
        };
        Ok((
            base,
            Smooth::Fixed {
                m: finite_m(strength)?,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unreachable
    )]
    use super::*;

    #[test]
    fn indexed_prior_matches_rescanning_through_updates_and_exclusions() {
        let mut moments = vec![TrainingMoment::default(); 256];
        let mut index = TrainingPriorIndex::new(moments.len()).unwrap();
        for step in 0..4096 {
            let id = (step * 37) % moments.len();
            moments[id] = if step % 7 == 0 {
                TrainingMoment::default()
            } else {
                TrainingMoment {
                    weight: (step % 23 + 1) as f64,
                    mean: (step % 97) as f64 * 0.7,
                    within: (step % 43) as f64,
                }
            };
            index.set(id, moments[id]).unwrap();
            for smooth in [Smooth::Fixed { m: 10.0 }, Smooth::Auto] {
                let (base, Smooth::Fixed { m }) = index.prior(smooth).unwrap() else {
                    unreachable!()
                };
                let (expected, Smooth::Fixed { m: expected_m }) =
                    super::super::training_prior(&moments, smooth).unwrap()
                else {
                    unreachable!()
                };
                assert!((base - expected).abs() < 1e-5);
                assert!((m - expected_m).abs() <= 1e-5 * expected_m.abs().max(1.0));
            }
        }
    }

    #[test]
    fn high_cardinality_credible_queries_prune_instead_of_scanning_levels() {
        let mut index = TrainingPriorIndex::new(32768).unwrap();
        for id in 0..32768 {
            index
                .set(
                    id,
                    TrainingMoment {
                        weight: 2.0,
                        mean: id as f64,
                        within: 1.0,
                    },
                )
                .unwrap();
        }
        let mut visits = 0;
        let result = index
            .credible(index.root, 16384.0, 18.0, &mut visits)
            .unwrap();
        assert_eq!(result.count, 32768 - 7);
        assert!(visits < 256, "visited {visits} nodes for a two-tail query");
    }
}
