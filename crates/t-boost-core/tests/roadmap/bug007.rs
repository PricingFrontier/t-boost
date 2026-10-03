use t_boost_core::engine::MultiClassModel;
use t_boost_core::explain::{fixture_model, fixture_serve, RefMeasure};
use t_boost_core::prune::{prune_multiclass_to_tables, PruneConfig};
use t_boost_core::serialize::SCHEMA_VERSION_UNLIFTED;

/// Direct weighted cross-entropy of the FULL model on each round-robin fold, computed from
/// the fixture tree by hand: class a logit = g(x) (6,2,2,0 for rows 0..4), classes b,c = 0.
fn direct_fold_losses(w: &[f32; 4]) -> Vec<f64> {
    let g = [6.0_f64, 2.0, 2.0, 0.0];
    let labels = [1usize, 0, 0, 0];
    let folds = [[0usize, 2], [1, 3]];
    folds
        .iter()
        .map(|rows| {
            let (mut acc, mut sw) = (0.0, 0.0);
            for &r in rows {
                let logits = [g[r], 0.0, 0.0];
                let denom: f64 = logits.iter().map(|v| v.exp()).sum();
                let p = logits[labels[r]].exp() / denom;
                acc += f64::from(w[r]) * (-p.ln());
                sw += f64::from(w[r]);
            }
            acc / sw
        })
        .collect()
}

#[test]
fn bug007_regression() {
    let base = fixture_model();
    let mut zero = base.clone();
    zero.trees.clear();
    let mc = MultiClassModel {
        classes: vec![base.clone(), zero.clone(), zero],
        class_labels: vec!["a".into(), "b".into(), "c".into()],
        schema_version: SCHEMA_VERSION_UNLIFTED,
        cell_refit: None,
    };
    let serve = fixture_serve();
    let labels = [1u32, 0, 0, 0];
    let cfg = PruneConfig {
        se_rule: 0.0,
        ..Default::default()
    };
    let mut reference: Option<(Vec<f64>, Vec<t_boost_core::explain::FeatureSet>)> = None;
    for w in [[1.0f32; 4], [100.0, 1.0, 100.0, 1.0]] {
        let (_tm, rep) = prune_multiclass_to_tables(
            &mc,
            &serve,
            &labels,
            &w,
            RefMeasure::Uniform,
            &[0, 1, 2, 3],
            2,
            &cfg,
        )
        .unwrap();
        let losses: Vec<f64> = rep.path.iter().map(|p| p.mean_deviance).collect();
        if let Some((expected, kept)) = &reference {
            assert_eq!(&rep.kept, kept, "fold-local scaling changed keep set");
            for (a, b) in losses.iter().zip(expected) {
                assert!((a - b).abs() < 1e-6);
            }
        } else {
            reference = Some((losses.clone(), rep.kept.clone()));
        }
        println!("w={w:?}");
        println!("  reported path losses: {losses:?}");
        println!(
            "  path n_tables: {:?}",
            rep.path.iter().map(|p| p.n_tables).collect::<Vec<_>>()
        );
        println!("  kept: {:?}", rep.kept);
        println!("  dropped: {:?}", rep.dropped);
        for ts in &rep.table_scores {
            println!(
                "  table {:?} mean_gain={:.7} se_gain={:.7} selected={}",
                ts.u, ts.mean_gain, ts.se_gain, ts.selected
            );
        }
        let direct = direct_fold_losses(&w);
        let direct_mean = direct.iter().sum::<f64>() / direct.len() as f64;
        assert!((losses[0] - direct_mean).abs() < 1e-6);
        let fold_sw: Vec<f64> = [[0usize, 2], [1, 3]]
            .iter()
            .map(|rows| rows.iter().map(|&r| f64::from(w[r])).sum::<f64>())
            .collect();
        let double: Vec<f64> = direct.iter().zip(&fold_sw).map(|(d, s)| d / s).collect();
        println!(
            "  direct per-fold weighted-mean CE (full model): {direct:?} -> mean {direct_mean:.7}"
        );
        println!(
            "  same divided again by fold weight sums {fold_sw:?}: {double:?} -> mean {:.7}",
            double.iter().sum::<f64>() / 2.0
        );
    }
}
