// BUG-004 harness: own-target invariance of training encodings under nonzero smoothing.
use t_boost_core::cat::{
    fit_cat_encoder, CatFitSpec, LeakageScheme, Smooth, TsConfig, TsEncodingId,
};
use t_boost_core::data::FeatureId;

fn train0(levels: &[String], y: &[f32], cfg: &TsConfig) -> (f32, f32) {
    let spec = CatFitSpec {
        raw: FeatureId(0),
        id: TsEncodingId(0),
        weight: None,
        exposure: None,
        config: cfg,
        seed: 123,
    };
    let (enc, train) = fit_cat_encoder(levels, y, spec).unwrap();
    (train[0], enc.base)
}

#[test]
fn bug004_regression() {
    let levels = vec!["a".to_string(); 30];
    let y: Vec<f32> = (0..30).map(|i| i as f32).collect();
    let mut changed = y.clone();
    changed[0] += 10_000.0;
    for (name, leakage) in [
        ("KFold k=5", LeakageScheme::KFold { k: 5 }),
        ("LeaveOneOut", LeakageScheme::LeaveOneOut),
        ("Ordered n_perms=1", LeakageScheme::Ordered { n_perms: 1 }),
    ] {
        for smooth in [
            Smooth::Fixed { m: 10.0 },
            Smooth::Fixed { m: 0.0 },
            Smooth::Auto,
        ] {
            let cfg = TsConfig {
                leakage,
                smooth,
                direct_max_levels: 0,
                ..TsConfig::default()
            };
            let (a, base_a) = train0(&levels, &y, &cfg);
            let (b, base_b) = train0(&levels, &changed, &cfg);
            assert_eq!(
                a, b,
                "own target changed encoding under {name}, smoothing {smooth:?}"
            );
            println!(
                "{name:<18} {smooth:?}: train[0] original={a} (base {base_a})  after y[0]+=10000: {b} (base {base_b})  changed={}",
                a != b
            );
        }
    }
}
