//! Gate G-MDS: the in-fit rate-collapse gate on `max_delta_step` (spec §05.6 addendum, av35).
//!
//! The shipping promise has two halves, and both are asserted here:
//!
//!  1. **Silent ⇒ bit-identical.** A fit whose detector never trips must produce the same
//!     model, down to the serialized bytes, as a fit built with the gate switched off. This is
//!     what lets an objective-aware default change ride into a benchmark without invalidating
//!     a single cached cell that did not collapse.
//!  2. **Fired ⇒ exactly a mid-fit `max_delta_step` swap.** Nothing else about the fit changes:
//!     the trees grown BEFORE the engaging round are bit-identical to the ungated fit's, and
//!     the only difference from the engaging round on is the tighter leaf-step clamp.
//!
//! Plus the scope and determinism guards: gamma/poisson ship no gate, an explicit
//! `max_delta_step` outranks the gate, and the decision is thread-count invariant.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use t_boost_core::{
    bin_columns, encode_model, BinConfig, Booster, Config, CredibilityFloor, FitSpec, Gamma,
    GatedDeltaStep, GatedStepPolicy, InteractionPolicy, Loss, Model, MonotoneMap, Poisson, Tweedie,
    TWEEDIE_GATED_DELTA_STEP,
};

fn spec<'a>(loss: &'a dyn Loss, weight: Option<&'a [f32]>) -> FitSpec<'a> {
    FitSpec {
        loss,
        weight,
        exposure: None,
        monotone: MonotoneMap::new(),
        interaction: InteractionPolicy::default(),
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed: 7,
    }
}

fn cfg(policy: GatedStepPolicy, max_delta_step: Option<f32>, n_trees: u32) -> Config {
    Config {
        n_trees,
        learning_rate: 0.5,
        lambda: 0.1,
        max_delta_step,
        max_delta_step_gated: policy,
        // No ES carve: every arm must grow exactly `n_trees` trees so tree-prefix comparison is
        // meaningful (a truncating holdout would confound "the cap changed" with "ES stopped").
        validation_fraction: None,
        ..Config::default()
    }
}

/// A zero-inflated aggregated-exposure fixture that reproduces the production failure mode:
/// one small, cleanly-separable subgroup (`x0 == 0`) has NO claims at all, so boosting drives
/// its predicted rate toward zero without bound, while the rest of the portfolio carries a
/// heavy-tailed positive severity. `weight` is the exposure.
fn collapse_fixture(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut x0 = Vec::with_capacity(n);
    let mut x1 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let group = if i % 11 == 0 { 0.0 } else { (i % 7 + 1) as f32 };
        x0.push(group);
        x1.push((i % 5) as f32);
        w.push(1.0 + (i % 3) as f32);
        // The collapsing subgroup is exactly zero-claim; everyone else has a lumpy rate.
        y.push(if group == 0.0 {
            0.0
        } else if i % 4 == 0 {
            120.0 * group
        } else {
            0.0
        });
    }
    (x0, x1, y, w)
}

/// A benign fixture: every subgroup has claims, so no predicted rate ever runs away.
fn benign_fixture(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut x0 = Vec::with_capacity(n);
    let mut x1 = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let g = (i % 6 + 1) as f32;
        x0.push(g);
        x1.push((i % 4) as f32);
        w.push(1.0 + (i % 3) as f32);
        y.push(if i % 3 == 0 { 40.0 * g } else { 5.0 * g });
    }
    (x0, x1, y, w)
}

fn fit(
    fixture: &(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>),
    loss: &dyn Loss,
    config: Config,
) -> Model {
    let (x0, x1, y, w) = fixture;
    let refs: Vec<&[f32]> = vec![x0, x1];
    let x = bin_columns(&refs, None, &BinConfig::default(), 0).unwrap();
    Booster::with_config(config)
        .fit(&x, y, &spec(loss, Some(w)))
        .unwrap()
}

fn tweedie() -> Tweedie {
    Tweedie::new(1.7).unwrap()
}

// ---------------------------------------------------------------------------------------
// 1. The gate fires on a real collapse, and firing is EXACTLY a mid-fit max_delta_step swap.
// ---------------------------------------------------------------------------------------

#[test]
fn gate_fires_on_collapse_and_only_changes_rounds_from_the_engaging_one_on() {
    let fx = collapse_fixture(600);
    let gated = fit(&fx, &tweedie(), cfg(GatedStepPolicy::Objective, None, 40));
    let ungated = fit(&fx, &tweedie(), cfg(GatedStepPolicy::Off, None, 40));

    let report = gated
        .delta_step_gate
        .expect("tweedie must arm the gate under the Objective policy");
    assert!(
        report.engaged,
        "the zero-claim-subgroup fixture must collapse: min ln(mu/mean) reached {}, threshold {}",
        report.min_log_rate_ratio, report.log_threshold
    );
    let r = report.engaged_round.expect("engaged ⇒ a round is recorded") as usize;
    // Round 0 can never fire: `raw` is seeded to f0 (+offset), so every s_i is exactly 0 and the
    // threshold is strictly negative. A collapse takes at least one tree to build.
    assert!(r >= 1, "the gate cannot fire at round 0, got {r}");
    assert!(
        r < 40,
        "the fixture must collapse inside the round budget, got {r}"
    );
    assert_eq!(report.bags_total, 1);
    assert_eq!(report.bags_engaged, 1);
    assert_eq!(
        report.capped_step,
        f64::from(TWEEDIE_GATED_DELTA_STEP.capped_step)
    );
    // The detector stops scanning the moment it fires — it is monotone, so a later observation
    // could not change the outcome.
    assert_eq!(report.rounds_checked, r as u64 + 1);

    // THE SEMANTICS CLAIM: trees grown before the engaging round are untouched...
    assert_eq!(
        gated.trees[..r],
        ungated.trees[..r],
        "the cap must apply from the engaging round ON — earlier trees must be bit-identical"
    );
    // ...and the fit really does diverge from there (otherwise the test would pass vacuously).
    assert_ne!(
        gated.trees[r..],
        ungated.trees[r..],
        "engaging a 0.3 clamp where 0.7 was in force must change the remaining trees"
    );
    // Every leaf grown from the engaging round on respects the tighter clamp.
    let bound = 0.5_f32 * TWEEDIE_GATED_DELTA_STEP.capped_step + 1e-5;
    for (_, tree) in &gated.trees[r..] {
        for &v in &tree.leaves {
            assert!(
                v.abs() <= bound,
                "post-engage leaf {v} exceeds lr·capped_step = {bound}"
            );
        }
    }
}

/// The `Objective` policy is nothing but a lookup of `Loss::gated_max_delta_step()` — spelling
/// the same numbers out by hand must produce the identical fit. This is what lets the arena
/// battery measure the shipped default through an explicit `On(..)` arm.
#[test]
fn the_objective_default_resolves_to_the_same_fit_as_the_explicit_policy() {
    let fx = collapse_fixture(600);
    let auto = fit(&fx, &tweedie(), cfg(GatedStepPolicy::Objective, None, 40));
    let explicit = fit(
        &fx,
        &tweedie(),
        cfg(GatedStepPolicy::On(TWEEDIE_GATED_DELTA_STEP), None, 40),
    );
    assert!(auto.delta_step_gate.expect("armed").engaged);
    assert_eq!(
        auto.delta_step_gate.map(|r| r.engaged_round),
        explicit.delta_step_gate.map(|r| r.engaged_round)
    );
    assert_eq!(
        encode_model(&auto).unwrap(),
        encode_model(&explicit).unwrap()
    );
}

/// The engage is a `min` against the resolved clamp, so a gate whose `capped_step` equals the
/// objective's standing `0.7` is arithmetically inert even though the detector fires. This
/// isolates "the gate swapped max_delta_step" from "the gate did something else".
#[test]
fn engaging_a_non_tightening_cap_is_bit_identical() {
    let fx = collapse_fixture(600);
    let inert = fit(
        &fx,
        &tweedie(),
        cfg(
            GatedStepPolicy::On(GatedDeltaStep {
                collapse_threshold: 0.01,
                capped_step: 0.7, // == Tweedie's standing max_delta_step
            }),
            None,
            40,
        ),
    );
    let ungated = fit(&fx, &tweedie(), cfg(GatedStepPolicy::Off, None, 40));
    assert!(
        inert.delta_step_gate.expect("armed").engaged,
        "detector must still fire"
    );
    assert_eq!(
        inert, ungated,
        "a non-tightening engage must not move the fit"
    );
    assert_eq!(
        encode_model(&inert).unwrap(),
        encode_model(&ungated).unwrap()
    );
}

// ---------------------------------------------------------------------------------------
// 2. Silent gate ⇒ bit-identical to the ungated engine.
// ---------------------------------------------------------------------------------------

#[test]
fn silent_gate_is_bit_identical_to_gate_off() {
    let fx = benign_fixture(600);
    let gated = fit(&fx, &tweedie(), cfg(GatedStepPolicy::Objective, None, 40));
    let ungated = fit(&fx, &tweedie(), cfg(GatedStepPolicy::Off, None, 40));

    let report = gated.delta_step_gate.expect("tweedie arms the gate");
    assert!(
        !report.engaged,
        "benign fixture must not collapse (min ln ratio {} vs threshold {})",
        report.min_log_rate_ratio, report.log_threshold
    );
    // A silent fit pays the scan every round — that IS the overhead being measured.
    assert_eq!(report.rounds_checked, 40);
    assert_eq!(gated, ungated, "silent gate must leave the model untouched");
    assert_eq!(
        encode_model(&gated).unwrap(),
        encode_model(&ungated).unwrap(),
        "silent gate must be identical down to the serialized bytes"
    );
}

/// The report is a runtime-only introspection aid: it must not leak into model identity, or
/// the bit-identity check above would be comparing the gate against itself.
#[test]
fn report_is_excluded_from_identity_and_from_the_wire() {
    let fx = collapse_fixture(600);
    let gated = fit(&fx, &tweedie(), cfg(GatedStepPolicy::Objective, None, 40));
    assert!(gated.delta_step_gate.is_some());
    let round_tripped = t_boost_core::decode_model(&encode_model(&gated).unwrap()).unwrap();
    assert!(
        round_tripped.delta_step_gate.is_none(),
        "the report must not survive serialization"
    );
    assert_eq!(
        round_tripped, gated,
        "§10.7 decode(encode(m)) == m must still hold with the report dropped"
    );
}

// ---------------------------------------------------------------------------------------
// 3. Precedence: an explicit max_delta_step outranks the gate.
// ---------------------------------------------------------------------------------------

#[test]
fn explicit_max_delta_step_disables_the_gate() {
    let fx = collapse_fixture(600);
    let explicit = fit(
        &fx,
        &tweedie(),
        cfg(GatedStepPolicy::Objective, Some(0.7), 40),
    );
    assert!(
        explicit.delta_step_gate.is_none(),
        "a caller-named cap must not be gated"
    );
    // ...and it really is the plain 0.7 fit, not a gated one that happened to stay silent.
    let plain = fit(&fx, &tweedie(), cfg(GatedStepPolicy::Off, Some(0.7), 40));
    assert_eq!(explicit, plain);

    // Same for an explicit cap that is TIGHTER than the gate's: still ungated, still literal.
    let tight = fit(
        &fx,
        &tweedie(),
        cfg(GatedStepPolicy::Objective, Some(0.2), 40),
    );
    let tight_off = fit(&fx, &tweedie(), cfg(GatedStepPolicy::Off, Some(0.2), 40));
    assert!(tight.delta_step_gate.is_none());
    assert_eq!(tight, tight_off);
    assert_ne!(tight, explicit);
}

#[test]
fn policy_off_disables_the_gate_on_a_collapsing_fit() {
    let fx = collapse_fixture(600);
    let off = fit(&fx, &tweedie(), cfg(GatedStepPolicy::Off, None, 40));
    assert!(
        off.delta_step_gate.is_none(),
        "Off ⇒ no detector, no report"
    );
}

// ---------------------------------------------------------------------------------------
// 4. Scope: tweedie ONLY.
// ---------------------------------------------------------------------------------------

#[test]
fn only_tweedie_ships_a_gated_default() {
    // Gamma was MEASURED HARMFUL under a blanket 0.3 (freclaimdam −0.0105 skill, t = −2.97) and
    // Poisson is untested, so both keep the plain 0.7 for the whole fit.
    assert!(Poisson.gated_max_delta_step().is_none());
    assert!(Gamma.gated_max_delta_step().is_none());
    assert!(t_boost_core::SquaredError.gated_max_delta_step().is_none());
    assert!(t_boost_core::Logistic.gated_max_delta_step().is_none());
    let tw = tweedie()
        .gated_max_delta_step()
        .expect("tweedie ships a gate");
    assert_eq!(tw, TWEEDIE_GATED_DELTA_STEP);
    assert_eq!(tw.collapse_threshold, 1e-3);
    assert_eq!(tw.capped_step, 0.3);
    // The standing caps are untouched by all of this.
    assert_eq!(Poisson.max_delta_step(), Some(0.7));
    assert_eq!(Gamma.max_delta_step(), Some(0.7));
    assert_eq!(tweedie().max_delta_step(), Some(0.7));
}

#[test]
fn gamma_and_poisson_fits_are_untouched_by_the_default_policy() {
    let fx = collapse_fixture(600);
    // Gamma's strict `y > 0` domain: shift the fixture's zeros up, keeping the collapsing
    // subgroup's severity two orders of magnitude below everyone else's.
    let (x0, x1, y, w) = &fx;
    let gy: Vec<f32> = y
        .iter()
        .zip(x0)
        .map(|(&yi, &g)| if g == 0.0 { 0.01 } else { yi.max(1.0) })
        .collect();
    let gfx = (x0.clone(), x1.clone(), gy, w.clone());
    for (name, loss) in [
        ("gamma", &Gamma as &dyn Loss),
        ("poisson", &Poisson as &dyn Loss),
    ] {
        let auto = fit(&gfx, loss, cfg(GatedStepPolicy::Objective, None, 40));
        let off = fit(&gfx, loss, cfg(GatedStepPolicy::Off, None, 40));
        assert!(
            auto.delta_step_gate.is_none(),
            "{name} must not arm a gate under the default policy"
        );
        assert_eq!(
            auto, off,
            "{name} fit must be untouched by the gate default"
        );
        assert_eq!(
            encode_model(&auto).unwrap(),
            encode_model(&off).unwrap(),
            "{name} must be bit-identical"
        );
    }
}

/// The gate is opt-in-able on objectives that ship none — the knob is a policy, not a
/// tweedie-shaped special case wired into the round loop.
#[test]
fn an_explicit_policy_arms_the_gate_on_gamma() {
    let fx = collapse_fixture(600);
    let (x0, x1, y, w) = &fx;
    let gy: Vec<f32> = y
        .iter()
        .zip(x0)
        .map(|(&yi, &g)| if g == 0.0 { 0.01 } else { yi.max(1.0) })
        .collect();
    let gfx = (x0.clone(), x1.clone(), gy, w.clone());
    let armed = fit(
        &gfx,
        &Gamma,
        cfg(
            GatedStepPolicy::On(GatedDeltaStep {
                collapse_threshold: 0.5,
                capped_step: 0.3,
            }),
            None,
            40,
        ),
    );
    assert!(armed.delta_step_gate.expect("armed").engaged);
}

// ---------------------------------------------------------------------------------------
// 5. Determinism.
// ---------------------------------------------------------------------------------------

#[test]
fn gate_decision_is_thread_count_invariant() {
    let fx = collapse_fixture(2000);
    let run = |threads: usize| -> Model {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| fit(&fx, &tweedie(), cfg(GatedStepPolicy::Objective, None, 30)))
    };
    let a = run(1);
    let b = run(8);
    let ra = a.delta_step_gate.expect("armed");
    let rb = b.delta_step_gate.expect("armed");
    assert!(ra.engaged);
    assert_eq!(
        ra.engaged_round, rb.engaged_round,
        "engage round must not depend on threads"
    );
    assert_eq!(
        ra.min_log_rate_ratio, rb.min_log_rate_ratio,
        "the detector's f64 min is a sequential fold — bit-equal across thread counts"
    );
    assert_eq!(encode_model(&a).unwrap(), encode_model(&b).unwrap());
}

#[test]
fn gate_config_is_validated() {
    for bad in [
        GatedDeltaStep {
            collapse_threshold: 0.0,
            capped_step: 0.3,
        },
        GatedDeltaStep {
            collapse_threshold: 1.0,
            capped_step: 0.3,
        },
        GatedDeltaStep {
            collapse_threshold: f32::NAN,
            capped_step: 0.3,
        },
        GatedDeltaStep {
            collapse_threshold: 0.01,
            capped_step: 0.0,
        },
        GatedDeltaStep {
            collapse_threshold: 0.01,
            capped_step: f32::INFINITY,
        },
    ] {
        assert!(bad.validate().is_err(), "{bad:?} must be rejected");
        assert!(
            cfg(GatedStepPolicy::On(bad), None, 5).validate().is_err(),
            "Config::validate must reject {bad:?}"
        );
    }
    assert!(TWEEDIE_GATED_DELTA_STEP.validate().is_ok());
}
