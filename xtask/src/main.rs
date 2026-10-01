//! `xtask` — t-boost's dev-only task runner (plan F0/F4).
//!
//! This crate ships **no** library code (it is `publish = false` and is not a
//! dependency of `t-boost-core`/`t-boost-py`), so it is in the unwrap-allowed
//! set `{tests, benches, xtask}` and does NOT inherit the workspace `[lints]`
//! panic-gate.
//!
//! It hosts the source-scanning *grep-gates* that CI runs over the shipped crates
//! (`crates/*/src`, excluding `tests/`/`benches/` and this crate). Doing them here
//! — rather than as shell `grep` lines in `ci.yml` — keeps them block-aware,
//! cross-platform, and unit-testable:
//!
//! * `check-no-box-dyn` — no `Box<dyn Error>` in any shipped signature (§13.8 / §02.4).
//! * `check-justified` — every `unwrap`/`expect`/`panic`/`unreachable` and every form-(b)
//!   `#[allow(clippy::indexing_slicing|arithmetic_side_effects)]` in shipped, non-`#[cfg(test)]`
//!   code carries a `// JUSTIFIED:` proof (§13.8).
//! * `check-no-usize-serialized` — no `usize`/`isize` field on a serialized type (§13.4 wire-width).
//! * `check-no-hashmap-serialized` — no `HashMap`/`HashSet` field on a serialized type (order).
//! * `check-all` — run every gate; non-zero exit if any fails.
//! * `accuracy` — deterministic, exactness-gated smoke harness (§13.7).
//! * `release-preflight` — lever/exactness checkpoint run before a release.
//! * `bit-repro` — fit+encode a fixture model once; CI runs it twice as separate
//!   process invocations and diffs the bytes (the promised cross-run/two-process
//!   reproducibility gate, §13.4).
//!
//! Each gate prints `file:line` for every violation and returns a non-zero
//! `ExitCode`, so CI fails closed.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use serde_json::json;
use t_boost_core::{
    assert_exact_decomposition, bin, bin_columns, check_feature_budget, encode_model, BinConfig,
    BinnedMatrix, Booster, BoosterConfig, Config, CredibilityFloor, ExactnessMode, FitSpec,
    HistPrecision, InteractionPolicy, Loss, MonotoneMap, NesterovSpec, OverflowPolicy, PbError,
    RefMeasure, RefitSpec, Sampling, ServeBinnedMatrix, SquaredError, Stage, TableBudget,
};

/// A grep-gate: scans the shipped sources and returns any violations found.
type GateFn = fn(&[SourceFile]) -> Vec<Violation>;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("--help");
    match cmd {
        "--help" | "-h" | "help" => {
            print_usage();
            ExitCode::SUCCESS
        }
        "accuracy" => match run_accuracy_cli(&args[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("xtask accuracy: {err}");
                ExitCode::FAILURE
            }
        },
        "release-preflight" => match run_release_preflight_cli(&args[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("xtask release-preflight: {err}");
                ExitCode::FAILURE
            }
        },
        "bit-repro" => match run_bit_repro_cli(&args[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("xtask bit-repro: {err}");
                ExitCode::FAILURE
            }
        },
        "check-no-box-dyn" => run_gate(check_no_box_dyn),
        "check-justified" => run_gate(check_justified),
        "check-no-usize-serialized" => run_gate(check_no_usize_serialized),
        "check-no-hashmap-serialized" => run_gate(check_no_hashmap_serialized),
        "check-all" => {
            let gates: [(&str, GateFn); 4] = [
                ("check-no-box-dyn", check_no_box_dyn),
                ("check-justified", check_justified),
                ("check-no-usize-serialized", check_no_usize_serialized),
                ("check-no-hashmap-serialized", check_no_hashmap_serialized),
            ];
            let files = match load_shipped_sources() {
                Ok(files) => files,
                Err(err) => {
                    eprintln!("xtask: could not enumerate shipped sources: {err}");
                    return ExitCode::FAILURE;
                }
            };
            let mut failed = false;
            for (name, gate) in gates {
                let violations = gate(&files);
                if violations.is_empty() {
                    println!("[ok]   {name}");
                } else {
                    failed = true;
                    println!("[FAIL] {name}: {} violation(s)", violations.len());
                    for v in &violations {
                        println!("    {v}");
                    }
                }
            }
            if failed {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        other => {
            eprintln!("xtask: unknown command `{other}`\n");
            print_usage();
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    println!(
        "t-boost xtask — dev-only task runner (ships no library code)\n\n\
         USAGE: cargo run -p xtask -- <command>\n\n\
         COMMANDS:\n\
         \x20 check-all                     run every grep-gate (CI entrypoint)\n\
         \x20 check-no-box-dyn              forbid `Box<dyn Error>` in shipped signatures\n\
         \x20 check-justified              require `// JUSTIFIED:` on unwrap/expect/panic/allow\n\
         \x20 check-no-usize-serialized     forbid usize/isize on serialized types\n\
         \x20 check-no-hashmap-serialized   forbid HashMap/HashSet on serialized types\n\
         \x20 accuracy [--seed N] [--output PATH] [--adversarial]\n\
         \x20                               exactness-gated deterministic smoke\n\
         \x20 release-preflight [--seed N] [--output PATH]\n\
         \x20                               lever/exactness checkpoint\n\
         \x20 bit-repro --output PATH [--seed N]\n\
         \x20                               fit+encode the fixture once (§13.4 cross-run gate;\n\
         \x20                               CI runs this twice and diffs the two outputs)\n\
         \x20 --help                        show this message"
    );
}

// ---------------------------------------------------------------------------
// `xtask accuracy`: dev-only deterministic smoke harness.
// ---------------------------------------------------------------------------

type XtaskResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone, PartialEq)]
struct AccuracyOptions {
    seed: u64,
    output: Option<PathBuf>,
    adversarial: bool,
}

impl Default for AccuracyOptions {
    fn default() -> Self {
        Self {
            seed: 20260622,
            output: None,
            adversarial: false,
        }
    }
}

fn run_accuracy_cli(args: &[String]) -> XtaskResult<()> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_accuracy_usage();
        return Ok(());
    }
    let opts = parse_accuracy_options(args)?;
    let artifact = if opts.adversarial {
        run_adversarial_fixture(opts.seed)?
    } else {
        run_accuracy_fixture(opts.seed)?
    };
    let text = serde_json::to_string_pretty(&artifact)?;
    match opts.output {
        Some(path) => {
            fs::write(&path, format!("{text}\n"))?;
            println!("wrote {}", path.display());
        }
        None => println!("{text}"),
    }
    Ok(())
}

fn run_release_preflight_cli(args: &[String]) -> XtaskResult<()> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_release_preflight_usage();
        return Ok(());
    }
    let opts = parse_accuracy_options(args)?;
    if opts.adversarial {
        return Err("--adversarial belongs to `xtask accuracy`, not `release-preflight`".into());
    }
    let artifact = run_release_preflight(opts.seed)?;
    let text = serde_json::to_string_pretty(&artifact)?;
    match opts.output {
        Some(path) => {
            fs::write(&path, format!("{text}\n"))?;
            println!("wrote {}", path.display());
        }
        None => println!("{text}"),
    }
    Ok(())
}

fn run_bit_repro_cli(args: &[String]) -> XtaskResult<()> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_bit_repro_usage();
        return Ok(());
    }
    let opts = parse_accuracy_options(args)?;
    if opts.adversarial {
        return Err("--adversarial belongs to `xtask accuracy`, not `bit-repro`".into());
    }
    let output = opts.output.ok_or(
        "bit-repro requires --output PATH (run it twice with the same --seed \
                and diff the two files to exercise the cross-run gate)",
    )?;
    let bytes = run_bit_repro_fixture(opts.seed)?;
    fs::write(&output, &bytes)?;
    println!("wrote {} ({} bytes)", output.display(), bytes.len());
    Ok(())
}

fn parse_accuracy_options(args: &[String]) -> XtaskResult<AccuracyOptions> {
    let mut opts = AccuracyOptions::default();
    let mut i = 0usize;
    while i < args.len() {
        match args.get(i).map(String::as_str) {
            Some("--seed") => {
                let raw = args.get(i + 1).ok_or("missing value after --seed")?;
                opts.seed = raw.parse::<u64>()?;
                i += 2;
            }
            Some("--output") => {
                let raw = args.get(i + 1).ok_or("missing value after --output")?;
                opts.output = Some(PathBuf::from(raw));
                i += 2;
            }
            Some("--adversarial") => {
                opts.adversarial = true;
                i += 1;
            }
            Some(other) => return Err(format!("unknown accuracy option `{other}`").into()),
            None => break,
        }
    }
    Ok(opts)
}

fn print_bit_repro_usage() {
    println!(
        "USAGE: cargo run -p xtask -- bit-repro --output PATH [--seed N]\n\n\
         Fits the committed synthetic order-3 fixture once (default seed 20260622) and\n\
         writes its encoded bytes to PATH. This is spec §13.4's promised cross-run\n\
         (two-process) reproducibility gate: invoke it twice, as two separate `cargo\n\
         run` process invocations with the same --seed but different --output paths,\n\
         then diff the two files — byte-for-byte equality guards against ambient\n\
         nondeterminism a single in-process thread-count test cannot see (hash-seed/\n\
         ASLR effects entering an unnoticed path, env-sensitive branches, future\n\
         getrandom leakage). CI wires this up in the `test` job of ci.yml."
    );
}

fn print_accuracy_usage() {
    println!(
        "USAGE: cargo run -p xtask -- accuracy [--seed N] [--output PATH] [--adversarial]\n\n\
         Fits the committed synthetic order-3 fixture, verifies Exact mode + all five\
         decomposition gates, then emits deviance/lift/ordered-Gini JSON. With\
         --adversarial, forces the §08 SparseFallback table-budget path and emits\
         a budget-firewall artifact."
    );
}

fn print_release_preflight_usage() {
    println!(
        "USAGE: cargo run -p xtask -- release-preflight [--seed N] [--output PATH]\n\n\
         Runs the lever checkpoint: exactness-gated accuracy smoke, MVS+QHIST smoke,\
         ensemble fork candidates, and SparseFallback adversarial budget stress."
    );
}

fn run_accuracy_fixture(seed: u64) -> XtaskResult<serde_json::Value> {
    let raw = synthetic_fixture(seed, 192)?;
    let split = deterministic_split(seed, raw.y.len())?;
    let train = select_fixture_rows(&raw, &split.train);
    let test = select_fixture_rows(&raw, &split.test);
    let x_train = build_train_matrix(&train)?;

    let loss = SquaredError;
    let baseline = fit_score_candidate(
        "baseline",
        accuracy_config(BoosterConfig::default()),
        &x_train,
        &train,
        &test,
        seed,
        &loss,
    )?;
    let refit = fit_score_candidate(
        "refit_ridge",
        accuracy_config(BoosterConfig {
            refit_leaves: RefitSpec::Ridge {
                l2: 1.0e-6,
                max_iter: 2,
                every_k_trees: None,
            },
            ..BoosterConfig::default()
        }),
        &x_train,
        &train,
        &test,
        seed,
        &loss,
    )?;
    let agbm = fit_score_candidate(
        "agbm",
        accuracy_config(BoosterConfig {
            nesterov: NesterovSpec::Agbm {
                momentum_correction: false,
            },
            ..BoosterConfig::default()
        }),
        &x_train,
        &train,
        &test,
        seed,
        &loss,
    )?;
    let agbm_correction = fit_score_candidate(
        "agbm_momentum_correction",
        accuracy_config(BoosterConfig {
            nesterov: NesterovSpec::Agbm {
                momentum_correction: true,
            },
            ..BoosterConfig::default()
        }),
        &x_train,
        &train,
        &test,
        seed,
        &loss,
    )?;
    let candidates = vec![
        baseline.to_fork_json(),
        refit.to_fork_json(),
        agbm.to_fork_json(),
        agbm_correction.to_fork_json(),
    ];
    let best = [&baseline, &refit, &agbm, &agbm_correction]
        .into_iter()
        .min_by(|a, b| {
            a.deviance
                .total_cmp(&b.deviance)
                .then_with(|| a.name.cmp(b.name))
        })
        .ok_or("accuracy fork candidate set is empty")?;

    Ok(json!({
        "schema_version": 1,
        "fixture": raw.name,
        "objective": "squared_error",
        "seed": seed,
        "split": {
            "train_rows": train.y.len(),
            "test_rows": test.y.len()
        },
        "exactness": {
            "mode": "Exact",
            "feature_budget": true,
            "decomposition": true
        },
        "model": {
            "n_trees": baseline.n_trees,
            "n_features": baseline.n_features,
            "hist_precision": "QuantizedI32",
            "sampling": "Mvs",
            "table_budget_beta": f64::from(InteractionPolicy::default().table_budget_beta),
            "bincode_bytes": baseline.encoded_len
        },
        "metrics": {
            "deviance": baseline.deviance,
            "ordered_gini": baseline.ordered_gini,
            "lift": baseline.lift
        },
        "fork_resolution": {
            "candidates": candidates,
            "best_smoke_candidate": best.name,
            "decision": {
                "refit_default": "off",
                "agbm_default": "off",
                "reason": "Smoke results are recorded for regression visibility; default-on requires repayment on the external accuracy corpus."
            }
        }
    }))
}

fn run_adversarial_fixture(seed: u64) -> XtaskResult<serde_json::Value> {
    let raw = synthetic_fixture(seed, 192)?;
    let x_train = build_train_matrix(&raw)?;
    let loss = SquaredError;
    let spec = FitSpec {
        loss: &loss,
        weight: None,
        exposure: None,
        monotone: MonotoneMap::new(),
        interaction: InteractionPolicy::default(),
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed,
    };
    let model = Booster::with_config(accuracy_config(BoosterConfig::default()))
        .fit(&x_train, &raw.y, &spec)?;
    if model.mode != ExactnessMode::Exact {
        return Err("adversarial harness refuses to score a non-Exact model".into());
    }
    check_feature_budget(&model)?;

    let serve = ServeBinnedMatrix(x_train);
    let budget = TableBudget {
        max_table_cells: 1,
        max_bank_cells: 1,
        on_overflow: OverflowPolicy::SparseFallback {
            density_threshold: 1.0,
        },
    };
    let started = Instant::now();
    let bank = model.explain_with_budget(&serve, RefMeasure::default(), budget)?;
    let purification_ms = started.elapsed().as_secs_f64() * 1000.0;
    assert_exact_decomposition(&model, &bank, &serve)?;

    let sparse_tables = bank
        .tables
        .iter()
        .filter(|table| table.values.is_sparse())
        .count();
    // Named `triples` historically; it is really "sparse tables of an INTERACTION order",
    // i.e. everything the readability budget is meant to bite on. Pinning it at `== 3` would
    // under-count the moment order 4..8 exists, and this metric is the adversarial-budget
    // report's evidence that SparseFallback engaged where it matters.
    let sparse_triples = bank
        .tables
        .iter()
        .filter(|table| {
            table.u.order() >= t_boost_core::engine::LEGACY_MAX_ORDER && table.values.is_sparse()
        })
        .count();
    if sparse_tables == 0 {
        return Err("adversarial budget did not engage SparseFallback".into());
    }

    Ok(json!({
        "schema_version": 1,
        "fixture": raw.name,
        "mode": "adversarial_table_budget",
        "seed": seed,
        "exactness": {
            "mode": "Exact",
            "feature_budget": true,
            "decomposition": true
        },
        "budget": {
            "max_table_cells": budget.max_table_cells,
            "max_bank_cells": budget.max_bank_cells,
            "overflow_policy": "SparseFallback",
            "density_threshold": 1.0
        },
        "tables": {
            "total": bank.tables.len(),
            "sparse": sparse_tables,
            "sparse_triples": sparse_triples
        },
        "perf": {
            "purification_ms": finite_or_zero(purification_ms)
        }
    }))
}

fn run_release_preflight(seed: u64) -> XtaskResult<serde_json::Value> {
    let accuracy = run_accuracy_fixture(seed)?;
    let adversarial = run_adversarial_fixture(seed)?;

    let candidates = accuracy["fork_resolution"]["candidates"]
        .as_array()
        .ok_or("accuracy artifact missing fork candidates")?;
    let fork_candidates_exact = candidates
        .iter()
        .all(|candidate| candidate["exact"].as_bool().unwrap_or(false));
    let qhist_on = accuracy["model"]["hist_precision"] == "QuantizedI32";
    let mvs_on = accuracy["model"]["sampling"] == "Mvs";
    let table_budget_beta_on = accuracy["model"]["table_budget_beta"]
        .as_f64()
        .is_some_and(|beta| beta > 0.0);
    let sparse_on = adversarial["tables"]["sparse"].as_u64().unwrap_or(0) > 0;
    let sparse_triple_on = adversarial["tables"]["sparse_triples"]
        .as_u64()
        .unwrap_or(0)
        > 0;
    let exactness_green = accuracy["exactness"]["mode"] == "Exact"
        && accuracy["exactness"]["decomposition"] == true
        && accuracy["exactness"]["feature_budget"] == true
        && adversarial["exactness"]["mode"] == "Exact"
        && adversarial["exactness"]["decomposition"] == true
        && adversarial["exactness"]["feature_budget"] == true;

    if !(fork_candidates_exact
        && qhist_on
        && mvs_on
        && table_budget_beta_on
        && sparse_on
        && sparse_triple_on
        && exactness_green)
    {
        return Err("internal release preflight did not clear every local lever gate".into());
    }

    Ok(json!({
        "schema_version": 1,
        "seed": seed,
        "status": "green",
        "internal": {
            "exactness": true,
            "mvs": true,
            "quantized_histograms": true,
            "table_budget_beta": true,
            "sparse_fallback": true,
            "sparse_triples": true,
            "ensemble_candidates_exact": true
        },
        "artifacts": {
            "accuracy": accuracy,
            "adversarial": adversarial
        }
    }))
}

#[derive(Debug, Clone)]
struct CandidateMetrics {
    name: &'static str,
    n_trees: usize,
    n_features: usize,
    encoded_len: usize,
    deviance: f32,
    ordered_gini: f64,
    lift: Vec<serde_json::Value>,
}

impl CandidateMetrics {
    fn to_fork_json(&self) -> serde_json::Value {
        json!({
            "name": self.name,
            "deviance": self.deviance,
            "ordered_gini": self.ordered_gini,
            "n_trees": self.n_trees,
            "bincode_bytes": self.encoded_len,
            "exact": true
        })
    }
}

// ---------------------------------------------------------------------------
// `xtask bit-repro`: spec §13.4's promised cross-run (two-process) reproducibility
// gate. Fits the same fixture as `accuracy` but only encodes it — no scoring, no
// fork comparison — so the CLI can be invoked twice as separate OS processes and the
// two output files diffed byte-for-byte. In-process thread-count determinism (the
// existing `tests/determinism.rs` harness) cannot see ambient sources that are fixed
// within a single process (hash-seed selection, ASLR, env-sensitive branches); this
// closes that gap by actually spawning two processes (in CI) and comparing outputs.
// ---------------------------------------------------------------------------

fn run_bit_repro_fixture(seed: u64) -> XtaskResult<Vec<u8>> {
    let raw = synthetic_fixture(seed, 192)?;
    let x_train = build_train_matrix(&raw)?;
    let loss = SquaredError;
    let booster = Booster::with_config(accuracy_config(BoosterConfig::default()));
    let spec = FitSpec {
        loss: &loss,
        weight: None,
        exposure: None,
        monotone: MonotoneMap::new(),
        interaction: InteractionPolicy::default(),
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed,
    };
    let model = booster.fit(&x_train, &raw.y, &spec)?;
    if model.mode != ExactnessMode::Exact {
        return Err("bit-repro harness refuses to encode a non-Exact model".into());
    }
    Ok(encode_model(&model)?)
}

fn accuracy_config(boosters: BoosterConfig) -> Config {
    Config {
        lambda_scale_invariant: false,
        max_delta_step_gated: Default::default(),
        n_trees: 40,
        learning_rate: 0.25,
        lambda: 1.0,
        min_split_gain: 0.0,
        max_delta_step: None,
        sampling: Sampling::Mvs {
            rate: 0.75,
            min_rows: 48,
        },
        l1_leaf: 0.0,
        colsample_bytree: 1.0,
        learning_rate_decay: 0.0,
        validation_fraction: None,
        early_stopping_rounds: 50,
        early_stopping_adaptive: None,
        early_stopping_min_delta: 0.0,
        interaction_gain_hurdle: 0.0,
        interaction_gain_hurdle_mode: t_boost_core::InteractionGainHurdleMode::Fixed,
        leaf_refine_steps: 0,
        leaf_refine_backtracks: 4,
        refine_closed_form_tier2: false,
        incremental_mu: false,
        hist_precision: HistPrecision::QuantizedI32,
        boosters,
    }
}

fn fit_score_candidate(
    name: &'static str,
    config: Config,
    x_train: &BinnedMatrix,
    train: &RawFixture,
    test: &RawFixture,
    seed: u64,
    loss: &SquaredError,
) -> XtaskResult<CandidateMetrics> {
    let booster = Booster::with_config(config);
    let spec = FitSpec {
        loss,
        weight: None,
        exposure: None,
        monotone: MonotoneMap::new(),
        interaction: InteractionPolicy::default(),
        credibility: CredibilityFloor::default(),
        fixed_holdout: None,
        bag_groups: None,
        seed,
    };
    let model = booster.fit(x_train, &train.y, &spec)?;

    let serve_train = ServeBinnedMatrix(x_train.clone());
    let bank = model.explain(&serve_train, RefMeasure::default())?;
    if model.mode != ExactnessMode::Exact {
        return Err("accuracy harness refuses to score a non-Exact model".into());
    }
    check_feature_budget(&model)?;
    assert_exact_decomposition(&model, &bank, &serve_train)?;

    let x_test = bin_like_model(test, &model.grids, &model.provenance)?;
    let mut raw_pred = vec![0.0_f32; x_test.n_rows as usize];
    model.score_trees(&x_test, None, &mut raw_pred)?;
    let pred = model.predict_binned(&x_test, None)?;
    let weight = vec![1.0_f32; test.y.len()];
    let deviance = loss.deviance(&test.y, &raw_pred, &weight)?;
    let lift = lift_curve(&test.y, &pred, &weight, 10);
    let ordered_gini = ordered_gini(&test.y, &pred, &weight);
    let encoded_len = encode_model(&model)?.len();

    Ok(CandidateMetrics {
        name,
        n_trees: model.trees.len(),
        n_features: model.grids.len(),
        encoded_len,
        deviance,
        ordered_gini,
        lift,
    })
}

#[derive(Debug, Clone)]
struct RawFixture {
    name: &'static str,
    columns: Vec<Vec<f32>>,
    y: Vec<f32>,
}

fn synthetic_fixture(seed: u64, n_rows: usize) -> XtaskResult<RawFixture> {
    let mut columns = vec![
        Vec::with_capacity(n_rows),
        Vec::with_capacity(n_rows),
        Vec::with_capacity(n_rows),
    ];
    let mut y = Vec::with_capacity(n_rows);
    for row in 0..n_rows {
        let block = u32::try_from(row)?;
        let jitter = unit_from_seed(seed, 0, Stage::Binning, block) as f32;
        let x0 = ((row * 17 + 3) % 29) as f32 + 0.01 * jitter;
        let x1 = ((row * 11 + 5) % 23) as f32 + 0.02 * (1.0 - jitter);
        let x2 = ((row * 7 + 13) % 19) as f32;
        let f0 = if x0 <= 10.0 { 1.2 } else { -0.4 };
        let f1 = if x1 <= 8.0 { 0.8 } else { -0.3 };
        let f2 = if x2 <= 6.0 { 0.5 } else { -0.1 };
        let pair = if x0 <= 10.0 && x1 > 8.0 { 0.7 } else { -0.2 };
        let triple = if x0 > 10.0 && x1 <= 8.0 && x2 <= 6.0 {
            0.9
        } else {
            0.0
        };
        columns[0].push(x0);
        columns[1].push(x1);
        columns[2].push(x2);
        y.push(10.0 + f0 + f1 + f2 + pair + triple);
    }
    Ok(RawFixture {
        name: "synthetic_order3",
        columns,
        y,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct SplitRows {
    train: Vec<usize>,
    test: Vec<usize>,
}

fn deterministic_split(seed: u64, n_rows: usize) -> XtaskResult<SplitRows> {
    let mut keyed = Vec::with_capacity(n_rows);
    for row in 0..n_rows {
        keyed.push((
            t_boost_core::pb_seed(seed, 0, Stage::Sample as u32, u32::try_from(row)?),
            row,
        ));
    }
    keyed.sort_unstable_by_key(|(key, row)| (*key, *row));
    let n_test = (n_rows / 4).max(1);
    let mut test: Vec<usize> = keyed.iter().take(n_test).map(|(_, row)| *row).collect();
    let mut train: Vec<usize> = keyed.iter().skip(n_test).map(|(_, row)| *row).collect();
    train.sort_unstable();
    test.sort_unstable();
    Ok(SplitRows { train, test })
}

fn select_fixture_rows(fixture: &RawFixture, rows: &[usize]) -> RawFixture {
    let columns = fixture
        .columns
        .iter()
        .map(|col| rows.iter().filter_map(|&r| col.get(r).copied()).collect())
        .collect();
    let y = rows
        .iter()
        .filter_map(|&r| fixture.y.get(r).copied())
        .collect();
    RawFixture {
        name: fixture.name,
        columns,
        y,
    }
}

fn build_train_matrix(fixture: &RawFixture) -> Result<BinnedMatrix, PbError> {
    let refs: Vec<&[f32]> = fixture.columns.iter().map(Vec::as_slice).collect();
    bin_columns(&refs, None, &BinConfig::default(), 0)
}

fn bin_like_model(
    fixture: &RawFixture,
    grids: &[t_boost_core::BorderGrid],
    provenance: &[t_boost_core::AxisProvenance],
) -> Result<BinnedMatrix, PbError> {
    if fixture.columns.len() != grids.len() {
        return Err(PbError::ShapeMismatch {
            what: format!(
                "fixture columns {} != model grids {}",
                fixture.columns.len(),
                grids.len()
            ),
        });
    }
    let mut data = Vec::with_capacity(fixture.columns.len());
    for (axis, col) in fixture.columns.iter().enumerate() {
        let grid = grids.get(axis).ok_or_else(|| PbError::Internal {
            what: "grid disappeared while binning fixture".into(),
        })?;
        let bins: Result<Vec<u8>, PbError> = col.iter().map(|&v| bin(v, grid)).collect();
        data.push(bins?);
    }
    Ok(BinnedMatrix {
        data,
        n_rows: u32::try_from(fixture.y.len()).map_err(|_| PbError::InvalidInput {
            what: "fixture has more than u32::MAX rows".into(),
        })?,
        grids: grids.to_vec(),
        provenance: provenance.to_vec(),
    })
}

fn unit_from_seed(seed: u64, round: u32, stage: Stage, block: u32) -> f64 {
    let bits = t_boost_core::pb_seed(seed, round, stage as u32, block);
    ((bits >> 11) as f64) / ((1_u64 << 53) as f64)
}

fn lift_curve(y: &[f32], pred: &[f32], weight: &[f32], buckets: usize) -> Vec<serde_json::Value> {
    if y.is_empty() || pred.len() != y.len() || weight.len() != y.len() || buckets == 0 {
        return Vec::new();
    }
    let total_w: f64 = weight.iter().map(|&w| f64::from(w.max(0.0))).sum();
    let total_yw: f64 = y
        .iter()
        .zip(weight)
        .map(|(&yi, &wi)| f64::from(yi) * f64::from(wi.max(0.0)))
        .sum();
    let overall = if total_w > 0.0 {
        total_yw / total_w
    } else {
        0.0
    };
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_by(|&a, &b| pred[b].total_cmp(&pred[a]).then_with(|| a.cmp(&b)));
    let mut out = Vec::new();
    let n = y.len();
    for bucket in 0..buckets {
        let start = bucket * n / buckets;
        let end = ((bucket + 1) * n / buckets).min(n);
        if start >= end {
            continue;
        }
        let mut w_sum = 0.0_f64;
        let mut y_sum = 0.0_f64;
        let mut p_sum = 0.0_f64;
        for &idx in &order[start..end] {
            let w = f64::from(weight[idx].max(0.0));
            w_sum += w;
            y_sum += f64::from(y[idx]) * w;
            p_sum += f64::from(pred[idx]) * w;
        }
        let mean_y = if w_sum > 0.0 { y_sum / w_sum } else { 0.0 };
        let mean_pred = if w_sum > 0.0 { p_sum / w_sum } else { 0.0 };
        let lift = if overall.abs() > f64::EPSILON {
            mean_y / overall
        } else {
            0.0
        };
        out.push(json!({
            "bucket": bucket + 1,
            "rows": end - start,
            "mean_y": finite_or_zero(mean_y),
            "mean_pred": finite_or_zero(mean_pred),
            "lift": finite_or_zero(lift)
        }));
    }
    out
}

fn ordered_gini(y: &[f32], pred: &[f32], weight: &[f32]) -> f64 {
    let model = concentration_gini(y, pred, weight);
    let perfect_scores: Vec<f32> = y.to_vec();
    let perfect = concentration_gini(y, &perfect_scores, weight);
    if perfect.abs() <= f64::EPSILON {
        0.0
    } else {
        finite_or_zero(model / perfect)
    }
}

fn concentration_gini(y: &[f32], score: &[f32], weight: &[f32]) -> f64 {
    if y.is_empty() || score.len() != y.len() || weight.len() != y.len() {
        return 0.0;
    }
    let total_w: f64 = weight.iter().map(|&w| f64::from(w.max(0.0))).sum();
    let total_yw: f64 = y
        .iter()
        .zip(weight)
        .map(|(&yi, &wi)| f64::from(yi.max(0.0)) * f64::from(wi.max(0.0)))
        .sum();
    if total_w <= 0.0 || total_yw <= 0.0 {
        return 0.0;
    }
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then_with(|| a.cmp(&b)));

    let mut prev_x = 0.0_f64;
    let mut prev_y = 0.0_f64;
    let mut cum_w = 0.0_f64;
    let mut cum_y = 0.0_f64;
    let mut area = 0.0_f64;
    for idx in order {
        let w = f64::from(weight[idx].max(0.0));
        cum_w += w;
        cum_y += f64::from(y[idx].max(0.0)) * w;
        let x = cum_w / total_w;
        let yy = cum_y / total_yw;
        area += (x - prev_x) * (yy + prev_y) * 0.5;
        prev_x = x;
        prev_y = yy;
    }
    finite_or_zero(2.0 * area - 1.0)
}

fn finite_or_zero(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Run one gate over the shipped sources and translate its findings into an exit code.
fn run_gate(gate: GateFn) -> ExitCode {
    let files = match load_shipped_sources() {
        Ok(files) => files,
        Err(err) => {
            eprintln!("xtask: could not enumerate shipped sources: {err}");
            return ExitCode::FAILURE;
        }
    };
    let violations = gate(&files);
    if violations.is_empty() {
        ExitCode::SUCCESS
    } else {
        for v in &violations {
            println!("{v}");
        }
        eprintln!("xtask: {} violation(s)", violations.len());
        ExitCode::FAILURE
    }
}

/// A loaded shipped source file: its display path plus its lines.
struct SourceFile {
    path: PathBuf,
    lines: Vec<String>,
}

/// One gate violation, rendered as `path:line: message`.
struct Violation {
    path: PathBuf,
    line: usize,
    message: String,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}: {}", self.path.display(), self.line, self.message)
    }
}

/// Locate the workspace root from `CARGO_MANIFEST_DIR` (xtask's own dir → parent),
/// falling back to the current directory.
fn workspace_root() -> PathBuf {
    if let Ok(dir) = env::var("CARGO_MANIFEST_DIR") {
        if let Some(parent) = Path::new(&dir).parent() {
            return parent.to_path_buf();
        }
    }
    env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Collect every shipped Rust source file (`crates/*/src/**/*.rs`), excluding any
/// `tests/`/`benches/` directory and this `xtask` crate — i.e. exactly the files
/// the no-panic / wire-width gates govern (unwrap-allowed set = `{tests, benches, xtask}`).
fn load_shipped_sources() -> std::io::Result<Vec<SourceFile>> {
    let root = workspace_root();
    let crates = root.join("crates");
    let mut rs_files = Vec::new();
    if crates.is_dir() {
        collect_rs(&crates, &mut rs_files)?;
    }
    let mut out = Vec::new();
    for path in rs_files {
        let text = fs::read_to_string(&path)?;
        let lines = text.lines().map(str::to_owned).collect();
        let display = path.strip_prefix(&root).unwrap_or(&path).to_path_buf();
        out.push(SourceFile {
            path: display,
            lines,
        });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

/// Recursively gather `.rs` files under `dir`, skipping `tests`/`benches` subtrees.
fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            let name = entry.file_name();
            if name == "tests" || name == "benches" || name == "target" {
                continue;
            }
            collect_rs(&path, out)?;
        } else if file_type.is_file() && path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Gate: no `Box<dyn Error>` in shipped code (§02.4 — the error model is `PbError`).
// ---------------------------------------------------------------------------

fn check_no_box_dyn(files: &[SourceFile]) -> Vec<Violation> {
    let mut out = Vec::new();
    for file in files {
        for (i, line) in file.lines.iter().enumerate() {
            let code = strip_line_comment(line);
            if code.contains("Box<dyn") && code.contains("Error") {
                out.push(Violation {
                    path: file.path.clone(),
                    line: i + 1,
                    message: "`Box<dyn Error>` is forbidden in shipped code; return `PbError`"
                        .into(),
                });
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Gate: `// JUSTIFIED:` proof required on panic-adjacent sites and form-(b) allows.
// Skips `#[cfg(test)]` modules (the unwrap-allowed set covers test code).
// ---------------------------------------------------------------------------

fn check_justified(files: &[SourceFile]) -> Vec<Violation> {
    // The last two triggers match the bare lint path, not a literal `#[allow(...)]`
    // string: a single-line trigger string can never match the module-blanket
    // `#![allow(\n    clippy::indexing_slicing,\n    ...\n)]` form (inner-attribute `!`,
    // and the lint name is on its own line), so shipped modules that scope-allow the
    // lint at module level sailed through unexamined. Matching the lint path alone
    // catches `#[allow(...)]`, `#![allow(...)]`, and any single- or multi-line spelling
    // of either form.
    let triggers: [&str; 6] = [
        ".unwrap(",
        ".expect(",
        "panic!",
        "unreachable!",
        "clippy::indexing_slicing",
        "clippy::arithmetic_side_effects",
    ];
    let mut out = Vec::new();
    for file in files {
        let in_test = test_module_mask(&file.lines);
        for (i, line) in file.lines.iter().enumerate() {
            if in_test[i] {
                continue;
            }
            let code = strip_line_comment(line);
            let trimmed = code.trim_start();
            // Skip doc/comment lines entirely (only real code triggers the gate).
            if trimmed.starts_with("//") {
                continue;
            }
            if triggers.iter().any(|t| code.contains(t)) && !has_justification(&file.lines, i) {
                out.push(Violation {
                    path: file.path.clone(),
                    line: i + 1,
                    message: "panic-adjacent site or proven-unchecked `#[allow]` lacks a `// JUSTIFIED:` proof"
                        .into(),
                });
            }
        }
    }
    out
}

/// A site is justified if a `// JUSTIFIED:` comment is on the same line or on the
/// nearest preceding non-blank line.
fn has_justification(lines: &[String], idx: usize) -> bool {
    if lines.get(idx).is_some_and(|l| l.contains("// JUSTIFIED:")) {
        return true;
    }
    let mut j = idx;
    while j > 0 {
        j -= 1;
        let prev = match lines.get(j) {
            Some(p) => p.trim(),
            None => return false,
        };
        if prev.is_empty() {
            continue;
        }
        return prev.contains("// JUSTIFIED:");
    }
    false
}

/// Mark every line that falls inside a `#[cfg(test)]` module body (brace-tracked).
/// Such lines are in the unwrap-allowed set and are exempt from `check_justified`.
fn test_module_mask(lines: &[String]) -> Vec<bool> {
    let mut mask = vec![false; lines.len()];
    let mut i = 0;
    while i < lines.len() {
        let line = match lines.get(i) {
            Some(l) => l,
            None => break,
        };
        if line.contains("#[cfg(test)]") {
            // Find the opening brace of the following `mod`, then skip to its match.
            let mut j = i;
            let mut depth = 0usize;
            let mut started = false;
            while j < lines.len() {
                if let Some(l) = lines.get(j) {
                    let opens = l.matches('{').count();
                    let closes = l.matches('}').count();
                    if opens > 0 {
                        started = true;
                    }
                    depth = depth.saturating_add(opens).saturating_sub(closes);
                    if let Some(m) = mask.get_mut(j) {
                        *m = true;
                    }
                }
                if started && depth == 0 {
                    break;
                }
                j += 1;
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    mask
}

// ---------------------------------------------------------------------------
// Gates: forbidden field types on serialized (de)serialize-deriving items.
// ---------------------------------------------------------------------------

fn check_no_usize_serialized(files: &[SourceFile]) -> Vec<Violation> {
    serialized_field_gate(
        files,
        &["usize", "isize"],
        "`usize`/`isize` on a serialized type breaks cross-platform wire width (§13.4); use a fixed-width int",
    )
}

fn check_no_hashmap_serialized(files: &[SourceFile]) -> Vec<Violation> {
    serialized_field_gate(
        files,
        &["HashMap", "HashSet"],
        "`HashMap`/`HashSet` on a serialized type has nondeterministic order; use `BTreeMap`/`Vec`",
    )
}

/// Scan each `Serialize`/`Deserialize`-deriving struct/enum body and flag any field
/// line that mentions one of `forbidden` as a whole token.
fn serialized_field_gate(
    files: &[SourceFile],
    forbidden: &[&str],
    message: &str,
) -> Vec<Violation> {
    let mut out = Vec::new();
    for file in files {
        let mut serialized_pending = false;
        let mut i = 0;
        while i < file.lines.len() {
            let line = match file.lines.get(i) {
                Some(l) => l,
                None => break,
            };
            let code = strip_line_comment(line);
            let trimmed = code.trim_start();

            if trimmed.starts_with("#[")
                && (code.contains("Serialize") || code.contains("Deserialize"))
            {
                serialized_pending = true;
                i += 1;
                continue;
            }

            let is_item = trimmed.starts_with("struct ")
                || trimmed.starts_with("pub struct ")
                || trimmed.starts_with("pub(crate) struct ")
                || trimmed.starts_with("enum ")
                || trimmed.starts_with("pub enum ")
                || trimmed.starts_with("pub(crate) enum ");

            if is_item {
                if serialized_pending {
                    i = scan_item_body(file, i, forbidden, message, &mut out);
                    serialized_pending = false;
                    continue;
                }
                serialized_pending = false;
            } else if !trimmed.is_empty()
                && !trimmed.starts_with("//")
                && !trimmed.starts_with("#[")
            {
                // A non-attribute, non-item code line breaks the derive→item adjacency.
                serialized_pending = false;
            }
            i += 1;
        }
    }
    out
}

/// Scan a single item body starting at the item keyword line. Handles brace bodies
/// `{ .. }` and one-line tuple/unit forms ending in `;`. Returns the index just past
/// the body.
fn scan_item_body(
    file: &SourceFile,
    start: usize,
    forbidden: &[&str],
    message: &str,
    out: &mut Vec<Violation>,
) -> usize {
    // Find the first delimiter to decide brace-body vs tuple/one-liner.
    let mut i = start;
    let mut depth = 0usize;
    let mut started = false;
    while i < file.lines.len() {
        let line = match file.lines.get(i) {
            Some(l) => l,
            None => break,
        };
        let code = strip_line_comment(line);
        let opens = code.matches('{').count();
        let closes = code.matches('}').count();
        if opens > 0 {
            started = true;
        }
        if i > start {
            // Inside the body — check field tokens (skip the item keyword line itself).
            // Applies whether the body is brace-delimited (`started`) or a brace-less
            // multi-line tuple struct spanning to a `;` (`!started`) — a field on any
            // line after the item keyword must be checked either way.
            flag_forbidden(file, i, &code, forbidden, message, out);
        } else if i == start {
            // Tuple/unit struct on the same line, e.g. `pub struct X(pub usize);`.
            if !code.contains('{') {
                flag_forbidden(file, i, &code, forbidden, message, out);
                if code.contains(';') {
                    return i + 1;
                }
            }
        }
        depth = depth.saturating_add(opens).saturating_sub(closes);
        if started && depth == 0 {
            return i + 1;
        }
        // Tuple struct spanning to a `;` without braces — this line was already
        // checked by the unconditional `i > start` branch above.
        if !started && code.contains(';') && i > start {
            return i + 1;
        }
        i += 1;
    }
    i
}

/// Push a violation for each forbidden whole-token found on this line.
fn flag_forbidden(
    file: &SourceFile,
    idx: usize,
    code: &str,
    forbidden: &[&str],
    message: &str,
    out: &mut Vec<Violation>,
) {
    for tok in forbidden {
        if contains_token(code, tok) {
            out.push(Violation {
                path: file.path.clone(),
                line: idx + 1,
                message: message.into(),
            });
            break;
        }
    }
}

/// Whole-token containment: `tok` bounded by non-identifier characters (so `usize`
/// matches `n: usize` but not `my_usize_thing`).
fn contains_token(haystack: &str, tok: &str) -> bool {
    let bytes = haystack.as_bytes();
    let mut from = 0;
    while let Some(rel) = haystack.get(from..).and_then(|s| s.find(tok)) {
        let at = from + rel;
        // `map_or(true, ..)` rather than `is_none_or` — the latter is stable only
        // since Rust 1.82, above this workspace's 1.74 MSRV.
        let before_ok = at == 0 || bytes.get(at - 1).map_or(true, |b| !is_ident_byte(*b));
        let after = at + tok.len();
        let after_ok = bytes.get(after).map_or(true, |b| !is_ident_byte(*b));
        if before_ok && after_ok {
            return true;
        }
        from = at + tok.len();
    }
    false
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Strip a trailing `//` line comment (best-effort: ignores `//` inside string
/// literals, which the gates do not need to parse precisely).
fn strip_line_comment(line: &str) -> String {
    match line.find("//") {
        Some(idx) => line.get(..idx).unwrap_or("").to_string(),
        None => line.to_string(),
    }
}

#[cfg(test)]
mod accuracy_tests {
    use super::*;

    #[test]
    fn ordered_gini_pins_perfect_and_reversed_rankings() {
        let y = [10.0_f32, 0.0, 0.0, 0.0];
        let w = [1.0_f32; 4];
        let perfect = [4.0_f32, 3.0, 2.0, 1.0];
        let reversed = [1.0_f32, 2.0, 3.0, 4.0];
        assert!((ordered_gini(&y, &perfect, &w) - 1.0).abs() < 1e-12);
        assert!((ordered_gini(&y, &reversed, &w) + 1.0).abs() < 1e-12);
        assert_eq!(ordered_gini(&[0.0, 0.0], &[2.0, 1.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn lift_curve_pins_bucket_means() {
        let y = [8.0_f32, 4.0, 2.0, 2.0];
        let pred = [0.9_f32, 0.8, 0.2, 0.1];
        let w = [1.0_f32; 4];
        let lift = lift_curve(&y, &pred, &w, 2);
        assert_eq!(lift.len(), 2);
        assert_eq!(lift[0]["rows"], 2);
        assert!((lift[0]["mean_y"].as_f64().unwrap() - 6.0).abs() < 1e-12);
        assert!((lift[0]["lift"].as_f64().unwrap() - 1.5).abs() < 1e-12);
        assert!((lift[1]["mean_y"].as_f64().unwrap() - 2.0).abs() < 1e-12);
        assert!((lift[1]["lift"].as_f64().unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn accuracy_artifact_is_seed_stable_and_exactness_gated() {
        let a = run_accuracy_fixture(7).unwrap();
        let b = run_accuracy_fixture(7).unwrap();
        assert_eq!(a, b);
        assert_eq!(a["exactness"]["mode"], "Exact");
        assert_eq!(a["exactness"]["decomposition"], true);
        assert_eq!(a["exactness"]["feature_budget"], true);
        assert!(a["metrics"]["deviance"].as_f64().unwrap().is_finite());
        assert!(a["metrics"]["ordered_gini"].as_f64().unwrap().is_finite());
        assert_eq!(a["fork_resolution"]["decision"]["refit_default"], "off");
        assert_eq!(a["fork_resolution"]["decision"]["agbm_default"], "off");
        assert_eq!(
            a["fork_resolution"]["candidates"].as_array().unwrap().len(),
            4
        );
        for candidate in a["fork_resolution"]["candidates"].as_array().unwrap() {
            assert_eq!(candidate["exact"], true);
            assert!(candidate["deviance"].as_f64().unwrap().is_finite());
        }
    }

    #[test]
    fn adversarial_artifact_is_seed_stable_and_uses_sparse_fallback() {
        let a = run_adversarial_fixture(7).unwrap();
        let b = run_adversarial_fixture(7).unwrap();
        assert_eq!(a["fixture"], b["fixture"]);
        assert_eq!(a["seed"], b["seed"]);
        assert_eq!(a["exactness"], b["exactness"]);
        assert_eq!(a["budget"], b["budget"]);
        assert_eq!(a["tables"], b["tables"]);
        assert_eq!(a["mode"], "adversarial_table_budget");
        assert_eq!(a["exactness"]["mode"], "Exact");
        assert_eq!(a["exactness"]["decomposition"], true);
        assert_eq!(a["budget"]["overflow_policy"], "SparseFallback");
        assert!(a["tables"]["sparse"].as_u64().unwrap() > 0);
        assert!(a["perf"]["purification_ms"].as_f64().unwrap().is_finite());
    }

    #[test]
    fn release_preflight_clears_every_lever_gate() {
        let artifact = run_release_preflight(7).unwrap();
        assert_eq!(artifact["status"], "green");
        assert_eq!(artifact["internal"]["exactness"], true);
        assert_eq!(artifact["internal"]["mvs"], true);
        assert_eq!(artifact["internal"]["quantized_histograms"], true);
        assert_eq!(artifact["internal"]["table_budget_beta"], true);
        assert_eq!(artifact["internal"]["sparse_fallback"], true);
        assert_eq!(artifact["internal"]["sparse_triples"], true);
        assert_eq!(artifact["internal"]["ensemble_candidates_exact"], true);
    }

    #[test]
    fn bit_repro_fixture_is_seed_stable() {
        // In-process sanity check on the harness itself. The gate this subcommand
        // exists for (spec §13.4's cross-run/two-process guarantee) is exercised by
        // CI invoking the CLI twice as separate processes and diffing the two output
        // files — a single process cannot observe the ambient nondeterminism that
        // gate guards against (hash-seed/ASLR effects), so this test only pins that
        // the fitted bytes are a pure function of the seed within one process.
        let a = run_bit_repro_fixture(7).unwrap();
        let b = run_bit_repro_fixture(7).unwrap();
        assert_eq!(a, b);
        assert!(!a.is_empty());
        assert_ne!(a, run_bit_repro_fixture(8).unwrap());
    }

    #[test]
    fn serialized_field_gates_catch_forbidden_fields_and_pass_clean_types() {
        // A Serialize-deriving struct with a HashMap field is flagged.
        let dirty_map = [SourceFile {
            path: PathBuf::from("crates/t-boost-core/src/x.rs"),
            lines: vec![
                "#[derive(Serialize, Deserialize)]".to_owned(),
                "pub struct Bad {".to_owned(),
                "    pub m: HashMap<u32, u32>,".to_owned(),
                "}".to_owned(),
            ],
        }];
        assert_eq!(check_no_hashmap_serialized(&dirty_map).len(), 1);

        // A `usize` field on a serialized struct is flagged.
        let dirty_usize = [SourceFile {
            path: PathBuf::from("x.rs"),
            lines: vec![
                "#[derive(Serialize)]".to_owned(),
                "struct B {".to_owned(),
                "    pub n: usize,".to_owned(),
                "}".to_owned(),
            ],
        }];
        assert_eq!(check_no_usize_serialized(&dirty_usize).len(), 1);

        // A clean serialized struct (BTreeMap + fixed-width) is NOT flagged.
        let clean = [SourceFile {
            path: PathBuf::from("x.rs"),
            lines: vec![
                "#[derive(Serialize, Deserialize)]".to_owned(),
                "pub struct Good {".to_owned(),
                "    pub m: BTreeMap<u32, u32>,".to_owned(),
                "    pub n: u64,".to_owned(),
                "}".to_owned(),
            ],
        }];
        assert!(check_no_hashmap_serialized(&clean).is_empty());
        assert!(check_no_usize_serialized(&clean).is_empty());

        // A NON-serialized struct with a HashMap is out of scope (gate is derive-scoped).
        let non_serialized = [SourceFile {
            path: PathBuf::from("x.rs"),
            lines: vec![
                "pub struct Scratch {".to_owned(),
                "    pub m: HashMap<u32, u32>,".to_owned(),
                "    pub n: usize,".to_owned(),
                "}".to_owned(),
            ],
        }];
        assert!(check_no_hashmap_serialized(&non_serialized).is_empty());
        assert!(check_no_usize_serialized(&non_serialized).is_empty());
    }

    #[test]
    fn serialized_field_gate_catches_brace_less_multiline_tuple_struct() {
        // rustfmt wraps a tuple struct across multiple brace-less lines once it exceeds
        // the line-width limit. Every intermediate field line must still be scanned —
        // previously only the item-keyword line and the final `;` line were checked,
        // so a HashMap/usize buried in the middle sailed through silently.
        let dirty = [SourceFile {
            path: PathBuf::from("x.rs"),
            lines: vec![
                "#[derive(Serialize, Deserialize)]".to_owned(),
                "pub struct Foo(".to_owned(),
                "    pub HashMap<u32, u32>,".to_owned(),
                "    pub usize,".to_owned(),
                ");".to_owned(),
            ],
        }];
        assert_eq!(check_no_hashmap_serialized(&dirty).len(), 1);
        assert_eq!(check_no_usize_serialized(&dirty).len(), 1);

        // A clean brace-less multi-line tuple struct still passes.
        let clean = [SourceFile {
            path: PathBuf::from("x.rs"),
            lines: vec![
                "#[derive(Serialize, Deserialize)]".to_owned(),
                "pub struct Good(".to_owned(),
                "    pub BTreeMap<u32, u32>,".to_owned(),
                "    pub u64,".to_owned(),
                ");".to_owned(),
            ],
        }];
        assert!(check_no_hashmap_serialized(&clean).is_empty());
        assert!(check_no_usize_serialized(&clean).is_empty());
    }

    #[test]
    fn justified_gate_catches_module_blanket_allow_and_accepts_a_justified_one() {
        // The multi-line, inner-attribute `#![allow(...)]` form used for module-level
        // scope-allows was previously invisible to the gate: the single-line literal
        // trigger `"#[allow(clippy::indexing_slicing)]"` cannot match an inner `!`
        // attribute whose lint name sits on its own line. An unjustified module
        // blanket must now be flagged...
        let unjustified = [SourceFile {
            path: PathBuf::from("crates/t-boost-core/src/x.rs"),
            lines: vec![
                "//! A module.".to_owned(),
                String::new(),
                "#![allow(".to_owned(),
                "    clippy::indexing_slicing,".to_owned(),
                "    clippy::needless_range_loop".to_owned(),
                ")]".to_owned(),
            ],
        }];
        let violations = check_justified(&unjustified);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].line, 4);

        // ...while the same form with a `// JUSTIFIED:` comment immediately above the
        // triggering line is accepted, and the pre-existing single-line `#[allow(...)]`
        // form (already working) keeps working.
        let justified = [SourceFile {
            path: PathBuf::from("crates/t-boost-core/src/y.rs"),
            lines: vec![
                "//! A module.".to_owned(),
                String::new(),
                "#![allow(".to_owned(),
                "    // JUSTIFIED: pre-existing module-scoped debt, tracked separately.".to_owned(),
                "    clippy::indexing_slicing,".to_owned(),
                ")]".to_owned(),
                "fn f() {".to_owned(),
                "    // JUSTIFIED: idx built from a fixed 0..8 range.".to_owned(),
                "    #[allow(clippy::indexing_slicing)]".to_owned(),
                "    let _ = 1;".to_owned(),
                "}".to_owned(),
            ],
        }];
        assert!(check_justified(&justified).is_empty());
    }
}
