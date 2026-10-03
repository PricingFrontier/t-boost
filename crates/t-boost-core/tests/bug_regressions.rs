//! Runtime counterexamples from the confirmed bug roadmap, with correctness oracles.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::float_cmp
)]

#[path = "roadmap/bug004.rs"]
mod bug004;
#[path = "roadmap/bug005.rs"]
mod bug005;
#[path = "roadmap/bug006.rs"]
mod bug006;
#[path = "roadmap/bug007.rs"]
mod bug007;
#[path = "roadmap/bug009.rs"]
mod bug009;
#[path = "roadmap/bug010.rs"]
mod bug010;
#[path = "roadmap/bug012.rs"]
mod bug012;
#[path = "roadmap/bug013.rs"]
mod bug013;
#[path = "roadmap/bug031.rs"]
mod bug031;
#[path = "roadmap/bug032.rs"]
mod bug032;
#[path = "roadmap/bug033.rs"]
mod bug033;
#[path = "roadmap/bug035.rs"]
mod bug035;
#[path = "roadmap/bug036.rs"]
mod bug036;
#[path = "roadmap/bug040.rs"]
mod bug040;
#[path = "roadmap/bug048.rs"]
mod bug048;
#[path = "roadmap/bug050.rs"]
mod bug050;
#[path = "roadmap/bug051.rs"]
mod bug051;
#[path = "roadmap/bug052.rs"]
mod bug052;
#[path = "roadmap/bug053.rs"]
mod bug053;
#[path = "roadmap/bug060.rs"]
mod bug060;
#[path = "roadmap/bug061.rs"]
mod bug061;
#[path = "roadmap/bug062.rs"]
mod bug062;
#[path = "roadmap/bug063.rs"]
mod bug063;
#[path = "roadmap/bug064.rs"]
mod bug064;
#[path = "roadmap/input_validation.rs"]
mod input_validation;
