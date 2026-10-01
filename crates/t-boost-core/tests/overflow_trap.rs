//! `overflow-checks` liveness probe (spec §02.3a / §02.10(6), plan F8). Confirms the
//! `[profile.*] overflow-checks = true` setting is actually in force: a runtime
//! integer overflow MUST trap (panic), never silently wrap. If overflow-checks were
//! off, `x + y` would wrap to 0 and this test would FAIL (no panic) — so it is a live
//! guard on the profile, the runtime half of the integer-safety contract.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

#[test]
#[should_panic(expected = "overflow")]
fn overflow_checks_are_live_in_the_test_profile() {
    // `black_box` defeats const-eval so the add happens at runtime (a const overflow
    // would be a compile error, not the runtime trap we are asserting).
    let x: u8 = std::hint::black_box(255);
    let y: u8 = std::hint::black_box(1);
    let z = x + y;
    std::hint::black_box(z);
}
