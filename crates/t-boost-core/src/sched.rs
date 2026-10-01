//! Scheduling constants for the fit hot loop: how rayon groups work items into tasks.
//! Nothing here can change any computed value.

/// Minimum CHUNKS per rayon task at the round loop's fixed-chunk parallel sites
/// (histogram row-chunks, grad/hess fills, deviance folds). Those tasks measured ~180µs,
/// small enough that per-round fork-join wake/steal churn eats wide pools; grouping `k`
/// chunks per task cuts the task count k-fold without touching the work items themselves.
///
/// Byte-identity holds at every value: `with_min_len` only constrains how rayon SPLITS the
/// indexed iterator — chunk boundaries stay `PAR_DEVIANCE_CHUNK`/`ROW_PAR_CHUNK_ROWS`, every
/// per-chunk partial is computed from the same rows, and partials are still combined in
/// chunk order (or written to disjoint slices).
///
/// 8 (2026-07-16): the sweep was monotone through 8 — 8-bag 406k MTPL fit 8.76s -> 6.86s
/// (-22%), 1-bag 200k @ 22 threads 4.14s -> 3.78s (-9%) — and the byte gate passed at 1, 4
/// and 8 against the pinned baseline artifacts.
pub(crate) const fn min_chunks_per_task() -> usize {
    8
}
