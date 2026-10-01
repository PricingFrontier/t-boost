//! Performance engineering kernels (spec §11).
//!
//! This module owns runtime scoring kernels whose only admissible behavior is
//! bit-equality with the canonical scalar scorer. The first shipped kernel is a
//! deterministic row-tiled path over [`crate::ScoringBank`]; it is intentionally
//! scalar internally today. `multiversion` runtime CPU-feature dispatch is **not** a
//! dependency of this crate yet — spec §13.4's bit-reproducibility-over-serve-speed
//! policy defers it until specific kernels are proven order-exact (bit-identical
//! regardless of vector width); this module's job in the meantime is to give that
//! future work a stable API and test oracle to optimize behind.

use crate::data::BinnedMatrix;
use crate::error::PbError;
use crate::scoring::ScoringBank;

/// Chunk size demonstrating the fixed-order-fold pattern (spec §11 / §02.5) used by
/// the determinism harness (`tests/determinism.rs`): reduce over `par_chunks(CHUNK_ROWS)`
/// combined in index order — never a steal-order rayon `reduce` — so results are
/// byte-identical regardless of thread count.
///
/// This constant does **not** govern the production histogram fold: `engine::hist`'s
/// row-chunk-parallel build uses its own private `ROW_PAR_CHUNK_ROWS = 8_192`
/// (`crates/t-boost-core/src/engine/hist.rs`), which is the value actually frozen by
/// the reproducibility `[GATE]` for histogram accumulation. `CHUNK_ROWS` freezes only
/// the harness's own fold pattern — changing it does not change training's bit-exact
/// output.
pub const CHUNK_ROWS: usize = 4096;

/// Score a row tile through a runtime [`ScoringBank`] in raw-score space.
///
/// Rows are visited in the caller-supplied order, and trees are always streamed in
/// stored model order inside [`ScoringBank::score_row`]. That makes this a strict
/// re-encoding of scalar path-A scoring; future ISA-specific implementations must keep
/// this byte contract.
///
/// Like [`ScoringBank::score_row`], this never adds the model's intercept `f0` on its
/// own: `offset[r]` (when `offset` is `Some`) is used verbatim per row, so callers MUST
/// pre-fold `f0` into every entry (`offset[r] = model.f0 + row_offset[r]`) to match
/// [`crate::Model::score_trees`]'s raw scores; `offset = None` scores relative to zero,
/// not `f0`.
///
/// # Errors
/// [`PbError::ShapeMismatch`] if `out`, `rows`, `offset`, or matrix columns disagree;
/// plus propagated scorer errors if a row lacks a referenced axis.
pub fn score_tile(
    bank: &ScoringBank,
    x: &BinnedMatrix,
    rows: &[u32],
    offset: Option<&[f32]>,
    out: &mut [f32],
) -> Result<(), PbError> {
    if out.len() != rows.len() {
        return Err(PbError::ShapeMismatch {
            what: format!("out len {} != rows len {}", out.len(), rows.len()),
        });
    }
    if let Some(off) = offset {
        if off.len() != x.n_rows as usize {
            return Err(PbError::ShapeMismatch {
                what: format!("offset len {} != n_rows {}", off.len(), x.n_rows),
            });
        }
    }
    let n_rows = x.n_rows as usize;
    for (axis, col) in x.data.iter().enumerate() {
        if col.len() != n_rows {
            return Err(PbError::ShapeMismatch {
                what: format!("column {axis} len {} != n_rows {n_rows}", col.len()),
            });
        }
    }

    let mut row_bins = vec![0u8; x.data.len()];
    for (&row, dst) in rows.iter().zip(out) {
        let r = row as usize;
        if r >= n_rows {
            return Err(PbError::ShapeMismatch {
                what: format!("row id {row} outside n_rows {n_rows}"),
            });
        }
        for (slot, col) in row_bins.iter_mut().zip(&x.data) {
            *slot = *col.get(r).ok_or_else(|| PbError::Internal {
                what: "validated tile row escaped column".into(),
            })?;
        }
        let off = offset.and_then(|v| v.get(r).copied()).unwrap_or(0.0);
        *dst = bank.score_row(&row_bins, off)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic
    )]

    use super::*;
    use crate::engine::{CorrectionBank, CorrectionTable};
    use crate::explain::{fixture_model, fixture_serve};

    #[test]
    fn tiled_scoring_matches_scalar_bank_bit_exactly() {
        let model = fixture_model();
        let x = fixture_serve().0;
        let bank = ScoringBank::from_model(&model).unwrap();
        let rows: Vec<u32> = (0..x.n_rows).rev().collect();
        let offset: Vec<f32> = (0..x.n_rows).map(|i| i as f32 * 0.125).collect();
        let mut out = vec![0.0_f32; rows.len()];
        score_tile(&bank, &x, &rows, Some(&offset), &mut out).unwrap();

        let mut row_bins = vec![0_u8; x.data.len()];
        for (&row, score) in rows.iter().zip(&out) {
            let r = row as usize;
            for (slot, col) in row_bins.iter_mut().zip(&x.data) {
                *slot = col[r];
            }
            let expected = bank.score_row(&row_bins, offset[r]).unwrap();
            assert_eq!(score.to_bits(), expected.to_bits());
        }
    }

    /// `score_tile` delegates straight to `ScoringBank::score_row`; pin that a bank
    /// carrying a §G1 correction (see `scoring.rs`'s own bit-equality test) stays
    /// tile-vs-bank exact too, so a future ISA-specific rewrite can't silently drop it.
    /// TWO tables (not one) — a single-table bank has no summation order to get wrong.
    #[test]
    fn tiled_scoring_matches_scalar_bank_bit_exactly_with_correction() {
        let mut model = fixture_model();
        model.correction = Some(CorrectionBank {
            tables: vec![
                CorrectionTable {
                    axes: vec![0, 1],
                    shape: vec![3, 3],
                    bin_to_cell: vec![vec![0, 1, 2], vec![0, 1, 2]],
                    values: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                },
                CorrectionTable {
                    axes: vec![1],
                    shape: vec![3],
                    bin_to_cell: vec![vec![0, 1, 2]],
                    values: vec![0.0, 0.1, 0.2],
                },
            ],
        });
        let x = fixture_serve().0;
        let bank = ScoringBank::from_model(&model).unwrap();
        let rows: Vec<u32> = (0..x.n_rows).rev().collect();
        let offset: Vec<f32> = (0..x.n_rows).map(|i| i as f32 * 0.125).collect();
        let mut out = vec![0.0_f32; rows.len()];
        score_tile(&bank, &x, &rows, Some(&offset), &mut out).unwrap();

        let mut row_bins = vec![0_u8; x.data.len()];
        for (&row, score) in rows.iter().zip(&out) {
            let r = row as usize;
            for (slot, col) in row_bins.iter_mut().zip(&x.data) {
                *slot = col[r];
            }
            let expected = bank.score_row(&row_bins, offset[r]).unwrap();
            assert_eq!(score.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn tiled_scoring_rejects_bad_shapes() {
        let model = fixture_model();
        let x = fixture_serve().0;
        let bank = ScoringBank::from_model(&model).unwrap();
        let mut out = vec![0.0_f32; 1];
        assert!(matches!(
            score_tile(&bank, &x, &[0, 1], None, &mut out),
            Err(PbError::ShapeMismatch { .. })
        ));
    }
}
