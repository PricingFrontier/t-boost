//! Exposure / offset plumbing (spec §03.7). `FitSpec.exposure` sets a per-row
//! `offset = ln(e)` added to the raw score before the inverse link every iteration.
//! This module validates and transforms exposure; §05/§09 consume the offset. The
//! offset never enters binning or the grids — it is a per-row score shift, orthogonal
//! to tree shape, so I1/I2 are untouched.

use crate::error::PbError;

/// Compute the per-row offset `ln(exposure[i])` (spec §03.7), validating that every
/// exposure is finite and strictly positive.
///
/// # Errors
/// [`PbError::ShapeMismatch`] if `exposure.len() != n_rows`; [`PbError::InvalidInput`]
/// if any entry is non-finite or `<= 0`.
pub fn compute_offset(exposure: &[f32], n_rows: usize) -> Result<Vec<f32>, PbError> {
    if exposure.len() != n_rows {
        return Err(PbError::ShapeMismatch {
            what: format!("exposure len {} != n_rows {n_rows}", exposure.len()),
        });
    }
    let mut offset = Vec::with_capacity(exposure.len());
    for &e in exposure {
        if !e.is_finite() || e <= 0.0 {
            return Err(PbError::InvalidInput {
                what: "exposure entries must be finite and > 0".into(),
            });
        }
        offset.push(e.ln());
    }
    Ok(offset)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::float_cmp
    )]
    use super::*;

    #[test]
    fn offset_is_log_exposure() {
        let e = [1.0_f32, std::f32::consts::E, 1.0];
        let off = compute_offset(&e, 3).unwrap();
        assert_eq!(off[0], 0.0); // ln(1) = 0 ⇒ base level e^0 = 1.000
        assert!((off[1] - 1.0).abs() < 1e-6); // ln(e) = 1
        assert_eq!(off[2], 0.0);
    }

    #[test]
    fn non_positive_or_nonfinite_exposure_errors() {
        assert!(matches!(
            compute_offset(&[1.0, 0.0], 2),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            compute_offset(&[1.0, -2.0], 2),
            Err(PbError::InvalidInput { .. })
        ));
        assert!(matches!(
            compute_offset(&[1.0, f32::NAN], 2),
            Err(PbError::InvalidInput { .. })
        ));
    }

    #[test]
    fn length_mismatch_errors() {
        assert!(matches!(
            compute_offset(&[1.0, 2.0], 3),
            Err(PbError::ShapeMismatch { .. })
        ));
    }
}
