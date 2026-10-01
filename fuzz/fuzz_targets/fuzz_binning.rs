#![no_main]

use libfuzzer_sys::fuzz_target;
use t_boost_core::{bin, build_grid, BinConfig, BorderFamily, FeatureId};

fuzz_target!(|data: &[u8]| {
    let (cfg, seed, values, weights) = decode_case(data);
    if let Ok(grid) = build_grid(&values, weights.as_deref(), &cfg, seed, FeatureId(0)) {
        for &v in &values {
            let _ = bin(v, &grid);
        }
    }
});

fn decode_case(data: &[u8]) -> (BinConfig, u64, Vec<f32>, Option<Vec<f32>>) {
    let max_bin = data.first().copied().unwrap_or(8).clamp(2, 254);
    let subsample_for_binning = u32::from(data.get(1).copied().unwrap_or(16)).max(1);
    let min_data_per_bin = u32::from(data.get(2).copied().unwrap_or(0) % 16);
    let cfg = BinConfig {
        max_bin,
        subsample_for_binning,
        min_data_per_bin,
        border_family: BorderFamily::EqualCount,
    };

    let mut seed_bytes = [0_u8; 8];
    for (dst, src) in seed_bytes.iter_mut().zip(data.iter().skip(3)) {
        *dst = *src;
    }
    let seed = u64::from_le_bytes(seed_bytes);

    let payload = data.get(11..).unwrap_or(&[]);
    let mut values = Vec::with_capacity(payload.len() / 4);
    for chunk in payload.chunks(4) {
        let mut bytes = [0_u8; 4];
        for (dst, src) in bytes.iter_mut().zip(chunk) {
            *dst = *src;
        }
        values.push(f32::from_bits(u32::from_le_bytes(bytes)));
    }
    if values.is_empty() {
        values.push(f32::NAN);
    }

    let weights = if data.get(2).is_some_and(|b| b & 0x80 != 0) {
        Some(
            values
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let raw = data.get(11 + i).copied().unwrap_or(1);
                    if raw == 0 {
                        0.0
                    } else {
                        f32::from(raw) / 16.0
                    }
                })
                .collect(),
        )
    } else {
        None
    };

    (cfg, seed, values, weights)
}
