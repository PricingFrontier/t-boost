use t_boost_core::explain::{fixture_model, fixture_serve, RefMeasure};
use t_boost_core::table_model::TableModel;

fn doc_with_factored(k: usize) -> String {
    let tm = TableModel::from_model(&fixture_model(), &fixture_serve(), RefMeasure::Uniform).unwrap();
    let mut doc: serde_json::Value = serde_json::from_str(&tm.to_json().unwrap()).unwrap();
    let ids: Vec<u32> = (0..k as u32).collect();
    let axes: Vec<serde_json::Value> = ids
        .iter()
        .map(|&i| {
            serde_json::json!({"raw": i, "borders": [], "cells": 2, "joint_channels": null, "band_of": null})
        })
        .collect();
    let per_axis_w: Vec<serde_json::Value> = ids.iter().map(|_| serde_json::json!([0.5, 0.5])).collect();
    let fe = serde_json::json!({
        "u": ids,
        "axes": axes,
        "per_axis_w": per_axis_w,
        "boxes": [],
        "variance": 0.0
    });
    doc["model"]["bank"]["factored"] = serde_json::json!([fe]);
    doc.to_string()
}

fn try_load(k: usize) {
    let s = doc_with_factored(k);
    let r = std::panic::catch_unwind(|| TableModel::from_json(&s).map(|m| m.bank.factored.len()));
    match r {
        Ok(Ok(n)) => println!("k={k}: loaded Ok, {n} factored effect(s)"),
        Ok(Err(e)) => println!("k={k}: typed error: {e}"),
        Err(p) => {
            let msg = p
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".into());
            println!("k={k}: PANIC: {msg}");
        }
    }
}

fn main() {
    std::panic::set_hook(Box::new(|info| {
        eprintln!("[panic hook] {info}");
    }));
    for k in [3usize, 9, 63, 64, 65, 200] {
        try_load(k);
    }
}
