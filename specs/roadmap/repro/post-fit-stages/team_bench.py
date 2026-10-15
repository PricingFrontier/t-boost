"""Shared benchmark harness for ../../post-fit-stages.md (the 2026-10-14 agent team).

Not part of the test suite. The dataset paths below are local to the maintainer's machine.

Run from a t-boost checkout so its own build is imported:
    uv run --no-sync python specs/roadmap/repro/post-fit-stages/team_bench.py \
        --datasets norway,surrender --splits 0,1 \
        --configs '{"base": {}, "nograd": {"graduate": false}}'

Each (dataset, split, config) prints one JSON line: test/train mean deviance (lower is better;
log loss for classification), fit seconds, and the post-fit reports the issues are about.
Import it to reuse `load`, `fit`, `deviance` in custom experiments.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

CACHE = Path("/home/ralph/suite/hsv18-throwaway-cache/datasets")
FREMTPL2 = Path("/home/ralph/suite/t-boost-report-demo/data/fremtpl2freq.parquet")

# name -> (cache id or None, task, use exposure)
DATASETS: dict[str, tuple[str | None, str, bool]] = {
    # Poisson claim frequency with exposure
    "fremtpl2": (None, "poisson", True),                        # 678k rows, the slow one
    "norway": ("af8ad88eb912a0769cc2117f", "poisson", True),     # 184k, 4 features
    "swmc": ("7359e233c972ae2b11a2a29d", "poisson", True),       # Swedish motorcycle, 62k
    # Gamma severity
    "ausbi": ("9d9a76495a9190d6e90caabd", "gamma", False),       # 22k
    "ussev": ("53e4a20d42b621dbe9f6e1e8", "gamma", False),       # 6.8k
    # binary
    "surrender": ("b8b10eb4c1ae6bc1cb3b4f16", "binary", False),  # 22k, 36% positive
    "lifesurr": ("a693b79c5d5c488d6cb44a06", "binary", False),   # 29k, 38% positive
    "lapse": ("bc6038a280a98b8176e891ab", "binary", False),      # 23k
    # multiclass (K>=3)
    "pg16_fleet": ("37ea3865126ec30642cdee26", "multiclass", False),   # 87k, 3 classes
    "pg17_instal": ("bfce0f0f42ebbe2cfad7ef0a", "multiclass", False),  # 100k, 4 classes
    "pg17_cover": ("247ea0b94fc140352d97367e", "multiclass", False),   # 100k, 4 classes
    "fr_instal": ("a1df2d5700a99a20fafdbd43", "multiclass", False),    # 52k, 4 classes
    "prudential": ("95d35ecb7cea120f24110598", "multiclass", False),   # 59k, 8 classes, 139 cols
}
FRE_FEATURES = ["Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas",
                "Density", "Region"]


def load(name: str, split: int = 0) -> dict[str, Any]:
    """Train/test parts of a dataset: X (polars), y, exposure (or None), weight (or None).
    The test set is the rows with `default_rng(split).uniform() >= 0.8`."""
    cache_id, task, use_exposure = DATASETS[name]
    if cache_id is None:
        df = (pl.read_parquet(FREMTPL2)
              .with_columns(pl.col("ClaimNb").clip(upper_bound=4),
                            pl.col("Exposure").clip(upper_bound=1.0),
                            pl.col("VehGas").str.strip_chars("'"),
                            *[pl.col(c).cast(pl.String) for c in ("Area", "VehBrand", "Region")]))
        X = df.select(FRE_FEATURES)
        y = df["ClaimNb"].to_numpy().astype(float)
        e = df["Exposure"].to_numpy().astype(float)
        w = None
    else:
        d = CACHE / cache_id
        meta = json.loads((d / "meta.json").read_text())
        arrays = np.load(d / "arrays.npz", allow_pickle=True)
        X = pl.read_parquet(d / "X.parquet")
        cats = [c for c in meta.get("cat_features") or [] if c in X.columns]
        X = X.with_columns([pl.col(c).cast(pl.String) for c in cats])
        y = arrays["y"].astype(float)
        e = arrays["exposure"].astype(float) if (use_exposure and arrays["exposure"].size) else None
        w = arrays["sample_weight"].astype(float) if arrays["sample_weight"].size else None
        if w is not None and (np.allclose(w, 1.0) or (e is not None and np.allclose(w, e))):
            w = None  # the exposure already carries it (or it is all ones)
    if task == "multiclass":
        y = y.astype(int)
    test = np.random.default_rng(split).uniform(size=len(y)) >= 0.8
    part = lambda a, m: None if a is None else a[m]  # noqa: E731
    return {"task": task,
            "train": (X.filter(pl.Series(~test)), y[~test], part(e, ~test), part(w, ~test)),
            "test": (X.filter(pl.Series(test)), y[test], part(e, test), part(w, test))}


def deviance(task: str, y: np.ndarray, pred: np.ndarray, e: np.ndarray | None,
             w: np.ndarray | None) -> float:
    """Weighted mean unit deviance; `pred` is predict() (rate) or predict_proba()."""
    w = np.ones(len(y)) if w is None else w
    if task == "poisson":
        mu = pred * (1.0 if e is None else e)
        t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
        unit = 2.0 * (t - (y - mu))
    elif task == "gamma":
        unit = 2.0 * (-np.log(y / pred) + (y - pred) / pred)
    elif task == "binary":
        p = np.clip(pred[:, 1] if pred.ndim == 2 else pred, 1e-15, 1 - 1e-15)
        unit = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    else:
        unit = -np.log(np.clip(pred[np.arange(len(y)), y], 1e-15, None))
    return float(np.sum(w * unit) / np.sum(w))


def fit(task: str, kwargs: dict[str, Any], part: tuple, n_jobs: int | None = 8):
    from t_boost import TBoostClassifier, TBoostRegressor
    X, y, e, w = part
    kw = {"seed": 0, "n_jobs": n_jobs, **kwargs}
    if task in ("poisson", "gamma"):
        est = TBoostRegressor(objective=task, **kw)
        return est.fit(X, y, exposure=e, sample_weight=w)
    est = TBoostClassifier(**kw)
    return est.fit(X, y.astype(int), sample_weight=w)


def predict(task: str, est, X):
    return est.predict(X) if task in ("poisson", "gamma") else est.predict_proba(X)


def reports(est) -> dict[str, Any]:
    rep = getattr(est, "pruning_report_", None) or {}
    band = rep.get("banding")
    if isinstance(band, list):  # multiclass: one per class
        band = {"classes": len(band), "budget": [b.get("budget") for b in band]}
    elif isinstance(band, dict):
        band = {k: band.get(k) for k in ("skipped", "budget", "noise_budget", "deviance_budget",
                                         "selected_mse", "combined_mse")}
    gv = getattr(est, "graduation_validation_", None) or {}
    return {"selector": rep.get("selector"), "kept": len(rep.get("kept") or []),
            "rebalance": rep.get("rebalance"), "banding": band,
            "graduation": {k: gv.get(k) for k in ("adopted", "skipped", "rejected",
                                                  "n_smoothed", "candidate_count")}}


def run(name: str, split: int, config: str, kwargs: dict[str, Any], n_jobs: int | None) -> dict:
    data = load(name, split)
    task = data["task"]
    t0 = time.time()
    est = fit(task, kwargs, data["train"], n_jobs)
    secs = time.time() - t0
    out = {"dataset": name, "split": split, "config": config, "fit_s": round(secs, 1)}
    for side in ("test", "train"):
        X, y, e, w = data[side]
        out[f"{side}_dev"] = deviance(task, y, predict(task, est, X), e, w)
    out.update(reports(est))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--splits", default="0")
    ap.add_argument("--configs", default='{"base": {}}', help="JSON: name -> estimator kwargs")
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    configs = json.loads(a.configs)
    for name in a.datasets.split(","):
        for split in (int(s) for s in a.splits.split(",")):
            for config, kwargs in configs.items():
                print(json.dumps(run(name, split, config, kwargs, a.n_jobs), default=str),
                      flush=True)


if __name__ == "__main__":
    main()
