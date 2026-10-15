"""Settings-matrix validation of the post-fit stages (2026-10-15, PF-15's coverage gap).

Fits every objective under every setting the post-fit work touches, on data that carries the
three kinds of categorical the stages treat differently: one whose levels share bins (300 levels,
scored at its out-of-fold bins by PF-05), one with a bin per level (40 levels) and a binary one.
Each fit must either run or be refused with a deliberate message, and when it runs:

- predictions are finite (probabilities sum to 1);
- the reports are well formed (selector, kept tables, banding, rebalance, graduation, the
  boosting report);
- the native model bytes are identical at ``n_jobs`` 1 and 4;
- ``to_bytes``/``from_bytes`` and ``to_json``/``from_json`` reproduce the predictions.

    uv run --no-sync python specs/roadmap/repro/post-fit-stages/validation_matrix.py \
        --out specs/roadmap/repro/post-fit-stages/results/validation_matrix.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "python" / "tests"))
from _artifact import model_bytes  # noqa: E402

from t_boost import TBoostClassifier, TBoostRegressor  # noqa: E402

warnings.filterwarnings("ignore")
STOP_REASONS = {"early_stopping", "max_trees", "no_split", "callback"}


def make_data(task: str, n: int = 5000, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n), rng.uniform(-1, 1, size=n)
    shared_levels = [f"S{i}" for i in range(300)]   # more levels than the 254 bins
    own_levels = [f"O{i}" for i in range(40)]       # one bin per level
    s_eff = dict(zip(shared_levels, rng.normal(scale=0.4, size=300)))
    o_eff = dict(zip(own_levels, rng.normal(scale=0.3, size=40)))
    s = rng.choice(shared_levels, n)
    o = rng.choice(own_levels, n)
    g = rng.choice(["m", "f"], n)
    eta = (0.4 * a + 0.3 * np.sin(2 * b) + np.array([s_eff[v] for v in s])
           + np.array([o_eff[v] for v in o]) + 0.2 * (g == "m") + 0.2 * a * b)
    X = pl.DataFrame({"a": a, "b": b, "s": s, "o": o, "g": g})
    exposure = rng.uniform(0.3, 1.0, size=n)
    weight = rng.uniform(0.5, 2.0, size=n)
    groups = rng.integers(0, n // 5, size=n)  # ~5 rows per entity
    offset = rng.normal(scale=0.1, size=n)
    if task == "poisson":
        y = rng.poisson(np.exp(eta - 1.0) * exposure).astype(float)
    elif task == "tweedie":
        counts = rng.poisson(np.exp(eta - 1.0) * exposure)
        y = np.array([rng.gamma(2.0, 0.5, size=c).sum() for c in counts])
    elif task == "gamma":
        y = rng.gamma(3.0, np.exp(eta) / 3.0)
    elif task == "squared_error":
        y = eta + rng.normal(scale=0.5, size=n)
    elif task == "binary":
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(int)
    else:  # multiclass, 3 classes
        logits = np.stack([np.zeros(n), eta, -0.5 * eta + 0.3 * b], axis=1)
        p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        y = (rng.uniform(size=(n, 1)) > np.cumsum(p, axis=1)).sum(axis=1)
    cut = int(n * 0.8)
    return {"X": X, "y": y, "exposure": exposure, "weight": weight, "groups": groups,
            "offset": offset, "cut": cut}


SETTINGS: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    # name: (constructor kwargs, fit kwargs keys to pass from the data)
    "default": ({}, {}),
    "weights": ({}, {"sample_weight": True}),
    "exposure+weights": ({}, {"exposure": True, "sample_weight": True}),
    "offset": ({}, {"offset": True}),
    "groups": ({}, {"groups": True}),
    "eval_set": ({}, {"eval_set": True}),
    "cell_refit": ({"cell_refit_base": 4000.0}, {}),
    "one_bag": ({"n_bags": 1}, {}),
    "unpruned": ({"prune": False}, {}),
    "monotone": ({"monotone_constraints": {"a": 1}}, {}),
    "multichannel": ({"cat_channels": ["mean", "count"]}, {}),
    "class_freq": ({"cat_channels": ["mean", "class_freq"]}, {}),
    "callback": ({}, {"callbacks": True}),
    "fixed_graduation": ({"graduation_alpha": 30.0}, {}),
    "no_reanchor": ({"reanchor": False}, {}),
}
TASKS = ["poisson", "gamma", "tweedie", "squared_error", "binary", "multiclass"]


def build(task: str, ctor: dict[str, Any], n_jobs: int):
    common = {"n_trees": 150, "seed": 0, "n_jobs": n_jobs, **ctor}
    if task in ("binary", "multiclass"):
        return TBoostClassifier(**common)
    return TBoostRegressor(objective=task, **common)


def fit_kwargs(task: str, keys: dict[str, Any], d: dict[str, Any]) -> dict[str, Any]:
    cut = d["cut"]
    out: dict[str, Any] = {}
    if task in ("poisson", "tweedie") and "exposure" not in keys:
        out["exposure"] = d["exposure"][:cut]
    for k in keys:
        if k == "eval_set":
            out["eval_set"] = (d["X"][cut:], d["y"][cut:])
            if task in ("poisson", "tweedie"):
                out["eval_exposure"] = d["exposure"][cut:]
        elif k == "callbacks":
            out["callbacks"] = lambda info: info["round"] >= 20
        else:
            out[k] = d[{"sample_weight": "weight"}.get(k, k)][:cut]
    return out


def predict(est, task: str, X, d: dict[str, Any], keys: dict[str, Any]) -> np.ndarray:
    if task in ("binary", "multiclass"):
        return est.predict_proba(X)
    kw = {"offset": d["offset"][d["cut"]:]} if "offset" in keys else {}
    return est.predict(X, **kw)


def report_checks(est, task: str, ctor: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    rep = est.pruning_report_ or {}
    if not rep.get("selector"):
        problems.append("pruning_report_ has no selector")
    if not rep.get("kept"):
        problems.append("pruning_report_ keeps no tables")
    if "banding" not in rep:
        problems.append("no banding report")
    if ctor.get("prune", True) and "rebalance" not in rep:
        problems.append("no rebalance report on a pruned fit")
    gv = getattr(est, "graduation_validation_", None) or {}
    if "adopted" not in gv or ("candidate_count" not in gv and gv.get("skipped") != "disabled"):
        problems.append(f"graduation_validation_ incomplete: {sorted(gv)}")
    per_bag = getattr(est, "n_trees_per_bag_", None)
    reasons = getattr(est, "stopping_reason_per_bag_", None)
    n_bags = ctor.get("n_bags", 8)
    if per_bag is None or reasons is None or len(per_bag) != n_bags or len(reasons) != n_bags:
        problems.append(f"boosting report missing or wrong length: {per_bag} {reasons}")
    elif not set(reasons) <= STOP_REASONS:
        problems.append(f"unknown stop reasons {reasons}")
    return problems


def run(task: str, setting: str) -> dict[str, Any]:
    ctor, keys = SETTINGS[setting]
    d = make_data(task)
    cut = d["cut"]
    X_tr, y_tr, X_te = d["X"][:cut], d["y"][:cut], d["X"][cut:]
    out: dict[str, Any] = {"task": task, "setting": setting}
    t0 = time.time()
    try:
        est = build(task, ctor, 4).fit(X_tr, y_tr, **fit_kwargs(task, keys, d))
    except Exception as exc:  # recorded: a refusal must be deliberate
        out.update(status="refused", error=f"{type(exc).__name__}: {exc}"[:400])
        return out
    out["fit_s"] = round(time.time() - t0, 2)
    problems: list[str] = []
    try:
        pred = predict(est, task, X_te, d, keys)
        if not np.all(np.isfinite(pred)):
            problems.append("non-finite predictions")
        if task in ("binary", "multiclass") and not np.allclose(pred.sum(axis=1), 1.0, atol=1e-5):
            problems.append("probabilities do not sum to 1")
        problems += report_checks(est, task, ctor)
        again = build(task, ctor, 1).fit(X_tr, y_tr, **fit_kwargs(task, keys, d))
        h4 = hashlib.sha256(model_bytes(est)).hexdigest()[:16]
        h1 = hashlib.sha256(model_bytes(again)).hexdigest()[:16]
        out["bytes"] = h4
        if h1 != h4:
            problems.append(f"n_jobs 1 vs 4 differ ({h1} vs {h4})")
        cls = type(est)
        for loaded in (cls.from_bytes(est.to_bytes()), cls.from_json(est.to_json())):
            p2 = predict(loaded, task, X_te, d, keys)
            if not np.allclose(pred, p2, rtol=1e-6, atol=1e-9):
                problems.append("reloaded model predicts differently")
            if getattr(loaded, "n_trees_per_bag_", None) != getattr(est, "n_trees_per_bag_", None):
                problems.append("boosting report lost on reload")
        rb = (est.pruning_report_ or {}).get("rebalance") or {}
        out["rebalance"] = {k: rb.get(k) for k in ("adopted", "supports", "excluded")}
        out["stop"] = getattr(est, "stopping_reason_", None)
    except Exception:
        problems.append("check raised: " + traceback.format_exc(limit=3)[-600:])
    out.update(status="ok" if not problems else "problem", problems=problems)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "a") as fh:
        for task in a.tasks.split(","):
            for setting in a.settings.split(","):
                r = run(task, setting)
                fh.write(json.dumps(r, default=str) + "\n")
                fh.flush()
                brief = r.get("error") or "; ".join(r.get("problems") or []) or ""
                print(f"{task:14s} {setting:17s} {r['status']:8s} {r.get('fit_s', '')} {brief[:160]}",
                      flush=True)


if __name__ == "__main__":
    main()
