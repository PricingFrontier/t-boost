"""The post-fit stages under the settings they meet in practice (2026-10-15).

A categorical with more levels than bins has its fit rows scored at their out-of-fold bins
(`Model::fit_cat_bins`); that path is recorded by every fit and read by the ranked path's evidence
and the rebalance, so it must hold under an evaluation set, panel groups, a fit-time cell
correction, weights with exposure, offsets, and on binary and multiclass fits. Each case must
run, re-solve the categorical's tables with the rest, and give byte-identical models at any thread
count. The full matrix (every objective x every setting, with the model report) is
`specs/roadmap/repro/post-fit-stages/validation_matrix.py`.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from _artifact import model_bytes
from t_boost import TBoostClassifier, TBoostRegressor


def _data(task: str) -> dict:
    rng = np.random.default_rng(0)
    n = 5000
    a = rng.normal(size=n)
    levels = [f"S{i}" for i in range(300)]  # more levels than the 254 bins: levels share bins
    effect = dict(zip(levels, rng.normal(scale=0.4, size=300)))
    s = rng.choice(levels, n)
    eta = 0.4 * a + np.array([effect[v] for v in s])
    exposure = rng.uniform(0.3, 1.0, size=n)
    if task == "poisson":
        y = rng.poisson(np.exp(eta - 1.0) * exposure).astype(float)
    elif task == "gamma":
        y = rng.gamma(3.0, np.exp(eta) / 3.0)
    elif task == "squared_error":
        y = eta + rng.normal(scale=0.5, size=n)
    elif task == "binary":
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(int)
    else:
        logits = np.stack([np.zeros(n), 2.5 * eta, -1.5 * eta], axis=1)
        p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        y = (rng.uniform(size=(n, 1)) > np.cumsum(p, axis=1)).sum(axis=1)
    return {"X": pl.DataFrame({"a": a, "s": s}), "y": y, "exposure": exposure,
            "weight": rng.uniform(0.5, 2.0, size=n), "groups": rng.integers(0, n // 5, size=n),
            "offset": rng.normal(scale=0.1, size=n)}


CASES = [
    ("poisson", "eval_set"),
    ("poisson", "groups"),
    ("poisson", "exposure+weights"),
    ("gamma", "cell_refit"),
    ("squared_error", "offset"),
    ("binary", "groups"),
    ("multiclass", "groups"),
    ("multiclass", "cell_refit"),
]


def _fit(task: str, setting: str, d: dict, n_jobs: int):
    cut = 4000
    ctor: dict = {"n_trees": 120, "seed": 0, "n_jobs": n_jobs}
    kw: dict = {}
    if setting == "cell_refit":
        ctor["cell_refit_base"] = 4000.0
    if task == "poisson":
        kw["exposure"] = d["exposure"][:cut]
    if setting == "eval_set":
        kw.update(eval_set=(d["X"][cut:], d["y"][cut:]), eval_exposure=d["exposure"][cut:])
    elif setting == "groups":
        kw["groups"] = d["groups"][:cut]
    elif setting == "exposure+weights":
        kw["sample_weight"] = d["weight"][:cut]
    elif setting == "offset":
        kw["offset"] = d["offset"][:cut]
    est = (TBoostClassifier(**ctor) if task in ("binary", "multiclass")
           else TBoostRegressor(objective=task, **ctor))
    return est.fit(d["X"][:cut], d["y"][:cut], **kw)


@pytest.mark.parametrize(("task", "setting"), CASES, ids=[f"{t}-{s}" for t, s in CASES])
def test_shared_bin_categorical_under_setting(task: str, setting: str) -> None:
    d = _data(task)
    est = _fit(task, setting, d, n_jobs=4)
    rep = est.pruning_report_
    # The categorical (raw 1) is single-channel, so its tables are re-solved with the rest.
    assert rep["rebalance"]["excluded"] == []
    assert any(1 in u for u in rep["kept"])
    out = (est.predict_proba(d["X"][4000:]) if task in ("binary", "multiclass")
           else est.predict(d["X"][4000:]))
    assert np.isfinite(out).all()
    assert len(est.n_trees_per_bag_) == 8
    assert model_bytes(est) == model_bytes(_fit(task, setting, d, n_jobs=1))
