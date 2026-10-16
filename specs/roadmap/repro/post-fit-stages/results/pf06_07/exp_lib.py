"""PF-06/PF-07 experiment library: one fit per (dataset, split, n_bags), then post-fit variants
on the same soup without refitting.

The worktree build carries experiment-only hooks (not for landing):
  * env TBOOST_EXP_MCBAND_DUMP=<dir>: the multiclass banding stage writes the pre-banding model
    (preband.bin), its per-class inputs (meta.json: sigmas, dev_mean, hs) and every bag's
    kept-bank class scores (bagscores_<k>.f64, [bag][row] little-endian f64);
  * `_MultiClassTableModel.exp_band(x, y, w, sigmas, tolerance, cap, mode, cat_x, n_jobs)`:
    re-band a model with a budget mode (see `mc_band_apply` in the binding).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, Timer, census, tb  # noqa: E402

from t_boost._t_boost import _MultiClassTableModel  # noqa: E402

DUMPS = RES / "dumps"
N_JOBS = 8


class Stage:
    """One fitted soup's post-fit state: the estimator (graduate=False), the pre-banding model U,
    the shipped banded model B, designs and labels for train and test."""

    def __init__(self, name: str, split: int, n_bags: int, refit: bool = False,
                 tag: str = "", fit_kwargs: dict[str, Any] | None = None) -> None:
        self.name, self.split, self.n_bags = name, split, n_bags
        self.dir = DUMPS / f"{name}_s{split}_b{n_bags}{('_' + tag) if tag else ''}"
        self.fit_kwargs = dict(fit_kwargs or {})
        d = tb.load(name, split)
        self.Xtr, self.ytr, _, self.wtr = d["train"]
        self.Xte, self.yte, _, self.wte = d["test"]
        t = Timer()
        est_path = self.dir / "est.bin"
        if est_path.exists() and not refit:
            from t_boost import TBoostClassifier
            self.est = TBoostClassifier.from_bytes(est_path.read_bytes())
            self.est.n_jobs = N_JOBS  # set_params would clear the fitted state
            self.fit_s = json.loads((self.dir / "fit.json").read_text())["fit_s"]
        else:
            self.dir.mkdir(parents=True, exist_ok=True)
            os.environ["TBOOST_EXP_MCBAND_DUMP"] = str(self.dir)
            try:
                self.est = tb.fit("multiclass",
                                  {"graduate": False, "n_bags": n_bags, **self.fit_kwargs},
                                  d["train"], N_JOBS)
            finally:
                os.environ.pop("TBOOST_EXP_MCBAND_DUMP", None)
            self.fit_s = t.lap()
            est_path.write_bytes(self.est.to_bytes())
            (self.dir / "fit.json").write_text(json.dumps({"fit_s": self.fit_s}))
            (self.dir / "pruning_report.json").write_text(
                json.dumps(self.est.pruning_report_, default=float))
        est = self.est
        self.pruning_report = json.loads((self.dir / "pruning_report.json").read_text())
        self.B = est._multi_model
        self.U = _MultiClassTableModel.from_bytes((self.dir / "preband.bin").read_bytes())
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.K = int(self.U.n_classes)
        names = list(self.Xtr.columns)
        self.x_tr, self.cat_tr, _ = est._split_columns(self.Xtr, list(est._cat_indices_), names)
        self.y_tr = np.searchsorted(est.classes_, self.ytr).astype(np.float32)
        self.y_te = np.searchsorted(est.classes_, self.yte).astype(np.int64)
        self.w_tr = np.ascontiguousarray(
            self.wtr.astype(np.float32) if self.wtr is not None
            else np.ones(len(self.ytr), dtype=np.float32))
        self.w_te = (self.wte.astype(np.float64) if self.wte is not None
                     else np.ones(len(self.yte)))
        self.x_te, self.kw_te = est._serve_kwargs(self.Xte, self.U)
        self.yi_tr = self.y_tr.astype(np.int64)
        # the shipped banded model's per-class reports (one per class)
        self.band_reports = self.pruning_report.get("banding")
        self._cache: dict[int, tuple[Any, np.ndarray, np.ndarray]] = {}

    # -- scoring -------------------------------------------------------------------------------
    def logits(self, model: Any) -> tuple[np.ndarray, np.ndarray]:
        key = id(model)
        if key not in self._cache:
            a = np.asarray(model.predict_raw(self.x_tr, cat_x=self.cat_tr, n_jobs=N_JOBS),
                           dtype=np.float64)
            b = np.asarray(model.predict_raw(self.x_te, **self.kw_te, n_jobs=N_JOBS),
                           dtype=np.float64)
            # the model is held with its logits, so its id cannot be reused while cached
            self._cache[key] = (model, a, b)
        _, a, b = self._cache[key]
        return a, b

    def forget(self, model: Any) -> None:
        self._cache.pop(id(model), None)

    def ce(self, model: Any) -> tuple[float, float]:
        a, b = self.logits(model)
        return (ce_logits(a, self.yi_tr, self.w_tr.astype(np.float64)),
                ce_logits(b, self.y_te, self.w_te))

    def sigmas(self) -> list[float | None]:
        return self.meta["sigmas"]

    def band(self, mode: str, cap: float = 0.001, tol: float = 0.75) -> tuple[Any, list[dict]]:
        m, rep = self.U.exp_band(self.x_tr, self.y_tr, self.w_tr, self.sigmas(), tol, cap, mode,
                                 cat_x=self.cat_tr, n_jobs=N_JOBS)
        return m, json.loads(rep)

    def bag_scores(self, k: int) -> np.ndarray | None:
        p = self.dir / f"bagscores_{k}.f64"
        if not p.exists():
            return None
        v = np.fromfile(p, dtype="<f8")
        n = int(self.meta["n"])
        return v.reshape(-1, n)


def ce_logits(z: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    m = z.max(axis=1, keepdims=True)
    lse = (m[:, 0] + np.log(np.exp(z - m).sum(axis=1)))
    nll = lse - z[np.arange(len(y)), y]
    return float(np.sum(w * nll) / np.sum(w))


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def decompose(a: np.ndarray, b: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    """Second-order decomposition of the move a -> b (logits) in NLL units, as % of the CE at a:
    realized change, first-order Σ(p-y)δ, joint softmax quadratic ½[Σpδ² - (Σpδ)²], its diagonal
    stand-in ½Σp(1-p)δ², and per-class first-order and diagonal terms."""
    w = np.asarray(w, dtype=np.float64)
    sw = w.sum()
    base = ce_logits(a, y, w)
    p = softmax(a)
    d = b - a
    Y = np.zeros_like(p)
    Y[np.arange(len(y)), y] = 1.0
    g = p - Y
    first_k = (w[:, None] * g * d).sum(axis=0) / sw
    diag_k = 0.5 * (w[:, None] * p * (1 - p) * d * d).sum(axis=0) / sw
    pd = (p * d).sum(axis=1)
    joint = 0.5 * float(np.sum(w * ((p * d * d).sum(axis=1) - pd * pd)) / sw)
    common = 0.5 * float(np.sum(w * pd * pd) / sw)  # the part of Σpδ² that is a common shift
    realized = ce_logits(b, y, w) - base
    pc = lambda v: 100.0 * float(v) / base  # noqa: E731
    return {"base": base, "realized": pc(realized), "first": pc(first_k.sum()),
            "joint2": pc(joint), "diag2": pc(diag_k.sum()), "common2": pc(common),
            "rem": pc(realized - first_k.sum() - joint),
            "first_k": [pc(v) for v in first_k], "diag2_k": [pc(v) for v in diag_k]}


def band_summary(reports: list[dict] | None) -> list[dict[str, Any]] | None:
    if not reports:
        return None
    keys = ("budget", "noise_budget", "deviance_budget", "selected_mse", "combined_mse",
            "tables", "largest_after", "skipped")
    out = []
    for r in reports:
        e = {k: r.get(k) for k in keys}
        e["cells_before"] = sum(int(v) for v in (r.get("cells_before") or {}).values())
        e["cells_after"] = sum(int(v) for v in (r.get("cells_after") or {}).values())
        out.append(e)
    return out
