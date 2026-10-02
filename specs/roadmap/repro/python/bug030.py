import json
import numpy as np
from t_boost import TBoostRegressor

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "out"  # consumed by rust/src/bin/bug032.rs
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(80)
x = rng.uniform(-2, 2, size=(600, 2)).astype(np.float32)
y = (np.sin(3*x[:, 0])*np.cos(2*x[:, 1])
     + 0.1*rng.normal(size=600)).astype(np.float32)
m = TBoostRegressor(
    n_trees=40, n_bags=2, max_depth=4, interaction_gain_hurdle=0.0,
    validation_fraction=None, n_jobs=1, band_tolerance=0.75,
    graduate=False,
).fit(x, y)
tm = m._model
open(OUT / "tboost_review_banded.json", "w").write(tm.to_json())
print("saved json")
print("band docstring:", (tm.band.__doc__ or "")[:2000])
p = np.asarray(tm.predict_raw(x), dtype=np.float64)
for tol in [1e-8, 0.01]:
    b, report = tm.band(
        x, np.ones(600), np.ones(600), p, 1.0,
        tolerance=tol, pseudo_rows=1000, n_jobs=1,
    )
    p2 = np.asarray(b.predict_raw(x), dtype=np.float64)
    r = json.loads(report)
    print("tol", tol)
    print("  skipped", r.get('skipped'), "combined_mse", r.get('combined_mse'), "budget", r.get('budget'))
    print("  actual MSE", np.mean((p2-p)**2), "max move", np.max(np.abs(p2-p)))
    print("  report keys", sorted(r.keys()))
# inspect shapes of the model
try:
    tj = json.loads(tm.to_json())
    print("top keys", list(tj.keys())[:20])
    bank = tj.get("bank", tj)
    for t in bank.get("tables", []):
        u = t.get("u")
        axes = t.get("axes")
        vals = t.get("values")
        shape = None
        if isinstance(vals, dict):
            shape = vals.get("shape") or vals.get("dims")
        print("  table u", u, "axes extents", [ (len(a.get("borders", [])) if isinstance(a, dict) else None) for a in (axes or [])],
              "banded", [ (a.get("band_of") is not None) if isinstance(a, dict) else None for a in (axes or [])], "shape", shape)
except Exception as ex:
    print("inspect failed", ex)
