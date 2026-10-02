# Control for BUG-030: band an UNBANDED fit (band_tolerance=None) once, then band the banded result again.
import json
import numpy as np
from t_boost import TBoostRegressor

rng = np.random.default_rng(80)
x = rng.uniform(-2, 2, size=(600, 2)).astype(np.float32)
y = (np.sin(3*x[:, 0])*np.cos(2*x[:, 1]) + 0.1*rng.normal(size=600)).astype(np.float32)
m = TBoostRegressor(
    n_trees=40, n_bags=2, max_depth=4, interaction_gain_hurdle=0.0,
    validation_fraction=None, n_jobs=1, band_tolerance=None, graduate=False,
).fit(x, y)
tm = m._model
p = np.asarray(tm.predict_raw(x), dtype=np.float64)
for tol in [1e-8, 0.01]:
    b, report = tm.band(x, np.ones(600), np.ones(600), p, 1.0, tolerance=tol, pseudo_rows=1000, n_jobs=1)
    p2 = np.asarray(b.predict_raw(x), dtype=np.float64)
    r = json.loads(report)
    print("FIRST pass tol", tol, "reported", r['combined_mse'], "budget", r['budget'], "actual MSE", np.mean((p2-p)**2), "max move", np.max(np.abs(p2-p)))
    # second pass on the banded output
    b2, report2 = b.band(x, np.ones(600), np.ones(600), p2, 1.0, tolerance=tol, pseudo_rows=1000, n_jobs=1)
    p3 = np.asarray(b2.predict_raw(x), dtype=np.float64)
    r2 = json.loads(report2)
    print("SECOND pass tol", tol, "reported", r2['combined_mse'], "budget", r2['budget'], "actual MSE", np.mean((p3-p2)**2), "max move", np.max(np.abs(p3-p2)))
