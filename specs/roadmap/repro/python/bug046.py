import numpy as np
from t_boost import TBoostRegressor
x = np.arange(20, dtype=np.float32).reshape(-1, 1)
for val in (-3, 3):
    y = np.full(20, val, dtype=np.float32)
    m = TBoostRegressor(n_trees=5, n_bags=1, prune=False, validation_fraction=None, n_jobs=1).fit(x, y)
    print("y =", val)
    print(m.actual_vs_expected(x, y)[0])
    print("pricing ae:", m.pricing_report(x, y)['actual_vs_expected'][0]['ae'])
