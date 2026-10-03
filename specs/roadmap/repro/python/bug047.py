import numpy as np
from t_boost import TBoostRegressor
x = np.arange(20, dtype=np.float32).reshape(-1, 1)
m = TBoostRegressor(n_trees=5, n_bags=1, prune=False, validation_fraction=None, n_jobs=1).fit(x, x[:, 0])
print("predict empty:", m.predict(x[:0]).shape)
print("cell_indices empty shape:", np.asarray(m._model.cell_indices(x[:0])).shape)
print("cell_indices full shape:", np.asarray(m._model.cell_indices(x)).shape)
try:
    print(m.actual_vs_expected(x[:0], x[:0, 0]))
except Exception as e:
    print("ERROR", type(e).__name__, e)
x3 = np.arange(60, dtype=np.float32).reshape(-1, 3)
m3 = TBoostRegressor(n_trees=5, n_bags=1, prune=False, validation_fraction=None, n_jobs=1).fit(x3, x3[:, 0])
print("3-feat cell_indices empty shape:", np.asarray(m3._model.cell_indices(x3[:0])).shape)
