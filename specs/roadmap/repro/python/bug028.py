import warnings
import numpy as np
from t_boost import TBoostRegressor
warnings.simplefilter('ignore')
x = np.arange(80, dtype=np.float32).reshape(-1, 1)
for cats in [[], (), np.array([], dtype=int), None]:
    try:
        TBoostRegressor(categorical_features=cats, n_trees=5, n_bags=1, prune=False, validation_fraction=None, n_jobs=1).fit(x, x[:, 0])
        print(repr(cats), "-> fit OK")
    except Exception as exc:
        print(repr(cats), "->", type(exc).__name__, str(exc)[:120])
