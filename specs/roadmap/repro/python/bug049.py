import numpy as np
from t_boost import TBoostRegressor

X = np.arange(100, dtype=np.float32).reshape(-1, 1)
y = np.arange(100, dtype=np.float32)
w = np.zeros(100, dtype=np.float32)
w[17] = 1
for bags in [1, 2, 8]:
    for seed in [0, 1, 2]:
        try:
            model = TBoostRegressor(
                n_trees=3, n_bags=bags, prune=False, graduate=False,
                validation_fraction=None, n_jobs=1, seed=seed,
            ).fit(X, y, sample_weight=w)
            print(bags, seed, "ok", model.predict(X)[0])
        except ValueError as error:
            print(bags, seed, type(error).__name__, str(error))
