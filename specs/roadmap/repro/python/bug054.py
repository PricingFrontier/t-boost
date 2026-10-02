import numpy as np
from t_boost import TBoostRegressor

x = np.repeat(np.array([0, 1], dtype=np.float32), 100).reshape(-1, 1)
y = np.repeat(np.array([1e12, 7.9e13], dtype=np.float32), 100)
for n_trees in [2, 10, 20, 40]:
    predictions = []
    for incremental in [False, True]:
        model = TBoostRegressor(
            objective="poisson", n_trees=n_trees, n_bags=1,
            prune=False, graduate=False, band_tolerance=None,
            validation_fraction=None, n_jobs=1, learning_rate=0.1,
            leaf_refine_steps=0, reanchor=False, incremental_mu=incremental,
        ).fit(x, y)
        pred = model.predict(x)
        predictions.append(pred)
        print(n_trees, incremental, pred[[0, -1]], model.predict_raw(x)[[0, -1]])
    print("max relative difference",
          np.max(np.abs(predictions[0] - predictions[1]) / predictions[0]))
print("log(mean y) =", np.log(y.astype(np.float64).mean()), " exp(30)=", np.exp(30))
