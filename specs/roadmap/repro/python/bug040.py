import numpy as np
from t_boost import TBoostRegressor

x = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32), (20, 1))
y = np.tile(np.array([6, 2, 2, 0], dtype=np.float32), 20)
for n_trees in [20, 100]:
    for shift in [0.0, 10_000_000.0]:
        model = TBoostRegressor(
            n_trees=n_trees, n_bags=1, prune=False, graduate=False,
            band_tolerance=None, validation_fraction=None, n_jobs=1,
            learning_rate=0.2, leaf_refine_steps=0, reanchor=False,
        ).fit(x, y + np.float32(shift))
        centered = model.predict(x) - shift
        print(n_trees, shift, centered[:4], np.mean((centered - y) ** 2))
        # independent accumulation check
        target = (y + np.float32(shift)).astype(np.float32)
        native = model._new_booster(fit_pool_width=1).fit(x, target)
        tables = model._full_tables(native, x, target, None, None, None)
        nraw = np.asarray(native.predict_raw(x))
        traw = np.asarray(tables.predict_raw(x))
        print("   native predict_raw-shift[:4]", (nraw - shift)[:4], " table predict_raw-shift[:4]", (traw - shift)[:4],
              " max|native-table|", np.max(np.abs(nraw.astype(np.float64) - traw.astype(np.float64))))
        print("   public predict == table predict_raw:", np.allclose(model.predict(x), traw), " native dtype", nraw.dtype, "table dtype", traw.dtype)
