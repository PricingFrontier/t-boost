import warnings
import numpy as np
import pandas as pd
import polars as pl
from t_boost import TBoostRegressor
warnings.simplefilter('ignore')
opts = dict(n_trees=20, n_bags=1, prune=False, validation_fraction=None,
            n_jobs=1, min_data_in_leaf=1, max_depth=3)
x = np.tile(np.array([0.1, 0.2], dtype=np.float32), 100).reshape(-1, 1)
y = np.tile(np.array([0., 10.], dtype=np.float32), 100)
frames = [('numpy', x), ('pandas', pd.DataFrame(x, columns=['f'])),
          ('polars', pl.DataFrame({'f': x[:, 0]}))]
for train_name, frame in frames:
    m = TBoostRegressor(categorical_features=[0], **opts).fit(frame, y)
    for serve_name, other in frames:
        print(train_name, serve_name, m.predict(other)[:2])
print("str(np.float32(0.1)) =", str(np.float32(0.1)), "| str(float(np.float32(0.1))) =", str(float(np.float32(0.1))))
