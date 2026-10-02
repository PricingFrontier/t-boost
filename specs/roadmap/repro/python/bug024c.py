import numpy as np
import pandas as pd
from t_boost import TBoostRegressor
x = pd.DataFrame({'cat': pd.Categorical(
    np.tile(np.array([.1, .2], dtype=np.float32), 100)
)})
y = np.tile([0., 10.], 100).astype(np.float32)
m = TBoostRegressor(categorical_features=[0], n_trees=10, n_bags=1,
    prune=False, validation_fraction=None, n_jobs=1).fit(x, y)
for n in [1, 63, 64, 100]:
    batch = pd.concat([x.iloc[[1]]] * n, ignore_index=True)
    print(n, m.predict(batch)[0])
print("categories dtype:", x['cat'].cat.categories.dtype, [type(c) for c in x['cat'].cat.categories][:1], [str(c) for c in x['cat'].cat.categories])
print("to_numpy scalar:", type(x['cat'].to_numpy()[1]), str(x['cat'].to_numpy()[1]))
