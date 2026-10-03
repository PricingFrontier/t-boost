import json
import numpy as np
import polars as pl
from t_boost import TBoostRegressor
x = pl.DataFrame({'c': ['<rare>']*100 + ['a']*100 + ['thin']*2})
y = np.array([0]*100 + [2]*100 + [8]*2, dtype=np.float32)
m = TBoostRegressor(n_trees=30, n_bags=1, prune=False,
    validation_fraction=None, n_jobs=1).fit(x, y)
print(json.loads(m.tables(x))['tables'][0]['axes'][0]['levels'])
print(m.predict(pl.DataFrame({'c': ['<rare>', 'thin']})))
