import json
import numpy as np
import polars as pl
from t_boost import TBoostRegressor
opts = dict(n_trees=30, n_bags=1, prune=False, validation_fraction=None,
            n_jobs=1, graduate=False)
y = np.array([0]*100 + [2]*100 + [4]*2 + [6]*20, dtype=np.float32)
a = pl.DataFrame({'cat': ['a']*100 + ['b']*100 + ['rareA']*2 + [None]*20})
b = pl.DataFrame({'cat': ['a']*100 + ['b']*100 + ['rareB']*2 + [None]*20})
ma = TBoostRegressor(**opts).fit(a, y)
mb = TBoostRegressor(**opts).fit(b, y)
ea, eb = json.loads(ma.tables(a)), json.loads(mb.tables(b))
print(ea == eb)
print(ea['tables'][0]['axes'][0]['levels'])
query = pl.DataFrame({'cat': ['rareA', 'rareB', 'never-seen', None]})
print(ma.predict(query))
print(mb.predict(query))
print("axis keys:", sorted(ea['tables'][0]['axes'][0].keys()))
print("table keys:", sorted(ea['tables'][0].keys()))
print("top keys:", sorted(ea.keys()))
