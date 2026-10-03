import numpy as np
import polars as pl
from t_boost import TBoostRegressor
opts = dict(n_trees=10, n_bags=1, prune=False,
            validation_fraction=None, n_jobs=1)
y = np.repeat([0., 10.], 100).astype(np.float32)
for label in ['__t_boost_missing__', 'ordinary-value']:
    x = pl.DataFrame({'cat': [None]*100 + [label]*100})
    m = TBoostRegressor(**opts).fit(x, y)
    print(label, m.predict(x)[[0, -1]])
# numpy object path (uses _cat_level)
x = np.empty((200, 1), dtype=object); x[:, 0] = [None]*100 + ['__t_boost_missing__']*100
m = TBoostRegressor(categorical_features=[0], **opts).fit(x, y)
print("numpy object path", m.predict(x)[[0, -1]])
# control: __t_boost_rare__ literal
try:
    x = pl.DataFrame({'cat': ['a']*100 + ['__t_boost_rare__']*100})
    m = TBoostRegressor(**opts).fit(x, y)
    print("__t_boost_rare__ accepted", m.predict(x)[[0, -1]])
except Exception as ex:
    print("__t_boost_rare__ ->", type(ex).__name__, ex)
# __t_boost_rare__ with cat_min_data_per_group=0 (guard is after the min<=0 early return)
try:
    m = TBoostRegressor(cat_min_data_per_group=0, **opts).fit(x, y)
    print("__t_boost_rare__ with cat_min_data_per_group=0 accepted", m.predict(x)[[0, -1]])
except Exception as ex:
    print("__t_boost_rare__ cat_min_data_per_group=0 ->", type(ex).__name__, ex)
