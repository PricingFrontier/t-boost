import sys, warnings
if len(sys.argv) > 1:
    sys.modules['pandas'] = None
import numpy as np
from t_boost import TBoostRegressor
from t_boost._ingest import _cat_level
warnings.simplefilter('ignore')
x = np.empty((300, 1), dtype=object)
x[:, 0] = [np.float32('nan'), 'a', 'b'] * 100
y = np.tile([0, 5, 10], 100).astype(np.float32)
from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / 'out'
OUT.mkdir(exist_ok=True)
path = OUT / 'tboost_deep_no_pandas.bin'
if len(sys.argv) > 1:
    m = TBoostRegressor.from_bytes(open(path, 'rb').read())
else:
    m = TBoostRegressor(categorical_features=[0], n_trees=20, n_bags=1,
        prune=False, validation_fraction=None, n_jobs=1).fit(x, y)
    open(path, 'wb').write(m.to_bytes())
print("pandas blocked" if len(sys.argv) > 1 else "pandas available", _cat_level(np.float32('nan')), m.predict(x)[:3])
