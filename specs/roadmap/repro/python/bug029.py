import sys, warnings
if "--block" in sys.argv:
    sys.modules['sklearn'] = None
import numpy as np
from t_boost import TBoostRegressor
from t_boost._compat import SKLEARN_AVAILABLE
warnings.simplefilter('ignore')
print("SKLEARN_AVAILABLE =", SKLEARN_AVAILABLE)
x = np.arange(80, dtype=np.float32).reshape(-1, 1)
m = TBoostRegressor(n_trees=5, n_bags=1, prune=False, validation_fraction=None, n_jobs=1).fit(x, x[:, 0])
print(m.predict(x[:1]))
m.set_params()
print("fitted after set_params():", m.__sklearn_is_fitted__())
try:
    print(m.predict(x[:1]))
except Exception as exc:
    print(type(exc).__name__, str(exc)[:120])
