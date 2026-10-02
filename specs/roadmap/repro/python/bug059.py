import sys
if "--block" in sys.argv:
    sys.modules['sklearn'] = None
import numpy as np
from t_boost import TBoostRegressor
from t_boost._compat import SKLEARN_AVAILABLE
print("SKLEARN_AVAILABLE =", SKLEARN_AVAILABLE)
for params in [dict(monotone_constraints=np.array([1, 0])), dict(categorical_features=np.array([True, False])), dict(monotone_constraints=[1, 0])]:
    m = TBoostRegressor(**params)
    try:
        print(repr(m)[:120])
    except Exception as exc:
        print(list(params), "->", type(exc).__name__, str(exc)[:120])
