import sys, warnings
BLOCK = "--block" in sys.argv
if BLOCK:
    sys.modules['sklearn'] = None
import numpy as np
from t_boost import TBoostClassifier
from t_boost._compat import SKLEARN_AVAILABLE
print("SKLEARN_AVAILABLE =", SKLEARN_AVAILABLE)
warnings.simplefilter('ignore')
for labels in [[0., np.inf], [0., 1., np.nan], [0., 1., np.inf], [0., np.nan]]:
    y = np.tile(labels, 100)
    x = np.tile(np.arange(len(labels), dtype=np.float32), 100).reshape(-1, 1)
    try:
        m = TBoostClassifier(n_trees=20, n_bags=1, prune=False, validation_fraction=None, n_jobs=1).fit(x, y)
        print(labels, "->", m.classes_, m.predict(x)[:len(labels)])
    except Exception as exc:
        print(labels, "-> raised", type(exc).__name__, str(exc)[:140])
