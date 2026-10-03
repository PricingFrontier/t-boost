import sys, warnings
if "--block" in sys.argv:
    class BlockSklearn:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "sklearn" or fullname.startswith("sklearn."):
                raise ImportError("simulate runtime-only installation")
    sys.meta_path.insert(0, BlockSklearn())
import numpy as np
from t_boost import TBoostClassifier, TBoostRegressor
from t_boost._compat import SKLEARN_AVAILABLE
print("sklearn", SKLEARN_AVAILABLE)
x = np.arange(20, dtype=np.float32).reshape(-1, 1)
options = dict(n_trees=5, n_bags=1, prune=False, graduate=False, validation_fraction=None, n_jobs=1)
for model, y in [(TBoostRegressor(**options), np.zeros(20)), (TBoostClassifier(**options), np.tile([0, 1], 10))]:
    model.fit(x, y)
    for design, target in [(x, np.array([0.])), (x[:1], y[:1]), (x, np.column_stack([y, y])[:10])]:
        try:
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter("always")
                s = model.score(design, target)
            print(type(model).__name__, design.shape, target.shape, s, [str(w.message)[:60] for w in wl])
        except ValueError as error:
            print(type(model).__name__, type(error).__name__, str(error)[:120])
