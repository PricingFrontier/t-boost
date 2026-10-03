import warnings
import numpy as np
import polars as pl
from t_boost import TBoostClassifier, TBoostRegressor
warnings.simplefilter("ignore")

OPTIONS = dict(n_trees=8, n_bags=1, prune=False, validation_fraction=None, min_data_in_leaf=2, n_jobs=1)
X = np.arange(80, dtype=np.float32).reshape(-1, 1)

m = TBoostClassifier(**OPTIONS).fit(X, np.repeat(["a", "b"], 40))
print("classes before:", m.classes_)
try:
    m.fit(X, np.repeat(["wrong1", "wrong2"], 40), sample_weight=np.ones(79))
    print("NO EXCEPTION on bad refit")
except Exception as exc:
    print("refit raised:", type(exc).__name__, str(exc)[:120])
print("fitted:", m.__sklearn_is_fitted__())
print("classes after:", m.classes_)
print("predict unique:", np.unique(m.predict(X)))

frame = pl.DataFrame({"a": X[:, 0], "b": np.zeros(80, dtype=np.float32)})
r = TBoostRegressor(**OPTIONS).fit(frame, X[:, 0])
print("names before:", hasattr(r, "feature_names_in_"), getattr(r, "feature_names_in_", None))
reordered = frame.select(["b", "a"])
before = r.predict(reordered)
try:
    r.fit(frame, X[:-1, 0])
    print("NO EXCEPTION on bad regressor refit")
except Exception as exc:
    print("regressor refit raised:", type(exc).__name__, str(exc)[:120])
print("fitted:", r.__sklearn_is_fitted__())
print("has feature_names_in_:", hasattr(r, "feature_names_in_"))
print("max abs diff:", np.max(np.abs(before - r.predict(reordered))))
