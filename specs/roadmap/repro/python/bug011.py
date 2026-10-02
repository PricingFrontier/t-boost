import warnings
import numpy as np
import polars as pl
from t_boost import TBoostClassifier, TBoostRegressor
warnings.simplefilter("ignore")

OPT = dict(n_trees=1, n_bags=1, prune=False, validation_fraction=None, n_jobs=1)
for name in ["t-boost-multiclass", "t-boost-multiclass-tables", "t-boost-tables"]:
    frame = pl.DataFrame({name: list(range(20))})
    m = TBoostClassifier(**OPT).fit(frame, [0, 1] * 10)
    try:
        j = TBoostClassifier.from_json(m.to_json())
        print(name, "json: OK", np.array_equal(j.predict(frame), m.predict(frame)))
    except Exception as exc:
        print(name, "json:", type(exc).__name__, str(exc)[:150])
    try:
        b = TBoostClassifier.from_bytes(m.to_bytes())
        print(name, "bytes: OK preds equal", np.array_equal(b.predict(frame), m.predict(frame)))
    except Exception as exc:
        print(name, "bytes:", type(exc).__name__, str(exc)[:150])
    # regressor too
    r = TBoostRegressor(**OPT).fit(frame, np.arange(20, dtype=np.float32))
    try:
        TBoostRegressor.from_json(r.to_json()); print(name, "regressor json: OK")
    except Exception as exc:
        print(name, "regressor json:", type(exc).__name__, str(exc)[:150])
