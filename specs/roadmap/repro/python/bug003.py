import json
import numpy as np
from t_boost._t_boost import _Booster

x = np.empty((160, 0), dtype=np.float32)
cats = [["a"] * 40 + ["b"] * 40 + ["c"] * 40 + ["heldout_only"] * 40]
holdout = [False] * 120 + [True] * 40
y = np.array([0] * 40 + [1] * 40 + [2] * 40 + [0] * 40, dtype=np.float32)
changed = y.copy()
changed[120:] = 2
booster = _Booster(n_trees=1, n_jobs=1, cat_smooth=0, cat_min_data_per_group=0)

for target in (y, changed):
    m = booster.fit_multiclass(
        x, target, 3, ["a", "b", "c"], cat_x=cats, es_holdout=holdout
    )
    enc = json.loads(m.to_json())["model"]["classes"][0]["schema"]
    enc = enc["cat_encoders"]["encoders"][0]
    print([(v["label"], v["encoding"]) for v in enc["levels"]])
    print(enc["base"])

# Scalar control: same data through the scalar path with es_holdout
print("--- scalar control (fit with es_holdout) ---")
for target in (y, changed):
    m = booster.fit(x, target, cat_x=cats, es_holdout=holdout)
    enc = json.loads(m.to_json())["model"]["schema"]["cat_encoders"]["encoders"][0]
    print([(v["label"], v["encoding"]) for v in enc["levels"]])
    print(enc["base"])
