from pathlib import Path
import numpy as np
from t_boost import TBoostClassifier

def ticks():
    return {
        p.name: sum(map(int, (p / "stat").read_text().split()[13:15]))
        for p in Path("/proc/self/task").iterdir()
    }

x = np.arange(180, dtype=np.float32).reshape(-1, 1)
m = TBoostClassifier(
    n_trees=8, n_bags=1, prune=False, graduate=False,
    band_tolerance=None, validation_fraction=None, n_jobs=4,
).fit(x, np.repeat(np.arange(3), 60))
m.n_jobs = 1
query = np.tile(x, (10000, 1))
m.predict_proba(query)
before = ticks()
for _ in range(12):
    m.predict_proba(query)
after = ticks()
print("threads total:", len(after))
print({k: after[k] - before.get(k, 0)
       for k in after if after[k] > before.get(k, 0)})
# n_jobs=0 observation
m.n_jobs = 0
try:
    p = m.predict_proba(query[:5]); print("n_jobs=0 multiclass predict_proba OK, shape", p.shape)
except Exception as e:
    print("n_jobs=0 multiclass raised:", type(e).__name__, e)
try:
    d = m.decision_function(query[:5]); print("n_jobs=0 multiclass decision_function OK, shape", d.shape)
except Exception as e:
    print("n_jobs=0 multiclass decision_function raised:", type(e).__name__, e)
# binary control
b = TBoostClassifier(n_trees=8, n_bags=1, prune=False, graduate=False,
    band_tolerance=None, validation_fraction=None, n_jobs=4).fit(x, (x[:,0] > 90).astype(int))
b.n_jobs = 0
try:
    b.predict_proba(query[:5]); print("n_jobs=0 binary predict_proba OK (unexpected)")
except Exception as e:
    print("n_jobs=0 binary raised:", type(e).__name__, e)
