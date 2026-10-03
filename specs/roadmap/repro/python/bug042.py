import numpy as np
from t_boost import TBoostClassifier
x = np.tile(np.array([-1, -1, -1, 1, 1, 1], dtype=np.float32), 100).reshape(-1, 1)
y = np.tile([0, 1, 1, 0, 2, 2], 100)
m = TBoostClassifier(n_trees=10, n_bags=1, prune=False, validation_fraction=None,
    n_jobs=1, colsample_bytree=1, leaf_refine_steps=0).fit(x, y)
records = m.predict_contributions(x[:2])
print([[len(c['contributions']) for c in row['classes']] for row in records])
print("proba finite:", np.all(np.isfinite(m.predict_proba(x[:2]))))
try:
    print(m.predict_contributions(x[:2], return_format='dataframe'))
except Exception as e:
    print("ERROR", type(e).__name__, str(e)[:300])
