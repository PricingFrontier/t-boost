import numpy as np
from t_boost import TBoostClassifier
x = np.ones((200, 1), dtype=np.float32)
y = np.repeat([0, 1], 100)
e = np.repeat([1., 4.], 100).astype(np.float32)
for prune in [False, True]:
    m = TBoostClassifier(n_trees=5, n_bags=1, prune=prune, validation_fraction=None, n_jobs=1).fit(x, y, exposure=e)
    raw = m.decision_function(x).astype(float)
    correct = 1 / (1 + np.exp(-(raw + np.log(e))))
    a = m.actual_vs_expected(x, y, exposure=e)[0]
    proba = m.predict_proba(x)[:, 1]
    print("prune", prune, "raw[0]", raw[0], "proba[0]", proba[0], "proba[199]", proba[199],
          "actual", sum(a['actual']), "expected", sum(a['expected']), "offset-aware", sum(correct),
          "mass", sum(a['mass']), "_ae_requires_exposure_", getattr(m, "_ae_requires_exposure_", None))
    print(" rows", a['rows'], "actual", a['actual'], "expected", a['expected'])
