import numpy as np
from t_boost import TBoostClassifier
OPTIONS = dict(n_trees=8, n_bags=1, prune=False, validation_fraction=None, min_data_in_leaf=2, n_jobs=1)
X = np.arange(80, dtype=np.float32).reshape(-1, 1)
for label, y in [("1/2", np.repeat([1, 2], 40)), ("0/1", np.repeat([0, 1], 40)), ("-1/1", np.repeat([-1, 1], 40)), ("bool", np.repeat([False, True], 40))]:
    try:
        m = TBoostClassifier(**OPTIONS).fit(X, y)
        report = m.actual_vs_expected(X, y)[0]
        print(label, "classes_", m.classes_, "count pos", np.count_nonzero(y == m.classes_[1]),
              "sum actual", sum(report["actual"]), "sum expected", round(sum(report["expected"]), 4))
    except Exception as e:
        print(label, "ERROR", type(e).__name__, str(e)[:120])
y = np.repeat(["a", "b"], 40)
try:
    m = TBoostClassifier(**OPTIONS).fit(X, y)
    print("strings fit ok, classes_", m.classes_)
    report = m.actual_vs_expected(X, y)[0]
    print("strings sum actual", sum(report["actual"]))
except Exception as e:
    print("strings ERROR", type(e).__name__, str(e)[:160])
