import warnings
import numpy as np
from t_boost import TBoostClassifier, TBoostRegressor
warnings.simplefilter("ignore")

OPTIONS = dict(n_trees=8, n_bags=1, prune=False, validation_fraction=None, min_data_in_leaf=2, n_jobs=1)
X = np.arange(80, dtype=np.float32).reshape(-1, 1)

for name in ["squared_error", "squared-error", "l2", "regression", "SQUARED_ERROR"]:
    try:
        m = TBoostRegressor(objective=name, **OPTIONS).fit(X, X[:, 0] + 1)
    except Exception as exc:
        print(name, "fit failed:", type(exc).__name__, str(exc)[:100]); continue
    print(name, "link =", m.link, "| objective attr =", m.objective)
    try:
        m.predict_contributions(X[:1]); print("   contributions OK")
    except Exception as exc:
        print("   contributions:", type(exc).__name__, str(exc)[:160])
    loaded = TBoostRegressor.from_bytes(m.to_bytes())
    print("   after from_bytes: link =", loaded.link, "objective =", loaded.objective)

for name in ["logistic", "Logistic"]:
    c = TBoostClassifier(objective=name, **OPTIONS).fit(X, np.repeat([0, 1], 40))
    print(name, "classifier link =", c.link)
    try:
        c.predict_contributions(X[:1]); print("   contributions OK")
    except Exception as exc:
        print("   contributions:", type(exc).__name__, str(exc)[:160])

exposure = np.full(80, 2.0, dtype=np.float32)
y = np.ones(80, dtype=np.float32)
for name in ["poisson", "POISSON"]:
    m = TBoostRegressor(objective=name, **OPTIONS).fit(X, y, exposure=exposure)
    rep = m.actual_vs_expected(X, y, exposure=exposure)
    print(name, "ae type:", type(rep).__name__)
    report = rep[0]
    print(name, "sum expected:", sum(report["expected"]), "| sum pred*exposure:", float(np.sum(m.predict(X) * exposure)))
