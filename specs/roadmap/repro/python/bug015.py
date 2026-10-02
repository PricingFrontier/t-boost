from sklearn.base import clone
from t_boost import TBoostRegressor

a = TBoostRegressor(early_stopping=20)
b = TBoostRegressor().set_params(early_stopping=20)
print("rounds ctor/set_params:", a.early_stopping_rounds, b.early_stopping_rounds)
p = b.get_params()
print("b get_params pair:", p["early_stopping"], p["early_stopping_rounds"], p["early_stopping_adaptive"])
try:
    clone(b); print("clone(b) OK")
except Exception as exc:
    print("clone(b):", type(exc).__name__, str(exc)[:200])
try:
    clone(a); print("clone(a) OK")
except Exception as exc:
    print("clone(a):", type(exc).__name__, str(exc)[:200])

a = TBoostRegressor(prune_size_penalty=3.0)
b = TBoostRegressor().set_params(prune_size_penalty=3.0)
print("lambda ctor/set_params:", a.prune_lambda_tables, b.prune_lambda_tables)
try:
    clone(b); print("clone(b) OK")
except Exception as exc:
    print("clone(b):", type(exc).__name__, str(exc)[:200])
try:
    clone(a); print("clone(a) OK")
except Exception as exc:
    print("clone(a):", type(exc).__name__, str(exc)[:200])
# float form
c = TBoostRegressor().set_params(early_stopping=1.2)
print("adaptive ctor/set_params:", TBoostRegressor(early_stopping=1.2).early_stopping_adaptive, c.early_stopping_adaptive)
