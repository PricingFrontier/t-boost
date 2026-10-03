import math
import numpy as np
from t_boost.sklearn import _guard_mu, _guard_reanchor

raw = np.zeros(100)
y = np.r_[np.ones(10), np.zeros(90)]
shifted = _guard_reanchor("logistic", raw, y, np.ones(100), True)
print("mean mu after helper shift:", _guard_mu("logistic", shifted).mean())
print("y.mean():", y.mean())
print("helper delta:", shifted[0] - raw[0], " expected log(1/9)=", math.log(0.1 / 0.9))
# native-style bisection on the same data
def bisect(raw, y, w):
    lo, hi = -60.0, 60.0
    obs = float((w * y).sum())
    for _ in range(96):
        mid = 0.5 * (lo + hi)
        pred = float((w / (1 + np.exp(-(raw + mid)))).sum())
        if pred < obs: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)
print("bisection delta:", bisect(raw, y, np.ones(100)))
# varying logits / weights
rng = np.random.default_rng(0)
raw2 = rng.normal(size=100)
w2 = rng.uniform(0.5, 2, size=100)
s2 = _guard_reanchor("logistic", raw2, y, w2, True)
print("varying: helper delta", s2[0]-raw2[0], " bisect delta", bisect(raw2, y, w2))
print("varying: balance after helper", (w2*_guard_mu("logistic", s2)).sum(), " target", (w2*y).sum())
# control: poisson log link closed form
raw3 = np.log(np.full(100, 0.5))
s3 = _guard_reanchor("poisson", raw3, y, np.ones(100), True)
print("poisson control: balance", _guard_mu("poisson", s3).sum(), " target", y.sum())
# disabled path
print("disabled unchanged:", np.array_equal(_guard_reanchor("logistic", raw, y, np.ones(100), False), raw))
