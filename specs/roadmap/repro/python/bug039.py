import numpy as np
from t_boost import TBoostRegressor

x = np.column_stack([
    np.r_[np.arange(300), np.full(100, np.nan)],
    np.arange(400),
]).astype(np.float32)
y = x[:, 1].copy()
w = np.r_[np.zeros(300), np.ones(100)].astype(np.float32)
options = dict(
    n_trees=5, n_bags=1, prune=False, graduate=False,
    band_tolerance=None, validation_fraction=None, n_jobs=1,
)
for label, design, target, mass in [
    ("original", x, y, w),
    ("drop zero-weight rows", x[w > 0], y[w > 0], w[w > 0]),
    ("drop uninformative feature", x[:, 1:], y, w),
]:
    try:
        model = TBoostRegressor(**options).fit(design, target, sample_weight=mass)
        print(label, "ok", model.predict(design[:2]))
    except ValueError as error:
        print(label, type(error).__name__, str(error))
# extra: distinct <= max_bin variant (midpoint path) -- doc says failure requires > max_bin distinct
x2 = x.copy(); x2[:300, 0] = np.arange(300) % 10
try:
    model = TBoostRegressor(**options).fit(x2, y, sample_weight=w)
    print("10-distinct zero-weight feature ok", model.predict(x2[:2]))
except ValueError as error:
    print("10-distinct zero-weight feature", type(error).__name__, str(error))
