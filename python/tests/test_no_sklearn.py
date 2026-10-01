"""The estimators run WITHOUT scikit-learn installed (the `_compat` shim path).

scikit-learn is present in the test environment, so the no-sklearn world is reproduced in
a subprocess whose meta-path blocks every `sklearn` import — the only faithful way to
exercise the shims end to end (in-process unimporting is unreliable once sklearn is
loaded). One subprocess runs the whole scenario battery to keep the suite fast.
"""

from __future__ import annotations

import subprocess
import sys

_SCENARIO = r"""
import sys


class _BlockSklearn:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sklearn" or fullname.startswith("sklearn."):
            raise ImportError(f"{fullname} is blocked: this subprocess simulates no-sklearn")
        return None


sys.meta_path.insert(0, _BlockSklearn())

import numpy as np
import polars as pl

from t_boost import TBoostClassifier, TBoostRegressor
from t_boost._compat import SKLEARN_AVAILABLE, NotFittedError
from t_boost.metrics import mean_poisson_deviance

assert not SKLEARN_AVAILABLE, "the import blocker failed; this test proves nothing"
assert "sklearn" not in sys.modules

rng = np.random.default_rng(0)
n = 300
df = pl.DataFrame(
    {
        "age": rng.uniform(18, 80, n).astype(np.float32),
        "region": rng.choice(["N", "S", "E", "W"], n),
        "expo": rng.uniform(0.1, 1.0, n).astype(np.float32),
        "claims": rng.poisson(0.2, n).astype(np.float32),
    }
)

cheap = dict(n_trees=40, n_bags=1, validation_fraction=None, seed=0)

# NotFittedError before fit (shim class: same ValueError+AttributeError bases as sklearn's)
unfitted = TBoostRegressor(**cheap)
try:
    unfitted.predict(df)
except NotFittedError as exc:
    assert isinstance(exc, (ValueError, AttributeError))
else:
    raise AssertionError("predict before fit did not raise NotFittedError")

# polars-native fit with column-name y/exposure + auto-categorical
reg = TBoostRegressor(objective="poisson", **cheap).fit(df, "claims", exposure="expo")
pred = reg.predict(df)
assert np.isfinite(pred).all()
assert list(reg.feature_names_in_) == ["age", "region"]
assert mean_poisson_deviance(df["claims"].to_numpy(), pred, df["expo"].to_numpy()) > 0.0

# get_params/set_params shims round-trip and validate
params = reg.get_params()
assert params["objective"] == "poisson" and params["n_bags"] == 1
reg2 = TBoostRegressor(**cheap).set_params(objective="poisson")
assert reg2.get_params()["objective"] == "poisson"
try:
    reg2.set_params(nonsense=1)
except ValueError:
    pass
else:
    raise AssertionError("set_params accepted an invalid parameter")
repr(reg2)  # shim __repr__ must not crash

# score() shims: R^2 (regressor) and accuracy (classifier)
sq = TBoostRegressor(**cheap).fit(df.select("age", "claims"), "claims")
r2 = sq.score(df.select("age"), df["claims"].to_numpy())
assert -1.0 <= r2 <= 1.0

df_bin = df.with_columns((pl.col("claims") > 0).cast(pl.String).alias("label"))
clf = TBoostClassifier(**cheap).fit(df_bin.drop("claims", "expo"), "label")
proba = clf.predict_proba(df_bin)
assert proba.shape == (n, 2) and np.allclose(proba.sum(axis=1), 1.0, rtol=1e-6)
acc = clf.score(df_bin.select("age", "region"), df_bin["label"].to_numpy())
assert 0.0 <= acc <= 1.0

# multiclass path exercises the type_of_target shim (and rejects continuous y through it)
df_mc = df.with_columns(pl.Series("kind", rng.choice(["a", "b", "c"], n)))
mc = TBoostClassifier(**cheap).fit(df_mc.drop("claims", "expo"), "kind")
assert mc.predict_proba(df_mc).shape == (n, 3)
try:
    TBoostClassifier(**cheap).fit(df.select("age"), rng.uniform(0, 1, n))
except ValueError as exc:
    assert "continuous" in str(exc) or "label type" in str(exc)
else:
    raise AssertionError("continuous y was not rejected without sklearn")

# serialization + tables still work
loaded = TBoostRegressor.from_bytes(reg.to_bytes())
assert np.array_equal(loaded.predict(df), pred)
import json
bank = json.loads(reg.tables(df, exposure="expo"))
assert bank["mode"] == "Exact"

print("NO-SKLEARN-OK")
"""


def test_estimators_work_without_sklearn() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _SCENARIO],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "NO-SKLEARN-OK" in result.stdout
