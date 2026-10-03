import json
import numpy as np
from t_boost import TBoostRegressor
rng = np.random.default_rng(5)
x = rng.normal(size=(300, 4)).astype(np.float32)
y = (2 * np.sign(x[:, 0]) * np.sign(x[:, 1]) * np.sign(x[:, 2]) + 0.7 * x[:, 3] + 0.1 * rng.normal(size=300))
m = TBoostRegressor(n_trees=10, n_bags=1, max_depth=6, interaction_gain_hurdle=0.0,
    validation_fraction=None, n_jobs=1, prune_selector="fold_vote", band_tolerance=None, graduate=False).fit(x, y)
bank = json.loads(m._model.to_json())["model"]["bank"]
print("factored:", [(f["u"], len(f["boxes"])) for f in bank["factored"]])
print("dense tables:", [t["u"] for t in bank["tables"]])
print("deployed:", m.pruning_report_["deployed"])
print("kept_not_deployed:", m.pruning_report_["kept_not_deployed"])
print("kept:", m.pruning_report_["kept"])
print("deployed_supports():", m._model.deployed_supports())
print("deployed_census():", m._model.deployed_census())
