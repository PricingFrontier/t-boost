import warnings, json, pickle
import numpy as np
from t_boost import TBoostRegressor
warnings.simplefilter('ignore')
x = np.tile(np.array([0., 1.], dtype=np.float32), 100).reshape(-1, 1)
e = np.tile(np.array([1., 10.], dtype=np.float32), 100)
y = np.tile(np.array([1., 30.], dtype=np.float32), 100)
m = TBoostRegressor(objective='poisson', n_trees=20, n_bags=1, prune=False, validation_fraction=None, n_jobs=1).fit(x, y, exposure=e)
for name, estimator in [('original', m),
    ('bytes', TBoostRegressor.from_bytes(m.to_bytes())),
    ('json', TBoostRegressor.from_json(m.to_json())),
    ('pickle', pickle.loads(pickle.dumps(m)))]:
    print(name, "has _fit_exposure_:", hasattr(estimator, "_fit_exposure_"), "_ae_requires_exposure_:", getattr(estimator, "_ae_requires_exposure_", None))
    bank = json.loads(estimator.tables(x, sample_weight=np.ones(200, dtype=np.float32)))
    print("  ", bank['tables'][0]['support'], bank['f0'], bank['tables'][0]['values'])
    bank0 = json.loads(estimator.tables(x))
    print("   no-arg:", bank0['tables'][0]['support'], bank0['f0'])
    print("   preds equal:", np.allclose(estimator.predict(x[:2]), m.predict(x[:2])))
