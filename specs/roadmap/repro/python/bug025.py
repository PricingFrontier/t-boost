import warnings
import pickle
import numpy as np
from t_boost import TBoostClassifier
warnings.simplefilter('ignore')
print("numpy", np.__version__, "coercion dtype:", np.asarray([1, 2**63 + 1]).dtype)
x = np.tile(np.array([0., 1.], dtype=np.float32), 100).reshape(-1, 1)
y = np.tile(np.array([1, 2**63 + 1], dtype=np.uint64), 100)
m = TBoostClassifier(n_trees=20, n_bags=1, prune=False, validation_fraction=None, n_jobs=1).fit(x, y)
print("orig classes:", m.classes_.dtype, list(map(int, m.classes_)))
for name, restored in [
    ('bytes', TBoostClassifier.from_bytes(m.to_bytes())),
    ('json', TBoostClassifier.from_json(m.to_json())),
    ('pickle', pickle.loads(pickle.dumps(m))),
]:
    print(name, m.classes_.dtype, restored.classes_.dtype, list(map(int, restored.classes_)))
    print("  preds orig/restored:", list(map(int, m.predict(x[:2]))), list(map(int, restored.predict(x[:2]))))
