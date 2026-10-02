import json, warnings
import numpy as np
import polars as pl
from t_boost import TBoostClassifier, TBoostRegressor
warnings.simplefilter('ignore')
x = pl.DataFrame({
    'signal': np.tile([0., 1.], 100).astype(np.float32),
    'noise': np.zeros(200, dtype=np.float32),
})
y = np.tile([0, 1], 100)
m = TBoostClassifier(n_trees=20, n_bags=1, prune=False, validation_fraction=None, n_jobs=1).fit(x, y)
print("== A")
for labels in [['A', 'B', 'C'], []]:
    doc = json.loads(m.to_json())
    doc['classes_'] = labels
    try:
        loaded = TBoostClassifier.from_json(json.dumps(doc))
    except Exception as exc:
        print("load raised", type(exc).__name__, str(exc)[:120]); continue
    print('accepted classes', loaded.classes_, 'probability shape', loaded.predict_proba(x[:2]).shape)
    try:
        print(loaded.predict(x[:2]))
    except Exception as exc:
        print(type(exc).__name__, str(exc))

print("== bytes variant")
blob = m.to_bytes()
length = int.from_bytes(blob[4:8], 'big')
header = json.loads(blob[8:8+length])
header['classes_'] = ['A', 'B', 'C']
encoded = json.dumps(header).encode()
corrupt = blob[:4] + len(encoded).to_bytes(4, 'big') + encoded + blob[8+length:]
loaded = TBoostClassifier.from_bytes(corrupt)
print(loaded.classes_.tolist(), loaded.predict_proba(x[:1]).shape)

print("== B")
m = TBoostRegressor(n_trees=20, n_bags=1, prune=False, validation_fraction=None, n_jobs=1).fit(x, y.astype(np.float32)*10)
doc = json.loads(m.to_json())
print(doc['feature_names_in_'])
print(json.loads(doc['model'])['model']['schema']['feature_names'])
doc['feature_names_in_'] = ['noise', 'signal']
loaded = TBoostRegressor.from_json(json.dumps(doc))
print(m.predict(x[:2]), loaded.predict(x[:2]))

print("== control: mixed categorical fit")
rng = np.random.default_rng(0)
cf = pl.DataFrame({
    'category': rng.choice(['p', 'q', 'r'], 200).tolist(),
    'value': rng.uniform(0, 1, 200).astype(np.float32),
})
cy = cf['value'].to_numpy() * 10 + (cf['category'] == 'p').to_numpy()
cm = TBoostRegressor(n_trees=20, n_bags=1, prune=False, validation_fraction=None, n_jobs=1).fit(cf, cy)
cdoc = json.loads(cm.to_json())
print("header feature_names_in_:", cdoc['feature_names_in_'])
print("native feature_names:   ", json.loads(cdoc['model'])['model']['schema']['feature_names'])
print("cat_indices:", cdoc.get('cat_indices'), "n_features_in_ key:", cdoc.get('n_features_in_'))
print("round trip preds equal:", np.allclose(TBoostRegressor.from_json(cm.to_json()).predict(cf), cm.predict(cf)))
