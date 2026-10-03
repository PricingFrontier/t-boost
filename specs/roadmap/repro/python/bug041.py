import numpy as np
from t_boost import TBoostRegressor
rng = np.random.default_rng(80)
x = rng.uniform(-2, 2, size=(600, 2)).astype(np.float32)
y = (0.5 * (x[:, 0] + x[:, 1]) + np.sin(3*x[:, 0])*np.cos(2*x[:, 1]) + 0.1*rng.normal(size=600)).astype(np.float32)
m = TBoostRegressor(n_trees=40, n_bags=2, max_depth=4, interaction_gain_hurdle=0,
    validation_fraction=None, n_jobs=1, band_tolerance=.75, graduate=False).fit(x, y)
r = m.pricing_report(x, y)
print([(t['feature_set'], t['shape']) for t in r['tables']['tables']])
for a in r['actual_vs_expected']:
    print(a['feature'], len(a['rows']), a['axis']['cells'], sum(a['mass'][a['axis']['cells']:]))
# which table's axis is attached per raw feature
for t in r['tables']['tables']:
    print(t['feature_set'], [(ax['raw'], ax['cells'], len(ax['borders'])) for ax in t['axes']])
