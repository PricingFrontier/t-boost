import numpy as np
from sklearn.metrics import roc_auc_score
from t_boost.metrics import ordered_gini
p = np.ones(4)
for y in [np.array([0., 0., 1., 1.]), np.array([1., 1., 0., 0.]), np.array([0., 1., 0., 1.])]:
    print(y, ordered_gini(y, p), 2 * roc_auc_score(y, p) - 1)
# permutation invariance with tied groups
rng = np.random.default_rng(1)
y = (rng.random(200) < 0.3).astype(float)
s = np.round(rng.random(200), 1)  # many ties
perm = rng.permutation(200)
print("ties: gini", ordered_gini(y, s), "permuted", ordered_gini(y[perm], s[perm]), "2AUC-1", 2*roc_auc_score(y, s)-1)
# no-ties control
s2 = rng.random(200)
print("no ties: gini", ordered_gini(y, s2), "permuted", ordered_gini(y[perm], s2[perm]), "2AUC-1", 2*roc_auc_score(y, s2)-1)
