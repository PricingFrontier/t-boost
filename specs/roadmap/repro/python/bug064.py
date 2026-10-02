import json
import numpy as np
from t_boost import TBoostRegressor

x = np.tile(np.array([[0,0],[0,1],[1,0],[1,1]], np.float32), (20,1))
y = x[:,0]+x[:,1]
m = TBoostRegressor(n_trees=100,n_bags=1,max_depth=3,max_interaction_order=1,prune=False,
    graduate=False,band_tolerance=None,validation_fraction=None,n_jobs=1,
    leaf_refine_steps=0,reanchor=False).fit(x,y)
for correlated in [False, True]:
    z=x.copy()
    if correlated:z[:,1]=z[:,0]
    e=json.loads(m.tables(z,ref_measure='joint',sample_weight=np.ones(len(z),np.float32)))
    variance=np.var(m.predict_raw(z))
    print('correlated',correlated,'modelvariance',variance)
    for t in e['tables']:
        print(t['feature_set'],t['variance'],t['sobol'],'variance/modelvariance',t['variance']/variance)
    print('sum sobol', sum(t['sobol'] for t in e['tables']))
