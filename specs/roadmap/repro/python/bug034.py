import numpy as np
from t_boost import TBoostRegressor

X = np.ones((100, 1), dtype=np.float32)
y = np.full(100, 10.0)
e = np.tile([1.0, 10.0], 50)

def gamma_dev(y, mu):
    return 2 * np.sum((y - mu) / mu - np.log(y / mu))
def tweedie_dev(y, mu, p):
    return 2 * np.sum(y**(2-p)/((1-p)*(2-p)) - y*mu**(1-p)/(1-p) + mu**(2-p)/(2-p))
def poisson_dev(y, mu):
    return 2 * np.sum(y*np.log(y/mu) - (y - mu))

for objective, power in [("poisson", 1), ("gamma", 2), ("tweedie", 1.5)]:
    options = dict(
        objective=objective, n_trees=20, n_bags=1, prune=False,
        graduate=False, validation_fraction=None, n_jobs=1,
        reanchor=False, reanchor_slope=False,
    )
    offset_model = TBoostRegressor(**options).fit(X, y, exposure=e)
    equivalent_model = TBoostRegressor(**options).fit(
        X, y / e, sample_weight=e ** (2 - power)
    )
    r_off = offset_model.predict(X)[0]
    r_eq = equivalent_model.predict(X)[0]
    print(objective, r_off, r_eq)
    if objective == "gamma":
        print("  gamma deviance fitted", gamma_dev(y, r_off * e), "optimum(5.5)", gamma_dev(y, 5.5 * e), "eq-model", gamma_dev(y, r_eq*e))
    elif objective == "tweedie":
        opt = np.sum(y * e**(1-1.5)) / np.sum(e**(2-1.5))
        print("  tweedie deviance fitted", tweedie_dev(y, r_off * e, 1.5), "optimum(%g)" % opt, tweedie_dev(y, opt * e, 1.5), "eq-model", tweedie_dev(y, r_eq*e, 1.5))
    else:
        print("  poisson deviance fitted", poisson_dev(y, r_off*e), "optimum", poisson_dev(y, (np.sum(y)/np.sum(e))*e))
    # reanchor variants
    for ra in [None, False, True]:
        o2 = dict(options); o2['reanchor'] = ra
        try:
            mm = TBoostRegressor(**o2).fit(X, y, exposure=e)
            print("  reanchor=%s -> %r" % (ra, mm.predict(X)[0]))
        except Exception as ex:
            print("  reanchor=%s -> ERR %s" % (ra, ex))
    # predict with exposure for the record
    try:
        print("  predict(X, exposure=e)[:2] =", offset_model.predict(X, exposure=e)[:2])
    except Exception as ex:
        print("  predict w/ exposure ERR", ex)
