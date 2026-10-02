# Variant: child uses n_jobs=None (ambient/global pool) and n_jobs=2 after a parent that fitted with n_jobs=2,
# plus a parent that never used n_jobs at all. Distinguishes serve_pool-cache from generic rayon-after-fork.
import multiprocessing as mp, warnings
import numpy as np
from t_boost import TBoostRegressor

def predict_child(model, x, queue, width):
    model.n_jobs = width
    queue.put(model.predict(x).tolist())

def trial(m, x, method, width, label):
    ctx = mp.get_context(method)
    queue = ctx.Queue()
    child = ctx.Process(target=predict_child, args=(m, x[:2], queue, width))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        child.start()
    child.join(5)
    alive = child.is_alive()
    print(label, method, width, "still running:", alive, "| warnings:", [str(x.message)[:80] for x in w])
    if alive:
        child.terminate(); child.join(2)
    else:
        print("   ->", queue.get(timeout=1))
    queue.close()

if __name__ == "__main__":
    x = np.arange(80, dtype=np.float32).reshape(-1, 1)
    m2 = TBoostRegressor(n_trees=4, n_bags=1, prune=False, graduate=False,
        band_tolerance=None, validation_fraction=None, n_jobs=2).fit(x, x[:, 0])
    m2.predict(x[:2])
    trial(m2, x, "fork", None, "parent-n_jobs=2 child-None")
    trial(m2, x, "fork", 2, "parent-n_jobs=2 child-2")
    trial(m2, x, "fork", 4, "parent-n_jobs=2 child-4(fresh)")
