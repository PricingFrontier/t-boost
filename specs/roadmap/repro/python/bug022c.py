# Control: forked child with n_jobs=None on a parallelism-sized query (10000x80 rows).
import multiprocessing as mp
import numpy as np
from t_boost import TBoostRegressor

def predict_child(model, x, queue, width):
    model.n_jobs = width
    queue.put(model.predict(x)[:2].tolist())

if __name__ == "__main__":
    x = np.arange(80, dtype=np.float32).reshape(-1, 1)
    m = TBoostRegressor(n_trees=16, n_bags=1, prune=False, graduate=False,
        band_tolerance=None, validation_fraction=None, n_jobs=2).fit(x, x[:, 0])
    big = np.tile(x, (10000, 1))
    m.predict(big)
    for width in [None, 2]:
        ctx = mp.get_context("fork")
        queue = ctx.Queue()
        child = ctx.Process(target=predict_child, args=(m, big, queue, width))
        child.start(); child.join(10)
        print("fork", width, "big query still running:", child.is_alive())
        if child.is_alive():
            child.terminate(); child.join(2)
        else:
            print("   ->", queue.get(timeout=1))
        queue.close()
