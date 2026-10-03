import multiprocessing as mp
import numpy as np
from t_boost import TBoostRegressor

def predict_child(model, x, queue, width):
    model.n_jobs = width
    queue.put(model.predict(x).tolist())

if __name__ == "__main__":
    x = np.arange(80, dtype=np.float32).reshape(-1, 1)
    m = TBoostRegressor(
        n_trees=4, n_bags=1, prune=False, graduate=False,
        band_tolerance=None, validation_fraction=None, n_jobs=2,
    ).fit(x, x[:, 0])
    print("parent", m.predict(x[:2]))
    for method, width in [("fork", 2), ("fork", 3), ("spawn", 2)]:
        ctx = mp.get_context(method)
        queue = ctx.Queue()
        child = ctx.Process(target=predict_child, args=(m, x[:2], queue, width))
        child.start()
        child.join(5)
        print(method, width, "still running:", child.is_alive())
        if child.is_alive():
            child.terminate()
            child.join(2)
        else:
            print(queue.get(timeout=1))
        queue.close()
