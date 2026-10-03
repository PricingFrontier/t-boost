import numpy as np, warnings
from t_boost.metrics import mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance
warnings.simplefilter("ignore")
for label, f in [
    ("poisson nan y", lambda: mean_poisson_deviance([np.nan, 1.], [1., 1.])),
    ("gamma inf pred", lambda: mean_gamma_deviance([1., 1.], [np.inf, 1.])),
    ("poisson nan weight", lambda: mean_poisson_deviance([1., 1.], [1., 1.], [np.nan, 1.])),
    ("tweedie power nan", lambda: mean_tweedie_deviance([2.], [5.], power=np.nan)),
    ("tweedie power inf", lambda: mean_tweedie_deviance([2.], [5.], power=np.inf)),
    ("sq err inf y", lambda: mean_tweedie_deviance([np.inf], [1.], power=0)),
    ("zero weights", lambda: mean_poisson_deviance([1., 1.], [1., 1.], [0., 0.])),
]:
    try:
        print(label, f())
    except Exception as e:
        print(label, "ERROR", type(e).__name__, str(e)[:100])
