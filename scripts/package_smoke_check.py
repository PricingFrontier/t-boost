"""Installed-package smoke check for the Release workflow's wheel and sdist lanes.

Run with the interpreter of a fresh environment that has only the built distribution (and
its declared dependencies) installed — never from the repository root, where the source
tree's ``t_boost`` would shadow the installed one.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", required=True, help="the version the package must report")
    args = parser.parse_args(argv)

    import numpy as np
    import polars as pl

    import t_boost
    from t_boost import TBoostClassifier, TBoostRegressor

    assert "site-packages" in (t_boost.__file__ or ""), f"t_boost imported from {t_boost.__file__}"
    assert t_boost.__version__ == args.version, f"{t_boost.__version__} != {args.version}"
    assert t_boost._t_boost.BUILD_PROFILE == "release", t_boost._t_boost.BUILD_PROFILE

    rng = np.random.default_rng(0)
    n = 2000
    df = pl.DataFrame({
        "Age": rng.integers(18, 80, n).astype(float),
        "Power": rng.uniform(4.0, 15.0, n),
        "Region": rng.choice(["A", "B", "C"], n),
        "Exposure": rng.uniform(0.1, 1.0, n),
    })
    lam = np.exp(-2.0 + 0.01 * df["Age"].to_numpy()) * df["Exposure"].to_numpy()
    df = df.with_columns(
        ClaimCount=pl.Series(rng.poisson(lam).astype(float)),
        Lapsed=pl.Series(rng.choice(["stay", "lapse", "switch"], n)),
    )
    features = ["Age", "Power", "Region"]

    freq = TBoostRegressor(objective="poisson", n_trees=50).fit(
        df.select(features + ["ClaimCount", "Exposure"]), "ClaimCount", exposure="Exposure"
    )
    rate = freq.predict(df)
    assert rate.shape == (n,) and np.all(np.isfinite(rate)) and np.all(rate > 0)
    tables = json.loads(freq.tables(df))
    assert tables["objective"]["loss"] == "Poisson", tables["objective"]

    clf = TBoostClassifier(n_trees=50).fit(df.select(features + ["Lapsed"]), "Lapsed")
    proba = clf.predict_proba(df)
    assert proba.shape == (n, 3) and np.allclose(proba.sum(axis=1), 1.0)

    print(f"package smoke ok: t-boost {t_boost.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
