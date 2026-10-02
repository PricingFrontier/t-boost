"""`predict_contributions` (rustystats' record contract) and `feature_importances_`.

The deployed model is its rating tables, so contributions are not an approximation: the base
value plus a row's contributions is the served raw score, and the inverse link of that is the
prediction.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from t_boost import TBoostClassifier, TBoostRegressor

warnings.filterwarnings("ignore", category=UserWarning)

RECORD_KEYS = {
    "family", "link", "output_space", "prediction_space", "base_value", "sum_contributions",
    "prediction_from_contributions", "prediction_value", "contributions",
}
ENTRY_KEYS = {"term", "term_type", "feature", "feature_value", "contribution", "rank"}


def _frame(n: int = 3000, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    region = pl.Series(rng.choice(["a", "b", "c", None], n).tolist(), dtype=pl.String)
    df = pl.DataFrame(
        {
            "region": region,
            "age": rng.normal(size=n),
            "veh": rng.normal(size=n),
            "Exposure": rng.uniform(0.2, 1.0, size=n),
        }
    )
    age, veh = df["age"].to_numpy(), df["veh"].to_numpy()
    is_a = (df["region"] == "a").fill_null(False).to_numpy()
    mu = np.exp(0.3 * age + 0.4 * is_a + 0.3 * age * veh) * df["Exposure"].to_numpy()
    return df.with_columns(ClaimCount=pl.Series(rng.poisson(mu).astype(np.float64)))


@pytest.fixture(scope="module")
def frequency() -> tuple[TBoostRegressor, pl.DataFrame]:
    df = _frame()
    model = TBoostRegressor(objective="poisson", n_trees=300, n_bags=1, seed=0)
    return model.fit(df, "ClaimCount", exposure="Exposure"), df


def test_records_reproduce_the_served_score_exactly(frequency) -> None:
    model, df = frequency
    rows = df.head(200)
    records = model.predict_contributions(rows)
    raw = model.predict_raw(rows)
    pred = model.predict(rows)
    assert len(records) == 200
    for i, rec in enumerate(records):
        assert set(rec) == RECORD_KEYS
        assert rec["family"] == "poisson" and rec["link"] == "log"
        assert rec["output_space"] == "linear_predictor" and rec["prediction_space"] == "response"
        assert all(set(c) == ENTRY_KEYS for c in rec["contributions"])
        # Bit-exact: the decomposition is the scorer's own sum, before its float32 rounding.
        assert np.float32(rec["prediction_from_contributions"]) == np.float32(raw[i])
        total = rec["base_value"] + sum(c["contribution"] for c in rec["contributions"])
        assert total == pytest.approx(rec["prediction_from_contributions"], abs=1e-12)
        assert rec["prediction_value"] == pytest.approx(pred[i], rel=1e-6)
        ranks = sorted(c["rank"] for c in rec["contributions"])
        assert ranks == list(range(1, len(ranks) + 1))


def test_effect_terms_name_main_effects_and_interactions(frequency) -> None:
    model, df = frequency
    rec = model.predict_contributions(df.head(1))[0]
    for c in rec["contributions"]:
        parts = c["term"].split(":")
        assert c["term_type"] == ("main" if len(parts) == 1 else "interaction")
        assert c["feature"] == c["term"]
        assert set(parts) <= {"region", "age", "veh"}
        if len(parts) == 1:
            assert c["feature_value"] == df[parts[0]][0]
        else:
            assert c["feature_value"] == {p: df[p][0] for p in parts}


def test_split_interactions_gives_shapley_values_per_feature(frequency) -> None:
    model, df = frequency
    rows = df.head(50)
    effects = model.predict_contributions(rows)
    shapley = model.predict_contributions(rows, split_interactions=True)
    for e, s in zip(effects, shapley):
        assert [c["term"] for c in s["contributions"]] == ["region", "age", "veh"]
        assert all(c["term_type"] == "feature" for c in s["contributions"])
        assert s["prediction_from_contributions"] == e["prediction_from_contributions"]
        assert s["sum_contributions"] == pytest.approx(e["sum_contributions"], abs=1e-12)


def test_exposure_adds_a_log_exposure_term(frequency) -> None:
    model, df = frequency
    rows = df.head(20)
    records = model.predict_contributions(rows, exposure="Exposure")
    expected = model.predict(rows) * rows["Exposure"].to_numpy()
    for i, rec in enumerate(records):
        last = rec["contributions"][-1]
        assert last["term"] == "exposure" and last["term_type"] == "exposure"
        assert last["feature_value"] == rows["Exposure"][i]
        assert last["contribution"] == pytest.approx(np.log(rows["Exposure"][i]))
        assert rec["prediction_value"] == pytest.approx(expected[i], rel=1e-6)
    by_array = model.predict_contributions(rows, exposure=rows["Exposure"].to_numpy())
    assert [r["prediction_value"] for r in by_array] == [r["prediction_value"] for r in records]


def test_exposure_is_validated(frequency) -> None:
    model, df = frequency
    rows = df.head(3)
    with pytest.raises(ValueError, match="strictly positive"):
        model.predict_contributions(rows, exposure=np.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="rows"):
        model.predict_contributions(rows, exposure=np.ones(2))
    gaussian = TBoostRegressor(n_trees=50, n_bags=1).fit(df.drop("Exposure"), "ClaimCount")
    with pytest.raises(ValueError, match="log-link"):
        gaussian.predict_contributions(rows.drop("Exposure"), exposure=np.ones(3))


def test_dataframe_format_matches_records(frequency) -> None:
    model, df = frequency
    rows = df.head(30)
    records = model.predict_contributions(rows)
    long = model.predict_contributions(rows.lazy(), return_format="dataframe")
    n_terms = len(records[0]["contributions"])
    assert long.height == 30 * n_terms
    assert long["contribution"].to_list() == [
        c["contribution"] for r in records for c in r["contributions"]
    ]
    assert long["prediction_value"].to_list()[::n_terms] == [r["prediction_value"] for r in records]
    with pytest.raises(ValueError, match="return_format"):
        model.predict_contributions(rows, return_format="table")


def test_contributions_survive_a_save_and_load(frequency) -> None:
    model, df = frequency
    rows = df.head(10)
    loaded = TBoostRegressor.from_bytes(model.to_bytes())
    assert loaded.predict_contributions(rows) == model.predict_contributions(rows)
    np.testing.assert_array_equal(loaded.feature_importances_, model.feature_importances_)


def test_binary_classifier_decomposes_the_logit() -> None:
    df = _frame(seed=1)
    x = df.drop("ClaimCount", "Exposure")
    clf = TBoostClassifier(n_trees=200, n_bags=1, seed=0).fit(x, (df["ClaimCount"] > 0).to_numpy())
    records = clf.predict_contributions(x.head(100))
    proba = clf.predict_proba(x.head(100))[:, 1]
    assert records[0]["link"] == "logit" and records[0]["family"] == "logistic"
    np.testing.assert_allclose([r["prediction_value"] for r in records], proba, rtol=1e-6)


def test_unnamed_input_with_categoricals_maps_back_to_input_columns() -> None:
    """The native model orders numeric features first; terms must still name and read the
    caller's own column positions."""
    df = _frame(seed=2)
    x = np.empty((df.height, 3), dtype=object)
    x[:, 0] = df["region"].to_numpy()
    x[:, 1] = df["age"].to_numpy()
    x[:, 2] = df["veh"].to_numpy()
    model = TBoostRegressor(objective="poisson", n_trees=200, n_bags=1, categorical_features=[0])
    model.fit(x, df["ClaimCount"].to_numpy())
    rec = model.predict_contributions(x[:1], split_interactions=True)[0]
    values = {c["term"]: c["feature_value"] for c in rec["contributions"]}
    assert values == {"f0": x[0, 0], "f1": x[0, 1], "f2": x[0, 2]}
    assert model.feature_importances_.shape == (3,)


def test_multichannel_categoricals_decompose_exactly() -> None:
    df = _frame(seed=3)
    model = TBoostRegressor(
        objective="poisson", n_trees=200, n_bags=1, cat_channels=["mean", "count"]
    ).fit(df, "ClaimCount", exposure="Exposure")
    rows = df.head(100)
    records = model.predict_contributions(rows)  # validate=True checks every row
    assert {c["term"] for c in records[0]["contributions"]} >= {"region"}


def test_feature_importances_are_sobol_shares_by_input_feature(frequency) -> None:
    model, _ = frequency
    imp = model.feature_importances_
    assert imp.shape == (3,) and np.all(imp >= 0)
    assert imp.sum() == pytest.approx(1.0)
    assert int(np.argmax(imp)) == 1  # age carries the strongest signal in the fixture


def test_unsupported_models_say_why() -> None:
    df = _frame(n=600, seed=4)
    unpruned = TBoostRegressor(n_trees=30, n_bags=1, prune=False).fit(
        df.drop("Exposure"), "ClaimCount"
    )
    assert not hasattr(unpruned, "feature_importances_")
    with pytest.raises(ValueError, match="prune=True"):
        unpruned.predict_contributions(df.head(2))
    multi = TBoostClassifier(n_trees=30, n_bags=1).fit(
        df.drop("ClaimCount", "Exposure"), np.arange(df.height) % 3
    )
    with pytest.raises(ValueError, match="multiclass"):
        multi.predict_contributions(df.head(2))
    with pytest.raises(Exception, match="not fitted"):
        TBoostRegressor().predict_contributions(df.head(2))
