"""The model report (`pricing_report` v2) and its HTML view (`html_report` / `report_html`)."""
from __future__ import annotations

import json
import math
import re
import types

import numpy as np
import pandas as pd
import polars as pl
import pytest

from t_boost import TBoostClassifier, TBoostRegressor, report_html
from t_boost import metrics
from t_boost import _report
from t_boost._html_report import (_W_FACTOR, _W_FULL, _W_HALF, _Log, _table_cells, cell_labels,
                                   run_label, span_label)
from t_boost._t_boost import _report_cell_quantiles, _report_cell_sums, _report_ranking

FAST = dict(n_trees=60, n_bags=3, n_jobs=1, seed=0)
_MLFLOW_KEY = re.compile(r"^[A-Za-z0-9_\-. /]{1,250}$")


def frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    df = pl.DataFrame({
        "age": rng.integers(18, 80, n).astype(float),
        "veh": rng.choice([f"V{i}" for i in range(6)], n),
        "dens": rng.lognormal(5, 1, n),
        "exp": rng.uniform(0.1, 1.0, n),
    })
    eta = -1.5 + 0.01 * (df["age"].to_numpy() - 45) ** 2 / 20 + (df["veh"] == "V3").to_numpy() * 0.4
    return df.with_columns(pl.Series("y", rng.poisson(np.exp(eta) * df["exp"].to_numpy())))


@pytest.fixture(scope="module")
def poisson():
    df = frame()
    tr, ev = df[:2200], df[2200:]
    m = TBoostRegressor(objective="poisson", **FAST).fit(
        tr.select("age", "veh", "dens"), tr["y"], exposure=tr["exp"])
    report = m.pricing_report(tr, "y", exposure="exp", eval_set=(ev, "y"),
                              eval_exposure="exp", name="Freq <v1>")
    assert [d["label"] for d in report["datasets"]] == ["train", "validation"]
    return m, tr, ev, report


def test_report_is_strict_json_and_renders_from_json(poisson):
    _, _, _, report = poisson
    assert report["report_version"] == 2
    text = json.dumps(report, allow_nan=False)
    html = report_html(report)
    assert report_html(json.loads(text)) == html
    assert html.startswith("<!doctype html>") and "Freq &lt;v1&gt;" in html
    # Usable where scripts are blocked and offline: no script, nothing fetched.
    assert "<script" not in html.lower()
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", html)
    with pytest.raises(ValueError, match="report_version"):
        report_html({**report, "report_version": 1})


def test_summary_metrics_are_mlflow_ready(poisson):
    _, _, _, report = poisson
    sm = report["summary_metrics"]
    assert {"train_deviance", "validation_deviance_explained", "validation_gini",
            "validation_ae"} <= set(sm)
    for key, value in sm.items():
        assert _MLFLOW_KEY.match(key) and isinstance(value, float) and np.isfinite(value)


def test_statistics_match_independent_computations(poisson):
    m, tr, _, report = poisson
    first = report["datasets"][0]["metrics"]
    y, e = tr["y"].to_numpy().astype(float), tr["exp"].to_numpy().astype(np.float32).astype(float)
    mu = m.predict(tr.select("age", "veh", "dens")) * e
    assert first["deviance"] == pytest.approx(metrics.mean_poisson_deviance(y, mu), rel=1e-6)
    # Poisson null: the constant rate Σy / Σe times the exposure.
    null = metrics.mean_poisson_deviance(y, y.sum() / e.sum() * e)
    assert first["null_deviance"] == pytest.approx(null, rel=1e-6)
    assert first["deviance_explained"] == pytest.approx(1 - first["deviance"] / null, rel=1e-6)
    rate = mu / e
    assert first["gini"] == pytest.approx(metrics.concentration_gini(y / e, rate, e), abs=1e-6)
    assert first["gini_normalised"] == pytest.approx(metrics.ordered_gini(y / e, rate, e), abs=1e-6)
    assert first["ae"] == pytest.approx(y.sum() / mu.sum(), rel=1e-6)
    lift = first["lift"]
    assert sum(b["mass"] for b in lift) == pytest.approx(e.sum())
    assert sum(b["actual"] * b["mass"] for b in lift) == pytest.approx(y.sum())


def test_lift_keeps_ties_together_and_ignores_row_order():
    rng = np.random.default_rng(1)
    rate = np.repeat([0.1, 0.2, 0.3], 100).astype(float)  # three distinct predictions
    y = rng.poisson(rate).astype(float)
    mass = np.ones_like(rate)
    a = _report_ranking(rate, y, mass, 10, 11)
    assert len(a["lift"]) == 3 and [b[0] for b in a["lift"]] == [100, 100, 100]
    perm = rng.permutation(len(rate))
    b = _report_ranking(rate[perm], y[perm], mass[perm], 10, 11)
    assert a["gini"] == pytest.approx(b["gini"], abs=1e-12)
    assert [x[1:4] for x in a["lift"]] == pytest.approx([x[1:4] for x in b["lift"]])
    assert a["gini"] == pytest.approx(metrics.concentration_gini(y, rate, mass), abs=1e-12)


def test_native_cell_sums_match_bincount():
    rng = np.random.default_rng(2)
    n = 500
    cells = np.ascontiguousarray(rng.integers(0, [4, 7], size=(n, 2)), dtype=np.uint32)
    y, p, w, e, v = (rng.uniform(size=n) for _ in range(5))
    out = _report_cell_sums(cells, [4, 7], y, p, w, e, v)
    for j, (act, exp, mass, rows, var) in enumerate(out):
        c = cells[:, j]
        np.testing.assert_allclose(act, np.bincount(c, w * y, minlength=[4, 7][j]))
        np.testing.assert_allclose(exp, np.bincount(c, w * p, minlength=[4, 7][j]))
        np.testing.assert_allclose(mass, np.bincount(c, w * e, minlength=[4, 7][j]))
        np.testing.assert_array_equal(rows, np.bincount(c, minlength=[4, 7][j]))
        np.testing.assert_allclose(var, np.bincount(c, w * v, minlength=[4, 7][j]))
    with pytest.raises(Exception):
        _report_cell_sums(cells, [3, 7], y, p, w, e)  # a cell outside its grid
    # The kernels read rows in C order; a Fortran-ordered array would be misread, not refused.
    with pytest.raises(ValueError, match="C-contiguous"):
        _report_cell_sums(np.asfortranarray(cells), [4, 7], y, p, w, e)
    with pytest.raises(ValueError, match="C-contiguous"):
        _report_cell_quantiles(np.asfortranarray(cells), [4, 7], np.asfortranarray(
            rng.uniform(size=(n, 2))), w, [0.5])


def _split(effects: np.ndarray, sets: list[list[int]], n_raw: int) -> np.ndarray:
    out = np.zeros((effects.shape[0], n_raw))
    for k, raws in enumerate(sets):
        for raw in raws:
            out[:, raw] += effects[:, k] / len(raws)
    return out


def test_native_shapley_split_and_cell_quantiles_match_numpy(poisson):
    m, tr, _, _ = poisson
    model = m._model
    x32, kw = m._serve_kwargs(tr.select("age", "veh", "dens"), model)
    f0, effects, sets = model.effect_contributions(x32, **kw)
    g0, feat = model.feature_contributions(x32, **kw)
    assert g0 == f0
    np.testing.assert_allclose(feat, _split(effects, sets, feat.shape[1]), atol=1e-12)
    cells = np.ascontiguousarray(model.cell_indices(x32, **kw), dtype=np.uint32)
    n_cells = [int(a["cells"]) for a in json.loads(model.merged_axes())]
    w = np.random.default_rng(3).uniform(0.2, 1.0, feat.shape[0])
    out = _report_cell_quantiles(cells, n_cells, np.ascontiguousarray(feat), w, [0.1, 0.9])
    for j, (weight, mean, q) in enumerate(out):
        for c in range(n_cells[j]):
            rows = cells[:, j] == c
            if not rows.any():
                assert weight[c] == 0 and math.isnan(mean[c])
                continue
            v, ww = feat[rows, j], w[rows]
            assert weight[c] == pytest.approx(ww.sum())
            assert mean[c] == pytest.approx(float((v * ww).sum() / ww.sum()))
            order = np.argsort(v, kind="stable")  # ties in row order, as the kernel keeps them
            cum = np.cumsum(ww[order])
            for i, qq in enumerate((0.1, 0.9)):
                at = min(int(np.searchsorted(cum, qq * cum[-1])), len(v) - 1)
                assert q[c, i] == v[order][at]
    with pytest.raises(Exception):
        _report_cell_quantiles(cells, n_cells, np.ascontiguousarray(feat), w, [1.5])


@pytest.mark.parametrize("objective", ["gamma", "tweedie", "squared_error"])
def test_other_objectives_and_offsets(objective):
    rng = np.random.default_rng(3)
    n = 1500
    x = rng.normal(size=(n, 2))
    o = rng.normal(scale=0.2, size=n)
    if objective == "squared_error":
        y = x[:, 0] + o + rng.normal(scale=0.5, size=n)
    elif objective == "gamma":
        y = rng.gamma(2.0, np.exp(0.3 * x[:, 0] + o) / 2.0)
    else:
        y = rng.poisson(np.exp(0.3 * x[:, 0] + o)) * rng.gamma(2.0, 0.5, n)
    m = TBoostRegressor(objective=objective, **FAST).fit(x, y, offset=o)
    report = m.pricing_report(x, y, offset=o)
    first = report["datasets"][0]["metrics"]
    mu = m.predict(x, offset=o)
    power = {"squared_error": 0.0, "gamma": 2.0, "tweedie": m.tweedie_rho}[objective]
    assert first["deviance"] == pytest.approx(metrics.mean_tweedie_deviance(y, mu, power=power))
    if objective == "squared_error":  # identity null: mean of y - offset, plus the offset
        null = metrics.mean_tweedie_deviance(y, np.mean(y - o) + o, power=0.0)
    else:  # log-link null: c * exp(o), c = Σ y a^(1-p) / Σ a^(2-p)
        a = np.exp(o)
        c = np.sum(y * a ** (1 - power)) / np.sum(a ** (2 - power))
        null = metrics.mean_tweedie_deviance(y, c * a, power=power)
    assert first["null_deviance"] == pytest.approx(null, rel=1e-6)
    assert report["model"]["columns"] == ["f0", "f1"] and not report["model"]["columns_named"]
    report_html(report)


def test_binary_classifier_with_pandas_input():
    rng = np.random.default_rng(4)
    n = 1500
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.choice(["x", "y", "z"], n)})
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-X["a"].to_numpy()))).astype(int)
    m = TBoostClassifier(categorical_features=["b"], **FAST).fit(X, y)
    report = m.pricing_report(X, y)
    first = report["datasets"][0]["metrics"]
    p = m.predict_proba(X)[:, 1]
    assert first["deviance"] == pytest.approx(metrics.mean_binomial_deviance(y, p))
    assert first["null_deviance"] == pytest.approx(metrics.mean_binomial_deviance(y, np.full(n, y.mean())))
    assert "odds multiplier" in report_html(report)


def test_multiclass_report():
    rng = np.random.default_rng(5)
    n = 2400
    X = pl.DataFrame({"a": rng.normal(size=n), "c": rng.choice(["p", "q", "r", "s"], n)})
    y = np.where(X["a"].to_numpy() > 0.5, 2, rng.integers(0, 2, n))
    w = rng.uniform(0.5, 2.0, n)
    m = TBoostClassifier(**FAST).fit(X, y, sample_weight=w)
    with pytest.raises(ValueError, match="explicit sample_weight"):
        m.pricing_report(X, y)
    report = m.pricing_report(X[:1600], y[:1600], sample_weight=w[:1600],
                              eval_set=(X[1600:], y[1600:]), eval_sample_weight=w[1600:])
    first = report["datasets"][0]["metrics"]
    proba = m.predict_proba(X[:1600])
    codes = np.searchsorted(m.classes_, y[:1600])
    assert first["deviance"] == pytest.approx(
        metrics.mean_multinomial_deviance(codes, proba, w[:1600]))
    matrix = np.asarray(first["class_matrix"]["expected"])
    assert matrix.sum(axis=1) == pytest.approx(np.bincount(codes, w[:1600]))
    ae = m.actual_vs_expected(X[:1600], y[:1600], sample_weight=w[:1600])
    assert {r["class"] for r in ae} == set(m.classes_.tolist())
    for row in ae:
        k = list(m.classes_).index(row["class"])
        assert sum(row["expected"]) == pytest.approx(float(np.sum(w[:1600] * proba[:, k])))
    assert {"train_ae_0", "train_gini_2", "validation_accuracy"} <= set(report["summary_metrics"])
    html = report_html(report)
    assert 'name="cl"' in html and "logit contribution" in html
    mm = m._multi_model
    x32, kw = m._serve_kwargs(X, mm)
    per_class = mm.feature_contributions(x32, **kw)
    assert len(per_class) == len(m.classes_)
    for (f0, effects, sets), (g0, feat) in zip(mm.effect_contributions(x32, **kw), per_class):
        assert g0 == f0
        np.testing.assert_allclose(feat, _split(effects, sets, feat.shape[1]), atol=1e-12)
    # The multiclass selection is recorded as `selection`, and drawn like the single-output path.
    assert report["fit"]["pruning"]["value"]["selection"]["sizes"]
    assert 'aria-label="share of the out-of-bag gain by bank size"' in html
    with pytest.raises(ValueError, match="not supported for multiclass"):
        m.pricing_report(X, y, sample_weight=w, exposure=np.ones(n))


def test_multiclass_cell_indices_coded_and_labelled_agree_with_merged_axes():
    rng = np.random.default_rng(6)
    n = 1200
    X = pl.DataFrame({"a": rng.normal(size=n), "c": rng.choice(["p", "q", "r"], n)})
    y = rng.integers(0, 3, n)
    m = TBoostClassifier(**FAST).fit(X, y)
    mm = m._multi_model
    x32, coded = m._serve_kwargs(X, mm)
    x32b, labels = m._serve_design(X)
    raw = mm.raw_feature_names().index("c")
    for k in range(3):
        a = np.asarray(mm.cell_indices(x32, class_index=k, **coded))
        b = np.asarray(mm.cell_indices(x32b, cat_x=labels, class_index=k))
        np.testing.assert_array_equal(a, b)
        axis = json.loads(mm.merged_axes(class_index=k))[raw]
        for level in axis["levels"]:
            rows = X["c"].to_numpy() == level["label"]
            assert set(a[rows, raw]) == {level["cell"]}


def test_coded_tables_export_matches_labelled():
    df = frame(n=1200)
    X = df.select("age", "veh", "dens")
    m = TBoostRegressor(objective="poisson", **FAST).fit(X, df["y"], exposure=df["exp"])
    x32, labels = m._serve_design(X)
    e = df["exp"].to_numpy().astype(np.float32)
    want = m._model.tables(x32, cat_x=labels, exposure=e, ref_measure="exposure")
    x32c, coded = m._serve_kwargs(X, m._model)
    assert m._model.tables(x32c, exposure=e, ref_measure="exposure", **coded) == want


def test_loaded_model_keeps_fit_time_and_pruning_but_not_history(poisson):
    m, tr, ev, _ = poisson
    assert isinstance(m.fitted_at_, str) and m.fit_duration_ > 0
    assert isinstance(m.graduation_report_, list) and m.graduation_report_
    report_now = m.pricing_report(tr, "y", exposure="exp")
    for blob in (m.to_bytes(), m.to_json()):
        loaded = (TBoostRegressor.from_bytes(blob) if isinstance(blob, bytes)
                  else TBoostRegressor.from_json(blob))
        assert loaded.fitted_at_ == m.fitted_at_ and loaded.fit_duration_ == m.fit_duration_
        assert loaded.pruning_report_["path"] == json.loads(json.dumps(m.pruning_report_["path"]))
        assert not hasattr(loaded, "evals_result_")
        assert loaded.graduation_report_ == json.loads(json.dumps(m.graduation_report_))
        report = loaded.pricing_report(tr, "y", exposure="exp")
        assert report["fit"]["pruning"]["available"]
        assert report["fit"]["graduation_tables"]["value"] == report_now["fit"]["graduation_tables"]["value"]
        assert not report["fit"]["evals_result"]["available"]
        report_html(report)


def test_refit_and_set_params_reset_fit_time():
    df = frame(n=800)
    X = df.select("age", "veh", "dens")
    m = TBoostRegressor(objective="poisson", **FAST).fit(X, df["y"], exposure=df["exp"])
    first = m.fitted_at_
    m.set_params(learning_rate=0.2)
    assert not hasattr(m, "fitted_at_") and not hasattr(m, "fit_duration_")
    m.fit(X, df["y"], exposure=df["exp"])
    assert m.fitted_at_ >= first


def test_ranked_path_records_the_cutoff(poisson):
    m = poisson[0]
    path = m.pruning_report_["path"]
    devs = path["oob_deviance"]
    assert path["total_gain"] == pytest.approx(devs[0] - min(devs))
    assert path["min_step"] == int(np.argmin(devs))
    if path["fraction_step"] is not None:
        assert path["best"] == max(path["fraction_step"], path["tolerance_step"])
    steps = [s["step"] for s in path["sequence"]]
    assert steps == sorted(steps) and all(1 <= s < len(path["sizes"]) for s in steps)


def test_thresholds_are_validated_and_drive_the_flags(poisson):
    m, tr, ev, _ = poisson
    kw = dict(exposure="exp", eval_set=(ev, "y"), eval_exposure="exp")
    with pytest.raises(ValueError, match="unknown threshold"):
        m.pricing_report(tr, "y", thresholds={"nope": 1}, **kw)
    with pytest.raises(ValueError, match="psi_warn"):
        m.pricing_report(tr, "y", thresholds={"psi_warn": 0.5, "psi_critical": 0.1}, **kw)
    with pytest.raises(ValueError, match="finite and non-negative"):
        m.pricing_report(tr, "y", thresholds={"ae_test": float("nan")}, **kw)
    with pytest.raises(ValueError, match="unknown threshold"):
        m.pricing_report(tr, "y", thresholds={"ae_eval": 0.05}, **kw)
    strict = m.pricing_report(tr, "y", thresholds={"psi_warn": 0.0, "psi_critical": 1e9}, **kw)
    assert {f["status"] for f in strict["review"] if f["check"] == "psi"} == {"warning"}
    assert strict["thresholds"]["psi_warn"] == 0.0
    off = m.pricing_report(tr, "y", thresholds={"psi_warn": None, "psi_critical": None}, **kw)
    assert not [f for f in off["review"] if f["check"] == "psi"]


def test_labels_name_the_datasets_and_metric_keys(poisson):
    m, tr, ev, _ = poisson
    r = m.pricing_report(tr, "y", exposure="exp", eval_set=(ev, "y"), eval_exposure="exp",
                         labels=("fit", "holdout 2024"))
    assert [d["label"] for d in r["datasets"]] == ["fit", "holdout 2024"]
    assert "holdout_2024_gini" in r["summary_metrics"]
    with pytest.raises(ValueError, match="labels"):
        m.pricing_report(tr, "y", exposure="exp", eval_set=(ev, "y"), eval_exposure="exp",
                         labels=("a", "a"))
    with pytest.raises(ValueError, match="labels"):
        m.pricing_report(tr, "y", exposure="exp", eval_set=(ev, "y"), eval_exposure="exp",
                         labels=("only one",))


def test_train_validation_and_test(poisson):
    m, tr, ev, _ = poisson
    va, te = ev[:400], ev[400:]
    r = m.pricing_report(tr, "y", exposure="exp", eval_set=[(va, "y"), (te, "y")],
                         eval_exposure=["exp", "exp"])
    assert [d["label"] for d in r["datasets"]] == ["train", "validation", "test"]
    assert [(p["reference"], p["compared"]) for p in r["drift"]] == [
        ("train", "validation"), ("train", "test")]
    assert {"test_gini", "validation_ae", "train_ae"} <= set(r["summary_metrics"])
    checks = {(f["check"], f["dataset"]) for f in r["review"]}
    assert {("beats_null", "validation"), ("beats_null", "test"), ("train_gap", "test"),
            ("overall_ae", "test")} <= checks
    ae_test = next(f for f in r["review"] if f["check"] == "overall_ae" and f["dataset"] == "test")
    assert ae_test["threshold"] == r["thresholds"]["ae_test"]
    html = report_html(r)
    assert "train → test" in html and 'id="ds-2" checked' in html
    # numpy vectors per dataset; None where a dataset has none
    e = ev["exp"].to_numpy()
    r2 = m.pricing_report(tr, "y", exposure="exp",
                          eval_set=[(va.select("age", "veh", "dens"), va["y"]),
                                    (te.select("age", "veh", "dens"), te["y"])],
                          eval_exposure=[e[:400], e[400:]])
    assert r2["datasets"][2]["metrics"]["deviance"] == pytest.approx(
        r["datasets"][2]["metrics"]["deviance"])
    with pytest.raises(ValueError, match="list of 2 entries"):
        m.pricing_report(tr, "y", exposure="exp", eval_set=[(va, "y"), (te, "y")],
                         eval_exposure="exp")
    with pytest.raises(ValueError, match="up to two pairs"):
        m.pricing_report(tr, "y", exposure="exp", eval_set=[(va, "y")] * 3,
                         eval_exposure=["exp"] * 3)


def test_hostile_names_and_levels_are_escaped():
    rng = np.random.default_rng(7)
    n = 1200
    X = pl.DataFrame({"<script>alert(1)</script>": rng.normal(size=n),
                      'lvl"><img src=x onerror=1>': rng.choice(['"><img src=x>', "</style>", "a&b"], n)})
    y = rng.poisson(1.0, n)
    m = TBoostRegressor(objective="poisson", **FAST).fit(X, y)
    html = m.html_report(X, y, name="<b>x</b>")
    assert "<script" not in html.lower() and "<img" not in html.lower()
    assert "</style>" not in html.split("</style>", 1)[1]  # only the page's own style block ends
    assert "&lt;b&gt;x&lt;/b&gt;" in html


def test_unscorable_evaluation_data_is_flagged_not_raised():
    rng = np.random.default_rng(8)
    n = 1200
    X = pl.DataFrame({"a": rng.normal(size=n), "c": rng.choice(["p", "q"], n)})
    y = rng.poisson(1.0, n)
    m = TBoostRegressor(objective="poisson", unknown_category="error", **FAST).fit(X, y)
    Xe = X.with_columns(pl.Series("c", ["new"] * 10 + ["p"] * (n - 10)))
    report = m.pricing_report(X, y, eval_set=(Xe, y))
    flags = [f for f in report["review"] if f["check"] == "scoring"]
    assert flags and flags[0]["status"] == "critical"
    assert report["datasets"][1]["unseen"][0]["value"] == "new"
    report_html(report)


def test_unpruned_bank_report_stays_small():
    rng = np.random.default_rng(9)
    n = 3000
    X = rng.normal(size=(n, 8))
    y = rng.poisson(np.exp(0.2 * X[:, 0] * X[:, 1]))
    m = TBoostRegressor(objective="poisson", prune=False, n_trees=120, n_bags=2, n_jobs=1,
                        seed=0).fit(X, y)
    html = report_html(m.pricing_report(X, y))
    assert len(html) < 5_000_000
    assert "Pruning" in html and "prune=False" in html
    # The Tables tab holds numbers, not heatmaps: no chart at all.
    tables = html.split('id="tables">', 1)[1].split("</section>", 1)[0]
    assert "<table" in tables and " × " in tables and "<svg" not in tables


def test_factor_check_reads_equal_mass_bands_shared_by_every_dataset(poisson):
    _, _, _, report = poisson
    first, second = report["datasets"]
    for a, b in zip(first["actual_vs_expected"], second["actual_vs_expected"]):
        assert 1 <= len(a["bands"]) <= 10
        assert [x["cells"] for x in a["bands"]] == [x["cells"] for x in b["bands"]]
        assert sum(x["actual"] for x in a["bands"]) == pytest.approx(sum(a["actual"]))
        assert sum(x["mass"] for x in a["bands"]) == pytest.approx(sum(a["mass"]))
        assert "_var" not in a
    passing = [f for f in report["review"] if f["check"] == "factor_ae" and f["status"] == "pass"]
    assert passing and all("of the mass" in f["message"] for f in passing)
    html = report_html(report)
    assert "A/E by band (the review check)" in html and "furthest from balance" in html


def test_page_is_css_only_tabs_with_one_dataset_switch(poisson):
    _, _, _, report = poisson
    html = report_html(report)
    tabs = re.findall(r'name="tb" id="tb-(\d)"( checked)?', html)
    assert [int(i) for i, _ in tabs] == list(range(7)) and tabs[0][1] == " checked"
    for i, name in enumerate(("overview", "performance", "factors", "interactions", "tables",
                              "build", "checks")):
        assert f'<section class="tab tb tb-{i}" id="{name}">' in html
        assert f'<label for="tb-{i}">' in html
    # One dataset switch for the whole page; the last (most independent) dataset selected.
    assert re.findall(r'name="ds" id="ds-(\d)"( checked)?', html) == [("0", ""), ("1", " checked")]
    # Every chart is drawn at the pixel width of its card: half, full and factor width.
    widths = {int(w) for w in re.findall(r'<svg class="chart" [^>]*?width="(\d+)"', html)}
    assert {_W_HALF, _W_FULL, _W_FACTOR} <= widths


def test_cell_labels_read_as_integers_on_an_integer_feature(poisson):
    integer = {"cells": 6, "borders": [18.5, 20.0, 25.5, 99.0], "integer": True}
    assert cell_labels(integer) == ["missing", "≤ 18", "19–20", "21–25", "26–99", "≥ 100"]
    assert run_label(integer, 2, 3) == "19–25"
    # Half-integer borders alone do not make a feature integer: without the flag, intervals.
    assert cell_labels({**integer, "integer": False})[1:3] == ["≤ 18.5", "(18.5, 20]"]
    continuous = {"cells": 4, "borders": [0.123456, 0.5]}
    assert cell_labels(continuous) == ["missing", "≤ 0.1235", "(0.1235, 0.5]", "> 0.5"]
    assert run_label(continuous, 1, 3) == "all" and span_label(None, None, True) == "all"
    levels = [{"label": "a", "cell": 1}, {"label": "b", "cell": 1},
              {"label": "<rare>", "cell": 2, "synthetic": True, "members": ["c", "d"]}]
    assert cell_labels({"cells": 3, "levels": levels}) == ["missing", "a, b", "<rare> (2 levels)"]
    # The report's bands use the same labels: age is an integer feature.
    _, _, _, report = poisson
    age = next(e for e in report["datasets"][0]["actual_vs_expected"] if e["feature"] == "age")
    assert all(re.fullmatch(r"missing|≤ \d+|≥ \d+|\d+|\d+–\d+", b["label"]) for b in age["bands"])


def test_log_axis_ticks_cover_the_range():
    wide = _Log(0.8, 6.85, 200, 0)
    assert 1.0 in wide.ticks and len(wide.ticks) >= 3 and max(wide.ticks) >= 4
    narrow = _Log(0.86, 1.3, 200, 0, max_ticks=4)
    assert 2 <= len(narrow.ticks) <= 4 and 1.0 in narrow.ticks
    tiny = _Log(0.99, 1.01, 200, 0)
    assert 1.0 in tiny.ticks and len(tiny.ticks) >= 2


def test_features_and_decile_uncertainty(poisson):
    _, _, _, report = poisson
    feats = report["features"]
    importance = [f["importance"] for f in feats]
    assert importance == sorted(importance, reverse=True)
    assert sum(importance) == pytest.approx(sum(t["sobol"] for t in report["tables"]["tables"]))
    for f in feats:
        assert f["importance"] == pytest.approx(f["own_share"] + f["interaction_share"])
        ow = f.get("one_way")
        if ow:
            t = next(t for t in report["tables"]["tables"] if t["feature_names"] == [f["feature"]])
            assert ow["cells"] == len(t["relativities"])
            assert ow["missing"]["effect"] == pytest.approx(t["relativities"][0])
    for d in report["datasets"]:
        for b in d["metrics"]["lift"]:
            # z = (A - E) / σ and sd = σ / E, so z = (A/E - 1) / sd.
            assert b["z"] == pytest.approx((b["ae"] - 1.0) / b["sd"])


def test_missing_level_and_decile_checks_follow_their_thresholds(poisson):
    m, tr, ev, _ = poisson
    kw = dict(exposure="exp", eval_set=(ev, "y"), eval_exposure="exp")
    r = m.pricing_report(tr, "y", thresholds={"missing_effect": 0.0}, **kw)
    want = {f["feature"] for f in r["features"] if f.get("one_way")
            and f["one_way"]["missing"]["support"] == 0 and f["one_way"]["missing"]["effect"] != 1.0}
    assert {f["feature"] for f in r["review"] if f["check"] == "missing_level"} == want
    off = m.pricing_report(tr, "y", thresholds={"missing_effect": None}, **kw)
    assert not [f for f in off["review"] if f["check"] == "missing_level"]
    # With no tolerance and no z bar, every decile off balance is a warning.
    r = m.pricing_report(tr, "y", thresholds={"decile_ae": 0.0, "factor_z": 0.0}, **kw)
    for d in r["datasets"]:
        off_balance = [b for b in d["metrics"]["lift"] if b["ae"] != 1.0 and b["z"] is not None]
        flagged = [f for f in r["review"] if f["check"] == "decile_ae"
                   and f["dataset"] == d["label"] and f["status"] == "warning"]
        assert len(flagged) == len(off_balance)


def test_shap_by_level_adds_up_to_the_score(poisson):
    m, tr, _, report = poisson
    first = report["datasets"][0]
    assert all("shap" in e for e in first["actual_vs_expected"])
    assert all("shap" not in e for d in report["datasets"][1:] for e in d["actual_vs_expected"])
    # Shapley efficiency: the features' values add up to each row's score less the intercept, so
    # their exposure-weighted means per level, pooled, give the mean score less the intercept.
    pooled = 0.0
    for e in first["actual_vs_expected"]:
        sh = e["shap"]
        assert len(sh["mean"]) == len(e["mass"])
        for c, mass in enumerate(e["mass"]):
            if mass > 0:
                assert sh["p10"][c] <= sh["p90"][c]
                pooled += mass * sh["mean"][c]
        total = sum(e["mass"])
    e_tr = tr["exp"].to_numpy().astype(np.float32).astype(float)
    raw = np.asarray(m.predict_raw(tr.select("age", "veh", "dens")), dtype=float)
    f0 = report["tables"]["f0"]
    assert pooled / total == pytest.approx(float(np.sum(e_tr * (raw - f0)) / e_tr.sum()), abs=1e-5)


def test_interactions_summary_and_view(poisson):
    _, _, _, report = poisson
    inters = report["interactions"]
    tables = report["tables"]["tables"]
    mains = sum(t["sobol"] for t in tables if len(t["feature_names"]) == 1)
    assert mains + sum(i["share"] for i in inters) == pytest.approx(1.0)
    shares = [i["share"] for i in inters]
    assert shares == sorted(shares, reverse=True)
    for i in inters:
        assert i["order"] == len(i["features"]) >= 2
        lo, hi = i["range"]
        assert lo - 1e-12 <= i["spread"][0] <= i["spread"][1] <= hi + 1e-12
    html = report_html(report)
    pane = html.split('id="interactions">', 1)[1].split("</section>", 1)[0]
    if inters:
        assert pane.count('name="ix"') == 0 and html.count('name="ix"') == len(inters)
        assert " × ".join(inters[0]["features"]) in pane and "one-way table" in pane
    # The Tables tab tints relativities: green above no effect, yellow below.
    tables_tab = html.split('id="tables">', 1)[1].split("</section>", 1)[0]
    assert re.search(r'class="n [gy][1-5]"', tables_tab)


def test_tints_follow_the_effect_relative_to_the_largest():
    from t_boost._html_report import _tones

    assert _tones([1.0, 2.0, 0.5, 1.2, 3.0], [1, 1, 1, 1, 0], True) == ["", "g5", "y5", "g3", "muted"]
    assert _tones([0.0, 0.4, -0.1], [1, 1, 1], False) == ["", "g5", "y3"]
    assert _tones([1.0, 1.0], [1, 1], True) == ["", ""]


def test_integer_labels_need_whole_values_not_just_half_integer_borders():
    # A continuous feature split at 0.5 (values 0.1 and 0.9) reads as intervals, not "≤ 0" and
    # "≥ 1"; a whole-valued one as integers, with a categorical column before both.
    rng = np.random.default_rng(4)
    n = 2000
    X = pl.DataFrame({"c": rng.choice(["a", "b"], n), "k": rng.integers(0, 6, n).astype(float),
                      "x": rng.choice([0.1, 0.9], n)})
    y = rng.poisson(np.exp(-1 + 0.3 * X["k"].to_numpy() + 0.8 * (X["x"].to_numpy() > 0.5)))
    m = TBoostRegressor(objective="poisson", **FAST).fit(X, y)
    report = m.pricing_report(X, y)
    ae = {e["feature"]: e for e in report["datasets"][0]["actual_vs_expected"]}
    assert ae["k"]["axis"]["integer"] is True and ae["x"]["axis"]["integer"] is False
    assert "integer" not in ae["c"]["axis"]
    labels = {b["label"] for b in ae["x"]["bands"]} - {"missing"}
    assert labels and labels <= {"≤ 0.5", "> 0.5", "all"}
    assert all(re.fullmatch(r"missing|≤ \d+|≥ \d+|\d+|\d+–\d+", b["label"])
               for b in ae["k"]["bands"])
    for t in report["tables"]["tables"]:
        for a in t["axes"]:
            if not a.get("levels"):
                assert a["integer"] is (a["name"] == "k")
    html = report_html(report)
    assert "x ≤ 0.5" in html and "x ≤ 0:" not in html and "x ≥ 1:" not in html


def test_merged_cells_map_to_table_cells_by_exact_borders():
    merged = {"cells": 5, "borders": [5e-11, 1.5e-10, 2.5e-10]}
    assert _table_cells(merged, dict(merged)) == [0, 1, 2, 3, 4]
    assert _table_cells(merged, {"cells": 3, "borders": [1.5e-10]}) == [0, 1, 1, 2, 2]
    # `merged_axes` writes float32's shortest form and `tables` the float64 of the same float32:
    # the two parse to different floats, which an exact or a 1e-9-tolerant comparison mis-maps.
    borders = np.unique(np.random.default_rng(9).normal(size=60).astype(np.float32))
    short = {"cells": len(borders) + 2, "borders": [float(str(b)) for b in borders]}
    wide = {"cells": len(borders) + 2, "borders": [float(b) for b in borders]}
    assert short["borders"] != wide["borders"]
    assert _table_cells(short, wide) == list(range(len(borders) + 2))
    assert _table_cells(wide, short) == list(range(len(borders) + 2))


@pytest.mark.parametrize("multiclass", [False, True])
def test_shap_reads_the_bank_the_tables_are_exported_from(multiclass):
    # With no interactions a feature's Shapley value is its one-way table, so on every level the
    # report's SHAP mean equals the exported table, also when the report's data carry a mass the
    # fit did not (the export then re-centres the tables on it).
    rng = np.random.default_rng(6)
    n = 3000
    X = pl.DataFrame({"a": rng.normal(size=n), "c": rng.choice(["p", "q", "r"], n)})
    q = (X["c"] == "q").to_numpy()
    if multiclass:
        y = np.where(X["a"].to_numpy() > 0.5, 2, q.astype(int))
        m = TBoostClassifier(max_interaction_order=1, **FAST).fit(X, y)
    else:
        y = rng.poisson(np.exp(0.4 * X["a"].to_numpy() + 0.5 * q))
        m = TBoostRegressor(objective="poisson", max_interaction_order=1, **FAST).fit(X, y)
    heavy = np.where(X["c"].to_numpy() == "p", 50.0, 1.0)
    values = {}
    for weight in (None, heavy):
        report = m.pricing_report(X, y, sample_weight=weight)
        banks = report["tables"] if multiclass else {None: report["tables"]}
        for e in report["datasets"][0]["actual_vs_expected"]:
            t = next((t for t in banks[e.get("class")]["tables"]
                      if t["feature_names"] == [e["feature"]]), None)
            to_table = _table_cells(e["axis"], t["axes"][0]) if t is not None else None
            for c, mass in enumerate(e["mass"]):
                if mass > 0:  # a class with no table on the feature gives it no effect
                    want = t["values"][to_table[c]] if t is not None else 0.0
                    assert e["shap"]["mean"][c] == pytest.approx(want, abs=1e-9)
            values[weight is None, e.get("class"), e["feature"]] = t and t["values"]
    # The weights moved the tables, so the re-centred bank was exercised.
    assert any(v != values[True, k, f] for (unweighted, k, f), v in values.items()
               if not unweighted)


def test_decile_and_factor_checks_with_the_z_condition_off():
    # With `factor_z=None` the A/E tolerance alone flags a decile, as it does a factor band, and a
    # band with no z (no variance) still gets its message.
    rng = np.random.default_rng(7)
    n = 3000
    X = rng.normal(size=(n, 2))
    y = (rng.random(n) < 1 / (1 + np.exp(-X[:, 0]))).astype(int)
    m = TBoostClassifier(**FAST).fit(X, y)
    emptied = y.copy()
    emptied[np.argsort(m.predict_proba(X)[:, 1])[-n // 10:]] = 0  # no events in the top decile
    report = m.pricing_report(X, emptied, thresholds={"factor_z": None})
    decile = [f for f in report["review"] if f["check"] == "decile_ae"]
    assert any(f["status"] == "warning" for f in decile)
    assert not any(f["status"] == "pass" for f in decile)
    for d in report["datasets"]:
        for e in d["actual_vs_expected"]:
            for b in e["bands"]:
                b["z"] = None
    th = _report.resolve_thresholds({"factor_z": None, "factor_ae": 1e-9})
    factor = [f for f in _report.review(m, report, th) if f["check"] == "factor_ae"]
    assert any(f["status"] == "warning" and ", z" not in f["message"] for f in factor)


def test_logistic_null_deviance_with_a_large_offset():
    # Under an offset of 5 the constant model is the intercept -5 (probability 0.5 on balanced
    # outcomes); Newton from logit(mean) overshot it and stopped at probability 1.
    n = 400
    y = np.tile([0.0, 1.0], n // 2)
    p = types.SimpleNamespace(y=y, w=np.ones(n), e=np.ones(n), offset=np.full(n, 5.0))
    mu = _report._null_prediction("logistic", "logit", p, 0.0)
    np.testing.assert_allclose(mu, 0.5, atol=1e-12)
    assert metrics.mean_binomial_deviance(y, mu) == pytest.approx(2 * math.log(2), abs=1e-9)
    # Row offsets, exposure and weights: the root of the intercept's score equation.
    rng = np.random.default_rng(8)
    w, yr = rng.uniform(0.5, 2.0, n), (rng.random(n) < 0.3).astype(float)
    p = types.SimpleNamespace(y=yr, w=w, e=rng.uniform(0.2, 1.0, n), offset=rng.normal(0, 3, n))
    mu = _report._null_prediction("logistic", "logit", p, 0.0)
    assert float(np.sum(w * (yr - mu))) == pytest.approx(0.0, abs=1e-8)


def test_multinomial_deviance_needs_probability_rows():
    with pytest.raises(ValueError, match="sum to 1"):
        metrics.mean_multinomial_deviance(np.array([0, 1]), [[1.0, 1.0], [1.0, 1.0]])
    proba = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float32).astype(np.float64)
    assert metrics.mean_multinomial_deviance(np.array([1, 0]), proba) == pytest.approx(
        -math.log(0.8) - math.log(0.6), rel=1e-6)
