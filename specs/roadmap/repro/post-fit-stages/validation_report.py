"""Companion to validation_matrix.py: the model report (pricing_report, html_report) must build
for every supported objective x setting, with the training rows and a held-out set.

    uv run --no-sync python specs/roadmap/repro/post-fit-stages/validation_report.py \
        --out specs/roadmap/repro/post-fit-stages/results/validation_report.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import validation_matrix as vm  # noqa: E402


def run(task: str, setting: str) -> dict:
    ctor, keys = vm.SETTINGS[setting]
    d = vm.make_data(task)
    cut = d["cut"]
    out = {"task": task, "setting": setting}
    try:
        est = vm.build(task, ctor, 4).fit(d["X"][:cut], d["y"][:cut], **vm.fit_kwargs(task, keys, d))
    except Exception as exc:
        out.update(status="refused", error=f"{type(exc).__name__}: {exc}"[:200])
        return out
    kw, ev = {}, {}
    if task in ("poisson", "tweedie"):
        kw["exposure"], ev["eval_exposure"] = d["exposure"][:cut], d["exposure"][cut:]
    if "sample_weight" in keys:
        kw["sample_weight"], ev["eval_sample_weight"] = d["weight"][:cut], d["weight"][cut:]
    if "exposure" in keys:
        kw["exposure"], ev["eval_exposure"] = d["exposure"][:cut], d["exposure"][cut:]
    if "offset" in keys:
        kw["offset"], ev["eval_offset"] = d["offset"][:cut], d["offset"][cut:]
    try:
        rep = est.pricing_report(d["X"][:cut], d["y"][:cut], eval_set=(d["X"][cut:], d["y"][cut:]),
                                 **kw, **ev)
        html = est.html_report(d["X"][:cut], d["y"][:cut], eval_set=(d["X"][cut:], d["y"][cut:]),
                               **kw, **ev)
        crit = [f["message"] for f in rep["review"] if f["status"] == "critical"]
        es = next((f for f in rep["review"] if f["check"] == "early_stopping"), None)
        out.update(status="ok", html_kb=round(len(html) / 1024), critical=crit,
                   early_stopping=None if es is None else [es["status"], es["message"][:90]])
    except Exception:
        out.update(status="problem", error=traceback.format_exc(limit=4)[-700:])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(vm.TASKS))
    ap.add_argument("--settings", default=",".join(vm.SETTINGS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "a") as fh:
        for task in a.tasks.split(","):
            for setting in a.settings.split(","):
                r = run(task, setting)
                fh.write(json.dumps(r, default=str) + "\n")
                fh.flush()
                print(f"{task:14s} {setting:17s} {r['status']:8s} {r.get('early_stopping') or ''} "
                      f"{(r.get('error') or '')[-200:]}", flush=True)


if __name__ == "__main__":
    main()
