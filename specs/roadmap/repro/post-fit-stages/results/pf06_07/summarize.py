"""Summaries of band_exp.jsonl and grad_exp.jsonl (last record wins per key).

    summarize.py band     # banding variants vs the shipped banding, per dataset and pooled
    summarize.py grad     # graduation rules per base model
    summarize.py facts    # the PF-06/PF-07 facts table (band on/off x graduation on/off)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent


def load(name: str, key: tuple[str, ...]) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    p = RES / name
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        r = json.loads(line)
        r.setdefault("tag", "")
        if r["tag"] is None:
            r["tag"] = ""
        out[tuple(r.get(k) for k in key)] = r
    return out


def mse(v: list[float]) -> str:
    a = np.asarray(v, dtype=float)
    if a.size == 0:
        return "-"
    se = a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else float("nan")
    return f"{a.mean():+.3f} ± {se:.3f}"


def band() -> None:
    rows = load("band_exp.jsonl", ("dataset", "split", "n_bags", "variant", "tag"))
    by: dict[tuple, dict] = defaultdict(dict)
    for (ds, sp, nb, var, tag), r in rows.items():
        by[(ds + (("/" + tag) if tag else ""), nb)].setdefault(sp, {})[var] = r
    variants = sorted({k[3] for k in rows if not k[3].startswith("_")} - {"unbanded", "shipped", "base"})
    print("| dataset | bags | splits | banding vs none (test %) | cells none → shipped | "
          + " | ".join(f"{v} vs shipped (test %, better/worse, cells ×)" for v in variants) + " |")
    print("|---" * (5 + len(variants)) + "|")
    pooled: dict[str, list[float]] = defaultdict(list)
    for (ds, nb) in sorted(by):
        sp = by[(ds, nb)]
        bvn = [100 * (s["shipped"]["test"] - s["unbanded"]["test"]) / s["unbanded"]["test"]
               for s in sp.values() if "shipped" in s and "unbanded" in s]
        cells = [(s["unbanded"]["cells_cells"], s["shipped"]["cells_cells"])
                 for s in sp.values() if "shipped" in s and "unbanded" in s]
        line = [ds, str(nb), str(len(sp)), mse(bvn),
                " / ".join(f"{a}→{b}" for a, b in cells)]
        for v in variants:
            d = [s[v]["d_test_vs_ship_pct"] for s in sp.values() if v in s]
            cx = [s[v]["cells_cells"] / s["shipped"]["cells_cells"] for s in sp.values()
                  if v in s and "shipped" in s]
            pooled[f"{v}@{nb}"].extend(d)
            bw = f"{sum(x < -1e-9 for x in d)}/{sum(x > 1e-9 for x in d)}"
            line.append(f"{mse(d)} ({bw}) ×{np.mean(cx):.2f}" if d else "-")
        print("| " + " | ".join(line) + " |")
    print()
    for k, v in sorted(pooled.items()):
        print(f"pooled {k}: {mse(v)} over {len(v)} fits, better/worse "
              f"{sum(x < -1e-9 for x in v)}/{sum(x > 1e-9 for x in v)}")


def dec() -> None:
    rows = load("band_exp.jsonl", ("dataset", "split", "n_bags", "variant"))
    print("| dataset | split | bags | variant | test Δ% | 1st | joint 2nd | diag 2nd | common | train Δ% | train 1st | per-class test 1st |")
    print("|---" * 12 + "|")
    for k in sorted(rows, key=lambda k: (k[0], k[2], k[1], k[3])):
        r = rows[k]
        if k[3].startswith("_") or k[3] == "unbanded":
            continue
        dt, dr = r["dec_test"], r["dec_train"]
        print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {dt['realized']:+.3f} | {dt['first']:+.3f} | "
              f"{dt['joint2']:.3f} | {dt['diag2']:.3f} | {dt['common2']:.3f} | {dr['realized']:+.3f} | "
              f"{dr['first']:+.3f} | {', '.join(f'{x:+.3f}' for x in dt['first_k'])} |")


def grad() -> None:
    rows = load("grad_exp.jsonl", ("dataset", "split", "n_bags", "base", "rule", "tag"))
    by: dict[tuple, list[dict]] = defaultdict(list)
    for k, r in rows.items():
        by[(k[0], k[2], k[3], k[4], k[5] or "")].append(r)
    print("| dataset | bags | base | rule | splits | ungated test % | train ratio | brake adopts | gated test % | smoothed | ceiling |")
    print("|---" * 11 + "|")
    for k in sorted(by):
        rs = by[k]
        d = [r["d_test_pct"] for r in rs]
        g = [r["d_test_pct"] if r["adopted_by_brake"] else 0.0 for r in rs]
        ratios = ", ".join("%.4f" % r["train_ratio"] for r in rs)
        adopted = sum(r["adopted_by_brake"] for r in rs)
        print(f"| {k[0]}{('/' + k[4]) if k[4] else ''} | {k[1]} | {k[2]} | {k[3]} | {len(rs)} | "
              f"{mse(d)} | {ratios} | {adopted}/{len(rs)} | {mse(g)} | "
              f"{np.mean([r['n_smoothed'] for r in rs]):.0f} | "
              f"{np.mean([r.get('n_ceiling', 0) for r in rs]):.0f} |")


def es_rounds() -> dict[tuple, list[tuple[int, int]]]:
    """Per-bag (trained, best) rounds of every fitted stage, from the queue logs."""
    import re
    out: dict[tuple, list[tuple[int, int]]] = {}
    for f in sorted(RES.glob("queue_*.log")):
        cur = None
        for line in f.read_text().splitlines():
            m = re.match(r"=== \S+ (band|grad|es) (\S+) (\d+) (\d+)(.*)", line)
            if m:
                kind, ds, sp, nb, rest = m.groups()
                cur = ((ds, int(sp), int(nb), "nt12k" if "nt12k" in rest else "")
                       if kind == "band" else None)
                continue
            m = re.match(r"\[mc-es\] trained_rounds=(\d+) best_round=(\d+)", line)
            if m and cur:
                out.setdefault(cur, []).append((int(m.group(1)), int(m.group(2))))
    # standalone runs outside the queues (same seed and settings as the cached stages)
    for f, key in (("time_fr_instal.log", ("fr_instal", 0, 8, "")),
                   ("time_pg16_fleet.log", ("pg16_fleet", 0, 8, "")),
                   ("logs_band_fr_instal_s0_b8_nt12k.txt", ("fr_instal", 0, 8, "nt12k"))):
        p = RES / f
        if p.exists() and key not in out:
            for line in p.read_text(errors="replace").splitlines():
                m = re.match(r"\[mc-es\] trained_rounds=(\d+) best_round=(\d+)", line)
                if m:
                    out.setdefault(key, []).append((int(m.group(1)), int(m.group(2))))
    return out


def facts() -> None:
    """Default-setting facts per (dataset, split): test CE of the four band x graduation
    combinations at 8 bags (graduation gated by the shipped 1% brake), cells, and banding at one
    bag. Changes are % of the band-off, graduation-off test CE."""
    band_rows = load("band_exp.jsonl", ("dataset", "split", "n_bags", "variant", "tag"))
    grad_rows = load("grad_exp.jsonl", ("dataset", "split", "n_bags", "base", "rule", "tag"))
    es = es_rounds()
    print("| dataset | split | none (test CE) | band only | band off, grad as shipped (brake-gated) | "
          "default = band + brake-gated grad | "
          "grad ungated on banded / unbanded (train ratio) | cells none → banded | "
          "1 bag: band vs none | 1 bag cells | best rounds (8 bags) |")
    print("|---" * 11 + "|")
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    keys = sorted({(k[0], k[1]) for k in band_rows if k[2] == 8 and not k[4]})
    for ds, sp in keys:
        u = band_rows.get((ds, sp, 8, "unbanded", ""))
        b = band_rows.get((ds, sp, 8, "shipped", ""))
        if not (u and b):
            continue
        base = u["test"]
        pc = lambda x: 100 * (x - base) / base  # noqa: E731
        gb = grad_rows.get((ds, sp, 8, "shipped", "shipped", "")) or grad_rows.get(
            (ds, sp, 8, "shipped", "shipped", None))
        gu = grad_rows.get((ds, sp, 8, "unbanded", "shipped", "")) or grad_rows.get(
            (ds, sp, 8, "unbanded", "shipped", None))
        band = pc(b["test"])
        cells = f"{u['cells_cells']} → {b['cells_cells']}"
        line = [ds, str(sp), f"{base:.6f}", f"{band:+.3f}%"]
        agg[ds]["band"].append(band)
        if gu:
            g_only = pc(gu["test"]) if gu["adopted_by_brake"] else 0.0
            line.append(f"{g_only:+.3f}%{'' if gu['adopted_by_brake'] else ' (declined)'}")
            agg[ds]["grad"].append(g_only)
        else:
            line.append("-")
        if gb:
            default = pc(gb["test"]) if gb["adopted_by_brake"] else band
            line.append(f"{default:+.3f}%{'' if gb['adopted_by_brake'] else ' (grad declined)'}")
            agg[ds]["default"].append(default)
            line.append(f"{gb['d_test_pct']:+.3f}% (×{gb['train_ratio']:.4f}) / "
                        + (f"{gu['d_test_pct']:+.3f}% (×{gu['train_ratio']:.4f})" if gu else "-"))
            agg[ds]["grad_ungated_banded"].append(gb["d_test_pct"])
            if gu:
                agg[ds]["grad_ungated_unbanded"].append(gu["d_test_pct"])
        else:
            line += ["-", "-"]
        line.append(cells)
        u1 = band_rows.get((ds, sp, 1, "unbanded", ""))
        b1 = band_rows.get((ds, sp, 1, "shipped", ""))
        if u1 and b1:
            d1 = 100 * (b1["test"] - u1["test"]) / u1["test"]
            agg[ds]["band1"].append(d1)
            line += [f"{d1:+.3f}%", f"{u1['cells_cells']} → {b1['cells_cells']}"]
        else:
            line += ["-", "-"]
        r = es.get((ds, sp, 8, ""))
        line.append(f"{min(x for _, x in r)}–{max(x for _, x in r)} of {max(t for t, _ in r)}"
                    if r else "-")
        print("| " + " | ".join(line) + " |")
    print()
    print("| dataset | splits | band only | band off, grad gated | default (band + gated grad) | "
          "grad ungated on banded | "
          "grad ungated on unbanded | band at 1 bag |")
    print("|---" * 8 + "|")
    for ds, a in agg.items():
        print(f"| {ds} | {len(a['band'])} | {mse(a['band'])} | {mse(a['grad'])} | "
              f"{mse(a['default'])} | {mse(a['grad_ungated_banded'])} | "
              f"{mse(a['grad_ungated_unbanded'])} | {mse(a['band1'])} |")


def conv() -> None:
    """Convergence of each fitted soup against banding's held-out cost and its first-order share."""
    band_rows = load("band_exp.jsonl", ("dataset", "split", "n_bags", "variant", "tag"))
    es = es_rounds()
    print("| dataset | split | trees | bags at cap | best rounds | unbanded test CE | banding test % | "
          "first-order / second-order (test) |")
    print("|---" * 8 + "|")
    for (ds, sp, nb, tag), r in sorted(es.items()):
        if nb != 8:
            continue
        u = band_rows.get((ds, sp, 8, "unbanded", tag))
        b = band_rows.get((ds, sp, 8, "shipped", tag))
        if not (u and b):
            continue
        cap = 12000 if tag == "nt12k" else 4000
        at_cap = sum(1 for t, _ in r if t >= cap)
        d = b["dec_test"]
        ratio = d["first"] / d["joint2"] if d["joint2"] > 1e-9 else float("nan")
        print(f"| {ds} | {sp} | {cap} | {at_cap}/{len(r)} | {min(x for _, x in r)}–"
              f"{max(x for _, x in r)} | {u['test']:.6f} | {d['realized']:+.3f} | "
              f"{d['first']:+.3f} / {d['joint2']:.3f} ({ratio:.1f}x) |")


def budgets() -> None:
    """Which budget binds per class in the shipped banding: noise (tolerance x sigma)^2 or the
    class's share of the deviance cap; the ratio noise/cap (>1: the cap binds)."""
    rows = load("band_exp.jsonl", ("dataset", "split", "n_bags", "variant", "tag"))
    print("| dataset | split | bags | noise / cap budget per class | binds | cells after per class |")
    print("|---|---|---|---|---|---|")
    for k in sorted(rows):
        r = rows[k]
        if k[3] != "shipped" or not r.get("band"):
            continue
        ratios, binds = [], []
        for b in r["band"]:
            nb, db = b.get("noise_budget"), b.get("deviance_budget")
            if nb is None or db is None or not db:
                ratios.append("inf")
                binds.append("cap")
                continue
            ratios.append(f"{nb / db:.2f}")
            binds.append("cap" if db <= nb else "noise")
        cells = [b.get("cells_after") for b in r["band"]]
        tag = f"/{k[4]}" if k[4] else ""
        print(f"| {k[0]}{tag} | {k[1]} | {k[2]} | {', '.join(ratios)} | {', '.join(binds)} | {cells} |")


def side() -> None:
    """The side analyses, one block each."""
    def lines(fname: str) -> list[dict]:
        p = RES / fname
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    print("### Banding's held-out first-order term by table order (test rows, % of CE)\n")
    print("| dataset | split | banding | intercept | mains | pairs | triples |")
    print("|---|---|---|---|---|---|---|")
    for r in lines("order_split.jsonl"):
        f = r["first_by_order"]
        print(f"| {r['dataset']} | {r['split']} | {r['banding']} | {f[0]:+.3f} | {f[1]:+.3f} | "
              f"{f[2]:+.3f} | {f[3]:+.3f} |")
    print("\n### Out-of-bag estimate of banding's first-order term vs test and training\n")
    for r in lines("oob_first.jsonl"):
        for k, v in r.items():
            if isinstance(v, dict):
                print(f"- {r['dataset']} s{r['split']} {k}: OOB {v['first_oob']:+.3f}, test "
                      f"{v['first_test']:+.3f}, train {v['first_train']:+.3f}")
    print("\n### GCV's implied cell noise vs the between-bag noise (unbanded bank)\n")
    for r in lines("gcv_noise.jsonl"):
        print(f"- {r['dataset']} s{r['split']}: {r['tables_matched']} tables matched, GCV smoothed "
              f"{r['tables_smoothed_gcv']}; median sigma2_gcv / sigma2_bag on smoothed tables "
              f"{r['median_ratio_smoothed']:.2f} (all tables {r['median_ratio_all']:.2f}); "
              f"Cp at the bag noise: " + ", ".join(
                  f"{k} {r[k]['n']} tables {r[k]['d_test_pct']:+.3f}% (x{r[k]['train_ratio']:.4f})"
                  for k in ("cp_bag", "cp_bag4", "cp_bag16")))
    print("\n### Graduation per table (each smoothed table applied alone, unbanded bank)\n")
    for r in lines("per_table.jsonl"):
        t = r["tables"]
        ceil = sum(e["d_test_pct"] for e in t if e["ceiling"])
        print(f"- {r['dataset']} s{r['split']} {r['base']}: {len(t)} tables, sum {r['sum_d_test_pct']:+.3f}% "
              f"(ceiling tables {ceil:+.3f}%). Worst: " + "; ".join(
                  f"class {e['k']} {'x'.join(e['u'])} alpha {e['alpha']:g} {e['d_test_pct']:+.3f}%"
                  for e in t[:4]))
    print("\n### Null-space refusal rule on single-output fits (as shipped, brake on)\n")
    rows = lines("null_single.jsonl")
    for r in rows:
        print(f"- {r['dataset']} s{r['split']}: {r['d_test_pct']:+.3f}% (tables "
              f"{r['shipped']['n_smoothed']} -> {r['null_refuse']['n_smoothed']})")
    if rows:
        d = [r["d_test_pct"] for r in rows]
        print(f"- mean {np.mean(d):+.4f}% over {len(d)} fits, better/worse "
              f"{sum(x < -1e-9 for x in d)}/{sum(x > 1e-9 for x in d)}")
    print("\n### Convergence of single-output and one-vs-rest fits (kept trees per bag)\n")
    for r in lines("es_check.jsonl") + lines("ovr_check.jsonl"):
        lab = r["dataset"] + (f" class {r['ovr_class']} vs rest" if "ovr_class" in r else "")
        print(f"- {lab} s{r['split']}: {r['n_trees_per_bag']} ({sorted(set(r['stop']))})")
    print("\n### Row-level paired SEs (% of the reference test CE; mean, SE)\n")
    for r in lines("paired_se.jsonl"):
        parts = [f"{k} {v[0]:+.3f} ± {v[1]:.3f}" for k, v in r.items()
                 if isinstance(v, list) and abs(v[0]) + abs(v[1]) > 0]
        print(f"- {r['dataset']} s{r['split']}: " + "; ".join(parts))


def write_all(name: str) -> None:
    """Every summary into one markdown file in the results directory."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for title, fn in (("Facts (default settings)", facts), ("Convergence", conv),
                          ("Banding variants", band), ("Banding decomposition", dec),
                          ("Budgets binding in the shipped banding", budgets),
                          ("Graduation rules", grad), ("Side analyses", side)):
            print(f"## {title}\n")
            fn()
            print()
    (RES / name).write_text("# PF-06 / PF-07 summaries (generated by summarize.py)\n\n" + buf.getvalue())


if __name__ == "__main__":
    if sys.argv[1] == "all":
        write_all(sys.argv[2] if len(sys.argv) > 2 else "summary.md")
    else:
        {"band": band, "dec": dec, "grad": grad, "facts": facts, "conv": conv,
         "side": side, "budgets": budgets}[sys.argv[1]]()
