"""Summarise a pf05_ab.py JSONL file: paired test-deviance change of each arm against arm A."""
import collections
import json
import statistics as st
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
base = sys.argv[2] if len(sys.argv) > 2 else "A"
by = collections.defaultdict(dict)
for r in rows:
    by[(r["dataset"], r["split"])][r["arm"]] = r
arms = sorted({r["arm"] for r in rows} - {base})
per = collections.defaultdict(lambda: collections.defaultdict(list))
print(f"{'dataset':12s} {'s':>2s} {base + ' test':>10s} " + " ".join(f"{a + ' %':>8s}" for a in arms)
      + "   kept " + "/".join([base] + arms) + "   excluded(" + base + ")")
for (d, s), v in sorted(by.items()):
    if base not in v:
        continue
    b = v[base]
    cells = []
    for a in arms:
        if a in v:
            x = 100 * (v[a]["test_dev"] - b["test_dev"]) / b["test_dev"]
            per[a][d].append(x)
            cells.append(f"{x:+8.3f}")
        else:
            cells.append(f"{'-':>8s}")
    kept = "/".join(str(v[a]["kept"]) if a in v else "-" for a in [base] + arms)
    exc = len((b.get("rebalance") or {}).get("excluded") or [])
    print(f"{d:12s} {s:2d} {b['test_dev']:10.6f} " + " ".join(cells) + f"   {kept}   {exc}")
print()
for a in arms:
    print(f"arm {a} vs {base}:")
    means = []
    for d, xs in sorted(per[a].items()):
        means.append(st.mean(xs))
        print(f"   {d:12s} n={len(xs)} mean {st.mean(xs):+.4f}%  better {sum(x < -0.02 for x in xs)}"
              f" worse {sum(x > 0.02 for x in xs)}")
    print(f"   mean of dataset means {st.mean(means):+.4f}%")
