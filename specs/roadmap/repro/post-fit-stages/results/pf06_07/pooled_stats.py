"""Pooled statistics over every banding move in band_exp.jsonl: the first-order share of the
realized test cost, and the joint / diagonal second-order ratio, split by convergence class."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent
rows = {}
for line in (RES / "band_exp.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r["variant"].startswith("_") or r["variant"] == "unbanded":
        continue
    rows[(r["dataset"], r["split"], r["n_bags"], r["variant"], r.get("tag") or "")] = r
truncated = {"fr_instal", "pg16_fleet"}
for label, sel in (("all", lambda k: True),
                   ("truncated soups (fr_instal, pg16_fleet; 4000 trees)",
                    lambda k: k[0] in truncated and k[4] == ""),
                   ("12000-tree soups", lambda k: k[4] == "nt12k"),
                   ("converged soups (pg17_cover, pg17_instal, prudential)",
                    lambda k: k[0] not in truncated)):
    ks = [k for k in rows if sel(k)]
    jr = [rows[k]["dec_test"]["joint2"] / rows[k]["dec_test"]["diag2"] for k in ks
          if rows[k]["dec_test"]["diag2"] > 1e-4]
    fs = [rows[k]["dec_test"]["first"] / rows[k]["dec_test"]["realized"] for k in ks
          if rows[k]["dec_test"]["realized"] > 0.02]
    fr = [rows[k]["dec_test"]["first"] / rows[k]["dec_test"]["joint2"] for k in ks
          if rows[k]["dec_test"]["joint2"] > 1e-3]
    print(f"{label}: {len(ks)} moves")
    if jr:
        print(f"  joint/diag test: median {np.median(jr):.3f}, range {min(jr):.3f}-{max(jr):.3f}")
    if fs:
        print(f"  first-order share of the realized cost (cost > 0.02%): median {np.median(fs):.2f}, "
              f"range {min(fs):.2f}-{max(fs):.2f} over {len(fs)}")
    if fr:
        print(f"  first / joint second-order: median {np.median(fr):.2f}, range {min(fr):.2f}-{max(fr):.2f}")
