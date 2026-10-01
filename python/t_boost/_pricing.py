"""Display metadata for pricing review; never used by model scoring."""
from __future__ import annotations

import json
from typing import Any

import numpy as np


def annotate_joint_export(payload: str, measure: str | None) -> str:
    """Describe the regularized hybrid and measure residuals on observed support."""
    if measure != "joint":
        return payload
    export = json.loads(payload)
    # Multiclass exports contain one bank per class. Visit banks without assuming
    # that the surrounding container has the single-output layout.
    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if "tables" in value and "f0" in value:
                residuals = []
                for table in value["tables"]:
                    if len(table["shape"]) != 2:
                        continue
                    shape = tuple(table["shape"])
                    mass = np.asarray(table["support"], dtype=float).reshape(shape)
                    effect = np.asarray(table["values"], dtype=float).reshape(shape)
                    maximum = 0.0
                    for axis in (0, 1):
                        den = mass.sum(axis=axis)
                        means = np.divide((mass * effect).sum(axis=axis), den,
                                          out=np.zeros_like(den), where=den > 0)
                        maximum = max(maximum, float(np.max(np.abs(means), initial=0.0)))
                    residuals.append({"features": table["feature_names"],
                                      "max_abs_observed_conditional_mean": maximum})
                value["joint_reallocation"] = {
                    "kind": "regularized_pairwise_hybrid",
                    "ridge": 0.001, "joint_floor": 0.000001,
                    "exact_conditional_purity": False,
                    "shapley_interpretation": False,
                    "residual_measure": "observed support, without the positivity floor",
                    "pairs": residuals,
                }
            else:
                for item in value.values():
                    visit(item)
    visit(export)
    return json.dumps(export, allow_nan=False)
