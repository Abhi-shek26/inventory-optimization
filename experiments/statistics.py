"""Paired statistical comparison of every method against DP.

Why this exists: mean costs on shared demand paths are *paired* observations, so
the right question is not "is 513.9 bigger than 487.8?" but "is the per-episode
difference consistently positive, and how large could it plausibly be?".

For each scenario and each method we report the paired difference versus DP
(method - DP, so positive = worse than DP), its 95% confidence interval and the
paired t-test p-value, plus the cost components that explain a difference.

Writes results/paired_tests.json and results/paired_tests.csv.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = "results"
BASELINE = "DP"


def _paired(a, b):
    """Paired comparison of method `a` against baseline `b`."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diff)
    mean = float(diff.mean())
    if n < 2 or np.allclose(diff, diff[0]):
        return {"mean_diff": mean, "pct": 0.0, "ci_low": mean, "ci_high": mean,
                "p_value": 1.0 if np.allclose(diff, 0) else 0.0,
                "wins": int((diff < 0).sum()), "n": int(n)}
    sem = stats.sem(diff)
    lo, hi = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
    p_value = float(stats.ttest_rel(a, b).pvalue)
    return {"mean_diff": mean, "pct": 100.0 * mean / float(np.mean(b)),
            "ci_low": float(lo), "ci_high": float(hi), "p_value": p_value,
            "wins": int((diff < 0).sum()), "n": int(n)}


def _verdict(result):
    """Plain-language reading of the interval."""
    if result["ci_low"] <= 0.0 <= result["ci_high"]:
        return "indistinguishable from DP"
    return "worse than DP" if result["mean_diff"] > 0 else "better than DP"


def run():
    frame = pd.read_csv(os.path.join(RESULTS, "evaluation_episodes.csv"))
    methods = [m for m in frame["method"].unique() if m != BASELINE]
    scenarios = list(frame["scenario"].unique())

    payload = {"baseline": BASELINE, "methods": methods,
               "scenarios": {sc: {} for sc in scenarios},
               "components": {}, "episodes": {}}

    rows = []
    for sc in scenarios:
        subset = frame[frame["scenario"] == sc]
        wide = subset.pivot(index="episode", columns="method", values="total_cost")
        payload["episodes"][sc] = int(len(wide))
        for method in methods:
            res = _paired(wide[method], wide[BASELINE])
            res["verdict"] = _verdict(res)
            payload["scenarios"][sc][method] = res
            rows.append({"scenario": sc, "method": method, **res})

    # Component-level explanation on the stationary scenario.
    base = frame[(frame["scenario"] == scenarios[0])]
    comp = {}
    for method in [BASELINE] + methods:
        g = base[base["method"] == method]
        comp[method] = {k: float(g[k].mean()) for k in
                        ("total_cost", "holding_cost", "shortage_cost",
                         "ordering_cost", "service_level", "avg_inventory",
                         "compute_time_s")}
        comp[method]["service_level_pct"] = 100.0 * comp[method]["service_level"]
        comp[method]["compute_time_ms"] = 1000.0 * comp[method]["compute_time_s"]
    payload["components"] = comp
    payload["component_deltas"] = {
        method: {k: comp[method][k] - comp[BASELINE][k]
                 for k in ("holding_cost", "shortage_cost", "ordering_cost", "total_cost")}
        for method in methods}

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "paired_tests.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "paired_tests.csv"), index=False)

    print(f"Paired comparison against {BASELINE} "
          f"({payload['episodes'][scenarios[0]]} episodes per scenario)")
    print(pd.DataFrame(rows)[["scenario", "method", "mean_diff", "pct", "ci_low",
                              "ci_high", "p_value", "wins", "verdict"]]
          .round(3).to_string(index=False))
    print("\nCost components, Exp 1 (mean per episode)")
    print(pd.DataFrame({
        m: {"total": comp[m]["total_cost"], "holding": comp[m]["holding_cost"],
            "shortage": comp[m]["shortage_cost"], "ordering": comp[m]["ordering_cost"],
            "service %": comp[m]["service_level_pct"],
            "time ms": comp[m]["compute_time_ms"]}
        for m in comp}).T.round(2).to_string())
    return payload


if __name__ == "__main__":
    run()
