"""Paired statistical comparison of all evaluated methods against Stochastic DP."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = "results"


def run_statistics(episodes_csv=os.path.join(RESULTS, "evaluation_episodes.csv")):
    if not os.path.exists(episodes_csv):
        raise FileNotFoundError(
            f"{episodes_csv} not found. Run `python -m experiments.evaluate` first."
        )

    df = pd.read_csv(episodes_csv)
    rows = []

    for scenario, group in df.groupby("scenario", sort=False):
        pivoted = group.pivot(index="episode", columns="method", values="total_cost")
        if "DP" not in pivoted.columns:
            continue
        dp_costs = pivoted["DP"].to_numpy()
        n = len(dp_costs)

        for method in pivoted.columns:
            if method == "DP":
                continue
            m_costs = pivoted[method].to_numpy()
            diff = m_costs - dp_costs
            mean_diff = float(diff.mean())
            sd_diff = float(diff.std(ddof=1))
            se_diff = sd_diff / np.sqrt(n)
            t_stat, p_val = stats.ttest_rel(m_costs, dp_costs)
            ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean_diff, scale=se_diff)
            pct_diff = 100.0 * mean_diff / float(dp_costs.mean())

            rows.append({
                "scenario": scenario,
                "method": method,
                "n_episodes": int(n),
                "mean_cost": round(float(m_costs.mean()), 2),
                "dp_mean_cost": round(float(dp_costs.mean()), 2),
                "mean_diff_vs_dp": round(mean_diff, 2),
                "pct_diff_vs_dp": round(pct_diff, 2),
                "ci95_low": round(float(ci_low), 2),
                "ci95_high": round(float(ci_high), 2),
                "t_stat": round(float(t_stat), 3),
                "p_value": round(float(p_val), 6),
            })

    out_df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    out_df.to_csv(os.path.join(RESULTS, "paired_tests.csv"), index=False)
    with open(os.path.join(RESULTS, "paired_tests.json"), "w") as fh:
        json.dump(rows, fh, indent=2)

    print(out_df.to_string(index=False))
    print(f"\nSaved paired statistical tests in ./{RESULTS}/paired_tests.csv and .json")
    return out_df


if __name__ == "__main__":
    run_statistics()
