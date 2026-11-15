"""Find the best (s, S) policy by grid search and save it.

Run from the repo root:
    python -m experiments.tune_ss

Writes results/ss_params.json, results/ss_grid.csv and results/ss_heatmap.png
"""

import json
import os

import numpy as np
import pandas as pd

from algorithms.ss_policy import SSPolicy, tune_ss
from config import Config
from environment import InventoryEnv
from utils.simulation import run_episode

RESULTS = "results"


def main():
    cfg = Config()
    env = InventoryEnv(cfg, seed=cfg.seed)
    os.makedirs(RESULTS, exist_ok=True)

    s, S, rows, demands = tune_ss(cfg, env)
    grid = pd.DataFrame(rows)
    grid.to_csv(os.path.join(RESULTS, "ss_grid.csv"), index=False)
    best_cost = grid.loc[(grid.s == s) & (grid.S == S), "mean_cost"].iloc[0]
    print(f"Best (s, S) = ({s}, {S}), mean training cost = {best_cost:.1f}")

    # Cross-check the fast numpy simulator against the shared run_episode.
    policy = SSPolicy(s, S)
    shared = np.mean([run_episode(policy, demands[i], cfg)["total_cost"] for i in range(200)])
    from algorithms.ss_policy import _simulate_costs
    fast = _simulate_costs(cfg, s, S, demands[:200]).mean()
    print(f"Cross-check on 200 paths: shared simulator = {shared:.2f}, fast simulator = {fast:.2f}")
    assert abs(shared - fast) < 1e-6, "fast tuning simulator disagrees with the shared one!"

    with open(os.path.join(RESULTS, "ss_params.json"), "w") as f:
        json.dump({"s": int(s), "S": int(S), "training_mean_cost": float(best_cost)}, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pivot = grid.pivot(index="s", columns="S", values="mean_cost")
        plt.figure(figsize=(7, 5))
        plt.imshow(pivot.values, origin="lower", aspect="auto",
                   extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
                   vmax=np.nanpercentile(pivot.values, 60))
        plt.colorbar(label="mean cost per episode")
        plt.scatter([S], [s], color="red", marker="x", s=80, label=f"best ({s}, {S})")
        plt.xlabel("S (order-up-to level)")
        plt.ylabel("s (reorder point)")
        plt.title("(s, S) grid search")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS, "ss_heatmap.png"), dpi=150)
        print("Saved results/ss_heatmap.png")
    except ImportError:
        print("matplotlib not installed - skipped the heatmap")


if __name__ == "__main__":
    main()
