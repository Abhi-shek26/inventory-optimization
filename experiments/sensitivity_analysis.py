"""Simple sensitivity analysis for shortage penalty (re-solves DP each time).

For a full study, retrain DQN for each setting rather than reusing a policy.
"""
from dataclasses import replace
import pandas as pd
from config import Config
from environment import InventoryEnv
from algorithms.dynamic_programming import DynamicProgrammingSolver

def run():
    base = Config()
    rows = []
    for penalty in [2.0, 4.0, 8.0, 12.0, 20.0]:
        cfg = replace(base, shortage_cost=penalty)
        env = InventoryEnv(cfg, seed=cfg.seed)
        solver = DynamicProgrammingSolver(cfg, env.values, env.pmf)
        value, _ = solver.solve()
        rows.append({"shortage_cost": penalty, "optimal_expected_cost": value,
                     "initial_order": solver.act(0, 0)})
    df = pd.DataFrame(rows)
    df.to_csv("results/dp_sensitivity.csv", index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run()
