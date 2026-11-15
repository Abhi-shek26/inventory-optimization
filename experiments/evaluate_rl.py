"""Evaluate (s,S), Q-learning and DQN against DP on the SHARED demand paths.

Run from the repo root (after tune_ss, train_q and/or train_dqn):
    python -m experiments.evaluate_rl
    python -m experiments.evaluate_rl --episodes 100

Uses the same evaluation seeds as the shared evaluate.py (seed + 10000 + episode)
and the shared run_episode(), so the numbers are comparable with her table.
It edits no shared file.  Whichever of results/ss_params.json, results/q_table.npy and
results/dqn_model.pt exist get evaluated.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from algorithms.dynamic_programming import DynamicProgrammingSolver
from config import Config
from environment import InventoryEnv
from experiments.demand_scenarios import SCENARIOS, SCENARIO_LABELS
from utils.simulation import run_episode

RESULTS = "results"


def build_policies(cfg, env):
    dp = DynamicProgrammingSolver(cfg, env.values, env.pmf)
    dp.solve()
    policies = {"DP": lambda t, inv: dp.act(t, inv)}

    ss_path = os.path.join(RESULTS, "ss_params.json")
    if os.path.exists(ss_path):
        from algorithms.ss_policy import SSPolicy
        params = json.load(open(ss_path))
        policies["(s,S)"] = SSPolicy(params["s"], params["S"])
    else:
        print("[skip] (s,S): results/ss_params.json not found (run experiments.tune_ss)")

    q_path = os.path.join(RESULTS, "q_table.npy")
    if os.path.exists(q_path):
        from algorithms.tabular_q_agent import TabularQAgent
        q_agent = TabularQAgent(cfg)
        q_agent.load(q_path)
        policies["Q-learning"] = q_agent.policy()
    else:
        print("[skip] Q-learning: results/q_table.npy not found")

    dqn_path = os.path.join(RESULTS, "dqn_model.pt")
    if os.path.exists(dqn_path):
        from algorithms.dqn_agent import DQNAgent
        dqn = DQNAgent(cfg)
        dqn.load(dqn_path)

        def dqn_policy(t, inv):
            state = np.array([inv / cfg.max_inventory, t / cfg.horizon], dtype=np.float32)
            return dqn.act(state, range(min(cfg.max_order, cfg.max_inventory - inv) + 1))
        policies["DQN"] = dqn_policy
    else:
        print("[skip] DQN: results/dqn_model.pt not found")
    return policies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=None,
                        help="evaluation paths per scenario (default: cfg.eval_episodes = 40)")
    args = parser.parse_args()

    cfg = Config()
    n = args.episodes or cfg.eval_episodes
    env = InventoryEnv(cfg, seed=cfg.seed)
    policies = build_policies(cfg, env)

    episode_rows, summary = [], []
    for name, builder in SCENARIOS.items():
        costs = {m: [] for m in policies}
        visited = set()
        for ep in range(n):
            rng = np.random.default_rng(cfg.seed + 10000 + ep)   # same seeds as evaluate.py
            path, _ = builder(rng, cfg, env.values, env.pmf)
            for method, policy in policies.items():
                out = run_episode(policy, path, cfg)
                costs[method].append(out["total_cost"])
                episode_rows.append({"scenario": name, "episode": ep,
                                     "method": method, "total_cost": out["total_cost"]})
                if name == "Exp1_stationary" and method == "DP":
                    visited.update((p["t"], p["inventory_start"]) for p in out["periods"])

        dp_costs = np.array(costs["DP"])
        for method in policies:
            c = np.array(costs[method])
            diff = c - dp_costs                          # paired: same demand paths
            half = 1.96 * diff.std(ddof=1) / np.sqrt(n) if method != "DP" else 0.0
            row = {"scenario": SCENARIO_LABELS[name], "method": method,
                   "mean cost": round(c.mean(), 1), "std": round(c.std(ddof=1), 1),
                   "vs DP": round(diff.mean(), 1), "95% CI": f"+/-{half:.1f}",
                   "vs DP %": round(100 * diff.mean() / dp_costs.mean(), 2)}
            if name == "Exp1_stationary" and method != "DP":
                same = np.mean([policies[method](t, i) == policies["DP"](t, i) for t, i in visited])
                row["same order as DP"] = f"{100 * same:.0f}%"
            summary.append(row)

    table = pd.DataFrame(summary)
    print(table.to_string(index=False))
    os.makedirs(RESULTS, exist_ok=True)
    table.to_csv(os.path.join(RESULTS, "rl_vs_dp_summary.csv"), index=False)
    pd.DataFrame(episode_rows).to_csv(os.path.join(RESULTS, "rl_vs_dp_episodes.csv"), index=False)
    print("\nSaved results/rl_vs_dp_summary.csv and rl_vs_dp_episodes.csv")
    print("'same order as DP' = share of the (period, stock) situations DP visits in Exp 1.")


if __name__ == "__main__":
    main()
