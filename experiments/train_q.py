"""Train the tabular Q-learning agent.

Run from the repo root:
    python -m experiments.train_q                    # default 300k episodes
    python -m experiments.train_q --episodes 50000   # quick test

Writes  results/q_table.npy  and  results/q_training_curve.csv/.png
"""

import argparse
import os
import time

import numpy as np

from algorithms.dynamic_programming import DynamicProgrammingSolver
from algorithms.tabular_q_agent import TabularQAgent
from config import Config
from environment import InventoryEnv

RESULTS = "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300_000)
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()

    cfg = Config()
    os.makedirs(RESULTS, exist_ok=True)

    # The environment is the shared one.  The training seed is far away from the
    # evaluation seeds (cfg.seed + 10000 + episode) so we never train on test paths.
    env = InventoryEnv(cfg, seed=cfg.seed)
    train_seed = cfg.seed + 5_000_000

    agent = TabularQAgent(cfg, gamma=args.gamma)
    print(f"Training for {args.episodes} episodes "
          f"(Q-table: {agent.Q.size} entries) ...")
    t0 = time.perf_counter()
    costs = agent.train(env, args.episodes, seed=train_seed,
                        log_every=max(1, args.episodes // 10))
    print(f"Done in {time.perf_counter() - t0:.0f} s")

    agent.save(os.path.join(RESULTS, "q_table.npy"))

    # Learning curve: average training cost per block of episodes.
    # (Training episodes still include random exploration, so this sits a bit
    #  above the pure greedy cost that evaluate_q.py reports.)
    block = max(1, args.episodes // 100)
    curve = costs[: (len(costs) // block) * block].reshape(-1, block).mean(axis=1)
    np.savetxt(os.path.join(RESULTS, "q_training_curve.csv"),
               np.column_stack([np.arange(len(curve)) * block, curve]),
               delimiter=",", header="episode,mean_training_cost", comments="")

    # Reference line: the DP optimum, from the shared DP solver.
    dp = DynamicProgrammingSolver(cfg, env.values, env.pmf)
    dp_cost, _ = dp.solve()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(np.arange(len(curve)) * block, curve, label="Q-learning (training, incl. exploration)")
        plt.axhline(dp_cost, color="red", linestyle="--", label=f"DP optimum ({dp_cost:.1f})")
        plt.xlabel("training episode")
        plt.ylabel("mean total cost per episode")
        plt.title("Q-learning learning curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS, "q_training_curve.png"), dpi=150)
        print("Saved results/q_training_curve.png")
    except ImportError:
        print("matplotlib not installed - skipped the plot (CSV was still saved)")


if __name__ == "__main__":
    main()
