"""Train the DQN agent.

Run from the repo root:
    python -m experiments.train_dqn                    # cfg.episodes (2500) episodes
    python -m experiments.train_dqn --episodes 6000    # longer run
    python -m experiments.train_dqn --gamma 0.98       # try the config's gamma

Writes results/dqn_model.pt (the path the shared evaluate.py looks for),
results/dqn_training_curve.csv and results/dqn_training_curve.png
"""

import argparse
import os
import time

import numpy as np

from algorithms.dqn_agent import DQNAgent
from algorithms.dynamic_programming import DynamicProgrammingSolver
from config import Config
from environment import InventoryEnv

RESULTS = "results"


def main():
    cfg = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=cfg.episodes)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    env = InventoryEnv(cfg, seed=cfg.seed)
    train_seed = cfg.seed + 5_000_000      # far from the evaluation seeds (seed + 10000 + ep)

    agent = DQNAgent(cfg, gamma=args.gamma, hidden=args.hidden)
    print(f"Training DQN for {args.episodes} episodes (gamma={args.gamma}) ...")
    t0 = time.perf_counter()
    costs = agent.train(env, args.episodes, seed=train_seed,
                        log_every=max(1, args.episodes // 10))
    print(f"Done in {time.perf_counter() - t0:.0f} s")
    agent.save(os.path.join(RESULTS, "dqn_model.pt"))

    block = max(1, args.episodes // 100)
    curve = costs[: (len(costs) // block) * block].reshape(-1, block).mean(axis=1)
    np.savetxt(os.path.join(RESULTS, "dqn_training_curve.csv"),
               np.column_stack([np.arange(len(curve)) * block, curve]),
               delimiter=",", header="episode,mean_training_cost", comments="")

    dp_cost, _ = DynamicProgrammingSolver(cfg, env.values, env.pmf).solve()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(np.arange(len(curve)) * block, curve, label="DQN (training, incl. exploration)")
        plt.axhline(dp_cost, color="red", linestyle="--", label=f"DP optimum ({dp_cost:.1f})")
        plt.xlabel("training episode")
        plt.ylabel("mean total cost per episode")
        plt.title("DQN learning curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS, "dqn_training_curve.png"), dpi=150)
        print("Saved results/dqn_training_curve.png")
    except ImportError:
        print("matplotlib not installed - skipped the plot")


if __name__ == "__main__":
    main()
