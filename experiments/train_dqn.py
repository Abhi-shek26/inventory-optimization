"""Train the DQN agent.

The agent is given the state (inventory, period), the feasible order range and
the resulting cost -- nothing else.  It never sees the demand distribution, only
realised demands, which is the whole point of the RL comparison.

Runtime note: this project runs on CPU.  `torch.set_num_threads(1)` is faster
for a network this small (thread synchronisation costs more than the matmuls),
and the gradient step is taken every `train_every` environment steps.
"""

from __future__ import annotations

import os
import random
import time

import numpy as np
import torch

from algorithms.dqn_agent import DQNAgent
from config import Config
from environment import InventoryEnv
from utils.visualization import plot_training

RESULTS = "results"


def train(config=None, episodes=None, train_every=2, log_every=250):
    cfg = config or Config()
    episodes = episodes or cfg.episodes
    torch.set_num_threads(1)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = InventoryEnv(cfg, seed=cfg.seed)
    agent = DQNAgent(cfg)
    episode_rewards = []
    t0 = time.perf_counter()

    for episode in range(episodes):
        state = env.reset()
        done, total_reward, step = False, 0.0, 0
        while not done:
            valid = env.valid_actions()
            action = agent.act(state, valid, explore=True)
            next_state, reward, done, _ = env.step(action)
            next_valid = env.valid_actions() if not done else [0]
            agent.remember(state, action, reward, next_state, done, next_valid)
            step += 1
            if step % train_every == 0:
                agent.learn()
            state = next_state
            total_reward += reward
        episode_rewards.append(total_reward)
        if (episode + 1) % log_every == 0:
            elapsed = time.perf_counter() - t0
            print(f"episode {episode + 1:5d}/{episodes} | "
                  f"cost (last 100) {-np.mean(episode_rewards[-100:]):8.1f} | "
                  f"epsilon {agent.epsilon():.3f} | {elapsed:6.1f}s", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    agent.save(os.path.join(RESULTS, "dqn_model.pt"))
    # The simulator returns rewards (negative cost); the log stores the positive
    # episode cost so that the column name and the sign agree for the reader.
    episode_costs = [-r for r in episode_rewards]
    np.savetxt(os.path.join(RESULTS, "training_rewards.csv"),
               episode_costs, delimiter=",", header="cost", comments="")
    plot_training(episode_costs, os.path.join(RESULTS, "dqn_training.png"))
    print(f"training finished in {time.perf_counter() - t0:.1f}s")
    return agent, episode_rewards


if __name__ == "__main__":
    train()
