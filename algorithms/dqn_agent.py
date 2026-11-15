"""Deep Q-Network (DQN) agent for the shared inventory problem.

Same idea as the Q-table, with one change: instead of a table that stores one
number per (period, stock, order), a small neural network *predicts* the Q-values
of all orders from the state [stock / Imax, period / T].

The class exposes exactly the interface the team's shared harness expects
(`DQNAgent(cfg)`, `.act(state, valid)`, `.load(path)`), so
`python -m experiments.evaluate` picks it up automatically once
results/dqn_model.pt exists -- no shared file needs editing.

Three extra ingredients make training stable (they are what "DQN" adds):
  1. Replay buffer   - learn from random old transitions, not only the latest one.
  2. Target network  - a slowly updated copy of the network provides the targets.
  3. Action masking  - orders that exceed the warehouse space are never chosen
                       and never used when computing the "best next value".
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Costs per episode are in the hundreds; dividing rewards by this keeps the
# network's targets near 1 and makes the learning rate behave sensibly.
REWARD_SCALE = 100.0


class QNetwork(nn.Module):
    def __init__(self, n_inputs, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size memory of (state, order, reward, next_state, done, next_max_order)."""

    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.hi2 = np.zeros(capacity, dtype=np.int64)   # largest valid order in next state
        self.pos = 0
        self.size = 0

    def add(self, s, a, r, s2, d, hi2):
        i = self.pos
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i], self.hi2[i] = s, a, r, s2, d, hi2
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]), torch.from_numpy(self.s2[idx]),
                torch.from_numpy(self.d[idx]), torch.from_numpy(self.hi2[idx]))


class DQNAgent:
    def __init__(self, cfg, gamma=1.0, hidden=64, seed=None):
        # gamma = 1.0 keeps the learning objective equal to the undiscounted total
        # cost used by the MILP, DP and the evaluation.  (cfg.gamma = 0.98 is the
        # value in the shared config; try it as an experiment.)
        self.cfg = cfg
        self.gamma = gamma
        self.n_actions = cfg.max_order + 1
        seed = cfg.seed if seed is None else seed
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.online = QNetwork(2, self.n_actions, hidden)
        self.target = QNetwork(2, self.n_actions, hidden)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.learning_rate)
        self.buffer = ReplayBuffer(cfg.replay_capacity, 2)
        self.grad_steps = 0

    # ------------------------------------------------------------------
    def max_valid_order(self, stock):
        return min(self.cfg.max_order, self.cfg.max_inventory - int(stock))

    def epsilon(self):
        """Exploration rate, decayed per gradient step (as in the shared config)."""
        c = self.cfg
        frac = min(1.0, self.grad_steps / c.epsilon_decay_steps)
        return c.epsilon_start + (c.epsilon_end - c.epsilon_start) * frac

    def act(self, state, valid, epsilon=0.0):
        """Choose an order.  `valid` = allowed orders (e.g. range(0, hi + 1))."""
        valid = list(valid)
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.choice(valid))
        with torch.no_grad():
            q = self.online(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))[0]
        return int(valid[int(torch.argmax(q[valid]))])

    # ------------------------------------------------------------------
    def update(self):
        """One gradient step on a random mini-batch from the replay buffer."""
        s, a, r, s2, d, hi2 = self.buffer.sample(self.cfg.batch_size, self.rng)

        q_pred = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.target(s2)
            invalid = torch.arange(self.n_actions).unsqueeze(0) > hi2.unsqueeze(1)
            q_next = q_next.masked_fill(invalid, float("-inf"))     # action masking
            best_next = q_next.max(dim=1).values
            target = r + self.gamma * (1.0 - d) * best_next         # no future after last period

        loss = F.smooth_l1_loss(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        self.grad_steps += 1
        if self.grad_steps % self.cfg.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return loss.item()

    def train(self, env, episodes, seed, learning_starts=1000, log_every=250):
        """Learn by trial and error in `env` (the shared InventoryEnv)."""
        rng = np.random.default_rng(seed)
        T = self.cfg.horizon
        costs = np.zeros(episodes)

        for ep in range(episodes):
            demands = rng.choice(env.values, size=T, p=env.pmf)
            state = env.reset()
            total_cost = 0.0
            for t in range(T):
                valid = range(self.max_valid_order(env.inventory) + 1)
                order = self.act(state, valid, self.epsilon())
                next_state, reward, done, _ = env.step(order, demand=int(demands[t]))
                self.buffer.add(state, order, reward / REWARD_SCALE, next_state,
                                float(done), self.max_valid_order(env.inventory))
                if self.buffer.size >= learning_starts:
                    self.update()
                state = next_state
                total_cost -= reward
            costs[ep] = total_cost

            if log_every and (ep + 1) % log_every == 0:
                recent = costs[ep + 1 - log_every: ep + 1].mean()
                print(f"  episode {ep + 1:>6}/{episodes}   eps={self.epsilon():.2f}   "
                      f"mean training cost (last {log_every}) = {recent:7.1f}")
        return costs

    # ------------------------------------------------------------------
    def save(self, path):
        torch.save(self.online.state_dict(), path)

    def load(self, path):
        self.online.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.target.load_state_dict(self.online.state_dict())
        self.online.eval()
