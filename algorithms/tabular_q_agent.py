"""Tabular Q-learning agent for the shared inventory problem.

Idea in one paragraph
---------------------
Keep a table Q[t, stock, order] = "expected total future reward (= minus cost)
if, in period t with `stock` units on the shelf, I order `order` units and then
behave well afterwards".  Play many simulated episodes.  After every period, nudge
the table entry for the (t, stock, order) just used towards

    reward_now + best Q-value of the next situation        (the "target")

After enough episodes, picking the order with the highest Q-value in each
situation gives a good policy.  Nothing about the demand distribution is given
to the agent: it only ever sees the stock level, the period, and the cost it
paid.  (The simulator samples the demand behind the scenes.)

This file only decides orders; the physics and costs come from the shared
environment.py, exactly as the team-sync rules require.
"""

import numpy as np


class TabularQAgent:
    def __init__(self, cfg, gamma=1.0):
        # gamma = 1.0: no discounting, so the agent optimises the same
        # undiscounted total cost that the MILP, DP and the evaluation use.
        self.cfg = cfg
        self.gamma = gamma
        T = cfg.horizon
        n_stock = cfg.max_inventory + 1          # stock 0..Imax
        n_orders = cfg.max_order + 1             # order 0..Qmax
        self.Q = np.zeros((T, n_stock, n_orders))
        self.N = np.zeros((T, n_stock, n_orders), dtype=np.int64)  # visit counts

    # ------------------------------------------------------------------
    def max_valid_order(self, stock):
        """Same action mask as the shared harness: min(Qmax, Imax - stock)."""
        return min(self.cfg.max_order, self.cfg.max_inventory - int(stock))

    def act(self, t, stock, epsilon=0.0, rng=None):
        """Pick an order.  With probability epsilon explore (random valid order)."""
        hi = self.max_valid_order(stock)
        if epsilon > 0.0 and rng.random() < epsilon:
            return int(rng.integers(0, hi + 1))
        q = self.Q[t, int(stock), : hi + 1]
        best = np.flatnonzero(q == q.max())      # all orders tied for best
        if rng is not None and len(best) > 1:
            return int(rng.choice(best))         # break ties randomly in training
        return int(best[0])                      # ...and deterministically at test time

    def policy(self):
        """Return policy(t, stock) -> order, the interface the shared harness uses."""
        return lambda t, stock: self.act(min(t, self.cfg.horizon - 1), stock)

    # ------------------------------------------------------------------
    def train(self, env, episodes, seed, eps_start=1.0, eps_end=0.05,
              decay_fraction=0.6, log_every=10000):
        """Learn by trial and error in `env` (an InventoryEnv).

        env.step(order, demand=d) is the shared physics.  We sample the demand
        ourselves (batch per episode, from the environment's own demand pmf) so
        training is reproducible and fast.
        """
        rng = np.random.default_rng(seed)
        T = self.cfg.horizon
        costs = np.zeros(episodes)

        for ep in range(episodes):
            # epsilon: explore a lot early, mostly exploit later
            frac = min(1.0, ep / (decay_fraction * episodes))
            epsilon = eps_start + (eps_end - eps_start) * frac

            demands = rng.choice(env.values, size=T, p=env.pmf)
            env.reset()
            total_cost = 0.0

            for t in range(T):
                stock = env.inventory
                order = self.act(t, stock, epsilon, rng)
                _, reward, done, _ = env.step(order, demand=int(demands[t]))
                next_stock = env.inventory

                if done:                                  # last period: no future
                    target = reward
                else:
                    hi = self.max_valid_order(next_stock)
                    target = reward + self.gamma * self.Q[t + 1, next_stock, : hi + 1].max()

                self.N[t, stock, order] += 1
                alpha = max(0.02, self.N[t, stock, order] ** -0.6)  # shrinking step size
                self.Q[t, stock, order] += alpha * (target - self.Q[t, stock, order])
                total_cost -= reward                       # reward = -cost

            costs[ep] = total_cost
            if log_every and (ep + 1) % log_every == 0:
                recent = costs[ep + 1 - log_every: ep + 1].mean()
                print(f"  episode {ep + 1:>7}/{episodes}   eps={epsilon:.2f}   "
                      f"mean training cost (last {log_every}) = {recent:7.1f}")
        return costs

    # ------------------------------------------------------------------
    def save(self, path):
        np.save(path, self.Q)

    def load(self, path):
        self.Q = np.load(path)
