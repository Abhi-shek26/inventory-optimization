"""(s, S) reorder policy: the classic industry baseline for inventory.

Rule:  if stock <= s  ->  order enough to bring stock up to S   (order = S - stock)
       otherwise      ->  order nothing

Two numbers define the whole policy.  The fixed ordering cost K is exactly why
this shape works: don't pay K for a tiny order, wait until stock is low, then
order a big batch.

We pick (s, S) by grid search on SIMULATED TRAINING demand (never on the shared
evaluation paths).  Like the RL agents, the policy never sees the demand
distribution; it only gets scored on samples drawn from it.

Limitation: the numbers are the same in every period, while the exact DP policy
changes near the end of the horizon (e.g. it stops ordering for leftovers).  So
the best (s, S) is expected to sit slightly above DP.
"""

import numpy as np


class SSPolicy:
    def __init__(self, s, S):
        self.s, self.S = int(s), int(S)

    def __call__(self, t, stock):
        # The shared harness clips the order to min(Qmax, Imax - stock).
        return self.S - int(stock) if stock <= self.s else 0


def _simulate_costs(cfg, s, S, demands):
    """Total cost of an (s,S) policy on many demand paths at once (fast, numpy).

    `demands` has shape (n_paths, horizon).  This duplicates the shared cost
    formula only to RANK candidates quickly.  Final reported numbers always come
    from the shared run_episode, and tune_ss() cross-checks the two.
    """
    n_paths, T = demands.shape
    stock = np.zeros(n_paths, dtype=int)
    total = np.zeros(n_paths)
    for t in range(T):
        order = np.where(stock <= s, S - stock, 0)
        order = np.minimum(order, np.minimum(cfg.max_order, cfg.max_inventory - stock))
        available = stock + order
        sales = np.minimum(available, demands[:, t])
        lost = demands[:, t] - sales
        stock = available - sales
        total += (cfg.fixed_order_cost * (order > 0) + cfg.order_cost * order
                  + cfg.holding_cost * stock + cfg.shortage_cost * lost)
    return total


def tune_ss(cfg, env, n_paths=2000, seed=None, max_s=25):
    """Grid-search (s, S); returns (best_s, best_S, table of mean costs)."""
    seed = cfg.seed + 7_000_000 if seed is None else seed   # distinct from eval and RL-training seeds
    rng = np.random.default_rng(seed)
    demands = rng.choice(env.values, size=(n_paths, cfg.horizon), p=env.pmf)

    rows, best = [], (None, None, float("inf"))
    for s in range(0, max_s + 1):
        for S in range(s + 1, cfg.max_inventory + 1):
            cost = float(_simulate_costs(cfg, s, S, demands).mean())
            rows.append({"s": s, "S": S, "mean_cost": cost})
            if cost < best[2]:
                best = (s, S, cost)
    return best[0], best[1], rows, demands
