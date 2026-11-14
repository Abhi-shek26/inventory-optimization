import numpy as np

class DynamicProgrammingSolver:
    """Finite-horizon stochastic DP with known demand probabilities."""
    def __init__(self, config, demand_values, demand_pmf):
        self.cfg = config
        self.values = np.asarray(demand_values, dtype=int)
        self.pmf = np.asarray(demand_pmf, dtype=float)
        self.pmf = self.pmf / self.pmf.sum()
        self.V = None
        self.policy = None

    def solve(self):
        T, Imax = self.cfg.horizon, self.cfg.max_inventory
        V = np.zeros((T + 1, Imax + 1), dtype=float)
        policy = np.zeros((T, Imax + 1), dtype=int)
        for t in range(T - 1, -1, -1):
            for inv in range(Imax + 1):
                max_q = min(self.cfg.max_order, Imax - inv)
                best_cost, best_q = float("inf"), 0
                for q in range(max_q + 1):
                    expected = 0.0
                    available = inv + q
                    for demand, prob in zip(self.values, self.pmf):
                        sales = min(available, int(demand))
                        lost = int(demand) - sales
                        ending = available - sales
                        immediate = ((self.cfg.fixed_order_cost if q > 0 else 0.0)
                                     + self.cfg.order_cost * q
                                     + self.cfg.holding_cost * ending
                                     + self.cfg.shortage_cost * lost)
                        expected += prob * (immediate + V[t + 1, ending])
                    if expected < best_cost:
                        best_cost, best_q = expected, q
                V[t, inv], policy[t, inv] = best_cost, best_q
        self.V, self.policy = V, policy
        return V[0, 0], policy

    def act(self, t, inventory):
        if self.policy is None:
            self.solve()
        return int(self.policy[min(t, self.cfg.horizon - 1), inventory])
