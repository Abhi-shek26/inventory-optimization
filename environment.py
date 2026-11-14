import numpy as np
import math

class InventoryEnv:
    """Single-item, periodic-review inventory system with lost sales.

    At each period: choose order quantity, demand is realized, unmet demand
    is lost, and leftover inventory incurs holding cost.  Placing a non-zero
    order also triggers the fixed ordering cost K (the setup decision y_t).
    """
    def __init__(self, config, demand_pmf=None, seed=None):
        self.cfg = config
        self.values = np.asarray(config.demand_values, dtype=int)
        if demand_pmf is None:
            weights = np.exp(-config.demand_lambda) * np.array(
                [config.demand_lambda ** int(d) / math.factorial(int(d))
                 for d in self.values], dtype=float)
            demand_pmf = weights / weights.sum()
        self.pmf = np.asarray(demand_pmf, dtype=float)
        if len(self.pmf) != len(self.values) or np.any(self.pmf < 0):
            raise ValueError("Invalid demand probability mass function.")
        self.pmf = self.pmf / self.pmf.sum()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.inventory = 0
        return self._state()

    def valid_actions(self):
        return np.arange(min(self.cfg.max_order,
                             self.cfg.max_inventory - self.inventory) + 1)

    def _state(self):
        return np.array([self.inventory / self.cfg.max_inventory,
                         self.t / self.cfg.horizon], dtype=np.float32)

    def step(self, action, demand=None):
        action = int(action)
        if action not in self.valid_actions():
            raise ValueError(f"Invalid order {action}; valid actions: {self.valid_actions()}")
        if demand is None:
            demand = int(self.rng.choice(self.values, p=self.pmf))
        available = self.inventory + action
        sales = min(available, int(demand))
        lost_sales = int(demand) - sales
        ending_inventory = available - sales
        # Fixed ordering cost K is paid once per period in which we order,
        # i.e. whenever the binary setup decision y_t is 1.
        fixed = self.cfg.fixed_order_cost if action > 0 else 0.0
        variable = self.cfg.order_cost * action
        ordering = fixed + variable
        holding = self.cfg.holding_cost * ending_inventory
        shortage = self.cfg.shortage_cost * lost_sales
        cost = ordering + holding + shortage
        self.inventory = ending_inventory
        self.t += 1
        done = self.t >= self.cfg.horizon
        info = {"demand": int(demand), "sales": sales, "lost_sales": lost_sales,
                "ordering_cost": ordering, "fixed_cost": fixed,
                "variable_cost": variable, "holding_cost": holding,
                "shortage_cost": shortage, "cost": cost,
                "ending_inventory": ending_inventory}
        return self._state(), -float(cost), done, info
