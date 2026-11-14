from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass
class Config:
    # --- problem structure -------------------------------------------------
    horizon: int = 20            # T   planning periods
    max_inventory: int = 40      # Imax warehouse capacity (units)
    max_order: int = 30          # Qmax largest order per period (units)
    # --- demand ------------------------------------------------------------
    demand_values: tuple = tuple(range(13))
    demand_lambda: float = 6.0   # Poisson mean before truncation
    # --- costs -------------------------------------------------------------
    fixed_order_cost: float = 25.0   # K  fixed cost paid whenever q_t > 0
    order_cost: float = 2.0          # c  variable (per-unit) purchase cost
    holding_cost: float = 0.5        # h  holding cost per unit per period
    shortage_cost: float = 8.0       # p  lost-sale penalty per unit
    # --- learning / evaluation ---------------------------------------------
    episodes: int = 2500
    eval_episodes: int = 40
    batch_size: int = 64
    replay_capacity: int = 30000
    learning_rate: float = 0.001
    gamma: float = 0.98
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 12000  # counted in gradient steps, not env steps
    target_update_steps: int = 500
    seed: int = 42

    def setup_big_m(self):
        """Smallest valid big-M for the linking constraint q_t <= M * y_t.

        Ordering more than the warehouse can hold is never useful, so the
        tightest valid M is min(Qmax, Imax).  A tight M gives a much stronger
        LP relaxation, hence far fewer Branch-and-Bound nodes.
        """
        return int(min(self.max_order, self.max_inventory))

    def save(self, path):
        data = asdict(self)
        data["demand_values"] = list(self.demand_values)
        Path(path).write_text(json.dumps(data, indent=2))

def load_config(path="config.json"):
    import json
    p = Path(path)
    if not p.exists():
        return Config()
    data = json.loads(p.read_text())
    if "demand_values" in data:
        data["demand_values"] = tuple(data["demand_values"])
    return Config(**data)
