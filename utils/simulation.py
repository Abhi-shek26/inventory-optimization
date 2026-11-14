"""One simulation engine used by DP, DQN and the MILP rolling plan.

Keeping a single rollout function means the three methods are scored on exactly
the same rules: same physics, same cost accounting, same demand path.
"""

from __future__ import annotations

import numpy as np


def _period_cost(q, demand, inventory, cfg):
    """Cost of one period and the resulting state, shared by every method."""
    available = inventory + q
    sales = min(available, demand)
    lost = demand - sales
    ending = available - sales
    fixed = cfg.fixed_order_cost if q > 0 else 0.0
    variable = cfg.order_cost * q
    holding = cfg.holding_cost * ending
    shortage = cfg.shortage_cost * lost
    return {
        "q": q, "sales": sales, "lost": lost, "ending": ending,
        "fixed": fixed, "variable": variable,
        "holding": holding, "shortage": shortage,
        "cost": fixed + variable + holding + shortage, "demand": demand,
    }


def run_episode(policy, demand_path, cfg, initial_inventory=0, max_order=None):
    """Roll a policy through one realised demand path.

    policy(t, inventory) -> order quantity
    Returns per-period detail plus episode aggregates.
    """
    demand_path = [int(d) for d in demand_path]
    cap = cfg.max_order if max_order is None else max_order
    inventory = int(initial_inventory)
    periods = []
    for t, demand in enumerate(demand_path):
        allowed = min(cap, cfg.max_inventory - inventory)
        start = inventory
        q = int(max(0, min(int(policy(t, inventory)), allowed)))
        row = _period_cost(q, demand, start, cfg)
        row["t"] = t
        row["inventory_start"] = start
        inventory = row["ending"]
        periods.append(row)
    return _aggregate(periods, cfg)


def replay_plan(orders, demand_path, cfg, initial_inventory=0):
    """Execute a fixed order plan (used for the clairvoyant MILP)."""
    orders = [int(q) for q in orders]
    it = iter(orders)

    def policy(t, inventory):
        return next(it, 0)

    return run_episode(policy, demand_path, cfg, initial_inventory=initial_inventory)


def _aggregate(periods, cfg):
    total_demand = sum(p["demand"] for p in periods)
    sold = sum(p["sales"] for p in periods)
    inventory_after = np.array([p["ending"] for p in periods], dtype=float)
    inventory_avg = float(inventory_after.mean()) if periods else 0.0
    total = float(sum(p["cost"] for p in periods))
    return {
        "total_cost": total,
        "fixed_cost": float(sum(p["fixed"] for p in periods)),
        "variable_cost": float(sum(p["variable"] for p in periods)),
        "ordering_cost": float(sum(p["fixed"] + p["variable"] for p in periods)),
        "holding_cost": float(sum(p["holding"] for p in periods)),
        "shortage_cost": float(sum(p["shortage"] for p in periods)),
        "service_level": sold / total_demand if total_demand else 1.0,
        "avg_inventory": inventory_avg,
        "ending_inventory": inventory_after,
        "lost_units": int(sum(p["lost"] for p in periods)),
        "num_orders": int(sum(1 for p in periods if p["q"] > 0)),
        "units_ordered": int(sum(p["q"] for p in periods)),
        "periods": periods,
    }


METRIC_COLUMNS = ["mean_total_cost", "std_total_cost", "mean_holding_cost",
                  "mean_shortage_cost", "mean_ordering_cost",
                  "mean_service_level", "mean_avg_inventory",
                  "mean_orders", "mean_compute_time_ms"]


def summarize(records):
    """Aggregate episode records into the metrics required by the brief.

    `records` are dicts produced by run_episode/replay_plan, optionally with a
    "compute_time_s" key (planning time per episode) and a "window_cost" key.
    """
    costs = np.array([r["total_cost"] for r in records], dtype=float)
    service = np.array([r["service_level"] for r in records], dtype=float)
    n = len(records)
    out = {
        "mean_total_cost": float(costs.mean()),
        "std_total_cost": float(costs.std(ddof=1)) if n > 1 else 0.0,
        "median_total_cost": float(np.median(costs)),
        "p95_total_cost": float(np.percentile(costs, 95)),
        "mean_holding_cost": float(np.mean([r["holding_cost"] for r in records])),
        "mean_shortage_cost": float(np.mean([r["shortage_cost"] for r in records])),
        "mean_ordering_cost": float(np.mean([r["ordering_cost"] for r in records])),
        "mean_fixed_cost": float(np.mean([r["fixed_cost"] for r in records])),
        "mean_service_level": float(service.mean()),
        "mean_avg_inventory": float(np.mean([r["avg_inventory"] for r in records])),
        "mean_orders": float(np.mean([r["num_orders"] for r in records])),
        "mean_lost_units": float(np.mean([r["lost_units"] for r in records])),
        "mean_compute_time_ms": 1000.0 * float(np.mean([r.get("compute_time_s", 0.0)
                                                        for r in records])),
        "episodes": n,
    }
    window = [r["window_cost"] for r in records if "window_cost" in r]
    if window:
        out["mean_window_cost"] = float(np.mean(window))
    return out

def summarize_metrics(records):
    """Backwards-compatible alias used by experiments/evaluate.py."""
    return summarize(records)
