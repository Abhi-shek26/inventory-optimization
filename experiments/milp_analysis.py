"""MILP-specific analysis: the evidence behind the "MILP / Branch-and-Bound" slides.

Produces
    results/milp_small_instance.csv      exhaustive 2^T setup-pattern table
    results/milp_bnb_stats.csv           Branch-and-Bound node statistics
    results/milp_big_m.csv               tight vs loose big-M (LP bound + nodes)
    results/milp_fixed_cost_sensitivity.csv
    results/milp_wagner_whitin_check.csv
    results/milp_analysis.json           everything the deck builder needs
    results/milp_*.png                   figures
"""

from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import replace

import numpy as np
import pulp

from algorithms.branch_and_bound import branch_and_bound
from algorithms.dynamic_programming import DynamicProgrammingSolver
from algorithms.lot_sizing import wagner_whitin_cost
from algorithms.milp_optimizer import (_build_model, _solve, solve_milp)
from config import Config
from environment import InventoryEnv
from utils.visualization import (plot_big_m_comparison, plot_bnb_trace,
                                 plot_fixed_cost_sensitivity,
                                 plot_integrality_gap)

RESULTS = "results"


def _val(var):
    raw = pulp.value(var)
    return 0.0 if raw is None else float(raw)


# ---------------------------------------------------------------- 1. worked example
def enumerate_setup_patterns(demand, cfg):
    """Cost of every possible 0/1 setup pattern (2^T LPs).

    This is the brute-force version of Branch and Bound: the optimal pattern is
    the cheapest feasible one.  With T=4 there are only 16 patterns, which makes
    the whole idea visible in a single table.
    """
    T = len(demand)
    rows = []
    for pattern in itertools.product((0, 1), repeat=T):
        prob, v = _build_model(demand, cfg, relax=True, name="pattern")
        for t, y in enumerate(pattern):
            v["y"][t].lowBound, v["y"][t].upBound = y, y
        status = _solve(prob)
        feasible = pulp.LpStatus[status] == "Optimal"
        rows.append({"pattern": "".join(str(p) for p in pattern),
                     "setups": sum(pattern),
                     "cost": _val(prob.objective) if feasible else float("nan"),
                     "feasible": feasible})
    best = min((r for r in rows if r["feasible"]), key=lambda r: r["cost"])
    for r in rows:
        r["is_optimal"] = r["pattern"] == best["pattern"]
    return rows, best


def small_instance_analysis(cfg):
    demo_cfg = replace(cfg, horizon=4, max_order=12, max_inventory=12)
    demand = [4, 7, 3, 6]
    rows, best = enumerate_setup_patterns(demand, demo_cfg)
    milp = solve_milp(demand, demo_cfg)
    bb = branch_and_bound(demand, demo_cfg)
    return {
        "config": {"horizon": demo_cfg.horizon, "max_order": demo_cfg.max_order,
                   "max_inventory": demo_cfg.max_inventory,
                   "fixed_order_cost": demo_cfg.fixed_order_cost,
                   "order_cost": demo_cfg.order_cost,
                   "holding_cost": demo_cfg.holding_cost,
                   "shortage_cost": demo_cfg.shortage_cost},
        "demand": demand,
        "patterns": rows,
        "best_pattern": best["pattern"],
        "best_cost": best["cost"],
        "milp_cost": milp["cost"],
        "milp_orders": milp["orders"],
        "bnb_cost": bb["cost"],
        "bnb_nodes": bb["nodes_explored"],
        "bnb_pruned": bb["nodes_pruned"],
        "bnb_root_bound": bb["root_lp_bound"],
        "bnb_trace": bb["trace"],
        "bnb_root_fractional_setups": bb["root_fractional_setups"],
    }


# ------------------------------------------------------- 2. integrality gap + B&B
def gap_and_node_analysis(cfg, n_paths=20, bnb_instances=3, bnb_max_nodes=400):
    """Root LP gap on the full 20-period model; B&B runs on a smaller instance.

    The teaching implementation of Branch and Bound solves the 8-period version
    to proven optimality; on the full 20-period model it needs thousands of nodes,
    which is exactly why production solvers add cutting planes and presolve.
    """
    env = InventoryEnv(cfg, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed + 7)
    bnb_cfg = replace(cfg, horizon=8, max_order=12, max_inventory=12)
    bnb_env = InventoryEnv(bnb_cfg, seed=cfg.seed)
    gaps, nodes = [], []
    bnb_rows = []
    trace_example = None
    for i in range(n_paths):
        demand = rng.choice(env.values, size=cfg.horizon, p=env.pmf).astype(int)
        res = solve_milp(demand, cfg, with_lp_bound=True)
        gaps.append(res["integrality_gap_pct"])
        if i < bnb_instances:
            small_demand = rng.choice(bnb_env.values, size=bnb_cfg.horizon,
                                      p=bnb_env.pmf).astype(int)
            res_small = solve_milp(small_demand, bnb_cfg)
            t0 = time.perf_counter()
            bb = branch_and_bound(small_demand, bnb_cfg, max_nodes=bnb_max_nodes)
            elapsed = time.perf_counter() - t0
            nodes.append(bb["nodes_explored"])
            res = res_small
            demand = small_demand
            bnb_rows.append({"instance": i, "horizon": bnb_cfg.horizon,
                             "milp_cost": res["cost"],
                             "bnb_cost": bb["cost"],
                             "root_lp_bound": bb["root_lp_bound"],
                             "root_gap_pct": bb["root_integrality_gap_pct"],
                             "nodes_explored": bb["nodes_explored"],
                             "nodes_pruned": bb["nodes_pruned"],
                             "finished": bb["optimal"],
                             "seconds": elapsed})
            if trace_example is None:
                trace_example = {"demand": [int(d) for d in demand],
                                 "trace": bb["trace"], "nodes": bb["nodes_explored"],
                                 "root_bound": bb["root_lp_bound"],
                                 "cost": bb["cost"]}
    return gaps, bnb_rows, trace_example


# --------------------------------------------------------------- 3. big-M study
def big_m_study(cfg):
    """Loose M weakens the LP relaxation -> weaker bound, more B&B work."""
    tight_cfg = replace(cfg, horizon=8, max_order=20, max_inventory=12)
    env = InventoryEnv(tight_cfg, seed=tight_cfg.seed)
    rng = np.random.default_rng(tight_cfg.seed + 11)
    demands = [rng.choice(env.values, size=tight_cfg.horizon, p=env.pmf).astype(int)
               for _ in range(3)]
    rows = []
    for label, big_m in [("tight M = min(Qmax, Imax)", tight_cfg.setup_big_m()),
                         ("loose M = 9999", 9999)]:
        milp_costs, lp_bounds, nodes = [], [], []
        for demand in demands:
            milp = solve_milp(demand, tight_cfg)
            lp = solve_milp(demand, tight_cfg, relax=True, big_m=big_m)
            bb = branch_and_bound(demand, tight_cfg, big_m=big_m, max_nodes=400)
            milp_costs.append(milp["cost"])
            lp_bounds.append(lp["cost"])
            nodes.append(bb["nodes_explored"])
        mean_milp = float(np.mean(milp_costs))
        mean_lp = float(np.mean(lp_bounds))
        rows.append({
            "label": label,
            "big_m": big_m,
            "milp_cost": mean_milp,
            "lp_root_bound": mean_lp,
            "root_gap_pct": 100.0 * (mean_milp - mean_lp) / mean_milp if mean_milp else 0.0,
            "bnb_nodes": float(np.mean(nodes)),
            "bnb_all_finished": all(n < 400 for n in nodes),
        })
    return rows


# -------------------------------------------------- 4. fixed cost K sensitivity
def fixed_cost_sensitivity(cfg, k_values=(0.0, 10.0, 25.0, 50.0, 100.0),
                           n_paths=12):
    env = InventoryEnv(cfg, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed + 13)
    demands = [rng.choice(env.values, size=cfg.horizon, p=env.pmf).astype(int)
               for _ in range(n_paths)]
    rows = []
    for k in k_values:
        trial = replace(cfg, fixed_order_cost=float(k))
        solver = DynamicProgrammingSolver(trial, env.values, env.pmf)
        dp_value, _ = solver.solve()
        results = [solve_milp(d, trial) for d in demands]
        rows.append({
            "fixed_order_cost": float(k),
            "dp_expected_cost": float(dp_value),
            "milp_mean_cost": float(np.mean([r["cost"] for r in results])),
            "milp_mean_orders": float(np.mean([r["num_orders"] for r in results])),
            "milp_mean_setups_per_10_periods":
                float(np.mean([r["num_orders"] for r in results]) * 10.0 / cfg.horizon),
        })
    return rows


# ------------------------------------------------ 5. Wagner-Whitin verification
def wagner_whitin_check(cfg, n_paths=8):
    """With capacity and shortage penalties switched off, the MILP must equal WW."""
    off = replace(cfg, max_order=10_000, max_inventory=10_000, shortage_cost=1e6)
    env = InventoryEnv(cfg, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed + 17)
    rows = []
    for i in range(n_paths):
        demand = rng.choice(env.values, size=cfg.horizon, p=env.pmf).astype(int)
        milp = solve_milp(demand, off)
        ww = wagner_whitin_cost(demand, off)
        rows.append({"instance": i,
                     "demand_total": int(demand.sum()),
                     "milp_cost": milp["cost"],
                     "wagner_whitin_cost": ww["cost"],
                     "difference": milp["cost"] - ww["cost"],
                     "milp_orders": milp["num_orders"],
                     "ww_orders": len(ww["order_periods"])})
    max_diff = max(abs(r["difference"]) for r in rows)
    return rows, max_diff


def run(config=None):
    cfg = config or Config()
    os.makedirs(RESULTS, exist_ok=True)
    env = InventoryEnv(cfg, seed=cfg.seed)

    print("1/5 worked example: enumerating all 2^4 setup patterns ...")
    small = small_instance_analysis(cfg)
    import pandas as pd
    pd.DataFrame(small["patterns"]).to_csv(
        os.path.join(RESULTS, "milp_small_instance.csv"), index=False)
    print(f"    brute force best pattern {small['best_pattern']} = {small['best_cost']:.2f}, "
          f"MILP = {small['milp_cost']:.2f}, "
          f"B&B = {small['bnb_cost']:.2f} in {small['bnb_nodes']} nodes")

    print("2/5 LP relaxation gap and Branch-and-Bound node counts ...")
    gaps, bnb_rows, trace = gap_and_node_analysis(cfg)
    pd.DataFrame(bnb_rows).to_csv(os.path.join(RESULTS, "milp_bnb_stats.csv"), index=False)
    plot_integrality_gap(gaps, os.path.join(RESULTS, "milp_integrality_gap.png"))
    if trace:
        plot_bnb_trace(trace["trace"], os.path.join(RESULTS, "milp_bnb_trace.png"),
                       title=(f"Branch and Bound on one demand path "
                              f"({trace['nodes']} nodes)"))

    print("3/5 tight vs loose big-M ...")
    big_m_rows = big_m_study(cfg)
    pd.DataFrame(big_m_rows).to_csv(os.path.join(RESULTS, "milp_big_m.csv"), index=False)
    plot_big_m_comparison(big_m_rows, os.path.join(RESULTS, "milp_big_m.png"))

    print("4/5 fixed ordering cost sensitivity ...")
    k_rows = fixed_cost_sensitivity(cfg)
    k_frame = pd.DataFrame(k_rows)
    k_frame.to_csv(os.path.join(RESULTS, "milp_fixed_cost_sensitivity.csv"), index=False)
    plot_fixed_cost_sensitivity(k_frame, os.path.join(RESULTS, "milp_fixed_cost.png"))

    print("5/5 Wagner-Whitin cross-check ...")
    ww_rows, max_diff = wagner_whitin_check(cfg)
    pd.DataFrame(ww_rows).to_csv(os.path.join(RESULTS, "milp_wagner_whitin_check.csv"),
                                 index=False)

    payload = {
        "small_instance": small,
        "integrality_gap": {
            "mean_pct": float(np.mean(gaps)),
            "median_pct": float(np.median(gaps)),
            "min_pct": float(np.min(gaps)),
            "max_pct": float(np.max(gaps)),
            "paths": len(gaps),
        },
        "bnb": {"rows": bnb_rows,
                "mean_nodes": float(np.mean([r["nodes_explored"] for r in bnb_rows]))
                if bnb_rows else 0.0},
        "big_m": big_m_rows,
        "fixed_cost": k_rows,
        "wagner_whitin": {"max_abs_difference": float(max_diff),
                          "instances": len(ww_rows)},
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in cfg.__dict__.items()},
    }
    with open(os.path.join(RESULTS, "milp_analysis.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nMean root integrality gap: {np.mean(gaps):.1f}%")
    print(pd.DataFrame(bnb_rows).round(2).to_string(index=False))
    print(pd.DataFrame(big_m_rows).round(2).to_string(index=False))
    print(f"Wagner-Whitin cross-check: max |MILP - WW| = {max_diff:.6f}")
    print(f"Saved MILP analysis in ./{RESULTS}/")
    return payload


if __name__ == "__main__":
    run()
