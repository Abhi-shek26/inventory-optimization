"""Evaluate DP, DQN and both MILP variants on identical demand paths.

Four methods are compared on the same realised demand sequence:

  DQN                     learned policy, no knowledge of the demand distribution
  DP                      exact optimal policy, knows the demand distribution
  MILP (rolling forecast) non-anticipative MILP: re-planned every period on a
                          point forecast -- the implementable "optimisation" baseline
  MILP (clairvoyant)      perfect foresight, solved once per episode -- an
                          optimistic lower bound, NOT an implementable policy

Outputs land in results/: episode-level CSV, scenario summaries, figures and a
JSON file that the slide deck builder reads.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from algorithms.dqn_agent import DQNAgent
from algorithms.dynamic_programming import DynamicProgrammingSolver
from algorithms.milp_optimizer import solve_milp
from config import Config
from environment import InventoryEnv
from experiments.demand_scenarios import SCENARIOS, SCENARIO_LABELS
from utils.simulation import replay_plan, run_episode, summarize
from utils.visualization import (plot_cost_breakdown, plot_cost_comparison,
                                 plot_cost_distributions,
                                 plot_inventory_trajectories, plot_order_patterns,
                                 plot_scenario_costs)

RESULTS = "results"
METHODS = ["DQN", "DP", "MILP (rolling forecast)", "MILP (clairvoyant)"]


def make_dqn_policy(agent, cfg):
    def policy(t, inventory):
        state = np.array([inventory / cfg.max_inventory, t / cfg.horizon], dtype=np.float32)
        valid = range(min(cfg.max_order, cfg.max_inventory - inventory) + 1)
        return agent.act(state, valid)
    return policy


def make_rolling_milp_policy(cfg, forecast_mean, timer, lookahead=5):
    """Model-predictive use of the MILP: forever re-plan, execute one period.

    Every period we forecast the next `lookahead` demands (a point forecast =
    the demand mean), solve the lot-sizing MILP for that window, and execute
    only the FIRST order.  This is non-anticipative -- it never sees the future
    -- so it is a fair, implementable benchmark against DP and DQN.
    """
    def policy(t, inventory):
        window = min(lookahead, cfg.horizon - t)
        if window <= 0:
            return 0
        forecast = [float(forecast_mean)] * window
        window_cfg = replace(cfg, horizon=window)
        t0 = time.perf_counter()
        plan = solve_milp(forecast, window_cfg, initial_inventory=inventory)
        timer["planner_time_s"] += time.perf_counter() - t0
        return plan["orders"][0]
    return policy


def evaluate_scenario(scenario_name, cfg, dp, agent, episodes, run_milp=True):
    env = InventoryEnv(cfg, seed=cfg.seed)
    builder = SCENARIOS[scenario_name]
    rows = []
    trajectories = {}

    for ep in range(episodes):
        rng = np.random.default_rng(cfg.seed + 10000 + ep)
        demand_path, window = builder(rng, cfg, env.values, env.pmf)

        timer = {"planner_time_s": 0.0}
        policies = {
            "DQN": (make_dqn_policy(agent, cfg), lambda: 0.0),
            "DP": (lambda t, inv: dp.act(t, inv), lambda: 0.0),
            "MILP (rolling forecast)": (make_rolling_milp_policy(cfg, cfg.demand_lambda, timer),
                                        lambda: timer["planner_time_s"]),
        }
        for method, (policy, elapsed) in policies.items():
            out = run_episode(policy, demand_path, cfg)
            out["window_cost"] = float(sum(p["cost"] for p in out["periods"][window]))
            rows.append({"scenario": scenario_name, "episode": ep, "method": method,
                         "compute_time_s": elapsed(), **out})
            if ep < 12:
                trajectories.setdefault(method, []).append(
                    {"demand": list(map(int, demand_path)),
                     "inventory": list(out["ending_inventory"]),
                     "orders": [p["q"] for p in out["periods"]]})

        if run_milp:
            t0 = time.perf_counter()
            plan = solve_milp(demand_path, cfg)
            elapsed = time.perf_counter() - t0
            out = replay_plan(plan["orders"], demand_path, cfg)
            out["window_cost"] = float(sum(p["cost"] for p in out["periods"][window]))
            out["num_orders"] = plan["num_orders"]
            rows.append({"scenario": scenario_name, "episode": ep,
                         "method": "MILP (clairvoyant)", "compute_time_s": elapsed, **out})
            if ep < 12:
                trajectories.setdefault("MILP (clairvoyant)", []).append(
                    {"demand": list(map(int, demand_path)),
                     "inventory": list(out["ending_inventory"]),
                     "orders": [p["q"] for p in out["periods"]]})

    frame = pd.DataFrame([{k: v for k, v in r.items() if k not in ("periods", "ending_inventory")}
                          for r in rows])
    return frame, trajectories


def evaluate(config=None, episodes=None, scenarios=None, run_milp=True,
             model_path=os.path.join(RESULTS, "dqn_model.pt")):
    cfg = config or Config()
    n = episodes or cfg.eval_episodes
    scenarios = scenarios or list(SCENARIOS)
    env = InventoryEnv(cfg, seed=cfg.seed)

    t0 = time.perf_counter()
    dp = DynamicProgrammingSolver(cfg, env.values, env.pmf)
    dp_value, _ = dp.solve()
    dp_time = time.perf_counter() - t0

    agent = DQNAgent(cfg)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Train DQN first: python -m experiments.train_dqn")
    agent.load(model_path)

    frames, trajectories = [], {}
    for name in scenarios:
        frame, traj = evaluate_scenario(name, cfg, dp, agent, n, run_milp=run_milp)
        frames.append(frame)
        trajectories[name] = traj
        print(f"  evaluated {name} ({n} episodes)")

    all_rows = pd.concat(frames, ignore_index=True)
    os.makedirs(RESULTS, exist_ok=True)
    all_rows.to_csv(os.path.join(RESULTS, "evaluation_episodes.csv"), index=False)

    summaries = {}
    for name in scenarios:
        subset = all_rows[all_rows["scenario"] == name]
        summaries[name] = {m: summarize(g.to_dict("records"))
                           for m, g in subset.groupby("method")}
    pd.DataFrame({(sc, m): v for sc, d in summaries.items() for m, v in d.items()}).T \
        .to_csv(os.path.join(RESULTS, "summary.csv"))

    payload = {
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in cfg.__dict__.items()},
        "dp_expected_cost": dp_value,
        "dp_solve_time_s": dp_time,
        "scenarios": {sc: SCENARIO_LABELS.get(sc, sc) for sc in scenarios},
        "methods": METHODS,
        "summaries": summaries,
        "episodes_per_scenario": n,
    }
    with open(os.path.join(RESULTS, "evaluation_summary.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    plot_cost_comparison(summaries["Exp1_stationary"],
                         os.path.join(RESULTS, "cost_comparison.png"))
    plot_cost_distributions(all_rows[all_rows["scenario"] == "Exp1_stationary"],
                            os.path.join(RESULTS, "cost_distributions.png"))
    plot_cost_breakdown(summaries["Exp1_stationary"],
                        os.path.join(RESULTS, "cost_breakdown.png"))
    plot_scenario_costs(summaries, os.path.join(RESULTS, "scenario_costs.png"))
    plot_inventory_trajectories(trajectories["Exp1_stationary"],
                                os.path.join(RESULTS, "inventory_trajectories.png"))
    plot_order_patterns(trajectories["Exp1_stationary"],
                        os.path.join(RESULTS, "order_patterns.png"))

    print(f"\nDP expected optimal cost (known distribution): {dp_value:.2f}")
    flat = pd.DataFrame({f"{sc[:4]} | {m}": v for sc, d in summaries.items()
                         for m, v in d.items()}).T
    cols = ["mean_total_cost", "mean_holding_cost", "mean_shortage_cost",
            "mean_ordering_cost", "mean_service_level", "mean_avg_inventory",
            "mean_compute_time_ms"]
    print(flat[cols].round(3).to_string())
    print(f"\nSaved evaluation outputs in ./{RESULTS}/")
    return all_rows, summaries


def rebuild_episode_figures(episodes=6):
    """Regenerate the per-episode trajectory figures without a full evaluation."""
    cfg = Config()
    env = InventoryEnv(cfg, seed=cfg.seed)
    dp = DynamicProgrammingSolver(cfg, env.values, env.pmf)
    dp.solve()
    agent = DQNAgent(cfg)
    agent.load(os.path.join(RESULTS, "dqn_model.pt"))
    _, trajectories = evaluate_scenario("Exp1_stationary", cfg, dp, agent,
                                        episodes, run_milp=True)
    plot_inventory_trajectories(trajectories, os.path.join(RESULTS, "inventory_trajectories.png"))
    plot_order_patterns(trajectories, os.path.join(RESULTS, "order_patterns.png"))
    print("rebuilt trajectory figures")


if __name__ == "__main__":
    import sys
    if "--figures-only" in sys.argv:
        rebuild_episode_figures()
    else:
        evaluate()
