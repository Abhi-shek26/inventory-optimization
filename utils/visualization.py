"""All figures for the project. Every plot is written to results/ as PNG."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_STYLE = {
    "(s,S)": "#8e44ad",
    "Q-learning": "#16a085",
    "DQN": "#2c7fb8",
    "DP": "#31a354",
    "MILP (rolling forecast)": "#f0a202",
    "MILP (clairvoyant)": "#c0392b",
}
DPI = 160


def _save(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()


def plot_training(rewards, path, window=100):
    rewards = np.asarray(rewards, dtype=float)
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, alpha=0.25, color="#888888", label="Episode cost")
    if len(rewards) >= window:
        s = pd.Series(rewards).rolling(window).mean()
        plt.plot(s, color="#2c7fb8", linewidth=2, label=f"{window}-episode moving average")
    plt.xlabel("Training episode")
    plt.ylabel("Episode cost (lower is better)")
    plt.title("DQN learning curve")
    plt.legend()
    _save(path)


def plot_cost_comparison(summary, path):
    names = [m for m in METHOD_STYLE if m in summary]
    means = [summary[n]["mean_total_cost"] for n in names]
    stds = [summary[n]["std_total_cost"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, means, yerr=stds, capsize=5,
                  color=[METHOD_STYLE[n] for n in names], alpha=0.9)
    for bar, m, sd in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + sd, f"{m:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean total cost per episode (lower is better)")
    ax.set_title("Cost comparison on identical demand paths (Exp 1)")
    ax.set_ylim(0, max(means) * 1.18)
    plt.xticks(rotation=12)
    _save(path)


def plot_cost_distributions(frame, path):
    methods = [m for m in METHOD_STYLE if m in set(frame["method"])]
    data = [frame[frame["method"] == m]["total_cost"].values for m in methods]
    fig, ax = plt.subplots(figsize=(7, 4))
    try:                       # matplotlib >= 3.9 renamed `labels` -> `tick_labels`
        parts = ax.boxplot(data, tick_labels=methods, patch_artist=True, widths=0.55)
    except TypeError:
        parts = ax.boxplot(data, labels=methods, patch_artist=True, widths=0.55)
    for patch, m in zip(parts["boxes"], methods):
        patch.set_facecolor(METHOD_STYLE[m])
        patch.set_alpha(0.75)
    ax.set_ylabel("Total cost per episode")
    ax.set_title("Cost distribution across test episodes (Exp 1)")
    plt.xticks(rotation=12)
    _save(path)


def plot_scenario_costs(summaries, path):
    scenarios = list(summaries)
    methods = [m for m in METHOD_STYLE if m in summaries[scenarios[0]]]
    x = np.arange(len(scenarios))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for i, m in enumerate(methods):
        vals = [summaries[s][m]["mean_total_cost"] for s in scenarios]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=m,
               color=METHOD_STYLE[m], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in scenarios], fontsize=9)
    ax.set_ylabel("Mean total cost per episode")
    ax.set_title("Cost across the three demand scenarios")
    ax.legend(fontsize=8, ncol=2)
    _save(path)


def plot_inventory_trajectories(trajectories, path, episodes=4):
    methods = [m for m in METHOD_STYLE if m in trajectories]
    n = min(episodes, min(len(trajectories[m]) for m in methods) if methods else 0)
    if n == 0:
        return
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6))
    for ax, ep in zip(axes.ravel(), range(n)):
        d = trajectories[methods[0]][ep]["demand"]
        ax.bar(range(len(d)), d, color="#dddddd", label="Demand", zorder=0)
        for m in methods:
            ax.plot(trajectories[m][ep]["inventory"], marker="o", markersize=3,
                    color=METHOD_STYLE[m], label=m)
        ax.set_title(f"Episode {ep + 1}", fontsize=10)
        ax.set_xlabel("Period")
        ax.set_ylabel("Ending inventory (units)")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("Inventory trajectories on identical demand paths", y=1.0)
    _save(path)


def plot_order_patterns(trajectories, path, episodes=4):
    methods = [m for m in METHOD_STYLE if m in trajectories]
    n = min(episodes, min(len(trajectories[m]) for m in methods) if methods else 0)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.9 * n), sharex=True)
    if n == 1:
        axes = [axes]
    width = 0.8 / len(methods)
    for ax, ep in zip(axes, range(n)):
        for i, m in enumerate(methods):
            orders = trajectories[m][ep]["orders"]
            ax.bar(np.arange(len(orders)) + i * width - 0.4 + width / 2, orders,
                   width, label=m, color=METHOD_STYLE[m])
        ax.set_ylabel("Order q$_t$")
        ax.set_title(f"Ordering plan, episode {ep + 1}", fontsize=9)
    axes[0].legend(fontsize=7, ncol=4)
    axes[-1].set_xlabel("Period")
    _save(path)


def plot_cost_breakdown(summary, path):
    methods = [m for m in METHOD_STYLE if m in summary]
    parts = [("mean_fixed_cost", "Fixed ordering", "#34495e"),
             ("mean_variable_cost", "Variable purchase", "#5dade2"),
             ("mean_holding_cost", "Holding", "#58d68d"),
             ("mean_shortage_cost", "Shortage / lost sales", "#e74c3c")]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bottom = np.zeros(len(methods))
    for key, label, colour in parts:
        vals = np.array([summary[m].get(key, 0.0) for m in methods])
        ax.bar(methods, vals, bottom=bottom, label=label, color=colour)
        bottom += vals
    for i, m in enumerate(methods):
        ax.text(i, bottom[i], f"{bottom[i]:.0f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_ylim(0, bottom.max() * 1.22)
    ax.set_ylabel("Mean cost per episode")
    ax.set_title("Where the money goes, by method (Exp 1)", pad=26)
    ax.legend(fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14),
              frameon=False)
    plt.xticks(rotation=12)
    _save(path)


def plot_bnb_trace(trace, path, title="Branch and Bound: bound vs incumbent"):
    nodes = [t[0] for t in trace]
    bounds = [t[1] for t in trace]
    incumbents = [t[2] for t in trace]
    best_bound = []
    running = float("inf")
    for b in bounds:
        running = min(running, b)
        best_bound.append(running)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(nodes, bounds, s=9, color="#e74c3c", alpha=0.35,
               label="LP bound of each node explored")
    ax.step(nodes, best_bound, where="post", color="#e74c3c", linewidth=2,
            label="Tightest remaining bound")
    ax.step(nodes, incumbents, where="post", color="#2c7fb8", linewidth=2,
            label="Best integer plan found (incumbent)")
    ax.set_xlabel("Node explored")
    ax.set_ylabel("Objective value")
    ax.set_title(title)
    ax.legend(fontsize=8)
    _save(path)


def plot_big_m_comparison(rows, path):
    labels = [r["label"] for r in rows]
    bounds = [r["lp_root_bound"] for r in rows]
    opt = [r["milp_cost"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, bounds, 0.4, label="Root LP relaxation bound", color="#e74c3c")
    ax.bar(x + 0.2, opt, 0.4, label="MILP optimum", color="#2c7fb8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Objective value")
    ax.set_title("Why a tight big-M matters: LP bound quality")
    ax.legend(fontsize=8)
    _save(path)


def plot_fixed_cost_sensitivity(df, path):
    fig, ax1 = plt.subplots(figsize=(7.5, 4))
    ax1.plot(df["fixed_order_cost"], df["dp_expected_cost"], marker="o",
             color="#31a354", label="DP expected cost")
    ax1.plot(df["fixed_order_cost"], df["milp_mean_cost"], marker="s",
             color="#c0392b", label="MILP clairvoyant mean cost")
    ax1.set_xlabel("Fixed ordering cost K")
    ax1.set_ylabel("Cost per episode")
    ax1.legend(fontsize=8, loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(df["fixed_order_cost"], df["milp_mean_orders"], marker="^",
             linestyle="--", color="#555555", label="Orders per episode (MILP)")
    ax2.set_ylabel("Orders placed per episode")
    ax2.legend(fontsize=8, loc="upper right")
    ax1.set_title("Fixed cost K drives batching: the reason MILP is needed")
    _save(path)


def plot_integrality_gap(gaps, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(gaps, bins=20, color="#f0a202", edgecolor="white")
    ax.axvline(np.mean(gaps), color="#c0392b", linestyle="--",
               label=f"mean gap = {np.mean(gaps):.1f}%")
    ax.set_xlabel("Root integrality gap (%)  = (MILP - LP) / MILP")
    ax.set_ylabel("Number of demand paths")
    ax.set_title("The LP relaxation never gives the right plan")
    ax.legend(fontsize=8)
    _save(path)
