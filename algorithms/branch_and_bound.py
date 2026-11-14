"""A small, readable Branch-and-Bound solver for the fixed-charge lot-sizing MILP.

Why write our own when CBC exists?  Because the presentation has to *show* what
Branch and Bound does, with real numbers from this problem:

  * the root LP relaxation of the fixed-charge model is fractional (y_t ~ 0.x),
    so it is not a real plan -- it is only a lower bound;
  * we then branch on one fractional y_t (force it to 0 or to 1), re-solve the
    LP, and prune nodes whose bound is already worse than the incumbent.

Branching only on y_t is *exact* here.  Once every y_t is fixed the remaining
model is a flow/transportation LP, whose constraint matrix is totally
unimodular, so its LP vertices are integral -- q_t, I_t, S_t come out as whole
units automatically.

>>> from config import Config
>>> from algorithms.branch_and_bound import branch_and_bound
>>> out = branch_and_bound([6,5,7,6,8], Config())
>>> out["cost"], out["nodes_explored"]
"""

from __future__ import annotations

import pulp

from algorithms.milp_optimizer import _build_model, _solve

EPS = 1e-6


def _fractional(value):
    """Distance of a y value from the nearest integer, as a fractionality score."""
    return abs(value - round(value))


def _replay_cost(orders, demand, inventory0, cfg):
    """True cost of a concrete order plan under the realised demand sequence."""
    inventory = float(inventory0)
    total, held, lost_total = 0.0, 0.0, 0.0
    setups = 0
    for q, d in zip(orders, demand):
        q = int(q)
        if q > 0:
            setups += 1
            total += cfg.fixed_order_cost
        available = inventory + q
        sales = min(available, d)
        lost = d - sales
        inventory = available - sales
        total += cfg.order_cost * q + cfg.holding_cost * inventory + cfg.shortage_cost * lost
        held += cfg.holding_cost * inventory
        lost_total += lost
    return {"cost": total, "holding_cost": held, "lost": lost_total,
            "num_orders": setups}


def branch_and_bound(demand, config, initial_inventory=0, max_nodes=5000,
                     big_m=None, solver_msg=False):
    """Solve the lot-sizing MILP by explicit Branch and Bound on the y_t.

    Returns the incumbent, the root LP bound, node statistics and a trace of
    (bound, incumbent) after every node, suitable for plotting.
    """
    demand = [int(d) for d in demand]
    T = len(demand)

    def solve_node(node):
        prob, v = _build_model(demand, config,
                               initial_inventory=initial_inventory,
                               relax=True, big_m=big_m, name="BNB_node")
        for t, target in node["y"].items():
            if target == 0:
                v["y"][t].lowBound, v["y"][t].upBound = 0, 0
            else:
                v["y"][t].lowBound, v["y"][t].upBound = 1, 1
        for t, lo in node["q_lo"].items():
            v["q"][t].lowBound = lo
        for t, hi in node["q_hi"].items():
            v["q"][t].upBound = hi
        status = _solve(prob, solver_msg=solver_msg)
        if pulp.LpStatus[status] != "Optimal":
            return None
        return {
            "obj": float(pulp.value(prob.objective)),
            "y": [_val(x) for x in v["y"]],
            "q": [_val(x) for x in v["q"]],
        }

    root_node = {"y": {}, "q_lo": {}, "q_hi": {}}
    root = solve_node(root_node)
    if root is None:
        raise RuntimeError("Root LP relaxation is infeasible.")
    root_bound = root["obj"]

    best = None                 # (cost, y, q)
    # Greedy incumbent: round every switched-on period to "on", which always
    # yields a feasible integer plan and gives Branch and Bound something to
    # prune against from the very first node.
    greedy = solve_node({"y": {t: (1 if root["y"][t] > 1e-6 else 0) for t in range(T)},
                         "q_lo": {}, "q_hi": {}})
    if greedy is not None:
        orders = [int(round(x)) for x in greedy["q"]]
        cost = _replay_cost(orders, demand, initial_inventory, config)["cost"]
        best = (cost, [int(round(x)) for x in greedy["y"]], orders)
    nodes_explored = 0
    nodes_pruned = 0
    trace = []                  # (node_index, tightest open bound, incumbent)
    # Each stack entry carries the LP bound of its parent, which is a valid
    # lower bound on everything inside that subtree.
    stack = [(root_node, root_bound)]

    while stack and nodes_explored < max_nodes:
        node, parent_bound = stack.pop()
        solution = solve_node(node)
        nodes_explored += 1
        if solution is None:
            nodes_pruned += 1
            continue
        # Prune: the bound cannot beat the incumbent already found.
        if best is not None and solution["obj"] >= best[0] - EPS:
            nodes_pruned += 1
            trace.append((nodes_explored, _open_bound(stack, best), best[0]))
            continue
        frac_y = [t for t in range(T) if _fractional(solution["y"][t]) > 1e-5]
        frac_q = [t for t in range(T) if _fractional(solution["q"][t]) > 1e-5]
        if not frac_y and not frac_q:
            orders = [int(round(x)) for x in solution["q"]]
            cost = _replay_cost(orders, demand, initial_inventory, config)["cost"]
            if best is None or cost < best[0] - EPS:
                best = (cost, [int(round(x)) for x in solution["y"]], orders)
            trace.append((nodes_explored, _open_bound(stack, best), best[0]))
            continue
        if not frac_y and frac_q:
            # Safety valve: with integer setups the LP should already be integral
            # (totally unimodular flow matrix); branch on a fractional quantity
            # if numerical noise ever produces one.
            t = max(frac_q, key=lambda i: _fractional(solution["q"][i]))
            lo, hi = node["q_lo"].get(t, 0), node["q_hi"].get(t, config.max_order)
            mid = int(solution["q"][t] // 1)
            for q_lo, q_hi in ((lo, mid), (mid + 1, hi)):
                child = {"y": dict(node["y"]),
                         "q_lo": dict(node["q_lo"]), "q_hi": dict(node["q_hi"])}
                child["q_lo"][t], child["q_hi"][t] = q_lo, q_hi
                stack.append((child, solution["obj"]))
            trace.append((nodes_explored, _open_bound(stack, best),
                          best[0] if best else float("nan")))
            continue
        # Branch on the earliest fractional setup: period order tends to produce
        # good incumbents quickly in a forward lot-sizing problem.
        branch = min(frac_y, key=lambda t: (_fractional(solution["y"][t]) <= 1e-5, t))
        for target in (1, 0):          # push 0 first so 1 is explored first
            child = {"y": dict(node["y"]),
                     "q_lo": dict(node["q_lo"]), "q_hi": dict(node["q_hi"])}
            child["y"][branch] = target
            stack.append((child, solution["obj"]))
        trace.append((nodes_explored, _open_bound(stack, best),
                      best[0] if best else float("nan")))

    if best is None:
        raise RuntimeError("Branch and Bound finished without an incumbent.")
    if not stack and trace:
        # Nothing open: the bound has met the incumbent, which proves optimality.
        trace.append((trace[-1][0], best[0], best[0]))

    cost, setups, orders = best
    return {
        "cost": cost,
        "setups": setups,
        "orders": orders,
        "num_orders": sum(setups),
        "nodes_explored": nodes_explored,
        "nodes_pruned": nodes_pruned,
        "root_lp_bound": root_bound,
        "root_integrality_gap_pct": 100.0 * (cost - root_bound) / cost if cost else 0.0,
        "root_fractional_setups": sum(1 for x in root["y"] if _fractional(x) > 1e-5),
        "trace": trace,
        "optimal": len(stack) == 0,
    }


def _open_bound(stack, best):
    """Tightest bound among the nodes still to be explored."""
    if not stack:
        return best[0] if best else float("nan")
    return min(bound for _, bound in stack)


def _val(var):
    raw = pulp.value(var)
    return 0.0 if raw is None else float(raw)
