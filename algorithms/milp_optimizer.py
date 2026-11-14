"""Fixed-charge multi-period lot-sizing MILP  --  the MILP / Branch-and-Bound part.

Why MILP?
---------
The problem contains a *fixed ordering cost* K: every period we must decide
"do we place an order at all?"  That is a yes/no (0/1) decision, so the model
is a Mixed-Integer Linear Program, not a plain LP.  The yes/no decision is
linked to the order quantity by the classic **big-M** constraint

        q_t  <=  M * y_t ,        y_t in {0, 1}

so q_t can only be positive if the setup indicator y_t is switched on.

Formulation (t = 1..T)
----------------------
Parameters
    D_t   demand in period t (known exactly for the clairvoyant variant)
    K     fixed cost per period in which an order is placed
    c     variable (per-unit) purchase cost
    h     holding cost per unit left in stock at the end of a period
    p     penalty per unit of demand that is lost
    Qmax  maximum order quantity per period
    Imax  warehouse capacity (maximum ending inventory)
    M     big-M, tightest valid value = min(Qmax, Imax)

Decision variables
    q_t >= 0      order quantity           (integer units)
    I_t >= 0      ending inventory
    S_t >= 0      lost sales (unmet demand)
    y_t in {0,1}  setup indicator (1 <=> an order is placed in period t)

Objective
    min  sum_t [ K*y_t + c*q_t + h*I_t + p*S_t ]

Constraints
    (1) flow balance   :  I_{t-1} + q_t - D_t + S_t = I_t ,  I_0 given
    (2) big-M linking  :  q_t <= M * y_t
    (3) capacity       :  0 <= q_t <= Qmax ,   0 <= I_t <= Imax
    (4) lost sales     :  0 <= S_t <= D_t

Note on constraint (1): `sales_t = D_t - S_t`, so the balance is the standard
lost-sales recursion  I_t = I_{t-1} + q_t - sales_t.
(the handwritten note "I_t - S_t = I_{t+1} + q_t - D_t" has a sign slip; the
form above is the version that keeps inventory non-negative.)

Three ways the same model is used in this project
-------------------------------------------------
1. `solve_milp`            -- generic solver for one demand sequence.
2. clairvoyant oracle      -- demand = the *realised* path (perfect foresight).
3. rolling forecast        -- at each period, forecast the remaining demand as
                              its mean and re-solve (deterministic-equivalent MPC).
                              This is the *fair*, non-anticipative use of MILP.
"""

from __future__ import annotations

import time

import pulp


def _build_model(demand, config, initial_inventory=0, relax=False,
                 big_m=None, name="Inventory"):
    """Assemble the fixed-charge lot-sizing model.

    relax=True turns y_t into a continuous variable in [0, 1], which gives the
    LP relaxation used to obtain a lower bound inside Branch and Bound.
    """
    demand = [int(d) for d in demand]
    T = len(demand)
    if big_m is None:
        big_m = config.setup_big_m()

    prob = pulp.LpProblem(name, pulp.LpMinimize)

    q = [pulp.LpVariable(f"q_{t}", lowBound=0, upBound=config.max_order,
                         cat=pulp.LpContinuous) for t in range(T)]
    inv = [pulp.LpVariable(f"I_{t}", lowBound=0, upBound=config.max_inventory,
                           cat=pulp.LpContinuous) for t in range(T)]
    lost = [pulp.LpVariable(f"S_{t}", lowBound=0, upBound=demand[t],
                            cat=pulp.LpContinuous) for t in range(T)]
    y = [pulp.LpVariable(f"y_{t}", lowBound=0, upBound=1,
                         cat=pulp.LpContinuous if relax else pulp.LpBinary)
         for t in range(T)]

    prev = float(initial_inventory)
    for t in range(T):
        # (1) flow balance
        prob += prev + q[t] - demand[t] + lost[t] == inv[t], f"balance_{t}"
        # (2) big-M: q_t <= M * y_t  -- links the quantity to the yes/no decision
        prob += q[t] <= big_m * y[t], f"setup_link_{t}"
        prev = inv[t]

    prob += pulp.lpSum(config.fixed_order_cost * y[t]
                       + config.order_cost * q[t]
                       + config.holding_cost * inv[t]
                       + config.shortage_cost * lost[t]
                       for t in range(T)), "total_cost"

    return prob, dict(q=q, inv=inv, lost=lost, y=y, T=T, big_m=big_m)


def _solve(prob, solver_msg=False, time_limit=None):
    kwargs = {"msg": bool(solver_msg)}
    if time_limit is not None:
        kwargs["timeLimit"] = time_limit
    return prob.solve(pulp.PULP_CBC_CMD(**kwargs))


def solve_milp(demand_sequence, config, initial_inventory=0,
               relax=False, big_m=None, with_lp_bound=False,
               solver_msg=False, time_limit=None):
    """Solve the fixed-charge lot-sizing MILP for one demand sequence.

    Returns a dict with the optimal plan, its cost breakdown and (optionally)
    the LP-relaxation bound so the integrality gap can be reported.
    """
    demand = [int(d) for d in demand_sequence]
    t0 = time.perf_counter()
    prob, v = _build_model(demand, config, initial_inventory=initial_inventory,
                           relax=relax, big_m=big_m,
                           name="LotSizing_relax" if relax else "LotSizing")
    status = _solve(prob, solver_msg=solver_msg, time_limit=time_limit)
    elapsed = time.perf_counter() - t0
    status_name = pulp.LpStatus[status]
    if status_name not in ("Optimal", "Not Solved"):  # "Not Solved" => time limit
        raise RuntimeError(f"MILP did not solve to optimality: {status_name}")

    def val(var):
        raw = pulp.value(var)
        return 0.0 if raw is None else float(raw)

    orders = [int(round(val(x))) for x in v["q"]]
    setups = [1 if val(x) > 0.5 else 0 for x in v["y"]]
    inventory = [val(x) for x in v["inv"]]
    lost = [val(x) for x in v["lost"]]

    result = {
        "cost": float(pulp.value(prob.objective)),
        "status": status_name,
        "orders": orders,
        "setups": setups,
        "inventory": inventory,
        "lost_sales": lost,
        "fixed_cost": config.fixed_order_cost * sum(setups),
        "variable_cost": config.order_cost * sum(orders),
        "holding_cost": config.holding_cost * sum(inventory),
        "shortage_cost": config.shortage_cost * sum(lost),
        "num_orders": sum(setups),
        "big_m": v["big_m"],
        "solve_time_s": elapsed,
        "relaxed": relax,
    }
    if with_lp_bound and not relax:
        bound = solve_milp(demand, config, initial_inventory=initial_inventory,
                           relax=True, big_m=big_m, solver_msg=solver_msg)
        result["lp_relaxation_cost"] = bound["cost"]
        if result["cost"] > 0:
            result["integrality_gap_pct"] = 100.0 * (result["cost"] - bound["cost"]) / result["cost"]
        else:
            result["integrality_gap_pct"] = 0.0
    return result


def solve_clairvoyant_milp(demand_sequence, config, solver_msg=False):
    """Perfect-information (oracle) MILP solved once per realised demand path.

    It sees the entire future demand sequence before choosing any order, so its
    cost is an optimistic LOWER BOUND -- not a policy that can be implemented.
    """
    return solve_milp(demand_sequence, config, with_lp_bound=False,
                      solver_msg=solver_msg)


def solve_rolling_milp(demand_sequence, config, forecast=None,
                       solver_msg=False):
    """Non-anticipative MILP: re-solve each period on a demand forecast.

    At period t the model only knows the past.  Future demand is replaced by a
    point forecast (the demand mean by default, i.e. the deterministic
    equivalent), the MILP is solved for the remaining horizon, and only the
    FIRST order of that plan is executed.  Then demand realises and we re-plan.
    This is the honest, implementable version of "the MILP benchmark".
    """
    demand = [int(d) for d in demand_sequence]
    T = len(demand)
    mean = float(forecast if forecast is not None else config.demand_lambda)

    inventory = 0.0
    total = 0.0
    planner_time = 0.0
    plan_orders, sales_per_period, lost_per_period = [], [], []
    for t in range(T):
        remaining = T - t
        forecast_path = [mean] * remaining
        plan = solve_milp(forecast_path, config, initial_inventory=inventory,
                          solver_msg=solver_msg)
        planner_time += plan["solve_time_s"]
        action = plan["orders"][0] if plan["orders"] else 0

        available = inventory + action
        sales = min(available, demand[t])
        lost = demand[t] - sales
        inventory = available - sales
        total += ((config.fixed_order_cost if action > 0 else 0.0)
                  + config.order_cost * action
                  + config.holding_cost * inventory
                  + config.shortage_cost * lost)
        plan_orders.append(action)
        sales_per_period.append(sales)
        lost_per_period.append(lost)

    return {"cost": total, "orders": plan_orders,
            "sales": sales_per_period, "lost_sales": lost_per_period,
            "planner_time_s": planner_time}
