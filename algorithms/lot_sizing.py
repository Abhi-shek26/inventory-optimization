"""Wagner-Whitin: the textbook exact algorithm for the uncapacitated
fixed-charge lot-sizing problem.

It exists in this project for one reason: it is an *independent*, analytically
proven optimum.  If the MILP reproduces the Wagner-Whitin cost on instances
where both apply (no capacity binding, unmet demand forbidden), then the MILP
formulation is verified -- not just "the solver said so".

Setting: no warehouse limit, no lost sales, zero lead time.
Recursion (backward-looking over the last order cycle):

    f(0) = 0
    f(t) = min_{1 <= j <= t} [ f(j-1) + K + c*sum_{i=j..t} D_i
                                       + h*sum_{k=j+1..t} (k-j)*D_k ]

`j` is the period in which the last order is placed and `j..t` is the block of
demand that order covers.  Complexity O(T^2).
"""


def wagner_whitin_cost(demand, config):
    demand = [int(d) for d in demand]
    T = len(demand)
    K, c, h = config.fixed_order_cost, config.order_cost, config.holding_cost

    # cjk[j][t] = cost of covering periods j..t with one order placed at j
    cjk = [[0.0] * T for _ in range(T)]
    for j in range(T):
        qty, hold = 0.0, 0.0
        for t in range(j, T):
            qty += demand[t]
            if t > j:
                hold += (t - j) * demand[t]
            cjk[j][t] = K + c * qty + h * hold

    f = [0.0] + [float("inf")] * T
    arg = [0] * (T + 1)
    for t in range(1, T + 1):
        for j in range(1, t + 1):
            cand = f[j - 1] + cjk[j - 1][t - 1]
            if cand < f[t] - 1e-12:
                f[t], arg[t] = cand, j

    # Recover the order periods from the DP choices.
    order_periods = []
    t = T
    while t > 0:
        j = arg[t]
        order_periods.append(j - 1)
        t = j - 1
    order_periods.reverse()

    return {"cost": f[T], "order_periods": order_periods,
            "orders": _orders_from_periods(order_periods, demand)}


def _orders_from_periods(order_periods, demand):
    """Convert 'order in these periods' into explicit quantities."""
    T = len(demand)
    orders = [0] * T
    for idx, start in enumerate(order_periods):
        end = order_periods[idx + 1] if idx + 1 < len(order_periods) else T
        orders[start] = int(sum(demand[start:end]))
    return orders
