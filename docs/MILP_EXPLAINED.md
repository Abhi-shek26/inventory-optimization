# Your part, explained from zero: the MILP and Branch-and-Bound benchmark

This document is written for the person presenting the MILP part. It assumes you
have not seen integer programming before. Everything here is implemented in
`algorithms/milp_optimizer.py` (the model) and `algorithms/branch_and_bound.py`
(the solver demonstration), and every number quoted comes from `results/`.

---

## 1. What problem are we solving?

A warehouse sells one product over `T = 20` periods (days). Each period:

1. you decide how much to order, `q_t`;
2. customer demand `D_t` happens;
3. demand you cannot serve is lost (lost sales), leftover stock is held.

Money is spent four ways:

| Symbol | Meaning | Value in this project |
|---|---|---|
| `K` | **fixed** cost each period you order at all | 25 |
| `c` | **variable** cost per unit ordered | 2 |
| `h` | holding cost per unit left over at the end of a period | 0.5 |
| `p` | penalty per unit of lost demand | 8 |

`K` is the interesting one. It does not depend on *how much* you order, only on
*whether* you order. That single fact is what turns a linear program into a
mixed-integer program.

## 2. Why is this a MILP and not an LP?

If every period you ordered some positive amount, the cost would be linear in the
quantities and a plain LP would do. But the optimal solution wants to *skip*
periods (order a big batch on Monday, nothing on Tuesday). "Skip or not skip" is a
yes/no decision, and yes/no decisions are binary variables:

```
y_t = 1   if an order is placed in period t
y_t = 0   if no order is placed in period t
```

A model with continuous variables (`q_t`, `I_t`, `S_t`) *and* binary variables
(`y_t`) is a **Mixed-Integer Linear Program**. Sound familiar? The yes/no
structure is exactly the "fixed-charge problem" from the syllabus.

If we ignored `K`, `y_t` would disappear and we would have a min-cost flow LP —
solvable by Simplex in milliseconds. The whole difficulty comes from `K`.

## 3. The big-M trick (the one concept to actually understand)

The binary `y_t` and the quantity `q_t` are separate variables, so nothing stops
the solver from saying "y_t = 0, but q_t = 18". We need a constraint that couples
them. That is what big-M does:

```
q_t  <=  M * y_t
```

Read it in two cases:

- **y_t = 1** (order placed): `q_t <= M`. Since `M` is huge, this is not binding —
  the real limit is `Qmax`. The solver is free to order.
- **y_t = 0** (no order): `q_t <= 0`, and because `q_t >= 0`, this forces `q_t = 0`.

So big-M is a switch: *"you may only produce if you paid the setup cost."* The
objective then charges `K*y_t`, so the solver pays `K` only when it actually
switches the period on. That is the entire link between the setup decision and the
production decision.

### How big should M be?

`M` must be at least as large as the largest possible `q_t`, otherwise you cut off
legal solutions (the model would be wrong). But bigger is not better:

- `M` too small → an optimal plan becomes infeasible (wrong answer).
- `M` too big → the LP relaxation (below) becomes extremely loose, so Branch and
  Bound has to work much harder.

The tightest valid value here is `M = min(Qmax, Imax) = min(30, 40) = 30`, because
ordering more than the warehouse can hold is never useful. That is what
`config.setup_big_m()` returns, and `results/milp_big_m.csv` measures the
difference against a deliberately loose `M = 9999`.

## 4. The complete model

$$\min \sum_{t=1}^{T} \big( K\,y_t + c\,q_t + h\,I_t + p\,S_t \big)$$

subject to, for every period `t`:

```
I_{t-1} + q_t - D_t + S_t = I_t        (1) flow balance
q_t        <= M * y_t                  (2) big-M linking
0 <= q_t   <= Qmax                     (3) order capacity
0 <= I_t   <= Imax                     (4) warehouse capacity
0 <= S_t   <= D_t                      (5) lost sales cannot exceed demand
y_t ∈ {0,1},  q_t, I_t, S_t  continuous
```

- `sales_t = D_t - S_t`, so (1) is the usual `I_t = I_{t-1} + q_t - sales_t`.
- `I_0 = 0` (we start empty).
- Cost is charged on *ending* inventory, matching the simulator exactly.

### The version you saw in the handwritten notes

The note says `I_t - S_t = I_{t+1} + q_t - D_t`. That has a sign slip on `q_t`;
the correct lost-sales balance is the one above. Worth being able to spot — it is
exactly the kind of thing a professor asks about.

### Assumptions (say these out loud in the viva)

1. one product, one warehouse;
2. periodic review, zero replenishment lead time (order arrives immediately);
3. lost sales, not backorders: unmet demand is gone forever;
4. demand is integer, `0..12`, drawn from a truncated Poisson;
5. `K`, `c`, `h`, `p` are known constants;
6. capacity is linear in units.

## 5. What "solving" it means: LP relaxation and Branch and Bound

### The LP relaxation

Drop the `y_t ∈ {0,1}` requirement and let `y_t` be any number in `[0,1]`. The
problem becomes an ordinary LP, cheap to solve, but its answer is nonsense as a
plan: `y_t = 0.4` means "40% of an order", which is not a decision anyone can
make. However, the LP value is always **≤** the true integer optimum, so it is a
valid *lower bound*.

How loose is it? Measured on this problem: the LP relaxation is on average
**24% below** the true MILP optimum (see `results/milp_analysis.json`,
`integrality_gap.mean_pct`). That gap is the reason we cannot just round the LP and
go home, and the reason Branch and Bound exists.

### Branch and Bound

```
1. Solve the LP relaxation (the "root" node).  Get bound z_LP.
2. If every y_t is already 0 or 1, stop - that is the answer.
3. Otherwise pick a fractional y_t (e.g. y_3 = 0.6) and split into two cases:
        branch A:  y_3 = 0
        branch B:  y_3 = 1
   Each branch is a smaller MILP, and each is *at least as expensive* as the root.
4. Solve the LP relaxation of each child node.
        - child bound >= best integer plan found so far  ->  prune (throw away)
        - child is infeasible                            ->  prune
        - child is fully integer                         ->  new incumbent
5. Repeat until nothing is left to explore.  The incumbent is optimal.
```

Two facts worth knowing for the viva:

- **Pruning is safe** because a child's LP bound is a lower bound on everything
  inside that branch. If it already loses to the incumbent, no deeper node inside
  can win.
- **Branching only on `y_t` is exact here.** Once all setups are fixed, what
  remains is a pure flow problem, whose constraint matrix is totally unimodular,
  so the LP solution is automatically integral. Our implementation still guards
  against a fractional `q_t` just in case of numerical noise.

`algorithms/branch_and_bound.py` is a readable implementation of exactly this
algorithm on this exact model, so you can run it and show the node counts:

```bash
python -c "
from dataclasses import replace
from config import Config
from algorithms.milp_optimizer import solve_milp
from algorithms.branch_and_bound import branch_and_bound
# an 8-period instance finishes in a few hundred nodes; the full 20-period
# model needs thousands, which is why real solvers add cuts and presolve
cfg = replace(Config(), horizon=8, max_order=12, max_inventory=12)
demand = [6, 5, 7, 6, 8, 4, 9, 6]
exact = solve_milp(demand, cfg)
out = branch_and_bound(demand, cfg)
print('MILP  ', exact['cost'])
print('B&B   ', out['cost'], '| nodes', out['nodes_explored'],
      '| pruned', out['nodes_pruned'], '| root bound', round(out['root_lp_bound'], 1))
"
```

## 6. The three ways the MILP is used in this project

This is the part most presentations get wrong, so be precise about it.

| Variant | What it knows | Fair? |
|---|---|---|
| **MILP (clairvoyant)** | the exact future demand of the episode | No — impossible in practice. It is an **optimistic lower bound**. |
| **MILP (rolling forecast)** | a point forecast (the mean demand) for the next 5 periods | Yes — re-planned every period, never sees the future. This is model-predictive control, and it is the honest "optimisation benchmark". |
| DP | the demand *distribution*, not the realisation | Yes. Exact optimum for a policy. |
| DQN | nothing but experience | Yes — and that is the point of the RL comparison. |

Why include the unfair one at all? Because it answers "how much money did
foreseeing the future earn?" — a clean upper bound on what any policy could ever
achieve. Use the phrase **perfect-information bound** and you are safe.

## 7. Where to look in the results

| File | What it shows |
|---|---|
| `results/milp_small_instance.csv` | all `2^4 = 16` setup patterns on a 4-period instance, with the optimal one marked. Brute force as a sanity check. |
| `results/milp_bnb_stats.csv` | node counts, pruned nodes, root LP bound, root gap per instance, plus a cross-check that our B&B equals CBC. |
| `results/milp_bnb_trace.png` | the classic B&B picture: bound and incumbent closing in as nodes are explored. |
| `results/milp_integrality_gap.png` | distribution of the root LP gap over many demand paths. |
| `results/milp_big_m.csv` / `.png` | tight `M = min(Qmax, Imax)` vs loose `M = 9999`: LP bound quality and B&B effort. |
| `results/milp_fixed_cost.png` | how `K` changes cost and the number of orders — proof that `K` drives batching. |
| `results/milp_wagner_whitin_check.csv` | with capacity and shortage penalties switched off, the MILP must reproduce the Wagner-Whitin optimum. `max |difference| = 0` means the formulation is verified against an independent exact algorithm. |
| `results/summary.csv` | the four methods on all three demand scenarios with every metric from the brief. |

## 8. Likely viva questions, with short answers

**Why not just use an LP?**
Because of the fixed order cost `K`, which needs a binary "do we order" decision.
An LP cannot represent "pay `K` only once per ordering period".

**What does big-M do?**
It links the binary setup variable to the quantity: `q_t <= M*y_t`. If `y_t = 0`
then `q_t = 0`; if `y_t = 1` the quantity is free up to `M`.

**What is a bad choice of M?**
Too small and you cut off feasible plans. Too large and the LP relaxation becomes
weak, so the solver branches far more. Measured in `milp_big_m.csv`.

**Why is the LP relaxation value below the optimum?**
Relaxing `y_t ∈ {0,1}` to `y_t ∈ [0,1]` enlarges the feasible set, so the minimum
can only decrease. A larger feasible region cannot increase a minimisation.

**How does Branch and Bound guarantee optimality?**
Every pruned node is pruned only when its LP bound is already worse than a known
integer plan, so nothing better can be hiding inside it. When the tree is empty,
the incumbent beats every remaining bound.

**Is MILP fair against RL?**
Only the rolling-forecast version is. The clairvoyant version is reported
separately, labelled as a perfect-information bound.

**Why can't MILP handle uncertainty like RL does?**
A deterministic MILP needs one demand number per period. Under uncertainty you
must either forecast (and re-plan, which is what we do) or move to stochastic
programming / scenario trees, which grows fast with the horizon.

**Is the MILP finding a policy?**
No. It finds a *plan* for one realised demand sequence. A policy maps states to
actions for every possible state — that is DP's job (exactly, given the
distribution) and DQN's job (approximately, from experience).

**Why does DP beat the rolling MILP?**
DP uses the full distribution and looks exactly to the end of the horizon; the
rolling MILP sees a 5-period Gaussian-free point forecast and re-plans myopically.
Forecasting the mean ignores variance, and in inventory variance is exactly what
you are paying for.

## 9. One-slide summary you can memorise

> The fixed ordering cost makes this a fixed-charge lot-sizing problem, so it is a
> MILP, not an LP. A binary `y_t` says whether we order, and big-M
> (`q_t <= M*y_t`) couples that yes/no decision to the quantity. Solving it means
> Branch and Bound: the LP relaxation is a valid but loose lower bound (about 24%
> loose here), and we branch on the fractional setups until the bound meets an
> integer plan. We use the model twice — once as a perfect-information lower
> bound, once as a receding-horizon planner that only sees a forecast — and the
> second one is the fair comparison against DP and the DQN.
