# Fixed-charge lot sizing as a MILP: formulation, Branch & Bound, and an exact benchmark

The **MILP part** of the team project *"RL vs. Optimization: Managing Inventory
Under Demand Uncertainty"*.

The single-product, periodic-review inventory problem contains a fixed ordering
cost `K`, which makes "do we order this period?" a yes/no decision. That is what
turns the exact model into a **Mixed-Integer Linear Program**, and it is the whole
subject of this part: the formulation, the big-M link, the LP relaxation, Branch
and Bound, and what the exact optimum is worth compared with the achievable
policies.

The RL (DQN) half of the project belongs to the other team member.
`experiments/evaluate.py` imports it lazily, so this repository runs and
demonstrates **three** methods — `DP`, `MILP (rolling forecast)`,
`MILP (clairvoyant)` — and automatically adds `DQN` in the joint repository where
the RL files and a trained model exist. Everything shared between the two halves
is frozen by [docs/TEAM_SYNC.md](docs/TEAM_SYNC.md).

## Model

Single item, periodic review, lost sales, zero replenishment lead time. At period
`t` we choose an order `q_t`, then demand `D_t` realises:

- available `I_t + q_t`; sales `min(I_t + q_t, D_t)`; lost `max(D_t - I_t - q_t, 0)`
- ending inventory `I_(t+1) = max(0, I_t + q_t - D_t)`
- period cost `K·1{q_t>0} + c·q_t + h·I_(t+1) + p·S_t`

Default instance (`config.py`): `T = 20`, `Imax = 40`, `Qmax = 30`, `K = 25`,
`c = 2`, `h = 0.5`, `p = 8`, truncated `Poisson(6)` demand on `0..12`
(the truncation keeps 99.1% of the probability mass; its mean is 5.93).

The MILP itself (`algorithms/milp_optimizer.py`):

```
min  sum_t ( K*y_t + c*q_t + h*I_t + p*S_t )
s.t. I_(t-1) + q_t - D_t + S_t = I_t          flow balance
     q_t <= M * y_t                            big-M (M = min(Qmax, Imax) = 30)
     0 <= q_t <= Qmax,  0 <= I_t <= Imax
     0 <= S_t <= D_t,    y_t in {0,1}
```

## Quickstart — what to run to show the results

```bash
pip install -r requirements.txt          # no torch needed for this part

# 1. The MILP evidence: brute force, LP gap, Branch & Bound, big-M, K sensitivity.
#    Self-contained, does not touch the RL part. ~2 min
python -m experiments.milp_analysis

# 2. Comparison on identical demand paths (DP + both MILP variants). ~3-4 min,
#    because the rolling planner re-solves the MILP 20 times per episode.
python -m experiments.evaluate

# 3. Paired significance tests against DP. ~2 s
python -m experiments.statistics
```

Optional: `python -m experiments.evaluate --figures-only` rebuilds only the
per-episode trajectory figures.

In the joint repository (RL files present) you additionally run
`python -m experiments.train_dqn` before step 2, and step 2 then reports four
methods.

All outputs are written to `./results/`, which is **created automatically** and is
excluded from git (`.gitignore`). Clone the repo, run the commands, and the folder
reappears with every CSV, JSON and figure.

## What the MILP part establishes

| Claim | Evidence |
|---|---|
| The formulation is correct, not just "solver said optimal" | `results/milp_small_instance.csv`: all 2⁴ setup patterns enumerated by brute force; MILP returns the same cheapest pattern |
| The LP relaxation is not a plan | `results/milp_integrality_gap.png`: average root gap ≈ 24% over 20 demand paths |
| Branch and Bound is implemented correctly | `results/milp_bnb_stats.csv`: same cost as CBC, 127–177 nodes, ~40% of nodes pruned |
| A tight big-M matters | `results/milp_big_m.csv`: tight `M` → 9.3% root gap; loose `M = 9999` → 55.5% gap, same optimum |
| `K` is what creates batching | `results/milp_fixed_cost.png`: orders per episode fall as `K` rises; at `K = 0` the problem is a plain LP |
| What perfect foresight is worth | `results/summary.csv` + `results/paired_tests.csv`: clairvoyant MILP is 9.3% below DP on all 40 paths — an unattainable bound |
| The implementable MILP is competitive | the rolling-forecast MILP is statistically indistinguishable from DP (−0.1%, p = 0.81) |

## Files

| File | Role |
|---|---|
| `config.py` | every parameter in one place; `setup_big_m()` returns `min(Qmax, Imax)` |
| `environment.py` | the simulator: physics + cost accounting, incl. the fixed charge |
| `algorithms/milp_optimizer.py` | **core**: MILP builder, clairvoyant solver, LP relaxation, rolling-forecast planner |
| `algorithms/branch_and_bound.py` | **core**: readable B&B over the `y_t`, with node statistics and a bound trace |
| `algorithms/dynamic_programming.py` | stochastic DP — the optimal-policy reference (shared) |
| `experiments/milp_analysis.py` | all MILP-only experiments and figures |
| `experiments/evaluate.py` | the shared comparison harness over identical demand paths |
| `experiments/statistics.py` | paired t-tests and confidence intervals vs DP |
| `experiments/demand_scenarios.py` | stationary / step-change / high-variance demand |
| `utils/simulation.py` | one rollout engine + metric summarisation (shared) |
| `utils/visualization.py` | every figure |
| `RUNBOOK.md` | command → file → printed output → generated results, step by step |
| `docs/TEAM_SYNC.md` | the frozen parameters, interface and information protocol shared with the RL part |

## Assumptions and limitations

- one product, zero lead time, integer units, lost sales (no backorders);
- demand is a truncated Poisson — stationary by construction, so the RL/DP
  comparison deliberately excludes seasonality and promotions;
- the clairvoyant MILP is a bound, never a policy;
- the rolling MILP uses a point forecast (mean, 5-period lookahead), so it ignores
  demand variance by construction;
- Branch and Bound is demonstrated on 4- and 8-period instances; on the full
  20-period model a naive implementation needs thousands of nodes, which is why
  production solvers add cutting planes and presolve;
- single training seed on the RL side, 40 evaluation episodes per scenario.
