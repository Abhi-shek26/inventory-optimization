# RL vs. Optimization: Managing Inventory Under Demand Uncertainty

A reproducible Python project comparing three ways to run a single-product,
periodic-review inventory system where demand is uncertain and every order
carries a **fixed cost**:

| Method | What it knows | Role |
|---|---|---|
| **MILP (fixed-charge lot sizing, Branch & Bound)** | the exact demand sequence | optimisation benchmark / perfect-information lower bound |
| **MILP (rolling forecast)** | a point forecast of the next 5 periods | fair, implementable optimisation baseline (re-plans every period) |
| **DP (Dynamic Programming)** | the demand *distribution* | exact optimal policy |
| **DQN (Deep Q-Network)** | nothing but experience | learned policy |

The MILP part (formulation, big-M, LP relaxation, Branch and Bound) is the
component documented in depth in [`docs/MILP_EXPLAINED.md`](docs/MILP_EXPLAINED.md).

## Model

Single item, periodic review, lost sales, zero replenishment lead time.
At period `t` we choose an order `q_t`, then demand `D_t` realises:

- available: `I_t + q_t`
- sales: `min(I_t + q_t, D_t)`, lost sales: `max(D_t - I_t - q_t, 0)`
- ending inventory: `max(0, I_t + q_t - D_t)`
- period cost: `K·1{q_t>0} + c·q_t + h·I_(t+1) + p·S_t`

`K` is what makes the exact model a **MILP** rather than an LP: it introduces a
binary setup decision `y_t` per period, linked to the quantity by the big-M
constraint `q_t <= M·y_t`.

Default parameters (`config.py`): `T=20`, `Imax=40`, `Qmax=30`, `K=25`, `c=2`,
`h=0.5`, `p=8`, truncated `Poisson(6)` demand.

### Information matters — be explicit about it

DP and DQN are **non-anticipative**: they decide with the present state only.
The clairvoyant MILP sees the whole future and is therefore an **optimistic
lower bound**, not a policy. The rolling-forecast MILP is the honest
optimisation comparison. Do not present the clairvoyant number as achievable.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

PuLP ships with the CBC solver, which is what actually solves the MILPs.
`pip install "PuLP==3.*"` — PuLP 4.x changed its variable API and the model code
here targets the documented 3.x API.

## Run the full pipeline

```bash
python -m experiments.train_dqn        # ~2 min on CPU, writes results/dqn_model.pt
python -m experiments.evaluate         # 4 methods x 3 scenarios (~3-4 min; MILP is the slow part)
python -m experiments.milp_analysis    # MILP-only evidence: gaps, B&B nodes, big-M, K sensitivity
python -m experiments.statistics       # paired t-tests of every method against DP
python -m presentation.build_deck      # builds the slide deck from the results
```

Optional extras:

```bash
python -m experiments.evaluate --figures-only     # regenerate trajectory figures only
python -m experiments.sensitivity_analysis        # DP sensitivity to the shortage penalty
python -m unittest discover -s tests -v           # 14 tests, ~2 s
```

## What the MILP part produces

| File | Content |
|---|---|
| `results/milp_small_instance.csv` | exhaustive 2^4 setup-pattern enumeration (brute-force check) |
| `results/milp_bnb_stats.csv` | B&B nodes, pruned nodes, root LP bound, root gap; cost matches CBC |
| `results/milp_bnb_trace.png` | bound vs incumbent convergence |
| `results/milp_integrality_gap.png` | distribution of the root LP relaxation gap |
| `results/milp_big_m.csv/.png` | tight `M = min(Qmax, Imax)` vs loose `M = 9999` |
| `results/milp_fixed_cost.png` | how `K` changes cost and the number of orders |
| `results/milp_wagner_whitin_check.csv` | MILP vs the Wagner-Whitin exact optimum |
| `results/paired_tests.csv` / `.json` | paired difference vs DP per scenario: mean, 95% CI, p-value, paths won |

Headline numbers from the current run (40 episodes per scenario, `K = 25`):

- the LP relaxation sits ~24% below the MILP optimum on average, which is why
  Branch and Bound is needed rather than rounding;
- with capacity and shortage penalties disabled the MILP reproduces the
  Wagner-Whitin optimum exactly (max absolute difference `0`);
- a loose `M = 9999` keeps the same optimum but weakens the root LP bound from a
  9% to a 55% gap — the practical reason for the tightest valid `M`;
- paired test on the same 40 paths: the rolling MILP is statistically
  indistinguishable from DP (-0.1%, p = 0.81), the DQN is reliably 5.3% dearer
  (p < 1e-6, better on 5/40 paths), and the clairvoyant MILP is 9.3% below DP on
  all 40 paths - the value of perfect foresight, unattainable by any policy;
- the DQN's entire gap is shortage cost (+31 units/episode): one slightly low
  reorder threshold, not instability.

## Repository layout

```
config.py                  parameters + tight big-M helper
environment.py             the simulator (step/cost/fixed-charge accounting)
algorithms/
  milp_optimizer.py        fixed-charge MILP: clairvoyant + rolling forecast + LP relaxation
  branch_and_bound.py      readable Branch and Bound on the y_t, with node trace
  lot_sizing.py            Wagner-Whitin exact DP (independent verification)
  dynamic_programming.py   finite-horizon stochastic DP
  dqn_agent.py             Q-network, replay buffer, target network
experiments/
  train_dqn.py             training loop
  evaluate.py              four methods on identical demand paths
  demand_scenarios.py      stationary / step-change / high-variance demand
  milp_analysis.py         all MILP-specific experiments
utils/
  simulation.py            one rollout engine + metric summarisation
  visualization.py         every figure
docs/MILP_EXPLAINED.md     the MILP part explained from zero, with viva questions
presentation/              build_deck.py + the generated .pptx (+ PNG previews)
tests/test_core.py         correctness tests, incl. MILP vs brute force vs Wagner-Whitin
```

## Assumptions and limitations

- one product, zero lead time, integer units, lost sales (no backorders);
- demand is drawn from a truncated Poisson — the truncation is explicit and
  should be reported;
- the DQN is trained on a single seed and on stationary demand; the scenario
  experiments therefore measure a *frozen* policy, which is the honest reading;
- the rolling MILP uses a point forecast, so it ignores demand variance by
  construction; a scenario-tree stochastic program would be the proper extension;
- the clairvoyant MILP is a bound, not a policy.
