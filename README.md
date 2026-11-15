# RL vs. Optimization: Managing Inventory Under Demand Uncertainty

A comparative study of **Reinforcement Learning**, **Classical Inventory Heuristics**, **Stochastic Dynamic Programming**, and **Mixed-Integer Linear Programming (MILP)** on a single-product, periodic-review lost-sales inventory control problem with fixed ordering costs.

---

## Problem Formulation

Single item, periodic review over horizon $T = 20$, lost sales (no backorders), zero replenishment lead time. At period $t$, the agent observes current inventory $I_t$ and chooses an order quantity $q_t \in \{0, 1, \dots, Q_{\max}\}$, after which stochastic demand $D_t$ is realized:

* **Available stock:** $I_t + q_t$
* **Sales:** $\min(I_t + q_t, D_t)$
* **Lost sales:** $S_t = \max(D_t - I_t - q_t, 0)$
* **Ending inventory:** $I_{t+1} = \max(0, I_t + q_t - D_t)$
* **Period cost:** $C_t = K \cdot \mathbb{I}\{q_t > 0\} + c \cdot q_t + h \cdot I_{t+1} + p \cdot S_t$

### Default Instance (`config.py`)
* **Horizon & Capacity:** $T = 20$, $I_{\max} = 40$, $Q_{\max} = 30$
* **Cost Parameters:** Fixed setup cost $K = 25$, unit purchase cost $c = 2$, holding cost $h = 0.5$, lost-sales penalty $p = 8$
* **Demand Process:** Truncated $\text{Poisson}(6)$ on $\{0, \dots, 12\}$ (retains 99.1% probability mass; mean $= 5.93$)

---

## Methods Compared

### 1. Reinforcement Learning & Heuristic Policies
* **$(s, S)$ Base-Stock Policy (`algorithms/ss_policy.py`, `experiments/tune_ss.py`):** Classical reorder-point / order-up-to heuristic tuned via grid search over $(s, S)$ pairs.
* **Tabular Q-Learning (`algorithms/tabular_q_agent.py`, `experiments/train_q.py`):** Model-free value iteration trained over 300,000 episodes ($\epsilon$-greedy decay to $0.05$, 25,420 state-action entries).
* **Deep Q-Network — DQN (`algorithms/dqn_agent.py`, `experiments/train_dqn.py`):** Neural network Q-function approximator trained over 2,500 episodes ($\gamma = 1.0$) with experience replay, action masking, and target network updates.

### 2. Exact & Rolling Optimization (`algorithms/milp_optimizer.py`, `algorithms/branch_and_bound.py`)
* **Stochastic Dynamic Programming (`DP`):** Exact backward-induction benchmark over the discrete state space.
* **Rolling-Horizon MILP (`MILP (rolling forecast)`):** Solves a 5-period lookahead MILP at each step using expected demand forecasts.
* **Clairvoyant MILP (`MILP (clairvoyant)`):** Solves the full-horizon MILP with perfect foresight of realized demand $D_t$ — serves as an unattainable lower bound on cost:

```text
min  sum_t ( K*y_t + c*q_t + h*I_t + p*S_t )
s.t. I_(t-1) + q_t - D_t + S_t = I_t          (flow balance)
     q_t <= M * y_t                           (big-M setup link, M = min(Qmax, Imax) = 30)
     0 <= q_t <= Qmax,  0 <= I_t <= Imax
     0 <= S_t <= D_t,   y_t in {0, 1}
```

---

## Quickstart — Installation & Running Instructions

### 1. Setup
```bash
git clone [https://github.com/Abhi-shek26/inventory-optimization.git](https://github.com/Abhi-shek26/inventory-optimization.git)
cd inventory-optimization
pip install -r requirements.txt
```
*(Alternatively, open `notebooks/rl_training_and_evaluation.ipynb` in Google Colab to train the RL agents on GPU.)*

### 2. Train RL Agents & Tune $(s, S)$ Policy
Run these scripts first so trained models and tuned parameters are saved to `./results/` before joint evaluation:
```bash
# Tune classical (s, S) heuristic via grid search
python -m experiments.tune_ss

# Train Tabular Q-Learning agent (300,000 episodes, ~2.5 min)
python -m experiments.train_q

# Train Deep Q-Network (DQN) agent (2,500 episodes, ~1 min on GPU)
python -m experiments.train_dqn
```

### 3. Run Optimization Analysis & Joint Evaluation
```bash
# 1. MILP structural & sensitivity analysis (brute force, LP gap, Branch & Bound, big-M, K sensitivity)
python -m experiments.milp_analysis

# 2. Joint benchmark across identical demand paths ((s, S), Q-learning, DQN, DP, Rolling MILP, Clairvoyant MILP)
python -m experiments.evaluate

# 3. Paired statistical significance tests against Stochastic DP
python -m experiments.statistics
```
*Optional:* Run `python -m experiments.evaluate_rl` for RL-specific policy agreement metrics against DP, or `python -m experiments.evaluate --figures-only` to regenerate per-episode trajectory plots.

All outputs (CSVs, JSONs, model weights, and PNG figures) are written automatically to `./results/` (excluded from version control via `.gitignore`).

---

## Key Results & Findings

### Reinforcement Learning & Heuristic Training Summary
| Policy | Training Configuration | Convergence / Tuned Result | Output Artifacts |
| :--- | :--- | :--- | :--- |
| **$(s, S)$ Policy** | Exhaustive grid search | Optimal $(s, S) = (3, 24)$; mean cost $= 515.2$ (200-path cross-check: $514.58$) | `results/ss_params.json`, `results/ss_grid.csv`, `results/ss_heatmap.png` |
| **Tabular Q-Learning** | 300,000 episodes, 25,420 Q-table entries | Converges from $976.2 \rightarrow 517.2$ mean episode cost | `results/q_table.npy`, `results/q_training_curve.csv`, `results/q_training_curve.png` |
| **Deep Q-Network (DQN)** | 2,500 episodes, $\gamma = 1.0$ | Converges from $954.3 \rightarrow 523.8$ mean episode cost | `results/dqn_model.pt`, `results/dqn_training_curve.csv`, `results/dqn_training_curve.png` |

### MILP & Exact Benchmark Findings
| Claim | Evidence |
| :--- | :--- |
| **Formulation correctness** | `results/milp_small_instance.csv`: all $2^4$ setup patterns enumerated by brute force; MILP matches the exact cheapest pattern. |
| **LP relaxation integrality gap** | `results/milp_integrality_gap.png`: average root LP gap $\approx 24\%$ across 20 demand paths. |
| **Custom Branch & Bound verification** | `results/milp_bnb_stats.csv`: matches CBC solver optimum across 127–177 nodes with ~40% pruning. |
| **Tight big-$M$ formulation impact** | `results/milp_big_m.csv`: tight $M = 30 \rightarrow 9.3\%$ root gap vs. loose $M = 9999 \rightarrow 55.5\%$ gap. |
| **Fixed cost $K$ drives order batching** | `results/milp_fixed_cost.png`: order frequency drops as $K$ increases; $K = 0$ reduces to a plain LP. |
| **Value of perfect foresight (Clairvoyant MILP)** | `results/summary.csv` & `results/paired_tests.csv`: 9.3% below DP across all 40 paths (unattainable offline bound). |
| **Implementable policies vs. DP** | Rolling-forecast MILP ($-0.1\%, p = 0.81$) and trained RL/$(s,S)$ policies closely track the Stochastic DP optimum. |

---

## Repository Structure

| File / Directory | Role |
| :--- | :--- |
| `config.py` | Frozen problem parameters, demand distribution, and `setup_big_m()` |
| `environment.py` | Periodic-review inventory simulator (dynamics and fixed-charge cost accounting) |
| `algorithms/milp_optimizer.py` | MILP builder, clairvoyant solver, LP relaxation, and rolling-forecast planner |
| `algorithms/branch_and_bound.py` | Custom Branch & Bound over binary setup variables $y_t$ with node/bound tracking |
| `algorithms/dynamic_programming.py` | Exact Stochastic Dynamic Programming reference policy |
| `algorithms/ss_policy.py` | Classic $(s, S)$ reorder-point / order-up-to heuristic and grid search |
| `algorithms/tabular_q_agent.py` | Model-free Tabular Q-Learning agent |
| `algorithms/dqn_agent.py` | Deep Q-Network (DQN) architecture, replay buffer, and training agent |
| `experiments/tune_ss.py` | Grid-search tuning and cross-check for the $(s, S)$ heuristic |
| `experiments/train_q.py` | Training script for Tabular Q-Learning |
| `experiments/train_dqn.py` | Training script for Deep Q-Network (DQN) |
| `experiments/evaluate_rl.py` | Focused RL & $(s, S)$ evaluation against Stochastic DP |
| `experiments/milp_analysis.py` | MILP verification, LP gap, Branch & Bound, big-$M$, and $K$-sensitivity experiments |
| `experiments/evaluate.py` | Unified evaluation harness comparing all policies on identical demand paths |
| `experiments/statistics.py` | Paired $t$-tests and confidence intervals against Stochastic DP |
| `experiments/demand_scenarios.py` | Stationary, step-change, and high-variance demand scenario generators |
| `utils/simulation.py` | Shared episode rollout engine and metric aggregation |
| `utils/visualization.py` | Figure generation for training curves, heatmaps, and policy comparisons |
| `notebooks/` | Google Colab notebook for GPU training and end-to-end experiment execution |
| `RUNBOOK.md` | Step-by-step execution log mapping commands to printed outputs and generated artifacts |
| `docs/TEAM_SYNC.md` | Frozen shared interface, parameters, and information protocol between team halves |

---

## Assumptions & Limitations

* Single product, zero replenishment lead time, integer order/inventory units, and lost sales (no backlogging).
* Stationary truncated Poisson demand in the baseline setting (non-stationary step-change and high-variance regimes are evaluated separately in `demand_scenarios.py`).
* Clairvoyant MILP assumes offline knowledge of future demand and serves strictly as a theoretical lower bound.
* Rolling-horizon MILP optimizes over a 5-period expected-demand lookahead and does not explicitly model demand variance.
* Custom Branch & Bound is evaluated on 4- and 8-period instances; full 20-period instances rely on CBC presolve and cutting planes.
