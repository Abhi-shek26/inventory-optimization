# Runbook — what runs when you run what

Everything in this project is produced by **three commands**. This page maps each
command to the Python file it starts, the functions that actually execute, the text
you will see in the terminal, and the files that land in `results/`.

Verified on Windows / Python 3.14 / PuLP 3.0.2 with the default configuration
(`T = 20`, `Imax = 40`, `Qmax = 30`, `K = 25`, `c = 2`, `h = 0.5`, `p = 8`,
truncated `Poisson(6)` demand).

---

## 0. One-time setup

```bash
pip install -r requirements.txt
```

No Python file runs; it installs `numpy`, `pandas`, `matplotlib`, `scipy` and
`PuLP < 4` (PuLP 4.x changed its variable API and will break the model).
`torch` is **not** needed — that is only for the RL part.

---

## 1. `python -m experiments.milp_analysis` — the MILP evidence  (~2 min)

**Entry file:** `experiments/milp_analysis.py` → `run()`

### What executes, in order

| Step | Function in `milp_analysis.py` | What it calls underneath |
|---|---|---|
| 1/4 | `small_instance_analysis()` | `enumerate_setup_patterns()` → `milp_optimizer._build_model()` + `_solve()` **16 times** (one per 0/1 pattern), then `solve_milp()` and `branch_and_bound()` for the cross-check |
| 2/4 | `gap_and_node_analysis()` | `solve_milp(..., with_lp_bound=True)` → **2 CBC solves per demand path, 20 paths**; `branch_and_bound()` on 3 smaller 8-period instances; `plot_integrality_gap()`, `plot_bnb_trace()` |
| 3/4 | `big_m_study()` | on 3 demand paths, for tight `M = 12` and loose `M = 9999`: `solve_milp()` (exact), `solve_milp(relax=True, big_m=…)` (LP bound) and `branch_and_bound(big_m=…)`; then `plot_big_m_comparison()` |
| 4/4 | `fixed_cost_sensitivity()` | `DynamicProgrammingSolver.solve()` once per `K`, and `solve_milp()` for 12 demand paths per `K`; `plot_fixed_cost_sensitivity()` |

Then it assembles `results/milp_analysis.json` from everything above.

### What you see printed

```
1/4 worked example: enumerating all 2^4 setup patterns ...
    brute force best pattern 1010 = 96.50, MILP = 96.50, B&B = 96.50 in 19 nodes
2/4 LP relaxation gap and Branch-and-Bound node counts ...
3/4 tight vs loose big-M ...
4/4 fixed ordering cost sensitivity ...

Mean root integrality gap: 24.5%
 instance  horizon  milp_cost  bnb_cost  root_lp_bound  root_gap_pct  nodes_explored  nodes_pruned  finished  seconds
        0        8      221.5     221.5         196.00         11.51             167            64      True    12.53
        1        8      222.0     222.0         200.08          9.87             177            70      True    13.82
        2        8      279.5     279.5         253.17          9.42             127            55      True     9.70
                    label  big_m  milp_cost  lp_root_bound  root_gap_pct  bnb_nodes  bnb_all_finished
tight M = min(Qmax, Imax)     12     220.67         200.08          9.33     123.00              True
           loose M = 9999   9999     220.67          98.12         55.53     202.33              True
Saved MILP analysis in ./results/
```

Read it as: **step 1** = the MILP agrees with exhaustive search; **step 2** = the
LP relaxation is 24.5% below the optimum and Branch & Bound closes it in ~130–180
nodes, matching CBC; **step 3** = a loose big-M keeps the same optimum but wrecks
the LP bound (9.3% → 55.5% gap); **step 4** = how `K` drives batching.

### Files written

| File | Contents | Produced by |
|---|---|---|
| `results/milp_small_instance.csv` | all 2⁴ rows: setup pattern, orders placed, cost, optimal flag | step 1 |
| `results/milp_bnb_stats.csv` | per instance: MILP cost, B&B cost, root LP bound, root gap %, nodes, pruned, finished, seconds | step 2 |
| `results/milp_integrality_gap.png` | histogram of the 20 root LP gaps | step 2 |
| `results/milp_bnb_trace.png` | open-node bound vs incumbent over nodes | step 2 |
| `results/milp_big_m.csv` | tight vs loose `M`: LP bound, root gap, B&B nodes | step 3 |
| `results/milp_big_m.png` | bar chart of that comparison | step 3 |
| `results/milp_fixed_cost_sensitivity.csv` | `K` → DP expected cost, MILP mean cost, orders/episode | step 4 |
| `results/milp_fixed_cost.png` | the two-line sensitivity chart | step 4 |
| `results/milp_analysis.json` | everything above, machine readable | end of `run()` |

**This command alone is a complete MILP submission.** It needs no DP results, no RL
part and no trained model.

---

## 2. `python -m experiments.evaluate` — the comparison  (~4 min)

**Entry file:** `experiments/evaluate.py` → `evaluate()`

### What executes, in order

1. `load_dqn_agent()` → tries `from algorithms.dqn_agent import DQNAgent`. The module
   is absent in this repo, so it prints a skip line **and continues**. (In the joint
   repo, with the RL files and `results/dqn_model.pt` present, DQN joins as a 4th
   method.)
2. `DynamicProgrammingSolver(...).solve()` → Bellman recursion over
   `T × (Imax+1) × (Qmax+1) × 13` states — one pass, then it is a lookup table.
3. For each of the three scenarios: `evaluate_scenario()`, and for each of the 40
   episodes:
   - `demand_scenarios.<scenario>()` builds the demand path from
     `default_rng(seed + 10000 + ep)` — **the same path for every method**;
   - `utils.simulation.run_episode(policy, path, cfg)` rolls each policy and scores it:
     - `DP` → `dp.act(t, inventory)`
     - `MILP (rolling forecast)` → `make_rolling_milp_policy()` → `solve_milp()` on a
       5-period mean forecast **every period = 20 solves per episode** (~1.1 s/episode;
       this is what makes the command slow)
     - `MILP (clairvoyant)` → one `solve_milp()` on the full realised path, then
       `replay_plan()` to execute that plan under the same cost accounting;
   - `summarize()` reduces the per-episode records to the 7 required metrics.
4. Six `plot_*` functions in `utils/visualization.py`, then JSON dump.

### What you see printed

```
  [skip] DQN: algorithms/dqn_agent.py not present in this repo
  evaluated Exp1_stationary (40 episodes)
  evaluated Exp2_step_change (40 episodes)
  evaluated Exp3_high_variance (40 episodes)

DP expected optimal cost (known distribution): 493.22
                                mean_total_cost  mean_holding_cost  mean_shortage_cost  mean_ordering_cost  mean_service_level  mean_avg_inventory  mean_compute_time_ms
Exp1 | DP                               487.825            111.425                23.0             353.400               0.976              11.143                 0.000
Exp1 | MILP (clairvoyant)               442.450             81.225                 0.2             361.025               1.000               8.122               472.763
Exp1 | MILP (rolling forecast)          487.138            122.838                28.2             336.100               0.971              12.284              1102.901
... (Exp2, Exp3 rows)

Saved evaluation outputs in ./results/
```

The `493.22` is the DP's **expected** cost from the model; the table's `487.825` is
the **realised** mean over the 40 sampled paths. They differ by sampling error —
quote the table when comparing methods, and the 493.22 only when discussing the DP
objective itself.

### Files written

| File | Contents | Produced by |
|---|---|---|
| `results/evaluation_episodes.csv` | one row per (scenario, episode, method): total/holding/shortage/ordering cost, service level, avg inventory, orders, compute time | `evaluate()` |
| `results/summary.csv` | the same, aggregated per (scenario, method) — **this is the table you quote** | `evaluate()` |
| `results/evaluation_summary.json` | summaries + the full config + DP expected cost + method list | `evaluate()` |
| `results/cost_comparison.png` | mean cost bars with error bars (Exp 1) | `plot_cost_comparison` |
| `results/cost_distributions.png` | boxplot of the 40×3 episode costs (Exp 1) | `plot_cost_distributions` |
| `results/cost_breakdown.png` | stacked fixed/variable/holding/shortage (Exp 1) | `plot_cost_breakdown` |
| `results/scenario_costs.png` | grouped bars across the three scenarios | `plot_scenario_costs` |
| `results/inventory_trajectories.png` | ending-inventory paths for the first 4 episodes | `plot_inventory_trajectories` |
| `results/order_patterns.png` | `q_t` per period, same episodes | `plot_order_patterns` |

### Variant

```bash
python -m experiments.evaluate --figures-only
```
Runs 6 episodes through `evaluate_scenario()` and rewrites only
`inventory_trajectories.png` and `order_patterns.png` (~20 s). It does **not** touch
the CSVs, JSON or the other figures.

---

## 3. `python -m experiments.statistics` — is the difference real?  (~2 s)

**Entry file:** `experiments/statistics.py` → `run()`

**Requires step 2** — it reads `results/evaluation_episodes.csv` and will raise
`FileNotFoundError` if `evaluate` has not been run.

### What executes

Reads the episode CSV → pivots to one column per method → `_paired()` per method vs
`DP`, which computes the per-episode difference, `scipy.stats.ttest_rel` for the
p-value and `stats.t.interval` for the 95% CI → `summarize`-style component means.

### What you see printed

```
Paired comparison against DP (40 episodes per scenario)
            scenario                   method  mean_diff    pct  ci_low  ci_high  p_value  wins                  verdict
   Exp1_stationary MILP (rolling forecast)     -0.688 -0.141  -6.424    5.049    0.810    25 indistinguishable from DP
   Exp1_stationary      MILP (clairvoyant)    -45.375 -9.301 -50.837  -39.913    0.000    40              better than DP
... (Exp2, Exp3 rows)

Cost components, Exp 1 (mean per episode)
                          total  holding  shortage  ordering  service %  time ms
DP                       487.82   111.42      23.0    353.40      97.60     0.00
MILP (rolling forecast)  487.14   122.84      28.2    336.10      97.10  1102.90
MILP (clairvoyant)       442.45    81.22       0.2    361.02      99.98   472.76
```

Note `p_value` prints as `0.000` when it is below the display precision — the exact
values (e.g. `4.8e-21`) are in `results/paired_tests.csv`.

### Files written

| File | Contents |
|---|---|
| `results/paired_tests.csv` | one row per (scenario, method): mean diff, %, CI bounds, p-value, paths won, verdict |
| `results/paired_tests.json` | the same plus the component-level means and deltas |

---

## 4. Order and dependencies

```
pip install -r requirements.txt
        │
        ├── python -m experiments.milp_analysis      (independent, ~2 min)
        │
        └── python -m experiments.evaluate           (~4 min)
                     │ writes evaluation_episodes.csv
                     ▼
            python -m experiments.statistics         (must run after evaluate)
```

- Only one hard dependency exists: **statistics needs evaluate's CSV**.
- `milp_analysis` is independent of both — you can run it first, last, or alone.
- Every command **overwrites** its own outputs; nothing needs cleaning between runs.
- Runtime scales with the number of CBC solves. `milp_analysis` is dominated by the
  Branch-and-Bound runs (~1,450 node LPs), plus 16 pattern LPs, 40 gap solves and 60
  `K`-sweep solves. `evaluate` is ≈ 3 scenarios × 40 episodes × (20 + 1) MILP solves,
  so raising `horizon` or `eval_episodes` in `config.py` raises the wall-clock time
  proportionally.

---

## 5. Which file answers which question

| Professor asks | Run | Open |
|---|---|---|
| Show me the formulation and the big-M constraint | — | `algorithms/milp_optimizer.py`, lines 62–101 (`_build_model`) |
| How does Branch and Bound actually work? | `milp_analysis` | `algorithms/branch_and_bound.py` + `results/milp_bnb_trace.png` |
| Why not just round the LP solution? | `milp_analysis` | `results/milp_integrality_gap.png` (24.5% average gap) |
| Why is your `M` that value, and does the choice matter? | `milp_analysis` | `results/milp_big_m.csv` (9.3% vs 55.5% root gap) |
| Is the model correct, or just "the solver said optimal"? | `milp_analysis` | `results/milp_small_instance.csv` (all 16 patterns enumerated) |
| What does the fixed cost actually change? | `milp_analysis` | `results/milp_fixed_cost.png` |
| How does MILP compare with the exact policy? | `evaluate` + `statistics` | `results/summary.csv`, `results/paired_tests.csv` |
| Is the difference statistically real? | `statistics` | `results/paired_tests.csv` (rolling MILP −0.1%, p = 0.81) |
| How much is perfect foresight worth? | `evaluate` | `results/summary.csv`: clairvoyant −9.3% vs DP, 40/40 paths |
